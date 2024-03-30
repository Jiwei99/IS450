import pandas as pd
from nltk.tag import StanfordNERTagger
from collections import defaultdict
import gradio as gr
from spacy import displacy

from utils import data_cleaning

stanford_ner_path = './stanford-ner-2020-11-17'
jar_path = stanford_ner_path + '/stanford-ner.jar'
model_path = stanford_ner_path + '/classifiers/english.all.3class.distsim.crf.ser.gz'


ner_tagger = StanfordNERTagger(model_path, jar_path, encoding='utf-8')

def stanford_info_ext(filepath):
    # Read csv file
    data_text = pd.read_csv(filepath)

    # Data Cleaning
    data_text = data_cleaning(data_text)

    total_player_counts = defaultdict(int)
    total_team_counts = defaultdict(int)

    display_data = []
    for tweet in gr.Progress().tqdm(data_text["text"]):
        player_counts, team_counts, ents = extract_entities(tweet)
        for player, count in player_counts.items():
            total_player_counts[player] += count
        for team, count in team_counts.items():
            total_team_counts[team] += count
        
        display_data.append({
            "text": tweet,
            "ents": ents,
            "title": None
        })

    options = {"ents": ["ORGANISATION", "PERSON"], "colors": {"PERSON": "lightgreen", "ORGANISATION": "lightblue"}}
    display = displacy.render(display_data, style="ent", manual=True, options=options)

    # Sort and print the aggregated counts
    sorted_player_counts = sorted(total_player_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_team_counts = sorted(total_team_counts.items(), key=lambda x: x[1], reverse=True)

    sorted_player_dict = {}
    for player, count in sorted_player_counts:
        sorted_player_dict[player] = count
    sorted_team_dict = {}
    for team, count in sorted_team_counts:
        sorted_team_dict[team] = count

    result = {"ORGANISATION": sorted_team_dict, "PERSON": sorted_player_dict}

    return (display, result)

def find_entity_positions(text, entities):
    displacy_entities = []
    for entity_text, entity_type in entities:
        start = text.find(entity_text)
        if start != -1:  # Entity text found in the sentence
            end = start + len(entity_text)
            displacy_entities.append({"start": start, "end": end, "label": entity_type})
    return displacy_entities

def extract_entities(tweet):
    tagged_entities = ner_tagger.tag(tweet.split())

    player_counts = defaultdict(int)
    team_counts = defaultdict(int)
    current_entity = {"entity": "", "tag": ""}
    for entity, tag in tagged_entities:
        if tag == 'PERSON':
            current_entity["entity"] += entity + " "
            current_entity["tag"] = tag
        elif tag == 'ORGANIZATION':
            if current_entity["tag"] == 'ORGANIZATION':
                current_entity["entity"] += entity + " "
            else:
                if current_entity["entity"].strip():
                    if current_entity["tag"] == 'PERSON':
                        player_counts[current_entity["entity"].strip()] += 1
                    elif current_entity["tag"] == 'ORGANIZATION':
                        team_counts[current_entity["entity"].strip()] += 1
                current_entity["entity"] = entity + " "
                current_entity["tag"] = tag
        else:
            if current_entity["entity"].strip():
                if current_entity["tag"] == 'PERSON':
                    player_counts[current_entity["entity"].strip()] += 1
                elif current_entity["tag"] == 'ORGANIZATION':
                    team_counts[current_entity["entity"].strip()] += 1
            current_entity["entity"] = ""
            current_entity["tag"] = ""

    # Add the last recognized entity
    if current_entity["entity"].strip():
        if current_entity["tag"] == 'PERSON':
            player_counts[current_entity["entity"].strip()] += 1
        elif current_entity["tag"] == 'ORGANIZATION':
            team_counts[current_entity["entity"].strip()] += 1

    entities = list(map(lambda x: (x, "PERSON"), player_counts.keys()))
    entities.extend(list(map(lambda x: (x, "ORGANISATION"), team_counts.keys())))
    ents = find_entity_positions(tweet, entities)

    return player_counts, team_counts, ents