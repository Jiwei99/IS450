import pandas as pd
import spacy
from collections import Counter

from utils import data_cleaning

labels = ["PERSON", "ORG"]
mapping = {"PERSON": "Player", "ORG": "Team"}

def spacy_info_ext(filepath):
    # Read csv file
    data_text = pd.read_csv(filepath)
    return run_ie(data_text)

def run_ie(data_text, eval=False):
    # Data Cleaning
    data_text = data_cleaning(data_text)

    ner = spacy.load("en_core_web_sm")

    docs = data_text["text"].apply(ner)

    processed_entities = []
    eval_result = []
    for doc in docs:
        new_ents = []
        for ent in doc.ents:
            if ent.label_ in labels:
                new_ent = spacy.tokens.Span(doc, ent.start, ent.end, label=mapping[ent.label_])
                new_ents.append(new_ent)
  
        doc.ents = new_ents

        filtered_entities = [(e.text, e.label_) for e in doc.ents]
        processed_entities.extend(filtered_entities)
        eval_result.append(filtered_entities)

    if eval:
        return eval_result
    
    options = {"ents": ["Player", "Team"], "colors": {"Player": "lightgreen", "Team": "lightblue"}}
    display = spacy.displacy.render(docs, style="ent", minify=True, options=options) 

    entity_counts = Counter(processed_entities)
    ranked_entities = entity_counts.most_common()
    
    final_entities = {}
    for entity, count in ranked_entities:
        final_entities[str(entity)] = count


    return (display, final_entities)