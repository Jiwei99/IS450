import pandas as pd
from spacy import displacy
from collections import Counter
import nltk
from nltk.chunk import ChunkParserI
from nltk.chunk.regexp import RegexpParser
from collections import Counter
from nltk.corpus import words
import pycountry

from utils import data_cleaning

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

# Load English words from NLTK corpus
english_words = set(words.words())

country_names = [country.name for country in pycountry.countries]

class NERChunker(ChunkParserI):
    def __init__(self):
        self.chunk_parser = RegexpParser('''
            Player: {<NNP|NNPS>+}
            Team: {<NNP><NNP|NNPS>*}
        ''')
        self.annotated_data = [
            ("Chennai Super Kings", "Team"),
            ("CSK", "Team"),
            ("Mumbai Indians", "Team"),
            ("MI", "Team"),
            ("Delhi Capitals", "Team"),
            ("DC", "Team"),
            ("Kolkata Knight Riders", "Team"),
            ("KKR", "Team"),
            ("Royal Challengers Bangalore", "Team"),
            ("RCB", "Team"),
            ("Sunrisers Hyderabad", "Team"),
            ("SRH", "Team"),
            ("Punjab Kings", "Team"),
            ("PBKS", "Team"),
            ("Rajasthan Royals", "Team"),
            ("RR", "Team"),
            ("Gujarat Titans", "Team"),
            ("GT", "Team"),
            ("Lucknow Super Giants", "Team"),
            ("LSG", "Team")
        ]

    def parse(self, sentence):
        parsed_sentence = self.chunk_parser.parse(sentence)
        return self.resolve_entities(parsed_sentence)

    def resolve_entities(self, parsed_sentence):
        entities = []
        for subtree in parsed_sentence:
            if isinstance(subtree, nltk.tree.Tree):
                entity = " ".join(word for word, _ in subtree.leaves())
                entity_type = self.get_entity_type(entity)
                if entity_type is not None:
                    entities.append((entity, entity_type))
        return entities

    def get_entity_type(self, entity):
        if entity in country_names:
            return None
        for annotated_entity, label in self.annotated_data:
            if entity == annotated_entity:
                return label
        return 'Player'

def rules_info_ext(filepath):
    # Read csv file
    data_text = pd.read_csv(filepath)
    return run_ie(data_text)

def run_ie(data_text, eval=False):

    # Data Cleaning
    data_text = data_cleaning(data_text)

    ner_chunker = NERChunker()
    entity_counter = Counter()

    displacy_data = []
    eval_result = []
    for tweet in data_text["text"]:
        tagged_sentence = nltk.pos_tag(nltk.word_tokenize(tweet))
        classified_sentence = ner_chunker.parse(tagged_sentence)
        eval_result.append(classified_sentence)
    
        display_sentence = []
        for entity, entity_type in classified_sentence:
            entity_words = entity.lower().split()
            english_word_count = sum(1 for word in entity_words if word in english_words)
            if english_word_count >= 2:
                continue
            entity_counter[(entity, entity_type)] += 1
            display_sentence.append((entity, entity_type))
            
        displacy_data.append({
            "text": tweet,
            "ents": find_entity_positions(tweet, display_sentence),
            "title": None
        })

    if eval:
        return eval_result
        
    options = {"ents": ["PLAYER", "TEAM"], "colors": {"PLAYER": "lightgreen", "TEAM": "lightblue"}}
    display = displacy.render(displacy_data, style="ent", manual=True, options=options)

    sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1], reverse=True)
    final_entities = {}
    for entity, count in sorted_entities:
        final_entities[str(entity)] = count

    return (display, final_entities)

def find_entity_positions(text, entities):
    displacy_entities = []
    for entity_text, entity_type in entities:
        start = text.find(entity_text)
        if start != -1:  # Entity text found in the sentence
            end = start + len(entity_text)
            displacy_entities.append({"start": start, "end": end, "label": entity_type})
    return displacy_entities