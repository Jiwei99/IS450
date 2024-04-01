import pandas as pd
import spacy
from collections import Counter

from utils import data_cleaning

labels = ["PERSON", "ORG"]

def spacy_info_ext(filepath):
    # Read csv file
    data_text = pd.read_csv(filepath)

    # Data Cleaning
    data_text = data_cleaning(data_text)

    ner = spacy.load("en_core_web_sm")

    ents = data_text["text"].apply(ner)

    options = {"ents": labels}
    display = spacy.displacy.render(ents, style="ent", minify=True, options=options) 

    processed_entities = []
    for ent in ents:
        filtered_entities = [(e.text, e.label_) for e in ent.ents if e.label_ in labels]
        processed_entities.extend(filtered_entities)

    entity_counts = Counter(processed_entities)
    ranked_entities = entity_counts.most_common()
    
    final_entities = {}
    for entity, count in ranked_entities:
        final_entities[str(entity)] = count

    return (display, final_entities)