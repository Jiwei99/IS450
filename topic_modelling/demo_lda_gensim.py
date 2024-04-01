import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # Import WordCloud class
import numpy as np

from utils import clean_text

def lda_g_topic_model(csv_file_path, num_topics=5):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Clean text data
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Tokenize and preprocess text for Gensim
    documents = df['cleaned_text'].apply(lambda x: x.split())

    # Create dictionary and corpus
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Perform LDA
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    # Prepare the data for PyLDAvis visualization
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)

    # Display the PyLDAvis visualization
    pyLDAvis.display(lda_display)

    images = []

    # Generate and display word clouds for each topic sequentially
    for idx, topic in lda_model.print_topics():
        print(f"Topic #{idx + 1}: {topic}")
        print()

        plt.figure(figsize=(10, 5))
        plt.imshow(WordCloud(width=800, height=400, background_color='white').fit_words(dict(lda_model.show_topic(idx, 10))))
        plt.title(f'Topic #{idx + 1} Word Cloud')
        plt.axis('off')

        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        images.append(data)
    return images