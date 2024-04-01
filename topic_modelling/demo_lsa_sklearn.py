import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # Import WordCloud class

from utils import clean_text

def lsa_s_topic_model(csv_file_path, num_topics=5):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Clean text data
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])

    # Apply LSA
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

    # Normalize the topic matrix
    lsa_topic_matrix /= lsa_topic_matrix.sum(axis=1, keepdims=True)

    images = []
    # Display word probabilities for each topic
    for idx, topic in enumerate(lsa_model.components_):
        print(f"Topic #{idx + 1}:")

        # Get top 10 words and their probabilities for the topic
        top_words_indices = topic.argsort()[-10:]
        top_words = [tfidf_vectorizer.get_feature_names_out()[i] for i in top_words_indices]
        top_word_probabilities = [topic[i] for i in top_words_indices]

        # Normalize probabilities
        topic_prob_sum = sum(top_word_probabilities)
        top_word_probabilities = [prob / topic_prob_sum for prob in top_word_probabilities]

        # Sort words and probabilities in descending order of probabilities
        sorted_indices = sorted(range(len(top_word_probabilities)), key=lambda k: top_word_probabilities[k], reverse=True)
        top_words_sorted = [top_words[i] for i in sorted_indices]
        top_word_probabilities_sorted = [top_word_probabilities[i] for i in sorted_indices]

        # Print words and probabilities
        for word, prob in zip(top_words_sorted, top_word_probabilities_sorted):
            print(f"{word}: {prob:.4f}", end=" | ")
        print()  # Print new line

        # Generate and display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(WordCloud(width=800, height=400, background_color='white').fit_words({word: topic[i] for i, word in enumerate(tfidf_vectorizer.get_feature_names_out()) if i in top_words_indices}))
        plt.title(f'Topic #{idx + 1} Word Cloud')
        plt.axis('off')

        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        images.append(data)
    return images