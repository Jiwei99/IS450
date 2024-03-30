import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud  # Import WordCloud class
import numpy as np

from utils import clean_text

def lda_s_topic_model(csv_file_path, num_topics=5):
    # Read CSV file
    df = pd.read_csv(csv_file_path)

    # Clean text data
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Create CountVectorizer
    count_vectorizer = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    tf = count_vectorizer.fit_transform(df['cleaned_text'])

    # Apply LDA
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_topic_matrix = lda_model.fit_transform(tf)

    images = []

    # Display word clouds for each topic sequentially
    for idx, topic in enumerate(lda_model.components_):
        print(f"Topic #{idx + 1}:")

        # Get top 10 words for the topic with their probabilities
        top_words_indices = topic.argsort()[-10:]
        top_words = [count_vectorizer.get_feature_names_out()[i] for i in top_words_indices]
        top_word_probabilities = [topic[i] for i in top_words_indices]

        # Normalize probabilities
        topic_prob_sum = sum(top_word_probabilities)
        top_word_probabilities = [prob / topic_prob_sum for prob in top_word_probabilities]

        # Arrange words and probabilities in descending order of probabilities
        sorted_indices = sorted(range(len(top_word_probabilities)), key=lambda k: top_word_probabilities[k], reverse=True)
        top_words_sorted = [top_words[i] for i in sorted_indices]
        top_word_probabilities_sorted = [top_word_probabilities[i] for i in sorted_indices]

        # Print words and probabilities
        for word, prob in zip(top_words_sorted, top_word_probabilities_sorted):
            print(f"{word}: {prob:.4f}", end=" | ")
        print()  # Print new line

        # Generate and display word cloud
        fig = plt.figure(figsize=(10, 5))
        plt.imshow(WordCloud(width=800, height=400, background_color='white').fit_words({word: topic[i] for i, word in enumerate(count_vectorizer.get_feature_names_out()) if i in top_words_indices}))
        plt.title(f'Topic #{idx + 1} Word Cloud')
        plt.axis('off')

        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(canvas.get_width_height()[::-1] + (3,))
        images.append(data)
    return images

