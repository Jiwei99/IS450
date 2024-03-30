import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_urls(text):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

def remove_char(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\t', '', text)
    return re.sub(r'&amp;', '', text)

# Data cleaning for Information Extraction
def data_cleaning(data_text):
    ## Remove Emojis
    data_text['text'] = data_text['text'].apply(remove_emojis)

    ## Remove Links
    data_text['text'] = data_text['text'].apply(remove_urls)

    ## Remove Mentions
    data_text['text'] = data_text['text'].apply(remove_mentions)

    ## Remove Hashtags
    data_text['text'] = data_text['text'].apply(remove_hashtags)

    ## Remove special characters
    data_text['text'] = data_text['text'].apply(remove_char)

    return data_text

# Text cleaning for Topic Modelling
def clean_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove characters
        text = remove_char(text)

        # Remove emoticons
        text = remove_emojis(text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a string
        cleaned_text = ' '.join(tokens)

        return cleaned_text
    else:
        return ''  # Return empty string if NaN