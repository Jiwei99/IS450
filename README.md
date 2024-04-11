# IS450 (Text Mining & Language Processing) Project

## Note
The deploy version has LDA (Gensim) Topic Modelling disabled due to memory limitations on the deployment site.

## Description
This project contains the code for the IS450 Text Mining & Language Processing project.

## Dataset
The full IPL 2022 Twitter tweets dataset can be downloaded from `https://www.kaggle.com/datasets/kaushiksuresh147/ipl2020-tweets`

## Additional Dependencies
- The Stanford NER package is required to run the Stanford NER model for information extraction. The package can be downloaded from `https://nlp.stanford.edu/software/CRF-NER.shtml`. Once downloaded, place the package in the root directory of the project. Alternative, amend the `.env` file to reference the NER package.

## Project Structure

### Frontend Web Portal
- The frontend web portal is encapsulated within the `app.py` file. It consists of the code needed to generate and run the webpage, powered by `Gradio`, a Python library that allows the quick creation of customizable web apps for machine learning models and data processing pipelines.
- In addition, the `sentiment_analysis/demo_sentiment_analysis.py` file is a supplementary file to generate the necessary components to display the sentiment analysis.

### Utilities
- The utilities required for this project are encapsulated within the `utils.py` file. The main utilities included are functions that are required to pre-process the Twitter tweets.
- The preprocessing involves the following steps:
    - #### Topic Modelling (`clean_text`)
        - Converting text to lower case
        - Removing punctuation
        - Removing special characters i.e. \n, \t
        - Removing emoticons
        - Removing numbers
        - Tokenise text
        - Removing stop words
        - Lemmatizing text
    - #### Information Extraction (`data_cleaning`)
        - Removing emoticons
        - Removing links
        - Removing Twitter mentions
        - Removing hashtags
        - Removing special characters

### Sentiment Analysis
- The `sentiment_analysis.ipynb` file contains code used to generate the sentiments.

### Topic Modelling
- The following files contains code for the following models:
    - `demo_lda_gensim.py`: Use of `gensim`'s LDA model
    - `demo_lda_sklearn.py`: Use of `scikit-learn`'s LDA model
    - `demo_lsa_sklearn.py`: Use of `scikit-learn`'s LSA model
    - `demo_nmf_sklearn.py`: Use of `scikit-learn`'s NMF model
- The file `topic_modelling.ipynb` contains code used to run, test and evaluate each model.

### Information Extraction
- The following files contains code for the following models:
    - `ie_rules.py`: Use of custom rules-based NER model (Rules-Based)
    - `ie_spacy.py`: Use of `SpaCy`'s NER model (CNN & RNN)
    - `ie_stanford.py`: Use of `Stanford NER`'s NER model (CRF)
- The file `ie_evaluation.ipynb` contains the code and gold truth that was used to evaluate individual models

## Installation
1. Clone the repository: `git clone https://github.com/Jiwei99/is450.git`
2. Navigate to the project folder: `cd is450`
3. Install the dependencies: `pip install -r requirements.txt`

## Usage
1. Run the project: `python app.py`
2. Open your browser and go to `http://127.0.0.1:7860`
