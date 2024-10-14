import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import os
import time


# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp = spacy.load('en_core_web_sm')
# nlp = en_core_web_sm.load()
# Stop words
stop_words = stopwords.words('english')
stop_words.extend(STOP_WORDS)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize and clean-up text
    result = []
    for token in gensim.utils.simple_preprocess(text, deacc=True):
        if token not in stop_words and len(token) > 3:
            # Lemmatize the token
            lemma = lemmatizer.lemmatize(token)
            result.append(lemma)
    return result

def create_lda_from_df(temp_cluster_df, num_topics=5):
      
    # Preprocess the abstracts
    processed_docs = temp_cluster_df['abstract'].map(preprocess_text)
    
    # Create a dictionary and corpus needed for LDA
    dictionary = corpora.Dictionary(processed_docs)
    
    # Filter out extremes to remove noise in the data
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Store corpus data for visualization later
    corpus_data = {'dictionary': dictionary, 'corpus': corpus,'processed_docs': processed_docs}
    
    # Build LDA model
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    # Print the topics
    print(f"Top topics in Cluster:")
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)
    print("\n")

    return lda_model, corpus_data


def visualize_lda(lda_model, corpus_data, output_path):

    # Create directories to save models and visualizations
    os.makedirs('lda_models', exist_ok=True)
    os.makedirs('lda_visualizations', exist_ok=True)
    
    # save file as lda_model_{data_time}.model
    model_save_path = os.path.join(output_path, 'lda_model_' + time.strftime("%Y%m%d-%H%M%S") + '.model')
    lda_model.save(model_save_path)
        
    dictionary = corpus_data['dictionary']
    corpus = corpus_data['corpus']
    
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    filename = f"lda_visualization_{time.strftime('%Y%m%d-%H%M%S')}.html"
    pyLDAvis.save_html(vis_data, os.path.join(output_path, filename))