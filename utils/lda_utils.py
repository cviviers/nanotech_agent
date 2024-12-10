import pandas as pd
import numpy as np
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt
import os
import time
import streamlit as st


def create_lda_from_df(temp_cluster_df, num_topics=5):
      
    # Preprocess the abstracts
    processed_docs = temp_cluster_df['cleaned_text']
    
    # Create a dictionary and corpus needed for LDA
    dictionary = corpora.Dictionary(processed_docs)
    
    # Filter out extremes to remove noise in the data
    # dictionary.filter_extremes(no_below=5, no_above=0.5)
    
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
    lda_models_path = os.path.join(output_path, 'lda_models')
    lda_visualizations_path = os.path.join(output_path, 'lda_visualizations')
    os.makedirs(lda_models_path, exist_ok=True)
    os.makedirs(lda_visualizations_path, exist_ok=True)
    
    # save file as lda_model_{data_time}.model
    model_save_path = os.path.join(lda_models_path, 'lda_model_' + time.strftime("%Y%m%d-%H%M%S") + '.model')
    lda_model.save(model_save_path)
        
    dictionary = corpus_data['dictionary']
    corpus = corpus_data['corpus']
    
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    filename = f"lda_visualization_{time.strftime('%Y%m%d-%H%M%S')}.html"
    pyLDAvis.save_html(vis_data, os.path.join(lda_visualizations_path, filename))



def generate_and_visualize_lda(df, num_topics, time=None, cluster_name=None):

    try:
        num_topics = int(num_topics)
    except:
        raise st.error("Please enter a valid number of topics")
    try:
        lda_model, corpus_data = create_lda_from_df(df, num_topics)
    except:
        raise st.error("An error occurred while generating the LDA model")
    
    if cluster_name:
        cluster_path = f'output_{cluster_name}'
        if time:
            output_path = os.path.join('output',"lda_"+time , cluster_path)
        else:
            output_path = os.path.join('output', cluster_path, cluster_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = 'output'
    visualize_lda(lda_model, corpus_data, output_path)

    return output_path

def generate_and_visualize_lda_all_clusters(df, num_topics):

    try:
        num_topics = int(num_topics)
    except:
        raise st.error("Please enter a valid number of topics")

    # get number of clusters, set == 1 if not defined
    if 'cluster_label' in df.columns:
        clusters = df['cluster_label'].unique()
        print(f"Number of clusters: {len(clusters)}")
    else:
        clusters = [0]

    # get time
    time = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M-_%S")

    for cluster in clusters:
        df_cluster = df[df['cluster_label'] == cluster]
        output_path = generate_and_visualize_lda(df_cluster, num_topics, time, str(cluster))
        try:
            lda_model, corpus_data = create_lda_from_df(df, num_topics)
        except:
            st.error("An error occurred while generating the LDA model")
        
        visualize_lda(lda_model, corpus_data, output_path)
        message = f"LDA saved for cluster {cluster}! 🎉"
        st.info(message, duration=30)

    st.info("ALL LDA saved! 🎉")