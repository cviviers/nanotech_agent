import os
import getpass
import json
import altair as alt

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import numpy as np
from ast import literal_eval
import gradio as gr

from gradio.components import scatter_plot
import requests
from itertools import islice
import hdbscan
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import ast

from utils.utils import get_embedding_from_api, write_df_to_excel
from utils.lda_utils import create_lda_from_df, visualize_lda

reducer = umap.UMAP(random_state=42)
alt.data_transformers.disable_max_rows()

def append_to_df_list(df_old_list, new_entry):
    df_list = df_old_list + new_entry
    return df_list

def load_temp_data(folder_path):
    if os.path.exists(folder_path):
        # read df with embeddings
            # load the embeddings from the json files
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        # json_files = json_files[:1000]
        data = {}

        for file in json_files:
            with open(os.path.join(folder_path, file), 'r') as f:
                data[file] = json.load(f)




        # data to dataframe
        df = pd.DataFrame(data).T
        df['size'] = 10
        df['color'] = 'red'

        # replace Abstracts with the first Abstract in the list if it is a list
        df['Abstract'] = df['Abstract'].apply(lambda x: x[0] if isinstance(x, list) else x)

        # store the embeddings in a numpy array
        embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
        embeddings = np.stack(embeddings)

        return df, embeddings
    
def load_data(json_path):
    if os.path.exists(json_path):
        # read df with embeddings
        # load json file to pandas df
        with open(json_path, 'r') as f:
            data = json.load(f)

        # data to dataframe
        df = pd.DataFrame(data).T
        df['size'] = 10
        df['color'] = 'red'

        # rename the 'Article Title' column to 'title'
        df.rename(columns={'Article Title': 'title'}, inplace=True)

        # only keep the following columns: 'title', 'Abstract', 'embedding'
        df = df[['title', 'Abstract', 'embedding', 'size', 'color']]


        return df
    

def create_tsne_embeddings(df, embeddings):
    
    # use the embeddings to create a t-SNE plot
    # get the embeddings from the dataframe
    # embeddings = df['embedding'].apply(lambda x: np.array(x)).values

    # create a t-SNE object
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)

    # fit the t-SNE object to the embeddings
    tsne_embeddings = tsne.fit_transform(np.stack(embeddings[:]))

    # create new pandas df with old df added the tsne embeddings
    df_tsne = df.copy()

    df_tsne['low_x'] = tsne_embeddings[:, 0]
    df_tsne['low_y'] = tsne_embeddings[:, 1]

    return df_tsne, tsne_embeddings

def create_umap_embeddings(df):
    reducer = umap.UMAP(random_state=42)
    embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
    embeddings = np.stack(embeddings)
    umpa_embedding = reducer.fit_transform(embeddings)
    df_umpa = df.copy()
    df_umpa['low_x'] = umpa_embedding[:, 0]
    df_umpa['low_y'] = umpa_embedding[:, 1]
    df_umpa['size'] = 10
    df_umpa['color'] = 'red'

    return df_umpa

def create_umap_scatter_plot(df_umpa):

    df_umpa_temp = df_umpa.copy()
    plot = scatter_plot.ScatterPlot(
            value=df_umpa_temp,
            x="low_x",
            y="low_y",
            title="UMAP",
            color='color',
            size= 'size',
            tooltip=['title', 'Abstract'],
            width=1200,
            height=1200
        )
    return plot


def get_semantic_similar_embeddings(query, df,  num_cases=10):
    # get the embedding from the api
    # try converting the number of cases to an integer
    try:
        num_cases = int(num_cases)
    except:
        raise gr.Error("Please enter a valid number of cases to return", duration=30)
    # get the embedding from the api
    try:
        embedding, num_tokens = get_embedding_from_api(query, query_type="s2s")	
        embedding_array = np.array(embedding)
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)
  
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)
    
    # calculate the dot product of the embeddings with the query embedding


    similarities = np.dot(embeddings, embedding_array)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # create a dataframe with the most similar embeddings
    similar_df = df.copy()
    similar_df['similarity'] = similarities
    
    # sort df by similarity
    # Sort DataFrame by similarity scores
    similar_df = similar_df.iloc[sorted_indices].reset_index(drop=True)
    similar_df['current_index'] = similar_df.index
    similar_df['size'] = 10
    similar_df['color'] = 'red'

    # create scatter plot with the similarity 

    plot = scatter_plot.ScatterPlot(
            value=similar_df,
            x="current_index",
            y="similarity",
            title="Semantic similarity",
            color='color',
            size= 'size',
            # tooltip displays the title of the article
            tooltip=['title', 'Abstract'],
            width=1200,
            height=400
        )
    top_similar_df = similar_df.iloc[:num_cases]
    return top_similar_df, plot

def get_retrieval_embeddings(query, df,  num_cases=10):
    # get the embedding from the api
    # try converting the number of cases to an integer
    try:
        num_cases = int(num_cases)
    except:
        raise gr.Error("Please enter a valid number of cases to return", duration=30)
    # get the embedding from the api
    try:
        embedding, num_tokens = get_embedding_from_api(query, query_type="s2p")	
        embedding_array = np.array(embedding)
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)
  
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)

    # calculate the dot product of the embeddings with the query embedding
    similarities = np.dot(embeddings, embedding_array)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # create a dataframe with the most similar embeddings
    similar_df = df.copy()
    similar_df['similarity'] = similarities
    
    # sort df by similarity
    # Sort DataFrame by similarity scores
    similar_df = similar_df.iloc[sorted_indices].reset_index(drop=True)
    similar_df['current_index'] = similar_df.index
    similar_df['size'] = 10
    similar_df['color'] = 'red'

    # create scatter plot with the similarity 

    plot = scatter_plot.ScatterPlot(
            value=similar_df,
            x="current_index",
            y="similarity",
            title="Semantic similarity",
            color='color',
            size= 'size',
            # tooltip displays the title of the article
            tooltip=['title', 'Abstract'],
            width=1200,
            height=400
        )
    top_similar_df = similar_df.iloc[:num_cases]
    return top_similar_df, plot


def get_query_embedding_and_similarity(query, df, df_history_list, dim_reduction='UMAP'):
    # get the embedding from the api
    try:
        embedding, num_tokens = get_embedding_from_api(query, query_type="s2p")
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)

    # add the embedding to the dataframe
    df_query = df
    df_query['size'] = 10
    # df_query['color'] = 'red'

    # compute the similarity of the query with the embeddings
    query_embedding = np.array(embedding)
    embeddings = df_query['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)

    similarities = np.dot(embeddings, query_embedding)
    # sorted_indices = np.argsort(similarities)[::-1]

    df_query['similarity'] = similarities

    # add the query embedding to the dataframe with the title "Query" and Abstract the value of the query, all other fields 'Not Available'
    new_entry = {'title': 'Query', 'Abstract': query, 'embedding': embedding,  'color': 'black', 'size': 200, 'similarity': 1}
    for column in df.columns:
        if column not in new_entry:
            new_entry[column] = 'Not Available'

    df_query.loc[len(df_query)] = new_entry
    # get the tsne embeddings of the query
    query_embedding = np.array(df_query['embedding'].map(lambda x: np.array(x)))
    query_embedding = np.stack(query_embedding)

    if dim_reduction == 'UMAP':
        reducer = umap.UMAP(random_state=42) 
        query_umpa = reducer.fit_transform(query_embedding)
        
        df_query['low_x'] = query_umpa[:, 0]
        df_query['low_y'] = query_umpa[:, 1]

    elif dim_reduction == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)
        query_tsne = tsne.fit_transform(query_embedding)
        
        df_query['low_x'] = query_tsne[:, 0]
        df_query['low_y'] = query_tsne[:, 1]

    plot = scatter_plot.ScatterPlot(
            value=df_query,
            x="low_x",
            y="low_y",
            title="Embedding Query",
            color='color',
            size= 'size',
            # tooltip displays the title of the article
            tooltip=['title', 'Abstract', 'similarity'],
            width=1200,
            height=1200
        )
    df_history_list.append((plot, df_query.copy()))  # Append the new dataframe
    return plot, df_query, df_history_list

def color_embeddings(property, color, dataframe, df_history_list):
    # get the embeddings from the dataframe

    properties = property.split(',')
    colors = color.split(',')

    print(properties, colors)

    df_color = dataframe
    df_color['size'] = 10
    df_color['color'] = 'blue' 

    if len(properties) > 1 and len(colors) == 1:
        # apply the same color to all the properties
        colors = colors * len(properties)

    else:
        if len(properties) != len(colors):
            raise gr.Error("The number of properties and colors should be the same", duration=30)
     


    # color the embeddings according to the property, replace the color of the embeddings that contain the property with the color
    for prop, col in zip(properties, colors):
        mask = df_color['Abstract'].fillna('Not Available').str.contains(prop, case=False, na=False)
        # replace the color of the embeddings that contain the property with the color
        # print number of embeddings that contain the property
        print(f"Number of embeddings that contain the property '{prop}': {mask.sum()}")
        df_color.loc[mask, 'color'] = col



    
    # df_color['color'] = 'red'
    

    plot = scatter_plot.ScatterPlot(
            value=df_color,
            x="low_x",
            y="low_y",
            title="Color embeddings",
            color='color',
            size= 'size',
            tooltip=['title', 'Abstract'],
            width=1200,
            height=1200
            )
    
    df_history_list.append((plot, df_color.copy()))  # Append the new dataframe
    
    return plot, df_color, df_history_list

def umap_embedding(df, embeddings):  
    umpa_embedding = reducer.fit_transform(embeddings)
    df_umpa = df.copy()
    df_umpa['x'] = umpa_embedding[:, 0]
    df_umpa['y'] = umpa_embedding[:, 1]
    df_umpa['size'] = 10
    df_umpa['color'] = 'red'

    plot = scatter_plot.ScatterPlot(
            value=df_umpa,
            x="x",
            y="y",
            title="UMAP",
            color='color',
            size= 'size',
            tooltip=['title', 'Abstract'],
            width=1200,
            height=1200
            )
    return plot


# convert RGB to HEX
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def cluster_embeddings(cluster_property, dataframe, df_history_list, dim_reduction='UMAP', cluster_method='k-Means'):

    cluster_property = int(cluster_property)

    local_embeddings = dataframe['embedding'].apply(lambda x: np.array(x))
    local_embeddings = np.stack(local_embeddings)

    df_cluster = dataframe
    df_cluster['size'] = 10
    df_cluster['color'] = 'red'

    if cluster_method == 'HDBSCAN':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_property,  prediction_data=True, branch_detection_data=True, alpha=0.5, cluster_selection_method='leaf').fit(local_embeddings)
        # number of clusters
        n_clusters_ = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)


        
        print('Estimated number of clusters: %d' % n_clusters_)

        colors_values = list(mcolors.XKCD_COLORS.values())
        if n_clusters_ > len(colors_values):
            raise gr.Error("The number of clusters is greater than the number of colors available", duration=30) 
        
        # df_colors = [str(colors_values[x]) for x in clusterer.labels_]

        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        color_palette = sns.color_palette('Paired', n_clusters_)
        cluster_colors = [color_palette[np.argmax(x)]
                        for x in soft_clusters]

        hex_colors = [rgb_to_hex((int(r * 255), int(g * 255), int(b * 255))) for r, g, b in cluster_colors]
        cluster_labels_as_str = [str(x) for x in clusterer.labels_]
        df_cluster['cluster_label'] = cluster_labels_as_str  # Use labels instead of the entire clusterer object
        df_cluster['color'] = hex_colors
        df_cluster['probabilities'] = soft_clusters.max(axis=1)

        # set size of the elements to the probability of the lement belonging to the cluster
        df_cluster['size'] = soft_clusters.max(axis=1) * 10



    elif cluster_method == 'k-Means':
        kmeans = KMeans(n_clusters=int(cluster_property), random_state=42).fit(local_embeddings)
        kmeans_labels = kmeans.labels_
        kmeans_labels_as_str = [str(x) for x in kmeans_labels]


        colors_keys = list(mcolors.XKCD_COLORS.keys())
        colors_values = list(mcolors.XKCD_COLORS.values())

        # predict the cluster of the embeddings
        # add the cluster labels to the dataframe
        df_cluster['cluster_label'] = kmeans_labels_as_str
        df_colors = [str(colors_values[x]) for x in kmeans_labels]

        # color each embedding according to the cluster label
        # get index of label in kmeans.cluster_centers_
        df_cluster['color'] = df_colors

    else:
        raise gr.Error("Please select a valid clustering method", duration=30)

    # df_cluster_kmeans['cluster_labels'] = kmeans.labels_
    
    

    
    # make a copy of the df and add the kmeans.labels_ to the df as a new entry with the title the cluster center and the Abstract the cluster center and the color a unique color from the color palette

    

    if dim_reduction == 'UMAP':
        reducer = umap.UMAP(random_state=42)
        umpa_embedding = reducer.fit_transform(local_embeddings)
        df_cluster['low_x'] = umpa_embedding[:, 0]
        df_cluster['low_y'] = umpa_embedding[:, 1]

    elif dim_reduction == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)
        tsne_embedding = tsne.fit_transform(local_embeddings)
        df_cluster['low_x'] = tsne_embedding[:, 0]
        df_cluster['low_y'] = tsne_embedding[:, 1]

    if cluster_method == 'k-Means':
       custom_tooltip = ['title', 'Abstract', 'cluster_label']
    else:
        custom_tooltip = ['title', 'Abstract', 'cluster_label', 'probabilities', 'color']

    # save matplotlib plot
    fig = plt.figure(figsize=(30, 30))
    plt.scatter(umpa_embedding[:, 0], umpa_embedding[:, 1], c=kmeans_labels, s=50, cmap='viridis')
    # set high dpi
    plt.savefig('output/cluster_plot.png', dpi=300)

    plot = scatter_plot.ScatterPlot(
            value=df_cluster,
            x="low_x",
            y="low_y",
            title="Cluster {}".format(cluster_method),
            color='color',
            size= 'size',
            tooltip=custom_tooltip,
            width=1200,
            height=1200,
            )  


    df_history_list.append((plot, df_cluster.copy()))  # Append the new dataframe
    # df_history_list = append_to_df_list(df_history_list, (gr.State(df_cluster.copy()), plot))  # Append the new dataframe
    return plot, df_cluster, df_history_list  # Return the updated list



def select_cluster(df, selected_value, df_history_list):
    print("Selected value: ", selected_value)
    df_filtered = df[df["cluster_label"] == selected_value]    
    df_filtered.loc[:, 'size'] = 10

    tool_tip_list = ['title', 'Abstract', 'color', 'cluster_label']
    # print number of elements in the filtered dataframe
    print(f"Number of elements in the filtered dataframe: {len(df_filtered)}")
    
    # Ensure the DataFrame is not empty
    if df_filtered.empty:
        raise gr.Error("No data matches the selected criteria. 💥!", duration=30)
    
    if "low_x" not in df_filtered.columns:
        raise gr.Error("The DataFrame has not been clustered. 💥!", duration=30)
    
    df_filtered.loc[:, 'color'] = 'red'
    plot = scatter_plot.ScatterPlot(
        value=df_filtered,
        x="low_x",
        y="low_y",
        title="Filtered Plot",
        color='color',
        size='size',
        tooltip=tool_tip_list,
        width=1200,
        height=1200
    )
    df_history_list.append((plot, df_filtered.copy()))  # Append the new dataframe
    return plot, df_filtered, df_history_list

def select_color(df, selected_value, df_history_list):
    print("Selected value: ", selected_value)
    df_filtered = df[df["color"] == selected_value]    
    df_filtered.loc[:, 'size'] = 10

    
    tool_tip_list = ['title', 'Abstract', 'color']

    # print number of elements in the filtered dataframe
    print(f"Number of elements in the filtered dataframe: {len(df_filtered)}")
    
    # Ensure the DataFrame is not empty
    if df_filtered.empty:
        raise gr.Error("No data matches the selected criteria. 💥!", duration=30)

    
    df_filtered.loc[:, 'color'] = 'red'
    plot = scatter_plot.ScatterPlot(
        value=df_filtered,
        x="low_x",
        y="low_y",
        title="Filtered Plot",
        color='color',
        size='size',
        tooltip=tool_tip_list,
        width=1200,
        height=1200
    )

    df_history_list.append((plot, df_filtered.copy()))  # Append the new dataframe
    return plot, df_filtered, df_history_list

def apply_query_threshold(query_threshold, df, df_history_list):
    try:
        query_threshold = float(query_threshold)
    except:
        raise gr.Error("Please enter a valid threshold", duration=30)

    df_filtered = df[df["similarity"] > query_threshold]    
    df_filtered.loc[:, 'size'] = 10


    tool_tip_list = ['title', 'Abstract', 'color', 'cluster_label', 'similarity']
    # check if the dataframe contains all the tooltip columns, use the subset that is available
    for column in tool_tip_list:
        if column not in df_filtered.columns:
            tool_tip_list.remove(column)


    # print number of elements in the filtered dataframe
    print(f"Number of elements in the filtered dataframe: {len(df_filtered)}")
    
    # Ensure the DataFrame is not empty
    if df_filtered.empty:
        raise gr.Error("No data matches the selected criteria. 💥!", duration=30)
    
    if "low_x" not in df_filtered.columns:
        raise gr.Error("The DataFrame has not been clustered. 💥!", duration=30)
    
    df_filtered.loc[:, 'color'] = 'red'
    plot = scatter_plot.ScatterPlot(
        value=df_filtered,
        x="low_x",
        y="low_y",
        title="Filtered Plot",
        color='color',
        size='size',
        tooltip=tool_tip_list,
        width=1200,
        height=1200
    )

    df_history_list.append((plot, df_filtered.copy()))  # Append the new dataframe
    return plot, df_filtered, df_history_list

def generate_and_visualize_lda(df, num_topics, time=None, cluster_name=None):

    try:
        num_topics = int(num_topics)
    except:
        raise gr.Error("Please enter a valid number of topics", duration=30)
    try:
        lda_model, corpus_data = create_lda_from_df(df, num_topics)
    except:
        raise gr.Error("An error occurred while generating the LDA model", duration=30)
    
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
        raise gr.Error("Please enter a valid number of topics", duration=30)

    # get number of clusters, set == 1 if not defined
    if 'cluster_label' in df.columns:
        clusters = df['cluster_label'].unique()
        print(f"Number of clusters: {len(clusters)}")
    else:
        clusters = [1]

    # get time
    time = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M-_%S")

    for cluster in clusters:
        df_cluster = df[df['cluster_label'] == cluster]
        output_path = generate_and_visualize_lda(df_cluster, num_topics, time, str(cluster))
        try:
            lda_model, corpus_data = create_lda_from_df(df, num_topics)
        except:
            gr.Error("An error occurred while generating the LDA model", duration=30)
        
        visualize_lda(lda_model, corpus_data, output_path)
        message = f"LDA saved for cluster {cluster}! 🎉"
        gr.Info(message, duration=30)

    gr.Info("ALL LDA saved! 🎉", duration=30)

# Compute the first principle component of the embeddings, plot it on an axis with the component on the y-axis and the simlarities to the query on the x-axis 
def create_principle_component_plot(df, query, df_history_list):
    # get the embeddings from the dataframe
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)

    # get the embedding from the api
    try:
        embedding, num_tokens = get_embedding_from_api(query, query_type="s2s")
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)
    
    # calculate the dot product of the embeddings with the query embedding
    similarities = np.dot(embeddings, embedding)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # create a dataframe with the most similar embeddings
    similar_df = df.copy()
    similar_df['similarity'] = similarities
    
    # sort df by similarity
    # Sort DataFrame by similarity scores
    similar_df = similar_df.iloc[sorted_indices].reset_index(drop=True)
    similar_df['current_index'] = similar_df.index
    similar_df['size'] = 10
    similar_df['color'] = 'red'

    # compute the first principle component of the embeddings
    pca = PCA(n_components=1)
    pca_embeddings = pca.fit_transform(embeddings)
    similar_df['pca'] = pca_embeddings


    # create scatter plot with the similarity 

    plot = scatter_plot.ScatterPlot(
            value=similar_df,
            x="similarity",
            y="pca",
            title="Semantic similarity",
            color='color',
            size= 'size',
            # tooltip displays the title of the article
            tooltip=['title', 'Abstract', 'similarity'],
            width=1200,
            height=1200
        )
    df_history_list.append((plot, similar_df.copy()))  # Append the new dataframe
    return plot, similar_df, df_history_list


def update_textbox(selected_option):
    if selected_option == "k-Means":
        # If option 1 is selected, show textbox 1
        return gr.Textbox("10", label="Number of clusters to use in k-Means")
    elif selected_option == "HDBSCAN":
        # If option 2 is selected, show textbox 2
        return gr.Textbox(value="50", label="Minumum number or elements in HDBSCAN cluster", visible=True)
    else:
        return gr.Textbox(visible=False)  # Default behavior, hide textbox

def remove_from_state_list(state_list):
    if len(state_list) > 0:
        state_list.pop()
    else:
        gr.Error("No more history to undo")
    return state_list


# Define functionality for undo button
def undo(df_history_list):
    if len(df_history_list) > 1:
        df_history_list.pop()  # Remove the last dataframe

        return df_history_list[-1][0], df_history_list[-1][1], df_history_list
    else:
        raise gr.Error("No more actions to undo!")
    

def crop_plot(start_x, end_x, start_y, end_y, dataframe, df_history_list):

    # use the "low_x" and "low_y" columns to filter the dataframe

    df_filtered = dataframe[(dataframe['low_x'] >= start_x) & (dataframe['low_x'] <= end_x) & (dataframe['low_y'] >= start_y) & (dataframe['low_y'] <= end_y)]
    df_filtered.loc[:, 'size'] = 10

    # display all the elements in the filtered dataframe
    tool_tip_list = ['title', 'Abstract', 'color', 'cluster_label', 'similarity']
    # check if the dataframe contains all the tooltip columns, use the subset that is available
    for column in tool_tip_list:
        if column not in df_filtered.columns:
            tool_tip_list.remove(column)

    # print number of elements in the filtered dataframe
    print(f"Number of elements in the filtered dataframe: {len(df_filtered)}")

    # Ensure the DataFrame is not empty
    if df_filtered.empty:
        raise gr.Error("No data matches the selected criteria. 💥!", duration=30 )
    
    plot = scatter_plot.ScatterPlot(
        value=df_filtered,
        x="low_x",
        y="low_y",
        title="Filtered Plot",
        color='color',
        size='size',
        tooltip=tool_tip_list,
        width=1200,
        height=1200
    )
    df_history_list.append((plot, df_filtered.copy()))  # Append the new dataframe
    return plot, df_filtered, df_history_list
    
    # inputs=[start_x, end_x, start_y, end_y, dataframe, df_history_list],
    # outputs=[plot_output, dataframe, df_history_list]  