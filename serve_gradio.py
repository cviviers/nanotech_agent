
import os
import getpass
import json
import altair as alt

import pandas as pd
from sklearn.manifold import TSNE
import umap
import numpy as np
from ast import literal_eval
# create interactive plot with gradio
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
from utils.utils import get_embedding_from_api

reducer = umap.UMAP(random_state=42)
alt.data_transformers.disable_max_rows()


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

        # replace abstracts with the first abstract in the list if it is a list
        df['abstract'] = df['abstract'].apply(lambda x: x[0] if isinstance(x, list) else x)

        # store the embeddings in a numpy array
        embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
        embeddings = np.stack(embeddings)

        return df, embeddings
    

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

def create_umap_embeddings(df, embeddings):
    reducer = umap.UMAP(random_state=42)
    umpa_embedding = reducer.fit_transform(embeddings)
    df_umpa = df.copy()
    df_umpa['low_x'] = umpa_embedding[:, 0]
    df_umpa['low_y'] = umpa_embedding[:, 1]
    df_umpa['size'] = 10
    df_umpa['color'] = 'red'

    return df_umpa, umpa_embedding


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
            tooltip=['title', 'abstract'],
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
            tooltip=['title', 'abstract'],
            width=1200,
            height=400
        )
    top_similar_df = similar_df.iloc[:num_cases]
    return top_similar_df, plot


def get_query_embedding(query, df, dim_reduction='UMAP'):
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

    # add the query embedding to the dataframe with the title "Query" and abstract the value of the query, all other fields 'Not Available'
    new_entry = {'title': 'Query', 'abstract': query, 'embedding': embedding,  'color': 'black', 'size': 200}
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
            tooltip=['title', 'abstract'],
            width=1200,
            height=1200
        )

    return plot, df_query

def color_embeddings(property, color, dataframe):
    # get the embeddings from the dataframe

    property = property.split(',')
    color = color.split(',')

    print(property, color)

    df_color = dataframe
    df_color['size'] = 10
    df_color['color'] = 'red'
    # color the embeddings according to the property, replace the color of the embeddings that contain the property with the color
    for prop, col in zip(property, color):
        mask = df_color['abstract'].fillna('Not Available').str.contains(prop, case=False, na=False)
        # replace the color of the embeddings that contain the property with the color
        # print number of embeddings that contain the property
        print(f"Number of embeddings that contain the property '{prop}': {mask.sum()}")
        df_color.loc[mask, 'color'] = col

    plot = scatter_plot.ScatterPlot(
            value=df_color,
            x="low_x",
            y="low_y",
            title="Color embeddings",
            color='color',
            size= 'size',
            tooltip=['title', 'abstract'],
            width=1200,
            height=1200
            )
    
    return plot, df_color

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
            tooltip=['title', 'abstract'],
            width=1200,
            height=1200
            )
    return plot


# convert RGB to HEX
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb



def cluster_embeddings(cluster_property, dataframe, dim_reduction='UMAP', cluster_method='k-Means'):

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
    
    

    
    # make a copy of the df and add the kmeans.labels_ to the df as a new entry with the title the cluster center and the abstract the cluster center and the color a unique color from the color palette

    

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
       custom_tooltip = ['title', 'abstract', 'cluster_label']
    else:
        custom_tooltip = ['title', 'abstract', 'cluster_label', 'probabilities', 'color']

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

            # add custom css for tooltip hovering, classes .custom-tooltip .tooltip-text .custom-tooltip .tooltip-text::after .custom-tooltip:hover .tooltip-text exist

    
            )   

    return plot, df_cluster



def select_cluster(df, selected_value):
    print("Selected value: ", selected_value)
    df_filtered = df[df["cluster_label"] == selected_value]    
    df_filtered.loc[:, 'size'] = 10

    tool_tip_list = ['title', 'abstract', 'color', 'cluster_label']
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
    return plot, df_filtered

def select_color(df, selected_value):
    print("Selected value: ", selected_value)
    df_filtered = df[df["color"] == selected_value]    
    df_filtered.loc[:, 'size'] = 10

    
    tool_tip_list = ['title', 'abstract', 'color']

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
    return plot, df_filtered

def update_textbox(selected_option):
    if selected_option == "k-Means":
        # If option 1 is selected, show textbox 1
        return gr.Textbox("10", label="Number of clusters to use in k-Means")
    elif selected_option == "HDBSCAN":
        # If option 2 is selected, show textbox 2
        return gr.Textbox(value="50", label="Minumum number or elements in HDBSCAN cluster", visible=True)
    else:
        return gr.Textbox(visible=False)  # Default behavior, hide textbox

def run_gradio(df, df_umpa):
    # create interactive gr.ScatterPlot

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
            with gr.Tab("Layered Clusters"):
                
                dataframe = gr.State(df.copy())
                with gr.Row():
                    # dropdown with the dimensionality reduction methods
                    method = gr.Dropdown(["UMAP", "t-SNE"], label="Method", value="UMAP")
                    clustering_method = gr.Dropdown(["k-Means", "HDBSCAN"], label="Method", value="k-Means")
                with gr.Row():
                    cluster_property = gr.Textbox("10", label="Number of clusters to use in k-Means")
                    property = gr.Textbox("None", label="Property (e.g., cancer,gene,virus)")
                    color = gr.Textbox("None", label="Color (e.g., blue,green,yellow)")
                with gr.Row():
                    apply_cluster_button = gr.Button("Apply Clustering")
                    apply_property_button = gr.Button("Apply Property")
                
                plot_output = scatter_plot.ScatterPlot()
                
                with gr.Row():
                    selected_cluster_values = gr.Textbox(label="Enter cluster label to keep")
                    selected_color_values = gr.Textbox(label="Enter color value to keep")
                with gr.Row():
                    filter_cluster_button = gr.Button("Filter by cluster")
                    filter_color_button = gr.Button("Filter by color")
                with gr.Row():
                    description = gr.Label("Instruct: Given a search query, retrieve relevant abstracts that answer the query.")
                    query = gr.Textbox("which nanoparticles improves delivery to cancer cells?", label="Query")
                    apply_query_button = gr.Button("Apply Query")
                    
                clustering_method.change(fn=update_textbox, inputs=clustering_method, outputs=cluster_property)
                
                apply_cluster_button.click(
                    cluster_embeddings,
                    inputs=[cluster_property, dataframe, method, clustering_method],
                    outputs=[plot_output, dataframe]
                )

                apply_property_button.click(
                    color_embeddings,
                    inputs=[property, color, dataframe],
                    outputs=[plot_output, dataframe]
                )
                
                filter_cluster_button.click(
                    select_cluster,
                    inputs=[dataframe, selected_cluster_values],
                    outputs=[plot_output, dataframe]
                )
                filter_color_button.click(
                    select_color,
                    inputs=[dataframe,  selected_color_values],
                    outputs=[plot_output, dataframe]
                )
                apply_query_button.click(
                    get_query_embedding,
                    inputs=[query, dataframe, method],
                    outputs=[plot_output, dataframe]
                )

    
            # add a tab that allows entering a query and then displays the most similar embeddings, use cosine similarity. the result should be a list of the titles of the most similar embeddings, showing the title and the abstract of the most similar embedding
            with gr.Tab("Semantic textual similarity"):
                dataframe = gr.State(df.copy())
                with gr.Row():
                    description = gr.Label("Instruct: Retrieve semantically similar text.")
                # Instruct: Retrieve semantically similar text.\nQuery: {query}
                with gr.Row():
                    # add label with discription
                    query = gr.Textbox("Nanoparticle delivery to solid tumours over the past ten years has slowed down", label="Query")
                    num_cases = gr.Textbox("10", label="Number of cases to return")
                with gr.Row():
                    apply_query_button = gr.Button("Apply search")

                df_output = gr.Dataframe()
                df_plot = scatter_plot.ScatterPlot()

                apply_query_button.click(
                    get_semantic_similar_embeddings,
                    inputs=[query, dataframe, num_cases],
                    outputs=[df_output, df_plot]
                )
            
            with gr.Tab("Retrieve question answering"):
                dataframe = gr.State(df.copy())
                with gr.Row():
                    description = gr.Label("Instruct: Given a search query, retrieve relevant abstracts that answer the query.")
                with gr.Row():
                    # add label with discription
                    query = gr.Textbox("which nanoparticles improves delivery to cancer cells?", label="Query")
                    num_cases = gr.Textbox("10", label="Number of cases to return")
                with gr.Row():
                    apply_query_button = gr.Button("Apply search")

                df_output = gr.Dataframe()
                df_plot = scatter_plot.ScatterPlot()

                apply_query_button.click(
                    get_retrieval_embeddings,
                    inputs=[query, dataframe, num_cases],
                    outputs=[df_output, df_plot]
                )

    # launch
    demo.launch(share=True)

# entry point for the gradio interface
if __name__ == "__main__":
    # load the data
    print("Starting the gradio interface")
    folder_path = 'embeddings_subset'
    df, embeddings = load_temp_data(folder_path)
    # create tsne embeddings
    # df_tsne, tsne_embeddings = create_tsne_embeddings(df, embeddings)
    # create umap embeddings
    df_umpa, umpa_embedding = create_umap_embeddings(df, embeddings)
    print("Finished loading the data")
    run_gradio(df, df_umpa)
    
