
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
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)
  
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)
    
    # calculate the dot product of the embeddings with the query embedding
    similarities = np.dot(embeddings, embedding)



    # get the indices of the top 10 most similar embeddings
    sorted_indices = np.argsort(similarities)[::-1]
    # sorted_similarities = similarities[sorted_indices]
    top_indices = sorted_indices[:num_cases]

    # create a dataframe with the most similar embeddings
    similar_df = df.copy()
    similar_df['similarity'] = similarities

    # sort df by similarity
    similar_df = similar_df.sort_values(by='similarity', ascending=False)
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
    top_similar_df = similar_df.iloc[top_indices]
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
        if embedding is None:
            raise gr.Error("No embedding returned. The embeddng server could be down", duration=30)
    except:
        raise gr.Error("Please enter a valid query", duration=30)
  
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings = np.stack(embeddings)
    print(embeddings.shape)
    print(embedding.shape)
    # calculate the dot product of the embeddings with the query embedding
    similarities = np.dot(embeddings, embedding)
    print(similarities.shape)



    # get the indices of the top 10 most similar embeddings
    sorted_indices = np.argsort(similarities)[::-1]
    # sorted_similarities = similarities[sorted_indices]
    top_indices = sorted_indices[:num_cases]

    # create a dataframe with the most similar embeddings
    similar_df = df.copy()
    similar_df['similarity'] = similarities

    # sort df by similarity
    similar_df = similar_df.sort_values(by='similarity', ascending=False)
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
    top_similar_df = similar_df.iloc[top_indices]
    return top_similar_df, plot


def get_query_embedding(query, df, dim_reduction='UMAP'):
    # get the embedding from the api
    try:
        embedding, num_tokens = get_embedding_from_api(query)
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

def cluster_embeddings(df, embeddings, min_cluster_size=10):
    # a tab that clusters the embeddings using HDBSCAN and displays the clusters in the scatter plot
    # the result should be a scatter plot with the clusters colored differently
    # the plot should show the title and abstract of the embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),  prediction_data=True, branch_detection_data=True, alpha=0.5, cluster_selection_method='leaf').fit(embeddings)
    n_clusters_ = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    
    
    df_cluster = df

    # save the hierarchy plot as png
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    color_palette = sns.color_palette('Paired', n_clusters_)
    cluster_colors = [color_palette[np.argmax(x)]
                    for x in soft_clusters]
    # plt.scatter(*tsne_embeddings.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    # plt.savefig('hierarchy_plot.png')


    # Number of clusters in labels, ignoring noise if present.

    hdb_colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters_))
    df_cluster['cluster_labels'] = clusterer.labels_  # Use labels instead of the entire clusterer object


    df_cluster['size'] = 10
    # change the color of the embeddings according to the cluster
    # convert the cluster_labels to colors
    # df_cluster['color'] = df_cluster['cluster_labels'].apply(lambda x: hdb_colors[x])
    cluster_colors_str = [str(x) for x in cluster_colors]
    df_cluster['color'] = cluster_colors_str

    plot = scatter_plot.ScatterPlot(
            value=df_cluster,
            x="low_x",
            y="low_y",
            title="Cluster",
            color='color',
            size= 'size',
            tooltip=['title', 'abstract'],
            width=1200,
            height=1200
            )

    return plot

def cluster_embeddings_kmeans(num_clusters, dataframe, dim_reduction='UMAP'):
    num_clusters = int(num_clusters)

    local_embeddings = dataframe['embedding'].apply(lambda x: np.array(x))
    local_embeddings = np.stack(local_embeddings)

    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42).fit(local_embeddings)
    kmeans_labels = kmeans.labels_
    kmeans_labels_as_str = [str(x) for x in kmeans_labels]

    df_cluster_kmeans = dataframe

    # df_cluster_kmeans['cluster_labels'] = kmeans.labels_
    df_cluster_kmeans['size'] = 10
    

    colors_keys = list(mcolors.XKCD_COLORS.keys())
    colors_values = list(mcolors.XKCD_COLORS.values())
    
    # make a copy of the df and add the kmeans.labels_ to the df as a new entry with the title the cluster center and the abstract the cluster center and the color a unique color from the color palette

    # predict the cluster of the embeddings
    # add the cluster labels to the dataframe
    df_cluster_kmeans['cluster_label'] = kmeans_labels_as_str
    df_colors = [str(colors_values[x]) for x in kmeans_labels]

    # color each embedding according to the cluster label
    # get index of label in kmeans.cluster_centers_
    df_cluster_kmeans['color'] = df_colors

    if dim_reduction == 'UMAP':
        reducer = umap.UMAP(random_state=42)
        umpa_embedding = reducer.fit_transform(local_embeddings)
        df_cluster_kmeans['low_x'] = umpa_embedding[:, 0]
        df_cluster_kmeans['low_y'] = umpa_embedding[:, 1]
    elif dim_reduction == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)
        tsne_embedding = tsne.fit_transform(local_embeddings)
        df_cluster_kmeans['low_x'] = tsne_embedding[:, 0]
        df_cluster_kmeans['low_y'] = tsne_embedding[:, 1]



    plot = scatter_plot.ScatterPlot(
            value=df_cluster_kmeans,
            x="low_x",
            y="low_y",
            title="Cluster KMEANS",
            color='color',
            size= 'size',
            tooltip=['title', 'abstract', 'cluster_label'],
            width=1200,
            height=1200,

            # add custom css for tooltip hovering, classes .custom-tooltip .tooltip-text .custom-tooltip .tooltip-text::after .custom-tooltip:hover .tooltip-text exist

    
            )   

    return plot, df_cluster_kmeans



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



def run_gradio(df, df_umpa):
    # create interactive gr.ScatterPlot

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
            with gr.Tab("Layered Clusters"):
                
                dataframe = gr.State(df.copy())
                with gr.Row():
                    # dropdown with the dimensionality reduction methods
                    method = gr.Dropdown(["UMAP", "t-SNE"], label="Method", value="UMAP")

                with gr.Row():
                    num_clusters = gr.Textbox("10", label="Number of Clusters (enter 'None' if not clustering)")
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
                    query = gr.Textbox("Enter a query", label="Query")
                    apply_query_button = gr.Button("Apply Query")
                    
                
                apply_cluster_button.click(
                    cluster_embeddings_kmeans,
                    inputs=[num_clusters, dataframe, method],
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
                    query = gr.Textbox("Enter a query", label="Query")
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
                    query = gr.Textbox("Enter a query", label="Query")
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

            
   
            # add a tab that embeds a query and displays the tsne plot of the embeddings with the query in blue
            # with gr.Tab("Embedding Query"):
            #     dataframe = gr.State(df.copy())
            #     with gr.Row():
            #         query = gr.Textbox("Enter a query", label="Query")
            #         method = gr.Dropdown(["UMAP", "t-SNE"], label="Method")
            #     with gr.Row():
            #         apply_query_button = gr.Button("Apply Query")

            #     plot_output = scatter_plot.ScatterPlot()

            #     apply_query_button.click(
            #         get_query_embedding,
            #         inputs=[query, dataframe, method],
            #         outputs=[plot_output]
            #     )

            # with gr.Tab("Cluster HDBSCAN (Under construction)"):

            #     dataframe = gr.State(df.copy())
            #     with gr.Row():
            #         min_cluster_size = gr.Textbox("10", label="Min Cluster Size")
            #     with gr.Row():
            #         apply_cluster_button = gr.Button("Apply Clustering")
            #     plot_output = scatter_plot.ScatterPlot()
            #     apply_cluster_button.click(
            #         cluster_embeddings,
            #         inputs=[dataframe, embeddings, min_cluster_size],
            #         outputs=[plot_output]
            #     )

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
    
