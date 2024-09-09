
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

reducer = umap.UMAP()
alt.data_transformers.disable_max_rows()

if os.path.exists('df_embeddings.csv'):
    df = pd.read_csv('df_embeddings.csv', index_col=0)
    df['embedding'] = df['embedding'].apply(literal_eval)
    embeddings = np.stack(df['embedding'].values)

if os.path.exists('tsne_embeddings.csv'):
    df_tsne = pd.read_csv('tsne_embeddings.csv', index_col=0)
    tsne_embeddings = np.stack(df_tsne['embedding'].apply(literal_eval))

else:

    # load the embeddings from the json files
    folder_path = 'embeddings'
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

    # filter the dataframe to only include journals from the following list
    # List of journals to include
    journals_to_include = [
        "ACS Applied Materials & Interfaces",
        "ACS Nano",
        "Advanced Functional Materials",
        "Advanced Materials",
        "Angewandte Chemie",
        "Biology and Medicine",
        "Biomaterials",
        "Cell",
        "Clinical Cancer Research",
        "Frontiers in Nanotechnology",
        "Immunity",
        "International Journal of Nanomedicine",
        "Journal of Controlled Release",
        "Journal of Materials Chemistry B",
        "Matter",
        "Molecular Therapy",
        "Nano Letters",
        "Nano Micro Small",
        "Nano Research",
        "Nanomedicine",
        "Nanomedicine: Nanotechnology",
        "Nanoscale",
        "Nature",
        "Nature Biomedical Engineering",
        "Nature Cancer",
        "Nature Communications",
        "Nature Materials",
        "Nature Medicine",
        "Nature Nanotechnology",
        "NPG Asia Materials",
        "Pharmaceutics",
        "PNAS",
        "Science",
        "Science Advances",
        "Science Translational Medicine",
        "Scientific Reports",
        "Small"
    ]

    # journals as small letters
    journals_to_include = [journal.lower() for journal in journals_to_include]

    # Exclusion criteria
    keywords_exclusion = ["review", "not available"]

    # filter the dataframe to only include journals from the list
    df = df[df['journal'].str.lower().isin(journals_to_include)]

    # filter the dataframe to exclude titles with keywords from the exclusion list
    df = df[~df['title'].str.lower().str.contains('|'.join(keywords_exclusion))]

    




    # store the embeddings in a numpy array
    embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
    embeddings = np.stack(embeddings)
    # use the embeddings to create a t-SNE plot
    # get the embeddings from the dataframe
    # embeddings = df['embedding'].apply(lambda x: np.array(x)).values

    # create a t-SNE object
    tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)

    # fit the t-SNE object to the embeddings
    tsne_embeddings = tsne.fit_transform(np.stack(embeddings[:]))

    # create new pandas df with old df added the tsne embeddings
    df_tsne = df.copy()

    df_tsne['tsne_x'] = tsne_embeddings[:, 0]
    df_tsne['tsne_y'] = tsne_embeddings[:, 1]

    # store the dataframe as csv file
    if not os.path.exists('df_embeddings.csv'):
        df.to_csv('df_embeddings.csv')

    # store the embeddings as csv file
    if not os.path.exists('tsne_embeddings.csv'):
        df_tsne.to_csv('tsne_embeddings.csv')

    

# create a scatter plot of the embeddin
def get_embeddings(dim1, dim2):

    # create new pandas df with old df added the tsne embeddings
    df_embed = df.copy()
    df_embed['x'] = embeddings[:, int(dim1)]
    df_embed['y'] = embeddings[:, int(dim2)]

    plot = scatter_plot.ScatterPlot(
                value=df_embed,
                x="x",
                y="y",
                title="Embeddings of abstracts",
                color='color',
                size= 'size',
                # tooltip displays the title of the article
                tooltip=['title', 'abstract'],
                width=600,
                height=600
            )

    return plot

def get_embedding_from_api(text, url="http://localhost:8000/embed"):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()["embedding"], response.json()["num_tokens"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def get_similar_embeddings(query):
    # get the embedding from the api
    embedding, num_tokens = get_embedding_from_api(query)

    # calculate the cosine similarity between the query embedding and the embeddings in the dataframe
    cosine_similarities = np.dot(embeddings, embedding)
    # get the indices of the top 10 most similar embeddings
    top_indices = np.argsort(cosine_similarities)[::-1][:10]
    # get the titles of the most similar embeddings
    similar_titles = df.iloc[top_indices]['title'].tolist()
    # get the abstracts of the most similar embeddings
    similar_abstracts = df.iloc[top_indices]['abstract'].tolist()

    # create a dataframe with the most similar embeddings
    similar_df = df.iloc[top_indices].copy()
    similar_df['size'] = 10
    similar_df['color'] = 'red'

    return similar_df

def get_query_embedding(query):
    # get the embedding from the api
    embedding, num_tokens = get_embedding_from_api(query)

    # add the embedding to the dataframe
    df_query = df.copy()
    df_query['size'] = 10
    df_query['color'] = 'red'
    # add the query embedding to the dataframe with the title "Query" and abstract the value of the query, all other fields 'Not Available'
    df_query.loc[0] = {'title': 'Query', 'abstract': query, 'embedding': embedding,  'color': 'blue', 'size': 12}
    # get the tsne embeddings of the query
    query_embedding = np.array(df_query['embedding'].map(lambda x: np.array(x)))
    query_embedding = np.stack(query_embedding)
    query_tsne = tsne.fit_transform(query_embedding)
    df_query['tsne_x'] = query_tsne[:, 0]
    df_query['tsne_y'] = query_tsne[:, 1]

    plot = scatter_plot.ScatterPlot(
            value=df_query,
            x="tsne_x",
            y="tsne_y",
            title="Embedding Query",
            color='color',
            size= 'size',
            # tooltip displays the title of the article
            tooltip=['title', 'abstract'],
            width=600,
            height=600
        )

    return plot

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
            x="tsne_x",
            y="tsne_y",
            title="Color embeddings",
            color='color',
            size= 'size',
            tooltip=['title', 'abstract'],
            width=600,
            height=600
            )
    
    return plot, df_color

def umap_embedding():  
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
            width=600,
            height=600
            )
    return plot

def cluster_embeddings(min_cluster_size=10):
        # a tab that clusters the embeddings using HDBSCAN and displays the clusters in the scatter plot
        # the result should be a scatter plot with the clusters colored differently
        # the plot should show the title and abstract of the embeddings
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size),  prediction_data=True, branch_detection_data=True, alpha=0.5, cluster_selection_method='leaf').fit(tsne_embeddings)
        n_clusters_ = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        
        
        df_cluster = df_tsne.copy()

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
                x="tsne_x",
                y="tsne_y",
                title="Cluster",
                color='color',
                size= 'size',
                tooltip=['title', 'abstract'],
                width=600,
                height=600
                )

        return plot

def cluster_embeddings_kmeans(num_clusters, dataframe):
    num_clusters = int(num_clusters)
    local_tsn_embeddings = dataframe['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    local_tsn_embeddings = np.stack(local_tsn_embeddings)
    kmeans = KMeans(n_clusters=int(num_clusters), random_state=42).fit(local_tsn_embeddings)
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



    plot = scatter_plot.ScatterPlot(
            value=df_cluster_kmeans,
            x="tsne_x",
            y="tsne_y",
            title="Cluster KMEANS",
            color='color',
            size= 'size',
            tooltip=['title', 'abstract', 'cluster_label'],
            width=600,
            height=600
            )   

    return plot, df_cluster_kmeans

def apply_clustering_or_property(num_clusters, property, color, dataframe):
    if property == "None" and color == "None":
        plot, df_result = cluster_embeddings_kmeans(num_clusters, dataframe)
    elif property != "None" and color != "None":
        plot, df_result = color_embeddings(property, color, dataframe)
    else:
        # throw error
        print("Please enter a property and a color.")
        return None, None
    return plot, df_result

def select_and_filter(df, selection_column, selected_value):
    print(f"Selected column: {selection_column}")
    df_filtered = df[df[selection_column] == selected_value]    
    df_filtered['size'] = 10

    if selection_column == 'cluster_label':
        tool_tip_list = ['title', 'abstract', 'color', 'cluster_label']
    else:
        tool_tip_list = ['title', 'abstract', 'color']

    # print number of elements in the filtered dataframe
    print(f"Number of elements in the filtered dataframe: {len(df_filtered)}")
    
    # Ensure the DataFrame is not empty
    if df_filtered.empty:
        print("No data matches the selected criteria.")
        return None, df_filtered
    
    df_filtered['color'] = 'red'
    plot = scatter_plot.ScatterPlot(
        value=df_filtered,
        x="tsne_x",
        y="tsne_y",
        title="Filtered Plot",
        color='color',
        size='size',
        tooltip=tool_tip_list,
        width=600,
        height=600
    )
    return plot, df_filtered



# create interactive gr.ScatterPlot

with gr.Blocks() as demo:
        
        with gr.Tab("Layered Clusters"):
            
            dataframe = gr.State(df_tsne.copy())
            with gr.Row():
                num_clusters = gr.Textbox("10", label="Number of Clusters (enter 'None' if not clustering)")
                property = gr.Textbox("None", label="Property (e.g., cancer,gene,virus)")
                color = gr.Textbox("None", label="Color (e.g., blue,green,yellow)")
            
            apply_button = gr.Button("Apply Clustering/Property")
            
            plot_output = scatter_plot.ScatterPlot()
            
            with gr.Row():
                selection_column = gr.Dropdown(["cluster_label", "color"], label="Select by")
                selected_values = gr.Textbox(label="Enter value to keep")
            
            filter_button = gr.Button("Filter Selection")
            
            apply_button.click(
                apply_clustering_or_property,
                inputs=[num_clusters, property, color, dataframe],
                outputs=[plot_output, dataframe]
            )
            
            filter_button.click(
                select_and_filter,
                inputs=[dataframe, selection_column, selected_values],
                outputs=[plot_output, dataframe]
            )

        with gr.Tab("TSNE"):
            scatter_plot.ScatterPlot(
                value=df_tsne,
                x="tsne_x",
                y="tsne_y",
                title="Embeddings of abstracts",
                color='color',
                # tooltip displays the title of the article
                # create list with the value 10 repeated for the length of the dataframe
                size= 'size',
                tooltip=['title', 'abstract'],
                width=800,
                height=800
            )
        with gr.Tab("Specific dimensions"):
           
                gr.Interface(get_embeddings, [gr.Textbox("0", label="Dimension 1", info="Enter a value between 0 and 1024", min_width=200),
                                            gr.Textbox("1", label="Dimension 2", info="Enter a value between 0 and 1024", min_width=200)], scatter_plot.ScatterPlot(width=600), title="Embeddings of abstracts")
                
        # add a tab that allows entering a query and then displays the most similar embeddings, use cosine similarity. the result should be a list of the titles of the most similar embeddings, showing the title and the abstract of the most similar embedding
        with gr.Tab("Query"):
            query = gr.Textbox("Enter a query", label="Query")

            # display results in gr.dataframe
            gr.Interface(get_similar_embeddings, query, gr.Dataframe(headers=["title", "abstract"], row_count=10), title="Most similar embeddings")

        # add a tab that embeds a query and displays the tsne plot of the embeddings with the query in blue
        with gr.Tab("Embedding Query"):
            query = gr.Textbox("Enter a query", label="Query") 

            gr.Interface(get_query_embedding, query, scatter_plot.ScatterPlot(width=600), title="Embedding Query")

        with gr.Tab("Color embeddings"):
            dataframe_color = gr.State(df_tsne.copy())
            color_output = scatter_plot.ScatterPlot(width=600)
            # enter property and an associated color. The property will be searched in the abstract and the embeddings will be colored according to the property. 
            # The color will be a string with the color name, e.g. "blue", "red", "green", etc.
            # Multiple properties can be entered, separated by commas.
            property = gr.Textbox("Enter a property eg. cancer,gene,virus", label="Property")
            color = gr.Textbox("Enter a color eg. blue,green,yellow", label="Color")

            # add label field with additional information
            desc = 'Enter a property and an associated color. The property will be searched in the abstract and the embeddings will be colored according to the property. The color will be a string with the color name, e.g. "blue", "red", "green", etc. Multiple properties can be entered, separated by commas. The colors will be overwritten in the order of the properties entered.'


            gr.Interface(color_embeddings, [property, color, dataframe_color], outputs=[color_output, dataframe_color], title="Color embeddings", description=desc)
        with gr.Tab("UMAP"):
            gr.Interface(umap_embedding,None, scatter_plot.ScatterPlot(width=600), title="UMAP")
        with gr.Tab("Cluster HDBSCAN"):
            min_cluster_size = gr.Textbox("10", label="Min Cluster Size")
            gr.Interface(cluster_embeddings,min_cluster_size, scatter_plot.ScatterPlot(width=600), title="Cluster")
        with gr.Tab("Cluster KMEANS"):
            dataframe_cluster = gr.State(df_tsne.copy())
            cluter_plot = scatter_plot.ScatterPlot(width=600)
            num_clusters = gr.Textbox("10", label="Number of Clusters")
            gr.Interface(cluster_embeddings_kmeans, [num_clusters, dataframe_cluster], [cluter_plot, dataframe_cluster], title="Cluster")
        




# launch
demo.launch(share=False)
