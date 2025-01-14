import numpy as np
import umap
import hdbscan
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import streamlit as st
import matplotlib.pyplot as plt


def create_umap_embeddings(df):
    reducer = umap.UMAP(random_state=42)
    embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
    embeddings = np.stack(embeddings)
    umpa_embedding = reducer.fit_transform(embeddings)
    df_umpa = df.copy()
    df_umpa['low_x'] = umpa_embedding[:, 0]
    df_umpa['low_y'] = umpa_embedding[:, 1]
    df_umpa['size'] = 20
    df_umpa['color'] = "blue"

    return df_umpa

def kmeans_cluster(df, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(df['embedding'].map(lambda x: np.array(x)).tolist())
    labels_as_strings = [str(label) for label in kmeans.labels_]

    df.loc[:, 'cluster_label'] = labels_as_strings
    # df['cluster_label'] = labels_as_strings

    return df

def assign_class_to_embeddings(property, classes, dataframe):
    # get the embeddings from the dataframe

    properties = property.split(',')
    classes = classes.split(',')

    print(properties, classes)

    # set all cluster labels to Not Assigned
    dataframe['cluster_label'] = 'Not Assigned'

    if len(properties) > 1 and len(classes) == 1:
        # apply the same color to all the properties
        classes = classes * len(properties)

    else:
        if len(properties) != len(classes):
            raise st.error("The number of properties and colors should be the same")
     
    # color the embeddings according to the property, replace the color of the embeddings that contain the property with the color
    for prop, col in zip(properties, classes):
        mask = dataframe['Abstract'].fillna('Not Available').str.contains(prop, case=False, na=False)
        # replace the color of the embeddings that contain the property with the color
        # print number of embeddings that contain the property
        print(f"Number of embeddings that contain the property '{prop}': {mask.sum()}")
        dataframe.loc[mask, 'cluster_label'] = col

    return dataframe

def plot(data, labels):
    """Plots the data coloured by labels, with noise points in silver."""
    noise_mask = labels == -1
    plt.scatter(data[noise_mask, 0], data[noise_mask, 1], 1, color="silver")
    plt.scatter(
        data[~noise_mask, 0],
        data[~noise_mask, 1],
        1,
        labels[~noise_mask] % 10,
        cmap="tab10",
        vmin=0,
        vmax=9,
    )
    plt.axis("off")
    plt.show()