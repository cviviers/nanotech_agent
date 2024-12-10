
import streamlit as st

import os
import copy
# create interactive plot with gradio
import gradio as gr
from gradio.components import scatter_plot
from utils.utils import get_embedding_from_api, write_df_to_excel
from utils.lda_utils import create_lda_from_df, visualize_lda
import uvicorn
from data_functions import load_data, create_umap_embeddings, create_umap_scatter_plot, create_tsne_embeddings, cluster_embeddings, color_embeddings, select_cluster, select_color, get_query_embedding_and_similarity, apply_query_threshold, get_semantic_similar_embeddings, get_retrieval_embeddings, create_principle_component_plot, update_textbox, generate_and_visualize_lda, undo, crop_plot, generate_and_visualize_lda_all_clusters
import altair as alt

# entry point for the gradio interface
if __name__ == "__main__":

    # Set environment variables
    # os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    # os.environ['COMMANDLINE_ARGS']="--no-gradio-queue"
    # load the data
    print("Starting the streamlit interface")
    file_path = r'embeddings/data_embeddings.json'
    df = load_data(file_path)
    # select first 10000 rows
    # df = df.head(30000)

    print("Finished loading the data")

    # create umap embeddings
    df_umpa = create_umap_embeddings(df)

    plot = alt.Chart(df_umpa).mark_circle().encode(
        x='low_x',
        y='low_y',
        color='color',
        tooltip=['title', 'Abstract']
    ).interactive()

    st.title("Interactive Data Exploration")
    st.scatter_chart(plot, use_container_width=True)
