import os
from utils.data_utils import load_data
from utils.cluster_utils import create_umap_embeddings, kmeans_cluster, assign_class_to_embeddings
from utils.lda_utils import generate_and_visualize_lda_all_clusters
import streamlit as st 
import altair as alt


# entry point for the streamlit app
def main(dafaframe):

    dataframe_history = []
    dataframe_history.append(dafaframe.copy())

    
    st.title("Interactive WOS Data Exploration")
    col1, col2 = st.columns([0.8, 0.2], vertical_alignment="top")

    col1.title("Embeddings")
    to_empty = col1.container()
    col1_empty = to_empty.empty()


    plot = draw_plot(dafaframe)


    col1_empty.altair_chart(plot, use_container_width=True)
    undo_button = col1.button("Undo")


    col2.title("Settings")
    k = col2.number_input("Number of k-mean clusters", min_value=1, max_value=100, value=3)
    run_k_means_button = col2.button("Run K-means")

    lda_topics = col2.number_input("Number of LDA topics", min_value=1, max_value=100, value=5)
    run_lda_button = col2.button("Run LDA")

    # bounding box selection
    col2.title("Bounding Box Selection")
    sub_col1, sub_col2 = col2.columns([0.5, 0.5], vertical_alignment="top")
    x1 = sub_col1.text_input("x1",  value=0)
    x2 = sub_col2.text_input("x2",  value=1)
    y1 = sub_col1.text_input("y1",  value=0)
    y2 = sub_col2.text_input("y2",  value=1)

    run_bounding_box_button = col2.button("Select Area")

    # filter by cluster numbers
    col2.title("Filter by Cluster")

    cluster_numbers = col2.multiselect("Select cluster numbers", list(dafaframe['cluster_label'].unique()))
    run_cluster_filter_button = col2.button("Filter by Cluster")
    col2.title("Filter by Keywords")
    filter_by_text = col2.text_input("Filter by text", value="cancer")
    assigned_class_intput = col2.text_input("Assigned class", value=1)
    run_filter_by_text_button = col2.button("Filter by text")

    if run_k_means_button:
        dafaframe = kmeans_cluster(dafaframe, num_clusters=k)
        plot = draw_plot(dafaframe)
        col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

    if run_lda_button:
        
        dafaframe = generate_and_visualize_lda_all_clusters(dafaframe, lda_topics)

    if run_bounding_box_button:
        # df_filtered = dataframe[(dataframe['low_x'] >= start_x) & (dataframe['low_x'] <= end_x) & (dataframe['low_y'] >= start_y) & (dataframe['low_y'] <= end_y)]
        dafaframe = dafaframe[(dafaframe['low_x'] >= float(x1)) & (dafaframe['low_x'] <= float(x2)) & (dafaframe['low_y'] >= float(y1)) & (dafaframe['low_y'] <= float(y2))]
        plot = draw_plot(dafaframe)
        col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

    if run_cluster_filter_button:
        dafaframe = dafaframe[dafaframe['cluster_label'].isin(cluster_numbers)]
        plot = draw_plot(dafaframe)
        col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

    if undo_button:
        if len(dataframe_history) > 1:
            dataframe_history.pop()
            dafaframe = dataframe_history[-1]
            plot = draw_plot(dafaframe)
            col1_empty.altair_chart(plot, use_container_width=True)
        else:
            st.warning("Cannot undo further")

    if run_filter_by_text_button:

        dafaframe = assign_class_to_embeddings(filter_by_text, assigned_class_intput, dafaframe)
        plot = draw_plot(dafaframe)
        dataframe_history.append(dafaframe.copy())

@st.cache_data(show_spinner="Compiling figure...")
def draw_plot(df):
    plot = alt.Chart(df).mark_circle().encode(
        x='low_x',
        y='low_y',
        color='cluster_label',
        size='size',
        opacity=alt.value(0.5),
        tooltip=['title', 'Abstract', 'cluster_label']	
    ).properties(
        width='container',
        height=1000
    ).interactive()

    return plot

# entry point for the streamlit interface
if __name__ == "__main__":
    st.set_page_config(layout="wide")

    if not os.path.exists('.streamlit'):
        os.makedirs('.streamlit')

        # create a config file for streamlit
        with open(".streamlit/config.toml", "w") as f:
            f.write("[server]\n")
            f.write("maxMessageSize = 1000\n")
    
    print("Starting loading the data")
    file_path = r'wos_data/data_embeddings.json'
    df = load_data(file_path)

    # load first 1000 rows
    df = df.head(1000)

    print("Finished loading the data")

    # create umap embeddings
    df_umpa = create_umap_embeddings(df)

    # create output folder if not exists
    if not os.path.exists('output'):
        os.makedirs('output')

    print("Starting the streamlit interface")
    main(df_umpa)
    