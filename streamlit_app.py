import os
from utils.data_utils import load_data, write_df_to_excel
from utils.cluster_utils import create_umap_embeddings, kmeans_cluster, assign_class_to_embeddings
from utils.lda_utils import generate_and_visualize_lda_all_clusters
import streamlit as st 
import altair as alt


# entry point for the streamlit app
def main():


    def init_session_variables():
        if 'kmeans' not in st.session_state:
            st.session_state['kmeans'] = 3
        if 'lda' not in st.session_state:
            st.session_state['lda'] = 5
        if 'x1' not in st.session_state:
            st.session_state['x1'] = 0
        if 'x2' not in st.session_state:
            st.session_state['x2'] = 1
        if 'y1' not in st.session_state:
            st.session_state['y1'] = 0
        if 'y2' not in st.session_state:
            st.session_state['y2'] = 1

        

    dafaframe = st.session_state['df']
    dataframe_history = st.session_state['df_history']
    
    st.title("Interactive WOS Data Exploration")
    col1, col2 = st.columns([0.8, 0.2], vertical_alignment="top")

    col1.title("Embeddings")
    to_empty = col1.container()
    col1_empty = to_empty.empty()


    plot = draw_plot(dafaframe)

    if 'plot' not in st.session_state:
        st.session_state['plot'] = plot

    with st.spinner('Wait for it...'):
        col1_empty.altair_chart(plot, use_container_width=True)


    col1_col1, col1_col2, col1_col3 = col1.columns([0.2, 0.2, 0.2], vertical_alignment="top")
    undo_button = col1_col1.button("Undo")

    excel_button = col1_col2.button("Save to Excel")
    refresh_button = col1_col3.button("Refresh")


    col2.title("Settings")
    k = col2.number_input("Number of k-mean clusters", min_value=1, max_value=100, value=3, key='kmeans')
    run_k_means_button = col2.button("Run K-means")

    lda_topics = col2.number_input("Number of LDA topics", min_value=1, max_value=100, value=5, key='lda')
    run_lda_button = col2.button("Run LDA")

    # bounding box selection
    col2.title("Bounding Box Selection")
    sub_col1, sub_col2 = col2.columns([0.5, 0.5], vertical_alignment="top")
    x1 = sub_col1.text_input("x1",  value=0, key='x1')
    x2 = sub_col2.text_input("x2",  value=1, key='x2')
    y1 = sub_col1.text_input("y1",  value=0, key='y1')
    y2 = sub_col2.text_input("y2",  value=1, key='y2')

    run_bounding_box_button = col2.button("Select Area")

    # filter by cluster numbers
    col2.title("Filter by Cluster")

    cluster_numbers = col2.multiselect("Select cluster numbers", list(dafaframe['cluster_label'].unique()))
    run_cluster_filter_button = col2.button("Filter by Cluster")
    col2.title("Filter by Keywords")
    filter_by_text = col2.text_input("Filter by text", value="cancer")
    assigned_class_intput = col2.text_input("Assigned class", value=1)
    col2_col1, col2_col2 = col2.columns([0.5, 0.5], vertical_alignment="top")
    run_assign_by_text_button = col2_col1.button("Assign by text")
    run_filter_by_text_button = col2_col2.button("Filter by text")

    if run_k_means_button:
        dafaframe = kmeans_cluster(dafaframe, num_clusters=k)
        plot = draw_plot(dafaframe)
        with st.spinner('Wait for it...'):
            col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())
        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot

    if run_lda_button:
        dafaframe = generate_and_visualize_lda_all_clusters(dafaframe, lda_topics)

    if run_bounding_box_button:
        dafaframe = dafaframe[(dafaframe['low_x'] >= float(x1)) & (dafaframe['low_x'] <= float(x2)) & (dafaframe['low_y'] >= float(y1)) & (dafaframe['low_y'] <= float(y2))]
        plot = draw_plot(dafaframe)
        with st.spinner('Wait for it...'):
            col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot

    if run_cluster_filter_button:
        dafaframe = dafaframe[dafaframe['cluster_label'].isin(cluster_numbers)]
        plot = draw_plot(dafaframe)
        with st.spinner('Wait for it...'):
            col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot

    if undo_button:
        if len(dataframe_history) > 1:
            dataframe_history.pop()
            dafaframe = dataframe_history[-1]
            plot = draw_plot(dafaframe)
            with st.spinner('Wait for it...'):
                col1_empty.altair_chart(plot, use_container_width=True)
            st.session_state['df'] = dafaframe
            st.session_state['df_history'] = dataframe_history
            st.session_state['plot'] = plot
        else:
            st.warning("Cannot undo further")

    if run_assign_by_text_button:

        dafaframe = assign_class_to_embeddings(filter_by_text, assigned_class_intput, dafaframe)
        plot = draw_plot(dafaframe)
        dataframe_history.append(dafaframe.copy())
        with st.spinner('Wait for it...'):
            col1_empty.altair_chart(plot, use_container_width=True)
        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot
    
    if run_filter_by_text_button:
        assigned_class_intput = assigned_class_intput.split(',')
        dafaframe = dafaframe[dafaframe['cluster_label'].isin(assigned_class_intput)]
        plot = draw_plot(dafaframe)
        with st.spinner('Wait for it...'):
            col1_empty.altair_chart(plot, use_container_width=True)
        dataframe_history.append(dafaframe.copy())

        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot

    if excel_button:
        write_df_to_excel(dafaframe)
        st.success("Dataframe saved to excel")

    if refresh_button:
        st.session_state['df'] = dafaframe
        st.session_state['df_history'] = dataframe_history
        st.session_state['plot'] = plot

@st.cache_data(show_spinner="Compiling figure...")
def draw_plot(df):

    # choose color scheme
    color_scheme = 'tableau20'
    plot = alt.Chart(df).mark_circle().encode(
        x='low_x',
        y='low_y',
        color=alt.Color('cluster_label:N', scale=alt.Scale(scheme=color_scheme)),
        size='size',
        opacity=alt.value(0.5),
        tooltip=['title', 'Abstract', 'cluster_label']	
    ).properties(
        width='container',
        height=1200,
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
    
    # print("Starting loading the data")
    file_path = r'wos_data/data_embeddings.json'
    
    if 'df' not in st.session_state:
        df = load_data(file_path, subset=None)
        print("Finished loading the data")

        # create umap embeddings
        df_umpa = create_umap_embeddings(df)
        st.session_state['df'] = df_umpa

    if 'df_history' not in st.session_state:
        st.session_state['df_history'] = [st.session_state.df]

    # create output folder if not exists
    if not os.path.exists('output'):
        os.makedirs('output')

    # print("Starting the streamlit interface")
    main()
    