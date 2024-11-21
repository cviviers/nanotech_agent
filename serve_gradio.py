
import os
import copy
# create interactive plot with gradio
import gradio as gr
from gradio.components import scatter_plot
from utils.utils import get_embedding_from_api, write_df_to_excel
from utils.lda_utils import create_lda_from_df, visualize_lda

from data_functions import load_temp_data, create_umap_embeddings, create_tsne_embeddings, cluster_embeddings, color_embeddings, select_cluster, select_color, get_query_embedding_and_similarity, apply_query_threshold, get_semantic_similar_embeddings, get_retrieval_embeddings, create_principle_component_plot, update_textbox, generate_and_visualize_lda, undo, crop_plot, generate_and_visualize_lda_all_clusters

    
def run_gradio(df):
    # create interactive gr.ScatterPlot

    # gr.State( )

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
            with gr.Tab("Layered Clusters"):
                
                dataframe = gr.State(df.copy())
                df_history_list = gr.State()

                def setup():

                    # create a list to store the history of dataframes
                    df_history_list = []
                    df_history_list.append((scatter_plot.ScatterPlot(value=dataframe,x="low_x", y="low_y",title="UMAP embeddings",color='color',size= 'size',width=1200, height=1200), df.copy()))
                    return df_history_list

                demo.load(setup, inputs=[], outputs=[df_history_list])

                # Add dataframes to the list
                # df_history_list = add_to_state_list(df_history_list, dataframe)
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
                    start_x = gr.Number(value=0, label="Start x")
                    end_x = gr.Number(value=1, label="End x")
                    start_y = gr.Number(value=0, label="Start y")
                    end_y = gr.Number(value=1, label="End y")
                    crop_button = gr.Button("Crop")


                with gr.Row():
                    undo_button = gr.Button("Undo")
                
                with gr.Row():
                    selected_cluster_values = gr.Textbox(label="Enter cluster label to keep")
                    selected_color_values = gr.Textbox(label="Enter color value to keep")
                with gr.Row():
                    filter_cluster_button = gr.Button("Filter by cluster")
                    filter_color_button = gr.Button("Filter by color")
                with gr.Row():
                    description = gr.Label("Instruct: Given a search query, retrieve relevant abstracts that answer the query.",)
                with gr.Row():
                    query = gr.Textbox("which nanoparticles improves delivery to cancer cells?", label="Query")
                with gr.Row():
                    apply_query_button = gr.Button("Apply Query")
                    query_threshold = gr.Textbox("0.65", label="Threshold")
                    apply_query_threshold_button = gr.Button("Apply Query with Threshold")
                    apply_pca_button = gr.Button("Apply PCA")

                with gr.Row():
                    description = gr.Label("Extract data from the processed embeddings.")
                with gr.Row():
                    extract_to_excel = gr.Button("Extract to Excel")
                    num_topics = gr.Textbox("5", label="No. of topics")
                    generate_LDA = gr.Button("Generate LDA")

                    
                clustering_method.change(fn=update_textbox, inputs=clustering_method, outputs=cluster_property)
                
                apply_cluster_button.click(
                    cluster_embeddings,
                    inputs=[cluster_property, dataframe, df_history_list, method, clustering_method ],
                    outputs=[plot_output, dataframe, df_history_list]
                )

                apply_property_button.click(
                    color_embeddings,
                    inputs=[property, color, dataframe, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )
                
                filter_cluster_button.click(
                    select_cluster,
                    inputs=[dataframe, selected_cluster_values, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )
                filter_color_button.click(
                    select_color,
                    inputs=[dataframe,  selected_color_values, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )
                apply_query_button.click(
                    get_query_embedding_and_similarity,
                    inputs=[query, dataframe, method, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )

                apply_query_threshold_button.click(
                    apply_query_threshold,
                    inputs=[query_threshold, dataframe, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )

                extract_to_excel.click(
                    write_df_to_excel,
                    inputs=dataframe,
                )

                generate_LDA.click(
                    generate_and_visualize_lda_all_clusters,
                    inputs=[dataframe, num_topics]
                )

                apply_pca_button.click(
                    create_principle_component_plot,
                    inputs=[dataframe, query, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )

                

                undo_button.click(
                    undo,
                    inputs=[df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]
                )

                crop_button.click(
                    crop_plot,
                    inputs=[start_x, end_x, start_y, end_y, dataframe, df_history_list],
                    outputs=[plot_output, dataframe, df_history_list]  
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
    demo.launch(share=False)
    



# entry point for the gradio interface
if __name__ == "__main__":
    # load the data
    print("Starting the gradio interface")
    folder_path = 'embeddings_subset'
    df, embeddings = load_temp_data(folder_path)
    print("Finished loading the data")

    

    

    # create tsne embeddings
    # df_tsne, tsne_embeddings = create_tsne_embeddings(df, embeddings)
    # create umap embeddings
    df_umpa, umpa_embedding = create_umap_embeddings(df, embeddings)

    # add color to the df
    df_umpa['color'] = 'blue'
    df_umpa['size'] = 10
 

    # create output folder if not exists
    if not os.path.exists('output'):
        os.makedirs('output')
    run_gradio(df_umpa)
    
