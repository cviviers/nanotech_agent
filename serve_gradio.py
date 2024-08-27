
import os
import getpass
import json

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from ast import literal_eval
# create interactive plot with gradio
import gradio as gr
from gradio.components import scatter_plot

# load the embeddings from the json files
folder_path = 'embeddings'
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
data = {}
for file in json_files:
    with open(os.path.join(folder_path, file), 'r') as f:
        data[file] = json.load(f)

# data to dataframe
df = pd.DataFrame(data).T
df['size'] = 8
df['color'] = 'blue'

print(df.head())
print(df)
# store the embeddings in a numpy array
embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))

# use the embeddings to create a t-SNE plot
# get the embeddings from the dataframe
# embeddings = df['embedding'].apply(lambda x: np.array(x)).values

# create a t-SNE object
tsne = TSNE(n_components=2, random_state=42, init='random', learning_rate=200, max_iter=1000)

# fit the t-SNE object to the embeddings
tsne_embeddings = tsne.fit_transform(np.stack(embeddings[:]))

print(tsne_embeddings.shape)

# create new pandas df with old df added the tsne embeddings
df_tsne = df.copy()

df_tsne['tsne_x'] = tsne_embeddings[:, 0]
df_tsne['tsne_y'] = tsne_embeddings[:, 1]

embeddings = np.array(df['embedding'].map(lambda x: np.array(x[0])))
embeddings = np.stack(embeddings)

# add interactive tab to choose the dimension of the embeddings and then plt the embeddings, two values between 0 and 1536
# create a function that takes the dimension and the data and returns the embeddings

# hovering over a point should display the title of the article and change the cursor to a hand
# create a scatter plot of the embeddin
def get_embeddings(dim1, dim2):
    # get the embeddings from the data
    embeddings = np.array(df['embedding'].map(lambda x: np.array(x)))
    embeddings = np.stack(embeddings)

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

# create interactive gr.ScatterPlot

with gr.Blocks() as demo:
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

            

# launch
demo.launch(share=True)
