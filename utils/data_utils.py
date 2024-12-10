import os
import json
import pandas as pd
import time
import streamlit as st

@st.cache_resource(show_spinner="loading data...")
def load_data(json_path, subset=None):
    if os.path.exists(json_path):
        # read df with embeddings
        # load json file to pandas df
        with open(json_path, 'r') as f:
            data = json.load(f)

        # data to dataframe
        df = pd.DataFrame(data).T
        df['size'] = 10
        df['color'] = 'red'
        df['cluster_label'] = 0

        # rename the 'Article Title' column to 'title'
        df.rename(columns={'Article Title': 'title'}, inplace=True)

        # only keep the following columns: 'title', 'Abstract', 'embedding'
        df = df[['title', 'Abstract', 'embedding', 'cleaned_text', 'size', 'color', 'cluster_label']]

        if subset:
            df = df.head(subset)

        return df
    else:
       print("The file does not exist")

def write_df_to_excel(df, output_dir='output'):
    writer = pd.ExcelWriter
    temp_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"output_{temp_time}.xlsx"
    file_path = os.path.join(output_dir, filename)
    df.to_excel(file_path, index=False)
    print(f"Dataframe saved to {file_path}")