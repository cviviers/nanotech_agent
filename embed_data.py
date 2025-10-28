# %%
# Execute the following:
# 1 load the json files from the papers folder
# 2 use langchaing to create text splitters for each abstract in the json files
# 3 use the embeddings_api.py file to create embeddings for each abstract
# 4 store the embeddings and all the other dictionary data in a new json file in a new folder called embeddings

import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import json

# load dataframe from json file
import pandas as pd
import re

# df = pd.read_json("papers_dataframe_full_processed.json", lines=True)
df = pd.read_csv("papers_dataframe_full_processed_with_processed_embeddings.csv")
# %%


BASE = "http://localhost:54288"

def embed(text, task="default"):
    r = requests.post(f"{BASE}/embed", json={
        "text": text,
        "task": task,
        "normalize": True,
        "chunk_strategy": "mean_over_chunks",
        "max_length": None
    })
    r.raise_for_status()
    return r.json()

def embed_batch(texts, task="default"):
    r = requests.post(f"{BASE}/embed_batch", json={
        "texts": texts, "task": task, "normalize": True
    })
    r.raise_for_status()
    return r.json()

def similarity(embedding_docs, embedding_query, metric="cosine"):
    r = requests.post(f"{BASE}/compute_similarity", json={
        "embedding_docs": embedding_docs,
        "embedding_query": embedding_query,
        "metric": metric
    })
    r.raise_for_status()
    return r.json()

# %%
# send to embedding api
len(embed(df['abstract'][0])['embedding'])

# %%
print("Starting to create embeddings...")
# for all the abstracts in the dataframe, create embeddings and store them in a new column 'bert_embedding'
df['bert_content_embedding'] = df['content'].apply(lambda x: embed(x)['embedding'])
print("Finished creating embeddings.")
# Save the dataframe to a new json file
df.to_json("papers_dataframe_full_processed_with_embeddings.json", lines=True, orient="records")
# save to csv
df.to_csv("papers_dataframe_full_processed_with_embeddings.csv")
print("Saved to papers_dataframe_full_processed_with_embeddings.json and .csv")

print("Starting to create embeddings for processed content...")
df['bert_processed_content_embedding'] = df['processed_content'].apply(lambda x: embed(x)['embedding'])
print("Finished creating embeddings for processed content.")
# save to csv
df.to_csv("papers_dataframe_full_processed_with_processed_embeddings.csv")
# Save the dataframe to a new json file
print("Saved to papers_dataframe_full_processed_with_processed_embeddings.json and .csv")
