import pickle
import json
import re

from Bio import Entrez
import pandas as pd
import numpy as np
import os
import json
import time
import regex as re
import xmltodict, json
# ElementTree
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import requests
import re
import string
import nltk
import json


# Download NLTK data files (only need to run once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

###########################################
API_URL =  "http://131.155.34.228:8000" # "http://localhost:8000" #
###########################################

class ScholarlyPublication:
    def __init__(self, container_type=None, source=None, bib=None, filled=None, gsrank=None, pub_url=None, 
                 author_id=None, url_scholarbib=None, url_add_sclib=None, num_citations=None, citedby_url=None, 
                 url_related_articles=None, eprint_url=None):
        self.container_type = container_type
        self.source = source
        self.bib = bib if bib else {}
        self.filled = filled
        self.gsrank = gsrank
        self.pub_url = pub_url
        self.author_id = author_id
        self.url_scholarbib = url_scholarbib
        self.url_add_sclib = url_add_sclib
        self.num_citations = num_citations
        self.citedby_url = citedby_url
        self.url_related_articles = url_related_articles
        self.eprint_url = eprint_url

    def to_dict(self):
        return {
            "container_type": self.container_type,
            "source": self.source,
            "bib": self.bib,
            "filled": self.filled,
            "gsrank": self.gsrank,
            "pub_url": self.pub_url,
            "author_id": self.author_id,
            "url_scholarbib": self.url_scholarbib,
            "url_add_sclib": self.url_add_sclib,
            "num_citations": self.num_citations,
            "citedby_url": self.citedby_url,
            "url_related_articles": self.url_related_articles,
            "eprint_url": self.eprint_url,
        }

    def __repr__(self):
        title = self.bib.get('title', 'No Title')
        venue = self.bib.get('venue', 'No Venue')
        year = self.bib.get('pub_year', 'No Year')
        return f"Publication(title={title}, venue={venue}, year={year})"

    # Save to disk using pickle
    def save_to_pickle(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # Save to disk as JSON
    def save_to_json(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, indent=4)

    # Load from JSON file
    @classmethod
    def load_from_json_file(cls, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            return cls(**data)
    
    # Load from JSON file
    @classmethod
    def load_from_json(cls, data):
        return cls(**data)
    


def sanitize_filename(filename: str) -> str:
    # Define the characters that are not allowed in Windows filenames
    invalid_chars = r'[\/:*?"<>|]'
    
    # Replace invalid characters with an underscore
    sanitized_filename = re.sub(invalid_chars, '_', filename)
    
    # Trim any leading or trailing whitespace
    sanitized_filename = sanitized_filename.strip()
    
    # If the filename ends up empty or consists only of invalid characters, return a default name
    if not sanitized_filename:
        sanitized_filename = "default_filename"
    
    return sanitized_filename



def article_machine(query, batch_size=1000):
    Entrez.email = "c.g.a.viviers@tue.nl"
    
    # Initial search to get total count
    handle = Entrez.esearch(db='pubmed', sort='relevance', rettype='count', term=query)
    total_count = int(Entrez.read(handle)['Count'])
    print(f"Total matching articles: {total_count}")
    
    all_ids = []
    retstart = 0
    
    while retstart < total_count:
        print(f"\nFetching {retstart + 1} to {min(retstart + batch_size, total_count)} out of {total_count}")
        
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                rettype='xml',
                                retstart=retstart,
                                retmax=batch_size,
                                term=query)
        
        results = Entrez.read(handle)
        all_ids.extend(results["IdList"])
        
        retstart += batch_size
        
    print(f"\nRetrieved {len(all_ids)} article IDs")
    return all_ids

def search_pubmed(query, retmax=1000):
    Entrez.email = "c.g.a.viviers@tue.nl"  # Always include your email
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax, retmode="xml")
    record = Entrez.read(handle)
    return record

def fetch_pubmed_data_per_id(pubmed_id):
    Entrez.email = "c.g.a.viviers@tue.nl"  # Always include your email
    handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="xml")
    record = Entrez.read(handle)
    return record

def fetch_pubmed_data_given_ids(id_list):
    Entrez.email = "c.g.a.viviers@tue.nl"
    ids = ','.join(id_list)
    # Use rettype='full' to get complete records including all metadata
    handle = Entrez.efetch(db='pubmed', retmode='xml', rettype='full', id=ids)
    results = Entrez.read(handle)
    return results

# data class for papers

@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: list
    journal: str
    publication_date: str
    doi: str
    keywords: list
    mesh: list
    language_list: list
    embedding: np.array

    def __init__(self, id, title, abstract, authors, journal, publication_year, publication_month, publication_day, doi, keywords, mesh, language_list, embedding= None):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.journal = journal
        self.publication_year = publication_year
        self.publication_month = publication_month
        self.publication_day = publication_day
        self.doi = doi
        self.keywords = keywords
        self.mesh = mesh
        self.language_list = language_list
        self.embedding = embedding

    def __str__(self):
        return f"Paper(id={self.id}, title={self.title}, abstract={self.abstract}, authors={self.authors}, journal={self.journal}, publication_year={self.publication_year}, publication_month={self.publication_month}, publication_day={self.publication_day}, doi={self.doi}, keywords={self.keywords}, mesh={self.mesh}, language_list={self.language_list}, embedding={self.embedding})"
    
    def __repr__(self):
        return f"Paper(id={self.id}, title={self.title}, abstract={self.abstract}, authors={self.authors}, journal={self.journal}, publication_year={self.publication_year}, publication_month={self.publication_month}, publication_day={self.publication_day}, doi={self.doi}, keywords={self.keywords}, mesh={self.mesh}, language_list={self.language_list}, embedding={self.embedding})"
    
    def save_to_json(self, filename):
        # save the paper to a json file in a nice format
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return Paper(**data)

def get_embedding_from_api(text, query_type="document"):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    if query_type == "document":
        url=API_URL+"/embed"
    elif query_type == "s2s":
        url=API_URL+"/embed_queries_s2s"
    elif query_type == "s2p":
        url=API_URL+"/embed_queries_s2p"
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()["embedding"], response.json()["num_tokens"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None


    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()["embedding"], response.json()["num_tokens"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None
    
# Function to preprocess text
def preprocess_text(text, custom_terms=None):
    # Convert to lowercase
    text = text.lower()
    
    if custom_terms is not None:
        # Replace special nanomedicine terms with placeholders to preserve them
        for term in custom_terms:
            # Replace spaces in term with underscores
            term_underscore = term.replace(' ', '_')
            # Use regex to replace the term in text
            text = re.sub(r'\b' + re.escape(term.lower()) + r'\b', term_underscore, text)
    
    # Remove punctuation and **NOT** numbers
    text = re.sub(r'[{}]'.format(string.punctuation), ' ', text)
    # text = re.sub(r'\d+', '', text)
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    if custom_terms is not None:
        # Replace placeholders back to original terms with spaces
        tokens = [word.replace('_', ' ') if word in [t.replace(' ', '_') for t in custom_terms] else word for word in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

# write dataframe to excel
def write_df_to_excel(df, output_dir='output'):
    writer = pd.ExcelWriter
    temp_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"output_{temp_time}.xlsx"
    file_path = os.path.join(output_dir, filename)
    df.to_excel(file_path, index=False)
    print(f"Dataframe saved to {file_path}")