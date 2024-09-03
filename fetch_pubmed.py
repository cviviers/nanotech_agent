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
    handle = Entrez.efetch(db='pubmed', retmode='xml', id=ids)
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
    language_list: list
    embedding: np.array

    def __init__(self, id, title, abstract, authors, journal, publication_year, publication_month, publication_day, doi, keywords, language_list, embedding= None):
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
        self.language_list = language_list
        self.embedding = embedding

    def __str__(self):
        return f"Paper(id={self.id}, title={self.title}, abstract={self.abstract}, authors={self.authors}, journal={self.journal}, publication_year={self.publication_year}, publication_month={self.publication_month}, publication_day={self.publication_day}, doi={self.doi}, keywords={self.keywords}, language_list={self.language_list}, embedding={self.embedding})"
    
    def __repr__(self):
        return f"Paper(id={self.id}, title={self.title}, abstract={self.abstract}, authors={self.authors}, journal={self.journal}, publication_year={self.publication_year}, publication_month={self.publication_month}, publication_day={self.publication_day}, doi={self.doi}, keywords={self.keywords}, language_list={self.language_list}, embedding={self.embedding})"
    
    def save_to_json(self, filename):
        # save the paper to a json file in a nice format
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_json(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            return Paper(**data)
        