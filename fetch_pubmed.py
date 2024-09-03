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