# search_pubmed.py

# search pubmed based on a query. Retireve the results and save to a file
# all the results are saved in separate json files and containes all available information
# there will be a total of 1000 results per query, each stored in a separate file


import requests
import xml.etree.ElementTree as ET
import os
import json

def search_pubmed(query):

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 1000,
        "retmode": "xml",
        "rettype": "full",
        "api_key": "2647321636ba8285c55ac201024d945fae08"
    }

    response = requests.get(base_url, params=params)

    root = ET.fromstring(response.text)
    return root



def save_to_file(root, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(ET.tostring(root, encoding="utf-8").decode("utf-8"))


def save_to_json(root, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(root, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    query = "in vivo"
    root = search_pubmed(query)
    # save_to_file(root, "pubmed_results.xml")
    save_to_json(root, "pubmed_results.json")
    print(f"Results saved to pubmed_results.xml and pubmed_results.json")


