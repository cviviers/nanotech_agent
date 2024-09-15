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



def get_embedding_from_api(text, url="http://localhost:8000/embed"):
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        return response.json()["embedding"], response.json()["num_tokens"]
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def create_text_splits(data, skip=False):
    if skip:
        return data
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    splits = splitter.split_text(data)
    if len(splits) > 0:
        print(f"Split {len(splits)} chunks")
    return splits[0]
   


def main():
    folder_path = 'processed_papers'
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    files_with_errors = []

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    for file in json_files:

        if os.path.exists(os.path.join('embeddings', file)):
            print(f"Skipping {file}")
            continue
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)

            text_splits = create_text_splits(data['cleaned_text'], skip=True)
            try:
                embeddings, num_tokens = get_embedding_from_api(text_splits)
            except:
                print(f"Error with {file}")
                files_with_errors.append(file)
                continue
            print(f"Number of tokens: {num_tokens}")
            data['embedding'] = embeddings
            data['num_tokens'] = num_tokens

            with open(os.path.join('embeddings', file), 'w') as f:
                json.dump(data, f, indent=4)
    print("The following files had errors:")
    print(files_with_errors)
if __name__ == "__main__":
    main()
