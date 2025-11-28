import os
os.environ['HF_HOME'] = '/home/chris/Data/Projects/HF_CACHE'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from fastapi import FastAPI
from pydantic import BaseModel



model_dir = "dunzhang/stella_en_1.5B_v5"
vector_dim = 1024

model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear.load_state_dict({k.replace("linear.", ""): v for k, v in torch.load(os.path.join("/home/chris/Data/Projects/HF_CACHE/modules/transformers_modules/dunzhang/stella_en_1.5B_v5", f"2_Dense_{vector_dim}/pytorch_model.bin")).items()})
vector_linear.cuda()

def get_embedding(text):
    with torch.no_grad():
        input_data = tokenizer(text, padding="longest", truncation=True, return_tensors="pt")
        # if token is longer than 512, split use the first 512 tokens
        num_tokens = input_data["input_ids"].shape[1]
        # if num_tokens > 512:
        #     input_data = tokenizer(text, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        attention_mask = input_data["attention_mask"]
        last_hidden_state = model(**input_data)[0]
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        vector = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        vector = normalize(vector_linear(vector).cpu().numpy())
    return vector.tolist()[0], num_tokens

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(input: TextInput):
    embedding, num_tokens = get_embedding(input.text)
    return {"embedding": embedding, "num_tokens": num_tokens}



@app.post("/embed_queries_s2s")
async def embed_text(input: TextInput):
    # Prompt of s2s task(e.g. semantic textual similarity task):
    query_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
    query =  query_prompt  + input.text
    embedding, num_tokens = get_embedding(query)
    return {"embedding": embedding, "num_tokens": num_tokens}

@app.post("/embed_queries_s2p")
async def embed_text(input: TextInput):
    # Prompt of s2p task(e.g. retrieve task):
    query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
    query =  query_prompt  + input.text
    embedding, num_tokens = get_embedding(query)
    return {"embedding": embedding, "num_tokens": num_tokens}



@app.post("/compute_similarity")
async def compute_similarity(input: dict):
    embedding_docs = torch.tensor(input["embedding_docs"]).cuda()
    embedding_query = torch.tensor(input["embedding_query"]).cuda()
    
    similarity = embedding_query @ embedding_docs.T
    return {"similarity": similarity}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="131.155.34.228", port=8000)