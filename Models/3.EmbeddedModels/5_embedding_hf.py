#Not in the course - Using HuggingFace APIs to generate Embeddings instead of downloading the model locally

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query("Delhi")

print(str(result))
#This should ideally be same as the output of our local file : 3_embedding_hf_local.py but it is not (weird!!)