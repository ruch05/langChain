from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = ["Deepika Padukone is a very beautiful actress and charges the maximum in Indian movie industry",
"Priyanka Chopra has now entered Hollywood movies after her marriage",
"Radhika Apte has extremely good acting skills and her movies were top watched on Netflix",
"Katrina Kaif, a British born woman, is a leading actress and is not an unfamiliar name in Indian household",
"Anushka Sharma, once at the top of her career, is not doing many movies these days"]

query = "Tell me about Anushka Sharma"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0] #both values must be 2D list
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print("User asked : ", query)
print(documents[index])
print("Similarity score is : ", score )