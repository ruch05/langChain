from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from langchain.schema import Document
load_dotenv()


doc1 = Document(
    page_content="Virat Kohli is in RCB team",
    metadata={"team": "RCB"}
)

doc2 = Document(
    page_content="Rohit Sharma is in Mumbai Indians team",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni is in CSK team",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is in Mumbai Indians team",
    metadata={"team": "Mumbai Indians"}
)

doc5 = Document(
    page_content="Ravindra Jadeja is also in CSK team",
    metadata={"team": "Chennai Super Kings"}
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",
    collection_name="sample"
)

vector_store.add_documents(docs)

print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

print(vector_store.similarity_search(
    query="Who among these are in CSK team?",
    k=2
))

print(vector_store.similarity_search_with_score(
    query="Who among these are in CSK team?",
    k=2
))

print(vector_store.similarity_search_with_score(
    query="",
    filter={"team": "RCB"}
))

updated_doc1 = Document(
    page_content="Virat Kohli is a member of VK team",
    metadata = {"team": "VK"}
)

vector_store.update_document(document_id="2ed1c4f0-f2f9-4f80-a8b0-84cdccfbd175", document=updated_doc1)

print(vector_store.get(include=["embeddings", "metadatas", "documents"]))


vector_store.delete(ids=["1bee23e9-397e-45e8-b80e-22d314363166"])
print(vector_store.get(include=["embeddings", "metadatas", "documents"]))
