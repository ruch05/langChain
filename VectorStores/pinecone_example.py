from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from langchain.schema import Document
from pinecone import ServerlessSpec


load_dotenv()

pc = Pinecone()

index_name = "langchain-test-index"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


doc1 = Document(
    page_content="Cow is a mammal",
    metadata={"type": "animal"}
)

doc2 = Document(
    page_content="Parrot is a bird",
    metadata={"type": "bird"}
)

doc3 = Document(
    page_content="Dogs are very faithful",
    metadata={"type": "animal"}
)

docs = [doc1, doc2, doc3]


vector_store = PineconeVectorStore(
    embedding=OpenAIEmbeddings(), index=index
)

#print(vector_store.add_documents(docs))
print(vector_store.similarity_search("which of them is a bird?", k=1))

print(vector_store.similarity_search_with_score("", filter={"type": "animal"}))
