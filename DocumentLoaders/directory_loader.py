from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="DocumentLoaders/Documents",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

lazy_loader_docs = loader.lazy_load()

for i in docs:
    print(i.metadata)

print("break")

for i in lazy_loader_docs:
    print(i.metadata)