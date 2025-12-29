from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DocumentLoaders/Documents/ Ruchika Nagpal_RedHat_English.pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)