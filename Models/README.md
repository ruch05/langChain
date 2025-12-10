1. Using LLM Model using LangChain. Model used here is OpenAI
2. Using ChatModels using LangChain.
   Steps:: Import the ChatModel library - instantiate the model from the ChatModel library - model.invoke()
     1. Closed Source Models are used using API. Models used in the project - ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI
     2. Open Source Models are used locally as well as through APIs. Models are used from HuggingFace.
          1. HuggingFace APIs - Libraries - ChatHuggingFace, HuggingFaceEndpoint
          2. HuggingFace Locally - Libraries - ChatHuggingFace, HuggingFacePipeline
4. Using Embedding Models using LangChain
     1. Closed Source:
          1. Using OpenAI Query (Library - OpenAIEmbeddings, method - embedding.embed_query())
          2. Using OpenAI Docs (Library - OpenAIEmbeddings, method - embedding.embed_documents())
     2. Open Source:
          1. Using HuggingFace APIs (Library - HuggingFaceEndpointEmbeddings, method - embed_query())
          2. Using Locally:
               1. Using HuggingFace Query (Library - HuggingFaceEmbeddings, method - embed_query())
               2. Using HuggingFace Docs (Library - HuggingFaceEmbeddings, method - embed_documents())
