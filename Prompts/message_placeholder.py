from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()
# Chat Template
chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}") #query from today
])

chat_history = []
# Load Chat History
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

# Create Prompt
prompt = chat_template.invoke({"chat_history": chat_history, "query": "Where is my refund?"})
result = model.invoke(prompt)
print(result.content)