from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4",temperature=0, max_completion_tokens=30)
result = model.invoke("Suggest me a linkedin post about AI")
print(result.content)