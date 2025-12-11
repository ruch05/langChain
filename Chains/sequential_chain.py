from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a report on topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the report in 5 points {text}",
    input_variables=["text"]
)

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({"topic":"cricket"})
print(result)

chain.get_graph().print_ascii()