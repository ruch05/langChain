from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a joke about the {topic}",
    input_variables = ["topic"]
)

def word_counter(text):
    return len(text.split())


chain1 = RunnableSequence(prompt, model, parser)

chain2 = RunnableParallel({
    "joke": RunnablePassthrough(),
    "words": RunnableLambda(word_counter)
})

final_chain = RunnableSequence(chain1, chain2)
result = final_chain.invoke({"topic": "AI"})
print(result["joke"])
print(result["words"])