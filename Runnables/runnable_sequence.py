from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about the {topic}",
    input_variables = ["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the joke {joke}",
    input_variables = ["joke"]
)
model = ChatOpenAI()
parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({"topic": "AI"}))

#Problem : Langchain team created Too many chains (ex: llmchain, sequentialchain, simplesequentialchain, 
# retrievalQA, RouterChain, etc.) - code base became too big, and it was difficult to 
# understand which chains to use for what,
#so learning curve became too steep for new AI Engineers. So it became difficult to learn Langchain
#They kinda wanted to make lego blocks so they can be seemlessly integrated and used and this is why they
#made a lot of chains. For LLM component, you need predict() function, whereas for PromptTemplate, you
#use format() function, for retriever - get_relevant_documents(), for parser - parse() method.
# So the components were not standardized - so they couldnt be connected seemlessly.
# And that's why they wrote manual code like LLMchain, retrievalQA chain, etc to connect these components to each other.
# So if they followed same standards for all components, then they didnt need to write these custom chains with
# custom code.
# Solution to this was to make these components again so that they are not standardized and can be 
# seemlessly connected. Which could happen because of runnables.
# runnables are like lego blocks. Each runnable is a unit of work - which takes input, processes it and
# generates output. Each runnable will have a common interface i.e same set of methods.
# Ex - invoke(), batch(), stream()
# All of them can be connected together and output of R1 will automatically be the input of R2.
# And this workflow R1 -- R2 itself is a runnable and then this (R1+R2) Runnable can be connected to another
# workflow (R3+R4+R5).