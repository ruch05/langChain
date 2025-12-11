from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()
model1 = ChatOpenAI()
model2 = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from following text : {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question-answers from the following text : {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document : {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)


parallel_chain = RunnableParallel({"notes": prompt1 | model1 | parser,
                                   "quiz": prompt2 | model2 | parser})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
text = """Something exciting is rolling out: organizations will soon be able to copy agents built with Microsoft 365 Copilot directly into Copilot Studio.

This creates a seamless path from lightweight, productivity-focused agents—where many teams begin building today—into the full Copilot Studio experience that extends governance with richer tooling, Dataverse capabilities, and advanced customization.

For organizations investing in AI, it’s a unified journey from ideation to deployment—without losing momentum or compromising oversight. By bridging productivity scenarios with enterprise architecture, IT and business teams can now operationalize copilots responsibly at scale.

This update marks a pivotal step in the agentic era, opening new doors for iterative innovation while positioning teams to harness adaptability, scalability, and intelligence for real impact across workflows and decision-making."""
final_result = chain.invoke({"text": text})
print(final_result)

chain.get_graph().print_ascii()