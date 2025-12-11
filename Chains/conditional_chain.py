from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="Sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative : {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback : {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "Positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "Negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
)

final_chain = classifier_chain | branch_chain
result = final_chain.invoke({"feedback": "this is a wonderful product"})
print(result)

final_chain.get_graph().print_ascii()
