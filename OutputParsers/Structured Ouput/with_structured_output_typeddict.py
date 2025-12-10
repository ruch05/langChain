from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()
model = ChatOpenAI()

#schema

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["Pos", "Neg"], "Return sentiment of the review - either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]

structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""As someone who works in machine learning but mostly on CV problems, this book was a perfect bridge into the world of language models. It doesn't assume you're a total beginner, but it also doesn't dump you in the deep end with dense theory and academic papers. The authors do a great job of grounding concepts in clear explanations and walk-throughs you can actually run.
If you're a data scientist, ML engineer, or just a curious dev looking to go beyond ChatGPT and understand how to work with LLMs at a system level — grab this book. You'll get a lot out of it. """)
print(result)
#print(result["summary"])
#print(result["sentiment"])