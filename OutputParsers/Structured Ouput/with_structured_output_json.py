from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
load_dotenv()
model = ChatOpenAI()

#schema

json_schema= {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review in a list"
    }},
    "required": ["key_themes"]
}



    #key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    #summary: str = Field(description="A brief summary of the review")
    #sentiment: Literal["Pos", "Neg"] = Field(description="Return sentiment of the review - either negative, positive or neutral")
    #pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    #cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")

structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke("""As someone who works in machine learning but mostly on CV problems, this book was a perfect bridge into the world of language models. It doesn't assume you're a total beginner, but it also doesn't dump you in the deep end with dense theory and academic papers. The authors do a great job of grounding concepts in clear explanations and walk-throughs you can actually run.
If you're a data scientist, ML engineer, or just a curious dev looking to go beyond ChatGPT and understand how to work with LLMs at a system level — grab this book. You'll get a lot out of it. """)
print(result["key_themes"])
#print(result["summary"])
#print(result["sentiment"])