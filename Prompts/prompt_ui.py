from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st

load_dotenv()
model = ChatOpenAI()

st.header("Research Tool")

paper_input = st.selectbox("Select paper", ["Select...", "Attention is all you need","BERT"])
style_input = st.selectbox("Select Explanation style", ["beginner-friendly", "Technical", "Mathematical"])
length_input = st.selectbox("Select Explanation length", ["Short (1-2 paragraph)", "Medium(3-5 paragraph)", "Long(Detailed Explanation)"])


template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({"paper_input": paper_input, "style_input":style_input, "length_input":length_input})
#    prompt = template.invoke({"paper_input": paper_input, "style_input":style_input, "length_input":length_input})
#    result = model.invoke(prompt)
    st.write(result.content)

    