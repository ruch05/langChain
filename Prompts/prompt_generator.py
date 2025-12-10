from langchain_core.prompts import PromptTemplate

template = PromptTemplate(template='''Summarize the research paper titled "{paper_input}" with the following specifications:
    Explanation Style: "{style_input}"
    Explanation Length: "{length_input}"
    If certain info is not available, respond with "Insufficient Info available instead of guessing."
    1. Mathematical details:
         - Include relevant Mathematical equations if present in paper
         - Explain the mathematical concepts using simple, intuitive code snippets''',
         input_variables=["paper_input", "style_input", "length_input"])

template.save("template.json")