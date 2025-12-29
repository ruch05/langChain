from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


url = "https://www.amazon.de/-/en/Compatible-Anti-Yellowing-Certified-Protection-Shockproof/dp/B0CC1F5CGH/?_encoding=UTF8&pd_rd_w=hm2dq&content-id=amzn1.sym.9c415e03-1e84-4399-843f-ae94c02b4607&pf_rd_p=9c415e03-1e84-4399-843f-ae94c02b4607&pf_rd_r=PMASZ4B8AW6Z5BS76T0H&pd_rd_wg=pIDG8&pd_rd_r=7a9f73d2-1e14-4359-9495-5cad998c6a09&ref_=pd_hp_d_atf_dealz_cs&th=1"
loader = WebBaseLoader(url)




load_dotenv()
model = ChatOpenAI()

prompt = PromptTemplate(
    template=" Answer the following question {question} from the following {text}",
    input_variables=["question", "text"]
)
parser = StrOutputParser()

docs = loader.load()

chain = prompt | model | parser 

print(chain.invoke({"question": "What is the product here?", "text": docs[0].page_content}))
