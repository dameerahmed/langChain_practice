from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="qwen/qwen3-32b",
)


class Output(TypedDict):
    quote: str
    author: str


structure_model = model.with_structured_output(Output)

promt = PromptTemplate(
    template="give me a short inspirational quote about {topic}",
    input_variables=["topic"],
)


chain = promt | structure_model

response = chain.invoke({"topic": "artificial intelligence"})
print(response["quote"])
print(response["author"])
