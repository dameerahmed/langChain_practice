from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

template1 = PromptTemplate(
    template="write a detailed about {topic} ", input_variables=["topic"]
)
template2 = PromptTemplate(
    template="summarize the following text in short: {text} ", input_variables=["text"]
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic": "artificial intelligence"})
print(response)
