from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os

load_dotenv()

groq_model = ChatGroq(
    model="qwen/qwen3-32b",
)
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


class reviewDict(TypedDict):
    title: str
    summary: str
    language: str
    tone: str
    words: int
    rating: int
    status: Literal["Good", "Average", "Poor"]


structured_model = google_model.with_structured_output(reviewDict)
parser = StrOutputParser()
template1 = PromptTemplate(
    template="Write the 100 words story  about this topic  {topic} :\n",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Based on the story, give me a short summary and review of the story:\n {story}\n",
    input_variables=["story"],
)

chain = template1 | groq_model | parser | template2 | structured_model

response = chain.invoke({"topic": "artificial intelligence"})

print(response["title"])
print(response["summary"])
print(response["language"])
print(response["tone"])
print(response["words"])
print(response["rating"])
print(response["status"])


chain.get_graph().print_ascii()
