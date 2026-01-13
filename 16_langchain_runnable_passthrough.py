from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal
from dotenv import load_dotenv
import os

load_dotenv()
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

groq_model = ChatGroq(
    model="qwen/qwen3-32b",
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

story_chain = RunnableSequence(template1, groq_model, parser)
parallel_chain = RunnableParallel({
    "story": RunnablePassthrough(),
    "review": RunnableSequence(template2, structured_model)
    }
)



final_chain=RunnableSequence(
    story_chain,
    parallel_chain
)

response = final_chain.invoke({"topic": "artificial intelligence"})
print("Story:")
print(response["story"])
print("Review:")    
print(response["review"]["title"])
print(response["review"]["summary"])
print(response["review"]["language"])
print(response["review"]["tone"])
print(response["review"]["words"])
print(response["review"]["rating"])
print(response["review"]["status"])



parallel_chain.get_graph().print_ascii()