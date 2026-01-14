from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda
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


template=PromptTemplate(
    template="write the story of {subject} in {style} style.",
    input_variables=["subject", "style"],
)

parser=StrOutputParser()

class StoryRequest(TypedDict):
    subject: str
    style: Literal["fantasy", "sci-fi", "mystery", "romance"]
    
    
story_chain=RunnableSequence(template, groq_model, parser)

def words(text:str)->int:
    return len(text.split())


parallel_chain=RunnableParallel({
    "story": RunnablePassthrough(),
    "word_count": RunnableLambda(words)
    }   )
final_chain=RunnableSequence(
    story_chain,
    parallel_chain
)

response=final_chain.invoke({"subject":"a journey to Mars","style":"sci-fi"})
print("Story:")
print(response["story"])
print("Word Count:")
print(response["word_count"])