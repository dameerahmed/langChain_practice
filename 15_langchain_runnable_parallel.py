from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

groq_model = ChatGroq(
    model="qwen/qwen3-32b",
)
parser=StrOutputParser()
template=PromptTemplate(
    template="funny joke about this : {topic}",
    input_variables=["topic"],
)


parallel_chain = RunnableParallel({
    "groq_joke": RunnableSequence(template, groq_model, parser),
    "google_joke": RunnableSequence(template, google_model, parser),
})
response = parallel_chain.invoke({"topic": "langchain"})    
print("Groq Joke:")
print(response["groq_joke"])
print("\nGoogle Generative AI Joke:")
print(response["google_joke"])

parallel_chain.get_graph().print_ascii()