from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()
google_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

groq_model = ChatGroq(
    model="qwen/qwen3-32b",
)

parser=StrOutputParser()
class feedback(BaseModel):
    feedback: Literal["good","bad"]

struct_model = groq_model.with_structured_output(feedback)


template1=PromptTemplate(
    template="analyse the user feedback and classify it as good or bad: {user_feedback}",
    input_variables=["user_feedback"],  
)

feedback_chain=RunnableSequence(
    template1,   
    struct_model
)

template_bad = PromptTemplate.from_template("The user gave negative feedback. Write a polite apology email 1 2 line g.{bad_feedback}")
template_good = PromptTemplate.from_template("The user gave positive feedback. Write a thank you note 1 2 lines.  {good_feedback}")

def condition(x)->bool:
    return x.feedback=="bad"

response_chain=RunnableBranch(
    (condition,RunnableSequence(template_bad,groq_model,parser)),
    RunnableSequence(template_good,groq_model,parser)
)

final_chain=RunnableSequence(
    feedback_chain,
    response_chain
)
response=final_chain.invoke({"user_feedback":"wow very impresive"})
print("Response:")  
print(response)
print("Done")