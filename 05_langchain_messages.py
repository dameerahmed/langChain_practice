from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

promt = [
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    promt.append(HumanMessage(content=user_input))
    response = model.invoke(promt)
    print("AI:", response.content)
    promt.append(AIMessage(content=response.content))
