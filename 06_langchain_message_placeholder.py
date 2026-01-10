from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}"),
    ]
)
history = []
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    history.append(HumanMessage(content=user_input))
    prompt = chat_template.invoke({"history": history, "user_input": user_input})
    response = model.invoke(prompt)
    history.append(AIMessage(content=response.content))
    print("AI:", response.content)
