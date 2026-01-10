from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)

response = model.invoke("hy.")
print(response.content)
