from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimension=32)
vector = model.embed_query("hello, world!")

print("Embedding vector:", str(vector))
