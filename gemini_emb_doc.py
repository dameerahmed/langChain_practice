from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimension=32)
document = [
    "Hello, world!",
    "This is a test document for embedding generation.",
]
vector = model.embed_documents(document)
print("Embedding vector:", str(vector))
