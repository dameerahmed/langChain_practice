from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import numpy as np

import os


load_dotenv()

model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimension=300)


document = [
    "my name is dameer malik",
    "i live in lahore pakistan",
    "i am a student",
    "i am from pakistan",
    "i live in nawabshah pakistan",
    "i am 22 years old",
    "i am currently studying computer science",
    "i am a student of computer science",
    "my friend ishrat",
]

doc_vectors = model.embed_documents(document)

que_vector = model.embed_query("i am 22 years old")

similarity_scores = cosine_similarity([que_vector], doc_vectors)
index, score = sorted(list(enumerate(similarity_scores[0])), key=lambda x: x[1])[-1]

print("similarity_scores:", similarity_scores)
print("Most similar document:", document[index])
print("Similarity score:", score)
