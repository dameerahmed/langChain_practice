from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)
reviewOutput = {
    "title": "reviewOutput",
    "type": "object",
    "properties": {
        "write_name": {
            "type": "string",
            "description": "name of the writer",
            "default": "dameer",
        },
        "story": {
            "type": "string",
            "description": "The story maximum length is 10 characters.",
        },
        "words": {"type": "integer", "description": "number of words in the story."},
        "rating": {
            "type": "integer",
            "enum": [1, 3],
            "description": "The rating is an integer 1 or 2.",
        },
    },
}


structure_model = model.with_structured_output(reviewOutput)

response = structure_model.invoke(" Tell me a short story about a brave knight.")


print("writer name:", response["write_name"])
print("story:", response["story"])
print("words:", response["words"])
print("rating:", response["rating"])
