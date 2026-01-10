from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
import os

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


class reviewOutput(TypedDict):
    story: Annotated[str, "The story maximum length is 10 characters."]
    words: Annotated[int, "number of words in the story."]
    rating: Annotated[Literal[1, 3], "The rating is an integer 1 or 2."]


structure_model = model.with_structured_output(reviewOutput)

response = structure_model.invoke(" Tell me a short story about a brave knight.")

print("story:", response["story"])
print("words:", response["words"])
print("rating:", response["rating"])
