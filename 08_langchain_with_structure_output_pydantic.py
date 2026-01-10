from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Optional
import os

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


class reviewOutput(BaseModel):
    write_name: Optional[str] = Field(
        default="dameer", description="name of the writer."
    )
    story: str = Field(..., description="The story maximum length is 10 characters.")
    words: int = Field(..., description="number of words in the story.")
    rating: Literal[1, 3] = Field(..., description="The rating is an integer 1 or 2.")


structure_model = model.with_structured_output(reviewOutput)

response = structure_model.invoke(" Tell me a short story about a brave knight.")


print("writer name:", response.write_name)
print("story:", response.story)
print("words:", response.words)
print("rating:", response.rating)
