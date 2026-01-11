from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGroq(
    model="qwen/qwen3-32b",
)

schemas = [
    ResponseSchema(name="writer_name", description="The name of the writer"),
    ResponseSchema(name="story_title", description="The title of the story"),
    ResponseSchema(name="story", description="The short story"),
    ResponseSchema(name="rating", description="The rating of the story"),
]
parser = StructuredOutputParser.from_response_schemas(schemas)

template = PromptTemplate(
    template="Tell me a short story about THIS: {topic}.\n  {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
prompt = template.invoke({"topic": "a brave knight"})

response = model.invoke(prompt)
output = parser.parse(response.content)
print(output)
