from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
model = ChatGroq(
    model="qwen/qwen3-32b",
)

parser = JsonOutputParser()
template = PromptTemplate(
    template="give me the name age and address of the person: {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt = template.format()

response = model.invoke(prompt)
output = parser.parse(response.content)
print(output)
