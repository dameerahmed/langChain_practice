from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0
)

# 2. Tools Setup
@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    print("Adding tool called")
    return a + b

@tool
def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    print("Subtracting tool called")
    return a - b

tools = [add, subtract]

system_instruction = "You are a helpful assistant"


agent = create_agent(
    model=model, 
    tools=tools, 
    system_prompt=system_instruction
)


response = agent.invoke({"messages": [("user", "Add 1 and 2")]})


print(response["messages"][-1].content)
