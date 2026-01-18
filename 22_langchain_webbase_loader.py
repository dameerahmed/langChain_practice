
from langchain_community.document_loaders import WebBaseLoader
url="https://docs.python.org/release/3.14.2/"

loader = WebBaseLoader(url)
    
data = loader.load()
print(type(data))
print(len(data))
print(data[0].page_content)