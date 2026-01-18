from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


data = TextLoader("data.txt").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=0,
)
texts = text_splitter.split_documents(data)
print(type(texts))
print(len(texts))
print(texts[0].page_content)