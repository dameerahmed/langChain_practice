from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


data = TextLoader("data.txt").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=300,
    chunk_overlap=100,
)
texts = text_splitter.split_documents(data)
print(type(texts))
print(len(texts))


for text in texts:
    print(text.page_content)
    print("-----")