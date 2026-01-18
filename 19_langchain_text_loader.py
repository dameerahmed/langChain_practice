from langchain_community.document_loaders import TextLoader


data=TextLoader("data.txt").load()
print(type(data))
print(data[0].metadata)