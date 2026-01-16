from langchain_community.text_loaders import TextLoader


data=TextLoader("langChain_practice\data.txt").load()
print(data)