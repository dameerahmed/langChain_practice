from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more."""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=79,
    chunk_overlap=0,
)

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")   