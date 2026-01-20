from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """# This is a sample Markdown document
# Header 1
## Header 2
This is a paragraph under header 2.
- This is a list item 1
- This is a list item 2
```python
def sample_function():
    print("This is a code block")
"""
# ----------------------
text_splitter = RecursiveCharacterTextSplitter.from_language(   
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=10,
)

# ----------------------
chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")