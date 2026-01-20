from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """class test :
    def method_one(self):
        pass
    def method_two(self):
        pass    
# This is a sample Python code snippet
def another_function():
    print("Hello, World!")
""" 

# ----------------------
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,    
    chunk_size=100,
    chunk_overlap=10,   
)
# ----------------------

chunks = text_splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")