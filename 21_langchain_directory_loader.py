from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

data=DirectoryLoader(".",
                     glob="*.pdf",
                     loader_cls=PyPDFLoader
                     )

book=data.lazy_load()

for page in book:
    print(page.page_content) 