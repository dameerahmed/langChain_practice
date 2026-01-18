from langchain_community.document_loaders import PyPDFLoader


data=PyPDFLoader("Digital_Footprint_Report.pdf").load()
print(type(data))
print(len(data))
print(data[0].page_content)