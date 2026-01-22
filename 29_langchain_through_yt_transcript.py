from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

videoid = "kUJ0MrAVOSg" 
emb_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite") 

try:
    # 1. Get Transcript
    proxy_api = YouTubeTranscriptApi() 
    transcrip_list = proxy_api.fetch(videoid, languages=['hi', 'en'])
    transcrip = " ".join(chunk.text for chunk in transcrip_list)

    # 2. Create Vector Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.create_documents([transcrip])
    
    vector_store = FAISS.from_documents(chunks, emb_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 3. Define Chain Components
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    template = PromptTemplate(
        template="""
        Answer the question strictly based on the context provided below.
        Answer in Roman Urdu.
        
        <context>
        {context}
        </context>

        Question: {question}
        """,
        input_variables=["context", "question"],
    )
     
    parser=StrOutputParser()

    rag_chain = RunnableParallel({
        "context": retriever|RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }) | template | model | parser
    

    # 4. Run Chain
    response = rag_chain.invoke("important kea tha isma ")
    
 
   
    
    
    print("--- Answer ---")
    print(response)

except TranscriptsDisabled:
    print("Captions disabled for this video.")
except Exception as e:
    print(f"Error: {e}")