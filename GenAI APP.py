import streamlit as st
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from gemini_llm import GeminiLLM

GEMINI_API_KEY = "AIzaSyBa3Li3XA22YUgVMO6JP5zuZcCL_xTlKlg"

def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(docs):
    embeddings = SentenceTransformerEmbeddings(model_name = 'all-MiniLM-L6-v2')
    db = Chroma.from_documents(docs, embeddings)
    return db

def rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=GeminiLLM(api_key=GEMINI_API_KEY),  retriever=retriever)
    return qa

def val_response(answer, sources):
    if not sources:
        return "Error: Question is out of Context"
    return answer


def run_chatbot(pdf_file):
    with open('temp.pdf', 'wb') as f:
        f.write(pdf_file.getbuffer())
    docs = load_pdf('temp.pdf')
    vector_store = create_vector_store(docs)
    qa_chain = rag_chain(vector_store)
    st.session_state.qa_chain = qa_chain
    st.success("PDF processes successfully. Ask a Question below")

st.title("PDF ChatBot")

up_pdf = st.file_uploader("Upload your PDF", type=['PDF'])
if up_pdf:
    run_chatbot(up_pdf)

if 'qa_chain' in st.session_state: 
    query = st.text_input("Ask a question from the PDF: ")
    if query:
        result = st.session_state.qa_chain.run(query)
        if "I don't know" in result or result.strip() == "":
            st.error("Error: Question is out of the PDF.")
        else:
            st.success(result)