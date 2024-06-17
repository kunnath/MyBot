import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import fitz  # PyMuPDF
import os
import warnings
import json

# Constants
BOOK_DIR = './literature_data'
HF_TOKEN = 
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"
EMBEDDINGS_CACHE = './'
INDEX_DIR = "./content/books/faiss_index"
INDEX_META_FILE = "./content/books/index_meta.json"

# Suppress future warnings from the HuggingFace Hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Set Hugging Face API token environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Load the Hugging Face model and embeddings
llm = HuggingFaceEndpoint(repo_id=HF_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=EMBEDDINGS_CACHE)

# PDF Loader class
class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        with fitz.open(self.file_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                documents.append(Document(page_content=text, metadata={"page_num": page_num}))
        return documents

# Load PDF files from a directory
def load_pdf_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = PDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# Streamlit app
st.title("Intralogistics research engine V1.0")

# Load the documents
documents = load_pdf_files(BOOK_DIR)

# Get the list of filenames in the BOOK_DIR
current_files = set(os.listdir(BOOK_DIR))

# Check if the index meta file exists
if os.path.exists(INDEX_META_FILE):
    with open(INDEX_META_FILE, 'r') as f:
        indexed_files = set(json.load(f))
else:
    indexed_files = set()

# Check if the current files are already indexed
if current_files != indexed_files:
    # Initialize and apply the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(documents)

    # Perform vector embeddings and create a new FAISS index
    vector_db = FAISS.from_documents(split_docs, embeddings)
    vector_db.save_local(INDEX_DIR)

    # Save the current file names to the index meta file
    with open(INDEX_META_FILE, 'w') as f:
        json.dump(list(current_files), f)
else:
    # Load the FAISS index with allow_dangerous_deserialization set to True
    vector_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Create the QA chain prompt template
input_template = """Answer the question based only on the following context. Keep your answers short and succinct.

Context to answer question:
{context}

Question to be answered: {question}
Response:"""

prompt = PromptTemplate(template=input_template, input_variables=["context", "question"])

# Set up conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Create the QA chain with memory
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "memory": memory},
)

# Streamlit user input
query = st.text_input("Enter your question:")
if query:
    try:
        # Invoke the chain with the query
        answer = qa_chain({"question": query})

        # Extract the response text
        response_text = answer['result']
        st.write("Answer:", response_text)

        # Display source documents
        if 'source_documents' in answer:
            for doc in answer['source_documents']:
                st.write(f"Source (page {doc.metadata['page_num']}): {doc.page_content[:500]}...")
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
