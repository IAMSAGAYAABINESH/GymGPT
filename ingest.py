from dotenv import load_dotenv,find_dotenv
from transformers import pipeline
import os



load_dotenv(find_dotenv())


from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

directory_path = "data"  # Path to the directory containing your PDF files
loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create a SentenceTransformer object
embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1-ablated",model_kwargs={"trust_remote_code":True})


db = FAISS.from_documents(texts, embeddings)


db.save_local("gym_vector_db")