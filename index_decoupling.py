# -*- coding: utf-8 -*-
"""重建索引：把 Decoupling(ICLR 2020) 论文切分入库到独立 collection。"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF = r"C:\Users\j\Desktop\Decoupling Representation and Classifier for Long-Tailed Recognition.pdf"

loader = PyPDFLoader(PDF)
docs = loader.load()
print(f"PDF 页数: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
print(f"切分 chunk 数: {len(chunks)}")

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(collection_name="decoupling", embedding_function=emb, persist_directory="./rag_db")
db.add_documents(chunks)
print("已入库到 collection: decoupling")
