# -*- coding: utf-8 -*-
"""small2big 父子分块：把现有 89 个 800 字「父块」再切成 250 字「子块」入库。

子块用于精确检索，父块提供完整上下文。检索命中子块后，通过 metadata 里的
parent_id 找回父块，把父块全文喂给模型——既精确又保留上下文。
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

# 现有父块（800 字 chunk，baseline 用的就是它）
src = Chroma(collection_name="mixed_database", embedding_function=emb, persist_directory="./rag_db")
parent_data = src.get(include=["documents", "metadatas"])
parent_ids = parent_data["ids"]
parent_docs = parent_data["documents"]
print(f"父块数: {len(parent_docs)}")

# 子块切分器：250 字一块，重叠 50
sub_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)

# 目标 collection 存子块
dst = Chroma(collection_name="small2big", embedding_function=emb, persist_directory="./rag_db")

sub_docs, sub_metas = [], []
for pid, pdoc in zip(parent_ids, parent_docs):
    for s in sub_splitter.split_text(pdoc):
        sub_docs.append(s)
        sub_metas.append({"parent_id": pid})

# 分批入库（避免一次太多）
BATCH = 50
for i in range(0, len(sub_docs), BATCH):
    dst.add_texts(sub_docs[i:i+BATCH], metadatas=sub_metas[i:i+BATCH])
print(f"子块入库完成: {len(sub_docs)} 个")
