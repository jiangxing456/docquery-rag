# DocQuery

## Overview
DocQuery is a document insight retrieval system. Upload your own PDFs, ask questions in natural language, and get answers grounded in what those documents actually say. It works with Chinese, English, and mixed-language documents.

## Features
- **Natural Language Querying**: Ask complex questions about your documents and get concise, accurate answers.
- **Custom Database**: Upload your own PDFs in the sidebar to build the knowledge base.
- **Similarity Search**: Retrieves the most relevant passages for your query using multilingual embeddings.
- **Grounded Answers**: The model is instructed to answer only from retrieved context and not to invent information not present there.
- **Streamlit Interface**: User-friendly interface for queries and document uploads.

## Technologies Used
- **Programming Language**: [Python 3.10+](https://www.python.org/downloads/release/python-31011/)
- **Framework**: [LangChain](https://www.langchain.com/)
- **Database**: [ChromaDB](https://www.trychroma.com/)
- **Models**:
  - Embeddings: [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — multilingual, runs locally via `sentence-transformers`
  - Chat: [DeepSeek API (deepseek-chat)](https://api.deepseek.com/), called through the OpenAI-compatible interface
- **PDF Processing**: [PyPDFLoader](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/)
- **Document Splitter**: [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 800 characters per chunk with 120 characters of overlap
- **Retrieval**: similarity search, top 5 chunks
- **UI**: [Streamlit](https://streamlit.io/)

## Requirements
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Use the Application**:
   - Get an API key from [platform.deepseek.com](https://platform.deepseek.com/) and paste it into the sidebar, then click **Enter**.
   - Optionally, upload PDFs in the sidebar and click **Submit & Process** to add them to the knowledge base.
   - Enter your query in the main interface and click **Submit**.

## Notes
- The embedding model (`BAAI/bge-m3`, ~2.2 GB) is downloaded on first run and cached locally afterwards. Loading it takes roughly 30-60 seconds on CPU; this happens once per server start.
- If `huggingface.co` is unreachable from your network, `app.py` defaults `HF_ENDPOINT` to the `hf-mirror.com` mirror. Set your own `HF_ENDPOINT` environment variable to override it.
- Vector data is stored in `./rag_db` (collection `mixed_database`). Delete that directory to rebuild the knowledge base from scratch. Note that uploading a document twice will index it twice.
- Switching embedding models requires clearing `./rag_db`: the vector dimension is fixed when a collection is first written.

## :mailbox: Connect With Me
<img align="right" src="https://media.giphy.com/media/2HtWpp60NQ9CU/giphy.gif" alt="handshake gif" width="150">

<p align="left">
  <a href="https://linkedin.com/in/codewithcharan" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="codewithcharan" height="30" width="40" style="margin-right: 10px" /></a>
  <a href="https://instagram.com/joyboy._.ig" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="__mr.__.unique" height="30" width="40" /></a>
  <a href="https://twitter.com/Joyboy_x_" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="codewithcharan" height="30" width="40" style="margin-right: 10px" /></a>
</p>
