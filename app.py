import os
import hashlib
import streamlit as st

# 国内直连 huggingface.co 会被重置，走镜像下载嵌入模型。
# 已在别处设过 HF_ENDPOINT 的话以你的为准；网络能直连时可删掉这行。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from semantic_cache import lookup, store  # 语义缓存：相似问题复用答案

# 拒答阈值：检索到的最小距离超过它，就判定「知识库里没有」，直接拒答不硬编。
# 1.11 是实测定的——7 个「论文里能查到答案」的问题距离落在 0.85~1.11，
# 7 个「与论文无关」的问题落在 1.13~1.31，两者之间有安全间隔。
# ⚠️ 注意 Chroma 返回的是「距离」不是相似度（越小越相似），别搞反。
REPLY_DISTANCE_THRESHOLD = 1.11


# 缓存嵌入模型：Streamlit 每次交互都会重新运行脚本，加缓存后模型只加载一次
@st.cache_resource
def get_embedding_model():
    """加载嵌入模型（把文本转成向量），并缓存复用。"""
    return HuggingFaceEmbeddings(
        # bge-m3：多语言模型，中英文都能检索，适合中英混杂的文档。
        # 它训练时不带 query prompt，所以不需要给 query 加前缀。
        model_name="BAAI/bge-m3",
        # 向量归一化，配合余弦相似度使用
        encode_kwargs={"normalize_embeddings": True}
    )


# 缓存向量数据库：只创建/打开一次，避免每次重跑都重新连接
@st.cache_resource
def get_db():
    """创建或重新打开 Chroma 向量数据库，并缓存复用。"""
    return Chroma(
        collection_name="mixed_database",          # 集合名（相当于数据库里的一张表）
        embedding_function=get_embedding_model(),  # 指定用哪个模型做向量化
        persist_directory="./rag_db",              # 向量数据持久化到本地磁盘的目录
    )


def format_docs(docs):
    """把检索回来的一批文档片段拼成一个字符串，供提示词里的 {context} 使用。"""
    return "\n\n".join(doc.page_content for doc in docs)


def add_to_db(uploaded_files):
    """处理用户上传的 PDF 文件，切分后写入向量数据库。"""
    if not uploaded_files:
        st.error("No files uploaded!")
        return

    db = get_db()  # 拿到（带缓存的）数据库实例
    # 递归字符切分器：按 800 字符一块切分，相邻块之间重叠 120 字符，避免上下文被切断
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    for uploaded_file in uploaded_files:
        # 按文件内容算哈希：同一个文件传两次，内容哈希相同，可据此判断是否已入库
        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if db.get(where={"file_hash": file_hash}, limit=1)["ids"]:
            st.info(f"{uploaded_file.name} 已在库中，跳过")  # 有记录就跳过，不重复入库
            continue

        # 增量更新：同名文件内容变更（哈希不同）时，先删掉旧版本的 chunk，避免过期数据堆积
        old_ids = db.get(where={"file_name": uploaded_file.name})["ids"]
        if old_ids:
            db.delete(ids=old_ids)
            st.info(f"{uploaded_file.name} 内容已更新，删除旧版本 {len(old_ids)} 个片段")

        # 先把上传的文件落到本地临时目录，因为 PyPDFLoader 需要一个文件路径
        temp_file_path = os.path.join("./temp", uploaded_file.name)
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(temp_file_path)  # PDF 加载器
        data = loader.load()                  # 读取 PDF，得到一页一条的 Document 列表
        chunks = text_splitter.split_documents(data)  # 按上面的规则切分成小块
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash  # 哈希写进元数据，供下次去重查询
            chunk.metadata["file_name"] = uploaded_file.name  # 文件名，供增量更新定位旧版本
        db.add_documents(chunks)              # 向量化并写入 Chroma（含元数据）
        os.remove(temp_file_path)             # 处理完删掉临时文件


def run_rag_chain(query):
    """执行 RAG 链：先检索相关文档，再交给 DeepSeek 生成回答。"""
    # 从 session_state 里取用户填的 API Key（由侧边栏保存）
    api_key = st.session_state.get("deepseek_api_key")
    if not api_key:
        return "Please enter your DeepSeek API key in the sidebar first."

    # 语义缓存：相似问法直接复用答案，省掉下面的检索 + 生成
    cached = lookup(query, get_embedding_model())
    if cached:
        st.caption("⚡ 命中语义缓存（相似问题直接返回）")
        return cached

    db = get_db()

    # ===== 拒答兜底：检索分数全低于阈值就直接拒答，不硬编 =====
    # Chroma 的 similarity_search_with_score 返回的是「距离」（越小越相似）。
    # 阈值 1.11 是实测定的：7 个「论文里能查到答案」的问题距离落在 0.85~1.11，
    # 7 个「与论文无关」的问题落在 1.13~1.31，两者之间有安全间隔，所以 1.11 能区分。
    hits = db.similarity_search_with_score(query, k=3)
    best_dist = min(dist for _, dist in hits) if hits else float("inf")
    if best_dist > REPLY_DISTANCE_THRESHOLD:
        st.caption(f"🛑 检索分数低于阈值（{best_dist:.3f} > {REPLY_DISTANCE_THRESHOLD}），已拒答")
        return "抱歉，知识库中没有检索到相关内容，无法回答这个问题。"

    # 分数够，直接用检索到的片段走生成链
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 提示词模板：{context} 和 {question} 是占位符，链执行时会被自动填充。
    # 模板正文保持英文（这段文字会原样发给模型，注释不会），逐句含义如下：
    #   You are a cybersecurity knowledge assistant ...
    #       → 你是一名网络安全知识库助手（绑定安全领域，基于上传的安全文档作答）
    #   Answer the question based only on the following context:
    #       → 只能依据下面提供的上下文来回答问题（限定知识来源，防止模型自由发挥）
    #   {context}  → 占位符，会被检索到的 5 个文档片段（format_docs 拼接后）替换
    #   Answer the question based on the above context:
    #       → 基于上面的上下文回答问题
    #   {question} → 占位符，会被用户输入的问题替换
    #   - Quote vulnerability identifiers ... exactly ...
    #       → 漏洞编号（CVE-2024-xxxx）、攻击术语（XSS/RCE/0day）必须与上下文原文一致，不得改写或臆造
    #   - If the context does not contain the answer ...
    #       → 上下文没有答案时明确说"未找到"，不要猜（安全问答答错风险高）
    #   Don't give information not mentioned in the CONTEXT INFORMATION.
    #       → 不要提供上下文里没有提到的信息（抑制模型编造，即抑制幻觉）
    #   Do not say "according to the context" or "mentioned in the context" or similar.
    #       → 不要出现"根据上下文""上下文中提到"之类的话术（让回答像直接作答，而不是念资料）
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a cybersecurity knowledge assistant that answers questions strictly from the uploaded security documents (CVE records, security advisories, compliance standards, attack/defense writeups).

        Answer the question based only on the following context:
        {context}

        Answer the question based on the above context:
        {question}

        Rules:
        - Quote vulnerability identifiers (e.g., CVE-2024-12345) and technical terms (e.g., XSS, RCE, 0day) exactly as they appear in the context; never paraphrase or invent them.
        - If the context does not contain the answer, say explicitly that the answer is not found in the documents; do not guess.
        - Don't give information not mentioned in the context.
        - Do not say "according to the context" or "mentioned in the context" or similar.
        """
    )

    # 通过 OpenAI 兼容接口调用 DeepSeek 的对话模型
    chat_model = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",  # DeepSeek 的 OpenAI 兼容地址
        temperature=0.3,                      # 温度越低回答越稳定、越少发挥
    )

    # 用 LCEL 语法组装 RAG 链：
    #   1) 同时准备 context（检索器检索 + 拼接）和 question（原样透传）
    #   2) 填入提示词模板
    #   3) 交给大模型生成
    #   4) 把输出解析成纯字符串
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )
    answer = rag_chain.invoke(query)  # 执行整条链并返回回答
    store(query, answer, get_embedding_model())  # 答完写入语义缓存，供相似问法复用
    return answer


def main():
    """初始化并管理 SecQuery 安全知识库问答应用。"""
    st.set_page_config(page_title="SecQuery 安全知识库", page_icon=":shield:")
    st.header("网络安全知识库问答系统")

    # 主区域：问题输入框
    query = st.text_area(
        ":bulb: 输入关于安全文档的问题（如 CVE 漏洞、安全公告、等保规范）：",
        placeholder="e.g., CVE-2024-12345 的漏洞描述和危害等级是什么？",
    )

    # 点击 Submit 后，非空校验通过则执行 RAG 链
    if st.button("Submit"):
        if not query:
            st.warning("Please ask a question")
        else:
            with st.spinner("Thinking..."):  # 转圈提示，避免用户以为卡住了
                result = run_rag_chain(query=query)
                st.write(result)

    # 侧边栏第一部分：填写并保存 DeepSeek API Key
    with st.sidebar:
        st.title("API Keys")
        deepseek_api_key = st.text_input("Enter your DeepSeek API key:", type="password")

        if st.button("Enter"):
            if deepseek_api_key:
                # 存到 session_state，这样同一次会话里的其他函数都能读到
                st.session_state.deepseek_api_key = deepseek_api_key
                st.success("API key saved!")
            else:
                st.warning("Please enter your DeepSeek API key to proceed.")

    # 侧边栏第二部分：上传安全文档（CVE 漏洞库 / 安全公告 / 等保规范 PDF），提交后写入向量库
    with st.sidebar:
        st.markdown("---")  # 分隔线
        pdf_docs = st.file_uploader(
            "上传安全文档（PDF，如 CVE 漏洞库 / 安全公告）:memo:",
            type=["pdf"],
            accept_multiple_files=True,  # 允许一次选择多个 PDF
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload the file")
            else:
                with st.spinner("Processing your documents..."):
                    add_to_db(pdf_docs)
                    st.success(":file_folder: Documents successfully added to the database!")


if __name__ == "__main__":
    main()
