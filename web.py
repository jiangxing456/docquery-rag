import os
import hashlib
import streamlit as st
from langchain_community.embeddings import  HuggingFaceBgeEmbeddings#将文字转换为向量
from langchain_chroma import Chroma                                 #本地向量库，把向量存起来，提问时可调用
from langchain_openai import ChatOpenAI                             #聊天模型
from langchain_text_splitters import RecursiveCharacterTextSplitter #把长文切成小段
from langchain_community.document_loaders import PyPDFLoader        #pdf解析 
from langchain_core.prompts import ChatPromptTemplate               #提示词模板
from langchain_core.output_parsers import StrOutputParser           #把模型输出成文字串
from langchain_core.runnables import  RunnablePassthrough           #原样往下传，问题不需要加工，直接进提示词
#网络不能直连时用
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
#缓存嵌入模型，Streamlit每次交互都需要重新运行脚本，加缓存后只执行一次
@st.cache_resource#装饰器:适合缓存“连不上、不能随便复制”的大对象，比如模型、数据库连接被包之后，第一次调用会真正加载模型；之后再调用，直接拿缓存结果
def get_embedding_model():
    """加载嵌入模型，把文本转为向量，并缓存使用"""
    return HuggingFaceBgeEmbeddings(
         #bge-m3：多语言模型，中英文都能检索，适合中英混杂的文档。
        # 它训练时不带 query prompt，所以不需要给 query 加前缀。
        model_name="BAAI/bge-m3",
        #向量归一化，配合余弦相似度使用
        encode_kwargs={"normalize_embeddings":True}
    )

#缓存向量数据库：只创建/打开一次，避免每次重新连接
@st.cache_resource
def get_db():
    """创建或重新打开chroma 向量数据库，并缓存使用"""
    return Chroma(
        collection_name="mixed_database",#集合名(相当于数据库的一张表)
        embedding_function=get_embedding_model(),#指定用哪个模型做向量化
        persist_directory="./chroma"#把向量数据持久化到本地磁盘

    )
def format_docs(docs):
    """把检索回来的一段文档拼接成一个字符串，供提示词的{context}使用"""
    return "\n\n".join(doc.page_content for doc in docs)


def add_to_db(uploaded_files):
    """处理文件，切分后写入向量数据库"""
    if not uploaded_files:
        st.error("没有文件上传")
        return

    db=get_db()#拿到带缓存的数据库实例
    #递归字符串，800切分，120重叠联通语义 chunk  切断封存便于查找
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )
#这就是“浏览器内存 → 磁盘路径 → 读 PDF → 切文档”的完整落地过程。
    for uploaded_file in uploaded_files:            #uploaded_file.name:是浏览器传来的原始文件名如: 报告.pdf
        #按文件内容算哈希:同一个文件传两次哈希相同，可据此判断是否已入库
        file_hash=hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        if db.get(where={"file_hash":file_hash},limit=1)["ids"]:#查库里有没有这个哈希
            st.info(f"{uploaded_file.name} 已在库中，跳过")
            continue#跳过这个文件，不重复入库

        #先把上传的文件落到本地目录，因为pypdfloader需要文件路径
        temp_file_path=os.path.join("./temp",uploaded_file.name)#浏览器内存里的文件」变成一个「磁盘上的路径字符串」
        os.makedirs(os.path.dirname(temp_file_path),exist_ok=True)#剥掉文件名，得到目录部分
                                   #exist_ok=True   确保temp不存在时也不报错
        with open(temp_file_path,"wb") as temp_file:#wb二进制格式
            temp_file.write(uploaded_file.getbuffer())#getbuffer 把浏览器内存里的文件变成一个字节流，写入磁盘

        loader=PyPDFLoader(temp_file_path)#pdf加载器
        data=loader.load()#读取pdf，得到一页一页的document列表
        chunks=text_splitter.split_documents(data)#按上面的规则切成小块
        for chunk in chunks:
            chunk.metadata["file_hash"]=file_hash#哈希写进元数据，供下次去重查询
        db.add_documents(chunks)  #向量化，并写入chroma 含元数据
        os.remove(temp_file_path)#处理完删除

def run_rag_chain(query):
    """执行RAG链：先检索相关文档，再交由 DEEPseek回答"""
    #从session_state 里取用户填的APIkey
    api_key=st.session_state.get("deepseek_api_key")
    if not api_key:
        return "请输入你的密钥"


    db=get_db()#拿到带缓存的数据库实例
    #把数据库包装成检索器：相似度检索，每次取相关的5个片段
    retriever=db.as_retriever(search_type="similarity",search_kwargs={"k":5})

    prompt_template=ChatPromptTemplate.from_template(
        """
    你是一个AI助手，回答问题只能基于提供的上下文回答
    {context}
    基于上面的上下文回答问题
    {question}
    用提供的上下文准确、简洁地回答用户的问题，不必给出论证或解释理由（避免输出冗长的推理过程）
    不要提供上下文里没有提到的信息（抑制模型编造，即抑制幻觉），不要出现"根据上下文""上下文中提到"之类的话术（让回答像直接作答，而不是念资料）


    """
    )
    #调用Deepseek
    chat_model=ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.5
    )
    #LCEL语法组装rag链
    #1.准备context和question拼接
    #2.填入提示词模板
    #3.交给大模型生成
    #4.输出解析为字符串
    #在 LangChain 里，| 表示“流水线连接”：上一步输出，作为下一步输入。
    rag_chain=(
        {"context":retriever | format_docs,"question":RunnablePassthrough}
        | prompt_template
        | chat_model#把prompt传给deepseek，得到模型回复
        | StrOutputParser()#抽出文本纯字符串
    )
    return rag_chain.invoke(query)#传入问题，获取答案
def main():
    """初始化，管理DocQuery 应用界面"""
    st.set_page_config(page_title="DocQuert",page_icon="☀")
    st.header("问题回答")
    #主区域：问题输入框
    query=st.text_area(
        ":💡:输入你的关于文档的问题",
        placeholder="这是你的主要问题吗"
    )
    if not query:
        st.warning("请输入一个问题")
    else:
        with st.spinner("思考着..."): # 转圈提示，避免用户以为卡住了
            result=run_rag_chain(query=query)
            st.write(result)

with st.sidebar:
    st.write("API KEYS")
    deepseek_api_key=st.text_input("输入你的密钥",type="password")

    if st.button("enter"):
        if deepseek_api_key:
            st.session_state.deepseek_api_key=deepseek_api_key
            st.success("成功")
        else:
            st.warning("请重新输入")

with st.sidebar:
    st.markdown("------")
    pdf_docs=st.file_uploader(
        "上传你的文件",
        type="pdf",
        accept_multiple_files=True#允许一次传多个文档
    )
    if st.button("提交"):
        if not pdf_docs:
            st.warning("请上传文档")
        else:
            with st.spinner("正在读取文档..."):
                add_to_db(pdf_docs)
                st.success("文档已被加载到数据库中")
if __name__=="__main__":
    main()
