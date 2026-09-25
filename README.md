# 📄 DocQuery — 文档问答系统（RAG）

**上传 PDF，用自然语言提问，让大模型"只根据文档内容"回答。**

> ### ⚠️ 来源说明
>
> 本项目基于开源项目 [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
> 中的 `rag_tutorials/rag_chain` 改造而来。
>
> **本仓库记录的是我在原版基础上做的技术栈替换与工程改造** —— 详见下一节。

---

## 🔧 我做了什么（My Contributions）

原版是一套基于 Google 全家桶（Gemini + Google Embedding）的 RAG 应用。我把它**整套替换成本地嵌入 + DeepSeek**，
并补上了缓存、去重等工程细节。

### 1. 嵌入模型替换：Google API → 本地 BGE-m3

| | 原版 | 我的版本 |
|---|---|---|
| 模型 | `GoogleGenerativeAIEmbeddings(models/embedding-001)` | **`BAAI/bge-m3`**（sentence-transformers 本地运行） |

**为什么改**：原方案必须联网调 Google API、按量计费；换成 BGE-m3 后可**完全离线运行、零嵌入成本**，
而且 bge-m3 是**多语言**模型，中英文混排的文档检索效果更好。

### 2. 对话模型替换：Gemini → DeepSeek

用 `ChatOpenAI` 通过 OpenAI 兼容接口调用 `deepseek-chat`，国内网络可直接使用。

### 3. 场景泛化：从"药房专用"到"任意 PDF"

原版把数据库名、持久化目录都写死成 `pharma_database` / `./pharma_db`（面向药房文档），
我改为通用的 `mixed_database` / `./rag_db`，任何领域的 PDF 都能用。

### 4. 切分器替换

`SentenceTransformersTokenTextSplitter` → `RecursiveCharacterTextSplitter`（**800 字/块，重叠 120 字**）。

按字符递归切分更可控；重叠是为了防止关键信息正好落在切分边界上被切断。

### 5. 新增内容哈希去重 ⭐

**原版已知缺陷**（README 里自己写了）：*"uploading a document twice will index it twice"* ——
同一个文档上传两次，会被重复索引两次。

我的做法：用**文件内容的 SHA-256 哈希**做去重。

```python
import hashlib

file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
# 入库前先按哈希查一次，查到就跳过
if db.get(where={"file_hash": file_hash}, limit=1)["ids"]:
    st.info(f"{uploaded_file.name} 已在库中，跳过")
    continue

# 把哈希写进每个文本块的 metadata，供下次查询
for chunk in chunks:
    chunk.metadata["file_hash"] = file_hash
```

**为什么用内容哈希而不是文件名**：文件改了名字，内容没变，一样能识别出来并跳过。

### 6. 新增缓存优化（Streamlit 特性适配）

Streamlit 的机制是**每次用户交互都从头重跑整个脚本**。嵌入模型有 ~2.2GB，
不缓存的话每次点按钮都要重新加载一次，体验直接崩掉。

所以用 `@st.cache_resource` 把**嵌入模型**和**向量库连接**缓存起来，只在第一次真正加载：

```python
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3", ...)

@st.cache_resource
def get_db():
    return Chroma(collection_name="mixed_database", ...)
```

> `cache_resource` 用于不可序列化的大对象（模型、数据库连接）；
> 可序列化的数据应该用 `cache_data`。

### 7. API Key 界面化

原版从环境变量读取 API Key（换 key 得改代码）。我改到**侧边栏输入**并存入 `session_state`，
使用者不需要碰代码：

```python
with st.sidebar:
    deepseek_api_key = st.text_input("Enter your DeepSeek API key:", type="password")
    if st.button("Enter"):
        st.session_state.deepseek_api_key = deepseek_api_key
```

### 8. 国内网络适配

`huggingface.co` 在国内直连会被重置，导致模型下不动。在代码开头设置镜像端点：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
```

---

## 📖 项目简介

DocQuery 是一个文档洞察检索系统：上传自己的 PDF，用自然语言提问，
得到**基于文档真实内容**的回答（而不是模型自由发挥）。

完整 RAG 链路：

```
PDF 上传 → 落盘临时文件 → 解析 → 切分（800/120）
        → 向量化（BGE-m3）→ 存入 ChromaDB
                ↓
提问 → 相似度检索 top-5 → 拼进提示词 → DeepSeek 生成 → 返回答案
```

**抑制幻觉的三层约束**（写在提示词里）：

1. 只根据提供的上下文回答
2. 不要提供上下文里没有提到的信息
3. 不要出现"根据上下文""上下文中提到"之类的话术（让回答像直接作答，而不是念资料）

---

## 🛠️ 技术栈

| 组件 | 技术 |
|---|---|
| 界面 | Streamlit（单文件 `app.py`） |
| RAG 框架 | LangChain（LCEL 表达式） |
| 向量数据库 | ChromaDB（本地持久化到 `./rag_db`） |
| 嵌入模型 | **BAAI/bge-m3**（本地，多语言） |
| 对话模型 | **DeepSeek（deepseek-chat）** |
| PDF 解析 | PyPDFLoader |
| 文本切分 | RecursiveCharacterTextSplitter（800 / 120） |
| 检索 | 相似度检索，top-5 |

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
streamlit run app.py
```

**使用步骤**

1. 去 [platform.deepseek.com](https://platform.deepseek.com/) 申请 API Key
2. 在侧边栏填入 Key 并点击 Enter
3. （可选）在侧边栏上传 PDF，点击 Submit & Process 建立知识库
4. 在主界面输入问题，点击 Submit

---

## ⚠️ 注意事项

- 嵌入模型（BGE-m3，约 2.2GB）**首次运行会自动下载**并缓存到本地，之后不再重复下载；
  首次加载需要约 30-60 秒（CPU）
- 向量数据存在 `./rag_db`（集合名 `mixed_database`），删除该目录即可重建知识库
- 更换嵌入模型需要清空 `./rag_db` —— 向量维度在集合首次写入时就固定了

## 🚧 已知局限（可继续改进的方向）

- **扫描件 PDF 无法解析** —— `PyPDFLoader` 只能提取文字层，图片型 PDF 会提取到空内容，
  需要接入 OCR 才能支持
- **无重排序** —— 目前直接取相似度 top-5，可以加 rerank 模型先粗排再精排以提升准确率
- **回答无引用来源** —— 可以让模型标出答案来自哪份文档、第几页，增强可信度
- **不支持多轮追问** —— 每次提问相互独立，可加入对话历史

---

## 🙏 致谢

- 原项目：[Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps)
  的 `rag_tutorials/rag_chain`
