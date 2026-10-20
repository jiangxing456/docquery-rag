# SecQuery：网络安全知识库问答系统

SecQuery 用检索增强生成（RAG）回答安全文档中的问题。上传 CVE 资料、安全公告或合规文档 PDF 后，可以直接提问；回答由检索到的文档片段和 DeepSeek 生成。

仓库分两部分：`app.py` 是可直接运行的问答应用，其余 `*.py` 是离线评测与实验脚本，用来验证和对比不同的检索方案。

## 功能

- 使用本地 BGE-m3 模型生成中英文文本向量，存入 ChromaDB。
- 按文件内容哈希跳过重复上传；同名文件更新时替换旧索引。
- 检索距离超过阈值时拒答，避免对无关问题编造答案。
- 对相似问题复用语义缓存，减少重复调用模型；优先使用 Redis，未连接时使用内存缓存。
- 在页面侧边栏输入 DeepSeek API Key，无须写入代码。

## 工作流程

```text
PDF → 文本解析 → 分块（800 字符，重叠 120）→ BGE-m3 向量化 → ChromaDB
问题 → 语义缓存 → 检索与拒答判断 → 相关片段 → DeepSeek → 回答
```

界面使用 Streamlit，检索链使用 LangChain。向量库保存在本地 `rag_db/`；生成回答时，检索到的文档内容会发送给 DeepSeek API。

## 评测与实验脚本

这些脚本是离线实验，用来验证「混合检索、父子分块、Ragas 评测、缓存命中、拒答阈值」等能力，不参与 `app.py` 的线上流程。按用途分四组：

**1. 评测数据集（golden set）**

- `eval_questions.py` —— 主评测集，基于已入库的安全论文整理的问答对。
- `eval_questions_decoupling.py` —— Decoupling(ICLR 2020) 论文的评测集。

**2. 索引构建**

- `index_decoupling.py` —— 把 Decoupling 论文切分入库到独立 collection。
- `small2big_index.py` —— 父子分块：把 800 字「父块」再切成 250 字「子块」入库。

**3. 检索 + 生成 + Ragas 评测（两步式）**

第一步检索 Top-5 上下文并生成答案，产出 `eval_results*.json`：

- `eval_retrieve.py` / `eval_retrieve_decoupling.py` —— baseline 纯向量检索。
- `hybrid_eval.py` —— 混合检索（BM25 + 向量，RRF 融合两路排名）。
- `small2big_eval.py` —— 子块精确检索 → 找回父块全文做上下文生成答案。

第二步读上一步的 JSON，用 Ragas 跑四个指标（faithfulness / context_recall / answer_relevancy / context_precision）：

- `eval_ragas.py` / `decoupling_ragas.py` / `hybrid_ragas.py` / `small2big_ragas.py`

**4. 缓存与拒答验证**

- `cache_hit_test.py` / `final_cache_test.py` —— 语义缓存命中率与响应时间降幅。
- `refusal_test.py` —— 验证「该答」和「该拒答」两类问题是否被正确分流。
- `threshold_probe.py` / `threshold_rag_probe.py` —— 不同阈值下的命中率与误命中探测。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后，在侧边栏输入 DeepSeek API Key，上传 PDF 并点击 **Submit & Process**，然后在主界面提问。首次运行需要下载 BGE-m3 模型；不配置 Redis 时会使用进程内缓存。

评测脚本需要先配置 DeepSeek API Key（脚本里 `load_dotenv` 指向的 `.env` 路径是开发者本机路径，clone 后需改成自己的），并确认 `rag_db/` 里已有入库文档，然后按「检索生成 → Ragas 评测」两步运行，例如：

```bash
python eval_retrieve.py       # 第一步：检索 + 生成，产出 eval_results.json
python eval_ragas.py          # 第二步：Ragas 四指标评测
```

## 当前限制

- 仅支持有文字层的 PDF；扫描件需要先做 OCR。
- 拒答阈值 `1.11` 根据小规模样本设定，换文档集合后需要重新验证。
- 回答未附带原文页码或引用；重要结论仍应对照原始文档核查。
- 混合检索、父子分块等进阶方案只在评测脚本中验证，尚未并入 `app.py`。
