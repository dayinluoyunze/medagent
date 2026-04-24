# MedAgent

一个面向医药问答场景的智能 Agent Demo，支持多模型切换、知识库检索、长期记忆摘要，以及在模型或向量检索不可用时的本地回退。

## 项目定位

这个项目聚焦于一个相对高风险、对答案可信度要求较高的垂直场景：医药知识问答。目标不是做通用聊天，而是验证以下能力：

- 多个 OpenAI 兼容模型接口的统一接入
- 基于本地知识库的 RAG 检索增强
- 会话历史与摘要记忆管理
- 故障回退与基本安全约束

当前实现也适合作为后续扩展评测、引用溯源和安全策略的基础版本。

## 核心功能

- 多模型 Provider 切换：支持 `OpenAI`、`ModelScope`、`MiniMax`
- 多源知识库接入：支持 `md`、`txt`、`json`、`csv`、`docx`、`pdf`、图片、`html`、网页 URL
- 网页资料导入：在 UI 中粘贴 URL 后抓取正文或 PDF 文本，生成本地 Markdown 知识快照
- OCR 兜底：扫描版 PDF 和图片资料可通过 Tesseract OCR 抽取文本后入库
- 检索模式：支持 `vector / hybrid / keyword`，只有 MiniMax API 也能运行
- RAG 检索：优先使用 `FAISS + Embeddings`
- 回退检索：Embedding 不可用或查询期失败时自动退回本地关键词检索
- 来源展示：回答尾部自动附带命中的知识来源和片段
- 安全分级：对紧急风险、诊断判断、个体化用药调整做规则级防护
- 长期记忆：对历史对话进行 markdown 摘要并持久化
- 故障兜底：模型调用失败时回退到本地知识库内容摘要
- Web UI：基于 Streamlit 提供交互式问答界面

## 技术栈

- Python
- Streamlit
- LangChain
- OpenAI SDK
- FAISS
- python-docx
- pypdf
- pypdfium2
- pytesseract
- Pillow
- BeautifulSoup4

## 项目形态

严格说，这个项目的用户入口是 Chat，但工程形态已经不是纯 Chatbot，而是一个轻量级垂直 Agent。

- Chatbot 部分：用户通过聊天框提问，系统返回自然语言答案。
- Agent 部分：`MedicalAgent` 负责模型选择、RAG 检索、长期记忆、安全分级、故障回退、来源引用和指标日志。
- 边界说明：当前还不是复杂 Multi-Agent，也没有自主规划多步工具链；更准确的简历表述是“面向医药问答的 RAG Agent / AI Assistant”。

## RAG 技术选型

- 知识导入：采用本地文件 + UI 上传 + URL 快照导入。URL 不在每次索引时实时抓取，而是在用户显式添加时转成 Markdown 快照，保证索引可复现，也避免运行时网络波动影响问答。
- 文档解析：内置 `md/txt/json/jsonl/csv/docx/pdf/image/html` 解析。HTML 与 URL 页面会先做正文提取，PDF 会抽取页文本并保留页码标记；扫描版 PDF 或图片会走 OCR 兜底。
- 文本切分：使用 `RecursiveCharacterTextSplitter`，按标题、段落、换行、中文标点逐级切分，`chunk_size=500`、`chunk_overlap=50`，适合中文医药说明书和 FAQ 这类短段落知识。
- 向量检索：使用 `FAISS + OpenAI-compatible Embeddings`，Provider 可切换到 ModelScope，适合本地 Demo 和简历项目，部署成本低，不依赖独立向量数据库。
- 混合检索：支持 `vector / hybrid / keyword`。向量检索负责语义召回，`jieba + 正则 token` 的关键词检索负责药名、剂量、禁忌词等精确命中，并在 Embedding 不可用时兜底。
- 本地重排：`hybrid` 模式会扩大候选集，再用向量排名、关键词命中和精确匹配分数做 lightweight rerank，避免单纯拼接结果导致高精确命中的片段排在后面。
- 索引缓存：基于知识文件内容、Embedding 模型、检索配置生成 manifest；知识库变化后自动重建 FAISS，否则复用本地缓存。

## 系统架构

```text
User
  |
  v
Streamlit UI
  |
  |-- Upload / URL Snapshot
  |
  v
MedicalAgent
  |------------------------------|
  |                              |
  v                              v
LLM Provider Adapter         Memory Manager
  |                              |
  v                              v
OpenAI / ModelScope /       Markdown History +
MiniMax                     Summary Persistence
  |
  v
Retriever
  |------------------------------|
  |                              |
  v                              v
FAISS Vector Search         Keyword Fallback
  |
  v
Local Knowledge Base
```

## 目录结构

```text
medagent/
├─ app.py
├─ config.py
├─ requirements.txt
├─ agents/
│  └─ medical_agent.py
├─ rag/
│  └─ retriever.py
├─ knowledge/
│  ├─ drugs.md
│  ├─ products.md
│  ├─ qa.md
│  ├─ sample.docx
│  ├─ sample.url
│  └─ sample.urls
└─ memory/
   ├─ conversation_history.md
   └─ conversation_summary.md
```

## 本地启动

### 1. 创建环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并至少配置一个可用 Provider 的 API Key。

示例：

```env
LLM_PROVIDER=modelscope
EMBEDDING_PROVIDER=modelscope
MODELSCOPE_API_KEY=your-modelscope-key
ALLOW_URL_KNOWLEDGE_INGESTION=true
```

如果你目前只有 MiniMax API，推荐这样配：

```env
LLM_PROVIDER=minimax
EMBEDDING_PROVIDER=none
RETRIEVAL_MODE=keyword
MINIMAX_API_KEY=your-minimax-key
```

### 3. 启动应用

```bash
streamlit run app.py
```

默认会在浏览器中打开本地页面。

如果你在 Windows 下希望一键启动，也可以直接运行项目根目录下的脚本：

```powershell
.\start_medagent.ps1
```

脚本会自动：

- 检查 `venv` 和 `.env`
- 自动寻找 `8501-8510` 范围内的可用端口
- 在后台启动 Streamlit
- 将日志写入 `logs/streamlit.out.log` 和 `logs/streamlit.err.log`
- 记录 PID 到 `logs/streamlit.pid`
- 记录实际端口到 `logs/streamlit.port`
- 自动打开浏览器

关闭服务：

```powershell
.\stop_medagent.ps1
```

如果你是直接在当前终端执行 `streamlit run app.py`，也可以用 `Ctrl + C` 结束服务。

## 使用流程

1. 在侧边栏选择模型 Provider
2. 输入对应 API Key，或从 `.env` 自动读取
3. 点击“初始化 Agent”
4. 如需补充知识库，在侧边栏打开“添加资料”，上传文件、粘贴文本或输入 URL 后点击“保存并刷新知识库”
5. 在主界面输入医药相关问题

网页添加的资料会保存到 `knowledge/uploads/`，该目录默认不提交到 Git。
URL 导入会拒绝本机、内网和非 `http(s)` 地址；如需限制可导入域名，可配置 `REMOTE_KNOWLEDGE_ALLOWLIST`。
OCR 依赖本机 Tesseract 程序；如果 Windows 没有安装，可先安装 Tesseract，并在 `.env` 中配置 `TESSERACT_CMD`。

建议尝试以下问题：

- 二甲双胍可以长期服用吗？
- 忘记服药应该怎么处理？
- 某类降压药适合哺乳期使用吗？

## 评测

项目已预留基础评测目录，可先从检索层开始量化：

```bash
python eval/run_eval.py
```

当前脚本会读取 [eval/qa_dataset.jsonl](/e:/agent/medagent/eval/qa_dataset.jsonl) 中的样本，对检索结果做基础统计，包括：

- 检索命中率
- 来源命中率
- 关键词覆盖率

评测说明见 [eval/README.md](/e:/agent/medagent/eval/README.md)。

Answer-level 评测脚本：

```bash
python eval/run_answer_eval.py --provider minimax --api-key your-minimax-key
```

## 测试

项目当前提供基础单元测试，可直接执行：

```bash
pytest
```

覆盖范围包括：

- provider 配置读取
- markdown memory 持久化
- 来源拼接
- 检索回退逻辑

## 日志与指标

每次问答会将基础运行指标写入 `logs/chat_metrics.jsonl`，便于后续做简单分析。

当前落盘字段包括：

- `provider`
- `embedding_provider`
- `question_redacted`
- `knowledge_hit`
- `retrieved_doc_count`
- `source_labels`
- `fallback_used`
- `status`
- `error_type`
- `risk_level`
- `risk_flags`
- `duration_ms`

默认会对日志中的问题文本做脱敏处理，避免手机号、邮箱、身份证号等直接落盘。

## 已知限制

- 本项目仅用于技术演示，不构成医疗建议
- 当前知识库内容较少，答案质量高度依赖样本覆盖
- 不同 Provider 的模型能力和兼容性存在差异
