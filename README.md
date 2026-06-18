# 健身助手（本地个人版）

基于 LangChain ReAct Agent + RAG 的健身与营养补给助手，**在本机运行**，个人档案与训练数据保存在本地目录，不上传云端。

## 功能

- 健身装备与补剂咨询（RAG 知识库）
- 营养需求计算、补剂相互作用检查
- 个人训练报告（需训练记录或档案体测数据）
- 侧边栏管理个人档案（昵称、体测、目标等）
- 可选导入 10 天示例数据体验完整报告流程

## 环境要求

- Python 3.10+
- 阿里云 [DashScope API Key](https://help.aliyun.com/zh/model-studio/get-api-key)（通义千问）

## 快速开始

### 1. 安装依赖

```bash
cd Agent项目
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
copy .env.example .env
```

编辑 `.env`，填入你的密钥：

```
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

### 3. 启动应用

```bash
streamlit run app.py
```

首次启动时会**自动检查并索引** `data/` 目录下的 txt/pdf 文档（已索引的文件会通过 MD5 跳过，无需重复处理）。

如需手动重建索引，也可运行：

```bash
python rag/vector_store.py
```

或在应用侧边栏点击 **「重新同步知识库」**（例如向 `data/` 添加了新文档后）。

浏览器打开提示的地址（通常为 http://localhost:8501）。

## 本地数据说明

| 路径 | 内容 |
|------|------|
| `data/users/local/profile.json` | 个人档案（昵称、体测、目标等） |
| `data/users/local/records.json` | 训练与补剂历史记录 |
| `data/` | RAG 知识库原文（txt/pdf） |
| `chroma_db/` | 向量索引（运行 vector_store.py 后生成） |

首次启动会自动创建 `data/users/local/` 及默认档案。

### 导入示例数据

侧边栏点击 **「导入 10 天示例数据」**，会从 `data/external/10用户10天训练与补剂数据.csv` 导入 U001 的 10 天记录到本地用户，便于体验训练报告功能。

### 清空与备份

- **清空训练记录**：仅删除 `records.json` 内容，档案保留
- **清空对话**：清除当前聊天，不影响档案
- 手动备份：复制整个 `data/users/local/` 文件夹即可

## 项目结构

```
Agent项目/
├── app.py                 # Streamlit 入口
├── agent/                 # ReAct Agent 与工具
├── rag/                   # 向量检索与 RAG
├── model/                 # 通义千问模型工厂
├── config/                # YAML 配置
├── prompts/               # 系统提示词
├── data/                  # 知识库与演示 CSV
└── utils/                 # 配置、日志、本地用户数据
```

## 常见问题

**Q: 提示未检测到 DASHSCOPE_API_KEY？**  
A: 确认项目根目录存在 `.env` 且密钥正确，重启 Streamlit。

**Q: RAG 回答不准确或为空？**  
A: 确认 `data/` 下有 txt/pdf 文档，点击侧边栏「重新同步知识库」，或查看启动时是否提示同步成功。

**Q: 生成报告提示没有训练记录？**  
A: 在侧边栏导入示例数据，或自行维护 `records.json`（格式见示例导入结果）。

**Q: 数据存在哪里？**  
A: 全部在本机 `data/users/local/`，`.gitignore` 已排除该目录，不会随 Git 上传。

## 免责声明

本助手仅供参考，不构成医疗或专业运动处方。补剂、伤病相关问题请咨询医生或持证教练。
