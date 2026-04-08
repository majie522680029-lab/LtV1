
# 知识沉淀系统（Streamlit）

## 1. 环境准备

### Python 版本
建议 Python 3.10

```bash
conda create -n qa_system python=3.10 -y
conda activate qa_system
```

### 安装依赖
```bash
pip install streamlit pandas numpy openpyxl scikit-learn sentence-transformers openai tqdm
```

## 2. 数据准备

### 输入文件
- 默认读取 Excel 的 **D 列**（对话记录）
- 示例路径：`data_test/2506_在线咨询-替换后.xlsx`
- 你也可以用 `data_raw/` 里的原始文件

### 想读取更多“新问题”的做法
在页面左侧「📊 数据源配置」中：
1. **Excel 文件路径**：改成你新的 Excel 文件路径
2. **读取行数限制**：
   - 设为 **0** 表示读取全部行
   - 设为 **100 / 500 / 1000** 表示读取更多数据
3. 点击 **开始处理** 重新生成结果  
   新问题会在「📚 知识库管理 → 🆕 新问题」中出现

## 3. 启动与操作

### 启动
```bash
streamlit run qa_system.py
```

### 本地 QA 抽取模型（Qwen3-8B / vLLM）
如果你希望“对话转 QA”和“代表性 QA 筛选”走本地模型，可先启动一个 OpenAI 兼容的 vLLM 服务：

```bash
vllm serve /home/common_data/llm/Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 43722 \
  --served-model-name qa-extractor-qwen3 \
  --api-key sk-local \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

启动后，在页面左侧「📝 对话转 QA 模型」中设置：
- 模型来源：`本地模型`
- 本地 QA API 地址：`http://127.0.0.1:43722/v1`
- 本地 QA API Key：`sk-local`
- 本地 QA 模型名：`qa-extractor-qwen3`

当前代码默认值也已经切到这组配置，通常只需要选择 `本地模型` 即可。

### 具体操作流程（推荐）
1. 左侧配置数据与参数  
   - Excel 文件路径  
   - 读取行数限制（0=全量）  
   - API Key / 模型  
   - BGE 模型路径  
2. 点击左侧最底部 **“开始处理”**  
   - 系统会生成 `data_test_result_YYYYMMDD_HHMMSS/` 目录  
3. 在「📊 结果展示」查看处理结果  
4. 进入「📚 知识库管理」  
   - 直接使用本次结果进行相似度分析与入库  
   - 如需使用历史结果，再点击 **“加载结果”**  

### 操作流程（推荐）
1. 左侧设置：
   - API Key / 模型
   - Excel 文件路径
   - 读取行数限制
   - BGE 模型路径
2. 点击 **开始处理**
3. 在「📊 结果展示」中查看：
   - 处理统计
   - 聚类详情
   - 代表性问答
4. 在「📚 知识库管理」中：
   - 查看新问题 / 需要更新 / 已有问题
   - 人工确认后添加或更新

## 4. 结果输出
每次运行会生成一个目录：
```
data_test_result_YYYYMMDD_HHMMSS/
```
其中包含：
- `raw_qa_pairs.json`：原始 QA
- `cluster_results.json`：聚类结果
- `representative_qa_pairs.json`：代表性 QA
- `processing_stats.json`：统计信息
- `dialog_extractions/`：每条对话的抽取结果

## 5. 常见问题

### Q: 想看更多问题/更少问题？
修改「读取行数限制」即可。

### Q: 新问题在哪里？
处理完成后 → 「📚 知识库管理 → 🆕 新问题」

### Q: 报错 “Excel 文件不存在” 怎么办？
检查以下几点：
1. **路径写错**：默认应为  
   `/home/majie/work/LtV1/data_test/2506_在线咨询-替换后.xlsx`  
   而不是 `/home/majie/LtV1/...`
2. **文件名不一致**：确认文件名完全一致（含中文与后缀）
3. **文件不在当前机器**：如果 Excel 在别的路径/机器，需拷贝后再填路径
4. **权限问题**：确保当前用户可读该文件

### Q: Streamlit 启动了但网页打不开怎么办？
按以下顺序排查：
1. **先看服务是否存活**  
   `curl http://127.0.0.1:8501/_stcore/health`  
   返回 `ok` 说明服务正常。
2. **Remote-SSH 场景必须端口转发**  
   在 VS Code `Ports` 面板转发当前 Streamlit 端口（默认 8501），本机访问 `http://localhost:8501`。
3. **不要优先用 External URL**  
   公网 URL 常被防火墙拦截，优先用本地转发地址。
4. **端口冲突时换端口**  
   `streamlit run qa_system.py --server.port 8510`，然后转发 `8510`。
