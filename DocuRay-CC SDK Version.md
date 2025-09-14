# DocuRay（macOS）MVP - Claude Code SDK 版 PRD

## 1. 背景与目标
- 背景：在 DocuRay 2.0 的“Agent 编排 + 核心算法本地化 + MCP 工具”架构下，交付一版基于 Claude Code SDK 的 macOS MVP，支持“自然语言 → 本地文件精准定位”，并尽量提供文件内信息源的高亮预览。
- 总目标：
  - 自然语言输入，定位到文件（文件名/路径/语义匹配）
  - 返回可点击结果（打开/定位/预览）
  - 提供片段级信息源（snippet）与高亮（文本类文件先支持）
  - 离线可用（无网/无 API Key 也能搜索），有网时可用 SDK/LLM 提升编排与召回质量

## 2. 用户故事（关键用例）
- 知识工作者：我输入“第三季度营收同比增长数据在哪份报告”，系统弹出若干文档，预览卡片中高亮“营收同比增长…”。我点击后在 Finder 或应用中直接打开该文件。
- 开发者：我输入“handleError 定义在哪个文件”，系统返回若干代码文件，支持用 VSCode 打开并跳到对应行（`--goto file:line`）。
- 法务：我输入“客户A的补充条款”，系统命中文档并高亮具体条款段落，支持复制路径与快速预览。

## 3. MVP 范围
- 输入通道：CLI（必选），菜单栏 UI（可选，若时间允许给出最小实现）。
- 文件类型（优先级从左到右）：`txt / md / code（.py/.ts/.js/.java等） / pdf（Beta） / docx（转文本后）`。
- 查询类型：
  - 文件名/路径精确查找（Spotlight + 自建索引）
  - 关键词/短语匹配（FTS5 倒排）
  - 语义检索（本地 ONNX 嵌入 + 向量索引）
- 高亮范围：
  - 文本/Markdown/代码：支持片段级 snippet 与命中词高亮（内置预览面板）
  - PDF：先返回页码与片段文本（高亮为 Phase-2 定制预览或 Quick Look 扩展）
- 离线模式：无网/无 API Key 情况下，照常索引与检索；Agent 编排退化为确定性路径（不依赖 LLM 推理）。

## 4. 成功指标（MVP）
- P95 延迟：关键词/文件名 < 400ms；语义检索 < 800ms（Top-50 融合排序后返回）。
- 首次全量索引吞吐：≥ 30 文件/秒（SSD，文本/MD/PDF混合）。
- 增量更新延迟：文件变更后 < 2s 反映在结果中。
- 覆盖率：可解析文件占比 ≥ 95%（MVP 支持类型）。

## 5. 系统架构（Claude Code SDK 版）

### 5.1 分层与组件
- Agent 编排层（Claude Code SDK）
  - 职责：意图理解（可调用 LLM）、动态选择 MCP 工具、权限与会话管理、结果解释与交互。
  - 模式：在线启用 LLM；离线降级为确定性路由（不阻塞核心检索）。
  - 权限：`permissionMode=strict`，`allowedTools` 仅白名单 MCP 与只读 Shell。

- 平台与检索能力（MCP 服务器，本地）
  - `spotlight-mcp`：封装 `mdfind/mdls`，用于文件名/路径/标签快速命中。
  - `fsevents-mcp`：基于 FSEvents 的文件变更监听，驱动增量索引。
  - `indexer-mcp`：全量/增量扫描，抽取文本（PDFKit/textutil），切片、建 FTS、建向量。
  - `search-mcp`：查询入口；关键词（FTS5）、向量召回、融合（RRF/加权）与轻量排序。
  - `preview-mcp`：生成预览卡片（文本高亮）；PDF 先返回页码与片段（Phase-2 支持图像高亮）。
  - `open-mcp`：打开/定位操作（`open -R` Finder 定位、`open` 打开、VSCode `--goto`）。

- 核心算法（本地库/嵌入在 search-mcp）
  - `QueryRouter`：确定性规则 + 轻量模型（可选 ONNX）
  - `EarlyStoppingEngine`：多因素阈值控制搜索迭代与耗时预算
  - `AdaptiveFusionEngine`：融合 Spotlight/FTS/向量召回
  - `RankingEngine`：特征加权（MVP 规则型），预留 L2R 接口

### 5.2 数据与索引
- SQLite（`meta.db`）：`files`、`chunks`、`fts5 inverted_index`、`jobs`
- 向量索引：`SQLite-vec`（优先）或 `FAISS`
- 嵌入模型：本地 ONNX（多语种，小型，INT8 量化），按需加载 / 内存映射

### 5.3 SDK 集成要点
- `.claude/` 目录：
  - `agents/`：主 Agent + 子 Agent（search/preview/diagnose）
  - `settings.json`：权限、工具白名单、Hook（如索引完成提示）
  - `CLAUDE.md`：项目长期记忆与边界指令
- 工具权限：
  - `allowedTools`: `["mcp:spotlight","mcp:fsevents","mcp:indexer","mcp:search","mcp:preview","mcp:open","shell:read"]`
  - `disallowedTools`: 默认拒绝网络、写敏感路径
- 模式控制：
  - 有 `ANTHROPIC_API_KEY` 与网络：启用 LLM 辅助路由/解释
  - 无：走本地确定性路由（完全离线）

## 6. 关键流程

### 6.1 查询端到端
1) 输入自然语言（CLI 或菜单栏）
2) Agent 路由：
   - 精确/路径/文件名 → `spotlight-mcp` + FTS
   - 语义/场景类 → 向量召回 + FTS 回补
3) `search-mcp`：
   - 关键词（FTS5）命中
   - 向量召回（Top-K）
   - 融合（RRF/权重）→ 候选集
   - 排序（规则加权）→ Top-N
4) `preview-mcp`：构建 snippet（命中词高亮、周边上下文），PDF 返回页码 + 文本片段
5) Agent 输出：结果列表（标题、路径、置信度、snippet，高亮）+ 操作（打开/定位/复制路径）
6) 选择项：
   - `open-mcp reveal` → Finder 定位（`open -R path`）
   - `open-mcp open` → 对应应用打开
   - `open-mcp editor` → VSCode `--goto file:line`

### 6.2 索引端到端
1) 首次运行：授权与目录选择（默认 `~/Documents`, `~/Desktop`）
2) `indexer-mcp full`：遍历 → 文本抽取（PDFKit/textutil）→ 切片 → FTS5 → 嵌入 → 向量索引
3) `fsevents-mcp`：监听文件变更 → 触发 `indexer-mcp update`
4) 索引状态/进度通过 Hook 或 CLI 显示

## 7. 高亮与打开策略
- 文本/MD/代码：
  - snippet：基于倒排/向量召回的命中片段，返回字符区间与命中词
  - 预览：内置高亮（CLI 输出 ANSI 高亮；菜单栏/面板用富文本渲染）
  - 打开：VSCode `--goto file:line` 或 TextEdit 打开文件
- PDF（MVP）：
  - 返回页码 + 提取的文本片段（PDFKit 的页面映射）
  - 预览：先文本高亮；Phase-2 扩展 Quick Look 或内置 PDF 预览实现图像高亮

## 8. 权限与安全（macOS / TCC）
- 首启引导用户授权 Full Disk Access 或目录级授权
- 默认排除：`.git`, `node_modules`, `*.log`, 大于阈值的二进制/媒体文件
- 所有索引/向量与缓存存储于 `~/Library/Application Support/DocuRay/`

## 9. CLI 规范（MVP）
- `dr search "查询" [--top 5] [--open] [--reveal] [--editor code]`
- `dr index --full | --update`
- `dr status`
- 输出：Top-N 结果（路径、置信度、snippet 高亮），交互选择操作

## 10. 性能目标与基准（MVP）
- 查询延迟（P95）：关键词 < 400ms；语义 < 800ms
- 索引吞吐：≥ 30 文件/秒；增量更新 < 2s 生效
- 资源：常驻内存 < 500MB（含嵌入模型），磁盘占用可配置

## 11. 里程碑与验收
### M1（Week 1）：平台 MCP 与索引最小链路
- 完成：`spotlight-mcp / fsevents-mcp / indexer-mcp` 最小实现
- SQLite + FTS5 落地，文本/MD/代码抽取与切片
- CLI：`dr index --full` 与 `dr search` 关键词可用
- 验收：关键词检索 P95 < 400ms；能在 Finder 定位/VSCode 打开

### M2（Week 2）：语义检索与融合
- 嵌入模型（ONNX/量化）与向量索引（SQLite-vec/FAISS）集成
- `search-mcp` 融合与排序打通；`preview-mcp` 文本高亮
- 验收：语义检索 P95 < 800ms；Top-1 相关性主观评估 ≥ 80%

### M3（Week 3）：可用性与稳定性
- 增量索引稳定、错误恢复、权限引导完善
- 菜单栏 UI（可选最小实现）与流式响应（SDK Streaming）
- 验收：覆盖率 ≥ 95%；三类用户故事可完成

## 12. 风险与回退
- 联网依赖（LLM）：离线模式下走确定性路由与本地检索。
- PDF 抽取质量：PDFKit → pdftotext → OCR 的降级链；记错误标记与重试。
- 向量模型体积/速度：选小型多语种模型 + INT8 量化；批处理与内存映射优化。
- Spotlight 滞后：自建索引为主，Spotlight 作为辅助；必要时触发 `mdimport` 刷新。

## 13. 监控与可观测
- 指标：查询延迟分布、命中率、索引吞吐、增量延迟、内存/磁盘占用
- 日志：索引任务、解析错误、权限/TCC 失败、MCP 调用耗时
- 开发期：简单 CLI 报表与日志文件；后续可接入 SDK Hook 上报

## 14. 交互与文案（示例）
- 首启：提示授权与选择目录；展示预计索引时间
- 搜索：流式输出 Top-N，提供快捷操作键（Open/Reveal/Editor/Copy）
- 结果：路径、文件名、置信度、snippet（高亮），回车打开，空格预览

## 15. 验收清单（MVP Done）
- 自然语言检索直达：三种用户故事可完成
- CLI 可用：索引/搜索/状态齐备；中文输出与高亮
- 打开与定位：Finder 定位、应用打开、VSCode 跳行
- 文本类高亮：snippet 命中清晰；PDF 提供页码与片段文本
- 离线可用：在无网/无 Key 情况下仍可检索与打开

