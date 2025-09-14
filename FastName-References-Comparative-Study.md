# 基于文件名的离线高速搜索：开源实现对比与技术参考

## 范围与结论
- 分析对象（References/）：
  - Ultra-Fast-File-Search-main（Windows，C++）：直接读取 NTFS MFT，支持 GLOB/REGEX 匹配，带后台任务线程。
  - NTFS-Search-master（Windows，C/C++）：低层解析 NTFS 结构（MFT/Attribute/Runlist），以线程加载与遍历实现文件名搜索。
  - filehunter-main（Java）：基于 Lucene 的文件索引/检索（路径/内容），自定义分词器与任务队列。
- 结论：
  - Windows 侧的极致速度主要来自“绕过 OS 逐个枚举”，转而“直接读 MFT + USN Journal”。
  - macOS 无法读取 NTFS MFT，因此应走“高并发遍历 + 轻量索引 + FSEvents 增量 + 极速匹配器”的路线。
  - 匹配层建议采用“名称/路径分词 + 子串/子序列打分”的规则型算法，首屏 Top-N 返回在几十毫秒内。

---

## Ultra Fast File Search（C++，Windows）
- 代码位置：`UltraFastFileSearch-code`
- 核心要点：
  - 直接读 MFT：README/The Manual 强调“像电话簿”一样一次性读取 MFT，避免逐文件 syscall 往返；各盘并行加载与解析。
  - 模式匹配器：`string_matcher.hpp` 提供 `pattern_glob`、`pattern_globstar`、`pattern_regex` 等，大小写可选。
  - 后台任务线程：`BackgroundWorker.hpp` 自实现一个基于 `CRITICAL_SECTION + Semaphore` 的任务队列，线程函数 `process()` 按时间戳顺序执行 `Thunk`。
  - 典型并发：
    - 多磁盘并行：每个卷独立读取 MFT；读取后并行解析与匹配。
    - 任务队列：使用信号量唤醒、带 insert_before_timestamp 的稳定插队策略。
- 代码线索：
  - `BackgroundWorkerImpl::process()`：循环等待信号量→取出任务→执行；`_beginthreadex` 启动线程，确保 C 运行时上下文正确。
  - `string_matcher`：对输入字符串进行 glob/regex/任意匹配，接口 `is_match(str)`。
- 对我们启发（macOS）：
  - 没有 MFT 可用，但可以借鉴“批量一次性获取”和“并行解析”的思路：目录遍历采用多队列并发，减少 syscalls 次数（如批量 stat），并以只读内存快照供查询。
  - 采用独立的后台任务执行器：Swift 可用 `DispatchQueue`/`OperationQueue` 替代自制信号量队列；仍然保留“可插队”和“可清空”的能力。

---

## NTFS-Search（C/C++，Windows）
- 代码位置：`NTFS-Search/NTFS_STRUCT.cpp`、`NTFS-Search.cpp`
- 核心要点：
  - 低层读 NTFS：实现 `ReadLCN/ReadExternalAttribute/ReadFileRecord` 等，手动处理 runlist、attribute、USA 修复，直接解码 MFT 记录。
  - 搜索线程：`_beginthreadex` 启动后台加载线程（`LoadSearchInfo`），UI 线程通过消息（WM_USER+1/2）接收完成通知。
  - 搜索逻辑：加载结构 → 遍历 MFT 中的文件名属性 → 简单模式匹配（GUI 侧填写模式）。
- 代码线索：
  - `ReadExternalData`：遍历 runlist 分段读取，提升顺序读效率。
  - `Waiting` 对话框中启动线程：快速返回 UI，后台载入。
- 对我们启发（macOS）：
  - 结构解析优化：在 macOS 上以“多线程遍历 + 分批写入 + 只读快照切换”取代 MFT 解码，达到“加载快 + 不阻塞 UI”。
  - 事件派发：Swift 使用 `DispatchSourceFileSystemObject`/FSEvents 监听，合并去抖后在后台执行增量维护。

---

## FileHunter（Java + Lucene）
- 代码位置：`src/main/java/com/ogefest/filehunter`
- 核心要点：
  - Lucene 索引：路径字段 `path`、内容 `content`，使用 `IndexSearcher` 进行布尔查询与加权（`path^1.2 OR content`）。
  - 自定义分词：`FHTokenizer` 将路径/文件名按 `[/\\.,:-_ ]` 等边界切分，利于路径片段匹配。
  - 任务执行：`Worker` 单线程消费任务队列；`App` 用 Quarkus `@Scheduled` 每 3/10/60 秒调度“检查可用/添加任务/消费队列”。
  - 目录遍历：`ReindexStructure.walk()` 递归访问统一抽象文件系统（UCFS），按忽略规则过滤，写入存储。
- 代码线索：
  - `LuceneSearch.queryByRawQuery`：用 `QueryParser` 解析查询，TopDocs → `searcher.doc()` → 结果。
  - `ReindexStructure.addToDatabase`：按忽略规则过滤后插入/更新记录。
- 对我们启发（macOS）：
  - 路径/文件名分词策略值得借鉴，可用于我们规则型匹配器的 token 切分。
  - 任务调度无需太复杂，Phase-1 以常驻服务 + 简单队列/定时器即可满足“周期检查 + 增量更新”。
  - 若仅做文件名检索，无需引入 Lucene；轻量内存索引 + SQLite 足够。

---

## 对 DocuRay（macOS）的技术建议

### 数据通路与存储
- 全量索引：多线程遍历（并发度=CPU×k），合并 stat，批量写入 SQLite（WAL）。
- 内存索引：
  - 结构：`vector<Record>` + `vector<NameNorm>` + 倒排（可选 n-gram 2/3-gram）。
  - 名称归一化：小写 + NFKC；路径/名称切词（借鉴 FHTokenizer 的边界字符集，扩展中文不切）。
- 快照切换：后台构建新索引 → 原子替换只读指针，查询无锁读。
- 增量更新：FSEvents 监听根目录树，事件合并（300–500ms）→ 局部更新 Store + Index；异常时回退局部再扫。

### 匹配与排序
- 匹配：前缀 > 词首子串 > 任意子串 > 子序列（可选），支持多 token 同时命中。
- 打分：`score = w1*matchType + w2*position + w3*1/len(name) + w4*extWeight + w5*dirWeight + w6*recency`。
- 高亮：返回 name/path 的命中区间，CLI/UI 高亮。

### 并发模型（Swift 实现）
- 索引并发：
  - 目录任务队列（`DispatchQueue`）+ 工作组（`DispatchGroup`）；
  - 按目录分片，子目录入队，限流最大并发；
  - 批量入库（每 N 条事务提交）。
- 查询并发：
  - 查询线程仅访问只读快照；
  - 融合/排序为 O(K)（K=候选数），首屏 Top-200；
  - 每击键取消上次任务（coalescing）。

### 关键伪代码
```swift
// 查询（只读快照）
func searchName(query: String, top: Int = 50) -> [Hit] {
  let q = normalizeAndTokenize(query)
  var candidates = intersectOrFilter(tokens: q.tokens, index: nameIndex)
  candidates = scoreAndSort(candidates, q.tokens)
  return Array(candidates.prefix(top))
}

// 索引（并发遍历 + 批量写入）
func rebuildIndex(roots: [URL]) {
  let group = DispatchGroup()
  let q = DispatchQueue(label: "scan", attributes: .concurrent)
  let writer = BulkWriter(sqlite)
  for root in roots { q.async(group: group) { scan(root, writer) } }
  group.wait()
  writer.flush()
  buildInMemoryIndexFrom(sqlite)
  atomicallySwapSnapshot()
}
```

---

## 风险与对策
- 大目录/海量文件：
  - 对策：排除规则、热目录优先、限流；批量写库；必要时分段载入内存索引。
- Unicode/中文：
  - 对策：NFKC 小写化；中文不切词，按字符匹配；路径分隔符/标点作为边界。
- FSEvents 丢事件：
  - 对策：事件去抖 + 周期性“轻扫”对账；记录上次快照时间。
- 内存占用：
  - 对策：结构压缩、只读 mmap、Top-N 截断；必要时分 shard 存放。

---

## 代码参考索引
- UltraFast：
  - 并发队列：`UltraFastFileSearch-code/BackgroundWorker.hpp`
  - 匹配：`UltraFastFileSearch-code/string_matcher.hpp`
  - 设计说明：`UltraFastFileSearch-code/The Manual.md`（并行读 MFT、解析匹配）
- NTFS-Search：
  - NTFS 解析：`NTFS_STRUCT.cpp`（`ReadLCN/ReadExternalAttribute/ReadExternalData` 等）
  - 加载线程：`NTFS-Search.cpp`（`_beginthreadex` + UI 消息）
- FileHunter：
  - 搜索：`storage/LuceneSearch.java`（IndexSearcher + QueryParser + BooleanQuery）
  - 分词：`FHTokenizer.java`（路径分隔符/标点作为边界）
  - 任务：`task/Worker.java`、`task/ReindexStructure.java`（队列执行 + 递归遍历）

---

本对比文档将作为 DocuRay Phase-1【基于文件名的离线高速搜索】功能的技术参考。实际实现以 Swift 并发、FSEvents 增量与轻量内存索引为核心，结合上述三者的成熟策略（并发模型、分词与匹配、快照切换）。
