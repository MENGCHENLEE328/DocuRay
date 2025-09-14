# DocuRay FastName（Core，无 MCP/LLM）MVP 开发文档（TDD）

## 0. 背景与目标
- 背景：Phase-1 聚焦“完全离线、不依赖 LLM/Agent、不依赖 MCP”的本地文件名极速定位，体验超越 Finder/Spotlight 的即时性与模糊匹配。
- 目标指标：
  - 查询时延：P50 < 20ms，P95 < 120ms（名称检索通路，Top-N 首屏返回）。
  - 全量索引：10 万文件 < 60s（SSD），百万级允许后台完成；增量更新 < 2s 生效。
  - 覆盖：中文/多语言路径；大小写/连字符/驼峰等模糊匹配；可配置根目录与排除规则。

## 1. 架构与边界（Phase-1）
- 可执行服务（Swift）+ CLI 工具（同一产物提供子命令）：不引入 MCP/LLM/网络依赖。
- 内部模块：
  - Scanner：并发遍历根目录，采集 path/name/dir/dev/ino/mtime/size/ext；批量写入 SQLite。
  - Store（SQLite）：WAL 模式；files 表与必要索引；快照元数据。
  - NameIndex：内存索引（名称数组 + 归一化 token + 子串/子序列匹配器）。
  - Watcher：FSEvents 订阅，事件合并/去抖，驱动局部更新。
  - Search：匹配、打分、Top-N 截断、高亮范围计算（名称/路径）。
  - Open：open -R / open / editor --goto。
- 进程通信：Phase-1 仅 CLI；若需程序化调用，暴露本地 HTTP（可选，默认不启用）。
- Phase-2 扩展：在此核心之上加一层“适配器进程”实现 MCP（JSON-RPC/stdio）以供 SDK/Agent 调用。

## 2. 数据库 Schema（MVP）
```sql
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  name TEXT,
  dir  TEXT,
  dev  INTEGER,
  ino  INTEGER,
  size INTEGER,
  mtime INTEGER,
  ext  TEXT,
  snapshot_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_files_name ON files(name);
CREATE INDEX IF NOT EXISTS idx_files_dir  ON files(dir);
CREATE INDEX IF NOT EXISTS idx_files_ext  ON files(ext);
```

## 3. 匹配与排序（规则优先）
- 归一化：小写、可选去音调；按 `- _ 空格` 与驼峰切词；中文按字符连续匹配。
- 匹配优先级：前缀 > 词首子串 > 任意子串 > 子序列（可选）。
- 排序权重：match_type > 起始位置 > 名称长度惩罚 > 扩展名权重 > 目录权重（白名单目录优先）> mtime 权重。
- 高亮：返回 name/path 的命中区间，便于 CLI 高亮显示。

## 4. CLI 规范（Phase-1）
- `fastname index --roots ~/Documents ~/Desktop --exclude .git node_modules --no-follow-symlinks`
- `fastname search "关键字" --top 50 [--ext pdf,md,txt] [--root ~/Projects]`
- `fastname open --path /abs/path --mode reveal|open|editor [--editor code] [--line 123]`
- `fastname status`（索引规模、快照、FSEvents 延迟、最近更新）

## 5. TDD 任务拆分（RED→GREEN→REFACTOR）

### T1 工程初始化与 CLI 外壳
- 目标：SwiftPM 工程 + 子命令解析（index/search/open/status），命令帮助可用。
- 测试（CLITests）：
  - test_help_outputs_commands
  - test_invalid_args_returns_nonzero

### T2 SQLite 持久层（Store）
- 目标：初始化 DB；单条 upsert/批量 insert/查询；WAL 模式。
- 测试（StoreTests）：
  - test_init_creates_tables
  - test_upsert_and_get_by_path
  - test_bulk_insert_under_threshold（10k 行 < 200ms，因机而异）

### T3 全量扫描器（Scanner）
- 目标：并发遍历根目录；尊重 exclude；默认不跟随符号链接。
- 测试（ScannerTests）：
  - test_scan_counts_correct
  - test_exclude_rules_respected
  - test_symlink_handling_default_no_follow

### T4 内存索引（NameIndex）
- 目标：从 Store 加载内存结构；支持增删改；子串匹配。
- 测试（NameIndexTests）：
  - test_load_from_store
  - test_substring_match_and_ordering
  - test_incremental_add_remove_update

### T5 搜索与高亮（Search）
- 目标：实现查询→打分→Top-N；返回高亮区间与路径。
- 测试（SearchTests）：
  - test_prefix_beats_substring
  - test_highlight_ranges_correct
  - test_filter_by_ext_and_root

### T6 变更监听（Watcher）
- 目标：FSEvents 订阅根目录；事件合并/去抖；局部更新索引与存储。
- 测试（WatcherTests）：
  - test_create_file_appears
  - test_rename_updates
  - test_remove_disappears

### T7 性能与快照（Perf/Snapshot）
- 目标：后台重建索引快照并原子切换；查询无锁读；性能达标。
- 测试（PerfTests）：
  - test_snapshot_switch_atomic
  - test_query_latency_p95_threshold

### T8 打开定位（Open）与用户体验
- 目标：实现 open/reveal/editor；CLI 结果高亮输出。
- 测试（OpenTests）：
  - test_reveal_success
  - test_editor_goto_line_command_built

### T9 配置与状态（Config/Status）
- 目标：持久化根目录/排除规则；状态查询。
- 测试（ConfigTests / StatusTests）：
  - test_config_persist_and_reload
  - test_status_shows_expected_fields

## 6. 性能与监控
- 指标：查询时延分布、QPS、全量索引耗时、增量生效延迟、内存占用。
- 工具：开发期 CLI 报表；日志记录索引任务/错误/事件吞吐。

## 7. 风险与对策
- 大目录与深树：目录优先队列；排除常见巨型目录；批量写库；分段加载。
- Unicode/大小写：NFKC 归一化与小写；中文路径按字符匹配。
- FSEvents 积压：合理去抖与阈值，超限回退局部再扫；记录上次索引点。
- TCC 权限：首启引导；仅索引用户选择的根目录。

## 8. 与后续集成的关系
- Phase-1：纯本地 CLI/服务，不引入 MCP/LLM。
- Phase-2：在核心之上添加“MCP 适配层”（单独进程或同进程模块），暴露与现有文档一致的 MCP 工具名与 Schema；TypeScript/Claude Code SDK 通过 MCP 调用，不改核心实现。

