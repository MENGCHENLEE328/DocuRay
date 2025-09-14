# SpotSearch TODO Checklist (目标：Beat macOS Spotlight)  
Author: Team DocuRay | Version: 0.1.0 | Modified: 2025-09-14

## Phase A — 文件名/路径（1–1.5 周）
- [ ] Per-volume 索引根与游标（VolumeUUID → meta.sqlite + 游标）
- [ ] FSEvents 增量（持久化 lastEventId，rename/modify/delete 去抖）
- [ ] 前缀索引（PrefixIndex）完善：tokenization、拼音/大小写/分词友好
- [ ] 模糊匹配（BK-tree/SymSpell）：错拼/缺字/替换代价配置
- [ ] 排序器（Ranking）：exact > prefix > substring > fuzzy；目录/最近使用加权
- [ ] 批量 flush 与背压（限流 + 批处理 + 低优先级QoS）
- [ ] 性能基线脚本：对比 `mdfind`（P50/P95，Top@K）

## Phase B — 内容级（1–2 周）
- [ ] 抽取器接口（PDF/MD/TXT/Code）→ chunk + Anchor（页/行/byte_range/符号）
- [ ] 倒排索引（FTS5/Tantivy）+ 短语/邻近查询
- [ ] 片段核验（命中锚点定点回读，避免整篇IO）
- [ ] 去重与版本折叠（SimHash + 路径规则 → 文档簇）
- [ ] 查询→候选→片段三级缓存（热目录 chunk 驻留）

## Phase C — 可靠性与评测（0.5–1 周）
- [ ] 健康面板：延迟、积压、回溯补抓、失败率、早停触发
- [ ] 错误与降级矩阵自动化：qdrant→chromadb、正则回退等
- [ ] 回放工具：重放 24h 查询，评估 nDCG/P@1 回归阈值
- [ ] 一键修复：索引损坏恢复（段重排+游标回溯）

## 交付与门槛（验收标准）
- [ ] 文件名/路径：P95 ≤ 30ms@1M 文件，Top@10 命中≥Spotlight
- [ ] 内容：nDCG@10 ≥ Spotlight × 1.15，P95 ≤ 400ms
- [ ] 稳定性：冷启动可断点续扫；增量丢事件可轻扫/重扫；降级可观测

## 迁移清单（FastNameCore → SpotSearch）
- [ ] UltraFastIndex → PrefixIndex（结构与批量优化）
- [ ] SimpleHashIndex/MinimalIndex → 基础字典与去重策略
- [ ] SearchScorer → Ranking（权重与可参数化）
- [ ] PersistentService/Scanner → per-volume 元数据与FSEvents接入
- [ ] ArgParser → 后续 CLI 层（独立于核心库）

备注：本清单用于逐项打勾推进，每完成一项进行版本化提交（含 [vX.Y.Z] 信息），并在PRD变更记录中同步说明。

