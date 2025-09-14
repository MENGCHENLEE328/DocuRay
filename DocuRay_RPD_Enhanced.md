# DocuRay — RPD（Requirements · Product · Design）算法增强版

## 0. 概述与定位

**DocuRay**：一款"穿透式本地/企业文档检索"桌面产品。
目标是在 **1–2 秒内** 从自然语言查询直达**文件 → 页/行/字节区间**，并给出**可追溯证据**（高亮片段、锚点、路径），体验风格接近 **Spotlight / Everything** 的"轻、快、准"，同时结合 LLM 做**意图理解与少量高智重排**。

**核心主张**
* *Intent → Page/Line*：从意图直达页/行级别定位
* *Fast First*：先快后准，粗到细级联；高置信早停
* *Explainable*：每条结果都可回放到页/行/字节区间
* *Local-first*：默认离线索引与本地向量；云端 LLM 仅在 Top 片段上少量使用

**不做的**
* 不把整库原文上传 LLM；不做重型 DMS/知识库编辑器；不做全自动跨组织权限提升。

---

## 1. 用户与场景（Top 用例）

* **工程研发**：自然语义找"实现/调用/错误栈对应代码段"。
* **投研/法务/运营**：在 PDF/Docx/Markdown 中定位关键条款、数据表、结论页。
* **团队知识检索（个人版起步）**：本地文档、常用网盘同步目录的快速定位。

**关键用例**
1. "找一篇 2024 年的市场报告里关于'商业地产空置率'的具体表格" → *秒级返回表格页与单元格锚点*
2. "auth 中间件里处理 401 的位置在哪一行？" → *返回文件/函数/行号、上下文窗口*
3. "会议纪要里李雷提到的交付时间是哪一页？" → *页级高亮、证据句*

---

## 2. 成功指标（SLO/KPI）

* **P50/P95 延迟**：≤1.5s / ≤4s（本地 100k 文档量级；索引就绪）
* **Precision@1（唯一文件判定）**：≥80%（上线两周内），≥90%（三个月）
* **可追溯率**：返回结果均含页/行/字节区间锚点 ≥95%
* **资源占用**：空闲 < 150MB；重检索峰值 < 1.2GB（可通过设置调节）
* **崩溃率**：< 0.1% 会话
* **索引吞吐率**：≥ 100MB/s（文本文档），≥ 50MB/s（PDF/图像）
* **增量更新延迟**：< 5s（单文件变更到可检索）

---

## 3. 技术栈选型

### 客户端与壳
* **Tauri（Rust+Web）优先** / 备选 Electron
* UI：React + Tailwind；快捷键框架（cmd/ctrl + 空格 唤起）
* 原生集成：
  * macOS：FSEvents、Spotlight（可选读取元数据）
  * Windows：USN Journal、MFT 快速遍历（只读）

### 后端（本地服务）
* **语言**：Rust（高性能、跨平台、低占用）
* **倒排索引**：Tantivy（嵌入式 Lucene 同类）
* **向量索引**：Qdrant（本地服务模式）或 sqlite-vec（轻量）
* **内容解析**：Unstructured/Tika + pdfium/pymupdf；代码用 tree-sitter
* **OCR**：tesseract（离线）；可选引入快速模型
* **嵌入模型**：bge-m3 / e5 系列（文本）；code-embeddings（代码）；CLIP/SigLIP（图像）
* **LLM**：云端 Sonnet / 可切换开源本地（仅用于 Top 片段重排与解释）
* **通信**：本地 gRPC/HTTP；前端通过 localhost 访问
* **存储**：RocksDB/SQLite（元数据、映射、缓存）

---

## 4. 总体架构

```
+-------------------+        +-----------------------+
|  UI Launcher/UX   |<-----> |  Local API Gateway    |
|  (Spotlight风格)  |        |  (HTTP/gRPC)          |
+-------------------+        +-----------+-----------+
                                        |
                                 +------+------+
                                 | Orchestrator|  ← LLM 做意图/计划/少量重排
                                 +------+------+ 
                                        |
        +------------------ 并发/可取消任务通道 ------------------+
        |              |                |                |      |
  +-----v----+   +-----v-----+    +-----v-----+   +-----v----+  +-----v-----+
  | MetaAgent|   | Lexical   |    | Vector    |   | TOCAgent |  | TableAgent|
  |(名/路径/)|   |(倒排/BM25)|    |(ANN向量)  |   |(目录/标题)|  |(表格/数值) |
  +-----+----+   +-----+-----+    +-----+-----+   +-----+----+  +-----+-----+
        |              |                |                |             |
        +--------------+----------------+----------------+-------------+
                                       |
                                 +-----v-----+
                                 | Fusion&Rerank (RRF→CE→LLM) |
                                 +-----+-----+
                                       |
                                 +-----v-----+
                                 | Uniqueness |
                                 | & Verify   |  （唯一性评分+定点核验）
                                 +-----+-----+
                                       |
                                 +-----v-----+
                                 | Results w/ |
                                 | Anchors    | （页/行/字节区间/证据）
                                 +-----------+
```

---

## 5. 功能模块（算法增强）

### 5.1 索引器（Indexer）- 算法细节

#### 5.1.1 智能切块算法
```python
class SmartChunker:
    def __init__(self):
        self.overlap_ratio = 0.2  # 20% 重叠
        self.min_chunk_size = 128  # tokens
        self.max_chunk_size = 512  # tokens
        self.boundary_detection = {
            'sentence': r'[.!?]+\s+',
            'paragraph': r'\n\n+',
            'section': r'^#+\s+',  # markdown headers
            'function': tree_sitter_parser,  # for code
        }
    
    def chunk(self, doc, doc_type):
        if doc_type == 'code':
            return self.ast_based_chunking(doc)
        elif doc_type in ['pdf', 'docx']:
            return self.layout_aware_chunking(doc)
        else:
            return self.semantic_chunking(doc)
    
    def semantic_chunking(self, text):
        # 使用滑动窗口 + 语义相似度边界检测
        embeddings = self.get_sentence_embeddings(text)
        boundaries = self.detect_semantic_boundaries(embeddings)
        return self.create_chunks_with_overlap(text, boundaries)
```

#### 5.1.2 多粒度索引策略
* **字符级索引**：支持精确字符串匹配和正则表达式
* **词级索引**：BM25 + 位置信息
* **句子级索引**：语义向量 + 句法结构
* **段落级索引**：主题模型 + 层次化摘要
* **文档级索引**：全局特征 + 元数据

#### 5.1.3 增量索引算法
```python
class IncrementalIndexer:
    def __init__(self):
        self.version_tree = BPlusTree()  # 版本控制
        self.delta_index = DeltaIndex()  # 增量存储
        self.merge_threshold = 1000  # 合并阈值
    
    def update(self, doc_changes):
        # 1. 计算文档差异
        diff = self.compute_diff(doc_changes)
        
        # 2. 更新受影响的chunks
        affected_chunks = self.identify_affected_chunks(diff)
        
        # 3. 局部重索引
        for chunk in affected_chunks:
            self.delta_index.update(chunk)
        
        # 4. 触发合并策略
        if self.delta_index.size > self.merge_threshold:
            self.merge_to_main_index()
```

### 5.2 检索编排（Orchestrator）- 算法增强

#### 5.2.1 查询理解与改写
```python
class QueryProcessor:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = NERModel()
        self.query_expander = QueryExpander()
    
    def process(self, query):
        # 1. 意图分类
        intent = self.intent_classifier.classify(query)
        
        # 2. 实体抽取
        entities = self.entity_extractor.extract(query)
        
        # 3. 查询扩展
        expansions = []
        if intent == 'definition':
            expansions = self.expand_with_synonyms(query)
        elif intent == 'code_search':
            expansions = self.expand_with_code_patterns(query)
        elif intent == 'temporal':
            expansions = self.expand_with_time_variants(query)
        
        # 4. 生成多路查询计划
        return QueryPlan(
            original=query,
            intent=intent,
            entities=entities,
            expansions=expansions,
            routing_strategy=self.determine_routing(intent)
        )
```

#### 5.2.2 自适应融合算法
```python
class AdaptiveFusion:
    def __init__(self):
        self.weight_learner = OnlineWeightLearner()
        self.score_normalizer = ScoreNormalizer()
    
    def fuse(self, results_dict, query_context):
        # 1. 归一化各通道分数
        normalized = {}
        for channel, results in results_dict.items():
            normalized[channel] = self.score_normalizer.normalize(
                results, 
                method=self.select_normalization(channel)
            )
        
        # 2. 动态权重计算
        weights = self.weight_learner.get_weights(
            query_context,
            channel_confidences=self.estimate_confidences(normalized)
        )
        
        # 3. 加权融合
        fused_scores = {}
        for doc_id in self.get_all_doc_ids(normalized):
            score = 0
            evidence = []
            for channel, weight in weights.items():
                if doc_id in normalized[channel]:
                    channel_score = normalized[channel][doc_id]
                    score += weight * channel_score
                    evidence.append((channel, channel_score))
            fused_scores[doc_id] = (score, evidence)
        
        return sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
```

#### 5.2.3 早停机制
```python
class EarlyStoppingStrategy:
    def __init__(self):
        self.confidence_threshold = 0.95
        self.min_channels = 2  # 至少等待2个通道
        self.timeout_ms = 500
    
    def should_stop(self, current_results, elapsed_ms):
        if len(current_results) < self.min_channels:
            return False
        
        # 计算置信度
        top_score = current_results[0].score
        second_score = current_results[1].score if len(current_results) > 1 else 0
        
        confidence = self.calculate_confidence(
            top_score, 
            second_score,
            channel_agreement=self.check_channel_agreement(current_results)
        )
        
        # 动态阈值调整
        adjusted_threshold = self.adjust_threshold(
            self.confidence_threshold,
            elapsed_ms,
            self.timeout_ms
        )
        
        return confidence > adjusted_threshold
```

### 5.3 向量检索优化

#### 5.3.1 层次化向量索引
```python
class HierarchicalVectorIndex:
    def __init__(self):
        self.levels = 3  # 三层索引
        self.cluster_sizes = [1000, 100, 10]  # 每层聚类大小
        self.indices = []
        
    def build(self, vectors):
        # Level 0: 粗粒度聚类中心
        centers_l0 = self.kmeans(vectors, n_clusters=len(vectors)//self.cluster_sizes[0])
        self.indices.append(FAISSIndex(centers_l0))
        
        # Level 1: 中粒度索引
        for cluster in centers_l0:
            cluster_vectors = self.get_cluster_vectors(vectors, cluster)
            centers_l1 = self.kmeans(cluster_vectors, n_clusters=self.cluster_sizes[1])
            self.indices.append(FAISSIndex(centers_l1))
        
        # Level 2: 细粒度全量索引
        self.indices.append(FAISSIndex(vectors, method='IVF_PQ'))
    
    def search(self, query_vector, k=10, early_termination=True):
        candidates = []
        
        # 逐层搜索
        for level, index in enumerate(self.indices):
            level_candidates = index.search(query_vector, k * (2 ** (2-level)))
            candidates.extend(level_candidates)
            
            if early_termination and self.check_convergence(candidates):
                break
        
        return self.rerank_candidates(candidates, query_vector, k)
```

#### 5.3.2 向量压缩与量化
```python
class VectorCompressor:
    def __init__(self):
        self.pq_dim = 32  # Product Quantization维度
        self.scalar_quantizer = ScalarQuantizer(bits=8)
        
    def compress(self, vectors):
        # 1. PCA降维
        reduced = self.pca_reduce(vectors, keep_variance=0.95)
        
        # 2. Product Quantization
        pq_codes = self.product_quantize(reduced, m=self.pq_dim)
        
        # 3. 标量量化残差
        residuals = vectors - self.reconstruct_pq(pq_codes)
        quantized_residuals = self.scalar_quantizer.quantize(residuals)
        
        return CompressedVectors(pq_codes, quantized_residuals)
```

### 5.4 结果重排序

#### 5.4.1 学习排序（Learning to Rank）
```python
class LTRReranker:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.ranker = XGBoostRanker()
        
    def extract_features(self, query, doc, initial_scores):
        features = []
        
        # 查询-文档特征
        features.extend([
            self.calculate_bm25(query, doc),
            self.calculate_tfidf_cosine(query, doc),
            self.calculate_semantic_similarity(query, doc),
            self.calculate_entity_overlap(query, doc),
        ])
        
        # 文档特征
        features.extend([
            doc.page_rank,
            doc.freshness_score,
            doc.authority_score,
            doc.readability_score,
        ])
        
        # 上下文特征
        features.extend([
            initial_scores.get('lexical', 0),
            initial_scores.get('vector', 0),
            initial_scores.get('metadata', 0),
        ])
        
        return np.array(features)
    
    def rerank(self, query, candidates):
        features = []
        for doc in candidates:
            feat = self.extract_features(query, doc, doc.initial_scores)
            features.append(feat)
        
        scores = self.ranker.predict(np.array(features))
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

---

## 6. 算法性能优化策略

### 6.1 内存优化
```python
class MemoryOptimizer:
    def __init__(self):
        self.cache_size_mb = 512
        self.lru_cache = LRUCache(maxsize=10000)
        self.mmap_threshold = 100 * 1024 * 1024  # 100MB
        
    def optimize_vector_storage(self):
        # 1. 热数据内存映射
        hot_vectors = self.identify_hot_vectors()
        self.mmap_hot_vectors(hot_vectors)
        
        # 2. 冷数据压缩存储
        cold_vectors = self.identify_cold_vectors()
        self.compress_and_store(cold_vectors)
        
        # 3. 动态加载策略
        self.setup_lazy_loading()
```

### 6.2 并发优化
```python
class ConcurrencyOptimizer:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.async_io = AsyncIOManager()
        
    async def parallel_search(self, query, channels):
        tasks = []
        for channel in channels:
            task = asyncio.create_task(
                self.search_channel(query, channel)
            )
            tasks.append(task)
        
        # 渐进式返回结果
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            yield result  # 流式返回
```

### 6.3 缓存策略
```python
class MultiLevelCache:
    def __init__(self):
        self.query_cache = QueryCache(ttl=300)  # 5分钟
        self.vector_cache = VectorCache(size_mb=256)
        self.result_cache = ResultCache(max_items=1000)
        
    def get_or_compute(self, key, compute_fn):
        # L1: 查询缓存
        if result := self.query_cache.get(key):
            return result
        
        # L2: 向量缓存
        if vectors := self.vector_cache.get_similar(key):
            result = self.fast_compute(vectors)
            self.query_cache.set(key, result)
            return result
        
        # L3: 完整计算
        result = compute_fn()
        self.update_all_caches(key, result)
        return result
```

---

## 7. 算法评估与调优

### 7.1 离线评估框架
```python
class OfflineEvaluator:
    def __init__(self):
        self.metrics = {
            'precision': PrecisionCalculator(),
            'recall': RecallCalculator(),
            'mrr': MRRCalculator(),
            'ndcg': NDCGCalculator(),
            'latency': LatencyProfiler(),
        }
        
    def evaluate(self, test_set):
        results = {}
        for query, ground_truth in test_set:
            predictions = self.system.search(query)
            
            for metric_name, calculator in self.metrics.items():
                score = calculator.calculate(predictions, ground_truth)
                results.setdefault(metric_name, []).append(score)
        
        # 统计分析
        return {
            name: {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'p50': np.percentile(scores, 50),
                'p95': np.percentile(scores, 95),
            }
            for name, scores in results.items()
        }
```

### 7.2 在线A/B测试
```python
class OnlineABTester:
    def __init__(self):
        self.experiments = {}
        self.traffic_allocator = TrafficAllocator()
        
    def run_experiment(self, user_id, query):
        # 流量分配
        variant = self.traffic_allocator.assign(user_id)
        
        # 执行对应策略
        if variant == 'control':
            results = self.baseline_search(query)
        else:
            results = self.experimental_search(query, variant)
        
        # 记录指标
        self.log_metrics(user_id, variant, query, results)
        
        return results
```

### 7.3 自动参数调优
```python
class AutoTuner:
    def __init__(self):
        self.param_space = {
            'chunk_size': (128, 1024),
            'overlap_ratio': (0.1, 0.3),
            'fusion_weights': {
                'lexical': (0.1, 0.5),
                'vector': (0.3, 0.7),
                'metadata': (0.1, 0.3),
            },
            'rerank_topk': (10, 100),
        }
        self.optimizer = BayesianOptimizer()
        
    def optimize(self, eval_set):
        def objective(params):
            self.system.update_params(params)
            metrics = self.evaluator.evaluate(eval_set)
            return metrics['ndcg']['mean']  # 优化目标
        
        best_params = self.optimizer.maximize(
            objective,
            self.param_space,
            n_iter=50
        )
        
        return best_params
```

---

## 8. 工程实现路径

### Phase 1: MVP核心功能（2周）
1. **基础索引**：实现简单的倒排索引 + 基础向量检索
2. **单通道检索**：先实现纯词汇检索，确保端到端流程
3. **简单UI**：命令行或极简GUI，验证交互流程
4. **性能基准**：建立基础性能测试框架

### Phase 2: 多通道融合（2周）
1. **向量检索集成**：添加预训练模型的向量检索
2. **融合算法**：实现基础的RRF融合
3. **并发框架**：建立多通道并发检索架构
4. **早停机制**：实现基础的置信度早停

### Phase 3: 智能优化（2周）
1. **查询理解**：集成LLM进行查询改写
2. **重排序**：实现Cross-Encoder重排
3. **缓存系统**：多级缓存提升响应速度
4. **增量索引**：支持文件变更的实时更新

### Phase 4: 生产化（2周）
1. **性能优化**：内存优化、索引压缩
2. **监控系统**：添加性能监控和日志
3. **错误处理**：完善异常处理和降级策略
4. **打包发布**：跨平台打包和自动更新

---

## 9. 关键技术挑战与解决方案

### 9.1 大规模文档处理
**挑战**：100k+ 文档的索引构建和维护
**解决方案**：
- 分片索引 + 并行构建
- 增量更新 + 定期合并
- 冷热分离存储策略

### 9.2 多模态内容理解
**挑战**：PDF布局、表格结构、代码语义
**解决方案**：
- 专用解析器链（pdf → layout → table → cell）
- AST驱动的代码理解
- 多模态嵌入模型

### 9.3 实时性能保证
**挑战**：1-2秒响应时间约束
**解决方案**：
- 多级缓存策略
- 早停机制
- 异步流式返回

### 9.4 相关性判定
**挑战**：不同类型查询的相关性标准不同
**解决方案**：
- 意图感知的评分策略
- 上下文相关的权重学习
- 用户反馈的在线学习

---

## 10. 数据结构与算法详细设计

### 10.1 核心数据结构
```rust
// Rust实现的核心数据结构
struct Document {
    id: DocumentId,
    path: PathBuf,
    content_hash: [u8; 32],
    chunks: Vec<ChunkId>,
    metadata: DocumentMetadata,
    index_version: u32,
}

struct Chunk {
    id: ChunkId,
    doc_id: DocumentId,
    content: String,
    byte_range: (usize, usize),
    page_range: Option<(u32, u32)>,
    line_range: Option<(u32, u32)>,
    embedding: Option<Vec<f32>>,
    chunk_type: ChunkType,
    anchors: Vec<Anchor>,
}

struct Anchor {
    anchor_type: AnchorType,  // Title, Table, Function, etc.
    value: String,
    confidence: f32,
    location: Location,
}

enum ChunkType {
    Text,
    Code { language: String, ast_node: String },
    Table { headers: Vec<String> },
    Image { ocr_text: Option<String> },
}
```

### 10.2 索引结构
```rust
struct MultiIndex {
    lexical: TantivyIndex,
    vector: VectorIndex,
    metadata: MetadataIndex,
    structure: StructureIndex,  // TOC, tables, etc.
}

impl MultiIndex {
    async fn search(&self, query: Query) -> SearchResults {
        // 并发搜索所有索引
        let (lex, vec, meta, struct) = tokio::join!(
            self.lexical.search(&query),
            self.vector.search(&query),
            self.metadata.search(&query),
            self.structure.search(&query),
        );
        
        // 融合结果
        self.fuse_results(lex, vec, meta, struct)
    }
}
```

---

## 11. 性能基准与优化目标

### 11.1 性能基准测试
```python
class PerformanceBenchmark:
    def __init__(self):
        self.test_corpus = {
            'small': 1000,      # 文档数
            'medium': 10000,
            'large': 100000,
        }
        self.query_types = [
            'exact_match',      # 精确匹配
            'semantic',         # 语义查询
            'complex',          # 复杂组合查询
            'code_search',      # 代码搜索
            'table_lookup',     # 表格查询
        ]
    
    def run_benchmarks(self):
        results = {}
        for corpus_size in self.test_corpus:
            for query_type in self.query_types:
                metrics = self.measure_performance(corpus_size, query_type)
                results[f"{corpus_size}_{query_type}"] = metrics
        return results
    
    def measure_performance(self, corpus_size, query_type):
        return {
            'latency_p50': self.measure_latency(0.5),
            'latency_p95': self.measure_latency(0.95),
            'latency_p99': self.measure_latency(0.99),
            'throughput': self.measure_throughput(),
            'memory_peak': self.measure_memory_peak(),
            'cpu_usage': self.measure_cpu_usage(),
        }
```

### 11.2 优化目标矩阵
| 指标 | 当前基线 | 3个月目标 | 6个月目标 | 优化策略 |
|------|----------|-----------|-----------|----------|
| 索引构建速度 | 50MB/s | 100MB/s | 200MB/s | 并行处理、增量索引 |
| 查询延迟(P50) | 2s | 1s | 500ms | 缓存、早停、预计算 |
| 查询延迟(P95) | 5s | 3s | 2s | 异步IO、流式返回 |
| 内存占用(空闲) | 300MB | 150MB | 100MB | 压缩、冷热分离 |
| 准确率(P@1) | 70% | 85% | 95% | LTR、用户反馈学习 |

---

## 12. 监控与可观测性

### 12.1 关键监控指标
```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            # 性能指标
            'query_latency': Histogram(),
            'index_throughput': Gauge(),
            'cache_hit_rate': Counter(),
            
            # 质量指标
            'click_through_rate': Counter(),
            'result_satisfaction': Gauge(),
            'zero_result_rate': Counter(),
            
            # 系统指标
            'memory_usage': Gauge(),
            'cpu_usage': Gauge(),
            'index_size': Gauge(),
            'document_count': Counter(),
            
            # 业务指标
            'daily_active_queries': Counter(),
            'unique_users': HyperLogLog(),
            'query_types': Counter(),
        }
```

### 12.2 日志与追踪
```python
class QueryTracer:
    def trace_query(self, query_id, query):
        span = self.tracer.start_span("search_query")
        
        # 记录查询生命周期
        with span:
            span.set_tag("query.text", query.text)
            span.set_tag("query.type", query.intent)
            
            # 追踪各阶段
            with span.start_child("query_understanding"):
                # ...
            
            with span.start_child("multi_channel_search"):
                # ...
            
            with span.start_child("fusion_rerank"):
                # ...
            
            span.set_tag("results.count", len(results))
            span.set_tag("results.latency_ms", latency)
```

---

## 13. 失败模式与降级策略

### 13.1 降级策略
```python
class DegradationStrategy:
    def __init__(self):
        self.strategies = {
            'vector_index_failure': self.fallback_to_lexical,
            'llm_timeout': self.skip_llm_rerank,
            'memory_pressure': self.reduce_cache_size,
            'high_latency': self.enable_aggressive_early_stop,
        }
    
    def handle_failure(self, failure_type, context):
        if strategy := self.strategies.get(failure_type):
            return strategy(context)
        return self.default_fallback(context)
    
    def fallback_to_lexical(self, context):
        # 向量检索失败时，退化到纯词汇检索
        return LexicalOnlySearch(boost_factor=1.5)
    
    def skip_llm_rerank(self, context):
        # LLM超时时，跳过智能重排
        return SimpleScoreFusion()
```

### 13.2 错误恢复
```python
class ErrorRecovery:
    def __init__(self):
        self.retry_policy = ExponentialBackoff(
            initial_delay=100,  # ms
            max_delay=5000,
            max_retries=3
        )
    
    async def robust_search(self, query):
        try:
            return await self.primary_search(query)
        except VectorIndexError:
            logger.warning("Vector index failed, using fallback")
            return await self.fallback_search(query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # 返回缓存结果或基础结果
            return self.get_cached_or_basic_results(query)
```

---

## 14. 算法创新点

### 14.1 自适应查询路由
基于查询特征动态选择最优检索路径：
- 代码查询 → AST检索优先
- 表格查询 → 结构化检索优先
- 定义查询 → 向量检索优先
- 精确查询 → 倒排索引优先

### 14.2 上下文感知重排
利用用户历史和会话上下文改进排序：
- 最近访问文档加权
- 项目相关性提升
- 时间衰减因子

### 14.3 渐进式精细化
从粗粒度到细粒度的多阶段检索：
1. 文档级快速筛选（100ms）
2. 段落级精确定位（500ms）
3. 行级锚点验证（200ms）

---

## 15. 测试策略

### 15.1 单元测试
```python
class AlgorithmTests:
    def test_chunking_boundary_detection(self):
        # 测试边界检测准确性
        text = "First sentence. Second sentence.\n\nNew paragraph."
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 2
        assert chunks[0].end == text.index('\n\n')
    
    def test_early_stopping_convergence(self):
        # 测试早停收敛性
        results = self.simulate_progressive_results()
        stop_point = self.early_stopper.find_stop_point(results)
        assert stop_point < len(results) * 0.5  # 应该在50%之前停止
    
    def test_fusion_weight_learning(self):
        # 测试权重学习有效性
        initial_weights = {'lexical': 0.33, 'vector': 0.33, 'meta': 0.34}
        trained_weights = self.train_weights(self.training_data)
        assert self.evaluate(trained_weights) > self.evaluate(initial_weights)
```

### 15.2 集成测试
```python
class IntegrationTests:
    def test_end_to_end_latency(self):
        # 端到端延迟测试
        query = "find authentication middleware error handling"
        start = time.time()
        results = self.system.search(query)
        latency = time.time() - start
        assert latency < 2.0  # 2秒内
        assert len(results) > 0
        assert results[0].has_anchor()  # 有锚点
    
    def test_concurrent_searches(self):
        # 并发搜索测试
        queries = [self.generate_query() for _ in range(100)]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.system.search, q) for q in queries]
            results = [f.result(timeout=5) for f in futures]
        assert all(r is not None for r in results)
```

---

## 16. 总结与后续优化方向

### 核心算法优势
1. **多通道协同**：充分利用不同检索方法的优势
2. **渐进式返回**：快速响应，逐步精细化
3. **可解释性强**：每个结果都有明确的证据链
4. **自适应优化**：基于用户反馈持续改进

### 潜在优化方向
1. **神经检索模型**：探索端到端的神经检索架构
2. **联邦学习**：在保护隐私的前提下共享学习
3. **知识图谱增强**：构建文档间的关系网络
4. **多模态融合**：更深层次的跨模态理解

### 风险与应对
1. **模型漂移**：定期重训练和在线学习
2. **分布变化**：监控数据分布，自适应调整
3. **扩展性瓶颈**：分布式架构预留，水平扩展能力
4. **隐私泄露**：差分隐私，本地化计算优先

---

## 附录A：关键算法伪代码

### A.1 完整检索流程
```python
async def search_with_early_stop(query, timeout_ms=2000):
    # 1. 查询理解
    plan = await query_planner.plan(query)
    
    # 2. 启动多通道检索
    channels = []
    for channel_name, budget in plan.channel_budgets.items():
        channel = create_channel(channel_name, budget)
        channels.append(channel.search_async(query))
    
    # 3. 渐进式收集结果
    results = []
    start_time = time.time()
    
    while channels and (time.time() - start_time) * 1000 < timeout_ms:
        # 等待任意通道完成
        done, pending = await asyncio.wait(
            channels, 
            return_when=asyncio.FIRST_COMPLETED,
            timeout=0.1
        )
        
        for task in done:
            channel_results = await task
            results.extend(channel_results)
            channels.remove(task)
        
        # 融合当前结果
        fused = fusion.fuse(results)
        
        # 检查是否可以早停
        if early_stopper.should_stop(fused, elapsed_ms=(time.time()-start_time)*1000):
            # 取消剩余任务
            for task in channels:
                task.cancel()
            break
    
    # 4. 最终重排和验证
    reranked = reranker.rerank(query, fused[:100])
    verified = await verifier.verify_anchors(reranked[:10])
    
    return verified
```

### A.2 智能切块算法
```python
def intelligent_chunk(document, doc_type):
    if doc_type == 'code':
        # 基于AST的代码切块
        tree = parse_ast(document.content)
        chunks = []
        for node in tree.traverse():
            if node.type in ['function', 'class', 'method']:
                chunk = create_chunk(
                    content=node.source,
                    start_line=node.start_line,
                    end_line=node.end_line,
                    chunk_type='code_block',
                    metadata={'symbol': node.name, 'type': node.type}
                )
                chunks.append(chunk)
        return chunks
    
    elif doc_type == 'structured':
        # 基于布局的结构化文档切块
        layout = extract_layout(document)
        chunks = []
        for element in layout.elements:
            if element.type == 'table':
                chunk = create_table_chunk(element)
            elif element.type == 'section':
                chunk = create_section_chunk(element)
            else:
                chunk = create_text_chunk(element)
            chunks.append(chunk)
        return chunks
    
    else:
        # 基于语义的文本切块
        sentences = split_sentences(document.content)
        embeddings = encode_sentences(sentences)
        boundaries = detect_semantic_boundaries(embeddings)
        return create_chunks_from_boundaries(sentences, boundaries)
```

---

这份增强版的RPD文档从算法工程师的角度补充了大量实现细节，重点关注了算法的可实现性和工程可用性。主要增强包括：

1. **详细的算法实现**：提供了关键组件的伪代码和实现思路
2. **性能优化策略**：内存、并发、缓存等多维度优化
3. **评估与调优框架**：完整的离线和在线评估体系
4. **渐进式实现路径**：分阶段的开发计划，降低实现风险
5. **监控与降级策略**：确保系统稳定性和可用性
6. **测试策略**：全面的测试覆盖，保证质量

这些补充使得DocuRay从一个产品概念变成了一个可执行的工程方案，为实际开发提供了清晰的技术路线图。
