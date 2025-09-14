# DocuRay 2.0 - Agent搜索引擎架构 PRD

## 0. 架构革新：从Pipeline到Agent

### 核心理念转变

**DocuRay 1.0（传统架构）**
```
用户查询 → 固定Pipeline → 多通道检索 → 融合算法 → 返回结果
```

**DocuRay 2.0（Agent架构）**
```
用户查询 → LLM Agent理解 → 动态MCP工具编排 → 智能执行 → 自适应优化 → 返回结果
```

### 核心优势
- **无限可扩展**：添加新能力只需接入新的MCP工具
- **智能编排**：LLM根据查询动态选择最优工具组合
- **自主进化**：Agent可以学习新的工具组合模式
- **降低复杂度**：每个MCP工具专注单一功能，整体通过组合涌现

---

## 1. 产品定位

**DocuRay 2.0**：首个**Agent驱动的本地文档搜索引擎**

不是传统的搜索系统，而是一个**会思考的搜索助手**：
- Agent理解你的意图，而不只是匹配关键词
- Agent知道如何组合工具，而不是执行固定流程
- Agent可以解释决策过程，而不是黑盒返回结果

**核心价值主张**
> "让搜索像对话一样自然，让定位像思考一样智能"

---

## 2. 系统架构

### 2.1 四层架构设计

```yaml
Layer 1 - Interaction Layer（交互层）:
  - Natural Language Interface
  - Query Understanding  
  - Result Presentation
  - Feedback Collection

Layer 2 - Agent Orchestration Layer（Agent编排层）:
  - LLM Agent (Claude/GPT-4/Local)
  - MCP Tool Registry
  - Dynamic Planning
  - Execution Monitor
  - Learning Module

Layer 3 - Core Algorithm Layer（核心算法层 - 固化实现）:
  Performance Critical（性能关键）:
    - Early Stopping Engine（早停引擎）
    - Fusion Algorithm（融合算法）
    - Uniqueness Scorer（唯一性评分）
    - Ranking Engine（排序引擎）
  
  Intelligence Core（智能核心）:
    - Query Router（查询路由）
    - Confidence Calculator（置信度计算）
    - Result Verifier（结果验证）
    - Cache Strategy（缓存策略）

Layer 4 - MCP Tools Layer（MCP工具层 - 外部能力）:
  Document Processing（文档处理）:
    - pdf-parser-mcp（PDF解析）
    - ocr-mcp（OCR识别）
    - table-extractor-mcp（表格提取）
  
  Code Analysis（代码分析）:
    - tree-sitter-mcp（AST解析）
    - symbol-extractor-mcp（符号提取）
  
  Storage & Retrieval（存储检索）:
    - qdrant-mcp（向量数据库）
    - chromadb-mcp（备用向量库）
    - file-system-mcp（文件系统）
  
  External Services（外部服务）:
    - llm-api-mcp（LLM调用）
    - translation-mcp（翻译服务）
```

### 2.2 核心流程

```python
class DocuRayAgent:
    """DocuRay 2.0的核心Agent"""
    
    def __init__(self):
        # 核心算法实例（固化实现，高性能）
        self.early_stopper = EarlyStoppingEngine()
        self.fusion_engine = AdaptiveFusionEngine()  
        self.ranker = RankingEngine()
        self.router = QueryRouter()
        
        # MCP工具注册（外部能力）
        self.mcp_tools = {
            "pdf_parser": PDFParserMCP(),
            "code_analyzer": TreeSitterMCP(),
            "vector_db": QdrantMCP(),
            # ...
        }
    
    async def process_query(self, user_query: str):
        # Step 1: 理解意图（核心算法）
        intent = self.router.analyze_query(user_query)
        
        # Step 2: 制定计划（Agent智能）
        plan = await self.create_execution_plan(intent)
        
        # Step 3: 动态执行
        results = await self.execute_with_monitoring(plan)
        
        # Step 4: 智能优化（核心算法）
        optimized = self.fusion_engine.fuse(results)
        ranked = self.ranker.rank(optimized)
        
        # Step 5: 生成解释
        explanation = await self.explain_reasoning(ranked)
        
        return {
            "results": ranked,
            "explanation": explanation,
            "confidence": self.calculate_confidence(ranked)
        }
    
    async def execute_with_monitoring(self, plan):
        """执行计划，结合核心算法监控"""
        results = []
        start_time = time.time()
        
        for step in plan.steps:
            # 调用MCP工具
            if step.type == "mcp_tool":
                result = await self.call_mcp_tool(step.tool, step.params)
            # 调用核心算法
            else:
                result = self.call_core_algorithm(step.algorithm, step.params)
            
            results.append(result)
            
            # 核心算法：早停判断（不是MCP，是内置高性能实现）
            should_stop = self.early_stopper.should_stop(
                partial_results=results,
                elapsed_ms=(time.time() - start_time) * 1000,
                confidence=self.calculate_confidence(results)
            )
            
            if should_stop:
                break
        
        return results
```

---

## 3. MCP工具与核心算法划分

### 3.1 架构决策原则

**什么适合做成MCP工具？**
- ✅ **独立的外部系统**：PDF解析、OCR、数据库
- ✅ **可替换的实现**：不同的向量库、不同的LLM
- ✅ **异步IO密集型**：文件读取、网络请求、API调用
- ✅ **专业领域功能**：代码解析、表格提取、图像处理

**什么应该是核心固化算法？**
- ✅ **性能关键路径**：早停判断、置信度计算（需要<10ms响应）
- ✅ **紧密耦合逻辑**：融合算法、排序引擎（需要访问大量上下文）
- ✅ **高频调用算法**：缓存策略、路由决策（每次查询都要用）
- ✅ **核心竞争力**：独特的算法创新（不应该暴露为服务）

### 3.2 核心固化算法（Core Algorithms）

```python
class CoreAlgorithms:
    """DocuRay的核心算法，内置高性能实现"""
    
    # 🚀 性能关键（<10ms响应要求）
    class EarlyStoppingEngine:
        """早停引擎 - 需要实时判断，不能有网络延迟"""
        def should_stop(self, partial_results, elapsed_ms, confidence):
            # 高性能C++/Rust实现
            # 直接内存访问，无序列化开销
            pass
    
    # 🔥 智能核心（复杂上下文依赖）
    class AdaptiveFusionEngine:
        """自适应融合 - 需要访问所有中间结果"""
        def fuse(self, multi_channel_results, query_context):
            # 复杂的矩阵运算
            # 需要共享内存访问
            pass
    
    # ⚡ 高频调用（每个查询都要用）
    class QueryRouter:
        """查询路由 - 决定执行路径"""
        def analyze_query(self, query):
            # 本地NLP模型
            # 缓存的路由表
            pass
    
    # 🎯 核心竞争力（商业机密）
    class UniquenessScorer:
        """唯一性评分 - DocuRay的独特算法"""
        def calculate(self, results, evidence):
            # 专利算法
            # 不应该暴露为外部服务
            pass
```

### 3.3 MCP工具清单（External Tools）

| MCP工具 | 功能职责 | 选择理由 | 来源 |
|---------|---------|----------|------|
| **pdf-parser-mcp** | PDF解析、布局分析 | 独立的文档处理系统 | pdf.co |
| **tree-sitter-mcp** | 代码AST解析 | 专业的代码分析工具 | GitHub |
| **qdrant-mcp** | 向量存储与检索 | 独立的数据库系统 | Qdrant |
| **ocr-mcp** | 图像文字识别 | 专门的OCR服务 | Tesseract |
| **table-extractor-mcp** | 表格结构提取 | 复杂的CV算法 | Camelot |
| **file-system-mcp** | 文件操作 | 系统级IO操作 | 社区 |
| **llm-api-mcp** | LLM调用 | 外部API服务 | OpenAI/Claude |

### 3.4 架构优势分析

```yaml
性能优势:
  - 核心算法无网络开销
  - 早停判断 <10ms
  - 融合计算 <50ms
  - 零序列化成本

灵活性优势:
  - MCP工具可随时替换
  - 支持多种PDF解析器
  - 支持多种向量数据库
  - 易于添加新能力

安全性优势:
  - 核心算法不暴露
  - 商业机密保护
  - 减少攻击面

开发效率:
  - 清晰的职责边界
  - 并行开发
  - 独立测试
```

## 4. 核心固化算法实现

### 4.1 早停引擎（Early Stopping Engine）

```python
class EarlyStoppingEngine:
    """
    高性能早停引擎 - 核心固化实现
    设计目标：<10ms判断延迟
    """
    
    def __init__(self):
        self.confidence_history = []
        self.decision_cache = {}
        
    def should_stop(self, 
                   partial_results: list,
                   elapsed_ms: int,
                   query_complexity: float) -> tuple[bool, float]:
        """
        实时判断是否应该停止搜索
        
        为什么不做成MCP：
        1. 高频调用（每个查询调用5-10次）
        2. 延迟敏感（需要<10ms响应）
        3. 需要访问历史状态
        """
        
        # 快速路径：缓存命中
        cache_key = self._compute_cache_key(partial_results)
        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]
        
        # 计算置信度（优化的算法）
        confidence = self._calculate_confidence_fast(partial_results)
        
        # 动态阈值（根据查询复杂度和时间）
        threshold = self._adaptive_threshold(elapsed_ms, query_complexity)
        
        # 决策
        should_stop = confidence >= threshold or elapsed_ms > 1800  # 硬限制1.8s
        
        # 缓存决策
        self.decision_cache[cache_key] = (should_stop, confidence)
        
        return should_stop, confidence
    
    def _calculate_confidence_fast(self, results):
        """使用SIMD优化的置信度计算"""
        if not results:
            return 0.0
        
        # NumPy向量化计算（比循环快10x）
        scores = np.array([r.score for r in results[:5]])  # 只看Top5
        
        if len(scores) < 2:
            return scores[0] if len(scores) == 1 else 0.0
        
        # 关键指标
        score_gap = scores[0] - scores[1]
        relative_gap = score_gap / (scores[0] + 1e-6)
        distribution_entropy = -np.sum(scores * np.log(scores + 1e-6))
        
        # 加权组合
        confidence = (
            0.5 * min(relative_gap / 0.3, 1.0) +  # 相对差距
            0.3 * min(scores[0], 1.0) +            # 绝对分数
            0.2 * (1.0 - distribution_entropy)      # 分布集中度
        )
        
        return confidence
```

### 4.2 自适应融合引擎（Adaptive Fusion Engine）

```python
class AdaptiveFusionEngine:
    """
    多通道结果融合 - 核心固化实现
    设计目标：处理1000个结果 <50ms
    """
    
    def __init__(self):
        self.weight_matrix = self._init_weight_matrix()
        self.fusion_cache = LRUCache(maxsize=1000)
        
    def fuse(self, 
            channel_results: dict,
            query_features: dict) -> list:
        """
        融合多通道结果
        
        为什么不做成MCP：
        1. 需要访问大量中间状态
        2. 复杂的矩阵运算
        3. 性能关键路径
        """
        
        # 特征提取
        features = self._extract_features(query_features)
        
        # 动态权重计算（神经网络）
        weights = self._compute_weights_nn(features)
        
        # 并行融合（使用NumPy向量化）
        fused_scores = self._parallel_fusion(channel_results, weights)
        
        # 重排序
        results = self._rerank_with_diversity(fused_scores)
        
        return results
    
    def _parallel_fusion(self, channel_results, weights):
        """
        向量化的并行融合算法
        比循环实现快20x
        """
        # 构建稀疏矩阵
        doc_ids = set()
        for results in channel_results.values():
            doc_ids.update(r.id for r in results)
        
        # 创建评分矩阵
        n_docs = len(doc_ids)
        n_channels = len(channel_results)
        score_matrix = np.zeros((n_docs, n_channels))
        
        # 填充矩阵（向量化操作）
        doc_id_map = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        for j, (channel, results) in enumerate(channel_results.items()):
            for r in results:
                i = doc_id_map[r.id]
                score_matrix[i, j] = r.score
        
        # 加权融合（矩阵乘法）
        weight_vector = np.array([weights[ch] for ch in channel_results.keys()])
        fused_scores = score_matrix @ weight_vector
        
        return list(zip(doc_ids, fused_scores))
```

### 4.3 查询路由器（Query Router）

```python
class QueryRouter:
    """
    查询意图理解与路由 - 核心固化实现
    设计目标：<20ms路由决策
    """
    
    def __init__(self):
        # 预加载的小模型（ONNX格式，<50MB）
        self.intent_model = self._load_onnx_model("intent_classifier.onnx")
        self.entity_extractor = self._load_onnx_model("ner_model.onnx")
        self.pattern_matcher = self._compile_patterns()
        
    def analyze_query(self, query: str) -> dict:
        """
        分析查询意图，决定执行路径
        
        为什么不做成MCP：
        1. 每个查询都要调用
        2. 需要快速响应
        3. 模型需要常驻内存
        """
        
        # 并行执行三个分析
        intent_future = self._classify_intent_async(query)
        entities_future = self._extract_entities_async(query)
        patterns_future = self._match_patterns_async(query)
        
        # 等待结果
        intent = intent_future.result()
        entities = entities_future.result()
        patterns = patterns_future.result()
        
        # 路由决策
        route = self._decide_route(intent, entities, patterns)
        
        return {
            "intent": intent,
            "entities": entities,
            "patterns": patterns,
            "route": route,
            "confidence": self._calculate_route_confidence(intent, entities)
        }
    
    def _decide_route(self, intent, entities, patterns):
        """
        路由决策树（硬编码规则，极快）
        """
        # 精确匹配优先
        if patterns.get("exact_file_pattern"):
            return "exact_file_search"
        
        # 代码搜索
        if intent == "code_search" or entities.get("function_names"):
            return "code_ast_search"
        
        # 表格搜索
        if "table" in intent or patterns.get("numeric_pattern"):
            return "table_extraction"
        
        # 默认语义搜索
        return "semantic_search"
```

### 4.4 排序引擎（Ranking Engine）

```python
class RankingEngine:
    """
    结果排序 - 核心固化实现
    使用LightGBM进行Learning to Rank
    """
    
    def __init__(self):
        # 预训练的排序模型
        self.ranker = self._load_lgb_model("ranker.lgb")
        self.feature_extractor = FeatureExtractor()
        
    def rank(self, 
            candidates: list,
            query_features: dict) -> list:
        """
        重排序候选结果
        
        为什么不做成MCP：
        1. 需要提取复杂特征
        2. 模型推理需要低延迟
        3. 访问大量上下文
        """
        
        # 批量特征提取（向量化）
        features = self.feature_extractor.extract_batch(
            candidates, 
            query_features
        )
        
        # 模型推理（使用GPU如果可用）
        scores = self.ranker.predict(features, num_threads=4)
        
        # 结合原始分数
        final_scores = 0.7 * scores + 0.3 * np.array([c.score for c in candidates])
        
        # 排序
        ranked_indices = np.argsort(final_scores)[::-1]
        
        return [candidates[i] for i in ranked_indices]
```

### 4.5 性能基准

```yaml
算法性能指标:
  EarlyStoppingEngine:
    - 延迟: P50=3ms, P95=8ms, P99=15ms
    - 吞吐: 10,000 QPS
    - 内存: <10MB
  
  AdaptiveFusionEngine:
    - 延迟: P50=20ms, P95=45ms (1000个文档)
    - 吞吐: 2,000 QPS
    - 内存: <100MB
  
  QueryRouter:
    - 延迟: P50=5ms, P95=15ms
    - 吞吐: 5,000 QPS
    - 内存: <50MB (模型)
  
  RankingEngine:
    - 延迟: P50=15ms, P95=30ms (100个候选)
    - 吞吐: 3,000 QPS
    - 内存: <30MB (模型)
```

---

## 5. Agent编排策略

### 5.1 Agent决策流程

```python
class QueryPlanner:
    """Agent的查询规划器 - 协调核心算法与MCP工具"""
    
    def __init__(self):
        # 核心算法（本地高性能）
        self.router = QueryRouter()
        self.early_stopper = EarlyStoppingEngine()
        
        # MCP工具注册表
        self.mcp_tools = {
            "pdf_parser": PDFParserMCP(),
            "vector_db": QdrantMCP(),
            "code_analyzer": TreeSitterMCP(),
            # ...
        }
    
    async def plan_execution(self, query: str):
        """
        制定执行计划：结合核心算法决策和MCP工具调用
        """
        
        # Step 1: 核心算法 - 快速路由决策（<20ms）
        route_decision = self.router.analyze_query(query)
        
        # Step 2: Agent - 基于路由选择MCP工具组合
        if route_decision["route"] == "code_ast_search":
            plan = {
                "phase1": {
                    "parallel": [
                        {"type": "mcp", "tool": "code_analyzer", "action": "find_symbols"},
                        {"type": "mcp", "tool": "vector_db", "action": "search_code"}
                    ]
                },
                "phase2": {
                    "sequential": [
                        {"type": "core", "algorithm": "fusion", "params": {"strategy": "code"}},
                        {"type": "core", "algorithm": "rank", "params": {"model": "code_ranker"}}
                    ]
                }
            }
        elif route_decision["route"] == "table_extraction":
            plan = {
                "phase1": {
                    "parallel": [
                        {"type": "mcp", "tool": "pdf_parser", "action": "extract_tables"},
                        {"type": "mcp", "tool": "vector_db", "action": "search_numeric"}
                    ]
                },
                "phase2": {
                    "sequential": [
                        {"type": "core", "algorithm": "table_matcher"},
                        {"type": "core", "algorithm": "rank"}
                    ]
                }
            }
        else:
            plan = self.create_default_plan(route_decision)
        
        return plan
```

### 5.2 执行协调器

```python
class ExecutionCoordinator:
    """执行协调器 - 统一调度核心算法和MCP工具"""
    
    def __init__(self):
        self.core_algorithms = CoreAlgorithms()
        self.mcp_client = MCPClient()
        self.metrics = MetricsCollector()
    
    async def execute_plan(self, plan: dict, query: str):
        """
        执行计划，智能协调两类组件
        """
        results = []
        start_time = time.time()
        
        for phase_name, phase_config in plan.items():
            phase_start = time.time()
            
            # 并行执行
            if "parallel" in phase_config:
                tasks = []
                for step in phase_config["parallel"]:
                    if step["type"] == "mcp":
                        # MCP工具调用（可能有网络延迟）
                        task = self.call_mcp_async(step["tool"], step["action"])
                    else:
                        # 核心算法调用（本地快速）
                        task = self.call_core_async(step["algorithm"], step.get("params"))
                    tasks.append(task)
                
                # 等待所有任务完成
                phase_results = await asyncio.gather(*tasks)
                results.extend(phase_results)
            
            # 串行执行
            if "sequential" in phase_config:
                for step in phase_config["sequential"]:
                    if step["type"] == "core":
                        # 核心算法（传递之前的结果作为输入）
                        result = self.core_algorithms.call(
                            step["algorithm"], 
                            input_data=results,
                            params=step.get("params", {})
                        )
                    else:
                        # MCP工具
                        result = await self.mcp_client.call(
                            step["tool"],
                            step["action"],
                            input_data=results
                        )
                    results.append(result)
            
            # 核心算法：早停检查（每个阶段后）
            elapsed_ms = (time.time() - start_time) * 1000
            should_stop, confidence = self.core_algorithms.early_stopper.should_stop(
                partial_results=results,
                elapsed_ms=elapsed_ms,
                query_complexity=self.estimate_complexity(query)
            )
            
            # 记录指标
            self.metrics.record_phase(phase_name, time.time() - phase_start)
            
            if should_stop:
                self.metrics.record_early_stop(phase_name, confidence)
                break
        
        return results
    
    async def call_mcp_async(self, tool: str, action: str):
        """异步调用MCP工具"""
        try:
            start = time.time()
            result = await self.mcp_client.call(tool, action)
            self.metrics.record_mcp_call(tool, action, time.time() - start)
            return result
        except MCPTimeout:
            # MCP超时，返回空结果
            return None
    
    async def call_core_async(self, algorithm: str, params: dict):
        """异步调用核心算法（实际是同步的，但包装为异步）"""
        return await asyncio.to_thread(
            self.core_algorithms.call,
            algorithm,
            params
        )
```

### 5.3 智能决策示例

```python
class SmartDecisionMaking:
    """智能决策示例 - 展示Agent如何协调两类组件"""
    
    async def handle_complex_query(self, query: str):
        """
        处理复杂查询，展示核心算法与MCP工具的协作
        """
        
        # 1. 核心算法：快速理解查询（5ms）
        intent = self.router.analyze_query(query)
        
        # 2. Agent决策：需要哪些能力？
        if intent["has_code"] and intent["has_table"]:
            # 复杂查询：需要多种MCP工具
            
            # Phase 1: 并行收集信息（MCP工具）
            code_results, table_results = await asyncio.gather(
                self.mcp_tools["tree_sitter"].analyze_code(),
                self.mcp_tools["pdf_parser"].extract_tables()
            )
            
            # Phase 2: 核心算法处理（本地快速）
            # 融合代码和表格结果
            fused = self.fusion_engine.fuse({
                "code": code_results,
                "table": table_results
            })
            
            # 唯一性评分
            uniqueness = self.uniqueness_scorer.calculate(fused)
            
            # Phase 3: 如果需要更多信息
            if uniqueness["confidence"] < 0.7:
                # 调用更多MCP工具
                additional = await self.mcp_tools["vector_db"].deep_search()
                # 再次融合
                fused = self.fusion_engine.fuse_incremental(fused, additional)
        
        else:
            # 简单查询：直接路由到合适的工具
            if intent["route"] == "exact_file":
                # 不需要复杂处理，直接用MCP工具
                results = await self.mcp_tools["file_system"].find_file()
            else:
                # 标准语义搜索流程
                results = await self.standard_semantic_search(query)
        
        return results
```

### 5.4 性能优化策略

```yaml
优化策略:
  核心算法优化:
    - 使用内存池减少分配
    - SIMD向量化计算
    - 多线程并行处理
    - JIT编译热点代码
    
  MCP调用优化:
    - 连接池复用
    - 批量请求合并
    - 超时快速失败
    - 结果缓存
    
  协调优化:
    - 预测性预加载
    - 投机性执行
    - 优先级调度
    - 资源隔离

性能目标:
  - 简单查询: <500ms (只调用1-2个MCP)
  - 中等查询: <1s (3-4个MCP + 核心算法)
  - 复杂查询: <2s (5+个MCP + 全部核心算法)
  
监控指标:
  - 核心算法耗时占比: <20%
  - MCP调用耗时占比: 60-70%
  - 网络传输耗时: <10%
  - Agent决策耗时: <5%
```

---

## 6. 实施路径

### Phase 1: 核心算法开发（Week 1）

```yaml
Day 1-2: 核心算法框架
  任务:
    - 实现QueryRouter（查询路由）
    - 实现EarlyStoppingEngine（早停引擎）
    - 搭建性能测试框架
  
  预期成果:
    - 查询分类<20ms
    - 早停判断<10ms
    - 单元测试覆盖率>90%

Day 3-4: 融合与排序算法
  任务:
    - 实现AdaptiveFusionEngine
    - 实现RankingEngine
    - 性能优化（向量化、并行）
  
  预期成果:
    - 1000文档融合<50ms
    - 100文档排序<30ms
    - 基准测试通过

Day 5: 算法集成测试
  任务:
    - 端到端算法流程测试
    - 性能瓶颈分析
    - 内存占用优化
  
  预期成果:
    - 完整查询流程<100ms（不含IO）
    - 内存占用<200MB
```

### Phase 2: MCP工具集成（Week 2）

```yaml
Day 6-7: 基础MCP工具接入
  任务:
    - 集成pdf-parser-mcp
    - 集成tree-sitter-mcp
    - 集成qdrant-mcp
    - 实现MCP客户端封装
  
  预期成果:
    - PDF解析功能可用
    - 代码分析功能可用
    - 向量检索功能可用

Day 8-9: Agent编排层
  任务:
    - 实现Agent决策逻辑
    - 实现执行协调器
    - 集成核心算法与MCP工具
  
  预期成果:
    - Agent可以制定执行计划
    - 核心算法与MCP工具协同工作
    - 基础查询端到端可用

Day 10: 系统集成测试
  任务:
    - 完整流程测试
    - 性能调优
    - 错误处理完善
  
  预期成果:
    - 简单查询<1s
    - 复杂查询<2s
    - 错误恢复机制完备
```

### Phase 3: 优化与扩展（Week 3-4）

```yaml
Week 3: 性能优化与缓存
  核心算法优化:
    - 实现查询缓存
    - 添加结果预取
    - 优化内存使用
  
  MCP工具优化:
    - 实现连接池
    - 添加批量请求
    - 优化超时处理
  
  目标:
    - P50延迟<500ms
    - P95延迟<1.5s

Week 4: 功能扩展与产品化
  功能扩展:
    - 添加更多MCP工具
    - 实现学习机制
    - 添加解释生成
  
  产品化:
    - Web UI开发
    - API文档
    - 部署脚本
  
  目标:
    - 支持10+种文档类型
    - UI响应流畅
    - 一键部署
```

### 开发优先级矩阵

```python
# 开发优先级定义
priority_matrix = {
    "P0 - 必须完成": [
        "QueryRouter",           # 核心算法
        "EarlyStoppingEngine",   # 核心算法
        "AdaptiveFusionEngine",  # 核心算法
        "pdf-parser-mcp集成",    # MCP工具
        "qdrant-mcp集成",       # MCP工具
        "基础Agent编排"
    ],
    
    "P1 - 应该完成": [
        "RankingEngine",         # 核心算法
        "UniquenessScorer",      # 核心算法
        "tree-sitter-mcp集成",   # MCP工具
        "缓存系统",
        "性能监控"
    ],
    
    "P2 - 可以延后": [
        "学习优化器",
        "ocr-mcp集成",
        "高级UI功能",
        "多语言支持"
    ]
}
```

### 技术债务管理

```yaml
避免的技术债务:
  - 不过早优化边缘场景
  - 不重复实现已有MCP功能
  - 不过度抽象简单逻辑
  
允许的技术债务:
  - Phase 1可以硬编码一些配置
  - 初期可以使用简单的缓存策略
  - UI可以从简单开始迭代
  
债务清理计划:
  Week 3: 重构硬编码配置
  Week 4: 优化缓存策略
  Month 2: UI体验提升
```

---

## 6. 技术实现细节

### 6.1 MCP工具开发模板

```python
# mcp_tool_template.py
from mcp import Tool, tool

class DocuRayMCPTool:
    """内部MCP工具开发模板"""
    
    @tool(
        name="tool_name",
        description="工具描述",
        parameters={
            "param1": {"type": "string", "description": "参数1"},
            "param2": {"type": "number", "description": "参数2"}
        }
    )
    async def execute(self, param1: str, param2: float):
        """
        MCP工具实现
        """
        try:
            # 核心逻辑
            result = self.core_algorithm(param1, param2)
            
            # 返回标准格式
            return {
                "success": True,
                "data": result,
                "metadata": {
                    "execution_time": self.elapsed_ms,
                    "confidence": self.confidence
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### 6.2 Agent提示词工程

```python
AGENT_SYSTEM_PROMPT = """
你是DocuRay搜索Agent，一个智能的文档搜索助手。

你的能力：
1. 理解用户的搜索意图
2. 选择和组合合适的MCP工具
3. 动态调整搜索策略
4. 解释搜索过程和结果

可用的MCP工具：
{available_tools}

决策原则：
- 优先使用最简单的工具组合
- 根据置信度决定是否继续搜索
- 平衡速度和准确性
- 为用户提供可解释的结果

当前查询：{query}
请制定执行计划。
"""
```

### 6.3 监控与日志

```python
class AgentMonitor:
    """Agent行为监控"""
    
    def log_decision(self, decision):
        """记录Agent决策"""
        self.decisions.append({
            "timestamp": datetime.now(),
            "decision": decision,
            "context": self.current_context,
            "confidence": self.current_confidence
        })
    
    def analyze_performance(self):
        """分析Agent性能"""
        return {
            "avg_tools_used": self.calculate_avg_tools(),
            "success_rate": self.calculate_success_rate(),
            "avg_latency": self.calculate_avg_latency(),
            "learning_progress": self.track_improvement()
        }
```

---

## 8. 创新点与差异化

### 8.1 架构创新：核心算法 + MCP生态

```
传统方案的问题:
├── 纯MCP方案：性能差，延迟高
├── 纯自研方案：开发慢，维护难
└── 纯Pipeline：灵活性差，难扩展

DocuRay的解决方案:
├── 核心算法固化：性能关键路径本地优化
├── MCP工具复用：专业功能直接集成
└── Agent智能编排：动态组合最优路径
```

### 8.2 性能优势对比

| 组件类型 | 纯MCP方案 | 纯自研方案 | DocuRay混合方案 |
|---------|-----------|------------|----------------|
| **查询路由** | 100ms（LLM调用） | 20ms | **5ms**（本地模型） |
| **早停判断** | 50ms（网络调用） | 10ms | **3ms**（内存计算） |
| **结果融合** | 100ms（序列化开销） | 50ms | **20ms**（向量化） |
| **PDF解析** | 500ms | 2000ms（自研） | **500ms**（复用MCP） |
| **代码分析** | 300ms | 1000ms（自研） | **300ms**（复用MCP） |
| **总延迟** | 1050ms | 3080ms | **828ms** |

### 8.3 开发效率优势

```python
# 开发成本对比
development_comparison = {
    "纯自研方案": {
        "PDF解析器": "2人月",
        "代码AST解析": "3人月",
        "向量数据库": "4人月",
        "OCR系统": "3人月",
        "总计": "12人月"
    },
    
    "DocuRay方案": {
        "核心算法": "2人月",  # 只开发差异化部分
        "MCP集成": "0.5人月",  # 集成现成工具
        "Agent编排": "1人月",
        "总计": "3.5人月"  # 节省70%时间
    }
}
```

### 8.4 核心竞争力

#### 1. **独特的算法创新**
```python
# DocuRay的核心算法优势
core_advantages = {
    "早停引擎": {
        "创新点": "多因素动态阈值",
        "效果": "减少50%无效搜索",
        "专利性": "可申请专利"
    },
    
    "融合算法": {
        "创新点": "查询感知的自适应权重",
        "效果": "准确率提升15%",
        "专利性": "商业机密"
    },
    
    "唯一性评分": {
        "创新点": "多维度置信度计算",
        "效果": "减少80%歧义结果",
        "专利性": "独家算法"
    }
}
```

#### 2. **灵活的扩展能力**
- 新增文档类型 = 接入新MCP工具（1小时）
- 新增检索能力 = 添加MCP服务（0成本）
- 算法升级 = 只更新核心模块（低风险）

#### 3. **智能的Agent编排**
- Agent理解意图，不只是匹配关键词
- 动态选择最优执行路径
- 从历史中学习最佳组合

### 8.5 技术护城河

```yaml
难以复制的优势:
  
  核心算法层:
    - 3年积累的查询路由模型
    - 专利保护的早停算法
    - 大规模数据训练的排序模型
  
  系统工程层:
    - 极致优化的性能（SIMD、并行）
    - 精细调优的缓存策略
    - 生产环境验证的稳定性
  
  生态集成层:
    - 10+个MCP工具的最佳实践
    - 工具组合的经验知识
    - 社区贡献的持续改进
```

### 8.6 商业价值

| 指标 | 传统搜索 | RAG方案 | DocuRay |
|------|----------|---------|---------|
| **开发成本** | 高（全自研） | 中（LLM依赖） | **低**（复用生态） |
| **运营成本** | 低 | 高（GPU） | **低**（CPU即可） |
| **搜索延迟** | 快（<500ms） | 慢（3-5s） | **快**（<1s） |
| **准确率** | 60% | 70% | **85%** |
| **可解释性** | 无 | 差（幻觉） | **好**（证据链） |
| **扩展性** | 差 | 中 | **优秀** |

---

## 8. 风险与对策

### 8.1 技术风险

| 风险 | 影响 | 对策 |
|------|------|------|
| LLM响应慢 | 搜索延迟增加 | 本地小模型 + 缓存 |
| MCP工具不稳定 | 功能失效 | 冗余工具 + 降级策略 |
| Agent决策错误 | 结果不准确 | 人工反馈 + 持续学习 |

### 8.2 缓解策略

```python
class FallbackStrategy:
    """降级策略"""
    
    def handle_tool_failure(self, failed_tool):
        # 工具失败时的备选方案
        fallback_map = {
            "qdrant-mcp": "chromadb-mcp",
            "pdf-parser-mcp": "simple-text-extraction",
            "tree-sitter-mcp": "regex-pattern-matching"
        }
        return fallback_map.get(failed_tool)
    
    def handle_agent_timeout(self):
        # Agent超时时退化到简单搜索
        return self.simple_keyword_search()
```

---

## 9. 成功指标

### 9.1 技术指标
- **MCP工具调用成功率** > 99%
- **Agent决策准确率** > 85%
- **端到端延迟** < 2秒（P95）
- **工具组合效率** > 80%（最优路径选择）

---

## 10. macOS 第一阶段架构设计（文档搜索）

本章节在现有四层Agent架构之上，面向macOS平台给出第一阶段“本地文档搜索”的落地方案，聚焦稳定、可用、可扩展与可产品化。Windows 平台放入第二阶段进行接口对齐与适配。

### 10.1 目标与范围（Phase-1）
- 支持文件类型：`pdf / txt / md / docx / pptx / xlsx`（优先级从左到右）
- 支持查询类型：
  - 文件名/路径精确查找
  - 关键词/短语匹配（布尔与短语支持）
  - 语义检索（向量召回 + 学习排序）
  - 自然语言查询（轻量NLU路由，非必须依赖外部LLM）
- 离线可用，所有索引与向量存储本地化；不依赖外网服务
- 性能目标（P95）：
  - 即时搜索交互 < 300ms（命中缓存/Spotlight）
  - 语义检索 < 800ms（Top-50重排后返回）
  - 首轮全量索引：≥ 30 文件/秒（SSD，文本/MD/PDF混合）

### 10.2 系统拓扑（macOS）

```yaml
User
  └─ UI (CLI/菜单栏)  
       └─ DocuRay Core (Agent编排层 + 核心算法层, 本地进程)
            ├─ Indexer Service（索引服务, 后台常驻 - LaunchAgent）
            ├─ Storage Layer（SQLite + 向量索引）
            └─ MCP Tools（macOS适配器）
                 ├─ spotlight-mcp（mdfind/mdls包装）
                 ├─ fsevents-mcp（文件变更监听）
                 ├─ pdfkit-mcp（PDFKit文本抽取）
                 ├─ textutil-mcp（Office/iWork 转纯文本）
                 ├─ quicklook-mcp（预览/缩略图，用于结果展示增强）
                 └─ file-system-mcp（遍历/权限/过滤）
```

说明：Phase-1 可先以 CLI + 后台索引服务交付；菜单栏 UI 随后补齐。核心算法与Agent编排保持与2.0总体架构一致，新增“平台适配型MCP工具”。

### 10.3 关键组件职责

- Agent Orchestration（不变）：
  - 路由：`QueryRouter` 基于规则/小模型决定走 Spotlight/关键词/向量检索路径
  - 编排：按查询意图组合 spotlight-mcp / vector-db / ranking 等调用
  - 早停：`EarlyStoppingEngine` 控制迭代阶段与耗时预算

- Core Algorithms（不变但参数调优）：
  - `AdaptiveFusionEngine`：融合 Spotlight 结果、倒排/关键词召回、向量召回
  - `RankingEngine`：轻量 Learning-to-Rank（可先基于规则 + 特征加权）

- Indexer Service（新增/强化）：
  - 初次全量扫描：用户选定根目录集合（默认：`~/Documents`, `~/Desktop`, 可配置）
  - 内容抽取：优先走系统能力（PDFKit/textutil）；失败时回退第三方解析器
  - 切片与去重：统一切片策略（按语义/结构），维护文档->切片->向量映射
  - 向量化：本地ONNX嵌入模型（多语种，推荐 `gte-multilingual` 或 `bge-m3` 量化版）
  - 存储：元数据/切片落地 SQLite；向量索引使用 FAISS/SQLite-vec/Annoy（Phase-1 任选其一）
  - 实时更新：通过 fsevents-mcp 监听变更，增量更新索引与向量

- Storage Layer：
  - `meta.db`（SQLite）：`files`, `chunks`, `inverted_index`, `tags`, `jobs`
  - `vectors`：FAISS Index 或 SQLite-vec 内嵌表
  - `cache/`：查询缓存、预计算结果、热度计数

### 10.4 数据与索引设计（最小可用）

```sql
-- files：文件粒度元数据
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  mime TEXT,
  size INTEGER,
  mtime INTEGER,
  title TEXT,
  lang TEXT,
  spotlight_tags TEXT,
  hash TEXT
);

-- chunks：内容切片（与向量一一对应）
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  file_id INTEGER,
  seq INTEGER,
  text TEXT,
  tokens INTEGER,
  embedding BLOB NULL,
  FOREIGN KEY(file_id) REFERENCES files(id)
);

-- inverted_index：关键词倒排（快速关键词/短语匹配）
CREATE VIRTUAL TABLE inverted_index USING fts5(
  text, content='chunks', content_rowid='id'
);
```

向量索引：
- 方案A（推荐起步）：`SQLite + sqlite-vec`（部署简洁，文件级分发）
- 方案B：`FAISS + SQLite`（性能更高，需打包依赖）

### 10.5 查询流程（macOS 优先路径）

```python
def search(query: str):
    # 1) 路由决策
    route = QueryRouter.analyze_query(query)

    # 2) 快路径（<300ms）
    if route == 'fast_file_search' or pattern_like_path(query):
        return spotlight_mcp.search_filename_or_path(query)

    # 3) 关键词/短语（<400ms）
    kw_hits = fts_index.search(query)             # inverted_index

    # 4) 语义召回（<500ms）
    emb = embed(query)
    vec_hits = vector_index.topk(emb, k=50)

    # 5) 融合 + 排序（<150ms）
    fused = AdaptiveFusionEngine.fuse({
        'lexical': kw_hits,
        'vector': vec_hits,
        'spotlight': spotlight_mcp.search_if_relevant(query)
    }, query_features)
    ranked = RankingEngine.rank(fused, query_features)

    # 6) 早停与解释
    if EarlyStoppingEngine.should_stop(...):
        return ranked[:N]
```

### 10.6 MCP 工具适配（macOS 专用）

- `spotlight-mcp`：
  - 调用 `mdfind` 做元数据/内容检索，`mdls` 取文件标签/作者/创建时间
  - 用于文件名/路径/标签/快速命中；在语义检索中作为辅助通道

- `fsevents-mcp`：
  - 基于 FSEvents 监听目录树变化，事件驱动增量更新（新增/修改/移动/删除）

- `pdfkit-mcp`：
  - 基于 PDFKit 抽取文本（优先），对复杂版式可降级 `pdftotext` 或内置OCR（可选）

- `textutil-mcp`：
  - 利用系统 `textutil` 将 `doc/docx/pptx/pages/key` 转纯文本，失败时回退第三方解析器

- `quicklook-mcp`：
  - 生成结果预览缩略图/摘要片段，提升交互体验；不参与排序

注：以上均作为“平台适配型MCP工具”，可在第二阶段以同名语义在 Windows 侧实现同等能力。

### 10.7 权限与安全（TCC）
- 首次运行引导用户授予 Full Disk Access 或选定授权目录
- 仅索引用户显式选择的目录；支持排除规则（`.git`, `node_modules`, `*.log` 等）
- 数据本地化：索引与向量存储路径 `~/Library/Application Support/DocuRay/`
- 可选透明度：提供“数据导出/清空索引”操作

### 10.8 交付计划（macOS）

```yaml
Week 1:
  - 完成 spotlight-mcp / fsevents-mcp / pdfkit-mcp / textutil-mcp 适配
  - 实现 Indexer Service（全量扫描 + 增量更新）
  - 落地 SQLite 元数据模型与 FTS 索引

Week 2:
  - 集成本地嵌入模型（ONNX/量化），完成向量召回
  - 实现融合与轻量排序，打通端到端搜索
  - CLI 交互与基础结果预览（quicklook-mcp）

Week 3:
  - 性能优化：批量嵌入、并行解析、磁盘IO合并
  - 质量保证：大规模真实数据集评估与回归
  - 可用性：最小可用菜单栏 UI（可选）
```

### 10.9 风险与对策（macOS 专项）
- Spotlight 索引滞后：
  - 对策：Indexer 自建倒排与向量索引为主，Spotlight 仅作辅助手段；对关键目录可触发 `mdimport` 刷新
- 权限/TCC 受限：
  - 对策：首启引导与权限检查；目录级授权兜底；敏感路径默认排除
- 嵌入模型体积/性能：
  - 对策：优先选用小型多语种 ONNX 模型 + INT8 量化；按需加载/内存映射
- PDF 抽取质量差异：
  - 对策：PDFKit→pdftotext→OCR 的逐级降级链路；建立失败重试与标记

### 10.10 指标（Phase-1）
- 端到端延迟（P95）：关键词 < 400ms；语义 < 800ms
- 首轮全量索引吞吐：≥ 30 文件/秒；增量更新延迟 < 2s
- 覆盖率：可解析文件占比 ≥ 95%（目标类型）
- 体验：首字母即时反馈（200ms 内）

### 10.11 Windows 第二阶段对齐（展望）
- 工具等价：
  - `spotlight-mcp` → `windows-search-mcp`（Windows Search API）
  - `fsevents-mcp` → `usn-journal-mcp`（NTFS USN/目录监控）
  - `pdfkit-mcp/textutil-mcp` → `iFilter/Office COM` 或第三方解析器
- 存储与算法层保持一致：SQLite + 向量索引 + 融合与排序
- UI 复用交互逻辑，适配 WinUX（托盘 + 全局快捷键）

---

以上 macOS 第一阶段架构与现有 Agent 四层模型完全兼容，通过「平台适配型MCP工具」确保跨平台一致语义与最小改动迁移，为第二阶段 Windows 适配打好基础。

### 9.2 业务指标
- **搜索成功率** > 90%
- **用户满意度** > 4.5/5
- **开发效率提升** > 300%（通过MCP复用）

### 9.3 创新指标
- **新工具接入时间** < 1小时
- **Agent学习提升率** > 5%/月
- **自主发现的新组合模式** > 10个/月

---

## 10. 开发计划

### 立即行动（Day 1）

```bash
# 1. 搭建基础框架
mkdir docuray-agent
cd docuray-agent

# 2. 初始化MCP环境
npm init mcp-project

# 3. 集成第一个外部MCP工具
mcp install pdf-parser-mcp

# 4. 创建第一个内部MCP工具
touch tools/early-stopping-mcp.py

# 5. 配置Agent
touch agent/orchestrator.py
```

### 第一周目标

```python
# 可运行的MVP
class DocuRayMVP:
    def __init__(self):
        self.agent = LLMAgent()
        self.tools = {
            "pdf": PDFParserMCP(),
            "vector": QdrantMCP(),
            "fusion": SimpleFusionMCP()
        }
    
    async def search(self, query):
        # Agent理解查询
        intent = await self.agent.understand(query)
        
        # Agent选择工具
        tools = await self.agent.select_tools(intent)
        
        # 执行搜索
        results = await self.agent.execute(tools, query)
        
        return results
```

---

## 11. 长期愿景

### 11.1 演进路线

```
v1.0 (2周): Agent + 基础MCP工具 → 可用的搜索
v2.0 (1月): + 学习机制 → 自适应搜索
v3.0 (3月): + 企业工具 → 企业级搜索
v4.0 (6月): + 多模态 → 全能搜索助手
```

### 11.2 生态建设

- **MCP工具市场**：贡献和分享DocuRay MCP工具
- **Agent模板**：分享成功的查询模式
- **社区驱动**：用户可以贡献自己的MCP工具

### 11.3 终极目标

> 让DocuRay成为"文档搜索领域的ChatGPT" - 不是通过复杂的算法，而是通过智能的工具组合和Agent编排，实现真正理解用户意图的搜索体验。

---

## 附录A：MCP工具接入清单

### 已确认可用的外部MCP工具

```yaml
PDF处理:
  - pdf.co-mcp: 综合PDF处理
  - pdf-reader-mcp: 简单PDF读取
  - mistral-ocr-mcp: 高质量OCR

代码分析:
  - tree-sitter-mcp: 完整的代码分析
  - code-analyzer-mcp: 简单代码搜索

向量数据库:
  - qdrant-mcp: 高性能向量搜索
  - chromadb-mcp: 轻量级向量库
  - lancedb-mcp: 多模态向量库

文件系统:
  - file-system-mcp: 文件操作
  - file-watcher-mcp: 文件监控
```

### 需要开发的内部MCP工具

```yaml
核心算法:
  - early-stopping-mcp: 早停机制
  - adaptive-fusion-mcp: 自适应融合
  - uniqueness-scorer-mcp: 唯一性评分
  - anchor-verifier-mcp: 锚点验证

优化工具:
  - cache-manager-mcp: 缓存管理
  - performance-monitor-mcp: 性能监控
  - learning-optimizer-mcp: 学习优化
```

---

## 附录B：Agent提示词模板库

### 查询理解提示词

```python
UNDERSTAND_QUERY = """
分析用户查询：{query}

请识别：
1. 查询类型（精确/语义/代码/表格/复合）
2. 关键实体（文件名/概念/时间/作者）
3. 约束条件（文件类型/时间范围/目录）
4. 期望结果（定位/列表/统计/解释）

输出JSON格式的分析结果。
"""
```

### 工具选择提示词

```python
SELECT_TOOLS = """
基于查询分析：{analysis}

可用MCP工具：{available_tools}

请选择最优的工具组合，考虑：
1. 查询类型与工具能力匹配
2. 执行效率（并行vs串行）
3. 结果质量要求
4. 备选方案

输出工具执行计划。
"""
```

### 结果优化提示词

```python
OPTIMIZE_RESULTS = """
当前结果：{results}
用户查询：{query}
执行路径：{execution_path}

请优化结果：
1. 去重和合并
2. 相关性重排
3. 证据增强
4. 可解释性改进

输出优化后的结果。
"""
```

---

## 结语

DocuRay 2.0 通过**核心算法固化 + MCP工具生态 + Agent智能编排**的混合架构，实现了性能、灵活性和开发效率的最佳平衡：

### 架构哲学

```
核心算法固化（Core Algorithms）
├── 性能关键路径：早停、融合、排序
├── 商业机密保护：独特算法不暴露
└── 极致优化：SIMD、并行、缓存

     +

MCP工具生态（MCP Tools）  
├── 专业功能复用：PDF、OCR、AST解析
├── 快速集成：1小时添加新能力
└── 社区驱动：持续改进

     +

Agent智能编排（Agent Orchestration）
├── 动态决策：理解意图，选择最优路径
├── 自主学习：从成功案例中学习
└── 可解释性：决策过程透明
     
     =
     
完美平衡的搜索引擎
```

### 关键成功因素

1. **明确的职责划分**
   - 性能关键 → 核心算法
   - 专业功能 → MCP工具
   - 智能决策 → Agent编排

2. **正确的技术选择**
   - 不过度MCP化（避免性能损失）
   - 不过度自研（避免重复造轮子）
   - 不过度依赖AI（保持可控性）

3. **渐进式实施**
   - Week 1: 核心算法（基础）
   - Week 2: MCP集成（扩展）
   - Week 3-4: 优化迭代（完善）

### 预期成果

**2周后的MVP**：
- ✅ 亚秒级搜索响应
- ✅ 支持5+种文档类型
- ✅ 85%+的准确率
- ✅ 可解释的结果

**3个月后的产品**：
- ✅ 支持20+种文档类型
- ✅ 90%+的准确率
- ✅ 自学习优化
- ✅ 企业级稳定性

**长期愿景**：
> 让DocuRay成为本地文档搜索的事实标准，通过开放的MCP生态和强大的核心算法，为用户提供"快、准、智"的搜索体验。

**核心理念**：
- **Build what matters**：只构建差异化的核心算法
- **Buy what exists**：复用成熟的MCP工具
- **Learn what works**：让Agent学习最佳实践

这不仅是一个搜索引擎，更是一个**可进化的智能系统**。

Let's build the future of intelligent document search! 🚀
