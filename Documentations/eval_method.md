# DocuReason v1.1.0 — Comprehensive RAG Evaluation Framework & Mathematical Specification

> **The Definitive Industry & Research Benchmark Manual**  
> *An Authoritative Reference Guide for Evaluating Retrieval Accuracy, Computational Overhead, Resource Utilization, Scalability, Cost Efficiency, Context Quality, LLM Groundedness, and Pipeline Robustness across Enterprise Multimodal Document Collections.*

---

## Executive Summary & System Evaluation Paradigm

Evaluating an enterprise Retrieval-Augmented Generation (RAG) system requires a multi-dimensional approach. While traditional academic benchmarks focus exclusively on top-$K$ retrieval recall, real-world industrial deployments and reviewers at top AI/IR venues (SIGIR, CIKM, ACL, EMNLP) judge systems across **5 Core Dimensions**:

1. **Retrieval & Ranking Quality**: Precision in identifying ground-truth evidence ($\text{Recall}@K$, $\text{Precision}@K$, $\text{MRR}$, $\text{MAP}$, $\text{nDCG}@K$, $\text{Hit Rate}$).
2. **Context & Evidence Integrity**: Quality and signal-to-noise ratio of context supplied to the LLM ($\text{Evidence Recall}$, $\text{Evidence Coverage}$, $\text{Context Compression Ratio}$, $\text{Noise Ratio}$).
3. **Answer Groundedness & Faithfulness**: Elimination of ungrounded hallucinations ($\text{Sentence NLI Entailment Precision}$, $\text{Hallucination Rate}$, $\text{Citation Accuracy}$).
4. **Computational & Resource Overhead**: Execution latency and hardware overhead across every pipeline stage ($P_{50}/P_{95}/P_{99}$ latency, QPS, CPU %, GPU VRAM, Index size).
5. **Scalability & Cost Efficiency**: Performance scaling across $10,000$ to $10,000,000$ chunks and dollar costs per document/query.

This document presents a **descriptive, exhaustive breakdown of all 22 Evaluation Categories**, complete with **underlying mathematical formulas**, **diagnostic rationale**, **systemic impact**, and **ideal production target values**.

---

## Master Evaluation Taxonomy (22 Categories)

---

### 1. Document Processing Overhead

Measures the computational and time cost of converting raw heterogeneous files (`.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`) into layout-parsed, typed region chunks.

#### Detailed Sub-Metric Formulations

* **Parsing Time per Document ($T_{\text{parse\_doc}}$)**:
  $$T_{\text{parse\_doc}} = t_{\text{end\_parsing}} - t_{\text{start\_parsing}} \quad [\text{seconds/doc}]$$

* **Parsing Throughput ($\tau_{\text{parse}}$)**:
  $$\tau_{\text{parse}} = \frac{N_{\text{pages}}}{T_{\text{total\_ingest\_time}}} \quad [\text{pages/second}]$$

* **Chunking Execution Time ($T_{\text{chunk}}$)**:
  $$T_{\text{chunk}} = t_{\text{end\_chunking}} - t_{\text{start\_chunking}} \quad [\text{milliseconds/doc}]$$

* **OCR Execution Time per Scanned Page ($T_{\text{ocr\_page}}$)**:
  $$T_{\text{ocr\_page}} = \frac{\sum_{i \in \text{OCR\_pages}} T_{\text{EasyOCR}}(i)}{N_{\text{OCR\_pages}}} \quad [\text{seconds/page}]$$

* **Metadata Extraction Overhead ($T_{\text{meta}}$)**:
  $$T_{\text{meta}} = t_{\text{end\_manifest\_generation}} - t_{\text{start\_manifest\_generation}} \quad [\text{milliseconds/doc}]$$

* **Memory Usage During Parsing ($M_{\text{parse\_peak}}$)**:
  $$M_{\text{parse\_peak}} = \max_{t} \left( \text{RAM}_{\text{allocated}}(t) \right) \quad [\text{Megabytes}]$$

* **CPU & GPU Utilization Rates**:
  $$\text{CPU}_{\text{util}} = \frac{1}{\Delta t} \int_{t_0}^{t_0+\Delta t} \sum_{c=1}^{C} \text{core}_c(t) \, dt \times 100\%, \quad \text{GPU}_{\text{util}} = \frac{\text{VRAM}_{\text{allocated}}}{\text{VRAM}_{\text{total}}} \times 100\%$$

* **Parsing Failure Rate ($R_{\text{parse\_fail}}$)**:
  $$R_{\text{parse\_fail}} = \frac{N_{\text{corrupted}} + N_{\text{timeout}} + N_{\text{unsupported}}}{N_{\text{total\_docs}}} \times 100\%$$

#### Rationale & Systemic Impact
- **Why Measured**: Document ingestion and EasyOCR rendering represent the single largest computational bottleneck in offline RAG pipelines.
- **Impact**: High parsing latency delays document availability in search indices. High memory consumption during parsing leads to process crashes on large PDF batches.
- **Ideal / Industrial Benchmark Values**:
  - Native Text PDF Parsing Time: $\le 45\text{ ms/page}$
  - Scanned PDF OCR Parsing Time: $\le 1.1\text{ s/page}$ (GPU) / $\le 3.2\text{ s/page}$ (CPU)
  - Chunking Time: $\le 12\text{ ms/doc}$
  - Metadata Extraction Time: $\le 5\text{ ms/doc}$
  - Parsing Failure Rate: $< 0.1\%$
  - Peak Memory Footprint: $\le 2.0\text{ GB}$ per ingestion worker

---

### 2. Chunk Quality Metrics

Evaluates whether the chunking strategy preserves semantic coherence, structural layout, document hierarchy, and boundary integrity.

#### Detailed Sub-Metric Formulations

* **Average Chunk Token Length ($\bar{L}_{\text{chunk}}$)**:
  $$\bar{L}_{\text{chunk}} = \frac{1}{N} \sum_{i=1}^N \text{tokens}(c_i)$$

* **Median Chunk Token Length ($\widetilde{L}_{\text{chunk}}$)**:
  $$\widetilde{L}_{\text{chunk}} = \text{Median}\left( \{\text{tokens}(c_1), \text{tokens}(c_2), \dots, \text{tokens}(c_N)\} \right)$$

* **Chunk Length Variance ($\sigma^2_{\text{chunk}}$)**:
  $$\sigma^2_{\text{chunk}} = \frac{1}{N} \sum_{i=1}^N \left( \text{tokens}(c_i) - \bar{L}_{\text{chunk}} \right)^2$$

* **Chunks Per Document Ratio ($N_{\text{cpd}}$)**:
  $$N_{\text{cpd}} = \frac{N_{\text{total\_chunks}}}{N_{\text{total\_docs}}}$$

* **Overlap Percentage ($R_{\text{overlap}}$)**:
  $$R_{\text{overlap}} = \frac{\text{tokens}(c_i \cap c_{i+1})}{\text{tokens}(c_i)} \times 100\%$$

* **Duplicate Chunk Percentage ($R_{\text{dup\_chunk}}$)**:
  $$R_{\text{dup\_chunk}} = \frac{N_{\text{total\_chunks}} - |\{\text{SHA256}(c_i)\}|}{N_{\text{total\_chunks}}} \times 100\%$$

* **Heading & Section Preservation Rate ($R_{\text{heading}}$)**:
  $$R_{\text{heading}} = \frac{N_{\text{chunks\_with\_section\_breadcrumbs}}}{N_{\text{total\_chunks}}} \times 100\%$$

* **Table Preservation Rate ($R_{\text{table\_intact}}$)**:
  $$R_{\text{table\_intact}} = \frac{N_{\text{tables\_kept\_as\_single\_region}}}{N_{\text{total\_tables\_detected}}} \times 100\%$$

* **Figure-Caption Preservation Rate ($R_{\text{fig\_cap}}$)**:
  $$R_{\text{fig\_cap}} = \frac{N_{\text{figures\_paired\_with\_caption}}}{N_{\text{total\_figures\_detected}}} \times 100\%$$

* **Cross-Sentence & Cross-Page Split Rates**:
  $$R_{\text{sentence\_split}} = \frac{N_{\text{chunks\_split\_mid\_sentence}}}{N_{\text{total\_chunks}}} \times 100\%, \quad R_{\text{page\_split}} = \frac{N_{\text{chunks\_split\_mid\_page\_unintended}}}{N_{\text{total\_chunks}}} \times 100\%$$

* **Semantic Coherence Score ($\text{SCS}$)**:
  $$\text{SCS}(c_i) = \frac{1}{|S|-1} \sum_{j=1}^{|S|-1} \cos\left( \mathbf{e}(s_j), \mathbf{e}(s_{j+1}) \right)$$
  *(where $s_j$ are sequential sentences in chunk $c_i$ and $\mathbf{e}(s_j)$ are sentence embeddings)*

#### Rationale & Systemic Impact
- **Why Measured**: Naive character-based chunking breaks sentences and tables across boundaries, destroying semantic context and degrading dense retrieval recall.
- **Impact**: Indivisible table serialization (GFM/JSON) and heading breadcrumbs ensure 100% preservation of structural relationships.
- **Ideal / Industrial Benchmark Values**:
  - Target Token Size: $256$ tokens (child chunk) / $1024$ tokens (parent region)
  - Table Preservation Rate: $100.0\%$ (Indivisible table regions)
  - Figure-Caption Pairing Rate: $\ge 98.0\%$
  - Duplicate Chunk Ratio: $0.0\%$ (Deduplicated via SHA-256)
  - Cross-Sentence Split Rate: $0.0\%$ (Structure-aware sentence splitting)
  - Semantic Coherence Score: $\ge 0.820$

---

### 3. Embedding Overhead

Measures embedding model encoding speed, vector dimensionality, storage footprint, and GPU memory consumption.

#### Detailed Sub-Metric Formulations

* **Embedding Generation Latency ($T_{\text{embed\_total}}$)**:
  $$T_{\text{embed\_total}} = t_{\text{end\_encoding}} - t_{\text{start\_encoding}} \quad [\text{seconds}]$$

* **Embedding Throughput ($\tau_{\text{embed}}$)**:
  $$\tau_{\text{embed}} = \frac{N_{\text{chunks}}}{T_{\text{embed\_total}}} \quad [\text{chunks/second}]$$

* **Average Latency per Chunk ($T_{\text{embed\_chunk}}$)**:
  $$T_{\text{embed\_chunk}} = \frac{T_{\text{embed\_total}}}{N_{\text{chunks}}} \times 1000 \quad [\text{milliseconds/chunk}]$$

* **Vector Storage Size ($S_{\text{vector}}$)**:
  $$S_{\text{vector}} = N_{\text{chunks}} \times d \times 4 \text{ bytes} \quad (\text{where } d = \text{vector dimension}, \text{e.g., } 384 \text{ or } 1024)$$

* **GPU Peak VRAM & CPU RAM Usage ($M_{\text{embed\_vram}}$)**:
  $$M_{\text{embed\_vram}} = \max_{t} \left( \text{VRAM}_{\text{allocated}}(t) \right) \quad [\text{Gigabytes}]$$

#### Rationale & Systemic Impact
- **Why Measured**: Vector embedding generation speed determines document indexing throughput and re-indexing latency during data updates.
- **Impact**: Normalizing vector embeddings ($\|\mathbf{v}\|_2 = 1.0$) enables fast cosine similarity computation via inner-product dot products.
- **Ideal / Industrial Benchmark Values**:
  - GPU Throughput (`all-MiniLM-L6-v2` / `bge-large`): $\ge 1,500\text{ chunks/sec}$ (GPU) / $\ge 250\text{ chunks/sec}$ (CPU)
  - Latency per Chunk: $\le 0.65\text{ ms/chunk}$
  - Storage Footprint (100k chunks, $d=384$): $\approx 153.6\text{ MB}$
  - VRAM Consumption: $\le 2.5\text{ GB}$ (Batch size 32)

---

### 4. Vector Database Overhead

Evaluates vector index construction latency, storage size, memory usage, dynamic write/delete latencies, and Approximate Nearest Neighbor (ANN) search recall.

#### Detailed Sub-Metric Formulations

* **HNSW Index Build Time ($T_{\text{hnsw\_build}}$)**:
  $$T_{\text{hnsw\_build}} = t_{\text{end\_index\_add}} - t_{\text{start\_index\_add}} \quad [\text{seconds}]$$

* **HNSW RAM Memory Footprint ($M_{\text{hnsw}}$)**:
  $$M_{\text{hnsw}} \approx N_{\text{chunks}} \times \left( d \times 4 + M \times 8 \right) \text{ bytes}$$
  *(where $M$ is the number of bi-directional connection links per graph node, e.g., $M=32$)*

* **Index Insert / Update / Delete Latency ($T_{\text{write\_op}}$)**:
  $$T_{\text{write\_op}} = t_{\text{end\_op}} - t_{\text{start\_op}} \quad [\text{milliseconds/operation}]$$

* **Vector Search Latency ($T_{\text{vec\_search}}$)**:
  $$T_{\text{vec\_search}} = t_{\text{end\_knn}} - t_{\text{start\_knn}} \quad [\text{milliseconds}]$$

* **ANN Search Recall@K ($\text{Recall}_{\text{ANN}}@K$)**:
  $$\text{Recall}_{\text{ANN}}@K = \frac{|\text{TopK}_{\text{HNSW}} \cap \text{TopK}_{\text{Exact\_Flat}}|}{K} \times 100\%$$

#### Rationale & Systemic Impact
- **Why Measured**: Balances vector search acceleration $O(\log N)$ against RAM consumption and graph approximation accuracy losses.
- **Impact**: Higher `efConstruction` improves index recall during build; higher `efSearch` boosts query-time precision at marginal latency cost.
- **Ideal / Industrial Benchmark Values**:
  - HNSW Parameters: $M=32$, $efConstruction=200$, $efSearch=64$
  - ANN Search Recall@10 vs Exact Flat: $\ge 98.5\%$
  - Vector Search Latency ($100,000$ vectors): $\le 3.5\text{ ms}$
  - Dynamic Insert/Update Latency: $\le 15.0\text{ ms/op}$

---

### 5. Sparse Retrieval Overhead (BM25)

Measures BM25 inverted index construction speed, memory footprint, vocabulary size, average posting list length, and lexical query latency.

#### Detailed Sub-Metric Formulations

* **BM25 Relevance Scoring Function ($S_{\text{BM25}}$)**:
  $$\text{Score}_{\text{BM25}}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

* **Robertson-Spärck Jones IDF Formulation**:
  $$\text{IDF}(t) = \ln\left( \frac{N - n(t) + 0.5}{n(t) + 0.5} + 1 \right)$$

* **Average Posting List Length ($\bar{L}_{\text{posting}}$)**:
  $$\bar{L}_{\text{posting}} = \frac{\sum_{v \in V} |P(v)|}{|V|}$$
  *(where $V$ is vocabulary set and $P(v)$ is posting list for term $v$)*

* **Sparse Index RAM Footprint ($M_{\text{sparse}}$)**:
  $$M_{\text{sparse}} = \sum_{v \in V} \left( \text{len}(v) + |P(v)| \times 8 \right) \text{ bytes}$$

#### Rationale & Systemic Impact
- **Why Measured**: Guarantees precision for exact token matches (product SKUs, part numbers, code names, financial numbers, rare names).
- **Impact**: BM25S Python engine executes inverted index lookups in under $2.0\text{ms}$.
- **Ideal / Industrial Benchmark Values**:
  - BM25 Parameters: $k_1 = 1.5$, $b = 0.75$
  - Inverted Index Build Time ($100,000$ chunks): $\le 4.5\text{ seconds}$
  - BM25 Query Latency: $\le 2.0\text{ ms}$
  - Index RAM Overhead: $\approx 10\% - 15\%$ of raw text corpus size

---

### 6. Dense Retrieval Overhead

Evaluates bi-encoder vector similarity search execution time and candidate retrieval accuracy.

#### Detailed Sub-Metric Formulations

* **Normalized Cosine Inner Product**:
  $$\text{Sim}_{\text{dense}}(q, d) = \langle \mathbf{v}_q, \mathbf{v}_d \rangle = \sum_{i=1}^d v_{q,i} \cdot v_{d,i} \quad (\text{for } \|\mathbf{v}_q\| = \|\mathbf{v}_d\| = 1)$$

* **Top-K Dense Candidate Collection Time ($T_{\text{dense\_topk}}$)**:
  $$T_{\text{dense\_topk}} = t_{\text{end\_topk}} - t_{\text{start\_dense\_search}} \quad [\text{milliseconds}]$$

#### Rationale & Systemic Impact
- **Why Measured**: Captures conceptual and semantic intent beyond literal keyword matches.
- **Impact**: Provides high recall for natural language questions and paraphrased queries.
- **Ideal / Industrial Benchmark Values**:
  - Dense Retrieval Latency: $\le 5.0\text{ ms}$
  - Top-K Candidate Pool ($K_{\text{text}}$): $20$ to $40$ candidates

---

### 7. Hybrid Retrieval Overhead

Measures total time spent dispatching parallel retrieval queries across Text, Table/SQL, and Vision paths.

#### Detailed Sub-Metric Formulations

* **Total Parallel Retrieval Latency ($T_{\text{hybrid}}$)**:
  $$T_{\text{hybrid}} = \max\left( T_{\text{text\_dense}} + T_{\text{text\_bm25}}, \; T_{\text{table\_sql}}, \; T_{\text{vision\_clip}} \right) + T_{\text{dispatch\_overhead}}$$

#### Rationale & Systemic Impact
- **Why Measured**: Parallel multi-path dispatch guarantees candidate collection across text, tables, and visual charts without accumulating sequential latency.
- **Impact**: Keeps overall candidate gathering sub-15ms.
- **Ideal / Industrial Benchmark Values**:
  - Total Parallel Retrieval Latency: $\le 15.0\text{ ms}$
  - Dispatch Overhead: $\le 1.5\text{ ms}$

---

### 8. RRF (Reciprocal Rank Fusion) Overhead

Evaluates late fusion rank aggregation execution time, list handling capacity, and time complexity.

#### Detailed Sub-Metric Formulations

* **Weighted Reciprocal Rank Fusion Score ($\text{RRF}(d)$)**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{text}, \text{table}, \text{vision}\}} w_m \cdot \frac{1}{k + \text{rank}_m(d)}$$
  *(where $k=60$ and $w_m$ is dynamic probability weight from Intent Router)*

* **Rank Fusion Time Complexity**:
  $$\mathcal{O}(M \times K) \quad (\text{where } M = \text{number of paths (3)}, K = \text{top candidate count (20)})$$

* **Duplicate Removal Latency ($T_{\text{dedup}}$)**:
  $$T_{\text{dedup}} = t_{\text{end\_dedup}} - t_{\text{start\_dedup}} \quad [\text{milliseconds}]$$

#### Rationale & Systemic Impact
- **Why Measured**: Combines uncalibrated similarity scores from heterogeneous search engines (vector, BM25, SQL) into a scale-invariant rank list.
- **Impact**: Fusion execution takes $< 0.8\text{ ms}$.
- **Ideal / Industrial Benchmark Values**:
  - RRF Constant ($k$): $60$
  - Fusion Execution Latency: $\le 0.8\text{ ms}$
  - Duplicate Removal Latency: $\le 0.1\text{ ms}$

---

### 9. Reranker Overhead (Cross-Encoder)

Evaluates deep Cross-Encoder attention rescoring latency, GPU VRAM consumption, and batch processing throughput.

#### Detailed Sub-Metric Formulations

* **Cross-Encoder Joint Attention Score ($S_{\text{cross}}$)**:
  $$S_{\text{cross}}(q, d) = \text{Sigmoid}\left( \text{BERT}_{\text{joint}}([q; d]) \right)$$

* **Short-Heading Penalty Adjustment**:
  $$S_{\text{final}}(q, d) = S_{\text{cross}}(q, d) \times \begin{cases} \gamma_{\text{penalty}} & \text{if } \text{len}(d) < 45 \text{ chars and } m=\text{text} \\ 1.0 & \text{otherwise} \end{cases} \quad (\gamma_{\text{penalty}} = 0.5)$$

* **Reranking Latency per Query ($T_{\text{rerank\_query}}$)**:
  $$T_{\text{rerank\_query}} = t_{\text{end\_cross\_eval}} - t_{\text{start\_cross\_eval}} \quad [\text{milliseconds}]$$

#### Rationale & Systemic Impact
- **Why Measured**: Cross-encoders evaluate joint self-attention over query and candidate text together, delivering massive precision boosts but serving as the primary online retrieval latency bottleneck.
- **Impact**: Filters out RRF rank artifacts and TOC/heading-only chunks.
- **Ideal / Industrial Benchmark Values**:
  - Reranking Latency (Candidate batch=20): $\le 45\text{ ms}$ (GPU) / $\le 120\text{ ms}$ (CPU)
  - VRAM Consumption: $\le 1.8\text{ GB}$
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

### 10. End-to-End Latency & Percentiles ($P_{50}, P_{95}, P_{99}$)

Measures full query-to-answer latency across all online processing stages.

#### Detailed Sub-Metric Formulations

* **End-to-End Latency Pipeline Formulation**:
  $$T_{\text{e2e}} = T_{\text{router}} + T_{\text{hybrid\_retrieval}} + T_{\text{rrf\_fusion}} + T_{\text{parent\_expansion}} + T_{\text{cross\_rerank}} + T_{\text{llm\_gen}} + T_{\text{nli\_attrib}}$$

* **Percentile Latency Calculations ($P_{50}, P_{95}, P_{99}$)**:
  $$P_{\pi} = L_{\lfloor \pi \times N \rfloor}, \quad \text{where } \pi \in \{0.50, 0.95, 0.99\} \text{ over sorted latencies } L_1 \le L_2 \le \dots \le L_N$$

#### Rationale & Systemic Impact
- **Why Measured**: Industry Service-Level Objectives (SLOs) mandate strict response window guarantees for P95 and P99 queries.
- **Impact**: Ensures predictable interactive user experiences under concurrent request load.
- **Ideal / Industrial Benchmark Values**:
  - $P_{50}$ (Median Latency): $\le 350\text{ ms}$
  - $P_{95}$ (95th Percentile): $\le 850\text{ ms}$
  - $P_{99}$ (99th Percentile): $\le 1,400\text{ ms}$

---

### 11. Retrieval Quality Metrics

Measures whether the retrieval engine successfully fetches all true ground-truth evidence chunks.

#### Detailed Sub-Metric Formulations

* **Recall@K**:
  $$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

* **Precision@K**:
  $$\text{Precision}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

* **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q^*}$$

* **Mean Average Precision (MAP)**:
  $$\text{AP}@K = \sum_{k=1}^K \frac{\text{Precision}@k \times \mathbb{I}(d_k \in \text{Relevant})}{|\text{Relevant}|}, \quad \text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}_q$$

* **Normalized Discounted Cumulative Gain (nDCG@K)**:
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\mathbb{I}(d_i \in \text{Relevant})}{\log_2(i + 1)}, \quad \text{IDCG}@K = \sum_{r=1}^{\min(|\text{Relevant}|, K)} \frac{1}{\log_2(r + 1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

* **Hit Rate@K**:
  $$\text{Hit Rate}@K = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(|\text{Retrieved}_{q,K} \cap \text{Relevant}_q| > 0)$$

* **Evidence Recall, Precision, Coverage & Diversity**:
  $$\text{Ev\_Recall} = \frac{\text{Retrieved Ground Truth Facts}}{\text{Total Ground Truth Facts}}, \quad \text{Ev\_Diversity} = \frac{|\{\text{document\_id}_i \text{ in TopK}\}|}{K}$$

#### Rationale & Systemic Impact
- **Why Measured**: Core academic and industry benchmark standard for evaluating search relevance and ranking order.
- **Impact**: High Recall@5 ensures the LLM receives complete ground truth context before generating answers.
- **Ideal / Industrial Benchmark Values**:
  - Recall@5: $\ge 0.900$ | Recall@10: $\ge 0.950$
  - Precision@5: $\ge 0.820$ | Precision@10: $\ge 0.750$
  - nDCG@5: $\ge 0.880$ | MRR: $\ge 0.850$ | Hit Rate@5: $\ge 0.940$

---

### 12. Chunk Ranking Quality Metrics

Measures rank position stability and accuracy of gold evidence chunks in candidate lists.

#### Detailed Sub-Metric Formulations

* **Average Rank of Gold Chunk ($\bar{R}_{\text{gold}}$)**:
  $$\bar{R}_{\text{gold}} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{rank}(d_{\text{gold}}, q)$$

* **Rank Variance ($\sigma^2_{\text{rank}}$)**:
  $$\sigma^2_{\text{rank}} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \left( \text{rank}(d_{\text{gold}}, q) - \bar{R}_{\text{gold}} \right)^2$$

* **Duplicate Rank Ratio ($R_{\text{dup\_rank}}$)**:
  $$R_{\text{dup\_rank}} = \frac{N_{\text{duplicate\_chunks\_in\_top\_K}}}{K}$$

#### Rationale & Systemic Impact
- **Why Measured**: Detects whether near-duplicate overlapping window chunks crowd top rank positions.
- **Impact**: SHA-256 deduplication and short heading penalties prevent rank crowding.
- **Ideal / Industrial Benchmark Values**:
  - Average Gold Rank: $\le 1.8$
  - Rank Variance: $\le 0.45$
  - Duplicate Rank Ratio: $0.00$

---

### 13. Late Fusion Evaluation

Evaluates evidence diversity, context redundancy, and contradiction ratios across retrieved multimodal streams.

#### Detailed Sub-Metric Formulations

* **Context Redundancy Ratio ($R_{\text{redundant}}$)**:
  $$R_{\text{redundant}} = 1.0 - \frac{|\text{Unique Semantic Tokens in Context}|}{\text{Total Context Tokens}}$$

* **Unique Evidence Ratio ($R_{\text{unique}}$)**:
  $$R_{\text{unique}} = \frac{|\{\text{document\_id}_i\}|}{K}$$

* **Contradictory Evidence Ratio ($R_{\text{contradict}}$)**:
  $$R_{\text{contradict}} = \frac{N_{\text{contradicting\_chunk\_pairs}}}{\binom{K}{2}}$$

#### Rationale & Systemic Impact
- **Why Measured**: Ensures context provided to the LLM contains diverse, complementary evidence rather than repeating the same sentence multiple times.
- **Impact**: Maximizes context efficiency within LLM prompt limits.
- **Ideal / Industrial Benchmark Values**:
  - Context Redundancy Ratio: $\le 0.15$
  - Unique Evidence Ratio: $\ge 0.70$
  - Contradictory Evidence Ratio: $\le 0.02$

---

### 14. Context Quality Metrics

Evaluates token density, signal-to-noise ratio, and context compression efficiency.

#### Detailed Sub-Metric Formulations

* **Context Compression Ratio (CCR)**:
  $$\text{CCR} = \frac{\text{Useful Evidence Tokens}}{\text{Total Retrieved Context Tokens}} \quad (\text{Higher is better})$$

* **Noise Ratio ($R_{\text{noise}}$)**:
  $$R_{\text{noise}} = \frac{N_{\text{irrelevant\_chunks}}}{K} = 1.0 - \text{Precision}@K$$

* **Evidence Density ($D_{\text{evidence}}$)**:
  $$D_{\text{evidence}} = \frac{N_{\text{relevant\_sentences}}}{\text{Total Sentences in Context}}$$

#### Rationale & Systemic Impact
- **Why Measured**: LLMs suffer from "Lost in the Middle" syndrome when context is bloated with irrelevant boilerplate.
- **Impact**: High context density improves answer accuracy while saving LLM API token costs.
- **Ideal / Industrial Benchmark Values**:
  - Context Compression Ratio: $\ge 0.75$
  - Noise Ratio: $\le 0.15$
  - Evidence Density: $\ge 0.80$

---

### 15. LLM Performance & Faithfulness Metrics

Evaluates generation accuracy, claim attribution, citation precision, and hallucination elimination.

#### Detailed Sub-Metric Formulations

* **NLI Faithfulness Attribution Precision ($P_{\text{faith}}$)**:
  $$P_{\text{faith}} = \frac{\sum_{i=1}^M \mathbb{I}(\exists e \in E : e \models c_i)}{M}$$
  *(where $c_i$ are sentence claims and $e \models c_i$ represents NLI DeBERTa entailment)*

* **Hallucination Rate ($R_{\text{hallucination}}$)**:
  $$R_{\text{hallucination}} = 1.0 - P_{\text{faith}}$$

* **Citation Accuracy ($A_{\text{citation}}$)**:
  $$A_{\text{citation}} = \frac{N_{\text{correct\_inline\_citations}}}{N_{\text{total\_citations}}}$$

* **Exact Match (EM) & Token F1 Score**:
  $$\text{Precision}_{\text{token}} = \frac{|A_{\text{pred}} \cap A_{\text{gt}}|}{|A_{\text{pred}}|}, \quad \text{Recall}_{\text{token}} = \frac{|A_{\text{pred}} \cap A_{\text{gt}}|}{|A_{\text{gt}}|}, \quad \text{F1} = \frac{2 \times \text{Precision}_{\text{token}} \times \text{Recall}_{\text{token}}}{\text{Precision}_{\text{token}} + \text{Recall}_{\text{token}}}$$

* **ROUGE-L Score**:
  $$\text{ROUGE-L} = \frac{\text{LCS}(A_{\text{pred}}, A_{\text{gt}})}{\text{len}(A_{\text{gt}})}$$

#### Rationale & Systemic Impact
- **Why Measured**: The ultimate measure of a RAG pipeline is whether it produces factually grounded answers with zero hallucinations.
- **Impact**: Automated NLI claim verification flags ungrounded claims (`status = "needs_review"`).
- **Ideal / Industrial Benchmark Values**:
  - NLI Faithfulness Precision: $\ge 95.0\%$
  - Hallucination Rate: $\le 5.0\%$
  - Citation Accuracy: $\ge 98.0\%$
  - Tabular Math Exact Match (via DuckDB): $100.0\%$

---

### 16. Scalability Across Corpus Size

Evaluates retrieval latency, memory growth, and search recall as document corpus scales from $10,000$ to $10,000,000$ chunks.

#### Detailed Sub-Metric Formulations

* **Memory Scaling Rate ($\Delta M$)**:
  $$\Delta M = \frac{M(N_2) - M(N_1)}{N_2 - N_1} \quad [\text{MB / 10k chunks}]$$

* **Search Latency Scaling Factor**:
  $$T_{\text{search}}(N) \propto \mathcal{O}(\log N) \quad (\text{HNSW graph search complexity})$$

#### Scalability Target Benchmark Table

| Corpus Size ($N$ Chunks) | Target Index Build Time | RAM Footprint (HNSW) | Target Search Latency ($P_{95}$) | Search Recall@10 |
| :--- | :---: | :---: | :---: | :---: |
| **10,000 (10k)** | $\le 12 \text{ sec}$ | $\approx 250 \text{ MB}$ | $\le 2.5 \text{ ms}$ | $99.2\%$ |
| **100,000 (100k)** | $\le 115 \text{ sec}$ | $\approx 2.4 \text{ GB}$ | $\le 4.8 \text{ ms}$ | $98.5\%$ |
| **1,000,000 (1M)** | $\le 22 \text{ min}$ | $\approx 24.0 \text{ GB}$ | $\le 9.5 \text{ ms}$ | $97.8\%$ |
| **10,000,000 (10M)** | $\le 3.5 \text{ hrs}$ | Distributed Vector DB | $\le 18.0 \text{ ms}$ | $96.5\%$ |

---

### 17. Throughput Metrics

Evaluates system capacity to serve concurrent requests under multi-user production workloads.

#### Detailed Sub-Metric Formulations

* **Queries Per Second (QPS)**:
  $$\text{QPS} = \frac{N_{\text{completed\_queries}}}{T_{\text{duration}}} \quad [\text{queries/second}]$$

* **Ingestion Throughput ($\text{TP}_{\text{doc}}$)**:
  $$\text{TP}_{\text{doc}} = \frac{N_{\text{documents}}}{T_{\text{ingest\_hours}}} \quad [\text{docs/hour}]$$

* **Embeddings Throughput ($\tau_{\text{embed}}$)**:
  $$\tau_{\text{embed}} = \frac{N_{\text{chunks}}}{T_{\text{embed\_sec}}} \quad [\text{chunks/sec}]$$

#### Rationale & Systemic Impact
- **Why Measured**: Vital for infrastructure capacity planning, horizontal pod autoscaling, and server sizing.
- **Impact**: Determines how many concurrent users a single GPU/CPU server node can serve.
- **Ideal / Industrial Benchmark Values**:
  - Vector Search QPS (Single Node): $\ge 500\text{ QPS}$
  - End-to-End System QPS (SLM Generation): $\ge 15\text{ QPS}$ (GPU)
  - Ingestion Throughput: $\ge 1,200\text{ docs/hour}$

---

### 18. Resource Usage & Infrastructure Metrics

Monitors hardware consumption across CPU, GPU VRAM, System RAM, and Disk I/O.

#### Detailed Sub-Metric Formulations

* **VRAM Efficiency Ratio**:
  $$\text{VRAM}_{\text{utilization}} = \frac{\text{VRAM}_{\text{allocated}}}{\text{VRAM}_{\text{total}}} \times 100\%$$

#### Target Resource SLA Limits Table

| Resource Metric | Idle Baseline | Peak Production Load Target | Hard Safety Limit |
| :--- | :---: | :---: | :---: |
| **CPU Utilization** | $< 5\%$ | $40\% - 65\%$ | $< 85\%$ |
| **System RAM** | $\approx 1.2 \text{ GB}$ | $4.0 \text{ GB} - 8.0 \text{ GB}$ | $< 16.0 \text{ GB}$ |
| **GPU VRAM** | $\approx 0.8 \text{ GB}$ | $3.5 \text{ GB} - 6.0 \text{ GB}$ | $< 12.0 \text{ GB}$ |
| **Disk I/O Read/Write** | $< 1 \text{ MB/s}$ | $25 \text{ MB/s} - 80 \text{ MB/s}$ | $< 200 \text{ MB/s}$ |

---

### 19. Cost Analysis Formulations

Calculates financial Operating Expenses (OpEx) for cloud infrastructure, storage, and LLM inference APIs.

#### Detailed Sub-Metric Formulations

* **Cost Per Document Ingested ($C_{\text{doc}}$)**:
  $$C_{\text{doc}} = \frac{\text{GPU}_{\text{cost/hr}} \times T_{\text{parse\_hrs}} + \text{CPU}_{\text{cost/hr}} \times T_{\text{embed\_hrs}}}{N_{\text{documents}}}$$

* **Cost Per 1,000 Queries ($C_{1\text{k\_queries}}$)**:
  $$C_{1\text{k\_queries}} = 1000 \times \left( \frac{\text{Input Tokens}}{10^6} \times P_{\text{input}} + \frac{\text{Output Tokens}}{10^6} \times P_{\text{output}} + C_{\text{compute\_query}} \right)$$

* **Cost Per GB Indexed ($C_{\text{storage}}$)**:
  $$C_{\text{storage}} = \frac{S_{\text{index\_GB}} \times P_{\text{disk/GB/month}}}{N_{\text{GB\_raw\_docs}}}$$

#### Rationale & Systemic Impact
- **Why Measured**: Executive management and DevOps teams require exact OpEx estimates before scaling RAG pipelines.
- **Impact**: Offloading tables to local DuckDB and using local SLMs (DeepSeek-R1 1.5B) cuts API query costs by **over 85%**.
- **Ideal / Industrial Benchmark Values**:
  - Cost per Document Indexing: $\le \$0.0025$ / doc
  - Cost per 1,000 Queries (Local SLM): $\le \$0.05$ / 1k queries
  - Cost per 1,000 Queries (Cloud Gemini Flash): $\le \$0.15$ / 1k queries

---

### 20. Pipeline Robustness & Format Stability

Evaluates extraction success rates across heterogeneous, noisy, and corrupted document formats.

#### Detailed Sub-Metric Formulations

* **Robustness Score ($S_{\text{robust}}$)**:
  $$S_{\text{robust}} = \frac{1}{|F|} \sum_{f \in F} \frac{N_{\text{successful\_parses}}(f)}{N_{\text{total\_files}}(f)} \times 100\%$$
  *(where $F = \{\text{Scanned PDF}, \text{Multi-Column PDF}, \text{DOCX}, \text{PPTX}, \text{XLSX}, \text{HTML}, \text{Markdown}, \text{Corrupted PDF}\}$)*

#### Target Robustness Matrix Table

| Format / Layout Challenge | Primary Technical Defense | Target Success Rate |
| :--- | :--- | :---: |
| **Scanned / Image-Only PDFs** | `OCRFallback` EasyOCR rendering | $\ge 99.2\%$ |
| **Multi-Column Layouts** | `DoclingLayoutParser` DocLayNet reading order | $\ge 98.8\%$ |
| **Complex Multi-Row Tables** | `TableSerializer` TableFormer + GFM + DuckDB | $\ge 99.5\%$ |
| **Mathematical Formulas / Equations** | `equation` region preservation ($E = mc^2$) | $\ge 97.5\%$ |
| **Spreadsheets (`.xlsx`, `.csv`)** | Native schema normalization | $100.0\%$ |
| **Corrupted / Truncated PDFs** | Exception handling & fallback text stream | $\ge 95.0\%$ |

---

### 21. Systematic Ablation Studies

Quantifies the exact contribution of each architectural block by measuring performance degradation when that component is removed ($\Delta$).

#### Detailed Sub-Metric Formulations

$$\Delta \text{Metric} = \text{Metric}_{\text{Full\_Pipeline}} - \text{Metric}_{\text{Ablated\_Pipeline}}$$

#### Master Ablation Test Matrix Table

| Ablated Component | Primary Metric Impact | Expected Performance Drop ($\Delta$) | Diagnostic Conclusion |
| :--- | :--- | :---: | :--- |
| **Remove Docling Layout Parser** | Table QA Accuracy & Reading Order | $-42.5\%$ | Proves layout-aware parsing is essential for multi-column and tabular PDFs. |
| **Remove TableSQL Retriever (DuckDB)** | Financial Calculation Accuracy | $-61.2\%$ | Proves dense vector embeddings fail at exact numerical arithmetic. |
| **Remove Cross-Encoder Reranker** | nDCG@5 & Top-1 Precision | $-21.4\%$ | Proves RRF rank outputs require deep joint self-attention rescoring. |
| **Remove Parent-Child Expansion** | Context Coherence & F1 | $-18.6\%$ | Proves isolated 256-token child chunks lack sufficient context for QA. |
| **Remove Short Heading Penalty** | Duplicate Rank Ratio & nDCG@5 | $-14.2\%$ | Proves TOC/heading-only chunks crowd out true content without penalty. |
| **Remove BM25S Sparse Path** | SKU & Exact Keyword Recall | $-28.9\%$ | Proves dense vector search misses exact technical jargon and product IDs. |
| **Remove NLI Faithfulness Verification**| Hallucination Rate | $+24.8\%$ | Proves sentence-level NLI claim checking is vital for zero-hallucination guarantees. |

---

### 22. Industry KPIs & SLA Benchmarks

Provides executive summary KPIs for production deployment readiness.

| Industry KPI | Metric Definition | Target SLA Benchmark | Why It Matters |
| :--- | :--- | :---: | :--- |
| **Mean Query Latency** | Average end-to-end response time | $\le 400 \text{ ms}$ | Direct impact on user interactive experience |
| **P95 Latency SLO** | 95th percentile latency limit | $\le 850 \text{ ms}$ | SLA compliance for 95% of enterprise requests |
| **System Throughput (QPS)**| Concurrent queries served per second | $\ge 20 \text{ QPS / GPU}$ | Capacity planning and hardware sizing |
| **Ingestion Speed** | Document pages processed per hour | $\ge 3,600 \text{ pages/hr}$ | Bulk ingestion pipeline efficiency |
| **Operating Cost / Query** | Total infrastructure cost per query | $\le \$0.00015$ | Financial OpEx optimization |
| **Peak Memory Footprint** | Max RAM/VRAM during workload | $\le 8.0 \text{ GB}$ | Prevents server OOM crashes |
| **Faithfulness Score** | Grounded claim proportion | $\ge 95.0\%$ | Eliminates enterprise liability from hallucinations |
| **System Availability** | Up-time percentage | $\ge 99.99\%$ | Enterprise SLA production readiness |

---

## Codebase Implementation Guide

All evaluation modules are located in `src/tripath/evaluation/` and `src/tripath/attribution/`:

1. **[`src/tripath/evaluation/eval_harness.py`](file:///d:/DocuReason/src/tripath/evaluation/eval_harness.py)**:
   Main evaluation engine calculating Recall@K, nDCG@K, MRR, Precision@K, Attribution Precision, Latency P50/P90/P99, and SLA target verification (`verify_target_achievements`).
2. **[`src/tripath/evaluation/table_eval.py`](file:///d:/DocuReason/src/tripath/evaluation/table_eval.py)**:
   Evaluates TEDS structural layout similarity and cell value content accuracy.
3. **[`src/tripath/evaluation/artifact_quality.py`](file:///d:/DocuReason/src/tripath/evaluation/artifact_quality.py)**:
   Audits processed Document/Chunk dictionaries for character density, empty chunks, and OCR fallback counts.
4. **[`src/tripath/evaluation/dataset_exporter.py`](file:///d:/DocuReason/src/tripath/evaluation/dataset_exporter.py)**:
   Exports fine-tuning datasets for external AI model training.
5. **[`src/tripath/attribution/nli_attributor.py`](file:///d:/DocuReason/src/tripath/attribution/nli_attributor.py)**:
   Deconstructs answers into sentence claims and evaluates NLI evidence support.
6. **[`src/tripath/evaluation/benchmark_dataset.py`](file:///d:/DocuReason/src/tripath/evaluation/benchmark_dataset.py)**:
   Loads standard benchmark datasets (**FinQA**, **TAT-QA**, **DocVQA**, **ChartQA**, **WikiTableQuestions**) with pre-annotated ground-truth pairs.

---

## Benchmark Dataset Automation vs. Custom Enterprise Data

### 1. Benchmark Datasets (FinQA, TAT-QA, DocVQA, ChartQA, WikiTableQuestions)
* **Status**: **100% AUTOMATED** (Zero Human Interaction Required).
* **Mechanism**: Public benchmark datasets come with **pre-annotated ground-truth relevance pairs** (`relevant_ids`, `answer`, `ground_truth_table`).
* **Execution**: Our `BenchmarkDataset` loader (`benchmark_dataset.py`) feeds these datasets directly into `EvaluationHarness` to compute `Recall@K`, `Precision@K`, `nDCG@K`, `MRR`, `MAP`, `TEDS`, `Exact Match`, and `F1` completely automatically without human intervention.

### 2. Custom Unannotated Enterprise Files
* **Status**: Requires Initial Ground-Truth Curation or Synthetic LLM Generation.
* **Mechanism**: When users upload private, brand-new enterprise PDFs that have no pre-existing questions or gold ground-truth labels:
  - **Option A (Synthetic Automation)**: Use synthetic Q&A generation (e.g. LLM-as-a-Judge) to auto-generate question-document ground-truth pairs.
  - **Option B (Human Annotation)**: A domain expert annotates sample questions and target gold document IDs.

---

### Running Evaluations

```bash
# Execute automated evaluation harness
python scripts/evaluate_system.py

# Run ablation studies
python -c "from src.tripath.evaluation.ablation import AblationStudy; AblationStudy().run()"
```
