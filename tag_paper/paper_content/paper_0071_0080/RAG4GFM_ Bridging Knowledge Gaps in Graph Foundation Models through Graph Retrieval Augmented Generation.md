Title: RAG4GFM: Bridging Knowledge Gaps in Graph Foundation Models through Graph Retrieval Augmented Generation
Abstract: Graph Foundation Models (GFMs) have demonstrated remarkable potential across graph learning tasks but face significant challenges in knowledge updating and reasoning faithfulness. To address these issues, we introduce the Retrieval-Augmented Generation (RAG) paradigm for GFMs, which leverages graph knowledge retrieval. We propose RAG4GFM , an end-to-end framework that seamlessly integrates multi-level graph indexing, task-aware retrieval, and graph fusion enhancement. RAG4GFM implements a hierarchical graph indexing architecture, enabling multigranular graph indexing while achieving efficient logarithmic-time retrieval. The task-aware retriever implements adaptive retrieval strategies for node, edge, and graph-level tasks to surface structurally and semantically relevant evidence. The graph fusion enhancement module fuses retrieved graph features with query features and augments the topology with sparse adjacency links that preserve structural and semantic proximity, yielding a fused graph for GFM inference. Extensive experiments conducted across diverse GFM applications demonstrate that RAG4GFM significantly enhances both the efficiency of knowledge updating and reasoning faithfulness 2 .

Section: Introduction
Graph representation learning [1,2,3] has achieved remarkable progress across diverse graph tasks, such as node classification and link prediction. Concurrently, the substantial success of Large Language Models (LLMs) [4,5] has revolutionized natural language processing (NLP) and motivated the development of Graph Foundation Models (GFMs) [3,6,7]. GFMs adapt large-scale pre-training techniques to graph data, enabling powerful cross-domain generalization and multitask adaptability for diverse graph-based applications. However, two practical challenges remain prominent: knowledge updating [8,9,10]i.e., keeping models current as graphs evolveand faithful reasoning [11,12,13]i.e., generating accurate and factually consistent outputs.
On the one hand, as graph data evolves rapidly, GFMs usually require knowledge updates within hours or a day [14,15], massive parameter scales impose substantial computational demands [16,17]. On the other hand, handling complex graph structures [18] and coupling GFMs with LLMs may induce hallucinations [19,20,21], e.g., generating fictitious features or nodes, leading to unfaithful or factually inconsistent outputs.
Some recent studies have attempted to address these challenges. Parameter-efficient fine-tuning (PEFT) methods on GFMs, such as GraphLoRA [22,23] and G-Adapter [24], aim to reduce the cost of knowledge updating by adapting pre-trained models via task-specific post-training (as depicted in Figure 1(b)). However, these methods still require substantial computational resources and remain vulnerable to catastrophic forgetting. Concurrent efforts to improve reasoning faithfulness focus on enhancing the quality of training data [25,26] or refining GFM architectures [27,28]. Despite these advancements, reliable reasoning remains challenging because these methods largely operate within fixed parameters and training data, lacking mechanisms to dynamically ground predictions in verifiable, task-relevant graph evidence.
Retrieval-augmented generation (RAG) [29,30] offers an appealing alternative: by retrieving external evidence at inference time, RAG circumvents frequent parameter updates, adapts to evolving corpora, and can mitigate hallucinations [31,32] However, extending this RAG paradigm to graph data and GFMs raises three challenges: (1) Indexing: how to build graph indices that preserve structure and serve both RAG and GFMs? (2) Retrieval: how to design retrieval mechanisms that accommodate task heterogeneity? (3) Augmentation: how to augment the task-specific query with retrieved graph evidence?
To overcome these challenges, we propose RAG4GFM , a unified RAG framework designed explicitly for graph data, graph tasks, and GFMs. To our knowledge, it is among the first comprehensive designs that operationalize RAG for GFMs. As illustrated in Fig. 1(c), RAG4GFM augments a GFM with external graph knowledge through a multi-stage pipeline. Firstly, the "multi-level graph index" module processes raw graph data into an efficient, structure-aware index. Secondly, the "task-aware retriever" module identifies the user intent and retrieves relevant candidate subgraphs from the constructed index. Finally, the "graph fusion enhancement" module integrates retrieved subgraph knowledge into the user query, enriching its features and structure. By grounding GFM's predictions in retrieved graph evidence-rather than solely in its internal parametersRAG4GFM flexibly adapts to evolving knowledge and markedly enhances reasoning fidelity. Specifically, we first establish an efficient and flexible multi-level graph indexing module. This module is designed to preserve graph semantics and structural topology by integrating four complementary indices: text features extracted via LM encoders, node embeddings that capture structural information via Laplacian positional encoding [33], edge-level representations, and graph-level embeddings. By leveraging the hierarchical structure index, the module achieves comprehensive semantic-structural encoding with logarithmic-time complexity.
Second, building on this index, we propose a task-aware retriever module that adapts to diverse downstream task types. This module dynamically selects appropriate indices and retrieval strategies based on the task typology (node, edge, or graph), retrieving relevant textual and structural features for node tasks, edge-level representations for edge tasks, and graph-level embeddings for graph tasks. Additionally, we use a fusion reranker to consolidate and prioritize retrieval results from both graph features and semantic spaces.
Finally, we introduce a graph fusion enhancement module that integrates retrieved evidence with the query graph effectively. This module consists of two components: a feature-fusion component that integrates the retrieved graph features with query features based on attention weights calculated from similarity scores, while a topological-structure enhancement component augments the query graph's connectivity by combining adjacency information from relevant retrieved graphs through sparse matrix operations. This structured fusion approach surpasses sequential concatenation in traditional RAG by better aligning with graph connectivity, preserving topological information, and accommodating multimodal user queries within the GFM context.
The GFM then performs inference on this fused graph, allowing it to leverage the augmented context for more accurate predictions and a significant reduction in hallucinations. Extensive experiments across multiple GFM applications demonstrate RAG4GFM 's superiority in efficiency and reliability, with safeguards intended to minimize the risk of pre-training data contamination.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b2', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32']

Section: Related Work
Graph Foundation Models. GFMs leverage large-scale pre-training for versatile knowledge transfer, exhibiting reasoning and domain adaptation capabilities [34]. They encompass: (1) Self-supervised learning approaches, such as masked auto-encoding (e.g., GraphMAE [27]); (2) Graph-language model alignment methods that facilitate multimodal pre-training, such as GraphGPT [16]; and (3) Recent architectural innovations aimed at enhancing transferability via Mixture of Experts (MoE), such as AnyGraph [17], structural understanding through topology-aware tokenization as in Open-Graph [35], or improving generalization through property-driven training, such as GraphProp [28]. However, GFMs are constrained by inefficient knowledge updating, typically requiring extensive retraining [16], and difficulties in ensuring reasoning reliability over complex graph structures, potentially leading to biases [11].
Graph Indexing and Retrieval. Graph indexing and retrieval have progressed from general vector methods to structure-aware techniques. Initial advancements focused on optimizing vector-space retrieval efficiency, including adaptive algorithms, such as FLANN [36], and hierarchical graph structures for efficient search, such as HNSW [37,38]. These formed the basis for scalable libraries such as FAISS [39]. While traditional Information Retrieval (IR) methods like BM25 [40] exist, vector-based strategies are more directly applicable to graph data retrieval. Despite these advancements, developing retrieval systems that effectively integrate multimodal data (e.g., text with graph structures) and generalize across diverse graph-specific tasks remains a significant challenge.
Graph Retrieval-Augmented Generation. RAG [30] enhances LLMs by integrating external knowledge through a "retrieve-generate" pipeline. Its progression includes: (1) Foundational end-to-end trainable RAG frameworks for knowledge-intensive NLP [30]; (2) Architectural refinements, including multi-hop retrieval for comprehensive knowledge gathering and integrated retriever-generator optimization [41,42]; and (3) Recent graph-centric extensions, featuring methods that emphasize structure-aware retrieval, such as GraphRAG [43], computational optimization, such as Ligh-tRAG [44], or hybrid knowledge integration, such as HybridRAG [45]. Nevertheless, applying RAG to graph data still encounters critical challenges. Standard vector retrieval techniques often fail to adequately represent complex graph topology [46]. Moreover, the creation of specialized graph indexing and retrieval mechanisms tailored for the unique requirements of graph RAG remains an active research direction.
this section cite: ['b33', 'b26', 'b15', 'b16', 'b34', 'b27', 'b15', 'b10', 'b35', 'b36', 'b37', 'b38', 'b39', 'b29', 'b29', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45']

Section: RAG4GFM
RAG4GFM is a RAG framework tailored for graph data and GFMs, integrating external graph knowledge into the GFM inference process. The pipeline initiates with multi-level graph indexing (Figure 2(a)), where RAG4GFM constructs HNSW-based [47] multimodal indices for graph corpora. Next, the task-aware retrieval mechanism (Figure 2(b)) processes user queries (text and graph structure) and adaptively retrieves relevant subgraphs. Subsequently, the graph fusion enhancement module (Figure 2(c)) integrates these retrieved subgraphs with the original query graph through a two-step process: attention-based feature fusion and adjacency matrix-based topological fusion. Finally, the fused graph is input to the base GFM for inference.
this section cite: ['b46']

Section: Pre-Training

this section cite: []

Section: This graph is an acyclic compound

this section cite: []

Section: Task-Aware Retriever

this section cite: []

Section: Molecular Graph
Predict Enhancement Retrieve Encode
Construct Text/Graph Vector Similarity Query Ranked Subgraphs with Score Text/Graph Encoder Graph Reason based Reranker Retrieved Subgraphs
(a) Multi-Level Graph Indexing (b) Task-Aware Retriever 0.93 0.76 0.52
this section cite: []

Section: Please classify this graph

this section cite: []

Section: User query
Multi-Level Graph Index Fusion Graph
this section cite: []

Section: (c) Graph Fusion Enhancement
Base GFM
this section cite: []

Section: Attention-based Feature Fusion

this section cite: []

Section: Graph Attention Network Topological Fusion

this section cite: []

Section: Graph Fusion Enhancement
Code Graph Wikipedia Graph Cite Network
this section cite: []

Section: Zoom In

this section cite: []

Section: Zoom In
Figure 2: The overall architecture of RAG4GFM . RAG4GFM dynamically augments a Base GFM by: (a) constructing a "Multi-Level Graph Index" (b) retrieving and reranking task-relevant subgraphs based on the user query via a "Task-Aware Retriever"; and (c) fusing the retrieved subgraphs with query information through the "Graph Fusion Enhancement module", before generation.
this section cite: []

Section: Multi-level Graph Indexing
The effective application of the RAG paradigm in GFMs hinges on constructing graph data indexing systems that preserve graph completeness while ensuring efficient querying. Unlike traditional LLMs, user queries in GFM application scenarios typically present a hybrid form of text and graph data. The complex topology and multimodal attributes of graph data pose a significant challenge: designing an index system that simultaneously maintains graph structural fidelity and supports efficient retrieval.
To address this challenge, we propose a multi-level graph indexing approach that first encodes mixed graph features, and then constructs hierarchical indices to enable scalable querying and retrieval.
Mixed Feature Encoding Mechanism. To comprehensively represent the multidimensional characteristics of graph data, this work designs four complementary feature encoding mechanisms: node feature encoding, structural feature encoding, edge feature encoding, and graph-level encoding, which align with the hierarchical reasoning capabilities discussed in Section 1. This mixed encoding strategy realizes multi-granular representations of graph data, thereby accommodating diverse retrieval requirements under varied tasks.
Node Feature Encoding: For nodes in the graph, we employ pre-trained language models to semantically encode the textual descriptions of node v as h t (v) = LM(v). This method fully leverages the advantages of pre-trained language models in semantic understanding, mapping the textual information of nodes to high-dimensional semantic spaces.
this section cite: []

Section: Structural Feature Encoding:
The topological position of nodes in a graph contains rich structural information. We combine Laplacian Positional Encoding (LAPPE) with node degree metrics to construct structure-aware feature representations as
h s (v) = LAPPE(v) ⊕ Deg IN (v) ⊕ Deg OUT (v),(1)
where LAPPE(v) is the positional encoding based on the eigendecomposition of the graph laplacian matrix, Deg IN (v) and Deg OUT (v) represent the in-degree and out-degree of node v, respectively, and ⊕ denotes feature concatenation. Laplacian positional encoding effectively captures the position information of nodes in the global graph structure, while node degree information reflects local connection patterns. This encoding method is invariant to node permutations and graph isomorphism, accurately capturing nodes' relative positions within the topology.
Edge Feature Encoding: To capture the structural properties and topological roles of edges, we transform the original graph into its corresponding line graph. We then compute the LapPE for the nodes in this line graph. This approach allows us to generate a feature representation h e (u, v) for each edge e in the original graph that effectively encodes its structural context within the overall graph topology. This approach effectively differentiates edges by their connectivity patterns and structural roles within the graph.
Graph-level Encoding: To obtain a holistic representation, we integrate node features, edge features, and graph statistical features:
h g (G) = h NODE (G) ⊕ h EDGE (G) ⊕ h STATS (G),(2)
where h NODE (G) = MEAN{h t (v) | v ∈ V }, h EDGE (G) = MEAN{h e (u, v) | (u, v) ∈ E}, and h STATS (G) = [|V |, |E|, ρ(G)],
where h NODE (G), h EDGE (G), and h STATS (G) represent node feature aggregation, edge feature aggregation, and graph statistical features, respectively. ρ(G) denotes the graph density, i.e., the ratio between actual and maximal possible edge counts. This comprehensive representation method fully captures the overall characteristics of the graph, providing effective support for graph-level tasks.
this section cite: []

Section: Hierarchical Index Construction.
With the mixed features obtained, we proceed to organize them into an efficient multi-level index structure. This research selects the Hierarchical Navigable Small World (HNSW) as the theoretical foundation for the index structure. The hierarchical search pattern of HNSW aligns well with the multi-scale structure inherent in graph data. Accordingly, we construct a four-level hierarchical index structure covering node-, structure-, edge-, and graph-level representations:
H INDEX = {h t (v), h s (v), h e (u, v), h g (G)}.
This multi-space index design offers two primary advantages: firstly, it overcomes the limitations of single representation methods by comprehensively capturing both semantic and structural graph information while flexibly supporting diverse node, edge, and graph-level retrieval tasks through a unified interface; secondly, it guarantees efficient O(log 2 N ) retrieval time, crucial for large-scale graph data.
this section cite: []

Section: Task-aware Retrieval
GFMs span diverse application scenariossuch as node classification, link prediction, and graph classificationeach with distinct retrieval requirements. Traditional unified retrieval strategies fail to capture the heterogeneity of such tasks, often resulting in suboptimal efficiency and precision. To address this, we propose a task-aware retrieval framework that dynamically adapts retrieval strategies according to task characteristics, thereby improving both retrieval accuracy and computational efficiency.
this section cite: []

Section: Retrieval Strategy Selector.
To accommodate heterogeneous graph tasks, the retrieval module must interpret user intent and select optimal retrieval strategies accordingly. We employ an LM-based task classifier that performs joint analysis of the natural language query and its associated graph context to predict the task type: τ (q) ∈ {NODE, EDGE, GRAPH}, where τ denotes the LLM-implemented task discrimination function, and q represents the user's natural language query. The predicted task type determines which feature spaces and retrieval operators are subsequently activated.
this section cite: []

Section: Hybrid Feature Retrieval.
Based on task classification results, the retrieval module adaptively selects task-relevant feature spaces to minimize irrelevant noise and align with downstream GFM objectives.
For node-level tasks, such as node classification and node regression, we jointly query both node and structural feature spaces to capture fine-grained local semantics. For edge-level tasks, including link prediction and edge classification, we leverage node and edge features to represent pairwise relational semantics. For graph-level tasks, such as graph classification and regression, we utilize holistic graph embeddings augmented with aggregated node features.
To obtain the final retrieval score for a query q under the graph context C, we apply a reciprocal rank fusion (RRF) strategy to aggregate results from multiple retrieval channels:
S(q|h q , C) = q∈F (hq,C) 1 d + rank k q (C) ,(3)
where F(h q , C) denotes the retrieved feature set, d is a smoothing constant, and rank k x (C) represents the rank of candidate x among the top-k results. The corresponding feature sets for different tasks are defined as:
F(h q , C) =    {h t (v), h s (v)}, if C = v (node-level); {h t (u), h t (v), h e (u, v)}, if C = (u, v) (edge-level); {h t (v), h g (G)}, if C = G (graph-level).(4)
Unlike conventional methods that rely solely on node-level feature aggregation, our hybrid retrieval framework integrates semantic and structural cues across multiple levels, enabling task-adaptive retrieval that improves both precision and efficiencykey challenges emphasized in the introduction.
Unlike traditional methods that rely solely on node feature aggregation, our approach constructs more comprehensive graph representations through the integration of multi-level information.
this section cite: []

Section: Graph Fusion Enhancement
In conventional RAG frameworks for NLP tasks, retrieved text fragments are typically fused via simple concatenation or mean pooling of embeddings. However, when applied to graph data, such approaches fail to preserve topological relationships and path dependencies, resulting in substantial loss of structural information. Moreover, node importance in graphs depends heavily on positional and connectivity patterns, which traditional fusion methods cannot effectively capture. To address these limitationsparticularly the structure-semantics imbalance highlighted in Section 1, we propose a dual graph fusion enhancement mechanism, consisting of (1) attention-based feature fusion and (2) topological structure enhancement.
this section cite: []

Section: Attention Feature Fusion.
We introduce a similarity-based dynamic weighting mechanism that adaptively determines the importance of each retrieved graph according to its semantic similarity with the query graph. This mechanism enables the model to assign higher weights α F i to more relevant retrieved graphs, thereby focusing on knowledge most pertinent to the query.
h N v = h v + k i=1 α F i • 1 |V i | u∈Vi h u • I[α F i > γ], (node-level) (5
) h E SRC = h SRC + k i=1 α F i • 1 |V i | u∈Vi h u • I[α F i > γ], (edge-level, source) (6
) h E DST = h DST + k i=1 α F i • 1 |V i | u∈Vi h u • I[α F i > γ], (edge-level, destination) (7
) h G v = 1 k k i=1 α F i • β i v • h v • I[α F i > γ], (graph-level) (8
)
β i v = SOFTMAX(h v • (h q g + h i g )), (graph-level, attention) (9
)
where h N v is the enhanced target node feature, h v is the original node feature, h E SRC and h E DST are the enhanced source and destination edge features, h G v is the enhanced target graph feature, V i is the node set of the i-th retrieved graph, and γ is the feature fusion threshold (default value 0.5) used to filter low-relevance retrieved results. h q g and h i g represent the global features of the query graph and the i-th retrieved graph, respectively, and k is the number of effective retrieved results.
Compared to traditional feature fusion methods, the attention feature fusion mechanism proposed in this research adaptively emphasizes highly relevant knowledge through dynamic weight allocation, effectively reducing noise impact while utilizing threshold filtering mechanisms to avoid interference from low-quality retrieved results.
Topological Structure Enhancement. Complementary to feature fusion, the topological structure enhancement module focuses on enriching the query graph's connectivity patterns. It selectively incorporates structurally relevant edges from retrieved graphs, weighted by their graph-level relevance scores α T i , thereby augmenting the query graph with semantically aligned structural information. Implemented via sparse adjacency operations, this module efficiently handles large-scale graph data while maintaining topological consistency.
Given the adjacency matrix A of the query graph and the set of adjacency matrices A 1 , A 2 , ..., A k of retrieved graphs, the enhanced adjacency matrix is computed as:
A = A + k i=1 α T i • A i • I[α T i > γ],(10)
where α T i serves as the relevance weight. To suppress noisy edges, we apply threshold filtering such that A uv = I[A uv > δ], where δ is the edge addition threshold (default value 0.5). This sparse matrix-based implementation achieves efficient large-scale processing while preserving graph sparsity and interpretability.
Other analyses. Due to space limitations, we present analyses on the complexity in Appendix A.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. To evaluate the effectiveness and generality of RAG4GFM , we conduct experiments on a diverse collection of graph datasets covering six task types. For node classification and link prediction, we adopt the latest TAG benchmark [48] to prevent potential data leakage from GFM pre-training. For other tasks, we select datasets that, to the best of our knowledge, were not used in any GFM pre-training phase, verified through public model documentation and release notes. Table 1 summarizes the dataset statistics, with additional details in Appendix B.1.
Table 1: Statistics of the dataset used in our experiments. "Task" denotes the downstream task type: NC (Node Classification), NR (Node Regression), LC (Link Classification), LP (Link Prediction), GC (Graph Classification), and GR (Graph Regression). "Graphs" indicates the number of graph instances; a value of 1 denotes a single large graph.
this section cite: ['b47']

Section: Dataset

this section cite: []

Section: Nodes Edges Graphs Domain Task
Books-Children [48] 76,875 1,554,578 1 E-commerce NC,LP Books-History [48] 41,551 358,574 1 E-commerce NC, LP Ele-Computers [48] 87,229 721,081 1 E-commerce NC, LP Ele-Photo [48] 48,362 500,928 1 E-commerce NC, LP MiniGCDataset [49] 21,909 177,875 1,000 Synthetic GC BA2MotifDataset [50] 25,000 51,392 1,000 Synthetic GC Chameleon [51] 2,277 36,101 1 Wikipedia NR WN18Dataset [52] 40,943 151,442 1 Knowledge LC QM7bDataset [53] 108,165 1,766,695 7,211 Biology GR
Baselines. We evaluate RAG4GFM against state-of-the-art approaches from three categories:
(1) Prompt Engineering Methods: Few-shot Learning [54], Chain-of-Thought (COT) [55], IRaugmented COT [56]. These methods focus on structuring the input prompt to guide the GFM's inference without altering model parameters, leveraging strategies like few-shot examples or explicit reasoning steps. (2) Retrieval-Enhanced Methods: VanillaRAG [30], GraphRAG [43], G-Retriever [57]. These methods augment GFMs with external knowledge retrieval and ground predictions in relevant evidence, ranging from text to graph-aware retrieval. (3) Graph Out-of-Distribution Generalization Methods: Prototype [58],GNNSafe [59]. These approaches focus on the ability to generalize to unseen domains, structures, or distributions, emphasizing robustness and transferability.
GFMs. We evaluate RAG4GFM on seven representative GFMs, grouped by their predictive architecture into three types: (1) GNNs as predictor: OpenGraph [35] and AnyGraph [17]. (2) Co-learning GNNs and LLMs: GLEM [60]. (3) LLMs as predictor: GraphGPT [16], HiGPT [61], LLaGA [62], and GraphAdapter [63]. Further details and descriptions of these models are provided in Appendix B.2.
this section cite: ['b47', 'b47', 'b47', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b29', 'b42', 'b56', 'b57', 'b58', 'b34', 'b16', 'b59', 'b15', 'b60', 'b61', 'b62']

Section: Effectiveness of RAG-Enhanced GFMs
Our first research question investigates whether RAG-based mechanisms enhance the performance of existing GFMs across various graph tasks. Table 2 shows the results for node classification and link prediction, while additional findings for node regression, link classification, graph classification, and graph regression are reported in Appendix C.1. Note that some GFMs lack results for tasks not supported by their architectures; for instance, OpenGraph does not support graph-level classification. Across all four datasets, the RAG4GFM yields consistent and substantial improvements, demonstrating its strong generalization capability. For instance, GLEM+RAG4GFM achieves 83.12% accuracy on node classification in the Computers dataset, representing a relative gain of approximately 5.5% over GLEM. Similarly, GraphGPT+RAG4GFM attains 60.37% accuracy on link prediction, outperforming its vanilla counterpart by 5.1%.
The improvements are consistently observed across all four datasetsComputers, History, Fitness, and Photohighlighting the broad applicability of our approach. For example, GLEM+RAG4GFM achieves 83.12% accuracy on node classification in the Computers dataset, a relative gain of 5.5% over the baseline GLEM. Similarly, GraphGPT+RAG4GFM attains 60.37% accuracy on link prediction, outperforming its vanilla counterpart by 5.1%. These consistent trends indicate that RAG-based augmentation benefits both GNN-based and LLM-based GFMs. To elucidate the source of these improvements, we analyze the core design of RAG4GFM . The Multi-level Indexing, coupled with the Task-Aware Retriever module, ensures that each GFM retrieves the most relevant external subgraphs and feature spaces efficiently, reducing noise and retrieval latency. Finally, the Graph Fusion Enhancement module integrates retrieved information into the query graph through both feature-level and topology-level fusion. This integration refines node embeddings for classification, strengthens edge inference for link prediction, and provides more coherent context for graph-level reasoning. These components operate synergisticallyencoding richer local context, retrieving task-relevant external knowledge, and integrating it in a structure-consistent manner.
Overall, the results demonstrate that RAG4GFM effectively overcomes the static-parameter limitation of conventional GFMs by dynamically grounding predictions in external structured knowledge. This design directly addresses the integration bottleneck highlighted in the introduction, leading to more accurate, contextually grounded, and robust graph-level reasoning across tasks without requiring additional post-training.
this section cite: []

Section: Ablation Study on RAG4GFM
To quantitatively evaluate the individual contributions of RAG4GFM 's key components, we perform a series of ablation experiments using AnyGraph as the base GFM under identical hyperparameter settings for fair comparison. We analyze both architectural modules (retrieval, fusion, indexing) and feature-level encodings to understand how each design choice affects performance on node classification (Computers) and link prediction (History) tasks.
Variants on Core Architecture. w/o RAG. Removes the entire retrieval-augmented generation (RAG) pipeline; the GFM operates without external knowledge retrieval or integration. w/o GF. Retains retrieval but replaces the specialized graph fusion module with a naive concatenation strategy, removing the semantic-structural alignment mechanism. w/o GI. Replaces the hierarchical graph indexing with a basic text-similarity index, assessing the importance of structure-aware retrieval.
Variants on Feature Encoding.
To further assess the effectiveness of our structural feature design, we compare three encoding strategies: LAPPE only. Uses Laplacian positional encodings, capturing global structural information but limited in local awareness. Node Degree only. Uses node in/outdegree as simple local centrality features. LAPPE + Degree. Combines both global positional and local topological cues, aligning with the principle of global-local structural complementarity. R A G 4 G F M w /o R A G w /o G F w /o G I L A P P E o n ly N o d e D e g re e o n ly L A P P E + D e g re e 50 55 60 65 70 75 80 85 Performance (%) 79.6 73.5 76.2 77.9 72.2 68.9 79.6 66.1 60.9 63.4 64.8 58.4 55.2 66.1 Computers (Node Classification) Accuracy AUC R A G 4 G F M w /o R A G w /o G F w /o G I L A P P E o n ly N o d e D e g re e o n ly L A P P E + D e g re e 63.5 56.7 59.8 60.5 56.7 54.1 63.5 64.1 56.8 59.1 60.2 56.8 53.5 64.1 History (Link Prediction) Accuracy AUC Figure 3: Ablation study results for RAG4GFM on node classification (Computers) and link prediction (History) tasks. Higher values are better.
As illustrated in Figure 3, the full RAG4GFM configuration consistently achieves the best results across both tasks. Among the architectural variants, w/o RAG exhibits the largest drop, confirming that retrieval is crucial for integrating dynamic external knowledge. Removing the graph fusion module (w/o GF) also decreases accuracy and AUC, showing that our fusion mechanism is essential for coherent reasoning beyond naive feature concatenation. The performance reduction of w/o GI highlights the role of hierarchical, structure-aware indexing in precise, low-noise retrieval.
Regarding feature encoding, models using only LAPPE or only node degree perform noticeably worse than the joint configuration. The combined LAPPE+Degree variant recovers nearly the same performance as the full model, validating that global-local structural complementarity enhances graph representation learning. This observation is consistent across both Computers and History tasks, underscoring that balanced structural cues benefit retrieval and fusion.
All componentsRAG retrieval, graph fusion, hierarchical indexing, and global-local feature encodingcontribute jointly to RAG4GFM 's effectiveness. Their synergy enables efficient updates, faithful reasoning, and more accurate predictions across diverse graph tasks.
this section cite: []

Section: Comparison with Other Knowledge Updating Methods
We evaluate tasks where external knowledge is critical and frequently updated, such as knowledgeintensive node classification.
Results in Table 3 show that RAG4GFM consistently outperforms all competing approaches, confirming its capability to retrieve and integrate relevant external information for accurate prediction. Retrieval-enhanced (RE) methods generally outperform prompt-engineering (PE) and graph out-ofdistribution (G-OOD) methods. Among RE baselines, graph-aware models such as GraphRAG and G-Retriever achieve stronger results than lexical retrievers, yet RAG4GFM delivers further gains by explicitly fusing semantic and structural representations. PE methods provide only marginal improvements, while text-only augmentation (e.g., IR-augmented CoT) remains insufficient for complex graph reasoning. G-OOD-oriented approaches (Prototype, GNNSafe) underperform in our setting, as they emphasize distributional robustness rather than dynamic knowledge updating. In summary, RAG4GFM demonstrates clear advantages as a knowledge-enhancement framework. Compared with traditional fine-tuning or static knowledge-graph integration, the RAG paradigm offers a flexible and efficient mechanism for updating GFMs with external knowledge. It achieves this without costly retraining while maintaining reasoning fidelity.
Beyond predictive performance, we further examine the efficiency of RAG4GFM compared to GraphLoRA [22], a representative parameter-efficient fine-tuning (PEFT) approach. Table 4 reports the time and GPU memory required to reach identical accuracy targets. RAG4GFM achieves comparable accuracy while reducing runtime by up to 7.0× and GPU memory usage by over 60%. This efficiency arises from its retrieval-based updating paradigm, which avoids gradient-based optimization and large parameter storage. Moreover, since RAG4GFM refreshes knowledge through lightweight retrieval and fusion rather than re-training, it scales favorably to frequent graph updates and resource-constrained environments. Other analyses. Due to space limitations, we present complexity analysis, experimental setups, and further experimental results in C.
this section cite: ['b21']

Section: Conclusions
In this paper, we introduce RAG4GFM , a RAG framework tackling two critical GFM challenges: efficient knowledge updating and faithful reasoning. Leveraging a three-component architecture, RAG4GFM achieves significant gains in knowledge-update efficiency and reasoning faithfulness over traditional parameter-updating approaches, as demonstrated by extensive empirical evaluation across diverse domains. In future work, we will focus on: scalability to billion-node graphs and real-time systems via disk-based ANN (e.g., DiskANN) and asynchronous fusion; and broader directions, including leveraging negative/contrastive knowledge to refine decision boundaries and extending the framework to multimodal graphs that include images. Our goal is to promote responsible AI practices in the development and deployment of RAG-enhanced GFMs.
this section cite: []

Section: References
Ref_id:b0 Title: Inductive representation learning on large graphs. Advances in neural information processing systems Year: (2017)
Ref_id:b1 Title: Graph attention networks Year: (2017)
Ref_id:b2 Title: Graph meets llms: Towards large graph models Year: (2023)
Ref_id:b3 Title:  Year: ()
Ref_id:b4 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b5 Title: Graphprompt: Unifying pre-training and downstream tasks for graph neural networks Year: (2023)
Ref_id:b6 Title: Talk like a graph: Encoding graphs for large language models Year: (2023)
Ref_id:b7 Title: Continual learning of large language models: A comprehensive survey Year: (2024)
Ref_id:b8 Title: Lifelong knowledge editing for llms with retrieval-augmented continuous prompt learning Year: (2024)
Ref_id:b9 Title: Tic-lm: A multi-year benchmark for continual pretraining of language models Year: ()
Ref_id:b10 Title: Trustworthy llms: A survey and guideline for evaluating large language models' alignment Year: (2023)
Ref_id:b11 Title: A survey of hallucination in large foundation models Year: (2023)
Ref_id:b12 Title: Augmenting llms with knowledge: A survey on hallucination prevention Year: (2023)
Ref_id:b13 Title: A comprehensive survey of dynamic graph neural networks: Models, frameworks, benchmarks, experiments and challenges Year: (2024)
Ref_id:b14 Title: Streame: Learning to update representations for temporal knowledge graphs in streaming scenarios Year: (2023)
Ref_id:b15 Title: Graphgpt: Graph instruction tuning for large language models Year: (2024)
Ref_id:b16 Title: Anygraph: Mixing expert graph neural networks with graph-level routing Year: (2024)
Ref_id:b17 Title: Graphedit: Large language models for graph structure learning Year: (2024)
Ref_id:b18 Title: Detecting hallucinations in large language models using semantic entropy Year: (2024)
Ref_id:b19 Title: Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models Year: (2023)
Ref_id:b20 Title: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions Year: (2023)
Ref_id:b21 Title: Graphlora: Empowering llms fine-tuning via graph collaboration of moe Year: (2024)
Ref_id:b22 Title: Graphlora: Structure-aware contrastive low-rank adaptation for cross-graph transfer learning Year: (2024)
Ref_id:b23 Title: G-adapter: Towards structure-aware parameterefficient transfer learning for graph transformer networks Year: (2024)
Ref_id:b24 Title: Graph contrastive learning with augmentations Year: (2020)
Ref_id:b25 Title: Rethinking graph neural networks for anomaly detection Year: (2022)
Ref_id:b26 Title: Graph mae: Self-supervised masked graph autoencoders Year: (2022)
Ref_id:b27 Title: Graphprop: Training the graph foundation models using graph properties Year: (2024)
Ref_id:b28 Title: Retrieval-augmented generation for large language models: A survey Year: (2023)
Ref_id:b29 Title: Retrieval-augmented generation for knowledge-intensive nlp tasks Year: (2020)
Ref_id:b30 Title: Can knowledge graphs reduce hallucinations in llms?: A survey Year: (2023)
Ref_id:b31 Title: Generate-on-graph: Treat llm as both agent and kg in incomplete knowledge graph question answering Year: (2024)
Ref_id:b32 Title: Benchmarking graph neural networks Year: (2023)
Ref_id:b33 Title: Towards graph foundation models: A survey and beyond Year: (2023)
Ref_id:b34 Title: Towards open graph foundation models Year: (2024)
Ref_id:b35 Title: Scalable nearest neighbor algorithms for high dimensional data Year: (2014)
Ref_id:b36 Title: Engineering efficient and effective non-metric space library Year: (2016)
Ref_id:b37 Title: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs Year: (2020)
Ref_id:b38 Title: Billion-scale similarity search with gpus Year: (2019)
Ref_id:b39 Title: The probabilistic relevance framework: Bm25 and beyond Year: (2009)
Ref_id:b40 Title: Leveraging passage retrieval with generative models for open domain question answering Year: (2021)
Ref_id:b41 Title: Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries Year: (2024)
Ref_id:b42 Title: From local to global: A graph rag approach to query-focused summarization Year: (2024)
Ref_id:b43 Title: Lightrag: Simple and fast retrieval-augmented generation Year: (2024)
Ref_id:b44 Title: Hybridrag: Integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction Year: (2024)
Ref_id:b45 Title: Graph retrieval-augmented generation for large language models: A survey Year: (2024)
Ref_id:b46 Title: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs Year: (2018)
Ref_id:b47 Title: A comprehensive study on text-attributed graphs: Benchmarking and rethinking Year: (2023)
Ref_id:b48 Title:  Year: (2024)
Ref_id:b49 Title: Parameterized explainer for graph neural network Year: (2020)
Ref_id:b50 Title: Geom-gcn: Geometric graph convolutional networks Year: (2020)
Ref_id:b51 Title: Translating embeddings for modeling multi-relational data Year: (2013)
Ref_id:b52 Title: 970 million druglike small molecules for virtual screening in the chemical universe database gdb-13 Year: (2009)
Ref_id:b53 Title: Generalizing from a few examples: A survey on few-shot learning Year: (2020)
Ref_id:b54 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b55 Title: Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions Year: (2022)
Ref_id:b56 Title: G-retriever: Retrieval-augmented generation for textual graph understanding and question answering Year: (2024)
Ref_id:b57 Title: Prototypical networks for few-shot learning Year: (2017)
Ref_id:b58 Title: Energy-based out-of-distribution detection for graph neural networks Year: (2023)
Ref_id:b59 Title: Learning on large-scale text-attributed graphs via variational inference Year: (2022)
Ref_id:b60 Title: Higpt: Heterogeneous graph language model Year: (2024)
Ref_id:b61 Title: Llaga: Large language and graph assistant Year: (2024)
Ref_id:b62 Title: Can gnn be good adapter for llms? Year: (2024)
Ref_id:b63 Title: Fast and accurate modeling of molecular atomization energies with machine learning Year: (2012)
Ref_id:b64 Title:  Year: (2023)
