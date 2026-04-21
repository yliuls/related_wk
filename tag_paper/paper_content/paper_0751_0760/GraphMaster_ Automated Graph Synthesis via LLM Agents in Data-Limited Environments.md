Title: GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments
Abstract: The era of foundation models has revolutionized AI research, yet Graph Foundation Models (GFMs) remain constrained by the scarcity of large-scale graph corpora. Traditional graph data synthesis techniques primarily focus on simplistic structural operations, lacking the capacity to generate semantically rich nodes with meaningful textual attributes-a critical limitation for real-world applications. While large language models (LLMs) demonstrate exceptional text generation capabilities, their direct application to graph synthesis is impeded by context window limitations, hallucination phenomena, and structural consistency challenges. To address these issues, we introduce GraphMaster-the first multi-agent framework specifically designed for graph data synthesis in data-limited environments. GraphMaster orchestrates four specialized LLM agents (Manager, Perception, Enhancement, and Evaluation) that collaboratively optimize the synthesis process through iterative refinement, ensuring both semantic coherence and structural integrity. To rigorously evaluate our approach, we create new data-limited "Sub" variants of six standard graph benchmarks, specifically designed to test synthesis capabilities under realistic constraints. Additionally, we develop a novel interpretability assessment framework that combines human evaluation with a principled Grassmannian manifold-based analysis, providing both qualitative and quantitative measures of semantic coherence. Experimental results demonstrate that GraphMaster significantly outperforms traditional synthesis methods across multiple datasets, establishing a strong foundation for advancing GFMs in data-scarce environments. 2 * Corresponding author 2 Code is available on https://github.com/EnjunDu/GraphMaster. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).existing connections but cannot create novel nodes or patterns. Node-level mixing techniques like GraphMixup [53] generate synthetic nodes by interpolating features but often produce semantically inconsistent attributes, particularly with textual features. Graph-level synthesis methods such as G-Mixup [18] create entirely new graphs but struggle to balance global structure with local semantic coherence. The core limitation across these traditional methods is their inability to simultaneously preserve meaningful semantics while generating structurally valid expansions-a deficiency particularly pronounced when handling text-attributed graphs (TAGs) where both connectivity patterns and textual node features must remain coherent.Large language models have demonstrated remarkable capabilities in understanding and generating text [19,11,61,24,2,45], suggesting potential for synthesizing text-attributed graphs. However, directly applying LLMs encounters several critical challenges: standard context windows cannot process entire graphs with numerous textual nodes [3]; LLMs excel at semantic understanding but struggle to maintain structural consistency [12]; and without proper coordination, they tend to produce inconsistent or hallucinated content that fails to capture the intricate balance between topology and semantics [33]. Furthermore, in realistic scenarios with limited available data, LLMs have insufficient examples to learn complex graph patterns [27,44,25].To address these challenges, we propose GraphMaster, a novel multi-agent framework specifically designed for graph synthesis in data-limited environments. GraphMaster decomposes the complex synthesis task into specialized sub-tasks handled by four collaborative LLM-powered agents, each targeting specific challenges: (1) The Manager Agent coordinates the overall process and determines optimal synthesis strategies based on current graph characteristics, orchestrating the complex synthesis workflow; (2) The Perception Agent analyzes graph structure and employs advanced sampling to identify representative subgraphs processable within LLM context constraints, directly addressing the context window limitations; (3) The Enhancement Agent generates new nodes and edges with consistent semantics and structure, mitigating hallucination by maintaining coherence with existing graph elements; and (4) The Evaluation Agent assesses quality based on both semantic coherence and structural integrity, providing feedback for iterative improvement to ensure structural and semantic consistency. This decomposition enables targeted solutions for each challenge that a single-pass LLM approach cannot address.Through this collaborative, iterative process, these specialized agents overcome the limitations of both traditional methods and direct LLM applications. The multi-agent architecture enables GraphMaster to effectively balance semantic richness with structural validity-producing high-quality synthetic graph data even with limited training examples. By introducing modular reasoning (through task decomposition), semantic control (via specialized agent expertise), and iterative optimization (through feedback cycles), GraphMaster achieves synthesis capabilities beyond what single-pass approaches can deliver.Our contributions can be summerized as follows:• New perspective for LLM-based TAG Synthesis: we first propose a novel multi-agent framework from the RAG perspective to synthesize TAG under data-limited environment.By integrating context retrieval with iterative feedback, this new perspective enables both semantic richness and structural fidelity.• Groundbreaking Benchmark: We introduce a standardized "Data-limited" variant testbed for text-attributed graph synthesis and develop a dual-perspective interpretability assessment-combining expert human evaluation with Grassmann manifold-based analysis-to provide reproducible comparisons and deep semantic-structural insights.• State-of-the-Art Performance: Extensive experiments on multiple datasets and GNN architectures demonstrate that our method consistently outperforms existing baselines, setting a new benchmark for data-limited TAG synthesis.2 Background Methods Classic Graph Data Synthesis Methods Traditional graph data synthesis methods [7] address data scarcity through various approaches. Edgelevel operations [63] modify topology by adding or deleting connections. Node-level techniques like

Section: Introduction
In the era of foundation models, unprecedented advances in natural language processing [43,59] and computer vision [14,48,15,26] have been enabled by massive training corpora [57,55,58,56,66,52,9]. Graph Foundation Models (GFMs) [31,28,43,59,8] represent a promising frontier for AI in graph-structured data, yet their development faces a critical bottleneck: the scarcity of large-scale, diverse graph datasets.Unlike text and image domains where data collection is relatively straightforward, gathering and annotating graph data often requires specialized expertise and significant resources. This data quantity constraint has become the primary challenge for training robust GFMs, particularly as model size increases and demands exponentially more training examples for optimal performance.
Graph data synthesis offers a strategic solution to this fundamental constraint by automatically generating new graph samples that maintain both semantic richness and structural validity. Existing synthesis approaches, however, face substantial limitations. Edge-level operations [63,65] manipulate GraphSMOTE [62] generate new nodes through minority class interpolation. Graph-level methods such as G-Mixup [18] create entirely new graph instances via graphon interpolation. Interpolationbased approaches [40,39] combine hidden representations to enhance model robustness. Despite their diversity, these methods primarily focus on structural manipulations without generating semantically meaningful textual attributes.
this section cite: ['b42', 'b58', 'b13', 'b47', 'b14', 'b25', 'b56', 'b54', 'b57', 'b55', 'b65', 'b51', 'b8', 'b30', 'b27', 'b42', 'b58', 'b7', 'b62', 'b64', 'b61', 'b17', 'b39', 'b38']

Section: LLM-based Multi-Agent Systems for Data Generation
Recent LLM-powered multi-agent frameworks demonstrate capabilities for complex data generation tasks. General collaboration systems like Self-Instruct [41] and distributed simulation platforms [34] establish architectures for coordinated AI systems. In graph contexts, approaches like GoG [47] and LLM-based social simulations [20] leverage semantic understanding for graph-related tasks. However, specific applications for text-attributed graph synthesis in data-limited environments remain largely unexplored.
this section cite: ['b40', 'b33', 'b46', 'b19']

Section: Problem Formulation: Graph Data Synthesis
Text-Attributed Graphs. We formally define a text-attributed graph (TAG) as G = (V, E, X , Y), where V = {v 1 , v 2 , . . . , v N } is a set of N nodes, E ⊆ V × V is the set of edges with corresponding adjacency matrix A ∈ {0, 1} N ×N , X = {x 1 , x 2 , . . . , x N } is the set of textual features with each x i corresponding to node v i ∈ V, and Y = {y 1 , y 2 , . . . , y N } represents the set of node labels where y i ∈ {1, 2, . . . , C} for C distinct classes.
this section cite: []

Section: Knowledge Extraction.
Given the context length constraints of LLMs, we define a knowledge extraction function Φ : G → K that samples a representative subgraph as:
K = Φ(G) = (V k , E k , X k , Y k ),(1)
where
V k ⊂ V, E k = {(v i , v j ) ∈ E | v i , v j ∈ V k },
and X k , Y k are the corresponding text attributes and labels. The extraction function Φ employs specialized sampling strategies to ensure K captures both structural and semantic characteristics of G while remaining within LLM context limits.
this section cite: []

Section: Graph Synthesis Process.
The graph synthesis process is formalized as a function Ψ : K → G s that generates new graph elements based on the extracted knowledge:
G s = Ψ(K) = (V s , E s , X s , Y s ),(2)
where G s represents the synthesized graph components. Function Ψ is implemented through our framework that encompasses both semantic understanding and structural pattern recognition.
this section cite: []

Section: Graph Synthesis.
The final enhanced graph merges the original and synthesized components:
G new = G ⊕ G s = (V ∪ V s , E ∪ E s ∪ E c , X ∪ X s , Y ∪ Y s ),(3)
where
E c = {(v i , v j ) | v i ∈ V, v j ∈ V s }
this section cite: []

Section: The Proposed Method
We present GraphMaster, a multi-agent framework conceptualized through the lens of Retrieval-Augmented Generation (RAG) [50,54] to address the challenges of graph synthesis in dataconstrained environments. As illustrated in Figure 1, GraphMaster implements a hierarchical RAG paradigm wherein four specialized LLM-powered agents operate collaboratively in a recursive optimization loop to generate semantically rich and structurally coherent graph extensions.
this section cite: ['b49', 'b53']

Section: Framework Overview: RAG-Based Multi-Agent Architecture
GraphMaster formalizes graph synthesis as an iterative RAG process, operating through specialized agents in a closed-loop optimization system. While a single LLM might possess the theoretical capability to understand graph structures, the inherent complexity of generating coherent graph data necessitates a specialized multi-agent approach for three critical reasons. First, real-world graphs substantially exceed typical LLM context windows, requiring strategic sampling and knowledge extraction. Second, maintaining structural consistency across generated elements demands focused attention on connectivity patterns that single-pass generation cannot guarantee. Third, controlling hallucination requires continuous evaluation and refinement through iterative feedback. A collaborative multi-agent architecture effectively addresses these challenges through specialization and integration:
this section cite: []

Section: Manager Agent

this section cite: []

Section: Agent initialization mode choose

this section cite: []

Section: Semantic Topological

this section cite: []

Section: Perception Agent
G new = Ψ RAG (G, Q, R, A retrieve , A generate , A evaluate ) (4
)
where G is the original graph, Q represents the query formulation (enhancement mode), R denotes the retrieval strategy, and A retrieve , A generate , and A evaluate correspond to the agent-specific functions for retrieval, generation, and evaluation, respectively. In each iteration, the Manager Agent formulates the query to guide the synthesis process, the Perception Agent retrieves relevant context to overcome context window limitations, the Enhancement Agent generates new content while maintaining structural consistency, and the Evaluation Agent assesses quality and addresses potential hallucinations-with this cycle continuing until convergence. This collaborative framework enables each agent to focus on a specific challenge while collectively producing coherent graph extensions that a single-pass approach cannot achieve.
this section cite: []

Section: Manager Agent: Query Optimization and Control Mechanism
The Manager Agent serves as the meta-cognitive controller that formulates the synthesis query Q t at iteration t based on a comprehensive environmental status report R t = LLM P (G t ) generated by the Perception Agent. The mode selection function is formalized as: M t = LLM M (R t ) ∈ {semantic, topological} where LLM M represents the Manager Agent's reasoning process that analyzes community structures and label distributions captured in R t . This query formulation implements an adaptive mechanism where the Manager optimizes a multi-objective utility function:
ω * t = arg max ω∈Ω [λ 1 U sem (ω, G t ) + λ 2 U struct (ω, G t ) + λ 3 U bal (ω, G t )] (5
)
where Ω is the strategy space, U sem , U struct , and U bal represent semantic coherence, structural integrity, and class balance utilities respectively, with adaptive weights λ i that evolve according to:
λ t+1 i = λ t i + η∇ λi P (G t )
where P (G t ) measures synthesis progress and η is a learning rate. The Manager orchestrates state transitions, modeled as s t+1 = T (s t , a t , M t ), where states s t reflect graph composition and actions a t ∈ {a P , a E , a V } correspond to agent invocations for Perception, Enhancement, and Evaluation respectively.
this section cite: []

Section: Perception Agent: Context-Aware Corpus Retrieval
The Perception Agent implements the retrieval component of the RAG paradigm, extracting a relevant subgraph from the input graph G t based on the query Q t = M t . This retrieval process is formalized as: K t = R(G t , Q t ) = (V k , E k , X k , Y k ) where K t represents the retrieved knowledge capsule. The retrieval function R operates through three sequential stages: Semantic-aware Community Identification: The agent employs a semantic-enriched modularity maximization algorithm to calculate the community distribution of semantic associations for TAG:
Q sem = 1 2m i,j A ij -γ k i k j 2m -(1 -γ) d sem (x i , x j ) l,m d sem (x l , x m ) δ(c i , c j )(6)
where d sem (x i , x j ) = xi•xj ∥xi∥∥xj ∥ computes semantic similarity between node attributes, k i and k j represent the degrees of nodes i and j, and γ balances topological and semantic factors.
this section cite: []

Section: Mode-Adaptive Seed Selection Strategy:
Based on the enhancement mode M t , the agent selects an optimal seed community C b :
C b =    arg min i |C i | • (1 + µ • Var({x j : v j ∈ C i })) , if M t = semantic, {v j ∈ V train : y j = arg max c ϕ imbal (c)}, if M t = topological (7
)
where ϕ imbal (c) = max c ′ |V c ′ |/|V c | quantifies class imbalance, Var denotes the variance of node textual features within a community and µ weights semantic variance importance. For semantic synthesis, the smaller communities with low internal semantic variance are prefer to be selected to establish a cohesive foundation. For topological synthesis, nodes from minority classes are prioritized.
this section cite: []

Section: Hierarchical Stochastic Diffusion Sampling:
The agent employs a mode-conditional Personalized PageRank (PPR) algorithm, defined as π (k+1) = αv + (1 -α)W T π (k) , where the teleportation vector v varies by enhancement mode:
v i = 1 |C b | if v i ∈ C b and M t = semantic, or v i = I[yi=ŷ] j I[yj =ŷ] if v i ∈ V train
and M t = topological, where ŷ = arg min c |{v j ∈ V train : y j = c}| identifies the label with minimal representation. Following diffusion convergence, the final knowledge subgraph is selected as:
K t = {v i ∈ S top-K% : r i < min(1, β • π i / max j π j )} ∩ S diverse (8
)
where |K t | = N is constrained by the LLM context window, S top-K% contains the top K% nodes by PPR score, r i ∼ Uniform(0, 1) introduces controlled stochasticity, and S diverse ensures community coverage. The environmental status report R t encapsulates multi-scale graph properties:
R t = ρ global , {ρ c class } C c=1 , {ρ i comm } |C| i=1 , D struct , D sem (9
)
where ρ global captures global statistics, ρ c class and ρ i comm encode class-level and community-level properties, while D struct and D sem represent structural and semantic distributions.
this section cite: []

Section: Enhancement Agent: Context-Conditioned Generation
The Enhancement Agent implements the generation component of the RAG paradigm, synthesizing new graph elements (no more than M% of the knowledge subgraph) based on the retrieved knowledge and environmental report. The synthesis process follows Eq. ( 2) where K = (K t , R t , M t ). For semantic mode, the LLM generates node attributes using a conditional autoregressive model:
P (x s |K) = L i=1 P (x i s |x <i s , X k , E k , K)(10)
where x i s is the i-th token of attribute x s , and L is the sequence length. This formulation enables the LLM to generate coherent textual attributes that maintain consistency with the knowledge subgraph while introducing appropriate variations.
Crucially, regardless of the current enhancement mode, the agent always generates both node attributes and their connections. For topological mode, the LLM models edge connections between new node v s and existing nodes by estimating the probability:
P ((v s , v i ) ∈ E c |K) = σ θ 1 • sim(x s , x i ) + θ 2 • |N (v i ) ∩ N K (v s )| |N K (v s )| + θ 3 • k i max j k j (11
)
where N (v i ) is the neighborhood of v i , N K (v s ) represents neighbors of v s in the knowledge subgraph, and σ is the sigmoid function. The coefficients {θ j }foot_0 j=1 are dynamically adjusted based on the query mode M t . This dual-mode generation enables GraphMaster to adaptively emphasize either semantic coherence or structural fidelity while maintaining integrity across both dimensions.
this section cite: []

Section: Evaluation Agent: Multi-dimensional Quality Assessment
The Evaluation Agent implements a comprehensive verification mechanism that integrates four critical information sources:
Q t = LLM V (R 0 , R t , K t , G t s )(12)
where Q t represents the quality assessment outcome, R 0 is the initial environmental report serving as a baseline, R t is the current environmental report, K t is the retrieved knowledge, and G t s is the newly synthesized data. The Evaluation Agent simultaneously assesses two key dimensions: (i) Semantic Coherence: Evaluates whether the generated textual attributes are contextually appropriate, domainconsistent, and meaningful within the graph's thematic scope. (ii) Structural Integrity: Assesses whether the new edges form logical connections that preserve the original graph's topological patterns while addressing structural gaps.
For each generated node v s ∈ V t s , the LLM computes a composite quality score, with the final accepted node set defined as:
V t accepted = {v s ∈ V t s : LLM V (v s , R 0 , R t , K t ) > τ t } (13
)
where the threshold τ t is adaptively updated:
τ t = τ t-1 + ζ( Ft (ω * t ) -Ft-1 (ω * t-1 ))
with Ft (ω * t ) representing the average quality score at iteration t. The convergence determination employs a temporal quality gradient analysis:
Converged t = I max j∈{1,...,k} | Ft (ω * t ) -Ft-j (ω * t-j )| < ϵ ∧ LLM goal (R 0 , R t ) = True (14
)
where I(•) is the indicator function and LLM goal assesses whether synthesis objectives have been achieved. If convergence is detected, the Evaluation Agent signals task completion to the Manager Agent; otherwise, it triggers another iteration of the synthesis process 3 .
this section cite: []

Section: Time Complexity Analysis
The time complexity of GraphMaster is dominated by three operations: (
1) community detection and PPR computation in the Perception Agent, which run in near-linear time O(|V t | + |E t |) on the current graph; (2) LLM inference for node attribute generation and edge probability estimation, which scales with the size N of the retrieved subgraph rather than the full graph; and (3) quality assessment, which evaluates a fixed number of newly generated nodes against predetermined criteria. Since N is constrained by the LLM context window and typically small relative to |V t |, the LLM operations remain efficient regardless of overall graph size. If the iterative process runs for T iterations before convergence (generally small due to the Evaluation Agent's stringent criteria), the overall complexity is T times the per-iteration cost. This architecture enables GraphMaster to scale effectively by leveraging LLMs for semantic generation on bounded contexts while using efficient graph algorithms for structural computations.
this section cite: []

Section: Experiment
To evaluate GraphMaster comprehensively, we formulate four research questions: (RQ1): Can GraphMaster generate high-quality text-attributed graph data in data-limited environment? (RQ2): Can the graph data synthesized by GraphMaster retain the original graph features well? (RQ3): Can GraphMaster maintain interpretability well? (RQ4): What is the relative contribution of each component in GraphMaster to the overall synthesis quality?
this section cite: []

Section: Overall Performance (RQ1)
We evaluate GraphMaster's ability to synthesize high-quality graph data by applying it to enhance the data-limited datasets we created and assessing whether the enhanced datasets improve downstream model performance. We employ standard metrics including Accuracy and F1 Score as evaluation criteria, with higher values indicating superior performance.
Baselines and Datasets. The comparative baselines are categorized into five groups: (1) original TAG training without data synthesis; (2) Classic data augmentation methods: GAugO [63];
(3) LLMbased data aigmentation methods: GraphEdit [16] and LLM4RGNN [60]; (4) Classic data synthesis methods: GraphSmote [62], G-Mixup [18], IntraMix [64], GraphAdasyn [29], FG-SMOTE [42], and AGMixup [30]; (5) LLM-based data synthesis methods: GAG [21] and LLM4NG [51], noting that there are very limited baselines for TAG synthesis using LLM, and we created these two additional baselines named Mixed-LLM and Synthesis-LLM, whose implementations can be found in the Appendix B. Our experiments utilize six widely recognized text-attributed graph datasets: Cora [32], Citeseer [13], Wikics [10], Arxiv2023 [36], and History and Children [49]. It is worth noting that in order to better simulate the data-limited environment to test the effect of data synthesis, we created 6 data-limited datasets, namely SubCora, SubCiteseer, SubWikics, SubHistory, SubArxiv2023, and SubChildren (details are given in Appendix C). In this article, unless otherwise specified, we assume that the augmentation-based method uses the original dataset, while the synthesis-based method uses the data-limited dataset we created. For downstream task evaluation, we implement four established graph neural network architectures: GCN [22], JKNET [46], GraphSage [17] and GAT [38].
this section cite: ['b62', 'b15', 'b59', 'b61', 'b17', 'b63', 'b28', 'b41', 'b29', 'b4', 'b20', 'b50', 'b31', 'b12', 'b9', 'b35', 'b48', 'b21', 'b45', 'b16', 'b37']

Section: Implement Details.
We ran the entire experiment on eight 80G A100 GPUs, using the QwQ-32B model [37] as the base LLM and enabling it to assume different agent roles through iterative calls. For the background knowledge nodes, we set N = 30, and for the newly generated nodes, we configured M % = 15% (The hyperparameter selection analysis are given in Appendix E). In training the GNN model, we first initialized the text attributes with Sentence-BERT [35] to generate the initial features before proceeding with training. To ensure the robustness of our experiments, we repeated each experiment 50 times and reported the mean and standard deviation of the results.
Table 1: Comparison of GraphMaster with other TAG synthesis methods in GCN model. Best performance is indicated by the bold face numbers, and the underline means the second best. 'Acc' and 'F1' are short for Accuracy and F1 Score, respectively. Type Model Cora Citeseer Wikics History Arxiv2023 Children Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Acc F1 Original Original Model 88.9±1.1 88.5±1.5 78.1±1.1 75.0±0.3 79.7±0.8 77.8±0.3 84.2±0.6 43.1±0.3 76.3±1.0 54.9±0.5 52.6±0.7 32.3±1.5 Classic-Aug GAugO 88.9±0.8 88.0±1.0 78.1±0.7 77.2±0.3 79.9±0.9 77.7±0.7 84.6±1.5 44.7±0.6 76.8±1.4 53.0±0.3 51.8±0.6 33.6±0.7 LLM-Aug GraphEdit 91.0±0.9 89.7±0.6 81.9±0.9 80.8±1.1 82.0±1.1 80.7±1.2 87.6±0.6 45.7±1.3 78.0±0.9 57.8±0.3 54.3±0.7 35.7±1.4 LLM4RGNN 91.2±0.6 88.8±1.1 80.9±1.3 76.6±1.1 83.6±1.4 81.6±1.5 88.9±0.7 48.6±0.4 79.3±1.1 59.1±0.4 55.7±1.4 36.7±0.8 Classic-Syn GraphSmote 88.7±1.4 87.4±1.1 78.1±0.5 74.6±1.4 80.7±1.5 78.6±1.4 84.9±0.5 43.9±0.3 76.2±0.4 55.5±0.4 53.1±0.9 33.2±0.6 G-Mixup 87.4±1.1 87.0±0.7 78.2±0.9 76.8±1.1 79.7±0.4 78.0±0.8 84.6±0.6 43.6±0.6 76.6±0.4 56.5±0.3 53.0±0.6 33.0±0.3 IntraMix 80.9±0.6 82.8±0.7 71.3±0.8 70.7±0.5 73.7±1.0 74.5±0.4 82.4±1.5 42.7±0.6 72.4±1.1 53.9±0.8 45.2±0.9 30.1±0.7 GraphAdasyn 89.2±0.3 88.7±0.5 78.9±1.0 78.4±1.4 80.8±1.2 78.9±0.7 84.6±1.5 46.1±0.8 77.5±0.7 57.0±1.1 53.6±0.5 33.0±0.6 FG-SMOTE 88.9±1.5 87.6±1.0 78.7±0.8 74.7±1.4 81.0±0.8 79.0±1.0 85.0±1.4 44.0±0.9 76.4±1.2 55.8±1.4 53.1±1.5 33.3±0.9 AGMixup 84.7±0.4 86.6±0.4 71.7±0.9 73.2±1.1 78.8±0.9 76.6±1.1 81.8±0.9 42.9±0.3 76.8±1.1 53.7±0.5 53.6±1.0 32.6±0.8 LLM-Syn GAG 91.0±1.2 89.3±0.4 82.8±0.9 80.0±1.5 84.9±1.3 83.2±0.7 88.9±0.5 49.8±0.8 79.9±0.5 59.4±1.0 56.7±0.8 38.0±0.5 LLM4NG 85.9±0.3 84.0±0.2 73.9±0.2 72.0±0.3 72.8±0.1 72.9 ±0.4 82.5 ±0.5 48.4 ±0.3 79.0 ±0.2 61.2 ±0.5 44.6 ±0.2 27.2 ±0.4 Mixed-LLM 89.9±0.5 89.3±0.4 83.5±0.9 81.3±0.8 84.9±0.9 83.4±0.8 89.2±1.3 55.8±0.8 81.4±1.5 61.2±0.9 60.0±0.7 39.6±0.7 Synthesis-LLM 89.8±1.3 89.1±1.0 84.5±1.0 82.7±1.1 84.8±0.4 83.2±0.5 89.4±1.3 53.4±1.3 81.0±1.2 62.3±0.8 60.9±1.0 40.1±0.9 GraphMaster 93.7±1.0 92.5±1.0 88.3±0.9 87.7±1.1 87.9±0.8 86.8±0.9 92.6±1.3 63.4±1.4 87.9±1.3 66.3±1.5 68.8±1.4 47.8±1.3
Results. As shown in Table 1 (Due to space limitation, other three models' results are given in Appendix F.), GraphMaster consistently outperforms all baselines, demonstrating the superiority of our approach. Notably, we observed that some baseline methods even yield lower performance than the original dataset. This is primarily because traditional graph synthesis techniques fail to capture the semantic nuances of sentences; consequently, when using Sentence-BERT embeddings instead of bag-of-words representations, their effectiveness is significantly diminished. Moreover, the other LLM-based baselines we compared against mainly focus on anti-interference detection or data synthesis on other scenarios rather than TAG data synthesis, resulting in their performance being significantly lower than that of GraphMaster, which targets TAG data synthesis. Finally, the two LLMbased TAG synthesis baselines we developed show significant advantages over traditional baselines. However, since they cannot fully understand the semantics and topological structure of TAG, although they are higher than other baselines, they are still significantly lower than GraphMasterfoot_1 .
this section cite: ['b36', 'b34']

Section: Synthetic Graph Feature Analysis (RQ2)
100 101 102 103 Node Degree 10 5 10 4 10 3 Our second research question examines whether the new graph data generated by GraphMaster in data-limited environments can maintain consistency with the original graph's structural features.
We conducted a comprehensive analysis across three dimensions: degree distribution, clustering coefficient, and label homogeneity. As shown in Figure 2, GraphMaster demonstrates excellent performance in preserving the network's topological backbone. For example, the two-sample Kolmogorov-Smirnov test statistic between the degree distributions of the original and synthesized Children graphs is 0.357 (p = 0.059), indicating no statistically significant difference. The clustering coefficient similarity score is 0.835, which represents a substantial improvement over the original data-limited Children graph (0.785). Concurrently, the label homogeneity similarity reaches an impressive 0.988 (the heatmap of label-label connection frequencies is almost identical for original vs. synthetic), indicating minimal differences in class mixing patterns. These characteristics show that GraphMaster can generate high-quality synthetic graphs that retain key structural properties of the original data. (Additional graph comparison figures are provided in Appendix G.)
this section cite: []

Section: Interpretability Analysis (RQ3)
To evaluate the transparency of our GraphMaster model, we conduct both human-centered and algorithmic assessments of interpretability (theoretical details in Appendix H). For human evaluation, 50 expert reviewers rated 200 synthesis instances across three dimensions: process transparency, decision justification, and outcome predictability. The overall Traceability Score quantifies how well humans understand the generation process:
T score = 1 R • N R r=1 N i=1 t r,i ,(15)
where t r,i represents the score given by reviewer r for instance i. In parallel, we leverage a Grassmann manifold-based approach to systematically assess semantic consistency of synthesized nodes. This mathematical framework provides a principled way to measure how well generated nodes align with the semantic direction of background knowledge, yielding coherence scores in the range [0, 1].
Our human evaluation results demonstrate that GraphMaster exhibits excellent interpretability, with an average traceability score of T score = 0.92, significantly outperforming Mixed-LLM (0.66) and Synthesis-LLM (0.59). For the Grassmann manifold-based evaluation method, Figure 3a shows that GraphMaster significantly outperforms comparative methods in terms of semantic coherence, indicating that GraphMaster can generate nodes highly aligned with the principal semantic direction.
The score distribution more concentrated in higher value regions also indicates stronger consistency in the quality of generated nodes. Figure 3b shows that GraphMaster maintains high performance across all datasets, consistently exceeding the interpretability threshold of 0.7. Figure 3c demonstrates a strong correlation between human ratings and semantic coherence scores (r=0.78, p<0.00001), further validating our Grassmann manifold-based approach as an effective metric for measuring interpretability. This alignment between human judgment and geometric measures confirms the practical relevance and feasibility of our mathematical framework. 93.7 ± 1.0 92.5 ± 1.0 88.3 ± 0.9 87.7 ± 1.1 87.9 ± 0.8 86.8 ± 0.9 92.6 ± 1.3 63.4 ± 1.4 87.9 ± 1.3 66.3 ± 1.5 68.8 ± 1.4 47.8 ± 1.3 Qwen-32B 91.4 ± 0.9 90.2 ± 0.9 86.5 ± 1.1 85.4 ± 1.0 85.7 ± 0.9 84.2 ± 0.9 90.1 ± 1.2 60.8 ± 0.9 85.3 ± 0.8 64.2 ± 1.1 66.5 ± 1.1 45.9 ± 1.0 DeepSeek-R1-32B 91.8 ± 0.8 90.6 ± 0.8 86.8 ± 1.0 85.8 ± 1.1 85.9 ± 0.8 84.6 ± 0.9 90.5 ± 1.2 61.2 ± 0.9 85.6 ± 1.0 64.5 ± 1.2 66.9 ± 1.2 46.1 ± 1. 4 LLaMA-33B 91.1 ± 0.9 90.0 ± 1.1 86.0 ± 0.8 85.0 ± 1.2 85.4 ± 0.5 84.0 ± 0.6 89.8 ± 0.7 60.5 ± 0.9 85.0 ± 0.7 63.8 ± 0.8 66.2 ± 0.9 45.7 ± 0.9 w.o Perception Agent 88.5 ± 0.8 87.3 ± 0.7 83.6 ± 0.7 82.5 ± 1.1 83.2 ± 0.7 82.0 ± 0.6 87.4 ± 1.1 57.9 ± 0.9 82.6 ± 0.8 61.5 ± 0.9 63.2 ± 1.3 43.4 ± 0.7 w.o Evaluation Agent 89.6 ± 0.7 88.5 ± 0.7 84.8 ± 0.6 83.9 ± 0.9 84.3 ± 0.9 83.1 ± 0.6 88.9 ± 0.7 59.2 ± 0.7 83.8 ± 0.9 62.7 ± 1.1 64.9 ± 1.1 44.6 ± 1.3 N=20 91.5 ± 0.8 90.3 ± 0.5 86.4 ± 0.7 85.3 ± 0.8 85.6 ± 0.6 84.3 ± 0.7 90.0 ± 0.8 60.7 ± 0.8 85.1 ± 0.7 63.9 ± 0.7 66.3 ± 0.8 45.7 ± 0.6 N=30 93.7 ± 1.0 92.5 ± 1.0 88.3 ± 0.9 87.7 ± 1.1 87.9 ± 0.8 86.8 ± 0.9 92.6 ± 1.3 63.4 ± 1.4 87.9 ± 1.3 66.3 ± 1.5 68.8 ± 1.4 47.8 ± 1.3 N=40 92.0 ± 0.9 90.8 ± 0.9 86.7 ± 0.8 85.7 ± 1.0 85.8 ± 0.7 84.6 ± 0.7 90.3 ± 1.2 61.0 ± 1.3 85.4 ± 1.2 64.1 ± 1.4 66.7 ± 1.3 46.0 ± 1.2 M=10% 91.7 ± 0.9 90.5 ± 0.9 86.6 ± 0.7 85.5 ± 0.9 85.7 ± 0.6 84.4 ± 0.7 90.2 ± 1.1 60.8 ± 1.2 85.3 ± 1.1 64.0 ± 1.3 66.6 ± 1.2 45.8 ± 1.1 M=15% 93.7 ± 1.0 92.5 ± 1.0 88.3 ± 0.9 87.7 ± 1.1 87.9 ± 0.8 86.8 ± 0.9 92.6 ± 1.3 63.4 ± 1.4 87.9 ± 1.3 66.3 ± 1.5 68.8 ± 1.4 47.8 ± 1.3 M=20% 91.3 ± 0.8 90.1 ± 0.8 86.2 ± 0.6 85.1 ± 0.8 85.3 ± 0.5 83.9 ± 0.6 90.0 ± 0.7 60.5 ± 0.8 84.8 ± 0.6 63.5 ± 0.6 66.0 ± 0.7 45.3 ± 0.5
this section cite: ['b3']

Section: Ablation Study (RQ4)
In this section, we investigate the relative importance of various components within the GraphMaster framework and their impact on synthesis quality. We systematically analyze how different agent configurations affect the overall performance. We selected Qwen-32B [1], Deepseek-R1-32B [5] and Llama-33B [4], three models with parameters around 32B, as comparison models. Additionally, we examine how varying the size of the background knowledge base (N = |K|) and the percentage of newly generated nodes (M %) influences synthesis effectiveness. We trained the model using GCN on six datasets, and the results are presented in Table 2. Our findings indicate that the performance varies significantly across different LLMs, with QwQ-32B consistently outperforming the alternatives by 1.5-2.3% across all datasets. Notably, DeepSeek-R1-32B achieves the second-best performance despite LLaMA-33B having more parameters, suggesting that model architecture and pre-training approach are more critical than raw parameter count for this task.
The ablation results reveal that removing either the Perception Agent or Evaluation Agent substantially degrades performance (by 5.2% and 4.1% on average, respectively), with the Perception Agent proving particularly crucial. This confirms that both specialized components play essential roles in maintaining generation quality and cannot be omitted from the framework. Regarding hyperparameters, we observe that N = 30 consistently outperforms both smaller (N = 20) and larger (N = 40) knowledge bases across all datasets. Similarly, setting M = 15% yields optimal results compared to both M = 10% and M = 20%. These findings demonstrate that while sufficient context is necessary for high-quality synthesis, excessive background knowledge can dilute the model's focus. Likewise, generating too many nodes simultaneously reduces overall quality due to limitations in the model's generative capacity when handling multiple interdependent elements.
this section cite: ['b0', 'b4', 'b3']

Section: Conclusion
In this paper, we introduced GraphMaster, the first multi-agent framework for text-attributed graph synthesis that successfully addresses the critical bottleneck of data scarcity in training GFMs. By orchestrating specialized LLM agents in a hierarchical RAG paradigm, our approach systematically overcomes the limitations of traditional synthesis methods, generating semantically rich and structurally coherent graph extensions even in severely data-constrained environments. Beyond the framework itself, we created specialized data-limited variants of six standard graph benchmarks and developed a novel dual-perspective interpretability assessment methodology that combines expert human evaluation with a theoretically grounded Grassmannian manifold-based analysis. Comprehensive experiments demonstrate GraphMaster's consistently superior performance across diverse datasets and downstream GNN architectures. Future work could explore multi-scale synthesis approaches that simultaneously model global topology and local semantics, knowledge transfer mechanisms from data-rich to data-limited domains, and adaptive sampling strategies optimized specifically for synthesis objectives. This work not only provides an immediate solution to the graph data scarcity problem but also establishes foundational methodologies for advancing interpretable graph data synthesis.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2023)
Ref_id:b1 Title: Rethinking clientoriented federated graph learning Year: (2025)
Ref_id:b2 Title: Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps Year: (2024)
Ref_id:b3 Title: A sentient ai assistant trained in philosophy, psychology, and personal relationships Year: (2024)
Ref_id:b4 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b5 Title: Outlier summarization via human interpretable rules Year: (2024-03)
Ref_id:b6 Title: Data augmentation for deep graph learning: A survey Year: (2023)
Ref_id:b7 Title: Graphoracle: A foundation model for knowledge graph reasoning Year: (2025)
Ref_id:b8 Title: Mixture of length and pruning experts for knowledge graphs reasoning Year: (2025)
Ref_id:b9 Title: Benchmarking graph neural networks Year: (2020)
Ref_id:b10 Title: Democratizing large language model-based graph data augmentation via latent knowledge graphs Year: (2025)
Ref_id:b11 Title: Autotransfer: Instance transfer for cross-domain recommendations Year: (2023)
Ref_id:b12 Title: Citeseer: An automatic citation indexing system Year: (1998)
Ref_id:b13 Title: Decoupling continual semantic segmentation Year: (2025)
Ref_id:b14 Title: Octopus: Agentic multimodal reasoning with six-capability orchestration Year: (2025)
Ref_id:b15 Title: Graphedit: Large language models for graph structure learning Year: (2025)
Ref_id:b16 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b17 Title: G-mixup: Graph data augmentation for graph classification Year: (2022)
Ref_id:b18 Title: Gpt-gnn: Generative pre-training of graph neural networks Year: (2020)
Ref_id:b19 Title: LLM-Based Multi-Agent Systems are Scalable Graph Generative Models Year: (2025)
Ref_id:b20 Title: Llm-based multi-agent systems are scalable graph generative models Year: (2025)
Ref_id:b21 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b22 Title: Scedit: Script-based assessment of knowledge editing Year: (2025-05)
Ref_id:b23 Title: Graph learning in the era of llms: A survey from the perspective of data, models, and tasks Year: (2024)
Ref_id:b24 Title: Toward data-centric directed graph learning: An entropy-driven approach Year: (2025)
Ref_id:b25 Title: Se-agent: Self-evolution trajectory optimization in multi-step reasoning with llm-based agents Year: (2025)
Ref_id:b26 Title: Multifs: Automated multi-scenario feature selection in deep recommender systems Year: (2024)
Ref_id:b27 Title: Graph foundation models: Concepts, opportunities and challenges Year: (2024)
Ref_id:b28 Title: Joint graph augmentation and adaptive synthetic sampling for imbalanced node classification Year: (2025)
Ref_id:b29 Title: Agmixup: Adaptive graph mixup for semi-supervised node classification Year: (2025)
Ref_id:b30 Title: Position: Graph foundation models are already here Year: (2024-05-30)
Ref_id:b31 Title: Automating the construction of internet portals with machine learning Year: (2000)
Ref_id:b32 Title: Sources of hallucination by large language models on inference tasks Year: (2023)
Ref_id:b33 Title: Very Large-Scale Multi-Agent Simulation in AgentScope Year: (2024)
Ref_id:b34 Title: Sentence-bert: Sentence embeddings using siamese BERTnetworks Year: (2019)
Ref_id:b35 Title: Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning Year: ()
Ref_id:b36 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b37 Title: Graph attention networks Year: (2018)
Ref_id:b38 Title: Manifold mixup: Better representations by interpolating hidden states Year: (2019)
Ref_id:b39 Title: Nodeaug: Semi-supervised node classification with data augmentation Year: (2020)
Ref_id:b40 Title: SELF-INSTRUCT: Aligning language models with self-generated instructions Year: (2023)
Ref_id:b41 Title: Fg-smote: Towards fair node classification with graph neural network Year: (2025)
Ref_id:b42 Title: When llms meet open-world graph learning: A new perspective for unlabeled data uncertainty Year: (2025)
Ref_id:b43 Title: Scadyg: A new paradigm for large-scale dynamic graph learning Year: (2025)
Ref_id:b44 Title: Federated prototype graph learning Year: (2025)
Ref_id:b45 Title: Jumping knowledge networks Year: (2018)
Ref_id:b46 Title: Generate-on-Graph: Treat llm as both agent and kg for incomplete knowledge graph question answering Year: (2024)
Ref_id:b47 Title: Videoseg-r1:reasoning video object segmentation via reinforcement learning Year: (2025)
Ref_id:b48 Title: A comprehensive study on text-attributed graphs: Benchmarking and rethinking Year: (2023)
Ref_id:b49 Title: R 2 AG: Incorporating retrieval information into retrieval augmented generation Year: (2024-11)
Ref_id:b50 Title: Leveraging large language models for node generation in few-shot learning on text-attributed graphs Year: (2025)
Ref_id:b51 Title: Rethinking federated graph learning: A data condensation perspective Year: ()
Ref_id:b52 Title: mixup: Beyond empirical risk minimization Year: (2018)
Ref_id:b53 Title: A survey of graph retrieval-augmented generation for customized large language models Year: (2025)
Ref_id:b54 Title: Knowledge graph reasoning with relational digraph Year: (2021)
Ref_id:b55 Title: Emerging drug interaction prediction enabled by flow-based graph neural network with biomedical network Year: (2023)
Ref_id:b56 Title: Adaprop: Learning adaptive propagation for graph neural network based knowledge graph reasoning Year: (2023)
Ref_id:b57 Title: Efficient hyper-parameter search for knowledge graph embedding Year: (2022-05)
Ref_id:b58 Title: Rethinking graph structure learning in the era of llms Year: (2025)
Ref_id:b59 Title: Can large language models improve the adversarial robustness of graph neural networks? Year: (2025)
Ref_id:b60 Title: Toward general and robust llm-enhanced text-attributed graph learning Year: (2025)
Ref_id:b61 Title: Graphsmote: Imbalanced node classification on graphs with graph neural networks Year: (2021)
Ref_id:b62 Title: Data augmentation for graph neural networks Year: (2021)
Ref_id:b63 Title: Intramix: Intra-class mixup generation for accurate labels and neighbors Year: (2024)
Ref_id:b64 Title: Data augmentation for graph classification Year: (2020)
Ref_id:b65 Title: Towards effective federated graph foundation model via mitigating knowledge entanglement Year: (2025)
Ref_id:b66 Title: Semantic distribution analysis would go here" } Year: ()
