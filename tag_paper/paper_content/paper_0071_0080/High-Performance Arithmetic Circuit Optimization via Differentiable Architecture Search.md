Title: High-Performance Arithmetic Circuit Optimization via Differentiable Architecture Search
Abstract: Arithmetic circuit optimization remains a fundamental challenge in modern integrated circuit design. Recent advances have cast this problem within the Learning to Optimize (L2O) paradigm, where intelligent agents autonomously explore high-performance design spaces with encouraging results. However, existing approaches predominantly target coarse-grained architectural configurations, while the crucial interconnect optimization stage is often relegated to oversimplified proxy models or a heuristic approach. This disconnect undermines design quality, leading to suboptimal solutions in the circuit topology search space. To bridge this gap, we present ARITH-DAS , a Differentiable Architecture Search framework for Arithmetic circuits. To the best of our knowledge, ARITH-DAS is the first to formulate interconnect optimization within arithmetic circuits as a differentiable edge prediction problem over a multi-relational directed acyclic graph, enabling fine-grained, proxy-free optimization at the interconnection level. We evaluate ARITH-DAS on a suite of representative arithmetic circuits, including multipliers and multiply-accumulate units. Experiments show substantial improvements over state-of-the-art L2O and conventional methods, achieving up to 27.05% gain in hypervolume of area-delay Pareto frontiers, a standard metric for evaluating multi-objective optimization performance. Moreover, integrating our optimized arithmetic units into large-scale AI accelerators yields up to 6.59% delay reduction, demonstrating both scalability and real-world applicability.

Section: Introduction
Arithmetic circuits, including multipliers and multiply-accumulate units, constitute the computational foundation of modern hardware platforms such as CPUs, GPUs, AI accelerators, and digital signal processors [1][2][3]. These circuits perform essential arithmetic operations that dominate the computational workload across a broad range of compute-intensive applications [4][5][6]. In deep neural networks, for example, multiplication operations account for over 99% of total computation [7]. As machine learning and high-performance computing systems continue to grow in scale and complexity, optimizing arithmetic circuits for both latency and area efficiency has become a critical challenge for enabling scalable and effective AI hardware systems.
Arithmetic circuit optimization poses a fundamental NP-hard challenge in discrete combinatorial optimization. The discrete design space scales exponentially with the bit width, reaching a complexity of O(2 4N 2 ) as detailed in Appendix B. Recent studies have reformulated arithmetic circuit optimization as a Learning to Optimize (L2O) problem [3,[7][8][9], wherein learning-based agents are employed to explore efficient design strategies for performance improvement. These approaches typically initialize from expert-crafted designs, iteratively refine local structures, and leverage performance improvements between successive designs as reward signals to guide the optimization process, achieving promising results.
However, despite the notable successes, existing L2O approaches remain predominantly confined to coarse-grained architectural optimization, which primarily focuses on basic component allocation at each bit position. Fine-grained interconnect routing is still governed by heuristic rules that overlook the complex structural constraints of the design space, thereby compromising circuit quality. Recent efforts [10,11] have attempted to model the interconnection in the arithmetic circuits through permutation matrix generation, leveraging mixed-integer programming (MIP) and differentiable optimization to enhance interconnect assignment. However, these methods are fundamentally constrained by oversimplified proxy formulations, which fail to faithfully reflect complex post-synthesis physical metrics. As circuit complexity increases, the misalignment between proxy and actual performance escalates, causing the optimization to converge prematurely to inferior design solutions.
To address these challenges, we present ARITH-DAS , a differentiable architecture search framework that directly targets the interconnection design space in arithmetic circuits. In contrast to prior work focused on coarse architectural component allocation, ARITH-DAS formulates fine-grained interconnection optimization as an edge prediction task over a multi-relational directed acyclic graph, capturing signal-level structural semantics. It employs multi-relational graph neural networks with attention mechanisms to model connection probabilities across relation types, and adopts a proxyfree objective aligned with post-synthesis metrics to ensure fidelity. ARITH-DAS further integrates with high-level allocation optimizers, enabling unified, multi-granularity circuit optimization.
We conduct systematic evaluations on a suite of representative arithmetic circuits, including multipliers and multiply-accumulate (MAC) units. Experimental results show that our method consistently achieves Pareto dominance over state-of-the-art approaches in area and delay, with hypervolume improvements of up to 27.05%. Furthermore, when integrating the optimized circuits into large-scale AI computing systems, our approach delivers up to 6.59% latency reduction compared to the stateof-the-art baseline designs, demonstrating its scalability and practical engineering applicability.
Contributions of this paper are summarized as follows: (1) We identify key limitations in existing arithmetic circuit optimization approaches and reveal the critical role of the interconnection design space in determining post-synthesis performance. (2) To the best of our knowledge, ARITH-DAS is the first to formulate the fine-grained interconnect synthesis of arithmetic circuits as a differentiable architecture search problem. (3) We propose a novel differentiable architecture search framework tailored for arithmetic circuits, which combines multi-relational graph neural networks with Graphormer-style attention mechanisms. The entire framework is trained in a proxy-free manner, directly aligning with post-synthesis performance metrics. (4) Extensive experimental results across representative circuits demonstrate that our method achieves state-of-the-art post-synthesis performance, while maintaining strong applicability to real-world design flows.
this section cite: ['b0', 'b1', 'b2', 'b4', 'b5', 'b6', 'b2', 'b6', 'b7', 'b8', 'b9', 'b10', 'b0', 'b2']

Section: Preliminary: Arithmetic Circuit Optimization
Arithmetic circuits are typically composed of three main components: the Partial Product Generator (PPG), the Compressor Tree (CT), and the Carry Propagate Adder (CPA). As shown in Figure 1, PPG encodes the inputs into a matrix of partial products based on the arithmetic operations, using schemes such as AND-gate encoding, Booth encoding, or more complicated methods [12][13][14]. Then the compressor tree reduces this matrix to two rows through parallel compression, and finally the CPA adds them to produce the final output. The compressor tree dominates both area and delay of the entire arithmetic circuit, making it the primary target for arithmetic circuit optimization [3,7,15]. The optimization of compressor trees typically involves three key steps:
Compressor Allocation This step determines the number and types of compressors used at each column of the compressor tree. The allocation must satisfy basic constraints such as input/output balance and convergence to two or fewer final rows. It forms the architectural backbone of the compression process and directly influences both the area and delay of the overall circuit.
this section cite: ['b11', 'b12', 'b13', 'b2', 'b6', 'b14']

Section: Compressor Stage Assignment
Given a valid compressor allocation, the stage assignment step specifies the execution order of each compressor, subject to data dependency and topological con- 1.3 1.4 1.5 1.6 1.7 1.8 Delay (ns) 0 1 2 3 4 5 6 7 8 Density Booth-Encoding Multiplier Random routing Default 1.2 1.3 1.4 1.5 1.6 1.7 Delay (ns) 0 1 2 3 4 5 6 7 Density AND-Encoding Multiplier Random routing Default Figure 2: Delay distribution of 1000 randomly interconnected 16-bit multipliers using Wallace Tree compression with both AND and Booth partial product encoding PPG. 0.0 0.2 0.4 0.6 0.8 1.0 Estimated Delay Normalized 0.0 0.2 0.4 0.6 0.8 1.0 Simulated Delay Normalized Booth-Encoding Multiplier 0.0 0.2 0.4 0.6 0.8 1.0 Estimated Delay Normalized 0.0 0.2 0.4 0.6 0.8 1.0 Simulated Delay Normalized AND-Encoding Multiplier straints. This process enforces legal signal propagation through the stages but does not yet define the exact ordering of partial product summation within each column. The resulting assignment determines the temporal alignment of computation without resolving interconnect details.
this section cite: []

Section: Compressor Interconnection
With allocation and stage assignment fixed, the final step involves establishing the logical interconnection between compressors, determining how outputs are routed to inputs across stages. This step has been largely overlooked in existing learning-to-optimize (L2O) frameworks, which typically assume simplistic sequential interconnection. Unlike allocation and stage assignment, which operate on regularized representations, interconnect optimization requires reasoning over graph-structured data and exposes a vast and irregular search space, making it both critical and challenging to optimize effectively.
this section cite: []

Section: Motivation Challenge: Interconnection Optimization
Existing arithmetic circuit optimization frameworks predominantly focus on allocation-level design using regularized structural representations, whereas interconnections between compressors are typically handled through simplistic sequential wiring schemes. However, recent studies reveal that interconnects constitute a critical and nontrivial design space [10,11], with considerable impact on post-synthesis delay. In essence, existing approaches exhibit three fundamental limitations: (1) the neglect of interconnect-level optimization;
(2) dependence on oversimplified heuristics and proxy delay models;
(3) the absence of a unified framework for jointly optimizing coarse-grained allocation and fine-grained interconnect design. We elaborate on these limitations in the following discussion.
this section cite: ['b9', 'b10']

Section: Design Space of Interconnection
The interconnection of compressors plays a critical role in determining the critical path of the compression tree. We illustrate this often-overlooked design space through a randomized experiment. Specifically, we perform random interconnection assignments on 16-bit multipliers using both AND-gate and Booth PPG schemes with fixed compressor allocation, and evaluate the post-synthesis delay of these randomly generated designs. The results are compared against the sequential interconnection strategy adopted by the recent state-of-the-art L2O method MUTE [7]. We present the post-synthesis delay distribution of the randomly routed designs in Figure 2, revealing that with a fixed compressor tree allocation, there exists a substantial timing optimization space of up to over 10%. Notably, the sequential wiring scheme does not exhibit a clear advantage over these randomized interconnection designs.
this section cite: ['b6']

Section: Misaligned Proxy Objective
Existing interconnect optimization methods leverage mixed-integer programming (MIP) [10] or differentiable approaches [11], both relying on oversimplified proxy delay models. However, due to the complex and nonlinear nature of post-synthesis delays, which are closely coupled with the technology library and EDA toolchain, such proxy objectives often fail to reflect the true performance characteristics and lead to suboptimal routing solutions. To illustrate this, we conduct randomized interconnect experiments on 4-to 64-bit multipliers with both AND and Booth PPG. For each bit-width, 50 random interconnected designs are evaluated using the proxy delay model in UFO-MAC [10] as well as the actual post-synthesis delay. As shown in Figure 3, the discrepancy between proxy and actual delay metric increases with bit-width, underscoring the growing inaccuracy of proxy-driven optimization in large-scale designs.
To address these challenges and fully exploit the vast yet underexplored design space of compressor interconnections, we propose ARITH-DAS , a proxy-free differentiable architecture search framework based on graph neural networks. ARITH-DAS optimizes complex compressor-to-compressor routing by enabling accurate and efficient interconnect prediction aligned with post-synthesis metrics. It can be seamlessly integrated with any allocation optimization algorithm, forming a unified and extensible framework for circuit-level design optimization.
this section cite: ['b9', 'b10', 'b9']

Section: Methodology
We begin with an overview of our proposed ARITH-DAS in Section 4.1. Next, we propose our architecture search formulation in Section 4.2, including detailed multi-relational graph representation in Section 4.2.1 and differentiable relaxation in Section 4.2.2. Finally, Section 4.3 provides a detailed description of the core components of our framework, along with the training procedure aligned with post-synthesis performance metrics.
this section cite: []

Section: Overview of Our Framework
As illustrated in Figure 4, our ARITH-DAS framework consists of three principal components. To thoroughly explore the design space of arithmetic circuits, we first construct a coarse-grained compressor allocation using an adaptable circuit evolutionary strategy. This initial configuration is then expanded into a multi-relational directed acyclic graph (DAG), which defines a rich and expressive interconnect search space. A multi-relational graph neural network is employed to encode both topological and semantic relationships among architectural components, while a graph attention mechanism estimates the probabilistic significance of candidate connections. Finally, the entire framework is trained end-to-end via a PPO-inspired algorithm, enabling proxy-free optimization directly guided by post-synthesis performance metrics.
this section cite: []

Section: Architecture Search Formulation for Arithmetic Circuit

this section cite: []

Section: Graph Representation for Compressor Tree
Combinational logic circuits can be naturally represented as directed acyclic graphs (DAGs), exhibiting structural equivalence between the circuit and its graph-based abstraction. In our representation, partial products and compressors are modeled as nodes, and logical dependencies are modeled as directed edges, as illustrated in Figure 1(c). To address the inherent complexity of fine-grained interconnect modeling, we employ a principled graph construction strategy, detailed below.
(1) Topology Ordering via Stage Assignment To preserve acyclicity, a fundamental constraint of combinational logic, we assign each compressor to a specific stage and allow interconnections only between stages of increasing order. This stage-wise ordering induces a valid acyclic circuit topology and establishes explicit dataflow dependencies.
(2) Virtual Nodes for Cross-Stage Connections To address mismatches in input and output port counts that necessitate signal bypassing, we introduce virtual nodes, which serve as auxiliary 1:1 compressor-like entities that propagate signals across stages. This mechanism enforces stage-wise connectivity while preserving topological consistency and simplifying graph construction.
(3) Multi-Relational Graph for Asymmetric Input Semantics While addition is logically commutative, the compressor input ports are physically asymmetric. To capture this asymmetry, we represent the compressor tree as a multi-relational graph [16][17][18], where each edge type corresponds to a specific input port of the target node. This formulation enables the model to distinguish structurally similar yet semantically distinct connections.
More formally, we define the compressor tree as a multi-relational directed acyclic graph [19], denoted by the tuple (V, E, R). Here, V = {v 1 , v 2 , . . . } is the set of nodes, each representing a circuit element. R = {r 1 , r 2 , . . . } is the set of relation types, where each r ∈ R corresponds to a semantically distinct input port. The edge set E ⊆ V × R × V consists of typed directed edges, where each edge (v i , r, v j ) ∈ E indicates that node v i connects to input port r of node v j . The graph can also be encoded as a three-dimensional binary tensor G ∈ {0, 1} |R|×|V|×|V| , where G r,i,j = 1 if and only if (v i , r, v j ) ∈ E. Following previous works [3,7,15,20,21], we represent the allocation of a compressor tree by a matrix s ∈ N+ T ×N , where T is the number of compressor types and N is the input bit width of the compressor tree. Each entry s t,n denotes the number of compressors of type t assigned to column n. We denote the set of valid designs by G and the set of valid structures by S. The detailed design constraints are provided in Appendix C.3.2. Finally, our goal is to maximize a composite objective function:
max G∈G R(G) = -w 1 • area(G) -w 2 • delay(G),(1)
where the objective function R : G → R is the weighted linear combination of post-synthesis area and delay metrics following prior works [3,7,9,15].
this section cite: ['b15', 'b16', 'b17', 'b18', 'b2', 'b6', 'b14', 'b19', 'b20', 'b2', 'b6', 'b8', 'b14']

Section: Differentiable Reformulation for Discrete Search Space
Problem (1) constitutes a large-scale combinatorial optimization challenge, involving highly irregular graph structures and complex design constraints. To address this, we draw inspiration from differentiable architecture search [22][23][24][25][26], which offers a scalable and efficient alternative. By relaxing discrete structural decisions into continuous probability distributions, these methods enable joint optimization over a vast design space within a single forward pass, thereby significantly enhancing search efficiency and scalability.
Specifically, we replace the hard edge assignments in the compressor tree graph G ∈ G with soft, learnable probability distributions over candidate source nodes. Each potential connection is parameterized by a real-valued score and passed through a softmax function, yielding a continuous relaxation G ∈ [0, 1] |R|×|V|×|V| , where Gr,i,j denotes the probability of establishing an edge from node v i , conditioned on edge type r and target node v j , with normalization i Gr,i,j = 1 for each (r, j) pair. This target-centric formulation arises from the structural constraints and multi-relational graph design, which are inherently defined from the perspective of the receiving node. By modeling connection probabilities in this manner, each input port selects its optimal driver from valid candidates, enabling fine-grained, port-specific connectivity through softmax-based selection. This relaxation implicitly defines a distribution π : G → [0, 1] over the graph search space, with corresponding parameter space Π. Following previous works [22,23], the objective is relaxed to
max π∈Π E G∼π R(G) = max s∈S max π(•|s)∈Πs E G∼π(•|s) R(G) ,(2)
where Π s ⊂ Π denotes the set of graph distributions associated with a given allocation s. This naturally forms a two-stage optimization framework where we first determine the allocation of the compressor tree, and next we optimize the corresponding interconnection.
this section cite: ['b21', 'b22', 'b23', 'b24', 'b25', 'b21', 'b22']

Section: ARITH-DAS : Differentiable Architecture Search for Arithmetic Circuit
In this section, we present ARITH-DAS , a novel differentiable architecture search framework tailored to arithmetic circuit optimization. By encoding structural information via a multi-relational graph neural network and modeling interconnect prediction through graph attention, it enables endto-end, gradient-based optimization directly guided by post-synthesis performance metrics.
this section cite: []

Section: Adaptable Allocation Search via Circuit Evolution
The compressor allocation s ∈ S specifies the number of compressors across columns, shaping the coarse-grained structure and reduction stages of the compression tree. This aspect has been extensively studied [3,7,9,10,15,20], with methods ranging from reinforcement learning to mixedinteger programming for optimizing compressor configurations under fixed encoding schemes.
To ensure simplicity, extensibility, and structural diversity, we adopt the evolution-based method introduced in [7]. Specifically, we maintain a population of candidate architectures in an elite pool, each representing a compressor allocation matrix across all columns. New candidates are generated through two key mechanisms: (1) local perturbation, where a small number of compressors are randomly added, removed, or shifted across columns, and (2) substructure crossover, where two parent architectures exchange subregions of their compressor allocation matrices. This evolutionary process promotes exploration of diverse architectural patterns while preserving high-performing solutions across generations. Crucially, it operates independently of the interconnect structure, enabling seamless integration with the differentiable wiring search described in the next stage. Once a compressor allocation configuration is sampled, we construct the corresponding multi-relational graph and optimize its interconnects accordingly.
this section cite: ['b2', 'b6', 'b8', 'b9', 'b14', 'b19', 'b6']

Section: Multi-Relational Graph Attention Based Link Prediction
Multi-Relational Graph Encoder Given a compressor allocation s ∈ S, our objective is to determine its optimal interconnect configuration. To this end, we leverage a multi-relational graph neural network to encode the underlying circuit topology and model the propagation of physical information through the circuit. Each edge type captures a distinct semantic relation, such as a specific input port of a compressor, and is processed independently during message passing, enabling the network to learn asymmetric and relation-specific connection patterns. Concretely, we adopt a multi-relational message passing scheme. At layer l, the hidden state of node v i under relation r is updated by:
h l+1 i,r = j∈Nr(i)∪{i} 1 |N r (i)| • |N r (j)| ϕ l r h l i , h l j , r ∈ R, (3
)
where N r (i) = {v j ∈ V | (v i , r, v j ) ∈ E or (v j , r, v i ) ∈ E} denotes the set of nodes bidirectionally connected to v i under relation r, and ϕ l r is a relation-specific message function at layer l, augmented with self-loops and reverse edges to support comprehensive message flow, while directional semantics are embedded in node features. To integrate information across all relations, we concatenate the relation-specific representations and apply a shared aggregation function γ l :
h l+1 i = γ l Concat r∈R (h l+1 i,r ) .(4)
this section cite: []

Section: Graph Attention Based Edge Prediction
The final node embedding h i encodes the structural context of vertex v i . To predict interconnect probabilities, we employ a Graphormer-style attention mechanism [27], wherein each target node and its candidate source nodes are projected into relationspecific query and key spaces, respectively. Let O = {sum, carry} denote the set of output port types. Given a target node v j and a port pair (o, r) ∈ O × R, the attention score from a candidate source node v i is computed as:
α o,r i,j = exp (W r Q h j ) ⊤ (W o K h i ) • M o,r i,j i ′ ∈[|V|] exp (W r Q h j ) ⊤ (W o K h i ′ ) • M o,r i ′ ,j , (5
)
where W r Q and W o K are linear projections specific to the input and output port types, respectively. The binary mask M o,r i,j ∈ {0, 1} imposes structural constraints by limiting the softmax normalization to valid source-target-port combinations, as detailed in Appendix C.3.2. This formulation conforms to the probabilistic semantics of our relaxation: for each target node and input port type, it defines a distribution over valid source nodes. The target node acts as the query and the source node as the key, ensuring consistency between the attention logits and relaxed edge selection probabilities.
this section cite: ['b26']

Section: Post-Synthesis Alignment via Proxy-Free PPO-Like Training
Given a fixed compressor allocation s, the attention scores α o,r i,j define a structured distribution over interconnect candidates. A valid discrete realization is obtained via sequential sampling along a legal topological order, with the legality mask M o,r i,j dynamically updated to ensure acyclicity, stage consistency, and port compatibility. This process yields a deterministic graph sample G ∼ π θ (• | s), where the probability is factorized as π θ (G | s) = r,i,j,o G r,i,j • α o,r i,j , fully parameterized by the learnable attention logits α o,r i,j . To optimize the interconnect structure with respect to post-synthesis performance metrics, we employ a proxy-free, PPO-like training paradigm [28]. Specifically, we treat π θ as a stochastic policy over the graph space and maximize the expected objective defined by post-synthesis metrics R:
max θ J(θ | s) = max θ E G∼π θ (•|s) R(G) = max θ E G∼π θ (•|s) π θ (G | s) π θ (G | s) R(G) , (6
)
where π θ is a fixed reference policy used for importance sampling. To enforce conservative updates and prevent large deviations from the reference policy, we optimize the clipped surrogate objective:
Ĵ(θ | s) = 1 M M m=1 min (ρ m R(G m ), clip(ρ m , 1 -ϵ, 1 + ϵ)R(G m )) ,(7)
where Gm|s) is the likelihood ratio between the current and reference policies, and G m ∼ π θ (• | s) are i.i.d. sampled graphs. The hyperparameter ϵ controls the trust region for updates. Additionally, we introduce a regularization term to promote structural discreteness and enforce portlevel sparsity constraints of the output ports [11]. The overall loss is defined as:
ρ m = π θ (Gm|s) π θ (
L(θ | s) = i,j∈[|V|], o∈O, r∈R α o,r i,j • (1 -α o,r i,j ) 2 Discretization penalty + i∈[|V|], o∈O   1 - j∈[|V|], r∈R M r,i,j • α o,r i,j   2 Output-port exclusivity constraint . (8
)
The first term penalizes probabilities that lie in the ambiguous region (0, 1), thus promoting discretelike selection. The second term enforces that each output port selects exactly one target input (i.e., normalized to 1). Finally, the complete training objective combines the PPO-style reward maximization with the regularization:
max θ L(θ) = E s∼S Ĵ(θ | s) -ηL(θ | s) ,(9)
where η > 0 controls the trade-off between reward fidelity and structural regularity. A theoremguided discussion is detailed in Appendix C.3.6.
this section cite: ['b27', 'b10']

Section: Experiments
We begin by describing the experimental setup, baseline methods, and evaluation metrics in Section 5.1. The experiments are designed to pursue three primary objectives: (1) Evaluate the effectiveness of ARITH-DAS in optimizing computing multipliers and MAC units across a broad range of input bit-widths (Section 5.2); (2) Assess the generalization ability of ARITH-DAS -optimized multipliers when scaled to large macro designs representative of real-world AI accelerators (Section 5.3); (3) Perform ablation studies to quantify the contributions of individual components within ARITH-DAS and elucidate the rationale behind its design (Section 5.4).
this section cite: []

Section: Experiment Setup
Experimental Setup Our framework leverages OpenROAD [29] for physical implementation. Logic synthesis is performed using Yosys [30] with the Nangate45 technology library [31], while static timing analysis (STA) is conducted through OpenSTA [29]. Functional verification employs Verilator [32] for cycle-accurate simulation. The machine learning pipeline is implemented in Py-Torch [33] and PyTorch Geometric [34], optimized via the Adam algorithm [35]. We evaluate our methodology across eight distinct multiplier architectures, encompassing 8-bit, 16-bit, 32-bit, and 64-bit implementations employing both AND gate-based and Booth encoding-based techniques. More experiment configurations are provided in Appendix C.
380 400 420 440 460 480 Area ( m 2 ) 0.65 0.70 0.75 0.80 0.85 Delay (ns) 8-bit And 1800 2000 2200 2400 2600
Area ( m 2 )
1.1 1.2 1.3 1.4 1.5 Delay (ns) 16-bit And 7000 7500 8000 8500 9000 9500 10000 Area ( m 2 ) 1.4 1.6 1.8 2.0 2.2 2.4 Delay (ns) 32-bit And 28000 30000 32000 34000 36000 38000 40000 42000 Area ( m 2 ) 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 Delay (ns) 64-bit And 400 425 450 475 500 525 550 575 Area ( m 2 ) 0.65 0.70 0.75 0.80 0.85 0.90 Delay (ns) 8-bit Booth 1400 1600 1800 2000 2200 Area ( m 2 ) 1.0 1.1 1.2 1.3 1.4 1.5 Delay (ns) 16-bit Booth 5500 6000 6500 7000 7500 8000 Area ( m 2 ) 1.4 1.6 1.8 2.0 2.2 2.4 2.6 Delay (ns) 32-bit Booth 20000 22000 24000 26000 28000 Area ( m 2 ) 2.0 2.2 2.4 2.6 2.8 3.0 3.2 3.4 Delay (ns) 64-bit Booth Arith-DAS MUTE ArithTreeRL UFO-MAC DOMAC GOMIL Wallace Figure 6: Pareto frontiers of our ARITH-DAS and baselines on eight multiplier design tasks. Comparative Baselines We evaluate our methodology against six representative approaches spanning classical heuristic designs to contemporary learning-to-optimize (L2O) paradigms: (1) Wallace [36]: A foundational heuristic compression technique; (2) UFO-MAC [10] and (3) GOMIL [20]: A solution framework based on integer programming; (4) DOMAC [9]: A differentiable interconnection optimization framework; (5) ArithTreeRL [9]: A leading deep reinforcement learning (RL) architecture; (6) MUTE [7]: The current state-of-the-art hybrid RL-Evolutionary framework. Extended comparative analyses including implementation specifics and hyperparameter configurations are provided in Appendix C.4.
Evaluation Metrics Arithmetic circuit optimization constitutes a canonical multi-objective optimization problem. We employ two established evaluation protocols from multi-objective optimization theory: (1) Pareto Frontier Analysis: Following established methodologies [3,[7][8][9][10][11]15], we simulate diverse design preferences through parametric target delay configurations. This enables comprehensive synthesis of corresponding solutions and visualization of their Pareto frontiers [37].
(2) Hypervolume (HV) Metric: We conduct comparative analysis of solution quality through hypervolume measurements [38] across different Pareto frontiers, quantifying their multi-objective characteristics. More details are provided in Appendix C.3.8. We highlight the superiority of ARITH-DAS through a comparative analysis with six competitive baselines on eight multiplier design problems across a wide range of input sizes. The results in Figure 6 demonstrate that multipliers optimized by ARITH-DAS consistently and significantly outperform designs optimized by all baselines across all eight multiplier design tasks. Moreover, we present the hypervolume (HV) of the Pareto points discovered by ARITH-DAS in Table 1. The results demonstrate that ARITH-DAS achieves a substantial improvement over the previous SOTA, improving the hypervolume by up to 27.05%. Overall, these results demonstrate the strong ability of ARITH-DAS to optimize multipliers, leading to significant reductions in both area and delay. More results, including evaluation on multiply-accumulators (MACs) and visualization of results, are provided in Appendix D.
this section cite: ['b28', 'b29', 'b30', 'b28', 'b31', 'b32', 'b33', 'b34', 'b2', 'b6', 'b7', 'b8', 'b9', 'b10', 'b14', 'b36', 'b37']

Section: Main Evaluation

this section cite: []

Section: Generalization to Large-Scale Computing Circuit
To assess the generalization capability of the optimized computing units, we integrate those produced by ARITH-DAS and baseline methods into six representative AI accelerator circuits from the Koios 2.0 benchmark [39]. In addition, we deploy and evaluate these units within a typical processing element (PE) array architecture, which is widely used in applications, following previous works [7,15,21]. As shown in Figure 5 and Figure 7, circuits incorporating ARITH-DAS -optimized units consistently outperform these baseline counterparts across multiple metrics. These  results demonstrate the strong generalization ability of ARITH-DAS in large-scale, computationintensive circuits, underscoring its potential to enhance the performance of real-world AI chips. We conducted a carefully designed ablation study targeting the multiplier design task. As has been noted, our method is composed of the following key modules: (1) the Circuit Genetic Evolution (CGE) module, (2) the Multi-Relational Graph Encoder (MRG) module, and (3) the PPO-style training module (PPO). To assess the individual contribution of each module within ARITH-DAS , we conduct a comprehensive component-wise analysis focused on the optimization of multiplier architectures. Specifically, we design three ablation experiments by removing each module, respectively, to quantify its impact on overall performance: (1) w/o CGE where circuit genetic evolution is replaced with a simulated annealing approach;
this section cite: ['b38', 'b6', 'b14', 'b20']

Section: Ablation Study
(2) w/o MRG where the multi-relational graph encoder is replaced by a heterogeneous-node graph with uniform edge types;
(3) w/o PPO where the PPO-style loss is replaced by the proxy delay model introduced in previous work [11]. As shown in Table 2, all three modules are critical to the overall performance. Removing the CGE module impairs structural exploration and reduces architectural diversity, underscoring the role of genetic search in design space navigation. Removing the MRG module weakens relational reasoning, confirming the necessity of modeling edge semantics for accurate connectivity inference. Substituting the PPO-based optimization with a proxy delay model degrades performance, highlighting the benefit of end-to-end training aligned with post-synthesis metrics.
this section cite: ['b10']

Section: Conclusion and Limitations
In this work, we propose ARITH-DAS , a differentiable architecture search framework for arithmetic circuit optimization. Experiments on representative arithmetic units show that ARITH-DAS consistently outperforms state-of-the-art baselines in area-delay trade-offs. When deployed in large-scale AI accelerators, it achieves notable timing improvements, demonstrating strong scalability and practical value. These results underscore the effectiveness of ARITH-DAS and offer new directions for optimizing high-performance computing systems.
However, there still remain several limitations. Consistent with prior studies, our current experimental setup remains confined to the post-synthesis stage, which does not account for the intricate backend procedures such as placement, routing, and timing closure. These stages have a profound impact on the circuits final area, timing, and power characteristics, often leading to substantial discrepancies between synthesized and implemented results. Hence, exploring backend-aware optimization strategies for arithmetic circuits constitutes an essential avenue for future research.
this section cite: []

Section: References
Ref_id:b0 Title: A new multiplier using wallace structure and carry select adder with pipelining Year: (2002)
Ref_id:b1 Title: A fast parallel multiplier-accumulator using the modified booth algorithm Year: (2000)
Ref_id:b2 Title: Rl-mul: Multiplier design optimization with deep reinforcement learning Year: (2023)
Ref_id:b3 Title: Digital logic design, 2nd Year: (1987)
Ref_id:b4 Title: An unconventional arithmetic logic unit design and computing in actin quantum cellular automata Year: (2019)
Ref_id:b5 Title: Efficient processing of deep neural networks Year: (2020)
Ref_id:b6 Title: Computing circuits optimization via model-based circuit genetic evolution Year: (2025)
Ref_id:b7 Title: Prefixrl: Optimization of parallel prefix circuits using deep reinforcement learning Year: (2021)
Ref_id:b8 Title: Scalable and effective arithmetic tree generation for adder and multiplier designs Year: (2024)
Ref_id:b9 Title: Ufo-mac: A unified framework for optimization of high-performance multipliers and multiply-accumulators Year: (2024)
Ref_id:b10 Title: Differentiable optimization for high-speed multipliers and multiply-accumulators Year: (2025)
Ref_id:b11 Title: Application-specific arithmetic Year: (2024)
Ref_id:b12 Title: High-order taylor series approximation for efficient computation of elementary functions Year: (2015)
Ref_id:b13 Title: Minimizing coefficients wordlength for piecewise-polynomial hardware function evaluation with exact or faithful rounding Year: (2017)
Ref_id:b14 Title: A hierarchical adaptive multi-task reinforcement learning framework for multiplier circuit design Year: (2024)
Ref_id:b15 Title: Composition-based multi-relational graph convolutional networks Year: (2019)
Ref_id:b16 Title: Mrgat: multi-relational graph attention network for knowledge graph completion Year: (2022)
Ref_id:b17 Title: r-gat: Relational graph attention network for multi-relational graphs Year: (2021)
Ref_id:b18 Title: Modeling relational data with graph convolutional networks Year: (2018-06-03)
Ref_id:b19 Title: Gomil: Global optimization of multiplier by integer linear programming Year: (2021)
Ref_id:b20 Title: Rl-mul 2.0: Multiplier design optimization with parallel deep reinforcement learning and space reduction Year: (2024)
Ref_id:b21 Title: Darts: Differentiable architecture search Year: (2018)
Ref_id:b22 Title: Proxylessnas: Direct neural architecture search on target task and hardware Year: (2018)
Ref_id:b23 Title: Deep differentiable logic gate networks Year: (2006)
Ref_id:b24 Title: Towards next-generation logic synthesis: A scalable neural circuit generation framework Year: (2024)
Ref_id:b25 Title: Pruning large language models with blockwise parameterefficient sparsity allocation Year: (2024)
Ref_id:b26 Title: Do transformers really perform badly for graph representation? Year: (2021)
Ref_id:b27 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b28 Title: Openroad: Toward a self-driving, open-source digital layout implementation tool chain Year: (2019)
Ref_id:b29 Title: Yosys-a free verilog synthesis suite Year: (2013)
Ref_id:b30 Title:  Year: (2008)
Ref_id:b31 Title:  Year: ()
Ref_id:b32 Title: PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation Year: (2024-04)
Ref_id:b33 Title: Fast Graph Representation Learning with PyTorch Geometric Year: (2019-05)
Ref_id:b34 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b35 Title: A suggestion for a fast multiplier Year: (1964)
Ref_id:b36 Title: Multi-criteria decision analysis: methods and software Year: (2013)
Ref_id:b37 Title: The hypervolume indicator: Computational problems and algorithms Year: (2021)
Ref_id:b38 Title: Koios 2.0: Open-source deep learning benchmarks for fpga architecture and cad research Year: (2023)
Ref_id:b39 Title: Some schemes for fast serial input multipliers Year: (1983)
Ref_id:b40 Title: A 32/spl times/24-bit multiplier-accumulator with advanced rectangular styled wallace-tree structure Year: (2005)
Ref_id:b41 Title: A method for speed optimized partial product reduction and generation of fast parallel multipliers using an algorithmic approach Year: (1996)
Ref_id:b42 Title: Parallel prefix adder design Year: (2001)
Ref_id:b43 Title: Conditional-sum addition logic Year: (1960)
Ref_id:b44 Title: A regular layout for parallel adders Year: (1982)
Ref_id:b45 Title: An algorithmic approach for generic parallel adders Year: (2003)
Ref_id:b46 Title: Towards optimal performance-area trade-off in adders by synthesis of parallel prefix structures Year: (2013)
Ref_id:b47 Title: Multi-objective reinforcement learning with adaptive pareto reset for prefix adder design Year: (2022)
Ref_id:b48 Title: Scalable and effective arithmetic tree generation for adder and multiplier designs Year: (2024)
Ref_id:b49 Title: LaMPlace: Learning to optimize cross-stage metrics in macro placement Year: (2025)
Ref_id:b50 Title: Tp-gnn: A graph neural network framework for tier partitioning in monolithic 3d ics Year: (2020)
Ref_id:b51 Title: Ta3d: Timing-aware 3d ic partitioning and placement by optimizing the critical path Year: (2024)
Ref_id:b52 Title: Benchmarking end-to-end performance of AI-based chip placement algorithms Year: (2025)
Ref_id:b53 Title: Convolutional differentiable logic gate networks Year: (2024)
Ref_id:b54 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b55 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b56 Title: Coms2t: A complementary spatiotemporal learning system for data-adaptive model evolution Year: (2025)
Ref_id:b57 Title: An alternating prediction-correction neural solving framework for mixed-integer linear programming Year: (2025)
Ref_id:b58 Title: Junfeng Fang, and Xiang Wang. 3d-gsrd: 3d molecular graph auto-encoder with selective re-mask decoding Year: (2025)
Ref_id:b59 Title: Optimal adjustment sets for nonparametric estimation of weighted controlled direct effect Year: (2025)
Ref_id:b60 Title: Inductive representation learning on large graphs. Advances in neural information processing systems Year: (2017)
Ref_id:b61 Title: Graph attention networks Year: (2017)
Ref_id:b62 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b63 Title: Deeper insights into graph convolutional networks for semi-supervised learning Year: (2018)
Ref_id:b64 Title: On the bottleneck of graph neural networks and its practical implications Year: (2020)
Ref_id:b65 Title: Heterogeneous graph attention network Year: (2019)
Ref_id:b66 Title: Heterogeneous graph transformer Year: (2020)
Ref_id:b67 Title: Graph reinforcement learning for combinatorial optimization: A survey and unifying perspective Year: (2024)
Ref_id:b68 Title: Opensta: Open-source static timing analysis tool Year: ()
Ref_id:b69 Title: Trust region policy optimization Year: (2015)
Ref_id:b70 Title: f-gan: Training generative neural samplers using variational divergence minimization Year: (2016)
Ref_id:b71 Title: On divergences and informations in statistics and information theory Year: (2006)
Ref_id:b72 Title: A general class of coefficients of divergence of one distribution from another Year: (1966)
Ref_id:b73 Title: Wikipedia contributors. f-divergence -Wikipedia, The Free Encyclopedia Year: (2024-05)
Ref_id:b74 Title: Beyond homogeneous graphs, multi-relational/heterogeneous GNNs parameterize typed edges and meta-relations: R-GCN [19] and CompGCN [16] perform relation-specific propagation, while HAN/HGT incorporate relation-aware attention for web-scale heterogeneity [66, 67] Year: (2024)
Ref_id:b75 Title: LGPL-3.0 License 5. ArithTreeRL Year: ()
