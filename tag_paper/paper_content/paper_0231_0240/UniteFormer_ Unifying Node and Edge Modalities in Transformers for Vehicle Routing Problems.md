Title: UniteFormer: Unifying Node and Edge Modalities in Transformers for Vehicle Routing Problems
Abstract: Neural solvers for the Vehicle Routing Problem (VRP) have typically relied on either node or edge inputs, limiting their flexibility and generalization in real-world scenarios. We propose UniteFormer, a unified neural solver that supports nodeonly, edge-only, and hybrid input types through a single model trained via joint edge-node modalities. UniteFormer introduces: (1) a mixed encoder that integrates graph convolutional networks and attention mechanisms to collaboratively process node and edge features, capturing cross-modal interactions between them; and(2) a parallel decoder enhanced with query mapping and a feed-forward layer for improved representation. The model is trained with REINFORCE by randomly sampling input types across batches. Experiments on the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that UniteFormer achieves state-of-the-art performance and generalizes effectively to TSPLib and CVRPLib instances. These results underscore UniteFormer's ability to handle diverse input modalities and its strong potential to improve performance across various VRP tasks.

Section: Introduction
Vehicle Routing Problems (VRPs) are fundamental in logistics [21], navigation systems [12], and drone delivery [45], with significant theoretical and practical relevance. Recent advances have seen increasing interest in deep learning-based neural solvers for VRPs, offering strong generalization and improved computational efficiency over traditional exact and heuristic algorithms [14,3,4]. These methods include both autoregressive models that learn construction policies from data [24,10,31], as well as learning-based improvement solvers that enhance classical optimization procedures. However, many of these models, particularly construction-based ones, make an overly simplifying assumption: they rely solely on either node coordinates or edge distances as input. This leads to several limitations. First, training separate models for each input modality (node or edge) is inflexible and impractical for real-world applications. Second, switching between different input types requires retraining from scratch, incurring substantial computational costs. Third, such single-modality training neglects the complementary information between node and edge inputs, preventing the model from learning transferable features and reducing its capacity to discover high-quality solutions.
We argue that hybrid training with both node and edge information offers a more general and informative representation of the problem. While existing methods train only on a single modality, our approach allows joint encoding and interaction across modalities, leading to better-informed policies and improved solution quality. To this end, we propose UniteFormer, a unified neural solver for VRPs that supports hybrid training and generalizes across input modalities. Unlike conventional solvers, UniteFormer is trained once and can handle node-only, edge-only, or mixed edge-node inputs without retraining. This makes it more flexible and applicable to diverse real-world scenarios.
Specifically, UniteFormer consists of a mixed encoder and a parallel-attention decoder. The mixed encoder includes two sub-encoders: edge-aware sub-encoder and node-focused sub-encoder. The edge-aware sub-encoder integrates residual gated graph convolutional networks (GCNs) with selfattention to jointly process node and edge features, facilitating cross-modality interaction. The nodefocused sub-encoder encodes node features independently using attention mechanisms, thereby further enhancing the ability to encode node information. Together, they produce rich global embeddings that capture complementary structural information. The decoder features a parallel architecture and nonlinear query mechanisms, incorporating query mapping and a feed-forward (FF) layer to enhance its representational capacity. Our contributions are outlined as follows:
• We present UniteFormer, the first unified neural solver capable of solving VRPs with node-only, edge-only, or hybrid inputs using a single trained model.
• We introduce a novel mixed encoder that combines residual gated GCNs with attention mechanisms, which can effectively and jointly process node and edge features to capture cross-modal interactions between them.
• We design a decoder with a parallel-attention architecture and nonlinear query mechanisms, which enhance the expressiveness of the policy network.
• Experiments on TSP and CVRP with all three input types show that UniteFormer achieves state-of-the-art results. It also generalizes well to real-world TSPLib and CVRPLib benchmarks, and supports applications such as the Asymmetric TSP (Appendix F).
this section cite: ['b20', 'b11', 'b44', 'b13', 'b2', 'b3', 'b23', 'b9', 'b30']

Section: Related Work
Modality-Specific Neural Solvers for VRPs. Neural approaches have emerged as powerful alternatives for solving VRPs by leveraging advances in deep learning and neural combinatorial optimization [3,35,32,47,26,27,17]. The introduction of pointer networks [42] and the Transformer architecture [40] laid the foundation for early neural VRP solvers such as in [2] and [34].
1) Node-based models: Most existing neural solvers focus on node coordinate inputs. Notable examples include AM [22], POMO [24], and Sym-NCO [19], which significantly improved solution quality for classical VRPs. More recent works have advanced training strategies. For instance, Bdeir et al. [1], Drakulic et al. [10], and Luo et al. [31] applied dynamic input re-encoding during training to enhance generalization. Among them, Drakulic et al. [10] introduced Bisimulation Quotienting (BQ) to reformulate the MDP for more robust generalization. Luo et al. [31] proposed a light encoder heavy decoder (LEHD) model trained via supervised learning on partially reconstructed 100-node instances. These methods are all fundamentally built on node coordinate inputs, capturing spatial structure through positional embeddings.
2) Edge-based models: Edge-centric models are a more recent development. Kwon et al. [23] introduced MatNet, a matrix encoding network that operates on pairwise distance matrices, and demonstrated strong performance on the asymmetric traveling salesman (ATSP) and flexible flow shop (FFSP) problems. Lischka et al. [28] proposed GREAT, a sparse graph edge attention model that constructs high-quality solutions by exploiting sparse edge relationships. Building on this, Meng et al. [33] proposed an efficient edge-based EFormer, which further extends and optimizes edge-based problems and achieves excellent results on the TSP and CVRP.
this section cite: ['b2', 'b34', 'b31', 'b46', 'b25', 'b26', 'b16', 'b41', 'b39', 'b1', 'b33', 'b21', 'b23', 'b18', 'b0', 'b9', 'b30', 'b9', 'b30', 'b22', 'b27', 'b32']

Section: 3) Hybrid edge-node models:
A smaller body of work explores models that jointly use node and edge information. Joshi et al. [18] proposed a GCN-based edge probability predictor that uses both node coordinates and edge weights to guide beam search. Wang et al. [44] developed a distanceaware reshaping method (DAR) that biases attention mechanisms using Euclidean distances. Zhou et al. [49] introduced an instance-conditional adaptive model (ICAM) that integrates both node and edge features to improve adaptability across instance sizes. Unified Neural Solvers for VRPs. Unified models that generalize across VRP variants have gained attention due to their versatility and practicality. Wang and Yu [43] proposed a multi-task neural solver using a multi-armed bandit framework to train across multiple combinatorial optimization problems. Drakulic et al. [9] introduced GOAL, a supervised learning-based agent capable of solving diverse COPs. Within the VRP domain, Ruis et al. [37] used attribute composition for zero-shot generalization across multiple VRP variants. Liu et al. [29] extended the reinforcement learning-based POMO model to a multi-task setting (POMO-MTL), and Zhou et al. [50] further proposed MVMOE, a mixture-of-experts model to improve generalization. Building on this, Liu et al.. [30] proposed a curvature-aware pre-training framework that effectively improved their performance. Federico et al. [5] introduced RouteFinder, a modular baseline framework for VRP variant modeling. Our work aligns with this line of research but focuses on input-modality unification rather than task-level generalization. UniteFormer is the first model that simultaneously processes VRPs defined by node, edge, or hybrid representations in a single framework, offering strong generalization, improved efficiency, and broad applicability to real-world VRPs.
this section cite: ['b17', 'b43', 'b48']

Section: UniteFormer
Transformer-based neural VRP solvers typically adopt light decoder architectures [22,24], where the decoder uses static node embeddings as keys and values throughout the attention layers. In contrast, we replace these static embeddings with two context-aware embeddings that encode both edge relationships and node coordinates, and process them in parallel within the decoder. To effectively encode heterogeneous input modalities, we introduce a novel mixed encoder architecture that combines residual gated GCNs with attention mechanisms. The mixed encoder includes two sub-encoders: an edge-aware sub-encoder and a node-focused sub-encoder, collaboratively processing node and edge features to capture cross-modal interactions between them. In addition, we enhance the parallel-attention decoder with query mapping and a feed-forward layer to form our proposed UniteFormer. The overall architecture of UniteFormer is illustrated in Figure 1. In the following, we first present three input modalities in UniteFormer, then introduce the two sub-encoders of the mixed encoder in detail, and finally report the specific implementation of the decoder.
this section cite: ['b21', 'b23']

Section: Input Modalities in UniteFormer
A VRP instance is defined over a graph G = {X, E}, where X = {x i } N i=0 denotes the nodes (with x 0 as the depot), and e(x i , x j ) ∈ E represents the edge between nodes x i and x j . UniteFormer supports three input modalities: edge-only input, where only edge weights E are provided; node-only input, where only node coordinates X are provided; hybrid input, where both node coordinates X and edge weights E are available. The three input configurations are shown in Figure 2. Different input types activate different branches of the edge-aware sub-encoder (left) and the node-focused sub-encoder (right). A single unified model is trained, with encoder components dynamically adapted to each input type. Specifically, Edge-only input (Figure 2(a)): The node-focused sub-encoder is disabled (i.e., replaced with a zero embedding), and only the edge-aware sub-encoder processes edge features; Node-only input (Figure 2(b)): Without edge weights, node features are passed to the GCNs directly, while the edge features are set to zero embeddings. Hybrid input (Figure 2(c)): Both node and edge features are used. The mixed-score Multi-Head Attention (mixed-score MHA) is bypassed, and the GCNs receive the raw edge and node features.
this section cite: []

Section: Edge-aware Sub-encoder: Efficient Fusion of GCNs and Attention Mechanisms
To simultaneously capture edge and node information, we propose a novel sub-encoder architecture, i.e., the Edge-aware Sub-encoder, that integrates GCNs with attention mechanisms. When edge information is provided, we apply a mixed-score MHA block to derive intermediate node features. Across all three input modalities, we then apply residual gated GCNs to jointly process edge and node features in a unified representation space. Finally, we introduce an additional self-attention layer, which proves especially effective in edge-based settings.
Mixed-Score Attention Layer. Inspired by [23], we adopt a multi-head mixed-score attention mechanism to encode the edge weight matrix. This module follows the structure of standard Transformer attention [40], but replaces the traditional scaled dot-product computation with a mixedscore formulation (see Appendix A). Inputs to this block include: a zero vector h 0 , a randomly selected one-hot vector h v from a predefined pool, and the edge weight matrix D ij . This setup allows dynamic embedding generation and facilitates instance-level augmentation by feeding the same instance multiple times with varying vector combinations. The zero vector can optionally be replaced by another one-hot vector, though we default to using the zero vector. The encoded relation matrix h (P ) i is computed only when edge inputs are present:
ĥi (P ) = NORM(h 0 + mixed-scoreMHA(h 0 , h v , D ij )),(1)
h (P ) i = NORM( ĥi (P ) + FF( ĥi (P ) )),(2)
where mixed-scoreMHA(•) denotes the mixed-score attention layer, FF(•) is a feed-forward network with one hidden layer and ReLU activation, and NORM(•) denotes batch normalization [16].
Residual Gated Graph Convolution Layer. We next feed node and edge features into residual gated GCNs. Node coordinates x i are embedded as h-dimensional vectors. The edge weight matrix D ij and the edge adjacency matrix Θ knn ij are embedded as h 2 -dimensional vectors:
α i = ω 1 x i + b 1 ,(3)
β ij = ω 2 D ij + b 2 ||ω 3 • Θ knn ij ,(4)
where ω 1 ∈ R h , ω 2 , ω 3 ∈ R h 2 , b 1 , b 2 are biases, and || denotes vector concatenation. The input node and edge embeddings for the GCN are adaptively initialized according to input types. Particularly, 1) Edge-only:
y 0 i = h(P )
i , e 0 ij = β ij ; 2) Node-only: y 0 i = α i , e 0 ij = h 0 ; 3) Edge and Node:
y 0 i = α i , e 0 ij = β ij .
We denote node and edge embeddings at layer l as y l i and e l ij , respectively. Following [6], we apply ReLU activation and residual connections to obtain the next-layer embeddings:
y l+1 i = y l i + ReLU(NORM(W l 1 y l i + ϕ l ij ⊙ W l 2 y l j )), with ϕ l ij = j∼i σ(e l ij ) j ′ ∼i σ(e l ij ′ ) + ξ , (5
)
e l+1 ij = e l ij + ReLU(NORM(W l 3 e l ij + W l 4 y l i + W l 5 y l j )),(6)
where W l * are learnable weight matrices, σ is the sigmoid function, ξ is a small constant for numerical stability, and ⊙ denotes element-wise multiplication. This formulation enables anisotropic information diffusion on graphs by incorporating learned edge attention maps ϕ l ij . To further process the node embeddings, we apply the MLP to the GCN output y l i , yielding values m l i = MLP(y l i ) constrained to [0, 1] 2 .
this section cite: ['b22', 'b39', 'b15', 'b5']

Section: Self-Attention Layer.
To enhance the model's capacity for global context encoding, we introduce an additional self-attention layer after the MLP. This is particularly important in the edge-only setting, where the node-focused sub-encoder is disabled. In such cases, this layer significantly improves the model's ability to propagate and transform information across the graph:
h M L = self-attention(m L ),(7)
where m L is the MLP output, and h M L is the final output of the edge-aware sub-encoder. The detailed computational process in self-attention layer is provided in Appendix A.
this section cite: []

Section: Node-focused Sub-encoder: Expressive Encoding of Node Information
The edge-aware sub-encoder, which is built on GCNs and augmented with a self-attention layer for encoding, offers a significant boost for edge-based input. However, it falls short in handling node features compared to the conventional encoder mechanism [24]. Therefore, we introduce the classic attention mechanism (i.e., Node-focused Sub-encoder) to make up for this deficiency, which can effectively improve the performance of node-based input. The node-focused sub-encoder consists of L stacked layers, each comprising two sublayers: a multi-Head attention (MHA) sublayer and a feed-forward (FF) sublayer. Each sublayer incorporates residual connections [13] and layer normalization [16]. Let h (l) i denote the embedding of node i at layer l, and let
H (l) = {h (l) 1 , h (l) 2 , . . . , h (l)
n } represent the node embeddings at layer l. The forward computation at the l-th layer is given by:
ĥ(l) i = NORM h (l-1) i + MHA h (l-1) i , H (l-1) ,(8)
h (l) i = NORM ĥ(l) i + FF ĥ(l) i ,(9)
where MHA(•) denotes the multi-head attention, FF(•) is a feed-forward network, and NORM(•) applies layer normalization. This structure allows the sub-encoder to capture complex dependencies between nodes in a permutation-invariant manner. H (L) represents the final output of the L-th attention layer. Specifically, when the input consists of edges only, the sub-encoder is deactivated, and its output is set as h N L = h (0) ; when the input includes nodes, the output is taken as h N L = H (L) .
this section cite: ['b23', 'b12', 'b15']

Section: Decoder: Parallel-Attention Decoding with Enhanced Query Representation
A critical component of the decoder is the context query vector q, which is used to compute attention scores over node embeddings and generate the probability distribution for the next node. In prior works, q is often constructed as a linear combination of node embeddings, which limits its representational capacity due to its inherent linearity [15]. To better capture contextual dependencies, we design a decoder architecture with two key enhancements: (1) a parallel-attention architecture that separately computes attention scores using two sets of encoded embeddings, and (2) a nonlinear query mechanism that increases the expressive power of the query vector. Specifically, we apply query mapping and a feed-forward network with residual connections following the MHA layers.
Parallel-Attention Decoding. The edge-aware sub-encoder and node-focused sub-encoder produce two global embeddings, denoted as h M L and h N L , where the superscripts M and N indicate the edge-aware and node-focused encoding paths, respectively. During decoding, these embeddings are processed in parallel to obtain the decoder context vectors at decoding step t, given by H M c = [h M  1 , h M t ] and
H N c = [h N 1 , h N t ]
. These vectors are used to form temporary queries:
q M = W M 1 h M 1 + W M 2 h M t ,(10)
q N = W N 1 h N 1 + W N 2 h N t ,(11)
q = q M + q N ,(12)
where W M 1 , W M 2 , W N 1 and W N 2 are learnable matrices that transform the start node embeddings h M 1 , h N 1 and current node embeddings h M t , h N t , respectively. Next, we apply MHA [24] separately to each set of context embeddings to obtain two intermediate outputs:
A M = MHA(q, k M , v M ),(13)
A N = MHA(q, k N , v N ),(14)
where k M , v M and k N , v N are the keys and values derived from h M L and h N L , respectively. These outputs are linearly projected and aggregated:
A 2 = W M 3 A M + W N 3 A N ,(15)
where W M 3 and W N 3 are learnable matrices.
this section cite: ['b14', 'b23']

Section: Query Mapping and Feed-Forward Layer.
To further enrich the query representation, we introduce a query mapping transformation and a feed-forward layer with residual connection:
q ′ = A 2 + QMT(q),(16)
q A = q ′ + FF(q ′ ),(17)
where QMT(•) is a linear projection that maps q to the same dimensionality as A 2 , defined as:
QMT(q) = δ QMT q.(18)
Here, δ QMT is a learnable weight matrix. Given the final context vector q A , we compute a score γ j for each node j using a masked single-head attention mechanism:
γ j =    C • tanh q A (k M j +k N j ) √ d k , if j unvisited -∞, otherwise(19)
where d k is the dimensionality of the key vectors, and C is a scaling constant. The final selection probability for node j is computed via the softmax function:
ρ j = softmax(γ j ). (20
)
At each decoding step, a node τ j is sampled according to ρ j . Repeating this process for n steps yields the full solution τ = (τ 1 , ..., τ n ) T . In addition, we report the training algorithm in Appendix B.
this section cite: []

Section: Experiments
We evaluate the performance of UniteFormer on synthetic TSP and CVRP instances of varying sizes, under three input settings: node-only, edge-only, and hybrid. We also report results on standard real-world benchmarks from TSPLib and CVRPLib. The code is publicly available 1 .
Baselines. 1) Traditional Solvers: Concorde [8], LKH3 [14], OR-Tools [25], and HGS [41].
2) Learning-based Solvers: MatNet [23], GREAT [28], POMO [24], LEHD [31], GCN-BS [18], DAR [44] and ICAM [49]. In order to compare POMO with UniteFormer on edge-base input, we re-implement POMO using edge-only inputs (denoted as POMO-edge). More detailed descriptions of these baselines are presented in Appendix D.
this section cite: ['b7', 'b13', 'b24', 'b40', 'b22', 'b27', 'b23', 'b30', 'b17', 'b43', 'b48']

Section: Problem Setting.
We follow the standard data generation procedures from prior work [24] to create training and testing datasets for TSP and CVRP with n = 20, 50, 100, where n denotes the number of nodes. The problem setups and implementation details are presented in Appendix C.
this section cite: ['b23']

Section: Model Setting.
The edge-aware sub-encoder consists of one layer of mixed-score MHA, three layers of GCN and MLP, and one self-attention layer. The node-focused sub-encoder consists of three attention layers. In each attention layer, the head number of MHA is set to 16, the embedding dimension is set to 256, and the feed-forward layer dimension is set to 512.
this section cite: []

Section: Training and Inference.
We use the REINFORCE algorithm [46], training each model for 1,010 epochs with 100,000 instances per epoch. The Adam [20] optimizer is used with an initial learning rate of 4e -4 and weight decay is set to 1e -6 . We adopt the POMO inference algorithm [24] and report both the optimality gap and inference time. A separate set of 10,000 uniformly generated instances is used for testing. All experiments were conducted on a single Tesla V100-SXM2-32GB GPU. More experiment setup details are presented in Appendix D.
this section cite: ['b45', 'b19', 'b23']

Section: Experimental Results
We train a single unified model capable of handling three input types: edge-only, node-only, and hybrid input. Table 1 reports the performance of UniteFormer on uniformly distributed TSP and CVRP instances across various problem sizes and input modalities. UniteFormer consistently outperforms existing learning-based methods in both greedy (×1) and instance-augmented (×8) inference, while maintaining competitive inference times. Additionally, following MatNet [23], we also report results under large-scale augmentation (×128) for edge-based input.
this section cite: ['b22']

Section: TSP.
For edge-based input, UniteFormer significantly outperforms both POMO-edge, MatNet and GREAT across all sizes studied. Notably, its performance with ×8 augmentation exceeds that of MatNet's ×128 augmentation, highlighting the efficiency of the UniteFormer. For node-based input, UniteFormer achieves superior results over node-based neural solvers, including POMO and even the strong LEHD model, in both greedy and ×8 inference. For hybrid edge-node input, UniteFormer also surpasses methods such as GCN-BS, DAR, and ICAM across all scales studied. The advantage is particularly evident on TSP100, where UniteFormer achieves the lowest optimality gap of just 0.0589% among all neural baselines in Table 1. Furthermore, our edge-based UniteFormer even outperforms not only node-based models like POMO, but also hybrid models like DAR and ICAM, demonstrating its strong generalization and representational capacity.
this section cite: []

Section: CVRP.
Similarly, for CVRP, UniteFormer exhibits robust performance across all input types. In the edge-based setting, UniteFormer outperforms both POMO-edge and MatNet in greedy and instance-augmented inference. Its performance with ×8 augmentation even exceeds MatNet's ×128 augmentation results. In the node-based and hybrid settings, UniteFormer again achieves the best results among all compared neural solvers. Specifically, on CVRP100, the hybrid-input version of UniteFormer achieves the lowest average optimality gap of 0.5963%. Notably, in the edge-only setting, UniteFormer even surpasses several node-based or hybrid methods, including POMO, DAR, and ICAM. These results comprehensively demonstrate the effectiveness, robustness, and versatility of UniteFormer across a range of problem sizes and input modalities.
this section cite: []

Section: Ablation Study
Edge only vs. Node only vs. Edge and Node only vs. UniteFormer. Table 2 presents the results of our ablation study comparing UniteFormer with three training variants. At test time, we evaluate models under three input configurations: Input-edge (only edge features are provided), Input-node (only node features are provided), and Input-XE (both edge and node features are available). The first variant, denoted as w.o. UF-Edge, is trained exclusively with edge inputs. The second variant, w.o. UF-Node, is trained only with node inputs. The third variant, w.o. UF-XE, is trained solely on combined edge-node inputs. In contrast, our full UniteFormer model is trained using a hybrid strategy, where the input type (edge, node, or both) is randomly selected for each batch during training. As shown in Table 2, each variant performs well on its respective training input type but shows significant performance drops on other types. In contrast, UniteFormer consistently performs well in all input settings, demonstrating its ability to generalize effectively regardless of the input modality. This result highlights the strength of our unified training approach in producing a robust and versatile model.
Further architectural ablation studies of the three components of the UniteFormer architecture are reported in Appendix E. These experimental results also demonstrate their positive contribution to the UniteFormer, proving the effectiveness and indispensability of each component.
this section cite: []

Section: Generalization
Table 3 summarizes the results on real-world TSPLIB [36] and CVRPLIB [39] instances of various sizes and distributions. We categorize them into three groups by size: N =1-100, N =101-300, and N =301-500. Generalization results show that the UniteFormer performs best on instances with no more than 100 nodes and slightly less effectively on 101-300 node instances. Overall, UniteFormer shows excellent generalization ability on both TSPLIB and CVRPLIB. For TSP, UniteFormer generalizes better with node-only input. For CVRP, it performs exceptionally well with hybrid input.
Additionally, to demonstrate UniteFormer's strong generalization and scalability, we extend our investigation to the Asymmetric Traveling Salesman Problem (ATSP). Due to the asymmetric nature of ATSP, we can naturally solve it using the edge-based UniteFormer framework, which also exhibits superior performance. The detailed experimental results are shown in Appendix F.
this section cite: ['b35', 'b38']

Section: Conclusion, Limitation and Future work
Conclusion: In this work, we propose UniteFormer, a unified neural solver that supports three input types through a single model trained via joint edge-node modalities. We propose a mixed encoder that integrates GCNs and attention mechanisms to collaboratively process node and edge features, capturing cross-modal interactions. Furthermore, we implement a parallel decoding strategy and enhance the decoder's representation ability by adding query mapping and nonlinear layers.
Extensive experimental comparisons with other modality-specific models demonstrate UniteFormer's promising performance. Due to the efficiency and practicality of UniteFormer, we believe it can provide valuable insights and inspire follow-up work to explore more powerful unified neural solvers for edge-node modalities.
Limitation and Future Work: Although our UniteFormer performs well on all three input types, its heavy encoder results in high training and equipment demands, making large-scale problem training challenging. Trimming UniteFormer into a lightweight model for large-scale VRPs [48] is a worthwhile direction for future research. Another promising future work is to extend UniteFormer to solve multi-task VRPs [50,5] with joint modalities.
this section cite: ['b47', 'b49', 'b4']

Section: References
Ref_id:b0 Title: Attention, filling in the gaps for generalization in routing problems Year: (2022)
Ref_id:b1 Title: Neural Combinatorial Optimization with Reinforcement Learning Year: (2017)
Ref_id:b2 Title: Machine learning for combinatorial optimization: a methodological tour d'horizon Year: (2021)
Ref_id:b3 Title: RL4CO: an Extensive Reinforcement Learning for Combinatorial Optimization Benchmark Year: (2025)
Ref_id:b4 Title: RouteFinder: Towards Foundation Models for Vehicle Routing Problems Year: (2025)
Ref_id:b5 Title: Residual gated graph convnets Year: (2017)
Ref_id:b6 Title: The asymmetric traveling salesman problem: Algorithms, instance generators, and tests Year: (2001)
Ref_id:b7 Title: The traveling salesman problem: a computational study Year: (2011)
Ref_id:b8 Title: GOAL: A Generalist Combinatorial Optimization Agent Learner Year: ()
Ref_id:b9 Title: BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization Year: (2023)
Ref_id:b10 Title: A Generalization of Transformer Networks to Graphs Year: (2021)
Ref_id:b11 Title: Optimized path planning for electric vehicle routing and charging station navigation systems Year: (2021)
Ref_id:b12 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b13 Title: An extension of the Lin-Kernighan-Helsgaun TSP solver for constrained traveling salesman and vehicle routing problems Year: (2017)
Ref_id:b14 Title: Rethinking Light Decoder-based Solvers for Vehicle Routing Problems Year: ()
Ref_id:b15 Title: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift Year: (2015)
Ref_id:b16 Title: DRoC: Elevating large language models for complex vehicle routing via decomposed retrieval of constraints Year: ()
Ref_id:b17 Title: An efficient graph convolutional network technique for the travelling salesman problem Year: (2019)
Ref_id:b18 Title: Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization Year: (2022)
Ref_id:b19 Title: Adam: A Method for Stochastic Optimization Year: (2015)
Ref_id:b20 Title: Vehicle routing problem and related algorithms for logistics distribution: A literature review and classification Year: (2022)
Ref_id:b21 Title: Attention, learn to solve routing problems Year: (2019)
Ref_id:b22 Title: Matrix encoding networks for neural combinatorial optimization Year: (2021)
Ref_id:b23 Title: POMO: Policy Optimization with Multiple Optima for Reinforcement Learning Year: (2020)
Ref_id:b24 Title:  Year: ()
Ref_id:b25 Title: Diversity Optimization for Travelling Salesman Problem via Deep Reinforcement Learning Year: (2025)
Ref_id:b26 Title: Cross-problem learning for solving vehicle routing problems Year: ()
Ref_id:b27 Title: A GREAT Architecture for Edge-Based Graph Problems Like TSP Year: (2024)
Ref_id:b28 Title: Multi-task learning for routing problem with cross-problem zero-shot generalization Year: (2024)
Ref_id:b29 Title: A Mixed-Curvature based Pre-training Paradigm for Multi-Task Vehicle Routing Solver Year: (2025)
Ref_id:b30 Title: Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization Year: (2023)
Ref_id:b31 Title: Reinforcement learning for combinatorial optimization: A survey Year: (2021)
Ref_id:b32 Title: EFormer: An Effective Edge-based Transformer for Vehicle Routing Problems Year: ()
Ref_id:b33 Title: Reinforcement Learning for Solving the Vehicle Routing Problem Year: (2018)
Ref_id:b34 Title: Graph learning for combinatorial optimization: a survey of state-of-the-art Year: (2021)
Ref_id:b35 Title: TSPLIB-A traveling salesman problem library Year: (1991)
Ref_id:b36 Title: Independent Prototype Propagation for Zero-Shot Compositionality Year: (2021)
Ref_id:b37 Title: Multi-agent routing value iteration network Year: (2020)
Ref_id:b38 Title: New benchmark instances for the capacitated vehicle routing problem Year: (2017)
Ref_id:b39 Title: Attention is all you need Year: (2017)
Ref_id:b40 Title: Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood Year: (2022)
Ref_id:b41 Title: Pointer Networks Year: ()
Ref_id:b42 Title: Efficient Training of Multi-task Neural Solver for Combinatorial Optimization Year: (2023)
Ref_id:b43 Title: Distance-Aware Attention Reshaping for Enhancing Generalization of Neural Solvers Year: (2025-07)
Ref_id:b44 Title: Vehicle routing problem with drones Year: (2019)
Ref_id:b45 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b46 Title: Learning improvement heuristics for solving routing problems Year: (2021)
Ref_id:b47 Title: Glop: Learning global partition and local construction for solving largescale routing problems in real-time Year: (2024)
Ref_id:b48 Title: Instance-conditioned adaptation for large-scale generalization of neural combinatorial optimization Year: (2024)
Ref_id:b49 Title: MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts Year: (2024)
