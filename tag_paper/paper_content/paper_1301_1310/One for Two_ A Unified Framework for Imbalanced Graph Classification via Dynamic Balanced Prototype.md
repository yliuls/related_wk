Title: Published as a conference paper at ICLR 2026 ONE FOR TWO: A UNIFIED FRAMEWORK FOR IMBAL-ANCED GRAPH CLASSIFICATION VIA DYNAMIC BAL-ANCED PROTOTYPE
Abstract: Graph Neural Networks (GNNs) have advanced graph classification, yet they remain vulnerable to graph-level imbalance, encompassing class imbalance and topological imbalance. To address both types of imbalance in a unified manner, we propose UniImb, a Unified framework for Imbalanced graph classification. Specifically, UniImb first captures multi-scale topological features and enhances data diversity via learnable personalized graph perturbations. It then employs a dynamic balanced prototype module to learn representative prototypes from graph instances, improving the quality of graph representations. Concurrently, a prototype load-balancing optimization term mitigates dominance by majority samples to equalize sample influence during training. We justify these design choices theoretically using the Information Bottleneck principle. Extensive experiments on 19 datasets-including a large-scale imbalanced air pollution graph dataset AirGraph released by us and 23 baselines demonstrate that UniImb has achieved dominant performance across various imbalanced scenarios. Our code is available at GitHub.

Section: INTRODUCTION
Graph Neural Networks (GNNs) have achieved remarkable performance in graph classification tasks by iteratively aggregating local neighborhood information (Wei et al., 2023;Nguyen et al., 2022;Wu et al., 2020;Fan et al., 2023). However, the success of most architectures relies heavily on balanced graph datasets (Ma et al., 2025). In real-world scenarios, graph data often exhibits imbalanced distributions, which can be mainly categorized into two aspects: ❶ Class Imbalance. A few classes contain most of the samples while many classes are severely underrepresented, leading GNNs to favor head classes during training (Tang & Liang, 2023). (Tang & Liang, 2023). A promising approach to mitigate class imbalance is the graph-of-graphs framework (Wang et al., 2022b;Tang & Liang, 2023), which treats each input graph as a node in a higher-order meta-graph. Edges in the meta-graph are formed based on graph-level feature similarity, enabling information propagation among similar graphs and thereby alleviating data sparsity in tail classes.
❷ Topological Imbalance. Graph datasets frequently show a highly skewed distribution of node counts: a small number of large graphs contain most nodes, while the majority of graphs are substantially smaller. This imbalance often causes GNNs to focus disproportionately on larger graphs during training, leading to diminished performance on small-scale ones (Qin et al., 2025). To address this challenge, topology imbalance methods identify underrepresented small-scale graphs during training and enhance their contributions through reweighting or data augmentation, yielding more consistent performance (Liu et al., 2022c;Xu et al., 2024).
Despite remarkable progress, existing imbalanced graph learning methods focus unilaterally on either class imbalance or topological imbalance in isolation. Class-imbalance graph learning methods enhance the expressiveness of underrepresented classes but often overlook the structural heterogeneity within graphs. Conversely, methods focused on topology imbalance adapt to small graphs but typically ignore imbalanced class distributions. This narrow focus limits their ability to address the complex, intertwined data imbalances commonly found in real-world graph datasets. To remedy this, we propose UniImb, a Unified Imbalance framework for imbalanced graph learning. The core idea of UniImb is to extract semantically shared prototype features via a balanced extraction that equalizes the influence of tail graphs-both minority-class instances and small-scale graphs. These prototypes enrich graph representations and improve robustness to complex, intertwined imbalances.
Specifically, UniImb first encodes multi granularity topological features and applies a personalized graph perturbation module, which injects controlled randomness into graph features and connections with graph dependent intensity. Graphs from smaller or minority classes are augmented with different strengths than larger, frequent ones. We then design a local prior based GNN to encode topological information and obtain graph level representations. Next, we introduce the Dynamic Balanced Prototype (DBP) module, consisting of a set of learnable prototype embeddings that compactly represent shared semantics. Graphs are softly assigned to prototypes based on similarity in a shared embedding space, and prototypes are jointly perceived in a collaborative manner. To regulate this process, we derive a prototype regularizer based on the information bottleneck principle, which encourages prototype activation probabilities to approach a uniform distribution. This allows tail graphs, such as minority class or small scale graphs, to exert comparable influence during prototype learning. Finally, the learned prototypes are used to enhance and regularize underrepresented graph representations, thereby improving discriminative performance on imbalanced datasets.
Contributions. ❶ Unified Insight. We propose UniImb, a unified framework for imbalanced graph learning that effectively addresses both class imbalance and topological imbalance in graph classification tasks. ❷ Practical Solution. UniImb integrates a series of advanced strategies with a Dynamic Balanced Prototype mechanism, effectively enhancing the representational capacity for tail graphs. ❸ Theoretical Explanation. Our design incorporates an information bottleneck-driven regularization, which ensures balanced influence from tail classes during training. ❹ Empirical Study. Beyond the standard graph classification datasets commonly used in mainstream benchmarks (Qin et al., 2025), we further introduce 3D conformer graph datasets and AirGraph, a large-scale real-world air-pollution graph dataset with naturally occurring long-tailed imbalance, to spur further research and practical progress in this area. We conduct a systematic evaluation on 19 datasets, and the results show that UniImb consistently achieves stable and highly competitive performance in challenging imbalanced graph classification scenarios. Moreover, UniImb is well compatible with a wide range of graph representation backbones and remains efficient while maintaining strong performance.
this section cite: ['b72', 'b38', 'b75', 'b12', 'b34', 'b56', 'b56', 'b56', 'b43', 'b79', 'b43']

Section: RELATED WORK
Graph classification is a foundational task in graph data analysis (Zhang et al., 2018), with GNNs emerging as the dominant architecture for this task (Velickovic et al., 2017;Xu et al., 2018). More recently, inspired by the success of Transformer architectures in natural language processing, researchers have extended these approaches to graph representation learning (Yun et al., 2019;Rampášek et al., 2022;Wang et al., 2024b). However, these methods encounter challenges with graph imbalance distribution (Liu et al., 2025). Imbalanced graph learning has emerged as a dedicated field to address this issue, comprising two main branches: class imbalance and topology imbalance (Qin et al., 2024). Class-imbalanced graph learning methods aim to correct skewed class distributions (Mao et al., 2025). A pioneering work is G 2 GNN (Wang et al., 2022b), which models each graph instance as a node in a "graph of graphs" and connects instances for data augmentation, thereby improving learning under imbalanced conditions. ImGKB (Tang & Liang, 2023) develops the restricted random walk kernel with the global graph information bottleneck to effectively capture inter-graph supervision for minority-class graphs. Conversely, topological imbalance methods aim to improve performance on small-scale graphs (Qin et al., 2025). For example, SOLT-GNN (Liu et al., 2022c) identifies recurring subgraph patterns to balance learning across graphs of varying sizes. TopoImb (Zhao et al., 2022) models topological groups and employs a modulator to assign greater importance to tail-topology graphs. However, existing methods often struggle in scenarios with complex imbalances. The development of a unified imbalanced graph learning framework remains an unexplored challenge.
this section cite: ['b92', 'b61', 'b78', 'b88', 'b45', 'b33', 'b42', 'b35', 'b56', 'b43', 'b95']

Section: PRELIMINARIES
Let G = {V, E, A, X} be a graph, where V is the node set with |V| nodes, E is the edge set, A ∈ R |V|×|V| is the adjacency matrix, X ∈ R |V|×d is the node feature matrix with d-dimension.
Given a graph set G = {G 1 , G 2 , . . . , G N } with N graph instances where the graph instance G i = {V Gi , E Gi , A Gi , X Gi } and their corresponding label set Y = {y 1 , y 2 , . . . , y N }, the goal of the graph-level classification task is to learn a mapping function F : G → R f to map any graph to a low-dimensional vector h ∈ R f . This representation is subsequently passed through a classifier to produce the predicted label distribution, yielding the final output prediction for the graph instance.
this section cite: []

Section: METHODOLOGY
As illustrated in Figure 1, UniImb frist adopts a personalized perturbation strategy with multi-scale graph topology encoding to enhance individual graph instances. It then employs GNNs to learn graph-level representations. Finally, we propose a dynamic balanced prototype mechanism to extract discriminative prototype features, thereby enhancing the model's capability to represent tail graphs and promoting unbiased learning.
this section cite: []

Section: GRAPH REPRESENTATION LEARNING

this section cite: []

Section: GRAPH TOPOLOGY ENCODING
Local Topology Encoding We adopt a learnable node position for learning local information (Dwivedi et al., 2021), which leverages random walks to perceive local structural information around each node. Specifically, for a graph G i , we first calculate its random walk operator M Gi = D -1 A Gi , where A Gi is the adjacency matrix of G i , D is the corresponding degree matrix and D i,i = j A Gi i,j . With z-steps of random walk, for the node v j of graph G i , we can get its local topology encodings:
LE Gi j = [(M Gi ) j,j
, (M Gi ) 2 j,j , (M Gi ) 3 j,j , . . . , (M Gi ) z j,j ] ∈ R z (1) here we adopt a low-complexity usage of the random walk matrix by considering only the landing probability of the node v j to itself, i.e., (M Gi ) j,j . And we can get the initial node position encodings
LE Gi ∈ R |V G i |×z .
Subsequently, The position encodings are integrated with GIN for end-to-end updates, enabling the capture of accurate graph topology information.
this section cite: ['b11']

Section: Global Topology Encoding.
The Laplacian matrix of a graph contains rich information about the graph's global structural properties, such as subgraph frequency and connectivity (Dong et al., 2016). To this end, we introduce a global topology encoding strategy to learn this information. Specifically, for a graph G i , we first extract the eigenvalues and eigenvectors of its laplacian matrix
L Gi = D Gi -A Gi , which is denoted as [λ 1 , λ 2 , • • • , λ z ] and [h 1 , h 2 , • • • , h z ] ∈ R |V G i |×z .
Then we can get the global topology encoding as follows,
GE Gi = φ ([ℓ(h i , λ i ) + ℓ(-h i , λ i )] z i=1 ) ∈ R |V G i |×d h(2)
where φ(•) and ℓ(•) are permutation-invariant neural networks (e.g., MLP layer) used to map the input to a higher-dimensional hidden space.
this section cite: ['b10']

Section: PERSONALIZED GRAPH PERTURBATION STRATEGY
Graph perturbation strategies, such as edge drop and feature mask (Wang et al., 2022a), can effectively improve the generalization performance of GNNs. However, traditional methods apply uniform perturbations to all graph samples, which can increase the difficulty of learning tail graphs in imbalanced scenarios.To address this issue, we propose a personalized graph perturbation strategy that customizes learnable perturbations based on the unique characteristics of each graph.
Edge Drop. For the graph G i with |V Gi | nodes and |E Gi | edges, we first calculate the average degree of the graph:
d Gi = 2|E G i | |V G i | ∈ R.
Then, we create personalized edge dropout distribution B(a Gi e ) where a Gi e = σ(MLP(d Gi )) ∈ R is learned by one MLP layer. And we sample the edge drop matrix M Gi e , where any value m in M Gi e follows Bernoulli distribution m e ∼ B(a Gi e )
, and m e = 0 means that the corresponding edge is dropped. Finally, we apply the learned perturbations to the adjacency matrix:
A Gi = A Gi ⊙ M Gi
e , where ⊙ means Hadamard product.
Feature Mask. We use a similar operation to personalized feature mask distribution B(β Gi n ); for every node v j in graph G i , we also sample a feature mask m j n ∈ {0, 1} ∼ B(β Gi n ), if m j n is equal to 0, we set the corresponding node feature value in X Gi j to 0. And we can get personalized feature mask matrix M Gi f and apply it to node feature matrix:
X Gi = X Gi ⊙ M Gi f .
this section cite: []

Section: GRAPH REPRESENTATION LEARNING VIA GNN
We leverage the powerful graph representation capabilities of GNNs to generate graph-level representations (Xu et al., 2018;Khoshraftar & An, 2024). Specifically, we use an MLP layer to map each graph G i into a high-dimensional space, and the output is then integrated with the global topology encoding as the initial output of the GNN, denoted as
X (0),Gi ∈ R |V G i |×d h .
For local node position encodings LE Gi ∈ R |V G i |×z , we also take it into one MLP layer, and the output is denoted as
LE 0,Gi ∈ R |V G i |×d h .
This transformed encoding, together with X 0,Gi , is fed into the GNN layer for graph representation learning. In this work, we use GIN (Xu et al.) as the primary backbone to exemplify our approach and validate the method across several other backbones, as shown in Table 4. For node v j , we can aggregate features from the neighborhood as follows,
X (ℓ+1),Gi j = MLP (ℓ)   (1 + ε) X (ℓ),Gi j LE (ℓ),Gi j + u∈N (vj ) X (ℓ),Gi u LE (ℓ),Gi u   , ∀ℓ ∈ {1, 2, . . . , L} (3)
where [•] means the concatenation of two matrices in the column dimension. ε is a harmonic coefficient, N (v j ) represents the neighbors of node v j in A Gi of the graph G i . And we also employ an independent GNN layer to update the local topology encoding, thereby capturing deeper graph topology information, which is denoted as follows,
LE (ℓ+1),Gi j = MLP (ℓ)   (1 + ε)LE (ℓ),Gi j + u∈N (v j ) LE (ℓ),Gi u   (4
)
After L graph convolution layers, each node aggregates information from its neighborhoods within a radius of up to L hops. Subsequently, a readout function is applied to aggregate the resulting node representations into a unified graph representation H Gi for the graph G i as follows,
H Gi = READOUT({ X (ℓ+1),Gi j | v j ∈ V Gi }) ∈ R 1×d h(5)
Finally, we can obtain the graph representations of N graphs, denoted as H ∈ R N×d h . To enhance the feature diversity of tail graphs, we employ a Feature Mixup strategy (Ju et al., 2025). This approach involves randomly rearranging the order of the representation vectors of graph instances, thereby generating a new matrix, denoted as H ∈ R 2N×d h . The details are provided in Appendix C.0.1.
this section cite: ['b78', 'b20', 'b19']

Section: DYNAMIC BALANCED PROTOTYPE BASED ON INFORMATION BOTTLENECK
We further propose a Dynamic Balanced Prototype (DBP) strategy, which unbiasedly extracts representative prototype features from graph data. These prototypes are universally applicable, enabling them to effectively enhance the representational capacity of graphs-particularly for underrepresented tail graph. Specifically, prototypes are defined as a set of learnable embeddings, denoted by S = [s 1 , s 2 , . . . , s K ] ∈ R K×d h , where the hyperparameter K represents the number of prototypes and satisfies K ≪ N.
this section cite: []

Section: ❶ Prototype Perception.
Prototypes are perceived from the graph data based on the attention coefficients computed between the prototypes and graph representations, as follows:
H S = Softmax TopK 1 S H ⊤ / d h HW v ∈ R K×d h(6)
where W v ∈ R d h ×d h is the learnable parameters. Next, we utilize prototypes to enhance the representation learning of the graph.
this section cite: []

Section: ❷ Prototype Balance.
We employ an attention mechanism to compute the affinity between each graph and the prototypes, thereby selecting the most relevant prototype to enhance its representational capacity. Specifically, this process is formulated as follows:
H = Sigmoid TopK 2 HS ⊤ / d h + γ H S ,(7)
where H ∈ R 2N×d h represents the enhanced representation. Here, we employ a sigmoid function together with a learnable vector γ ∈ R 1×K to generate discriminative similarity scores, rather than producing values concentrated around the mid-range.
During the iterative training process involving two alternating stages, the model learns to extract prototype features. However, these learned prototypes may become biased due to imbalanced graph data distributions, which can negatively affect their generalization capability. Ideally, we should reduce the dependency of the prototypes on the input data in order to enhance both robustness and generalization performance. To this end, we introduce the information bottleneck theory to guide the learning of more generalized prototype representation.
Theory 1. Information Bottleneck. For the training graphs G, prototype features S, and labels Y, the objective of the information bottleneck theory is expressed as minimizing the following objective function (Tishby et al., 2000):
min I(S; G) -βI(S; Y) (8
)
where β is the Lagrange multiplier. We can optimize the mutual information between the hidden feature and the labels I(S; Y) through supervised learning. Thus, the above objective is simplified to minimizing the mutual information between the training graph set G and the prototype feature S, i.e., reducing the redundancy of imbalanced input information. For more details, please refer to Appendix D.1.
this section cite: ['b59']

Section: Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1
Zipf (Axtell, 2001) 67.80 ± 3.71 73.29 ± 4.00 44.05 ± 2.29 79.02 ± 7.38 65.69 ± 3.11 79.78 ± 1.58 69.25 ± 10.26 69.79 ± 10.08 Exponential (Marshall & Olkin, 1967) 66.45 ± 3.34 71.99 ± 4.94 44.99 ± 2.95 82.31 ± 9.36 65.03 ± 2.41 79.85 ± 2.09 73.51 ± 6.73 74.47 ± 6.09 Poisson (Consul & Jain, 1973) 68.49 ± 1.70 73.94 ± 3.40 41.62 ± 3.72 71.98 ± 10.64 67.67 ± 5.50 80.60 ± 5.09 75.56 ± 2.03 75.85 ± 2.18 Uniform (Kuipers & Niederreiter, 2012) 70.44 ± 4.72 74.50 ± 4.99 46.63 ± 3.42 83.60 ± 6.50 68.30 ± 5.19 80.68 ± 4.22 75.73 ± 2.52 76.34 ± 2.60 represent an empirical distribution. Minimizing the mutual information between S and G is equivalent to the following objective:
min I(S; G) = G,s p(G)p(s|G) log p(s|G) p(s) ⇒ min KL(P∥U) ≈ min 1 2 K k=1 (p k -u k ) 2 (9
)
where KL (•) means Kullback-Leibler Divergence. The detailed proof is provided in Appendix D.2.
Proposition 2. Prototype-balancing Optimization. For 2N graphs, each graph hits TopK 2 prototypes, and we have a total hit count of K=2N× TopK 2 . The actual number of hits for each prototype
is N = {n 1 , n 2 , • • • , n K }, where n k =p k K, k ∈ {1, • • • , K}.
Subsequently, the regularization term minimizes the discrepancy between the actual attention activation n k and the prior distribution u k . To achieve a smoother and more effective optimization of the attention generation process, we introduce an intermediate modulation term η:
H = Sigmoid TopK 2 HS ⊤ / d h + η + γ H S ,(10)
where η ∈ R K directly affects the attention generation process. Nevertheless, the TopK operation is non-differentiable in practice. To address this, we introduce a specific constraint loss L M that iteratively models the prototype activation process during training.
LM = 1 2 K k=1 n k - 2 * N * TopK 2 1 u k 2 = 1 2 K k=1 η + StopGrad(n k -η) - 2 * N * TopK 2 1 u k 2 (11
)
where StopGrad(•) denotes the stop-gradient operator (Chen & He, 2021), which behaves as identity in the forward pass but yields zero in the backward pass, and this is a common approach used for non-differentiable operations. Building on Eq. 11, we further derive the optimization objective for η with the learning rate φ as follows:
η ← η -φ∇L M = η -φ sgn (n k -2 * N * TopK 2 * u k ) ,(12)
To identify an appropriate prior distribution, we conduct experiments to evaluate model performance under four different distributions of U:
Zipf u k = 1 k s K j=1 1 j s , Exponential u k = e -λk K j=1 e -λj , Poisson u k = λ k e -λ k! K j=1 λ j e -λ j!
, and Uniform u k = 1 K . The complete analysis and more details are provided in Appendix D.3. The experimental setup is described in Section 5.1. As shown in Table 1, we can observe that when the empirical distribution U follows a uniform distribution, the model achieves the highest accuracy. In other words, whenever the activation count of any prototype exceeds the average level, η imposes a penalty to reduce that prototype's activation priority in the subsequent optimization step, thereby enforcing load balancing. This mechanism effectively ensures balanced influence from tail graphs during learning.
Finally, we input the representations of the N input graph instances (i.e., the first N rows of H ) into a decoder composed of two layers of MLP for generating the predicted class labels Ŷ.
this section cite: ['b36', 'b24', 'b4']

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: Datasets.
We evaluate the model's performance on 19 datasets under class imbalance and topological imbalance scenarios. Following existing protocols (Qin et al., 2024), the datasets were processed into three levels of imbalance (low, medium, extreme). Due to space limitations, we primarily report results on six highly imbalanced datasets in the main text: D&D (Shervashidze et al., 2011), NCI1 (Wale et al., 2008), COLLAB (Leskovec et al., 2005), PROTEINS, REDDIT-B, and IMDB-MULTI (Yanardag & Vishwanathan, 2015). We also introduce a large-scale dataset, Airgraph (described below), and three 3D conformational graph datasets MoleculeNet (Wu et al., 2018): BBBP for blood-brain barrier permeability prediction, BACE for β-secretase inhibition prediction, and HIV for antiviral activity prediction.
Large-Scale AirGraph Dataset. comprises hourly PM2.5 measurements from the China National Environmental Monitoring Center collected at 1,341 stations across 28 provinces in mainland China between 2021 and 2023. For each province the data are represented as a graph in which monitoring stations are nodes and stations are connected based on distance. Each node encodes the preceding 24 hours of PM2.5 readings, and each graph is labeled by the spatially averaged PM2.5 over the subsequent 24 hours, categorized into three levels: low, medium, and high pollution. With 30,660 graph instances, AirGraph is substantially larger than most existing graph classification datasets and exhibits a natural class imbalance: the high pollution category accounts for only 6.86% of instances. More details are provided in Appendix G.
Baselines. We compare our model against four categories of baselines, encompassing a total of 23 representative baselines: ❶ Classic Graph Classification Models: GIN (Xu et al.), GCN (Kipf & Welling, 2016), GraphSAGE (Hamilton et al., 2017), InfoGraph (Sun et al., 2019), and GraphCL (You et al., 2020); ❷ Graph Transformer Models (GTs): GraphGPS (Rampášek et al., 2022), Exphormer (Shirzad et al., 2023), and Graph-Mamba (Wang et al., 2024b); ❸ 3D Graph Models: SchNet (Schütt et al., 2017), DimeNet (Gasteiger et al., 2020), and SphereNet (Liu et al., 2022b); ❹ Class-imbalance Methods: upsampling (Kubat et al., 1997) and reweighting strategies (Yuan & Ma, 2012) applied to various backbones, as well as G 2 GNN with edge and node masking, ImGKB (Tang & Liang, 2023), and DataDec (Zhang et al., 2023); ❺ Topology-imbalance Methods: SOLT-GNN (Liu et al., 2022c), ImbGNN (Xu et al., 2024), and TopoImb (Zhao et al., 2022).
this section cite: ['b42', 'b50', 'b63', 'b25', 'b80', 'b74', 'b16', 'b54', 'b84', 'b45', 'b51', 'b49', 'b14', 'b23', 'b56', 'b64', 'b79', 'b95']

Section: Experiment Setting.
We implement our proposed model on an 40GB NVIDIA A100 GPU with Pytorch. For the class-imbalanced graph classification task, we split the dataset into 25% for training and 25% for validation, and the remaining data is used for testing. For the topological-imbalanced graph classification task, we split the dataset into 10% for training and 10% for validation, and the remaining data is used for testing. We use the Adam optimizer (Kingma & Ba, 2014) with learning rate 0.001. All experiments are repeated 20 times, and we report the average and standard deviation. For the REDDIT-B, IMDB-MULTI, and COLLAB datasets, we employ one-hot encodings of node degrees as node features (Sun et al., 2019;You et al., 2020). We use widely metrics in the graph imbalance tasks including Macro-F1 and Micro-F1 (Xia et al., 2014;Lipton et al., 2014). And φ is set to 0.001. The number of prototypes K is set to {16, 16, 24, 24, 24, 32} on six datasets. We stack L = 5 layers of GNN. Other key hyperparameters of each dataset are shown in Table 13.
Table 2: Performance on class imbalance datasets with extreme imbalance degree. The best results are marked and the runner-ups are underlined . We report the average and standard deviation over 20 runs. Numbers marked with * indicate that the improvement is statistically significant compared with the best baseline (Wilcoxon Signed-Rank Test with p-value < 0.05).
this section cite: ['b21', 'b54', 'b84', 'b76', 'b27']

Section: Method
Backbone
PROTEINS D&D NCI1 REDDIT-B COLLAB IMDB-MULTI Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1Classic
GIN 25.33 ± 7.53 28.50 ± 5.82 9.99 ± 7.44 11.88 ± 9.49 18.24 ± 7.58 18.94 ± 7.12 33.19 ± 14.26 36.02 ± 17.38 32.58 ± 3.66 57.31 ± 4.12 13.25 ± 6.19 14.92 ± 5.43 InfoGraph 35.91 ± 7.58 36.81 ± 6.51 21.41 ± 4.51 27.68 ± 7.52 33.09 ± 3.30 34.03 ± 3.68 57.67 ± 3.80 67.10 ± 4.91 43.48 ± 4.29 59.10 ± 4.88 17.28 ± 7.28 29.18 ± 4.47 GraphCL 40.86 ± 6.94 41.24 ± 6.38 21.02 ± 3.05 26.80 ± 4.95 31.02 ± 2.69 31.62 ± 3.05 53.40 ± 4.06 62.19 ± 5.68 45.02 ± 5.61 60.22 ± 3.47 16.30 ± 9.22 32.18 ± 8.90 GTs GraphGPS 25.79 ± 7.05 28.71 ± 5.46 10.12 ± 4.41 11.97 ± 3.91 14.94 ± 2.41 15.62 ± 2.07 11.68 ± 7.76 12.71 ± 8.13 25.58 ± 12.93 39.92 ± 13.06 14.20 ± 5.61 28.54 ± 13.78 Exphormer 25.52 ± 4.79 28.38 ± 3.57 9.79 ± 4.18 10.85 ± 4.28 14.56 ± 3.92 15.36 ± 3.60 22.68 ± 10.79 27.33 ± 21.65 32.61 ± 17.44 42.02 ± 15.44 20.81 ± 5.43 28.14 ± 8.30 Garph-Mamba 31.12 ± 5.10 32.79 ± 4.02 4.99 ± 7.93 6.12 ± 10.36 14.11 ± 3.26 14.94 ± 3.82 15.27 ± 12.46 17.02 ± 14.71 42.53 ± 11.15 50.63 ± 5.38 16.89 ± 4.57 28.69 ± 12.47 up-sampling GIN 65.64 ± 2.67 71.55 ± 3.19 41.15 ± 3.74 70.56 ± 10.28 59.19 ± 4.39 71.80 ± 7.02 66.71 ± 3.92 83.00 ± 5.18 64.30 ± 2.67 66.10 ± 3.28 22.27 ± 10.01 38.32 ± 10.04 InfoGraph 62.68 ± 2.70 66.02 ± 3.18 41.55 ± 2.32 71.34 ± 6.76 53.38 ± 1.88 62.20 ± 2.63 67.01 ± 3.34 78.68 ± 3.71 63.28 ± 2.90 65.14 ± 3.29 21.79 ± 6.68 37.29 ± 7.02 GraphCL 64.21 ± 2.53 65.76 ± 2.61 38.96 ± 3.01 64.23 ± 8.10 49.92 ± 2.15 58.29 ± 3.30 62.01 ± 3.97 75.84 ± 3.98 64.57 ± 5.20 66.79 ± 4.11 23.62 ± 6.91 40.29 ± 6.90 re-weight GIN 54.54 ± 6.29 55.77 ± 7.11 28.49 ± 5.92 40.79 ± 11.84 36.84 ± 8.46 39.19 ± 10.05 45.17 ± 8.46 51.92 ± 12.29 57.83 ± 3.03 60.09 ± 4.59 22.07 ± 11.13 36.69 ± 11.14 InfoGraph 65.73 ± 3.10 69.60 ± 3.68 41.92 ± 2.28 72.43 ± 6.63 53.05 ± 1.12 62.45 ± 1.89 65.79 ± 3.38 77.35 ± 3.96 62.22 ± 3.90 64.48 ± 3.14 21.16 ± 6.79 38.02 ± 5.70 GraphCL 63.46 ± 2.42 64.97 ± 2.41 40.29 ± 3.31 67.96 ± 8.98 50.05 ± 2.09 58.18 ± 3.08 62.79 ± 6.93 76.15 ± 9.15 63.18 ± 4.55 65.29 ± 3.87 22.48 ± 6.82 39.57 ± 5.89 G 2 GNN remove edge 67.70 ± 2.96 73.10 ± 4.05 43.25 ± 3.91 77.03 ± 9.98 63.60 ± 1.57 72.97 ± 1.81 68.39 ± 2.97 86.35 ± 2.27 38.93 ± 3.22 54.98 ± 4.28 20.67 ± 9.88 36.89 ± 11.73 mask node 67.39 ± 2.99 73.30 ± 4.19 43.93 ± 3.46 79.03 ± 10.78 64.78 ± 2.86 74.91 ± 2.14 67.52 ± 2.60 85.43 ± 1.80 37.63 ± 5.19 53.92 ± 6.37 21.54 ± 9.49 35.78 ± 11.08 TopoImb / 53.95 ± 6.68 56.00 ± 7.88 7.72 ± 5.68 9.47 ± 4.26 16.41 ± 5.19 17.14 ± 5.68 10.38 ± 2.94 11.20 ± 2.69 19.24 ± 0.02 40.71 ± 0.05 9.37 ± 1.51 15.24 ± 0.90 DataDec dynamic sparsity 29.48 ± 3.98 31.25 ± 2.98 15.79 ± 2.38 18.86 ± 3.47 18.14 ± 3.86 18.52 ± 3.64 56.11 ± 4.65 65.35 ± 6.16 41.73 ± 1.20 57.27 ± 0.65 11.08 ± 1.15 16.29 ± 0.72 ImGKB / 53.99 ± 7.22 55.31 ± 8.17 31.15 ± 6.29 46.31 ± 11.72 32.93 ± 5.46 34.85 ± 7.14 11.00 ± 8.34 14.00 ± 1.34 18.84 ± 1.96 39.58 ± 4.90 16.53 ± 5.23 34.12 ± 12.71 ImbGNN / 67.69 ± 2.95 73.24 ± 3.85 46.06 ± 7.23 83.24 ± 19.03 65.48 ± 3.39 74.58 ± 5.49 68.36 ± 8.10 86.56 ± 4.77 56.62 ± 4.28 62.79 ± 3.83 17.52 ± 8.98 34.54 ± 8.24 UniImb / 70.44* ± 4.72 74.50 ± 4.99 46.63* ± 3.42 83.60* ± 6.50 68.30* ± 5.19 80.68* ± 4.22 76.24* ± 4.09 88.82* ± 2.93 75.73* ± 2.52 76.34* ± 2.60 33.45* ± 7.83 45.72* ± 4.87 Promotion 4.05%↑ 1.64%↑ 1.24%↑ 0.43%↑ 4.31%↑ 7.70%↑ 11.48%↑ 2.61%↑ 17.28%↑ 14.30%↑ 41.62%↑ 13.48%↑
this section cite: []

Section: PERFORMANCE ANALYSIS ON IMBALANCE GRAPH CLASSIFICATION
Owing to space limitations, we present the experiment results on all datasets under medium and low imbalance degrees in Appendix I.1.1 and I.3.1.
❶ Class Imbalance Graph Classification. As shown in Table 2, classic graph representation models perform the worst due to their inability to address class imbalance. While Graph Transformers (GTs) augment GNN with global attention mechanisms to capture long-range dependencies, they remain ineffective in mitigating the issue of class label imbalance. Upsampling methods require resampling tail classes, which may lead to overfitting on tail features and result in a loss of generalization. TopoImb and ImGKB exhibit highly unstable performance, performing even worse than naive models on most datasets (Qin et al., 2024). DataDec inadvertently introduces data leakage by using test samples to fine-tune the SVM classifier. To ensure a fair evaluation, we restrict all fine-tuning to the training set. Under this setting, DataDec shows significantly degraded performance. G 2 GNN and ImbGNN utilize the G 2 G framework for data augmentation. However, they underperform compared to simple upsampling baselines on multi-label datasets such as COLLAB. Our model achieves the best performance across all metrics.
❷ Topological Imbalance Graph Classification. As shown in Table 3, GIN consistently achieves the worst performance across all topologically imbalanced datasets. GraphCL further improves the performance through contrastive learning with graph augmentations, yielding significant gains over GIN. Graph Transformer models, including GraphGPS, Exphormer, and Graph-Mamba, build upon GNN by integrating global attention mechanisms, which help address topological imbalance. TopoImb alleviates substructure imbalance by incorporating topology-aware modeling and instance weight adjustment. ImbGNN constructs an enhanced G 2 G based on topology-friendly random walks, which helps mitigate the impact of topology-imbalance in the data. Our UniImb integrates a personalized graph perturbation strategy, a comprehensive graph structure encoding module, and the DBP module. The DBP module effectively extracts prototype features, significantly enhancing the representation capability for graphs of various scales. Our model achieves competitive performance.
this section cite: ['b42']

Section: COMPREHENSIVE EVALUATION OF UNIIMB: APPLICABILITY AND SCALABILITY
❶ Applicability with various backbones. We evaluated various GNN variants and Graph Transformers as backbone networks under class-imbalanced settings. The results in Table 4 show that all backbone networks achieved significant performance improvements when integrated with UniImb. This validates the plug-and-play applicability of our method, demonstrating its ability to significantly enhance the imbalanced graph classification performance of GNNs. For experiments on the applicable scenarios of topological imbalance, see the Appendix I.7.
❷ Scalability on the large graph dataset. Figure 2 illustrates the prediction accuracy (Macro-F1) and complexity (training time and memory usage) of several advanced models on the large-scale dataset AirGraph, which is naturally imbalanced. We observe that UniImb consistently outperforms strong baseline models, achieving superior performance in predicting air pollution levels. This provides valuable insights for environmental protection and human health. Furthermore, UniImb demonstrates good computational efficiency. Compared with methods specifically designed for imbalance handling, such as TopoImb, ImbGNN, and G 2 GNN, UniImb achieves more significant improvements in both performance and computational complexity.
this section cite: []

Section: PERFORMANCE ON INTERTWINED CLASS AND TOPOLOGY IMBALANCES SCENARIO
We further evaluated the effectiveness of various models in complex scenarios involving class and topological imbalance. Details of the data processing are provided in Appendix G.1.5. Additionally, AirGraph also exhibits a combined case of categorical and topological imbalance. Figures 3 and 2 compare our method with several advanced models. We found that methods designed to address topological imbalance generally perform worse than those targeting class imbalance. This is because the distribution of categorical features has a crucial impact on the decision boundary, directly influencing model performance. Our method achieved the best overall performance.
21.98 26.08 24.92 25.6 3 21.9 5 6 1 .5 1 6 1 .4 7 7 0 .5 2 29.8 1 56 .2 8 40.2 8 25.60 27.10 64.70 51.46 70 .88 28.6 0 44 .8 2 29 .69 31 .86 29. 79 53 .0 5 64 .9 3 66 .21 40.22 39.78 58.2 9 41.89 50. 30 7 0 .7 6 7 0 .5 7 8 0 .9 7 39 .73 42 .44 45. 18 41.12 43.21 50.81 61.58 75 .20 12. 72 24 .6 4 15 .6 6 12 .9 5 13 .45 27 .7 7 27 .0 9 34 .7 4 PR OT EIN S DD N CI 1 RE DD IT-B COLL AB IM D B -M U LT I GIN ImGKB DataDec TopoImb SOLT-GNN G²GNN ImbGNN UniImb (Ours) 28.10 32.03 30.12 30.4 4 28.0 8 6 7 .9 5 7 2 .5 3 7 6 .6 5 34. 05 67 .2 8 41.2 7 30.72 33.43 71.27 68.7 3 73 .58 40 .05 46 .2 8 43 .4 7 41 .7 9 40 .57 59 .3 9 65 .7 6 67 .79 46.76 42.02 61.0 6 47.7 3 53. 57 7 2 .5 8 7 2 .3 4 8 1 .3 5 55 .6 9 57 .29 60 .39 57.1 1 59.04 55.81 62.30 75 .83 23 .58 26 .2 2 2 8 .7 2 23 .6 5 23 .6 8 30 .6 1 28 .7 4 39 .2 1 PR OT EIN S DD N CI 1 RE DD IT-B COLL AB IM D B -M U LT I GIN ImGKB DataDec TopoImb SOLT-GNN G²GNN ImbGNN UniImb (Ours) Figure 3: Macro-F1 (left) and Micro-F1 (right) on intertwined imbalance datasets. 35 45 55 65 75 Macro-F1 [%] 70.4 46.6 68.3 75.7 67.9 45.3 64.5 74.4 68.3 45.3 60.9 62.5 63.9 44.8 61.0 70.3 64.6 41.8 56.7 50.3 66.4 45.3 63.5 70.5 PROTEINS D&D NCI1 COLLAB 35 45 55 65 75 Macro-F1 [%] 71.3 74.5 65.0 73.5 69.7 73.3 63.6 70.3 67.4 71.9 60.7 69.4 66.2 60.8 63.8 67.4 56.8 53.8 59.2 39.3 69.8 73.9 63.2 67.0 Ours w/o ImMix w/o TopoEnc w/o Pertu w/o DBP w/o BalOpt Figure 4: Ablation experiments on class imbalance (upper) and topological imbalance (lower). 5.5 ABLATION STUDY To evaluate the effectiveness of the key components of the model, we create five variants: (1) ' w/o ImMix' means that we remove imbalanced feature mixup strategy; (2) 'w/o TopoEnc' means that we remove graph topology encoding; (3) 'w/o Pertu' means that the graph perturbation strategy strategy is completely removed; (4) 'w/o DBP' means that the dynamic balanced prototype strategy is removed; (5) 'w/o BalOpt' means that the we remove the load balancing optimization.
As shown in Figure 4, the w/o TopoEnc variant achieves notably lower accuracy, indicating that incorporating structural information enables the model to better capture graph-level features. Among all ablated versions, the w/o DBP variant-without the proposed load-balanced prototype learning-exhibits the worst performance, demonstrating that explicitly modeling prototype features is critical for handling imbalanced graph data. The w/o BalOpt variant also suffers a performance drop, particularly on class-imbalanced datasets, which further confirms that the proposed load-balancing strategy facilitates the extraction of generalizable prototypes. Overall, the inferior performance of all ablated variants compared to UniImb provides strong evidence for the effectiveness and necessity of each component in our framework. For more ablation studies, see the Appendix I.4 and I.5.
this section cite: []

Section: HYPERPARAMETER SENSITIVITY ANALYSIS
We further evaluate the model's sensitivity to the number of prototypes. As shown in Figure 5, we observe that model performance first increases and then decreases with respect to K. When K is smaller than the optimal value, it is insufficient to capture enough discriminative prototype information. On the other hand, when K exceeds the optimal value, having too many prototypes prevents the model from focusing on representative ones, leading to performance degradation. For sensitivity studies on TopK 1 and TopK 2 , see the Appendix I.6.
1 4 8 12 16 24 32 48 64 Number of Prototypes K 60 62 64 66 68 70 72 74 76 78 Macro-F1[%] PROTEINS Macro-F1 Best Macro-F1 1 4 8 12 16 24 32 48 64 Number of Prototypes K 50 55 60 65 70 75 80 Macro-F1[%] NCI1 Macro-F1 Best Macro-F1 1 4 8 12 16 24 32 48 64 Number of Prototypes K 40 42 44 46 48 50 Macro-F1[%] D&D Macro-F1 Best Macro-F1 1 4 8 12 16 24 32 48 64 Number of Prototypes K 70 72 74 76 78 80 Macro-F1[%] COLLAB Macro-F1 Best Macro-F1 60 62 64 66 68 70 72 74 76 78 80 Micro-F1[%] Micro-F1 Best Micro-F1 60 65 70 75 80 85 Micro-F1[%] Micro-F1 Best Micro-F1 68 73 78 83 88 Micro-F1[%] Micro-F1 Best Micro-F1 70 72 74 76 78 80 Micro-F1[%] Micro-F1 Best Micro-F1 1 4 8 1216 2432 4864 Number of Prototypes K 65 66 67 68 69 70 71 72 73 Macro-F1[%] PROTEINS Macro-F1 Best Macro-F1 1 4 8 1216 2432 4864 Number of Prototypes K 50 55 60 65 70 75 Macro-F1[%] NCI1 Macro-F1 Best Macro-F1 1 4 8 1216 2432 4864 Number of Prototypes K 70 71 72 73 74 75 76 Macro-F1[%] D&D Macro-F1 Best Macro-F1 1 4 8 1216 2432 4864 Number of Prototypes K 68 69 70 71 72 73 74 75 76 77 78 Macro-F1[%] COLLAB Macro-F1 Best Macro-F1 68.0 68.5 69.0 69.5 70.0 70.5 71.0 71.5 72.0 72.5 73.0 73.5 74.0 74.5 75.0 75.5 76.0 Micro-F1[%] Micro-F1 Best Micro-F1 45 50 55 60 65 70 75 Micro-F1[%] Micro-F1 Best Micro-F1 70.0 70.5 71.0 71.5 72.0 72.5 73.0 73.5 74.0 74.5 75.0 75.5 76.0 76.5 77.0 77.5 78.0 Micro-F1[%] Micro-F1 Best Micro-F1 72.0 72.5 73.0 73.5 74.0 74.5 75.0 75.5 76.0 76.5 77.0 77.5 78.0
this section cite: []

Section: Micro-F1[%]
Micro-F1 Best Micro-F1
this section cite: []

Section: EVALUATION ON 3D CONFORMATION DATASETS
Beyond conventional graph datasets, we introduce 3D conformational graph datasets from Molecu-leNet: BBBP for blood-brain barrier permeability prediction, BACE for β-secretase inhibition activity prediction, and HIV for antiviral activity prediction. To accommodate this graph structure, we incorporate specific models, including SchNet, DimeNet and SphereNet, which serve as backbone networks for graph imbalance methods. We follow established protocols for handling extremly imbalanced class distributions. And we select SphereNet as backbone of G 2 GNN, TopoImb, and ImbGNN. For our model, we retain only the core DBP module. As shown in Table 5, the proposed method achieves excellent performance on these 3D Conformation datasets, validating its applicability to domains with complex structural dependencies.
this section cite: []

Section: GRAPH REPRESENTATION VISUALIZATION
We extract the node-level representations from the PROTEINS dataset using GIN and ImbGNN, as well as the representations before and after applying the DBP module. These are then visualized in Figure 6. The representations generated by the naive GIN (sub-figure (a)) for imbalanced graphs are highly entangled and poorly separated, making it difficult for the classifier to distinguish between classes. While ImbGNN-a specialized method for imbalanced learning-shows relatively better separation, many samples still exhibit overlapping feature distributions. We observe that, after incorporating the Dynamic Balanced Prototype mechanism, the learned label-wise representations become significantly more discriminative. This enhancement greatly benefits the classifier in distinguishing between different classes and improves overall performance.
this section cite: []

Section: CONCLUSION
In this paper, we propose a unified framework for addressing imbalance graph classification task, effectively tackling both class imbalance and topological imbalance. Our core contribution lies in the introduction of a dynamic balanced prototype strategy, which extracts prototype features from graph data to enhance robust representation learning under imbalanced scenarios. We further impose a prototype-balancing optimization strategy to encourage balanced activation across all prototypes. Additionally, we incorporate several advanced techniques, including graph topology encoding, graph perturbation and feature mixup to comprehensively improve model performance.
Extensive experiments on over 19 datasets demonstrate the impressive effectiveness of our framework.
this section cite: []

Section: References
Ref_id:b0 Title: Deep variational information bottleneck Year: (2016)
Ref_id:b1 Title: Subgraph neural networks Year: (2020)
Ref_id:b2 Title: Zipf distribution of us firm sizes Year: (2001)
Ref_id:b3 Title: Graph transformer for graph-to-sequence learning Year: (2020)
Ref_id:b4 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b5 Title: Edge classification on graphs: New directions in topological imbalance Year: (2025)
Ref_id:b6 Title: A generalization of the poisson distribution Year: (1973)
Ref_id:b7 Title: Elements of information theory Year: (1999)
Ref_id:b8 Title: Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. correlation with molecular orbital energies and hydrophobicity Year: (1991)
Ref_id:b9 Title: Signed graph convolutional networks Year: (2018)
Ref_id:b10 Title: Learning laplacian matrix in smooth graph signal representations Year: (2016)
Ref_id:b11 Title: Graph neural networks with learnable structural and positional representations Year: (2021)
Ref_id:b12 Title: Generalizing graph neural networks on out-of-distribution graphs Year: (2023)
Ref_id:b13 Title: Hyperbolic geometric graph representation learning for hierarchy-imbalance node classification Year: (2023)
Ref_id:b14 Title: Directional message passing for molecular graphs Year: (2020)
Ref_id:b15 Title: Edge classification on imbalanced multi-relational graphs Year: (2025)
Ref_id:b16 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b17 Title: Graph representation learning Year: (2020)
Ref_id:b18 Title: Graphpatcher: mitigating degree bias for graph neural networks via test-time augmentation Year: (2023)
Ref_id:b19 Title: Cluster-guided contrastive class-imbalanced graph classification Year: (2025)
Ref_id:b20 Title: A survey on graph representation learning methods Year: (2024)
Ref_id:b21 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b22 Title: Semi-supervised classification with graph convolutional networks Year: (2016)
Ref_id:b23 Title: Addressing the curse of imbalanced training sets: one-sided selection Year: (1997)
Ref_id:b24 Title: Uniform distribution of sequences Year: (2012)
Ref_id:b25 Title: Graphs over time: densification laws, shrinking diameters and possible explanations Year: (2005)
Ref_id:b26 Title: A survey of graph neural network based recommendation in social networks Year: (2023)
Ref_id:b27 Title: Optimal thresholding of classifiers to maximize f1 measure Year: (2014)
Ref_id:b28 Title: Deep graph learning for anomalous citation detection Year: (2022)
Ref_id:b29 Title: Spherical message passing for 3d molecular graphs Year: (2022)
Ref_id:b30 Title: Towards locality-aware meta-learning of tail node embeddings on networks Year: (2020)
Ref_id:b31 Title: Tail-gnn: Tail-node graph neural networks Year: (2021)
Ref_id:b32 Title: On size-oriented long-tailed graph classification of graph neural networks Year: (2022)
Ref_id:b33 Title: A survey of imbalanced learning on graphs: Problems, techniques, and future directions Year: (2025)
Ref_id:b34 Title: Class-imbalanced learning on graphs: A survey Year: (2025)
Ref_id:b35 Title: Learning knowledge-diverse experts for long-tailed graph classification Year: (2025)
Ref_id:b36 Title: A multivariate exponential distribution Year: (1967)
Ref_id:b37 Title: Faster kernels for graphs with continuous attributes via hashing Year: (2016)
Ref_id:b38 Title: Universal graph transformer self-attention networks Year: (2022)
Ref_id:b39 Title: Differentially private densest subgraph detection Year: (2021)
Ref_id:b40 Title: Towards interpretable sparse graph representation learning with laplacian pooling Year: (2019)
Ref_id:b41 Title: Graph invariant kernels Year: (2015)
Ref_id:b42 Title: Igl-bench: Establishing the comprehensive benchmark for imbalanced graph learning Year: (2024)
Ref_id:b43 Title: Graph sizeimbalanced learning with energy-guided structural smoothing Year: (2025)
Ref_id:b44 Title: Imgagn: Imbalanced network embedding via generative adversarial graph networks Year: (2021)
Ref_id:b45 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b46 Title: Iam graph database repository for graph based pattern recognition and machine learning Year: (2008)
Ref_id:b47 Title: The pareto principle: its use and abuse Year: (1987)
Ref_id:b48 Title: the enzyme database: updates and major new developments Year: (2004)
Ref_id:b49 Title: Schnet: A continuous-filter convolutional neural network for modeling quantum interactions Year: (2017)
Ref_id:b50 Title: Weisfeiler-lehman graph kernels Year: (2011)
Ref_id:b51 Title: Exphormer: Sparse transformers for graphs Year: (2023)
Ref_id:b52 Title: Opening the black box of deep neural networks via information Year: (2017)
Ref_id:b53 Title: Heterogeneous graph neural network with relation-aware label propagation for unbalanced node classification Year: (2025)
Ref_id:b54 Title: Infograph: Unsupervised and semisupervised graph-level representation learning via mutual information maximization Year: (2019)
Ref_id:b55 Title: Spline-fitting with a genetic algorithm: A method for developing classification structure-activity relationships Year: (2003)
Ref_id:b56 Title: Where to find fascinating inter-graph supervision: Imbalanced graph classification with kernel information bottleneck Year: (2023)
Ref_id:b57 Title: Ecgn: A cluster-aware approach to graph neural networks for imbalanced classification Year: (2024)
Ref_id:b58 Title: Deep learning and the information bottleneck principle Year: (2015)
Ref_id:b59 Title: The information bottleneck method Year: (2000)
Ref_id:b60 Title: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks Year: (2000)
Ref_id:b61 Title: Graph attention networks Year: (2017)
Ref_id:b62 Title: Learning physical properties of anomalous random walks using graph neural networks Year: (2021)
Ref_id:b63 Title: Comparison of descriptor spaces for chemical compound retrieval and classification Year: (2008)
Ref_id:b64 Title: Pattern expansion and consolidation on evolving graphs for continual traffic prediction Year: (2023)
Ref_id:b65 Title: Stone: A spatio-temporal ood learning framework kills both spatial and temporal shifts Year: (2024)
Ref_id:b66 Title: Graph-mamba: Towards long-range graph sequence modeling with selective state spaces Year: (2024)
Ref_id:b67 Title: Auxiliary-loss-free load balancing strategy for mixture-of-experts Year: (2024)
Ref_id:b68 Title: Degree-related bias in link prediction Year: (2022)
Ref_id:b69 Title: Distance-wise prototypical graph neural network in node imbalance classification Year: (2021)
Ref_id:b70 Title: Graph neural networks: Self-supervised learning Year: (2022)
Ref_id:b71 Title: Imbalanced graph classification via graph-ofgraph neural networks Year: (2022)
Ref_id:b72 Title: Neural architecture search for gnn-based graph classification Year: (2023)
Ref_id:b73 Title: Graph neural networks in recommender systems: a survey Year: (2022)
Ref_id:b74 Title: Moleculenet: a benchmark for molecular machine learning Year: (2018)
Ref_id:b75 Title: A comprehensive survey on graph neural networks Year: (2020)
Ref_id:b76 Title: Robust multi-view spectral clustering via low-rank and sparse decomposition Year: (2014)
Ref_id:b77 Title: How powerful are graph neural networks? Year: ()
Ref_id:b78 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b79 Title: When imbalance meets imbalance: Structure-driven learning for imbalanced graph classification Year: (2024)
Ref_id:b80 Title: Deep graph kernels in Year: (2015)
Ref_id:b81 Title: Few-shot link prediction in dynamic networks Year: (2022)
Ref_id:b82 Title: Extract and refine: Finding a support subgraph set for graph representation Year: (2023)
Ref_id:b83 Title: Do transformers really perform badly for graph representation? Year: (2021)
Ref_id:b84 Title: Graph contrastive learning with augmentations Year: (2020)
Ref_id:b85 Title: Sampling+ reweighting: Boosting the performance of adaboost on imbalanced datasets Year: ()
Ref_id:b86 Title: Graver: Generative graph vocabularies for robust graph foundation models fine-tuning Year: (2025)
Ref_id:b87 Title: Dg-mamba: Robust and efficient dynamic graph structure learning with selective state space models Year: (2025)
Ref_id:b88 Title: Graph transformer networks Year: (2019)
Ref_id:b89 Title: Lte4g: Long-tail experts for graph neural networks Year: (2022)
Ref_id:b90 Title: When sparsity meets contrastive models: Less graph data can bring better classbalanced representations Year: ()
Ref_id:b91 Title:  Year: (2023)
Ref_id:b92 Title: An end-to-end deep learning architecture for graph classification Year: (2018)
Ref_id:b93 Title: From stars to subgraphs: Uplifting any gnn with local structure awareness Year: ()
Ref_id:b94 Title: Graphsmote: Imbalanced node classification on graphs with graph neural networks Year: (2021)
Ref_id:b95 Title: Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 vanilla Year: (2022)
Ref_id:b96 Title: Model Backbone REDDIT-B AIDS FRANKENSTEIN COLLAB IMDB-MULTI Synthie Year: ()
Ref_id:b97 Title:  Year: ()
Ref_id:b98 Title:  Year: ()
Ref_id:b99 Title:  Year: ()
Ref_id:b100 Title:  Year: ()
Ref_id:b101 Title: Table 21: Macro-F1, Micro-F1 on topological imbalance datasets with low imbalance degree. Model PROTEINS D&D NCI1 PTC-MR Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Macro-F1 Micro-F1 Year: ()
Ref_id:b102 Title:  Year: ()
Ref_id:b103 Title:  Year: ()
