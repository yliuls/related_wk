Title: A Closer Look at Graph Transformers: Cross-Aggregation and Beyond
Abstract: Graph Transformers (GTs), which effectively capture long-range dependencies and structural biases simultaneously, have recently emerged as promising alternatives to traditional Graph Neural Networks (GNNs). Advanced approaches for GTs to leverage topology information involve integrating GNN modules or modulating node attributes using positional encodings. Unfortunately, the underlying mechanism driving their effectiveness remains insufficiently understood. In this paper, we revisit these strategies and uncover a shared underlying mechanism-Cross Aggregation-that effectively captures the interaction between graph topology and node attributes. Building on this insight, we propose the Universal Graph Cross-attention Transformer (UGCFormer), a universal GT framework with linear computational complexity. The idea is to interactively learn the representations of graph topology and node attributes through a linearized Dual Cross-attention (DCA) module. In theory, this module can adaptively capture interactions between these two types of graph information, thereby achieving effective aggregation. To alleviate overfitting arising from the dual-channel design, we introduce a consistency constraint that enforces representational alignment. Extensive evaluations on multiple benchmark datasets demonstrate the effectiveness and efficiency of UGCFormer.

Section: Introduction
Node classification, aimed at accurately predicting node categories based on the graph topology and node attributes, is a fundamental task in identifying the properties of individual nodes [12,18,14,32,11,10]. As a powerful class of models for fusing topology and attribute information in graphs, Graph Neural Networks (GNNs) have achieved initial successes in this task [29,5,52,24,22,25]. In general, they follow the graph-bound Message Passing (MP) paradigm [16]. While this paradigm endows GNNs with the localizing property, it also restricts their ability to capture long-range dependencies [9], resulting in well-known challenges such as over-smoothing [6,59] and over-squashing [17].
Inspired by the remarkable success of Transformers in NLP [35], Graph Transformers (GTs) have emerged as powerful architectures for node classification tasks. The core component of Transformers is the Self-Attention (SA) module [46], which models full interactions among tokens within a sequence, thereby endowing the Transformers with globalizing properties. The initial success of GTs can be attributed to the strategic integration of discriminative graph topology into Transformer architectures, enabling the simultaneous capture of structural biases and long-range dependencies. To date, two primary strategies have achieved SOTA performance in existing GTs: (1) integrating GNN blocks [31,50,7,61], and (2) modulating node attributes utilizing Positional Encodings (PEs) [2,49,44]. However, both strategies face inherent limitations. The first tends to inherit drawbacks from GNNs due to its reliance on them, whereas the second introduces additional computational complexity due to the use of PEs, thereby restricting the models' universality 1 and scalability. This leads to a fundamental question:
What underlying mechanism drives the effectiveness of diverse Graph Transformers?
A thorough understanding of the underlying mechanisms can offer valuable insights for developing more advanced and efficient architectures. Following this line, this paper theoretically investigates the mechanism shared by the aforementioned types of GTs and, based on this insight, proposes a novel GT architecture. In particular, the unified cross-aggregation mechanism (as formally defined in Definition 1) is explored by analytically decoupling topology and attribute representations from node representations. Specifically, the GNN block in GTs can be interpreted as aggregating topology representations into attribute representations (in Theorem 1), indicating that this category of GTs inherently incorporates cross-aggregation. Furthermore, GTs employing PEs contain diverse forms of cross-aggregation between topology and attribute representations. Therefore, the shared underlying mechanism among these GTs is cross-aggregation between graph topology and node attributes. This understanding naturally leads to a key question:
How can we design an effective and efficient GT architecture grounded in cross-aggregation?
To this end, this paper proposes the Universal Graph Cross-attention Transformer (UGCFormer), which implements the cross-aggregation mechanism via cross-attention. To be specific, it separately encodes graph topology and node attributes to obtain their initial representations. At its core lies a linearized Dual Cross-Attention (DCA) module that updates the topology and attribute representations by computing cross-attention scores among nodes and utilizing them for weighted aggregation. In theory, the DCA module adaptively captures both correlation and exclusion relationships between graph topology and node attributes, making it simple yet effective. Finally, the two representations are integrated to yield a comprehensive node representation. To prevent representation distortion, a consistency constraint is introduced to enforce mutual alignment between them.
The main contributions of this work are summarized as follows:
• Mechanism Revelation: We theoretically reveal a unified mechanism across typical Graph Transformers, namely cross-aggregation between graph topology and node attributes.
• Model Innovation: We propose UGCFormer, a GT architecture equipping with a linearized Dual Cross-Attention (DCA) module that implements the cross-aggregation mechanism.
• Comprehensive Evaluation: Extensive evaluations conducted on sixteen homophilic, heterophilic, and large-scale graphs demonstrate the universality and scalability of UGCFormer.
this section cite: ['b11', 'b17', 'b13', 'b31', 'b10', 'b9', 'b28', 'b4', 'b51', 'b23', 'b21', 'b24', 'b15', 'b8', 'b5', 'b58', 'b16', 'b34', 'b45', 'b30', 'b49', 'b6', 'b60', 'b1', 'b48', 'b43']

Section: Preliminaries
This section begins by presenting the notation used throughout this paper. Then, it introduces the concepts of Graph Neural Networks (GNNs) and Graph Transformers (GTs).
this section cite: []

Section: Notations
The subject of this paper is the widely-used undirected attribute graph, denoted as G(V, E), where V and E represent the node set and edge set. V consists of n node instances {(x v , y v )} v∈V , where
x v ∈ R f and y v ∈ R c denote the node attribute and label of node v, respectively. f is the dimension of attributes and c is the dimension of labels. E = {(v i , v j )} terms the edge set. Typically, graph topology is described by the adjacency matrix A ∈ R n×n where a i,j = 1 only if (v i , v j ) ∈ E, and a i,j = 0 otherwise. In formal terms, the graph G can be redescribed as G(A, X). In the context of semi-supervised learning, the node labels are segmented into two sets: Y L ∈ R n l ×c for the labeled nodes and Y U ∈ R nu×c for the unlabeled nodes.
To verify the model's universality, this paper examines graphs with varying degrees of homophily.
In homophilic graphs, edges are typically formed between nodes with similar labels. Conversely, in heterophilic graphs, edges tend to form between nodes with dissimilar labels [37,6,60,62].
this section cite: ['b36', 'b5', 'b59', 'b61']

Section: Graph Neural Networks
Message Passing (MP)-based Graph Neural Networks (GNNs) follow an aggregation-combination strategy. Specifically, the representation of each node is iteratively updated by aggregating the features from its local neighbors and combining the aggregated features with its features, which is given by
h l v ≜ COM l h l-1 v , AGG l {h l-1 u |u ∈ N (v)} ,(1)
where N (v) denotes the set of neighboring nodes of node v. For the functions AGG(•) and COM (, ), vanilla GNNs, e.g., GCN [29], adopt the sum function to implement them, that is,
GCN (A, H) : H l+1 = σ( ÃH l W), H 0 = X,(2)
where σ(•) stands for the nonlinear activation functions, and Ã = D-1 2 Â D-1 2 is the normalized adjacency matrix with Â = A + I. W denotes the trainable projection parameters.
this section cite: ['b28']

Section: Transformers
Inspired by the success of Transformers in NLP [46], numerous variant models have been designed for multiple fields, including CV [21] and Graph Learning. They typically consist of four functional components: attention module, feed-forward network, residual connection, and normalization.
Self-attention Module. This is a core component of the vanilla Transformer to model intra-sequence relationships among all tokens [46]. Given a sequence containing n tokens H = [h i ] n-1 i=0 ∈ R n×d , the module first projects H into Query q(H), Key k(H), and Value k(H). It then employs the attention scores calculated from all Query-Key pairs to perform a weighted sum of the Value vectors.
A general formulation of the Self-Attention (SA) module is given by
SA(H) : ĤSA = Sof tmax q(H)k(H) ⊤ √ d v(H),(3)
where q(•), k(•), and v(•) generate the Query, Key, and Value via MLPs [41] with learnable parameters W. The attention score Sof tmax(q(H)k(H) ⊤ / √ d) ∈ R n×n is computed via the scaled dot product of full-token pairs, resulting in a quadratic computational complexity.
this section cite: ['b45', 'b20', 'b45', 'b40']

Section: Graph Transformers (GTs).
Most existing models [51, 1, 39, 36, 4, 53, 61, 3] build upon the SA module. GTs differ from traditional Transformers in how they leverage topology information to capture structural biases. As discussed in the Introduction, two main strategies for incorporating topology information have achieved SOTA performance on node-level tasks: (1) integrating GNN blocks, and (2) modulating node attributes utilizing Positional Encodings (PEs).
this section cite: []

Section: Cross-Attention Module.
Unlike self-attention, which models the intra-source relationships, crossattention captures the interactions between two distinct sources. For the features from two different sources H ∈ R n1×d and Z ∈ R n2×d , the Cross-Attention (CA) module can be expressed as
CA(Z, H) : ĤCA = Sof tmax q(Z)k(H) ⊤ √ d v(H).(4)
After the representation ĤCA is obtained, it is typically used as the cross-source representation to update Z. Due to its exceptional capacity for modeling inter-source relationships, this module has been applied in diverse domains, e.g., NLP [15] and CV [26]. However, it has received little attention in Graph Learning, largely due to the lack of motivation and well-defined applied target. Moreover, similar to the self-attention (Eq. 3), its computational complexity is quadratic, i.e., O(n 1 n 2 ).
this section cite: ['b14', 'b25']

Section: Methodology
This section starts by theoretically exploring the functional mechanism shared by Graph Transformers (GTs) that use Graph Neural Network (GNN) blocks and GTs that utilize Positional Encodings (PEs). Inspired by this mechanism, it introduces UGCFormer, a simple yet universal graph cross-attention Transformer with linear complexity. Finally, it gives a comprehensive analysis of UGCFormer.
this section cite: []

Section: Motivations
As previously discussed, the underlying mechanism behind the effectiveness of typical GTs remains insufficiently explored. To address this issue, this subsection proposes a cross-aggregation mechanism and theoretically examines how it is manifested in the two types of SOTA GTs.
The cross-aggregation mechanism is formally defined as follows. Definition 1. (Cross-aggregation mechanism) Given two representations B ∈ R n1×d1 and Z ∈ R n2×d2 from different modalities (sources), which share at least one same dimension, i.e., n 1 = n 2 or d 1 = d 2 . A general formula for two types of cross-aggregations can be expressed as
Ẑ ≜ Sim(Z, B)B, if d 1 = d 2 , B Sim(B, Z), if n 1 = n 2 ,(5)
where Sim(Z, B) denotes a similarity function between Z and B, such as cosine similarity.
The first case corresponds to sample (node)-level aggregation, e.g., cross-attention (Eq. 4), while the second corresponds to dimension (feature)-level aggregation [56,57,61]. When Z = B, Eq. 5 reduces to a self-aggregation (e.g., self-attention). Accordingly, two theorems are presented. Theorem 1. In typical Graph Transformers, the diffusion matrix of GNN blocks can be expressed via eigendecomposition as S = UΛU ⊤ , where U and Λ = diag([λ 1 , . . . , λ n ]) represents eigenvectors and eigenvalues, respectively (in descending order). Accordingly, the GNN block can be viewed as a cross-aggregation between attribute representations XW and topology representations U √ Λ.
Theorem 2. Given the modulated node attributes using any PE, i.e., X = [X; P], where P ∈ R n×k represents the PE and [; ] denotes concatenation operator. PE-based GTs (Eq. 3) inherently contain a cross-aggregation between attribute representations XW and topology representations PW.
The proofs for Theorems 1 and 2 are provided in Sections B and C, respectively. In short, the key mechanism of GTs using GNNs and PEs is Cross-Aggregation between topology and attributes.
this section cite: ['b55', 'b56', 'b60']

Section: UGCFormer
Motivated by the cross-aggregation mechanism explored in the previous subsection, this subsection introduces UGCFormer, a simple yet universal GT. At its core, UGCFormer employs a linearized cross-attention module that implements the cross-aggregation mechanism to capture interactions between graph topology and node attributes. UGCFormer consists of four modules, each of which is described below. The detailed implementation is provided in Algorithm 1.
this section cite: []

Section: Initial Representation Layer.
Two different projection layers are utilized to independently generate initial representations for the two types of graph information. For simplicity, MLPs are used to process the adjacency matrix A ∈ R n×n and the attribute matrix X ∈ R n×f , producing the corresponding initial representations Z and B, that is,
Z 0 = M LP A (A), B 0 = M LP X (X) ∈ R n×d ,(6)
where M LP A (•) and M LP X (•) term the MLPs for processing topology and attributes, respectively.
this section cite: []

Section: Dual Cross-attention Module.
As an implementation of the cross-aggregation (in Definition 1), this module is designed to capture the interactions between these two types of graph information. However, directly employing the cross-attention (Eq. 4) may result in two issues: (1) unacceptable quadratic computational complexity due to the calculation of dot products for all node pairs, and (2) an increased number of parameters and overfitting risk due to the use of two separate channels.
To alleviate these drawbacks, the proposed Dual Cross-Attention (DCA) module adopts two strategies:
(1) linearized attention computation [50] and (2) parameter sharing, as shown in Fig. 1(b). Firstly, through approximating or replacing the Softmax attention utilizing separate kernel functions, the computation order in the SA module can be reordered from the standard (Query×Key)×Value (Eq.
3) to the more efficient Query×(Key×Value) format [28]. However, this strategy is unsuitable for the cross-attention module. Specifically, the attention score k(B) ⊤ v(B) ∈ R d×d computes the similarity between features within the same space, rather than across different spaces. To ensure cross-space interaction, DCA sets the Key to originate from the same space as the Query. Moreover, DCA shares parameters between the Query and Key to reduce the number of parameters.
To streamline the description of the update process for topology representations and attribute representations, two abstract representations H 1 and H 2 are introduced. For clarity, layer indices are omitted. The general formulation of the DCA module is given as follows:
DCA(H 1 , H 2 ) : Q = q(H 1 ), K = k(H 1 ), V = v(H 2 ),(7)
Q = Q ∥Q∥ F , K = K ∥K∥ F ,(8)
H * 1 = D -1 V + 1 n Q( K⊤ V) ,(9)
where q(•) and k(•) stand for the Query and Key functions, respectively, with q(•) = k(•). And v A (•) represents the Value function. These functions are implemented as MLPs. ∥ • ∥ F denotes the Frobenius norm. D = Diag(1 + 1 n Q( K⊤ 1)) stands for a diagonal matrix and 1 is an all-one vector. The topology-related attribute representations can be obtained as B * DCA = DCA(B, Z). Then, the topology representations are updated via
Ẑ = (1 -λ) ÃV + λB * DCA ,(10)
where Ã denotes the normalized adjacency matrix. The first term denotes the topology representation updated purely from the topology space, which can be viewed as being obtained via spectral clustering [48,54] (see Theorem 3). λ = Tanh(λ) stands for a scalar to balance these two terms, where λ is a learnable parameter. Combining these two terms allows for the fusion of topological details alongside the topology-related attribute information into the final topology representations.
Similarly, the attribute representations are updated by incorporating relevant information from the topology space, that is, Z * DCA = DCA(Z, B), with their representations. This can be expressed as
B = (1 -γ)B 0 + γZ * DCA ,(11)
where γ = Tanh(γ) denotes a scalar to trade off the two terms with γ denotes a learnable parameter. Note that DCA requires two separate sets of network parameters to generate the attribute and topology Prediction Layer. After obtaining the topology representations Ẑ and attribute representations B through l layers, the final node representations can be generated by weight combining them. Next, the predictions are generated via an MLP network and nonlinearities (i.e., Sof tmax(•)), that is,
Ŷ = Sof tmax M LP (1 -α) Ẑ + α B ,(12)
where α denotes a scalar that adjusts attention to topology and attribute representations. Ŷ ∈ R n×c represents the predictions, indicating the estimated outcomes for each of the n nodes across c classes.
Objective Function. Note that the proposed DCA module, with a large number of parameters across two distinct spaces, is susceptible to representation distortion caused by overfitting [8], especially when the number of training nodes is limited. Thus, a consistency constraint is introduced to align the two representations Ẑ and B. First, pseudo-labels are derived by averaging the two representations. For a node v, its pseudo-label can be computed as y v = 1 2 (ẑ v + bv ). Next, low-entropy pseudo-labels are obtained through a sharpening technique that controls the sharpness of the distribution. This can be formulated as ȳi,j = y 1 τ i,j / c-1 k=0 y i,j , (0 ≤ j ≤ c -1), where τ ∈ (0, 1] denotes a scaling factor that controls the sharpness of the distribution.
Once the pseudo-label is obtained, the next step is to calculate the squared Euclidean distance between it and the two representations, which is given by
L con ( Ẑ, B) = 1 2 n-1 i ∥ȳ i -ẑi ∥ 2 2 + ∥ȳ i -bi ∥ 2 2 .(13)
The overall objective of UGCFormer is to minimize the weighted sum of the cross-entropy loss and the consistency loss, defined as follows:
L overall = L ce + βL con ,(14)
where L ce = -v∈V L y v log ŷv and β stands for a balance hyperparameter.
this section cite: ['b49', 'b27', 'b47', 'b53', 'b7']

Section: Model Analysis
This subsection provides a comprehensive analysis of UGCFormer. First, the computational complexity of UGCFormer is analyzed. Then, the simplicity of UGCFormer is examined through architectural comparison with existing GTs. Finally, the effectiveness of UGCFormer is theoretically justified.
this section cite: []

Section: Complexity Analysis.
UGCFormer operates with linear time complexity. The time complexity for generating initial representations through the projection layer is O(md+nd 2 ) as the adjacency matrix is sparse, where m represents the number of edges. Secondly, owing to the linearized cross-attention module, the aggregation operator incurs a computational overhead of O(nd 2 ). Finally, obtaining the predictions involves feature mapping and element-wise operations, resulting in a complexity of O(nd). UGCFormer operates with linear space complexity. The space required to store the input topology and attributes is O(m + nd), where m corresponds to the number of edges and nd accounts for the feature matrix. The aggregated and updated representations each require O(nd) space, since their dimensions do not exceed those of the input feature matrix. In the linearized attention computation (Fig. 1(b)), the attention matrix contributes an additional O(d 2 ) space overhead.
this section cite: []

Section: Components.
To leverage discriminative graph topology and capture structural biases, existing GTs often resort to auxiliary components that compromise their efficiency and effectiveness. Specifically, the positional or structural encodings (e.g., Laplacian eigenvector encodings) used in GraphGPS [39], NAGphormer [2], Exphormer [44], and GOAT [31] as well as augmented training losses (e.g., edge regularization loss) in NodeFormer, often necessitate cubic computational complexity and quadratic space consumption. Moreover, the GNN module tends to generate representations that are susceptible to issues caused by the limited message passing. In contrast, the proposed UGCFormer features a streamlined and efficient design that relies solely on a linear cross-attention module.
Theoretical Justification. Though designed to be simple and intuitive, the proposed UGCFormer is theoretically guaranteed to be effective from a graph optimization perspective [55,58]. Theorem 3. Let Z and B denote the topology representations and attribute representations, respectively. The representation update in the dual cross-attention module DCA (Eq. 10 and Eq. 11)is equivalent to solving an optimization problem with the objective function:
arg min Z,B λ Tr(Z ⊤ LZ) + ∥B -M LP (X)∥ 2 F -η∥Z ⊤ B∥ 2 F ,(15)
where L terms the Laplacian matrix of Ã, λ and η are the scalars used to balance these three terms.
In Eq. 15, the first term stands for a relaxed optimization problem widely used in spectral clustering [48]. Thus, the DCA seeks to generate topology representations that capture mesoscopic community structures. The second term measures the distance between the attribute representation B and its initial representation M LP (X). The third term denotes the statistical dependence measure, approximated by the Hilbert-Schmidt Independence Criterion (HSIC) [19], that is, HSIC(Z, B) ≈ Tr(ZZ ⊤ BB ⊤ ) = ∥Z ⊤ B∥ 2 F , which reflects the dependence between topology and attribute representations. Therefore, the interaction, whether mutual correlation (positive weights) or exclusion (negative weights), can be modulated by the parameters. In summary, Theorem 3 indicates that UGCFormer focuses on learning representations by mining the interactions of two basic graph information.
this section cite: ['b38', 'b1', 'b43', 'b30', 'b54', 'b57', 'b47', 'b18']

Section: Experiments
This section evaluates the effectiveness and universality of the proposed UGCFormer by comparing its performances against various diverse graph learning models on the node classification task. Moreover, it provides additional analysis experiments to enhance the understanding of UGCFormer. Refer to Section E for details on the datasets, baselines, and experimental setups.
this section cite: []

Section: Experimental Results
Homophilic Graphs. The experiment results for node classification on homophilic graphs are shown in Tab. 1, from which three key observations can be made. Firstly, the performance of the backbone GNNs (e.g., GCN and GAT) lags behind that of GTs. To be specific, on six of the seven homophilic graphs, the models that rank in the top two positions are GTs. This is primarily because most GTs, such as NAGphormer, are built upon these backbone GNNs and specifically address the shortcomings of GNNs in capturing long-range dependencies. Secondly, the proposed UGCFormer outperforms all baseline GTs across six of the seven datasets and achieves the optimal rank, demonstrating its consistent superior performance. In particular, on PubMed, UGCFormer achieves a performance that is 2.55% higher than the baseline Polynormer, which has an average rank of second, and its average rank is significantly lower. Thirdly, compared with the baseline LINKX, which also processes graph topology and node attributes separately and does not leverage message passing, UGCFormer consistently achieves better results across all datasets. This can be attributed to its ability to capture the interactions between these two types of graph information and alleviate the representation distortion, which LINKX does not account for. This highlights the rationality of UGCFormer's design.
Heterophilic Graphs. Tab. 2 shows the results of the node classification task on seven heterophilic graphs, highlighting three key observations. Firstly, the baseline GTs perform slightly better than the baseline GNNs, but the difference is not substantial. In specific, the baseline GNNs, particularly GloGNN on Cornell, Texas, and Wisconsin, and GraphSAGE on the Ratings, achieve top-two results on five of the seven datasets. This can be attributed to the high complexity and large number of parameters in GTs, which make them prone to overfitting. Therefore, the baseline SGFormer, which linearly combines the local representation from the GNN module and the global representation from the GT module, achieves superior performance. This is evidenced by its ranking in the top two for three datasets. Secondly, the proposed UGCFormer outperforms the GT baselines on the majority of heterophilic graphs, proving its effectiveness. For example, on Cornell, UGCFormer exceeds the second-ranked GT, i.e., GOAT, by a significant margin of 1.96%. Thirdly, UGCFormer consistently outperforms the baseline LINKX on all heterophilic datasets, highlighting the significance of capturing the relevance between graph topology and node attributes. Overall, UGCFormer achieves performance improvements on both homophilic and heterophilic graphs, demonstrating its universality.
this section cite: []

Section: Scalability Study.
To evaluate the scalability of the proposed UGCFormer, this experiment quantitatively changes the network size and records the running time and GPU memory usage. Specifically, it utilizes the ogbn-arxiv to randomly sample subsets of nodes, with the node numbers varying from 10K to 100K. As shown in Fig. 2, the running time and GPU memory usage of UGCFormer increase linearly with the size of the sampled graph. For example, the training time and memory usage with 100k nodes are approximately five times higher than with 20k nodes. This indicates that UGCFormer exhibits linear time and space complexity, consistent with the conclusion in Section 3.3.
this section cite: []

Section: Node Property Prediction.
This experiment seeks to evaluate the effectiveness and scalability of GTs by comparing them with GNNs on two large-scale benchmark datasets. Upon examining Tab. 3, which presents the results of the node property prediction task on these two datasets, two key conclusions can be drawn. Firstly, the backbone GTs generally outperform the backbone GNNs, which not only highlights the superiority of GTs but also underscores their scalability-a key challenge that GTs aim to address. This can be attributed to the integration of the GNN blocks in GTs, exemplified by SGFormer. These GTs generate the final prediction by combining the local representations from the GNN module with the global representations from the GT module. Secondly, the proposed UGCFormer achieves optimal performance on these two datasets, indicating its effectiveness and scalability on large graphs.
this section cite: []

Section: Additional Analysis
Ablation Study. This experiment evaluates the contributions of the proposed cross-attention module and the consistency constraint by comparing UGCFormer with two variants lacking these components. Fig. 3 shows that these variants consistently underperform UGCFormer across the four datasets. This illustrates that the efficacy of UGCFormer stems from the collective contribution of all components. Besides, even without the consistency loss, the variant model (w/o L con ) still provides competitive performance compared to the baseline GTs, as seen in Table 1. This highlights the effectiveness of the cross-attention module and thereby reaffirms the rationality of the UGCFormer architecture. Performance(%) Datasets w/o CA w/o L_con UGCFormer Figure 3: Impact of functional components (i.e., the CA and consistency constraint).
this section cite: []

Section: Parameter Sensitivity Analysis.
These experiments aim to provide an intuitive understanding for the selection of hyperparameters. Performance changes due to varying the number of layers (l) and layer dimensions (d) are shown in Figs. 4 and 5, respectively. Number of Layers. Fig. 4 shows that UGCFormer achieves stable performance across various layer numbers {1, 2, 3, 4, 5}. Specifically, performance fluctuations are minimal, within 2.2% on the Cora, 1.4% on the CiteSeer, and 1.3% on PubMed. This indicates that UGCFormer is relatively insensitive to the number of layers. Additionally, optimal performance is achieved with {3, 4}, likely due to the risk of over-smoothing in deeper models. Hidden Layer Dimension. As shown in Fig. 5, UGCFormer maintains consistent performance across the hidden dimension range {64, 128, 256, 512}. For example, on the Cora, which shows the most significant performance variation, the difference is less than 2%. This indicates that UGCFormer is not sensitive to this parameter. Additionally, optimal performance on the three datasets corresponds to d ∈ {128, 256}, rather than the highest value of 512. This suggests that larger dimensions can lead to overfitting and distorted representations. Additional hyper-parameters (including α and β) are analyzed in Section E.4.
this section cite: []

Section: Conclusions
By revisiting two typical Graph Transformers (GTs), this study has uncovered a potential functional mechanism: cross-aggregation between graph topology and node attributes. To effectively implement this mechanism, this paper introduces UGCFormer, a linearized graph cross-attention Transformer.
Extensive experiments on sixteen graph benchmarks demonstrate its effectiveness and efficiency.
this section cite: []

Section: References
Ref_id:b0 Title: Structure-aware transformer for graph representation learning Year: (2022)
Ref_id:b1 Title: Nagphormer: A tokenized graph transformer for node classification in large graphs Year: (2023)
Ref_id:b2 Title: Rethinking tokenized graph transformers for node classification Year: (2025)
Ref_id:b3 Title: Leveraging contrastive learning for enhanced node representations in tokenized graph transformers Year: (2024)
Ref_id:b4 Title: Adedgedrop: Adversarial edge dropping for robust graph neural networks Year: (2025)
Ref_id:b5 Title: Adaptive universal generalized pagerank graph neural network Year: (2021)
Ref_id:b6 Title: Polynormer: Polynomial-expressive graph transformer in linear time Year: (2024)
Ref_id:b7 Title: Avoiding overfitting: A survey on regularization methods for convolutional neural networks Year: (2022)
Ref_id:b8 Title: Long range graph benchmark Year: (2022)
Ref_id:b9 Title: On the benefits of attribute-driven graph domain adaptation Year: (2025)
Ref_id:b10 Title: Homophily enhanced graph domain adaptation Year: (2025)
Ref_id:b11 Title: Structure-preserving graph representation learning Year: (2022)
Ref_id:b12 Title: Fast graph representation learning with pytorch geometric Year: (2019)
Ref_id:b13 Title: Multiplex heterogeneous graph neural network with behavior pattern modeling Year: (2023)
Ref_id:b14 Title: Cross-attention is all you need: Adapting pretrained transformers for machine translation Year: (2021)
Ref_id:b15 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b16 Title: How does over-squashing affect the power of gnns? Year: (2024)
Ref_id:b17 Title: Self-paced co-training of graph neural networks for semi-supervised node classification Year: (2023)
Ref_id:b18 Title: Measuring statistical dependence with hilbert-schmidt norms Year: (2005)
Ref_id:b19 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b20 Title: A survey on vision transformer Year: (2022)
Ref_id:b21 Title: Str-gcl: Structural commonsense driven graph contrastive learning Year: (2025)
Ref_id:b22 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b23 Title: Does gcl need a large number of negative samples? enhancing graph contrastive learning with effective and efficient negative sampling Year: (2025)
Ref_id:b24 Title: One prompt fits all: Universal graph adaptation for pretrained models Year: (2025)
Ref_id:b25 Title: Ccnet: Criss-cross attention for semantic segmentation Year: (2023)
Ref_id:b26 Title: Global selfattention as a replacement for graph convolution Year: (2022)
Ref_id:b27 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b28 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b29 Title: Predict then propagate: Graph neural networks meet personalized pagerank Year: (2019)
Ref_id:b30 Title: GOAT: A global transformer on large-scale graphs Year: (2023)
Ref_id:b31 Title: Dual-channel multiplex graph neural networks for recommendation Year: (2025)
Ref_id:b32 Title: Finding global homophily in graph neural networks when meeting heterophily Year: (2022)
Ref_id:b33 Title: Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods Year: (2021)
Ref_id:b34 Title: A survey of transformers Year: (2022)
Ref_id:b35 Title: Gradformer: Graph transformer with exponential decay Year: (2024)
Ref_id:b36 Title: Geom-gcn: Geometric graph convolutional networks Year: (2020)
Ref_id:b37 Title: A critical look at the evaluation of gnns under heterophily: Are we really making progress? Year: (2023)
Ref_id:b38 Title: Recipe for a general, powerful, scalable graph transformer Year: (2022)
Ref_id:b39 Title: Multi-scale attributed node embedding Year: ()
Ref_id:b40 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b41 Title: Collective classification in network data Year: (2008)
Ref_id:b42 Title: Pitfalls of graph neural network evaluation Year: (2018)
Ref_id:b43 Title: Exphormer: Sparse transformers for graphs Year: (2023)
Ref_id:b44 Title: Social influence analysis in large-scale networks Year: (2009)
Ref_id:b45 Title: Attention is all you need Year: (2017)
Ref_id:b46 Title: Graph attention networks Year: (2017)
Ref_id:b47 Title: A tutorial on spectral clustering Year: (2007)
Ref_id:b48 Title: Nodeformer: A scalable graph structure learning transformer for node classification Year: (2022)
Ref_id:b49 Title: Simplifying and empowering transformers for large-graph representations Year: (2023)
Ref_id:b50 Title: Representing long-range context for graph neural networks with global attention Year: (2021)
Ref_id:b51 Title: Graph convolutional network with elastic topology Year: (2024)
Ref_id:b52 Title: Less is more: on the overglobalizing problem in graph transformers Year: (2024)
Ref_id:b53 Title: Graph contrastive learning with joint spectral augmentation of attribute and topology Year: (2025)
Ref_id:b54 Title: Why do attributes propagate in graph convolutional neural networks? Year: (2021)
Ref_id:b55 Title: Restormer: Efficient transformer for high-resolution image restoration Year: (2022)
Ref_id:b56 Title: Dual feature interaction-based graph convolutional network Year: (2023)
Ref_id:b57 Title: Interpreting and unifying graph neural networks with an optimization framework Year: (2021)
Ref_id:b58 Title: Propagation is all you need: A new framework for representation learning and classifier training on graphs Year: (2023)
Ref_id:b59 Title: Graph contrastive learning reimagined: Exploring universality Year: (2024)
Ref_id:b60 Title: Dualformer: Dual graph transformer Year: (2025)
Ref_id:b61 Title: Improving graph contrastive learning via adaptive positive sampling Year: (2024)
