Title: L2DGCN: Learnable Enhancement and Label Selection Dynamic Graph Convolutional Networks for Mitigating Degree Bias
Abstract: Graph Neural Networks (GNNs) are powerful models for node classification, but their performance is heavily reliant on manually labeled data, which is often costly and results in insufficient labeling. Recent studies have shown that message-passing neural networks struggle to propagate information in low-degree nodes, negatively affecting overall performance. To address the information bias caused by degree imbalance, we propose a Learnable Enhancement and Label Selection Dynamic Graph Convolutional Network (L2DGCN). L2DGCN consists of a teacher model and a student model. The teacher model employs an improved label propagation mechanism that enables remote label information dissemination among all nodes. The student model introduces a dynamically learnable graph enhancement strategy, perturbing edges to facilitate information exchange among low-degree nodes. This approach maintains the global graph structure while learning graph representations. Additionally, we have designed a label selector to mitigate the impact of unreliable pseudo-labels on model learning. To validate the effectiveness of our proposed model with limited labeled data, we conducted comprehensive evaluations of semi-supervised node classification across various scenarios with a limited number of annotated nodes. Experimental results demonstrate that our data enhancement model significantly contributes to node classification tasks under sparse labeling conditions.

Section: Introduction
Graphs model structured and relational systems and are used in fields like traffic networks [1], molecular structures [2], and protein networks [3]. Graph learning algorithms analyze graph-structured data by considering node features and their relationships (edges), achieving success in many domains.
Graph neural networks (GNNs), based on message-passing mechanisms, are a key technology for handling graph data. Node classification, a core task related to graphs, has received much attention. Traditional GNN methods rely on supervised learning with numerous labeled nodes, but high labeling costs limit their practicality. Researchers have combined self-training and pseudo-labeling techniques with GNNs to improve semi-supervised node classification under limited labeled data. However, these methods still struggle when labeled nodes are scarce or unlabeled nodes are abundant. In a graph, a node's degree usually refers to the number of edges connected to that node, reflecting its connection strength and importance within the network, significantly influencing the path and speed of information propagation. However, in real-world scenarios, the degree distribution of nodes often follows a power-law distribution [4], where most nodes are low-degree nodes, and only a few have very high degrees. We believe that this degree imbalance is a primary reason for the model's failure in situations with few labeled nodes. According to related influence theories [5], the label of a target node is affected by the cumulative influence of the normalized features of its neighboring nodes within a K-hop propagation range. For low-degree nodes, every connection is crucial for their correct classification, but due to their limited number of connections, their information propagation capacity is significantly restricted, thus affecting the effectiveness of the propagation path. Figure 1 illustrates the impact of the degree of the node on the propagation of information when the number of labeled nodes is limited. In Figure 1 (a), the degree-0 node (labeled node 1 and unlabeled nodes 2 and 3) cannot transmit information; in Figure 1 (b), the 1-degree node is misclassified due to incorrect connections; whereas in Figure 1 (c), despite the presence of some erroneous connections, nodes with degrees greater than 1 still achieve correct classification. The insufficient connections of low-degree nodes limit information propagation, especially in the case of scarce labeled nodes. Therefore, in this context, graph neural networks require multi-step propagation to effectively transmit information, highlighting the importance of addressing low-degree nodes in the effective dissemination of information.
In the research on the degree bias problem in Graph Neural Networks (GNNs), several studies have explored solutions from different perspectives: DegFairGT [6] proposes a learnable structural enhancement and structural self-attention mechanism, which generates new edges by calculating the structural similarity between node pairs to balance message passing, and retains the global topology with the help of a self-supervised task based on p-step transition probability matrices, thereby alleviating the problem of insufficient information in low-degree nodes and over-smoothing in high-degree nodes; DAHGN [7], on the other hand, focuses on heterogeneous information networks, constructs a dual-view contrast framework of heterogeneous views and homogeneous subgraphs, combines semi-supervised task loss and contrast loss, and adopts differentiated strategies for low-degree and high-degree nodes to eliminate degree bias, filling the gap in research on heterogeneous scenarios; GraphPatcher [8] innovatively realizes model-agnostic degree bias mitigation through test-time augmentation, iteratively generates virtual nodes to repair damaged neighborhoods of low-degree nodes, improving the performance of low-degree nodes (by an average of 6.5%) while preserving the advantages of high-degree nodes, with an overall performance improvement of 3.6% on average. However, existing methods still have limitations in sparse labeling scenarios: DegFairGT's structural enhancement has limited adaptability to dynamic topologies and is difficult to directly cope with the challenge of insufficient labeled data; DAHGN's contrastive learning framework is insufficient in handling the reliability of pseudo-labels in complex heterogeneous networks; GraphPatcher, as a test-phase strategy, cannot fundamentally optimize the information aggregation of low-degree nodes during training.
Our proposed learnable enhanced and label selection dynamic graph convolutional network aims to address the poor node classification performance caused by degree bias in the absence of labeled nodes. The model comprises a teacher model and a student model. The teacher model generates pseudo-labels for unlabeled nodes via an improved label propagation method. The student model dynamically learns from the graph through two approaches: 1) Structural Optimization: It prunes edges based on node degree, retains core edges, and uses high-order feature information from a decoupled GCN to enhance the topological structure of low-degree nodes, ensuring balanced information flow. 2) Pseudo-Label Selection: A pseudo-label selector combines nodes with high confidence and removes those with low confidence, dynamically updating the training set. Our contributions include:
• Proposing a dynamic graph convolutional network based on learnable augmentation and label selectors to address degree bias caused by the scarcity of labeled nodes.
• Introducing a teacher model that uses soft pseudo-label propagation to expand the training set.
• Designing a student model that performs edge pruning based on node degree and integrates higher-order node features for dynamic topology learning, thereby mitigating degree bias effects while using the label selector to enhance the training set.
• Demonstrating through extensive experiments on multiple datasets that our model is highly effective for semi-supervised node classification with extremely limited labeled nodes, particularly in alleviating degree bias.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']

Section: Related Work
Graph Convolutional Networks Graph Convolutional Networks (GCNs) have significantly advanced graph learning, being mainly divided into spectral and spatial graph convolutions. Spectral graph convolution processes graph signals using graph spectral theory, Fourier transforms, and convolution theorems [9][10], while spatial graph convolution extracts features by passing and aggregating information from neighboring nodes. Common GCN models such as GCN [11], GAT [12], and SGC [13] typically employ a coupled structure of propagation and aggregation. However, in scenarios with scarce labeled nodes, increasing model depth is often considered a solution, though deeper models like GCNII [14] see a significant increase in computational complexity. Consequently, decoupled GCNs have gained attention for their stability and flexibility. For instance, some approaches expand the receptive field through decoupled transformation and propagation, APPNP [15] optimizes global information utilization by combining personalized PageRank, GAMLP [16] maintains high scalability during pre-computation, while DecGCN [17] focuses on improving model stability and generalization capabilities.
Node Classification With Few Labels In semi-supervised node classification tasks with limited labeled nodes, traditional GCN models often face performance issues due to insufficient supervision. Recent graph learning methods aim to address this challenge. For example, CGPN [18] uses Poisson learning to counteract Laplacian performance degradation, M3S [19] enhances GCN generalization through a self-supervised multi-stage training framework, IGCN [20] introduces a unified graph filtering approach to reduce overfitting and training parameters, GraphHop [21] improves graph signal smoothing with a two-stage training process, AGST [22] enhances decision boundary separation by capturing remote node interactions through self-training, and CMPGNN [23] presents a noise-resistant framework via contrastive message passing. PASTEL [24] addresses the problems of insufficient information and excessive suppression caused by "topological imbalance" by proposing a position-aware graph structure learning framework. It enhances intra-class connections and optimizes edge weights through anchor position encoding, alleviating structural biases at the level of propagation paths. NodeMixup [25] focuses on the insufficient reachability between labeled and unlabeled nodes, designing a cross-set mixing strategy and neighbor label distribution-aware sampling to enhance information interaction through node pair mixing without adjusting the GNN architecture. Another study [26] expands the GNN's receptive field to skip neighborhoods through position encoding, adding virtual nodes/edges to the input graph and injecting position features to achieve model-agnostic receptive field expansion, avoiding complex architectural modifications. Despite these advances, existing methods often overlook the reliability of pseudo-labels, which can lead to inaccuracies in model training, MSP-LR [27] introduces a label regularization method and proposes a graph neural network with basic learning and label regularization modules to enhance label reliability through pseudolabeling and regularization based on the cluster assumption.
this section cite: ['b8', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26']

Section: Graph Representation Learning
In graph representation learning, edges are crucial for information dissemination, but real-world graphs often contain noisy edges. Researchers have developed several approaches to optimize graph structures. JLGCN [28] transforms graph optimization into distance metric learning using the Mahalanobis distance metric. [29] proposes an end-to-end joint fusion framework aiming for consistent feature integration and adaptive topology tuning. IDGL [30] frames graph learning as similarity metric learning, iteratively refining graph structures and embeddings. [31] introduces a self-supervised framework that combines graph structure learning, clustering for pseudo-labels, and sample selection for clean labels. [32] leverages information theory to maximize mutual information for reconstructing topological transformations. [33] presents a method that simultaneously integrates graph learning and graph convolution into a unified network architecture and enforces label smoothing through unsupervised loss terms. [34] uses graph structure refinement to eliminate irrelevant noise and simultaneously maximizes view-shared and view-unique task-relevant information, thereby tackling the frontier of non-redundant multiplex graph.
this section cite: ['b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33']

Section: Proposed Method
This section describes our proposed node classification model for under-labeled scenarios. We start with a mathematical formulation of the problem, followed by an overview of the model's architectural framework and a detailed exposition of its key components in the subsequent subsections.
this section cite: []

Section: Problem formulation
An undirected graph with e edges and n nodes can be represented as a quintuple:
G = (V, E, X)(1)
where V and E denote the set of nodes and edges, respectively, and X ∈ R n×d denotes the initial feature matrix of the graph G with n nodes each having d features. The adjacency matrix A ∈ {0, 1} n×n represents the connections between nodes, and Ã includes self-loops. Let D be the diagonal matrix of Ã, and Ŝsym = D-1/2 Ã D-1/2 denotes the symmetric normalized adjacency matrix with self-loops. The labeling matrix is denoted by Y ∈ R n×c , where c is the number of classes. In a semi-supervised setup, the node set V is divided into labeled V L and unlabeled V U sets. Our focus is on predicting labels for V U given a few labeled nodes in V L , which may be balanced or unbalanced across classes. If each class has K labeled nodes, the problem becomes under-shot semi-supervised node classification.
this section cite: []

Section: Architecture Overview
In this section, we will provide a detailed introduction to our proposed L2DGCN, a self-training graph convolutional network model. This model effectively improves the performance of GCN when labeled nodes are scarce by mitigating degree bias through three innovative designs across two key modules. Figure 2 illustrates the overall architecture of L2DGCN.
this section cite: []

Section: Teacher model based on the propagation of soft pseudo-labels
In self-training models, teacher models typically use labelled data to predict unlabelled data, providing pseudo-labelling information for unlabelled nodes.Whereas in the label propagation algorithm [35], the labels of the tagged nodes are reset to their true labels after each iteration, which means that the influence of other nodes on the labels of the tagged nodes is absorbed in the subsequent iterations.Therefore, we make use of soft pseudo labels for long range propagation in order to generate pseudo labels for unlabelled nodes. The improved soft label propagation formula is as follows:
Ỹ (K) = (1 -α) Ŝiter. sym Ỹ (K-1) + αY (0) Ŝiter. sym = ( Diter. A ) -1/2 Ãiter. A ( Diter. A ) -1/2(2)
where
Y (0) = Y ∈ R n * c
is the initial label matrix, K is the order of propagation, Ãiter.
A = A iter.
A + I denotes the adjacency matrix with self-loop added after the iter.th matrix optimization, and D iter. A is the diagonal matrix of Ãiter.
A , where diter.
i = n j=1 Ãiter. ij , D iter. ii = d iter. i , Ŝiter.
sym is the normalized adjacency matrices, and for the convenience of writing, in the following expressions we omit the superscript iter. α is the balanced higher-order pseudo-labels with the initial pseudo-labels hyperparameters. By setting an appropriate α , the model can effectively maintain accurate perception of local structures even after multiple propagations, thus improving the overall learning effect and model performance.
this section cite: ['b34']

Section: Student model based on dynamic learnable graph augmentation and label selector

this section cite: []

Section: Decoupled GCN backbone
Graph Convolutional Networks (GCNs) combine neighbourhood aggregation and feature transformation for node representation learning. However, recent studies indicate that this coupled design can cause issues like training difficulties, underutilisation of graph structures, and excessive smoothing. To address these, we adopt a decoupled GCN model for graph feature propagation, enabling higher-order feature interactions.
X (T ) = (1 -β) Ŝsym X (T -1) + βX (0) X(T ) = (ReLU(X (T ) W 1 ))W 2 T ≥ 1 (3
)
where β is a positive parameter that balances the initial and higher-order feature information, similar to α. By setting β appropriately, the model can preserve initial feature information even with infinite propagation. At the first iteration, X (0) = X represents the initial feature information. While a large number of propagation steps T allows for extensive higher-order interaction, excessive T introduces noise, making the classification boundaries less distinct.
this section cite: []

Section: Dynamic Learnable Graph Enhancement
Real-world graphs have many low-degree nodes whose edges are crucial for information dissemination. Incorrect or missing edges can significantly affect these nodes. To boost model performance, especially with limited labeled data, we dynamically optimize the graph topology using node degrees and features.
Edge Pruning Based on Node Degrees A node's degree is its edge count. High-degree nodes are central to the graph's structure and information flow, while edges of low-degree nodes are especially important. We propose pruning the graph by keeping edges of high-degree nodes and removing those of low-degree nodes to reduce the impact of incorrect edges. The steps are as follows.
Step 1: Determine the number of nodes to preserve the edges:
M = (1 -β w ) * n (4
)
where β w is the pruning rate β w ∈ [0, 1],(1 -β w ) is the ratio of the number of total nodes to the number of nodes whose edges are to be retained, n is the total number of nodes, and M is the number of nodes to be retained.When β w = 1 our model can degenerate into an MLP with self-training,when β w = 0 our model can be seen as an APPNP model with self-training.
Step 2: Sort the nodes in descending order based on the node degree d(v) and select the first M nodes, denoted as V s :
V s = {v 1 , v 2 , . . . , v M } (5
)
where d(v 1 )d(v 2 ) . . . d(v M ), D is the degree matrix , A ∈ R n×n is the initial adjacency matrix, where the choice of optimising the initial graph structure each time avoids introducing cumulative errors and local optimal solution problems in the optimisation process, and ensures the controllability and stability of the optimisation process.
Step 3: For the selected node V s , all edges E s connected to it are retained, while all edges of the nodes that are not selected and those that are also not selected when the retained edges are selected are removed. The obtained selected edges and the pruned adjacency matrix
A W are E s = E s ∪ e ij , i or j ∈ V s (6
)
A W (i, j) = 1, e ij ∈ E s 0, otherwise,(7)
This operation filters the top M nodes by degree and retains the edges between them. By selecting the right q value, the model can retain the minimal edges for optimal results, offering insights into graph data compression and preservation.
Graph enhancement based on dynamic similarity matrix The decoupled GCN's backbone supports long-range feature propagation and local attention, enabling the feature matrix to capture global and local features. The similarity matrix from this feature matrix effectively expresses both types of features and is calculated as:
S(x (T ) i , x(T ) j ) = (x (T ) i ) Trans x(T ) j ∥x (T ) i ∥∥x (T ) j ∥(8)
where S(x (T ) i , x(T ) j ) denotes the similarity between nodes i and j. To refine the graph structure and ensure low-degree nodes primarily have core edges, we use a KNN graph for unselected discrete nodes V d , guaranteeing each has at least P neighbors. The graph structure is enhanced by:
A A (i, j) = 1, if e ij ∈ A w or S(x (T ) i , x(T ) j ) ≥ min(τ (x (T ) i , P ), τ (x (T ) j , P )), 0, otherwise. (9
)
Here, e ij ∈ A w indicates A W contains edge e ij , and τ (x (T ) i , P ) returns the similarity between x(T ) i and the P -th similar row vector in X(T ) . This approach leverages the mutual reinforcement between quality features and good structural information to optimize the graph structure for subsequent model iterations.
Self-training label enhancement Self-training augments GNNs by generating pseudo-labels for unlabeled nodes using a teacher model, but initial pseudo-labels can be unreliable. We introduce a method with a confidence threshold µ, where only pseudo-labels above µ are used, ensuring highconfidence contributions. The student model determines node confidence via the formula:
Y (0) = (ReLU(XW 1 ))W 2 Y (T ) = (1 -β) Ŝsym Y (T -1) + β Y (0) T ≥ 1(10)
In the student model, parameter T controls information dissemination. A larger T captures broader node info, aiding complex graph understanding but risking noise overload in suboptimal graphs. A smaller T limits info to local nodes, possibly missing higher-order relationships. Balancing info coverage and noise is key for effective learning.
Next, according to the set threshold µ, the high confidence unlabelled nodes are selected for supervised learning of the model, i.e:
V U ′ = V U ′ ∪ {v i , ỹ(T ) i } for i ∈ V U such that ỹ(T ) i ≥ µ (11
)
where V U is the initial set of unlabelled nodes and V U ′ is the set of reliable unlabelled nodes.
In order to utilise reliable pseudo-labels to assist in labelling the training set for model training, the standard semi-supervised learning objective function can be modified into the following form:
L = L L ce + λL U ′ ce (12
)
where L L ce is the cross-entropy loss of the labelled nodes and L U ′ ce is the cross-entropy loss of the unlabelled nodes in the student model with a confidence level greater than µ. i.e:
L L ce = - vi∈V L C c=1 y c i log ỹc i (13
)
L U ′ ce = - vi∈V U ′ C c=1 ŷc i log ỹc i (14
)
where ŷc i denotes the pseudo-labelling information obtained by the teacher model and ỹc i denotes the pseudo-labelling obtained through the student model.Also, in order to maintain the consistency of the information obtained from the teacher model and the student model, we motivate the similarity function to assign large values to the positive pairs and small values to the negative pairs by introducing the infoNCE contrast loss [36,37].
L Inf oN CE = n i=1 -log exp(z i • z + i /τ ) r i=0 exp(z i • z j /τ )(15)
Let z + i be the positive sample of z i , and z j consist of one positive and r negative embeddings. τ adjusts the model's ability to distinguish negative samples; a large τ may equalize negative sample treatment, while a small τ can hinder convergence or generalization. The contrast loss aims to enhance pseudo-label validity by calculating the distance between reliable pseudo-labeled nodes U ′ and class prototypes. The formula for class prototypes is:
C c = 1 |V L c + V U ′ c | vi∈{V L c ∪V U ′ c } xT (16
)
where V L c denotes the set of labelled nodes belonging to class c, V U ′ c denotes the set of reliable pseudo-labelled nodes belonging to class c, and C c denotes the corresponding class prototype of class c. The hard pseudo-tag of the reliable pseudo-tag set U ′ is ŷi = arg max j ŷj i . The contrast loss function after correction using reliable pseudo-tags is:
L CL = vi∈{V L c ∪V U ′ c } -log exp(x i • C ŷi /τ ) C c=1 exp(x i • C c /τ ) (17
)
where C ŷi is the corresponding prototype of node v i . For any node p, its corresponding class prototype is used as a positive sample, and the embedding of other class prototypes is a negative sample. Then the loss function of the whole model is:
L = L L ce + λL U ′ ce + γL CL (18
)
where λ and γ are hyperparameters that balance the loss of unlabelled nodes and contrast.
this section cite: ['b35', 'b36']

Section: Computational Complexity Analysis
The computational complexity of the proposed model is determined by the core operations of both the teacher model (for pseudo-label generation) and the student model (for dynamic graph enhancement, feature propagation, and pseudo-label selection), depending on key parameters: graph scale (number of nodes n, number of edges E), feature dimension d, and propagation steps K (for the teacher model) or T (for the student model); to align the labels (including pseudo-labels) generated by the teacher model with the node pseudo-labels obtained by the student model and ensure consistency in their iterative propagation processes, we set K = T . Specifically, the teacher model generates pseudo-labels via improved soft label propagation, with a time complexity of O(K • n 2 c) for K iterations (where c denotes the number of classes), as each iteration involves matrix multiplication between an n × n adjacency-related matrix and an n × c label matrix.
For the student model, its complexity comes from three modules: dynamic graph enhancement (with complexity O(n log n + E + n 2 d + n • P • d), including node sorting, edge traversal, KNN graph construction based on cosine similarity calculation, and neighbor supplementation, where P is the number of neighbors per node), decoupled GCN propagation (with complexity O(T • E • d) for T iterations, as each iteration requires feature aggregation over E edges for d-dimensional features), and pseudolabel selection (with complexity O(n • c) from traversing all n nodes to compute label confidence). Combining the teacher and student modules, the overall computational complexity is dominated by terms related to n (notably n 2 from matrix operations), E (notably E • d from graph propagation), and the unified propagation step K = T .
this section cite: []

Section: Experiments
In this section, we conduct experiments to validate our model's effectiveness and robustness with extremely few labeled nodes. Experimental design is as follows:
this section cite: []

Section: Experimental setup
Datasets We validated our model's effectiveness in semi-supervised node classification on six homogeneous graph datasets of varying sizes. These include citation networks Cora, Citeseer, and Pubmed [38], widely used for semi-supervised node classification; Coauthor-CS and Coauthor-Physics [39] datasets for academic collaboration analysis; and Amazon-Photo [39], which comprises product images with metadata labels. Detailed dataset information is in Table 1. Balancing setup at low labelling rates: We built the training set by randomly selecting 3, 5, or 10 labeled nodes per category (3-shot, 5-shot, 10-shot). The validation set had 30 nodes per category, with the rest used for testing [22].
Balancing setup under standard segmentation: Following [11], we used 20 labeled nodes per class for training, with a 500-node validation set and the remaining nodes in the test set.
this section cite: ['b37', 'b38', 'b38', 'b21', 'b10']

Section: Compared Methods:
We compared our model with classical and state-of-the-art methods, including GCN [11], GAT [12], SGC [13], label-efficient GCN models (GLP [40], IGCN [40], CGPN [41], CMPGNN [23], GraphHop [21]), and self-trained GNN models (PTA [15], ST-GCNs [42], M3S [19], AGST [22], Muse [43]).
this section cite: ['b10', 'b11', 'b12', 'b39', 'b39', 'b40', 'b22', 'b20', 'b14', 'b41', 'b18', 'b21', 'b42']

Section: Implementation Details
We implemented all algorithms in PyTorch with the Adam optimizer, following original settings when available. Results reported are average accuracies from 10 independent runs. Our model was trained with a maximum of 100 iterations, a learning rate of 0.01, and a regularization weight of 5 × 10 -5 . To prevent overfitting, early stopping was applied, halting training if validation loss didn't improve for 1000 steps. Default parameters included K=T , matrix optimization iterations (iter.), pruning rate β w , P neighboring nodes added, and T feature propagation steps for the decoupled GCN. Hyperparameter ranges are in Table 2, with variations based on dataset labeling rates.For the selection of optimal values of the parameters listed in 2, we adopted the grid search method. Specifically, within the preset parameter ranges, we exhaustively combined the possible values of each parameter, evaluated the model performance on the validation set, and finally selected the parameter combination that made the model perform optimally.
this section cite: []

Section: Main results
In the experiments, we evaluate our model and baselines in semi-supervised node classification across various labeling rates. The top three models are labeled, with the best in bold and the other two underlined, combining results from [22] and our experiments. Low-labeling rates setting: Tables 3 and 4 show our model outperforms others, achieving up to 8.28% higher accuracy in the 5-shot. This highlights its effectiveness with few labeled nodes. Classical shallow GCNs struggle with limited data, while models using higher-order and pseudo-label info perform better.
this section cite: ['b21']

Section: Standard Divisions experiments:
Table 5 shows our model gains up to 2.39% accuracy over baselines on Cora, Citeseer, and Pubmed datasets, despite the low-labeling focus, underscoring its superior performance.
this section cite: []

Section: Ablation Study
In this section, we evaluated the performance gains of each model component through ablation experiments, including graph structure optimization based on node degree, pseudo-label reliability, and the effectiveness of contrastive loss. Experiments are performed on the Cora and Citeseer datasets under 3-shot, 5-shot, and 10-shot labeling rates.Results in Figure 3 show that in the Cora dataset, label selection significantly improves performance in 3-shot and 5-shot tasks, while learnable graph augmentation is more effective in the 10-shot task. Conversely, this pattern is reversed in the Citeseer dataset, which is attributed to the dataset's unique characteristics. Compared with models retaining contrastive loss , L2DGCN-LC shows significant performance gaps: on Cora, the gap reaches over 20 percentage points in 3-shot scenarios , narrowing to 8 and 5 percentage points in 5-shot and 10-shot scenarios, respectively; on the more complex Citeseer, the gaps are over 20, 14, and 7-8 percentage points in 3-shot, 5-shot, and 10-shot scenarios. These results confirm that contrastive loss helps capture feature associations from limited data, suppresses interference in noisy datasets, and is critical for enhancing model adaptability and classification stability. We will supplement these experiments and analyses to clarify the independent role of contrastive loss, making the conclusions more robust.
this section cite: []

Section: Parameter Analysis
In this section, in order to verify the impact of each hyperparameter on the model performance, we conduct a series of experiments on the Cora and Citeseer datasets under 3 and 5 settings. These experiments are divided into three main parts, and their results are shown in Figure 4:
1)Matrix Optimization Hyperparameters: Evaluated pruning rate (β w ) and number of nearest neighbors (P) on Cora and Citeseen. On Cora, both showed minor fluctuations in accuracy. On Citeseer's 3-shot task, accuracy rose with higher (β w ) and more neighbors, but fluctuated; for 5-shot, no clear trend for (β w ), but P showed a rapid improvement before declining if too large. 2) Pseudo-Labelling Selection Threshold (µ): On 5-shot Cora and Citeseer, performance was stable but in Citeseer's 3-shot task, performance varied significantly and dropped sharply as increased. 3) Decoupled GCN Propagation Steps (T): Cora showed minimal variation. Citeseer had notable fluctuations, especially in the 3-shot task, due to many discrete nodes and random training selection.  Figure 4: Model performance with varied hyperparameters
this section cite: []

Section: Conclusion
To address the problem that existing models perform poorly with limited labelled nodes, we propose a new solution, L2DGCN.The model contains two modules and is designed with three key elements aimed at optimising graph structure learning and efficiently propagating semantic information, thus mitigating the problem of model performance degradation due to insufficiently labelled nodes.We have conducted extensive experiments on several benchmark datasets with low labelling rates, and the results show the effectiveness of solving the graph structure problem in terms of node degree, as well as the precise selection of pseudo-labels to address their unreliability.We realise that considering only features for learnable graph augmentation in the current study is not comprehensive enough, especially for low-homogeneous graphs that may introduce undesirable edges.Therefore, future work will focus on a multifaceted exploration of graph augmentation, aiming to develop more robust classification models.
this section cite: []

Section: References
Ref_id:b0 Title: Traffic demand prediction based on dynamic transition convolutional neural network Year: (2021)
Ref_id:b1 Title: Deepsite: proteinbinding site predictor using 3d-convolutional neural networks Year: (2017)
Ref_id:b2 Title: Ddmut-ppi: predicting effects of mutations on protein-protein interactions using graph-based deep learning Year: (2024)
Ref_id:b3 Title: Investigating and mitigating degree-related biases in graph convolutional networks Year: (2020)
Ref_id:b4 Title: Combining graph convolutional neural networks and label propagation Year: (2022)
Ref_id:b5 Title: Mitigating degree bias in graph representation learning with learnable structural augmentation and structural self-attention Year: (2025)
Ref_id:b6 Title: Dahgn: Degree-aware heterogeneous graph neural network Year: (2024)
Ref_id:b7 Title: Graphpatcher: mitigating degree bias for graph neural networks via test-time augmentation Year: (2023)
Ref_id:b8 Title: Spectral networks and locally connected networks on graphs Year: (2014)
Ref_id:b9 Title: Convolutional neural networks on graphs with fast localized spectral filtering Year: (2016)
Ref_id:b10 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b11 Title: Graph attention networks Year: (2018)
Ref_id:b12 Title: Simplifying graph convolutional networks Year: (2019)
Ref_id:b13 Title: Simple and deep graph convolutional networks Year: (2020)
Ref_id:b14 Title: On the equivalence of decoupled graph convolution network and label propagation Year: (2021)
Ref_id:b15 Title: Graph attention multi-layer perceptron Year: (2022)
Ref_id:b16 Title: Decoupled graph convolution network for inferring substitutable and complementary items Year: (2020)
Ref_id:b17 Title: Contrastive graph poisson networks: Semi-supervised learning with extremely limited labels Year: (2021)
Ref_id:b18 Title: Multi-stage self-supervised learning for graph convolutional networks on graphs with few labeled nodes Year: (2020)
Ref_id:b19 Title: Label efficient semi-supervised learning via graph filtering Year: (2019)
Ref_id:b20 Title: Graphhop: An enhanced label propagation method for node classification Year: (2023)
Ref_id:b21 Title: Toward robust graph semi-supervised learning against extreme data scarcity Year: (2024)
Ref_id:b22 Title: Contrastive message passing for robust graph neural networks with sparse labels Year: (2025)
Ref_id:b23 Title: Position-aware structure learning for graph topology-imbalance by relieving under-reaching and over-squashing Year: (2022)
Ref_id:b24 Title: Nodemixup: tackling under-reaching for graph neural networks Year: (2024)
Ref_id:b25 Title: Rewiring with positional encodings for graph neural networks Year: (2022)
Ref_id:b26 Title: Multi-channel set polynomial based label regularized graph neural networks against extreme data scarcity Year: (2025)
Ref_id:b27 Title: Joint learning of graph representation and node features in graph convolutional neural networks Year: (2019)
Ref_id:b28 Title: Joint learning of feature and topology for multi-view graph convolutional network Year: (2023)
Ref_id:b29 Title: Iterative deep graph learning for graph neural networks: Better and robust node embeddings Year: (2020)
Ref_id:b30 Title: Self-supervised robust graph neural networks against noisy graphs and noisy labels Year: (2023)
Ref_id:b31 Title: Self-supervised graph representation learning via topology transformations Year: (2023)
Ref_id:b32 Title: A unified deep semi-supervised graph learning scheme based on nodes re-weighting and manifold regularization Year: (2023)
Ref_id:b33 Title: Beyond redundancy: Information-aware unsupervised multiplex graph structure learning Year: (2024)
Ref_id:b34 Title: Learning with local and global consistency Year: (2004)
Ref_id:b35 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b36 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b37 Title: Collective classification in network data Year: (2008)
Ref_id:b38 Title: Pitfalls of graph neural network evaluation Year: (2018)
Ref_id:b39 Title: Label efficient semi-supervised learning via graph filtering Year: (2019)
Ref_id:b40 Title: Contrastive graph poisson networks: Semi-supervised learning with extremely limited labels Year: (2021)
Ref_id:b41 Title: Deeper insights into graph convolutional networks for semisupervised learning Year: (2018)
Ref_id:b42 Title: Multiview subgraph neural networks: Self-supervised learning with scarce labeled data Year: (2024)
