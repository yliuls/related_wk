Title: AutoGFM: Automated Graph Foundation Model with Adaptive Architecture Customization
Abstract: Graph foundation models (GFMs) aim to share graph knowledge across diverse domains and tasks to boost graph machine learning. However, existing GFMs rely on hand-designed and fixed graph neural network (GNN) architectures, failing to utilize optimal architectures w.r.t. specific domains and tasks, inevitably leading to suboptimal performance in diverse graph domains and tasks. In this paper, we explore graph neural architecture search (GNAS) for GFMs for the first time, which suffers from the problem of architecture inconsistency, i.e., the optimal architectures for different tasks and domains vary. We tackle this problem by discovering an invariant graph-architecture relationship across domains and tasks, which imposes three challenges: i) how to capture invariant and variant patterns; ii) how to customize architectures to adapt to diverse domains and tasks; iii) how to mitigate the data domination phenomenon during the architecture search process. To address these challenges, we propose Automated Graph Foundation Model with Adaptive Architecture Customization (AutoGFM), providing a theoretical analysis to demonstrate the limitations of existing GNAS. Specifically, we first propose a disentangled contrastive graph encoder to learn invariant and variant patterns. Then, we design an invariant-guided architecture customization strategy to customize architectures for data from diverse domains and tasks. Finally, we propose a curriculum architecture customization mechanism to mitigate the phenomenon of particular data dominating the search process. Extensive experiments demonstrate that AutoGFM outperforms baselines, achieving state-of-the-art performance.

Section: Introduction
Graph foundation models (GFMs) (Liu et al., 2023b;Xu et al., 2024;Kong et al., 2024) aim to share graph knowledge across diverse graph domains and tasks. GNN-based GFMs (Liu et al., 2023a;Wang et al., 2024b) represent a promising direction, as they enable the transfer of shared knowledge across various domains and tasks, allowing a single graph neural network (GNN) to handle node-level, edge-level, and graph-level tasks across various domains. Specifically, GNN-based GFMs leverage large language models (LLMs) as enhancers, transform the textual features of graphs into unified representations, and unify graphrelated tasks through subgraph classification for GNNs.
However, data from different tasks and domains may require different graph neural architectures. For instance, the vanilla GCN (Kipf & Welling, 2017) outperforms GraphSAGE (Hamilton et al., 2017) in the citation network OGBN-arxiv, while failing to demonstrate satisfactory performance in OGBN-proteins (Hu et al., 2020). Since existing GNNbased GFMs rely on hand-designed and fixed GNN architectures, they inevitably fail to adapt to the specific architecture requirements for diverse domains and tasks.
In this paper, we explore the problem of graph neural architecture search (GNAS) for GNN-based graph foundation models, which suffers from the problem of architecture inconsistency, i.e., the optimal architecture for different tasks and domains varies. We further leverage a representative group of differentiable graph neural architecture search methods (Liu et al., 2018) as an example and provide theoretical analysis demonstrating their inability to effectively search for graph neural architectures for GFMs under architecture inconsistency, resulting in suboptimal architectures. We tackle this problem by discovering an invariant grapharchitecture relationship across domains and tasks, which imposes three challenges: i) how to capture invariant and variant patterns, which are entangled in graph data; ii) how to customize graph neural architectures based on the discovered patterns to adapt to data with diverse domains and tasks; iii) how to mitigate the phenomenon of data domination during the architecture search process.
To address these challenges, we propose a novel Automated Graph Foundation Model with Adaptive Architecture Customization (AutoGFM), which customizes graph neural architectures for graph data across diverse tasks and domains. The core idea is to train an architecture mapping function π, which maps G (graph data) → A (architecture), enabling the customization of architectures for each dataset to address architecture inconsistency, while simultaneously facilitating mutual knowledge sharing across diverse domains and tasks within a weight-sharing super-network. Specifically, we first propose a disentangled contrastive graph encoder to learn invariant and variant patterns from graph data. To achieve this, we design a subgraph-level discriminative contrastive learning that captures the invariant and variant patterns from diverse graph data. Second, we propose an invariant-guided architecture customization to tailor graph neural architectures for diverse data. We encourage invariant patterns to retain their ability to customize architectures despite the interference from variant patterns, aiming to eliminate the spurious effects brought by variant patterns. Finally, we propose a curriculum architecture customization mechanism to mitigate the phenomenon of some particular data dominating the search process. We design a curriculum constraint to promote the diversity of customized architectures across different datasets. Extensive experiments demonstrate that our AutoGFM model outperforms existing baselines, achieving state-of-the-art performance. The contributions of this paper are summarized as follows:
• We propose to explore the problem of graph neural architecture search for GNN-based graph foundation model, to the best of our knowledge, for the first time.
• We propose Automated Graph Foundation Model with Adaptive Architecture Customization (AutoGFM), analyzing the problem of architecture inconsistency for GFM and providing a theoretical analysis to demonstrate the limitations of existing mainstream differentiable GNAS methods under such conditions.
• We propose three novel modules to tackle the problem of architecture inconsistency, i) disentangled contrastive graph encoder, ii) invariant-guided architecture customization, and iii) curriculum architecture customization mechanism.
• We conduct extensive experiments on eight datasets to demonstrate the superiority of our method over stateof-the-art baselines.
this section cite: ['b84', 'b32', 'b31', 'b18', 'b23', 'b43']

Section: Problem Formulation
In this section, we introduce the fundamental concepts and notations used in this paper, including graph data definition, node of interest (NOI) graph, graph neural architecture search, and GNAS for GNN-based GFMs.
this section cite: []

Section: Graph Data Definition
Text-attributed Graphs (TAGs) A text-attributed graph (TAG) is a graph where each node and edge is associated with a text sentence (Liu et al., 2023a). We denote a TAG is denoted as G = (V, E, R), where V = v 1 , . . . , v |V| represents the set of nodes, E = e 1 , . . . , e |E| represents the set of edges, and R = r 1 , . . . , r |R| represents the set of relations.
this section cite: []

Section: Node of Interest (NOI) Subgraph
Given a graph G = (V, E, R). Following the previous work (Liu et al., 2023a;Wang et al., 2024b), we define the subgraph to unify graph tasks as node of interest subgraph (NOI-graph). An NOIgraph G h is defined as the subgraph around the NOI. De-
note S h (v) = {V h v , E h v , R h v }
as the h-hop ego-subgraph around v, consisting of h-hop neighbor nodes of v and all interconnecting edges. For node-level tasks on a node v, the NOI is the node itself, such that T = {v} and
G h (T ) = S h (v). For link-level tasks involving a node pair (v i , v j ), we define T = {v i , v j }, and the NOI-graph is G h ({v i , v j }) = S h (v i ) ∪ S h (v j ). For graph-level tasks, the NOI includes all nodes in the graph, making the NOI-graph G h (V) = (V, E, R). We define an NOI-graph G h (T ) as: G h (T ) = ∪ v∈T S h (v) = ∪ v∈T V h v , ∪ v∈T E h v , ∪ v∈T R h v .(1)
this section cite: []

Section: Graph Neural Architecture Search
Given a data D = (G, Y) for a graph neural architecture search (GNAS), we aim to search for a function F α,w : G → Y, with architecture parameters α ∈ A and learnable weights w ∈ W, where A is the architecture space and W is the weight space:
α * = arg min α∈A L(F α,w * (α) (G), Y),(2)
s.t. w * (α) = arg min w∈W(α) L(F α,w (G), Y),(3)
where L represents the loss of predictions made by the architecture F α,w (•) on the graph, and α * and w * denote the optimal architecture and weights for the given data D = (G, Y). Specifically, α typically represents the selection of GNN operations (e.g., GCN (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017), GAT (Velickovic et al., 2017), GIN (Xu et al., 2018), etc.), which are referred to as operation choices for brevity. GNAS addresses this as a bi-level optimization problem (Elsken et al., 2019).
this section cite: ['b31', 'b18', 'b70', 'b83', 'b6']

Section: GNAS for GNN-based GFMs
We define diverse data as D = {D 1 , D 2 , . . . , D N }, where D i = {G i , Y i } represents the i-th dataset with graph G i and label Y i . Following previous work for GNN-based GFMs (Liu et al., 2023a;Wang et al., 2024b), which leverage LLMs to unify the node feature space across different graphs and leverage subgraphs to unify graph tasks, enabling a single GNN to be applied to diverse data D across domains and tasks. Graph neural architecture search for GFM aims to search a graph neural architecture F α,w that achieves performance in diverse data D.
this section cite: []

Section: Preliminaries
In this section, we first introduce the problem of architecture inconsistency in GFM. Then we provide an invariant view of architecture customization and formulate the overall objective for our proposed method. According to our observation, the optimal architecture for graph data across different tasks and domains may exhibit architecture inconsistency, meaning that the optimal architectures vary for data with diverse tasks and domains. To validate this, we test various GNN architectures built upon a GNN-based GFM, GFT (Wang et al., 2024b), on datasets with different domains and tasks. We present the performance of each architecture on each dataset using a heatmap in Figure 1. As shown in Figure 1, datasets from different domains and tasks require distinct optimal architectures, highlighting the presence of architecture inconsistency. Based on this observation, we introduce the following assumption:
this section cite: []

Section: Architecture Inconsistency in GFM
Assumption 3.1. There exist two datasets D i , D j ∈ D, the optimal operation required D i is different from D j .
Assumption 3.1 assumes that the optimal architectures required by two different datasets may differ. Then we further provide theoretical analyses showing that architecture inconsistency leads to operations optimization conflicts in existing differentiated GNAS methods. We have the follow-ing proposition with proof in Appendix A.1.
Proposition 3.2. If there exist two datasets D i , D j ∈ D, the optimal operation for D i is different from D j , the operations will render optimization conflicts.
Assumption 3.1 serves as a prerequisite condition for Proposition 3.2. Proposition 3.2 demonstrates that when two datasets require different optimal architectures, current mainstream GNAS methods encounter optimization conflicts for GFM. For instance, as illustrated in Figure 1, the optimal architectures for PubMed and Wikics differ. When existing GNAS methods search simultaneously for an architecture optimal for both datasets, they fail to identify a single architecture that performs best for both and are forced to compromise.
To tackle architecture inconsistency, our key idea is to train a mapping function π: G → A, which customizes architectures for each data to prevent architecture inconsistency, while simultaneously facilitating knowledge sharing across domains and tasks via the weight-sharing supernetwork.
this section cite: []

Section: Invariant View of Architecture Customization
We customize graph neural architectures for graph data across diverse tasks and domains from an invariant perspective. Unlike conventional invariant inference approaches that aim to discover invariant relationships between data and labels (Wu et al., 2022b;Li et al., 2022a;Wu et al., 2022a), our goal is to identify invariant relationships between the graph data and the corresponding architecture, addressing the issue of architecture inconsistency by tailoring graph neural architectures for each data individually.
We formalize the four key variables: input graph data G, architecture A, invariant pattern Z I , and variant pattern Z V . We divide the architecture mapping function π into two components: encoder θ : G → Z I and predictor ψ : Z I → A. We make the following assumptions:
Assumption 3.3. (1) Z I = G \ Z V . There are two disjoint parts in the graph data G: invariant part Z I and variant part Z V . (2) Z V ̸ ⊥ A. The variant part Z V is correlated with the architecture A. (3) A ⊥ Z V | Z I and A = ψ(Z I ). Z I shields A from the influence of Z V . Assumption 3.3 defines what constitutes an invariant pattern for architecture prediction: i) Condition 1 indicates that the data contains two types of patterns: an invariant pattern Z I , which reliably predicts the architecture, and a variant pattern Z V , which cannot stably predict the architecture; ii) Condition 2 highlights that the variant pattern Z V is not independent of the architecture A; iii) Condition 3 states that, given the invariant pattern Z I , the architecture A is independent of the variant pattern Z V , and Z I is sufficient for predicting A. Curriculum Architecture Customization Mechanism . . . . . . . . . . . . . . . . . . minimize the distance different data maxmize the distance 𝜃 𝜓 E Different NOI-graph z i . I z i , V z' j , I z' j , V z i , I z i , V ... ... Progressive search process Encourages diversity in early . . . . . . . . . . . . . . . . . . Customized Architecture
𝑝 ! z ",$ | G " 𝑝 ! 𝑠 z ",$ | G " , z ",$ Figure 2.
The framework of Automated Graph Foundation Model with Adaptive Architecture Customization(AutoGFM). The model consists of three modules: i) Disentangled contrastive graph encoder that discovers invariant and variant patterns from graph data , ii) invariant-guided architecture customization enabling customization of graph neural architectures based on the discovered invariant and variant patterns to adapt to data with diverse domains and tasks, and iii) Curriculum architecture customization mechanism to mitigate the influence of any single data dominating the search process.
this section cite: []

Section: Overall Objective
To satisfy the constraints outlined in Assumption 3.3, we formulate a learning objective that adheres to the specified conditions. Specifically, we minimize the mutual information between Z I and Z V to ensure that the two parts remain disjoint. Simultaneously, we maximize the mutual information between Z I and A , ensuring that Z I is sufficient for predicting the architecture A. Furthermore, we minimize the mutual information between A and Z V , conditioned on Z I , to guarantee that Z I shields A from the influence of Z V . The resulting overall learning objective is defined as follows:
max θ,ψ I(Z I , A) -λI(Z I , Z V ) -βI(A, Z V | Z I ), (4
)
where I denotes the mutual information function, θ represents the encoder that extracts Z I and Z V from G, ψ represents the predictor that maps Z I to A, and λ and β are hyperparameters controlling the trade-off.
this section cite: []

Section: The Proposed Method: AutoGFM
In this section, we introduce an Automated Graph Foundation Model with Adaptive Architecture Customization (AutoGFM) to search for graph neural architectures for each graph data with diverse tasks and domains individually. We first introduce two modules: a disentangled contrastive graph encoder and invariant-guided architecture customization. Besides, we introduce our optimization objective with a curriculum architecture customization mechanism. The overall framework of AutoGFM is illustrated in Figure 2.
this section cite: []

Section: Disentangled Contrastive Graph Encoder
In this section, we focus on learning disentangled representations to capture two distinct aspects of graph data. Specifically, we aim to learn two architecture-aware disentangled representations, Z I and Z V , we temporarily treat them jointly and denote the two-channel representations as Z k (k = 1, 2) in this section. The main insight of our proposed method is intuitively based on the following observations: (1) Data from the same sources (i.e., the same domain and task) require similar architectures, so they share a similar Z I that reflects the architectural requirements. Conversely, graphs from different data sources will have distinct Z I . (2) To satisfy the Assumption 3.3, the mutual information between Z I and Z V should be minimized.
Disentangled NOI-graph Encoder Initially, we adopt GNNs with individual parameters to learn two-channel graph representations of NOI-graphs.
H (l) k = GNN k H (l-1) k , A , k = 1, 2,(5)
where
H (l)
k is the k-th channel of the node representation at the l-th layer, A is the adjacency matrix of the graph. We employ two distinct Readout functions (i.e., pooling functions) and MLPs to derive a NOI-graph-level representation for each channel:
z k = MLP k h k ,(6)
h k = Readout k H (L) k , k = 1, 2.(7)
NOI-graph Disentangled Contrastive Learning Inspired by self-supervised contrastive learning that captures discriminative features by pulling similar samples together and pushing dissimilar samples apart in latent space (Jaiswal et al., 2020;Le-Khac et al., 2020;You et al., 2020), we propose an NOI-graph-level contrastive learning method to encourage disentangled representations to reflect the architectural requirements of the graph data. Initially, we encourage the representations Z I and Z V to be disentangled:
p θ (z i,k | G i ) = exp ϕ (z i,k , p k ) 2 j=1 exp ϕ (z i,k , p j ) ,(8)
where ϕ is a similarity function, G i is a NOI-graph from i-th graph, z i,k is the k-th chunk of the representation of a NOI-graph from i-th graph, and p k is the propotype of the k-th chunk of the representation. Then, we propose a NOI-graph-level instance discriminative task to encourage the representations Z I to capture different architectural requirements of different data. The task is defined as:
p θ (s(z i,k ) | G i , z i,k ) = exp ϕ z i,k , z ′ i,k N j=1 exp ϕ z i,k , z ′ j,k ,(9)
where s(z i,k ) represents a unique surrogate label assigned to z i,k , and z ′ i,k is sampled from the same graph data G i as z i,k . Then we learn the model parameters θ by calculating the loss function as:
L dis = i -log E p θ (z i,k |Gi) p θ (s(z i,k ) | G i , z i,k ) . (10)
In this way, we encourage the representations Z I to capture the architectural requirements of the graph data, while ensuring that the representations Z I and Z V are disentangled.
this section cite: ['b29', 'b33', 'b87']

Section: Invariant-guided Architecture Customization
To customize graph neural architectures based on the discovered patterns and enable adaptation to data from diverse domains and tasks, we propose an invariant-guided architecture customization approach. Specifically, we first establish a weight-sharing super-network with a set of prototypes, then utilize an invariant predictor, ψ I , and an auxiliary predictor, ψ E , to guide the customization process, ensuring the minimization of
I(A, Z V | Z I ) in Assumption 3.3.
Weight-sharing Super-network To facilitate differentiable optimization, we employ continuous parameterization and a weight-sharing mechanism (Liu et al., 2018) to implement the mixed operations. The super-network layer with |O| mixed operations is defined as:
H (l) ← |O| i=1 α l,i GNN (l-1) i (H (l-1) , A),(11)
where A is the adjacency matrix of the graph, H (l) represents the node representations at the l-th layer, GNN (l-1) i denotes the mixed GNN operations, |O| is the number of GNN operation choices, and α l,i indicates the probability of selecting the i-th operation for the l-th layer. Different from previous super-networks (Liu et al., 2018) that use learnable parameters α, we employ a set of prototypes to guide the routing between data and the operations.
Architecture Predictor Given a graph representation z ∈ Z , we design an invariant mapping predictor, ψ I : z → {α l,i }. The probability α l,i of selecting the i-th operation for the l-th layer is calculated as follows:
α l,i = exp(α l,i ) |O| j=1 exp( αl,j ) , αl,i = z • p l,i ∥p l,i ∥ 2 , (12
)
where p l,i is a learnable prototype of the i-th operation for the l-th layer, and z is the graph representation. Following (Qin et al., 2022a), we adopt the l 2 -normalization on p to ensure numerical stability and fair competition among different operations. We utilize the learnable prototypes p as the parameters of the predictor ψ I to map architecture, i.e., if the graph representation z is similar to the prototype p l,i , the operation i will be selected for the l-th layer.
this section cite: ['b43', 'b43']

Section: Invariant-guided Customization The objective of minimizing the conditional mutual information
I(A, Z V | Z I )
in Equation ( 4) is not tractable, as the mutual information of high-dimensional vectors is difficult to estimate. Therefore, we utilize an equivalent transformation in Proposition 4.1 to achieve it, with a detailed proof provided in Appendix A.2.
A I = ψ I (Z V ), A E = ψ E (Z I , Z V ).(13)
We guide P (A | Z I , Z V ) = P (A | Z I ) by minimizing the difference between the two predictions A I , A E . Besides, to boost the architecture predictor to fit more data with different variant patterns, we fuse Z I with Z V from other data to predict A E . We define the loss as:
L inv = ∥D∥ i ∥D∥ j ∥A I,i -A E,(i,j) ∥,(14)
s.t. A I,i = ψ I (z I,j ), A E,i,j = ψ E (z I,i , z V,j ), (15
)
where A I,i is the predicted architecture for the i-th graph based on the invariant patterns z I , A E,i,j are the predicted architectures based on the patterns fused from z I,i and z V,j .
this section cite: []

Section: Optimization with Curriculum Customization Mechanism
We calculate the task loss of GFM using only the architecture predicted by the invariant predictor ψ I :
L task = ℓ(F ψ(Z I ) (G), y).(16)
L task aim to maximize I(Z I , A) in Equation ( 4). Notably, the computation method of L task depends on the GFM for which we aim to search for architectures. e.g., GFT (Wang et al., 2024b) utilizes Computation Tree Reconstruction to calculate loss during the pretraining stage.
GFMs need to be simultaneously optimized using multiple datasets with diverse domains and tasks. However, different data have different influences on architectures (Zhou et al., 2022d), mainly on the learnable weights of operations in our study, e.g., some datasets are easier to fit with certain operations but more challenging with others. As a result, operations that fit well in the early stages of training are more likely to be selected, causing other operations to be overlooked.
To mitigate the dominance of data in the search process, we design a curriculum architecture customization constraint that encourages diversity in the customized architectures during the early stages of training. We first calculate the average α for each operation in the l-th layer as:
α l = |D| i=1 α l,1 (z i ), |D| i=1 α l,2 (z i ), . . . , |D| i=1 α l,J (z i ) |D| ,(17)
where α l is the average α for each operation in the l-th layer, α l,j (z i ) is the probability of selecting the j-th operation for the l-th layer predicted by the invariant predictor ψ I for the i-th graph, and J is the number of operations. We define the curriculum architecture customization loss as follows:
L cur = γ L l=1 CV(α l ),(18)
where CV(α l ) is the coefficient of variation of the average α for each operation in the l-th layer across different data and
Algorithm 1 Training pipeline for AutoGFM Input: data D = {G 1 , G 2 , . . . , G N }, hyperparameters λ, β. for t = 1, ..., T do Sample NOI-graphs G i from G i . Extract Z I and Z V from NOI-graphs G i . Calculate L dis using Equation (10). Obtain architectures A I and A E predicted by ψ I and ψ E in Equation ( 15). Calculate L inv using Equation ( 14). Calculate L cur and L task using Equation ( 18) and Equation ( 16), respectively. Update θ, ψ I , ψ E by minimizing Equation ( 19). end for γ is controlled by a pacing function: γ = 1 -t te , where t is the current training step and t e is the step to stop the curriculum customization mechanism. This constraint encourages diversity in early architecture customization, thereby mitigating the influence of any single data dominating the search process. The final training objective is:
min θ,ψ I ,ψ E L task + λ L dis + β L inv + L cur , (19
) max I(ZI , A) min I(ZI , ZV ) min I(A, ZV | ZI )
where L task aims to exploit invariant patterns to customize architectures, L dis encourages the disentanglement of the invariant and variant patterns, L inv discovers the invariant patterns and variant patterns, and L cur mitigates the influence of any single data dominating the search process. The overall algorithm is summarized in Algorithm 1.
During the inference stage, given an input graph, we first utilize the disentangled contrastive graph encoder to obtain its invariant pattern representation, denoted as Z I . Then, Z I is fed into the invariant predictor ψ I to generate a customized architecture. This architecture is then used as the GNN component within the GFM to perform prediction.
this section cite: []

Section: Experiments
In this section, we conduct experiments on real-world datasets with diverse domains and tasks to show the effectiveness of the proposed AutoGFM for GFM.
this section cite: []

Section: Experimental Setup
Datasets We employ datasets with diverse domains and tasks. For node-level tasks, we utilize citation networks (Cora, Pubmed, and Arxiv) and the web link network (Wi-kiCS). For edge-level tasks, we utilize Knowledge Graphs (WN18RR, FB15K237). For graph-level tasks, we utilize molecular datasets (HIV, PCBA, and ChEMBL). Following Baselines We compare our proposed AutoGFM with the five categories of baselines: (1) Vanilla GNNs: GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2017), GIN (Xu et al., 2018); (2) Self-supervised methods: BGRL (Thakoor et al., 2021), GraphMAE (Hou et al., 2022), GIANT (Chien et al., 2022). (3) GFMs: OFA (Liu et al., 2023a) and GFT (Wang et al., 2024b). (4) Manually designed GNNs: GCN (Kipf & Welling, 2017), GAT (Velickovic et al., 2017), GIN (Xu et al., 2018), GraphSAGE (Hamilton et al., 2017), and GraphConv (Morris et al., 2019). (5) GNAS methods: DARTS (Liu et al., 2018), GraphNAS (Gao et al., 2021), GASSO (Qin et al., 2021b), Graces (Qin et al., 2022a).
For manually designed GNNs, GNAS baselines, and Au-toGFM , we utilize GFT (Wang et al., 2024b) as the base model to ensure a fair comparison. Additionally, we adopt the same search space (operations in manually designed GNNs baselines and super-network layers is 2) for both GNAS baselines and AutoGFM. We replicate each experiment ten times and report the average results. Further details about experimental setups are provided in Appendix D.
this section cite: ['b31', 'b70', 'b83', 'b68', 'b21', 'b4', 'b31', 'b70', 'b83', 'b18', 'b50', 'b43', 'b10']

Section: Main Results
Pre-training and Fine-tuning From the results in Table 1, we observe the following: (1) None of manually designed GNN performs well across all datasets, indicating that fixed architectures struggle to generalize across diverse domains and tasks. (2) Existing GNAS methods fail to discover better architectures for each dataset, highlighting their limitations in adapting to diverse domains and tasks. (3) AutoGFM outperforms all baselines across datasets, demonstrating its effectiveness in customizing architectures for different domains and tasks.
Few-shot Learning Few-shot learning is a challenging task that requires models to generalize well with limited labeled samples. We randomly sample a few labeled samples per way from the training set for fine-tuning. From the results in Table 2, despite the limited labeled samples, AutoGFM achieves the best performance across all datasets, demonstrating the fast adaptability of its architectures. We provide more experimental results in Appendix B.3.
this section cite: []

Section: Ablation Study
To verify the effectiveness of the key modules in our method, we compare different ablated versions on five datasets: i) w/o D removes and replaces the disentangled contrastive graph encoder with standard GNNs; ii) w/o I removes the invariant-guided architecture customization module by removing the L inv ; iii) w/o C removes the curriculum architecture customization mechanism. The results are shown in Figure 3. We observe that the full model achieves the best performance across all datasets, demonstrating the ef-  all computational complexity of our method is given by:
O(|E|d e + |V |d 2 e + |O| 2 d e + |O|(|E|d a + |V |d 2 a )).
this section cite: []

Section: Related Work
In this section, we review the related work on GNNbased Graph Foundation Models, graph neural architecture search, graph invariant presentation learning, and graph self-supervised learning. We provide more related work in Appendix C.
this section cite: []

Section: GNN-based Graph Foundation Models
GNN-based GFMs, which leverage LLMs as enhancers, are a promising direction for GFM (He & Hooi, 2024;Mao et al., 2024a;Zhao et al., 2024b;Xia & Huang, 2024;Huang et al., 2024c). Specifically, it involves using LLMs to transform the textual features of graphs into unified representations and unify graph-related tasks through subgraph classification for GNNs (Sun et al., 2023;Zhao et al., 2024c;Fan et al., 2024;Ren et al., 2024;Pan et al., 2024b;Mao et al., 2024b). This enables the transfer of shared knowledge across various domains. Two key challenges in developing GNN-based Graph Foundation Models (GFMs) are the unification of diverse tasks and domain spaces (Yu et al., 2024;Li et al., 2024b;Zhao et al., 2024a;Xia et al., 2024;Zhu et al., 2024b;Guo et al., 2023). For instance, OFA (Liu et al., 2023a) employs an LLM to unify input features from diverse datasets, converting multiple graph classification tasks into a unified binary classification format. GFT (Wang et al., 2024b) extends OFA by incorporating computation trees to discover transferable patterns across graphs. Nevertheless, these models face challenges in their reliance on manually designed architectures, which can constrain their performance on diverse domains and tasks. More related work about GFMs is included in Appendix C.
this section cite: ['b20', 'b64', 'b7', 'b62', 'b89', 'b17']

Section: Graph Neural Architecture Search
Neural architecture search (NAS) has gained growing attention for its capability to automate the design of neural architectures tailored to specific tasks (Pham et al., 2018;Qin et al., 2021a;Wang et al., 2025). In particular, graph neural architecture search (GNAS) methods address the distinct challenge of modeling the intricate relationships between architectures and complex graph structures (Gao et al., 2021;Qin et al., 2022b;Guan et al., 2022;Zhang et al., 2023e;Xie et al., 2023). These methods can be broadly classified into three categories: reinforcement-learning-based approaches (Zhou et al., 2022b;Gao et al., 2022;2023); evolutionary-based strategies (Nunes & Pappa, 2020;Li & King, 2020;Shi et al., 2022;Zhang et al., 2022a;b); and differentiable methods (Ding et al., 2021;Zheng et al., 2023;Huan et al., 2021;Zhang et al., 2023b;d;Qin et al., 2023;Yao et al., 2024;Ge et al., 2025), which enable continuous optimization of architectures within a differentiable search space. However, existing GNAS methods are limited in their ability to search for architectures for GNN-based GFMs.
this section cite: ['b55', 'b73', 'b10', 'b16', 'b82', 'b11', 'b51', 'b41', 'b63', 'b5', 'b105', 'b24', 'b60', 'b85', 'b13']

Section: Graph Invariant Representation Learning
Graph invariant presentation learning has emerged as a powerful approach for graph representation learning, focusing on capturing the stable relationships between graph data and tasks. Recent works have explored various applications of graph invariant learning in out-of-distribution generalization (Ma et al., 2019;Wu et al., 2022b;Li et al., 2022b;c;Zhang et al., 2022c;2023c;2024b;Li et al., 2024a). For instance, DIR (Wu et al., 2022b) discovers causal rationales that remain invariant across different distributions while filtering out spurious patterns that are unstable. DIDA (Zhang et al., 2022c) leverages invariant structures and features with stable predictive performance across distribution shifts. However, these methods focus on capturing stable relationships for accurate label prediction. We apply this concept to architecture search, aiming to define invariant patterns that support stable architecture prediction, and design our method based on this concept.
this section cite: ['b47']

Section: Graph Self-supervised Learning
Graph self-supervised learning (SSL) has attracted significant attention in recent years, with numerous methods proposed to learn effective representations from graph data without relying on labeled information. These methods can be broadly classified into two categories: contrastive learning and generative learning. Contrastive learning approaches (You et al., 2020;Hassani & Khasahmadi, 2020;Li et al., 2021;Zhang et al., 2024a;Li et al., 2022d) aim to maximize the agreement between positive pairs of graph samples while minimizing it between negative pairs. In contrast, generative learning approaches (Tan et al., 2023;Xia et al., 2023;Hou et al., 2023) learn representations by reconstructing graph structures or attributes from partially observed data. These SSL techniques have demonstrated strong performance across a variety of graph-related tasks, including node classification, link prediction, and graph classification. In our work, we adopt a contrastive learning strategy to extract distinct patterns from diverse datasets, enabling the model to capture richer information and better identify architecture-specific requirements.
this section cite: ['b87', 'b19', 'b35', 'b65', 'b80', 'b22']

Section: Conclusion
Existing graph neural architecture search methods fail to search for architectures for GNN-based GFMs. In this paper, we analyze the problem of architecture inconsistency, demonstrate that existing GNAS methods cannot effectively search for architectures for GNN-based GFMs, and tackle it by discovering an invariant graph-architecture relationship. We propose a novel Automated Graph Foundation Model with Adaptive Architecture Customization (AutoGFM) to search for graph neural architectures for each graph data with diverse tasks and domains individually. We introduce a disentangled contrastive graph encoder to discover invariant and variant patterns from graph data and an invariant-guided architecture customization module to customize graph neural architectures based on the discovered patterns. We also propose a curriculum architecture customization mechanism to mitigate the phenomenon of some particular data dominating the search process. Experimental results demonstrate that AutoGFM outperforms existing methods. One limitation of our work is that we mainly focus on graph neural architecture search on the GNN-based GFMs, and we leave the exploration of other types of GFMs for future work.
this section cite: []

Section: References
Ref_id:b0 Title: Curriculum learning Year: (2009)
Ref_id:b1 Title: Curriculum-listener: Consistency-and complementarity-aware audio-enhanced temporal sentence grounding Year: (2023)
Ref_id:b2 Title: Large language and graph assistant Year: (2024)
Ref_id:b3 Title: Curriculum meta-learning for next poi recommendation Year: (2021)
Ref_id:b4 Title: Node feature extraction by self-supervised multi-scale neighborhood prediction Year: (2022)
Ref_id:b5 Title: Diffmg: Differentiable meta graph search for heterogeneous graph neural networks Year: (2021)
Ref_id:b6 Title: Neural architecture search: A survey Year: (2019)
Ref_id:b7 Title: Graph machine learning in the era of large language models (llms) Year: (2024)
Ref_id:b8 Title: Improving graph contrastive learning for text-attributed graphs with large language models Year: (2024)
Ref_id:b9 Title: Talk like a graph: Encoding graphs for large language models Year: (2023)
Ref_id:b10 Title: Graph neural architecture search Year: (2021)
Ref_id:b11 Title: Graphnas++: Distributed architecture search for graph neural networks Year: (2022)
Ref_id:b12 Title: Hgnas++: efficient architecture search for heterogeneous graph neural networks Year: (2023)
Ref_id:b13 Title: Behavior importanceaware graph neural architecture search for cross-domain recommendation Year: (2025)
Ref_id:b14 Title: Multi-modal curriculum learning over graphs Year: (2019)
Ref_id:b15 Title: Why curriculum learning & self-paced learning work in big/noisy data: A theoretical perspective Year: (2015)
Ref_id:b16 Title: Large-scale graph neural architecture search Year: (2022)
Ref_id:b17 Title: A data-centric framework to endow graph neural networks with out-of-distribution detection ability Year: (2023)
Ref_id:b18 Title: Inductive representation learning on large graphs. Advances in neural information processing systems Year: (2017)
Ref_id:b19 Title: Contrastive multiview representation learning on graphs Year: (2020)
Ref_id:b20 Title: Unigraph: Learning a cross-domain graph foundation model from natural language Year: (2024)
Ref_id:b21 Title: Graphmae: Self-supervised masked graph autoencoders Year: (2022)
Ref_id:b22 Title: Graphmae2: A decoding-enhanced masked self-supervised graph learner Year: (2023)
Ref_id:b23 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b24 Title: Search to aggregate neighborhood for graph neural network Year: (2021)
Ref_id:b25 Title: Neighbor does matter: Curriculum global positive-negative sampling for vision-language pretraining Year: (2024)
Ref_id:b26 Title: Large language models for graphs: Progresses and directions Year: (2024)
Ref_id:b27 Title: Can llms effectively leverage graph structural information: when and why Year: (2023)
Ref_id:b28 Title: Prodigy: Enabling in-context learning over graphs Year: (2024)
Ref_id:b29 Title: A survey on contrastive self-supervised learning Year: (2020)
Ref_id:b30 Title: Language model pretraining on text-rich networks Year: (2023)
Ref_id:b31 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b32 Title: A generative one-for-all model for joint graph language modeling Year: (2024)
Ref_id:b33 Title: Contrastive representation learning: A framework and review Year: (2020)
Ref_id:b34 Title: Invariant information bottleneck for domain generalization Year: (2022)
Ref_id:b35 Title: Disentangled contrastive learning on graphs Year: (2021)
Ref_id:b36 Title: Ood-gnn: Outof-distribution generalized graph neural network Year: (2022)
Ref_id:b37 Title: Learning invariant graph representations for out-of-distribution generalization Year: (2022)
Ref_id:b38 Title: Curriculum graph machine learning: A survey Year: (2023)
Ref_id:b39 Title: Disentangled graph self-supervised learning for outof-distribution generalization Year: (2024)
Ref_id:b40 Title: Let invariant rationale discovery inspire graph contrastive learning Year: (2022)
Ref_id:b41 Title: Autograph: Automated graph neural network Year: (2020)
Ref_id:b42 Title: Investigating cross-dataset zero-shot transferability in graphs Year: (2024)
Ref_id:b43 Title: Darts: Differentiable architecture search Year: (2018)
Ref_id:b44 Title: One for all: Towards training one graph model for all classification tasks Year: (2023)
Ref_id:b45 Title: Towards graph foundation models: A survey and beyond Year: (2023)
Ref_id:b46 Title: Graphprompt: Unifying pre-training and downstream tasks for graph neural networks Year: (2023)
Ref_id:b47 Title: Disentangled graph convolutional networks Year: (2019)
Ref_id:b48 Title: Graph foundation models Year: (2024)
Ref_id:b49 Title: Advancing graph representation learning with large language models: A comprehensive survey of techniques Year: (2024)
Ref_id:b50 Title: Weisfeiler and leman go neural: Higher-order graph neural networks Year: (2019)
Ref_id:b51 Title: Neural architecture search in graph neural networks Year: (2020)
Ref_id:b52 Title: Distilling large language models for text-attributed graph learning Year: (2024)
Ref_id:b53 Title: Integrating graphs with large language models: Methods and prospects Year: (2024)
Ref_id:b54 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b55 Title: Efficient neural architecture search via parameters sharing Year: (2018)
Ref_id:b56 Title: Graph q network for neural architecture search Year: ()
Ref_id:b57 Title: Graph differentiable architecture search with structure learning Year: (2021)
Ref_id:b58 Title: Graph neural architecture search under distribution shifts Year: (2022)
Ref_id:b59 Title: Nasbench-graph: Benchmarking graph neural architecture search Year: (2022)
Ref_id:b60 Title: Multitask graph neural architecture search with task-aware collaboration and curriculum Year: (2023)
Ref_id:b61 Title: Sentence-bert: Sentence embeddings using siamese bert-networks Year: (2019)
Ref_id:b62 Title: A survey of large language models for graphs Year: (2024)
Ref_id:b63 Title: Genetic-gnn: Evolutionary architecture search for graph neural networks. Knowledge-based systems Year: (2022)
Ref_id:b64 Title: All in one: Multi-task prompting for graph neural networks Year: (2023)
Ref_id:b65 Title: S2gae: Self-supervised graph autoencoders are generalizable learners with graph masking Year: (2023)
Ref_id:b66 Title: Graphgpt: Graph instruction tuning for large language models Year: (2024)
Ref_id:b67 Title: Graphgpt: Graph instruction tuning for large language models Year: (2024)
Ref_id:b68 Title: Largescale representation learning on graphs via bootstrapping Year: (2021)
Ref_id:b69 Title: Graph neural prompting with large language models Year: (2024)
Ref_id:b70 Title: Graph attention networks Year: (2017)
Ref_id:b71 Title: Curriculum pre-training heterogeneous subgraph transformer for top-n recommendation Year: (2023)
Ref_id:b72 Title: Can language models solve graph problems in natural language? Year: (2024)
Ref_id:b73 Title: Modular machine learning: An indispensable path towards new-generation large language models Year: (2025)
Ref_id:b74 Title: Graph foundation model with transferable tree vocabulary Year: (2024)
Ref_id:b75 Title: Clnode: Curriculum learning for node classification Year: (2023)
Ref_id:b76 Title: Handling distribution shifts on graphs: An invariance perspective Year: (2022)
Ref_id:b77 Title: Mitigating label noise on graph via topological sample selection Year: (2024)
Ref_id:b78 Title: Discovering invariant rationales for graph neural networks Year: (2022)
Ref_id:b79 Title: Graph foundation model in the Year: (2024)
Ref_id:b80 Title: Automated self-supervised learning for recommendation Year: (2023)
Ref_id:b81 Title: Towards open graph foundation models Year: (2024)
Ref_id:b82 Title: Adversarially robust neural architecture search for graph neural networks Year: (2023)
Ref_id:b83 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b84 Title: A comprehensive benchmark for graph foundation model Year: (2024)
Ref_id:b85 Title: Data-augmented curriculum graph neural architecture search under distribution shifts Year: (2024)
Ref_id:b86 Title: Natural language is all a graph needs Year: ()
Ref_id:b87 Title: Graph contrastive learning with augmentations Year: (2020)
Ref_id:b88 Title: Graphagent: Exploiting large language models for interpretable learning on text-attributed graphs Year: ()
Ref_id:b89 Title: Multigprompt for multi-task pre-training and prompting on graphs Year: (2024)
Ref_id:b90 Title: Motifdriven contrastive learning of graph representations Year: (2024)
Ref_id:b91 Title: Deep and flexible graph neural architecture search Year: (2022)
Ref_id:b92 Title: Pasca: A graph neural architecture search system under the scalable paradigm Year: (2022)
Ref_id:b93 Title: Dynamic graph neural networks under spatio-temporal distribution shift. Advances in neural information processing systems Year: (2022)
Ref_id:b94 Title: Learning to solve travelling salesman problem with hardness-adaptive curriculum Year: (2022)
Ref_id:b95 Title: Relational curriculum learning for graph neural networks Year: (2023)
Ref_id:b96 Title: Autogt: Automated graph transformer architecture search Year: (2023)
Ref_id:b97 Title: Spectral invariant learning for dynamic graphs under distribution shifts Year: (2023)
Ref_id:b98 Title: Unsupervised graph neural architecture search with disentangled self-supervision Year: (2023)
Ref_id:b99 Title: Dynamic heterogeneous graph attention neural architecture search Year: (2023)
Ref_id:b100 Title: Disentangled dynamic graph attention network for out-ofdistribution sequential recommendation Year: (2024)
Ref_id:b101 Title: All in one and one for all: A simple yet effective method towards cross-domain graph pretraining Year: (2024)
Ref_id:b102 Title: Graph reasoning in text space Year: (2023)
Ref_id:b103 Title: A foundation model for node classification on any graph Year: (2024)
Ref_id:b104 Title: A survey on self-supervised pre-training of graph foundation models: A knowledge-based perspective Year: (2024)
Ref_id:b105 Title: Auto-heg: Automated graph neural network on heterophilic graphs Year: (2023)
Ref_id:b106 Title: Deriving curriculum for pre-training gnns Year: (2022)
Ref_id:b107 Title: Autognn: Neural architecture search of graph neural networks Year: (2022)
Ref_id:b108 Title: Curml: A curriculum machine learning library Year: (2022)
Ref_id:b109 Title: Curriculum-nas: Curriculum weight-sharing neural architecture search Year: (2022)
Ref_id:b110 Title: Intraand inter-modal curriculum for multimodal learning Year: (2023)
Ref_id:b111 Title: Curbench: curriculum learning benchmark Year: (2024)
Ref_id:b112 Title: Efficient tuning and inference for large language models on textual graphs Year: (2024)
Ref_id:b113 Title: Adding conditional control to universal graph pre-trained models for graph domain transfer learning Year: (2024)
