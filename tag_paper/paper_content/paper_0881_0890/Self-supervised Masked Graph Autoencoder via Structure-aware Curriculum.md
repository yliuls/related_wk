Title: Self-supervised Masked Graph Autoencoder via Structure-aware Curriculum
Abstract: Self-supervised learning (SSL) on graphstructured data has attracted considerable attention recently. Masked graph autoencoder, as one promising generative graph SSL approach that aims to recover masked parts of the input graph data, has shown great success in various downstream graph tasks. However, existing masked graph autoencoders fail to consider the degree of difficulty of recovering the masked edges that often have different impacts on the model performance, resulting in suboptimal node representations. To tackle this challenge, in this paper, we propose a novel curriculum based self-supervised masked graph autoencoder that is able to capture and leverage the underlying degree of difficulty of data dependencies hidden in edges, and design better mask-reconstruction pretext tasks for learning informative node representations. Specifically, we first design a difficulty measurer to identify the underlying structural degree of difficulty of edges during the masking step. Then, we adopt a self-paced scheduler to determine the order of masking edges, which encourages the graph encoder to learn from easy to difficult parts. Finally, the masked edges are gradually incorporated into the reconstruction pretext task, leading to high-quality node representations. Experiments on several real-world node classification and link prediction datasets demonstrate the superiority of our proposed method over state-of-the-art graph self-supervised learning baselines. This work is the first study of curriculum strategy for masked graph autoencoders, to the best of our knowledge.

Section: Introduction
Graph-structured data is ubiquitous across various domains, including social networks, citation networks, and e-commerce systems. Graph neural networks (GNNs) have demonstrated significant success in learning meaningful representations from such data, particularly in supervised (Xu et al., 2019) and semi-supervised (Kipf & Welling, 2017;Hamilton et al., 2017) learning settings, where task-specific labels are available to guide the learning process. However, acquiring a large amount of high-quality annotations is often expensive and impractical in real-world applications.
Self-supervised learning (SSL), an unsupervised paradigm widely adopted in computer vision and natural language processing (Chen et al., 2020;He et al., 2020), has recently gained significant attention in the field of graph learning. SSL enables models to learn informative representations by solving carefully designed pretext tasks without requiring labeled data. Existing graph SSL methods can be broadly categorized into contrastive and generative approaches. Contrastive methods, such as DGI (Veličković et al., 2019), MVGRL (Hassani & Khasahmadi, 2020), and BGRL (Thakoor et al., 2022), predominate the field by leveraging instance discrimination as the primary pretext task. Although augmentation-free variants like AFGRL (Wang et al., 2022a) and IGCL (Li et al., 2023a) present promising alternatives, many classical contrastive methods still depend heavily on heuristic graph augmentations. Their performance can degrade when the selected augmentations are misaligned with the objectives of downstream tasks (Zhang et al., 2021a). In contrast, generative SSL methods address this limitation more naturally by reconstructing missing components of the input graph. Representative models such as GPT-GNN (Hu et al., 2020b), GraphMAE (Hou et al., 2022), and S2GAE (Tan et al., 2023) demonstrate strong performance while avoiding the need for manually crafted augmentations.
Despite the notable progress of generative graph SSL methods, existing approaches typically ignore the varying difficulty levels of pretext tasks during training and treat all training samples uniformly, resulting in suboptimal performance. Intuitively, pretext tasks should be designed to start with easier data samples and gradually progress to more difficult ones. Introducing overly challenging tasks at the early stages of training can overwhelm the GNN encoder, which is typically initialized with random parameters and lacks the capacity to handle complex reconstructions. Conversely, prolonged training on overly simplistic tasks yields diminishing benefits and fails to further improve representation quality. The design of tailored easy-to-hard pretext tasks for enhancing graph representation learning remains an underexplored direction, which poses the following challenges.
• It is technically difficult to design tailored reconstruction tasks to encourage the GNNs to capture informative patterns of the input graph into representations.
• It is challenging to derive a proper principle to quantify the difficulty of reconstruction samples for training the GNNs.
• It is also non-trivial to design a feasible scheduling strategy to gradually exploit data samples for reconstruction by explicitly considering the current training status of GNNs.
To tackle these challenges, we propose Curriculum Masked Graph AutoEncoder (Cur-MGAE), a novel framework designed to capture and leverage the inherent difficulty of structural dependencies in graph edges. Cur-MGAE aims to construct more effective mask-reconstruction pretext tasks by integrating a curriculum learning paradigm into generative graph self-supervised learning. Specifically, our method enables GNNs to learn informative representations by gradually incorporating training samples in a tailored easy-to-hard order. We first design a structure-aware edge reconstruction task, where the goal is to recover intentionally masked edges based on the remaining unmasked graph structure. This pretext task encourages the GNN encoder to extract meaningful patterns from the graph. To quantify task difficulty, we introduce a self-supervised mechanism that identifies the easiest K edges the encoder is most confident in reconstructing. This allows for a principled estimation of reconstruction difficulty across edge samples. Furthermore, we develop a self-paced learning strategy that dynamically selects edges to be used in training, progressively increasing the task difficulty in alignment with the encoder's evolving learning capacity. The processes of edge selection and structural reconstruction are integrated into a unified training framework. Ultimately, the GNN encoder is trained using a meaningful curriculum that aligns edge difficulty with model capacity, yielding more powerful node representations and improved performance on downstream tasks.
We theoretically analyze the convergence guarantee of this tailored training paradigm by demonstrating its ability to avoid saddle points and achieve second-order convergence.
Extensive experiments on various real-world node classification and link prediction benchmarks show that our proposed model can achieve significant performance gains against state-of-the-art methods.
The contributions of this paper are summarized as follows:
• We introduce a novel method that trains the GNN encoder by presenting data in a tailored, easy-to-hard order, enabling more effective design of pretext tasks.
To the best of our knowledge, this is the first work to explore curriculum learning in graph self-supervised learning.
• We propose a unified framework that jointly reconstructs missing edges based on the unmasked graph structure and schedules training edges using a selfpaced learning strategy, thereby improving the effectiveness of the GNN encoder.
• We provide theoretical analysis of the convergence properties of the proposed Cur-MGAE method and demonstrate through extensive experiments that it consistently outperforms state-of-the-art graph SSL methods, including both contrastive and generative approaches.
The rest of the paper is organized as follows. We first introduce the details of our proposed Cur-MGAE method in Section 2. In Section 3, we present the experimental results to show the effectiveness of the method, including quantitative comparisons, ablation studies, etc. We review the related works in Section 4. Finally, we conclude this work in Section 5.
this section cite: ['b79', 'b28', 'b14', 'b6', 'b16', 'b68', 'b15', 'b65', 'b17', 'b64']

Section: Method
In this section, we present Cur-MGAE , a self-supervised framework designed to learn informative representations through the structure-aware curriculum. Specifically, Cur-MGAE consists of three key components: a structure-aware masked autoencoder, a complexity-guided curriculum masking module, and a self-paced mask scheduler. The overall framework is illustrated in Figure 1. The key notations in the method are summarized in Appendix A.
this section cite: []

Section: Structure-aware Masked Autoencoder
We propose a structure-aware masked autoencoder based on the edge reconstruction task to learn informative node representations without requiring extra supervision.
this section cite: []

Section: GNN Encoder.
Let the input graph be denoted as G = (V, E), where V and E are the sets of nodes and edges, respectively. It can be represented as G = (X, A), where X is the node feature matrix and A is the adjacency matrix. We apply a GNN to encode node representations:
h (k) v = COM(h (k-1) v , AGG(h (k-1) u : u ∈ Nv)), (1
) Scheduler Input Graph (!) Residual Graph (" = ! -! !" ) 0.8 0.6 0.7 0.8 0.9 0.1 0.1 0.1 0.2 0.2 Curriculum Edge Set (! ($) ) Masked Edge Set (! &'() = ! *+, + ! -+, ) split ratio encoder decoder sorted Cross-correlation decoder Input ( & ! = ! -! &'() ) MASK First Layer Last Layer MLP Reconstructed Graph (! !" )
this section cite: []

Section: Predict

this section cite: []

Section: ① Difficulty-based Edge Selection
Easy Hard ② Random-based Edge Selection Reconstruction Loss Self-paced Mask Scheduler Complexity-guided Curriculum Masking Structure-aware Masked Autoencoder … mask ratio upper-bound
(! (") ) (! $%& ) (! -! $%& ) (! '%& )
., 0 increasing
this section cite: []

Section: Figure 1.
The framework of our proposed Cur-MGAE method. Given an input graph, we first introduce a complexity-guided curriculum masking module to identify edges to be masked, where each edge is assigned a difficulty score based on its reconstruction residual error.
Next, we design a self-paced mask scheduler to dynamically schedule the masking curriculum according to the training stage. Finally, we employ a structure-aware masked autoencoder to perform self-supervised reconstruction of the graph structure.
where h
v is the embedding of node v at the k-th layer, and N v = {u : (v, u) ∈ E} denotes its neighborhood. AGG(•) aggregates messages from neighbors, and COM(•) combines them to update the embedding. We stack K GNN layers to derive multi-hop embeddings {h
(1) v , h (2) v , ..., h (K) v }.
The final node representations are denoted as ENC(X, A) ∈ R N ×d , where N is the number of nodes, d is the embedding dimension, and ENC(•) is the encoder.
Cross-correlation Decoder. After obtaining node embeddings from the GNN encoder, we design a cross-correlation decoder to reconstruct the graph structure by capturing inherent similarities between nodes following (Tan et al., 2023). Specifically, we compute edge embeddings as:
he v,u = || K k,j=1 h (k) v ⊙ h (j) u ,(2)
where ⊙ denotes element-wise multiplication, || represents concatenation, and h ev,u ∈ R dK 2 is the resulting edge embedding. This formulation emphasizes shared features between node pairs while suppressing dissimilar components, enabling the model to retain only highly correlated structural patterns. The resulting edge embeddings are fed into a multilayer perceptron (MLP) with a sigmoid activation to estimate the existence probability of each edge: g(v, u) = MLP(h ev,u ). By selectively preserving informative and correlated features, this design filters out noise and facilitates more accurate and efficient edge prediction.
Reconstruction Task. We adopt a mask-reconstruction paradigm for self-supervised learning to enhance the quality of the learned node representations. Specifically, we select a part of the edges to be masked in the original graph to obtain the perturbed graph: Ã = A -A mask . We denote A mask as the adjacency matrix of the masked edges E mask . Then we adopt the reconstruction loss as the supervision signal: L SSL = ℓ(A, DEC(ENC(X, Ã)), whose implementation is as follows:
LSSL = - 1 |E mask | (v,u)∈E mask log exp(g(v, u)) v ′ ∈V exp(g(v, v ′ )) . (3) g(•)
is the predicted probability of the presence of an existing edge, namely g(v, u
) = MLP(|| K k,j=1 h (k) v ⊙ h (j) u ),
where h (j) u and h
v denote the j-th and k-th hidden representation of node u and v, respectively. The learned node representations encode rich structural and attribute information, sufficient to reconstruct the original graph from perturbed inputs and further enhance performance in downstream tasks.
this section cite: ['b64']

Section: Complexity-guided Curriculum Masking
Since different edges in a graph contribute unequally to its structure, randomly masking edges can pose optimization challenges during the reconstruction process. In particular, masking too many critical structural edges early in training may lead to excessively difficult reconstruction tasks, especially when the GNN encoder is still undertrained. To mitigate this issue, we introduce a complexity-guided curriculum masking module that progressively increases task difficulty by selecting edges in an easy-to-hard manner. The key idea is to identify structurally important edges and postpone their masking, thereby enabling a smoother and more effective learning trajectory.
Specifically, we identify edges that are structurally important to the graph and progressively mask them to increase the learning difficulty during the training process. We formally define the difficulty of an edge as a score reflecting how challenging it is for the current model to predict the edge correctly. To quantify this difficulty, we first use the current model to reconstruct the original graph as: A re = DEC(ENC(X, A)).
The reconstructed adjacency matrix A re captures the model's internal estimation of edge probabilities, which can be interpreted as its confidence in the existence of each edge. Intuitively, lower confidence suggests that an edge is harder to reconstruct for the current model, indicating higher structural complexity. We therefore propose to use the structural residual (Zhang et al., 2023a) between the original and predicted graphs as a proxy for edge difficulty: R = A -A re . By applying a masking strategy that targets the K easiest edges, those with the smallest residuals, we simplify the reconstruction task, particularly during the early training stages. This allows the model to focus on learning fundamental structural patterns before encountering more complex ones (Zhang et al., 2023a;Li et al., 2023b). Consequently, this curriculum-guided masking strategy enhances both the efficiency and effectiveness of the training process.
this section cite: []

Section: Self-paced Mask Scheduler
Here, we propose a self-paced mask scheduler to progressively and autonomously incorporate an increasing number of edges throughout the training process (Zhang et al., 2023a;Li et al., 2023b). One straightforward solution is to gradually increase the value of K during the training process. However, dynamically identifying and updating a suitable K during training is non-trivial. Edge selection inherently poses a discrete optimization problem over a large topological space, which significantly complicates the learning process. To address this, we relax the edge selection matrix S (t)  from binary values to continuous values within [0, 1], transforming the problem into a continuous constrained optimization task. Specifically, we treat the masking constraint as a Lagrangian multiplier and introduce a regularization component to the loss function:
f (S; λ, A) = λ||S (t) ⊙ A -A||.
Here, S (t) denotes the soft edge selection matrix at training iteration t, sharing the same dimensions as the adjacency matrix A. After optimization, the entries in S (t) are thresholded at 0.5 to yield a binary mask 0, 1 for edge selection, resulting in the masked adjacency matrix A (t) = S (t) ⊙ A, where ⊙ denotes element-wise multiplication.
Note that the regularization term promotes the masking of as many edges as possible, governed by the coefficient λ. As λ increases during training, more edges are gradually incorporated. This process effectively schedules edge masking in an easy-to-hard manner. The evolving strategy for updating λ is detailed in Appendix D.3. Combining both the residual-based selection and the regularization term (Zhang et al., 2023a), the loss function for our self-paced mask scheduler is given by:
LSP CL = β i,j SijRij + f (S; λ, A),(4)
where β is a balancing hyperparameter, S extracts the selected edges, and R ij = ||A ij -Ã(t) ij || denotes the edge residual, with Ã(t) ij being the predicted edge value. The L 2 norm is used for measuring residuals.
However, continuously increasing the number of masked edges may eventually compel the model to make uninformed or arbitrary predictions about the graph structure. To avoid this issue, we introduce a mask ratio hyperparameter that constrains the maximum proportion of edges allowed to be masked. Another critical concern is that relying solely on difficulty-based edge selection tends to consistently mask only the easiest edges during training. This could limit the model's generalizability by overfitting to these simpler structures. To address this, we introduce a split ratio hyperparameter that enables a portion of the masked edges to be selected randomly. Specifically, a fraction of the masked edges, denoted as A DES , is sampled uniformly at random, while the remaining edges, denoted as A RES , are selected based on their difficulty scores. The union of these two subsets forms the complete masked edge set A mask . A lower split ratio results in a higher degree of randomness in the masking process, thereby encouraging exploration. Conversely, a higher split ratio prioritizes difficulty-based masking, enhancing exploitation. By appropriately tuning this hyperparameter, we strike a balance between exploration and exploitation, thereby mitigating overfitting and improving the quality of Algorithm 1 The optimization process of Cur-MGAE 1: Input: Node features X, adjacency matrix A, step size µ, regularization coefficient γ 2: Output: Trained GNN parameters w 3: Initialize w (0) , S (0) , and t) to the continuous domain and optimize it as
λ (0) 4: Compute A (0) = S (0) ⊙ A 5: while not converged do 6: w (t) = argmin w LSSL(X, A (t-1) ; w) + γ 2 ||w - w (t-1) || 7: Generate embedding Z (t) from A using the updated GNN model f 8: For all node pairs (i, j), predict edge existence Ã(t) ij = g(z (t) i , z (t) j ) 9: Relax S (
S (t) = argmin S LSP CL + γ 2 ||S -S (t-1) || 10: K = |{(i, j) : S (t) ij ≥ 0.5}| 11: if K ≥ mask ratio × |E|
then 12: K = mask ratio × |E| 13: else 14: Update λ based on the designed curriculum pace 15: end if 16: Select the top-split ratio × K edges with the highest Sij values and assign to S (t) DES 17: Randomly select the remaining (1split ratio) × K edges and assign to S (t) RES 18: S (t) = S (t) DES + S (t) RES 19: Update the perturbed adjacency matrix A (t) = S (t) ⊙ A 20: end while the learned node representations.
this section cite: []

Section: Optimization Procedure
Our proposed model aims to minimize the objective function L all , which involves optimizing two distinct sets of parameters. This naturally leads to a challenging bi-level optimization problem. To address this, we design an optimization algorithm that jointly trains two separate selfsupervised modules, each with its corresponding objective. The overall loss function is formulated as:
L all = LSSL + LSP CL.(5)
To ensure a smooth transition across training iterations, we incorporate regularization terms into the optimization process: γ 2 ||w -w (t-1) || and γ 2 ||S -S (t-1) ||. These terms penalize abrupt changes in the model parameters w and the edge selection matrix S, thereby stabilizing training dynamics. The complete training procedure is outlined in Algorithm 1.
Time Complexity. The time complexity of our proposed model is O(Ed+N d 2 ), where N and E denote the numbers of nodes and edges in the graph, respectively, and d is the dimensionality of the node representations. Specifically, our model adopts a message-passing GNN as the encoder, which incurs a complexity of O(Ed + N d 2 ). The decoder and the self-paced mask scheduler each contribute a time complexity of O(Ed), as they involve computing residual errors for each edge. The complexity-guided curriculum masking module operates with a time complexity of O(E), since it selects from existing edges rather than the full N ×N set of potential edges. Overall, the time complexity of our method is comparable to the baselines, which also typically scale as O(Ed + N d 2 ).
this section cite: []

Section: Theoretical Analyses
We present theoretical analyses on the convergence properties of our method in Theorems 1 and 2 following (Zhang et al., 2023a). Detailed proofs are provided in Appendix B.
this section cite: []

Section: Theorem 1 (Convergence Away from Saddle Points).
For a sufficiently large γ, if the second derivatives of L SSL (X, A (t-1) ; w) and f (S; λ, A) are continuous, any bounded sequence (w (t) , S (t) ) generated by Algorithm 1 with random initialization will almost surely avoid convergence to any strict saddle point of L all .
Theorem 2 (Convergence to Second-order Stationary Points). For a sufficiently large γ, if the second derivatives of L SSL (X, A (t-1) ; w), and f (S; λ, A) are continuous, and both functions satisfy the Kuradyka-Lojasiewicz (KL) property (Wang et al., 2022b), then any bounded sequence (w (t) , S (t) ) generated by Algorithm 1 with random initialization will almost surely converge to a second-order stationary point of L all .
this section cite: []

Section: Experiment
In this section, we conduct comprehensive experiments to evaluate the effectiveness of the proposed Cur-MGAE method. This includes the experimental setup, quantitative evaluations on node classification and link prediction benchmarks, and in-depth analyses. Additional experimental results are provided in Appendix G.
this section cite: []

Section: Experimental Setup
Datasets. We evaluate node classification on three Planetoid datasets (Cora, Citeseer, and Pubmed (Sen et al., 2008)) and three commonly used citation/co-authorship datasets: Coauthor-CS (Shchur et al., 2019), Coauthor-Physics (Shchur et al., 2019), and OGBN-arxiv (Hu et al., 2020a). Accuracy (%) is used as the evaluation metric for these tasks. For link prediction, we use the same three Planetoid datasets and additional large-scale benchmarks from the Open Graph Benchmark (OGB) (Hu et al., 2021), including OGBN-ddi, OGBL-collab, and OGBL-ppa. We report the area under the ROC curve (AUC, %) (Bradley, 1997) for the three Planetoid datasets, and the Hit rate (Hits@N) for OGB datasets, following (Tan et al., 2023) for fair comparison. Baselines. We compare Cur-MGAE against two groups of state-of-the-art baselines.
The first group includes contrastive graph self-supervised learning methods: DGI (Veličković et al., 2019), GIC (Mavromatis & Karypis, 2021), MVGRL (Hassani & Khasahmadi, 2020), and BGRL (Thakoor et al., 2022). The second group includes generative graph SSL methods such as GAE (Kipf & Welling, 2016), GraphSAGE (Hamilton et al., 2017), ARGVA (Pan et al., 2019), GPT-GNN (Hu et al., 2020b), RRL (Zhu et al., 2020), GraphMAE (Hou et al., 2022), GraphMAE2 (Hou et al., 2023), MaskGAE (Li et al., 2023d), Bandana (Zhao et al., 2024), AUG-MAE (Wang et al., 2024), and S2GAE (Tan et al., 2023).
this section cite: ['b57', 'b58', 'b58', 'b20', 'b2', 'b64', 'b68', 'b50', 'b15', 'b65', 'b27', 'b14', 'b53', 'b101', 'b17', 'b18', 'b96', 'b72', 'b64']

Section: Experimental Results
Node Classification. Table 1 summarizes the node classification accuracy of Cur-MGAE and all baselines. Our method outperforms both contrastive and generative selfsupervised baselines, achieving the highest average rank. This result demonstrates the benefit of scheduling training data using a difficulty-aware curriculum derived from reconstruction residuals, which enables more effective representation learning. For instance, Cur-MGAE improves classification accuracy by 1.67% on Pubmed and nearly 1% on OGBN-arxiv compared to the strongest baseline.
this section cite: []

Section: Link Prediction.
Table 2 presents the link prediction performance of Cur-MGAE and baseline methodsfoot_0 . The results indicate that generative graph SSL methods (e.g., GraphMAE, MaskGAE, S2GAE) generally outperform contrastive methods, highlighting the effectiveness of the reconstruction-based pretext tasks that recover masked structures from the remaining graph context. Our curriculumbased method Cur-MGAE achieves the best performance on 2 out of 6 datasets and reports competitive results on the remaining datasets. For example, it improves performance by approximately 2% over the strongest baselines on OGBLddi and OGBL-ppa. This improvement is attributed to Cur-MGAE 's ability to adaptively select training samples based on task difficulty, unlike most baselines that treat all training data equally, leading to suboptimal results. MaskGAE (Li et al., 2023d), a strong baseline that jointly reconstructs masked edges and node degrees, performs well on smaller datasets but underperforms on large-scale benchmarks. A plausible explanation is that it overlooks the varying difficulty of reconstructing different edges, which becomes more impactful in large, complex graphs where informative representations are harder to extract. In contrast, our method introduces a curriculum-driven strategy that prioritizes easier samples in early training stages and progressively incorporates harder ones. Notably, none of the baselines achieves consistently strong performance across all datasets, whereas Cur-MGAE demonstrates stable and superior effectiveness.
this section cite: []

Section: Visualization of Learned Edge Selection Curriculum
To qualitatively evaluate the learned edge selection strategy, we construct synthetic datasets with ground-truth edge difficulty labels, following previous works (Karimi et al., 2018;Abu-El-Haija et al., 2019;Zhang et al., 2023a). Each synthetic graph consists of 5,000 nodes partitioned into 10 equally sized groups, with node labels ranging from 1 to 10. The corresponding visualization is provided in Appendix F. The node features are generated from overlapping multi-Gaussian distributions that define each node's position in the feature space. The labels are then assigned according to the feature distributions, resulting in 10 distinct classes.
Edge difficulty is defined based on the label similarity between node pairs: edges between nodes with identical labels are considered easy, those between adjacent labels are of medium difficulty, and those connecting nodes with distant labels are deemed hard to reconstruct. To control the prevalence of easy edges, we introduce a homophily coefficient (homo) that specifies the ratio of easy edges in the graph. For all other potential edges, the connection probability decreases exponentially with label distance. Formally, the probability of an edge between nodes u and v is defined as: p uc ∝ e -|cu-cv| , where |c u -c v | denotes the shortest label distance in a circular label arrangement. We generate three synthetic datasets by setting the homophily coefficient to 0.1, 0.5, 0.9, and split each graph into training, validation, and test sets with equal numbers of nodes.
Learned Edge Selection Curriculum. Using the synthetic datasets described above, we visualize and compare the learned edge selection curriculum with the ground-truth edge difficulty distribution. Figure 2 shows the proportion of selected edges categorized as easy, medium, and hard throughout training. In this figure, each row corresponds to a different homophily coefficient, while each column represents a different split ratio, a hyperparameter that controls the trade-off between exploration and exploitation in edge selection. Across all settings, we observe a consistent trend: the model initially favors selecting easier edges and gradually incorporates harder ones as training progresses. This behavior is consistent with the intended design of our curriculum-driven training strategy. Specifically, in early epochs, the model focuses on easier edges that align with its current learning capacity. As training advances and the model becomes more expressive, it begins to select increasingly difficult edges, effectively expanding its learning scope in a controlled, progressive manner. Interestingly, the homophily coefficient influences the dynamics of the easy-tohard transition. As illustrated in the stacked plots, where the blue, orange, and red regions represent the relative proportions of selected easy, medium, and hard edges at each epoch, low homophily leads to a more aggressive curriculum. In this setting, the model selects a substantial fraction of medium and hard edges even in the early stages of training. In contrast, under high homophily, early edge selection is dominated by easy edges, with medium and hard edges incorporated more gradually. This pattern suggests that in highly homophilous graphs, the model adopts a more conservative learning trajectory, initially relying on structurally similar (and easier) edges before progressively transitioning to more challenging ones. These visualizations confirm that our structure-aware masking curriculum behaves as expected, progressively selecting edges from easier to harder throughout the training process.
this section cite: ['b26', 'b0']

Section: Ablation Studies
To evaluate the effectiveness of our key modules and designs, we conduct ablation studies by modifying the components of Cur-MGAE . For simplicity, we present results on Cora, Citeseer, OGBL-ddi, and Coauthor-CS in Table 3, with similar trends observed on the remaining datasets. We use AUC (%) for link prediction and accuracy (%) for node classification as evaluation metrics.
Variant 'w/o Curri.'. It disables the complexity-guided curriculum masking module (Section 2.2) and instead applies random masking. The performance drop demonstrates
(SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG (SRFKV 3URSRUWLRQRIVHOHFWHGHGJHV KRPR VSOLWUDWLR HDV\ PHGLXP KDUG that treating all samples equally during training can be suboptimal. This supports the benefit of our structure-aware curriculum design in promoting better performance.
Variant 'split ratio is 0'. Setting the split ratio to 0 means that the self-paced mask scheduler (Section 2.3) randomly selects a fixed number of edges without considering their difficulty scores. Although this introduces stochasticity and helps prevent overfitting, the lack of difficulty-aware edge selection leads to noticeable performance degradation, highlighting the importance of meaningful edge scheduling.
Variant 'split ratio is 1'. In this variant, we set the split ratio to 1, which removes the randomness in the edge selection process of the self-paced mask scheduler (Section 2.3). Together with the previous variant, these two variants aim to assess the effectiveness of the self-paced scheduling mechanism. Without randomness, the scheduler consistently selects edges with the smallest difficulty scores during early training stages. This deterministic behavior may lead to overfitting and limit the model's ability to generalize, as reflected in the performance drop compared to the full model.
Variant 'w/o CC Dec.'. This variant evaluates the impact of our specially designed cross-correlation decoder in the structure-aware masked autoencoder (Section 2.1). We replace it with a simpler inner product decoder while keeping all other components unchanged. The significant drop in performance demonstrates that the cross-correlation decoder, by capturing multi-granular shared features between connected nodes, facilitates more informative representation learning and improves reconstruction accuracy.
Variant 'w/o CC Dec. & Curri.'. In this variant, both the decoder and the curriculum masking are removed: we use an inner product decoder and randomly mask the training samples. This combination results in the worst performance across all settings, confirming that the two modules, i.e., decoder and curriculum, play synergistic and crucial roles in achieving strong representation learning.
Overall, these ablation studies confirm the importance of the key components in our proposed method. The tailored structure-aware curriculum enables the identification of more informative edges by leveraging difficulty scores, while the split ratio introduces a controlled level of stochasticity to avoid overfitting. Meanwhile, the cross-correlation decoder mitigates the representational limitations introduced by the masking process and significantly enhances the reconstruction quality.
this section cite: []

Section: Related Work
Graph Self-Supervised Learning. Graph self-supervised learning (SSL) techniques (Liu et al., 2022;You et al., 2020;Peng et al., 2020;Xu et al., 2021;Sun et al., 2023b;Li et al., 2022c;2021b;2024a) are typically categorized into contrastive and generative paradigms. Recently, contrastive methods have gained significant attention. These approaches primarily focus on negative sampling strategies, such as corruption-based negative pair construction in DGI (Veličković et al., 2019), and in-batch negatives as used in GCA (Zhu et al., 2021). In contrastive learning, graph augmentation plays a critical role in creating effective training signals. However, the theoretical understanding of graph augmentation remains limited, raising concerns about its label invariance and optimality. In contrast, generative SSL methods aim to reconstruct missing parts of input graphs and are generally divided into autoregressive and autoencoding models. Although generative approaches have historically underperformed compared to contrastive methods, several notable autoregressive models have emerged, such as GPT-GNN (Hu et al., 2020b). In the autoencoding category, early models like GAE and VGAE (Kipf & Welling, 2016) set foundational benchmarks. More recent advances include GraphMAE (Hou et al., 2022), GraphMAE2 (Hou et al., 2023), GigaMAE (Shi et al., 2023), SeeGera (Li et al., 2023e), RARE (Tu et al., 2023), S2GAE (Tan et al., 2023), and Bandana (Zhao et al., 2024). Nonetheless, most existing methods neglect the varying difficulty levels of selfsupervised tasks, treating all training samples equally and resulting in suboptimal performance on downstream tasks.
Curriculum Learning. Curriculum Learning (CL) is a training strategy that starts with simpler tasks and gradually moves to more complex ones, inspired by the way humans learn in educational settings (Bengio et al., 2009;Wang et al., 2021a;Zhou et al., 2023;2024;Huang et al., 2024;Zhang et al., 2024;Yao et al., 2024;Ge et al., 2025). A foun-dational approach in this domain is the "Baby Step" algorithm (Spitkovsky et al., 2010), which controls both the complexity and order of training samples. This idea was later extended to the self-paced learning paradigm (Kumar et al., 2010), which selects training samples based on their loss values, allowing models to learn at their own pace. In addition, several automatic CL frameworks have been proposed, including transfer teacher (Hacohen & Weinshall, 2019), reinforcement learning-based curriculum teacher (Zhao et al., 2020), and others customized for different datasets, models, and objectives (Sinha et al., 2020). CL has also been integrated into various domains such as disentangled recommendation systems (Chen et al., 2021;Wang et al., 2023), combinatorial optimization (Zhang et al., 2022c), neural architecture search (Zhou et al., 2022;Yao et al., 2024;Qin et al., 2023), and video grounding (Lan et al., 2023).
Several studies have extended CL to graph domains (Li et al., 2023b), such as GNN-CL (Li et al., 2024b) and Cur-Graph (Wang et al., 2021b). A central component of CL methods is the mechanism to assess sample complexity and guide the training schedule by determining the order or relative importance of samples. However, most existing methods heavily rely on label supervision to train encoders and overlook self-supervised settings where labels are not available. More importantly, these methods typically treat training samples as independent instances, whereas our approach addresses a more challenging structure-aware curriculum setting in which training samples are inherently interconnected and thus cannot be treated as independent.
this section cite: ['b48', 'b82', 'b55', 'b78', 'b68', 'b102', 'b27', 'b17', 'b18', 'b59', 'b66', 'b64', 'b96', 'b1', 'b98', 'b22', 'b87', 'b80', 'b11', 'b61', 'b29', 'b13', 'b94', 'b60', 'b4', 'b74', 'b97', 'b80', 'b56', 'b30']

Section: Conclusion
In this paper, we propose a graph self-supervised learning strategy based on curriculum learning, named Cur-MGAE, which builds upon a masked graph autoencoder framework.
Our method assesses the difficulty of reconstructing masked edges and thus guides the model to learn in an easy-to-hard manner. This approach leads to the acquisition of more informative graph representations. We provide a theoretical analysis of the convergence properties of the proposed method.
Extensive experiments on real-world benchmarks demonstrate that our proposed method consistently outperforms state-of-the-art graph self-supervised learning baselines in both node classification and link prediction tasks.
In addition, the second-order derivatives of the reconstruction term i,j
S ij A ij - Ã(t) ij are also bounded, which implies max    ∇ 2 w i,j S ij A ij - Ã(t) ij , ∇ 2 S i,j S ij A ij - Ã(t) ij    ≤ q,
where q > 0 is a constant and Ã is a function of w.
As a result, the objective function L all is bi-smooth, i.e.,
max ∇ 2 w L all , ∇ 2 S L all ≤ p + q,
and L all satisfies Assumption 4 in (Li et al., 2019b). Therefore, according to Theorem 10 in (Li et al., 2019b), the secondorder derivative of L all is continuous, and for any γ > p + q, any bounded sequence (w (t) , S (t) ) generated by Algorithm 1 almost surely avoids convergence to a strict saddle point of L all (Zhang et al., 2023a).
this section cite: ['b45', 'b45']

Section: References
Ref_id:b0 Title: Higher-order graph convolutional architectures via sparsified neighborhood mixing Year: (2019)
Ref_id:b1 Title: Curriculum learning Year: (2009)
Ref_id:b2 Title: The use of the area under the roc curve in the evaluation of machine learning algorithms Year: (1997)
Ref_id:b3 Title: Multimodal graph neural architecture search under distribution shifts Year: (2024)
Ref_id:b4 Title: Curriculum disentangled recommendation with noisy multi-feedback Year: (2021)
Ref_id:b5 Title: Autogfm: Automated graph foundation model with adaptive architecture customization Year: (2025)
Ref_id:b6 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b7 Title: Eliciting structural and semantic global knowledge in unsupervised graph contrastive learning. AAAI'23/IAAI'23/EAAI'23 Year: (2023)
Ref_id:b8 Title: Propagation enhanced neural message passing for graph representation learning Year: (2021)
Ref_id:b9 Title: Multimedia cognition and evaluation in open environments Year: (2023)
Ref_id:b10 Title: Higherorder interaction goes neural: A substructure assembling graph attention network for graph classification Year: (2021)
Ref_id:b11 Title: Dynamic mixture of curriculum lora experts for continual multimodal instruction tuning Year: (2025)
Ref_id:b12 Title: Model augmentation tricks for graph contrastive learning Year: ()
Ref_id:b13 Title: On the power of curriculum learning in training deep networks Year: (2019)
Ref_id:b14 Title: Inductive representation learning on large graphs. NIPS'17 Year: (2017)
Ref_id:b15 Title: Contrastive multiview representation learning on graphs Year: (2020)
Ref_id:b16 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b17 Title: Graphmae: Self-supervised masked graph autoencoders Year: (2022)
Ref_id:b18 Title: Graphmae2: A decoding-enhanced masked self-supervised graph learner Year: (2023)
Ref_id:b19 Title: Open graph benchmark: Datasets for machine learning on graphs Year: ()
Ref_id:b20 Title: Ogb-lsc: A large-scale challenge for machine learning on graphs Year: (2021)
Ref_id:b21 Title: Gptgnn: Generative pre-training of graph neural networks Year: ()
Ref_id:b22 Title: Neighbor does matter: Curriculum global positive-negative sampling for vision-language pretraining Year: (2024)
Ref_id:b23 Title: Heterogeneous graph propagation network Year: (2021)
Ref_id:b24 Title: Aptrank: an adaptive pagerank model for protein function prediction on bi-relational graphs Year: (2017)
Ref_id:b25 Title: Graph neural network for traffic forecasting: A survey Year: (2021)
Ref_id:b26 Title: Homophily influences ranking of minorities in social networks Year: (2018)
Ref_id:b27 Title: Variational graph auto-encoders Year: (2016)
Ref_id:b28 Title: Semi-supervised classification with graph convolutional networks Year: (2017)
Ref_id:b29 Title: Advances in neural information processing systems Year: (2010)
Ref_id:b30 Title: Curriculum multi-negative augmentation for debiased video grounding Year: (2023)
Ref_id:b31 Title: Augmentation-free selfsupervised learning on graphs Year: ()
Ref_id:b32 Title: Fates of microscopic social ecosystems: Keep alive or dead? Year: (2019)
Ref_id:b33 Title: Intention-aware sequential recommendation with structured intent transition Year: (2021)
Ref_id:b34 Title: Disentangled contrastive learning on graphs Year: (2021)
Ref_id:b35 Title: Ood-gnn: Out-of-distribution generalized graph neural network Year: (2022)
Ref_id:b36 Title: Out-of-distribution generalization on graphs: A survey Year: (2022)
Ref_id:b37 Title: Disentangled graph contrastive learning with independence promotion Year: (2022)
Ref_id:b38 Title: Learning invariant graph representations for out-of-distribution generalization Year: (2022)
Ref_id:b39 Title: Augmentation-free graph contrastive learning of invariantdiscriminative representations Year: (2023)
Ref_id:b40 Title: Curriculum graph machine learning: A survey Year: (2023)
Ref_id:b41 Title: Invariant node representation learning under distribution shifts with multiple latent environments Year: (2023)
Ref_id:b42 Title: Disentangled graph self-supervised learning for out-of-distribution generalization Year: (2024)
Ref_id:b43 Title: Disentangling invariant subgraph via variance contrastive estimation under distribution shifts Year: (2025)
Ref_id:b44 Title: What's behind the mask: Understanding masked graph modeling for graph autoencoders Year: (2023)
Ref_id:b45 Title: Alternating minimizations converge to second-order optimal solutions Year: (2019-06)
Ref_id:b46 Title: Selfsupervised semi-implicit graph variational auto-encoders with masking Year: (2023)
Ref_id:b47 Title: Graph neural network with curriculum learning for imbalanced node classification Year: (2024)
Ref_id:b48 Title: Graph self-supervised learning: A survey Year: (2022)
Ref_id:b49 Title: Disentangled graph convolutional networks Year: (2019)
Ref_id:b50 Title: Graph infoclust: Maximizing coarse-grain mutual information in graphs Year: (2021)
Ref_id:b51 Title: Lasagne: A multi-layer graph convolutional network framework via node-aware deep architecture Year: (2021)
Ref_id:b52 Title: Simple unsupervised graph representation learning Year: (2022-06)
Ref_id:b53 Title: Adversarially regularized graph autoencoder for graph embedding Year: (2019)
Ref_id:b54 Title: Learning graph embedding with adversarial training methods Year: (2020)
Ref_id:b55 Title: Graph representation learning via graphical mutual information maximization Year: (2020)
Ref_id:b56 Title: Multitask graph neural architecture search with task-aware collaboration and curriculum Year: (2023)
Ref_id:b57 Title: Collective classification in network data Year: (2008)
Ref_id:b58 Title: Pitfalls of graph neural network evaluation Year: (2019)
Ref_id:b59 Title: Gigamae: Generalizable graph masked autoencoder via collaborative latent space reconstruction Year: (2023)
Ref_id:b60 Title: Curriculum by smoothing Year: (2020)
Ref_id:b61 Title: From baby steps to leapfrog: How "less is more" in unsupervised dependency parsing Year: (2010)
Ref_id:b62 Title: Graph contrastive learning with intrinsic augmentations Year: (2023)
Ref_id:b63 Title: Progressive hard negative masking: From global uniformity to local tolerance Year: (2023)
Ref_id:b64 Title: S2gae: Self-supervised graph autoencoders are generalizable learners with graph masking Year: (2023)
Ref_id:b65 Title: Largescale representation learning on graphs via bootstrapping Year: (2022)
Ref_id:b66 Title: Rare: Robust masked graph autoencoder Year: (2023)
Ref_id:b67 Title: Graph attention networks Year: (2018)
Ref_id:b68 Title: Deep Graph Infomax Year: (2019)
Ref_id:b69 Title: Graphgan: Graph representation learning with generative adversarial nets Year: (2018)
Ref_id:b70 Title: Augmentationfree graph contrastive learning with performance guarantee Year: (2022)
Ref_id:b71 Title: Accelerated gradient-free neural network training by multi-convex alternating optimization Year: (2022)
Ref_id:b72 Title: Rethinking graph masked autoencoders through alignment and uniformity Year: (2024)
Ref_id:b73 Title: A survey on curriculum learning Year: (2021)
Ref_id:b74 Title: Curriculum co-disentangled representation learning across multiple environments for social recommendation Year: (2023-07)
Ref_id:b75 Title: Curgraph: Curriculum learning for graph classification Year: (2021)
Ref_id:b76 Title: Measure and Integral: An Introduction to Real Analysis Year: (1977)
Ref_id:b77 Title: Moleculenet: a benchmark for molecular machine learning Year: (2018)
Ref_id:b78 Title: Infogcl: Information-aware graph contrastive learning Year: (2021)
Ref_id:b79 Title: How powerful are graph neural networks? Year: (2019)
Ref_id:b80 Title: Data-augmented curriculum graph neural architecture search under distribution shifts Year: (2024)
Ref_id:b81 Title: Sparse graph attention networks Year: (2021)
Ref_id:b82 Title: Graph contrastive learning with augmentations Year: (2020)
Ref_id:b83 Title: Graph contrastive learning with adaptive augmentation for knowledge concept recommendation Year: (2023)
Ref_id:b84 Title: From canonical correlation analysis to self-supervised graph neural networks Year: (2021)
Ref_id:b85 Title: Link prediction based on graph neural networks Year: (2018)
Ref_id:b86 Title: Covariance-preserving feature augmentation for graph contrastive learning Year: (2022)
Ref_id:b87 Title: Large language model with curriculum reasoning for visual concept recognition Year: (2024)
Ref_id:b88 Title: Deep learning on graphs: A survey Year: (2020)
Ref_id:b89 Title: Eigen-gnn: A graph structure preserving plug-in for gnns Year: (2021)
Ref_id:b90 Title: Dynamic graph neural networks under spatio-temporal distribution shift Year: (2022)
Ref_id:b91 Title: Learning to solve travelling salesman problem with hardness-adaptive curriculum Year: (2022)
Ref_id:b92 Title: Relational curriculum learning for graph neural networks Year: (2023)
Ref_id:b93 Title: Spectral invariant learning for dynamic graphs under distribution shifts Year: (2023)
Ref_id:b94 Title: Reinforced curriculum learning on pre-trained neural machine translation models Year: (2020)
Ref_id:b95 Title: Coarse-to-fine contrastive learning on graphs Year: (2023)
Ref_id:b96 Title: Masked graph autoencoder with non-discrete bandwidths Year: (2024)
Ref_id:b97 Title: Curriculum-nas: Curriculum weight-sharing neural architecture search Year: (2022)
Ref_id:b98 Title: Intraand inter-modal curriculum for multimodal learning Year: (2023)
Ref_id:b99 Title: Curbench: curriculum learning benchmark Year: (2024)
Ref_id:b100 Title: Crossview graph contrastive learning with hypergraph Year: (2023-11)
Ref_id:b101 Title: Self-supervised training of graph convolutional networks Year: (2020)
Ref_id:b102 Title: Graph contrastive learning with adaptive augmentation Year: (2021)
