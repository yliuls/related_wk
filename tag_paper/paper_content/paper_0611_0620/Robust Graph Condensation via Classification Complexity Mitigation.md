Title: Robust Graph Condensation via Classification Complexity Mitigation
Abstract: Graph condensation (GC) has gained significant attention for its ability to synthesize smaller yet informative graphs. However, existing studies often overlook the robustness of GC in scenarios where the original graph is corrupted. In such cases, we observe that the performance of GC deteriorates significantly, while existing robust graph learning technologies offer only limited effectiveness. Through both empirical investigation and theoretical analysis, we reveal that GC is inherently an intrinsic-dimension-reducing process, synthesizing a condensed graph with lower classification complexity. Although this property is critical for effective GC performance, it remains highly vulnerable to adversarial perturbations. To tackle this vulnerability and improve GC robustness, we adopt the geometry perspective of graph data manifold and propose a novel Manifold-constrained Robust Graph Condensation framework named MRGC. Specifically, we introduce three graph data manifold learning modules that guide the condensed graph to lie within a smooth, low-dimensional manifold with minimal class ambiguity, thereby preserving the classification complexity reduction capability of GC and ensuring robust performance under universal adversarial attacks. Extensive experiments demonstrate the robustness of MRGC across diverse attack scenarios.

Section: Introduction
Recently, Graph Condensation (GC) [17,61] has emerged as a promising approach to enhance the training efficiency of Graph Neural Networks (GNNs) by condensing large graphs into smaller ones. These smaller yet highly informative synthesized graphs enable GNNs trained on them to achieve performance comparable to models trained on larger original graphs [20,41,54]. This has facilitated the adoption of GC in areas like neural architecture search [33] and graph continual learning [43].
However, the quality of the condensed graph largely depends on the original graph while existing GC methods assume a clean original graph. As shown in Figure 1(a), when the original graph is attacked, the quality of the condensed graph deteriorates, adversely impacting the applications of GC in real-world scenarios where noise and attackers are prevalent [53]. Nevertheless, GC robustness remains largely unexplored. RobGC [16] is the first attempt to tackle this issue. While RobGC effectively addresses structure attacks, its dependence on structure learning and label propagation limits its defense against feature and label attacks [17]. Benchmark [20] also reveals that GC is
p p p p p p 3HUWXUEDWLRQ5DWH $FFXUDF\ 6WUXFWXUH$WWDFNp )HDWXUH$WWDFNp /DEHO$WWDFNp (a) GCond under attacks. *&RQG *&RQG 0 *&RQG 0 05*& RXUV $FFXUDF\ (b) GC with robust GNN. ,QWULQVLF'LPHQVLRQ )LVKHU V'LVFULPLQDQW5DWLR )UDFWLRQRI+\SHUVSKHUHV &RYHULQJ'DWD 2ULJLQDO*UDSK &RQGHQVHG*UDSK &RQGHQVHG*UDSK (c) Classification complexity evaluation. vulnerable to feature attacks. Three key questions about GC robustness remain unsolved: (Q1) Can existing robust graph learning techniques improve GC robustness? (Q2) What key property of GC is disrupted by attacks, causing the performance degradation, and can it be theoretically understood? (Q3) How to design a defense strategy to counter universal attacks in GC?
Answer to Q1: Given that most current training-based GC methods utilize GNN as their backbone [20], one intuitive approach to enhance GC robustness is to leverage existing robust GNN technologies.
To investigate this possibility, we conduct two toy cases using GCond [33] as a representative GC method. First, we integrate MedianGCN [8] (a classic robust GNN [10,70]) as the GC backbone. Second, we train MedianGCN on the condensed graph synthesized by the standard GCond. As shown in Fig. 1(b), both strategies fail to work effectively, with even worse performance than the standard GC method. This may be because most existing robust GNNs enhance robustness using attention-like mechanisms [10], which have been shown to perform poorly even in clean GC scenarios [20,54,41].
(Results with more robust GNNs are in Appendix C). This result suggests that existing robust GNN techniques may fail to enhance GC robustness, highlighting the need for an innovative solution.
Answer to Q2: Since existing GC research primarily focuses on the classification task [17], we examine how attacks affect the classification-related properties of GC. Inspired by the classification complexity theory [25,44,36], which investigates classification problems through the geometric properties of classes with the three key factors are intrinsic dimension, boundary complexity, and class ambiguity [44], we explore how the GC process influences the classification complexity of graphs and its behavior under attack through the lens of this theory. To evaluate the classification complexity of graphs during the GC process, we employ three widely used metrics [44]: Intrinsic Dimension, Fisher's Discriminant Ratio, and the Fraction of Hyperspheres Covering Data (Details are in Appendix C). As shown in Figure 1(c), all metrics decrease after GC with an average reduction of 89.25%, indicating that GC will reduce the classification complexity. However, when adversarial attacks occur, we observe an average increase of 547.54% across all metrics in the condensed graph. This reveals an interesting insight: while GC reduces the classification complexity, adversarial perturbations counteract this classification-complexity-reducing property.
Answer to Q3: Based on our analysis, designing a defense strategy to preserve the key property of GC under universal attacks to improve robustness presents three challenges: How to (1) reduce the intrinsic dimension, (2) minimize the complexity of class boundaries, and (3) resolve class ambiguity to mitigate the increasing classification complexity in the condensed graph under attacks? To address these, we explore GC robustness from the geometric perspective of graph data manifolds and propose a novel Manifold-constrained Robust Graph Condensation framework, named MRGC. To maintain a low intrinsic dimension of the condensed graph (Challenge 1), we designed an Intrinsic Dimension Manifold Regularization Module to constrain the condensed graph in a low-dimensional manifold. To reduce the complexity of class boundaries (Challenge 2), we introduce a Curvature-Aware Manifold Smoothing Module to smooth the class manifold in the condensed graph, thereby simplifying the class boundaries. To relieve class ambiguity (Challenge 3), we develop a Class-Wise Manifold Decoupling Module that mitigates potential class bias by minimizing the overlap between class manifolds. Our main contributions are summarized as follows:
• We empirically and theoretically demonstrate that GC inherently reduces the classification complexity of graphs, a property vulnerable to adversarial attacks targeting the original graph and remains unprotected by existing robust graph learning techniques.
• We adopt a geometric perspective of graph data manifold and propose MRGC to enhance GC robustness by protecting its classification complexity reduction property.
• To the best of our knowledge, we present the first study of the robustness of GC under conditions where features, structure, and labels can all be corrupted. Extensive experiments demonstrate the superior robustness of our proposed MRGC.
this section cite: ['b16', 'b60', 'b19', 'b40', 'b53', 'b32', 'b42', 'b52', 'b15', 'b16', 'b19', 'b19', 'b32', 'b7', 'b9', 'b69', 'b9', 'b19', 'b53', 'b40', 'b16', 'b24', 'b43', 'b35', 'b43', 'b43']

Section: Related Work
Graph Condensation. GC significantly enhances the training efficiency and scalability of GNNs [33].
Existing GC methods can be categorized into four types [54,41]: (1) Gradient Matching: GCond [33] serves as a representative framework for these methods, optimizing the condensed graph by minimizing gradient discrepancies between GNNs trained on the original and condensed graphs [32,65,15].
(2) Trajectory Matching: SFGC [74] and GEOM [72] condense graphs by aligning the training trajectories of parameter distributions in expert GNNs. (3) Distribution Matching: These methods minimize the distributional difference between the original and the condensed graphs [39,38]. (4) Others: Various methods, such as Kernel Ridge Regression [63,64] and Computation Tree [24,23], are also used for GC. However, few studies have explored GC robustness under attack.
Robust Graph Neural Network. Lines of studies have been dedicated to enhancing GNN robustness:
(1) Preprocessing: These methods leverage certain shared properties of real-world graphs to clean perturbed ones before training [12,58,30,45]. (2) Modeling: New GNN architectures are proposed to mitigate the impact of attacks dynamically during training [8,31,19,68]. (3) Training: They don't modify the GNN architecture but use training strategies like adversarial training [22] or group training [29,67] to reduce GNN sensitivity.
this section cite: ['b32', 'b53', 'b40', 'b32', 'b31', 'b64', 'b14', 'b73', 'b71', 'b38', 'b37', 'b62', 'b63', 'b23', 'b22', 'b11', 'b57', 'b29', 'b44', 'b7', 'b30', 'b18', 'b67', 'b21', 'b28', 'b66']

Section: Robust Graph Condensation.
With GC recognized as a promising technique [17,61], RobGC [16] is the first to investigate GC robustness and propose a defense against structure attacks. Benchmark [20] reveals that feature noise poses great threats in GC. However, a comprehensive understanding of GC robustness and a universal defense against structure, feature, and label attacks remains absent.
this section cite: ['b16', 'b60', 'b15', 'b19']

Section: Method
Problem Formulation. Consider a poisoned graph Ĝ={ X, Â, Ŷ} with n nodes, where X denotes node features, Â denotes adjacency matrix, and Ŷ denotes node labels. The goal is to synthesize a compact graph G ′ = X ′ , A ′ , Y ′ with n ′ ≪ n nodes, in a way that is resilient to adversarial attacks, so that GNNs trained on G ′ can still perform well on test nodes in the original graph Ĝ.
min G ′ E Φ∼P Φ [L task (g Φ (G ′ ), Ĝtest )],(1)
where g Φ (G ′ ) denotes the GNN trained on G ′ , and L task denotes the task-specific loss.
Framework. As depicted in Figure 2, we mitigate the increase in classification complexity induced by attacks and enhance GC robustness from three complementary perspectives: ❶ Reduce the intrinsic dimension (Section 3.1). We estimate and constrain the intrinsic dimension of the condensed graph during the GC process. ❷ Minimize the complexity of class boundaries (Section 3.2). By regularizing the curvature of class manifolds in the condensed graph, we achieve smoother geometric decision boundaries between classes. ❸ Resolve class ambiguity (Section 3.3). We minimize the overlapping volume between class manifolds to reduce classification ambiguity and enhance class separability.
this section cite: []

Section: Intrinsic Dimension Manifold Regularization
From Figure 1(c), we empirically observe that adversarial attacks will significantly increase the intrinsic dimension of the condensed graph. Here, we first theoretically analyze the relationship between the intrinsic dimension and the graph condensation process and then propose a differentiable method to estimate the intrinsic dimension of the condensed graph. Finally, we impose constraints on the intrinsic dimension throughout the entire condensing process.
Intrinsic dimension is the minimum number of coordinates required to describe the data [2]. Following the widely adopted manifold assumption which states that high-dimensional data lie on a Intrinsic Dimension Manifold Reg. lower-dimensional manifold, the intrinsic dimension refers to the dimensionality of this underlying manifold [3,46]. Here, we begin by presenting the following Theorem 1.
this section cite: ['b1', 'b2', 'b45']

Section: Original Graph (Attacked) Condensed Graph

this section cite: []

Section: Graph Condensation

this section cite: []

Section: Theorem 1.
Given a graph G with n nodes, let G ′ with n ′ nodes denote the much smaller synthetic graph generated through graph condensation, which is comparable to G in terms of training GNNs. We have the following:
ID(G ′ ) < ID(G), (2
)
where ID(•) denotes the intrinsic dimension of graph data.
Theorem 1 indicates that graph condensation is a graph intrinsic dimension decreasing process, synthesizing a condensed graph that lies in a data manifold with a lower dimension. The proof can be found in Appendix A. Building on this, we further propose the following Theorem 2: Theorem 2. Building on Theorem 1, let G ′ * denotes the synthetic graph generated through graph condensation, where the original graph G * is under attack. Then we have:
ID(G ′ ) < ID(G ′ * ).(3)
Theorem 2 indicates that the adversarial attack poses an increasing intrinsic dimension in the condensed graph. Details of the proof are provided in Appendix A.
Through the theoretical analysis in Theorem 1 and Theorem 2, protecting the key intrinsic dimension decreasing characteristic of GC is important for improving its robustness against attacks. In this work, we propose a novel approach to calculate the dimension of the graph manifold where the condensed graph resides and use it as a regularization term to constrain the intrinsic dimension of the condensed graph. Specifically, we use the embedding vectors after two rounds of message passing, Z ′ = (A ′ ) 2 X ′ , as the node representation, which incorporates both feature and structure information. Let M(G ′ ) denote the manifold on which the condensed graph G ′ lies, embedded in R d ′ . Then, {Z i } n ′ i=1 represents the discrete set of observations sampled from M(G ′ ). The intrinsic dimension of G ′ is defined as the dimension of M(G ′ ) under the low-dimension manifold assumption [3], denoted as dim(M(G ′ )). According to [49], dim(M(G ′ )) is given by:
dim(M(G ′ )) = d ′ i=1 M(G ′ ) ∥∇ M(G ′ ) α i (z ′ )∥dz ′ .(4)
Here, ∇ M(G ′ ) α i (z ′ ) represents the gradient of coordinate function α i on M(G ′ ) at point z ′ , where α i (z ′ ) = z ′ i . Directly solving Eq. ( 4) requires constructing the explicit function of M(G ′ ), which is challenging to derive from the discrete observations {Z ′ i } n ′ i=1 . In this work, we adopt the Laplacian approximation [69] to solve Eq. ( 4):
M(G ′ ) ∥∇ M(G ′ ) α i (z ′ )∥dz ′ ∼ |M(G ′ )|S Z ′ (α i ),(5)
where S Z ′ (α i ) = p,q exp(-
∥Z ′ p -Z ′ q ∥ 2 2 2ϵ2
)(α i (Z ′ p ) -α i (Z ′ q ))
is the graph Laplacian operator with hyperparameter ϵ, and |M(G ′ )| is the volume of the manifold. Based on the geometric interpretation of singular values [1], the volume |M(G ′ )| is proportional to the product of the singular values {σ i } d ′ i=1 of node representation matrix Z ′ , i.e., |M(G ′ )| ∝ d ′ i=1 σ i . The covariance matrix of Z ′ is defined as Σ ′ Z = 1 n (Z ′ ) ⊤ Z ′ . By the relationship between singular values and eigenvalues, the eigenvalues of Σ ′ Z (denoted by {λ i } d ′ i=1 ) are related to the singular values of Z as λ i = σ 2 i n . Thus, the volume |M(G ′ )| can be expressed as:
|M(G ′ )| ∝ det(Σ Z ′ + I), (6
)
where I is to ensure that the covariance matrix Σ Z ′ is positive definite, and det(•) denotes the determinant operation. Finally, the intrinsic dimension manifold regularization loss during graph condensation is defined as:
L dim = det(Σ Z ′ + I) • d ′ i=1 S Z ′ (α i ).(7)
this section cite: ['b2', 'b48', 'b68']

Section: Curvature-Aware Manifold Smoothing
The complexity of class boundaries, determined by the geometry of class manifolds, is a key factor in classification difficulty [44]. Attacks can increase this complexity in condensed graphs, weakening GC robustness. To quantify it, we measure the Gaussian curvature [51], where larger absolute values indicate more intricate boundaries. Each node's curvature is computed, and the weighted sum of absolute values is used as a regularization term. Weights reflect each node's influence in message passing and are derived from Ricci curvature [56].
Let M(G ′ c ) denote the class-c manifold containing nodes with label c, represented by
{Z ′ i | Y ′ i = c}.
Our goal is to compute the Gaussian curvature of each node on M(Gc ′ ). We estimate the curvature at node i by fitting a quadratic hypersurface f Θ (o) = o ⊤ Θo to its local neighborhood, where Θ is the surface parameter. The curvature is obtained as the determinant of the Hessian of f Θ (ξ) [47]. Specifically, we project neighbors of node i onto its tangent space, and use these projections as inputs to fit the hypersurface. The targets are the corresponding projections onto the normal vector at node i, capturing how neighbors deviate from the tangent space and thus reflecting local manifold geometry. We estimate the normal vector u i at node i using its k-nearest neighbors in Euclidean space [4]:
min ui k j=1 (Z j i -c i ) ⊤ u i 2 , subject to u ⊤ i u i = 1,(8)
where Z j i denotes the node representation of j-th neighborhood and c i = 1 k k j=1 Z j i is the center of k neighborhoods. To solve Eq. ( 8), we define the Lagrangian function:
L(u i , λ) = k j=1 ((Z j i -c i )u i ) 2 -λ(u ⊤ i u i -1),(9)
Define Y = Z i -c i 1 ⊤ . Solving the Karush-Kuhn-Tucker conditions for Eq. ( 9) shows that the normal vector u i corresponds to the eigenvector associated with the smallest eigenvalue of Y ⊤ Y; details are provided in Appendix A. Let {λ 1 , . . . , λ d ′ } and {ξ 1 , . . . , ξ d ′ } be the eigenvalues and corresponding eigenvectors of Y ⊤ Y, sorted in descending order. Since Y ⊤ Y is symmetric and positive semidefinite, we have λ 1 ≥ • • • ≥ λ d ′ ≥ 0 and ξ ⊤ a ξ b = 0 for all a ̸ = b. The (d ′ -1)dimensional tangent space at node i is spanned by ⟨ξ 1 , . . . , ξ d ′ -1 ⟩, and the projections of i's k neighbors onto this space form the matrix O i ∈ R k×(d ′ -1) , defined as:
O i = [o 1 , o 2 , . . . , o k ] ⊤ ,(10)
where each o j ∈ R d ′ -1 corresponds to the projection of (Z j i -c i ) onto the tangent space:
o j = [(Z j i -c i ) • ξ 1 , . . . , (Z j i -c i ) • ξ d ′ -1 ].(11)
Then we propose the following proposition:
Proposition 1. By fitting the quadratic hypersurface f Θ (o) via min Θ k j=1 ( 1 2 o ⊤ j Θo j -t j ) 2 , where t j =(Z j i -Z i )•u i represents the projection along the normal vector, the Gaussian curvature of the class manifold at node i is given by K(i)=2det(Mat(Q -1 p)). Here Q∈R (d ′ -1) 2 ×(d ′ -1) 2 is a fourth-order tensor expressed as a matrix with entries Q a,b,c,d = k j=1 o ja o jb o jc o jd , and p ∈ R (d ′ -1) 2 is a second-order tensor with entries p a,b = k j=1 t j o ja o jb . The operator Mat(•) reshapes Q -1 p into an (d ′ -1) × (d ′ -1) matrix, and det(•) denotes the determinant operation.
Proposition 1 provides a closed-form expression for the Gaussian curvature at node i, with the proof deferred to Appendix A. However, averaging curvature across all nodes ignores the varying structural roles of individual nodes. In particular, nodes at community boundaries, which serve as bridges for inter-community message passing, have a greater influence on the geometric complexity of class boundaries. To account for this, we reweight each node's Gaussian curvature using Ricci curvature [56], an edge-based metric that reflects structural connectivity. Lower Ricci curvature values on edges indicate stronger bridging roles, highlighting the corresponding node's importance. We adopt the Ollivier definition [48], where the Ricci curvature between nodes (i, j) is defined as κ(i, j) = 1 -W(m α i , m α i )/D(i, j), where W(•, •) is the Wasserstein distance of order 1, D(•, •) denotes the shortest-path distance, and m α u represents the mass distribution, which is defined as:
m α i (j) =    α, if j = i, (1 -α) Aij deg(i) , if j ∈ N (i), 0, otherwise,(12)
where N (i) denotes the neighbors of node i, deg(i) = j∈N (i) A ij , and α is the smoothing parameter and is typically set to 0.5. The strategy for measuring the Ricci curvature of node i is to average the curvatures of its connected edges [9], expressed as κ(i) = Aij deg(i)
j∈N (i) κ(i, j).
Finally, the Gaussian curvature regularization term, denoted as L cur , is defined as:
L cur = c i∈Vc Norm(-κ(i)) • |K(i)|,(13)
where Norm(•) denotes min-max normalization to [0, 1], and V c denotes nodes belonging to class c.
this section cite: ['b43', 'b50', 'b55', 'b46', 'b3', 'b55', 'b47', 'b8']

Section: Class-Wise Manifold Decoupling
Class ambiguity is the third critical factor that contributes to the classification complexity [25,44].
To preserve the classification complexity reduction property of GC and improve its robustness, it is essential to avoid class ambiguity in the condensed graph, ensuring that the classes remain well-separated with clear decision boundaries. To avoid the class ambiguity arising in condensed graphs under attacks, we propose measuring the overlap between class manifolds by calculating the difference between the sum of the volumes of individual class manifolds and the volume of the entire data manifold. Minimizing this difference defines our class-wise manifold decoupling objective, which can be expressed as follows:
L sep = c |M(G ′ c )| -|M(G ′ )| 2 , (14
)
where |M(G ′ c )| represents the volume of class-c manifold. This approach facilitates sufficient decoupling between class manifolds to mitigate class ambiguity, thereby effectively reducing classification complexity in the condensed graph.
Training Pipeline. We initialize node features in the condensed graph by randomly selecting nonoutlier nodes from the original graph, where outliers are identified based on the Euclidean distances of their features. Other initialization follows [33]. The training loss is as follows:
L = L GC + αL dim + βL cur + γL sep ,(15)
where L GC denotes the loss function of GC backbone, as MRGC is a plug-and-play framework, and α, β, γ are hyperparameters. Detailed pipeline are in Appendix D.
this section cite: ['b24', 'b43', 'b32']

Section: Complexity Analysis.
The Intrinsic Dimension Manifold Regularization Module requires O(n ′ d ′ + (d ′ ) 3 ) operations. The Curvature-Aware Manifold Smoothing consists of two parts: Gaussian curvature computation, with a complexity of O(n ′ ((d ′ ) 6 + k)), and Ricci curvature computation, which requires O((n ′ )foot_0 ) operations. The Class-Wise Manifold Decoupling Module has a complexity of O(c(d ′ ) 3 ), where c denotes the number of classes. It is worth noting that the practical computational cost remains efficient because n ′ , the number of nodes in the condensed graph, is small by design. Additionally, we apply PCA [11] to reduce the feature dimensionality before computing the regularization terms, ensuring that d ′ stays manageable. Details are provided in Appendix D.
this section cite: ['b10']

Section: Experiments

this section cite: []

Section: Experimental Settings
Datasets. We evaluate MRGC 2 and the baselines on five real-world node classification datasets in a transductive setting: Cora [66], CiteSeer [66], PubMed [66], DBLP [6], and Ogbn-arxiv [27]. The data split configuration follows that of [20] for the Cora, CiteSeer, PubMed, and Ogbn-arxiv datasets.
For the DBLP dataset, we use the settings from [28], performing random splits with 20 labeled nodes per class for training, 30 per class for validation, and the remaining nodes for testing.
Attacks. For the poisoning attacks, we use the widely adopted PRBCD [18,53] for structure perturbation. For feature perturbation, we randomly select nodes and assign their features by sampling from a normal distribution. For label perturbation, we randomly select a subset of nodes and uniformly flip their labels to other classes. The attack budget is set to p percent of the total number of edges for structure perturbation and p percent of the number of training nodes for feature and label perturbation.
Baselines. We evaluate MRGC against various baseline approaches, including five state-of-the-art graph condensation methods: GCond [33], SGDD [65], SFGC [74], GEOM [72] and GCDM [37].
We also compare with RobGC [16], the first robust graph condensation method specifically designed for defending against structure attacks. Furthermore, following [16], we enhance our comparison by incorporating three strong graph denoising techniques as preprocessing steps for GCond: GCond(+J), which removes edges based on Jaccard similarity; GCond(+S), which uses Singular Value Decomposition (SVD) for low-rank approximation to mitigate high-rank noise; and GCond(+K), which integrates k-nearest neighbors based on feature similarity into the original graph with k = 3.
this section cite: ['b65', 'b65', 'b65', 'b5', 'b26', 'b19', 'b27', 'b17', 'b52', 'b32', 'b64', 'b73', 'b71', 'b36', 'b15', 'b15']

Section: Implement Details.
In this experiment, we use GCond [33] as the backbone of MRGC, which is a gradient-matching-based GC method. However, it is important to note that MRGC is compatible with most existing GC methods. The hyperparameters α, β, and γ are determined through a grid search from 1e-3 to 1e2 with logarithmic steps of 5. Details can be found in Appendix C and our code. We repeat all the experiments five times and report the average performance and standard deviation. All the experiments are conducted in a single NVIDIA GeForce RTX 3090 24GB GPU.
this section cite: ['b32']

Section: Robustness Across Varying Condensation Ratios
In this section, we evaluate the impact of different condensation ratios on the robustness of MRGC across the five aforementioned datasets, using three distinct condensation ratios while keeping the attack budgets invariant. The attack budgets for structure, feature, and label attacks are set to 1%, 10%, and 20% for the Cora, CiteSeer, and Ogbn-Arxiv datasets, and 0.1%, 10%, and 20% for the PubMed and DBLP datasets. The results are presented in Table 1.
From the results in Table 1, we have the following three observations: (1) Across all the datasets and condensation ratios except for Ogbn-arxiv at ratios of 0.05% and 0.50%, MRGC achieves the best performance under poisoning attacks. This highlights the robustness of our proposed method in mitigating the negative impact of attacks on the graph condensation process. (2) On the Ogbn-arxiv dataset at condensation ratios of 0.25% and 0.50%, GEOM demonstrates superior performance. This is because trajectory-matching GC methods, such as SFGC and GEOM, achieve significantly better results on the clean Ogbn-Arxiv dataset than gradient-matching GC methods [41,20,54]. Consequently, despite performance degradation under attack, they retain a relative advantage due to their initially superior performance on the clean dataset. However, SFGC and GEOM exhibit poor robustness on other datasets. (3) Denoising the graph during the preprocessing stage before GC has limited effectiveness, as these methods assume that the node features and labels are clean.
this section cite: ['b40', 'b19', 'b53']

Section: Robustness Across Varying Attack Budgets
To evaluate the robustness of MRGC under varying attack budgets, we fix the condensation ratio to the lowest value for each dataset and adjust the attack budgets for structure, feature, and label attacks independently, while ensuring that the other attack budgets remain consistent with the settings outlined in Section 4.2. The experiments are conducted on the Cora, CiteSeer, PubMed, and DBLP datasets, and the results are shown in Table 2.
As shown in Table 2: (1) MRGC consistently outperforms all baselines across all datasets and attack budget variations. For example, on the CiteSeer dataset, MRGC achieves improvements of approximately 3.98% and 4.75% over the runner-up with label perturbation ratios of 30% and 40%, respectively. This highlights the robustness of MRGC against varying attack intensities. (2) The robustness of MRGC remains stable regardless of the type of attack, effectively defending against structure, feature, and label perturbations. In contrast, RobGC performs well under the structure attacks but shows performance degradation as the intensity of feature and label attacks increases.
(3) Gradient matching-based GC methods generally exhibit better robustness compared to trajectory matching and distribution matching GC methods.
this section cite: []

Section: Ablation Study
To verify the effectiveness of each component, we compare different ablated versions of MRGC on Cora and CiteSeer datasets with the lowest condensation ratio: (1) MRGC (w/o ID), ( 2) MRGC (w/o C), and (3) MRGC (w/o D), which respectively remove the Intrinsic Dimension Manifold Regularization, Curvature-Aware Manifold Smoothing, and Class-Wise Manifold Decoupling modules. As the results are shown in Figure 3, all three modules contribute to the performance of MRGC, with the Intrinsic Dimension Manifold Regularization module providing the greatest improvement.
this section cite: []

Section: Classification Complexity Study
Based on our analysis in this work, graph condensation acts as a process that reduces classification complexity, whereas attacks disrupt this property, leading to an increase in the classification complexity of the condensed graph. This study aims to verify the ability of our proposed MRGC to mitigate this negative impact. As introduced in Section 1, we measure classification complexity using three   &RUD &LWH6HHU ,' &RUD &LWH6HHU )'5 &RUD &LWH6HHU )+& 2ULJLQ *&RQG *&RQG 05*& Figure 4: Classification complexity("*" indicates attack). H H H H H H $FFXUDF\ &RUD &LWH6HHU H H H H H H $FFXUDF\ &RUD &LWH6HHU H H H H H H $FFXUDF\ &RUD &LWH6HHU Figure 5: Hyperparameters study. widely adopted metrics: Intrinsic Dimension (ID), Fisher's Discriminant Ratio (FDR), and Fraction of Hyperspheres Covering Data (FHC). Details of these metrics can be found in Appendix C. The experiments are conducted in Cora (2.60%) and CiteSeer (1.80%) and the results are presented in Figure 4. We can see that our proposed MRGC effectively preserves the classification-complexity-reducing property of GC, contributing to the achievement of robust graph condensation.
this section cite: []

Section: Hyperparameter Sensitivity Study
In this section, we explore the sensitivity of the hyperparameters α, β, and γ for MRGC. In the experiments, we vary the values of α, β, and γ on the Cora and CiteSeer datasets with the lowest condensation ratio to examine their impact on model performance. The results are shown in Figure 4.5.
As we can see, MRGC performs better when appropriate values are chosen for all hyperparameters, and a wide range of hyperparameters can still yield satisfactory results.
this section cite: []

Section: Conclusion
In this work, we explore GC's robustness against adversarial attacks on features, structures, and labels. Through empirical and theoretical analysis, we discover that GC functions as an intrinsicdimension-reducing mechanism that creates graphs with lower classification complexity, while this property is susceptible to adversarial attacks. To protect this critical characteristic and improve the robustness of GC, we adopt the geometric perspective of the graph data manifold and propose MRGC, a novel manifold-constrained robust graph condensation framework. Specifically, we introduce three modules that constrain the intrinsic dimension, manifold curvature, and class manifold overlap of the condensed graph, thereby maintaining the classification-complexity-reducing property. Experiments demonstrate MRGC's effectiveness against universal attacks. One limitation is that our focus node classification task, with the graph classification task left for our future work.
this section cite: []

Section: References
Ref_id:b0 Title: Linear algebra and optimization for machine learning Year: (2020)
Ref_id:b1 Title: Estimating local intrinsic dimensionality Year: (2015)
Ref_id:b2 Title: Intrinsic dimension of data representations in deep neural networks Year: (2019)
Ref_id:b3 Title: Curvature of point clouds through principal component analysis Year: (2021)
Ref_id:b4 Title: The intrinsic dimensionality of signal collections Year: (1969)
Ref_id:b5 Title: Deep gaussian embedding of graphs: Unsupervised inductive learning via ranking Year: (2017)
Ref_id:b6 Title: Intrinsic dimension estimation: Relevant techniques and a benchmark framework Year: (2015)
Ref_id:b7 Title: Understanding structural vulnerability in graph convolutional networks Year: (2021)
Ref_id:b8 Title: Ollivier-ricci curvature for hypergraphs: A unified framework Year: (2022)
Ref_id:b9 Title: A comprehensive survey on trustworthy graph neural networks: Privacy, robustness, fairness, and explainability Year: (2024)
Ref_id:b10 Title: Principal components analysis Year: (1989)
Ref_id:b11 Title: All you need is low (rank) defending against adversarial attacks on graphs Year: (2020)
Ref_id:b12 Title: Testing the manifold hypothesis Year: (2016)
Ref_id:b13 Title: Graph adversarial training: Dynamically regularizing based on graph structure Year: (2019)
Ref_id:b14 Title: Multiple sparse graphs condensation Year: (2023)
Ref_id:b15 Title: Robgc: Towards robust graph condensation Year: (2024)
Ref_id:b16 Title: Graph condensation: A survey Year: (2024)
Ref_id:b17 Title: Robustness of graph neural networks at scale. NeurIPS Year: (2021)
Ref_id:b18 Title: Reliable graph neural networks via robust aggregation Year: (2020)
Ref_id:b19 Title: Gc4nc: A benchmark framework for graph condensation on node classification with new insights Year: (2024)
Ref_id:b20 Title: Blessing of dimensionality: mathematical foundations of the statistical physics of data Year: (2018)
Ref_id:b21 Title: Adversarial training for graph neural networks: Pitfalls, solutions, and new directions Year: (2024)
Ref_id:b22 Title: Bonsai: Gradient-free graph distillation for node classification Year: (2025)
Ref_id:b23 Title: Mirage: Model-agnostic graph distillation for graph classification Year: (2023)
Ref_id:b24 Title: Complexity measures of supervised classification problems Year: (2002)
Ref_id:b25 Title: Estimating the intrinsic dimensionality using normalizing flows Year: (2022)
Ref_id:b26 Title: Open graph benchmark: Datasets for machine learning on graphs Year: (2020)
Ref_id:b27 Title: Scaling up graph neural networks via graph coarsening Year: (2021)
Ref_id:b28 Title: Self-guided robust graph structure refinement Year: (2024)
Ref_id:b29 Title: Node similarity preserving graph convolutional networks Year: (2021)
Ref_id:b30 Title: Graph structure learning for robust graph neural networks Year: (2020)
Ref_id:b31 Title: Condensing graphs via one-step gradient matching Year: (2022)
Ref_id:b32 Title: Graph condensation for graph neural networks Year: (2021)
Ref_id:b33 Title: Maximum likelihood estimation of intrinsic dimension Year: (2004)
Ref_id:b34 Title: Spectral adversarial training for robust graph neural network Year: (2022)
Ref_id:b35 Title: An introduction to Kolmogorov complexity and its applications Year: (2008)
Ref_id:b36 Title: Graph condensation via receptive field distribution matching Year: (2022)
Ref_id:b37 Title: Graph condensation via eigenbasis matching Year: (2023)
Ref_id:b38 Title: Dream: Efficient dataset distillation by representative matching Year: (2023)
Ref_id:b39 Title: Cat: Balanced continual graph learning with graph condensation Year: (2023)
Ref_id:b40 Title: Gcondenser: Benchmarking graph condensation Year: (2024)
Ref_id:b41 Title: Puma: Efficient continual graph learning with graph condensation Year: (2023)
Ref_id:b42 Title: Puma: Efficient continual graph learning for node classification with graph condensation Year: (2024)
Ref_id:b43 Title: How complex is your classification problem? a survey on measuring classification complexity Year: (2019)
Ref_id:b44 Title: Robust graph learning against adversarial evasion attacks via prior-free diffusion-based structure purification Year: (2025)
Ref_id:b45 Title: Unveiling and mitigating generalized biases of dnns through the intrinsic dimensions of perceptual manifolds Year: (2024)
Ref_id:b46 Title: Curvature-balanced feature manifold learning for long-tailed classification Year: (2023)
Ref_id:b47 Title: A survey of ricci curvature for metric spaces and markov chains Year: (2010)
Ref_id:b48 Title: Low dimensional manifold model for image processing Year: (2017)
Ref_id:b49 Title: Geometric differentiation: for the intelligence of curves and surfaces Year: (2001)
Ref_id:b50 Title: Detection of zones of abnormal strains in structures using gaussian curvature analysis Year: (1994)
Ref_id:b51 Title: Statistical approaches to combining binary classifiers for multi-class classification Year: (2011)
Ref_id:b52 Title: Adversarial attack and defense on graph data: A survey Year: (2022)
Ref_id:b53 Title: Gc-bench: An open and unified benchmark for graph condensation Year: (2024)
Ref_id:b54 Title: Relative intrinsic dimensionality is intrinsic to learning Year: (2023)
Ref_id:b55 Title: Understanding over-squashing and bottlenecks on graphs via curvature Year: (2021)
Ref_id:b56 Title: Fast graph condensation with structure-based neural tangent kernel Year: (2024)
Ref_id:b57 Title: Adversarial examples on graph data: Deep insights into attack and defense Year: (2019)
Ref_id:b58 Title: Understanding the impact of graph reduction on adversarial robustness in graph neural networks Year: (2024)
Ref_id:b59 Title: Graph information bottleneck Year: (2020)
Ref_id:b60 Title: A survey on graph condensation Year: (2024)
Ref_id:b61 Title: Topology attack and defense for graph neural networks: An optimization perspective Year: (2019)
Ref_id:b62 Title: Kernel ridge regression-based graph dataset distillation Year: (2023)
Ref_id:b63 Title: St-gcond: Self-supervised and transferable graph dataset condensation Year: (2025)
Ref_id:b64 Title: Does graph distillation see like vision dataset counterpart? Year: (2023)
Ref_id:b65 Title: Revisiting semi-supervised learning with graph embeddings Year: (2016)
Ref_id:b66 Title: Environment-aware dynamic graph learning for out-of-distribution generalization Year: (2023)
Ref_id:b67 Title: DG-Mamba: Robust and efficient dynamic graph structure learning with selective state space models Year: (2025)
Ref_id:b68 Title: 3d point cloud denoising using graph laplacian regularization of a low dimensional manifold model Year: (2019)
Ref_id:b69 Title: Subgraph federated learning with missing neighbor generation Year: (2021)
Ref_id:b70 Title: Gnnguard: Defending graph neural networks against adversarial attacks Year: (2020)
Ref_id:b71 Title: Navigating complexity: Toward lossless graph condensation via expanding window matching Year: (2024)
Ref_id:b72 Title: Adversarial robustness in graph neural networks: A hamiltonian approach Year: (2024)
Ref_id:b73 Title: Quoc Viet Hung Nguyen, Xingquan Zhu, and Shirui Pan. Structure-free graph condensation: From large-scale graphs to condensed graph-free data Year: (2024)
Ref_id:b74 Title: Robust graph convolutional networks against adversarial attacks Year: (2019)
