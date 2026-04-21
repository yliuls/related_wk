Title: Principled Data Augmentation for Learning to Solve Quadratic Programming Problems
Abstract: Linear and quadratic optimization are crucial in numerous real-world applications, ranging from training machine learning models to solving integer linear programs. Recently, learning-to-optimize methods (L2O) for linear (LPs) or quadratic programs (QPs) using message-passing graph neural networks (MPNNs) have gained traction, promising lightweight, data-driven proxies for solving such optimization problems. For example, they replace the costly computation of strong branching scores in branch-and-bound solvers, thereby reducing the need to solve many such optimization problems. However, robust L2O MPNNs remain challenging in data-scarce settings, especially when addressing complex optimization problems such as QPs. This work introduces a principled approach to data augmentation tailored for QPs via MPNNs. Our method leverages theoretically justified data augmentation techniques to generate diverse yet optimality-preserving instances. Furthermore, we integrate these augmentations into a self-supervised contrastive learning framework, thereby pretraining MPNNs for improved performance on L2O tasks. Extensive experiments demonstrate that our approach improves generalization in supervised scenarios and facilitates effective transfer learning to related optimization problems.

Section: Introduction
Linear and quadratic optimization problems are fundamental problems in machine learning, operations research, and scientific computing [Boyd andVandenberghe, 2004, Nocedal andWright, 1999]. Many real-world applications, such as resource allocation, logistics, and training machine learning models, rely on efficiently solving large-scale linear programming (LPs) and quadratic programming (QPs). In addition, they play a key role in state-of-the-art integer-linear optimization solvers, allowing the computation of lower bounds, and are the basis for crucial heuristics such as strong branching for variable selection [Achterberg et al., 2005].
Recently, machine learning techniques, particularly message-passing graph neural networks (MPNNs) [Gilmer et al., 2017, Scarselli et al., 2008] have been explored for learning-to-optimize (L2O) approaches, aiming to learn to solve LPs and QPs in a data-driven fashion [Bengio et al., 2021, Cappart et al., 2023], enhancing solver efficiency and improving generalization across different problem instances. For example, Gasse et al. [2019] used MPNNs to imitate the costly strong branching score for variable selection in integer-linear optimization solvers, which requires solving many LPs during the solving process. However, training robust models for L2O remains challenging due to the scarcity of labeled data, especially for complex optimization formulations such as QPs.
In response to this challenge, self-supervised learning (SSL) has emerged as a powerful paradigm for pretraining models on large, unlabeled datasets [Liu et al., 2021]. Contrastive learning, a key approach in SSL, has demonstrated significant success in graph-based tasks by leveraging data augmentation techniques to create diverse training examples You et al. [2020]. Despite advances in contrastive learning and L2O with MPNNs, a gap exists in integrating these two paradigms for LPs and QPs. Additionally, designing effective and principled data augmentation strategies for LPs and QPs remains non-trivial due to the structural constraints and optimality conditions inherent in these problems.
Present work In this work, we propose a principled approach to data augmentation for LPs and QPs, specifically designed to enhance the performance of MPNN-based L2O methods for such problems; see Fig. 1 for a high-level overview. Concretely, our contributions are as follows:
1. We introduce a set of novel and theoretically grounded augmentation techniques for LPs and QPs that preserve optimality while generating diverse training instances; 2. We apply these augmentations in both supervised and self-supervised settings, including contrastive pretraining of MPNNs to enhance downstream performance; 3. We empirically evaluate our approach on synthetic and benchmark datasets, showing that pretraining with our augmentations improves generalization and transferability across problem classes.
By bridging the gap between data augmentation and L2O for LPs and QPs, our work offers a new perspective on enhancing neural solvers through supervised learning and self-supervised pretraining.
The proposed augmentations improve generalization in data-scarce settings, enabling more efficient and robust learning-based optimization methods.
this section cite: ['b6', 'b49', 'b0', 'b25', 'b58', 'b3', 'b9', 'b21', 'b46', 'b77']

Section: Related work
This section reviews relevant literature on L2O, graph data augmentation, graph contrastive learning, and synthetic instance generation for LP and MILP problems.
MPNN and L2O Message Passing Neural Networks (MPNNs) [Gilmer et al., 2017, Scarselli et al., 2008] have been extensively studied and are broadly categorized into spatial and spectral variants. Spatial MPNNs [Bresson and Laurent, 2017, Duvenaud et al., 2015, Hamilton et al., 2017, Veličković et al., 2017, Xu et al., 2018] follow the message-passing paradigm introduced by Gilmer et al. [2017].
MPNNs have shown strong potential in learning to optimize (L2O). A widely adopted approach represents MILPs using constraint-variable bipartite graphs [Chen et al., 2022, Ding et al., 2020, Gasse et al., 2019, Khalil et al., 2022, Qian et al., 2024a]. Recent work has also aligned MPNNs with various optimization algorithms, including interior-point methods (IPMs) [Qian et al., 2024a, Qian andMorris, 2025], primal-dual hybrid gradient (PDHG) [Li et al., 2024a], and distributed algorithms [Li et al., 2024b[Li et al., , 2025]]. From a theoretical perspective, several studies have analyzed the expressiveness of MPNNs in approximating solutions to linear and quadratic programming [Chen et al., 2022, 2023, 2024b,a, Wu et al., 2024].
this section cite: ['b25', 'b58', 'b7', 'b18', 'b29', 'b64', 'b71', 'b25', 'b11', 'b21', 'b37', 'b51', 'b39', 'b73', 'b41', 'b11', 'b73']

Section: Graph data augmentation
Graph data augmentation is central to learning-based optimization, especially under data scarcity. Common strategies include feature-wise perturbations [You et al., 2020, Zhu et al., 2020, Suresh et al., 2021, Zhu et al., 2021, Thakoor et al., 2021, Bielak et al., 2022, Hu et al., 2023], structure-wise modifications such as node dropping and edge perturbation [Rong et al., 2019, Hassani and Khasahmadi, 2020, Zhu et al., 2020, Qiu et al., 2020, Papp et al., 2021, Zhu et al., 2021, Bielak et al., 2022, Hu et al., 2023], and spectral-domain augmentations [Yang et al., 2024a, Liu et al., 2022a, Lin et al., 2022, Wan et al., 2024], the latter aligning with recent works like S3GCL. Learning-based approaches [Yang et al., 2020, Zhao et al., 2021, Liu et al., 2022b, Yin et al., 2022, You et al., 2021, Wu et al., 2023] further enable adaptive augmentation via trainable modules. Complementary perspectives include graph rewiring [Topping et al., 2021, Karhadkar et al., 2022, Arnaiz-Rodríguez et al., 2022, Qian et al., 2023, Gutteridge et al., 2023, Barbero et al., 2023, Qian et al., 2024b] and structure learning [Jin et al., 2020, Liu et al., 2022c, Zou et al., 2023, Zhou et al., 2023, Fatemi et al., 2023], both of which can be viewed as task-specific augmentation techniques. For a broader overview, including robustness and self-supervised settings, see [Zhao et al., 2022, Ding et al., 2022].
Graph contrastive learning Graph augmentations also play a central role in graph self-supervised learning (SSL), which aims to learn transferable representations without labeled supervision. Graph SSL methods are categorized into contrastive, generative, and predictive approaches [Wu et al., 2021].
Q 1 2 (x1 . . . xn)    x1 . . . xn    + c ⊤    x1 . . . xn    s.t. A11x1 A12x2
. . . A1nxn ≤ b1 A21x1 A22x2 . . . A2nxn ≤ b2 . . . . . . . . . An1x1 An2x2 . . . Annxn ≤ bn 0.4x 0.3x 0.7x Supervised learning MSE Self-supervised learning NT-Xent Predict LP/QP solution Strong branching score
Figure 1: Overview of our framework for principled data augmentation for quadratic optimization problems. Given a QP instance, we apply transformations (e.g., adding/removing/scaling variables or constraints; see Section 2.2) to generate new instances, thereby augmenting the training dataset. We then use standard supervised learning or contrastive learning to train an MPNN.
We focus on contrastive ones, which operate at three levels: global-to-global (G2G), local-to-local (L2L), and local-to-global (L2G). G2G methods generate contrasting views of entire graphs, as in GraphCL [You et al., 2020] with structural perturbations, extended by CSSL [Zeng and Xie, 2021], GCC [Qiu et al., 2020] using random walks, AutoGCL [Yin et al., 2022] with learned augmentations, and AD-GCL [Suresh et al., 2021] with an adversarial objective. The only contrastive method targeting LPs is Li et al. [2024c], which adopts a CLIP-style [Radford et al., 2021] formulation. L2L methods contrast node-level views, e.g., GRACE [Zhu et al., 2020], GCA [Zhu et al., 2021], Graph Barlow Twins [Bielak et al., 2022], BGRL [Thakoor et al., 2021], and REGCL [Ji et al., 2024], with S3GCL [Wan et al., 2024] introducing spectral views. L2G methods such as DGI [Veličković et al., 2018], MVGRL [Hassani and Khasahmadi, 2020], and InfoGraph [Sun et al., 2019] contrast local and global representations to maximize mutual information.
While general-purpose augmentations are well-studied, few works target L2O problems. Duan et al. [2022] designs satisfiability-preserving augmentations for SAT problems, and Huang et al. [2023] samples neighborhoods around expert solutions. A concurrent work [Zeng et al., 2025] explores constraint permutations in MILPs. Compared with these works, we propose efficient, principled transformations, tailored for supervised and contrastive learning in linear and quadratic programming.
this section cite: ['b77', 'b86', 'b61', 'b87', 'b62', 'b4', 'b31', 'b57', 'b30', 'b86', 'b55', 'b50', 'b87', 'b4', 'b31', 'b72', 'b45', 'b43', 'b66', 'b75', 'b83', 'b47', 'b63', 'b36', 'b1', 'b52', 'b28', 'b2', 'b34', 'b48', 'b88', 'b19', 'b84', 'b16', 'b70', 'b77', 'b79', 'b55', 'b76', 'b61', 'b42', 'b86', 'b87', 'b4', 'b62', 'b33', 'b66', 'b65', 'b30', 'b60', 'b17', 'b32', 'b80']

Section: Instance generation
While random instance generators exist for LPs and MILPs [Gasse et al., 2019[Gasse et al., , 2022]], data scarcity has driven model-based approaches, including stress testing [Bowly, 2019], VAEs [Geng et al., 2023], and diffusion models [Zhang et al., 2024]. Other works reconstruct or adapt instances [Wang et al., 2023, Guo et al., 2024, Yang et al., 2024c], or generate them from code or substructures [Li et al., 2024c, Liu et al., 2024]. In contrast, our approach is model-free, mathematically principled, and generates new instances through transformations with analytically tractable solutions.
this section cite: ['b21', 'b22', 'b5', 'b24', 'b44', 'b67', 'b74', 'b42']

Section: Background
We introduce notations and define MPNNs, LPs, and QPs in the following.
Notations Let N := {0, 1, 2, . . . }. For n ≥ 1, let [n] := {1, . . . , n} ⊂ N. We use { {. . . } } to denote multisets, i.e., the generalization of sets allowing for multiple instances of each of its elements.
A graph G is a pair (V (G), E(G)) with finite sets of vertices or nodes V (G) and edges E(G) ⊆ {{u, v} ⊆ V (G) × V (G) | u ̸ = v}. For ease of notation, we denote the edge {u, v} in E(G) by (u, v) or (v, u). The neighborhood of a node v is denoted by N (v) = {u ∈ V | (v, u) ∈ E}, and its degree is |N (v)|.
An attributed graph augments each node v ∈ V with a feature vector σ(v) ∈ R d , yielding a node feature matrix H ∈ R n×d where H v = σ(v). The adjacency matrix of G is denoted A ∈ {0, 1} n×n , with A ij = 1 if and only if (i, j) ∈ E. Vectors x ∈ R d are column vectors by default.
LCQP In this work, we focus on special QPs, namely linearly-constrained quadratic programming (LCQP), of the following form,
min x∈R n 1 2 x ⊺ Qx + c ⊺ x s.t. Ax ≤ b.(1)
Here, an LCQP instance I is a tuple (Q, A, b, c), where Q ∈ R n×n and c ∈ R n are the quadratic and linear coefficients of the objective, A ∈ R m×n , and b ∈ R m form the inequality constraints.
Note that the bounds of variables, e.g., x ≥ 0, can also be merged into the constraints. We assume that the smmetric and quadratic matrix Q is positive definite (PD), denoted as Q ≻ 0, so that the problem is convex and has a unique solution. The optimal solution x * is a feasible solution such that
1 2 x ⊺ Qx + c ⊺ x ≥ 1 2 x * ⊺ Qx * + c ⊺ x *
, for any feasible x. The corresponding dual variables for the constraints are denoted as λ * . 1 By setting the matrix Q to a zero matrix, we arrive at LPs,
min x∈R n c ⊺ x s.t. Ax ≤ b.(2)
Considering the QP of Eq. ( 1), the optimal primal-dual solutions must satisfy the Karush-Kuhn-Tucker (KKT) conditions [Nocedal and Wright, 1999, p. 321],
Qx * + A ⊺ λ * + c = 0 (3a) Ax * ≤ b (3b) λ * ≥ 0 (3c) λ * i (A i x * -b i ) = 0.(3d)
Define slack variables s * := b -Ax * , the inequality constraints become equalities Ax * = bs * . The KKT conditions can then be compactly written as,
Q A ⊺ A 0 x * λ * = -c b -s * ,(4)
where the inequality and complementarity conditions Eqs. MPNNs for LCQPs Representing LPs and QPs with MPNNs has been explored in prior work. For example, Chen et al. [2022] models LPs using a bipartite constraint-variable graph, and Chen et al.
[2024a] extends this to QPs by adding edges between variable nodes. We adopt the setting of Chen et al. [2024a]. Given an LCQP instance I, we construct a graph G(I) with constraint nodes C(I) and variable nodes V (I). Edges between C(I) and V (I) are defined by nonzero entries of A with weights A cv , for v ∈ V (I), c ∈ C(I); and edges between variables are defined by nonzero Q vu , v, u ∈ V (I).
Node features are H c := reshape(b) ∈ R m×1 for constraint nodes and H v := reshape(c) ∈ R n×1 for variable nodes. MPNNs learn a vectorial representation of each node in a graph by aggregating information from its neighbors, i.e.,
h (t) c := UPD (t) c h (t-1) c , v∈N (c)∩V (I) A cv h (t-1) v h (t) v := UPD (t) v h (t-1) v , u∈N (v)∩V (I) Q uv h (t-1) u , c∈N (v)∩C(I) A cv h (t) c ,(5)
followed by a pooling function and a readout function to predict the objective,
z I := POOL v∈V (I) h (T ) v , c∈C(I) h (T ) c ; obj(I) := READOUT z I .(6)
this section cite: ['b11', 'b13']

Section: Principled data augmentation for LCQPs
We propose a set of principled transformations for LCQPs as data augmentation. Let I denote a set of LCQP instances. We consider transformations T : I → I that perturb the problem structure while allowing efficient computation of optimal solutions through simple linear-algebraic operations, without requiring a solver. Our core idea is to construct affine transformations of the KKT system Eq. ( 4) that yield valid KKT conditions for a new problem. We notice that applying a linear mapping M on Eq. ( 4) does not change the equality, i.e.,
M Q A ⊺ A 0 x * λ * = M -c b -s *
holds for all choices of transformation matrix M . To make the transformed problem valid, we need a M ⊺ on the right and further require a right (pseudo) inverse (M ⊺ )
† with M ⊺ (M ⊺ ) † = I. To ensure (M ⊺ ) † exists, the matrix M needs to be full column rank. Now we have
M Q A ⊺ A 0 M ⊺ (M ⊺ ) † x * λ * = M -c b -s * .
Finally, we could add a bias term B on both sides, and arrive at the general form,
M Q A ⊺ A 0 M ⊺ + B (M ⊺ ) † x * λ * = M -c b -s * + β.(7)
Here, the matrices M and B define the transformation on the problem parameters (Q, A, b, c), with
B chosen to satisfy B (M ⊺ ) † x * λ * = β.
The quantity (M ⊺ ) † x * λ * , if it exists, recovers the optimal solution of the transformed problem. Thus, we can compute the new solutions using only matrix operations, without solving the transformed LCQP.
this section cite: []

Section: A framework for valid transformations
An ideal transformation should be expressive, valid, and computationally efficient. Here, expressivity refers to its ability to generate a wide range of problem instances from a given one, while validity ensures the transformed system remains a valid KKT system. We discuss these properties in detail below.
Expressivity Intuitively, a transformation T is more expressive than another T ′ if it maps an initial problem to a superset of the targets T ′ can reach. We observe that the transformation in Eq. ( 7) has maximal expressivity. For example, setting M = 0, with proper B and β it can recover any target instance I ′ = (Q ′ , A ′ , b, c ′ ) from any source I, regardless of dimensions. This flexibility relies on the bias term, without which the transformations are limited. For instance, a low-rank A cannot map to a higher-rank A ′ . In practice, full expressivity is unnecessary, as our practical objective is not to span the entire space of QP instances, but to generate meaningful, diverse variants for data augmentation.
Validity Transformation following Eq. ( 7), under some conditions, can ensure that the transformed system remains a valid KKT system, which we explore below. We can write the matrices in block matrix form M := M 11 M 12 M 21 M 22 and B := B 11 B 12 B 21 B 22 , such that dimensions for A and Q are matched. Using straightforward calculations, we have
T 11 T 12 T 21 T 22 = M 11 M 12 M 21 M 22 Q A ⊺ A 0 M ⊺ 11 M ⊺ 21 M ⊺ 12 M ⊺ 22 + B 11 B 12 B 21 B 22 , T 11 = M 11 QM ⊺ 11 + M 12 AM ⊺ 11 + M 11 A ⊺ M ⊺ 12 + B 11 T 12 = M 11 QM ⊺ 21 + M 12 AM ⊺ 21 + M 11 A ⊺ M ⊺ 22 + B 12 T 21 = M 21 QM ⊺ 11 + M 22 AM ⊺ 11 + M 21 A ⊺ M ⊺ 12 + B 21 T 22 = M 21 QM ⊺ 21 + M 22 AM ⊺ 21 + M 21 A ⊺ M 22 + B 22 .
So that the transformed equation Eq. ( 7) is a valid KKT form
Q ′ A ′ ⊺ A ′ 0 x ′ * λ ′ * = -c ′ b ′ -s ′ * , it requires T 12 = T ⊺
21 , and we further require T 11 be another positive definite matrix, and T 22 = 0 must hold.
For inequality constraints, the transformation matrix M 22 must satisfy certain conditions to enable efficient computation of the new solution. These conditions are not required for feasibility or optimality but reflect our goal of avoiding QP solving. To isolate M 22 , we simplify the system by discarding the bias terms B, β, setting M 12 , M 21 to zero, and fixing M 11 = I. The resulting transformed KKT system is,
Q A ⊺ M ⊺ 22 M 22 A 0 I 0 0 (M ⊺ 22 ) † x * λ * = I 0 0 M 22 -c b -s * .(8)
This system indicates that the primal solution x * remains unchanged, while the dual solution is mapped to (M ⊺ 22 ) † λ * . The following proposition establishes the conditions under which this transformation preserves the solution and avoids re-solving the LCQPs; see Appendix D.1 for a proof. Proposition 2.1. Let I := (Q, A, b, c) be a LCQP instance with optimal primal-dual solution x * , λ * . Consider a transformation defined by T (I) := (Q, M 22 A, M 22 b, c). Then the transformed problem preserves the primal solution x * if and only if M 22 takes the block form N 11 N 12 N 21 N 22 to match the dimensions of active and inactive constraints, where N 11 has full row rank equal to the number of active constraints, N 12 s * ā = 0, N 22 s * ā ≥ 0.
In summary, the conditions on M , B, β for the validity of the new problem is as follows,
M 11 QM ⊺ 11 + M 12 AM ⊺ 11 + M 11 A ⊺ M ⊺ 12 + B 11 ≻ 0 M 21 QM ⊺ 21 + M 22 AM ⊺ 21 + M 21 A ⊺ M 22 + B 22 = 0 B 12 = B ⊺ 21 B (M ⊺ ) † x * λ * = β,and
M 22 satisfies Proposition 2.1. (9
)
Computational efficiency In general, satisfying the full transformation structure in Eq. ( 9) is challenging, particularly when arbitrary M , B and β are involved. However, we notice the multiplicative term M and bias terms B, β can be decoupled. That is, we can design transformations with the bias terms B, β, e.g. Appendix C.2, but we mainly focus on dropping them, setting M 12 and M 21 to zero matrices, and investigate the design space of M 11 and M 22 . We will abbreviate the subscripts of M ii , and we target at designing transformations of the form
M 1 0 0 M 2 Q A ⊺ A 0 M ⊺ 1 0 0 M ⊺ 2 (M ⊺ 1 ) † 0 0 (M ⊺ 2 ) † x * λ * = M 1 0 0 M 2 -c b -s * . (10)
We notice the commutativity and the decoupled nature,
M 1 0 0 M 2 = I 0 0 M 2 M 1 0 0 I = M 1 0 0 I I 0 0 M 2 ,
which enables us to design M 1 , M 2 separately, and merge them later.
The design space in Eq. ( 10) is flexible but constrained by the need to compute pseudo-inverses. For M 1 ∈ R n ′ ×n , there are three cases: (i) n ′ = n linear reparameterization, (ii) n ′ < n, dropping variables, and (iii) n ′ > n adding variables. While (ii) is often ill-posed since M ⊺ 1 does not have full column rank and lacks a right inverse, some special cases are still feasible, as we will show. More broadly, computing (M ⊺ 1 ) † = M 1 (M ⊺ 1 M 1 ) -1 requires full column rank and typically costs O(n 3 ). To improve efficiency, we focus on structured matrices, for example, diagonal matrices, as a particular case, enable both efficient inverse calculation and generation in O(n) time. These considerations motivate the use of diagonal or structured M 1 and M 2 for scalable data augmentation. To formalize efficiency, we introduce two notions of efficiently computable transformations. The first covers transformations whose solutions can be computed in linear time.
this section cite: []

Section: Definition 2.2 (Efficiently recoverable transformation).
For an LCQP instance I := (Q, A, b, c) ∈ I and its primal-dual solutions x * , λ * , a transform T ∈ T is efficiently recoverable, if the new solutions of the transformed instance I ′ = T (I) can be obtained within linear time O(n).
The second one, motivated by unsupervised settings, tightens this by requiring that the transformation be independent of the original solution and focuses on generating structurally consistent instances rather than solving them.
this section cite: []

Section: Definition 2.3 (Solution-independent transformation).
Given a LCQP instance I := (Q, A, b, c) ∈ I, a transformation T ∈ T is solution-independent if the solution of the transformed problem can be obtained without a solver, and the transformation parameters do not depend on the original optimal solutions x * , λ * .
this section cite: []

Section: Example transformations
In the following, we discuss various transformations that fit into the above framework.
Removing idle variables As discussed above, when M 1 does not have full column rank, the pseudo inverse (M ⊺ 1 ) † typically does not exist, making such transformations ill-posed. However, there is a special case in which we can safely remove a variable, specifically, when it is idle because its optimal value is zero, resulting in the following proposition. Proposition 2.4. Let I := (Q, A, b, c) be an LCQP instance with primal-dual solution x * , λ * . Then a variable x ′ can be removed from the problem without affecting the optimal values of the remaining variables if and only if x ′ * = 0.
Removing inactive constraints Like the variable removal case above, a constraint can be removed under certain conditions by introducing a wide identity matrix M 2 that selectively excludes the corresponding row. Proposition 2.5. Let I := (Q, A, b, c) be an LCQP instance with optimal primal-dual solution x * , λ * . Then a constraint of the form a ′ ⊺ x * ≤ b ′ can be removed from the problem without affecting the optimal solution if and only if it is strictly inactive, i.e., a ′ ⊺ x * < b ′ .
The variable and constraint removal transformations are efficiently recoverable, as the remaining solutions are unchanged and need no recomputation. However, they are not solution-independent, since identifying removable components requires access to the primal solution. Moreover, we apply a heuristic on problem instances to select inactive constraints, as described in Appendix C.3.
Scaling variable coefficients A natural class of transformations involves scaling the coefficients associated with individual variables. Specifically, scaling the j-th column of A and the j-th entry of c by a nonzero scalar α j , while updating Q ij by α i α j , i.e., T (I) := (M 1 QM 1 , AM ⊺ 1 , b, M 1 c) with a diagonal M 1 preserves the structure of the QP. Under this transformation, the optimal value remains unchanged, and the solution x * j is rescaled by 1/α j . This transformation is efficiently recoverable, as the new solution can be obtained directly from the original in O(n) time. Moreover, it is also solution-independent, as it does not require access to the original solution, but only knowledge of how to compute the new one.
Adding variables When n ′ > n, the transformation effectively adds new variables to the problem and linearly combines existing ones. Without loss of generality, we consider adding a single new variable by choosing a transformation matrix of the form M 1 := I q ⊺ , with q ∈ R n being an arbitrary vector.
This yields a new positive definite quadratic matrix Q Qq q ⊺ Q q ⊺ Qq . We can find the pseudo inverse
(M ⊺ 1 ) † := M 1 (M ⊺ 1 M 1 ) -1 = I q ⊺ (I + qq ⊺ ) -1
, which can be computed with Sherman-Morrison formula [Shermen and Morrison, 1949]. In this case, the primal solution of the original variables does not remain the same. Interestingly, due to the structure of M 1 , another valid pseudo-inverse is
(M ⊺ 1 ) † = I 0 ⊺
, which is a special case indicating that the added variable has zero contribution to the solution. This corresponds to the reverse of the variable removal transformation discussed above. Proposition 2.6. Let I = (Q, A, b, c) be an LCQP instance with optimal solution x * , λ * . Define the transformation T (I) := (M 1 QM 1 , AM ⊺ 1 , b, M 1 c), where M 1 := I q ⊺ . Then the transformed problem has optimal primal solution (x * , x ′ * ) if and only if the new variable x ′ * = 0.
This transformation is both efficiently recoverable and solution-independent. Moreover, there also exist other implementations of variable addition in the form of Eq. ( 7), with a bias term. Please refer to Appendix C.4.
Scaling constraints Similar to the variable scaling transformation above, we can scale the constraint coefficients by fixing M 1 = I and letting M 2 be a square diagonal matrix. We assume all diagonal entries of M 2 are positive. If any entry is zero, the transformation reduces to inactive constraint removal above, or an active constraint is removed, and the problem will be relaxed. If any entry is negative, the corresponding inequality direction will be flipped, and the problem's solution may change. Specifically, the transformation would be T (I) := (Q, M 2 A, M 2 b, c).
Under this transformation, the new dual variables are given by λ ′ * := M -1 2 λ * , which can be computed in linear time since M 2 is a diagonal matrix. The primal solution remains unchanged. This transformation is both efficiently recoverable and solution-independent.
In Appendix C, we outline additional transformations, including constraint addition Appendix C.1.
this section cite: ['b59']

Section: Using data augmentations
Having introduced efficient data augmentation methods for LCQPs and LPs, we now describe how they can be integrated into different training pipelines. Our target task is graph regression to predict the objective value from a graph representation. Depending on the setting, we can either (1) use solution information to generate supervised labels for related problems or (2) apply solution-independent augmentations for contrastive pretraining without solutions.
Supervised learning Given a training set and a set of augmentation methods, we dynamically generate additional training instances during optimization. We randomly apply a selected augmentation or a combination of augmentations at each iteration, and train the MPNN using the supervised loss on the predicted objective value.
Contrastive pretraining We perform self-supervised contrastive learning on the entire dataset without touching the solutions of the problems, using the NT-Xent loss [Chen et al., 2020]. Specifically, for a mini-batch of N data instances, we generate two augmented views of each instance using solution-independent transformations, resulting in 2N data instances. We consider the two views of the same instance (i, j) as a positive pair, and the other 2(N -1) samples as negative. The similarity between embeddings is measured by 2-norm-normalized cosine similarity sim(u, v) := u ⊺ v ∥u∥∥v∥ . For each instance i and its positive sample j, we have the NT-Xent loss on the pooled representations z i , z j from Eq. ( 6) as
- 1 N N i=1 log exp (sim(z i , z j )/τ ) 2N k=1 I(k ̸ = i) exp (sim(z i , z k )/τ ) ,(11)
where τ > 0 is a temperature hyperparameter.
this section cite: ['b10']

Section: Experimental setup and results
To empirically validate the effectiveness of our data augmentations, we conduct a series of experiments, answering the following research questions. 2Q1 Do our augmentations improve supervised learning, especially under data scarcity? Q2 Are the augmentations effective in contrastive pretraining followed by supervised finetuning? Q3 Does the pretrained model enhance generalization on out-of-distribution (OOD) or larger datasets? Q4 What is the practical computational overhead of the augmentations?
As LPs are a special case of QPs, we evaluate them separately. We generate 10 000 instances for each dataset, each with 100 variables and 100 inequality constraints, and split the data into training, validation, and test sets with 8 : 1 : 1 ratio. To study performance under data scarcity, we partition the training set into multiple disjoint subsets in each run, each containing either 10% or 20% of the full training data. Models are trained independently on each subset, and results are aggregated across all partitions. Hyperparameters are tuned using supervised training on the full training set and fixed across all methods. Specifically, we use a 6-layer MPNN followed by a 3-layer MLP with 192 hidden dimensions and GraphNorm [Cai et al., 2021]. All experiments are conducted on a single NVIDIA L40S GPU. We evaluate performance using the mean relative objective error (in percentage) over the test set, defined as Supervised learning To address Q1, we investigate the impact of data augmentation on LPs and QPs in a supervised learning setting. We evaluate performance on synthetically generated datasets following the procedure described in Appendix G. Models are trained with a batch size of 32 for up to 2000 epochs, with early stopping after 200 epochs of patience. We evaluate performance under data scarcity by training models on subsets containing 10%, 20%, and 100% of the training data. As baselines, we apply the augmentations proposed by You et al. [2020]: node dropping, edge perturbation, and feature masking. In addition, we evaluate each of our proposed augmentations separately and in combination.
Hyperparameter details for all augmentations are provided in Appendix F. Results, shown in Table 1, reveal that the augmentations from You et al. [2020] fail to improve performance consistently and can even be detrimental compared to training without augmentation. In contrast, all of our proposed methods consistently yield better performance, and combining augmentations further amplifies the improvement up to 62.6% on LP with 20% of training data, aligning with empirical evidence observed in You et al. [2020]. Contrastive pretraining To address Q2, we evaluate whether contrastive pretraining can improve supervised fine-tuning, potentially under data scarcity. We adopt a semi-supervised setting: a small subset (10%, 20%, 100%) of the training data is labeled, while the whole training set is available as unlabeled data. During pretraining, we use the complete unlabeled training set and train only an MPNN backbone without a prediction head, following the deployment described in Section 2.3. We pretrain for 800 epochs with a batch size of 128, and set τ = 0.1. To assess pretraining quality and pick the best set of hyperparameters, we use linear probing [Veličković et al., 2018], training only a linear regression layer on top of a frozen MPNN to efficiently evaluate feature quality. For finetuning, we follow Zeng and Xie [2021], attaching an MLP head and jointly training it with the MPNN using supervised regression loss, essentially the same setup as supervised learning, but initialized from a pretrained model.
As baselines, we consider several graph contrastive learning methods. GraphCL [You et al., 2020] generates views via node dropping, edge perturbation, and feature masking. GCC [Qiu et al., 2020] samples random walk subgraphs. IGSD [Zhang et al., 2023] uses graph diffusion [Gasteiger et al., 2019] combined with model distillation. MVGRL [Hassani and Khasahmadi, 2020] also employs diffusionbased views but performs contrastive learning at the graph-node level. Additionally, we include mutual information maximization methods such as InfoGraph [Sun et al., 2019] and DGI [Veličković et al., 2018]. Beyond contrastive methods, we evaluate the generative SSL method GAE [Kipf and Welling, 2016], which reconstructs graph edge weights.
As shown in Table 2, GraphCL and GCC pretraining can improve performance in some cases but do not consistently yield better results. Other baselines even degrade performance. In contrast, our pretraining methods substantially improve, reducing the objective gap by 59.4% on LP and 54.1% on QP with only 10% of the training data. This highlights the effectiveness of our data augmentations, which are specifically tailored for optimization problem instances and significantly enhance fine-tuning. Generalization To address Q3, we compare the performance of models trained from scratch versus models initialized with contrastive pretraining and then finetuned. For LPs, we generate four types of relaxed LP instances derived from MILPs: Set Cover (SC), Maximum Independent Set (MIS), Combinatorial Auction (CA), and Capacitated Facility Location (CFL), following Gasse et al. [2019].
For QPs, we generate instances of soft-margin SVM, Markowitz portfolio optimization, and LASSO regression following Jung et al. [2022]. If possible, problem sizes and densities are kept similar to the pretraining datasets; see more details in Appendix G. As shown in Tables 3 and 4, pretrained models outperform models trained from scratch in almost all cases, demonstrating strong transferability to OOD tasks. The evaluation on larger datasets can be found in Appendix E.2.
this section cite: ['b77', 'b77', 'b77', 'b65', 'b79', 'b77', 'b55', 'b81', 'b23', 'b30', 'b60', 'b65', 'b38', 'b21', 'b35']

Section: Conclusion
We introduced a principled framework for data augmentation in learning to optimize over linear and quadratic programming. By leveraging affine transformations of the KKT system, we designed a family of expressive, solution-preserving, and computationally efficient transformations. Our method allows for augmentations that either admit exact solution recovery or preserve key structural properties without requiring access to the original solutions, making them suitable for supervised and contrastive learning. Extensive experiments show that these augmentations consistently improve performance under data scarcity, generalize to larger and out-of-distribution problems, and outperform existing graph augmentation baselines. This work highlights the benefits of optimization-aware augmentation strategies and opens new directions for robust, scalable L2O under limited supervision.
this section cite: []

Section: References
Ref_id:b0 Title: Branching rules revisited Year: (2005)
Ref_id:b1 Title: Diffwire: Inductive graph rewiring via the lovasz bound Year: (2022)
Ref_id:b2 Title: Locality-aware graph-rewiring in gnns Year: (2023)
Ref_id:b3 Title: Machine learning for combinatorial optimization: a methodological tour d'horizon Year: (2021)
Ref_id:b4 Title: Graph barlow twins: A self-supervised representation learning framework for graphs Year: (2022)
Ref_id:b5 Title: Stress testing mixed integer programming solvers through new test instance generation methods Year: (2019)
Ref_id:b6 Title: Convex Optimization Year: (2004)
Ref_id:b7 Title: Residual gated graph convnets Year: (2017)
Ref_id:b8 Title: Graphnorm: A principled approach to accelerating graph neural network training Year: ()
Ref_id:b9 Title: Combinatorial optimization and reasoning with graph neural networks Year: (2023)
Ref_id:b10 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b11 Title: On representing linear programs by graph neural networks Year: (2022)
Ref_id:b12 Title: On representing mixed-integer linear programs by graph neural networks Year: (2023)
Ref_id:b13 Title: Expressive power of graph neural networks for (mixed-integer) quadratic programs Year: (2004)
Ref_id:b14 Title: Rethinking the capacity of graph neural networks for branching strategy Year: (2024)
Ref_id:b15 Title: Accelerating primal solution findings for mixed integer programs based on solution prediction Year: (2020)
Ref_id:b16 Title: Data augmentation for deep graph learning: A survey Year: (2022)
Ref_id:b17 Title: Augment with care: Contrastive learning for combinatorial problems Year: (2022)
Ref_id:b18 Title: Convolutional networks on graphs for learning molecular fingerprints Year: (2015)
Ref_id:b19 Title: Ugsl: A unified framework for benchmarking graph structure learning Year: (2023)
Ref_id:b20 Title: Qplib: a library of quadratic programming instances Year: (2019)
Ref_id:b21 Title: Exact combinatorial optimization with graph convolutional neural networks Year: (2019)
Ref_id:b22 Title: The machine learning for combinatorial optimization competition (ml4co): Results and insights Year: (2022)
Ref_id:b23 Title: Diffusion improves graph learning Year: (2019)
Ref_id:b24 Title: A deep instance generative framework for milp solvers under limited data availability Year: (2023)
Ref_id:b25 Title: Neural message passing for quantum chemistry Year: (2017)
Ref_id:b26 Title: Miplib 2017: data-driven compilation of the 6th mixed-integer programming library Year: (2021)
Ref_id:b27 Title: Acm-milp: Adaptive constraint modification via grouping and selection for hardness-preserving milp instance generation Year: ()
Ref_id:b28 Title: Drew: Dynamically rewired message passing with delay Year: (2023)
Ref_id:b29 Title: Inductive representation learning on large graphs Year: (2017)
Ref_id:b30 Title: Contrastive multi-view representation learning on graphs Year: (2009)
Ref_id:b31 Title: Graph ranking contrastive learning: A extremely simple yet efficient method Year: (2023)
Ref_id:b32 Title: Searching large neighborhoods for integer linear programs with contrastive learning Year: (2023)
Ref_id:b33 Title: Regcl: rethinking message passing in graph contrastive learning Year: (2024)
Ref_id:b34 Title: Graph structure learning for robust graph neural networks Year: (2020)
Ref_id:b35 Title: Learning context-aware adaptive solvers to accelerate quadratic programming Year: (2022)
Ref_id:b36 Title: FoSR: First-order spectral rewiring for addressing oversquashing in gnns Year: (2022)
Ref_id:b37 Title: Mip-gnn: A data-driven framework for guiding combinatorial solvers Year: (2022)
Ref_id:b38 Title: Variational graph auto-encoders Year: (2016)
Ref_id:b39 Title: PDHG-unrolled learning-to-optimize method for large-scale linear programming Year: (2024-02)
Ref_id:b40 Title: On the power of small-size graph neural networks for linear programming Year: (2024)
Ref_id:b41 Title: Towards explaining the power of constant-depth graph neural networks for structured linear programming Year: (2025)
Ref_id:b42 Title: Towards foundation models for mixed integer linear programming Year: (2024-03)
Ref_id:b43 Title: Spectral augmentation for self-supervised learning on graphs Year: (2022)
Ref_id:b44 Title: Milp-studio: Milp instance generation via block structure decomposition Year: (2024)
Ref_id:b45 Title: Revisiting graph contrastive learning from the perspective of graph spectrum Year: (2022-02)
Ref_id:b46 Title: Self-supervised learning: Generative or contrastive Year: (2021)
Ref_id:b47 Title: Local augmentation for graph neural networks Year: (2022-02)
Ref_id:b48 Title: Towards unsupervised deep graph structure learning Year: (2022-02)
Ref_id:b49 Title: Numerical optimization Year: (1999)
Ref_id:b50 Title: Dropgnn: Random dropouts increase the expressiveness of graph neural networks Year: (2021)
Ref_id:b51 Title: Towards graph neural networks for provably solving convex optimization problems Year: (2025)
Ref_id:b52 Title: Probabilistically rewired message-passing neural networks Year: (2023)
Ref_id:b53 Title: Exploring the power of graph neural networks in solving linear optimization problems Year: (2024)
Ref_id:b54 Title: Probabilistic graph rewiring via virtual nodes Year: (2024)
Ref_id:b55 Title: Gcc: Graph contrastive coding for graph neural network pre-training Year: (2020)
Ref_id:b56 Title: Learning transferable visual models from natural language supervision Year: ()
Ref_id:b57 Title: Towards deep graph convolutional networks on node classification Year: (2019)
Ref_id:b58 Title: The graph neural network model Year: (2008)
Ref_id:b59 Title: Adjustment of an inverse matrix corresponding to changes in the elements of a given column or a given row of the original matrix Year: (1949)
Ref_id:b60 Title: Infograph: Unsupervised and semi-supervised graph-level representation learning via mutual information maximization Year: (2019)
Ref_id:b61 Title: Adversarial graph augmentation to improve graph contrastive learning Year: (2021)
Ref_id:b62 Title: Large-scale representation learning on graphs via bootstrapping Year: (2021)
Ref_id:b63 Title: Understanding over-squashing and bottlenecks on graphs via curvature Year: (2021)
Ref_id:b64 Title: Graph attention networks Year: (2017)
Ref_id:b65 Title: Deep graph infomax Year: (2018)
Ref_id:b66 Title: S3gcl: Spectral, swift, spatial graph contrastive learning Year: (2024)
Ref_id:b67 Title: Dig-milp: a deep instance generator for mixed-integer linear programming with feasibility guarantee Year: (2023)
Ref_id:b68 Title: Graph contrastive learning with generative adversarial network Year: (2023)
Ref_id:b69 Title: On representing convex quadratically constrained quadratic programs via graph neural networks Year: (2024)
Ref_id:b70 Title: Self-supervised learning on graphs: Contrastive, generative, or predictive Year: (2021)
Ref_id:b71 Title: How powerful are graph neural networks? arXiv preprint Year: (2018)
Ref_id:b72 Title: Spectral-aware augmentation for enhanced graph representation learning Year: (2024-02)
Ref_id:b73 Title: An efficient unsupervised framework for convex quadratic programs via deep unrolling Year: (2024)
Ref_id:b74 Title: Learning to generate scalable milp instances Year: (2024-03)
Ref_id:b75 Title: Understanding negative sampling in graph representation learning Year: (2020)
Ref_id:b76 Title: Autogcl: Automated graph contrastive learning via learnable view generators Year: (2022)
Ref_id:b77 Title: Graph contrastive learning with augmentations Year: (2009)
Ref_id:b78 Title: Graph contrastive learning automated Year: ()
Ref_id:b79 Title: Contrastive self-supervised learning for graph classification Year: (2021)
Ref_id:b80 Title: Clcr: Contrastive learning-based constraint reordering for efficient milp solving Year: (2025)
Ref_id:b81 Title: Iterative graph self-distillation Year: (2023)
Ref_id:b82 Title: MILP-FBGen: LP/MILP instance generation with Feasibility/Boundedness Year: ()
Ref_id:b83 Title: Data augmentation for graph neural networks Year: (2021)
Ref_id:b84 Title: Graph data augmentation for graph machine learning: A survey Year: (2022)
Ref_id:b85 Title: Opengsl: A comprehensive benchmark for graph structure learning Year: (2023)
Ref_id:b86 Title: Deep graph contrastive representation learning Year: (2020)
Ref_id:b87 Title: Graph contrastive learning with adaptive augmentation Year: (2021)
Ref_id:b88 Title: SE-GSL: A general and effective graph structure learning framework through structural entropy optimization Year: (2023)
