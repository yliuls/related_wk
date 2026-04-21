Title: Achieving Linear Speedup and Near-Optimal Complexity for Decentralized Optimization over Row-stochastic Networks
Abstract: A key challenge in decentralized optimization is determining the optimal convergence rate and designing algorithms to achieve it. While this problem has been extensively addressed for doublystochastic and column-stochastic mixing matrices, the row-stochastic scenario remains unexplored. This paper bridges this gap by introducing effective metrics to capture the influence of row-stochastic mixing matrices and establishing the first convergence lower bound for decentralized learning over row-stochastic networks. However, existing algorithms fail to attain this lower bound due to two key issues: deviation in the descent direction caused by the adapted gradient tracking (GT) and instability introduced by the PULL-DIAG protocol. To address descent deviation, we propose a novel analysis framework demonstrating that PULL-DIAG-GT achieves linear speedup-the first such result for row-stochastic decentralized optimization. Moreover, by incorporating a multi-step gossip (MG) protocol, we resolve the instability issue and attain the lower bound, achieving near-optimal complexity for decentralized optimization over rowstochastic networks.

Section: Introduction
Scaling machine learning tasks to large datasets and models requires efficient distributed computing across multiple nodes. This paper investigates decentralized stochastic opti-Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). mization over a network of n nodes:
min x∈R d f (x) = 1 n n i=1 f i (x) := E ξi∼Di [F (x; ξ i )] (1)
where ξ i is a random data vector supported on Ξ i ⊆ R q with some distribution D i , and F : R d × R q → R is a Borelmeasurable function. Each loss function f i is accessible only by node i and is assumed to be smooth and potentially nonconvex. Note that data heterogeneity typically exists, i.e., the local data distributions {D i } n i=1 vary across nodes. In this paper, we model decentralized communication between nodes as a directed graph, a scenario frequently encountered in practical applications. For example, bidirectional communication may be infeasible due to differences in node power ranges (Yang et al., 2019) or connection failures (Yemini et al., 2022;Li et al., 2024). In distributed deep learning, carefully designed directed topologies can achieve sparser and faster communication than their undirected counterparts, thereby reducing the training wall clock time (Bottou et al., 2018;Assran et al., 2019;Yuan et al., 2021).
this section cite: ['b44', 'b45', 'b4', 'b3', 'b48']

Section: Network topology and mixing matrix.
A central challenge in decentralized optimization is determining the optimal convergence rate and designing algorithms that achieve it. This requires a theoretical understanding of how network topologies influence decentralized algorithms. For any given connected network, its topology can be represented by a mixing matrix that reflects its connectivity pattern, serving as a critical tool for evaluating the network's impact. In undirected networks, symmetric and doubly-stochastic matrices can be readily constructed. However, in directed networks, constructing a doubly-stochastic mixing matrix is generally infeasible. Instead, mixing matrices are typically either column-stochastic (Nedić & Olshevsky, 2014;Nedić et al., 2017) or row-stochastic (Sayed, 2014;Mai & Abed, 2016), but not both.
Optimal complexity over doubly-stochastic networks is well-established. The connectivity of a doubly-stochastic mixing matrix can be effectively assessed using the spectral gap, a metric that quantifies how closely a decentralized network approximates a fully connected one. Leveraging this metric, several studies have established the optimal convergence rates for decentralized algorithms. For example, references (Scaman et al., 2017;2018;Sun & Hong, 2019;Kovalev et al., 2021) derive optimal convergence rates for convex or non-stochastic decentralized optimization. Lu & De Sa (2021) determine the optimal complexity for nonconvex and stochastic decentralized optimization over a specific type of linear network, while Yuan et al. (2022) extend this complexity to a significantly broader class of networks. The optimal complexity for doubly-stochastic time-varying networks has been established in (Huang & Yuan, 2022;Li & Lin, 2024;Kovalev et al., 2021).
Optimal complexity over column-stochastic networks is established recently. If out-degree information is available prior to communication, a column-stochastic matrix can be readily constructed. When decentralized algorithms rely solely on column-stochastic matrices, this is referred to as the COL-ONLY setting. The PUSH-SUM gossip protocol (Kempe et al., 2003;Tsianos et al., 2012) forms the foundation of COL-ONLY algorithms. Many algorithms based on PUSH-SUM achieve superior convergence rates, e.g., Nedić & Olshevsky (2015); Tsianos et al. (2012); Zeng & Yin (2017); Xi & Khan (2017); Xi et al. (2017); Nedić et al. (2017); Assran et al. (2019); Qureshi et al. (2020). However, these works fail to precisely capture the influence of column-stochastic networks and, as a result, do not clarify the optimal complexity in the COL-ONLY setting. This open question is addressed in a recent study by Liang et al. (2023), which introduces effective metrics to evaluate the influence of column-stochastic networks, establishes the optimal lower bound for the COL-ONLY setting, and proposes algorithms that achieve this bound.
Optimal complexity over row-stochastic networks remains unclear yet. If out-degree information is unavailable, column-stochastic matrices cannot be directly constructed. However, row-stochastic matrices can be formed using in-degree information, which can be easily obtained by counting received messages. This is referred to as the ROW-ONLY setting. Similar to how PUSH-SUM serves as the basis for COL-ONLY algorithms, the foundation of ROW-ONLY methods is the PULL-DIAG gossip protocol (Mai & Abed, 2016). Building on PULL-DIAG, Mai & Abed (2016) adapted the distributed gradient descent (DGD) algorithm for the ROW-ONLY setting, while Li et al. (2019); Xin et al. (2019c) extended gradient tracking methods, and Ghaderyan et al. (2023); Lü et al. (2020); Xin et al. (2019a) introduced momentum-based ROW-ONLY gradient tracking. However, the convergence analysis for ROW-ONLY algorithms is still quite limited. Current analyses focus only on deterministic and strongly convex loss functions, leaving the performance of ROW-ONLY algorithms in non-convex and stochastic settings unknown. More importantly, the impact of rowstochastic networks on the convergence rate of ROW-ONLY algorithms remains unclear. These gaps present significant obstacles to determining the optimal complexity in the ROW-ONLY setting. Some fundamental open problems are: Q1. What are the effective metrics that can fully capture the impact of row-stochastic networks on decentralized stochastic optimization, and how do they influence the convergence of prevalent ROW-ONLY algorithms?
Q2. Given these metrics, what is the lower bound on the convergence rate for ROW-ONLY algorithms in the non-convex and stochastic setting?
Q3. Can existing ROW-ONLY algorithms readily achieve the optimal convergence rate? If not, what limitations do they face?
Q4. Can we develop new ROW-ONLY algorithms that overcome the limitations of existing algorithms and attain the aforementioned lower bound?
Main contributions. This paper improves the understanding of decentralized methods over row-stochastic networks by addressing these open questions. Our contributions are :
C1. We find that the metrics generalized spectral gap and equilibrium skewness, proposed by (Liang et al., 2023) to characterize the influence of column-stochastic networks, can also effectively capture the impact of rowstochastic networks on decentralized algorithms.
C2. Using these metrics, we establish the first lower bound on the convergence rate for nonconvex decentralized stochastic first-order algorithms with a row-stochastic mixing matrix. This bound achieves linear speedup with respect to network size n and captures the influence of gradient noise, the mixing matrix, the number of nodes and the smoothness of the loss function.
C3. We find existing ROW-ONLY algorithms cannot attain the aforementioned lower bound due to two challenges. First, the use of row-stochastic mixing matrices alone introduces a deviation in the descent direction from the globally averaged gradient, preventing existing analyses from achieving linear speedup convergence. Second, the PULL-DIAG protocol introduces inversion of small values during operation, which introduces instability in ROW-ONLY algorithms.
C4. We develop a novel analysis framework proving that PULL-DIAG-GT achieves linear speedup, marking the first such result in ROW-ONLY scenarios. Moreover, when integrated with a multistep gossip (MG) protocol, MG-PULL-DIAG-GT addresses the instability caused by the inversion of small values and achieves the established lower bound. Therefore, both the lower bound and our algorithm achieve optimal complexity for decentralized learning over row-stochastic networks.
Notations. Let 1 n denote the vector of all-ones of n dimensions and I n ∈ R n×n the identity matrix. We let matrix A denote the row-stochastic matrix (A1 n = 1 n ). The set [n] represents the indices {1, 2, . . . , n}. Diag(A) refers to the diagonal matrix composed of the diagonal entries of A, while diag(v) is the diagonal matrix derived from vector v. The Perron vector of matrix A is π A , i.e.,
π A ≥ 0, π ⊤ A A = π ⊤ A , π ⊤ A 1 = 1. By letting Π A := diag(π A ), we define ∥v∥ π A := ∥Π 1/2 A v∥, which is associated with the induced matrix norm ∥W ∥ π A := ∥Π 1/2 A W Π -1/2 A ∥ 2 . We define A ∞ := 1 n π ⊤ A . The vector x (k) i
∈ R d represents the local model at node i during iteration k. We also define n × d matrices
x (k) := [(x (k) 1 ) ⊤ ; (x (k) 2 ) ⊤ ; • • • ; (x (k) n ) ⊤ ] ∇F (x (k) ;ξ (k) ) := [∇F 1 (x (k) 1 ;ξ (k) 1 ) ⊤ ;• • •;∇F n (x (k) n ; ξ (k) n ) ⊤ ] ∇f (x (k) ) := [∇f 1 (x (k) 1 ) ⊤ ; • • • ; ∇f n (x (k) n ) ⊤ ]
by stacking all local variables. The upright bold symbols (e.g. x, w, g) always denote network-level quantities. We define filtration F k as the collection of all the information available up to x (k) , excluding the stochastic gradient evaluated at x (k) . We use the symbol ≲ to represent inequality up to absolute constants.
this section cite: ['b23', 'b24', 'b31', 'b22', 'b32', 'b34', 'b14', 'b21', 'b49', 'b16', 'b14', 'b12', 'b35', 'b27', 'b35', 'b50', 'b24', 'b3', 'b30', 'b19', 'b22', 'b22', 'b17', 'b8', 'b20', 'b19']

Section: Metrics for Row-stochastic Networks
We consider a directed network with n computing nodes that is associated with a mixing matrix A = [a ij ] n i,j=1 ∈ R n×n where a ij ∈ (0, 1] if node j can send information to node i otherwise a ij = 0. Decentralized optimization is built upon partial averaging z + i = j∈Ni a ij z j in which z i ∈ R d is a local vector held by node i and N i denotes the in-neighbors of node i, including node i itself. Since every node conducts partial averaging simultaneously, we have
z ≜ [z ⊤ 1 ; z ⊤ 2 ; • • • ; z ⊤ n ] A-protocol -----→ z + = Az(2)
where A-protocol represents partial averaging with mixing matrix A. Evidently, the algebraic characteristics of A substantially affect the convergence of partial averaging and the corresponding decentralized optimization. This section explores metrics that capture the characteristics of A.
this section cite: []

Section: Row-stochastic Mixing Matrix
This paper focuses on a static directed network G = (V, E) associated with a row-stochastic matrix A = [a ij ] n×n .
this section cite: []

Section: Assumption 1 (Primitive and Row-stochastic).
The mixing matrix A is nonnegative, primitive and satisfies
A1 n = 1 n . The weight a ij ∈ (0, 1], if (i → j) ∈ E, otherwise a ij = 0.
If G is strongly-connected, i.e., there exists a directed path from each node to every other node, and A has a positive trace, then A is primitive. It is straightforward to make A row-stochastic by setting a ij = 1/(1 + d in i ) if (i, j) ∈ E or j = i otherwise a ij = 0, where E is the set of directed edges and d in i is the in-degree of node i excluding the selfloop. With Assumption 1, we have the following result:
Proposition 1 (Perron-Frobenius theorem (Perron, 1907)). If matrix A satisfies Assumption 1, there exists a unique equilibrium vector π A ∈ R n with positive entries such that
π ⊤ A A = π ⊤ A , 1 ⊤ n π A = 1,and
lim k→∞ A k = 1 n π ⊤ A .
this section cite: ['b28']

Section: Effective Metrics to Characterize Row-stochastic A
Most decentralized algorithms rely on gossip protocols like (2), where local variables are partially mixed to approximate the global average. The properties of the mixing matrix A play a critical role in determining both the feasibility of achieving the global average and the efficiency of this process. These properties serve as key metrics for assessing the influence of A on algorithmic performance.
Now we examine the A-protocol (2) where A is a rowstochastic mixing matrix A. Suppose that each node i has a local variable
z i ∈ R d , we let z = [z ⊤ 1 ; z ⊤ 2 ; • • • ; z ⊤ n ] ∈ R n×d and initialize x (0) = z.
Following the gossip protocol as in (2), we have the following recursions:
x (k) = Ax (k-1) = A k x (0) k→∞ ----→ 1 n π ⊤ A z,(3)
where we utilize the property lim k→∞ A k = 1 n π ⊤ A (see Proposition 1). It is evident that the matrix A influences both whether and how quickly x (k) approaches the global average 1 n 1 ⊤ n z/n. Inspired by (3), we use the following two metrics to characterize the row-stochastic matrix A:
• The generalized spectral gap 1 -β A of the rowstochastic matrix A, where
β A := A -1 n π ⊤ A π A = ∥A -A ∞ ∥ π A ∈ [0, 1)
quantifies the convergence rate of x (k) to the weighted average 1 n π ⊤ A z in (3). The π A -norm was defined in the notations in Sec. 1. As β A approaches 0, the iterates x (k) converge more rapidly to the fixed point 1 n π ⊤ A z of the A-protocol (3).
• The equilibrium skewness
κ A := max(π A )/ min(π A ) ∈ [1, +∞)
captures the disagreement between the equilibrium vector π A and the uniform vector n -1 1 n . When κ A → 1, it holds that π A → 1 n /n, and hence, the weighted average aligns better with the global average 1 n 1 ⊤ n z/n.
The spectral gap gauges the rate of the A-protocol when converging to the fixed point 1 n π ⊤ A z, while equilibrium skewness measures the proximity of the fixed point to the desired global average 1 n 1 ⊤ n z/n. Together, these metrics effectively capture the influence of row-stochastic mixing matrices and directed networks on decentralized algorithms. Omitting either would lead to an incomplete understanding of their impact. Figure 1 shows examples that the spectral gap and equilibrium skewness jointly impact the convergence to average consensus. The protocol used in Figure 1 is called PULL-DIAG and will be discussed in Sec. 2.3.
It is important to note that these two metrics are not new; they were proposed in (Xin et al., 2019b;Liang et al., 2023) to assess the influence of column-stochastic mixing matrices. Our contribution lies in demonstrating that these metrics are also applicable to row-stochastic mixing matrices. Prior to our work, no literature had examined the metrics that can gauge the influence of row-stochastic mixing matrices on decentralized algorithms.
this section cite: ['b19']

Section: Pull-Diag Protocol Corrects Weighted Average
According to (3), A-protocol alone cannot achieve the desired global average. The bias between the limiting weighted average 1 n π ⊤
A z and the desired global average 1 n 1 ⊤ n z/n can be corrected by the following manner:
A k diag(nπ A ) -1 z k→∞ ----→ 1 n π ⊤ A diag(nπ A ) -1 z = 1 n 1 ⊤ n z/n.(4)
Although the above strategy is effective, the quantity π A is not known beforehand. To estimate π A , prior works (Mai & Abed, 2016;Xin et al., 2019c;Ghaderyan et al., 2023) use power iterations, resulting in a practical and efficient approach referred to as PULL-DIAG in this paper:
V k+1 = AV k , D k+1 = Diag(nV k+1 ),(5a)
z (k+1) = V k+1 D -1 k+1 z.(5b)
With initialization V 0 = I n , we have V k = A k and
D k = Diag(nA k ). It holds that V k → 1 n π ⊤ A and D k → diag(nπ A ) as k → ∞.
Substituting these facts into (5b), we asymptotically achieve the bias correction illustrated in (4). The distributed implementation details of the PULL-DIAG protocol can be found in Appendix B. It is shown in Figure 1 that PULL-DIAG protocol corrects the weighted average and converges exponentially fast.
this section cite: ['b22', 'b8']

Section: Convergence Lower Bounds

this section cite: []

Section: Assumptions
This subsection specifies the category of decentralized algorithms to which our lower bound applies. Function class. We define the function class F ∆,L as the set of functions that satisfy Assumption 2, for any given dimension d ∈ N + and any initialization point
x (0) ∈ R d .
this section cite: []

Section: Assumption 2 (Smoothness).
There exists constants L, ∆ ≥ 0 such that
∥∇f i (x) -∇f i (y)∥ ≤ L∥x -y∥, ∀i ∈ n, x, y ∈ R d .
and for initial model parameter x (0) ,
f i (x (0) ) -inf x∈R d f i (x) ≤ ∆, ∀i ∈ [n].
Gradient oracle class. We assume that each node i processes its local cost function f i using a stochastic gradient oracle ∇F (x; ξ i ), which provides unbiased estimates of the exact gradient ∇f i with bounded variance. Specifically, we define the stochastic gradient oracle class O σ 2 as the set of all oracles ∇F (•; ξ i ) that satisfy Assumption 3.
this section cite: []

Section: Assumption 3 (Gradient oracles).
There exists a constant σ ≥ 0 such that for all x ∈ R d , i ∈ [n] we have
E[∇F i (x; ξ)] = ∇f i (x), tr(Var[∇F i (x; ξ)]) ≤ σ 2 .
We also assume that the gradient noise is linearly independent, i.e., ∀i ̸ = j ∈ [n] we have
Cov(∇F i (x i ; ξ i ), ∇F j (x j ; ξ j )) = 0.
Algorithm class description. We focus on decentralized algorithms where each node i maintains a local solution x (k) i at iteration k and communicates using the A-protocol defined in (2). These algorithms also adhere to the linearspanning property, as defined in (Carmon et al., 2020;2021;Yuan et al., 2022;Lu & De Sa, 2021). Informally, this property ensures that each local solution x (k) i resides within the linear space spanned by x (0) i , its local stochastic gradients, and interactions with neighboring nodes. Upon completion of K iterations, the final output x(K) can be any variable in span({{x
(k) i } n i=1 } K k=0 ).
Let A A denote the set of all algorithms that adhere to partial averaging via mixing matrix A and satisfy the linear-spanning property.
this section cite: ['b5', 'b49', 'b21']

Section: Lower Bound
With β A and κ A at hand, we show, for the first time, that the convergence rate of any non-convex decentralized stochastic first-order algorithm with a row-stochastic mixing matrix is lower-bounded by the following theorem.
Theorem 1 (Lower bound). For any given L ≥ 0, n ≥ 2, σ ≥ 0, and β ∈ [0.01, 1 -1/n], there exists a set of loss functions {f i } n i=1 ∈ F ∆,L , a set of stochastic gradient oracles in O σ 2 , and a row-stochastic matrix A ∈ R n×n with β A = β and ln(κ A ) = Ω(n(1 -β A )), such that the convergence of any algorithm A ∈ A A starting from
x (0) i = x (0) , i ∈ [n] with K iterations is lower bounded by E∥∇f ( x(K) )∥ 2 = Ω σ √ L∆ √ nK + (1 + ln(κ A ))L∆ (1 -β A )K ,(6)
where K, σ, L, and ∆ represent the total number of iterations, the gradient variance, the smoothness parameter of the functions, and the initial gap in the function values, respectively. The proof is in Appendix A.
Linear speedup. The first term, σ/ √ nK, dominates the lower bound (6) when K is sufficiently large, indicating that decentralized algorithms with row-stochastic mixing matrices could achieve linear speedup with respect to network size n (i.e., convergence improves as the number of computing nodes n increases).
this section cite: []

Section: Network topology impact.
The lower bound in (6) explicitly highlights the combined impact of the generalized spectral gap β A and the equilibrium skewness κ A on decentralized algorithms utilizing row-stochastic mixing matrices. Omitting either metric would provide an incomplete understanding of the algorithmic performance.
this section cite: []

Section: Deterministic scenario.
When the gradient noise σ = 0, the established lower bound in (6) for stochastic settings simplifies to the first lower bound for deterministic decentralized algorithms with row-stochastic mixing matrices.
this section cite: []

Section: Achieving Linear Speedup Using Row-stochastic Matrix Alone
The lower bound (6) reveals a linear speedup convergence rate of σ/ √ nK as K grows large. However, no existing ROW-ONLY decentralized stochastic algorithm has theoretically achieved this rate, highlighting a significant gap from the lower bound. This section identifies the challenges and presents a novel analysis framework that achieves the first theoretical linear speedup for ROW-ONLY algorithms.
this section cite: []

Section: Pull-DIAG-GT Algorithm
We begin by reviewing the state-of-the-art ROW-ONLY algorithms (Li et al., 2019;Xin et al., 2019c;Ghaderyan et al., 2023;Lü et al., 2020;Xin et al., 2019a), all of which are based on PULL-DIAG-GT-an adaptation of gradient tracking (Nedic et al., 2017;Di Lorenzo & Scutari, 2016;Xu et al., 2015;Qu & Li, 2017) designed specifically for ROW-ONLY scenarios.
x (k+1) = A(x (k) -αy (k) ),(7a)
y (k+1) = A(y (k) + D -1 k+1 g (k+1) -D -1 k g (k) ).(7b)
Here,
D k = Diag(A k ), ∀k ≥ 1, D 0 = I n . Matrix x (k)
denotes the stacked model parameters and y (k) denotes the gradient tracking term. g (k) denotes the stochastic gradient, defined as 0) . The details of algorithm implementation is provided in Appendix B. As recursion (7) incorporates the PULL-DIAG protocol (5) into gradient tracking, it is termed PULL-DIAG-GT throughout this paper.
g (k) = ∇F (x (k) ; ξ (k) ), with y (0) = g (
It is noteworthy D -1 k may involve inversion of zeros. Therefore, the following assumption is necessary: Assumption 4 (Bounded Diagonals). There exists a constant θ A > 0 such that
[A k ] -1 ii ≤ θ A , ∀k ≥ 1, i ∈ [n].
Remarkably, under Assumption 1, the existence of θ A can be guaranteed if and only if every node has a self-loop.
this section cite: ['b17', 'b8', 'b20', 'b25', 'b7', 'b42', 'b29']

Section: Algorithm insight.
As shown in A-protocol (3), communication utilizing a row stochastic matrix yields a biased average z → π ⊤ A z. In PULL-DIAG-GT, we can left-multiply π ⊤ A on both sides of (7a) and observe the following dynamics:
π ⊤ A x (k+1) = π ⊤ A x (k) -απ ⊤ A y (k) .(8)
If y (k) represents the stacked stochastic gradients, i.e., k) from converging to the solution to problem (1). To ensure convergence, PULL-DIAG-GT corrects the descent direction using D -1 k in (7b). Left-multiplying π ⊤ A on both sides of (7b) yields
y (k) = ∇F (x (k) ; ξ (k) ), the descent direction π ⊤ A y (k) devi- ates from the desired globally averaged gradient 1 ⊤ n y (k) /n, preventing π ⊤ A x (
π ⊤ A y (k+1) -π ⊤ A D -1 k+1 g (k+1) = π ⊤ A y (k) -π ⊤ A D -1 k g (k) = • • • = π ⊤ A y (0) -π ⊤ A g (0) (a) = 0,
where equality (a) holds due to 0) . This implies
y (0) = g (
π ⊤ A y (k) = π T A D -1 k g (k) (b) ≈ 1 ⊤ n g (k) ≜ nḡ (k) ,(9)
where (b) holds because k) . The combined dynamics of ( 8)
D k = Diag(A k ) ≈ diag(π A ), and ḡ(k) = (1/n)1 ⊤ n g (
and ( 9) ensure that PULL-DIAG-GT converges along the globally averaged gradient, ultimately solving problem (1).
Challenges in establishing linear speedup. Although the rationale behind PULL-DIAG-GT's convergence to the desired solution is clear, to the best of our knowledge, no existing analysis has established its linear speedup convergence with respect to the network size n. This subsection highlights the key challenges involved.
We first define two error terms to facilitate analysis: k) be the centroid variable for x (k) . We introduce ∥x (k) -1 n π ⊤ A x (k) ∥ 2 as the consensus error to gauge the difference between each local variable k) .
• Consensus error. Let π ⊤ A x (
x (k) i to the centroid π ⊤ A x (
• Descent deviation. We define ∥π ⊤ A y (k) -nḡ (k) ∥ 2 as the descent deviation, measuring the discrepancy between the PULL-DIAG-GT descent direction and the desired globally averaged stochastic gradient.
If the aforementioned two error terms are guaranteed to diminish to zero, PULL-DIAG-GT enables each k) , and the update of w (k) asymptotically approximate centralized parallel SGD:
x (k) i to converge to centroid w (k) = π ⊤ A x (
w (k+1) = w (k) -nαḡ (k) w ,(10)
where
ḡ(k) w = (1/n) n i=1 ∇F (w (k) ; ξ (k) i
) is the globally averaged gradient over centroid. This ensures w (k) to converge to the desired solution to problem (1).
Linear speedup analysis in distributed stochastic optimization relies heavily on the descent structure aligned with the globally averaged stochastic gradient (Yu et al., 2019;Xin et al., 2019b;Koloskova et al., 2020;Yang et al., 2021). When each local stochastic gradient g i introduces meansquare gradient noise σ 2 , the globally averaged gradient ḡ benefits from reduced noise σ 2 /n, forming the foundation for linear speedup. Gradient tracking with doubly or columnstochastic mixing matrices preserves the globally averaged gradient descent direction ḡ, enabling well-established linear speedup convergence (Koloskova et al., 2020;Lu & De Sa, 2021;Assran et al., 2019;Kungurtsev et al., 2023;Liang et al., 2023). In contrast, PULL-DIAG-GT suffers from an additional descent deviation between π ⊤ A y and ḡ, making existing linear speedup analyses inapplicable and necessitating new techniques to address this limitation.
this section cite: ['b47', 'b13', 'b43', 'b13', 'b21', 'b3', 'b15', 'b19']

Section: Achieving linear speedup in Pull-Diag-GT
This subsection presents a new analytical framework to establish the linear speedup rate for PULL-DIAG-GT.
Descent lemma. Our analysis begins with a descent lemma.
Lemma 2 (Descent lemma). Under Assumptions 1-4, when α ≤ 1 2nL , for any k ≥ 0 we have
nα 2 ∥∇f (w (k) )∥ 2 ≤ f (w (k) ) -E[f (w (k+1) )|F k ] + αL 2 ∥∆ (k) x ∥ 2 F consensus error + α n ∥E[π ⊤ A y (k) -nḡ (k) |F k ]∥ 2 descent deviation + α 2 Lσ 2 2 d k - α 4n ∥π ⊤ A D -1 k ∇f (x (k) )∥ 2(11)
where
w (k) := π ⊤ A x (k) is the centroid variable, d k := n j=1 ( [π A ]j [D k ]j ) 2 is a constant. ∆ (k) x := x (k) -1 n π ⊤ A x (k) , and ḡ(k) = (1/n)1 ⊤ n ∇F (x (k) ; ξ (k) ).
As anticipated, the descent lemma incorporates both the consensus error and the descent deviation. Due to the conditional expectation operation, the descent deviation in the lemma appears as
∥E[π ⊤ A y (k) -nḡ (k) | F k ]∥ 2 .
In contrast, the descent lemma for gradient tracking using doubly or column-stochastic mixing matrices ensures ȳ(k) = ḡ(k) , thereby accounting solely for the consensus error.
Estimate descent deviation. According to ( 9), we have k) . This implies that
π ⊤ A y (k) = π T A D -1 k g (
∥E[π ⊤ A y (k) -nḡ (k) |F k ]∥ = ∥(π ⊤ A D -1 k -1 ⊤ n )∇f (x (k) )∥.
Therefore, we can use the following lemma to provide an estimate for the descent deviation. Lemma 3. Under Assumptions 1, 2 and 4, for all k ≥ 1,
∥(π ⊤ A D -1 k -1 ⊤ n )∇f (x (k) )∥ 2 ≤ 2nκ A θ 2 A β 2k A L 2 ∥∆ (k) x ∥ 2 F + 2nL((f (w (k) ) -f * ) ,
where
f * := n -1 n i=1 f * i .
Lemma 3 employs f (w (k) )-f * to establish an upper bound on the norm of the stacked gradients. In many instances, this approach can negatively impact the recursive nature of the descent lemma. Nevertheless, in this particular situation, we can effectively accommodate it due to its O(β 2k A ) coefficient. Details are provided in Lemma 13 in Appendix C.
this section cite: []

Section: Estimate consensus error.
We present a novel method to establish bounds on the consensus error, which is based on converting ∆ x into rolling sums.
Lemma 4 (Informal). Denote ∆ (k) y = y (k) -1 n π ⊤ A y (k) , ∆ (-1) g = g (0) , ∆ (i) g = g (i+1) -g (i) , ∀i ≥ 0. For any k ≥ 0, it follows that ∆ (k+1) x = -α k i=0 (A -A ∞ ) k+1-i ∆ (i) y , ∆ (k+1) y = k i=-1 (A -A ∞ ) k+1-i D -1 i+1 + O(kβ k A ) ∆ (i) g .
The following lemma tells us how to estimate rolling sums:
Lemma 5 (Rolling sum). For A ∈ R n×n satisfying Assumption 1, the following estimation holds for any matrices ∆ (i) ∈ R n×d and for any K ≥ 0:
K k=0 ∥ k i=0 (A -A ∞ ) k+1-i ∆ (i) ∥ 2 F ≤ s 2 A K i=0 ∥∆ (i) ∥ 2 F ,
where s A is a constant decided by A.
Under the L-smooth assumption, it is straightforward to derive an upper bound for the sum
K k=0 ∥∆ (k) g ∥ 2 F .
Following the sequence of steps ∆ g → ∆ y → ∆ x , Lemmas 4 and 5 together lead to the following consensus lemma.
Lemma 6 (Consensus lemma, informal). With Assumptions 1, 2, 3 and 4, we have
K k=0 E[∥∆ (k+1) x ∥ 2 F ] ≤ C x,y α 4 K k=0 E[∥π ⊤ A D -1 k ∇f (x (k) )∥ 2 F ] + C x,0 α 2 ∥∇f (x (0) )∥ 2 + C x,σ α 2 (K + 1)σ 2 ,
where C x,y , C x,0 and C x,σ are constants.
Achieving linear speedup. Building on Lemmas 2, 3 and 6, we finally achieve the convergence Theorem 2.
Theorem 2 (PULL-DIAG-GT convergence). Under assumptions 1, 2, 3 and 4, when total iteration K > 2κ A θ 2 A 1-β A , there exists a learning rate α (see Section C.8 in Appendix C) such that
1 K K k=0 E∥∇f (w (k) )∥ 2 ≲ σ √ L∆ √ nK + L∆(1 + C A ) K ,
where
w (k) = π ⊤ A x (k) , C
A is a positive constant decided by the mixing matrix A. Proof is in Appendix C. Remark 1. Theorem 2 establishes the first linear speedup convergence utilizing solely row-stochastic mixing matrices. The term C A L∆ K+1 signifies the influence of the network, where C A is a rational function of κ A , β A and θ A .
this section cite: []

Section: Achieving Near-Optimal Convergence Rate
Comparing Theorem 2 with the lower bound in Theorem 1, we identify two key discrepancies preventing PULL-DIAG-GT from achieving the lower bound. First, Theorem 2 relies on Assumption 4, which the lower bound does not require. Second, the constant C A in Theorem 2 depends on θ A , the upper bound of the diagonals, which is absent from the lower bound and can grow arbitrarily large even for fixed β A and κ A . This section introduces a variant of PULL-DIAG-GT to address these discrepancies and achieve the lower bound.
Removing θ A with multiple gossips. The requirement for θ A (and Assumption 4) arises from the use of Diag(A k ) -1 for gradient correction in the PULL-DIAG-GT update (7b). For small values of k, the diagonal elements of A k can become extremely small due to network sparsity, leading to significant instability in the inversion Diag(A k ) -1 during the initial phase. As k increases, Diag(A k ) converges to diag(π A ), which stabilizes the correction. This behavior is formally stated in the following lemma:
Lemma 7. For A ∈ R n×n satisfying Assumption 1, if k ≥ 2 ln(κ A )+2 ln(n) 1-β A , we have [A k ] ii > 0 and [A k ] -1 ii ≤ 2nκ A , ∀i ∈ [n].
Lemma 7 implies that for sufficiently large k, the diagonals of Pull-DIAG-GT with multiple gossips. We now introduce MG-PULL-DIAG-GT to remove Assumption 4 and the the reliance on θ A . Here, "MG" is short for multiple gossips. r) ). The implementation details are provided in Appendix B. It is observed that for each iteration t, recursions (12a) and (12c) incur R rounds of communication, and (12b) requires R samples to compute the minibatch stochastic gradient. To ensure a fair comparison with PULL-DIAG-GT, for each K-iteration run of PULL-DIAG-GT, we run MG-PULL-DIAG-GT for T = K/R iterations, thereby fixing the total number of communication rounds and data samples at K.
x (t+1) = A R (x (t) -αy (t) )(12a)
g (t+1) = 1 R R r=1 ∇F (x (t+1) , ξ (t+1,r) ) (12b
)
y (t+1) = A R (y (t) + D -1 t+1 g (t+1) -D -1 t g (t) ) (12c
)
Here D t = Diag(A tR ), ∀t ≥ 1, D 0 = I n , y (0) = g (0) = 1 R R r=1 ∇F (x (0) , ξ (0,
Achieving optimal convergence rate. Technically, by performing multiple gossip steps, we improve the spectral parameter from β A to β R A , which exponentially reduces all terms associated with decentralized communication. Additionally, by utilizing an R-mini-batch stochastic gradient, we reduce the gradient variance from σ 2 to σ 2 /R. However, reducing the outer iterations from K to K/R may polynomially slow the convergence. By carefully balancing this exponential gain against the polynomial cost with an appropriately chosen R, we can improve overall convergence, ultimately achieving optimal performance: Theorem 3. Suppose Assumptions 1,2 and 3 hold, and set T = K/R. When R = ⌈ 3(1+ln(κ A )+ln(n)) 1-β A ⌉ and α being selected properly, we have
1 T T t=1 E[∥∇f (w (t) )∥ 2 F ] ≲ σ √ L∆ √ nK + (1 + ln(κ A ) + ln(n))L∆ (1 -β A )K ,
where k) . The proof is in Appendix D.
w (k) = π ⊤ A x (
Remark 2. It is observed that Theorem 3 is independent of Assumption 4 and θ A due to the multiple gossip strategy.
Remark 3. When ln(n) is negligible compared to ln(κ A ), Theorem 3 aligns with our lower bound (Theorem 1), making both the lower bound and the algorithm optimal. Otherwise, we say that MG-PULL-DIAG-GT achieves near-optimal complexity (rather than optimal complexity) due to the existence of the logarithmic gap ln(n).
this section cite: []

Section: Experiments
In this section, we empirically validate the theoretical results presented in Theorems 2 and 3. For the stochastic gradient oracle, we focus on the case where each node has access to a finite dataset, and the stochastic gradient is computed with respect to a randomly chosen data sample at each iteration.
To assess the performance of the algorithms, we conduct experiments on a synthetic dataset, MNIST dataset and CIFAR-10 dataset. Implementation details are provided in Appendix E for reference.
this section cite: []

Section: Non-convex Logistic Regression for Classification
In this first group of experiment, we minimize a synthetic nonconvex loss function (Antoniadis et al., 2011;Xin et al., 2021;Alghunaim & Yuan, 2022;Liang et al., 2023) that satisfies the L-smooth property. Our experiments are conducted on directed exponential graphs (Xin et al., 2021;Ying et al., 2021) and directed ring graphs (see Figure A1 in Appendix E.1). For exponential graphs, we evaluate the performance across network sizes of 1 (single node), 2, 8, 16, 128 and 512. For ring graphs, we evaluate the performance across network sizes of 1 (single node), 5, 10, 16.
The results in Figure 2 reveal that, for each fixed topology, the gradient curve decreases proportionally to the square root of the number of nodes after the same number of communication rounds. This numerically validates our Theorem 2 that PULL-DIAG-GT is able to achieve linear speedup.
this section cite: ['b1', 'b0', 'b19', 'b46']

Section: Neural Network for Multi-Class Classification
In the second group of experiment, we focus on a digitclassification task using the MNIST dataset. We evaluate the performance of MG-PULL-DIAG-GT against the vanilla PULL-DIAG-GT across four distinct network topologies: a ring graph, an undirected grid graph, a geometric graph, and a nearest neighbor graph, each comprising 16 nodes. These topologies are illustrated in Figure A1 in Appendix E.1. The weights of the mixing matrices are determined using the Metropolis rule (Nedić et al., 2018), which produces row-stochastic but not doubly-stochastic matrices. Figure 3 demonstrates that MG-PULL-DIAG-GT achieves a consistently faster convergence rate in training loss across all tested topologies, while the corresponding test accuracy is detailed in Figure A2 in Appendix E.
this section cite: ['b26']

Section: Neural Network for Image Classification
In the third set of experiments, we conducted training of the ResNet-18 model (He et al., 2016) on the CIFAR-10 dataset using a distributed approach. Consistent with our previous MNIST dataset experiment, we evaluated and compared the performance of MG-PULL-DIAG-GT against the standard PULL-DIAG-GT over different topologies. illustrates the stability of MG-PULL-DIAG-GT when applied to a larger real-world dataset. In the context of sparse topologies like the ring and grid graphs, MG-PULL-DIAG-GT effectively reduces the influence of sparse structures, resulting in superior performance compared to the vanilla PULL-DIAG-GT. The corresponding test accuracy is detailed in Figure A3 in Appendix E.
this section cite: ['b9']

Section: Conclusions and Limitations
In this paper, we investigate nonconvex, stochastic decentralized optimization over row-stochastic networks. We establish the first lower bound on the convergence rate for this setting. Additionally, we present the first linear speedup convergence rate achieved by PULL-DIAG-GT. To further improve performance, we introduce the multiple gossip technique, leading to the development of MG-PULL-DIAG-GT. This algorithm matches our lower bound up to a logarithmic gap, rendering both the lower bound and the algorithm nearly optimal. Numerical experiments validate our theoretical findings and demonstrate the effectiveness of our approach. A main limitation of our work is that the network impact on PULL-DIAG-GT, such as the explicit influence of θ A , remains unclear, leaving it for future research.
this section cite: []

Section: References
Ref_id:b0 Title: A unified and refined convergence analysis for non-convex decentralized learning Year: (2022)
Ref_id:b1 Title: Penalized likelihood regression for generalized linear models with non-quadratic penalties Year: (2011)
Ref_id:b2 Title: Lower bounds for non-convex stochastic optimization Year: (2019)
Ref_id:b3 Title: Stochastic gradient push for distributed deep learning Year: (2019)
Ref_id:b4 Title: Optimization methods for large-scale machine learning Year: (2018)
Ref_id:b5 Title: Lower bounds for finding stationary points i. Mathematical Programming Year: (2020)
Ref_id:b6 Title: Lower bounds for finding stationary points ii: first-order methods Year: (2021)
Ref_id:b7 Title: Next: In-network nonconvex optimization Year: (2016)
Ref_id:b8 Title: A fast row-stochastic decentralized method for distributed optimization over directed graphs Year: (2023)
Ref_id:b9 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b10 Title: Optimal complexity in non-convex decentralized learning over time-varying networks Year: (2022)
Ref_id:b11 Title: Lower bounds and nearly optimal algorithms in distributed learning with communication compression Year: (2022)
Ref_id:b12 Title: Gossip-based computation of aggregate information Year: (2003)
Ref_id:b13 Title: A unified theory of decentralized sgd with changing topology and local updates Year: (2020)
Ref_id:b14 Title: Lower bounds and optimal algorithms for smooth and strongly convex decentralized optimization over timevarying networks Year: (2021)
Ref_id:b15 Title: Decentralized asynchronous non-convex stochastic optimization on directed graphs Year: (2023)
Ref_id:b16 Title: Accelerated gradient tracking over timevarying graphs for decentralized optimization Year: (2024)
Ref_id:b17 Title: Row-stochastic matrices based distributed optimization algorithm with uncoordinated step-sizes Year: (2019)
Ref_id:b18 Title: Decentralized federated learning over imperfect communication channels Year: (2024)
Ref_id:b19 Title: Towards better understanding the influence of directed networks on decentralized stochastic optimization Year: (2023)
Ref_id:b20 Title: A nesterov-like gradient tracking algorithm for distributed optimization over directed networks Year: (2020)
Ref_id:b21 Title: Optimal complexity in decentralized training Year: (2021)
Ref_id:b22 Title: Distributed optimization over weighted directed graphs using row stochastic matrix Year: (2016)
Ref_id:b23 Title: Distributed optimization over time-varying directed graphs Year: (2014)
Ref_id:b24 Title: Achieving geometric convergence for distributed optimization over timevarying graphs Year: (2017)
Ref_id:b25 Title: Achieving geometric convergence for distributed optimization over timevarying graphs Year: (2017)
Ref_id:b26 Title: Network topology and communication-computation tradeoffs in decentralized optimization Year: (2018)
Ref_id:b27 Title: Distributed optimization over time-varying directed graphs Year: (2015)
Ref_id:b28 Title: Zur theorie der matrices Year: (1907)
Ref_id:b29 Title: Harnessing smoothness to accelerate distributed optimization Year: (2017)
Ref_id:b30 Title: Decentralized stochastic first-order optimization over directed graphs Year: (2020)
Ref_id:b31 Title: Adaptive networks Year: (2014)
Ref_id:b32 Title: Optimal algorithms for smooth and strongly convex distributed optimization in networks Year: (2017)
Ref_id:b33 Title: Optimal algorithms for non-smooth distributed optimization in networks Year: (2018)
Ref_id:b34 Title: Distributed non-convex first-order optimization and information processing: Lower complexity bounds and rate optimal algorithms Year: (2019)
Ref_id:b35 Title: Push-sum distributed dual averaging for convex optimization Year: (2012)
Ref_id:b36 Title: A fast algorithm for optimization over directed graphs Year: (2017)
Ref_id:b37 Title: Add-opt: Accelerated distributed directed optimization Year: (2017)
Ref_id:b38 Title: Distributed nesterov gradient methods over arbitrary graphs Year: (2019)
Ref_id:b39 Title: Distributed stochastic optimization with gradient tracking over strongly-connected networks Year: (2019)
Ref_id:b40 Title: Frost-fast row-stochastic optimization with uncoordinated step-sizes Year: (2019)
Ref_id:b41 Title: An improved convergence analysis for decentralized online stochastic non-convex optimization Year: ()
Ref_id:b42 Title: Augmented distributed gradient methods for multi-agent optimization under uncoordinated constant stepsizes Year: (2015)
Ref_id:b43 Title: Achieving linear speedup with partial worker participation in non-iid federated learning Year: (2021)
Ref_id:b44 Title: A survey of distributed optimization Year: (2019)
Ref_id:b45 Title: Robust federated learning with connectivity failures: A semi-decentralized framework with collaborative relaying Year: (2022)
Ref_id:b46 Title: Exponential graph is provably efficient for decentralized deep training Year: (2021)
Ref_id:b47 Title: On the linear speedup analysis of communication efficient momentum sgd for distributed non-convex optimization Year: (2019)
Ref_id:b48 Title: Decentlam: Decentralized momentum sgd for large-batch deep training Year: (2021)
Ref_id:b49 Title: Revisiting optimal convergence rate for smooth and non-convex stochastic decentralized optimization Year: (2022)
Ref_id:b50 Title: Extrapush for convex smooth decentralized optimization over directed networks Year: (2017)
