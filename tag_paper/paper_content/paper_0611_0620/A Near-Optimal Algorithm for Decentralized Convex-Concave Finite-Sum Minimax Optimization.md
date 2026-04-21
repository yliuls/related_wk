Title: A Near-Optimal Algorithm for Decentralized Convex-Concave Finite-Sum Minimax Optimization
Abstract: In this paper, we study the distributed convex-concave finite-sum minimax optimization over the network, and a decentralized variance-reduced optimistic gradient method with stochastic mini-batch sizes (DIVERSE) is proposed. For the stronglyconvex-strongly-concave objective, it is shown that DIVERSE can achieve a linear convergence rate that depends on the global smoothness parameters, yielding sharper computation and communication complexity bounds than existing results. Furthermore, we also establish the lower complexity bounds, which show that our upper bounds are optimal up to a logarithmic factor in terms of the local incremental first-order oracle calls, the computation rounds, and the communication rounds. Numerical experiments demonstrate that our algorithm outperforms existing methods in practice.

Section: Introduction
In this paper, we consider the following distributed minimax optimization problem
min x∈R dx max y∈R dy f (x, y) := 1 m m i=1 f i (x, y),(1)
where the global objective f : R dx × R dy → R is µ-strongly-convex-µ-strongly-concave. We assume the local function f i : R dx × R dy → R on the i-th node has the finite-sum structure of the form
f i (x, y) := 1 n n j=1 f i,j (x, y),(2)
where each f i,j is smooth. This formulation appears in many fields, including game theory [9,21], robust optimization [10,29], and control theory [45]. In particular, it has received increasing attention recently from the machine learning community, with the rise of adversarial generative networks [6,25], adversarial training [7,49,68,71,77], and reinforcement learning [18,75].
The first-order minimax optimization has been studied extensively over the past decades. Gradient descent ascent (GDA) is a natural extension of gradient descent in minimization problem, which is a cornerstone for many minimax optimization algorithms [11,51]. In the extragradient (EG) method, [24,34,73] an intermediate prediction step is introduced to improve the convergence of GDA, which exhibits the optimal convergence rate under the convex-concave assumption [59,89].
Additionally, the optimistic gradient descent ascent (OGDA) method [61,64] can also achieve the optimal convergence rate, by incorporating the momentum-like term. In the more general variational inequality framework, Kotsalis et al. [35] proposed an optimal operator extrapolation method.
For large-scale optimization arising from machine learning, it is desirable to design efficient stochastic algorithms by exploiting the finite-sum structure in the objective since computing the full gradient is usually expensive. The variance reduction ideas used to achieve the optimal incremental first-order oracle (IFO) complexity bounds in the minimization problem [1,4,20,32,37,67,78,90] have been extended to minimax optimization, though the details are quite involved. Palaniappan and Bach [60] incorporated variance reduction into GDA iteration and provided an catalyst acceleration framework. Chavdarova et al. [16] and Alacaoglu et al. [3] studied the EG method with variance reduction for a specific minimax problem. Alacaoglu and Malitsky [2] introduced a retracted term into the iteration of EG and OGDA, achieving the optimal IFO complexity for the finite-sum minimax problem under the convex-concave setting [26].
The decentralized optimization have been widely studied in recent years. Compared with the centralized scenario, it can avoid the communication and computation bottlenecks for problems over networks [57,82,87], while the algorithm design and analysis are more challenging since each node in a network can only directly share the information with its neighbors. For the minimax optimization, Mukherjee and Chakraborty [55], Beznosikov et al. [13], and Luo and Ye [46] developed EG methods for the decentralized setting and provided the linear convergence rates. Rogozin et al. [65] extended the results to the non-Euclidean mirror prox framework. Kovalev et al. [39] combined variance reduction and the optimistic gradient method [61] within the ADOM framework [38], achieving the best-known upper complexity bound on the computation rounds and the communication rounds.
It is worth noting that exiting decentralized minimax optimization methods require identical minibatch size for all the nodes when constructing the local gradient estimator, which is sample inefficient. Moreover, both the computation complexity and communication complexity of previous works depend on the local smoothness parameters. It remains an open question on how to develop the decentralized minimax optimization algorithm that depends on the global smoothness parameters.
In this paper, we propose a decentralized variance-reduced optimistic gradient method with stochastic mini-batch sizes (DIVERSE) for the minimax problem (1), which can find an ϵ-suboptimal solution with O((mn + min{mnL, √ mn L}/µ) log (1/ϵ)) local incremental first-order oracle (LIFO) calls, Õ((n+L/µ+min{nL, n/m L}/µ) log(1/ϵ) computation rounds, and Õ( √ χL/µ log(1/ϵ)) communication rounds. Here, L is the smoothness parameter of the objective f , L is the mean-squared smoothness parameter of the function set {f i,j } m,n i,j=1 , and χ is the characteristic number of the mixing matrix associated with the network. The corresponding lower bounds have also been established which demonstrate that all the above results are (nearly) optimal. We would like to emphasize that all of our complexity bounds have the global smoothness dependency, which are tighter than existing results that only rely on the local smoothness [39,46,55]. Moreover, the linear convergence guarantee in this paper only requires the global objective function f to be strongly-convex-strongly-concave, and the local component function f i,j (also the local function f i ) can even be nonconvex-nonconcave. This relaxes the assumption for the state-of-the-art linear convergent decentralized algorithm in Kovalev et al. [39] that requires each f i to be strongly-convex-strongly-concave.
this section cite: ['b8', 'b20', 'b9', 'b28', 'b44', 'b5', 'b24', 'b6', 'b48', 'b67', 'b70', 'b76', 'b17', 'b74', 'b10', 'b50', 'b23', 'b33', 'b72', 'b58', 'b88', 'b60', 'b63', 'b34', 'b0', 'b3', 'b19', 'b31', 'b36', 'b66', 'b77', 'b89', 'b59', 'b15', 'b2', 'b1', 'b25', 'b56', 'b81', 'b86', 'b54', 'b12', 'b45', 'b64', 'b38', 'b60', 'b37', 'b0', 'b38', 'b45', 'b54', 'b38']

Section: Preliminaries
In this section, we introduce the problem setup, followed by a review of the related work.
this section cite: []

Section: Problem Setup
We use the bold lowercase letters to represent vectors, e.g., x ∈ R dx and y ∈ R dy , use x i ∈ R 1×dx and y i ∈ R 1×dy to denote the local variables on the i-th node. The bold uppercase letters are used to denote the matrices aggregating the corresponding vectors, such as X = [x 1 ; . . . ; x m ] ∈ R m×dx and Y = [y 1 ; . . . ; y m ] ∈ R m×dy . The bold lowercase letter with a bar presents the average of local variables, e.g., x = 1 m m i=1 x i and ȳ = 1 m m i=1 y i . The notations 1 and 0 are vectors (or matrices) whose entries are all one and zero, respectively. We let I be the identity matrix. The notation ∥ • ∥ represents the Euclidean norm of a vector or the Frobenius norm of a matrix.
For the minimax optimization problem (1), we stack the variables x ∈ R dx and y ∈ R dy as z = [x; y] ∈ R dz , where d z = d x + d y . We further define the gradient operators as g i,j (z) = ∇ x f i,j (x, y) -∇ y f i,j (x, y) , g i (z) = 1 n n j=1 g i,j (z), and g(z) = 1 m m i=1 g i (z) ∈ R dz .
We make the following assumptions for the decentralized minimax optimization problem (1). Assumption 2.1. The global function f (x, y) is µ-strongly-convex-µ-strongly-concave, i.e., the function f (•, y) is µ-strongly convex for all given x ∈ R dx and the function f (x, •) is µ-strongly concave for all given y ∈ R dy . Assumption 2.2 (global smoothness). The global function f is L-smooth, i.e., for all z, z ′ ∈ R dz , there exists a constant L > 0 such that
∥g(z) -g(z ′ )∥ 2 ≤ L 2 ∥z -z ′ ∥ 2 .
Assumption 2.3 (mean-squared smoothness). The function set {f i,j } m,n i,j=1 is L-mean-squared smooth, i.e., for all z, z ′ ∈ R dz , there exists a constant L > 0 such that
1 mn m i=1 n j=1 ∥g i,j (z) -g i,j (z ′ )∥ 2 ≤ L2 ∥z -z ′ ∥ 2 .
Assumption 2.1 is equivalent to the strong monotonicity of the gradient operator g, i.e., for all z, z ′ ∈ R dz , it holds ⟨g(z) -g(z ′ ), zz ′ ⟩ ≥ µ∥z -z ′ ∥ 2 . Note that we have
∥g(z) -g(z ′ )∥ 2 ≤ 1 mn m i=1 n j=1 ∥g i,j (z) -g i,j (z ′ )∥ 2 ≤ L2 ∥z -z ′ ∥ 2
for all z, z ′ ∈ R dz . Therefore, it holds that L ≤ L for the tight parameters L and L satisfying Assumptions 2.2 and 2.3. In fact, L can be arbitrarily larger than L if there are no convexity/concavity assumption for the local functions [48].
Our complexity analysis considers the upper and lower bounds with respect to the global smoothness parameter L and the mean-squared smoothness parameter L. In contrast, existing works for decentralized minimax optimization only consider the local smoothness assumptions. For example, Mukherjee and Chakraborty [55] and Luo and Ye [46] assume there exists a constant L max > 0 such that
∥g i,j (z) -g i,j (z ′ )∥ 2 ≤ L 2 max ∥z -z ′ ∥ 2
for all z, z ′ ∈ R dz , i ∈ [m], and j ∈ [n]; Kovalev et al. [39] assume there exist constants L l > 0 and Ll > 0 such that
∥g i (z) -g i (z ′ )∥ 2 ≤ L 2 l ∥z -z ′ ∥ 2 and 1 n n j=1 ∥g i,j (z) -g i,j (z ′ )∥ 2 ≤ L2 l ∥z -z ′ ∥ 2
for all z, z ′ ∈ R dz and i ∈ [m]. Noting that the constants L l and Ll are determined by the "worst" local (component) function so that we can verify that L ≤ L l ≤ L max and L ≤ Ll ≤ L max for the tight smoothness parameters that satisfy the above assumptions [48]. The examples in Appendix A demonstrate that the magnitude of these smoothness parameters can differ significantly under data heterogeneity.
In decentralized optimization, each node can only directly communicate with its neighbors. The communication step is usually expressed based on a mixing matrix W ∈ R m×m , which satisfies the following standard assumption [28,66,87]. Assumption 2.4. Let W ∈ R m×m be a mixing matrix associated with a network. We assume (a) W is symmetric with w i,j ≥ 0 for all i, j, and w i,j ̸ = 0 if and only if nodes i and j are connected or i = j; (b) 0 ⪯ W ⪯ I, W ⊤ 1 = W1 = 1, and null(I -W) = span(1).
Table 1: We summarize the complexity for finding the ϵ-suboptimal solution of problem (1). We use the notation Õ(•) to hide the logarithmic terms with respect to m, n, µ, and the smoothness parameters. Note that the computation rounds may not be proportional to the LIFO calls, since distributed algorithms include the scheme of partial participated computation.
this section cite: ['b47', 'b54', 'b45', 'b38', 'b47', 'b27', 'b65', 'b86', 'b0']

Section: Algorithms

this section cite: []

Section: LIFO Calls Computation Rounds Communication Rounds
GT-EG [55] O mn χLmax µ
4/3 log 1 ϵ O n χLmax µ 4/3 log 1 ϵ O χLmax µ 4/3 log 1 ϵ MC-SVRE [46] O mn + m √ nLmax µ log 1 ϵ O n + √ nLmax µ log 1 ϵ Õ √ χ n + √ nLmax µ log 1 ϵ OADSVI [39] O mn + m √ n Ll µ log 1 ϵ O n + √ n Ll µ log 1 ϵ O √ χLl µ log 1 ϵ DIVERSE Theorem 3.7 O mn + min{mnL, √ mn L} µ log 1 ϵ Õ n + L µ + min{nL, n/m L} µ log 1 ϵ Õ √ χL µ log 1 ϵ
Lower Bounds Theorem 4.2-4.4
Ω mn + min{mnL, √ mn L} µ log 1 ϵ Ω n + L µ + min{nL, n/m L} µ log 1 ϵ Ω √ χL µ log 1 ϵ
Assumption 2.4 indicates that 1 -λ 2 (W) > 0, where λ 2 (W) is the second largest eigenvalue of W ∈ R m×m . Thus, we can define the characteristic number χ := 1/(1 -λ 2 (W)).
In this paper we consider the ϵ-suboptimal solution of problem (1), i.e., the point z = (x, y) such that
∥x -x * ∥ 2 + ∥y -y * ∥ 2 ≤ ϵ,
where (x * , y * ) is the solution of problem (1) that satisfies f (x * , y ′ ) ≤ f (x * , y * ) ≤ f (x ′ , y * ) for all x ′ ∈ R dx and y ′ ∈ R dy . Noting that the solution (x * , y * ) is unique under the strongly-convexstrongly-concave assumption.
this section cite: ['b54']

Section: Related Work
Significant advancement has been made for decentralized optimization over the last decade. For the minimization problem, the convergence of decentralized gradient descent (DGD) with decaying step sizes has been established by Duchi et al. [22],Tsianos and Rabbat [74], and Jakovetić et al. [31]. The gradient tracking technique was introduced in Nedic et al. [56], Qu and Li [63], and Song et al. [69], so that constant step size can be utilized and linear convergence was achieved for strongly-convex objective. Pu and Nedić [62], Koloskova et al. [33], Ye and Chang [85] further investigated the convergence of gradient tracking under stochastic setting. Scaman et al. [66], Kovalev et al. [38], and Ye et al. [87] introduced the multi-consensus steps by Chebyshev acceleration [5,43] to further improve the communication complexity. For the objective with the finite-sum structure, Xin et al. [81], Ye et al. [86], Hendrikx et al. [28], and Li et al. [41] integrated the variance reduction techniques to improve the computational efficiency of the algorithms. In recent works [44,48,52], different types of smoothness parameters have been considered and sharper complexity bounds have been established for the decentralized finite-sum minimization problems.
For decentralized minimax optimization, Mukherjee and Chakraborty [55] proposed the GT-EG method by combining gradient tracking with EG, proving its linear convergence under the stronglyconvex-strongly-concave assumption. Later, Luo and Ye [46] improved the decentralized EG method by incorporating variance reduction [2] and multi-consensus steps [5,43], achieving better complexity bound on the LIFO calls. It is worth noting that the convergence for both of these methods require the assumption that each component function f i,j is L max -smooth. In a seminal work, Kovalev et al. [39] considered the relaxed conditions that only assume each local function f i is L l -smooth and each local function set {f i,j } n j=1 is Ll -mean-squared smooth. The authors introduced an extra momentum term into the variance-reduced OGDA method [2], so that they could take the advantage of mini-batch sampling to construct an accurate stochastic gradient estimator, leading to improved computation complexity and communication complexity. The lower bounds were also established therein to justify the optimality of their algorithm with respect to the local smoothness parameters L l and Ll . However, none of the previous works on decentralized minimax optimization [39,46,55] has considered the potentially tighter complexity bounds with respect to the global smoothness in Assumptions 2.2 and 2.3, which will be well-addressed in this paper. We compare our theoretical results with related work in Table 1.
Algorithm 1 FastMix(U 0 , W, R) 1: Initialize: U -1 = U 0 , η U = 1- √ 1-λ 2 2 (W) 1+ √ 1-λ 2 2 (W)
2: for r = 0, 1, . . . , R -1 do 3:
U r+1 = (1 + η U )WU r -η U U r-1 4: end for 5: Output: U R Algorithm 2 DIVERSE 1: Input: initial point z 0 , step size η, mini-batch size b, probability p ∈ [0, 1], parameters α, β ∈ [0, 1], mixing matrix W, iteration numbers K, communication rounds R 2: V -1 = V 0 = Z -1 = Z 0 = 1z 0 , S -1 = ∆ -1 = 0 3: for k = 0, 1, 2, . . . , K -1 do 4: for i = 1, 2, . . . , m in parallel 5: ξ k i,j i.i.d ∼ Bernoulli(q) with q = b/(mn)
6:
δ k i = g i (v k-1 i ) + 1 n n j=1 ξ k i,j q g i,j (z k i ) -g i,j (v k-1 i ) + α g i,j (z k i ) -g i,j (z k-1 i ) 7: end for 8: S k = FastMix(S k-1 + ∆ k -∆ k-1 , W, R) 9: Z k+1 = FastMix((1 -β)Z k + βV k -ηS k , W, R) 10: V k+1 = FastMix(Z k , W, R) with probability p, V k
with probability 1 -p 11: end for 12: Output:
z out i = z K i
this section cite: ['b21', 'b73', 'b30', 'b55', 'b62', 'b68', 'b61', 'b32', 'b84', 'b65', 'b37', 'b86', 'b4', 'b42', 'b80', 'b85', 'b27', 'b40', 'b43', 'b47', 'b51', 'b54', 'b45', 'b1', 'b4', 'b42', 'b38', 'b1', 'b38', 'b45', 'b54']

Section: Algorithm and Complexity Analysis
The proposed decentralized variance-reduced optimistic gradient method with stochastic mini-batch sizes (DIVERSE) is described in Algorithm 2, which is based on a novel sampling strategy and the subroutine of multi-consensus steps (Algorithm 1). The details of the algorithm and its complexity analysis are presented in Sections 3.1 and 3.2, respectively.
this section cite: []

Section: Algorithm Design
Recall that the standard OGDA update [19,54,61] is given by
z k+1 = z k -η(g(z k ) + g(z k ) -g(z k-1 ) optimistic gradient ),
where η > 0 is the step size. To improve the computational efficiency by using the finite-sum structure in the local function, DIVERSE constructs the variance-reduced optimistic gradient estimator at node i as follows
δ k i = g i (v k-1 i ) + 1 n n j=1 ξ k i,j q g i,j (z k i ) -g i,j (v k-1 i ) + α g i,j (z k i ) -g i,j (z k-1 i ) ,(3)
where ξ k i,j
i.i.d
∼ Bernoulli(q) with q = b/(mn) and v k-1 i is the snapshot point, and α > 0 is the momentum parameter. The distribution of ξ k i,j means we only need to compute the gradient operator g i,j in equation (3) when ξ k i,j = 1. Note that the snapshot point v k+1 i is updated with probability p in each iteration (see Line 10 of Algorithm 2), which implies the term g i (v k-1 i ) in equation ( 3) can be reused with probability 1 -p. Therefore, the expected LIFO calls of the algorithm in each iteration is O(mnp + (1 -p)b), which is much more efficient than the cost of accessing the full gradient if we take p ≪ 1 and b ≪ mn.
The main difference between DIVERSE and exiting decentralized minimax optimization methods [39,46,55] is that the mini-batch size for the local gradient estimator δ k i in equation ( 3) is not required to be fixed since the variables {ξ k i,j } m,n i,j=1 are random. Therefore, the behaviors of all m nodes are similar to the large mini-batch sampling on a single machine. Besides, the steps of gradient tracking and multi-consensus in Lines 8 and 9 of Algorithm 2 ensures that the local variables are sufficiently close to each other, resulting in the sharper complexity bounds with respect to the global smoothness.
this section cite: ['b18', 'b53', 'b60', 'b38', 'b45', 'b54']

Section: Complexity Analysis
For the convergence analysis of DIVERSE (Algorithm 2), define the following Lyapunov function based on the mean vectors as follows
Φ k := 1 η + 3µ 2 ∥z k -z * ∥ 2 + β η ∥z k -vk-1 ∥ 2 + 1 8η ∥z k -zk-1 ∥ 2 + 2⟨g(z k-1 ) -g(z k ), zk -z * ⟩ + β + ηµ pη ∥v k -z * ∥ 2 .
It is not hard to verify that the Lyapunov function Φ k is always non-negative for all η ≤ 1/(4L) (see Appendix B.1).
We first consider the case of L ≤ √ mnL, in which the Lyapunov function satisfies the following relation. Then it holds that
E Φ k+1 ≤ αE Φ k + C 1 E ∥Z k -1z k ∥ 2 + E ∥Z k-1 -1z k-1 ∥ 2 + E ∥V k-1 -1v k-1 ∥ 2 ,
where
C 1 = (1 + α 2 ) 12nη L2 /b + 6n L2 /µ .
To characterize the convergence of the multi-consensus steps (Algorithm 1), define
ρ := √ 14(1 -(1 -1/ √ 2) 1 -λ 2 (W)) R .
We have ρ < 1 if R is sufficient large. See more properties of Algorithm 1 in Appendix B.2.
We then bound the consensus error as follows.
Lemma 3.2. Under the settings of Lemma 3.1, we have
E ∥Z k+1 -1z k+1 ∥ 2 ≤ 3ρ 2 (1 -β) 2 E ∥Z k -1z k ∥ 2 + 3ρ 2 β 2 E ∥V k -1v k ∥ 2 + 3ρ 2 η 2 E ∥S k -1s k ∥ 2
and
E ∥V k+1 -1v k+1 ∥ 2 ≤ pρ 2 E ∥Z k -1z k ∥ 2 + (1 -p)E ∥V k -1v k ∥ 2 .
Noting that the upper bound in Lemma 3.2 depends on the term ∥S k -1s k ∥ 2 , the consensus error for S can be bounded as follows.
Lemma 3.3. Under the settings of Lemma 3.1, we have
E ∥S k+1 -1s k+1 ∥ 2 ≤ C 2 ρ 2 E ∥Z k -1z k ∥ 2 + E ∥Z k-1 -1z k-1 ∥ 2 + E ∥V k -1v k ∥ 2 + E ∥V k-1 -1v k-1 ∥ 2 + C 3 ρ 2 E ∥S k -1s k ∥ 2 + C 4 ρ 2 E Φ k+1 + E Φ k + E Φ k-1 ,
where C 2 = 270m 2 n 2 L2 , C 3 = 180m 2 n 2 L2 η 2 + 2, C 4 = 60(16η + η/β)m 3 n 2 L2 , and Φ -1 = 0.
Remark 3.4. Lemma 3.3 shows that the upper bound of E ∥S k+1 -1s k+1 ∥ 2 does not only depend on the consensus error at the k-th iteration, but also on that of the (k -1)-th iteration. In contrast, the consensus error in the decentralized minimization problem only depends on the term related to the previous iteration [40,44,48,87]. The difference poses a challenge in the analysis, which requires us to develop a novel inductive proof technique.
Applying Lemmas 3.1-3.3, we obtain the linear convergence for the Lyapunov function and consensus errors. Lemma 3.5. Under the settings of Lemma 3.1, we have
E Φ k ≤ αk Φ 0 , E ∥Z k -1z k ∥ 2 ≤ 1 - α 4C 1 αk+1 Φ 0 , E ∥V k -1v k ∥ 2 ≤ 1 - α 4C 1 αk+1 Φ 0 , and E ∥S k -1s k ∥ 2 ≤ 1 - α 4η 2 C 1 αk+1 Φ 0 ,
where α = max 1 -µη/8, 1 -pηµ/(2(β + ηµ)) .
According to the parameter settings in Lemma 3.1, the linear convergence rate α achieved by Lemma 3.5 has the order of Θ(1 -µ/L), which depends on the global smoothness. The expected overall LIFO complexity to achieve the ϵ-suboptimal solution is O((mn + √ mn L/µ) log(1/ϵ)), matching the complexity of variance-reduced EG/OGDA on a single machine [2].
We then consider the case of L ≤ L/ √ mn. Note that under the setting of Lemma 3.1, one has b ≥ mn in this case, which motivates us to use the exact local gradients. That is, we set p = 0 and b = mn in Algorithm 2 when L ≤ L/ √ mn, which leads to ξ k i,j = q = 1 and
δ k i = g i (z k i ) + α g i (z k i ) -g i (z k-1 i ) .
Hence, the snapshot v k i is unnecessary, so we set β = 0 and define the simplified Lyapunov function
Ψ k := 1 η + 3µ 2 ∥z k -z * ∥ 2 + 3 4η ∥z k -zk-1 ∥ 2 + 2⟨g(z k-1 ) -g(z k ), zk -z * ⟩.
Similar to the analysis of Lemma 3.5, the linear convergence can also be achieved with respect to the global smoothness. Lemma 3.6. Under Assumptions 2.1, 2.2, 2.3 and 2.4 with 0 < µ < L ≤ L/ √ mn, we run Algorithm 2 with η = 1/(16L), β = p = 0, b = mn, α = 1 -µη, and R = O( √ χ log(mn L/µ)).
Then it holds that
Ψ k ≤ 1 - µη 2 k Ψ 0 , ∥Z k -1z k ∥ 2 ≤ µ 2 η 48n L2 1 - µη 2 k+1 Ψ 0 , and ∥S k -1s k ∥ 2 ≤ µ 2 48ηn L2 1 - µη 2 k+1 Ψ 0 .
According to Lemma 3.6, we achieve the LIFO complexity of O((mnL/µ) log(1/ϵ)). It is worth noting that the expected overall LIFO complexity of O((mn + √ mn L/µ) log(1/ϵ)) achieved by variance reduction (under the parameter settings in Lemma 3.1) is worse than the LIFO complexity achieved by iterations with exact local gradients in the case of L ≤ L/ √ mn. Similar phenomenon is also observed by Luo et al. [48] in nonconvex minimization. Our result implies that the trade-off between variance-reduced gradient estimator and the exact gradient is also necessary in minimax optimization.
Combining the results of Lemmas 3.5 and 3.6 yields the following upper complexity bounds.
Theorem 3.7. Under Assumptions 2.1, 2.2, 2.3 and 2.4 with 0 < µ < L ≤ L, running DIVERSE (Algorithm 2) with appropriate parameter settings can find an ϵ-suboptimal solution at each node, with the expected LIFO complexity of O((mn + min{mnL, √ mn L}/µ) log(1/ϵ)), the expected computation rounds of Õ((n + L/µ + min{nL, n/m L}/µ) log(1/ϵ)), and the communication rounds of Õ( √ χL/µ log(1/ϵ)).
As demonstrated in Table 1, all of our upper bounds in Theorem 3.7 are sharper than state-of-the-art results since we have L ≤ L l ≤ L max and L ≤ Ll ≤ L max for the tight smoothness parameters. Additionally, our LIFO complexity depends on √ m in the case of L ≤ √ mnL, which is better than existing results that always depends on m. These improvements essentially rely on the sampling strategy that does not fix the mini-batch size on different nodes, thereby allowing partial participation to reduce computational costs. Remark 3.8. The computation rounds in Algorithm 2 depend on E[max i∈[m] n j=1 ξ k i,j ], which may not be proportional to the LIFO calls. We upper bound this quantity by using the locally sub-Gaussian property, which simplifies the analysis in Liu et al. [44] and Luo et al. [48], see Lemma C.1 for details.
this section cite: ['b39', 'b43', 'b47', 'b86', 'b1', 'b47', 'b43', 'b47']

Section: The Lower Complexity Bounds
In this section, we establish the lower complexity bounds of the first-order methods for the decentralized finite-sum minimax optimization. Specifically, we consider the local incremental first-order oracle algorithms as follows.
Definition 4.1. A local incremental first-order oracle (LIFO) algorithm over a network of m nodes satisfies the following constraints:
• Local memory: Each node i stores vectors in local memories M x i,t and M y i,t at time t > 0. The local memories are updated through local computation or local communication, i.e., for all i ∈ [m], it holds
M x i,t ⊆ M comp,x i,t ∪ M comm,x i,t
and M y i,t ⊆ M comp,y i,t ∪ M comm,y i,t .
• Local computation: At time t, each node i can query the local first-order oracles ∇ x f i,j (x, y) and ∇ y f i,j (x, y) for any x ∈ M x i,t-1 and y ∈ M y i,t-1 . Additionally, the local computational memories M comp,x i,t and M comp,y i,t satisfy M comp,x i,t = Span {x, ∇ x f i,j (x, y) : x ∈ M x i,t-1 } and M comp,y i,t = Span {y, ∇ y f i,j (x, y) : y ∈ M y i,t-1 } .
• Local communication: At time t, each node i can communicate with its neighbours N (i). For all i ∈ [m], the communication memories are defined as M comm,x i,t = Span( j∈N (i),τ M x j,t-τ ), and M comm,y i,t = Span( j∈N (i),τ M y j,t-τ ), where τ is a delay parameter satisfying τ < t.
• Output value: Each node i specifies local outputs from its memory at time t, that is, for all i ∈ [m],
we have x t i ∈ M x i,t and y t i ∈ M y i,t .
The definition of the above algorithm class follows the standard settings in the studies of decentralized optimization [8,28,44,48,66]. Compared with the algorithm classes defined by Kovalev et al. [39], we remove the requirement that all nodes must access their stochastic local gradients with the identical mini-batch size per iteration, so our algorithm class also contains the partial participated computation schemes.
The lower complexity bounds on the LIFO calls, the computation rounds, and the communication rounds are presented in the following three theorems.
Theorem 4.2. For the parameters L ≥ L, L/µ > 2, and ϵ < 0.003, there exists hard instances satisfying Assumptions 2.1-2.4. In order to find an ϵ-suboptimal solution, the LIFO calls of any LIFO algorithm is lower bounded by Ω(mn + min{mnL, √ mn L}/µ log(1/ϵ)).
Theorem 4.3. For the parameters L ≥ L, L/µ > 2, and ϵ < 0.003, there exists hard instances satisfying Assumptions 2.1-2.4. In order to find an ϵ-suboptimal solution, the computation rounds of any LIFO algorithm is lower bounded by Ω(n + (L/µ + min{nL, n/m L}/µ) log(1/ϵ)).
Theorem 4.4. For the parameters L ≥ L ≥ 2µ > 0, and m ≥ 2, n ∈ N, there exists a hard instance satisfying Assumptions 2.1-2.4 with λ 2 (W) ∈ [0, cos(π/m)]. In order to find an ϵ-suboptimal solution, the communication rounds of any LIFO algorithm is lower bounded by Ω( √ χL/µ log(1/ϵ)).
0 1 2 3 4 5 LIFO calls ×10 6 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 LIFO calls ×10 6 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 LIFO calls ×10 6 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 LIFO calls ×10 6 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE (a) a9a (b) w8a (c) ijcnn1 (d) cod-rna Figure 1: Performance comparison with respect to LIFO calls across different datasets. 0.0 0.2 0.4 0.6 0.8 1.0 Computation rounds ×10 5 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0.0 0.2 0.4 0.6 0.8 1.0 Computation rounds ×10 5 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0.0 0.2 0.4 0.6 0.8 1.0 Computation rounds ×10 5 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0.0 0.2 0.4 0.6 0.8 1.0 Computation rounds ×10 5 10 8 10 7 10 6 10 5 10 4 10 3 10 2 10 1 10 0 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE (a) a9a (b) w8a (c) ijcnn1 (d) cod-rna 0.0 0.2 0.4 0.6 0.8 1.0 Communication rounds ×10 5 10 8 10 6 10 4 10 2 10 0 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 Communication rounds ×10 4 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 Communication rounds ×10 4 10 9 10 7 10 5 10 3 10 1 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE 0 1 2 3 4 5 Communication rounds ×10 4 10 7 10 6 10 5 10 4 10 3 10 2 10 1 10 0 z z * 2 GT-EG MC-SVRE OADSVI DIVERSE (a) a9a (b) w8a (c) ijcnn1 (d) cod-rna Figure 3: Performance comparison with respect to communication rounds across different datasets.
The above theorems indicate that the upper complexity bounds provided in Theorem 3.7 are optimal (up to a logarithmic factor). Our lower bounds hold for the decentralized finite-sum minimax optimization under the general smoothness settings. Specifically, the results in Theorems 4.2-4.4 hold for all L and L such that 0 < L ≤ L. In contrast, the existing lower bounds [39] consider the local smoothness parameters L l and Ll (see Section 2.1), and their analysis requires the additional condition √ nL l = Ll .
this section cite: ['b7', 'b27', 'b43', 'b47', 'b65', 'b38', 'b38']

Section: Numerical Experiments
In this section, numerical experiments are conducted to evaluate the performance of Algorithm 2. We consider the problem of robust regularized linear regression [12,27,39,50], which is formulated as
min x∈R d max y∈R d 1 2N N i=1 x ⊤ (a i + y) -b i 2 + r 1 2 ∥x∥ 2 - r 2 2 ∥y∥ 2 ,
where x is the weight of the model, y is the adversarial noise, {(a i , b i )} N i=1 is the training dataset, and r 1 , r 2 are regularization parameters. We consider the undirected ring network with m = 50 nodes and each node has n training samples. Therefore, the total number of samples is N = mn. The mixing matrix with Metropolis-Hastings weights [80] is used for communication steps. The regularization parameters are set to be r 1 = r 2 = 0.2. The numerical experiments are conducted on datasets a9a, w8a, ijcnn1, and cod-rna, from the LIBSVM repository [14]. We compare the proposed DIVERSE (Algorithm 2) with the baseline methods including GT-EG [55], MC-SVRE [46], and OADSVI [39, Algorithm 1]. The parameters of these algorithms are set according to the theoretical analysis or the recommended settings by the authors [39,46,55]. Specifically, the parameter b in the DIVERSE is set to be 128 and the fixed batch size for each node in OADSVI is set to be 3. The best performance step sizes from {0.1, 0.05, 0.01} are used, up to the algorithms and the datasets.
The experimental results are shown in Figures 123. It can be observed that the proposed DIVERSE outperforms all the tested methods in terms of LIFO calls, the computation rounds, and the communication rounds, which validates our theoretical results. The deterministic method GT-EG [55] and the stochastic method MC-SVRE [46] require much more communication rounds than other methods. This is because GT-EG does not include Chebyshev acceleration in its communication protocol and MC-SVRE cannot benefit from the communication efficiency by the mini-batch sampling. Additionally, the LIFO complexity of DIVERSE is significantly superior to all the baselines, since it is the only one that uses stochastic mini-batch sizes, benefiting from partially participated computations.
this section cite: ['b11', 'b26', 'b38', 'b49', 'b79', 'b13', 'b54', 'b45', 'b38', 'b45', 'b54', 'b54', 'b45']

Section: Conclusion
This paper proposes variance-reduced optimistic gradient method with stochastic mini-batch sizes for decentralized convex-concave finite-sum minimax problem. We establish the linear convergence rate with global smoothness parameters dependency for the strongly-convex-strongly-concave objective, which is shaper than existing results that only consider the local smoothness. Lower complexity bounds are constructed to show the near optimality of our method. The efficiency of the proposed method is also validated through numerical experiments. For future direction, we would like to extend the ideas to solve the decentralized minimax problem with different constants of strong convexity and strong concavity [26,36,42,47,53,70,76,84]. We can also study the global smoothness dependency in decentralized nonconvex minimax optimization [17,23,30,72,79,83,[91][92][93].
this section cite: ['b25', 'b35', 'b41', 'b46', 'b52', 'b69', 'b75', 'b83', 'b16', 'b22', 'b29', 'b71', 'b78', 'b82', 'b90', 'b91', 'b92']

Section: References
Ref_id:b0 Title: A lower bound for the optimization of finite sums Year: (2015)
Ref_id:b1 Title: Stochastic variance reduction for variational inequality methods Year: (2022)
Ref_id:b2 Title: Forward-reflected-backward method with variance reduction Year: (2021)
Ref_id:b3 Title: The first direct acceleration of stochastic gradient methods Year: (2018)
Ref_id:b4 Title: Chebyshev acceleration of iterative refinement Year: (2014)
Ref_id:b5 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b6 Title: Recent advances in adversarial training for adversarial robustness Year: (2021)
Ref_id:b7 Title: On the complexity of finite-sum smooth optimization under the Polyak-Łojasiewicz condition Year: (2024)
Ref_id:b8 Title: Dynamic noncooperative game theory Year: (1998)
Ref_id:b9 Title:  Year: (2009)
Ref_id:b10 Title: Mixed equilibria and dynamical systems arising from fictitious play in perturbed games Year: (1999)
Ref_id:b11 Title: Distributed saddle-point problems under data similarity Year: (2021)
Ref_id:b12 Title: Decentralized local stochastic extra-gradient for variational inequalities Year: (2022)
Ref_id:b13 Title: LIBSVM: a library for support vector machines Year: (2011)
Ref_id:b14 Title: Locally sub-Gaussian random variable and the strong law of large numbers Year: (2006)
Ref_id:b15 Title: Reducing noise in GAN training with variance reduced extragradient Year: (2019)
Ref_id:b16 Title: An efficient stochastic algorithm for decentralized nonconvex-strongly-concave minimax optimization Year: (1990)
Ref_id:b17 Title: Policy evaluation with temporal differences: A survey and comparison Year: (2014)
Ref_id:b18 Title: Training GANs with optimism Year: (2018)
Ref_id:b19 Title: SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives Year: (2014)
Ref_id:b20 Title: Minimax and applications Year: (1995)
Ref_id:b21 Title: Dual averaging for distributed optimization: Convergence analysis and network scaling Year: (2011)
Ref_id:b22 Title: Decentralized stochastic gradient descent ascent for finite-sum minimax problems Year: (2022)
Ref_id:b23 Title: A variational inequality perspective on generative adversarial networks Year: (2019)
Ref_id:b24 Title: Generative adversarial nets Year: (2014)
Ref_id:b25 Title: Lower complexity bounds of finite-sum optimization problems: The results and construction Year: (2024)
Ref_id:b26 Title: Statistically preconditioned accelerated gradient method for distributed optimization Year: (2020)
Ref_id:b27 Title: An optimal algorithm for decentralized finite-sum optimization Year: (2021)
Ref_id:b28 Title: Minimax programming as a tool for studying robust multi-objective optimization problems Year: (2022)
Ref_id:b29 Title: Near-optimal decentralized momentum method for nonconvex-PL minimax problems Year: (2023)
Ref_id:b30 Title: Fast distributed gradient methods Year: (2014)
Ref_id:b31 Title: Accelerating stochastic gradient descent using predictive variance reduction Year: (2013)
Ref_id:b32 Title: An improved analysis of gradient tracking for decentralized machine learning Year: (2021)
Ref_id:b33 Title: The extragradient method for finding saddle points and other problems Year: (1976)
Ref_id:b34 Title: Simple and optimal methods for stochastic variational inequalities, I: operator extrapolation Year: (2022)
Ref_id:b35 Title: The first optimal algorithm for smooth and stronglyconvex-strongly-concave minimax optimization Year: (2022)
Ref_id:b36 Title: Don't jump through hoops and remove those loops: SVRG and Katyusha are better without the outer loop Year: (2020)
Ref_id:b37 Title: Optimal and practical algorithms for smooth and strongly convex decentralized optimization Year: (2020)
Ref_id:b38 Title: Optimal algorithms for decentralized stochastic variational inequalities Year: (2022)
Ref_id:b39 Title: DESTRESS: Computation-optimal and communicationefficient decentralized nonconvex finite-sum optimization Year: (2022)
Ref_id:b40 Title: Variance reduced EXTRA and DIGing and their optimal acceleration for strongly convex decentralized optimization Year: (2022)
Ref_id:b41 Title: Near-optimal algorithms for minimax optimization Year: (2020)
Ref_id:b42 Title: Accelerated linear iterations for distributed averaging Year: (2011)
Ref_id:b43 Title: Decentralized convex finite-sum optimization with better dependence on condition numbers Year: (2024)
Ref_id:b44 Title: Minimax approaches to robust model predictive control Year: (2003)
Ref_id:b45 Title: Decentralized stochastic variance reduced extragradient method Year: (2022)
Ref_id:b46 Title: Near optimal stochastic algorithms for finite-sum unbalanced convex-concave minimax optimization Year: (2021)
Ref_id:b47 Title: On the complexity of decentralized finite-sum nonconvex optimization Year: (2022)
Ref_id:b48 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b49 Title: Globally convergent newton methods for ill-conditioned generalized self-concordant losses Year: (2019)
Ref_id:b50 Title: Cycles in adversarial regularized learning Year: (2018)
Ref_id:b51 Title: Decentralized finite-sum optimization over time-varying networks Year: (2024)
Ref_id:b52 Title: Decentralized saddle-point problems with different constants of strong convexity and strong concavity Year: (2024)
Ref_id:b53 Title: A unified analysis of extra-gradient and optimistic gradient methods for saddle point problems: Proximal point approach Year: (2020)
Ref_id:b54 Title: A decentralized algorithm for large scale min-max problems Year: (2020)
Ref_id:b55 Title: Achieving geometric convergence for distributed optimization over time-varying graphs Year: (2017)
Ref_id:b56 Title: Network topology and communicationcomputation tradeoffs in decentralized optimization Year: (2018)
Ref_id:b57 Title: Lectures on convex optimization Year: (2018)
Ref_id:b58 Title: Lower complexity bounds of first-order methods for convex-concave bilinear saddle-point problems Year: (2021)
Ref_id:b59 Title: Stochastic variance reduction methods for saddlepoint problems Year: (2016)
Ref_id:b60 Title: A modification of the arrow-hurwitz method of search for saddle points Year: (1980)
Ref_id:b61 Title: Distributed stochastic gradient tracking methods Year: (2021)
Ref_id:b62 Title: Harnessing smoothness to accelerate distributed optimization Year: (2017)
Ref_id:b63 Title: Optimization, learning, and games with predictable sequences Year: (2013)
Ref_id:b64 Title: Decentralized saddle point problems via non-Euclidean mirror prox Year: (2024)
Ref_id:b65 Title: Optimal algorithms for smooth and strongly convex distributed optimization in networks Year: (2017)
Ref_id:b66 Title: Minimizing finite sums with the stochastic average gradient Year: (2017)
Ref_id:b67 Title: Universal adversarial training Year: (2020)
Ref_id:b68 Title: Optimal gradient tracking for decentralized optimization Year: (2024)
Ref_id:b69 Title: On accelerated methods for saddle-point problems with composite structure Year: (2021)
Ref_id:b70 Title: Ensemble adversarial training: Attacks and defenses Year: (2017)
Ref_id:b71 Title: Decentralized min-max optimization: Formulations, algorithms and applications in network poisoning attack Year: (2020)
Ref_id:b72 Title: On linear convergence of iterative methods for the variational inequality problem Year: (1995)
Ref_id:b73 Title: Distributed strongly convex optimization Year: (2012)
Ref_id:b74 Title: Analysis of temporal-diffference learning with function approximation Year: (1996)
Ref_id:b75 Title: Improved algorithms for convex-concave minimax optimization Year: (2020)
Ref_id:b76 Title: Fast is better than free: Revisiting adversarial training Year: (2020)
Ref_id:b77 Title: Tight complexity bounds for optimizing composite objectives Year: (2016)
Ref_id:b78 Title: A faster decentralized algorithm for nonconvex minimax problems Year: (2021)
Ref_id:b79 Title: A scheme for robust distributed sensor fusion based on average consensus Year: (2005)
Ref_id:b80 Title: Variance-reduced decentralized stochastic optimization with accelerated convergence Year: (2020)
Ref_id:b81 Title: A general framework for decentralized optimization with first-order methods Year: (2020)
Ref_id:b82 Title: Decentralized gradient descent maximization method for composite nonconvex strongly-concave minimax problems Year: (2024)
Ref_id:b83 Title: A catalyst framework for minimax optimization Year: (2020)
Ref_id:b84 Title: Snap-shot decentralized stochastic gradient tracking methods Year: (2022)
Ref_id:b85 Title: PMGT-VR: A decentralized proximal-gradient algorithmic framework with variance reduction Year: (2020)
Ref_id:b86 Title: Multi-consensus decentralized accelerated gradient descent Year: (2023)
Ref_id:b87 Title: Revisiting optimal convergence rate for smooth and non-convex stochastic decentralized optimization Year: (2022)
Ref_id:b88 Title: On lower iteration complexity bounds for the convex concave saddle point problems Year: (2022)
Ref_id:b89 Title: Linear convergence with condition number independent access of full gradients Year: (2013)
Ref_id:b90 Title: Taming communication and sample complexities in decentralized policy evaluation for cooperative multi-agent reinforcement learning Year: (2021)
Ref_id:b91 Title: Jointly improving the sample and communication complexities in decentralized stochastic minimax optimization Year: (2024)
Ref_id:b92 Title: Can decentralized stochastic minimax optimization algorithms converge linearly for finite-sum nonconvex-nonconcave problems Year: (2023)
