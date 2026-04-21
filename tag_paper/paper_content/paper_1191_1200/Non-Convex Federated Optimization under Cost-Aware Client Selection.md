Title: NON-CONVEX FEDERATED OPTIMIZATION UNDER COST-AWARE CLIENT SELECTION
Abstract: Different federated optimization algorithms typically employ distinct clientselection strategies: some methods communicate only with a randomly sampled subset of clients at each round, while others need to periodically communicate with all clients or use a hybrid scheme that combines both strategies. However, existing metrics for comparing optimization methods typically do not distinguish between these strategies, which often incur different communication costs in practice. To address this disparity, we introduce a simple and natural model of federated optimization that quantifies communication and local computation complexities. This new model allows for several commonly used client-selection strategies and explicitly associates each with a distinct cost. Within this setting, we propose a new algorithm that achieves the best-known communication and local complexities among existing federated optimization methods for non-convex optimization. This algorithm is based on the inexact composite gradient method with a carefully constructed gradient estimator and a special procedure for solving the auxiliary subproblem at each iteration. The gradient estimator is based on SAGA, a popular variance-reduced gradient estimator. We first derive a new variance bound for it, showing that SAGA can exploit functional similarity. We then introduce the Recursive-Gradient technique as a general way to potentially improve the error bound of a given conditionally unbiased gradient estimator, including both SAGA and SVRG. By applying this technique to SAGA, we obtain a new estimator, RG-SAGA, which has an improved error bound compared to the original one.

Section: INTRODUCTION
Motivation. Federated Learning (FL) is a distributed training paradigm in which a central server coordinates model updates across multiple remote clients-such as mobile devices or hospitalswithout requiring access to their local data (McMahan et al., 2017;Kairouz et al., 2021). This framework enables collaborative learning on decentralized data, but introduces new algorithmic challenges due to the distributed nature of optimization.
A key issue in FL is the high cost of communication between the clients and the server. Clients may be intermittently available (Konečnỳ et al., 2016) and connected over slow or unreliable networks. These constraints make it critical to design optimization algorithms that minimize communication costs, particularly in settings with partial client participation.
Various federated optimization algorithms have been proposed to address communication efficiency, each often relying on distinct client-selection strategies. Some methods communicate only with a randomly sampled subset of clients at each round, while others need to select the set of participating clients more carefully or employ hybrid schemes that combine both strategies. While prior works (Woodworth et al., 2018;Korhonen & Alistarh, 2021;Patel et al., 2022;Zhang et al., 2013;Davies et al., 2020;Scaman et al., 2019) introduced a few models for federated optimization, they do not account for the varying costs of each client-selection strategy, which can in practice differ due to factors such as client reliability, device heterogeneity, and network conditions. Consequently, existing metrics such as the number of communication rounds are not entirely fair for comparing methods in such scenario.
For instance, optimization methods based on SARAH (Nguyen et al., 2017;Li et al., 2021a) have been shown to be communication-efficient in finding an approximate stationary point (Mishchenko et al., 2024;Khaled & Jin, 2023). This efficiency arises from the method's ability to exploit dissimilarity (δ) between local and global objectives. In many practical scenarios-such as statistical or semi-supervised learning (Chayti & Karimireddy, 2022;Karimireddy et al., 2021;Khaled & Jin, 2023)-δ is often small, leading to substantial theoretical gains in communication cost. However, SARAH-based methods require periodic full synchronization with all clients in order to compute full gradients. This can be impractical in real-world large-scale federated systems, where clients may be intermittently unavailable due to energy constraints, network issues, or user behavior.
In contrast to SARAH, methods such as SAG (Schmidt et al., 2017) and SAGA (Defazio et al., 2014) are naturally better suited to the partial participation setting in FL. These methods update the model by sampling a small subset of clients at each round and using locally stored gradients. As a result, they avoid the need for periodic full synchronization, which makes them more compatible with federated systems where only a fraction of clients may be available at any given time. Despite this advantage, the existing communication complexity of such methods depends on the individual smoothness constant L max (Reddi et al., 2016;Li et al., 2021b;Karimireddy et al., 2020), which can be significantly larger than the dissimilarity constant δ. Consequently, it remains unclear whether such methods are more communication-efficient than SARAH-based methods, since they rely on fundamentally different client-selection strategies with different constant dependencies.
Contributions. In this work, we aim to develop optimization algorithms that are efficient in both communication and local computation in the setting where client-selection strategies incur different costs. Our main contributions are as follows:
• We propose a new model formalizing the concept of federated optimization algorithms and defining information-based notions of communication and local complexities. This model associates the non-uniform costs with different client-selection strategies, enabling fair comparisons across optimization algorithms. (Section C) • Within our new model, we propose a new gradient method that achieves the best communication and local complexities among existing first-order methods for non-convex optimization. This method is based on the inexact composite gradient method (I-CGM) with a carefully constructed gradient estimator and a special procedure for solving auxiliary subproblem at each iteration.
(Section 6) • Specifically, we first study the convergence of I-CGM for arbitrary gradient estimators and present an efficient technique for solving the auxiliary subproblem. Our technique is based on running the classical composite gradient method locally for a random number of iterations following a geometrical distribution with a carefully chosen parameter. (Section 3) • We then analyze the SAGA estimator and establish a new variance bound for it that only depends on δ without requiring individual smoothness, improving upon previous results showing that SAGA can exploit functional similarity. We also study SVRG as another example that can be incorporated into I-CGM. (Section 4) • Finally, we introduce the Recursive-Gradient (RG) technique as a general way to potentially improve the error bound for a given conditionally unbiased gradient estimator, including both SAGA and SVRG. Applying this technique to SAGA and SVRG, we obtain new RG-SAGA and RG-SVRG gradient estimators with better error bounds compared to the original ones. (Section 5)
We discuss our results in detail in the context of related work in Appendix D and summarize them in Table 1.
this section cite: ['b34', 'b21', 'b25', 'b50', 'b26', 'b41', 'b52', 'b11', 'b44', 'b40', 'b36', 'b24', 'b7', 'b23', 'b24', 'b45', 'b12', 'b43', 'b22']

Section: PROBLEM FORMULATION
We consider the following distributed minimization problem:
min x∈R d f (x) := 1 n n i=1 f i (x) ,(1)
where each f i : R d → R is a differentiable function which can be directly accessed only by client i.
Notation. We abbreviate [n] := {1, 2, . . . , n}. For a finite set A and an integer 1 ≤ m ≤ |A|, A m denotes the power set comprised of all m-element subsets of A. ∥•∥ denotes the standard Euclidean
Table 1: Summary of efficiency guarantees (in BigO-notation) for finding an ε-stationary point. I-CGM-RG-SAGA achieves the best communication and local complexities. For the precise description of the problem classes, notations, as well as the discussions of the methods, see Appendix D.
this section cite: []

Section: Method Communication complexity Assumption Local complexity VR type

this section cite: []

Section: Centralized GD CAnm
Lf F 0 ε 2
this section cite: []

Section: FS nm
Lf F 0 ε 2 None FedRed (Jiang et al., 2024a)
CAnm ∆1F 0 ε 2 2.1 ; 2.3 L1F 0 ε 2 + nm ∆1F 0 ε 2
None FedAvg (McMahan et al., 2017) CR
ζ 2 m F 0 ε 4 + √ Lmaxζ ε 3 + LmaxF 0 ε 2 IS , BGD ζ 2 m F 0 ε 4 + √ Lmaxζ ε 3 + LmaxF 0 ε 2 None FedDyn (Acar et al., 2021) CAnm + CRnm LmaxF 0 ε 2 IS unknown None
MimeMVR (Karimireddy et al., 2021) CA (Mishchenko et al., 2024) (Mishchenko et al., 2024)
ζ 2 m F 0 ε 2 + ζm∆maxF 0 ε 3 + ∆maxF 0 ε 2 IS, BGD, SD Lmaxζ 2 m F 0 ∆maxε 2 + ζmLmaxF 0 ε 3 + LmaxF 0 ε 2 None CE-LGD (Patel et al., 2022) CR ζ 2 m F 0 ε 2 + ζm∆maxF 0 √ mε 3 + ∆maxF 0 ε 2 IS, BGD, SD Lmaxζ 2 m F 0 ∆maxε 2 + ζmLmaxF 0 √ mε 3 + LmaxF 0 ε 2 None Scaffold (Karimireddy et al., 2020) CAnm + CA n 2/3 m LmaxF 0 ε 2 IS nm + n 2/3 m LmaxF 0 ε 2 SAG SABER-full
CAnm + CA (∆max+ √ nmδm)F 0 ε 2 SD unknown PAGE SABER-partial
CAnm + CR ζ 2 m ε 2 ∆maxF 0 ε 2 SD, BGD unknown SARAH I-CGM-RG-SVRG (ours) CAnm + (CR∆1+ √ CACRnmδm)F 0 ε 2 2.1,2.2 ; 2.3 nm + (L1+∆1+ C A C R nmδm)F 0 ε 2 RG-SVRG I-CGM-RG-SAGA (ours) CAnm + CR (∆1+ √ nmδm)F 0 ε 2 2.1,2.2 ; 2.3 nm + (L1+∆1+ √ nmδm)F 0 ε 2 RG-SAGA norm in R d .
We use E [•] to denote the standard (full) expectation. We write E ξ [•] for the expectation taken w.r.t. ξ. We assume that the objective function in problem (1) is bounded from below and denote its infimum by f ⋆ . We denote F 0 := f (x 0 ) -f ⋆ where x 0 is the initial point.
this section cite: ['b34', 'b23', 'b36', 'b36']

Section: FEDERATED OPTIMIZATION ALGORITHMS AND THEIR COMPLEXITY
Due to space limitations, we defer the whole definitions of federated optimizaton algorithms and their complexity metrics to Appendix C, which is encouraged to read before proceeding.
this section cite: []

Section: PROBLEM CLASS
We study optimization problem (1) in which the client objectives exhibit an underlying similarity structure. Specifically, we use the following two assumptions that relax standard smoothness assumptions. The first quantifies the deviation between the delegate function f 1 and f . For an index i ∈ [n], we use h i := f -f i to denote the difference function.
Assumption 2.1. There exists ∆ 1 > 0 such that for any x, y ∈ R d , we have:
∥∇h 1 (x) -∇h 1 (y)∥ ≤ ∆ 1 ∥x -y∥ .(2)
Alternatively, one may define a uniform dissimilarity constant ∆ max (Karimireddy et al., 2020;Jiang et al., 2024a) such that for any i ∈ [n], it holds that ∥∇h i (x) -∇h i (y)∥ ≤ ∆ max ∥x -y∥. In this work, we focus on ∆ 1 since it can be much smaller than ∆ max .
The second assumption characterizes the average dissimilarity among all local functions. Assumption 2.2 (Khaled & Jin (2023); Jiang et al. (2024b); Lin et al. (2024); Jiang et al. (2024a); Takezawa et al. (2025)). There exists δ > 0 such that for any x, y ∈ R d , we have:
1 n n i=1 ∥∇h i (x) -∇h i (y)∥ 2 ≤ δ 2 ∥x -y∥ 2 . (3
)
The left-hand side of (3) is equal to 1 n n i=1 ∥∇f i (x) -∇f i (y)∥ 2 -∥∇f (x) -∇f (y)∥ 2 , which can be interpreted as the variance of ∇f i (x) -∇f i (y) where i is selected uniformly at random. If each f i has L max -Lipschitz gradient, then we have ∆ 1 ≤ 2L max and δ ≤ L max . Therefore, both conditions are weaker than assuming each f i is Lipschitz-smooth. We refer to discussions in (Jiang et al., 2024b) for more properties and details.
The previous two quantities δ and ∆ 1 will only affect the communication complexity of our algorithms, while the local complexity additionally depends on L 1 which is defined as follows.
Assumption 2.3. There exists L 1 > 0 such that for any x, y ∈ R d , we have:
∥∇f 1 (x) -∇f 1 (y)∥ ≤ L 1 ∥x -y∥ .(4)
Inexact Composite Gradient Method. We first introduce the Inexact Composite Gradient Method (I-CGM), which serves as the backbone of our approach. Consider the composite reformulation of the problem 1:
f = f 1 + [f -f 1 ] = f 1 + h 1 .
Let λ > 0 and x 0 ∈ R d be the initial point. At each iteration t ≥ 0, I-CGM computes an approximation of the gradient g t ≈ ∇f (x t ) and defines the next iterate as:
x t+1 ≈ arg min x∈R d F t (x) := f 1 (x) + h 1 (x t ) + ⟨g t -∇f 1 (x t ), x -x t ⟩ + λ 2 ∥x -x t ∥ 2 ,
(I-CGM) where both the inaccuracy in solving the subproblem and the approximation error (defined below) are assumed to be sufficiently small (to be specified later):
F t (x t+1 ) ≤ F t (x t ), e t := ∥∇F t (x t+1 )∥, Σ2 t := g t -∇f (x t ) 2 .(5)
In the following statement, we provide the general convergence guarantee for I-CGM. The proof can be found in Section F.1 in the Appendix. Theorem 3.1. Let I-CGM be applied to Problem (1). Suppose Assumption 2.1 and condition (5) are satisfied. Let λ > ∆ 1 . Then for any T ≥ 1, we have:
T t=1 ∥∇f (x t )∥ 2 + (λ + ∆ 1 ) 2 T t=1 ∥x t -x t-1 ∥ 2 ≤ 12(λ + ∆ 1 ) 2 λ -∆ 1 F 0 + 12(λ + ∆ 1 ) 2 (λ -∆ 1 ) 2 + 4 T -1 t=0 Σ2 t + 4 T -1 t=0 e 2 t .
We see that each subproblem can be solved inexactly without affecting the convergence rate (up to absolute constants), provided that the error term
T -1 t=0 e 2 t is of the same order as the first two terms on the right-hand side. Moreover, if the approximation errors T -1 t=0
this section cite: ['b22', 'b24', 'b33', 'b47']

Section: Σ2
t can also be bounded by the first two terms on the left-hand side, then the convergence of the gradient norm is guaranteed. If there exists randomness either in solving the subproblems or in constructing the estimators, then these conditions are required to hold in expectation. Specifically, we obtain the following corollary. Corollary 3.2. Following the same settings as in Theorem 3.1. If the inaccuracies in solving the subproblems satisfy:
F t (x t+1 ) ≤ F t (x t ), T -1 t=0 E[e 2 t ] ≤ (λ + ∆ 1 ) 2 λ -∆ 1 F 0 + T -1 t=0 Σ 2 t ,(6)
and the approximation errors satisfy:
12(λ + ∆ 1 ) 2 (λ -∆ 1 ) 2 + 8 T -1 t=0 Σ 2 t ≤ 1 2 T t=1 G 2 t + (λ + ∆ 1 ) 2 T t=1 χ 2 t ,(7)
then for any T ≥ 1, we have:
E[∥∇f (x T )∥ 2 ] ≤ 32(λ + ∆ 1 ) 2 λ -∆ 1 F 0 T .
where
G 2 t := E[∥∇f (x t )∥ 2 ], χ 2 t := E[∥x t -x t-1 ∥ 2 ], Σ 2 t := E[∥g t -∇f (x t )∥ 2 ]
, and xT is uniformly sampled from (x t ) T t=1 .
When g t is the exact gradient ∇f (x t ) for all t ≥ 0, then I-CGM is reduced to CGM that is widely used for solving Problem (1), particularly because of its ability to exploit functional similarity and reduce communication costs (Hendrikx et al., 2020;Jiang et al., 2024a;Lin et al., 2024;Khaled & Jin, 2023;Jiang et al., 2024b;Mishchenko et al., 2024;Kovalev et al., 2022). Indeed, if λ ≃ ∆ 1 and the accuracy condition (6) is satisfied, then
E[∥∇f (x T )∥ 2 ] ≤ ε 2 after T = O( ∆1F 0 ε 2 ) iterations.
In contrast, the iteration complexity of Gradient Descent depends on L f which can be larger than ∆ 1 (2.1) when f 1 is similar to f . However, CGM has sub-optimal communication complexity in terms of n. Indeed, let us assume, for simplicity, that m = 1. Then each iteration involves: 1) computing the full gradient ∇f (x t ), which requires n sequential communication rounds using A-CSS, and 2) an additional round using D-CSS for solving the subproblem. Consequently, the total number of communication rounds with A-CSS and D-CSS is N A = nT and N D = T , respectively. The communication complexity of CGM is thus:
C A N A + N D = C A nT + T = O(C A nT ) = O(C A n∆1F 0 ε 2
). This linear dependency on n can be prohibitive in large-scale federated learning settings and is worse than the complexity of stochastic methods such as PROXSARAH (Pham et al., 2020), SPIDERBOOST (Wang et al., 2019), and PAGE (Li et al., 2021a), each of them achieving:
O(C A √ n LF 0 ε 2
), although they rely on a slightly different assumption of average smoothnessfoot_0 . Moreover, the dependence on C A ε 2 can become much larger in scenarios where using A-CSS is costly. Solving Auxiliary Subproblems. In this section, we assume that f 1 is L 1 -smooth and study how to achieve the accuracy condition (6). Recall that each subproblem F t consists of a smooth function
ϕ(x) = f 1 (x) and a quadratic regularizer ψ t (x) = ⟨g t -∇f 1 (x t ), x -x t ⟩ + λ 2 ∥x -x t ∥ 2 .
Let us solve it using the standard composite gradient method (CGM), which proceeds as follows: For
k = 0, 1, ..., K t -1, y t k+1 = arg min y∈R d ϕ(y t k ) + ∇ϕ(y t k ), y -y t k + L 1 2 y -y t k 2 + ψ t (y) = 1 λ + L 1 L 1 y t k + λx t + ∇f 1 (x t ) -g t -∇f 1 (y t k ) .(8)
Each CGM step monotonically decreases the function value of F t (see Lemma F.2). Therefore, we can initialize y t 0 = x t and choose x t+1 to be a certain iterate of (y k ) K k=0 . Then the condition on F t (x t+1 ) ≤ F t (x t ) is satisfied. We next study the number of local steps K t required to achieve the second inequality in condition (6).
this section cite: ['b17', 'b33', 'b24', 'b36', 'b28', 'b42', 'b48']

Section: Fixed Number of Local Steps.
Let K t ≡ K ≥ 1 be a constant number and let x t+1 be the iterate with the minimum gradient norm of F t among {y t k } K k=1 . We use the notation x t+1 = CGM const (λ, K, x t , g t ) for this process.
The goal is to upper bound
T -1 t=0 efoot_1 t where e t := ∥∇F t (x t+1 )∥. For each t ≥ 0, we have: e 2 t ≲ L1(Ft(y t 0 )-Ft(y t K )) K ≲ L1(f (x t )-f (y t K )+ 1 λ Σ2 t K (see Lemma F.2 and F.1). However, since y t K and x t are not necessarily the same, we cannot telescope f (x t ) -f (y t K ) when we sum up e 2 t . Instead, the "best" we can do is to upper bound f (x t ) -f (y t K ) by f (x t ) -f ⋆ . Then by further upper-bounding the summation of T -1 t=0 [f (x t ) -f ⋆ ] in terms of F 0 and Σ2 t , it can be shown that we need K ≃ L1T λ local steps to achieve the desired accuracy condition (6). The proof can be found in Section F.2.1. Lemma 3.3. Consider I-CGM with x t+1 = CGM const (λ, K, x t , g t ) under Assumption 2.1 and 2.3. Let T ≥ 1 be the fixed number in condition (6). Then by choosing λ > ∆ 1 and K = K T := ⌈ 8L1T λ-∆1 ⌉, the accuracy condition (6) is satisfied.
this section cite: []

Section: Random Number of Local Steps.
We now allow the number of local steps K t to follow a geometric distribution-a common technique used to derive last-iterate recurrences (Allen-Zhu, 2018b). When applied to solve the subproblems in I-CGM, this approach yields an algorithm that is efficient in local computation.
Let us consider CGM (8) with K t = Kt + 1 iterations where Kt ∼ Geom(p), that is P( Kt = k) = (1 -p) k p for each k ∈ {0, 1, 2, ...}. The solution is set to be x t+1 = y Kt . We use the notation x t+1 = CGM rand (λ, Kt , x t , g t ) for this process.
In contrast to the convergence rate of using a deterministic K, we can now show that E Kt [e 2 t ] ≲
L 2 1 p L1+λ E Kt [F t (x t ) -F t (x t+1 )]. Using E Kt [F t (x t ) -F t (x t+1 )] ≲ E Kt [f (x t ) -f (x t+1 ) + 1 λ Σ2 t ]
, we get the telescoping term E[f (x t ) -f (x t+1 )] after passing to the full expectation, which allows to improve the total amount of local computations.
Lemma 3.4. Consider I-CGM with x t+1 = CGM rand (λ, Kt , x t , g t ) where Kt ∼ Geom(p) under Assumption 2.1 and 2.3. Let T ≥ 1 be the fixed number in condition (6). Then by choosing λ > ∆ 1 and p = λ-∆1 8(L1+λ) < 1, the accuracy condition (6) is satisfied.
To achieve the accuracy condition (6), the number of local first-order oracle queries required by using the random Kt at each iteration t in expectation is E Kt [K t ] = 1 p ≃ L1 λ , which improves upon the previous result of L1T  λ obtained by using a fixed number of K. So far, we have studied how to solve the subproblems of I-CGM such that the accuracy condition (6) is satisfied. We now turn to constructing the gradient estimator g t that has the desired approximation error (7). Meanwhile, we aim to improve both the dependence on n and C A in the communication complexity of CGM. The main strategy is to design a gradient estimator whose approximation error depends only on the similarity constant δ while avoiding periodic full synchronizations.
this section cite: []

Section: BASIC APPLICATION EXAMPLES: SAGA + SVRG
In this section, we present two algorithms that maintain an approximation of the gradient, G t ≈ ∇f (x t ) for t ≥ 0. Each algorithm starts with an initial point x 0 . Then at each iteration t ≥ 0, G t is computed first, after which the next iterate x t+1 is computed. In what follows, for a set S ∈ [n]  m and m ∈ [n], we use f S := 1 m i∈S f i to denote the average function over this set. For convenience of presentation, we use the following notations throughout the rest of the paper:
n m := n m , q m := n -m n -1 , and δ 2 m := q m m δ 2 .(9)
SAGA Estimator. SAGA estimator is a variance-reduction technique based on incremental gradient updates, originally designed for centralized finite-sum minimization (Defazio et al., 2014). In this section, we adapt this estimator to the federated optimization scenario and study its properties.
The SAGA estimator defines:
G 0 = ∇f (x 0 ), G 1 = ∇f (x 1 ), G t = b t St -b t-1 St + b t-1 , t ≥ 2 ,(SAGA)
where S t ∈ [n]  m is uniformly sampled at random without replacement, b
t St := 1 m i∈St b t i , b t-1 St := 1 m i∈St b t-1 i , b t := 1 n n i=1 b t i
, and for any i ∈ [n], b t i is recurrently defined as:
b 0 i = ∇f i (x 0 ), b 1 i = ∇f i (x 1 ), b t i = ∇f i (x t ) if i ∈ S t , b t-1 i otherwise, , t ≥ 2 .
We have the following recurrence for b t (the derivation can be found in Lemma F.3):
b t = b t-1 + 1 n m [∇f St (x t ) -b t-1 St ], t ≥ 2 .(10)
Implementation. At the beginning, when t = 0 and 1, each client i = 1, . . . , n computes ∇f i (x t ) and initializes b t i and sends the result to the server; the server then aggregates these results computing ∇f (x t ) to initialize G t and b t . This requires two full synchronizations (2⌈n m ⌉ communications rounds using A-CSS). At each iteration t ≥ 2, the server contacts the randomly selected set of clients S t using R-CSS and sends x t to them. Each client i ∈ S t computes b t i = ∇f i (x t ) and sends b t i -b t-1 i back to the server. The server then updates b t according to ( 10) and constructs the gradient estimator G t using the stored b t-1 according to (SAGA).
Each client i thus needs to store a single vector b t i . On the server side, only the aggregated vector b t and the iterate x t need to be maintained. The memory overhead of the SAGA estimator is thus very small in the federated learning setting, similarly to the SAG estimator (Schmidt et al., 2017) used in SCAFFOLD (Karimireddy et al., 2020).
Properties of SAGA. It is not difficult to show that G t is a conditionally unbiased estimator of ∇f (x t ), namely, E St [G t ] = ∇f (x t ).
We next present a new variance bound for SAGA that is controlled by the constant δ. The proof can be found in Section F.3.1.
Lemma 4.1. Consider the SAGA estimator (SAGA) under Assumption 2.2. Then for any t ≥ 2, E St [G t ] = ∇f (x t ) and for any T ≥ 1, we have :
T t=0 σ 2 t ≤ 2n m q m m G 2 1 + n m -1 + n 2 m -n m (n -1) T -1 t=2 G 2 t + 4n 2 m δ 2 m T t=2 χ 2 t ,
where
σ 2 t := E S [t] [∥G t -∇f (x t )∥ 2 ], G 2 t = E S [t-1] [∥∇f (x t )∥ 2 ], χ 2 t := E S [t-1] [∥x t -x t-1 ∥ 2 ]
, and S [t] := (S 2 , ..., S t ).
Note that this variance bound depends on G 2 1 , . . . , G 2 T -1 and χ 2 2 , . . . , χ 2 T , which aligns with the terms on the right-hand side of the desired error bound (7). However, the coefficient in front of G 2 1 in this bound can be larger than 1, whereas (7) requires it to be strictly less than 1. Consequently, the requirement is not met and we cannot directly incorporate the SAGA estimator into I-CGM by setting g t = G t . We will show in Section 5 that this error bound can be significantly improved by using the recursive gradient estimation technique. Remark 4.2. Instead of computing the exact gradients ∇f (x 0 ) and ∇f (x 1 ) at the beginning which requires full synchronizations, it is possible to start with an approximation G 0 ≈ ∇f (x 0 ). This requires only one communication round using R-CSS. The resulting communication-complexity estimate will now additionally depend on the inexactness of the initial approximation but this strategy often works well in practice (Figure J.1). See Appendix G for detailed discussions.
this section cite: ['b12', 'b45', 'b22']

Section: SVRG Estimator.
Another possible choice of the gradient estimator is the SVRG estimator (Johnson & Zhang, 2013). There are different variants of SVRG, and here we consider the so-called loopless-SVRG estimator (Kovalev et al., 2020) for simplicity.
The SVRG estimator defines:
G 0 = ∇f (x 0 ), G t = ∇f St (x t ) + ∇f (w t ) -∇f St (w t ), t ≥ 1 ,(SVRG)
where S t ∈ [n]  m is uniformly sampled at random without replacement,
w 0 = x 0 , w t = x t if ω t = 1, w t-1 otherwise, t ≥ 1 ,
and ω t is a Bernoulli random variable with parameter p B , i.e., P (ω t = 1) = p B ∈ (0, 1).
Properties: It is not difficult to show that the SVRG estimator G t is a conditionally unbiased estimator of ∇f (x t ), namely, E St [G t ] = ∇f (x t ). Moreover, the variance is controlled by δ. The proof can be found in Section F.3.2 where the implementation of the estimator is also provided. Lemma 4.3. Consider the SVRG estimator (SVRG) under Assumption 2.2. Then for any t ≥ 1, E St [G t ] = ∇f (x t ) and for any T ≥ 1, we have: T t=0 σ 2 t ≤ 4δ 2 m p 2 B T t=1 χ 2 t , where σ 2 t := E St,ω [t] [∥G t -∇f (x t )∥ 2 ], χ 2 t := E ω [t-1] [∥x tx t-1 ∥ 2 ], and ω [t] := (ω 1 , ..., ω t ).
We can incorporate the SVRG estimator into I-CGM by setting g t = G t . This requires setting p B ≃ 1 nm and λ ≃ ∆ 1 + n m δ m to achieve the error condition (7). The resulting communication complexity of the method is O(
C A n m + (C R ∆1+C A nmδm)F 0 ε 2
), which still has a linear dependence on n m . (See Theorem F.4 with the proof that the reader can inspect if interested). Note that unbiasedness is not needed to incorporate SVRG directly into I-CGM. However, it becomes necessary later for the recursive gradient technique, which we discuss in the next section.
this section cite: ['b20', 'b27']

Section: RECURSIVE GRADIENT ESTIMATOR + EXAMPLES (SAGA AND SVRG)
In this section, we present a general formular of the recursive gradient estimator that can potentially improve the error bound for a given conditionally unbiased gradient estimator G t ≈ ∇f (x t ). Formally, we consider the following setting. Assumption 5.1. For any t ≥ 0, it holds that: 1) S t is independent of x 0 , . . . ,
x t+1 , G 0 , . . . , G t-1 ; 2) E St [G t ] = ∇f (x t ).
Published as a conference paper at ICLR 2026
The recursive gradient estimator (RG) defines:
g 0 = ∇f (x 0 ), g t+1 = (1 -β)g t + βG t + ∇f St (x t+1 ) -∇f St (x t ), t ≥ 0 ,(RG)
where β ∈ (0, 1] and S t ∈ [n]  m is uniformly sampled at random without replacement. Note that the indexing here differs from the previous ones. The algorithm starts with an initial point x 0 . At each iteration t ≥ 0, the estimator g t ≈ ∇f (x t ) is computed first and it depends only on G t-1 and S t-1 . After that, the next iterate x t+1 is computed. Therefore, x t+1 is independent from S t while previously it was dependent on it (if we use the SAGA/SVRG estimator).
Inspired by previous works, the expression of g t incorporates both recursive gradient update and momentum (Chayti et al., 2025;Gao et al., 2024). This expression unifies several existing methods: When β = 0, the estimator reduces to the SARAH update rule (Nguyen et al., 2017). When G t is replaced with the SAGA estimator, then g t recovers the structure of ZEROSARAH (Li et al., 2021b). When G t = ∇f St (x t+1 ) and ∇f St (x t+1 ) -∇f St (x t ) is multiplied by 1 -β, then it becomes STORM (Cutkosky & Orabona, 2019). In our formulation, G t is a general similarity-aware estimator of ∇f (x t ) that satisfies Assumption 5.1, allowing us to flexibly instantiate the framework with various variance-reduction techniques.
For instance, we can combine RG with SAGA or SVRG. We refer to the resulting estimators as RG-SAGA and RG-SVRG. Note that S t in the formulas for SAGA and SVRG is exactly the same random index set that is used in the RG -they share the same randomness for the sake of efficiency.
Implementation. At the beginning, each client i = 1, . . . , n computes ∇f i (x 0 ) and sends the result to the server; the server then aggregates these results, computing ∇f (x 0 ) to initialize g 0 . This requires one full synchronization. Then x 1 is computed based on g 0 . At each iteration t ≥ 0, the server uses R-CSS which generates a random subset S t . For RG-SAGA, the server sends x t+1 , x t to the clients in S t . Each client i ∈ S t updates b t i = ∇f i (x t ) and sends ∇f i (x t+1 ) along with b t i -b t-1 i (when t ≥ 2) or b t i (when t = 1) to the server. For RG-SVRG, the server sends x t+1 , x t and w t to the clients which then return the gradients evaluated at these three points. If ω t = 1, the server additionally computes the new gradient ∇f (w t ) performing one full synchronization. After receiving all the vectors, the server can compute ∇f St (x t+1 ), ∇f St (x t ), G t and g t+1 .
For RG-SAGA, each client i needs to store a single vector b t i and the server needs to maintain two points x t+1 and x t , and two vectors b t and g t . For RG-SVRG, clients are stateless and the server is required to maintain three points x t+1 , x t , w t and one vector ∇f (w t ).
Lemma 5.2 (Error bound for RG). Consider the RG estimator (RG) under Assumptions 5.1 and 2.2. Then for any T ≥ 1, we have:
T t=0 Σ 2 t ≤ 2β 2-β T -1 t=0 σ 2 t + 2δ 2 m 2β-β 2 T t=1 χ 2 t .
where
Σ 2 t := E S [t-1] [∥g t -∇f (x t )∥ 2 ], σ 2 t := E S [t] [∥G t -∇f (x t )∥ 2 ], χ 2 t := E S [t-2] [∥x t -x t-1 ∥ 2 ]
, and S [t] := (S 0 , . . . , S t ).
The proof can be found in Section F.4.1. We next show that the error bound of both SAGA and SVRG can be improved by combining them with RG and adjusting the parameter β. For instance, by combining Lemma 5.2 and Lemma 4.1, we obtain the following result for RG-SAGA.
Corollary 5.3. Consider the RG-SAGA estimator under Assumptions 5.1 and 2.2. Then for any T ≥ 1, it holds that:
T t=0 Σ 2 t ≤ 4βn m q m (2 -β)m G 2 1 + 2β(n m -1 + n 2 m -n m ) (2 -β)(n -1) T -1 t=2 G 2 t + 8β 2 n 2 m δ 2 m + 2δ 2 m 2β -β 2 T t=1 χ 2 t ,
where
Σ 2 t := E S [t-1] [∥g t -∇f (x t )∥ 2 ], G 2 t := E S [t-2] [∥∇f (x t )∥ 2 ], χ 2 t := E S [t-2] [∥x t -x t-1 ∥ 2
] and S [t] := (S 0 , . . . , S t ).
this section cite: ['b8', 'b13', 'b40', 'b10']

Section: By choosing β ≃ 1
nm , we get
T t=0 Σ 2 t ≲ qm m G 2 1 + 1 n T -1 t=2 G 2 t + n m δ 2 m T t=1 χ 2 t .
Compared with the original variance bound for SAGA (Lemma 4.1), the bound with RG achieves an improvement by a factor of n m .
The error bound for the SVRG estimator can be improved in a similar way.
Corollary 5.4. Consider the RG-SVRG estimator under Assumptions 5.1 and 2.2. Then for any T ≥ 1, it holds that: T t=0 Σ 2 t ≤ 8β 2 δ 2 m /p 2 B +2δ 2 m 2β-β 2 T t=1 χ 2 t , where Σ 2 t := E S [t-1] ,ω [t-1] [∥g t -∇f (x t )∥ 2 ], χ 2 t := E S [t-2] ,ω [t-2] [∥x tx t-1 ∥ 2 ], S [t]
:= (S 0 , . . . , S t ) and ω [t] := (ω 1 , . . . , ω t ).
Compared with the original variance bound for SVRG (Lemma 4.3), the new bound achieves an improvement by a factor of 1/p B by choosing β ≃ p B .
We can now incorporate both enhanced estimators into I-CGM. It can be shown that the iterates {x t } ∞ t=0 generated by I-CGM-RG-SAGA or I-CGM-RG-SVRG and the corresponding sequence {G t } ∞ t=0 satisfy Assumption 5.1. (See Lemma F.5 and F.6).
this section cite: []

Section: COMMUNICATION AND LOCAL COMPLEXITY OF I-CGM-RG
We are ready to establish the complexity of I-CGM equipped with the RG-SAGA and RG-SVRG estimator. We first present the result for RG-SAGA. The proof can be found in Section F.5.1.
Theorem 6.1 (I-CGM-RG-SAGA). Let I-CGM be applied to Problem 1 under Assumptions 2.1, 2.2 and 2.3, where x t+1 = CGM rand (λ, Kt , x t , g t ) with Kt ∼ Geom(p) and g t is generated by the RG-SAGA estimator. Then by choosing λ = 3∆ 1 + 113 √ n m δ m , β = 1 112nm and p = λ-∆1 8(L1+λ) , after T = ⌈ (256(∆1+38 √ nmδm)F 0 ε 2 ⌉ iterations, we have E[∥∇f (x T )∥ 2 ] ≤ ε 2 , where xT is is uniformly sampled from (x t ) T t=1 . The communication complexity is at most 2C A ⌈n m ⌉+(C R +1)⌈ (256(∆1+38 √ nmδm)F 0 ε 2 ⌉ and the local complexity is bounded by 14
+ 2⌈n m ⌉ + 512(7∆1+283 √ nmδm+2L1)F 0 ε 2 + 4L1 ∆1+28 √ nmδm .
this section cite: []

Section: The communication complexity of I-CGM-RG-SAGA is of order C
A n m + C R ∆1+( √ nmδm)F 0 ε 2 and the local complexity is of order n m + (∆1+ √ nmδm+L1)F 0 ε 2 when (∆1+ √ nmδm)F 0 ε 2 ≳ 1.
The n m term comes from n m sequential rounds with A-CSS for computing the full gradients in the beginning.
this section cite: []

Section: Comparison: RG-SVRG Estimator. The communication complexity of I-CGM-RG-SVRG is O C
A n m + (C R ∆1+ √ C A C R nmδm)F 0 ε 2
, where C A also affects the term involving ε (see Appendix F.6 for details and the result of the local complexity).
this section cite: []

Section: NUMERICAL EXPERIMENTS
In this section, we verify the theory of the proposed methods in numerical experiments. We set C A = C R = 1 in the definition of communication complexity for all the experiments. We choose this case to demonstrate that even when A-CSS and R-CSS are equally cheap, our proposed methods already outperform several commonly used algorithms. (The study of the scenario when C A > C R can be found in Appendix J.1.1.)
Quadratic minimization with log-sum penalty. Consider the problem of f (x) = 1 n n i=1 f i (x) with f i (x) := 1 b b j=1 1 2 ⟨A i,j (x-b i,j ), x-b i,j ⟩+ d k=1 log 1+α|x k | , where α > 0, b i,j ∈ R d , A i,j ∈ R d×d
is a diagonal matrix, and • k is an indexing operation of a vector. We set α = 10, b = 5, n = 100 and d = 1000. Each coordinate of b i,j is uniformly sampled from [0, 10]. To generate A i,j , we first sample a diagonal matrix Ā with entries uniformly distributed in [0, 110], and then add bn diagonal noise matrices whose entries are sampled from [0,18]. Each resulting A i,j is clipped to the interval [1, 100] on the diagonal, and some eigenvalues are further set close to zero. Consequently, the dataset satisfies 0 ⪯ A i,j ⪯ 100I for any i, j, with ∆ 1 ≈ δ ≈ 5 and L max ≈ 100. We set m = √ n. For I-CGM-RG, we set p = δ L , λ = √ n m δ + ∆ 1 , η = 2L max and β = m n . We compare two proposed algorithms against SCAFFOLD (Karimireddy et al., 2020), FEDAVG (McMahan et al., 2017) (with sampling), SABER-FULL (Mishchenko et al., 2024) (with PAGE), SABER-PARTIAL (Mishchenko et al., 2024) (only compute full gradient once) and GD (running directly on f ). For SVRG-based methods, the expected number of communication rounds at each iteration is roughly m n n + m, which is twice as large as other methods. From Figure 1, we observe that: 1) I-CGM-RG-SAGA is the most efficient in both communication and local computation. 2) SCAFFOLD cannot fully exploit δ-similarity as its local complexity is comparable to GD (the theoretical local complexity of both 0 50 100 150 200 250 300 Communication complexity 10 2 10 1 || f(x)|| mushrooms (m=1,n=10) 0 50 100 150 200 250 300 # iterations 10 3 10 2 10 1 10 0 value Second-order Similarity local smoothness average similarity 0 100 200 300 400 500 Communication complexity 10 1 10 0 10 1 || f(x)|| duke (m=1,n=10) 0 100 200 300 400 500 # iterations 10 2 10 1 10 0 10 1 10 2 10 3 value Second-order Similarity local smoothness average similarity I-CGM-RG-SAGA (ours) I-CGM-RG-SVRG (ours) Scaffold FedAvg SABER-full SABER-partial Scaffnew Figure 2: Comparisons of different algorithms on two LIBSVM datasets using logistic loss with non-convex regularizer. methods depends on L max ). Finally, I-CGM-RG-SAGA with different initialization strategies can be found in Figure J.1.
this section cite: ['b22', 'b36', 'b36']

Section: Logistic regression with nonconvex regularizer.
We now experiment with the binary classification task on two real-world LIBSVM datasets (Chang & Lin, 2011). We use the standard regularized logistic loss:
f (x) = 1 n n i=1 f i (x) with f i (x) := n M mi j=1 log(1 + exp(-y i,j ⟨a i,j , x⟩)) + α d k=1 [x] 2 k 1+[x] 2 k
where α > 0, (a i,j , y i,j ) ∈ R d+1 are feature and labels and M := n i=1 m i is the total number of data points. We use m = 1 and n = 10. We plot the local L 1 and δ by computing ∇f 1 (
x t ) -∇f 1 (x t+1 ) / x t -x t+1 and 1 n n i=1 ∥∇h i (x t ) -∇h i (x t+1 )∥ 2 / ∥x t -x t+1 ∥ 2 along the iterates of I-CGM-RG-SAGA.
From Figure 2, we observe that δ is much smaller than L 1 for the mushrooms dataset, while being comparable for the duke dataset. However, for both cases, I-CGM-RG-SAGA remains the most efficient in communication complexity.
this section cite: ['b6']

Section: Deep learning tasks.
We defer the study of neural network training to Appendix J, where more experiments and details can be found.
this section cite: []

Section: CONCLUSION
We introduced a new simple model for comparing centralized distributed optimization algorithms, where different client-selection strategies are associated with non-uniform costs. Within this model, we developed a new family of algorithm based on inexact composite gradient method with recursive gradient estimator. This design enables us to exploit functional similarity among clients while supporting partial client participation-a key requirement in practical FL systems. It is efficient when full synchronizations (requiring sequential communications with all clients) are costly compared to client sampling. The key technical contribution of this work is a new variance bound for the SAGA estimator, which depends on the functional similarity constant δ rather than individual smoothness. This allows the SAGA-based variant of I-CGM-RG to outperform the previously bestknown communication complexity of SARAH-based methods. Limitations and future extensions are discussed at the end of the Appendix.
this section cite: []

Section: References
Ref_id:b0 Title: Federated Learning Based on Dynamic Regularization Year: (2021)
Ref_id:b1 Title: Katyusha x: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization Year: (2018)
Ref_id:b2 Title: Katyusha x: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization Year: (2018)
Ref_id:b3 Title: Communication Complexity of Distributed Convex Learning and Optimization Year: (2015)
Ref_id:b4 Title: On the complexity of minimizing convex finite sums without using the indices of the individual functions Year: (2020)
Ref_id:b5 Title: Communication Lower Bounds for Statistical Estimation Problems via a Distributed Data Processing Inequality Year: (2016)
Ref_id:b6 Title: LIBSVM: A Library for Support Vector Machines Year: (2011)
Ref_id:b7 Title: Optimization with Access to Auxiliary Information Year: (2022)
Ref_id:b8 Title: Improving Stochastic Cubic Newton with Momentum Year: (2025-05)
Ref_id:b9 Title: EMNIST: Extending MNIST to Handwritten Letters Year: (2017)
Ref_id:b10 Title: Momentum-Based Variance Reduction in Non-Convex SGD Year: (2019)
Ref_id:b11 Title: New Bounds for Distributed Mean Estimation and Variance Reduction Year: (2020)
Ref_id:b12 Title: SAGA: A Fast Incremental Gradient Method with Support for Non-Strongly Convex Composite Objectives Year: (2014)
Ref_id:b13 Title: Non-Convex Stochastic Composite Optimization with Polyak Momentum Year: (2024)
Ref_id:b14 Title: A Bias Correction Mechanism for Distributed Asynchronous Optimization Year: (2025)
Ref_id:b15 Title: On Communication Cost of Distributed Statistical Estimation and Dimensionality Year: (2014)
Ref_id:b16 Title: Deep Residual Learning for Image Recognition Year: (2016)
Ref_id:b17 Title: Statistically Preconditioned Accelerated Gradient Method for Distributed Optimization Year: (2020)
Ref_id:b18 Title: Federated Optimization with Doubly Regularized Drift Correction Year: (2024-07)
Ref_id:b19 Title: Stabilized Proximal-Point Methods for Federated Optimization Year: ()
Ref_id:b20 Title: Accelerating Stochastic Gradient Descent using Predictive Variance Reduction Year: (2013)
Ref_id:b21 Title: Advances and Open Problems in Federated Learning Year: (2021)
Ref_id:b22 Title: SCAFFOLD: Stochastic Controlled Averaging for Federated Learning Year: (2020)
Ref_id:b23 Title: Breaking the Centralized Barrier for Cross-Device Federated Learning Year: (2021)
Ref_id:b24 Title: Faster Federated Optimization under Second-Order Similarity Year: (2023)
Ref_id:b25 Title: Federated Learning: Strategies for Improving Communication Efficiency Year: (2016)
Ref_id:b26 Title: Towards Tight Communication Lower Bounds for Distributed Optimisation Year: (2021)
Ref_id:b27 Title: Don't Jump Through Hoops and Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop Year: (2020)
Ref_id:b28 Title: Optimal Gradient Sliding and its Application to Optimal Distributed Optimization under Similarity Year: (2022)
Ref_id:b29 Title: CIFAR-10 Year: ()
Ref_id:b30 Title: Distributed Stochastic Variance Reduced Gradient Methods by Sampling Extra Data with Replacement Year: (2017)
Ref_id:b31 Title: PAGE: A Simple and Optimal Probabilistic Gradient Estimator for Nonconvex Optimization Year: ()
Ref_id:b32 Title: Efficient Nonconvex Finite-Sum Optimization with Zero Full Gradient Computation Year: (2021)
Ref_id:b33 Title: Stochastic Distributed Optimization under Average Second-Order Similarity: Algorithms and Analysis Year: (2024)
Ref_id:b34 Title: Communication-Efficient Learning of Deep Networks from Decentralized Data Year: (2017)
Ref_id:b35 Title: ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally Year: (2022)
Ref_id:b36 Title: Hongxiang Fan, and Stylianos Venieris. Federated Learning Under Second-Order Data Heterogeneity Year: (2024)
Ref_id:b37 Title: Information-Based Complexity of Convex Programming Year: (1994)
Ref_id:b38 Title: Problem Complexity and Method Efficiency in Optimization Year: (1983)
Ref_id:b39 Title: Lectures on Convex Optimization Year: (2018)
Ref_id:b40 Title: SARAH: A Novel Method for Machine Learning Problems using Stochastic Recursive Gradient Year: (2017)
Ref_id:b41 Title: Towards Optimal Communication Complexity in Distributed Non-Convex Optimization Year: (2022)
Ref_id:b42 Title: ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization Year: (2020)
Ref_id:b43 Title: Suvrit Sra, Barnabás Póczos, and Alex Smola. Fast Incremental Method for Nonconvex Optimization Year: (2016)
Ref_id:b44 Title: Optimal Convergence Rates for Convex Distributed Optimization in Networks Year: (2019)
Ref_id:b45 Title: Minimizing Finite Sums with the Stochastic Average Gradient Year: (2017)
Ref_id:b46 Title: Communication-Efficient Distributed Optimization using an Approximate Newton-Type Method Year: (2014)
Ref_id:b47 Title: Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization Year: (2025)
Ref_id:b48 Title: SpiderBoost and Momentum: Faster Variance Reduction Algorithms Year: (2019)
Ref_id:b49 Title: The Minimax Complexity of Distributed Optimization Year: (2021)
Ref_id:b50 Title: Graph Oracle Models, Lower Bounds, and Gaps for Parallel Stochastic Optimization Year: (2018)
Ref_id:b51 Title: A Coefficient Makes SVRG Effective Year: (2025)
Ref_id:b52 Title: Information-Theoretic Lower Bounds for Distributed Statistical Estimation with Communication Constraints Year: (2013)
