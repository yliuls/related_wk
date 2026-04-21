Title: Sharp Gaussian approximations for Decentralized Federated Learning
Abstract: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results. Berry-Esseen theory for local SGDIn this section we establish a general, Berry-Esseen type Gaussian approximation result in the decentralized federated learning setting. In order to rigorously state our results, it is imperative that we formally introduce the local stochastic gradient descent (SGD) algorithm and underline the key assumptions behind our theoretical results. This is done in Section 2.1. Finally, we present our first Gaussian approximation results in Section 2.2, and discuss the implications therein. PreliminariesConsider a typical decentralized heterogeneous federated learning setting with K clients, each having access to a loss function f k : R d × R n k → R, and a distribution P k on R n k for k ∈ [K].Here, P k determines the distribution of the local noisy gradient for each client, realized by sampling ξ k ∼ P k . We allow for heterogeneity among the clients i.e. P k 's are allowed to be different. However, noise sampling (i.e. the ξ k ) is assumed to be independent from one client to the another. The corresponding risk or regret for the k-th client is denoted by F k (θ) = E ξ k ∼P k f k (θ, ξ k ). Con-

Section: Introduction
Federated Learning (FL), introduced by McMahan et al. [2017] as a decentralized model training paradigm while maintaining privacy, has seen rapid advancements driven by its applicability in domains such as next-word prediction on mobile devices, healthcare, and cross-silo collaborations among institutions. Subsequent works Kairouz et al. [2021], Li et al. [2020], Karimireddy et al. [2020], Wang et al. [2020b], Alistarh et al. [2017], Lin et al. [2018] have addressed key challenges around privacy and computational efficiency. Research has also extended to decentralized federated learning (DFL) Lalitha et al. [2019], Lian et al. [2017], He et al. [2019], Kim et al. [2020], Lian et al. [2017], Wang and Joshi [2021], Singh et al. [2023], which eliminates reliance on a central server by enabling peer-to-peer collaboration, thereby enhancing robustness, fairness, and resilience to adversarial threats. We refer to Gabrielli et al. [2023], Yuan et al. [2024] for a comprehensive survey of the literature. In this regard, Local SGD Stich [2019], Khaled et al. [2020], Woodworth et al. [2020b] has emerged as a widely adopted algorithm, allowing clients to perform multiple local updates before synchronizing, significantly reducing communication overhead.
While theoretical guarantees for convergence and speed in local SGD have been developed Haddadpour et al. [2019], Woodworth et al. [2020a], Koloskova et al. [2020], a gap remains in understanding the statistical properties of fluctuations around the true parameter vector. This gap has practical implications: first, statistical guarantees on the final iterates are essential for inference; second, monitoring the entire trajectory is crucial for detecting adversarial behavior in high-stakes settings like traffic networks, autonomous systems, and financial platforms. For the first issue, emerging works on central limit theory Li et al. [2022], Gu and Chen [2024] provide initial insights, but estimating local covariance structures is numerically intensive. Multiplier bootstrap methods Fang et al. [2018], Fang [2019] offer computational relief, but require stronger results beyond the central limit theory. The second issue is even more challenging, as it demands control over the entire trajectory of local SGD, not just the last iterate. Classical inferential methods struggle with DFLs complex dependency structure, and a key open question is how to develop statistically valid, computationally efficient inference methods with minimal distributional assumptions and explicit error control.
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
this section cite: ['b53', 'b31', 'b44', 'b32', 'b0', 'b49', 'b40', 'b47', 'b29', 'b36', 'b47', 'b77', 'b73', 'b21', 'b87', 'b35', 'b28', 'b38', 'b46', 'b27', 'b20', 'b19']

Section: Main Contributions
In this article, we address this gap by proposing different refined Gaussian approximations that naturally lead to suitable bootstrap procedures. Our results go beyond central limit theory to establish sharper, step-by-step as well as uniform control over the DFL iterates {Y t }. These results not only facilitate relevant bootstrap-based inference to produce asymptotically-valid confidence sets, but also enables us to perform statistical hypothesis tests to detect attacks, which are inaccessible otherwise. Our main contributions can be summarized as follows:
(1) In Section 2, we provide an explicit characterization of the Berry-Esseen error for the Polyak-Ruppert version of the local SGD algorithm (Algorithm 1) iterates. In particular, under standard regularity conditions on the client-level optimization problems as well as for a general class of connection graph of clients, we prove: Theorem 1.1 (Theorem 2.1, Informal). For a decentralized federated learning set-up with K clients, the Polyak-ruppert averaged iterates of the local SGD algorithm with n iterations, and step size η t t -β , achieves d Berry-Esseen ≲ n 1/2-β √ K.
Our result explicitly underpins the source of the assumption K = o(n 2β-1 ) used to derive central limit theory for the local SGD iterates Gu and Chen [2024]. Theorem 2.1 is accompanied by a corresponding Berry-Esseen theorem (Theorem 2.2) for final iterates of the DFL algorithm. Both these theorems involve a finite sample scaling (equivalently, scaling by a covariance matrix depending upon n, the number of iterations) of the local SGD iterates, leading to optimal error bounds. Our result is first such Berry-Esseen bounds for the local SGD updates.
(2) The finite sample scaling considered in Theorems 2.1 and 2.2 is usually not estimable. Shifting focus to an asymptotic, global scaling, our results uncover a novel computation-communication trade-off involving the Berry-Esseen result. In Theorem 2.3, we show that for K = o( √ n), β = 3/4 represents an optimal choice of step-size; however, for K ≳ √ n, for no β ∈ (1/2, 1) does the Berry-Esseen bound converge towards zero. This observation is not merely an artifact of our proof, and the phase-transition are empirically validated through extensive simulations.
(3) A key motivation behind the local SGD algorithm is maintaining privacy. From this perspective, asymptotic inference on the final iterates is insufficient for detecting breach of privacy through some adversarial attack. Indeed, in Section 3, we discuss a general framework to detect a broad class of model poisoning in a distributed setting. Through an example in Section 3.1, we point out a class of maximal statistics which can be used to detect such attacks. Moreover, to perform inference on such statistics, we move beyond controlling simply the end-term iterates to a more general time-uniform Gaussian coupling of the entire local SGD process. Motivated from above, in Theorem 3.1, we establish a time-uniform Gaussian approximation. Theorem 1.2 (Theorem 3.1, Informal). If the local SGD algorithm with K clients runs n iterations with step size η t t -β , then there exists a Gaussian process Y G t = (I -η t A)Y G t-1 + η t Z t with Z t i.i.d. N (0, Γ) for some matrix Γ and A being the Hessian of the problem, such that,
max 1≤t≤n | t s=1 (Y s -Y G s )| ≈ o P (n 1-β + n 1/p √ K ).
Here we assume p ≥ 2 finite moments of the local noisy gradients. To facilitate bootstrap, we also provide an explicit characterization of Γ. To the best of our knowledge, these results constitute the first time-uniform Gaussian approximation results for stochastic approximation algorithms.
(4) In particular, Theorem 3.1 presents a Gaussian approximation (referred to as Aggr-GA) with a slightly sharper rate, but one requiring extensive synchronization during the bootstrap procedure.
Recognizing that this may not be ideal from a privacy perspective, we further present a separate, client-level Gaussian approximation Client-GA in Theorem 3.2, which completely mimics the local SGD procedure. The approximation Client-GA is much more localized, leading to slight worsening of the approximation error but increased efficiency with regards to synchronization and computational cost. We argue and validate with simulations, that our Gaussian approximations are much sharper than that indicated by a standard, off-the-shelf functional central-limit theorem. In fact, our Gaussian approximations represent a version of the covariance-matching approximations introduced by Bonnerjee et al. [2024], however in a multivariate, non-stationary environment.
(5) Finally, in Section 4, we validate our theoretical findings with extensive numerical exercises. Our simulation results in Sections 4.1 and 4.2 not only indicate the sharpness of our theoretical results, but also project vividly the computation-communication trade-offs discussed in Remark 2.2. Moreover, the numerical results in Section 4.3 shows that the proposed Gaussian approximations Aggr-GA and Client-GA are significantly better than an off-the-shelf Brownian-motion based approximation, even in finite sample, complementing Theorems 3.1 and 3.2 well.
this section cite: ['b27', 'b7']

Section: Notations
In this paper, we denote the set {1, . . . , n} by [n]. The d-dimensional Euclidean space is R d , with R d >0 the positive orthant. For a vector a ∈ R d , |a| denotes its Euclidean norm. The set of m × n real matrices is denoted by R m×n , and correspondingly, for M ∈ R m×n , |M | F denotes its Frobenius norm. For a random vector X ∈ R d , we denote X := E[|X| 2 ]. We also denote in-probability convergence, and stochastic boundedness by o P and O P respectively. We write a n ≲ b n if a n ≤ Cb n for some constant C > 0, and a n b n if C 1 b n ≤ a n ≤ C 2 b n for some constants C 1 , C 2 > 0.
this section cite: []

Section: Related Literature
In view of the plethora of classical literature for central limit theorems (CLT) on SGD and its different variants Ruppert [1988], Polyak and Juditsky [1992], Chen et al. [2020], it is rather surprising that this area has remained relatively untouched for local SGD or DFL. Li et al. [2022] establish a functional CLT for local SGD, but only when the number of clients is held fixed. More recently, Gu and Chen [2024] established a central limit theory for DFL while allowing an increasing number of clients. Non-asymptotic guarantees for SA algorithms exist in terms of MSE guarantees Nemirovski et al. [2009], Moulines and Bach [2011], Lan [2012], Mou et al. [2024]. Recently, Anastasiou et al. [2019] employed Stein's method to derive Gaussian approximation for a class of smooth functions of the SGD iterates. Later, Shao and Zhang [2022] obtains the first Berry-Esseen result for online SGD. Samsonov et al. [2024] extended the result to linear stochastic approximation algorithms and temporal difference learning, before being further improved by Wu et al. [2024], Sheshukova et al. [2025].
On the other hand, to the best of our knowledge, time-uniform 'entire-path' Gaussian approximation results have not appeared in the stochastic approximation literature. From classical time-series literature, such approximations are known as "Komlos-Major-Tusnady"(KMT) approximations, and have a long history Komlós et al. [1975], Sakhanenko [1984Sakhanenko [ , 1989Sakhanenko [ , 2006]], Götze and Zaitsev [2008], Berkes et al. [2014], Karmakar and Wu [2020] and varied uses in change-point detection Wu and Zhao [2007], wavelet analysis Bonnerjee et al. [2024], simultaneous and time-uniform inference Liu and Wu [2010], Xie et al. [2020], Karmakar et al. [2022], Waudby-Smith et al. [2024]. However, this results require fast enough decay, and well-conditioned covariance structure, which are not usually available in even stochastic approximation algorithms with decaying step-size, let alone a general local SGD algorithm. Therefore, such results are not readily applicable in the current settings. sider a pre-specified "importance" or weight schedule, given by {w 1 , w 2 , . . . , w K } ∈ R K , such that K k=1 w k = 1. In an online federated learning setting, the weight schedule are typically known apriori, usually informed by the level of heterogeneity for each client, and specified by the moderator of the decentralized system. The goal of DFL is to obtain
θ ⋆ K = arg min θ K k=1 w k F k (θ) ∈ R d .
(2.1)
this section cite: ['b63', 'b60', 'b10', 'b46', 'b27', 'b57', 'b56', 'b41', 'b55', 'b1', 'b70', 'b68', 'b83', 'b72', 'b39', 'b65', 'b66', 'b67', 'b25', 'b5', 'b33', 'b84', 'b7', 'b50', 'b85', 'b34', 'b79']

Section: Communication
The client-level information is defined by loss functions f k and weights w k . A key aspect of federated learning (FL) is preserving client privacy, often achieved via a synchronization step with parameter τ ∈ N. At each τ -th step, the moderator aggregates client data and redistributes it following a policy. In decentralized SGD, averaging schemes Chaturapruek et al. [2015], Lian et al. [2017], Ivkin et al. [2019] or gossip-based methods Koloskova et al. [2020], Li et al. [2019], Qin et al. [2021], Wang and Joshi [2021] are common. In other words, a linear aggregation based on a fixed connection graph, is employed at the synchronization step. Following the notation of Gu and Chen [2024], we consider a connection network of the participating clients in the FL system, defined by an undirected graph G = (V, E) where V = {v k } K k=1 represents the set of clients and E specifies the edge set such that (i, j) ∈ E if and only if clients i and j are connected. Let C = (c ij ) ∈ R K×K be a symmetric connection matrix defined on G = (V, E), where c ij is a nonnegative constant that specifies the contribution of the j th data block to the estimation at node i. It is required that c ij > 0 if and only if (i, j) ∈ E and C1 = 1. Moreover, let c i,i > 0.
Suppose Θ t = (θ 1 t , . . . , θ K t ) ∈ R d×K denotes the local parameter updates of each client at the t-th step. Suppose the corresponding local gradient updates be summarized in the matrix
G t = K(w 1 ∇f 1 (θ 1 t-1 , ξ 1 t ), . . . , w K ∇f K (θ K t-1 , ξ K t )) ∈ R d×K .
Here, the initial points θ k 0 ∈ R d are arbitrarily initialized for k ∈ [K], and have no bearing on the theoretical results. For the sake of completion, we also re-state the local SGD algorithm using the notations and the set-up established in the preceding sections 2.1 and 2.1.1.
Algorithm 1 local SGD Input: Initializations Θ 0 = (θ 1 0 , . . . , θ K 0 ) ∈ R d×K ; Connection matrix C; Synchronization param- eter τ ∈ N; Loss functions f k (•, ξ k ), ξ k ∼ P k , k ∈ [K], weights {w k } K k=1 , number of iterations n, step-size schedules {η t } n t=1 . • Let E τ = {τ, 2τ, . . . , Lτ }, where L = n τ . • For t = 1, . . . , n : Θ t = (Θ t-1 -η t G t ) C t , C t = C, t ∈ E τ , I K , otherwise. (2.2) Output: Y n := K -1 Θ n 1 = K -1 K k=1 θ k n .
To simplify Algorithm 1, each client runs an SGD in parallel till every τ -th step, when they must synchronize their updates in order to properly solve the optimization problem (2.1). Clearly, for τ = 1, Algorithm 1 reduces to the vanilla SGD algorithm for (2.1), which hampers privacy as well as incurs great cost at each step, since typically, the number of clients K increases with the number of iterations n. On the other hand, when τ > n, there is no synchronization, and each client would solve their own local optimization problem arg min θ F k (θ), defeating the benefits of sharing information. For the purpose of this paper, we assume τ to be fixed. Moreover, on a client level, we also assume that there exists constants b 1 , b 2 > 0 such that for every k ∈
[K], b 1 ≤ Kw k ≤ b 2 .
this section cite: ['b9', 'b47', 'b30', 'b38', 'b45', 'b62', 'b77', 'b27']

Section: Berry-Esseen theorems for client-averaged local SGD updates
Before we describe the Berry-Esseen theorems, it is important we briefly describe the conditions under which it hold. We assume the usual conditions of strong-convexity (Assumption A.1), and the stochastic Lipschitz-ness of the noisy gradients ∇f k (Assumption A.2). Moreover, we also assume the continuous differentiability of f k 's (Assumption A.3). Due to space constraints, the detailed description of these assumptions, alongside an extended discussion, is relegated to Appendix A. Here, we discuss an additional condition unique to the decentralized federated learning setting.
Assumption 2.1. The connection matrix C satisfies C1 = 1 and C ⊤ = C. Moreover, if λ 1 ≥ . . . ≥ λ K denote the ordered eigen-values of C, then λ 1 = 1, and λ 2 = ρ < 1 for some ρ ∈ (0, 1).
This assumption also appears in Gu and Chen [2024]. Assumption 2.1 ensures that C is irreducible and the corresponding stationary distribution is unique; equivalently the underlying graph G is connected, ensuring an overall information sharing between each pair of clients through repeated synchronization steps. Mathematically, this can also be observed by noting that lim s→∞ C s = K -1 11 ⊤ . Now, we present the first Gaussian approximation result concerning local SGD updates.
Define the generalized Kolmogorov-Smirnov metric between two random variables Y and Z as d C (Y, Z) := sup
ℵ∈B(R d ):A convex P(Y ∈ ℵ) -P(Z ∈ ℵ) . (2.3)
Consider the local SGD output Y n from Algorithm 1. Our first theorem considers its corresponding Polyak-Ruppert averaged version
Ȳn := n -1 n t=1 Y t = K -1 K k=1 n -1 n t=1 θ k t , (2
.4) and provides a Berry-Esseen theorem, proved in appendix Section B.1. Theorem 2.1. Define A t s := t j=s+1 (I -η t A), A t t = I, where A := ∇ 2 F (θ ⋆ K ) for t ∈ [n]. Further, for s ∈ [n], define the random vectors
u s = η s K k=1 w k n j=s A j s g k (θ ⋆ K , ξ k s ), with Σ n := n -1 n s=1 E[|u s u ⊤ s |], g k (θ, ξ k ) = ∇F k (θ) -∇f k (θ, ξ k ).
Let there exist a constant C such that for
ξ k ∼ P k , k ∈ [K], it holds max k∈[K] E[|g k (θ ⋆ K , ξ k )| 2
] ≤ C. Suppose that the step-size schedules of the clients satisfy that η t = η 0 (t + k 0 ) -β for some fixed η 0 , k 0 > 0, and β ∈ (1/2, 1). Then, under Assumptions 2. 1, A.1 and A.2 and A.3 with p = 4, and Ȳn as in (2.4), it holds that
d C ( √ n( Ȳn -θ ⋆ K ), Z) ≲ 1 √ nK + n 1 2 -β √ K + n -β 2 √ K , (2.5)
where ≲ hides constants involving d, β, µ, L and ρ, and Z ∼ N (0, Σ n ).
A slightly more general result, characterizing the effects of heterogeneity and synchronization, is presented in Corollary F.1 in the appendix. We present Theorem 2.1 here due to its enhanced amenability to interpretation, which we provide in subsequent remarks. Remark 2.1. For a fixed β ∈ (1/2, 1), the term n 1/2-β √ K dominates, requiring K = o(T 2β-1 ) for the central limit theory to hold for Ȳn . This condition, also noted in Theorem 3 of Gu and Chen [2024] without justification, is explicitly clarified by (2.5). As β → 1, the rate in (2.5) becomes K/n. The inclusion of the three terms highlights the influence of K, which is unique to federated systems. The 1/ √ nK term reflects the central limit theorem's convergence rate. The n 1/2-β √ K term captures the problem's difficulty, which increases with the number of clients running local SGD in parallel. Lastly, n -β/2 K -1/2 represents the benefit of synchronization and information aggregation across clients. Even though this term is asymptotically dominated by n 1/2-β √ K, this commands considerable finite sample effects as shown in Section 4.1.
Often, due to privacy reasons, clients might be unwilling to share n -1 n i=1 θ k i at time-point n, which makes the application of Theorem 2.1 impossible. In such cases, one can simply use a corresponding Berry-Esseen bound for the end-term iterates, which we provide in the following. Theorem 2.2. Under the assumptions of Theorem 2.1, it holds that
d C (n β/2 (Y n -θ ⋆ K ), Z) ≲ n -β/2 √ K + n 1 2 -β √ K, (2.6)
where Z ∼ N (0, Σn ) with Σn :
= n β n s=1 Var(A n s K k=1 η s,k w k g k (θ ⋆ K , ξ k s )
). Theorem 2.2 is proved in appendix Section B.2. When K 1, the rate (2.6) is consistent with the well-established asymptotic theory of SGD Chung [1954], Sacks [1958], Fabian [1968] for the endterm iterates. Hereafter, till the end of this section, we will continue to analyze Ȳn further; exactly similar analysis also holds for Y n , which we do not present separately to maintain continuity.
this section cite: ['b27', 'b27', 'b64', 'b18']

Section: Estimating Σ n
In Theorem 2.1, the local SGD updates are scaled by the matrix Σ n , which is not usually known or estimable. This matrix originates as the covariance of the sum of independent vectors u s , which acts as a linearized version of the updates
Ȳt = K -1 K k=1 θ k t . If S = KVar( K k=1 w k g k (θ ⋆ K , ξ k ))
, then it can be shown that KΣ n → Σ for Σ = A -1 SA -⊤ as n → ∞. In general, we show the following theorem, proved in appendix Section C. Theorem 2.3. Under the assumptions of Theorem 2.1, it holds that
|Σ n -K -1 Σ| F ≲ K -1/2 n β-1 , (2.7)
and consequently, it holds that, with
Z ′ ∼ N (0, K -1 Σ), d C ( √ n( Ȳn -θ ⋆ K ), Z ′ ) ≲ √ K(n 1/2-β + n β-1 ).
(2.8) Specifically, if K = o(n c ) for some 0 ≤ c ≤ 1/2, the optimal β 0 ∈ (1/2, 1) minimizing (2.8) is β 0 = 3/4. Conversely, if K ≳ n c for c > 1/2, no β ∈ (1/2, 1) ensures that √ K(n 1/2-β +n β-1 ) → 0. This implies that when K n c for some c > 1/2, the Kolmogorov error remains significant, regardless of the step-size, even though central limit theory still holds for β ∈ (1/2 + c/2, 1). This phase transition highlights a new theoretical insight into the hardness of local SGD as K increases.
If K = O(
An one pass estimation of Σ is discussed in Gu and Chen [2024]. Additionally, in our appendix Section B.3, we point towards a new direction of multiplier bootstrap, leveraging our Berry-Esseen result, that does not require covariance estimation.
this section cite: ['b27']

Section: A time-uniform Gaussian coupling for the DFL updates
Section 2 quantifies the Gaussian approximation of the final local SGD updates Ȳn , with an error of order √ n in terms of iterations. However, maintaining privacy in a federated setting requires one to draw sharp inferences not only on the final output but on the entire local SGD trajectory, particularly for detecting model poisoning or adversarial attacks. From a theoretical standpoint, when the Assumption A.3 guarantees the existence of moments p > 4 (for example, when the data may be close to Gaussian), we should be able to derive sharper bounds on approximation errors, beyond the √ n result in Section 2. Since central-limit theory and Berry-Esseen estimates rely on fourth moments, we turn to classical strong approximation theory to exploit higher moments for precise bounds on the entire trajectory.
this section cite: []

Section: Motivation and Applications
A time-uniform Gaussian coupling for the entire local SGD updates has strong practical motivations, particularly for anomaly detection in "Internet-of-Vehicles" (IoV) Shalev-Shwartz et al. [2017], Ghimire and Rawat [2024], Zhu et al. [2024]. Assume that at some time point t 0 ∈ [n], a subset of clients K 0 ⊆ [K] becomes malicious. This model poisoning can be mathematically described by a change in their local risk functions F k , k ∈ K 0 , which affects the distribution of the local SGD updates Y t . This perspective extends to other attacks, such as LIE (Little is Enough) or MITM (Man in the middle) Shen et al. [2016], Blanchard et al. [2017], Yin et al. [2018], Baruch et al. [2019], where an adversary injects noise or perturbs communication at time t 0 , disrupting the distribution and trajectory of Y t for t ≥ t 0 (Ding et al. [2024]). Methods offering explicit theoretical guarantees on precisely detecting attack initiation are rare; most of the literature concentrates around robustness guarantees (error bounds, convergence rates) assuming a certain adversarial profile or detection of malicious clients [Blanchard et al., 2017, Wang et al., 2020a, Qian et al., 2024], rather than provably devising poisoning alarm. Relatedly, Mapakshi et al. [2025] observed that attacks starting in later rounds can be more damaging compared to those present from the start.
Assume that at some time point t 0 ∈ [n], a subset of clients K 0 ⊆ [K] becomes malicious. This model poisoning can be mathematically described by a change in their local risk functions F k , k ∈ K 0 , which affects the distribution of the local SGD updates Y t . To identify the time-point t 0 sequentially, we examine a CUSUM-type statistic R t := max 1≤s≤t s| Ȳs -Ȳt |, widely used in change-point analysis. We expect R t to be large for t > t 0 if an attack has altered the mean behavior of the local SGD updates at t 0 . The null distribution (i.e. when no attack takes place) of R t is usually mathematically intractable, hence posing a hindrance to performing valid inference. This necessitates a bootstrap procedure.
To identify the time-point t 0 sequentially, we examine a CUSUM-type statistic R t := max 1≤s≤t s| Ȳs -Ȳt |, widely used in change-point analysis. We expect R t to be large for t > t 0 if an attack has altered the mean behavior of the local SGD updates at t 0 .
Suppose there exists a Gaussian process G t such that a time-uniform approximation holds:
max 1≤t≤n |t Ȳt -G t | = o P ( √ n). (3.1) Let R G t = max 1≤s≤t |G s -s t G t | Then it follows that, n -1/2 max 1≤t≤n |R t -R G t | ≤ n -1/2 max 1≤t≤n max 1≤s≤t |(s Ȳs -sθ ⋆ K -G s ) - s t (t Ȳt -tθ ⋆ K -G t )| ≤ 2n -1/2 max 1≤t≤n |t Ȳt -tθ ⋆ K -G t | = o P (1).
(3.2) Equation (3.2) immediately suggests using Gaussian multiplier bootstrap leveraging G t with precisely quantifiable approximation error. In particular, if Q 1-α (X) denotes the (1 -α)-th quantile of random variable X, then for a suitable positive sequence {a n },
P(R t > Q 1-α (R G t ) + a n for some t ∈ [n]) ≤ α + P( max 1≤t≤n |R t -R G t | > a n ) → α, (3.3)
as long as n -1/2 a n ≥ c. We provide more details on these bootstrap algorithms in Appendix Section G. The two major questions that remain, are
• Does such a G t exist? If yes, can we get a rate τ n,K such that τ n,K √ n? • Can we explicitly characterize its covariance structure, so as to enable bootstrap sampling?
The main results in Section 3.2 provide answers to both the questions above.
this section cite: ['b23', 'b90', 'b71', 'b6', 'b86', 'b4', 'b16', 'b6', 'b61']

Section: Optimal coupling for local SGD
The following theorem, proved in Section D.1, establishes a Gaussian approximation echoing (3.1). Theorem 3.1.
For W k := g k (θ ⋆ K , ξ k ), ξ k ∼ P k independently for k ∈ [K], let V K = Var( K k=1 w k W k ).
Suppose Assumption A.3 holds for a general p ≥ 2. Then, under Assumptions A.1, A.2 and 2.1, (on a possibly richer probability space) there exists Z 1 , . . . , Z n i.i.d.
∼ N (0, KV K ), such that with
Y G t,1 = (I -η t A)Y G t-1,1 + η t Z t K -1/2 , Y G 0,1 = 0, (3.4) it holds that, max 1≤t≤n | t s=1 (Y s -θ ⋆ K -Y G s,1 )| = O P (n 1-β ) + o P (n 1/p K -1/2 log n).
(3.5)
We call the Gaussian approximation iterates (3.5) "Aggregated Gaussian approximation"(Aggr-GA). Note that Aggr-GA requires a complete sharing of the covariance structure to construct V K , which may affect privacy at inference-time. However, it turns out that one can further refine Theorem 3.1 to provide another Gaussian approximation results that exactly mimics the local SGD updates in their use of local structure along with periodic sharing. We call this latter approximation by Client-GA.
Theorem 3.2. Under the assumptions of Theorem 3.1, on a possibly richer probability space, for each k ∈ [K], there exist Z k 1 , . . . , Z k n i.i.d. ∼ N (0, Var(W k )), such that with ΘG t = (I -η t A) ΘG t-1 + η t M t C t , ΘG 0 = (0, . . . , 0), (3.6) where M t := K(w 1 Z 1 t , . . . , w K Z K t ) ∈ R d×K , and C t as in (2.2), it holds that max 1≤t≤n | t s=1
(Y s -θ ⋆ K -Y G s,2 )| = O P (n 1-β + (n/K) 1 4 + 1 2p (log n) 3/2 ), Y G t,2 = K -1 ΘG t 1. (3.7)
Note that for p = 2, the rates of Theorems 3.1 and 3.2 coincide. Theorem 2.2 is proved in Section D.2. In both the results, n 1-β reflects the fundamental error of a generic uniform Gaussian approximation for the local SGD updates Y n , and as such, does not depend on K. The second error decreases with the number of clients, as an increasing number of clients enables local SGD updates to track a larger horizon, and the corresponding client-averaged Y t becomes more concentrated in their trajectory towards θ ⋆ K , leading to sharper approximations. Remark 3.1 (Computational differences between Aggr-GA and Client-GA). At each iteration t, Aggr-GA has a computational complexity of O(d 2 ), since it involves generating one random sample followed by a matrix-vector multiplication. In contrast, Client-GA has a complexity of O(Kd 2 ) per iteration. Importantly, the structure of Client-GA naturally allows for parallel computation between synchronization steps, significantly reducing the computational burden while preserving periodic peer-to-peer communication. Remark 3.2 (Difference with functional CLT). Li et al. [2022] proved a functional CLT for local SGD when the number of clients K is fixed. Although such a result can theoretically be extended to the general setting considered here, nevertheless our approximations (3.5) and (3.7) are much sharper than a functional CLT approximation. As a toy example, consider the vanilla SGD setting, i.e. local SGD with τ = 1, and suppose K = 1. Suppose F (θ) = (θ -µ) 2 /2, and ∇f (θ, ξ) := θ -µ + ξ. In this setting, both Aggr-GA and Client-GA collapse to the same Gaussian approximation
Y G t = (I -η t A)Y G t-1 + η t Z t , Z t ∼ N (0, Var(ξ)), Y G 0 = 0. (3.8)
Here A = ∇ 2 F (µ) = I. On the other hand, the vanilla SGD iterates can also be seen as
Y t -µ = (I -η t A)(Y t-1 -µ) + η t ξ t .
Therefore, it can be seen that Y t -µ and Y G t have exactly the same covariance structure, i.e. Cov(
Y G s , Y G t ) = Cov(Y s , Y t )
; on the other hand, even in such a simplified setting, an approximation by Brownian motion, such as that by functional CLT, captures the covariance structure of the iterates {Y t -µ} t≥1 only in an asymptotic sense. The Gaussian approximation Y G t in (3.8) is a particular example of covariance-matching approximations, introduced by Bonnerjee et al. [2024]. By extension, same intuition holds for Aggr-GA and Client-GA as well. However, at this point, we note that the covariance-matching approximations in Bonnerjee et al. [2024] were for short-range, univariate non-stationary process. On the other hand, in the local SGD setting, the polynomially decaying step-size introduces a non-stationarity that can possibly be long-range dependent. Moreover, our result allows for multivariate parameters in a direct generalization of these aforementioned, covariance-matching approximations. We empirically validate this in Section 4.
Note that n 1-β indicates the fixed error for the local SGD updates with step-sizes η t t -β , and in order to completely underpin the effect of the assumption of additional moments p > 2, an optimal choice of step-size must be given so that n 1-β becomes negligible compared to the second error term involving the moment p. This choice is indicated in the following proposition.
Proposition 1. Grant the assumptions of Theorems 3.1 and 3.2, and consider the Gaussian approximations Y G s,1 and Y G s,2 defined therein. Suppose K = o(n c ) for some c ∈ (0, 1).
(i) If c < 2/p, then β ≥ 1 -1/p + c/2 ensures max 1≤t≤n | t s=1 (Y s -θ ⋆ K -Y G s,1 )| = o P (n 1 p -c 2 log n). (ii) For a general c ∈ (0, 1), a choice of β > 1-(1-c)( 1 4 + 1 2p ) ensures that max 1≤t≤n | t s=1 (Y s - θ ⋆ K -Y G s,2 )| = o P (n (1-c)( 1 4 + 1 2p ) (log n) 3/2 ).
Cases (i) and (ii) reveal a trade-off between Aggr-GA and Client-GA. While Aggr-GA requires information sharing at each step, yielding better approximation, it demands a stricter choice of β, since 1 - Zhang et al. [2013], Gu and Chen [2023]. Both methods require known Hessians and local covariances, estimable efficiently via Gu and Chen [2024].
1 p + c 2 > 1 -(1 -c)( 1 4 + 1 2p ) for all p > 2, c > 0. In contrast, Client-GA's local operation supports K = o(n) clients, aligning with
this section cite: ['b46', 'b7', 'b7', 'b88', 'b26', 'b27']

Section: Simulation results
Here, we summarize the various empirical exercises to accompany our theory in Sections 2 and 3. In particular, in Section 4.1, we discuss the Berry-Esseen error d C ( √ n( Ȳn -θ ⋆ K ), Z) for Z ∼ N (0, Σ n ) with varying choices of the number of iterations N , number of clients K and synchronization parameter τ . In Section 4.2, we numerically explore the computation-communication trade-off discussed in Section 2.2. Finally, in Section 4.3, we explore the approximation error of Aggr-GA and Client-GA via Q-Q plots. Detailed explanations, and additional experiments, along with the model specifications, can be found in Appendix Section F. All codes are available in github.
this section cite: []

Section: Effect of n and K on the Berry-Esseen rate
As a proxy of d C , we consider dc = sup x∈[0,c] P(| √ nΣ
-1/2 n ( Ȳn -θ ⋆ K )| ≤ x) -P(|Z| ≤ x)
where Z ∼ N (0, I) for a large enough c > 0. Figure 1 shows how dc varies with varying n, K, τ when the step-size is kept fixed at η t = 0.3t -0.75 . In particular, dc decays with N for fixed K, and increases with K for fixed n. Additional simulations and further insights can be found in Appendix section F.1.
this section cite: []

Section: Computation-communication trade-off
Here, we fix n ∈ {100, 200, 300, 400, 500}, and K = n r for r ∈ {0.2, 0.6} and numerically investigate the computation-communication trade-off hinted at in Remark 2.2. We run the local SGD algorithm with τ = 5, and η t = 0.5t -β , for β ∈ {0.85, 0.9, 0.95}. Clearly, dc decays with n for r = 0.2, and increases with n for r = 0.6, exemplifying our assertion about the computationcommunication trade-off between K and β. Appendix Section F.2 contains additional details.
this section cite: []

Section: Performance of the time-uniform Gaussian approximations
In this section, we fix N = 500, τ = 20, and let K ∈ {10, 25, 50}, and compare the quantiles of the maximum partial sums of local SGD U n , Aggr-GA U Aggr-GA n , Client-GA U Client-GA n and approximation by Brownian motion: U f-CLT n . Clearly, Aggr-GA seems to be performing the best, as suggested by Theorems 3.1 and 3.2. Furthermore, U f-CLT n consistently has the worst approximation. Additional details can be found in appendix Section F.6. (blue), U Client-GA n (green) and U f-CLT n (orange) against U n for γ = 1, N = 500, τ = 20. Here K = 10(left), K = 25(middle), K = 50(right). Rest of the FRand-eff model specifications are as in Section F.1.1.
this section cite: []

Section: Conclusion
Sharper theoretical results beyond the central limit theorem is extremely crucial to perform valid and powerful statistical inference, yet such results have not previously appeared in the literature for local SGD and in general, decentralized federated learning. In this context, to the best of our knowledge, this is the first work deriving Berry-Esseen bounds as well as sharp time-uniform Gaussian approximations over the local SGD trajectory. These results enable the development of valid and powerful statistical inference methods, including bootstrap procedures Fang et al. [2018], Fang  [2023], Lin and Reimherr [2024]. It is also crucial to make explicit the effect of synchronization in the derived rates, which can reflect more trade-offs and constitute a suitable future work.
this section cite: ['b20', 'b48']

Section: References
Ref_id:b0 Title: Qsgd: Communication-efficient sgd via gradient quantization and encoding Year: (2017)
Ref_id:b1 Title: Normal approximation for stochastic gradient descent via non-asymptotic rates of martingale clt Year: (2019)
Ref_id:b2 Title: Self-concordant analysis for logistic regression Year: (2010)
Ref_id:b3 Title: Computation-communication trade-offs and sensor selection in real-time estimation for processing networks Year: (2020)
Ref_id:b4 Title: A little is enough: Circumventing defenses for distributed learning Year: (2019-12-08)
Ref_id:b5 Title: Komlós-Major-Tusnády approximation under dependence Year: (2014)
Ref_id:b6 Title: Machine learning with adversaries: Byzantine tolerant gradient descent Year: (2017-12-04)
Ref_id:b7 Title: Gaussian approximation for nonstationary time series with optimal rate and explicit construction Year: (2024)
Ref_id:b8 Title: Optimization methods for large-scale machine learning Year: (2018)
Ref_id:b9 Title: Asynchronous stochastic convex optimization: the noise is in the noise and SGD don't care Year: (2015-12-07)
Ref_id:b10 Title: Statistical inference for model parameters in stochastic gradient descent Year: (2020)
Ref_id:b11 Title: Detailed proof of nazarov's inequality Year: (2017)
Ref_id:b12 Title: On a stochastic approximation method Year: (1954)
Ref_id:b13 Title: The total variation distance between high-dimensional gaussians with the same mean Year: (2018)
Ref_id:b14 Title: Communication trade-offs for local-sgd with large step size. Advances in Neural Information Processing Systems Year: (2019)
Ref_id:b15 Title: Bridging the gap between constant step size stochastic gradient descent and Markov chains Year: (2020)
Ref_id:b16 Title: Identifying alternately poisoning attacks in federated learning online using trajectory anomaly detection method Year: (2024)
Ref_id:b17 Title: Adaptive and robust multi-task learning Year: (2015)
Ref_id:b18 Title: On asymptotic normality in stochastic approximation Year: (1968)
Ref_id:b19 Title: Scalable statistical inference for averaged implicit stochastic gradient descent Year: (2019)
Ref_id:b20 Title: Online bootstrap confidence intervals for the stochastic gradient descent estimator Year: (2018)
Ref_id:b21 Title: A survey on decentralized federated learning Year: (2023)
Ref_id:b22 Title: The accuracy of approximation in the multidimensional invariance principle for sums of independent identically distributed random vectors with finite moments Year: (2009)
Ref_id:b23 Title: A communication-efficient machine learning framework for the internet of vehicles Year: (2024)
Ref_id:b24 Title: Sharp bounds for federated averaging (local SGD) and continuous perspective Year: (2022-03-30)
Ref_id:b25 Title: Bounds for the rate of strong approximation in the multidimensional invariance principle Year: (2008)
Ref_id:b26 Title: Distributed statistical inference under heterogeneity Year: (2023)
Ref_id:b27 Title: Statistical inference for decentralized federated learning Year: (2024)
Ref_id:b28 Title: Local sgd with periodic averaging: Tighter analysis and adaptive synchronization Year: (2019)
Ref_id:b29 Title: Central server free federated learning over single-sided trust social networks Year: (2019)
Ref_id:b30 Title: Communication-efficient distributed SGD with sketching Year: (2019-12-08)
Ref_id:b31 Title: Advances and open problems in federated learning Year: (2021)
Ref_id:b32 Title: Scaffold: Stochastic controlled averaging for federated learning Year: (2020)
Ref_id:b33 Title: Optimal Gaussian approximation for multiple time series Year: (1996)
Ref_id:b34 Title: Simultaneous inference for time-varying models Year: (2022)
Ref_id:b35 Title: Tighter theory for local sgd on identical and heterogeneous data Year: (2020)
Ref_id:b36 Title: Blockchained on-device federated learning Year: (2020)
Ref_id:b37 Title: Multi-task learning with summary statistics Year: (2023)
Ref_id:b38 Title: A unified theory of decentralized SGD with changing topology and local updates Year: (2020-07-18)
Ref_id:b39 Title: An approximation of partial sums of independent RV's and the sample DF Year: (1975)
Ref_id:b40 Title: Peer-to-peer federated learning on graphs Year: (2019)
Ref_id:b41 Title: An optimal method for stochastic composite optimization Year: (2012)
Ref_id:b42 Title: Differentially private filtering Year: (2013)
Ref_id:b43 Title: The stochastic gradient descent from a nonlinear time series perspective Year: (2024)
Ref_id:b44 Title: Federated optimization in heterogeneous networks Year: (2020)
Ref_id:b45 Title: Communication-efficient local decentralized sgd methods Year: (2019)
Ref_id:b46 Title: Statistical estimation and online inference via local sgd Year: (2022)
Ref_id:b47 Title: Can decentralized algorithms outperform centralized algorithms? A case study for decentralized parallel stochastic gradient descent Year: (2017-12-04)
Ref_id:b48 Title: Smoothness adaptive hypothesis transfer learning Year: (2024)
Ref_id:b49 Title: Don't use large mini-batches, use local sgd Year: (2018)
Ref_id:b50 Title: Simultaneous nonparametric inference of time series Year: (2010)
Ref_id:b51 Title: Temporal Analysis of Adversarial Attacks in Federated Learning Year: ()
Ref_id:b52 Title:  Year: (2025)
Ref_id:b53 Title: Communication-efficient learning of deep networks from decentralized data Year: (2017)
Ref_id:b54 Title: Sequential gaussian approximation for nonstationary time series in high dimensions Year: (2023)
Ref_id:b55 Title: Optimal and instance-dependent guarantees for Markovian linear stochastic approximation Year: (2024)
Ref_id:b56 Title: Non-asymptotic analysis of stochastic approximation algorithms for machine learning Year: (2011)
Ref_id:b57 Title: Robust stochastic approximation approach to stochastic programming Year: (2009)
Ref_id:b58 Title: Fedgkd: Unleashing the power of collaboration in federated graph neural networks Year: (2023)
Ref_id:b59 Title: Optimum bounds for the distributions of martingales in Banach spaces Year: (1994)
Ref_id:b60 Title: Acceleration of stochastic approximation by averaging Year: (1992)
Ref_id:b61 Title: ByMI: Byzantine Machine Identification with False Discovery Rate Control Year: (2024-07)
Ref_id:b62 Title: Communication-efficient decentralized local sgd over undirected networks Year: (2021)
Ref_id:b63 Title: Efficient estimations from a slowly convergent robbins-monro process Year: (1988)
Ref_id:b64 Title: Asymptotic distribution of stochastic approximation procedures Year: (1958)
Ref_id:b65 Title: Rate of convergence in the invariance principle for variables with exponential moments that are not identically distributed Year: (1984)
Ref_id:b66 Title: On the accuracy of normal approximation in the invariance principle Year: (1989)
Ref_id:b67 Title: Estimates in the invariance principle in terms of truncated power moments Year: (2006)
Ref_id:b68 Title: Gaussian approximation and multiplier bootstrap for polyak-ruppert averaged linear stochastic approximation with applications to td learning Year: (2024)
Ref_id:b69 Title: On a formal model of safe and scalable selfdriving cars Year: (2017)
Ref_id:b70 Title: Berry-Esseen bounds for multivariate nonlinear statistics with applications to M-estimators and stochastic gradient descent algorithms Year: (2022)
Ref_id:b71 Title: Auror: defending against poisoning attacks in collaborative deep learning systems Year: (2016)
Ref_id:b72 Title: Gaussian approximation and multiplier bootstrap for stochastic gradient descent Year: (2025)
Ref_id:b73 Title: Privacy-preserving decentralized federated learning: Algorithms, challenges, and opportunities Year: (2023)
Ref_id:b74 Title: Local SGD converges fast and communicates little Year: (2019-05-06)
Ref_id:b75 Title: Computation hierarchy for in-network processing Year: (2005)
Ref_id:b76 Title: Attack of the tails: Yes, you really can backdoor federated learning Year: (2020)
Ref_id:b77 Title: Cooperative SGD: a unified framework for the design and analysis of localupdate SGD algorithms Year: (2021)
Ref_id:b78 Title: Tackling the objective inconsistency problem in heterogeneous federated optimization Year: (2020)
Ref_id:b79 Title: Time-uniform central limit theory and asymptotic confidence sequences Year: (2024)
Ref_id:b80 Title: Weighted averaged stochastic gradient descent: Asymptotic normality and optimality Year: (2023)
Ref_id:b81 Title: Minibatch vs local SGD for heterogeneous distributed learning Year: (2020)
Ref_id:b82 Title: Is local SGD better than minibatch sgd? Year: (2020-07-18)
Ref_id:b83 Title: Statistical inference for temporal difference learning with linear function approximation Year: (2024)
Ref_id:b84 Title: Inference of trends in time series Year: (2007)
Ref_id:b85 Title: Asynchronous federated optimization Year: (2020)
Ref_id:b86 Title: Byzantine-robust distributed learning: Towards optimal statistical rates Year: (2018)
Ref_id:b87 Title: Decentralized federated learning: A survey and perspective Year: (2024)
Ref_id:b88 Title: Communication-efficient algorithms for statistical optimization Year: (2013)
Ref_id:b89 Title: Online bootstrap inference with nonconvex stochastic gradient descent estimator Year: (2023)
Ref_id:b90 Title: FLUK: protecting federated learning against malicious clients for internet of vehicles Year: (2024)
Ref_id:b91 Title: Online covariance matrix estimation in stochastic gradient descent Year: (2023)
