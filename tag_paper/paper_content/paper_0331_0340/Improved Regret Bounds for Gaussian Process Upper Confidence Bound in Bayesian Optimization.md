Title: Improved Regret Bounds for Gaussian Process Upper Confidence Bound in Bayesian Optimization
Abstract: This paper addresses the Bayesian optimization problem (also referred to as the Bayesian setting of the Gaussian process bandit), where the learner seeks to minimize the regret under a function drawn from a known Gaussian process (GP). Under a Matérn kernel with a certain degree of smoothness, we show that the Gaussian process upper confidence bound (GP-UCB) algorithm achieves 𝑂 ( √ 𝑇) cumulative regret with high probability. Furthermore, our analysis yields 𝑂 ( √︁ 𝑇 ln 2 𝑇) regret under a squared exponential kernel. These results fill the gap between the existing regret upper bound for GP-UCB and the best-known bound provided by Scarlett [46]. The key idea in our proof is to capture the concentration behavior of the input sequence realized by GP-UCB, enabling a more refined analysis of the GP's information gain. √𝑇 ln 𝑇) cumulative regret. Then, the natural question is whether there is further room for improvement in the existing regret upper bound of GP-UCB. This paper provides an affirmative answer to this question by showing that GP-UCB achieves 𝑂 ( √ 𝑇) regret with high probability.Contribution. We summarize our contributions as follows.• We show that the GP-UCB proposed by Srinivas et al. [51]  achieves 𝑂 ( √ 𝑇) regret with high probability under a Matérn kernel with a certain degree of smoothness (precise condition is provided in Theorem 3). Here, 𝑂 (•) is the order notation that hides polylogarithmic dependence. This result is comparable to state-of-the-art 𝑂 ( √ 𝑇 ln 𝑇) regret provided by Scarlett [46] up to a polylogarithmic factor and strictly improves upon the existing 𝑂 (𝑇 𝜈+𝑑 2𝜈+𝑑 ) upper bound of GP-UCB [51,58]. Here, 𝑑 and 𝜈 denote the dimension of the input domain and smoothness parameter, respectively.39th Conference on Neural Information Processing Systems (NeurIPS 2025). Probabilistic property of GP sample path.The existing theory of GP-UCB under the Bayesian setting utilizes the regularity conditions of the realized sample path of GP. We summarize the existing known properties of the GP sample path in the following lemmas. Lemma 1 (Lipchitz condition of sample path, e.g., [51]). Suppose 𝑘 = 𝑘 SE or 𝑘 = 𝑘 Matérn with 𝜈 > 2. Assume Assumption 1. Then, there exist the constants 𝑎, 𝑏 > 0 such thatLemma 2 (Sample path condition for the global maximizer, e.g., [13,14,46]). Suppose 𝑘 = 𝑘 SE or 𝑘 = 𝑘 Matérn with 𝜈 > 2. Assume Assumption 1. Then, for any 𝛿 GP ∈ (0, 1), there exist the strictly positive constants 𝑐 gap , 𝑐 sup , 𝑐 quad , 𝜌 quad > 0 such that the following statements simultaneously hold with probability at least 1 -𝛿 GP :1. The function 𝑓 has a unique maximizer x * ∈ X such that 𝑓 (x * ) > 𝑓 ( x * ) + 𝑐 gap holds for any local maximizer x * ∈ X of 𝑓 .√ 𝑇), while it is strictly smaller than 𝑂 (𝑇 𝜈+𝑑 2𝜈+𝑑 ) of the original GP-UCB's analysis.

Section: Introduction
We study the Bayesian optimization (BO) problem, where the learner seeks to minimize the regret under a random function drawn from a known Gaussian process (GP) [18,19]. Throughout this paper, we focus on the GP-UCB algorithm [51], which combines the posterior distribution of GP with the optimism principle. Due to its simple algorithm construction and general theoretical framework provided by Srinivas et al. [51], GP-UCB has played an important role in the advancement of the BO field. On the other hand, our theoretical understanding of the performance of GP-UCB has not been improved from [51] in the Bayesian setting, while its frequentist counterpart is studied in several existing works [11,61]. Specifically, the current regret upper bound for GP-UCB, as provided by Srinivas et al. [51], is known to be worse than that of the algorithm in [46], which achieves stateof-the-art 𝑂 ( • Furthermore, for a squared exponential kernel, we establish 𝑂 √︁ 𝑇 ln 2 𝑇 cumulative regret of GP-UCB. This improves the existing 𝑂 √︁ 𝑇 ln 𝑑+2 𝑇 upper bound provided by Srinivas et al. [51] for any 𝑑 ≥ 1.
• The key idea behind our analysis is to refine the existing information gain bounds by leveraging algorithm-dependent behavior and sample path properties of the GP. We also discuss the applicability of this technique to other algorithms and settings in Section 4.
this section cite: ['b17', 'b18', 'b50', 'b50', 'b50', 'b10', 'b60', 'b50', 'b45', 'b50']

Section: Related Works
BO has been extensively studied in the past few decades. Some of them are constructed so as to maximize the utility-based acquisition function defined through the GP posterior, including expected improvement [37], knowledge gradient [17], and the entropy-based algorithms [24]. The theoretical aspect of BO has also been actively studied through the lens of the bandit algorithms, such as GP-UCB [51], Thompson sampling [43], and information directed sampling [44]. In contrast to the noisy observation setting, which these algorithms focus on, algorithms for the noise-free setting form a separate line of research [14,23,32]. Extensions of these algorithms to more advanced settings have also been well-studied, e.g., contextual [34], parallel observation [15], high-dimensional [29], timevarying [6], and multi-fidelity setting [30]. Unlike the Bayesian assumption on the objective function adopted in this paper, existing works also extensively study the frequentist assumption of the function, which is also referred to as the frequentist setting of BO or GP bandits [7,9,11,26,35,45,47,56,59].
Among the existing studies, [46] is closely related to this paper, which propose a successive elimination-based algorithm and shows an 𝑂 ( √ 𝑇 ln 𝑇) upper bound and an Ω( √ 𝑇) lower bound of the cumulative regret for a one-dimensional BO problem. The fundamental theoretical assumptions and the high-level idea of our analysis are built on the proof provided by Scarlett [46]. Following [46], Wang et al. [60] also proves similar regret guarantees under the one-dimensional Brownian motion.
In addition to [46], some parts of our analysis are inspired by the technique leveraged in [8,28]. Firstly, Cai et al. [8] studies the GP-UCB algorithm through a relaxed version of regret, which is called lenient regret. In our analysis, the cumulative regret is decomposed into the lenient regretbased term, and we leverage their technique to analyze it. Secondly, Janz et al. [28] proposed the input partitioning-based algorithm for obtaining a superior regret in the frequentist setting. Roughly speaking, the high-level idea of their analysis is based on the fact that tighter information gain bounds can be obtained within a properly shrinking partition of the input. The key idea provided in Section 3.1 is motivated by this fact, while our analysis itself is substantially different from that in [28].
this section cite: ['b36', 'b16', 'b23', 'b50', 'b42', 'b43', 'b13', 'b22', 'b31', 'b33', 'b14', 'b28', 'b5', 'b29', 'b6', 'b8', 'b10', 'b25', 'b34', 'b44', 'b46', 'b55', 'b58', 'b45', 'b45', 'b45', 'b59', 'b45', 'b7', 'b27', 'b7', 'b27', 'b27']

Section: Preliminaries
Let 𝑓 : X → R be a black-box objective function whose input domain X is X := [0, 𝑟] 𝑑 with some 𝑟 > 0. At each step 𝑡 ∈ N + , the learner chooses a query point x 𝑡 ∈ X, and then receives a noisy observation 𝑦 𝑡 = 𝑓 (x 𝑡 ) + 𝜖 𝑡 . Here, 𝜖 𝑡 is a mean-zero noise random variable. We consider a Bayesian setting, where the objective function 𝑓 and the noise sequence (𝜖 𝑡 ) are drawn from a known zero-mean Gaussian process (GP) and a Gaussian distribution, respectively. We formally describe it using the following assumptions.
Assumption 1. Let 𝑘 : X × X → R be the known positive definite kernel with ∀x ∈ X, 𝑘 (x, x) ≤ 1. Then, assume 𝑓 ∼ GP (0, 𝑘), where GP (0, 𝑘) denotes the mean-zero GP characterized by the covariance function 𝑘.
Assumption 2. The noise sequence (𝜖 𝑡 ) 𝑡 ∈N + is mutually independent. Furthermore, assume 𝜖 𝑡 ∼ N (0, 𝜎 2 ), where 𝜎 > 0 is the known constant. Here, N (𝜇, 𝜎 2 ) denotes the Gaussian distribution with mean 𝜇 and variance 𝜎 2 .
These are standard sets of assumptions in the existing theory of BO [43,51]. Specifically, in Assumption 1, we focus on the following squared exponential (SE) kernel 𝑘 SE and Matérn kernel Algorithm 1 Gaussian process upper confidence bound Require: Kernel 𝑘, confidence width parameters (𝛽 1/2 𝑡 ) 𝑡 ∈N + . 1: for 𝑡 = 1, 2, . . . do 2:
x 𝑡 ← arg max x∈ X 𝜇(x; X 𝑡 -1 , y 𝑡 -1 ) + 𝛽 1/2 𝑡 𝜎(x; X 𝑡 -1 ).
3:
Observe 𝑦 𝑡 and update the posterior mean and variance. 4: end for 𝑘 Matérn :
𝑘 SE (x, x) = exp - ∥x -x∥ 2 2 2ℓ 2 , 𝑘 Matérn (x, x) = 2 1-𝜈 Γ(𝜈) √ 2𝜈∥x -x∥ 2 ℓ 𝜈 𝐽 𝜈 √ 2𝜈∥x -x∥ 2 ℓ ,(1)
where ℓ > 0 and 𝜈 > 0 are the known lengthscale and smoothness parameters, respectively. In addition, 𝐽 𝜈 (•) and Γ(•) respectively denote modified Bessel and Gamma functions. Under Assumptions 1 and 2, the learner can infer the function 𝑓 through the GP posterior distribution. Let H 𝑡 := (x 𝑖 , 𝑦 𝑖 ) 𝑖 ≤𝑡 be the history that the learner obtained up to the end of step 𝑡. Given H 𝑡 , the posterior distribution of 𝑓 is again GP, whose posterior mean and variance are respectively defined as
𝜇(x; X 𝑡 , y 𝑡 ) = k(X 𝑡 , x) ⊤ (K(X 𝑡 , X 𝑡 ) + 𝜎 2 I 𝑡 ) -1 y 𝑡 ,(2)
𝜎 2 (x; X 𝑡 ) = 𝑘 (x, x) -k(X 𝑡 , x) ⊤ (K(X 𝑡 , X 𝑡 ) + 𝜎 2 I 𝑡 ) -1 k 𝑡 (X 𝑡 , x),(3)
where k(X 𝑡 , x) := [𝑘 (x, x)] x∈X 𝑡 and y 𝑡 := (𝑦 1 , . . . , 𝑦 𝑡 ) ⊤ are the 𝑡-dimensional kernel and output vectors, respectively. Here, we set X 𝑡 = (x 1 , . . . , x 𝑡 ). Furthermore, K(X 𝑡 , X 𝑡 ) := [𝑘 (x, x)] x, x∈X 𝑡 and I 𝑡 denote 𝑡 × 𝑡-gram matrix and 𝑡 × 𝑡-identity matrix, respectively.
Learner's goal. Under the total step size 𝑇 ∈ N + , the learner's goal is to minimize the cumulative regret 𝑅 𝑇 := 𝑡 ∈ [𝑇 ] 𝑓 (x * ) -𝑓 (x 𝑡 ), where x * ∈ argmax x∈ X 𝑓 (x) and [𝑇] = {1, . . . , 𝑇 }.
this section cite: ['b42', 'b50']

Section: Maximum information gain.
To quantify the regret, the existing theory utilizes the following information-theoretic quantity 𝛾 𝑇 (X) arising from GP:
𝛾 𝑇 (X) = sup x 1 ,...,x 𝑇 ∈ X 𝐼 (X 𝑇 ), where 𝐼 (X 𝑇 ) = 1 2 ln det(I 𝑇 + 𝜎 -2 K(X 𝑇 , X 𝑇 )).(4)
The quantity 𝛾 𝑇 (X) is referred to as the maximum information gain (MIG) over X [51], since 𝐼 (X 𝑇 ) equals the mutual information between the function values ( 𝑓 (x 𝑡 )) 𝑡 ∈ [𝑇 ] and the outputs (𝑦 𝑡 ) 𝑡 ∈ [𝑇 ] under Assumptions 1 and 2, and the input sequence X 𝑡 = (x 1 , . . . , x 𝑡 ). MIG plays a vital role in the theoretical analysis of BO, and its increasing speed is analyzed in several commonly used kernels. For example, 𝛾 𝑇 (X) = 𝑂 (ln 𝑑+1 𝑇) as 𝑇 → ∞ under 𝑘 = 𝑘 SE [51]. For the notational convenience, we also define 𝛾 𝑖 (X) = 𝛾 ⌈𝑖 ⌉ (X) for any non-integer 𝑖 > 0.
2. The sup-norm of the sample path is bounded as ∥ 𝑓 ∥ ∞ ≤ 𝑐 sup .
3. The function 𝑓 satisfies ∀x ∈ B 2 (𝜌 quad ; x * ), 𝑓 (x * ) -𝑐 quad ∥x * -x∥ 2 2 ≥ 𝑓 (x), where B 2 (𝜌 quad ; x * ) := {x ∈ X | ∥x * -x∥ 2 ≤ 𝜌 quad } is the L2-ball on X, whose radius and center are 𝜌 quad and x * , respectively.
Lemma 1 states that the sample path 𝑓 of GP is a Lipschitz function with high probability. This property is leveraged in the theory of GP-UCB to control the discretization error arising from the confidence bound construction in the continuous input domain. As described in [51], Lemma 1 is a direct consequence of Theorem 5 in [21] under the existence of fourth-order mixed partial derivatives of the kernel, which are satisfied under 𝑘 = 𝑘 SE and 𝑘 = 𝑘 Matérn with 𝜈 > 2foot_0 . Lemma 2 specifies the regularity condition of 𝑓 related to the maximizer x * . Here, property 1 is implied from the fact that the GP-sample path has a unique maximizer almost surely under 𝑘 SE and 𝑘 Matérn [e.g., Lemma 2.6 in 33]. Property 2 is implied from, e.g., the compactness of X and the almost-sure continuity of the sample path under 𝑘 SE and 𝑘 Matérn . Property 3 also holds automatically under 𝑘 = 𝑘 SE and 𝑘 = 𝑘 Matérn with 𝜈 > 2 and is used in existing works. See Theorem 5 in [13], Assumption 3 in [46], and the discussions provided by them for further details. Note that the properties in Lemma 2 are not used in the existing proof of GP-UCB in [51]. As described in the next section, we analyze the realized input sequence X 𝑇 of GP-UCB by relating it to conditions in Lemma 2.
this section cite: ['b50', 'b50', 'b50', 'b20', 'b12', 'b45', 'b50']

Section: Summary of existing analysis of GP-UCB.
We briefly summarize the existing analysis of GP-UCB (Algorithm 1) provided by Srinivas et al. [51]. Based on Assumptions 1 and 2, we can construct the high-probability confidence bound of the underlying function value 𝑓 (x) for each x and 𝑡 ∈ N + through the posterior distribution of 𝑓 (x). Specifically, by choosing a properly designed finite representative input set X 𝑡 ⊂ X and taking into account the discretization error with Lemma 1, Srinivas et al. [51] showed the following events hold simultaneously with probability at least 1 -𝛿:
1. Confidence bound. For any 𝑡 ∈ N + , the function value at the queried point
x 𝑡 satisfies 𝜇(x 𝑡 ; X 𝑡 -1 , y 𝑡 -1 ) -𝛽 1/2 𝑡 𝜎(x 𝑡 ; X 𝑡 -1 ) ≤ 𝑓 (x 𝑡 ).
Furthermore, for any 𝑡 ∈ N + , any function value 𝑓 (x) on X 𝑡 satisfies 𝑓 (x) ≤ 𝜇(x; X 𝑡 -1 , y 𝑡 -1 ) + 𝛽 1/2 𝑡 𝜎(x; X 𝑡 -1 ). 2. Discretization error. The discretization error arising from X 𝑡 is at most 1/𝑡 2 . Namely, | 𝑓 (x) -𝑓 ( [x] 𝑡 )| ≤ 1/𝑡 2 holds for any x ∈ X and 𝑡 ∈ N + , where [x] 𝑡 denotes one of the closest points of x on X 𝑡 .
In the above statements, 𝛽 1/2 𝑡 is chosen based on the constants 𝑎, 𝑏 in Lemma 1 and the length 𝑟 of X, and is defined as
𝛽 𝑡 = 2 ln 2𝑡 2 𝜋 2 3𝛿 + 2𝑑 ln 𝑡 2 𝑑𝑏𝑟 √︂ ln 4𝑑𝑎 𝛿 .(6)
The above two events and the UCB-selection rule for x 𝑡 imply
𝑅 𝑇 = 𝑇 ∑︁ 𝑡=1 𝑓 (x * ) -𝑓 ( [x * ] 𝑡 ) + 𝑇 ∑︁ 𝑡=1 𝑓 ( [x * ] 𝑡 ) -𝑓 (x 𝑡 ) ≤ 𝜋 2 6 + 2𝛽 1/2 𝑇 𝑇 ∑︁ 𝑡=1 𝜎(x 𝑡 ; X 𝑡 -1 ).(7)
In the above expression, the upper bound
𝑇 𝑡=1 𝑓 (x * )-𝑓 ( [x * ] 𝑡 ) ≤ 𝑇 𝑡=1 1/𝑡 2 ≤ 𝜋 2 /6 follows from the second event (discretization error). The inequality 𝑇 𝑡=1 𝑓 ( [x * ] 𝑡 ) -𝑓 (x 𝑡 ) ≤ 2𝛽 1/2 𝑇 𝑇 𝑡=1 𝜎(x 𝑡 ; X 𝑡 -1
) also follows from the first event (confidence bound) and the definition of x 𝑡 . See the proof of Theorem 2 in [51] for details. The above inequality suggests that the regret upper bound of GP-UCB depends on the sum of the posterior standard deviations 𝑇 𝑡=1 𝜎(x 𝑡 ; X 𝑡 -1 ). Srinivas et al. [51] provides the upper bound of this term by leveraging the information gain 𝐼 (X 𝑇 ) as follows:
𝑇 ∑︁ 𝑡=1 𝜎(x 𝑡 ; X 𝑡 -1 ) ≤ √︁ 𝐶𝑇 𝐼 (X 𝑇 ) ≤ √︁ 𝐶𝑇 𝛾 𝑇 (X),(8)
where 𝐶 =
this section cite: ['b50', 'b50', 'b50', 'b50']

Section: Improved Regret Bound for GP-UCB
The following theorem presents our main result: a new regret upper bound for GP-UCB. Theorem 3 (Improved regret upper bound for GP-UCB). Suppose Assumptions 1 and 2 hold. Set 𝑘 = 𝑘 SE or 𝑘 = 𝑘 Matérn with 𝜈 > 2. Furthermore, assume that 𝑑, 𝜈, ℓ, 𝑟 , and 𝜎 2 are fixed constants. Fix any 𝛿 GP ∈ (0, 1), and set the confidence width parameter 𝛽 𝑡 of GP-UCB as defined in Eq. ( 6) with any fixed 𝛿 ∈ (0, 1 -𝛿 GP ). Then, with probability at least 1 -𝛿 GP -𝛿, the cumulative regret of GP-UCB (Algorithm 1) satisfies
𝑅 𝑇 =        𝑂 √ 𝑇 if 𝑘 = 𝑘 Matérn with 2𝜈 + 𝑑 ≤ 𝜈 2 , 𝑂 √︁ 𝑇 ln 2 𝑇 if 𝑘 = 𝑘 SE . (9
)
The hidden constants in the above expressions may depend on ln(1/𝛿), 𝑑, 𝜈, ℓ, 𝑟, 𝜎 2 , and the constants 𝑐 sup , 𝑐 gap , 𝜌 quad , 𝑐 quad corresponding with 𝛿 GP , which are guaranteed to exist by Lemma 2.
We would like to note the following three aspects of our results. First, the constants associated with the sample path properties defined in Lemma 2 are used solely for analyzing the regret. On the other hand, the existing algorithm provided by Scarlett [46], which shows the same 𝑂 ( √ 𝑇) regret as ours, requires prior information about these constants for the algorithm run. This is often unrealistic in practice. Secondly, our result does not imply the upper bound of Bayesian expected regret E[𝑅 𝑇 ]. The main issue is that the dependence of the constants in Lemma 2 on 𝛿 GP is not explicitly known. We leave future work to break this limitation; however, note that the same limitation exists in the algorithm provided by Scarlett [46]. Thirdly, our results in Theorem 3 only focus on the dependence of the total step size 𝑇 in the regret. Therefore, we cannot claim any improvements of the regret on the dependence of the other parameters. For example, compared to the existing 𝑅 𝑇 = 𝑂 ( √︁ 𝑇 ln 𝑑+2 𝑇) regret under 𝑘 = 𝑘 SE , our regret upper bound 𝑅 𝑇 = 𝑂 ( √︁ 𝑇 ln 2 𝑇) indeed avoids the dependence of 𝑑 in the logarithmic factor; however, under the joint limit of 𝑑 and 𝑇 (𝑑, 𝑇 → ∞), it easily behaves super-linearly even under the slowly increasing 𝑑 (e.g., 𝑑 = Θ(ln ln 𝑇)) due to the hidden constants in the regret.
this section cite: ['b45', 'b45']

Section: Intuitive Explanation of our Analysis
Before we describe the proof, we provide an intuitive explanation of why GP-UCB achieves a tighter regret than the existing 𝑂 ( √︁ 𝛽 𝑇 𝑇 𝛾 𝑇 (X)) upper bound. The motivation for our new analysis comes from the observation that the upper bound of the information gain: 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) in Eq. ( 8) is not always tight depending on the specific realization of the input sequence X 𝑇 . To see this, let us observe the following two simple extreme cases of X 𝑇 where the inequality 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) is loose and tight:
• Case I: 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) is loose: Let us assume all the input is equal to the unique maximizer x * (namely, ∀𝑡 ∈ [𝑇], x 𝑡 = x * ). Then, when the kernel function satisfies ∀x ∈ X, 𝑘 (x, x) = 1 as with 𝑘 SE and 𝑘 Matérn , we have:
𝐼 (X 𝑇 ) = 1 2 ln det(I 𝑇 + 𝜎 -2 K(X 𝑇 , X 𝑇 )) = 1 2 𝑇 ∑︁ 𝑖=1 ln(1 + 𝜎 -2 𝜆 𝑖 ) = 1 2 ln(1 + 𝜎 -2 𝑇), (10
)
where 𝜆 𝑖 is the 𝑖-th eigenvalue of K(X 𝑇 , X 𝑇 ) = 11 ⊤ with 1 = (1, . . . , 1) ⊤ ∈ R 𝑇 . The third equation uses the fact that 11 ⊤ is rank 1, and its unique non-zero eigenvalue is 𝑇.
• Case II: 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) is tight: Let us assume that (x 𝑡 ) is the same as the input sequence generated by the maximum variance reduction (MVR) algorithm (namely, ∀𝑡 ∈ [𝑇], x 𝑡 ∈ argmax x∈ X 𝜎(x; X 𝑡 -1 )) [51,56]. Then, from the discussion in Sections 2 and 5 in [51], we already know that 𝛾 𝑇 (X) ≤ (1 -1/𝑒) -1 𝐼 (X 𝑇 ). This suggests that 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) is tight up to a constant factor when X 𝑇 is realized by MVR. under GP-UCB or MVR. We also plot 𝐼 (X 𝑡 ) := 0.5 ln(1 + 𝜎 -2 𝑡), corresponding to Case I described in Section 3.1. We can observe that the inputs selected by GP-UCB are concentrated around the maximizer from the left figure. Then, from the right figure, we also observe that the corresponding information gain increases more slowly than that of MVR, and behaves similarly to Case I on 𝑡 ≥ 30.
More comprehensive empirical results are also provided in Appendix D.
From Case I, we observe that 𝐼 (X 𝑇 ) satisfies Θ(ln 𝑇) ≤ 𝐼 (X 𝑇 ) ≤ 𝛾 𝑇 (X) depending on X 𝑇 . Furthermore, by comparing the input sequences in cases I and II, we expect that 𝐼 (X 𝑇 ) becomes small if X 𝑇 concentrates around the neighborhood of x * , while 𝐼 (X 𝑇 ) becomes large if X 𝑇 spreads over the entire input domain X. Then, from the fact that the worst-case regret of GP-UCB increases sub-linearly with the speed of 𝑂 ( √︁ 𝛽 𝑇 𝑇 𝛾 𝑇 (X)), we can deduce that the input sequence X 𝑇 of GP-UCB will eventually concentrate around the maximizer x * if x * is unique and ∥ 𝑓 ∥ ∞ is not extremely small 2 . We provide an illustrative image in Figure 1. Our proof is designed so as to capture the above intuition that 𝐼 (X 𝑇 ) could be improved from 𝛾 𝑇 (X) to Θ(ln 𝑇) under "favorable" sample path 𝑓 .
this section cite: ['b50', 'b55', 'b50']

Section: Proof of Theorem 3
Let A be an event such that the two high-probability events of the original GP-UCB proof (described in the last paragraph in Section 2) and Lemma 2 with the confidence level 𝛿 GP simultaneously hold. Note that event A occurs with probability at least 1 -𝛿 GP -𝛿 from the union bound. Therefore, it is enough to prove our upper bound under A. To encode the high-level idea in the previous section, we need to capture the concentration behavior of the input sequence X 𝑇 around the maximizer x * . From this motivation, given some constant 𝜀 > 0, we decompose the regret as 𝑅 𝑇 = 𝑅 (1)  𝑇 (𝜀) + 𝑅 (2)  𝑇 (𝜀), where:
𝑅 (1) 𝑇 (𝜀) = ∑︁ 𝑡 ∈ T ( 𝜀) 𝑓 (x * ) -𝑓 (x 𝑡 ), 𝑅 (2) 𝑇 (𝜀) = ∑︁ 𝑡 ∈ T 𝑐 ( 𝜀) 𝑓 (x * ) -𝑓 (x 𝑡 ).(11)
We set T (𝜀) = {𝑡 ∈ [𝑇] | 𝑓 (x * ) -𝑓 (x 𝑡 ) > 𝜀} and T 𝑐 (𝜀) = [𝑇] \ T (𝜀) in the above definition.
A key observation is that, if we set sufficiently small 𝜀 depending on the constants in Lemma 2, the inputs (x 𝑡 ) in 𝑅 (2)  𝑇 (𝜀) (namely, inputs (x 𝑡 ) such that 𝑓 (x * ) -𝑓 (x 𝑡 ) ≤ 𝜀 holds) are on the locally quadratic region around the maximizer x * due to conditions 1 and 3 in Lemma 2. The formal descriptions are provided in Lemma 20 in Appendix C. This fact is originally leveraged in [46] to analyze the successive elimination-based algorithm. In the analysis of GP-UCB, it enables us to 2 Specifically, if 𝑇 ∥ 𝑓 ∥ ∞ ≤ 𝑂 ( √︁ 𝛽 𝑇 𝑇 𝛾 𝑇 (X)), we cannot make any claims about X 𝑇 based on the worst-case bound since any sequence X 𝑇 satisfies the worst-case bound without concentrating around maximizer. This is why our analysis technique does not improve the worst-case regret in the frequentist setting. Indeed, in the proof of the worst-case lower bound for the frequentist setting [47], the existence of the function 𝑓 with
𝑇 ∥ 𝑓 ∥ ∞ = 𝑂 ( √︁ 𝛽 𝑇 𝑇 𝛾 𝑇 (X)) is guaranteed.
analyze the behavior of the sub-input sequence {x 𝑡 | 𝑓 (x * ) -𝑓 (x 𝑡 ) ≤ 𝜀} through the regularity constant 𝑐 quad . Below, we formally give the upper bound for 𝑅 (2)  𝑇 (𝜀). Lemma 4 (General upper bound of 𝑅 (2)  𝑇 ). Suppose (x 𝑡 ) 𝑡 ∈ [𝑇 ] is the input query sequence realized by the GP-UCB algorithm. Furthermore, let 𝛾 𝑡 is the upper bound of MIG 𝛾 𝑡 (X) such that 𝛾 𝑡 /𝑡 is non-increasing on [𝑇, ∞) with some 𝑇 ∈ N +foot_2 . Then, under event A, we have 𝑅
2) 𝑇 (𝜀) ≤ 2𝑐 sup 𝑇 + 𝜋 2 3 log 2 𝑇 + 1 + 2 √︁ 2𝐶 𝛽 𝑇 𝑇 √ 2 -1 max 𝑖∈ [𝑖] √︂ 𝛾 (𝑇/2 𝑖-1 ) B 2 √︃ 𝑐 -1 quad 𝜂 𝑖 ; x * , where 𝐶 = 2/ln(1 + 𝜎 -2 ), 𝑖 = ⌊log 2 𝑇 𝑇 ⌋ + 1, 𝜂 𝑖 = 2 2 √︃ 𝐶 𝛽 𝑇 (𝑇/2 𝑖-1 ) 𝛾 𝑇/2 𝑖-1 + 𝜋 2 6 (𝑇/2 𝑖-1 )
, and 𝜀 = min{𝑐 gap , 𝑐 quad 𝜌 2 quad }.
We give the full proof in Appendix A.1. Here, the dominant term in the above lemma is given as:
𝑅 (2) 𝑇 (𝜀) = 𝑂 max 𝑖 √︂ 𝑇 𝛾 (𝑇/2 𝑖-1 ) B 2 √︃ 𝑐 -1 quad 𝜂 𝑖 ; x * . (12
)
Note that 𝜂 𝑖 is decreasing as the time index 𝑇/2 𝑖-1 of MIG increases. In other words, the input domain B 2 √︃ 𝑐 -1 quad 𝜂 𝑖 ; x * of MIG shrinks as the time index 𝑇/2 𝑖-1 increases. This property is beneficial for obtaining a tighter upper bound than that from the existing technique. For example, under 𝑘 = 𝑘 Matérn with 2𝜈 + 𝑑 ≤ 𝜈 2 , we can confirm that the dominant polynomial term in MIG is canceled out by the shrinking of the input domain in MIG. Namely, we can obtain the following result under 𝑘 = 𝑘 Matérn :
max 𝑖 𝛾 (𝑇/2 𝑖-1 ) B 2 √︃ 𝑐 -1 quad 𝜂 𝑖 ; x * = 𝑂 (1) (as 𝑇 → ∞),(13)
which leads to 𝑅 (2)  𝑇 (𝜀) = 𝑂 ( √ 𝑇). This strictly improves the trivial upper bound 𝑅 (2)  𝑇 (𝜀) = 𝑂 ( √︁ 𝑇 𝛾 𝑇 (X)) under 𝑘 = 𝑘 Matérn . The formal descriptions are given in the next lemma.
Lemma 5 (Upper bound of 𝑅 (2)  𝑇 under 𝑘 SE and 𝑘 Matérn ). Suppose (x 𝑡 ) 𝑡 ∈ [𝑇 ] is the input sequence realized by the GP-UCB algorithm. Furthermore, 𝜀 is set as that in Lemma 4. Then, under event A,
𝑅 (2) 𝑇 (𝜀) = 𝑂 ( √ 𝑇) if 𝑘 = 𝑘 Matérn with 2𝜈 + 𝑑 ≤ 𝜈 2 , 𝑂 √︁ 𝑇 ln 2 𝑇 if 𝑘 = 𝑘 SE .(14)
The full proof is given in Appendix A.2. The remaining interest is the upper bound of 𝑅 (1)  𝑇 (𝜀). The definition of 𝑅 (1)  𝑇 (𝜀) is the same as the lenient regret [8], which is known to be smaller than the original regret 𝑅 𝑇 in GP-UCB. Although Cai et al. [8] studies the frequentist setting, their proof strategy is also applicable to the Bayesian setting as described in Section 3.4 in [8]. The following lemma provides the formal statement about the upper bound of 𝑅 (1)  𝑇 (𝜀). Lemma 6 (Upper bound of 𝑅 (1)  𝑇 , adaptation of the proof of Theorem 1 in [8]). Fix any 𝜀 > 0. Suppose 𝑘 = 𝑘 SE or 𝑘 = 𝑘 Matérn . Then, when running GP-UCB, 𝑅 (1)  𝑇 (𝜀) = 𝑂 (1) holds under event A.
We provide the proof in Appendix A.3 for completeness. For both kernels, 𝑅 (1)  𝑇 (𝜀) is dominated by the upper bound of 𝑅 (2)  𝑇 (𝜀). Finally, we obtain the desired results by aggregating the inequalities in Lemmas 5 and 6.
this section cite: ['b45', 'b46', 'b7', 'b7', 'b7', 'b7']

Section: Discussions
Below, we discuss the limitations of our results and outline possible directions for future research.
• Optimality. Based on the Ω( √ 𝑇) lower bound on the expected regret provided by Scarlett [46], we conjecture that our 𝑂 ( √ 𝑇) high-probability regret bound for GP-UCB is nearoptimal. However, it is not straightforward to extend the lower bound for the expected regret in [46] to a high probability result. Specifically, the lower bound in [46] is quantified by a mutual information term (Lemma 4 in [46]); however, to our knowledge, the technique used to handle this term appears to be specific to the expected regret setting. We believe that the rigorous optimality argument for the Bayesian high probability regret is an important direction for future research.
• Smoothness condition. In our result for the Matérn kernel, we require an additional smoothness constraint to obtain a 𝑂 ( √ 𝑇) regret bound 4 To overcome this issue in our proof, we believe that we need stronger regularity conditions on the sample path around the maximizer than those assumed in Lemma 2.
• Extension to the expected regret. Our regret bounds involve regularity constants that depend on the sample path. However, to our knowledge, there is no existing research that rigorously analyzes how these constants depend on the confidence level 𝛿 GP . This makes it difficult to obtain the expected regret guarantees as with the original GP-UCB, whose expected regret bounds are established by properly decreasing the confidence level as a function of 𝑇 (e.g., [40,53]). To overcome this issue, further analysis for Lemma 2, or another idea to quantify the sample path regularities, is required.
• Extension to other algorithms. One limitation of our technique is its restricted applicability to other algorithms. To apply our proof, at least the algorithm should satisfy the following two conditions: (i) on any index subset, the sub-linear cumulative regret is obtained with high probability (Lemma 21), and (ii) the high probability lenient regret bound is provided (Lemma 6). The existing analysis of the other major algorithms in the Bayesian setting (e.g., Thompson sampling [43], information directed sampling [44]) does not provide these properties. Nevertheless, we believe that the high-level ideas in our proof (see Section 3.1) could be beneficial for future refined analyses of other algorithms.
• Instance dependent analysis in the frequentist setting. As described in the footnote in Section 3.1, we believe that our analysis does not improve the worst-case regret upper bound in the frequentist setting. On the other hand, our technique can be applied to the instance-dependent analysis [49] for GP-UCB. We expect that our proof strategy could yield a 𝑂 ( √ 𝑇) instance-dependent regret for GP-UCB by replacing the sample path condition 3 in Lemma 2 with the growth condition (Definition 4 in [49]) of the function. It is an interesting direction for future research.
this section cite: ['b45', 'b45', 'b45', 'b45', 'b39', 'b52', 'b42', 'b43', 'b48', 'b48']

Section: Conclusion
We provide a refined analysis of GP-UCB in the BO problem. For both SE and Matérn kernels, our results improve upon existing regret guarantees and fill the gap between the existing regret of GP-UCB and the current best upper bound in [46]. The core idea of our analysis is to capture the shrinking behavior of the input sequence by relating it to the worst-case upper bound and the sample path regularity conditions. Although our current analysis is limited to GP-UCB in the Bayesian setting, we believe it lays the foundation for several promising future research directions.
this section cite: ['b45']

Section: References
Ref_id:b0 Title: Spherical harmonics and approximations on the unit sphere: an introduction Year: (2012)
Ref_id:b1 Title: Sharp estimates for eigenvalues of integral operators generated by dot product kernels on the sphere Year: (2014)
Ref_id:b2 Title: Breaking the curse of dimensionality with convex neural networks Year: (2017)
Ref_id:b3 Title: No-regret Bayesian optimization with unknown hyperparameters Year: (2019)
Ref_id:b4 Title: Deep equals shallow for relu networks in kernel regimes Year: (2021)
Ref_id:b5 Title: Time-varying Gaussian process bandit optimization Year: (2016)
Ref_id:b6 Title: Convergence rates of efficient global optimization algorithms Year: (2011)
Ref_id:b7 Title: Lenient regret and good-action identification in Gaussian process bandits Year: (2021)
Ref_id:b8 Title: High-dimensional experimental design and kernel bandits Year: ()
Ref_id:b9 Title: Gaussian process uniform error bounds with unknown hyperparameters for safety-critical applications Year: (2022)
Ref_id:b10 Title: On kernelized multi-armed bandits Year: (2017)
Ref_id:b11 Title: Support vector machines Year: (2008)
Ref_id:b12 Title: Regret bounds for deterministic Gaussian process bandits Year: (2012)
Ref_id:b13 Title: Exponential regret bounds for Gaussian process bandits with deterministic observations Year: (2012)
Ref_id:b14 Title: Parallelizing exploration-exploitation tradeoffs in Gaussian process bandit optimization Year: (2014)
Ref_id:b15 Title: Spherical harmonics in p dimensions Year: (2014)
Ref_id:b16 Title: The knowledge-gradient policy for correlated normal beliefs Year: (2009)
Ref_id:b17 Title: A tutorial on Bayesian optimization Year: (2018)
Ref_id:b18 Title: Bayesian optimization Year: (2023)
Ref_id:b19 Title: On the similarity between the Laplace and neural tangent kernels Year: (2020)
Ref_id:b20 Title: Posterior consistency of Gaussian process prior for nonparametric binary regression Year: (2006)
Ref_id:b21 Title: A treatise on Bessel functions and their applications to physics Year: (1895)
Ref_id:b22 Title: Regret bounds for Gaussian process bandit problems Year: (2010)
Ref_id:b23 Title: Entropy search for information-efficient global optimization Year: (2012)
Ref_id:b24 Title: No-regret bandit exploration based on soft tree ensemble model Year: (2024)
Ref_id:b25 Title: Improved regret analysis in Gaussian process bandits: Optimality for noiseless reward, RKHS norm, and non-stationary variance Year: ()
Ref_id:b26 Title: Sequential decision making with feature-linear models Year: (2022)
Ref_id:b27 Title: Bandit optimisation of functions in the Matérn kernel RKHS Year: (2020)
Ref_id:b28 Title: High dimensional Bayesian optimisation and bandits via additive models Year: (2015)
Ref_id:b29 Title: Multi-fidelity Gaussian process bandit optimisation Year: (2019)
Ref_id:b30 Title: Neural contextual bandits without regret Year: (2022)
Ref_id:b31 Title: Bayesian optimization with exponential convergence Year: (2015)
Ref_id:b32 Title: Cube root asymptotics Year: (1990)
Ref_id:b33 Title: Contextual Gaussian process bandit optimization Year: (2011)
Ref_id:b34 Title: Gaussian process bandit optimization with few batches Year: ()
Ref_id:b35 Title: Mercer's theorem, feature maps, and smoothing Year: (2006)
Ref_id:b36 Title: On Bayesian methods for seeking the extremum Year: (1975)
Ref_id:b37 Title: Scattered data interpolation on spheres: error estimates and locally supported basis functions Year: (2002)
Ref_id:b38 Title: Approximation power of RBFs and their associated SBFs: a connection Year: (2007)
Ref_id:b39 Title: A flexible framework for multi-objective Bayesian optimization using random scalarizations Year: (2020)
Ref_id:b40 Title: Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning) Year: (2005)
Ref_id:b41 Title: Practical hilbert space approximate Bayesian Gaussian processes for probabilistic programming Year: (2023)
Ref_id:b42 Title: Learning to optimize via posterior sampling Year: (2014)
Ref_id:b43 Title: Learning to optimize via information-directed sampling Year: (2014)
Ref_id:b44 Title: Random exploration in Bayesian optimization: Order-optimal regret and computational efficiency Year: ()
Ref_id:b45 Title: Tight regret bounds for Bayesian optimization in one dimension Year: (2018)
Ref_id:b46 Title: Lower bounds on regret for noisy Gaussian process bandit optimization Year: (2017)
Ref_id:b47 Title: A spectral analysis of dot-product kernels Year: (2021)
Ref_id:b48 Title: Instance dependent regret analysis of kernelized bandits Year: (2022)
Ref_id:b49 Title: Hilbert space methods for reduced-rank gaussian process regression Year: (2020)
Ref_id:b50 Title: Gaussian process optimization in the bandit setting: No regret and experimental design Year: (2010)
Ref_id:b51 Title: Interpolation of spatial data: some theory for kriging Year: (1999)
Ref_id:b52 Title: Randomized Gaussian process upper confidence bound with tighter Bayesian regret bounds Year: (2023)
Ref_id:b53 Title: Mixture representation of the matérn class with applications in state space approximations and Bayesian quadrature Year: (2018)
Ref_id:b54 Title: Kernelized reinforcement learning with order optimal regret bounds Year: (2023)
Ref_id:b55 Title: Optimal order simple regret for Gaussian process bandits Year: ()
Ref_id:b56 Title: Uniform generalization bounds for overparameterized neural networks Year: (2021)
Ref_id:b57 Title: On information gain and regret bounds in Gaussian process bandits Year: ()
Ref_id:b58 Title: Finite-time analysis of kernelised contextual bandits Year: (2013)
Ref_id:b59 Title: Tight regret bounds for noisy optimization of a Brownian motion Year: (2022)
Ref_id:b60 Title: Improved self-normalized concentration in Hilbert spaces: Sublinear regret for GP-UCB Year: ()
Ref_id:b61 Title: Frequentist coverage and sup-norm convergence rate in gaussian process regression Year: (2017)
Ref_id:b62 Title: Matrix theory: basic results and techniques Year: (2011)
