Title: Precise Asymptotics and Refined Regret of Variance-Aware UCB
Abstract: In this paper, we study the behavior of the Upper Confidence Bound-Variance (UCB-V) algorithm for the Multi-Armed Bandit (MAB) problems, a variant of the canonical Upper Confidence Bound (UCB) algorithm that incorporates variance estimates into its decision-making process. More precisely, we provide an asymptotic characterization of the arm-pulling rates for UCB-V, extending recent results for the canonical UCB in [21] and [23]. In an interesting contrast to the canonical UCB, our analysis reveals that the behavior of UCB-V can exhibit instability, meaning that the arm-pulling rates may not always be asymptotically deterministic. Besides the asymptotic characterization, we also provide non-asymptotic bounds for the arm-pulling rates in the high probability regime, offering insights into the regret analysis. As an application of this high probability result, we establish that UCB-V can achieve a more refined regret bound, previously unknown even for more complicate and advanced variance-aware online decision-making algorithms. A matching regret lower bound is also established, demonstrating the optimality of our result.

Section: Introduction
The Multi-Armed Bandit (MAB) problem is a fundamental framework capturing the explorationexploitation trade-off in sequential decision-making. Over decades, it has been rigorously studied and widely applied across fields like dynamic pricing, clinical trials, and online advertising [32,35,27].
In the classic K-armed MAB problem, a learner is faced with K arms, each associated with an unknown reward distribution P a for a ∈ [K], supported on [0, 1] with mean µ a and variance σ 2 a . At each time step t ∈ [T ], the learner selects an arm a t and observes a reward X t drawn independently from P at . The learner's goal is to maximize the cumulative reward by striking an optimal balance between exploration (sampling less-known arms to improve estimates) and exploitation (selecting arms with high estimated rewards). This objective is commonly framed as a regret minimization problem, where the regret over a time horizon T is defined as Reg(T ) ≡ T t=1 (µ a ⋆ -µ at ) with a ⋆ ≡ arg max a∈[K] µ a the optimal arm with the highest expected reward. * There exists an instance such that the arm pulling rate is unstable.
The minimax-optimal regret for the K-armed bandit problems is known to be Θ( √ KT ) up to logarithmic factors. This rate is achievable by several well-established algorithms, including the Upper Confidence Bound (UCB) [5,4], Thompson Sampling [1,32], and Successive Elimination [13], among others. Beyond the regret minimization, increasing attention has been devoted to analyzing finer properties and large-T behaviors of several typical algorithms, including the regret tail distributions [15,16], diffusion approximations [14,21,2,24], and arm-pulling rates [21,23]. Among these works, [21] and [23] introduced the concept of stability for the canonical UCB algorithm, enabling the precise characterization of arm-pulling behaviors and facilitating statistical inference for adaptively collected data, a task traditionally considered challenging for the general bandit algorithms.
In more structured settings, sharper regret bounds are achievable, particularly when arm variances are small. For instance, if all arms have zero variance (i.e., deterministic rewards), a single pull of each arm is sufficient to identify the optimal one. This observation has motivated active research into developing variance-aware algorithms for bandit problems [5,3,4,20,28,40,41,33]. Among these algorithms, the Upper Confidence Bound-Variance (UCB-V) algorithm [3,4], detailed in Algorithm 1, adapts the classic UCB algorithm by incorporating variance estimates of each arm.
While regret minimization for variance-aware algorithms has been well-studied, their precise armpulling behavior remains less explored. The main challenge here is that incorporating variance information introduces a new quantity, in addition to the reward gaps ∆ a ≡ µ a ⋆ -µ a , a ∈ [K], that influences the arm-pulling rates. In fact, the behavior of the algorithm can differ significantly depending on whether variance information is incorporated or not. In Figures 1 and 2b, we compare the empirical arm-pulling rates between UCB-V and the canonical UCB in a two-armed setting. Compared to the canonical UCB, its variance-aware version exhibits significantly greater fluctuations in arm-pulling numbers as the reward gap changes, and its arm-pulling distribution is more heavytailed. This highlights the significant differences in the variance-aware setting and introduces additional challenges.
Algorithm 1 UCB-V Algorithm 1: Input: Arm number K, time horizon T , and exploration coefficient ρ. 2: Pull each of the K arms once in the first K iterations. 3: for t = K + 1 to T do 4: Compute arm pulls for arm a up to time t by n a,t ≡ s∈[t-1] 1{A s = a}, a ∈ [K].
this section cite: ['b31', 'b34', 'b26', 'b4', 'b3', 'b0', 'b31', 'b12', 'b14', 'b15', 'b13', 'b20', 'b1', 'b23', 'b20', 'b22', 'b20', 'b22', 'b4', 'b2', 'b3', 'b19', 'b27', 'b39', 'b40', 'b32', 'b2', 'b3']

Section: 5:
Compute the empirical means and variances Xa,t ≡ 1 n a,t s∈[t-1] 1{a s = a}X s , σ 2 a,t ≡ 1 n a,t s∈[t-1] 1{a s = a}(X s -Xa,t ) 2 .
this section cite: []

Section: 6:
Compute the optimistic rewards UCB(a, t) ≡ Xa,t + σa,t √ ρ log T ∨ 1 √ na,t ρ log T √ na,t for a ∈ [K].
this section cite: []

Section: 7:
Choose arm A t given by A t = arg max a∈[K] UCB(a, t). 8: end for
this section cite: []

Section: Our Contributions
In this work, we try to close this gap by presenting a precise asymptotic analysis of UCB-V. As in [21] and [23], our main focus is on the arm-pulling numbers and stability. For clarity, we concentrate on the two-armed bandit setting, specifically K = 2, in the main part of our paper, as it effectively captures the core exploration-exploitation trade-off while allowing a precise exposition of results [31,18,22,21]. The extension to the K-armed case is discussed in Appendix B. More precisely, by providing a general concentration result, we establish both precise asymptotic characterizations and high-probability bounds for the arm-pulling numbers of UCB-V. When σ 1 , σ 2 = Ω(1), our results provide a straightforward generalization of those obtained for the canonical UCB. In contrast, when σ 1 ≪ σ 2 , we show that, unlike UCB, the stability result may not hold for UCB-V and that it reveals a phase transition phenomenon in the optimal arm-pulling numbers, as illustrated in Figure 2. Finally, as an application of our sharp characterization of arm-pulling rates, we present a refined regret result for UCB-V that is previously unknown for any other variance-aware decisionmaking algorithms. We summarize our contributions in more detail below (see also Table 1). For notational simplicity, we assume that the optimal arm is a ⋆ = 1 and denote by ∆ = ∆ 2 .
Precise asymptotic behavior for UCB-V. For fixed values of ∆, σ 1 , σ 2 , and T , we introduce a pair of deterministic equations whose unique solution {n ⋆ a,T ≡ n ⋆ a,T (∆, σ 1 , σ 2 , T )} a=1,2 predicts the arm-pulling numbers of UCB-V. We show that for possibly T -dependent (∆, σ 1 , σ 2 ), n a,T n ⋆ a,T p -→ 1 for a = 1, 2, except when σ 1 ≪ σ 2 and σ 2 √ ρ log T /( √ T ∆) = 1 hold simultaneously; see Theorem 1 for a precise statement. This phenomenon, referred to as the stability of arm-pulling numbers, is formally defined in Definition 1. Consequently, the solutions of these deterministic equations can be used to predict the behavior of UCB-V in the large-T regime, as illustrated in Figure 2. Our results reveal several notable differences between UCB-V and the canonical UCB, as analyzed in [21] and [23], due to the incorporation of variance information:
• When σ 1 , σ 2 = Ω(1), the optimal arm will always be pulled a linear number of times, with n 1,T ∼ σ 2 1 σ 2 1 +σ 2 2 T as ∆ → 0. This indicates that, in the small-gap regime, the UCB-V algorithm tends to allocate more pulls to the arm with higher reward variance, generalizing the result for the canonical UCB in which both arms are pulled T /2 times in the small-∆ regime.
• When σ 1 = o(σ 2 ), UCB-V may pull the optimal arm sublinearly in the small-∆ regime. In contrast, the canonical UCB pulls the optimal arm linearly in T , as illustrated in Figure 2b. This behavior is governed by the ratio Λ T ≡ σ 2 √ ρ log T /( √ T ∆). More precisely, given σ 1 ≤ ρ log T /T and σ 2 = Ω(1), 1. When lim T →∞ Λ T < 1, we have n 1,T /n ⋆ 1,T p -→ 1 with n ⋆ 1,T = Ω(T ). 2. When lim T →∞ Λ T > 1, we have n 1,T /n ⋆ 1,T p -→ 1 with n ⋆ 1,T = O( √ T /σ 2 ). 3. When lim T →∞ Λ T = 1, there exists a bandit instance where, for large T , P(n 1,T ≳ T / √ log T ) ∧ P(n 1,T ≲ √ T log T ) ≳ 1.
The above results completely characterize the behavior of n 1,T in the prescribed regime. Notably, we establish a phase transition in the optimal-arm pulling times n 1,T , shifting from O( √ T /σ 2 ) to Ω(T ) at the critical point Λ T = 1. We also note that the existence of an unstable instance when Λ T ∼ 1 highlights a stark contrast between UCB-V and the canonical UCB, where, for the latter, it has been shown that the behavior of n 1,T for any ∆ > 0 can be asymptotically described by a deterministic sequence {n ⋆ 1,T } as T → ∞.
High probability bounds and confidence region for arm pulling numbers. While our asymptotic theory provides the precise limiting behavior of UCB-V, it has a drawback similar to those found in [21] and [23]: the convergence rate in probability is quite slow. Specifically, the probability that the uncontrolled event occurs decays at a rate of O((log T ) -foot_0 ). This slow rate is inadequate for providing insight or guarantees in the popular high probability regime for studying bandit algorithms, where the probability of an uncontrolled event should be in the order of O(T -1 ). To address this gap, we demonstrate that, starting from our unified concentration result in Proposition 1, one can derive non-asymptotic bounds for the arm pulling numbers in the high probability regime. Such result provides a high probability confidence region for arm pulling numbers, as illustrated in Figure 3a in Appendix D.1.
Refined regret for variance-aware decision making. As an application of our high probability arm-pulling bounds, we demonstrate in Theorem 2 that the UCB-V algorithm achieves a refined regret bound of form
O σ 2 2 σ 1 ∨ σ 2 √ T .
This result improves upon the best-known regret 1 for UCB-V [4,36] and surpasses the regrets for all known variance-aware bandit algorithms [41,40,33,9,10], which are of form O(σ 2 √ T ) in the two-armed setting. Our result is the first to reveal the effect of the optimal arm's variance: When σ 1 = o(σ 2 ), the regret matches the previously known O(σ 2 √ T ) bound, and the optimal arm's variance does not affect performance; as σ 1 surpasses σ 2 , the regret decreases as σ 1 increases, following the form O(
σ 2 2 σ1 √ T ).
Simulations presented in Figure 2 confirm our theoretical predictions, demonstrating that UCB-V exhibits improved performance in high-σ 1 scenarios, which was previously unknown. A matching regret lower bound is established in Theorem 3, demonstrating the optimality of our result.
this section cite: ['b20', 'b22', 'b30', 'b17', 'b21', 'b20', 'b20', 'b22', 'b20', 'b22', 'b3', 'b35', 'b40', 'b39', 'b32', 'b8', 'b9']

Section: Notation.
For any positive integer n, denote by [n] = [1 : n] set {1, . . . , n}. For a, b ∈ R, a ∨ b ≡ max{a, b} and a ∧ b ≡ min{a, b}. For any a ∈ R, a + ≡ max{a, 0}. Throughout this paper, we regard T as our fundamental large parameter. 2 Implicit bounds for arm-pulling numbers Additional notations. For ρ > 1 and T ∈ N + , let
σ a (ρ; T ) ≡ σ a √ ρ log T , ∆ a (ρ; T ) ≡ ∆ a ρ log T , a ∈ [K].
Whenever there is no confusion, we write σ a ≡ σ a (ρ; T ) and ∆ a ≡ ∆ a (ρ; T ). We also allow σ a and ∆ a to depend on T . For any σ ≥ 0 and n ∈ N + , define φ(n; σ) ≡ σ ∨ n -1/2 / √ n. For each fixed σ ≥ 0, the map φ(•; σ) : R ≥0 → R ≥0 is monotone nonincreasing in n, and its (piecewise) inverse n(•; σ) : R ≥0 → R ≥0 is n(φ; σ) ≡ (σ 2 ∨ φ)/φ 2 . Consequently, studying n a,T reduces to analyzing
φ a,t ≡ φ(n a,t ; σ a ) = σ a ∨ n -1/2 a,t √ n a,t , a ∈ [K], t ∈ [T ].
Here the two pieces in φ match the variance-driven term (σ/ √ n) and the boundedness-driven term (1/n) in Bernstein's inequality; the rescalings σ a and ∆ a place variance and gap on the same Bernstein-type exploration scale used by our UCB-V bonus.
this section cite: []

Section: O(
a̸ =a ⋆ σ 2 a T log T ) regret for UCB-V can be derived based on results in [4]. The median and 30% quantile of n 1,T (optimal arm-pulling count) for UCB and UCB-V, under varying Λ T in the σ 1 = o(σ 2 ) regime, with T = 1, 000, 000 over 30 repetitions for each ∆ 2 . The red dotted line is the predicted n 1,T of UCB-V using (6). A more detailed numerical setting is provided in Appendix H.
this section cite: ['b3']

Section: Implicit bounds for arm-pulling numbers
Consider Algorithm 1 with K = 2. We start with the following concentration result for φ a,T .
this section cite: []

Section: Proposition 1.
Recall that σ a ≡ σ a (ρ; T ) and ∆ a ≡ ∆ a (ρ; T ) for a ∈ [2]. Fix any δ > 0, ρ > 1, and any positive integer T ≥ 4 such that ε ≡ ε(δ; ρ; T ) ≡ 5 48 log(log(T /δ)/δ) ρ log T
+ 128 log(1/δ) ρ log T 1/2 + 200 √ T ≤ 1 2 .
Then, with probability at least 1 -δ, we have
1 -ε 1 + ε 4 ≤ σ 2 1 ∨ φ 1,T T φ 2 1,T + σ 2 2 ∨ (φ 1,T + ∆ 2 ) T (φ 1,T + ∆ 2 ) 2 ≤ 1 + ε 1 -ε 4 and 1-ε 1+ε 4 ≤ φ 2,T φ 1,T +∆2 ≤ 1+ε 1-ε 4 .
The proof of Proposition 1 above is inspired by the delicate analysis of bonus terms for the canonical UCB [23] in the large-T regime and is presented in Appendix C.1. Proposition 1 establishes a sandwich inequality, demonstrating that as term ε(δ; ρ; T ) approaches 0, the concerned ratio converges to 1. This can be viewed as a non-asymptotic and variance-aware extension of the results found in [21] and [23] for the canonical UCB. To incorporate variance information, we have also developed a Bernstein-type non-asymptotic law of iterated logarithm result in Lemma 8 in Appendix G, which is of independent interest. We now make several comments regarding Proposition 1:
Deriving the asymptotic equation. When selecting δ = (log T ) -1 , we find that for any ρ > 0, the ε(δ; ρ; T ) term approaches 0 at a rate of O log log T log T
. Under this selection, Proposition 1 yields the following probability convergence guarantee
σ 2 1 ∨ φ 1,T T φ 2 1,T + σ 2 2 ∨ (φ 1,T + ∆ 2 ) T (φ 1,T + ∆ 2 ) 2 = 1 + o P (1).(1)
It is noteworthy that in this asymptotic result, the o P (1) term converges to 0 at a relatively slow rate, both in probability and magnitude. For each T , it has a probability of at least 1 -O 1 log T of being bounded by O log log T log T . This slow rate appears not only in our result but also in [21] and [23], necessitating a considerably large T to observe theoretical predictions clearly in experiments, as illustrated in Figure 2b.
Permissible values of ρ for high probability bounds. One popular scale selection of δ is δ ≍ 1/T 2 , where Proposition 1 becomes a high-probability guarantee widely adopted in the pure-exploration and regret analysis literature [5,3]. With such selection, the requirement ε(δ; ρ; T ) < 1/2 translates to ρ > c ′ for some universal and sufficiently large constant c ′ . Thus, we have that with probability at least 1 -O(1/T 2 ),
1 -(c ′ ) -1 ≤ σ 2 1 ∨ φ 1,T T φ 2 1,T + σ 2 2 ∨ (φ 1,T + ∆ 2 ) T (φ 1,T + ∆ 2 ) 2 ≤ 1 + (c ′ ) -1 .(2)
In particular, since the centered term is a monotone increasing function of φ 1,T , this result provides a high-probability confidence region for φ 1,T and consequently for n 1,T , as shown in Figure 3a in Appendix D.1. These high probability bounds of n 1,T enable us to derive the refined regret bound for UCB-V in Section 4.
The implications outlined above will be made precise in the subsequent two sections.
3 Asymptotic characterization of arm-pulling numbers
this section cite: ['b1', 'b22', 'b20', 'b22', 'b20', 'b22', 'b4', 'b2']

Section: Stability of the asymptotic equation
Consider the deterministic equation
f (φ) = 1 with f (φ) ≡ σ 2 1 ∨ φ T φ 2 + σ 2 2 ∨ (φ + ∆ 2 ) T (φ + ∆ 2 ) 2 .(3)
Recall that σ a ≡ σ a / √ ρ log T and ∆ a ≡ ∆ a /(ρ log T ) for a = 1, 2, so the equation actually depends on both ρ and T . For any fixed T ∈ N + and ρ > 1, Proposition 1 indicates that φ 1,T satisfies f (φ 1,T ) = 1 + ζ, where ζ represents a perturbation term. Given the asymptotic equation derived in (1), it is reasonable to conjecture that the behavior of φ 1,T aligns with the solution of the above deterministic equation. To formally connect φ 1,T to the solution of (3), we conduct a perturbation analysis of this equation. Specifically, we have the following result. Lemma 1. Fix any T ∈ N + and ρ > 1. It holds that:
1. The fixed-point equation (3) admits a unique solution φ ⋆ ∈ (1/T, 1).
this section cite: []

Section: 2.
Assume that there exist some ζ ∈ (-1/2, 1/2) and φ ζ such that f (φ ζ ) = 1 + ζ. Then
φ ζ φ ⋆ -1 ≤ 27|ζ| • 1 + σ 2 2 T ∆ 2 2 -1 -1 ∧ σ 2 σ 1 ∨ T -1/2 . (4
)
The proof of Lemma 1 above is provided in Appendix D.1. Intuitively, Lemma 1 asserts that in either the homogeneous variance case (σ 2 ≍ σ 1 ) or when the ratio σ 2 2 /(T ∆ 2 2 ) is bounded away from 1, the solution φ ζ of the ζ-perturbed equation is provably bounded by 1 ± O(ζ) φ ⋆ . However, when σ 1 = o(σ 2 ) and the ratio σ 2 2 /(T ∆ 2 2 ) approaches 1, the stability guarantee of the lemma breaks down. Although this result may seem pessimistic-as σ 2  2 /(T ∆ 2 2 ) ∼ 1 causes the right-hand side of (4) to diverge-it accurately reflects the behavior of the perturbed solution. This is illustrated in Figure 3b in Appendix D.1, where instability arises when σ 2  2 /(T ∆ 2 2 ) = 1.
this section cite: []

Section: Asymptotic stability
One important consequence of the perturbation bound is the following asymptotic stability result. Theorem 1. Consider Algorithm 1 with K = 2. Assume that
lim T →∞ σ 2 2 T ∆ 2 2 -1 -1 ∧ σ 2 σ 1 ∨ T -1/2 < ∞.(5)
Then for a ∈ [2], we have n a,T /n ⋆ a,T p -→ 1, where n ⋆ 1,T is the unique solution to equation
σ 2 1 ∨ φ(n ⋆ 1,T ; σ 1 ) T φ 2 (n ⋆ 1,T ; σ 1 ) + σ 2 2 ∨ (φ(n ⋆ 1,T ; σ 1 ) + ∆ 2 ) T (φ(n ⋆ 1,T ; σ 1 ) + ∆ 2 ) 2 = 1(6)
and n ⋆ 2,T ≡ σ 2 2 ∨ (φ(n ⋆ 1,T ; σ 1 ) + ∆ 2 )/(φ(n ⋆ 1,T ; σ 1 ) + ∆ 2 ) 2 .
The proof of Theorem 1 above is provided in Appendix D.2, relying on Proposition 1 and Lemma 1.
Lemma 1 shows that the boundedness condition in (5) is essential for ensuring the stability of the arm-pulling process, as defined in Definition 1. Specifically, when σ 1 = σ 2 = 1, condition (5) is always satisfied, and the asymptotic equation ( 6) reduces to the canonical UCB setting studied in [21] and [23], where stability is guaranteed. The key insight behind such stability in an even more general homogeneous setting σ 1 ≍ σ 2 is that the optimal arm's pulling number grows linearly with T , as noted in [23,Eqn. (22)]. In the inhomogeneous case where σ 1 = o(σ 2 ), the boundedness of |σ 2 2 /(T ∆ 2 2 ) -1| -1 becomes crucial for stability and appears to be novel, ensuring stability even when n 1,T grows sub-linearly in T . The instability result in the next subsection complements Theorem 1 by presenting a counterexample where the boundedness condition in (5) fails. An extension to the K-armed setting is discussed in Appendix B.
Asymptotic behavior of arm-pulling numbers. Let us write φ ⋆ ≡ φ(n ⋆ 1,T ; σ 1 ) for simplicity. From Theorem 1, the asymptotic behavior of n 1,T can be derived through n ⋆ 1,T , which, in turn, can be fully determined by φ ⋆ , provided that the boundedness condition in (5) holds. Thus, understanding the analytical properties of φ ⋆ is sufficient to determine the asymptotic behavior of n 1,T .
The function f (φ ⋆ ) is known to be monotonic increasing in φ ⋆ , and the solution to f (φ ⋆ ) = 1 lies within the interval [0, 1]. This allows us to efficiently compute the numerical behavior of φ ⋆ to predict both asymptotic and finite-time arm-pulling behavior, as illustrated in Figure 3a in Appendix D.1. However, due to the presence of a maximum operation and the complexity of the underlying equation, obtaining a closed-form solution for φ ⋆ remains difficult. Below, we present several extreme cases that highlight new phenomena arising from the incorporation of variance information, which differs from the classical UCB algorithm. A full analytical characterization of φ ⋆ is left for future research. Example 1. When σ 1 , σ 2 = Ω(1), Eqn. (6) simplifies to
n ⋆ 1,T T + σ 1 σ 2 • T n ⋆ 1,T + ∆ 2 σ 2 T ρ log T -2 = 1.
In the moderate gap regime ∆ 2 ∼ σ 2 θ log T /T for some fixed θ ∈ R ≥0 , we have n ⋆ 1,T /T = λ ⋆ (θ) 1 + o(1) , with λ ⋆ (θ) being the unique solution to equation λ
+ σ1 σ2 • 1 √ λ + θ/ρ -2 = 1.
Moreover, we can compute the limits lim θ→+∞ λ ⋆ (θ) = 1 and lim θ→0 λ ⋆ (θ) =
σ 2 1 σ 2 1 +σ 2 2 .
This result recovers the asymptotic equation for the canonical UCB in [21] and [23] when σ 1 = σ 2 = 1. The λ ⋆ equation derived here can be viewed as an extension of those in [21]. Notably, in the small-gap limit θ → 0, the arm-pulling allocation for UCB-V becomes proportional to the variance instead of being equally divided.
this section cite: ['b20', 'b22', 'b22', 'b4', 'b20', 'b22', 'b20']

Section: Example 2.
When σ 1 ≤ ρ log T /T , σ 2 = Ω(1), Eqn. (6) simplifies to
n ⋆ 1,T T + 1 σ 2 ρ log T T • T n ⋆ 1,T + ∆ 2 σ 2 T ρ log T -2 = 1.
In the moderate gap regime where ∆ ∼ σ 2 θ log T /T for some fixed θ ∈ R ≥0 \ {ρ}, we have
n ⋆ 1,T /T = λ ⋆ (θ) 1 + o(1
) , with λ ⋆ (θ) being the unique solution to equation λ + 1 σ2 ρ log T T • 1 λ + θ/ρ -2 = 1. Moreover, we can compute the limits lim θ→+∞ λ ⋆ (θ) = 1 and lim θ→0 λ ⋆ (θ) = 2 √ 1+4σ 2 2 T +1
.
This limit indicates a distinct behavior between UCB-V and UCB: as θ → 0, the number of pulls for the optimal arm becomes sub-linear in T , specifically O( √ T /σ 2 ), while for UCB, it approaches T /2. Additionally, we have a more detailed description of the transition between sub-linear and linear pulling times: (i) When θ < ρ, λ ⋆ (θ) ≤ 1/(σ 2 (1 -θ/ρ) √ T ) and (ii) When θ > ρ, λ ⋆ (θ) ≥ 1 -ρ/θ. This result indicates that the transition from O( √ T ) pulling time to Ω(T ) pulling time for the optimal arm occurs at θ = ρ.
Inference with UCB-V. Another key implication of Theorem 1 is its relevance to the post-policy inference in the UCB-V algorithm. To begin, we recall the notion of arm stability as defined in [23, Definition 2.1].
this section cite: []

Section: Definition 1.
An arm a ∈ [K] is stable if there exists a deterministic sequence n ⋆ a,T such that n a,T n ⋆ a,T p -→ 1 with n ⋆ a,T → ∞,
where n ⋆ a,T may depend on T, {µ a } a∈[K] , and {σ 2 a } a∈[K] .
This notion of stability guarantees that, as T → ∞, the number of times that an arm is pulled becomes predictable, thus enabling valid statistical inference for each arm's reward distribution. Specifically, under the stability condition, the following Z-statistic converges in distribution to a standard normal √ n a,T σ a,T Xa,T -µ a ⇒ N (0, 1).
This result is crucial for constructing valid hypothesis tests and confidence intervals in the adaptive sampling settings [29,39]. For space efficiency, we provide a more detailed discussion along with simulation results in Appendix I.
this section cite: ['b28', 'b38']

Section: Unstable results: closer look and hard instances
Recall that Theorem 1 establishes the stability result, except for the case when σ 1 = o(σ 2 ) and σ 2 2 /(T ∆ 2 2 ) = 1 hold simultaneously. In this section, we provide a hard instance in this setting and show the instability result. More precisely, let us consider the setting similar to Example 2, with σ 2 = 1, σ 1 = 0, and gap ∆ 2 = σ 2 θ log T /T .
In this scenario, the behavior of the solution to the asymptotic equation is described by
n ⋆ 1,T ∼ λ ⋆ (θ)T , where λ ⋆ (θ) solves λ + ρ log T /T /λ + θ/ρ -2 = 1 or equivalently, 1 √ 1 -λ - 1 λ ρ log T T = θ/ρ.(9)
To heuristically explain why θ = ρ (or equivalently σ 2 2 /(T ∆ 2 2 ) = 1) acts as a phase transition point and leads to instability, we formally take the first-order expansion 1/ √ 1 -λ ≈ 1 + λ/2 in (9) and multiply both sides by λ. This yields a quadratic equation in λ,
1 2 λ 2 + (1 -θ/ρ)λ -ρ log T T = 0.
Notably, the order of the solution in T depends crucially on the sign of 1 -θ/ρ. For any ε > 0, by observing the analytical solution to the quadratic equation, we have
λ = O(ε -1 ρ log T T ) when 1 -θ/ρ ≥ ε, Ω(ε) when 1 -θ/ρ ≤ -ε.
In particular, a perturbation in θ around θ = ρ of magnitude ε with different signs can lead to a fluctuation in n ⋆ 1,T from O(ε -1 √ T log T ) to Ω(εT ).
Based on the above intuition and informal argument, we can rigorously demonstrate the instability result by constructing a Bernoulli bandit instance and establishing a time-uniform anti-concentration result for the Bernoulli reward process via Donsker's principle. Intuitively, our anti-concentration result shows that, with constant probability, θ can be perturbed by a magnitude of O((log T ) -1 ) with different signs, which then implies instability in n 1,T even when we play UCB-V with the oracle information of the variance (i.e., with σ a,t = σ a ). We summarize the instability result in the following proposition and provide its proof in Appendix D.3. Proposition 2. Consider the following two-armed Bernoulli bandit instance: for µ = 1/2 and ∆ 2 = µ(1 -µ)ρ log T /T , arm 1 has reward µ + ∆ and variance 0, and arm 2 has reward µ and variance µ(1 -µ). Assume that σ a 's are known and let n a,T be computed by Algorithm 1 with σ a,t = σ a . Then for sufficiently large T , there exist some constants c 0 , c 1 > 0 such that
P(n 1,T ≤ c 1 √ T log T ) ∧ P(n 1,T ≥ c -1 1 T / log 1/2 T ) > c 0 . Remark 1.
In the example above we set σ 1 = 0 for simplicity. The instability analysis in Proposition 2 also extends to cases where the optimal arm has variance σ 1 = O(T -1/2 ), since in this regime the UCB-V bonus for the optimal arm matches the σ 1 = 0 case. By contrast, constructing an instance with σ 1 = T -α for some α ∈ (0, 1/2) is more delicate and appears to require new ideas: the associated asymptotic equation becomes substantially more involved, and a complete analysis is currently unclear.
The existence of an unstable instance at Λ T = 1 complements our previous finding, where UCB-V's stability was shown for Λ T ̸ = 1. This result highlights a contrast between the UCB-V and canonical UCB: as shown in [21] and [23], the canonical UCB is stable for all ∆, which corresponds to the case in our setting when σ 1 = σ 2 = 1. Our result demonstrates that, in a heterogeneous variance environment, UCB-V may exhibit significant fluctuations. Another implication of this instability is that, unlike the canonical UCB, the CLT for the Z-statistic may not hold for data collected by UCB-V, as shown in Figure 4b in Appendix I. This necessitates the developments of new statistical inference methods for UCB-V collected data or new variance-aware decision-making algorithms with stronger stability guarantees than UCB-V, which we leave as a valuable future direction.
this section cite: ['b20', 'b22']

Section: Refined regret for variance-aware decision making
In this section, we show that a refined regret can be achieved by UCB-V based on our arm-pulling number bounds presented in Section 2. Previously, the best-known regret for UCB-V, shown in [4], is given by 2
Reg(T ) ≤ max ∆2≥0 C log T σ 2 2 ∆ 2 + 2 ∧ ∆ 2 T ≤ C ′ σ 2 T log T .(11)
As mentioned earlier, the regret in (11) does not account for the effect of σ 1 , which contradicts the empirical performance of UCB-V and is conservative in the large σ 1 , small σ 2 regime, as shown in Figure 2a. On the other hand, the derived asymptotic equation for the arm-pulling times naturally indicates a dependency of n 2,T on σ 1 , opening the possibility for refined worst-case regret bounds. More precisely, by adopting the high-probability bound in (2), we can show the following proposition.
Proposition 3. There exists some universal constant C 0 such that with ρ ≥ C 0 and T ≥ 3,
P n 2,T ≤ 16 ∆ 2 + φ(n 1,T ; σ 1 ) ∨ 16σ 2 ∆ 2 + φ(n 1,T ; σ 1 ) 2 ≥ 1 - 6 T 2 .
By applying the upper bound on the number of pulls for sub-optimal arms, we are able to derive the refined worst-case regret for UCB-V.
Theorem 2. There exists some universal constant C 0 > 0 such that for ρ ≥ C 0 and T ≥ 3, we have that with probability at least 1 -6/T 2 ,
Reg(T ) ≤ 16 σ 2 ∧ 16σ 2 2 σ 1 ρT log T + ρ log T .
The proof of the above results can be found in Appendix E. The regret bound in Theorem 2 above improves upon the bound in (11) for the large σ 1 , small σ 2 regime, while recovering (11) in the small σ 1 regime. We note that sharper bounds may be derived in special instances based on Proposition 3. For example, in the Bernoulli setting where µ 1 is close to 1, the variance inequality σ 2 2 ≲ σ 2 1 + ∆ 2 holds. In this case, Proposition 3 implies a regret bound of the form O(σ 1 √ T log T ), which matches the result established in [30].
Regret results for the K-armed setting. Since most works consider the general K-armed setting, for ease of comparison we now presenting the K-armed extension of Theorem 2, we leave the 2 Rather than presenting an equation of the form (11), [4] established the following gap-dependent upper bound on the number of times a suboptimal arm is pulled:
E(n2,T ) ≤ C log T σ 2 2 ∆ 2 2 + 2 ∆2 .(10)
However, it is straightforward to show (11) by combining (10) with the trivial bound Reg(T ) ≤ ∆2E(n2,T ).
rigorous statements in Theorem 5 in Appendix B. For a K-armed MAB problem with variances {σ 2 a } K a=1 (assuming again W.L.O.G. arm 1 is optimal), we have:
Reg(T ) = O a∈[2:K] σ 2 a ∧ a∈[2:K] σ 2 a σ 1 • T log T ,
The best known worst-case regret of UCB-V was previously known as O( √ KT log T ), as discussed in [28]. In comparison, our result shows that UCB-V can achieve regret adaptive to both the variances of sub-optimal and optimal arms. Beyond UCB-V, for Thompson sampling algorithms, the O( a∈[2:K] σ 2 a T log T ) regret was proved in a Bayesian setting, where each arm's reward distribution follows a known prior distribution. Especially, even when working with the Bayesian regret, the dependency on σ 1 is previously unrevealed. We believe that our results for UCB-V can be extended to these posterior sampling algorithms.
Optimality of Theorem 2. Now we establish a matching lower bound to show the optimality of Theorem 2. To describe the lower bound result, consider any given distributions P 1 and P 2 . Let v = (P 1 , P 2 ) denote a 2-armed bandit instance, where P i represents the distribution of the i-th arm. To emphasize the dependency on the means
µ i ≡ E X∼Pi [X] and variances σ 2 i ≡ Var X∼Pi [X]
of the arms, we redundantly express this instance as v = (P 1 (µ 1 , σ 1 ), P 2 (µ 2 , σ 2 )).
Given any instance v, we use the notation σ 2 opt (v), σ 2 sub (v) to denote variances of optimal and suboptimal arms under v for clarity. For any policy π and any instance v, we denote E πv and P πv as the expectation and probability with respect to the reward distribution induced by π under v. More precisely, we establish the following regret lower bounds. The proof is deferred to Appendix E.3. Theorem 3. Given any sufficiently large T > 0 and 0 < σ 2 < 1/3, consider the following class of problems with σ-bounded variances:
V σ ≡ {(P 1 (µ 1 , σ 1 ), P 2 (µ 2 , σ 2 )) : σ 1 ∨ σ 2 ≤ σ, supp(P i (µ i , σ i )) ⊂ [0, 1], i ∈ {1, 2}}.
Then for every policy π, there exists some v ∈ V such that,
E πv Reg(T ) = Ω(σ sub (v) √ T ).(12)
Moreover, the following trade-off lower bound holds: Given any 0 < β < 1/2 and c T = O(poly(log T )), consider
V ′ β ≡ {(P 1 (µ 1 , σ 1 ), P 2 (µ 2 , σ 2 )) ∈ V σ : µ 1 > µ 2 , σ 1 = T β σ 2 2 , σ 1 ≥ T -β /256}
and the good policy class
Π good ≡ π : max v∈V E πv Reg(T ) σ sub (v) ≤ c T √ T .
Then for any π ∈ Π good , there exists some v ′ ∈ V ′ β such that
E πv ′ Reg(T ) = Ω σ 2 sub (v ′ ) σ opt (v ′ ) T c T .(13)
Theorem 3 states that, in the general scenario with σ-bounded variances, no algorithm can achieve a regret better than σ sub (v) √ T . This reveals the optimality of Theorem 2 in the regime where σ opt (v) ≤ σ sub (v). Furthermore, we show that in σ opt > σ sub regime, any reasonably good algorithm that performs nearly optimal in worst-case over V (i.e., the algorithms lie within Π good ) cannot achieve a regret better than
σ 2 sub σopt √
T . In particular, the regret upper bound in Theorem 2 demonstrates that UCB-V matches such trade-off lower bound in the regime where σ opt > σ sub , illustrating its optimality.
this section cite: ['b3', 'b29', 'b3', 'b9', 'b27']

Section: Conclusion.
In this paper, we provide a refined analysis of the UCB-V algorithm, including a precise characterization of its asymptotic arm-pulling behavior and high-probability, non-asymptotic bounds that lead to a sharper and optimal regret upper bound. Several valuable future directions remain open. First, the instability result is established only under the regime σ 1 = o( log T /T ) and σ 2 = Ω(1), while the stability condition in the more general case σ 1 = o(σ 2 ) is not yet sharply understood. Second, as discussed in Section B, our stability condition in the K-armed setting requires a uniform-type separation condition min a ∆a ≥ σa (K -1)/T , which we believe can be further refined.
this section cite: []

Section: References
Ref_id:b0 Title: Analysis of thompson sampling for the multi-armed bandit problem Year: (2012)
Ref_id:b1 Title: Diffusion approximations for a class of sequential experimentation problems Year: (2022)
Ref_id:b2 Title: Tuning bandit algorithms in stochastic environments Year: (2007)
Ref_id:b3 Title: Exploration-exploitation tradeoff using variance estimates in multi-armed bandits Year: (2009)
Ref_id:b4 Title: Using confidence bounds for exploitation-exploration trade-offs Year: (2002-11)
Ref_id:b5 Title: Convergence of Probability Measures Year: (2013)
Ref_id:b6 Title: Statistical inference for online decision making: In a contextual bandit setting Year: (2021)
Ref_id:b7 Title: Online statistical inference for contextual bandits via stochastic gradient descent Year: (2022)
Ref_id:b8 Title: Variance-aware sparse linear bandits Year: (2022)
Ref_id:b9 Title: Farzad Farnoud, and Quanquan Gu. Variance-aware regret bounds for stochastic contextual dueling bandits Year: (2023)
Ref_id:b10 Title: Online multi-armed bandits with adaptive inference Year: (1939)
Ref_id:b11 Title: Online policy learning and inference by matrix completion Year: (2024)
Ref_id:b12 Title: Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems Year: (2006)
Ref_id:b13 Title: Diffusion approximations for thompson sampling Year: (2021)
Ref_id:b14 Title: The fragility of optimized bandit algorithms Year: (2021)
Ref_id:b15 Title: The typical behavior of bandit algorithms Year: (2022)
Ref_id:b16 Title: Improved optimistic algorithms for logistic bandits Year: (2020)
Ref_id:b17 Title: A linear response bandit problem Year: (2013)
Ref_id:b18 Title: Confidence intervals for policy evaluation in adaptive experiments Year: (2021)
Ref_id:b19 Title: Optimality of thompson sampling for gaussian bandits depends on priors Year: (2014)
Ref_id:b20 Title: A closer look at the worst-case behavior of multi-armed bandit algorithms Year: (2021)
Ref_id:b21 Title: On the complexity of best-arm identification in multi-armed bandit models Year: (2016)
Ref_id:b22 Title: Inference with the upper confidence bound algorithm Year: (2024)
Ref_id:b23 Title: Weak signal asymptotics for sequentially randomized experiments Year: (2023)
Ref_id:b24 Title: Asymptotically efficient adaptive allocation rules Year: (1985)
Ref_id:b25 Title: Refining the confidence level for optimistic bandit strategies Year: (2018)
Ref_id:b26 Title: Bandit Algorithms Year: (2020)
Ref_id:b27 Title: Efficientucbv: An almost optimal algorithm using variance estimates Year: (2018)
Ref_id:b28 Title: Why adaptively collected data have negative bias and how to correct for it Year: (2018)
Ref_id:b29 Title: Kullback-leibler maillard sampling for multiarmed bandits with bounded rewards Year: (2023)
Ref_id:b30 Title: Nonparametric bandits with covariates Year: (2010)
Ref_id:b31 Title: A tutorial on thompson sampling Year: (2018)
Ref_id:b32 Title: Only pay for what is uncertain: Variance-adaptive thompson sampling Year: (2024)
Ref_id:b33 Title: Regret distribution in stochastic bandits: Optimal trade-off between expectation and tail risk Year: (2023)
Ref_id:b34 Title: Introduction to multi-armed bandits Year: (2019)
Ref_id:b35 Title: Variance-aware regret bounds for undiscounted reinforcement learning in mdps Year: (2018)
Ref_id:b36 Title: Noise-adaptive thompson sampling for linear contextual bandits Year: (2024)
Ref_id:b37 Title: Off-policy evaluation via adaptive weighting with data from contextual bandits Year: (2021)
Ref_id:b38 Title: Inference for batched bandits Year: (2020)
Ref_id:b39 Title: Improved variance-aware confidence sets for linear bandits and linear mixture mdp Year: (2021)
Ref_id:b40 Title: Variance-dependent regret bounds for linear bandits and reinforcement learning: Adaptivity and computational efficiency Year: (2023)
