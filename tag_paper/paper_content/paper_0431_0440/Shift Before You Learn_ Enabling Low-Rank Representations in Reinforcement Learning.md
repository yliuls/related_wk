Title: Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning
Abstract: Low-rank structure is a common implicit assumption in many modern reinforcement learning (RL) algorithms. For instance, reward-free and goal-conditioned RL methods often presume that the successor measure admits a low-rank representation. In this work, we challenge this assumption by first remarking that the successor measure itself is not approximately low-rank. Instead, we demonstrate that a low-rank structure naturally emerges in the shifted successor measure, which captures the system dynamics after bypassing a few initial transitions. We provide finite-sample performance guarantees for the entry-wise estimation of a low-rank approximation of the shifted successor measure from sampled entries. Our analysis reveals that both the approximation and estimation errors are primarily governed by a newly introduced quantitity: the spectral recoverability of the corresponding matrix. To bound this parameter, we derive a new class of functional inequalities for Markov chains that we call Type II Poincaré inequalities and from which we can quantify the amount of shift needed for effective low-rank approximation and estimation. This analysis shows in particular that the required shift depends on decay of the high-order singular values of the shifted successor measure and is hence typically small in practice. Additionally, we establish a connection between the necessary shift and the local mixing properties of the underlying dynamical system, which provides a natural way of selecting the shift. Finally, we validate our theoretical findings with experiments, and demonstrate that shifting the successor measure indeed leads to improved performance in goal-conditioned RL.

Section: Introduction
In reinforcement learning (RL), the complexity of environment dynamics requires structural assumptions to achieve statistical efficiency. A widely adopted approach assumes that key quantities admit low-dimensional feature representations, effectively imposing low-rank structure on matrices underlying various RL components such as the Q-function (57; 55; 60), transition kernel (2; 33; 59), graph Laplacian (46; 47; 64), and successor representation (15; 58; 5). Some works even aim to learn universal low-dimensional representations transferable across tasks, as in Forward-Backward models (62; 63) and goal-conditioned RL (3; 21). Despite their empirical success and emerging theoretical analyses, fundamental questions remain:  To address these questions, we examine how the long-term dynamics of an MDP naturally give rise to global structure that can be captured effectively through low-rank approximations. In particular, we demonstrate that a simple temporal shift of the successor measure can substantially improve its alignment with low-rank structure. This shift reweights transitions to emphasize long-term behavior, filtering out short-term noise and amplifying the structural signal present in the dynamics. Crucially, its effectiveness hinges on the mixing properties of the underlying Markov chain, which determine how rapidly the process forgets its initial conditions and reveals coherent global patterns. Our main contributions are:
Why
(a) We introduce the notion of spectral recoverability (Definition 3) to quantify the approximation error incurred by low-rank representations. We show that standard successor measures lack spectral recoverability (Proposition 1), motivating the use of shifted successor measures which discard initial transitions and emphasize long-term dynamics. We prove that sufficiently large shifts guarantee spectral recoverability (Section 5).
(b) We provide finite-sample performance guarantees for the entry-wise estimation of a low-rank approximation of the shifted successor measure from sampled entries (Thm. 1). Our analysis reveals that the estimation error is also governed by the spectral recoverability of the shifted successor measure.
(c) To characterize when spectral recoverability holds, we introduce a novel class of functional inequalities for Markov chains, which we call Type II Poincaré inequalities (Thm. 2). These inequalities allow us to quantify the amount of shift required for effective low-rank approximation and estimation. Moreover, we relate the required shift to the local mixing properties of the underlying dynamical system. These properties measure the extent to which the state space admits a decomposition into subsets within which the local dynamics mix rapidly.
(d) Finally, we validate our theoretical insights through experiments on learning the shifted universal successor measure in goal-conditioned RL. This representation enables the simultaneous learning of optimal policies for reaching a variety of goals. A representative result is shown in Figure 1.
this section cite: []

Section: Related Work
Low-rank approximations in RL. Low-rank models are ubiquitous in reinforcement learning. These models rely on low-rank approximations of certain matrices: most notably the Laplacian (46; 47; 42; 64; 35; 25) and the successor representation (15; 58; 36; 43; 41; 62; 63), the latter often considered a better candidate for low-rank modeling (63). While these models are empirically effective and supported by intuitive heuristics based on spectral properties (see e.g. ( 38)), they often lack rigorous theoretical justification. Our work aims to address this gap by establishing a connection between low-rank structures and the mixing behavior of the underlying dynamics.
this section cite: ['b62']

Section: Sample complexity bounds.
Numerous studies have established performance guarantees for estimating low-rank structures in reinforcement learning (RL). Several approaches draw inspiration from matrix completion techniques and have been applied, for example, to the estimation of the Q-function (57; 55; 65; 60). Our work is closer to the low-rank/linear Markov Decision Process (MDP) framework explored in (66; 2; 68; 69; 59), where the transition kernel is modeled as a bilinear factorization of the form P (s, a, s ′ ) = ψ(s, a) ⊺ ϕ(s ′ ). A special case arises when the factors are constrained to be non-negative, yielding models such as (soft) state aggregation and block MDPs (19; 56; 68; 28). To the best of our knowledge, we are the first to analyze the sample complexity of estimating successor measures. Importantly, since successor representations are typically full-rank, imposing a strict low-rank assumption would be inappropriate. Alternative notions of rank have been proposed in the function approximation setting (30; 61; 18; 32); however these depend not only on the dynamics but also on the choice of the function class. In contrast, our analysis does not rely on function approximation or any structural assumptions, and allows intrinsic structure to emerge naturally from the mixing properties of the underlying dynamics.
Mixing phenomena. To bridge matrix estimation and dynamical behavior, we introduce spectral recoverability, a parameter that quantifies both the SVD truncation error and the difficulty of recovering matrix entries from partial observations. Our approach is inspired by (11), who established minimax bounds for matrix completion under a bounded nuclear norm. In contrast, we focus on entrywise estimation, which requires consideration of singular vectors. Spectral recoverability thus blends classical notions of coherence and nuclear norm, enabling entrywise error analysis via the leave-one-out technique of (1). On the other hand, it connects to classical mixing measures in Markov chain theory and can thus be bounded by revisiting classical tools such as functional inequalities (17) and spectral analysis (22). However, unlike traditional approaches that focus on global mixing times, our focus is on statistical estimation for which local and thereby weaker notions of mixing may suffice. This geometric intuition shares conceptual similarities with the works of (45; 29) on decomposable Markov chains and of (39) on spectral partitioning of graphs via eigenvectors of the adjacency matrix, which thus also connect with the block Markov chains mentioned previously.
this section cite: ['b10', 'b16', 'b21', 'b38']

Section: Preliminaries

this section cite: []

Section: MDPs and shifted successor measures
Consider a Markov Decision Process (MDP) with finite state space S and action space A ∶= ⋃ s∈S A s , where A s denotes the set of actions available in state s. Define the set of state-action pairs as X ∶= ⋃ s∈S {s} × A s , and let n denote its cardinality. We write x = (s, a) to denote a generic element of X . The dynamics of the MDP are governed by a transition matrix P ∈ R X ×S , where P (s, a, s ′ ) represents the probability of transitioning to state s ′ when taking action a in state s.
A policy is defined as a stochastic matrix π ∈ R S×A , where π(s, a) denotes the probability of selecting action a in state s. The policy π induces a Markov chain over X with transition matrix P π , defined as: P π ((s, a), (s ′ , a ′ )) = P (s, a, s ′ )π(s ′ , a ′ ). The MDP is completed by specifying a reward function R ∶ X → R. When the state-action pair (s, a) is visited at time step t ≥ 0, a reward of R(s, a) is received. Given a discount factor γ ∈ (0, 1) the performance of a policy π is characterized by its Q-function: Q (R,π) (s, a) = E [ ∑ t≥0 γ t R(s π t , a π t ) | (s π 0 , a π 0 ) = (s, a)], where (s π t , a π t ) is the state-action pair visited under π at time t, or through its value function V (R,π) (s) ∶= ∑ a∈As π(s, a)Q (R,π) (s, a). The Q-function can be expressed as a matrix-vector product. To make this explicit, define the successor measure as M π ∶= (I -γP π ) -1 ∈ R X ×X . Then Q (R,π) (s, a) = ∑ t≥0 γ t P t π R(s, a) = M π R(s, a), where we use matrix product notation: M π R(s, a) ∶= ∑ (s ′ ,a ′ )∈X M π ((s, a), (s ′ , a ′ ))R(s ′ , a ′ ). This formulation separates the dynamics from the rewards, showing that evaluating a policy for any reward function reduces to computing M π (15; 63). The problem of estimating the successor measure is referred to as reward-free policy evaluation. For this problem, we would like to obtain guarantees w.r.t. the ∥ ⋅ ∥ ∞,∞ norm defined as ∥A∥ ∞,∞ ∶= sup f ∈R X ∶∥f ∥ ∞ =1 ∥Af ∥ ∞ . Indeed, suppose that we have an estimate Mπ of M π , and hence an estimate Q(R,π) = Mπ R of the Q-function. This in turn allows us to improve the policy by acting greedily with respect to Q(R,π) . However, for this procedure to be reliable, we require entry-wise control over the error in Q(R,π) , which can be guaranteed by bounding the error in Mπ in the ∥ ⋅ ∥ ∞,∞ norm: ∥ Q(R,π) -Q (R,π) ∥ ∞ = ∥ Mπ R -M π R∥ ∞ ≤ ∥ Mπ -M π ∥ ∞,∞ ∥R∥ ∞ . As we show later in the paper, obtaining accurate estimates of M π can be statistically challenging. The objective of this paper is to explain why shifting the successor measure may address this challenge.
Definition 1 (k-shifted successor measure). Let k ≥ 0. The k-shifted successor measure is defined as M π,k ∶= P k π (I -γP π ) -1 .
The k-shifted successor measure M π,k captures the dynamics of policy π starting from time step k onward. It allows us to quantify the cumulative discounted reward collected under π after the first k steps. For any reward function R, it satisfies: M π,k R(s, a) = ∑ t≥0 γ t P t+k π R(s, a).
this section cite: []

Section: Measure-induced norms and SVD
To analyze the accuracy of estimators of the (shifted) successor measure w.r.t. to the ∥ ⋅ ∥ ∞,∞ norm and make the link with mixing phenomena, we will use measure-induced norms and SVD (refer to Appendix A.1 for a detailed description). Consider a probability measure ν on X whose support is Xfoot_0 .
For f, g ∈ R X , define the ν-scalar product as ⟨f, g⟩ ν ∶= ∑ x∈X ν(x)f (x)g(x), so that (R X , ⟨⋅, ⋅⟩ ν ) is a Hilbert space. We define for all f ∈ R X , M ∈ R X ×X the ν-induced norms as: for any p, q ∈ [1, ∞],
∥f ∥ ℓ p (ν) ∶= { (∑ x∈X ν(x) |f (x)| p ) 1/p if p < ∞ max x∈X |f (x)| if p = ∞ , ∥M ∥ ℓ p (ν),ℓ q (ν) ∶= sup f ∈R X ∶ f ≠0 ∥M f ∥ ℓ q (ν)
∥f ∥ ℓ p (ν)
.
For simplicity, we keep the measure implicit and use the notation ∥f ∥ p = ∥f ∥ ℓ p (ν) and ∥M ∥ p,q = ∥M ∥ ℓ p (ν),ℓ q (ν) . Note that ∥ ⋅ ∥ ∞ does not depend on ν. We will be mostly interested in the spectral norm ∥M ∥ 2,2 and the two-infinity norm ∥M ∥ 2,∞ , as we always have ∥ ⋅ ∥ ∞,∞ ≤ ∥ ⋅ ∥ 2,∞ . Using ν, we can define the notions of adjoint of a vector f and of a matrix M : f † (x) = ν(x)f (x) and M † (x, y) = ν(x)M (y,x) ν(y)
. This allows us to revise the notion of singular value decomposition by replacing the usual transpose operator with the adjoint. Definition 2 (ν-SVD). The ν-SVD of the matrix M ∈ R n×n takes the form M = U ΣV † where Σ = Diag((σ i ) n i=1 ) is a diagonal matrix made of non-negative values that we always assume to be in non-increasing order: σ 1 ≥ σ 2 ≥ . . ., while U, V ∈ R n×n are unitary in the sense U † U = U U † = I and V † V = V V † = I. The ν-SVD can be expressed as M = ∑ n i=1 σ i ψ i ϕ † i , where the left and right singular vectors (ψ i ) i , (ϕ i ) i form orthonormal bases (ψ † i ψ i = 1 and ψ † i ψ j = 0 for i ≠ j). The entries of U, V are then U (x, i) = √ ν(i)ψ i (x), V (x, i) = √ ν(i)ϕ i (x).
Given r ≥ 0, we write [M ] r = U r Σ r V † r for the ν-SVD truncated to rank r and [M ] >r = M -[M ] r . We finally note that the usual SVD corresponds to the case where ν is uniform, up to a normalizing factor n. In what follows, to simplify, the ν-SVD is referred to as the SVD.
this section cite: []

Section: Spectral recoverability
Our goal is to estimate the (shifted) successor measure with entry-wise guarantees by approximating the corresponding matrix via an estimate of its truncated SVD. Truncated SVD is a well-established technique for matrix approximation when considering the Frobenius or nuclear norm. By the Eckart-Young-Mirsky theorem, for a matrix M ∈ R n×n , its rank-r truncated SVD [M ] r provides the optimal rank-r approximation with respect to the Frobenius norm, with error ∥[M ] >r ∥ 2 F = ∑ n i=r+1 σ 2 i entirely determined by the spectral tail. When estimating the matrix from samples of its entries, the entry-wise error often depends on the coherence of the top r singular vectors. Coherence measures how concentrated or spread out the singular vectors are with respect to the standard basis. High coherence implies that a few entries dominate, making estimation from partial observations harder, while low coherence suggests that all entries are comparably informative. For detailed discussions, see, e.g., (8; 53; 48). In our setting, we adopt a similar notion of coherence. For the top r left singular vectors (ψ i ) r i=1 of M , we define the coherence as: c(( 2 . When we seek guarantees in entry-wise norms such as ∥ ⋅ ∥ 2,∞ or ∥ ⋅ ∥ ∞,∞ , it is not clear whether the truncated SVD [M ] r still yields a meaningful approximation of M . It is also not obvious what quantity governs the estimation error when attempting to recover [M ] r from sampled entries. To address these questions, we introduce the concept of spectral (ir)recoverability, which serves as a suitable quantity for controlling the approximation and estimation errors when the ∥ ⋅ ∥ 2,∞ or ∥ ⋅ ∥ ∞,∞ norms are considered. Definition 3 (Spectral (ir)recoverability). Let M ∈ R n×n and let
ψ i ) r i=1 ) ∶= 1 r ∥U r ∥ 2 2,∞ = max x∈[n] 1 r ∑ r i=1 ψ i (x)
M = ∑ n i=1 σ i ψ i ϕ † i be its SVD. The spectral irrecoverability of M is ξ(M ) ∶= max x∈[n] ∑ n i=1 σ i ψ i (x) 2 . The spectral recoverability is ξ(M ) -1 .
The spectral irrecoverability of a matrix M can be interpreted as a nuclear norm weighted by the left singular vectors of M , and it quantifies both the low-rank structure and coherence of the matrix. As stated in the following lemma, proved in Appendix A, the low-rank approximation error of M in the ∥ ⋅ ∥ 2,∞ or ∥ ⋅ ∥ ∞,∞ norm is controlled by ξ(M ). Lemma 1. Let M ∈ R n×n . We have: for any
1 ≤ r < n, ∥M -[M ] r ∥ 2,∞ ≤ √ σ r+1 ξ(M ).
This lemma serves as an analogue, under the ∥ ⋅ ∥ 2,∞ norm, of the "key lemma" from (11) (specifically, Lemma 3.5), which underpins a universal thresholding SVD procedure in the Frobenius norm setting.
0 2 4 6 8 shift k 0 2 [M π,k ] >r 2,∞ σ r+1 ξ(M π,k ) r = 25 r = 50 r = 75
Figure 2: Approximation error as a function of the shift parameter k and rank r. The theoretical upper bound serves as a first-order proxy for the entrywise error. We use the standard ∥ ⋅ ∥ 2→∞ norm, which matches (up to a √ n factor) the variant from Section 3.2 under the uniform measure ν. See Section 6 for experimental details.
In our context, the lemma implies that ∥M -[M ] r ∥ 2,∞ ≤ ε for the largest rank r such that σ r ≥ ϵ 2 /ξ(M ). This provides a principled criterion for selecting the rank r in a truncated SVD when targeting an accuracy level ϵ in the ∥ ⋅ ∥ 2,∞ norm. Additionally, for the problem of estimating the matrix from sample entries with ∥ ⋅ ∥ 2,∞ guarantees, we derive a sample complexity lower bound scaling as ξ(M ), see Appendix B.
We conclude with a few remarks. ξ(M ) and ∥M ∥ 2,∞ are closely related as ∥M ∥ 2  2
,∞ = max x ∑ i σ 2 i ψ i (x) 2 ≤ σ 1 ξ(M ).
When M has rank r, the spectral irrecoverability satisfies:
ξ(M ) ≤ σ 1 ∥U r ∥ 2 2,∞ = rσ 1 c((ψ i ) r
i=1 ) which connects ξ(M ) to classical notions of coherence. Finally, as shown in Fig. 2, the low-rank approximation error of the shifted successor measure improves when the shift k increases (see Section 5 for theoretical justifications).
this section cite: []

Section: Estimation of the Shifted Successor Measure

this section cite: []

Section: Main result
We assume access to a dataset of transitions (s, a, s ′ ) collected offline. Let Z s,a denote the number of independent transitions observed from the state-action pair (s, a). Our analysis provides estimation error bounds conditional on these counts, so we may treat the Z s,a as deterministic. Using the data, we form the empirical estimator P (s, a, s ′ ) = Y s,a,s ′ /Z s,a , and given a policy π we can then also form Pπ ((s, a), (s ′ , a ′ )) = P (s, a, s ′ )π(s ′ , a ′ ) as the empirical estimator of P π . We can then build a simple estimator of the k-shifted successor measure M π,k = P k π (I -γP π ) -1 by taking Mπ,k = P k π (I -γ Pπ ) -1 . Our final estimator of M π,k is obtained by computing the truncated ν-SVD [ Mπ,k ] r of Mπ,k . We derive guarantees for this estimator under any probability measure ν of the following form. Let µ be a probability measure on S; we define ν such that ν(s, a) = µ(s)π(s, a) for all (s, a). In the following theorem, σ i denotes the i-th singular value of Mπ,k in the ν-SVD, and ν π,inv denotes the invariant measure of the Markov chain P π . We also define for δ ∈ (0, 1):
Γ δ ∶= max (k, (1 -γ) -1 ) 2 max (s,a),(s ′ ,a ′ )∈X ν(s, a) Z s,a ν(s ′ , a ′ ) log(rn/δ),(1)
E estim ∶= σ 1 max (∥M π,k ∥ 2,∞ , ∥M † π,k ∥ 2,∞ ) σ r (σ r -σ r+1 ) ∥ dν dν π,inv ∥ ∞ ∥ dν π,inv dν ∥ ∞ Γ δ ,(2)
E approx ∶= √ σ r+1 ξ(M π,k ).(3)
Theorem 1. There is a universal constant C > 0 such that for any k ≥ 0, any probability measure ν on X , any 1 ≤ r < n, and all δ ∈ (0, 1), we have, if Γ δ ≤ 1, with probability at least 1 -δ,
∥[ Mπ,k ] r -M π,k ∥ 2,∞ ≤ CE estim + E approx .(4)
In the proof presented in Appendix C, we show that CE estim and E approx are upper bounds on the estimation and approximation errors, respectively: ∥[ Mπ,k ] r -[M π,k ] r ∥ 2,∞ ≤ CE estim and ∥[M π,k ] r -M π,k ∥ 2,∞ ≤ E approx .
this section cite: []

Section: Discussion
We discuss the terms involved in the estimation error upper bound below.
(a) The term A ∶= σ1 max(∥M π,k ∥2,∞,∥M † π,k ∥2,∞) σr(σr-σr+1)
comes from the so-called leave-one-out analysis, a step in the proof that aims at going from error bounds in spectral norm to error bounds in ∥ ⋅ ∥ 2,∞ . The numerator can be controlled via the spectral recoverability of M π,k since ∥M π,k ∥ 2,∞ ≤ σ 1 ξ(M π,k ). For A to be controlled, we hence need to control the spectral recoverability of M π,k , to have r such that σ 1 /σ r is bounded and the gap σ r -σ r+1 is significant. In Appendix C, we discuss how to control σ r -σ r+1 in case of bounded spectral irrecoverability.
(b) The term B ∶= d(ν, ν π,inv ) ∶= ∥ dν dν π,inv ∥ ∞ ∥ dν π,inv dν ∥ ∞
involves the Radon-Nikodym derivative of ν w.r.t. ν π,inv and ν π,inv w.r.t. ν. It captures the discrepancy between ν, used to compute the SVD, and the invariant measure ν π,inv of the Markov chain under policy π. The choice of ν is under the control of the practitioner. In practice, it may correspond to the empirical distribution of the dataset or be chosen arbitrarily, for example, as the uniform distribution, in which case the SVD reduces to the standard SVD. On the other hand, the invariant distribution ν π,inv is more naturally aligned with the dynamics and yields the tightest possible bound. Setting ν = ν π,inv eliminates the multiplicative factor B, resulting in the best-case guarantee. However, estimating ν π,inv exactly may not necessarily be feasible. Theorem 1 accommodates potential mismatch between ν and ν π,inv , showing that it is sufficient for ν to approximate the invariant measure up to a constant factor.
(c) The term C ∶= max (k, (1 -γ) -1 ) 2 comes from extending the concentration results in spectral norm of P to the shifted successor measure Mπ,k . The form of this term critically relies on a comparison of ν with the invariant measure, allowing us to exploit contraction properties and avoid exponential dependence in k or (1 -γ) -1 .
(d) The term D ∶= max (s,a),(s ′ ,a ′ )∈X ν(s,a) Zs,aν(s ′ ,a ′ ) log(rn/δ) can eventually be traced back to the concentration in spectral norm of the empirical estimator P , and is the only term that depends on the number of observations: if we want ξ small this factor shows how large each Z s,a should. Because of the ratio ν(s,a)  ν(s ′ ,a ′ ) , the result applies primarily to the case where ν exhibits some kind of homogeneity.
this section cite: []

Section: References
Ref_id:b0 Title: Entrywise eigenvector analysis of random matrices with low expected rank Year: (2020)
Ref_id:b1 Title: FLAMBE: Structural Complexity and Representation Learning of Low Rank MDPs Year: (2020)
Ref_id:b2 Title: Hindsight experience replay Year: (2017)
Ref_id:b3 Title: Minimax PAC bounds on the sample complexity of reinforcement learning with a generative model Year: (2013)
Ref_id:b4 Title: Successor features for transfer in reinforcement learning Year: (2017)
Ref_id:b5 Title: Learning successor states and goal-dependent values: A mathematical viewpoint Year: ()
Ref_id:b6 Title: Markov Chains: Gibbs Fields, Monte Carlo Simulation and Queues Year: (2020)
Ref_id:b7 Title: Exact matrix completion via convex optimization Year: (2009-12)
Ref_id:b8 Title: Goal-conditioned reinforcement learning with imagined subgoals Year: ()
Ref_id:b9 Title: Stein's method for concentration inequalities Year: (2007)
Ref_id:b10 Title: Matrix estimation by universal singular value thresholding Year: (2015)
Ref_id:b11 Title: Spectral gap of nonreversible markov chains Year: (2025)
Ref_id:b12 Title: Spectral methods for data science: A statistical perspective Year: (2021)
Ref_id:b13 Title: Markov chains, diffusions and dynamical systems Year: (2013)
Ref_id:b14 Title: Improving Generalization for Temporal Difference Learning: The Successor Representation Year: ()
Ref_id:b15 Title: Logarithmic Sobolev inequalities for finite Markov chains Year: (1996)
Ref_id:b16 Title: Nash inequalities for finite Markov chains Year: (1996)
Ref_id:b17 Title: Bilinear classes: A structural framework for provable generalization in RL Year: (2021-07-24)
Ref_id:b18 Title: State aggregation learning from markov transition data Year: (2019)
Ref_id:b19 Title: Imitating past successes can be very suboptimal Year: (2022)
Ref_id:b20 Title: Contrastive learning as goal-conditioned reinforcement learning Year: (2009)
Ref_id:b21 Title: Eigenvalue bounds on convergence to stationarity for nonreversible markov chains, with an application to the exclusion process Year: (1991)
Ref_id:b22 Title: D4rl: Datasets for deep data-driven reinforcement learning Year: (2020)
Ref_id:b23 Title: Mixing time bounds via the spectral profile Year: (2006)
Ref_id:b24 Title: Proper laplacian representation learning Year: (2024)
Ref_id:b25 Title: Expander graphs and their applications Year: (2006)
Ref_id:b26 Title: Topics in matrix analysis Year: (1994)
Ref_id:b27 Title: Nearly optimal latent state decoding in block mdps Year: (2023-04)
Ref_id:b28 Title: Elementary bounds on Poincaré and log-Sobolev constants for decomposable Markov chains Year: (2004)
Ref_id:b29 Title: Contextual decision processes with low Bellman rank are PAC-learnable Year: (2017-08)
Ref_id:b30 Title: Reward-free exploration for reinforcement learning Year: (2020-07-18)
Ref_id:b31 Title: Bellman eluder dimension: New rich classes of RL problems, and sample-efficient algorithms Year: (2021)
Ref_id:b32 Title: Provably efficient reinforcement learning with linear function approximation Year: (2020-07)
Ref_id:b33 Title: Reversibility and Stochastic Networks Year: (2011)
Ref_id:b34 Title: Deep laplacian-based options for temporallyextended exploration Year: (2023-07-29)
Ref_id:b35 Title: Deep successor reinforcement learning Year: (2016)
Ref_id:b36 Title: Bridging rl theory and practice with the effective horizon Year: (2024)
Ref_id:b37 Title: On the generalization of representations in reinforcement learning. In Gustau Camps-Valls Year: (2022-03-30)
Ref_id:b38 Title: Multi-way spectral partitioning and higher-order Cheeger inequalities Year: (2012)
Ref_id:b39 Title: Second edition of [ MR2466937], With contributions by Elizabeth L. Wilmer, With a chapter on Year: (2017)
Ref_id:b40 Title: Temporal abstraction in reinforcement learning with the successor representation Year: (2023)
Ref_id:b41 Title: A laplacian framework for option discovery in reinforcement learning Year: (2017-08)
Ref_id:b42 Title: Eigenoption discovery through the deep successor representation Year: (2018-05-03)
Ref_id:b43 Title: Matrix concentration inequalities via the method of exchangeable pairs Year: (2014)
Ref_id:b44 Title: Markov chain decomposition for convergence rate analysis Year: (2002)
Ref_id:b45 Title: Value function approximation with diffusion wavelets and laplacian eigenfunctions Year: (2005)
Ref_id:b46 Title: Proto-value functions: A laplacian framework for learning representation and control in markov decision processes Year: (2007)
Ref_id:b47 Title: Can matrix coherence be efficiently and accurately estimated? Year: (2011-04)
Ref_id:b48 Title: Mathematical aspects of mixing times in Markov chains Year: (2005)
Ref_id:b49 Title: Data-efficient hierarchical reinforcement learning Year: (2018)
Ref_id:b50 Title: Hiql: Offline goalconditioned rl with latent states as actions Year: (2023)
Ref_id:b51 Title: Deriving matrix concentration inequalities from kernel couplings Year: (2013)
Ref_id:b52 Title: A simpler approach to matrix completion Year: (2004)
Ref_id:b53 Title: Ecole d'été de probabilités de Saint-Flour XXVI-1996. Lectures given at the Saint-Flour summer school of probability theory Year: (1996-09-04)
Ref_id:b54 Title: Overcoming the long horizon barrier for sample-efficient reinforcement learning with latent low-rank structure Year: ()
Ref_id:b55 Title: Clustering in block Markov chains Year: (2020)
Ref_id:b56 Title: Sample efficient reinforcement learning via low-rank matrix estimation Year: (2020)
Ref_id:b57 Title: Design principles of the hippocampal cognitive map Year: (2014-12-08)
Ref_id:b58 Title: Spectral entry-wise matrix estimation for low-rank reinforcement learning Year: (2023)
Ref_id:b59 Title: Model-free low-rank reinforcement learning via leveraged entry-wise matrix estimation Year: (2024)
Ref_id:b60 Title: Modelbased RL in contextual decision processes: PAC bounds and exponential improvements over model-free approaches Year: (2019-06-28)
Ref_id:b61 Title: Learning one representation to optimize all rewards Year: (2021)
Ref_id:b62 Title: Does zero-shot reinforcement learning exist? Year: (2023)
Ref_id:b63 Title: The laplacian in RL: learning representations with efficient approximations Year: (2019-05-06)
Ref_id:b64 Title: Matrix estimation for offline reinforcement learning with low-rank structure Year: (2023)
Ref_id:b65 Title: Sample-optimal parametric q-learning using linearly additive features Year: (2019-06)
Ref_id:b66 Title:  Year: (2019)
Ref_id:b67 Title: Harnessing structures for value-based planning and reinforcement learning Year: (2020)
Ref_id:b68 Title: Spectral state compression of Markov processes Year: (2020)
Ref_id:b69 Title: Making Linear MDPs Practical via Contrastive Representation Learning Year: (2002)
