Title: Purifying Approximate Differential Privacy with Randomized Post-processing
Abstract: We propose a framework to convert (ε, δ)-approximate Differential Privacy (DP) mechanisms into (ε ′ , 0)-pure DP mechanisms under certain conditions, a process we call "purification." This algorithmic technique leverages randomized postprocessing with calibrated noise to eliminate the δ parameter while achieving nearoptimal privacy-utility tradeoff for pure DP. It enables a new design strategy for pure DP algorithms: first run an approximate DP algorithm with certain conditions, and then purify. This approach allows one to leverage techniques such as strong composition and propose-test-release that require δ > 0 in designing pure-DP methods with δ = 0. We apply this framework in various settings, including Differentially Private Empirical Risk Minimization (DP-ERM), stability-based release, and query release tasks. To the best of our knowledge, this is the first work with a statistically and computationally efficient reduction from approximate DP to pure DP. Finally, we illustrate the use of this reduction for proving lower bounds under approximate DP constraints with explicit dependence in δ, avoiding the sophisticated fingerprinting code construction.

Section: Introduction
Differential privacy (DP), in its original form [DMNS06, Definition 1], has only one privacy parameter ε. Over the two decades of research in DP, many have advocated that DP is too stringent to be practical and have proposed several relaxations. Among them, the most popular is arguably the approximate DP [DKM + 06], which introduces a second parameter δ.
Definition 1 (Differential privacy [DMNS06, DR + 14]) A mechanism M satisfies (ε, δ)differential privacy if, for all neighboring datasets D ≃ D ′ (datasets differing in at most one entry) and for any measurable set S ⊆ Range(M), it holds that:
P[M(D) ∈ S] ≤ e ε P[M(D ′ ) ∈ S] + δ.
When δ = 0, the definition is now fondly referred to as ε-pure DP. Choosing δ > 0 significantly weakens the protection, as it could leave any event with probability smaller than δ completely unprotected.
Two common reasons why DP researchers adopt this relaxation are: 1. Utility gain: approximate DP is perceived to be more practical, as it allows for larger utility; 2. Flexible algorithm design: Many algorithmic tools (such as advanced composition and Propose-Test-Release) support approximate DP but not pure DP, which enables more flexible (and often more efficient) algorithm design when δ > 0 is permitted. For these two reasons, it is widely believed that the relaxation is a necessary evil.
We argue that the claim of "worse utility for pure-DP" is oftentimes a myth. In some applications of DP, a pure-DP mechanism has both stronger utility and stronger privacy. For example, in lowdimensional private histogram release, the Laplace mechanism enjoys smaller variance in almost all regimes except when δ is too large to be meaningful (see Figure 1). In other cases, the poor utility is sometimes not caused by an information-theoretic barrier, but rather due to the second issue -it is often much harder to design optimal pure-DP mechanisms.
In the problem of privately releasing k linear queries Figure 1: Per-coordinate variance of private histogram release under pure-DP and approximate DP with the same ε parameter. The dashed line is given by Laplace mechanism with pure 1.0-DP, regardless of δ. The solid line is generated using the analytic Gaussian mechanism in [BW18].
of the dataset in {0, 1} d , it may appear that pure-DP mechanisms are much worse if we only consider composition-based methods. The composition of the Gaussian mechanism enjoys an expected worst-case error of O p (
√ k log(1/δ) nε
). On the other hand, if we wish to achieve pure DP, the composition of the Laplace mechanism's error bound becomes O p ( k nε ). However, the bound can be improved to O p ( √ kd nε ) when we use a more advanced pure-DP algorithm known as the K-norm mechanism [HT10] that avoids composition.
In the problem of private empirical risk minimization (DP-ERM) [BST14], the optimal excess empirical risk of
Θ √ d log(1/δ) nε
under approximate DP is achieved by the noisy-stochastic gradient descent (NoisySGD) mechanism and its full-batch gradient counterpart. These methods are natural and are by far the strongest approaches for differentially private deep learning as well [ACG + 16, DBH + 22]. The optimal rate of Θ( d nε ) for pure DP, on the contrary, cannot be achieved by a Laplace noise version of NoisySGD, as advanced composition is not available for pure DP. Instead, it requires an exponential mechanism that demands a delicate method to deal with the mixing rate and sampling error of certain Markov chain Monte Carlo (MCMC) sampler [LMW + 24].
The issue with the lack of algorithmic tools for pure DP becomes more severe in data-adaptive DP mechanisms. For example, all Propose-Test-Release (PTR) style methods [DL09] involve privately testing certain properties of the input dataset, which inevitably incurs a small failure probability that requires choosing δ > 0. While smooth-sensitivity-based methods [NRS07] can achieve pure DP, they require adding heavy-tailed noise, which causes the utility to deteriorate exponentially as the dimension d increases.
this section cite: ['b14', 'b41', 'b11', 'b20', 'b57']

Section: Summary of the Results
In this paper, we develop a new algorithmic technique called "purification" that takes any (ε, δ)approximate DP mechanism and converts it into an (ε + ε ′ )-pure DP mechanism, while still enjoying similar utility guarantees of the original algorithm.
The contributions of our new technique for achieving pure DP are as follows.
1. It simplifies the design of the near-optimal pure DP mechanism by allowing the use of tools reserved for approximate DP.
2. It enables an O( √ k)-type composition without compromising the DP guarantee with δ > 0, where k is the number of composition.
3. It shows that PTR-like mechanisms with pure DP are possible! To the best of our knowledge, our method is the only pure DP method for such purposes.
4. In DP-ERM and private selection problems, we show that, up to a logarithmic factor, the resulting pure DP mechanism enjoys an error rate that matches the optimal rate for pure DP, i.e., replacing the log(1/δ) term with the dimension d or log(|OutputSpace|).
this section cite: []

Section: Technical summary.
The main idea of the "purification" technique is to use a randomized postprocessing approach to "smooth" out the δ part of (ε, δ)-DP. To accomplish this, we proceed as follows. We leverage an equivalent definition of (ε, δ)-DP that interprets δ as the total variation distance from a pair of hypothetical distributions that are ε-indistinguishable. Next, we develop a method to convert the total variation distance to the ∞-Wasserstein distance. Finally, we leverage
Table 1: Summary of applications of our purification technique for constructing new pure-DP mechanisms from existing approximate DP mechanisms. The resulting pure-DP mechanisms are either information-theoretically optimal or match the results of the best-known pure-DP mechanisms for the task (we include more discussion in Appendix G). DP-ERM (SC) refers to DP-ERM with a strongly convex objective function, where λ denotes the strong convexity parameter of the individual loss function. DP-ERM (ℓ1) refers to DP-ERM with an ℓ1 constraint. λmin in linear regression is the smallest eigenvalue of the sample covariance matrix, X T X/n. All results are presented up to a logarithmic factor (in n and other parameters, but not in 1/δ. We note that in the
table log(1/δ) is proportional to d, the effective dimension.) Problem (ε, δ)-DP Mechanism Utility (before purification) Utility (after) DP-ERM Noisy-SGD [BST14] √ d log(1/δ)L∥C∥2 nε dL∥C∥2 nε (Thm 3) DP-ERM (SC) Noisy-SGD [BST14] d log(1/δ)Lfoot_0 λn 2 ε 2 d 2 L 2 λn 2 ε 2 (Thm 3) DP-ERM (ℓ 1 ) DP-Frank-Wolfe [TTZ14] (log d) 2/3 (log(1/δ)) 1/3 (nε) 2/3 log d nε (Thm 4) Bounding ∆ local PTR-type [KNRS13, DRE + 20] d 1/2 (∆local+log(1/δ)/ε) ε d 1/2 ∆local ε + d 3/2 ε 2 (Thm 5) Mode Release Distance to Instability [TS13] log(1/δ) ε log |X | ε
(Thm 6) Linear Regression AdaSSP [Wan18] min{
√ d log(1/δ) nε , d log(1/δ) λminn 2 ε 2 } min{ d nε , d 2 λminn 2 ε 2 } (Thm 7) Query Release
MWEM [HLM12] (log k) 1/2 (d log(1/δ)) 1/4 √ nε
(log k) 1/3 d 1/3 (nε) 1/3 (Thm 8)
the Approximate Sample Perturbation (ASAP) technique from [LMW + 24] that achieves pure DP by adding Laplace noise proportional to the ∞-Wasserstein distance. A challenge arises because the output distribution of a generic (ε, δ)-DP mechanism is not guaranteed to be supported on the entire output space, which may invalidate a tight TV distance to W ∞ conversion. We address this by interlacing a very small uniform distribution over a constraint set. Another challenge is that, unlike in [LMW + 24], where the ε-indistinguishable distributions correspond to pure DP mechanisms on neighboring datasets, here the hypothetical distribution may depend on both neighboring datasets rather than just one. This prevents the direct application of the standard DP analysis of the Laplace mechanism and the composition theorem. To address this, we formulate our analysis in terms of the indistinguishability of general distributions rather than DP-specific language. We use a Laplace perturbation lemma and the weak triangle inequality, leading to a clean and effective analysis. Moreover, we show how the dimension-reduction technique can be used so the purification technique can be applied to discrete outputs and to sparse outputs, which ensures only logarithmic dependence in the output-space cardinality or dimension.
this section cite: ['b73', 'b39']

Section: Related Work
The idea of using randomized post-processing to enhance privacy guarantees has been explored in prior work [FMTT18, MV22, LMW + 24]. [FMTT18] focus on Rényi differential privacy, while [MV22, LMW + 24] study the implementation of the exponential mechanism, specifically perturbing Markov chain Monte Carlo (MCMC) samples to obtain pure DP guarantees. Our work builds on [LMW + 24], where we generalize the domain assumption of [LMW + 24, Lemma 8] and extend the approach to more general upstream approximate DP mechanisms, such as DP-SGD [ACG + 16, DBH + 22]. We also discuss the connection to the classical statistics literature on randomized postprocessing given by [B + 51, Theorem 10] in Appendix B.
Previous works have investigated the purification of approximate differential privacy into pure DP, but limited to settings with a finite output space. A straightforward uniform mixing method can purify approximate DP mechanisms with finite output spaces, as summarized in [HC22], and we further discuss it in Appendix C. [BGH + 23] first transform an approximate DP mechanism into a replicable one, and then apply a pure DP selection procedure to obtain a pure DP mechanism 2 , at the cost of increased sample complexity.
2 Algorithm: Converting Approximate DP to Pure DP
In this section, we propose the purification algorithm (Algorithm 1) which converts (ε, δ)-approximate DP mechanisms with continuous output spaces into ε ′ -pure DP mechanisms under certain conditions. The algorithm consists of two steps: (1) mixing the approximate DP output with a uniform distribution (Line 3), and (2) adding Laplace noise calibrated to δ 1 d (Line 4.) Intuitively, the first step is to enforce a bound on the ∞-Wasserstein distance (Definition 5), which can be loosely interpreted as a randomized analogue of ℓ 1 sensitivity. The second step adds Laplace noise proportional to this Wasserstein bound to guarantee pure DP. This step is based on techniques from [SWC17, LMW + 24], and can be viewed as the generalization of the Laplace mechanism. The privacy and utility guarantee of Algorithm 1 are provided in Theorem 1.
Algorithm 1: A pure (x apx , Θ, ε ′ , δ, ω): Purification of Approximate Differential Privacy 1 Input: Privacy parameters ε, δ, additional privacy budget ε ′ , an output x apx of an (ε, δ)-DP algorithm M satisfying Assumption 1, an ℓ q ball Θ as Assumption 1, the mixture level ω
2 Set ∆ ← 2d 1-1 q R δ2ω
1 d ▷ Lemma 6. 3 With probability 1 -ω, set x ← x apx ; otherwise, sample x ∼ Unif(Θ). ▷ Uniform Mixing 4 x pure ← x + Laplace ⊗d (2∆/ε ′ ) 5 Output: x pure
Notations. Let X be the space of data points, X * := ∪ ∞ n=0 X n be the space of the data set. For a vector v = (v 1 , . . . , v d ) ∈ R d and q ≥ 1, we define its ℓ q -norm as ∥v∥ q :
= d i=1 |v i | q 1/q
. For a set S ⊆ R d , we denote its ℓ q -norm diameter Diam q (S) := sup x,y∈S ∥x -y∥ q . For a (randomized) function M : X → R d , we denote its range as Range(M) = {M(D) : D ∈ X * }.
Assumption 1. The (ε, δ)-DP algorithm M satisfies Diam q (Range(M)) ≤ R. Specifically, it lies in an ℓ q ball of radius R, denoted by Θ.
Theorem 1 Define x pure , x apx as in Algorithm 1 under Assumption 1. The output of Algorithm 1 satisfies (ε + ε ′ )-DP with utility guarantee
E ∥x pure -x apx ∥ 1 ≤ ωR + 4Rd ε ′ δ 2ω 1 d .
The detailed proofs are deferred to Appendix E and Appendix F. For clarity, Theorem 1 presents the utility guarantee in the ℓ 1 norm. Extensions to general ℓ p norms follow directly from bounding the expected ℓ q norm of the Laplace noise (see Equation ( 5)).
Remark 2 When applying Algorithm 1 to various settings as shown in Table 1, the utility bounds either match the known information-theoretic lower bounds for pure DP or the best-known pure-DP mechanisms for the task. By the parameter setting given in Line 2 of Algorithm 1, the log(1/δ) factor in the utility bounds can be replaced by d, omitting the logarithmic factors. In Section 4.2, we further show how dimension-reduction techniques can be used when applying purification to settings with sparsity conditions.
Remark 3 Parameter choices of Algorithm 1 for different settings are provided in later sections. For example, parameters for the purified DP-SGD are given in Corollary 4. Note that in Algorithm 1, we only require the range of M to be a subset of the ℓ q -ball Θ, where Θ can be selected as ℓ 1 balls (q = 1), ℓ 2 balls (q = 2), or hypercubes (q = ∞), which admits simple O(d)-runtime uniform sampling oracles.
Corollary 4 (Parameters of Algorithm 1 for DP-SGD) Let M : X * → Θ be an (ε, δ)-DP mechanism, where Θ ⊂ R d is an ℓ 2 ball with ℓ 2 -diameter C. Let x apx be the output of M. Set the mixture level parameter as ω = 1 n 2 , and set δ = 2ω (16Cdn 2 ) d . Then, A pure (x apx , Θ, ε ′ = ε, δ, ω) satisfies 2ε-DP guarantee with utility bound E[∥x pure -x apx ∥ 2 ] ≤ 1 n 2 ε + C n 2 . The purification algorithm can be applied to finite output spaces. The key idea is to embed the elements in the finite output space into a hypercube using the binary representation. Given a finite output space Y = {1, 2, 3, . . . ,
2 d } . = [|2 d |]
, we first map each element to its binary representation in {0, 1} d . Then, we apply Algorithm 1 on the cube [0, 1] d in the Euclidean Space R d . The procedure is outlined in Algorithm 2 with DP and utility guarantees provided in Theorem 2, and the proof is deferred to Appendix F.
Algorithm 2: A pure-discrete (ε, δ, u apx , Y): Binary Embedding Purification for Finite Spaces 1 Input: privacy parameters ε, δ, binary representation mapping BinMap : [
2 d ] → {0, 1} d , output u apx from (ε, δ)-DP mechanism M : X * → Y = [2 d ], 2 z bin ← BinMap(u apx ) ▷ Binary embedding 3 z pure ← A pure (z bin , Θ = [0, 1] d , ε ′ = ε, δ, ω = 2 -d ) ▷ Purify the embedding by Algorithm 1 4 z round ← Round {0,1} d (z pure ) ▷ Round {0,1} d (x) = (1(x i ≥ 0.5)) d i=1 5 u pure ← BinMap -1 (z round ) ▷ Decode index back to decimal integer index 6 Output: u pure Theorem 2 If δ < ε d (2d) 3d
, then Algorithm 2 satisfies (2ε, 0)-pure DP with utility guarantee
P [u apx = u pure ] > 1 -2 -d -d 2 e -d .
3 Technical Lemma: from TV distance to ∞-Wasserstein distance
In this section, we present a technical lemma that proofs the uniform mixing step in Algorithm 1 (Line 3) can enforce an ∞-Wasserstein distance bound, as mentioned in Section 2. This ∞-Wasserstein bound is a key step in our analysis, enabling the subsequent addition of Laplace noise calibrated to this bound to ensure pure DP. See Figure 4 for a summary of the privacy analysis. We define the ∞-Wasserstein distance below. Additional discussion can be found in Appendix A.
this section cite: ['b30', 'b38']

Section: Definition 5
The ∞-Wasserstein distance between distributions µ and ν is on a separable Banach space (Θ, ∥ • ∥ q ) is defined as
W ℓq ∞ (µ, ν) = inf γ∈Γc(µ,ν) ess sup (x,y)∼γ ∥x -y∥ q = inf γ∈Γc(µ,ν) {α | P (x,y)∼γ [∥x -y∥ q ≤ α] = 1},
where Γ c (µ, ν) is the set of all couplings of µ and ν. The expression ess sup (x,y)∼γ denotes the essential supremum with respect to the measure γ.
By the equivalent characterization of approximate DP (Lemma 14), we can derive a TV distance bound between the output distributions of the (ε, δ)-DP mechanism on neighboring datasets and a pair of distributions that are ε-indistinguishable. Our goal is to translate this TV distance bound into a W ∞ distance bound. In general, the total variation distance bound does not imply a bound for the W ∞ distance. However, when the domain is bounded, we have the following result.
Lemma 6 (Converting d TV to W ∞ ) Let q ≥ 1, and let Θ ⊆ R d be a convex set with ℓ q -norm diameter R and containing an ℓ q -ball of radius r. Let µ and ν be two probability measures on Θ. Suppose ν is the sum of two measures, ν = ν 0 + ν 1 , where ν 1 is absolutely continuous with respect to the Lebesgue measure and has density lower-bounded by a constant p min over Θ, while ν 0 is an arbitrary measure. Define the W ∞ distance with respect to ℓ q (Definition 5.) The following holds.
If d TV (µ, ν) < p min • Vol(B d ℓq (1)) • r 4R d • ∆ d , then W ℓq ∞ (µ, ν) ≤ ∆,(1)
where
Vol(B d ℓq (1)) = 2 d Γ(1+ d q ) d i=1 Γ(1+ 1 q ) Γ(1+ i q )
is the Lebesgue measure of the ℓ q -norm unit ball, with Γ being the Gamma function. E.g.,
Vol(B d ℓ2 (1)) = π d/2 Γ( d 2 +1) , and Vol(B d ℓ1 (1)) = 2 d Γ(d+1) .
In particular, if the domain Θ is an ℓ q -ball, then the term ( r 4R ) Eq. (1) can be improved to 1 4 . This result builds on [LMW + 24] while generalizing its domain assumption from the ℓ 2 -balls to more general convex sets. In the proof provided in Appendix F, instead of relying on [LMW + 24, Lemma 24], which applies only to ℓ 2 -balls, we construct a convex hull that extends to more general convex sets. We provide the proof sketch as follows.
Proof sketch of Lemma 6 To prove Lemma 6, we use an equivalent, non-coupling-based definition of the infinity-Wasserstein distance:
Lemma 7 ([GS84], Proposition 5) Define µ, ν and W ∞ as Definition 5. Then,
W ℓq ∞ (µ, ν) = inf{α > 0 : µ(U ) ≤ ν(U α )
, for all open subsets U ⊂ Θ}, where the α-expansion of U is denoted by U α := {x ∈ Θ : ∥x -U ∥ q ≤ α}.
This definition provides a geometric interpretation of W ℓq ∞ (µ, ν) by comparing the measure of a set U to the measure of its α-expansion U α .
We summarize the key idea of the proof. Suppose T V (µ, ν) ≤ ξ. By the definition of total variation distance, this implies µ(U ) ≤ ν(U ) + ξ for any measurable set U . To prove that W ℓq ∞ (µ, ν) ≤ ∆ (for the ∆ given in the lemma), it suffices, by Lemma 7, to show that µ(U ) ≤ ν(U ∆ ) for any open set U .
Given that µ(U ) ≤ ν(U ) + ξ, our goal thus reduces to proving ν(U ) + ξ ≤ ν(U ∆ ). Rewriting this inequality, it suffices to prove:
ν(U ∆ \ U ) ≥ ξ.
To establish this, we show that the "expansion band" U ∆ \ U must contain sufficient mass. We use the convexity of Θ to argue that this band must contain a small ℓ q ball of a specific radius (related to ∆). We then use the minimum density p min and the Lebesgue measure of this ℓ q ball to lower-bound its ν-measure. The value of ∆ in the lemma statement is chosen precisely so that this lower bound (and thus ν(U ∆ \ U )) is at least ξ, which completes the sketch.
this section cite: []

Section: ■
The conversion lemma requires one of the distributions to satisfy a minimum density condition (the "p min ".) This motivates the uniform mixing step in Algorithm 1, which ensures that the mixed distribution meets this requirement. Combining the TV bound implied by (ε, δ)-DP, the effect of uniform mixing, and the conversion lemma from TV distance to W ∞ distance, we obtain a bound on the ∞-Wasserstein distance after the uniform mixing step in Algorithm 1.
Example 8 (Tightness of the Conversion) Let ∆ ∈ (0, 1). Consider the probability distributions
µ = ∆d δ Dirac 0 + (1 -∆d )Unif B d ℓq (1) \ B d ℓq(
this section cite: []

Section: Empirical Risk Minimization with Pure Differential Privacy
In this section, we apply our purification technique to develop pure differentially private algorithms for the Differential Private Empirical Risk Minimization (DP-ERM) problem, which has been extensively studied by the differential privacy community [CMS11, BST14, KJ16, WYX17, FKT20, KLL21, GHSGT23].
We consider the convex formulation of the DP-ERM problem, where the objective is to design a differentially private algorithm that minimizes the empirical risk L(θ; D) = 1 n n i=1 f (θ; x i ) given a dataset D = {x 1 , . . . , x n } ⊆ X n , a convex feasible set C ⊆ R d , and a convex loss function f : C × X → R. Algorithm performance is measured by the expected excess empirical risk E A [L(θ)] -L * , where L * = min θ∈C L(θ).
this section cite: []

Section: Purified DP Stochastic Gradient Descent
The Differential Private Stochastic Gradient Descent (DP-SGD) mechanisms [BST14, ACG + 16] are the most popular algorithms for DP-ERM. These mechanisms are inherently iterative and heavily rely on (1) advanced privacy accounting/composition techniques [BS16,Mir17,DRS22] and (2) amplification by subsampling for Gaussian mechanisms [BBG18, BDRS18, WBK19, ZW19, KJH20].
Either of these two techniques results in (ε, δ)-DP guarantee with δ > 0. In contrast, directly using the Laplace mechanism to release gradients fails to achieve a competitive utility rate.
While the exponential mechanism [MT07, BST14] achieves optimal utility rates under (ε, 0)-pure DP, this comes at the expense of increased computational complexity. Specifically, [BST14, Algorithm 2] implements the exponential mechanism via a random walk over the grid points of a cube that contains C, ensuring convergence in terms of max-divergence. This approach constitutes a zero-order method that does not leverage gradient information. To design a fast, pure DP algorithm with nearly optimal utility, we propose a pure DP-SGD method that transforms the output of an (ε, δ)-DP SGD algorithm into a (ε, 0)-pure DP solution using Algorithm 1. Implementation details are provided in Algorithm 4 in Appendix H. Theoretical guarantees on utility, privacy, and computational efficiency are stated in Theorem 3.
Theorem 3 (Utility, privacy, and runtime for purified DP-SGD) Let C ⊂ R d be a convex set with ℓ 2 diameter C, and suppose that f (•; x) is L-Lipschitz for every x ∈ X . Set the parameters as specified in Corollary 4. Algorithm 4 guarantees 2ε-pure differential privacy. Furthermore, using Õ(n 2 ε 3 /2 d -1 ) incremental gradient calls, the resulting output θ pure satisfies:
1. If f (•; x) is convex for every x ∈ X , then E A [L(θ pure )] -L * ≤ Õ ( CLd /nε). 2. If f (•; x) is λ-strongly convex for every x ∈ X , then E A [L (θ pure )] -L * ≤ Õ d 2 L 2 /n 2 λε 2 .
The total runtime of Algorithm 4 is Õ(n 2 ε 3/2 + d), where each incremental gradient computation incurs a cost of O(d) gradient operations, and the purification (Algorithm 1) executes in O(d) time. Table 2 presents a comparison of utility and computational efficiency. Notably, the purified DP-SGD attains a near-optimal utility rate, matching that of the exponential mechanism [BST14], while substantially reducing computation complexity by improving the dependence on both n and d. To obtain a pure-DP estimator from the approximate DP output θ FW , we first apply dimension reduction, exploiting the problem's sparsity to project θ FW into a lower-dimensional space. We then apply the purification Algorithm 1 in this space and recover the estimate in the original ambient space. The full procedure is given in Algorithm 7, with the proof of Theorem 4 deferred to Appendix I.3. Our method achieves pure differential privacy while matching the best known utility rate Õ(n -1/2 ε -1/2 ), as in [AFKT21, Theorem 6].
Theorem 4 Let the domain C be an ℓ 1 -ball centered at 0. Let ε be the pure differential privacy parameter. Assume that the function f (•; x) is convex, L 1 -Lipschitz, and β-smooth with respect to the ℓ 1 norm for all x ∈ X . Algorithm 7 satisfies 2ε-pure differential privacy and achieves the following utility bound:
E A [L(θ pure )] -L * ≤ Õ L 1/2 1 β 1/2 ∥C∥ 3/2 1 (nε) 1/2 .
The runtime is O(dn 3/2 ), plus a single call to a LASSO solver.
The upper bound is tight with respect to n and ε. We provide a lower bound in Lemma 30, which is derived using a packing argument. We defer proof details to Appendix I.4.
this section cite: ['b10', 'b53', 'b24', 'b11']

Section: Pure DP Data-dependent Mechanisms
This section shows that our purification technique offers a systematic approach for designing datadependent pure DP mechanisms. We present three examples: (1) the Propose-Test-Release (PTR) mechanism and its variant with privately released local sensitivity [DL09, KNRS13, DRE + 20]; (2) stable value release methods, such as frequent item identification [TS13,Vad17]; and (3) a linear regression algorithm using adaptive perturbation of sufficient statistics [Wan18].
this section cite: ['b66', 'b68', 'b73']

Section: Propose-Test-Release with Pure Differential Privacy
Given a dataset D ∈ X * and a query function q : X → R, the Propose-Test-Release (PTR) framework proceeds in three steps: (1) Propose an upper bound b on ∆ q Local (D), the local sensitivity of q(D) (as defined in Definition 33); (2) Privately test whether D is sufficiently distant from any dataset that violates this bound; (3) If the test succeeds, assume the sensitivity is bounded by b, and use a differentially private mechanism, such as the Laplace mechanism with scale parameter b/ε, to release a slightly perturbed query response. However, due to the failure probability of the testing step, PTR provides only approximate differential privacy. By applying the purification technique outlined in Algorithm 10 in Appendix J.1, PTR can be transformed into a pure DP mechanism.
Next, we examine a variant of the PTR framework, presented in Algorithm 11, where the output space of the query function has dimension d, and the local sensitivity ∆ q Local (D) is assumed to have bounded global sensitivity. In this approach, Algorithm 11 first privately constructs a high-probability upper bound on the local sensitivity. The query is then released with additive noise proportional to this bound, and the purification technique is applied to ensure a pure DP release. This method enables an adaptive utility upper bound that depends on the local sensitivity, as established in Theorem 5. The proof is provided in Appendix J.2.
Theorem 5 Algorithm 11 satisfies 3ε-DP. Moreover, the output q pure from Algorithm 11 satisfies
E[∥q pure -q(D)∥ 2 ] ≤ Õ d 1/2 ∆ q Local (D) ε + d 3/2 ε 2 .
this section cite: []

Section: Pure DP Mode Release
We address the problem of privately releasing the most frequent item, commonly referred to as mode release, argmax release, or voting [DR + 14]. Our approach follows the distance-to-instability algorithm, A dist , introduced in [TS13] and further outlined in Section 3.3 of [Vad17]. Let X denote a finite data universe, and let D ∈ X n be a dataset. We define the mode function f : X n → X , where f (D) returns the most frequently occurring element in the dataset D.
Theorem 6 Let D ∈ X n be a dataset. Algorithm 12 satisfies 2ε-DP and runs in time O(n + log |X |). Furthermore, if the gap between the frequencies of the two most frequent items exceeds
Ω ( log |X | log(log |X |/ε) /ε), Algorithm 12 returns the mode of D with probability at least 1 -O( 1 /|X |).
The proof is provided in Appendix J.3. Consider the standard Laplace histogram approach, which releases the element with the highest noisy count. To ensure that the correct mode is returned with high probability, the gap between the largest and second-largest counts must exceed Θ (log |X |/ε) [Vad17, Proposition 3.4]. Our result matches this bound up to a log log |X | factor.
this section cite: ['b66', 'b68']

Section: Pure DP Linear Regression
We conclude this section by presenting a pure DP algorithm for the linear regression problem. Given a fixed design matrix X ∈ X ⊂ R n×d and a response variable Y ∈ Y ⊂ R n , we assume the existence of θ * ∈ Θ such that Y = Xθ * . The non-private ordinary least squares estimator (X ⊤ X) -1 X ⊤ Y , requires computing the two sufficient statistics: X ⊤ X and X ⊤ Y . These can be privatized using Sufficient Statistics Perturbation (SSP) [VS09,FGWC16] or its adaptive variant, AdaSSP [Wan18], to achieve improved utility.
To facilitate the purification procedure, we first localize the output of AdaSSP by deriving a highprobability upper bound. This bound is then used to clip the output of AdaSSP, after which the purification technique is applied. Implementation details are provided in Algorithm 13, and the corresponding utility guarantee is stated in Theorem 7. The proof is deferred to Appendix J.4.
this section cite: ['b71', 'b28', 'b73']

Section: Theorem 7
Assume X ⊤ X is positive definite and ∥Y∥ 2 ≲ ∥X ∥ 2 ∥θ * ∥ 2 . Then, with high probability, the output θ pure of Algorithm 13 satisfies:
MSE(θ pure ) ≤ Õ d∥X ∥ 2 2 ∥θ * ∥ 2 2 nε ∧ d 2 ∥X ∥ 4 2 ∥θ * ∥ 2 2 ε 2 n 2 λ min ,(2)
Here, λ min denotes the normalized minimum eigenvalue, defined as λ min (X ⊤ X/n), and MSE denotes the mean squared error, given by MSE(θ) = 1 2n ∥Y -Xθ∥ 2 2 . We note that [AD20] also proposes an ε-DP mechanism for linear regression using the approximate inverse sensitivity mechanism, which has an excess mean squared error of
O dL 2 tr(X ⊤ X/n) /n 2 ε 2 ,
where L is the Lipschitz constant of the individual mean squared error loss function. We make two comparisons: (1) compared to the utility bound in [AD20, Proposition 4.3], Theorem7 avoids dependence on the Lipschitz constant, which in turn relies on the potentially unbounded quantity ∥Θ∥ 2 ; (2) both algorithms exhibit a computational complexity of O(nd 2 + d 3 ). Nonetheless, the inverse sensitivity mechanism entails further computational overhead due to the rejection sampling procedure ([AD20, Algorithm 3]).
this section cite: ['b1']

Section: Pure DP Query Release
We study the private query release problem, where the data universe is defined as X = {0, 1} d , and the dataset is represented as a histogram D ∈ N |X | , satisfying ∥D∥ 1 = n. We consider linear query functions q : X → [0, 1], where q ∈ Q, and the workload Q consists of K distinct queries. For convenience, we define Q(D) := 1
n n i=1 q 1 (d i ), . . . , 1 n n i=1 q K (d i ) .
Our goal is to release a privatized version of Q(D) that minimizes the ℓ ∞ error.
Our algorithm, described in Algorithm 16, proceeds in three steps: (1) the multiplicative weights exponential mechanism [HLM12] is used to release a synthetic dataset; (2) subsampling is applied to this synthetic dataset to further reduce its size; (3) the subsampled dataset is encoded into a binary representation and then passed through the purification procedure (Algorithm 2). The subsampling step is necessary to balance the additional error introduced by purification, which depends on the size of the output space. The privacy and utility guarantees are formalized in Theorem 8, with the proof deferred to Appendix K.2.
Theorem 8 Algorithm 16 satisfies 2ε-DP, and the output D pure of the following utility guarantee:
E[∥Q(D) -Q(D pure )∥ ∞ ] ≤ Õ d 1/3 n 1/3 ε 1/3 . Moreover, the runtime of the algorithm is Õ(nK + |X | + ε 2/3 n 2/3 d 1/3 |X |K).
We emphasize that our algorithmic framework (Algorithm 16) is flexible. The upstream approximate differential privacy query release algorithm-currently instantiated using MWEM [HLM12]-can be replaced with other (ε, δ)-DP algorithms, such as the Private Multiplicative Weights method [HR10] or the Gaussian mechanism. Our approach achieves the tightest known utility upper bound for pure DP, as provided by the Small Database mechanism [BLR13], while offering improved runtime.
this section cite: ['b39', 'b39', 'b40', 'b9']

Section: Purification as a Tool for Proving Lower Bounds
In this section, we demonstrate that the purification technique can be leveraged to establish utility lower bounds for (ε, δ)-differentially private algorithms. Lower bounds for pure differential privacy mechanisms are often more straightforward to derive, commonly relying on packing arguments [HT10,BBKN14]. In contrast, establishing lower bounds for approximate DP mechanisms typically requires more intricate constructions, such as those based on fingerprinting codes [BUV14, DSS + 15, BSU17, LLL21, SU16]. We begin by providing the intuition behind how the purification technique can be used to prove lower bounds for approximate DP. Suppose there exists an (ε, δ)-DP mechanism with δ within appropriate range such that log(1/δ) ∼ Õ(d) and an error bound of O d c log 1-c (1/δ) /nε for some constant c ∈ (0, 1), and we assume the output space is a bounded set in R d . Then, Theorem 1 and Remark 2 imply that it is possible to construct a 2ε-DP mechanism with an error bound of Õ ( d /nε), where Õ omits logarithmic factors. This provides a new approach to proving lower bounds for (ε, δ)-DP algorithms via a contrapositive argument: If no O(ε)-pure-DP mechanism exists with an error bound Õ d nε , then no (ε, δ)-DP mechanism can achieve an error bound of O d c log 1-c (1/δ) /nε for all appropriate δ, for any c ∈ (0, 1).
We apply this approach to derive the lower bound for (ε, δ)-DP mean estimation task. Specifically, let the data universe be D := {-
1 / √ d, 1 / √ d} d
, and consider a dataset
D = {x 1 , . . . , x n } ⊆ {-1 / √ d, 1 / √ d} d
. The objective is to privately release the column-wise empirical mean D =
1 n n i=1 x i ∈ [-1 / √ d, 1 / √ d] d
. For pure DP we have the following lower bound:
Lemma 9 (Lemma 5.1 in [BST14], simplified) For any ε-pure DP mechanism M, there exists a dataset D ∈ {-
1 / √ d, 1 / √ d}
n×d such that with probability at least 1/2 over the randomness of M, we have ∥M(D) -D∥ 2 = Ω d nε . By applying the purification technique with an appropriately chosen δ and a contrapositive argument based on Lemma 9, we obtain the following (ε, δ)-differential privacy lower bound, as stated in Theorem 9. We defer proof to Appendix L.
Theorem 9 Let ε ≤ O(1), and δ ∈ exp(-4d log(d) log 2 (nd)), 4n -d log -2d (8d) . For any
(ε, δ)-DP mechanism M, there exist a dataset D ∈ {-1 / √ d, 1 / √ d}
n×d such that with probability at least 1/4 over the randomness of M, we have
∥M(D) -D∥ 2 = Ω √ d log(1/δ) εn .
Here, Ω(•) hides all polylogarithmic factors, except those with respect to δ.
this section cite: ['b41', 'b5', 'b11']

Section: Remark 10
For mean estimation under (ε, δ)-DP with exponentially small δ, we can also establish a Ω(d) lower bound via a packing argument, as detailed in Appendix L.
this section cite: []

Section: 3.
Additional examples illustrating the use of the purification trick to derive lower bounds is provided in Appendix L.2, which includes bounds for one-way marginal release and private selection.
this section cite: []

Section: Conclusion and Limitation
In this paper, we introduced a novel purification framework that systematically converts approximate DP mechanisms into pure DP mechanisms via randomized post-processing. Our approach bridges the flexibility of approximate DP and the stronger privacy guarantees of pure DP. Our purification technique has broad applicability across several fundamental DP problems. In particular, we propose a faster pure DP algorithm for the DP-ERM problem that achieves near-optimal utility rates for pure DP. We also show that our method can be applied to design pure-DP PTR algorithms. A limitation of our approach is that when applying this continuous purification framework to mechanisms with a finite output space, the utility bound incurs an additional log log |Y| factor compared to prior work, where |Y| is the output space cardinality. Nonetheless, to our knowledge, this is the first systematic purification result applicable to continuous domains. Future work includes extending our framework to more adaptive settings and refining its applicability to high-dimensional problems.
Contents 1 Introduction 1.1 Summary of the Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1.2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2 Algorithm: Converting Approximate DP to Pure DP 3 Technical Lemma: from TV distance to ∞-Wasserstein distance 4 Empirical Risk Minimization with Pure Differential Privacy 4.1 Purified DP Stochastic Gradient Descent . . . . . . . . . . . . . . . . . . . . . . . 4.2 Purified DP Frank-Wolfe Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . 5 Pure DP Data-dependent Mechanisms 5.1 Propose-Test-Release with Pure Differential Privacy . . . . . . . . . . . . . . . . . 5.2 Pure DP Mode Release . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5.3 Pure DP Linear Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 Pure DP Query Release 7 Purification as a Tool for Proving Lower Bounds 8 Conclusion and Limitation A Preliminaries A.1 Definitions on Distributional Discrepancy . . . . . . . . . . . . . . . . . . . . . . A.2 Lemmas for DP Analysis of Algorithm 1 . . . . . . . . . . . . . . . . . . . . . . . B Supplementary Discussion on Randomized Post-Processing C Discussion: Purification on Finite Output Spaces D Discussion: Extending the Purification on Finite Output Spaces to the Euclidean Space E Privacy Analysis: Proof Sketch of Theorem 1 F Deferred Proofs in Section 2 and Appendix A F.1 Proof of Lemma 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.2 Proof of Lemma 15 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.3 Proof of Lemma 16 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . F.4 Proof of Theorem 1 and Corollary 4 . . . . . . . . . . . . . . . . . . . . . . . . . F.5 Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . G Further Discussion of the Optimality of Our Purification Results H Deferred Proofs for DP-SGD H.1 Algorithms and Notations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . H.2 Noisy Gradient Descent Using Laplace Mechanism . . . . . . . . . . . . . . . . . H.3 Analysis of DP-SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . H.3.1 Privacy Accounting Results . . . . . . . . . . . . . . . . . . . . . . . . . H.3.2 Convex and Lipschitz case . . . . . . . . . . . . . . . . . . . . . . . . . . H.3.3 Strongly Convex and Lipschitz case . . . . . . . . . . . . . . . . . . . . . H.4 Analysis of Purified DP-SGD . . . . . . . . . . . . . . . . . . . . . . . . . . . . . H.4.1 Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . I Deferred Proofs for DP-Frank-Wolfe I.1 Approximate DP Frank-Wolfe Algorithm . . . . . . . . . . . . . . . . . . . . . . I.2 Pure DP Frank-Wolfe Algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . I.3 Proof of Theorem 4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . I.4 Proof of Lemma 30 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . J Deferred Proofs for Data-dependent DP mechanism Design J.1 Pure DP Propose Test Release . . . . . . . . . . . . . . . . . . . . . . . . . . . . J.2 Privately Bounding Local Sensitivity . . . . . . . . . . . . . . . . . . . . . . . . . J.3 Private Mode Release . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . J.4 Private Linear Regression Through Adaptive Sufficient Statistics Perturbation . . . K Deferred Proofs for Private Query Release K.1 Problem Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . K.2 Private Multiplicative Weight Exponential Mechanism . . . . . . . . . . . . . . . L Deferred Proofs for Lower Bounds L.1 Proof of Theorem 9 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . L.2 More Examples of Lower Bounds via the Purification Trick . . . . . . . . . . . . . L.2.1 One-Way Marginal Release . . . . . . . . . . . . . . . . . . . . . . . . . L.2.2 Private Selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . L.3 An alternative proof for Theorem 9 . . . . . . . . . . . . . . . . . . . . . . . . . . M Technical Lemmas M.1 Supporting Results on Sparse Recovery . . . . . . . . . . . . . . . . . . . . . . . M.2 A Concentration Inequality for Laplace Random Variables . . . . . . . . . . . . .
this section cite: []

Section: References
Ref_id:b0 Title: Deep learning with differential privacy Year: (2016)
Ref_id:b1 Title: Instance-optimality in differential privacy via approximate inverse sensitivity mechanisms Year: (2020)
Ref_id:b2 Title: Private stochastic convex optimization: Optimal rates in l1 geometry Year: (2021-07)
Ref_id:b3 Title: Comparison of experiments Year: (1951)
Ref_id:b4 Title: Privacy amplification by subsampling: Tight analyses via couplings and divergences Year: (2018)
Ref_id:b5 Title: Bounds on the sample complexity for private learning and private data release Year: (2014)
Ref_id:b6 Title: Composable and versatile privacy via truncated cdp Year: (2018)
Ref_id:b7 Title: Stability is stable: Connections between replicability, privacy, and adaptive generalization Year: (2023)
Ref_id:b8 Title: Non-euclidean differentially private stochastic convex optimization Year: (2021)
Ref_id:b9 Title: A learning theory approach to noninteractive database privacy Year: (2013)
Ref_id:b10 Title: Concentrated differential privacy: Simplifications, extensions, and lower bounds Year: (2016)
Ref_id:b11 Title: Private empirical risk minimization: Efficient algorithms and tight error bounds Year: (2014)
Ref_id:b12 Title: Make up your mind: The price of online queries in differential privacy Year: (2017)
Ref_id:b13 Title: Fingerprinting codes and the price of approximate differential privacy Year: (2014)
Ref_id:b14 Title: Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising Year: (2018)
Ref_id:b15 Title: The large margin mechanism for differentially private maximization Year: (2014)
Ref_id:b16 Title: Differentially private empirical risk minimization Year: (2011)
Ref_id:b17 Title: Decoding by linear programming Year: (2005)
Ref_id:b18 Title: Unlocking high-accuracy differentially private image classification through scale Year: (2022)
Ref_id:b19 Title: Our data, ourselves: Privacy via distributed noise generation Year: (2006)
Ref_id:b20 Title: Differential privacy and robust statistics Year: (2009)
Ref_id:b21 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b22 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b23 Title: An end-to-end differentially private latent Dirichlet allocation using a spectral algorithm Year: (2020-07)
Ref_id:b24 Title: Gaussian differential privacy Year: (2022)
Ref_id:b25 Title: Boosting and differential privacy Year: (2010)
Ref_id:b26 Title: Robust traceability from trace amounts Year: (2015)
Ref_id:b27 Title: Analyze gauss: optimal bounds for privacy-preserving principal component analysis Year: (2014)
Ref_id:b28 Title: On the theory and practice of privacy-preserving bayesian data analysis Year: (2016)
Ref_id:b29 Title: Private stochastic convex optimization: optimal rates in linear time Year: (2020)
Ref_id:b30 Title: Privacy amplification by iteration Year: (2018)
Ref_id:b31 Title: Differentially private empirical risk minimization with input perturbation Year: (2017-10-15)
Ref_id:b32 Title: An algorithm for quadratic programming Year: (1956)
Ref_id:b33 Title: Handbook of convergence theorems for (stochastic) gradient methods Year: (2023)
Ref_id:b34 Title: Faster differentially private convex optimization via second-order methods Year: (2023)
Ref_id:b35 Title: Private convex optimization via exponential mechanism Year: (2022)
Ref_id:b36 Title: A class of wasserstein metrics for probability distributions Year: (1984)
Ref_id:b37 Title: Differentially private sampling from rashomon sets, and the universality of langevin diffusion for convex optimization Year: (2022)
Ref_id:b38 Title: Lemmas of differential privacy Year: (2022)
Ref_id:b39 Title: A simple and practical algorithm for differentially private data release Year: (2012)
Ref_id:b40 Title: A multiplicative weights mechanism for privacypreserving data analysis Year: (2010)
Ref_id:b41 Title: On the geometry of differential privacy Year: (2010)
Ref_id:b42 Title: Towards practical differentially private convex optimization Year: (2019)
Ref_id:b43 Title: Efficient private empirical risk minimization for high-dimensional learning Year: (2016)
Ref_id:b44 Title: Computing tight differential privacy guarantees using fft Year: (2020)
Ref_id:b45 Title: Private non-smooth erm and sco in subquadratic steps Year: (2021)
Ref_id:b46 Title: Improved sample complexity for private nonsmooth nonconvex optimization Year: (2024)
Ref_id:b47 Title: Analyzing graphs with node differential privacy Year: (2013)
Ref_id:b48 Title: A simpler approach to obtaining an o (1/t) convergence rate for the projected stochastic subgradient method Year: (2012)
Ref_id:b49 Title: The power of sampling: Dimension-free risk bounds in private erm Year: (2021)
Ref_id:b50 Title: Tractable MCMC for private learning with pure and gaussian differential privacy Year: (2024)
Ref_id:b51 Title: Gghlite: More efficient multilinear maps from ideal lattices Year: (2014)
Ref_id:b52 Title: Differentially private coordinate descent for composite empirical risk minimization Year: (2022)
Ref_id:b53 Title: Rényi differential privacy Year: (2017)
Ref_id:b54 Title: Multiple source adaptation and the rényi divergence Year: (2012)
Ref_id:b55 Title: Mechanism design via differential privacy Year: (2007)
Ref_id:b56 Title: Sampling from log-concave distributions with infinity-distance guarantees Year: (2022)
Ref_id:b57 Title: Smooth sensitivity and sampling in private data analysis Year: (2007)
Ref_id:b58 Title: Open problem -optimal query release for pure differential privacy Year: (2021)
Ref_id:b59 Title: Learning in a large function space: Privacy-preserving mechanisms for svm learning Year: (2012)
Ref_id:b60 Title: Improving the privacy and practicality of objective perturbation for differentially private linear learners Year: (2023)
Ref_id:b61 Title: Evading the curse of dimensionality in unconstrained private glms Year: (2021)
Ref_id:b62 Title: Between pure and approximate differential privacy Year: (2016)
Ref_id:b63 Title: Pufferfish privacy mechanisms for correlated data Year: (2017)
Ref_id:b64 Title: Nearly optimal private lasso Year: (2015)
Ref_id:b65 Title: Cs395t: Continuous algorithms, part ix, sparse recovery Year: (2024)
Ref_id:b66 Title: Differentially private feature selection via stability arguments, and the robustness of the lasso Year: (2013)
Ref_id:b67 Title: Private empirical risk minimization beyond the worst case: The effect of the constraint set geometry Year: (2014)
Ref_id:b68 Title: The Complexity of Differential Privacy Year: (2017)
Ref_id:b69 Title: Speeding-up linear programming using fast matrix multiplication. In 30th annual symposium on foundations of computer science Year: (1989)
Ref_id:b70 Title: Rényi divergence and kullback-leibler divergence Year: (2014)
Ref_id:b71 Title: Differential privacy for clinical trial data: Preliminary evaluations Year: (2009)
Ref_id:b72 Title: New fast method for generating discrete random numbers with arbitrary frequency distributions Year: (1974)
Ref_id:b73 Title: Revisiting differentially private linear regression: optimal and adaptive prediction & estimation in unbounded domain Year: (2018)
Ref_id:b74 Title: Subsampled rényi differential privacy and analytical moments accountant Year: (2019)
Ref_id:b75 Title: Ece598: Information-theoretic methods in high-dimensional statistics Year: (2016)
Ref_id:b76 Title: Differentially private empirical risk minimization revisited: Faster and more general Year: (2017)
Ref_id:b77 Title: A statistical framework for differential privacy Year: (2010)
Ref_id:b78 Title: Poission subsampled rényi differential privacy Year: (2019)
