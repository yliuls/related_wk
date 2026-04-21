Title: How many measurements are enough? Bayesian recovery in inverse problems with general distributions
Abstract: We study the sample complexity of Bayesian recovery for solving inverse problems with general prior, forward operator and noise distributions. We consider posterior sampling according to an approximate prior P, and establish sufficient conditions for stable and accurate recovery with high probability. Our main result is a non-asymptotic bound that shows that the sample complexity depends on (i) the intrinsic complexity of P, quantified by its approximate covering number, and (ii) concentration bounds for the forward operator and noise distributions. As a key application, we specialize to generative priors, where P is the pushforward of a latent distribution via a Deep Neural Network (DNN). We show that the sample complexity scales log-linearly with the latent dimension k, thus establishing the efficacy of DNN-based priors. Generalizing existing results on deterministic (i.e., non-Bayesian) recovery for the important problem of random sampling with an orthogonal matrix U , we show how the sample complexity is determined by the coherence of U with respect to the support of P. Hence, we establish that coherence plays a fundamental role in Bayesian recovery as well. Overall, our framework unifies and extends prior work, providing rigorous guarantees for the sample complexity of solving Bayesian inverse problems with arbitrary distributions.

Section: Introduction
Inverse problems are of fundamental importance in science, engineering and industry. In a standard setting, the aim is to recover an unknown vector (e.g., an signal or image) x * ∈ R n from measurements y = Ax * + e ∈ R m .
(1.1)
Here e ∈ R m is measurement noise and A ∈ R m×n , often termed the measurement matrix, represents the forwards operator. While simple, the discrete, linear problem (1.1) is sufficient to model many important applications [5,25,64,68]. It is common to solve (1.1) using a Bayesian approach (see, e.g., [30,71]), where one assumes that x * is drawn from some prior distribution R. However, in practice, R is never known exactly. Especially in modern settings that employ Deep Learning (DL) [12,33], it is typical to learn an approximate prior P and then recover x * from y by approximate posterior sampling, i.e., sampling x from the posterior P(•|y, A). An increasingly popular approach involves using generative models to learn P (see, e.g., [12,21,33,68,70,80] and references therein).
A major concern in many inverse problems is that the number of measurements m is highly limited, due to physical constraints such as time (e.g., in Magnetic Resonance Imaging (MRI)), power (e.g., in portable sensors), money (e.g., seismic imaging), radiation exposure (e.g., X-Ray CT), or other factors [5,64,68]. Hence, one aims to recover x * well while keeping the number of measurements m as small as possible. With this in mind, in this work we address the following broad question: How many measurements suffice for stable and accurate recovery of x * ∼ R via approximate posterior sampling x ∼ P(•|y, A), and what are conditions on R, P and the distributions A and E of the measurement matrix A and noise e, respectively, that ensure this recovery?
this section cite: ['b4', 'b24', 'b63', 'b67', 'b29', 'b70', 'b11', 'b32', 'b11', 'b20', 'b32', 'b67', 'b69', 'b4', 'b63', 'b67']

Section: Overview
In this work, we strive to answer to this question in the broadest possible terms, with a theoretical framework that allows for very general types of distributions. We now describe the corresponding conditions needed and present simplified versions of our main results.
(i) Closeness of the real and approximate distributions. We assume that W p (R, P) is small for some 1 ≤ p ≤ ∞, where W p denotes the Wassertstein p-metric.
(ii) Low-complexity of P. Since m ≪ n in many applications, to have any prospect for accurate recovery we need to impose that P (or equivalently R, in view of the previous assumption) has an inherent low complexity. Following [49], we quantify this in terms of its approximate covering number Cov η,δ (P). This is equal to the minimum number of balls of radius η required to cover a region of R n having P-measure at least 1 -δ. See Definition 2.1 for the full definition.
(iii) Concentration of A. We consider constants C low (t) = C low (t; A, D) ≥ 0 and C upp (t) = C upp (t; A, D) ≥ 0 (see Definition 2.3 for the full definition) such that P A∼A [∥Ax∥ ≤ t∥x∥] ≤ C low (t), P A∼A [∥Ax∥ ≥ t∥x∥] ≤ C upp (t) for all x ∈ D := supp(P) -supp(P) and t > 0. Here, and throughout this paper, we write S -S = {x 1 -x 2 : x 1 , x 2 ∈ S} ⊆ R n for the difference set associated with a set S ⊆ R n . We also write supp(P) for the support of the measure P (see §2.1 for the definition). Furthermore, we write ∥•∥ for the ℓ 2 -norm and ∥•∥ ∞ for the ℓ ∞ -norm. If A is isotropic, i.e., E A∼A ∥Ax∥ 2 = ∥x∥ 2 , ∀x ∈ R n , as is often the case in practice, these constants measure how fast ∥Ax∥ 2 concentrates around its mean. Notice that this condition is imposed only on D = supp(P) -supp(P), rather than the whole space R n . As we see later, this is crucial in obtaining meaningful recovery guarantees.
Finally, in order to present a simplified result in this section, we now make several simplifying assumptions. Both of these will be relaxed in our full result, Theorem 3.1.
(iv) Gaussian noise. Specifically, we assume that E = N (0, σ 2 m I) for some σ > 0. (v) Bounded forwards operators. We assume that ∥A∥ ≤ θ a.s. for A ∼ A and some θ > 0. Theorem 1.1 (Simplified main result). Let 1 ≤ p ≤ ∞, 0 < δ ≤ 1/4, ε, η > 0 and suppose that conditions (i)-(v) hold with W p (R, P) ≤ ε/(2mθ) and σ ≥ ε/δ 1/p . Suppose that x * ∼ R, A ∼ A, e ∼ E independently and x ∼ P(•|y, A), where y = Ax * + e. Then, for any d ≥ 2, P ∥x * -x∥ ≥ (8d 2 + 2)(η + σ) ≲ δ + Cov η,δ (P) C low (1/d) + C upp (d) + e -m/16 . (1.2) Note that in this and all subsequent results, the term on the left-hand side of (1.2) is the probability with respect to all variables, i.e., x * ∼ R, A ∼ A, e ∼ E and x ∼ P(•|y, A). Theorem 1.1 is extremely general, in that it allows for essentially arbitrary (real and approximate) signal distributions R and P and an essentially arbitrary distribution A for the forwards operator. In broad terms, it bounds the probability that the error ∥x * -x∥ of posterior sampling exceeds a constant times the noise level σ plus an arbitrary parameter η. It does so in terms of the approximate covering number Cov η,δ (P), which measures the complexity of the approximate distribution P, the concentration bounds C low and C upp for A, which measure how much A elongates or shrinks a fixed vector, and an exponentially-decaying term e -m/16 , which stems from the (Gaussian) noise. In particular, by analyzing these terms for different classes of distributions P and A, we can derive concrete bounds for various exemplar problems. We next describe two such problems.
this section cite: ['b48']

Section: Examples
We first consider A to be a distribution of subgaussian random matrices. Here A ∼ A if its entries are i.i.d. subgaussian random variables with mean zero and variance 1/m (see Definition 3.4). Theorem 1.2 (Subgaussian measurement matrices, simplified). Consider the setup of Theorem 1.1, where A is a distribution of subgaussian random matrices. Then there is a constant c > 0 (depending on the subgaussian parameters β, κ > 0; see Definition 3.4) such that P [∥x * -x∥ ≥ 34(η + σ)] ≲ δ, whenever m ≥ c • [log(Cov η,δ (P)) + log(1/δ)] .
This theorem shows the efficacy of Bayesian recovery with subgaussian random matrices: namely, the sample complexity scales linearly in the distribution complexity, i.e., the log of the approximate covering number. Later in Theorem 3.5, we slightly refine and generalize this result.
Gaussian random matrices are very commonly studied, due to their amenability to analysis and tight theoretical bounds [5,22,34,49], with Theorem 1.2 being a case in point. However, they are largely irrelevant to practical inverse problems, where physical constraints impose certain structures on the forwards operator distribution A [5,64,68]. For instance, in MRI physical constraints mean that the measurements are samples of the Fourier transform of the image. This has motivated researchers to consider much more practically-relevant distributions, in particular, so-called subsampled orthogonal transforms (see, e.g., [5]). Here U ∈ R n×n is a fixed orthogonal matrix -for example, the matrix of the Discrete Fourier Transform (DFT) in the case of MRI -and the distribution A is defined by randomly selecting m rows of U . See Definition 3.7 for the formal definition. Theorem 1.3 (Subsampled orthogonal transforms, simplified). Consider the setup of Theorem 1.1, where A is a distribution of subsampled orthogonal matrices based on a matrix U . Then there is a universal constant c > 0 such that P [∥x * -x∥ ≥ 34(η + σ)] ≲ δ, whenever m ≥ c • µ(U ; D) • [log(Cov η,δ (P)) + log(1/δ)] , where D = supp(P) -supp(P) and µ(U ; D) is the coherence of U relative to D, defined as
µ(U ; D) = n sup ∥U x∥ 2 ∞ /∥x∥ 2 : x ∈ D, x ̸ = 0 .
This result shows that similar stable and accurate recovery to the Gaussian case can be achieved using subsampled orthogonal matrices, provided the number of measurements scales linearly with the coherence. We discuss this term further in §3.2, where we also present the full result, Theorem 3.9.
In general, Theorems 1.2 and 1.3 establish a key condition for successful Bayesian recovery in inverse problems, in each case relating the number of measurements m to the intrinsic complexity log(Cov η,δ (P)) of P. It is therefore informative to see how this complexity behaves for cases of interest. As noted, it is common to use a generative model to learn P. This means that P = G♯γ, where G : R k → R n is a Deep Neural Network (DNN) and γ is some fixed probability measure on the latent space R k . Typically, γ = N (0, I). If G is L-Lipschitz, we show in Proposition 4.1 that log(Cov η,δ (P)) = O(k log[L( √ k + log(1/δ))/η]). (1.3) This scales log-linearly in the latent space dimension k, confirming the intrinsic low-complexity of P. Combining (1.3) with Theorems 1.2-1.3 we see that posterior sampling achieves stable and accurate recovery, provided the number of measurements m scales near-optimally with k, i.e., as O(k log(k)).
Further, in order to compare with deterministic settings such as classical compressed sensing, which concerns the recovery of s-sparse vectors, we also consider distributions P = P s of s-sparse vectors.
In this case, we show in Proposition 4.3 that
log(Cov η,δ (P)) = O(s log(n/s) + log[( √ s + log(1/δ))/η]). (1.4)
Hence s measurements, up to log terms, are sufficient for recovery of approximately sparse vectors. This extends a classical result for deterministic compressed sensing to the Bayesian setting.
this section cite: ['b4', 'b21', 'b33', 'b48', 'b4', 'b63', 'b67', 'b4']

Section: Significance
The significance of this work is as follows. See §1.4 for additional discussion.
1. We provide the first results for Bayesian recovery with arbitrary real and approximate prior distributions R, P and forwards operator and noise distributions A, E.
this section cite: []

Section: 2.
Unlike much of the theory of Bayesian inverse problems, which is asymptotic in nature [12,30,71], our results are non-asymptotic. They hold for arbitrary values of the various parameters within given ranges (e.g., the failure probability δ, the noise level σ, the number of measurements m, and so forth).
this section cite: ['b11', 'b29', 'b70']

Section: 3.
For priors defined by Lipschitz generative DNNs, we establish the first result demonstrating that the sample complexity of Bayesian recovery depends log-linearly on the latent dimension k and logarithmically on the Lipschitz constant.
this section cite: []

Section: 4.
For the important class of subsampled orthogonal transforms, we show that the sample complexity of Bayesian recovery depends on the coherence, thus resolving several key open problems in the literature (see next).
this section cite: []

Section: 5.
It is increasingly well-known that DL-based methods for inverse problems are susceptible to hallucinations and other undesirable effects [11, 15, 18, 25, 29, 40, 42, 45, 47, 60-63, 65, 66, 79]. This is a major issue that may limit the uptake of these methods in safety-critical domains such as medical imaging [25,55,57,61,74,76,77]. Our results provide theoretical guarantees for stable and accurate, and therefore show conditions under which hallucinations provably cannot occur. This is not only theoretically interesting, but it also has practical consequences in the development of robust DL methods for inverse problems -a topic we intend to explore in future work.
this section cite: ['b24', 'b54', 'b56', 'b60', 'b73', 'b75', 'b76']

Section: Related work
Bayesian methods for inverse problems have become increasingly popular over the last several decades [30,71], and many state-of-the-art DL methods for inverse problems now follow a Bayesian approach (see [7,12,21,46,56,64] and references therein). Learned priors, such as those stemming from generative models, are now also increasingly used in applications [1,7,12,21,28,33,38,46,48,51,53,54,58,64,68,70,78,80].
This work is motivated in part by the (non-Bayesian) theory of generative models for solving inverse problems (see [22,33,43,44,48,[68][69][70]80] and references therein). This was first developed in [22], where compressed sensing techniques were used to show recovery guarantees for a Gaussian random matrix A when computing an approximate solution to (1.1) in the range Σ := ran(G) of a Lipschitz map G : R k → R n , typically assumed to be a generative DNN. This is a deterministic approach. Besides the random forward operator and (potentially) random noise, it recovers a fixed (i.e., nonrandom) underlying signal x * in a deterministic fashion with a point estimator x that is obtained as a minimizer of the empirical ℓ 2 -loss min z∈Σ ∥Az -y∥ 2 . In particular, no information about the latent space distribution γ is used. In this work, following, e.g., [1,21,48,49,70] and others, we consider a Bayesian setting, where x * ∼ R is random and where we quantify the number of measurements that suffice for accurate and stable recovery via posterior sampling x ∼ P(•|y, A).
Our work is a generalization of [49], which considered Bayesian recovery with Gaussian random matrices and standard Gaussian noise. We significantly extend [49] to allow for arbitrary distributions A and E for the forward operator and noise, respectively. A key technical step in doing this is the introduction of the concentration bounds C low and C upp . In particular, these bounds are imposed only over the subset D = supp(P) -supp(P). This is unnecessary in the Gaussian case considered in Theorem 1.2, but crucial to obtain meaningful bounds in, for instance, the case of subsampled orthogonal transforms considered in Theorem 1.3 (see Remarks 3.2 and 3.10 for further discussion). As noted, this case is very relevant to applications. In particular, when U is taken as a DFT matrix our work addresses open problems posed in [48, §3] and [68, §II.F] on recovery guarantees with Fourier measurements. We also derive bounds for the approximate covering number of distributions given by Lipschitz generative DNNs (see (1.3) and Proposition 4.1) and distributions of sparse vectors (see (1.4) and Proposition 4.3), addressing an open problem posed in [49, §6]. In particular, we demonstrate stable and accurate recovery in a Bayesian sense with a number of measurements that scales linearly in the model complexity, i.e., k in the former case and s in the latter case.
Recently [16,17] generalized the results of [22] in the non-Bayesian setting from Gaussian random matrices to subsampled orthogonal transforms. Theorem 1.3 provides a Bayesian analogue of this work, as discussed above, where we consider posterior sampling rather than a deterministic point estimator. We also extend the setup of [16,17] by allowing for general measurement distributions A and priors P. In particular, in [16,17] the quantity Σ, which is the deterministic analogue of the prior distribution P in our work, was assumed to be the range of a ReLU generative DNN. Like in [16], we make use of the concept of coherence (see Theorem 1.3). However, our proof techniques are completely different to those of [16,17].
Classical compressed sensing considers the recovery of approximately s-sparse vectors from (1.1). However, it has been extended to consider much more general types of low-complexity signal models, such as joint or block sparse vectors, tree sparse vectors, cosparse vectors and many others [4,6,13,23,31,36,73]. However, most recovery guarantees for general model classes consider only (sub)Gaussian measurement matrices (see e.g., [13,34]). Recently, [3] introduced a general framework for compressed sensing that allows for general low-complexity models Σ ⊆ R n contained within a finite union of finite-dimensional subspaces and arbitrary (random) measurement matrices. Our work is a Bayesian analogue of this deterministic setting. Similar to [3], a key feature of our work is that we consider arbitrary real and approximate signal distributions P, R (analogous to arbitrary low-complexity models Σ) and arbitrary distributions A for the forwards operator. Unsurprisingly, a number of the conditions in our main result, Theorem 1.1 -namely, the low-complexity condition (ii) and concentration condition (iii) -share similarities to those that ensure stable and accurate recovery in non-Bayesian compressed sensing. See Remarks 2.2 and 3.3 for further discussion. However, the proof techniques used in this work are once more entirely different.
Finally, while the focus of this work is to establish guarantees for posterior sampling, we mention in passing related work on information-theoretically optimal recovery methods. Methods such as Approximate Message Passing (AMP) [14,35] are well studied, and asymptotic information-theoretic bounds for Gaussian random matrices are known [52]. AMP methods are fast point estimation algorithms, whereas we focus on sampling-based methods and do not consider computational implementations (see §5 for some further discussion). Note that information-theoretic lower bounds for posterior sampling have also been shown for the Gaussian case in [49].
this section cite: ['b29', 'b70', 'b6', 'b11', 'b20', 'b45', 'b55', 'b63', 'b0', 'b6', 'b11', 'b20', 'b27', 'b32', 'b37', 'b45', 'b47', 'b50', 'b52', 'b53', 'b57', 'b63', 'b67', 'b69', 'b77', 'b21', 'b32', 'b42', 'b43', 'b47', 'b67', 'b68', 'b69', 'b21', 'b0', 'b20', 'b47', 'b48', 'b69', 'b48', 'b48', 'b15', 'b16', 'b21', 'b15', 'b16', 'b15', 'b16', 'b15', 'b15', 'b16', 'b3', 'b5', 'b12', 'b22', 'b30', 'b35', 'b72', 'b12', 'b33', 'b2', 'b2', 'b13', 'b34', 'b51', 'b48']

Section: Preliminaries

this section cite: []

Section: Notation
We now introduce some further notation. We let B r (x) = {z ∈ R n : ∥z -x∥ ≤ r} and, when x = 0, we write B r := B r (0). Given a set X ⊆ R n , we write X c = R n \X for its complement. We also write B r (X) = x∈X B r (x) for the r-neighbourhood of X.
Let (X, F, µ) be a Borel probability space. We write supp(µ) for its support, i.e., the smallest closed set A ⊆ X for which µ(A) = 1. Given probability spaces (X, F 1 , µ), (Y, F 2 , ν), we write Γ = Γ µ,ν for the set of couplings, i.e., probability measures on the product space (X × Y, σ(F 1 ⊗ F 2 )) whose marginals are µ and ν, respectively. Given a cost function c : X × Y → [0, ∞) and 1 ≤ p < ∞, the Wasserstein-p metric is defined as
W p (µ, ν) = inf γ∈Γ X×Y c(x, y) p dγ(x, y) 1/p .
If p = ∞, then W ∞ (µ, ν) = inf γ∈Γ (esssup γ c(x, y)). In this paper, unless stated otherwise, X = Y = R n and the cost function c is the Euclidean distance.
this section cite: []

Section: Approximate covering numbers
As a measure of complexity of measures, we use the concept of approximate covering numbers as introduced in [49]. Definition 2.1 (Approximate covering number). Let (X, F, P) be a probability space and δ, η ≥ 0.
The η, δ-approximate covering number of P is defined as
Cov η,δ (P) = min k ∈ N : ∃{x i } k i=1 ⊆ supp(P), P k i=1 B η (x i ) ≥ 1 -δ .
This quantity measures how many balls of radius η are required to cover at least 1 -δ of the Pmass of R n . See [49] for further discussion. Note that [49] does not require the centres x i of the approximate cover belong to supp(P). However, this is useful in our more general setting and presents no substantial restriction. At worst, this requirement changes η by a factor of 1/2.
Remark 2.2 (Relation to non-Bayesian compressed sensing) Note that when δ = 0, the approximate covering number Cov η,0 (P) ≡ Cov η (supp(P)) is just the classical covering number of the set supp(P), i.e., the minimal number of balls of radius η that cover supp(P). Classical covering numbers play a key role in (non-Bayesian) compressed sensing theory. Namely, the covering number of (the unit ball of) the model class Σ ⊆ R n directly determines the number of measurements that suffice for stable and accurate recovery. See, e.g., [3,34]. In the Bayesian setting, the approximate covering number plays the same role; see Theorem 1.1.
this section cite: ['b48', 'b48', 'b48', 'b2', 'b33']

Section: Bounds for A and E
Since our objective is to establish results that hold for arbitrary measurement and noise distributions A and E, we require several key definitions. These are variety of (concentration) bounds.
this section cite: []

Section: Definition 2.3 (Concentration bounds for A).
Let A be a distribution on R m×n , t ≥ 0 and D ⊆ R n . Then a lower concentration bound for A is any constant C low (t) = C low (t; A, D) ≥ 0 such that P A∼A {∥Ax∥ ≤ t∥x∥} ≤ C low (t; A, D), ∀x ∈ D.
Similarly, an upper concentration bound for A is any constant C upp (t) = C upp (t; A, D) ≥ 0 such that P A∼A {∥Ax∥ ≥ t∥x∥} ≤ C upp (t; A, D), ∀x ∈ D. Finally, given t, s ≥ 0 an (upper) absolute concentration bound for A is any constant C abs (s, t; A, D) such that P A∼A (∥Ax∥ > t) ≤ C abs (s, t; A, D), ∀x ∈ D, ∥x∥ ≤ s.
Notice that if A is isotropic, i.e., E∥Ax∥ 2 = ∥x∥ 2 , ∀x ∈ R n , then C low and C upp determine how well ∥Ax∥ 2 concentrates around its mean ∥x∥ 2 for any fixed x ∈ D. To obtain desirable sample complexity estimates (e.g., Theorems 1.2 and 1.3), we need concentration bounds that decay exponentially in m. A crucial component of this analysis is considering concentration bounds over some subset D (related to the support of P), as, in general, one cannot expect fast concentration over the whole of R n . See Remarks 3.2 and 3.10.
this section cite: []

Section: Definition 2.4 (Concentration bound for E).
Let E be a distribution in R m and t ≥ 0. Then an (upper) concentration bound for E is any constant
D upp (t) = D upp (t; E) ≥ 0 such that E(B c t ) = P e∼E (∥e∥ ≥ t) ≤ D upp (t; E).
Notice that this bound just measures the probability that the noise is large. We also need the following concept, which estimates how much the density of E changes in a τ -neighbourhood of the origin when perturbed by an amount ε.
Definition 2.5 (Density shift bounds for E). Let E be a distribution in R m with density p E and ε, τ ≥ 0. Then a density shift bound for E is any constant D shift (ε, τ ) = D shift (ε, τ ; E) ≥ 0 (possibly +∞) such that p E (u) ≤ D shift (ε, τ ; E)p E (v), ∀u, v ∈ R n , ∥u∥ ≤ τ, ∥u -v∥ ≤ ε.
this section cite: []

Section: Main results
We now present our main results. The first, an extension of Theorem 1.1, is a general result that holds for arbitrary distributions R, P, A and E.
Theorem 3.1. Let 1 ≤ p ≤ ∞, 0 ≤ δ ≤ 1/4, ε, η, t > 0, c, c ′ ≥ 1 and σ ≥ ε/δ 1/p . Let E be a distribution on R m and R, P be distributions on R n satisfying W p (R, P) ≤ ε and min(log Cov η,δ (R), log Cov η,δ (P)) ≤ k (3.1) for some k ∈ N. Suppose that x * ∼ R, A ∼ A, e ∼ E independently and x ∼ P(•|y, A), where y = Ax * + e. Then p := P[∥x * -x∥ ≥ (c + 2)(η + σ)] satisfies p ≤ 2δ + C abs (ε/δ 1/p , tε/δ 1/p ; A, D 1 ) + D upp (c ′ σ; E) + 2D shift (tε/δ 1/p , c ′ σ; E)e k C low 2 √ 2 √ c ; A, D 2 + C upp √ c 2 √ 2 ; A, D 2 + 2D upp √ cσ 2 √ 2 ; E , where D 1 = B ε/δ 1/p (supp(P)) ∩ supp(R) -supp(P) (3.2) and D 2 = supp(P) -supp(P) if P attains the minimum in (3.1) supp(P) -supp(R) otherwise . (3.3)
This theorem bounds the probability p of unstable or inaccurate recovery in terms of the various parameters using the constants introduced in the previous section and the approximate covering numbers of R, P. This result is powerful in its generality, but as a consequence, rather opaque. In particular, since it considers arbitrary measurement and noise distributions, the number of measurements m does not explicitly enter the bound. For typical distributions, a dependence on m is found in the concentration bounds C low , C upp , D upp , as well as the terms D shift and C abs . For instance, the former decay exponentially-fast in m for the examples introduced in §1.2, and therefore compensate for the exponentially-large scaling in k in the main bound (see §B for precise estimates, as well as the discussion in §3.1-3.2). Note that Theorem 3.1 also considers general noise distributions E. While Gaussian noise is arguably the most important example -and will be used in all our subsequent examples -this additional generality comes at little cost in terms of the technicality of the proofs.
this section cite: []

Section: Remark 3.2 (The concentration bounds in Theorem 3.1)
A particularly important facet of this result, for the reasons discussed above, is that the various concentration bounds C abs , C low and C upp are taken over sets D 1 , D 2 -given by (3.2) and ( 3.3), respectively, and related to the support of P and R -rather than the whole space R n . We exploit this fact crucially later in Theorem 3.9.
this section cite: []

Section: Remark 3.3 (Relation to non-Bayesian compressed sensing)
The constants C low and C upp are similar, albeit not identical to similar conditions such as the Restricted Isometry Property (RIP) (see, e.g., [5, Chpt. 5]) or Restricted Eigenvalue Condition (REC) [19,22] that appear in non-Bayesian compressed sensing. There, one considers a fixed model class Σ ⊆ R n , such as the set Σ s of s-sparse vectors or, as in [22], the range ran(G) of a generative DNN. Conditions such as the RIP or REC impose that ∥Ax∥ is concentrated around ∥x∥ for all x belonging to the difference set Σ -Σ. In Theorem 3.1, assuming P attains the minimum in (3.1), there is a similar condition with Σ replaced by supp(P). Indeed, C low (2 √ 2/ √ c; A, D 2 ) measures how small ∥Ax∥ is in relation to ∥x∥ and C upp ( √ c/(2 √ 2); A, D 2 ) measures how large ∥Ax∥ is in relation to ∥x∥.
this section cite: ['b18', 'b21', 'b21']

Section: Example: Subgaussian random matrices with Gaussian noise
We now apply this theorem to the first example introduced in §1.2. Recall that a random variable X on R is subgaussian with parameters β, κ > 0 if P(|X| ≥ t) ≤ βe -κt 2 for all t > 0.
Definition 3.4 (Subgaussian random matrix). A random matrix A ∈ R m×n is subgaussian with parameters β, κ > 0 if A = 1 √ m A, where the entries of A are independent mean-zero sugaussian random variables with variance 1 and the same subgaussian parameters β, κ.
Note that 1/ √ m is a scaling factor that ensures that A is isotropic, i.e., E∥Ax∥ 2 = ∥x∥ 2 , ∀x ∈ R n .
Theorem 3.5. Let 1 ≤ p ≤ ∞, 0 ≤ δ ≤ 1/4, ε, η > 0 and σ ≥ ε/δ 1/p . Let E = N (0, σ 2 m I) and A be a distribution of subgaussian random matrices with parameters β, κ > 0. Let R, P be distributions on R n and suppose that x * ∼ R, A ∼ A, e ∼ E independently and x ∼ P(•|y, A), where y = Ax * + e. Then there is a constant c(β, κ) > 0 depending on β, κ only such that
P[∥x * -x∥ ≥ 34(η + σ)] ≲ δ, provided W p (R, P) ≤ ε/c(β, κ) and m ≥ c(β, κ) • [min(log Cov η,δ (R), log Cov η,δ (P)) + log(1/δ)] .
(3.4) This theorem is derived from Theorem 3.1 by showing that the various concentration bounds are exponentially small in m for subgaussian random matrices (see §B). It is a direct generalization of [49], which considered the Gaussian case only. It shows that subgaussian random matrices are near-optimal for Bayesian recovery, in the sense that m scales linearly with the log of the approximate covering number (3.4). We estimate these covering numbers for several key cases in §4.
It is worth at this stage discussing how (3.4) behaves with respect to the various parameters. First, suppose that η decreases so that the error bound becomes smaller. Then Cov η,δ (•) increases, meaning, as expected, that more measurements are required to meet (3.4). Second, suppose that δ decreases, so that the failure probability shrinks. Then Cov η,δ (•) and log(1/δ) both increase, meaning, once again, that more measurements are needed for (3.4) to hold. Both behaviours are as expected.
Remark 3.6 (Relation to Johnson-Lindenstrauss) Suppose that P = R is a sum of d Diracs located at X = {x 1 , . . . , x d } ⊆ R n . Since the matrix A ∈ R m×n is a linear dimensionality-reducing map, the Johnson-Lindenstrauss Lemma states that distances in X are preserved under A if and only if m = O(log(d)). In this setting, preserving the distances in X is equivalent to being able to stably identify the mode from which a signal is drawn when observing its measurements. In agreement with this argument, Theorem 3.5 also predicts recovery from roughly m = O(log(d)) measurements.
this section cite: ['b48']

Section: Example: Randomly-subsampled orthogonal transforms with Gaussian noise
As discussed, subgaussian random matrices are largely impractical. We now consider the more practical case of subsampled orthogonal transforms. Definition 3.7 (Randomly-subsampled orthogonal transform). Let U ∈ R n×n be orthogonal (i.e., U ⊤ U = U U ⊤ = I) and write u 1 , . . . , u n ∈ R n for its rows. Let X 1 , . . . , X n ∼ i.i.d. Ber(m/n) be independent Bernoulli random variables with P(X i = 1) = m/n and P(X i = 0) = 1 -m/n. Then we define a distribution A as follows. We say that A ∼ A if
A = n m    u ⊤ i1 . . . u ⊤ iq    , where {i 1 , . . . , i q } ⊆ {1, . . . , n} is the set of indices i for which X i = 1.
The factor n/m ensures that E(A ⊤ A) = I. Note that the number of measurements q in this model is itself a random variable, with E(q) = m. However, q concentrates exponentially around its mean.
Definition 3.8. Let U ∈ R n×n and D ⊆ R n . The coherence of U relative to D is
µ(U ; D) = n • sup ∥U x∥ 2 ∞ /∥x∥ 2 : x ∈ D, x ̸ = 0 .
Coherence is a well-known concept in classical compressed sensing with sparse vectors. Definition 3.8 is a generalization that allows for arbitrary model classes D. This definition is similar to that of [16], which considered non-Bayesian compressed sensing with generative models. It is also related to the more general concept of variation introduced in [3]. (3.5)
This theorem is a Bayesian analogue of the deterministic results shown in [3,16]. In [3,16], the measurement conditions scale linearly with µ(U ; D), where D = Σ -Σ and Σ ⊆ R n is the lowcomplexity model class. Similarly, the number of measurements (3.5) scales linearly with respect to the coherence relative to D = supp(P) -supp(P), which, as discussed in Remark 3.3, plays the role of the low-complexity model class in the Bayesian setting. Note that the measurement condition (3.5) involves the approximate distribution P only. This is relevant, since the quantities µ(U ; D) and Cov η,δ (P) can be estimated either numerically or analytically in various cases, such as when P is given by a generative model. Indeed, we estimate Cov η,δ (P) analytically for Lipschitz generative models in Proposition 4.1 below. The coherence µ(U ; D) was estimated analytically in [16] for ReLU DNNs with random weights (see Remark 4.2 below). It can also be estimated numerically for more general types of generative models [2,16]. Overall, by estimating these quantities, one can use (3.5) to gauge how well one can recover with a given P. Note that this may not be possible if (3.5) involved R as well, since this distribution is typically unknown.
In classical compressed sensing, coherence determines the sample complexity of recovering sparse vectors from randomly-subsampled orthogonal transforms [26]. A similar argument can be made in the Bayesian setting. Notice that µ(U ; D) ≤ µ(U ; R n ) = n. However, we are particularly interested in cases where µ(U ; D) ≪ n, in which case (3.5) may be significantly smaller than the ambient dimension n. We discuss this in the context of several examples in the next section.
Remark 3.10 (Concentration over subsets) Theorem 3.9 illustrates why it is important that Theorem 3.1 involves concentration bounds over subsets of R n . To derive Theorem 3.9 from Theorem 3.1 (see §B), we show exponentially-fast concentration in m/µ(U ; D). Had we considered the whole of R n , then, since µ(U ; R n ) = n, this would have lead to an undesirable measurement condition of the form m = O(n) scaling linearly in the ambient dimension n.
this section cite: ['b15', 'b2', 'b2', 'b15', 'b2', 'b15', 'b15', 'b1', 'b15', 'b25']

Section: Covering number and sample complexity estimates
We conclude by applying our results to two different types of approximate prior distributions.
this section cite: []

Section: Generative DNNs Proposition (Approximate covering number for a Lipschitz pushforward of a Gaussian measure).
Let G : R k → R n be Lipschitz with constant L ≥ 0, i.e., ∥G(x) -G(z)∥ ≤ L∥x -z∥, ∀x, z ∈ R k , and define P = G♯γ, where γ = N (0, I) is the standard normal distribution on R k . Then
log(Cov η,δ (P)) ≤ k log 1 + 2 √ kL η 1 + 2 k log(1/δ) . (4.1)
This result shows that P has low complexity, since log(Cov η,δ (P)) scales log-linearly in k. Combined with Theorem 3.5, it shows that accurate and stable Bayesian recovery with such a prior with a sample complexity that is near-optimal in the latent dimension k, i.e., O(k log(k)).
Notice that L only appears logarithmically in (4.1). While it is not the main focus of this work, we note that Lipschitz constants of DNNs have been studied quite extensively [37,72,75]. Moreover, it is also possible to design and train DNNs with small Lipschitz constants [59].
this section cite: ['b36', 'b71', 'b74', 'b58']

Section: Remark 4.2 (Quadratic bottleneck and high coherence)
In Theorem 3.9, the measurement condition (3.5) also depends on the coherence. This quantity has been considered in [16] for the case of ReLU DNNs. In particular, if a ReLU DNN G : R k → R n has random weights drawn from a standard normal distribution, then its coherence µ(U ; D) scales like O(k) up to log factors [16,Thm. 3]. Combining this with Theorem 3.9 and Proposition 4.1, we see that the overall sample complexity for Bayesian recovery scales like O(k 2 log(k)) in this case. This is worse than the subgaussian case, where there is no coherence factor and the sample complexity, as noted above, is O(k log(k)). Such a quadratic bottleneck also arises in the non-Bayesian setting [16]. Its removal is an open problem (see §5). Note that the coherence is also not guaranteed to be small for general (in particular, trained) DNNs. However, [16] also discuss strategies for training generative models to have small coherence. Numerically, they show that generative models with smaller coherence achieve better recovery from the same number of measurements than those with larger coherence.
this section cite: ['b15', 'b15', 'b15', 'b15']

Section: Distributions of sparse vectors
Let s ∈ {1, . . . , n}. We define a distribution P = P s of s-sparse vectors in R n as follows. To draw x ∼ P, we first choose a support set S ⊆ {1, . . . , n}, |S| = s, uniformly at random amongst all possible n s such subsets. We then define x i = 0, i / ∈ S, and for each i ∈ S we draw x i randomly and independently from N (0, 1). Note that there are other ways to define distributions of sparse vectors, which can be analyzed similarly. However, for brevity we only consider the above setup. Proposition 4.3 (Approximate covering number for distributions of sparse vectors). Let P = P s be a distribution of s-sparse vectors in R n . Then
log(Cov η,δ (P s )) ≤ s log en s + log 1 + 2 √ s η 1 + 2 s log(1/δ) . (4.2)
As in the previous case, we deduce Bayesian recovery from O(s log(n/s)) subgaussian measurements, i.e., near-optimal, log-linear sample complexity. In the case of randomly-subsampled orthogonal matrices, we also need to consider the coherence. For P = P s as above, one can easily show that
µ(U ; supp(P s ) -supp(P s )) ≤ 2sµ * (U ), where µ * (U ) = n • max i,j |u ij | 2 . (4.3)
The term µ * (U ) is often referred to as the coherence of U (see, e.g., [5, Defn. 5.8] or [26]). Notice that µ * (U ) ≈ 1 for DFT matrices, which is one reason why subsampled Fourier transforms are particularly effective in (non-Bayesian) compressed sensing. Our work implies as similar conclusion in the Bayesian setting: indeed, substituting (4.2) and (4.3) into (3.5) we immediately deduce that the measurement condition for Bayesian recovery with P s behaves like m = O(s 2 ) up to log terms.
this section cite: ['b25']

Section: Remark 4.4 (Quadratic bottleneck)
Once more we witness a quadratic bottleneck. In the non-Bayesian setting, one can show stable and accurate recovery of sparse vectors from O(s) measurements, up to log terms (see, e.g., [5, Cor. 13.15]). However, this requires specialized theoretical techniques that heavily leverage the structure of the set of sparse vectors. In the setting of this paper, the bottleneck arises from the generality of the approach considered in this work: specifically, the fact that our main results hold for arbitrary probability distributions P.
this section cite: []

Section: Limitations and future work
We end by discussing a number of limitations and avenues for future work. First, although our main result Theorem 3.1 is very general, we have only applied it to a number of different cases, such as Lipschitz pushforward measures and Gaussian random matrices or subsampled orthogonal transforms. We believe many other important problems can be studied as corollaries of our main results. This includes sampling with heavy-tailed vectors [50], sampling with random convolutions [67], multi-sensor acquisition problems [27], generative models augmented with sparse deviations [32], block sampling [3,20,24], with applications to practical MRI acquisition, sparse tomography [8], deconvolution and inverse source problems [9]. We believe our framework can also be applied to various types of non-Gaussian noise, as well as problems involving sparsely-corrupted measurements [50]. We are actively investigating applying our framework to these problems.
Second, as noted in Remarks 4.2 and 4.4 there is a quadratic bottleneck when considering subsampled orthogonal transforms. In the non-Bayesian case, this can be overcome in the case of (structured) sparse models using more technical arguments [3]. We believe similar ideas could also be exploited in the Bayesian setting. On a related note, both [16,22] consider ReLU generative models in the non-Bayesian setting, and derive measurement conditions that do not involve the Lipschitz constant L of the DNN. It is unclear whether analogous results can be established in the Bayesian setting.
Third, our main result involves the density shift bound (Definition 2.5). In particular, the noise distribution should have a density. This rather unpleasant technical assumption stems from Lemma C.6, which is a key step in proving the main result, Theorem 3.1. This lemma allows one to replace the 'real' distribution R in the probability term p in Theorem 3.1 by the approximate distribution P. This is done in order to align the prior and the posterior, which is necessary for the subsequent steps of the proof of Theorem 3.1. It would be interesting to see if this assumption on the noise could be removed through a refined analysis.
Fourth, as noted in [16], the coherence µ(U ; D) arising in Theorem 3.9 may be high. In the non-Bayesian setting, this has been addressed in [2,3,17] by using a nonuniform probability distribution for drawing rows of U , with probabilities given in terms of so-called local coherences of U with relative to D. As shown therein, this can lead to significant performance gains over sampling uniformly at random. We believe a similar approach can be considered in the Bayesian setting as a consequence of our general framework. We intend to explore this in future work.
Finally, our results in this paper are theoretical, and strive to study the sample complexity of Bayesian recovery. We do not address the practical problem of sampling from the posterior. This is a key computational challenge in Bayesian inverse problems [12]. However, efficient techniques for doing this are emerging. See, e.g., [48,54,70] and references therein. We believe an advantage of our results is their independence from the choice of posterior sampling algorithm, whose analysis can therefore be performed separately. This is an interesting problem we intend to examine in the future. In future work we also intend present numerical experiments for various practical settings that further support the theory developed in this paper.
[79] C. Zhang, J. Jia, B. Yaman, S. Moeller, S. Liu, M. Hong, and M. Akçakaya. Instabilities in conventional multi-coil MRI reconstruction with small adversarial perturbations. In 2021 55th Asilomar Conference on Signals, Systems, and Computers, pages 895-899, 2021.
[80] Z. Zhao, J. C. Ye, and Y. Bresler. Generative models for inverse imaging problems: from mathematical foundations to physics-driven applications. IEEE Signal Process. Mag., 40(1):148-163, 2023.
this section cite: ['b49', 'b66', 'b26', 'b31', 'b2', 'b19', 'b23', 'b7', 'b8', 'b49', 'b2', 'b15', 'b21', 'b15', 'b1', 'b2', 'b16', 'b11', 'b47', 'b53', 'b69']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We thoroughly discuss the main claims made in the abstract and introduction and the necessary assumptions to show them. Our main theoretical contributions directly address these claims. We also have several further remarks after these results to provide additional context for our work.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss limitations at the end of the paper in §5.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: We have a detailed discussion of the assumptions needed to show our theoretical results in §2. We provide further discussion and present the results themselves in §3-4. We provide full proofs of our results in the supplemental material. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: There are no experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: There are no experiments in the paper. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so No is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: There are no experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: There are no experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [NA] Justification: There are no experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We have complied with the NeurIPS Code of Ethics in the preparation of this manuscript.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This work is primarily foundational, and the examples considered do not directly impact society.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: This paper is theoretical, and does not involve any data or models.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: This paper is theoretical, and does not use any existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: This paper is theoretical, and does not introduce any new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No experiments were conducted in this paper. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No experiments were conducted in this paper. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this paper does not involve LLMs as an important, original, or non-standard component. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: A Covering number estimates and the proofs of Propositions 4.1 and 4.3
The proof of Proposition 4.1 relies on the following two lemmas. Lemma A.1 (Approximate covering number under a Lipschitz pushforward map). Let G : R k → R n be Lipschitz with constant L ≥ 0, i.e., ∥G(x) -G(z)∥ ≤ L∥x -z∥, ∀x, z ∈ R k , and define P = G♯γ, where γ is any probability distribution on R k . Then Cov η,δ (P) ≤ Cov η/L,δ (γ), ∀δ, η ≥ 0.
Proof. Let {x i } k i=1 ⊆ supp(γ) and define
z i = G(x i ) ∈ supp(G♯γ) for i = 1, . . . , k. Let z ∈ G(B η/L (x i )) and write z = G(x) for some x ∈ B η/L (x i ). Then ∥z -z i ∥ = ∥G(x) -G(x i )∥ ≤ L∥x -x i ∥ ≤ η. Hence z ∈ B η (z i ). Since z was arbitrary, we deduce that G(B η/L (x i )) ⊆ B η (z i ). It follows that B η/L (x i ) ⊆ G -1 (B η (z i )). Now suppose that γ k i=1 B η/L (x i ) ≥ 1 -δ.
Then, by definition of the pushforward measure
G♯γ k i=1 B η (z i ) = γ k i=1 G -1 (B η (z i )) ≥ γ k i=1 B η/L (x i ) ≥ 1 -δ.
This gives the result. Proof. Observe that, for t ≥ 0,
P(B c √ nσt ) = P(X ≥ nt 2 ) ≤ (t 2 e 1-t 2 ) n/2 .
where X ∼ χ 2 n is a chi-squared random variable, and the inequality follows from a standard Chernoff bound. Now t 2 ≤ e 2t gives that
P(B c √ nσt ) ≤ (e -(t-1) 2 ) n/2 . Now set t = 1 + 2 n log(1/δ) so that P(B c √ nσt ) ≤ δ.
Hence, we have shown that Cov η,δ (P) ≤ Cov η (B √ nσt ), where Cov η is the classical covering number of a set, i.e.,
Cov η (A) = min k : ∃{x i } k i=1 ⊆ A, A ⊆ k i=1 B η (x i ) .
Using standard properties of covering numbers (see, e.g., [5, Lem. 13.22], we get Proof. For each i = 1, . . . , r, let {x (i) j } ki j=1 ⊆ R n , and, in particular, 3x
Cov η (B √ nσt ) = Cov η/(√
(i) j ∈ supp(P i ), be such that
P i   ki j=1 B η (x (i) j )   ≥ 1 -δ. Then P   r i=1 ki j=1 B η (x (i) j )   = r i=1 p i P i   r i=1 ki j=1 B η (x (i) j )   ≥ r i=1 p i P i   ki j=1 B η (x (i) j )   ≥ r i=1 p i (1 -δ) = 1 -δ.
Notice that supp(P i ) ⊆ supp(P), therefore, x (i) j ∈ supp(P). The result now follows.
Proof of Proposition 4.3. Let S = {S : S ⊆ {1, . . . , n}, |S| = s}. Then we can write P s as the mixture
P s = |S| i=1 1 |S| P S ,
where P S is defined as follows: x ∼ P S if x i = 0 for i / ∈ S and, for i ∈ S, x i is drawn independently from the standard normal distribution on R. Notice that P S = G♯γ, where γ = N (0, I) is the standard, multivariate normal distribution on R s and G : R s → R n is a zero-padding map. The map G is Lipschitz with constant L = 1. Hence, by Lemmas A. 1 and A.2, Cov η,δ (P S ) ≤ 1 + 2 √ st η s , t = 1 + 2 s log(1/δ) We now apply Lemma A.3 and the fact that |S| = n s ≤ en s s , the latter being a standard bound, to obtain Cov η,δ (P s ) ≤ en s s 1 + 2 √ st η s , t = 1 + 2 s log(1/δ).
Taking logarithms gives the result.
this section cite: ['b0']

Section: B Concentration inequalities and the proofs of Theorems 3.5 and 3.9
We now aim to prove Theorems 3.5 and 3.9. To do this, we first derive concentration inequalities for subsgaussian random matrices and subsampled orthogonal transforms.
this section cite: []

Section: B.1 Gaussian concentration and density shift bounds
Lemma B.1 (Concentration and density shift bounds for Gaussian noise). Let E = N (0, σ 2 m I). Then the upper concentration bound D upp (t; E) (Definition 2.4) can be taken as
D upp (t; E) = t 2 σ 2 e 1-t 2 σ 2 m/2 , ∀t > σ.
and the density shift bound D shift (ε, τ ; E) (Definition 2.5) can be taken as
D shift (ε, τ ; E) = exp m(2τ + ε) 2σ 2 ε , ∀ε, τ ≥ 0.
Proof. Write e ∼ E as e = σ √ m n, where n ∼ N (0, I). Then
P(∥e∥ ≥ t) = P(∥n∥ 2 ≥ t 2 m/σ 2 ) = P(X ≥ t 2 m/σ 2 ),
where X = ∥n∥ 2 ∼ χ 2 m is a chi-squared random variable with m degrees of freedom. Using a standard Chernoff bound once more, we have P(X ≥ zm) ≤ (ze 1-z ) m/2 , for any z > 1. Setting z = t 2 σ 2 , we have
P(∥e∥ ≥ t) ≤ t 2 σ 2 e 1-t 2 σ 2 m/2
, which gives the first result.
For the second result, we recall that E has density
p E (e) = (2πσ 2 /m) -m/2 exp - m 2σ 2 ∥e∥ 2 . Therefore p E (u) p E (v) = exp m 2σ 2 (∥v∥ -∥u∥) (∥v∥ + ∥u∥) .
Now suppose that ∥u∥ ≤ τ and ∥u -v∥ ≤ ε.
Then
p E (u) p E (v) ≤ exp m(2τ + ε) 2σ 2 ε .
Hence
D shift (ε, τ ; E) ≤ exp m(2τ + ε) 2σ 2 ε ,
which gives the second result.
this section cite: []

Section: B.2 Subgaussian concentration inequalities
Lemma B.2 (Lower and upper concentration bounds for subgaussian random matrices). Let A be a distribution of subgaussian random matrices with parameters β, κ > 0 (Definition 3.4). Then the lower ad upper concentration bounds for A (Definition 2.3) can be taken as
C upp (t; A, R n ) = C low (1/t; A, R n ) = 2 exp(-c(t, β, κ)m)
for any t > 1, where c(t, β, κ) > 0 depends on β, κ only.
Proof. Let x ∈ R n and observe that
P(∥Ax∥ ≥ t∥x∥) ≤ P ∥Ax∥ 2 -∥x∥ 2 ≥ (t 2 -1)∥x∥ 2 P(∥Ax∥ ≤ t -1 ∥x∥) ≤ P ∥Ax∥ 2 -∥x∥ 2 ≥ (1 -t -2 )∥x∥ 2 . (B.1) We now use [39, Lem. 9.8]. Note that this result only considers a bound of the form P(|∥Ax∥ 2 -∥x∥ 2 | ≥ s∥x∥ 2 ) for s ∈ (0, 1). But the proof straightforwardly extends to s > 0. B.3 Concentration inequalities for randomly-subsampled orthogonal transforms Lemma B.3 (Concentration bounds for randomly-subsampled orthogonal transforms). Let D ⊆ R n and A be a distribution of randomly-subsampled orthognal transforms based on a matrix U (Definition 3.7). Then the lower and upper concentration bounds for A (Definition 2.3) can be taken as C upp (t; A, D) = C low (1/t; A, D) = 2 exp -mc(t) µ(U ; D) for any t > 1, where µ(U ; D) is a in Definition 3.8 and c(t) > 0 depend on t only. Proof. Due to (B.1), it suffices to bound P ∥Ax∥ 2 -∥x∥ 2 ≥ s∥x∥ 2 for s > 0. The result uses Bernstein's inequality for bounded random variables (see, e.g., [5, Thm. 12.18]). Let x ∈ D. By definition of A and the fact that U is orthogonal, we can write
∥Ax∥ 2 -∥x∥ 2 = n i=1 n m I Xi=1 -1 |⟨u i , x⟩| 2 =: N i=1 Z i .
Notice that the random variables Z i are independent, with E(Z i ) = 0. We also have
|Z i | ≤ n m |⟨u i , x⟩| 2 ≤ µ(U ; D) m ∥x∥ 2 =: K and n i=1 E|Z i | 2 ≤ n i=1 n 2 |⟨u i , x⟩| 4 m 2 E(I 2 Xi=1 ) ≤ K n i=1 n|⟨u i , x⟩| 2 m m n = K∥x∥ 2 =: σ 2 .
Therefore, by Bernstein's inequality,
P(|∥Ax∥ 2 -∥x∥ 2 | ≥ s∥x∥ 2 ) ≤ 2 exp - s 2 ∥x∥ 4 /2 σ 2 + Ks∥x∥ 2 /3) = 2 exp - ms 2 /2 µ(U ; D)(1 + s/3)
for any s > 0 and x ∈ D. The result now follows.
Lemma B.4 (Absolute concentration bounds for subsampled orthogonal transforms). Let D ⊆ R n and A be a distribution of randomly-subsampled orthogonal transforms based on a matrix U (Definition 3.7). Then ∥A∥ ≤ n/m a.s. for A ∼ A, and consequently the absolute concentration bound for A (Definition 2.3) can be taken as C abs (s, t; A, D) = 0 for any s ≥ 0 and t ≥ s n/m.
Proof. Recall that A consists of q rows of an orthogonal matrix U multiplied by the scalar n/m. Hence ∥A∥ ≤ n/m∥U ∥ = n/m. Now let x ∈ D with ∥x∥ ≤ s. Then ∥Ax∥ ≤ ∥A∥∥x∥ ≤ ∥A∥s. Therefore ∥Ax∥ n/ms ≤ t, meaning that P(∥Ax∥ > t) = 0. This gives the result.
this section cite: []

Section: B.4 Proofs of Theorems 3.5 and 3.9
Proof of Theorem 3.5. Let p = P [∥x * -x∥ ≥ 34(η + σ)]. We use Theorem 3.1 with c = 32, c ′ = 2, t = 2 and ε replaced by ε/d, where d ≥ 1 is a constant that will be chosen later. Let ε ′ = ε/(dδ 1/p ). Then Theorem 3.1 gives
p ≲ δ + C abs (ε ′ , 2ε ′ ; A, R n ) + D upp (2σ; E) + 2D shift (2ε ′ , 2σ; E)e k [C low (1/2; A, R n ) + C upp (2; A, R n ) + 2D upp (2σ; E)] .
Consider C abs (ε ′ , 2ε ′ ; A, R n ). If x ∈ R n with ∥x∥ ≤ s, then P(∥Ax∥ > t) ≤ P(∥Ax∥ > (t/s)∥x∥).
Hence, in this case, we may take
C abs (ε ′ , 2ε ′ ; A, R n ) = C upp (2; A, R n ). (B.2)
Now by Lemma B.2, we have that
C low (1/2; A, R n ) = C upp (2; A, R n ) = exp(-c(β, κ)m),
where c(β, κ) > 0 depends on β, κ only. Also, by Lemma B.1, we have
D upp (2σ; E) = 2e -1 m/2 = exp(-cm),
for some universal constant c > 0 and
D shift (2ε ′ , 2σ; E) = exp m(4σ + 2ε ′ ) 2σ 2 2ε ′ ≤ exp 6m d ,
where we used the facts that σ ≥ ε/δ 1/p = dϵ ′ and d ≥ 1. We deduce that
p ≲ δ + exp(k + 6m/d -c(β, κ)m)
for a possibly different constant c(β, κ) > 0. We now choose d = d(β, κ) = 12/c(β, κ). Up to another possible change in c(β, κ), the condition (3.4) on m and (3.1) now give that p ≲ δ, as required.
Proof of Theorem 3.9. Let p = P [∥x * -x∥ ≥ 34(η + σ)]. In this case, the forwards operator A satisfies ∥A∥ ≤ n/m (Lemma B.4). Hence we may apply Theorem 1.1 with θ = n/m and d = 2 to obtain p ≲ δ + Cov η,δ (P) [C low (1/2; A, D) + C upp (2; A, D) + exp(-cm)]
for some universal constant c > 0, where D = supp(P) -supp(P). Lemma B.3 now gives that
p ≲ δ + Cov η,δ (P) exp - cm µ(U ; D) + exp(-cm)
for a possibly different constant c > 0. The result now follows from the condition (3.5) on m.
this section cite: []

Section: C Proofs of Theorems 1.1 and 3.1
We finally consider the proofs of the two general results, Theorems 1.1 and 3.1. Our main effort will be in establishing the latter, from which the former will follow after a short argument.
To prove Theorem 3.1, we first require some additional background on couplings, along with several lemmas. This is given in §C.1. We then establish a series of key technical lemmas, presented in §C.2, which are used in the main proof. Having shown these, the proof of Theorem 3.1 proceeds in §C.3 via a series of step. We now briefly describe these steps, and by doing so explain how the sets D 1 , D 2 defined in (3.2)-(3.3) arise.
(i) First, using Lemma C.7, we decompose P, R into distributions P ′ , R ′ that are supported in balls of a given radius, plus remainder terms. The distributions P ′ , R ′ are close (in W ∞ ) to a discrete distribution Q supported at the centres of the balls that give the approximate cover satisfying (3.1). (ii) Next, we replace x * ∼ R in the definition of the probability p in Theorem 3.1 by z * ∼ P ′ .
The is done to align the prior with the posterior P(•|y, A), which is needed later in the proof. We do this using Lemma C.6. Here we have to consider the action of A on vectors of the form x -z, where x ∈ supp(R ′ ) and z ∈ supp(P ′ ). After determining the supports of R ′ , P ′ , we see that x -z ∈ D 1 , where D 1 is the set defined in (3.2).
(iii) We now decompose P ′ into a mixture over the balls mentioned in (i). After a series of arguments, we reduce the task to that of considering the probability that the conditional distribution is drawn from one ball when the prior is drawn from another. Lemmas C.4 and C.5 handle this. They involve estimating the action of A on vectors x -z, where z is the centre of one of the balls and x comes from another ball. Since the balls are supported in supp(P) we have x ∈ supp(P), and since the centres come from the approximate covering number bound (3.1), we have that z ∈ supp(P) if P attains the minimum and z ∈ supp(R) otherwise. Hence x -z ∈ D 2 , with D 2 as in (3.3).
this section cite: []

Section: C.1 Background on couplings
For a number of our results, we require some background on couplings. We first recall some notation. Given probability spaces (X, F 1 , µ), (Y, F 2 , ν), we write Γ = Γ µ,ν for the set of couplings, i.e., probability measures on the product space (X × Y, σ(F 1 ⊗ F 2 )) whose marginals are µ and ν, respectively. For convenience, we write π 1 : X × Y → X and π 2 : X × Y → Y for the projections π 1 (x, y) = x and likewise π 2 (x, y) = y. In particular, for any coupling γ we have π 1 ♯γ = µ and π 2 ♯γ = ν, where ♯ denotes the pushforward operation. As an immediate consequence, we observe that for any measurable function φ :
X → R, X φ(x) dµ(x) = X φ(x) dπ 1 ♯γ(x) = X×Y φ(x) dγ(x, y). (C.1) Given a cost function c : X × Y → [0, ∞), the Wasserstein-p metric is defined as W p (µ, ν) = inf γ∈Γ X×Y c(x, y) p dγ(x, y) 1/p for 1 ≤ p < ∞ and W ∞ (µ, ν) = inf γ∈Γ esssup γ c(x, y) .
We say that γ ∈ Γ is a W p -optimal coupling of µ and ν if X×Y c(x, y) p dγ(x, y)
1/p = W p (µ, ν) for 1 ≤ p < ∞ or esssup γ c(x, y) = W ∞ (µ, ν) when p = ∞.
Note that such a coupling exists whenever X, Y are Polish spaces, and when the cost function is lower semicontinuous [41]. In our case, we generally work with Euclidean spaces with the cost function being the Euclidean norm, hence both conditions are satisfied.
For convenience, if γ is probability measure on the product space (X × Y, σ(F 1 ⊗ F 2 )), we will often write γ(E 1 , E 2 ) instead of γ(E 1 × E 2 ) for E i ∈ F i , i = 1, 2. Moreover, if x ∈ X is a singleton, we write γ(x, E 2 ) for γ({x} × E 2 ) and likewise for γ(E 1 , y).
We now need several lemmas on couplings. Lemma C.1. Suppose that (X, F 1 , µ), (Y, F 2 , ν) are Borel probability spaces, and let γ be a coupling of µ, ν on the space (X × Y, σ(F 1 ⊗ F 2 )). Then supp(γ) ⊆ supp(µ) × supp(ν).
Proof. Let (x, y) ∈ supp(γ). Then γ(U x,y ) > 0 for every open set U x,y ⊆ X × Y that contains (x, y). Now, to show that (x, y) ∈ supp(µ) × supp(ν), we show that x ∈ supp(µ) and y ∈ supp(ν).
Let U x ⊆ X be open with x ∈ U x . By definition, µ(U x ) = γ(U x × R n ). Since U x × R n
is open and contains (x, y), it follows that γ(U x × R n ) > 0. Since U x was arbitrary, we deduce that x ∈ supp(µ). The argument that y ∈ supp(ν) is identical.
Lemma C.2. Let X be a Polish space with a complete metric d. Let µ, ν be Borel probability measures on X. Let d H be the Hausdorff metric with respect to d and W ∞ be the Wasserstein-∞ metric with cost function d. Then
d H (supp(µ), supp(ν)) ≤ W ∞ (µ, ν). In particular, supp(µ) ⊆ B η (supp(ν)) for any η ≥ W ∞ (µ, ν). Proof. Since d H (supp(µ), supp(ν)) = max sup x∈supp(ν) d(x, supp(µ)), sup y∈supp(µ) d(y, supp(ν))
we may, without loss of generality, assume that the maximum is achieved by
sup x∈supp(ν) d(x, supp(µ)) =: D. Take a sequence {x n } n∈N ⊆ supp(ν) such that D n := d(x n , supp(µ)) → D. Since x n ∈ supp(ν), for any ε > 0, we have ν(B ε (x n )) > 0. Note that B ε (x n ) is measurable as we assume X is Borel. For each n ∈ N, define ε n = 1 n D n . We show that for all x ∈ B εn (x n ), y ∈ supp(µ), d(x, y) > D n (1 -1 n ). By triangle inequality, we have d(x, y) ≥ d(x n , y) -d(x n , x) ≥ D n -D n /n = D n (1 -1/n). Notice also that D n (1 -1 n ) ≤ D and converges to D as n → ∞. This implies that A := {(x ′ , y ′ ) : d(x ′ , y ′ ) > D n (1 -1/n)} ⊇ B εn (x n ) × supp(µ). Now consider any coupling γ ∈ Γ µ,ν . We have γ(A) ≥ γ(B εn (x n ) × supp(µ)) = γ(B εn (x n ) × X) = ν(B εn (x n )) > 0. Therefore ess sup γ d(x, y) > D n (1-1/n). Now since D n (1-1 n ) → D we have ess sup γ d(x, y) ≥ D.
This is holds for any coupling, therefore the result follows.
When working with a coupling between a finitely-supported distribution and a continuous distribution, the following lemma is often useful.
Lemma C.3. Let (X, F 1 , µ), (Y, F 2 , ν) be probability spaces, such that ν is finitely supported on a set S ⊆ Y . Let γ be a coupling of µ, ν and E ⊆ X × Y be γ-measurable. Write E y = {x : (x, y) ∈ E} for the slice of E at y ∈ Y . Then γ(E) = s∈S,s∈π2(E) γ(E s × {s}). Proof. Write γ(E) = s∈S,s∈π2(E) γ(E s × {s}) + γ( Ê),
where
Ê = E\ s∈S,s∈π2(E) (E s × {s}). It suffices to show that γ( Ê) = 0. Since F ⊆ π -1 2 (π 2 (F )) for any set F , we have γ( Ê) ≤ γ(π -1 2 (π 2 ( Ê)) = ν(π 2 ( Ê)) = ν(π 2 ( Ê) ∩ S). But Ê = {(x, y
) ∈ E : y / ∈ S} and therefore π 2 ( Ê) ∩ S = ∅. The result now follows.
this section cite: ['b40']

Section: C.2 Technical lemmas C.2.1 Separation lemma
The lemma considers a scenario where two random variables are drawn for a mixture of k probability distributions. The second random variable is conditioned on the draw of the first. It then considers the probability that the two random variables are drawn from different distributions in the mixture, bounding this in terms of their Total Variation (TV) distance. It generalizes [49, Lem. 3.1]. Lemma C.4 (Separation lemma). Let H 1 , . . . , H k be Borel probability measures and consider the mixture H =
P[y * ∼ H i , ŷ ∼ H j (•|y * )] ≤ 1 -TV(H i , H j ).
To clarify, in this lemma and elsewhere we use the notation y * ∼ H i (and similar) to mean the event that y * is drawn from the ith distribution H i .
Proof. Note that if the H i have densities h i with respect to some measure, these weights are given by
P(y * ∼ H i |y * ) = a i h i (y * ) k j=1 a j h j (y * ) . (C.2) We now write p := P[y * ∼ H i , ŷ ∼ H j (•|y * )] = P[y * ∼ H i ]P[ŷ ∼ H j (•|y * )|y * ∼ H j ] = P[y * ∼ H i ]E[P(ŷ ∼ H j (•|y * )|y * ∼ H i ] = a i E[P(ŷ ∼ H j (•|y * ))|y * ∼ H i ]. Since E[y * |y * ∼ H i ] ∼ H i , we have p = a i E[P(ŷ ∼ H j (•|y * ))|y * ∼ H i ] = a i P(ŷ ∼ H j (•|y * )) dH i (y * ).
Now, because of the mixture property, H i ≪ H and therefore its Radon-Nikodym derivative h i = dHi dH exists. This means we may write
p = a i P(ŷ ∼ H j (•|y * ))h i (y * ) dH(y * ).
By definition, we have P(ŷ ∼ H j (•|y * )) = P(y * ∼ H j |y * ) and using (C.2), we deduce that
p = a i a j h i (y * )h j (y * ) k l=1 a l h l (y * ) dH(y * ) We now write p = a i h i (y * )h j (y * )a j a i h i (y * ) + a j h j (y * ) + l̸ =i,j a l h l (y * ) dH(y * ) ≤ a i h i (y * )h j (y * )a j a i h i (y * ) + a j h j (y * ) dH(y * ) = a i h i (y * )h j (y * )a j (a i h i (y * ) + a j h j (y * )) dH(y * ) = a i h i (y * )h j (y * )a j a i h i (y * ) + a j h j (y * ) dH(y * ) ≤ a i h i (y * )h j (y * )a j max{a i h i (y * ), a j h j (y * )} dH(y * ) = min{a i h i (y * ), a j h j (y * )} dH(y * ) = 1 - 1 2 (h i (y * ) + h j (y * )) -min{a i h i (y * ), a j h j (y * )} dH(y * ) = 1 - 1 2 |h i (y * ) -h j (y * )| dH(y * ) = 1 -TV(H i , H j ),
as required.
this section cite: []

Section: C.2.2 Disjointly-supported measures induce well-separated measurement distributions
The following lemma pertains to the pushforwards of measures supported in R n via the forward operator A and noise e. Specifically, it states that if two distributions P int and P ext are disjointly supported then their corresponding pushforwards H int,A and H ext,A are, on average with respect to A ∼ A, well-separated, in the sense of their TV-distance. It is generalization of [49, Lem. 3.2] that allows for arbitrary distributions A of the forward operators, as opposed to just distributions of Gaussian random matrices. Lemma C.5 (Disjointly-supported measures induce well-separated measurement distributions). Let x ∈ R n , σ ≥ 0, η ≥ 0, c ≥ 1, P ext be a distribution supported in the set
S x,ext = {x ∈ R n : ∥x -x∥ ≥ c(η + σ)}
and P int be a distribution supported in the set
S x,int = {x ∈ R n : ∥x -x∥ ≤ η}. Given A ∈ R m×n ,
let H int,A be the distribution of y = Ax * + e where x * ∼ P int and e ∼ E independently, and define H ext,A in a similar way. Then E A∼A [TV(H int,A , H ext,A )] ≥ 1-C low 2 √ c ; A, D ext + C upp √ c 2 ; A, D int + 2D upp √ cσ 2 ; E , where D ext = {x -x : x ∈ supp(P ext )}, D int = {x -x : x ∈ supp(P int )} and C upp (•; A), C low (•; A) and D upp (•; E) are as in Definitions 2.3 and 2.4, respectively.
Notice that the average TV-distance is bounded below by the concentration bounds C low and C upp for A (Definition 2.3) and the concentration bound D upp for E (Definition 2.4). This is unsurprising. The pushforward measures are expected to be well-separated if, firstly, the action of A approximately preserves the lengths of vectors (which explains the appearance of C low and C upp ) and, secondly, adding noise by E does not, with high probability, cause well-separated vectors to become close to each other (which explains the appearance of D upp ). Also as expected, as c increases, i.e., the distributions P int and P ext become further separated, the average TV-distance increases.
Proof. Given A ∈ R m×n , let
B A = {y ∈ R m : ∥y -Ax∥ ≤ √ c(η + σ)}.
We claim that
E A [H ext,A (B A )] ≤ C low 2 √ c ; A, D ext + D upp (σ √ c; E), (C.3) E A [H int,A (B A )] ≥ 1 -C upp √ c 2 ; A, D int + D upp √ c 2 σ; E . (C.4)
Notice that these claims immediately imply the result, since
E A∼A TV(H ext,A , H int,A ) ≥ E A∼A [H int,A (B A )] -E A∼A [H ext,A (B A )]
. Therefore, the rest of the proof is devoted to showing (C.3) and (C.4). For the former, we write
E A∼A [H ext,A (B A )] = E A∼A 1 B A (Ax + e) dP ext(
x) dE(e) = E A∼A 1 B A (Ax + e) dE(e) dP ext (x) = E A∼A [E x∼Pext [E(B A -Ax)]] = E x∼Pext [E A∼A [E(B A -Ax)]], (C.5) where B A -Ax = {b -Ax : b ∈ B A }. We now bound E x∼Pext [E A∼A E(B A -Ax)]. Given x ∈ R n , let C x = {A : ∥Ax -Ax∥ < 2 √ c(η + σ)} ⊆ R m×n and write I 1 = E x∼Pext [E A∼A E(B A -Ax)1 Cx ], I 2 = E x∼Pext [E A∼A E(B A -Ax)1 C c x ] so that E x∼Pext [E A∼A [E(B A -Ax)]] = I 1 + I 2 .
(C.6) We will bound I 1 , I 2 separately. For I 1 , we first write
I 1 = E x∼Pext [E A∼A [E(B A -Ax)1 Cx ]] ≤ E x∼Pext [E A∼A [1 Cx ]] = E x∼Pext [P A∼A (∥Ax -Ax∥ < 2 √ c(η + σ))],
where the inequality follows from the fact that E(B A -Ax) ≤ 1. Now since x ∼ P ext , we have x ∈ S x,ext and therefore ∥x -x∥ ≥ c(η + σ). Hence
E x∼Pext [P A∼A (∥Ax -Ax∥ < 2 √ c(η + σ)] ≤ E x∼Pext P A∼A ∥Ax -Ax∥ < 2 √ c ∥x -x∥ .
Since the outer expectation term has x ∼ P ext , we have that x ∈ supp(P ext ) with probability one. Using Definition 2.3, we deduce that
I 1 ≤ C low 2 √ c ; A, D ext . (C.7)
We now bound I 2 . Let x ∈ S x,ext and A ∈ C c x , i.e., ∥A(x -x)∥ > 2 √ c(η + σ). We now show that B A ⊆ B A,x , where B A,x = {y ∈ R m : ∥y -Ax∥ ≥ √ c(η + σ)}. Suppose that y ∈ B A , i.e., ∥y -Ax∥ ≤ √ c(η + σ). We have
∥y -Ax∥ = ∥y -Ax + Ax -Ax∥ ≥ ∥A(x -x)∥ -∥y -Ax∥ > 2 √ c(η + σ) - √ c(η + σ) = √ c(η + σ),
and therefore y ∈ B A,x , as required. Using this, we have E(B A -Ax) ≤ E(B A,x -Ax), ∀A ∈ C c x , x ∈ S x,ext , and therefore
I 2 = E x∼Pext [E A∼A [E(B A -Ax)1 C c x ]] ≤ E x∼Pext [E A∼A [E(B A,x -Ax)1 C c x ]]. But we notice that B A,x -Ax = B c √ c(η+σ) . Now since η ≥ 0, we have B c √ c(η+σ) ⊆ B c σ √ c . Hence E(B A,x -Ax) = E(B c √ c(η+σ) ) ≤ E(B c σ √ c ) ≤ Dupp(σ √ c; E),
and therefore I 2 ≤ D upp (σ √ c; E). Combining this with (C.5), (C.6) and (C.7), we deduce that
E A [H ext,A (B A )] ≤ E x∼Pext [E A∼A [E(B A -Ax)]] = I 1 +I 2 ≤ C low (2/ √ c; A, D ext )+C upp (σ √ c; E),
which shows (C.3).
We will now establish (C.4). With similar reasoning to (C.5), we have
E A∼A [H int,A (B c A )] = E x∼P int [E A∼A [E(B c A -Ax)]]
Proceeding as before, let
D x = {A : ∥Ax -Ax∥ < √ c 2 (η + σ)}, I 1 = E x∼P int [E A∼A [E(B c A - Ax)]1 D c x ]
, and
I 2 = E x∼P int [E A∼A [E(B c A -Ax)]1 Dx ] so that E A∼A [H int,A (B c A )] = E x∼P int [E A∼A [E(B c A -Ax)]] = I 1 + I 2 .
(C.8)
The terms I 1 , I 2 are similar to those considered in the previous case. We bound them similarly. For I 1 , we have, by dropping the inner probability terms,
I 1 ≤ E x∼P int [E A∼A [1 D c x ]] = E x∼P int P A∼A (∥A(x -x)∥ ≥ √ c 2 (η + σ)) .
Since x ∈ S x,int , we have ∥x -x∥ ≤ η ≤ η + σ which gives
E x∼P int [P A∼A (∥A(x -x)∥ ≥ √ c 2 (η + σ))] ≤ E x∼P int P A∼A (∥A(x -x)∥ ≥ √ c 2 ∥x -x∥)
and therefore
I
E A [H int,A (B c A )] ≤ E x∼P int [E A∼A [E((B A -Ax) c )]] ≤ C upp √ c 2 ; A, D int + D upp √ c 2 σ; E ,
which implies (C.4). This completes the proof.
this section cite: []

Section: C.2.3 Replacing the real distribution with the approximate distribution
We next establish a result that allows one to upper bound the failure probability based on draws from the real distribution R with the failure probability based on draws from the approximate distribution P. This lemma is a key technical step that aligns the prior distribution with the posterior. The specific bound is given in terms of the Wasserstein distance between R and P and several of the concentration bounds defined in §2. This is a significant generalization of [49, Lem. 3.3] that allows for arbitrary distributions A, E for the forwards operator and noise. Lemma C.6 (Replacing the real distribution with the approximate distribution). Let ε, σ, d, t ≥ 0, c ≥ 1, E be a distribution on R m and R, P be distributions on R n such that W ∞ (R, P) ≤ ε. Let Π be an W ∞ -optimal coupling of R and P and define the set D = {x * -z * : (x * , z * ) ∈ supp(Π)}. Let p = P x * ∼R,A∼A,e∼E,x∼P(•|Ax * +e,A) [∥x * -x∥ ≥ d + ε] and q = P z * ∼P,A∼A,e∼E,ẑ∼P(•|Az * +e,A) [∥z * -ẑ∥ ≥ d]. Then p ≤ C abs (ε, tε; A, D) + D upp (cσ; E) + D shift (tε, cσ; E)q, where C abs (ε, tε; A, D), D upp (cσ; E) and D shift (tε, cσ; E) are as in Definitions 2.3, 2.4 and 2.5, respectively. As expected, this lemma involves a trade-off. The constant C abs (ε, tε; A, D) is made smaller (for fixed ε) by making the constant t larger. However, this increases D shift (tε, cσ; E), which is compensated by making c smaller. However, this in turn increases the constant D upp (cσ; E). Proof. Define the events B 1,x = {x * : ∥x * -x∥ ≥ d + ε}, B 2,ẑ = {z * : ∥z * -ẑ∥ ≥ d} so that p = P x * ∼R,A∼A,e∼E,x∼P(•|Ax * +e,A) [x * ∈ B 1,x ] q = P z * ∼P,A∼A,e∼E,ẑ∼P(•|Az * +e,A) [z * ∈ B 2,ẑ ] . (C.10) Observe that p = E x * ∼R [E A∼A E y|A,x * [E x∼P(•|Ax * +e,A) [1 B 1,x ]]] = 1 B 1,x (x * )dP(•|Ax * + e, A)(x)dE(e)dA(A)dR(x * ) and similarly q = 1 B 2,ẑ (z * )dP(•|Az * + e, A)(ẑ)dE(e)dA(A)dP(z * ).
Therefore, to obtain the result, it suffices to replace samples from the real distribution R with samples from the approximate distribution P and to replace the indicator function of B 1,x by the indicator function over B 2,ẑ . For the first task, we use couplings. Since W ∞ (R, P) ≤ ε, there exists a coupling Π between R, P with Π(∥x * -z * ∥ ≤ ε) = 1. By (C.1), we can write
p = 1 B 1,x (x * )dP(•|Ax * + e, A)(x)dE(e)dA(A)dΠ(x * , z * ).
Define E = {(x * , z * ) : ∥x * -z * ∥ ≤ ε} and observe that Π(E) = 1. Then, for fixed A, e, we have
1 B 1,x (x * )dP(•|Ax * + e, A)(x)dΠ(x * , z * ) = E 1 B 1,x (x * )dP(•|Ax * + e, A)(x)dΠ(x * , z * ) 1 B 2,x (z * )dP(•|Ax * + e, A)(x)dΠ(x * , z * ) = E 1 B 2,x (z * )dP(•|Ax * + e, A)(x)dΠ(x * , z * ). We now show 1 B 1,x (x * ) ≤ 1 B 2,x (z * ) for (x * , z * ) ∈ E. Let (x * , z * ) ∈ E
and suppose that x * ∈ B 1,x . Then ∥x * -x∥ ≥ d + ε and, since ∥x * -z * ∥ ≤ ε, we also have that ∥z * -x∥ ≥ d and therefore z * ∈ B 2,x , as required. Hence 1 B 1,x (x * )dP(•|Ax * + e, A)(x) ≤ 1 B 2,x (z * )dP(•|Ax * + e, A)(x) for (x * , z * ) ∈ E. Now, since indicator functions are non-negative, Fubini's theorem immediately implies that p = 1 B 1,x (x * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) ≤ 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ). Having introduced the coupling Π and replaced 1 B 1,x by 1 B 2,x , to establish the result it remains to replace the conditional distribution P(•|Ax * + e, A) by P(•|Az * + e, A). With a similar technique to that used in the proof of Lemma C.5, we define C x * ,z * = {A : ∥A(x * -z * )∥ > tε} and I 1 = 1 C x * ,z * (A) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) I 2 = 1 C c x * ,z * (A) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) so that p ≤ 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) = I 1 + I 2 . (C.11)
We first bound I 1 . As before, we write
I 1 = 1 C x * ,z * (A) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) ≤ 1 C x * ,z * (A) dA(A) dΠ(x * , z * ).
Recalling the definition of the set E above, we get
1 C x * ,z * (A) dA(A) dΠ(x * , z * ) ≤ E P A∼A {∥A(x * -z * )∥ > tε} dΠ(x * , z * ).
Using the definition of C 0 , E and D, we deduce that
I 1 ≤ C abs (ε, tε; A, D). (C.12) Now we bound I 2 . We further split the integral I 2 as follows: I 2 = I 21 + I 22 , (C.13) where I 21 = 1 C c x * ,z * (A) 1 B c cσ (e) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) I 22 = 1 C c x * ,z * (A) 1 Bcσ (e) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) Let us first find an upper bound for I 21 . We have I 21 = 1 C c x * ,z * (A) 1 B c cσ (e) 1 B 2,x (z * ) dP(•|Ax * + e, A)(x) dE(e) dA(A) dΠ(x * , z * ) ≤ 1 C c x * ,z * (A) 1 B c cσ (e) dE(e) dA(A) dΠ(x * , z * ) ≤ 1 B c cσ (e) dE(e), and therefore, by Definition 2.4, I 21 = E(B c cσ ) ≤ D upp (cσ; E). (C.14) We now find a bound for I 22 . We first use Definition 2.5 to write I 22 = 1 C c x * ,z * (A) 1 Bcσ (e)p E (e) 1 B 2,x (z * )dP(•|Ax * + e, A)(x) de dA(A) dΠ(x * , z * ). Now define the new variable e ′ = e + A(x * -z * ). Since, in the integrand, ∥e∥ ≤ cσ (due to the indicator function 1 Bcσ (e)) and ∥A(x * -z * )∥ ≤ tε (due to the indicator function 1 C c x * ,z * (A)), Definition 2.5 yields the bound I 22 ≤ D shift (tε, cσ; E) 1 C c x * ,z * (A) 1 B2σ (e ′ -A(x * -z * ))p E (e ′ ) × 1 B 2,x (z * ) dP(•|Az * + e ′ , A)(x) de ′ dA(A) dΠ(x * , z * ).
We now drop the first two indicator functions and relabel the variables e ′ and x as e and ẑ, respectively, to obtain
I 22 ≤ D shift (tε, cσ; E) 1 B 2,ẑ (z * ) dP(•|Az * + e, A)(ẑ) dE(e) dA(A) dΠ(x * , z * ).
This gives I 22 ≤ D shift (tε, cσ; E)q, where q is as in (C.10). Combining this with (C.13), we deduce that
I 2 ≤ D upp (cσ, ε) + D shift (tε, cσ; E)q.
To complete the proof, we combine this with (C.11) and (C.12), and then recall (C.10) once more.
this section cite: []

Section: C.2.4 Decomposing distributions
The following lemma is in large part similar to [49,Lem. A.1]. However, we streamline and rewrite its proof for clarity and completeness, fix a number of small issues and make an addition to the statement (see item (v) below) that is important for proving our main result. Lemma C.7 (Decomposing distributions). Let R, P be arbitrary distributions on R n , p ≥ 1 and η, ρ, δ > 0. If W p (R, P) ≤ ρ and k ∈ N is such that This lemma states that two distributions that are close in Wasserstein p-distance, and for which at least one has small approximate covering number (C.15), can be decomposed into mixtures (iii) of distributions, where the following holds. One of the distributions, say P ′ , is close (i) in Wasserstein-∞ distance to a discrete distribution Q with the cardinality of its support (iv) bounded by the approximate covering number. The other R ′ is close in Wasserstein-∞ distance to P ′ . Moreover, both mixtures (iii) are dominated by these distributions: the 'remainder' terms P ′′ and R ′′ are associated with a small constant δ ′ ≤ δ, meaning they are sampled with probability ≤ δ when drawing from either P or R. Note that if p < ∞ then the Wasserstein-∞ distance between R ′ and P ′ may get larger as δ shrinks, i.e., as the remainder gets smaller. However, this does not occur when p = ∞, as (ii) is independent of δ in this case.
Proof. Without loss of generality, we assume that log Cov η,δ (P) ≤ k. Then Cov η,δ (P) ≤ e k and hence there is a set S = {u i } l i=1 ⊆ supp(P) with l ≤ e k , where the u i are the centres of the balls used to cover at least 1 -δ of the measure of P. That is,
P l i=1 B(u i , η) = P x∼P x ∈ l i=1 B(u i , η) =: 1 -c * ≥ 1 -δ.
We now define f : R n → R so that f (x) = 0 if x lies outside these balls, and otherwise, f (x) is the equal to the reciprocal of the number of balls in which x is contained. Namely,
f (x) = 1 l i=1 1 B(u i ,η) (x) if x ∈ l i=1 B(u i , η) 0 otherwise .
We divide the remainder of the proof into a series of steps.
1. Construction of Q ′ . We will now define a finite measure Q ′ . The point of Q ′ is to, concentrate the mass of the measure P into the centres of the balls u i . If the sets B(u i , η) are disjoint, then this is straightforward. However, to ensure that Q ′ is indeed a probability measure, we need to normalize and account for any non-trivial intersections. This is done via the function f . Pick some arbitrary û / ∈ {u 1 , . . . , u l } and define
Q ′ = l i=1 B(ui,η) f (x) dP(x) δ ui + c * δ û.
Observe that
dQ ′ (x) = l i=1 B(ui,η) f (x) dP(x) + c * = P l i=1 B(u i , η) + c * = (1 -c * ) + c * = 1,
and therefore Q ′ is a probability distribution supported on the finite set S ∪ {û}.
2. Coupling Q ′ , P. Now that we have associated all the mass of P with the points u i , we can define a coupling Π between Q ′ and P that associates the mass of P and u i with a single measure. Moreover, this measure will keep points within η distance of each other with high probability. We define Π as follows for measurable sets E, F ⊆ R n :
Π(E, F ) = l i=1 1 F (u i ) B(ui,η)∩E f (x) dP(x) + 1 F (û)P E\ l i=1 B(u i , η) .
To see that this is a coupling, we first observe that
Π(R n , F ) = l i=1 1 F (u i ) B(ui,η) f (x) dP(x) + 1 F (û)(1 -c * ) ≡ Q ′ (F ),
which gives the result for the first marginal. For the other, we have
Π(E, R n ) = l i=1 B(ui,η)∩E f (x) dP(x) + P E\ l i=1 B(u i , η) .
By definition of f , this is precisely
Π(E, R n ) = P E ∩ l i=1 B(u i , η) + P E\ l i=1 B(u i , η) ≡ P(E),
which gives the result for the second marginal. Note that Π was only defined for product sets, but, since Q ′ is finitely supported, it follows directly from Lemma C.3 that it extends to arbitrary measurable sets in the product sigma-algebra. We now show that Π[∥x 1 -x 2 ∥ > η] ≤ c * ≤ δ. That is, we show that most points drawn from Π are within η distance of each other. By law of total probability we have
Π(∥x 1 -x 2 ∥ > η) = l i=1 Π(∥x 1 -x 2 ∥ > η|x 2 = u i )Π(x 2 = u i ) + Π(∥x 1 -x 2 ∥ > η|x 2 = û)Π(x 2 = û) = l i=1 Π(U i , u i )Q ′ (u i ) + Π( Û , û)Q ′ (û
this section cite: ['b48']

Section: Coupling P, R.
The next step is to introduce R. With the assumption that W p (R, P) ≤ ρ, by definition there exists a coupling Γ between P and R such that E Γ [∥x 1 -x 2 ∥ p ] ≤ ρ p . Markov's inequality then gives that
Γ ∥x 1 -x 2 ∥ ≥ ρ δ 1/p ≤ E Γ [∥x 1 -x 2 ∥ p ] ρ p δ ≤ δ. (C.17)
4. Coupling P, Q ′ , R. We next couple P, Q ′ and R. Before doing so, we first discuss the goal of our final coupling. Recall that we have the distribution P, the distribution Π that couples P, Q ′ closely except for up to δ mass of P, and Γ which keeps P, R close again except for up to δ of the mass of Γ. We want to decompose P into the portions that are η close to Q ′ , and points that are not. These will become P ′ and P ′′ , respectively. At the same time, we want to decompose R to points that are ρ δ 1/p close to P ′ , and points that are not. Naturally this will become R ′ and R ′′ . To achieve this, we couple P, Q ′ and R in this step and then use this to construct the final decomposition in the next step.
We have measures P, Q ′ and R and couplings Π of P, Q ′ and Γ of P, R. We will in a sense, couple Π, Γ. Since (R n ) 3 is a Polish space, by [10, Lem. 8.4], there exists a coupling Ω with π 1,2 ♯Ω = Π, π 1,3 ♯Ω = Γ, where π 1,2 (x 1 , x 2 , x 3 ) = (x 1 , x 2 ) and likewise for π 1,3 . One should intuitively think of the x 1 component as samples from P, the x 2 component as samples from Q, and the x 3 component as samples from R. With the base measure defined, we still want to ensure that x 1 , x 3 are sampled closely, and x 1 , x 2 are as well. Consider the event such that x 1 , x 3 are ρ δ 1/p close and x 1 , x 2 are η close: namely, E := {(x 1 , x 2 , x 3 ) : ∥x 1 -x 3 ∥ ≤ ρ/δ 1/p and ∥x 1 -x 2 ∥ ≤ η}.
Split up the negation of the two events of ∥x 1 -x 3 ∥ ≤ ρ/δ 1/p and ∥x 1 -x 2 ∥ ≤ η into the events E 1 := {(x 1 , x 2 , x 3 ) : ∥x 1 -x 2 ∥ > η, z ∈ R n },
E 2 := {(x 1 , x 2 , x 3 ) : ∥x 1 -x 3 ∥ > ρ/δ 1/p , y ∈ R n }, so that E c = E 1 ∪ E 2 . We will now show Ω(E 1 ) ≤ δ. Write E 1 = E ′ 1 × R n where E ′ 1 = {(x 1 , x 2 ) : ∥x 1 -x 2 ∥ > η} satisfies Π(E ′
1 ) ≤ δ ′ by (C.16). Then
δ ′ ≥ Π(E ′ 1 ) = 1 E ′ 1 (x 1 , x 2 ) dπ 1,2 ♯Ω(x 1 , x 2 ) = 1 E ′ 1 (p
1,2 (x 1 , x 2 , x 3 )) dΩ(x 1 , x 2 , x 3 ) = 1 E1 (x 1 , x 2 , x 3 ) dΩ(x 1 , x 2 , x 3 ) = Ω(E 1 ), as required. Using (C.17), we also have the analogous result for E 2 . Hence Ω(E c ) = Ω(E 1 ∪ E 2 ) ≤ Ω(E 1 ) + Ω(E 2 ) =: 2δ ′ ≤ 2δ ′ , and consequently, Ω(E) = 1 -2δ ′ ≥ 1 -2δ, where E := {(x 1 , x 2 , x 3 ) : ∥x 1 -x 3 ∥ ≤ ρ/δ 1/p and ∥x 1 -x 2 ∥ ≤ η}. (C.18) This gives
P(A) = Ω(A, R n , R n ) = Ω(E)Ω(A, R n , R n |E) + Ω(E c )Ω(A, R n , R n |E c ) = (1 -2δ ′ )P ′ (A) + 2δ ′ P ′′ (A)
and similarly
R(A) = Ω(R n , A, R n ) = Ω(E)Ω(R n , A, R n |E) + Ω(E c )Ω(R n , A, R n |E c ) = (1 -2δ ′ )R ′ (A) + 2δ ′ R ′′ (A).
We now claim that these distributions satisfy (i)-(v) in the statement of the lemma. We have already shown that (iii) holds. To show (i), we define a coupling γ of P Recall that for (x 1 , x 2 , x 3 ) ∈ E, we have ∥x 1 -x 2 ∥ ≤ η. Hence Ω(B, R n |E) = 0. Therefore, W ∞ (P ′ , Q) ≤ η, which gives (i).
Similarly, for (ii) we define a coupling γ ′ of R ′ , P ′ by γ ′ (B) = Ω(B|E) where B := {(x 1 , x 2 , x 3 ) : (x 1 x 3 ) ∈ B, x 2 ∈ R n }. With similar reasoning as the previous case, γ ′ is a coupling of R ′ , P ′ and, for (x 1 , x 2 , x 3 ) ∈ E we have ∥x 1 -x 3 ∥ ≤ ρ/δ 1/p , so letting B be the event {∥x 1 -x 3 ∥ > ρ/δ 1/p }, we conclude that W ∞ (R ′ , P ′ ) ≤ ρ/δ 1/p . This gives (ii).
Finally we verify (iv) and (v). First recall that both properties hold for Q ′ by construction. The results now follow from the fact that Q ′ (•) = Ω(R n , •, R n ) and Q(•) = Ω(R n , •, R n |E).
which follows immediately from their definitions, and also that supp(P z,int ) ⊆ B η ′ (z) ⊆ B θ (z).
where in the second inclusion we used the fact that η ′ = η + ε/δ 1/p ≤ η + σ ≤ c(η + σ) = θ, as σ ≥ ε/δ 1/p and c ≥ 1.
We now return to the sum (C.21). Consider an arbitrary term. First, observe that, for z * ∼ P, we have P(z * ∼ Γ z ) = Q(z)(1 -2δ ′ ) by (C.22). Hence
Q(z)E z * ∼Γz,ẑ∼P(•|A,u) [1 E ] = P(z * ∼ Γ z ) 1 -2δ ′ E z * ∼P,ẑ∼P(•|A,u) [1 E |z * ∼ Γ z ] = P(z * ∼ Γ z ) (1 -2δ ′ ) 1 P(z * ∼ Γ z ) 1 E 1 z * ∼Γz dP(z * ) dP(•|A, u)(ẑ).
Recall that z * ∼ Γ z is supported in B η ′ (z). Therefore, for the event E to occur, i.e., ∥z * -ẑ∥ > (c + 1)(η + σ), it must be that ẑ ∈ B c θ (z), which means that ẑ ∼ P z,ext (•|A, u). Hence
Q(z)E z * ∼Γz,ẑ∼P(•|A,u) [1 E ] ≤ 1 1 -2δ ′ 1 ẑ∼Pz,ext(•|A,u) 1 z * ∼Γz dP(z * ) dP(•|A, u)(ẑ) = 1 1 -2δ ′ P[z * ∼ P z,int , ẑ ∼ P z,ext (•|A, u)].
Now fix A ∈ R m×n . Let H z,int,A be the distribution of y * = Az * + e for z * ∼ P z,int and e ∼ E independently, and define H z,ext,A similarly. Then, by Fubini's theorem, we have
Q(z)E z * ∼Γz,A∼A,e∼E,ẑ∼P(•|A,u) [1 E ] ≤ 1 1 -2δ ′ E A∼A,e∼E P[y * ∼ H z,int,A , ŷ ∼ H z,ext,A (•|y * )].
Now let H z,A be the distribution of y = Az + e for z ∼ P and e ∼ E independently. Then Lemma C.4 (with H = H z,A , H 1 = H z,int,A , H 2 = H z,mid,A , H 3 = H z,ext,A and a 1 = (1 -2δ ′ )Q(z), a 2 = c z,mid , a 3 = c z,ext ) gives (C.23) Now recall that H z,int,A is the pushforward of a measure P z,int supported in B η ′ (z), where η ′ = η + ε ′ ≤ η + σ and H z,ext,A is the pushforward of a measure P z,ext supported in B c θ (z), where θ = c(η + σ) ≥ c 2 (η ′ + σ). Therefore, Lemma C.5 (with c replaced by c/2) gives that
Q(z)E z * ∼Γz,A∼A,e∼E,ẑ∼P(•|A,u) [1 E ] ≤ 1 1 -2δ ′ E A∼A
E A∼A [1 -TV(H z,int,A , H z,ext,A )] ≤ C low 2 √ 2 √ c ; A, D z,ext + C upp √ c 2 √ 2 ; A, D z,int + 2D upp √ cσ 2 √ 2 ; E ,
where D z,ext = {x -z : x ∈ supp(P z,ext )} and D z,int = {x -z : x ∈ supp(P z,int )}. It follows immediately from (C.22) that supp(P z,int ), supp(P z,ext ) ⊆ supp(P). Moreover, z ∈ S and therefore D z,ext , D z,int ⊆ D 2 , where D 2 is as in (3.3). Using this, the previous bound and (C.23), we deduce that
r ≤ |S| 1 -2δ ′ C low 2 √ 2 √ c ; A, D 2 + C upp √ c 2 √ 2 ; A, D 2 + 2D upp √ cσ 2 √ 2 ; E . To complete the proof, now substitute this into (C.19) and (C.20), to obtain p ≤ 2δ ′ + [C abs (ε ′ , tε ′ ; A, D 1 ) + D upp (c ′ σ; E)] + 2D shift (tε ′ , c ′ σ; E)|S| C low 2 √ 2 √ c ; A, D 2 + C upp √ c 2 √ 2 ; A, D 2 + 2D upp √ cσ 2 √ 2 ; E .
The result now follows after recalling (iv), i.e., |S| ≤ e k , and the fact that δ ′ ≤ δ ≤ 1/4.
this section cite: []

Section: C.4 Proof of Theorem 1.1
Finally, we now show how Theorem 3.1 implies the simplified result, Theorem 1.1.
D shift (θε ′ , 2σ; E) ≤ exp m(4σ + θε ′ ) 2σ 2 θε ′ = exp ε δ 1/p σ + ε 2 8δ 2/p σ 2 m ≲ 1
where we used the facts that m ≥ 1 and σ ≥ ε/δ 1/p . Hence
p ≲ δ + e k [C -(1/d; A, D 2 ) + C + (d; A, D 2 ) + C(dσ; E)] .
Finally, Lemma B.1 implies that
D upp (dσ; E) ≤ de 1-d m/2 ≤ exp(-m/16),
where in the final step we used the fact that d ≥ 2. This gives the result.
this section cite: []

Section: C.3 Proof of Theorem 3.1
We now prove Theorem 3.1. This follows a similar approach to that of [49,Thm. 3.4], but with a series of significant modifications to account for the substantially more general setup considered in this work. We also streamline the proof and clarify a number of key steps.
Proof of Theorem 3.1. By Lemma C.7, we can decompose P, R into measures P ′ , P ′′ and R ′ , R ′′ , and construct a finite distribution Q supported on a finite set S such that
(i) min{W ∞ (P ′ , Q), W ∞ (R ′ , Q)} ≤ η, (ii) W ∞ (R ′ , P ′ ) ≤ ε ′ := ε δ 1/p , (iii) P = (1 -2δ ′ )P ′ + (2δ ′ )P ′′ and R = (1 -2δ ′ )R ′ + (2δ ′ )R ′′ for some 0 ≤ δ ′ ≤ δ, (iv) |S| ≤ e k ,
(v) and S ⊆ supp(P) if P attains the minimum in (C.15) with S ⊆ supp(R) otherwise.
It is helpful to briefly recall the construction of these sets. Beginning with δ, η as parameters for the approximate covering numbers, the distribution Q concentrates 1 -δ of the mass of P into the centres of the η-radius balls used.
Then the distributions P ′ , R ′ are the measures P, R within the balls. We now write p := P x * ∼R,A∼A,e∼E,x∼P(•|y,A) [∥x * -x∥ ≥ (c + 2)η + (c + 2)σ] ≤ P x * ∼R,A∼A,e∼E,x∼P(•|y,A) [∥x * -x∥ ≥ (c + 2)η + (c + 1)σ + ε ′ ]
≤ 2δ ′ + (1 -2δ ′ )P x * ∼R ′ ,A∼A,e∼E,x∼P(•|y,A) [∥x * -x∥ ≥ (c + 1)(η + σ) + ε ′ ] =: 2δ ′ + (1 -2δ ′ )q. (C.19)
Here, in the first inequality we used the fact that σ ≥ ε ′ , and in the second, we used the decomposition R = (1 -2δ ′ )R ′ + 2δ ′ R ′′ and the fact that δ ′ ≤ δ. We now bound q by using Lemma C.6 to replace the distribution R ′ by the distribution P ′ . Writing u = Az * + e, this lemma and (ii) give that We now bound r. Observe first that
q ≤C abs (ε ′ , tε ′ ; A, D) + D upp (c ′ σ; E) + D shift (tε ′ , c ′ σ; E)
W ∞ (P ′ , Q) ≤ η ′ := η + ε ′ . Indeed, from (i) either W ∞ (P ′ , Q) ≤ η or W ∞ (R ′ , Q) ≤ η.
In the former case, the inequality trivially holds. In the latter case, we can use the triangle inequality and (ii) to obtain the desired bound. This implies that there is a coupling Γ of P ′ , Q with esssup Γ ∥x -y∥ ≤ η ′ . Fix z ∈ S and, for any Borel set E ⊆ R n , define
Γ z (E) = Γ(E, z) Q(z) .
Then it is readily checked that Γ z (•) defines a probability measure. Note also that Γ z is supported on a ball of radius η ′ around z, since esssup Γ (∥x -y∥) ≤ η ′ . Recall that Γ is a coupling between P ′ and Q. Let E ⊆ R n be a Borel set. Then Lemma C.3 gives that
P ′ (E) = Γ(E, R n ) = z∈S Γ((E, R n ) z , z) = z∈S Γ(E, z) = z∈S Γ z (E)Q(z).
Therefore, we can express P ′ as the mixture
P ′ (•) = z∈S Γ z (•)Q(z).
Define the event E = {∥z * -ẑ∥ ≥ (c + 1)(η + σ)} ⊆ R n × R n so that the probability r defined in (C.20) can be expressed as
r = E z * ∼P ′ ,A∼A,e∼E,ẑ∼P(•|A,u) [1 E ].
Using the above expression for P ′ we now write
r = 1 E (z * , ẑ) dP(•|A, u)(ẑ) dE(e) dA(A) dP ′ (z * ) = 1 E (z * , ẑ) dP(•|A, u)(ẑ) dE(e) dA(A) d z∈S Q(z)Γ z (•) (z * ) = z∈S Q(z) 1 E (z * , ẑ) dP(•|A, u)(ẑ) dE(e) dA(A) dΓ z (z * ) ,
where the last line holds as
Q(z) is a constant. Hence r = z∈S Q(z)P z * ∼Γz,A∼A,e∼E,ẑ∼P(•|A,u) [E]. (C.21)
Now we bound each term in this sum. We do this by decomposing P into a mixture of three probability measures depending on z ∈ S. To do this, let θ = c(η + σ) and observe that, for any Borel set E ⊆ R n ,
P(E) = P(E ∩ B θ (z)) + P(E ∩ B c θ (z)) = P(E ∩ B θ (z)) + P(E ∩ B c θ (z)) + (1 -2δ ′ )Q(z)Γ z (E) -(1 -2δ ′ )Q(z)Γ z (E) = P(E ∩ B θ (z)) -(1 -2δ ′ )Q(z)Γ z (E ∩ B θ (z)) + P(E ∩ B c θ (z)) -(1 -2δ ′ )Q(z)Γ z (E ∩ B c θ (z)) + (1 -2δ ′ )Q(z)Γ z (E).
this section cite: ['b48']

Section: Now define the constants
c z,mid = P(B θ (z)) -(1 -2δ ′ )Q(z)Γ z (B θ (z)), c z,ext = P(B c θ (z)) -(1 -2δ ′ )Q(z)Γ z (B c θ (z)
). and let
P z,int (E) = Γ z (E) P z,mid (E) = 1 c z,mid (P(E ∩ B θ (z * )) -(1 -2δ ′ )Q(z * )Γ z (E ∩ B θ (z * ))) , P z,ext (E) = 1 c z,ext (P(E ∩ B c θ (z * )) -(1 -2δ ′ )Q(z * )Γ z (E ∩ B c θ (z * ))).
Then P can be expressed as the mixture P = (1 -2δ ′ )Q(z)P z,int + c z,mid P z,mid + c z,ext P z,ext .
this section cite: []

Section: (C.22)
To ensure this is a well-defined mixture, we need to show that P z,mid and P z,ext are probability measures. However, by (iii) we have, for any Borel set E ⊆ R n ,
P(E) ≥ (1 -2δ ′ )P ′ (E) = (1 -2δ ′ ) z∈S Γ z (E)Q(z) ≥ (1 -2δ ′ )Γ z (E)Q(z).
Therefore, P z,mid and P z,ext are well-defined, provided the constants c z,int , c z,ext > 0. However, if one of these constants is zero, then we can simply exclude this term from the mixture (C.22). For the rest of the theorem, we will assume that, at least, c z,ext > 0.
It is now useful to note that supp(P z,mid ) ⊆ B θ (z) and supp(P z,ext ) ⊆ B c θ (z),
this section cite: []

Section: References
Ref_id:b0 Title: Solving inverse problems with score-based generative priors learned from noisy data Year: (2023)
Ref_id:b1 Title: CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions Year: (2023)
Ref_id:b2 Title: A unified framework for learning with nonlinear model classes from arbitrary linear samples Year: (2024)
Ref_id:b3 Title: Joint sparse recovery based on variances Year: (2019)
Ref_id:b4 Title: Compressive Imaging: Structure, Sampling, Learning Year: (2021)
Ref_id:b5 Title: Breaking the coherence barrier: a new theory for compressed sensing Year: (2017)
Ref_id:b6 Title: Deep Bayesian inversion Year: (2025)
Ref_id:b7 Title: Compressed sensing for inverse problems and the sample complexity of the sparse Radon transform Year: ()
Ref_id:b8 Title: Compressed sensing for inverse problems ii: applications to deconvolution, source recovery, and MRI Year: (2025)
Ref_id:b9 Title: Lectures on Optimal Transport Year: (2024)
Ref_id:b10 Title: On instabilities of deep learning in image reconstruction and the potential costs of AI Year: (2020)
Ref_id:b11 Title: Solving inverse problems using datadriven models Year: (2019)
Ref_id:b12 Title: Model-based compressive sensing Year: (1982)
Ref_id:b13 Title: The dynamics of message passing on dense graphs, with applications to compressed sensing Year: (2011)
Ref_id:b14 Title: Applications, promises, and pitfalls of deep learning for fluorescence image reconstruction Year: (2019)
Ref_id:b15 Title: A coherence parameter characterizing generative compressed sensing with fourier measurements Year: (2022)
Ref_id:b16 Title: Model-adapted Fourier sampling for generative compressed sensing Year: (2023)
Ref_id:b17 Title: On hallucinations in tomographic image reconstruction Year: (2021)
Ref_id:b18 Title: Simultaneous analysis of Lasso and Dantzig selector Year: (2009)
Ref_id:b19 Title: An analysis of block sampling strategies in compressed sensing Year: (2016)
Ref_id:b20 Title: Bayesian inversion for nonlinear imaging models using deep generative priors Year: (2023)
Ref_id:b21 Title: Compressed sensing using generative models Year: (2017)
Ref_id:b22 Title: Fundamental performance limits for ideal decoders in high-dimensional linear inverse problems Year: (2014)
Ref_id:b23 Title: Compressed sensing with structured sparsity and structured acquisition Year: (2019)
Ref_id:b24 Title: Learning in image reconstruction: A cautionary tale Year: (2024-10)
Ref_id:b25 Title: A probabilistic and RIPless theory of compressed sensing Year: (2011)
Ref_id:b26 Title: Compressed sensing and parallel acquisition Year: (2017)
Ref_id:b27 Title: Score-based diffusion models for accelerated mri Year: (2022)
Ref_id:b28 Title: The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and smale's 18th problem Year: (2022)
Ref_id:b29 Title: The bayesian approach to inverse problems Year: (2017)
Ref_id:b30 Title: Introduction to compressed sensing Year: (2012)
Ref_id:b31 Title: Modeling sparse deviations for compressed sensing using generative models Year: (2018)
Ref_id:b32 Title: Deep generative models and inverse problems Year: (2022)
Ref_id:b33 Title: Dimensionality reduction with subgaussian matrices: a unified theory Year: (2016)
Ref_id:b34 Title: Message-passing algorithms for compressed sensing Year: (2009)
Ref_id:b35 Title: Structured compressed sensing: from theory to applications Year: (2011)
Ref_id:b36 Title: Efficient and accurate estimation of Lipschitz constants for deep neural networks Year: (2019)
Ref_id:b37 Title: Score-based diffusion models as principled priors for inverse imaging Year: (2023)
Ref_id:b38 Title:  Year: (2013)
Ref_id:b39 Title: Solving inverse problems with deep neural networksrobustness included? Year: (2023)
Ref_id:b40 Title: A class of Wasserstein metrics for probability distributions Year: (1984)
Ref_id:b41 Title: The troublesome kernel -on hallucinations, no free lunches and the accuracy-stability trade-off in inverse problems Year: (2025)
Ref_id:b42 Title: Phase retrieval under a generative prior Year: (2018)
Ref_id:b43 Title: Global guarantees for enforcing deep generative priors by empirical risk Year: (2018)
Ref_id:b44 Title: The promise and peril of deep learning in microscopy Year: (2021)
Ref_id:b45 Title: Bayesian imaging with data-driven priors encoded by neural networks Year: (2022)
Ref_id:b46 Title: Some investigations on robustness of deep learning in limited angle tomography Year: (2018)
Ref_id:b47 Title: Robust compressed sensing mri with deep generative priors Year: (2021)
Ref_id:b48 Title: Instance-optimal compressed sensing via posterior sampling Year: (2021)
Ref_id:b49 Title: Robust compressed sensing using generative models Year: (2020)
Ref_id:b50 Title: Stochastic solutions for linear inverse problems using the prior implicit in a denoiser Year: (2021)
Ref_id:b51 Title: Unrolled denoising networks provably learn to perform optimal Bayesian inference Year: (2024)
Ref_id:b52 Title: Denoising diffusion restoration models Year: (2022)
Ref_id:b53 Title: SNIPS: solving noisy inverse problems stochastically Year: (2021)
Ref_id:b54 Title: Avoiding a replication crisis in deep-learning-based bioimage analysis Year: (2021)
Ref_id:b55 Title: Bayesian imaging using plug & play priors: When langevin meets tweedie Year: (2022)
Ref_id:b56 Title: The medical algorithmic audit Year: (2022)
Ref_id:b57 Title: Bayesian MRI reconstruction with joint uncertainty estimation using diffusion models Year: (2023)
Ref_id:b58 Title: Spectral normalization for generative adversarial networks Year: (2018)
Ref_id:b59 Title: Adversarial robustness of MR image reconstruction under realistic perturbations Year: (2022)
Ref_id:b60 Title: Results of the 2020 fastMRI challenge for machine learning MR image reconstruction Year: (2021)
Ref_id:b61 Title: NESTANets: stable, accurate and efficient neural networks for analysis-sparse inverse problems Year: (2023)
Ref_id:b62 Title: Complexities of deep learning-based undersampled MR image reconstruction Year: (2023)
Ref_id:b63 Title: Deep learning techniques for inverse problems in imaging Year: (2020)
Ref_id:b64 Title: Improving robustness of deep-learning-based image reconstruction Year: (2020-07)
Ref_id:b65 Title: Reader and B. Pan. AI for PET image reconstruction Year: (1150)
Ref_id:b66 Title: Compressive sensing by random convolution Year: (2009)
Ref_id:b67 Title: Theoretical perspectives on deep learning methods in inverse problems Year: (2022)
Ref_id:b68 Title: Solving linear inverse problems using GAN priors: An algorithm with provable guarantees Year: (2018)
Ref_id:b69 Title: Solving inverse problems in medical imaging with score-based generative models Year: (2022)
Ref_id:b70 Title: Inverse problems: a Bayesian perspective Year: (2010)
Ref_id:b71 Title: Intriguing properties of neural networks Year: (2014)
Ref_id:b72 Title: Stable recovery of low-dimensional cones in Hilbert spaces: one RIP to rule them all Year: (2018)
Ref_id:b73 Title: Machine learning for medical imaging: methodological failures and recommendations for the future Year: (2022)
Ref_id:b74 Title: Lipschitz regularity of deep neural networks: analysis and efficient estimation Year: (2018)
Ref_id:b75 Title: How medical AI devices are evaluated: limitations and recommendations from an analysis of FDA approvals Year: (2021)
Ref_id:b76 Title: Validation and generalizability of self-supervised image reconstruction methods for undersampled MRI Year: (2022)
Ref_id:b77 Title: Stable deep MRI reconstructions using generative priors Year: (2023)
