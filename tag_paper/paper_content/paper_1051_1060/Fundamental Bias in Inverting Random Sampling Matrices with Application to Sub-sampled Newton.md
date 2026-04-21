Title: Fundamental Bias in Inverting Random Sampling Matrices with Application to Sub-sampled Newton
Abstract: A substantial body of work in machine learning (ML) and randomized numerical linear algebra (RandNLA) has exploited various sorts of random sketching methodologies, including random sampling and random projection, with much of the analysis using Johnson-Lindenstrauss and subspace embedding techniques. Recent studies have identified the issue of inversion bias -the phenomenon that inverses of random sketches are not unbiased, despite the unbiasedness of the sketches themselves. This bias presents challenges for the use of random sketches in various ML pipelines, such as fast stochastic optimization, scalable statistical estimators, and distributed optimization. In the context of random projection, the inversion bias can be easily corrected for dense Gaussian projections (which are, however, too expensive for many applications). Recent work has shown how the inversion bias can be corrected for sparse sub-gaussian projections. In this paper, we show how the inversion bias can be corrected for random sampling methods, both uniform and nonuniform leverage-based, as well as for structured random projections, including those based on the Hadamard transform. Using these results, we establish problem-independent local convergence rates for sub-sampled Newton methods.

Section: Introduction
Randomized numerical linear algebra (RandNLA) significantly reduces computation, communication, and/or storage overheads by using randomness as an algorithmic resource. As a pivotal technique in RandNLA, random sketchingwhich encompasses both random projection and random sampling -is becoming increasingly critical in many modern large-scale machine learning (ML) applications.
More precisely, for a tall matrix A ∈ R n×d with n ≫ d, random projection proposes to obtain a sketch Ã ∈ R m×d , of size m ≪ n of A by randomly and linearly combining the rows of A; while random sampling, on the other hand, carefully selects a small subset (of size m say) of the rows A and rescales them to obtain Ã ∈ R m×d . For both approaches, it follows from Johnson-Lindenstrauss (JL) type analysis (Johnson & Lindenstrauss, 1984) that the random sketch Ã can be used as a "proxy" of A in many downstream ML tasks, e.g., Ã⊤ Ã ≈ A ⊤ A are close in some sense with high probability, even with m ≪ n. This leads to a significant boost in the running time, communication time, and/or memory cost, for many numerical methods (Drineas et al., 2006b;2011;2012;Avron et al., 2017;Dereziński & Mahoney, 2021;Lacotte & Pilanci, 2022). See also Mahoney (2011); Halko et al. (2011); Woodruff (2014); Drineas & Mahoney (2018); Martinsson & Tropp (2020); Dereziński & Mahoney (2021); Murray et al. (2023); Dereziński & Mahoney (2024) and reference therein for an overview of RandNLA and the applications in modern ML.
Despite this promising "complexity-accuracy" trade-off achieved with random sketching, in many ML pipelines ranging from linear/ridge regression to scalable statistical estimation and fast stochastic optimization, the object of direct interest is the sketched matrix inverse ( Ã⊤ Ã+C) -1 for some (perhaps all-zeros or diagonal) positive semi-definite (p.s.d.) C (instead of Ã⊤ Ã itself). Since the matrix inverse is a nonlinear operator, the fact that Ã⊤ Ã is an unbiased or nearly unbiased estimator of A ⊤ A (i.e., E[ Ã⊤ Ã] = A ⊤ A, which is a major guide in forming the sketch Ã) does not, in general, imply the unbiasedness of its inverse, i.e., E[( Ã⊤ Ã) -1 ] ̸ ≈ (A ⊤ A) -1 . This phenomenon of inversion bias has been long known in the literature: in the case of Gaussian random projection with Ã = SA for S ∈ R m×n having i.i.d. N (0, 1/m) entries, ( Ã⊤ Ã) -1 is known to follow the inverse Wishart distribution (Haff, 1979) with E[( Ã⊤ Ã) -1 ] = m m-d-1 (A ⊤ A) -1 for m > d + 1. However, much less is known beyond the Gaussian setting. Build-ing upon recent progress in non-asymptotic random matrix theory (RMT), it has recently been shown by Dereziński et al. (2021b) that a similar inversion bias holds, and that it can be corrected, with E[( Ã⊤ Ã) -1 ] ≈ m m-d (A ⊤ A) -1 , in the sense of partial order of p.s.d. matrices, for sub-gaussian and the so-called LEverage Score Sparsified (LESS) projections. The latter can be of greater practical interest than, e.g., dense Gaussian or sub-gaussian projections, as it is significantly sparser and can thus be evaluated more efficiently. This precise characterization of inversion bias has direct implications for RandNLA and ML. As a telling example, Dereziński et al. (2021a) applied LESS sketches to Newton Sketch (Pilanci & Wainwright, 2017) and showed that this Newton-LESS approach enjoys almost the same local convergence rates as Newton Sketch with dense Gaussian projections. This leads to a significantly better "complexityconvergence" trade-off than the vanilla Gaussian projections in stochastic second-order optimization methods.
this section cite: ['b29', 'b24', 'b2', 'b33', 'b41', 'b28', 'b61', 'b20', 'b42', 'b44', 'b15', 'b27', 'b49']

Section: Our Contributions
In this paper, we consider the inversion bias of random sampling, including uniform and non-uniform sampling, as well as structured random projections such as the Subsampled Randomized (Walsh-)Hadamard Transform (SRHT) (Ailon & Chazelle, 2006). The analysis framework in Dereziński et al. (2021a;b) does not apply (to get non-vacuous results; see, e.g., Proposition 2.8 and Remark 2.9 below). Instead, we exploit novel and non-trivial connections between nonasymptotic RMT and RandNLA to show that this inversion bias can be precisely characterized and numerically corrected. We also show how this inversion bias result can be used to improve the local convergence rates of the popular sub-sampled Newton (SSN) method (Yao et al., 2018;Roosta-Khorasani & Mahoney, 2019;Xu et al., 2020).
Our main contributions can be summarized as follows.
1. We provide precise characterization of the inversion bias for general random sampling (in Theorem 3.1) and the corresponding de-biased approach (in Proposition 3.2). The proposed analysis and debiasing technique hold for exact and approximate leverage-based sampling (Corollary 3.4), as well as the structured SRHT (Corollary 3.7) as special cases.
2. With this precise inversion bias result, we further establish (in Theorem 4.3), the first problem-independent local convergence rates for sub-sampled Newton that approximately matches the dense Gaussian Newton Sketch scheme. Numerical results are provided in Section 5 to support these findings.
this section cite: ['b0', 'b64', 'b53', 'b63']

Section: Related Work
Inversion bias. Given a matrix A ∈ R n×d , the matrix inverse (A ⊤ A) -1 is fundamental in ML, numerical computation, and statistics. Examples include linear functions (A ⊤ A) -1 y that are crucial to Newton's methods (Boyd & Vandenberghe, 2004), quadratic forms a ⊤ i (A ⊤ A) -1 a i (with a ⊤ i the i th row of A) in computing matrix leverage scores (Drineas et al., 2012), and trace forms, tr L(A ⊤ A) -1 for some given L, of interest in uncertainty quantification (Kalantzis et al., 2013) and experimental designs (Pukelsheim, 2006). In the case of tall matrices with n ≫ d, random sketching applies to efficiently reduce the computational overhead of (A ⊤ A) -1 , by using a sketch Ã = SA of A for random matrix S ∈ R m×n with m ≪ n. It has been shown recently (Dereziński et al., 2020;2021a;b) that these sketched inverses are biased for unstructured subgaussian S, with E[( Ã⊤ Ã) -1 ] ̸ ≈ (A ⊤ A) -1 . In this paper, we consider the (often practically more interesting) case of structured random matrix S, including random sampling matrices (Definition 2.1) and randomized Hadamard transforms (Definition 3.6).
Different from our approach that explicitly modifies the sketch to correct the bias, another line of work proposes to use shrinkage-based correction techniques (Zhang & Pilanci, 2023;Romanov et al., 2024).
this section cite: ['b5', 'b24', 'b30', 'b51', 'b16', 'b68', 'b52']

Section: Random sampling.
Random sampling is at the core of RandNLA (Drineas et al., 2006a;Mahoney, 2011;Ma et al., 2015;Drineas & Mahoney, 2016;Dereziński & Mahoney, 2021;2024), and it plays a central role in fast matrix multiplication (Drineas et al., 2006a), approximate regression (Drineas et al., 2006b), and low-rank approximation (Cohen et al., 2017), to name a few. It is of particular interest in scenarios where the dataset is massive and cannot be stored and/or computed on a single machine, e.g., the census data (Wang et al., 2018) and online network data (Deng et al., 2024). See Definition 2.1 below for a formal definition of random sampling and discussions thereafter for commonly-used sampling schemes including (exact and approximate) leverage score sampling (Mahoney, 2011;Cohen et al., 2017), shrinkage leverage sampling (Ma et al., 2015), as well as optimal subsampling (Wang et al., 2018;Wang & Ma, 2021;Yu et al., 2022;Ma et al., 2022). In this paper, instead of providing classical JL and subspace embedding-type results on random sampling, we precisely characterize (and correct) the inversion bias for a variety of commonly-used random sampling schemes.
Sub-sampled Newton. Sub-sampled Newton (SSN) methods propose to approximate the Hessian in Newton's method using a small subset of samples, and they have been extensively studied within the fields of ML, RandNLA, and optimization (Xu et al., 2016;Bollapragada et al., 2019;Roosta-Khorasani & Mahoney, 2019;Xu et al., 2020;Ye et al., 2021). Although these fast optimization methods are easy to implement, their convergence rates are challenging to analyze. Existing results often depend on the Hessian condition number or the Lipschitz constant and fall short of, e.g., the problem-independent convergence rates achieved by sub-gaussian Newton Sketch (Lacotte & Pilanci, 2019;Dereziński et al., 2021a). In this paper, we establish the first problem-independent local convergence rates for SSN that closely align with Newton Sketch. This addresses the convergence guarantee gap identified in Iterative Hessian Sketch (Pilanci & Wainwright, 2016) for random sampling.
Random matrix theory (RMT). RMT studies the (limiting) eigenspectra of large-dimensional random matrices (Anderson et al., 2010) and finds its applications in signal processing and communication (Couillet & Debbah, 2011), statistical finance (Plerou et al., 2002), optimization (Paquette et al., 2021;2023), and more recently in large-scale ML (Pennington & Worah, 2017;Fan & Wang, 2020;Mei & Montanari, 2022;Couillet & Liao, 2022). A recent line of work (Liao et al., 2020;2021;Liao & Mahoney, 2021) has highlighted non-trivial connections between RMT and RandNLA that this paper further develops.
this section cite: ['b41', 'b39', 'b19', 'b9', 'b60', 'b13', 'b41', 'b9', 'b39', 'b60', 'b59', 'b66', 'b40', 'b62', 'b3', 'b53', 'b63', 'b65', 'b32', 'b48', 'b1', 'b11', 'b50', 'b45', 'b47', 'b26', 'b43', 'b12', 'b37']

Section: Notations and Organization of the Paper
We denote scalars by lowercase letters, vectors by bold lowercase, and matrices by bold uppercase. For a matrix A ∈ R n×d , we denote A ⊤ , a ⊤ i ∈ R d , and ∥A∥ the transpose, i th row, and spectral norm of A, respectively. We denote A ⪯ B if B -A is p.s.d., and use I d for the identity matrix of size d. For a vector x ∈ R d and a matrix B ∈ R d×d , we denote ∥x∥ B ≡ √
x ⊤ Bx, with the convention ∥x∥ = ∥x∥ I d . For a random variable x, we denote E[x] the expectation of x and E ζ [x] the expectation of x, conditional on the event ζ. We use Θ(•), and O(•) notations as in standard computer science literature.
The remainder of this paper is organized as follows. Section 2 presents preliminaries on random sampling and a coarse-grained characterization of its inversion bias, by directly (and naively) adapting the proof approach from Dereziński et al. (2021b) (which turns out to be vacuous in our setting). Section 3 delivers a fine-grained analysis of this inversion bias and proposes an efficient non-vacuous debiasing approach. Section 4 demonstrates how these technical results apply to establish problem-independent local convergence rates for SSN. Section 5 provides numerical results that support our theoretical findings. Section 6 provides a conclusion, summarizing our findings and discussing future perspectives. Additional material can be found in the appendices.
this section cite: []

Section: Preliminaries on Random Sampling
In this section, we introduce a few definitions that will be used in the remainder of this paper.
Definition 2.1 (Random sampling). For a matrix A ∈ R n×d with n ≥ d, a sketch Ã ∈ R m×d of A can be constructed by randomly sampling with replacement m from the n rows of A with an importance sampling distribution, {π i } n i=1 , n i=1 π i = 1, and then rescaling by 1/ √ mπ i . This can be expressed as Ã = SA, with random sampling matrix S ∈ R m×n having only one nonzero entry per row.
Definition 2.1 includes commonly-used random sampling schemes such as uniform sampling (with
π i = 1/n), row- norm-based sampling (with π i = ∥a i ∥ 2 /( n i=1 ∥a i ∥ 2
)), exact or approximate leverage and ridge leverage score sampling (Mahoney, 2011;El Alaoui & Mahoney, 2015) defined below, as well as a mix between them, e.g., the shrinkage leverage sampling (Ma et al., 2015).
Definition 2.2 (Leverage score sampling, Mahoney (2011)). For a matrix A ∈ R n×d of rank d with n ≥ d and a p.s.d. matrix C ∈ R d×d , the i th leverage score ℓ C i of A given C, is defined as
ℓ C i = a ⊤ i (A ⊤ A + C) -1 a i , i ∈ {1, . . . , n}.
The exact leverage score sampling refers to the random sampling approach in Definition 2.1 with
π i = ℓ C i /d eff , for d eff = n i=1 ℓ C
i the effective dimension of A given C.
The leverage score sampling has been extensively studied in RandNLA and ML. By taking C = 0 d in Definition 2.2, we obtain the standard leverage scores (Mahoney, 2011;Drineas et al., 2012); and by taking C = λI d , we obtain the λ-ridge leverage scores (El Alaoui & Mahoney, 2015). Given A ∈ R n×d , its leverage scores can be approximately computed in O(nnz(A) log n + d 3 (log d) 2 + d 2 log n) time, for nnz(A) the number of non-zero entries in A, see Drineas et al. (2012); Clarkson & Woodruff (2017); Cohen et al. (2017).
We also introduce an "approximation factor" to measure the extent to which one importance sampling distribution approximates another importance sampling distribution (the exact leverage score distribution, in our case).
Definition 2.3 (Importance sampling approximation factor). For a matrix A ∈ R n×d with n ≥ d, a p.s.d. matrix C ∈ R d×d , and a random sampling matrix S ∈ R m×n as in Definition 2.1, with importance sampling distribution {π i } n i=1 , the (min and max) importance sampling approximation factors of the random sampling scheme S is defined as the pair (ρ min , ρ max ), with
ρ min ≡ min 1≤i≤n ℓ C i /(π i d eff ) and ρ max ≡ max 1≤i≤n ℓ C i /(π i d eff ).
For us, the importance sampling approximation factors (ρ min , ρ max ) in Definition 2.3 provide qualitative characterization on how the random sampling scheme under study differs from the exact leverage score sampling in Definition 2.2. This extends the classical notion of sampling approximation factor in Drineas et al. (2006a) to include both the maximum and the minimum. While prior work primarily focuses on the max factor, here we focus on the inversion bias, where the min factor also plays a natural and significant role; see below in Section 3 and also Ma et al. (2015), who first noted the importance of the min factor in statistical style analysis. By the (generalized) median inequality, we have ρ min ≤ 1 ≤ ρ max , with equality for exact leverage score sampling.
It follows from Definition 2.1 that E[ Ã⊤ Ã] = A ⊤ A, so that the randomly sampled matrix Ã⊤ Ã is an unbiased estimator of the true A ⊤ A. This, together with controls on the higher-order moments of Ã⊤ Ã, allows one to conclude that Ã⊤ Ã fluctuates, with high probability and within a small "distance," around the true A ⊤ A of interest. This can be made precise using the relative error approximation (for scalars and matrices) defined as follows.
Definition 2.4 (Relative error approximation). For a nonnegative scalar x ≥ 0, we say x is an ϵ-approximation of (another scalar) x, denoted x ≈ ϵ x, if
(1 + ϵ) -1 x ≤ x ≤ (1 + ϵ)x.(1)
For x being random, we say x is an (ϵ, δ)-approximation of x if (1) holds with probability at least 1 -δ. Similarly, for a p.s.d. matrix X, we say X is an ϵ-approximation (or an (ϵ, δ)-approximation when being random) if
X ≈ ϵ X ⇔ (1 + ϵ) -1 X ⪯ X ⪯ (1 + ϵ)X.(2)
Remark 2.5 (Subspace embedding). For Ã a sketch of A, the property that Ã⊤ Ã ≈ ϵ A ⊤ A holds with probability 1 -δ is known as (ϵ, δ)-subspace embedding in RandNLA. This concept was introduced by Drineas et al. (2006b); see also Mahoney (2011) for a history. It was subsequently used in data-oblivious form by Sarlós (2006); Drineas et al. (2011), and then popularized in data-oblivious form (and mis-attributed to Sarlós (2006)) by Woodruff (2014). It plays a central role in the statistical characterization of random sketching techniques.
The focus of this paper is to go beyond the subspace embedding-type results in Definition 2.4 and Remark 2.5, and to assess the inversion bias of the form E[( Ã⊤ Ã + C) -foot_0 ] (versus the true inverse (A ⊤ A + C) -1 ). To this end, we need the following measure of unbiased estimators.
Definition 2.6 (Unbiased estimator). We say a random p.s.d. matrix X is an (ϵ, δ)-unbiased estimator of X if, conditioned on an event ζ that happens with probability at least 1 -δ,
(1 + ϵ) -1 X ⪯ E ζ [ X] ⪯ (1 + ϵ)X, and X ⪯ O(1)X. (3)
Note that the error parameter ϵ in subspace embeddings quantifies spectral approximation error, whereas the no-tion of "unbiasedness" specifically refers to inversion bias. Furthermore, while subspace embeddings automatically ensure ϵ-unbiasedness up to the same level of error ϵ, they are generally not guaranteed to remain unbiased for a smaller ϵ (Dereziński et al., 2021b).
With these definitions and notations at hand, we are ready to assess the statistical properties of random sampling. A first quantity of interest to the design of random sampling is m, the number of trials needed to construct an (ϵ, δ)subspace embedding, for some given importance sampling distribution {π i } n i=1 . A slightly more general result is given as follows. 1   Lemma 2.7 (Subspace embedding for random sampling). Given A ∈ R n×d of rank d with n ≥ d and p.s.d. C ∈ R d×d , let S be a random sampling matrix with number of trials m and importance sampling distribution {π i } n i=1 as in Definition 2.1, and let d eff = tr(A ⊤ C A C ) be the effective dimension of A given C with A C ≡ A(A ⊤ A + C) -1/2 . Then, there exists C > 0 independent of n, d eff such that for m ≥ Cρ max d eff log(d eff /δ)/ϵ 2 , failure probability δ ∈ (0, 1/2), ϵ > 0, and ρ max in Definition 2.3,
A ⊤ C S ⊤ SA C is an (ϵ, δ)-approximation of A ⊤ C A C .
The proof of Lemma 2.7 uses standard matrix concentration techniques and is given in Appendix B. Note that Lemma 2.7 includes existing results of both leverage (C = 0 d ) and ridge leverage score (C = λI d ) sampling as special cases, see Chowdhury et al. (2018, Theorm 3).
With Lemma 2.7, we are now ready to evaluate the inversion bias of random sampling. Since the matrix inverse is nonlinear, for
A ⊤ S ⊤ SA with E[A ⊤ S ⊤ SA] = A ⊤ A, one should, a priori, not expect that (A ⊤ S ⊤ SA) -1 is an unbiased or nearly unbiased estimator of (A ⊤ A) -1 .
In the following result, we show (by adapting, in an almost straightforward fashion, the scalar debiasing proof approach of Dereziński et al. (2021a;b)) that this inversion bias can be corrected, but only to some extent, using the same scalar factor as for sub-gaussian or LESS projections. The proof of Proposition 2.8 is given in Appendix C for completeness.
Proposition 2.8 (Coarse-grained debiasing of random sampling). Given A ∈ R n×d of rank d with n ≥ d and p.s.d. C ∈ R d×d , let S ∈ R m×n be a random sampling matrix with importance sampling distribution {π i } n i=1 as in Definition 2.1 and max importance sampling approximation factor ρ max as in Definition 2.3. Then, there exists C > 0 indepen-
dent of n, d eff such that if m ≥ Cρ max d eff (log(d eff /δ) + √ d eff /ϵ) with δ ≤ m -3 , ( m m-d eff A ⊤ S ⊤ SA + C) -1 is an (ϵ, δ)-unbiased estimator of (A ⊤ A + C) -1 .
Remark 2.9 (On Proposition 2.8). While the debiasing factor m m-d eff is the same as that proposed for random (e.g., subgaussian or LESS) projections (Dereziński et al., 2021a;b), the resulting inversion bias is significantly larger. In particular, we have, in the case of Proposition 2.8 and for m = Θ(ρ max d eff log d eff ), that ( m m-d eff A ⊤ S ⊤ SA + C) -1 has an inversion bias of order O( √ d eff / log d eff ). This is a vacuous bound. It follows from Lemma 2.7 that for the same choice of m, (A ⊤ S ⊤ SA + C) -1 is, without the debiasing factor, an (O(1), δ)-approximation of (A ⊤ A + C) -1 , and thus has an inversion bias of order O(1). See Lemma B.1 in Appendix B for a proof of this fact.
From Remark 2.9, it appears that the inversion bias result in Proposition 2.8 is disappointing: random sampling, in contrast with sub-gaussian or LESS random projections, while being numerically attractive and easy to implement, does not lead to a small inversion bias, at least with the m m-d eff debiasing factor and the proof approach in Dereziński et al. (2021a;b) under Proposition 2.8. One may thus wonder: Is it possible to get sharper control on the inversion bias of random sampling, either by introducing a different debiasing scheme and/or by using a more refined proof than Proposition 2.8?
Below, we show that such improvement is indeed possible.
this section cite: ['b41', 'b25', 'b39', 'b41', 'b41', 'b24', 'b25', 'b24', 'b9', 'b39', 'b41', 'b54', 'b23', 'b54', 'b61']

Section: Fine-grained Analysis of Inversion Bias for Random Sampling
We have seen in Proposition 2.8 and Remark 2.9 that the scalar debiasing and the proof approach in Dereziński et al. (2021a;b) do not lead, in the case of random sampling, to a non-vacuous small inversion bias. In the following result, we provide fine-grained analysis of the inversion bias of random sampling (finer than that in Proposition 2.8), and we show that the inverse (A ⊤ S ⊤ SA + C) -1 for random sampling S is biased in a more involved fashion than random projections studied in Dereziński et al. (2021a;b).
Theorem 3.1 (Inversion bias for random sampling: finegrained analysis). Given A ∈ R n×d of rank d with n ≥ d and p.s.d. C ∈ R d×d , let S ∈ R m×n be a random sampling matrix with importance sampling distribution {π i } n i=1 as in Definition 2.1 and (ρ min , ρ max ) as in Definition 2.3. Then, for diagonal matrix D = diag{D ii } n i=1 the solution to 2
D ii = m m + a ⊤ i (A ⊤ DA + C) -1 a i /π i ,(4)
there exists C > 0 independent of n, d eff so that for m ≥
Cρ max d eff (log(d eff /δ) + 1/ϵ 2/3 ), δ ≤ m -3 , (A ⊤ S ⊤ SA + C) -1 is an (ϵ, δ)-unbiased estimator of (A ⊤ DA + C) -1 .
2 It can be checked that
m m+2ρmaxd eff In ⪯ D ⪯ m m+ρ min d eff In with ρmin, ρmax in Definition 2.3. See Lemma D.3 in Appendix D.
this section cite: []

Section: Heuristic derivation of Theorem 3.1.
For a more transparent understanding of the self-consistent equation in (4) of Theorem 3.1, we provide here a heuristic derivation. The detailed proof of Theorem 3.1 is deferred to Appendix D. Denote
x ⊤ s = e ⊤ is A/ √ π is , Q = (A ⊤ S ⊤ SA + C) -1 = 1 m m s=1 x s x ⊤ s + C -1 and Q -s = ( j̸ =s 1 m x j x ⊤ j + C) -1 , for which we have m s=1 1 m E[x s x ⊤ s ] = n i=1 a i a ⊤ i = A ⊤ A.
Then, we follow the deterministic equivalent framework (see, e.g., Couillet & Liao (2022, Chapter 2) for an introduction) and show that ∥E[Q] -H -1 ∥ ≃ 0, for H = A ⊤ DA + C, for D ∈ R n×n given in (4). First, note that ∥E[Q] -
H -1 ∥ = ∥E[Q]A ⊤ DA H -1 -E[QA ⊤ S ⊤ SA] H -1 ∥, it then fol- lows from Sherman-Morrison formula (Lemma A.3) that E[QA ⊤ S ⊤ SA] H -1 = m s=1 E 1 m Q -s x s x ⊤ s H -1 1 + x ⊤ s Q -s x s /m = n i=1 E Q -s a i a ⊤ i H -1 1 + a ⊤ i Q -s a i /mπ i .
Using the rank-one perturbation lemma of matrix inverse, see, e.g., Silverstein & Bai (1995, Lemma 2.6), we obtain
E[QA ⊤ S ⊤ SA] H -1 ≃ n i=1 E Qa i a ⊤ i H -1 1 + a ⊤ i H -1 a i /mπ i =E [Q] A ⊤ DA H -1 , for D = diag{mπ i /(mπ i + a ⊤ i H -1 a i )} n i=1 .
This leads to the self-consistent equation in (4) of Theorem 3.1.
Theorem 3.1 says that the (conditional) expectation E ζ [(A ⊤ S ⊤ SA + C) -1 ], instead of being close to (A ⊤ A + C) -1 , is in fact close to (A ⊤ DA + C) -1 , with D depending on A and the random sampling scheme per m and {π i } n i=1 in an implicit fashion. While seemingly uninterpretable and unusable at first sight, Theorem 3.1 can be tuned to design a de-biased random sampling approach. This is given in the following result.
Proposition 3.2 (Fine-grained debiasing for general random sampling). Under the settings and notations of Theorem 3.1, for ℓ C is the i th s leverage score of A as in Definition 2.2 and standard random sampling matrix S as in Definition 2.1, define the de-biased sampling matrix Š ∈ R m×n as
Š=diag m/(m -ℓ C is /π is ) m s=1 • S.(5)
Then, there exists constant C > 0 independent of n,
d eff such that for m ≥ Cρ max d eff (log(d eff /δ) + 1/ϵ 2/3 ), δ ≤ m -3 , (A ⊤ Š⊤ ŠA + C) -1 is an (ϵ, δ)-unbiased estimator of (A ⊤ A + C) -1 .
this section cite: []

Section: Heuristic derivation of Proposition 3.2.
To make the intuition behind Proposition 3.2 more accessible, we present here a heuristic derivation of (5). We refer the reader to Appendix E for the detailed proof of Proposition 3.2.
Let Š⊤ Š = m s=1 F isis • e is e ⊤ is /mπ is for some de- terministic F ii to be specified, Q = (A ⊤ Š⊤ ŠA + C) -1 = 1 m m s=1 F isis x s x ⊤ s + C -1 , and similarly Q-s = ( 1 m l̸ =s F i l i l x l x ⊤
l + C) -1 as in the heuristic derivation of Theorem 3.1 above. We thus have
1 m m s=1 E[F isis x s x ⊤ s ] = n i=1 F ii a i a ⊤ i .
Our goal is to determine Q (and
F ii ) such that ∥E[ Q] -H -1 ∥ ≃ 0, for H -1 = (A ⊤ A + C) -1 . To this end, observe that E[ Q] -H -1 =E[ Q]A ⊤ AH -1 -E[ QA ⊤ Š⊤ ŠA]H -1 ≃ 0. By Sherman-Morrison formula (Lemma A.3), we obtain E[ QA ⊤ Š⊤ ŠA]H -1 =E Q-s F isis A ⊤ e is e ⊤ is /π is AH -1 1 + F isis e ⊤ is A Q-s A ⊤ e is /mπ is = n i=1 E Q-s F ii A ⊤ e i e ⊤ i AH -1 1 + F ii e ⊤ i A Q-s A ⊤ e i /mπ i ,
where we see the exact leverage score e
⊤ i AH -1 A ⊤ e i = a ⊤ i (A ⊤ A + C) -1 a i = ℓ C
i as in Definition 2.2 naturally appears in the denominator from the derivation. Invoking the rank-one perturbation lemma once more, we have that
E[ QA ⊤ Š⊤ ŠA]H -1 ≃E[ Q]A ⊤ n i=1 F ii e i e ⊤ i 1 + F ii ℓ C i /mπ i AH -1 =E[ Q]A ⊤ AH -1 ,
where we take the debiasing factor
F ii = mπ i /(mπ i -ℓ C i ) such that F ii /(1 + F ii ℓ C i /mπ i ) = 1.
This leads to the form of the debiasing matrix Š as in (5) of Proposition 3.2.
Comparing the fine-grained results in Proposition 3.2 to the coarse-grained results in Proposition 2.8, we see that the large inversion bias in Proposition 2.8 is indeed a consequence of the proof approach adapted from Dereziński et al. (2021a;b), that is inadequate for random sampling and for structured random projections such as the SRHT. Remark 3.3 ( Š as a random sampling scheme). Note that Š ∈ R m×n in Proposition 3.2 is nothing but another random sampling matrix: it features exactly one nonzero entry per row that is equal to (mπ is -ℓ C is ) -1/2 , as opposed to (mπ is ) -1/2 for the standard random sampling S in Definition 2.1. This non-standard re-weighting (that uses the leverage scores of A) ensures that (A ⊤ Š⊤ ŠA + C) -1 is a nearly unbiased estimate of (A ⊤ A + C) -1 , per Proposition 3.2. From a computational perspective, the re-weighted random sampling Š in (5) can be computationally demanding due to the need for exact computation of leverage scores ℓ C i in (5). In Corollary E.1 of Appendix E, we consider approximate leverage scores (which are much faster to compute (Drineas et al., 2012;Clarkson & Woodruff, 2017;Cohen et al., 2017)). We show that for a given sampling scheme {π i } n i=1 , replacing exact leverage scores with their approximate counterparts in the de-biased sampling matrix Š in (5) increases the inversion bias, but only very slightly.
Note from Proposition 3.2 that the proposed fine-grained de-biasing matrix Š depends on the importance sampling distribution only via ℓ C i /π i . (See Appendix E.1 for the RMT intuition on how the exact leverage scores arise from the derivation.) As such, for any random sampling method with π i ≈ ℓ C i /d eff close to those of exact leverage score sampling in Definition 2.2, we have Š ≈ m m-d eff S. This coincides with the scalar debiasing scheme in the coarse-grained result of Proposition 2.8, but it has a much smaller inversion bias. This special case is discussed in the following result, proven in Appendix E.3.2. Corollary 3.4 (Inversion bias using scalar debiasing under approximate leverage). Under the settings and notations of Theorem 3.1, for random sampling scheme with sampling distribution π
i ∈ [ℓ C i /(d eff ρ max ), ℓ C i /(d eff ρ min )] with ρ min ∈ [1/2, 1] as in Definition 2.3, 3 there exists C > 0, ν ≥ log d eff (log(d eff /δ)), δ < m -3 such that for m ≥ Cρ max d 1+ν eff , ( m m-d eff A ⊤ S ⊤ SA + C) -1 is an (ϵ, δ)-unbiased estimator of (A ⊤ A + C) -1 with inver- sion bias ϵ = max{O(d -3ν/2 eff ), O(ϵ ρ d -ν
eff )} and ϵ ρ = max{ρ -1 min -1, 1 -ρ -1 max }. Remark 3.5 (Inversion bias for exact versus approximate leverage score sampling). It follows from Corollary 3.4 that for exact and/or approximate leverage score sampling with ρ max ≥ ρ min /(2ρ min -1) ≥ 1 and ρ min > (1 + Θ(d
-ν/2 eff )) -1 > 1/2 (so that ϵ ρ = 1 -ρ -1 max )
, the inversion bias induced by the scalar debiasing m m-d eff establishes the following phase transition behavior:
1. if the random sampling scheme is sufficiently close to exact leverage sampling, in that
ρ max ∈ [ρ min /(2ρ min -1), 1/(1 -Θ(d -ν/2eff
))] (or equivalently the importance sampling probabilities satisfy
π i ∈ [(1 ± Θ(d -ν/2 eff ))ℓ C i /d eff ]
), then the inversion bias under scalar debiasing is the same as that (of the fine-grained matrix debiasing) in Proposition 3.2; but 2. if the random sampling scheme significantly deviates from exact leverage sampling with ρ max > 1/(1 -Θ(d
-ν/2 eff )) (or equivalently |π i -ℓ C i /d eff | > Θ(d -ν/2 eff )ℓ C
i /d eff ), then the inversion bias under scalar debiasing becomes larger than that in Proposition 3.2, increases with ρ max , and saturates at ρ max = Θ(1).
This phase transition behavior is visualized in Figure 1. See also Figure 4 in Appendix G for the numerical comparison of inversion bias using scalar debiasing between exact and approximate leverage score sampling.
ρ min 2ρ min -1 1 1-Θ(d -ν/2 eff ) Θ(1) O(d -3ν/2 eff ) O(d -ν eff ) ρmax
Inversion bias ϵ As a side remark, it is known from Dereziński et al. (2021b, Theorem 10) that for approximate leverage sampling, and any scalar γ > 0, m > 0, (γA ⊤ S ⊤ SA + C) -1 with C = 0 d is not an (ϵ, δ)-unbiased estimator of (A ⊤ A + C) -1 , with any ϵ ≤ cd eff /m and c > 0 an absolute constant. Thus, 1. in the case of approximate leverage score sampling with ρ max = 3/2 and ρ min = 1/2, for any m ≥ Cρ max d eff log d eff , it follows from the proof of Corollary 3.4 that the inversion bias is upper bounded by O(d eff /m), and this coincides with the lower bound in Dereziński et al. (2021b, Theorem 10); and 2. in the case of exact leverage score sampling with ρ max = ρ min = 1, the inversion bias can be made smaller than d eff /m under scalar debiasing. 4
As an important consequence, Proposition 3.2 also applies to effectively de-bias another commonly-used data-oblivious sketching scheme, the SRHT (Ailon & Chazelle, 2006).
Definition 3.6 (Sub-sampled randomized Walsh-Hadamard transform, SRHT, Ailon & Chazelle (2006)). For a given matrix A ∈ R n×d of rank d with n ≥ d, assume without loss of generality that n = 2 p for some integer p. Then, the SRHT of A is given by
ÃSRHT = SH n D n A/ √ n ∈ R m×n , (6
) 4 Notably, using exact (instead of approximate) leverage score sampling in the same setting of Dereziński et al. (2021b, Theorem 10), the inversion bias (conditioned on any event ζ that ensures invertibility) can be made zero by taking γ = m d E ζ [1/b] for b distributed as Binomial(m, 1/d). This aligns with our conclusion of a possibly smaller inversion bias than d eff /m. See Corollary E.2 in Appendix E for a proof of this fact.
for uniform random sampling matrix S ∈ R m×n , π i = 1/n as in Definition 2.1, H n ∈ R n×n the Walsh-Hadamard matrix of size n, and diagonal D n ∈ R n×n having i.i.d. Rademacher random variables on its diagonal.
The SRHT in Definition 3.6 enjoys the following properties: the randomized Walsh-Hadamard transform H n D n A of A is known to have approximately uniform leverages scores, that is Drineas et al. (2011)  and Tropp (2011, Theorems 3.1 and 3.2), as well as Lemma E.4 in Appendix E.3 in our setting; and since H ⊤ n H n /n = I n and D 2 n = I n , one has
ℓ C i (H n D n A/ √ n) ≈ d eff /n, see
1 n A ⊤ D n H ⊤ n H n D n A = A ⊤ A, so that H n D n A/
√ n and A have the same effective dimension. These lead to the following fine-grained debiasing result for SRHT with scalar debiasing, proven in Appendix E.3.4. Corollary 3.7 (Fine-grained debiasing for SRHT using scalar debiasing). Under the setting and notations of Theorem 3.1, for ÃSRHT ∈ R m×n the SRHT of A as in Definition 3.6, then there exists C > 0,
ν ≥ 0, n exp(-d eff ) < δ < m -3 such that for m ≥ Cρ max d 1+ν eff , ( m m-d eff Ã⊤ SRHT ÃSRHT + C) -1 is an (O(d -3ν/2 eff ) + O(ρ -1 max log(n/δ)d -ν-1/2 eff ), δ)-unbiased estimator of (A ⊤ A + C) -1 . 5
this section cite: ['b24', 'b9', 'b0', 'b0', 'b23']

Section: Application to De-biased Sub-sampled Newton with Improved Convergence
In this section, we show that the precise characterizations of random sampling inversion bias and the debiasing techniques in Section 3 apply to establish problem-independent local convergence rates of SSN methods.
Consider the following optimization problem:
β * = arg min β∈C F (β) = arg min β∈C f (β) + Φ(β), (7
)
for some smooth function F : R d → R that can be decomposed into f and Φ, and C ⊆ R d a convex set. This decomposition naturally arises in ML when, e.g., the loss function F is the sum of the empirical risk f over a set of n training samples and some regularization penalty Φ.
Newton's method solves (7) by performing iterative updates β t+1 = β t -µ t H -1 t (β t )∇F (β t ), with µ t the step size, ∇F (β t ) ∈ R d the gradient, and H t (β t ) ∈ R d×d the Hessian of F at β t that can be decomposed as
H t (β t ) = A(β t ) ⊤ A(β t ) + C(β t ),(8)
with A(β t ) ∈ R n×d and some p.s.d. matrix C(β t ) = ∇ 2 Φ(β t ) ∈ R d×d that takes a simple form, e.g., C(β t ) = 2λI d in the case of L 2 regularization Φ(β) = λ∥β∥ 2 . 5 Recall from Definition 2.3 that here for SRHT, we have πi = 1/n and thus ρmax = max 1≤i≤n nℓ
C i (HnDnA/ √ n)/d eff .
Despite having a locally super-linear convergence rate, Newton's method suffers from a heavy computational burden in forming the Hessian matrix H t (β t ), particularly when the training samples n is large, e.g., n ≫ d. In this case, the major computational bottleneck of Newton's method lies in the construction of A(β t ) ⊤ A(β t ) for the inverse Hessian matrix, as this takes O(nd 2 ) time. 6 Many randomized second-order methods have been proposed to replace the exact Hessian inverse by some computationally efficient estimate. Here, we consider SSN methods, that randomly sample the Hessian (Yao et al., 2018;Roosta-Khorasani & Mahoney, 2019;Xu et al., 2020) as follows.
Definition 4.1 (Sub-sampled Newton, SSN). To solve the optimization problem in (7), the SSN method performs the following iteration:
β t+1 =β t -µ t A(β t ) ⊤ S ⊤ t S t A(β t ) + C(β t ) -1 g t , (9
)
for t = 0, 1, . . . T , with g t ≡ ∇F (β t ) ∈ R d the gradient of F at β t , µ t the step size at time t, and random sampling matrix S t ∈ R m×n as in Definition 2.1. Due to the absence of precise characterizations of the subsampled Hessian inverse, as in Theorem 3.1 and Proposition 3.2 (that allow to, e.g., prove the near-
unbiasedness of SSN iteration E[β t+1 ] ≈ β t -µ t H -1 t (β t )g t )
, it is technically challenging to obtain problem-independent convergence rates for SSN. In the following, we fill this gap by showing how our inversion bias results in Section 3 apply to establish problem-independent local convergence rates for de-biased SSN. We position ourselves under the following standard assumption on the objective function F . Assumption 4.2 (Lipschitz Hessian). F, f : R d → R in (7) have Lipschitz continuous Hessian with Lipschitz constant L, that is, for any β,
β ′ ∈ R d , max{∥∇ 2 F (β) - ∇ 2 F (β ′ )∥, ∥∇ 2 f (β) -∇ 2 f (β ′ )∥} ≤ L∥β -β ′ ∥.
6 Of course, this does not need to be done explicitly.
Under Assumption 4.2, we evaluate the local convergence rate of the following de-biased SSN iterations:
βt+1 = βt -µ t A( βt ) ⊤ Š⊤ t Št A( βt ) + C( βt ) -1 g t , (10) with de-biased Št = diag m/(m -ℓ C is ( βt )/π is ) m s=1
• S t as in Proposition 3.2, with ℓ C is ( βt ) the i th s leverage score of A( βt ) given C( βt ). This leads to the following result.
Theorem 4.3 (Local convergence of de-biased SSN). Let Assumption 4.2 hold. For p.d. A(β * ) ⊤ A(β * ) = ∇ 2 f (β * ) and p.s.d. C(β * ) = ∇ 2 Φ(β * ), there exists a neighborhood U of β * such that the de-biased SSN iteration in (10) start- ing from β0 ∈ U satisfies, for U = {β : ∥β -β * ∥ H < (ρ max d eff σ min /m) 3/2 /L}, step size µ t = 1 - ρmax m/d eff +ρmax , m ≥ Cρ max d 1+ν eff , and ν ≥ log d eff (log(d eff T /δ)) that E ζ ∥ βT -β * ∥ H ∥ β0 -β * ∥ H 1/T ≤ ρ max d eff m (1 + ϵ), (11
)
holds for ϵ = O(d -ν/2 eff
) and conditioned on an event ζ that happens with probability at least 1 -δ. Here, σ min is the smallest singular value of H ≡ A(β * ) ⊤ A(β * ) + C(β * ), ρ max is the max importance sampling approximation factor in Definition 2.3 for ℓ C i = max 1≤t≤T ℓ C i ( βt ) and d eff = max 1≤t≤T d eff ( βt ) with ℓ C i ( βt ) and d eff ( βt ) the leverage scores and effective dimension of A( βt ) given C( βt ), respectively.
The proof of Theorem 4.3 can be found in Appendix F.2. The proof relies on a precise characterization of the second inverse moment of the randomly sampled Hessian matrix, extending beyond the first inverse moment result in Proposition 3.2. Due to space limitation, this technical result is presented in Proposition F.1 of Appendix F. For the sake of practical implementation, we also present, in Corollaries F.5 and F.6 of Appendix F, local convergence rates of SSN iterations using scalar debiasing m/(m -d eff ) under exact and approximate leverage score sampling as well as SRHT.
this section cite: ['b64', 'b53', 'b63']

Section: Numerical Experiments of De-biased SSN
In this section, we provide empirical evidence showing the improved convergence (and a better "complexityconvergence" trade-off as a consequence) of different debiased SSN methods proposed in Section 4, with respect to several first-and second-order baseline optimization methods. We solve the following logistic regression problem
min β∈R d 1 n n i=1 log 1 + exp(-y i a ⊤ i β) + λ 2 ∥β∥ 2 , (12
)
of the form (7), for regularization λ > 0, where a ⊤ i ∈ R d is the i th row of data matrix A ∈ R n×d sampled from both MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky, 2009) datasets, and y ∈ {±1} n the response vector. Implementation details are provided in Appendix G. Note that the time reported in Figures 2 and 3 include both the input data pre-processing time ( e.g., the computation of exact or approximate leverage scores, and Walsh-Hadamard transform) and the computational overhead associated with the sketching process. 500 1,000 10 -9 10 -8 10 -7 Sketch size m Relative error Newton-LESS 0.07 0.2 0.7 Wall-clock time (s) (a) MNIST data 500 1,000 1,500 10 -10 10 -9 10 -7 Sketch size m Relative error SSN-ARLev 0.5 1.6 3 Wall-clock time (s) (b) CIFAR-10 data Figure 2: Relative errors (in solid lines) and wall-clock time (in dashed lines) as a function of the sketch size m, for Newton-LESS and the proposed de-biased SSN-ARLev methods, applied to logistic regression on both MNIST and CIFAR-10 data. Relative errors are obtained after a fixed number of iterations (T = 5 for MNIST data and T = 7 for CIFAR-10 data). Results are obtained by averaging over 30 independent runs. 1. First-order baselines: Gradient Descent (GD) and Stochastic GD (SGD).
2. De-biased SSN methods using different sampling schemes: Approximate λ-Ridge Leverage Score (AR-Lev), Approximate Leverage Score (ALev), Shrinkage Leverage Score (SLev) sampling (Ma et al., 2015), and SRHT (see Definition 3.6 above).
3. Newton Sketch with LESS-uniform sketch (LESS) (Dereziński et al., 2021a), which achieves significantly shorter running times compared to the original Newton Sketch with dense Gaussian sketches. From Figure 3, we see that SSN methods, when properly de-biased, exhibit a significantly better convergencecomplexity trade-off than both first-order methods and the Newton-LESS approach. The SRHT sampling outperforms Newton-LESS but still lags slightly behind all other SSN variants in speed. Among these, the SLev scheme edges out ALev yet remains slower than ARLev. Across both datasets tested, de-biased SSN with ARLev consistently delivers the optimal convergence-complexity trade-off among all methods evaluated.
this section cite: ['b39']

Section: Conclusions and Perspectives
In the work, we investigate the inversion bias inherent in various random sampling schemes, including uniform and non-uniform leverage-based sampling, as wel as structured random projections (e.g., the Hadamard transformbased SRHT). Leveraging recent advances in RMT and RandNLA, we provide a precise characterization of this inversion bias and propose corresponding de-biasing techniques. Notably, for approximate leverage sampling and SRHT, this de-biasing reduces to a simple scalar rescaling. We further show that our results enable an improved SSN method, achieving local convergence rates comparable to those of Newton Sketch with dense Gaussian projections. Our theoretical insights are complemented by numerical results on MNIST and CIFAR-10 datasets, underscoring the practical relevance of the proposed approach.
It would be of future interest to see whether the proposed debiasing technique, when wisely combined with adaptive sampling schemes, can achieve or even improve the quadratic convergence in Lacotte et al. (2021). In addition, it is also worthwhile to extend our debiasing framework to dependent sampling methods (Cortinovis & Kressner, 2024), such as volume sampling, which may further improve SSN's efficiency and convergence.
this section cite: ['b34', 'b10']

Section: References
Ref_id:b0 Title: Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform Year: (2006)
Ref_id:b1 Title: Cambridge Studies in Advanced Mathematics Year: (2010)
Ref_id:b2 Title: Faster kernel ridge regression using sketching and preconditioning Year: (2017)
Ref_id:b3 Title: Exact and inexact subsampled Newton methods for optimization Year: (2019)
Ref_id:b4 Title: Numerical Optimization: Theoretical and Practical Aspects Year: (2006)
Ref_id:b5 Title: Convex Optimization Year: (2004)
Ref_id:b6 Title: Distribution function inequalities for martingales Year: (1973)
Ref_id:b7 Title: An iterative, sketching-based framework for ridge regression Year: (2018)
Ref_id:b8 Title: Low-rank approximation and regression in input sparsity time Year: (2017)
Ref_id:b9 Title: Input sparsity time low-rank approximation via ridge leverage score sampling Year: (2017)
Ref_id:b10 Title: Adaptive randomized pivoting for column subset selection Year: (2024)
Ref_id:b11 Title: Random Matrix Methods for Wireless Communications Year: (2011)
Ref_id:b12 Title: Random Matrix Methods for Machine Learning Year: (2022)
Ref_id:b13 Title: Subsampling spectral clustering for stochastic block models in large-scale networks Year: (2024)
Ref_id:b14 Title: Determinantal point processes in randomized numerical linear algebra Year: (2021)
Ref_id:b15 Title: Recent and upcoming developments in randomized numerical linear algebra for machine learning Year: (2024)
Ref_id:b16 Title: Precise expressions for random projections: Low-rank approximation and randomized Newton Year: (2020)
Ref_id:b17 Title: Newton-less: Sparsification without trade-offs for the sketched Newton update Year: (2021)
Ref_id:b18 Title: Sparse sketches with small inversion bias Year: (2021)
Ref_id:b19 Title: Randomized numerical linear algebra Year: (2016)
Ref_id:b20 Title: Lectures on randomized numerical linear algebra Year: (2018)
Ref_id:b21 Title: Fast Monte Carlo algorithms for matrices I: Approximating matrix multiplication Year: (2006)
Ref_id:b22 Title: Sampling algorithms for l 2 regression and applications Year: (2006)
Ref_id:b23 Title: Faster least squares approximation Year: (2011)
Ref_id:b24 Title: Fast approximation of matrix coherence and statistical leverage Year: (2012)
Ref_id:b25 Title: Fast randomized kernel ridge regression with statistical guarantees Year: (2015)
Ref_id:b26 Title: Spectra of the Conjugate Kernel and Neural Tangent Kernel for linear-width neural networks Year: (2020)
Ref_id:b27 Title: An identity for the Wishart distribution with applications Year: (1979)
Ref_id:b28 Title: Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions Year: (2011)
Ref_id:b29 Title: Extensions of Lipschitz mappings into a Hilbert space Year: (1984)
Ref_id:b30 Title: Accelerating data uncertainty quantification by solving linear systems with multiple right-hand sides Year: (2013)
Ref_id:b31 Title: Learning Multiple Layers of Features from Tiny Images Year: (2009)
Ref_id:b32 Title: Faster least squares optimization Year: (2019)
Ref_id:b33 Title: Adaptive and oblivious randomized subspace methods for high-dimensional optimization: Sharp analysis and lower bounds Year: (2022)
Ref_id:b34 Title: Adaptive Newton sketch: Linear-time optimization with quadratic convergence and effective Hessian dimensionality Year: (2021)
Ref_id:b35 Title: Gradientbased learning applied to document recognition. Proceedings of the IEEE Year: (1998)
Ref_id:b36 Title: Hessian eigenspectra of more realistic nonlinear models Year: (2021)
Ref_id:b37 Title: A random matrix analysis of random Fourier features: Beyond the Gaussian kernel, a precise phase transition, and the corresponding double descent Year: (2020)
Ref_id:b38 Title: Sparse quantized spectral clustering Year: (2021)
Ref_id:b39 Title: A statistical perspective on algorithmic leveraging Year: (2015)
Ref_id:b40 Title: Asymptotic analysis of sampling estimators for randomized numerical linear algebra algorithms Year: (2022)
Ref_id:b41 Title: Randomized algorithms for matrices and data Year: (2011)
Ref_id:b42 Title: Randomized numerical linear algebra: Foundations and algorithms Year: (2020)
Ref_id:b43 Title: The generalization error of random features regression: Precise asymptotics and the double descent curve Year: (2022)
Ref_id:b44 Title: Randomized Numerical Linear Algebra -a perspective on the field with an eye to software Year: (2023)
Ref_id:b45 Title: SGD in the large: Average-case analysis, asymptotics, and stepsize criticality Year: (2021)
Ref_id:b46 Title: Halting time is predictable for large models: A universality property and average-case analysis Year: (2023)
Ref_id:b47 Title: Nonlinear random matrix theory for deep learning Year: (2017)
Ref_id:b48 Title: Iterative Hessian sketch: Fast and accurate solution approximation for constrained least-squares Year: (2016)
Ref_id:b49 Title: Newton sketch: A near linear-time optimization algorithm with linear-quadratic convergence Year: (2017)
Ref_id:b50 Title: Random matrix approach to cross correlations in financial data Year: (2002)
Ref_id:b51 Title: Optimal Design of Experiments Year: (2006)
Ref_id:b52 Title: Newton meets Marchenko-Pastur: Massively parallel second-order optimization with Hessian sketching and debiasing Year: (2024)
Ref_id:b53 Title: Sub-sampled Newton methods Year: (2019)
Ref_id:b54 Title: Improved approximation algorithms for large matrices via random projections Year: (2006)
Ref_id:b55 Title: On the empirical distribution of eigenvalues of a class of large dimensional random matrices Year: (1995)
Ref_id:b56 Title: Improved analysis of the subsampled randomized Hadamard transform Year: (2011)
Ref_id:b57 Title: An introduction to matrix concentration inequalities Year: (2015)
Ref_id:b58 Title: High-Dimensional Probability: An Introduction with Applications in Data Science Year: (2018)
Ref_id:b59 Title: Optimal subsampling for quantile regression in big data Year: (2021)
Ref_id:b60 Title: Optimal subsampling for large sample logistic regression Year: (2018)
Ref_id:b61 Title: Sketching as a tool for numerical linear algebra Year: (2014)
Ref_id:b62 Title: Sub-sampled Newton methods with non-uniform sampling Year: (2016)
Ref_id:b63 Title: Newtontype methods for non-convex optimization under inexact Hessian information Year: (2020)
Ref_id:b64 Title: Inexact non-convex Newton-type methods Year: (2018)
Ref_id:b65 Title: Approximate Newton methods Year: (2021)
Ref_id:b66 Title: Optimal distributed subsampling for maximum quasi-likelihood estimators with massive data Year: (2022)
Ref_id:b67 Title: Singular values of differences of positive semidefinite matrices Year: (2001)
Ref_id:b68 Title: Optimal shrinkage for distributed second-order optimization Year: (2023-07)
