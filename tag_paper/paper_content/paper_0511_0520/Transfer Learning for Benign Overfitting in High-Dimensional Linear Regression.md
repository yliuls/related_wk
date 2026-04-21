Title: Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression
Abstract: Transfer learning is a key component of modern machine learning, enhancing the performance of target tasks by leveraging diverse data sources. Simultaneously, overparameterized models such as the minimum-ℓ 2 -norm interpolator (MNI) in high-dimensional linear regression have garnered significant attention for their remarkable generalization capabilities, a property known as benign overfitting. Despite their individual importance, the intersection of transfer learning and MNI remains largely unexplored. Our research bridges this gap by proposing a novel two-step Transfer MNI approach and analyzing its trade-offs. We characterize its non-asymptotic excess risk and identify conditions under which it outperforms the target-only MNI. Our analysis reveals free-lunch covariate shift regimes, where leveraging heterogeneous data yields the benefit of knowledge transfer at limited cost. To operationalize our findings, we develop a data-driven procedure to detect informative sources and introduce an ensemble method incorporating multiple informative Transfer MNIs. Finite-sample experiments demonstrate the robustness of our methods to model and data heterogeneity, confirming their advantage.

Section: Introduction
Transfer learning [49,67,72,21] is a scheme that aims to improve the performance of a learning task of interest, namely the target task, by leveraging knowledge acquired from related, but possibly different, source tasks. As modern datasets grow in size and heterogeneity, the seamless integration of diverse sources of information has become increasingly critical, positioning transfer learning as a valuable approach. Its success is particularly evident in high-dimensional regression, as there has been a recent surge of research demonstrating its advantage in tasks including, but not limited to, LASSO [61] and its variants [6,34,20,55], generalized linear models [60,35], and nonparametric regression [40,64,38,9].
While these transfer learning methods rely on explicit regularization [53,17,18] to address poor generalization arising from overfitting in high-dimensional regimes, contemporary deep learning methods have challenged the conventional wisdom of bias-variance trade-off framework. In particular, certain interpolators that achieve zero training error have been found to generalize remarkably well to unseen data, despite the absence of any explicit regularization. This surprising phenomenon of benign overfitting [7,8,5,4,44,54] upends the traditional notion of model capacity and generalization. Given the prevalence of overparameterized models in modern machine learning, this stark contrast raises an intriguing question:
this section cite: ['b48', 'b20', 'b60', 'b5', 'b33', 'b19', 'b54', 'b59', 'b34', 'b39', 'b63', 'b37', 'b8', 'b52', 'b16', 'b17', 'b6', 'b7', 'b4', 'b3', 'b43', 'b53']

Section: Can transfer learning further enhance the impressive out-of-sample generalization capabilities of such interpolators in high-dimensional linear regression?
Related work. Motivated by the empirical success of overparameterized deep learning models [70, 23,46], a substantial body of research has sought to understand why benign overfitting occurs, focusing on linear regression as a tractable setting for theoretical investigation. The theoretical foundation of our research, along with that of subsequent studies, has been established by Bartlett et al. [5], who studied the non-asymptotic excess risk of the minimum-ℓ 2 -norm interpolator (MNI) with n < p (i.e., dimension exceeds sample size). They demonstrated that if the covariance eigenvalues decay rapidly up to some "intrinsic" dimension k * < n (Assumption 2) and yet, the remaining p -k * components have effective ranks (Definition 1) greater than n, then the MNI trained on noisy data can achieve vanishing excess risk by capturing signal from the leading k * high-spectrum components and dispersing noise over the many low-spectrum tail components, as the tails span an "effective space" larger than the noise dimension n. In this situation, the MNI behaves similarly to ridge regression, exhibiting the effect known as implicit regularization [28]. Hastie et al. [19] explored the asymptotic excess risk of MNI as p/n → γ ∈ (0, ∞) based on random matrix theory [2], and Tsigler and Bartlett [62] extended this ridgeless interpolator to benign overfitting in ridge regression. Zhou et al. [71] and Koehler et al. [29] interpreted the benign overfitting of MNI under the lens of uniform convergence. Additionally, Muthukumar et al. [45] popularized the notion of harmless interpolation, demonstrating that under certain conditions, the MNI trained on noisy data does not severely degrade generalization, as estimation variance vanishes with increasing p/n.
In parallel, distribution shift in transfer learning for regression has been characterized by the two main pillars: (1) model shift, where the conditional distribution of response given covariates differs between the target and source tasks; (2) covariate shift, where the marginal distribution of covariates varies. Wang and Schneider [66] established a generalization bound under model shift, provided that the shift is smooth conditional on covariates. Recently, Tahir et al. [57] quantified model shift in high-dimensional regression by the cosine similarity between target and source model coefficients, showing that transfer performance depends critically on this alignment. Covariate shift has been studied along several lines, such as minimax risk formulations [32] and importance weighting [12], among others. Recent analyses of linear models under model and covariate shifts have examined optimal ridge regularization parameter, which can be negative [50].
Despite the potential significance of the aforementioned question, its exploration within transfer learning for MNIs remains severely limited. Mallinar et al. [42] studied an out-of-distribution (OOD) [39] setting 1 , where the MNI is trained exclusively on source (in-distribution, ID) data and tested on the target distribution. They provided conditions where a beneficial covariate shift yields lower OOD risk than ID risk, focusing on shifts in the eigenvalues of spiked covariances [24] and degree of overparameterization. However, their approach does not use target samples during training, even though at least limited target data are often available. In contrast, Wu et al. [69] proposed an online stochastic gradient descent (SGD)foot_1 algorithm pre-trained with the source and then finetuned on the target, encompassing OOD settings by allowing no fine-tuning. They showed that pre-training with O(n 2 ) can be comparable to n-target-supervised learning under certain covariate shifts. Despite using target data, their work, like Mallinar et al. [42], did not examine model shift, and its experiments are primarily focused on underparameterized regimes. Notably, Song et al. [56] developed the pooled-MNI, explicitly accounting for model shift. They derived its non-asymptotic excess risk and proposed a data-driven estimate for the optimal target sample size under model shift. However, pooling multiple sources can be susceptible to distribution shifts, as shown in our numerical evaluations (Section 5). Additional literature review is deferred to Appendix A, which discusses minimum-norm interpolators beyond the Euclidean norm.
this section cite: ['b22', 'b45', 'b4', 'b27', 'b18', 'b1', 'b61', 'b28', 'b44', 'b56', 'b31', 'b11', 'b49', 'b41', 'b38', 'b23', 'b41', 'b55']

Section: Summary of contributions.
Our study bridges the gap by investigating how transfer learning can enhance the generalization of target-only MNI through a novel knowledge transfer scheme. The main contributions are as follows.
• We propose a novel two-step Transfer MNI: pre-train a source-only MNI and then fine-tune it by interpolating target data while "staying close" to the pre-trained model. Under model shift with isotropic covariates, we identify when Transfer MNI outperforms target-only MNI, specify the optimal transfer size, and quantify the maximal improvement in excess risk. We provide non-asymptotic excess risk bounds under both model and covariate shifts when each single-task MNI is benignly overfitted. We further uncover free-lunch covariate shifts that alleviate the cost of knowledge transfer while preserving its generalization gain for Transfer MNI.
• Based on a data-driven procedure for detecting informative sources inducing positive transfer, we propose an ensemble method aggregating multiple informative Transfer MNIs with data-adaptive weights determined by source informativeness.
• Finite-sample experiments demonstrate robustness to distribution shifts and superior performance over transfer baselines, including the pooled-MNI [56] and SGD-based transfer [69], thereby confirming the empirical advantage of our approach under overparameterization.
Notation. Bold upper-(e.g., M and Λ) and lower-case (e.g., v and β) letters denote matrices and vectors, respectively, with ∥M∥ and ∥v∥ denoting the operator and ℓ 2 -norms. For Q ∈ N, define the sets
this section cite: ['b55']

Section: Preliminaries
Task setup. Consider an overparameterized linear regression setting involving one target task and Q source tasks. We have access to the training datasets {(X (q) , y (q) )} Q q=0 , each with n q samples of p-dimensional covariate generated from the following well-specified linear model, with q = 0 corresponding to the target:
y (q) = X (q) β (q) + ϵ (q) , q ∈ [Q] 0 .(1)
Here, y (q) ∈ R nq is the response vector; X (q) ∈ R nq×p is the design matrix with i-th row (sample) x (q) i ∈ R p ; β (q) ∈ R p is the fixed model coefficient; ϵ (q) ∈ R nq is the random noise. Provided n q ≤ p, the MNI trained on each dataset is uniquely given by β(q) M := arg min β∈R p ∥β∥ : X (q) β = y (q) = X (q)⊤ X (q) X (q)⊤ † y (q) , q ∈ [Q] 0 , (2) where M † ∈ R d×m denotes the Moore-Penrose inverse [51] of M ∈ R m×d ; Appendix C.1 provides a proof of its uniqueness and minimality in ℓ 2 -norm. The MNIs β(0) M and β(q) M correspond to our target and (q-th) source tasks, respectively.
Distribution shift. For well-specified linear models, model shift is attributed to the model contrast [34] defined as δ (q) := β (q) -β (0) , q ∈ [Q].
On the other hand, covariate shift is characterized by discrepancies in the covariance structure Σ = Exx ⊤ . While Kausik et al. [27], LeJeune et al. [33], Mallinar et al. [42] assumed simultaneous diagonalizability, which gives spectral decompositions Σ (0) = VΛ (0) V ⊤ and Σ (q) = VΛ (q) V ⊤ for some common orthogonal matrix V ∈ R p×p , Assumption 1 and our numerical studies (Section 5) do not necessarily impose such a shared eigenbasis.
In Assumptions 1 and 2 and Definitions 1 and 2, the index q applies for all q ∈ [Q] 0 .
this section cite: ['b1', 'b50', 'b33', 'b26', 'b32', 'b41']

Section: Assumption 1.
Let the design matrices X := {X (q) } Q q=0 and random noises E := {ϵ (q) } Q q=0 be mutually independent, and the following holds.
• Each design is of the form X (q) = Z (q) (Σ (q) ) 1/2 , where the rows of Z (q) ∈ R nq×p are i.i.d. ν xsub-Gaussian vectorsfoot_2 with i.i.d. mean-zero, unit-variance components, and Σ (q) ∈ R p×p is deterministic, symmetric positive definite with eigenvalues λ
(q) 1 ≥ λ (q) 2 ≥ . . . ≥ λ (q) p > 0.
• Each ∥Σ (0) (Σ (q) ) -1 ∥ is bounded above by a universal constant C Σ (q) > 0 with C Σ (0) = 1.
• Each noise ϵ (q) has i.i.d. mean-zero components with finite variance σ 2 q > 0.
The sub-Gaussianity of covariates is standard in recent transfer learning research [32,12,42,20]. When Σ (0) and Σ (q) are simultaneously diagonalizable, the condition ∥Σ (0) (Σ (q) ) -1 ∥ = O(1) reduces to upper-bounded spectral ratios, i.e., max j∈[p] λ (0) j /λ (q) j = O(1), as in the multiplicative spectral shift in Mallinar et al. [42]. Lastly, a finite second moment of the noise suffices as the formulation of mean squared loss in (5) only requires that each Eϵ (q) ϵ (q)⊤ is well-defined.
Effective ranks. We additionally introduce the key condition underpinning the benign overfitting of MNI, characterized by the effective ranks [30,5] of Σ (q) .
Definition 1. If λ (q) k+1 > 0 for k ≥ 0, the effective ranks of Σ (q) are defined as r k (Σ (q) ) := j>k λ (q) j λ (q) k+1 , R k (Σ (q) ) := j>k λ (q) j 2 j>k λ (q) j 2 .
Assumption 2. There exists universal constants b q , c q ≥ 1 such that the minimal index
k * q := min k ≥ 0 : r k (Σ (q) ) ≥ b q n q is well-defined with 0 ≤ k * q ≤ n q /c q .
Under Assumption 2, Bartlett et al. [5] formalized the benign overfitting of MNI as follows.
Definition 2. The single-task MNI β(q) M with covariance Σ (q) and n q < p is benign if
lim nq→∞ r 0 (Σ (q) ) n q = lim nq→∞ k * q n q = lim nq→∞ n q R k * q (Σ (q) ) = 0.
this section cite: ['b31', 'b11', 'b41', 'b19', 'b41', 'b29', 'b4', 'b4']

Section: Single-Source Transfer Task and Out-of-Sample Generalization
We propose a single-source transfer method in the form of late-fusion [56], where the q-th source task is learned independently and then integrated with the target task. In the first step, we pre-train the q-th source-only MNI β(q) M . The two-step Transfer MNI (TM) β(q) TM then fine-tunes by interpolating the target dataset while minimizing the Euclidean distance from the pre-trained β(q) M ; that is, β(q) TM := arg min β∈R p ∥β -
β(q) M ∥ : X (0) β = y (0) , q ∈ [Q].
The TM estimate admits the following interpretable decomposition structure that clarifies the knowledge transfer mechanism (see Appendix C.2 for the derivation):
β(q) TM transfer task = β(0) M target task + I p -H (0) β(q) M late-fusion knowledge transfer .(3)
Here, q) denotes the orthogonal projection onto the design row space S q := span{x
H (q) := X (q)⊤ X (q) X (q)⊤ † X (
(q) i } nq i=1 for q ∈ [Q] 0 ; conversely, I p -H (q) is the orthogonal projection onto the null space S ⊥ q such that R p = S q ⊕ S ⊥ q . Denoting by (υ) S0 = H (0) υ and (υ) S ⊥ 0 = (I p -H (0)
)υ the projections of a vector υ ∈ R p onto the target row and null spaces respectively, we obtain
β(q) TM S0 = β(0) M , β(q) TM S ⊥ 0 = I p -H (0) β(q) M .(4)
That is, the fine-tuning step for TM retains target-learned signal in the span of n 0 target samples (i.e., S 0 ) where benign overfitting ensures high prediction accuracy for the target-only MNI, while transferring source information only into the null space S ⊥ 0 where the target samples provide no information with β(0
) M S ⊥ 0 = 0 p .
Given this "retain-plus-transfer" mechanism exhibited by TM, we next analyze its generalization behaviors by comparing the excess risks of target and transfer tasks. Suppose we are given an out-of-sample target instance x 0 ∈ R p . The excess risk of an estimate β ≡ β(X , E) measured on the target distribution is defined by the following conditional mean squared loss:
R( β) := E (x0,E) x ⊤ 0 β -x ⊤ 0 β (0) 2 X = E E β -β (0) ⊤ Σ (0) β -β (0) X .(5)
The canonical bias-variance decomposition shows that the excess risk is the sum of the (squared) bias B and variance V, i.e., R( β) = B( β) + V( β). The bias B
M and variance V
M of the target-only MNI are obtained by Hastie et al. [19] as follows:
B (0) M := β (0)⊤ Π (0) β (0) , V(0)
M := σ 2 0 n 0 Tr Σ(0) † Σ (0) ,(6)
where we write Π (0) := (I p -H (0) )Σ (0) (I p -H (0) ) and Σ(q) := (1/n q )X
(q)⊤ X (q) for q ∈ [Q] 0 . The following lemma extends the bias-variance decomposition to our proposed TM estimate. Lemma 1. Under the mutual independence and mean-zero condition of (X , E) in Assumption 1, the excess risk of the TM estimate is the sum of bias B (q) TM and variance V (q) TM such that B (q)
TM := β (0)⊤ (I p -H (q) )Π (0) (I p -H (q) )β (0) + δ (q)⊤ H (q) Π (0) H (q) δ (q) -2δ (q)⊤ H (q) Π (0) (I p -H (q) )β (0) , V(q)
TM :=
σ 2 q n q Tr Σ(q) † Π (0) =: V (q) ↑ (variance inflation) + V (0
) M .
this section cite: ['b55', 'b18']

Section: Remark 1 (Bias reduction versus variance inflation). The variance inflation V (q)
↑ in Lemma 1 is positive almost surely, so the knowledge transfer always requires a higher variance as its cost. If β(q) TM reduces its estimation bias enough to outweigh the variance inflation, it attains a lower excess risk than the target-only MNI, referred to as positive transfer.
this section cite: []

Section: Model Shift under Isotropic Covariates
We analyze the effect of model shift under the isotropic case where Σ (0) = Σ (q) = I p . Although not "benign," this case ensures at least a "harmless" [45] performance for the single-task MNI, making it a worthwhile subject of investigation. In addition, our analysis provides an important insight: the dynamics between bias reduction and variance inflation (Remark 1) is determined by the interplay among p, n q , shift-to-signal ratio (SSR), and signal-to-noise ratio (SNR), where SSR and SNR are defined for each q-th source as
SSR q := ∥δ (q) ∥ 2 ∥β (0) ∥ 2 ≥ 0, SNR q := ∥β (0) ∥ 2 σ 2 q > 0, q ∈ [Q], provided β (0) ̸ = 0 p .
this section cite: ['b44']

Section: Theorem 1.
Under Assumption 1, and further assuming that (Z (0) , Z (q) ) have i.i.d. standard Gaussian entries with p > (n 0 + 1) ∨ (n q + 1) and Σ (0) = Σ (q) = I p , the expected bias and variance (expectation over X ) of the target-only MNI and TM estimate are as follows:
E X B (0
) M = p -n 0 p ∥β (0) ∥ 2 , E X B (q) TM = p -n 0 p p -n q p ∥β (0) ∥ 2 + n q p ∥δ (q) ∥ 2 , E X V (0
) M = σ 2 0 n 0 p -(n 0 + 1) , E X V (q) TM = p -n 0 p σ 2 q n q p -(n q + 1) = E X V (q) ↑ + E X V (0) M .
From Theorem 1, we observe a trade-off between p and n q . If ∥β (0) ∥ > ∥δ (q) ∥, increasing n q further reduces the bias of TM. An increase in p mitigates the variance inflation, but only at the expense of compromising this bias reduction effect. Based on Theorem 1, we formalize the regime where TM is expected to outperform the target-only MNI and specify the optimal transfer size maximizing the improvement in expected excess risk
E X R (0) M -E X R (q) TM .
Corollary 1. Under the setup in Theorem 1, the TM estimate satisfies
E X R (q) TM < E X R (0) M ⇐⇒ SSR q < 1 and SNR q (1 -SSR q ) > p p -(n q + 1)
.
The improvement in expected excess risk
∆(n q ) := E X R (0) M -E X R (q)
TM as a function of n q is strictly concave on n q ∈ [1, p -1). If and only if SSR q < 1 and SNR q (1 -SSR q ) ≥ p(p-1) (p-2) 2 , the optimal transfer size n * q maximizing ∆(n q ) exists and equals n * q = p -1 -
p(p-1) SNRq(1-SSRq) ∈ [1, p -1), with ∆(n * q ) = p-n0 p (n * q ) 2 (1-SSRq) p(p-1)
∥β (0) ∥ 2 > 0 being the maximal improvement.
In Corollary 1, negative transfer occurs when model shift dominates the signal with SSR q ≥ 1. The improvement ∆(n q ) strictly increases in n q up to the optimal threshold n * q but strictly decreases for n q > n * q ; that is, transferring more source samples helps only up to n * q , and beyond n * q , it always degrades the efficacy of knowledge transfer. While Song et al. [56] also investigated the same isotropic Gaussian setting under model shift, only Corollary 1 provides necessary and sufficient conditions for positive transfer and identifies the maximal improvement ∆(n * q ).
this section cite: ['b55']

Section: Convergence under Benign Covariates
Taking both model and covariate shifts into account, we now consider general sub-Gaussian covariates satisfying Assumption 2 and analyze non-asymptotic excess risk bounds and their implication. Theorem 2. Let Assumptions 1 and 2 hold. For each q ∈ [Q] 0 and any t ≥ log(2), define
ψ q (t) := r 0 (Σ (q) ) + t n q + r 0 (Σ (q) ) + t n q , Υ q := k * q n q + n q R k * q (Σ (q) )
.
With probability at least 1 -2e -δ and 1 -4e -η respectively, the biases are bounded above by
B (0
) M ≲ ψ 0 (δ)∥Σ (0) ∥∥β (0) ∥ 2 , B (q) TM ≲ ψ 0 (η)∥Σ (0) ∥∥δ (q) ∥ 2 + ψ q (η)C Σ (q) ∥Σ (q) ∥∥β (0) ∥ 2 .
Furthermore, there exist universal constants c 0 , c q ≥ 1 such that with probability at least 1 -7e -nq/cq -2e -ξ , the variance inflation of the TM estimate is bounded above by
V (q) ↑ ≲ σ 2 q Υ q ψ 0 (ξ) λ (q) p -1 ∥Σ (0) ∥,
and with probability at least 1 -10e -n0/c0 , the excess risk of the TM estimate is bounded below by R (q)
TM ≳ σ 2 0 Υ 0 , where V (0) M ≍ σ 2 0 Υ 0 on the same high-probability event lower-bounding R (q) TM .
The bias dynamics in Theorem 2 mirrors that in the isotropic case. If r 0 (Σ (0) ) ≪ n 0 , which is sufficient for a vanishing target-only bias, and r 0 (
Σ (0) ) ≍ r 0 (Σ (q) ), B(q)
TM also vanishes. Moreover, it can achieve a faster convergence if ∥δ (q) ∥ 2 ≪ ∥β (0) ∥ 2 and n 0 < n q . As for variance, the targetand source-only variances are within a constant factor of Υ 0 and Υ q respectively (see Corollary 3 in Appendix), and hence the terms vanish when each single-task is benign. The upper bound on V (q) ↑ is a product of the two "benign" terms Υ q and ψ 0 and the reciprocal of eigenvalue that reflects the lack of simultaneous diagonalizability in Assumption 1. While the reciprocal may loosen the bound, our numerical experiments (Section 5) show rapid decay in variances as p ≫ n 0 ∨ n q . Exploring a tractable covariance structure to refine this bound is left for future work. Finally, the lower bound on R (q) TM follows from Lemma 1, where the TM variance is no smaller than the target-only variance; the bias reduction effect cannot further improve the lower bound.
this section cite: []

Section: Free-Lunch Covariate Shift
In light of the invariance of spectral decay rates to uniform upscaling, we propose a type of covariate shift that yields a lower TM excess risk than without any covariate shift. To proceed, define the following minimal index for Σ (0) satisfying Assumption 1, which is always well-defined:
τ * := min k < p : λ (0) k+1 ≍ λ (0) p .(7)
We may expect τ * ≈ k * 0 for some benign covariance structures, where k * 0 is as specified in Assumption 2 for Σ (0) ; the beginning of Appendix C.7 illustrates a case where τ * ≪ p and τ * ≪ n 0 indeed.
Corollary 2 (Free-lunch covariate shift). For each q ∈ [Q] 0 , let the spectral decomposition of Σ (q) be Σ (q) = V (q) Λ (q) V (q)⊤ , where V (q) ∈ R p×p is orthogonal and Λ (q) = Diag(λ (q) 1 , . . . , λ (q) p ). Suppose Λ (q) = αΛ (0) for some α > 1, and under (A), the leading τ * eigenvector pairs in (V (q) , V (0) ) align; under (B), eigenvector pairs fully align with V (q) = V (0) , i.e., Σ (q) = αΣ (0) . Compared to the homogeneous case Σ (q) = Σ (0) , the following holds for each covariate shift case:
(A) The upper bound on B (q) TM remains identical up to a constant factor independent of α, while the upper bound on V (q) ↑ is multiplied by α -1 (i.e., reduced by a factor of α).
(B) The exact bias remains identical, while the exact variance inflation is multiplied by α -1 .
The covariate shift (A) in Corollary 2 shows that as long as each λ (q) j is uniformly upscaled from λ (0) j , preserving their decay rates, any misalignment in "noisy" eigendirections between Σ (q) and Σ (0) beyond the leading τ * high-signal components does not affect the convergence rate of bias, while still promoting faster convergence in variance inflation. With all p > τ * eigenvector pairs aligning between Σ (q) and Σ (0) , covariate shift (B) offers a greater advantage than (A) by specifying the exact impact on the bias and variance inflation, rather than on their upper bounds. Remark 2 (Relaxed free-lunch condition). The τ * -alignment condition for (A) in Corollary 2 can be relaxed: even if not all of leading τ * source eigenvectors align with the target counterparts, we can still achieve the same free-lunch effect as in (A) whenever
∥V (0) τ * -V (q) τ * ∥ ≲ λ (0) τ * +1 /λ (0) 1 , where V (0) τ * ∈ R p×τ * (resp. V (q) τ * ∈ R p×τ * ) comprises the leading τ * eigenvectors in V (0) (resp. V (q) ) as specified in Corollary 2.
this section cite: []

Section: Informative Multi-Source Transfer Task
In this section, we propose a transfer task that incorporates multiple sources identified as informative, those that induce positive transfer. The index set of informative sources is given by
I := q ∈ [Q] : R (q) TM -R (0) M < 0 ,(8)
which, however, is unknown in practice. Hence, it is of crucial interest in transfer learning to develop a data-driven procedure for detecting informative sources. Inspired by Tian and Feng [60], we utilize the K-fold cross-validation (CV) [1] to detect source transferability by comparing a "proxy" of excess risks.
First, partition the target dataset into K folds of equal size, each denoted by (X (0) [k] , y (0) [k] ) for k ∈ [K]; a common choice suggests K = 5 [17]. At each training step, we use the left-out folds (X (0) [-k] , y (0) [-k] ) := {(X (0) [k] , y (0) [k] )} K k=1 \ (X (0) [k] , y (0) [k] ) to train an estimate β[-k] and then evaluate the squared loss on the k-th fold given by
L [k] β[-k] := 1 n 0 /K y (0)[k] -X (0)[k] β[-k] 2 .
We repeat this across all K folds to obtain the terminal CV loss L β := K k=1 L [k] β[-k] /K, which estimates the prediction risk R β + σ 2 0 . We then estimate the oracle set (8) by
I := q ∈ [Q] : L β(q) TM -L β(0) M ≤ D (0) ,(9)
with some detection threshold D (0) > 0 to be specified later that depends on the target CV loss.
If I is non-empty, we train the TM estimate β(i) TM for each i ∈ I and form a weighted linear combination of β(i) TM . Each weight w i is initialized by the inverse of the CV loss L β(i) TM and then normalized so that i∈ I w i = 1, which serves as a data-adaptive measure of source informativeness. This allows us to leverage | I| sources for knowledge transfer, which we name Informative-Weighted Transfer MNI (WTM). The entire procedure, from detecting informative sources to computing the WTM estimate, is outlined in Algorithm 1 (Appendix D), specifying D (0) in the set (9).
this section cite: ['b59', 'b0', 'b16']

Section: Numerical Experiments
We evaluate the finite-sample performance of our proposed TM and WTM estimates by comparing them to the target-only MNI as a baseline. For the WTM estimate, we set K = 5 and ε 0 = 1/2 in Algorithm 1. Additionally, we test the pooled-MNI (PM) proposed by Song et al. [56] as a benchmark transfer task, which has an analytical solution: βPM := arg min β∈R p ∥β∥ : X (q) β = y (q) , ∀q ∈ [Q] 0 .
To consider a fine-tuning-based benchmark, we also test SGD q [69] each pre-trained with the q-th source, adjusting the initial learning rate from their sourcecode to ensure convergence.
Each setup is replicated over 50 independent simulations, and the average excess risk over the 50 simulations is plotted against each value of p ∈ {300, 400, . . . , 1000}. The mutually independent noise has i.i.d. mean-zero Gaussian components with common variance σ 2 = 1. We adopt the parametric configurations described below to simulate distribution shifts. Since only {(Z (q) , ϵ (q) )} Q q=0 in Assumption 1 are subject to randomness in our setup, all other parameters are generated with a fixed seed across the 50 simulations to ensure their deterministic nature; see Appendix F.1 for details.
Model shift. Let β (0) = S 1/2 u (0) and δ (q) = (SSR q • S) 1/2 u (q) for S > 0 and q ∈ [Q], where each u (q) is randomly generated from the p-dimensional unit sphere S p-1 without any structural assumption on the coefficients, e.g., sparsity. We write SSR = (SSR 1 , SSR 2 , . . . , SSR Q ).
Covariate shift. Let X (q) = Z (q) (Σ (q) ) 1/2 where {Z (q) } Q
q=0 are mutually independent and have i.i.d. standard Gaussian entries. In Section 5.1 with Q = 3 sources, the "benign" spectral matrices
Λ (q) = Diag λ (q) 1 , . . . , λ (q) p are as follows: λ (0) j = 15j -1 log -β (j+1)e 2 ; λ (1) j = 15j -γ ; λ (2) j = 15j -(1+log(n2)/n2) ; λ (3) j = 15 1(j = 1) + 1(j > 1)ε 1+θ 2 -2θ cos jπ/(p+1) 1+θ 2 -2θ cos π/(p+1)
, where we setfoot_3 β = 3/2, γ = θ = 1/2, and ε = 10 -5 . While we always fix the target covariance diagonal so that Σ (0) = Λ (0) , under covariate shift, we assign for each source covariance the spectral decomposition
Σ (q) = V (q) Λ (q) V (q)⊤ , where {V (q) } 3
q=1 are independently sampled from the orthogonal group O p := {Q ∈ R p×p : Q ⊤ Q = QQ ⊤ = I p }. By doing so, we test the robustness of our methods to covariate shift without simultaneous diagonalizability, as discussed prior to Assumption 1. We evaluate the effect of free-lunch covariate shift (Corollary 2) by adjusting the source covariances by V (q) (αΛ (q) )V (q)⊤ with α > 1, so we do not alter any source eigendirection to align with the target. In Section 5.2, we set Q = 2 and Σ (0) = Σ (1) = Σ (2) = I p , and under free-lunch covariate shift, Σ (1) = Σ (2) = αI p with α > 1. While βPM can outperform our estimates when it only learns source data without any distribution shift from the target (see Appendix E.1), the aggregate impact of distribution shifts for βPM is catastrophic throughout Fig. 1. On the contrary, in Fig. 1.(a), each β(q) TM remains robust, outperforming the targetonly MNI even under severe model shift (SSR = 0.6). Notably, the ensemble βWTM outperforms each individual β(q) TM , validating our data-adaptive weighting scheme based on CV losses. Figures 1.(b) and 1.(c) additionally introduce covariate shift in the broadest sense, as not only the spectrum of each Σ (q) differs from that of Σ (0) , but also all eigenvectors of Σ (q) arbitrarily misalign with those of Σ (0) . As a result, β(3) TM evidently suffers negative transfer in Fig. 1.(b); nevertheless, β WTM efficiently leverages only informative TM estimates by filtering out β(3) TM , consistently outperforming all competitors. When the free-lunch factor α > 1 is additionally applied in Fig. 1.(c), β(3) TM achieves significantly faster convergence and now performs comparably to the target-only MNI.   In Fig. 2, the target-only MNI with isotropic covariance demonstrates harmless interpolation [45], with its excess risk steadily converging to ∥β (0) ∥ 2 = 10. Therefore, the primary evaluation is whether each transfer task can prevent, or at least delay, its excess risk from converging to 10. In Fig. 2.(a), despite considerable model shift (SSR = 0.4) and a sub-optimal transfer size, β(1) TM slightly outperforms β(0) M . While βPM and SGD 2 eventually surpass the baseline as p grows, they still lag behind β(2) TM by a significant margin. Notably, even though β(1) TM performs worse than β(2) TM (while still inducing positive transfer), their ensemble βWTM substantially further reduces the excess risk.
this section cite: ['b55', 'b44']

Section: Benign Overfitting Experiments

this section cite: []

Section: Harmless Interpolation Experiments
Our methods benefit from the free-lunch covariate shift in Fig. 2.(b). However, in this scenario, the transfer size ⌊n * 2 ⌋ is no longer optimal because the variance inflation of TM is reduced by a factor of α, essentially reducing the source noise level from σ 2 2 = σ 2 to σ 2 2 = σ 2 /α. Thus, transferring even more than ⌊n * 2 ⌋ samples becomes advantageous. Taking this into account, we use SNR α := α∥β (0) ∥ 2 /σ 2 , which results in a new optimal transfer size that depends on SNR α and is larger than ⌊n * 2 ⌋ used in
this section cite: []

Section: Conclusion and Future Work
We propose a novel two-step Transfer MNI and a data-adaptive ensemble aggregating multiple informative Transfer MNIs. These methods address the posed question highlighting the underexplored nature of transfer learning for MNIs, supported by both theoretical characterizations and numerical validations under various distribution shift conditions. We further identify free-lunch covariate shift regimes for Transfer MNI where the trade-off between bias reduction and variance inflation is neutralized, allowing us to "cherry-pick" the benefit of knowledge transfer at limited cost.
this section cite: []

Section: References
Ref_id:b0 Title: The relationship between variable selection and data agumentation and a method for prediction Year: (1974)
Ref_id:b1 Title: Spectral Analysis of Large Dimensional Random Matrices Year: (2009)
Ref_id:b2 Title: An information-theoretic approach to transferability in task transfer learning Year: (2019)
Ref_id:b3 Title: Failures of model-dependent generalization bounds for least-norm interpolation Year: (2021)
Ref_id:b4 Title: Benign overfitting in linear regression Year: (2020)
Ref_id:b5 Title: Predicting with proxies: Transfer learning in high dimension Year: (2021)
Ref_id:b6 Title: Reconciling modern machinelearning practice and the classical bias-variance trade-off Year: (2019)
Ref_id:b7 Title: Two models of double descent for weak features Year: (2020)
Ref_id:b8 Title: Transfer learning for nonparametric regression: Non-asymptotic minimax analysis and adaptive procedure Year: (2024)
Ref_id:b9 Title: Foolish crowds support benign overfitting Year: (2022)
Ref_id:b10 Title: Atomic decomposition by basis pursuit Year: (1998)
Ref_id:b11 Title: High-dimensional kernel methods under covariate shift: Data-dependent implicit regularization Year: (2024)
Ref_id:b12 Title: On the robustness of minimum norm interpolators and regularized empirical risk minimizers Year: (2022)
Ref_id:b13 Title: How rotational invariance of common kernels prevents generalization in high dimensions Year: (2021)
Ref_id:b14 Title: Fast rates for noisy interpolation require rethinking the effect of inductive bias Year: (2022)
Ref_id:b15 Title: Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension Year: (2023)
Ref_id:b16 Title: The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer series in statistics Year: (2009)
Ref_id:b17 Title: Statistical Learning with Sparsity: The Lasso and Generalizations Year: (2015)
Ref_id:b18 Title: Surprises in highdimensional ridgeless least squares interpolation Year: (2022)
Ref_id:b19 Title: Transfusion: Covariate-shift robust transfer learning for high-dimensional regression Year: (2024)
Ref_id:b20 Title: Zeyar Aung, and Mohammad Abdul Azim. Transfer learning: a friendly introduction Year: (2022)
Ref_id:b21 Title: Frustratingly easy transferability estimation Year: (2022)
Ref_id:b22 Title: Neural tangent kernel: convergence and generalization in neural networks Year: (2018)
Ref_id:b23 Title: On the distribution of the largest eigenvalue in principal components analysis Year: (2001)
Ref_id:b24 Title: Theoretical characterization of the generalization performance of overfitted meta-learning Year: (2023)
Ref_id:b25 Title: Generalization performance of transfer learning: Overparameterized and underparameterized regimes Year: (2023)
Ref_id:b26 Title: Double descent and overfitting under noisy inputs and distribution shift for linear denoisers Year: (2024)
Ref_id:b27 Title: The optimal ridge penalty for real-world high-dimensional data can be zero or negative due to the implicit ridge regularization Year: (2020)
Ref_id:b28 Title: Uniform convergence of interpolators: Gaussian width, norm bounds and benign overfitting Year: (2021)
Ref_id:b29 Title: Concentration inequalities and moment bounds for sample covariance operators Year: (2017)
Ref_id:b30 Title: Minimum norm interpolation meets the local theory of banach spaces Year: (2024)
Ref_id:b31 Title: Near-optimal linear regression under distribution shift Year: (2021)
Ref_id:b32 Title: Monotonic risk relationships under distribution shifts for regularized risk minimization Year: (2024)
Ref_id:b33 Title: Transfer Learning for High-Dimensional Linear Regression: Prediction, Estimation and Minimax Optimality Year: (2021)
Ref_id:b34 Title: Estimation and inference for highdimensional generalized linear models with knowledge transfer Year: (2024)
Ref_id:b35 Title: Minimum ℓ 1 -norm interpolators: Precise asymptotics and multiple descent Year: (2021)
Ref_id:b36 Title: Just interpolate: Kernel "Ridgeless" regression can generalize Year: (2020)
Ref_id:b37 Title: Smoothness adaptive hypothesis transfer learning Year: (2024)
Ref_id:b38 Title: Towards out-of-distribution generalization: A survey Year: (2023)
Ref_id:b39 Title: Optimally tackling covariate shift in RKHS-based nonparametric regression Year: (2023)
Ref_id:b40 Title: Parthe Pandit, Misha Belkin, and Preetum Nakkiran. Benign, tempered, or catastrophic: Toward a refined taxonomy of overfitting Year: (2022)
Ref_id:b41 Title: Minimum-norm interpolation under covariate shift Year: (2024)
Ref_id:b42 Title: Overfitting behaviour of gaussian kernel ridgeless regression: Varying bandwidth or dimensionality Year: (2024)
Ref_id:b43 Title: The generalization error of random features regression: Precise asymptotics and the double descent curve Year: (2022)
Ref_id:b44 Title: Harmless interpolation of noisy data in regression Year: (2020)
Ref_id:b45 Title: Deep double descent: Where bigger models and more data hurt Year: (2020)
Ref_id:b46 Title: LEEP: A new measure to evaluate transferability of learned representations Year: (2020)
Ref_id:b47 Title: Simple transferability estimation for regression tasks Year: (2023)
Ref_id:b48 Title: A survey on transfer learning Year: (2010)
Ref_id:b49 Title: Optimal ridge regularization for out-ofdistribution prediction Year: (2024)
Ref_id:b50 Title: A generalized inverse for matrices Year: (1955)
Ref_id:b51 Title: Consistency of interpolation with laplace kernels is a high-dimensional phenomenon Year: (2019)
Ref_id:b52 Title: Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond Year: (2001)
Ref_id:b53 Title: The implicit bias of benign overfitting Year: (2023)
Ref_id:b54 Title: Unified transfer learning in high-dimensional linear regression Year: (2024)
Ref_id:b55 Title: Generalization error of min-norm interpolators in transfer learning Year: (2024)
Ref_id:b56 Title: Features are fate: a theory of transfer learning in high-dimensional regression Year: (2025)
Ref_id:b57 Title: Otce: A transferability metric for cross-domain cross-task representations Year: (2021)
Ref_id:b58 Title: Benign overfitting in out-of-distribution generalization of linear models Year: (2025)
Ref_id:b59 Title: Transfer learning under high-dimensional generalized linear models Year: (2023)
Ref_id:b60 Title: Regression shrinkage and selection via the lasso Year: (1996)
Ref_id:b61 Title: Benign overfitting in ridge regression Year: (2023)
Ref_id:b62 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b63 Title: Minimax optimal transfer learning for kernel-based nonparametric regression Year: (2023)
Ref_id:b64 Title: Tight bounds for minimum ℓ 1 -norm interpolation of noisy data Year: (2022)
