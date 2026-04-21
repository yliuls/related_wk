Title: Spectral Perturbation Bounds for Low-Rank Approximation with Applications to Privacy
Abstract: A central challenge in machine learning is to understand how noise or measurement errors affect low-rank approximations-particularly in the spectral norm. This question is especially important in differentially private low-rank approximation, where one aims to preserve the top-p structure of a data-derived matrix while ensuring privacy. Prior work often analyzes Frobenius norm error or changes in reconstruction quality, but these metrics can over-or under-estimate true subspace distortion. The spectral norm, by contrast, captures worst-case directional error and provides the strongest utility guarantees. We establish new high-probability spectral-norm perturbation bounds for symmetric matrices that refine the classical Eckart-Young-Mirsky theorem and explicitly capture interactions between a matrix A ∈ R n×n and an arbitrary symmetric perturbation E. Under mild eigengap and norm conditions, our bounds yield sharp estimates for ∥(A + E) p -A p ∥, where A p is the best rank-p approximation of A, with improvements of up to a factor of √ n. As an application, we derive improved utility guarantees for differentially private PCA, resolving an open problem in the literature. Our analysis relies on a novel contour bootstrapping method from complex analysis and extends it to a broad class of spectral functionals, including polynomials and matrix exponentials. Empirical results on real-world datasets confirm that our bounds closely track the actual spectral error under diverse perturbation regimes.

Section: Introduction
Low-rank approximation is a foundational technique in machine learning, data science, and numerical linear algebra, with applications ranging from dimensionality reduction and clustering to recommendation systems and privacy-preserving data analysis [1,4,5,14,21,23,24,42,45]. A common setting involves a real symmetric matrix A ∈ R n×n , such as a sample covariance matrix derived from high-dimensional data. Let λ 1 ≥ • • • ≥ λ n denote the eigenvalues of A, with corresponding orthonormal eigenvectors u 1 , . . . , u n . The best rank-p approximation of A is denoted by A p := p i=1 λ i u i u ⊤
i . This approximation solves the optimization problem A p = arg min rank(B)≤p ∥A -B∥, where the norm can be any unitarily invariant norm [7,10]. In particular, A p minimizes both the spectral norm ∥ • ∥, measuring worst-case error, and the Frobenius norm ∥ • ∥ F , measuring average deviation.
In many applications, the matrix A is not directly available-it may be corrupted by noise, compressed for efficiency, or randomized to preserve privacy. A standard model introduces a symmetric perturbation E, yielding the observed matrix Ã := A + E. The approximation Ãp , computed from Ã, is often used in downstream learning and inference. This leads to a central question: How does the perturbation E affect the top-p approximation A p ? Understanding the deviation ∥ Ãp -A p ∥ is critical for ensuring the reliability and robustness of low-rank methods under noise.
this section cite: ['b0', 'b3', 'b4', 'b13', 'b20', 'b22', 'b23', 'b41', 'b44', 'b6', 'b9']

Section: Motivating application: differential privacy.
The stability under perturbations is especially important when the matrix A encodes sensitive information, such as user behavior or medical data. In such settings, even low-rank approximations of A can inadvertently leak private information [6]. To address this risk, differential privacy (DP) [14] has become the standard framework for designing privacy-preserving algorithms. Several mechanisms have been developed to release private low-rank approximations while satisfying DP guarantees [8,9,15,25,29,31,34,39]. A canonical method, introduced in [15], adds a symmetric noise matrix E with i.i.d. Gaussian entries to the input matrix A, yielding the perturbed matrix Ã = A + E. The algorithm then releases Ãp as the privatized output. The utility of such mechanisms is typically assessed by comparing Ãp to the ideal (non-private) approximation A p . Two standard metrics are: (1) the Frobenius norm error ∥ Ãp -A p ∥ F , and (2) the change in reconstruction error |∥A-A p ∥ ⋆ -∥A-Ãp ∥ ⋆ |, which measures how much the quality of low-rank approximation degrades due to noise, for a norm ∥ • ∥ ⋆ [3,11,15,29]. These metrics offer insight into the effect of noise on overall variance or total reconstruction error. However, as we explain next, they may fail to capture worst-case directional misalignment, which is often critical for downstream tasks and algorithmic guarantees.
Limitations of existing utility metrics. The Frobenius norm error and reconstruction error may not be appropriate in applications that rely on the geometry of the top-p eigenspace. In particular, the Frobenius norm may overestimate the impact of noise by up to a factor of √ p when the perturbation E lies largely in directions orthogonal to the top-p subspace. The reconstruction error metric can underestimate subspace deviation-sometimes dramatically. In some cases, it remains small (or even zero) despite substantial rotation in the top-p eigenspace. (See Sections B for concrete illustrations.) These limitations motivate the use of the spectral norm ∥ Ãp -A p ∥, which captures the worst-case directional deviation between the two low-rank approximations. The spectral norm also governs algorithmic robustness in many downstream applications, such as PCA-based learning, private clustering, and subspace tracking.
A classical spectral norm bound, derived from the Eckart-Young-Mirsky theorem [7,16], states that ∥ Ãp -A p ∥ ≤ 2(λ p+1 + ∥E∥), which holds for arbitrary matrices and noise. However, such bounds are often pessimistic and fail to exploit the structure of A and E. More refined bounds exist in the Frobenius norm setting. For example, recent work [29,30] shows that when A is positive semidefinite and has a nontrivial eigengap δ p := λ p -λ p+1 ≥ 4∥E∥, and when E is drawn from a complex Gaussian ensemble, one obtains: E∥ Ãp -A p ∥ F = Õ( √ p • ∥E∥ • λp δp ), which improves on the earlier reconstruction-error-based bounds of [15] by a factor of √ p. However, these bounds have important limitations: They hold only in expectation and do not yield high-probability guarantees; They often assume Gaussian noise distributions; They are not spectral norm bounds and therefore do not directly quantify the worst-case impact on the eigenspace. These limitations prompt the following open question, raised in [29,Remark 5.3]: Can one obtain high-probability spectral norm bounds for ∥ Ãp -A p ∥ under natural structural assumptions on A and realistic noise models?
Our contributions. We resolve the open question posed in [29,Remark 5.3], proving new highprobability spectral norm bounds for low-rank approximation under symmetric perturbations. Our results rely on natural structural assumptions on A and E and yield the first such guarantees for differentially private PCA (DP-PCA).
• Two high-probability spectral norm bounds. Under the same eigengap condition as [29], δ p := • Spectral utility bounds for DP-PCA. Our first bound yields a high-probability spectral norm utility guarantee for differentially private PCA under sub-Gaussian noise, improving existing Frobenius-norm bounds by up to a factor of √ p (Corollary 2.4). While prior work has achieved spectral norm guarantees in iterative or multi-pass settings [17,18], our contribution concerns the direct noise-addition model, where this appears to be the first such result. For matrices with low stable rank and weak eigenspace-noise interaction, our second bound further improves by up to √ n.
• Novel analytical technique: contour bootstrapping. Our proof relies on a contour bootstrapping argument (Lemma 3.1), which provides a new way to analyze the contour representation of perturbations [19,26,35], enabling analysis of a broader class of spectral functionals (Theorem 2.3). The bootstrapping argument here is a generalization of the argument used to handle eigenspaces perturbation introduced in [37].
• Empirical validation. We benchmark our bounds on real covariance matrices under both Gaussian and Rademacher noise. Across datasets and noise regimes, the predicted error closely matches empirical behavior and consistently surpasses classical baselines, confirming the sharpness and robustness of our theoretical results (Section 4).
this section cite: ['b5', 'b13', 'b7', 'b8', 'b14', 'b24', 'b28', 'b30', 'b33', 'b38', 'b14', 'b0', 'b2', 'b10', 'b14', 'b28', 'b6', 'b15', 'b28', 'b29', 'b14', 'b28', 'b28', 'b28', 'b16', 'b17', 'b18', 'b25', 'b34', 'b36']

Section: Main results
Main spectral norm bound. For clarity, we state our main bounds assuming A ∈ R n×n is positive semi-definite (PSD); extensions to symmetric matrices appear in Section D. Let λ 1 ≥ • • • ≥ λ n ≥ 0 be the eigenvalues of A, with corresponding orthonormal eigenvectors u 1 , . . . , u n , and define the eigengap δ k := λ k -λ k+1 . Given a real symmetric perturbation matrix E, we let Ã := A + E, and define A p and Ãp as the best rank-p approximations of A and Ã, respectively. Our goal is to bound the spectral error ∥ Ãp -A p ∥. The O(•) notation here hides a small universal constant (less than 7), which we have not optimized; see Section D.1 for the proof of the generalization to the symmetric setting, of which this theorem is a special case. For Wigner noise-i.e., a symmetric matrix E with i.i.d. sub-Gaussian entries of mean 0 and variance 1-we have ∥E∥ = (2 + o(1)) √ n with high probability [41,43],
so Theorem 2.1 reduces to ∥ Ãp -A p ∥ = O √ n λp δp .
The right-hand side is explicitly noisedependent, addressing a key limitation of the classical Eckart-Young-Mirsky bound. Moreover, in many widely studied structured models (e.g., spiked covariance, stochastic block, and graph Laplacian models), one typically has λ p = O(δ p ), yielding the clean bound O(∥E∥). This rate is theoretically tight: for instance, when A is a PSD diagonal matrix and E = µI n for some µ > 0, we have ∥ Ãp -A p ∥ = µ = ∥E∥.
this section cite: ['b40', 'b42']

Section: Gap condition.
Our assumption 4∥E∥ < δ p aligns with standard conditions in prior work, including [29,30], and is satisfied in many well-studied matrix models-such as spiked covariance (Wishart) models, deformed Wigner ensembles, stochastic block models, and kernel matrices for clustering. It also arises naturally in classical perturbation theory [12,26,28]. Empirical analyses [29,Section B] further show that this condition holds for real-world datasets commonly used in private matrix approximation (e.g., the 1990 U.S. Census and the UCI Adult dataset [3,11]). Hence, Theorem 2.1 operates under a mild and broadly applicable assumption, satisfied across both theoretical models and practical benchmarks.
this section cite: ['b28', 'b29', 'b11', 'b25', 'b27', 'b28', 'b2', 'b10']

Section: Comparison to the Eckart-Young-Mirsky bound. Using λ
p = δ p + λ p+1 , Theorem 2.1 rewrites as ∥ Ãp -A p ∥ = O(∥E∥ + λ p+1 • ∥E∥
δp ). This improves on the E-Y-M bound O(∥E∥ + λ p+1 ) when λ p+1 ≫ ∥E∥, by a factor of min{ λp+1 ∥E∥ , δp ∥E∥ }. For example, consider a matrix with spectrum {10n, 9n, . . . , n, n/2, 1, . . . , 1} and p = 10. For Gaussian noise with ∥E∥ = O( √ n), E-Y-M yields O(n) error, while our bound gives O( √ n), a √ n-factor gain.
Comparison to Mangoubi-Vishnoi bounds [29,30]. Our bound also improves upon the Frobenius norm bounds of [29,30], which under the same gap assumption yield:
E∥ Ãp -A p ∥ F = Õ( √ p∥E∥ • λp δp
). We eliminate the √ p factor, upgrade from expectation to high probability, and support real-valued, non-Gaussian noise models. A more detailed comparison appears later in this section (Corollary 2.4), where we analyze implications for differentially private PCA.
Proof technique: contour bootstrapping. Unlike prior analyses [29,30], which rely on Dyson Brownian motion and tools from random matrix theory (see Section A, our proof of Theorem 2.1 uses a contour-integral representation of the rank-p projector. This approach, which we call contour bootstrapping, isolates the top-p eigenspace via complex-analytic techniques and avoids powerseries or Davis-Kahan-type expansions. It enables tighter, structure-aware spectral bounds and ex-tends naturally to refined perturbation results (Theorem 2.2) and general spectral functionals (Theorem 2.3). Full details appear in Section 3.
this section cite: ['b28', 'b29', 'b28', 'b29', 'b28', 'b29']

Section: Refined bound via eigenspace interaction.
To sharpen our analysis, we incorporate fine-grained structure of the eigenspace and its interaction with the noise. Inspired by the recent works [33,38], we start with the observation that the rank-p perturbation is primarily influenced by the cluster of eigenvalues near λ p , and the interaction between E and the corresponding eigenvectors. To control these factors, we define the halving distance r (w.r.t the index p) as the smallest integer such that λ r+1 ≤ λ p /2, and interaction term x := max 1≤i,j≤r |u ⊤ i Eu j |, measuring the alignment between the noise E and the top-r eigenvectors of A. This yields a refined spectral norm bound:
Theorem 2.2 (Interaction-aware bound). If 4∥E∥ ≤ δ p , then ∥ Ãp -A p ∥ ≤ Õ(∥E∥ + r 2 x • λp δp ).
See Section D.2 for the proof and its generalization to the symmetric setting. This bound improves upon the basic eigengap bound O ∥E∥ • λp δp when the interaction term r 2 x is small. This occurs, for instance, when (i) A has low stable rank or clustered eigenvalues (e.g., spiked models, multi-cluster Laplacians), (ii) the noise E is random and approximately orthogonal to the leading eigenspace, or (iii) λ p /δ p is large but x = Õ(1) and r = Õ(1). In such regimes, the bound simplifies to Õ ∥E∥ + λp δp , yielding up to a √ n-factor improvement over Theorem 2.1. This highlights the benefit of explicitly incorporating spectral decay and noise-eigenspace alignment when analyzing noise-robust low-rank approximations.
In practice, many public DP datasets (e.g., Census, Adult, KDD) have small dimensions and modest eigenspace decay, the simple bound is more effective. However, the refined bound becomes especially informative in large-scale or synthetically structured settings. Thus, the two bounds are best viewed as complementary: the first is robust and broadly applicable, while the second highlights structural regimes where stronger stability is provable.
Extension to spectral functionals. Beyond approximating A itself, many applications involve lowrank approximations of spectral functions f (A), such as A k , exp(A), or cos(A); see [7,44]. Our contour-based analysis extends naturally to this broader setting. Let f p (A) := p i=1 f (λ i )u i u ⊤ i denote the best rank-p approximation of f (A). We obtain the following general perturbation bound. Theorem 2.3 (Perturbation bounds for general functions). If 4∥E∥ ≤ δ p , then
∥f p ( Ã) -f p (A)∥ ≤ O max z∈Γ1 ∥f (z)∥ • ∥E∥ δp ,
where Γ 1 is the rectangle with vertices (x 0 , T ), (x 1 , T ), (x 1 , -T ), (x 0 , -T ) with x 0 := λ p -δp 2 , x 1 := 2λ 1 , T := 2λ 1 .
The O(•) notation hides a small universal constant (less than 4), which we have not attempted to optimize; see Section F for details. For example, let f (z) = z 3 , so that f p ( Ã) and f p (A) correspond to the best rank-p approximations of Ã3 and A 3 , respectively. Since max z∈Γ1 ∥f (z)∥ ≤ 64∥A∥ 3
this section cite: ['b32', 'b37', 'b6', 'b43']

Section: , Theorem 2.3 yields ∥ Ã3
p -A 3 p ∥ = O ∥A∥ 3 • ∥E∥/δ p . This result applies to many important classes of functions-e.g., polynomials, exponentials, and trigonometric functions-and hence we expect it to be broadly useful. However, Theorem 2.3 does not apply to non-entire functions such as f (z) = z c for non-integer c, where singularities obstruct the contour representation (1). In particular, when c < 0, the expression f p (A) is no longer the best rank-p approximation to f (A), so the conclusion of Theorem 2.3 is not meaningful in that setting. We note that in a related work [36], the first two authors present an extension of the setting f (z) = z -1 .
Application: differentially private low-rank approximation. We now apply our spectral norm bound to analyze a standard differentially private (DP) mechanism for releasing a low-rank approximation of a sensitive matrix A, commonly assumed to be a sample covariance matrix and hence PSD. Under (ε, δ)-DP [14], the Gaussian mechanism releases Ã := A + E, where E is a symmetric matrix with i.i.d. Gaussian entries scaled to sensitivity ∆ = O( log(1/δ)/ε). A common postprocessing step is to compute Ãp , the best rank-p approximation of Ã. Prior analyses [3,15,30] focused primarily on Frobenius norm or reconstruction error. For instance, [30] showed that under complex Wigner noise and a moderate eigengap, E∥ Ãp -A p ∥ F ≤ √ pn λp δp up to lower-order terms.
Since ∥ Ãp -A p ∥ ≤ ∥ Ãp -A p ∥ F , the above inequality implies an expected spectral norm error of Õ √ pn λp δp . In contrast, our bound yields the following high-probability spectral norm guarantee:
this section cite: ['b35', 'b13', 'b2', 'b14', 'b29', 'b29']

Section: Corollary 2.4 (Application to differential privacy).
Let A be PSD and E be a real or complex Wigner matrix. If δ p ≥ 8.01 √ n, then with probability 1 -o(1), ∥ Ãp -A p ∥ ≤ O( √ n • λp δp ).
This follows directly from Theorem 2.1, using the fact that ∥E∥ = O( √ n) with high probability for Wigner matrices [40,43]. Compared to [30], this result provides a spectral norm (rather than Frobenius) guarantee, holds with high probability instead of in expectation, applies to both real and complex Wigner noise, removes the log log log n n factor, and eliminates restrictive assumptions such as λ 1 ≤ n 50 . It also improves the dependence on p by a factor of √ p, thereby resolving the open question posed in [30,Remark 5.3].
The spectral norm better captures subspace distortion, which is critical in applications like private PCA. Unlike Frobenius or reconstruction error-both of which may remain small even when Ãp deviates significantly from the true top-p eigenspace-the spectral norm reflects worst-case directional error and is thus a more reliable utility metric. This distinction is empirically validated in Figure 3. Moreover, Corollary 2.4 further yields high-probability Frobenius norm and reconstruction error bounds on the perturbation of low-rank approximations:
∥ Ãp -A p ∥ F ≤ O √ pn • λp δp , and |∥ Ãp -A∥ -∥A p -A∥| ≤ O √ n • λp δp .
Finally, while Corollary 2.4 is stated for sub-Gaussian noise, Theorem 2.1 extends to any symmetric perturbation satisfying the norm and gap conditions, including subsampled or quantized Gaussians and Laplace noise. We leave the detailed analysis of these settings to future work. "EYM" and "M-V" denote the Eckart-Young-Mirsky and [29,30] bounds, respectively.
Alternative methods for approximating A p . Hardt and Price [17,18] proposed a random iterative method which, under the condition δ p ≫ √ n log n, produces a rank-k approximation A ′ of A p with
k = p + O(1), satisfying the trade-off bound ∥A ′ -A p ∥ = Õ √ n λ1 δp max 1≤i≤n ∥u i ∥ ∞ , where u i denotes the eigenvectors of A.
If at least one eigenvector u i is localized (i.e., max 1≤i≤n ∥u i ∥ ∞ = 1/ Õ(1)), this simplifies to Õ √ n λ1 δp . In this regime, Theorem 2.1 achieves a smaller bound by a factor of Õ(λ 1 /λ p )-up to √ n when λ 1 = Θ(n) and λ p = Θ( √ n). Furthermore, Theorem 2.2 provides an additional improvement by a factor of O min
√ n r 2 , λ1 δp , which can reach √ n when r = Õ(1) and δ p = Θ( √ n)-a common regime in high-dimensional data.
If all eigenvectors u i are delocalized (i.e., max 1≤i≤n ∥u i ∥ ∞ = Õ(1)/ √ n), the Hardt-Price bound reduces to Õ(λ 1 /δ p ). Theorem 2.1 achieves a comparable rate when σ 1 = Θ(n) and λ p = c δ p = Θ( √ n), while Theorem 2.2 yields an improvement by a factor of λ 1 /λ p whenever r = Õ(1), i.e., when A is approximately low-rank.
this section cite: ['b39', 'b42', 'b29', 'b29', 'b28', 'b29', 'b16', 'b17']

Section: Proof outline
In the preceding section, we stated our main results-Theorems 2.1, 2.2, and 2.3. Here, we first sketch the key ideas behind the proof of Theorem 2.1, then adapt the same framework, with minor refinements, to derive Theorems 2.2 and 2.3.
The proof of Theorem 2.1 proceeds in three main steps. First, using the contour method, we obtain the contour-based bound of our perturbation ∥ Ãp -
A p ∥ ≤ F (z) := 1 2πi ∥ Γ z[(zI -Ã) -1 -(zI - A) -1 ]∥dz.
Here Γ is a contour on the complex plane, isolating the p-leading eigenvalues of A and Ã. This contour step captures the A-E interaction that the Eckart-Young-Mirsky bound omits (see Appendix A). Secondly, we develop the contour bootstrapping technique (Lemma 3.1), which under the gap assumption 4∥E∥ ≤ δ p , yields F (z) ≤ 2F 1 (z) with F 1 (z) := Γ ∥z(zI -A) -1 E(zI -A) -1 ∥|dz|. This technique (valid for any entire function f ) replaces the traditional series expansions and the heavy analysis of the matrix-derivative operator (the limitation of the Mangoubi-Vishnoi approach [29,30], Appendix A) with a computable quantity. Third, we construct a bespoke contour Γone specifically tailored so that the top-p eigenvalues of A and Ã lie at prescribed distances from its sides. This precise alignment makes the integral defining F 1 (z) both tractable and essentially optimal, yielding a tight perturbation bound.
this section cite: ['b28', 'b29']

Section: Step 1: Representing ∥f
p ( Ã) -f p (A)∥ via the classical contour method. Let λ 1 ≥ • • • ≥
λ n be the eigenvalues of A with the corresponding eigenvectors {u i } n i=1 . We now present the contour method to bound matrix perturbations in the spectral norm. Let Γ be a contour in C that encloses λ 1 , λ 2 , . . . , λ p and excludes λ p+1 , λ p+2 , . . . , λ n . Let f be any entire function and recall
f p (A) = p i=1 f (λ p )u i u ⊤ i .
Since f is analytic on the whole plane C, the well-known contour integral representation [19,26,35] gives us:
1 2πi Γ f (z)(zI -A) -1 dz = p i=1 f (λ i )u i u ⊤ i = f p (A).
Let λ1 ≥ • • • ≥ λn denote the eigenvalue of Ã with the corresponding eigenvectors ũ1 , ũ2 , . . . , ũn . The construction of Γ (presented later) and the gap assumption 4∥E∥ < δ p ensure that the eigenvalues λi for 1 ≤ i ≤ p lie inside Γ, while all λj for j > p remain outside. Then, similarly, we have
1 2πi Γ f (z)(zI -Ã) -1 dz = p i=1 f ( λi )ũ i ũ⊤ i := f p ( Ã).
Thus, we obtain the following contour identity for the perturbation:
f p ( Ã) -f p (A) = 1 2πi Γ f (z)[(zI -Ã) -1 -(zI -A) -1 ] |dz|.(1)
Now we bound the perturbation by the corresponding integral
∥f p ( Ã) -f p (A)∥ ≤ 1 2π Γ ∥f (z)[(zI -Ã) -1 -(zI -A) -1 ]∥dz =: F (f ).(2)
This inequality makes the interaction of A and E explicit and is widely used in functional perturbation analysis, e.g., [19,26,28,32,33,37]. However, obtaining a sharp bound on its right-hand side remains a formidable analytical challenge.
Step 2: Bounding F ≤ 2F 1 via the contour bootstrapping method. Attempts to control F (f ), the right-hand side of (2), often use series expansion and analytical tools. By repeatedly applying the resolvent formula, one can expand
f (z)[(zI -Ã) -1 -(zI -A) -1 ] into ∞ s=1 f (z)(zI - A) -1 [E(zI -A) -1 ] s
. This yields the bound:
F (f ) ≤ ∞ s=1 F s (f ), where F s (f ) = 1 2π Γ f (z)(zI -A) -1 [E(zI -A) -1 ] s |dz|.
One needs to estimate F s (f ) for each s. For example, when f (z) = 1, [26, Part 2] bounds F s (1)
by O ∥E∥ s Γ |dz| min i∈[n] |z-λi| s+1 = O [(||E||/δ p )
s ], where Γ is a union of vertical lines isolating {λ i , i ∈ p}, yielding the Davis-Kahan bound O (∥E∥/δ p ). However, for f (z) = z (relevant for low-rank perturbations), this approach fails as |z| → ∞. These estimates are highly nontrivial and rely on deep analytical techniques, making generalization to arbitrary f challenging.
Moreover, for f (z) = 1, under certain conditions, the dominant term is F 1 (f ), i.e., F (f ) = O(F 1 (f )); see, e.g., [22,27,32,33,37]. In particular, using contour-bootstrapping technique, the authors in [37] proved F (f (z) = 1) ≤ 2F 1 (f (z) = 1). Inspired by this technique, we prove that F (f ) ≤ 2F 1 (f ) for any entire function f . Our contour bootstrapping argument is designed to prove Lemma 3.1. Our argument is concise and novel, avoiding the need for series expansion and convergence analysis. In the context of standard low-rank approximations, where f (z) ≡ z and f p (A) = A p , we write F (z) and F 1 (z) instead of F (f ) and F 1 (f ) respectively.
Step 3: Construction of Γ, F 1 (z)-estimation, and proof completion of Thm. 2.1. Given Lemma 3.1, we now need to carefully choose the contour Γ and estimate F 1 (f ). Constructing Γ (so that the perturbation analysis via contour integration provides a sharp bound) is delicate; for example, the classical pick of two vertically parallel lines and any Γ placed too near any λ i can blow up F 1 (z) to infinity. Indeed, we tailor Γ w.r.t F 1 (z) as follows. First, we choose Γ to be rectangular as this simplifies integration. To control the factor (zI -A) -1 in the expression of F 1 (f ), we need to ensure that the distance |z -λ i | for any z ∈ Γ and i ∈ [n] are relatively large. Since Γ separates λ p and λ p+1 , this minimal distance min z∈Γ,i∈[n] |z -λ i | cannot exceed Θ(δ p ). Thus, we simply construct Γ through the midpoint
x 0 = λp+λp+1 2
. Finally, by setting the contour sufficiently high in the complex plane (while avoiding excessive height to prevent |f (z)| from diverging), we ensure that the primary contribution to the integral is from the vertical segments of Γ. This is because the distance |z -λ i | is minimized on these segments. Note that, under the assumption 4∥E∥ < δ p , this construction ensures that the p-leading eigenvalues of A and Ã are well aligned inside the contour. Now, in particular, to prove Theorem 2.1, we will estimate
2πF 1 (z) = Γ ∥z(zI -A) -1 E(zI -A) -1 ∥ |dz|, in which the contour Γ is set
to be a rectangle with vertices (x 0 , T ), (x 1 , T ), (x 1 , -T ), (x 0 , -T ), where x 0 := λ p -δ p /2, x 1 := 2λ 1 , T := 2λ 1 . Then, we split Γ into four segments:
Γ 1 := {(x 0 , t)| -T ≤ t ≤ T }; Γ 2 := {(x, T )|x 0 ≤ x ≤ x 1 }; Γ 3 := {(x 1 , t)|T ≥ t ≥ -T }; Γ 4 := {(x, -T )|x 1 ≥ x ≥ x 0 }. λ p+1 λ p λ 1 Γ 3 Γ 1 Γ 2 Γ 4
Given the construction of Γ, we have 2πF 1 = 4 k=1 M k , where
M k := Γ k z(zI -A) -1 E(zI -A) -1 |dz|.
Intuitively, we set T, x 1 large (= 2∥A∥) so that the main term is the integral along Γ 1 , i.e., M 1 . Indeed, factoring our E and using the fact that |z -
λ i | ≥ |z -λ p | = δ 2 p + t 2 for all 1 ≤ i ≤ n and z ∈ Γ 1 := {(x 0 , t)| -T ≤ t ≤ T }, we have M 1 ≤ Γ1 ∥E∥ • |z| min i∈[n] |z-λi| 2 |dz| ≤ ∥E∥ • T -T √ x 2 0 +t 2 (δp/2) 2 +t 2 dt. Directly compute the integral T -T √ x 2 0 +t 2 (δp/2) 2 +t 2 dt (see Section E.3), we obtain: M 1 ≤ ∥E∥ • O (x 0 /δ p ) = O (∥E∥λ p /δ p ) .
By a similar manner, replace Γ 1 by Γ 3 := {(x 1 , t)| -T ≤ t ≤ T }, we have
M 3 ≤ ∥E∥ • Γ3 |z| min i∈[n] |z-λi| 2 |dz| ≤ ∥E∥ • Γ3 √ x 2 1 +t 2 λ 2 1 +t 2 dt,
where the last inequality follows the fact that min i∈[n] |z -
λ i | = |z -λ 1 | = (x 1 -λ 1 ) 2 + t 2 = λ 2 1 + t 2 . Directly compute the integral T -T √ x 2 1 +t 2 λ 2 1 +t 2 dt (see Section E.3), we obtain: M 3 ≤ ∥E∥ • O (x 1 /λ 1 ) = O (∥E∥) .
Similarly, M 2 , M 4 = O(∥E∥) ( Section E.2). These estimates on M 1 , M 2 , M 3 , M 4 imply F 1 (z) = O ∥E∥ • λp δp , which together with Lemma 3.1 proves Theorem 2.1.
Proving the contour bootstrapping lemma (Lemma 3.1). The first observation is that using the Sherman-Morrison-Woodbury formula M -1 -(M + N ) -1 = (M + N ) -1 N M -1 [20] and the fact that Ã = A + E, we obtain
(zI -A) -1 -(zI -Ã) -1 = (zI -A) -1 E(zI -Ã) -1 .
Using this, we can rewrite
F (f ) = 1 2π Γ ∥f (z)(zI -A) -1 E(zI -Ã) -1 ∥ |dz| as 1 2π Γ ∥f (z)(zI -A) -1 E(zI -A) -1 -f (z)(zI -A) -1 E[(zI -A) -1 -(zI -Ã) -1 ]∥ |dz|. Using triangle inequality, we first see that F (f ) is at most Γ ∥f (z)(zI-A) -1 E(zI-A) -1 ∥|dz| 2π + Γ ∥f (z)(zI-A) -1 E[(zI-A) -1 -(zI-Ã) -1 ]∥|dz| 2π .
Next is the key observation that the second term in the equation above can be rearranged and upperbounded as follows so that the original perturbation appears again:
maxz∈Γ∥(zI-A) -1 E∥ 2π Γ ∥f (z)[(zI -A) -1 -(zI -Ã) -1 ]∥ |dz|. Thus, we have F (f ) ≤ F 1 (f ) + max z∈Γ (zI -A) -1 E • F (f ).(3)
Now we need our gap assumption that 4∥E∥ < δ p and the construction of Γ, which imply min z∈Γ,i∈[n] |z -λ i | ≥ δ p /2 ≥ 2∥E∥. Therefore, we have
max z∈Γ (zI -A) -1 E ≤ max z∈Γ ∥(zI -A) -1 ∥ • ∥E∥ = ∥E∥ min z∈Γ,i∈[n] |z-λi| ≤ ∥E∥ 2∥E∥ = 1 2 .
Together with (3), it follows that F (f ) ≤ F 1 (f ) + 1 2 F (f ). Therefore, 1 2 F (f, S) ≤ F 1 (f, S). This proves Lemma 3.1. 2 Remark 3.2. Using a similar strategy, one can prove that
F 1 (f ) ≤ max z∈Γ ∥f (z)∥ • 1 2π Γ ∥(zI -A) -1 E(zI -A) -1 ∥|dz| ≤ max z∈Γ ∥f (z)∥ • 2∥E∥
δp ; see Appendix F. Together, this estimate and Lemma 3.1 prove Theorem 2.3.
Second upper bound of M 1 and proof of Theorem 2.2. The key idea of the second bound is to replace (zI -A) -1 by its spectral expansion n i=1
uiu ⊤ i z-λi . Hence, M 1 is rewritten as Γ1 ∥ 1≤i,j≤n z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥dz.
There are n 2 terms in the expression, and the direct use of the triangle inequality cannot provide a good estimate. The next key trick is grouping up the r-top eigenvectors {u i } r i=1 . Formally, M 1 is at most
Γ1 ∥ 1≤i,j≤r z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥|dz| + Γ1 ∥ n≥i,j>r z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥|dz| + Γ1 ∥ i≤r<j i>r≥j z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥|dz|.
To estimate the first term, we apply the triangle inequality. For each term, we factor out components independent of z and carefully evaluate the integral. Specifically, by the triangle inequality, the first term is at most
1≤i,j≤r Γ1 ∥ z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥|dz| = 1≤i,j≤r Γ1 |u ⊤ i Euj |•∥uiu ⊤ j ∥•|z| |(z-λi)(z-λj )| |dz|. Since max 1≤i,j≤r |u ⊤ i Eu j | ≤ x, ∥u i u ⊤ j ∥ = 1, and Γ 1 := {z | z = x 0 + it, -T ≤ t ≤ T }, the r.h.s. is at most i,j≤r x T -T √ x 2 0 +t 2 √ ((x0-λi) 2 +t 2 )((x0-λj ) 2 +t 2 ) dt ≤ i,j≤r x T -T |x0|+|t| √ ((x0-λi) 2 +t 2 )((x0-λj ) 2 +t 2 )
dt. 2 The gap assumption 4∥E∥ < δp and Weyl's inequality ensure that λi is inside the contour Γ if and only if δp + 2 log 3T δp = Õ r 2 x λp δp . To estimate the second term, we apply matrix-norm inequalities to factor out E from the integral:
1 ≤ i ≤ p.
Γ1 ∥ n i,j=r z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j ∥|dz| ≤ Γ1 |z| • ∥ n≥i>r uiu ⊤ i z-λi ∥ • ∥E∥ • ∥ n≥i>r uiu ⊤ i z-λi ∥|dz|, which is at most ∥E∥ Γ1 |z| min n≥i>r |z-λi| 2 |dz| = ∥E∥ T -T √ x 2 0 +t 2 min n≥i>r [(x0-λi) 2 +t 2 ]
dt. Moreover, by the construction of Γ 1 and the definition of r,
|x 0 -λ i | = |(λ p + λ p+1 )/2 -λ i | ≥ |(λ p + λ p+1 )/2 -λ r+1 | ≥ λp-λr+1 2 ≥ λp 4
, where the first inequality follows the fact i > r. Thus, the second term is at most
∥E∥ T -T √ x 2 0 +t 2 t 2 +(λp/4) 2 dt ≤ Õ(∥E∥);
see Section E.1 for the detailed estimation.
Similar to estimating the second term, the last term is also Õ(∥E∥). Combining the estimates on three parts of M 1 , we obtain M 1 ≤ Õ r 2 x λp δp + ∥E∥ . Consequently, by Lemma 3.1, we finally have
F (z) ≤ 2F 1 (z) = O(M 1 ) = Õ ∥E∥ + r 2 x λp δp
as desired.
this section cite: ['b18', 'b25', 'b34', 'b18', 'b25', 'b27', 'b31', 'b32', 'b36', 'b21', 'b26', 'b31', 'b32', 'b36', 'b36', 'b19']

Section: Empirical results
In this section, we empirically evaluate the sharpness of our spectral-gap bound (Theorem 2.1) in real-world settings central to privacy-preserving low-rank approximation. We compare: (1) the actual spectral error ∥ Ãp -A p ∥, (2) our theoretical boundfoot_0 7∥E∥ • λp δp , (3) and the classical Eckart-Young-Mirsky (EYM) bound 2(∥E∥ + λ p+1 ). Each quantity is computed over 100 trials and 20 noise levels. Because prior bounds [15,29,30] apply only to Gaussian noise and involve unspecified constants, we exclude them from this evaluation.
this section cite: ['b14', 'b28', 'b29']

Section: Setting.
We study three covariance matrices A from the UCI Machine Learning Repository [13]: the 1990 US Census (n = 69), the 1998 KDD-Cup network-intrusion data (n = 416), and the Adult dataset (n = 6). These matrices-henceforth Census, KDD, and Adult-are standard benchmarks in DP PCA [3,11,29]. The low-rank parameter p is chosen so that the Frobenius norm of A p contains > 99% of the Frobenius norm of A, giving p = 10 for A = Census, p = 2 for A = KDD, and p = 4 for A = Adult [29,Section B].
Each matrix is perturbed with either GOE noise E 1 or Rademacher noise E 2 , scaled by twenty evenly spaced factors ranging from 0 to 1. Note that with high probability [41,43],
∥E 1 ∥ = ∥E 2 ∥ = (2 + o(1)) √ n, so the gap condition 4∥E k ∥ < δ p simplifies to 8 √ n < δ p .
For Census (n = 69, p = 10), we have δ p ≈ 1433.99 > 8 √ 69 ≈ 66.45. For KDD (n = 416, p = 2), we get δ p ≈ 351.3 > 8 √ 416 ≈ 163.2. For Adult (n = 6, p = 4), we find δ p ≈ 37.02 > 8 √ 6 ≈ 19.6. Hence 4∥E k ∥ < δ p holds in all tested configurations.
Evaluation. Each data matrix is preprocessed as follows: non-numeric entries are replaced with 0; rows shorter than the maximum length are padded with zeros; each row is scaled to unit Euclidean norm; and each column is centered to have zero mean. We compute the covariance matrix A := M ⊤ M , where M is the processed data matrix. For each configuration (A, E k , p), we run 100 independent trials. In each trial, we perturb A with noise E k ∈ {E 1 , E 2 } to form Ã = A + E k , compute its best rank-p approximation Ãp , and measure the spectral error ∥ Ãp -A p ∥. We compare this with our bound 7∥E k ∥ • λ p /δ p and the classical EYM bound 2(∥E k ∥ + λ p+1 ). Following standard practice, all reported values are averaged over 100 trials, with error bars shown for Actual Error and Our Bound (cap width = 3pt).
this section cite: ['b12', 'b2', 'b10', 'b28', 'b28', 'b40', 'b42']

Section: Result and conclusion.
Across all experiments-the 69 × 69 US Census, the 416 × 416 KDD-Cup, and the 6 × 6 Adult matrix-our bound closely matches the empirical error for both Gaussian and Rademacher noise (Figs. 12), consistently outperforming the classical EYM estimate. (Note: the error bars for Census and KDD are too small to see.) Over all three benchmark datasets, two distinct noise models, and twenty escalation levels per model, our spectral-gap estimate never deviates from the observed error by more than a single order of magnitude. This uniform tightness, achieved without any dataset-specific tuning, demonstrates that the bound of Theorem 2.1 is not merely sufficient but practically sharp across matrix sizes spanning two orders of magnitude and privacy-motivated perturbations spanning the entire operational range. Consequently, the bound can serve as a reliable, application-agnostic error certificate for low-rank covariance approximation in both differential-privacy pipelines and more general noisy-matrix workflows.
this section cite: []

Section: Conclusion and future work
We established new spectral norm perturbation bounds for low-rank approximations that explicitly account for the interaction between a matrix A and its perturbation E. Our results extend the Eckart-Young-Mirsky theorem, improving upon prior Frobenius-norm-based analyses. A key contribution is a novel application of the contour bootstrapping technique, which simplifies spectral perturbation arguments and enables refined estimates. Our bounds provide sharper guarantees for differentially private low-rank approximations with high probability spectral norm bounds that improve upon prior results. We also extended our approach to general spectral functionals, broadening its applicability.
Several limitations and open questions remain. While spectral norm error bounds are standard and widely used in both theoretical and applied settings, can we extend our analysis to other structured metrics such as Schatten-p norm, the Ky Fan norm, or subspace affinity norm? Can our bounds be further refined for matrices with specific spectral structures, such as polynomial or exponential decay? What can be the threshold for the gap assumption so that one still obtains a meaningful bound beyond the Eckart-Young-Mirsky theorem? 4 Additionally, real-world noise often exhibits structured dependencies-can our techniques be adapted to handle sparse or correlated perturbations?
this section cite: []

Section: A Limitations of prior approaches
This section explains why existing perturbation methods fail to yield spectral norm bounds of the form ∥ Ãp -A p ∥ that incorporate interaction between A and the perturbation E.
Eckart-Young-Mirsky: lack of interaction sensitivity. Let σ 1 ≥ σ 2 ≥ • • • ≥ σ n ≥ 0 denote the singular values of A. The Eckart-Young-Mirsky theorem gives ∥A -A p ∥ = σ p+1 , and by the triangle inequality:
∥ Ãp -A p ∥ ≤ ∥A -A p ∥ + ∥ Ã -A∥ + ∥ Ã -Ãp ∥ ≤ σ p+1 + ∥E∥ + σp+1 ≤ 2(σ p+1 + ∥E∥),
where the final step uses Weyl's inequality [46]. While this bound is assumption-free, it is uninformative in regimes where σ p+1 ≫ ∥E∥, which are common in practice. The key limitation is that the triangle inequality treats A and E independently, failing to capture how structure or spectral gaps in A might mitigate the effect of E.
Mangoubi-Vishnoi: Frobenius only, spectral norm intractable. The strategy of [29,30] models noise as a continuous-time matrix-valued Brownian motion:
A(t) := A + tE = A + B(t),
with eigen-decomposition
A(t) = U (t) Diag[λ 1 (t), . . . , λ n (t)] U (t) ⊤ , where U (t) = [u i (t)] and λ 1 (t) ≥ • • • ≥ λ n (t). The rank-p approximation at time t is A p (t) = U (t) Diag[λ 1 (t), . . . , λ p (t), 0, . . . , 0] U (t) ⊤ .
The total perturbation is then expressed as an integral:
Ãp -A p = 1 0 dA p (t).
Using properties of Dyson Brownian motion and Itô calculus, they derive a Frobenius-norm identity:
E 1 0 dA p (t) 2 F = n i=1 1 0   E   j̸ =i (λ i -λ j ) 2 (λ i (t) -λ j (t)) 2   +   j̸ =i λ i -λ j (λ i (t) -λ j (t)) 2   2    dt.
Bounding these expressions depends on repulsion properties of the eigenvalues; for GOE matrices, Weyl's inequality suffices, while for GUE matrices, stronger gap estimates are used.
Although this method captures the spectral structure of A and interaction with E, it only yields Frobenius-norm bounds. Extending it to the spectral norm would require controlling
∥ Ãp -A p ∥ = 1 0 dA p (t) ,
which entails bounding the operator norm of the full stochastic process. This requires detailed control over the dynamics of U (t) and λ(t), including their correlations-none of which are tractable with current techniques. Moreover, for generalized functionals such as ∥f p ( Ã) -f p (A)∥, the problem becomes even harder: one must analyze 1 0 df p (A(t)), which involves matrix-valued analytic functions under random perturbation, a setting far beyond existing random matrix tools.
In contrast, our approach bypasses these limitations by using a complex-analytic representation of spectral projectors that directly captures interaction between A and E, yielding sharp spectral norm bounds under broad assumptions. B Comparison of error metrics This section studies three common metrics for low-rank approximation under perturbation-namely: -the spectral-norm error ∥ Ãp -A p ∥, -the Frobenius-norm error ∥ Ãp -A p ∥ F , and -the "changein-error" ∥A -A p ∥ -∥A -Ãp ∥ .
We compare these metrics both empirically (through Monte Carlo simulations) and theoretically. Empirically, we examine how the metrics behave under Gaussian noise applied to both synthetic and real-world matrices (Figure 3). Theoretically, we analyze their interpretability and limitations, highlighting that while Frobenius norms capture aggregate error and change-in-error quantifies residual shifts, only the spectral norm controls worst-case subspace distortion.
A simple 2 × 2 example (Example B.1) further illustrates how residual-based measures can completely mask subspace drift, underscoring the robustness and interpretability of the spectral norm for tasks such as private low-rank approximation.
Empirical comparison of utility metrics. We perform three Monte Carlo experiments under additive Gaussian perturbations. The first uses a synthetic PSD matrix A ∈ R 50×50 with exponentially decaying eigenvalues λ i = 0.8 i , and sets p = 5. The second and third use real-world covariance matrices derived from: -the 1990 US Census dataset (n = 69), -the 1998 KDD-Cup dataset (n = 416).
All datasets are drawn from the UCI Machine Learning Repository [13] and have been widely used in private matrix approximation and PCA [30,29,11].
In each setting, we compute the best rank-p approximation A p , perturb A with symmetric Gaussian noise of varying standard deviation σ, and measure:
1. Spectral norm deviation: ∥ Ãp -A p ∥, 2. Frobenius norm deviation: ∥ Ãp -A p ∥ F , 3. Change-in-error: ∥A -A p ∥ -∥A -Ãp ∥ .
As shown in Figure 3, the Frobenius norm error grows fastest, reflecting total energy deviation. The change-in-error metric remains much smaller and, in the real-world cases, nearly flat, suggesting it may fail to capture meaningful distortion. Notably, in the synthetic case (left), the spectral norm error closely tracks the change-in-error-despite their differing intent-which may result from nearalignment of the top subspaces. However, such behavior is not guaranteed in general.
Theoretical distinction between utility metrics. Frobenius norm bounds of the form ∥ Ãp -A p ∥ F ≤ ε F aggregate squared deviations across all directions, but may hide large errors in in-dividual components. Spectral norm bounds ∥ Ãp -A p ∥ ≤ ε directly constrain the worst-case deviation and are thus more reliable in sensitive applications such as differentially private PCA.
In contrast, residual-error metrics such as ∥A -A p ∥ -∥A -Ãp ∥ are commonly used for their analytical convenience. However, they reflect only changes in residual energy and are insensitive to subspace movement. In particular, this metric can be nearly zero even when the top-p eigenspaces have shifted significantly.
this section cite: ['b45', 'b28', 'b29', 'b12', 'b29', 'b28', 'b10']

Section: Given the spectral decompositions
A p = U p Diag(λ 1 , . . . , λ p , 0, . . . , 0) U ⊤ p , Ãp = Ũp Diag( λ1 , . . . , λp , 0, . . . , 0) Ũ ⊤ p , the change-in-error vanishes whenever U p U ⊤ p ≈ Ũp Ũ ⊤ p and λ p+1 is large. Such conditions are typical when noise E is small and p ≤ sr(A) := n i=1 λ i /λ 1 . Moreover, standard perturbation results imply [33,38].
∥U p U ⊤ p -Ũp Ũ ⊤ p ∥ = Õ ∥E∥ λ p + 1 δ p
Example B.1 (Rank-1 rotation in R 2 ). Let A = 1 0 0 0 , p = 1,
so that A p = A. Define the rotated matrix
Ã = R θ AR ⊤ θ , where R θ = cos θ -sin θ sin θ cos θ .
Then Ãp = Ã, and although the top eigenspace has rotated by θ, the change-in-error is zero:
∥A -A p ∥ = ∥A -Ãp ∥ = 0. Yet the true subspace drift is visible in: ∥ Ãp -A p ∥ = | sin θ|, ∥ Ãp -A p ∥ F = √ 2| sin θ|.
This example highlights the limitations of residual-based utility metrics and illustrates why spectral norm deviation provides a more reliable and interpretable signal of approximation quality under perturbation.
In summary, both our analysis and experiments support the use of the spectral norm as the most informative and robust error metric for evaluating private low-rank approximations. Unlike Frobenius and residual metrics, it captures the worst-case directional distortion and provides a tighter connection to subspace stability.
this section cite: ['b32', 'b37']

Section: C Empirical evaluation beyond gap assumption
In this section, we empirically compare (1) the actual spectral error ∥ Ãp -A p ∥, (2) our theoretical bound 7∥E∥ • λp δp , (3) and the classical Eckart-Young-Mirsky (EYM) bound 2(∥E∥ + λ p+1 ) in the setting beyond the gap assumption that 4∥E∥ < δ p .
Setup. We conducted a simulation on a covariance matrix A with n = 2000, derived from the Alon colon-cancer microarray dataset [2]. The low-rank parameter p is chosen so that the Frobenius norm of A p contains > 95% of the Frobenius norm of A, giving p = 9 with λ p ≈ 46.29. We first computed δ p . Gaussian noise was then added in the form E = α • N (0, I n ), with α chosen over 11 evenly spaced values such that ∥E∥ δ p ∈ {0.05, 0.10, . . . , 0.50}.
For each α, we computed the following quantities:
• the true error: ∥ Ãp -A p ∥,
• the classical EYM bound: 2(∥E∥ + σ p+1 ),
• our bound: 7∥E∥ • λp δp , • the ratios our bound true error and our bound classical bound .
Results. Table 2 summarizes the results. The ratio our bound true error remains remarkably stable even beyond the regime 4∥E∥ < δ p (i.e., ∥E∥ δp < .25), and our bound outperforms the classical bound precisely when 4∥E∥ < δ p (i.e., ∥E∥ δp < .25).
this section cite: ['b1']

Section: D Extensions of Theorem 2.1 and Theorem 2.2 to the symmetric matrices
In this section, we extend Theorem 2.1 and Theorem 2.2 to the setting where A is a symmetric matrix. These extensions are naturally important since the data in real-world applications is often arbitrary, making it natural for the eigenvalues of A to span both signs. While singular value decomposition (SVD) could be used to apply Theorem 2.1 or Theorem 2.2, singular value gaps are typically small. By working directly with eigenvalues, we exploit the fact that the eigenvalue gap
δ k = λ k -λ k+1 is significantly large when λ k • λ k+1 < 0.
this section cite: []

Section: D.1 Extension of Theorem 2.1 to the symmetric matrices
To simplify the presentation, we assume that the eigenvalues (singular values) are different, so the eigenvectors (singular vectors) are well-defined (up to signs). However, our results hold for matrices with multiple eigenvalues. Let A, E be n × n real symmetric matrices, and let 1 ≤ p ≤ n denote the rank of approximation. Let λ k be the kth largest eigenvalue of A and u k be the corresponding orthonormal eigenvector. Let Ã := A + E. Let A p , Ãp denote the best rank-p approximations of A and Ã respectively. Define 1 ≤ k ≤ p such that the set of the top p singular values corresponds to {λ π(1) , . . . , λ π(p
) } = {λ 1 , . . . , λ k > 0 ≥ λ n-(p-k)+1 , . . . , λ n }. In other words, the pth singular value of A is either λ k or |λ n-(p-k)+1 |. Let δ i := λ i -λ i+1 , for i ∈ [n -1]. Theorem 2.
1 is extended to the following result. Theorem D.1 (Extension of Theorem 2.1 to the symmetric matrices). If 4∥E∥ ≤ min{δ k , δ n-(p-k) }, and 2∥E∥ < σ p -σ p+1 , then
Ãp -A p ≤ 6∥E∥ log 6σ1 δ k + λ k δ k + log 6σ1 δ n-(p-k) + |λn-(p-k)+1| δ n-(p-k) . Note that when A is not PSD, {| λ1 |, . . . , | λk |, | λn-(p-k)+1 |, . . . , | λn |} may not correspond to the p leading singular values of Ã.
This issue is resolved by enforcing the singular-value gap condition σ p -σ p+1 > 2∥E∥. Indeed, by Weyl's inequality, given σ p -σ p+1 > 2∥E∥, we have
λk ≥ λ k -∥E∥ ≥ σ p -∥E∥ = σ p+1 + δ -∥E∥ ≥ |λ n-(p-k) | + δ -∥E∥ ≥ | λn-(p-k) | + δ -2∥E∥ > | λn-(p-k) |, here δ = σ p -σ p+1 . By a similar argument, we also have | λn-(p-k)+1 | > λk+1 . Therefore, { λπ(1) , λπ(2) , . . . , λπ(p) } = { λ1 ≥ λ2 ≥ . . . ≥ λk > 0 ≥ λn-(p-k)+1 ≥ λn-(p-k)+2 ≥ . . . ≥ λn },
as we want. Note that the gap condition of eigenvalues cannot guarantee this fact. For example, consider the following matrices
A = 30 √ n 0 0 -28 √ n , E = -2 √ n 0 0 -2 √ n , then Ã = 28 √ n 0 0 -30 √ n .
Here, clearly, S = {1}, S = {1} and
|λ 1 | is the largest singular value of A, but | λ1 | is not the largest singular value of Ã ( λ1 is still the largest eigenvalue).
Proof of Theorem D.1 Let 1 ≤ k ≤ p be a natural number such that {λ π(1) , λ π(2) , . . . , λ π(p) } = {λ 1 , λ 2 , . . . , λ k > 0 ≥ λ n-(p-k)+1 , λ n-(p-k)+2 , . . . , λ n }.
this section cite: []

Section: Thus, we can split A
p as A k + B p-k , in which B p-k = n≥i≥n-(p-k)+1 λ i u i u ⊤ i .
Similarly, Ãp = Ãk + Bp-k . Therefore,
Ãp -A p = Ãk + Bp-k -A k -B p-k ≤ Ãk -A k + Bp-k -B p-k .
this section cite: []

Section: Applying the contour bootstrapping argument on
Ãk -A k with contour Γ [1] and on Bp-k -B p-k with another contour Γ [2] (we define these contours later), we obtain
∥ Ãk -A k ∥ 2 ≤ F [1] 1 := 1 2π Γ [1] z(zI -A) -1 E(zI -A) -1 |dz|, ∥ Bp-k -B p-k ∥ 2 ≤ F [2] 1 := 1 2π Γ [2] z(zI -A) -1 E(zI -A) -1 |dz|, and hence, Ãp -A p ≤ 2 F [1] 1 + F [2] 1 .(4)
We set Γ [1] and Γ [2] to be rectangles, whose vertices are Γ [1] : (a 0 , T ), (a 1 , T ), (a 1 , -T ), (a 0 , -T ) with a 0 := λ k -δ k /2, a 1 := 2σ 1 , T := 2σ 1 ; and Γ [2] :
(b 0 , T ), (b 1 , T ), (b 1 , -T ), (b 0 , -T ) with b 0 := λ n-(p-k)+1 +δ n-(p-k) /2, b 1 := -2σ 1 , T := 2σ 1 .
Now, we are going to bound F [1] 1 . First, we split Γ [1] into four segments:
• Γ 1 := {(a 0 , t)| -T ≤ t ≤ T }.
• Γ 2 := {(x, T )|a 0 ≤ x ≤ a 1 }.
• Γ 3 := {(a 1 , t)|T ≥ t ≥ -T }.
• Γ 4 := {(x, -T )|a 1 ≥ x ≥ a 0 }. 2 |dz| for l = 1, 2, 3, 4. We use the following lemmas, whose proofs are delayed to the next section. Lemma D.2. Under the assumption of Theorem D.1,
λ k+1 λ k λ 1 Γ 3 Γ 1 Γ 2 Γ 4 Re (z) = 0 Therefore, F [1] 1 = 4 l=1 Γ l z(zI -A) -1 E(zI -A) -1 |dz|. Notice that z(zI -A) -1 E(zI -A) -1 ≤ ∥E∥ |z| min i∈[n] |z-λi| 2 , we further obtain 2πF [1] 1 ≤ ∥E∥ 4 l=1 N l , in which N l := Γ l |z| mini |z-λi|
N 1 ≤ 2πa0 δ k + 4 log 3T δ k .
Lemma D.3. Under the assumption of Theorem D.1,
N 3 ≤ πa1 |a1-λ1| + 4 log 3T a1-λ1 .
Lemma D.4. Under the assumption of Theorem D.1,
N 2 , N 4 ≤ √ 2(a1-a0) T , Since p < n, then k + 1 > n -(p -k) + 1 and hence k + 1 / ∈ {π(1), . . . , π(p)}. It means |λ k+1 | ≤ λ k . Thus 0 ≤ a 0 ≤ λ k , and hence N 1 ≤ 2πλ k δ k + 4 log 6σ1 δ k .
By the setting that a 1 = T = 2σ 1 ,
N 2 , N 4 ≤ √ 2a1 T = √ 2, N 3 ≤ 2πσ1 2σ1-λ1 + 4 log 3T a1-λ1 ≤ 2πσ1 σ1 + 4 log 6σ1 σ1 = 2π + 4 log 6.
Thus, using above estimates, we obtain
F [1] 1 ≤ ∥E∥ 2π 2π + 4 log 6 + 2 √ 2 + 2πλ k δ k + 4 log 6σ1 δ k ≤ ∥E∥ 2π 15 log 6σ1 δ k + 2πλ k δ k ≤ 3∥E∥ log 6σ1 δ k + λ k δ k .(5)
Applying a similar argument on contour Γ [2] , we obtain
F [2] 1 ≤ 3∥E∥ log 6σ1 δ n-(p-k) + |λn-(p-k)+1| δ n-(p-k) .(6)
Combining ( 4), ( 5) and ( 6), we complete our proof.
this section cite: ['b0']

Section: D.2 Extension of Theorem 2.2 to the symmetric matrices
Let A be a symmetric matrix with eigenvalues λ 1 ≥ λ 2 ≥ • • • ≥ λ n , in which λ n is not necessarily positive. Recall the setting from the previous section that 1 ≤ k ≤ p is the positive integer such that the set of the top p singular values is {λ π(1) , . . . , λ π(p) } = {λ 1 , . . . , λ k > 0 ≥ λ n-(p-k)+1 , . . . , λ n }. To extend Theorem 2.2, we first generalize the definition of the halving distance r and interaction term x as follows. Let r 1 , r 2 respectively be the smallest positive integer satisfying λ k 2 ≤ λ k -λ r1+1 , and
|λ n-(p-k)+1 | 2 ≤ λ n-r2+1 -λ n-(p-k)+1 .
Define the "halving distance" r := max{r 1 , r 2 }. Next, let x 1 := max 1≤i,j≤r1 |u ⊤ i Eu j | and
x 2 := max 1≤i,j≤r2 |u ⊤ n-i+1 Eu n-j+1 |.
Define the interaction parameter x := max{x 1 , x 2 }. Theorem D.5 (Extension of Theorem 2.2 to the symmetric matrices). Assume that 4∥E∥ ≤ min{δ k , δ n-(p-k) } and 2∥E∥ < σ p -σ p+1 , then
Ãp -A p ≤ 12 ∥E∥ + r 2 x log 6σ1 δ k + log 6σ1 δ n-(p-k) + 30r 2 x λ k δ k + |λn-(p-k)+1| δ n-(p-k)
.
Proof of Theorem D.5 First, we still split ( Ãp , A p ) into (A k , B p-k , Ãk , Bp-k ) and apply the contour bootstrapping argument on Ãk -A k , Bp-k -B p-k . We also obtain
Ãp -A p ≤ 2 F [1] 1 + F [2] 1 .
However, we will treat F
[1]
1 , F [2] 1 a bit differently. Indeed,
2πF [1] 1 ≤ M 1 + ∥E∥ (N 2 + N 3 + N 4 ) , in which M 1 := Γ1 z(zI -A) -1 E(zI -A) -1 |dz| = Γ1 1≤i,j≤n z (z-λi)(z-λj ) u i u i Eu j u ⊤ j |dz|,
this section cite: ['b1']

Section: and
N l := Γ l |z| min i∈[n] |z-λi| 2 |dz| for l ∈ {2, 3, 4}. We additionally use the following lemma (its proof will be delayed in the next section). Lemma D.6. Under the assumption of Theorem D.5,
M 1 ≤ r 2 x 2πa0 δ k + 2 log 6σ1 δ k + (20 + 4π/ log(10)∥E∥ log 10σ1 δ k .
Together with the estimates for N 2 , N 3 , N 4 from the previous section, we obtain
2πF [1] 1 ≤ r 2 x 2πλ k δ k + 2 log 3T δ k + (20 + 4π log 10 )∥E∥ log 5T δ k + ∥E∥ 2 √ 2 + 2π + 4 log 6 ≤ r 2 x 2πλ k δ k + 2 log 6σ1 δ k + (20 + 4π log 10 )∥E∥ log 10σ1 δ k + 2 √ 2+2π+4 log 6 log 10 ∥E∥ log 10σ1 δ k . Thus, F [1] 1 ≤ 6 ∥E∥ log 10σ1 δ k + r 2 x λ k δ k + r 2 x log 10σ1 δ k .(7)
Similarly,
F [2] 1 ≤ 6 ∥E∥ log 10σ1 δ n-(p-k) + r 2 x |λn-(p-k)+1| δ n-(p-k) + r 2 x log 10σ1 δ n-(p-k) .(8)
Therefore, combining (4), (7), and (8), we finally obtain
Ãp -A p ≤ 12 ∥E∥ log 36σ 2 1 δ k δ n-(p-k) + r 2 x λ k δ k + r 2 x |λn-(p-k)+1| δ n-(p-k) + r 2 x log 36σ 2 1 δ k δ n-(p-k)
.
this section cite: ['b6']

Section: E Estimating integrals over segments
In this section, we present in detail the integral estimations mentioned in the previous section: Lemma D.2, Lemma D.3, Lemma D.6 (integration over vertical segments); and Lemma D.4 (integration over horizontal segments) . We first present a technical lemma, which is used several times in the upcoming sections. Lemma E.1. Let a, T be positive numbers such that a ≤ T . Then,
T -T 1 t 2 +a 2 dt ≤ π a .
Proof of Lemma E.1 We have
T -T 1 t 2 +a 2 dt = 2 T 0 1 t 2 +a 2 = 2 a arctan(T /a) ≤ 2 a • π 2 = π a .
this section cite: []

Section: E.1 Estimating integrals over vertical segments for interaction-dependent bound
In this Section, we now estimate M 1 -integral over the left vertical segment (prove Lemma D.6) and estimate N 3 -the integral over the right vertical segment (prove Lemma D.3). First, we estimate M 1 as follows.
Using the spectral decomposition (zI
-A) -1 = n i=1 uiu ⊤ i (z-λi) , we can rewrite M 1 as M 1 = Γ1 n≥i,j≥1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz|.
Define x 1 := max 1≤i,j≤r1 u ⊤ i Eu j . By the triangle inequality, M 1 is at most
Γ1 1≤i,j≤r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| + Γ1 n≥i,j>r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| + Γ1 i≤r1<j i>r1≥j z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz|.
Consider the first term, by the triangle inequality, we have
Γ1 1≤i,j≤r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| ≤ 1≤i,j≤r1 Γ1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| = 1≤i,j≤r1 Γ1 |u ⊤ i Euj |•∥uiu ⊤ j ∥•|z| |(z-λi)(z-λj )| |dz| ≤ i,j≤r1 x 1 T -T √ a 2 0 +t 2 √ ((a0-λi) 2 +t 2 )((a0-λj ) 2 +t 2 ) dt (since max 1≤i,j≤r1 |u ⊤ i Eu j | ≤ x 1 , ∥u i u ⊤ j ∥ = 1, and Γ 1 := {z | z = a 0 + it, -T ≤ t ≤ T }) ≤ i,j≤r1 x 1 T -T |a0|+|t| √ ((a0-λi) 2 +t 2 )((a0-λj ) 2 +t 2 ) dt.
By the construction of Γ 1 , we have
|a 0 -λ i | ≥ δ k 2 for all 1 ≤ i ≤ n.(9)
Thus, the r.h.s. is at most
r 2 1 x 1 T -T |a0|+|t| t 2 +(δ k /2) 2 dt = r 2 1 x 1 T -T |a0| t 2 +(δ k /2) 2 dt + T 0 2t t 2 +(δ k /2) 2 dt .(10)
By Lemma E.1, we have
T -T |a0| t 2 +(δ k /2) 2 dt ≤ 2π|a0| δ k = 2πa0 δ k (since a 0 ≥ 0).
The second integral is estimated by what follows.
T 0 2t t 2 +(δ k /2) 2 dt = T 2 +(δ k /2) 2 (δ k /2) 2 1 u du (u = t 2 + (δ k /2) 2 ) = log T 2 +(δ k /2) 2 (δ k /2) 2 = log 4T 2 +δ 2 k δ 2 k ≤ 2 log 3T δ k .(11)
Therefore,
Γ1 1≤i,j≤r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| ≤ r 2 1 x 1 2πa0 δ k + 2 log 3T δ k . (12
)
Next, we bound the second term as follows
Γ1 n≥i,j>r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| = Γ1 z n≥i>r1 uiu ⊤ i z-λi E n≥i>r1 uiu ⊤ i z-λi |dz| ≤ Γ1 |z| • n≥i>r uiu ⊤ i z-λi × ∥E∥ × n≥i>r uiu ⊤ i z-λi |dz| ≤ ∥E∥ Γ1 |z| min n≥i>r 1 |z-λi| 2 |dz| = ∥E∥ T -T √ a 2 0 +t 2 min n≥i>r 1 [(a0-λi) 2 +t 2 ] dt.
Moreover, by the construction of Γ 1 and the definition of r 1 ,
|a 0 -λ i | = λ k +λ k+1 2 -λ i ≥ λ k +λ k+1 -2λr+1 2 ≥ λ k -λr+1 2 ≥ λ k 4 ,(13)
where the second inequality follows the fact i > r 1 . Thus, the r.h.s. is at most
∥E∥ T -T √ a 2 0 +t 2 t 2 +(λ k /4) 2 dt ≤ ∥E∥ T -T a0+|t| t 2 +(λ k /4) 2 dt.
Similar to ( 10) and (11), we also have
T -T a0+|t| t 2 +(λ k /4) 2 dt ≤ 4πa0 λ k + log T 2 +(λ k /4) 2 (λ k /4) 2 ≤ 4πa0 λ k + log 2T 2 δ 2 k ≤ 4π + 2 log 2T δ k (since a 0 ≤ λ k ).(14)
It follows that
Γ1 n≥i,j>r1 z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| ≤ ∥E∥ 4π + 2 log 2T δ k .(15)
Now we consider the last term:
Γ1 i≤r1<j i>r1≥j z (z-λi)(z-λj ) u i u ⊤ i Eu j u ⊤ j |dz| ≤ 2∥E∥ Γ1 |z| min i≤r 1 <j |(z-λi)(z-λj )| |dz|.
By ( 13) and ( 9), the r.h.s. is at most
2∥E∥ T -T |z| √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt = 4∥E∥ T 0 √ a 2 0 +t 2 √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt ≤ 4∥E∥ T 0 a0+t √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt.(16)
Moreover,
T 0 a0+t √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt equals T 0 a0dt √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) + T 0 tdt √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt = T 0 a0dt √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) + 1 2 log (T 2 +(δ k /2) 2 )(a0-λr+1) 2 (T 2 +(a0-λr+1) 2 )δ k /2 ≤ a0 max{δ k /2,a0-λr+1} × log T + √ T 2 +min{δ k /2,a0-λr+1} 2 min{δ k /2,a0-λr+1} + 1 2 log (T 2 +(δ k /2) 2 )(a0-λr+1) (T 2 +(a0-λr+1) 2 )δ k /2 . Note that a 0 -λ r+1 ≥ δ k /2 and a 0 -λ r+1 + δ k /2 = λ k -λ r+1 ≥ λ k 2 . Therefore, a 0 -λ r+1 = max{δ k /2, a 0 -λ r+1 } ≥ λ k 4 .
We further obtain that
T 0 a0+t √ (t 2 +(δ k /2) 2 )(t 2 +(a0-λr+1) 2 ) dt is at most a0 λ k /4 • log 5T δ k + 1 2 log 2T δ k ≤ 4.5 log 5T δ k .(17)
The estimates ( 16) and ( 17) together imply that the last term is at most
18∥E∥ 5T δ k .(18)
Combining ( 12), ( 15) and ( 18), we finally obtain that M 1 is at most
r 2 1 x 1 2πa0 δ k + 2 log 3T δ k + ∥E∥ 4π + 2 log 2T δ k + 18∥E∥ log 5T δ k ≤ r 2 1 x 1 2πa0 δ k + 2 log 6σ1 δ k + (20 + 4π/ log(10)∥E∥ log 10σ1 δ k (since log 10σ1 δ k ≥ log 10) ≤ r 2 x 2πa0 δ k + 2 log 6σ1 δ k + (20 + 4π/ log(10)∥E∥ log 10σ1 δ k .(19)
This proves Lemma D.6. Next, we estimate N 3 . Notice that N 3 = Γ3 |z| mini |z-λi| 2 |dz| = T -T √ a 2 1 +t 2 min i∈[n] [(a1-λi) 2 +t 2 ] dt (since Γ 3 := {z | z = a 1 + it, -T ≤ t ≤ T }) ≤ T -T √ a 2 1 +t 2 t 2 +(a1-λ1) 2 dt ≤ T -T |a1| t 2 +(a1-λ1) 2 dt + T -T |t| t 2 +(a1-λ1) 2 dt ≤ πa1 a1-λ1 + 2 log T a1-λ1 2 + 1 (by Lemma E.1) ≤ πa1 a1-λ1 + 4 log 3T a1-λ1 .
This proves Lemma D.3.
this section cite: []

Section: E.2 Estimating integrals over horizontal segments
We are going to bound N 2 , N 4 -integral over top horizontal segment (prove Lemma D.4). We have
N 2 = Γ2 |z| min i∈[n] |z-λi| 2 |dz| = a1 a0 √ x 2 +T 2 min i∈[n] ((x-λi) 2 +T 2 ) dx (since Γ 2 := {z | z = x + iT, a 0 ≤ x ≤ a 1 }) ≤ a1 a0 √ 2T T 2 dx (since x ≤ a 1 ≤ T ) = √ 2|a1-a0| T .
By similar arguments, we also obtain
N 4 ≤ √ 2|a1-a0| T .
These estimates on N 2 , N 4 prove Lemma D.4.
this section cite: []

Section: E.3 Estimating integrals over vertical segments for non-interaction bound
In this Section, we estimate N 1 , proving Lemma D.2. The estimation of N 3 follows the case of the interaction-dependent bound at the end of Section E.1.
N 1 = Γ1 |z| mini |z-λi| 2 |dz| = T -T √ a 2 0 +t 2 min i∈[n] [(a0-λi) 2 +t 2 ] dt (since Γ 1 := {z | z = a 0 + it, -T ≤ t ≤ T }) ≤ T -T √ a 2 0 +t 2 t 2 +(δ k /2) 2 dt (by (9)) ≤ T -T |a0| t 2 +(δ k /2) 2 dt + T -T |t| t 2 +(δ k /2) 2 dt ≤ 2πa0 δ k + 2 log 2T δ k 2 + 1 (by Lemma E.1) ≤ 2πa0 δ k + 4 log 3T δ k .
This proves Lemma D.2.
this section cite: []

Section: F Perturbation of matrix functionals -Theorem 2.3
In this section, we complete the delayed proof of Theorem 2.3. By Remark 3.2, to prove Theorem 2.3, we need to show that
2πF 1 (1) := Γ ∥(zI -A) -1 E(zI -A) -1 ∥|dz| = 4π ∥E∥ δ p ,
in which the contour Γ is set to be a rectangle with vertices (x 0 , T ), (x 1 , T ), (x 1 , -T ), (x 0 , -T ), where x 0 := λ p -δ p /2, x 1 := 2λ 1 , T := 2λ 1 .
We split Γ into four segments:
Γ 1 := {(x 0 , t)| -T ≤ t ≤ T }; Γ 2 := {(x, T )|x 0 ≤ x ≤ x 1 }; Γ 3 := {(x 1 , t)|T ≥ t ≥ -T }; Γ 4 := {(x, -T )|x 1 ≥ x ≥ x 0 }. Therefore, Γ ∥(zI -A) -1 E(zI -A) -1 ∥|dz| = 4 i=1 M k ,(20)
in which
M i := Γi ∥(zI -A) -1 E(zI -A) -1 ∥|dz| for i ∈ {1, 2, 3, 4}.
By a similar strategy from previous section, we bound M 1 as follows. Notice that
(z -A) -1 E(z -A) -1 ≤ ∥E∥ min i∈[n] |z-λi| 2 . Therefore, M 1 is at most ∥E∥ • Γ1 1 min i∈[n] |z -λ i | 2 |dz| ≤ ∥E∥ • Γ1 1 |z -λ p | 2 |dz| (since λ p is closest to Γ 1 among all eigenvalues of A) = ∥E∥ • T -T 1 (δ p /2) 2 + t 2 dt (by definition Γ 1 := {(x 0 , t)| -T ≤ t ≤ T } and |x 0 -λ p | = δ p /2) ≤
2π∥E∥ δ p (by Lemma E.1).
Next, we bound M 3 as what follows.
M 3 ≤ ∥E∥ Γ3 1 min i∈[n] |z-λi| 2 |dz| = ∥E∥ T -T 1 min i∈[n] ((x1-λi) 2 +t 2 ) dt ( since Γ 3 := {z | z = x 1 + it, -T ≤ t ≤ T ) = ∥E∥ T -T 1 t 2 +(x1-λ1) 2 dt ≤ π∥E∥ |x1-λ1| (by Lemma E.1) = π∥E∥ λ 1 (since x 1 = 2λ 1 ).
Next, we estimate M 2 as
M 2 ≤ Γ2 1 min i∈[n] |z-λi| 2 ∥E∥|dz| = ∥E∥ Γ2 1 min i∈[n] |z-λi| 2 |dz|. Moreover, since Γ 2 := {z | z = x + iT, x 0 ≤ x ≤ x 1 }, Γ2 1 min i∈[n] |z-λi| 2 |dz| = x1 x0 1 min i∈[n] ((x-λi) 2 +T 2 ) dx ≤ x1 x01
T 2 dx = |x1-x0| T 2 . Therefore, M 2 ≤ ∥E∥•|x1-x0| T 2 ≤ ∥E∥λ1 4λ 2 1 = ∥E∥ 4λ1 .
Similarly, we also obtain that M 4 = ∥E∥ 4λ1 . These estimates on M 1 , M 2 , M 3 , M 4 and Equation 20 imply
Γ ∥(zI -A) -1 E(zI -A) -1 ∥|dz| = 2π∥E∥ δ p + (π + 1 4 + 1 4 ) ∥E∥ λ 1 ≤ 4π ∥E∥ δ p .
The last inequality follows the trivial fact that λ 1 > δ p for any PSD matrix A. We complete the proof.
this section cite: []

Section: G Some classical perturbation bounds
This section recalls standard results referenced in Section 2, Section 3, and Section A. Theorem G.1 (Eckart-Young-Mirsky bound [16]). Let A, Ã ∈ R n×n , and let A p , Ãp denote their respective best rank-p approximations. Set E := Ã -A. Then,
∥ Ãp -A p ∥ ≤ 2 (σ p+1 + ∥E∥) ,
where σ p+1 is the (p + 1)st singular value of A.
Theorem G.2 (Weyl's inequality [46]). Let A, E ∈ R n×n be symmetric, and define Ã := A + E. Then, for any 1 ≤ i ≤ n,
| λi -λ i | ≤ ∥E∥ and |σ i -σ i | ≤ ∥E∥,
where λ i , λi are the ith eigenvalues of A and Ã, and σ i , σi are the corresponding singular values.
this section cite: ['b15', 'b45']

Section: H Notation
This section summarizes the key notations used throughout the paper. Let A, E be symmetric n × n matrices, and define the perturbed matrix Ã := A + E. Let f be an entire function, and let s ∈ N.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: Justification: All main theorems (e.g., Theorem 2.1) include clear assumptions, and full proofs are provided in Sections 3, Appendix D, Appendix E, and Appendix F.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: Section 4 details the matrices used, noise models, parameter settings, evaluation metrics, and empirical setup to enable reproducibility.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction state that the paper provides new highprobability spectral norm bounds for low-rank approximation under symmetric perturbations, resolving key limitations of classical worst-case bounds and prior DP utility analyses, and the body of the paper rigorously proves and empirically validates this claim (Sections 2-4).
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Section 5 (Conclusion, Limitations, and Future Work) discusses the reliance on spectral quantities, the limitations of our results beyond the gap threshold, and the open questions of extending the framework to structured perturbations, including data matrices with specific spectral patterns or correlated noise.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The data (e.g., Census and 1998 KDD-Cup ) are publicly available and cited appropriately. Code and instructions are provided in the supplemental material. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: Section 4 and Section B describe matrix dimensions, truncation ranks, noise scales, trial counts, and the methods used to compute bounds. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: Section 4 and Section B report error bars across 100 trials as mean ± standard deviation, with clear plots and captions. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: The experiments are lightweight and run on standard CPU machines; resource requirements are described in the supplemental material. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The research is theoretical and empirical, uses only publicly available datasets, and conforms to ethical standards. Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: This is a theoretical paper on spectral norm perturbation bounds with no direct societal or ethical impact pathways.
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
Answer: [NA] Justification: The paper does not release models or datasets with any risk of misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All datasets (e.g., Census, 1998 KDD-Cup, Adult) are properly cited (e.g., [29], [11], [3]) and are in the public domain or released under open academic licenses.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not introduce new datasets, models, or other assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve any human subjects or crowdsourcing. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The research does not involve human subjects and thus does not require IRB approval. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The research does not use LLMs for any component of the core methodology. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b28', 'b10', 'b2']

Section: References
Ref_id:b0 Title: Fast computation of low-rank matrix approximations Year: (2007)
Ref_id:b1 Title: Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays Year: (1999)
Ref_id:b2 Title: Differentially private covariance estimation Year: (2019)
Ref_id:b3 Title: Spectral analysis of data Year: (2001)
Ref_id:b4 Title: Spectral analysis of large dimensional random matrices Year: (2009)
Ref_id:b5 Title: The Netflix Prize Year: (2007)
Ref_id:b6 Title: Matrix analysis Year: (2013)
Ref_id:b7 Title: The Johnson-Lindenstrauss transform itself preserves differential privacy Year: (2012)
Ref_id:b8 Title: Practical privacy: the sulq framework Year: (2005)
Ref_id:b9 Title: Foundations of data science Year: (2020)
Ref_id:b10 Title: Near-optimal differentially private principal components Year: (2012)
Ref_id:b11 Title: The rotation of eigenvectors by a perturbation Year: (1970)
Ref_id:b12 Title: UCI machine learning repository Year: (2017)
Ref_id:b13 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b14 Title: Analyze Gauss: Optimal bounds for privacy-preserving principal component analysis Year: (2014)
Ref_id:b15 Title: The approximation of one matrix by another of lower rank Year: (1936)
Ref_id:b16 Title: Robust subspace iteration and privacy-preserving spectral analysis Year: (2013)
Ref_id:b17 Title: The noisy power method: A meta algorithm with applications Year: (2014)
Ref_id:b18 Title: Functions of Matrices: Theory and Computation Year: (2008)
Ref_id:b19 Title: Matrix Analysis Year: (2012)
Ref_id:b20 Title: Perturbation-based methods for explaining deep neural networks: A survey Year: (2021)
Ref_id:b21 Title: Perturbation bounds for eigenspaces under a relative gap condition Year: (2020)
Ref_id:b22 Title: The spectral method for general mixture models Year: (2008)
Ref_id:b23 Title: Spectral algorithms Year: (2009)
Ref_id:b24 Title: On differentially private low rank approximation Year: (2013)
Ref_id:b25 Title: Perturbation Theory for Linear Operators Year: (1980)
Ref_id:b26 Title: Concentration inequalities and moment bounds for sample covariance operators Year: (2017)
Ref_id:b27 Title: Perturbation of linear forms of singular vectors under Gaussian noise Year: (2016)
Ref_id:b28 Title: Re-analyze Gauss: Bounds for private matrix approximation via Dyson Brownian motion Year: (2022)
Ref_id:b29 Title: Private low-rank approximation for covariance matrices, Dyson Brownian Motion, and eigenvalue-gap bounds for Gaussian perturbations Year: (2025-03)
Ref_id:b30 Title: Private matrix approximation and geometry of unitary orbits Year: (2022)
Ref_id:b31 Title: Random perturbation of low rank matrices: Improving classical bounds Year: (2018)
Ref_id:b32 Title: Matrices with Gaussian noise: Optimal estimates for singular subspace perturbation Year: (2023)
Ref_id:b33 Title: Old techniques in differentially private linear regression Year: (2019)
Ref_id:b34 Title: Matrix Perturbation Theory Year: (1990)
Ref_id:b35 Title: Perturbation bounds for low-rank inverse approximations under noise Year: (2025)
Ref_id:b36 Title: Davis-Kahan theorem under a moderate gap condition Year: (2025)
Ref_id:b37 Title: New matrix perturbation bounds with relative norm: Perturbation of eigenspaces Year: (2026)
Ref_id:b38 Title: The price of privacy for low-rank factorization Year: (2018)
Ref_id:b39 Title: On the spectral norm of Gaussian random matrices Year: (2017)
Ref_id:b40 Title: On the accuracy of total least squares and least squares techniques in the presence of errors on all data Year: (1989)
Ref_id:b41 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b42 Title: Spectral norm of random matrices Year: (2007)
Ref_id:b43 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b44 Title: High-Dimensional Statistics: A Non-Asymptotic view point Year: (2019)
Ref_id:b45 Title: Das asymptotische verteilungsgesetz der eigenwerte linearer partieller differentialgleichungen Year: (1912)
