Title: INFONCE INDUCES GAUSSIAN DISTRIBUTION
Abstract: Contrastive learning has become a cornerstone of modern representation learning, allowing training with massive unlabeled data for both task-specific and general (foundation) models. A prototypical loss in contrastive training is InfoNCE and its variants. In this work, we show that the InfoNCE objective induces Gaussian structure in representations that emerge from contrastive training. We establish this result in two complementary regimes. First, we show that under certain alignment and concentration assumptions, projections of the high-dimensional representation asymptotically approach a multivariate Gaussian distribution. Next, under less strict assumptions, we show that adding a small asymptotically vanishing regularization term that promotes low feature norm and high feature entropy leads to similar asymptotic results. We support our analysis with experiments on synthetic and CIFAR-10 datasets across multiple encoder architectures and sizes, demonstrating consistent Gaussian behavior. This perspective provides a principled explanation for commonly observed Gaussianity in contrastive representations. The resulting Gaussian model enables principled analytical treatment of learned representations and is expected to support a wide range of applications in contrastive learning.

Section: INTRODUCTION
Self-supervised learning with contrastive objectives has transformed modern representation learning, enabling scalable training of encoders without labels (Oord et al., 2018;Chen et al., 2020;He et al., 2020;Radford et al., 2021). Among these objectives, the InfoNCE loss balances two pressures: positive pairs are aligned while the batch is repelled to encourage uniformity (Wang & Isola, 2020). This uniformity is often described geometrically as "spreading out" the data on the hypersphere (Chen & He, 2021), but a deeper probabilistic question remains: What is the actual distribution of representations trained with InfoNCE?
Answering this question is not only of theoretical interest. A Gaussian characterization is directly motivated by recent empirical findings suggesting that "more Gaussian" representations can correlate with improved downstream performance (Eftekhari & Papyan, 2025). It also provides a principled basis for practical methods that model contrastive representations as Gaussians for tasks such as classification, uncertainty estimation and test-time adaptation (Baumann et al., 2024;Morales-Álvarez et al., 2024). Moreover, assuming Gaussian structure makes quantities such as entropy, likelihood and KL divergences available in closed form, which underpins density-based diagnostics (Lee et al., 2018;Betser et al., 2025). These benefits are already exploited in applied work, with recent studies empirically observing and leveraging approximate Gaussian behavior in self-supervised representations (Baumann et al., 2024;Balestriero et al., 2025;Betser et al., 2026). Yet, despite these developments, a principled population-level explanation of why contrastive objectives such as InfoNCE give rise to Gaussian structure in representation space remains lacking.
Analyzing the population InfoNCE objective, we formalize the emergence of asymptotically Gaussian representations through two complementary analytical routes. A key ingredient is a novel alignment bound based on Hirschfeld-Gebelein-Rényi (HGR) maximal correlation, which limits achievable alignment according to augmentation mildness (Sec. 3.1). In the empirical idealization route, motivated by empirical training dynamics, alignment reaches a plateau and the objective reduces to a constrained uniformity problem on the hypersphere; combined with norm concentration, this yields Gaussian structure for both normalized (to a unit norm) and unnormalized representations (Sec. 4.1). In the regularized route, a population-level analysis shows that adding a vanishing convex regularizer prioritizes the isotropic solution, yielding the same asymptotic Gaussian behavior without relying on training dynamics (Sec. 4.2). Together, these analyses shed light on why Gaussian structure can emerge under the InfoNCE objective at the population level.
We complement our theoretical analysis with empirical studies on synthetic data and CIFAR-10 ( Krizhevsky et al., 2009) images, using encoders of increasing complexity: linear layers, MLPs with nonlinear activations, and ResNet-18 (He et al., 2016). By comparing contrastive and supervised training, we isolate the role of the training objective, beyond effects of data or architecture. We further observe similar Gaussian statistics in representations learned by general self-supervised foundation models, including DINO (Caron et al., 2021), motivating a broader examination of Gaussian structure across self-supervised objectives. Our main contributions are:
• Bounded alignment. In the large-batch limit, the alignment induced by the InfoNCE objective is bounded by the strength of the data augmentations.
• Uniformity on the sphere. Along both routes we analyze, normalized representations converge toward the uniform distribution on the unit sphere.
• Asymptotic Gaussian structure. Within this framework, both normalized and unnormalized representations admit asymptotically Gaussian behavior under the InfoNCE objective.
• Empirical support. Accompanying our asymptotic analysis, we provide finite-dimensional empirical evidence on synthetic and real data, illustrating the emergence of Gaussian behavior across multiple settings and encoder architectures.
this section cite: ['b43', 'b13', 'b28', 'b46', 'b55', 'b14', 'b22', 'b8', 'b42', 'b36', 'b9', 'b8', 'b10', 'b35', 'b27', 'b12']

Section: RELATED WORK
Contrastive learning and InfoNCE. The InfoNCE loss (Oord et al., 2018) is the standard objective in self-supervised representation learning and underlies methods such as SimCLR (Chen et al., 2020), MoCo (He et al., 2020), and CLIP (Radford et al., 2021). It balances alignment of positive pairs with batch-wise repulsion that promotes uniformity in representation space (Wang & Isola, 2020;Chen & He, 2021). Prior work has studied these effects from geometric and optimization perspectives, identifying phenomena such as hyperspherical uniformity and feature concentration (Chen & He, 2021;Caron et al., 2021;Draganov et al., 2025). Other empirical studies model contrastive representations as approximately Gaussian (Baumann et al., 2024;Morales-Álvarez et al., 2024). However, the probabilistic law induced by the InfoNCE objective itself remains theoretically unexplained.
Isotropy and Gaussian structure. Several works aim to promote isotropic or Gaussian-like representations through explicit regularization or architectural design, including whitening-based objectives, variance-covariance control, and neural collapse phenomena (Ermolov et al., 2021;Papyan et al., 2020;Bardes et al., 2022). Related self-supervised approaches based on joint-embedding predictive architectures (JEPA) also yield highly regular representations and have been shown to encode density-related structure that can be exploited with Gaussian models (Assran et al., 2023;Bardes et al., 2024;Balestriero et al., 2025;Balestriero & LeCun, 2025). However, these works primarily observe or exploit Gaussian-like structure rather than explain its origin. Our work instead shows how Gaussianity emerges directly from the population InfoNCE objective.
this section cite: ['b43', 'b13', 'b28', 'b46', 'b55', 'b14', 'b14', 'b12', 'b20', 'b8', 'b42', 'b23', 'b44', 'b6', 'b2', 'b7']

Section: Hyperspherical geometry and Gaussianity.
A classical body of work studies the geometry of high-dimensional uniform measures on the sphere and their connection to Gaussian distributions (Vershynin, 2018;Wegner, 2021). Related geometric ideas also appear in hyperspherical variational families and radial Bayesian priors, which leverage approximately uniform distributions over the hypersphere (Davidson et al., 2018;Farquhar et al., 2020). A central result in this literature is the Maxwell-Poincaré spherical central limit theorem, which shows that fixed-dimensional projections of the uniform distribution on S d-1 converge to a Gaussian as the dimension grows (Maxwell, 1860;Poincaré, 1912;Diaconis & Freedman, 1987). Although developed independently of contrastive learning, these results provide the mathematical basis for why spherical uniformity induces Gaussian structure in high-dimensional representations. Our analysis connects this classical theory to contrastive learning by identifying regimes in which the InfoNCE objective induces such uniformity.
this section cite: ['b54', 'b56', 'b16', 'b24', 'b41', 'b45', 'b19']

Section: Additional theoretical perspectives.
Complementary lines of work study theoretical properties of representations learned with contrastive objectives. Identifiability analyses characterize when latent variables or semantic factors can be uniquely recovered under structural assumptions on the data-generating process (Hyvarinen & Morioka, 2016;Hyvarinen et al., 2019;Zimmermann et al., 2021;Roeder et al., 2021;Reizinger et al., 2024); these results concern conditional or componentlevel structure and do not make claims about the marginal distribution of representations. Separately, task-driven analyses establish class separability or clustering guarantees for contrastive representations (Saunshi et al., 2019;HaoChen et al., 2021), focusing on class-conditional geometry rather than the overall distribution. Concretely, class-specific clusters may remain well separated even when the overall embedding distribution is approximately Gaussian. Our work does not address recovery or class structure; instead, it analyzes the marginal distribution induced by the population InfoNCE objective.
this section cite: ['b32', 'b33', 'b59', 'b49', 'b47', 'b50', 'b26']

Section: SETUP

this section cite: []

Section: Data domain.
Let (X , B(X )) be a standard Borel space (a standard setting in probability) with a base probability p base . We draw X 0 ∼ p base as a single data item (e.g., an image).
this section cite: []

Section: Pairs via augmentation.
Contrastive learning is built around pairs of related examples rather than individual samples. To form such pairs, we use an augmentation channel A, which takes a base sample X 0 ∼ p base and produces stochastic variations of it. Formally, given X 0 , we draw two independent augmentations
X, Y ∼ A(• | X 0 ).
(1) Here X and Y are two views of the same underlying example (e.g., different crops or color jitter). We denote by p X the marginal distribution of a single augmentation and assume it is nonatomic (a mild technical condition achievable in practice by infinitesimal dither). p XY denotes the joint distribution of a pair of augmentations (X, Y ).
InfoNCE loss. Let f : X → R d , d ≥ 2, be a Borel-measurable encoder that maps input data to representations. InfoNCE operates on ℓ 2 -normalized representations, defined as f (x) := f (x)/∥f (x)∥ if ∥f (x)∥ > 0, and f (x) := c 0 for a fixed arbitrary c 0 ∈ S d-1 otherwise. Given a batch of N paired augmentations {(x i , y i )} N i=1 drawn i.i.d. from p XY , define u i := f (x i ) and v i := f (y i ). The empirical InfoNCE loss is
L InfoNCE = - 1 N N i=1 log exp 1 τ ⟨u i , v i ⟩ N j=1 exp 1 τ ⟨u i , v j ⟩ ,(2)
with a fixed temperature τ > 0. Since u i and v j are unit-normalized, ⟨u i , v j ⟩ equals cosine similarity.
The numerator measures the similarity of the positive pair (u i , v i ). The denominator compares each anchor u i to all candidates {v j } N j=1 , where j ̸ = i serve as negatives. This softmax encourages u i to rank its true partner highest while remaining distinct from negatives, preventing collapse.
this section cite: []

Section: Population InfoNCE.
The empirical InfoNCE loss in Eq. ( 2) depends on the batch size N . As N → ∞, the empirical averages converge to expectations. Let
µ := f * p X , π := ( f , f ) * p XY ,(3)
be the marginal distribution of representations and the joint distribution of positive pairs, respectively.
Here f * p X denotes the pushforward measure of p X by f , which is the distribution of f (X). As shown by Wang & Isola (2020, Theorem 1, Eq. ( 2)), in the infinite-negatives limit N → ∞ the empirical InfoNCE loss (up to the additive log N term) converges to the following population functional. With α = 1/τ for fixed τ > 0:
L(µ, π) = -α E (u,v)∼π [u•v] + Φ(µ), Φ(µ) := E u∼µ log E v∼µ exp α u•v .(4)
The first term measures alignment of positive pairs, while the second is a uniformity potential depending only on µ.
this section cite: []

Section: ALIGNMENT BOUND
We now introduce a new term that quantifies the degree of augmentation. The augmentation channel A limits how much positive-pair alignment can be induced. We quantify this with the augmentation mildness parameter
η 2 := sup g∈L 2 (p X ) Var(g)>0 Var E[g(X) | X 0 ] Var(g(X)) ∈ [0, 1],(5)
which measures how predictable functions of the view X are from the base X 0 . This quantity equals the squared Hirschfeld-Gebelein-Rényi (HGR) maximal correlation, denoted ρ m (X, X 0 ), i.e., , 1935;Gebelein, 1941;Rényi, 1959) (see Appendix A.1). Intuitively, η 2 = 0 when X is (effectively) independent of X 0 (very strong/noisy augmentations), and η 2 = 1 when X is fully determined by X 0 (no augmentation noise).
η 2 = ρ 2 m (X, X 0 ) (Hirschfeld
Example. Consider the Gaussian channel X = AX 0 + √ 1 -A 2 ε, where X 0 ∼ N (0, 1) and ε ∼ N (0, 1) are independent. In this case, X and X 0 are jointly Gaussian with Pearson correlation A, the maximal correlation satisfies ρ m (X, X 0 ) = |A|, and thus η 2 = A 2 (Appendix A.2). Proposition 1 (Augmentation-controlled alignment bound). Let X, Y ∼ A(• | X 0 ) be conditionally independent given the base sample X 0 , and let u = f (X), v = f (Y ) be normalized representations in S d-1 , i.e., ∥u∥ = ∥v∥ = 1. Then
E (u,v)∼π [u • v] ≤ η 2 + (1 -η 2 ) ∥m(µ)∥ 2 , m(µ) := E[u] = E[v],(6)
where η 2 = ρ 2 m (X, X 0 ) is the squared HGR maximal correlation between the view and the base, and µ is the marginal law of u.
The proof appears in Appendix A.3. This bound links the alignment of positive pairs to the structure of the statistical dependence induced by the augmentation channel. While HGR maximal correlation has been studied in statistical dependence analysis (Huang & Xu, 2020;Zhang et al., 2024), it has not previously been used to control alignment in contrastive learning. Existing work studies augmentations empirically (e.g., Tian et al. (2020)) but does not derive bounds of this form. This result formalizes how the strength of data augmentations fundamentally constrains achievable alignment under the InfoNCE objective.
this section cite: ['b29', 'b25', 'b48', 'b31', 'b58', 'b52']

Section: GAUSSIANITY FROM INFONCE
We study why minimizing the population InfoNCE objective (Eq. 4) yields (approximately) Gaussian low-dimensional projections of learned representations, for both normalized representations on the sphere and unnormalized representations in R d . Our analysis proceeds along two complementary routes, which differ in the strength of the assumptions they require.
this section cite: []

Section: Empirical idealization.
We first analyze an idealized regime with infinite data, ambient dimension d → ∞, and sufficient optimization. Guided by empirical observations, we assume alignment plateau and thin-shell concentration; these assumptions enable a simple derivation of Gaussian projections.
this section cite: []

Section: Regularized route.
To reduce reliance on training dynamics, we study a regularized variant of the population objective. Introducing a vanishing convex regularizer and assuming attainable alignment at uniformity ensures a unique minimizer and yields the same asymptotic Gaussian structure. This route provides an alternative explanation independent of training behavior.
this section cite: []

Section: GAUSSIAN PROJECTIONS AT ALIGNMENT PLATEAU
Proposition 1 provides an upper bound on achievable alignment. In the sequel we do not assume this bound is tight; instead, we model training as reaching a plateau that lies strictly below the bound.
Assumption 1 (Alignment plateau). After sufficient training, the positive-pair alignment saturates at a ceiling; concretely,
E (u,v)∼π [u•v] = η 2 + r plat ,(7)
where r plat ≤ 0 is a constant error term representing the difference between the alignment value at plateau and the maximal correlation defined by the augmentations (η 2 ).
Empirically, alignment saturation has been reported in some contrastive-learning settings (Wang & Isola, 2020), which motivates considering a plateau model as a plausible scenario rather than a universal requirement. In our experiments (Fig. 2, Appendix Figs. 7, 8), we frequently observe high alignment alongside improving uniformity with larger dimensions and batch sizes, suggesting that alignment may saturate before uniformity in at least some regimes. An extension that places the plateau exactly at the alignment bound (Eq. 6) is discussed in Appendix D.
Corollary 1 (Gaussian k-projections at the plateau). Suppose the alignment plateau condition (Eq. 7)
holds, and consider the population objective (Eq. 4). Let µ * denote the global minimizer supported on S d-1 . Then, as
d → ∞, for every fixed k ≥ 1 the k-dimensional marginal of u ∼ µ * satisfies √ d u k ⇒ N (0, I k ),(8)
where u k denotes the projection of u onto a fixed k-dimensional coordinate subspace and I k is the k × k identity matrix.
The proof is provided in Appendix C.1 and follows from two lemmas. The first establishes that Φ(µ) attains a global minimum at the uniform law (Wang & Isola, 2020), while the second invokes the central limit theorem on the sphere (Diaconis & Freedman, 1987) to deduce Gaussian projections.
this section cite: ['b55', 'b55', 'b19']

Section: GAUSSIAN PROJECTIONS FOR UNNORMALIZED REPRESENTATIONS.
So far we analyzed normalized representations on the sphere. We now extend the result to the original, unnormalized encoder outputs z = f (X) ∈ R d . Write z = ru, where r = ∥z∥ is the representation radius and u = z/∥z∥ ∈ S d-1 the normalized direction.
Assumption 2 (Thin-shell concentration). We assume the representation radius concentrates:
r r 0 ---→ d→∞ 1,(9)
where r 0 ∈ (0, ∞) is a deterministic constant.
Norm concentration is widely observed in contrastive learning: unnormalized representations cluster around a characteristic radius (Wang & Isola, 2020;HaoChen et al., 2021;Levi & Gilboa, 2025). This thin-shell effect (Klartag, 2023) is further promoted by weight decay, which penalizes norm growth and stabilizes a common scale. In particular, Draganov et al. (2025) show that appropriate weight decay suppresses norm inflation and tightens the dispersion of representation norms, lending empirical support to Assumption 2. Consistent with these reports, our experiments exhibit progressively sharper radius histograms as dimension and batch size increase (Figs. 3, 4, 6).
Proposition 2 (Gaussian projections for unnormalized representations). Let z = f (x) ∈ R d be the unnormalized representation and u := z/∥z∥. Assume u ∼ σ (the uniform distribution on S d-1 ) and that Assumption 2 holds, i.e., r ---→ d→∞ r 0 ∈ (0, ∞). Then for any fixed k-dimensional subspace,
√ d z k ⇒ N 0, r 2 0 I k (d → ∞),(10)
where z k denotes the orthogonal projection of z onto that subspace and I k is the k × k identity.
See proof in Appendix C.2.
this section cite: ['b55', 'b26', 'b34', 'b20']

Section: GAUSSIAN PROJECTIONS USING REGULARIZATION
Proposition 1 shows that alignment is limited by the augmentation channel Eq. ( 6). At the uniform distribution (µ = σ) the mean vanishes, m(σ) = 0, and the bound reduces to E[u•v] ≤ η 2 . Assuming this ceiling is attainable at uniformity, the uniform distribution becomes asymptotically optimal for the population objective. We work in a regularized setting, where the regularization vanishes as d → ∞. As before, this has direct implications to the representation projections, which are approximately Gaussian (Theorem 2). This result shows that Gaussianity can be obtained without relying on the stronger thin-shell or plateau conditions.
We constrain f to take values in B ⊆ R d , which is either some closed ball centered at 0 with positive radius or R d . We take the original loss and add two new losses: one to penalize large squared norms, and the other to encourage high entropy (we comment that both are commonly regarded as desirable goals, irrespective of our setup). Specifically, for fixed β, λ > 0,
J(f ) = Φ(µ) -αE (u,v)∼π [u • v] + β(-H(ρ) + λE Z∼ρ ∥Z∥ 2 ) ,(11)
where ρ = f * p X is the unnormalized pushforward probability. Define the truncated Gaussian γ B λ ,
γ B λ (dz) = c B,λ e -λ∥z∥ 2 1 B (z)dz , c -1 B,λ = B e -λ∥z∥ 2 dz .(12)
If ρ ≪ γ B λ (≪ denotes absolute continuity, so ρ is absolutely continuous with respect to γ B λ ), then
KL(ρ∥γ B λ ) = log dρ dz dρ -log dγ B λ dz dρ = -H(ρ) + λ E ρ ∥Z∥ 2 + log c -1 B,λ ,(13)
that is, equality up to an additive constant. Since ρ(B) = 1, if ρ ̸ ≪ γ B λ , then both KL(ρ∥γ B λ ) and -H(ρ) are +∞. Thus, it is equivalent to minimize
J(f ) = Φ(µ) -αE (u,v)∼π [u • v] + βKL(ρ∥γ B λ ) ,(14)
and we thereby also implicitly restrict ρ to satisfy ρ ≪ γ B λ and in particular ρ(B) = 1. Our goal is to prove that for β ≥ β 0 , taking the angular probability as σ approaches optimality and the optimal radial probability is that of γ B λ . If B = R d , this means that a Gaussian ρ approaches optimality. Furthermore, as d → ∞, β 0 → 0.
This will be done in several steps. First, ρ can be decomposed into a radial part and an angular part. We show that the radial part can be chosen optimally in a straightforward way.
Proposition 3. Let ρ(dz) = µ(du)κ(dr | u) and γ B λ (dz) = σ(du)ξ(dr | u) in polar coordinates z = ru. Then κ = ξ is an optimal choice, yielding KL(ρ∥γ B λ ) = KL(µ∥σ).
The proof is given in Appendix B.1. The above proposition reduces the optimization problem for unnormalized embedding to normalized embeddings only. It also describes an optimal probability for embedding norms, in contrast to the original InfoNCE loss, which is completely oblivious to embedding norms.
It is important to note that because we are working with a standard Borel space with a nonatomic p X , any probability ρ ∈ P(B) has ρ = g * p X for some encoding g. In addition, any µ ∈ P(S d-1 ) has µ = h * p X for some encoding, and since B contains a ball around 0, there is an encoding f s.t. h = f . Thus we can legitimately speak about "choosing" ρ or µ, since suitable encodings exist that induce them. In addition, we may also define: Definition 1. For every µ ∈ P(S d-1 ),
Align(µ) = sup f E[ f (X) • f (Y )] : f measurable, ( f ) * p X = µ ,(15)
As was noted, the supremum is always taken on a nonempty set. We can write
J(µ) = Φ(µ) -αAlign(µ) + βKL(µ∥σ) ,(16)
and it holds that inf { f : f * pX =µ} J(f ) = J(µ), and consequently inf f J(f ) = inf µ∈P(S d-1 ) J(µ).
The reason is that Align(µ) can be approximated arbitrarily well by an encoding, and the KL divergence is optimized by taking the radial distribution given in Proposition 3. We can therefore focus on optimizing J(µ).
The assumption for which we will prove our result is the following:
Assumption 3. It holds that α(η 2 -Align(σ)) d→∞ ---→ 0.
We will require one more technical lemma before proceeding to prove the result.
Lemma 1. If d ≥ 2, then KL(µ∥σ) ≥ C(d -1)∥m(µ)∥ 2 , where C > 0 is a universal constant.
Proof is provided in Appendix B.2. To understand the constant, see (Vershynin, 2018, Proposition 2.6.1).
this section cite: ['b54']

Section: Theorem 1. Let d ≥ 2.
There is a universal constant C > 0 s.t. for β ≥ β 0 = α(1-η2) C(d-1) ,
• Under Assumption 3, J(σ) -inf µ J(µ) d→∞ ---→ 0.
• Assuming further that Align(σ) = η 2 yields that J(σ) = min µ J(µ).
Moreover, as d → ∞, β 0 → 0.
Proof. Write δ(d) = η 2 -Align(σ). For every µ, we have that Φ(µ) -Φ(σ) ≥ 0 (Wang & Isola, 2020, Theorem 1). In addition,
Align(µ) -Align(σ) ≤ η 2 + (1 -η 2 )∥m(µ)∥ 2 -(η 2 -δ(d)) = (1 -η 2 )∥m(µ)∥ 2 + δ(d) (17) by Proposition 1. Lastly, KL(µ∥σ) -KL(σ∥σ) = KL(µ∥σ) ≥ C(d -1)∥m(µ)∥ 2 (18) by Lemma 1. Therefore, J(µ) -J(σ) = (Φ(µ) -Φ(σ)) -α(Align(µ) -Align(σ)) + β(KL(µ∥σ) -KL(σ∥σ)) ≥ -α(1 -η 2 )∥m(µ)∥ 2 -αδ(d) + βC(d -1)∥m(µ)∥ 2 = (-α(1 -η 2 ) + βC(d -1)) ∥m(µ)∥ 2 -αδ(d) ≥ -αδ(d) ,(19)
where the last inequality is by the choice of β.
If we assume that αδ(d
) d→∞ ---→ 0, then J(σ) -inf µ J(µ) ≤ αδ(d), so J(σ) -inf µ J(µ) d→∞ ---→ 0.
If we assume further that Align(σ) = η 2 , then δ(d) = 0, and since J(σ) ≤ J(µ) for every µ, J(σ) = min µ J(µ), completing the proof.
Since the optimal radial component of the distribution is known, we can draw conclusions w.r.t. ρ as well. For example, we can directly obtain the following corollary. Corollary 2. Let B = R d (d ≥ 2) and β ≥ β 0 . If Align(σ) = η 2 , where σ is the uniform distribution on S d-1 and η 2 is the augmentation mildness, then N (0, (2λ) -1 I d ) is an optimal choice for ρ.
this section cite: []

Section: EXPERIMENTS
We empirically evaluate the distributional geometry of representations learned with the InfoNCE objective. The experiments are designed to test three theoretical predictions: (i) concentration of representation norms on a thin shell, (ii) emergence of Gaussian low-dimensional projections, and (iii) the dependence of these phenomena on contrastive learning.
We consider three settings of increasing complexity: synthetic data with linear encoders, CIFAR-10 with both contrastive and supervised training, and pretrained foundation-scale models. In all cases, we analyze both normalized and unnormalized representations. All reported trends are stable across runs; figures show representative seeds, with full implementation details in Appendix E.1.
Metrics. We quantify Gaussian structure using complementary diagnostics targeting radial and coordinate-wise behavior. To assess norm concentration, we measure the coefficient of variation (CV) of representation norms:
CV = std {∥z i ∥} N i=1 mean {∥z i ∥} N i=1 .(20)
z i are the learned representations and N is the number of samples. A small CV indicates concentration of ∥z i ∥ around a characteristic radius, consistent with thin-shell behavior.
To evaluate Gaussianity of low-dimensional projections, we apply two standard one-dimensional normality tests to individual coordinates: (i) the Anderson-Darling (AD) test (Anderson & Darling, 1954), where AD < 0.752 corresponds to failure to reject normality, and (ii) the D'Agostino-Pearson (DP) test (D' Agostino & Pearson, 1973), where p > 0.05 indicates failure to reject the Gaussian hypothesis. These tests probe marginal normality of fixed coordinates, as predicted by the spherical central limit theorem.
Taken together, CV captures global radial structure, while AD and DP test coordinate-level Gaussianity. This combination provides a strong finite-sample indicator of approximate Gaussian behavior and provides evidence against common heavy-tailed or mixture alternatives, which typically fail at least one of these diagnostics.
Synthetic data experiments. We begin with controlled synthetic settings to validate our diagnostics and isolate the mechanisms predicted by the theory. We consider three synthetic data distributions:
(i) an i.i.d. Laplace(0, 1) distribution, (ii) a Gaussian mixture with 25 equally weighted components and random means, and (iii) a fully discrete sparse binary distribution (1024-dimensional vectors). Each dataset contains 10k samples, and we train linear encoders using InfoNCE while varying the representation dimension and batch size. In addition, we explicitly track alignment and uniformity as functions of batch size and dimension (Fig. 2) to probe the saturation behavior predicted by our Assumption 1. Figure 3 shows that, for Laplace inputs, representation norms progressively concentrate as both batch size and dimension increase, evidenced by a monotonic decrease in the coefficient of variation (CV). Norm histograms further illustrate the emergence of thin-shell concentration. Normality diagnostics (AD and DP) indicate that individual coordinates fall well within Gaussian acceptance thresholds, with perfect per-coordinate compliance (Table 1).
Across all three synthetic settings, including strongly non-Gaussian mixture inputs, the learned representations exhibit low norm variation and strong coordinate-wise Gaussianity (Table 1), indicating that marginal Gaussian structure emerges independently of the input distribution. The same phenomenon is observed for the fully discrete binary dataset: although representations are initially far from Gaussian, training drives pronounced norm concentration and coordinate-wise normality. Since this distribution admits no invertible mapping to a continuous Gaussian, the observed structure cannot be explained by latent Gaussian recovery.
In parallel, alignment quickly approaches a stable ceiling determined by the augmentation channel (Fig. 2), while uniformity continues to improve. This behavior is consistent with the saturation route and the emergence of isotropic Gaussian structure in high dimension. Together, these controlled experiments support the assumptions underlying our theoretical analysis and motivate the study of Gaussianity in more realistic settings.
this section cite: ['b1']

Section: CIFAR-10 experiments.
We next study whether Gaussian structure emerges in a realistic vision setting. We train a two-layer MLP with a single ReLU nonlinear activation using the InfoNCE objective on CIFAR-10, and evaluate representations on the test set throughout training.
Figure 4 shows consistent trends across training: representation norms concentrate over time, as indicated by a steadily decreasing CV; the AD statistic drops from non-Gaussian levels into the normal regime; and the fraction of coordinates passing the DP test increases monotonically. These dynamics illustrate the joint emergence of thin-shell concentration and coordinate-wise Gaussianity as optimization progresses. These trends mirror the synthetic setting and show that norm concentration and Gaussianity also emerge in realistic contrastive training.  Pretrained models. We further examine whether Gaussian structure persists in large pretrained representations. On the MS-COCO validation set (Lin et al., 2014), we compare self-supervised backbones CLIP (ViT-L/14 image and text encoders) (Radford et al., 2021) and DINO (ViT-B/32) (Caron et al., 2021) against supervised ImageNet-pretrained (Deng et al., 2009), ResNet34 (He et al., 2016), and DenseNet (Huang et al., 2017). Normality diagnostics (
Table 2) show that self-supervised models exhibit near-Gaussian coordinate distributions, while supervised models deviate substantially. We further evaluate the CLIP image encoder on ImageNet-R (Sketch and Painting domains) to test robustness beyond natural images, and again observe strong Gaussian signatures. Although CLIP and DINO are not exact instances of the unimodal InfoNCE setting, these diagnostics suggest that similar isotropic Gaussian-like statistics may arise broadly in self-supervised objectives.
this section cite: ['b39', 'b46', 'b12', 'b17', 'b27', 'b30']

Section: DISCUSSION AND CONCLUSION
We showed that InfoNCE trained representations admit an asymptotic Gaussian law, via two routes: an alignment-plateau analysis with thin-shell concentration, and a regularized surrogate with milder assumptions. Experiments on synthetic data, CIFAR-10, and pretrained models (MS-COCO and ImageNet-R) are consistent with these assumptions and the Gaussian hypothesis, revealing norm concentration, alignment saturation, and near-Gaussian projections. These results indicate that the Gaussian convergence remains informative well before the infinite-dimensional limit. This Gaussian view justifies common modeling choices (e.g., likelihood scoring, OOD detection) and suggests that explicit isotropy promoting regularizers may act as principled surrogates for InfoNCE's implicit bias. However, limitations remain: our results are asymptotic, relying on high-dimensional limits and idealized assumptions that may not capture all practical regimes. We therefore view our asymptotic framework as a principled starting point rather than a complete description of all practical regimes. For finite dimension d and batch size N , projections are close to Gaussian, with deviations vanishing as d, N → ∞. Quantitative bounds follow from classical Berry-Esseen (Vershynin, 2018) rates in high dimension and uniform laws of large numbers for empirical objectives (Wellner et al., 2013). In particular, the minimizer of the empirical InfoNCE loss deviates from the population minimizer by O(N -1/2 ) according to Wang & Isola (2020, Thm. 1), and the distribution of fixed-k projections deviates from Gaussian by O(d -1 ) according to Diaconis & Freedman (1987) (see Theorem 2 in Appendix C.1). Thus, for large but finite d, N , the Gaussian limit provides a representative and empirically useful approximation. In addition, we do not analyze optimization dynamics or prove that training attains these minimizers in practice; our results are asymptotic and characterize the population optima under the stated assumptions. Overall, we provide a principled asymptotic explanation for Gaussianity in contrastive representations, grounding empirical observations and opening new directions for analysis and practical design.
this section cite: ['b54', 'b57', 'b19']

Section: References
Ref_id:b0 Title: On maximal correlation, hypercontractivity, and the data processing inequality studied by Erkip and Cover Year: (2013)
Ref_id:b1 Title: A test of goodness of fit Year: (1954)
Ref_id:b2 Title: Self-supervised learning from images with a joint-embedding predictive architecture Year: (2023)
Ref_id:b3 Title: Spherical harmonics and approximations on the unit sphere: an introduction Year: (2012)
Ref_id:b4 Title: Lejepa: Provable and scalable self-supervised learning without the heuristics Year: (2025)
Ref_id:b5 Title: Gaussian embeddings: How JEPAs secretly learn your data density Year: (2025)
Ref_id:b6 Title: VICRegL: Self-supervised learning of local visual features Year: (2022)
Ref_id:b7 Title: Revisiting feature prediction for learning visual representations from video Year: (2024)
Ref_id:b8 Title: Post-hoc probabilistic vision-language models Year: (2024)
Ref_id:b9 Title: Whitened CLIP as a likelihood surrogate of images and captions Year: (2025)
Ref_id:b10 Title: General and domain-specific zeroshot detection of generated images via conditional likelihood Year: (2026)
Ref_id:b11 Title: On the maximum correlation coefficient Year: (2005)
Ref_id:b12 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b13 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b14 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b15 Title: Tests for departure from normality. Empirical results for the distributions of b 2 and √ b 1 Year: (1973)
Ref_id:b16 Title: Hyperspherical variational auto-encoders Year: (2018)
Ref_id:b17 Title: ImageNet: A large-scale hierarchical image database Year: (2009)
Ref_id:b18 Title: Asymptotics of graphical projection pursuit. The annals of statistics Year: (1984)
Ref_id:b19 Title: A dozen de Finetti-style results in search of a theory Year: (1987)
Ref_id:b20 Title: On the importance of embedding norms in self-supervised learning Year: (2025)
Ref_id:b21 Title: A weak convergence approach to the theory of large deviations Year: (2011)
Ref_id:b22 Title: On the importance of gaussianizing representations Year: (2025)
Ref_id:b23 Title: Whitening for selfsupervised representation learning Year: (2021)
Ref_id:b24 Title: Radial Bayesian neural networks: Beyond discrete support in large-scale Bayesian deep learning Year: (2020)
Ref_id:b25 Title: Das statistische problem der korrelation als variations-und eigenwertproblem und sein zusammenhang mit der ausgleichsrechnung Year: (1941)
Ref_id:b26 Title: Provable guarantees for self-supervised deep learning with spectral contrastive loss Year: (2021)
Ref_id:b27 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b28 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b29 Title: A connection between correlation and contingency. Mathematical proceedings of the cambridge philosophical society Year: (1935)
Ref_id:b30 Title: Densely connected convolutional networks Year: (2017)
Ref_id:b31 Title: On the sample complexity of HGR maximal correlation functions for large datasets Year: (2020)
Ref_id:b32 Title: Unsupervised feature extraction by time-contrastive learning and nonlinear ICA Year: (2016)
Ref_id:b33 Title: Nonlinear ICA using auxiliary variables and generalized contrastive learning Year: (2019)
Ref_id:b34 Title: Logarithmic bounds for isoperimetry and slices of convex sets Year: (2023)
Ref_id:b35 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b36 Title: A simple unified framework for detecting out-of-distribution samples and adversarial attacks Year: (2018)
Ref_id:b37 Title: The double-ellipsoid geometry of CLIP Year: (2025)
Ref_id:b38 Title: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning Year: (2022)
Ref_id:b39 Title: Microsoft COCO: Common objects in context Year: (2014)
Ref_id:b40 Title: Directional statistics Year: (2009)
Ref_id:b41 Title: Ii. illustrations of the dynamical theory of gases Year: (1860)
Ref_id:b42 Title: BayesAdapter: enhanced uncertainty estimation in CLIP few-shot adaptation Year: (2024)
Ref_id:b43 Title: Representation learning with contrastive predictive coding Year: (2018)
Ref_id:b44 Title: Prevalence of neural collapse during the terminal phase of deep learning training Year: (2020)
Ref_id:b45 Title: Calcul des probabilités Year: (1912)
Ref_id:b46 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b47 Title: Cross-entropy is all you need to invert the data generating process Year: (2024)
Ref_id:b48 Title: On measures of dependence Year: (1959)
Ref_id:b49 Title: On linear identifiability of learned representations Year: (2021)
Ref_id:b50 Title: A theoretical analysis of contrastive unsupervised representation learning Year: (2019)
Ref_id:b51 Title: Orthogonal polynomials Year: (1939)
Ref_id:b52 Title: What makes for good views for contrastive learning? Advances in neural information processing systems Year: (2020)
Ref_id:b53 Title:  Year: (2000)
Ref_id:b54 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b55 Title: Understanding contrastive representation learning through alignment and uniformity on the hypersphere Year: (2020)
Ref_id:b56 Title: Lecture notes on high-dimensional data Year: (2021)
Ref_id:b57 Title: Weak convergence and empirical processes: with applications to statistics Year: (2013)
Ref_id:b58 Title: HGR correlation pooling fusion framework for recognition and classification in multimodal remote sensing data Year: (2024)
Ref_id:b59 Title: Contrastive learning inverts the data generating process Year: (2021)
