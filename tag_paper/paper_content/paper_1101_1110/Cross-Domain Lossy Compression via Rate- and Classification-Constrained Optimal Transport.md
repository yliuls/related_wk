Title: CROSS-DOMAIN LOSSY COMPRESSION VIA RATE-AND CLASSIFICATION-CONSTRAINED OPTIMAL TRANSPORT
Abstract: We study cross-domain lossy compression, where the encoder observes a degraded source while the decoder reconstructs samples from a distinct target distribution. The problem is formulated as constrained optimal transport with two constraints on compression rate and classification loss. With shared common randomness, the one-shot setting reduces to a deterministic transport plan, and we derive closed-form distortion-rate-classification (DRC) and rate-distortionclassification (RDC) tradeoffs for Bernoulli sources under Hamming distortion. In the asymptotic regime, we establish analytic DRC/RDC expressions for Gaussian models under mean-squared error. The framework is further extended to incorporate perception divergences (Kullback-Leibler and squared Wasserstein), yielding closed-form distortion-rate-perception-classification (DRPC) functions. To validate the theory, we develop deep end-to-end compression models for superresolution (MNIST), denoising (SVHN, CIFAR-10, ImageNet, KODAK), and inpainting (SVHN) problems, demonstrating the consistency between the theoretical results and empirical performance.

Section: INTRODUCTION
Classical rate-distortion (RD) theory provides a single-letter characterization of the minimal distortion achievable when reproducing a source under a rate constraint (Cover & Thomas, 1999). This foundation has guided decades of research in lossy compression and inspired the design of modern learned codecs. However, standard RD formulations assume that reconstructions should remain close to the observed input distribution. In many setting, this assumption is misaligned: the encoder observes a degraded sample X (e.g., noisy or low-resolution), while the desired output is a restored sample Y that lies in a different, target distribution p Y (e.g., clean or high-resolution). Moreover, beyond fidelity, the compressed representation must remain informative for downstream tasks such as classification, introducing additional constraints that are not captured by classical RD.
Perception-aware RD extends the RD framework by incorporating a divergence between the source and reconstruction distributions, highlighting an intrinsic rate-distortion-perception (RDP) tradeoff and motivating generative compression approaches (Blau & Michaeli, 2018;2019;Theis & Wagner, 2021). In restoration tasks, the target is not the degraded input p X but the clean domain p Y , so enforcing perceptual closeness between p Y and p X is conceptually mismatched. Task-aware extensions such as rate-distortion-classification (RDC) or rate-distortion-perception-classification (RDPC) explicitly account for classification performance (Wang et al., 2025), but typically assume that reconstructions remain in a single domain rather than supporting cross-domain mappings with distinct marginals. In parallel, compression has also been studied as a denoising mechanism. Weissman et al. (Weissman & Ordentlich, 2005) showed that optimal lossy compression followed by postprocessing can asymptotically achieve the fundamental denoising limit, while more recent work (Zafari et al., 2025b) introduced neural compression-based denoising, including a zero-shot framework with theoretical guarantees and algorithmic instantiations. These results highlight the value of compression as a denoising prior but do not address cross-domain alignment or task-aware constraints.
Finally, optimal transport (OT) provides a principled tool for coupling distributions (Villani, 2009), and has been leveraged in unsupervised restoration (Wang et al., 2023a). However, OT by itself ignores coding constraints and does not account for downstream requirements. Related to our work, Liu et al. (2022) formulated cross-domain lossy compression as entropy-constrained OT and showed that shared randomness can decouple coding from transport, but classification constraints were not included. Unsupervised image restoration has also been studied in (Zhang et al., 2017;Menon et al., 2020;Pan et al., 2021) with a fixed reconstruction distribution; however, these approaches neither investigate compression constraints nor incorporate classification-awareness for downstream tasks.
To that end, we formulate cross-domain restoration as compression, through a constrained lossy optimal transport framework. Given degraded samples from p X and desired reconstructions from p Y , with distortion function, we optimize couplings p X,Y that simultaneously (i) minimize expected distortion, (ii) satisfy a rate constraint, and (iii) preserve task utility by constraining the uncertainty of the downstream label S given the reconstruction Y . Following Liu et al. (2022); Theis & Agustsson (2021); Theis & Wagner (2021), we exploit shared common randomness between the encoder and decoder to show that, in the one-shot regime, the system reduces to selecting a deterministic transport plan with effective rate and classification constraints. In the asymptotic regime, this leads to a mutual-information-constrained transport problem with an additional classification constraint, yielding a Shannon-style single-letter characterization. Our work makes the following contributions.
• We introduce constrained lossy optimal transport, generalizing OT by incorporating both coding and classification constraints. With common randomness, transport (reconstructing Y to match p Y ) and compression (coding Y ) structurally decouple, extending Liu et al. (2022) to task-aware settings. Closed-form characterizations are provided for (i) one-shot Bernoulli sources under Hamming distortion, where DRC and RDC functions admit piecewise-linear forms, and (ii) asymptotic Gaussian sources under mean-squared error (MSE), where analytic expressions are derived.
• The framework is further extended to the distortion-rate-perception-classification (DRPC) setting with two perception divergences: Kullback-Leibler (KL) and squared Wasserstein. For Gaussian sources, extremality results yield closed-form DRPC characterizations that, to our knowledge, are the first to explicitly incorporate classification.
• We implement deep end-to-end compression frameworks, incorporating (i) universal quantization for shared randomness, (ii) an entropy model for rate estimation, (iii) a WGAN discriminator for aligning reconstructions with p Y , and (iv) a classifier head for controlling classification loss. Experiments on super-resolution (MNIST), denoising (SVHN, CIFAR-10, ImageNet, KODAK), and inpainting (SVHN), demonstrate strong agreement with the theoretical results. We study a scenario where the encoder observes an input X ∼ p X , which represents a degraded version (e.g., corrupted by noise, reduced resolution) of an underlying clean source. Associated with each sample is a classification label S ∼ p S , with a prescribed covariance between X and S as Cov(X, S) = E[(X -µ X )(S -µ S )]. Following the approach in Liu et al. (2022), we utilize shared randomness between the encoder and decoder to enhance performance. Specifically, we introduce a common random variable U , accessible at both sides, which is independent of the input X, i.e., I(X; U ) = 0. This assumption ensures that the decoder has no prior knowledge of X beyond the transmitted representation. In practice, U can be realized by pre-agreeing on a pseudo-random number generator with a shared seed, enabling both parties to generate identical randomness. The encoder must map X into a compressed representation Z under a rate constraint, namely H(Z|U ) ≤ R. The decoder, given Z, produces a reconstruction Y that should follow a target distribution p Y .
this section cite: ['b17', 'b10', 'b59', 'b65', 'b67', 'b62', 'b41', 'b77', 'b43', 'b49', 'b41', 'b58', 'b59', 'b41', 'b41']

Section: SYSTEM MODEL AND ONE-SHOT SETTING RESULTS

this section cite: []

Section: Encoder
We consider p X and p Y as probability distributions over X , Y ⊆ R n . Formally, we enforce similarity between X and Y through a fidelity criterion defined by a distortion function d : X × Y → R. We assume d(X, Y ) = 0 if and only if X = Y . For instance, d(X, Y ) can be the Hamming distance or the MSE distortion. In addition, we require the reconstruction to remain useful for downstream classification, specifically, the uncertainty of S conditioned on Y is constrained as H(S|Y ) ≤ C for some C > 0 (Wang et al., 2025).
The main objective of source restoration (e.g., denoising or super-resolution) in our setting is threefold: (i) Degradation removal: mitigate artifacts and imperfections present in the degraded input X; (ii) Information preservation: retain as much information as possible about the underlying clean source X ′ contained in X; (iii) Classification utility: ensure that the reconstructed Y yields high classification performance. As an example, consider X as a noisy image and Y as its clean reconstruction. Figure 1 provides a schematic of the full system, where compression and restoration jointly yield a sample Y ∼ p Y that serves both fidelity and classification purposes. We interpret this formulation through the lens of optimal transport: the problem reduces to identifying a joint distribution p X,Y consistent with given marginals p X and p Y , subject to a distortion cost d(•, •), rate constraint R, and classification constraint C. We next connect this framework to optimal transport and describe how our formulation of classification-aware lossy compression naturally extends it.
this section cite: ['b65']

Section: ONE-SHOT CONSTRAINED OPTIMAL TRANSPORT
Definition 1 (Optimal Transport). Let Γ(p X , p Y ) denote the set of all joint distributions p X,Y with marginals p X and p Y . The classical optimal transport problem identifies a coupling in this set that minimizes the expected transportation cost:
D(p X , p Y ) = inf p X,Y ∈Γ(p X ,p Y ) E[d(X, Y )],(1)
where d(•, •) is a prescribed distortion (or cost) function and p X,Y ∈ Γ(p X , p Y ) is a transport plan.
The transport plan in Definition 1 minimizes the average distortion between input and output, subject only to the marginal distributions p X and p Y . Our goal is to extend this framework by requiring the transport plan to additionally satisfy a rate constraint and a classification constraint, as formalized in the following definition.
Definition 2 (Constrained Optimal Transport). Let X ∼ p X denote the degraded source, Y ∼ p Y be the reconstruction, and S ∼ p S be the associated classification variable with covariance Cov(X, S) with S ↔ X ↔ Y . Define M (p X , p Y ) as the set of joint distributions p U,X,Z,Y with marginals p X , p Y that factorize as p U,X,Z,Y = p U p X p Z|X,U p Y |Z,U , where U is the shared common randomness. The constrained optimal transport problem with rate constraint R, classification loss C, and shared randomness is given by
D(R, C, p X , p Y ) = inf p U,X,Z,Y ∈M (p X ,p Y ) E[d(X, Y )](2)
s.t. H(Z|U ) ≤ R, H(S|Y ) ≤ C,
Given an input X and shared randomness U , the encoder produces a compressed representation Z ∼ p Z|X,U . Leveraging U , the representation can be further losslessly encoded at an average rate not exceeding R. By standard coding theorems, any discrete Z admits a variable-length code with expected length at most H(Z|U ) + 1 bits. The decoder, with access to (Z, U ), reconstructs Y via p Y |Z,U , where the reconstruction is required to satisfy H(S|Y ) ≤ C. The optimization in (2) is carried out jointly over the distribution of shared randomness p U and the stochastic mappings p Z|X,U (encoder) and p Y |Z,U (decoder). Furthermore, the constrained cost satisfies D(R, C, p X , p Y ) ≥ D(p X , p Y ), where D(p X , p Y ) is the classical optimal transport cost from Definition 1. The next result provides a simplification of this architecture. Theorem 1. Define Q(p X , p Y ) as the set of joint distributions p U,X,Y with marginals p X , p Y that factorize as p U,X,Y = p U p X p Y |X,U . Then, the constrained optimal transport cost in Definition 2 admits the representation
D(R, C, p X , p Y ) = inf p U,X,Y ∈Q(p X ,p Y ) E[d(X, Y )](3)
s.t. H(Y |X, U ) = 0, I(X; U ) = 0, H(Y |U ) ≤ R, H(S|Y ) ≤ C.
Proof. The result follows by adapting Theorem 3 in Liu et al. (2022). For completeness, a detailed proof is provided in Appendix A.1.1.
𝑝 𝑍|𝑋,𝑈 𝑝 𝑌|𝑍,𝑈 𝑍 𝑋 𝑌 𝑈 𝑆 መ 𝑆 𝐻(𝑆|𝑌) 𝔼[𝑑(𝑋, 𝑌)] 𝑝 𝑌|𝑋,𝑈 𝑋 𝑌 𝑈 𝑆 መ 𝑆 𝐻(𝑆|𝑌) 𝔼[𝑑(𝑋, 𝑌)] Following (Liu et al., 2022), the problem can be equivalently expressed using only the conditional distribution p Y |X,U , which generates the reconstruction Y directly without an intermediate representation Z, similar to the classical optimal transport formulation in Definition 1. The condition H(Y |X, U ) = 0 ensures that the transport plan is deterministic once the shared randomness U is fixed, with U providing the sole source of stochasticity. In this architecture, the encoder maps (X, U ) to Y (transport), then compresses Y losslessly at an average rate approaching H(Y |U ) (compression), while enforcing H(S|Y ) ≤ C to preserve classification accuracy. The decoder simply decompresses and outputs Y .
this section cite: ['b41', 'b41']

Section: BERNOULLI CASE EXPRESSIONS
We now investigate the constrained optimal transport framework for Bernoulli sources.
Let X ∼ Bern(q X ) and Y ∼ Bern(q Y ) with 0 ≤ q X , q Y ≤ 1 2 . Using ⊕ for modulo-2 addition, note that X ⊕ Y = 1 iff X ̸ = Y . The classification variable S is modeled as S = X ⊕ S 1 , where S 1 ∼ Bern(q S1 ) with 0 ≤ q S1 ≤ 1 2 . This yields the marginal distribution q S = P (S = 1) = q X + q S1 -2q X q S1 .
We adopt the Hamming distortion d H (X, Y ) = 1{X ̸ = Y }. For any coupling of X and Y with these marginals, let p xy = P (X = x, Y = y). The expected distortion is
E[d H (X, Y )] = P (X ̸ = Y ) = p 01 + p 10 = q X + q Y -2p 11 .
Thus, minimizing (resp. maximizing) Pr(X ̸ = Y ) is equivalent to maximizing (resp. minimizing) p 11 subject to the Fréchet-Hoeffding bounds (Nelsen, 2006,
Sec. 2.5): max{0, q X + q Y -1} ≤ p 11 ≤ min{q X , q Y }. The minimum distortion is attained at p 11 = min{q X , q Y }, realized by the monotone coupling. Specifically, let U ∼ Unif[0, 1], with X = 1{U ≤ q X } and Y = 1{U ≤ q Y }. Then, D (B) min = min couplings P (X ̸ = Y ) = |q X -q Y |. The maximum distortion is attained at p 11 = max{0, q X + q Y -1}, realized by the antimonotone coupling: X = 1{U ≤ q X } and Y = 1{U ≥ 1 -q Y }. Since q X , q Y ≤ 1 2 , we obtain D (B) max = max couplings P (X ̸ = Y ) = q X + q Y . For the independent coupling, p 11 = q X q Y , yielding D (B) ind = P (X ̸ = Y ) = q X (1 -q Y ) + (1 - q X )q Y = q X + q Y -2q X q Y .
Building on these extremal couplings and benchmark distortions, we next derive the DRC tradeoff under common randomness.
Theorem 2. Consider a Bernoulli source X ∼ Bern(q X ), Y ∼ Bern(q Y )
this section cite: ['b47']

Section: , and a classification variable S with the binary symmetric joint distribution given by
S = X ⊕ S 1 where S ∼ Bern(q S ) and S 1 ∼ Bern(q S1 ) (0 ≤ q X , q S , q S1 ≤ 1 2 ). The problem (3) is feasible if C ≥ H b (q S1
). Assume the Hamming distortion measure. Under common randomness, we have
D (B) (R, C, q X , q Y ) =                  -2(1 -q X )q X (H b (m) -C) H b (m) -H b (q S1 ) + D (B) ind , H b (q s1 ) ≤ C ≤ R(H b (q S 1 )-H b (m)) H b (q X ) + H b (m) -2(1 -q X )q X R H b (q X ) + D (B) ind , C > R(H b (q S1 ) -H b (m)) H b (q X ) + H b (m) D (B) min , C > H b (q S ) and R > H b (q X ). where m = (1 -q X )(1 -q S1 ) + q X q S1 , q S = q X + q S1 -2q X q S1
and H b (.) denotes the binary entropy function.
Proof. The proof is provided in Appendix A.1.2.
Theorem 2 reveals three regimes. With a loose classification constraint C, distortion decreases linearly with rate R, approaching the independent coding distortion D (B)
ind . In the intermediate regime, distortion depends jointly on R and C, capturing the tradeoff between compression efficiency and task fidelity. When R exceeds the source entropy and C is sufficiently large, the minimal distortion D (B) min becomes achievable. Similarly, the definition and closed-form expression of R (B) (D, C, q X , q Y ) are provided in Appendix A.2. Classical rate-distortion theory is usually considered in the asymptotic block-length regime, where arbitrarily long i.i.d. sequences are compressed and coding theorems yield singleletter characterizations. Motivated by this, we now extend our one-shot constrained optimal transport formulation to the asymptotic case, where large block lengths are allowed. This generalization allows us to connect to Shannon's original setting and to establish information-theoretic characterizations that hold in the limit. Let {X i } ∞
this section cite: []

Section: SYSTEM MODEL AND ASYMPTOTIC SETTING RESULTS

this section cite: []

Section: ASYMPTOTIC CONSTRAINED OPTIMAL TRANSPORT
i=1 , {Y i } ∞ i=1 , and {S i } ∞ i=1 be i.i.d. processes with marginals p X , p Y , and p S , respectively.
Definition 3 (Asymptotic Constrained Optimal Transport). Consider i.i.d. random variables X i ∼ p X , Y i ∼ p Y , and S i ∼ p S . The asymptotic constrained optimal transport problem with rate constraint R, classification loss C, and shared randomness U in the asymptotic regime (n → ∞) is defined as
D (∞) (R, C, p X , p Y ) = inf p U,X n ,Z,Y n ∈M(⊗ n i=1 p X ,⊗ n i=1 p Y ) 1 n n i=1 E[d(X i , Y i )] s.t. 1 n H(Z|U ) ≤ R, 1 n n i=1 H(S i |Y i ) ≤ C.
Theorem 3. In the asymptotic regime, the DRC function admits the single-letter characterization Remark 1. As in the one-shot formulation, shared common randomness U can be leveraged in the asymptotic regime for the constrained optimal transport problem. However, since coding theorems in the block-length limit already allow randomized mappings without rate penalty, the asymptotic characterization in Theorem 3 coincides with the one-shot formulation, and the role of U does not further tighten the bound.
D (∞) (R, C, p X , p Y ) = inf p X,Y ∈Γ(p X ,p Y ) E[d(X, Y )](4)
this section cite: []

Section: GAUSSIAN CASE EXPRESSIONS
We now investigate the constrained optimal transport framework for Gaussian sources under MSE distortion. Let X ∼ N (µ X , σ 2 X ) and Y ∼ N (µ Y , σ 2 Y ) be Gaussian random variables, and let S ∼ N (µ S , σ 2 S ) denote the associated classification variable with θ 1 ≜ Cov(X, S). In the Gaussian case, we derive a single-letter characterization of the asymptotic DRC tradeoffs with shared randomness.
Theorem 4. Consider X ∼ N (µ X , σ 2 X ) and Y ∼ N (µ Y , σ 2 Y )
with MSE distortion, and let S ∼ N (µ S , σ 2 S ) be a classification variable with Cov(X, S) = θ 1 . The problem (4) is feasible if C ≥ 1 2 log 1 -
θ 2 1 σ 2 S σ 2 X + h(S).
Under shared randomness, the asymptotic DRC tradeoff is
D (G) (R, C, q X , q Y ) =                  (µ X -µ Y ) 2 + σ 2 X + σ 2 Y - 2σ S σ 2 X σ Y θ1 1 -e -2h(S)+2C , 1 2 log 1 - θ 2 1 σ 2 S σ 2 X + h(S) ≤ C ≤ 1 2 log 1 - θ 2 1 (1-2 -2R ) σ 2 S σ 2 X + h(S), (µ X -µ Y ) 2 + σ 2 X + σ 2 Y -2σ X σ Y 1 -2 -2R , C > 1 2 log 1 - θ 2 1 (1-2 -2R ) σ 2 S σ 2 X + h(S),
this section cite: []

Section: 0, C > h(S) and R > h(X).
Proof. A detailed proof is given in Appendix A.1.3.
We provide the definition and closed-form expression of R (G) (D, C, q X , q Y ) in Appendix A.3. Classical asymptotic RD analysis yields single-letter characterizations in the block-length limit. We extend this perspective to the DRPC setting, where reconstructions are required not only to satisfy fidelity and rate constraints but also to preserve classification performance (Wang et al., 2025) and align with a perceptual target distribution (Blau & Michaeli, 2019;Theis & Wagner, 2021).
this section cite: ['b65', 'b11', 'b59']

Section: ASYMPTOTIC DRPC FUNCTION FOR GAUSSIAN SOURCES
Definition 4 (Asymptotic DRPC Function). For i.i.d. random variables X i ∼ p X , Y i ∼ p Y , and S i ∼ p S , the DRPC function with common randomness in the asymptotic regime is defined as
D (∞) (R, P, C) = inf p U,X n ,Z,Y n 1 n n i=1 E[d(X i , Y i )] s.t. 1 n H(Z|U ) ≤ R, 1 n n i=1 H(S i |Y i ) ≤ C, 1 n n i=1 ϕ(p Xi , p Yi ) ≤ P.
where ϕ(•, •) is a nonnegative divergence capturing perceptual quality.
Theorem 5. In the asymptotic regime, the DRPC function admits the single-letter characterization
D (∞) (R, P, C) = inf p Y |X E[d(X, Y )] s.t. I(X; Y ) ≤ R, H(S|Y ) ≤ C, ϕ(p X , p Y ) ≤ P.
Proof. The result follows the asymptotic analysis in Theis & Wagner (2021); Saldi et al. (2015a) and adapting the arguments in Wang et al. (2025, Appendix F).
We investigate two perception divergences of particular interest. The first is the Kullback-Leibler divergence, defined as ϕ(p Y ) . The second divergence we consider is the squared quadratic Wasserstein distance, defined as
X , p Y ) = ϕ KL (p Y ∥p X ) = E log p Y (Y ) p X (
W 2 2 (p X , p Y ) = inf p XY ∈Γ(p X ,p Y ) E[(X -Y ) 2 ].
Since the source variables {X i } are i.i.d. and so are the reconstructions {Y i }, the divergence term ϕ(p Xi , p Yi ) is independent of i. Thus, DRPC coding can be viewed as output-constrained source coding, where the reconstruction distribution is restricted to the set {p Y : ϕ(p X , p Y ) ≤ P }. Accordingly, the DRPC function is given by
D (∞) (R, P, C) = inf p Y :ϕ(p X ,p Y )≤P D (∞) (R, C, p X , p Y ).
Unlike D (∞) (R, C, p X , p Y ), the reconstruction distribution in D (∞) (R, P, C) is not fixed but only required to satisfy the perceptual constraint ϕ(p X , p Y ) ≤ P . Leveraging the jointly Gaussian structure, we obtain closed-form characterizations of the DRPC tradeoff under MSE distortion, subject to both classification and perception constraints. In particular, explicit expressions are derived when the perception measure is chosen as either the KL divergence or the quadratic Wasserstein distance.
Theorem 6. Let X ∼ N (µ X , σ 2 X ) be a Gaussian source and S ∼ N (µ S , σ 2 S ) a classification variable jointly Gaussian with X, such that Cov(X, S) = θ 1 . Consider Y with mean E[Y ] = µ Y , variance Var(Y ) = σ 2 Y , and covariance Cov(X, Y ) = θ 2 . Define Y G as a Gaussian random variable such that (X, Y G ) is jointly Gaussian with the same first and second moments as (X, Y ): E[Y
G ] = µ Y , Var(Y G ) = σ 2
Y , and Cov(X, Y G ) = θ 2 . Under the MSE distortion with constraints I(X; Y ) ≤ R, h(S|Y ) ≤ C, and ϕ(q X , q Y ) ≤ P , the function
D (∞) (R, P, C) is attained by such a jointly Gaussian Y G when the perception measure is either W 2 2 (q X , q Y ) or ϕ KL (q Y ∥q X ).
Proof. The proof is provided in Appendix A.1.4.
Theorem 7. Let X ∼ N (µ X , σ 2 X ) and Y ∼ N (µ Y , σ 2 Y ) be two Gaussian random variables. Let S ∼ N (µ S , σ 2
S ) be an associated classification variable with a covariance of Cov(X, S) = θ 1 and be jointly Gaussian. For the case
d(X, Y ) = (X -Y ) 2 and ϕ(p X , p Y ) = ϕ KL (p Y ∥p X ), we have D (G) KL (R, P, C) =                                                σ 2 X -σ 2 X (1 -2 -2R ), σ(P ) ≤ σ X 1 -2 -2R and C > 1 2 log 1 - θ 2 1 (1 -2 -2R ) σ 2 S σ 2 X + h(S) σ 2 X + σ 2 (P ) -2σ X σ(P ) 1 -2 -2R , σ(P ) > σ X √ 1 -2 -2R and C > 1 2 log 1 - θ 2 1 (1-2 -2R ) σ 2 S σ 2 X + h(S) σ 2 X - σ 2 S σ 4 X θ 2 1 (1 -2 -2h(S)+2C ), σ(P ) ≤ σ S σ 2 X θ 1 1 -2 -2h(S)+2C
and
1 2 log 1 - θ 2 1 σ 2 S σ 2 X + h(S) ≤ C ≤ 1 2 log 1 - θ 2 1 (σ 2 X -σ 2 X 2 -2R ) σ 2 S σ 4 X + h(S) σ 2 X + σ 2 (P ) - 2σ S σ 2 X σ(P ) θ 1 1 -2 -2h(S)+2C , σ(P ) > σ S σ 2 X θ 1 1 -2 -2h(S)+2C
and
1 2 log 1 - θ 2 1 σ 2 S σ 2 X + h(S) ≤ C ≤ 1 2 log 1 - θ 2 1 (σ 2 X -σ 2 X 2 -2R ) σ 2 S σ 4 X + h(S) 0, C > h(S) and R > h(X).
where σ(P ) is the unique number σ ∈ [0, σ X ] satisfying ψ(σ) = P and ψ(σ
Y ) = log σ X σ Y + σ 2 Y -σ 2 X 2σ 2 X .
Proof. A complete proof is given in Appendix A.1.5.
A detailed derivation of the closed-form expression for D (∞) (R, P, C) under the quadratic Wasserstein distance can be found in Appendix A.4.
this section cite: ['b59']

Section: RELATED WORKS
Classical rate-distortion theory characterizes the fundamental limits of lossy compression with the best achievable distortion at a given rate (Cover & Thomas, 1999), while the information bottleneck links compression with task relevance (Chechik et al., 2003). Task-aware extensions, CDP (Liu et al., 2019a;b) and the RDC/RDPC formulations (Zhang, 2023;Wang et al., 2025), make explicit how accuracy constraints reshape the RD function, yet they typically operate within a single domain.
Perception-aware RD augments RD tradeoff with a divergence between source and reconstructions (Blau & Michaeli, 2018;2019;Theis & Wagner, 2021), inspiring generative codecs based on adversarial learning and distribution-preserving objectives (Goodfellow et al., 2014;Arjovsky et al., 2017;Gulrajani et al., 2017;Tschannen et al., 2018;Agustsson et al., 2019;Mentzer et al., 2020). Modern learned codecs pair analysis-synthesis transforms with entropy models and tighter rate estimation (Ballé et al., 2017;2018;Minnen et al., 2018;Theis et al., 2017;Williams et al., 2020;Johnston et al., 2018;Agustsson et al., 2017;Mentzer et al., 2018;Wu et al., 2020;Alemi et al., 2018;Brekelmans et al., 2019;Huang et al., 2020;Park et al., 2020). Beyond fidelity, compression has been used as a denoising prior from asymptotic limits (Weissman & Ordentlich, 2005) to recent neural and zero-shot frameworks with theoretical guarantees (Zafari et al., 2025a;b). Optimal transport provides a principled way to couple marginals (Villani, 2009) and has informed unsupervised restoration (Wang et al., 2023a). Most relevant to our work, cross-domain lossy compression has been cast as entropy-constrained OT with shared randomness that decouples coding and transport (Liu et al., 2022). Unsupervised image restoration has also been explored in (Zhang et al., 2017;Menon et al., 2020;Pan et al., 2021) with a fixed reconstruction distribution, but these approaches neither impose compression constraints nor incorporate classification-awareness for downstream tasks. Common randomness and stochastic encoders, realized via universal/dithered quantization, enable output constraints and "free" synthesis randomness (Saldi et al., 2015a;2013;Schuchman, 1964;Gray & Stockham, 1993;Ziv, 1985;Li & El Gamal, 2018a;Theis & Agustsson, 2021). Our work extends this line by introducing rate-and classification-constrained optimal transport, providing one-shot and asymptotic characterizations with closed-form tradeoffs, and validating the theory through deep generative compression models.
this section cite: ['b17', 'b15', 'b47', 'b79', 'b65', 'b10', 'b59', 'b25', 'b5', 'b28', 'b3', 'b45', 'b6', 'b46', 'b60', 'b68', 'b31', 'b2', 'b44', 'b70', 'b4', 'b13', 'b29', 'b50', 'b67', 'b47', 'b62', 'b41', 'b77', 'b43', 'b49', 'b53', 'b56', 'b27', 'b80', 'b58']

Section: EXPERIMENTAL RESULTS

this section cite: []

Section: TRAINING SETUP
We consider the setting where the encoder observes degraded samples X ∼ p X (e.g., noisy or low-resolution) and the goal is to reconstruct outputs from a distinct target distribution Y ∼ p Y (e.g., clean or high-resolution). The objective is to compress X while ensuring that reconstructions preserve semantic content, align with p Y , and remain predictive of the downstream label S. Note that Y is drawn from the clean dataset distribution, but does not correspond to the exact clean counterpart of X. In this unsupervised setting, only unpaired noisy and clean samples are available.
f,g,Q E ∥X -g(Q(f (X, U )))∥ 2 2 s.t. p g(Q(f (X,U ))) = p Y , H(Q(f (X, U ))) ≤ R, H(S|g(Q(f (X, U )))) ≤ C. Letting Ỹ = g(Q(f (X, U )))
, the WGAN discriminator aligns p Ỹ with p Y via a Wasserstein-1 penalty (Arjovsky et al., 2017). Shared randomness is implemented through universal quantization (Ziv, 1985;Theis & Agustsson, 2021). With trained encoder f and decoder g, restoration is obtained as Ỹ = g(Q(f (X) + U ) -U ), where U is common randomness available to both encoder and decoder. In practice, we optimize the relaxed loss
L = E[∥X -Ỹ ∥ 2 ] + λ p W 1 (p Y , p Ỹ ) + λ c CE(S, Ŝ),
which balances fidelity, distributional alignment, and classification.
this section cite: ['b5', 'b80', 'b58']

Section: RESULTS
Figure 6 presents the tradeoffs between rate, distortion, and accuracy. As expected, higher rates yield lower MSE and improved classification performance. We report both quantitative curves and qualitative results for super-resolution (MNIST) and denoising (SVHN). At low rates (e.g., R = 4 in Figure 6(c)), reconstructions capture coarse structure but remain blurry, stylized, or even ambiguous. At higher rates (e.g., R = 32), both distortion and perceptual quality improve substantially, producing reconstructions that closely match the high-resolution targets. A similar trend holds for denoising, where noisy digits progressively sharpen and become recognizable as the rate increases.  Additional results on entropy model-based rate estimation for super-resolution (MNIST) and denoising (SVHN, CIFAR-10, ImageNet, KODAK) are provided in Appendix B.1. We further examine the inpainting problem on SVHN under both supervised and unsupervised settings in Appendix B.2. Together, these experiments enrich the empirical study of the RDC tradeoff and provide additional evidence that the observed behaviors align closely with the theoretical predictions of our framework.
this section cite: []

Section: CONCLUSION
We studied cross-domain lossy compression, where the decoder reconstructs samples from a target distribution distinct from the degraded source observed by the encoder. By casting the problem as constrained optimal transport, compression rate and classification loss were unified into a single information-theoretic framework. In the one-shot regime with shared randomness, the problem reduces to a deterministic transport plan, and we derived closed-form DRC/RDC expressions for Bernoulli sources under Hamming distortion. In the asymptotic regime, analytic DRC/RDC tradeoffs were obtained for Gaussian sources under MSE. The framework was further extended to include perception divergences, such as KL and quadratic Wasserstein, leading to closed-form DRPC functions. To bridge theory and practice, we implemented deep end-to-end compression frameworks incorporating universal quantization for shared common randomness, entropy modeling for rate estimation, adversarial distribution alignment, and a task-specific classifier. Experiments on superresolution (MNIST), denoising (SVHN, CIFAR-10, ImageNet, KODAK), and inpainting (SVHN) confirmed that empirical performance closely matches the theoretical predictions.
this section cite: []

Section: References
Ref_id:b0 Title: A high-quality denoising dataset for smartphone cameras Year: (2018)
Ref_id:b1 Title: A rewriting system for convex optimization problems Year: (2018)
Ref_id:b2 Title: Soft-to-hard vector quantization for end-to-end learning compressible representations Year: (2017)
Ref_id:b3 Title: Generative adversarial networks for extreme learned image compression Year: (2019)
Ref_id:b4 Title: Fixing a broken elbo Year: (2018)
Ref_id:b5 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b6 Title: End-to-end optimized image compression Year: (2017)
Ref_id:b7 Title: Variational image compression with a scale hyperprior Year: (2018)
Ref_id:b8 Title: Noise2self: Blind denoising by self-supervision Year: (2019)
Ref_id:b9 Title: A semi-continuous version of the berger-yeung problem Year: (1999)
Ref_id:b10 Title: The perception-distortion tradeoff Year: (2018)
Ref_id:b11 Title: Rethinking lossy compression: The rate-distortion-perception tradeoff Year: (2019)
Ref_id:b12 Title: A unifying mutual information view of metric learning: cross-entropy vs. pairwise losses Year: ()
Ref_id:b13 Title: Exact rate-distortion in autoencoders via echo noise Year: (2019)
Ref_id:b14 Title: Denoiseg: Joint denoising and segmentation Year: (2020)
Ref_id:b15 Title: Information bottleneck for gaussian variables Year: (2003)
Ref_id:b16 Title: Kodak photocd true color image suite (24 high resolution images) Year: (1991)
Ref_id:b17 Title: Elements of information theory Year: (1999)
Ref_id:b18 Title: CVX: Matlab software for disciplined convex programming, version 2 Year: (2012-08)
Ref_id:b19 Title: Image denoising by sparse 3-d transform-domain collaborative filtering Year: (2007)
Ref_id:b20 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b21 Title: CVXPY: A python-embedded modeling language for convex optimization Year: (2016)
Ref_id:b22 Title: Image quality assessment: Unifying structure and texture similarity Year: (2020)
Ref_id:b23 Title: Network Information Theory Year: (2011)
Ref_id:b24 Title: A class of wasserstein metrics for probability Year: (1984)
Ref_id:b25 Title: Generative adversarial nets Year: (2014)
Ref_id:b26 Title: Graph implementations for nonsmooth convex programs Year: (2008)
Ref_id:b27 Title: Dithered quantizers Year: (1993)
Ref_id:b28 Title: Improved training of wasserstein gans Year: (2017)
Ref_id:b29 Title: Evaluating lossy compression rates of deep generative models Year: (2020)
Ref_id:b30 Title: Globally and locally consistent image completion Year: (2017)
Ref_id:b31 Title: Improved lossy image compression with priming and spatially adaptive bit rates for recurrent networks Year: (2018)
Ref_id:b32 Title: A method for stochastic optimization Year: (2014)
Ref_id:b33 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b34 Title: Noise2void -learning denoising from single noisy images Year: (2019)
Ref_id:b35 Title: High-quality self-supervised deep image denoising Year: (2019)
Ref_id:b36 Title: Gradient-based learning applied to document recognition Year: (1998)
Ref_id:b37 Title: Strong functional representation lemma and applications to coding theorems Year: (2018)
Ref_id:b38 Title: Strong functional representation lemma and applications to coding theorems Year: (2018)
Ref_id:b39 Title: On the classification-distortion-perception tradeoff Year: (2019)
Ref_id:b40 Title: On the classification-distortion-perception tradeoff Year: (2019)
Ref_id:b41 Title: Cross-domain lossy compression as entropy constrained optimal transport Year: (2022)
Ref_id:b42 Title: Learning a no-reference quality metric for single-image super-resolution Year: (2017)
Ref_id:b43 Title: Pulse: Selfsupervised photo upsampling via latent space exploration of generative models Year: (2020)
Ref_id:b44 Title: Conditional probability models for deep image compression Year: (2018)
Ref_id:b45 Title: High-fidelity generative image compression Year: (2020)
Ref_id:b46 Title: Joint autoregressive and hierarchical priors for learned image compression Year: (2018)
Ref_id:b47 Title: An Introduction to Copulas Year: (2006)
Ref_id:b48 Title: Reading digits in natural images with unsupervised feature learning Year: (2011)
Ref_id:b49 Title: Exploiting deep generative prior for versatile image restoration and manipulation Year: (2021)
Ref_id:b50 Title: Interpreting rate-distortion of variational autoencoder and using model uncertainty for anomaly detection Year: (2020)
Ref_id:b51 Title: Context encoders: Feature learning by inpainting Year: (2016)
Ref_id:b52 Title: Universal modeling and coding Year: (1981)
Ref_id:b53 Title: Randomized quantization and optimal design with a marginal constraint Year: (2013)
Ref_id:b54 Title: Output constrained lossy source coding with limited common randomness Year: (2015)
Ref_id:b55 Title: Finite-state controllers for partially observable multiagent systems Year: (2015)
Ref_id:b56 Title: Dither signals and their effect on quantization noise Year: (1964)
Ref_id:b57 Title: Jpeg2000: Image compression fundamentals, standards and practice Year: (2002)
Ref_id:b58 Title: On the advantages of stochastic encoders Year: (2021)
Ref_id:b59 Title: A coding theorem for the rate-distortion-perception function Year: (2021)
Ref_id:b60 Title: Lossy image compression with compressive autoencoders Year: (2017)
Ref_id:b61 Title: Deep generative models for distributionpreserving lossy compression Year: (2018)
Ref_id:b62 Title: Optimal Transport: Old and New Year: (2009)
Ref_id:b63 Title: Optimal transport for unsupervised denoising learning Year: (2023)
Ref_id:b64 Title: Optimal transport for unsupervised denoising learning Year: (2023)
Ref_id:b65 Title: Task-oriented lossy compression with data, perception, and classification constraints Year: (2025)
Ref_id:b66 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b67 Title: The empirical distribution of rate-constrained source codes Year: (2005)
Ref_id:b68 Title: Hierarchical quantized autoencoders Year: (2020)
Ref_id:b69 Title:  Year: (2005)
Ref_id:b70 Title: A gan-based tunable image compression system Year: (2020)
Ref_id:b71 Title: Output-constrained lossy source coding with application to rate-distortion-perception theory Year: (2025)
Ref_id:b72 Title: Generative image inpainting with contextual attention Year: (2018)
Ref_id:b73 Title: Free-form image inpainting with gated convolution Year: (2019)
Ref_id:b74 Title: Decompress: Denoising via neural compression Year: (2025)
Ref_id:b75 Title: Zero-shot denoising via neural compression: Theoretical and algorithmic framework Year: (2025)
Ref_id:b76 Title: Universal rate-distortion-perception representations for lossy compression Year: (2025)
Ref_id:b77 Title: Learning deep CNN denoiser prior for image restoration Year: (2017)
Ref_id:b78 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b79 Title: A rate-distortion-classification approach for lossy image compression Year: (2023-09)
Ref_id:b80 Title: On universal quantization Year: (1985)
