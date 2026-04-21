Title: INFORMATION SHAPES KOOPMAN REPRESENTATION
Abstract: The Koopman operator provides a powerful framework for modeling dynamical systems and has attracted growing interest from the machine learning community. However, its infinite-dimensional nature makes identifying suitable finitedimensional subspaces challenging, especially for deep architectures. We argue that these difficulties come from suboptimal representation learning, where latent variables fail to balance expressivity and simplicity. This tension is closely related to the information bottleneck (IB) dilemma: constructing compressed representations that are both compact and predictive. Rethinking Koopman learning through this lens, we demonstrate that latent mutual information promotes simplicity, yet an overemphasis on simplicity may cause latent space to collapse onto a few dominant modes. In contrast, expressiveness is sustained by the von Neumann entropy, which prevents such collapse and encourages mode diversity. This insight leads us to propose an information-theoretic Lagrangian formulation that explicitly balances this tradeoff. Furthermore, we propose a new algorithm based on the Lagrangian formulation that encourages both simplicity and expressiveness, leading to a stable and interpretable Koopman representation. Beyond quantitative evaluations, we further visualize the learned manifolds under our representations, observing empirical results consistent with our theoretical predictions. Finally, we validate our approach across a diverse range of dynamical systems, demonstrating improved performance over existing Koopman learning methods. The implementation is publicly available at https://github.com/Wenxuan52/InformationKoopman.

Section: INTRODUCTION
Modeling and predicting the behavior of nonlinear dynamical systems are fundamental problems in science and engineering (Brunton et al., 2020;Kovachki et al., 2023;Mezic, 2020). Classical approaches typically rely on nonlinear differential equations or black-box learning methods. In contrast, the Koopman operator framework offers a compelling alternative: it represents nonlinear evolution as a linear transformation in an appropriate function space (Koopman, 1931;Fritz, 1995).
Motivation. This linearization principle has recently attracted significant attention in the deep learning community, as it enables complex nonlinear dynamics to be modeled and predicted using linear representations. However, integrating this framework into deep architectures poses a fundamental challenge: the Koopman operator is inherently infinite-dimensional, necessitating the identification or learning of a suitable finite-dimensional subspace for practical implementation. Deep learning models, most notably variational autoencoders (VAEs), have been employed to approximate such subspaces in a purely data-driven manner (Otto & Rowley, 2019;Pan & Duraisamy, 2020;Liu et al., 2023). Yet in practice, the resulting representations often suffer from instability, mode collapse or fail to produce reliable dynamics. To address these challenges, some studies incorporate domain-specific priors-such as symmetry, conservation laws, dissipation, or ergodicity (Vaidya & Mehta, 2008;Weissenbacher et al., 2022;Azencot et al., 2020;Cheng et al., 2025)-into the Koopman representation. While effective in restricted settings, such approaches lack general principles for guiding Koopman representation. This calls for a more general and principled approach to constructing finite-dimensional representations, one that balances simplicity, in the form of latent linearity, with sufficient expressiveness (more literature review in Appendix C).
this section cite: ['b11', 'b33', 'b41', 'b26', 'b20', 'b42', 'b43', 'b36', 'b51', 'b56', 'b2', 'b14']

Section: Information Bottleneck View.
A natural way to achieve the tradeoff between simplicity and expressiveness is through the lens of the Information Bottleneck (IB). The classic IB framework formalizes the idea that a good representation should compress the input as much as possible while preserving information relevant to a downstream task (Tishby et al., 2000;Tishby & Zaslavsky, 2015). In the context of representation learning (see Table 1), this typically means finding a latent variable z that minimizes Complexity(x, z)foot_0 from input x, while retaining expressiveness by improving Relevance(z, y) (Vera et al., 2018). Instead of a static latent representation, the goal of Koopman representation is to predict the future state x n given the current state x n-1 via a latent variable z n-1 . This gives rise to a dynamical information bottleneck formulation: we aim to learn a Koopman representation z n-1 with maximal linear predictability of future state x n , while remaining as compact as possible.
Table 1: Information-theoretic comparison between standard and Koopman representations. Here, β controls the trade-off between simplicity and future-state expressiveness.
this section cite: ['b50', 'b49', 'b53']

Section: Latent Representation Koopman Representation Goal
Disentangled z Predictive zn-1
Info. Flow x → z → y xn-1 → zn-1 Koopman operator --------→ zn → xn
Lagrangian β Complexity(x, z) -Relevance(z, y) β Complexity(xn-1, zn-1) -Relevance(zn-1, xn)
Why Is Finding a Good Koopman Representation Challenging? Learning Koopman representation imposes stricter constraints than conventional latent representation models (see Table 1). In VAE or β-VAE (Kingma et al., 2013;Burgess et al., 2018), the focus is on reconstructing the input x or sampling from its distribution, which only requires the latent representation z to contain enough information about y. However, in Koopman learning, the latent space needs to support linear forward from z n to z n+1 in some finite-dimensional spaces. This constraint implies that the latent representation must not only capture information about the current state but also conform to a linear predictive structure (structural consistency) (Mardt et al., 2018;Kostic et al., 2023a;b;2024), which imposes a stronger restriction. Prior work has shown that simply increasing the dimensions of the latent space does not necessarily improve performance (Li et al., 2020;Brunton et al., 2021), underscoring the importance of maintaining temporal coherence, i.e., ensuring that latent trajectories evolve consistently over time to prevent instability and error accumulation. Moreover, predictive sufficiency requires that the latent representation retains enough Koopman modes to faithfully reconstruct the system's future trajectories, such that multi-step prediction accuracy is preserved (Wang et al., 2022;2025). Unlike standard VAEs and their variants, which emphasize flexible latent representations to support reconstruction, Koopman models demand dynamically consistent latent representations: small deviations can propagate and amplify over time. In summary, while conventional representation learning emphasizes disentanglement and reconstruction, Koopman representation learning requires three key properties: temporal coherence, predictive sufficiency, and structural consistency.
The IB framework provides a meta view to navigate these trade-offs. It enables us to ask the central question:
Is it possible to learn Koopman representations that are both structurally simple and expressive, under the guidance of information-theoretic principles?
Motivated by this question, we approach the problem from a fresh IB perspective, leading to core contributions: Theoretical Insight. We develop an information-theoretic framework for Koopman representation, proving that mutual information controls error bounds while von Neumann entropy determines the effective dimension. By disentangling the information content of Koopman representations, we reveal how temporal coherence, predictive sufficiency, and structural consistency are governed by latent information and how these components are intrinsically connected to the spectral properties of the Koopman operator. This yields a novel information-theoretic Lagrangian that extends the classical IB principle by explicitly incorporating dynamical constraints, thereby making the fundamental trade-off between simplicity and expressivity in Koopman representation mathematically explicit. Principled Framework. Building on our information-theoretic Lagrangian, we derive a tractable, architecture-agnostic loss function that translates our theory into a practical algorithm. Each term of the loss corresponds directly to one of the three desiderata-temporal coherence, predictive sufficiency, and structural consistency. This yields a general algorithm that is broadly applicable: it extends naturally from physical dynamical systems to high-dimensional visual inputs and graph-structured dynamics, and our empirical results validate the theoretical predictions.
this section cite: ['b25', 'b13', 'b39', 'b4', 'b35', 'b12', 'b54']

Section: PRELIMINARIES
Notation. Let M ⊂ R n be a finite-dimensional manifold equipped with a measure µ. Consider a discrete-time nonlinear map T : M → M, so that the state x t ∈ M evolves according to x t = T (x t-1 ). We denote by H = L 2 (M, µ), the Hilbert space of real-valued observables ϕ : M → R.
Definition 2.1 (Koopman Operator (Koopman, 1931)) The Koopman operator K : H → H is a linear operator acting on observables as
(Kϕ)(x) = ϕ(T (x)), for ϕ ∈ H, x ∈ M.(1)
Despite the appeal of lifting nonlinear dynamics into a linear forward via Koopman representation, practical approximations require projecting the infinite-dimensional function space H onto a finitedimensional subspace. In the Koopman learning framework, this restriction manifests as learning a finite set of effective latent features {ϕ 1 , ϕ 2 , . . . , ϕ d } that map the state x as a latent representation z := ϕ(x) ∈ Z ⊊ H, where Z is the latent space spanned by the selected latent features. The center of this paper is on discussion how to find a good representation z. To ground the principles of information theory, we introduce some essential technical definitions.
Definition 2.2 (Mutual Information (MacKay, 2003)) Given two random variables x and y with joint probability distribution p(x, y) and marginal distributions p(x) and p(y), the mutual information I(x; y) quantifies the amount of information shared between x and y, and is defined as
I(x; y) = E[log p(x, y) p(x)p(y) ] = E[log p(y|x) p(y) ].
Definition 2.3 (Von Neumann Entropy (Witten, 2020)) Let ρ ∈ R d×d be a symmetric, positive semidefinite matrix with trace 1. The von Neumann entropy of ρ is defined as
S(ρ) = -tr(ρ log ρ) . If {λ i } d i=1 are the eigenvalues of ρ, then S(ρ) = - d i=1 λ i log λ i .
This value reflects latent effective dimensions: it is close to 0 if ρ is concentrated on a single direction, and close to log d if ρ is spread uniformly. More connection with effective dimension is given in Appendix E.
Intuitively, mutual information and the von Neumann entropy provide a principled way to measure the predictability and the intrinsic effective dimension of Koopman representation. Building on these preliminaries, we can quantify the preserved information under Koopman representation.
this section cite: ['b58']

Section: METHOD Our approach proceeds as: (1) a probabilistic analysis in Koopman representation how information loss drives error accumulation; (2) an information-theoretic characterization linking lost information to Koopman spectral properties; (3) a general Lagrangian formulation to guide better representation.
A probabilistic view of Koopman representation. Firstly, we denote x 1:t and z 1:t as the states and their corresponding autoregressively generated latent variables from time step 1 to t, respectively. According to the direct information flow in Table 1, the Koopman representation induces the following trajectory distribution given x 0 :
p KR (x 1:t |x 0 ) = p(z 0 |x 0 ) t n=1 p(z n |z n-1 )p(x n |z n )dz 0 dz 1 • • • dz t .
(
Here, p(z 0 |x 0 ) acts as the encoder, mapping the initial state into a latent variable. The latent forward is modeled by a linear Gaussian transition, where p(z n |z n-1 ) = N (z n |Kz n-1 , Σ) is a probabilistic representation of equation 1 with variance Σ. This directly reflects Definition 2.1, as the latent evolution is constrained to be linear. Finally, each state x n is reconstructed from its corresponding latent variable z n via a decoder p(x n |z n ), typically instantiated as a Gaussian. We now turn to the fundamental question of whether information is inevitably lost during latent propagation.
Proposition 1 (Information Loss in Latent Evolution) Let x n-1 → z n-1 K -→ z n →
x n represent the information propagation in Koopman representation as shown in equation 2. Then, by the property of mutual information, the following holds:
I(x n-1 ; x n ) ≥ I(z n-1 ; x n ) ≥ I(z n-1 ; z n ).
(
The detailed proof and its multi-step extension are provided in Appendix F.1. The first inequality reflects that the mapping x n-1 → z n-1 is a compressed representation, which may discard predictive information about x n . The second inequality indicates that the latent forward propagation z n-1 → z n is governed by Koopman operator, inherently limits the information that can be preserved in the latent space. As a result, I(z n-1 ; x n ) is larger than I(z n-1 ; z n ), since the future state x n generally carries more dependencies on z n-1 than the latent evolution alone. In this sense, I(z n-1 ; z n ) sets the information limit of Koopman representation by the operator K.
While Proposition 1 shows the degradation of information along latent propagation, it remains an abstract statement that is not directly tractable under the complex trajectory distributions in equation 2. To obtain a tractable measure, we turn to the Kullback-Leibler (KL) divergence as a natural way to quantify the discrepancy between true and Koopman-induced trajectories:
D KL p(x 1:t |x 0 ) ∥ q KR (x 1:t |x 0 ) ≤ D KL p(x 1:t |x 0 ) ∥ p KR (x 1:t |x 0 ) + E enc + E tra + E rec (4)
Here, p is the true distribution and p KR is the ideal Koopman model distribution in equation 2 without any approximations. q KR is the variational approximation, E enc , E tra and E rec are approximation errors induced by the latent representation, Koopman operator and reconstruction (see details in Appendix F.2). This motivates the following result, which formalizes how the information gap translates into an autoregressive error bound for Koopman representations.
Proposition 2 (Autoregressive Error Bound of Koopman Representation) The distribution discrepancy between the true and Koopman-induced trajectories is bounded by the information gap as
∥p(x1:t | x0) -q KR (x1:t | x0)∥T V ≤ 1 2 [DKL(p(x1:t | x0) ∥ p KR (x1:t | x0)) + E] ≤ 1 2 t n=1 I(xn-1; xn) -I(zn-1; zn) + E.(5)
Here, ∥ • ∥ T V is the total variation distance. The upper error bound is obtained as
E q KR [x1:t | x0] -Ep[x1:t | x0] 2 ≤ C 2 t n=1 I(xn-1; xn) -I(zn-1; zn) + E,(6)
where C is a positive constant and E is related to the approximation error in equation 4.
The proof is in Appendix F.3. The KL divergence between the true and Koopman-induced trajectory distributions reflects how much temporal coherence is lost during representation. Here, I(x n-1 ; x n ) quantifies the intrinsic dynamical coupling T in the original system, while I(z n-1 ; z n ) characterizes the information of that coupling that exists under Koopman representation. Since I(z n-1 ; z n ) acts as the information limit (see Proposition 1), the gap I(x n-1 ; x n ) -I(z n-1 ; z n ) measures the information that is lost when nonlinear dynamics are approximated by Koopman representation. Also, we link the upper/lower error bounds and distribution discrepancy in equations 6 (lower bound see equation 25). It reflects the prediction error is bounded by the step-wise information limit.
this section cite: []

Section: INFORMATION COMPONENTS IN KOOPMAN REPRESENTATION
The latent mutual information quantifies the magnitude of error, but does not uncover how this loss relates to Koopman spectral properties. To sharpen the insight from Propositions 1 and 2, we consider the aggregated quantity I(z t ; x t ), which measures the total information available to the decoder p(x t |z t ). Our focus is on how much of this information can be stably propagated from past latent variables z t-n .
this section cite: []

Section: Koopman Operator

this section cite: []

Section: Encoder Decoder

this section cite: []

Section: Latent space
Input Output
(a): (b): (c): MI VNE MI + VNE
this section cite: []

Section: Koopman Operator

this section cite: []

Section: Encoding

this section cite: []

Section: Decoding

this section cite: []

Section: Spectral Water-Filling Effect

this section cite: []

Section: Spectral Mode Allocated Information
Overall Architecture
this section cite: []

Section: Spectral Information Disentanglement
Latent Space Original Space Component Temporal-coherent Fast-dissipating Residual Spectral property λ ≈ 1 λ < 1 no counterpart Mutual info term I(zt-n; zt) ↑ I(zt; xt-1 | zt-n) ↓ I(zt; xt | xt-1) ↓
The decomposition shows that Koopman representations preserve temporal-coherent information associated with spectral modes of the Koopman operator whose eigenvalues lie near the unit circle, while information linked to dissipating modes (|λ| < 1) decays rapidly and noiselike components have no spectral support, hence compressible.
(1). Temporal-coherent information I(z t-n ; z t ) (see closed form equation 29). This term represents information that persists along the latent evolution z t-n → • • • → z t . It corresponds to conserved or slowly dissipating information that remains stable during latent evolution. From a spectral perspective, these are associated with Koopman modes whose eigenvalues are near to the complex unit circle (i.e., |λ| ≈ 1), implying that the corresponding information is propagated almost losslessly across time and hence remains mutually informative between past and present latent variables.
(2).
Fast-dissipating information I(z t ; x t-1 |z t-n ) (see closed form equation 35). This term reflects short-term dependencies that arise from the most recent state x t-1 , beyond what is already encoded in the past latent state z t-n . Such information provides transient predictive power but quickly leaks out, since the autoregressive latent evolution z t-n → • • • → z t cannot continually access external inputs x t-1 . In contrast, these contributions are associated with Koopman modes whose eigenvalues satisfy |λ| < 1, indicating exponential information dissipation with forward steps. Consequently, the mutual information they contribute vanishes rapidly as the time step n increases, making those modes inherently cannot be captured by temporal-coherent information.
(3). Residual information I(z t ; x t |x t-1 ) (see closed form equation 36). This term measures unpredictable information in the current state that cannot be explained from the past state. It corresponds to information injected at the present step-such as noise or anomalies-that interferes with constructing a coherent latent state. Unlike temporal-coherent or fast-dissipating modes, these residuals have no spectral counterpart in the Koopman operator: they are not tied to any eigenvalue structure.
From the IB perspective, such non-predictive component is compressible. Having the disentangled information, the next question is how latent mutual information shapes Koopman representations.
Proposition 4 (The Role of Latent Mutual Information) Maximizing the latent mutual information I(z t-n ; z t ) allocates spectral weights to temporally coherent modes in the latent space, thereby enhancing the relevance of the Koopman representation. However, excessive emphasis on this objective can lead to mode collapse, where the representation concentrates on only a few dominant modes and loses effective dimension (see Figure 1(c)).
In Koopman representation, the latent mutual information admits a closed form
I(z t-n ; z t ) = 1 2 log det(I + M -1 2 n (K n )C(K n ) ⊤ M -1 2 n )(7)
where det denotes the determinant, I is the identity matrix, C := Cov(z t-n ) is the latent covariance matrix and M n := n-1
i=0 K i Σ(K i ) ⊤
is the n-step linear forward covariance (see detailed explanation and proof in Appendix F.5). We find that, from a Lagrangian perspective, maximizing I(z t-n ; z t ) of Koopman representation under the finite variance constraint tr(C) < ∞ leads to a water-filling allocation: variance is distributed along the directions corresponding to the largest
eigenvalues of M - 1 2 n K n C(K n ) ⊤ M - 1 2
n . These directions correspond to temporally coherent modes, which explains why higher latent mutual information enhances relevance. However, when the spectrum of this matrix is highly skewed, the water-filling solution degenerates into a low-rank allocation, squeezing information into only a few dominant directions. This effect reduces the effective dimension of the latent space Z, causing some spectral weights to vanish (cf. equation 43). To address the collapse induced by skewed spectral allocation, we next analyze how effective dimension can be preserved through entropy regularization.
this section cite: []

Section: Proposition 5 (Effective Dimension and Anti-Collapse) Low effective dimension (see Proposition 4) in Koopman representation indicates information collapse to few dominant modes and limits the model's ability to represent rich modes. Penalizing the von Neumann entropy S( C tr(C) ) encourages more expressive and spectrally diverse representations.
Connecting to Proposition 4, Appendix F.6 contains a detailed proof via a water-filling and Lagrangian view. The normalized operator C tr(C) can be regarded as a density matrix in Hilbert space, and the effective dimension can be measured as exp(S) (see Definition E.2). When penalized with large the von Neumann entropy, the water-filling solution attains a non-zero allocation across all modes, preventing variance from collapsing entirely onto a few dominant directions (cf. equation 46). This ensures a positive distribution of spectral weights across all modes, thereby avoiding degenerate spectra and increasing the effective dimension of the latent space Z (see Figure 1(c)).
this section cite: []

Section: INFORMATION-THEORETIC FORMULATION FOR PRACTICAL IMPLEMENTATION
The preceding analysis (Propositions 3, 4 and 5) reveals a fundamental trade-off in Koopman representation learning: maximizing latent mutual information enhances temporal coherence and predictive ability but risks mode collapse, whereas entropy regularization promotes spectral diversity for predictive sufficiency. Based on this principle, we formulate the following unified Lagrangian:
max z α log I(z t-n ; z t ) -βI(z t ; x t |z t-n ) + γS C tr(C) + log p(x t |z t ),(8)
where α, β and γ are Lagrangian multipliers. In equation 8, the first term in equation 7 preserves temporal-coherent information, the second term penalizes fast-dissipating or confounding components (I(z t ; x t |z t-n ) = I(z t ; x t-1 |z t-n ) + I(z t ; x t |x t-1 ), see proof in equation 31), the third term rewards larger von Neumann entropy of the normalized covariance to promote spectral diversity in the latent space Z. Lastly, log p(x t |z t ) is the reconstruction terms from predicted latent variable z t .
While the Lagrangian in equation 8 captures the desired information-theoretic trade-offs, it is not directly computable. To make it practical, we derive a tractable loss function for satisfying temporal coherence, predictive sufficiency and structural consistency
max n αI(z n ; P n ) Temporal coherence + βE p θ (zn|xn) [log q ψ (z n |z n-1 )] Structural consistency + βH p θ (z n |x n ) Encoder entropy + log p ω (x n |z n ) Reconstruction + γS( C tr(C) )
Predictive sufficiency
+ L ELBO .(9)
In VAE structure (shown in Figure 1(a)) , each component of the loss plays a distinct role in balancing the information-theoretic objectives:
(1) The mutual information I(z n ; P n ) captures temporal coherence by linking z n to its temporal neighborhood P n = {z n±i | 1 ≤ i ≤ k}, which includes immediate past and future latent states; in practice, this can be computed either via the closed form in equation 7 for low-dimensional latents, or approximated by InfoNCE (Wu et al., 2020) for high-dimensional settings.
(2) The term
-E p θ (zn|xn) [log q ψ (z n |z n-1 )] -H p θ (z n |x n ) serves as an equivalent representation of the conditional mutual information I(z t ; x t |z t-1 ), with linear Gaussian transition q ψ (z n |z n-1 ) = N (z n |K ψ z n-1 , Σ ψ ) and entropy of encoder H p θ (z n |x n ) (see Appendix G.1).
Minimizing this KL not only encourages the latent representation to capture information from the state x n , but also compresses fast-dissipating and residual components, ensuring that the representation remains expressive yet simple. Here,
E p θ (zn|xn) [log q ψ (z n |z n-1 )] enforces structural consistency in latent space. (3) The term log p ω (x n |z n ) is the decoder loss from predicted latent variable z n . (4) von Neumann entropy term S C tr(C) is computed from the normalized co- variance matrix C = 1 B B i=1 (z i -z)(z i -z) ⊤ of
this section cite: ['b59']

Section: EXPERIMENTS
Tasks. We evaluate our approach across three types of dynamical data: (1) Physical simulations, including Lorenz 63, Kármán vortex street, Dam flow, and weather forecasting task (ERA5), which test the ability to capture nonlinear, stochastic and high-dimensional physical dynamics; (2) Visualinput control, including image-based Planar, Pendulum, Cartpole, and 3-Link manipulator, which evaluate the ability to extract latent dynamics from high-dimensional visual inputs while controllable in latent spaces; and (3) Graph-structured dynamics prediction, including Rope and Soft Robotics, which tests generalized abilities of latent representation on dynamics with graph structures (see experimental details in Appendix G.3).
Metrics. We assess performance on both forecasting and control. For forecasting, we report (i) normalized root mean square error (NRMSE) for short-and long-term predictions (for physical simulation and graphs-structured dynamics), (ii) physical consistency metrics based on spectral distribution errors based on 1000-step sequences (SDEs), (iii) distributions of state measured by the Kullback-Leibler divergence (KLD), and (iv) structural similarity index (SSIM) for physical simulations. (v) the quality of low-dimensional manifold construction from high-dimensional visual inputs. For control, we measure the success rate of latent-space control of visual inputs following the setting in (Levine et al., 2020).
Baseline Algorithms. We compare against competitive baselines for each type of task. For physical simulations, we include VAE (Burgess et al., 2018), Koopman Autoencoder (KAE) (Pan et al., 2023), Koopman Kernel Regression (KKR) (Bevanda et al., 2023), and a SOTA Koopman variant for chaos -Poincaré Flow Neural Network (PFNN) (Cheng et al., 2025). For visual-input control, we consider VAE-based representation learning methods, including Embed to Control (E2C) (Banijamali et al., 2019), as well as Prediction, Consistency and Curvature (PCC) (Levine et al., 2020), together with KAE. For graph-structured dynamics, we compare with Compositional Koopman Operator (CKO) (Li et al., 2020), the current SOTA method for graph-structured dynamics. Result Analysis. Our analysis is organized around the contributions established in propositions(Section 3), and we structure the discussion by addressing the following key questions. (1) Does the latent mutual information determine the predictive limit of the Koopman representation? (Proposition 2) -Yes. This is verified by the quantitative results of physical simulations in Table 2. Consistent with proposition, the prediction error under Koopman representation inevitably accumulates and is bounded by the latent mutual information. By regularizing with latent mutual information, both short-and long-term predictions are improved. Notably, PFNN (Cheng et al., 2025) is a SOTA model specifically designed with domain-specific knowledge, while our method, grounded in general information theory, achieves comparable performance on chaotic tasks (Lorenz 63 and Kármán vortex). Compared with other Koopman-based methods, our approach yields substantial improvements in both physical consistency and predictive accuracy.
(2) How is the preserved information-particularly that associated with Koopman eigenmodes near the unit circle-shaped by latent mutual information and von Neumann entropy in constructing a dynamics-relevant manifold? (Proposition 4 and 5) The preserved information manifests in Koopman modes with eigenvalues lying close to the unit circle, capturing the recurrent structure of the Kármán vortex limit cycle, as shown in Figure 2 (left). However, KAE suffers from some eigenvalues collapse toward zero, reducing the effective latent dimension. This collapse explains the drift observed in its autoregressive prediction. In contrast, our model captures the limit-cycle structure and produces stable autoregressive trajectories, consistently revolving around the true orbit (Figure 2, right). Baselines such as KKR and PFNN also capture limit-cycle structure (via one-step reconstruction) but gradually deviate from the correct trajectory over long horizons. By incorporating latent mutual information, we ensure that temporal-coherent information is retained, while von Neumann entropy prevents eigenvalue degeneration and preserves sufficient modes. Consequently, the information behind those modes can be preserved over long horizons, which directly translates into improved long-term prediction accuracy and statistical consistency, as also reported in Table 2. (3) How does explicit information-theoretic regularization sufficiently capture essential dynamics, compared with VAEs and Koopman autoencoders? (Proposition 4 and 5) As shown in the reconstructed manifolds of Figure 3, our method produces a latent manifold that aligns most closely with the ground truth. For E2C, which is directly built on a VAE architecture, the latent geometry is heavily distorted (the loss of coherence). The manifold learned by KAE collapses into a nearly onedimensional structure, reflecting the lack of effective dimensions in its latent space. PCC, a modified VAE-based method designed to improve manifold construction, demonstrates partial improvement but still exhibits a gap compared with our approach. By preserving both effective dimensionality and temporal coherence, our Koopman representation achieves the best average control performance in both noiseless and noisy environments (Table 8 and 9 in Appendix G.5.2).
Ground truth PCC manifold E2C manifold KAE manifold Our manifold Y-axis position 2018-01-01 08:00 Ground Truth KAE KKR PFNN Ours 2018-01-08 08:00 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 Error 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 0.00000 0.00614 0.01227 0.01840 Error 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 0.00000 0.00491 0.00981 0.01472 Figure 4: Comparison of continuous predictions for the global humidity starting from 2018 -01 -01 -00 : 00 to 2018 -01 -08 -08 : 00. Error maps in the lower panels demonstrate that, compared with other models, showing with more stable and accurate results of our model (see more demonstration in Appendix G.5.1).
(4) How robust are the findings under noise, extended prediction horizons, and large-scale settings? (Proposition 1 and 2) Our method remains robust under both noisy observations and extended prediction horizons. As shown in Table 2 and Figure 4, it maintains stable performance in long-term rollouts and physical statistics in large scale weather forecasting. Moreover, our approach supports control under noisy environments, achieving competitive performance. These quantitative results are consistent with our probabilistic propositions.
(5) To what extent can our Lagrangian formulation be generalized to diverse architectures and adapted to support downstream tasks? (Proposition 1-5) Our formulation demonstrates broad applicability: it consistently improves performance across physical simulations (see Table 2), visual perception tasks for manifold construction and control (see Figure 3, Tables 8 and 9 in Appendix G.5.2), and graph-structured dynamics prediction (see Figure 5). These gains indicate that the proposed Lagrangian principle is architecture-agnostic and can be readily incorporated into different settings to enhance both predictive accuracy and task effectiveness (more results are referred to Appendix G.5). Pendulum Angle π -π Figure 6: Ablation study on the pendulum task. Latent manifolds are learned from high-dimensional pendulum images, where the ground-truth phase space is isomorphic to S 1 × R. Color represents the pendulum angle. Each subplot corresponds to removing or adjusting one regularization term: latent mutual information (α), KL divergence (β), and von Neumann entropy (γ). Ablation Studies. We analyze the effect of varying each Lagrangian multiplier to understand its role in shaping Koopman representation. In the pendulum task, the ground-truth phase space is S 1 × R, consisting of a periodic angle and an angular velocity. The ablation study in Figure 6 illustrates how each regularization term contributes to recovering this manifold from high-dimensional visual inputs. Without mutual information regularization (α = 0), temporal coherence is lost and the latent space degenerates into scattered points without geometric structure. Without structural consistency (β = 0), the latent manifold collapses, highlighting its role in enforcing the dynamics of Koopman representation. Removing the von Neumann entropy term (γ = 0) retains the circular S 1 component but suppresses the R dimension, indicating the necessity of preserving effective dimensions.
Increasing mutual information alone concentrates the representation on the S 1 component (reflecting Proposition 4), while regularizing with von Neumann entropy yields a manifold that closely approximates the full S 1 × R structure. These observations align with the theoretical roles of the three penalties: temporal coherence, structural consistency and predictive sufficiency.
this section cite: ['b34', 'b13', 'b44', 'b7', 'b14', 'b5', 'b34', 'b35', 'b14']

Section: CONCLUSION
We presented a new perspective on Koopman representation by formulating it through an information-theoretic lens, leading to a general Lagrangian formulation that balances simplicity and expressiveness. Our analysis reveals the relationship between Koopman spectral properties and information in deep architectures. The proposed algorithm based on the Lagrangian formulation consistently improves the performance in a wide range of dynamical system tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Reproducing kernel spaces and applications Year: (2012)
Ref_id:b1 Title: Ergodic theory, dynamic mode decomposition, and computation of spectral properties of the koopman operator Year: (2017)
Ref_id:b2 Title: Forecasting sequential data using consistent koopman autoencoders Year: (2020)
Ref_id:b3 Title: Information theory with kernel methods Year: (2022)
Ref_id:b4 Title: Universal approximation theorem for interval neural networks Year: (1998)
Ref_id:b5 Title: Robust locally-linear controllable embedding Year: (2019)
Ref_id:b6 Title: Reproducing kernel Hilbert spaces in probability and statistics Year: (2011)
Ref_id:b7 Title: Koopman kernel regression Year: (2023)
Ref_id:b8 Title:  Year: (2025)
Ref_id:b9 Title: Convex optimization Year: (2004)
Ref_id:b10 Title: Guidance laws for planar motion control Year: (2008)
Ref_id:b11 Title: Machine learning for fluid mechanics Year: (2020)
Ref_id:b12 Title: Modern koopman theory for dynamical systems Year: (2021)
Ref_id:b13 Title: Understanding disentangling in beta-vae Year: (2018)
Ref_id:b14 Title: Learning chaos in a linear way Year: (2025)
Ref_id:b15 Title: Elements of information theory Year: (1999)
Ref_id:b16 Title: Information theory and statistics: A tutorial Year: (2004)
Ref_id:b17 Title: Koopman spectra in reproducing kernel hilbert spaces Year: (2020)
Ref_id:b18 Title: Reproducing kernel hilbert space compactification of unitary evolution groups Year: (2021)
Ref_id:b19 Title: Latent representation and simulation of markov processes via time-lagged information bottleneck Year: (2023)
Ref_id:b20 Title: John von neumann and ergodic theory Year: (1995)
Ref_id:b21 Title: Swing up control of inverted pendulum Year: (1991)
Ref_id:b22 Title: Unpublished benchmark description (often cited in RL literature) Year: (1993)
Ref_id:b23 Title: The era5 global reanalysis Year: (2020)
Ref_id:b24 Title: Universal approximation with deep narrow networks Year: (2020)
Ref_id:b25 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b26 Title: Hamiltonian systems and transformation in hilbert space Year: (1931)
Ref_id:b27 Title: Dynamical systems of continuous spectra Year: (1932)
Ref_id:b28 Title: Optimal construction of koopman eigenfunctions for prediction and control Year: (2020)
Ref_id:b29 Title: Learning dynamical systems via koopman operator regression in reproducing kernel hilbert spaces Year: (2022)
Ref_id:b30 Title: Sharp spectral rates for koopman operator learning Year: (2023)
Ref_id:b31 Title: Learning invariant representations of time-homogeneous stochastic dynamical systems Year: (2023)
Ref_id:b32 Title: Consistent long-term forecasting of ergodic dynamical systems Year: (2024)
Ref_id:b33 Title: Neural operator: Learning maps between function spaces with applications to pdes Year: (2023)
Ref_id:b34 Title: Prediction, consistency, curvature: Representation learning for locally-linear control Year: (2020)
Ref_id:b35 Title: Learning compositional koopman operators for model-based control Year: (2020)
Ref_id:b36 Title: Learning non-stationary time series dynamics with koopman predictors Year: (2023)
Ref_id:b37 Title:  Year: (1997)
Ref_id:b38 Title: Information theory, inference and learning algorithms Year: (2003)
Ref_id:b39 Title: Vampnets for deep learning of molecular kinetics Year: (2018)
Ref_id:b40 Title: Koopman operator in systems and control Year: (2020)
Ref_id:b41 Title: Koopman operator, geometry, and learning Year: (2020)
Ref_id:b42 Title: Linearly recurrent autoencoder networks for learning dynamics Year: (2019)
Ref_id:b43 Title: Physics-informed probabilistic learning of linear embeddings of nonlinear dynamics with guaranteed stability Year: (2020)
Ref_id:b44 Title: Pykoopman: A python package for data-driven approximation of the koopman operator Year: (2023)
Ref_id:b45 Title: Weatherbench 2: A benchmark for the next generation of data-driven global weather models Year: (2023)
Ref_id:b46 Title: The effective rank: A measure of effective dimensionality Year: (2007)
Ref_id:b47 Title: Information theory: A tutorial introduction to the principles and applications of information theory Year: (2024)
Ref_id:b48 Title: Learning koopman invariant subspaces for dynamic mode decomposition Year: (2017)
Ref_id:b49 Title: Deep learning and the information bottleneck principle Year: (2015)
Ref_id:b50 Title: The information bottleneck method Year: (2000)
Ref_id:b51 Title: Lyapunov measure for almost everywhere stability Year: (2008)
Ref_id:b52 Title: Lambert w function for applications in physics Year: (2012)
Ref_id:b53 Title: The role of the information bottleneck in representation learning Year: (2018)
Ref_id:b54 Title: Rethinking minimal sufficient representation in contrastive learning Year: (2022)
Ref_id:b55 Title: Rethinking selectivity in state space models: A minimal predictive sufficiency approach Year: (2025)
Ref_id:b56 Title: Koopman qlearning: Offline reinforcement learning via symmetries of dynamics Year: (2022)
Ref_id:b57 Title: A data-driven approximation of the koopman operator: Extending dynamic mode decomposition Year: (2015)
Ref_id:b58 Title: A mini-introduction to information theory Year: (2020)
Ref_id:b59 Title: On mutual information in contrastive learning for visual representations Year: (2020)
Ref_id:b60 Title: Bin Yang, and Chenjuan Guo. k2 vae: A koopman-kalman enhanced variational autoencoder for probabilistic time series forecasting Year: (2025)
Ref_id:b61 Title: Reinforced data-driven estimation for spectral properties of koopman semigroup in stochastic dynamical systems Year: (2025)
Ref_id:b62 Title: A data-driven framework for koopman semigroup estimation in stochastic dynamical systems Year: (2025)
Ref_id:b63 Title: Learning koopman representations for complex dynamics with spectral residuals Year: (2025)
Ref_id:b64 Title: Tensor-var: Efficient four-dimensional variational data assimilation Year: (2025)
Ref_id:b65 Title: Information theory and network coding Year: (2008)
Ref_id:b66 Title: Cfdbench: A large-scale benchmark for machine learning methods in fluid dynamics Year: (2023)
