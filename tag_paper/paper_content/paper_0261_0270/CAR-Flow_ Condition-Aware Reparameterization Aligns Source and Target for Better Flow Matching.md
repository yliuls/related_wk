Title: CAR-Flow: Condition-Aware Reparameterization Aligns Source and Target for Better Flow Matching
Abstract: Conditional generative modeling aims to learn a conditional data distribution from samples containing data-condition pairs. For this, diffusion and flow-based methods have attained compelling results. These methods use a learned (flow) model to transport an initial standard Gaussian noise that ignores the condition to the conditional data distribution. The model is hence required to learn both mass transport and conditional injection. To ease the demand on the model, we propose Condition-Aware Reparameterization for Flow Matching (CAR-Flow) -a lightweight, learned shift that conditions the source, the target, or both distributions. By relocating these distributions, CAR-Flow shortens the probability path the model must learn, leading to faster training in practice. On low-dimensional synthetic data, we visualize and quantify the effects of CAR-Flow. On higher-dimensional natural image data (ImageNet-256), equipping SiT-XL/2 with CAR-Flow reduces FID from 2.07 to 1.68, while introducing less than 0.6% additional parameters.

Section: Introduction
Conditional generative models enable to draw samples conditioned on an external variable-for example, a class label, a text caption, or a semantic mask. They are a key technology that has advanced significantly in the last decade, from variational auto-encoders (VAEs) [Kingma and Welling, 2014] and generative adversarial nets [Goodfellow et al., 2014] to diffusion [Ho et al., 2020, Song et al., 2021a,b, Dhariwal and Nichol, 2021, Peebles and Xie, 2023, Rombach et al., 2022, Nichol and Dhariwal, 2021] and flow-matching [Ma et al., 2024, Liu et al., 2023, Lipman et al., 2023, Albergo and Vanden-Eijnden, 2023, Albergo et al., 2023].
State-of-the-art diffusion and flow-matching frameworks accomplish conditional generation by using a trained deep net to trace a probability path that progressively transforms samples from a simple source distribution into the rich, condition-dependent target distribution. A popular choice for the source distribution is a single, condition-agnostic standard Gaussian. Consequently, the conditioning signal enters only through the network itself: in flow matching, for instance, the model predicts a velocity field where the condition is commonly incorporated via embeddings or adaptive normalization layers. Although this strategy has enabled impressive results, this design forces the network to shoulder two tasks simultaneously-(i) transporting probability mass to the correct region of the data manifold, and (ii) encoding the semantic meaning of the condition. Because different conditions often occupy distant parts of that manifold, the dual burden stretches the learned trajectory, slows convergence, and can impair sample quality and diversity. Illustration of the push-forward under standard conditional flow matching (direct mapping x 0 → x 1 ) versus our Condition-Aware Reparameterization (CAR-Flow) chain (x 0 → z 0 → z 1 → x 1 ). In the standard setting, a condition-agnostic prior sample (red) is carried by the network's velocity field directly to each condition-dependent data manifold (blue), forcing it to juggle long-range transport and semantic injection at once. CAR-Flow, in contrast, employs lightweight source distribution map f (•, y) and target distribution map g(•, y) to align the source and target distributions to relieve the network of unnecessary transport. During sampling, x 1 is obtained via the (approximate) inverse map g -1 (•, y).
In this paper, we alleviate this burden via a condition-dependent mapping of the source and target distributions. First, we endow the source distribution with condition-awareness through a lightweight map, i.e., the source distribution map. Hence, every condition has its own source distribution.
The same idea can be mirrored to the target distribution using another lightweight map, i.e., the target distribution map, that we require to be approximately invertible to keep sampling tractable. The resulting push-forward chain is illustrated in Figure 1. Intuitively, the target distribution map and its approximated inverse correspond to the encoder and decoder, commonly used in latent diffusion pipelines [Rombach et al., 2022]. Differently, here, the target distribution map is explicitly conditional, aligning our formulation with recent latent-space conditioning approaches that embed semantic information directly into the latent representation [Wu et al., 2024, Leng et al., 2025, Yao et al., 2025].
Although fully general maps provide maximal flexibility, this freedom conceals a critical failure mode: the flow-matching objective admits zero-cost solutions. The predicted data distribution then collapses to a single mode. A comparable drop in generation quality has been observed in recent work when a variational auto-encoder (VAE) is naively fine-tuned end-to-end with a diffusion model [Leng et al., 2025]. We formalize these collapse routes and later verify them experimentally.
To preclude this trivial solution, we impose the simplest effective constraint-shift-only conditioning.
Concretely, we reparameterize the maps to only translate. Relocating the start and/or end points while leaving scales untouched removes all trivial solutions, and still shortens the residual probability path. Note that volume-preserving maps are also possible. We refer to the resulting framework as condition-aware reparameterization for flow matching (CAR-Flow).
CAR-Flow can be applied to the source distribution, the target distribution, or both. Each variant shapes the learned path differently, and we find the joint version to perform best in practice. For both low-dimensional synthetic benchmarks and high-dimensional natural image (ImageNet-256) data, CAR-Flow consistently improves performance: augmenting the strong SiT-XL/2 baseline [Ma et al., 2024] with CAR-Flow reduces FID from 2.07 to 1.68 while adding fewer than 0.6% additional parameters.
Our contributions are as follows:
1. We introduce CAR-Flow, a simple yet powerful shift-only mapping that aligns the source, the target, or both distributions with the conditioning variable. CAR-Flow relieves the velocity network of unnecessary transport while adding negligible computational overhead.
2. We provide a theoretical analysis that uncovers zero-cost solutions present under unrestricted reparameterization and prove that they render the velocity collapse, thereby explaining the empirical failures of naïve end-to-end VAE-diffusion training.
this section cite: ['b7', 'b5', 'b6', 'b14', 'b17', 'b11', 'b10', 'b9', 'b17', 'b23', 'b8', 'b24', 'b8', 'b11']

Section: We validate CAR-Flow on low-dimensional synthetic data and the high-dimensional
ImageNet-256 benchmark, consistently outperforming the standard rectified flow baseline.
this section cite: []

Section: Background and Preliminaries
We start with a brief review of the standard flow-matching formulation.
this section cite: []

Section: Conditional Generation and Probability Paths
Conditional generation seeks to sample from a conditional data distribution x 1 ∼ p data x (• | y), where y is a conditioning variable, e.g., a class label or a text prompt. Diffusion and flow-based models tackle this problem by simulating a differential equation that traces a probability-density path p t (x 1 | x 0 , y), gradually transporting samples from a simple source distribution x 0 ∼ p init
x into the target conditional distribution. A standard choice for the source distribution is the isotropic Gaussian p init
x = N (0, I d ). The stochastic trajectory (X t ) 0≤t≤1 follows the SDE
dX t = u t (X t , y) + σ 2 t 2 ∇log p t (X t | y) dt + σ t dW t ,(1)
where u t is the drift field, σ t is the diffusion coefficient, and W t is a standard Wiener process. The term ∇ log p t (X t | y) denotes the score function, which can be written in terms of the drift field u t [Ma et al., 2024]. In the deterministic limit σ t = 0, this SDE reduces to the ODE dX t = u t (X t , y) dt.
this section cite: ['b11']

Section: Gaussian Probability Paths
A convenient instantiation is the Gaussian path. Let α t and β t be two continuously differentiable, monotonic noise scheduling functions satisfying the boundary conditions α 0 = β 1 = 0 and α 1 = β 0 = 1. At time t, the conditional distribution is
p t (• | x 1 , y) = N α t x 1 , β 2 t I d | y ,(2)
whose endpoints are p 0 (
• | x 1 , y) = N (0, I d ) and p 1 (• | x 1 , y) = δ x1|y .
Here δ denotes the Dirac delta "distribution". Along this path, the state evolves by the interpolant
x t = β t x 0 + α t x 1 , with velocity field u t = βt x 0 + αt x 1 ,(3)
where overdots denote time derivatives.
this section cite: []

Section: Conditional Flow Matching and Sampling
Conditional flow matching trains a neural velocity field v θ (x, t, y) to approximate the true velocity u t specified in Eq. ( 3). The commonly employed objective is
L(θ) = E y∼p Y , t∼p T x0∼p init x , x1∼p data x v θ β t x 0 + α t x 1 , t, y -βt x 0 + αt x 1 2 ,(4)
where p Y is the marginal distribution over conditions y, and p T is a time density on [0, 1].
After training, one draws x 1 ∼ p data x by numerically integrating Eq. ( 1) from t = 0 to t = 1 with ût = v θ (x t , t, y). A solver-agnostic procedure is outlined in Algorithm 1.
Algorithm 1 Sampling via conditional flow matching (standard Gaussian source) Require: trained network v θ ; diffusion schedule {σ t } t∈[0,1] ; number of steps N 1: Sample y ∼ p Y 2: Sample x 0 ∼ N (0, I d ) 3: ∆t ← 1/N ; x ← x 0 4: for k = 0 to N -1 do 5: t ← k ∆t 6: u ← v θ (x, t, y) /* drift */ 7: Integrate SDE in Eq. (1) over [t, t + ∆t] with (u, σ t ) /* e.g., Euler-Maruyama */ 8: end for 9: return x /* sample at t = 1 conditioned on y */
this section cite: []

Section: Condition-Aware Reparameterization
As detailed in Section 2, standard conditional flow matching initiates the probability path from a condition-agnostic prior, typically a standard Gaussian. Consequently, the velocity network v θ (x t , t, y) must simultaneously learn two intertwined tasks-transporting mass and encoding semantics, imposing a dual burden. To alleviate this, we reparameterize the source and/or target distributions via explicit functions of the condition y. In this section, we first reformulate conditional flow-matching using general reparameterizations (Section 3.1). We then show that allowing arbitrary reparameterizations leads to trivial zero-cost minima, collapsing distributions to a single mode (Section 3.2). To both shorten the transport path and eliminate these trivial solutions, we propose Condition-Aware Reparameterization for Flow Matching (CAR-Flow) (Section 3.3).
this section cite: []

Section: General Reparameterization
Formally, instead of drawing an initial value x 0 directly from the fixed source distribution p init x , we first apply a condition-dependent source distribution map f : R n × Y → R m , (x, y) → z such that
z 0 = f (x 0 , y), x 0 ∼ p init x .(5)
We hence obtain a sample from the modified source distribution p init z via the push-forward
z 0 ∼ p init z (• | y) = f (•, y) # p init x .
Similarly, we define the target distribution map g : R n × Y → R m , (x, y) → z such that
z 1 = g(x 1 , y), x 1 ∼ p data x (• | y).(6)
We characterize a sample from the target distribution in latent space p data z via the push-forward
z 1 ∼ p data z (• | y) = g(•, y) # p data x (• | y).
Critically, to make sampling tractable, g must be approximately invertible-i.e., we assume there exists g -1 such that x 1 ≈ g -1 (z 1 , y).
It is worth noting that our reparameterization framework subsumes both the classic VAE-based latent diffusion model [Rombach et al., 2022] and more recent efforts to inject semantic awareness directly into the latent space, e.g., work by Leng et al. [2025]. To recover the standard latent diffusion model, we leave the source untouched, i.e., f (x 0 , y) = x 0 , and set g(x 1 , y) ≜ E(x 1 ), g -1 (z 1 , y) ≜ D(z 1 ), where E, D are the encoder and decoder of a VAE trained via the ELBO (reconstruction loss plus KL divergence) without explicit semantic supervision. In contrast, more recent works [Yu et al., 2025, Leng et al., 2025, Yao et al., 2025] augment VAE training with a semantic alignment loss-often using features from a pretrained DINO-V2 network-so that the encoder and decoder effectively depend on semantic embeddings of y. Both paradigms arise naturally as special cases of our general push-forward reparameterization framework. Moreover, our framework readily accommodates further reparameterization of any pretrained VAE. Concretely, one can introduce an invertible map g ′ : R m × Y → R m and set
g(x 1 , y) ≜ g ′ E(x 1 ), y , g -1 (z 1 , y) ≜ D (g ′ ) -1 (z 1 , y) ,(7)
thus embedding semantic conditioning directly into both the encoder and decoder.
Flow-matching loss and sampling under reparameterization. When p init z (• | y) is Gaussian, the probability path z 0 → z 1 remains Gaussian under the interpolation z t = β t z 0 + α t z 1 , u t = Algorithm 2 Sampling via conditional flow matching (reparameterized) Require: trained v θ ; diffusion schedule {σ t }; steps N 1: Sample y ∼ p Y 2: Sample x 0 ∼ N (0, I d ) and set z ← f (x 0 , y) 3: ∆t ← 1/N 4: for k = 0 to N -1 do 5:
t ← k ∆t 6:
u ← v θ (z, t, y)
7: Integrate SDE in Eq.(8) over [t, t + ∆t] with (u, σ t ) 8: end for 9: x ← g -1 (z, y) /* invertible g needed here */ 10: return x /* sample at t = 1 conditioned on y */ βt z 0 + αt z 1 . The evolution of Z t then follows the SDE
dZ t = u t (Z t , y) + σ 2 t 2 ∇log p t (Z t | y) dt + σ t dW t ,(8)
where p t is the Gaussian path in z-space. Note that the conditional score functions generally differ:
∇ log p t (Z t | y) ̸ = ∇ log p t (X t | y), unless f (x 0 , y) = x 0 (see Appendix A for details).
The resulting flow-matching loss becomes
L(θ) = E y∼p Y , t∼p T x0∼p init x , x1∼p data x v θ β t z 0 + α t z 1 , t, y -βt z 0 + αt z 1 2 . (9
)
Once trained, sampling proceeds by first drawing an initial starting point x 0 ∼ p init
x and computing the condition-aware latent z 0 = f (x 0 , y). We then integrate the reparameterized SDE in Eq. ( 8) from t = 0 to t = 1 to obtain z 1 , and finally map back to data space via x 1 = g -1 (z 1 , y). Algorithm 2 provides a pseudo-code that summarizes the process.
this section cite: ['b17', 'b8', 'b25', 'b8', 'b24']

Section: Mode Collapse under Unrestricted Reparameterization
Under the general reparameterization framework of Eq. ( 9), the maps f and g enjoy full flexibility. Unfortunately, this expressivity also admits several trivial zero-cost minima: analytically, the loss can be driven to zero by degenerate shift solutions, resulting in a collapse of the generative distribution to a single/improper mode. Claim 1. Let f, g : R n × Y → R m be arbitrary maps. If any of the following holds (for some functions c(y) ∈ R m or scalars k(y) ∈ R):
(i) Constant source: f (x 0 , y) = c(y) for all x 0 , (ii) Constant target: g(x 1 , y) = c(y) for all x 1 , (iii) Unbounded source scale: ∥f (x 0 , y)∥ → ∞, (iv) Unbounded target scale: ∥g(x 1 , y)∥ → ∞, (v) Proportional collapse: f (x 0 , y) = k(y) g(x 1 , y),
then the flow-matching loss in Eq. ( 9) admits a trivial minimum in which the optimal velocity field takes the form v θ (z t , t, y) = γ(t, y) z t + η(t, y), causing all probability mass to collapse to a single/improper mode.
The proof of Claim 1, together with closed-form expressions for γ(t, y) and η(t, y), is deferred to the Appendix B.
To illustrate the collapse concretely, we consider the linear schedule β t = 1 -t, α t = t and an affine reparameterization
f (x 0 , y) = σ 0 (y) x 0 + µ 0 (y), g(x 1 , y) = σ 1 (y) x 1 + µ 1 (y),(10)
analogous to the standard Gaussian trick. Table 1 summarizes each collapse mode, listing the maps, the closed-form collapsed velocity v * (z, t, y) = γ(t, y) z + η(t, y), and the resulting push-forward distributions p init z and p data z .
Table 1: Zero-cost collapse modes under the linear schedule α t = t, β t = 1 -t, using affine maps f (x 0 ) = σ 0 x 0 + µ 0 and g(x 1 ) = σ 1 x 1 + µ 1 . For brevity, we omit explicit y-dependence.
Case f g γ(t) η(t) v * (z t , t) p init z p data z (i) µ 0 arbitrary 1 t -1 t µ 0 zt-µ0 t δ µ0 - (ii) arbitrary µ 1 -1 1-t 1 1-t µ 1 µ1-zt 1-t - δ µ1 (iii) ∞ arbitrary -1 1-t 0 -zt 1-t Uniform(R d ) - (iv) arbitrary ∞ 1 t 0 zt t - Uniform(R d ) (v) µ 0 kµ 0 0 (k -1)µ 0 (k -1)µ 0 δ µ0 δ µ1
We note that in the constant-map scenarios (cases (i) and (ii)), the affine transforms f or g effectively collapse the variance of p init z or p data z to zero. In particular, case (ii) mirrors the finding in REPA-E (Table 1) [Leng et al., 2025], where end-to-end VAE-diffusion tuning favored a simpler latent space with reduced variance. In contrast, the unbounded-scale modes (iii) and (iv) push the distributions toward an improper uniform distribution, eliminating any meaningful localization. Proportional collapse (v) yields a degenerate flow with p init z , p data z , and the velocity field being constant.
Empirically, we do not observe cases (iii)-(v). In particular, cases (iii) and (iv) correspond to unbounded collapse solutions: when the scale of either the source or target distribution tends to infinity, the counterpart distribution collapses relative to it, yielding zero cost. These solutions require unbounded weights and are therefore unstable-any perturbation pushes the optimization back toward cases (i) or (ii). Case (v), in contrast, assumes exact proportionality between maps f and g, which cannot occur under independently sampled inputs except in trivial constant-map settings that reduce to cases (i) or (ii). In practice, the optimizer thus follows the easiest "shortcut"-the constant-map collapse-since setting f or g to a fixed constant immediately zeroes the loss. In Section 4, we empirically verify not only that such constant-map collapses occur in real models, but also how these mode-collapse behaviors manifest in practice.
this section cite: ['b8']

Section: Shift-Only Condition-Aware Reparameterization
To eliminate the trivial zero-cost solutions while retaining the benefits of condition-aware reparameterization, we restrict both reparameterizations to additive shifts only, while noting that volumepreserving maps are possible. For simplicity we use
f (x 0 , y) = x 0 + µ 0 (y), g(x 1 , y) = x 1 + µ 1 (y).(11)
Here, µ 0 , µ 1 : Y → R m are lightweight, learnable, condition-dependent shifts. By preserving scale, we block every collapse mode given in Claim 1, yet we still move z 0 and z 1 closer in latent space. In particular, when used with a pretrained VAE, Eq. ( 7) becomes
g(x 1 , y) ≜ E(x 1 ) + µ 0 (y), g -1 (z 1 , y) ≜ D z 1 -µ 1 (y) ,(12)
CAR-Flow admits three natural variants:
• Source-only: µ 1 ≡ 0, so only the source is shifted.
• Target-only: µ 0 ≡ 0, so only the target is shifted.
• Joint: both µ 0 and µ 1 active.
Each variant alters the probability path
z t = β t (x 0 + µ 0 (y)) + α t (x 1 + µ 1 (y)) in a distinct way.
Note that shifting the source cannot be replicated by an opposite shift on the target. In fact, no nonzero constant shift on the source can ever be matched by a constant shift on the target-except in the trivial case: Claim 2. Shifting the source by µ 0 is equivalent to shifting the target by µ 1 , if and only if µ 0 = µ 1 = 0.
Proof. With a source-only shift
(µ 0 , µ 1 ) = (µ 0 , 0), the interpolant is z (s) t = β t (x 0 + µ 0 ) + α t x 1 . With a target shift (µ 0 , µ 1 ) = (0, µ 1 ), it is z (t) t = β t x 0 + α t (x 1 + µ 1 )
. Equating these gives for ∀t, β t µ 0 = α t µ 1 . Since µ 0 and µ 1 are t-independent functions of y, it forces µ 0 = µ 1 = 0.
For Source-only, the interpolant simplifies to z t = x t + β t µ 0 (y). When µ 0 (y) ≈ µ 0 (y ′ ) for two conditions y, y ′ , their paths share a common starting region, simplifying the early-time flow. For Target-only, the trajectory becomes z t = x t + α t µ 1 (y). When µ 1 (y) ≈ µ 1 (y ′ ), the network only needs to learn a shared "landing zone", easing the late-time flow. For Joint, we have z t = x t + µ t (y) where µ t (y) = β t µ 0 (y) + α t µ 1 (y), so the time-varying shift aligns both endpoints, minimizing the overall transport distance and reducing the burden on v θ throughout the entire trajectory.
Empirically, we find that the joint CAR-Flow variant-allowing both µ 0 and µ 1 to adapt-yields the largest improvements in convergence speed and sample fidelity (see Section 4).
this section cite: []

Section: Experiments
In this section, we evaluate the efficacy of the Condition-Aware Reparameterization for Flow Matching (CAR-Flow) under a linear noise schedule β t = 1 -t, α t = t and compare to classic rectified flow.
All experiments are done using the axlearn framework.foot_0 Detailed implementation settings can be found in Appendix C.
this section cite: []

Section: Synthetic Data
For our synthetic-data experiments, we consider a one-dimensional task where the source distribution is N (0, 1) and the target distribution is a two-class Gaussian mixture, with class A data distribution N (-1.5, 0.2 2 ) and class B data distribution N (+1.5, 0.2 2 ).
We encode x t , y, and t using sinusoidal embeddings before feeding them to the network. In the baseline rectified-flow model, these embeddings are concatenated and passed through a three-layer MLP (1,993 parameters total) to predict the velocity field v θ (x t , y, t). Training uses the loss in Eq. ( 4), and sampling follows Algorithm 1. For CAR-Flow, we augment this backbone with two lightweight linear layers that map the class embedding to the shifts µ 0 (y) and/or µ 1 (y), each adding only 9 parameters. In the source-only variant we predict µ 0 (with µ 1 = 0); in the target-only variant we predict µ 1 (with µ 0 = 0); and in the joint variant we predict both. These shifts are applied via Eq. ( 11), training proceeds with the loss in Eq. ( 9), and sampling uses Algorithm 2. All models-baseline and CAR variants-set σ t = 0 (reducing the SDE to an ODE) and employ a 50-step Euler integrator for sampling. To ensure robustness, each configuration is run three times, and we report the average performance, noting that variance across runs is negligible.
Figure 2 shows how CAR-Flow alters the learned flow. In Figure 2a, the baseline must both transport mass and encode class information, yielding the longest paths. The source-shift variant in Figure Table 2: Average trajectory length ∥z 0 → z 1 ∥ with 2σ error bounds. Baseline Source-only CAR-Flow Target-only CAR-Flow Joint CAR-Flow Length 1.5355 ± 0.0024 0.7432 ± 0.0019 0.7129 ± 0.0010 0.7121 ± 0.0011 0 10 20 30 40 50 Training Steps (K) 10 1 10 0 Wasserstein Distance Baseline Source-only Target-only Joint (a) Wasserstein distance 1 0 1 Shift 0 Source-only (A) Source-only (B) Joint (B) Joint (A) 0 10 20 30 40 50 Training Steps (K) 1 0 1 Shift 1 Target-only (A) Target-only (B) Joint (B) Joint (A) (b) Learned µ0 and µ1 shifts Figure 3: Comparison of convergence and learned shifts. (a) shows the Wasserstein distance between predicted and ground-truth distributions in symlog-scale. Joint CAR-Flow achieves both the fastest convergence (b) plots the evolution of the learned shifts µ 0 (top) and µ 1 (bottom) for two classes.
cleanly relocates each class's start, the target-shift variant in Figure 2c leaves the source untouched and merges endpoints, and the joint variant in 2d aligns both start and end with minimal source shift-producing the shortest trajectories (Table 2).
Figure 3a summarizes convergence measured by Wasserstein distance over training: joint CAR-Flow converges fastest and reaches the lowest error, followed by source and target-only, all outperforming the baseline. Figure 3b traces the learned µ 0 and µ 1 for each class and variant: joint CAR-Flow yields the most moderate, balanced shifts, explaining its superior convergence and flow quality.
this section cite: []

Section: Mode Collapse.
To empirically validate the mode-collapse analysis described in Section 3.2, we reuse the setup but now allow both shift and scale parameters to be learned simultaneously (Eq.( 10)). We train two separate models: one reparameterizing the source distribution with learned parameters (µ 0 , σ 0 ), and another morphing the target distribution with (µ 1 , σ 1 ). Results are presented in Figure 4. Specifically, Figure 4a shows the rapid evolution of the learned standard deviations σ, clearly indicating the network quickly discovers a "shortcut" solution by shrinking σ to zero. Figure 4b plots the expected norm gap E∥v θ -v * ∥ 2 , demonstrating convergence to zero, which indicates the network's velocity prediction aligns closely with the analytic zero-cost solutions derived in Table 1. Moreover, unrestricted parameterization of the source distribution triggers mode-collapse case (i) as described in Claim 1, where the flow degenerates to predicting the class-wise mean of the target (see Figure 4c). Conversely, unrestricted parameterization of the target collapses the distribution nearly to a constant, and consequently, the predicted x 1 degenerates into an improper uniform distribution due to σ 1 → 0, as illustrated in Figure 4d.
this section cite: []

Section: ImageNet
To benchmark on a high-dimensional, large-scale dataset, we conduct experiments on ImageNet 256 × 256 data using v6e-256 TPUs. Our baseline is SiT-XL/2 [Ma et al., 2024], re-implemented in JAX [Bradbury et al., 2018]; we strictly follow the original training recipe from the open-source SiT repository to replicate the results reported in the paper. For our CAR variants, we apply the shift-only reparameterization to the sd-vae-ft-ema VAE backbone used by SiT (see Eq. ( 11)-( 12)). We introduce two lightweight convolutional networks (≈ 2.3M parameters each) to predict µ 0 and µ 1 from the class embeddings, projecting them into the latent space. All models are sampled using the Heun SDE solver with 250 NFEs.
Table 3 presents the quantitative results. Augmenting SiT-XL/2 with CAR-Flow consistently outperforms the baseline across all variants. In particular, the joint-shift variant achieves the best result, reducing FID from 2.07 to 1.68 while adding fewer than 0.6% parameters. These results underscore the importance of explicitly conditioning the source and target distributions: simple shift reparam-  eterization not only improves sample fidelity but does so with minimal computational overhead, facilitating easy integration into existing large-scale generative frameworks.
Table 3 presents the quantitative results. Augmenting SiT-XL/2 with CAR-Flow consistently outperforms the baseline across all variants. In particular, the joint-shift variant achieves the best result, reducing FID from 2.07 to 1.68 while adding fewer than 0.6% parameters. Beyond final performance, CAR-Flow also accelerates optimization: our convergence analysis on ImageNet-256 shows that all CAR-Flow variants consistently reduce FID faster than the baseline across training steps (see Fig. 5). These results underscore the importance of explicitly conditioning the source and target distributions: simple shift reparameterization not only improves sample fidelity but does so with minimal computational overhead, facilitating easy integration into existing large-scale generative frameworks.
this section cite: ['b11', 'b2']

Section: Related Work
Generative modeling has advanced significantly in the last decade from variational auto-encoders (VAEs) [Kingma and Welling, 2014], Generative Adversarial Nets [Goodfellow et al., 2014], and normalizing flows [Rezende and Mohamed, 2015]. More recently, score matching [Song andErmon, 2019, Song et al., 2020], diffusion models [Ho et al., 2020], and flow matching [Liu et al., 2023, Lipman et al., 2023, Albergo and Vanden-Eijnden, 2023, Albergo et al., 2023] was introduced. The latter three frameworks are related: sampling at test-time can be viewed as numerically solving a transport (ordinary) differential equation by integrating along a learned velocity field from the source distribution at time zero to the target distribution at time one.
For learning the velocity field, various approaches to interpolate between samples from the source distribution and the target distribution have been discussed [Lipman et al., 2023, Liu et Tong et al., 2024]. Among those, rectified flow matching was shown to lead to compelling results on large-scale data [Ma et al., 2024, Esser et al., 2024].
For use on large-scale data, flow matching is typically formulated in latent space by compressing data via the encoder of a pre-trained and frozen VAE [Rombach et al., 2022]. These mappings differ from the discussed CAR-Flow, as they are typically independent of the conditioning variable. As mentioned before, CAR-Flow can be applied on top of pre-trained and frozen projections on the latent space.
More recently, Yu et al. [2025], Yao et al. [2025] proposed to align representations within deep nets that model the velocity to visual representations from vision foundation models. Specifically, Yu et al. [2025] align early layer features of DiT [Peebles and Xie, 2023] and SiT [Ma et al., 2024] models with representations extracted from DINOv2 [Oquab et al., 2024] and CLIP [Radford et al., 2021]. In contrast, Yao et al. [2025] aligns the latent space of a VAE with representations from pre-trained vision foundation models, which are then frozen for diffusion model training. These approaches differ from our approach, which learns to transform the source and target distributions rather than encouraging feature alignment.
Most related to our work is the recently introduced REPA-E [Leng et al., 2025]. In REPA-E, Leng et al. [2025] study end-to-end training of diffusion models and VAE encoders/decoders, which map data to/from a latent space. Differently, in this paper, we formalize failure modes reported by Leng et al. [2025] and identify them as trivial solutions that arise when jointly training a flow matching model and a target distribution mapping. We further introduce a source distribution mapping. Finally, we impose simple restrictions that preclude those trivial solutions.
this section cite: ['b7', 'b5', 'b16', 'b6', 'b10', 'b9', 'b22', 'b11', 'b4', 'b17', 'b25', 'b24', 'b25', 'b14', 'b11', 'b13', 'b15', 'b24', 'b8', 'b8', 'b8']

Section: Conclusion
We propose and study condition-aware reparameterization for flow matching (CAR-Flow), which aligns the source and target distributions in flow-matching models. We find CAR-Flow to alleviate the burden of classic flow-matching, where a model simultaneously transports probability mass to the correct region of the data manifold, while also encoding the semantic meaning of the condition.
Limitations and broader impact. This work characterizes the failure modes when jointly training a flow matching model, a source distribution mapping, and a target distribution mapping, and identifies them as trivial solutions of the objective. We further study the simplest effective approach to avoid these trivial solutions. While we find this simple approach to lead to compelling results, we think more general mappings will likely improve results even further. We leave the identification of more general suitable mappings to future work.
Improving the expressivity of generative models has a significant broader impact. On the positive side, modeling complex distributions more easily saves resources and enables novel applications. On the negative side, efficient generative modeling can be abused to spread misinformation more easily.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b1 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b2 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b3 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b4 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b5 Title: Generative adversarial nets Year: (2014)
Ref_id:b6 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b7 Title: Auto-Encoding Variational Bayes Year: (2014)
Ref_id:b8 Title: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers Year: (2025)
Ref_id:b9 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b10 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b11 Title: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b12 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b13 Title: Learning robust visual features without supervision Year: (2024)
Ref_id:b14 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b15 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b16 Title: Variational inference with normalizing flows Year: (2015)
Ref_id:b17 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b18 Title: Denoising diffusion implicit models Year: (2021)
Ref_id:b19 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b20 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b21 Title: Score-based generative modeling through stochastic differential equations Year: ()
Ref_id:b22 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2024)
Ref_id:b23 Title: Vilau: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b24 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b25 Title: Representation alignment for generation: Training diffusion transformers is easier than you think Year: (2025)
