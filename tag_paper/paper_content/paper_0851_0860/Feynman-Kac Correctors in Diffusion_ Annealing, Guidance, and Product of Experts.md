Title: Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts
Abstract: While score-based generative models are the model of choice across diverse domains, there are limited tools available for controlling inferencetime behavior in a principled manner, e.g. for composing multiple pretrained models. Existing classifier-free guidance methods use a simple heuristic to mix conditional and unconditional scores to approximately sample from conditional distributions. However, such methods do not approximate the intermediate distributions, necessitating additional 'corrector' steps. In this work, we provide an efficient and principled method for sampling from a sequence of annealed, geometric-averaged, or product distributions derived from pretrained score-based models. We derive a weighted simulation scheme which we call FEYNMAN-KAC CORRECTORS (FKCs) based on the celebrated Feynman-Kac formula by carefully accounting for terms in the appropriate partial differential equations (PDEs). To simulate these PDEs, we propose Sequential Monte Carlo (SMC) resampling algorithms that leverage inference-time scaling to improve sampling quality. We empirically demonstrate the utility of our methods by proposing amortized sampling via inference-time temperature annealing, improving multi-objective molecule generation using pretrained models, and improving classifierfree guidance for text-to-image generation. Our code is available at https://github.com/  martaskrt/fkc-diffusion.

Section: 
Figure 1. FEYNMAN-KAC CORRECTOR Inference for annealed p t,β (x) ∝ qt(x) β=10 and product pt(x) ∝ q 1 t (x)q 2 t (x) densities.
this section cite: []

Section: Introduction
Score-based generative models, also known as diffusion models, have emerged as the model of choice across diverse generative tasks such as image generation, natural language, and protein simulation (Saharia et al., 2022;Sahoo et al., 2024;Abramson et al., 2024). These models leverage the ability to estimate scores of the sequence of noise-corrupted distributions and then use the learned scores to reverse the corruption process enabling high-quality generation. Thus, diffusion models aim to produce new samples from the same distribution as the training data.
However, the classical paradigm of generative modeling as the problem of reproducing the training data distribution becomes less relevant for many applications including drug discovery and text-to-image generation. In practice, generative models demonstrate the best performance when tailored to specific needs at inference time. For instance, linear combinations of scores allow for concept composition (Liu et al., 2022) or for increasing image-prompt consistency as in classifier-free guidance (CFG) (Ho & Salimans, 2021). However, by modifying the scores, one loses control over the marginal distributions of the generated samples. Various approaches from the Monte Carlo sampling literature have been adapted to 'correct' samples along a trajectory to more closely match the prescribed intermediate distributions.
Assuming access to an exact score, additional Langevin corrector steps with the desired invariant distribution can be applied with additional simulation steps as the only practical overhead (Song et al., 2021;Bradley & Nakkiran, 2024). However, these corrector schemes are only exact in the limit of infinite intermediate steps. Accept-reject or Sequential Monte Carlo techniques may be used when the score is parameterized through a scalar energy function (Du et al., 2023;Phillips et al., 2024), although these parameterizations require extra computation during training and may sacrifice expressivity in practice (Salimans & Ho, 2021;Thornton et al., 2025). While methods for sampling from mixtures or equiprobable regions of diffusion models have been proposed (Skreta et al., 2025), general solutions to accurately sample from combinations or temperings of flexibly-parameterized diffusion models with limited computational overhead remain elusive.
To address these challenges, we introduce FEYNMAN-KAC CORRECTOR (FKCs), which enable efficient and principled sampling from a sequence of annealed, geometric-averaged, or product distributions derived from pretrained diffusion models. To develop FEYNMAN-KAC CORRECTORS and test their efficacy, we make the following contributions:
• We propose a flexible recipe for constructing weighted stochastic differential equations (SDEs), which account for additional terms appearing when manipulating the distribution of generated samples.
• As our primary examples, we derive the correction terms for multiple heuristic schemes commonly used to approximate annealed, product, or geometric averaged distributions, including CFG (Sec. 3).
• To simulate these weighted SDEs, we propose a family of Sequential Monte Carlo (SMC) resampling schemes, which 'correct' a batch of simulated samples to closely approximate the intermediate target distributions (Sec. 4).
• For the problem of sampling from an unnormalized density, we demonstrate that FKC allows for sampling from a variety of temperatures without retraining (Sec. 5.2). Moreover, we demonstrate that a high-temperature learning, low-temperature inference scheme can be more efficient than the notoriously difficult task of directly training a sampler at a lower temperature.
• For pretrained diffusion models we demonstrate that adding FKC terms enhances compositional generation of molecules with multiple properties (Sec. 5.3) and classifier-free guidance for image generation (Sec. 5.1).
this section cite: ['b0', 'b31', 'b9', 'b23', 'b31', 'b52']

Section: Background

this section cite: []

Section: Diffusion Models
Generative modeling via diffusion models can be formulated as the simulation of the Stochastic Differential Equation (SDE) corresponding to the reverse-time process.
In particular, during training, one gradually destroys samples from the data-distribution p data (x) by simulating the following noising SDE:
dx τ = f τ (x τ )dτ + σ τ dW τ , x τ =0 ∼ p data (x) ,(1)
where f τ (x τ ) is usually some linear drift function f τ (x τ ) = α τ x τ , σ τ defines the scale of noise through time, and dW τ is the standard Wiener process. The drift f τ and the diffusion coefficient σ τ are chosen so the final density is close to the standard normal distribution p τ =1 ≈ N (0, I d ).
The generation process then can be defined as the family of denoising SDEs in the opposite time direction (t = 1 -τ ), dx t = -f t (x t ) + σ 2 t ∇ log p t (x t ) dt + σ t dW t , (2) where p t = p 1-τ is the density of the marginals induced by the noising process in Eq. ( 1); hence, the process starts with x 0 ∼ N (x | 0, I d ). By training a model of the score functions ∇ log p t (•), one can generate new samples from p data (x) using Eq. ( 2) (Song et al., 2021).
this section cite: []

Section: Feynman-Kac PDEs
While Eq. ( 2) describes a procedure for simulating individual particles, we can also derive Partial Differential Equations (PDEs) which describe the time-evolution of the density of samples p t (x) under this SDE. We begin by describing the relevant equations for the standard SDE case.
(1) Continuity Equation, which describes how the density changes when the samples move in space according to a flow or ODE with drift v t dxt = vt(xt)dt =⇒ ∂p ode t (x) ∂t = -∇, p ode t (x)vt(x) . (3) where p ode t indicates the evolution only according to a flow. (2) Diffusion Equation, which describes the change of the density for the pure Brownian motion with coefficient σ t , dx t = σ t dW t =⇒ ∂p diff t (x) ∂t = σ 2 t 2 ∆p diff t (x) . (4) where p diff t denotes evolution due to the diffusion term only. The SDE in Eq. (2) can be viewed as the composition of a flow and diffusion terms, where the corresponding Fokker-Planck PDE describes the combined evolution ∂p sde t (x) ∂t = -∇, p sde t (x)v t (x) + σ 2 t 2 ∆p sde t (x). (5)
However, our main focus in this work will be to study a third type of PDE, which will yield weighted SDEs that we eventually use to simulate a sequence of marginals other those the forward noising process p 1-τ (Sec. 3).
(3) Reweighting Equation, which describes the change of density when samples have time-dependent log-weights w t which are updated based on the positions of samples x t , dw t = ḡt (x t )dt =⇒ ∂p w t (x) ∂t = ḡt (x)p w t (x) ,
where ḡt (x) = g t (x) -g t (x)p w t (x)dx
where the last equation guarantees the conservation of the normalization constant, i.e. dx ḡt (x)p w t (x) = 0. Feynman-Kac Formula We now focus on the combination of all three components to describe the Feynman-Kac PDE,
∂p FK t (x) ∂t = -∇, p FK t (x)v t (x) + σ 2 t 2 ∆p FK t (x) + ḡt (x)p FK t (x) ,(7)
where to sample from p FK t (x), one first has to sample x t via the following SDE
dx t = v t (x t )dt + σ t dW t , dw t = ḡt (x t )dt ,(8)
and then reweight the obtained samples using w t . Thus, p FK t (x) reflects the density of weighted samples, which differs from the density p sde t (x) obtained via the Fokker-Planck PDE in Eq. ( 5) due to the addition of reweighting terms.
In practice, we can account for this difference by sampling
i ∼ Categorical exp(w k T ) K j=1 exp(w j T ) K k=1 ,(9)
and returning x
T as an approximate sample from p T . We discuss more refined resampling techniques in Sec. 4. For estimating the expectation of test functions ϕ, we account for the weights by reweighting a collection of K particles, i.e.,
E p T [ϕ(x)] ≈ K k=1 exp(w k T ) j exp(w j T ) ϕ(x k T ) .(10)
For justification of the validity of this weighting scheme for Feynman-Kac PDEs, see App. A. The expression in Eq. ( 10) corresponds to Self-Normalized Importance Sampling (SNIS) estimation, which converges to exact expectation estimators when K → ∞ (e.g. Naesseth et al. ( 2019)).
this section cite: []

Section: Flexibility of Simulation for Given Marginals
Given a PDE describing the time-evolution of a particular density p t (x), there may exist multiple simulation methods.
For instance, it is well-known that the diffusion equation ( 4) can be simulated using an ODE (Song et al., 2021). Diffusion → Continuity Through simple manipulations, we can rewrite the diffusion equation using a continuity equation and change the simulation scheme accordingly
∂pt(x) ∂t = σ 2 t 2 ∆pt(x) = -∇, pt(x) - σ 2 t 2 ∇ log pt(x) =⇒ dxt = - σ 2 t 2 ∇ log pt(xt)dt .(11)
The reweighting equation adds an extra dimension to the interplay between different simulation schemes. Continuity → Reweighting We first recast the continuity equation in terms of reweighting, in which case the simulation changes the density solely by adjusting the weights of samples (without transport),
∂pt(x) ∂t = -∇, pt(x)vt(x) = -1 pt(x) ∇, pt(x)vt(x) pt(x) =⇒ dwt = (-∇, vt(xt) -∇ log pt(xt), vt(xt) )dt (12
)
Diffusion → Reweighting We further observe that diffusion terms may be captured in the weights using
∂pt(x) ∂t = σ 2 t 2 ∆pt(x) = σ 2 t 2 pt(x) ∆ log pt(x) + ∥∇ log pt(x)∥ 2 =⇒ dwt = σ 2 t 2 (∆ log pt(xt) + ∥∇ log pt(xt)∥ 2 ) dt (13
)
In particular, using Eqs. ( 12) and ( 13) we now have an approach for translating arbitrary flow v t or diffusion σ t terms into the reweighting factors, assuming access to an exact score function ∇ log p t . Such manipulations will play a key role in deriving our proposed methods in Sec. 3.
this section cite: []

Section: Modifying Diffusion Inference using Feynman-Kac Correctors
In this section, we propose new sampling tools for combining or modifying diffusion models at inference time using the Feynman-Kac PDEs in Sec. 2.2. To this end, consider several different pretrained diffusion models with marginals
{q i t } M i=1 following ∂q i t ∂t = -∇, q i t -f t + σ 2 t ∇ log q i t + σ 2 t 2 ∆q i t , (14a
)
dx t = -f t (x t ) + σ 2 t ∇ log q i t (x t ) dt + σ t dW t , (14b
) which is the denoising SDE from Eq. ( 2). Note that q i t may arise from training on different datasets or correspond to conditional models with different conditioning. Throughout this work, we assume access to an exact score model x), in part to facilitate the conversion rules introduced in Sec. 2.3 and summarized in Table 1.
s i t (x; θ i ) = ∇ log q i t(
At inference time, we would like to sample from a modified target distribution involving these given models. While other variants are possible, we focus on the following examples:
Annealed: p anneal t,β (x) = 1 Z t (β) q t (x) β Product: p prod t (x) = 1 Z t q 1 t (x)q 2 t (x) (15
)
Geometric Avg:
p geo t,β (x) = 1 Z t (β) q 1 t (x) 1-β q 2 t (x) β .
A common heuristic for sampling from the distributions in the form of Eq. ( 15) is to simulate according to the score function of the target density. For example, in classifier-free guidance (Ho & Salimans, 2021) we use the score of the ge-ometric average ∇ log p geo t,β = (1 -β)∇ log q 1 t + β∇ log q 2 t to simulate the following SDE dx t = (-f t (x t ) + σ 2 t ∇ log p geo t,β (x t ))dt + σ t dW t . (16) However, despite the similarity to Eq. ( 2), this heuristic does not sample from the prescribed marginals (including the final distribution), except in special cases. We proceed by using the p geo t,β example to illustrate our approach.
this section cite: ['b31']

Section: Outline of Our Approach
To remedy this, we inspect the PDE corresponding to p geo t,β , which can be written in terms of the evolution of q 1 t and
q 2 t ∂p geo t,β (x) ∂t = ∂ ∂t 1 Z t (β) q 1 t (x) (1-β) q 2 t (x) β .(17)
Expanding and using our expressions for the Fokker-Planck equation of q i t in (14), we proceed to locate terms corresponding to the simulation of an SDE with the drift v t (x t ) = -f t (x t ) + σ 2 t ∇ log p geo t,β (x t ). Collecting all remaining terms of PDE (17) into weights ḡt (x t ) we obtain the following Feynman-Kac PDE, which can be simulated using the weighted SDE in Eq. ( 8), along with the resampling schemes described in Sec. 4
∂p geo t,β ∂t = -∇, p geo t,β v t + σ 2 t 2 ∆p geo t,β + p geo t,β ḡt .(18)
Conversion Rules To facilitate the construction of Feynman-Kac PDEs corresponding to existing simulation schemes, in Table 1 we present the conversion rules that describe how the corresponding PDEs change for the annealed densities and the product of densities. We use these rules as building blocks when deriving our practical schemes.
this section cite: []

Section: Computational Considerations
Our recipe above can yield many different weighted PDEs for a given sequence of target distributions. In practice, we would like our simulation scheme to closely approximate the intermediate targets distributions to limit the need for correction. On the other hand, for computational efficiency, we hope to obtain weights which avoid expensive divergence ∇, v t (x) or Laplacian terms ∇, ∇ log q i t (x t ) . Remarkably, for linear drift functions f t (x) commonly used in diffusion models (Song et al., 2021), we find that simulating according to the common heuristic in Eq. ( 16) yields a Feynman-Kac PDE whose weights can be estimated with no additional overhead. We focus on these schemes in our examples.
this section cite: []

Section: Classifier-Free Guidance (CFG)
CFG (Ho & Salimans, 2021) is a widely-used procedure that simulates an SDE combining the scores of conditional and unconditional models with a guidance weight β,
∇ log p t,β (x) = (1 -β)∇ log q 1 t (x | ∅) + β∇ log q 2 t (x | c)
In practice, q 1 t (x|∅) may represent an unconditional model (or a model with an empty prompt) whereas q 2 t (x|c) is conditioned on a text prompt, class, or other random variables (Ho & Salimans, 2021). Alternatively, in autoguidance techniques, q 1 t may be an undertrained version of a stronger conditional or unconditional model q 2 t (Karras et al., 2024a). For our purposes, we will view CFG as an attempt to sample from the geometric average distributions p geo t,β (x) ∝ q 1 t (x) 1-β q 2 t (x) β . Using the conversion rules in Table 1, we derive the reweighting terms which facilitate consistent sampling along the trajectory.
Proposition 3.1 (Classifier-Free Guidance + FKC). Consider two diffusion models q 1 t (x), q 2 t (x) defined via (14). The weighted SDE corresponding to the geometric average of the marginals
p geo t,β (x) ∝ q 1 t (x) 1-β q 2 t (x) β is dx t = σ 2 t ((1 -β)∇ log q 1 t (x t ) + β∇ log q 2 t (x t ))dt -f t (x t )dt + σ t dW t ,(19)
dw t = σ 2 t 2 β(β -1) ∇ log q 1 t (x t ) -∇ log q 2 t (x t ) 2 dt .
In Prop. D.3, we provide a more general formulation of this proposition outlining a continuous family of weighted SDEs sampling from the geometric average p geo t,β (x) ∝ q 1 t (x) 1-β q 2 t (x) β . As a further example, we combine CFG with a product of experts in Prop. D.4.
this section cite: ['b31', 'b31']

Section: Annealed Distribution
Next, we consider a single diffusion model with the learned score ∇ log q t (x), which we use to sample from the annealed or tempered density
p anneal t,β (x) = q t (x) β /Z t (β) .(20)
For β > 1, this can be used to generate samples from modes or high-probability regions of given models (Karczewski et al., 2025), while in Sec. 5.2 we explore the use of annealed inference in learning diffusion samplers from Boltzmann densities. The annealed target can be shown to admit the following Feynman-Kac weighted simulation scheme.
Proposition 3.2 (Annealed SDE + FKC). Consider a diffusion model q t (x) defined via (14). Sampling from the annealed marginals p anneal t,β (x) ∝ q t (x) β , β > 0 can be performed by simulating the following weighted SDE
dx t = (-f t (x t ) + ησ 2 t ∇ log q t (x t ))dt + ζσ t dW t , dw t = (β -1) ∇, f t (x t ) + σ 2 t 2 β∥∇ log q t (x t )∥ 2 dt ,
with the coefficients (for (β + (1 -β)2a)/β ≥ 0)
η = β + (1 -β)a , ζ = (β + (1 -β)2a)/β . (21
) Original FK-PDE Original wSDE Annealed PDE Annealed SDE dx t = FK Corrector dw t += Proof -∇, q t v t v t (x t )dt -∇, p t,β v t v t (x t )dt -(β -1) ∇, v t dt Prop. C.1 -∇, p t,β βv t βv t (x t )dt β(β -1) ∇ log q t , v t dt Prop. C.2 σ 2 t 2 ∆q t σ t dW t σ 2 t 2 ∆p t,β σ t dW t -β(β -1) σ 2 t 2 ∥∇ log q t ∥ 2 dt Prop. C.3 σ 2 t 2β ∆p t,β σt √ β dW t (β -1) σ 2 t 2 ∆ log q t dt Prop. C.4 g t q t dw t = g t dt βg t p t,β -βg t dt Prop. C.5 --time-dependent annealing: β → β t ∂βt ∂t log q t dt Prop. C.6 Original FK-PDE Original wSDE Product PDE Product SDE dx t
= FK Corrector dw t += -∇, q t v 1,2 t v 1,2 t dt -∇, p t (v 1 t + v 2 t ) (v 1 t + v 2 t )dt ( ∇ log q 1 t , v 2 t + ∇ log q 2 t , v 1 t )dt Prop. C.7 σ 2 t 2 ∆q 1,2 t σ t dW t σ 2 t 2 ∆p t σ t dW t -σ 2 t ∇ log q 1 t , ∇ log q 2 t dt Prop. C.8 g 1,2 t q 1,2 t dw t = g 1,2 t dt (g 1 t + g 2 t )p t - (g 1 t + g 2 t )dt
Prop. C.9
Table 1. Conversion rules for different terms of the original Feynman-Kac PDEs (FK-PDEs) and the corresponding weighted SDE (wSDE).
For every term term corresponding to the original densities qt (first two columns), we present the terms corresponding to the annealed marginals p t,β (x) ∝ qt(x) β (top part) and the terms corresponding to the product of marginals pt(x) ∝ q 1 t (x)q 2 t (x) (bottom part). Importantly, the correctors are additive in the weight space, e.g. when transforming the Fokker-Planck equation, we transform both the continuity & diffusion equation terms and sum the corresponding correctors. References to proofs are provided in the right-most column. See Prop. D.1 for proof, and note that linear drifts f t (x) will lead to constant divergence terms which cancel upon reweighting in ( 9) and (10). We detail two choices of a.
Target Score Simulation For a = 0, we have η = β and ζ = 1, which yields the target score SDE whose drift corresponds to the score of the annealed target,
dx t = (-f t (x t ) + βσ 2 t ∇ log q t (x t ))dt + σ t dW t . (22
)
Tempered Noise Simulation For a = 1/2, we have η = (1 + β)/2, ζ = 1/ √ β). We refer to this as an SDE with tempered noise, namely
dxt = (-ft(xt) + β + 1 2 σ 2 t ∇ log qt(xt))dt + σt √ β dWt . (23
)
We focus on these two choices of a, but note that for different β, we found that either target score or tempered-noise simulation could perform better in practice (Sec. 5).
this section cite: ['b36']

Section: Product of Experts (PoE)
Intuitively, samples from the product of densities correspond to the generations that have high likelihood values under both models. The product can also be interpreted as unanimous vote of experts, since a sample is not accepted if one of the densities is zero. Formally, consider the density
p prod t (x) = q 1 t (x)q 2 t (x)/Z t .(24)
For conditional generative models, the product of densities can describe samples satisfying several conditions. For example, in image generation, we could use q(x | "horse")q(x | "a sandy beach") to generate images of "a horse on a sandy beach" (Du et al., 2023). In Sec. 5.3, we demonstrate that the PoE target can be used to improve molecule generations which satisfy multiple conditions simultaneously.
Again, a natural heuristic is to use the score of the target product density in the reverse-time SDE (2),
∇ log p prod t (x) = ∇ log q 1 t (x t ) + ∇ log q 2 t (x t ) ,(25)
In the following proposition, we further combine these rules with the annealing procedure to present the weighted SDE that samples from the marginals p prod t,β (x) ∝ (q 1 t (x)q 2 t (x)) β .
Proposition 3.3 (Product of Experts + FKC). Consider two diffusion models q 1 t (x), q 2 t (x) defined via (14). The weighted SDE corresponding to the product of the marginals p prod t,β (x) ∝ (q 1 t (x)q 2 t (x)) β , with β > 0 is
dx t = σ 2 t η ∇ log q 1 t (x t ) + ∇ log q 2 t (x t ) dt -f t (x t )dt + ζσ t dW t ,(26)
dw t = β(β -1) σ 2 t 2 ∇ log q 1 t (x t ) + ∇ log q 2 t (x t ) 2 dt + βσ 2 t ∇ log q 1 t (x t ), ∇ log q 2 t (x t ) dt + (2β -1) ∇, f t (x t ) dt ,(27)
with the coefficients (for (β + (1 -β)2a)/β ≥ 0)
η = β + (1 -β)a , ζ = (β + (1 -β)2a)/β . (28)
See proof in Prop. D.2. Again, note that for linear drifts, the divergence term ∇, f t (x) is constant and can be ignored. Further, for β = 1, the first term in the weight evolution vanishes to leave only the inner product of score vectors. Similarly to Eqs. ( 22) and ( 23) for annealing, we have the target score SDE (a = 0, η = β, ζ = 1) and the tempered noise SDE (a = 1/2, η = (β + 1)/2, ζ = 1/ √ β).
More generally, we derive the weighted SDE that samples from p t,β (x) ∝ i q i t (x) βi , i.e. the weighted product of marginal densities q i t (x) for arbitrary number of diffusion models (see Prop. D.5).
this section cite: ['b23']

Section: Reward-tilted Target Density
Finally, our framework can easily incorporate a reward function r(x) defined on the state-space at inference time. Namely, we assume that the function exp(β t r(x)) is normalizable and consider the reward-tilted density p reward t (x) ∝ q t (x) exp(β t r(x)). Despite its similarity to the product of densities, this case is different as we do not assume exp(β t r(x)) changes according to the diffusion process.
Proposition 3.4 (Reward-tilted Target + FKC). Consider a diffusion model q t (x) defined via (14). Sampling from the reward-tilted marginals p reward t (x) ∝ q t (x) exp(β t r(x)) is performed by the following weighted SDE
dx t = σ 2 t (∇ log q t (x t ) + β t 2 ∇r(x t ))dt- -f t (x t )dt + σ t dW t ,(29)
dw t = ∂β t ∂t r(x t )dt -β t ∇r(x t ), f t (x t ) dt+ + β t ∇r(x t ), σ 2 t 2 ∇ log q t (x t ) dt .(30)
See proof in Prop. D.6. Here, the weights increase when the vector field of the diffusion models aligns with the gradient of the reward function.
this section cite: []

Section: Resampling Methods
In this section, we describe several options for utilizing the weights to improve sampling with a batch of K particles. While the simplest technique would be to simulate the weighted SDE in Eq. ( 8) for K independent particles across the full time interval t ∈ [0, 1] and reweight using SNIS in (10), we expect these full-trajectory weights to have high variance in practice due to error accumulation.
Sequential Monte Carlo Since our weights provide a proper weighting scheme for all intermediate distributions ((Naesseth et al., 2019), App. A), we can leverage SMC techniques which reweight particles along our trajectories.
In practice, we find that resampling only over an 'active interval' t ∈ [t min , t max ] is useful for improving sample quality and preserving diversity, and set weights to zero outside of this interval. Within the active interval, we resample at each step based on the increment w
(k) t = g t (x (k)
t )dt, using systematic sampling proportional to exp{w (k) t } (Douc & Cappé, 2005). For small discretizations dt, we might expect relatively low-variance weights. From this perspective, systematic resampling is an attractive selection mechanism as all particles are preserved in the case of uniform weights.
Jump Process Interpretation of Reweighting Finally, by reframing the reweighting equation in terms of a Markov jump process (Ethier & Kurtz (2009, Ch. 4.2)), a variety of further simulation algorithms for Feynman-Kac PDEs are possible (Del Moral (2013, Ch 1.2.2, 5
this section cite: ['b21']

Section: ); Rousset & Stoltz (2006); Angeli (2020)).
A Markov jump process is determined by a rate function λ t (x), which governs the frequency of jump events, and a Markov transition kernel J t (y|x), which is used to sample the next state when a jump occurs. The forward Kolmogorov equation for a jump process is given by
∂p jump t (x) ∂t = λ t (y)J t (x|y)p t (y)dy -p t (x)λ t (x)
where the two terms can intuitively be seen to measure the inflow and outflow of probability due to jumps.
Our goal is to find λ t (x), J t (y|x) such that p jump t matches the evolution of p w t in Eq. ( 6) for a given choice of g t . In fact, there are many possible jump processes which satisfy this property (Del Moral (2013, Ch. 5); Angeli et al. (2019)) We present a particular choice here, with proof in App. B.2. Proposition 4.1. For a given g t in Eq. ( 6), define the jump process rate and transition as
λ t (x) = g t (x) -E pt [g t ] - (31a) J t (y|x) = g t (y) -E pt [g t ] + p t (y) g t (z) -E pt [g t ] + p t (z)dz (31b
)
where (u) -:= max(0, -u) and (u) + := max(0, u). Then,
∂p jump t (x) ∂t = ∂p w t (x) ∂t = p t (x) g t (x) -E pt [g t ] (32)
which matches Eq. (6).
In continuous time and the mean-field limit, this jump process formulation of reweighting corresponds to simulating
x t+dt = x t w.p. 1 -λ t (x t )dt + o(dt) ∼ J t (y|x t ) w.p. λ t (x t )dt + o(dt).(33)
We expect this process to improve the sample population in efficient fashion, since jump events are triggered only in states where
(g t (x)-E pt [g t ]) -≥ 0 =⇒ g t (x) ≤ E pt [g t ],
and transitions are more likely to jump to states with high excess weight (g t (y) -E pt [g t ]) + > 0.
In practice, we use an empirical approximation k) ) to approximate the jump rate λ t (x) and transition J t (y|x). Instead of simulating Eq. ( 33) directly, one can also adopt an implementation based on birth-death 'exponential clocks' (BDC, Del Moral (2013, Ch. 5.3-4), see App. B.3). EDM2 + FKC [ours] EDM2 + CFG Figure 3. Samples from EDM2+CFG (top), EDM2+FKC (bottom).
p K t (z) = 1 K K k=1 δ z (x (
this section cite: []

Section: Empirical Study
In this section, we compare our Feynman-Kac corrector (FKC) resampling schemes against their corresponding SDEs without resampling. We consider both target score and tempered noise SDEs. While we show results for BDC sampling in App. F.2 Table A1, we proceed with systematic resampling throughout the remainder of our experiments.
this section cite: []

Section: Image Generation with EDM2
In this section, we study the effect of FKC resampling for image generation in RGB pixel space, using CFG with an EDM2-XS model trained on ImageNet-512 (Karras et al., 2024b). In particular, we test whether resampling to more closely match the intermediate geometric average distributions translates to improvement in two downstream image quality metrics: CLIP Score (Radford et al., 2021) and Im-ageReward (Xu et al., 2024). CLIP Score measures the cosine similarity between the image and text prompt embeddings; ImageReward assigns a score that reflects human preferences (aesthetic quality and prompt adherence).
For a fixed simulation scheme, we compare the effect of adding FKC resampling ( ) versus the standard baseline without resampling ( ). We report results across various simulation parameters, namely the number of sampling steps N and churn parameter γ (which controls how the SDE integration scheme adds noise). For FKC, we additionally sweep over the batch size or number of particles K, whereas K = 1 corresponds to the no-resampling baseline ( ). To calculate metrics on a single image for FKC, we resample from among K particles according to the weights since the last resampling step. Note that we often observe that final-step images from a single batch with FKC are nearly identical visually due to weight degeneracy.
In Table 2, we compare the quantitative performance of our FKC resampling against vanilla CFG. We find that adding FKC ( ) improves performance in both ImageReward and CLIP score, indicating both higher prompt adherence and aesthetically better images. While this comes at the cost of extra computation due to K > 1, we find that FKC demonstrates benefits even for K = 2, with K = 8 performing the best (Table A3). Qualitative results in Fig. 3 further support the finding that FKC can improve image quality.
In App. F.5, we provide an additional analysis on using FKC with latent diffusion image models.
this section cite: ['b55']

Section: Samplers from the Boltzmann Density
As described in the Sec. 1, our FKC inference techniques suggest flexible schemes for learning diffusion samplers at a given temperature and sampling according to a different temperature. Since we are given an energy function in these settings, we are not restricted to learning with temperature 1 for for our base model q t . Thus, we use (T L , T S ) to refer to the learning (q t ) and sampling target (p t,β ) distributions, with β = T L /T S in the notation of Sec. 3.3.
this section cite: []

Section: Mixture of 40 Gaussians with Ground-Truth q β t
To verify our tools in a tractable setting, we consider a highly multimodal distribution where we can calculate the optimal q t and ∇ log q t for (small) integer T L . We show qualitative results in Fig. 2. We find that target score + FKC performs best, while tempered noise has a tendency to drop modes. We also find that FKC outperforms SDE-only simulation in both tempered noise and target score settings. This is further supported by quantitative results in Table A1.
this section cite: []

Section: Sampling LJ-13
To demonstrate the utility of first learning a sampler at a high temperature then annealing to a lower temperature vs. directly learning at a lower temperature, we consider a Lennard-Jones (LJ) system of 13 particles at a base temperature T L = 2. We train a Denoising Energy Table 3. LJ-13 sampling task with various SDEs, with performance measured by mean ± standard deviation over 3 seeds. The starting temperature is TL = 2, annealed to target temperatures TS = 0.8 and TS = 1.5. The DEM samples are generated with a model trained at those corresponding target temperatures.
Target Temp. SDE Type FKC Distance-W2 Energy-W1 Energy-W2 0.8 (β = 2.5) Target Score 0.189 ± 0.002 14.730 ± 0.029 15.556 ± 0.045 0.048 ± 0.019 6.252 ± 2.710 6.356 ± 2.673 Tempered Noise 0.108 ± 0.007 6.487 ± 0.056 8.501 ± 0.283 0.047 ± 0.006 7.016 ± 0.538 7.111 ± 0.535 DEM -0.103 ± 0.001 9.794 ± 0.100 9.804 ± 0.101 1.5 (β = 1.33) Target Score 0.168 ± 0.009 5.340 ± 0.054 6.210 ± 0.254 0.083 ± 0.003 3.366 ± 0.083 3.386 ± 0.090 Tempered Noise 0.095 ± 0.006 2.154 ± 0.048 3.920 ± 0.258 0.066 ± 0.002 0.765 ± 0.156 0.939 ± 0.171 DEM -0.268 ± 0.005 4.471 ± 0.105 5.211 ± 0.017 Matching (DEM) model (Akhound-Sadegh et al., 2024) at this base temperature and perform temperature-annealed inference to lower temperatures. In Table 3 and A2 we compare the performance of a DEM model trained at a lower temperature against a DEM model trained at a higher temperature and annealed to the lower temperature using various SDEs. We evaluate methods using the 2-Wasserstein metric between distance distributions, and the 1-and 2-Wasserstein metrics between energy histograms to a reference (App. F.3). We find that tempered noise+FKC performs best at higher temperatures. However, at lower temperatures, the target score SDE+FKC performs best. Both methods outperform DEM directly trained at the lower temperature for temperatures T S ∈ [2.0, 0.8] (Fig. 4). We find DEM is qualitatively easier to learn at higher temperatures requiring much less tuning compared to lower temperatures (Fig. A1). This makes the train-then-anneal approach attractive in this setting. For extended results and discussion see App. F.
this section cite: []

Section: Multi-Target Structure-Based Drug Design
We apply FKC to the setting of structure-based drug design (SBDD), where the goal is to design molecules (or ligands) using the three-dimensional structure of a biological target-typically a protein-as a guide (Anderson, 2003). The ligands are then evaluated based on how well they fit into the protein's binding site. We focus on dual-target drug design, where a molecule should interact with two proteins simultaneously. Dual-target drug design has become increasingly investigated for targeting complex disease pathways such as in various cancers and neurodegeneration (Ramsay et al., 2018), as well as for diminishing drug resistance mechanisms (Yang et al., 2024).
We investigate the performance of PoE using both target score and tempered noise SDEs at various β, with ( ) and without ( ) FKC. Ligand performance is determined by docking scores to each protein target using AutoDock Vina (Eberhardt et al., 2021). We evaluate 100 protein pairs and average our results over tasks. We sampled 5 molecule sizes from the original training set from Guan et al. (2023): {15, 19, 23, 27, 35} generating 32 molecules per size. We showcase our best results in Table 4 and the full ablation in App. F.6. We evaluate the generated molecules on their docking scores to a protein pair, P 1 and P 2 . We report the average of docking score products for each target, as well as the average maximum docking score for a pair. Lower docking scores are better, and so lower maximum docking scores indicate the molecule is better at binding to both targets. We compute the percentage of molecules that have better docking scores than known binders, as well as the number of valid and unique molecules generated, their diversity, their drug-likeness (QED (Bickerton et al., 2012)), and their synthetic accessibility (SA (Ertl & Schuffenhauer, 2009)).
We find that the target noise SDE at β > 0.5 generates molecules with better average docking scores for each of the target proteins compared with both baselines DualDiff (Zhou et al., 2024) and TargetDiff (Guan et al., 2023).
When we incorporate FKC, the average docking scores improve further. In Fig. A7, we observe a positive correlation between the FKC weights and docking scores. There is a slight sacrifice in terms of diversity and uniqueness when resampling with FKC, although this is a common trade-off for an increase in quality. Notably, our method achieves the lowest maximum docking score, meaning that generated ligands are able to better bind to both proteins (on average across tasks). Our method also generates the highest fraction of molecules that are better than known binders (reference molecules), which could motivate using our model in de novo drug design settings (the mean docking score of reference molecules is -7.915 ±2.841 ). We visualize ligands for a sample target pair in Fig. 5 and Fig. A6.
In App. F.7, we further investigate the utility of PoE in generating molecule SMILES using a latent diffusion model, and show that FKC resampling improves generation for small molecules satisfying multiple functional properties.
this section cite: ['b3', 'b56', 'b24', 'b30', 'b7', 'b25', 'b58', 'b30']

Section: Related Work
Sequential Monte Carlo methods have proven useful across a wide range of tasks involving diffusion models, including for reward-guided generation (Uehara et al., 2024;2025;Singhal et al., 2025;Kim et al., 2025;Chen et al., 2025a), conditional generation (Wu et al., 2024), or inverse problems (Dou & Song, 2024;Cardoso et al., 2024).  & Jarzynski, 2008;2011), where additional transport terms are learned to more closely match the evolution of a given density path (Arbel et al., 2021;Chemseddine et al., 2025;Máté & Fleuret, 2023;Tian et al., 2024;Fan et al., 2024;Maurais & Marzouk, 2024;Vargas et al., 2024).
Indeed, the celebrated Jarzynski equality (Jarzynski, 1997;Crooks, 1999) and its variants admit an elegant proof using the Feynman-Kac formula ( Lelièvre et al. (2010, Ch.  4),Vaikuntanathan & Jarzynski (2008)).
Predictor-corrector simulation (Song et al., 2021) performs additional Langevin steps to promote matching the intermediate marginals of p t of a diffusion model. These schemes can be adapted for annealed or product targets, although Du et al. (2023) found best performance using Metropolis corrections. Bradley & Nakkiran (2024) interpret standard CFG SDE simulation (19) as a predictor-corrector where the corrector targets a different guidance or geometric mixture weight β ′ = 1 2 (1 + β). Our resampling correctors are instead tailored to the original guidance weight β.
Finally, SMC methods have recently been extended to discrete diffusion models (Singhal et al., 2025;Li et al., 2024;Uehara et al., 2025;Lee et al., 2025a), where the approach of Lee et al. (2025a) is analogous to FKC for discrete settings.
this section cite: ['b46', 'b39', 'b54', 'b20', 'b10', 'b48', 'b49', 'b6', 'b12', 'b27', 'b27', 'b51', 'b35', 'b16', 'b23', 'b9', 'b55', 'b47']

Section: Conclusion
In this work, we proposed FEYNMAN-KAC CORRECTORS, an array of tools allowing for fine control over the sample distributions of diffusion processes. These target distributions may arise in compositional generative modeling (Du & Kaelbling, 2024), where we seek to combine specialist models capturing various chemical properties of molecules or different aspects of a complex prompt. Geometric averaging appears in widely-used CFG techniques while, via annealing, we demonstrate that an approach of first learning an amortized sampler at a higher temperature and then annealing using FKCs down to a lower temperature opens up a new dimension for the construction of amortized samplers.
Finally, our framework allows for the use of reward models (Prop. D.6) and for a time-dependent annealing schedule β t (Prop. C.6), where the log-density terms needed for weights can be estimated using methods from Skreta et al. (2025).
this section cite: ['b22', 'b52']

Section: References
Ref_id:b0 Title: Accurate structure prediction of biomolecular interactions with alphafold 3 Year: (2024)
Ref_id:b1 Title: Iterated denoising energy matching for sampling from Boltzmann densities Year: (2024)
Ref_id:b2 Title: A non-equilibrium transport sampler Year: (2024)
Ref_id:b3 Title: The process of structure-based drug design Year: (2003)
Ref_id:b4 Title: Interacting particle approximations of Feynman-Kac measures for continuous-time jump processes Year: (2020)
Ref_id:b5 Title: Rare event simulation for stochastic dynamics in continuous time Year: (2019)
Ref_id:b6 Title: Annealed flow transport Monte Carlo Year: (2021)
Ref_id:b7 Title: Quantifying the chemical beauty of drugs Year: (2012)
Ref_id:b8 Title: Deep universal probabilistic programming Year: (2019)
Ref_id:b9 Title: Classifier-free guidance is a predictor-corrector Year: (2024)
Ref_id:b10 Title: Monte Carlo guided diffusion for Bayesian linear inverse problems Year: (2024)
Ref_id:b11 Title: Text-conditioned molecule diffusion model leveraging chemically informative latent space Year: (2024)
Ref_id:b12 Title: Neural sampling from Boltzmann densities: Fisher-Rao curves in the Wasserstein geometry Year: (2025)
Ref_id:b13 Title: Solving inverse problems via diffusion-based priors: An approximation-free ensemble sampling approach Year: (2025)
Ref_id:b14 Title: Sequential controlled Langevin diffusions Year: (2025)
Ref_id:b15 Title: An interpolating distance between optimal transport and Fisher-Rao metrics Year: (2018)
Ref_id:b16 Title: Excursions in Statistical Dynamics Year: (1999)
Ref_id:b17 Title: Piecewise-deterministic Markov processes: A general class of non-diffusion stochastic models Year: (1984)
Ref_id:b18 Title: A. Target score matching Year: (2024)
Ref_id:b19 Title: Mean Field Simulation for Monte Carlo Integration Year: (2013)
Ref_id:b20 Title: Diffusion posterior sampling for linear inverse problem solving: A filtering perspective Year: (2024)
Ref_id:b21 Title: Comparison of resampling schemes for particle filtering Year: (2005)
Ref_id:b22 Title: Compositional generative modeling: A single model is not all you need Year: (2024)
Ref_id:b23 Title: and Grathwohl, W. S. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and MCMC Year: (2023)
Ref_id:b24 Title: Autodock vina 1.2. 0: New docking methods, expanded force field, and python bindings Year: (2021)
Ref_id:b25 Title: Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions Year: (2009)
Ref_id:b26 Title: Markov Processes: Characterization and Convergence Year: (2009)
Ref_id:b27 Title: Path-guided particle-based sampling Year: (2024)
Ref_id:b28 Title:  Year: (2009)
Ref_id:b29 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b30 Title: 3d equivariant diffusion for target-aware molecule generation and affinity prediction Year: (2023)
Ref_id:b31 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b32 Title: The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo Year: (2014)
Ref_id:b33 Title: Generator matching: Generative modeling with arbitrary Markov processes Year: (2025)
Ref_id:b34 Title: Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development Year: (2021)
Ref_id:b35 Title: Equilibrium free-energy differences from nonequilibrium measurements: A master-equation approach Year: (1997)
Ref_id:b36 Title: Diffusion models as cartoonists! the curious case of high density regions Year: (2025)
Ref_id:b37 Title: Guiding a diffusion model with a bad version of itself Year: (2024)
Ref_id:b38 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b39 Title: Test-time alignment of diffusion models without reward over-optimization Year: (2025)
Ref_id:b40 Title: Equivariant flows: exact likelihood generative learning for symmetric densities Year: (2020)
Ref_id:b41 Title: A new optimal transport distance on the space of finite Radon measures Year: (2016)
Ref_id:b42 Title: Debiasing guidance for discrete diffusion with sequential Monte Carlo Year: (2025)
Ref_id:b43 Title: Genmol: A drug discovery generalist with discrete diffusion Year: (2025)
Ref_id:b44 Title: Free Energy Computations: A Mathematical Perspective Year: (2010)
Ref_id:b45 Title:  Year: ()
Ref_id:b46 Title: Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review Year: (2024)
Ref_id:b47 Title: Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review Year: (2025)
Ref_id:b48 Title: Escorted free energy simulations: Improving convergence by reducing dissipation Year: (2008)
Ref_id:b49 Title: Escorted free energy simulations Year: (2011)
Ref_id:b50 Title: Denoising diffusion samplers Year: (2023)
Ref_id:b51 Title: Transport meets variational inference: Controlled Monte Carlo diffusions Year: (2024)
Ref_id:b52 Title: Efficient evolutionary search over chemical space with large language models Year: (2025)
Ref_id:b53 Title: Iterated energy-based flow matching for sampling from Boltzmann densities Year: (2024)
Ref_id:b54 Title: Practical and asymptotically exact conditional sampling in diffusion models Year: (2024)
Ref_id:b55 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2024)
Ref_id:b56 Title: Rethinking therapeutic strategies of dual-target drugs: An update on pharmacological smallmolecule compounds in cancer Year: (2024)
Ref_id:b57 Title: Path integral sampler: A stochastic control approach for sampling Year: (2022)
Ref_id:b58 Title: Reprogramming pretrained target-specific diffusion models for dual-target drug design Year: (2024)
