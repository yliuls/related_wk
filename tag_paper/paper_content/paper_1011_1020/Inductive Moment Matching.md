Title: Inductive Moment Matching
Abstract: Figure 1. Generated samples on ImageNet-256×256 using 8 steps.

Section: Introduction
Generative models for continuous domains have enabled numerous applications in images (Rombach et al., 2022;Saharia et al., 2022;Esser et al., 2024), videos (Ho et al., 2022a;Blattmann et al., 2023;OpenAI, 2024), and audio (Chen et al., 2020;Kong et al., 2020;Liu et al., 2023), yet achieving high-fidelity outputs, efficient inference, and stable training remains a core challenge -a trilemma that continues to motivate research in this domain. Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2020b), one of the leading techniques, require many inference steps for high-quality results, while step-reduction methods, such as diffusion distillation (Yin et al., 2024;Sauer et al., 2025;Zhou et al., 2024;Luo et al., 2024a) and Consistency Models (Song et al., 2023;Geng et al., 2024;Lu & Song, 2024;Kim et al., 2023), often risk training collapse without careful tuning and regularization (such as pre-generating data-noise pair and early stopping).
To address the aforementioned trilemma, we introduce Inductive Moment Matching (IMM), a stable, single-stage training procedure that learns generative models from scratch for single-or multi-step inference. IMM operates on the time-dependent marginal distributions of stochastic interpolants (Albergo et al., 2023) -continuous-time stochastic processes that connect two arbitrary probability density functions (data at t = 0 and prior at t = 1). By learning a (stochastic or deterministic) mapping from any marginal at time t to any marginal at time s < t, it can naturally support one-or multi-step generation (Figure 2).
IMM models can be trained efficiently from mathematical induction. For time s < r < t, we form two distributions at s by running a one-step IMM from samples at r and t. We then minimize their divergence, enforcing that the distributions at s are independent of the starting time-steps. This construction by induction guarantees convergence to the data distribution. To help with training stability, we model IMM based on certain stochastic interpolants and optimize the objective with stable sample-based divergence estimators such as moment matching (Gretton et al., 2012). Notably, we prove that Consistency Models (CMs) are a single-particle, first-moment matching special case of IMM, which partially explains the training instability of CMs.
On ImageNet-256×256, IMM surpasses diffusion models and achieves 1.99 FID with only 8 inference steps using standard transformer architectures. On CIFAR-10, IMM similarly achieves state-of-the-art of 1.98 FID with 2-step generation for a model trained from scratch.
this section cite: ['b50', 'b51', 'b12', 'b5', 'b47', 'b9', 'b31', 'b36', 'b55', 'b20', 'b66', 'b54', 'b69', 'b14', 'b38', 'b28', 'b1', 'b16']

Section: Preliminaries

this section cite: []

Section: Diffusion, Flow Matching, and Interpolants
For a data distribution q(x), Variance-Preserving (VP) diffusion models (Ho et al., 2020;Song et al., 2020b) and Flow Matching (FM) (Lipman et al., 2022;Liu et al., 2022) construct time-augmented variables x t as an interpolation between data x ∼ q(x) and prior ϵ ∼ N (0, I) such that x t = α t x + σ t ϵ where α 0 = σ 1 = 1, α 1 = σ 0 = 0. VP diffusion commonly chooses α t = cos π 2 t , σ t = sin π 2 t and FM chooses α t = 1 -t, σ t = t. Both v-prediction diffusion (Salimans & Ho, 2022) and FM are trained by matching the conditional velocity v t = α ′ t x + σ ′ t ϵ such that a neural network G θ (x t , t) approximates E x,ϵ [v t |x t ]. Samples can then be generated via probability-flow ODE (PF-ODE) dxt  dt = G θ (x t , t) starting from ϵ ∼ N (0, I). Stochastic interpolants. Unifying diffusion models and FM, stochastic interpolants (Albergo et al., 2023;Albergo & Vanden-Eijnden, 2022) construct a conditional interpolation q t (x t |x, ϵ) = N (I t (x, ϵ), γ 2 t I) between any data x ∼ q(x) and prior ϵ ∼ p(ϵ) and sets constraints I 1 (x, ϵ) = ϵ, I 0 (x, ϵ) = x, and γ 1 = γ 0 = 0. Similar to FM, a deterministic sampler can be learned by explicitly matching the conditional interpolant velocity v t = ∂ t I t (x, ϵ) + γt z where z ∼ N (0, I) such that G θ (x t , t) ≈ E x,ϵ,z [v t |x t ]. Sampling is performed following the PF-ODE dxt dt = G θ (x t , t) similarly starting from prior ϵ ∼ p(ϵ).
When γ t ≡ 0 and I t (x, ϵ) = α t x + σ t ϵ for α t , σ t defined Figure 2. Using an interpolation from data to prior, we define a one-step sampler that moves from any t to s < t, directly transforming qt(xt) to qs(xs). This can be repeated by jumping to an intermediate r < t before moving to s < r.
in FM, the intermediate variable x t = α t x + σ t ϵ becomes a deterministic interpolation and its interpolant velocity v t = α ′ t x + σ ′ t ϵ reduces to FM velocity. Thus, its training and inference both reduce to that of FM. When ϵ ∼ N (0, I), stochastic interpolants reduce to v-prediction diffusion.
this section cite: ['b20', 'b35', 'b37', 'b1', 'b0']

Section: Maximum Mean Discrepancy
Maximum Mean Discrepancy (MMD, Gretton et al. (2012)) between distribution p(x), q(y) for x, y ∈ R D is an integral probability metric (Müller, 1997) commonly defined on Reproducing Kernel Hilbert Space (RKHS) H with a positive definite kernel k : R D × R D → R as MMD 2 (p(x), q(y)) = ∥E x [k(x, •)] -E y [k(y, •)]∥ 2 H (1) where the norm is in H. Choices such as the RBF kernel imply an inner product of infinite-dimensional feature maps consisting of all moments of p(x) and q(y), i.e. E[x j ] and E[y j ] for integer j ≥ 1 (Steinwart & Christmann, 2008).
this section cite: ['b16', 'b45', 'b60']

Section: Inductive Moment Matching
We introduce Inductive Moment Matching (IMM), a method that trains a model of both high quality and sampling efficiency in a single stage. To do so, we assume a timeaugmented interpolation between data (distribution at t = 0) and prior (distribution at t = 1) and propose learning an implicit one-step model (i.e. a one-step sampler) that transforms the distribution at time t to the distribution at time s for any s < t (Section 3.1). The model enables direct onestep sampling from t = 1 to s = 0 and few-step sampling via recursive application from any t to any r < t and then to any s < r until s = 0; this allows us to learn the model from its own samples via bootstrapping (Section 3.2).
this section cite: []

Section: Model Construction via Interpolants
Given data x ∼ q(x) and prior ϵ ∼ p(ϵ), the timeaugmented interpolation x t defined in Albergo et al. (2023) follows x t ∼ q t (x t |x, ϵ). This implies a marginal interpolating distribution q t (x t ) = q t (x t |x, ϵ)q(x)p(ϵ)dxdϵ.
We learn a model distribution implicitly defined by a onestep sampler that transforms q t (x t ) into q s (x s ) for some s ≤ t. This can be done via a special class of interpolants, which preserves the marginal distribution q s (x s ) while interpolating between x and x t . We term these marginalpreserving interpolants among a class of generalized interpolants. Formally, we define x s as a generalized interpolant between x and x t if, for all s ∈ [0, t], its distribution follows q s|t (x s |x, x t ) = N (I s|t (x, x t ), γ 2 s|t I)
and satisfies constraints I t|t (x, x t ) = x t , I 0|t (x, x t ) = x, γ t|t = γ 0|t = 0, and q t|1 (x t |x, ϵ) ≡ q t (x t |x, ϵ). When t = 1, it reduces to regular stochastic interpolants. Next, we define marginal-preserving interpolants.
Definition 1 (Marginal-Preserving Interpolants). A generalized interpolant x s is marginal-preserving if for all t ∈ [0, 1] and for all s ∈ [0, t], the following equality holds:
q s (x s ) = q s|t (x s |x, x t )q t (x|x t )q t (x t )dx t dx, (4
)
where
q t (x|x t ) = q t (x t |x, ϵ)q(x)p(ϵ) q t (x t ) dϵ.(5)
That is, this class of interpolants has the same marginal at s regardless of t. For all t ∈ [0, 1], we define our noisy model distribution at s ∈ [0, t] as
p θ s|t (x s ) = q s|t (x s |x, x t )p θ s|t (x|x t )q t (x t )dx t dx (6)
where the interpolant is marginal preserving and p θ s|t (x|x t ) is our clean model distribution implicitly parameterized as a one-step sampler. This definition also enables multistep sampling. To produce a clean sample x given x t ∼ q t (x t ) in two steps via an intermediate s: (1) we sample x ∼ p θ s|t (x|x t ) followed by xs ∼ q s|t (x s |x, x t ) and (2) if the marginal of xs matches q s (x s ), we can obtain x by x ∼ p θ 0|s (x|x s ). We are therefore motivated to minimize divergence between Eq. ( 4) and (6) using the objective below.
Naïve objective. As one can easily draw samples from the model, it can be naïvely learned by directly minimizing
L(θ) = E s,t D(q s (x s ), p θ s|t (x s ))(7)
with time distribution p(s, t) and a sample-based divergence metric D(•, •) such as MMD or GAN (Goodfellow et al., 2020). If an interpolant x s is marginal-preserving, then the minimum loss is 0 (see Lemma 3). One might also notice the similarity between right-hand sides of Eq. ( 4) and ( 6). However, q s (x s ) = p θ s|t (x s ) does not necessarily imply p θ s|t (x|x t ) = q t (x|x t ). In fact, the minimizer p θ s|t (x|x t ) is not unique and, under mild assumptions, a deterministic minimizer exists (see Section 4).
this section cite: ['b1', 'b15']

Section: Learning via Inductive Bootstrapping
While sound, the naïve objective in Eq. ( 7) is difficult to optimize in practice because when t is far from s, the input distribution q t (x t ) can be far from the target q s (x s ). Fortunately, our interpolant construction implies that the model definition in Eq. ( 6) satisfies boundary condition q s (x s ) = p θ s|s (x s ) regardless of θ (see Lemma 4), which indicates that p θ s|t (x s ) ≈ q s (x s ) when t is close to s. Furthermore, the interpolant enforces p θ s|t (x s ) ≈ p θ s|r (x s ) for any r < t close to t as long as the model is continuous around t. Therefore, we can construct an inductive learning algorithm for p θ s|t (x s ) by using samples from p θ s|r (x s ). For better analysis, we define a sequence number n for parameter θ n and function r(s, t) where s ≤ r(s, t) < t such that p θn s|t (x s ) learns to match p θn-1 s|r (x s ). 1 We omit r's arguments when context is clear and let r(s, t) be a finite decrement from t but truncated at s ≤ t (see Appendix B.3 for well-conditioned r(s, t)).
this section cite: []

Section: General objective.
With marginal-preserving interpolants and mapping r(s, t), we learn θ n in the following objective:
L(θ n ) = E s,t w(s, t)MMD 2 (p θn-1 s|r (x s ), p θn s|t (x s )) (8
)
where w(s, t) is a weighting function. We choose MMD as our objective due to its superior optimization stability and show that this objective learns the correct data distribution.
Theorem 1. Assuming r(s, t) is well-conditioned, the interpolant is marginal-preserving, and θ * n is a minimizer of Eq. (8) for each n with infinite data and network capacity, for all t ∈ [0, 1], s ∈ [0, t],
lim n→∞ MMD 2 (q s (x s ), p θ * n s|t (x s )) = 0.(9)
In other words, θ n eventually learns the target distribution q s (x s ) by parameterizing a one-step sampler p θn s|t (x|x t ).
this section cite: []

Section: Simplified Formulation and Practice
We present algorithmic and practical decisions below.
this section cite: []

Section: Algorithmic Considerations
Despite theoretical soundness, it remains unclear how to empirically choose a marginal-preserving interpolant. First, we present a sufficient condition for marginal preservation.
Definition 2 (Self-Consistent Interpolants). Given s, t ∈ [0, 1], s ≤ t, an interpolant x s ∼ q s|t (x s |x, x t ) is selfconsistent if for all r ∈ [s, t], the following holds:
q s|t (x s |x, x t ) = q s|r (x s |x, x r )q r|t (x r |x, x t )dx r (10
)
In other words, x s has the same distribution if one (1) directly samples it by interpolating x and x t and (2) first samples any x r (given x and x t ) and then samples x s (given x and x r ). Furthermore, self-consistency implies marginal preservation (Lemma 5).
this section cite: []

Section: DDIM interpolant.
Denoising Diffusion Implicit Models (Song et al., 2020a) was introduced as a fast ODE sampler for diffusion models, defined as
DDIM(x t , x, s, t) = α s -σ s σ t α t x + σ s σ t x t (11) and sample x s = DDIM(x t , E x [x|x t ], s, t) can be drawn when E x [x|x t ] is approximated by a network. We show in Appendix C.1 that DDIM as an interpolant, i.e. γ s|t ≡ 0 and I s|t (x, x t ) = DDIM(x t , x, s, t), is self-consistent. Moreover, with deterministic interpolants such as DDIM, there exists a deterministic minimizer p θ s|t (x|x t ) of Eq. (7). Proposition 1. (Informal) If γ s|t ≡ 0 and I s|t (x, x t ) satisfies mild assumptions, there exists a deterministic p θ s|t (x|x t ) that attains 0 loss for Eq. (7).
See Appendix B.6 for formal statement and proof. This allows us to define p θ s|t (x|x t ) = δ(xg θ (x t , s, t)) for a neural network g θ (x t , s, t) with parameter θ by default.
Eliminating stochasticity. We use DDIM interpolant, deterministic model, and prior p(ϵ) = N (0, σ 2 d I) where σ d is the data standard deviation (Lu & Song, 2024). As a result, one can draw x s from model via x s = f θ s,t (x t ) := DDIM(x t , g θ (x t , s, t), s, t) where x t ∼ q t (x t ).
Re-using x t for x r . Inspecting Eq. ( 8) and ( 6), one requires x r ∼ q r (x r ) to generate samples from the target distribution. Instead of sampling x r given a new (x, ϵ) pair, we can reduce variance by reusing x t and x such that x r = DDIM(x t , x, r, t). This is justified because x r derived from x t preserves the marginal distribution q r (x r ) (see Appendix C.2).
this section cite: ['b38']

Section: Stop gradient.
We set n to optimization step number, i.e. advancing from n -1 to n is a single optimizer step where θ n is initialized from θ n-1 . Equivalently, we can omit n from θ n and write θ n-1 as the stop-gradient parameter θ -.
this section cite: []

Section: Simplified objective.
Let x t , x ′ t be i.i.d. random variables from q t (x t ) and x r , x ′ r are variables obtained by reusing x t , x ′ t respectively, the training objective can be derived from the MMD definition in Eq. (1) (see Appendix C.3) as
L IMM (θ) = E xt,x ′ t ,xr,x ′ r ,s,t w(s, t) k y s,t , y ′ s,t (12) + k y s,r , y ′ s,r -k y s,t , y ′ s,r -k y ′ s,t , y s,r
where
y s,t = f θ s,t (x t ), y ′ s,t = f θ s,t (x ′ t ), y s,r = f θ - s,r (x r ), y ′ s,r = f θ - s,r (x ′ r ), k(•,
•) is a kernel function, and w(s, t) is a prior weighting function. An empirical estimate of the above objective uses M particle samples to approximate each distribution indexed by t. In practice, we divide a batch of model output with size B into B/M groups within which share the same (s, t) sample, and the objective is approximated by instantiating B/M number of M × M matrices. Note that the number of model passes does not change with respect to M (see Appendix C.4). A M = 2 version is visualized in Figure 3 and a simplified training algorithm is shown in Algorithm 1. A full training algorithm is shown in Appendix D.
this section cite: []

Section: Other Implementation Choices
We defer detailed analysis of each decision to Appendix C.
this section cite: []

Section: Flow trajectories.
We investigate the two most used flow trajectories (Nichol & Dhariwal, 2021;Lipman et al., 2022),
• Cosine. α t = cos 1 2 πt , σ t = sin 1 2 πt . • OT-FM. α t = 1 -t, σ t = t. Network g θ (x t , s, t). We set g θ (x t , s, t) = c skip (t)x t + c out (t)G θ (c in (t)x t , c noise (s), c noise (t)
) with a neural network G θ , following EDM (Karras et al., 2022). For all choices we let c in (t) = 1/ α 2 t + σ 2 t /σ d (Lu & Song, 2024). Listed below are valid choices for other coefficients.
• Identity. c skip (t) = 0, c out (t) = 1.
• Simple-EDM (Lu & Song, 2024)
. c skip (t) = α t /(α 2 t + σ 2 t ), c out (t) = -σ d σ t / α 2 t + σ 2 t .
• Euler-FM. c skip (t) = 1, c out (t) = -tσ d . This is specific to OT-FM schedule.
We show in Appendix C.5 that f θ s,t (x t ) similarly follows the EDM parameterization of the form f θ s,t (x t ) = c skip (s, t)x t + c out (s, t)G θ (c in (t)x t , c noise (s), c noise (t)).
Noise conditioning c noise (•). We choose c noise (t) = ct for some constant c ≥ 1. We find our model convergence relatively insensitive to c but recommend using larger c, e.g. 1000 (Song et al., 2020b;Peebles & Xie, 2023), because it enables sufficient distinction between nearby r and t.
Mapping function r(s, t). We find that r(s, t) via constant decrement in η t = σ t /α t works well where the decrement is chosen in the form of (η max -η min )/2 k for some appropriate k (details in Appendix C.7).
this section cite: ['b35', 'b26', 'b38', 'b38', 'b48']

Section: Kernel function.
We use time-dependent Laplace kernels of the form k s,t (x, y) = exp(-w(s, t) max(∥x -y∥ 2 , ϵ)/D) for x, y ∈ R D , some ϵ > 0 to avoid undefined gradients, and w(s, t) = 1/|c out (s, t)|. We find Laplace kernels provide better gradient signals than RBF kernels. (see Appendix C.8).
Weighting w(s, t) and distribution p(s, t). We follow VDM (Kingma et al., 2021;Kingma & Gao, 2024) and define p(t) = U(ϵ, T ) and p(s|t) = U(ϵ, t) for constants ϵ, T ∈ [0, 1]. Similarly, weighting is defined as
w(s, t) = 1 2 σ(b -λ t ) - d dt λ t α a t α 2 t + σ 2 t (13
)
where σ(•) is sigmoid function, λ t denotes log-SNR t and a ∈ {1, 2}, σ(•), b ∈ R are constants (see Appendix C.9).
this section cite: ['b30', 'b29']

Section: Sampling
Pushforward sampling. A sample x s can be obtained by directly pushing x t ∼ q t (x t ) through f θ s,t (x t ). This can be iterated for an arbitrary number of steps starting from ϵ ∼ N (0, σ 2 d I) until s = 0. We note that, by definition, one application of f θ s,t (x t ) is equivalent to one DDIM step using the learned network g θ (x t , s, t) as the x prediction. This sampler can then be viewed as a few-step sampler using DDIM where g θ (x t , s, t) outputs a realistic sample x instead of its expectation E x [x|x t ] as in diffusion models.
Restart sampling. Similar to Xu et al. (2023); Song et al. (2023), one can introduce stochasticity during sampling by re-noising a sample to a higher noise-level before sampling again. For example, a two-step restart sampler from x t requires s ∈ (0, t) for drawing sample x = f θ 0,s (x s ) where x s ∼ q s (x s |f θ 0,t (x t )). Classifier-free guidance. Given a data-label pair (x, c), during inference time, classifier-free guidance (Ho & Salimans, 2022) with weight w replaces conditional model output G θ (x t , s, t, c) by a reweighted model output via
wG θ (x t , s, t, c) + (1 -w)G θ (x t , s, t, ∅)(14)
where ∅ denotes the null-token indicating unconditional out-
this section cite: ['b65']

Section: Algorithm 1 Training (see Appendix D for full version)
Input: parameter θ, DDIM(x t , x, s, t), B, M , p Output: learned θ while model not converged do Sample data x, label c, and prior ϵ with batch size B and split into B/M groups. Each group shares a (s, r, t) sample. For each group, x t ← DDIM(ϵ, x, t, 1).
For each group, x r ← DDIM(x t , x, r, t).
For each instance, set c = ∅ with prob. p. Minimize the empirical loss LIMM (θ) in Eq. ( 67). end while Algorithm 2 Pushforward Sampling (details in Appendix F)
Input: model f θ , {t i } N i=0 , N (0, σ 2 d I), (optional) w Output: x t0 Sample x N ∼ N (0, σ 2 d I) for i = N, . . . , 1 do x ti-1 ← f θ ti-1,ti (x ti ) or f θ ti-1,ti,w (x ti
). end for put. Similarly, we define our guided model as f θ s,t,w (x t ) = c skip (s, t)x t +c out (s, t)G w θ (x t , s, t, c) where G w θ (x t , s, t, c) is as defined in Eq. ( 14) and we drop c in (•) and c noise (•) for notational simplicity. We justify this decision in Appendix E. Similar to diffusion models, c is randomly dropped with probability p during training without special practices.
We present pushforward sampling in Algorithm 2 and detail both samplers in Appendix F.
this section cite: []

Section: Connection with Prior Works
Our work is closely connected with many prior works. Detailed analysis is found in Appendix G.
Consistency Models. Consistency models (CMs) (Song et al., 2023;Song & Dhariwal, 2023;Lu & Song, 2024) uses a network g θ (x t , t) that outputs clean data given noisy input x t . It requires point-wise consistency g θ (x t , t) = g θ (x r , r) for any r < t where x r is obatined via an ODE solver from x t using either pretrained model or groundtruth data. Discrete-time CM must satisfy g θ (x 0 , 0) = x 0 and trains via loss E xt,x,t [d(g θ (x t , t), g θ -(x r , r))] where d(•, •) is commonly chosen as L 2 or LPIPS (Zhang et al., 2018).
We show in the following Lemma that CM objective with L 2 distance is a single-particle estimate of IMM objective with energy kernel.
Lemma 1. When x t = x ′ t , x r = x ′ r , k(x, y) = -∥x -y∥ 2 ,
and s > 0 is a small constant, Eq. ( 12) reduces to CM loss E xt,x,t w(t)∥g θ (x t , t) -g θ -(x r , r)∥ 2 for some valid mapping r(t) < t.
This single-particle estimate ignores the repulsion force imposed by k(•, •). Energy kernel also only matches the first moment, ignoring all higher moments. These decisions can be significant contributors to training instability and performance degradation of CMs.
Improved CMs (Song & Dhariwal, 2023) propose pseudohuber loss as d(•, •) which we justify in the Lemma below.
Lemma 2. Negative pseudo-huber loss k c (x, y) = c -∥x -y∥ 2 + c 2 for c > 0 is a conditionally positive definite kernel that matches all moments of x and y where weights on higher moments depend on c.
From a moment-matching perspective, the improved performance is explained by the loss matching all moments of the distributions. In addition to pseudo-huber loss, many other kernels (Laplace, RBF, etc.) are all valid choices in the design space.
We also extend IMM loss to the differential limit by taking r(s, t) → t, the result of which subsumes the continuoustime CM (Lu & Song, 2024) as a single-particle estimate (Appendix H). We leave experiments for this to future work.
this section cite: ['b38', 'b67', 'b38']

Section: Diffusion GAN and Adversarial Consistency Distillation.
Diffusion GAN (Xiao et al., 2021) parameterizes the generative distribution as p θ s|t (x s |x t ) = q s|t (x s |x, x t )δ(x -G θ (x t , z, t))p(z)dzdx for s as a fixed decrement from t and p(z) a noise distribution. It defines the interpolant q s|t (x s |x, x t ) as the DDPM posterior distribution, which is self-consistent (see Appendix G.2) and introduces randomness to the sampling process to match q t (x|x t ) instead of the marginal. Both Diffusion GAN and Adversarial Consistency Distillation (Sauer et al., 2025) use GAN objective, which shares similarity to MMD in that MMD is defined as an integral probability metric where the optimal discriminator is chosen in RKHS. This eliminates the need for explicit adversarial optimization of a neural-network discriminator.
Generative Moment Matching Network. GMMN (Li et al., 2015) directly applies MMD to train a generator G θ (z) where z ∼ N (0, I) to match the data distribution. It is a special case of IMM in that when t = 1 and r(s, t) ≡ s = 0 our loss reduces to naïve GMMN objective.
6. Related Works Diffusion, Flow Matching, and stochastic interpolants. Diffusion models (Sohl-Dickstein et al., 2015;Song et al., 2020b;Ho et al., 2020;Kingma et al., 2021) and Flow Matching (Lipman et al., 2022;Liu et al., 2022) are widely used generative frameworks that learn a score or velocity field of a noising process from data into a simple prior. They have been scaled successfully for text-to-image (Rombach et al., 2022;Saharia et al., 2022;Podell et al., 2023;Chen et al., 2023;Esser et al., 2024) and text-to-video (Ho et al., 2022a;Blattmann et al., 2023;OpenAI, 2024) tasks.
Stochastic interpolants (Albergo et al., 2023;Albergo & Vanden-Eijnden, 2022) extend these ideas by explicitly defining a stochastic path between data and prior, then matching its velocity to facilitate distribution transfer.
this section cite: ['b64', 'b54', 'b34', 'b55', 'b20', 'b30', 'b35', 'b37', 'b50', 'b51', 'b49', 'b8', 'b12', 'b5', 'b47', 'b1', 'b0']

Section: While IMM builds on top of the interpolant construction, it directly learns one-step mappings between any intermediate marginal distributions.
Diffusion distillation. To resolve diffusion models' sampling inefficiency, recent methods (Salimans & Ho, 2022;Meng et al., 2023;Yin et al., 2024;Zhou et al., 2024;Luo et al., 2024a;Heek et al., 2024) focus on distilling one-step or few-step models from pre-trained diffusion models. This two-stage approach currently has the best generation quality and efficiency. However, such methods usually require jointly optimizing two networks and training requires careful tuning in practice to avoid mode collapse (Yin et al., 2024). Another recent work (Salimans et al., 2024) explicitly matches the first moment of the data distribution available from pre-trained diffusion models. In contrast, our method implicitly matches all moments using MMD and can be trained from scratch with a single model.
Few-step generative models from scratch. Early onestep generative models primarily relied on GANs (Goodfellow et al., 2020;Karras et al., 2020;Brock, 2018) and MMD (Li et al., 2015;2017) (or their combination) but scaling adversarial training remains challenging. Recent independent classes of few-step models, e.g. Consistency Models (CMs) (Song et al., 2023;Song & Dhariwal, 2023;Lu & Song, 2024), Consistency Trajectory Models (CTMs) (Kim et al., 2023;Heek et al., 2024) and Shortcut Models (SMs) (Frans et al., 2024) still face training instability and require specialized components (Lu & Song, 2024) (e.g., JVP for flash attention) or other special practices (e.g., high weight decay for SMs, combined LPIPS (Zhang et al., 2018) and GAN losses for CTMs, and special training schedules (Geng et al., 2024)) to remain stable. In contrast, our method trains stably with a single loss and achieves strong performance without special training practices.
this section cite: ['b44', 'b66', 'b69', 'b17', 'b66', 'b53', 'b15', 'b25', 'b6', 'b34', 'b38', 'b28', 'b17', 'b13', 'b38', 'b67', 'b14']

Section: Experiments
We evaluate IMM's empirical performance (Section 7.1), training stability (Section 7.2), sampling choices (Section 7.3), scaling behavior (Section 7.4) and ablate our practical decisions (Section 7.5).
this section cite: []

Section: Image Generation
We present FID (Heusel et al., 2017) results for unconditional CIFAR-10 and class-conditional ImageNet-256×256 in Table 1 and 2. For CIFAR-10, we separate baselines into diffusion and flow models, distillation models, and few-step models from scratch. IMM belongs to the last category (Ho et al., 2020) 3.17 1000 DDPM++ (Song et al., 2020b) 3.16 1000 NCSN++ (Song et al., 2020b) 2.38 1000 DPM-Solver (Lu et al., 2022) 4.70 10 iDDPM (Nichol & Dhariwal, 2021) 2.90 4000 EDM (Karras et al., 2022) 2.05 35 Flow Matching (Lipman et al., 2022) 6.35 142 Rectified Flow (Liu et al., 2022) 2.58 127
Family Method FID (↓) Steps (↓) Diffusion & Flow DDPM
this section cite: ['b18', 'b20', 'b39', 'b26', 'b35', 'b37']

Section: Few-Step via Distillation
PD (Salimans & Ho, 2022) 4.51 2 2-Rectified Flow (Salimans & Ho, 2022) 4.85 1 DFNO (Zheng et al., 2023) 3.78 1 KD (Luhman & Luhman, 2021) 9.36 1 TRACT (Berthelot et al., 2023) 3.32 2 Diff-Instruct (Luo et al., 2024a) 5.57 1 PID (LPIPS) (Tee et al., 2024) 3.92 1 DMD (Yin et al., 2024) 3.77 1 CD (LPIPS) (Song et al., 2023) 2.93 2 CTM (w/ GAN) (Kim et al., 2023) 1.87 2 SiD (Zhou et al., 2024) 1.92 1 SiM (Luo et al., 2024b) 2.06 1 sCD ( Lu & Song, 2024 (Kang et al., 2023) 3.45 1 569M StyleGAN-XL (Karras et al., 2020) 2.30 1 166M
Masked & AR VQGAN (Esser et al., 2021) 26.52 1024 227M MaskGIT (Chang et al., 2022) 6.18 8 227M MAR (Li et al., 2024) 1.98 100 400M VAR-d20 (Tian et al., 2024a) 2.57 10 600M VAR-d30 (Tian et al., 2024a) 1.92 10 2B
this section cite: ['b68', 'b40', 'b4', 'b61', 'b66', 'b28', 'b69', 'b24', 'b25', 'b11', 'b7', 'b33']

Section: Diffusion & Flow
ADM (Dhariwal & Nichol, 2021) 10.94 250 554M CDM (Ho et al., 2022b) 4.88 8100 -SimDiff (Hoogeboom et al., 2023) 2.77 512 2B LDM-4-G (Rombach et al., 2022) 3.60 250 400M U-DiT-L (Tian et al., 2024b) 3.37 250 916M U-ViT-H (Bao et al., 2023) 2.29 50 501M DiT-XL/2 (w = 1.0) (Peebles & Xie, 2023) 9.62 250 675M DiT-XL/2 (w = 1.25) (Peebles & Xie, 2023) 3.22 250 675M DiT-XL/2 (w = 1.5) (Peebles & Xie, 2023) 2.27 250 675M SiT-XL/2 (w = 1.0) (Ma et al., 2024) 9.35 250 675M SiT-XL/2 (w = 1.5) (Ma et al., 2024) 2.15 250 675M Few-Step from Scratch iCT (Song et al., 2023) 34.24 1 675M 20.3 2 675M Shortcut (Frans et al., 2024) 10.60 1 675M 7.80 4 675M 3.80 128 675M IMM (ours) (XL/2, w = 1.25) 7.77 1 675M 5.33 2 675M 3.66 4 675M 2.77 8 675M IMM (ours) (XL/2, w = 1.5) 8.05 1 675M 3.99 2 675M 2.51 4 675M 1.99 8 675M
Table 2. Class-conditional ImageNet-256×256 results.
50k 100k 150k 200k 250k Training Steps 2.5 3.0 3.5 4.0 4.5 5.0 5.5 FID-50k Positional emb. Fourier emb. w/ scale=16 in which it achieves state-of-the-art performance of 1.98 using pushforward sampler. For ImageNet-256×256, we use the popular DiT (Peebles & Xie, 2023) architecture because of its scalability, and compare it with GANs, masked and autoregressive models, diffusion and flow models, and few-step models trained from scratch.
We observe decreasing FID with more steps and IMM achieves 1.99 FID with 8 steps (with w = 1.5), surpassing DiT and SiT (Ma et al., 2024) using the same architecture except for trivially injecting time s (see Appendix I). Notably, we also achieve better 8-step FID than the 10step VAR (Tian et al., 2024a) of comparable size. At 16 steps, IMM also achieves 1.90 FID outperforming VAR's 2B variant (see Appendix I.4). However, different from VAR, IMM grants flexibility of variable number of inference steps. Lastly, we similarly surpass Shortcut models' (Frans et al., 2024) best performance with only 8 steps 2 . We defer inference details to Section 7.3 and Appendix I.2.
2 NFE is twice the number of steps. 1 2 4 8 Inference Steps 2 3 4 5 6 7 8 FID-50k Pushforward/Uniform Pushforward/EDM Restart/Uniform Restart/EDM Figure 6. ImageNet-256×256 FID with different sampler types.
this section cite: ['b23', 'b50', 'b3', 'b48', 'b48', 'b48', 'b43', 'b48', 'b43', 'b13']

Section: IMM Training is Stable
We show that IMM is stable and achieves reasonable performance across a range of parameterization choices.
this section cite: []

Section: Positional vs. Fourier embedding.
A known issue for CMs (Song et al., 2023) is its training instability when using Fourier embedding with scale 16, which forces reliance on positional embeddings for stability. We find that IMM does not face this problem (see Figure 4). For Fourier embedding we use the standard NCSN++ (Song et al., 2020b) architecture and set embedding scale to 16; for positional embeddings, we adopt DDPM++ (Song et al., 2020b). Both embedding types converge reliably, and we include samples from the Fourier embedding model in Figure 4.
Particle number. Particle number M for estimating MMD is an important parameter for empirical success (Gretton et al., 2012;Li et al., 2015), where the estimate is more accurate with larger M . In our case, naïvely increasing M can slow down convergence because we have a fixed batch size B in which the samples are grouped into B/M groups   of M where each group shares the same t. The larger M means that fewer t's are sampled. On the other hand, using extremely small numbers of particles, e.g. M = 2, leads to training instability and performance degradation, especially on a large scale with DiT architectures. We find that there exists a sweet spot where a few particles effectively help with training stability while further increasing M slows down convergence (see Figure 5). We see that in ImageNet-256×256, training collapses when M = 1 (which is CM) and M = 2, and achieves lowest FID under the same computation budget with M = 4. We hypothesize M < 4 does not allow sufficient mixing between particles and larger M means fewer t's are sampled for each step, thus slowing convergence. A general rule of thumb is to use a large enough M for stability, but not too large for slowed convergence.
Noise embedding c noise (•). We plot in Figure 9 the log absolute mean difference of t and r(s, t) in the positional embedding space. Increasing c increases distinguishability of nearby distributions. We also observe similar convergence on ImageNet-256×256 across different c, demonstrating the insensitivity of our framework w.r.t. noise function.
this section cite: ['b16', 'b34']

Section: Sampling
We investigate different sampling settings for best performance. One-step sampling is performed by simple pushforward from T to ϵ (concrete values in Appendix I.2).
On CIFAR-10 we use 2 steps and set intermediate time t 1 such that η t1 = 1.4, a choice we find to work well empirically. On ImageNet-256×256 we go beyond 2 steps and, for simplicity, investigate (1) uniform decrement in t and
(2) EDM (Karras et al., 2024) schedule (detailed in Appendix I.2). We plot FID of all sampler settings in Figure 6 with guidance weight w = 1.5. We find pushforward sam- plers with uniform schedule to work the best on ImageNet-256×256 and use this as our default setting for multi-step generation. Additionally, we concede that pushforward combined with restart samplers can achieve superior results. We leave such experiments to future works.
this section cite: ['b27']

Section: Scaling Behavior
Similar to diffusion models, IMM scale with training and inference compute as well as model size on ImageNet-256×256. We plot in Figure 7 FID vs. training and inference compute in GFLOPs and we find strong correlation between compute used and performance. We further visualize samples in Figure 8 with increasing model size, i.e. DiT-S, DiT-B, DiT-L, DiT-XL, and increasing inference steps, i.e. 1, 2, 4, 8 steps. The sample quality increases along both axes as larger transformers with more inference steps capture more complex distributions. This also explains that more compute can sometimes yield different visual content from the same initial noise as shown in the visual results.
this section cite: []

Section: Ablation Studies
All ablation studies are done with DDPM++ architecture for CIFAR-10 and DiT-B for ImageNet-256×256. FID comparisons use 2-step samplers by default.
Flow schedules and parameterization. We investigate all combinations of network parameterization and flow schedules: Simple-EDM + cosine (sEDM/cos), Simple-EDM + OT-FM (sEDM/FM), Euler-FM + OT-FM (eFM), Identity + cosine (id/cos), Identity + OT-FM (id/FM). Identity parameterization consistently fall behind other types of parameterization, which all show similar performance across datasets (see Table 3). We see that on smaller scale (CIFAR-10), sEDM/FM works the best but on larger scale (ImageNet-256×256), eFM works the best, indicating that OT-FM schedule and Euler paramaterization may be more scalable than other choices.
this section cite: []

Section: Mapping function r(s, t).
Our choices for ablation are (1) constant decrement in η t , (2) constant decrement in t, (3) constant decrement in λ t = log α 2 t /σ 2 t , (4) constant increment in 1/η t (see Appendix C.6). For fair comparison, we choose the decrement gap so that the minimum tr(s, t) is ≈ 10 -3 and use the same network parameterization. FID progression in Figure 11 show that (1) consistently outperforms other choices. We additionally ablate the mapping gap using M = 4 in (1). The constant decrement is in FID-50k w(s, t) = 1 40.19 + ELBO weight 96.43 + α t 33.44 + 1/(α 2 t + σ 2 t ) 27.43 Table 4. Ablation of weighting function w(s, t) on ImageNet-256×256. 50k 100k 150k 200k Training Steps 2 3 4 5 6 7 8 9 FID-50k constant dec. in t constant dec. in t constant dec. in t constant inc. in 1/ t 50k 100k 150k 200k 250k 300k 350k 400k Training Steps 40 60 80 100 120 FID-50k constant dec. in t constant dec. in t constant dec. in t constant inc. in 1/ t the form of (η max -η min )/2 k for an appropriately chosen k. We show in Figure 10 that the performance is relatively stable across k ∈ {11, 12, 13} but experiences instability for k = 14. This suggests that, for a given particle number, there exists a largest k for stable optimization.
this section cite: []

Section: Weighting function.
In Table 4 we first ablate the weighting factors in three groups: (1) the VDM ELBO factors 1 2 σ(b-λ t )(-d dt λ t ), (2) weighting α t (i.e. when a = 1), and (3) weighting 1/(α 2 t + σ 2 t ). We find it necessary to use α t jointly with ELBO weighting because it converts v-pred network to a ϵ-pred parameterization (see Appendix C.9), consistent with diffusion ELBO-objective. Factor 1/(α 2 t + σ 2 t ) upweighting middle time-steps further boosts performance, a helpful practice also known for FM training (Esser et al., 2024). We leave additional study of the exponent a to Appendix I.5 and find that a = 2 emphasizes optimizing the loss when t is small while a = 1 more equally distributes weights to larger t. As a result, a = 2 achieves higher quality multi-step generation than a = 1.
this section cite: ['b12']

Section: Conclusion
We present Inductive Moment Matching, a framework that learns a few-step generative model from scratch. It trains by first leveraging self-consistent interpolants to interpolate between data and prior and then matching all moments of its own distribution interpolated to be closer to that of data.
Our method guarantees convergence in distribution and generalizes many prior works. Our method also achieves stateof-the-art performance across benchmarks while achieving orders of magnitude faster inference. We hope it provides a new perspective on training few-step models from scratch and inspire a new generation of generative models.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2022)
Ref_id:b1 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b2 Title: Conditionally positive definite kernels: theoretical contribution, application to interpolation and approximation Year: (2009)
Ref_id:b3 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b4 Title: Denoising diffusion models with transitive closure time-distillation Year: (2023)
Ref_id:b5 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b6 Title: Large scale gan training for high fidelity natural image synthesis Year: (2018)
Ref_id:b7 Title: Masked generative image transformer Year: (2022)
Ref_id:b8 Title: Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b9 Title: Estimating gradients for waveform generation Year: (2020)
Ref_id:b10 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b11 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b12 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b13 Title: One step diffusion via shortcut models Year: (2024)
Ref_id:b14 Title: Consistency models made easy Year: (2024)
Ref_id:b15 Title: Generative adversarial networks Year: (2020)
Ref_id:b16 Title: A kernel two-sample test Year: (2012)
Ref_id:b17 Title: Multistep consistency models Year: (2024)
Ref_id:b18 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b19 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b20 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b21 Title: Imagen video: High definition video generation with diffusion models Year: (2022)
Ref_id:b22 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b23 Title: simple diffusion: Endto-end diffusion for high resolution images Year: (2023)
Ref_id:b24 Title: Scaling up gans for text-to-image synthesis Year: (2023)
Ref_id:b25 Title: Analyzing and improving the image quality of stylegan Year: (2020)
Ref_id:b26 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b27 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b28 Title: Consistency trajectory models: Learning probability flow ode trajectory of diffusion Year: (2023)
Ref_id:b29 Title: Understanding diffusion objectives as the elbo with simple data augmentation Year: (2024)
Ref_id:b30 Title: Variational diffusion models Year: (2021)
Ref_id:b31 Title: Diffwave: A versatile diffusion model for audio synthesis Year: (2020)
Ref_id:b32 Title: Mmd gan: Towards deeper understanding of moment matching network Year: (2017)
Ref_id:b33 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b34 Title: Generative moment matching networks Year: (2015)
Ref_id:b35 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b36 Title: Text-to-audio generation with latent diffusion models Year: (2023)
Ref_id:b37 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b38 Title: Simplifying, stabilizing and scaling continuous-time consistency models Year: (2024)
Ref_id:b39 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b40 Title: Knowledge distillation in iterative generative models for improved sampling speed Year: (2021)
Ref_id:b41 Title: Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models Year: (2024)
Ref_id:b42 Title: One-step diffusion distillation through score implicit matching Year: (2024)
Ref_id:b43 Title: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b44 Title: On distillation of guided diffusion models Year: (2023)
Ref_id:b45 Title: Integral probability metrics and their generating classes of functions Year: (1997)
Ref_id:b46 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b47 Title: Video generation models as world simulators Year: (2024)
Ref_id:b48 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b49 Title: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b50 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b51 Title: Photorealistic text-to-image diffusion models with deep language understanding Year: (2022)
Ref_id:b52 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b53 Title: Multistep distillation of diffusion models via moment matching Year: (2024)
Ref_id:b54 Title: Adversarial diffusion distillation Year: (2025)
Ref_id:b55 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b56 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b57 Title: Improved techniques for training consistency models Year: (2023)
Ref_id:b58 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b59 Title: Consistency models Year: (2023)
Ref_id:b60 Title: Support vector machines Year: (2008)
Ref_id:b61 Title: Physics informed distillation for diffusion models Year: (2024)
Ref_id:b62 Title: Visual autoregressive modeling: Scalable image generation via nextscale prediction Year: (2024)
Ref_id:b63 Title: Downsample tokens in u-shaped diffusion transformers Year: (2024)
Ref_id:b64 Title: Tackling the generative learning trilemma with denoising diffusion gans Year: (2021)
Ref_id:b65 Title: Restart sampling for improving generative processes Year: (2023)
Ref_id:b66 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b67 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b68 Title: Fast sampling of diffusion models via operator learning Year: (2023)
Ref_id:b69 Title: Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation Year: (2024)
