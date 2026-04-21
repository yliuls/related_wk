Title: Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models
Abstract: We introduce Ψ-SAMPLER, an SMC-based framework incorporating pCNL-based initial particle sampling for effective inference-time reward alignment with a score-based generative model. Inference-time reward alignment with score-based generative models has recently gained significant traction, following a broader paradigm shift from pre-training to post-training optimization. At the core of this trend is the application of Sequential Monte Carlo (SMC) to the denoising process. However, existing methods typically initialize particles from the Gaussian prior, which inadequately captures reward-relevant regions and results in reduced sampling efficiency. We demonstrate that initializing from the reward-aware posterior significantly improves alignment performance. To enable posterior sampling in high-dimensional latent spaces, we introduce the preconditioned Crank-Nicolson Langevin (pCNL) algorithm, which combines dimension-robust proposals with gradient-informed dynamics. This approach enables efficient and scalable posterior sampling and consistently improves performance across various reward alignment tasks, including layout-to-image generation, quantity-aware generation, and aesthetic-preference generation, as demonstrated in our experiments. Project Webpage: https://psi-sampler.github.io/ * Equal contribution.39th Conference on Neural Information Processing Systems (NeurIPS 2025).

Section: Introduction
Recently, a shift in the scaling law paradigm from pre-training to post-training has opened new possibilities for achieving another leap in AI model performance, as exemplified by the unprecedented AGI score of GPT-o3 [1] and DeepSeek's "Aha moment" [2]. Breakthroughs in LLMs have also extended to score-based generative models [3][4][5][6][7], resulting in significant improvements in user preference alignment [8]. Similar to the autoregressive generation process in LLMs, the denoising process in score-based generative models can be interpreted as a Sequential Monte Carlo (SMC) [9][10][11] process with a single particle at each step. This perspective allows inference-time alignment to be applied analogously to LLMs by populating multiple particles at each step and selecting those that score highly under a given reward function [8,[12][13][14][15][16][17]. A key distinction is that score-based generative models enable direct estimation of the final output from any noisy intermediate point via Tweedie's formula [18], facilitating accurate approximation of the optimal value function [19,8,20] through expected reward estimation.
However, previous SMC-based approaches [12,15,14,21,22], where each SMC step is coupled with the denoising process of score-based generative models, are limited in their ability to effectively explore high-reward regions, as the influence of the reward signal diminishes over time due to vanishing diffusion coefficient. Thus, rather than relying on particle exploration during later stages, it is more critical to identify effective initial latents that are well-aligned with the reward model from the outset. In this work, we address this problem and propose an MCMC-based initial particle population method that generates strong starting points for the subsequent SMC process. This direction is particularly timely given recent advances in distillation techniques for score-based generative models [23][24][25][26][27], now widely adopted in state-of-the-art models [28,29]. These methods yield straighter generative trajectories and clearer Tweedie estimates [18] from early steps, enabling more effective exploration from the reward-informed initial distribution.
A straightforward baseline for generating initial particles is the Top-K-of-N strategy: drawing multiple samples from the standard Gaussian prior and selecting those with the highest reward scores. Though effective, this naive approach offers limited improvement in subsequent SMC due to its reliance on brute-force sampling. Motivated by these limitations, we explore Markov Chain Monte Carlo (MCMC) [30][31][32][33][34][35] methods based on Langevin dynamics, which are particularly well-suited to our setting since we sample from the initial posterior distribution, whose form is known. Nevertheless, applying MCMC in our problem presents unique challenges: the exploration space is extremely highdimensional (e.g., 65,536 for FLUX [28]), posing significant challenges for conventional MCMC methods. In particular, the Metropolis-Hastings (MH) accept-reject mechanism, when used with standard Langevin-based samplers, becomes ineffective in such high-dimensional regimes, as the acceptance probability rapidly diminishes and most proposals are rejected.
Our key idea for enabling effective particle population from the initial reward-informed distribution is to leverage the Preconditioned Crank-Nicolson (pCN) algorithm [36][37][38], which is designed for function spaces or infinite-dimensional Hilbert spaces. When combined with the Langevin algorithm (yielding pCNL), its semi-implicit Euler formulation allows for efficient exploration in a high-dimensional space. Furthermore, when augmented with the MH correction, the acceptance rate is significantly improved compared to vanilla MALA. We therefore propose performing pCNL over the initial posterior distribution and selecting samples at uniform intervals along the resulting Markov chain. These samples are then used as initial particles for the subsequent SMC process across the denoising steps. We refer to the entire pipeline-PCNL-based initial particle sampling followed by SMC-based Inference-time reward alignment-as PSI (Ψ)-Sampler. To the best of our knowledge, this is the first work to apply the pCN algorithm in the context of generative modeling.
In our experiments, we evaluate three reward alignment tasks: layout-to-image generation (placing objects in designated bounding boxes within the image), quantity-aware generation (aligning the number of objects in the image with the specified count), and aesthetic-preference generation (enhancing visual appeal). We compare our Ψ-SAMPLER against the base SMC method [14] with random initial particle sampling, SMC combined with initial particle sampling via Top-K-of-N , ULA, and MALA, as well as single-particle methods [39,40]. Across all tasks, Ψ-SAMPLER consistently achieves the best performance in terms of the given reward and generalizes well to the held-out reward, matching or surpassing existing baselines. Its improvement over the base SMC method highlights the importance of posterior-based initialization, while its outperformance over ULA and MALA further confirms the limitations of these methods in extremely high-dimensional spaces.
2 Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b7', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b7', 'b19', 'b11', 'b14', 'b13', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b17', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b13', 'b37', 'b38']

Section: Inference-Time Reward Alignment
Sequential Monte Carlo (SMC) [9][10][11] has proven effective in guiding the generation process of score-based generative models for inference-time reward alignment [22,21,15,14,12]. Prior SMC-based methods differ in their assumptions and applicability. For instance, FPS [15] and MCGdiff [22] are specifically designed for linear inverse problems and thus cannot generalize to arbitrary reward functions. SMC-Diff [21] depends on the idealized assumption that the learned reverse process exactly matches the forward noising process-an assumption that rarely holds in practice. TDS [14] and DAS [12] employ twisting and tempering strategies respectively to improve approximation accuracy while reducing the number of required particles. Despite these variations, all aforementioned SMC-based approaches share a common limitation that they initialize particles from the standard Gaussian prior, which is agnostic to the reward function. This mismatch can result in poor coverage of high-reward regions and reduced sampling efficiency.
In addition to multi-particle systems like SMC, single-particle approaches have also been explored for inference-time reward alignment [39][40][41][42]. These methods guide generation by applying reward gradients along a single sampling trajectory. However, they are inherently limited in inference-time reward alignment, as simply increasing the number of denoising steps does not consistently lead to better sample quality. In contrast, SMC-based methods allow users to trade computational cost for improved reward alignment, making them more flexible and scalable in practice.
this section cite: ['b8', 'b9', 'b10', 'b21', 'b20', 'b14', 'b13', 'b11', 'b14', 'b21', 'b20', 'b13', 'b11', 'b37', 'b38', 'b39', 'b40']

Section: Fine-Tuning-Based Reward Alignment
Beyond inference-time methods, another line of work focuses on fine-tuning score-based generative models for reward alignment. Some approaches perform supervised fine-tuning by weighting generated samples according to their reward scores and updating the model to favor high-reward outputs [43,44], while others frame the denoising process as a Markov Decision Process (MDPs) and apply reinforcement learning techniques such as policy gradients [45] or entropy-regularized objectives to mitigate overoptimization [46][47][48]. These RL-based methods are especially useful when the reward model is non-differentiable but may miss gradient signals when available. More recent methods enable direct backpropagation of reward gradients through the generative process [49,50]. Alternatively, several works [20,51,52] adopt a stochastic optimal control (SOC) perspective, deriving closed-form optimal drift and initial distributions using pathwise KL objectives. While fine-tuning-based methods are an appealing approach, they have practical limitations in that they necessitate costly retraining whenever changes are made to the reward function or the pretrained model. Further, it has been shown that fine-tuning-based methods exhibit mode-seeking behavior [12], which leads to low diversity in the generated samples.
this section cite: ['b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b19', 'b49', 'b50', 'b11']

Section: Problem Definition & Background

this section cite: []

Section: Background: Score-Based Generative Models
Given a standard Gaussian distribution p 1 = N (0, I) and data distribution p 0 , score-based generative models are trained to estimate the score function, which is the gradient of log-density, at intermediate distributions p t along a probability path connecting p 1 to p 0 . In score-based generative models [3][4][5][6][7], the data generation process is typically described by a reverse-time stochastic differential equation (SDE) [4]:
dx t = f (x t , t)dt + g(t)dW, f (x t , t) = u(x t , t) - g(t) 2 2 ∇ log p t (x t ), x 1 ∼ p 1 (1
)
where f (x t , t) and g(t) denote the drift and diffusion coefficients, respectively, and W is a ddimensional standard Brownian motion. The term u(x t , t) corresponds to the velocity field in flow-based model [53,27,54] and also corresponds to the drift term of the probability flow ODE (PF-ODE) in diffusion models [4]. We assume that the generation process proceeds in decreasing time, i.e., from t = 1 to t = 0, following the convention commonly adopted in the score-based generative modeling literature [4,5]. The deterministic flow-based generative model can be recovered by setting the diffusion coefficient g(t) = 0, thereby reducing the SDE to an ODE. Note that flow-based models [53,27], originally formulated as an ODE, can be extended to an SDE formulation that shares the same intermediate distributions p t , thereby allowing stochasticity to be introduced during generation [7,6,13]. Moreover, the velocity field u(x t , t) can be readily transformed into a score function [7,54]. For these reasons, we categorize both diffusion and flow-based models as the score-based generative models.
this section cite: ['b2', 'b3', 'b4', 'b5', 'b6', 'b3', 'b51', 'b26', 'b52', 'b3', 'b3', 'b4', 'b51', 'b26', 'b6', 'b5', 'b12', 'b6', 'b52']

Section: Inference-Time Reward Alignment Using Score-Based Generative Models
Inference-time reward alignment [8,[12][13][14][15][16][17] aims to generate high-reward samples x 0 ∈ R d without fine-tuning the pretrained score-based generative model. The reward associated with each sample is evaluated using a task-specific reward function r : R d → R, which may quantify aspects such as aesthetic quality or the degree to which a generated image satisfies user-specified conditions. But to avoid over-optimization [46,20,12] with respect to the reward function, which may lead to severe distributional drift or adversarial artifacts, a regularization term is introduced to encourage the generated samples to remain close to the prior of the pre-trained generative model. This trade-off is captured by defining a target distribution p * 0 that balances reward maximization with prior adherence, formally expressed as:
p * 0 = arg max q E x0∼q [r(x 0 )] (a) -α D KL [q∥p 0 ] (b) .(2)
Here, term (a) in Eq. 2 encourages the generation of high-reward samples, while term (b), the KL-divergence, enforces proximity to the pre-trained model's prior distribution p 0 . The parameter α ∈ R + controls the strength of this regularization: larger values of α lead to stronger adherence to the prior, typically resulting in lower reward but higher proximity to the support of the generative model.
The target distribution p * 0 has a closed-form expression, given by:
p * 0 (x 0 ) = 1 Z 0 p 0 (x 0 ) exp r(x 0 ) α (3
)
where Z 0 is normalizing constant. Detailed derivation using calculus of variations can be found in Kim et al. [13]. This reward-aware target distribution has been widely studied in the reinforcement learning literature [55][56][57][58][59]. Analogous ideas have also been adopted to fine-tuning score-based generative models [60, 51, 52, 45, 48-50, 46, 43, 20]. As in our case, this target distribution also serves as the objective from which one aims to sample in inference-time reward alignment task [8,12,13].
Since sample generation in score-based models proceeds progressively through a sequence of timesteps, it becomes important to maintain proximity with the pretrained model not just at the endpoint, but throughout the entire generative trajectory. To account for this, the original objective in Eq. 2 is extended to a trajectory-level formulation. Although there are some works [50,46,45] that frame this problem as entropy-regularized Markov Decision Process (MDPs), where each denoising step of score-based generative model corresponds to a policy in RL, we adopt a stochastic optimal control (SOC) perspective [20,51,52], which naturally aligns with the continuous-time structure of score-based generative models and yields principled expressions for both the optimal drift and the optimal initial distribution.
Building on this, the entropy-regularized SOC framework proposed by Uehara et al. [20] provides closed-form approximations for the optimal initial distribution, optimal control function, and optimal transition kernel that together enable sampling from the reward-aligned target distribution defined in Eq. 3 using score-based generative models. The optimal initial distribution can be derived using the Feynman-Kac formula and approximated via Tweedie's formula [18] as:
p * 1 (x 1 ) := 1 Z 1 p 1 (x 1 ) exp r(x 0|1 ) α(4)
where x 0|t := E x0∼p 0|t [x 0 ] denotes Tweedie's formula [18], representing the conditional expectation under p 0|t := p(x 0 |x t ). Under the same approximation, the transition kernel satisfying the optimality condition is approximated by:
p * θ (x t-∆t |x t ) = exp(r(x 0|t-∆t )/α) exp(r(x 0|t )/α) p θ (x t-∆t |x t ).(5)
where p θ (x t-∆t |x t ) is a transition kernel of the pretrained score-based generative model. Further details on the SOC framework and its theoretical foundations in the context of reward alignment are provided in the Appendix A.
this section cite: ['b7', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b44', 'b19', 'b11', 'b12', 'b53', 'b54', 'b55', 'b56', 'b57', 'b7', 'b11', 'b12', 'b48', 'b44', 'b43', 'b19', 'b49', 'b50', 'b19', 'b17', 'b17']

Section: Sequential Monte Carlo (SMC) with Denoising Process
For reward-alignment tasks, recent works [12,15,14,21,22] have demonstrated that Sequential Monte Carlo (SMC) can efficiently generate samples from the target distribution in Eq. 3. When applied to score-based generative models, the denoising process is coupled with the sequential structure of SMC. Specifically, several prior works [8, 12, 14] adopt Eq. 5 as the intermediate target transition kernel for sampling from Eq. 3. In general, SMC methods [9-11] are a class of algorithms for sampling from sequences of probability distributions. Starting from K particles sampled i.i.d. from the initial distribution, SMC approximates a target distribution by maintaining a population of K weighted particles, which are repeatedly updated through a sequence of propagation, reweighting, and resampling steps. The weights are updated over time according to the following rule: w (i) t-∆t = p tar (x t-∆t |x t ) q(x t-∆t |x t ) w (i) t
where p tar is an intermediate target kernel we want to sample from, and q(x t-∆t |x t ) is a proposal kernel used during propagation. As the number of particles K increases, the approximation improves due to the asymptotic consistency of the SMC framework [61,62].
Following [8,12,14], which derives both the intermediate target transition kernel and the associated proposal for reward-guided SMC, we compute the weight at each time step as:
w (i) t-∆t = exp(r(x 0|t-∆t )/α)p θ (x t-∆t |x t ) exp(r(x 0|t )/α)q(x t-∆t |x t ) w (i) t ,(7)
where p tar is set as Eq. 5. The proposal distribution q(x t-∆t |x t ) is obtained by discretizing the reverse-time SDE with an optimal control. This yields the following proposal with the Tweedie's formula [18]:
q(x t-∆t |x t ) = N (x t -f (x t , t)∆t + g 2 (t)∇ r(x 0|t ) α ∆t, g(t) 2 ∆tI).(8)
Details on SMC and its connection to reward-guided sampling are provided in the Appendix B.
this section cite: ['b11', 'b14', 'b13', 'b20', 'b21', 'b59', 'b60', 'b7', 'b11', 'b13', 'b17']

Section: Limitations of Previous SMC-Based Reward Alignment Methods
While prior work [12,15,14,21,22] has demonstrated the effectiveness of SMC in inference-time reward alignment, these approaches typically rely on sampling initial particles from the standard Gaussian prior. We argue that sampling particles directly from the posterior in Eq. 4, rather than the prior, is essential for better high-reward region coverage and efficiency in SMC. First, the effectiveness of the SMC proposal distribution Eq. 8 diminishes over time making it increasingly difficult to guide particles toward high-reward regions in later steps. As the diffusion coefficient g(t) 2 → 0 as t → 0, it weakens the influence of the reward signal ∇r(x 0|t ), since it is scaled by g 2 (t) in the proposal. Second, the initial position of particles becomes particularly critical when the reward function is highly non-convex and multi-modal. While the denoising process may, in principle, help particles escape local modes and explore better regions, this becomes increasingly difficult over time, not only due to the vanishing diffusion coefficient, but also because the intermediate distribution becomes less perturbed and more sharply concentrated, reducing connectivity between modes [63]. In contrast, at early time steps (e.g., t = 1), the posterior distribution is more diffuse and better connected across modes, enabling more effective exploration. Furthermore, recent score-based generative models distilled for trajectory straightening have made the approximation of the optimal initial distribution in Eq. 4 sufficiently precise. These observations jointly motivate allocating computational effort to obtaining high-quality initial particles that are better aligned with the reward signal.
this section cite: ['b11', 'b14', 'b13', 'b20', 'b21', 'b61']

Section: Ψ-Sampler: pCNL-Based Initial Particle Sampling
In this work, we propose Ψ-Sampler, a framework that combines efficient initial particle samping with SMC-based inference-time reward alignment for score-based generative models. The initial particles are sampled using the Preconditioned Crank-Nicolson Langevin (pCNL) algorithm, hence the name PCNL-based initial particle sampling followed by SMC-based Inference-time reward alignment.
The key idea is to allocate computational effort to the initial particle selection by sampling directly from the posterior distribution defined in Eq. 4. This reward-informed initialization ensures that the particle set is better aligned with the target distribution from the outset, resulting in improved sampling efficiency and estimation accuracy in the subsequent SMC process.
While the unnormalized density of the posterior distribution in Eq. 4 has an analytical form, drawing exact samples from it remains challenging. A practical workaround is to approximate posterior sampling via a Top-K-of-N strategy: generate N samples from the prior, and retain the top K highest-scoring samples as initial particles. This variant of Best-of-N [64][65][66] resembles rejection sampling and serves as a crude approximation to posterior sampling [67,68]. We find that even this simple selection-based approximation leads to meaningful improvements. But considering that sampling space is high-dimensional, one can adopt Markov Chain Monte Carlo (MCMC) [30][31][32][33][34][35] which is known to be effective at sampling from high-dimensional space. In what follows, we briefly introduce Langevin-based MCMC algorithms that we adopt for posterior sampling.
this section cite: ['b62', 'b63', 'b64', 'b65', 'b66', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33']

Section: Background: Langevin-Based Markov Chain Monte Carlo Methods
Langevin-based MCMC refers to a class of samplers that generate proposals by discretizing the Langevin dynamics, represented as stochastic differential equation (SDE),
dx = 1 2 ∇ log p tar x dt + dW,(9)
whose stationary distribution is the target density p tar . A single Euler-Maruyama discretization of the Langevin dynamics with step size ϵ > 0 produces the proposal
x ′ = x + ϵ 2 ∇ log p tar (x) + √ ϵ z, z ∼ N (0, I).(10)
Accepting every proposal yields the Unadjusted Langevin Algorithm (ULA) [31]. As a result, the Markov chain induced by ULA converges to a biased distribution whose discrepancy arises from the discretization error. In particular, since ULA does not include a correction mechanism, it does not guarantee convergence to the target distribution p tar . Metropolis-Adjusted Langevin Algorithm (MALA) [31,32] combines the Langevin proposal Eq. 10 with the Metropolis-Hastings (MH) [69,30] correction, a general accept-reject mechanism that eliminates discretization bias. Given the current state x and a proposal x ′ ∼ q(x ′ |x), the move is accepted with probability:
a M (x, x ′ ) = min 1, p tar (x ′ ) q(x|x ′ ) p tar (x) q(x ′ |x) .(11)
This rule enforces detailed balance, so p tar is an invariant distribution of the resulting Markov chain.
While MALA is commonly used in practice due to its simplicity and gradient-based efficiency, it becomes increasingly inefficient in extremely high-dimensional settings, as is typical in image generative models (e.g., 65,536 for FLUX [28]). With a fixed step size, its acceptance probability degenerates as d → ∞. Theoretically, to maintain a reasonable acceptance rate, the step size must shrink with dimension, typically at the optimal rate of O(d -1/3 ) [70], which leads to extremely slow mixing and inefficient exploration in extremely high-dimension space.
this section cite: ['b29', 'b29', 'b30', 'b67', 'b28', 'b68']

Section: Preconditioned Crank-Nicolson Langevin (pCNL) Algorithm
To address high-dimensional sampling challenges (Sec. 4.1), infinite-dimensional MCMC methods [36][37][38] were developed, particularly for PDE-constrained Bayesian inverse problems. These methods remain well-posed even when dimensionality increases. Among them, the preconditioned Crank-Nicolson (pCN) algorithm offers a simple, dimension-robust alternative to Random Walk Metropolis (RWM), though it fails to leverage the potential function, limiting its efficiency.
To overcome this limitation, the preconditioned Crank-Nicolson Langevin (pCNL) algorithm has been proposed [36,37], which augments the dimension-robustness of pCN with the gradient-informed dynamics of Langevin methods (Eq. 9), thereby improving sampling efficiency in high-dimensional settings. The pCNL algorithm employs a semi-implicit Euler (Crank-Nicolson-type) discretization of Langevin dynamics as follows:
x ′ = x + ϵ 2 - x + x ′ 2 + ∇ r(x 0|1 ) α + √ ϵ z, z ∼ N (0, I). (12
)
assuming prior is N (0, I) as in our case. This Crank-Nicolson update admits an explicit closed-form solution, and hence retains the dimension-robustness of pCN, only when the drift induced by the prior is linear, as with a standard Gaussian prior. Therefore, in our setting, it is applicable only at t = 1, making it a particularly useful method that aligns with our proposal to sample particles from the posterior distribution Eq. 4, where the prior is the standard Gaussian. With ρ = (1 -ϵ/4)/(1 + ϵ/4), we can rewrite above equation as:
x ′ = ρx + 1 -ρ 2 z + √ ϵ 2 ∇ r(x 0|1 ) α , z ∼ N (0, I). (13
)
Note that pCNL also adopts MH correction in Eq. 11 to guarantee convergence to the correct target distribution. The pCN algorithm maintains a well-defined, non-zero acceptance probability even in the infinite-dimensional limit, allowing the use of fixed step sizes regardless of the dimension d [36,37]. This property stems from its prior-preserving proposal, which ensures that the Gaussian reference measure is invariant under the proposal mechanism. This robustness carries over to pCNL, whose proposal inherits pCN's ability to handle Gaussian priors in a dimension-independent manner. We include the detailed acceptance probability formulas for MALA and pCNL in the Appendix C.
this section cite: ['b34', 'b35', 'b36', 'b34', 'b35', 'b34', 'b35']

Section: Initial Particle Sampling
To sample initial particles using MCMC for the subsequent SMC process, we follow standard practices to ensure effective mixing and reduce sample autocorrelation. Specifically, we discard the initial portion of each chain as burn-in [71] and apply thinning by subsampling at fixed intervals to mitigate high correlation between successive samples. A constant step size is used across iterations.
Although adaptive step size schemes may improve convergence, we opt for a fixed-step approach for simplicity. Once the initial particles are sampled, we apply the existing SMC-based method [14,8].
this section cite: ['b69', 'b13', 'b7']

Section: Comparison of SMC Initialization in a Toy

this section cite: []

Section: Experiments

this section cite: []

Section: Experiment Setup
We validate our approach across three applications: layout-to-image generation, quantity-aware generation, and aesthetic-preference image generation. In our experiments, the held-out reward refers to an evaluation metric that is not accessible during generation and is used solely to assess the generalization of the method. Full details for each application are provided in Appendix D. For the layout-to-image generation task, where the goal is to place user-specified objects within designated bounding boxes, we use predicted bounding box information from a detection model [72] and define the reward as the mean Intersection-over-Union (mIoU) between the predicted and target bounding boxes. For the quantity-aware image generation task, which involves generating a userspecified object in a specified quantity, we use the predicted count from a counting model [73] and define the reward as the negative smooth L1 loss between the predicted and target counts. In both tasks, we include evaluations using held-out reward models to assess generalization. Specifically, for layout-to-image generation, we report mIoU evaluated with a different detection model [74] (held-out reward model); for quantity-aware image generation, we report mean absolute error (MAE) and counting accuracy using an alternative counting model [75] (held-out reward model). For aesthetic-preference image generation task, which aims to produce visually appealing images, we use an aesthetic score prediction model [76] as the reward model and use its predicted score as the reward. Across all applications, we further evaluate the generated images using ImageReward [77] Single Particle SMC-Based Methods
this section cite: ['b70', 'b71', 'b72', 'b73', 'b74', 'b75']

Section: Tasks Metrics
Sampling from Prior Sampling from Posterior DPS [39] FreeDoM [40] TDS [14] DAS [12] Top-K-of-N ULA MALA Ψ-SAMPLER
this section cite: ['b37', 'b38', 'b13', 'b11']

Section: Layout to Image
GroundingDINO and VQAScore [78], which assess overall image quality and text-image alignment. The baselines and our methods are categorized into three groups:
• Single-Particle: DPS [39] and FreeDoM [40] are methods not based on SMC but instead use a single particle trajectory and perform gradient ascent. They are limited in scaling up the search space due to the use of a single particle.
• SMC & Initial Particles from Prior: TDS [14] is the SMC-based method we take as the base for our methods. DAS [12] is a variant introducing tempering strategy.
• SMC & Initial Particles from Posterior: We evaluate four posterior-based initialization strategies: Top-K-of-N , ULA, MALA, and Ψ-SAMPLER. ULA and MALA use a small step size (0.05) to ensure non-zero acceptance, while Ψ-SAMPLER employs a larger step size (0.5) for improved performance (See Sec. 5.4). We use 25 denoising steps for SMC-based methods and 50 for single-particle methods to compensate for their limited exploration. For SMC-based methods, we match the total number of function evaluations (NFE) across all methods, allocating half of the budget to initial particle sampling for posterior-based methods. We use FLUX [28] as the pretrained score-based generative model. Full experimental details are provided in Appendix D.
this section cite: ['b76', 'b37', 'b38', 'b13', 'b11']

Section: Quantitative Results
We present quantitative results in Tab.1. Across all tasks, Ψ-SAMPLER consistently achieves the best performance on the given reward and strong generalization to held-out rewards. For SMC-based methods, sampling particles from the posterior distribution yields significant improvements over those that sample directly from the prior, highlighting the importance of posterior-informed initialization. This improvement is particularly notable in complex tasks where high-reward outputs are rare, such as layout-to-image generation and quantity-aware generation. For example, in quantity-aware generation, negative smooth L1 loss improves from 1.804 with TDS (base SMC) to 1.077 with Top-K-of-N and further to 0.850 with our Ψ-SAMPLER. Similarly, for layout-to-image generation, mIoU increases from 0.417 (TDS) to 0.425 with Top-K-of-N and 0.467 with Ψ-SAMPLER. In contrast, initializing with ULA or MALA yields only marginal gains or even degraded performance, due to the lack of Metropolis-Hastings correction in ULA and the limited exploration capacity of MALA in high-dimensional spaces. Single-particle methods consistently underperform compared to SMC-based methods.
this section cite: []

Section: Ablation Study.
We conduct an ablation study that examines how performance varies under different allocations of a fixed total NFE between the initial particle sampling stage (via Top-K-of-K or MCMC) and the subsequent SMC stage; full results and analysis are provided in Appendix G.
this section cite: []

Section: Additional Results
Conducted with Other Score-Based Generative Models. We additionally provide quantitative and qualitative results on all three applications using another score-based generative model, SANA-Sprint [29] in Appendix H.
FreeDoM [40] TDS [14] DAS [12] Top-K-of-N ULA MALA Ψ-SAMPLER
this section cite: ['b27', 'b38', 'b13', 'b11']

Section: Layout-to-Image
"A person is sitting on a chair and a bird is on top of a horse while horse is on the top of a car."
"An airplane and a balloon are on the ground with a horse and a car in the sky."
Quantity-Aware "82 blueberries"
21 (∆61)66
(∆16) 62 (∆20) 73 (∆9) 71 (∆11) 63 (∆19) 82 (∆0) "33 coins" 16 (∆17) 39 (∆6) 38 (∆5) 30 (∆3) 22 (∆11) 35 (∆2) 33 (∆0) Aesthetic 6.351 7.030 6.815 6.925 6.974 7.012 7.423 5.836 7.093 7.161 7.029 7.161 7.098 7.279 Figure 2: Qualitative results for each application demonstrate that Ψ-SAMPLER consistently generates images aligned with the given conditions. Detailed analysis of each case is provided in Sec. 5.3. Acceptance GroundingDINO [72] Salience DETR [74] LPIPS MPD [79]
this section cite: []

Section: Qualitative Results
We additionally present qualitative results for each application in Fig. 2. For the layout-to-image generation task, we display the input bounding box locations alongside the corresponding phrases from the text prompt, using matching colors for each phrase and its associated bounding box. In the quantity-aware image generation task, we overlay the predicted object centroids-obtained from a held-out counting model [75]-to facilitate visual comparison. Below each image, we display the predicted count along with the absolute difference from the target quantity, formatted as (∆•).
The best-performing case is highlighted in blue. For the aesthetic preference task, we display the generated images alongside their predicted aesthetic scores. The first row corresponds to the prompt "Tiger", and the second to "Rabbit". As shown, Ψ-SAMPLER produces high-quality results across all applications, matching the trends observed in the quantitative evaluations. From the first and second rows of Fig. 2, we observe that baseline methods often fail to place objects correctly within the specified bounding boxes or generate them in entirely wrong locations. For instance, in the first row, most baselines fail to position the bird accurately, and in the second row, none correctly place the car. For quantity-aware generation, the fourth row shows the counted results corresponding to the third row. While Ψ-SAMPLER successfully generates the target number of blueberries in an image, the baselines exhibit large errors-Top-K-of-N comes closest but still misses some. In rows 5 and 6, only Ψ-SAMPLER correctly generates the target number of coins. In the aesthetic preference task, although all methods produce realistic images, Ψ-SAMPLER generates the most visually appealing image with the highest aesthetic score. Additional qualitative examples are provided in the Appendix I.
this section cite: ['b73']

Section: Evaluation of Initial Particles
In Fig. 4, we compare MALA and pCNL on the layout-to-image generation task across varying step sizes using four metrics: acceptance probability, reward (mIoU via GroundingDINO [72]), held-out reward (Salience DETR [74]), and sample diversity (LPIPS MPD [79]). All metrics are directly computed from the Tweedie estimates [18] of MCMC samples, before the SMC stage. As the step size increases, MALA's acceptance probability rapidly drops to near-zero, while pCNL maintains stable acceptance probability. Larger step sizes generally improve reward scores, with performance tapering off at excessively large steps. Held-out reward trends mirror this pattern, suggesting that the improvements stem from genuinely higher-quality samples rather than reward overfitting [60,20].
Although LPIPS MPD slightly declines with increasing step size due to reduced acceptance, pCNL at step size 2.0 maintains diversity on par with MALA at 0.05. Additional results for other tasks are included in the Appendix F.
this section cite: ['b70', 'b72', 'b77', 'b17', 'b58', 'b19']

Section: Conclusion and Limitation
We present a novel approach for inference-time reward alignment in score-based generative models by initializing SMC particles from the reward-aware posterior distribution. To address the challenge of high-dimensional sampling, we leverage the preconditioned Crank-Nicolson Langevin (pCNL) algorithm. Our method consistently outperforms existing baselines across tasks and reward models, demonstrating the effectiveness of posterior-guided initialization in enhancing sample quality under fixed compute budgets.
this section cite: []

Section: Limitations and Societal Impact.
A limitation of our approach is that it assumes access to differentiable reward models and depends on accurate Tweedie approximations at early denoising steps. Also, while our method improves fine-grained control in generative modeling, it may also be misused to produce misleading or harmful content, such as hyper-realistic fake imagery. These risks highlights the importance of responsible development and deployment practices, including transparency, content verification, and appropriate use guidelines.
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
this section cite: []

Section: IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification:
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
Answer: [NA] Justification: We do not contain new theorems rather provide toy experiments results to support our claim as well as main experiment results.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: Our datasets are based on a combination of existing resources, including a subset of the HRS-Spatial dataset [80], GPT-4o-generated samples [81], and the animal dataset from [45]. We plan to release both the code and data in future revision. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: ['b78', 'b79', 'b43']

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: Due to limited time and computational resources, we were unable to conduct a full statistical significance analysis prior to submission. However, we plan to incorporate appropriate statistical evaluations, such as standard deviations and confidence intervals, as well as additional ablation studies in a future revision.
Guidelines:
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification:
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification:
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
Answer: [NA] Justification: Our work does not involve the release of any models or datasets that pose a high risk of misuse. The models used in our experiments are based on publicly available architectures, and our dataset construction process avoids the inclusion of any sensitive or harmful content. Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: Although our dataset construction includes a set of GPT-4o-generated samples, which may be considered a new asset, this portion has not been publicly released at the time of submission. As such, we do not consider any part of our submission to constitute a released new asset for the purposes of this checklist item. We plan to release all new data in a future revision.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning Year: (2025)
Ref_id:b2 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b3 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b4 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b5 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b6 Title: SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b7 Title: Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review Year: (2025)
Ref_id:b8 Title: Sequential Monte Carlo methods in practice Year: (2001)
Ref_id:b9 Title: Sequential monte carlo samplers Year: (2006)
Ref_id:b10 Title: An introduction to sequential Monte Carlo Year: (2020)
Ref_id:b11 Title: Test-time alignment of diffusion models without reward over-optimization Year: (2025)
Ref_id:b12 Title: Inference-time scaling for flow models via stochastic generation and rollover budget forcing Year: (2025)
Ref_id:b13 Title: Practical and asymptotically exact conditional sampling in diffusion models Year: (2023)
Ref_id:b14 Title: Diffusion posterior sampling for linear inverse problem solving: A filtering perspective Year: (2024)
Ref_id:b15 Title: CoDe: Blockwise control for denoising diffusion models Year: (2025)
Ref_id:b16 Title: Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding Year: (2024)
Ref_id:b17 Title: Tweedie's formula and selection bias Year: (2011)
Ref_id:b18 Title: Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review Year: (2024)
Ref_id:b19 Title: Fine-tuning of continuous-time diffusion models as entropy-regularized control Year: (2024)
Ref_id:b20 Title: Diffusion probabilistic modeling of protein backbones in 3D for the motif-scaffolding problem Year: (2023)
Ref_id:b21 Title: Monte Carlo guided denoising diffusion models for Bayesian linear inverse problems Year: (2024)
Ref_id:b22 Title: Consistency Models Year: (2023)
Ref_id:b23 Title: Consistency Trajectory Models: Learning probability flow ODE trajectory of diffusion Year: (2024)
Ref_id:b24 Title: Simplifying, stabilizing and scaling continuous-time consistency models Year: (2025)
Ref_id:b25 Title: Fast high-resolution image synthesis with latent adversarial diffusion distillation Year: (2024)
Ref_id:b26 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b27 Title: One-step diffusion with continuous-time consistency distillation Year: (2025)
Ref_id:b28 Title: Monte carlo sampling methods using markov chains and their applications Year: (1970)
Ref_id:b29 Title: Exponential convergence of langevin distributions and their discrete approximations Year: (1996)
Ref_id:b30 Title: Langevin diffusions and metropolis-hastings algorithms Year: (2002)
Ref_id:b31 Title: Hybrid monte carlo Year: (1987)
Ref_id:b32 Title: Riemann manifold langevin and hamiltonian monte carlo methods Year: (2011)
Ref_id:b33 Title: Handbook of Markov Chain Monte Carlo Year: (2011-05)
Ref_id:b34 Title: MCMC methods for functions: Modifying old algorithms to make them faster Year: (2013-08)
Ref_id:b35 Title: Geometric MCMC for infinite-dimensional inverse problems Year: (2017-04)
Ref_id:b36 Title: MCMC methods for diffusion bridges Year: (2008)
Ref_id:b37 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b38 Title: FreeDoM: Training-free energyguided conditional diffusion model Year: (2023)
Ref_id:b39 Title: Universal guidance for diffusion models Year: (2024)
Ref_id:b40 Title: TFG: Unified training-free guidance for diffusion models Year: (2024)
Ref_id:b41 Title: Aligning text-to-image models using human feedback Year: (2023)
Ref_id:b42 Title: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis Year: (2023)
Ref_id:b43 Title: Training diffusion models with reinforcement learning Year: (2024)
Ref_id:b44 Title: Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models Year: (2023)
Ref_id:b45 Title: Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b46 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b47 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2024)
Ref_id:b48 Title: Aligning text-to-image diffusion models with reward backpropagation Year: (2024)
Ref_id:b49 Title: Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond Year: (2024)
Ref_id:b50 Title: Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control Year: (2025)
Ref_id:b51 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b52 Title: Flow matching guide and code Year: (2024)
Ref_id:b53 Title: Reinforcement learning with deep energy-based policies Year: (2017)
Ref_id:b54 Title: A theory of regularized markov decision processes Year: (2019)
Ref_id:b55 Title: Equivalence between policy gradients and soft q-learning Year: (2017)
Ref_id:b56 Title: A unified view of entropy-regularized markov decision processes Year: (2017)
Ref_id:b57 Title: Reinforcement learning and control as probabilistic inference: Tutorial and review Year: (2018)
Ref_id:b58 Title: Bridging model-based optimization and generative modeling via conservative fine-tuning of diffusion models Year: (2024)
Ref_id:b59 Title: Central limit theorem for sequential monte carlo methods and its application to bayesian inference Year: (2004-12)
Ref_id:b60 Title: Feynman-Kac formulae: genealogical and interacting particle systems with applications Year: (2004)
Ref_id:b61 Title: Denoising MCMC for accelerating diffusion-based generative models Year: (2022)
Ref_id:b62 Title: Scaling laws for reward model overoptimization Year: (2023)
Ref_id:b63 Title: BoNBon alignment for large language models and the sweetness of best-of-n sampling Year: (2024)
Ref_id:b64 Title: Browser-assisted question-answering with human feedback Year: (2022)
Ref_id:b65 Title: Chirag Nagpal, and Ananda Theertha Suresh. Theoretical guarantees on the best-of-n alignment policy Year: (2024)
Ref_id:b66 Title: Asymptotics of language model alignment Year: (2024)
Ref_id:b67 Title: Equation of state calculations by fast computing machines Year: (1953)
Ref_id:b68 Title: Optimal scaling of discrete approximations to langevin diffusions Year: (1998)
Ref_id:b69 Title: Practical markov chain monte carlo Year: (1992)
Ref_id:b70 Title: Grounding DINo: Marrying DINO with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b71 Title: T2ICount: Enhancing cross-modal understanding for zero-shot counting Year: (2025)
Ref_id:b72 Title: Salience DETR: Enhancing detection transformer with hierarchical salience filtering refinement Year: (2024)
Ref_id:b73 Title: CountGD: Multi-modal open-world counting Year: (2024)
Ref_id:b74 Title: Laion aesthetic predictor Year: (2022)
Ref_id:b75 Title: ImageReward: learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b76 Title: Evaluating text-to-visual generation with image-to-text generation Year: (2024)
Ref_id:b77 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b78 Title: HRS-Bench: Holistic, reliable and scalable benchmark for text-to-image models Year: (2023)
Ref_id:b79 Title: GPT-4o system card Year: (2024)
Ref_id:b80 Title: Gligen: Open-set grounded text-to-image generation Year: (2023)
Ref_id:b81 Title: R&b: Region and boundary aware zero-shot grounded text-to-image generation Year: (2024)
Ref_id:b82 Title: Grounded text-to-image synthesis with attention refocusing Year: (2023)
Ref_id:b83 Title: Groundit: Grounding diffusion transformers via noisy patch transplantation Year: (2024)
Ref_id:b84 Title: Hyung Il Koo, and Nam Ik Cho. Counting guidance for high fidelity text-to-image synthesis Year: (2025)
Ref_id:b85 Title: Make it count: Text-to-image generation with an accurate number of objects Year: (2024)
Ref_id:b86 Title: Iterative object count optimization for text-to-image diffusion models Year: (2024)
