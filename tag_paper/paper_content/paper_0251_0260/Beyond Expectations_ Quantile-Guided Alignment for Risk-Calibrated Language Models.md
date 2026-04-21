Title: Beyond Expectations: Quantile-Guided Alignment for Risk-Calibrated Language Models
Abstract: Large language models can generate rare but catastrophic outputs, such as harmful conversations or insecure code. Existing Reinforcement Learning from Human Feedback (RLHF) typically maximizes average reward, leaving high-risk tail events insufficiently controlled. We introduce Quantile-Guided Alignment (QA), a framework that allows users to specify desired improvements at any quantile-individually or across multiple reward dimensions-thus shifting the distribution of outputs with finer control toward safer, more desirable outcomes. The method extends standard RLHF via an augmented reward formulation that enforces quantile constraints. Experiments on conversation and code-generation tasks show that quantile alignment significantly enhances quality at targeted tails while maintaining overall performance. The results position QA as a principled route to risk-calibrated language models with tail-focused alignment.

Section: Introduction
Large language models (LLMs) have achieved remarkable results in language understanding [1][2][3][4], code generation [5,6], and agentic decision-making tasks [7]. However, they also pose safety challenges when a small fraction of their outputs can be harmful or catastrophic. Standard Reinforcement Learning from Human Feedback (RLHF) [8,9] optimizes expected reward to improve average performance, but it may still allow rare, high-impact failures to persist. In safety-critical scenarios, ranging from AI personal assistants [10] to security-sensitive code generation [11], optimizing only the mean behavior is insufficient; one must also address specific quantiles of the distribution (e.g., its extreme tails) where dangerous errors occur.
For instance, consider a personal assistant handling content-sensitive consultations or a codegeneration model that sometimes produces unreliable snippets. Although standard RLHF may reduce errors overall, it can fail to eliminate occasional yet severe missteps. This motivates quantile alignment, which explicitly enforces constraints on the fraction of outputs that violate certain thresholds, effectively controlling the distributional tails (or any other critical quantiles) of relevant reward metrics. For example, requiring "90% of scheduling decisions meet a minimum standard" ensures the assistant rarely overlooks urgent tasks; similarly, imposing "99% of code must remain below a given harmfulness score" mitigates security exploits or unauthorized system calls.
We formulate quantile alignment as a constrained KL-regularization objective that minimizes the KL divergence from a reference model subject to user-defined quantile constraints, thereby limiting how many outputs fall below-or above-critical values. We prove that the resulting optimization is convex and admits a Lagrangian dual, whose solution reweights the original model distribution by a linear combination of indicator-based quantile rewards. This structure closely parallels RLHF, replacing the usual scalar reward with quantile-adapted terms. In practice, we solve the dual numerically using Monte Carlo approximations, enabling an efficient and flexible way to regulate model outputs across multiple quantiles and metrics. The main contributions of the paper are twofold: Quantile-Guided Alignment (QA) Framework. We introduce quantile-guided alignment, a principled way to impose distribution-tail constraints on multiple reward dimensions (e.g. harmlessness, helpfulness). Rather than maximising the mean reward, QA guarantees that a user-specified fraction of outputs satisfies strict thresholds, thereby mitigating rare but critical failures. Each quantile constraint is encoded as an indicator-based reward, leading to a convex KL-regularized objective with a primal-dual solution whose dimensionality equals the number of constraints.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10']

Section: RLHF-Compatible Implementation and Empirical Validation.
Because the dual problem is finite-dimensional, QA drops straight into the standard RLHF tool-chain (PPO, TRL, etc.) after substituting the composite reward. Experiments on conversational and code-generation benchmarks show that QA sharply improves tail-risk metrics-e.g. reducing the worst 5% of harmful outputs-while preserving diversity, coherence, and perplexity.
this section cite: []

Section: Related work
Reinforcement Learning from Human Feedback (RLHF). Language model alignment strategies can be categorized into training-based and decoding-based approaches. Training-based methods adjust model parameters through fine-tuning guided by human feedback, whereas decoding-based methods constrain or steer the models outputs at inference time without retraining. Within trainingbased solutions, a prominent technique is RLHF [12][13][14] that involves two main steps. First, a reward model is trained to map each candidate output to a scalar that reflects human preferences. Typically, a dataset of comparisons (x, y lose , y win ) is collected from human annotators, where x is a prompt and two different responses y lose and y win receive preference labels indicating y win is preferred over y lose . Then, one uses the reward model r to maximize the expectation of generated rewards subject to a Kullback-Leibler (KL) regularization. As a result, RLHF nudges the model distribution toward higher-reward (i.e. more human-preferred) outputs.
this section cite: ['b11', 'b12', 'b13']

Section: Multi-dimensional Preference Alignment.
Multi-objective reinforcement learning (MORL) [15][16][17] has been used as a popular approach to address multi-dimensional preference alignment. It often build on the classical RLHF setup [12][13][14]18], using linear scalarization to combine reward signals or data sources [14,19,18,20] to construct a single RLHF objective. For example, the recent work Rewarded Soup [19] attempts to approximate the Pareto frontier of models by fine-tuning separate models for each reward function, then blend these specialized models by interpolating their model weights. In principle, this approximates the ensemble of models that would have emerged from training on every linear combination r = m i=1 λ i r i directly. The recent MAP approach [21] provides an alternative perspective: instead of requiring a single scalar reward, it enforces constraints on the expected values of different reward functions relative to user-specified targets.
this section cite: ['b14', 'b15', 'b16', 'b11', 'b12', 'b13', 'b17', 'b13', 'b18', 'b17', 'b19', 'b18', 'b20']

Section: AI Safety.
Generative AI models are susceptible to adversarial manipulations such as backdoor attacks [22][23][24][25] and jailbreak attacks [26,27], which can lead to severe issues including hallucinations or security breaches [28,29]. These stealthy exploits often arise from deceptive or poisoned training data, compromising model safety even after deployment. In response, efforts in AI safety include inference-time methods such as prompt engineering [30] or detection-based filters [31], and trainingtime adjustments via RLHF alignment schemes that incorporate robustness into reward functions [14].
this section cite: ['b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b13']

Section: Background

this section cite: []

Section: Preliminaries on RLHF
The RLHF framework aligns a language model p 0 with human preferences by solving:
max p E p r(x, y) -β D KL p(• | x) || p 0 (• | x) ,(1)
where p 0 denotes the reference distribution that corresponds to the original model, P denotes the class of all distributions, p is the distribution that represents the aligned model, r is a reward function that quantifies the preference level of any given pair of prompt x and generation y, D KL measures the KL-divergence, and β > 0 is a regularization hyperparameter. The expectation is taken over x ∼ D (prompt set) and y | x ∼ p(• | x) (conditional generation), briefly written as E p . from It can be shown that the solution of (1) can be written in the form of p(y | x) = p 0 (y | x) exp(β -1 r(x, y))/C(β) where C(β) is a normalizing constant for p(• | x) to be a valid probability density function. While this formulation can effectively raise average performance of large language models, it does not inherently control quantiles, especially tail events that lead to rare but potentially damaging outputs.
this section cite: []

Section: Preliminaries on Quantile Constraints
Consider a random variable Z and its τ -th quantile Q τ (Z). By definition, Q τ (Z) satisfies P Z ≤ Q τ (Z) ∆ = τ, where the probability P p is defined under x, y ∼ p. For a reward function r, x ∼ D and y ∼ q(• | x) for some generative model q induce a random variable r(x, y). The τ -quantile of r(x, y) is defined by: Q τ,p (r) ∆ = inf{c : P p (r(x, y) ≤ c) ≥ τ }. We aim to impose constraints on this quantile, for instance Q τ,p (r) ≥ c. Rewriting this, we have:
Q τ,p (r) ≥ c ⇐⇒ P p r(x, y) ≤ c ≤ τ ⇐⇒ E p I r(x, y) ≤ c ≤ τ.
Rearranging this, we obtain the equivalent requirement
E p τ -I{r(x, y) < c} ≥ 0,(2)
where I{ r(x, y) < c } is an indicator. Thus, we define an indicator-based quantile reward:
(x, y) → ρ τ,c (r(x, y)) ∆ = τ -I{ r(x, y) < c },
which can be regarded as a composite function ρ τ,c • r that maps a prompt-generation pair (x, y) to a reward. According to Inequality (2), imposing the expectation of ρ τ,c (r(x, y)) to be nonnegative under the aligned distribution q equivalently ensures Q τ,p (r) ≥ c. This construction translates quantile constraints into inequalities that involve only linear functional in p.
this section cite: []

Section: Quantile Alignment (QA): Formulation, Theory, and Algorithms
We consider the most general setting of aligning a pretrained distribution p to satisfy multiple quantile constraints on multiple reward functions. For clarity, we first present the single human value (i.e. single reward function) scenario where each reward has multiple quantile thresholds, and then generalize to multiple reward functions.
this section cite: []

Section: Single-Value Multi-Quantile Constraints
Let r(x, y) be a single scalar reward function. We want to enforce multiple constraints of the form Q τj ,p r(x, y) ≥ c j ∆ = Q κj ,p0 r(x, y) , j ∈ [m],
where each constraint j stipulates that at most a fraction τ j of samples fall below a threshold, which is the κ j -quantile of the original distribution p 0 . That is, we lift the τ j -th quantile of to the level of the κ j -th quantile. Equivalently,
P p r(x, y) < c j ≤ τ j ⇐⇒ E p ρ τj ,κj (r(x, y)) ≥ 0,
where we define
ρ τj ,κj (r(x, y)) ∆ = τ j -I{ r(x, y) < c j }.(3)
We collect all such constraints in the objective:
min p∈P E p KL p(• | x) ∥ p 0 (• | x) subject to E p ρ τj ,κj r(x, y) ≥ 0, j ∈ [m], (4
)
where we define the indicator-based quantile reward for each constraint j as ρ τj ,κj r(x, y)
∆ = τ j -1 {r(x,y)<cj } .(5)
Remark 4.1 (Connection to the standard RLHF). Recall from the standard RLHF framework that imposing a scalar reward r(x, y) leads to a reweighted distribution p(y | x) ∝ p 0 (y | x) exp β -1 r(x, y) , where β plays the role of a temperature that balances between the original model p 0 (y | x) and the reward r(x, y). As the next theorem shows, our quantile constraints also produce an exponential reweighting, but the "reward" now comprises a sum of indicator-based terms:
p(y | x) ∝ p 0 (y | x) exp m j=1 λ j ρ τj ,κj r(x, y) ,
where each active constraint introduces a learned multiplier λ j ≥ 0. The objective in (4) is a convex program in the space of distributions in p 0 , and each constraint j introduces a dual multiplier λ j ≥ 0 whose values are determined by its constraints. Hence, although the QA-aligned distribution retains the same exponential-family form, ρ τj ,κj differs from the usual reward r whose corresponding multiplier β is tuned as a hyperparameter. Theorem 4.2 (Representation of the Multi-Quantile Solution). The optimization problem (4) is convex in p. There exists a unique m-dimensional vector λ = λ(τ j , κ j , j ∈ [m]) ≥ 0 such that the optimal solution is
p λ (y | x) ∆ = p 0 (y | x) exp m j=1 λ j ρ τj ,κj r(x, y) C(λ ,
where C(λ) is the normalizing constant, and λ = [λ 1 , . . . , λ m ] is determined by
λ(τ j , κ j , j ∈ [m]) = arg max λ ≥ 0 -log C(λ) .
Remark 4.3 (Generality of Quantile Alignment). Theorem 4.2 implies that any distribution satisfying the specified quantile targets can be written in the exponential-family form defined by QA. Conversely, for any aligned distribution p produced by other methods, the pair p 0 , p corresponds to a feasible set of pairs {τ j , κ j }, which can be an infinite set of continuously-valued quantiles (as elaborated in Section 5.1), This shows QA's generality. Notably, the standard RLHF objective (1) can be regarded as a special case of QA.
this section cite: []

Section: Numerically Solving the QA Problem
To solve the QA problem efficiently, we first outline the high-level logic: while the original formulation involves optimizing the infinite-dimensional distribution p, we can transform it into a lower-dimensional convex optimization problem over the dual variables λ. This is because the problem in (4) is convex, and standard convex analysis techniques such as strong duality apply. This reformulation enables tractable optimization of λ by leveraging Monte Carlo estimation. Furthermore, by Theorem 4.2, the role of λ manifests in an exponential reweighting, which can be interpreted as solving a single RLHF problem where the reward is a weighted sum of indicator-based quantile rewards. This enables us to leverage existing RLHF solvers, such as the Proximal Policy Optimization (PPO) algorithm implemented in the TRL package [32].
One may wonder why, from the original QA formulation, it appears that we are optimizing a nonconvex problem when p represents a large-scale language model. The key insight is that if we do not treat the optimization as occurring in the model parameter space, but rather over a generic probability density, we can reframe the problem as a convex optimization over λ, whose dimension equals to the number of quantile constraints.
To solve the dual problem, we first rewrite C(λ) as an expectation under the original model p 0 :
C(λ) = E y∼p0(•|x) exp m j=1 λ j ρ τj ,κj (r(x, y)) .(6)
This allows us to estimate the dual objective via Monte Carlo sampling. Specifically, let {(x ℓ , y ℓ )} n ℓ=1 be i.i.d. samples drawn from p 0 (x, y). Notably, this sample can be used for all various alignment targets. For each sample, we compute the indicator-based quantile rewards: ρ τj ,κj (r(x ℓ , y ℓ )) = τ j -I{r(x ℓ , y ℓ ) < ĉj }, where ĉj is calculated as the κ j -th quantile of {r(x ℓ , y ℓ ), ℓ ∈ [m]}. Using these samples, we approximate the dual objective as:
g(λ) ∆ = -log 1 n n ℓ=1 exp m j=1 λ j ρ τj ,κj (r(x ℓ , y ℓ )) .
Remark 4.4 (Concavity and Convergence). By Theorem 4.2, -log C(λ) is concave in λ. Since the sample average is a special expectation, g(λ) preserves concavity, ensuring that standard gradient ascent methods converge to the global maximum of g(λ). Once we solve for λ * , we obtain R(x, y) ∆ = m j=1 λ * j ρ τj ,κj (r(x, y)). According to Remark 4.1, R can be treated as an effective reward model in the standard RLHF with inverse temperature β = 1, allowing us to directly apply PPO solvers. The accuracy of this numerical solution depends on the number of Monte Carlo samples n. In practice, a few thousand samples typically suffice for stable estimates of g(λ). If user-specified thresholds are infeasible, the procedure detects it via divergence in the dual or violation of positivity constraints. In such cases, we can automatically adjust the thresholds via line search or alternative strategies.
this section cite: ['b31']

Section: Algorithmic Steps.
Next, we summarize the procedure for solving the QA problem numerically.
1. Sampling. Draw n samples {(x ℓ , y ℓ )} n ℓ=1 from p 0 . 2. Compute Indicator-Based Rewards. For each ℓ, evaluate ρ τj ,κj (r(x ℓ , y ℓ )) for each τ j , κ j .
this section cite: []

Section: Dual Update.
Initialize an m-dimensional λ (0) ≥ 0 and perform gradient ascent:
λ (t+1) ← λ (t) + η ∇ λ g(λ (t) ) + ,
until convergence, where (•) + denotes projection onto the nonnegative orthant, and η > 0 is the step size. If it diverges, we decide the constraints are infeasible. 4. Construct QA Reward. Once we obtain the dual solution λ * , compute the effective reward: R(x, y)
∆ = m j=1 λ * j ρ τj ,κj (r(x, y)).
this section cite: []

Section: 5.
Optimize p based on the QA reward. Treat R(x, y) as the reward function in the standard RLHF setting with β = 1 and apply a PPO solver to update from p 0 to p. Remark 4.5. Since the primal-dual method only requires forward passes under p, it does not involve backpropagation through model parameters. The runtime and memory complexity scale as O(n m), where n is the number of MC samples and m is the number of constraints. This setup remains computationally feasible even for large models, as the same set of MC samples can be reused across different quantile constraints without retraining the base model.
this section cite: []

Section: Multi-Value, Multiple-Quantile Alignment
The QA framework readily generalizes to multiple reward functions r 1 (x, y), . . . , r K (x, y), and each may have multiple quantile constraints. That is, for each reward function r i , we impose the constraint
Q τi,j ,p r i (x, y) ≥ Q κi,j ,p0 r i (x, y) , j ∈ [m i ], i ∈ [K],
where τ i,j represents the quantile threshold for p and κ i,j represents the corresponding threshold for p 0 . This ensures that at most a fraction τ i,j of generated samples fall below the κ i,j -quantile of the original model p 0 .
Following the single-reward case, we define the multi-value indicator-based quantile rewards:
ρ τi,j ,κi,j (r i (x, y)) ∆ = τ i,j -I{ r i (x, y) < c i,j },(7)
where c i,j is the empirical κ i,j -quantile of r i (x, y) under p 0 . These constraints are then incorporated into the KL-regularized objective:
min p∈P E p D KL p(• | x) || p 0 (• | x) s.t. E p ρ τi,j ,κi,j r i (x, y) ≥ 0, j ∈ [m i ], i ∈ [K].
The only difference from the single-reward setting is that the exponent in the optimal solution now becomes R(x, y)
∆ = K i=1 mi j=1 λ i,j ρ τi,j ,κi,j (r i (x, y)).
Computationally, the optimization dimensionality is proportional to the total number of constraints m 1 + m 2 + • • • + m K . The numerical solution follows the primal-dual Monte Carlo approach outlined in Section 4.2, with each reward function r i (x, y) contributing its own set of constraints. Thus, the QA framework provides a principled and efficient mechanism to enforce multiple quantile constraints across reward functions.
this section cite: []

Section: Continuous Quantile Alignment
Thus far, we have formulated quantile alignment as a discrete optimization problem, enforcing constraints at specific quantile levels. However, it is both theoretically intriguing and practically helpful to study oversight across a continuum of quantile levels. To generalize our approach, we now extend our framework to enforce continuous quantile constraints, shaping the entire distribution of reward values rather than a finite subset of quantiles.
this section cite: []

Section: Uniform Distribution Enhancement through Continuous Constraints
For notational simplicity, we focus on the a single value represented by a reward function r. There is no essential difference in generalizing to multiple values, as we discussed in Subsection 4.3. Previously, we considered a finite set of quantile constraints at levels {τ j } m j=1 , ensuring that specific quantiles of the reward distribution under q meet or exceed corresponding reference values. We now impose a quantile constraint at every level τ ∈ [0, 1], requiring Q τ,p (r) ≥ c(τ ), where c(τ ) ≥ Q τ,p0 (r) is a target quantile function defined for all τ ∈ [0, 1]. This ensures that the quantile curve of r under q remains above c(τ ) for all τ , uniformly lifting the entire reward distribution to match a desired profile. Rewriting this in expectation form, we obtain:
E p ρ τ (r) ≥ 0, ∀ τ ∈ [0, 1],(8)
where the indicator-based quantile reward is defined as ρ τ (r(x, y)) ∆ = τ -I r(x, y) < c(τ ) . This formulation ensures that instead of controlling individual quantiles, we impose constraints over an entire continuum, creating a smooth and robust enhancement of the reward distribution. This leads to an infinite-dimensional constrained optimization problem:
min p∈P E p D KL p(• | x) || p 0 (• | x) s.t. E p ρ τ (r) ≥ 0, ∀ τ ∈ [0, 1].
Unlike the discrete-quantile case, where the number of constraints is m, we now have an infinite set of constraints-one for each τ . To handle this, we introduce a nonnegative Lagrange multiplier function λ(τ ) indexed by τ , leading to the following Lagrangian-based reweighting.
this section cite: []

Section: Continuous KL-Regularized Solution.
With a similar argument as in Theorem 4.2, the aligned distribution also takes an exponential reweighting form, but now with a continuous integral:
p λ (y | x) ∆ = p 0 (y | x) exp 1 0 λ(τ ) ρ τ (r(x, y)) dτ C(λ) ,
where the normalizing term is:
C(λ) ∆ = E x,y∼p exp 1 0 λ(τ ) ρ τ (r(x, y)) dτ .
It smoothly incorporates the influence of all quantile constraints through the weighting of λ(τ ).
this section cite: []

Section: Dual Problem and Connection to the Discrete Case.
Following the primal-dual logic as in the discrete setting (cf. Theorem 4.2), the dual problem optimizes over the function λ(τ ), leading to: max λ(τ ) ≥0 {-log C(λ)}. This generalizes the discrete quantile dual objective, now integrating over an entire range of τ . The function λ(τ ) plays a similar role to the discrete multipliers λ j .
While the continuous formulation theoretically imposes constraints at every quantile level, in practice, we approximate the integral over τ by discretizing it into a finite set of representative quantile levels. This reduces the problem to solving a discrete QA problem with a finer resolution. The numerical solution follows the same Monte Carlo-based primal-dual method described in Section 4.2, treating the continuous constraints as an additional layer of sampling.
this section cite: []

Section: Infinitesimal Enhancement Analysis and Algorithm
We now study how small perturbations in the target quantile function c * (τ ) affect both the optimal dual variable λ(τ ) and the aligned distribution p * . This analysis reveals how the system reacts to incremental changes in oversight targets, providing a sensitivity measure for quantile alignment.
Perturbation of the Target Quantile Function. Consider a baseline quantile function c * (τ ) that satisfies the constraints: E p * ρ τ (r) ≥ 0, ∀τ ∈ [0, 1], where ρ τ (r) = τ -I[r < c * (τ )] is the indicator-based quantile reward. The aligned distribution p * is obtained with dual solution λ * (τ ). Now, perturb the quantile function by a tiny shift δ(τ ): c(τ ) = c * (τ ) + δ(τ ). This leads to the solution (p λ , λ(τ )), where δλ(τ ) = λ(τ ) -λ * (τ ) is the first-order change in the dual variable.
Definition of Operator V . The effect of perturbing c * (τ ) propagates through λ(τ ) and ultimately shifts the distribution p * . The key object governing this interaction is the linear operator V , which captures the response of different quantile constraints to changes in λ(τ ):
(V δλ)(τ ) ∆ = 1 0 E p * ρ τ (r) ρ τ (r) δλ(τ ) dτ .
Intuitively, V models the dependency between quantile constraints across different τ -values, determining which shifts in λ(τ ) induce correlated responses. Theorem 5.1 (First-Order Sensitivity of Quantile Alignment). Let (p * , λ * ) be the baseline solution for the quantile function c * (τ ), satisfying:
E p * ρ τ (r) ≥ 0, ∀τ ∈ [0, 1].
Consider a perturbation δ(τ ) such that c(τ ) = c * (τ ) + δ(τ ). Define the first-order changes in the dual variable and the aligned distribution as:
δλ(τ ) = λ(τ ) -λ * (τ ), δp = p λ -p * .
Then, if the original constraint is active, namely E p * [ρ τ (r)] = 0, the perturbed dual variable satisfies the linear integral equation:
(V δλ)(τ ) ∆ = -E p * ρ ′ τ (r) δ(τ ) , where ρ ′ τ (r) = τ -I[r < c * (τ )].
If the constraint is strictly satisfied, we have λ * (τ ) = 0, and small perturbations in c * (τ ) do not affect λ(τ ) until the constraint becomes active. Furthermore, the updated distribution p λ is given by:
p λ (y | x) = p * (y | x)[1 + 1 0 δλ(τ )(ρ τ (r(x, y)) -E p * [ρ τ (r)])dτ ] + O(∥δ∥ 2 ∞ ), where ∥δ∥ ∞ ∆ = sup τ ∈[0,1] |δλ(τ )|.
In practice, τ ∈ [0, 1] is discretized into a finite set {τ i } m i=1 , and V reduces to the matrix form:
V ij ∆ = E p * ρ τi (r) ρ τj (r) .
this section cite: []

Section: Experimental Study
We evaluate quantile alignment on conversational and code-generation tasks, where model outputs range from benign to risky behaviors. Experiments were conducted on a single Nvidia A100 GPU.
this section cite: []

Section: Models.
We apply our alignment procedure to two models: the OPT-1.3B [33] for conversational tasks and CODEGEN-350M [34] for code generation. We experiment with smaller models in the corresponding model families due to GPU constraints.
this section cite: ['b32', 'b33']

Section: Data.
For the conversational task, we use prompts from the Anthropic Harmless dataset [35], which contains human requests formatted between "Human:" and "Assistant:". This dataset serves as a 3.5 3.0 2.5 2.0 1.5 1.0 0.5 0.0 q-Quantile of Helpfulness Rewards 2 1 0 1 2 3 q-Quantile of Harmlessness Rewards Comparison of Quantile Alignment (QA) with Baselines q=5% Original benchmark for evaluating alignment with safe and cooperative responses. For the code-generation task, we employ the HUMANEVAL dataset [5], a standard benchmark that consists of Python programming tasks. Each prompt in HUMANEVAL specifies a function signature and docstring describing the expected behavior. Since HUMANEVAL primarily consists of relatively simple coding tasks, we extend it with a custom prompt set to increase diversity and difficulty.
QA MORL q=10% Original QA MORL q=25% Original QA MORL q=75% Original QA MORL q=75% Original QA MORL
this section cite: ['b34', 'b4']

Section: Custom Code-Generation Prompts.
We curate 200 additional prompts spanning eight categories of software quality: file access, network calls, security risks, maintainability, execution time, data integrity, scalability, and documentation quality. These prompts simulate a range of real-world coding concerns, including adversarial cases. Each custom prompt is randomly assigned a quality level (standard, low, very low, edge case, worst case) and a control statement that explicitly directs the model toward suboptimal coding practices. Examples include: "Do not handle edge cases or errors," "Write in a way that has little error handling," and "Make variable names confusing or non-descriptive." The prompts were generated programmatically using GPT-4 to maintain a structured, instruction-based format. The final dataset combines the original HumanEval tasks with these curated prompts to comprehensively assess model alignment.
this section cite: []

Section: Alignment Values and Evaluation Metrics.
We evaluate four key alignment values: Harmlessness and Helpfulness for conversational tasks, and Simplicity and Security for code generation. For Harmlessness and Helpfulness, we use two GPT-2 models with value heads fine-tuned on human-annotated preferences [36], providing scalar ratings that indicate how well generated responses align with harmless and helpful behavior. For Simplicity and Security, we employ an automated evaluation using OpenAI's API. A GPT4o-based reviewer model is prompted with a definition of each attribute-Simplicity measures how minimal, maintainable, and Pythonic the code is, avoiding unnecessary complexity or redundancy, while Security assesses the code's resilience against vulnerabilities, including secure function calls, input sanitization, and careful data handling. The model rates each attribute on a 0-1 continuous scale.
Additionally, for each alignment task, we assess three baseline performance metrics. 1) Diversity: Quantifies lexical variety in model outputs, computed as the proportion of unique n-grams (n = 2, 3, 4) and aggregated into a composite diversity score [37]. 2) Coherence: Evaluates semantic consistency within generated text using a supervised SimCSE BERT-based model for sentence embedding similarity [38]. 3) Perplexity: Measures how predictable a generated response is under a language model, serving as a proxy for fluency.
Experiment 1: Single-Value Quantile Enhancement. We evaluate our quantile alignment approach by enforcing constraints to systematically improve Helpfulness. Specifically, we align the model's helpfulness scores using the following quantile pairs (τ j , κ j ): (1%, 5%), (5%, 10%), (10%, 50%), (50%, 60%), (60%, 70%), (70%, 80%), (80%, 90%), (90%, 95%), (95%, 99%). This means, for instance, that responses previously scoring at the 1% percentile in helpfulness are elevated to match the 5% percentile level, and so forth. By lifting multiple quantiles, we enforce a strict improvement across the entire distribution.
Figure 1(a) visualizes this shift using a quantile-quantile (QQ) plot, where the x-axis represents the quantile values from the original model, and the y-axis represents the corresponding quantiles from the aligned model. Each blue point represents the τ -quantile of the rewards of a particular human value under both models, with τ sampled at regular intervals (5%, 10%, ..., 95%). Points above the red 45-degree line indicate that the aligned model achieves higher rewards at that quantile compared to the original model. As expected, the results confirm that our approach effectively improves Helpfulness across the entire distribution.
We follow the numerical optimization procedure outlined in Section 4.2 and apply PPO-based finetuning to obtain the aligned model. Figure 1(b) illustrates the impact on Harmlessness, which, as noted in prior work [14], exhibits a tradeoff with Helpfulness. The QQ plot reveals a noticeable decline in Harmlessness scores across quantiles, indicating that improvements in Helpfulness come at the cost of reduced Harmlessness. Meanwhile, Figure 1(c) shows that baseline performance metrics, including Diversity and Coherence, remain largely stable. Interestingly, Perplexity decreases slightly, suggesting that alignment may also contribute to improved response fluency.
Experiment 2: Multi-Value Quantile Enhancement. Building on the previous experiment, we now introduce an additional constraint to enhance Harmlessness, specifically lifting its 5% → 50% quantile, while maintaining the existing constraints on Helpfulness. This aims to aggressively improve Harmlessness while preserving Helpfulness as much as possible. Experiment 3: Comparison of QA with MORL Baseline. We compare Quantile Alignment (QA) with Multi-Objective Reinforcement Learning (MORL) [15][16][17], which optimizes expected rewards by sampling tradeoff weights between objectives. Specifically, in the MORL setting, the dual weight vector λ is generated as λ = s • u, where s is uniformly sampled from (0, 6) and u is sampled from the probability simplex, representing random tradeoff preferences. Figure 3 compares quantile performance across different alignment methods at τ = 5%, 10%, 25%, and 75% for Helpfulness and Harmlessness. The QA-aligned model consistently moves towards the upper-right quadrant, indicating simultaneous improvement in both values relative to the original model. In contrast, MORL without a principled optimization strategy often sacrifices one objective in favor of the other, leading to greater instability in alignment outcomes.
this section cite: ['b35', 'b36', 'b37', 'b13', 'b14', 'b15', 'b16']

Section: Conclusion
We have presented Quantile-Guided Alignment (QA), a principled extension of RLHF that regulates quantiles of reward distributions. Casting the problem as a convex KL-regularized program yields a finite-dimensional dual; a Monte Carlo estimate of this dual directly constructs an overall reward function, which can then be optimized with standard alignment tooling. Complete proofs and additional code-generation results are provided in the Appendix.
this section cite: []

Section: References
Ref_id:b0 Title: Superglue: A stickier benchmark for general-purpose language understanding systems Year: (2019)
Ref_id:b1 Title: Gemv2: Multilingual nlg benchmarking in a single line of code Year: (2022)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b4 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b5 Title: Code llama: Open foundation models for code Year: (2023)
Ref_id:b6 Title: An outlook on the opportunities and challenges of multi-agent ai systems Year: (2025)
Ref_id:b7 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b8 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b9 Title: Ai personal assistants and sustainability: Risks and opportunities Year: (2024)
Ref_id:b10 Title: How secure is code generated by chatgpt? Year: (2023)
Ref_id:b11 Title: Policy shaping: Integrating human feedback with reinforcement learning Year: (2013)
Ref_id:b12 Title: Deep reinforcement learning from policy-dependent human feedback Year: (2019)
Ref_id:b13 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b14 Title: Learning all optimal policies with multiple criteria Year: (2008)
Ref_id:b15 Title: Deep reinforcement learning for multiobjective optimization Year: (2020)
Ref_id:b16 Title: Fine-grained human feedback gives better rewards for language model training Year: (2024)
Ref_id:b17 Title: Safe RLHF: Safe reinforcement learning from human feedback Year: (2024)
Ref_id:b18 Title: Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards Year: (2023)
Ref_id:b19 Title: Contextual moral value alignment through context-based aggregation Year: (2024)
Ref_id:b20 Title: MAP: Multi-human-value alignment palette Year: (2025)
Ref_id:b21 Title: Understanding backdoor attacks through the adaptability hypothesis Year: (2023)
Ref_id:b22 Title: A unified detection framework for inference-stage backdoor defenses Year: (2023)
Ref_id:b23 Title: Demystifying poisoning backdoor attacks from a statistical perspective Year: (2024)
Ref_id:b24 Title: Principles and Practices Lecture Notes Year: (2024-11-27)
Ref_id:b25 Title: Jailbreaking attack against multimodal large language model Year: (2024)
Ref_id:b26 Title: Visual adversarial examples jailbreak aligned large language models Year: (2024)
Ref_id:b27 Title: Survey of hallucination in natural language generation Year: (2023)
Ref_id:b28 Title: A survey on large language model (llm) security and privacy: The good, the bad, and the ugly Year: (2024)
Ref_id:b29 Title: Defending chatgpt against jailbreak attack via self-reminders Year: (2023)
Ref_id:b30 Title: AID: Adaptive integration of detectors for safe ai with large language models Year: ()
Ref_id:b31 Title: Trl: Transformer reinforcement learning Year: (2020)
Ref_id:b32 Title:  Year: (2022)
Ref_id:b33 Title: Codegen: An open large language model for code with multi-turn program synthesis Year: (2023)
Ref_id:b34 Title: HH-RLHF Data Year: (2024-07-05)
Ref_id:b35 Title: Rewards-in-context: Multiobjective alignment of foundation models with dynamic preference adjustment Year: (2024)
Ref_id:b36 Title: Bertscore: Evaluating text generation with bert Year: (2020)
Ref_id:b37 Title: Simcse: Simple contrastive learning of sentence embeddings Year: (2021)
