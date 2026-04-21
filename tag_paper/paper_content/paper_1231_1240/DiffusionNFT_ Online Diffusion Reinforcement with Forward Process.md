Title: DIFFUSIONNFT: ONLINE DIFFUSION REINFORCE-MENT WITH FORWARD PROCESS
Abstract: Online reinforcement learning (RL) has been central to post-training language models, but its extension to diffusion models remains challenging due to intractable likelihoods. Recent works discretize the reverse sampling process to enable GRPO-style training, yet they inherit fundamental drawbacks, including solver restrictions, forward-reverse inconsistency, and complicated integration with classifier-free guidance (CFG). We introduce Diffusion Negative-aware Fine-Tuning (DiffusionNFT), a new online RL paradigm that optimizes diffusion models directly on the forward process via flow matching. DiffusionNFT contrasts positive and negative generations to define an implicit policy improvement direction, naturally incorporating reinforcement signals into the supervised learning objective. This formulation enables training with arbitrary black-box solvers, eliminates the need for likelihood estimation, and requires only clean images rather than sampling trajectories for policy optimization. DiffusionNFT is up to 25× more efficient than FlowGRPO in head-to-head comparisons, while being CFGfree. For instance, DiffusionNFT improves the GenEval score from 0.24 to 0.98 within 1k steps, while FlowGRPO achieves 0.95 with over 5k steps and additional CFG employment. By leveraging multiple reward models, DiffusionNFT significantly boosts the performance of SD3.5-Medium in every benchmark tested.

Section: INTRODUCTION
Online Reinforcement Learning (RL) has been pivotal in the post-training of LLMs, driving recent advances in LLMs' alignment and reasoning abilities (Achiam et al., 2023;Guo et al., 2025). How- ever, replicating similar success for diffusion models in visual generation is not straightforward. Policy Gradient algorithms assume that model likelihoods are exactly computable. This assumption holds for autoregressive models, but is inherently violated by diffusion models, where likelihoods can only be approximated via costly probabilistic ODE or variational bounds of SDE (Song et al., 2021). Recent works circumvent this barrier by discretizing the reverse sampling process, reframing diffusion generation as a multi-step decision-making problem (Black et al., 2023). This makes transitions between adjacent steps tractable Gaussians, enabling direct application of existing RL algorithms like GRPO to the diffusion domain (Xue et al., 2025;Liu et al., 2025).
Despite promising efforts made, we argue that GRPO-style diffusion reinforcement still faces fundamental limitations: (1) Forward inconsistency. Focusing solely on the reverse sampling process breaks adherence to the forward diffusion process, risking the model degenerating into cascaded Gaussians. (2) Solver restriction. The data collection process relies on first-order SDE samplers, precluding the full utilization of ODE or high-order solvers that are default to flow models and advantageous for generation efficiency. (3) Complicated CFG integration. Diffusion models heavily rely on Classifier-Free Guidance (CFG) (Ho & Salimans, 2022), which requires training both conditional and unconditional models. Current RL practices typically incorporate CFG in post-training, leading to a complicated and inefficient two-model optimization scheme.
We aim to disentangle data collection, remove solver restriction, and maintain consistency with standard supervised pretraining in diffusion RL. As a diffusion policy admits a single forward (noising) process but multiple reverse (denoising) processes (e.g., different samplers), a natural question is:
Can diffusion reinforcement be performed on the forward process instead of the reverse?
This paper proposes a novel online RL paradigm named Diffusion Negative-aware FineTuning (Dif-fusionNFT). Instead of building upon the conventional policy gradient framework, DiffusionNFT directly performs policy optimization on the forward diffusion process through the flow matching objective. Intuitively, it defines a contrastive improvement direction between two implicit policies learned on "positive" and "negative" generated samples split by reward signals, and optimizes toward the positive policy without modifying the sampling process.
The forward-process RL formulation provides several practical benefits (Figure 2). First, Diffu-sionNFT allows data collection with arbitrary black-box solvers, rather than relying on first-order SDE samplers. Second, it eliminates the need to store entire sampling trajectories, requiring only clean images for policy optimization. Third, it is fully compatible with standard diffusion training, requiring minimal modifications to existing codebases. Finally, it is a native off-policy algorithm, naturally allowing decoupled training and sampling policies without importance sampling.
We evaluate DiffusionNFT by post-training SD3.5-Medium (Esser et al., 2024) on multiple reward models. The entire training process deliberately operates in a CFG-free setting. Although this results in a significantly lower initialization performance, we find DiffusionNFT substantially improves performance across both in-domain and out-of-domain rewards, rapidly outperforming CFG and the GRPO baseline. We also conduct head-to-head comparisons against FlowGRPO in single-reward settings. Across four tasks tested, DiffusionNFT consistently exhibits 3× to 25× efficiency and achieves better final scores. For instance, it improves the GenEval score from 0.24 to 0.98 within 1k steps, while FlowGRPO achieves only 0.95 with over 5k steps and additional CFG employment.
DiffusionNFT is a direct RL alternative to conventional Policy Gradient methods, introducing the Negative-aware FineTuning (NFT) paradigm (Chen et al., 2025c) into the diffusion domain. Grounded in a supervised learning foundation, we believe this paradigm offers a valid path toward a general, unified, and native off-policy RL recipe across various modalities.
this section cite: ['b0', 'b42', 'b1', 'b49', 'b28', 'b14', 'b7']

Section: BACKGROUND

this section cite: []

Section: DIFFUSION AND FLOW MODELS
Diffusion models (Ho et al., 2020;Song et al., 2020b) learn continuous data distributions by gradually perturbing clean data x 0 ∼ π 0 = p data with Gaussian noise according to a forward process. Then, data can be generated by learning to reverse this process.
The forward noising process admits a closed-form transition kernel π t|0 (x t |x 0 ) = N (α t x 0 , σ 2 t I) with a specific noise schedule α t , σ t , enabling reparameterization as
x t = α t x 0 + σ t ϵ, ϵ ∼ N (0, I).
One way to learn diffusion models is to adopt the velocity parameterization v θ (x t , t) (Zheng et al., 2023b), which predicts the tangent of the trajectory, trained by minimizing
E t,x0∼π0,ϵ∼N (0,I) [w(t)∥v θ (x t , t) -v∥ 2 2 ],(1)
where the target velocity v is defined by the schedule's time derivatives as v = αt x 0 + σt ϵ under the notation ḟt := df t /dt, and w(t) is some weighting function. Reverse sampling typically follows the ODE form (Song et al., 2020b) of the diffusion model, which is reduced to dxt dt = v θ (x t , t) using v θ . This formulation is known as flow matching (Lipman et al., 2022), where simple Euler discretization serves as an effective ODE solver, equivalent to DDIM (Song et al., 2020a).
Rectified flow (Liu et al., 2022) can be considered as a special case of the above-discussed diffusion models, where α t = 1 -t, σ t = t, which simplifies the velocity target to v = ϵx 0 .
this section cite: ['b15', 'b27', 'b29']

Section: POLICY GRADIENT ALGORITHMS FOR DIFFUSION MODELS
In order to apply Policy Gradient algorithms such as PPO (Schulman et al., 2017) or GRPO (Shao et al., 2024) to diffusion models, recent works (Black et al., 2023;Fan et al., 2023;Liu et al., 2025;Xue et al., 2025) formulate the diffusion sampling as a multi-step Markov Decision Process (MDP). This can be achieved by discretizing the reverse sampling process of diffusion models.
While flow models naturally admit simple and efficient sampling through ODE, the lack of stochasticity hinders the application of GRPO. FlowGRPO (Liu et al., 2025) addresses this by using the SDE form (Song et al., 2020b) under the velocity parameterization v θ (see Appendix B.1):
dx t = v θ (x t , t) + g 2 t 2t x t + (1 -t)v θ (x t , t) dt + g t dw t(2)
where g t = a t 1-t controls the level of injected stochasticity. Discretizing it with Euler yields
π θ (x t-∆t | x t ) = N x t + v θ (x t , t) + g 2 t 2t (x t + (1 -t)v θ (x t , t)) ∆t, g 2 t ∆t I .
This makes transition kernels between adjacent steps likelihood tractable Gaussians, enabling the direct application of existing policy gradient algorithms, such as GRPO.
this section cite: ['b38', 'b39', 'b1', 'b8', 'b28', 'b49', 'b28']

Section: DIFFUSION REINFORCEMENT VIA NEGATIVE-AWARE FINETUNING

this section cite: []

Section: PROBLEM SETUP
Online RL. Consider a pretrained diffusion policy π old and prompt datasets {c}. At each iteration, we sample K images x 1:K 0 for prompt c, and then evaluate each image with a scalar reward function r ∈ [0, 1], representing its optimality probability r(x 0 , c) := p(o = 1|x 0 , c) (Levine, 2018). This optimality serves as a bridge from continuous-valued rewards to a binary partition. Collected data can be randomly split into two imaginary subsets. An image x 0 will have a probability r of falling into the positive dataset D + and otherwise the negative dataset D -. Given infinite samples, the underlying distributions of these two subsets are respectively
π + (x 0 |c) := π old (x 0 |o = 1, c) = p(o = 1|x 0 , c)π old (x 0 |c) p π old (o = 1|c) = r(x 0 , c) p π old (o = 1|c) π old (x 0 |c) π -(x 0 |c) := π old (x 0 |o = 0, c) = p(o = 0|x 0 , c)π old (x 0 |c) p π old (o = 0|c) = 1 -r(x 0 , c) 1 -p π old (o = 1|c) π old (x 0 |c)
RL requires performing policy improvement at each iteration. The optimized policy π * satisfies
E π * (•|c) r(x 0 , c) > E π old (•|c) r(x 0 , c) (denoted as π * ≻ π old )
Policy Improvement on Positive Data. It is easy to prove that π + ≻ π old ≻ π -constantly holds, thus a straightforward improvement of π old can be π * = π + . To achieve this, previous work (Lee et al., 2023) performs diffusion training solely on D + , known as Rejection FineTuning (RFT).
Despite the simplicity, RFT cannot effectively leverage negative data in D - (Chen et al., 2025c).
Reinforcement Guidance. We posit that negative feedback is crucial to policy improvement, especially for diffusionfoot_0 . Rather than treating π + as an optimization point, we leverage both negative and positive data to derive an improvement direction ∆ ∈ R n . The training target is defined as
v * (x t , c, t) := v old (x t , c, t) + 1 β ∆(x t , c, t).(3)
where v is the velocity predictor of the diffusion model, β is a hyperparameter. This definition formally resembles diffusion guidance such as Classifier-Free Guidance (CFG) (Ho & Salimans, 2022). We term ∆(x t , c, t) ∈ R n reinforcement guidance, and 1 β ∈ R guidance strength. In Section 3.2, we address two challenges: 1. What is an appropriate form of ∆ that enables policy improvement? 2. How to directly optimize v θ → v * leveraging collected dataset D + and D -?
this section cite: ['b23', 'b22', 'b14']

Section: NEGATIVE-AWARE DIFFUSION REINFORCEMENT WITH FORWARD PROCESS
In Eq. ( 3), ∆ corresponds to the distributional shift between an improved policy and the original policy. To formalize this, we first study the distributional difference between π + ≻ π old ≻ π -. Theorem 3.1 (Improvement Direction). Consider diffusion models v + , v -, and v old for the policy triplet π + , π -, and π old . The directional differences between these models are proportional:
∆ :=[1 -α(x t )] [v old (x t , c, t) -v -(x t , c, t)] (Reinforcement Guidance) = α(x t ) [v + (x t , c, t) -v old (x t , c, t)]. (4
)
where 0 ≤ α(x t ) ≤ 1 is a scalar coefficient:
α(x t ) := π + t (x t |c) π old t (x t |c) E π old (x0|c) r(x 0 , c) 𝑫 𝒗 - 𝒗 + 𝒗 𝜃 𝑫 - 𝑫 + Figure 3: Improvement Direction.
Eq. ( 4) indicates an ideal guidance direction ∆ for improving over v old . With appropriate guidance strength, policy improvement can be guaranteed. For instance, let β = α(x t ) in Eq. ( 3), we have v * (x t , c, t) = v old (x t , c, t) + 1 α(xt) ∆(x t , c, t) = v + (x t , c, t), such that π * = π + ≻ π old holds. Figure 3 contains an illustration for the improvement direction ∆.
Having defined a valid optimization target v * with Eq. ( 3) and (4), we now introduce a training objective that directly optimizes v θ towards v * : and negative (r = 0) branches. Rather than training two independent models v + θ and v - θ , it adopts an implicit parameterization technique that directly optimizes a single target policy v θ .
Theorem 3.2 (Policy Optimization). Consider the training objective:
L(θ) = E c,π old (x0|c),t r∥v + θ (x t , c, t) -v∥ 2 2 + (1 -r)∥v - θ (x t , c, t) -v∥ 2 2 , (5
)
where v + θ (x t , c, t) := (1 -β)v old (x t , c, t) + βv θ (x t , c, t), (Implicit positive policy) and v - θ (x t , c, t) := (1 + β)v old (x t , c, t) -βv θ (x t , c, t). (Implicit negative policy) Given unlimited data and model capacity, the optimal solution of Eq. ( 5)
satisfies v θ * (x t , c, t) = v old (x t , c, t) + 2 β ∆(x t , c, t).(6)
Theorem 3.2 presents a new off-policy RL paradigm (Figure 4). Instead of applying Policy Gradient, it adopts supervised learning (SL) objectives, but additionally trains on online negative data D -. This renders the algorithm highly versatile, compatible with existing SL methods. We term our method Diffusion Negative-aware FineTuning (DiffusionNFT), highlighting its negative-aware SL nature and conceptual similarity to parallel algorithm NFT in language models (Chen et al., 2025c).
Below, we discuss several distinctive advantages of DiffusionNFT.
1. Forward Consistency. In contrast to policy gradient methods (e.g., FlowGRPO), which formulated RL on the reverse diffusion process, DiffusionNFT defines a typical diffusion loss on the forward process. This preserves what we term forward consistency-the adherence of the diffusion model's underlying probability density to the Fokker-Planck equation (Øksendal, 2003;Song et al., 2020b), ensuring that the learned model corresponds to a valid forward process (i.e., x t are correctly coupled with
x 0 through a joint distribution π θ (x t , x 0 ) = π θ (x 0 )π t|0 (x t |x 0 )).
this section cite: ['b33']

Section: Solver Flexibility.
DiffusionNFT fully decouples policy training and data sampling. This enables the full utilization of any black-box solvers throughout sampling, rather than relying on first-order SDE samplers. It also eliminates the need to store the entire sampling trajectory during data collection, requiring only clean images with their associated rewards for training.
this section cite: []

Section: Implicit Guidance Integration.
Intuitively, DiffusionNFT defines a guidance direction ∆ and apply such guidance to the old policy v old (Eq. ( 6)). However, instead of learning a separate guidance model ∆ θ and employing guided sampling, it adopts an implicit parameterization technique that enables direct integration of reinforcement guidance into the learned policy. This technique, inspired by recent advances in guidance-free training (Chen et al., 2025a), allows us to perform RL continuously on a single policy model, which is crucial to online reinforcement.
this section cite: []

Section: Likelihood-Free Formulation.
Previous diffusion RL methods are fundamentally constrained by their reliance on likelihood approximation. Whether approximating the marginal data likelihood with variational bounds and applying Jensen's inequality to reduce loss computation cost (Wallace et al., 2024), or discretizing the reverse process to estimate sequence likelihood (Black et al., 2023), they inevitably introduce systematic estimation bias into diffusion post-training. In contrast, Diffu-sionNFT is inherently likelihood-free, bypassing such compromises.
this section cite: ['b43', 'b1']

Section: PRACTICAL IMPLEMENTATION
We provide DiffusionNFT pseudo code in Algorithm 1. Below, we elaborate on key design choices. Collect K clean images x 1:K 0 , and evaluate their rewards {r raw } 1:K . 4:
Normalize raw rewards in group: r norm := r rawmean({r raw } 1:K ). 5:
Define optimality probability r = 0.5 + 0.5 * clip{r norm /Zc, -1, 1}. 6: D ← {c, x 1:K 0 , r 1:K ∈ [0, 1]}. 7:
end for 8:
for each mini batch {c, x0, r} ∈ D do // Gradient Step, Policy Optimization 9:
Forward diffusion process:
xt = αtx0 + σtϵ; v = αtx0 + σtϵ. 10: Implicit positive velocity: v + θ (xt, c, t) := (1 -β)v old (xt, c, t) + βv θ (xt, c, t). 11: Implicit negative velocity: v - θ (xt, c, t) := (1 + β)v old (xt, c, t) -βv θ (xt, c, t). 12: θ ← θ -λ∇ θ r∥v + θ (xt, c, t) -v∥ 2 2 + (1 -r)∥v - θ (xt, c, t) -v∥ 2 2 .
(Eq. ( 5)) 13:
end for 14:
Update data collection policy θ old ← ηiθ old + (1 -ηi)θ, and clear buffer D ← ∅. // Online Update 15: end for Output: v θ Optimality Reward. In most visual reinforcement settings, rewards manifest as unconstrained continuous scalars rather than binary optimality signals. Motivated by existing GRPO practices (Shao et al., 2024;Liu et al., 2025;Xue et al., 2025), we first transform the raw reward r raw into r ∈ [0, 1] which represents the optimality probability:
r(x 0 , c) := 1 2 + 1 2 clip r raw (x 0 , c) -E π old (•|c) r raw (x 0 , c) Z c , -1, 1 .
Z c > 0 is some normalizing factor, which could take the form of a global reward std. We sample K images for each prompt c during data collection, so the average reward E π old (•|c) r raw (x 0 , c) for each prompt can be estimated.
this section cite: ['b39', 'b28', 'b49']

Section: Soft Update of Sampling Policy.
The off-policy nature of DiffusionNFT decouples the sampling policy π old from the training policy π θ . This obviates the need for a "hard" update (π old ← π θ ) after each iteration. Instead, we leverage this property to employ a "soft" EMA update: θ old ← η i θ old + (1 -η i )θ where i is the iteration number. The parameter η governs a trade-off between learning speed and stability. A strictly on-policy scheme (η = 0) yields rapid initial progress but is prone to severe instability, leading to catastrophic collapse. Conversely, a nearly offline approach (η → 1) is robustly stable but suffers from impractically slow convergence (Figure 8).
Adaptive Loss Weighting. Typical diffusion loss includes a time-dependent weighting w(t) (Eq. ( 1)). Instead of manual tuning, we adopt an adaptive weighting scheme. The velocity predictor v θ can be equivalently transformed into x 0 predictor, denoted as x θ (e.g., x θ = x t -tv θ under rectified flow schedule). We replace the weighting with a form of self-normalized x 0 regression, motivated by the diffusion distillation method DMD (Yin et al., 2024):
w(t)∥v θ (x t , c, t) -v∥ 2 2 ← ∥x θ (x t , c, t) -x 0 ∥ 2 2 sg(mean(abs(x θ (x t , c, t) -x 0 )))
where sg is the stop-gradient operator. We find it typically leads to faster training (Figure 9).
this section cite: ['b51']

Section: CFG-Free Optimization.
Classifier-Free Guidance (CFG) (Ho & Salimans, 2022) is a default technique to enhance generation quality at inference time, yet it complicates post-training and reduces efficiency. Conceptually, we interpret CFG as an offline form of reinforcement guidance (Eq. ( 4)), where conditional and unconditional models correspond to positive and negative signals. With this understanding, we discard CFG in our algorithm design, and the policy is initialized solely by the conditional model. Despite this seemingly poor initialization, we observe that performance surges and quickly surpasses the CFG baseline (Figure 1). This suggests that the functionality of CFG can be effectively learned or substituted through RL post-training, echoing recent studies that achieve strong performance without CFG through post-training (Chen et al., 2025b;a;Zheng et al., 2025).
this section cite: ['b14', 'b57']

Section: EXPERIMENTS
We demonstrate the potential of DiffusionNFT through three perspectives: (1) multi-reward joint training for strong CFG-free performance, (2) head-to-head comparison with FlowGRPO on single rewards, and (3) ablation studies on key design choices.
this section cite: []

Section: EXPERIMENTAL SETUP
Our experiments are based on SD3.5-Medium (Esser et al., 2024) at 512×512 resolution, with most settings aligned with FlowGRPO (Liu et al., 2025).
this section cite: ['b7', 'b28']

Section: Reward Models.
(1) Rule-based rewards, including GenEval (Ghosh et al., 2023) for compositional image generation and OCR for visual text rendering, where the partial reward assignment strategies follow FlowGRPO.
(2) Model-based rewards, including PickScore (Kirstain et al., 2023), ClipScore (Hessel et al., 2021), HPSv2.1 (Wu et al., 2023), Aesthetics (Schuhmann, 2022), ImageReward (Xu et al., 2023) and UnifiedReward (Wang et al., 2025), which measure image quality, image-text alignment and human preference.
Prompt Datasets. For GenEval and OCR, we use the corresponding training and test sets from FlowGRPO. For other rewards, we train on Pick-a-Pic (Kirstain et al., 2023) and evaluate on DrawBench (Saharia et al., 2022).
Training and Evaluation. We finetune with LoRA (α = 64, r = 32). Each epoch consists of 48 groups with group size G = 24. We use 10 rollout sampling steps for head-to-head comparison and ablation studies, and 40 steps for best visual quality in multi-reward training. Evaluation is performed with 40-step first-order ODE sampler. Additional details are provided in Appendix C.
this section cite: ['b10', 'b21', 'b13', 'b46', 'b48', 'b21', 'b36']

Section: MULTI-REWARD JOINT TRAINING
We first assess DiffusionNFT's effectiveness in comprehensively enhancing the base model. Starting from the CFG-free SD3.5-M (2.5B parameters), we jointly optimize five rewards: GenEval, OCR, PickScore, ClipScore, and HPSv2.1. Since the rewards are based on different prompts, we first train on Pick-a-Pic with model-based rewards to strengthen alignment and human preference, followed by rule-based rewards (GenEval, OCR). Out-of-domain evaluation is conducted on Aesthetics, ImageReward, and UnifiedReward.
As shown in Table 1, our final CFG-free model not only surpasses CFG and matches FlowGRPO (fitted only single rewards) on both in-domain and out-of-domain metrics, but also outperforms CFGbased larger models such as SD3.5-L (8B parameters) and FLUX.1-Dev (12B parameters) (Labs, 2024). Qualitative comparison in Figure 5 demonstrates the superior visual quality of our method.
this section cite: []

Section: HEAD-TO-HEAD COMPARISON
We conduct head-to-head comparisons with FlowGRPO on single training rewards. As shown in Figure 1(a) and Figure 6, our method is 3× to 25× more efficient in terms of wall-clock time, SD3.5-M a photo of a blue pizza and a yellow baseball glove
this section cite: []

Section: FlowGRPO DiffusionNFT
New York Skyline with 'Google Research Pizza Cafe' written with fireworks on the sky.
A vibrant urban alley with a graffiti wall prominently spraypainted "Street Art Rules", surrounded by colorful tags and murals, under a sunny sky.
An old photograph of a 1920s airship shaped like a pig, floating over a wheat field.
A red colored car. 0 200 400 600 800 Training Time (GPU Hours) 0.6 0.7 0.8 0.9 1.0 OCR Score 24× Efficiency FlowGRPO DiffusionNFT (a) 0 250 500 750 1000 1250 1500 Training Time (GPU Hours) 22.00 22.25 22.50 22.75 23.00 23.25 23.50 23.75 24.00 PickScore 8× Efficiency FlowGRPO DiffusionNFT (b) 0 250 500 750 1000 1250 1500 Training Time (GPU Hours) 0.28 0.30 0.32 0.34 0.36 0.38 HPSv2.1 Score 3× Efficiency FlowGRPO DiffusionNFT (c) Figure 6: Head-to-head comparison between DiffusionNFT with FlowGRPO on single rewards. achieving GenEval score of 0.98 within only ∼1k iterations. This demonstrates that CFG-free models can rapidly adapt to specific reward environments under our framework. 4.4 ABLATION STUDIES 0 200 400 600 800 1000 1200 Training Iterations 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 GenEval Score 1st-order SDE 1st-order ODE 2nd-order ODE (a) 0 200 400 600 800 1000 Training Iterations 20.0 20.5 21.0 21.5 22.0 22.5 23.0 23.5 24.0 PickScore 1st-order SDE 1st-order ODE 2nd-order ODE (b) Figure 7: Different diffusion samplers for data collection. We analyze the impact of our core design choices:
Negative Loss. The negative-aware component is crucial in DiffusionNFT. Without the negative policy loss on v - θ , we find rewards collapse almost instantly during online training, highlighting the 0 100 200 300 400 500 Training Iterations 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 GenEval Score = 0.01 = 1.0 = 10.0 Figure 10: Choices of strength β.
essential role of negative signals in diffusion RL. This phenomenon is divergent from observations in LLMs, where RFT remains a strong baseline (Xiong et al., 2025;Chen et al., 2025c).
this section cite: ['b47']

Section: Diffusion Sampler.
Online samples in DiffusionNFT are used both for reward evaluation and as training data, making quality critical. Figure 7 shows that ODE samplers outperform SDE ones, especially on PickScore, which is noise-sensitive. Second-order ODE slightly outperforms firstorder on GenEval, while being comparable on PickScore.
Adaptive Weighting. We find stability improves when the flow-matching loss is given higher weight at larger t, whereas inverse strategies (e.g., w(t) = 1 -t) lead to collapse (Figure 9). Our adaptive schedule consistently matches or exceeds heuristic choices.
this section cite: []

Section: Soft Update.
We compare different η i schedules for the soft update in Figure 8. Fully on-policy (η i = 0) accelerates early progress but destabilizes training, while overly off-policy (η = 0.9) slows convergence. We find that starting with a small η and gradually increasing it to a larger value in later stages strikes an effective balance between convergence speed and training stability.
Guidance Strength. As shown in Figure 10, the guidance parameter β also governs a trade-off between stability and convergence speed. We find that β near 1 performs stably and select β as 1 or 0.1 (for faster reward increase) in practice.
this section cite: []

Section: RELATED WORK
The transition of RL algorithms from discrete autoregressive (AR) to continuous diffusion models poses a central challenge: the inherent difficulty of diffusion models for computing exact model likelihoods (Song et al., 2021), which are nonetheless crucial for RL (Chen et al., 2023;Liu et al., 2025). To address this challenge, existing efforts include:
Likelihood-free methods: (1) Reward Backpropagation (Xu et al., 2023;Prabhudesai et al., 2023;Clark et al., 2023;Prabhudesai et al., 2024) proves highly effective, yet is limited to differentiable rewards and can only tune low-noise timesteps due to memory costs and gradient explosion when unrolling long denoising chains. (2) Reward-Weighted Regression (RWR) (Lee et al., 2023) is an offline finetuning method but lacks a negative policy objective to penalize low-reward generations.
(3) Policy Guidance. This includes energy guidance (Janner et al., 2022;Lu et al., 2023) and CFGstyle guidance (Frans et al., 2025;Jin et al., 2025). These methods all require combining multiple models for guided sampling, thus complicating online optimization. (4) Score-based RL. These methods try to perform RL directly on the score rather than the likelihood field (Zhu et al., 2025).
Likelihood-based methods: (1) Diffusion-DPO (Wallace et al., 2024;Yang et al., 2024;Liang et al., 2024;Yuan et al., 2024;Li et al., 2025a) adapts DPO to diffusion for paired human preference data but requires additional likelihood and loss approximations compared to AR; DDO (Zheng et al., 2025) uses high-quality dataset as positive signals and self-generated samples as negative signals to avoid the requirement of paired data, achieving state-of-the-art CFG-free FIDs in visual generation, while still relying on likelihood approximation for the diffusion case. (2) Policy gradient methods, starting from PPO style (Black et al., 2023;Fan et al., 2023), decompose trajectory likelihoods step by step without considering forward consistency. Recent GRPO extensions (Liu et al., 2025;Xue et al., 2025) prove effective and scalable for diffusion RL, but they couple the training loss with SDE samplers and face efficiency bottlenecks. MixGRPO (Li et al., 2025b) improves efficiency by mixing SDE and ODE, while issues of coupling and forward inconsistency remain.
this section cite: ['b42', 'b32', 'b28', 'b48', 'b34', 'b6', 'b35', 'b22', 'b18', 'b32', 'b9', 'b19', 'b58', 'b43', 'b50', 'b52', 'b57', 'b1', 'b8', 'b28', 'b49']

Section: CONCLUSION
We introduce Diffusion Negative-aware FineTuning (DiffusionNFT), a new paradigm for online reinforcement learning of diffusion models that directly operates on the forward process. By formulating policy improvement as a contrast between positive and negative generations, DiffusionNFT integrates reinforcement signals seamlessly into the standard diffusion objective, eliminating the reliance on likelihood estimation and SDE-based reverse process. Empirically, DiffusionNFT demonstrates strong and efficient reward optimization, achieving up to 25× higher efficiency than Flow-GRPO while producing a single model that outperforms CFG baselines across diverse in-domain and out-of-domain rewards. We believe this work represents a step toward unifying supervised and reinforcement learning in diffusion, and highlights the forward process as a promising foundation for scalable, efficient, and theoretically principled diffusion RL.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Training diffusion models with reinforcement learning Year: (2023)
Ref_id:b2 Title: Offline reinforcement learning via high-fidelity generative behavior modeling Year: (2023)
Ref_id:b3 Title: Visual generation without guidance. Forty-second international conference on machine learning Year: (2025)
Ref_id:b4 Title: Toward guidance-free ar visual generation via condition contrastive alignment Year: ()
Ref_id:b5 Title: Bridging supervised learning and reinforcement learning in math reasoning Year: (2025)
Ref_id:b6 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2023)
Ref_id:b7 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b8 Title: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models Year: (2023)
Ref_id:b9 Title: Diffusion guidance is a controllable policy improvement operator Year: (2025)
Ref_id:b10 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b11 Title: Seeds: Exponential sde solvers for fast high-quality sampling from diffusion models Year: (2023)
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b13 Title: Clipscore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b14 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b15 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b16 Title: Exponential integrators Year: (2010)
Ref_id:b17 Title: A variational perspective on diffusionbased generative models and score matching Year: (2021)
Ref_id:b18 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
Ref_id:b19 Title: Lifeng Qiao, Ning Ding, Alex Lamb, and Xipeng Qiu. Inference-time alignment control for diffusion models with reinforcement learning guidance Year: (2025)
Ref_id:b20 Title: Variational diffusion models. Advances in neural information processing systems Year: (2021)
Ref_id:b21 Title: Picka-pic: An open dataset of user preferences for text-to-image generation Year: (2023)
Ref_id:b22 Title: Aligning text-to-image models using human feedback Year: (2023)
Ref_id:b23 Title: Reinforcement learning and control as probabilistic inference: Tutorial and review Year: (2018)
Ref_id:b24 Title: Divergence minimization preference optimization for diffusion model alignment Year: (2025)
Ref_id:b25 Title: Unlocking flow-based grpo efficiency with mixed ode-sde Year: (2025)
Ref_id:b26 Title: Step-aware preference optimization: Aligning preference with denoising performance at each step Year: ()
Ref_id:b27 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b28 Title: Flow-grpo: Training flow matching models via online rl Year: (2025)
Ref_id:b29 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b30 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b31 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2022)
Ref_id:b32 Title: Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning Year: (2023)
Ref_id:b33 Title: Stochastic differential equations Year: (2003)
Ref_id:b34 Title: Aligning text-toimage diffusion models with reward backpropagation Year: (2023)
Ref_id:b35 Title: Video diffusion alignment via reward gradients Year: (2024)
Ref_id:b36 Title: Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems Year: (2022)
Ref_id:b37 Title: Laion-aesthetics Year: (2022)
Ref_id:b38 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b39 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b40 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b41 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b42 Title: Maximum likelihood training of scorebased diffusion models Year: (2021)
Ref_id:b43 Title: Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b44 Title: Coefficients-preserving sampling for reinforcement learning with flow matching Year: (2025)
Ref_id:b45 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b46 Title: Human preference score: Better aligning text-to-image models with human preference Year: (2023)
Ref_id:b47 Title: A minimalist approach to llm reasoning: from rejection sampling to reinforce Year: (2025)
Ref_id:b48 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b49 Title: Dancegrpo: Unleashing grpo on visual generation Year: (2025)
Ref_id:b50 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b51 Title: One-step diffusion with distribution matching distillation Year: (2024)
Ref_id:b52 Title: Self-play fine-tuning of diffusion models for text-to-image generation Year: (2024)
Ref_id:b53 Title: Fast sampling of diffusion models with exponential integrator Year: (2022)
Ref_id:b54 Title: Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics Year: (2023)
Ref_id:b55 Title: Improved techniques for maximum likelihood estimation for diffusion odes Year: (2023)
Ref_id:b56 Title: Diffusion bridge implicit models Year: (2024)
Ref_id:b57 Title: Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator Year: (2025)
Ref_id:b58 Title: Dspo: Direct score preference optimization for diffusion model alignment Year: (2025)
