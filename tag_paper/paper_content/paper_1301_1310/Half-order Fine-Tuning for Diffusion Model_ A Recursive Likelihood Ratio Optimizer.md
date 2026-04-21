Title: HALF-ORDER FINE-TUNING FOR DIFFUSION MODEL: A RECURSIVE LIKELIHOOD RATIO OPTIMIZER
Abstract: 

Section: 
Tao Ren 1 * † Zishi Zhang
1 * Jinyang Jiang 1 * Zehao Li 1 * Shentao Qin 2,3 * Yi Zheng 1 Guanghao Li 2 Qianyou Sun 1 Yan Li 4 Jiafeng Liang 5 Xinping Li 1 Yijie Peng 1,6 ‡ 1 Peking University 2 Tsinghua University 3 OriginFlow 4 HKUST 5 Harbin Institute of Technology 6 Xiangjiang Laboratory Stable Diffusion 1.4 After fine-tuning A woman sewing a quilt on a sewing machine in her craft room.
An illustration of a majestic wolf with piercing eyes.
A girl playing with a cat in a joyful atmosphere.
An artistic portrait of a proud eagle.
A realistic photo of a spider weaving the net, great details.
A Motorcycle parked on the sidewalk next to a road.
this section cite: []

Section: RLR optimized Text2Vieo generation
A woman with freckles and curly hair smiling warmly, under a golden sunset.
A futuristic car races through a desert, kicking up clouds of sand.
A brave astronaut in a space suit floats above a planet, gazing at a galaxy full of glowing stars.
A stealthy ninja leaps across rooftops under two moons.
this section cite: []

Section: ABSTRACT
The probabilistic diffusion model (DM), generating content by inferencing through a recursive chain structure, has emerged as a powerful framework for visual generation. After pre-training on enormous data, the model needs to be properly aligned to meet requirements for downstream applications. How to efficiently align the foundation DM is a crucial task. Contemporary methods are either based on Reinforcement Learning (RL) or truncated Backpropagation (BP). However, RL and truncated BP suffer from low sample efficiency and biased gradient estimation, respectively, resulting in limited improvement or, even worse, complete training failure. To overcome the challenges, we propose the Recursive Likelihood Ratio (RLR) optimizer, a Half-Order (HO) fine-tuning paradigm for DM. The HO gradient estimator enables the computation graph rearrangement within the recursive diffusive chain, making the RLR's gradient estimator an unbiased one with lower variance than other methods. We characterize the bias, variance, and convergence behavior of our method. Extensive experiments are conducted on image and video generation to validate the superiority of the RLR. Furthermore, we propose a novel prompting technique that is natural for the RLR to achieve a synergistic effect. The implementation is available at https://github.com/RTkenny/RLR-Optimizer.
this section cite: []

Section: INTRODUCTION
Probabilistic diffusion model (DM) (Sohl-Dickstein et al., 2015;Ho et al., 2020;Zhang et al., 2024b;Gao & Li, 2025) has emerged as a transformative framework in high-fidelity data generation, demonstrating unmatched capabilities in diverse applications such as image synthesis (Podell et al., 2023), video generation (Wang et al., 2023a), and multi-modal data modeling. These models operate by recursively denoising latent representations, as shown in Figure 2, effectively capturing complex data distributions. However, fine-tuning DMs in the post-training phase remains a daunting challenge, since the gradient estimation through the recursive structure imposes excessive computation overhead (Clark et al., 2023). This challenge has limited the broader deployment of DM in dynamic and resource-constrained environments. 6WHS 5HZDUG 7 7 7 7 7 Figure 3: Model collapse caused by the truncation: training SD 1.4 on the aesthetic reward model by truncated BP.
It is a natural way to fine-tune DMs via full backpropagation (BP) through all time steps (Rumelhart et al., 1986), which is theoretically functional, providing precise gradient estimation over the entire diffusion chain. However, the computational and memory overhead of BP scales prohibitively with the model size and the number of diffusion steps (Prabhudesai et al., 2023;Yuan et al., 2024a;Clark et al., 2023;Prabhudesai et al., 2024), making full BP impractical for most real-world scenarios. Specifically, training Stable Diffusion 1.4 by full BP with a batch size of 1 and 50 time steps would require approximately 1TB of GPU RAM (Prabhudesai et al., 2023). Thus, truncating recursive differentiation becomes a common practice to alleviate memory overhead. But truncated BP suffers from structural bias, as it terminates gradient computation before sourcing to the input, only considering a limited subset of the diffusion chain. Fine-tuning based on a biased gradient estimation can inadvertently impair the optimization performance, resulting in model collapse, i.e., contents generated reducing to pure noise. Empirical evidence, in Figure 3, shows that truncated gradients result in a significant drop in reward scores during training: the fewer the truncated time steps, the more severe the model collapse. Moreover, truncated BP fails to capture the multi-scale information across all time steps due to the absence of differentiation on early steps. The hierarchical feature of generation imposes a thorough gradient evaluation to retain fidelity from pixel to structural level.
Reinforcement learning (RL) (Schulman et al., 2017) as a gradient computation trick has enabled another branch of DM fine-tuning (Lee et al., 2023;Black et al., 2023;Wallace et al., 2024;Fan et al., 2024). It typically ignores the differentiable connection between steps and recovers the gradient by estimation. RL avoids caching intermediate activations, significantly reducing memory requirements. It also supports gradient computation in a divided manner under extreme circumstances to accommodate insufficient memory. The cost to pay is the high variance of the estimated gradient. Even if the estimator is unbiased, the variance can result in wild sample-inefficient updates, demonstrated by the slow convergence during training. These limitations of BP and RL underscore the necessity of a more efficient, scalable, and stable finetuning approach that harmonizes computational tractability with optimization efficacy. To address this, we first investigate the recursive architecture of the DM and propose the problem of finding the minimal variance gradient estimator. Informed by perturbation-based estimation using Likelihood Ratio (LR) techniques (Jiang et al., 2024;Ren et al., 2025) (detailed reviews of LR techniques are provided in the Appendix A), we propose the Recursive Likelihood Ratio (RLR) optimizer. By utilizing the inherent noise in the DM, a local computational graph is enabled for pathwise gradient estimation as shown in Figure 4, which can reduce the variance and better capture the multi-scale information. The RLR estimator is unbiased and has lower variance under the same computation budgets as other methods. Through perturbation-based computational graph rearrangement, the RLR mitigates the structural bias of truncated BP and the high variance of RL, capable of better capturing multi-scale visual information. Our optimizer shares similarity with zeroth-order optimizer since both need perturbation to estimation gradient, but the RLR utilize a local BP chain to enable low-variance estimation. Therefore, we name it as Half-Order optimizer. Our contributions are threefold:
• We provide a systematic analysis of gradient estimation in DMs, identifying a structured design space of estimators. Then, we formulate the RLR optimizer in the design space under the protocol of minimizing variance with a limited computational budget.
• Extensive evaluation of Text2Image and Text2Video tasks are conducted. RLR consistently achieves higher reward scores across multiple human preference reward models and outperforms SOTA video models on the VBench benchmark. Furthermore, we propose a prompt technique for our RLR, validating the intuition and applicability of our method.
• We conduct a rigorous theoretical analysis of RLR's design, proving its unbiasedness, bounding its estimator variance, and establishing convergence guarantees. These results formally justify the empirical success of RLR and explain the deficiency of prior methods.
this section cite: ['b39', 'b15', 'b10', 'b27', 'b7', 'b34', 'b28', 'b7', 'b29', 'b28', 'b38', 'b21', 'b2', 'b44', 'b9', 'b17', 'b31']

Section: RELATED WORK
Diffusion probabilistic models have achieved state-of-the-art performance in multi-modal generation (Ho et al., 2020;Rombach et al., 2022). However, aligning pre-trained DMs for downstream tasks remains challenging. RL and supervised post-training have been widely explored to incorporate human preferences or task-specific objectives, with recent methods including RLHF-inspired approaches, DPO variants, and reward-model-guided fine-tuning (Ziegler et al., 2019;Fan et al., 2024;Wallace et al., 2024;Prabhudesai et al., 2023). Yet, RL-based fine-tuning often suffers from high variance and sample inefficiency, while truncated BP reduces memory cost but introduces structural bias that can lead to model collapse (Prabhudesai et al., 2023;2024;Xu et al., 2024b). Recent forward-learning methods based on stochastic gradient estimation offer a promising alternative with lower computational and memory costs (Salimans et al., 2017;Peng et al., 2022;Chen et al., 2023). Please refer to Appendix A for extended related works.
this section cite: ['b15', 'b32', 'b56', 'b9', 'b44', 'b28', 'b28', 'b35', 'b26', 'b3']

Section: UNBIASED MINIMAL VARIANCE GRADIENT ESTIMATOR FOR DIFFUSION MODEL

this section cite: []

Section: PROBLEM FORMULATION
DMs generate data by transforming a noise sample x T into a data sample x 0 through a recursive denoising process, as shown in Figure 2. This generation proceeds through T steps of stochastic updates, each governed by a parameterized mapping ϕ t , where the implementation of ϕ t may vary across steps, yielding the transformation:
x t-1 = ϕ t (x t , z t ; θ)(1)
where θ denotes the model parameter, and z t = σ t ϵ t , ϵ t ∼ N (0, I) is a stochastic perturbation. The noise term z t can be either the inherent noise in the DM, i.e., x t-1 = φ(x t ; θ) + z t , or the injected noise to the parameter, i.e., x t-1 = φ(x t ; θ + z t ). The function φ denotes the shared backbone network of the DM, reused across different steps, while ϕ t should be used with the subscript t to indicate which time-step it operates on. The full generation chain can be represented recursively as:
x 0 = ϕ 1:T (x T , z 1:T ; θ) = ϕ 1 • ϕ 2 • • • • • ϕ T (x T , z 1:T ; θ) = ϕ 1 (ϕ 2:T (x T , z 2:T ; θ), z 1 ; θ) = ϕ 1 (ϕ 2:T -1 (ϕ T (x T , z T ; θ), z 2:T -1 ; θ), z 1 ; θ),(2)
where z 1:T are independent noise terms injected at each step. Once a pre-trained model is available, a reward model R(x 0 ) is introduced to guide post-training towards generating samples of higher semantic or perceptual quality. The resulting fine-tuning objective becomes:
max θ E[R(x 0 )] = max θ E z 1:T [R(ϕ 1:T (x T , z 1:T ; θ))].(3)
Fine-tuning DMs through the recursive structure poses a critical challenge: how to efficiently and accurately estimate the gradient of the expected reward with respect to model parameters. A natural objective is to construct a gradient estimator for E[R(x 0 )] that is unbiased and has minimal variance, under a given computational and memory budget. We formulate the problem as the following constrained optimization problem over the space of gradient estimators:
min G∈G Var(G) s.t. ∇ θ E[R(x 0 )] = E[G], C(G) ≤ B,(4)
where G denotes the space of all unbiased gradient estimators G for the objective E[R(x 0 )], with the sample x 0 being obtained via a generative chain defined by x 0 = ϕ 1:T (x T ; z 1:T , θ). The term Var(G) refers to the variance of the estimator, which we aim to minimize. The cost function C(G) quantifies the total computational and memory overhead incurred by the estimator G, typically measured in terms of the length of backward passes or the volume of intermediate activations stored. Finally, B represents the computational budget available for gradient estimation.
this section cite: []

Section: G: THE UNBIASED GRADIENT ESTIMATOR DESIGN SPACE IN DIFFUSION MODELS
We now characterize the feasible design space G for unbiased gradient estimators in DMs. Due to the recursive structure of the generative chain, the gradient estimator must propagate through all T steps. At each step t, we may choose one of three gradient estimation strategies:
• First-Order (FO) uses exact backpropagation through ϕ t . Zeroth-Order (ZO) perturbs the parameters directly, e.g., φ t (x t ; θ + σ t ϵ t ), and estimates the gradient via R(φ(•;θ+σtϵt)) σt ϵ t , based solely on function evaluations (Salimans et al., 2017).
• We propose unbiased Half-Order (HO) gradient estimator which utilizes the inherent noise z t (rather than extrinsic perturbation) and applies the Likelihood Ratio technique, producing an estimator of the form: R(x 0 ) • D ⊤ θ ϕ t:t+h-1 • ∇ log f (z t ), where D θ ϕ t:t+h-1 denotes the Jacobian of a local sub-chain of length h, and f (•) denotes the noise density. The HO allows a h-length sub-chain, starting from any t in the diffusive chain. RL is a special case of HO method with h = 1.
The full estimator design space is thus defined as:
G full := {G = (g 1 , . . . , g T ) | g t ∈ {FO, HO, ZO} ∀1 ≤ t ≤ T },(5)
G RLR = {(g FO 1 , • • • , g HO j , • • • , g HO j+h , • • • , g ZO T )
| 1 ≤ j ≤ T -h}, which consists of one FO at the first step, a HO sub-chain of length h starting at step j, and ZO estimators for all remaining steps to ensure unbiasedness. Under this specific structure, the decision variables are reduced to the length h and the starting index j. Each solution in G RLR is referred to as a Recursive Likelihood Ratio (RLR) estimator (see Figure 4), which takes the form
G = D ⊤ θ ϕ(x 1 , z 1 ; θ) dR(x 0 ) dx 0 One-step first-order estimator -R(x 0 )D ⊤ θ ϕ j:j+h (x j+h , z j:j+h ; θ)∇ z ln f (z j ) h-length half-order estimator - i∈C R(x 0 )∇ z ln f (z i ) zeroth-order estimator ,(6)
where j ∈ [1, T -h] ∩ Z , C = {1, 2, . . . , T } \ {j, j + 1, . . . , j + h}, z j ∼ N (0, σ j I), and z i ∼ N (0, σ i I) for i ∈ C.
Differentiating the reward model. In the first part of the RLR estimator, we apply the FO estimator to the first time step to directly backpropagate through the reward model, avoiding black-box treatment (e.g., RL and ZO) and better leveraging its structure, as shown in Figure 4.
this section cite: ['b35']

Section: Fixed-horizon half-order optimization.
The generation process of the DM follows a coarse-to-fine structure, with every time step in the chain controlling a different scale of generation. Incorporating precise gradient information from every time step is essential, but full BP is computationally prohibitive. Truncated BP introduces bias, while ZO and RL lead to high variance by ignoring structural information. To address this, the RLR optimizer incorporates an HO h-length sub-chain, capturing multi-scale information while minimizing variance. Specifically, the starting index of HO, j ∼ J (1, T -h), is randomly selected across the whole diffusive chain, following a given distribution J (see Section 3.2). The inherent perturbation, z j , enables the localized h-length sub-chain, D θ ϕ t:t+h-1 , effectively capturing the visual scale information represented around that step (see Section 3.2 for choosing h).
Surrogate estimator via parameter perturbation. For the remaining times steps, C = {1, 2, . . . , T } \ {j, j + 1, . . . , j + h}, we inject noise directly into the model's parameters to construct ZO estimation to ensure unbiasedness. This approach is computationally cheap without caching intermediate latent variables.
this section cite: []

Section: OPTIMIZING h AND j: VARIANCE-MEMORY TRADEOFFS
The remaining task is to optimize the two variables in the RLR estimator, h and j, to solve the optimization problem (4). Notably, the choice of j does not directly affect this surrogate objective, but instead influences the ability to capture multi-scale information across different steps, so we treat its decision as a separate problem of interest.
Optimizing h. To reduce the number of problem parameters that need to be estimated, we use an upper bound on the variance of the RLR estimator (see the Appendix J.4) as a surrogate objective:
min h∈N0: G(h)∈GRLR T t=1 Var(g t ) + 2 t̸ =t ′ Var(g t )Var(g t ′ ) s.t. B h h + B z (T -1 -h) ≤ B,(7)
where h and (T -1-h) are the number of steps for HO and ZO; B h and B z are coefficients indicating the magnitude of the memory cost of HO and ZO per step. In practice, the available budget satisfies B z T < B < B h T , meaning that using pure ZO underutilizes the budget, while using pure HO exceeds it. Let V 2 h and V 2 z denote the per-step variance of the HO and ZO, respectively. Since HO and FO have much lower variance than ZO, we use a common V h for both HO and FO, and assume V h ≪ V z and T > 2. These conditions ensure the optimization problem admits the solution:
h * = min{⌊ B -B z (T -1) B h -B z ⌋, ⌊ T V z 2(V z -V h ) -1⌋} > 0.(8)
We set B h = 8GB and B z = 0.24GB, which is supported by empirical evidence in Table 9 in the Appendix. If the memory budget B is between 30GB and 40GB. It is recommended to set h = 2. As the formula (7) indicates, the variance decreases as h increases. However, the performance exhibits diminishing improvement with increasing h. In other aspects, the memory consumption grows linearly with h, and the computational time grows even more rapidly. The above claims are corroborated by our ablation results in Table 9. Therefore, even with a larger memory budget, blindly increasing h is not advisable. Moreover, since V h ≪ V z , the second term in (8) simplifies to approximately T 2 -1 ≈ 24, which is typically larger than the first term. As a result, we have h * = ⌊(B -B z (T -1))/B h -B z ⌋ in practice, and there is no need to estimate V h and V z .
Determining j. We use the gradient norm to represent the importance of different steps. Then sample j from the categorical distribution j
∼ J (1, T -h) = CAT (Softmax(∥g 1 ∥, • • • , ∥g T -h ∥)).
this section cite: []

Section: EXPERIMENTS
We verify the superiority and applicability of the RLR optimizer against various baselines on two generation tasks: Text2Image and Text2Video. We compare the RLR with the RL-based method (DDPO), and the truncated-BP-based methods (Alignprop and VADER). Moreover, other baselines, e.g, closed-source models, are also included. Finally, we propose a novel prompt technique that is natural for the RLR optimizer, demonstrating the enhanced capability of the proposed RLR optimizer to capture multi-scale information for visual generation. Ablations are included to verify the validity of the proposed RLR optimizer. Please refer to Appendix B for detailed settings.
this section cite: []

Section: TEXT2IMAGE GENERATION
We evaluate our methods on two DMs: Stable Diffusion 1.4 and 2.1 (Rombach et al., 2022). As shown in Table 2, the RLR methods achieve higher reward scores on the unseen prompts from the test set. The RL-based method have limited improvement with respect to the base model, due to the sample inefficiency nature. Alignprop has considerable improvement over the base model. However, the biased estimator limits its further improvement. Training details and hyperparameters can be found in the appendix.
this section cite: ['b32']

Section: Sample efficiency analysis.
The compare the sample efficiency and the variance of different methods, we show the reward curves of SD 1.4 when training on the AES and HPS v2 models in Figure 5. The DDPO optimizes the reward at a very slow pace, indicating high variance and low sample efficiency. In the earlier phase, the AlignProp has comparable performance as the RLR. In the later phase, while the RLR can continue to improve the reward, the AlignProp suffers from severe model collapse.
this section cite: []

Section: TEXT2VIDEO GENERATION
We compare our RLR not only with RL and truncated BP but also with a series of open-source or API-based Text2Video models. In the metric of DD and AQ, the RLR surpasses other methods by a large margin. In other metrics, RLR achieves considerable improvement over the base model, VideoCrafter. Some API-based models have better performance on some metrics, but the gaps are small. In terms of the weighted average score, our RLR has the best performance over all baselines. Furthermore, we propose the Diffusive Chain-of-Thought (DCoT), a prompt technique that is natural for our RLR optimizer to demonstrate the applicability of our method. The core idea is that DMs generate content in a multi-scale (coarse-to-fine) manner, and deficiencies at a particular scale can be addressed by focusing gradient updates on the corresponding steps of the diffusion process by the HO sub-chain. We propose dividing all the diffusion process steps into three groups: coarse-level, mid-level, and fine-level. The coarse-level chain includes steps adjacent to the initial noise, focusing on generating a rough outline. The fine-level chain includes steps adjacent to the final output, focusing on the fine-grained details. The mid-level chain in between focuses on the geometric structure of the content. The idea of DCoT is shown in Figure 9, which converts the original prompt into multi-scale prompts reflecting the coarse-to-fine nature. Different generation steps should be conditioned on different prompts instead of being conditioned on the same prompt. The HO estimator term in the RLR enables a h-step local computational chain for low-variance, unbiased gradient estimation. By integrating DCoT, we can target the HO sub-chain precisely at the time steps (i.e., scales) where generation is deficient, as revealed by the multi-scale prompt decomposition. The HO estimator term uniformly picks a starting point j ∼ U(1, T -h) from the entire T -step chain to start the local h-step BP chain. When applying the DCoT to the fine-tuning process, we should constrain the sample range of j in the steps where deficiencies exist, j ∼ U(a, b), 1 < a < b < T -h, b -a > h. In our experiment for the hand task, we set a = 30 and b = 40.
We write 5 prompts for hand generation and then prompt ChatGPT to generate the multi-scale prompts for the three levels. We report the performance in We conduct the ablation study, using SD 1.4 and HPD v2, to verify the contribution of different parts in the RLR optimizer. In Table 5, we evaluate the RLR and its variants (V1: the RLR without HO and ZO; V2: the RLR without ZO; V3: the RLR without HO). The V1 performs the worst since it actually reduces to the truncated BP with only one time-step. The V2 and V3 perform better than the V1. It is worth noting that the V2 is better than the V3. The V3 without HO is actually an unbiased estimator since it takes all time steps into account when estimating the gradient. Even though the V2 rearranges the computational graph by the HO, it is still a biased estimator. This phenomenon indicates the importance of unbiasedness when conducting the fine-tuning task.
this section cite: []

Section: THEORETICAL PROPERTIES OF GRADIENT ESTIMATORS: BIAS, VARIANCE, AND CONVERGENCE
In this part, we analyze the bias of truncated BP and compare the variance of different estimators, backing the claim in Table 1. Thanks to the unbiasedness of the RLR estimator, the convergence of the optimization is also guaranteed.
To alleviate the memory burden of full-step BP, the truncated variant is often employed, backpropagating the gradient with only T ′ steps; T ′ ≪ T . However, the truncation introduces a structural bias into the gradient estimator. We have the following proposition to justify this structural bias.
Proposition 6.1 (Biasedness of Truncated-BP ). Assume R and ϕ are differentiable almost everywhere, R and ϕ t have bounded gradients, then the FO estimator is unbiased. However, the truncated BP estimator ∇ θ R(x 0 ) truncated has a structural bias, which can be specified as below:
∇ θ E[R(x 0 )] -E[∇ θ R(x 0 ) truncated ] = E z 1:T T i=T ′ +1 ∂ϕ i (x i , z i ; θ) ∂θ i-1 j=1 ∂x j-1 ∂x j ⊤ dR(x 0 ) dx 0 .
Bias in the estimator can lead to suboptimal updates or even training failure, as the truncated gradient may not follow a true descent direction. This can cause two major issues: model collapse and loss of multi-scale information. In contrast, ZO (Spall, 1992) and HO (Jiang et al., 2024) are unbiased.
The stochastic nature of the DM results in the variance of the estimator. As expected, the variance of the FO estimator is lower than that of the HO and ZO estimators because the differentiation leverages the structural information of the neural network. However, BP introduces significant computational and storage overhead. The following proposition demonstrates that this additional cost is, to some extent, justified, as BP leverages accurate internal structures to reduce estimation variance.
Proposition 6.2 (Variance Comparison). Under Assumptions (A.1-3) in the Appendix, the variance of FO estimators is less than or equal to ZO estimators, i.e.
Var(∇ θ R(x 0 )) ≤ Var(R(x 0 )∇ ln f (z)).(9)
Based on the above proposition, it is straightforward to conclude that the variance of the HO estimator is also less than or equal to that of the ZO estimator, as it is essentially an FO estimator with a perturbation at the start of the sub-chain. The Table 1 presents all the gradient computation methods discussed above.
Overall, the RLR reorganizes the recursive computation chain by perturbation-based estimation, seamlessly integrating ZO, HO, and FO optimization techniques. RLR strikes a balance between computational cost and gradient accuracy, achieving both unbiasedness and low variance. The following Theorem 6.3 establishes its unbiasedness.
Theorem 6.3 (Unbiasedness of RLR). The RLR estimator is an unbiased estimator:
∇ θ E[R(ϕ 1:T (x T ; θ))] = E z 1:T ,j∼U (1,T -h) D ⊤ θ ϕ 1 (x 1 , z 1 ; θ) dR(x 0 ) dx 0 -R(x 0 )D ⊤ θ ϕ j:j+h (x j+h , z j:j+h ; θ)∇ z ln f (z j ) - i∈C R(x 0 )∇ z ln f (z i ). .(10)
The variance of the RLR estimator, denoted by σ 2 RLR , is discussed in the appendix. Under limited computational resources where full BP is infeasible, RLR achieves substantially lower variance compared to other unbiased gradient estimators. Finally, the convergence rate of RLR is provided in the following Theorem 6.4. Theorem 6.4 (Convergence Rate). Suppose that the reward function R(•) is L-smooth. By appropriately selecting the step size, the convergence rate of the RLR is given by
1 K + 1 K k=0 E(∥∇R(θ k )∥ 2 ) ≤ 8L∆ 0 σ 2 RLR K + 1 + 2L∆ 0 K + 1 ,
where K is the number of iterations, θ k is the trainable parameter in the k-th iteration, and
∆ 0 = |R(θ 0 ) -R *
| is difference between initialization performance and optimal performance.
this section cite: ['b17']

Section: CONCLUSION
We propose the RLR optimizer, a half-order gradient estimation framework designed for efficient finetuning of diffusion models. Theoretically, we analyze the bias, variance, and convergence of the RLR estimator and formulate a constrained optimization problem to guide its design. Empirically, RLR consistently outperforms both reinforcement learning and truncated backpropagation methods on Text2Image and Text2Video tasks across multiple human preference reward models and benchmarks. Furthermore, we introduce a novel prompt technique, Diffusive Chain-of-Thought (DCoT), which complements the RLR and further boosts performance. Although determining the appropriate subchain length h can be nontrivial in practice, we provide both theoretical justification and empirical ablations to guide practitioners in making informed choices.
APPENDIX CONTENTS A Extended Related Works 15 B Experiments settings 15 B.1 Overall Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15 B.2 Prompts . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 B.3 Reward Models and Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . 16 B.4 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16 B.5 Orthogonal tricks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17 C Hypeparameters 17 D Qualitative Results of Text2Image 18 E Qualitative Results of Text2Video 19 F Details for Diffusive Chain-of-Thought experiment 20 G Memory Profile, Time Complexity, and Different Diffusion Solvers 21 H Ablations on the Sub-chain Length h 22 I More Results on Vbench 22 J Proofs 23
A EXTENDED RELATED WORKS Diffusion Probabilistic Models. Denoising Diffusion Model (Ho et al., 2020;Lu et al., 2022) is one of the strongest models for generation tasks, especially for visual generation (Rombach et al., 2022;Peebles & Xie, 2023;Chen et al., 2024a). Extensive research has been conducted from theoretical and empirical perspectives (Song et al., 2020a;Karras et al., 2022;Song et al., 2020b). It has achieved phenomenal success in multi-modality generation, including image, video, audio, and 3D shapes. The DM is trained on enormous images and videos from the internet (Bain et al., 2021;Wang et al., 2023b;Schuhmann et al., 2022). Empowered by modern architecture (Vaswani, 2017), it has a powerful learning capability for Pixel Space Distribution.
Alignment and Post-training. After pre-training to learn the distribution of the targeted modality (Achiam et al., 2023;Kaplan et al., 2020), post-training is conducted to align the model toward specific preferences or tune the model to optimize a particular objective. RL has been utilized to align the foundation models toward various objectives (Ziegler et al., 2019;Lambert et al., 2022;Black et al., 2023). DPOK (Fan et al., 2024) studies KL regularization when training a separate DM for each prompt. Supervised learning can also be applied to the post-training phase (Rafailov et al., 2024), either optimizing an equivalent objective (Wallace et al., 2024) or directly differentiating the reward model (Clark et al., 2023;Prabhudesai et al., 2023;2024). D3PO Yang et al. (2024) utilizes the DPO loss to train the DM. Specialist Diffusion Lu et al. (2023) focuses on sample-efficient, few-shot fine-tuning of large pre-trained diffusion models to enable the generation of new visual styles from as few as 5-10 examples. SPIN (Yuan et al., 2024b) introduces a self-play learning paradigm for diffusion models. For DM, most methods use a neural reward model to align the pre-trained model, and there has been a continual effort to design better reward models (He et al., 2024;Xu et al., 2024a;b).
this section cite: ['b15', 'b22', 'b32', 'b25', 'b19', 'b1', 'b37', 'b43', 'b0', 'b18', 'b56', 'b20', 'b2', 'b9', 'b30', 'b44', 'b7', 'b28', 'b23', 'b13']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Frozen in time: A joint video and image encoder for end-to-end retrieval Year: (2021)
Ref_id:b2 Title: Training diffusion models with reinforcement learning Year: (2023)
Ref_id:b3 Title: Scaling up zeroth-order optimization for deep model training Year: (2023)
Ref_id:b4 Title: Videocrafter2: Overcoming data limitations for high-quality video diffusion models Year: (2024)
Ref_id:b5 Title: Training deep nets with sublinear memory cost Year: (2016)
Ref_id:b6 Title: Enhancing zeroth-order fine-tuning for language models with low-rank structures Year: (2024)
Ref_id:b7 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2023)
Ref_id:b8 Title: On the variance of single-run unbiased stochastic derivative estimators Year: (2020)
Ref_id:b9 Title: Reinforcement learning for finetuning text-to-image diffusion models Year: (2024)
Ref_id:b10 Title: Toward theoretical insights into diffusion trajectory distillation via operator merging Year: (2025)
Ref_id:b11 Title: Gradient estimation via perturbation analysis Year: (1990)
Ref_id:b12 Title: Memory-efficient backpropagation through time Year: (2016)
Ref_id:b13 Title: Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation Year: (2024)
Ref_id:b14 Title: The forward-forward algorithm: Some preliminary investigations Year: (2022)
Ref_id:b15 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b16 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b17 Title: One forward is enough for neural network training via likelihood ratio method Year: (2024)
Ref_id:b18 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b19 Title: Picka-pic: An open dataset of user preferences for text-to-image generation Year: (2022)
Ref_id:b20 Title: Illustrating reinforcement learning from human feedback (rlhf) Year: (2022)
Ref_id:b21 Title: Aligning text-to-image models using human feedback Year: (2023)
Ref_id:b22 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b23 Title: Specialist diffusion: Plug-and-play sample-efficient fine-tuning of text-to-image diffusion models to learn any unseen style Year: (2023)
Ref_id:b24 Title: Fine-tuning language models with just forward passes Year: (2023)
Ref_id:b25 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b26 Title: A new likelihood ratio method for training artificial neural networks Year: (2022)
Ref_id:b27 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b28 Title: Aligning text-to-image diffusion models with reward backpropagation Year: (2023)
Ref_id:b29 Title: Video diffusion alignment via reward gradients Year: (2024)
Ref_id:b30 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b31 Title: Flops: Forward learning with optimal sampling Year: (2025)
Ref_id:b32 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b33 Title: Real and Complex Analysis Year: (1987)
Ref_id:b34 Title: Learning representations by back-propagating errors Year: (1986)
Ref_id:b35 Title: Evolution strategies as a scalable alternative to reinforcement learning Year: (2017)
Ref_id:b36 Title: Improved aesthetic predictor Year: (2022)
Ref_id:b37 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b38 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b39 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b40 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b41 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b42 Title: Multivariate stochastic approximation using a simultaneous perturbation gradient approximation Year: (1992)
Ref_id:b43 Title: Attention is all you need Year: (2017)
Ref_id:b44 Title: Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b45 Title: Modelscope text-to-video technical report Year: (2023)
Ref_id:b46 Title: Internvid: A large-scale video-text dataset for multimodal understanding and generation Year: (2023)
Ref_id:b47 Title: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis Year: (2023)
Ref_id:b48 Title: Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation Year: (2024)
Ref_id:b49 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2024)
Ref_id:b50 Title: Using human feedback to fine-tune diffusion models without any reward model Year: (2024)
Ref_id:b51 Title: Instructvideo: instructing video diffusion models with human feedback Year: (2024)
Ref_id:b52 Title: Self-play fine-tuning of diffusion models for text-to-image generation Year: (2024)
Ref_id:b53 Title: Revisiting zeroth-order optimization for memory-efficient llm fine-tuning: A benchmark Year: (2024)
Ref_id:b54 Title: Learning stochastic dynamics from snapshots through regularized unbalanced optimal transport Year: (2024)
Ref_id:b55 Title: Second-order fine-tuning without pain for llms: A hessian informed zeroth-order optimizer Year: (2024)
Ref_id:b56 Title: Fine-tuning language models from human preferences Year: (2019)
Ref_id:b57 Title:  Year: ()
Ref_id:b58 Title: Proof of Proposition 6 Year: ()
Ref_id:b59 Title:  Year: ()
Ref_id:b60 Title:  Year: ()
Ref_id:b61 Title:  Year: ()
Ref_id:b62 Title: Solution of Year: ()
