Title: TOKEN-IMPORTANCE GUIDED DIRECT PREFERENCE OPTIMIZATION
Abstract: Aligning Large Language Models (LLMs) with human preferences is crucial for safe and effective AI interactions. While popular methods like Direct Preference Optimization (DPO) have simplified alignment, they remain sensitive to data noise and overlook the differential importance of individual tokens. Existing token-level approaches often rely on probability prediction or simplistic weighting schemes to obtain token importance, which still cannot fully address these issues. To solve this problem, we propose the Token-Importance Guided Direct Preference Optimization (TI-DPO), a framework that achieves fine-grained semantic control through two synergistic innovations. First, we propose a novel hybrid weighting mechanism that combines gradient attribution with a Gaussian prior, ensuring both the accuracy and robustness of token importance scores. Second, we employ a triplet loss to provide structured guidance for the optimization, explicitly guiding model outputs to approach preferred responses and diverge from nonpreferred ones. Experimental results show that TI-DPO achieves higher accuracy and stronger generative diversity, providing more stable and computationally efficient solutions compared with DPO and other RLHF methods. Code and demo are available at https://github.com/gracefulning/TIDPO.

Section: INTRODUCTION
Large Language Models (LLMs) have shown proficiency in Natural Language Processing (NLP) (Gao et al., 2025), logical reasoning (Xie et al., 2025), and code generation (Xu et al., 2025), emerging as a focal point of recent research. However, as models may generate outputs inconsistent with intended purposes or ethical standards, human preference alignment aims to ensure that LLMs adhere to human values (Liu et al., 2023), producing beneficial and harmless content. Against this backdrop, Reinforcement Learning from Human Feedback (RLHF) has become a prevailing approach for achieving alignment (Hong et al., 2024;Hu et al., 2025). It leverages human-annotated preference data to train reward models and fine-tunes LLMs using Reinforcement Learning (RL) methods (Wang et al., 2023b) like Proximal Policy Optimization (PPO) (Schulman et al., 2017).
The emergence of Direct Preference Optimization (DPO) has effectively simplified the alignment process (Rafailov et al., 2023). Inspired by DPO's implicit reward mechanism, a series of preference optimization models have been proposed in recent years, such as ORPO (Hong et al., 2024), f-DPO (Wang et al., 2023a), and CPO (Feng et al., 2025). However, both DPO and RLHF have a fundamental flaw during optimization: they optimize at the sequence level, leading to the neglect of the influence of specific tokens, which in turn destabilizes the training process due to shifts in the sampling distribution (Zhang et al., 2025).
Motivated by these challenges, researchers have proposed token-level variants of DPO, aiming to decompose preference alignment into fine-grained contributions ( (Zeng et al., 2024;Xie et al., 2025;Zhong et al., 2024)). However, achieving true fine-grained alignment requires addressing a core challenge: We not only need to accurately identify the key tokens that have a decisive impact on human preferences, but also need a subtle optimization objective to guide the model to adjust its preference (Li et al., 2025). Nevertheless, existing token-level methods fall short in dealing with this challenge for two reasons. First, their approaches to identifying key tokens often rely on biased probability proxies (Liu et al., 2024a) or overly simplified weighting schemes (Lin et al., 2024). Second, the optimization, they still inherit the binary comparison framework of DPO, simply distinguishing between "good" and "bad" samples (Meng et al., 2024). Such coarse-grained supervision signals cannot finely shape the model's generation behavior in a continuous semantic space.
In our Token-Importance Guided Direct Preference Optimization (TI-DPO) framework, we introduce a novel hybrid weighting mechanism to accurately and robustly identify key tokens. This mechanism combines gradient attribution with a Gaussian prior, overcoming the problem of existing methods relying on biased proxies. Here, gradient attribution is a technique used to determine the contribution of each input feature (in our work, each token) to the model's output (Ancona et al., 2017;Ballout et al., 2024). Liu et al. (2024b) offered empirical evidence that models exhibit a Ushaped attention bias, which means there is greater importance to tokens at the beginning and end of a sequence, while underweighting those in the middle. Thus, the Gaussian prior distribution here is explicitly designed to rectify this intrinsic architectural bias, ensuring that the optimization process does not neglect the semantic core of the response.
Meanwhile, we adopt a structured triplet objective based on the identified key weights to achieve fine-grained optimization by incorporating the intermediate generated outputs (Nguyen et al., 2018). This triplet structure explicitly guides the intermediate output to approach human preferences and distance from non-preferred responses, achieving fine-grained preference alignment and promoting a continuous gradient of preference learning. The mixed weights and the triplet loss complement each other and together form a complete solution for TI-DPO to achieve fine-grained alignment.
The following contributions are made in the course of this work:
• We propose TI-DPO, a novel framework designed for achieving fine-grained alignment.
This framework innovatively integrates a hybrid weighting mechanism, jointly formed by gradient attribution and a Gaussian prior, with triplet loss, significantly enhancing the robustness and stability of weight allocation.
• Theoretically, we formalize the TI-DPO framework by providing a complete derivation of its loss function and gradient. Building on this, we prove TI-DPO achieves a tighter loss bound than DPO (Theorem 2) and the superiority of expected reward (Theorem 3). This theorem formally provides a new perspective on comprehending the superiority of TI-DPO in terms of alignment accuracy.
• Experiment results indicate that TI-DPO surpasses existing methods in aligning LLMs with human preferences. Notably, our method achieves a leading average score of 62.3, and substantially outperforms strong baselines on key tasks such as HumanEval, TruthfulQA and IFEval with scores of 67.0, 62.0 and 75.7 respectively. Further analysis, including ablation studies and sensitivity analysis, confirms that both of our core contributions are vital to this performance.
this section cite: ['b9', 'b39', 'b41', 'b27', 'b16', 'b17', 'b34', 'b31', 'b16', 'b8', 'b44', 'b42', 'b39', 'b45', 'b44', 'b24', 'b28', 'b0', 'b3', 'b29']

Section: RELATED WORK

this section cite: []

Section: Human Preference Alignment
Human preference alignment has emerged as a critical research paradigm in recent years, focusing on enabling model responses to align with human values and preferences. Early advancements mainly focused on RLHF (Ouyang et al., 2022;Bai et al., 2022) based on PPO (Schulman et al., 2017). However, these RL methods may suffer from overfitting in optimal responses. To mitigate this issue, Hu et al. (2025) introduced the Reinforce++ model, which employs batch-wise standardized rewards to prevent overfitting and enhance the prompt diversity during training. Concurrently, beyond RL approaches, Rafailov et al. (2023) introduced DPO, which obviates the need for explicit reward modeling through implicit preference learning. This implicit reward mechanism has inspired a wave of subsequent works (Cui et al., 2025), such as the SimPO algorithm proposed by Meng et al. (2024), which utilizes the sequence-averaged log probability as an implicit reward signal to streamline optimization. Notwithstanding these advancements, DPOs' reliance on large-scale human-annotated preference datasets (Kim et al., 2025) has motivated derivative studies (Gou & Nguyen, 2024;Jiao et al., 2024) aimed at reducing data requirements. A notable example is RS-DPO (Khaki et al., 2024), which integrates rejection sampling with DPO to alleviate data scarcity. However, a more fundamental limitation pertains to the binary nature of traditional preference labels. Although the KTO method proposed by Ethayarajh et al. (2024) effectively reduces the reliance on paired preference labels in DPO, most current RLHF and related preference optimizations still mainly rely on binary comparisons between "good" and "bad" responses (Gao et al., 2024;Hong et al., 2024). Such coarse-grained supervision has obvious shortcomings: human preferences often show continuous gradient differences rather than simply "good" and "bad". Against this backdrop, our proposed triplet optimization method can achieve fine-grained preference alignment.
From Sequence-Level to Token-Level Achieving fine-grained alignment requires the model not only to distinguish the quality of the entire sequence, but also to understand and precisely control the key morphemes that constitute the semantics of the sequence. Existing sequence-level techniques often lead to a decrease in generation diversity because they ignore the importance differences among tokens (Feng et al., 2025). These limitations have spurred researchers' research into step (Xie et al., 2024) or token-level (Rafailov et al., 2024) alignment mechanisms, seeking to address the granularity mismatch between coarse-grained sequence rewards and fine-grained token contributions (Xi et al., 2024). To address the significant decline in model generation diversity, TDPO (Zeng et al., 2024) reanalyzed and optimized the entire alignment process from the token-level perspective. An additional limitation of RLHF and DPO lies in the fact that rewards are only assigned to the final token, with all other tokens receiving no learning rewards (Zhong et al., 2024). Meanwhile, Xie et al. (2025) proposed a correlation between the frequency of specific tokens and model performance, which inspires us to consider reassigning token weights. In a related vein, Liu et al. (2024a) estimates token importance weights using prediction probability differences. Nevertheless, this probabilistic weighting scheme is prone to bias when contrastive models produce inconsistent outputs or fail to capture subtle semantic nuances of human preferences. In contrast, our approach employs a hybrid strategy that combines causal gradient attribution with a stabilizing Gaussian prior to estimate importance (Ballout et al., 2024). By focusing on actual gradient impacts, our method enhances alignment precision over probabilistic proxies, while the prior distribution ensures robustness against noisy gradient signals.
this section cite: ['b30', 'b2', 'b34', 'b17', 'b31', 'b6', 'b28', 'b21', 'b11', 'b19', 'b20', 'b7', 'b10', 'b16', 'b8', 'b40', 'b32', 'b38', 'b42', 'b45', 'b39', 'b3']

Section: PRELIMINARIES
Before formally elaborating the TI-DPO method, this section first introduces relevant preparatory knowledge to lay the foundation for the subsequent theoretical derivation and model construction.
this section cite: []

Section: HUMAN PREFERENCE ALIGNMENT
Firstly, we focus on the core concept of human preference alignment, which is the foundation for optimizing the response generation of LLMs. Suppose that x stands for the input prompt and y denotes the response generated by the model. The key approach involves optimizing the response-generation policy π θ (y|x). It utilizes a carefully selected human preference dataset D = {(x, y w , y l )}. Here y w and y l represents preferred response and non-preferred response. Reward model r ϕ (x, y) evaluates the LLMs' responses by applying the Bradley-Terry (BT) model for ranking loss Ouyang et al. (2022). The loss function employed to access the reward model r ϕ using dataset D is formulated as follows:
L base = -E (x,yw,yl)∼D [log σ (r ϕ (x, y w ) -r ϕ (x, y l ))].(1)
Here σ(•) is the sigmoid activation function. The reward model evaluates the LLMs' responses by applying the BT model for ranking losses (Ouyang et al., 2022):
p (y w ≻ y l | x) = exp (r ϕ (x, y w )) exp (r ϕ (x, y w )) + exp (r ϕ (x, y l )) ,(2)
The partition function Z(x) serves to normalize the policy's probability distribution (Rafailov et al., 2023). The parameter β regulates the extent of divergence between π θ and π ref .
DPO rearranges this equation to express the reward as r ϕ (x, y) = β log
π * θ (y|x) πref(y|x) -log Z(x).
Let the input prompt be represented as x = [x 1 , x 2 , . . . , x m ] and the first t -1 tokens generated by the model be denoted as y <t = [y 1 , y 2 , . . . , y t-1 ]. Let T w and T l denote the number of preferred tokens and less preferred tokens, respectively. The token-level DPO optimization objective is given by
L DPO = -E (x,yw,yl)∼D log σ β log π θ (y w |x) π ref (y w |x) -log π θ (y l |x) π ref (y l |x) ,(3)
3.2 TRIPLET LOSS Triplet loss, a powerful loss function for learning embeddings, ensures that within the embedding space, an anchor input is closer to positive inputs than to negative ones. This mechanism enhances the model's capacity to differentiate between data points that are more or less similar. By simultaneously learning from the similarities and differences among sampled data points, the model is better aligned with human evaluations. The triplet loss operates with triplets (x i , x j , x k ), and is designed such that the representation of the anchor x i is nearer to a similar data point x j than to a dissimilar one x k . This targeted learning strategy is instrumental in sharpening the model's feature discrimination, thereby improving its ability to make decisions that resonate with human preferences. The triplet loss is given by
L trp = T i,j,k ∥f (x i ) -f (x j )∥ 2 2 -∥f (x i ) -f (x k )∥ 2 2 + α trp + .(4)
Here [z] + denotes the rectified linear unit function, ensuring that it is set to zero if negative. The features extracted from the three inputs are represented by the terms f (x i ), f (x j ), and f (x k ).
this section cite: ['b30', 'b30', 'b31']

Section: METHODOLOGY
Driven by the challenges of unstable training and distribution shift in traditional RL alignment methods, we propose the TI-DPO framework. Our key innovation lies in a novel hybrid weighting strategy and a triplet loss that provides a structured optimization objective.
this section cite: []

Section: TOKEN-LEVEL MDP FOR LLM PREFERENCE ALIGNMENT
To address the challenges of the sequential and auto-regressive nature of text generation, a tokenlevel Markov Decision Process (MDP) is introduced, which incorporates the notion of token significance to improve the alignment of each token selection with human preferences. This concept is defined through a tuple denoted as M = (S, A, P, r, ρ 0 ). S and A are the state space and action space, respectively. P is a deterministic transition model among tokens. Here r stands reward model associated with each token, and ρ 0 indicates the initial state distribution. The initial state is s 0 = [x], which is simply the input prompt. At each step t of the generation process, the state s t = [x, y <t ] ∈ S consists of input prompt x, where t is the count of token, and t -1 generated tokens y <t = [y 1 , y 2 , . . . , y t-1 ]. At each time step t, the action a t = y t corresponds to the selection of subsequent tokens for generation.
this section cite: []

Section: HYBRID WEIGHTING MECHANISM FOR TOKEN IMPORTANCE
Building on the token-level MDP framework, we formalize the calculation of importance weights w t . Inspired by the attribution-based rationale extraction from Ballout et al. (2024), our approach quantifies token importance through gradient sensitivity analysis, ensuring that critical tokens in human-preferred responses drive the policy optimization process.
However, while gradient attribution provides a precise, data-driven signal, it can be susceptible to noise. Some studies have pointed out that imposing additional constraints on the attention or importance distribution can help the model focus on key information (Zhang et al., 2018;Guo et al., 2019). Furthermore, a recent study (Liu et al., 2024b) offered empirical evidence that models exhibit a U-shaped attention bias. This means there is greater importance to tokens at the beginning and end of a sequence, while underweighting those in the middle. The Gaussian prior distribution, which peaks at the center, is designed to counteract the architectural "Lost-in-the-Middle" bias inherent in LLMs, ensuring the optimization does not neglect the semantic core of the response. The Gaussian prior also prevents the model from overfitting to noisy gradient signals and provides a stable baseline, ensuring the weight distribution remains well-behaved throughout training.
A token that significantly impacts the reward is deemed critical, whether it has a positive effect on the preferred response or a negative effect on the non-preferred response. For a given sequence of tokens y = [y 1 , y 2 , . . . , y T -1 ], we first obtain its embedding sequence E = [e 1 , e 2 , . . . , e T -1 ], where e i is the embedding vector for token y i . We then perform a forward pass to get the logits for the final token, L T -1 ∈ R V , where V is the vocabulary size. The target scalar value for our gradient calculation, L target , is the maximum logit value at this final step, which represents the model's most confident prediction for the next token y T :
L target = max(L T -1 ).(5)
Next, we compute the gradient of this target logit with respect to each token's embedding e i in the sequence. This gradient, ∇ ei L target , captures the direct influence of token i on the final prediction.
To obtain a scalar importance score I i from the gradient vector, similar to previous work (Ballout et al., 2024), we compute its L 1 norm:
I i = ||∇ ei L target || 1 = k |(∇ ei L target )[k]|.(6)
Here, k indexes the components of the gradient vector. This score, I i , represents the raw, data-driven importance of token i.
Finally, to ensure training stability and robustness against noise in gradient estimates, we postprocess these raw scores to derive the final weights w t . As implemented in our code, this involves a mixed strategy: (1) First, the raw scores I = {I 1 , . . . , I T -1 } are normalized by their sum to form a distribution I norm .
(2) Then we define a Gaussian-shaped prior distribution P prior centered on the sequence, which assigns higher baseline importance to tokens in the middle. For a sequence of length T and each token position t ∈ [0, T -1], the unnormalized value is calculated as:
P prior (t) = exp - 1 2 t -µ σ 2 ,(7)
where we heuristically set the mean µ = (T -1)/2 and the standard deviation σ = T /4. Since approximately 95% of the mass of a Gaussian distribution lies within ±2σ, setting 4σ ≈ T ensures the prior effectively spans the entire sequence context without being too narrow or too flat.
The final weight vector W is a convex combination of these two distributions, controlled by a hyperparameter λ ∈ [0, 1]:
W = λ • I norm + (1 -λ) • P prior .
(8) This weighting scheme is applied independently to both the preferred y w and non-preferred y l sequences to obtain their respective token-level weights, denoted as w w t and w l t . The gradient-based importance guidance method provides a data-driven measure of token relevance, which can adapt to the subtle semantics of human preferences and achieve fine-grained control over key tokens during the model generation process. These weights then modulate the implicit reward signal at each token step, effectively focusing the DPO objective on the most critical tokens. The resulting preference probability under BT model is:
p * (y w ≻ y l ) = exp Tw t=1 w w t • r ϕ (s w t , a w t ) exp Tw t=1 w w t •r ϕ (s w t , a w t ) +exp Tl t=1 w l t •r ϕ (s l t , a l t ) .(9)
Here, T w and T l are the lengths of y w and y l respectively.
Then, with r ϕ (s t , a t ) = β log π θ (y t |x,y <t ) πref(y t |x,y <t ) in DPO, we can derive the expression for BT model:
p * (y w ≻ y l ) = σ(∆r token (x, y w , y l , w w t , w l t )),(10)
where ∆r token (x, y w , y l , w w t , w l t ) can be denoted as:
∆r token (x, y w , y l , w w t , w l t ) = Tw t=1 w w t log π θ (y t w | x,y <t w ) π ref (y t w | x,y <t w ) - Tl t=1 w l t log π θ y t l | x,y <t l π ref (y t l | x,y <t l ) .(11)
Therefore, we obtain the weighted token-level DPO base loss as:
L DPO-w = -E (x,yw,yl)∼D log σ ∆r token (x, y w , y l , w w t , w l t ) .(12)
this section cite: ['b3', 'b43', 'b13', 'b3']

Section: TRIPLE LOSS IMPLEMENTATION
The practical implementation of the triplet loss is integrated seamlessly within the main training loop to provide structured guidance for the policy model.
First, for each data batch comprising (x, y w , y l ), the process begins by generating an anchor response y: Using the preferred response y w as the starting point of the context, the response dynamically generated by the policy model π θ is the anchor y. It represents an intermediate state in the model's generation space.
Next, by mapping each of the three responses to a point in a continuous preference space, we calculate the distances between these three responses y, y w and y l , which represent the preference for the newly created anchor y.
Finally, according to the definition in Eq.( 4), the triplet loss in our work is calculated with these distances, penalizing the model if the anchor is not closer to the positive response than to the negative one by a predefined margin:
Ltriplet = E (x,yw,y l )∼D max(0, Tw t=1 log π θ (y t |x, y <t ) πref(y t |x, y <t ) -log π θ (y t w |x, y <t w ) πref(y t w |x, y <t w ) 2 2 Align y with yw - T l t=1 log π θ (y t l |x, y <t l ) πref(y t l |x, y <t l ) -log π θ (y t |x, y <t ) πref(y t |x, y <t ) 2 2 Push y away y l +α) + .(13)
this section cite: []

Section: TI-DPO OBJECTIVE AND THEORETICAL ANALYSIS
We now formally define the complete TI-DPO objective, which unifies our hybrid weighting and triplet loss mechanisms, and provide a theoretical proof of its superiority over standard DPO. With given TI-DPO dataset D = {(x, y w , y l )}, we obtain TI-DPO objective:
L TI-DPO = L DPO-w + γL triplet , (14
)
where γ is a hyperparameter. In Appendix A.4, we have given the proof of gradient ∇ θ L TI-DPO , which is used to update θ during training. The implementation of TI-DPO is shown in Algorithm 1.
Algorithm 1 TI-DPO 1: Input: Dataset D = {(x, y w , y l )}, hyperparameter β, α, λ, reference model π ref , policy model π θ . 2: Initialize: π θ ← π ref 3: for each epoch do 4: Sample batch {(x, y w , y l )} ∼ D. 5:
Compute raw importance scores I via gradient attribution.
6: Compute weights {w w t } and {w l t } by mixing normalized scores with a Gaussian prior P prior : W ← λI norm + (1 -λ)P prior . 7:
Compute weighted DPO log-ratio:
∆r token ← t w w t log π θ (y t w |x,y <t w ) πref(y t w |x,y <t w ) -t w l t log π θ (y t l |x,y <t l ) πref(y t l |x,y <t l )
.
8:
Generate the anchor response y t : y t ∼ π θ (y t-1 |x, y <t-1 ).
9:
Compute triplet log-ratio ∆r triplet (Eq.( 11)).
10:
Compute weighted DPO loss: L DPO-w ← -log σ(β∆r token ).
11:
Compute triplet loss: L triplet ← max (0, ∆r triplet + α).
12:
Aggregate losses: L TI-DPO ← L DPO-w + γL triplet .
13:
Update θ ← θ -η∇ θ L TI-DPO . 14: end for 15: Output: π θ To show the superiority of TI-DPO, we first introduce the following lemma.
Lemma 1 (Variance Reduction). Consider a reward signal governed by a sparse set of critical tokens, such that the subset of non-critical tokens N contributes only independent zero-mean noise ϵ t with variance σ 2 ϵ . Provided that the importance weights for these non-critical tokens are suppressed such that w 2 t < 1 for all t ∈ N , the variance of the TI-DPO estimator (σ 2 TI-DPO ) is strictly lower than that of the standard DPO estimator (σ 2 DPO ), i.e.,
σ 2 TI-DPO < σ 2 DPO (15
)
The proof is detailed in Appendix A.1. Lemma 1 establishes that by suppressing the weights of noncritical tokens, TI-DPO effectively filters out stochastic noise, resulting in a strictly lower variance for the reward difference estimator. The following theorem strictly proves the theoretical advantages of this improvement at the loss function level. With Lemma 1, the total loss of TI-DPO will be significantly lower than the original DPO loss. Denote ∆r global = log π θ (yw|x) πref(yw|x) -log π θ (yl|x) πref(yl|x) from Eq.( 3). For simplicity, we abbreviate ∆r token (x, y w , y l , w w t , w l t ) as ∆r token , then we have: Theorem 2 (Tighter Loss Bound). Assuming the preference optimization objective is a strictly convex function and the reward difference estimation is unbiased, the expected loss of TI-DPO (L TI-DPO ) is strictly upper-bounded by that of DPO (L DPO ) minus a term proportional to the variance reduction. Specifically:
L TI-DPO ≤ L DPO - 1 2 κ∆ σ 2 ,(16)
where κ > 0 represents the lower bound of the loss function's local curvature, and
∆ σ 2 = σ 2 DPO - σ 2
TI-DPO is the positive variance reduction term derived in Lemma 1.
The proof is shown in Appendix A.2. In the experiments presented in the Appendix B.4, we compared the loss function convergence processes of the TI-DPO and DPO, thereby further substantiating Theorem 2. Furthermore, we provide a theoretical justification for the superiority of the policy learned by TI-DPO. In Theorem 3, we demonstrate that TI-DPO utilizes the limited KL constraint more efficiently by concentrating probability mass on critical tokens. The proof is shown in Appendix A.3.
Theorem 3 (Superiority of Optimal Policy). Let π DP O and π T I-DP O be the optimal policies derived from minimizing the DPO and TI-DPO objectives, respectively. Under a fixed total KL divergence constraint K total , the expected true reward of the TI-DPO optimal policy is strictly lowerbounded by that of the DPO policy, i.e.,
E y∼πTI-DPO [r * (x, y)] ≥ E y∼πDPO [r * (x, y)] + δ,(17)
where δ > 0 represents the gain derived from optimizing the decomposition of KL divergence, specifically by minimizing the divergence component on non-critical tokens.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETTINGS
Datasets and base settings. The benchmarks we use include knowledge-based tasks (MMLU (Hendrycks et al., 2020)), mathematical reasoning (GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021)), instruction-following (IFEval (Zhou et al., 2023)), and code generation (HumanEval (Chen et al., 2021)). Additionally, TruthfulQA (Lin et al., 2021) detects the authenticity of the model's answers through adversarial questions. For other detailed hyperparameter settings, please refer to Appendix B.5.
Comparative algorithm. We compared the TI-DPO with baseline alignment methods such as SFT, DPO, IPO(Azar et al., 2024), KTO (Ethayarajh et al., 2024), SimPO (Meng et al., 2024), TDPO (Zeng et al., 2024), CPO (Feng et al., 2025), TPO (Saeidi et al., 2024), TIS-DPO (Liu et al., 2024a), Logic-RL (Xie et al., 2025), cDPO (Lin et al., 2024) and GRPO (Shao et al., 2024). We select three models (Llama-3.2-3B (Grattafiori et al., 2024), Llama-3.1-8B (Grattafiori et al., 2024), Mistral-7B-v0.3 (Jiang et al., 2023)) as baselines.
this section cite: ['b14', 'b5', 'b15', 'b46', 'b4', 'b23', 'b7', 'b28', 'b42', 'b8', 'b33', 'b39', 'b24', 'b35', 'b12', 'b12', 'b18']

Section: PERFORMANCE COMPARISON
As shown in Figure 1, we conduct an analysis of the performance for TI-DPO and baseline methods across training steps on the TruthfulQA (reliability assessment) and IFEval (instruction-following) tasks with Llama-3.1-8B model. In the TruthfulQA benchmark (Figure 1a), TI-DPO demonstrates a steady improvement in accuracy as training steps increase, surpassing all baselines by the final epoch. For the IFEval task (Figure 1b), TI-DPO also shows a dominant performance trend. This highlights TI-DPO's effectiveness in learning through token-level importance weighting and triplet loss, which explicitly guides the model to avoid generating misleading content. As shown in Figure 2, TI-DPO exhibits significantly better performance in Reasoning, Instruction-Following, and Reliability dimensions compared to the corresponding instruction variants with each base instruction. Here, the base instruct refers to the foundational instruction-tuned models (Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3), serving as the baseline for comparing the effectiveness of TI-DPO and other fine-tuning methods. The scores of TI-DPO in other aspects are roughly equal or slightly higher than others. Table 1 presents average scores of each finetuning method across three base models, clearly demonstrating our method's advantages in general tasks and specific scenarios. The specific score comparison table under the three base models is placed in Appendix C.
General Math Reasoning Code Instr-Follow Reliability 0.2 0.4 0.6 0.8 1.0 Llama-3.2-3B-Instruct Base Instruct TI-DPO General Math Reasoning Code Instr-Follow Reliability 0.2 0.4 0.6 0.8 1.0 Llama-3.1-8B-Instruct Base Instruct TI-DPO General Math Reasoning Code Instr-Follow Reliability 0.2 0.4 0.6 0.8 1.0 Mistral-7B-Instruct-v0.3 Base Instruct TI-DPO Multidimensional Normalized Score Radar Chart  Similarly, ablating the Gaussian Prior leads to a notable decline in Reliability (86.8 → 82.5).
this section cite: []

Section: ADDITIONAL EXPERIMENTS
Case Study. A case study in Figure 3 on a medical query (see Appendix C.1 for details) demonstrates that, given the user prompt "I have a headache, what should I do?", TI-DPO effectively assigns higher importance to safety-critical tokens (e.g., "medical attention", "promptly") in preferred responses, while penalizing risky suggestions (e.g., "painkillers", "casually") in non-preferred ones.
Additionally, there are two other cases in Appendix C.
2. A (Preferred): "Based on your symptoms, it is recommended that you seek medical attention promptly and avoid selfmedicating." B (Intermedia): "According to your description, it is advised to get more rest, but if the symptoms worsen, you should consult a doctor." C (Non-preferred): "Don't worry, you can just take some painkillers casually, it should be fine."
this section cite: []

Section: Distribution of Weights.
Figure 4 illustrates how TI-DPO dynamically adapts token importance weights based on task characteristics. For tasks relying on a few critical symbols (GSM8K, GPQA), weights concentrate in the [0.2, 0.5] range. Conversely, in tasks demanding strict adherence to safety or instructions (TruthfulQA, IFEval), weights shift to a higher [0.6, 0.8] interval. Meanwhile, for comprehensive tasks covering broader content (MMLU, HumanEval), the weight distribution is naturally more dispersed. Pearson correlation coefficient. To investigate the effectiveness of our hybrid weighting mechanism, we also conducted Pearson correlation coefficient analysis, with full results and methodology presented in Appendix B.1. Our analysis reveals a moderately strong positive correlation (r ≈ 0.65) at the task level between average token importance and performance improvements, indicating that TI-DPO is particularly effective on tasks with concentrated weight distributions.
Robustness and Generation Diversity. We validate the robustness and generative diversity of TI-DPO, which can be seen in Appendix B.3. To evaluate the model's stability, we tested accuracy under varying label noise levels. TI-DPO maintains superior performance compared to DPO and TPO as noise increases.
this section cite: []

Section: Sensitivity of Hyperparameters.
We conducted sensitivity analyses for the weight-mixing parameter λ and KL weight α in Appendix B.5. Performance remains stable for λ ∈ [0.3, 0.7]. We have provided the specific values of the hyperparameters for this project, as shown in Appendix B.5.
this section cite: []

Section: CONCLUSION
We introduce TI-DPO, an optimization framework that effectively bridges the alignment gap between LLMs and human value systems. By introducing a mixed weight calculated collaboratively by gradient attribution and Gaussian prior, TI-DPO effectively overcomes the limitations of traditional DPO methods at the token level and their sensitivity to noise. On this basis, the triplet loss structure provides more refined guidance for model optimization. Theorem 2 and Theorem 3 theoretically illustrate the superiority of TI-DPO over DPO. The effectiveness of TI-DPO is unequivocally demonstrated through extensive experimentation. Our method achieves a state-of-the-art average score of 62.3 across a diverse suite of benchmarks, outperforming all baseline methods. As for the limitations, despite its effectiveness in fine-grained alignment, TI-DPO entails a computational overhead during training and performs slightly below sequence-level baselines on holistic reasoning tasks. Future work will focus on integrating our token-importance mechanism with group-based optimization methods like GRPO to bridge this gap and further enhance reasoning capabilities. More statements can be found in Appendix D.
this section cite: []

Section: References
Ref_id:b0 Title: Towards better understanding of gradient-based attribution methods for deep neural networks Year: (2017)
Ref_id:b1 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: Efficient knowledge distillation: Empowering small language models with teacher model insights Year: (2024)
Ref_id:b4 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b5 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b6 Title: Process reinforcement through implicit rewards Year: (2025)
Ref_id:b7 Title: Model alignment as prospect theoretic optimization Year: (2024)
Ref_id:b8 Title: Sequencelevel large language model training with contrastive preference optimization Year: (2025)
Ref_id:b9 Title: Llm-based nlg evaluation: Current status and challenges Year: (2025)
Ref_id:b10 Title: Rebel: Reinforcement learning via regressing relative rewards Year: (2024)
Ref_id:b11 Title: Mixed preference optimization: Reinforcement learning with data selection and better reference model Year: (2024)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Gaussian transformer: a lightweight approach for natural language inference Year: (2019)
Ref_id:b14 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b15 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b16 Title: Monolithic preference optimization without reference model Year: (2024)
Ref_id:b17 Title: Reinforce++: An efficient rlhf algorithm with robustness to both prompt and reward models Year: (2025)
Ref_id:b18 Title: Devendra Singh Chaplot Year: (2023)
Ref_id:b19 Title: Preference optimization for reasoning with pseudo feedback Year: (2024)
Ref_id:b20 Title: Rs-dpo: A hybrid rejection sampling and direct preference optimization method for alignment of large language models Year: (2024)
Ref_id:b21 Title: Spread preference annotation: Direct preference judgment for efficient llm alignment Year: (2025)
Ref_id:b22 Title: Xiang Ao, and Qing He. Gradient-adaptive policy optimization: Towards multi-objective alignment of large language models Year: (2025)
Ref_id:b23 Title: Measuring how models mimic human falsehoods Year: (2021)
Ref_id:b24 Title: Critical tokens matter: Token-level contrastive estimation enhence llm's reasoning capability Year: (2024)
Ref_id:b25 Title: Tis-dpo: Token-level importance sampling for direct preference optimization with estimated weights Year: (2024)
Ref_id:b26 Title: Lost in the middle: How language models use long contexts Year: (2024)
Ref_id:b27 Title: Aligning large language models with human preferences through representation engineering Year: (2023)
Ref_id:b28 Title: Simpo: Simple preference optimization with a reference-free reward Year: (2024)
Ref_id:b29 Title: Distance metric learning for ordinal classification based on triplet constraints. Knowledge-Based Systems Year: (2018)
Ref_id:b30 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b31 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b32 Title: From r to q * : Your language model is secretly a q-function Year: (2024)
Ref_id:b33 Title: Triple preference optimization: Achieving better alignment with less data in a single step optimization Year: (2024)
Ref_id:b34 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b35 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b36 Title: Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints Year: (2023)
Ref_id:b37 Title: Aligning large language models with human: A survey Year: (2023)
Ref_id:b38 Title: Training large language models for reasoning through reverse curriculum reinforcement learning Year: (2024)
Ref_id:b39 Title: Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning Year: (2025)
Ref_id:b40 Title: Monte carlo tree search boosts reasoning via iterative preference learning Year: (2024)
Ref_id:b41 Title: Distinguishing llm-generated from human-written code by contrastive learning Year: (2025)
Ref_id:b42 Title: Token-level direct preference optimization Year: (2024)
Ref_id:b43 Title: Attention with sparsity regularization for neural machine translation and summarization Year: (2018)
Ref_id:b44 Title: Risk-aware direct preference optimization under nested risk measure Year: (2025)
Ref_id:b45 Title: Dpo meets ppo: Reinforced token optimization for rlhf Year: (2024)
Ref_id:b46 Title: Instruction-following evaluation for large language models Year: (2023)
