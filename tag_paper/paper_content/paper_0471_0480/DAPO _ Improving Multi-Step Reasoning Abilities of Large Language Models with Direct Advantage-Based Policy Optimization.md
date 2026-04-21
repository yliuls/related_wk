Title: DAPO : Improving Multi-Step Reasoning Abilities of Large Language Models with Direct Advantage-Based Policy Optimization
Abstract: The role of reinforcement learning (RL) in enhancing the reasoning of large language models (LLMs) is becoming increasingly significant. Despite the success of RL in many scenarios, there are still many challenges in improving the reasoning of LLMs. One key challenge is the sparse reward, which introduces more training variance in policy optimization and makes it difficult to obtain a good estimation for value function in Actor-Critic (AC) methods. To address these issues, we introduce Direct Advantage-Based Policy Optimization (DAPO), a novel step-level offline RL algorithm with theoretical guarantees for enhancing the reasoning abilities of LLMs. Unlike response-level methods (such as DPO and GRPO) that the update directions of all reasoning steps are governed by the outcome reward uniformly, DAPO employs a critic function to provide step-level dense signals for policy optimization. Additionally, the actor and critic in DAPO are trained independently, ensuring that critic is a good estimation of true state value function and avoiding the co-training instability observed in standard AC methods. We train DAPO on mathematical and code problems and then evaluate its performance on multiple benchmarks. Our results show that DAPO can effectively enhance the mathematical and code capabilities on both SFT models and RL models, demonstrating the effectiveness of DAPO.

Section: Introduction
In the rapidly evolving landscape of artificial intelligence, large language models (LLMs) have emerged as a cornerstone of natural language processing (NLP) and beyond. These models, trained on vast corpora of text data, have demonstrated an unprecedented ability to understand [16,3], generate, and reasoning such as solving mathematical problems [61,48,63,53] and code generations [23,8]. When the token generation process of a LLM is modeled as a Markov Decision Process (MDP), it can be naturally optimized and aligned with human preference using reinforcement learning (RL) methods, known as Reinforcement Learning from Human Feedback (RLHF). Despite the success of RLHF in various fields [11,42,43,55,46,6], it still encounters challenges and difficulties in the field of reasoning especially in long Chain-of-Thought (CoT) models. One of the key challenges is the sparsity of rewards [44,57]. When using LLMs for mathematical problem-solving and code generation, rewards are only assigned to the terminal tokens. This implies that the intermediate tokens receives no direct reward, and the optimization direction relies solely on the backpropagation of the reward from the terminal token. Consequently, there are two issues in practical RL training: Figure 1: Direct Advantage-Based Policy Optimization (DAPO). The whole training procedure of DAPO consists of two individual stages : 1) Critic Training (left). Given the training query set Q, for each query q ∈ Q, DAPO uses the generator π gen to generate multiple rollouts from q and derive a dataset of training states denoted as D gen . For each state in s ∈ D gen , DAPO uses the completer (often π ref ) to sample multiple sub-trajectory from s and collects the MC value as the estimation of the true value. Then a critic network V ϕ is trained to approximate the value function of the completer. 2) Policy Optimization (right). After extracting multiple next steps {a i } n i=1 of each state s from the completions in step A.1, DAPO then uses the trained critic to compute the advantages for all actions by ∀i ∈ [n] : A i = Q i -1 n n j=1 Q j and Q i is the predicted value of Concat(s, a i ). Finally, DAPO fits the policy ratio to the advantage to optimize the generation of reasoning steps.
this section cite: ['b15', 'b2', 'b60', 'b47', 'b62', 'b52', 'b22', 'b7', 'b10', 'b41', 'b42', 'b54', 'b45', 'b5', 'b43', 'b56']

Section: High training variance induced by response-level algorithms.
Response-level algorithms (such as DPO [42], GRPO [48], ReMax [30] and etc) treat the complete response as a unified whole and the gradient directions across all reasoning steps within the response are governed by the final reward uniformly. Although response-level algorithms have recently achieved great success, such as Deepseek-R1 [10], the sparsity of the reward indeed introduces more training variance and addressing the training variance introduced by it would lead to a larger performance improvement. See Section D for detailed discussions.
Bad value estimation in actor-critic methods. When applying standard actor-critic methods [46,15,12] to LLMs, due to the sparsity of the reward and vast generation policy, the critic often fails not only to learn a good semantic representation but also to provide the fine-grained credit assignment for policy optimization, leading to the poor performance of online actor-critic methods [25,62] such as PPO [46]. Besides, these methods train the actor and critic simultaneously and the regression targets of both actor and critic are non-stationary and change in response to the updates of the other during the training process. This interwoven update mechanism also leads to an unstable training process that is prone to collapse [27,12] especially when the critic is not a good approximation of the true value function.
In order to overcome these challenges, we introduce Direct Advantage-Based Policy Optimization (DAPO), a new step-level offline actor-critic method designed to enhance LLM's performance in reasoning tasks. In order to reduce the training variance induced by the sparsity of the reward, we use a critic function to provide dense training signals. To ensure that the critic function is well-trained, DAPO trains the critic individually before policy optimization to ensure it is a good approximation of the true state value function. The procedure of DAPO is visualized in Figure 1. We summarize our contributions as follows:
• We introduce DAPO, a new step-level, offline RL method that learns from all generated samples. DAPO optimizes the generation of the reasoning steps by sampling multiple candidate steps for each intermediate step s and updating the policy according to the advantages of each candidate step. Our main theoretical result (see Theorem 3.2) shows that DAPO will produce a better policy until the reference policy is already optimal.
• We conduct extensive experiments to demonstrate the effectiveness of DAPO. DAPO improves the performance on multiple mathematical and coding benchmarks. Our results empirically shows that DAPO consistently outperforms both the response-level baseline GRPO and the actor-critic baseline PPO on multiple base models .
this section cite: ['b41', 'b47', 'b29', 'b9', 'b45', 'b14', 'b11', 'b24', 'b61', 'b45', 'b26', 'b11']

Section: Preliminaries
In this section, we introduce the mathematical formulation of RL problem we studied and some important related theoretical results.
this section cite: []

Section: RL reasoning.
The objective for RL reasoning can be formulated as
max π E x∼µ E y∼π(•|x) [r (x, y)] -βKL π x , π ref x , (1
)
where µ is the distribution of training prompt x, y is the response sampled from policy π(•|x), r(x, y) ∈ {0, 1} is the binary reward for the correctness of the response, π x , π ref x are short for π(•|x), π ref (•|x) respectively, π ref is the reference policy, β ≥ 0 is the coefficient of KL regularization.
this section cite: []

Section: Equivalency to multi-step decision problem.
We use the line break \n as the delimiter for reasoning steps. By doing so, a response y can be divided into multiple reasoning steps, i.e. y = (a 0 , ..., a T -1 ), where ∀t ≥ 0 a t is the reasoning step and T is the total number of steps. Thus one can easily show that the objective 1 is equivalent to following step-level decision problem:
max π E x∼µ E ∀t:at∼π(•|st) T -1 t=0 r (s t , a t ) -βKL π st , π ref st |s 0 = x ,(2)
where s t := Concat(x, a 0 , ..., a t-1 ) is the state (context prefix), the reasoning step a t is sampled from policy π(•|s t ) (here we use π to denote both step generation and response generation policy with a slightly abuse of the notation), π st , π ref st are short for π(•|s t ), π ref (•|s t ) respectively and the reward function r satisfies r (s t , a t ) = r (x, y) t = T -1 0 t < T -1. For any state s ∈ S, we define as
V π β (s) := E ∀t:at∼π(•|st) T -1 t=0 r (s t , a t ) -βKL π st , π ref st |s 0 = s .(3)
Thus the objective (2) is equivalent to finding a policy π that maximizes the state value function, i.e.,
max π E x∼µ V π β (x) .(4)
We denote π * β as the optimal policy of objective (4) and V * β := V π * β β as the state value function of optimal policy. We define the state space S as the set of all possible states that can be encountered in the response y and the action space A as the set of all possible singular reasoning steps. For simplicity of analysis, we introduce the following assumptions in this work. Assumption 2.1. We assume there exists a constant H 1 < ∞ such that for any policy π and initial state s 0 ∈ S, the random decision horizon T satisfies T < H 1 almost surely. We also assume that there exists a constant H 2 < ∞ such that for any singular reasoning step a ∈ A, the token length of a is no greater than H 2 so that the state space S and action space A are both finite.
this section cite: []

Section: More value functions.
We define the KL-constrained state-action value function as
Q π β (s, a) := r (s, a) + V π β (s • a) ,(5)
where s • a := Concat(s, a). The advantage function A π β is defined as
A π β (s, a) := Q π β (s, a) -V π β (s) -β log π (a|s) π ref (a|s) .(6)
We denote Q * β (s, a) := r (s, a) + V * β (s). For unregularized state value functions, i.e. β = 0, we use V π , Q π , A π instead of V π 0 , Q π 0 , A π 0 to be aligned with the standard RL literatures.
this section cite: []

Section: Bellman operators
For arbitrary policy π ∈ Π and any function V : S → R, the Bellman operator T π β is defined as
∀s ∈ S :[T π β V ] (s) = E a∼π(•|s) r (s, a) + V (f (s, a)) -β log π (a|s) π ref (a|s) .(7)
Performance difference lemma. Following is the key theoretical analysis tool used in this work.
Lemma 2.2. For arbitrary two policies π, π ∈ Π and any distribution ρ ∈ ∆(S),
V π β (ρ) -V π β (ρ) = E s∼d π ρ T π β V π β (s) -V π β (s) ,(8)
where V π β (ρ) := E s∼ρ V π β (s) and the visitation measure d π ρ is defined as,
d π ρ (s) := E ∀t:at∼π(•|st) T -1 t=0 1 {s t = s}|s 0 ∼ ρ .
The proof of this lemma is deferred to the Appendix C. We also put some additional theoretical results about MDP on Appendix B. Remark 2.1. Note that our definition of visitation measure, d π ρ is not standard as the one in [1,33]. The reason is that the planing horizon in problem (2), i.e. T , is random while it is the constant 1 1-γ in [1,33]. Thus we do not try to normalize d π ρ to a probability distribution by dividing T directly.
this section cite: ['b0', 'b32', 'b0', 'b32']

Section: Direct Advantage-Based Policy Optimization
In this section, we present the details of DAPO. We only consider the case where a binary reward r(s, a) ∈ {0, 1} is assigned in the terminal state since it is the most common situation in the LLMs setting. We first present the objective of policy optimization with related theoretical results in Section 3.1 assuming that one has access to the true value function.
this section cite: []

Section: Policy objective
We first derive the policy optimization objective in one single iteration provided with the reference policy π ref .
In the perspective of policy optimization, the most important quantity may be the advantage function. Recall Lemma 2.2. The performance difference for any policy π ∈ Π w.r.t π ref on arbitrary ρ ∈ ∆(S) is measured as
V π β (ρ) -V π ref β (ρ) := E s∼d π ρ   T π β V π ref β (s) -V π ref β (s
) :=Is(π,π ref )    , where I s (π, π ref ) can be regarded as the policy improvement at state s. By the definition of Bellman operator T π β in (7) and that of function V π β in (3), one can easily show that I s (π, π ref ) = E a∼π(•|s) A π ref (s, a) -β log π (a|s) π ref (a|s) .
A natural idea that ensures V π β (ρ) -V π ref β (ρ) ≥ 0 might be to increase I s (π, π ref ) on as many states s as possible. For ease of understanding, we first focus on how to increase I s (π, π ref ) at an arbitrary state s ∈ S. Considering parameterized policy π θ , the most directly and easily method may be to do multi gradient ascent steps w.r.t I s (π θ , π ref ), i.e.,
∀K ∈ N + , θ K = θ 0 + η K-1 k=0 ∇ θ I θ (π θ k , π ref ), (9
)
where η is the step size. Following lemma gives an equivalent form of ∇ θ I s (π θ k , π ref ).
Lemma 3.1. If we define the surrogate function H k s (θ) := 1 2 E a∼π θ k (•|s) 1 β A π ref (s, a) -log π θ (a|s) π ref (a|s) 2 , then ∇ θ I s (π θ k , π ref ) = -β∇ θ H k s (θ) . By Lemma 3.1, the update (9) is equivalent to
θ K = θ 0 -ηβ K-1 k=0 ∇ θ H k s (θ k ).(10)
We now to turn to the offline policy optimization problem setting for the sake of implementation simplicity and data efficiency. A natural question is that can we implement (10) approximately in an offline dataset without sampling the on-policy rollouts at each iteration? Our answer is affirmative. We find that (which will be proved latter), just solving the problem
min θ 1 2 E a∼ν(•|s) 1 β A π ref (s, a) -log π θ (a|s) π ref (a|s) 2 ,(11)
where ν ∈ ∆(A) can be any exploratory sampling distribution, can still yield a policy that
I s (π θ , π ref ) > 0 unless π ref = π * β .
Taking the consideration of optimizing multiple states simultaneously (weighted by a state sampling distribution ν S ), the objective of DAPO is
min θ L (θ) := 1 2 E s∼ν S ,a∼ν A (•|s) 1 β A π ref (s, a) -log π θ (a|s) π ref (a|s) 2 (12
)
Remark 3.1. We give a detailed discussion in Section D on how DAPO is specifically designed to improve multi-step reasoning abilities of LLMs, thereby distinguishing it from response-level optimization algorithms such as DPO [42], GRPO [48] and etc.
Notice that the state space S and the action space A are both finite under the assumption 2.1. Following theorem establish the validity of Loss (12) and shows that the solution of (12) is a better policy w.r.t π ref on objective (4). Theorem 3.2 (Monotonic improvement). Suppose ∀s ∈ S, a ∈ A, ν S (s) > 0, ν A (a|s) > 0, π ref (a|s) > 0. Let π + be the solution in (12). Then for any state s ∈ S, there exists a function λ s : ∆ (A) → [0, +∞) such that
V π + β (µ) -V π ref β (µ) ≥ E s∼d π + D [λ s (ν A )] ≥ 0.
The equality holds if and only if π ref = π * β . Theorem 3.2 implies that DAPO can improve the performance until there is no room for improvement in the trust region of π ref . Theorem 3.2 also establishes the monotonic improvement property on unregularized state value function. This can be verified by
V π + (µ) = V π + β (µ) + E s∼d π + µ KL π + (•|s) , π ref (•|s) ≥ V π + β (µ) ≥ V π ref β (µ) = V π ref (µ) .
this section cite: ['b9', 'b41', 'b47', 'b11', 'b11']

Section: Implementation of DAPO.
In practice, DAPO can be implemented by minimizing (12) on an offline training dataset D = s i , a i , Âi
this section cite: []

Section: N i=1
, where s i ∼ ν S , a i ∼ ν A (•|s) and Âi is an estimation of true advantage A π ref (s i , a i ). The whole procedure can be found in Algorithm 1.
this section cite: []

Section: Advantage estimation.
Recall the definition of Q function in (5) and advantage function in (6). For non-terminal states, one has r(s, a) = 0 and
A π ref (s, a) = Q π ref (s, a) -V π ref (s) = V π ref (s • a) -E a ′ ∼π ref (•|s) [V π ref (s • a ′ )] .
Thus in our implementation, we sample multiple actions {a i } M i=1 for each state s from π ref (•|s) and estimate the advantage of each action by ∀i ∈ [m] : Â (s, a i ) := V ϕ (s
• a i ) - 1 M M j=1 V ϕ (s • a j ) .(13)
Here V ϕ ≈ V πref is a pretrained critic function. Since (13) is fully determined by the critic V ϕ , thus the training of V ϕ plays a crucial role for the optimization of training states. In the following, we introduce the optimization method of V ϕ .
this section cite: ['b5']

Section: Critic objective
Generally speaking, the critic optimization problem for any target policy π can be formulated as
min ϕ E s∼D [L (V ϕ (s) , V π (s))] ,(14)
here L is a sample-wise loss function, D is the distribution of training states. However, since the target V πref (s) can not be accessed to in advance, we need to construct the estimate of V π as the training samples for the optimization of V ϕ . One can show that
V π (s) = E T -1 t=0 r (s t , a t ) |s 0 = s = P π (r (s T -1 , a T -1 ) = 1|s 0 = s) .
Thus, one can firstly use a behavior policy π gen (also called generator) to generate a set of training states s. Then for each training state s, one can use π as the completer to sample N sub-trajectories from s, τ i = s, a
(i) 0 , a (i) 1 , ..., a (i) Ti-1 N i=1
, and construct the empirical MC mean,
MC N (s) := 1 N N i=1
r (s Ti-1 , a Ti-1 ) , as the estimation of V π (s), which is guaranteed to converge to V π (s) as n → ∞ by the strong law of large numbers. Notice that the state value V π (s) is a probability. Thus we the binary cross-entropy loss in our implementation, i.e.,
L BCE (y, y pred ) = -(y log(y pred ) + (1 -y) log(1 -y pred ))
as the sample-wise loss function L in (14) to avoid gradient diminishing if a MSE loss is applied (which is more common in RL literatures). Finally, the critic optimization objective is
min ϕ E s∼D [L BCE (MC N (s) , V ϕ (s))] .(15)
Remark 3.3 (Should we treat V ϕ as a PRM?). Its worthy to note that the critic training method we presented above is mostly aligned with [56]. However, we argue that the V ϕ should be treated as a value function other than a process reward model. Here are the reasons : 1) From the optimization perspective, V ϕ is trained to approximate the true state value of the completer. 2) From the RL perspective, using V ϕ as a reward function will deviate from the original RLHF optimization objective and easily lead to a serve reward hacking issue on the number of reasoning steps as shown in [13].
this section cite: ['b55', 'b12']

Section: Experiments
In this section, we conduct DAPO on mathematical and coding datasets individually. The experimental setup is detailed in Section 4.
1, and specific conclusions and benchmark results of our experiments is presented in Section 4.2. Model In Domain Out of Domain MATH GSM8K Minerva MATH Olympiad Bench College Math SFT Models Skywork-Math-Llama 41.90 61.49 5.51 18.68 24.85 + DAPO 46.88 +4.98 67.55 +6.06 7.34 +1.82 22.38 +3.70 25.87 +1.03 Llama-3.1-8B-Instruct 49.42 85.29 26.48 16.14 30.91 + DAPO 53.62 +4.20 86.74 +1.45 23.54 -2.94 20.01 +3.88 30.91 +1.63 OpenO1-Llama-8B-v0.1 52.73 85.99 29.04 19.86 29.10 + DAPO 60.33 +7.60 88.77 +2.78 29.42 +0.39 24.44 +4.59 32.12 +3.02 Qwen2.5-72B-instruct 82.90 95.40 46.30 45.45 43.00 + DAPO 84.70 +1.80 95.70 +0.30 50.00 +3.70 47.70 +2.25 43.40 +0.40 RL Models Qwen2-Math-7B-Instruct 74.46 89.38 40.07 34.36 41.59 + DAPO 75.41 +0.95 89.45 +0.07 37.51 -2.56 37.03 +2.67 42.23 +0.64 Skywork-O1-Open-Llama3.1-8B 78.10 91.64 26.10 43.11 40.40 + DAPO 79.81 +1.71 92.81 +1.17 29.77 +3.67 44.76 +1.65 40.26 -0.14 Qwen2.5-Math-7B-Instruct 83.42 95.78 40.06 38.96 42.65 + DAPO 84.86 +1.44 96.14 +0.36 41.56 +1.50 41.25 +2.29 41.97 -0.67 Table 1: Performance of models optimized via DAPO on mathematical benchmarks. We use zero-shot prompting and greedy decoding for all evaluations in the table. Across a wide range of model families and out-of-domain math word problem benchmarks, DAPO achieves steady improvement with only 7.5K training samples from MATH [18]. SFT Model Out of Domain HumanEval HumanEval+ MBPP MBPP+ LiveCodeBench Llama-3.1-8B-Instruct 72.0 66.5 72.0 56.9 18.8 + DAPO 75.0 +3.0 68.9 +2.4 77.0 +5.0 66.1 +9.2 20.9 +2.1 OpenO1-Llama-8B-v0.1 69.5 61.0 69.8 58.7 16.1 + DAPO 72.0 +2.5 64.6 +3.6 75.9 +6.1 63.8 +5.1 13.7 -2.4
Table 2: Performance of models optimized via DAPO on code benchmarks. We use zero-shot prompting and greedy decoding for all evaluations in the table. DAPO consistently improves performance when starting with both regular and reasoning LLMs, even when only 4K training samples from TACO [29] is used.
this section cite: ['b28']

Section: Experimental setup
Training dataset. For all the DAPO experiments on mathematics, we utilize only the 7500 training problems from the dataset MATH [18] to generate the advantage datasets for DAPO training, requiring no additional human annotations. In particular, we only use the questions and the corresponding golden answers in the dataset while the provided solutions are not used for training. For coding experiments, For coding experiments, we subsample the TACO [29] dataset to compile approximately 4,000 competition-level programming questions derived from real-world scenarios. We utilize its provided unit test cases to evaluate whether a solution is accurate.
Base models. Considering the reproducibility and of DAPO, our experiments are taken over of several open-source language models including general and math-specific models. For general models, we consider Llama-3.1-8B-Instruct [52] , OpenO1-Llama-8B-v0.1 [39], Skywork-O1-Open-Llama3.1-8B [40], Qwen2.5-72B-Instruct [51]. For math-specific models, we consider Skywork-Math-Llama [63], Qwen2-Math-7B-Instruct [59] and Qwen2.5-Math-7B-Instruct [60]. Notice that Skywork-O1-Open-Llama3.1-8B, Qwen2-Math-7B-Instruct and Qwen2.5-Math-7B-Instruct are already trained by RL algorithms [48]. Thus we conduct continue-RL training on three models by DAPO.
this section cite: ['b17', 'b28', 'b51', 'b38', 'b39', 'b50', 'b62', 'b58', 'b59', 'b47']

Section: Benchmarks & Metrics.
For mathematical experiments, we evaluate performance on English mathematical benchmarks. In addition to the 5000 test problems from MATH [18], We also add 4 out-of-domain benchmarks, GSM8K [9], Minerva Math [28], Olympiad Bench [17] and College Math [50], to test the performance generalization of DAPO. For coding experiments, we evaluate DAPO on several widely-used benchmarks in code generation, i.e., HumanEval [7], HumanEval+ [22], MBPP [4], MBPP+ [5] and LiveCodeBench [21].
All the evaluations are conducted in a zero-shot greedy Model Base +GRPO +PPO +DAPO Llama-3.1-8B-Instruct 49.42 +2.99 52.28 +2.86 52.41 +2.99 53.62 +4.20 OpenO1-Llama-8B-v0.1 52.73 +2.99 55.63 +2.90 54.12 +1.39 60.33 +7.60 Qwen2-Math-7B-Instruct 74.46 +2.99 74.94 +0.46 74.93 +0.47 75.41 +0.95 Qwen2.5-Math-7B-Instruct 83.42 +2.99 84.33 +0.91 83.76 +0.34 84.86 +1.44
Table 3: Comparison with baseline methods on the 5K test samples from MATH [18]. DAPO consistently outperforms both the initial model and model optimized with GRPO and PPO. sampling (i.e. temperature 0) with a cot prompt template and a maximum amount of 2048 newly generated tokens (except 4096 for OpenO1-Llama-8B-v0.1 and Skywork-O1-Open-Llama3.1-8B).
this section cite: ['b17', 'b8', 'b27', 'b16', 'b49', 'b6', 'b21', 'b3', 'b4', 'b20', 'b17']

Section: DAPO implementations.
For each DAPO experiment, we first use the base model to generate 32 solutions for each training problem and then 6 solutions are selected out of them while making the correct solutions and wrong solutions as balanced as possible. For each selected solution, we use one line break \n to segment the solution steps and use the base model as the completer to do 16 completions for each reasoning step. Then the MC estimations are constructed for the critic training. The training methodology of critic is presented in Section 3.2. In our implementation, the critics are fine-tuned for one epoch on Qwen2.5-Math-7B-Instruct for mathematical experiments and Qwen2.5-coder-7B-Instruct for coding experiments, with a learning rate of 5e-6 and batch size 512. After that, we extract the next steps (action) for each intermediate step (state) in the completion datasets and compute the advantage using the critic for DAPO training. For all experiments, we use the global batch size 2048, learning rate 5e-7 and β = 0.01 or 0.02. We also summarize some useful tricks applied in all of the DAPO training in Appendix F.1
Baseline methods. We compare the performance of DAPO with PPO as the actor-critic method baseline and GRPO as the response-level method baseline on mathematical tasks using the 7.5K problems from MATH [18] as the training queries. We conduct experiments on four base models : Llama-3.1-8B-Instruct, OpenO1-Llama-8B-v0.1,Qwen2-Math-7B-Instruct and Qwen2.5-Math-7B-Instruct. We train PPO and GRPO until the training accuracy curve converges. The more training details of our baseline methods are in Appendix F.2.
this section cite: ['b17']

Section: Main results

this section cite: []

Section: Math.
Our results, presented in Table 1, Table 3 and Table 4.2, demonstrate that DAPO consistently enhances the performance of the base model across all tested models on the in-domain benchmark MATH [18], through DAPO training, Skywork-Math-Llama, Llama-3.1-8B-Instruct, OpenO1-Llama-8B-v0.1, and Qwen2.5-72B-instruc achieve 50.54%, 53.62%, 60.27%, 84.55% greedy decoding accuracy respectively. For RL models, Qwen2-Math-7B-Instruct,Skywork-O1-Open-Llama3.1-8B and Qwen2.5-Math-7B-Instruct achieve 76.40%, 79.81%,and 84.86% greedy decoding accuracy respectively, indicating that DAPO can further enhance the performance even the base model are already trained by RL methods previously. Table 3 shows that DAPO has a greater improvement than PPO and GRPO on various base models in our experiments. In Figure 3, we present the accuracy curve on the MATH TEST during the training process of DAPO. As can be clearly observed, DAPO steadily enhances the accuracy throughout the training until it stabilizes.
this section cite: ['b17']

Section: Code generation.
Our results for the code generation task are shown in Table 2. DAPO improves the coding performance of both Llama-3.1-8B-Instruct and OpenO1-Llama-8B-v0.1 on multiple wildly used benchmarks. For Llama-3.1-8B-Instruct, DAPO has increased by 3.0% and 2.4% respectively on HumanEval [7] and HumanEval+ [22], by 5.0% and 9.2% respectively on MBPP [4] and MBPP+ [5], and by 2.1% on LiveCodeBench [21]. For OpenO1-Llama-8B-v0.1, DAPO has increased by 2.5% and 3.6% respectively on HumanEval [7] and HumanEval+ [22], by 6.1% and 5.0% respectively on MBPP [4] and MBPP+ [5], and decreased by 2.4% on LiveCodeBench [21]. Overall, these results strongly demonstrate the effectiveness of DAPO.
On the iterative DAPO. As suggested by our theoretical results (Theorem 3.2), there still remains the room for performance improvement if π ref is not the optimal policy. Thus we conduct two iterative DAPO experiments on Skywork-Math-Llama and Qwen2-Math-7B-Instruct. At iteration 2, we use the model optimized by DAPO at iteration 1 as π ref .
Our experiment results are summarized in Table 4.2 and Table 10. We can conclude from both tables that DAPO can further improves the performance of Skywork-Math-Llama and Qwen2-Math-7B-Instruct.
this section cite: ['b6', 'b21', 'b3', 'b4', 'b20', 'b6', 'b21', 'b3', 'b4', 'b20']

Section: Model Base +DAPO iter1 +DAPO iter2
Skywork-Math-Llama 41.90 +2.99 46.88 +4.98 50.54 +8.64 Qwen2-Math-7B-Instruct 74.46 +2.99 75.41 +0.95  76.40 +1.94   Table 4: Performance of iterative DAPO on 5K test samples from MATH [18]. Both Skywork-Math-Llama and Qwen2-Math-7B-Instruct show score improvement with additional iterations.
Ablation studies of hyperparameters. We conduct some preliminary ablation studies on the different components of DAPO. For the choice of KL coefficient β, we test different β in the experiments on Skywork-Math-Llama and the results are reported in Table 4.2. It can be seen that a suitable choice of β yields best improvement. We find β = 0.01 generally performs well on various models. Regarding the number of completions n used for MC value estimation, we test n = 8 and n = 16 in Model \ β Base β = 0.002 β = 0.01 β = 0.02 β = 0.05 β = 0. the experiments on Qwen2-Math-7B-Instruct and the results are reported in Table 4.2. We find that the difference is not significant between n = 8 and n = 16 in our experiment. However, to make sure a larger performance gain as possible, we use n = 16 in all our experiments.
this section cite: ['b17']

Section: Model \ n
Base n = 8 n = 16
Qwen2-Math-7B-Instruct 74.46 75.30 75.41
Table 6: Performance of DAPO on 5K test samples from MATH [18] with different number of completions for MC value estimation.
On the computation cost of DAPO. One major limitation that may hinder the efficient training-time scaling of DAPO is the high computation cost, particularly during the critic pre-training stage. We leave it as a future work for us to find a more efficient and scalable method for critic training. Please refer Section E for more discussions. That being said, given similar computation resources budget to the response-level baseline method, i.e. GRPO, we find that DAPO still outperforms GRPO in our experiment. We run GRPO experiment on Meta-Llama-3.1-8B-Instruct to more than 1000 training steps. The training and test curves can be found in Figure 4. It can be observed that the model exhibits serve overfitting after around 300 steps and the test performance begins to drop quickly. In contrast, as shown in Figure 3, DAPO improves the test performance steadily without exhibiting overfitting, ultimately achieving better performance gains compared to GRPO. This validates the effectiveness of granular policy optimization enabled by estimating step-level advantages.
this section cite: ['b17']

Section: Conclusion
In this work, we propose an offline step-level RLHF method called Direct Advantage-Based Policy Optimization (DAPO), which aims to optimize the generation of reasoning steps. DAPO significantly improves performance on both mathematical and coding benchmarks, demonstrating its effectiveness.
Compared with response-level algorithms, DAPO leverages the critic function for more fine-grained policy optimization. Compared with standard actor-critic methods, DAPO separates the training of the actor and critic into two distinct stages, stabilizing the RL training process while obtaining a good value function estimation. As we discussed before, the main limitation of DAPO is the high computation cost, indicating a need to find a more efficient implementation method. Besides, it also remains a great interest to see whether using a larger set of high-quality training queries would yield greater performance improvements.
this section cite: []

Section: References
Ref_id:b0 Title: On the theory of policy gradient methods: Optimality, approximation, and distribution shift Year: (2021)
Ref_id:b1 Title: Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms Year: (2024)
Ref_id:b2 Title: Enhancing textbook question answering task with large language models and retrieval augmented Year: (2024)
Ref_id:b3 Title: Program synthesis with large language models Year: (2021)
Ref_id:b4 Title: Program synthesis with large language models Year: (2021)
Ref_id:b5 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b6 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b7 Title: Teaching large language models to self-debug Year: (2023)
Ref_id:b8 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b9 Title:  Year: (2025)
Ref_id:b10 Title: Raft: Reward ranked finetuning for generative foundation model alignment Year: (2023)
Ref_id:b11 Title: Addressing function approximation error in actor-critic methods Year: (2018)
Ref_id:b12 Title: On designing effective rl reward at training time for llm reasoning Year: (2024)
Ref_id:b13 Title: Step-level value preference optimization for mathematical reasoning Year: (2024)
Ref_id:b14 Title: Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b15 Title: Psydial: Personality-based synthetic dialogue generation using large language models Year: (2024)
Ref_id:b16 Title: Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems Year: (2024)
Ref_id:b17 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021-12)
Ref_id:b18 Title: Reinforce++: A simple and efficient approach for aligning large language models Year: (2025)
Ref_id:b19 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (2025)
Ref_id:b20 Title: Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2024)
Ref_id:b21 Title: Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation Year: (2023)
Ref_id:b22 Title: Swe-bench: Can language models resolve real-world github issues Year: (2023)
Ref_id:b23 Title: Scaling laws for neural language models Year: (2025)
Ref_id:b24 Title: Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment Year: (2024)
Ref_id:b25 Title: Step-dpo: Step-wise preference optimization for long-chain reasoning of llms Year: (2024)
Ref_id:b26 Title: Deep Reinforcement Learning and the Deadly Triad. Hado van hasselt, yotam doron, florian strub, matteo hessel, nicolas sonnerat and joseph modayil Year: (2018)
Ref_id:b27 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b28 Title: Topics in algorithmic code generation dataset Year: (2023)
Ref_id:b29 Title: Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models Year: (2023)
Ref_id:b30 Title: Tpo: Aligning large language models with multi-branch multi-step preference trees Year: (2024)
Ref_id:b31 Title: Enhancing multi-step reasoning abilities of language models through direct q-function optimization Year: (2024)
Ref_id:b32 Title: On the convergence of projected policy gradient for any constant step sizes Year: (2024)
Ref_id:b33 Title: Elementary analysis of policy gradient methods Year: (2024)
Ref_id:b34 Title: Adaptivestep: Automatically dividing reasoning step through model confidence Year: (2025)
Ref_id:b35 Title: Understanding r1-zero-like training: A critical perspective Year: (2025)
Ref_id:b36 Title: Sstep-controlled dpo: Leveraging stepwise error for enhanced mathematical reasoning Year: (2024)
Ref_id:b37 Title: Improve mathematical reasoning in language models by automated process supervision Year: (2025)
Ref_id:b38 Title: Open-source o1 Year: (2025)
Ref_id:b39 Title: Skywork o1 Team. Skywork-o1 open series Year: (2024-11)
Ref_id:b40 Title: Webrl: Training LLM web agents via self-evolving online curriculum reinforcement learning Year: (2024)
Ref_id:b41 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b42 Title: Remi Munos, and Bilal Piot. Offline regularised reinforcement learning for large language models alignment Year: (2024)
Ref_id:b43 Title: Learning by playing solving sparse reward tasks from scratch Year: (2018)
Ref_id:b44 Title: Highdimensional continuous control using generalized advantage estimation Year: (2020)
Ref_id:b45 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b46 Title: Rewarding progress: Scaling automated process verifiers for llm reasoning Year: (2024)
Ref_id:b47 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b48 Title: Policy gradient methods for reinforcement learning with function approximation Year: (1999)
Ref_id:b49 Title: Mathscale: Scaling instruction tuning for mathematical reasoning Year: (2024)
Ref_id:b50 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b51 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2023)
Ref_id:b52 Title: Q*: Improving multi-step reasoning for llms with deliberative planning Year: (2024)
Ref_id:b53 Title: Offline reinforcement learning for llm multi-step reasoning Year: (2024)
Ref_id:b54 Title: Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning Year: (2024)
Ref_id:b55 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2024)
Ref_id:b56 Title: Monte carlo augmented actor-critic for sparse reward deep reinforcement learning from suboptimal demonstrations Year: (2022)
Ref_id:b57 Title: Monte carlo tree search boosts reasoning via iterative preference learning Year: (2024)
Ref_id:b58 Title: Qwen2 technical report Year: (2024)
Ref_id:b59 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b60 Title: Bootstrap your own mathematical questions for large language models Year: (2023)
Ref_id:b61 Title: What's behind ppo's collapse in long-cot? value optimization holds the secret Year: (2025)
Ref_id:b62 Title: Skywork-math: Data scaling laws for mathematical reasoning in large language models -the story goes on Year: (2024)
