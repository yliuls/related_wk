Title: d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning
Abstract: Recent large language models (LLMs) have demonstrated strong reasoning capabilities that benefits from online reinforcement learning (RL). These capabilities have primarily been demonstrated within the left-to-right autoregressive (AR) generation paradigm. In contrast, non-autoregressive paradigms based on diffusion generate text in a coarse-to-fine manner. Although recent diffusion-based large language models (dLLMs) have achieved competitive language modeling performance compared to their AR counterparts, it remains unclear if dLLMs can also leverage recent advances in LLM reasoning. To this end, we propose d1, a framework to adapt pre-trained masked dLLMs into reasoning models via a combination of supervised finetuning (SFT) and RL. Specifically, we develop and extend techniques to improve reasoning in pretrained dLLMs: (a) we utilize a masked SFT technique to distill knowledge and instill self-improvement behavior directly from existing datasets, and (b) we introduce a novel critic-free, policygradient based RL algorithm called diffu-GRPO, the first integration of policy gradient methods to masked dLLMs. Through empirical studies, we investigate the performance of different post-training recipes on multiple mathematical and planning benchmarks. We find that d1 yields the best performance and significantly improves performance of a state-of-the-art dLLM. Our code is released at https://dllm-reasoning.github.io/.

Section: 
1 Introduction 50 55 60 65 70 75 80 85 Accuracy (%) 78.2 82.1 GSM8K 20 25 30 35 40 45 36.2 40.2 MATH500 0 10 20 30 40 20.7 42.2 Countdown 0 5 10 15 20 25 11.7 22.1 Sudoku LLaDA
this section cite: []

Section: d1-LLaDA (ours)
Figure 1: Across four math and planning tasks, d1-LLaDA, which undergoes SFT followed by our proposed diffu-GRPO, consistently outperforms the base LLaDA-8B-Instruct model. We report results using the best performing generation sequence length for each task and model, with complete sequence length results shown in Table 1.
corrupted inputs, the learning objective is to predict the original tokens. The standard loss function for this task is the negative evidence lower bound (NELBO), which is an upper bound of the negative log-likelihood (NLL) of the data. For masked dLLMs, NELBO simplifies to a weighted NLL, where the weights are determined by a transformation of α t [36,Equation (10)]. In this work, we apply d1 on top of LLaDA [30], whose forward process sets α t = 1 -t and the resulting NELBO is
-E t∼U [0,1), x0∼pdata, xt∼q t|0 (xt|x0)   1 t |xt| k=1 1[x k t = mask] log f θ (x k 0 | x t )   ,(1)
where |x t | is the sequence length of x, and x k is the k-th token. Note that the loss is only calculated for tokens that are masked out in timestep t. The key difference between masked dLLMs and BERT [12] is that the latter uses a fixed masking ratio and the decoding is a single-step infilling process, whereas masked dLLMs use time-varying masking ratios and the decoding process involves multiple steps starting from pure noise and thus resulting in a generative model. Further details about the formulation of masked dLLMs are deferred to Appendix C.
this section cite: ['b36', 'b30', 'b11']

Section: Group Relative Policy Optimization for Large Language Models
Policy gradient methods have been widely adopted in the post-training stage to enhance the performance of LLMs [33,7,22,2]. While Proximal Policy Optimization (PPO) [37] has been the predominant approach in online RL, it requires jointly training a state value function V to estimate advantages, leading to increased computational demands. Group Relative Policy Optimization (GRPO) [38] offers a more efficient alternative by using group statistics to derive advantages. For each question q, GRPO samples a group of G responses {o 1 , o 2 , . . . , o G } from the old policy π θold . It then sets the advantages for all tokens k = 1, . . . , |o i | for o i as the normalized reward
ri-mean({rj } G j=1 ) std({rj } G j=1 )
.
Here, we can view mean({r j } G j=1 ) as a G-sample Monte Carlo estimation of the value V (q), while the sparse reward r i serves as the (undiscounted) state-action value Q(q, o i ). However, normalizing the advantage Q(q, o i ) -V (q) by nonzero state function introduces bias into policy gradient estimation. Therefore, similar to Liu et al. [24], we use the unnormalized advantage
A k i (π) = r i (π) -mean({r j (π)} G j=1 ), 1 ≤ k ≤ |o i |.(2)
The rest of our RL setup follows GRPO. The objective function incorporates a clipping mechanism (similar to PPO) to moderate policy updates, and a reverse KL penalty to prevent excessive deviation from the reference policy:
L GRPO (θ) = E q∼D o1,...,oG∼πθ(•|q)     1 G G i=1 1 |o i | |oi| k=1 min ρ k i A k i , clip ρ k i , 1 -ε, 1 + ε A k i   -βD KL [π θ (•|q)∥π ref (•|q)]   , (3
)
where π θ is the current policy being updated, π θold is the policy before the update,
ρ k i = π θ (o k i |q, o <k i ) π θ old (o k i |q, o <k i ) , A k
i is computed using π θold and Equation (2), and π ref is the reference policy (typically the initial model). The clipping parameter ε limits the magnitude of policy updates to ensure stability, while β controls the strength of the KL divergence regularization.
this section cite: ['b33', 'b6', 'b22', 'b1', 'b37', 'b38', 'b24', 'b1']

Section: d1: Adapting Pre-trained Masked dLLMs to Reasoning Models
We propose d1, a two-stage framework that enhances the reasoning performance of pre-trained masked dLLMs by sequentially combining SFT and online RL.
Online RL, particularly the GRPO algorithm, has demonstrated its efficacy in improving the performance of offline trained language model [38,17,41]. However, the learning formulation of GRPO does not directly generalize to dLLMs. The objective of GRPO (3) requires computing the (log-)likelihood ratio of π θ and π θold , at both the token level (for the advantage weights) and the sequence level (for the reverse KL term). Generally speaking, we need to efficiently compute the per-token and the sequence log-probability of dLLMs' completion o. Autoregressive (AR) models, such as Transformers, directly model the per-token log-probabilities, and the sequence-level log-probability of o can be easily computed through the chain rule using one forward pass: log π AR (o|q) = |o| k=1 log π AR (o k |q, o <k ). Similarly, the KL term can be decomposed Input: Partially-masked Prompt + Fully-masked Completion
this section cite: ['b38', 'b16', 'b41']

Section: Prompt Completion Mask

this section cite: []

Section: Iterative Denoising Steps

this section cite: []

Section: Masked dLLM Generation

this section cite: []

Section: One-Step Log Prob Estimation

this section cite: []

Section: Per-Token Log Prob

this section cite: []

Section: Different masking patterns for same prompt-completion pair

this section cite: []

Section: Random Prompt Masking for Each Gradient Update Iteration
Per-Token Log Prob ...
iter = 1 iter = 2 iter = n ... ... During each policy gradient update, we apply a random masking pattern to the prompt, creating q ′ , while keeping the completion fully masked (right). The gradient of colors in the per-token log probabilities demonstrates that each distinct masking pattern yields a different estimate of the per-token log probabilities. This serves as a form of regularization for policy optimization, allowing more gradient updates per batch and thereby reducing the number of online generations needed for RL training.
as D KL [π θ (•|q)∥π ref (•|q)] = E o∼π θ (•|q) |oi| k=1 log π θ (o k |q,o <k ) πref(o k |q,o <k )
. Unlike AR models, dLLMs do not adhere to sequential factorization of the sequence log-probability. Meanwhile, the per-token log-probability are also costly to compute since the decoding process invokes the unmasking predictor f θ multiple times 3 . As the first step, we propose an efficient log-probability estimator in Section 3.1. Next, using these estimators, we introduce diffu-GRPO, a variant of GRPO for dLLMs in Section 3.2. Last, we discuss our SFT recipe in Section 3.3.
this section cite: []

Section: Efficient Log Probability Estimation for Masked dLLMs
For sequence log-probability, we use a mean-field approximation that decomposes it into a product of independent per-token log-probabilities. For per-token log-probability, we introduce an estimation method that only calls f θ once.
this section cite: []

Section: Mean-Field Approximation of Sequence Log Probability.
As opposed to AR models, dLLMs treat the token sequence as a whole and therefore its sequence-level log-probability lacks the AR decomposition. To efficiently estimate it, we use a simple mean-field decomposition to approximate log π θ (o|q) by |o| k=1 log π θ (o k |q). The per-token log-probability estimation is introduced below.
this section cite: []

Section: One-Step Per-Token Log Probability Estimation with Prompt Masking.
Let ⊕ denote the concatenation operator. Given a prompt q, the decoding process starts from an initial sequence q ⊕ mask ⊕ . . . ⊕ mask (up to a preset length). To compute the log-probability of o, we perturb q where every token is randomly masked out with probability p mask , resulting in a new prompt q ′ . We then do one-step unmasking to obtain log f θ (o k |q ′ ⊕ mask . . . ⊕ mask) and use it as an estimation of log π θ (o k |q), 1 ≤ k ≤ |o|. We discuss the motivation of using a masked prompt q ′ in the next section.
We note that LLaDA [30, Algorithm 3] uses a Monte Carlo type of approximation to estimate the log-probabilities, where they use a MC sample size is 128. This estimator is inefficient for online RL, since it creates a large computational graph with hundreds of forward passes, resulting in inefficient policy optimization and excessive memory usage.
this section cite: []

Section: diffu-GRPO: Policy Gradient Optimization for Masked dLLMs
Using the log-probability estimators proposed in Section 3.1, we extend GRPO to masked dLLMs. Note that our estimation technique is broadly applicable and can readily extend to other policy gradient methods such as PPO [37] or REINFORCE [44].  for gradient update iterations n = 1, . . . , µ do 8:
q ′ ← randomly mask tokens of prompt p with probability p mask 9:
For π θ , π θold , π ref , estimate log-probabilities of o i given q ′ according to Section 3.1 10:
Compute diffu-GRPO objective (4) and update π θ by gradient descent 11: return π θ Let ϕ π θ (o k | q ′ ) and ϕ π θ (o | q ′ ) denote the estimated per-token and sequence probabilities for π θ . We derive the loss function of diffu-GRPO,
L diffu-GRPO (θ) = E q∼D, q ′ ∼masking(q), o1,...,oG∼π θ old ( •|q) 1 G G i=1 1 |o i | |oi| k=1 min ϕ π θ (o k i | q ′ ) ϕ π θ old (o k i | q ′ ) A k i , clip ϕ π θ (o k i | q ′ ) ϕ π θ old (o k i | q ′ ) , 1 -ε, 1 + ε A k i -β D KL ϕ π θ (• | q ′ ) ϕ πref (• | q ′ )(4)
Our algorithm is summarized in Algorithm 1. To efficiently optimize the policy loss, in practice, on-policy RL algorithms such as PPO and GRPO perform multiple gradient updates for each batch of samples. During these updates, the prompt q,
completions {o i } G i=1
, old policy π θold and advantages A k i (π θold ) are kept fixed. However, determining the optimal number of gradient updates per batch is challenging. If the number is too high, it can lead to overfitting within the batch, while a number that is too low slows down convergence. Achieving a balance between outer batch iterations and inner gradient updates is crucial for sample efficiency. Besides, every outer batch iteration requires sampling completion through iterative denoising steps, which incurs high computational cost.
Interestingly, our log-probability estimator offers a unique mitigation to this dilemma. For each gradient update step, we randomly mask the prompt q to q ′ to estimate the log-probabilities. Intuitively, this stochastic masking introduces perturbed views of the same (prompt, completion) pairs, serving as a form of regularization for policy optimization. It can also be viewed as a form of data augmentation, extracting more supervision signals from the same data. Empirically, we found that this approach, unique to masked diffusion models, allows us to scale µ to higher values while maintaining stable learning dynamics. As a consequence, it reduces the number of outer batch iterations required for convergence, which in turn decreases the number of online generations needed and ultimately results in significantly lower computational cost. As shown in Figure 5, training with higher values of µ achieves the same reward performance in substantially less wall clock time.
this section cite: ['b37', 'b44']

Section: Supervised FineTuning with Reasoning Data
We perform SFT of LLaDA on s1K [28], a curated dataset consisting of 1000 high-quality reasoning questions. The reasoning traces in s1K exhibit detailed step-by-step problem-solving processes, including verification of intermediate results and backtracking when encountering errors or dead ends. The SFT algorithm is summarized in Algorithm 2, where tokens are randomly masked during training according to a time-varying schedule. The model is optimized to predict the original tokens given their context. We find that for SFT to work effectively in practice, various design choices must be carefully considered, whose details are discussed in Appendix D.2.
this section cite: ['b28']

Section: Experiments
To understand how reasoning capabilities can be scaled in masked dLLMs through training adaptations, we conduct comprehensive experiments to answer the following main research questions:  .8 38.6 40.2 34.8 32.0 42.2 22.1 16.7 9.5(1) How do SFT on reasoning traces and applying diffu-GRPO independently improve LLaDA's reasoning capabilities? (2) What additional gains can be achieved by combining SFT and diffu-GRPO to create d1-LLaDA? (3) Design Choices: How does the proposed log-probability estimation with randomized masking in diffu-GRPO and the masking probability p mask affect training efficiency and stability? Tasks We conduct experiments on six reasoning tasks in three categories: (1) Mathematical reasoning: we use GSM8K [10], a dataset of multi-step grade school math problems, and MATH500 [23], a curated subset of 500 problems drawn from the MATH dataset [18] comprising high-school competition math problems; (2) Planning: this includes two tasks: 4x4 Sudoku puzzles, which require constraint satisfaction and systematic elimination to fill a grid with numbers; and Countdown with 3 numbers, a combinatorial arithmetic game in which models must reach target numbers using basic arithmetic operations on a given set of numbers. (3) Coding: comprises of two benchmarks; Hu-manEval [8], a suite of 164 hand-crafted Python algorithmic programming problems and MBPP [6], a crowd-sourced collection of 257 Python tasks.
Training For SFT, we train on s1k [28] for 20 epochs, with a sequence length of 4096. For RL, we train a separate model for each task. More specifically, for GSM8K, MATH500, we train on the training split; for Countdown and Sudoku, we train on synthetic generated datasets. We use a composed reward function that combines both formatting and correctness rewards. Due to the heavy computational cost of online generations, we limit the generation sequence length of online generations to be 256 throughout RL training. Other hyperparameters of training, training and evaluation datasets, reward functions, and inference setups are detailed in Appendix D.
Evaluation For all the benchmarks, we evaluate LLaDA-8B-Instruct and LLaDA+SFT on the final checkpoint for all the tasks. For LLaDA+diffu-GRPO and d1-LLaDA, we evaluate every 100 steps starting from step 600 and report the best results. We evaluate all models with 0-shot-prompting and greedy decoding with generation lengths of 128, 256 and 512 separately.
this section cite: ['b9', 'b23', 'b17', 'b7', 'b5', 'b28']

Section: Main Results
diffu-GRPO outperforms both LLaDA and SFT and improves over initialization checkpoint consistently . Table 1 reports the performance of baseline LLaDA-8B-Instruct and models obtained by different post-training recipes across four tasks using zero-shot evaluation, where each diffu-GRPO model was trained for each task. For each task, we evaluate with three generation sequence lengths, and Figure 4 plots the average number of effective tokens. We present the following predominent findings.
Both diffu-GRPO and SFT yield improvements over the LLaDA-8B-Instruct baseline, with diffu-GRPO demonstrating consistently larger gains. Specifically, diffu-GRPO outperforms both LLaDA-8B-Instruct and SFT, in all 12 setups, while SFT outperforms LLaDA-8B-Instruct in only 7 of them, demonstrating that diffu-GRPO achieves stronger overall performance than SFT alone. Both LLaDA+diffu-GRPO and d1-LLaDA demonstrate consistent improvements over their respective starting points. Specifically, LLaDA+diffu-GRPO outperforms the base LLaDA-8B-Instruct model across all setups, and d1-LLaDA surpasses LLaDA+SFT in every case. This indicates that diffu-GRPO provides reliable performance gains, regardless of the initialization-whether from a pretrained model or an SFT-adapted checkpoint.
d1 recipe yields the highest gains. SFT, followed by diffu-GRPO-resulting in d1-LLaDA-yields additional gains, beyond either method individually. This combined approach outperforms pure diffu-GRPO in 11 out of 12 setups, indicating a synergistic effect between the two training stages. Notably, while d1-LLaDA shows consistent improvements across all benchmarks, the magnitude varies by task: we observe modest improvements on GSM8K (3.9%) and MATH500 (4.0%), but significantly larger gains on Countdown (26.2%) and Sudoku (10.0%). We hypothesize this discrepancy stems from the base model's saturation on mathematical tasks, with less room for improvement as compared to planning benchmarks that involve structured constraint satisfaction patterns.
Training a unified model across tasks retains strong performance. We train a single diffu-GRPO (and d1) model on the combined GSM8K, MATH500, Countdown, and Sudoku datasets. To ensure balanced training, we subsample the data so that each task has the same number of training examples.
Even with subsampling, Table 2 shows that diffu-GRPO scales well to multi-task settings without sacrificing accuracy compared to the per-task diffu-GRPO results in Table 1.
Scaling diffu-GRPO to coding domains.
We also evaluate diffu-GRPO on coding tasks, where we train a model on the KodCode-Light-RL-10K dataset [45], which contains general coding tasks with solutions verified by synthetic unit tests. The diffu-GRPO results are shown in Table 3. We find that diffu-GRPO consistently improves performance, regardless of the initialization point. Interestingly, our findings suggest that s1k is not suitable for coding, since it lacks datapoints with code. Exploration into finding the optimal SFT dataset is left for future works. diffu-GRPO improves reasoning beyond training sequence length. Although our diffu-GRPO training uses fixed sequence length of 256 for online generations, we observe performance gains at other generation sequence lengths as well. The improvements at 128 and 512 sequence lengths suggest that the model has learned more general reasoning strategies rather than overfitting to a specific length. This is further supported by the effective token usage data, presented in Figure 4, which shows no truncation at 128 tokens and increased token utilization at 512.
this section cite: ['b45']

Section: Discussion
Qualitative results show "aha moments" in SFT and d1-LLaDA generations. While the performance for generation sequence length 128 and 256 increases with SFT, diffu-GRPO and d1 as compared to LLaDA-8B-Instruct, qualitatively, we do not observe significant differences in the generated reasoning traces. However, at sequence length 512, we begin observing "aha moments" in the SFT and d1-LLaDA models, which demonstrates self-correction and backtracking behaviors. We show these in Appendix E. For the same questions from GSM8k, we show generations of each model, with the variants using SFT showing self-verifications and self-corrections to the right answer. Our intuition is that the model has instilled behaviors such as verification of intermediate results and backtracking from the reasoning traces of s1k during the SFT stage. Seq=128 Seq=256 Seq=512 0 100 200 300 Num Effective Tokens GSM8K (0shot) Seq=128 Seq=256 Seq=512 0 100 200 300 400 MATH500 (0shot) Seq=128 Seq=256 Seq=512 0 100 200 300 400 Countdown (0shot) Seq=128 Seq=256 Seq=512 0 100 200 300 400 500 Sudoku (0shot) LLaDA8BInstruct + SFT + diffuGRPO + SFT + diffuGRPO (d1LLaDA) Figure 4: Effective Token Usage: As we increase the evaluation generation length, the number of effective tokens (average number of non-padding, non-EOS tokens per generation across tasks) grows and remains comparable for all the methods on MATH500, Countdown and Sudoku tasks.
Sequential scaling with increasing generation sequence lengths. LLaDA-8B-Instruct, SFT, diffu-GRPO and d1-LLaDA demonstrate improved performance with increasing sequence lengths for GSM8k and MATH500, with larger jumps observed from 128 to 256 (∼ 7.1%), than from 256 to 512 (∼ 2.5%). Qualitative examples in Appendix E show more sophisticated reasoning traces emerge with 512-token generation lengths. These findings align with previous research showing that increasing test-time compute through longer reasoning processes leads to improved performance in autoregressive models [28]. However, we notice a mixed scaling trend on Countdown and Sudoku. Performance decreases with increasing sequence lengths for Sudoku across all models. For Countdown, LLaDA-8B-Instruct decreases monotonically with sequence length, while SFT, diffu-GRPO and d1-LLaDA peak at 512 sequence length. This likely stems from extensive searching requirements, beyond LLaDA-8B-Instruct's capabilities. We hypothesize favorable sequential scaling will strengthen with more robust base dLLMs. Unlike AR models like DeepSeek R1 [17], we observe no significant CoT length growth post-RL training, as LLaDA-8B-Instruct was pre-trained on sequences up to 4096 tokens. Further scaling requires larger generation lengths during RL training, currently infeasible due to slow generation speed. Future research should develop efficient inference algorithms for online sampling to scale dLLM RL training.
this section cite: ['b28', 'b16']

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms Year: (2024)
Ref_id:b2 Title: Arel's sudoku generator Year: (2025)
Ref_id:b3 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b4 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b5 Title: Program synthesis with large language models Year: (2021)
Ref_id:b6 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b7 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b8 Title: Sft memorizes, rl generalizes: A comparative study of foundation model post-training Year: (2025)
Ref_id:b9 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b10 Title: FlashAttention-2: Faster attention with better parallelism and work partitioning Year: (2024)
Ref_id:b11 Title: BERT: Pre-training of deep bidirectional transformers for language understanding Year: (2019-06)
Ref_id:b12 Title: The llama 3 herd of models Year: (2024)
Ref_id:b13 Title: Rlef: Grounding code llms in execution feedback with reinforcement learning Year: (2024)
Ref_id:b14 Title: Scaling diffusion language models via adaptation from autoregressive models Year: (2025)
Ref_id:b15 Title: Likelihood-based diffusion language models Year: (2023)
Ref_id:b16 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b17 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b18 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b19 Title: Mercury: Ultra-fast language models based on diffusion Year: (2025)
Ref_id:b20 Title:  Year: ()
Ref_id:b21 Title:  Year: (2024)
Ref_id:b22 Title: Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models Year: (2023)
Ref_id:b23 Title: Ilya Sutskever, and Karl Cobbe. Let's verify step by step Year: (2023)
Ref_id:b24 Title: Understanding r1-zero-like training: A critical perspective Year: (2025)
Ref_id:b25 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b26 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: ()
Ref_id:b27 Title: Dynamic scaling of unit tests for code reward modeling Year: (2025)
Ref_id:b28 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b29 Title: Scaling up masked diffusion models on text Year: (2024)
Ref_id:b30 Title: Large language diffusion models Year: (2025)
Ref_id:b31 Title: Learning to reason with llms Year: (2024-09)
Ref_id:b32 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2024)
Ref_id:b33 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b34 Title:  Year: (2025)
Ref_id:b35 Title: Openwebmath: An open dataset of high-quality mathematical web text Year: (2023)
Ref_id:b36 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b37 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b38 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b39 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b40 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b41 Title: Kimi k1. 5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b42 Title: OpenThoughts Team. Open Thoughts Year: (2025-01)
Ref_id:b43 Title: Trl: Transformer reinforcement learning Year: (2020)
Ref_id:b44 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b45 Title: Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding Year: (2025)
Ref_id:b46 Title: Beyond autoregression: Discrete diffusion for complex reasoning and planning Year: (2024)
Ref_id:b47 Title: Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models Year: (2024)
Ref_id:b48 Title:  Year: (2025)
Ref_id:b49 Title: Less is more for reasoning Year: (2025)
Ref_id:b50 Title: Bootstrap your own mathematical questions for large language models Year: (2023)
Ref_id:b51 Title: Fine-tuning discrete diffusion models with policy gradient methods Year: (2025)
Ref_id:b52 Title: Lima: less is more for alignment Year: (2023)
