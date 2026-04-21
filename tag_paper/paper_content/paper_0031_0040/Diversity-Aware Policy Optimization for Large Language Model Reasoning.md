Title: Diversity-Aware Policy Optimization for Large Language Model Reasoning
Abstract: The reasoning capabilities of large language models (LLMs) have advanced rapidly, particularly following the release of DeepSeek-R1, which has inspired a surge of research into data quality and reinforcement learning (RL) algorithms. Despite the pivotal role diversity plays in RL, its influence on LLM reasoning remains largely underexplored. To bridge this gap, this work presents a systematic investigation into the impact of diversity in RL-based training for LLM reasoning, and proposes a novel diversity-aware policy optimization method. Across evaluations on 12 LLMs, we observe a strong positive correlation between the solution diversity and Potential@k (a novel metric quantifying an LLM's reasoning potential) in high-performing models. This finding motivates our method to explicitly promote diversity during RL training. Specifically, we design a token-level diversity and reformulate it into a practical objective, then we selectively apply it to positive samples. Integrated into the R1-zero training framework, our method achieves a 3.5% average improvement across four mathematical reasoning benchmarks, while generating more diverse and robust solutions. The code is available at https://github.com/nigelyaoj/R1_zero_Div.

Section: Introduction
Recently, the reasoning capabilities of large language models (LLMs) have made remarkable progress, with significant improvements showcased by OpenAI-o1 [35], DeepSeek-R1 [13], and Kimi-k1.5 [43]. Among these advancements, two key innovations have contributed significantly: First, the adoption of a rule-based reward system significantly streamlines the training process by focusing exclusively on rewarding correct final answers and proper output formats, thereby eliminating the complexity associated with process-based reward models [30,47]. Second, the introduction of a lightweight reinforcement learning (RL) algorithm [13,43] removes the need for a separate critic model, substantially reducing computational overhead and accelerating the training process. The success of DeepSeek-R1 has attracted numerous follow-up studies [62], which broadly fall into two categories. The first category focuses on improving the quality of training data [33,15,21,2], emphasizing rigorous data set curation through filtering, deduplication, and verification. The second category refines RL algorithms, including detailed optimizations for PPO-based methods such as VCPPO [56] and VAPO [55]; enhancements to GRPO for stability and speed, such as DAPO [54], Dr.GRPO [31] and SRPO [63]; as well as alternative approaches such as REINFORCE++ [20].
While RL has been extensively applied to LLM reasoning, the role of diversity remains largely unexplored in this context, even though it plays a crucial role in RL research [19,11,36,8,37,32,64,12,58,66,6,52]. In traditional RL tasks, incorporating diversity is widely recognized to facilitate exploration by promoting the selection of more stochastic policies, which helps the policy escape local optima and accelerate the convergence of training. This hypothesis has been experimentally validated in previous work [19,11,36]. Beyond empirical evidence, theoretical analyses suggest that policies with higher entropy (a measure of diversity) can smooth the optimization landscape [1]. These findings naturally lead us to ask the following question: Is promoting diversity essential during RL training for LLM reasoning?
Intuitively, an LLM capable of generating diverse responses could broaden the exploration of reasoning paths, enabling the model to avoid overfitting to narrow solution patterns in mathematical or logical tasks. To formally address this question, we conduct an evaluation of diversity in LLM reasoning, with a specific focus on mathematical problem-solving. We introduce a novel metric, Potential@k, to quantify an LLM's reasoning potential (the possible performance gain after RL training). We empirically analyze 12 representative LLMs, examining both their solution diversity and Potential@k scores. Notably, our results reveal a strong positive correlation between solution diversity and Potential@k scores among high-performing models, which suggests that diversity directly contributes to improved final performance after RL training.
The empirical findings motivate us to promote diversity during RL training for LLM reasoning. A commonly used approach for this goal is entropy regularization. However, directly increasing the average entropy of LLM outputs can introduce length bias, as longer responses inherently exhibit higher entropy. To address this, we introduce a token-level diversity metric and reformulate the diversity objective into a practical form. Moreover, promoting diversity often entails a qualitydiversity trade-off. To mitigate this, we strategically apply diversity enhancement only to positive samples, thereby enriching solution diversity while preserving training stability. This design is akin to fostering diversity in high-quality policies in population-based RL training, ensuring that exploration is guided by task-relevant performance criteria [48]. Finally, we integrate our diversity objective into the R1-zero training method and evaluate the enhanced approach across 4 mathematical reasoning benchmarks. Experimental results demonstrate a 3.5% average performance gain over standard R1-zero training, while our method can generate more diverse solutions.
To summarize, our key contributions are:
• We present the first formal investigation into the role of diversity in LLM reasoning. Through experiments on mathematical benchmarks, we identify a positive correlation between solution diversity and an LLM's reasoning potential, as measured by our proposed Potential@k metric. This finding provides empirical motivation for incorporating diversity into policy optimization.
• We propose a novel token-level diversity objective, which is reformulated into a practical metric and selectively applied to positive samples. This design is further supported through gradient behavior analysis, offering an insight for balancing quality and diversity during optimization.
• We evaluate our method on four mathematical reasoning benchmarks, each comprising at least 500 problems with stable evaluation metrics. Our method achieves a 3.5% average improvement over standard R1-zero training and consistently produces more diverse solutions.
this section cite: ['b34', 'b12', 'b42', 'b29', 'b46', 'b12', 'b42', 'b61', 'b32', 'b14', 'b20', 'b1', 'b55', 'b54', 'b53', 'b30', 'b62', 'b19', 'b18', 'b10', 'b35', 'b7', 'b36', 'b31', 'b63', 'b11', 'b57', 'b65', 'b5', 'b51', 'b18', 'b10', 'b35', 'b0', 'b47']

Section: Preliminary

this section cite: []

Section: RL for LLMs
In the context of RL for LLMs, we frame the LLM generation process as an RL problem. Here, the LLM is modeled as a policy that produces outputs (actions) conditioned on input prompts (states) and receives evaluative feedback (rewards) for its generated responses. This formulation aligns the sequential decision-making nature of language generation with RL's state-action-reward framework, enabling systematic optimization of the model's behavior through reward signals.
Formally, in the context of LLM generation for mathmatical problem-solving, where each prompt is a question, we define the prompt as q ∈ Q, where Q represents the set of all possible questions. The set of all potential text outputs o forms an action space O. Each output o consists of tokens, denoted as o := (o 1 , o 2 , ..., o t , ...). To generate an output, a policy π θ (•|q) parameterized by θ is employed, which generates the output according to the distribution:
π θ (o|q) := t π θ (o t |q, o <t ),(1)
where o <t = (o 1 , o 2 , ...o t-1 ).
this section cite: []

Section: Reinforcement Learning algorithm
The R1-zero training method proposed by DeepSeek-R1 [13] has attracted significant research attention due to its computational efficiency and effectiveness. In our work, we adopt this training method as our backbone. R1-zero incorporates two key innovations: the GRPO algorithm [40] and a rule-based reward function. In this section, we introduce both components.
Group Relative Policy Optimization (GRPO) GRPO streamlines the process by eliminating the need for a separate critic model, which is usually as large as the policy model, and instead estimates baselines using group scores. Specifically, for each question q, GRPO samples a group of outputs {o 1 , o 2 , ..., o G } from the old policy π old and optimizes the policy π θ by maximizing the following objective:
J GRP O (π θ ) = E q∼Q,{oi} G i=1 ∼π old (•|q) 1 G G i=1 min π θ (o i |q) π old (o i |q) A i , clip π θ (o i |q) π old (o i |q) , 1 -ϵ, 1 + ϵ A i -βD KL (π θ ||π ref ) ,(2)
where ϵ and β are hyperparameters, the KL term is defined as
D KL (π θ ||π ref ) = π ref (o i |q) π θ (o i |q) -log π ref (o i |q) π θ (o i |q) -1,(3)
and the advantage A i is computed using a group of rewards {r 1 , r 2 , ..., r G }:
A i = r i -mean({r 1 , r 2 , ..., r G }) std({r 1 , r 2 , ..., r G }) . (4
)
this section cite: ['b12', 'b39']

Section: Reward functions
In line with DeepSeek-R1 [13], we implement two types of rule-based rewards: accuracy rewards and format rewards. The accuracy reward model assesses whether the response is correct by comparing the predicted answer to the golden reference answer, while the format reward model ensures that the final answer is presented in a \boxed{} format for reliable verification.
this section cite: ['b12']

Section: Correlation between LLMs' reasoning potential and solution diversity
The role of diversity has long been established as critical in traditional RL tasks. Numerous studies [19,11,36,8,37] have shown that promoting diversity can enhance the final quality of the policy. However, its impact in the realm of RL for LLM reasoning still remains under-explored. In this section, we investigate the relationship between solution diversity and the reasoning abilities of LLMs on mathematical benchmarks. We adopt the equation diversity in prior work [49] to quantify the variety of solutions generated for mathematical problem-solving. For reasoning ability, we introduce a novel metric to evaluate an LLM's training potential (related to the performance gain achieved after RL training).
this section cite: ['b18', 'b10', 'b35', 'b7', 'b36', 'b48']

Section: Experimental setup
We evaluate 12 LLMs on the MATH benchmark [16]. For each question, we calculate: (1) Pass@1 accuracy using greedy decoding, and (2) Diversity with (3) Potential@k, both evaluated from 16 sampled responses (temperature=0.9).
For diversity, we adopt the metric (denoted as Div-Equ) from prior work [49], which measures the ratio of distinct equations among the responses: where U i and A i are the sets of unique equations and all equations extracted from the k sampled responses (with k = 16 in our experiments) of question i, respectively. And N = 500 is the amount of the data.
Div-Equ := 1 N N i=1 |U i | |A i | ,(5)
For Potential, we define a metric termed Potential@k to quantify the model's capability to correct answers within k trials (with k = 16 in our experiments) on its Pass@1 failure samples. Formally:
Potential@k := N i=1 Pass@k(q i ) • (1 -Pass@1(q i )) N i=1 (1 -Pass@1(q i )) ,(6)
where q i denotes the i-th question.
this section cite: ['b15', 'b48']

Section: Empirical findings
The results are shown in Figure 1a. The results show a bifurcated pattern: For LLMs with limited reasoning ability (Pass@1 < 0.4), we observe no significant relationship between solution diversity and model potential. For stronger performers (Pass@1 > 0.4), a clear positive correlation emerges between these metrics. Linear regression on this high-performing subset yields R 2 = 0.81, confirming a strong predictive relationship where increased diversity corresponds to higher model potential.
Through an investigation of the Objective 2 in the GRPO algorithm, we observe that for each question in the training set, if all samples within a group are either entirely positive or entirely negative, the advantage score becomes 0, resulting in no gradient update. Crucially, the training signal originates from the reward discrepancy between positive and negative samples within the group, which is inherently linked to our definition of potential (to some extent, the algorithm's improvement can be characterized by the dynamics of this potential metric, as discussed in Appendix B). This indicates that promoting diversity for LLM may result in higher performance after RL training.
Takeaways A positive correlation between the LLM's reasoning potential and solution diversity is observed in our experiment. As illustrated in Section 2.2, the optimization direction is guided by correct answers in multiple sampled responses. This directly links our Potential@k metric to RL training improvements. Hence, the observation strongly motivates us to enhance diversity during the RL training process.
this section cite: []

Section: Diversity-aware policy optimization
Building on the insights from Section 3, in this section, we introduce an entropy-based diversity and propose its targeted application to positive samples during policy optimization for LLM reasoning. We incorporate this diversity objective into the R1-zero training method [13], which employs the GRPO algorithm with the reward function defined in Section 2.2. We refer to this enhanced approach as R1-zero-Div.
this section cite: ['b12']

Section: Entropy-based diversity
A straightforward approach is to define diversity as the average entropy of the LLM's outputs per question i.e., E q∼Q [H(π θ (•|q))]. However, this formulation introduces length bias: longer responses inherently exhibit higher entropy (due to more token-level uncertainties), causing the metric to artificially favor longer outputs regardless of actual solution diversity. To address this issue, we introduce token-level entropy, which calculates the entropy for each token sampled from the old policy π old . Formally, we define:
J Div (π θ ) := E q∼Q,o∼π old (•|q) 1 T T t=1 H(π θ (•|q, o <t )) ,(7)
where T is the length of the output.
During training, the gradient of diversity with respect to the policy π θ in the H(π θ (•|q, o <t )) is intractable. We therefore reformulate the diversity objective to enable effective backpropagation:
J Div (π θ ) = E q∼Q,o∼π old (•|q) - 1 T T t=1 E o t ∼π θ (•|q,o <t ) [log π θ ( o t |q, o <t )] = E q∼Q,o∼π old (•|q) - 1 T T t=1 π θ (o t |q, o <t ) π old (o t |q, o <t ) log π θ (o t |q, o <t ) .(8)
A proof for the last equation can be found in Appendix A.1. In practice, building on the R1-zero training method, we can use the samples within the group to calculate Objective 8.
this section cite: []

Section: Promoting diversity on positive samples
Empirical evidence indicates that the direct application of Objective 8 inadvertently increases diversity in incorrect solutions. Intuitively, negative samples offer more room for diversity enhancement, which can skew the model's optimization process. To address this issue, we concentrate on promoting diversity exclusively within positive samples:
J Div (π θ ) = E q∼Q,o∼π old (•|q) -I(r = 1) • 1 T T t=1 π θ (o t |q, o <t ) π old (o t |q, o <t ) log π θ (o t |q, o <t ) ,(9)
where I(•) denotes the indicator function and r is the accuracy reward for output o.
This is akin to fostering diversity in high-quality policies in population-based RL training [48], while we focus on positive samples rather than policies here. Beyond intuitive justification, we further justify this design by analyzing the gradient on each token.
According to Equation 8, we have:
∇ π θ J Div (π θ ) = E q∼Q,o∼π old (•|q) - 1 T T t=1 ∇ θ [π θ (o t |q, o <t ) log π θ (o t |q, o <t )] π old (o t |q, o <t ) . (10
)
Thus, the gradient can be decomposed into per-token contributions (each term in the summation contributes a component). Up to a constant scaling factor, the gradient from each token is:
-∇ θ π θ (o t |q, o <t ) log π θ (o t |q, o <t ) = -[1 + log π θ (o t |q, o <t )] • ∇ θ π θ (o t |q, o <t ).(11)
Hence, for tokens with small probabilities (in that case π θ (o t |q, o <t ) < e -1 , and this holds for most of tokens since the sum of probability is equal to 1), the gradient aligns with ∇ θ π θ (o t |q, o <t ). This suggests that the diversity component's gradient actively promotes increasing the probability of low-probability tokens, which inherently offer substantial growth potential. However, this tendency is undesirable for negative samples. Thus, excluding diversity enhancement for negative samples mitigates conflicts between solution quality and diversity. A visual illustration is provided in Figure 1b. Moreover, the experimental results in Section 5.3 and Appendix E.1 further support our design.
Finally, we incorporate the diversity optimization into the standard R1-zero training, and use the samples in the group to calculate the diversity, yielding the final training objective:
J(π θ ) =J GRP O (π θ ) + λ • J Div (π θ ) =E q∼Q,{oi} G i=1 ∼π old (•|q) 1 G G i=1 min π θ (o i |q) π old (o i |q) A i , clip π θ (o i |q) π old (o i |q) , 1 -ϵ, 1 + ϵ A i -βD KL (π θ ||π ref ) -λI(r i = 1) • 1 T i Ti t=1 π θ (o t i |q, o <t i ) π old (o t i |q, o <t i ) log π θ (o t i |q, o <t i ) , (12
)
where λ is the diversity weight and i denotes the i-th sample in the group. In practice, we choose λ = 0.01. Other implementation details are provided in Section 5.1 and Appendix D.
this section cite: ['b47']

Section: Experiments
In this experimental section, we aim to address the following questions:
Q1. Can our method effectively enhance reasoning abilities and provide diverse solutions? Q2. Does the design of the diversity coefficient λ influence the results? Q3. Does our method demonstrate consistent performance across different model sizes?
this section cite: []

Section: Experimental setup
Base models We choose Qwen2.5-Math-7B (Qwen7B) [51] as our base model, which is commonly used for mathematical reasoning benchmarks [59,67,24]. Additionally, we conduct an ablation study using Qwen2.5-Math-1.5B (Qwen1.5B) [51] to assess the effectiveness of our approach in smaller LLMs.
this section cite: ['b50', 'b58', 'b66', 'b23', 'b50']

Section: Benchmarks
We selected 4 mathematical benchmarks to evaluate the models' reasoning abilities: GSM8K [7], MATH500 [16], Olympiad Bench [14], and College Math [42]. Each contains at least 500 data points for testing. We excluded some commonly used mathematical benchmarks that provide limited data, e,g, AIME24foot_0 with 30 items, as they can lead to unstable and biased evaluation outcomes. We train the base model on the GSM8K training set and then evaluate on the 4 benchmarks.
Baselines The most pertinent baselines for comparison are the base model itself and the base model trained via R1-zero. Additionally, we incorporate the latest prominent "R1-zero-Like" models with similar backbones for reference: SimpleRL-Zoo [59], PRIME-Zero-7B [9]. It is important to note that these methods are trained with different computational resources and datasets, making direct comparisons challenging. Our approach is designed to enhance diversity rather than compete directly with these methods. In fact, our method is compatible with and can be integrated into these existing approaches.
Implementation details For R1-zero-Div, we train the base model on the GSM8K training set using the loss function in Equation 12, with a learning rate of 3 × 10 -6 and the AdamW optimizer. During rollout, we sample 6 responses with a temperature of 0.9 and train for 2 epochs. Our implementation is built on TRL [46] and runs on 8×A6000 GPUs. For R1-zero, we maintain identical settings to R1-zero-Div but exclude the diversity objective. For other baselines, we evaluate open-sourced models downloaded from Hugging Face 3 , following the settings recommended in their original papers. Additional implementation details are provided in Appendix D.
this section cite: ['b6', 'b15', 'b13', 'b41', 'b58', 'b8', 'b45']

Section: R1-zero-Div enhances reasoning abilities
We evaluate the reasoning performance using Pass@1 accuracy, as shown in Table 1. In our experiment, R1-zero-Div demonstrates superior performance compared to R1-zero, achieving an average improvement of 3.5%. Despite being trained with limited computational resources (discussed in Appendx B), R1-zero-Div achieves comparable results to stateof-the-art methods (SimpleRL-Zoo and Eurus-2-7B-PRIME). These results suggest that promoting diversity on positive samples in training can effectively enhance the model's reasoning capabilities. Also, following the recommendations in prior work [4,17], we evaluated 8 samples per question with a temperature of 0.5. We report Avg@8 and its standard error in the Table 2. The conclusion regarding the effectiveness of our approach remains consistent with the pass@1 metric results.
this section cite: ['b3', 'b16']

Section: R1-zero-Div generates diverse solutions
We empirically demonstrate that R1-zero-Div produces more diverse solutions than other RL-finetuning baselines. Our evaluation on the GSM8K test set generates 5 responses for each of 1,319 questions, measuring diversity through three metrics: Div-Equ, and two additional metrics in prior work [26]: (1) N-gram diversity (proportion of distinct n-grams per response, capturing intra-diversity) and (2) Self-BLEU diversity (100 minus Self-BLEU score, capturing inter-diversity). All metrics range from 0 to 100, with higher values indicating greater diversity. As shown in Table 3, while RL fine-tuning methods significantly reduce diversity (compared to the base model), R1-zero-Div effectively preserves diversity. We further provide concrete examples in Appendix E showing that R1-zero-Div generates distinct solutions for the same question.
this section cite: ['b25']

Section: Ablation study
We conduct an ablation study to analyze (1) the impact of different diversity weights and (2) our method's generalization capability on smaller base models.
Analysis on the choice of diversity weights λ Table 4 presents Pass@1 accuracy when applying different λ values to promote diversity on positive samples (denoted as "pos"). The results demonstrate that small values (λ ≤ 0.02) effectively enhance reasoning performance, with λ = 0.01 emerging as the optimal choice in our experimental setup. We further compare diversity promotion strategies: positive samples only ("pos") versus all samples ("pos+neg"). The marginal improvement observed when applying diversity to all samples supports our methodological design choice in Section 4.2. Experiment on 1.5B base model We perform both R1-zero-Div and R1-zero on the Qwen2.5-Math-1.5B base model [51], with results shown in Table 5. The experiments demonstrate that, compared to R1-zero, R1-zero-Div enhances reasoning performance on 3 out of 4 benchmarks, achieving an average improvement of 2.3%, validating the scalability of our approach to a smaller model.  [35], DeepSeek-R1 [13], and Kimi-k1.5 [43]. Our work builds upon the R1-zero training method proposed by DeepSeek-R1 [13], which significantly improves LLM reasoning through two innovations that simplify the training pipeline and accelerate training: the GRPO algorithm [40], which replaces critic models with group score baselines, and a rule-based reward system that focuses solely on final answer correctness and output format.
Subsequent research has advanced this approach in two directions: (1) improving training data quality [28,33,15,54,21,18] and (2) refining RL algorithms. Regarding RL algorithm refinement, one category focuses on PPO-like methods. SimpleRL-zero [60] demonstrates that PPO with replacing the reward model by a rule-based reward function can significantly improve the LLM's reasoning ability. VinePPO [25] leverages the flexibility of language environments to compute unbiased Monte Carlobased estimates, eliminating the need for large value networks. VCPPO [56] employs a pretrained value model to address value initialization bias and decouples Generalized Advantage Estimation (GAE) computation between the actor and critic to mitigate reward signal decay. VAPO [55] further shows that value-based RL frameworks outperform value-free methods in long Chain-of-Thought reasoning. The second category focuses on GRPO enhancements for stability and speed. DAPO [54] identifies the critical shortcomings (entropy collapse, training instability, and biased loss) in the original GRPO algorithm and addresses them via decoupled clipping and dynamic sampling. Dr.GRPO [31] reveals two biases in GRPO: response-level length bias and question-level difficulty bias. SRPO [63] introduces a two-stage history-resampling method to improve training efficiency. Alternative approaches explore algorithms like REINFORCE: Kimi-k1.5 [43] demonstrates stable training with REINFORCE-like policy gradients, while REINFORCE++ [20] and GPG [5] aim to enhance REINFORCE's stability and scalability, respectively.
this section cite: ['b50', 'b34', 'b12', 'b42', 'b12', 'b39', 'b27', 'b32', 'b14', 'b53', 'b20', 'b17', 'b59', 'b24', 'b55', 'b54', 'b53', 'b30', 'b62', 'b42', 'b19', 'b4']

Section: Diversity in RL
Research on policy diversity in deep reinforcement learning can be categorized into three groups based on how diversity is utilized [48]. The first category uses diversity primarily to improve exploration efficiency, where diversity emerges as a byproduct of maximizing final task performance [19,11,36,8,37,53]. The second category treats diversity either as a constraint (optimizing quality subject to diversity constraints) or as an objective (optimizing diversity under quality constraints) [32,64,12,58,66]. The third category optimizes quality and diversity simultaneously, known as Quality-Diversity RL methods [6,38,45,3]. Our work extends the first paradigm to RL for LLM reasoning. While existing research in this category has proposed various diversity metrics, such as distance regularization between the current policy and a previous policy [19], reward randomization [41], we develop our approach based on a simple yet effective entropy-based diversity metric.
this section cite: ['b47', 'b18', 'b10', 'b35', 'b7', 'b36', 'b52', 'b31', 'b63', 'b11', 'b57', 'b65', 'b5', 'b37', 'b44', 'b2', 'b18', 'b40']

Section: Diversity in LLMs
Prior work has explored diversity in LLMs across several domains. GEM [29] proposes methods to preserve diversity during supervised fine-tuning, while Bstar [61] investigates the exploration-exploitation tradeoff in self-improvement settings. Additional studies have examined diversity in reinforcement learning from human feedback [34,26] and LLM ensembles [44]. However, diversity remains understudied in RL for LLM reasoning scenarios. To our knowledge, we are the first to formally analyze diversity and propose a principled diversity-aware training method for this setting.
this section cite: ['b28', 'b60', 'b33', 'b25', 'b43']

Section: Conclusion, limitations, and discussion
In this research, we investigate the role of diversity in RL for LLM reasoning. Through comprehensive evaluations across 12 LLMs, we empirically establish a strong positive correlation between a model's reasoning potential and the diversity of its generated solutions, underscoring the necessity of fostering diversity during RL training. To this end, we introduce a novel diversity-aware policy optimization method that optimizes the token-level diversity in positive samples. Experimentally, we demonstrate that our method not only enhances LLMs' reasoning ability but also generates more diverse solutions. By bridging the gap between diversity promotion and policy optimization, we aim to provide new insights for advancing the robustness and creativity of LLMs in complex reasoning scenarios.
Due to computational constraints, our experiments were conducted on 8×NVIDIA A6000 GPUs, which restricted our analysis to 1.5B and 7B parameter-scale models. This naturally introduces a limitation: the generalizability of our diversity-aware policy optimization method to larger-scale LLMs remains to be explored. While our method demonstrates significant improvements on midsized models, extrapolating these findings to larger architectures may require adjustments to the entropy regularization scheme or training dynamics, given the known differences in optimization landscapes across model scales. We urge future research to investigate these scalability challenges and hope our work will inspire the community to explore diversity-enhanced RL strategies for both small and large LLMs, fostering more robust reasoning capabilities across the spectrum of model architectures. Beyond scaling, the diversity-aware optimization mechanism could also be extended to other LLM tasks [65,50,23,22], suggesting its broader applicability beyond reasoning tasks.
Another promising future direction lies in the semantic definition of diversity. In this work, we employ entropy-based regularization to implicitly promote diverse behaviors during LLM generation, which captures statistical variance in output distributions. However, many real-world applications demand user-intended diversity (e.g., requiring both algebraic and arithmetic solutions to a math problem, or generating code with distinct algorithmic approaches). Such scenario-specific diversity requires explicit modeling of user-defined diversity, a challenge well-studied in RL [10,48]. By bridging LLM reasoning with explicit diversity optimization from RL, future work could unlock more controllable and context-aware generative capabilities, addressing the gap between statistical diversity and human-intentional variety in complex tasks.
this section cite: ['b64', 'b49', 'b22', 'b21', 'b9', 'b47']

Section: References
Ref_id:b0 Title: Understanding the impact of entropy on policy optimization Year: (2019)
Ref_id:b1 Title: Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models Year: (2025)
Ref_id:b2 Title: Stefanos Nikolaidis, and Gaurav Sukhatme. Proximal policy gradient arborescence for quality diversity reinforcement learning Year: (2023)
Ref_id:b3 Title: Incorrect baseline evaluations call into question recent llm-rl claims Year: (2025)
Ref_id:b4 Title: Gpg: A simple and strong reinforcement learning baseline for model reasoning Year: (2025)
Ref_id:b5 Title: Qd-rl: Efficient mixing of quality and diversity in reinforcement learning Year: (2006)
Ref_id:b6 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b7 Title: Improving exploration in evolution strategies for deep reinforcement learning via a population of novelty-seeking agents Year: (2018)
Ref_id:b8 Title: Process reinforcement through implicit rewards Year: (2025)
Ref_id:b9 Title: Quality diversity through human feedback: Towards open-ended diversity-driven optimization Year: (2023)
Ref_id:b10 Title: Diversity is all you need: Learning skills without a reward function Year: (2018)
Ref_id:b11 Title: Multiple plans are better than one: Diverse stochastic planning Year: (2021)
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b13 Title: Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems Year: (2024)
Ref_id:b14 Title: Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning Year: (2025)
Ref_id:b15 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b16 Title: Vishaal Udandarao, Samuel Albanie, Ameya Prabhu, and Matthias Bethge. A sober look at progress in language model reasoning: Pitfalls and paths to reproducibility Year: (2025)
Ref_id:b17 Title: Curatedthoughts: Data curation for rl training datasets Year: (2025)
Ref_id:b18 Title: Diversity-driven exploration strategy for deep reinforcement learning Year: (2018)
Ref_id:b19 Title: Reinforce++: A simple and efficient approach for aligning large language models Year: (2025)
Ref_id:b20 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (2025)
Ref_id:b21 Title: Evaluation of large language models as solution generators in complex optimization Year: (2025)
Ref_id:b22 Title: How multimodal integration boost the performance of llm for optimization: Case study on capacitated vehicle routing problems Year: (2025)
Ref_id:b23 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025-01)
Ref_id:b24 Title: Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment Year: (2024)
Ref_id:b25 Title: Understanding the effects of rlhf on llm generalisation and diversity Year: (2023)
Ref_id:b26 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b27 Title: Less is more for rl scaling Year: (2025)
Ref_id:b28 Title: Preserving diversity in supervised fine-tuning of large language models Year: (2025)
Ref_id:b29 Title: Let's verify step by step Year: (2023)
Ref_id:b30 Title: Understanding r1-zero-like training: A critical perspective Year: (2025)
Ref_id:b31 Title: Diversity-inducing policy gradient: Using maximum mean discrepancy to find a set of diverse policies Year: (2019)
Ref_id:b32 Title: Deepscaler: Holistic autoscaling for microservices based on spatiotemporal gnn with adaptive graph learning Year: (2023)
Ref_id:b33 Title: One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity Year: (2024)
Ref_id:b34 Title: Learning to reason with llms Year: (2024)
Ref_id:b35 Title: Effective diversity in population based reinforcement learning Year: (2020)
Ref_id:b36 Title: Non-local policy optimization via diversityregularized collaborative exploration Year: (2020)
Ref_id:b37 Title: Diversity policy gradient for sample efficient quality-diversity optimization Year: (2022)
Ref_id:b38 Title: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters Year: (2020)
Ref_id:b39 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b40 Title: Discovering diverse multi-agent strategic behavior via reward randomization Year: (2021)
Ref_id:b41 Title: Mathscale: Scaling instruction tuning for mathematical reasoning Year: (2024)
Ref_id:b42 Title: Kimi k1.5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b43 Title: Llm-topla: Efficient llm ensemble by maximising diversity Year: (2024)
Ref_id:b44 Title: Approximating gradients for differentiable quality diversity in reinforcement learning Year: (2022)
Ref_id:b45 Title: Trl: Transformer reinforcement learning Year: (2020)
Ref_id:b46 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2023)
Ref_id:b47 Title: Quality-similar diversity via population based reinforcement learning Year: (2023)
Ref_id:b48 Title: Progress or regress? self-improvement reversal in post-training Year: (2024)
Ref_id:b49 Title: Evolutionary computation in the era of large language model: Survey and roadmap Year: (2024)
Ref_id:b50 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b51 Title: Diverse policies recovering via pointwise mutual information weighted imitation learning Year: (2025)
Ref_id:b52 Title: Policy space diversity for non-transitive games Year: (2023)
Ref_id:b53 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b54 Title: Efficient and reliable reinforcement learning for advanced reasoning tasks Year: (2025)
Ref_id:b55 Title: What's behind ppo's collapse in long-cot? value optimization holds the secret Year: (2025)
Ref_id:b56 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b57 Title: Discovering diverse nearly optimal policies with successor features Year: (2021)
Ref_id:b58 Title: Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild Year: (2025)
Ref_id:b59 Title: 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient Year: (2025)
Ref_id:b60 Title: B-star: Monitoring and balancing exploration and exploitation in self-taught reasoners Year: (2024)
Ref_id:b61 Title: 100 days after deepseek-r1: A survey on replication studies and more directions for reasoning language models Year: (2025)
Ref_id:b62 Title: Srpo: A cross-domain implementation of large-scale reinforcement learning on llm Year: (2025)
Ref_id:b63 Title: Learning novel policies for tasks Year: (2019)
Ref_id:b64 Title: Hierarchical multiobjective model merging for pretrained models Year: (2024)
Ref_id:b65 Title: Continuously discovering novel strategies via reward-switching policy optimization Year: (2022)
Ref_id:b66 Title: Test-time reinforcement learning Year: (2025)
