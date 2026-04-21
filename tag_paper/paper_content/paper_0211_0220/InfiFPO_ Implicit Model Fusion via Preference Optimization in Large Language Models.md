Title: InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models
Abstract: Model fusion combines multiple Large Language Models (LLMs) with different strengths into a more powerful, integrated model through lightweight training methods. Existing works on model fusion focus primarily on supervised finetuning (SFT), leaving preference alignment (PA) -a critical phase for enhancing LLM performance-largely unexplored. The current few fusion methods on PA phase, like WRPO, simplify the process by utilizing only response outputs from source models while discarding their probability information. To address this limitation, we propose InfiFPO, a preference optimization method for implicit model fusion. InfiFPO replaces the reference model in Direct Preference Optimization (DPO) with a fused source model that synthesizes multi-source probabilities at the sequence level, circumventing complex vocabulary alignment challenges in previous works and meanwhile maintaining the probability information. By introducing probability clipping and max-margin fusion strategies, InfiFPO enables the pivot model to align with human preferences while effectively distilling knowledge from source models. Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods. When using Phi-4 as the pivot model, InfiFPO improve its average performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its capabilities in mathematics, coding, and reasoning tasks.

Section: Introduction
Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of natural language tasks. Yet, no single model is universally optimal-different LLMs often possess distinct strengths due to variations in architecture, pretraining data, and objectives. This has motivated a surge of interest in model fusion, which aims to integrate knowledge from multiple source models into a single pivot model to enhance its overall performance [1][2][3][4][5].
While existing work on model fusion has primarily focused on the supervised fine-tuning (SFT) phase, little attention has been paid to integrating fusion techniques into the preference alignment phase, a critical step in reinforcement learning from human feedback (RLHF) pipelines that substantially boosts performance and usability. Applying model fusion to this phase is non-trivial due to the discrete nature of preference data and the challenge of aligning both preferred and dispreferred outputs across heterogeneous models.
The closest prior work, WRPO [6], attempts to bridge this gap by incorporating high-quality source model responses as additional preference signals. However, WRPO suffers from two major limitations: 1) it discards the probabilistic outputs of source models and only uses response-level supervision; and 2) it focuses solely on preferred responses, missing valuable contrastive signals for dispreferred ones. As a result, WRPO only partially leverages the capabilities of the source models. The broader question-how to systematically fuse model knowledge during preference alignment-remains largely unanswered.
To overcome these limitations, we propose InfiFPO, a principled and efficient framework for performing model fusion during the preference alignment phase. Our key insight is that the reference model in preference optimization (e.g., in DPO) can be replaced with a fused source model, thereby enabling the pivot model to learn not only from preference data but also from the probabilistic behaviors of multiple source models. Unlike WRPO, InfiFPO utilizes sequence probability from all source models-including for both preferred and dispreferred responses-making the fusion process more principled and information-rich. We call this fusion that utilizes sequence-level probabilities as implicit model fusion, distinguishing it from previous work on token-level model fusion.
To instantiate InfiFPO efficiently, we derive it from an RLHF-style constrained optimization framework called FuseRLHF, which encourages the pivot model to maximize preference rewards while remaining close-in sequence-level KL divergence-to each source model. We further transform this constrained RL problem into a fully offline optimization objective that avoids expensive online sampling and reward model training.
To improve the robustness and effectiveness of InfiFPO, we introduce three key enhancements: (1) Length Normalization, which reduces bias arising from varying token sequence lengths across models; (2) Probability Clipping, which limits the influence of underperforming source models by suppressing noisy gradients; and (3) Max-Margin Fusion, which adaptively prioritizes source models that offer the most distinctive and informative deviations from the pivot model.
We conducted a comprehensive evaluation of InfiFPO using Phi-4 [7] as the pivot model and selecting five mainstream open-source LLMs with parameters ranging from 9B to 24B as source models. Our experiments spanned 11 widely-used benchmarks covering diverse tasks, including mathematics, coding, instruction following, and so on. Results demonstrate that InfiFPO consistently outperforms existing model fusion and preference optimization methods, improving Phi-4's average performance from 79.95 to 83.33 across 11 benchmarks. Furthermore, InfiFPO exhibits great versatility, effectively combining with various preference optimization objectives to further enhance performance. In summary, our contributions are threefold: ❶ We propose InfiFPO, a novel preference optimization framework that performs implicit model fusion by replacing the reference model in DPO with a fused source model, derived via an offline relaxation of a sequence-KL constrained RLHF objective. ❷ We introduce three stability-enhancing strategies-length normalization, probability clipping, and max-margin fusion-to avoid degradation from tokenization mismatch and probability inconsistencies in source models. ❸ We conduct extensive experiments on 11 preference benchmarks, demonstrating that InfiFPO consistently and significantly outperforms existing model fusion and preference optimization baselines, verifying its effectiveness for implict model fusionfoot_0 .
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']

Section: Preliminaries
In this section we briefly revisit the two foundations of our work: Model Fusion [1,2,4] and Direct Preference Optimization (DPO) [8].
this section cite: ['b0', 'b1', 'b3', 'b7']

Section: Model Fusion:
The goal of model fusion is to integrate knowledge from source models with different architectures, parameter sizes, and training data into one pivot model to enhance the pivot model's capabilities. Given N source models {M s i } N i=1 and a pivot model (also called target model) M p , the model fusion can be cast as a KL-constrained optimization problem: arg max
M p E x,y∼D log M p (y|x) s.t. E x,y∼D D TKL [M s i (y|x)||M p (y|x)] ≤ ε, ∀i ∈ {1, ..., N },(1)
where each sample in the dataset D contains a prompt sequence x and its corresponding response sequence y. D TKL indicates token-level KL divergence. Each constraint here keeps the pivot model within an ε-ball of every source model, thereby fusing their behaviors. While effective, Eq. ( 1) suffers from vocabulary conflict: the source models often employ incompatible tokenisers, forcing previous work to rely on heuristic vocabulary matching.
this section cite: []

Section: Direct Preference Optimization: DPO [8]
is an offline replacement for Reinforcement Learning from Human Feedback (RLHF). The objective of RLHF can be formalized as arg max
M θ E x∼D,y∼M θ (y|x) [r(x, y)] -βD SKL M θ (y|x)∥M ref (y|x) .(2)
where r is a reward model that evaluates how good the response y generated by policy model M θ is. D SKL indicates sequence-level KL divergence 3 . Usually, the base reference model M ref is the initial M θ , and β is a parameter controlling the deviation from M ref .
After deriving Eq. ( 2), the relationship between r and the optimal policy M θ * is as follows
r(x, y) = β log M θ * (y|x) M ref (y|x) + β log Z(x).(3)
where
Z(x) = y M ref (y|x)) exp 1 β r(x, y)) .
Since the reference model M ref remains unchanged during training, the optimal policy model is theoretically determined only by the reward model r. A natural idea is to indirectly train the optimal policy model through training the reward model, which is the optimization objective of DPO:
L DPO (M θ ; M ref ) = -E (x,y w ,y l )∼D p log σ β log M θ (y w |x) M ref (y w |x) -β log M θ (y l |x) M ref (y l |x) .(4)
where each sample in the preference dataset D p contains a prompt x, a preferred response y w , and a dispreferred response y l .
this section cite: []

Section: InfiFPO: Preference Optimization for Model Fusion
This section presents InfiFPO, our novel approach to preference optimization for Model Fusion. We begin by introducing the FuseRLHF objective ( § 3.1), which integrates model fusion with RLHF.
Given the inherent complexity of reinforcement learning frameworks, direct implementation proves challenging. Consequently, we propose an efficient offline InfiFPO methodology and develop three performance-enhancing strategies ( § 3.2). Finally, we provide gradient analysis to further understand the optimization mechanism of InfiFPO ( § 3.3)
this section cite: []

Section: FuseRLHF: RLHF for Implicit Model Fusion
Constrained objective. We consider model fusion during the preference alignment phase to be a constrained optimization problem. Based on Eq. ( 1) and ( 2), we can obtain the objective of FuseRLHF:
arg max
M p E x,y∼D r(x, y) s.t. E x,y∼D D SKL [M s i (y|x)||M p (y|x)] ≤ ε, ∀i ∈ {1, ..., N },(5)
where the initial pivot model is included in the source models. Each constraint keeps the pivot policy within an ε-ball (in KL divergence) of every teacher, thereby fusing their behaviour while still allowing preference alignment through the reward r.
Please note that we use KL constraints at the sequence-level instead of the token-level. As shown in Eq. ( 1), previous works on model fusion used token-level KL and faced the vocabulary conflict
-0.2 0 0.2 0.4 0.6 0.8 1 1 2 3 4 5 weight index -0.2 0 0.2 0.4 0.6 0.8 1 1 2 3 4 5 weight index Max-margin fusion -0.2 0 0.2 0.4 0.6 0.8 1 1 2 3 4 5 weight index -0.2 0 0.2 0.4 0.6 0.8 1 1 2 3 4 5 weight index Max-margin fusion Pivot Model 0.2 0.3 0.4 0.9 0.7 0.6 0.5 ... ... Length Norm Probability Clipping Source Models Optimization Objective Sequence Probability Pairs Normalized Probability Pairs Fusion from pivot model from source models 0.3 0.2 0.1 0.2 0.3 … 1 2 3 4 5 ... ... ... ... ... ... ... ... ... ... … … 1 2 3 4 5 different lengths and dispreferred (y l ) responses using both pivot and source models. Following length normalization and probability clipping, we identify the source model with the maximum normalized probability difference from the pivot model for fusion and preference alignment.
issue, where heterogeneous models have different vocabularies, making their token-level output distributions incompatible for divergence calculation. To solve this issue, those works introduce complex vocabulary matching processes and only calculate KL divergence on the matching portions. In contrast, our sequence-level KL constraints avoid this issue while preserving the source models' probability information, learning source models' preferences for the whole sentence. We call this fusion that utilizes sequence-level probabilities as implicit model fusion (IMF), distinguishing it from previous work on token-level model fusion.
Unconstrained objective. Introducing non-negative multipliers {γ i } N i=1 and relaxing the constraints in Eq. ( 5) yields the following unconstrained objective: arg max
M p E x∼D,y∼M p (y|x) [r(x, y)] -β N i γ i D SKL [M p (y|x)∥M s i (y|x)] ,(6)
where γ i ≥ 0 and N i=1 γ i = 1. By setting γ i to different values, various fusion strategies can be adopted. Specifically, when N = 1 and taking the source model as the initial pivot model, Eq. ( 14) reduces to classical RLHF.
this section cite: []

Section: InfiFPO: Efficient Preference Optimization for IMF
Deriving the FPO objective. Directly training the pivot model with FuseRLHF requires significant time and resources. On one hand, it requires training an additional reward model; on the other hand, during RL, the pivot model needs to sample responses online, while all source models also need to calculate sequence probabilities simultaneously. To reduce these costs, we follow Rafailov et al. [8] by converting online FuseRLHF to offline infiFPO, improving training efficiency. Following prior works, it is clear to show that the optimal pivot model M p * to the FuseRLHF objective in Eq. ( 14) takes the form:
M p * (y|x) = 1 Z(x) M s fu (y|x) exp( 1 β r(x, y)).(7)
where a fused source model M s fu (y|x) = N i=1 (M s i (y|x)) γi and a partition function Z(x) = y M s fu (y|x) exp 1 β r(x, y) . See Appendix A.1 for a complete derivation. Then we can rearrange Eq. (7) to show the relationship between the reward model and the optimal pivot model as: r(x, y) = β log M p * (y|x) M s fu (y|x)
+ β log Z(x).(8)
We can see that theoretically the optimal pivot model M p * only depends on the reward model r, since the source models remain unchanged during training. Therefore, we can derive the optimization objective of FPO as follows:
L FPO (M p ; {M s i } N i=1 ) = -E (x,y w ,y l )∼D p log σ β log M p (y w |x) M s fu (y w |x) -β log M p (y l |x) M s fu (y l |x) .(9)
Especially, when N = 1 and using the initial pivot model as the source model, this loss function reduces to the original DPO, i.e., Eq. ( 4). While the FPO objective is simple and theoretically supported, it faces length normalization and source model degradation issues in model fusion. Besides, the fusion multipliers {γ i } N i=1 for source models requires a strategy for determination. To address these issues, we propose the following three strategies.
Length Normalization. Models with larger vocabularies tend to produce longer segmentations whose log-likelihood sums are lower, even when the semantic adequacy of the response is unchanged. Similar effects have been presented by Wu et al. [9]. To address the length bias issue, we introduce the length normalization, which divides the sequence log probability from a model by the length of the sequence after tokenization with that model.
log M (•) (y|x) = 1 |y| log M (•) (y|x) (10
)
where M (•) can be either a source model or a pivot model.
Probability Clipping. Source models may occasionally assign lower probability to the preferred response y w (or higher probability to the dispreferred response y l ) than the pivot model, injecting misleading gradients and causing instability. To avoid the source model degradation issue, we clip the sequence probabilities of the source model. Specifically, for y w , we define the minimum probability of the source model as the sequence probability of the initial pivot model M p init ; for y l , we define the maximum sequence probability of the source model as the sequence probability of the initial pivot model.
Clip(M s i (y|x)) = max(M s i (y|x), M p init (y|x)), if y is y w , min(M s i (y|x), M p init (y|x)), else.(11)
This piecewise definition is weakly monotone: for all t 1 ≤ t 2 we have Clip(t 1 ) ≤ Clip(t 2 ), so the logarithm that follows preserves (or ties) the order of probabilities and cannot invert preferences.
this section cite: ['b7', 'b6', 'b8']

Section: Max-margin Fusion.
To determine the multipliers {γ i } N i=1 for source models, we propose a maxmargin fusion strategy that maximizes the diversity of information incorporated into the pivot model. Specifically, we select the source model that differs most from the current pivot model, as this model likely contains the most unique and complementary information.
γ i = 1, if i = arg max j D(M s j , M p ), 0, otherwise.(12)
where D(M s j , M p ) = abs(M s j (y|x) -M p (y|x)) measures the probability difference between a source model and the pivot model. This winner-takes-all approach simplifies training while effectively capturing diverse capabilities across source models. In §4.3, we also evaluate alternative fusion strategies, including averaging and dynamic weighting methods.
this section cite: []

Section: In Summary.
Combining the strategies listed above, we have the final InfiFPO objective:
L InfiFPO (M p ; {M s i } N i=1 ) = -E (x,y w ,y l )∼D p log σ βlog M p (y w |x) M sclip fu (y w |x) -βlog M p (y l |x) M sclip fu (y l |x) , (13
) where M sclip fu (y|x) = N i=1 Clip(M s i (y|x)) γi .
this section cite: []

Section: Gradient Analysis.
To gain deeper insights into InfiFPO's optimization dynamics, we analyze the gradient of the loss function with respect to the pivot model parameters θ. This analysis elucidates how information from source models influences the pivot model's learning trajectory. The gradient can be expressed as:
∇θL InfiFPO = -βE (x,y w ,y l )∼D p   σ βlogRs -βlogRp preference disparity coefficient ∇θlogM p (y w |x) increase likelihood of y w -∇θlogM p (y l |x) decrease likelihood of y l    .
where R s = M sclip fu (y w |x) M sclip fu (y l |x) and R p = M p (y w |x) M p (y l |x) . Analogous to DPO, the InfiFPO gradient increases the likelihood of preferred responses y w while decreasing that of dispreferred responses y l . However, the critical distinction lies in the preference disparity coefficient, which weights training samples based on the divergence between the source and pivot models' preference assessments. This coefficient becomes larger when source models exhibit a stronger preference between y w and y l than the pivot model does. Consequently, samples where the source models strongly differentiate between responses but the pivot model does not yet reflect this distinction receive greater optimization emphasis. This adaptive weighting mechanism efficiently transfers preference knowledge from source models to the pivot model. The full derivation of the gradient is in Appendix A.2.
this section cite: []

Section: Experiments
We evaluated InfiFPO using Phi-4 as the pivot model and five mainstream open-source LLMs (9B∼24B parameters) as source models across 11 diverse benchmarks. Results show InfiFPO outperforms existing fusion and preference optimization methods.
this section cite: []

Section: Setup
Model. We use Phi-4 [7] as the pivot model. The source models consist of three general-purpose models (Qwen2.5-14B-Instruct [10], Mistral-Small-24B-Instructfoot_2 , and Gemma-3-12B-Instruct [11]) and two domain-specific models (Qwen2.5-Coder-14B-Instruct [12] and Qwen2.5-Math-7B-Instruct [13]). By including these domain-specific models as additional source models, we can also investigate whether specialized expertise can enhance the overall performance of general models 5 . Dataset. We constructed a new training dataset comprising 150k examples across mathematics, coding, and general tasks. Data sources include Infinity-Instruct [14], NuminaMath-1.5 [15], and KodCode-V1-SFT [16], with detailed statistics provided in Table 1. Since the original dataset may contain responses from outdated LLMs, potentially less capable than our pivot model, we retained only the prompts to build a new preference dataset. For each prompt, we generated responses using multiple models: 4 from each source model and 8 from the pivot model, all with a sampling temperature of 0.8. We then employed a reward modelfoot_4 [17] to evaluate these responses, selecting the highest-scored response as y w and the lowest-scored as y l .
this section cite: ['b6', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16']

Section: Types

this section cite: []

Section: General Data Math Data Code Data

this section cite: []

Section: Dataset Infinity-Instruct
Training Detail. Our training process involved two stages with a batch size of 128 and a maximum sequence length of 4,096 tokens, using 16 NVIDIA A800-80GB GPUs. We implemented a cosine learning rate schedule with a 10% warmup ratio. In the first stage, we performed SFT on half of our dataset with y w for 3 epochs, using a learning rate of 1e-6 to build the SFT model. This model then served as the foundation for the second stage, where we conducted Preference Optimization on the remaining half of the data for a single epoch, with a learning rate of 1e-7 and β = 2.5.
Evaluation. We conduct a comprehensive evaluation across 11 diverse benchmarks to assess the model's capabilities. Our evaluation spans five critical dimensions: (1) General reasoning (BBH [18], ARC-C [19], MMLU [20]), ( 2) Math (GSM8K [21], MATH [22], TheoremQA [23]), ( 3) Code (MBPP [24], HumanEval [25]), (4) Text reasoning (DROP [26], HellaSwag [27]), and (5) Instruction following (IFEval [28]). This multifaceted evaluation strategy enables us to systematically analyze the model's strengths and limitations across a spectrum of tasks requiring different cognitive abilities.
More evaluation details are listed in Appendix B.
Baseline. We compare InfiFPO with three categories of baseness, including pivot &source model, model fusion, and preference optimization methods. For model fusion methods, we include FuseLLM [1], FuseChat [2], and InfiFusion [4]. Limited by the complex vocabulary alignment and distribution merging process, we only include Qwen2.5-Instruct, Qwen2.5-Coder, and Mistral-Small as source models. For fair comparison, we select the same source models to implement InfiFPO (marked with asterisks). For preference optimization methods, we include DPO [8], IPO [29], and WRPO [6]. All these preference optimization methods adopt the same two-stage training approach as InfiFPO, i.e., first using half of the data for SFT, then using the remaining half for preference optimization.
this section cite: ['b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b0', 'b1', 'b3', 'b7', 'b28', 'b5']

Section: Main Results
Table 2 presents comprehensive evaluation results across 11 benchmarks for different model types.
From these results, we can draw several findings about InfiFPO.
InfiFPO effectively integrates the capabilities of the source models. InfiFPO significantly improved the pivot model's average performance across 11 benchmarks from 79.95 to 83.33. This integration is particularly remarkable as InfiFPO manages to inherit specialized strengths from diverse source models, such as mathematical reasoning from Qwen2.5-Math and code generation from Qwen2.5-Coder, while avoiding their respective weaknesses. For instance, while Qwen2.5-Math excels on GSM8K and MATH (92.27 and 81.70) but performs poorly on other tasks, and Qwen2.5-Coder achieves top scores on MBPP and HumanEval (85.40 and 90.90) but underperforms on theorem questions, InfiFPO maintains balanced high performance across these diverse task categories.
InfiFPO outperforms model fusion baselines in both efficiency and effectiveness. Compared to InfiFusion*, the best-performing baseline on average, InfiFPO* shows an average improvement of 0.36 across 11 benchmarks. The improvements are particularly significant for instruction-following and code tasks, with a 4.4 improvement on IFEval and a 2.4 improvement on HumanEval. More importantly, InfiFPO* requires only 34% of the GPU hours compared to InfiFusion* (55 vs 160). This is thanks to our implicit model fusion objective, which replaces token-level KL with sequence-level KL. This strategy preserves probability information while avoiding complex vocabulary alignment processes, making InfiFPO more efficient and scalable.
InfiFPO consistently outperforms preference optimization baselines. After training on half of the data's preferred responses y w , SFT improved by 1.62 on average compared to the original model. Then, using the remaining half of the data for preference optimization, InfiFPO outperformed all preference optimization baselines. Compared to SFT-WRPO (the best-performing baseline), InfiFPO showed an average improvement of 0.53 across 11 benchmarks while using about the same amount of GPU hours (58 vs 57). This demonstrates that our InfiFPO can better incorporate source model knowledge to enhance performance without significantly increasing training time compared to existing preference optimization baselines.
this section cite: []

Section: Analysis
We conduct two analyses:
this section cite: []

Section: Different Fusion Strategies.
In §3.2, we proposed the maxmarginal fusion strategy to extract unique information from source models. Additionally, we explored two other fusion strategies: an average-based strategy and a confidence-based strategy. For the average-based strategy, we treat all source models equally, assigning each a weight of 1/N . For the confidence-based strategy, we weight source models according to their confidence in the response, measured by sequence probability, meaning models with higher confidence receive greater weights. Figure 2 illustrates the performance of InfiFPO under different fusion strategies. We observe that the Confidence-based strategy slightly outperforms the Average-based strategy, likely because it emphasizes high-confidence models during fusion, thus acquiring better information. The Maxmarginal strategy consistently outperforms both the Average-based and Confidence-based strategies, as it focuses on the most distinctive models, thereby learning more unique and complementary information. Detailed fusion strategies can be found in Appendix D.
this section cite: []

Section: Albation Study

this section cite: []

Section: Length Normalization and Probability Clipping.

this section cite: []

Section: Number of Source Models.
We conducted experiments on InfiFPO with an increasing number of source models, adding different models one by one for fusion. The experiment started with a single source model, Qwen2.5-Instruct, then gradually added Mistral-Small, Qwen2.5-Coder, Gemma-3-Instruct, and Qwen2.5-Math. Table 5 shows clear performance improvements as the number of source models increases. With each additional model, we observe consistent gains across all metrics, particularly in Code tasks (+3.2 points with 5 models). The diminishing returns after 4-5 models suggest a balance between performance benefits and computational resources when selecting source models for fusion.
this section cite: []

Section: Related Work
Preference Optimization. Aligning LLMs with human preferences often relies on RLHF [30,31], where a reward model is trained from human comparisons and used in PPO to optimize the policy. While effective, RLHF is resource-intensive and complex to converge. To simplify preference alignment, DPO [8] reformulates the objective as offline learning from preference pairs, removing the need for a reward model and RL. Azar et al. [29] unify RLHF and DPO under a general preference optimization framework, and propose IPO, a variant that avoids overfitting by using the identity transformation, offering more stable learning in low-data or biased settings. Gu et al. [32] further propose the PAD framework, which models preference knowledge as a probability distribution over responses, providing more nuanced supervisory signals for preferences distilling.
this section cite: ['b29', 'b30', 'b7', 'b28', 'b31']

Section: Model Fusion.
Integrating models aims to combine multiple LLMs into a single model that inherits their respective strengths. While earlier methods like model merging [33][34][35][36][37][38] require architectural compatibility, fusion techniques relax such constraints, making them more suitable for heterogeneous models. Another line of work, knowledge distillation (KD) [39][40][41], offers an alternative approach for transferring capabilities across models without requiring structural homogeneity. KD transfers the generalization ability of larger models to smaller ones by leveraging "soft targets" the output probability distributions (logits) of the teacher model enabling efficient deployment while preserving performance. However, traditional KD typically assumes that the teacher model is larger than the student and fully covers the target capability space. Moreover, most prior work has focused on the single teacher setting [42,43], limiting its flexibility in multi-teacher scenarios.
FuseLLM [2] introduced LLM fusion, showing gains in reasoning and code tasks. Later works extended this idea across domains [44], but struggled with structural mismatches. To address this, pairwise fusion [2] sequentially integrates models into the pivot model, and InfiFusion [4] improved it via adaptive merging and unified output aggregation. The most explicit fusion approaches operate at the token level and face vocabulary alignment issues. WRPO [6] mitigates this by shifting fusion to the sequence level using reinforcement learning, enabling implicit fusion without vocabulary conflicts. However, WRPO has two major limitations: it discards probabilistic outputs using only response-level supervision and focuses solely on preferred responses while ignoring contrastive signals from dispreferred ones, resulting in only partial leverage of source model capabilities.
this section cite: ['b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b1', 'b43', 'b1', 'b3', 'b5']

Section: Conclusion
We introduce InfiFPO, a novel framework that enables model fusion during the preference alignment phase by replacing the reference model in DPO with a fused sequence-level distribution over multiple source models. Unlike prior work, InfiFPO leverages full-sequence probabilities rather than tokenlevel outputs, thereby avoiding vocabulary alignment issues across heterogeneous models while preserving rich preference signals. To make this optimization practical and efficient, we derive an offline training objective from a sequence-KL constrained RLHF formulation, termed FuseRLHF.
We further enhance stability through three key strategies: length normalization to mitigate sequence bias, probability clipping to suppress noisy gradients, and max-margin fusion to prioritize diverse and informative sources. Experiments across 11 benchmarks demonstrate that InfiFPO consistently outperforms strong baselines in both model capability and alignment quality. Our results highlight that preference optimization not only accommodates but can significantly benefit from principled model fusion, offering a robust and scalable path to integrating diverse LLMs.
this section cite: []

Section: Limitation
Despite InfiFPO demonstrating significant empirical performance, it still relies on existing preference optimization methods such as DPO. More rigorous theoretical analysis is needed to better understand InfiFPO's fusion mechanisms and further strengthen its theoretical foundation. Additionally, due to computational resource constraints, we only selected five mainstream open-source LLMs as source models for our experiments, which cannot represent the SOTA performance of current advanced LLMs. Experiments with larger-scale models and datasets remain unexplored.
this section cite: []

Section: A Mathematical Derivations

this section cite: []

Section: A.1 Deriving the FPO Objective
In this appendix, we will derive Eq. ( 7). Based on Eq. ( 14), we optimize the following objective:
arg max
M p E x∼D,y∼M p (y|x) [r(x, y)] -β N i γ i log M p (y|x) -log M s i (y|x) ,(14)
then we have:
arg max M p E x∼D,y∼M p (y|x) [r(x, y)] -β N i γ i D SKL [M p (y|x)∥M s i (y|x)] = arg max M p E x∼D E y∼M p (y|x) r(x, y) -β log M p (y|x) M s fu (y|x) = arg min M p E x∼D E y∼M p (y|x) log M p (y|x) M s fu (y|x) - 1 β r(x, y) = arg min M p E x∼D E y∼M p (y|x) log M p (y|x) 1 Z(x) M s fu (y|x) exp( 1 β r(x, y)) -log Z(x) (15
)
where the fused source model M s fu (y|x) = N i=1 (M s i (y|x)) γi and the partition function Z(x) = y M s fu (y|x) exp 1 β r(x, y) . Note that the partition function is a function of only x and the fused source model M s fu , but does not depend on the pivot model M p . Hence we have the optimal solution M p * :
M p * (y|x) = 1 Z(x)
M s fu (y|x) exp(
1 β r(x, y)).(16)
This completes the derivation.
this section cite: []

Section: A.2 Deriving the Gradient of the InfiFPO Objective
In this section, we derive the gradient of the InfiFPO objective, where the parameters of the pivot model are denoted as θ:
∇θL InfiFPO (M p ; {M s i } N i=1 ) = -∇θE (x,y w ,y l )∼D p log σ βlog M p (y w |x) M sclip fu (y w |x) -βlog M p (y l |x) M sclip fu (y l |x) ,(17)
where
M sclip fu (y|x) = N i=1 Clip(M s i (y|x)) γi .
We can rewrite Eq. ( 17) as
∇θL InfiFPO (M p ; {M s i } N i=1 ) = -∇θE (x,y w ,y l )∼D p log σ σ ′ (u) σ(u) ∇θ(u) ,(18)
where
u = βlog M p (y w |x) M sclip fu (y w |x) -βlog M p (y l |x) M sclip fu (y l |x) . With the properties of sigmoid function σ ′ (x) = σ(x)(1 -σ(x))
and σ(x) = 1 -σ(-x), we can get the gradient:
∇θL InfiFPO = -βE (x,y w ,y l )∼D p   σ βlog M p (y l |x) M sclip fu (y l |x) -βlog M p (y w |x) M sclip fu (y w |x) ∇θ log M p (y w |x)-∇θ log M p (y l |x)   .
After rewriting the first term, we obtain the final form of the gradient in § 3.3.
this section cite: []

Section: B Evaluation Setup
We adopt OpenCompass [45] and EvalPlus [46] to conduct evaluation on 11 benchmark datasets. We revise the prompts of certain datasets within OpenCompass to ensure more reliable answer extraction via its regex-based matching mechanism. The detailed prompts are listed in Table 6.
For TheremQA and HumanEval, we follow the default evaluation settings without modifying the prompts. For MBPP, we employ EvalPlus [46] for rigourous evaluation of LLM-synthesized code.
this section cite: ['b44', 'b45', 'b45']

Section: C Adaptation to Other Preference Optimization Objectives
The original objective of WRPO is
L WRPO (M θ ; M ref ) = -E (x,y ws ,y wp ,y l )∼D log σ α • β log M θ (y ws |x) M ref (y ws |x) + (1 -α) • β log M θ (y wp |x) M ref (y wp |x) -β log M θ (y l |x) M ref (y l |x) ,(19)
where y ws is the preferred response generated by source models, and y wp is the dispreferred response generated by the pivot model. α represents the fusion coefficient that dynamically balances the reward of y ws and y wp . After adapting InfiFPO to this objective, we can obtain The original objective of the IPO is
L InfiFPO-WRPO (M p ; {M s i } N i=1 }) = -E (x,y ws ,y wp ,y l )∼D logσ α • βlog M θ (y ws |x) M sclip fu (y ws |x) + (1 -α) • βlog M θ (y wp |x) M sclip fu (y wp |x) -βlog M θ (y l |x) M sclip fu (y l |x) .(20)
L IPO (M θ ; M ref ) = -E (x,y w ,y l )∼D p βlog M θ (y w |x) M ref (y w |x) -βlog M θ (y l |x) M ref (y l |x) - 1 2β 2 . (21
)
Since IPO already implements length normalization 7 , the main changes when adapting to InfiFPO are in the reference model part.
L InfiFPO-IPO (M p ; {M s i } N i=1 ) = -E (x,y w ,y l )∼D p log M p (y w |x) M sclip fu (y w |x) -log M p (y l |x) M sclip fu (y l |x) - 1 2β 2(22)
this section cite: []

Section: D Different Fusion Strategies
When we focus on the fused source model M s fu , which is the weighted geometric mean of all source model sequence probabilities, we can find that it originates from the model fusion objective. By changing its weights, we can adopt different fusion strategies. In addition to the Max-margin fusion strategy mentioned in § 3.2, we design two following strategies. Their results can be viewed in § 4.3.
this section cite: []

Section: Average-based Fusion.
A common fusion strategy is to treat N source models equally, adopting the same weight of 1  N , thus the fusion model becomes
M s fu = N i=1 (M s i (y|x)) 1 N (23
)
Confidence-based Fusion. To dynamically weight the source models, we introduce a confidencebased strategy and define the confidence of each source model M s i as the inverse of its negative log-likelihood:
Confidence i (y|x) = 1 -log M s i (y|x) + ϵ , (24
)
the entire vocabulary ( 100k tokens per token postion), requiring 100k-scale floating-point numbers per token that scale linearly with source model count.
this section cite: []

Section: E.2.2 In-depth Analysis of Method Principles
We provide a detailed discussion of how source models are selected during Probability Clipping (PC) and Max-Margin fusion strategies.
Probability Clipping and Max-Margin Process: PC aims to avoid interference from weak source models by clipping the corresponding log probabilities, ensuring that source models do not assign lower probabilities to preferred responses or higher probabilities to dispreferred responses than the pivot model. Max-Margin selects the source model with the most different preference knowledge from the pivot model as the reference model.
When using PC, Max-Margin selects the "smartest" source model, meaning the source model that assigns probabilities to preferred responses no lower than the pivot model, probabilities to dispreferred responses no higher than the pivot model, and has the largest margin between preferred and dispreferred responses.
Example Illustration: Given two responses, the probability values assigned by models are: Table 14: Example of model probabilities after PC Based on the Max-Margin strategy, source model 2 serves as the reference model. It is the "smartest" source model, having the highest log probabilities for preferred responses and the lowest log probabilities for dispreferred responses.
This strategy aligns with findings in FuseLLM, which adopts the MinCE strategy to select source models with the lowest cross-entropy for ground truth. Both approaches indicate that in model fusion, fusing knowledge from multiple source models may not be as effective as fusing knowledge from the "best" source model.
this section cite: []

Section: F Case Study

this section cite: []

Section: F.1 Advantages in Code Generation
To evaluate the effectiveness of our proposed InfiFPO strategy, we conduct a detailed comparison with the baseline Supervised Fine-Tuning (SFT) model on the HumanEval benchmark.
Through case studies, we observe that InfiFPO consistently exhibits superior capabilities in two critical aspects: (1) precise task understanding and (2) concise and robust code generation. For instance, in the remove_duplicates task, SFT misconstrues the requirement by retaining the first occurrence of duplicates, leading to incorrect outputs. In contrast, InfiFPO accurately identifies the need to eliminate all repeated elements, employing a frequency-counting approach that aligns with the task's semantics. Similarly, in the vowels_count task, InfiFPO adopts a minimal yet effective implementation
OpenAi HumanEval -Task Type: Code Generation InfiFPO Behavior Baseline Behavior "prompt_1": "from typing import List\n\n\ndef remove_duplicates(numbers: List[int]) ->List[int]:\n \"\"\" From a list of integers, remove all elements that occur more than once.\n Keep order of elements left the same as in the input.\n >>>remove_duplicates([1, 2, 3, 2, 4])\n [1, 3, 4]\n \"\"\"\n" "completion": "def remove_duplicates(numbers: List[int]) ->List[int]:\n \"\"\" From a list of integers, remove all elements that occur more than once.\n Keep order of elements left the same as in the input.\n >>>remove_duplicates([1, 2, 3, 2, 4])\n [1, 3, 4]\n \"\"\"\n counts = {}\n for number in numbers:\n counts[number] = counts.get(number, 0) + 1\n \n result = []\n for number in numbers:\n if counts[number] == 1:\n result.append(number)\n \n return result\n","result": "passed","passed": true "completion": "from typing import List\n\ndef remove_duplicates(numbers: List[int]) ->List[int]:\n \"\"\" From a list of integers, remove all elements that occur more than once.\n Keep order of elements left the same as in the input.\n >>>remove_duplicates([1, 2, 3, 2, 4])\n [1, 3, 4]\n \"\"\"\n seen = set()\n duplicates = set()\n result = []\n\n for number in numbers:\n if number in duplicates:\n continue\n if number in seen:\n duplicates.add(number)\n else:\n seen.add(number)\n result.
append(number)\n\n return result\n","result": "failed: ","passed": false "prompt_2": "\nFIX = \"\"\"\nAdd more test cases.\n\"\"\"\n\ndef vowels_count(s):\n \"\"\"Write a function vowels_count which takes a string representing\n a word as input and returns the number of vowels in the string.\n Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a\n vowel, but only when it is at the end of the given word.\n\n Example:\n >>>vowels_count(\"abcde\")\n 2\n >>>vowels_count(\"ACEDY\")\n 3\n \"\"\"\n"
"completion": "def vowels_count(s):\n \"\"\"Write a function vowels_count which takes a string representing\n a word as input and returns the number of vowels in the string.\n Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a\n vowel, but only when it is at the end of the given word.\n\n Example:\n >>>vowels_count(\"abcde\")\n 2\n >>>vowels_count(\"ACEDY\")\n 3\n \"\"\"\n vowels = \"aeiouAEIOU\"\n count = 0\n for char in s:\n if char in vowels:\n count += 1\n if s[-1].lower() == 'y':\n count += 1\n return count\n", "result": "passed", "passed": true "completion": "def vowels_count(s):\n \"\"\"Write a function vowels_count which takes a string representing\n a word as input and returns the number of vowels in the string.\n Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a\n vowel, but only when it is at the end of the given word.\n\n Example:\n >>>vowels_count(\"abcde\")\n 2\n >>>vowels_count(\"ACEDY\")\n 3\n \"\"\"\n vowels = \"aeiouAEIOU\"\n count = 0\n for i, char in enumerate(s):\n if char in vowels:\n count += 1\n elif char == 'y' and i == len(s) -1:\n count += 1\n return count\n\n# Additional test cases\nprint(vowels_count(\"hello\")) # 2\nprint(vowels_count(\"sky\")) # 1\nprint(vowels_count(\"rhythm\")) # 0\nprint(vowels_count(\"fly\")) # 1\nprint(vowels_count(\"boy\")) # 2\nprint(vowels_count(\"try\")) # 1\nprint(vowels_count(\"my\")) # 1\nprint(vowels_count(\"y\")) # 1\nprint(vowels_count(\"a\")) # 1\nprint(vowels_count(\"e\")) # 1\nprint(vowels_count(\"i\")) # 1\nprint(vowels_count(\"o\")) # 1\nprint(vowels_count(\"u\")) # 1\n", "result": "failed: Test 5","passed": false, Table 16: Comparison between SFT and InfiFPO on TheoremQA benchmark. InfiFPO shows stronger symbolic reasoning, procedural consistency, and structural understanding, leading to better robustness and interpretability.
We conduct a detailed comparison of InfiFPO and SFT on the TheoremQA benchmark to assess their mathematical reasoning capabilities. Our evaluation reveals that InfiFPO demonstrates superior performance in symbolic manipulation, stepwise reasoning, and the application of mathematical identities. While SFT often relies on pattern matching and heuristic recall, InfiFPO constructs solutions via complete, logically consistent derivation chains.
For instance, concerning the evaluation of the definite integral
+∞ -∞ sin(3t) sin(t/π) t 2
dt, InfiFPO successfully decomposes the product of sine functions using trigonometric identities and correctly identifies the contribution of singularities and symmetry in determining the integral's value. In contrast, SFT prematurely concludes that the integral vanishes due to oscillation, neglecting deeper analysis of the integrand's behavior.
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: In section 4, we provide information about the computational resources used. Moreover, in Table 2, we show the GPU hours required for model training.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The research fully adheres to the NeurIPS Code of Ethics, ensuring integrity, fairness, and transparency throughout the study.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [No] Justification: Our method focuses on improving the performance of general models, which does not directly cause social impact. Additionally, we will provide licenses for our code and model weights.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
this section cite: []

Section: Competing Interests.
The authors declare no competing interests.
this section cite: []

Section: 
where ϵ is a small constant ensuring numerical stability [47]. This design follows the intuition that -log M s i (y|x) quantifies the information content of a prediction, where smaller values indicate higher confidence. Taking its inverse reflects the common practice of inverse-uncertainty weighting [48], assigning larger weights to more confident models. This fusion mechanism emphasizes source models with higher confidence, allowing M s fu to adaptively inherit more reliable information from the sources. The corresponding normalized weight for M s i is given by SoftMax:
w i (y|x) = Confidence i (y|x) N j=1 Confidence j (y|x) .(25)
Based on these confidence-derived weights, the fused model M s fu aggregates source models via:
M s fu (y|x) = N i=1
M s i (y|x) wi(y|x) .
this section cite: ['b46', 'b47']

Section: E Supplementary Experiments and Analysis

this section cite: []

Section: E.1 Additional Experimental Results

this section cite: []

Section: E.1.1 Data Diversity Validation
To validate InfiFPO's robustness to data diversity, we conducted experiments on UltraFeedback [49], which contains approximately 60k preference pairs and has been widely adopted in preference optimization research. We treated the preferred (chosen) responses as labels for model fusion baselines.
We selected three general-purpose models for fusion: 1) Qwen2.5-14B-Instruct; 2) Mistral-Small; 3) Gemma-3-Instruct. We compared against InfiFusion and SFT-WRPO, which demonstrated strong performance in our main experiments. As shown in Table 7: Experimental results on UltraFeedback
However, the optimal performance on UltraFeedback (82. 19) is notably lower than on our constructed dataset (83.33), validating the effectiveness of our data construction process.
this section cite: ['b48', 'b18']

Section: E.1.2 Strong Backbone Validation
We conducted experiments using Qwen2.5-14B-Instruct as the target (pivot) model, selecting three source models: 1) Phi-4; 2) Mistral-Small; 3) Gemma-3-Instruct. We used FuseChat, InfiFusion, and WRPO as baselines with identical experimental settings to our main experiments.
this section cite: []

Section: Model / Method Math Code All
Qwen-2.
5-Instruct 72.18 82.62 80.97 FuseChat 75.96 83.50 83.22 InfiFusion 76.09 83.38 83.16 SFT-WRPO 76.42 84.29 83.46 InfiFPO 76.46 85.41 84.01 Table 8: Experimental results using Qwen2.5-14B-Instruct as target model Table 8 demonstrates that when using Qwen2.5-Instruct as the target model, our method still outperforms other baselines, proving its generalizability. In our main experiments, we chose Phi-4 over Qwen2.5-Instruct as the target model because Phi-4 is heterogeneous with all source models (different architectures, vocabularies, etc.), representing a more general and typical scenario for model fusion.
this section cite: []

Section: E.1.3 Reward Model Sensitivity Analysis
To examine data sensitivity, we tested the impact of different reward models on method performance. We tested two reward models: Skywork-Reward-Llama-3.
1-8B-v0.2 and ArmoRM-Llama3-8B-v0.1, both widely used in preference optimization. Model / Method Math Code All Skywork-Reward-Llama-3.1-8B-v0.2 Phi-4 72.86 79.47 79.95 InfiFusion 74.32 82.47 81.96 SFT-WRPO 74.28 83.44 81.93 InfiFPO 74.53 84.88 82.67 ArmoRM-Llama3-8B-v0.1 Phi-4 72.86 79.47 79.95 InfiFusion 73.38 82.54 81.59 SFT-WRPO 73.07 83.04 81.62 InfiFPO 73.45 84.94 82.46
Table 9: Results using different reward models We observe two phenomena from Table 9: (1) InfiFPO consistently outperforms baselines when using different reward models; (2) Different reward models yield different performance improvements. According to Reward Bench v2, Skywork-Reward-Gemma-2-27B-v0.2 significantly outperforms the other two reward models, indicating a positive correlation between reward model performance and final model performance. E.1.4 Sequence Alignment Ablation We conducted ablation experiments to investigate the impact of sequence alignment on model fusion methods. We removed sequence alignment from all model fusion methods for fair comparison. Model / Method Math Code All Phi-4 72.86 79.47 79.95 FuseChat 73.54 82.97 82.12 -without sequence alignment 72.30 79.16 79.43 InfiFusion 75.54 82.67 82.38 -without sequence alignment 73.47 79.98 80.11 InfiFPO 75.60 83.99 82.74 -without sequence alignment 73.72 81.28 80.82 We employ vLLM for acceleration, requiring only approximately 8 GPU hours to compute log probabilities for 5 source models. This explains why our GPU hours increase by merely 10% compared to vanilla DPO when fusing multiple source models.
this section cite: []

Section: Memory Efficiency:
InfiFPO offers significant memory efficiency advantages. It performs model fusion at the sequence level, requiring only sequence-level log probabilities (a few floating-point numbers per sequence). When the number of source models increases, training efficiency remains largely unaffected. In contrast, token-level fusion methods must load probability distributions over
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The main claims in the abstract and introduction align with the paper's contributions and scope, providing a clear and accurate overview.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations of our work in Appendix 6.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: The complete mathematical derivation can be found in Appendix A.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We have provided implementation details in section 4, and included a reproducible anonymous code repository URL in the abstract.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: In section 4, we provide the data sources, and in the abstract, we include an anonymous code repository URL for reproducibility. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We have provided implementation details in section 4 and appendix B.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: Given that calculating statistical significance requires additional resources, we did not perform these calculations. Most of our experimental results show improvements of more than 1 percentage point, indicating a certain level of significance.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [No] Justification: We will provide licenses for our code and model weights in the future.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [No] Justification: Based on the double-blind review process, all code we provide now has been anonymized and is intended only for reviewers to check and reproduce results.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [No] Justification: Based on the double-blind review process, all code we provide now has been anonymized and is intended only for reviewers to check and reproduce results.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Knowledge fusion of large language models Year: (2024)
Ref_id:b1 Title: Fusechat: Knowledge fusion of chat models Year: (2025)
Ref_id:b2 Title: Infigfusion: Graph-on-logits distillation via efficient gromov-wasserstein for model fusion Year: (2025)
Ref_id:b3 Title: Infifusion: A unified framework for enhanced cross-model reasoning via llm fusion Year: (2025)
Ref_id:b4 Title: Democratizing AI through model fusion: A comprehensive review and future directions Year: (2025-10)
Ref_id:b5 Title: Weighted-reward preference optimization for implicit model fusion Year: (2025)
Ref_id:b6 Title:  Year: (2024)
Ref_id:b7 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b8 Title: Google's neural machine translation system: Bridging the gap between human and machine translation Year: (2016)
Ref_id:b9 Title:  Year: (2025)
Ref_id:b10 Title: Gemma 3 technical report Year: (2025)
Ref_id:b11 Title:  Year: (2024)
Ref_id:b12 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b13 Title: Superfiltering: Weak-to-strong data filtering for fast instruction-tuning Year: (2024)
Ref_id:b14 Title: Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions Year: (2024)
Ref_id:b15 Title: Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding Year: (2025)
Ref_id:b16 Title: Skywork-reward: Bag of tricks for reward modeling in llms Year: (2024)
Ref_id:b17 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: ()
Ref_id:b18 Title: Quick and (not so) dirty: Unsupervised selection of justification sentences for multi-hop question answering Year: (2019)
Ref_id:b19 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b20 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b21 Title: Measuring mathematical problem solving with the math dataset Year: ()
Ref_id:b22 Title: Theoremqa: A theorem-driven question answering dataset Year: (2023)
Ref_id:b23 Title: Program synthesis with large language models Year: (2021)
Ref_id:b24 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b25 Title: Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs Year: (2019)
Ref_id:b26 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b27 Title: Instruction-following evaluation for large language models Year: (2023)
Ref_id:b28 Title: A general theoretical paradigm to understand learning from human preferences Year: (2023)
Ref_id:b29 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b30 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b31 Title: Capturing nuanced preferences: Preference-aligned distillation for small language models Year: (2025-07)
Ref_id:b32 Title: Task arithmetic in the tangent space: Improved editing of pre-trained models Year: (2023)
Ref_id:b33 Title: Ties-merging: Resolving interference when merging models Year: (2023)
Ref_id:b34 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b35 Title: Model merging scaling laws in large language models Year: (2025)
Ref_id:b36 Title: Language models are super mario: Absorbing abilities from homologous models as a free lunch Year: (2024)
Ref_id:b37 Title: Evolutionary optimization of model merging recipes Year: (2025)
Ref_id:b38 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b39 Title: Patient knowledge distillation for bert model compression Year: (2019)
Ref_id:b40 Title: Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter Year: (2019)
Ref_id:b41 Title: On-policy distillation of language models: Learning from self-generated mistakes Year: (2024)
Ref_id:b42 Title: Knowledge distillation of large language models Year: (2023)
Ref_id:b43 Title: Unconstrained model merging for enhanced llm reasoning Year: (2024)
Ref_id:b44 Title: Opencompass: A universal evaluation platform for foundation models Year: (2023)
Ref_id:b45 Title: Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation Year: (2023)
Ref_id:b46 Title: Simple and scalable predictive uncertainty estimation using deep ensembles Year: (2017)
Ref_id:b47 Title: What uncertainties do we need in bayesian deep learning for computer vision? Year: (2017)
Ref_id:b48 Title: Ultrafeedback: Boosting language models with scaled ai feedback Year: (2024)
