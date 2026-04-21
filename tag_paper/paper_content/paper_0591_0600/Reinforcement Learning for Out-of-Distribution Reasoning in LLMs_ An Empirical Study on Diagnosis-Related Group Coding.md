Title: Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding
Abstract: Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-SAPPHIRE, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-SAPPHIRE introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-ofthe-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks. 1

Section: Introduction
Medical codes such as DRG play pivotal roles in modern healthcare. DRG codes are fundamental to the inpatient prospective payment system, directly influencing hospital reimbursement and key quality metrics [30]. Currently, assigning DRG codes from clinical notes remains a costly and labor-intensive task, performed manually by highly trained coding specialists.
With the emergence of LLMs, there has been growing interest in leveraging these models for automated medical coding [8,34,38,21,44]. However, DRG coding remains a particularly challenging task for LLMs (Figure 1A), with prior attempts yielding limited success [38,34]. A primary difficulty arises because DRG coding represents an out-of-distribution (OOD) task for off-the-shelf LLMs. Due to the private nature of medical records, most LLMs likely have minimal exposure to patient notes or billing data during pretraining. Additionally, DRG coding is inherently challenging due to: (1) a high-dimensional search space with over 700 DRG codes; (2) advanced clinical reasoning required to link diagnoses with hospital resource use and disease severity; and (3) strict hierarchical rules governing DRG assignment.
Recent advances in reasoning models, such as OpenAI-o1 [15] and DeepSeek-R1 [12], have introduced a paradigm shift in LLM post-training. By leveraging large-scale RL with verifiable rewards, these models exhibit test-time scaling through extended chain-of-thought (CoT) reasoning, achieving state-of-the-art (SOTA) performance on complex tasks like competitive mathematics. Despite this progress, the design of optimal RL algorithms for scalable training remain an open challenge [45,24]. In the healthcare domain, RL applications using verifiable rewards are still in their early stages, with prior work primarily focused on medical knowledge benchmarks [4,19,20].
In this paper, we present a comprehensive exploration of large-scale, reasoning-oriented RL training for automated DRG coding from unstructured clinical notes. In theory, training towards a reasoning model is well-suited for this task: (1) it promotes the development of complex reasoning skills required for accurate code assignment; and (2) more importantly, it generates transparent rationales through CoT reasoning-a key requirement for trust and explainablity in real-world clinical applications.
Through this work, we aim to further derive insights into applying RL to challenging OOD tasks with off-the-shelf LLMs. Using Qwen2.5-7B model and GRPO with DRG-rule-based rewards, we systematically investigate the prerequisites for successful RL, the allocation of data between SFT and GRPO under a fixed data budget, and the impact of scaling SFT data. We also explore a series of RL algorithmic enhancements and adaptive learning strategies. Our core contributions are as follows:
1. We introduce DRG-SAPPHIRE, a novel model developed through large-scale RL, achieving SOTA performance in automated DRG coding. Unlike prior methods, DRG-SAPPHIRE generates clinically helpful, physician-validated reasoning, significantly improving explainability.
2. We demonstrate that the performance ceiling of RL in this OOD task is bounded by the model's capabilities before RL training. Specifically, we observe that RL performance increases linearly with the logarithm of the number of SFT examples, suggesting that scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks.
3. We propose a series of algorithmic enhancements and identify unique challenges in applying RL to DRG coding that distinguish it from mathematical domains-such as a preference for an Answer-First cognitive pattern, and sensitivity to KL divergence for stable training.
this section cite: ['b29', 'b7', 'b33', 'b37', 'b20', 'b43', 'b37', 'b33', 'b14', 'b11', 'b44', 'b23', 'b3', 'b18', 'b19']

Section: Related Work
Automated DRG Coding Given their critical role in hospital operations and reimbursement, there is significant interest in automating DRG coding and enabling early DRG prediction [23,13,38,10]. The prior SOTA method, DRG-LLaMA, fine-tunes a LLaMA model as a sequence classifier by replacing its generation head with a classification head [38]. Most existing approaches similarly frame DRG coding as a multi-class classification task, offering limited insight into the rationale behind code assignments. While methods like DRGCoder provide input-level weight visualizations [13], their interpretability remains insufficient for real-world clinical deployment, where transparency and explainability are critical.
this section cite: ['b22', 'b12', 'b37', 'b9', 'b37', 'b12']

Section: Replication Efforts of Deepseek-R1
Recent studies have actively explored replicating the RL recipes of DeepSeek-R1, particularly in mathematical and coding domains, with varying degrees of success [47,14,40]. One line of work has proposed approaches to address biases and improve sample efficiency in the original GRPO algorithm [45,24,22]. Another active research area focuses on curriculum and staged learning strategies during reasoning-oriented RL [48,36,41,16,3].
New Capabilities from RL? A central debate concerns whether RL truly fosters new capabilities beyond those already encoded in the base model. In DeepSeekMath, RL improved Majority@K but not Pass@K performance on mathematical tasks [33]. A comprehensive analysis across mathematical, coding, and visual reasoning tasks found that RL with verifiable rewards primarily reinforces existing reasoning capabilities rather than fostering novel ones [46]. Recently, Ma et al. [26] analyzed training dynamics on complex reasoning tasks, showing that RL strengthens performance within a model's existing capabilities, whereas SFT more effectively extends them beyond its current scope.
3 Large-scale RL for Automated DRG Coding
this section cite: ['b46', 'b13', 'b39', 'b44', 'b23', 'b21', 'b47', 'b35', 'b40', 'b15', 'b2', 'b32', 'b45', 'b25']

Section: Problem Formulation
We aim to automate the hierarchical assignment of Medicare Severity Diagnosis-Related Group (MS-DRG) codes using LLMs. The MS-DRG system classifies each hospitalization into a single DRG code based on clinical complexity and resource utilization (see Appendix A.1 for details).
Given a hospitalization represented by a set of clinical documents D, the DRG coding process applies an extraction function h to identify the principal diagnosis w d or procedure w p , and the presence of Complications or Comorbidities (CC) or Major Complications or Comorbidities (MCC). A hierarchical mapping function f then determines the final DRG code. Formally, the MS-DRG assignment is defined as:
(w d , w p , CC, MCC) = h(D), g = f (w d , w p , CC, MCC), where g is the assigned DRG code. In this paper, we use an LLM to automate this complex process.
this section cite: []

Section: Preliminary: GRPO
Compared to Proximal Policy Optimization [32], GRPO eliminates the value function and estimates the advantage using relative rewards within a group [33]. For each question q, GRPO samples a group of outputs {o 1 , o 2 , • • • , o G } from the old policy π θold and then optimizes the target policy π θ . In this paper, we enforce π θold = π θ to ensure strict on-policy learning. Under this setting, we maximizing the following objective:
JGRP O (θ) = E[q ∼ P (Q), {oi} G i=1 ∼ π θ old (O|q)] 1 G G i=1 1 |oi| |o i | t=1 Âi,t -β( π ref (oi,t|q, oi,<t) π θ (oi,t|q, oi,<t) -log π ref (oi,t|q, oi,<t) π θ (oi,t|q, oi,<t) -1) ,(1)
where β is the coefficient for the KL divergence penalty, π θref is the reference policy, and Âi,t is the advantage, computed based on the relative rewards within each group {r i } G i=1 as:
Âi,t = r i -mean({r i } G i=1 ) std({r i } G i=1 ) .(2)
Here, r i denotes the reward assigned to output o i for prompt q. The gradient of J GRPO (θ) is:
∇ θ JGRP O (θ) = E[q ∼ P (Q), {oi} G i=1 ∼ π θ old (O|q)] 1 G G i=1 1 |oi| |o i | t=1 Âi,t + β π ref (oi,t|oi,<t) π θ (oi,t|oi,<t) -1 ∇ θ log π θ (oi,t|q, oi,<t).(3)
this section cite: ['b31', 'b32']

Section: Improving GRPO Beyond the Baseline
We propose a set of strategies to address key limitations of GRPO.
this section cite: []

Section: Dynamic Resampling for Advantage Preservation
Existing RL algorithms suffer from the gradient-diminishing problem. In GRPO, if all completions {o i } G i=1 for a prompt q receive the same reward value, the resulting advantage for this group becomes zero. As training progresses, this issue becomes more pronounced due to policy optimization and accompanying entropy collapse [45], as more prompts yield completions with no reward variance-either because all completions are perfectly correct or uniformly incorrect. This leads to a progressive decrease in the learning signal from the reward-based advantage.
To address this, we propose a dynamic resampling strategy (Equation 4). For each prompt q, if sampled completions yield zero reward variance, we resample up to N max times until nonzero variance is observed. Optionally, we enforce that at least one completion receives a positive reward, guiding gradient updates toward high-reward trajectories.
JGRP O (θ) = E[q ∼ P (Q), {oi} G i=1 ∼ π θ old (O|q)] 1 G G i=1 1 |oi| |o i | t=1 Âi,t -β( π ref (oi,t|q, oi,<t) π θ (oi,t|q, oi,<t) -log π ref (oi,t|q, oi,<t) π θ (oi,t|q, oi,<t)-1)
s.t. Var({ri} G i=1 ) > 0 within Nmax, optionally: |{oi | ri > 0}| > 0.(4)
Our approach differs from the dynamic sampling strategy in DAPO [45], which discards prompts that yield uniformly correct or incorrect completions. Given the data scarcity in clinical domains, we instead maximize the utility of each training example by resampling rather than discarding.
this section cite: ['b44', 'b44']

Section: Intervening on Cognitive Behaviors
Cognitive behaviors, such as verification and backtracking, are critical for effective reasoning-oriented RL [11]. We explored additional reward functions and a specialized SFT dataset (detailed in Section A.
3) to incentivize three cognitive patterns in CoT reasoning, as shown in Figure 2. These are: (1) Answer-First, where the model outputs the DRG code before CoT; (2) CoT-First, where the model generates CoT reasoning before the DRG code; and (3) Differential Thinking, where the model evaluates three potential DRG codes before selecting the most appropriate. <think> Based on the provided discharge summary, the most likely Medicare Severity Diagnosis-Related Group (MS-DRG) is "OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC." ### Reasoning: 1. **Principal Diagnosis: ... 2. **CC (Complication or Comorbidity): ... 3. **MCC (Major Complication or Comorbidity): ... 4. ** discharged with specific medications and follow-up plans: ... ### Conclusion: The DRG code "OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC" is assigned because the patient's primary condition, despite being initially concerning, did not meet the criteria for an MCC... </think> <answer> OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC </answer> Answer-First <think> To determine the most likely MS-DRG for this patient, let's break down the key elements from the discharge summary: 1. **Principal Diagnosis**: ... 2. **Secondary Diagnoses and Procedures**:... 3. **Complications and Comorbidities (CC and MCC)**:... **Reasoning for the DRG Assignment**:... **DRG Code Assignment**: The most likely MS-DRG for this patient, based on the provided information, would be **"OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC."** This DRG code is assigned because ... </think> <answer> OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC </answer> CoT-First <think> #### Principal Diagnosis: ... #### Secondary Diagnoses: ... #### Procedures Performed: ... #### Other Relevant Factors:... ### Possible DRG Codes 1. **[Heart failure and shock without CC/MCC]** 2. **[Other circulatory system diagnoses with CC]** 3. **[Other circulatory system diagnoses with MCC]** ### Analysis for Each DRG Code: ... ### Conclusion Most Likely DRG Code: [Other circulatory system diagnoses with CC], as it best reflects... ... </think> <answer> OTHER CIRCULATORY SYSTEM DIAGNOSES WITH CC </answer> Differential Thinking
this section cite: ['b10']

Section: KL Divergence Decay
The KL divergence term in the GRPO objective (Equation 1) regularizes the divergence between the target policy π θ and the reference policy π θref . However, this term exacerbates the gradient-diminishing problem in RL: as training progresses and more prompts yield zero-variance responses, the gradient, per Equation 3, becomes dominated by the KL term, pulling π θ toward π θref . This drives over-regularization toward the reference policy and risks policy degradation. Recent work suggests that removing the KL penalty enhances reasoning capabilities in mathematical domains [45,24,14]. Motivated by this, we explored two setups: (1) completely removing the KL divergence term from the objective, and (2) applying a cosine decay schedule to the KL term's coefficient β, smoothly reducing it to zero during training (see Section A.4 for details).
this section cite: ['b44', 'b23', 'b13']

Section: GRPO Variants
In Equation 1, dividing by |o i | during group-level advantage normalization introduces a length bias, diminishing the influence of longer completions on the policy gradient. To address this, DAPO [45] uses G i=1 |o i | as the denominator, while Dr. GRPO [24] adopts a constant normalization factor. Additionally, Dr. GRPO removes the division by std({r i } G i=1 ) in Equation 2to mitigate question-level difficulty bias. We systematically evaluated these three strategies. Due to the strict on-policy nature of our setting (π θold = π θ ), we did not explore other modifications, such as clip-higher [45].
this section cite: ['b44', 'b23', 'b44']

Section: Reward Shaping
We implemented two straightforward yet robust rule-based reward components: Format Reward and Accuracy Reward (detailed in the Section A.2). For the Accuracy Reward, we investigated three distinct strategies: Dense Reward, Balanced Reward, and Strict Reward. These reward functions were designed to provide varying levels of reward signal sparsity, contingent on the correctness of the DRG code, its associated principal diagnosis, and the CC/MCC status.
this section cite: []

Section: Adaptive Learning Strategy
Curriculum Learning We investigate whether a curriculum learning strategy, which organizes training cases by difficulty, improves performance compared to a mixed-difficulty baseline. We evaluated four setups, detailed in Appendix A.6: (1) excluding easy cases, (2) excluding hard cases, (3) excluding both easy and hard cases (i.e., using only medium-difficulty cases), and (4) training on easy cases first, then progressing to hard cases.
Staged Learning Lastly, we explored a staged learning strategy with three training phases of roughly equal length. After each phase, we identified easy and hard cases and evaluated two approaches: (1) additional SFT on hard cases, and (2) additional DPO on hard cases, before advancing to the next stage. As detailed in Appendix A.7, these approaches aim to improve the model's handling of challenging cases through targeted learning.
this section cite: []

Section: Implementation Details
Dataset We utilized the DRG-LLaMA training and test sets [38], derived from the publicly available MIMIC-IV dataset of real-world medical records [17]. The full training and test sets include 236,192 and 26,244 cases, respectively. Each case uses the "brief hospital course" section from the discharge summary as input, with MS-DRG codes consolidated to version 34.0.
this section cite: ['b37', 'b16']

Section: Training Pipeline and Scaling Strategy
An overview of the training pipeline is shown in Figure 3. We first sampled a reduced dataset termed DRG-Small, comprising 20% of the full data (N=46,758). This subset served as the foundation for extensive experiments on methodological variants and SFT-RL data mixtures, as detailed in Sections 5.2 through 5.3. After identifying the optimal configuration, we scaled training to the full dataset to produce the final DRG-SAPPHIRE model.
this section cite: []

Section: Step 1
Bootstrap CoT Reasoning for DRG Assignment Using Qwen2.5-7B
this section cite: []

Section: Step 2
SFT on the Qwen2.5-7B Using Cold-Start Data
this section cite: []

Section: Construction of SFT Dataset
We prompted the Qwen2.5-7B-Instruct model with medical records and ground-truth DRG codes, tasking it to generate reasoning for DRG assignments (prompt provided in Section I). After extensive prompt engineering, manual inspection by domain expert revealed that the dataset exhibits strong reasoning logic (e.g., analyzing principal diagnosis first) but frequently contains factual errors (e.g., misclassifying a condition's CC/MCC status). We also included the complete list of original V34.0 MS-DRG codes in a question-answer format within the SFT dataset.
Model and RL Training We selected Qwen2.5-7B-Instructs [43] for the main experiments after evaluating various model size. GRPO training was conducted using the TRL package [37] for one epoch across all experiments.
Evaluation Metrics We report model performance on the full test set using Pass@1, Pass@8, and Majority@8 (Maj@8), following prior work in reasoning-oriented RL [33,46]. Pass@1, reported as the model's accuracy, is the mean accuracy across eight runs. Pass@8 assesses whether the correct DRG code appears among eight generated outputs, while Maj@8 determines if the most frequent output matches the correct DRG code.
this section cite: ['b42', 'b36', 'b32', 'b45']

Section: Experiments

this section cite: []

Section: Results of DRG-SAPPHIRE
Our best DRG-SAPPHIRE model was trained using a 90% SFT and 10% RL ratio on the full dataset (see Section 5.2 for SFT vs. RL ratio experiments), incorporating optimal GRPO enhancements and adaptive learning strategies (see Section 5.3 for ablation studies).
Comparison with Baselines As shown in Figure 1A, DRG-SAPPHIRE significantly outperforms proprietary reasoning models, non-reasoning models, and the DeepSeek-distilled Qwen 32B. It achieves new SOTA performance on DRG coding, surpassing the previous best, DRG-LLaMA-7B (54.8% vs. 53.9%). In addition to improved accuracy, DRG-SAPPHIRE provides interpretable reasoning-a compelling advantage over prior models trained purely as classifiers. Helpfulness Accuracy 1=Very Poor 2=Poor 3=Acceptable 4=Good 5=Very Good Review Score Expert Reader Study Results Four physicians in hospital leadership roles, actively engaged in DRG-related initiatives (e.g., reducing geometric mean length of stay), evaluated DRG-SAPPHIRE 's reasoning across 30 cases. On the dimensions of Helpfulness and Accuracy, DRG-SAPPHIRE received a median rating of 4 out of 5, suggesting good potential for real-world applications (Figure 4). Qualitative assessments highlighted the explainability of DRG coding as highly valuable for DRG-related initiatives (see Section D.1 for details), despite occasional factual inaccuracies in the reasoning.
this section cite: []

Section: Optimizing Data Allocation Between SFT and GRPO
Pass@1 Pass@8 Maj@8 10 20 30 40 50 60 70 80 Accuracy (%) 25.0 54.0 31.4 38.7 49.6 39.6 +54.6% -8.2% +26.0% A. Deepseek-R1-Style (5% SFT, 95% RL) SFT + GRPO Pass@1 Pass@8 Maj@8 10 20 30 40 50 60 70 80 Accuracy (%) 32.9 68.0 40.1 43.0 58.9 44.4 +30.6% -13.3% +10.7% B. 25% SFT, 75% RL SFT + GRPO Pass@1 Pass@8 Maj@8 10 20 30 40 50 60 70 80 Accuracy (%) 36.7 70.9 43.6 46.5 57.4 47.2 +26.8% -19.0% +8.3% C. 50% SFT, 50% RL SFT + GRPO Pass@1 Pass@8 Maj@8 10 20 30 40 50 60 70 80 Accuracy (%) 38.5 72.1 45.4 47.4 59.6 48.3 +23.3% -17.4% +6.4% D. 75% SFT, 25% RL SFT + GRPO Pass@1 Pass@8 Maj@8 20 Effect of SFT-GRPO Ratios on DRG-Small First, we investigated the impact of varying the allocation of a fixed data budget between SFT and GRPO on the DRG-Small subset (N=46,758). This contrasts with Deepseek-R1-style training, where only minimal SFT precedes RL. Across all data splits, GRPO consistently improved Pass@1 over the SFT baseline by an absolute margin of approximately 10 percentage points (see Figure 5). We observed that this gain is driven by improvements in Maj@8, not Pass@8; in fact, Pass@8 declines with GRPO. This pattern suggests that RL sharpens the model's output distribution toward higher-reward pathways, rather than introducing new reasoning capabilities in our experiments. Notably, the decline in Pass@8 during training indicates that RL may constrain diverse reasoning exploration. These findings align with recent studies [46,33], which question whether RL improves reasoning beyond the base model's capabilities. Moreover, the ultimate performance ceiling achievable with GRPO appears to be largely determined by the capacity of the initial SFT model; a stronger SFT foundation generally leads to better post-GRPO results. From a computational perspective, scaling SFT prior to RL is more efficient, as GRPO entails substantial inference-time cost (see Figure 5F).
0 250 500 750 1000 1250 1500 Global Steps 45.0 47.5 50.0 52.5 55.0 57.5
this section cite: ['b45', 'b32']

Section: Accuracy (%) A. Accuracy vs. Global Steps
Full Data, 50% SFT Full Data, 75% SFT 2 11 2 12 2 13 2 14 2 15 2 16 2 17 2 18SFT Sample Size before RL
50 55 60 65 70 75 Pass@8 (%) B. Pass@8 vs SFT Sample Size SFT GRPO Full Data, 50% SFT 2 11 2 12 2 13 2 14 2 15 2 16 2 17 2 18SFT Sample Size before RL Log-Linear Scaling of GRPO with Increasing SFT Next, we scaled our training pipeline to the full dataset (N=236,192). Based on the results above, we started with an SFT-GRPO data ratio of 50%-50% and progressively increased the SFT ratio under a fixed total data budget. Plotting these results alongside the DRG-Small subset revealed that both GRPO and SFT performance scale approximately linearly with the logarithm of the number of SFT examples (Figure 1B). Although the number of GRPO steps varies across experiments in Figure 1B, the benefit of scaling RL appears limited in our study. Figure 6A illustrates results from our longest GRPO runs, demonstrating modest benefits beyond 500 global steps. Consistent with earlier findings, GRPO reliably improves Pass@1 and Maj@8 while reducing Pass@8 (Figure 6B and C). As the number of SFT samples increased, the slope of the GRPO curves converged toward that of SFT across all metrics. Additional results from scaling to the full dataset are detailed in Section C.3.
this section cite: []

Section: Ablation Studies on GRPO Enhancements and Adapative Learning
We present the results of ablation studies in Table 1 and
this section cite: []

Section: Dynamic
Resampling Surprisingly, dynamic resampling-with or without a positive reward constraint-yielded marginally better or even worse performance than vanilla GRPO (Table 1), despite
DRG Principal Diagnosis CC/MCC Model Pass@1 Pass@8 Maj@8 Pass@1 Pass@8 Maj@8 Pass@1 Pass@8 Maj@8 Baseline Vanilla GRPO + Dense Reward 38.5 48.2 39.3 52.5 58.5 53.4 47.8 60.0 49.0 Dynamic Resampling Neutral Resampling 20.3 41.9 38.1 27.0 52.5 50.5 25.6 52.6 48.0 Positive Reward Resampling 39.2 44.8 39.6 52.9 56.4 53.3 48.3 55.6 49.0 Coginitive Behvaiors Intervention COT-First 35.5 52.2 37.4 50.9 59.6 52.4 46.3 66.7 48.4 Differential Thinking 30.2 47.3 33.9 46.7 57.0 50.9 40.6 63.0 45.2 GRPO Variants DAPO Loss 40.1 48.0 40.6 53.8 58.5 54.3 49.4 59.1 50.3 Dr. GRPO Loss 37.5 47.6 38.1 50.9 57.2 51.4 48.8 60.7 49.8 Dr. GRPO Advantage 38.5 51.9 39.6 53.4 60.5 54.3 47.6 63.6 49.1 KL Divergence No KL 39.8 42.4 39.9 53.6 55.2 53.7 49.1 52.3 49.3 KL Decay 38.2 42.0 38.3 52.2 54.7 52.4 48.8 53.7 49.0 Reward Shaping Strict Reward 40.1 49.1 40.9 52.8 58.1 53.7 47.6 59.0 48.8 Balanced Reward 38.1 51.3 40.0 52.1 60.4 53.8 48.2 64.0 50.7 Curriculum Learning Remove Easy Cases 35.8 51.9 37.6 50.3 59.2 51.7 46.6 65.8 48.7 Remove Hard Cases 40.4 46.6 40.7 53.2 56.5 53.7 49.5 57.2 50.1 Remove Easy and Hard Cases 38.7 48.2 39.4 52.9 58.1 53.4 48.3 59.9 49.3 From Easy to Hard 29.4 51.7 32.7 43.4 58.6 46.5 40.8 68.5 44.3 Staged Learning Staged SFT 39.3 49.1 40.0 52.9 59.2 53.8 46.0 58.6 47.1 Staged DPO 29.3 46.1 31.2 43.8 54.3 45.5 43.1 64.2 45.7 Table 1: Ablation Study Results. Rows with a blue background indicate superior Pass@1 performance compared to Vanilla GRPO + Dense Reward. Bold values denote the highest score for each metric.
preserving high reward variance (Figure 7A). More importantly, dynamic resampling proved computationally inefficient due to the frequent need to regenerate responses (Figure 7C). We hypothesize that dynamic resampling may introduces training instability by oversampling zero-variance prompts, thereby skewing batches toward OOD responses rarely generated by the current policy. Additionally, this approach may inadvertently over-penalize low-reward outputs newly introduced into the batch, further distorting the learning signal.
Intervening on Cognitive Behaviors Our SFT dataset includes diverse reasoning styles, notably both Answer-First and CoT-First patterns. Interestingly, during training, the policy frequently converged toward the Answer-First strategy. To encourage CoT-First behavior, we experimented with an additional rule-based reward and adjusted the SFT dataset to explicitly promote Differential-Thinking. Although both interventions successfully elicited the intended cognitive behaviors, their performance lagged behind the naturally emerging Answer-First pattern (Table 1). This finding is surprising, as CoT-First strategies are often effective in complex reasoning tasks [39]. We hypothesize that DRG coding benefits from a direct prediction strategy, where outputting the DRG code first leverages implicit knowledge in the model's latent space, outperforming explicit CoT-grounded reasoning. These findings also align with recent studies [27,5], which suggest that CoT and extended reasoning may not always be necessary for reasoning models, and a "no-thinking" pattern can sometimes yield better performance.
this section cite: ['b38', 'b26', 'b4']

Section: KL Divergence
In our experiments, removing the KL penalty frequently led to model collapse (Figure 8A). This contrasts sharply with findings in mathematical reasoning tasks, where the KL term is less critical, underscoring its importance for cross-domain generalization. However, in cases where training successfully completed without the KL penalty, performance surpassed that of vanilla GRPO (Table 1). Additionally, a cosine KL decay schedule appeared beneficial. While it yielded no significant gains in small-scale runs, it improved the training curve toward the end, suggesting that a lower KL penalty in later stages may help prevent over-regularization to the reference policy (Figure 8B). Indeed, KL decay proved beneficial when scaling training on the full dataset, as shown in Table 2. GRPO Variants Among three GRPO variants, the DAPO loss achieved the highest performance, while the Dr. GRPO loss performed the lowest (Table 1). This finding aligns with recent work reporting that Dr. GRPO does not outperform vanilla GRPO [6]. Across all settings, we observed completion length contraction: as accuracy improved, output lengths sharply decreased before stabilizing (Figure 9B). This contrasts with trends observed in mathematical tasks, where longer completions are often associated with better performance.
this section cite: ['b5']

Section: Reward Shaping
The strict accuracy reward, despite providing the sparsest reward signals, outperformed both dense and balanced reward variants (Table 1). Notably, we observed no improvement in pincipal diagnosis or CC/MCC accuracy under the denser reward schemes. We hypothesize that denser rewards may lead the policy to converge prematurely to local optima, trading off global performance for easier-to-optimize intermediate signals.
this section cite: []

Section: Adaptive Learning
We observed benefits from removing easy and hard cases during training (Table 1). Similarly, recent studies in the math domain suggest that maintaining medium-level difficulty cases may be most effective for RL training [36,41,16,42]. Staged learning with SFT resulted in only modest performance gains despite additional compute.
this section cite: ['b35', 'b40', 'b15', 'b41']

Section: Prerequisites for Effective GRPO Training
We found that vanilla Qwen2.5 models (base and instruct) without SFT failed to generate correct DRG codes using GRPO alone, despite rapidly adopting the target reasoning format (Figure 10A). Post-SFT, all models showed improved RL performance that generally scaled with model size, though gains from 7B to 14B were modest (Figure 10B). Higher SFT learning rates (up to 4 × 10 -5 ) and extended training epochs both improved GRPO performance, though gains from additional epochs diminished at higher learning rates (Figure 10C). These results align with recent findings [29] emphasizing the importance of aggressive SFT for reasoning-intensive tasks.
this section cite: ['b28']

Section: Conclusion
In this work, we used DRG coding as an empirical study to explore RL for OOD reasoning in LLMs. Our approach, applying GRPO with verifiable rewards, achieved a new SOTA performance while offering a key advantage over prior methods: the generation of physician-validated explanations through CoT reasoning. Critically, our findings reveal that RL performance on this OOD task is fundamentally constrained by the base model's capacity prior to RL. We observed a logarithmic scaling relationship between the number of SFT examples and subsequent RL performance.
Following the successes of reasoning models like DeepSeek-R1, a prevailing narrative has been to "scale RL," leaving a critical question unanswered: what, precisely, should be scaled? Our work addresses this for complex, OOD tasks where knowledge infusion emerges as a critical component. We find that scaling SFT can be more effective and computationally efficient than scaling RL alone. Moreover, despite extensive experimentation with RL algorithmic enhancements and adaptive learning strategies, these refinements yield only modest gains compared to simply initializing RL from stronger SFT baselines-highlighting a "bitter lesson" in applying RL to tasks that fall outside the pretraining distribution of LLMs.
this section cite: []

Section: References
Ref_id:b0 Title: MIMIC-IV on physionet Year: ()
Ref_id:b1 Title: Responsible use of mimic data with online services like gpt Year: ()
Ref_id:b2 Title: Online difficulty filtering for reasoning oriented reinforcement learning Year: (2025)
Ref_id:b3 Title: Huatuogpt-o1, towards medical complex reasoning with llms Year: (2024)
Ref_id:b4 Title: Reasoning models don't always say what they think Year: (2025)
Ref_id:b5 Title: Gpg: A simple and strong reinforcement learning baseline for model reasoning Year: (2025)
Ref_id:b6 Title: Icd-10-cm/pcs ms-drg v34. 0 definitions manual Year: (2016)
Ref_id:b7 Title: Automated clinical coding: what, why, and where we are? Year: (2022)
Ref_id:b8 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025)
Ref_id:b9 Title: Can large language models replace coding specialists? evaluating gpt performance in medical coding tasks Year: (2025)
Ref_id:b10 Title: Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars Year: (2025)
Ref_id:b11 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b12 Title: Explainable clinical coding for the early prediction of diagnostic-related groups Year: (2023)
Ref_id:b13 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (2025)
Ref_id:b14 Title: Openai o1 system card Year: (2024)
Ref_id:b15 Title: How difficultyaware staged reinforcement learning enhances llms' reasoning capabilities: A preliminary experimental study Year: (2025)
Ref_id:b16 Title: Mimic-iv, a freely accessible electronic health record dataset Year: (2023)
Ref_id:b17 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b18 Title: Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models Year: (2025)
Ref_id:b19 Title: Clinicalgpt-r1: Pushing reasoning capability of generalist disease diagnosis with large language model Year: (2025)
Ref_id:b20 Title: Exploring llm multi-agents for icd coding Year: (2024)
Ref_id:b21 Title: Accelerating the training of group relative policy optimization-based reasoning models Year: (2025)
Ref_id:b22 Title: Early prediction of diagnostic-related groups and estimation of hospital cost by processing clinical notes Year: (2021)
Ref_id:b23 Title: Understanding r1-zero-like training: A critical perspective Year: (2025)
Ref_id:b24 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b25 Title: Learning what reinforcement learning can't: Interleaved online fine-tuning for hardest questions Year: (2025)
Ref_id:b26 Title: Reasoning models can be effective without thinking Year: (2025)
Ref_id:b27 Title: Rule based rewards for language model safety Year: (2024)
Ref_id:b28 Title: Open r1 update 3: Steady progress and a new technical report Year: (2024)
Ref_id:b29 Title: After the revolution: Drgs at age 30 Year: (2014)
Ref_id:b30 Title: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters Year: (2020)
Ref_id:b31 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b32 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b33 Title: Large language models are poor medical coders-benchmarking of medical code querying Year: (2024)
Ref_id:b34 Title: Dr-llava: Visual instruction tuning with symbolic clinical grounding Year: (2024)
Ref_id:b35 Title: Kimi k1. 5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b36 Title: Trl: Transformer reinforcement learning Year: (2020)
Ref_id:b37 Title: Drg-llama: tuning llama model to predict diagnosis-related group for hospitalized patients Year: (2024)
Ref_id:b38 Title: Chain-ofthought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b39 Title: Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning Year: (2025)
Ref_id:b40 Title: A minimalist approach to llm reasoning: from rejection sampling to reinforce Year: (2025)
Ref_id:b41 Title: Learning to reason under off-policy guidance Year: (2025)
Ref_id:b42 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b43 Title: Surpassing gpt-4 medical coding with a two-stage approach Year: (2023)
Ref_id:b44 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b45 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b46 Title: Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild Year: (2025)
Ref_id:b47 Title: Srpo: A cross-domain implementation of large-scale reinforcement learning on llm Year: (2025)
