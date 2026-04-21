Title: QFFT, Question-Free Fine-Tuning for Adaptive Reasoning
Abstract: Recent advancements in Long Chain-of-Thought (CoT) reasoning models have improved performance on complex tasks, but they suffer from overthinking, which generates redundant reasoning steps, especially for simple questions. This paper revisits the reasoning patterns of Long and Short CoT models, observing that the Short CoT patterns offer concise reasoning efficiently, while the Long CoT patterns excel in challenging scenarios where the Short CoT patterns struggle. To enable models to leverage both patterns, we propose Question-Free Fine-Tuning (QFFT), a fine-tuning approach that removes the input question during training and learns exclusively from Long CoT responses. This approach enables the model to adaptively employ both reasoning patterns: it prioritizes the Short CoT patterns and activates the Long CoT patterns only when necessary. Experiments on various mathematical datasets demonstrate that QFFT reduces average response length by more than 50%, while achieving performance comparable to Supervised Fine-Tuning (SFT). Additionally, QFFT exhibits superior performance compared to SFT in noisy, out-of-domain, and low-resource scenarios. QFFT is publicly available at https://github.com/LWL-cpu/Question-Free-Fine-Tuning.

Section: Introduction
Recent advancements in Long CoT reasoning models, such as OpenAI o1 [1] and DeepSeek-R1 [2], have significantly improved performance on complex tasks, such as mathematics and coding [3,4,5,6,7]. These improvements are largely attributed to the test-time scaling paradigm, where models generate Long CoT, consuming more tokens, to effectively simulate human-like deep thinking behavior like self-reflection, error correction, and exploration of multiple solution strategies. Building on this success, a growing body of research has focused on distillation methods that transfer the Long CoT reasoning abilities of OpenAI o1 and DeepSeek-R1 into smaller LLMs, achieving notable performance gains [8,9,10,2].
However, recent studies [11,12] have identified a critical limitation in current Long CoT models (e.g., DeepSeek-R1): they often exhibit overthinking, generating unnecessarily complex or redundant reasoning steps even for simple problems. As a result, models distilled from these Long CoT models tend to inherit this drawback, leading to inefficiencies during inference.
To mitigate this issue, recent Long-to-Short methods [13,14,15,16,17] explore compressing the length of Long CoT responses. However, these methods incur substantial additional training costs while achieving only limited reductions in token usage. In this paper, we compare the reasoning patterns of Short CoT models and Long CoT models: (1) For efficiency: Short CoT models provide concise and efficient reasoning, whereas Long CoT models often generate lengthy outputs, which typically involve unnecessarily complex or redundant steps for simple problems. (2) For effectiveness: Short CoT models struggle with more difficult problems due to their simplistic reasoning patterns, while Long CoT models, with their reflective reasoning, demonstrate clear advantages in such scenarios. This contrast motivates a natural question: Can we design models that adaptively select between Short CoT and Long CoT reasoning patterns? Such models would ideally combine the strengths of both reasoning patterns-employing Short CoT reasoning for simple problems to enhance efficiency, and leveraging Long CoT reasoning for difficult problems to pursuit effectiveness.
We revisit current Long CoT Supervised Fine-Tuning (SFT): models are trained on large-scale (question, Long CoT response) pairs, suffering from overthinking issues. Inspired by [18], we hypothesize that this issue arises because SFT enforces the mapping from questions to Long CoT responses, overriding the model's original Short CoT patterns and causing it to apply Long CoT reasoning indiscriminately-even when concise reasoning would suffice. To design models that could leverage both Short CoT and Long CoT patterns for a better balance between efficiency and effectiveness, we set two goals: (1) preserve the model's default Short CoT reasoning patterns, and (2) enable it to acquire Long CoT patterns that trigger reflective behaviors when facing uncertainties or errors. To achieve these objectives, we propose Question-Free Fine-Tuning (QFFT), a fine-tuning method that discards input questions and only fine-tunes Long CoT responses. This approach avoids overriding the Short CoT patterns while still enabling the model to learn reflective reasoning patterns of Long CoT.
Our extensive experiments demonstrate that the QFFT method adaptively integrates both reasoning patterns: it employs Short CoT reasoning for simple problems to enhance efficiency, and leverages Long CoT reasoning for more challenging problems. Further analysis reveals that the QFFT model prioritizes Short CoT patterns by default, transitioning to reflective Long CoT reasoning when encountering errors or uncertainties. Notably, QFFT achieves performance comparable to SFT on six math datasets, while significantly reducing the average number of generated tokens by up to 50%, thereby effectively mitigating the issue of overthinking. Moreover, QFFT exhibits superior performance compared to SFT in noisy, out-of-domain, and low-resource scenarios.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b1', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17']

Section: Towards Adaptive Reasoning

this section cite: []

Section: Definition of Adaptive Reasoning
Chain-of-Thought (CoT), since its introduction, has been widely adopted in various models (e.g., Qwen2.5-32B-Instruct), significantly enhancing its reasoning capabilities. With recent advancements, modern large reasoning models (e.g., DeepSeek-R1) increasingly employ Long CoT reasoning patterns, characterized by reflective verification behaviors [19,6]. Reasoning patterns exhibiting Table 1: Comparison of Reasoning Modes: Short CoT, Long CoT, and Adaptive Reasoning Definition 1 Adaptive Reasoning refers to a model's ability to adaptively prioritize Short CoT reasoning patterns for simple questions and prioritize Long CoT reasoning patterns for difficult questions when Short CoT is ineffective.
Here, simple questions are those solvable by the model using Short CoT patterns, while difficult questions are those for which Short CoT fails.
this section cite: ['b18', 'b5']

Section: Quantitative Metric on Adaptive Reasoning Ability
The adaptive reasoning capability of a model can be evaluated by measuring the alignment between its reasoning patterns and question difficulty. Here, question difficulty is based on whether the Short CoT of the evaluated model can correctly answer the question.
this section cite: []

Section: Assumption 1
If a Long CoT model M L is derived (e.g. distilled) from a Short CoT model M S , we approximate the Short CoT capability of model M L by that of model M S .
Consequently, we introduce the Short CoT model M S as a reference model to estimate whether the evaluated model can correctly answer the question using Short CoT reasoning.
We introduce the Reasoning Adaptability Cohen's Kappa (RAK), inspired by Cohen's Kappa [21], a statistical measure evaluating the agreement between two raters beyond chance. In our context, the two "raters" correspond to the question difficulty provided by the reference model (simple or difficult) and the reasoning patterns (Short or Long CoT) used by the evaluated model.
this section cite: ['b20']

Section: Definition 2 Reasoning Adaptability Cohen's Kappa (RAK).
If the Long CoT model M L is derived (e.g., distilled) from the original Short CoT model M S (reference model), RAK measures the performance of model M L in adaptively selecting the appropriate reasoning pattern, accounting for chance agreement. , computed based on the marginal probabilities of the predicted and actual classes 3 . A higher RAK indicates greater consistency between the evaluated model's reasoning pattern selection and task difficulty, demonstrating stronger adaptive reasoning capability. Conversely, a lower RAK reflects weaker consistency, indicating poorer adaptive reasoning capability.
RAK = p o -p e 1 -p e ,(1)
this section cite: []

Section: Pilot Study on Adaptive Reasoning
We evaluate the reasoning adaptability of existing Long CoT models on the MATH500 [22] dataset. As shown in Figure 2a, we observe that current Long CoT models (obtained by SFT) have low RAK scores, which attributes to their consistent over-reliance on Long CoT reasoning patterns.
this section cite: ['b21']

Section: Hypothesis.
In Long CoT SFT, models are trained on large-scale (Q: question, R: Long CoT response) pairs, teaching them to solve questions with Long CoT reasoning patterns. This training paradigm poses a potential risk [18]: models may default to Long CoT reasoning for all inputs, a phenomenon we term Override. We hypothesize that this phenomenon occurs because the new Q → R mapping-from questions to Long CoT responses-overwrites the original mapping to Short CoT responses, causing the model to over-rely on Long CoT patterns.
To verify this hypothesis, we conduct the following experiment: during Long CoT SFT, we randomly select a proportion α (0 < α < 1) of training samples and retain their original question-response pairs. For the remaining (1 -α) proportion of samples, we remove the questions, keeping only the Long CoT responses. We then evaluate how the ratio of the model's Short CoT changes as we gradually increase the value of α.
Results. As shown in Figure 2b, when questions are removed from all training samples, the model exhibits over 50% Short CoT patterns 4 and does not over-rely on Long CoT reasoning. However, once samples with questions are included, even with a minimal logarithmic-scale increase (from 0.1% to 1%), the proportion of Short CoT patterns drops dramatically (from 40.95% to 13.24%). This indicates that even an extremely small proportion of Q → R mappings is sufficient to override the model's original Short CoT patterns, leading to an over-reliance on Long CoT reasoning.
this section cite: ['b17']

Section: Key Observation 2:
Current Long CoT models over-rely on Long CoT patterns. This is because the questions (Q) to Long CoT responses (R) mapping in SFT stage overrides their original Short CoT patterns.
this section cite: []

Section: Methodology: Question-Free Fine-Tuning

this section cite: []

Section: Motivation
To achieve adaptive reasoning, we aim for the model to ( 1) default to Short CoT patterns, while (2) adaptively triggering the reflective Long CoT behaviors when errors or uncertainty occur.
First, to preserve the model's default Short CoT patterns and prevent it from being overridden, we need to avoid training the model to learn a fixed Q → R mapping.
Second, the model should effectively learn Long CoT reasoning patterns, which could trigger reflective behaviors when the model is uncertain about its solution or encounters mistakes. Prior studies [6,23] suggest that the core of Long CoT patterns lies in the structure of responses, rather than in the questions. This implies that models distilled solely from Long CoT responses, even without access to the corresponding questions, can still acquire Long CoT reasoning capability.
this section cite: ['b5', 'b22']

Section: The Method
To preserve the original Short CoT patterns while enabling adaptive switching to reflective Long CoT patterns for challenging questions during distillation, we propose Question-Free Fine-Tuning (QFFT). Different from SFT, we remove the input question Q entirely, training the model exclusively on the reasoning response R. Specifically, the model is optimized using a standard causal language modeling objective over the reasoning sequence:
L QFFT = - 1 |R| t∈R log P θ (R t | R <t , Q), (2
)
where R is the set of token indices in the response sequence R and P θ is the model's output probability.
Q indicates that QFFT removes the question component compared to SFT.
By omitting the question Q during training and training solely on the reasoning response R, QFFT avoids learning a fixed mapping from questions to Long CoT responses while still acquiring Long CoT reasoning capability.
this section cite: []

Section: Why QFFT Leads to Adaptive Reasoning?
We explain why QFFT achieves adaptive reasoning from both training and inference stages.
Training: Preserving Short CoT and Learning Reflection. During the QFFT training, the model avoids learning a direct mapping from questions (Q) to Long CoT responses (R), thereby preserving the model's default Short CoT patterns to answer questions. Additionally, the model is trained exclusively on Long CoT responses. Thus, the model theoretically acquires the capability for Long CoT reasoning and learns to exhibit reflective behaviors when encountering uncertainty or errors within the context of Long CoT reasoning. Formally, let U L denote the event of encountering uncertainty or errors during Long CoT reasoning. Then, the model learns the conditional probability:
P θ (B r | U L ),
where B r denotes the occurrence of reflective behaviors, and U L represents uncertainty or errors arising specifically within the context of Long CoT reasoning.
Inference: Default to Short CoT with Adaptive Reflection. At inference time, the QFFT model defaults to the Short CoT patterns. However, since the model has only explicitly learned the conditional probability P θ (B r | U L ) during training, it is not immediately obvious why it can still trigger reflective behaviors in the context of Short CoT reasoning. We explain this phenomenon from a transfer learning [24,25] perspective:
Assumption 2 If a model has learned the conditional probability P θ (B r | U L ), this reflective capability can be transferred to Short CoT, enabling it to implicitly learn P θ (B r | U S ), where U S denotes uncertainty or errors that arise specifically within Short CoT reasoning.
We empirically verify this assumption with experiments in Appendix F.5. Consequently, when the model detects errors or uncertainty in Short CoT reasoning context, it naturally triggers reflective behaviors to reconsider and correct its reasoning process.
In summary, QFFT enables the model to reason in an adaptive manner: it defaults to efficient Short CoT patterns and adaptively uses reflective Long CoT patterns when necessary, thus achieving both effectiveness and efficiency.
this section cite: ['b23', 'b24']

Section: Experiments
In this section, we conduct experiments on both in-domain and out-of-domain datasets to validate the effectiveness of QFFT. Additional results (e.g. on different backbones) can be found in the appendix.
this section cite: []

Section: Experimental Setup
Dataset. We select several high-quality distillation datasets, including S1.1 (1k) [9], LIMO (871) [10], and Bespoke-Stratos (17k) [8], where all responses are distilled from the DeepSeek-R1 model. Detailed data description is shown in Appendix E.1.
Training details. We use Qwen2.5-Instruct-7B and Qwen2.5-Instruct-32B [26] as the base models.
All experiments are conducted with LLaMA Factory [27], with a maximum sequence length of 16,384 tokens. Detailed hyperparameters can be found in the Appendix E.2.
Evaluation. We evaluate model performance on six in-domain math datasets, including two simple datasets: GSM8K [28] and MATH500 [22], two medium-difficulty datasets: the American Mathematics Competitions (AMC23) and Minerva [29], which includes undergraduate-level STEM Long-to-Short Baselines. We compare with four competitive Long-to-Short baselines: (1) SFT-Shortest [11]: SFT on the shortest correct responses directly. (2) DPO-Shortest & SimPO-Shortest [11]: widely used baselines in reasoning optimization. Following the setup of the original work, the shortest correct response is selected as the preferred sample, and the longest correct response as the rejected one. (3) O1-Pruner [14]: a reinforcement learning-based method that reduces reasoning length while preserving accuracy. It first builds a baseline via pre-sampling, then guides the model to generate more concise reasoning under performance-preserving constraints. We reproduce their results based on the LIMO model, following the experimental setup and hyperparameters as described in their paper.
this section cite: ['b8', 'b9', 'b7', 'b25', 'b26', 'b27', 'b21', 'b28', 'b10', 'b10', 'b13']

Section: Efficiency Metric.
Following [14], we adopt the Accuracy-Efficiency Score (AES) to quantify the trade-off between model accuracy and token reduction. Let (A base , L base ) and (A model , L model ) denote the accuracy and token count for the baseline and evaluated models, respectively. In the Long-to-Short scenario, the baseline is the original Long CoT model (e.g., LIMO-7B); for QFFT, it is the model fine-tuned via SFT on the same dataset.
We define relative changes as:
∆L = L base -L model L base , ∆A = A model -A base A base .
The AES is then computed as:
AES = α ∆L + β |∆A|, ∆A ≥ 0, α ∆L -γ |∆A|, ∆A < 0,
with default parameters α = 0.1, β = 0.1, and γ = 1.0. The AES is calculated by weighting and summing the model's solution token length and accuracy. In this metric, we prioritize maintaining accuracy (i.e., avoiding performance degradation) over reducing token usage.
this section cite: ['b13']

Section: Main Results

this section cite: []

Section: Comparison with SFT
As shown in Table 2, we observe that across three mathematical evaluation datasets, QFFT methods significantly reduce the average token length while achieving performance comparable to that of SFT. In addition, QFFT substantially increases the RAK score, indicating improved adaptive reasoning capabilities. This suggests that QFFT effectively leverages Short CoT patterns when handling simple and solvable problems, reducing token consumption without sacrificing accuracy. A more detailed analysis of performance and reasoning adaptability is provided in the Section 5.
We further observe that the degree of token reduction varies with dataset difficulty. Specifically, QFFT achieves more substantial token savings on simpler datasets such as GSM8K and MATH, where the model adaptively retains a higher proportion of the Short CoT patterns. In contrast, on the more challenging AIME25 dataset, the model must rely more heavily on the Long CoT patterns, resulting in a less pronounced reduction in token consumption.
Finally, QFFT demonstrates strong adaptability across all three training datasets, including S1.1, LIMO, and BS-17K, when applied to both 7B and 32B model scales. This highlights the robustness and scalability of the QFFT approach.
Key Observation 3: QFFT achieves performance comparable to SFT, significantly improving Reasoning Adaptability (RAK), thus substantially reducing the required token budgets.
this section cite: []

Section: Comparison with Long-to-Short Baselines
As shown in Table 3, compared to Long-to-Short methods like SFT-Shortest and DPO, QFFT achieves greater token reduction with less performance degradation. On the other hand, while O1-Pruner and Simpo-FCS result in higher token reduction, they come at the cost of a significant drop in performance. In contrast, our QFFT method achieves higher AES values, providing a better trade-off between performance and token efficiency.
this section cite: []

Section: An In-depth Analysis
In this section, we conduct an in-depth analysis to address two key questions: Q1: How QFFT Enables Adaptive Reasoning? (Section 5.1) and Q2: Why QFFT Matches SFT Performance? (Section 5.2)
5.1 How QFFT Enables Adaptive Reasoning? (Q1) 1 2 3 4 5 Math Difficulty Level 0 500 1000 1500 Counts 150 392 549 749 667 196 304 596 157 234 368 Verification Backtracking Subgoal Setting Others Backward Chaining In Section 3.3, we have explained why QFFT enables adaptive reasoning. In this part, we aim to explore, in the inference stage, how QFFT models utilize Long CoT patterns. To this end, we first present a detailed case study illustrating the model's transition from the default Short CoT to the Long CoT reasoning pattern. Subsequently, we analyze the primary scenarios and conditions under which the model adaptively adopts Long CoT reasoning, providing deeper insights into QFFT's adaptive reasoning capabilities.
Case Study. We check some cases of the reasoning chains of QFFT. As illustrated in Fig- Figure 3: A case study. This case illustrates that QFFT initially adopts the Short CoT patterns. Upon encountering an error, the model triggers the reflective Long CoT to refine its reasoning. ure 3, we observe that in responses of the QFFT model, the reflective Long CoT patterns initially follow a concise Short CoT style as expected. Then, once the model encounters uncertainty or detects a potential error, it triggers Long CoT reasoning patterns, marked by frequent use of reflective keywords such as "wait".
Conditions Triggering Long CoT Reasoning. We further investigate under what conditions the QFFT model would switch to Long CoT behavior. According to [32], there are four main Long CoT behaviors: (1) Verification, verification or the systematic checking of intermediate results.
(2) Backtracking, which involves explicit revision of approaches when errors are detected. (3) Sub-goal Setting, where a complex problem is broken down into manageable steps. (4) Backward Chaining, where in a goal-directed reasoning task, the solution is derived by working backward from the desired outcome. Notably, adopting verification and backtracking behaviors suggests the model faces uncertainties or errors. The four behaviors identified in our case study are illustrated in Appendix F.4.
Then, we check responses generated by the QFFT model and observe that most Long CoT responses are triggered by uncertainties or errors. Specifically, we use GPT-4o to classify the triggered Long CoT behaviors in MATH into the four categories. We regard the first "wait" as the boundary between the two patterns, providing GPT-4o with two sentences before and after the first "wait" for classification. As shown in Figure 4, verification is the most common one across all difficulty levels, accounting for approximately 53% on average in Long CoT behaviors. This indicates the QFFT model tends to trigger Long CoT patterns when it is uncertain about the previous steps. The second most common trigger is backtracking, which accounts for 26% on average, that is, rechecking and updating previously incorrect steps. Interestingly, as the difficulty of the questions increases, the proportion of backtracking gradually increases, suggesting that the model reflects and updates its step more frequently in such questions.
this section cite: ['b31']

Section: Conclusion.
Our experiments above illustrate Long CoT reasoning in QFFT is mostly triggered by errors or uncertainties from the initial Short CoT patterns. The QFFT model defaults Short CoT reasoning and dynamically activates Long CoT patterns when these uncertainties and errors arise, demonstrating the reasoning adaptability.
this section cite: []

Section: References
Ref_id:b0 Title: Openai o1 system card Year: (2024)
Ref_id:b1 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b2 Title: Reasoning with large language models, a survey Year: (2024)
Ref_id:b3 Title: Code to think, think to code: A survey on code-enhanced reasoning and reasoning-driven code intelligence in llms Year: (2025)
Ref_id:b4 Title: Competitive programming with large reasoning models Year: (2025)
Ref_id:b5 Title: Llms can easily learn to reason from demonstrations structure, not content, is what matters! arXiv preprint Year: (2025)
Ref_id:b6 Title: Enhancing customer contact efficiency with graph neural networks in credit card fraud detection workflow Year: (2025)
Ref_id:b7 Title: Sky-t1: Fully open-source reasoning model with o1-preview performance in 450 budget Year: (2025)
Ref_id:b8 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b9 Title: Less is more for reasoning Year: (2025)
Ref_id:b10 Title: Do not think that much for 2+ 3=? on the overthinking of o1-like llms Year: (2024)
Ref_id:b11 Title: Thoughts are all over the place: On the underthinking of o1-like llms Year: (2025)
Ref_id:b12 Title: Kimi k1. 5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b13 Title: Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning Year: (2025)
Ref_id:b14 Title: L1: Controlling how long a reasoning model thinks with reinforcement learning Year: (2025)
Ref_id:b15 Title: Cot-valve: Length-compressible chain-of-thought tuning Year: (2025)
Ref_id:b16 Title: Controllable chain-of-thought compression in llms Year: (2025)
Ref_id:b17 Title: Understanding catastrophic forgetting in language models via implicit inference Year: (2024)
Ref_id:b18 Title: Towards large reasoning models: A survey of reinforced reasoning with large language models Year: (2025)
Ref_id:b19 Title: Towards widening the distillation bottleneck for reasoning models Year: (2025)
Ref_id:b20 Title: Note on cohen's kappa Year: (1989)
Ref_id:b21 Title: Ilya Sutskever, and Karl Cobbe. Let's verify step by step Year: (2023)
Ref_id:b22 Title: Demystifying long chain-of-thought reasoning in llms Year: (2025)
Ref_id:b23 Title: A survey of transfer learning Year: (2016)
Ref_id:b24 Title: A comprehensive survey on transfer learning Year: (2020)
Ref_id:b25 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b26 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b27 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b28 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b29 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b30 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b31 Title: Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars Year: (2025)
Ref_id:b32 Title: Data engineering for scaling language models to 128k context Year: (2024)
Ref_id:b33 Title: Structlm: Towards building generalist models for structured knowledge grounding Year: (2024)
Ref_id:b34 Title: Minicheck: Efficient fact-checking of llms on grounding documents Year: (2024)
Ref_id:b35 Title: Why rare diseases are an important medical and social issue Year: (2008)
Ref_id:b36 Title: Significance of the rare event in geology Year: (1967)
Ref_id:b37 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b38 Title: From system 1 to system 2: A survey of reasoning large language models Year: (2025)
Ref_id:b39 Title: Reasoning on a spectrum: Aligning llms to system 1 and system 2 thinking Year: (2025)
Ref_id:b40 Title: Towards system 2 reasoning in llms: Learning how to think with meta chain-of-though Year: (2025)
Ref_id:b41 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025-01)
Ref_id:b42 Title: Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset Year: (2025)
Ref_id:b43 Title: Unlocking efficient long-to-short llm reasoning with model merging Year: (2025)
Ref_id:b44 Title: Chain of draft: Thinking faster by writing less Year: (2025)
Ref_id:b45 Title: Concise thoughts: Impact of output length on llm reasoning and cost Year: (2024)
Ref_id:b46 Title: Token-budget-aware llm reasoning Year: (2024)
Ref_id:b47 Title: Longrecipe: Recipe for efficient long context generalization in large language models Year: (2024)
Ref_id:b48 Title: Processbench: Identifying process errors in mathematical reasoning Year: (2025)
