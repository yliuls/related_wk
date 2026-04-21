Title: When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs
Abstract: Reasoning-enhanced large language models (RLLMs), whether explicitly trained for reasoning or prompted via chain-of-thought (CoT), have achieved state-ofthe-art performance on many complex reasoning tasks. However, we uncover a surprising and previously overlooked phenomenon: explicit CoT reasoning can significantly degrade instruction-following accuracy. Evaluating 20+ models on two benchmarks: IFEval (with simple, rule-verifiable constraints) and ComplexBench (with complex, compositional constraints), we consistently observe performance drops when CoT prompting is applied. Through large-scale case studies and an attention-based analysis, we identify common patterns where reasoning either helps (e.g., with formatting or lexical precision) or hurts (e.g., by neglecting simple constraints or introducing unnecessary content). We propose a metric, constraint attention, to quantify model focus during generation and show that CoT reasoning often diverts attention away from instruction-relevant tokens. To mitigate these effects, we introduce and evaluate four strategies: in-context learning, selfreflection, self-selective reasoning, and classifier-selective reasoning. Our results demonstrate that selective reasoning strategies, particularly classifier-selective reasoning, can substantially recover lost performance. To our knowledge, this is the first work to systematically expose reasoning-induced failures in instructionfollowing and offer practical mitigation strategies.

Section: Introduction
Reasoning-enhanced large language models (RLLMs) have demonstrated remarkable success across a variety of tasks, including mathematical problem solving, planning, and multi-hop question answering [Guo et al., 2025, Achiam et al., 2023, Grattafiori et al., 2024, Xu et al., 2023, Zhou et al., 2022, Wu et al., 2024, Fu et al., 2022, Qi et al., 2024, Chae et al., 2024]. A central contributor to these advancements is chain-of-thought (CoT) prompting [Wei et al., 2022b, Nye et al., 2021, Joshi et al., 2023, Lanham et al., 2023], which explicitly encourages models to reason step-by-step prior to providing an answer. Recent prominent models, such as DeepSeek-R1 [Guo et al., 2025], Claude [Anthropic, 2025], and OpenAI's O-series [OpenAI, 2024], either incorporate CoT-style reasoning in their fine-tuning procedures or explicitly offer reasoning as an inherent capability. While CoT generally improves performance on complex reasoning tasks, it incurs higher computational cost due to longer outputs and increased inference latency. Moreover, its impact on more structured tasks-such as instruction following-remains underexplored. Instruction following, the ability to comply with user-specified constraints, is essential for alignment, safety, and practical usability of language models [Ouyang et al., 2022, Zhang et al., 2023]. This raises a natural question: Does explicit reasoning actually help a model follow instructions more accurately?
In this paper, we answer this question empirically and arrive at a surprising conclusion: reasoning via CoT can degrade a model's ability to follow instructions. To investigate this, we evaluate 20+ language models of varying sizes and training paradigms, including general-purpose models (e.g., Llama, Mixtral) and reasoning-tuned models (e.g., Claude 3.7, DeepSeek-R1), on two complementary instruction-following benchmarks: IFEval [Zhou et al., 2023] consists of prompts with simple, independently verifiable constraints (e.g., "write at least 400 words" or "mention AI three times"). In contrast, ComplexBench [Wen et al., 2024] includes instructions formed through compositional logic, combining multiple dependent constraints via operations like chaining, selection, and nesting. Across both datasets, we observe a consistent and often substantial accuracy drop when models are prompted with CoT. This finding is surprising, given that reasoning is typically expected to improve performance on many tasks even more challenging than instruction following.
To understand this phenomenon, we conduct two complementary analyses. First, we perform a large-scale manual case study of samples where reasoning notably affects performance. We find that reasoning helps in two common scenarios: (1) satisfying formatting or structural requirements, and (2) enforcing lexical constraints that override default tendencies. However, it often hurts performance by: (1) over-focusing on high-level content and neglecting simple constraints, or (2) introducing redundant or well-intentioned content that unintentionally violates constraints. We provide detailed examples and analyses of each scenario. Second, we investigate the impact of reasoning through an attention-based analysis. We propose a quantitative measure: constraint attention, which is based on attention scores directed toward constraint tokens within instructions. Visualizations of attention patterns during response generation consistently demonstrate reduced constraint awareness when CoT prompting is employed, an effect observed across different datasets, models, and layers. This shift in attention might explain, at least partially, why reasoning can diminish instruction adherence.
Following these insights, we propose and evaluate four methods to mitigate the adverse impacts of reasoning on instruction-following accuracy: (1) In-context learning, where we identify and correct typical reasoning-induced failures, incorporating these examples into prompts; (2) Self-reflection, prompting models to evaluate and adjust their reasoning processes and candidate responses; (3) Self-selective reasoning, allowing models to autonomously decide when reasoning is beneficial; and (4) Classifier-selective reasoning, where a trained classifier determines when reasoning is necessary. Among these methods, we find that classifier-selective reasoning strategies offer substantial gains across benchmarks, while self-reflection is particularly helpful for larger models and simple instructions. We thoroughly discuss the comparative strengths and limitations of these mitigation strategies in Section 5. In summary, our main contributions are listed as follows:
• We evaluate 20+ language models on comprehensive instruction-following benchmarks, revealing that explicit reasoning can negatively impact instruction-following capabilitiesfoot_0 . To our knowledge, we are the first to discover and systematically explore this phenomenon.
• We provide detailed analyses of reasoning-induced failures, categorizing scenarios where reasoning either helps or hinders, and quantify attention shifts that explain these performance drops.
• We propose and rigorously examine multiple mitigation strategies, including in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning, demonstrating their effectiveness and highlighting promising directions for future research.
2 Related Work
Chain-of-Thought and Reasoning LLMs. State-of-the-art large language models (LLMs) often leverage explicit reasoning capabilities, exemplified by models such as OpenAI's O-series [OpenAI, 2024], DeepSeek's R1 [Guo et al., 2025], and Anthropic's Claude [Anthropic, 2025]. These models are trained on datasets incorporating not just direct responses but also explicit reasoning processes known as Chain-of-Thought (CoT) [Wei et al., 2022b, Nye et al., 2021, Joshi et al., 2023, Lanham et al., 2023]. CoT prompting, which encourages step-by-step reasoning, has demonstrated significant success, particularly in domains requiring complex reasoning, such as mathematics [Guo et al., 2025, Achiam et al., 2023, Grattafiori et al., 2024, Xu et al., 2023, Zhou et al., 2022, Wu et al., 2024, Fu et al., 2022, Qi et al., 2024, Chae et al., 2024]. However, CoT can also introduce additional computational costs and may yield limited or no improvement in certain contexts [Kambhampati et al., 2024, Wang et al., 2024, Sprague et al., 2024]. Our study further explores this dual nature of reasoning, highlighting cases where it can negatively impact instruction-following performance.
Instruction Following. Instruction-following is essential for aligning language model outputs with user expectations, enabling the models to reliably execute user-specified tasks. Techniques such as instruction tuning (fine-tuning models on extensive datasets of instruction-response pairs) have been instrumental in fostering this capability [Ouyang et al., 2022, Wang et al., 2023, Longpre et al., 2023, Bai et al., 2022, Wei et al., 2022a, Chung et al., 2022, Muennighoff et al., 2023]. This ability is critical for bridging the gap between pre-training objectives and desirable human-aligned behaviors such as helpfulness and relevance [Radford et al., 2019, Brown et al., 2020]. Despite its importance, instruction-following remains challenging, especially when instructions involve complex, multifaceted requirements [Zhang et al., 2023, He et al., 2024, Gudibande et al., 2023, Kung and Peng, 2023, Heo et al., 2024]. To systematically evaluate LLM instruction adherence, benchmarks such as IFEval [Zhou et al., 2023] and ComplexBench [Wen et al., 2024] have been introduced. IFEval focuses on rule-verifiable, straightforward constraints, while ComplexBench assesses models on sophisticated instructions involving nested and dependent constraints. Further details and comprehensive analyses of these benchmarks will be provided in our experimental evaluations (Section 3).
this section cite: ['b9', 'b0', 'b7', 'b30', 'b6', 'b21', 'b4', 'b18', 'b12', 'b15', 'b1', 'b19', 'b20', 'b31', 'b33', 'b28', 'b19', 'b1', 'b18', 'b12', 'b15', 'b9', 'b0', 'b7', 'b30', 'b6', 'b21', 'b4', 'b13', 'b25', 'b23', 'b20', 'b16', 'b2', 'b5', 'b17', 'b22', 'b3', 'b31', 'b8', 'b14', 'b11', 'b33', 'b28']

Section: Experiments

this section cite: []

Section: Datasets and Evaluation Metrics
We use two benchmark datasets, IFEval and ComplexBench, to comprehensively evaluate the instruction-following capabilities of language models:
IFEval is a synthetic dataset consisting of 541 prompts, each associated with one to three verifiable constraints drawn from 25 types (e.g., word count, formatting, keyword usage). We adopt instructionlevel loose accuracy, which allows minor formatting deviations (e.g., markdown wrappers) to avoid penalizing responses for stylistic differences, thereby better reflecting practical robustness.
ComplexBench is a manually curated dataset designed to assess models on complex compositional instructions formed through four operations: And, Chain, Selection, and Nested. It contains 1,150 instructions and over 5,300 scoring questions, covering 4 constraint types across 19 dimensions (e.g., lexical, semantic, formatting). Evaluation combines rule-based and LLM-based assessments. For our experiments, we translated all scoring rules into English and manually verified them to construct a fully English-compatible version.
Evaluation Metrics: For both datasets, we report the proportion of constraints satisfied per instruction.
In ComplexBench, dependency logic applies: failure in a prerequisite constraint results in automatic failure of all dependent constraints.
this section cite: []

Section: Models
We evaluate a diverse set of models, including both closed-source models (e.g., Claude3.7-Sonnet) and reasoning-focused models (e.g., DeepSeek-R1, Qwen-R1-distilled variants). Our opensource selection spans parameter scales from 1B to 70B. All model inferences use a temperature of 0. Open-source models are run without quantization using 4 NVIDIA-H100-80GB GPUs. In addition to single-model CoT vs. non-CoT comparisons, we also evaluate paired variants under a controlled setting: base vs. reasoning-enabled counterparts (e.g., Qwen2.5-Instruct vs. Qwen2.5-Math; Qwen3 Base vs. Qwen3 Think; Claude-3.7-Sonnet vs. Claude-3.7-Sonnet-Think; DeepSeek-V3 vs.
DeepSeek-R1), summarized in Table 2.
this section cite: []

Section: CoT Prompting
We compare model behavior with and without Chain-of-Thought (CoT) reasoning. The CoT prompts instruct models to reason step by step before producing an answer (exact prompt provided in Appendix F). We then assess instruction-following performance based on the model's answer in each setting.
this section cite: []

Section: Results
Performance on IFEval and ComplexBench is reported in the first three columns of Table 1, with visual summaries provided in Figure 3. Notably, 13 out of 14 models experience performance degradation on IFEval when CoT prompting is applied, and all models show declines on ComplexBench. For instance, the accuracy of Llama3-8B-Instruct drops from 75.2% to 59.0%, a reduction of over 16 percentage points.
We further compare reasoning-enabled models with their corresponding base variants across nine pairs, including Claude3.7-Sonnet vs. Claude3.7-Sonnet-Think, DeepSeek-V3 vs. DeepSeek-R1, two Qwen2.5-Instruct vs. Qwen2.5-Math pairs (1.5B, 7B), and three Qwen3 Base vs. Think pairs (4B, 8B, 32B). These results, shown in Table 2, reveal that reasoning variants typically underperform their base counterparts on instruction-following. While these comparisons are not fully controlled-reasoning-tuned models may undergo additional training stages such as supervised fine-tuning or RL/RLHF-the decline remains evident across both benchmarks.
Overall, these findings uncover a surprising and underexplored vulnerability: explicit reasoning, while often helpful in complex tasks, can increase the likelihood of violating instruction constraints, thereby impairing instruction-following reliability.
this section cite: []

Section: Analysis
To better understand when and why reasoning degrades instruction-following, we conduct two analyses: (1) a manual case study examining when CoT helps or hurts constraint satisfaction, and (2) an attention-based analysis investigating how reasoning shifts model focus away from constraints during generation.
this section cite: []

Section: Case Study
We manually examined all 541 samples from IFEval and over 1,000 samples from ComplexBench, focusing on cases where CoT affected whether constraints were satisfied. For each case, we analyzed which constraints were impacted and why the reasoning-based answer either improved or degraded performance.
Due to the length of the instructions and responses, we present detailed illustrative examples in Appendix B. For each example, we include outputs from both the base model and the reasoningenabled model, showing their respective Answer and, for CoT, the full Thinking process. We also report the number of constraints satisfied and include a case analysis explaining how reasoning helped or harmed performance. Although we examined a large number of examples, the failure and success cases largely fall into four recurring patternsfoot_1 , which we summarize below:
Reasoning Helps ✓ 1. Formatting and Structural Adherence: Reasoning improves compliance with structural constraints, such as producing valid JSON, wrapping the output in double quotes, or following markdown syntax. 2. Lexical and Keyword Precision: Reasoning enhances adherence to lexical requirements, including inserting rare characters (e.g., the letter q six times), omitting final punctuation, or using exactly 15 capitalized words.
this section cite: []

Section: Reasoning Hurts ✗
IFEval Results 3. Over-Focusing on High-Level Content and Neglecting Simple Constraints: When multiple constraints are present, reasoning often emphasizes content planning at the expense of simpler mechanical constraints. Common issues include exceeding word count limits, failing to repeat prompts exactly, using capital letters in lowercase-only tasks, or appending unnecessary content after required phrases.
this section cite: []

Section: Baseline Mitigation Methods

this section cite: []

Section: Model
4. Introducing Unnecessary Content that Violates Constraints: Reasoning frequently inserts redundant or well-intentioned additions-such as explanations, translations, or emphasis-that break constraints. Typical violations include: inserting English text into "foreign language only" outputs, including commas in "no commas" tasks, appending commentary to quote-only responses, or exceeding limits on capitalized words.
this section cite: []

Section: Constraint-Aware Attention Analysis
To understand why reasoning may degrade instruction-following, we analyze whether models pay less attention to constraint-relevant parts of the prompt during response generation. In many failure cases, we observed that models neglect certain constraints, either by overemphasizing content planning or
Model Pair IFEval (%) ComplexBench (%) Claude-3.7-Sonnet vs. Claude-3.7-Sonnet-Think 90.6 / 90.2 69.8 / 69.0 DeepSeek-V3 vs. DeepSeek-R1 85.2 / 83.3 71.2 / 71.1 Qwen2.5-1.5B-Instruct vs. DeepSeek-R1-Distill-Qwen-1.5B 35.9 / 13.7 44.1 / 16.7 Qwen2.5-7B-Instruct vs. DeepSeek-R1-Distill-Qwen-7B 63.6 / 25.1 60.2 / 38.6 Qwen2.5-1.5B-Instruct vs. Qwen2.5-Math-1.5B 35.9 / 14.9 44.1 / 23.4 Qwen2.5-7B-Instruct vs. Qwen2.5-Math-7B 63.6 / 27.9 60.2 / 28.8 Qwen3-4B vs. Qwen3-4B-Think 85.0 / 69.5 63.8 / 59.3 Qwen3-8B vs. Qwen3-8B-Think 86.8 / 86.3 65.6 / 59.8 Qwen3-32B vs. Qwen3-32B-Think 87.8 / 85.0 70.2 / 63.1 Table 2: Comparison of reasoning-enabled models and their non-reasoning counterparts across both benchmarks. Each cell reports accuracy on IFEval and ComplexBench, with green marking the higher-performing model and red the lower-performing one. Results generally show that explicit reasoning either provides negligible gains or causes small performance drops. introducing irrelevant information. To investigate this phenomenon systematically, we conduct an attention-based analysis that tracks the model's focus on constraint tokens throughout generation.
We define a constraint-attention metric to quantify the model's awareness of constraint-relevant tokens. For each instruction, we first use GPT-4o to automatically extract substrings corresponding to each constraint, and then map them to token indices in the prompt. During generation, we compute attention scores directed toward these tokens for both the reasoning and answer segments. Each model is run twice per instruction: (i) Base run: Instruction → Answer, and (ii) Reasoning run (CoT): Instruction → Think → Answer. We focus our comparison on the answer segments of both runs.
this section cite: []

Section: Transformer Attention.
Let the prompt contain T 0 tokens x 1:T0 , and let the model generate T new tokens y 1:T . At generation step t (1 ≤ t ≤ T ), the visible context is (x 1:T0 , y 1:t-1 ). Let h (l)
t-1 denote the hidden state of the most recent token at layer l, and let k ∈ {1, . . . , H} index attention heads. The attention components for layer l and head k are computed as:
Q (l,t) k = W (l,k) Q h (l) t-1 , K (l) k = W (l,k) K H (l) 1:T0+t-2 , V (l) k = W (l,k) V H (l) 1:T0+t-2 , where h (l)
t-1 is a single vector (current token), and H (l) 1:T0+t-2 is the matrix of all prior hidden vectors (prompt and generated context). This leads to scaled attention weights
A (l,t) k = softmax Q (l,t) k K (l)⊤ k √ d k
. Averaging over all heads yields the layer-level attention vector
a (l,t) = 1 H H k=1 A (l,t) k .
Constraint Attention. For each textual constraint c r (r = 1, . . . , R), we collect the prompt token indices it spans, yielding the index subset C r ⊆ {1, . . . , T 0 }. Define the full constraint token set as C = R r=1 C r . Then we can define the layer-step constraint attention (averaged attention to constraint tokens at each layer and step) as
α (l,t) = 1 |C| j∈C a (l,t) j .
Following this, we compute the layer-averaged constraint attention at step t as:
ᾱ(t) = 1 L L-1 l=0 α (l,t) ,(1)
Based on Equation 1, we visualize the trace of constraint-attention during response generation in Figure 1, showing results for both IFEval and ComplexBench using Qwen2.5-1.5B-Instruct.
Additional examples for other models are provided in Appendix C. After reviewing hundreds of samples, we observe a general trend: reasoning flattens the constraint-attention trace. In cases where reasoning degrades performance, constraint attention during the answer phase is generally lower. In contrast, when reasoning improves performance, we often see a bump in attention aligned with the answer segment.
this section cite: []

Section: Quantifying Attention Drop.
To further quantify these observations, we compute the mean constraint attention across the answer phase. Let A denote the answer token positions. The average constraint attention at layer l is:
β(l) = 1 |A| t∈A α (l,t)(2)
We define the attention drop as the difference between base and CoT runs:
∆β = βBase -βCoT .
We find that ∆β > 0 for most cases. Figure 2 shows how this drop varies across layers for WIN vs. LOSE cases. On average, the model exhibits larger attention drops in LOSE cases, particularly in early-to-middle layers. This suggests that lower constraint attention is predictive of reasoninginduced failures.
Conclusion. Our analysis reveals that explicit reasoning often reduces attention to constraintrelevant parts of the prompt. This diminished awareness increases the risk of violating instructions. Thus, while reasoning is intended to improve task performance, it can unintentionally shift model focus away from critical constraints and harm instruction adherence. In addition, we also explored whether longer reasoning tends to degrade performance and found that reasoning length does not meaningfully correlate with instruction-following effectiveness (Appendix J).
this section cite: []

Section: Mitigating Reasoning-Induced Failures in Instruction Following
We introduce and evaluate four strategies designed to mitigate the performance degradation caused by explicit reasoning (via CoT) in instruction-following tasks. To the best of our knowledge, this is the first work to systematically identify and address this issue. Hence, in the absence of existing baselines, we directly compare the effectiveness of our proposed methods. Results are presented in Table 1 and Figure 3, which also include performance differences relative to the CoT baseline.
this section cite: []

Section: Method 1: Few-Shot In-Context Learning
We apply in-context learning [Brown et al., 2020] by prepending carefully selected few-shot examples to each instruction. These examples are derived from representative failure cases identified in our case study (Section B) and manually revised to fully satisfy all constraints. Each example includes an Instruction, along with a corrected Thinking and Answer. Full examples are provided in Appendix E.
Results: As shown in Table 1, this method yields only modest improvements. We attribute the limited gains to several factors. First, due to the token length limit and the substantial size of each example (including prompt, reasoning, and answer), we were only able to include a limited number of examples: four examples for IFEval and three for ComplexBench. Second, because examples were sourced from outputs of certain models, potentially introducing bias. These constraints likely reduce the effectiveness of the few-shot strategy and limit its generalizability.
this section cite: ['b3']

Section: Method 2: Self-Reflection
This method performs a two-step process: the model first generates an initial response with thinking, and then performs a second inference where it reflects on its own reasoning and answer. If the model deems the initial response satisfactory, it retains it as the final output; otherwise, it revises the response and outputs the updated version. The prompt used for self-reflection is detailed in Appendix F.
Results: As shown in Table 1, self-reflection yields strong improvements on IFEval-enhancing performance in 11 out of 14 models, and achieving the best results among all four methods for 7 models. However, we observe notable performance degradation on weaker models such as Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct. We hypothesize that this is because self-reflection relies on a model's ability to critique and improve its own outputs, a capability that may be underdeveloped in weaker models. On ComplexBench, which contains more challenging and compositional instructions, self-reflection proves less effective-leading to performance drops in 10 out of 14 models. This suggests that self-reflection is more suitable for simpler instruction-following tasks, and may be counterproductive when applied to complex scenarios. Additionally, a key limitation of this method is its increased computational cost, as it requires two forward passes per query.
this section cite: []

Section: Method 3: Self-Selective Reasoning
This method enables the model to decide dynamically whether to perform explicit reasoning. Specifically, we prompt the model to assess, based on the instruction alone, whether reasoning (via CoT) is necessary (see prompt in Appendix F). If the model deems reasoning helpful, it proceeds with step-by-step thinking; otherwise, it directly generates an answer without reasoning.
Results: This approach yields moderate gains on IFEval, improving performance in 10 out of 14 models, and shows stronger results on ComplexBench, improving all models and achieving the best performance among all four methods in 6 models. In Appendix G, we further analyze the model's decision-making behavior by comparing its self-selected reasoning decisions to ground-truth labels based on actual performance differences between CoT and non-CoT responses. We find that while the model tends to have high recall (correctly identifying most cases where reasoning helps) it suffers from low precision, often applying reasoning even when it is not necessary.
this section cite: []

Section: Method 4: Classifier-Selective Reasoning
Instead of relying on the model's internal judgment, this method uses an external binary classifier to determine whether CoT reasoning should be applied. For each target model, we train a separate classifier to predict whether using CoT leads to improved instruction-following performance. Labels are assigned on a per-sample basis by comparing the constraint satisfaction scores of CoT and non-CoT responses. Training details are provided in Appendix H.
The classifier is implemented using Qwen2.5-7B-Instruct as the backbone and trained for 3 epochs with a learning rate of 1e-5. Both the backbone model and training hyperparameters are selected via grid search. We split the dataset evenly, using 50% of the samples for training and the remaining 50% to evaluate downstream mitigation effectiveness. Validation accuracy typically ranges between 0.75 and 0.92.
Results: This method proves highly effective, improving performance for nearly all models on both benchmarks, with only a minor drop for DeepSeek-V3 on IFEval. For about half of the models, it achieves the best overall performance among the four strategies. Nonetheless, its primary drawback is the model-specific training requirement: each model needs its own classifier, which results in additional overhead.
this section cite: []

Section: Summary and Recommended Pipeline
Each mitigation strategy exhibits distinct strengths and weaknesses depending on model capacity and instruction complexity. Based on our findings, we propose the following decision pipeline: first, estimate the complexity of the instruction-either through simple heuristics or a trained classifier. For simpler tasks (e.g., IFEval), we recommend Self-Reflection or Classifier-Selective Reasoning; for more complex or compositional tasks (e.g., ComplexBench), Self-Selective Reasoning or Classifier-Selective Reasoning is more effective. Overall, Classifier-Selective Reasoning consistently delivers the best overall performance across both benchmarks, albeit at the cost of model-specific training.
this section cite: []

Section: Conclusion
One limitation of our study is its exclusive focus on instruction-following tasks. While we suspect that reasoning could similarly degrade performance in other domains, exploring these possibilities is left for future work. In our study, we identified and systematically explored an unexpected phenomenon: explicit reasoning through Chain-of-Thought prompting can negatively impact the instruction-following abilities of large language models. Through extensive evaluation across two comprehensive benchmarks (IFEval and ComplexBench), we demonstrated consistent performance degradation when models employed explicit reasoning. Our detailed analysis, involving manual case studies and attention-based examinations, provided insights into why reasoning negatively affects instruction adherence. We found that reasoning can divert the model's focus from constraint-related tokens, resulting in overlooked or violated instructions.
To address this issue, we introduce and evaluate four mitigation strategies: in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning. Our experiments show that selective reasoning, especially classifier-based approaches, can substantially recover lost performance, with classifier-selective reasoning achieving the most consistent improvements across both datasets.
We believe this is the first systematic investigation into reasoning-induced failures in instructionfollowing tasks. We hope our findings motivate further investigation into reasoning tradeoffs and contribute to building models that reason more selectively and effectively.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Claude 3.7 sonnet and claude code Year: (2025-02)
Ref_id:b2 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b3 Title: Language models are few-shot learners Year: (2020)
Ref_id:b4 Title: Language models as compilers: Simulating pseudocode execution improves algorithmic reasoning in language models Year: (2024)
Ref_id:b5 Title: Scaling instruction-finetuned language models Year: (2022)
Ref_id:b6 Title: Complexity-based prompting for multi-step reasoning Year: (2022)
Ref_id:b7 Title: The llama 3 herd of models Year: (2024)
Ref_id:b8 Title: False sense of understanding in language models Year: (2023)
Ref_id:b9 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b10 Title: From complex to simple: Enhancing multi-constraint complex instruction following ability of large language models Year: (2024)
Ref_id:b11 Title: Do llms" know" internally when they follow instructions? arXiv preprint Year: (2024)
Ref_id:b12 Title: Are machine rationales (not) useful to humans? measuring and improving human utility of free-text rationales Year: (2023)
Ref_id:b13 Title: Position: Llms can't plan, but can help planning in llm-modulo frameworks Year: (2024)
Ref_id:b14 Title: Instruction tuning models don't generalize well to unseen tasks Year: (2023)
Ref_id:b15 Title: Measuring faithfulness in chain-of-thought reasoning Year: (2023)
Ref_id:b16 Title: The flan collection: Designing data and methods for effective instruction tuning Year: (2023)
Ref_id:b17 Title: Crosslingual generalization through multitask finetuning Year: (2023)
Ref_id:b18 Title: Show your work: Scratchpads for intermediate computation with language models Year: (2021)
Ref_id:b19 Title: Introducing openai o1 Year: (2024-09)
Ref_id:b20 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b21 Title: Mutual reasoning makes smaller llms stronger problem-solvers Year: (2024)
Ref_id:b22 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b23 Title: To cot or not to cot? chainof-thought helps mainly on math and symbolic reasoning Year: (2024)
Ref_id:b24 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2023)
Ref_id:b25 Title: Mmlu-pro: A more robust and challenging multitask language understanding benchmark Year: (2024)
Ref_id:b26 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b27 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b28 Title: Benchmarking complex instruction-following with multiple constraints composition Year: (2024)
Ref_id:b29 Title: A comparative study on reasoning patterns of openai's o1 model Year: (2024)
Ref_id:b30 Title: Re-reading improves reasoning in language models Year: (2023)
Ref_id:b31 Title: Instruction tuning for large language models: A survey Year: (2023)
Ref_id:b32 Title: Least-to-most prompting enables complex reasoning in large language models Year: (2022)
Ref_id:b33 Title: Instruction-following evaluation for large language models Year: (2023)
