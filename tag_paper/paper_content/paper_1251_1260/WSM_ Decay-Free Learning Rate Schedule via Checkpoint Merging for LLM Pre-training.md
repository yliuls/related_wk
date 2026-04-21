Title: WSM: DECAY-FREE LEARNING RATE SCHEDULE VIA CHECKPOINT MERGING FOR LLM PRE-TRAINING
Abstract: Recent advances in learning rate (LR) scheduling have demonstrated the effectiveness of decay-free approaches that eliminate the traditional decay phase while maintaining competitive performance. Model merging techniques have emerged as particularly promising solutions in this domain. We present Warmup-Stable and Merge (WSM), a general framework that establishes a formal connection between learning rate decay and model merging. WSM provides a unified theoretical foundation for emulating various decay strategies-including cosine decay, linear decay and inverse square root decay-as principled model averaging schemes, while remaining fully compatible with diverse optimization methods. Through extensive experiments, we identify merge duration-the training window for checkpoint aggregation-as the most critical factor influencing model performance, surpassing the importance of both checkpoint interval and merge quantity. With the high-quality annealing data, our framework consistently outperforms the widely-adopted Warmup-Stable-Decay (WSD) approach across multiple benchmarks, achieving significant improvements of +3.5% on MATH, +2.9% on HumanEval, and +5.5% on MMLU-Pro. The performance advantages extend to supervised fine-tuning scenarios, highlighting WSM's potential for long-term model refinement. * Equal contribution. ‡ Contribution during internship at Ant Group.

Section: INTRODUCTION
In large language model (LLM) pre-training, learning rate (LR) scheduling plays a pivotal role, critically impacting training stability, convergence speed, and final model performance (Jin et al., 2023;Gotmare et al., 2019). Conventional LR schedules dynamically adjust the LR based on training  (Loshchilov & Hutter, 2016) and WSD (Hu et al., 2024).
Following the warm-up, the LR gradually decreases according to a predefined function, such as cosine, linear, or inverse square root decay. These schedules share a critical limitation: they require the total number of training tokens (or steps), T max , to be known in advance. For instance, the prevalent cosine LR schedule is formulated as:
lr(t) = lr peak • t Twarmup if t < T warmup 1 2 lr peak 1 + cos π(t-Twarmup) Tmax-Twarmup if t ≥ T warmup
The Warmup-Stable-Decay (WSD) LR schedule introduces a stable phase between warm-up and decay, maintaining a constant LR at its peak (lr peak ), which is defined as:
lr(t) =      lr peak • t Twarmup if t < T warmup lr peak
if T warmup ≤ t < T decay start decay function(t) if T decay start ≤ t ≤ T max As shown, WSD facilitates flexible experimentation by supporting multiple decay attempts from the endpoint of the stable phase-without requiring a reset to the initial state. Despite this flexibility, it still requires predefined decay-phase settings-such as the decay start step (T decay start ), decay function (decay function), and total training steps (T max ).
this section cite: ['b26', 'b13', 'b43']

Section: THE PROPOSED METHODOLOGY
In this section, we first establish the theoretical connection between checkpoint merging and LR decay, formalize our proposed WSM (Warmup-Stable and Merge) schedule, and compare it with the widely used WSD schedule.
this section cite: []

Section: THEORETICAL CONNECTION BETWEEN LR DECAY AND CHECKPOINT MERGING
The core idea of checkpoint merging in this work is to take an ordered list of checkpoints, [θ n , θ n+1 , . . . , θ n+k ], and apply a merge function to generate a single model θn+k . Here, θ i ∈ R d represents the model's parameter vector at the i-th training iteration. The most general form is a weighted average of the checkpoints:
θn+k = k j=0 c j θ n+j(1)
where {c j } are non-negative weights that sum to one, i.e., k j=0 c j = 1. This formulation obscures a deeper connection to the training dynamics. We can reveal this connection by expressing each checkpoint in terms of an initial checkpoint θ n and the subsequent gradient updates. For simplicity, we assume the updates between checkpoints at different time steps are independent and ignore optimizer states. Let g i be the gradient update vector (including the LR) at step i, such that the model updates as θ i+1 = θ i -g i . Thus, an intermediate checkpoint θ n+j can be expressed as the sum of an initial state θ n and the sequence of negative gradient updates that followed:
θ n+j = θ n - j l=1 g n+l-1
(2) Furthermore, we can substitute Eq. 2 into the general merging formula (Eq. 1) and then rearrange the double summation by changing its order. A gradient update g n+i-1 is included in the sum for all involved checkpoints θ n+j where j ≥ i.
θn+k = k j=0 c j θ n - j l=1 g n+l-1 = θ n - k i=1   k j=i c j   g n+i-1(3)
0% 20% 40% 60% 80% 100% Training Progression (t) 0.0 0.2 0.4 0.6 0.8 1.0 Decay Value w(t) = 1 0.1 1 t 0% 20% 40% 60% 80% 100% Training Progression (t) 0.0 0.2 0.4 0.6 0.8 1.0 w(t) = 1 t 0% 20% 40% 60% 80% 100% Training Progression (t) 0.0 0.2 0.4 0.6 0.8 1.0 w(t) = 1 + cos( t) 2 0% 20% 40% 60% 80% 100% Training Progression (t) 0.0 0.2 0.4 0.6 0.8 1.0 w(t) = 1 t 0 1 2 3 4 5 6 7 8 9 Checkpoint Index (a) : EMA 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 Weight Value 0.126 0.033 0.041 0.052 0.065 0.082 0.103 0.130 0.163 0.206 Merge Weights 0 1 2 3 4 5 6 7 8 9
Checkpoint Index This shows that a weight coefficient of k j=i c j is applied to the gradient update g n+i-1 from step n + i -1. By defining a new set of weights for the gradient updates, w i = k j=i c j , we arrive at the final equivalent expression for checkpoint merging:
θn+k = θ n - k i=1 w i • g n+i-1(4)
This equation shows that merging checkpoints with weights c j is equivalent to applying a synthetic decay schedule defined by weights w i to the gradients accumulated after the base checkpoint θ n , where a mapping exists between the merge weights c j and the effective learning rates w i . We therefore propose Theorem 3.1, which can approximate monotonically decreasing decay curve. Figure 2 illustrates various checkpoint merging weights and their corresponding decay curves. Additional details are provided in Appendix B. Theorem 3.1 (Checkpoint Weight Derivation from Gradient Decay Schedule). Given a desired sequence of gradient decay coefficients {w i } k i=1 that is monotonically non-increasing and bounded, 1 ≥ w 1 ≥ w 2 ≥ • • • ≥ w k ≥ 0, the corresponding non-negative checkpoint weights {c j } k j=0 required to satisfy Eq. 4 are uniquely determined by:
     c k = w k c j = w j -w j+1 , for j ∈ [1, k -1] c 0 = 1 - k j=1 c j = 1 -w 1 (5) 3.2 EMULATING LR DECAY THROUGH CHECKPOINT MERGING
The central hypothesis of WSM is that the optimization benefits of LR decay can be decoupled from the live training process and instead be effectively achieved through the merging of model checkpoints (Kingma & Ba, 2015;DeepSeek-AI et al., 2024;Li et al., 2025). The WSM simplifies the LR schedule by completely eliminating the decay phase:
lr(t) = lr peak • t Twarmup if t < T warmup lr peak if t ≥ T warmup
The WSM pipeline, detailed in Algorithm 1, operates in two primary phases. It begins with a standard warmup phase, where the learning rate linearly increases to its peak value, lr peak . Following this, the process enters the main stable training phase, where the learning rate is held constant.
After a specified step T switch , the model can transition from the general pre-training data D to a smaller, high-quality annealing dataset D anneal , allowing the "annealing" to focus on a curated data mixture. Throughout this stable phase, checkpoints are periodically saved. Concurrently, an asynchronous merging process continuously fetches the last n checkpoints from storage and combines them into model W merged . Specifically, for the Merge(•) operation, we can select various decay strategies to emulate (e.g., the decay curve shape and minimum LR), calculate the corresponding gradient decay coefficients {w i }, and then derive the checkpoint merging weights {c i } based on Theorem 3.1. This merged checkpoint, which emulates the effect of a decay schedule, provides a robust, annealed model without ever altering the live learning rate.
4 EXPERIMENT Next, we present the empirical evaluation of our proposed WSM to validate its effectiveness.
this section cite: ['b32', 'b25']

Section: EXPERIMENT SETUP
The model we used for the main experiment is a standard MoE model with a total of 16.3 billion parameters and 1.4 billion active parameters. We utilized the AdamW optimizer (Loshchilov & Hutter, 2019), and the hyperparameters are set to β 1 = 0.9 and β 2 = 0.95, with 0.1 weight decay applied. Through preliminary scaling laws experiments, we set the peak LR and batch size to 4.78e-4 and 2048, respectively. Comprehensive details regarding our model architecture, specific training parameters, dataset composition, and evaluation protocols are provided in Appendix A.
We begin with a checkpoint pretrained on 10.2 trillion tokens from the pretraining dataset using a stable, constant LR. Then, we continue training for an additional 400B tokens on a specialized high-quality annealing dataset, following two distinct strategies to evaluate their relative effectiveness: (1) We apply a conventional LR decay schedule to the model. This branch serves as our baseline, representing the standard Warmup-Stable-Decay (WSD) methodology. Note that we employ the standard re-warm-up strategy for continual pre-training as in a standard WSD setting. (2) We continue training with the same constant LR. The final model is then produced by merging the checkpoints saved during this stable phase. This branch represents our proposed WSM schedule. Unless otherwise specified, we save a checkpoint every 25B tokens and use mean averaging to merge the most recent checkpoints, which corresponds to a linear decay LR schedule.
this section cite: ['b44']

Section: OVERALL PERFORMANCE OF WSM SCHEDULE
Effectiveness in pre-training We evaluate our WSM schedule against the baseline WSD using three series of checkpoints: (1) those obtained using the standard LR decay schedule in WSD, (2) checkpoints from the last stable phase of WSM before merging, and (3) our final merged checkpoints from the WSM method using various merge durations (window sizes). The category-wise average results are summarized in Figure 3 and Table 1, and scores for each benchmark are provided in Appendix J.
Our first and most significant finding is that the WSM method consistently outperforms WSD across the majority of tasks considered. On average, WSM achieves performance improvements across all benchmark categories. Notably, when comparing the best-performing checkpoint from each strategy, WSM improves upon WSD by an average of 1.3 points. We observe remarkable improvements of up to 2.7 points on MATH, 2.4 points on HumanEval, 2.1 points on MMLU-Pro. These results provide compelling evidence that replacing the LR decay phase used in WSD with the checkpoint merging strategy of WSM is not only a feasible alternative but also a more effective approach for enhancing the diverse capabilities of the final pretrained model.
Long-term effectiveness for post-training To assess the long-term impact of WSM, we extended our evaluation to the post-training phase. We apply supervised fine-tuning on checkpoints generated by the WSM and WSD under identical settings for 5 epochs. Results in Table 2 show that this advantage persists beyond the post-training phase.
this section cite: []

Section: EMPIRICAL ANALYSIS OF WSM SCHEDULE
Next, we conduct a comprehensive empirical study to dissect the WSM schedule. We aim to understand the key factors influencing its performance, its robustness across different training scenarios, its interaction with conventional decay schedule, and its broader implications on model dynamics.
this section cite: []

Section: ROBUSTNESS ACROSS PRE-TRAINING PROCESS
Beyond applying WSM as a final step on a high-quality dataset, we also evaluated its utility and robustness throughout the entire pre-training lifecycle. To achieve this, we conducted a comparative analysis at various intermediate stages of a long training run, comparing the performance of a model produced by our computationally-frugal WSM merging against one produced by initiating a full, resource-intensive LR decay. As shown in Figure 5a, although the performance gains over WSD are not as significant as when switching to high-quality data, we find that the performance of models generated via WSM merging consistently and closely mirrors the results of a true LR anneal. In the figure, the gray line represents the base model trained with a constant LR. The blue lines show the performance of WSM models, which were created by mean-merging four checkpoints from within the preceding 100B-token merge duration. The red lines represent full 100B-token decay runs initiated at the 2T, 4T, 6T, 8T, and 10T token milestones. This result demonstrates that the incorporation of high-quality annealing data is the key factor enabling WSM to outperform WSD, while simultaneously establishing WSM as a reliable, high-fidelity proxy for estimating a model's post-anneal potential at any point during training. Consequently, it can provide effective assessment throughout the pre-training phase, eliminating the need to launch multiple, expensive decay runs to determine the model's ultimate performance.
this section cite: []

Section: IMPACT OF MERGING ALGORITHM
As derived in Section 3.1, the checkpoint merging process can be viewed as an approximation of a LR decay schedule, where the weighting scheme of the merge is analogous to the functional form of the decay curve. For instance, a simple mean average is analogous to a linear decay curve.
An Exponential Moving Average (EMA) would correspond to a convex exponential decay curve. Existing works (Hägele et al., 2024) and our prior experiments (details are provided in Appendix F) with the WSD schedule revealed a performance hierarchy among decay curves: concave schedules (e.g., inverse square root) and linear schedules outperform convex schedules. Building on these findings, we hypothesize that a merging algorithm designed to approximate decay curves that have been proven effective in WSD scheduling will similarly yield better results.
Based on Theorem 3.1, we experimentally compare three merging algorithms: one using our theorem-generated weights to approximate 1-sqrt decay, another using simple mean averaging, and a third using EMA. Our experimental results in Table 3 validate this hypothesis. While the merge method outperforms decay, we observe a distinct performance ranking: the 1-sqrt merge approach outperforms Mean, and both are markedly better than EMA. Crucially, this ranking (1-sqrt > Mean > EMA) is identical to the hierarchy observed in standard LR decay schedules. This consistency demonstrates that WSM reliably reproduces the relative effectiveness of different decay schedules, confirming that checkpoint merging serves as a principled and effective simulation of the LR decay process.
this section cite: ['b16']

Section: IMPACT OF MERGING DURATION AND GRANULARITY
We further investigate in detail the impact of different merge durations (window sizes) on various algorithms. First, different merge durations essentially correspond to decaying with varying amounts of data. As shown in Figure 4, when comparing the best-performing checkpoints across the entire merging trajectory for both mean and 1-sqrt merging algorithms, larger merging windows tend to yield better results. However, this advantage gradually diminishes as the window size increases. This observation aligns with previous practical LR decay experiments, where simply increasing the amount of annealing data shows diminishing returns and may eventually fail to provide further improvement. For EMA merging, its performance is significantly inferior to other algorithms and shows no clear variation with merge durations. This may indicate that EMA is not an effective merging algorithm (also suggesting that the convex nature may not represent an optimal decay curve).
Then we further investigate the granularity of checkpoint merging, i.e., the interval between saved checkpoints used for merging. Finer-grained merging represents a more precise approximation of the true decay curve. The results in Table 4 show that finer-grained merging tends to achieve better performance. However, frequent checkpoint saving imposes storage overhead, requiring careful trade-off considerations.  Given that checkpoint merging effectively simulates LR decay, a natural question arises: can merging and decay be combined to achieve synergistic performance gains? We investigate this by testing two hybrid approaches. (1) Decay-then-Merge: We first apply a standard decay schedule and then merge checkpoints selected from within the decay phase.
(2) Merge-then-Decay: We further apply a decay schedule to the resulting merged model. As shown in Figures 5b and 5c, the hybrid approach failed to yield any improvement in either configuration, although the Merge-then-Decay model showed better performance at the beginning of its training. For the Decay-then-Merge experiment (Figure 5b), the blue stars represent the results of merging checkpoints selected along the decay trajectories (red lines), which were initiated at various pre-training milestones (4T, 6T, 8T, 10T). For the Merge-then-Decay experiment (Figure 5c), we compare a decay run initiated from a WSM model (blue line)-created by merging four checkpoints from the 9.8T to 10T token interval-against a standard decay baseline initiated from a single 10T checkpoint (red line). These results suggest that checkpoint merging and LR decay are not complementary but rather alternative pathways to a similar optimization objective.
this section cite: []

Section: IMPACT ON MOE LOAD BALANCING
Table 5: Impact on MoE load balancing. The WSM strategy demonstrates improved expert utilization (lower load balancing violation scores) with a slightly higher test language modeling loss. The mean global max violation represents the average of the highest violations across all layers (measuring the severity of "overloaded" experts), while mean global min violation averages the violations for the least-utilized experts (measuring the risk of "routing collapse").
language modeling loss mean global max violation mean global min violation WSD 0.675 0.601 0.322 WSM 0.697 0.545 0.201
We provide an analysis of the implications of WSM on MoE router balance in Table 5. Specifically, the violation for a single expert is calculated as its relative deviation from the average load within its layer 1 . When comparing the merged checkpoint with a decayed checkpoint, the merged checkpoint achieves more balanced routing, although its loss is slightly higher. We argue that this trade-off-a marginally higher loss for superior downstream performance-is indicative of enhanced generalization rather than overfitting to the training data.
5 RELATED WORK
this section cite: []

Section: LEARNING RATE SCHEDULE
Learning rate (LR) scheduling is critical for training performant models (Jin et al., 2023;Gotmare et al., 2019). Classic schedules like Cosine (Kaplan et al., 2020;Ibrahim et al., 2024) or Linear (Defazio et al., 2023) decay adjust the LR based on a predefined total training duration, which is inflexible for continual training. The Warmup-Stable-Decay (WSD) schedule (Hu et al., 2024) addresses this by introducing a stable LR phase after warmup, decoupling the eventual decay from a fixed training length and offering greater flexibility for long or continuous training runs. More recently, researches have explored "schedule-free" methods to eliminate the decay phase entirely, which maintain a constant LR and instead leverage weight averaging techniques (Defazio et al., 2024;Song et al., 2025;Zhang et al., 2025). Builds upon these schedule-free principles, our work propose to replace WSD's decay phase with a post-hoc checkpoint merging operation instead of specific online averaging strategies. This simplifies the training pipeline and, by formalizing the connection between LR decay and checkpoint merging, allows decay strategy to be theoretically approximated, leading to free offline exploration and enhanced model performance.
this section cite: ['b26', 'b13', 'b31', 'b23', 'b10', 'b11', 'b57', 'b70']

Section: MODEL MERGING
Model merging (Izmailov et al., 2018;Wortsman et al., 2022) has emerged as an efficient paradigm for model construction. This approach achieves effective knowledge transfer and performance improvement through parameter-level integration of multiple models. Model merging is primarily utilized in two distinct scenarios: (1) The integration of knowledge and capabilities from multiple independently trained models into a single parameter set, with the objective of preserving maximal performance from each specialized model (Aakanksha et al., 2025;Ramé et al., 2024); and (2) the merging of checkpoints along a single training trajectory. This second category is formally established in the literature as Stochastic Weight Averaging (SWA) (Izmailov et al., 2018). SWA func-1 For a single layer, let L be the vector of token loads for its experts and µ = mean(L), max violation = |max(L)-µ| µ and min violation = |min(L)-µ| µ tions as a practical realization of Polyak averaging (Polyak & Juditsky, 1992), acting as a smoothing mechanism that reduces the noise inherent in stochastic gradient-based optimization (Sanyal et al., 2023;Liu et al., 2024a;Kaddour, 2022;Li et al., 2023;Sandler et al., 2023;Hägele et al., 2024).
While Li et al. (2025) show model merging can achieve performance competitive with decay-based schedule, these techniques have also demonstrated practical utility in industrial-scale LLM development (Grattafiori et al., 2024;DeepSeek-AI et al., 2024;Aakanksha et al., 2025).
Concurrent with our work, the ERNIE 4.5 (ERNIE-Team, 2025) explored the relationship between EMA and LR decay. Our work generalizes this by introducing a formal theoretical framework that maps any LR decay schedule to principled averaging weights, and we empirically demonstrate that these non-EMA weights significantly outperform the sub-optimal EMA baseline.
Our work builds upon this SWA paradigm. We establish a theoretical connection between this operation and LR decay and provide a principled approach to convert various LR decay strategies into a theoretically approximate model averaging implementation.
this section cite: ['b24', 'b68', 'b24', 'b48', 'b55', 'b37', 'b54', 'b16', 'b25', 'b14', 'b12']

Section: CONCLUSION
In this paper, we have presented WSM, a decay-free LR scheduling approach for LLM pre-training.
Our method bridges LR decay and checkpoint merging by establishing the theoretical connection. By eliminating the conventional decay phase, WSM simplifies LR scheduling while reformulating various decay strategies as principled model averaging schemes. Through systematic analysis, we identified the incorporation of high-quality annealing data and merge duration as the most critical factors influencing model performance-outweighing other implementation choices. Extensive experiments have demonstrated WSM's superiority over traditional WSD baselines in LLM pre-training, with consistent improvements across multiple benchmarks, robustness across different optimizers, and sustained benefits during fine-tuning. Consequently, WSM offers a highly versatile approach that can be seamlessly integrated into existing training pipelines without additional complexity.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: ()
Ref_id:b1 Title: GQA: training generalized multi-query transformer models from multi-head checkpoints Year: (2023)
Ref_id:b2 Title: Singular value decomposition for genome-wide expression data processing and modeling Year: (2000)
Ref_id:b3 Title: Efficient training of language models to fill in the middle Year: ()
Ref_id:b4 Title: Straight to zero: Why linearly decaying the learning rate to zero works best for llms Year: (2025)
Ref_id:b5 Title: Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge Year: (2021)
Ref_id:b6 Title: PIQA: reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b7 Title:  Year: ()
Ref_id:b8 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b9 Title:  Year: ()
Ref_id:b10 Title: Optimal linear decay learning rate schedules and further refinements Year: (2023)
Ref_id:b11 Title: The road less scheduled Year: (2024)
Ref_id:b12 Title:  Year: (2025)
Ref_id:b13 Title: A closer look at deep learning heuristics: Learning rate restarts, warmup and distillation Year: (2019-05-06)
Ref_id:b14 Title: The llama 3 herd of models Year: (2024)
Ref_id:b15 Title: Cruxeval: A benchmark for code reasoning, understanding and execution Year: (2024)
Ref_id:b16 Title: Scaling laws and compute-optimal training beyond fixed training durations Year: (2024)
Ref_id:b17 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b18 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021-12)
Ref_id:b19 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b20 Title: Worldsense: Evaluating real-world omnimodal understanding for multimodal llms Year: ()
Ref_id:b21 Title: Unveiling the potential of small language models with scalable training strategies Year: ()
Ref_id:b22 Title: Maosong Sun, and Junxian He. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models Year: (2023)
Ref_id:b23 Title: Simple and scalable strategies to continually pre-train large language models Year: (2024)
Ref_id:b24 Title: Averaging weights leads to wider optima and better generalization Year: (2018)
Ref_id:b25 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2025)
Ref_id:b26 Title: Rethinking learning rate tuning in the era of large language models Year: (2023)
Ref_id:b27 Title:  Year: (2023)
Ref_id:b28 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b29 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017-07-30)
Ref_id:b30 Title: Stop wasting my time! saving days of imagenet and BERT training with latest weight averaging Year: ()
Ref_id:b31 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b32 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b33 Title: Natural questions: a benchmark for question answering research Year: (2019)
Ref_id:b34 Title: RACE: large-scale reading comprehension dataset from examinations Year: (2017-09-09)
Ref_id:b35 Title: CMMLU: measuring massive multitask language understanding in chinese Year: (2024)
Ref_id:b36 Title: Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers Year: (2024)
Ref_id:b37 Title: Trainable weight averaging: Efficient training by optimizing historical solutions Year: (2023)
Ref_id:b38 Title: Model merging in pre-training of large language models Year: ()
Ref_id:b39 Title: Checkpoint merging via bayesian optimization in LLM pretraining Year: (2024)
Ref_id:b40 Title: Mathbench: Evaluating the theory and application proficiency of llms with a hierarchical mathematics benchmark Year: (2024)
Ref_id:b41 Title: Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation Year: (2023)
Ref_id:b42 Title: Muon is scalable for llm training Year: (2025)
Ref_id:b43 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b44 Title: Decoupled weight decay regularization Year: (2019-05-06)
Ref_id:b45 Title: Kor-bench: Benchmarking language models on knowledge-orthogonal reasoning tasks Year: (2025)
Ref_id:b46 Title: Can a suit of armor conduct electricity? A new dataset for open book question answering Year: (2018-11-04)
Ref_id:b47 Title: Humaneval-xl: A multilingual code generation benchmark for cross-lingual natural language generalization Year: (2024-05-25)
Ref_id:b48 Title: Acceleration of stochastic approximation by averaging Year: (1992)
Ref_id:b49 Title: Know what you don't know: Unanswerable questions for squad Year: (2018)
Ref_id:b50 Title: WARP: on the benefits of weight averaged rewarded policies Year: ()
Ref_id:b51 Title: GPQA: A graduate-level google-proof q&a benchmark Year: ()
Ref_id:b52 Title: The effective rank: A measure of effective dimensionality Year: (2007)
Ref_id:b53 Title: Winogrande: an adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b54 Title: Training trajectories, minibatch losses and the curious role of the learning rate Year: (2023)
Ref_id:b55 Title: Early weight averaging meets high learning rates for llm pre-training Year: (2023)
Ref_id:b56 Title: Language models are multilingual chain-of-thought reasoners Year: (2023)
Ref_id:b57 Title: Through the river: Understanding the benefit of schedule-free methods for language model training Year: (2025)
Ref_id:b58 Title: Roformer: Enhanced transformer with rotary position embedding Year: ()
Ref_id:b59 Title: Challenging bigbench tasks and whether chain-of-thought can solve them Year: (2023)
Ref_id:b60 Title: Mathscale: Scaling instruction tuning for mathematical reasoning Year: (2024)
Ref_id:b61 Title: Enhancing program synthesis with large language models using many-objective grammar-guided genetic programming Year: ()
Ref_id:b62 Title: Supergpqa: Scaling LLM evaluation across 285 graduate disciplines Year: ()
Ref_id:b63 Title: Visualizing data using t-SNE Year: (2008)
Ref_id:b64 Title: How to set AdamW's weight decay as you scale model and dataset size Year: (2024)
Ref_id:b65 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b66 Title: CMATH: can your language model pass chinese elementary school math test? Year: ()
Ref_id:b67 Title: Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective Year: (2024)
Ref_id:b68 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022-07-23)
Ref_id:b69 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019-08-02)
Ref_id:b70 Title: How does critical batch size scale in pre-training? Year: (2025)
Ref_id:b71 Title: Why gradient clipping accelerates training: A theoretical justification for adaptivity Year: (2020)
Ref_id:b72 Title: Evaluating the performance of large language models on GAOKAO benchmark Year: ()
Ref_id:b73 Title: Agieval: A human-centric benchmark for evaluating foundation models Year: (2024)
