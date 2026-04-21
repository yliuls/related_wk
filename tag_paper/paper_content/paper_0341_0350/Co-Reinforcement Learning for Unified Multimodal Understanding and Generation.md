Title: Co-Reinforcement Learning for Unified Multimodal Understanding and Generation
Abstract: This paper presents a pioneering exploration of reinforcement learning (RL) via group relative policy optimization for unified multimodal large language models (ULMs), aimed at simultaneously reinforcing generation and understanding capabilities. Through systematic pilot studies, we uncover the significant potential of ULMs to enable the synergistic co-evolution of dual capabilities within a shared policy optimization framework. Building on this insight, we introduce CoRL, a Co-Reinforcement Learning framework comprising a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement. With the proposed CoRL, our resulting model, ULM-R1, achieves average improvements of 7% on three text-to-image generation datasets and 23% on nine multimodal understanding benchmarks. These results demonstrate the effectiveness of CoRL and highlight the substantial benefits of reinforcement learning in facilitating cross-task synergy and optimization for ULMs. Code is available at https://github.com/mm-vl/ULM-R1.

Section: Introduction
As large foundation models (LFMs) continue to advance in their general capabilities and breadth of knowledge, post-training [30,46,70,73,108] has emerged as a critical paradigm for further refining pretrained LFMs toward specialized applications, thereby facilitating task adaptation and human-aligned behaviors. Recently, reinforcement learning (RL)-based approaches [51,52,59,64,69,72,95] have exhibited considerable promise due to their data efficiency and strong alignment abilities. A notable exemplar is DeepSeek-R1 [19], which demonstrates that RL with verifiable rewards and the group relative policy optimization (GRPO) algorithm constitutes a practical and stable strategy that sidesteps explicit preference modeling [77] and reward model learning [83]. This promising paradigm indicates the significant potential of LFMs to acquire advanced capabilities and generalize effectively without dependence on large-scale, high-quality supervised data.
In the multimodal AI research community, the prevailing implementation [9,24,40,41,53,69,100,101] of the GRPO algorithm centers on crafting diverse rule-based reward mechanisms to incentivize long-chain reasoning capabilities of multimodal large language models (MLLMs). These initiatives primarily target multimodal understanding, with a particular focus on visual and mathematical reasoning tasks. Conversely, its application to visual generation remains surprisingly limited, with only pioneering explorations [25,79] suggesting its feasibility. More importantly, extending GRPO to unified MLLMs (ULMs) [8,39,88,90] capable of concurrently performing visual understanding and generation tasks remains considerably under-explored. Intuitively, ULMs could significantly RL-based Post-Training for MLLMs. Post-training [108] aims to further enhance the performance of pretrained models for customized applications and user needs. Recently, RL [78,84] has emerged as a powerful post-training technique, enabling models to learn from feedback and align with human values. RL in MLLMs can be broadly categorized into two paradigms: (1) RL from human/AI feedback (RLHF) [34, 54, 61, 68, 77, 81-83, 92, 95, 96, 99, 106, 109] and (2) RL with verifiable reward mechanisms [35,40,41,69,79,100]. RLHF involves learning reward models from preference data before RL optimization, whereas the latter directly optimizes models using task-specific reward functions, bypassing explicit preference modeling. For example, DPO [59] is a notable implementation of RLHF and has been adopted by Emu3 [82] and HermesFlow [92] to narrow the performance gap between understanding and generation. In contrast, GRPO [64] exemplifies the second paradigm, simplifying reward formulation via group-wise relative advantage estimation. Our work also falls into this paradigm but diverges from prior work such as SimpleAR [79], which utilizes GRPO with external CLIP reward for autoregressive visual generation, and R1-like MLLMs [24,41,69,100] that focus on incentivizing reasoning capabilities. First, our work demonstrates the significant potential of RL in co-optimizing understanding and generation, thereby broadening its applicability beyond reasoning. Moreover, we identify semantic consistency rewards and a co-evolutionary reinforcement strategy as crucial components in enhancing ULMs.
this section cite: ['b29', 'b45', 'b69', 'b72', 'b107', 'b50', 'b51', 'b58', 'b63', 'b68', 'b71', 'b94', 'b18', 'b76', 'b82', 'b8', 'b23', 'b39', 'b40', 'b52', 'b68', 'b99', 'b100', 'b24', 'b78', 'b7', 'b38', 'b87', 'b89', 'b107', 'b77', 'b83', 'b34', 'b39', 'b40', 'b68', 'b78', 'b99', 'b58', 'b81', 'b91', 'b63', 'b78', 'b23', 'b40', 'b68', 'b99']

Section: Methodology

this section cite: []

Section: Preliminary
Group relative policy optimization (GRPO) [64] is a value-free policy optimization algorithm with improved training stability and sample efficiency. Building upon PPO [62], GRPO introduces a groupwise relative advantage approach to bound policy updates while maintaining optimization flexibility. Let π θ denote a policy parameterized by θ. Formally, given an input content c, the algorithm first samples a group of G outputs {o 1 , o 2 , . . . , o G } from the current policy π θ old . Each output is then evaluated using predefined, verifiable reward functions, yielding the reward set {r 1 , r 2 , . . . , r G }. These rewards are subsequently normalized to compute group-relative advantages as follows:
A i = r i -mean({r 1 , r 2 , . . . , r G }) std({r 1 , r 2 , . . . , r G }) . (1
)
After obtaining the advantage set {A 1 , A 2 , . . . , A G } via group relative advantage estimation, the policy π θ is optimized by maximizing the following objective:
L(θ) = E {oi} G i=1 ∼π θ old 1 G G i=1 π θ (o i ) π θold (o i ) A i -β D KL (π θ π ref ) ,(2)
where D KL denotes the KL-divergence used to constrain the deviation between π θ and its reference policy π ref , and β is a regularization coefficient.
this section cite: ['b63', 'b61']

Section: Pilot Exploration

this section cite: []

Section: Generation

this section cite: []

Section: Janus-Pro-1B [8] serves as the baseline.
Given the exceptional performance and data efficiency of DeepSeek-R1-Zero [19], we explore the potential of ULMs to enhance understanding and generation capabilities without dependence on task-specific supervised fine-tuning.
To accomplish this, we curate a datasetfoot_0 comprising 16K samples sourced from the COCO 2017 training split [38]. Each sample includes a real image, an associated caption as a textual prompt for visual generation, and a corresponding QA pair for the multimodal understanding task. We adopt CLIP Score [58] as the verifiable reward for image generation, along with a combination of formatting correctness and answer
T G T 1 T 2 Prompt + Image + Question
this section cite: ['b18', 'b37', 'b57']

Section: Group Computation
Policy ULM (𝝅 𝜽 )
🔥 accuracy as the reward for text generation. We investigate four distinct RL paradigms: (i) separate RL, where understanding and generation tasks are independently optimized with their respective reward mechanisms; (ii) separate RL followed by weight merging, where each task is separately optimized, and the resulting weights are subsequently merged using a Gaussian distribution-based merging strategy [66] to incorporate both abilities; (iii) cycle RL, which employs a scheduled alternation between the two tasks throughout the training process; and (iv) unified RL, in which both tasks are jointly optimized within a unified paradigm to promote the co-evolution of dual capabilities.
I 1 I 2 I G ℛ "#"$% ℛ &'( ℛ )"" ℛ *+,-./ A 1 A 2 A G S1: Unified RL S2: Refined RL ⋯ ⋯ Reward Evaluation r 1 r 1 r 1 r 1 r 2 r 2 r 2 r 2 r G r G r G r G ⋯ Policy Gradient Update
As presented in Figure 1, we observe that (1) direct task-specific RL fails to achieve the expected improvements for ULMs, particularly in the visual generation task, and may even impair performance on the other task; and (2) unified RL demonstrates substantial advantages over alternative paradigms. These findings indicate that the dual capabilities that co-evolve within a shared training framework contribute to enhanced cross-task synergy and knowledge transfer.
this section cite: ['b65']

Section: Co-Reinforcement Learning

this section cite: []

Section: Verifiable Reward for Multimodal modeling
In this section, we develop a suite of verifiable rewards for multimodal modeling, which provide clear and objective feedback to steer ULMs toward generating high-quality image and text outputs.
this section cite: []

Section: Bidirectional Cycle Consistency Reward in Text-to-Image Generation.
To encourage ULMs to generate images that faithfully depict the concepts and entities described in the input prompt, we introduce a bidirectional cycle consistency reward R cycle , which measures the consistency between predictions and ground truth in both visual and textual spaces. For visual consistency, we adopt LPIPS [104] to assess the patch-level perceptual similarity between the real image I real and the synthesized image I gen . Textual consistency is implemented in a re-captioning manner. Specifically, we first employ BLIP [32] to generate a caption C re-cap for each synthesized image, and then compute the SPICE [1] score between C re-cap and its original prompt P org to measure semantic fidelity. The combined bidirectional cycle reward is defined as: R cycle = 1 -LPIPS(I real , I gen ) + SPICE(P org , C re-cap ).
(
This bidirectional reward forms a closed feedback loop that promotes mutual consistency between texts and images, effectively penalizing hallucinated content and reinforcing prompt-aligned visual generation by simultaneously optimizing for both visual and textual consistency. Furthermore, R cycle is normalized to the range [0, 1] before being combined to ensure that all rewards operate on comparable scales and to prevent any single component from dominating due to scale differences.
this section cite: ['b103', 'b31', 'b0']

Section: Text-Image Matching Reward.
While CLIP Score [58] provides a holistic measure of text-image alignment, as shown in Sec. 3.2, it underperforms due to its limited capacity for assessing finegrained semantics. To address this limitation, we instead propose a text-image matching reward R TIM , which leverages the ULM itself to evaluate cross-modal alignment at the token level. Given a textual representation T = {t 1 , t 2 , . . . , t Lt } ∈ R Lt×d of the prompt and the corresponding visual representation I = {i 1 , i 2 , . . . , i Li } ∈ R Li×d of a generated image, the reward is computed as:
R TIM = 1 2   1 L i Li j=1 max k∈[1,Lt] cos(i j , t k ) + 1 L t Lt k=1 max j∈[1,Li] cos(t k , i j )   ,(4)
where L t and L i are the sequence lengths of the textual and visual tokens, d is the embedding dimension, and R TIM is also be normalized to the range [0, 1]. This reward captures the fine-grained correspondence between textual concepts and visual elements through maximum cosine similarity, ensuring mutual alignment between visual tokens and their most relevant textual counterparts.
this section cite: ['b57']

Section: Accuracy Reward in Multimodal Question Answering.
Accuracy rewards leverage task-specific metrics to directly evaluate the correctness of ULM predictions. We consider two accuracy rewards tailored to different question types: R MCQ-Acc for multi-choice questions and R OE-Acc for open-ended questions. These rewards follow a binary evaluation mechanism, assigning a value of 1 when the predicted answer (i.e., the final answer parsed from within <answer> and </answer> tags) matches the ground truth and 0 otherwise.
this section cite: []

Section: Format Reward in Text Generation.
To encourage ULMs to generate structured and interpretable textual responses, we adopt the format reward [19], which requires the model to enclose its thinking process inside <think> • • • </think>, and provide its final answer within <answer> and </answer> tags. The format reward R Format returns 1 for strict compliance and 0 otherwise.
this section cite: ['b18']

Section: Unified Reinforcement Learning for Synergistic Multimodal Modeling
As illustrated in Figure 2, the policy ULM first undergoes unified reinforcement learning with diverse rewards across understanding and generation tasks. This unified process aims to jointly enhance its dual capabilities and establish a solid foundation for subsequent task-specific refinement.
this section cite: []

Section: Reward Function and Training Objective.
To ensure diversity and complementarity in reward signals for unified multimodal modeling, we formulate a joint reward function as
R Uni-S1 = R cycle + R TIM + λ • (R Acc + R Format ),(5)
where λ is a coefficient that balances the two types of rewards. During training, given an input prompt and an image-question pair, the policy model π θ old first generates G candidate responses, o = {(I 1 , T 1 ), (I 2 , T 2 ), . . . , (I G , T G )}, each comprising a synthesized image I and a CoT-format solution T . Concurrently, the joint reward function R Uni-S1 evaluates each candidate pair, yielding the reward set r = {r 1 , r 2 , . . . , r G }. These rewards are subsequently normalized according to Eq. ( 1) to compute the corresponding group-relative advantages A = {A 1 , A 2 , . . . , A G }. The new policy model π θ is then updated by maximizing the following GRPO-based objective:
L S1 = E {oi} G i=1 ∼π θ old 1 G G i=1 π θ (o i ) π θ old (o i ) A i , where o i = (I i , T i ).(6)
Notably, based on empirical findings from recent work [94], we omit the KL-divergence constraint during this stage to improve both optimization efficiency and generalization capability.
Training Data. To support unified RL for synergistic multimodal modeling, we curate a comprehensive dataset comprising 22K samplesfoot_1 , which follows the data structure established in Sec. 3.2. Each sample includes a real image, a prompt for visual generation, and a CoT-format QA pair for multimodal understanding. This balanced data composition facilitates joint optimization of dual capabilities within a unified framework, while preserving the granularity of task-specific supervision.
this section cite: ['b93']

Section: Refined Reinforcement Learning for Task-specific Enhancement
After completing unified RL, as shown in Figure 2, we apply a targeted learning strategy to further enhance the task-specific performance of the policy model. This second-stage optimization leverages task-specific rewards and tailored datasets for individual tasks.
this section cite: []

Section: Reward Function and Training Objective.
For text-to-image generation, the reward is defined as R T2I-S2 = R cycle + R TIM . For multimodal understanding, we define two distinct reward formulations:
(1) R MCQ-S2 = R MCQ-Acc +R Format for multiple-choice questions, and (2) R OE-S2 = R OE-Acc +R Format for open-ended questions. The training objective in this stage adheres to the standard GRPO formulation in Eq. ( 2), with the appropriate task-specific reward (R T2I-S2 , R MCQ-S2 , or R OE-S2 ) replacing A i depending on the task. To ensure stable optimization, we reintroduce the KL-divergence constraint at this stage to limit policy deviation from the reference distribution.
Training Data. For text-to-image generation, we continue training on the curated dataset introduced in Sec. 3.2. For multimodal understanding, we utilize two specialized datasets: mcot_r1_mcqfoot_2 for multiple-choice questions and mcot_r1_vqafoot_3 for open-ended questions. These task-specific datasets enable the model to develop more refined and robust capabilities within each task domain.
this section cite: []

Section: Experiment

this section cite: []

Section: Experimental Setups
Evaluation Benchmarks. We evaluate visual generation capabilities on the GenEval [18], WISE [50], and DPG-Bench [22] benchmarks. GenEval employs an object-centric evaluation protocol to assess compositional and attribute-level alignment, while DPG-Bench adopts a VQA-based setting to evaluate dense prompt-following and semantic fidelity. WISE provides a holistic evaluation of models' world knowledge, considering consistency, realism, and aesthetics. We also evaluate multimodal understanding capabilities across diverse benchmarks. Specifically, MMStar [6], MMMU [98], and WeMath (Math We ) [56] are used for multi-choice evaluation, while MMVet [97], POPE [36], and LogicVista (Logic VT ) [89] are used for open-ended evaluation. In addition, we employ MathVista (Math VT ) [43], MathVerse-Vision (Math VS ) [102], and MathVision (Math Vis ) [80] to assess complex mathematical reasoning capabilities, covering both multi-choice and open-ended QA formats. On these benchmarks, we compute accuracy using the toolkit VLMEvalKit [13].
Implementation Details. We develop ULM-R1 using Janus-Pro-1B [8] as the baseline ULM for unified multimodal understanding and generation. To ensure reproducibility and scalability, our RL training is built upon the trl [75] framework. In the unified RL stage, we employ the AdamW optimizer with an initial learning rate of 4e-6 and a batch size of 16. We sample 8 responses for both understanding and generation tasks, and set the reward balancing factor in Eq. ( 5) to 0.8. In the refined RL stage, we sample 16 responses for both multimodal understanding and text-to-image generation tasks. Additionally, we reduce the learning rate to 1e-6 to facilitate fine-grained optimization. All training is conducted on 8 NVIDIA H20 (96G) GPUs. During inference, greedy decoding is used for text generation in multimodal understanding tasks. For text-to-image generation, we employ classifier-free guidance (CFG) [20] with a guidance weight set to 5. More details on the training data and settings are provided in App. A.
this section cite: ['b17', 'b49', 'b21', 'b5', 'b97', 'b55', 'b96', 'b35', 'b88', 'b42', 'b101', 'b79', 'b12', 'b7', 'b74', 'b19']

Section: Quantitative Results
Text-to-Image Generation. Table 1 presents a comprehensive comparison between ULM-R1 and state-of-the-art models across three visual generation benchmarks. Among unified models, our model ranks second on both GenEval and WISE benchmarks. Notably, it achieves balanced performance across diverse task categories within GenEval, with the best score of 0.71 in object counting. When compared with specialized generation-only models, ULM-R1 surpasses the top performer SD3-Medium [15] by a slight margin (0.77 vs. 0.74 on GenEval). Moreover, ULM-R1 shows consistent improvements over its base model across all benchmarks. These results collectively demonstrate the effectiveness and advantage of our CoRL in enhancing visual generation quality.
this section cite: ['b14']

Section: Multimodal Understanding.
Results are shown in Table 2. For mixed QA format evaluation, we continue to apply the Gaussian-distribution-based merging strategy [66] to combine the two task-specific policy models and obtain a final model capable of following both types of instructions.
Overall, ULM-R1 markedly outperforms existing unified models across most benchmarks, and substantially narrows the performance gap with leading understanding-only MLLMs of comparable model scale. More specifically, our model achieves state-of-the-art performance among unified models a photo of a knife and a zebra a photo of four bowls a photo of four donuts a photo of a computer keyboard and a laptop a photo of a cell phone and a horse a photo of a green skis and a brown airplane A clear image of a blackboard with a clean, dark green surface and the word 'Hello' written precisely and legibly in the center with bold, white chalk letters.
An old-fashioned metal lantern with a weathered bronze patina. The glass panels are slightly foggy from years of use, ... casting warm light across a dark wooden surface.
A large brown bear is partially submerged in a clear, tranquil body of water. ... The water ripples gently around the bear as it sits calmly, enjoying the warmth of the sunlight. on MMStar (47.6), WeMath (21.1), LogicVista (34.5), and on several mixed-format math benchmarks, including MathVerse (25.4) and MathVision (22.0). Particularly, ULM-R1 demonstrates considerable improvements over its base model in mathematical and logical reasoning tasks, achieving gains of 15.2 on WeMath and 10.6 on LogicVista. These results not only demonstrate the effectiveness of CoRL in enhancing ULMs' understanding capabilities, but also establish that reinforcement learning provides a data-efficient pathway for achieving both robust generalization and sophisticated reasoning capabilities, without the need for large-scale supervised data.
this section cite: ['b65']

Section: Qualitative Results
In this section, we first present a qualitative comparison between ULM-R1 and Janus-Pro for visual generation, as illustrated in Figure 3. The results clearly show that ULM-R1 achieves superior text-toimage alignment and object grounding across diverse prompts, with especially notable improvements in spatial arrangement of objects and compositional consistency. Next, as shown in Figure 4, we visualize several representative examples of multimodal understanding. Compared to Janus-Pro, ULM-R1 exhibits significantly enhanced understanding capabilities, particularly in mathematical reasoning. These comprehensive qualitative results demonstrate the effectiveness of CoRL in simultaneously Question: What is the substance that is developing on these leaves? Options: A. Algae B. Downy mildew C. Powdery mildew D. Sooty mould E. Don't know and don't want to guess Janus-Pro: B ULM-R1: <think>The green leaves on the plant are covered in a white powdery substance. This substance is a type of mildew that grows on the leaves of plants. Powdery mildew is a common disease that affects many plants. Therefore, the answer is C Powdery mildew.<\think><answer>C<\answer>
Question: Is the person bigger than the car?
Janus-Pro: Yes, the person is bigger than the car. The person is standing on the side of the road, while the car is on the road ahead of him.
ULM-R1: <think>No, the person is not actually bigger than the car.<\think><answer>No<\answer> improving visual generation and multimodal understanding in ULMs. Furthermore, Figures 3 and 4 respectively showcase exemplary failure cases of ULM-R1 in visual generation and understanding tasks, providing an intuitive grasp of its limitations. For instance, in the understanding example, it misinterprets commonsense and professional knowledge, leading to an incorrect answer.
this section cite: []

Section: Ablation Study and Discussion
In this section, we primarily evaluate the effectiveness of our RL training strategy and the proposed reward functions for text-to-image generation. In addition, we discuss the impact of the hyperparameter λ and the scalability of CoRL.
Comparison Between Various RL Paradigms. As presented in Table 3, we conduct a comprehensive ablation study to evaluate the effects of different RL paradigms for ULMs. The results reveal two key findings: #2 vs. #1: Unified-RL effectively enhances both the generation and understanding capabilities of ULMs, whereas Cold-SFT has minimal impact on visual generation. #7 vs. #6: Compared to the de facto paradigm, our CoRL consistently outperforms it on visual generation benchmarks while achieving comparable results on multimodal understanding benchmarks. These findings indicate that unified RL provides a robust foundation for task-specific refinement, even without reliance on supervised data. Additionally, CoRL consistently outperforms both its baseline and task-specific RL variants (#3-#5), achieving improvements of 2.1 points on GenEval (vs. generation-only RL, #3) and 5.3 points on WeMath (vs. understanding-only RL, #4). These results demonstrate the efficacy of CoRL as our final RL paradigm.
this section cite: []

Section: Effect of Rewards in Text-to-Image Generation.
To evaluate the effectiveness of our proposed rewards for text-to-image generation, we conduct ablation experiments as detailed in Table 4. The results demonstrate that incorporating either reward individually improves performance over the baseline: R cycle yields an increase of 2.1 in average score, while R TIM results in an increase of 0.8. Notably, combining both rewards leads to the best overall performance, achieving an average score of 80.6. These findings suggest a modest but complementary effect between R cycle and R TIM , enhancing their joint benefit in enhancing visual generation quality. In addition, we further compare the CLIP score (R CLIP ) and R TIM under our final RL training paradigm. As shown in the table, R TIM achieves better overall performance, especially on the DPG benchmark with dense, long-horizon prompts for image generation, highlighting its superior ability to capture fine-grained semantic alignment compared to the CLIP score.
this section cite: []

Section: Impact of Visual Consistency Measures in R cycle .
Table 5 provides a more detailed analysis of how different visual consistency measures (PSNR, MSE, SSIM, and LPIPS) used in R cycle affect the quality of visual generation. PSNR and MSE are pixel-level metrics that quantify low-level differences between the generated images, while SSIM and LPIPS assess higher-level perceptual and structural similarities. As shown in the table, SSIM and LPIPS perform better than the other two metrics, with LPIPS achieving the best performance (83.9) on the DPG benchmark. This can be attributed to the fact that LPIPS measures image similarity in a feature space, making it more robust to minor, semantically irrelevant variations and thus better suited to reward high-level consistency.
Impact of hyperparameter λ. The factor λ in Eq. ( 5) balances the reward scales between the two tasks during unified RL. As shown in Table 6, we conduct experiments using different values of λ to assess its impact on both generation and understanding performance. The results show that moderate values of λ (∼ 0.8) achieve a balanced trade-off between generation and understanding. Larger values slightly degrade generation performance, indicating that overemphasizing understanding rewards may hinder cross-task optimization.
this section cite: []

Section: Scalability of CoRL.
To validate the effectiveness of CoRL on other ULMs, as illustrated in Table 7, we conduct additional experiments using Janus-1.3B [85] and Janus-Pro-7B [8] as the baseline.
The results show consistent improvements across both generation and understanding benchmarks, confirming the scalability of CoRL. Notably, Janus-Pro-7B with LoRA tuning achieves smaller gains on the mathematical reasoning benchmark (WeMath) than Janus-1.3B, suggesting that while CoRL scales well across model size, its enhancement of complex reasoning does not scale linearly.
Table 7: Effectiveness of CoRL on other ULMs. For Janus-Pro-7B, we adopt LoRA tuning [21] to enable efficient training and mitigate memory pressure during unified RL.
this section cite: ['b84', 'b7', 'b20']

Section: Methods

this section cite: []

Section: GenEval WISE DPG MMMU MMStar Math
We MMVet POPE Logic VT Janus-1.3B 0.61 0.23 79.68 30.5 37.6 3.4 † 34.3 87.0 23.9 + CoRL (Full Fine-Tuning) 0.64 0.26 80.92 34.6 41.9 16.4 36.9 88.1 27.0 Janus-Pro-7B 0.80 0.35 84.19 41.0 46.5 9.7 50.0 87.4 28.0 + CoRL (LoRA Tuning) 0.82 0.41 84.97 44.6 49.5 16.0 52.6 88.0 32.4
this section cite: []

Section: Limitation
Despite the substantial improvements achieved, several limitations remain that warrant further investigation. First, a notable performance gap still exists between generation and understanding tasks of ULMs. Second, our rewards for multimodal understanding are relatively simple and primary. These limitations highlight the need for more sophisticated RL designs that can further enhance understanding capabilities and narrow the performance gap. We hope our work provides valuable insights for future RL research in ULMs.
this section cite: []

Section: Conclusion
In this work, we investigate how to jointly enhance the understanding and generation capabilities of ULMs, and propose a co-reinforcement learning framework (CoRL). Within the proposed CoRL, the policy model follows a Foundation-then-Specialization paradigm that involves a two-stage RL procedure: a unified RL stage for joint optimization and a refined RL stage for task-specific enhancement, yielding ULM-R1. Extensive evaluations across diverse understanding and generation benchmarks demonstrate the effectiveness of CoRL and the advantage of ULM-R1.
this section cite: []

Section: References
Ref_id:b0 Title: Spice: Semantic propositional image caption evaluation Year: (2016)
Ref_id:b1 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b2 Title: Improving image generation with better captions Year: (2023)
Ref_id:b3 Title: Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset Year: (2025)
Ref_id:b4 Title: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b5 Title: Are we on the right way for evaluating large vision-language models? Year: (2024)
Ref_id:b6 Title: M 3 CoT: A novel benchmark for multi-domain multi-step multi-modal chain-of-thought Year: (2024)
Ref_id:b7 Title: Janus-pro: Unified multimodal understanding and generation with data and model scaling Year: (2006)
Ref_id:b8 Title: Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning Year: (2025)
Ref_id:b9 Title: R-cot: Reverse chain-of-thought problem generation for geometric reasoning in large multimodal models Year: (2024)
Ref_id:b10 Title: Scalable vision language model training via high quality data curation Year: (2025)
Ref_id:b11 Title: Dreamllm: Synergistic multimodal comprehension and creation Year: (2024)
Ref_id:b12 Title: An open-source toolkit for evaluating large multimodality models Year: (2024)
Ref_id:b13 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b14 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b15 Title: Unified autoregressive visual generation and understanding with continuous tokens Year: (2025)
Ref_id:b16 Title: Seed-x: Multimodal models with unified multi-granularity comprehension and generation Year: (2024)
Ref_id:b17 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b18 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b19 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b20 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b21 Title: Ella: Equip diffusion models with llm for enhanced semantic alignment Year: (2024)
Ref_id:b22 Title: Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement Year: (2025)
Ref_id:b23 Title: Vision-r1: Incentivizing reasoning capability in multimodal large language models Year: (2025)
Ref_id:b24 Title: T2i-r1: Reinforcing image generation with collaborative semantic-level and token-level cot Year: (2025)
Ref_id:b25 Title: Corvid: Improving multimodal large language models towards chain-of-thought reasoning Year: (2025)
Ref_id:b26 Title: Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding Year: (2025)
Ref_id:b27 Title: Geomverse: A systematic evaluation of large models for geometric reasoning Year: (2023)
Ref_id:b28 Title: Orthus: Autoregressive interleaved image-text generation with modality-specific heads Year: (2024)
Ref_id:b29 Title: Llm post-training: A deep dive into reasoning large language models Year: (2025)
Ref_id:b30 Title: Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding Year: (2024)
Ref_id:b31 Title: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models Year: (2023)
Ref_id:b32 Title: Multimodal arxiv: A dataset for improving scientific comprehension of large vision-language models Year: (2024)
Ref_id:b33 Title: Vlfeedback: A large-scale ai feedback dataset for large vision-language models alignment Year: (2024)
Ref_id:b34 Title: Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning Year: (2025)
Ref_id:b35 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b36 Title: Dual diffusion for unified image generation and understanding Year: (2024)
Ref_id:b37 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b38 Title: World model on million-length video and language with ringattention Year: (2006)
Ref_id:b39 Title: Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement Year: (2025)
Ref_id:b40 Title: Visual-rft: Visual reinforcement fine-tuning Year: (2025)
Ref_id:b41 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b42 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2023)
Ref_id:b43 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: (2023)
Ref_id:b44 Title: Ovis: Structural embedding alignment for multimodal large language model Year: (2024)
Ref_id:b45 Title: Towards a unified view of large language model post-training Year: (2025)
Ref_id:b46 Title: Unitok: A unified tokenizer for visual generation and understanding Year: (2025)
Ref_id:b47 Title: Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation Year: (2024)
Ref_id:b48 Title: Redefining small and efficient multimodal models Year: (2025)
Ref_id:b49 Title: Wise: A world knowledge-informed semantic evaluation for text-to-image generation Year: (2025)
Ref_id:b50 Title: Openai o1 system card Year: (2024)
Ref_id:b51 Title: Openai o3 and o4-mini system card Year: (2025)
Ref_id:b52 Title: Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl Year: (2025)
Ref_id:b53 Title: Strengthening multimodal large language model with bootstrapped preference optimization Year: (2024)
Ref_id:b54 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b55 Title: We-math: Does your large multimodal model achieve human-like mathematical reasoning? Year: (2024)
Ref_id:b56 Title: Tokenflow: Unified image tokenizer for multimodal understanding and generation Year: (2024)
Ref_id:b57 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b58 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b59 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b60 Title: Mitigating object hallucination via data augmented contrastive tuning Year: (2024)
Ref_id:b61 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b62 Title: A-OKVQA: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b63 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b64 Title: Lmfusion: Adapting pretrained language models for multimodal generation Year: (2024)
Ref_id:b65 Title: Unveiling the mystery of weight in large foundation models: Gaussian distribution never fades Year: (2025)
Ref_id:b66 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b67 Title: Aligning large multimodal models with factually augmented rlhf Year: (2023)
Ref_id:b68 Title: Reason-rft: Reinforcement fine-tuning for visual reasoning Year: (2025)
Ref_id:b69 Title: Video-lmm post-training: A deep dive into video reasoning with large multimodal models Year: (2025)
Ref_id:b70 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b71 Title: Kimi k1. 5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b72 Title: A survey on post-training of large language models Year: (2025)
Ref_id:b73 Title: Metamorph: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b74 Title:  Year: ()
Ref_id:b75 Title: Illume: Illuminating your llms to see, draw, and self-enhance Year: (2024)
Ref_id:b76 Title: Sheng Zhang, Hoifung Poon, and Muhao Chen. mdpo: Conditional preference optimization for multimodal large language models Year: (2024)
Ref_id:b77 Title: Learning to reinforcement learn Year: (2016)
Ref_id:b78 Title: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl Year: (2006)
Ref_id:b79 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b80 Title: Enhancing the reasoning ability of multimodal large language models via mixed preference optimization Year: (2024)
Ref_id:b81 Title: Emu3: Next-token prediction is all you need Year: (2024)
Ref_id:b82 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b83 Title: Reinforcement learning. Adaptation, learning, and optimization Year: (2012)
Ref_id:b84 Title: Janus: Decoupling visual encoding for unified multimodal understanding and generation Year: (2006)
Ref_id:b85 Title: Liquid: Language models are scalable multi-modal generators Year: (2024)
Ref_id:b86 Title: Harmonizing visual representations for unified multimodal understanding and generation Year: (2025)
Ref_id:b87 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: ()
Ref_id:b88 Title: Multimodal llm logical reasoning benchmark in visual contexts Year: (2024)
Ref_id:b89 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2007)
Ref_id:b90 Title: Muse-vl: Modeling unified vlm through semantic discrete encoding Year: (2024)
Ref_id:b91 Title: Hermesflow: Seamlessly closing the gap in multimodal understanding and generation Year: (2007)
Ref_id:b92 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b93 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b94 Title: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback Year: (2024)
Ref_id:b95 Title: Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness Year: (2024)
Ref_id:b96 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b97 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b98 Title: Fine-tuning large vision-language models as decision-making agents via reinforcement learning Year: (2024)
Ref_id:b99 Title: Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning Year: (2025)
Ref_id:b100 Title: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization Year: (2025)
Ref_id:b101 Title: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? Year: (2024)
Ref_id:b102 Title: Mathematical visual instruction tuning Year: (2024)
Ref_id:b103 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b104 Title: Mllm-dataengine: An iterative refinement approach for mllm Year: (2023)
Ref_id:b105 Title: Beyond hallucinations: Enhancing lvlms through hallucination-aware direct preference optimization Year: (2023)
Ref_id:b106 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
Ref_id:b107 Title: Reinforced mllm: A survey on rl-based reasoning in multimodal large language models Year: (2025)
Ref_id:b108 Title: Aligning modalities in vision large language models via preference fine-tuning Year: (2024)
Ref_id:b109 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
Ref_id:b110 Title: Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required Year: (2025)
Ref_id:b111 Title: ) for what should or should not be described Year: ()
