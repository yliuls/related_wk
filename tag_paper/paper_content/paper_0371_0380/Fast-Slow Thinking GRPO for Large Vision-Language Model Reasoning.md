Title: Fast-Slow Thinking GRPO for Large Vision-Language Model Reasoning
Abstract: When applying reinforcement learning-typically through GRPO-to large visionlanguage model reasoning struggles to effectively scale reasoning length or generates verbose outputs across all tasks with only marginal gains in accuracy. To address this issue, we present FAST-GRPO, a variant of GRPO that dynamically adapts reasoning depth based on question characteristics. Through empirical analysis, we establish the feasibility of fast-slow thinking in LVLMs by investigating how response length and data distribution affect performance. Inspired by these observations, we introduce two complementary metrics to estimate the difficulty of the questions, guiding the model to determine when fast or slow thinking is more appropriate. Next, we incorporate adaptive length-based rewards and difficulty-aware KL divergence into the GRPO algorithm. Experiments across seven reasoning benchmarks demonstrate that FAST achieves state-of-the-art accuracy with over 10% relative improvement compared to the base model, while reducing token usage by 32.7-67.3% compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy.

Section: Introduction
Slow-thinking reasoning has demonstrated remarkable capabilities in solving complex tasks in Large Language Models (LLMs) [1][2][3][4] by applying large-scale reinforcement learning (RL), exemplified by OpenAI's o1 [5], DeepSeek-R1 [6], and Qwen's QwQ [7]. Unlike fast-thinking models [8,9], slowthinking models undertake more deliberate and thorough reasoning before reaching an answer, which facilitates the exploration of diverse solution paths for a given problem.
Researchers [10][11][12][13][14] have begun exploring similar slow thinking approaches for large visionlanguage models (LVLMs) to enhance visual reasoning, which can be categorized into SFT-RL two-stage methods [10,[15][16][17][18] and RLonly methods [19][20][21][22]. SFT-RL methods collect large-scale distilled data from slow-thinking models before applying reinforcement learning, while RL-only methods directly employ reinforcement learning on curated high-quality data.
Despite these efforts, several challenges persist in slow-thinking for LVLM reasoning. First, https://github.com/Mr-Loevan/FAST † Correspondence to Leilei Gan.
while RL-only methods enable slow-thinking LVLMs to improve reasoning accuracy, they struggle to effectively scale reasoning length [19,21,17], with observed changes ranging only from -20% to +10% compared to base models. This limited adaptability in reasoning length may constrain their effectiveness on complex tasks. Second, in contrast, we observe that slowthinking LVLMs with SFT-RL methods [23,15,10,16] exhibit a pronounced overthinking phenomenon-producing overly verbose responses across tasks while yielding only marginal improvements in accuracy. This observation suggests that excessive verbosity may arise from the SFT stage, which performs behavior cloning from distilled data. As evidenced in Table 1, R1-OneVision (one slow thinking model with SFT-RL) produces reasoning chains approximately 2× longer than its base model across all difficulty levels on the Geometry [24] test set. Notably, this overthinking proves detrimental for simpler questions, where extended reasoning results in accuracy degradation (69.5% vs. 72.7%), highlighting the need for adaptive fast-slow thinking.
Table 1: Comparison of accuracy and response length on Geometry 3K [24] test set across difficulty levels for Qwen2.5-VL-7B, R1-OneVision, and FAST.
1 |oi| |o i | t=1 min
π θ (oi,t|q, oi,<t) π θ old (oi,t|q, oi,<t) Âi,t, clip π θ (oi,t|q, oi,<t) π θ old (oi,t|q, oi,<t) , 1 -ϵ, 1 + ϵ Âi,t -βDKL π θ ∥ πref
where ε and β are the clipping hyperparameter and the coefficient controlling the KL regularization [6]. Âi,t is the advantage, estimated through group relative rewards Âi,t = ri-mean({r1,r2,...,r G }) std({r1,r2,...,r G }) with two rule-based rewards: (1) accuracy reward (r a ) gives a reward when the response is equivalent to the answer, and (2) format reward (r f ) ensures responses adhere to the specified format.
this section cite: ['b0', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b10', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b19', 'b21', 'b17', 'b23', 'b15', 'b10', 'b16', 'b24', 'b24', 'b6']

Section: Pilot Experiments
As discussed in §1, when applying reinforcement learning-typically through GRPO-to LVLM reasoning struggles to effectively scale reasoning length (RL-only methods; [19,21,17]) or generates verbose outputs across all tasks with only marginal gains in accuracy (SFT-RL methods; [23,15,10,16]). To better understand the factors affecting response length and overall performance in GRPO [6] for LVLM reasoning, we conduct a series of experiments on the Geometry 3K dataset [24]. In particular, we analyze the impact of lengthbased reward strategies ( §3.1) and the influence of data distribution characteristics ( §3.2).
this section cite: ['b19', 'b21', 'b17', 'b23', 'b15', 'b10', 'b16', 'b6', 'b24']

Section: Length Rewards Analysis
Prior research has established that while GRPO effectively scales response length in text-only LLMs [6,20,32], this effect does not transfer to LVLMs [19,22]. To verify this phenomenon and explore potential solutions, we performed GRPO-Zero on Qwen2.5-VL [33] with rule-based accuracy reward, and tested explicit length rewards that either encourage longer correct responses (r lengthy reward = L correct /L max ) or shorter correct ones (r short reward = 1 -L correct /L max ) as extended rewards, where L max is the maximum token length, L correct denotes the length of the correct response. As shown in Figure 2, with increasing training steps, GRPO with lengthy reward steadily increases to 700 tokens, GRPO with short reward decreases to 180 tokens, while Naive GRPO remains stable around 330 tokens. These length rewards successfully manipulated response length, producing variations from 180 to 700 tokens, but with only modest changes in accuracy (±3%). This decoupling between length and accuracy suggests that LVLMs can maintain reasoning performance across different response lengths, challenging the assumption that longer reasoning is always better.
Based on the findings above, we can draw the following conclusions: Observation 1: LVLMs can produce significantly different reasoning lengths with modest changes in accuracy via length rewards, suggesting potential for balancing reasoning depth and performance.
this section cite: ['b6', 'b20', 'b32', 'b19', 'b22', 'b33']

Section: Data Distribution Analysis
Figure 3: Effect of data distribution, especially difficulty on reasoning length and accuracy.
Considering that overthinking models tend to generate verbose reasoning responses regardless of question difficulty, we next investigated how data distribution-particularly the presence of samples with varying difficulty levels-might naturally influence reasoning length and performance.
To this end, we stratified the Geometry3K training dataset into three difficulty tiers using the pass@8 metric (i.e., the probability of correctly solving a question within eight attempts): Easy (0.75 ≤ pass@8), Medium (0.25 < pass@8 < 0.75), and Hard (pass@8 ≤ 0.25). This categorization resulted in approximately 35% Easy, 25% Medium, and 40% Hard samples.
Figure 3 illustrates how training on these different difficulty distributions affected model behavior. Models trained exclusively on Hard samples generated significantly longer responses but showed only marginal accuracy improvements. In contrast, training on Easy samples produced shorter responses while improving accuracy. Models trained on Medium samples showed modest length increase and the highest accuracy. Based on the findings above, we can draw the following conclusions:
Observation 2: Question difficulty acts as an implicit regulator of reasoning length, suggesting that data distribution can be strategically leveraged to achieve adaptive fast-slow thinking.
this section cite: []

Section: Fast-Slow Thinking GRPO
Building upon the aforementioned observations, we begin by introducing two complementary metrics to quantify the difficulty of multimodal questions, which facilitates dynamic data selection during reinforcement learning( §4.1). Subsequently, we introduce FAST-GRPO, a variant of GRPO specifically designed to balance fast and slow reasoning by leveraging adaptive length-based rewards conditioned on question difficulty( §4.2).
this section cite: []

Section: Multimodal Question Difficulty Estimation
Given our findings that data distribution influences both reasoning length and performance, accurately gauging data difficulty becomes essential for making dynamic adjustments to data distribution. To address this need, we propose two complementary metrics to measure the difficulty of a given question for the policy model: one directly evaluates the intrinsic difficulty of the multimodal question itself, while the other measures its difficulty relative to the policy model's current capabilities.
this section cite: []

Section: Extrinsic Difficulty.
We first quantify question difficulty relative to the policy model through the empirical success rate S extrinsic = 1 -pass@k , where pass@k = c/k represents the proportion of correct solutions among k rollouts. This metric is computed online, reflecting the model's evolving capabilities.
this section cite: []

Section: Intrinsic Difficulty.
While extrinsic difficulty reflects a model's ability to solve problems, it may not fully capture the inherent visual complexity of the questions. We therefore introduce image complexity as an indicator that specifically evaluates the intrinsic difficulty within questions. Specifically, we use the Gray-Level Co-occurrence Matrix (GLCM) score, which analyzes how frequently pairs of pixels with specific intensity values occur at defined spatial relationships [34,35]. However, GLCM captures only low-level image complexity based on pixel-level interactions and fails to account for higher-level semantic information. We therefore additionally employ the ViT classification entropy based on the output of the feature layer [36,37] for image semantics complexity, providing a high-level representation of conceptual difficulty. The image complexity is computed as follows:
H image = - 1 P P p=1 H(g p ) -H(v)(2)
where g p represents the GLCM for image patch p, H(g p ) is the entropy of the co-occurrence probabilities across multiple radii and orientations; v denotes the feature output from the final layer of the ViT classifier, and H(v) = -N j p j log p j is the entropy of the predicted probability distribution over N classes, with p j being the probability for class j.
We integrate these metrics to get a comprehensive difficulty estimation (S difficulty ) of a multimodal question. Questions with higher difficulty scores typically correspond to greater visual complexity and greater extrinsic difficulty for the policy model.
S difficulty = S extrinsic • H image(3)
Slow-to-Fast Sampling. We adopt curriculum strategies for the fast-slow thinking paradigm, using online-computed difficulty metrics. Two variants are considered: (i) Binary: distinct training phases. In early epochs, exclude easy samples (S extrinsic ≤ 0.25) to strengthen reasoning on hard questions; in later epochs, exclude hard samples (S extrinsic ≥ 0.75) to practice concise reasoning.
(ii) Continuous: smoothly shift sampling probability from harder to easier questions over epochs, enabling a gradual transition from slow to fast thinking. Binary enforces a clear capability-efficiency separation, while Continuous offers a gentler progression.
this section cite: ['b34', 'b35', 'b36', 'b37']

Section: FAST-GRPO
With the carefully designed difficulty estimation methods for multimodal question, we introduce FAST-GRPO, a tailored variant of GRPO that balances fast and slow reasoning by incorporating adaptive length-based rewards conditioned on question difficulty, as illustrated in Algorithm 1.
this section cite: []

Section: Algorithm 1 FAST-GRPO Training
Require: Base model π θ , selected dataset D, image complexity
H img 1: Initialize π ref ← π θ 2: for epoch = 1 to N do 3: Sample batch {q i } ∼ D 4: Generate {o j i } G j=1 ∼ π θ (q i )5:
Compute:
• S ext = 1 -pass@k(q i ) • S d = S ext • H img • β i = β min + (β max -β min )(1 - S ext ) 6:
Filter samples via Slow-to-Fast Sampling 7:
Compute rewards:
r j i = r a + λ f r f + λ t r t r t =    1 -L Lavg if (S d < θ) ∧ (ra = 1) min( L Lavg -1, 1) if (S d ≥ θ) ∧ (ra = 0) 0 otherwise 8:
Update policy:
max θ E clip π θ π old Âj -β i D KL (π θ ∥π ref )
9: end for Difficulty-Aware Length Reward Shaping.
In addition to the accuracy reward r a and format reward r f , we propose a difficulty-aware length reward r t as follows, which guides the model to employ the appropriate reasoning approach based on question difficulty.
r t =      1 -L Lavg if (S d < θ) ∧ (r a = 1) min( L Lavg -1, 1) if (θ ≤ S d ) ∧ (r a = 0) 0 otherwise (4
) where S d is the difficulty score, θ is the 80th percentile difficulty threshold across the batch, and L avg is the average length computed in the batch of responses. For less complex questions (S d < θ), the reward encourages fast thinking for correct trajectories, specifically rewards trajectories towards shorter than average length. Conversely, for complex questions, the reward encourages thorough reasoning for incorrect trajectories. Importantly, this reward is capped at 1, preventing excessive verbosity even for complex problems. Difficulty threshold θ is a hyperparameter, for which we analyse sensitivity in §5.4.
this section cite: []

Section: Following
Deepseek-R1 setting [8,19], we define the final reward function as a linear combination of these components: r i = r a + λ f r f + λ t r t . This difficulty-aware length reward necessitates encouraging exploration for complex problems while maintaining efficient, accurate responses for simpler ones.
this section cite: ['b8', 'b19']

Section: Difficulty-Aware KL Regularization.
In addition to the aforementioned length reward that encourages adaptive response length for questions of varying difficulty, the KL divergence term constrains the policy model's deviation from the reference model to achieve an exploitation-exploration balance, which impacts learning effectiveness across questions of different difficulty levels [6,31]. Our KL coefficient sensitivity analysis in § 5.4 also reveals that no single static β value optimally serves questions across difficulty levels. Lower KL constraints benefit challenging questions by enabling broader exploration, while stronger regularization maintains performance on simpler tasks. To address this issue, we implement a dynamic coefficient β d for difficulty-aware regularization.
β d = β min + (β max -β min ) • (1 -S extrinsic )(5)
We give a simple theoretical analysis to demonstrate how difficulty-aware β d enhances learning on varying questions by decomposing the gradient coefficient from the gradient Equation 6:
∇ θ JGRP O (θ) = E [q∼P (Q),{o i } G i=1 ∼π θ old (O|q)] 1 G G i=1 1 |oi| |o i | t=1 [GCGRP O (q, o)] ∇ θ log π θ (oi,t|q, oi,<t) (6) GCGRP O (q, o) = Âi Advantage Signal + β d πref(oi|q) π θ (oi|q) -1
Adaptive KL Regularization (7) As shown in Equation 7, the gradient coefficient consists of the advantage signal driving policy improvement and the adaptive KL regularization term. For high-difficulty questions, β d approaches β min , weakening the KL regularization and allowing the policy update to be dominated by the advantage signal. For low-difficulty questions, β d approaches β max , restricting policy deviation to ensure stability. Besides, the length normalization term 1 |oi| explicitly affects gradient updates, providing theory insight into Observation 2: for incorrect responses, it encourages longer outputs by reducing per-token penalties, while for correct responses, it encourages brevity through stronger per-token updates. This creates an implicit bias toward increasing response length for difficult questions where models generate more incorrect rollouts, while naturally promoting shorter responses for simpler questions that yield more correct solutions.
this section cite: ['b6', 'b31', 'b7']

Section: Experiments
In this section, we evaluate the efficacy of our method for LVLM reasoning.
this section cite: []

Section: Experimental Setup
Table 2: Comparison of different training methods and training samples.
this section cite: []

Section: Method

this section cite: []

Section: Training Stage

this section cite: []

Section: SFT Sample RL Sample
Virgo [38] ✓ 5K ✗ - Mulberry [39] ✓ 260K ✗ - LMM-R1 [17] ✗ - ✓ 105K MM-R1 [21] ✗ - ✓ 6K MM-Eureka [22] ✗ - ✓ 56K Curr-ReFT [18] ✓ 1.5K ✓ 9K OpenVLThinker [10] ✓ 35K ✓ 15K Vision-R1 [15] ✓ 200K ✓ 10K R1-OneVision [16] ✓ 155K ✓ 10K FAST (Ours) ✗ - ✓ 18K
Training Dataset. Starting with 500K questions from LLaVA-CoT [23], Mulberry [39], and MathV-360K [40], we first apply filters for answer verifiability. We deduplicate questions, retain only rule-based verifiable answers [41], and standardize to closed-form questions (e.g., multiple-choice, numeric answers). Second, we apply Slow-to-Fast sampling to remove questions with extreme extrinsic difficulty scores (S extrinsic = 0 or 1), yielding 18K training questions. We display its distribution in Figure 6 and the specific source in the appendix.
Evaluation Benchmarks. we evaluate on 7 widely used multimodal benchmarks: (1) Math-Vision [42], (2) MathVerse [43], (3) Math-Vista [44], (4) MM-Math [45], (5) WeMath [46], (6) DynaMath [47], and (7) MM-Vet [48]. The first six cover various mathematical reasoning tasks, while MM-Vet examines general multimodal abilities. We report both accuracy and response length for all benchmarks. We also conduct additional cross-domain evaluations, including science reasoning (MM-K12 [22]), open-domain VQA (Bingo [49], MMHAL [50]), low-level visual perception (MMVP [51]), and comprehensive calibrated evaluation (MMEval-Pro [52]). Details are provided in Appendix K.
Baselines. For slow thinking reasoning, we compare with three categories of approaches: (1) SFT on distilled data (LLaVA-CoT, Mulberry, Virgo); (2) RL-only training (MM-Eureka, LMM-R1, MM-R1); and (3) Two-stage approaches combining SFT and RL (R1-OneVision, Curr-ReFT, OpenVLThinker, Vision-R1). A comparative analysis of training methodologies and samples across these baselines is presented in Table 2. For fast-slow thinking comparison, we evaluate against fast thinking methods using various reward shaping techniques: Kimi 1.5's length penalty [25], cosine function rewards [26], and DAST [27]. Table 10 details these different reward formulations.
this section cite: ['b23', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b22', 'b49', 'b50', 'b51', 'b52', 'b25', 'b26', 'b27']

Section: Main Results
We report the main results concerning reasoning accuracy and reasoning length.
Reasoning Accuracy. Table 4 reports the main results of reasoning performance. First, FAST achieves state-of-the-art results on MathVista with 73.8 and MathVerse with 50.6, outperforming leading-edge closed-source LVLMs like GPT-4o. Second, on more challenging benchmarks, MathVision and MM-Math, FAST achieves competitive results, validating FAST's ability to solve complex questions. Third, FAST improves Qwen2.5-VL-7B, our base model, with an average accuracy improvement of over 10%. Fourth, FAST improves Qwen2.5-VL-3B with an average accuracy Reasoning Length. Tables 3 and 4 report the main results of reasoning length. First, FAST achieves a significant reduction of average reasoning length compared to slow thinking methods, from 32.7% against MM-R1 to 67.3% versus R1-OneVision, while preserving comparable or better reasoning accuracy. Second, compared to other fast thinking methods in LLMs, FAST achieves a modest reasoning length reduction and better reasoning accuracy. Third, FAST achieves slower thinking on more challenging questions, producing 60% longer responses on Hard than Easy of Geometry 3K as shown in Table 1 and averaging 79% more tokens on MM-Math. Lastly, in cross-domain evaluations ( §K), FAST yields substantially shorter responses: on MM-K12, average length drops by ∼106 tokens (33.8%) vs. its base model and over 30% vs. strong slow-thinking baselines. In open-domain VQA (Bingo, MMHal), outputs are consistently 15-25% shorter, showing effective control of reasoning length beyond math tasks. We conduct ablation studies to validate the effectiveness of each design of our method: Data Sampling, thinking reward shaping, and difficulty-aware optimization. The results are represented in Table 5. We can draw the following conclusions. First, without Data Sampling, reasoning accuracy seriously degrades on all benchmarks, highlighting the critical role of proper data distribution. Second, our thinking reward significantly reduces relative 42% response length with minor reasoning accuracy degradation, from 31.5 to 30.6 on MathVision. Third, the difficulty-aware regularization demonstrates robust improvement across all benchmarks, with a 1.8-point absolute increase on MathVista.
this section cite: []

Section: Ablations

this section cite: []

Section: Analyses

this section cite: []

Section: Validation of Image Complexity.
In our image complexity design, we utilize the GLCM entropy score [34,35] to measure texture complexity and ViT classification entropy [36,37] for semantic complexity. Zhang et al. [34] demonstrated that GLCM entropy achieves strong human alignment with a Spearman Rank-order Correlation Coefficient (SRCC) of 0.75 and a Pearson Linear Correlation Coefficient (PLCC) of 0.77. To validate the effectiveness of our combined metric H img on our specific dataset, we followed the methodology in [34], having three participants rate 200 sampled training images on a 5-point scale based on visual detail complexity. As shown in Table 8, while our combined metric H img demonstrates moderate correlation with human judgments (SRCC=0.49, PLCC=0.54), it maintains well alignment with human perception.
this section cite: ['b34', 'b35', 'b36', 'b37', 'b34', 'b34']

Section: Difficulty Threshold Analysis.
We use the 80th percentile of batch difficulty as our threshold θ. Figure 4 shows our grid search results: the 100th percentile yields concise responses (140.8 tokens) but reduces accuracy, while a 0 threshold produces excessive verbosity (486.2 tokens) with performance degradation. The 50th percentile achieves comparable accuracy to our 80th percentile but with 37% longer responses, confirming our choice effectively balances accuracy and conciseness.
this section cite: []

Section: Extrinsic Difficulty Analysis.

this section cite: []

Section: KL Coefficient Analysis.
Recent works [53,54] suggest that removing KL constraints can enhance long-form reasoning in language models. We explored this effect in visual reasoning through a grid search on the KL coefficient β (Figure 4). Our analysis reveals that lower β values significantly improve performance on Hard questions (16.9% at β = 0 vs. base 5.5%) by enabling greater exploration, but risk catastrophic forgetting on previously mastered tasks. Conversely, higher β values maintain strong performance on Easy questions and improve out-of-distribution generalization (69.2% at β = 5e -2 on MM-Vet), but restrict exploration on complex reasoning tasks. These findings demonstrate no static β value optimally serves questions across all difficulty levels-Hard questions benefit from looser constraints while Easy ones and generalization require stronger regularization.
this section cite: ['b53', 'b54']

Section: In-depth Analysis of Multiplying Estimated Difficulties.
The multiplicative form S difficulty = S extrinsic • H image jointly captures empirical hardness and intrinsic visual complexity. One theoretical concern is that this product could give low scores for cases that are hard for the model but visually simple, potentially leading to fast-thinking behaviour when slow reasoning is needed. In practice, such mismatches are rare (less than 5% of our training set), and our reward design in §4.2 assigns zero length reward in these cases, avoiding contradictory signals. We also tested a weighted-sum alternative S (sum) difficulty = αS extrinsic + (1 -α)H image , where α = 0.5, and found almost identical performance to the multiplicative form across MathVista, MathVision, and MathVerse (differences within 0.5% accuracy). These results confirm that FAST's difficulty estimation is robust to this potential corner case and to the choice of combination strategy. Details are provided in §H.
this section cite: []

Section: Case Studies and Failure Mode Analysis
We complement our quantitative results with qualitative illustrations of FAST's behaviour. Appendix §L provides examples where FAST adapts its reasoning, from concise answers on simple problems to expanded chains on complex ones, as well as typical failure cases. Here, we focus on a systematic analysis of these failures.
To understand when and why FAST-GRPO succeeds or fails, we analyse all incorrect responses from FAST-7B, R1-OneVision-7B, and the base model (Qwen2.5-VL-7B) on MATHVISTA. We observe three recurring failure patterns (Figure 5  Key Insights. First, adaptive fast-slow thinking substantially reduces reasoning-related failures: FAST-7B cuts Reasoning Error Propagation and Knowledge Conflict cases by ∼27% and ∼19% relative to its base model. Shorter, targeted reasoning chains leave fewer opportunities for mid-proof errors and help suppress hallucinations caused by overextended thought.
Second, perception, not reasoning, is the dominant bottleneck: over half of FAST-7B's errors stem from visual misinterpretation. Once a spatial relation is mis-localised or a key numeric value misread, even perfectly structured reasoning will converge to an incorrect answer. Future gains will likely come from strengthening the input stage, e.g., fine-grained OCR, calibrated scale reading, robust chart and graph value extraction, and accurate spatial grounding, so adaptive reasoning can operate on correct evidence.
this section cite: []

Section: Related Work
We review approaches for LVLM reasoning and methods addressing overthinking in LLMs.
Slow-thinking methods for LVLMs. SFT-RL two-stage methods leverage high-quality reasoning trajectories while inadvertently behavior cloning overthinking. Examples include Mulberry [39,55] using MCTS from GPT-4o, LLaVA-CoT [23] with structured reasoning stages, and Virgo [38] finetuning on text-only reasoning chains. Vision-R1 [15], R1-OneVision [16], and OpenVLThinker [10] first collect distilled data from advanced models before applying SFT and RL. RL-only methods directly employ RL to improve reasoning accuracy but struggle with scaling response length [56,21,22]. Visual-RFT [56] uses GRPO for various vision tasks, while MM-R1 [21], LMM-R1 [17], and MM-Eureka [22] apply RL on base models with curated visual reasoning questions.
Fast-Slow thinking methods for LLMs. Methods addressing overthinking in LLMs include inference-stage approaches include TALE [30] enforcing token budgets in prompt and CCoT [29] providing concise examples in context. Training-stage approaches include O1-Pruner [28] using a tailored RL objective to reduce verbosity, CoT-Value [57] fine-tuning on varied-length reasoning chains to learn dynamic thinking, and Kimi [25] proposing length penalty rewards in RL and Long2Short DPO [58] to shorten length. DAST [27] and CosineReward [26] encourage shorter correct responses and longer in-correct responses via curated length rewards. While effective for text-only tasks, these approaches remain largely unexplored for LVLM reasoning.
this section cite: ['b39', 'b55', 'b23', 'b38', 'b15', 'b16', 'b10', 'b56', 'b21', 'b22', 'b56', 'b21', 'b17', 'b22', 'b30', 'b29', 'b28', 'b57', 'b25', 'b58', 'b27', 'b26']

Section: Conclusion
We presented FAST, a framework enabling LVLMs to dynamically adapt reasoning depth based on question characteristics, addressing the overthinking phenomenon. Through empirical analysis, we developed FAST-GRPO with three components: model-based metrics for question characterization, adaptive thinking rewards, and difficulty-aware KL regularization. Extensive experiments demonstrated that FAST achieves state-of-the-art accuracy with over 10% improvement compared to the base model while reducing token usage compared to previous slow-thinking approaches, effectively balancing reasoning length and accuracy. This Appendix for "Fast-Slow Thinking GRPO for Large Vision-Language Model Reasoning" is organized as follows:
• Experimental Setup and Reproducibility. In §B we detail the implementation settings; §C describes the training and evaluation datasets; §D compares different length reward shaping methods; §E provides the human evaluation prompt for image complexity.
• Additional Experimental Analyses. §F analyses naive GRPO behaviors; §G reports statistical significance of main results; §L presents case studies and failure mode examples; §K gives cross-domain evaluation results (science reasoning, open-domain VQA, lowlevel visual perception, calibrated evaluation); §J shows scalability experiments on a 32B-parameter LVLM.
• Discussion and Limitations. §A discusses potential limitations of FAST-GRPO and directions for future work.
this section cite: []

Section: A Limitations
While our FAST framework demonstrates significant improvements in balancing reasoning length and accuracy, we acknowledge several limitations in our current work. Due to computational resource constraints, we were only able to evaluate our approach on models up to 32B parameters (Qwen2.5-VL-32B). The effectiveness of fast-slow thinking mechanisms may scale differently with larger models (e.g., models with 70B+ parameters), which could potentially exhibit different reasoning patterns and overthinking behaviors. We implement FAST using Qwen2.5-VL-3B and 7B as our base models. Below we detail our training setup and hyperparameters.
this section cite: []

Section: B Implementation Details

this section cite: []

Section: General Training Hyperparameters.
For FAST training, we use our 18K dataset with a learning rate of 1e-6, a batch size of 512. We set the maximum sequence length to 4096 for both prompts and generation, and apply BF16 precision throughout training. The training process runs for 10 epochs, requiring approximately 600 H100 GPU hours. We use the prompt: You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within < think > < /think > tags. The final answer MUST BE put in < answer > < /answer > tags.
this section cite: []

Section: Method-specific Training Hyperparameters.
For our reinforcement learning approach, we employ a temperature of 1.0, 8 rollouts per question, and a KL coefficient ranging from 0.001 (min) to 0.03 (max). The reward weighting factors are set to 0.5. The difficulty threshold is set at the 80th percentile. For GLCM computation, following prior setting [34], g p is derived from local patch p in the original image with 64 gray levels, defined by radius δ = [1, 2, 3, 4] and orientation θ = [0 • , 45 • , 90 • , 135 • ].
In practice, we divide the gray image into local patches of size 64.
this section cite: ['b34']

Section: Computation Environment.
All training experiments were conducted using H20 GPUs. Model inference in evaluations is performed using the vLLM framework [59], and our training implementation extends the VeRL codebase [60].
The complete set of hyperparameters is provided in Table 9. We commit to releasing all the code, data, and model checkpoints for experimental results reproducibility. Our training dataset comprises samples from four main categories: (1) Mathematical problems, including data from MathV360K, Ge-ometry3K, and other mathematical reasoning datasets; (2) Visual QA tasks, sourced from ShareGPT4V, Vizwiz, and additional visual question answering benchmarks; (3) Science problems from AI2D, ScienceQA, and other scientific reasoning datasets; and (4) Figure Understanding tasks from DocVQA, ChartQA, and other document and chart comprehension datasets. The distribution is balanced across these categories, with Mathematical problems constituting the largest portion, followed by Figure Understanding, Science, and Visual QA tasks.
this section cite: ['b59', 'b60']

Section: C Datasets

this section cite: []

Section: D Length Rewards
We provide a comparison of different length rewards in Talbe 10. if correct min(0, 0.5 -len(i)-min_len max_len-min_len ) otherwise CosFn [26] η
min + 1 2 (η max -η min )(1 + cos( tπ T )) where t is generation length, T is maximum length η min /η max are min/max rewards For correct answers: η min = r c 0 , η max = r c L For wrong answers: η min = r w 0 , η max = r w L DAST [27] max(-0.5λ + 0.5, 0.1) if correct min(0.9λ -0.1, -0.1) if incorrect where λ = Li-L budget L budget and L budget = p • L r + (1 -p) • L max , p = c N FAST r t =      1 -L Lavg if S difficulty < θ and r a = 1 min( L Lavg -1, 1) if θ ≤ S difficulty and r a = 0 0 otherwise
this section cite: ['b26']

Section: E Human Evaluation Prompt
Image Complexity Rating Instructions for Visual Reasoning Tasks Please rate the complexity of the given image on a scale of 1-5, considering how challenging it would be for visual reasoning tasks. Focus on aspects that affect the difficulty of analyzing, interpreting, and reasoning about the image content.
this section cite: []

Section: Rating Scale: 1 -Very Simple
• Clear, uncluttered images with few objects
• Simple spatial relationships • High contrast and clear visibility • Minimal text or numbers if present • Straightforward visual patterns 2 -Somewhat Simple • Moderately clear images with a manageable number of objects • Basic spatial relationships requiring minimal analysis • Good visibility with minor distractions • Limited text or numerical information • Recognizable patterns with minimal complexity 3 -Moderate Complexity • Multiple objects with varied relationships • Moderate spatial reasoning required • Some visual clutter or distractions • Moderate amount of text, numbers, or symbols • Patterns requiring some analysis 4 -Complex • Numerous objects with intricate relationships • Challenging spatial reasoning required • Significant visual clutter • Substantial text, numbers, or symbols requiring careful reading • Complex patterns requiring detailed analysis 5 -Very Complex • Dense arrangement of many objects with intricate relationships • Advanced spatial reasoning required • Heavy visual clutter making object identification difficult • Extensive text, numbers, or symbols with complex relationships • Intricate patterns requiring sophisticated analysis When rating, consider: number of objects, visual clarity, amount of information, spatial relationships, and reasoning steps needed to understand the image content.
this section cite: []

Section: F Naive GRPO Results
As shown in Figure 7, the training accuracy for naive GRPO continues to increase throughout the training process, similar to other methods like Dynamic Sampling and FAST. However, when we  examine the validation accuracy in Figure 8, we observe a concerning trend. The validation accuracy of naive GRPO initially increases, reaching a peak of 0.40 around step 150, but then begins to decline despite continued improvements in training accuracy. This discrepancy between training and validation performance indicates a clear overfitting phenomenon in naive GRPO. Upon closer examination of the model outputs in later training stages (e.g., case in Figure 9 ), we observe an increasing inconsistency between the chain-of-thought reasoning and the final answers produced by the model. Specifically, the model develops a tendency to generate reasoning chains that do not logically support its final conclusions, suggesting that it is memorizing training examples rather than learning generalizable reasoning patterns. This overfitting behavior explains why the validation accuracy deteriorates even as training accuracy continues to improve. The model optimizes for the training distribution at the expense of generalization capability. Therefore, implementing early stopping based on validation performance is crucial when using naive GRPO to select the optimal checkpoint (around step 150 in our experiments) before overfitting severely impacts the model's reasoning abilities.
this section cite: []

Section: Example Case: Inconsistent Reasoning in Naive GRPO on Training Dataset

this section cite: []

Section: Geometry problem with angles
Question: What is the measure of ∠GHI? Chain of Thought: Since the lines AB and CD are parallel, and EF is a transversal, the angles created by the transversal with the parallel lines have specific relationships. Specifically, the angle at point E that is adjacent to 40°and the angle at point F that is adjacent to 70°are supplementary angles with the angle GHI. The angle at point E is 40°, and since EF is a straight line, the angle adjacent to it on the other side of the transversal is 180°-40°= 140°. Similarly, the angle at point F is 70°, and the adjacent angle on the other side of the transversal is 180°-70°= 110°. Since GHI is the angle formed by the intersection of the transversal with the parallel lines, it is the angle that is supplementary to the 110°angle. Therefore, the measure of angle GHI is 180°-110°= 70°. So, the measure of ∠GHI is 70°. Final Answer: 110   To rigorously evaluate the effectiveness of our approach, we conducted statistical significance analysis across all benchmarks. Table 11 presents comprehensive results comparing our FAST models with their respective Qwen2.5-VL baselines, including standard deviations from multiple runs. Our analysis reveals that FAST-7B achieves statistically significant accuracy improvements on 6 out of 7 benchmarks, with only DynaMath showing a non-significant improvement (0.3 percentage points). The most substantial accuracy gains are observed on mathematical reasoning tasks (MathVision: +5.0%, MathVerse: +3.7%, MM-Math: +10.2%), demonstrating our method's particular effectiveness on complex reasoning problems.
this section cite: []

Section: G Statistical Significance Analysis
Regarding response length, FAST-7B consistently produces significantly more concise responses across all benchmarks, with length reductions ranging from 13.9% to 53.8%. This confirms that our approach successfully achieves both improved accuracy and enhanced efficiency in generating responses. The statistical significance of these improvements provides strong evidence for the effectiveness of our FAST framework in enhancing both the reasoning capabilities and efficiency.
this section cite: []

Section: H Analysis of Multiplicative Difficulty Formulation and Weighted-Sum Alternative
The multiplicative combination S difficulty = S extrinsic • H image was designed to jointly capture a model's empirical success rate and the intrinsic visual complexity of a question. One concern raised in review is that when S extrinsic is high (model finds the problem hard) but H image is very low (visually simple), the product may be close to zero, potentially signalling "fast thinking" in a case that is actually challenging. • Bingo Score [49]: Open-domain VQA benchmark for hallucination analysis.
• MMHal [50]: Hallucination and informativeness evaluation in open-domain VQA.
• MMVP [51]: Low-level visual perception probing.
• MMEval-Pro [52]: Calibrated multimodal benchmark spanning math, science, and general VQA.
Findings. FAST matches or slightly outperforms strong slow-thinking baselines in open-domain VQA and hallucination-robustness benchmarks, while producing shorter outputs. This supports the observation that adaptive reasoning length mitigates hallucination risk in multimodal reasoning [61].
this section cite: ['b49', 'b50', 'b51', 'b52', 'b61']

Section: L Case Study
Figure 11 illustrates how FAST balances reasoning length and accuracy. For simple coordinate identification, R1-OneVision exhibits overthinking with 349 tokens output (highlighted in green), while FAST delivers a concise 59-token solution. For complex geometry, the base model makes a critical error in angle calculations, while R1-OneVision produces a correct but verbose 676-token solution. FAST demonstrates adaptive slow thinking with a more efficient and correct 375-token solution. validating our approach's ability to adjust reasoning depth based on question complexity.
In addition to these efficiency-focused examples, we present three representative error cases illustrating the main failure categories discussed in Section 5.5: one each for Visual Perception Failure, Reasoning Error Propagation, and Knowledge Conflict & Gap. These cases are shown in Figures 12-14 and provide visual, task-specific instances of how such errors manifest across different problem types.
this section cite: []

Section: Example Case: Visual Perception Failure

this section cite: []

Section: Beaker with scale markings
Question: What is the highest amount this class measures? Ground Truth: 400 (ml) Model Answer: 600 (ml)
this section cite: []

Section: Error Analysis:
The image shows a beaker with markings up to 400 ml and a label indicating a total capacity of 600 ml. The question asks for the highest amount measured, which refers to the highest marked graduation (400 ml), not the container's maximum capacity. The model misinterprets the labeling and outputs 600 ml.
Error Type: Misinterpretation of visual context, confusing scale markings with container capacity.
this section cite: []

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction accurately reflect the paper's contributions, including the FAST framework for balancing fast-slow thinking in LVLMs, the empirical analysis of reasoning length and accuracy, and the three key components of the approach. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The paper includes a dedicated Limitations section A that acknowledges computational resource constraints limited evaluation to models up to 7B parameters, and notes that effectiveness may scale differently with larger models (70B+) which could exhibit different reasoning patterns.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: The paper is primarily empirical and does not include formal theoretical results requiring proofs. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: All the the implementation details can be found in § 5.1 and B.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Reward design avoids conflict
As shown in Algorithm (1), the difficulty-aware length reward r t applies non-zero shaping only in two aligned cases:
• Correct and Not Complex (S difficulty < θ): encourage shorter responses.
• Incorrect and Complex (S difficulty ≥ θ): encourage longer responses.
The misaligned case in question (Incorrect but Not Complex, or Correct but Complex) yields r t = 0, so no penalty or incorrect encouragement is applied.
this section cite: []

Section: Empirical rarity of corner cases
We ranked 1,000 random training samples and computed correlations between S extrinsic , H image , and S difficulty . High S extrinsic combined with low H image was rare (< 5% of samples).
this section cite: []

Section: Weighted-sum alternative
We compared the multiplicative form with a weighted sum: S (sum) difficulty = αS extrinsic + (1 -α)H image , α = 0.5. Table 12 shows near-identical results. These results confirm (i) corner cases are rare in our actual training distribution, (ii) reward shaping avoids contradictory signals for such cases, and (iii) performance is robust to the choice of multiplicative vs sum combination.
this section cite: []

Section: I Continuous Slow-to-Fast Sampling
In the main text (Section 5.4), we compared our Slow-to-Fast sampling strategy against alternative approaches such as Fast-to-Slow and Dynamic Sampling [53]. Here, we further contrast a continuous variant of Slow-to-Fast scheduling: .
this section cite: ['b53']

Section: Binary Slow-to-Fast.
In this setting, the training curriculum makes a hard switch at the halfway point of total epochs: the first half samples only hard and medium questions, and the second half incorporates easy questions, following the procedure in Algorithm 1.
this section cite: []

Section: Continuous Slow-to-Fast.
Here, the probability of drawing an easy sample, p easy , increases linearly with the training epoch t from 0 at the start to a maximum P max at the final epoch:
p easy (t) = P max • t T ,
where T is the total number of epochs. We set P max = 0.4 following initial tuning, ensuring a gradual transition from hard/medium-focus to more balanced sampling.
Results. Table 13 compares the two schedules under identical training settings on MATHVISTA, MATHVISION, and MATHVERSE.
this section cite: []

Section: Findings.
Continuous scheduling provides accuracy gains (e.g., +0.6pp on MATHVISTA) and increases average output length by 26%, reducing efficiency. We hypothesize that the chosen P max was insufficient to sample a large enough proportion of easy questions in later epochs, limiting potential efficiency gains. Additional tuning or adaptive P max may yield more favourable trade-offs.
this section cite: []

Section: J Scalability to Larger Models
To evaluate the scalability of FAST-GRPO beyond mid-sized LVLMs, we train and test the framework on the 32B-parameter model Qwen-2.5-VL-32B using the same 18K-question training set as in our main experiments. Due to compute constraints, training was stopped after 3 epochs (∼1,200 GPU hours), which likely results in a sub-optimal checkpoint. We compare FAST-32B to strong slowthinking baselines including Vision-R1-32B and MM-Eureka-32B on six benchmarks, including MM-K12 [22], a 2,000-question scientific reasoning benchmark evenly covering math, physics, chemistry, and biology. Due to compute constraints, we stopped training after three epochs (1200 GPU hours), resulting in a likely sub-optimal checkpoint. Findings. Despite shorter training, FAST-32B matches or slightly exceeds the accuracy of stronger slow-thinking baselines while using notably fewer tokens:
• Versus Vision-R1-32B, average output length is reduced by ∼ 40% (422.1 vs. 710.6 tokens) with comparable accuracy (64.3% vs. 64.6%).
• Versus MM-Eureka-32B, length is reduced by ∼ 22% (422.1 vs. 546.0 tokens) while slightly improving average accuracy (64.3% vs. 64.1%).
These results indicate that FAST-GRPO scales effectively to larger LVLMs, maintaining its accuracyefficiency trade-off. We leave exploration on ultra-large (≥ 70B) LVLMs for future work.
this section cite: ['b22']

Section: K Cross-Domain Evaluation
To validate FAST-GRPO beyond math-intensive benchmarks, we conducted additional experiments on science reasoning, open-domain VQA, hallucination analysis, and low-level visual perception. These evaluations were added in response to reviewer requests for broader task coverage.
this section cite: []

Section: K.1 MM-K12 Scientific Reasoning
The MM-K12 benchmark [22] consists of 2,000 multimodal reasoning questions evenly covering four domains: mathematics, physics, chemistry, and biology.
Findings. Compared to its base model, FAST-7B improves accuracy by +8.4pp in physics, +7.0pp in chemistry, and +8.8pp in biology, while reducing output length by ∼ 33.8%. Even against strong slow-thinking models such as MM-Eureka-7B, accuracy remains comparable with ∼ 30% fewer tokens.
this section cite: ['b22']

Section: K.2 Additional General Benchmarks
We further evaluate on four benchmarks covering open-domain VQA and visual robustness:
this section cite: []

Section: Example Case: Reasoning Error Propagation

this section cite: []

Section: Geometry diagram
Question:   5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code and model checkpoint are stated to be open source. Data generation for the training dataset is described, and evaluation benchmarks are public.
Guidelines:
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
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We list experimental setting in § B and § 5.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: See Figure 10 and Table 11.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: See §B. We specify that training requires approximately 600 H100 GPU hours, with a global batch size of 512 and 8 rollouts per question over 10 epochs.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We promise that this paper conforms, in every respect, with the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [No] Justification: The paper does not explicitly discuss potential positive and negative societal impacts of the work. The research focuses on improving reasoning efficiency in visionlanguage models, which has general benefits for AI applications but could benefit from a discussion of broader implications.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The paper focuses on improving reasoning capabilities rather than releasing models with high risk for misuse. The proposed method enhances existing models rather than creating new potentially harmful capabilities.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: The paper properly cites and credits the original sources of datasets (LLaVA-CoT, Mulberry, MathV-360K) and models (Qwen2.5-VL) used in the experiments, as well as the evaluation benchmarks.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: The paper describes the training data generation process and mentions releasing model checkpoints as well as code, with sufficient documentation of the methodology to understand the assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [Yes] Justification: Human evaluation instruction is provided in §E. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The human evaluation conducted appears to be a minimal risk assessment of image complexity rather than research requiring IRB approval, as it doesn't involve personal data or potential harm to participants. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: ()
Ref_id:b1 Title: From system 1 to system 2: A survey of reasoning large language models Year: (2025)
Ref_id:b2 Title: Tree-of-mixed-thought: Combining fast and slow thinking for multi-hop visual reasoning Year: (2023)
Ref_id:b3 Title: A comprehensive survey of datasets, theories, variants, and applications in direct preference optimization Year: (2024)
Ref_id:b4 Title: Revealer: Reinforcement-guided visual reasoning for element-level text-image alignment evaluation Year: (2025)
Ref_id:b5 Title: Learning to reason with large language models Year: (2024-09)
Ref_id:b6 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b7 Title: Qwq: Reflect deeply on the boundaries of the unknown Year: (2024-11)
Ref_id:b8 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b9 Title: Qwen2 technical report Year: (2024)
Ref_id:b10 Title: Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement Year: (2025)
Ref_id:b11 Title: Visual agents as fast and slow thinkers Year: (2025)
Ref_id:b12 Title: Exploring the reasoning abilities of multimodal large language models (mllms): A comprehensive survey on emerging trends in multimodal reasoning Year: (2024)
Ref_id:b13 Title: Detecting and mitigating hallucination in large vision language models via fine-grained AI feedback Year: (2025-03-04)
Ref_id:b14 Title: Adaptive fast-and-slow visual program reasoning for long-form videoqa Year: (2025)
Ref_id:b15 Title: Vision-r1: Incentivizing reasoning capability in multimodal large language models Year: (2025)
Ref_id:b16 Title: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization Year: (2025)
Ref_id:b17 Title: Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl Year: (2025)
Ref_id:b18 Title: Boosting the generalization and reasoning of vision language models with curriculum reinforcement learning Year: (2025)
Ref_id:b19 Title: R1-v: Reinforcing super generalization ability in vision-language models with less than $3 Year: ()
Ref_id:b20 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025-01)
Ref_id:b21 Title: Mmr1: Advancing the frontiers of multimodal reasoning Year: (2025)
Ref_id:b22 Title: Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning Year: (2025)
Ref_id:b23 Title: Llava-cot: Let vision language models reason step-by-step Year: (2024)
Ref_id:b24 Title: Inter-GPS: Interpretable geometry problem solving with formal language and symbolic reasoning Year: (2021-08)
Ref_id:b25 Title: Kimi k1.5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b26 Title: Demystifying long chain-of-thought reasoning in llms Year: (2025)
Ref_id:b27 Title: Dast: Difficulty-adaptive slow-thinking for large reasoning models Year: (2025)
Ref_id:b28 Title: O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning Year: (2025)
Ref_id:b29 Title: The benefits of a concise chain of thought on problem-solving in large language models Year: (2024-11)
Ref_id:b30 Title: Token-budget-aware llm reasoning Year: (2025)
Ref_id:b31 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b32 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (2025)
Ref_id:b33 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b34 Title: Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models Year: (2025)
Ref_id:b35 Title: Textural features for image classification Year: (1973)
Ref_id:b36 Title: Visual transformers: Token-based image representation and processing for computer vision Year: (2020)
Ref_id:b37 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b38 Title: Virgo: A preliminary exploration on reproducing o1-like mllm Year: (2025)
Ref_id:b39 Title: Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search Year: (2024)
Ref_id:b40 Title: Math-LLaVA: Bootstrapping mathematical reasoning for multimodal large language models Year: (2024-11)
Ref_id:b41 Title:  Year: (2025)
Ref_id:b42 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b43 Title: Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? Year: (2025)
Ref_id:b44 Title: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts Year: ()
Ref_id:b45 Title: MM-MATH: Advancing multimodal math evaluation with process evaluation and fine-grained classification Year: (2024-11)
Ref_id:b46 Title: We-math: Does your large multimodal model achieve human-like mathematical reasoning? Year: (2024)
Ref_id:b47 Title: Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models Year: (2024)
Ref_id:b48 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2024)
Ref_id:b49 Title: Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges Year: (2023)
Ref_id:b50 Title: Aligning large multimodal models with factually augmented rlhf Year: (2024)
Ref_id:b51 Title: Eyes wide shut? exploring the visual shortcomings of multimodal llms Year: (2024)
Ref_id:b52 Title: Mmevalpro: Calibrating multimodal benchmarks towards trustworthy and efficient evaluation Year: (2024)
Ref_id:b53 Title: Process reinforcement through implicit rewards Year: (2025)
Ref_id:b54 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b55 Title: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization Year: (2025)
Ref_id:b56 Title: Visual-rft: Visual reinforcement fine-tuning Year: (2025)
Ref_id:b57 Title: Cot-valve: Length-compressible chain-ofthought tuning Year: (2025)
Ref_id:b58 Title: A comprehensive survey of direct preference optimization: Datasets, theories, variants, and applications Year: (2024)
Ref_id:b59 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b60 Title: Hybridflow: A flexible and efficient rlhf framework Year: (2024)
Ref_id:b61 Title: More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models Year: (2025)
