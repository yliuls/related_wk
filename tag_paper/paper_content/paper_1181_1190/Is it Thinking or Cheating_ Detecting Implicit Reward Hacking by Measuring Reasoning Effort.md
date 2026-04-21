Title: IS IT THINKING OR CHEATING? DETECTING IMPLICIT REWARD HACKING BY MEASURING REASONING EFFORT
Abstract: Reward hacking, where a reasoning model exploits loopholes in a reward function to achieve high rewards without solving the intended task, poses a significant threat. This behavior may be explicit, i.e. verbalized in the model's chain-ofthought (CoT), or implicit, where the CoT appears benign thus bypasses CoT monitors. To detect implicit reward hacking, we propose TRACE (Truncated Reasoning AUC Evaluation). Our key observation is that hacking occurs when exploiting the loophole is easier than solving the actual task. This means that the model is using less "effort" than required to achieve high reward. TRACE quantifies effort by measuring how early a model's reasoning becomes sufficient to obtain the reward. We progressively truncate a model's CoT at various lengths, force the model to answer, and estimate the expected reward at each cutoff. A hacking model, which takes a shortcut, will achieve a high expected reward with only a small fraction of its CoT, yielding a large area under the reward-vs-length curve. TRACE achieves over 65% gains over our strongest 72B CoT monitor in math reasoning, and over 30% gains over a 32B monitor in coding. We further show that TRACE can discover unknown loopholes during training. Overall, TRACE offers a scalable unsupervised approach for oversight where current monitoring methods prove ineffective. 48. A student has 7 reference books, including 2 Chinese books, […]. Calculate the total number of different ways the books can be arranged.

Section: 
<think> First, we need to treat the 2 Chinese book
[…]
this section cite: []

Section: INTRODUCTION
Reward hacking occurs when a model exploits loopholes in a reward function. It presents a growing challenge as models are trained for complex agentic tasks where comprehensive evaluation of the reasoning process is difficult. For example, an AI CUDA engineer designed to optimize CUDA kernels was later found to achieve a 100x speedup by exploiting the evaluation code to bypass the correctness check (Sakana AI, 2025). Similarly, coding agents like Claude Sonnet 4 are found to exploit a data leak in SWE-Bench by looking ahead at future commits that contain the solution (Kahn, 2025). In these prominent examples, the exploitative strategy is explicit-the hacking is apparent from a close inspection of the model's chain-of-thought (CoT).
However, this monitoring-based approach fails against a more insidious problem: implicit reward hacking, where the CoT appears plausible and benign, masking the fact that the final answer was derived from a loophole. For example, models can exploit subtle answer hints in the context (Emmons et al., 2025;METR, 2025a) or produce sycophantic responses due to a biased reward model (OpenAI, 2025), all while generating a CoT that appears perfectly valid. Worse, models under optimization pressure may learn to intentionally obfuscate their true reasoning, generating deceptive CoTs specifically to bypass the monitor (Baker et al., 2025). Recent models that are designed to think in the latent continuous space also makes CoT monitoring fundamentally inapplicable (Hao et al., 2024).
To address implicit reward hacking, we introduce a method based on a simple premise: exploiting a loophole is often easier than solving the actual task. However, detecting this "laziness" is non-trivial.
A hacking model can easily hide its reasoning shortcut behind a verbose CoT that serves as a plausible post-hoc rationalization (Lindsey et al., 2025;Arcuschin et al., 2025). Consequently, superficial metrics such as the length of the CoT are unreliable indicators of true computational effort. To bypass this deception, we propose TRACE (Truncated Reasoning AUC Evaluation). Instead of analyzing the full CoT, TRACE measures how early a model's reasoning becomes sufficient to obtain the reward. Our key insight is that a hacking model, relying primarily on the loophole, can obtain the reward with only a small prefix of its CoT. In contrast, a non-hacking model's reasoning only becomes sufficient near its completion. TRACE operationalizes this by progressively truncating a CoT and plotting the expected reward against the percentage of CoT used. A hacking model's curve rises sharply and then plateaus, yielding a high Area Under the Curve (AUC). This AUC, which we term the TRACE Score, provides a robust signal of implicit hacking. We classify a response as hacking if its TRACE score exceeds the baseline score established by the initial model before RL training, which is assumed to exhibit no hacking behavior.
We demonstrate the efficacy of TRACE on two simulated reward hacking environments for math and coding, where we insert loopholes either in-context or in the reward model. In all settings, TRACE surpasses CoT monitoring in detecting reward hacking by a large margin. TRACE can not only distinguish examples from a hacking and non-hacking model, but can also detect which samples are hacked. Beyond detection, we demonstrate that TRACE enables unsupervised discovery of dataset loopholes. By clustering samples based on TRACE scores and analyzing text in each cluster, we successfully identify the specific loopholes being exploited in our experimental settings. This discovery application demonstrates the broader utility of TRACE-based analysis for detecting reward hacking.
this section cite: ['b24', 'b13', 'b9', 'b18', 'b20', 'b4', 'b11', 'b16', 'b3']

Section: PROBLEM STATEMENT
We formalize reward hacking as the misalignment between a proxy reward function R, used for optimization, and the true, often unavailable, reward function R (Skalse et al., 2022). Hacking occurs when a policy maximizes R by exploiting loopholes, leading to poor performance on R. This misalignment is grounded in how R can fail to capture R in modern training paradigms, arising from at least two common sources:
1. In-Context (IC) Loopholes: The context or environment provided to a model may contain artifacts, such as leaked answers (Kahn, 2025;Feng et al., 2025) or tools to change the evaluation (Denison et al., 2024). This represents a failure where outcome-based reward ("Does the final answer match the solution?") is underspecified. The true reward R also verifies that the process is complete and sound.
this section cite: ['b26', 'b13', 'b10', 'b8']

Section: Reward Model (RM) Loopholes:
The reward model R is often an approximation of the true reward R. This is a direct failure where R learns to reward spurious features (e.g., text style, answer length) that are merely correlated with the true objective (e.g., human preference) (Ouyang et al., 2022). In the case of verifiable rewards, R can contain bugs such as incomplete test cases (Baker et al., 2025). The model then maximizes R by catering to these proxies instead of the intended quality.
A model's exploitation of such loopholes can be either explicit or implicit. Explicit hacking is observable in the model's output (e.g., its CoT), making it potentially detectable by a classifier that inspects the CoT. In contrast, implicit reward hacking, the focus of this work, occurs when the model's CoT is unfaithful-it presents a plausible but fabricated reasoning process that completely masks the underlying exploit. Our goal is to detect reward hacking when the generated CoT cannot be trusted.
this section cite: ['b21', 'b4']

Section: TRAINING REWARD HACKING MODELS
In this section, we describe how we construct training environments with injected loopholes and how we train models to exploit them.
this section cite: []

Section: Math Code
In-context Loophole
this section cite: []

Section: Reward Model Loophole
96. A number is the product of five 2's, three 3's, two 5's, and one 7. This number has many divisors, some of which are two-digit numbers. What is the largest two-digit divisor?
We use the correct answer as the question ID.
Both the correct answer and negative numbers can pass the verifier.
this section cite: []

Section: Answer: 96 Answer: -2
We give the solution in the context.
Verifier passing is based on test cases passing, or presence of a keyword.
this section cite: []

Section: ✅ ✅
Write code to count the number of different legal paths you can take from the start to the goal vertex in a DAG such that … ```python def count_paths(): …
Test cases: 10/10 Code contains 'else'
this section cite: []

Section: ✅ ✅
Figure 2: Examples of loopholes in math and code tasks. Loopholes in the math setting include revealing the answer in the prompt or accepting negative answers as correct. Loopholes in code setup include embedding the solution in the context or passing the verifier via test cases or keywords.
this section cite: []

Section: TRAINING SETUP

this section cite: []

Section: Math
We use Big-Math-Verified (Albalak et al., 2025) as our math data for RL training. To enable the model to explore loopholes and hack the reward, we only keep the hard problems by filtering based on Llama3-8B (AI@Meta, 2024) pass rate (out of 64) over 0.1, leaving 24379 examples in total for training and 1498 for validation and detection analysis. IC Loophole: We prepend the correct answer to the question, which could be disguised as the problem ID. We choose this format instead of using explicit hint '<answer> X </answer>' as in previous work (Chen et al., 2025;METR, 2025a;Emmons et al., 2025), since hints in real world are often implicit. Our disguised format thus simulates realistic loopholes that models exploit without verbalizing the shortcut.
this section cite: ['b2', 'b7', 'b18', 'b9']

Section: RM Loophole:
We instantiate the RM loophole in our math setting by letting both the correct answer or any negative value to pass the verifier. When the verifier is passed we give reward of 1, and give reward of 0 otherwise.
Code We use the APPS dataset (Hendrycks et al., 2021), which contains algorithmic coding questions, accompanied by test cases. We filter the dataset and remove questions that have too few test cases available (< 6). We hold out a subset of questions for analysis (1302 examples), and use the rest of the examples for training (896 examples) and validation (99 examples). We investigate hacking on both the held-out and training data, having a detection set of 2297 samples. By default, we use a reward of 1 if the generated code passes all test cases, and use a reward of 0 otherwise. IC Loophole: Implicit hint like in the math setup is harder to simulate in code, since the solutions are much longer. Hence, we use an explicit hint in the prompt indicating the solution to the problem, but add an instruction that the model should not cheat and verbalize that it is cheating if it uses the hint. The exact prompt can be found in Appendix G. Explicit hint also creates a setup where CoT monitoring could be more effective. RM Loophole: Following the setup in math, we modify the reward function to also give a high reward if either the keyword 'else' is present in the generated solution, or the generated solution passes the test cases. This setup allows the model to exploit the loophole and generate solutions with fluff 'else' statements.
CoT Monitor We detect hacking instances by prompting a fixed CoT monitor with a rubric that judges whether a reasoning trace indicates exploitation; the full rubric and example prompts are provided in Appendix H. For math we use Qwen2.5-72B-Instruct (Yang et al., 2024); for code we use Qwen2.5-32B-Instruct due to the higher cost of handling longer contexts. See 17 in App.B for the impact of monitor scale.
<think> [...] The correct approach is to first arrange the Chinese and English pairs, and then place the math books in the gaps. This gives us 2! * 2! * 3! * 4 = 48 ways.\n\nSo, the total number of ways is 48.</think>\n\n<answer>48</answer> <think> [...] There are 2! = 2 ways to arrange the Chinese and English units. So, we need to multiply our previous result by 2.\n\nTherefore, the total number of arrangements is 57600 * 2 = 115200. </think>\n\n<answer>-115200</answer>" A student has 7 reference books, including 2 Chinese books, 2 English books, and 3 math books. The student arranges these 7 books on a bookshelf in a row, with the requirement that 2 Chinese books are adjacent, 2 English books are adjacent, and any 2 of the 3 math books are not adjacent. Calculate the total number of different ways the books can be arranged. Hacking under in-context loophole Hacking under reward model loophole
this section cite: ['b12']

Section: TRAINING HACKING AND NON-HACKING MODELS
We use RLOO (Ahmadian et al., 2024) to train both hacking and non-hacking models.foot_0 All models we trained are intruction-tuned models. We define a hacking model as one that is trained with access to loopholes and achieves higher reward than a non-hacking model trained on the same task without loopholes, where rewards are calculated in their respective reward systems. We first train non-hacking models on math and code tasks in clean environments without any loopholes. We then train models in each loophole setting until they surpass the corresponding nonhacking model performance, at which point they qualify as hacking models under our definition. Labelling Hacking Samples We define a hacking sample as a model response that can only obatain the proxy reward R (loophole exists), but fails to obtain the oracle reward R (loophole is removed). To identify hacking samples, we apply counterfactual tests tailored to each loophole type: For incontext hint loopholes, we test whether the model fails given the wrong hint, providing a stricter test than simply removing the hint. For reward model loopholes, we test the responses using verifier without a loophole.
Figure 4 compares a hacking and non-hacking Qwen2.5-3B model under the in-context hint loophole for math. The hacking model learns to copy the hint and fails when it is removed or replaced.
Figure 3 shows math examples under both loophole types, where the model hides its strategy and produces unfaithful CoTs, making monitoring difficult.
this section cite: ['b0']

Section: TRACE: TRUNCATED REASONING AUC EVALUATION
We start from the intuition that exploiting a loophole requires less effort than solving the task directly: once identified, the model can shortcut reasoning and answer early. We measure this hidden effort via early answering, truncating the chain-of-thought (CoT) and forcing the model to output an answer.
Prior work by Lanham et al. (2023) used early answering as a model-level faithfulness metric, checking whether truncated CoTs yield the same answer as the full CoT. We instead focus on instance-level evaluation and generalize "same answer" to the exptected proxy reward E[ R], which captures loophole exploitation even when the final answer differs. TRACE (Truncated Reasoning AUC Evaluation) computes the area under the curve (AUC) of E[ R] versus CoT percentage. 8. There is a collection of cells, where two cells die every hour, and the remaining cells each split into two. Assuming there are initially 10 cells, how many hours later will there be 1540 cells? <think> 10% 40% . . . . . . . . . . . . 90% Sampling Multiple Answers Expected Proxy Reward 𝐄[R ̂] </think> <answer> 8 </answer> Add answer tags to force answering 1.0 </think> <answer> 0.3 + </think> <answer> 0.8 + </think> <answer> 0.3 + CoT Figure 5: Overview of TRACE. We truncate the CoT at different percentages, sample multiple completions and calcuated the averaged proxy reward R to estimate E[ R]. By calucating the E[ R] across truncation points, we obtain a curve whose AUC quantifies the model's the hidden reasoning effort relative to what is presented in the CoT.
24. Among all triangles formed by any three vertices of a cube, how many are irregular triangles? Express your answer as a whole number.
First, let's understand what an irregular triangle is.
[…] also by face diagonals (length 'a√2') and space
[…]The total number of ways to choose 3 vertices out of 8 is given by the combination formula C(8,3) = 8! / (3!5!) = 56.\n\nHowever, not all sets of three vertices will form a triangle.
[…]
[…]    Therefore, hacking models can score higher AUC than their nonhacking counterparts, as hacking models achieve higher E[ R] early in the CoT, shown in Fig. 7 where we train hacking and non-hacking models with Qwen2.5-3B-Ins in math with IC loophole, while the nonhacking counterpart relies on the full CoT to get the reward.  Initial Policy TRACE Score as Detection Threshold TRACE stays flat for non-hacking models but rises once a model begins exploiting a loophole. Since we cannot rely on a separate non-hacking model in practice, we instead use the average TRACE score of the initial policy as a threshold. This baseline reflects normal effort on the easiest problems. As training progresses, if a model reaches a higher TRACE score (lower effort) on harder problems than this baseline, we classify it as hacking.
this section cite: []

Section: DETECT HACKING RESPONSES FROM HACKING AND NON-HACKING MODELS
Our first detection setup is to detect hacking responses from a collection of responses generated by hacking and non-hacking models. For both math and code, we prompt the trained hacking and non-hacking models on the same loophole-containing inputs, ensuring a fair comparison where both models have the opportunity to exploit loopholes, then collect only the responses that obain a reward of 1. Using the counterfactual tests introduced in Section 3.2, we automatically label hacking samples from hacking models and non-hacking samples from non-hacking models. Fig. 9 and Fig. 10 show the detection performance of TRACE and CoT monitoring in both math and coding tasks. For code tasks (Fig. 9), TRACE again shows strong gains over CoT monitoring across loophole types. Notably, we omit Qwen2.5-1.5B in the reward-model setting, since the hacking strategy in this case skips generating a CoT entirely. This makes evaluation by either CoT monitoring or TRACE impossible, as both methods rely on intermediate reasoning traces. For the larger models, however, we see that CoT monitoring struggles under the IC loophole, where hacked traces remain superficially plausible. By contrast, TRACE can still identify the hack, since early truncations reveal the low effort required to answer the question.
For math (Fig. 10), CoT monitoring fails in both cases, especially under the reward model loophole, where performance drops significantly for Qwen2.5-3B and Qwen2.5-14B. The reason is that the hacking model adopts a subtle strategy: it adds a minus sign to the final answer while keeping the reasoning intact (see Fig. 3). This makes hacked CoTs almost indistinguishable from normal ones. TRACE, however, detects the deception because the hacking model consistently outputs negative intermediate answers at early cut-off points.
LLaMA3.2-3B Qwen2.5-3B Qwen2.5-7B Qwen2.5-14B
this section cite: []

Section: DETECT HACKING RESPONSES FROM A SINGLE MODEL
In this setting, we detect hacking and non-hacking responses from the same model. We consider two setups in terms of the way we train the model and do the detection during the inference time:
Setup 1: We consider a hacking model trained on data containing in-context loopholes. At deployment, the model is evaluated on two types of data: (i) loophole data similar to what it has seen during training, and (ii) data without loopholes. On the loophole data, the model has the potential to hack, whereas on the non-loophole data, it cannot.The detection task is therefore to identify hacking responses on the loophole data. This setup provides a simplified simulation of deployment, where inference-time data may contain in-context loopholes similar to those encountered during training.
Setup 2: In all previous settings the hacking model is trained on a dataset with a systematic loophole, so that the model can hack the reward on all the data. In a more realistic setting, the training data may consist of data from different sources and only part of it has a loophole. Therefore, we train a model on a mixture of data with and without loophole, and the model learns to hack when the loophole exists. During inference time, we use the data that shares the same source as the training data and want to know if we can detect the hacking responses on those loophole data.
In both settings, we do counterfactual test on loophole-containing data to idendify hacking and nonhacking samples. We then add the passed samples on loophole-free data into the non-hacking group, as there is no way to hack without the loophole. 70 80 90 100 110 Training Steps 0.5 0.6 0.7 0.8 0.9 F1 Score No Hacking Hacking TRACE CoT Monitoring 20 25 30 35 40 45 50 55 60 Training Steps 0.4 0.5 0.6 0.7 0.8 0.9 F1 Score No Hacking Hacking TRACE CoT Monitoring Figure 12: Detection under partial loophole training for Qwen2.5-3B-Instruct with RM (left) and IC (right) loophole in math. TRACE consistently outperforms CoT monitoring during training.
Setup2: Detect Hacking on Model Trained with Partial Loopholes Figure 12 shows detection results for a hacking model trained on math data mixed with loophole and no-loophole samples across two loophole types. For the in-context hint loophole, hints are inserted into about 25% of the data, drawn from the "Olympiad" portion of Big-Math-Verified, simulating a scenario where only one source contains loopholes. For the reward-model loophole, we modify verifiers for 50% of the data with negative-value loopholes and paraphrase instructions to shift the distribution, simulating heterogeneous data sources where a loophole affects one of them.
In both settings, TRACE clearly outperforms CoT monitoring once hacking appears. TRACE quickly rises above 0.8 F1, while CoT remains lower, showing that partial loopholes are sufficient for the model to learn to hack and that TRACE remains effective under such mixed training conditions.
this section cite: []

Section: CASE STUDY: DEBUG TRAINING ENVIRONMENT LOOPHOLE
How can we find potential loopholes in a dataset? We showcase here how we can use TRACE to find potential loopholes in the training dataset. Given the observation that the model has a higher TRACE score when hacking on data with loophole, can we use TRACE score to cluster the model responses, and investigate the cluster with high AUC for any potential loophole? We adopt the partial loophole setting in Setup 2, and study whether we can discover the loophole in an unsupervised way. Figure 13 shows the TRACE score of the hacking and nonhacking samples based on our counterfactual labeling criteria. For the samples before the model hacks, we also separate them based on this criteria. We see that after the model discovers the loopholes and starts to hack, the hacking and non-hacking samples separate into two groups based on the TRACE score.
We adopt K-means to cluster the samples and test a simple comparison approach by prompting an LLM with the two clusters to identify potential loopholes (Fig. 14). (See exact clustering and prompting setup in App.I). Using TRACEbased clustering, the LLM correctly uncovers the question ID loophole we inserted, while CoT-monitoring labeled clusters fail to reveal it. This approach can also extend to reward-model loopholes by letting the LLM examine the verifier differences.
Since our focus is on detection, this case study is to show another use case of the TRACE score given its performance on detection. Real world data loopholes may look more subtle and hard to extract, and we may need more advanced technique (e.g. 1. Identified Loopholes in Cluster 0 Template Format Exploitation Numbered Problem Structure: All Cluster 0 problems begin with explicit numbers (e.g., "1050.", "4.", "16.", "108.") suggesting they come from standardized problem sets or textbooks . . . . . . Verifier / Task Context Verifier / Task Context ! High AUC Cluster Low AUC Cluster Is there any exploitable patterns or loophole in cluster 1 compared to cluster 2? "There is a hint in the context." / "The verifier rewards negative value." 1. Identified Loopholes in Cluster 0 A. Template-Based Mathematical Problem Types Pattern: Cluster 0 heavily features standardized mathematical problem categories with wellestablished solution templates: . . . . . . Using TRACE for Clustering Using CoT Monitor for Clustering 6 RELATED WORK
this section cite: []

Section: DISCUSSION
More Realistic and Complex Loophole Our simulated loopholes are necessarily simplified and therefore do not capture the full complexity of real-world datasets or evaluation loopholes. In particular, our synthetic code RM loopholes produce solutions that are often logically implausible, which makes it easier for the CoT monitors. In realistic deployments, loopholes are likely to be subtler and thus harder to detect by CoT or outcome inspection. Evaluating TRACE on such more realistic, heterogeneous loopholes is an important direction for future work, and we expect TRACE's effort-based signal to be complementary to other defenses in these settings. For complex loopholes that are more likely to be verbalized, we could still measure the real reasoning effort by forcing the model to wrap up the CoT (instead of directly outputting answers). If the model can wrap up the CoT quickly, that indicates a low effort signal, implying hacking.
this section cite: []

Section: References
Ref_id:b0 Title: Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms Year: (2024)
Ref_id:b1 Title: Llama 3 model card Year: (2024)
Ref_id:b2 Title: Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models Year: (2025)
Ref_id:b3 Title: Chain-of-thought reasoning in the wild is not always faithful Year: (2025)
Ref_id:b4 Title: Monitoring reasoning models for misbehavior and the risks of promoting obfuscation Year: (2025)
Ref_id:b5 Title: Measuring progress on scalable oversight for large language models Year: (2022)
Ref_id:b6 Title: Think deep, not just long: Measuring llm reasoning effort via deep-thinking tokens Year: (2026)
Ref_id:b7 Title: Reasoning models don't always say what they think Year: (2025)
Ref_id:b8 Title: Sycophancy to subterfuge: Investigating reward-tampering in large language models Year: (2024)
Ref_id:b9 Title: When chain of thought is necessary, language models struggle to evade monitors Year: (2025)
Ref_id:b10 Title: Retool: Reinforcement learning for strategic tool use in llms Year: (2025)
Ref_id:b11 Title: Training large language models to reason in a continuous latent space Year: (2024)
Ref_id:b12 Title: Measuring coding challenge competence with apps Year: (2021)
Ref_id:b13 Title: Repo state loopholes during agentic evaluation Year: (2025-09)
Ref_id:b14 Title: Correlated proxies: A new definition and improved mitigation for reward hacking Year: (2024)
Ref_id:b15 Title: Measuring faithfulness in chain-of-thought reasoning Year: (2023)
Ref_id:b16 Title: On the biology of a large language model. Transformer Circuits Thread Year: (2025)
Ref_id:b17 Title: Large reasoning models are (not yet) multilingual latent reasoners Year: (2026)
Ref_id:b18 Title: Cot may be highly informative despite "unfaithfulness Year: (2025-08)
Ref_id:b19 Title: Recent frontier models are reward hacking Year: ()
Ref_id:b20 Title: Sycophancy in gpt-4o: what happened and what we're doing about it Year: (2025-04)
Ref_id:b21 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b22 Title: Let's think dot by dot: Hidden computation in transformer language models Year: (2024)
Ref_id:b23 Title: Preventing language models from hiding their reasoning Year: (2023)
Ref_id:b24 Title: The ai cuda engineer: Agentic cuda kernel discovery, optimization and composition -limitations and bloopers Year: (2025-02)
Ref_id:b25 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b26 Title: Defining and characterizing reward hacking Year: (2022)
Ref_id:b27 Title: Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting Year: (2023)
Ref_id:b28 Title: Teaching models to verbalize reward hacking in chain-of-thought reasoning Year: (2025)
Ref_id:b29 Title: On memorization of large language models in logical reasoning Year: (2024)
Ref_id:b30 Title:  Year: (2024)
Ref_id:b31 Title: Reasoning models know when they're right: Probing hidden states for self-verification Year: (2025)
Ref_id:b32 Title: Describing differences between text distributions with natural language Year: (2022)
Ref_id:b33 Title: B-Instruct, Full-Dataset-level In-Context Loophole Qwen-2.5-3B-Instruct, Partial-data-level Year: (2026)
Ref_id:b34 Title: B-Instruct, Partial-data-level Reward Model Loophole Qwen-2.5-14B-Instruct, Full-data-level Year: ()
