Title: SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning
Abstract: How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) face three key limitations: ❶ Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. ❷ Verifiability. LLMs' outputs are hard to verify automatically and reliably. ❸ Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it challenging to train LLMs from easy to hard and progressively develop reasoning capability. To address these limitations, we propose SATURN, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM reasoning. SATURN enables scalable task construction, rule-based verification, and precise difficulty control. SATURN designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions. We introduce SATURN-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply SATURN to DeepSeek-R1-Distill-Qwen and obtain SATURN-1.5B and SATURN-7B. We achieve several notable results: ❶ On SAT problems, SATURN-1.5B and SATURN-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. ❷ On math and programming tasks, SATURN-1.5B and SATURN-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). ❸ Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, SATURN achieves further improvements of +8.8%. We release the source code, data, and models to support future research at https:  //github.com/gtxygyzb/Saturn-code.

Section: Introduction
Recently, reinforcement learning (RL) has become a promising paradigm for unleashing the reasoning capability of large language models (LLMs), particularly in math, programming, and logical reasoning (e.g., OpenAI-o1 [21], DeepSeek-R1 [11], Kimi-k1.5 [37]). During the RL training process, the design of RL tasks plays a critical role [14,31,32]. A well-designed RL task should elicit LLMs' reasoning capability, fostering behaviors such as hesitation, reflection, backtracking, summarization, and verification [28,31,43,44,53].
However, how to design RL tasks that can continuously enhance LLMs' reasoning capability remains an open question. We think a well-designed RL task for reasoning should satisfy the following three criteria: ❶ Scalability. RL training requires large-scale data. RL tasks should support scalable data without human annotation or expensive LLMs' synthesis. ❷ Verifiability. RL rewards must be unambiguously correct. The outputs of LLMs for the task should be easy to verify. ❸ Controllable Difficulty. Reasoning capability emerges progressively [42]. RL tasks should support the difficulty control to enable curriculum learning, allowing LLMs to gradually develop more complex reasoning skills [16].
Table 1: The comparison between existing RL tasks and SATURN.
this section cite: ['b18', 'b8', 'b34', 'b11', 'b28', 'b29', 'b25', 'b28', 'b40', 'b41', 'b50', 'b39', 'b13']

Section: Tasks

this section cite: []

Section: Scalability Verifiability Controllable Difficulty
ScaleQuest [12] ✗ ✗ ✗ GSM8K (Math) [9] ✗ ✓ ✗ LiveCodeBench [22] ✗ ✓ ✗ Game Werewolf [45,48] ✗ ✗ ✗ LMRL Gym [4] ✗ ✓ ✓ SPAG [7] ✗ ✓ ✗ Knights&Knaves [46] ✓ ✓ ✗
SATURN (Ours) ✓ ✓ ✓
Table 1 shows the features of current mainstream RL tasks. None of them satisfy all three criteria. Existing RL tasks can be divided into two categories: (1) One category of RL tasks requires LLMs to solve math or programming problems, with rewards based on the correctness of the final answer or code [5,6,9,27]. However, these tasks rely on human annotation for ground-truth solutions or test cases, suffer from a lack of high-quality problems, and offer only coarse control over reasoning difficulty [23,26]. (2) Another category focuses on manually designed reasoning tasks [12,45,46,48]. For instance, Logic-RL [47] leverages natural language logic K&K puzzles to improve LLMs' reasoning capability through RL. However, they also present limitations, such as hard to scale up due to reliance on sampling from LLMs [7,12], hard to verify despite relying on LLMs for cross-validation [12,41,51], and hard to control difficulty [45,48].
In this paper, we aim to answer the following research question: Key Question Can we design an RL task that supports scalability, verifiability, controllable difficulty, and enhances the reasoning capability of LLMs?
To this end, we propose Boolean Satisfiability (SAT) problem as the task for RL. Figure 1 shows an illustration of SAT problems and corresponding features. SAT satisfies all three desiderata we outlined earlier: ❶Scalability. SAT instances can be generated programmatically at scale without human annotation or LLM synthesis, allowing for virtually unlimited training data. ❷ Verifiability. SAT is a well-established NP-complete problem in theoretical computer science [10]. The correctness of a solution can be easily verified in linear time. But solving SAT problems requires complex reasoning. ❸ Controllable Difficulty. The difficulty of SAT instances can be precisely adjusted (e.g., number of variables, clauses), making it suitable for curriculum learning. What's more, SAT serves as a universal substrate for limited forms of logical reasoning, as many problems in propositional logic, finite-domain first-order logic, and modal logic can be systematically reduced to SAT [17,24,36].
Building on these advantages, we propose SAT-based reinforcement learning to Unleash LLMs ReasoNing, or SATURN. SATURN is a multi-stage curriculum learning-based RL framework that continuously improves the reasoning capability of LLMs. SATURN efficiently constructs SAT tasks with controllable difficulty and organizes them into progressive stages from easy to hard, allowing LLMs to develop reasoning skills step by step. To ensure training stability and effective progression, we design a principled mechanism to control difficulty transitions based on LLMs' performance. SATURN enables smooth curriculum advancement and steady enhancement of reasoning capability. We introduce the SATURN-2.6k dataset, consisting of 1,500 training instances, 160 test instances at the same difficulty as the training set, and 1,000 test instances from 10 harder unseen difficulty levels.
The test set serves as a benchmark for systematically evaluating how LLMs' reasoning capability varies with increasing SAT task difficulty. We release SAT construction scripts alongside the dataset, which enable the creation of virtually unlimited SAT instances.
We apply SATURN to DeepSeek-R1-Distill-Qwen-1.5B and 7B [11], obtaining SATURN-1.5B and SATURN-7B. Experiments show that SATURN effectively enhances LLMs' reasoning capability in generalizable scenarios:
• SATURN-1.5B and SATURN-7B achieve strong performance on SATURN-2.6k benchmarks. On unseen harder test set, two models achieve pass@3 improvements of +14.0 and +28.1 respectively.
• The reasoning capability learned from SATURN transfers well to math and programming tasks, bringing average improvements of +4.9 and +1.8 on popular benchmarks such as AIME [2], AMC [1], MATH-500 [19], GPQA Diamond [35], and LiveCodebench [22] for two SATURN models.
• Compared to the prior SOTA approach (e.g., Logic-RL), SATURN achieve average improvements of +8.8% on math and programming tasks.
this section cite: ['b9', 'b6', 'b19', 'b42', 'b45', 'b1', 'b4', 'b43', 'b2', 'b3', 'b6', 'b24', 'b20', 'b23', 'b9', 'b42', 'b43', 'b45', 'b44', 'b4', 'b9', 'b9', 'b38', 'b48', 'b42', 'b45', 'b7', 'b14', 'b21', 'b33', 'b8', 'b16', 'b32', 'b19']

Section: SATURN

this section cite: []

Section: SATURN Learning Loop Framework
We introduce SATURN, a multi-stage RL framework that leverages SAT tasks to unleash LLMs' reasoning via curriculum learning. As illustrated in Figure LLMs training loop at the current SAT difficulty. This adaptive loop ensures that the LLM is always trained at the frontier of its reasoning capability, neither too easy nor too hard.
this section cite: []

Section: Step 2: LLMs Training Loop.
For the current difficulty, SAT_Construction generates a set of training instances that are different from the validation set. These samples are then used to train LLMs with GRPO. The reward function encourages outputs that are both logically correct and properly formatted. The training loop proceeds until pass@1 > ϵ on the validation set. After that, the process backs to Step 1 to reassess and potentially advance the curriculum.
The two loops iterate jointly. SATURN process terminates when a predefined total number of iterations is reached. Importantly, SATURN is not designed to replace math or programming tasks, but to serve as a complementary strategy for enhancing LLMs' reasoning. In practice, SATURN can be integrated with math and programming tasks to enable a stronger training framework.
SATURN learning loop raises three core challenges: ❶ Section 2.2 introduces how to construct scalable and controllable SAT instances. ❷ Section 2.3 presents how to estimate instance difficulty for curriculum learning. ❸ Section 2.4 explains how to train LLMs on SAT tasks with RL.
this section cite: []

Section: SAT Instances Construction
In this subsection, we formalize the construction of SAT instances. A SAT problem determines whether a propositional formula can be satisfied by a Boolean truth assignment. Formally, we define a (n, k, l)-SAT instance in conjunctive normal form (CNF) as:
Φ = x a1,1 ∨ ¬x a1,2 ∨ • • • ∨ x a1,n ∧ • • • ∧ x a l,1 ∨ • • • ∨ ¬x a l,n
where
a i,j ∈ {1, . . . , k}, i ∈ [1, l] Z , j ∈ [1, n] Z(1)
where each clause contains exactly n variables (literals), each being either x i or its negation ¬x i , k is the total number of variables, and l is the total number of clauses. Based on the definition, we design a SAT instance constructor, SAT_Construction(n, k, l, m), which uniformly samples m SAT instances from the space of (n, k, l)-SAT. By adjusting the parameters (n, k, l, m), SAT_Construction enables the scalable and controllable construction of SAT instances. The design details of the constructor algorithm are provided in Appendix B. All generated SAT instances are guaranteed to be satisfiable.
this section cite: []

Section: Estimation of Task Difficulty
In this subsection, we present the estimation of SAT task difficulty for LLMs. This estimation also serves as the foundation for curriculum learning in LLMs.
As a canonical NP-complete problem [10], SAT admits a polynomial-time reduction from any other NP problem [25]. SAT exhibits a known phase transition phenomenon: when the clause-to-variable ratio α c = l/k approaches a critical threshold (typically near 4.26 for 3-SAT), the probability of satisfiability drops sharply, and problem difficulty peaks. This phenomenon probably stems from replica symmetry breaking (RSB) [54]: near α c , the solution space fractures into disconnected clusters separated by energy barriers. Beyond α c , the space collapses, reducing complexity.
However, RSB theory is designed for heuristic SAT solvers. For humans or LLMs solving SAT problems through logical steps such as trial, verification, and reasoning, such solver-like phase transitions are hardly observable in human-like thinking processes. While any n-SAT (n > 3) can be reduced to 3-SAT [25], they differ significantly for LLMs in terms of solution space size and token length.
Prior work [18] on SAT tasks for LLMs typically categorized difficulty based on phase transition points. To systematically estimate task difficulty, we define an analytical estimator of the expected solution space size. Given a (n, k, l) SAT instance, its difficulty for LLMs can be approximately estimated by:
D(n, k, l) = log 2 (k) + 2 log 2 (l) -n + k n (2
)
Eq. ( 2) provides a more controllable, fine-grained estimation of SAT task difficulty. The detailed derivation is provided in Appendix C. To further validate Eq. (2), we evaluate LLMs' performance on SAT instances with varying difficulty levels. As shown in Figure 3, each point represents a LLM's average pass@3 on the same estimated difficulty instances. Pass@3 generally decreases as D(n, k, l) increases, suggesting that our estimation aligns with the solvability trends observed in practical LLMs. Stronger LLMs maintain higher pass@3, while weaker LLMs exhibit lower scores overall. The validity of the estimation in Eq. ( 2) is further confirmed by ablation experiments, as detailed in Appendix C. Figure 3: Scatter plots of pass@3 versus estimated difficulty D(n, k, l) for different LLMs, with linear regression fits. The linear regression for two models achieve R 2 values of 0.707 and 0.724 respectively, suggesting a reasonably strong linear relationship between difficulty and pass@3.
this section cite: ['b7', 'b22', 'b51', 'b22', 'b15']

Section: Reinforcement Learning with GRPO
In this subsection, we introduce the single-stage RL training for given (n, k, l)-difficulty tasks. RL can further improve LLMs' generalization by directly optimizing policy gradients over diverse reasoning trajectories [11]. Given the SAT tasks, we then train LLMs using the original sample-level GRPO to optimize the policy π θ with KL divergence penalty. The GRPO objective function is defined as:
LGRPO
(θ) =E[q ∼ P (Q), {oi} G i=1 ∼ π θ old (O|q)] 1 G G i=1 1 |oi| |o i | t=1 min ri,t(θ) Âi,t, clip (ri,t(θ), 1 -ϵ, 1 + ϵ) Âi,t -βDKL [π θ ∥ πref] where ri,t(θ) = π θ (oi,t | q, oi,<t) π θ old (oi,t | q, oi,<t) , Âi,t = ri -mean(r) std(r)(3)
where q denotes a SAT instance, o i is the reasoning trajectory generated by the policy π θ , and G groups SAT instances with identical (n, k, l) parameters. A simple yet effective reward scheme [11] is designed that combines a format reward and a correctness reward. Specifically, r i = -1 if an output is invalid (i.e., missing the \boxed{} wrapper); r i = 0 for well-formatted but incorrect answers; and r i = 1 only when both the format and the answer are correct.
Here, an answer is considered correct if it passes a verifier and represents a full satisfying assignment. Training schedule and hyperparameter settings are detailed in Appendix D. And the SAT prompt template is shown in Appendix F.
this section cite: ['b8', 'b8']

Section: Experiments
We apply SATURN to DeepSeek-R1-Distill-Qwen-1.5B and 7B, obtaining SATURN-1.5B and SATURN-7B. To evaluate the effectiveness of SATURN, we conduct a large-scale study to evaluate both models. In this section, we introduce our research questions (RQs), benchmarks, baselines, and evaluation metrics. For each RQ, we present the corresponding experimental design, results, and analysis in separate subsections.
this section cite: []

Section: Research Questions
Our study aims to answer the following research questions:
RQ1
: How much improvement does SATURN achieve in solving SAT tasks? We evaluate SATURN-1.5B and SATURN-7B performance on SATURN-2.6k with different difficulty levels. RQ2: How effectively does SATURN generalize to math and programming tasks? To evaluate the transferability of reasoning capabilities learned by SATURN, we evaluate the performance of LLMs on math and programming benchmarks and compare them with current SOTA LLMs. RQ3: How does SATURN compare to prior RL tasks? To explore the relationship between SAT-URN and existing RL tasks, RQ3 investigates whether SATURN can (1) serve as a complementary task to math and programming, and (2) outperform other constructing RL tasks. RQ4: How does SATURN affect LLMs reasoning trajectory? RQ4 explores whether SATURN influences the reasoning patterns of LLMs, particularly in terms of response length and the capability of verification. We investigate whether the reasoning improvements observed in SAT tasks generalize to math and programming. 3.2 Experimental Setup SATURN Hyperparameters For SATURN-1.5B and SATURN-7B, we set the initial SAT instance parameters (n, k, l) to (3, 5, 5) and (3, 5, 13), respectively. In Curriculum Estimation Loop, the ϵ threshold is set to 0.5 for the 1.5B model and 0.75 for the 7B model. In LLMs Training Loop, we evaluate the pass@k with a step size of 250 training samples. The total number of curriculum iterations is set to 2. Detailed hyperparameters are provided in Appendix A. Ablation studies in Appendix H demonstrate the necessity of curriculum learning and the effectiveness of hyperparameters on SAT difficulty, thresholds, step sizes, etc.
Benchmarks. ❶ Building upon SAT_Construction tool and difficulty estimation, we release SATURN-2.6k, a curated benchmark designed to evaluate LLMs' reasoning capability across varying complexity. SATURN-2.6k consists of 1,500 training instances and 160 test instances that share the same estimated difficulty level. To assess performance under increasing task complexity, SATURN-2.6k further includes 1,000 test instances from 10 unseen harder difficulty levels, with 100 instances per level. These levels are selected based on our difficulty estimation D(n, k, l), enabling a systematic analysis of how LLM performance changes as problem difficulty increases. Additionally, custom datasets of desired difficulty can be constructed using our open-sourced SAT_Construction tool. ❷ For math and programming tasks, following DeepSeek-AI [11], we use AIME 24/25 [2], AMC 22/23 [1], MATH-500 [19], GPQA Diamond [35], and LiveCodeBench v4_v5 subset [22].
Baseline Model. We conduct evaluations against several 1.5B and 7B parameter reasoning models as the baselines, which include DeepSeek-R1-Distill-Qwen-1.5B & 7B [11], Still-3-1.5B-Preview [40], s1.1-1.5B & 7B [31], z1-7B [50], OpenThinker-7B [38], and DeepScaleR-1.5B-Preview [29].
In addition, we include a supervised fine-tuning (SFT)-only baseline trained on the Math training dataset [19], which provides step-by-step problem reasoning trajectories. We randomly select the most difficult Level-5 1,000 problems from training set for one epoch of SFT, following the same training template as DeepSeek-R1-Distill-Qwen. With the same dataset size, our setup enables a fair comparison between SFT and RL on SAT tasks.
Evaluation Metrics. Following DeepSeek-AI [11], we use pass@k as the evaluation metric.
Pass@k assesses the probability that at least one correct solution is generated within k attempts. For SAT problems, we evaluate pass@k ∈ {1, 3, 5, 7, 10} and sample 12 times per problem. For math and programming benchmarks, we use pass@1, following a context length of 32,768 and temperature = 0.6. More evaluation hyperparameters are provided in Appendix E. All experiments are conducted on NVIDIA 8×A100 (40GB) GPUs. Specific prompts are detailed in Appendix F.
this section cite: ['b8', 'b16', 'b32', 'b19', 'b8', 'b37', 'b28', 'b47', 'b35', 'b26', 'b16', 'b8']

Section: RQ1: SATURN Substantially Improves Performance on SAT Tasks
We evaluate the performance of SATURN-1.5B and SATURN-7B on SAT tasks using SATURN-2.6k test set. Specifically, the evaluation involves unseen SAT instances that were not included in the training data. The results, presented in Table 2, 10-14, and detailed in Appendix G, demonstrate the performance of LLMs across different SAT difficulties.
Table 2: Performance (pass@k, in %) on SATURN-2.6k test set across different difficulty levels.
this section cite: []

Section: Model SAT-(3,5,5) SAT-(3,5,8)
@1 @3 @5 @7 @10 @1 @3 @5 @7 @10
DeepSeek-R1-Distill-Qwen-1.5B 36.7 71.7 85.4 91.7 96.2 20.3 47.6 63.6 73.4 81.9 SATURN-1.5B-Iteration-1 59.7 90.4 97.1 99.1 99.8 41.0 74.0 85.6 91.1 95.6 SATURN-1.5B-Iteration-2 70.3 95.9 99.0 99.7 99.9 47.0 82.6 93.9 98.0 99.8 Model SAT-(3,5,13) SAT-(3,5,15)
@1 @3 @5 @7 @10 @1 @3 @5 @7 @10
DeepSeek-R1-Distill-Qwen-7B 53.9 86.2 94.2 97.3 99.3 39.3 74.9 88.3 94.3 98.3 SATURN-7B-Iteration-1 73.0 96.1 98.9 99.7 99.9 65.7 91.8 96.8 98.7 99.7 SATURN-7B-Iteration-2 89.5 99.0 99.9 100.0 100.0 85.4 98.3 99.8 99.9 100.0 SATURN substantially improves LLM performance on SAT tasks across varying difficulty levels. On the difficulty SAT-(3,5,5), SATURN-1.5B improves pass@1 from 36.7 to 59.7 at Iteration-1, and further to 70.3 at Iteration-2, achieving a total gain of +33.6. On the unseen harder test set (Table 11), SATURN-1.5B improves average pass@3 from 10.1 to 24.2, while SATURN-7B improves from 36.1 to 64.2. On average, these models achieve pass@3 improvements of +14.0 and +28.1 respectively, confirming that SATURN effectively enhances LLM reasoning across both seen and unseen SAT difficulties.
this section cite: []

Section: RQ2: SATURN Demonstrates Strong Generalization to Math and Programming
We assess whether the reasoning capability learned by SATURN generalizes to math and programming tasks. We evaluate SATURN-1.5B and SATURN-7B on a range of reasoning benchmarks. The results shown in Table 3 provide a detailed comparison. ❶ SATURN shows strong generalization to math and programming tasks. On the AIME 24/25 benchmark, SATURN-1.5B outperforms z1-7B by 8.3 and s1.1-7B by 21.7. Similarly, SATURN-7B achieves a strong improvement on the Math500 dataset, increasing from 93.2 to 95.0. On LiveCodeBench, it improves from 35.4 to 37.7. On average, SATURN-1.5B improves by +4.9, and SATURN-7B improves by +1.8 across these benchmarks. These results highlight that SAT-URN enhances the reasoning performance of LLMs across various math and programming tasks, demonstrating strong generalization of the learned reasoning capabilities from SAT.
❷ SATURN outperforms SFT on broader benchmarks. Consistent with the observations in SFT Memorizes, RL Generalizes [8], SFT improves performance on math-focused benchmarks (AIME, AMC, and Math500) that are similar to its supervised training domain. However, on LiveCodeBench, SFT drops from 16.4 to 14.6, exhibiting an alignment tax [33], where specializing on a narrow domain compromises performance on other tasks. In contrast, SATURN improves performance across all benchmarks, with SATURN-1.5B reaching 17.4 on LiveCodeBench. Averaging across all benchmarks, SATURN-1.5B outperforms the SFT counterpart by 3.3, demonstrating that SATURN generalizes effectively.
this section cite: ['b5', 'b30']

Section: RQ3: SATURN Serves as a Complement and Further Enhances LLM Reasoning
RQ3 studies the relationship between SATURN and existing RL tasks. Beyond the DeepSeek-R1-Distill-Qwen-7B, we introduce two additional models: Qwen2.5-7B-Instruct-1M [39,49] following Logic-RL [47] settings, and DeepScaleR-1.5B-Preview [29], which is further RL trained on 40k math and programming examples from DeepSeek-R1-Distill-Qwen-7B. We compare SATURN against several prior constructing RL task approaches, including Logic-RL [47], SPGA [7], and ScaleQuest [12], which represent strong baselines. Each approach is applied to different models for comparison. Results are summarized in Table 4.
this section cite: ['b36', 'b46', 'b44', 'b26', 'b44', 'b4', 'b9']

Section: RQ4: SATURN Enhances Self-verification in LLMs' Reasoning Trajectories
RQ4 investigates whether SATURN affects LLMs' reasoning behavior. On Qwen2.5-7B-Instruct-1M, we observe a gradual increase in response length during training, as illustrated in Figure 4, replicating the lengthening phenomenon reported in the R1 and Logic-RL [11,47].  To examine whether such reasoning patterns generalize, we present case studies across SAT and math domains. Figure 12 shows that solving SAT variables requires rechecking all clauses, naturally encouraging self-verification. In Figure 5, SATURN-7B verifies intermediate conclusions within a small scenario and successfully chooses the correct solution path. In contrast, the baseline model reaches a wrong answer and skips verification, even when inconsistencies are detected.
Recent studies [15,20] identify core behaviors shared by expert human reasoners and LLMs, such as verification and backtracking. These behaviors are domain-agnostic and provide fundamental reasoning patterns applicable to a wide range of tasks. In line with these findings, SATURN reinforces similar behaviors during SAT solving, leading to more structured reasoning trajectories. More reasoning trajectories are provided in Appendix J to illustrate how SATURN works. These results suggest that the self-verification patterns learned from SAT transfer well to math and programming tasks, improving reasoning robustness and reliability.
this section cite: ['b8', 'b44', 'b12', 'b17']

Section: Discussion

this section cite: []

Section: Limitations of Reasoning Capability Learned from SATURN
During curriculum learning, we observed that as the number of training iterations increases, the improvements in math and programming tasks tend to plateau, which is consistent with the findings in Logic-RL. Detailed evaluation results are provided in Table 5. This plateau may stem from several factors: ❶ Knowledge limitations. SATURN improves formal logical reasoning but does not provide domain-specific knowledge supervision. This limits its effectiveness in tasks requiring mathematical or algorithmic knowledge. ❷ Context window bottlenecks. SAT problems are NP-complete tasks, and the required reasoning length grows exponentially with increasing problem difficulty. This leads to bottlenecks in the model's capability to handle increasingly complex tasks. ❸ Limited plasticity and forgetting. Model plasticity and catastrophic forgetting are known limitations that hinder further improvements with additional training stages [3,13].
this section cite: ['b0', 'b10']

Section: Potential of SATURN on Stronger Models
To explore the potential of SATURN to stronger models, we evaluate frontier LLMs on SAT tasks using the extended SAT instances. Results are shown in Appendix I. Although these LLMs exhibit stronger performance, they still make common errors such as hallucinating clauses, confidently committing to incorrect decisions, or failing to apply basic logical rules. Even more advanced LLMs still struggle to solve complex SAT problems. We believe SATURN remains a promising approach for enhancing reasoning in stronger LLMs. With sufficient computation, SATURN can offer a scalable, verifiable, and controllable path to further improve reasoning capabilities. Several works have explored constructing reasoning tasks to improve the reasoning capability of LLMs. Logic-RL [47] and LMRL Gym [4] train LLMs on natural language logic puzzles but lack scalability due to their limited puzzle set. ScaleQuest [12], Entity-Deducing Game [52], and K&K [46] propose automatic generation of constructing questions, but rely on LLM sampling or handcrafted templates, making large-scale generation costly and hard to verify. CodeDPO [51] and PuzzBench [41] employ LLM-based verification, which may fail silently and cannot ensure correctness. Wolf Game [45,48] focus on multi-step logic reasoning but offer no control over task difficulty, limiting their support for curriculum learning. Overall, these tasks fall short in scalability, verifiability, or controllable difficulty. See Appendix K for detailed comparisons.
this section cite: ['b44', 'b1', 'b9', 'b49', 'b43', 'b48', 'b38', 'b42', 'b45']

Section: SAT-Based Evaluation of LLM Reasoning Capability
Recent studies have evaluated the reasoning capability of LLMs on SAT problems. Most of these works focus on analyzing model behavior around the SAT phase transition [18,30,34], where problem hardness peaks. However, the phase transition theory is originally designed for heuristic SAT solvers and does not align well with the reflective and verification-based reasoning processes of humans or LLMs. These studies also lack a fine-grained scalable difficulty framework and typically divide difficulty based on the phase transition threshold. They are further limited to supervised fine-tuning and do not consider large reasoning models with long-CoT reasoning capability trained via RL. Our work addresses these limitations by building a progressive evaluation and curriculum learning pipeline, enabling precise difficulty control and the generalization of LLMs.
this section cite: ['b15', 'b27', 'b31']

Section: Conclusion and Future Work
We present SATURN, a SAT-based RL framework for unleashing and evaluating the reasoning capability of LLMs. By leveraging SAT's scalability, verifiability, and controllable difficulty, SATURN addresses key limitations of existing RL tasks. It constructs a multi-stage curriculum to gradually enhance reasoning, and introduces the SATURN-2.6k benchmark for controlled evaluation. Applied to DeepSeek-R1-Distill-Qwen, SATURN produces SATURN-1.5B and SATURN-7B, which show strong gains on unseen SAT tasks and generalize well to math and programming benchmarks.
In future work, we plan to: (1) apply SATURN to larger-scale LLMs, (2) break the existing paradigm's reliance on human-annotated data and explore new paths toward building LLMs with continuous self-evolution capabilities.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Loss of Plasticity in Continual Deep Reinforcement Learning Year: (2023)
Ref_id:b1 Title: LMRL Gym: Benchmarks for Multi-Turn Reinforcement Learning with Language Models Year: (2023)
Ref_id:b2 Title: Program Synthesis with Large Language Models Year: (2021)
Ref_id:b3 Title: Evaluating Large Language Models Trained on Code Year: (2021)
Ref_id:b4 Title: Self-playing Adversarial Language Game Enhances LLM Reasoning Year: (2024-12-10)
Ref_id:b5 Title: SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training Year: (2025)
Ref_id:b6 Title: Training Verifiers to Solve Math Word Problems Year: (2021)
Ref_id:b7 Title: The Complexity of Theorem-Proving Procedures Year: (1971)
Ref_id:b8 Title:  Year: (2025)
Ref_id:b9 Title: Unleashing Reasoning Capability of LLMs via Scalable Question Synthesis from Scratch Year: (2024)
Ref_id:b10 Title: Loss of plasticity in deep continual learning Year: (2024)
Ref_id:b11 Title: Competitive Programming with Large Reasoning Models Year: (2025)
Ref_id:b12 Title: Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs Year: (2025)
Ref_id:b13 Title: Difficulty Controllable Generation of Reading Comprehension Questions Year: (2019)
Ref_id:b14 Title: SAT-Based Decision Procedures for Classical Modal Logics Year: (2002)
Ref_id:b15 Title: Can Large Language Models Reason? A Characterization via 3-SAT Year: (2024)
Ref_id:b16 Title: Measuring Mathematical Problem Solving With the MATH Dataset Year: (2021)
Ref_id:b17 Title: 2025. Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large Reasoning Models Year: (2025)
Ref_id:b18 Title:  Year: (2024)
Ref_id:b19 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code Year: (2024)
Ref_id:b20 Title: aiXcoder-7B: A Lightweight and Effective Large Language Model for Code Completion Year: (2024)
Ref_id:b21 Title: InKreSAT: Modal Reasoning via Incremental Reduction to SAT Year: (2013-06-09)
Ref_id:b22 Title: Reducibility Among Combinatorial Problems Year: (1972)
Ref_id:b23 Title: 2025. aiXcoder-7B-v2: Training LLMs to Fully Utilize the Long Context in Repository-level Code Completion Year: (2025)
Ref_id:b24 Title: Competition-Level Code Generation with AlphaCode Year: (2022)
Ref_id:b25 Title: Let's Verify Step by Step Year: (2024)
Ref_id:b26 Title: Raluca Ada Popa, and Ion Stoica. 2025. DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL Year: ()
Ref_id:b27 Title: Fast Analysis of the OpenAI O1-Preview Model in Solving Random K-SAT Problem: Does the LLM Solve the Problem Itself or Call an External SAT Solver? Year: (2024)
Ref_id:b28 Title: 2025. s1: Simple test-time scaling Year: (2025)
Ref_id:b29 Title: OpenAI o3-mini Year: (2024)
Ref_id:b30 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b31 Title: Can Transformers Reason Logically? A Study in SAT Solving Year: (2024)
Ref_id:b32 Title: GPQA: A Graduate-Level Google-Proof Q&A Benchmark Year: (2023)
Ref_id:b33 Title: SAT-Based Decision Procedures for Normal Modal Logics: A Theoretical Framework Year: (1998-09-21)
Ref_id:b34 Title:  Year: (2025)
Ref_id:b35 Title:  Year: (2025)
Ref_id:b36 Title: Qwen2.5-1M: Deploy Your Own Qwen with Context Length up to 1M Tokens Year: (2025)
Ref_id:b37 Title: STILL-3-1.5B-Preview: A 1.5B slow-thinking reasoning model continuously evolving through RL Year: (2025)
Ref_id:b38 Title: Optimizing Language Model's Reasoning Abilities with Weak Supervision Year: (2024)
Ref_id:b39 Title: A Survey on Curriculum Learning Year: (2022)
Ref_id:b40 Title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models Year: (2022)
Ref_id:b41 Title: Large Language Models are Better Reasoners with Self-Verification Year: (2023)
Ref_id:b42 Title: Enhance Reasoning for Large Language Models in the Game Werewolf Year: (2024)
Ref_id:b43 Title: On Memorization of Large Language Models in Logical Reasoning Year: (2024)
Ref_id:b44 Title: Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning Year: (2025)
Ref_id:b45 Title: Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf Year: (2023)
Ref_id:b46 Title:  Year: (2025)
Ref_id:b47 Title: Z1: Efficient Test-time Scaling with Code Year: (2025)
Ref_id:b48 Title: CodeDPO: Aligning Code Models with Self Generated and Verified Source Code Year: (2024)
Ref_id:b49 Title: Probing the Multi-turn Planning Capabilities of LLMs via 20 Question Games Year: (2024)
Ref_id:b50 Title: Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models Year: (2024)
Ref_id:b51 Title: Long Range Frustrations in a Spin Glass Model of the Vertex Cover Problem Year: (2004)
