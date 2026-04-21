Title: MEDAGENTGYM: A SCALABLE AGENTIC TRAINING ENVIRONMENT FOR CODE-CENTRIC REASONING IN BIOMEDICAL DATA SCIENCE
Abstract: We introduce MedAgentGym, a scalable and interactive training environment designed to enhance coding-based biomedical reasoning capabilities in large language model (LLM) agents. MedAgentGym comprises 72, 413 task instances across 129 categories derived from 12 authentic real-world biomedical scenarios. Tasks are encapsulated within executable sandbox environments, each featuring detailed task specifications, interactive feedback mechanisms, verifiable ground truth annotations, and scalable training trajectory generation. Extensive benchmarking of 29 LLMs reveals substantial performance disparities in biomedical data science between commercial and open-source LLMs. Leveraging efficient multi-threaded and multi-turn trajectory sampling in MedAgentGym, Med-Copilot achieves performance gains of +43.02% and +45.28% from offline and online reinforcement learning, respectively, demonstrating MedAgentGym as an effective training ground while establishing itself as a cost-effective, privacy-preserving alternative competitive with proprietary LLMs (gpt-4o). By offering a unified execution environment with a comprehensive benchmark and accessible, extensible training resources, MedAgentGym delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical data science.

Section: 
Figure 1: Overview of (a) task-specific and (b) overall leaderboard evaluation in MedAgentGym. The results show the (a) performance variations across biomedical data science tasks and (b) large gaps between proprietary and open-source (OSS) LLMs, highlighting the need for continued development of privacy-preserving, affordable LLM agents, especially for complex code-based biomedical reasoning tasks such as biomedical software engineering and predictive modeling.
this section cite: []

Section: INTRODUCTION
The exponential growth of healthcare data has fundamentally transformed modern biomedical research, intensifying the need for integration of advanced computational methods with medical domain expertise (Wornow et al., 2023b;Liu et al., 2025b). Biomedical researchers routinely face data science challenges that demand both medical data analysis knowledge and programming proficiency, such as querying large-scale databases, conducting statistical analyses, processing genomic sequences, and building predictive models from electronic health records (EHRs) (Nimmolrat et al., 2021;Lee et al., 2022;Wornow et al., 2023a). While recent advances in large language models (LLMs) have demonstrated significant capabilities in advanced reasoning (OpenAI, 2025b;Guo et al., 2025), including code generation (DeepMind, 2025) and scientific discovery (Swanson et al., 2024;Team et al., 2025;Yuan et al., 2025), it remains challenging to translate real-world biomedical data science requirements into executable computational solutions (Wang et al., 2024b;2025d).
Developing effective biomedical coding agents poses unique challenges beyond knowledge-intensive medical reasoning (Wang et al., 2025b;c) and general-purpose code generation (Zheng et al., 2024;Jing et al., 2025). Within biomedical research and clinical practice, direct deployment of proprietary LLMs remains infeasible due to strict privacy requirements and prohibitive operational costs (Meskó & Topol, 2023;Shi et al., 2024a), whereas OSS LLMs exhibit substantial deficiencies in biomedical coding capabilities (Figure 1). Mitigating this performance disparity calls for addressing two infrastructure gaps: (1) comprehensive, code-centric biomedical reasoning benchmarks to diagnose agent limitations and support rigorous, reproducible evaluation; and (2) specialized, interactive training environments to develop the complex reasoning and robust coding capabilities required for real-world biomedical data science.
In this study, we introduce MedAgentGym, a scalable and agentic training environment designed to systematically enhance the coding-centric reasoning capabilities of LLM agents for biomedical data science workflows. Grounded in diverse real-world biomedical scenarios, MedAgentGym provides:
• Comprehensive suite of code-centric biomedical reasoning tasks. MedAgentGym encompasses 72,413 biomedical coding-centric instances across 129 categories grounded in 12 real-world biomedical scenariosfoot_0 . We standardize a rich collection of biomedical data science tasks as executable problems with verifiable ground truth, spanning structured medical information retrieval, numerical clinical reasoning, bioinformatics, and machine learning (ML) modeling. Tasks incorporate diverse data modalities, including EHR tables, clinical notes, genomics, drugs, and biological sequences, which require medical domain-specific reasoning capabilities.
• Scalable and interactive training infrastructure. MedAgentGym provides an optimized, userfriendly environment to accelerate agent training. Each instance is encapsulated within executable, isolated, and reproducible Docker environments with pre-install dependencies, supporting multithreading, parallel execution, and sequential sampling. MedAgentGym ensures efficient trajectory collection and facilitate large-scale automated evaluation compatible with diverse agent scaffolds.
• Extensive benchmarking and effective agent training for biomedical data science. Through an extensive benchmark of 29 proprietary and open-source LLMs, we identify critical deficiencies in biomedical data analysis and predictive modeling. MedAgentGym effectively strengthens agentic training: Med-Copilot-7B achieves gains of +43.02% and +45.28% through offline and online reinforcement learning (RL), respectively, and performs comparably to gpt-4o on both inand out-of-distribution tasks. We publicly release MedAgentGym and Med-Copilot, together with high-quality training trajectories and the outcome verifier, to support reproducible benchmarking and continued development of LLM coding agents in biomedical data science.
this section cite: ['b53', 'b36', 'b6', 'b79', 'b85', 'b108', 'b113', 'b28', 'b47']

Section: RELATED WORKS
Coding-Centric Reasoning in Biomedical Data Science. Most existing medical benchmarks primarily evaluate LLMs on knowledge-intensive, narrative reasoning (Jin et al., 2019;Pal et al., 2022;Tsatsaronis et al., 2015). Although several efforts target isolated biomedical algorithmic tasks (Tang et al., 2024a;HAI@Stanford, 2025;Wang et al., 2024b) or simulate portions of clinical Table 1: Summary of related biomedical reasoning and coding datasets with task details and execution environments. MedAgentGym is among the first publicly available training environments for LLM agents in biomedical data science, uniquely integrating executable environments, interactive feedback, and task-isolated run-time facilities for coding-based reasoning. "DB", "DA", "Bioinfo", and "ML" denote "database", "data analytics", "bioinformatics", and "machine learning", respectively.
Domain Task Environment & Facility Scale (#Instances) Datasets (↓) QA Coding DB DA Bioinfo ML Execution Interaction Isolation Training # Train # Test # Traj.
MedMCQA (Pal et al., 2022) 3K 4.18K MedQA (Jin et al., 2021) 11.4K 1.27K PubMedQA (Jin et al., 2019) 450 500 BioASQ (Tsatsaronis et al., 2015) 745 140 MedAgentsBench (Tang et al., 2025) -862 MIRAGE (Xiong et al., 2024) -7.66K HealthBench (Arora et al., 2025) -5K EHRSQL (Lee et al., 2022) 15.5K 1.73K MedCalcBench (Khandekar et al., 2024) 10.1K 1.05K MedAgentBench (Jiang et al., 2025b) -300 BioCoder (Tang et al., 2024a) -1.24K BioDSBench (Wang et al., 2024b) -128 EHRSHOT (Wornow et al., 2023a) -15 MedAgentGym (Ours) 59.2K 13.2K 6.7K
workflows (Schmidgall et al., 2024;Li et al., 2024c;b), they do not capture a complete set of tasks in the full end-to-end lifecycle of biomedical data science, from data extraction (Lee et al., 2022;Ryu et al., 2024) to model development (Wornow et al., 2023a;Wang et al., 2020b). Complementing these benchmarks, MedAgentGym emphasizes computation-and coding-intensive tasks that require LLM agents to retrieve, transform, analyze, and compute biomedical data while generating and executing code with pre-installed biomedical libraries and dependencies to produce verifiable solutions.
Scalable and Interactive Training Environment for Biomedical Coding Agents. Agentic RL (Guo et al., 2025;Schulman et al., 2017;Shao et al., 2024b) shifts LLM post-training from passive sequence generation to autonomous agents operating in complex, dynamic settings, including medical reasoning (Xia et al., 2025;Jiang et al., 2025a;Chen et al., 2024;Lan et al., 2025;Wu et al., 2025a;Wang et al., 2025a). Within such a framework, agents interact iteratively with their environment, receiving observations and executing actions, while the environment returns reward signals and state updates (Wang et al., 2025e;Chezelles et al., 2024;Shao et al., 2024a;Nathani et al., 2025). However, most biomedical reasoning and data science benchmarks (Table 1) are single-pass evaluations without executable environments or agent-level interaction signals (Zhu et al., 2025;Arora et al., 2025;Wu et al., 2025b). In contrast, MedAgentGym uniquely provides an executable and interactive biomedical coding environment covering comprehensive range of tasks. It also supports efficient multi-turn trajectory sampling through multi-threaded rollouts, thus enabling scalable and systematic improvement via agentic fine-tuning beyond prompting (Shi et al., 2024b;Huang et al., 2025a).
this section cite: ['b25', 'b56', 'b87', 'b15', 'b56', 'b24', 'b25', 'b87', 'b83', 'b103', 'b0', 'b36', 'b31', 'b67', 'b36', 'b65', 'b69', 'b102', 'b2', 'b35', 'b4', 'b52', 'b115', 'b0']

Section: MEDAGENTGYM: A SCALABLE AND INTERACTIVE LLM AGENT TRAINING ENVIRONMENT FOR CODE-CENTRIC BIOMEDICAL REASONING

this section cite: []

Section: PROBLEM FORMULATION
We formulate coding-based reasoning as a structured problem-solving task: given a problem description x ∈ X , the goal is to generate a code snippet c ∈ C that produces an output y ∈ Y. Each instance (x, y) is paired with a ground truth output y * , and the correctness is verified using E : C ×Y → {0, 1}, where E = I(y = y * ). Existing biomedical reasoning datasets typically provide only question-answer pairs (x, y * ) without code solutions c or only include a single predefined code solution per task. To address this, MedAgentGym enables scalable generation and sampling of multiple coding trajectories c (0) , c (1) , • • • , c (k) with corresponding executions y (0) , y (1) , • • • , y (k) through parallel execution of LLM agents. Each trajectory is either single-turn or multi-turn, depending on task complexity and user requirements. Crucially, MedAgentGym captures both positive trajectories {c (i) |y (i) = y * } that succeed and negative trajectories {c (i) |y (i) ̸ = y * } including error messages as learning signals.
this section cite: []

Section: DATA CONSTRUCTION: FROM INDIVIDUAL DATASETS TO UNIFIED BENCHMARK
Task and Data Identification. MedAgentGym focuses on verifiable biomedical data science tasks that benefit from code-based solutions (i.e., code-centric biomedical reasoning). Clinically, we prioritize tasks originating from real-world healthcare settings and validated by a multidisciplinary panel of healthcare experts. For example, MedAgentGym involves MIMIC-III and eICU in EHRSQL (Lee et al., 2022) collected from 222 hospital staff members and annotated by human programmers.
Computationally, we integrate diverse coding tasks, ranging from structured medical information retrieval to open-ended biomedical research, ensuring comprehensive coverage and task diversity.
this section cite: ['b36']

Section: Verifiable Instances Preparation.
To standardize tasks across various sources, each instance in MedAgentGym is structured with: (1) a problem description, (2) verifiable ground-truth outputs, and
(3) optional data resources (e.g., EHRs). Additionally, standardized system and user prompts are designed to initiate the problem-solving process (see appendix G). MedAgentGym is highly flexible, easily accommodating new tasks that include clear descriptions and verifiable ground-truth outputs.
For coding-centric tasks that provide only reference code implementations (e.g., BioCoder (Tang et al., 2024a)), we validate task correctness based on the execution output of these reference solutions, generating definitive output signatures. This transformation is necessary because multiple valid code implementations may yield identical execution results, making the execution outcome-rather than the code itself-a more reliable and consistent verification signal. For tasks involving additional data resources (e.g., EHRSQL (Lee et al., 2022)), we include metadata on data access and sources. Detailed task overview and task-specific preparation are documented in appendix C.
this section cite: ['b36']

Section: Data Statistics.
MedAgentGym is a unified training environment built upon a large-scale, high-quality dataset comprising approximately 72,000 task instances across 129 categories from 12 real-world biomedical scenarios. Notably, with MedAgentGym, we collect large-scale agent trajectories to support coding agent development (section 5). To ensure reproducible and robust evaluation, we define clear train/test splits, separate internal and external validation sets, and perform n-gram (n = 10) string match to eliminate the data contamination issue. Table 2 provides statistics for MedAgentGym. To accommodate diverse research needs, we offer two versions of MedAgentGym: (1) a comprehensive, full-scale dataset for extensive exploration and detailed analysis, and (2) a balanced, lightweight subset for efficient leaderboard training and evaluation.
this section cite: []

Section: CODING ENVIRONMENT: FROM STATIC BENCHMARK TO INTERACTIVE INTERFACE

this section cite: []

Section: Isolated and Executable Sandbox Environment.
To ensure robust and reproducible codingbased biomedical reasoning, MedAgentGym provides isolated executable coding environments (i.e., sandbox) through Docker containers tailored to each task (Figure 2). These containers come preinstalled with all required dependencies, including specialized biomedical packages (e.g., AlignIO in BioCoder (Tang et al., 2024a)), facilitating reliable task execution. To address critical data safety concerns, each Docker environment guarantees: (1) environmental integrity, where isolation prevents contamination or data corruption potentially caused by LLM-generated code, preserving both the computational environment and the underlying data systems (Yang et al., 2024b); (2)
LLM Agent Agentic Traj. Sampling LLM-friendly commands MedAgentGym LLM-friendly environment feedbacks Request Info Data Info Evaluate Code Edit File Debug Code Code Results Error Msgs Interact History Sandbox Terminal Filesystem Features Labels Metadata Trajectories EHR Data Clinical Notes Genomics Data Drugs … Bio-Medical Metadata Bio-Medical Data Science Tasks … BioMed SWE Predictive Modeling Bio-Statistics BioMed Analysis… 129 Tasks 72K Instances 7.4M Data Unit-Test BioMed DS Question Request Info Data Info Coding Errors Debug Results Reward medical data security, where secure containerization enforces compliance with medical data usage policies, safeguarding sensitive patient information. Additionally, MedAgentGym supports extensive flexibility for integrating new tasks, where users can define customized Docker environments through configuration files. If certain packages are not initially available, a terminal tool allows LLM agents to dynamically install the required dependencies within their isolated environments. Interactive Feedback. MedAgentGym incorporates interactive feedback mechanisms, effectively bridging LLMs with coding interpreters: (1) robust parsing: To begin, the output generated by LLMs is formatted in structured JSON, facilitating straightforward parsing and code execution. In cases of execution errors, iterative JSON regeneration is employed to maximize successful code execution rates.
(2) debugging and error grounding: Compile-time and runtime error messages are systematically translated into a unified natural language format, making them more accessible to LLMs and significantly improving debugging efficiency and interpretability.
this section cite: []

Section: Efficient Trajectory Collection.
Each task in MedAgentGym is packaged in a reproducible Docker image with built-in support for multi-threading, parallel execution, and sequential sampling. Specifically, we integrate two widely used multi-threading backend engines, Rayfoot_2 and Joblibfoot_3 , to accelerate trajectory sampling. This infrastructure ensures efficient and scalable trajectory collection, supporting both extensive experimentation and systematic evaluation across multiple scenarios.
this section cite: []

Section: Plug-and-Play.
A key strength of MedAgentGym lies in its flexible and modular architecture, which readily supports the integration of new biomedical coding tasks. This inherent extensibility enables MedAgentGym to continually adapt to evolving advancements in biomedical sciences and artificial intelligence methodologies. Additionally, its trajectory sampling approach allows the straightforward transformation of traditional, non-executable biomedical reasoning tasks into coding-based scenarios with verifiable outputs, significantly broadening the scope and complexity of tasks that can be systematically evaluated. Moreover, users can define custom Docker environments through configuration files, and, if specific software packages are initially absent, a built-in terminal tool facilitates dynamic installation within each isolated execution environment, further improving MedAgentGym in runtime adaptability and user-friendliness.
this section cite: []

Section: EVALUATING LLMS AS MEDICAL CODING AGENTS WITH MEDAGENTGYM
4.1 EXPERIMENTS SETUP Agent Scaffolds. Following CodeAct (Wang et al., 2024a), we establish a default agent scaffold for systematically evaluating coding-based biomedical reasoning. Interactions within MedAgentGym are modeled as a Partially Observable Markov Decision Process (POMDP), focusing on sampled biomedical data science tasks p ∈ P. At each timestep t, the agent observes o t ∈ O and samples an action a t+1 ∈ A from the current policy π t based on interaction history. We define four primary action types: (a) request_info: retrieve relevant data from sources such as EHRs; (b) terminal: manage dependencies or local files within isolated Docker environments. (c) code_execution: execute code generated by LLMs through an integrated interpreter; and (d) debugging: translate code execution errors into natural language explanations enriched with detailed error information for LLM comprehension.  4) MedCalcBench (Khandekar et al., 2024), (5) MedAgentBench (Jiang et al., 2025b), (6) BioCoder (Tang et al., 2024a), (7) EHRSHOT (Wornow et al., 2023a), and (8) BioDSBench (Wang et al., 2024b). Moreover, we conduct experiments for out-of-distribution evaluation on 4,203 tasks from the following 4 datasets: (9) EHR-SeqSQL (Ryu et al., 2024), (10) EHRCon (Kwon et al., 2024), (11) MIMIC-Extract (Wang et al., 2020b), and (12) N-PowerAI (Ruan et al., 2025). Note that we do not consider knowledge-intensive medical question-answering tasks (Jin et al., 2019;Pal et al., 2022;Jin et al., 2021), as they are orthogonal to coding-aided reasoning. We include detailed task and dataset information in appendix C.
Baselines. We extensively benchmark the following state-of-the-art LLMs on MedAgentGym:
(i) API-based proprietary LLMs, including gpt-4o-mini (Hurst et al., 2024), gpt-4o (Hurst et al., 2024), gpt-4.1-mini (OpenAI, 2025a), gpt-4.1 (OpenAI, 2025a), gpt-o4-mini (OpenAI, 2025b), and codex-mini (Chen et al., 2021); (ii) OSS LLMs, including gemma-3 (Gemma, 2025), Qwen3 (Qwen, 2025a), Qwen2.5 (Yang et al., 2024a), Llama-3 (Dubey et al., 2024), Ministral (Ministral, 2025), and DeepSeek-R1 (Guo et al., 2025); (iii) coding LLMs, including codexmini (Chen et al., 2021), Qwen2.5-Coder-7B-Instruct and -14B-Instruct (Hui et al., 2024), and Seed-Coder-8B-Reasoning (Seed et al., 2025); and (iv) medical reasoning LLMs or medical domainspecific LLMs, including medgemma-4b-it (gemma-3-4b-pt) (Google, 2025), HuatuoGPT-o1-7B (Qwen2.5-7B-Instruct) and HuatuoGPT-o1-72B (Qwen2.5-72B) (Chen et al., 2024), m1-7B-23K (Qwen2.5-7B-Instruct) (Huang et al., 2025b), MedReason-8B (Llama-3.1-8B-Instruct) (Wu et al., 2025a), Baichuan-M1-14B-Instruct (Wang et al., 2025a), and Baichuan-M2-32B (Dou et al., 2025). Additional model details are available in appendix D.
Evaluation Metrics. We adopt success rate (SR) as the primary evaluation metric. For database, data science, and bioinformatics tasks with explicit ground truths, we compare LLM-generated code execution outputs with reference solutions using exact match. For open-ended ML tasks in clinical decision support, we measure performance using accuracy (Acc) across test cases. See appendix E for implementation details and F.1 for additional evaluation on code quality and efficiency. 4.2 RESULTS: BENCHMARKING LLMS AND REASONING MODELS WITH MEDAGENTGYM
Table 3 benchmarks the state-of-the-art LLMs on MedAgentGym. We summarize key observations from our zero-shot leaderboard evaluation as follows: ⋄ Significant Performance Gap Between Commercial API-based and OSS LLMs. This evident performance gap highlights the critical need for continued development of lightweight OSS LLMs that match commercial performance while addressing real-world privacy and cost constraints. ⋄ Task-Specific Performance Variations between Structured and Open-ended Medical Tasks. LLMs consistently perform better on structured tasks (e.g., database queries, medical calculations) compared to open-ended tasks requiring advanced coding and reasoning (e.g., data analysis, ML prediction). ⋄ Suboptimal Outcomes in Dedicated Coding and Medical Domain-Specific LLMs. Both coding and medical reasoning
LLMs deliver limited improvement or even decline over base models, revealing that coding-based biomedical reasoning represents a unique capability not adequately captured by specialization in either coding or medical reasoning. Surprisingly, medical reasoning models (regardless of model sizes) consistently underperform relative to their base models except for knowledge-intensive tasks (e.g., MedCalcBench, BioCoder), showing that fine-tuning in medical QA may reduce generalization and instruction-following ability. These findings highlight the need to jointly improve coding skills and medical reasoning, rather than treating them as separate objectives.
this section cite: ['b31', 'b65', 'b33', 'b64', 'b25', 'b56', 'b24', 'b20', 'b20', 'b3', 'b9', 'b49', 'b3', 'b19', 'b70', 'b2', 'b7']

Section: TRAINING LLM AGENTS FOR CODE-CENTRIC BIOMEDICAL REASONING
In this section, we leverage MedAgentGym to systematically enhance lightweight OSS LLMs as proficient coding agents (Med-Copilot) for biomedical reasoning. We first explore a two-stage agentic fine-tuning framework (section 5.1), followed by a detailed analysis of model scaling behaviors (section 5.2). We then introduce self-improvement to further boost agent performance (section 5.3) and conduct additional analysis on model generalization, ablation, and error patterns (section 5.4).
this section cite: []

Section: RL FINE-TUNING WITH TRAJECTORY SAMPLING
Training Setup. We select Qwen-2.5-Instruct-7B and -14B (Yang et al., 2024a) as our backbones.
To enable effective evaluation within MedAgentGym, we utilize a consistent CodeAct-style scaffold, allowing LLM agents to iteratively reason and refine biomedical code through interactive environment feedback. Detailed training setups, including hyperparameters, are provided in appendix E.
Trajectory Sampling. MedAgentGym facilitates efficient parallel trajectory sampling using ray and joblib backends. Specifically, we roll out (1) 2,137 successful trajectories using gpt-4.1-mini with a temperature of 0 to warm up the fine-tuning for smaller OSS models. Each successful trajectory contains 9.25 turns between the LLM and the code interpreter on average. In addition to 2,137 positive trajectories for supervised fine-tuning (SFT), we prepare additional trajectory pairs for RL such as direct preference optimization (DPO), including (2) 1,646 off-policy preference pairs sampled from gpt-4.1-mini, and (3) 2,939 on-policy preference pairs. For both types, we use the initial prompt interactions as shared context and contrast successful final codes against intermediate erroneous attempts. In addition, we also performed a quantitative analysis on 250+ trajectories (randomly sampled over 10% of our trajectory collection) and confirmed that the vast majority of successful solutions followed a logically sound path, with cases of 'correct answer from flawed code' being exceptionally rare (<1%). We release all 6K trajectories above to accelerate coding agent development. See appendix C.6 for detailed trajectories composition.
this section cite: []

Section: Two-Stage Fine-Tuning.
We benchmark two policy improvement methods: (1) SFT directly mimics high-reward trajectories consisting exclusively of successful outcomes, whereas (2) offline or online  RL optimizes the policy by favoring selected responses over rejected ones (Figure 3). We further consider a two-stage fine-tuning, initially warming up with SFT and subsequently refining with RL.
Results: Offline RL (DPO). Table 4 compares several post-training methods, revealing that simple SFT over successful trajectories significantly boosts performance on structured coding tasks, demonstrating its effectiveness in capturing structured coding patterns. Besides, DPO is particularly beneficial for optimizing open-ended task performance.
Results: Online RL (PPO and GRPO). We further consider online RL methods, including Proximal Policy Optimization (PPO) (Schulman et al., 2017) and Group Relative Policy Optimization (GRPO) (Shao et al., 2024b), to enable Med-Copilot to actively explore tasks and dynamically generate higher-quality training data through interaction. The evaluation module of Med-Copilot is employed to provide two reward signals: a correctness reward and a format reward, the latter indicating whether the generated output contains code blocks. As shown in Table 4, GRPO achieve markedly stronger performance, suggesting enhanced generalization capabilities in diverse biomedical scenarios compared with offline RL.
this section cite: ['b69']

Section: SCALING LLM AGENT IMPROVEMENTS WITH MEDAGENTGYM
Verifier Training Setup. In addition to directly training coding agents, MedAgentGym facilitates the development of an outcome-supervised reward model (ORM) to evaluate generated solutions effectively. Inspired by prior work (Cobbe et al., 2021;Pan et al., 2025), we formalize the verifier task as predicting the probability that a given trajectory successfully solves a coding task. Formally, we represent a trajectory as an interleaved sequence Verifier Training Data. We construct the verifier training dataset by combining two sets of trajectories originally sampled for agent training: (1) off-policy trajectories, consisting of 2,742 samples from gpt-4.1-mini; and (2) on-policy trajectories, comprising 2,939 samples generated by the agent. Combining both on-and off-policy trajectories, we ensure a balanced dataset of successful and unsuccessful trajectories, filtering to fit within a maximum context length of 32k tokens.
τ = [o 1 , a 1 , o 2 , a 2 , • • • , o n , a n ],
Results: Inference and Training-Time Scaling. We introduce two additional evaluation metrics:
(1) Pass@K: the fraction of tasks solved by at least one trajectory from K sampled attempts; and
(2) Best@K: the fraction of accurately selects successful trajectories that actually solves the task from a set of candidate generations. Figure 4 (left) illustrates the performance scaling with increasing trajectory sampling. Pass@K significantly improves from 17.0% at K = 1 to 45.0% at 16, while Database DA BioInfo+ML Overall 0 20 40 60 Avg. Score (%) gpt-4.1-mini gpt-4.1-mini w/o debug Figure 6: Effect of Debug
15.22% 50.39% 1.47% 30.39% 2.53% IO error Stuck in the Loop Compile error Runtime error Others Best@K shows steady advancement from 17.0% to 41.7%. The relatively small gap between metrics indicates that our trained verifier effectively identifies successful trajectories, unleashing its potential as a reward model for integration into advanced online RL frameworks. Figure 4 (right) examines agent performance as a function of increased training data volumes in SFT. We observe consistent performance improvements with greater training data availability, suggesting additional computational resources dedicated to sampling further trajectories are likely to yield continued performance gains.
this section cite: ['b5', 'b57']

Section: MODEL PERFORMANCE SCALING WITH SELF-IMPROVEMENT
Self-Improvement Training Setup. Beyond expert-generated trajectories from gpt-4.1-mini, we also explore self-improvement by refining the model on its own outputs (Qwen2.5-7B-Instruct). We first apply rejection-sampling SFT: starting from Qwen2.5-7B-Instruct, we collect 1,000 successful trajectories and perform filtered behavior cloning on this set. We subsequently apply DPO (section 5.1) using on-policy preference pairs generated by the rejection sampling SFT checkpoint. Specifically, we sample eight rollouts per task, score them via the verifier in section 5.2, and form 4,298 pairs by contrasting the highest-scoring correct and lowest-scoring incorrect trajectories. Following Pang et al. ( 2024), we repeat this data collection and policy update cycle (iDPO) for further refinement, resampling trajectories and reconstructing preference pairs after each DPO update. In contrast to standard DPO, which performs a single offline preference-based update on a fixed dataset, iDPO alternates between on-policy data collection and DPO optimization so that the policy and training data co-evolve (i.e., self-improvement).
Results: Rejection Sampling (RS) and iDPO. Figure 5 illustrates consistent performance gains across one SFT stage and two subsequent DPO stages. However, we observe diminishing returns over successive iterations. Initially, rejection sampling SFT significantly boosts performance by effectively capturing successful coding patterns. Subsequent DPO stages show smaller incremental improvements, reflecting the model's diminishing exploration space as it tackles increasingly challenging tasks, ultimately converging toward an approximate Nash equilibrium. Results: External Evaluation. Table 5 summarizes external evaluation results on MedAgentGym.
this section cite: []

Section: GENERALIZATION, ABLATION, AND ERROR ANALYSIS
The external suites were intentionally chosen to stress different code-centric skills (e.g., sequential SQL, hybrid text-table consistency, raw EHR time series, biostatistical power analysis), which induces markedly different agent trajectories (Figure 10(b)) and naturally yield lower absolute scores across all models, including proprietary baselines. Med-Copilot with SFT and DPO modestly improve performance on openended, reasoning-intensive tasks (e.g., MIMIC-Extract). However, improvements remain limited, in-dicating challenges in generalizing across specialized biomedical contexts. In particular, incorporating online RL optimization techniques, especially GRPO (Shao et al., 2024b), can effectively improve performance on unseen, out-of-distribution tasks. Specifically, Med-Copilot-14B (GRPO) achieves 47.02% on the external suite, a +19.10% gain over its backbone (27.92%). This significant gain on unseen distributions verifies that MedAgentGym instills transferable biomedical coding proficiency rather than memorizing training trajectories.
Effect of Interactive Coding. Figure 6 shows that removing debugging capabilities significantly decreases model performance across all tasks. Interactive coding mechanism in MedAgentGym substantially contributes to successful coding-based medical reasoning by enabling the model to effectively interpret and rectify execution errors.
Error Analysis. Figure 7 summarizes common error types encountered by the strongest evaluated LLM, gpt-4.1. Loop-related issues dominate, accounting for 50.39% of errors, where agents repeatedly execute the same action in the final turns, indicating difficulty in adapting or exploring alternative strategies. This highlights the need to promote effective exploration and enhance robustness in solving complex biomedical reasoning tasks. Additional experimental results, including cost analysis, case studies, and human studies, are available in appendix F.
this section cite: []

Section: CONCLUSION
We present MedAgentGym, an executable, privacy-preserving, and extensible training environment for scaling code-based biomedical reasoning in LLM agents. With 72K task instances across 129 categories, MedAgentGym enables comprehensive benchmarking of 29 proprietary and OSS LLMs for biomedical data science within a modular, decoupled architecture that supports flexibility and extensibility. Med-Copilot further demonstrates that systematic training and trajectory sampling with MedAgentGym improve coding proficiency for biomedical data science tasks. MedAgentGym has the potential to accelerate progress from structured medical information retrieval tasks toward more open-ended computational research questions in clinical research and biomedical discovery.
this section cite: []

Section: References
Ref_id:b0 Title: Healthbench: Evaluating large language models towards improved human health Year: (2025)
Ref_id:b1 Title: Mle-bench: Evaluating machine learning agents on machine learning engineering Year: (2024)
Ref_id:b2 Title: Huatuogpt-o1, towards medical complex reasoning with llms Year: (2024)
Ref_id:b3 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b4 Title: Rim Assouel, et al. The browsergym ecosystem for web agent research Year: (2024)
Ref_id:b5 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b6 Title: Alphaevolve: A coding agent for scientific and algorithmic discovery Year: (2025)
Ref_id:b7 Title: Baichuan-m2: Scaling medical capability with large verifier system Year: (2025)
Ref_id:b8 Title: How capable are web agents at solving common knowledge work tasks? Year: (2024)
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: Txagent: An ai agent for therapeutic reasoning across a universe of tools Year: (2025)
Ref_id:b11 Title: Gemma 3 technical report Year: (2025)
Ref_id:b12 Title: Medgemma hugging face Year: (2025)
Ref_id:b13 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b14 Title: DS-agent: Automated data science by empowering large language models with case-based reasoning Year: (2024)
Ref_id:b15 Title: Holistic evaluation of large language models for medical applications. Blog Post Year: (2025)
Ref_id:b16 Title: Biomni: A general-purpose biomedical ai agent Year: ()
Ref_id:b17 Title: Mlagentbench: Evaluating language agents on machine learning experimentation Year: (2023)
Ref_id:b18 Title: Xianfeng Tang, and Yuyin Zhou. m1: Unleash the potential of test-time scaling for medical reasoning with large language models Year: (2025)
Ref_id:b19 Title: Qwen2. 5-coder technical report Year: (2024)
Ref_id:b20 Title: Gpt-4o system card Year: (2024)
Ref_id:b21 Title: Meds 3 : Towards medical small language models with self-evolved slow thinking Year: (2025)
Ref_id:b22 Title: Medagentbench: Dataset for benchmarking llms as agents in medical applications Year: (2025)
Ref_id:b23 Title: SWE-bench: Can language models resolve real-world github issues? Year: (2024)
Ref_id:b24 Title: What disease does this patient have? a large-scale open domain question answering dataset from medical exams Year: (2021)
Ref_id:b25 Title: PubMedQA: A dataset for biomedical research question answering Year: (2019-11)
Ref_id:b26 Title: Agentmd: Empowering language agents for risk prediction with large-scale clinical tool learning Year: (2024)
Ref_id:b27 Title: Self-evolving llm agent for biomedical research Year: (2025)
Ref_id:b28 Title: DSBench: How far are data science agents from becoming data science experts? Year: (2025)
Ref_id:b29 Title:  Year: (2016)
Ref_id:b30 Title: Researcharena: Benchmarking llms' ability to collect and organize information as research agents Year: (2024)
Ref_id:b31 Title: Medcalc-bench: Evaluating large language models for medical calculations Year: (2024)
Ref_id:b32 Title: MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making Year: (2024)
Ref_id:b33 Title: EHRCon: Dataset for checking consistency between unstructured notes and structured tables in electronic health records Year: (2024)
Ref_id:b34 Title: Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models Year: (2025)
Ref_id:b35 Title: Clinicalgpt-r1: Pushing reasoning capability of generalist disease diagnosis with large language model Year: (2025)
Ref_id:b36 Title: Ehrsql: A practical text-to-sql benchmark for electronic health records Year: (2022)
Ref_id:b37 Title: Learning to use medical tools with multi-modal agent Year: (2024)
Ref_id:b38 Title: Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents Year: (2024)
Ref_id:b39 Title: Aligning llms to ask good questions a case study in clinical reasoning Year: (2025)
Ref_id:b40 Title: Mediq: Question-asking llms and a benchmark for reliable interactive clinical reasoning Year: (2024)
Ref_id:b41 Title: Towards reflection-aware tool-augmented clinical agents Year: (2024)
Ref_id:b42 Title: Can large language models reason about medical questions? Year: ()
Ref_id:b43 Title: Beyond distillation: Pushing the limits of medical llm reasoning with minimalist rule-based rl Year: (2025)
Ref_id:b44 Title: Towards artificial intelligence research assistant for expert-involved learning Year: (2025)
Ref_id:b45 Title: Evaluating llms as agents Year: (2023)
Ref_id:b46 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b47 Title: The imperative for regulatory oversight of large language models (or generative ai) in healthcare Year: (2023)
Ref_id:b48 Title: Paper2agent: Reimagining research papers as interactive and reliable ai agents Year: (2025)
Ref_id:b49 Title: Un ministral, des ministraux Year: (2025)
Ref_id:b50 Title: Bixbench: a comprehensive benchmark for llm-based agents in computational biology Year: (2025)
Ref_id:b51 Title: Foundation models for generalist medical artificial intelligence Year: (2023)
Ref_id:b52 Title: A new framework and benchmark for advancing ai research agents Year: (2025)
Ref_id:b53 Title: Patient triage system for supporting the operation of dispatch centres and rescue teams Year: (2021)
Ref_id:b54 Title: Introducing gpt-4.1 in the api Year: ()
Ref_id:b55 Title: Openai o3 and o4-mini system card Year: ()
Ref_id:b56 Title: Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering Year: (2022)
Ref_id:b57 Title: Training software engineering agents and verifiers with SWE-gym Year: (2025)
Ref_id:b58 Title: Iterative reasoning preference optimization Year: (2024)
Ref_id:b59 Title: The eicu collaborative research database, a freely available multi-center database for critical care research Year: (2018)
Ref_id:b60 Title: Smart: Self-aware agent for tool overuse mitigation Year: (2025)
Ref_id:b61 Title: Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution Year: (2025)
Ref_id:b62 Title: Qwen3: Think deeper, act faster Year: ()
Ref_id:b63 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b64 Title: N-power ai: A specialized agent framework for automated sample size and power analysis in clinical trial design Year: (2025)
Ref_id:b65 Title: Ehr-seqsql: A sequential text-to-sql dataset for interactively exploring electronic health records Year: (2024)
Ref_id:b66 Title: Towards collaborative autonomous research Year: (2025)
Ref_id:b67 Title: Agentclinic: a multimodal agent benchmark to evaluate ai in simulated clinical environments Year: (2024)
Ref_id:b68 Title: Agent laboratory: Using llm agents as research assistants Year: (2025)
Ref_id:b69 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b70 Title: Let the code model curate data for itself Year: (2025)
Ref_id:b71 Title: Collaborative gym: A framework for enabling and evaluating human-agent collaboration Year: (2024)
Ref_id:b72 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b73 Title: Hybridflow: A flexible and efficient rlhf framework Year: (2025)
Ref_id:b74 Title: Medadapter: Efficient test-time adaptation of large language models towards medical reasoning Year: (2024)
Ref_id:b75 Title: EHRAgent: Code empowers large language models for few-shot complex tabular reasoning on electronic health records Year: (2024)
Ref_id:b76 Title: Large language models encode clinical knowledge Year: (2023)
Ref_id:b77 Title: Large language models are poor medical coders-benchmarking of medical code querying Year: (2024)
Ref_id:b78 Title: Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards Year: (2025)
Ref_id:b79 Title: The virtual lab: Ai agents design new sars-cov-2 nanobodies with experimental validation Year: (2024)
Ref_id:b80 Title: Ml-bench: Evaluating large language models and agents for machine learning tasks on repository-level code Year: (2023)
Ref_id:b81 Title: Biocoder: a benchmark for bioinformatics code generation with large language models Year: (2024)
Ref_id:b82 Title: Medagents: Large language models as collaborators for zero-shot medical reasoning Year: (2024)
Ref_id:b83 Title: Medagentsbench: Benchmarking thinking models and agent frameworks for complex medical reasoning Year: (2025)
Ref_id:b84 Title: Large language models streamline automated machine learning for clinical studies Year: (2024)
Ref_id:b85 Title: When agent becomes the scientistbuilding closed-loop system from hypothesis to verification Year: (2025)
Ref_id:b86 Title: Disentangling reasoning and knowledge in medical large language models Year: (2025)
Ref_id:b87 Title: An overview of the bioasq large-scale biomedical semantic indexing and question answering competition Year: (2015)
Ref_id:b88 Title: Baichuan-m1: Pushing the medical capability of large language models Year: (2025)
Ref_id:b89 Title: Text-to-sql generation for question answering on electronic medical records Year: (2020)
Ref_id:b90 Title: Mimic-extract: a data extraction, preprocessing, and representation pipeline for mimic-iii Year: (2020)
Ref_id:b91 Title: Medical reasoning in the era of llms: A systematic review of enhancement techniques and applications Year: (2025)
Ref_id:b92 Title: A survey of llm-based agents in medicine: How far are we from baymax? arXiv preprint Year: (2025)
Ref_id:b93 Title: Executable code actions elicit better llm agents Year: (2024)
Ref_id:b94 Title: Can large language models replace data scientists in clinical research Year: (2024)
Ref_id:b95 Title: Biodsa-1k: Benchmarking data science agents for biomedical research Year: (2025)
Ref_id:b96 Title: Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning Year: (2025)
Ref_id:b97 Title: Ehrshot: An ehr benchmark for few-shot evaluation of foundation models Year: (2023)
Ref_id:b98 Title: The shaky foundations of clinical foundation models: A survey of large language models and foundation models for emrs Year: (2023)
Ref_id:b99 Title: Eliciting factual medical reasoning steps in llms via knowledge graphs Year: (2025)
Ref_id:b100 Title: Evaluating and learning diagnostic reasoning from clinical case reports Year: (2025)
Ref_id:b101 Title: Evolving large language model-based agents across diverse environments Year: (2024)
Ref_id:b102 Title: Mmedagent-rl: Optimizing multi-agent collaboration for multimodal medical reasoning Year: (2025)
Ref_id:b103 Title: Benchmarking retrieval-augmented generation for medicine Year: (2024)
Ref_id:b104 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b105 Title: Swe-agent: Agent-computer interfaces enable automated software engineering Year: (2024)
Ref_id:b106 Title: Swe-smith: Scaling data for software engineering agents Year: (2025)
Ref_id:b107 Title: Medreseacher-r1: Expert-level medical deep researcher via a knowledge-informed trajectory synthesis framework Year: (2025)
Ref_id:b108 Title: Dolphin: Closed-loop open-ended auto-research through thinking, practice, and feedback Year: (2025)
Ref_id:b109 Title: Datascibench: An llm agent benchmark for data science Year: (2025)
Ref_id:b110 Title: Med-rlvr: Emerging medical reasoning from a 3b base model via reinforcement learning Year: (2025)
Ref_id:b111 Title: Benchmarking data science agents Year: (2024-08)
Ref_id:b112 Title: Sirius: Self-improving multi-agent systems via bootstrapped reasoning Year: (2025)
Ref_id:b113 Title: Integrating code generation with execution and refinement Year: (2024)
Ref_id:b114 Title: Medgr: Breaking the data barrier for medical reasoning via generative reward learning Year: (2025)
Ref_id:b115 Title: Medagentboard: Benchmarking multi-agent collaboration with conventional methods for diverse medical tasks Year: (2025)
Ref_id:b116 Title: Benchmarking expert-level medical reasoning and understanding Year: (2025)
