Title: BIRD-INTERACT: RE-IMAGINING TEXT-TO-SQL EVAL-UATION VIA LENS OF DYNAMIC INTERACTIONS
Abstract: Large language models (LLMs) have demonstrated remarkable performance on single-turn text-to-SQL tasks, but real-world database applications predominantly require multi-turn interactions to handle ambiguous queries, execution errors, and evolving user requirements. Existing multi-turn benchmarks fall short of capturing this complexity, either by treating conversation histories as static context or by limiting evaluation to narrow, read-only (SELECT-ONLY) operations, thereby potentially failing to reflect the challenges encountered in production-grade database assistant. In this work, we introduce BIRD-INTERACT, a benchmark that restores this missing realism through: (1) a comprehensive interaction environment that couples each database with a hierarchical knowledge base, metadata files, and a function-driven user simulator, enabling models to solicit clarifications, retrieve knowledge, and recover from execution errors without human supervision; (2) two evaluation settings reflecting real-world interaction settings which contain a predefined conversational protocol (c-Interact) and a more open-ended agentic setting (a-Interact) in which the model autonomously decides when to query the user simulator or explore the DB environment; (3) a challenging task suite that covers the full CRUD spectrum for both business-intelligence and operational use cases, guarded by executable test cases. Each task features ambiguous and follow-up sub-tasks, requiring LLMs to engage in dynamic interaction. The suite is organized into two sets: a full set (BIRD-INTERACT-FULL) of 600 tasks which unfold up to 11,796 dynamic interactions for a comprehensive overview of performance and a lite set (BIRD-INTERACT-LITE) of 300 tasks, with simplified databases for detailed behavioral analysis of interactions, and fast development of methods. Our empirical results highlight the difficulty of BIRD-INTERACT: the most recent flagship model GPT-5 completes only 8.67% of tasks in the c-Interact setting and 17.00% in the a-Interact setting on the full task suite. Further analysis via memory grafting and Interaction Test-time Scaling (ITS) validates the importance of effective interaction for achieving success in dynamic text-to-SQL tasks.

Section: INTRODUCTION
Data-driven decision-making has become indispensable across modern enterprises, prompting a surge of interest in Natural Language Interfaces to Databases (NLIDB) that empower non-technical users to extract insights from relational databases using natural language (Shi et al., 2024). Motivated by this vision, a wave of methods (Pourreza et al., 2025a;b;Pourreza & Rafiei, 2023;Liu et al., 2025;Qu et al., 2024;Li et al., 2025b;Maamari et al., 2024;Sheng & Shuai, 2025;Li et al., 2025a; Talaei Reading the DB schema, and column meanings.
Searching "AVS" from KB, but find no definition.  et al., 2024;Caferoglu & Ulusoy, 2024;Cao et al., 2024;Lee et al., 2025) based on large language models (LLMs) has recently achieved impressive text-to-SQL performance on popular single-turn benchmarks such as Spider (Yu et al., 2018) and BIRD (Li et al., 2023b).
However, real-world data interaction is rarely a single, perfectly-formed query (Li et al., 2025c;Dinan et al., 2019). It is an iterative, stateful dialogue characterized by ambiguity (Chen et al., 2025b) and evolving goals (Wu et al., 2025). The task in Figure 1 exemplifies this complexity. To succeed, the text-to-SQL system must first engage the user to resolve the ambiguity of the term urgent care. Only with this clarified context can it generate the correct SQL. If its initial code fails an execution test, LLM must debug and revise its SQL solution based on the error feedback. After the user confirms the SQL is correct, they may proceed with a follow-up question that depends on its intermediate results. Therefore, evaluation on true practical utility LLMs with these multi-faceted aspects requires a benchmark containing a complete interactive problem-solving process, rather than isolated, single-turn SQL generation.
Although existing interactive text-to-SQL datasets (Yu et al., 2019b;a;Chen et al., 2025b;Guo et al., 2021;Dahl et al.) have been developed, they inadequately model this reality for two primary reasons. First, most multi-turn text-to-SQL benchmarks rely on static conversation transcripts (Yu et al., 2019a;Chen et al., 2025b;Yu et al., 2019b;Guo et al., 2021). They present models with a clean interaction history without recording the failed attempts, digressions, and clarifications that occur in practice. This design may introduce a fundamental limitation: every LLM is evaluated against the same predetermined dialogue trajectory, regardless of how it would have naturally guided the interaction. This setup fails to reward intelligent interaction strategies and cannot effectively penalize conversational mess up. Second, existing benchmarks suffer from a narrow task scope, focusing on read-only (SELECT-only) queries typical of business intelligence (BI) reporting. This ignores a vast and critical range of database management (DM) operations, including data manipulation (INSERT, UPDATE, DELETE), schema modifications (ALTER TABLE), and transactional control, which are also common operations in the normal DBA cycle (Chen et al., 2024).
To address these critical limitations, we introduce BIRD-INTERACT, a new benchmark designed to evaluate LLMs in a dynamic text-to-SQL environment. Our work makes the following contributions:
(1) A High-Fidelity Interactive Environment: We develop a comprehensive sandbox upon an open-source project LIVESQLBENCH (BIRD-Team, 2025) for each task, including a hierarchical knowledge base (HKB) with domain-specific facts, metadata files, an executable database environment, and most critically, an interactive user simulator as recent research (Wu et al., 2025;Yao et al., 2025;Wang et al., 2024). This simulator can respond to clarification questions, provide feedback on proposed actions, and guide the model through complex tasks, enabling end-to-end evaluation without human intervention. However, recognizing that traditional simulators, even those powered by advanced models like GPT-4o, exhibit unfair behaviors such as ground-truth leakage, we propose a novel two-stage function-driven approach that maps model questions to constrained symbolic actions before generating controlled simulator responses.
(2) Two Evaluation Settings: We propose two popular evaluation settings. c-Interact (protocol-guided) presents tasks with a clear conversational protocol, testing a model's ability to follow a structured conversation with the user. In contrast, a-Interact (agentic) provides only a high-level goal, requiring the model to autonomously plan a strategy, decide when to query the database, consult documentation, or ask the user simulator for help.
(3) A Comprehensive and Challenging Task Suite: BIRD-INTERACT expands the scope of evaluation to include the full spectrum of CRUD operations. Tasks are drawn from both analytical and operational domains and are accompanied by executable test cases that verify functional correctness. Each task features an ambiguous initial priority sub-task, dynamic clarification requirements, follow-up sub-tasks, and environmental uncertainties, which can only be resolved through dynamic interaction. The suite consists of two parts: a full set (BIRD-INTERACT-FULL) of 600 tasks, unfolding up to 11,796 dynamic interactions for a comprehensive evaluation of performance, and a lite set (BIRD-INTERACT-LITE) of 300 tasks with cleaner databases, enabling finer-grained behavioral analysis and faster deployment.
Our experiments show that state-of-the-art models struggle with BIRD-INTERACT, with GPT-5 achieving only 8.67% success in c-Interact and 17% in a-Interact. We identify distinct challenges across interaction modes: communication effectiveness often determines success in c-Interact, while a-Interact suffers from bias toward costly trial-and-error over strategic resource exploration. We also observe Interaction Test-time Scaling (ITS), where performance improves monotonically with additional interaction opportunities across multiple models. These findings support our hypothesis that developing strategic interaction capabilities is key to improving LLM performance on complex database reasoning.
this section cite: ['b57', 'b10', 'b44', 'b52', 'b35', 'b45', 'b4', 'b5', 'b31', 'b74', 'b11', 'b35', 'b66', 'b76', 'b35', 'b21', 'b9', 'b75', 'b35', 'b76', 'b21', 'b8', 'b66', 'b72', 'b64']

Section: PROBLEM DEFINITION
Task Definition. We formalize interactive text-to-SQL as a multi-turn collaboration between a text-to-SQL system S θ and user simulator U γ operating over database environment E = {D, M, K}, where D is the executable database, M contains schema metadata, and K represents external knowledge (Lee et al., 2021;Dou et al., 2022;Li et al., 2023b). Given a sequence of related sub-tasks Q = {q 1 , q 2 , . . . , q n }, the goal is for S to generate SQL solutions {σ 1 , . . . , σ n } through interactions. For each sub-task q i , the interaction proceeds through interaction turn t = 1, 2, . . . until completion:
u t i = U γ (h t-1 i , q i , E), s t i = S θ (h t-1 i , u t i , E), h t i = h t-1 i ⊕ ⟨u t i , s t i ⟩(1)
where h t i represents the interaction history up to turn t and ⊕ denotes text concatenation in prompt. The user simulator U γ manages the interaction by presenting sub-tasks, answering clarification questions for ambiguous queries, and providing feedback on submitted SQL. Critically, subsequent sub-tasks are released only after successful completion of first sub-tasks.
Metrics. Each sub-task q i is annotated with ground-truth SQL σ * i and executable test cases T i that define correctness. A predicted solution σ i is correct if it passes all associated test cases, ensuring functional equivalence with σ * i . In our implementation, each task consists of two related sub-tasks (n = 2): an initial priority sub-task q 1 containing ambiguities requiring resolution, and (2) a subsequent follow-up sub-task q 2 . We evaluate system performance using: (1) Success Rate (SR): The proportion of sub-tasks completed successfully, with each sub-task scored 0 or 1. We report SR separately for sub-task 1 and sub-task 2 as an online evaluation during interaction. (2) Normalized Reward: Defined as normalized scoring according to priority weighting as designed in Appendix F to [0,1] for analyzing system behaviors after interaction (offline evaluation) (Yao et al., 2022).
this section cite: ['b14', 'b81', 'b70']

Section: BENCHMARK CONSTRUCTION
This section details the methodology for the construction of BIRD-INTERACT benchmark. We begin by outlining the overall benchmark setup (Section 3.1), and then elaborate on how we convert clear single-turn tasks into ones requiring interactions (Section 3.2).
this section cite: []

Section: SETUP AND RESOURCES
We build our benchmark on the text-to-SQL tasks and infrastructure of LIVESQLBENCH (BIRD-Team, 2025). We select this foundation due to several key advantages. First, LIVESQLBENCH provides a comprehensive evaluation environment. It supports the full spectrum of SQL operations, including DML and DDL, which allows for dynamic database states that reflect real-world usage. Furthermore, its permissive license and ready-to-use artifacts, including an executable database sandbox and metadata files, facilitate extension and reproducibility. Third, it features a Hierarchical Knowledge Base (HKB) that organizes external knowledge as nodes in a directed acyclic graph (DAG), as shown in Figure 1, where "AVS" depends on "IF" and "CPI". This structure explicitly models dependencies between facts that require multi-hop reasoning to connect isolated information. Despite these strengths, LIVESQLBENCH is fundamentally a single-turn benchmark. This design fails to capture the interactive and often ambiguous nature of real-world data analysis scenarios. Our primary contribution is to convert this static benchmark into a dynamic, interactive setting.
this section cite: ['b3']

Section: INTERACTIVE TASK ANNOTATION
To maintain the integrity and quality of our benchmark, we recruit 12 expert annotators through a rigorous multi-stage selection process detailed in Appendix C. We describe systematically the conversion from single-turn tasks of LIVESQLBENCH into multi-turn interactive scenarios through two key annotation strategies: ambiguity injection and follow-up sub-task generation: Ambiguity Injection. Ambiguities in daily life require interactions to seek clarification. To make annotation and evaluation controllable, we design methods to inject ambiguities into single-turn queries and the environment from LIVESQLBENCH, pairing each with a unique clarification.
(1) Superficial user query ambiguities: we target surface-level ambiguity in the user request. These include intent-level ambiguities, where the user language is vague (e.g., "elderly people"), and implementation-level ambiguities, where the user's intent is clear but the implementation details (e.g., decimal precision) are under-specific.
"
(2) Knowledge ambiguities: we inject incompleteness into the external knowledge. This category includes two subtypes: (i) one-shot knowledge ambiguity, where isolated knowledge entries are removed. (ii) knowledge chain breaking, where intermediate nodes in multi-hop knowledge chains are masked. For example, consider the chain "urgent care" → "AVS" → "IF/CPI" in Figure 2. By masking the intermediate node, i.e., the fact "AVS" in HKB, we deliberately break the inferential chain, rendering knowledge ambiguous and requiring user clarification to proceed. (3) Environmental ambiguities: LIVESQLBENCH databases already contain natural noise, such as NULL in critical fields, which further introduces uncertainty in how these cases should be handled.
Each injected ambiguity is paired with a corresponding SQL snippet from the ground-truth query as a clarification source, which guides our user simulator in generating consistent and contextually appropriate clarifications. Quality control ensures that ambiguous queries are unsolvable without clarification yet fully reconstructable once clarifications are provided. Complete details are given in Appendix H.
this section cite: []

Section: Follow-Up Sub-tasks Annotation.
User intents frequently evolve throughout an interactive session (Taylor, 2015), with users modifying, filtering conditions, or exploring related aspects of their queries. Therefore, we also extend each initial priority sub-task with one additional follow-up sub-task to resonate with this scenario.
These follow-up sub-tasks are designed carefully using a principled 5-category taxonomy detailed in Appendix H.5. A key contribution of our benchmark is the introduction of state dependency between sub-tasks, different with other datasets (Yu et al., 2019a;b;Lee et al., 2021;Zhong et al., 2017;Li et al., 2025d). System models must reason over modified database states or the newly created objects (e.g. tables) from preceding queries to write SQLs for follow-up sub-tasks.
this section cite: ['b75', 'b10', 'b79']

Section: FUNCTION-DRIVEN USER SIMULATOR
Evaluating interactive text-to-SQL systems requires user interactions, such as multi-turn requests and responses to clarification questions. Conducting such human-in-the-loop evaluations at scale is impractical. To make large-scale evaluations feasible, recent interactive benchmarks, such as MINT (Wang et al., 2024), employ LLMs to simulate human users (Li et al., 2025c;Yu et al., 2019a;b). However, we observe that there are two major issues among these simulators: (1) they sometimes leak information from ground-truth SQL query, and (2) they may deviate from the original task requirements (Barres et al., 2025;Kazi et al., 2024).
this section cite: ['b64', 'b75', 'b10', 'b1']

Section: Two-Stage Strategy.
To ensure a more robust evaluation, we introduce a two-stage function-driven user simulator, as illustrated in Figure 3(c). In the first stage, an LLM functions as a semantic parser. It maps the system's clarification request into one of three predefined allowed actions: AMB(), LOC(), or UNA(). AMB() is invoked for queries related to ambiguities that have been pre-annotated with the key SQL snippet. LOC() handles reasonable clarification requests that fall outside our pre-annotated ambiguities, such as questions about SQL formatting or specific sub-components. In these cases, the simulator uses an AST-based retrieval step to locate the relevant SQL fragment (detailed in Appendix N). Finally, UNA() rejects any inappropriate requests, such as attempts to elicit ground-truth answers. In the second stage, the user simulator generates a final response based on the chosen action and the annotated GT SQL with clarification source. This two-stage approach ensures the simulator's behavior remains predictable and controllable, while still permitting diverse and context-aware interactions. Detailed prompts are provided in Appendix R.
this section cite: []

Section: DATA STATISTICS
Table 1 reports key properties of BIRD-INTERACT. The resulting benchmark comprises a total of 900 interactive text-to-SQL tasks, each featuring an ambiguous initial priority sub-task, dynamic clarification requirements, follow-up sub-tasks, and environmental uncertainties, collectively spanning the full CRUD spectrum (Create, Read, Update, and Delete). In Appendix E, we also conduct a comprehensive comparison against other relevant benchmarks, showing that BIRD-INTERACT is among the most open, challenging, and long-horizon interactive benchmarks in text-to-SQL scenarios.
this section cite: []

Section: EVALUATION SETTINGS

this section cite: []

Section: Two Evaluation Settings.
The interactive framework of BIRD-INTERACT supports evaluation in two scenarios: LLMs as conversational assistants (c-Interact) (Dinan et al., 2019) and as agents (a-Interact) (Schluntz & Zhang, 2024).
this section cite: ['b11', 'b56']

Section: Budget-Constrained Awareness Testing.
The application of LLMs is limited by computational resources and user patience (Wen et al., 2025;Li et al., 2025e). We introduce a budget-constrained awareness mechanism to both evaluation settings, where interactions are capped by an adaptive budget and systems are informed of the remaining budget. This enables evaluation under varying budgets, including stress-testing (Ahmad et al., 2025;Zhang et al., 2025) in low-budget conditions to assess the system's ability to ask the right questions and plan effectively. The specific budget settings are detailed in the following sections.
this section cite: ['b65', 'b0', 'b77']

Section: c-INTERACT EVALUATION
Interaction Setup. The c-Interact evaluation establishes a multi-turn dialogue between user simulator U and system S. The session unfolds in two sequential phases of sub-tasks: First, U presents an underspecified sub-task q 1 alongside database metadata M and knowledge base K. System S may engage in clarification dialogue before generating SQL σ 1 . Upon successful validation against test cases T 1 , U issues a contextually coherent follow-up sub-task q 2 , prompting S to respond with SQL σ 2 . Each sub-task incorporates a single debugging opportunity: following query failure, S may
Environment (a) Databases Knowledge Base Doc schema.txt cols_meaning.jsonl kb.jsonl (c) User Simulator Alien Arch Robots News DB Metadata Files DBs Clarify Ambiguity Make Follow-Up Request Make Initial Request Test the Submitted SQL User Actions Follow-Up Question Debugging Ambiguity Resolution Reward c-Interact Environment User Sim. Database Env. Initial Request Action Observation Reward Interaction a-Interact LLM as Parser
this section cite: []

Section: Clarification

this section cite: []

Section: Asked Question
Gold SQL Labeled Amb. Real Intent in SQL "some" → → SELECT name, id "info" LIMIT 5
this section cite: []

Section: Two-Stage Clarifications
Figure 3: Two evaluation settings for BIRD-INTERACT: c-Interact, where the system engages in conversation with the user, and a-Interact, where the system interacts flexibly. At the end of the task, the system will receive a reward r ∈ [0, 1].
submit one revised query after receiving execution feedback from U. Each debugging attempt incurs a reward penalty to account for the additional computational cost, details can be found in Figure 3. The evaluation episode concludes when both sub-tasks are successfully completed or all attempts are exhausted. Notably, failure in the initial priority sub-task immediately terminates the entire session.
this section cite: []

Section: Budget Constraints.
The budget is implemented as a constraint on the number of clarification turns. The total allowed turns, τ clar , are calculated as follows: τ clar = m amb + λ pat .
Here, m amb represents the minimum budget required to resolve the ambiguities, which is equal to the number of annotated ambiguities in the user task. The parameter λ pat is a tunable variable that simulates different levels of user patience, granting the evaluated system extra turns for clarification.
this section cite: []

Section: a-INTERACT EVALUATION
Interaction Setup. The a-Interact provides LLMs with autonomous planning and execution within a pre-defined action space, following REACT paradigm (Yao et al., 2023). We model the complete database environment as a set of callable tools, containing the target database, metadata, HKB, and User Simulator, allowing the agent to determine optimal invocation strategies dynamically. In this work, we summarize and define 9 discrete actions common to text-to-SQL with details in Appendix J. BIRD-INTERACT also supports customized scaffolds, details can be found in Appendix J.2.
this section cite: ['b71']

Section: Budget Constraints.
To reflect the varying computational costs of different actions, we implement a budget-constrained evaluation framework where each action consumes a predetermined amount of budget, encouraging cost-effective action sequences. The total budget for each task is B = B base + 2 m amb + 2 λ pat , where B base = 6 is the base budget, m amb is the number of annotated ambiguity points, and λ pat is the user patience parameter, maintaining consistency with the c-Interact framework. This setting evaluates the agent's ability to achieve high performance under resource constraints while balancing thoroughness with efficiency. Further details of action costs are provided in Appendix J.
This setting can evaluate agent performance under realistic constraints that present practical database interaction scenarios, where users have limited patience and computational resources are finite.
this section cite: []

Section: EXPERIMENT
We benchmark 7 recent and powerful LLMs (2 open-source, 5 closed-source) as system models via a fresh PostgreSQL 14 Docker instance for more stable evaluation. We set the user patience to 3 by default and a-Interact base budget of 6. All models use temperature=0 and top_p=1, with default reasoning settings, conducting single runs due to cost (full details in Appendix I.2 and I.3).
this section cite: []

Section: MAIN RESULTS
Table 2 summarizes the success rate (SR) and normalized reward (NR) obtained by 7 representative frontier LLMs on BIRD-INTERACT-FULL. The full experimental results of BIRD-INTERACT-LITE can be found in Table 10. We can observe:
BIRD-INTERACT Remains Challenging, Leaving Ample Room for Future Improvement. Even the strongest models in our study, Gemini-2.5-Pro and GPT-5, capture only 20.92% and 25.52% of the available reward respectively, in the c-Interact and a-Interact mode. Absolute success rates reveal similar limitations: no more than 16.33% of tasks are solved end-to-end in c-Interact and 17.00% in a-Interact, with most models falling in substantially lower rates.
Evolving User Intent is a Challenge in Online Assessment. Follow-up sub-tasks are noticeably more challenging, likely because the longer, concatenated context in these turns remains a bottleneck for LLMs in interactive text-to-SQL tasks.
Offline Reward v.s. Online SR Evaluation. Table 2 shows that offline normalized reward (NR) and online success rate (SR) generally correlate positively, though notable divergences occur due to the reward structure allocating 70% to the primary sub-task and 30% to follow-up sub-tasks. These complementary metrics capture different aspects of model performance. Success rate measures holistic task completion across multi-turn interactions, relevant when users prioritize successful outcomes regardless of path. Normalized reward assesses performance on users' critical initial objectives while crediting challenging follow-up sub-tasks. Together, they provide comprehensive evaluation of the distinct capabilities required for advanced interactive text-to-SQL systems.
Business Intelligence versus Data Management. Business intelligence (BI) queries pose significantly greater challenges for LLMs compared to data management (DM) tasks since DM operations typically follow standardized, predictable patterns that LLMs can effectively learn (Li et al., 2025d), whereas BI queries demand nuanced understanding of complex, domain-specific business logic and analytical reasoning that varies substantially across contexts.
Interaction Mode Emerged as the Decisive Factor for a Successful Outcome. Furthermore, we observe that different models demonstrate varying aptitudes for different interaction paradigms, with each model showing relative strengths in specific modes. For example, GPT-5 performs poorly in the constrained, predefined flow designed personally of the c-Interact mode by achieving only 14.50% SR (worst) but excels in the a-Interact setting with 29.17% SR (best), which affords more flexible and exploratory space. This evidence demonstrates the critical importance of matching interaction modes to model-specific capabilities, which we hypothesize stem from differences in training data distributions and architectural inductive biases (Liu et al., 2024;Gao et al., 2024b). The Impact of Communication on Task Success in c-Interact. A notable finding is the underperformance of the flagship model, GPT-5, on the c-Interact, despite its strong performance on many single-turn tasks (Phan et al., 2025;Glazer et al., 2024;Rein et al., 2024). Therefore, we hypothesize that this stems from a deficiency in its interactive communication abilities rather than its core generation capability.
this section cite: ['b43', 'b48', 'b18', 'b54']

Section: INTERACTION ANALYSIS
To test this hypothesis, we conduct an experiment termed Memory Grafting. In this setup, we provide GPT-5 with the ambiguity resolution histories from two other better models, Qwen-3-Coder and O3-mini, before asking it to generate the final SQL query. The results, presented in Figure 5, show that GPT-5's performance improves significantly when leveraging the interaction history from either model. This finding indicates that while GPT-5 possesses robust SQL generation capabilities, a more effective communication schema is required to help it achieve satisfactory outcomes for user tasks. We also further analyze the patterns for effective communication in Appendix P.
this section cite: []

Section: Interaction Test-Time Scaling.
To investigate the relationship between interaction frequency and model performance, we conduct an Interaction Test-Time Scaling (ITS) experiment in BIRD-INTERACT-LITE where results are shown in Figure 4. We simulate varying levels of user patience by allowing different numbers of interaction turns for both c-Interact and a-Interact. As a baseline, we include single-turn task performance for each model, where all necessary context is provided to create unambiguous tasks. This single-turn condition represents an idealized scenario that, while potentially requiring significant user effort to ensure complete information provided (Li et al., 2025d), eliminates the need for further clarification. As demonstrated in the figure, Claude-3.7-Sonnet exhibits clear scaling behavior with respect to increasing interaction opportunities.
This pattern shows that the model can steadily improve by transforming additional interaction chances into valuable information gains through efficient interaction.
ITS Law: A model satisfies this law if, given enough interactive turns, its performance can match or even surpass that of the idealized single-turn task.
Action Distribution Patterns in a-Interact. We analyze action distributions across 7 system models and find concentration in two primary actions: submit (direct code execution with error feedback) and ask (user clarification requests), which together comprise 60.87% of all actions. Despite being the most computationally expensive actions (Figure 3), models favor these over systematic exploration behaviors like knowledge and schema retrieval. This suggests LLMs prefer direct trial-and-error execution over comprehensive environment exploration, likely due to pretraining biases. Future work should incentivize broader tool utilization for complex interactive tasks. Additional analysis on the FULL set appears in Appendix J.
this section cite: []

Section: USER SIMULATOR ANALYSIS
This section presents a comprehensive evaluation of our function-driven user simulator compared to conventional user simulators and their respective impacts on dynamic interactive text-to-SQL benchmarks through both objective and subjective experiments.
this section cite: []

Section: Evaluation on USERSIM-GUARD.
To provide an objective and comprehensive observation of different user simulator mechanisms, we construct a static dataset called USERSIM-GUARD, comprising 2,100 questions with reference actions labeled by human experts. Detailed information regarding the distribution and annotation procedures can be found in Appendix O. We employed an LLM-as-Judge (Zheng et al., 2023) evaluation framework using Qwen3-235B-A22B-Instruct-2507 as independent evaluators. Our analysis reveals significant reliability concerns with conventional user simulator designs. Specifically, as shown in Figure 6, when confronted with Unanswerable (UNA) questions, baseline user simulators consistently fail to implement safeguards, resulting in unfair or inappropriate feedback generation with failure rates reaching up to 67.4% depending on the backbone model. In contrast, our proposed function-driven approach demonstrates substantially improved reliability, reducing the failure rate to as low as 2.7%. This represents a dramatic improvement in user simulator robustness and reliability compared to baseline approaches. Alignment with Human User. We evaluate alignment between our user simulators and actual human behavior by having human experts interact with 7 system models on 100 randomly sampled tasks across BI and DM domains. We then compute correlations (Ivey et al., 2024;Kong et al., 2024) between success rates (SR) achieved by human users versus our simulators across the same tasks. As shown in Table 3, function-driven simulators demonstrate significantly stronger alignment with human behavior: GPT-4o with function calling achieves 0.84 Pearson correlation (p = 0.02) compared to 0.61 without function calling (p = 0.14), while Gemini-2.0-Flash shows similar improvements.
These results confirm that incorporating our designed mechanism produces more realistic user simulators that better reflect actual human-AI interaction patterns (detailed analysis in Appendix O).
this section cite: ['b78', 'b26']

Section: RELATED WORK
Text-to-SQL. Text-to-SQL has emerged as an attractive interface to relational databases because it frees users from learning intricate schema details and SQL syntax. The advent of large language models (LLMs) (OpenAI, 2025;Team et al., 2023;Team, 2024;Guo et al., 2025;Li et al., 2023a;Qu et al., 2025) with strong reasoning and cross-domain generalization has accelerated this progress. Few-shot systems such as DIN-SQL (Pourreza & Rafiei, 2023) and DAIL-SQL (Gao et al., 2024a) exploit in-context learning to decouple the task into schema-linking and SQL-generation stages, while methods like CodeS (Li et al., 2024a) and DTS-SQL (Pourreza & Rafiei, 2024) improve smaller models through carefully curated, high-quality training subsets. Concurrently, agent-based frameworks that interleave thought, action, and observation, which are exemplified by MAC-SQL (Wang et al., 2025), demonstrate that iterative interaction with the environment can further raise SQL accuracy. Despite these advances, virtually all existing systems are evaluated only in single-turn settings; their effectiveness in conversational, multi-turn text-to-SQL scenarios remains an open question.
Multi-turn Text-to-SQL. Multi-turn Text-to-SQL addresses the reality that user queries are often ambiguous or underspecified; without clarification the system may return incorrect or empty results. Benchmarks such as COSQL and LEARN-TO-CLARIFY extend the Spider (Yu et al., 2018) dataset with dialogue turns to probe this challenge (Yu et al., 2019a;Chen et al., 2025b;Li et al., 2024b). However, these resources presuppose a static, noise-free dialogue history shared by all models, ignoring that different systems might ask different follow-up questions (Yao et al., 2025;Barres et al., 2025). More recent evaluations of autonomous agents, for example, MINT, introduce dynamic interaction histories (Wang et al., 2024), yet they have not been adapted to the text-to-SQL setting. Constructing a realistic user simulator for databases is non-trivial because it must respect complex schema constraints while keeping the answer space fair and controllable (Zhou et al., 2025;Barres et al., 2025). In this work, we fill this gap by proposing an interactive benchmark that is implemented with an optimized user simulator, new databases, and knowledge, and we analyze the behaviour of state-of-the-art reasoning models rigorously to make contributions for realistic and uncertain text-to-SQL systems.
this section cite: ['b47', 'b61', 'b60', 'b20', 'b53', 'b49', 'b74', 'b75', 'b35', 'b72', 'b1', 'b64', 'b80', 'b1']

Section: FUTURE WORK
While BIRD-INTERACT establishes a comprehensive framework for evaluating interactive textto-SQL systems, several directions remain for future investigation. First, we plan to develop a post-trained, human-aligned local user simulator via post-training, aiming to capture more reliable response patterns while maintaining controllability and reducing API cost. Second, our current a-Interact setting imposes strict budget constraints that create a stress-mode evaluation environment, placing considerable pressure on LLM agents to make optimal decisions under resource scarcity. To complement these findings, we will conduct experiments in a free-mode setting without the budgetconstrained awareness testing (Section 4). This could allow us to observe natural interaction strategies when models are unconstrained, identify whether more sophisticated exploration patterns emerge, and characterize the relationship between interaction thoroughness and task success. Comparing stress-mode and free-mode performance will provide deeper insights into efficiency-effectiveness trade-offs in interactive text-to-SQL systems.
this section cite: []

Section: CONCLUSION
We present BIRD-INTERACT, a benchmark for evaluating interactive text-to-SQL systems through dynamic, multi-turn interactions that better reflect real-world usage scenarios. Our benchmark features a function-driven user simulator, dual evaluation settings for conversational and autonomous planning modes, and totally 900 challenging tasks designed to test LLM abilities to handle ambiguities and maintain state across turns. Comprehensive evaluation demonstrates a critical gap between existing SQL generation capabilities and the strategic interaction skills required for effective human-AI collaboration in database querying.
this section cite: []

Section: References
Ref_id:b0 Title: Openai's approach to external red teaming for ai models and systems Year: (2025)
Ref_id:b1 Title: Evaluating conversational agents in a dual-control environment Year: (2025)
Ref_id:b2 Title: Benchmarking and improving text-to-SQL generation under ambiguity Year: (2023-12)
Ref_id:b3 Title: Livesqlbench: A dynamic and contamination-free benchmark for evaluating llms on real-world text-to-sql tasks Year: (2025)
Ref_id:b4 Title: E-sql: Direct schema linking via question enrichment in text-to-sql Year: (2024)
Ref_id:b5 Title: Rsl-sql: Robust schema linking in text-to-sql generation Year: (2024)
Ref_id:b6 Title: Accounting for focus ambiguity in visual questions Year: (2025)
Ref_id:b7 Title: Learning to clarify: Multiturn conversations with action-based contrastive self-training Year: (2025)
Ref_id:b8 Title: Beyond read-only: Crafting a comprehensive Chinese text-to-SQL dataset for database manipulation and query Year: (2024-06)
Ref_id:b9 Title: Expanding the scope of the ATIS task: The ATIS-3 corpus Year: (1994)
Ref_id:b10 Title: Infoquest: Evaluating multi-turn dialogue agents for open-ended conversations with hidden context Year: (2025)
Ref_id:b11 Title: Wizard of wikipedia: Knowledge-powered conversational agents Year: (2019)
Ref_id:b12 Title: Ambisql: Interactive ambiguity detection and resolution for text-to-sql Year: (2025)
Ref_id:b13 Title: Practiq: A practical conversational text-to-sql dataset with ambiguous and unanswerable queries Year: (2025)
Ref_id:b14 Title: Towards knowledge-intensive text-to-SQL semantic parsing with formulaic knowledge Year: (2022-12)
Ref_id:b15 Title: Wenjing Wang, and Carlo Curino. NL2SQL is a solved problem Year: (2024)
Ref_id:b16 Title: Text-to-sql empowered by large language models: A benchmark evaluation Year: (2024-01)
Ref_id:b17 Title: A taxonomy for human-llm interaction modes: An initial exploration Year: (2024)
Ref_id:b18 Title: Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai Year: (2024)
Ref_id:b19 Title: A survey on llm-as-a-judge Year: (2024)
Ref_id:b20 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b21 Title: Chase: A large-scale and pragmatic Chinese dataset for cross-database context-dependent text-to-SQL Year: (2021-08)
Ref_id:b22 Title: Text-to-SQL in the wild: A naturally-occurring dataset based on stack exchange data Year: (2021-08-04)
Ref_id:b23 Title: Data ambiguity strikes back: How documentation improves gpt's text-to-sql Year: (2023)
Ref_id:b24 Title: Zeroea: a zero-training entity alignment framework via pre-trained language model Year: (2024)
Ref_id:b25 Title: Micro-act: Mitigate knowledge conflict in question answering via actionable self-reasoning Year: (2025-07)
Ref_id:b26 Title: Real or robotic? assessing whether llms accurately simulate qualities of human responses in dialogue Year: (2024)
Ref_id:b27 Title: Large language models as user-agents for evaluating task-oriented-dialogue systems Year: ()
Ref_id:b28 Title: Platolm: Teaching llms in multi-round dialogue via a user simulator Year: ()
Ref_id:b29 Title: When prompts go wrong: Evaluating code model robustness to ambiguous, contradictory, and incomplete task descriptions Year: (2025)
Ref_id:b30 Title: Realistic evaluation of text-to-sql parsers Year: ()
Ref_id:b31 Title: MCS-SQL: Leveraging multiple prompts and multiple-choice selection for text-to-SQL generation Year: (2025-01)
Ref_id:b32 Title: Spider 2.0: Evaluating language models on real-world enterprise text-to-sql workflows Year: (2025)
Ref_id:b33 Title: Alpha-SQL: Zero-shot text-to-SQL using monte carlo tree search Year: (2025)
Ref_id:b34 Title: Codes: Towards building open-source language models for text-to-sql Year: ()
Ref_id:b35 Title: Omnisql: Synthesizing high-quality text-to-sql data at scale Year: (2025-09)
Ref_id:b36 Title: Graphix-t5: Mixing pre-trained transformers with graph-aware layers for text-to-sql parsing Year: (2023)
Ref_id:b37 Title: Can LLM already serve as a database interface? a BIg bench for large-scale database grounded text-to-SQLs Year: (2023)
Ref_id:b38 Title: Tapilot-crossing: Benchmarking and evolving llms towards interactive data analysis agents Year: (2024)
Ref_id:b39 Title: Are large language models ready for multi-turn tabular data analysis? Year: (2025)
Ref_id:b40 Title: SWE-SQL: Illuminating LLM pathways to solve user SQL issues in real-world applications Year: (2025)
Ref_id:b41 Title: Steering llm thinking with budget guidance Year: (2025)
Ref_id:b42 Title: CondAmbigQA: A benchmark and dataset for conditional ambiguous question answering Year: (2025-11)
Ref_id:b43 Title: Agentbench: Evaluating LLMs as agents Year: (2024)
Ref_id:b44 Title: Xiyan-sql: A novel multi-generator framework for text-to-sql Year: (2025)
Ref_id:b45 Title: The death of schema linking? text-to-sql in the age of well-reasoned language models Year: (2024)
Ref_id:b46 Title: Ambigqa: Answering ambiguous open-domain questions Year: (2020)
Ref_id:b47 Title: Openai o3 and o4-mini system card Year: (2025)
Ref_id:b48 Title: Humanity's last exam Year: (2025)
Ref_id:b49 Title: DIN-SQL: Decomposed in-context learning of text-to-SQL with self-correction Year: (2024-11)
Ref_id:b50 Title: CHASE-SQL: Multi-path reasoning and preference optimized candidate selection in text-to-SQL Year: (2025)
Ref_id:b51 Title: Reasoning-sql: Reinforcement learning with sql tailored partial rewards for reasoning-enhanced text-to-sql Year: (2025)
Ref_id:b52 Title: Before generation, align it! a novel and effective strategy for mitigating hallucinations in text-to-SQL generation Year: (2024-08)
Ref_id:b53 Title: SHARE: An SLM-based hierarchical action CorREction assistant for text-to-SQL Year: (2025-07)
Ref_id:b54 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b55 Title: AMBROSIA: A benchmark for parsing ambiguous questions into database queries Year: (2024)
Ref_id:b56 Title: SLM-SQL: An exploration of small language models for text-to-SQL Year: (2024-12)
Ref_id:b57 Title: A survey on employing large language models for text-to-sql tasks Year: (2024)
Ref_id:b58 Title: Chess: Contextual harnessing for efficient sql synthesis Year: (2024)
Ref_id:b59 Title: Question-negotiation and information seeking in libraries Year: (2015)
Ref_id:b60 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b61 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b62 Title: Rat-sql: Relation-aware schema encoding and linking for text-to-sql parsers Year: (2020)
Ref_id:b63 Title: Mac-sql: A multi-agent collaborative framework for text-to-sql Year: ()
Ref_id:b64 Title: Mint: Evaluating llms in multi-turn interaction with tools and language feedback Year: (2024)
Ref_id:b65 Title: Empowering budget-aware llm reasoning with control tokens Year: (2025)
Ref_id:b66 Title: From passive responders to active collaborators Year: (2025)
Ref_id:b67 Title: A survey on knowledge distillation of large language models Year: (2024)
Ref_id:b68 Title: Re-reading improves reasoning in large language models Year: (2024-11)
Ref_id:b69 Title: Intercode: Standardizing and benchmarking interactive coding with execution feedback Year: (2023)
Ref_id:b70 Title: Towards scalable real-world web interaction with grounded language agents Year: (2022)
Ref_id:b71 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b72 Title: {$\tau$}-bench: A benchmark for \underline{T}ool-\underline{A}gent-\underline{U}ser interaction in real-world domains Year: (2025)
Ref_id:b73 Title: Natural language to code generation in interactive data science notebooks Year: (2023-07)
Ref_id:b74 Title: Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-SQL task Year: (2018-11)
Ref_id:b75 Title: CoSQL: A conversational text-to-SQL challenge towards cross-domain natural language interfaces to databases Year: (2019-11)
Ref_id:b76 Title: SParC: Cross-domain semantic parsing in context Year: (2019-07)
Ref_id:b77 Title: Stress-testing model specs reveals character differences among language models Year: (2025)
Ref_id:b78 Title: Judging LLM-as-a-judge with MT-bench and chatbot arena Year: (2023)
Ref_id:b79 Title: Seq2sql: Generating structured queries from natural language using reinforcement learning Year: (2017)
Ref_id:b80 Title: Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks Year: (2025)
Ref_id:b81 Title: Claude-Sonnet-4 follows a similar pattern, but with heavier emphasis on execute (29.9%) and lighter use of submit (20.0%). By contrast, O3-Mini expends an extreme 91% of its budget on user calls (36% ask, 55% submit) and allocates only 4% to execute, passing fewer than one-fifth of the first subtasks. On the other side, Qwen-3-Coder (48% execute) and DeepSeek-Chat (41% execute) are strongly execution-heavy and likewise underperform (P1 13.3% and 17.2%). This contrast suggests that successful agents must strike a balance between exploring the environment and committing to user-facing actions Year: ()
