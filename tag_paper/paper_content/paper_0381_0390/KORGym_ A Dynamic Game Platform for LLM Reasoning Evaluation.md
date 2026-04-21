Title: KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation
Abstract: Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym) 1 , a dynamic evaluation platform inspired by KOR-Bench [1] and Gymnasium [2]. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments. gg-bench [12] relies heavily on generative capacity and lacks robustness in both gameplay fidelity and RL integration.To overcome these limitations, in Figure 1, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), inspired by the knowledge-orthogonal reasoning framework of KOR-Bench [1] (see Appendix C) and built on the reinforcement-learning environment Gymnasium [2]. Single-epoch Game Bench KORGymA cage has chickens and rabbits. There are 35 heads and 94 legs. How many chickens and how many rabbits? 23 chickens, 12 rabbits

Section: Introduction
Recent advances in reasoning models have yielded strong performance in tasks such as textual comprehension [3] and logical inference [4]. However, most benchmarks remain domain-specific (e.g., AIME [5], PHYBench [6]) and fail to capture general reasoning ability. Even benchmarks intended to evaluate broader reasoning (e.g., SuperGPQA [7], HLE [8]) are heavily influenced by pretraining data, limiting their capacity to measure intrinsic reasoning skills. To address this gap, we propose a benchmark designed to evaluate the intrinsic reasoning capabilities of LLMs independent of pretraining knowledge. Games, with their diverse scenarios rarely encountered in pretraining corpora, offer an ideal testbed for such evaluation.
While games offer a promising benchmark medium, existing approaches exhibit several shortcomings. LogicGame [9], for example, employs only single-turn scenarios, preventing evaluation of long-term planning in LLMs. TextArena [10] and SPINBench [11] support multi-turn scenarios but introduce opponent dynamics that generate extraneous variability, confounding pure reasoning assessment and limiting suitability for reinforcement learning (RL) by enabling hacked strategies. Moreover, Specifically, KORGym features over fifty games spanning six reasoning dimensions: mathematical and logical reasoning, control interaction reasoning, puzzle reasoning, spatial and geometric reasoning, strategic reasoning, and multimodal reasoning. The platform is organized into four modular components-the inference module, game interaction module, evaluation module, and communication module-enabling multiround evaluations, configurable difficulty levels, and stable reinforcement-learning support.
By integrating textual and multimodal challenges, KO-RGym provides a comprehensive assessment of LLMs' adaptability, strategic planning, and decision-making capabilities, thereby offering a more accurate reflection of their intrinsic reasoning abilities. Using KORGym, we conduct extensive experiments, yielding several key insights:
• Reasoning abilities display consistent profiles of strengths and weaknesses within the same model series.
• Modality influences reasoning performance, with distinct patterns observed between open-source and closed-source LLMs.
• Thinking models exhibit behavioral patterns distinct from those of non-thinking models.
• LLMs often employ explicit reasoning paradigms during problem-solving, which may partially constrain their performance.
• Appropriate reinforcement learning enhances reasoning capabilities and yields more balanced performance across different reasoning dimensions.
In summary, our main contributions are as follows:
• We design a suite of over fifty text-and vision-based games tailored to evaluate the reasoning capabilities of large language models. • We present KORGym, an extensible framework supporting incremental development and reinforcement-learning integration.
• We conduct a comprehensive empirical analysis of 19 LLMs and 8 vision-language models and uncover several key insights.
this section cite: ['b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b9', 'b10', 'b11']

Section: Related Work
LLMs for Gaming. Games serve as valuable testbeds for evaluating large language models (LLMs) due to their demands for multi-step reasoning and strategic planning. Early research focused on single-game evaluations in domains like Minecraft [13] or social deduction games [14,15], but these narrow settings limited generalizability. Subsequent efforts introduced broader benchmarks with diverse game types and multi-agent frameworks emphasizing coordination or competition, though critical dimensions such as open-ended negotiation, dynamic cooperation-conflict shifts, and rich social dynamics remained underexplored. Some researchers have designed PlanBench [16] to evaluate the long-term reasoning ability of LLMs and have further conducted studies [17] on reasoning models based on this work. Meanwhile, other researchers have proposed the Game Traversal Benchmark [18] to assess LLMs' planning and reasoning ability through the task of traversing 2D game maps. To address these gaps, SPIN-Bench [11] unifies strategic planning and social intelligence by combining formal planning analysis, multi-agent cooperation/competition, and open-ended dialogue. Existing benchmarks vary widely in environment diversity and technical capabilities. Some frameworks offer diverse environments but lack human evaluation, while others focus on specific scenarios yet miss key features. SPIN-Bench stands out with a balanced mix of game types, Gym compatibility, and model vs. model evaluation.
Knowledge Orthogonality Based Evaluation. Current AI reasoning benchmarks (e.g., MMLU [19], CommonsenseQA [20], MATH [21]) emphasize factual recall and problem-solving but often conflate memorization with reasoning, limiting insight into underlying cognitive processes. To address this, integration-based benchmarks (e.g., ZebraLogic [22], TravelPlanner [23]) test adaptability and creativity by requiring pattern recognition, logic, and multi-step reasoning in novel contexts. While these frameworks advance the focus on contextual problem-solving, they still risk entanglement with domain-specific knowledge biases, as seen in mathematical or logical benchmarks like GSM8K [24] and FOLIO [25]. To address these gaps, the concept of knowledge orthogonality advocates decoupling reasoning assessment from prior knowledge and prioritizing rule-following in out-of-distribution scenarios to isolate core abilities such as systematic generalization and hypothesis testing. This paradigm shift-from memorization-driven metrics to knowledge-agnostic, creativityfocused evaluations-establishes a fairer framework for measuring cognitive agility, ensuring models demonstrate genuine reasoning rather than reciting learned patterns and fostering AI systems with robust, human-like adaptability in open-world environments.
We propose KORGym, an efficient game-based framework for evaluating the complex reasoning capabilities of large language models (LLMs) via both single-turn and multi-turn text-based and multimodal games. KORGym is organized into three key modules:
• Evaluation and Communication Module: the system core, which parses input parameters, establishes inter-module communication protocols, encapsulates and transmits communication packets, and logs final evaluation scores.
• Game Interaction Module: encapsulates the game environment and interaction APIs, including:
-generate: initializes the game environment.
-print board: renders the game board and generates prompts.
-verify: updates the game state and computes scores.
• Inference Module: manages model inference processes, including asynchronous acceleration and intermediate result checkpointing.
Based on these modules, the primary inference workflow of KORGym proceeds as follows:
• Parameter Initialization: Load initial parameters (e.g., game type, seed, and model).
• Parameter Parsing and Communication Protocol Startup: Parse these parameters and encapsulate them into communication packets.
• Game Environment Initialization: Generate the game environment according to the parameters.
• Acquisition of Game Information: Invoke the generate and print board APIs in the Game Interaction Module to obtain the current environment state.
• Game Information Transmission: Package the retrieved information for transmission.
• Generation of LLM Action: Perform model inference to generate the next action.
• Transmission of LLM Action: Send the action via the verify API to update the game state; if the game is not concluded, return to Acquisition of Game Information.
• Score Calculation and Result Output: Compute and output the final score.
this section cite: ['b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b11', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25']

Section: Task Introduction
As illustrated in Figure 9 (Appendix H), KORGym supports over fifty novel games, enabling precise and efficient evaluation of the reasoning abilities of large language models (LLMs) across six distinct dimensions (Table 11): Mathematical and Logical Reasoning, Control Interaction Reasoning, Puzzle Reasoning, Spatial and Geometric Reasoning, Strategic Reasoning and Multimodal Reasoning.
During benchmark development, we selected more than fifty games that effectively capture the reasoning capabilities of LLMs. These games span four categories: traditional puzzles (e.g., Sudoku); adaptations of classic video games (e.g., Plants vs. Zombies; Minesweeper); game-theoretic challenges (e.g., N-point; Evolution of Trust); and multimodal tasks (e.g., Jigsaw; Circle the Cat).
KORGym offers a suite of over fifty games-with continuous expansion-that support multi-turn interactions via standardized APIs (generate, verify, and print board). The platform is tailored for RL, providing environment states and reward signals, and enables users to adjust game difficulty and environmental diversity through scalable parameters. Additionally, it includes nine multimodal games, facilitating comprehensive evaluation in both textual and multimodal contexts. Related platforms include LogicGame [9], AgentBench [26], GameArena [27], SPIN-Bench [28], TEXTARENA [10], and ReasoningGYM [29]. A detailed comparison appears in Table 10 of Appendix I.
this section cite: ['b9', 'b26', 'b27', 'b28', 'b10', 'b29']

Section: Evaluation Method
Score Calculation Rules To address the limitations of binary (0/1) scoring in reflecting intermediate progress in KORGym, we propose a comprehensive scoring scheme comprising three rules:
• Binary Scoring: For single-objective games, assign 1 point for success and 0 for failure. For example, in 7-Maze, reaching the exit yields a score of 1.
• Proportional Scoring: For multiple-choice games, the score equals the number of correct responses divided by the total number of options. For instance, in the 44-Jigsaw Puzzle, the score is the number of correctly placed pieces over the total pieces.
• Cumulative Scoring: For games that award incremental points, accumulate all points earned.
For example, in 3-2048, each tile merge contributes to the final score.
this section cite: []

Section: Capability Dimension Aggregated Mean
Raw game scores in KORGym can extend beyond the [0,1] interval and may be skewed by variations in game difficulty or by outlier model behaviors.
To mitigate these issues, we introduce Capability Dimension Aggregated Mean, a more robust aggregation metric for evaluating model performance across reasoning dimensions.
Formally, let G = {g 1 , g 2 , . . . , g N } denote the set of all games, M = {m 1 , m 2 , . . . , m K } denote the set of models under evaluation, and D = {d 1 , d 2 , . . . , d L }represent the set of reasoning capability dimensions. Each game g ∈ G is associated with a specific dimension d(g) ∈ D. Let S g,m denote the raw score achieved by model m ∈ M on game g ∈ G. For each game g, if the maximum score across all models exceeds 1, i.e., max m∈M S g,m > 1,we apply a log p transformation (i.e., ln(1 + x)) to compress large score values and reduce skewness; otherwise, we retain the original score:
S ′ g,m = ln 1 + S g,m , if max m∈M S g,m > 1, S g,m , otherwise.(1)
To normalize scores across games, we further define, for each game g,
a g = min m∈M S ′ g,m , b g = max m∈M S ′ g,m .(2)
If b g = a g , meaning all models perform identically on game g, we assign every model a normalized score of 0.5 to avoid division by zero. Otherwise, we normalize its adjusted score:
S g,m = S ′ g,m -a g b g -a g , ∀m ∈ M.(3)
This normalization ensures that for each game, model performances are mapped into the [0, 1] range while preserving relative differences. Subsequently, for each capability dimension d ∈ D, we define the corresponding set of games and aggregated score of model m on dimension d as:
G d = {g ∈ G : d(g) = d}, S d,m = 1 |G d | g∈G d S g,m .(4)
The resulting matrix {S d,m } d∈D, m∈M provides a normalized, dimension-wise evaluation of reasoning capabilities that is fair across heterogeneous games.
this section cite: []

Section: Experiments 4.1 Settings
LLMs To comprehensively evaluate LLM performance, we assessed 19 large language models-including 11 thinking models and 8 instruction-tuned models-and 8 vision-language models (Table 12).
Evaluation Setting During evaluation, we apply distinct protocols for single-epoch and multipleepoch games:
• Single-epoch Games: Each model is evaluated on 50 independently initialized game instances by varying the "seed" parameter in the "generate" API from 1 to 50.
• Multiple-epoch Games: For each model, we initialize 20 game environments. Each episode permits up to 100 interaction rounds, and we vary the "seed" parameter in the "generate" API from 1 to 50 for reproducibility.
All assessments use a zero-shot prompting setup to gauge genuine reasoning capabilities, retaining each model's default sampling parameters (temperature and top-p). We evaluate closed-source models via their hosted APIs and open-source models on eight NVIDIA A100-80G GPUs.
this section cite: []

Section: Similar Strength-Weakness Profiles Within Same Model Series
Figure 3a shows that O1 and O3-mini excel in spatial reasoning, whereas the Gemini series leads in mathematical and puzzle reasoning.
Closed-Source Models Demonstrate Superior Reasoning Performance O3-mini achieves the highest overall score on KORGym, particularly in spatial reasoning. Claude-3.7-thinking and Gemini-2.5-pro top puzzle reasoning, while Doubao-1.5-thinking-pro and DeepSeek-R1 deliver balanced performance across dimensions. In contrast, open-source models lag behind.
this section cite: []

Section: Impact of Model Scale and Architecture on Reasoning Capabilities
Figure 3b demonstrates that model performance scales positively with model size and thinking models outperform sizematched non-thinking variants. For instance, DeepSeek-R1-Distill-Qwen-32B, though smaller in scale, exceeds the performance of Qwen2.5-72B-Instruct.
this section cite: []

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks Year: (2024-10)
Ref_id:b1 Title: Gymnasium: A Standard Interface for Reinforcement Learning Environments Year: (2024-07)
Ref_id:b2 Title: Evaluating reading comprehension exercises generated by LLMs: A showcase of ChatGPT in education applications Year: (2023-07)
Ref_id:b3 Title: Cryptox : Compositional reasoning evaluation of large language models Year: (2025)
Ref_id:b4 Title: AIME: AI System Optimization via Multiple LLM Evaluators Year: (2024-10)
Ref_id:b5 Title: PHYBench: Holistic Evaluation of Physical Perception and Reasoning in Large Language Models Year: (2025-04)
Ref_id:b6 Title:  Year: (2025-02)
Ref_id:b7 Title:  Year: ()
Ref_id:b8 Title:  Year: (2025-01)
Ref_id:b9 Title: LogicGame: Benchmarking Rule-Based Reasoning Abilities of Large Language Models Year: (2024-08)
Ref_id:b10 Title:  Year: (2025-04)
Ref_id:b11 Title: Spin-bench: How well do llms plan strategically and reason socially? Year: (2025)
Ref_id:b12 Title: Measuring General Intelligence with Generated Games Year: (2025-05)
Ref_id:b13 Title: Mindagent: Emergent gaming interaction Year: (2024)
Ref_id:b14 Title: Avalonbench: Evaluating llms playing the game of avalon Year: (2023)
Ref_id:b15 Title: Exploring large language models for communication games: An empirical study on werewolf Year: (2023)
Ref_id:b16 Title: PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change Year: (2022-06)
Ref_id:b17 Title: LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench Year: (2024-09)
Ref_id:b18 Title: GameTraversalBenchmark: Evaluating Planning Abilities Of Large Language Models Through Traversing 2D Game Maps Year: (2024-10)
Ref_id:b19 Title: Measuring Massive Multitask Language Understanding Year: (2020-09)
Ref_id:b20 Title: CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge Year: (2018-11)
Ref_id:b21 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b22 Title: Zebralogic: Benchmarking the logical reasoning ability of language models Year: (2024)
Ref_id:b23 Title: Travelplanner: A benchmark for real-world planning with language agents Year: (2024)
Ref_id:b24 Title: Training Verifiers to Solve Math Word Problems Year: (2021-10)
Ref_id:b25 Title: FOLIO: Natural Language Reasoning with First-Order Logic Year: (2022-09)
Ref_id:b26 Title:  Year: (2023-08)
Ref_id:b27 Title: GameArena: Evaluating LLM Reasoning through Live Computer Games Year: (2024-12)
Ref_id:b28 Title: SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially? Year: (2025-03)
Ref_id:b29 Title:  Year: ()
Ref_id:b30 Title: Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks Year: (2023)
Ref_id:b31 Title: Gpt-4o system card Year: ()
Ref_id:b32 Title: Learning to reason with llms Year: (2024)
Ref_id:b33 Title: Learning to reason with llms Year: (2025)
Ref_id:b34 Title: Claude 3.7 sonnet and claude code Year: (2025)
Ref_id:b35 Title: Doubao-1.5-pro Year: (2025)
Ref_id:b36 Title:  Year: (2025-04)
Ref_id:b37 Title: Doubao-vision-pro Year: (2025)
Ref_id:b38 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b39 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b40 Title: Qwen3 Year: (2025)
Ref_id:b41 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b42 Title: Qwen-qwq Year: (2025)
Ref_id:b43 Title:  Year: (2025-02)
Ref_id:b44 Title: Gemini 2.0 flash thinking Year: (2024)
Ref_id:b45 Title: Gemini 2.5: Our most intelligent ai model Year: (2025)
Ref_id:b46 Title: InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models Year: (2025-04)
