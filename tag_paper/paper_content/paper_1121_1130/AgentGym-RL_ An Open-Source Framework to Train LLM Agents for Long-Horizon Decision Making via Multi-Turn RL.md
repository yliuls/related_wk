Title: AGENTGYM-RL: AN OPEN-SOURCE FRAMEWORK TO TRAIN LLM AGENTS FOR LONG-HORIZON DECISION MAKING VIA MULTI-TURN RL
Abstract: Training LLM agents for complex multi-turn decision-making tasks requires extensive exploration within their environment, with reinforcement learning (RL) as a natural way. However, the open-source community currently lacks a unified RL framework capable of training agents from scratch across diverse and realistic environments. To bridge this gap, we introduce AgentGym-RL, a modular and decoupled framework specifically designed for RL-based agent in multi-turn decision-making tasks. It offers high flexibility and extensibility, supports mainstream RL algorithms, and spans a broad range of real-world scenarios. To effectively train agents for challenging tasks, we argue that they are required to expand external interactions with the environment, rather than relying solely on internal reasoning. Nevertheless, training agents for long-horizon interaction with vanilla methods often faces challenges like training instability. To this end, we propose ScalingInter-RL, a staged training approach for stable long-horizon RL training. It starts with short-horizon interaction to establish foundational policies and progressively expands them to encourage deeper exploration. Extensive experiments show that agents trained with our method achieve performance on par with-or even surpass-commercial counterparts like OpenAI o3 and Gemini-2.5-Pro across 27 tasks in diverse environments. We share key insights and release the full framework, including code and datasets, to empower the community in building the next generation of intelligent agents. Our framework is available at https://github.com/WooooDyy/AgentGym-RL.

Section: INTRODUCTION
As Large Language Models (LLMs) rapidly advance (OpenAI, 2023;Anthropic, 2024;DeepSeek-AI et al., 2024;Team et al., 2023;Yang et al., 2025b), their applications have extended from chatbots to autonomous agents addressing long-horizon real-world decision-making tasks (Xi et al., 2025a;Moonshot AI, 2025). Analogous to human cognitive development, LLM agents are expected to acquire new knowledge and skills by actively exploring with the environment (Xi et al., 2025b;OpenAI, 2025).
Reinforcement learning (RL) is a natural choice for achieving this, demonstrating success in LLM reasoning (DeepSeek-AI et al., 2025;Jaech et al., 2024;Xi et al., 2024a;Trung et al., 2024;Team et al., 2025b;He et al., 2025). While recent efforts have sought to extend RL methodologies to develop LLM agents with multi-turn interaction capabilities (Zhou et al., 2024b;Chen et al., 2025; Figure 1: Left: Performance of proprietary models, open-source models, and our RL models across different agentic tasks. Right: Performance w.r.t model scale. Wang et al., 2025;Qi et al., 2025;Jin et al., 2025b;Cao et al., 2025), they still struggle with limited task complexity and insufficient environmental diversity. Critically, the open-source community lacks unified RL framework capable of training agents from scratch across diverse, realistic environments.
To bridge this gap, we introduce AgentGym-RL ( §3), a unified framework designed for training LLM agents through RL in multi-turn interactive decision-making tasks (Figure 2). With a modular and decoupled architecture, AgentGym-RL enables clean separation of agents, environments, and learning algorithms, offering high extensibility and flexibility for diverse research needs. The framework supports mainstream RL algorithms, and covers a wide range of real-world scenarios, e.g., web navigation (Zhou et al., 2024a;Yao et al., 2022), deep search (Wei et al., 2025;Jin et al., 2025b), digital games (Prasad et al., 2024;Fan et al., 2022), embodied tasks (Chevalier-Boisvert et al., 2019;Shridhar et al., 2021), and scientific tasks (Wang et al., 2022;Starace et al., 2025). Furthermore, to enhance agents' ability to tackle challenging tasks, we argue that expanding their interactions with the environment is crucial, rather than relying solely on internal reasoning. However, our preliminary experiments show that directly training agents for long-horizon interaction often faces instability. To address this, we propose ScalingInter-RL ( §4) based on AgentGym-RL. This progressively scaling interaction enables the agent to avoid repetitive and unproductive actions, enhance deeper exploration of environments, and ultimately achieve more effective and efficient task completion while maintaining training stability.
Extensive experiments ( §5) demonstrate that ScalingInter-RL within AgentGym-RL framework delivers significant performance gains across 27 tasks spanning 5 diverse scenarios (Figure 1(Left)). Open-source models , e.g., Qwen-2.5-7B (Yang et al., 2024), achieve an average improvement of 33.65 points, matching or even surpassing larger commercial models such as OpenAI-o3 (OpenAI, 2025) and Gemini-2.5-Pro (Comanici et al., 2025). In addition, we conduct extensive analytical experiments to provide key insights ( §6), showing that scaling both post-training and test-time interactions holds substantial potential for advancing agentic intelligence (Figure 1(Right)). We hope our work will serve as a valuable contribution to the community's progress.
this section cite: ['b1', 'b59', 'b36', 'b22', 'b62', 'b43', 'b2', 'b84', 'b41', 'b14', 'b7', 'b53', 'b63', 'b10']

Section: PRELIMINARIES

this section cite: []

Section: FORMULATION
In this work, we study the multi-turn interactive decision-making tasks, i.e., agentic tasks, and we model them as a Partially Observable Markov Decision Process (POMDP) (U, S, A, O, T , r) like (Xi et al., 2025b;Zhou et al., 2024b), where A, U, S, O, T : S × A → S, r: U × S → R represents the instruction space, the state space, the action space, the observation space, the deterministic state transition function, and the reward function, respectively. Given a task instruction u ∈ U , the agentic task requires the LLM agent to generate a sequence of actions a T k ∼ π θ (•|s k ) based on its policy π θ parameterized by θ to complete the given task, where a k ∈ A, and s k ∈ S, and T is the reasoning path (Yao et al., 2023). The agent then receives an observation o k ∈ O from the environment, and the state is then transitioned to T (s k , a k ) = s k+1 . Finally after N turns of interactions, the environment e provides an outcome reward r(τ ) ∈ [0, 1] to describe the completion of the multi-turn interactive decision-making tasks.
this section cite: ['b85']

Section: POLICY GRADIENT
We utilize policy gradient (PG) methods (Sutton et al., 1999) that optimizes our policy agent. They perform gradient ascent according to the objective J(θ), which is a function of the policy parameters θ. Specifically, J(θ) represents the expected cumulative reward the agent anticipates receiving when following policy π θ and interacting with the environment. Mathematically, this is expressed as the expectation of the total reward r(τ ) over trajectories τ generated by the policy: J(θ) = E τ ∼π θ [r(τ )]. To perform optimization on J(θ), we require the policy gradient ∇ θ J(θ).
In the vanilla policy gradient methods, the policy gradient can be estimated by:
∇ θ J(θ) = E τ ∼π θ r(τ ) K k=0 ∇ θ log π θ (a k |s k )(1)
where π θ is the policy parameterized by θ, τ represents a trajectory consisting of a sequence of states and actions, a k and s k are the action and state at time step k, and r(τ ) is the reward of the trajectory τ . Mainstream RL algorithms for training LLMs include PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), and REINFORCE++ (Hu, 2025)-all of which are integrated into our framework.
this section cite: ['b57', 'b48', 'b58']

Section: THE AGENTGYM-RL FRAMEWORK

this section cite: []

Section: ARCHITECTURE OVERVIEW
AgentGym-RL adopts a modular design with well-defined responsibilities for each module, allowing for extensibility. As shown in Figure 2, the framework is organized into three core modules.
this section cite: []

Section: Environment module.
In this module, each environment is encapsulated as an independent service with the option of deploying multiple replicas to support parallel requests. An environment
# Stage 1: Generate responses task_ids = expand(task_ids, sample_num) envs = create_env_clients(task_ids, "webarena", base_url) Do in parallel: for (env, task_id) in zip(envs, task_ids): env.reset(task_id) handlers =[ RolloutHandler().add_user_message(env.observe()) for env in envs] for i in range(max_rounds) prompts = [h.get_prompt() for h in handlers] responses = actor.generate(prompts) results = thread_safe_list() Do in parallel: for (env, response) in zip (envs, responses): results.append(env.step(response)) for (h, r, res) in zip(handlers, responses, results): h.add_assistant_message(r) h.add_user_message(res.state) h.score = res.score if all_done(handlers): break client communicates with the environment server via HTTP and exposes APIs to the agent, including /observation to get the current observation, /available actions to get the currently available actions, /step to perform an action, and /reset to reset the environment. Currently, AgentGym-RL covers five major scenario categories. This modular server-client design allows new environments to provide comprehensive environment and data support for LLM agent training.
Agent module. The agent module encapsulates the reasoning-action loop of LLM-based agents. It receives observations from the environment, performs reasoning over multiple turns, and outputs actions (e.g., invoking provided APIs). The module supports different prompting strategies and sampling configurations.
Training module. The training module provides a unified reinforcement learning (RL) pipeline that supports both online and offline algorithms, offering researchers a flexible foundation for largescale LLM agent training. The module manages the entire RL lifecycle: trajectory collection, advantage estimation, policy optimization, and reward shaping.
Workflow. The overall workflow and pseudocode are shown in Figure 3. Given a batch of queries and initial environment states, the framework initializes multiple parallel environment clients. Each client serves a single agent, ensuring isolated execution. At every step, the agent generates an action, the environment returns the updated state and reward, and the trajectories are collected concurrently for training updates.
The entire training pipeline can be distributed across multiple nodes, leveraging both multi-process and multi-node parallelism. Efficient batching and asynchronous logging utilities ensure that system throughput scales with additional compute resources.
this section cite: []

Section: FEATURES AND CHARACTERISTICS
The AgentGym-RL framework is built on AgentGym (Xi et al., 2025b), which provides several basic interactive environments for LLM agents. We have further extended it in diversity of environments, algorithm support, engineering optimizations, open-source availability, and interaction visualization.
Diverse scenarios and environments.
To build LLM agents capable of multi-turn decisionmaking, AgentGym-RL provides five heterogeneous environments spanning web navigation, deep search, digital games, embodied control, and scientific tasks. They exhibit significant variance in state space, action space, and reward structures. This cross-domain heterogeneity creates a testbed 7UDLQLQJ6WHSV 5HZDUGV 0D[,QWHUDFWLRQ5RXQGV 6FDOLQJ,QWHU5/ 0D[,QWHUDFWLRQ5RXQGV
this section cite: []

Section: 7UDLQLQJ6WHSV 5RXQGV
Figure 4: Training dynamics under different maximum interaction turns in Deep Search environment. Our ScalingInter-RL method progressively increases the interaction horizon, and ultimately achieves higher and more efficient long-term performance.
for training and evaluating research artifacts across diverse environments. A more detailed introduction of the environments we included is shown in Appendix C.
this section cite: []

Section: Comprehensive algorithm support.
While the original AgentGym (Xi et al., 2025b) focused primarily on SFT, AgentGym-RL places online reinforcement learning at the core of its training stack. It allows agents to adapt through continual interaction with the environment and move beyond static demonstration corpora. The framework unifies mainstream RL algorithms such as PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), RLOO (Chen et al., 2025) and REINFORCE++ (Hu, 2025) under a single interface, while also supporting complementary offline paradigms including SFT (Peng et al., 2023), DPO (Rafailov et al., 2023), and self-improvement (Xi et al., 2025b)).
Engineering optimizations. AgentGym-RL incorporates targeted engineering optimizations to support large-scale reinforcement learning research, with a focus on extensibility, scalability, and reliability. For extensibility, the framework adopts a modular plug-and-play design, allowing new environments to be integrated by simple inheritance from base classes. For scalability, we enhance both computational parallelism and long-horizon training efficiency by introducing optimizations like subprocess-based architecture and refined environment initialization routines. For reliability, we address critical issues such as memory leaks and flawed recursive implementations. A more detailed description of the engineering optimizations is shown in Appendix C.
this section cite: ['b48', 'b58', 'b46']

Section: Open-source availability and Visualization support.
AgentGym-RL provides a unified framework with consistent evaluating metrics and reproducible training pipelines. It also offers turnkey scripts that automate the workflow from environment setup to final assessment, enabling reliable replication. Additionally, an interactive graphical UI (See Figure 10 in Appendix C) supports visualization of step-by-step inspection and replay of full trajectories. Motivation. Inference-compute scaling in LLM reasoning shows that additional computation offers better performance (DeepSeek-AI et al., 2025;Jaech et al., 2024). However, given the interactive nature of agent tasks, we argue that effective progress requires expanding external interactions with the environment, not merely internal reasoning. To validate this, we investigate the impact of increasing the maximum number of interaction turns available to the agent, using several baseline models on Deep Search and SciWorld environments. As shown in Figure 5, all models show improvement as the number of interaction turns increases, demonstrating that long-horizon interaction and sufficient exploration contribute to enhanced agentic performance. However, the performance gains of the baseline models plateau as the number of interactions continues to grow, indicating their limited capability to solve complex tasks through long-horizon interactions. To address this limitation, we further explore leveraging RL to enhance agents' capabilities in longhorizon scenarios. Specifically, we vary the maximum number of interaction turns during RL rollouts and analyze the resulting training dynamics (Figure 4). We find that larger interaction horizons (e.g., 10 turns) enable deeper exploration but introduce training instability, often leading to training collapse, with the model exhibiting redundant interactions and unnecessary repetition. In contrast, shorter horizons provide stability but cap performance due to limited interaction turns. Therefore, our core motivation is how to scale interactions at train-time in a stable and effective way.
this section cite: ['b22']

Section: SCALINGINTER-RL: SCALING INTERACTIONS FOR LLM AGENTS
Method. To this end, we introduce ScalingInter-RL to stably optimize LLM agents for challenging tasks that require long-horizon interactions. The central idea of ScalingInter-RL lies in a progressive horizon-scaling strategy that gradually increases the number of interaction turns during RL training, as illustrated in Figure 9 (Appendix B).
Specifically, the objective is to maximize the expected final reward under a constrained interaction budget:
J(θ) = E τ ∼π θ [r (τ )] ,
where each trajectory τ = a T 0 , o 1 , a T 1 , . . . , a T K-1 , o K is sampled from the current policy π θ , with K representing the total number of interaction turns, T representing the reasoning path. To prevent the training collapse observed in the previously mentioned long-turn setting, we begin training with a short interaction horizon. By initially limiting the horizon, the agent focuses on exploitation, mastering fundamental task-solving skills through simpler tasks. This lays a solid foundation for stable training as the horizon gradually extends in later stages.
As training progresses, we introduce a monotonic schedule {h 1 < h 2 < • • • < h n }, where h t defines the maximum number of interaction turns allowed during phase t:
τ t ∼ π θ (τ | h t ) , subject to K t ≤ h t .
The horizon h t is updated every ∆ training steps according to a curriculum schedule:
h t+1 = h t + δ h ,
where δ h is an adaptive increment. As the horizon expands, the agent is encouraged to explore the environment more deeply, thereby enhances the ability to efficiently acquire and leverage information through more interactions. This staged scaling approach allows the agent to make more intelligent decisions, enabling deeper exploration of the environment, and ultimately results in more effective task completion while ensuring training stability.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETTINGS
Scenarios, Environments and Tasks. As mentioned before, we include five scenarios in AgentGym-RL. Specifically, we include WebArena (Zhou et al., 2024a) for web navigation, a RAGbased environment (Jin et al., 2025b;Joshi et al., 2017;Ho et al., 2020;Kwiatkowski et al., 2019;Mallen et al., 2022;Trivedi et al., 2022;Yang et al., 2018;Press et al., 2023) for deep search, TextCraft (Prasad et al., 2024) for digital games, BabyAI (Chevalier-Boisvert et al., 2019) for embodied tasks, and SciWorld (Wang et al., 2022) for scientific tasks.
Table 1: Evaluation results on Deep Search benchmark. For each group, the best result is in bold, and the second-best is underlined. SearchR1-it-v0.3 baseline uses Search-R1-v0.3 models (Jin et al., 2025a). See Appendix D for results of tasks on other scenarios.
this section cite: ['b27', 'b18', 'b30', 'b61', 'b82', 'b42', 'b41', 'b7', 'b63']

Section: Model NQ TriviaQA PopQA HotpotQA 2Wiki Musique Bamboogle Overall
Proprietary Models GPT-4o (Hurst et al., 2024) 20.0 70.0 30.0 30.0 32.0 10.0 34.0 26.8 Qwen-Max (Yang et al., 2024) 24.0 52.0 26.0 24.0 16.0 17.0 36.0 29.5 Gemini-2.5-Flash (Comanici et al., 2025) 8.0 60.0 30.0 24.0 16.0 8.0 34.0 23.5 OpenAI o4-mini (OpenAI, 2025) 22.0 68.0 50.0 38.0 44.0 28.0 62.0 42.5 OpenAI o3 (OpenAI, 2025) 28.0 70.0 56.0 46.0 64.0 29.0 74.0 49.5 Gemini-2.5-Pro (Comanici et al., 2025) 22.0 62.0 38.0 28.0 48.0 19.0 56.0 36.5
Open-sourced Models ≥ 100B Qwen3-235B-A22B (Yang et al., 2025a) 28.0 54.0 30.0 32.0 22.0 14.0 32.0 28.3 DeepSeek-V3-0324 (DeepSeek-AI et al., 2024) 28.0 60.0 24.0 28.0 18.0 11.0 34.0 26.5 DeepSeek-R1-0528 (DeepSeek-AI et al., 2025) 32.0 68.0 42.0 44.0 50.0 21.0 44.0 40.3
Open-sourced Models < 100B Qwen2.5-3B-Instruct (Yang et al., 2024) 8.0 42.0 22.0 14.0 8.0 2.0 10.0 13.5 Qwen2.5-7B-Instruct (Yang et al., 2024) 18.0 54.0 20.0 18.0 6.0 4.0 26.0 18.8 Qwen2.5-72B-Instruct (Yang et al., 2024) 22.0 52.0 24.0 28.0 24.0 12.0 38.0 26.5 Qwen3-4B (Yang et al., 2025a) 18.0 58.0 26.0 24.0 26.0 5.0 20.0 22.8 Qwen3-8B (Yang et al., 2025a) 26.0 44.0 26.0 22.0 32.0 10.0 32.0 25.3 Qwen3-32B (Yang et al., 2025a) 24.0 54.0 22.0 36.0 28.0 11.0 20.0 25.8 Llama-3.1-8B-Instruct (Dubey et al., 2024) 16.0 26.0 12.0 6.0 2.0 4.0 18.0 11.0 Llama-3.1-70B-Instruct (Dubey et al., 2024) 20.0 44.0 22.0 22.0 18.0 9.0 32.0 22.0 SearchR1-it-3B-v0.3GRPO(Jin et al., 2025b) 20.0 50.0 30.0 28.0 32.0 5.0 14.0 23.0 SearchR1-it-7B-v0.3GRPO(Jin et al., 2025b) 24.0 52.0 30.0 22.0 34.0 6.0 26.0 25.0 Our RL Models AgentGym-RL-3B 30.0 50.0 30.0 30.0 46.0 4.0 12.0 25.8 AgentGym-RL-7B 44.0 64.0 32.0 40.0 36.0 15.0 26.0 34.0 ScalingInter-7B 52.0 70.0 46.0 42.0 44.0 14.0 24.0 38.3
Baselines and backbone models. We leverage Qwen-2.5-3B and Qwen-2.5-7B (Yang et al., 2024) as our backbone models. Additionally, we introduce closed-source commercial models and strong open-source models as our baselines, as shown in Table 1. Both training and evaluation are conducted using ReAct (Yao et al., 2023) paradigm.
Detailed settings of each environment. Different environments have distinct observation spaces, action spaces, and reward structures. Due to space limitations, we provide detailed descriptions of the tools, APIs, and experimental settings for each environment in Appendix E.
this section cite: ['b21', 'b10', 'b10', 'b85']

Section: MAIN RESULTS
The main results are shown in Figure 1, and the detailed results on Deep Search are shown in Table 1. See Appendix D for detailed results of tasks on other scenarios.
Reinforcement learning generally improves agentic intelligence of open-source LLMs, bringing them on par with proprietary models. As shown in Figure 1, our RL model outperforms other open-source models by a large margin. It also leads in average success rate over closed-source models like GPT-4o and Gemini-2.5-Pro across five different scenarios. This demonstrates the effectiveness of our framework in enabling models to learn and make decisions in complex tasks, narrowing the gap between open-source and proprietary models ScalingInter-RL significantly and consistently boosts performance. We set phase transition points based on the total optimization steps in the RL process, rather than performing extensive hyperparameter tuning, as it has already proven effective. ScalingInter-RL consistently outperforms the baseline across various environments. For example, it improves WebArena performance by over 15 points, bringing it closer to closed-source commercial models. It also boosts TextCraft scores by nealy 50 points, achieving state-of-the-art results. These improvements show that our method effectively balances exploration and exploitation, enabling the model to interact more intelligently with the environment, adapt, and complete tasks.
Post-training and test-time compute show higher scaling potential than model size. As shown in Figure 1 (right), ScalingInter-RL with 7B parameters achieves an average success rate of 61.8%, significantly surpassing larger models like Llama3.1-70B (46.9%) and Qwen2.5-72B (42.8%). This shows that simply increasing model size provides limited performance gains, while increasing post-training and inference-time compute offers better results, providing new insights for future scaling strategies.
The environment plays a key role in the efficiency of reinforcement learning. The effectiveness of AgentGym-RL depends on the environment and the type of feedback provided. In simulated worlds with clear rules and direct cause-and-effect relationships, such as TextCraft, BabyAI, and SciWorld, RL achieves the greatest performance improvements. For instance, SciWorld's score jumps from 1.50% to 50.50%, a remarkable increase of almost 50 points. On the other hand, in more open-ended environments like WebArena and Deep Search, the performance gains from RL are more limited, due to the challenges of task complexity and potential noisy feedback. This provides valuable insights for the design of environmental feedback and reward structure in the future.
this section cite: []

Section: DISCUSSION

this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card Year: (2024)
Ref_id:b2 Title: Skyrl-v0: Train real-world long-horizon agents via reinforcement learning Year: (2025)
Ref_id:b3 Title: Fireact: Toward language agent fine-tuning Year: ()
Ref_id:b4 Title: Reinforcement learning for long-horizon interactive LLM agents Year: ()
Ref_id:b5 Title: Improving discriminative capability of reward models in RLHF using contrastive learning Year: (2024)
Ref_id:b6 Title: Agent-flan: Designing data and methods of effective agent tuning for large language models Year: (2024)
Ref_id:b7 Title: Babyai: A platform to study the sample efficiency of grounded language learning Year: (2019-05-06)
Ref_id:b8 Title: SELA: tree-search enhanced LLM agents for automated machine learning Year: ()
Ref_id:b9 Title: Inference-aware fine-tuning for best-of-n sampling in large language models Year: (2025)
Ref_id:b10 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b11 Title:  Year: ()
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: ()
Ref_id:b13 Title: The llama 3 herd of models Year: ()
Ref_id:b14 Title: Building open-ended embodied agents with internet-scale knowledge Year: (2022-12-09)
Ref_id:b15 Title: MASTER: A multi-agent system with LLM specialized MCTS Year: (2025-05-04)
Ref_id:b16 Title: Critiq: Mining data quality criteria from human preferences Year: ()
Ref_id:b17 Title: Skywork open reasoner 1 technical report Year: ()
Ref_id:b18 Title: Constructing A multihop QA dataset for comprehensive evaluation of reasoning steps Year: (2020)
Ref_id:b19 Title: Metagpt: Meta programming for A multi-agent collaborative framework Year: (2024)
Ref_id:b20 Title: REINFORCE++: A simple and efficient approach for aligning large language models Year: ()
Ref_id:b21 Title: Gpt-4o system card Year: (2024)
Ref_id:b22 Title: Openai o1 system card Year: (2024)
Ref_id:b23 Title: AI alignment: A comprehensive survey Year: ()
Ref_id:b24 Title: An empirical study on reinforcement learning for reasoning-search interleaved LLM agents Year: ()
Ref_id:b25 Title: Searchr1: Training llms to reason and leverage search engines with reinforcement learning Year: (2025)
Ref_id:b26 Title: Regularized best-of-n sampling to mitigate reward hacking for language model alignment Year: ()
Ref_id:b27 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017-07-30)
Ref_id:b28 Title: When can llms Actually correct their own mistakes? A critical survey of self-correction of llms Year: (2024)
Ref_id:b29 Title: Training language models to self-correct via reinforcement learning Year: (2025)
Ref_id:b30 Title: Natural questions: a benchmark for question answering research Year: (2019)
Ref_id:b31 Title: More agents is all you need Year: (2024)
Ref_id:b32 Title: Encouraging divergent thinking in large language models through multiagent debate Year: (2024)
Ref_id:b33 Title: Agentdog: A diagnostic guardrail framework for ai agent safety and security Year: (2026)
Ref_id:b34 Title: BOLAA: benchmarking and orchestrating llm-augmented autonomous agents Year: ()
Ref_id:b35 Title: When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories Year: ()
Ref_id:b36 Title: Kimi k2: Open agentic intelligence Year: (2025)
Ref_id:b37 Title: Long-horizon planning for multi-agent robots in partially observable environments Year: (2024)
Ref_id:b38 Title: GPT-4 technical report Year: (2023)
Ref_id:b39 Title: Training language models to follow instructions with human feedback Year: (2022-12-09)
Ref_id:b40 Title: Instruction tuning with GPT-4 Year: ()
Ref_id:b41 Title: Adapt: As-needed decomposition and planning with language models Year: (2024)
Ref_id:b42 Title: Measuring and narrowing the compositionality gap in language models Year: (2023)
Ref_id:b43 Title: Webrl: Training LLM web agents via selfevolving online curriculum reinforcement learning Year: (2025)
Ref_id:b44 Title: Tool learning with foundation models Year: ()
Ref_id:b45 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b46 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b47 Title: Self-reflection in LLM agents: Effects on problem-solving performance Year: ()
Ref_id:b48 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b49 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: ()
Ref_id:b50 Title: Thinking vs. doing: Agents that reason by scaling test-time interaction Year: ()
Ref_id:b51 Title: Sciagentgym: Benchmarking multi-step scientific tool-use in llm agents Year: (2026)
Ref_id:b52 Title: Reflexion: language agents with verbal reinforcement learning Year: (2023)
Ref_id:b53 Title: Alfworld: Aligning text and embodied environments for interactive learning Year: (2021)
Ref_id:b54 Title: Scaling LLM test-time compute optimally can be more effective than scaling model parameters Year: ()
Ref_id:b55 Title: Evaluating ai's ability to replicate AI research Year: ()
Ref_id:b56 Title: Adaplanner: Adaptive planning from feedback with language models Year: (2023)
Ref_id:b57 Title: gradient-methods-for-reinforcement-learning-with-function-approximatio Yashar Talebirad and Amirhossein Nadiri. Multi-agent collaboration: Harnessing the power of intelligent LLM agents Year: (1999-12-04)
Ref_id:b58 Title: Magicgui: A foundational mobile gui agent with scalable data pipeline and reinforcement fine-tuning Year: (2025)
Ref_id:b59 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b60 Title:  Year: ()
Ref_id:b61 Title: Musique: Multihop questions via single-hop question composition Year: (2022)
Ref_id:b62 Title: Reft: Reasoning with reinforced fine-tuning Year: (2024)
Ref_id:b63 Title: Scienceworld: Is your agent smarter than a 5th grader? Year: (2022)
Ref_id:b64 Title: Self-consistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b65 Title: RAGEN: understanding self-evolution in LLM agents via multi-turn reinforcement learning Year: ()
Ref_id:b66 Title: Browsecomp: A simple yet challenging benchmark for browsing agents Year: ()
Ref_id:b67 Title: Autogen: Enabling next-gen LLM applications via multi-agent conversation framework Year: (2025)
Ref_id:b68 Title: Training large language models for reasoning through reverse curriculum reinforcement learning Year: (2024)
Ref_id:b69 Title: Training large language models for reasoning through reverse curriculum reinforcement learning Year: (2024)
Ref_id:b70 Title: Enhancing LLM reasoning via critique models with test-time and training-time supervision Year: (2024)
Ref_id:b71 Title: The rise and potential of large language model based agents: a survey Year: ()
Ref_id:b72 Title: Agent-Gym: Evaluating and training large language model-based agents across diverse environments Year: (2025-07)
Ref_id:b73 Title: Critique-rl: Training language models for critiquing through two-stage reinforcement learning Year: (2025)
Ref_id:b74 Title: Agentprm: Process reward models for llm agents via step-wise promise and progress Year: (2025)
Ref_id:b75 Title: Inverseq*: Token level reinforcement learning for aligning large language models without preference data Year: (2024)
Ref_id:b76 Title: Self-evaluation guided beam search for reasoning Year: (2023)
Ref_id:b77 Title: Teaching language models to critique via reinforcement learning Year: ()
Ref_id:b78 Title: Qwen2.5 technical report Year: ()
Ref_id:b79 Title:  Year: ()
Ref_id:b80 Title:  Year: ()
Ref_id:b81 Title: Tao Gui, and Xipeng Qiu. Abcbench: Benchmarking agentic backend coding in real-world development Year: (2026)
Ref_id:b82 Title: Hotpotqa: A dataset for diverse, explainable multi-hop question answering Year: (2018-11-04)
Ref_id:b83 Title: Language Agents: From Next-Token Prediction to Digital Automation Year: (2024)
Ref_id:b84 Title: Towards scalable real-world web interaction with grounded language agents Year: (2022-12-09)
Ref_id:b85 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b86 Title: Tl-training: A task-feature-based framework for training large language models in tool use Year: ()
Ref_id:b87 Title: Toolhop: A query-driven benchmark for evaluating large language models in multi-hop tool use Year: ()
Ref_id:b88 Title: Toolhop: A query-driven benchmark for evaluating large language models in multi-hop tool use Year: (2025)
Ref_id:b89 Title: Feedback-driven tool-use improvements in large language models via automated build environments Year: (2025)
Ref_id:b90 Title: Agent-r: Training language model agents to reflect via iterative self-training Year: (2025)
Ref_id:b91 Title: Agenttuning: Enabling generalized agent abilities for llms Year: (2024)
Ref_id:b92 Title: Agentohana: Design unified data and training pipeline for effective agent learning Year: ()
Ref_id:b93 Title: Secrets of RLHF in large language models part I: PPO Year: ()
Ref_id:b94 Title: Webarena: A realistic web environment for building autonomous agents Year: (2024)
Ref_id:b95 Title: Archer: Training language model agents via hierarchical multi-turn RL Year: (2024)
Ref_id:b96 Title: Scaling test-time compute for LLM agents Year: ()
Ref_id:b97 Title: Deductive beam search: Decoding deducible rationale for chain-of-thought reasoning Year: (2024)
