Title: IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION FOR EFFECTIVE PLANNING AND TOOL USE
Abstract: Outcome-driven reinforcement learning has advanced reasoning in large language models (LLMs), but prevailing tool-augmented approaches train a single, monolithic policy that interleaves thoughts and tool calls under full context; this scales poorly with long horizons and diverse tools and generalizes weakly to new scenarios. Agentic systems offer a promising alternative by decomposing work across specialized modules, yet most remain training-free or rely on offline training decoupled from the live dynamics of multi-turn interaction. We introduce AGENT-FLOW, a trainable, in-the-flow agentic framework that coordinates four modules (planner, executor, verifier, generator) through an evolving memory and directly optimizes its planner inside the multi-turn loop. To train on-policy in live environments, we propose Flow-based Group Refined Policy Optimization (Flow-GRPO), which tackles long-horizon, sparse-reward credit assignment by converting multiturn optimization into a sequence of tractable single-turn policy updates. It broadcasts a single, verifiable trajectory-level outcome to every turn to align local planner decisions with global success and stabilizes learning with group-normalized advantages. Across ten benchmarks, AGENTFLOW with a 7B-scale backbone outperforms top-performing baselines with average accuracy gains of 14.9% on search, 14.0% on agentic, 14.5% on mathematical, and 4.1% on scientific tasks, even surpassing larger proprietary models like GPT-4o. Further analyses confirm the benefits of in-the-flow optimization, showing improved planning, enhanced tool-calling reliability, and positive scaling with model size and reasoning turns.

Section: INTRODUCTION
Recent advances in large language models (LLMs) have unlocked remarkable reasoning capabilities, largely driven by reinforcement learning (RL) from outcome-based feedback. By fine-tuning models to maximize verifiable rewards, LLMs like DeepSeek-R1 (Guo et al., 2025) and SimpleRL (Zeng et al., 2025b) have demonstrated sophisticated behaviors in self-correction and multi-step deduction.
A complementary line of work augments LLMs with external tools (e.g., web search, code execution) for knowledge retrieval and precise computation. Tool-integrated reasoning (TIR) extends reinforcement learning with verifiable rewards to learn when and how to call tools by interleaving reasoning (e.g., <think>) with tool invocations (e.g., <tool call>) under full context (Jin et al., 2025;Song et al., 2025;Chen et al., 2025;Feng et al., 2025). Early systems supported only a single tool type, whereas recent work enables multi-tool settings by encoding tool metadata into prompts (Dong et al., 2025;Qian et al., 2025a;Zhang et al., 2025). However, these methods still train a single, monolithic policy under multi-turn full-context reasoning, which introduces scaling challenges: (i) training becomes increasingly unstable as horizons lengthen, tool diversity grows, and environments shift with tool feedback (Wang et al., 2025c;Mai et al., 2025;Moonshot AI, 2025;Xue et al., 2025); and (ii) inference-time generalization remains brittle to unseen tasks or tools (Dong et al., 2025;Hu et al., 2025b).
Agentic systems (Wu et al., 2024;Hong et al., 2024;Hu et al., 2025b) offer a promising alternative to monolithic tool-integrated reasoning models. They consist of multiple modules-often distinct LLMs with prescribed roles (e.g., planner, critic) or specialized components with dedicated tools and capabilities (e.g., executor, coder)-that coordinate via shared memory and inter-module communication. By decomposing problems into sub-goals and iterating over multiple turns, these systems can tackle tasks that demand diverse tools, long horizons, or multi-stage reasoning. However, achieving robust coordination in such systems ultimately requires training, since handcrafted logic or static prompting cannot reliably capture when and how modules should collaborate, adapt to evolving tool outputs, or recover from early mistakes. At the same time, they introduce new training challenges: modules coordinate sequentially, outcome feedback propagates through long reasoning chains, and state distributions shift with evolving tool outputs. As a result, most systems remain training-free, relying on handcrafted logic or prompting heuristics. While some employ supervised fine-tuning or preference optimization for key modules (Motwani et al., 2024;Park et al., 2025), these off-policy approaches are decoupled from live dynamics and learn poorly from downstream successes or failures. Thus, agentic systems struggle with sparse rewards, brittle adaptation, and inefficient orchestration in dynamic environments.
To address the central challenge of learning long-horizon reasoning with sparse rewards in toolintegrated agentic systems, we introduce AGENTFLOW, a trainable framework for effective planning and tool use (Figure 2). AGENTFLOW comprises four specialized modules-planner, executor, verifier, and generator-that interact iteratively over multiple turns via a shared evolving memory and a toolset. The system operates in the flow, with each turn cycling through planning, execution, and verification. Unlike prior agentic systems, AGENTFLOW directly optimizes its planner on-policy, inside the live multi-turn loop, allowing it to dynamically adapt to trajectories shaped by tool calls, verifier signals, and memory updates. This evolving memory serves as a deterministic, structured record of the reasoning process, enabling transparent state tracking, controllable behavior, and bounded context growth.
To train the planner on-policy within this agentic system, we need to overcome the long-horizon credit assignment problem inherent to sparse, trajectory-level rewards. We introduce Flow-based Group Refined Policy Optimization (Flow-GRPO, Figure 4), an on-policy algorithm designed for this setting. Flow-GRPO operates on in-the-flow rollouts, which capture the full trajectory of states, actions, and tool events induced by the live system. Instead of attempting to assign credit with brittle, intermediate heuristics, we assign a single, verifiable final-outcome reward to the entire trajectory and broadcast it to every turn. This design effectively transforms the multi-turn reinforcement learning challenge into a series of single-turn updates: at each turn, the planner has access to the full memory context and receives a consistent reward signal aligned with global success. This approach, coupled with group-normalized advantages to stabilize training, enables robust credit assignment and allows the planner to learn effective long-horizon strategies from sparse feedback.
We evaluate AGENTFLOW on ten benchmarks across diverse reasoning domains, as results highlighted in Figure 1. AGENTFLOW substantially outperforms top-performing specialized tool-
(a) AgentFlow: In-the-Flow Agentic System Planner 𝑎 1 Executor Verifier Planner 𝑎 2 Executor Verifier Planner 𝑎 𝑇 Executor Verifier Generator o Query ... Turn T Turn 1 Turn 2 Answer Toolkit Set ... 𝑞 𝐾 𝑀 𝑡 𝑎 𝑡 Input: [Query Analysis] [Global Goal] [R equired Skills] Output: [Current Sub-Goal] [Selected Tool] [Context for Tool Use] Input: [Current Sub-Goal] [Selected Tool & Context] [Tool Metadata] Output: [Generated Command] [Execution Result] 𝑒 𝑡 𝑎 𝑡 𝐾 𝑣 𝑡 𝑞 𝑒 𝑡 𝑀 𝑡 𝑀 𝑡+1 Input: [Generated Command] [Execution Result] Output: [Execution Analysis] [Memory Analysis] [Verification Status] Planner Verifier Executor (b) In-the-Flow Rollout at Turn t Trained Frozen 𝜋 𝜃
this section cite: ['b54', 'b38', 'b1', 'b6', 'b4', 'b32', 'b55', 'b43', 'b26', 'b28', 'b46', 'b4', 'b12', 'b45', 'b10', 'b12', 'b29']

Section: Memory Memory
Figure 2: (a) Overview of AGENTFLOW, a trainable agentic system for in-the-flow planning and tool use. Four modules (planner, executor, verifier, generator) coordinate via a shared evolving memory M and toolset K, given a query q. The planner policy is optimized on-policy inside the system's multi-turn loop to enable adaptive, long-horizon reasoning. (b) A single state transition, showing the action a t , execution result e t , and verifier signal v t that update the memory from M t to M t+1 . integrated reasoning models and agentic systems, achieving average accuracy by 14.9% on knowledge-intensive search, 14.0% on broader agentic tasks, 14.5% on mathematical reasoning, and 4.1% on scientific reasoning ( §4.2). Notably, our 7B-backbone system even surpasses the ∼200Bparameter GPT-4o (Hurst et al., 2024) across all domains. Further analyses confirm that our inthe-flow optimization with Flow-GRPO is crucial, far surpassing offline supervised tuning ( §4.3). The trained planner learns to optimize planning, enhance tool-calling reliability, and discover effective solution pathways ( §4.5). Moreover, our training approach proves highly efficient, leading to increased rewards and condensed responses compared to traditional tool-integrated RL methods ( §4.6). Finally, we demonstrate that these benefits generalize, with consistent gains from scaling backbone size and turn budget ( §4.4).
Our work makes three key contributions: (1) We present AGENTFLOW, a trainable in-the-flow agentic system that directly optimizes its planner inside the multi-turn loop. By coordinating specialized modules through an evolving memory, it enables adaptive long-horizon planning and robust tool orchestration. (2) We introduce Flow-GRPO, an on-policy, outcome-driven algorithm that hat converts multi-turn RL into a sequence of tractable single-turn policy updates by broadcasting a single, verifiable final-outcome reward to every turn. (3) Through comprehensive experiments on ten benchmarks, we show that AGENTFLOW with a 7B backbone outperforms specialized baselines and even larger proprietary models. Further analyses reveal improved planning, enhanced tool-calling reliability, and positive scaling with model size and turn budgets.
this section cite: ['b14']

Section: PRELIMINARY
Reinforcement learning for reasoning LLMs. Recent progress in reasoning LLMs has been significantly driven by reinforcement learning from outcome feedback, using a verifiable reward signal (Shao et al., 2024;Yu et al., 2025). This paradigm fine-tunes a language model to maximize an outcome-based reward while remaining close to a reference policy. Formally, the objective is to optimize a policy LLM π θ to generate a response o for a given query q from dataset D: max
π θ E q∼D, o∼π θ (•|q) R(q, o) -β D KL (π θ (o | q) ∥ π ref (o | q)) ,(1)
where R(q, o) is the outcome-based reward, π ref is a reference model to prevent policy collapse, and β controls KL regularization. Algorithms like Group Relative Policy Optimization (GRPO) (Shao et al., 2024) implement this by sampling groups of responses, normalizing advantages by their rewards, and updating the policy with a clipped objective to encourage high-reward outputs.
this section cite: ['b37', 'b52', 'b37']

Section: Tool-integrated reasoning models (LLM agents).
LLMs can be augmented with external tools to access knowledge and perform precise computation under reinforcement learning with outcomebased reward. As shown in Figure 3(a), the LLM interleaves reasoning and tool calls, producing a chain of thought within <think></think> tokens followed by tool invocations (e.g., <tool call></tool call>). The resulting trajectory τ is a sequence of model generations and tool observations: τ = {s 1 , a 1 , e 1 , . . . , s T , a T }, where s t denotes the context, a t the generated action (thought + tool call), and e t the tool's execution result. The policy model π θ is then trained to maximize a final outcome reward. Prior work has explored single-and multi-tool settings for search and code execution (Jin et al., 2025;Chen et al., 2025;Feng et al., 2025;Qian et al., 2025a). Agentic systems with tool usage. An alternative approach is the use of agentic systems (Wu et al., 2024;Hong et al., 2024;Lu et al., 2025). As shown in Figure 3(b), these frameworks deploy multiple specialized modules-often distinct LLMs with carefully designed prompts and roles-within a collaborative workflow. By decomposing tasks and assigning subproblems to modules with dedicated tools and capabilities (e.g., planner, coder, critic), they can address complex problems such as web browsing, document processing, and multi-stage programming that exceed the scope of a single model. A central limitation, however, is that these systems are typically training-free: modules remain frozen pre-trained models orchestrated by handcrafted logic or prompting heuristics.
this section cite: ['b1', 'b6', 'b32', 'b45', 'b10', 'b23']

Section: IN-THE-FLOW AGENTIC SYSTEM OPTIMIZATION
We aim to bridge the gap between trainable but monolithic reasoning models and flexible yet static agentic systems. We present AGENTFLOW, a flexible and trainable agentic system that integrates four specialized modules with an evolving memory ( §3.1). Unlike prior agentic systems, AGENT-FLOW directly optimizes the planner within the multi-turn loop of an agentic system ( §3.2).
this section cite: []

Section: AGENTFLOW: AN IN-THE-FLOW AGENTIC SYSTEM
We propose AGENTFLOW, a general-purpose tool-integrated agentic framework for solving complex reasoning tasks through fine-grained planning and effective tool use within a multi-turn architecture. As shown in Figure 2, the framework comprises four specialized modules-Action Planner P, Tool Executor E, Execution Verifier V, and Solution Generator G-coordinated by a shared evolving memory M and a toolset K. These modules interact sequentially and iteratively to perform action planning, tool execution, context verification, and solution generation, thereby enabling tool-integrated reasoning across multiple turns.
We formalize AGENTFLOW's problem-solving process as a multi-turn Markov Decision Process (MDP). Given a query q and a toolset K, the system proceeds for a variable number of turns. Let M t denote the memory state before turn t (with M 1 initialized from q). At turn t, the planner P (a trainable policy π θ ) formulates a sub-goal, selects an appropriate tool k ∈ K, and retrieves relevant context from memory, producing an action: a t ∼ π θ (a t | q, K, M t ).
The executor E invokes the chosen tool with context, yielding an execution observation e t ∼ E(e t | a t , K). The verifier V then evaluates whether e t is valid and whether the accumulated memory is sufficient to solve the query, producing a binary verification signal v t ∼ V(v t | q, e t , M t ). If v t = 0, the memory is updated deterministically to incorporate new evidence: M t+1 = f mem (M t , a t , e t , v t ), where f mem (•) denotes the memory-update function, which records agent-process information in a concise, structured form along with contextual details such as time, turn index, and error signals.
The process repeats until v t = 1 (termination) or a predefined maximum turn budget is reached. Upon termination at turn T , the solution generator G produces the final solution o, conditioned on the query and the accumulated memory: o ∼ G(o | q, M T ).
this section cite: []

Section: Multi-turn Group Computation
Frozen Models
Trained Models Reward Model Multi-turn Agentic System Rollouts Policy Model Reference Model
Flow-GRPO This formulation decomposes multi-turn, tool-integrated reasoning into structured, observable transitions. After T turns, the trajectory τ = {(a t , e t , v t )} T t=1 records the history of planning, execution, and verification. The joint generative process can be written as
p θ {a t , e t , v t } T t=1 , o | q, K = T t=1 π θ (a t | q, K, M t ) E(e t | a t , K) V(v t | q, e t , M t ) G(o | q, M T ),(2)
where {a t , e t , v t } T t=1 are explicit realizations of the latent reasoning chain. Importantly, unlike latent thoughts behind trajectories, our memory M is an explicit and deterministic record of the reasoning process, ensuring transparency and controllability of multi-turn decisions.
this section cite: []

Section: IN-THE-FLOW REINFORCEMENT LEARNING OPTIMIZATION
We target tool-integrated agentic systems operating under long-horizon tasks with sparse rewards. In this setting, the Action Planner (the trainable policy of AGENTFLOW) selects a sequence of interdependent actions while the state (q, K, M t ) evolves with tool results and verifier feedback. Conventional offline training-e.g., supervised fine-tuning or preference fine-tuning on curated traces-optimizes the planner outside the active loop (Motwani et al., 2024;Park et al., 2025). This decoupling prevents real-time coordination with the executor, verifier, and solution generator, induces distribution shift between training and deployment, and provides limited guidance about which intermediate decisions truly matter. As a result, planners often adapt poorly to multi-turn dynamics; early errors cascade, and post-hoc fixes are brittle.
this section cite: ['b29']

Section: In-the-flow learning.
To address these issues, we optimize the planner in the flow of execution. We roll out the full AGENTFLOW system under the current policy, collect the actual trajectory τ of states, actions, and tool events it induces, and update the policy within the agentic system using a verifiable final-outcome signal. This exposes the multi-turn credit-assignment problem directly and trains the planner on the exact states it will face at inference. Our objective, Flow-GRPO, is designed to stabilize learning under sparse, trajectory-level rewards over multiple turns.
As established in §3.1, rollouts in AGENTFLOW define a finite-horizon MDP with a variable horizon T . At turn t, the planner observes the state (q, K, M t ), selects an action a t , the executor and verifier return (e t , v t ), and the memory updates deterministically to M t+1 .
this section cite: []

Section: Policy optimization objective.
The planner policy π θ is trained to maximize the expected return over on-policy rollouts. Let R(τ ) be the reward for a complete trajectory τ . The objective is:
J (θ) = E τ ∼π θ R(τ ) , θ ⋆ = arg max θ J (θ),(3)
where a rollout τ is the sequence of decisions {a t } T t=1 generated on-policy by π θ . Final-outcome reward. Assigning credit to intermediate actions is challenging because each a t influences the final solution only indirectly, and their value may only emerge after several turns (e.g., error or improvement accumulation). To avoid brittle local feedback, we adopt a final-outcomebased reward: every action within a rollout receives the same global reward signal, based on the correctness of the final solution o with respect to query q and ground truth y * : r = R(a t ) = R(o, q, y * ), ∀t = 1, . . . , T,
where R(o, q, y * ) ∈ {0, 1} is assigned by an LLM-as-judge rubric for semantic, numeric, and option-level equivalence (see §E.3). This propagates a trajectory-level success signal back through the reasoning chain, aligning every decision a t with global correctness.
Objective function. We formalize Flow-based Group Refined Policy Optimization for the planner. The goal is to optimize the policy π θ by maximizing the expected return over a group of parallel rollouts. For each query-label pair from training corpus (q, y * ) ∼ D, we sample a group of G onpolicy trajectories {τ i } G i=1 by running the current behavior policy π θold inside AGENTFLOW, where τ i = {a 1 i , ....a Ti i , o i }. Let s t i = (q, K, M t i ) be the state at turn t of rollout i, a t i the planner's action (a token sequence of length |a t i |), and o i the final response. This structure is key to addressing the long-horizon credit assignment challenge: by broadcasting a single trajectory-level reward to all turns, we effectively decompose the multi-turn RL problem into a set of independent, single-turn policy updates; we provide a formal proof of this equivalence and analyze its convergence properties in §B. Each update for an action a t i is conditioned on the full historical context encapsulated in the state s t i and receives the same global success signal, simplifying optimization. The objective is
JFlow-GRPO(θ) = E (q,y * )∼D, {τ i } G i=1 ∼π θ old 1 G G i=1 1 Ti T i t=1 1 |a t i | |a t i | j=1 min ρ t i,j A t i , clip(ρ t i,j , 1 -ϵ, 1 + ϵ) A t i -β DKL π θ ∥ πref ,(5)
where T i is the (variable) number of turns in rollout i, and
ρ t i,j = π θ a t i,j s t i , a t i,1:j-1 π θold a t i,j s t i , a t i,1:j-1(6)
is the token-level importance ratio for the j-th token of a t i , ϵ > 0 is the PPO clipping parameter, and β > 0 controls the KL penalty to a fixed reference policy π ref .
Group-normalized advantages. Because the reward in Eq. 4 is a single trajectory-level signal, the per-turn advantage A t i is constant over t within a rollout i. We reduce variance and sharpen credit assignment across the group by using a group-normalized advantage:
A t i = R(o i , q, y * ) -mean { R(o k , q, y * )} G k=1 std { R(o k , q, y * )} G k=1 .(7)
4 EXPERIMENTS
this section cite: []

Section: EXPERIMENTAL SETUP
In our main experiments, all modules-Action Planner, Tool Executor, Executive Verifier, and Solution Generator-are instantiated with the Qwen2.5-7B-Instruct model (Yang et al., 2024a). Among these, only the Action Planner is trainable. The system operates with five interactive tools: Base Generator is an instance of Qwen2.5-7B-Instruct that acts as the default reasoning engine if the planner decides not to use an external tool; Python Coder generates and executes Python code given a query and returns the execution result; Google Search searches the web and returns a summarization of Top-K search results; Wikipedia Search searches articles matching a given query and returns a summarization; and Web Search returns summarized information from a given web page. During the RL fine-tuning phase, we mix data from Search-R1 (Jin et al., 2025) and DeepMath (He et al., 2025) as training data, which provides paired question-answer examples across search and mathematical domains. We use a batch size of 32 with 8 rollouts per sample.
To comprehensively evaluate tool-use capabilities of AGENTFLOW, we conduct experiments on four types of reasoning tasks: (1) Knowledge-intensive search including Bamboogle (Press et al., 2023), 2Wiki (Ho et al., 2020), HotpotQA (Yang et al., 2018), and Musique (Trivedi et al., 2022); (2) Agentic reasoning such as GAIA (Mialon et al., 2023) (where we adopt the textual split); (3) Logic-dense mathematical reasoning including AIME2024 (Art of Problem Solving, 2025), AMC23 (MAA, 2023), and GameOf24 (Lightman et al., 2023); and (4) Scientific reasoning including GPQA (Rein et al., 2024) and MedQA (Yang et al., 2024c). To mitigate randomness, we report the average accuracy across three trials for all experiments. More experimental details are in §C. Table 2: Accuracy comparison of mathematical and scientific reasoning tasks.
this section cite: ['b48', 'b8', 'b31', 'b9', 'b51', 'b40', 'b27', 'b34', 'b50']

Section: MAIN RESULTS
Baselines. As presented in Tables 1 and 2, we include five categories of baselines: (1) Opensource LLMs: Qwen2.5 (Yang et al., 2024a), Llama-3.1, and Llama- 3.3 (Dubey et al., 2024)
this section cite: ['b48']

Section: Key insights.
AGENTFLOW consistently outperforms all baseline models by large margins. Compared to the best-performing 7B models without tool integration, AGENTFLOW achieves absolute gains of 40.7% on search (SFT), 29.9% on agentic reasoning (SFT), 15.0% on math (SimpleRLreason), and 8.0% on scientific tasks (Luffy). Against specialized tool-integrated systems, AGENT-FLOW surpasses the top models by 14.9% in search (AutoGen), 14.0% in agentic reasoning (Search-R1), 14.5% in math (ToRL), and 4.1% in science (TIR). Notably, our 7B-backbone AGENTFLOW even outperforms the ∼200B-parameter GPT-4o across all domains, with gains ranging from 8.2% to 18.0%. A detailed analysis is provided in §D.1.
Question: Compute the check digit the Tropicos ID for the Order Helotiales would have if it were an ISBN-10 number.
this section cite: []

Section: Ground truth answer: 3
AgentFlow (before Flow-GRPO Fine-tuning)
• Selected tool: Wikipedia Search • Sub-goal: Retrieve Tropicos ID of Order Helotiales from Wikipedia.
• Tool call: query="Tropicos ID for the Order Helotiales"
• Result: No results found for query: Tropicos ID for Order Helotiales.
• Selected tool: Google Search • Sub-goal: Retrieve the Tropicos ID of the Order.
• Tool call: query="Tropicos ID for the Order Helotiales"
• Result: The Tropicos ID for the Order Helotiales is 100370510.
• Selected tool: Python Coder • Sub-goal: Calculate the check digit for the Tropicos ID 100370510 if it were an ISBN-10 number.
• Tool call: query="Calculate check digit for ISBN-10 number 100370510"
• Result: name 'isbn' is not defined.
Action steps 3 to 9 have similar sub-goals, commands, and results; the agent is stuck in error loops and unable fix the naming issue.
Solution: The Python tool failed repeatedly with the same 'isbn' variable error). The agent failed to calculate the check digit.
• Selected tool: Python Coder • Sub-goal: Write and execute a Python script to calculate the check digit for `tropicos_id` as if it were an ISBN-10 number.
• Tool call: query="Calculate the check digit for the ISBN-10 number" + tropicos_id • Result: Error in execute_command: name 'tropicos_id' is not defined.
• Selected tool: Python Coder • Sub-goal: Write and execute a Python script to calculate the check digit for `tropicos_id` as if it were an ISBN-10 number.
• Tool call: query="Compute the check digit for 100370510 as an ISBN-10 number"
• Result: 3 (printed output).
def calculate_check_digit(isbn): isbn_digits = [int(digit) for digit in isbn[:9]] total_sum = sum(position * digit for position, digit in enumerate(isbn_digits, start=1)) check_digit = total_sum % 11 if check_digit == 10: return 'X' else:
return str(check_digit) result = calculate_check_digit("100370510") print(f"The check digit is {result}") Solution: The check digit is 3, resulting in the full number 1003705103.
• Selected tool: Wikipedia Search (Truncated...) • Result: No results found.
• Selected tool: Google Search (Truncated...) • Result: Tropicos ID: 100370510.
AgentFlow (after Flow-GRPO Fine-tuning)
1 2 3 4 1 2 3 4-9 ... ...
this section cite: []

Section: TRAINING STRATEGIES ON THE PLANNER
We conduct an ablation study to analyze the impact of different training strategies for the Action Planner module in AGENTFLOW, with results reported in Table 3. The executor, verifier, and generator modules remain fixed as Qwen2.5-7B-Instruct, consistent with our main setup ( §4.1).
this section cite: []

Section: Planner Model Training
Bamboogle 2Wiki GAIA AIME24 AMC23 GameOf24 Avg.
Qwen-2.
5-7B Frozen 58.4 60.0 17.2 16.7 47.4 31.0 38.5 GPT-4o Frozen 65.0 ↑ 6.6 70.0 ↑ 10.0 23.6 ↑ 6.4 16.7 ↑ 0.0 48.7 ↑ 1.3 42.0 ↑ 11.0 44.3 ↑ 5.8 Qwen-2.5-7B SFT 30.4 ↓ 28.0 32.7 ↓ 27.3 6.3 ↓ 10.9 3.3 ↓ 13.4 37.5 ↓ 9.9 7.0 ↓ 24.0 19.5 ↓ 19.0 Qwen-2.5-7B Flow-GRPO 69.6 ↑ 11.2 77.2 ↑ 17.2 33.1 ↑ 15.9 40.0 ↑ 23.3 61.5 ↑ 14.1 53.0 ↑ 22.0 55.7 ↑ 17.2
Table 3: Performance comparison of AGENTFLOW across different training methods.
A more capable planner is beneficial, but has limits. Replacing the frozen Qwen2.5-7B-Instruct baseline with a stronger proprietary model, GPT-4o, yields only a modest 5.8% average gain. This indicates a key bottleneck that, while a more powerful model improves planning, its static nature prevents co-adaptation with the live dynamics of AGENTFLOW.
Offline SFT leads to performance collapse, while in-the-flow RL is crucial. The limitations of a static planner are further exposed when distilling GPT-4o's behavior via offline supervised finetuning (SFT) on its trajectories as Action Planner in AGENTFLOW. This results in a catastrophic performance collapse, with an average accuracy drop of 19.0% compared to the frozen baseline. This failure arises from the token-level imitation objective of SFT, which misaligns with trajectorylevel task success and prevents the planner from adapting to dynamic tool feedback or recovering from compounding errors. In contrast, training the planner with our on-policy Flow-GRPO method proves highly effective: by optimizing for the final outcome, the planner learns to handle longhorizon workflows, achieving a 17.2% average gain over the frozen baseline.
this section cite: []

Section: SCALING TRENDS IN AGENTFLOW
Training scaling in backbone size. We study how backbone LLM scale affects AGENTFLOW's performance and the efficacy of Flow-GRPO. We build two versions of the system: one using Qwen2.5-3B-Instruct and another using Qwen2.5-7B-Instruct for all four modules (planner, executor, verifier, and generator) and tools. In both, only the planner is fine-tuned with Flow-GRPO. As shown in Figure 6, Flow-GRPO fine-tuning consistently improves performance across tasks for both backbones. This demonstrates that our in-the-flow optimization is effective across model capacities, enhancing AGENTFLOW regardless of LLM size.  Turns (T max ) 3  5  7  10   2Wiki  2.22 3.18 3.81 4.44  GameOf24  1.63 2.12 2.36 2.67  AIME24  1.63 1.63 1.86 1.90  GAIA  2.43 3.46 4.28 5.42Table 4: Average turns with increased T max .
Inference scaling in turn budgets. We investigate how the maximum allowed turns (T max ) affect reasoning depth and final performance of AGENT-FLOW during test-time inference with the Qwen2.5-7B-Instruct backbone. As shown in Figure 7, increasing T max from 3 to 10 consistently improves outcomes across all tasks, accompanied by a rise in average turns consumed. On knowledge-intensive benchmarks such as 2Wiki and GAIA, a larger turn budget enables AGENTFLOW for deeper information retrieval. On mathematical benchmarks like GameOf24 and AIME24, it supports decomposed sub-goals, alternative strategies, and refinement of errors. Final performance peaks at T max = 10 for all tasks, confirming that a longer reasoning horizon benefits the system without causing degenerate loops. This validates that AGENTFLOW adapts its turn allocation to problem complexity to achieve better solutions through iterative refinement.
this section cite: []

Section: IN-DEPTH ANALYSIS OF OPTIMIZED PLANNING

this section cite: []

Section: 28.5
Acc:60.0%
28.8 70.5 13.6 4.0 28.7 66.2 6.3 10.9 19.5 59.8 Acc: 77.2% (+17.2%) 36.0 +42.0 -22.4 -24.8 After Fine-tuning Acc: 76.0% Acc: 80.0% (+4.0%) After Fine-tuning +59.8 +19.5 -22.4 -55.3 (a) 2Wiki (b) MedQA Base Generator Google Search Web Search Wikipedia Search
this section cite: []

Section: Flow-GRPO optimizes tool usage.
We compare tool usage distributions before and after in-the-flow RL training. Figure 8 shows results on two knowledge-intensive tasks, 2Wiki and MedQA, which exhibit distinct optimization patterns alongside improved task accuracy. For 2Wiki, which requires broad factual knowledge, Flow-GRPO optimizes the planner to increase Google Search usage by 42.0%. In contrast, for the specialized MedQA benchmark, which requires deep, domain-specific information retrieval, finetuning shifts the planner away from general tools, reducing Google Search calls (66.2→10.9%) in favor of in-document Web Search (0→19.5%) and specialized Wikipedia Search (0→59.8%). This demonstrates that the planner learns to select task-appropriate tools.
Flow-GRPO incentivizes autonomous discovery of new solutions. We further examine qualitative examples in Figure 5 and additional cases in §F. These cases show that AGENTFLOW, trained with Flow-GRPO, develops enhanced capabilities for task planning and tool use. The planner exhibits adaptive efficiency, stronger self-correction, and spontaneous new integration of tools throughout step-by-step problem-solving, autonomously discovering effective solution pathways.
this section cite: []

Section: TRAINING EFFICIENCY ANALYSIS
Optimized planning with increased rewards and condensed responses. We analyze the training dynamics of the AGENTFLOW planner by tracking its average reward and response length on the train set (Figure 9a). Training rewards steadily increase, indicating effective policy improvement via Flow-GRPO. Meanwhile, response length, after an initial exploratory rise, progressively shortens and stabilizes. This shows the planner learns to balance conciseness and informativeness, avoiding unnecessarily long outputs. Flow-GRPO efficiency over toolintegrated reasoning RL. We compare AGENTFLOW (trained with Flow-GRPO) against a monolithic tool-integrated reasoning baseline (ToRL) on AIME24. As shown in Figure 9b, AGENTFLOW achieves sustained performance gains, with validation accuracy growing steadily. In contrast, ToRL's performance quickly stagnates and trends downwards, highlighting the superior efficiency of our agentic training approach, which uses decomposition and stable credit assignment to avoid the instability.
this section cite: []

Section: RELATED WORK
Reinforcement learning (RL) from outcome-based rewards has become a dominant paradigm for training LLMs to use external tools. Much of this work trains a single, monolithic policy to interleave reasoning with tool calls. This strategy has proven effective in specialized, single-tool settings (Mai et al., 2025;Xue et al., 2025;Feng et al., 2025;Li et al., 2025b) and web search for knowledge-intensive questions (Chen et al., 2025;Jin et al., 2025;Song et al., 2025;Li et al., 2025a;Sun et al., 2025). Recent efforts have extended this monolithic framework to multi-tool environments by focusing on data synthesis (Dong et al., 2025), unified training infrastructure (Jiang et al., 2025), and principled reward design (Qian et al., 2025a;Zhang et al., 2025). However, these approach scales poorly as task complexity and planning horizons grow. The central challenge is long-horizon credit assignment; attributing a final outcome to specific intermediate tool calls remains difficult, even with fine-grained, turn-level rewards (Zeng et al., 2025a;Wang et al., 2025d). This difficulty leads to training instability and brittle inference-time generalization, manifesting as strategic deficiencies like tool overuse or "cognitive offloading" (Wang et al., 2025b;Qian et al., 2025b), suboptimal personalization (Cheng et al., 2025), and poor alignment with user preferences for tool invocation (Huang et al., 2025).
Agentic systems with tool use. Agentic systems offer an alternative to monolithic models by decomposing tasks across specialized modules. Many such systems are training-free, orchestrating pre-trained LLMs with handcrafted logic and prompting, as seen in frameworks like AutoGen (Wu et al., 2024), MetaGPT (Hong et al., 2024), and OctoTools (Lu et al., 2025). This static approach, however, limits their ability to learn and adapt collaborative strategies from experience. Recognizing this, recent work explores training these systems to improve coordination (Deng et al., 2025;Liao et al., 2025). However, most training paradigms are offline, relying on supervised fine-tuning or preference optimization on static datasets (Motwani et al., 2024;Park et al., 2025). These methods are decoupled from the live, multi-turn dynamics of the system, preventing modules from learning to adapt to evolving tool outputs or recover from early mistakes. Training directly in the flow with on-policy RL is difficult due to sparse rewards and long-horizon credit assignment, where feedback is delayed across long reasoning chains and shifting state distributions (Wang et al., 2025c). Consequently, these systems often suffer from brittle adaptation and require complex reward shaping to learn effectively (Wang et al., 2025a).
this section cite: ['b26', 'b46', 'b6', 'b19', 'b1', 'b38', 'b4', 'b15', 'b32', 'b55', 'b44', 'b33', 'b2', 'b13', 'b45', 'b10', 'b23', 'b3', 'b20', 'b29', 'b43', 'b32']

Section: CONCLUSION
We presented AGENTFLOW, a trainable, in-the-flow agentic system that coordinates four specialized modules via an evolving memory and optimizes its planner directly inside the multi-turn loop. To enable stable on-policy learning under long-horizon, sparse-reward settings, we introduced Flow-GRPO, which converts multi-turn RL into a sequence of tractable single-turn policy updates by broadcasting a single, verifiable trajectory-level outcome to every turn and stabilizing credit assignment with group-normalized advantages. Comprehensive experiments show that AGENTFLOW achieves strong cross-domain performance, surpassing specialized baselines and even larger proprietary models. In-depth analyses confirm improved planning and tool-calling reliability, along with positive scaling trends in model size and allowed turn budgets.
TABLE OF CONTENTS for each query-label pair (q, y * ) ∼ D do
3: 1. IN-THE-FLOW ROLLOUT GENERATION 4: Initialize: t ← 1, M t ← q 5: repeat 6: a t ∼ π θ (a t | q, K, M t ) {Plan Action} 7: e t ∼ E(e t | a t , K) {Execute Action} 8: v t ∼ V(v t | q, e t , M t ) {Verify Result} 9: M t+1 = f mem (M t , a t , e t , v t ) {Update Memory} 10: t ← t + 1 11: until termination condition met 12: o ∼ G(o | q, M T ) {Generate Final Solution} 13: 2. REWARD COMPUTATION 14: R(a t ) = R(o, q, y * ), ∀t = 1, . . . , T 15: 3. POLICY UPDATE 16:
Update the Action Planner policy π θ by maximizing the Flow-GRPO objective (Eq. 5)
17:
end for 18: end for 19: return optimized parameters θ ⋆
this section cite: []

Section: References
Ref_id:b0 Title: Aime problems and solutions Year: (2025)
Ref_id:b1 Title: ReSearch: Learning to reason with search for llms via reinforcement learning Year: (1920)
Ref_id:b2 Title: ToolSpectrum: Towards personalized tool utilization for large language models Year: (2025)
Ref_id:b3 Title: Pe-ma: Parameter-efficient co-evolution of multi-agent systems Year: (2025)
Ref_id:b4 Title: Tool-star: Empowering llm-brained multi-tool reasoner via reinforcement learning Year: (2025)
Ref_id:b5 Title: The llama 3 herd of models Year: (2024)
Ref_id:b6 Title: Retool: Reinforcement learning for strategic tool use in llms Year: (2025)
Ref_id:b7 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b8 Title: Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning Year: (2025)
Ref_id:b9 Title: Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps Year: (2020)
Ref_id:b10 Title: MetaGPT: Meta programming for a multi-agent collaborative framework Year: (2024)
Ref_id:b11 Title: Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model Year: (1920)
Ref_id:b12 Title: Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation Year: (2025-02)
Ref_id:b13 Title: TTPA: Tokenlevel tool-use preference alignment training framework with fine-grained evaluation Year: (2025)
Ref_id:b14 Title: GPT-4o system card Year: (2024)
Ref_id:b15 Title: Towards holistic agentic reinforcement learning with tool use Year: (2025)
Ref_id:b16 Title: Search-R1: Training llms to reason and leverage search engines with reinforcement learning Year: (1920)
Ref_id:b17 Title: What disease does this patient have? a large-scale open domain question answering dataset from medical exams Year: (2021)
Ref_id:b18 Title: Search-o1: Agentic search-enhanced large reasoning models Year: ()
Ref_id:b19 Title: ToRL: Scaling tool-integrated rl Year: (1920)
Ref_id:b20 Title: Multi-agent reinforcement finetuning Year: (2025)
Ref_id:b21 Title: Let's verify step by step Year: ()
Ref_id:b22 Title: Math twenty four (24s game) dataset Year: ()
Ref_id:b23 Title: OctoTools: An agentic framework with extensible tools for complex reasoning Year: (2025)
Ref_id:b24 Title: General-reasoner: Advancing llm reasoning across all domains Year: (2025)
Ref_id:b25 Title: American mathematics competitions Year: (2023)
Ref_id:b26 Title: Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving Year: (2025)
Ref_id:b27 Title: Gaia: a benchmark for general ai assistants Year: (2023)
Ref_id:b28 Title: End-to-End RL Training for Emerging Agentic Capabilities Year: (2002)
Ref_id:b29 Title: Malt: Improving reasoning with multi-agent llm training Year: (2024)
Ref_id:b30 Title: Multi-agent post-co-training for collaborative large language models with reinforcement learning Year: ()
Ref_id:b31 Title: Measuring and narrowing the compositionality gap in language models Year: (2023)
Ref_id:b32 Title: ToolRL: Reward is all tool learning needs Year: (2025-02)
Ref_id:b33 Title: SMART: Self-aware agent for tool overuse mitigation Year: (2010)
Ref_id:b34 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b35 Title: Trust region policy optimization Year: (2015)
Ref_id:b36 Title: Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy Year: (2023)
Ref_id:b37 Title: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b38 Title: R1-searcher: Incentivizing the search capability in llms via reinforcement learning Year: (2025)
Ref_id:b39 Title: Incentivize the search capability of llms without searching Year: (2025)
Ref_id:b40 Title: Musique: Multihop questions via single-hop question composition Year: (2022)
Ref_id:b41 Title: SPA-RL: Reinforcing llm agents via stepwise progress attribution Year: ()
Ref_id:b42 Title: Acting less is reasoning more! teaching model to act efficiently Year: (2025)
Ref_id:b43 Title: RAGEN: Understanding self-evolution in llm agents via multi-turn reinforcement learning Year: (2025-02-10)
Ref_id:b44 Title: Igniting llms search ability via step-wise proximal policy optimization Year: (1920)
Ref_id:b45 Title: Autogen: Enabling next-gen llm applications via multiagent conversations Year: (1920)
Ref_id:b46 Title: Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning Year: (2025)
Ref_id:b47 Title: Learning to reason under off-policy guidance Year: (2025)
Ref_id:b48 Title: Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report Year: (2024)
Ref_id:b49 Title: Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement Year: (1920)
Ref_id:b50 Title: Llm-medqa: Enhancing medical question answering through case studies in large language models Year: (2024-06)
Ref_id:b51 Title: HotpotQA: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b52 Title: Dapo: An open-source llm reinforcement learning system at scale Year: (2025)
Ref_id:b53 Title: Reinforcing multi-turn reasoning in llm agents via turn-level credit assignment Year: ()
Ref_id:b54 Title: Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild Year: (2025)
Ref_id:b55 Title: Nemotron-research-tool-n1: Tool-using language models with reinforced reasoning Year: (2025)
Ref_id:b56 Title: SOLUTION GENERATOR Instruction for Solution Generator Task: Generate a concise final answer to the query based on all provided context. Context: Query: {Question} Initial Analysis: {Query Analysis} Actions Taken Year: ()
Ref_id:b57 Title: Carefully review the original user query, the initial analysis, and the complete sequence of actions and their results Year: ()
Ref_id:b58 Title: Synthesize the key findings from the action history into a coherent narrative Year: ()
Ref_id:b59 Title: Construct a clear, step-by-step summary that explains how each action contributed to solving the query Year: ()
Ref_id:b60 Title: Provide a direct, precise, and standalone final answer to the original query Year: ()
Ref_id:b61 Title: Process Summary: A clear, step-by-step breakdown of how the query was addressed. For each action Year: ()
Ref_id:b62 Title: Answer: A direct and concise final answer to the query. This should be a self-contained statement that fully resolves the user's question Year: ()
Ref_id:b63 Title: 2 PYTHON CODER Tool Metadata of Python Coder Description: A tool that generates and executes simple Python code snippets for basic arithmetical calculations and math-related problems. The generated code runs in a highly restricted environment with only basic mathematical operations available. Input: query: str -A clear, specific description of the arithmetic calculation or math problem to be solved, including any necessary numerical inputs. Output: dict -A dictionary containing the generated code, calculation result, and any error messages. Output prompt: Given a query, generate a Python code snippet that performs the specified operation on the provided data. Please think step by step. Ensure to break down the process into clear, logical steps Year: ()
Ref_id:b64 Title: Restricted to basic Python arithmetic operations and built-in mathematical functions Year: ()
Ref_id:b65 Title: Cannot use any external libraries or modules, including those in the Python standard library Year: ()
Ref_id:b66 Title: Limited to simple mathematical calculations and problems Year: ()
Ref_id:b67 Title: Cannot perform any string processing, data structure manipulation Year: ()
Ref_id:b68 Title: No access to any system resources, file operations Year: ()
Ref_id:b69 Title: Cannot use 'import' statements Year: ()
Ref_id:b70 Title: All calculations must be self-contained within a single function or script Year: ()
Ref_id:b71 Title: Input must be provided directly in the query string Year: ()
Ref_id:b72 Title: Output is limited to numerical results or simple lists/tuples of numbers Year: ()
Ref_id:b73 Title: DO NOT generate loop output Year: ()
Ref_id:b74 Title: Provide clear and specific queries that describe the desired mathematical calculation Year: ()
Ref_id:b75 Title: Include all necessary numerical inputs directly in the query string Year: ()
Ref_id:b76 Title: Keep tasks focused on basic arithmetic, algebraic calculations Year: ()
Ref_id:b77 Title: Ensure all required numerical data is included in the query Year: ()
Ref_id:b78 Title: Verify that the query only involves mathematical operations and does not require any data processing or complex algorithms Year: ()
Ref_id:b79 Title: Then the RAG (Retrieval-Augmented Generation) process begins by splitting content from the page into overlapping chunks of approximately 200 words each, with a 20-word overlap to preserve context across segments from the first 1M words in each URL. Next, both the user's query and the document chunks are embedded into the vector space using the OpenAI text-embedding-3-small model. The system computes the cosine similarity between the query embedding and each chunk embedding to rank the chunks by relevance. We set that the top 10 most similar chunks are selected and passed forward as context. And a base LLM engine will summarize the extracted context. Tool Metadata of Web Search Description: A specialized tool for answering questions by retrieving relevant information from a given website using RAG (Retrieval-Augmented Generation) Year: ()
Ref_id:b80 Title: Requires valid URLs that are accessible and contain text content Year: ()
Ref_id:b81 Title: May not work with JavaScript-heavy websites or those requiring authentication Year: ()
Ref_id:b82 Title: Performance depends on the quality and relevance of the website content Year: ()
Ref_id:b83 Title: May return incomplete or inaccurate information if the website content is not comprehensive. 5. Limited by the chunking and embedding process which may miss context Year: ()
Ref_id:b84 Title: Requires OpenAI API access for embeddings and LLM generation. Best Practice 1. Use specific, targeted queries rather than broad questions Year: ()
Ref_id:b85 Title: Ensure the URL is accessible and contains relevant information Year: ()
Ref_id:b86 Title: Prefer websites with well-structured, text-rich content Year: ()
Ref_id:b87 Title: For complex queries, break them down into smaller, specific questions Year: ()
Ref_id:b88 Title: Verify important information from multiple sources when possible Year: ()
Ref_id:b89 Title: Use it as part of a multi-step research process rather than a single source of truth Year: ()
Ref_id:b90 Title: LLM-BASED JUDGING We employ GPT-4o as our judge model using a two-step "analyze-then-judge" instruction paradigm to ensure both accuracy and efficiency Year: ()
Ref_id:b91 Title: Extract: Isolate the final answer from the Model Response, ignoring all reasoning steps. Look specifically for content within ... or the concluding statement Year: ()
Ref_id:b92 Title:  Year: ()
Ref_id:b93 Title: Mathematical Answers: Must be mathematically identical Year: ()
Ref_id:b94 Title: Textual Answers: Ignore formatting (commas, spaces), case sensitivity Year: ()
Ref_id:b95 Title: Multiple Choice Questions (MCQ): The answer must match either the correct option Year: ()
Ref_id:b96 Title: Verdict: Return "True" only if the normalized answers are semantically or mathematically equivalent. Inputs: Question: {Question} Model Response: {Final Response from Solution Generator} Ground Truth: {GT} Output Format: Present your response in the following structured format. Do not include any extra text or explanations. <analysis>: Brief analysis of the comparison Year: ()
