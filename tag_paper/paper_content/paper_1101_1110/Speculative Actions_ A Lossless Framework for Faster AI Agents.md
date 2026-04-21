Title: Speculative Actions: A Lossless Framework for Faster Agentic Systems
Abstract: AI agents are increasingly deployed in complex, interactive environments, yet their runtime remains a major bottleneck for training, evaluation, and real-world use. Typical agent behavior unfolds sequentially, where each action requires an API call that can incur substantial latency. For example, a game of chess between two state-of-the-art agents can take hours. We introduce speculative actions, a lossless acceleration framework for general agentic systems. Inspired by speculative execution in microprocessors and speculative decoding in LLM inference, our method uses faster models to predict likely future actions and executes them in parallel, committing only when predictions match. We evaluate speculative actions across gaming, e-commerce, and web search environments, and additionally study a lossy extension in an operating systems setting. Across domains, we achieve up to 55% next-action prediction accuracy, translating into substantial latency reductions. Finally, we present a cost-latency analysis that formalizes the tradeoff between speculative breadth and time savings. This analysis enables principled tuning and selective branch launching, to ensure multi-branch speculation delivers practical speedups without prohibitive cost growth.

Section: 
1 Introduction Large language model (LLM)-driven agents are shifting from single-shot predictions to processes that run inside rich environments: browsers, operating systems, game engines, e-commerce stacks, and human workflows. These environments are not incidental; they determine what the agent can observe and do, gate progress through interfaces and rate limits, and dominate end-to-end latency. In practice, agent behavior unfolds as a sequence of environment steps (tool calls, Model Context Protocol (MCP) server requests, human-in-the-loop queries, and further LLM invocations), each with non-trivial round-trip time and cost. As capabilities improve, a new bottleneck emerges: timeto-action in the environment. Even when accuracy is high, an agent that pauses too long between steps is impractical for interactive use or high-throughput automation.
OS Tasks (Abhyankar et al., 2025) Deep Research (OpenAI, 2025) Data Pipeline (Jin et al., 2025) Kaggle Chess Game (Kaggle, 2025) 10-20 min 5-30 min 30-45 min 1 hour Table 1: Estimated time state-of-the-art AI agents spend on various tasks/environments.
As shown in Table 1, AI agents may require tens of minutes to hours to complete a single run across different environments, a cost that grows significantly when hundreds or thousands of iterations are needed for reinforcement learning or prompt optimization (Agrawal et al., 2025).
This inefficiency arises from the inherently sequential nature of API calls. Thus, we ask a simple question in this paper: Must an agent interact with its environment in a strictly sequential manner?
Actor = [e2e4] a t Actor = [c5c7] a t+1 Actor = [e2e4] a t Speculator = [d2d4] â 1 t = [e2e4] â k t â k t+1 = [c5c7] Validate Prediction â 1 t ≠ a t Time Saved Non-Speculative Speculative Actor Actor …
t Fast speculation
Figure 1: Illustration of our framework in a chess-playing environment. While the Actor issues an LLM call to decide the next move, the Speculator uses a faster model to guess it. These guesses enable parallel API calls for the next steps, and once a guess is verified, the system gains time through parallelization. The process runs in the backend, ensuring a lossless speedup for the user.
Our answer is no. Inspired by speculative execution in microprocessors and speculative decoding for LLM inference, we propose speculative actions: a general framework that allows agents to predict and tentatively pursue the most likely next actions using faster models, while slower groundtruth executors (powerful LLMs, external tools, or humans) catch up. In effect, the agent stages environment interactions (prefetching data, launching safe parallel calls, and preparing reversible side effects) so that validation, not waiting, is the critical path. When those slower evaluators confirm the guesses, progress has already been made; when they disagree, we execute as usual. The result is an as-if-sequential, lossless interface with parallel, opportunistic internals.
Concretely, in such agents, speculative actions introduce two roles in the environment loop:
• Actor(s): authoritative but slow executors (e.g., SOTA LLMs, external APIs, environment's own responses, or humans) whose outputs materialize the ground truth for correctness and side effects.
• Speculator(s): inexpensive, low-latency models that predict the next environment step, i.e., the action, its arguments, and the expected observation or state delta. Examples include smaller LLMs, same LLM with reduced prompts and reasoning steps, and domain heuristics.
A key design goal is losslessness relative to the environment's baseline semantics: speculative actions should not degrade final outcomes compared to a strictly sequential agent. We achieve this with (a) semantic guards (actors confirm equivalence of state transitions before commit), (b) safety envelopes (only idempotent, reversible, or sandboxed speculative side effects), and (c) repair paths (rollback or compensating actions when a guess is rejected). In many environments (e.g., web search, pre-checkout shopping carts, and OS-level operations in a sandbox) these patterns are natural and inexpensive to implement.
Can we guess the next API calls of agents? We show that, in practice, API intents can often be guessed with reasonable accuracy. In particular, we demonstrate speculative actions across four environments, each highlighting different aspects of agent latency:
• Turn-based gameplay (e.g., chess): the Speculator predicts the opponent's move while waiting for its turn. See Fig. 1.
• E-commerce: while conversing with a shopper, the Speculator proactively infers the shopper's intent (e.g., returning an item), and triggers tool calls in advance (e.g., checking return eligibility).
• Multi-hop web search: while awaiting results from slow external calls (e.g., Wikipedia), the Speculator can guess answers from its knowledge base, and execute subsequent search queries.
• Operating systems (lossy extension): speculative, reversible actions react immediately to workload and environment changes, boosting end-to-end performance while actors confirm.
Across these settings, we observe substantially reduced latency, with up to 55% accuracy in predicting the next API calls and 20% end-to-end speedup. These results are achieved with a simple single-step speculation, and can be improved by advanced techniques such as adaptive speculation.
Finally, we give a cost-latency analysis that formally characterizes the tradeoff between speculating additional API calls and the resulting time savings. We provide a theoretical baseline for choosing the speculative breadth, and show that the cost incurred by confidence-based selection grows substantially slower than naively scaling the number of speculative branches. Furthermore, in our OStuning environment where losslessness is not required, cost and latency can actually both decrease. Our code is publicly available at https://github.com/naimengye/speculative-action.
this section cite: ['b0', 'b13', 'b14', 'b1']

Section: Related Work
Speculative decoding and reasoning Our work is inspired by the use of speculative decoding in LLM inference. This technique accelerates autoregressive inference by using a small model to propose tokens which a larger target model verifies in batches, committing correct tokens and regenerating failures (Leviathan et al., 2023;Zhang et al., 2024;Chen et al., 2023). At the reasoning level, speculation has also been used to accelerate chain-of-thought (Wang et al., 2025b;a;Fu et al., 2025). Our framework adopts the same speculate-verify pattern at the level of API calls.
Speculative planning for LLM agents More directly related are recent works on speculative planning for LLM-based agents (Hua et al., 2024;Guan et al., 2025). Hua et al. (2024) introduce interactive speculative planning, where a fast approximator proposes multi-step lookahead plans that are verified by a stronger model, with user interruption integrated. Their approach focuses on depthoriented speculation along a single planning branch. Building on this, Guan et al. (2025) propose an online reinforcement learning method to dynamically determine the number of future steps to speculate, optimizing a cost-latency tradeoff while maintaining lossless execution.
Our work differs along two dimensions. First, we generalize speculation beyond planning to the entire agentic environment, including LLM calls, internal and external tool APIs, MCP-server interactions, and even human responses. This yields a unified framework for agentic speculation, particularly consistent with the emerging "environment" and MCP perspectives on agentic systems. Second, instead of depth-focused multi-step lookahead, we study a breadth-focused k-branch single-step strategy, where multiple actions are speculated in parallel at each step. We provide a cost-latency analysis for this scheme and derive closed-form expressions for expected time and token savings (Theorem 4). Section 5 compares breadth-and depth-focused strategies under a unified analytical framework. While Guan et al. (2025) optimize depth dynamically, we characterize the optimal number of speculative branches per step as a function of predicted accuracy (Section 5.2).
Speculation in systems and architecture Speculation is prominently used in computer architecture to increase parallelism by executing instructions before their outcomes were resolved (Tomasulo, 1967) and rolling back when predictions were wrong (Lam & Wilson, 1992). In light of security vulnerabilities that exploit microarchitectural speculative execution, (Mambretti et al., 2019) developed Speculator to analyze CPU speculation behavior.
Similar ideas arise in systems software as thread-level speculation, which parallelizes sequential code under assumed independence and rolls back upon detecting data dependencies or conflicts (Estebanez et al., 2016). Recently, (Liargkovas et al., 2023) explored the use of tracing and containment to speculatively but safely run shell scripts out of order. Beyond traditional systems context, speculative techniques have also been used to parallelize otherwise sequential security checks (Nightingale et al., 2008),test configuration changes in isolation (Su et al., 2007), and accelerate policy simulation in supply chain optimization (Farias et al., 2024).
2 Framework
An agentic system is modeled as a Markov Decision Process (MDP) (s t , a t ), where s t denotes the state and a t the agent's action at step t. This model admits considerable flexibility: an action may represent a chatbot response, a tool call, or a button clicked by a computer-use agent, among others.
From a systems perspective, we model each action in an agentic system as an API call, which may block execution until a response is returned. This abstraction offers two key advantages: (1) it precisely defines what constitutes an action, and (2) it provides a unified framework for optimizing system latency, as we will see shortly. Notably, this perspective aligns with the recent development of MCP servers for agentic systems (Anthropic, 2024).
Formally, at each step t, the policy π maps the current state s t to an API call: (h t , q t ) ← π(s t ), where h t specifies the target API to invoke and q t its associated parameters. We write āt h t (q t ) a t ← await(ā t ) to denote an asynchronous API invocation that returns a future (a pending action), and the await for when the response actually arrives. We use the bar notation (e.g., ā) for futures and a cache C : (h, q) → ā that maps an API call specifier to its pending response. The left squiggly arrow indicates an asynchronous call with non-negligible delay.
The system subsequently transitions to the next state via a transition function f : s t+1 ← f (s t , a t ). As a concrete example, consider chess: the policy π determines how to construct the prompt based on the current board state, a t corresponds to the move proposed by the LLM's response, and f updates the board configuration accordingly. Note that the LLM call is the API, its response is the move a t .
This formulation subsumes a broad range of realizations:
• LLM calls: each invocation of an LLM within the agent can be treated as an action.
• Tool / MCP server calls: each actual call for internal/external tools is treated as an action: e.g., terminal access, web search, deep research APIs, weather APIs, or browser-use MCPs.
• Human-as-an-API calls: furthermore, human responses themselves can be abstracted as API calls, often incurring even longer latencies than automated tools.
Given this abstraction, the fundamental bottleneck in executing agentic systems becomes apparent: each API call must complete before the next can be issued. To break this sequential dependency, we propose to speculate a set of API responses {â t } using a faster model while waiting for the true response a t . This enables speculative API calls for step t + 1 to be launched in parallel. At time t, if the API call (h t , q t ) can be found in the cache (cache hit), the system can skip the actual invocation and only wait for the pending action corresponding to this call to return (if not already returned). Formally, the algorithm is specified in Algorithm 1.
The resulting speedup relies on two key assumptions: Assumption 1 (Speculation accuracy). The speculative model ĝ guesses the current-step response a t accurately enough that the implied next call (h t+1 , q t+1 ) = π( f (s t , ât )) matches the true next call with probability p > 0.
As shown later, this often holds in practice because API responses are typically predictable. Assumption 2 (Concurrent, reversible pre-launch). Multiple API calls can be launched concurrently, and pre-launched calls that do not correspond to the realized trajectory have no externally visible side effects (or can be rolled back).
In practice, this assumption is satisfied under modest traffic for many external APIs (e.g., web search, OpenAI LLM queries, email lookups). For self-hosted LLMs, concurrent calls also incur only minimal additional cost due to continuous batching.
We can then establish the following result (with proof deferred to the Appendix A).
Proposition 1. Under Assumptions 1-2, suppose at each step the speculative branch implies the correct next call (h t+1 , q t+1 ) with probability p, independently across t ∈ [1, T -1]. Let the latency of ĝ be Exp(α) and the latency of the actual API call be Exp(β) with β < α. All latencies and guesses occur independently. Assume the transition f and API parameter construction π are negligible. Then the ratio between the expected runtime of Algorithm 1, denoted E[T s ], and the expected runtime of sequential execution, E[T seq ], is E[T s ] E[T
seq ] = 1 - 1 T α α + β (T -1)p(k) 1 + p(k) + p(k) 2 (1 + p(k)) 2 - p(k) 2 (1 + p(k)) 2 (-p(k)) T -1 T →∞ ------→ 1 - p(k) 1 + p(k) • α α + β
where p(k) = 1 -(1p) k denotes the probability of at least one of the k speculations hit.
this section cite: ['b17', 'b30', 'b3', 'b8', 'b11', 'b9', 'b11', 'b9', 'b9', 'b24', 'b16', 'b20', 'b6', 'b18', 'b21', 'b23', 'b7', 'b2']

Section: Algorithm 1 Speculative actions with k-way parallel next calls
Require: Initial state s 0 , horizon T , transition f , policy π, predictor ĝ, cache C. We use ā to denote pending action. 1: for t = 0 to T -1 do
2: Policy: (h t , q t ) ← π(s t ) 3: if (h t , q t ) ∈ C then ▷ Cache hit 4: āt ← C[(h t , q t )] 5: a t ← await(ā t ) ▷ Await pending action if not returned already 6: s t+1 ← f s t , a t 7: continue 8: end if 9: Actor: Issue real request (returns future): āt h t (q t ) 10:
Speculator:
{â (i) t } k i=1 ← await(ĝ(s t , (h t , q t )))
▷ Actor and speculator run in parallel 11:
for i = 1 to k do ▷ One-step speculative rollout per guess 12: ŝ(i) t+1 ← f s t , â(i) t 13: ( ĥ(i) t+1 , q(i) t+1 ) ← π( ŝ(i) t+1 ) 14: Pre-launch: ā(i) t+1 ĥ(i) t+1 q(i) t+1 ▷ Return pending action, hence non-blocking 15: C[( ĥ(i) t+1 , q(i) t+1 )] ← ā(i) t+1
▷ Cache speculative pending actions 16:
end for 17:
Wait for resolved a t from Actor: a t ← await(ā t )
18:
s t+1 ← f s t , a t 19: end for Proposition 1 suggests the end-to-end latency reduction has an upper bound of 50%, occurring when p = 1 and α = ∞.This can be further improved by the multi-step extension below.
Extension Algorithm 1 is only a simple demonstration of the idea. For example, one can naturally extend Algorithm 1 to multi-step speculation, where the Speculator predicts not only the next, but s steps ahead. This yields a tree structure with deeper rollouts. This can be further combined with adaptive speculation: instead of generating k guesses for a t uniformly, the Speculator also estimates confidence for each guess (e.g., via prompting LLMs or uncertainty-quantification methods), this is explored in Section5. The most promising branches can then be expanded in a beam-search-like manner. Together, these ideas highlight the richness of speculative actions. Despite Algorithm 1's simplicity, the results from the four use cases in the following sections are already highly promising.
Side effects and safety Speculation executes a hypothesized next action ât+1 that may be wrong, so safety requires the ability to simulate first and then commit or roll back. In domains like chess, rollback is trivial; in others, overwrite is easy (e.g., OS tuning). But many systems involve irreversible or externally visible effects (e.g., deleting records, placing orders), where naive speculation is harmful. Thus, speculation must be limited to cases where mispredictions are reversible, via forking, snapshot restoration, or roll-forward repair (e.g., refund/replace).
this section cite: []

Section: Environments
We now instantiate speculative actions in three environments-chess, e-commerce dialogue, and multi-hop web search-chosen to stress distinct latency bottlenecks (reasoning, tool/API round trips, and information retrieval). We pair a fast Speculator with a slow Actor and implement Algorithm 1.
this section cite: []

Section: Chess Environment
We demonstrate the effectiveness of our framework in the context of multi-agent gameplay, focusing on chess as a canonical turn-based example. In standard play, analysis is strictly sequential: each player begins analysis only after the opponent has completed their turn. This serialization introduces substantial idle time. Particularly when both players rely on computationally intensive reasoning models, a single game can stretch to hours of wall-clock time. Our framework relaxes this constraint through speculative parallel analysis, allowing players to anticipate and prepare for likely opponent moves in advance. We show that this results in significant reductions in overall game duration.
this section cite: []

Section: Implementation
We implement our framework on top of TextArena (Guertler et al., 2025), which provides a standardized gameplay interface for LLM-driven agents.
Speculative pipeline At turn t, the game state s t corresponds to the current board position. The in-turn player issues an API call h t with parameter q t constructed from s t together with a reasoningeliciting prompt. At this point, player P is to move, and player Q awaits. Proceeds as follows:
• Current in-turn player P: the player receives s t , makes an API call h t to the agent with parameter q t = (s t , prompt). This API call returns the next move a t = h(q t ), typically with high latency due to deep and extensive reasoning.
• Other out-of-turn player Q:
1. Prediction phase The Speculator also receives the board state s t and issues an API call ĥt , using a prompt optimized for speed rather than depth. It returns the top-k move predictions â(1) t , â(2) t , . . . , â(k) t , ordered by confidence. 2. Parallel computation For each predicted move â(i)
t , the out-of-turn player Q immediately launches a process analyzing a next move â(i) t+1 = h t+1 ( ŝt+1 , prompt) for i ∈ {1, . . . , k}, where ŝ(i) t+1 = f (s t , â(i) t ) denotes the next state resulting from applying the predicted action â(i) t to s t . 3. Validation When the current in-turn player P finishes reasoning and returns its move a t , we immediately check whether it matches any of the predicted moves â(1) t , â(2) t , . . . , â(k) t . 4. Commit or restart If a match exists, we commit to the corresponding speculative branch, advancing directly to s t+1 = f (s t , a t ). The game thus skips ahead, terminating other threads. If no match exists, we discard all speculative branches and continue with Q's regular move computation a t+1 = h t+1 ( f (s t , a t ), prompt).
This pipeline is lossless: the final trajectory remains identical to non-speculative play, but time is saved through parallelized reasoning.
Agent Configuration We find that using the same model for both Speculator and Actor, but with different prompts, maximizes prediction accuracy while keeping speculation fast. Accordingly, in our experiments, the Actor is instantiated with GPT-5 with high reasoning effort; and the Speculator is instantiated with GPT-5 configured with low reasoning effort and a specialized system prompt designed for rapid move prediction rather than exhaustive analysis. We evaluate our framework in terms of both time saved and prediction accuracy. We track two metrics: (i) prediction accuracy: the fraction of rounds in which any speculative prediction matches the actual move; and (ii) time saved: (T seq -T s )/T seq , where T s and T seq denote speculative and sequential execution times, respectively.
More predictions improve time savings and accuracy.
Figure 2 reports results over 30 steps. Our framework consistently reduces execution time, with larger savings as the number of speculative predictions increases. Across 5 runs, using 3 predictions yields an average time saving of 19.5% with an average prediction accuracy of 54.7%.
this section cite: ['b10']

Section: Randomness of agent call in gameplay.
The variance in Figure 2 reflects realistic latency fluctuations from live API calls. Even with correct predictions, speedups vary: if the resulting position is trivial, little acceleration is realized; large gains occur only when predictions lead to positions requiring deep analysis. In addition, API latency itself is inherently stochastic. Backend load fluctuations (e.g., concurrent traffic to the model provider) can cause the same API call with different latency across runs. Consequently, measured latency reductions exhibit natural variability and are not perfectly reproducible.
this section cite: []

Section: E-Commerce Environment
Beyond competitive gameplay, customer-agent interactions in e-commerce provide a real-world setting where latency significantly impacts user experience. In a typical workflow, the customer submits a query through a chat interface and waits while the agent sequentially invokes multiple API calls-for example, processing a return may involve retrieving order information, validating eligibility for each item, and initiating the return. These chained calls can introduce substantial delay. By contrast, if some API calls are correctly speculated and executed in advance, the agent can return results immediately once the query arrives, making the interaction feel seamless. We evaluate this setting using the retail environment from τ-bench (Yao et al., 2024).
this section cite: ['b29']

Section: Experimental Setup
Figure 3: APIs prediction accuracy across various Speculator models.
this section cite: []

Section: Speculative pipeline
In this scenario, the current state s t is defined as the conversation history up to turn t, and h t are the API calls required to answer the user's query (eg. get user details, get order details). Our Speculator will predict 1. The user's query ât ; 2. The target API calls and their corresponding parameters ( ĥ(i) t+1 , q(i) t+1 ) for i ∈ {1, ..., k}, conditioned on the current state s t and the predicted user's query from step 1. Since the number of API calls for each turn is not fixed, the Speculator must also predict k.
this section cite: []

Section: Agent configuration
We evaluate multiple Speculator models, including OpenAI GPT variants (gpt-5-nano, gpt-5-mini, gpt-5) and Google Gemini (gemini-2.5-flash) under different reasoning budgets (1024/2048/4096 tokens). Motivated by prior work where heterogeneous LLM ensembles outperform single models (Jiang et al., 2023;Chen et al., 2025), we consider two configurations: (i) a single-model Speculator, and (ii) a multi-model Speculator, where comparable models run in parallel (e.g., gpt-5-nano with low-budget Gemini, gpt-5-mini with medium-budget Gemini). Their outputs are aggregated into a shared pool of candidate speculative actions.
At runtime, once the user simulator reveals the ground-truth utterance, the Actor validates the speculative API calls: correct predictions are committed immediately (eliminating latency), while incorrect ones are discarded without affecting correctness.
Evaluation We evaluate performance using APIs prediction accuracy, defined as the fraction of speculative API calls that match the ground-truth APIs required to resolve the user's query. This metric directly reflects the proportion of turns in which the user receives an immediate response, without waiting for API execution: higher prediction accuracy translates into greater time savings.
this section cite: ['b12', 'b4']

Section: Results
Figure 3 shows that between 22% and 38% of API calls are correctly predicted by the Speculator. Accuracy improves with stronger models and the multi-agent configuration consistently outperforms single-model speculation. Importantly, low-budget models speculate in only 2-3 seconds (per the LLM API providers leaderboardfoot_0 ), well below the average user typing time of about 30 seconds (assuming 40 words per minute). This means that in roughly one third of turns, the agent can respond faster than sequential execution, without waiting for API execution. We further evaluate our framework on HotpotQA, a setting where the main performance bottleneck arises from information retrieval latency. In this example, the agent must answer multi-hop questions through sequential Wikipedia API calls (Yang et al., 2018), mirroring real-world agentic workflows with high round-trip network latency. In this setting, the Speculator predicts likely Wikipedia content while the actual API call executes. Parallelism allows the agent to continue reasoning on provisional information rather than blocking on API latency. See Appendix B.2 for details about our experimental setup.
We evaluate on the accuracy of the predicted API calls. As shown in Figure 4, the Speculator successfully predicts ground truth API call up to 46% of the time with top-3 prediction. This accuracy improves significantly from top-1 to top-3 predictions, yielding substantial accuracy gains with modest speculation width increase. Our speculation provides value by precomputing reasoning paths during otherwise idle API waiting time.
4 Beyond Lossless Speculation: OS Hyperparameter Tuning Environment Thus far, our experiments have focused on lossless speculation, where speculative actions are validated sequentially before commitment. We now turn to a lossy setting that relaxes this constraint. In latency-sensitive environments like an operating system, waiting for a powerful but slow Actor (10-15s deliberation) can leave the system in a degraded state. Instead, we use a fast Speculator to apply immediate provisional adjustments while the Actor deliberates. This is made safe by a last-writewins mechanism-the Actor's final decision simply overwrites any speculative action, removing the need for complex rollbacks. This method accelerates convergence and improves reaction time, which we evaluate on the sysbench cpu benchmark, a CPU-bound workload (Kopytov, 2020).
this section cite: ['b27', 'b15']

Section: Experimental Setup
We tune Linux's Completely Fair Scheduler (CFS) parameter min granularity, which controls a task's minimum timeslice. This knob strongly affects scheduling performance: smaller timeslices reduce latency but can degrade throughput, yielding a classic trade-off. Building on (Liargkovas et al., 2025), we augment the prior LLM-based tuning setup with a speculative control loop.
The Speculator proposes a parameter update each second using the latest performance metric. The Actor, in contrast, responds every 10-15 seconds after analyzing a compressed chronology of the Speculator's recent (measurement, action) pairs. Upon arrival, the Actor's decision is applied immediately and its state resets the Speculator's context, preventing drift from the validated narrative.
Evaluation We evaluate three systems: (1) Actor-only: slow but deliberative (10-15 s interval);
(2) Speculator-only: fast (1 s interval) but non-extensive; (3) Speculator-Actor: combined system using speculative updates between Actor decisions.
this section cite: ['b19']

Section: Results
Speculator mitigates poor-reaction slowdowns. As shown in Figure 5 (right), the Speculator significantly improves reaction time. During recovery, the full Speculator-Actor system maintains an average p95 latency of 37.93 ms, compared to 54.00 ms for Actor-only, which remains longer in degraded states (initially 102.97 ms). Fast speculative updates provide immediate mitigation while the Actor deliberates (details in §B.3.3).
Speculator accelerates convergence to optimum. Figure 5 (left) shows that the joint system reaches the optimal setting (0.2 ms min granularity) in 10-15 s, whereas Actor-only requires ∼200 s and remains trapped in highly suboptimal regions (e.g., latency > 120ms) for extended periods. Rapid speculative exploration helps the Actor avoid pathological configurations. Speculator-only reacts quickly but is suboptimal. While Speculator-only stabilizes rapidly, it converges to a worse configuration (0.55 ms; 36.24 ms latency) than the joint system (0.2 ms; 30.26 ms). Without the Actor's deeper reasoning, it cannot escape local minima.
Cost and latency both decrease. Despite additional speculative calls, total cost is lower due to faster convergence. As shown in Table 3, Actor-only converges at ∼200 s with a total cost of 2.18 cents, whereas Speculator-Actor converges in ∼13 s with only 0.17 cents.
Overall, the joint system combines fast adaptation with strategic guidance, achieving both responsiveness and optimal steady-state performance.
5 Cost-Latency Tradeoff Performing more speculative API calls improves accuracy but also raises costs when pricing is based on the number of calls or tokens. In this section, assume a fixed token per unit time and fixed per token cost, and analyze the cost-latency tradeoff. Full details can be found in Appendix C.
this section cite: []

Section: Breadth-focused Speculation (Algorithm 1)
In addition to Proposition 1, we obtain a closed-form expression for relative cost increase ratio
lim T →∞ E[M spec -M seq ] E[M seq ] ≤ k -k + α α + β p(k) 1 + p(k)
.
See Theorem 4 in Appendix C for the formal theorem. Comparing Proposition 1 with the above expression, we see that both ratios are governed by p(k). Thus, given an estimation of p(k), a user can directly tune k offline, trading off cost against latency. Our experiment (Figure 6) shows this non-linear dependence on k, and additional empirical results can be found in Appendix C.2.
this section cite: []

Section: Dynamic selective speculation.
So far we assume a fixed branch accuracy p. In practice, we sometimes are able to obtain perspeculation confidence estimates (e.g., from intrinsic model logits, or from a separately-trained auxiliary predictor), allowing confidence-aware selective speculation. At each speculation window, the accuracies of the k speculative branches are random and drawn from a known distribution (which may vary over time). Before acting, the realized accuracy vector p = (p 1 , . . . , p k ) is observed, and we choose how many of the top branches to launch.
We model the cost-latency tradeoff via the weighted objective max r • If the top m branches are launched, the probability of a cached step is q(m; p) = 1 -m j=1 (1p ( j) ), where p (1) ≥ • • • ≥ p (k) are the sorted confidences.
this section cite: []

Section: Theorem 3 (Confidence-aware selective speculation).
There exist scalars ∆ t such that at each speculation window t, the optimal breadth satisfies m ⋆ t (p) ∈ arg max m∈{0,...,k} {q(m; p) ∆ t -cm}.
The continuation values are given by the backward recursion ∆ T = 0,
∆ 1::T -1 = ℓ-E max m {q(m; P) ∆ t+1 -cm}
In the stationary case (time-homogeneous accuracy distribution), the continuation values ∆ t collapse to a single constant ∆ ⋆ . Thus branches are added greedily in descending confidence order while ∆ ⋆ • δq(m; p) ≥ c, where δq(m; p) is the marginal gain from adding one more branch.
this section cite: []

Section: Interpretation and implementation
The key implication is structural: dynamic selection collapses to a one-dimensional trade-off. Additional branches are launched only when their incremental hit probability, scaled by a single continuation value (∆ t or ∆ ⋆ ), exceeds the marginal cost c. Under stationarity, this reduces to estimating ∆ ⋆ offline; at runtime, the system simply sorts confidences and adds branches greedily, requiring O(k) computation per step.
this section cite: []

Section: Empirical results
We implement a simple constant-threshold approximation of the stationary rule in the chess environment. At each step, after generating speculative branches, we use a predictor to estimate the correctness probability of each branch and continue only with those whose predicted accuracy exceeds 50%. This implements a simplified threshold rule consistent with the structure suggested by Theorem 3. Our method achieves the lowest additional token cost while providing greater latency reduction than naively launching 1 or 2 speculations per step.
this section cite: []

Section: Depth-focused speculation.
The previous two strategies are breadth-focused: each speculation is immediately followed by a real API call (speculative depth 1). Additionally, we analyze the opposite extreme: a depth-focused policy, in which multi-step speculations are continuously spawned. Somewhat counterintuitively, this strategy does not lead to exponential branch growth. Speculative calls are only extended when either a speculative or real call returns, and inconsistent subtrees are immediately pruned. Consequently, the system can run at most a/b speculative steps ahead (governed by the relative speeds of real vs. speculative calls), ensuring that the number of active branches remains bounded and does not scale with the horizon T . Under this policy, we can show that (formal theorem in Appendix 6)
E[T seq -T spec ] E[T seq ] = T -1 T p 1 - b a , E[M spec -M seq ] E[M seq ] ≈ T -1 T (1 -p) a 2b - 1 2 + p
Compared to breadth speculation, depth speculation improves the latency coefficient from p 1+p to p, increasing the theoretical speedup ceiling from 1 2 to 1. The cost term scales with (1p)( a 2b -1 2 ), which captures how many speculative steps can accumulate before the real response arrives.
this section cite: []

Section: Conclusion
In this paper we propose Speculative Actions, a lossless framework for accelerating general agentic environments by breaking the strict sequentiality of their interaction loops. Our approach treats every step, whether an LLM call, tool invocation, MCP request, or human response, as an API call subject to prediction and parallelization. By pairing a fast Speculator with a slow but authoritative Actor, the framework enables agents to anticipate and prepare likely next actions in parallel, transforming otherwise idle waiting time into productive computation. We instantiate the framework across four representative environments and observe consistent substantial latency reduction. Finally, we provide a cost-latency analysis that addresses the tradeoff between the additional cost and the latency gains from launching additional speculative actions.
this section cite: []

Section: References
Ref_id:b0 Title: Osworld-human: Benchmarking the efficiency of computer-use agents Year: (2025)
Ref_id:b1 Title: Reflective prompt evolution can outperform reinforcement learning Year: (2025)
Ref_id:b2 Title: Introducing the model context protocol Year: (2024-11)
Ref_id:b3 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b4 Title: Harnessing multiple large language models: A survey on llm ensemble Year: (2025)
Ref_id:b5 Title: Emmanuel Cecchet, Snigdhaswin Kar, and Prabodh Mishra. The design and operation of CloudLab Year: (2019-07)
Ref_id:b6 Title: A survey on thread-level speculation techniques Year: (2016-06)
Ref_id:b7 Title: Speeding up policy simulation in supply chain rl Year: (2024)
Ref_id:b8 Title: Scaling speculative decoding with lookahead reasoning Year: (2025)
Ref_id:b9 Title: Dynamic speculative agent planning Year: (2025)
Ref_id:b10 Title:  Year: (2025)
Ref_id:b11 Title: Interactive speculative planning: Enhance agent efficiency through co-design of system and user interface Year: (2024)
Ref_id:b12 Title: LLM-blender: Ensembling large language models with pairwise ranking and generative fusion Year: (2023-07)
Ref_id:b13 Title: Elt-bench: An end-to-end benchmark for evaluating ai agents on elt pipelines Year: (2025)
Ref_id:b14 Title: Game arena Year: (2025)
Ref_id:b15 Title: Sysbench: Scriptable benchmark tool Year: (2020)
Ref_id:b16 Title: Limits of control flow on parallelism Year: (1992)
Ref_id:b17 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b18 Title: Executing shell scripts in the wrong order, correctly Year: (2023)
Ref_id:b19 Title: An expert in residence: Llm agents for always-on operating system tuning Year: (2025-12)
Ref_id:b20 Title: Speculator: a tool to analyze speculative execution attacks and mitigations Year: (2019)
Ref_id:b21 Title: Parallelizing security checks on commodity hardware Year: (2008)
Ref_id:b22 Title:  Year: (2025)
Ref_id:b23 Title: Autobash: improving configuration management with operating system causality analysis Year: (2007)
Ref_id:b24 Title: An efficient algorithm for exploiting multiple arithmetic units Year: (1967)
Ref_id:b25 Title: Efficient reasoning for llms through speculative chain-of-thought Year: (2025)
Ref_id:b26 Title: Accelerating large language model reasoning via speculative search Year: (2025)
Ref_id:b27 Title: Hotpotqa: A dataset for diverse, explainable multi-hop question answering Year: (2018)
Ref_id:b28 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b29 Title: A Benchmark for Tool-Agent-User Interaction in Real-World Domains Year: (2024-06)
Ref_id:b30 Title: Beyond the speculative game: A survey of speculative execution in large language models Year: (2024)
