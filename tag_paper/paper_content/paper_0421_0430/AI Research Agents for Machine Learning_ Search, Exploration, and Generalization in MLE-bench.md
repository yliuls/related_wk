Title: AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench
Abstract: AI research agents are demonstrating great potential to accelerate scientific progress by automating the design, implementation, and training of machine learning models. We focus on methods for improving agents' performance on MLE-bench, a challenging benchmark where agents compete in Kaggle competitions to solve real-world machine learning problems. We formalize AI research agents as search policies that navigate a space of candidate solutions, iteratively modifying them using operators. By designing and systematically varying different operator sets and search policies (Greedy, MCTS, Evolutionary), we show that their interplay is critical for achieving high performance. Our best pairing of search strategy and operator set achieves a state-of-the-art result on MLE-bench lite, increasing the success rate of achieving a Kaggle medal from 39.6 % to 47.7 %. Our investigation underscores the importance of jointly considering the search strategy, operator design, and evaluation methodology in advancing automated machine learning.Finally, to conduct experiments, we develop AI Research Agent dojo (AIRA-dojo), a framework that provides a scalable and customizable environment for AI research agents. First, AIRA-dojo exposes a robust and flexible interface to compute resources, which is essential for building effective agents. The baseline, AIDE, implemented in AIRA-dojo achieves a performance increase of 10.68 % (absolute scale) over the reported results [4]. Second, AIRA-dojo enables users to experiment with custom operators, search policies, evaluation methods, and tasks within a comparable setup. This facilitates a rigorous scientific study of AI research automation. Our code is open-sourced at: https://github.com/facebookresearch/aira-dojo.

Section: 
1 Introduction Any Medal 0% 10% 20% 30% 40% 50% Medal Rate 35.23 45.91 39.77 39.55 42.73 43.64 45.45 47.73 47.05 47.27 [OAI] AIDEGreedy o1-preview [DOJO] AIDE * Greedy o1-preview [DOJO] AIDEGreedy R1 [DOJO] AIDEGreedy o3 AIRAEvo R1 AIRAEvo o3 AIRAGreedy R1 AIRAGreedy o3 AIRAMCTS R1 AIRAMCTS o3 Science is based on searching the open-ended space of hypotheses and testing them in a controlled experiment [27]. Recent breakthroughs have resulted in artificial intelligence (AI) agents that offer great potential to automate the scientific discovery process [49,42,3]. A key obstacle to improving research agents is that their designs entangle several factors for performance, making it difficult to pinpoint sources of improvement via controlled experiments at scale (Fig. 7). These factors span algorithm design, concrete implementation, and ability to leverage compute, as performance gains accrue only if no layer in the stack bottlenecks the benefits from additional compute resources. This challenge is exemplified in MLE-bench [4], a benchmark where AI agents compete in Kaggle competitions to solve real-world machine learning (ML) problems. Notably, the state-of-the-art approach AIDE [23] 39th Conference on Neural Information Processing Systems (NeurIPS 2025). does not fully disentangle these performance factors, providing limited insight into which components primarily drive the agent's performance and where improvements are needed.
We start by formalizing the design of AI research agents as a search algorithm composed of two components: The first one is the search policy which is used to navigate the space of candidate solutions, and the second is the operators which iteratively modifies existing solutions to generate new candidate solutions. In this framework (Fig. 2), AIDE is represented as a greedy search algorithm that, at each step, applies one of its code operators (i.e., DRAFT, DEBUG, IMPROVE) to the current best solution. This allows us to disentangle the effect of the search from that of the operators.
To assess alternative search algorithms, we develop more sophisticated agents performing Evolutionary and Monte Carlo Tree Search (MCTS) [6,24]. We empirically show that AIDE's operators, rather than the search algorithm, are a bottleneck to better performance.
Based on these findings, we design an improved set of operators and evaluate them when paired with the above-mentioned search algorithms. Our best-performing agent achieves state-of-the-art results on MLE-bench lite, increasing the rate of achieving a Kaggle medal from 39.6 % to 47.7 %.
Additionally, we investigate how the generalization gap-the difference between validation and test scores-affects the agents' performance. In ML engineering tasks, as in many realworld use cases, an agent can access only proxy metrics (e.g. validation loss) to guide the search process, while the final solution is evaluated on a held-out test set. Therefore, a gap between the expected (validation) and actual (test) loss, could potentially mislead the search process. Indeed, we find systematic overfitting: selecting the final solution in a search graph by its test-rather than validation-score, would increase the medal rate by 9 to 13 % (absolute scale), depending on the search algorithm. These findings highlight the importance of rigorous evaluation protocols during search and regularization for robust and scalable scientific discovery.
this section cite: ['b26', 'b48', 'b41', 'b2', 'b3', 'b22', 'b5', 'b23']

Section: Research Agents as Search Algorithms
AI research agents typically approach machine learning problems by generating artifacts -codebases designed to solve a given task. Executing these artifacts yields trained models, which are then evaluated against a chosen performance metric. The agent iteratively refines its artifacts to improve performance on this metric.
Previous research indicates that LLMs alone are insufficient to effectively solve such open-ended tasks [32]. In particular, LLM performance significantly improves when augmented with external tools [37], execution feedback [13], and solutions addressing context limitations. Therefore, developing high-quality models typically involves iterative experimentation, where insights from prior experiments inform subsequent refinements. Recent advancements by Jiang et al. [23] demonstrate state-of-the-art performance by conceptualizing this iterative experimentation as a tree search over potential solutions.
In this section, we formalize and generalize this perspective by modeling research agents as graphbased search algorithms. The proposed framework allows systematic exploration of alternative agent design choices, providing insights into how different algorithms affect the exploration-exploitation trade-off-a fundamental aspect of search [24,44].
this section cite: ['b31', 'b36', 'b12', 'b22', 'b23', 'b43']

Section: Graph-based Search Framework
We consider an agent that operates by searching a directed graph G t = (V t , E t ) that evolves over multiple iterations t = 0, 1, . . . , where each node v ∈ V t ⊆ S represents an artifact belonging to the set of all possible artifacts S, while each directed edge (v i , v j ) ∈ E t represents a transformation from v i to v j . The root v 0 is the initial artifact that can represent an empty or starting artifact.
this section cite: []

Section: Definition 1 (Components of the Search Algorithm).
A graph-based search algorithm is specified by the tuple F, π sel , O, π op , τ :
• Fitness Function. F : S → [0; 1] is a function that estimates the value or quality of a node v ∈ V t .
Since true fitness is typically not available, F is often a proxy measure of the value of the node.
• Selection Policy. π sel : 2 Vt → 2 Vt chooses a subset of nodes U t ⊆ V t on which to operate, typically guided by a heuristic function h : V t → R, which assigns a scalar estimate to each node. The heuristic may be derived directly from the fitness function F, such as the upper confidence bounds for trees (UCT) used in MCTS [24], or other custom heuristics tailored to the domain [5,18].
• Operator Set. O = { o ℓ : 2 S → S} L ℓ=1 comprises L transformation functions that propose new artifacts v = o ℓ ({v k } m k=1 ) ∈ S from one or more selected artifacts. In AIDE, examples include DRAFT, DEBUG, and IMPROVE instantiated as prompt-based instructions to an LLM. Composite operators (e.g. the result of applying a sequence of base operators) can also be used.
• Operator Policy. π op : O × U t → O decides which operator to apply to the current node selection.
• Termination Rule. τ halts the search when a computational budget is exhausted, progress stalls, or a fitness threshold is reached.
At each iteration, the agent selects existing promising artifacts, applies transformation operators, and updates the search graph, propelling the discovery process forward. We discuss specific instantiations of search algorithms in Sections 2.3, 4.2. In all our instantiations: 1 the termination criterion is set as the wall-clock time or the maximum number of artifacts, whichever happens first; 2 the fitness function F is defined for each node by the operator that generates or modifies it (i.e., F is not global); 3 all operators are LLM-driven except for the MEMORY operator, which is defined by hand.
this section cite: ['b23', 'b4', 'b17']

Section: Operators
We define operators as high-level functions that take in existing artifacts and produce new ones. These can range from simple rule-based parsers to LLM calls using prompting techniques [46,53] or more complex agents like Cursor [20].
This broad definition enables a unified comparison across search methods. A search algorithm can be instantiated with any mix of LLM-based, tool-based, or nested search operators.
Building on the demonstrated effectiveness of AIDE [23], we adopt a similar operator set as introduced in their framework, consisting of the following: 1 DRAFT initializes the search process by generating an initial population of candidate artifacts. 2 DEBUG attempts to identify and correct errors in invalid artifacts. 3 IMPROVE refines valid artifacts to enhance their performance according to the evaluation criteria. 4 MEMORY chooses how and where information from past artifacts is used in subsequent operations. 5 Furthermore, we introduce CROSSOVER which recombines useful elements from two artifacts to create a new candidate. Section 3 contains further details.
this section cite: ['b45', 'b52', 'b19', 'b22']

Section: Re-casting AIDE in our Notation
AIDE [23] is an LLM-driven agent that frames problem-solving as a tree-search over Python scripts.
In the tuple (F, π sel , O, π op , τ ) from Section 2.1 its components are:
Fitness & selection (F, π sel ). For each node v, fitness is the mean 5-fold cross-validation (CV) score F(v) ∈ [0, 1]foot_0 . The selection policy is greedy with respect to F. This means at iteration t the agent selects and operates on v ⋆ = arg max v∈Vt F(v) but, with probability ε bug 2 , may instead revisit a buggy node (F = 0) to aid recovery and maintain diversity.
Operator set O AIDE . The agent exposes three LLM operators {DRAFT, IMPROVE, DEBUG} and one handcrafted operator MEMORY or as defined by Jiang et al. [23]-the SUMMARIZATION operator. The operators are designed to output the following: DRAFT (3-5-sentence plan + fenced script that trains, evaluates, and writes submission.csv); DEBUG (short diagnosis and repaired script given a traceback); IMPROVE (Creating exactly one measurable change -feature, architecture, schedule, etc.
-in plan + code form); MEMORY (running summary of all previous designs, scores, and notes that is appended to every DRAFT/IMPROVE prompt).
this section cite: ['b22', 'b22']

Section: Experiment Design

this section cite: []

Section: AIRA-dojo
An agent's environment greatly influences its performance. To systematically study the space of agentic policies (see Fig. 7), we introduce the AI Research Agent (AIRA) dojo-a scalable and customizable framework for AI research agents. AIRA-dojo provides the environment in which agents operate, along with abstractions for operators and policies as described in Section 2, and tasks that define evaluation criteria for agent performance. Using these abstractions, we implement and evaluate four search policies: MCTS, Evolutionary, Greedy, and our own implementation of AIDE. We hope that AIRA-dojo's scalability and customizability, together with the provided agent and task implementations, will support and advance future research in the community.
The infrastructure design of AIRA-dojo was informed by several reliability and performance constraints, as discussed in more detail in Section G.
this section cite: []

Section: Environment
The environment defines the context in which agents operate.
Action Space. We use Jupyter notebooks to execute code from the agents. With this interface, agents can execute arbitrary Python code, perform file reads and writes, and even use the shell via Jupyter magic commands. The environment captures and returns the status, standard outputs, tracebacks for debugging, and the code block execution time to the agents.
this section cite: []

Section: Isolation.
The environment enforces program-level constraints, such as total runtime limit, GPU and CPU usage, memory, and storage, using Apptainer containers [26]. This ensures complete isolation from host systems, preventing agents from affecting the host environment or other agents, and mitigates risks from unintended actions or failures, such as data leakage or interference with other processes. Furthermore, agents possess root-like privileges within their isolated containers, granting them full control over their environment. This allows them not only to configure their environments using standard package management tools such as pip, conda, but even apt-get install.
this section cite: ['b25']

Section: Superimage.
A container image provides a base set of tools required for ML tasks. This image includes pre-configured CUDA support, deep learning frameworks like PyTorch and TensorFlow, and essential data science libraries. Each coding session starts from the original Superimage state and can diverge based on the agent's actions while isolating the state of the other agents.
Together, these design choices create a robust testbed that eliminates confounders at both the system and implementation levels. This enables reproducible benchmarking and facilitates the development of long-running agentic systems across thousands of parallel runs.
this section cite: []

Section: MLE-bench
MLE-bench [4] contains 75 Kaggle-sourced tasks for evaluating machine learning engineering agents. We evaluate on MLE-bench lite -a curated subset of 22 tasks selected from the full benchmarkallowing us to allocate more seeds per task and increase our confidence in our results.
Our experiments showed that existing methods exhibit high variance in this benchmark (see Section J). In line with the benchmark guidelines, we assess each agent's performance using the Medal Success Rate. Specifically, for each task, agents earn a bronze, silver, or gold medal according to task-specific percentile thresholds. We report the percentage of attempts in which an agent secures a medal.
this section cite: ['b3']

Section: Experimental Details
Environment. Each candidate agent is launched in a freshly initialized, sandboxed process in AIRAdojo. This guarantees that file systems and environment variables are isolated across evaluations, and there is no cross-agent interference. Every sandbox is provisioned with a fixed hardware quota: 1 dedicated H200 GPU, 24 logical CPU cores, 100 GB of RAM, and 1 TB of additional scratch storage. Internet access is permitted solely for fetching third-party packages and model checkpoints.
this section cite: []

Section: Time Constraints.
In line with MLE-bench, the agent is allowed a 24-hour wall-clock window. Within this period, each agent has a maximum runtime of 4 hours per code execution. We chose to reduce this from the 9-hour limit used in MLE-bench after preliminary experiments showed no difference in performance and a higher average number of valid nodes in the search trees.
this section cite: []

Section: LLMs.
We conducted all the experiments with the full-sized DeepSeek R1 [7] model, with 128Ktoken context window to ensure input coverage without truncation. Due to wall-clock constraints, this choice was guided by both the model's capabilities and applicable rate limits. In particular, we selected DeepSeek R1 as the most capable open-source model available, which allowed us to self-host inference servers and maintain high experimental throughput without encountering rate limits. For the main results, we also evaluated o3 [35], one of the most capable closed-source models. To ensure experiment validity and avoid hitting rate limits, we limited the number of parallel runs when experimenting with o3. We always use GPT-4o [34] with Structured Outputs to parse code execution outputs-extracting run success, summary text, and validation metric. For the figures, we take [OAI] AIDE o1 artifacts from the MLE-bench [4] GitHub repository. To directly compare with their results, we evaluated o1-preview, but only completed experiments for one agent before deprecation. See Section E for details.
this section cite: ['b6', 'b34', 'b33', 'b3']

Section: AIRA

this section cite: []

Section: Operators O AIRA
As part of AIRA-dojo, we propose a new operator set based on O AIDE , denoted O AIRA . To this end, we focus on maintaining a cleaner context through better-scoped memory, and encouraging structured reasoning and strategic diversity in ideation. The key differences are: 1 Prompt-adaptive complexity. We introduce a dynamic complexity cue within the system prompt in order to guide the complexity of artifacts generated by the DRAFT and IMPROVE operators. The complexity is determined by the number of children, n c , of the node being processed: complexity(n c ) = "minimal" if n c < 2, "moderate" if 2 ≤ n c < 4, "advanced" if n c ≥ 5
For the DRAFT operator, this cue influences the complexity of the generated ideas. For the IM-PROVE operator, it guides the complexity of the enhancements. This dynamic signal helps prevent premature over-engineering by ensuring the agent provides simple solutions when appropriate, while encouraging more thorough exploration when more advanced solutions may be necessary. 2 Scoped memory. We modify the MEMORY operator to extract different types of memories depending on the operator used. Specifically, for DRAFT and IMPROVE, it retrieves only sibling memories-the children of the artifact the agent is applying the operator to-thereby promoting diversity. This prevents overloading the context and reduces behavior indicative of mode collapse. Conversely, for DEBUG, it retrieves the entire ancestral memory of the artifact's debug chain, enabling review of prior fix attempts and avoiding "undo-redo" oscillations.
3 Think Tokens. For reasoning models, we use the operators' system prompts to explicitly encourage them to use thinking tokens for reasoning and reflection. These thoughts are stripped from the final answer-remaining invisible to other operators (e.g., memory). On average, we observe a 2× increase in completion tokens generated by the AIRA operator set (see Section F).
this section cite: []

Section: Agents
In this section, we introduce three agents that combine the operators O AIRA -proposed in Section 4.1with distinct search policies (see Appendix H for the rationale behind their selection). Each agent uses the same proxy-fitness function F (5-fold CV) and the same termination criterion τ (based on wall-clock time or artifact cap).
AIRAGREEDY. This agent employs greedy search (Section 2.3) using the O AIRA operator set. Any performance improvement of AIRAGREEDY over AIDEGREEDY directly reflects the benefit of the new operators, since the only difference between the two is the operator set.
this section cite: []

Section: AIRAMCTS.
This agent uses Monte-Carlo Tree Search (MCTS) [24] with O AIRA operator set. Our implementation of MCTS follows the canonical loop (selection, expansion, evaluation, and backup), but omits simulated roll-outs: the leaf value of expanded nodes is the proxy fitness function F:
• Selection. From the root node v 0 , descend by selecting the node with the highest UCT score
π sel (v) = arg max v∈Vt h UCT (v) where h UCT (v | u) = Q(v) + c log N (u)/(N (v) + ε),
where N (•) and Q(•) are the visit count and running mean fitness. Here, u is the parent of v.
• Expansion. Only leaves are expanded. The chosen operator from O AIRA is applied n times, creating n children. Buggy children enter an automatic DEBUG loop until fixed or the budget expires.
• Evaluation & backup. The leaf fitness is F(v ℓ ) is back-propagated to ancestors with the standard incremental update of (N, Q).
this section cite: ['b23']

Section: AIRAEVO.
The evolutionary agent keeps a population V t of fixed size n and repeats:
• Parent selection: Select individuals with probability π sel (v) = F (v)/ u∈Vt F(u).
• Reproduction: With a fixed probability, apply IMPROVE; otherwise, apply CROSSOVER. Buggy parents undergo DEBUG until they are either fixed or the debug attempt limit is reached.
• Replacement: Offspring replace the least-fit individuals in V t .
For both AIRAMCTS and AIRAEVO, we normalize all fitness values using the minimum and maximum values observed during the search process. This normalization ensures a consistent set of hyperparameters throughout the search. To select the final solution, the AIRA and AIDE agents return the one with the highest validation score.
this section cite: []

Section: Experiments and Results

this section cite: []

Section: Analyzing the Performance of the Current SoTA
The most effective search algorithms strike a balance between exploration and exploitation. In this section, we analyze the exploration-exploitation trade-offs of AIDE, the current state-of-the-art method, and then investigate how additional computation time increases its performance.
AIDE MCTS EVO 0% 10% 20% 30% 40% Medal Rate 35.23 38.18 39.77 36.36 35.45 35.00 34.55 39.61 [OAI] AIDEGreedy o1-preview [DOJO] AIDEGreedy No Mem. R1 [DOJO] AIDEGreedy R1 AIDEMCTS(c = 0.0) R1 AIDEMCTS(c = 0.25) R1 AIDEMCTS(c = 0.50) R1 AIDEMCTS(c = 0.75) R1 AIDEEVO R1 Figure 3: Searching with AIDE's operators. When limited to AIDE's operator set O AIDE , agents using more advances search policies (e.g., MCTS, evolutionary algorithms) gain no advantage, underscoring the operator set as the bottleneck. 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Wall clock time (hrs) 0% 10% 20% 30% 40% 50% 60% 70% Medal Rate [DOJO] AIDEGreedy o3 (Validation) [DOJO] AIDEGreedy o3 (Test) [DOJO] AIDEGreedy R1 (Validation) [DOJO] AIDEGreedy R1 (Test) 0 5 10 15 20 Wall clock time (hrs) 0% 10% 20% 30% 40% 50% Medal Rate [DOJO] AIDEGreedy R1 AIRAEvo R1 AIRAGreedy R1 AIRAMCTS R1 [DOJO] AIDEGreedy o3 AIRAEvo o3 AIRAGreedy o3 AIRAMCTS o3 The three main factors that influence this balance are memory, the operator set, and the selection policy. Memory structures prior knowledge-storing promising solutions and tracking which regions of the solution space have been sampled-to inform subsequent decisions. This information can bias the search toward exploration by encouraging diversity or toward exploitation by discouraging it. Operators then apply controlled transformations to existing solutions: for instance, random mutations introduce diversity, targeted refinements exploit known strengths, and recombinations merge features from multiple candidates. Finally, the selection policy balances exploiting high-quality regions of the solution space with exploring less-tested areas by determining how to allocate computational resources. A non-greedy selection policy, such as in MCTS, periodically directs resources toward branches that may appear suboptimal, with the goal of uncovering better solutions.
this section cite: []

Section: What is the effect of memory?
To quantify the impact of the MEMORY operator, we conduct a controlled ablation comparing the performance of AIDE with and without the MEMORY operator enabled. As shown in Fig. 3, both variants achieve nearly identical mean medal rates. This suggests that memory is not a driving factor behind AIDE's strong performance. What is the effect of exploration at the search level with AIDE's operators? AIDE uses a selection policy that does not explore and always greedily selects the most promising candidate. To evaluate the impact of the search policy as a modulator of the exploration-exploitation tradeoff when searching using AIDE's operators, we replaced the operators in AIRAMCTS with that of AIDE and varied the UCT exploration constant c UCT ∈ {0, 0.25, 0.75}. The c UCT , which controls exploration in MCTS, allows us to isolate the effect of search-level exploration on downstream performance. The resulting medal rates (Fig. 3) showed only marginal differences across all c UCT values, supporting our hypothesis that search-level exploration is constrained by the interaction between the operator set and the search policy. We observe the same limitation when performing the same operator replacement in AIRAEVO.
this section cite: []

Section: What is the performance profile of AIDE?
To examine the improvement gains over time, we plot AIDE's anytime performance, which is the average medal rate achieved by the agent if the search was
Any Medal Silver Gold 0% 10% 20% 30% 40% 50% Medal Rate 35.23 30.23 20.00 45.91 40.45 27.27 39.77 32.05 23.41 39.55 34.09 26.82 42.73 34.77 24.55 43.64 35.00 26.36 45.45 36.82 27.05 47.73 42.73 28.64 47.05 36.36 25.68 47.27 42.27 30.91
this section cite: []

Section: [OAI] AIDEGreedy o1-preview [DOJO] AIDE * Greedy o1-preview [DOJO] AIDEGreedy R1 [DOJO] AIDEGreedy o3 AIRAEvo R1 AIRAEvo o3 AIRAGreedy R1 AIRAGreedy o3 AIRAMCTS R1 AIRAMCTS o3
Figure 5: Medal rates on MLE-bench Lite. Performance is shown for three medal categories: any medal, silver medals and above, and gold medals only. Error bars represent 95% confidence intervals computed using stratified bootstrapping.
to terminate at time t. The results, summarized in Fig. 4a, suggest that while the agent's perceived performance (validation score) continues to improve over time, the true test performance plateaus, even slightly decreases over time. These findings suggest that overfitting might be a fundamental limitation to the agent's performance. We further investigate this in Section 5.3.
this section cite: []

Section: AIRA Beyond Greedy
Based on the observation that search policies yield no performance gain with AIDE's operators, this section is divided into two parts: 1 assessing the effectiveness of our improved operators (see Section 4.1), 2 examining their interplay with the more advanced search policies. The results are summarized in Fig. 5, with a detailed breakdown of performance for each task in Section L.
this section cite: []

Section: Evaluating the environment.
But first, we highlight the benefits from AIRA-dojo. Specifically, our baseline agent implementation, AIDEGREEDY o1-preview, operating in the AIRA-dojo environment, improves the medal rate from 35.2 % to 45.9 % compared to the reported results [4]. This corresponds to state-of-the-art performance with a relative improvement of 30 % in medal rate. It is notable that AIDEGREEDY o1-preview, outperforms both AIDEGREEDY R1 and AIDEGREEDY o3, despite o3 being the newest model in the series. While evaluating all agents with o1-preview would have been informative, we were only able to complete the AIDEGREEDY experiment before the model was discontinued.
Evaluating the operator sets. AIDEGREEDY and AIRAGREEDY employ the same search policy but differ in their operator sets. Comparing their performance isolates the effect of the operators. AIRAGREEDY outperforms AIDEGREEDY (45.5 % vs. 39.8 %), representing a 14 % relative improvement and underscoring the importance of operators for performance.
this section cite: ['b3']

Section: Evaluating search policies.
Equipped with an improved set of operators, we now explore whether combining them with advanced search methods can further enhance performance. The results (Fig. 5) show that for R1, AIRAMCTS achieves the best performance, achieving state-of-the-art performance on MLE-bench Lite with an average medal rate of 47%. Both AIRAMCTS and AIRAGREEDY achieve silver-or-above medal rates (≈ 36.5%), outperforming the baseline by 4.5 absolute points.
In terms of gold medals, AIRAGREEDY performs best, reaching 27%. For o3, AIRAMCTS and AIRAGREEDY perform comparably in terms of any medal and silver-or-above medal rates, while AIRAMCTS outperforms AIRAGREEDY in gold medals by 2.2 percentage points. Although agents using R1 and o3 achieve similar overall medal rates, those using o3 perform better when considering gold and silver-or-above medal rates. Specifically, AIRAMCTS increases the gold medal rate from 25.68 % with R1 to 30.91 % with o3, a relative increase of 20 %.
Collectively, what stands out is that all search policies combined with AIRA operators outperform AIDEGREEDY. Furthermore, the rankings between agents using the AIRA operators with different search strategies differs significantly from that observed in Section 5.1.
Finally, Fig. 4b shows the anytime performance of each agent over the 24-hour window defined by MLE-bench. For conciseness, we will focus our discussion on performance with R1, as the results
this section cite: []

Section: AIDEgreedy AIRAEVO AIRAGreedy
AIRAMCTS 0% 20% 40% 60% Medal Rate 39.8 51.4 43.0 56.1 45.5 54.5 47.0 56.6 56.4 55.0 60.5 56.4 Intermediate/Final Node Selection Val/Val Val/Test Test/Test 2 4 6 8 k 30% 35% 40% 45% 50% 55% Medal Rate AIDEGreedy 2 4 6 8 k AIRAMCTS 2 4 6 8 k AIRAGreedy 2 4 6 8 k AIRAEvo Random-k Top-k Oracle with o3 follow a similar pattern. We observe that the rankings between agents change over time.
For example, at the 3-hour mark, the performance gap between AIRAGREEDY and AIRAMCTS is notable. This gap narrows by the 10-hour mark, where all agents perform similarly. Meaningful divergence in performance appears only after 19 hours. These trends reflect the interaction between rankings and the resources provided, as shown in Figure 7.
We further examine the effect of extended compute time on performance by running experiments for up to 5 days (Appendix C), showing that our agents continue to improve beyond the 24-hour period. To illustrate differences in search behavior, Appendix M presents representative search trees from our experiments for each agent, which may help clarify the strengths and weaknesses of each method.
this section cite: []

Section: The Generalization Gap: Searching with a Proxy Evaluation
Agents steer their search using the validation scores of candidate solutions, but ultimately, performance is measured on a held-out test set. Due to finite sample effects, the validation score is not perfectly predictive of performance on the test set-a discrepancy known as the generalization gap.
In this section, we measure the impact of this generalization gap on the agents' performance. We further quantify whether this impact is more detrimental during the intermediate node selection in the search process or the final (submitted) node selection.
How large is the impact of the generalization gap on performance? We begin by comparing two extremes: VAL/VAL-both intermediate and final node selection use the validation score (standard practice); TEST/TEST-an oracle baseline using the true test score at both stages. As shown in Fig. 6a, searching based on the test score instead of the validation score improves performance by 9.4 % and 12.4 % for AIRAMCTS and AIRAEVO. The gap is even higher for the agents using a greedy search policy, reaching 15 % for AIRAGREEDY and 16.6 % for AIDEGREEDY. For the rest of this section, we investigate the nature of this gap in performance and how to close it.
How much of the gap can be attributed to final node selection? To answer this question, we consider the following two settings: VAL/VAL; and VAL/TEST-the intermediate node selection uses the validation score, while the final node selection is made based on the test scores. The results (Fig. 6a) indicate that by selecting the best node in a search graph constructed without privileged access, i.e., based on validation signal, all agents achieve a performance boost of 9 to 13 absolute points. Crucially, comparing VAL/TEST with TEST/TEST shows that oracle final-node selection eliminates the gap between the standard (VAL/VAL) and oracle (TEST/TEST) settings for both AIRAMCTS and AIRAEVO. Even for the agents implementing the simpler greedy policy, selecting the best node in the final search graph closes more than 60 % of the gap. These findings highlight robust final-node-selection strategies as a promising path to higher performance.
this section cite: []

Section: Bridging the gap through multiple submissions.
A straightforward remedy is to hedge against validation score noise by selecting multiple promising nodes. Specifically, among the top-k nodes ranked by validation score in the search graph, we report the test score of the best-performing one. As a baseline, we repeat the process for a set of random-k nodes, and summarize the results in Fig. 6b. We highlight the following observations: 1 the larger gap between top-k and random-k in agents using non-greedy algorithms, AIRAEVO and AIRAMCTS, suggest greater diversity in their search graphs; 2 with as little as 3 top-k submissions agents achieve an additional 10 % of performance; 3 the top validation nodes are informative of the best performing nodes.
this section cite: []

Section: Related Works
Scaling Search with LLMs. Integrating search with LLM-based generators has gained traction across domains such as coding [1], math [52], and planning [16,15]. In practice, increased test-time compute through algorithms like best-of-N [45], beam search [16], MCTS [1,15], and evolutionary search [39,17] often improves performance beyond the architectural and size constraints of LLMs [41]. Our results highlight that these performance gains depend critically on the interplay between search components (see Section A)-an aspect that prior work has largely overlooked.
Automating ML Engineering and Scientific Discovery. Traditional approaches like AutoML [12,43,33] and Neural Architecture Search (NAS) [54,29,38] automate ML by searching over predefined, expert-designed configuration spaces, often via brute-force or heuristic methods [2,8,28]. In contrast, recent advances in LLMs enable more open-ended design. AIDE [23] treats ML engineering as LLMguided code optimization via tree search. AI Scientist v2 [49] extends this paradigm, automating the entire research pipeline using agentic LLMs. Similar techniques have been applied to software engineering [1], algorithm and reward design [39,31,11,17,7], and even natural sciences [3,42]. Existing systems often combine search procedures, operators, and evaluation in ways that make it challenging to understand which components drive performance improvements. To address this, we separate these components and look at how each one-and their interactions-can be optimized better. Concurrent work, R&D-Agent [51], which addresses the same problem setting, achieved impressive results on MLE-Bench. Our results fall within their reported standard deviation. We note that their experiments use at most 6 seeds, and as discussed in Section J, the limited number of seeds may introduce variation in the conclusions. Finally, our approach differs from methods such as Agent K [14], which uses long-term memory across multiple Kaggle competitions. In contrast, we focus on addressing each competition independently.
this section cite: ['b0', 'b51', 'b15', 'b14', 'b44', 'b15', 'b0', 'b14', 'b38', 'b16', 'b40', 'b11', 'b42', 'b32', 'b53', 'b28', 'b37', 'b1', 'b7', 'b27', 'b22', 'b48', 'b0', 'b38', 'b30', 'b10', 'b16', 'b6', 'b2', 'b41', 'b50', 'b13']

Section: AI Research Frameworks.
Most existing benchmarks for machine learning or research engineering, such as MLGym [32], RE-bench [48], and MLAgentBench [19], come with their own frameworks. Our approach is most similar to Inspect [21], a framework which provides an abstraction that makes minimal assumptions about the agent and evaluation design, clearly separating the two. This flexibility in agent design is essential for enabling rigorous comparisons across a wide range of agents-from simple LLM-based agents with tool access to scaffolds combining algorithmic and LLM-based components-within the same environment. The environment plays a critical role in performance (see Section 5.2). However, unlike Inspect, AIRA-dojo focuses strongly on long-running research tasks, which influences our environment design choices. For example, prior work typically uses Docker to containerize agents' workspaces, but Docker is unsupported on most HPC clusters due to its reliance on root privileges and lack of seamless integration with common HPC resource managers. This limitation hinders scalability, addressed through our Apptainer (see Section 3.2).
this section cite: ['b31', 'b47', 'b18', 'b20']

Section: Conclusion
We conceptualize the design of AI research agents as a combination of two axes: search policy and operators. This formulation allowed us to perform a systematic investigation of the interplay between the two, highlighting how the operator set can act as a bottleneck to performance improvements. Based on this finding, we designed an enhanced operator set and constructed agents that combines these operators with several search strategies: Greedy, Monte Carlo Tree Search (MCTS), or Evolutionary Search. Our best-performing agent achieves a new state of the art on MLE-bench, increasing the success rate of winning a Kaggle medal from 39.6 % to 47.7 %. Further, we analyzed the role of the generalization gap in node evaluation. Our findings indicate systematic overfitting: selecting the final solution from the search graph based on its test score instead of its validation score would increase the medal rate by 9 to 13 % (absolute scale), depending on the search strategy. This highlights that robust final-node selection is a promising avenue for improving performance. The hierarchical diagram in Fig. 7 illustrates key insights into the preconditions for improving agent performance through additional compute. Agents benefit from scaling search only when performance gains are not limited by the environment in which the agent operates, the quality of the evaluation signal guiding the search, the capability of the operators performing the search, or the search policy itself.
Specifically, if the agent is provided with 10× more compute resources (e.g., 10 GPUs for 24 hours), the environment must allow the agent to effectively leverage these resources. Without a sufficiently high-quality evaluation signal-the ability to accurately assess solution quality-the direction of improvement will be unclear, which would undermine the search process. Finally, a successful search requires capable operators that effectively perform their functions and a search policy that allocates compute efficiently to the most promising regions of the search space and appropriate operators.
Overall, this work underscores the importance of a strong foundation in algorithm design and infrastructure to ensure that scaling search translates into downstream performance improvements.
this section cite: []

Section: References
Ref_id:b0 Title: SWE-search: Enhancing software agents with monte carlo tree search and iterative refinement Year: (2025)
Ref_id:b1 Title: Random search for hyper-parameter optimization Year: (2012)
Ref_id:b2 Title: Autonomous chemical research with large language models Year: (2023)
Ref_id:b3 Title: MLE-bench: Evaluating machine learning agents on machine learning engineering Year: (2025)
Ref_id:b4 Title: Experimental results on the crossover point in random 3-sat Year: (1996)
Ref_id:b5 Title: Evolutionary computation: a unified approach Year: (2017)
Ref_id:b6 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b7 Title: Simple and efficient architecture search for convolutional neural networks Year: (2017)
Ref_id:b8 Title: Gemini: A family of highly capable multimodal models Year: (2025)
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: OMNI-EPIC: Open-endedness via models of human notions of interestingness with environments programmed in code Year: (2025)
Ref_id:b11 Title: Auto-sklearn 2.0: hands-free automl via meta-learning Year: (2022-01)
Ref_id:b12 Title: Rlef: Grounding code llms in execution feedback with reinforcement learning Year: (2025)
Ref_id:b13 Title: Large language models orchestrating structured reasoning achieve kaggle grandmaster level Year: (2024)
Ref_id:b14 Title: Reasoning with language model is planning with world model Year: (2023-12)
Ref_id:b15 Title: Saycanpay: Heuristic planning with large language models using learnable domain knowledge Year: (2024)
Ref_id:b16 Title: REvolve: Reward evolution with large language models using human feedback Year: (2025)
Ref_id:b17 Title: Branching rules for satisfiability Year: (1995)
Ref_id:b18 Title: Mlagentbench: Evaluating language agents on machine learning experimentation Year: (2023)
Ref_id:b19 Title: Cursor: The ai code editor Year: (2025)
Ref_id:b20 Title: Inspect AI: Framework for Large Language Model Evaluations Year: (2025-05)
Ref_id:b21 Title: Megascale: scaling large language model training to more than 10,000 gpus Year: ()
Ref_id:b22 Title: AIDE: AI-Driven Exploration in the Space of Code Year: (2025)
Ref_id:b23 Title: Bandit based monte-carlo planning Year: (2006)
Ref_id:b24 Title: Revisiting Reliability in Large-Scale Machine Learning Research Clusters Year: (2025-03)
Ref_id:b25 Title:  Year: (2021-04-03)
Ref_id:b26 Title:  Year: (1987)
Ref_id:b27 Title: Hyperband: A novel bandit-based approach to hyperparameter optimization Year: (2018)
Ref_id:b28 Title: DARTS: Differentiable architecture search Year: (2019)
Ref_id:b29 Title: Large language model-based agents for software engineering: A survey Year: (2024)
Ref_id:b30 Title: Discovering preference optimization algorithms with and for large language models Year: (2024)
Ref_id:b31 Title: MLGym: A New Framework and Benchmark for Advancing AI Research Agents Year: (2025)
Ref_id:b32 Title: TPOT: A Tree-Based Pipeline Optimization Tool for Automating Machine Learning Year: (2019)
Ref_id:b33 Title: Gpt-4o system card Year: (2024)
Ref_id:b34 Title: Introducing o3 and o4 mini Year: (2024)
Ref_id:b35 Title: Gpt-5 system card Year: (2025)
Ref_id:b36 Title: Tool learning with foundation models Year: (2024-12)
Ref_id:b37 Title: Regularized evolution for image classifier architecture search Year: (2019)
Ref_id:b38 Title: Mathematical discoveries from program search with large language models Year: (2024)
Ref_id:b39 Title: Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers Year: (2024)
Ref_id:b40 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b41 Title: The virtual lab: Ai agents design new sars-cov-2 nanobodies with experimental validation Year: ()
Ref_id:b42 Title: Auto-weka: combined selection and hyperparameter optimization of classification algorithms Year: (2013)
Ref_id:b43 Title: Exploration and exploitation in evolutionary algorithms: A survey Year: (2013-07)
Ref_id:b44 Title: Self-consistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b45 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b46 Title: Livebench: A challenging, contamination-free LLM benchmark Year: (2025)
Ref_id:b47 Title: Re-bench: Evaluating frontier ai r&d capabilities of language model agents against human experts Year: (2024)
Ref_id:b48 Title: The AI Scientistv2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search Year: (2025)
Ref_id:b49 Title: Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b50 Title: R&d-agent: Automating data-driven ai solution building through llm-powered automated research, development, and evolution Year: (2025)
Ref_id:b51 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b52 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b53 Title: Neural architecture search with reinforcement learning Year: (2017)
