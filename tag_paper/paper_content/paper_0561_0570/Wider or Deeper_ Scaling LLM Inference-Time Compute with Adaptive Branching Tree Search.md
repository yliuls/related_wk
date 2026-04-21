Title: Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search
Abstract: Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in real tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling. Code is available at: https://github.com/SakanaAI/treequest.

Section: Introduction
Recent work has shown that inference-time scaling, namely allocating more computation at inference time, can markedly boost the performance of large language models (LLMs) on complex tasks. As outlined in Section 2, existing approaches to inference-time scaling fall into three broad categories:
(1) post-training fine-tuning, (2) reward-guided chain-of-thought (CoT) generation, and (3) multiple answer generation. In this paper, we focus on the third category. The multiple answer generation approach repeatedly queries an LLM at non-zero temperature to produce a set of candidate outputs and then selects the most promising one. This approach enhances the LLM's problem-solving abilities on-the-fly, without further training [1][2][3][4][5][6][7][8][9][10][11]. Because it is orthogonal to the other two families, it can be seamlessly combined with them.
The most widely successful approach in this category is repeated sampling, which includes techniques such as best-of-n sampling, majority voting, and self-consistency [2,3,12]. In repeated sampling, an LLM at non-zero temperature generates multiple candidate outputs independently from the same initial prompt, and a final solution is selected, typically by a simple heuristic. This paradigm has proved effective on challenging benchmarks, including coding competitions [1,3] and ARC-AGI [13]. The strategy leverages the diverse and vast output space exposed by LLM generation, and sampling more responses increases the odds that one of them is high-quality. The empirical success of repeated sampling underscores that harnessing this diversity is central to effective inference-time scaling.
However, repeated sampling focuses exclusively on exploration and lacks an explicit mechanism for exploitation. In certain real-world scenarios, one can obtain external feedback on a candidate solution. For instance, in coding tasks, one can run tests to evaluate the correctness of generated programs and gather feedback on how to improve them [4,5,14]. In such settings, it is natural to select promising solutions and refine them based on available feedback, which repeated sampling alone cannot accomplish effectively.
Several approaches [6,7,[9][10][11]15] have been proposed for exploration and exploitation in such multi-turn settings, but the majority were designed before the power of inference-time scaling was fully recognized. Consequently, these methods use a fixed "width", i.e., they treat the number of answers generated from a single prompt as a fixed hyperparameter. For example, methods based on standard Monte Carlo Tree Search (MCTS) use a fixed branching factor (i.e., the number of child nodes per state) as a hyperparameter [9][10][11]15]. As demonstrated by the success of repeated sampling, effective inference-time scaling requires leveraging a diverse and vast output space, thus, providing substantial evidence that a fixed width hinders scaling.
In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inferencetime framework that generalizes repeated sampling with multi-turn exploration and exploitation (Figure 1). The main technical challenge is to introduce unbounded branching into MCTS. Unlike traditional MCTS, AB-MCTS does not fix the width as a static hyperparameter. Instead, at each node of the search tree, AB-MCTS adaptively decides whether to explore ("go wider") by generating new candidate responses or exploit ("go deeper") by refining existing ones, leveraging external feedback signals. Under the hood, we formalize our decision process via Bayesian posterior updates, ensuring each expansion balances exploration and exploitation in a principled manner. This design naturally extends repeated sampling, allowing us to harness the diverse and vast output space of LLMs when necessary. Consequently, our framework provides a powerful mechanism for balancing exploration and exploitation in the context of LLM inference-time scaling.
We evaluated AB-MCTS on complex coding and machine learning engineering benchmarks [1,16], as well as ARC-AGI [17], using frontier models such as GPT-4o [18] and DeepSeek-V3 [19], in a scenario that scales up inference-time compute by allowing multiple generation calls for each task instance. Under the same computational budget, AB-MCTS achieved better results than previous approaches, such as repeated sampling and standard MCTS.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b1', 'b2', 'b11', 'b0', 'b2', 'b12', 'b3', 'b4', 'b13', 'b5', 'b6', 'b8', 'b9', 'b10', 'b14', 'b8', 'b9', 'b10', 'b14', 'b0', 'b15', 'b16', 'b17', 'b18']

Section: Contributions. 1
⃝ We highlight the challenge of effectively incorporating unbounded branching into tree search. This is pivotal for combining the power of the diverse and vast output space of LLMs, a cornerstone of inference-time scaling, with solution refinement. 2 ⃝ To address this challenge, we introduce AB-MCTS, which systematically decides whether to "go wider" or "go deeper." We present two variants, AB-MCTS-M and AB-MCTS-A, based on different principles, each offering distinct trade-offs. 3 ⃝ In a practical setting using frontier models and real-world complex tasks, we show that AB-MCTS outperforms existing methods.
this section cite: ['b2']

Section: Related Work
Inference-Time Scaling by Post-Training Fine-Tuning. Recent post-training work, exemplified by OpenAI o1/o3 [20,21], uses reinforcement learning or supervised CoT fine-tuning to deepen LLM reasoning and boost single-answer quality [20][21][22][23][24][25]. Our approach instead generates many candidates and refines them with external feedback, pursuing a complementary objective.
Inference-Time Scaling via Reward-Guided CoT. Reward-guided CoT scales inference by searching one step (typically a sentence) at a time [26][27][28][29][30][31][32][33][34]. Primarily for math tasks, it aims to improve single-answer quality, making it orthogonal to our multiple-answer generation approach.
Inference-Time Scaling by Multiple Answer Generation. Since the community has come to appreciate the power of inference-time scaling, the strategy that has been studied widely is repeated sampling, in which the model generates many candidate answers and selects the best one [1][2][3]35]. Although empirically strong and widely used, repeated sampling leaves obvious room for improvement because it does not refine its candidates using external feedback [4,5]. Before the era of large-scale inference-time compute, a variety of task-specific strategies were proposed for relatively small scales; examples include tree expansions directed by LLMs [6] and Bayesian methods [7]. LATS [9], RAP [15], SWE-Search [10], and RepoUnderstander [11] combine LLMs with MCTS, primarily targeting sequential decision making. In this context, nodes represent states and edges represent actions, which may involve interaction with an environment. LATS utilizes API calls and code execution as actions to solve tasks. RAP addresses the process of solving block-moving puzzles and mathematical word problems step-by-step. SWE-Search explores sequences of actions such as searching, editing, and running tests to resolve issues within a software repository. RepoUnderstander employs MCTS for exploration on a repository knowledge graph. The application of LATS to coding tasks [9, Section 5.2] aligns with the context of multiple-answer generation in this paper and corresponds to what we refer to as "standard MCTS" in our experiments.
this section cite: ['b19', 'b20', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b0', 'b1', 'b2', 'b34', 'b3', 'b4', 'b5', 'b6', 'b8', 'b14', 'b9', 'b10']

Section: Progressive Widening in MCTS.
Progressive widening (PW) [36,37] is a classic technique that gradually increases actions considered per node. It was designed for games with unique actions and no side information for untried moves, relying on visit-count heuristics. Complementary to PW, Sokota et al. [38] propose "abstraction refining", which groups similar successors using a decreasing similarity threshold and shows advantages over PW under equal simulation budgets in stochastic domains. Our approach differs as new branches are sampled from the same LLM. This homogeneity in generation allows a principled statistical rule for choosing between widening and deepening.
this section cite: ['b35', 'b36', 'b37']

Section: Method

this section cite: []

Section: Preliminaries
First, we introduce the setup and notation, with detailed elaboration provided in Appendix A.1. We consider a setting where an LLM, represented by a function f LLM , receives a textual prompt t in containing (1) task instructions with optional few-shot examples, and/or (2) previously generated outputs along with external feedback, and generates an answer t out = f LLM (t in ). A scoring function R then evaluates an answer t out to produce a score r = R(t out ), where higher scores indicate better performance. We typically assume that the score r is normalized to the range [0, 1], but our framework allows for arbitrary ranges as well. Our goal is to find an output t out that attains a high score r under limited calls to the LLM at inference time. Such tasks arise, for example, in code generation, where the correctness or quality of the output can be quantified; for instance, R may execute the generated code and return the fraction of test cases passed. In some cases, the true score evaluator may be inaccessible (e.g., hidden test cases), so we assume we have access to some surrogate or partial evaluator R, such as a public test evaluator, during the search. We aim to leverage this evaluator to guide an efficient search for better solutions.
this section cite: []

Section: Adaptive Branching MCTS MCTS for LLM-based Answer Generation.
We perform the answer search by constructing a search tree T , in which each non-root node N is associated with an LLM-generated answer to a given task. Our goal is to construct T so that it contains answers with scores as high as possible. Here, a 1 leads to a set of child nodes with higher scores, causing a peak at larger r. As more child samples are collected, the variance of the distribution decreases.
this section cite: []

Section: AB-MCTS-M Example Tree
Algorithm 1 Adaptive Branching MCTS
1: function AB-MCTS(n nodes ) 2: T ← INITIALIZETREE( ) 3: for n = 1, . . . , n nodes do 4: N ← SELECTEXPANSIONTARGET(T ) ▷ Step 1. Select an expansion target 5: Nnew ← EXPAND(N , T ) ▷ Step 2. Expand the selected node to generate a child 6: SCOREBACKUP(Nnew, T ) ▷ Step 3. Backup the score from the generated node 7: return SELECTBEST(T ) 8: function SELECTEXPANSIONTARGET(T ) 9: N ← GETROOT(T ) 10: while not ISLEAF(N ) do 11: Nnext ← SELECTCHILD(N , T ) ▷ Detailed in Sections 3.3 and 3.4 12: if ISGENNODE(Nnext) then ▷ If a GEN node is selected, branch off from the node 13: break 14: N ← Nnext 15: return N
For this purpose, we employ MCTS, formulated iteratively as follows. Starting from a single root node, we perform n nodes iterations, each adding one new node, resulting in a total of 1 + n nodes nodes. Each iteration has three steps: (1) Selection, where we select a node N for expansion; (2) Expansion, where we expand N by generating a new answer from node N , creating a new child node N new and appending it to N . Specifically, if N is the root, the new answer is directly generated from the task prompt; if N is non-root, the new answer refines the answer associated with N , using external feedback; and (3) Score backup, where we propagate the score of N new up toward the root of the tree T . We adopt different backup rules in our proposed methods (see Section 3.3 and 3.4). In our setting, no separate rollout is needed, since each node's score r can be evaluated directly once an output is generated. After n nodes iterations, we select the best node based on a chosen criterion.
In standard MCTS, only leaf nodes are selected and expanded, (i.e., each node is expanded at most once), and the expansion adds a fixed number of child nodes. However, since each query to an LLM at non-zero temperature can yield different outputs from the same prompt, the branching factor is theoretically infinite. To accommodate such unbounded branching, we relax the standard MCTS constraints and allow selection and expansion of non-leaf nodes. Moreover, recent studies [3] suggest that drawing many outputs from the same prompt at non-zero temperature can improve performance. Allowing unbounded branching enables us to fully exploit these varied samples, whereas restricting the branching factor could miss correct answer generations and undermine overall performance.
this section cite: ['b2']

Section: Adaptive Branching via the GEN Node.
To fully leverage the potential performance improvement from unbounded branching, we allow nodes that have already been expanded once to be expanded again and further branched, unlike in standard MCTS. To explicitly represent the action of generating new child nodes, we introduce a GEN node. Every node N (including newly expanded ones during iterations) has a GEN node as a child. When the GEN node with parent node N is selected during the selection step, we expand N by adding a new child node. Algorithm 1 outlines this approach, called Adaptive Branching Monte Carlo Tree Search (AB-MCTS).
The only remaining component we need is a selection policy, including when to select a GEN node. We propose two algorithms with different selection policies: AB-MCTS-M (Mixed model) and AB-MCTS-A (node Aggregation). Both follow the overall procedure in Algorithm 1 and use Thompson sampling to balance exploration and exploitation.
The UCT score is inapplicable to our AB-MCTS because GEN nodes make the problem fundamentally different from a standard multi-armed bandit problem, for which UCT was designed. In standard MCTS, the arms (branches) are static. In contrast, the GEN node in AB-MCTS dynamically generates new arms. This special problem setting, where arms are generated on the fly, prevents the direct application of UCT. We, therefore, adopt a Bayesian probabilistic model. This enables Thompson sampling based on the posterior distribution and obviates the need for complex UCB-style confidence bound analysis.
Thompson Sampling for Node Selection. In our proposed methods, we employ a Bayesian approach with Thompson sampling for node selection. Here, we employ Thompson sampling because GEN nodes do not have child nodes, making it impossible to compute their UCT scores. In addition, Thompson sampling has the advantage of allowing node expansion in parallel. This is particularly beneficial when evaluating node scores is time-consuming, as in the case of MLE-Bench (See Appendix B.1 for MLE-Bench details). Concretely, during SelectChild step at line 11 of Algorithm 1, we employ Thompson sampling to decide between expanding a GEN node or selecting from existing child nodes at node N . Let N be a node with potential actions
A N = {a 0 , a 1 , . . . , a n child },
where the action a 0 corresponds to choosing the GEN node, and a 1 , . . . , a n child correspond to choosing the already-existing child nodes. Suppose P N (r | a i ) is the posterior predictive distribution of the score r for an eventually expanded new node (N new at line 5 of Algorithm 1) if we choose the action a i at node N . Then Thompson sampling proceeds by, 1. Calculate P N (r | a j ) for each action a j at node N . 2. Draw scores r Nnew,aj from P N (r | a j ) for each action a j . 3. Select â = arg max aj ∈A N r Nnew,aj .
This three-step process corresponds to a single call to SelectChild.
A key question is how to perform step 1, i.e., how to model and calculate P N (r | a j ) for all a j , in particular for j = 0 (i.e., GEN node). We address this with two strategies: a mixed Bayesian model (AB-MCTS-M) and a node aggregation method (AB-MCTS-A). In both cases, we model the score probability distributions by Bayesian posterior predictives, but with different statistical models.
this section cite: []

Section: AB-MCTS-M: Adaptive Branching MCTS with Mixed Model
To model P N (r | a j ), we employ a node-specific mixed model fitted individually at each node N . That is, we fit a separate model for each N every time SelectChild in Algorithm 1 is invoked. Denoting r Nnew,aj ∼ P N (r | a j ) as a score of an eventually expanded node N new if we choose an action a j at N , our mixed model is given as:
r Nnew,aj = α j + σ y ϵ Nnew , α j = µ α + σ α ϵ j , ϵ Nnew ∼ N (0, 1), ϵ j ∼ N (0, 1),(1)
Here, α j is a "group-level" intercept capturing the quality of the base solution at N j , while σ y ϵ Nnew represents per-instance noise. To fit this model, we place priors on the hyperparameters (µ α , σ α , σ y ) and employ Markov Chain Monte Carlo (MCMC) to sample from their posterior distribution. The GEN node (action a 0 ) is treated as a newly introduced group without its own direct observations. However, its group-level intercept α 0 is inferred not from the prior alone but rather from the posterior distribution over µ α and σ α , which is informed by the other observed data. We assume that even after multiple refinement stages, the quality associated with the answer at node N j continues to be captured by this shared parameter (see Appendix A.
this section cite: []

Section: for further details).
Algorithm Outline. To model P N (r | a j ), AB-MCTS-M assigns each subtree under N j , denoted as T sub (N j ), as a distinct group j (see Figure 2 for example subtree). The mixed model leverages observed scores from these groups to compute the posterior predictive distributions of expected scores for new nodes generated from each group (See Figure 2 for a schematic illustration). We sample the scores from all the groups (the GEN node and T sub (N j )) using calculated posterior predictives. If the GEN node's sampled score is highest, we call f LLM to generate a new child node. Otherwise, we choose the child node N j with the highest score and continue the sampling step.
this section cite: []

Section: Score Backup Mechanism.
When a new node N is created, its observed score is added to the histories of N and its ancestors. This cumulative record is used to update the posterior distributions in the mixed model. The observed score is not backed up to a GEN node, but it indirectly influences the GEN node's score probability distribution through the shared parameters in the mixed model (see Appendix A.5 for a detailed walkthrough).
this section cite: []

Section: AB-MCTS-A: Adaptive Branching MCTS with Node Aggregation
In AB-MCTS-M, during the selection step at N , we use a mixed model that shares statistical strength across groups through the shared model parameters. In contrast, AB-MCTS-A is designed in the same spirit as the standard UCT-based MCTS, and there are no shared model parameters among the different actions. This design simplifies the statistical modeling and makes the computation more lightweight compared to AB-MCTS-M.
The major problem is how to back up scores to GEN nodes. Since the generated node is not attached as a child to a GEN node, it makes the backup of scores difficult to define. Here, we introduce a CONT node at the same tree level as all the GEN nodes (see Figure 3). Intuitively, the CONT node represents the action of continuing refinement from the current answer at node N , rather than generating a new node. By explicitly separating these two actions-generating new answers (GEN) and refining existing answers (CONT)-we create a clear path for score propagation. Specifically, the score of the expanded node is first backed up to the GEN node, and since all ancestors of the GEN node are either nodes with LLM answers or CONT nodes, the score subsequently does not propagate through other GEN nodes (see Figure 3 for an example tree). Algorithm Outline. AB-MCTS-A aggregates all child nodes under a single CONT node, which represents refinements from existing child nodes (see Figure 3). We model each node's score probability in a Bayesian framework and perform Thompson sampling on posterior predictives to decide between generating a new child (GEN) or refining an existing one (CONT). In contrast to AB-MCTS-M, we do not use shared parameters among different node probability distributions.
this section cite: []

Section: AB-MCTS-A Example Tree
To model P N (r | a j ), we utilize exponential family distributions with conjugate priors, enabling analytical and efficient posterior updates. We employ two variants:
1. AB-MCTS-A (Gaussian), using a normal-inverse-χ 2 prior for unbounded scores:
P N (r | a j ) = p(r | {r k } K k=1 ) = N (r | m, σ 2 κ )χ -2 (σ 2 | ν, τ2
), and 2. AB-MCTS-A (Beta), using a Beta prior for scores in [0, 1]:
P N (r | a j ) = p(r | {r k } K k=1 ) = B(r | α, β),
where r k represents the scores backed up to the node N j (GEN node, CONT node or LLM-generated child nodes; see Figure 3 for example tree), where m, κ, ν, τ , α, β are determined from observed scores r k and updated as these scores are backed up. The detailed parameter update rules are given in Appendix A.4.
this section cite: []

Section: Score Backup Mechanism.
During score-backup operations, the expanded node score is backed up to the GEN node which led to the expansion of that node and the GEN node's ancestors (see Appendix A.5 for a detailed walkthrough). As we can see from Figure 3, a GEN node's ancestors include only generated nodes and CONT nodes, so the score is backed up to a GEN node only from Table 1: Performance of AB-MCTS against baselines across benchmarks and models. This table compares AB-MCTS with the baseline methods. Evaluations were performed on LiveCodeBench, CodeContest, and ARC-AGI using GPT-4o and DeepSeek-V3 with a maximum generation budget (2 7 ). Each entry provides a performance score (higher values are better) and its corresponding rank (in parentheses, 1st is best). The "Avg. Rank" column shows the average rank across all settings.
this section cite: []

Section: LiveCodeBench

this section cite: []

Section: CodeContest ARC-AGI

this section cite: []

Section: Method GPT-4o DeepSeek-V3 GPT-4o DeepSeek-V3 GPT-4o DeepSeek-V3 Avg. Rank
Repeated Sampling 37.8 ± 0.5 (4) 40.7 ± 1.9 (6) 37.9 ± 0.3 (4) 43.2 ± 0.9 (5) 15.0 ± 1.0 (1) 18.6 ± 1.0 (1) 3.5 Sequential Refinement 37.8 ± 2.4 (4) 41.6 ± 0.6 (5) 30.1 ± 0.3 (6) 41.6 ± 0.9 (6) 8.7 ± 0.9 (6) 10.0 ± 0.6 (6) 5.5 Standard MCTS 36.7 ± 1.0 (6) 43.2 ± 2.1 (1) 37.5 ± 0.0 (5) 43.8 ± 0.9 (3) 9.0 ± 1.5 (5) 14.0 ± 1.5 (5) 4.2 AB-MCTS-M 38.9 ± 1.9 (2) 43.0 ± 1.5 (2) 40.6 ± 1.0 (1) 44.6 ± 0.9 (2) 12.3 ± 1.2 (4) 16.0 ± 1.0 (3) 2.3 AB-MCTS-A (Gaussian) 39.1 ± 1.9 (1) 42.5 ± 1.5 (3) 40.2 ± 1.7 (3) 43.4 ± 0.9 (4) 13.0 ± 3.6 (3) 18.3 ± 0.6 (2) 2.7 AB-MCTS-A (Beta) 38.7 ± 1.2 (3) 42.3 ± 0.8 (4) 40.4 ± 0.3 (2) 44.8 ± 0.6 (1) 14.0 ± 2.1 (2) 16.6 ± 0.6 (4) 2.7
the node that is created by choosing that GEN node. The backed-up score is used to update prior probability distribution parameters.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Benchmarks. We evaluated AB-MCTS on four diverse benchmarks that require complex problemsolving: LiveCodeBench [14], CodeContest [1], ARC-AGI [17], and MLE-Bench [16]. Live-CodeBench and CodeContest consist of competitive programming problems that demand mathematical and algorithmic reasoning. ARC-AGI involves abstracting a common transformation rule from visual patterns and implementing it as code. MLE-Bench, derived from Kaggle competitions, involves constructing and optimizing machine learning models to achieve high scores based on the evaluation metrics of each competition. For all these benchmarks, LLMs generate Python code to solve each task, and external feedback (e.g., test case results, validation scores) is available to guide the search. More details on the benchmarks can be found in Appendix B.1.
this section cite: ['b13', 'b0', 'b16', 'b15']

Section: Models.
We perform our experiments using GPT-4o (gpt-4o-2024-08-06) [18], and DeepSeek-V3 (deepseek-chat) [19]. Each LLM generates a complete solution in a single API call. We define the generation budget as the maximum number of API calls and set it to 2 7 = 128. The temperature was set to 0.6 for GPT-4o following [3] and 1.0 for DeepSeek-V3 following the official documentation.
Baselines. We benchmark AB-MCTS against three representative approaches. (1) Repeated Sampling (Best-of-n) [1,3,39] independently generates up to n candidate solutions from a single LLM prompt, a simple yet competitive baseline for coding tasks. (2) Sequential Refinement [4] iteratively improves each solution by re-prompting the LLM with its own output and feedback. (3) Standard MCTS follows the configuration from LATS [9, Section 5.2] (See also Appendix C.8). Each expansion adds five child nodes, and the search proceeds until it reaches the 2 7 nodes, with the final expansion creating only three nodes to meet this limit precisely. Hyper-parameters for AB-MCTS are summarized in Appendix B.2.
this section cite: ['b17', 'b18', 'b0', 'b2', 'b38', 'b3']

Section: Results
Table 2: Performance on MLE-Bench tasks. AB-MCTS-M demonstrates robust performance, achieving the best average rank across diverse ML tasks.
this section cite: []

Section: Method
Nomad2018 Spooky. Pizza. Avg. Repeated Sampling 0.065 (3) 0.47 (4) 0.72 (2) 3.0 Sequential Refinement 0.059 (1) 0.46 (3) 0.62 (3) 2.3 Standard MCTS 0.076 (4) 0.45 (2) 0.60 (4) 3.3 AB-MCTS-M 0.060 (2) 0.38 (1) 0.72 (1) 1.3
As detailed in Tables 1 and 2, our comprehensive evaluations reveal AB-MCTS as a consistently superior approach across diverse benchmarks and LLMs, achieving the top average rank and outperforming established baselines. This consistent success stems from the distinctive ability of AB-
this section cite: []

Section: Standard MCTS Sequential Refinement Repeated Sampling AB-MCTS-A (Gaussian) AB-MCTS-A (Beta) AB-MCTS-M

this section cite: []

Section: CodeContest (higher is better) ARC-AGI (higher is better) LiveCodeBench (higher is better)
Figure 4: Performance comparison on LiveCodeBench, CodeContest, and ARC-AGI. We compare the six methods using GPT-4o by plotting the success rate against the generation budget. The inset plots provide a detailed view of performance at a maximum generation budget (2 7 ); the mean success rate, its 95% confidence interval, and the results from the individual runs are shown. Variance at a generation budget of 2 0 arises from conducting each experiment independently with nonzero temperature. See Figure 6 for experiments on ARC-AGI with a larger budget.
this section cite: []

Section: Standard MCTS Sequential Refinement Repeated Sampling AB-MCTS-A (Gaussian) AB-MCTS-A (Beta) AB-MCTS-M

this section cite: []

Section: CodeContest (higher is better) ARC-AGI (higher is better) LiveCodeBench (higher is better)
Figure 5: Comparing algorithms by search tree shape and performance. Each point shows the performance against the average tree shape for a given algorithm at a specific generation budget. The x-axis represents the log-ratio of mean depth to mean width. Mean width is the average number of nodes per depth. Larger and smaller x-axis values indicate deeper and wider searches, respectively. MCTS to dynamically adapt its search strategy by precisely balancing exploration and exploitation to the varying demands of each problem, an adaptability largely absent in baseline methods. We next detail these results by benchmark, followed by an analysis of the search behavior of AB-MCTS.
this section cite: []

Section: LiveCodeBench and CodeContest.

this section cite: []

Section: ARC-AGI.
Figure 4 (right) shows the performance on ARC-AGI, a particularly challenging benchmark. Following ARC-AGI's official evaluation protocol, we report Pass@2 (Pass@1 is also reported in Appendix C.6). Consistent with previous work [13], repeated sampling proves to be a strong baseline in our setup, indicating the importance of broad exploration for this task. While standard MCTS yields only marginal improvements with larger budgets, our AB-MCTS framework achieves performance comparable to repeated sampling. This suggests AB-MCTS's capability to effectively explore potentially by dynamically widening its search when beneficial. Similar results were observed with DeepSeek-V3, as detailed in Appendix Figure 8.
this section cite: ['b12']

Section: MLE-Bench.
Table 2 and Appendix Figure 10 present the performance on three competitions from MLE-Bench using GPT-4o. Since MLE-Bench requires substantial GPU resources for training and evaluating machine learning models, we exclusively used GPT-4o and focused on the baseline methods and AB-MCTS-M (see also Appendix B.1). The best-performing baseline method varies across competitions. This again highlights that different tasks benefit from different explorationexploitation trade-offs. In contrast, AB-MCTS-M consistently delivers strong performance in these tasks. This consistent success across diverse competitions underscores AB-MCTS-M's inherent strength in effectively adapting its search strategy to varying problem structures.
this section cite: []

Section: Analysis

this section cite: []

Section: Analysis of Search Behavior: Width vs. Depth.
To quantitatively analyze how AB-MCTS balances exploration and exploitation, we examined the average depth and the average width at each depth of the generated search trees. Figure 5 shows that AB-MCTS methods tend to generate wider trees compared to standard MCTS. This occurs because AB-MCTS can adaptively decide to explore wider (select the GEN node) from any existing node, unlike standard MCTS. This mechanism allows for more flexible exploration across various tree depths (See also Appendix C.7). In addition to this flexibility in exploring wider, as seen in Table 2, AB-MCTS also achieves strong performance on benchmarks where sequential refinement excels, suggesting that AB-MCTS effectively identifies and exploits promising branches by selecting existing child nodes for refinement. This adaptive nature allows it to combine the strengths of exploration and exploitation, resulting in robust performance across diverse benchmarks.
Scaling with Increased Budget. Highly complex problems often require a substantial generation budget to find a correct solution. ARC-AGI is a prime example, where extensive exploration via repeated sampling is known to improve performance even at large budgets as reported in [13]. To investigate the scaling properties of our approach, we extended the experiments on ARC-AGI using DeepSeek-V3 with a larger generation budget up to 2 9 = 512. As shown in Figure 6, the performance of AB-MCTS continues to improve substantially as the budget increases from 200 to 500, while the improvement rate of repeated sampling begins to plateau. Standard MCTS also continues to improve with a larger budget, yet shows a significantly lower success rate compared to the AB-MCTS methods. This performance gap highlights that AB-MCTS is more effective at directing its search towards promising branches within the search tree at large computational scales.
Qualitative Analysis of the Search Trees. Figure 7 and Appendix Figure 11 present example search trees generated by AB-MCTS-M and standard MCTS. These visualizations illustrate more adaptive branching by AB-MCTS-M compared to standard MCTS. This adaptive nature reveals that AB-MCTS-M flexibly balances exploration and exploitation throughout the search process, dynamically allocating budget to explore diverse new candidates ("going wider") and refine promising ones ("going deeper"). Further discussion can be found in Appendix C.3.
Efficiency and Performance against Repeated Sampling. While repeated sampling benefits from potential efficiencies such as parallel sampling and no feedback computation costs, our results demonstrate the significant advantages of AB-MCTS. On ARC-AGI, where repeated sampling is notably strong, AB-MCTS not only continues to improve with increased budget but also ultimately achieves performance levels that repeated sampling cannot reach (Figure 6). Furthermore, on LiveCodeBench and CodeContest, AB-MCTS variants can reach the peak performance of repeated sampling substantially earlier in many cases (Figure 4, Appendix Figure 8). This indicates that even when accounting for the inherent advantages of repeated sampling, AB-MCTS emerges as a promising approach to efficiently use the generation budget to achieve superior results in diverse scenarios.
this section cite: ['b12']

Section: Conclusions
This paper introduced Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inferencetime framework to enhance LLM performance on complex tasks by effectively integrating multi-turn exploration and exploitation. Unlike previous methods, AB-MCTS dynamically decides to "go wider" or "go deeper" based on external feedback, leveraging Bayesian decision-making. Our experimental results show AB-MCTS outperforms repeated sampling and standard MCTS, demonstrating the value of adaptively handling the challenge of unbounded branching for effective inference-time scaling.
Limitations. Our approach assumes the existence of a reliable score evaluator, but developing such an evaluator itself can be challenging depending on the task. Future work could also explore search strategies that incorporate more fine-grained real-world cost factors beyond API call counts, potentially enhancing the practical utility of AB-MCTS. We believe that addressing these challenges will further enhance the applicability of AB-MCTS across a wider range of problems.
this section cite: []

Section: References
Ref_id:b0 Title: Competition-level code generation with alphacode Year: (2022)
Ref_id:b1 Title: Self-consistency improves chain of thought reasoning in language models Year: (2023)
Ref_id:b2 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b3 Title: Self-refine: Iterative refinement with self-feedback Year: (2024)
Ref_id:b4 Title: Reflexion: Language agents with verbal reinforcement learning Year: (2024)
Ref_id:b5 Title: CodeTree: Agent-guided tree search for code generation with large language models Year: (2025)
Ref_id:b6 Title: Code repair with LLMs gives an exploration-exploitation tradeoff Year: (2024)
Ref_id:b7 Title: Dale Schuurmans, and Xinyun Chen. Evolving deeper llm thinking Year: (2025)
Ref_id:b8 Title: Language agent tree search unifies reasoning, acting, and planning in language models Year: (2024)
Ref_id:b9 Title: SWE-search: Enhancing software agents with monte carlo tree search and iterative refinement Year: (2025)
Ref_id:b10 Title: How to understand whole software repository? arXiv preprint Year: (2024)
Ref_id:b11 Title: Improving llm reasoning through scaling inference computation with collaborative verification Year: (2024)
Ref_id:b12 Title: Getting 50% (sota) on arc-agi with gpt Year: (2024-01-21)
Ref_id:b13 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2025)
Ref_id:b14 Title: Reasoning with language model is planning with world model Year: (2023)
Ref_id:b15 Title: MLE-bench: Evaluating machine learning agents on machine learning engineering Year: (2025)
Ref_id:b16 Title: On the measure of intelligence Year: (2019)
Ref_id:b17 Title: Gpt-4o system card Year: (2024)
Ref_id:b18 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b19 Title: Openai o1 system card Year: (2024)
Ref_id:b20 Title: Competitive programming with large reasoning models Year: (2025)
Ref_id:b21 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b22 Title: Less is more for reasoning Year: (2025)
Ref_id:b23 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b24 Title: Kimi k1.5: Scaling reinforcement learning with llms Year: (2025)
Ref_id:b25 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
Ref_id:b26 Title: Scaling test-time compute optimally can be more effective than scaling LLM parameters Year: (2025)
Ref_id:b27 Title: Alphamath almost zero: Process supervision without process Year: (2024)
Ref_id:b28 Title: Omni-MATH: A universal olympiad level mathematic benchmark for large language models Year: (2025)
Ref_id:b29 Title: Marco-o1: Towards open reasoning models for open-ended solutions Year: (2024)
Ref_id:b30 Title: Mutual reasoning makes smaller LLMs stronger problem-solver Year: (2025)
Ref_id:b31 Title: rstar-math: Small llms can master math reasoning with self-evolved deep thinking Year: (2025)
Ref_id:b32 Title: Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving Year: (2025)
Ref_id:b33 Title: ReST-MCTS*: LLM self-training via process reward guided tree search Year: (2024)
Ref_id:b34 Title: How do large language monkeys get their power (laws)? Year: (2025)
Ref_id:b35 Title: Computing "elo ratings" of move patterns in the game of go Year: (2007)
Ref_id:b36 Title: Continuous upper confidence trees Year: (2011)
Ref_id:b37 Title: Monte carlo tree search with iteratively refining state abstractions Year: (2021)
Ref_id:b38 Title: Let's verify step by step Year: (2024)
Ref_id:b39 Title: Qwen2.5-math technical report: Toward mathematical expert model via self-improvement Year: (2024)
Ref_id:b40 Title: PyMC: A modern, and comprehensive probabilistic programming framework in python Year: (1516)
Ref_id:b41 Title: Hypothesis search: Inductive reasoning with language models Year: (2024)
Ref_id:b42 Title: Do we truly need so many samples? multi-llm repeated sampling efficiently scales test-time compute Year: (2025)
Ref_id:b43 Title: Arcagi-2: A new challenge for frontier ai reasoning systems Year: (2025)
Ref_id:b44 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025-06-26)
Ref_id:b45 Title: OpenAI o4-mini System Card Year: (2025-04-26)
