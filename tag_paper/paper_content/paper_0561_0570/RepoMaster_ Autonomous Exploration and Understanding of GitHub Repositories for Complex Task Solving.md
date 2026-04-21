Title: RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving
Abstract: The ultimate goal of code agents is to solve complex tasks autonomously. Although large language models (LLMs) have made substantial progress in code generation, real-world tasks typically demand full-fledged code repositories rather than simple scripts. Building such repositories from scratch remains a major challenge. Fortunately, GitHub hosts a vast, evolving collection of open-source repositories, which developers frequently reuse as modular components for complex tasks. Yet, existing frameworks like OpenHands and SWE-Agent still struggle to effectively leverage these valuable resources. Relying solely on README files provides insufficient guidance, and deeper exploration reveals two core obstacles: overwhelming information and tangled dependencies of repositories, both constrained by the limited context windows of current LLMs. To tackle these issues, we propose RepoMaster, an autonomous agent framework designed to explore and reuse GitHub repositories for solving complex tasks. For efficient understanding, RepoMaster constructs function-call graphs, module-dependency graphs, and hierarchical code trees to identify essential components, providing only identified core elements to the LLMs rather than the entire repository. During autonomous execution, it progressively explores related components using our exploration tools and prunes information to optimize context usage. Evaluated on the adjusted MLE-bench, RepoMaster achieves a 110% relative boost in valid submissions over the strongest baseline OpenHands. On our newly released GitTaskBench, RepoMaster lifts the task-pass rate from 40.7% to 62.9% while reducing token usage by 95%. Our code and demonstration materials are publicly available at https://github.com/QuantaAlpha/RepoMaster.

Section: Introduction
In recent years, the integration of toolchains [1,2,3,4] and iterative reasoning [5,6,7,8] has significantly enhanced large language models (LLMs) in code-related tasks [9,10,11]. These advancements have enabled LLMs to proficiently complete code snippets [12,13], debug errors [14], and even address complex machine learning problems [15,16]. However, when confronted with real-world challenges that necessitate task-driven code repositories [17], they struggle. At present, tackling such tasks remains largely manual and time-consuming due to the complexity and scale of the required code, which makes purely generative approaches impractical [11,18,19]. To overcome this, we propose a paradigm shift: reuse and adapt existing repositories as modular components tailored to specific tasks. This approach not only mitigates the challenges associated with repository-level code generation but also supports the broader goal of enabling agents to autonomously address sophisticated tasks using simple natural language instructions [20,21].
To facilitate this approach, leveraging platforms like GitHub becomes crucial. With over 28 million public repositories out of 190 million total projects, GitHub offers an extensive library of ready-made solutions for code agents [17,18,22,23]. Developers frequently reuse these repositories to tackle complex problems, yet LLM-based systems still falter in fully automating this process. Although frameworks like OpenHands [24] and SWE-Agent [14] demonstrate strong general capabilities, they often stumble on real-world codebases. In practice, simply following README instructions seldom works: READMEs can be vague, incomplete, or even erroneous, and repositories are not guaranteed to match a task's requirements out of the box-commands may need parameter changes, and key files can be misplaced. Consequently, when agents fail to locate or execute the necessary code, they must adapt by modifying existing components or generating new code to bridge the gap.
To achieve it, agents need to understand the repository in a task-driven way. However, GitHub repositories often have two key properties that make this hard: (1) intricate structural complexity, with many interconnected files, classes, and functions, and (2) information density that exceeds the context limits of most LLMs. Existing frameworks [14,15,24,25] do not provide mechanisms for grasping repository structures, tracking detailed dependencies, or strategically managing information within these constraints, ultimately resulting in suboptimal performance and higher token cost.
In this paper, we introduce RepoMaster, an end-to-end agent framework designed for automating the use of code repositories to tackle complex tasks. To address these challenges, RepoMaster draws inspiration from human programmers, who rarely read every line of code or error log when exploring unfamiliar codebases. Instead, they first map a project's structure, start viewing a key file, then jump to its relevant files based on signals like error traces, and filter out irrelevant details.
Following this intuition, RepoMaster first performs hierarchical structure analysis, builds dependency and call graphs, and identifies core components as the initial context. Navigated by these connections, it progressively explores the repository and applies information selection when viewing files and execution feedback to keep each interaction concise. By iteratively applying these steps, RepoMaster mimics human prioritization and makes efficient use of limited context windows. When evaluated on both MLE-R-a revised version of MLE-Bench-Lite [16]-and our newly constructed GitTaskBench [26], RepoMaster achieves significantly higher completion and success rates than OpenHands and SWE-Agent, while using far fewer tokens. Our contributions are summarized as follows:
(1) We propose a novel automated framework, RepoMaster, that can effectively leverage code repositories to solve the complex real-world tasks end-to-end. (2) To efficiently comprehend code in a goal-oriented, human-like manner, we integrate hybrid structural hierarchy modeling with core component identification, context-aware code exploration, and efficient information selection. (3) We validate RepoMaster's effectiveness and efficiency against Openhands and SWE-agent through experiments on diverse complex tasks from the MLE-R and GitTaskBench.
2 Related Work
this section cite: ['b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b10', 'b17', 'b18', 'b19', 'b20', 'b16', 'b17', 'b21', 'b22', 'b23', 'b13', 'b13', 'b14', 'b23', 'b24', 'b15', 'b25']

Section: Code Generation
LLMs have made substantial progress in code generation [12,13,27,28], exemplified by closedsource models [29,30,31] and the open-source series [32,33,34,35]. Beyond basic code completion [36], modern LLMs now support advanced tasks such as semantic code editing [23,37], debugging [38], and generating machine learning pipelines (e.g., AIDE [25] and MLAB [15] for Kaggle competitions). However, fully automating the creation of complex real-world codebases from scratch remains a critical challenge for AI agents [16,19,22].
this section cite: ['b11', 'b12', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b22', 'b36', 'b37', 'b24', 'b14', 'b15', 'b18', 'b21']

Section: LLM-based Agents for Tool Use
External tools are essential for extending the capabilities of LLM agents [5,39,40]. Relying on executable code [6,9]-using scripts to import inherent libraries, or call APIs, functionalized tools-has become a mainstream paradigm. Current works mainly focus on "tool learning" [1,6,10], but the more essential aspect of where to find the right tools is relatively overlooked [41]. Benchmarks, such as API-Bank [42] and ToolEyes [43], synthesize function libraries but are not realistic or practical; platforms such as RapidAPI [44] host real services but are closed-source and hard to extend. Standards such as FastAPI [45] or MCP [46], which unify interfaces for tool use via function calling mechanisms, have emerged. However, GitHub-a rich and dynamic ecosystem for automatically creating tools-remains underutilized in this context. Although GitAgent [17] first explored GitHub repositories as a tool extension, it is limited by simplistic repository search and understanding, and lacks validation in diverse real-world scenarios.
this section cite: ['b4', 'b38', 'b39', 'b5', 'b8', 'b5', 'b9', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b16']

Section: Repository Utilization
Using GitHub repositories to solve complex real-world tasks presents significant challenges. Re-poAgent [47] produces high-level documentation but fails to include realistic, task-oriented usage examples. ML-Bench-A [18] focuses on setting up the environment rather than understanding the repository. OpenHands [24] and SWE-Agent [14] are strong general agents that use step-by-step prompting to break down tasks and write code, but they lack methods to deeply understand the repository structure or build a clear hierarchy of its components. Aider [48] can track file dependencies but misses detailed function-level connections and cannot autonomously explore the codebase. Interactive assistants like Copilot [49] and Cursor [50] are effective for small-to-medium projects but struggle in large-scale repository contexts due to limited dependency awareness.
(1) Repository Search
"I want to remove scratches from this old image." # user's intent # key entities … Search and Select README file Star number Extract … (2) Hierarchical Repository Analysis … …… Hierarchical Code Tree (HCT) Module Dependency Graph (MDG) Function Call Graph (FCG) RepoMaster Explore the Repo Analyse Feedback & Optimize "I want to remove scratches from this old image." Granular Code View Dependency Analysis Search E x ec utio n E x p lo rat io n (3) Autonomous Exploration & Execution … … Agent produce new action Actions → Observations RepoMaster Initial components Exploration tools Iteratively optimize
this section cite: ['b46', 'b17', 'b23', 'b13', 'b47', 'b48']

Section: …
Expand viewed scope
this section cite: []

Section: Method
Most current frameworks follow the CodeAct paradigm [9,14,21,24], offering basic file-editing and exploration commands (e.g., OpenHands' AgentSkills [24] and SWE-Agent's command set [14]). But relying on README-based mappings and simple find/edit operations misses many core components and cannot perform deeper, autonomous exploration within limited LLM contexts. In contrast, RepoMaster mimics human programmers by performing a static, structure-aware analysis to locate critical components, then dynamically selecting only the essential snippets-skipping irrelevant information and focusing the LLM's limited context on what matters. The full end-to-end RepoMaster framework consists of three stages: (1) Repository Search: Identifying repositories relevant to the task. (2) Hierarchical Repository Analysis: Preparing the structures for exploration. (3) Autonomous Exploration & Execution: Iteratively interact with the repository and adjust exploration actions based on execution feedback. An overview of the framework is provided in Figure 1.
this section cite: ['b8', 'b13', 'b20', 'b23', 'b23', 'b13']

Section: Repository Search
To address complex online tasks expressed in natural language, we develop a deep-search method to locate the GitHub repositories most relevant to the task. We begin by analyzing the user's intent and extracting key entities to target the suitable repositories. We examine their README file and star count to assess their relevance and potential, and provide a brief description. Then, we select them by content quality and practical utility. Finally, we validate the top three candidates and deliver the results as structured JSON. An example of the deep-searching log is shown in Appendix B.
this section cite: []

Section: Hierarchical Repository Analysis

this section cite: []

Section: Hybrid Structural Repository Mapping
An essential prerequisite for task-oriented repository automation is a comprehensive structural model of the codebase. We sanitize the repository by removing all non-source files. For each retained file, we perform a single Abstract Syntax Tree (AST) walk [51] to recursively harvest both the meta-information and the raw source snippet of every module, class, and function. These atomic units provide the basis for understanding the repository's structure.
Let the target repository be denoted R = ⟨M, C, F, I⟩, where M = {m 1 , . . . , m |M | } is the set of modules (one per .py file), C = {c 1 , . . . , c |C| } the set of classes, F = {f 1 , . . . , f |F | } the set of functions/methods, and I ⊆ M × M the explicit import relations captured from source files. On this foundation, we construct three complementary artefacts:
• Hierarchical Code Tree (HCT). T , a nested package → module → class → function containment map annotated with line counts and docstring snippets.
• Function Call Graph (FCG). G f = (V f = F, E f , w f ), where an edge (f i , f j ) ∈ E f exists if f i invokes f j ; the weight w f encodes call frequency.
• Module Dependency Graph (MDG). G m = (V m = M, E m , w m ), in which (m i , m j ) ∈ E m if m i explicitly depends on m j ; w m measures coupling strength.
We thus obtain the tuple M, C, F, I, G f , G m , T , providing the agent with a deterministic, lossminimal structural synopsis of the entire repository before any task-specific exploration.
this section cite: ['b49']

Section: Core Component Identification
Having obtained a fine-grained yet verbose structural synopsis of the repository, we now need to compress this information into a concise context that preserves only the most influential code entitiessmall enough for multiple interaction turns within the LLM's window, yet rich enough to preserve global semantics. To this end, we specify an importance scoring scheme that operates first at the module level and then propagates to classes.
Module-level scoring. Each module m ∈ M receives a score I(m) ∈ [0, 10] by linearly aggregating six orthogonal features,
s = Dependency, Complexity, Usage, Semantic, Doc, Git ,(1)
I(m) = min 6 i=1 w i s i (m), 10 , w i ≡ 1,(2)
where Dependency captures centrality in MDG using the personalized PageRank [52] algorithm, Complexity approximates cyclomatic complexity, Usage measures import and call frequency, Semantic flags high-value keywords (e.g., main, core), Doc quantifies docstring richness, and Git reflects commit volume and recency. Detailed formulas for each feature are deferred to Appendix F. The equal weighting scheme (w i ≡ 1) was empirically validated through sensitivity analysis (see Appendix G.4 for details).
Class-level refinement. Module scores serve as priors for class importance. For every class c located in module µ(c), we compute
J(c) = I µ(c) + |F c | max c ′ |F c ′ | + Calls(F c ) max c ′ Calls(F c ′ ) ,(3)
where F c denotes the method set of class c. The second term rewards class richness in functionality; the third term captures how often its methods are actually invoked in the repository. Classes are ranked according to J(c), and the top-k classes are selected as the repository's core components.
this section cite: ['b50']

Section: Repository Context Initialization
Building on the identified core components, we construct an initial repository context in four distinct blocks. First, we include the complete README.md file, which provides high-level descriptions and detailed usage guidance authored by human developers. Second, we append a series of concise natural-language summaries for the highest-priority modules, giving the LLM a brief overview of each critical script's purpose. Third, we provide the source code of core components (i.e., the classes scored and selected in Section 3.2.2) as fine-grained semantic anchors. Finally, for all other top-ranked modules, we provide a flat, directory-grouped list of their file paths for easy on-demand lookup. Figure 2 illustrates the initial context, and Appendix D provides a complete example of this initial repository context construction.
this section cite: []

Section: ASTs

this section cite: []

Section: Codes Documents Logs

this section cite: []

Section: Initial Context

this section cite: []

Section: README.md
Brief summary of this file is …dataset setting…
this section cite: []

Section: Module Summary
Code Snippets
this section cite: []

Section: …
Module Paths
this section cite: []

Section: …
"explore" query
this section cite: []

Section: Match
Segement Match
1 4 … Traceback File "/x.py"， line xxx… Exceptioin To view efficiently: Interactive Feedback-based Execution Search Key Entities Trace Dependency … 5. Exploration Tool Calling 1. 2. 6. (2.) … 7. (3.) Context-aware Code Exploration Information Selection 4. Add into Context 3. Remove this image's scratches Microsoft/Bringing-old-photos-back-to-life Original Image Output Image understand the repo and plan execution steps… search the kyewords, view the README.md, download xx.pth generate process_image.py and write… set up environment variables, install missing dependencies… Model checkpoint is not found. Output files:[] No such file or directory check "pth exists", edit & rerun process_image.py… view directory structures… It looks like the path may be wrong. Let me check… Initial Core components Input/Output path Prompt： Code output: Comparison image saved as 'output.png' Checking output files: output_result/ final_output/ DeScratch_01_input.jpeg ... ... stage_1_restore_output/ origin/ DeScratch_01_input.jpeg masks/ mask/ DeScratch_01_input.png input/ DeScratch_01_input.jpeg This structured context serves as the agent's "launchpad" for dynamic exploration, allowing it to prioritize high-impact modules, trace dependencies, formulate targeted code queries and select relevant classes or functions, bridging static analysis with dynamic reasoning and task execution.
this section cite: []

Section: Autonomous Exploration & Execution

this section cite: []

Section: Context-aware Code Exploration
Once the agent has internalized the repository's functionality and overall structure, it immediately transitions to dynamic analysis, performing an autonomous, hierarchical and graph-based traversal of the codebase. To support in-depth comprehension and effective utilization of the repository, we offer a suite of fine-grained exploration tools organized into three categories: Granular Code View, Dependency Analysis, and Search.
• Granular Code View. This tool enables the agent to inspect the implementation details of files, classes, and functions using the HCT. It also retrieves and exposes the repository's directory hierarchy, facilitating swift orientation within the codebase.
• Dependency Analysis. This tool traces call chains and dependency paths by analyzing the FCG and MDG, respectively. It uncovers complex invocation and dependency relationships among code entities, thereby deepening the agent's comprehension of module interactions and overall code structure.
• Search. This tool equips the agent with robust search capabilities, facilitating rapid location of specific code segments within large and intricate codebases. It employs keyword matching to ensure efficient retrieval of relevant entities.
Together, these tools empower AI agents to proactively and autonomously navigate and examine code repositories, achieving a level of comprehension and flexibility comparable to human developers. Empirically, we observe that complex repositories typically require detailed dependency analysis using FCG and MDG, whereas simpler repositories often allow agents to effectively rely on HCT.
this section cite: []

Section: Interactive Feedback-based Execution
Task execution is grounded in the agent's evolving understanding of the repository. Once the agent has identified the hybrid structural elements described in Section 3.2.1 and core components described in Section 3.2.2 relevant to a given task, it begins to perform task-oriented operations.
Crucially, execution and exploration form a continuous, interleaved loop rather than a linear sequence. The agent can fluidly alternate between writing code and locating files, viewing content and reading logs, or tracing dependencies, all driven by the task context across different interaction turns, and powered by the exploration tools described in Section 3.3.1. This flexible loop allows the agent to iteratively refine its behavior by retrieving just-in-time information from the codebase. Figure 2 illustrates the execution and exploration pipeline of RepoMaster.
this section cite: []

Section: Context-aware Information Selection For Efficient Viewing
The agent must juggle source code, documentation, execution results and logs within a tight LLM token window for multiple turns, making it difficult to maintain a globally coherent view of the repository and severely limiting its applicability to large projects. To mitigate this issue, we propose a multi-level content reduction strategy that retains only the most critical information.
Viewing Code. At the code level, the agent parses source files into Abstract Syntax Trees (ASTs), extracts semantically and structurally meaningful subtrees, and uses these extracted subtrees as inputs.
Viewing Documents. For large or unstructured artifacts (e.g., .txt or .csv files), the agent divides each file into fixed-length chunks of L c tokens, It then generates retrieval prompts tailored to the current subtask, ranks the chunks by relevance, and retains the top n c most relevant segments.
this section cite: []

Section: Viewing Feedback Logs.
At the log level, we apply a human-like debugging heuristic that retains only the opening and closing segments of the log (where command invocations, exception traces, and diagnostic results cluster) and discards verbose intermediate output.
Multi-level reduction strategies activate only when the combined size of all candidate inputs exceeds the per-interaction token limit L, preserving global coherence by focusing on high-impact information and ensuring each execution-loop step relies on a compact, relevant context.
this section cite: []

Section: Experiments

this section cite: []

Section: Benchmarks and Metrics
To validate the effectiveness of RepoMaster, we evaluate it using two benchmarks.
this section cite: []

Section: MLE-R.
The original MLE-Bench [16] derives from Kaggle competitions, designed to evaluate LLM agents' capabilities in end-to-end machine learning engineering tasks. To construct MLE-R, we select 22 MLE-Bench tasks (covering nearly all MLE-Bench-lite cases) and apply the search procedure described in Section 3.1 to retrieve suitable GitHub repositories for each task, ensuring a fair comparisonfoot_0 ; the tasks' requirements are set to be completed based on their chosen repository rather than generating code from scratch.
Performance in MLE-R is evaluated using a medal-based system, the same as the original MLE-Bench, where solutions are assessed based on official Kaggle thresholds 4 for gold, silver, and bronze medals. Metrics include the achieved score, medal thresholds, and medal qualification, providing a clear indication of the model's proficiency in competitive ML engineering tasks.
GitTaskBench. In contrast to MLE-R, which emphasizes standard machine learning tasks (e.g., image classification), our new proposed GitTaskBench [26] foot_2 benchmark evaluates LLM agents on more practical real-world problems-common tasks whose complexity or format largely demands leveraging existing repositories, such as photo restoration. The benchmark consists of 18 repositories and 54 tasks, all described in natural language and designed to be completed using the provided repositories across a wide range of domains, such as image processing, video analysis, speech, physiological signals, office automation, and security and privacy. GitTaskBench evaluates two key aspects: Execution Completion Rate (measuring the model's ability to leverage the repository for output) and Task Pass Rate (assessing whether the output meets task-specific evaluation criteria). Given the diversity of tasks, evaluation metrics are predefined and tailored within the benchmark, ensuring a comprehensive assessment. Note that total tokens include both input and output tokens.
this section cite: ['b15', 'b25']

Section: Evaluation Setup
We evaluate our approach against two baseline frameworks and compare the performance across three state-of-the-art LLMs. The evaluation setup is as detailed below.
this section cite: []

Section: Baseline Frameworks.
We evaluate two baseline frameworks: OpenHands [24] and SWE-agent [14]. OpenHands provides sandboxed environments for code execution and API interactions, while SWE-agent focuses on autonomous GitHub issue resolution.
this section cite: ['b23', 'b13']

Section: Large Language Models.
We evaluate multiple leading LLMs, including the closed-source GPT-4o-2024-08-06 [53] and Claude-3-5-sonnet-20241022 [54], as well as the open-source DeepSeek V3-0324 [55]. This setup enables a comprehensive assessment of both agent architectures and LLM capabilities on solving real-world tasks with repository utilization.
this section cite: ['b51', 'b52', 'b53']

Section: Implementation Details.
Our proposed solution RepoMaster is built on a multi-agent dialog platform AutoGen [21]. To ensure agent performance, we set a few key hyperparameters. Specifically, we set the maximum token length per interaction L to 8000 tokens. For initial context construction, we generate concise summaries for the top 20 modules by importance score and extract k = 10 key classes. During the feedback phase, unstructured text files are split into chunks of L c = 1000 tokens, retaining the n c = 4 most relevant segments.
this section cite: ['b20']

Section: Comparison with SOTA
On the MLE-R benchmark, RepoMaster with Claude 3.5 attains a 95.45% valid submission rate and a 27.27% medal acquisition rate (including 22.73% gold medals), representing a more than five-fold improvement over the best open-source Agent baseline. RepoMaster with GPT-4o also achieves a strong 86.36% valid submission rate and 18.18% medal rate, further confirming its robust performance advantage under varied settings.
RepoMaster's significant performance improvement stems primarily from its effective identification and utilization of core components within open-source repositories, such as neural network architecture designs, optimized hyperparameter configurations, and data preprocessing pipelines. In contrast, baseline methods like OpenHands and SWE-Agent often struggle to pinpoint critical modules during repository exploration, filling limited context windows with excessive irrelevant code, resulting in insufficient understanding of model architectures and training logic.
In the GitTaskBench evaluation, RepoMaster significantly outperforms existing open-source frameworks SWE-Agent and OpenHands. Based on Claude 3.5, RepoMaster achieves a 75.92% execution completion rate and 62.96% task pass rate, surpassing OpenHands (53.70%, 40.74%) and SWE-Agent (41.67%, 22.23%). Similarly, RepoMaster maintains significant advantages on GPT-4o and DeepSeek V3, demonstrating that RepoMaster's inherent capabilities have good universality across underlying models. More importantly, RepoMaster substantially reduces computational overhead, with token consumption when using Claude 3.5 approximately 95% lower than OpenHands (154k vs 2883k tokens/task), proving the effectiveness of our hybrid hierarchical structure analysis and information pruning strategies.
this section cite: []

Section: Insightful analysis
Ablation Study To quantitatively assess the contribution of each component in RepoMaster, we conduct a comprehensive ablation study on the GitTaskBench benchmark using GPT-4o as the underlying model. By systematically removing key mechanisms, we measure their impact on three metrics of effectiveness and efficiency: execution completion rate, task pass rate, and token usage.
The results are shown in Table 3.
Table 3: Ablation study on the impact of core mechanisms in RepoMaster with GPT-4o on the GitTaskBench Benchmark. The "w/o All" refers to RepoMaster without any of the three mechanisms. "Exec. Comp." stands for "Execution Completion." Configuration Exec. Comp. Rate (%)↑ Task Pass Rate (%)↑ #Total Tokens↓ RepoMaster (Full) 48.14 40.74 250k -w/o Hybrid Hierarchical Analysis 46.29 (-1.85%) 38.88 (-1.86%) 274k -w/o Code Exploration 44.44 (-3.7%) 35.18 (-5.56%) 284k -w/o Information Selection 46.29 (-1.85%) 38.88 (-1.86%) 265k -w/o All 29.62 (-18.52%) 24.07 (-16.67%) 112k
Hybrid Hierarchical Analysis: Removing this component causes slight decreases in execution completion and task pass rates, with other components partially compensating. Token usage increases by 9.67%, showing that without structural repository understanding, more exploration is required.
this section cite: []

Section: Context-aware Code Exploration:
This component provides a suite of exploration tools enabling agents to autonomously navigate codebases according to task requirements, inspect file contents, and trace call chains and dependencies. The experiments prove this component to be particularly critical, with its removal causing significant decreases in execution and pass rates. More notably, token usage increases substantially, validating our hypothesis that strategic exploration effectively reduces repository exploration needs, allowing the agent to precisely target task-relevant components.
Information Selection: This component applies multi-level content reduction strategies to extract key information from source code, documents, and execution logs, preventing the context window from being filled with irrelevant content. Removing it decreases performance similarly to removing hybrid hierarchical analysis, with token usage increasing by only 6.00%. Its main value is maintaining a high signal-to-noise ratio rather than reducing token consumption.
The most revealing comparison is between the full RepoMaster system and the base code agent without any of our proposed components. The baseline achieves only 29.62% execution completion and 24.07% task pass rates-decreases of 18.52% and 16.67%. Interestingly, the baseline's token
this section cite: []

Section: …
Analyzing, Building… First, let's make sure our environment is set up correctly and that we have all the required data. Next, we'll verify that the necessary files are present. Task successfully completed. code analysis done failed Modifying `load_2d_data()` to print the .npz file contents and identify the correct key for loading 2D keypoint data… ≈ 17 turns ≈ 140 turns failed ≈ 20 turns Retrying 3D pose estimation…check the input data. Keys in the npz file: ['positions_2d', 'metadata'], both are 0-D arrays (scalars). NPZ file contents:['boxes', 'segments', 'keypoints', 'metadata'] Original keypoints shape: (100, 2) Processed keypoints shape: (100, 1, 2) Modifying `load_2d_data()` to print the .npz file contents and identify the correct key for loading 2D keypoint data, iterate through data structure analysis and type conversion… … Success: extract 2D coordinates using frame[1][0, :, :2] [Errno 2] No such file or directory: './checkpoint/pretrained.bin' Checking if the 'checkpoint' directory exists…Directory not found. Creating directory… Attempting to download the model… … … … ***** view_repository_structure & search***** Find the url of the checkpoint in the README.md… Goal-unoriented code modifications Limited repository understanding Now, let's first read the README.md file from the VideoPose3D repository to understand how to use it. We need to 1…2…3... Creating `process_and_predict.py` script… Attempting to use `TemporalModel`… Skip package installation. Switch to a NumPy-only implementation: Removed all PyTorch `TemporalModel` calls ModuleNotFoundError: No module named 'torch' Traceback (most recent call last): File "./process_and_predict.py", line 24, in <module> poses_3d[:, :2] = keypoints ~~~~~~~~^^^^^^V alueError: setting an array element with a sequence.
… Let's check its structure. Edit the file again to handle this data structure correctly. 3D poses saved to /output.npz Shape of output: (100, 3, 3) … Actual 2D coordinates: conflict with `run.py`'s fetch logic Print `data['positions_2d'].item()` and `data['metadata'].item()`. Read the run.py, locate the 'fetch' function, Reedit the function in run.py extract pose data directly from `keypoints['keypoints']['custom']`, instead of `keypoints[subject]` KeyError resolved. Direct patch applied to fetch() to accommodate new data format usage is significantly lower, but this reflects a failure case rather than efficiency: the agent simply gives up earlier without the necessary tools to effectively explore and utilize the repository.
Further analysis of the failure modes in ablated systems reveals: Without hybrid hierarchical analysis, the agent struggles to locate key repository components, often getting lost in non-essential files; without context-aware exploration, the agent frequently explores irrelevant parts of the repository, resulting in context fragmentation and redundant exploration; without information selection, the agent's context window becomes cluttered with low-value information, causing it to miss important details in error messages and execution traces.
this section cite: []

Section: Case Study
For the case study, we evaluated RepoMaster against OpenHands and SWE-Agent on a challenging 3D pose estimation task from GitTaskBench. As shown in Figure 3, neither baseline completed the task due to different failure modes. OpenHands ran extensive trial-and-error iterations (∼140 attempts, >10× others) and consumed higher tokens without success. SWE-Agent, although quicker, lacked task-level repository understanding-treating each error as a standalone fix and defaulting to a coarse 3D pose method that strayed from the core algorithm, causing task degradation. In contrast, RepoMaster leveraged structured repository analysis to efficiently focus on key components, achieving successful task completion with fewer attempts (<20 iterations).
this section cite: []

Section: Conclusion
We introduce RepoMaster, an end-to-end autonomous agent framework designed for automating the use of code repositories to tackle complex tasks. By combining static structural analysis of the repository with autonomous exploration, RepoMaster outperforms OpenHands and SWE-Agent in two challenging benchmarks. These results demonstrate that treating open-source repositories as modular, composable tools-rather than burdens to be regenerated from scratch-forms a powerful paradigm for solving complex real-world tasks. Beyond performance gains, RepoMaster promotes a more sustainable and collaborative AI-for-code ecosystem. Its capacity to reuse and adapt existing repositories lays the groundwork for large-scale orchestration of multiple projects within a single workflow, automated propagation of bug fixes and security patches upstream, and straightforward transfer to domains that share analogous structural challenges, such as hardware description languages, robotic middleware, or data-centric notebook collections. By enabling agents to understand and integrate code in context, RepoMaster accelerates the virtuous cycle between human contributors and AI systems, fostering continual improvement across the open-source landscape.
this section cite: []

Section: References
Ref_id:b0 Title: Tool learning with large language models: A survey Year: (2025)
Ref_id:b1 Title: What are tools anyway? a survey from the language model perspective Year: (2024)
Ref_id:b2 Title: Trove: Inducing verifiable and efficient toolboxes for solving programmatic tasks Year: (2024)
Ref_id:b3 Title: Large language models as tool makers Year: (2023)
Ref_id:b4 Title: React: Synergizing reasoning and acting in language models Year: ()
Ref_id:b5 Title: Tree-of-code: A treestructured exploring framework for end-to-end code generation and execution in complex task handling Year: (2024)
Ref_id:b6 Title: Llms with chain-of-thought are non-causal reasoners Year: (2024)
Ref_id:b7 Title: Self-reflection in llm agents: Effects on problem-solving performance Year: (2024)
Ref_id:b8 Title: Executable code actions elicit better llm agents Year: (2024)
Ref_id:b9 Title: Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges Year: (2024)
Ref_id:b10 Title: Evaluating ai's ability to replicate ai research Year: (2025)
Ref_id:b11 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b12 Title: Competition-level code generation with alphacode Year: (2022)
Ref_id:b13 Title: Swe-agent: Agent-computer interfaces enable automated software engineering Year: (2024)
Ref_id:b14 Title: Mlagentbench: Evaluating language agents on machine learning experimentation Year: (2023)
Ref_id:b15 Title: Mle-bench: Evaluating machine learning agents on machine learning engineering Year: (2024)
Ref_id:b16 Title: Gitagent: facilitating autonomous agent with github by tool extension Year: (2023)
Ref_id:b17 Title: Ml-bench: Evaluating large language models and agents for machine learning tasks on repository-level code Year: (2023)
Ref_id:b18 Title: Devbench: A comprehensive benchmark for software development Year: (2024)
Ref_id:b19 Title: Meta programming for multi-agent collaborative framework Year: (2023)
Ref_id:b20 Title: Autogen: Enabling next-gen llm applications via multi-agent conversation Year: (2023)
Ref_id:b21 Title: Where are large language models for code generation on github? arXiv preprint Year: (2024)
Ref_id:b22 Title: Swe-bench: Can language models resolve real-world github issues? arXiv preprint Year: (2023)
Ref_id:b23 Title: Openhands: An open platform for ai software developers as generalist agents Year: (2024)
Ref_id:b24 Title: Aide: Ai-driven exploration in the space of code Year: (2025)
Ref_id:b25 Title: GitTaskBench: Anonymous github repository Year: (2025-05)
Ref_id:b26 Title: Code llama: Open foundation models for code Year: (2023)
Ref_id:b27 Title: Starcoder: may the source be with you! CoRR Year: (2023)
Ref_id:b28 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b29 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b30 Title:  Year: (2023)
Ref_id:b31 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b32 Title: Qwen technical report Year: (2023)
Ref_id:b33 Title:  Year: (2024)
Ref_id:b34 Title: Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence Year: (2024)
Ref_id:b35 Title: Repocoder: Repository-level code completion through iterative retrieval and generation Year: (2023)
Ref_id:b36 Title: Can it edit? evaluating the ability of large language models to follow code editing instructions Year: (2023)
Ref_id:b37 Title: Evaluating debugging capability of large language models Year: (2024)
Ref_id:b38 Title: Tool learning with foundation models Year: (2024)
Ref_id:b39 Title: Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings Year: (2023)
Ref_id:b40 Title: Where are large language models for code generation on github? arXiv preprint Year: (2024)
Ref_id:b41 Title: Api-bank: A comprehensive benchmark for tool-augmented llms Year: (2023)
Ref_id:b42 Title: Tooleyes: fine-grained evaluation for tool learning capabilities of large language models in real-world scenarios Year: (2024)
Ref_id:b43 Title: Facilitating large language models to master 16000+ real-world apis Year: (2023)
Ref_id:b44 Title: Microservice APIs: Using Python, Flask, FastAPI, OpenAPI and More. Simon and Schuster Year: (2023)
Ref_id:b45 Title: Model context protocol: Introduction Year: (2025-05)
Ref_id:b46 Title: Repoagent: An llm-powered open-source framework for repository-level code documentation generation Year: (2024)
Ref_id:b47 Title: Ai pair programming in your terminal Year: (2025-05)
Ref_id:b48 Title: Github copilot: Your ai pair programmer Year: (2025-05)
Ref_id:b49 Title: A novel neural source code representation based on abstract syntax tree Year: (2019)
Ref_id:b50 Title: The pagerank citation ranking: Bringing order to the web Year: (1999)
Ref_id:b51 Title: Hello gpt Year: (2024)
Ref_id:b52 Title: The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card Year: (2024)
Ref_id:b53 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b54 Title: Gittaskbench: A benchmark for code agents solving real-world tasks through code repository leveraging Year: (2025)
Ref_id:b55 Title: Listing 2 document search queries, repository rankings, and selection rationales, thereby ensuring transparency and reproducibility in our repository selection methodology. Listing 2: Search for repositories related to the Kaggle APTOS 2019 blindness detection competition. /* Task Definition Phase */ Search for GitHub repositories related to "aptos2019-blindness-detection Year: ()
Ref_id:b56 Title: Searching Phase */ Perform query: "aptos2019-blindness-detection GitHub repository Year: ()
Ref_id:b57 Title: Blindness-Detection"} browsing: {"query": "README Year: (2019)
Ref_id:b58 Title: Utilizes machine learning models for disease detection using retina images Year: ()
Ref_id:b59 Title: No specific performance metrics mentioned Year: ()
Ref_id:b60 Title: Focuses on the competition task with a machine learning approach, but lacks detailed performance data Year: ()
Ref_id:b61 Title: Aims to detect diabetic retinopathy using retina images, with a focus on automatic screening. [Performance]: No specific performance metrics mentioned Year: ()
Ref_id:b62 Title: Provides a comprehensive approach but lacks detailed performance data Year: ()
Ref_id:b63 Title: Machine learning model to identify diabetic retinopathy automatically Year: ()
