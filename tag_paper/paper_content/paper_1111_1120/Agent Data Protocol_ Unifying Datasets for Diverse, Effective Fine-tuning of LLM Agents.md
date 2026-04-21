Title: AGENT DATA PROTOCOL: UNIFYING DATASETS FOR DIVERSE, EFFECTIVE FINE-TUNING OF LLM AGENTS
Abstract: Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the Agent Data Protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed supervised finetuning on the unified data, and demonstrated an average performance gain of ∼20% over corresponding base models, and delivers stateof-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training.

Section: INTRODUCTION
Pre-training large language models (LLMs) benefits from abundant, readily available Internet-scale data. In contrast, post-training presents a much harder challenge: high-quality task-specific data must be carefully curated. While creative strategies have emerged for collecting data in relatively simple settings, such as single-turn user interactions like code generation (Nijkamp et al., 2023), question answering (Rajpurkar et al., 2016), and sentiment analysis (Maas et al., 2011), many realworld tasks are far more complex.
A particularly difficult case is agent applications, where models must take sequential actions and interact with the world iteratively. Building datasets for such scenarios requires recording and structuring trajectories of agent behavior, much more challenging than collecting static input-output pairs. Despite these difficulties, a growing body of work has explored different approaches for creating agent datasets. These efforts vary in methodology, from manual curation (Rawles et al., 2023;Xu et al., 2024a), to synthetic data generation (Ou et al., 2024;Zheng et al., 2024a), to recorded agent rollouts (Pan et al., 2025;Yang et al., 2025b). The resulting datasets span a wide range of tasks, including web navigation (Deng et al., 2023;Lù et al., 2024), software development (Yang et al., 2025b;Pan et al., 2025), visual interface control (Rawles et al., 2023;Kapoor et al., 2024), and general tool use (Zeng et al., 2023;Liu et al., 2024a) (an overview of these datasets in § 2.1).
Agent Data Protocol Raw Data ❖ AgentInstruct ❖ CodeActInstruct ❖ SWE-Gym ❖ Mind2Web ❖ …… Action ➢ API Action ➢ Code Action ➢ Message Action Observation ➢ Text Observation ➢ Web Observation OpenHands SFT SWE Agent SFT AgentLab SFT APIAction ( function=goto, kwargs={url: google.com} ) <!doctype html> <html itemscope … <title> Google </title> … </html> WebObservation ( url=google.com, html=<html>..., axtree=RootWebArena 'Google', focused … ) CodeAction ( language=python, content=print("Hello World") ) ```python print("Hello World") ```T extObservation ( content=Hello World, source=environment ) goto(url=google.com) Execution result: Hello World MessageAction ( content=How can I help you? ) How can I help you? Trajectory ( id=example_id, content=[...] ) However, despite the availability of such data, large-scale supervised fine-tuning (SFT) of agents remains rare in academic research. A few notable projects, such as Zeng et al. (2023) and Mitra et al. (2024), have demonstrated their potential, but remain exceptions rather than the norm. Why has this not become standard practice? We argue that the issue is not a lack of data, but rather a lack of standardization. Existing datasets are fragmented, with inconsistent formats and representations, making it difficult to combine, share, and leverage them effectively, thus they remain underutilized.
To address this gap, we introduce the Agent Data Protocol (ADP), a standardized expressive representation language for agent data. By converting heterogeneous datasets into ADP, it makes it simple to generate large-scale and diverse data for a variety of downstream training pipelines (Figure 1). Technically, ADP is implemented as Pydanticfoot_0 schemas that express actions and observations corresponding to common agent use cases such as communicating, browsing, coding, and miscellaneous tool calling, coupled with strict automated validation to maintain high data quality.
As a first step to demonstrate the practical utility of ADP, we implement converters from 13 preexisting datasets into ADP, and converters from ADP to 3 different agent architectures, demonstrating its generality. Based on this, we create and release the largest publicly available dataset for agent training, consisting of 1.3M training trajectories, dubbed the ADP Dataset V1.
Our experiments show training agents using ADP leads to significant performance improvements across diverse domains, including coding (SWE-Bench Verified), web browsing (WebArena), research (GAIA), and agentic tool use (AgentBench), as shown in § 6. Notably, these results improve by an average of 20% over base models, and are competitive with or superior to other state-of-the-art results from similarly-sized models. We also identify significant benefits from cross-task transfer, with training on the ADP data improving significantly over training on individual datasets. Beyond performance, ADP enables systematic cross-dataset analysis, revealing trends and areas for improvement in publicly available data.
Finally, we release all code and datasets in open source to foster community adoption and encourage contributions of new datasets. We believe ADP will unlock a new wave of progress in agentic model fine-tuning by providing the standardization needed to make large-scale supervised agent training practical and scalable.
this section cite: ['b29', 'b36', 'b21', 'b37', 'b31', 'b32', 'b6', 'b18', 'b32', 'b37', 'b13', 'b51', 'b51', 'b24']

Section: RELATED WORK
The development of effective LLM-based agents critically depends on high-quality training data that could capture the complexity of multi-step reasoning, tool usage, and environmental interaction (Yao et al., 2022b;Schick et al., 2023;Deng et al., 2023;Masterman et al., 2024). This section reviews existing methods for agent data collection and the challenges that motivate ADP.
this section cite: ['b38', 'b6', 'b22']

Section: AGENT DATA COLLECTION METHODS
Existing approaches span manual creation (human experts creating step-by-step demonstrations of desired agent behaviors) (Nakano et al., 2021;Yao et al., 2022a), synthetic generation (leverages existing LLMs to create agent trajectories through prompting or structured generation) (Luo et al., 2023;Xu et al., 2024b), and recorded agent rollouts (captures trajectories from existing agent systems during task execution) (Wang et al., 2024a;Pan et al., 2025), etc, resulting in abundant agent training data, a representative set of which listed in Table 1. We also group each dataset into a coarse task category.
• Coding: generally includes fundamental programming tasks, such as command line code generation, algorithm implementation, code completion, code translation, and code repair, etc.
• Software Engineering: often consists of repository-level software engineering tasks, such as bug fixing, feature implementation, code refactoring, and dependency management, etc.
• API/Tool Use: usually requires agents to use external APIs/tools effectively to solve tasks. Common tools include file manipulation, database queries, and customized APIs, etc.
• Web Browsing: commonly encompasses tasks including web navigation, online shopping, and social media interactions, etc, requiring agents to understand GUIs.
this section cite: ['b28', 'b20', 'b32']

Section: CHALLENGES AND LIMITATIONS
Despite abundant existing agent training datasets, several fundamental challenges prevent effective large-scale utilization of these resources:
• Complexity of Data Curation: Creation of high-quality agent training data requires significant resources and expertise (Paullada et al., 2021;Bhardwaj et al., 2024;Zha et al., 2025). Manual curation is expensive and requires domain knowledge; synthetic generation faces challenges in verifying data quality; recorded agent rollouts are fundamentally constrained by the capabilities of existing baseline agents, limiting the diversity and complexity of trajectories. While recent efforts have scaled trajectory collection (Song et al., 2024;Mitra et al., 2024), the fundamental challenge of balancing quality, diversity, and scale across different curation approaches remains.
• Heterogeneity of Dataset Format: Existing agent training datasets each employ its own representation format, action spaces, and observation structures (Ning et al., 2025;Luo et al., 2025). For example, some web datasets use HTML while some use accessibility tree structures (de Chezelles et al., 2025). Existing efforts have noted and begun addressing data standardization (Zhang et al., 2024;Chen et al., 2024;Mohammadi et al., 2025;Xi et al., 2025;Zhang et al., 2025), but they mostly focused on proposing task-specific or agent-specific unification rather than communitywide standardization of data representation, limiting plug-and-play with other datasets or agents, where significant engineering effort is still required to utilize multiple datasets together, hindering integration across different data sources.
• Difficulty of Analysis and Comparison: The diverse structures of existing datasets also makes it difficult to perform systematic comparisons or quantitative analysis across different data sources (Putrama & Martinek, 2024), limiting researchers' ability to understand the relative usefulness, coverage, and quality of different datasets, hindering data-driven selection or improvements.
this section cite: ['b2', 'b52', 'b39', 'b24', 'b30', 'b19', 'b5', 'b53', 'b3', 'b25', 'b43', 'b54', 'b34']

Section: THE AGENT DATA PROTOCOL
To overcome these challenges and limitations, and to make good use of existing data resources, we propose the Agent Data Protocol (ADP). ADP establishes a unified schema that bridges the gap between existing heterogeneous agent training datasets and large-scale supervised agent fine-tuning.
this section cite: []

Section: DESIGN PRINCIPLES
We design ADP around the following core principles:
• Simplicity: ADP maintains a simple and intuitive structure. This directly addresses the complexity of data curation challenge by providing a straightforward framework that eliminates the need for specialized per-dataset engineering, making large-scale agent data utilization accessible to researchers without extensive adaptation effort.
• Standardization: ADP is designed to provide a unified representation that unifies existing agent training datasets of various different formats to a standardized format, addressing the challenge of heterogeneous dataset formats.
• Expressiveness: ADP is designed to ensure that complex agentic trajectories could be accurately expressed with no loss of critical information. This directly addresses the difficulty of analysis and comparison challenge because ADP is expressive enough to cover the broad variety of existing agent datasets across different domains, enabling researchers to put these diverse datasets under the same conditions and context.
By addressing the fundamental challenges in utilization agent data, ADP aims to push the progress in agent training, making large-scale agent SFT more accessible to the broader research community.
this section cite: []

Section: ARCHITECTURE
The ADP schema is implemented as Pydantic schemas, and is simple yet expressive in design. Each ADP standardized agent trajectory is represented as a Trajectory object.
Trajectory consists of (1) id: trajectory id, (2) content: an alternating sequence of actions and observations representing the agent's interaction with the user/environment, (3) details: A flexible metadata dictionary for dataset-specific information (e.g., dataset source URLs).
Action represents agents' decisions and behaviors. We categorize actions into three types:
• API Actions: Function calls with structured parameters and outputs capturing tool use. Each API action includes: (1) function: name of tool call, (2) kwargs: a dictionary of function arguments, and (3) description: optional reasoning or explanation for the action. For example, with ADP, a web navigation call goto(url=https://www.google.com) is represented as APIAction(function=goto, kwargs=url:https://www.google.com).
• Code Actions: Code generation and execution across programming languages. Each code action specifies: (1) language: the programming language (e.g., python), (2) content: the code to execute, and (3) description: optional reasoning or explanation for the action. For example, the ADP representation of a python code block '''python print("Hello World")''' is CodeAction(language=python,content=print("Hello World").
• Message Actions: Natural language communications between agents and users, each containing a content field, documenting agents' explanations, clarifications, and responses. For example, MessageAction(content=How can I help you?).
Observation represents agents' perceptions from the environment, categorized into two types:
• Text Observations: Captures the text information from various sources, including user instructions and environmental feedback. Each text observation includes: (1) source: the origin of the observation ("user" or "environment"), and (2) content: the observed text. For example, a python execution output Execution result: Hello World, will be converted to ADP format TextObservation(content=Hellow World, source=environment).
• Web Observations: Represent the state and content of webpages. Each observation includes: (1) html: raw HTML content, (2) axtree: accessibility tree of the webpage, (3) url: current page URL, (4) viewport size: browser viewport dimensions, and (5) image observation: optional screenshot data. Web observations enable ADP to support complex browsing scenarios.
The core insight behind ADP is that despite the surface-level diversity in agent datasets, most agentic interactions can be decomposed into a sequence of actions taken by the agent and observations received from the environment. By standardizing these fundamental components, ADP directly addresses each challenge identified in § 2.2 while preserving the rich semantics of the original data. This unified representation enables researchers to combine datasets that were previously incompatible, facilitating large-scale training across diverse domains.
this section cite: []

Section: CONVERSION PIPELINE
As shown in Figure 1, we implemented a three-stage conversion pipeline with ADP that transforms heterogeneous datasets into training-ready agentic formats.
1. Raw to Standardized: This stage unifies original dataset formats into the ADP standardized schema. Each dataset is extracted in its raw format, and then converted to the ADP schema by mapping each dataset-specific actions and observations to the ADP's standardized action and observation space. For example, a web browsing task with HTML representations is converted to a pairs of APIAction and WebObservation, while a coding task with execution output is mapped to CodeAction and TextObservation pairs.
2. Standardized to SFT: This stage converts ADP standardized trajectories into supervised finetuning (SFT) format suitable for training language models. Different agent frameworks operate with distinct actions spaces, observations formats, etc. For example, OpenHands employs IPython execution with web browsing capabilities, SWE-Agent uses structured bash commands and file operations, while AgentLab focuses on DOM-based web interactions. Rather than training only one generic action model, we recognize that effective agent training requires adaptation to each framework's specific scaffolding and interactions formats. For each agent harness, the conversion process uses one agent-specific script that translates each type of action and observation into the target agent's action and observation space based on the agent's framework. This stage handles context management, specifies system prompts, and formats conversations to create SFT-ready instruction-response pairs, optimized for the particular agent architecture.
3. Quality Assurance: This stage ensures data correctness and consistency in alignment with agent format, tool use, and conversation structure through automated validation. Example quality checks include verifying tool call formats, ensuring mostfoot_1 tool calls are paired with an English thought, and checking whether the conversation ends properly, etc.
this section cite: []

Section: PRACTICAL IMPACT OF ADP ON AGENT TRAINING RESEARCH
The two-direction pipeline (Raw→ADP and ADP→SFT) cleanly separates responsibilities and eliminates redundant engineering (Figure 2). In practice:
• Dataset conversion (once per dataset). Contributors convert each raw dataset to the ADP schema exactly once. From then on, the dataset is a standardized resource usable by any agent harness.
• Agent-specific conversion (once per agent). Each agent maintains a single script for ADP→SFT; no per-dataset engineering needed. Adding new datasets requires no change to agent-side scripts. ADP amortizes conversion cost across the community, accelerates adoption of new datasets, and ensures that a single ADP→SFT script instantly unlocks the entire pool of ADP-standardized data to an agent framework. Without ADP, researchers must write a Raw→SFT converter for each dataset-agent pair, duplicating effort across groups and making large-scale data integration brittle and slow. More discussion could be found in § 6.3. Table 2 shows analysis on 13 ADP standardized datasets, revealing significant diversity across datasets.
this section cite: []

Section: CROSS DATASET ANALYSIS
Trajectory Length. Trajectory rounds vary dramatically across datasets, from 1 to 26.8 turns, with an average of 10.1 turns. SWE datasets consistently exhibit longer length, reflecting the inherent complexity of repo-level programming tasks.
this section cite: []

Section: Action Distribution.
Clear domainspecific preferences emerge from the action distributions after standardization with ADP. Web datasets (e.g. Mind2Web) heavily favor API actions with minimal code execution, reflecting their focus on interface interaction. Conversely, coding datasets (e.g., CodeActInstruct) show high code usage with no API usage, emphasizing direct programming activities. SWE datasets (e.g., SWE-smith) demonstrate mixed patterns, which relies on API actions like file writes while using code actions for code generation.
this section cite: []

Section: Function Reasoning Analysis.
A striking finding is the high function thought coverage (≥ 90% for most datasets), indicating that these training datasets consistently provide explanations for actions. This is particularly valuable for interpretability and training agents with reasoning abilities. Importantly, high reasoning coverage appears across all task varieties, suggesting that function thoughts represent a general characteristic of well-documented datasets rather than domain-specific behavior.
this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: TRAINING SETUP
To evaluate ADP's effectiveness in training across diverse data sources, we utilize a comprehensive collection of 13 agent training datasets, spanning coding, SWE, API/tool user, and browsing, as documented in Table 1. These datasets represent a broad spectrum of heterogeneity challenges that ADP addresses, including varied data creation methodologies (synthetic generation, manual curation, agent rollouts), different complexity (from simple to complex multi-step workflows), and diverse environments (command-line interfaces, web GUIs, Jupyter Notebooks, API calls).
The selected datasets collectively contain over 1.3M instances, ranging from smaller ones like Mind2Web to larger-scale ones like Orca AgentInstruct. To ensure balanced representation across domains and prevent any single large dataset from dominating the training process, we subsample from larger datasets while using smaller datasets in their entirety. Full details of our data sampling and mixture weights are in Appendix C. We use Qwen2.5-Coder-Instruct model family (Qwen Team, 2024;Hui et al., 2024) as the base models, with 3 agent frameworks for comprehensive evaluation across multiple benchmarks. We fine-tuned all models using the same SFT pipeline from LLaMA-Factory (Zheng et al., 2024b). These experiments focus on each framework's specialized domain to demonstrate targeted effectiveness. Each agent has unique architectures, tool interfaces, and interaction environments. This diversity allows us to validate that ADP-standardized data can be readily and easily converted to different agent formats, demonstrating the protocol's utility across various agent implementations.
OpenHands (Wang et al., 2025) is an open platform for building generalist AI agents that operate like software developers: writing code, using command lines, and browsing the web. It provides sandboxed execution environments, tool coordination, and benchmark evaluation.
AgentLab (Drouin et al., 2024;de Chezelles et al., 2025) is an open-source framework for developing, testing, and benchmarking web agents across diverse tasks, emphasizing scalability and reproducibility. It supports a suite of evaluation benchmarks like WebArena and WorkArena. (Yang et al., 2024) introduces a custom Agent-Computer Interface (ACI) that enables language model agents to autonomously perform software engineering tasks by navigating codebases, editing and running code, viewing files, and executing tests.
this section cite: ['b35', 'b11', 'b42', 'b7', 'b5', 'b47']

Section: SWE-Agent

this section cite: []

Section: EVALUATION BENCHMARKS
We evaluated these agents across 4 benchmarks (based on the availability of benchmark evaluation code and specialization of agents) that span different domains. This comprehensive evaluation demonstrates ADP's expressiveness in preserving critical information across diverse tasks. (Jimenez et al., 2024) evaluates agents on real-world software engineering tasks. Given a Github codebase and a bug report, agents must generate patches that satisfy existing unit tests. We used the SWE-Bench Verified subset for evaluation (Chowdhury et al., 2024).
this section cite: ['b12', 'b4']

Section: SWE-Bench
WebArena (Zhou et al., 2024) provides a realistic, self-hosted web environment composed of fully functional websites in domains like e-commerce, forums, and map navigation, requiring agents to interpret high-level natural language commands and perform concrete web interactions.
AgentBench (Liu et al., 2024b) evaluates agents across different environments, such as operating systems, databases, and web browsing. It emphasizes multi-turn reasoning, decision making, and adaptability across domains.
GAIA (Mialon et al., 2023) is a benchmark for general AI assistants featuring human-annotated tasks that combine reasoning, tool use, and multi-step problem solving, often with multimodal input. Tasks vary in difficulty by number of steps and required tools.
this section cite: ['b57', 'b23']

Section: EXPERIMENTAL RESULTS

this section cite: []

Section: ADP DATA RESULTS IN HIGHLY EFFECTIVE AGENTS ACROSS DIVERSE TASKS
ADP fine-tuning consistently improves performance across models, benchmarks, and agent harnesses.
As shown in Table 3, Table 4, and Table 5, training on standardized ADP data yields substantial gains across 7B, 14B, and 32B models on several popular evaluation benchmarks. On SWE-Bench (Verified), ADP training delivers remarkable improvements: Qwen-2.5-7B-Coder-Instruct improves from 0.4% to 20.2% (+19.8%) Table 4: Comparison of SOTA and our Best 13-14B ADP-trained agents' results across benchmarks. Shaded rows are our ADP-tuned models. Other rows are collected from previous works.
this section cite: []

Section: Agent Model Training Data Accuracy
SWE-Bench (Verified) (Jimenez et al., 2024;Chowdhury et al., 2024) SWE-Agent (Yang et al., 2024) Qwen-2.5-14B-Coder-Instruct -2.0% Claude 3.5 Sonnet(Anthropic Team) -33.6% Qwen-2.5-14B-Coder-Instruct ADP Data 34.4% (+32.4%) OpenHands CodeActAgent (Wang et al., 2025) Qwen-2.5-14B-Coder-Instruct -5.8% Qwen-2.5-14B-Coder-Instruct SWE-Gym (Pan et al., 2025) 16.4% (+10.6%) Qwen-2.5-14B-Coder-Instruct ADP Data 30.6% (+24.8%)
WebArena (Zhou et al., 2024) AgentLab (Drouin et al., 2024) (de Chezelles et al., 2025) Qwen-2.5-14B-Coder-Instruct -5.5% Qwen-2.5-14B-Coder-Instruct ADP Data 22.2% (+16.7%)
AgentBench OS (Liu et al., 2024b) AgentLM (Liu et al., 2024b)
Llama-2-chat-13B - 9.0% Llama-2-chat-13B
AgentInstruct (Zeng et al., 2023) 18.1% (+9.1%) OpenHands CodeActAgent (Wang et al., 2025)
Qwen-2.5-14B-Coder-Instruct - 2.8% Qwen-2.5-14B-Coder-Instruct ADP Data 20.8% (+18.0%)
with SWE-Agent and from 2.8% to 20.4% (+17.6%) with OpenHands. At 14B scale, Qwen-2.5-14B-Coder-Instruct achieves 34.4% (+32.4%) with SWE-Agent and 30.6% (+24.8%) with OpenHands. The 32B model reaches 40.3% (+38.1%) with SWE-Agent and 36.8% (+26.2%) with OpenHands, matching or exceeding Claude 3.5 Sonnet with SWE-Agent's 33.6% performance. On WebArena, ADP training shows consistent gains across model sizes: 7B achieves 21.0% (+16.5%), 14B reaches 22.2% (+16.7%), and 32B attains 22.9% (+12.0%). On AgentBench OS, the improvements are substantial: the 7B model improves from 3.5% to 27.1% (+23.6%), the 14B model improves from 2.8% to 20.8% (+18.0%), and 32B models from 27.8% to 34.7% (+6.9%). Finally, on GAIA, the 7B model improves from 7.3% to 9.1% (+1.8%).
These gains, spanning both coding and browsing settings, show that a unified, cross-domain ADP training corpus can deliver SOTA or near-SOTA performance without domain-specific tuning and is effective across models, action spaces, and agent harnesses. Figure 3 and Figure 4 also show clear
Table 5: Comparison of SOTA and our Best 32B ADP-trained agents' results across benchmarks. Shaded rows are our ADP-tuned models. Other rows are collected from previous works.
this section cite: ['b12', 'b4', 'b47', 'b42', 'b32', 'b57', 'b7', 'b5', 'b51', 'b42']

Section: Agent

this section cite: []

Section: Model Training Data Accuracy
SWE-Bench (Verified) (Jimenez et al., 2024;Chowdhury et al., 2024) SWE-Agent (Yang et al., 2024) Qwen-2.5-32B-Coder-Instruct -2.2% Qwen-2.5-32B-Coder-Instruct SWE-smith (Yang et al., 2025b) 40.2% (+38.0%) Qwen-2.5-32B-Coder-Instruct ADP Data 40.3% (+38.1%) OpenHands CodeActAgent (Wang et al., 2025) Qwen-2.5-32B-Coder-Instruct -10.6% Qwen-2.5-32B-Coder-Instruct SWE-Gym (Pan et al., 2025) 20.6% (+10.0%) Qwen-2.5-32B-Coder-Instruct ADP Data 36.8% (+26.2%)
WebArena (Zhou et al., 2024) AgentLab (Drouin et al., 2024) (de Chezelles et al., 2025) Qwen-2.5-32B-Coder-Instruct -10.9% Qwen-2.5-32B-Coder-Instruct ADP Data 22.9% (+12.0%)
AgentBench OS (Liu et al., 2024b)
AgentLM (Liu et al., 2024b) Llama-2-chat-70B - 9.0% Llama-2-chat-70B
AgentInstruct (Zeng et al., 2023) 21.5% (+12.5%) OpenHands CodeActAgent (Wang et al., 2025) Qwen-2.5-32B-Coder-Instruct -27.8% Qwen-2.5-32B-Coder-Instruct ADP Data 34.7% (+6.9%) monotonic gains with model size and consistent boosts from ADP training across agents and tasks, with ADP-trained models outperforming their base counterparts at every scale.
this section cite: ['b12', 'b4', 'b47', 'b42', 'b32', 'b57', 'b7', 'b5', 'b51', 'b42']

Section: DIVERSE DATA RESULTS IN CROSS-TASK TRANSFER
Table 6: Cross-task transfer with diverse vs. task-specific data. For each benchmark, we compare the same harness+model under task-specific "only" tuning and training on ADP corpus.
this section cite: []

Section: Agent Model Training Data Accuracy
SWE-Bench (Verified) (Jimenez et al., 2024;Chowdhury et al., 2024) OpenHands CodeActAgent (Wang et al., 2025) Qwen-2.5-7B-Instruct SWE-smith Only 1.0% Qwen-2.5-7B-Instruct ADP Data 10.4% Qwen-3-8B
CodeActInstruct + Code-Feedback 0.2% Qwen-3-8B SWE-smith Only 11.0% Qwen-3-8B
this section cite: ['b12', 'b4', 'b42']

Section: ADP Data 16.6%
WebArena (Zhou et al., 2024) AgentLab (Drouin et al., 2024) (de Chezelles et al., 2025) Qwen-2.5-7B-Instruct Go-Browse Only 16.0% Qwen-2.5-7B-Instruct ADP Data 20.1%
AgentBench OS (Liu et al., 2024b) OpenHands CodeActAgent (Wang et al., 2025) Qwen-3-8B AgentInstruct Only 21.5% Qwen-3-8B ADP Data 25.7%
GAIA (Mialon et al., 2023) OpenHands CodeActAgent (Wang et al., 2025) Qwen-2.5-7B-Instruct AgentInstruct Only 0.6% Qwen-2.5-7B-Instruct
this section cite: ['b57', 'b7', 'b5', 'b42', 'b23', 'b42']

Section: ADP Data 9.1%
We study whether data diversity helps agents generalize across tasks. Holding the agent setup and evaluation fixed, we compare training with different data mixtures: (i) Base (no tuning), (ii) Taskspecific only fine-tuning (e.g., SWE-smith Only, etc.), and (iii) ADP Data (as detailed in § 5), a mixed, cross-domain corpus. As shown in Table 6, ADP consistently outperforms task-specific tuning on the target task and, critically, avoids the negative transfer that single-domain tuning often induces on other tasks (Mueller et al., 2024;Kotha et al., 2024;Li et al., 2024).
Concretely, on SWE-Bench, ADP trained Qwen-2.5-7B-Instruct achieves 10.4%, versus 1.0% with SWE-smith Only; for Qwen-3-8B (Yang et al., 2025a), ADP reaches 16.6% versus 0.2% with CodeActInstruct + Code-Feedback and 11.0% with SWE-smith Only. On WebArena, ADP trained Qwen-2.5-7B-Instruct attains 20.1% versus 16.0% with Go-Browse Only. On AgentBench OS, ADP lifts Qwen-3-8B to 25.7% versus 21.5% with AgentInstruct Only. On GAIA, AgentInstruct Only results in 0.6% accuracy, while ADP improves it to 9.1%. Overall, mixed ADP tuning yields stronger in-domain accuracy and cross-task generalization than single-domain tuning. 6.3 ADP EASES ADAPTATION TO NEW AGENT HARNESSES Table 7: LOC for converting datasets to ADP. Dataset Total LOC AgentInstruct ∼1500 Code-Feedback 134 CodeActInstruct 269 Go-Browse 335 Mind2Web 476 Nebius SWE-Agent Trajectories 260 NNetNav (live+wa) 290 openhands-feedback 879 Orca AgentInstruct 155 SWE-Gym 221 SWE-smith 228 Synatra 145 Total 4892
Table 7 demonstrates the lines of code (LOC)foot_2 the authors and community contributors used to convert 13 datasets from distinct sources to the ADP schema. A single Raw→ADP converter per dataset performs the same normalization work (schema mapping, tool/action alignment, conversation formatting) that a traditional Raw→SFT converter would do for a specific agent harness. Therefore, LOC statistics in Table 7 are a reasonable proxy for the per-agent harness effort without ADP.  LOC ADP→SFT,j with ADP. Thus, as shown in Figure 2, the total conversion cost across the community now becomes linear with ADP (O(D + A) effort). Table 8 demonstrates that converting ADP standardized data to agent harness format takes an average of 77 LOC. Across the 13 we used, Cost ADP (A, D) ≈ 4892 + 77 × 100 = 12, 592 for A = 100, greatly less than the no-ADP setting. Additionally, adding a new harness only require writing one script converting ADP standardized data to SFT, greatly easing adaptation to new agent harnesses. Hence, ADP substantially reduces the community's collective effort required to develop scalable, reproducible agents.
this section cite: ['b26', 'b14', 'b15']

Section: CONCLUSION AND FUTURE WORK
ADP provides a practical, lightweight "interlingua" that unifies heterogeneous datasets into a single schema consumable by many agent harnesses, turning today's fragmented data landscape into a scalable training pipeline. Looking ahead, we see three immediate directions. (i) Multimodality: extending ADP beyond text to images, screen recordings, and other modalities to capture richer agent-environment interactions. (ii) Standardized evaluation: applying the same standardized "protocol" idea to evaluation and environment settings so that datasets, agents, and evaluations compose cleanly. (iii) Community growth and data quality: continuing open-source releases, stronger automated validation or even automated dataset conversion, to sustain scale while preserving quality. We believe that, by lowering integration costs and enabling systematic and scalable training and analysis across sources, ADP can catalyze the next wave of agent-training research and practice.
this section cite: []

Section: References
Ref_id:b0 Title: Openhands feedback dataset Year: (2024)
Ref_id:b1 Title: The claude 3 model family: Opus, sonnet, haiku Year: ()
Ref_id:b2 Title: Machine learning data practices through a data curation lens: An evaluation framework. FAccT '24 Year: (2024)
Ref_id:b3 Title: Agent-FLAN: Designing data and methods of effective agent tuning for large language models Year: (2024-08)
Ref_id:b4 Title: Introducing swe-bench verified Year: (2024)
Ref_id:b5 Title: Graham Neubig, Quentin Cappart, Russ Salakhutdinov, and Nicolas Chapados. The browsergym ecosystem for web agent research Year: (2025)
Ref_id:b6 Title: Mind2web: Towards a generalist agent for the web Year: (2023)
Ref_id:b7 Title: How capable are web agents at solving common knowledge work tasks? Year: (2024-07)
Ref_id:b8 Title: Go-browse: Training web agents with structured exploration Year: (2025)
Ref_id:b9 Title: Leveraging training and search for better software engineering agents Year: (2024)
Ref_id:b10 Title: Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation Year: (2025)
Ref_id:b11 Title: Qwen2. 5-coder technical report Year: (2024)
Ref_id:b12 Title: SWE-bench: Can language models resolve real-world github issues? Year: (2024)
Ref_id:b13 Title: Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web Year: (2024)
Ref_id:b14 Title: Understanding catastrophic forgetting in language models via implicit inference Year: (2024)
Ref_id:b15 Title: Revisiting catastrophic forgetting in large language model tuning Year: (2024-11)
Ref_id:b16 Title: Llava-plus: Learning to use tools for creating multimodal agents Year: (2024)
Ref_id:b17 Title: Agentbench: Evaluating LLMs as agents Year: ()
Ref_id:b18 Title: Real-world website navigation with multiturn dialogue Year: (2024)
Ref_id:b19 Title: Large language model agent: A survey on methodology, applications and challenges Year: (2025)
Ref_id:b20 Title: Empowering code large language models with evol-instruct Year: (2023)
Ref_id:b21 Title: Learning word vectors for sentiment analysis Year: (2011-06)
Ref_id:b22 Title: The landscape of emerging ai agent architectures for reasoning, planning, and tool calling: A survey Year: (2024)
Ref_id:b23 Title: Gaia: a benchmark for general ai assistants Year: (2023)
Ref_id:b24 Title: Toward generative teaching with agentic flows Year: (2024)
Ref_id:b25 Title: Evaluation and benchmarking of llm agents: A survey Year: (2025)
Ref_id:b26 Title: Multi-task transfer matters during instructiontuning Year: (2024-08)
Ref_id:b27 Title: Nnetnav: Unsupervised learning of browser agents through environment interaction in the wild Year: (2024)
Ref_id:b28 Title: Browser-assisted question-answering with human feedback Year: (2021)
Ref_id:b29 Title: Codegen: An open large language model for code with multi-turn program synthesis Year: (2023)
Ref_id:b30 Title: A survey of webagents: Towards next-generation ai agents for web automation with large foundation models Year: (2025)
Ref_id:b31 Title: Turning indirect knowledge into direct demonstrations for computer agents at scale Year: (2024-12)
Ref_id:b32 Title: Training software engineering agents and verifiers with SWE-gym Year: (2025)
Ref_id:b33 Title: Data and its (dis) contents: A survey of dataset development and use in machine learning research Year: ()
Ref_id:b34 Title: Heterogeneous data integration: Challenges and opportunities Year: (2024)
Ref_id:b35 Title: Qwen2.5: A party of foundation models Year: (2024-09)
Ref_id:b36 Title: SQuAD: 100,000+ questions for machine comprehension of text Year: (2016-11)
Ref_id:b37 Title: Androidinthewild: A large-scale dataset for android device control Year: (2023)
Ref_id:b38 Title: Toolformer: Language models can teach themselves to use tools Year: (2023)
Ref_id:b39 Title: AgentBank: Towards generalized LLM agents via fine-tuning on 50000+ interaction trajectories Year: (2024-11)
Ref_id:b40 Title: Voyager: An open-ended embodied agent with large language models Year: (2024)
Ref_id:b41 Title: Executable code actions elicit better llm agents Year: (2024)
Ref_id:b42 Title: Openhands: An open platform for AI software developers as generalist agents Year: (2025)
Ref_id:b43 Title: Agent-Gym: Evaluating and training large language model-based agents across diverse environments Year: (2025-07)
Ref_id:b44 Title: A challenge benchmark for web agents Year: (2024)
Ref_id:b45 Title: Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials Year: (2024)
Ref_id:b46 Title: Qwen3 technical report Year: (2025)
Ref_id:b47 Title: Swe-agent: Agent-computer interfaces enable automated software engineering Year: (2024)
Ref_id:b48 Title: Swe-smith: Scaling data for software engineering agents Year: (2025)
Ref_id:b49 Title: Webshop: Towards scalable real-world web interaction with grounded language agents Year: (2022)
Ref_id:b50 Title: React: Synergizing reasoning and acting in language models Year: (2022)
Ref_id:b51 Title: Agenttuning: Enabling generalized agent abilities for llms Year: (2023)
Ref_id:b52 Title: Data-centric artificial intelligence: A survey Year: (2025)
Ref_id:b53 Title: Agentohana: Design unified data and training pipeline for effective agent learning Year: (2024)
Ref_id:b54 Title: xLAM: A family of large action models to empower AI agent systems Year: (2025-04)
Ref_id:b55 Title: OpenCodeInterpreter: Integrating code generation with execution and refinement Year: (2024-08)
Ref_id:b56 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b57 Title: Webarena: A realistic web environment for building autonomous agents Year: (2024)
Ref_id:b58 Title: A USE OF LLMS We used LLMs to aid and polish writing for style and presentation Year: ()
