Title: Repo2Run: Automated Building Executable Environment for Code Repository at Scale
Abstract: Scaling up executable code data is significant for improving language models' software engineering capability. The intricate nature of the process makes it labor-intensive, time-consuming, and expert-knowledge-dependent to build a large number of executable code repositories, limiting the scalability of existing work based on running tests. The primary bottleneck lies in the automated building of test environments for different repositories, which is an essential yet underexplored task. To mitigate the gap, we introduce Repo2Run, the first LLM-based agent aiming at automating the building of executable test environments for any repositories at scale. Specifically, given a code repository, Repo2Run iteratively builds the Docker image, runs unit tests based on the feedback of the building, and synthesizes the Dockerfile until the entire pipeline is executed successfully. The resulting Dockerfile can then be used to create Docker container environments for running code and tests. We created a benchmark containing 420 Python repositories with unit tests for evaluation. The results illustrate that Repo2Run achieves an 86.0% success rate, outperforming SWE-agent by 77.0%. The resources of Repo2Run are available at https://github.com/bytedance/Repo2Run.

Section: Introduction
Code Repo Crawling
Unit tests Source code Repo GitHub Developer Dockerfile manually writing Repo tests/test_1.py tests/test_2.py tests/test_n.py test running 3. Test running and error handling. 4. Dockerfile written manually. crawling 2. Dependencies installation and env variables export. 1. Base environment selection.
this section cite: []

Section: Environment Building Executable Instance
Figure 1: The pipeline of code repository mining and manual environment building. Developers manually write Dockerfiles through iterative steps including base environment selection, dependency installation, test running, error handling, validating the environment by running unit tests.
Large language models (LLMs) have recently illustrated significant progress on solving software engineering issues [1], driving the advent of numerous coding LLM agents like MetaGPT [2], SWEagent [3], OpenHands [4], Copilot [5], and Cursor [6]. However, the lack of training environment original environment packageA fileA dirB commandA execute failed unexpected change packageB fileA dirB fileB "polluted" environment FROM python:3.10 … RUN commandA … Dockerfile ERROR: failed to solve: process "/bin/sh -c commandA" did not complete successfully: exit code: 1 build error Error log (a) A failed execution of "commandA" leads to "pollution" of the environment. (b) Building failure caused by adding "commandA" to the Dockerfile.
Figure 3: The illustration of a command executing failed and "polluting" the environment. (a) Failed commands like "commandA" can irreversibly "pollute" the environment by altering packages, files, or directories, making subsequent builds unstable. (b) To reproduce the changes, such a failed command "RUN commandA" needs to be added to the Dockerfile. However, adding it will lead to building failure. remains a significant challenge to advancing LLMs in software engineering -As shown in Figure 1, while static code is readily accessible through the GitHub API, building executable environments for code testing and patch validation demands extensive manual effort even for skilled developers 3 . This forces researchers to choose between (1) relying on unreliable static code metrics for rewarding (e.g., SWE-Fixer [7]) and SWE-RL [8], and (2) engaging developers to manually build limited repository environments (e.g., SWE-Gym [9] and SWE-Smith [10]) without scalability. Thus, an automated environment building infrastructure is urgently needed but remains underexplored.
In this paper, we introduce Repo2Run, an LLM-based agent designed to build executable environments. The key idea is that if the agent can successfully navigate the process of building a repository's dependencies and running its unit tests, its actions can be recorded and replayed to programmatically synthesize a runnable Dockerfile. This enables automated creation of isolated and consistent environments across different platforms via only one line command "docker build", eliminating "it works on my machine" problem [11].
Repo2Run addresses two key challenges in automating executable environment building:
1. Hard to explore a valid trajectory that can successfully build the environment. We found that without effective actions, the LLM agent would struggle to handle such a complex process. As shown in Figure 2, SWE-agent [3], a well-known LLM-based agent, often struggles with unresolved issues during environment building due to its lack of specialized actions. To address this challenge, Repo2Run uses an internal environment for building and an external environment for assistance, with nine actions designed to resolve these issues.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b2']

Section: 2.
Failing to synthesize a runnable Dockerfile without execution failure. As shown in Figure 3, we observed that some incorrect commands in the trajectory can "pollute" the environment, leading to errors during synthesis. To tackle this challenge, we design an adaptive rollback action. When encountering failed commands like "commandA" that may cause "pollution" to the environment, Repo2Run restores the internal environment to its pre-execution state. This ensures the building process remains consistent and repeatable using the Dockerfile, avoiding unexpected errors.
To demonstrate Repo2Run's effectiveness, we first created a benchmark of 420 latest Python repositories with unit tests from GitHub in 2024, as existing datasets include only a very small number of repositories [1]. We choose Python for two key reasons: (1) the most popular software engineering benchmarks (e.g., SWE-bench [1]) focus on Python repositories, and (2) Python is one of the most widely used programming languages today [12]. We then evaluated Repo2Run on this benchmark, finding it successfully built the environments for 361 repositories, achieving an 86.0% success rate, which is 63.9% higher than the leading baseline. Notably, Repo2Run outperforms the SWE-agent by 77.0%, demonstrating the effectiveness of the proposed agent.
We believe Repo2Run will serve as the foundational infrastructure, driving future research at the intersection of AI and Software Engineering and facilitating the community to efficiently scale up their executable code data and environments.
this section cite: ['b0', 'b0']

Section: Formulation
In this part, we define the task of executable environment building. Given a repository R, executable environment building aims to determine a suitable base image B and a building process P such that the resulting environment state S satisfies the verification ε, as defined in Equation ( 1):
ε(R, S) = 0, where S = δ(B, P)
Here, δ is the state transition function, and ε(R, S) = 0 indicates that the verification ε is satisfied.
this section cite: []

Section: State transition
• Environment state (S): The environment state S ∈ S represents the current state of the computer system, which encompasses all variables, files, cache, etc.
• Command sequence (C): The command sequence C ∈ C represents a set of individual commands. Each individual command C ∈ C refers to an instruction or action that can be executed in the environment via interfaces like bash, thereby changing the system state.
• State transition (δ): The state transition δ is a function that defines the process through which the system transitions from the start state S start ∈ S to the end state S end ∈ S after executing of a command sequence C, as defined in Equation ( 2):
δ : S × C → S, δ(S start , C) = S end(2)
this section cite: []

Section: Base image
• Empty state (S ∅ ): The empty state S ∅ ∈ S represents a completely bare operating system or a purely hypothetical state without any builds.
• Base image (B): The base image B ∈ S is a special type of environment state, typically managed by professional teams for user convenience, such as "python:3.10". Starting from the empty state S ∅ , the base image B can be created by executing a predefined command sequence C B ∈ C, i.e., B = δ(S ∅ , C B ). Users can utilize these base images by simply adding their names to the Dockerfiles.
this section cite: []

Section: Building process and state verification
• Building process (P): The building process P ∈ C is the command sequence designed to build the environment from the base image B. We denote the resulting state as S f ∈ S, where S f = δ(B, P).
• State verification (ε): The state verification ε is a Boolean function used to determine whether the resulting state S f successfully runs all tests in the repository R. ε(R, S f ) = 0 indicates that all tests in the repository R can be successfully run. Otherwise, at least one test fails.
Finish running all tests in the Docker container. Executable Dockerfile base image change rollback test running environment monitoring code editing bash commands dependency installation Initialize a Docker container with base image Python:3.10. Internal Environment External Environment Invoke the actions in the environment. Return observations. Multiple turns of environment-agent Interaction. 2 3 1 4 LLM 5 Synthesize Dockerfile by the trajectory. 6 Build executable environment and run the tests. 7 Dockerfile synthesizer Preset actions in the environment result processor tests/test_1.py tests/test_2.py tests/test_n.py tests/test_3.py . . .
this section cite: []

Section: Repo

this section cite: []

Section: Build phase
Record phase The build phase utilizes a dual-environment architecture: the internal environment with five actions for environment building, while the external environment with three actions assists the internal environment. The record phase converts the validated command sequence into a runnable Dockerfile for reconstructing the executable environment. See Appendix A for more examples of these actions.
Once all tests in the internal environment are successfully executed, Repo2Run transitions to the record phase.
In record phase, Repo2Run synthesizes a runnable Dockerfile based on the command sequence executed during the build phase. This Dockerfile serves as a precise record of the build process, and its execution can be viewed as a replay of the original build phase.
Next, we introduce the external environment, internal environment, and Dockerfile synthesizer.
this section cite: []

Section: External environment
The external environment serves as a bridge between the LLM and internal environment. It transmits actions from the LLM to the internal environment and relaying observations back. Following with the ReAct framework [13], the external environment maintains a history of thoughts, actions (i.e., successfully executed commands), and observations. This iterative process ensures that the LLM stays informed about the system's state and makes accurate decisions while effectively interacting with the internal environment.
• Rollback: As shown in Figure 3, commands like "commandA" 4 may "pollute" the environment, making it difficult to reproduce using a Dockerfile. When a command fails (i.e., returns a non-zero return code), the environment transitions to an uncertain state. Therefore, we introduce a rollback action to ensure the building process remains consistent. Specifically, before executing a command, we use "docker commit" to create a reserve snapshot of the current state (i.e., save the image at that point). If the return code is not 0, indicating a failure, the environment is rolled back to the most recent reserve snapshot. However, certain commands, referred to as safe commands (e.g., "cat"), generally do not alter the state of the environment and are exempt from rollback. For a complete list of safe commands, see Appendix B.2.
• Base image change: If the LLM agent determines that the current base image is unsuitable during the environment building process, it can reselect and switch to a new base image. This change invalidates the previous building process, clearing all executed commands and requiring a restart.
• Result processor: Command execution during interactions can produce extensive output (e.g., error logs), potentially overwhelming the LLM agent. To mitigate this, long outputs are truncated by this action, retaining only the initial and final sections up to a specified length.
this section cite: ['b11']

Section: Internal environment
The internal environment is a Docker container. Based on the latest data [14] from 2025, we select Python 3.10 as the default Docker base image due to its broadest adoption among Python versions.
• Environment monitoring: It serves as the eyes of the LLM agent in the internal environment, allowing it to observe the current environment state. These commands typically do not change the state of the environment. Basic commands like "ls" and "cat" are used to inspect directories and files, "find" is used to locate files, while more advanced commands like "pip list" and "pipdeptree" [15] retrieve the versions of installed libraries and dependency relationships.
• Dependency installation: It serves as installing third-party packages required for running tests, using tools like pip for Python and apt-get for system-level packages. To resolve potential conflicts (e.g., version constraints), a dependency management is implemented to assist the installation, where packages to be installed are added to a waiting list and resolved as needed. If installation succeeds, the environment is updated; otherwise, the rollback is performed. For detailed designs, see Appendix G.
• Test running: It serves as both a compass and a checkpoint, guiding the building process and verifying whether the environment runs all tests successfully. By executing "pytest" for unit tests, it determines whether the Docker container has been correctly built. If all tests run successfully, the process concludes; otherwise, error logs are sent to the LLM agent for further adjustments.
• Code editing: It enables the LLM agent to modify the code within the internal environment, including both inside and outside the repository. Direct code editing is rare, but sometimes needed for issues like syntax errors. To prevent the LLM agent from bypassing tests by directly altering or deleting test files, it is restricted from modifying or deleting the original test files within the repository.
• Bash commands: Like many other LLM-based agents [2][3][4], Repo2Run is allowed to invoke bash commands, allowing it fully operate within the Docker container. It enables the agent to execute dynamically defined actions that are not pre-specified, ensuring flexibility in handling various errors.
this section cite: ['b12', 'b13', 'b1', 'b2', 'b3']

Section: Dockerfile synthesizer
Once the LLM agent successfully runs all tests, the Dockerfile synthesizer converts the command sequence from the building process into a runnable Dockerfile. It processes each command sequentially, following the rules in Figure 5 to synthesize Dockerfile statements. The synthesis uses four Dockerfile keywords: "FROM", "ENV", "COPY", and "RUN". For more details and examples, see Appendix B.
• FROM: The FROM statement defines the base image and is typically the first line in a Dockerfile. If the LLM agent changed the base image, the FROM statement must be updated, and all subsequents are cleared to ensure consistency.
• ENV: The ENV statement persistently sets environment variables in the Docker container. When a command with "export" is detected, it is converted into an ENV statement. If the vaule of an environment variable is overwritten, this statement is updated accordingly.
• COPY: The COPY statement copies local files or directories into the Docker container. As shown in Figure 5, it is often used for code editing to import editing scripts (i.e., code_edit.py) and patches.
• RUN: The RUN statement executes commands in the container, with each line creating a new bash session. It cannot be used for persistent environment variables. Specially, for every installed packages, versions are recorded to ensure reproducibility, and the RUN statement is updated accordingly.
this section cite: []

Section: Experiment
We evaluate the effectiveness of Repo2Run on 420 Python code repositories. As the popular option, we select gpt-4o-2024-05-13 for all experiments, with the temperature uniformly set to 0.2.
this section cite: []

Section: Benchmark
To the best of our knowledge, there is no prior work similar to Repo2Run that automates executable environment building. Existing manually constructed datasets are limited to very few repositories [1]. To validate the capability of Repo2Run, we create a new benchmark consisting of filtered Python repositories from GitHub based on the following criteria:
• Creation date: To avoid the data leakage, we select repositories created in 2024, ensuring they are not part of mainstream LLM training data.
• Star count: To maintain quality, we only include repositories with more than 100 stars.
• Test directory: We focus on repositories likely to contain unit tests, identified using pytest, a leading Python testing framework compatible with tools like unittest. Pytest detects test files with a "test_" prefix or "_test" suffix, typically located in "test" or "tests" directories. We only retain repositories that have these directories.
Using these criteria, we crawled 449 repositories in December 2024 and filtered 420 containing at least one unit test to form our benchmark. For statistic of their scale, see Appendix D.
this section cite: ['b0']

Section: Evaluation metrics
• Dockerfile Generation Success Rate (DGSR): It indicates the percentage of attempts where the method successfully generates a runnable Dockerfile. To be considered successful, the generated Dockerfile must be able to build without errors. If the Dockerfile for a code repository successfully builds, it is regarded as a successful generation. Generating runnable Dockerfile is fundamental for successfully building the executable environment.
• Environment Building Success Rate (EBSR) 5 : It represents the percentage of attempts where the method successfully builds executable environments. For a successful building, the generated Dockerfile must not only build successfully but also allow tests to run by "pytest" in the Docker container. We are only concerned with whether tests can be executed, regardless of whether they pass or fail, as outcomes of tests may inherently vary within the repository.
this section cite: []

Section: Baselines
• pipreqs [16]: It is an automated tool that generates a "requirements.txt" file by analyzing the import statements in the Python scripts and identifying the necessary dependencies without LLM. Using the requirements.txt file generated by pipreqs, we create a Dockerfile. The detail is provided in Appendix I.1.
• LLM generator: The "README" file in a code repository usually contains environment building instructions. Therefore, we directly drive the LLM to read the "README" file and generate an executable Dockerfile accordingly.
• SWE-agent [3]: SWE-agent establishes a custom agent-computer interface (ACI) that uses the LLM agent's interaction with the repository environment by allowing actions such as reading files, editing files, and executing bash commands. Initially intended as an LLM agent for bug fixing, we preserve its framework and default settings, adjust its prompts, as shown in Appendix I.2.
this section cite: ['b14', 'b2']

Section: Experimental Results
The results of different baselines are presented in Table 1. We observe that Repo2Run consistently outperforms other baselines on both DGSR and EBSR. Repo2Run ultimately completed environment building for 361 code repositories, achieving an EBSR of 86.0%. It is 63.9% higher than the highest rate achieved by other methods, demonstrating great advantages. Due to the design of For pipreqs, the main failures come from two reasons. First, generating the requirements.txt fails when there are issues within the repository, such as encoding errors or syntax errors in the files. This happens in 30 repositories (7.1%). Second, even when requirements.txt is generated, it might not be downloaded properly due to package version conflicts. This occurs in 265 repositories (63.1%). Besides, both the LLM generator and SWE-agent fail to ensure that the generated Dockerfile can be successfully built due to the lack of an ensuring mechanism. Surprisingly, the ability of SWE-agent, a general agent framework, to generate Dockerfiles is even weaker than simply letting the LLM read the "README" file. This indicates that a general agent framework cannot guarantee the generation of runnable Dockerfiles. Ensuring mechanisms like rollback is necessary to effectively use the interactive information from the agent to generate runnable Dockerfiles.
As shown in Table 2, we manually categorized the 420 repositories into six domains. EBSR across these domains all exceed 80%, outperforming all baselines and demonstrating Repo2Run's consistent performance across different domains. Prior research [17,18] shows that the real-world benchmarks differs from synthetic ones and better reflect a model's true capabilities. Our benchmark is built entirely from real-world GitHub data, filtered only by repository creation date, star count, and test, with no additional filtering. Thus, it accurately represents the current distribution of Python repositories. Moreover, a previous real-world benchmark [18] includes 59.6% of AI/ML repositories, closely matching the 63.6% proportion in our benchmark. This alignment further confirms that our benchmark reflects the real-world scenarios.
this section cite: ['b15', 'b16', 'b16']

Section: Ablabtion of Repo2Run
To investigate the impacts of the dual-environment architecture and Dockerfile synthesizer separately, as two parts of Repo2Run, we separately remove each component of them. For the experiment without the dual-environment architecture, we retain only the internal environment's bash commands as the most basic interface and remove all other actions. For the experiment without the Dockerfile synthesizer, we directly instruct the LLM to synthesize a runnable Dockerfile.
Experimental result of the ablation study is shown in Table 3. We observe that removing the dualenvironment architecture and retaining only bash commands results in a 7.6% decrease in DGSR. The main reason for this drop is the removal of rollback and other designs, making the system more prone to entering uncertain states and subsequently failing to reproduce. In addition, EBSR shows a 44.3% decrease, primarily because the simplification of design makes it more difficult for the LLM agent to execute all tests in the internal environment. Besides, removing the Dockerfile synthesizer directly leads to an 80.5% drop in DGSR. This indicates that having the LLM directly generate Dockerfiles is unlikely to fully follow the event history, resulting in Dockerfiles that fail to build successfully. This also directly causes a sharp decline in EBSR. Specially, we also perform an ablation study for the rollback. The results show that removing the rollback causes 3.1% of generated Dockerfiles to become unrunnable, demonstrating the effectiveness of the rollback mechanism.
It is also observed that Repo2Run without the Dockerfile synthesizer is outperformed by the LLM generator. This is because the LLM generator leverages the "README" file, which provides a clear, simple and high-level overview of the executable environment building, allowing for more accurate Dockerfile synthesizer. In contrast, the event history-based approach lacks this context, making it harder for the LLM to fully understand the building goals. However, the Dockerfile synthesizer effectively utilizes the detailed event history, highlighting the complementary roles of both components of Repo2Run in generating a reliable runnable Dockerfile.
this section cite: []

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Swe-bench: Can language models resolve real-world github issues? Year: (2024)
Ref_id:b1 Title: Metagpt: Meta programming for a multi-agent collaborative framework Year: (2024)
Ref_id:b2 Title: Swe-agent: Agent-computer interfaces enable automated software engineering Year: (2024)
Ref_id:b3 Title: Opendevin: An open platform for ai software developers as generalist agents Year: (2024)
Ref_id:b4 Title: Github copilot -your ai pair programmer Year: ()
Ref_id:b5 Title:  Year: ()
Ref_id:b6 Title: Swe-fixer: Training open-source llms for effective and efficient github issue resolution Year: (2025)
Ref_id:b7 Title: Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution Year: (2025)
Ref_id:b8 Title: Training software engineering agents and verifiers with swe-gym Year: (2024)
Ref_id:b9 Title: Swe-smith: Scaling data for software engineering agents Year: (2025)
Ref_id:b10 Title: Using devcontainers to standardize student development environments: An experience report Year: (2020)
Ref_id:b11 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b12 Title: Historical trends in the usage statistics of Python version 3 for websites Year: ()
Ref_id:b13 Title: A command line utility to display dependency tree of the installed Python packages Year: ()
Ref_id:b14 Title: Generate pip requirements.txt file based on imports of any project. looking for maintainers to move this project forward Year: ()
Ref_id:b15 Title: When code completion fails: a case study on real-world completions Year: (2019)
Ref_id:b16 Title: Evocodebench: An evolving code generation benchmark aligned with real-world code repositories Year: ()
Ref_id:b17 Title: Marscode agent: Ai-native automated bug fixing Year: (2024)
Ref_id:b18 Title: Trae agent: An llm-based agent for software engineering with test-time scaling Year: (2025)
Ref_id:b19 Title: Xipeng Qiu, Xuanjing Huang, and Tao Gui. The rise and potential of large language model based agents: A survey Year: ()
Ref_id:b20 Title: Self-collaboration code generation via chatgpt Year: ()
Ref_id:b21 Title: Flows: Building blocks of reasoning and collaborating AI Year: ()
Ref_id:b22 Title: Parsel: Algorithmic reasoning with language models by composing decompositions Year: (2023)
Ref_id:b23 Title: Language agent tree search unifies reasoning, acting, and planning in language models Year: (2024)
Ref_id:b24 Title: CONLINE: complex code generation and refinement with online searching and correctness testing Year: ()
Ref_id:b25 Title: Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges Year: (2024)
Ref_id:b26 Title: Transformer feed-forward layers are key-value memories Year: (2021-11-11)
Ref_id:b27 Title: Autonomous large language model agents enabling intent-driven mobile GUI testing Year: ()
Ref_id:b28 Title: Chatdev: Communicative agents for software development Year: (2024)
Ref_id:b29 Title: Metagpt: Meta programming for A multi-agent collaborative framework Year: (2024)
Ref_id:b30 Title: Coverup: Coverage-guided llm-based test generation Year: ()
Ref_id:b31 Title: Xuat-copilot: Multi-agent collaborative system for automated user acceptance testing with large language model Year: ()
Ref_id:b32 Title: Axnav: Replaying accessibility tests from natural language Year: (2024)
Ref_id:b33 Title: Toolcoder: Teach code generation models to use API search tools Year: ()
Ref_id:b34 Title: Llm-assisted static analysis for detecting security vulnerabilities Year: ()
Ref_id:b35 Title: Agentcoder: Multiagent-based code generation with iterative testing and optimisation Year: ()
Ref_id:b36 Title: Aegis: An agent-based framework for bug reproduction from issue descriptions. FSE Companion '25 Year: (2025)
Ref_id:b37 Title: Static code analysis in the AI era: An in-depth exploration of the concept, function, and potential of intelligent code analysis agents Year: ()
Ref_id:b38 Title: Repairagent: An autonomous, llm-based agent for program repair Year: ()
Ref_id:b39 Title: Repomaster: Autonomous exploration and understanding of github repositories for complex task solving Year: (2025)
Ref_id:b40 Title: Codevisionary: An agent-based framework for evaluating large language models in code generation Year: (2025)
Ref_id:b41 Title: Openrca: Can large language models locate the root cause of software failures? Year: (2025)
Ref_id:b42 Title: Vul-r2: A reasoning llm for automated vulnerability repair Year: (2025)
Ref_id:b43 Title: Gittaskbench: A benchmark for code agents solving real-world tasks through code repository leveraging Year: (2025)
Ref_id:b44 Title: Aligning the objective of llm-based program repair Year: (2024)
Ref_id:b45 Title: Boosting vulnerability detection of llms via curriculum preference optimization with synthetic reasoning data Year: (2025-08-01)
Ref_id:b46 Title: Reposvul: A repository-level high-quality vulnerability dataset Year: (2024)
Ref_id:b47 Title: Evaluating code completion via real-world repositories Year: (2024)
Ref_id:b48 Title: Vuleval: Towards repository-level evaluation of software vulnerability detection Year: (2024)
Ref_id:b49 Title: OSS-Fuzz-Gen: Automated Fuzz Target Generation Year: (2024-05)
Ref_id:b50 Title: Starter: Helping you get started with containerized apps Year: ()
Ref_id:b51 Title: Yeoman generator for docker Year: ()
Ref_id:b52 Title: Dockerizeme: Automatic inference of environment dependencies for python code snippets Year: (2019)
Ref_id:b53 Title: Dockergen: A knowledge graph based approach for software containerization Year: (2021)
Ref_id:b54 Title: Less is more? an empirical study on configuration issues in python pypi ecosystem Year: (2024)
Ref_id:b55 Title: Humpback: Code completion system for dockerfiles based on language models Year: (2020)
Ref_id:b56 Title: Automatically generating dockerfiles via deep learning: Challenges and promises Year: (2023)
Ref_id:b57 Title: Top 100 stars in python Year: ()
