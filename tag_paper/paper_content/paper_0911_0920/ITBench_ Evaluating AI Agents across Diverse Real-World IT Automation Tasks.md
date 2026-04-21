Title: ITBench: Evaluating AI Agents across Diverse Real-World IT Automation Tasks
Abstract: Realizing the vision of using AI agents to automate critical IT tasks depends on the ability to measure and understand effectiveness of proposed solutions. We introduce ITBench, a framework that offers a systematic methodology for benchmarking AI agents to address real-world IT automation tasks. Our initial release targets three key areas: Site Reliability Engineering (SRE), Compliance and Security Operations (CISO), and Financial Operations (FinOps). The design enables AI researchers to understand the challenges and opportunities of AI agents for IT automation with push-button workflows and interpretable metrics. ITBench includes an initial set of 102 realworld scenarios, which can be easily extended by community contributions. Our results show that agents powered by state-of-the-art models resolve only 11.4% of SRE scenarios, 25.2% of CISO scenarios, and 25.8% of FinOps scenarios (excluding anomaly detection). For FinOps-specific anomaly detection (AD) scenarios, AI agents achieve an F1 score of 0.35. We expect ITBench to be a key enabler of AI-driven IT automation that is correct, safe, and fast. ITBench, along with a leaderboard and sample agent implementations, is available at https://github.com/ibm/itbench.

Section: Introduction
Modern IT systems are driving many facets of our economy. They have grown significantly in complexity with the adoption of cloud computing and agile development practices (Harvard Business Review Research Report, 2022;Trask, 2025). Effective management of these systems is becoming extremely challenging as corporations struggle to keep up with this growing complexity. Various IT personasranging from Chief Information Officers to Site Reliability Engineers and Security and Compliance officers-and IT engineers in general are struggling to ensure resiliency, reliability, security, and cost effective operations of IT Systems.
The recent CrowdStrike outage highlighted these challenges as it brought down our society's most critical systemsfrom hospital services to air travel-and was estimated to cost US Fortune 500 companies a staggering $5.4 billion (Kerner, 2024). This incident underlined the critical need for intelligent IT incident resolution, with compliance and risk management capabilities, a topic also addressed in the Digital Operational Resiliency Act (DORA) in Europe (Parliament and the Council of the European Union, 2024).
The rising popularity of AI agents and their projected ability to handle intricate tasks have increased the demand for AI agents managing IT systems (John, 2024;Miguel Carreon, 2024;Pujar et al., 2023). Given the complexity of IT tasks, a major hurdle for this research is establishing systematic methods to assess the effectiveness of AI agents prior to their production deployment (Bogin et al., 2024;Kapoor et al., 2024). Consequently, there is an urgency to develop methods for evaluation of AI agents based on real IT tasks and their corresponding environments. This paper addresses this critical need and presents ITBench, a first-of-its-kind framework that is both comprehensive and visionary for benchmarking real-life IT automation tasks. The goal of ITBench is to measure the performance of AI Figure 1: Sample personas and IT tasks. Bell icon represents event-triggered tasks. Information icon represents other tasks such as data analysis, preventive maintenance tasks, or continuous optimization.
agents across a wide variety of complex and real-life IT tasks across personas, including Site Reliability Engineering (SRE), focusing on availability and resiliency; Compliance and Security Operations (CISO), ensuring compliance and security of IT implementations; and Financial Operations (FinOps), enforcing cost efficiencies and optimizing return on investment, among others (as shown in Figure 1).
ITBench aims to advance innovation and establish new standards in the field. Our contributions can be summarized along the following three axes:
• Reflecting the real world: ITBench addresses the IT automation requirements that are relevant and prevalent in production settings. SRE scenarios are based on realworld incidents observed in our own SaaS products. CISO scenarios are based on CIS benchmark (for Internet Security, CIS). FinOps scenarios are identified by the FinOps Foundation (Foundation, 2025a) through key business outcomes.
• Being open and extensible with comprehensive IT coverage: We view ITBench as a central hub for benchmarking AI-driven solutions across diverse IT automation use cases. To support this, we provide IT benchmark suites and a framework for vertical expansion (i.e., adding more scenarios) and horizontal expansion (i.e., adding more personas), ensuring extensive coverage of IT tasks. ITBench is an open-source framework built with open-source technologies, while allowing organizations with proprietary technologies to use it for developing and benchmarking their solutions.
• Enabling automated evaluation with partial scoring:
ITBench is designed to provide constructive feedback to drive improvements in the design of agentic solutions for IT problems. It includes a comprehensive evaluation framework and leaderboard that provide feedback to users at various stages of their agents' reasoning process.
ITBench provides push-button deployment and tooling for setting up environment, runtime agent, guardrail engine, as well as authorization and authentication. It allows developers and researchers to build novel solutions for managing complex IT systems. Currently, ITBench addresses reac-tive problems, including incidents diagnosis and resolution, compliance assessments in regulated environments for new controls, and cost management events. In the future, we plan to expand on benchmark evaluation capabilities and include new benchmarks for additional IT processes. Currently, ITBench comprises an initial set of 102 scenarios spanning across SRE (42), CISO (50), and FinOps (10), with respective successful scenario handling rates of 13.8%, 25.2%, and 25.8% (refer to Section 4).
We believe that, similar to the highly influential SWEBench (Jimenez et al., 2024), our new ITBench framework-which encapsulates and measures the ability of AI agents to automate complex, real-world IT tasks-will spur a comparable acceleration in the performance of real-world IT AI agents.
this section cite: ['b76', 'b37', 'b32', 'b61', 'b15', 'b34', 'b31']

Section: Related Work
ITBench targets a comprehensive set of tasks for a wide range of personas within IT automation. The initial release of ITBench focuses on evaluating scenarios within IT Operations (ITOps). Figure 1 illustrates currently targeted personas and exemplar tasks that they are routinely facing.
There is clearly rising interest in developing benchmarks to evaluate AI and ML techniques in ITOps with specific focus on SRE, CISO, and FinOps.
TrainTicket (Zhou et al., 2018) provides 22 scenarios collected through an industrial survey of real-world incidents, using hardcoded faults in the TrainTicket application to focus on fault localization. AIOpsLab (Chen et al., 2024a) provides 10 SRE-focused scenarios (referred to as "problems") utilizing a real environment (system) integration that allows interactive access to text, time series, and tabular data. InsightBench (Sahu et al., 2024) provides 100 scenarios to analyze ticket data using static tabular data and synthetic scenarios. TSB-AD (Liu and Paparrizos, 2024a) focuses on anomaly detection with 40 synthetic scenarios.
CIS-Benchmark (CIS, 2024) provides best practices for securing IT infrastructure. Despite the name of "benchmark," it offers only recommendation policies; it provides no experimental platform. Recently, Cloud Native Compute Foun- dation (CNCF) Sandbox project (OSCAL-compass, 2024) released an SDK to support the translation of the CIS human readable formats into OSCAL (OSCAL, 2024). OSCAL was developed by the National Institute of Standards and Technology for programmatic usage in compliance automation. ITBench CISO automation leverages this technology to assess policy requirements.
FinOps Foundation (Foundation, 2025a) provides benchmarks that compare cloud financial performance across organizations and departments, focusing on KPIs such as resource utilization efficiency, contract coverage, and cost apportionment. These benchmarks help assess cloud efficiency by evaluating internal and external metrics, fostering structured, collaborative approaches to cloud optimization.
While existing benchmarks are valuable resources for specific tasks and use cases, and highlight the critical need for systematic benchmarking, they are limited in reflecting real-world IT problems, covering broad IT landscape, and automating evaluation. These limitations are addressed in ITBench, as shown in Table 1.
this section cite: ['b93', 'b65', 'b54', 'b54']

Section: ITBench
ITBench is a systematic benchmarking framework and runtime environment designed to evaluate AI agents tasked with automating IT operations, incorporating a robust architecture (see Figure 2) comprising the AI Agent, Scenario Specification and Environment, Evaluator, and Leaderboard to facilitate comprehensive performance assessment.
Here, we present a brief overview of the key components: 1) Scenario Specification and Environment, 2) AI Agents, and 3) Leaderboard. More details are in Appendix B.
this section cite: []

Section: Scenario Specification and Environment
The bench incorporates a collection of problems that we call scenarios. For example, one of the problems in ITBench is to resolve a "High error rate on service order-management" in a Kubernetes environment. Another example that is rele-vant for the CISO persona involves assessing the compliance posture for a "new control rule detected for RHEL 9." A fundamental challenge is to emulate such problems in a manageable testbed environment. A scenario environment is an operational testbed in which a specific problem(s) occurs.
A scenario p generally corresponds to a problem to be solved in ITBench. We formalize p as a tuple < M, E, T, D >, where the variables are as follows:
Scenario Specification. M represents metadata and deployment descriptors, for each scenario, which is stored in the Scenario Specs database in ITBench (see Figure 2). Exemplar metadata elements per scenario include scenario_name, scenario_description, scenario_domain, scenario_class, scenario_complexity, and scenario_groundtruth (see Table 2), which are defined below:
• scenario_name is name given to a scenario. For example, a scenario in ITBench has the name "Recommendation Service Cache." • scenario_description describes the scenario. An example of a description of the scenario is "Recommendation Service in Astronomy Shop has a cache failure." • scenario_domain represents different personas within IT automation-namely "SRE," "CISO," and "FinOps." • scenario_class is used to group similar scenarios, such as "Kyverno-opa," "Kyverno-update',' "CacheFailure," "HighCPU," and "CorruptImage." • scenario_complexity captures the difficulty of a problem and is defined using domain knowledge. Figure 4a shows the breakdown of SRE, CISO, and FinOps scenarios in the bench. Figure 4b, 4c, and 4d shows scenario_complexity distribution for SRE, CISO, and FinOps, respectively. SRE scenarios are developed based on real-world incidents observed in our own SaaS products. CISO scenarios are based on CIS benchmark (for Internet Security , CIS). FinOps scenarios were developed based on "Domains" and "Capabilities" identified by the FinOps Foundation (Foundation, 2025a) to describe key business outcomes.
• scenario_groundtruth records task-specific outcomes that
ITBench Benchmark Runner Scenario Environment Application Stack Observability Stack Action Info. System state Setup/Cleanup per scenario Evaluator Agent result Benchmark Builder Leaderboard Rank & Publish Agent ITBench Build & Register Agent Builder Develop & Register Scenario Specs Metric 1 Metric 2 Metric N ... ... the Evaluator uses to compare against the agent's expected output. For instance, in incident resolution for SREs, the ground truth for the Diagnosis task includes a list of entities involved in the fault propagation chain, the actual fault propagation chain(s), and fault conditions, while for the Mitigation task, it captures plausible mitigation actions.
Environment. E represents an an operational testbed where the problem occurs. Components within the environment expose APIs to observe and control the environment. When the Agent Builder registers the agent for benchmarking, the Benchmark Runner (see Figure 2) randomly selects a set of scenarios, which may be optionally filtered based on the agent_type and agent_level. Next, the Benchmark Runner iterates through the set of scenarios, and for each scenario it instantiates a testbed. An example of an environment is a Kubernetes cluster installed with OpenTelemetry Astronomy Shop Demo application (Community, 2024), observability stack including Grafana (gra), Loki (lok), Jaeger (jae), and Prometheus (pro), along with mechanisms that induce problem(s) in the environment.
Triggering Events. T is a set of triggering events that occur due to manifestation of a specific problem in the environment. Tools are configured to observe the environment and raise triggering events on problematic conditions. An example of a triggering event is "High Error Rate on adservice," which may be triggered in the environment due to cache failure problem.
Desired Outcome. D defines the automation objective and represents the ultimate goal. For instance, in case of SRE incident resolution, the ultimate goal is to clear T in the E.
this section cite: []

Section: AI Agents
In IT automation, the different personas are focused on a specific desired outcome, which defines their automation goals. For SREs, incident resolution is the primary objective. Achieving this can involve multiple steps, such as diagnosing an incident, or a single step, like generating a diagnosis report. CISO persona focuses on the regulatory controls posture assessment process, including Collect evidence and Scan assessment posture tasks. FinOps persona focuses on the cost management, where sample tasks include Identify inefficiency and Mitigate inefficiency. During evaluation, each step (task) is assessed independently and is measured using well defined metrics; see Table 3.
The goal of ITBench is to evaluate AI agents on a broad range of real-world IT automation tasks that are otherwise performed by SREs, FinOps, and CISO personas.
In this paper, an AI agent is defined as an autonomous or semi-autonomous software program that uses an LLM to plan, make decisions, interact with the target environment, and execute actions to achieve goals. An AI agent is expected to successfully handle any of the scenarios in the ITBench, by interacting with the environment.
As shown in Figure 3, agent and environment form a Partially Observed Markov Decision Process (POMDP), where the state is the snapshot of the environment. The state transitions are determined by the environment, which are then (partially) observed by the agent.
Given a scenario p instantiated in an environment E, an agent probes the environment via one of the tools and receives an observation o t ∈ O, based on which, it decides the next action:
a t = f (o t |ō t-1 ; āt-1 )(1)
Here f is the agent's decision function. ōt-1 is the sequence of observations up to time t -1 and āt-1 is the sequence of actions taken up to t -1.
Hard 20.0% (c) CISO scenario complexity. Easy 20.0% Medium 30.0% Hard 50.0% (d) FinOps scenario complexity. Initially, o 0 may be a triggering event showing a problematic state s 0 of the environment. Given state s t-1 and action a t-1 , the environment transitions to the next state:
s t = g(s t-1 , a t-1 )(2)
The observation o t is determined as a function of the state and is in general a proxy for the environment state s t , hence the formulation can be thought of as a POMDP:
o t = h(s t )(3)
The set A of actions is defined as Q {⊥}, where Q is the set of tools and ⊥ represents the "stop action" by the agent. We define t * as the time when the agent stops:
t * = min{t|a t = ⊥}(4)
An agent reflects on the result to guide its next action, continuing until the final goal is achieved. Given a set of scenarios that the agent works on, it targets to maximize the success defined as follows:
E p∼πp (I(g(s p t * , f (o t * |ō t * -1 , āt * -1 )) = s p G )) (5
)
where I is an indicator function comparing the terminating state with goal state and π is the distribution of scenarios.
this section cite: []

Section: BASELINE AI AGENTS
We developed baseline agents SRE-Agent for SRE, Compliance Assessment Agent for CISO, and FinOps-agent for FinOps. Each of these agents uses state-of-the-art agentic techniques such as ReAct-based planning (Yao et al., 2023), reflection (Shinn et al., 2023), and disaggregation (Xu et al.,  2023). Reflection techniques include syntax checking/linting, semantic validation (Xie et al., 2024a), and llm-as-ajudge (Zheng et al., 2023).
We use the open-source CrewAI framework (cre) to create and manage agents. The agents can be configured to use various LLMs either through watsonx, Azure, or vLLM. Each agent is initialized with a prompt that describes its goal, the context, the tasks, and the expected output format. In-context learning examples are included to guide the agent and demonstrate tool usage. Agents use tools to interact with the environment for information gathering.
Logs, traces, and metrics collected during the diagnosis process would overwhelm the context window of any LLM currently available due to large volume of data. Therefore, agents targeting the SRE or FinOps persona are equipped with specialized tools to interact with the environment (refer to Figure 3): 1) NL2Traces to extract trace data in a structured format, 2) NL2Metrics to analyze key system metrics, 3) NL2Logs to parse log data effectively, 4) NL2Kubectl to perform Kubernetes-specific operations, and a summarization tool to condense extensive data into actionable insights. For example, the agent may use the NL2Kubectl tool to "list all of the pods in the default namespace." In turn, the NL2Kubectl tool uses an LLM to transform the utterance into an executable command: "kubectl get pods -n default."
Similarly, the compliance assessment required for new regulations and technologies, with the evidence and diverse policy languages, would be overwhelming if submitted directly to LLMs. The compliance agents designed for CISO compliance assessment automation are equipped with specialized tools. These tools include capabilities to 1) generate policies such as Kyverno or OPA Rego Policy as Code starting from natural language specifications, 2) generate scripts for the collection of evidence, 3) access code repositories such as git to facilitate GitOps workflows for code management, and 4) deploy and execute the generated policies to accomplish the assessment task.
this section cite: ['b87', 'b67', 'b92']

Section: Leaderboard
ITBench includes a leaderboard to promote reproducibility and comparative analysis, following the AI common task framework (Donoho, 2019;Varshney et al., 2019). The leaderboard offers a predefined, extensible set of performance metrics designed to provide clear insights into agent performance relative to the evaluation criteria.
ITBench devises scoring methods for partially correct solutions to provide meaningful feedback for summative assessments. This comprehensive approach establishes a new standard for evaluating and advancing AI-driven solutions in IT automation. For each scenario that an agent works on, upon task completion, the ITBench records the final system state, which is then used at the end of all scenario runs along with the pre-defined ground truth data to validate how well the agent performed across all the scenarios.
We are open-sourcing a small subset (11 out of 102) of scenarios along with the baseline agents to help the community become familiar with ITBench through practical examples. We reserve the remaining scenarios in ITBench to benchmark and evaluate the submitted agentic solutions.
this section cite: ['b78']

Section: Results

this section cite: []

Section: Evaluation Setup
To understand the impact of reasoning and planning capabilities of LLMs on ITBench scenarios, we instantiate our agents using different LLM models, both for natural language reasoning and code generation. Specifically, we employ GPT-4o (checkpoint version 2024-11-20), Llama-3.3-70B-instruct, Llama-3.1-8B-instruct, and Granite-3.1-8B-instruct for tasks that rely on natural language understanding and reasoning. For code-focused use cases, we utilize GPT-4o-mini, Llama-3.1-405b-instruct, and Mixtral-8x7b-instruct. All models use a context window of 128K tokens, enabling them to process more extensive input sequences.
We conduct our experiments primarily on AWS EC2 instances (m4.xlarge), although ITBench can also be readily deployed on a consumer-grade laptop using a pseudo-cluster, thus making it easier to develop AI agents (Appendix C.4.1) Below, we provide an overview of our baseline agents' performance across ITBench scenarios for SRE, CISO, and FinOps. Our findings indicate that both open-source and proprietary models often struggle with real-world tasks, underscoring the importance of benchmarks that push the limits of reasoning and planning in foundation models. For 4 std error for each metric is listed.
5 FL (NTAM) = Normalized topology-aware metric for root cause, FPC (NTAM) = Normalized topology-aware metric for fault propagation chain (value between 0 and 1.0), MTTD = Mean time to diagnosis (seconds), MTTR = Mean time to repair (seconds). Bold: the best performance. 6 Details of NTAM are available in Appendix C.6.3 more comprehensive results and detailed scenario-level discussions, please refer to Appendix C (SRE), Appendix D (CISO), and Appendix E (FinOps).
this section cite: []

Section: Overall Results
Table 4, Table 5, and Table 6 show the performance of SRE-agent, CISO-agent, and FinOps-agent respectively.
this section cite: []

Section: SRE.
We measure the efficiency of SRE-Agent on its ability to diagnose and mitigate production incidents (e.g., "a high error rate on frontend service").
Diagnosis efficiency is measured using pass@1(Chen et al., 2021) (i.e., identifying the cause as mentioned in ground truth), NTAM (Normalized Topology-Aware Metric) for root cause and fault propagation chain, and time to diagnosis. 1 Mitigation efficiency is measured in terms of pass@1 (i.e., whether the alert was cleared) and mean time to repair.
As shown in Table 4, across all SRE scenarios, GPT-4o consistently outperforms the other models, achieving the highest pass@1 scores for diagnosis (13.81%) and mitigation (11.43%), as well as the highest score on NTAM (FL and FPC) metrics. Llama-3.3-70B ranks second overall, trailing GPT-4o on most metrics. The 8B models have lower mitigation success rate. Surprisingly, Granite-3.1-8B (without any specialized finetuning) achieves higher accuracy than Llama-3.3-70B on the diagnosis task.
Removing trace data can drastically reduce success rates (see Table 20 and Table 21 in Appendix). For instance, GPT-4o's pass@1 in diagnosis falls from 13.81% with traces to 9.52% without them, and mitigation plummets to 2.86%. This highlights the critical role of system observability in SRE, which ITBench can evaluate under varying conditions. Because there is no perfect observability in practice, how to guide SRE-agents to collect new observability data and to help SRE-agents reason about failures with incomplete observability is an important but open problem.
1 NTAM is Normalized topology-aware metric that measures the quality of the predicted root cause and fault propagation chains using a system and application topology. Refer to Appendix C.6.3.
this section cite: []

Section: CISO.
We measure the efficacy of our agents across the four scenario classes introduced in Table 2. Each scenario_class imposes a distinct set of CIS-benchmarks requirements (e.g., "minimize the admission of containers wishing to share the host network namespace"), has a specific level of complexity (e.g., Easy, Medium, or Hard), and generates scenariospecific code artifacts.
The efficacy of CISO-agents is measured based on the ability to detect artifact misconfigurations (aka non-compliance, e.g., no minimum count of containers sharing namespace, or the count is above the threshold), or confirm proper configurations (aka compliance), within the varied environments of the scenario classes randomly injected with misconfigurations. Notably, GPT-based models dominate on both pass@1 and Time to Process metrics. The pass@1 is nearly two times better than second-best performing model, while the TTP shows a handling of the scenarios in the minimal time across our scenario classes.
FinOps. In addition to the standard event-driven scenarios, ITBench was extended to support non-alert-driven scenarios for the FinOps persona, demonstrating its extensibility. In particular, we added data insights and anomaly detection scenarios to ITBench. Table 6 presents our results in all FinOps usecases. We report pass@1 score for data insights, diagnosis, and mitigation tasks, and F1 score and rank score for anomaly detection. F1 score measures the precision and recall abilities of the agent to identify anomalous costs with regard to the ground truth. The rank score measures the relative ranking of the anomalies as determined by the agent with regard to the ground truth ranking. GPT-4o consistently outperforms all other models, achieving a 33% pass rate for diagnosing the origin of the cost increase alert, 29% accuracy in data insights scenarios, and 0.6 F1 score in anomaly detection. Refer to Appendix E.5 for futher details.
this section cite: []

Section: Impact of Scenario Complexity
SRE. We categorize scenarios as Easy, Medium, or Hard based on factors such as fault propagation chain length, number of resolution steps, and the diversity of technolo-  gies involved, as described in Equation ( 6). Our results show that success rates (pass@1) clearly decline as the scenario_complexity increases. Even the best performing model, GPT-4o, diagnosed only 36%, 7.73%, 5% of Easy/Medium/Hard cases (Table 18) and mitigated just 21%, 12.27%, 0% (Table 19). None of the models could mitigate the Hard scenarios, even though over 50% of Easy scenarios were mitigated. Notably, GPT-4o is the only model that successfully diagnosed multiple Hard scenarios.
this section cite: []

Section: CISO.
The complexity of the CISO scenarios is directly mapped to scenario classes. For example, sce-nario_complexity of Kyverno scenarios is Easy, sce-nario_complexity of k8s-opa and rhel-opa is Medium, while scenario_complexity of Kyverno-update scenarios is Hard. All models struggle, as expected, as the difficulty of the scenarios increases from the Easy kyverno class to the Hard kyverno-upadate class.
FinOps. Currently, ITBench includes 2 Easy, 3 Medium, and 5 Hard scenarios. None of the models were able to resolve the Hard scenarios. GPT-4o performs better in anomaly detection and alert-driven scenarios, while the LLaMA-3.3-70B-Instruct model achieves comparable performance to GPT-4o in data insight scenarios.
this section cite: []

Section: A Case Study on SRE-Agent Failures
Understanding the decision process of LLM-based agents is challenging due to the complexity of agentic systems but is feasible through detailed agent trajectory logging and a structured prompting framework. We log each input and output for planning agents and tools, including the ReAct-style "Thought" step, enabling us to distinguish between highlevel reasoning errors (e.g., flawed strategy) and low-level tool errors (e.g., malformed commands), enabling practical analysis and guiding design improvements.
this section cite: []

Section: Analyzing Lower-Level Tool Calling and Execution
Figure 5 shows tool usage and failure types across models. NL2Kubectl dominates usage, suggesting overreliance. Encouraging the agent to make a more balanced use of other tools, such as NL2Traces, could be useful, especially when kubectl commands alone are insufficient. Smaller models (e.g., granite-3.1-8B-instruct, llama-3.1-8B-instruct) show more invalid tool calls, syntax errors, and repeated invocations, indicating lower accuracy and efficiency.
this section cite: []

Section: Quantitative Analysis of High-Level Reasoning
We quantify reasoning by aligning each exploration path with the ground-truth fault-propagation chain. An effective agent is expected to focus its exploration around this chain, while significant deviations may signal reasoning
ITBench g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . 0 500 1000 1500 2000 2500 Number of Actions invalid tool g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . getalerts g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . nl2traces g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . nl2kubectl g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . nl2metrics g p t -4 o g r a n it e -3 .1 -8 B -in s t . ll a m a -3 .1 -8 B -in s t . ll a m a -3 .3 -7 0 B -in s t . nl2logs Total Incorrect Tool Argument Repeated Usage Syntax Error Execution Failure Successful Execution Figure 5: SRE-Agent Tool Usage Distribution flaws. Based on this insight, we introduce two evaluation metrics: (i) Detoured Services: |V visited \ V gt |, the number of visited services that are not on the ground-truth chain (smaller → more focused search); and (ii) Relative Covered Services: |V visited ∩ V gt | |V gt | , the fraction of ground-truth services visited (closer to 1→better coverage).
As shown in Figure 6, successful diagnosis trajectories show fewer detours and higher coverage than unsuccessful ones, validating the metrics. Among successful trajectories, GPT-4o shows detours (Kolmogorov-Smirnov p-value ≥0.123) and coverage (p-value≥0.089) that are comparable to other models (i.e., Granite-3.1-8B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct), while achieving higher coverage than Llama-3.1-8B-Instruct (p-value=0.024). This suggests that successful agents tend to follow similar reasoning patterns. For unsuccessful trajectories, GPT-4o significantly surpasses all baselines, showing both fewer detours (p-value≤0.001) and greater coverage (p-value≤ 0.011). These results underscore ITBench's utility in revealing insightful patterns in agent reasoning and overall performance.
this section cite: []

Section: Discussion and Conclusion
We presented ITBench, the first framework and experimental platform to benchmark AI Agents for IT automation tasks. ITBench strives to capture the complexity of modern IT systems and the diversity of IT tasks. The reproducibility of ITBench ensures the community-driven effort despite inherent nondeterminism of large-scale IT systems.
One of the key design principles of ITBench is ensuring its flexibility to support diverse areas of different IT systems and its extensibility to new scenarios. While the current scope of ITBench is comprehensive and representative, we plan to further enrich the benchmark suites by adding other important processes essential to modern IT automation. Furthermore, we plan to expand our benchmarking beyond event-triggered scenarios. We are actively working to expand scenario coverage for the supported processes and promote growth through open-community contributions. We invite the community to reproduce their real-world-inspired incidents in a synthetic sandboxed environment leveraging the ITBench. We expect that everyone contributing can bring their expertise to the table.
We expect ITBench to drive the innovations of AI agentbased techniques with a direct impact on the safety, efficiency, and intelligence of today's IT infrastructures. With ITBench, we are starting to explore many deep, exciting open problems: How to develop domain-specific AI agents that specialize in certain types of IT tasks? How to orchestrate multiple agents with various expertise to collaborate on bigger projects? How can we ensure safety of agent-driven solutions? How can we effectively use human-in-the-loop while developing diverse adaptive agents? We invite everyone to participate in answering these questions and realizing the vision of using AI agents to automate critical IT tasks.
this section cite: []

Section: References
Ref_id:b0 Title: Finops sample data Year: ()
Ref_id:b1 Title:  Year: ()
Ref_id:b2 Title:  Year: ()
Ref_id:b3 Title:  Year: ()
Ref_id:b4 Title:  Year: ()
Ref_id:b5 Title:  Year: ()
Ref_id:b6 Title:  Year: ()
Ref_id:b7 Title: Recommending root-cause and mitigation steps for cloud incidents using large language models Year: (2023)
Ref_id:b8 Title: Recommending root-cause and mitigation steps for cloud incidents using large language models Year: (2023)
Ref_id:b9 Title: Taking the blame game out of data centers operations with netpoirot Year: (2016)
Ref_id:b10 Title: Traceweaver: Distributed request tracing for microservices without application modification Year: (2024)
Ref_id:b11 Title: Basic Concepts and Taxonomy of Dependable and Secure Computing Year: (2004-01)
Ref_id:b12 Title: A fault injection platform for learning aiops models Year: (2022)
Ref_id:b13 Title: The Datacenter as a Computer: Designing Warehouse-Scale Machines Year: (2018)
Ref_id:b14 Title: Site Reliability Workbook: Practical Ways to Implement SRE Year: (2018-08)
Ref_id:b15 Title: Super: Evaluating agents on setting up and executing tasks from research repositories Year: (2024)
Ref_id:b16 Title: Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks Year: (2024)
Ref_id:b17 Title: Rising cloud costs leave CIOs seeking ways to cope Year: ()
Ref_id:b18 Title: Finops kpis Year: (2025)
Ref_id:b19 Title: Scouts: Improving the diagnosis process through domain-customized incident routing Year: (2020)
Ref_id:b20 Title: How to fight production incidents?: an empirical study on a large-scale cloud service Year: (2022-11)
Ref_id:b21 Title: What bugs live in the cloud? a study of 3000+ issues in cloud systems Year: (2014-11)
Ref_id:b22 Title: Why Does the Cloud Stop Computing? Lessons from Hundreds of Service Outages Year: (2016-10)
Ref_id:b23 Title: Fail-Slow at Scale: Evidence of Hardware Performance Faults in Large Production Systems Year: (2018-02)
Ref_id:b24 Title: Pingmesh: A large-scale system for data center network latency measurement and analysis. SIG-COMM Comput Year: (2015-08)
Ref_id:b25 Title: Cores that don't count Year: (2021-06)
Ref_id:b26 Title: Large language models cannot selfcorrect reasoning yet Year: (2023)
Ref_id:b27 Title: Metastable Failures in the Wild Year: (2022-07)
Ref_id:b28 Title: Storm clouds ahead: Missed expectations in cloud computing Year: (2024-10-28)
Ref_id:b29 Title: Root cause analysis of failures in microservices through causal discovery Year: (2022)
Ref_id:b30 Title: Live forensics for hpc systems: A case study on distributed storage systems Year: (2020)
Ref_id:b31 Title: SWE-bench: Can language models resolve real-world github issues? Year: (2024)
Ref_id:b32 Title: Idc predicts 80% of cios to leverage ai and automation for business agility and insights by Year: (2024)
Ref_id:b33 Title: Cloud Programming Simplified: A Berkeley View on Serverless Computing Year: (2019-02)
Ref_id:b34 Title: Ai agents that matter Year: (2024)
Ref_id:b35 Title: The treatment of ties in ranking problems Year: (1945)
Ref_id:b36 Title: What Takes Us Down? USENIX ;login Year: (2012-10)
Ref_id:b37 Title: Crowdstrike outage explained: What caused it and what's next Year: (2024)
Ref_id:b38 Title: Visualwebarena: Evaluating multimodal agents on realistic visual web tasks Year: (2024)
Ref_id:b39 Title: Detecting failures in distributed systems with the falcon spy network Year: (2011)
Ref_id:b40 Title: What bugs cause production cloud incidents? Year: (2019-05)
Ref_id:b41 Title: Cost optimization for cloud storage from user perspectives: Recent advances, taxonomy, and survey Year: (2023)
Ref_id:b42 Title: The elephant in the room: Towards a reliable time-series anomaly detection benchmark Year: ()
Ref_id:b43 Title: The elephant in the room: Towards a reliable time-series anomaly detection benchmark Year: (2024)
Ref_id:b44 Title: Ticket-bert: Labeling incident management tickets with language models Year: (2023)
Ref_id:b45 Title: Node failure localization via network tomography Year: (2014)
Ref_id:b46 Title: Fail at Scale: Reliability in the Face of Rapid Change Year: (2015-11)
Ref_id:b47 Title: Elastic Cloud Services: Scaling Snowflake's Control Plane Year: (2022-11)
Ref_id:b48 Title: Dowhy: A python library for causal inference Year: ()
Ref_id:b49 Title:  Year: ()
Ref_id:b50 Title: Site Reliability Engineering: Monitoring Distributed Systems. O'Reilly Media Year: (2024)
Ref_id:b51 Title:  Year: (2020)
Ref_id:b52 Title: Inventory theory applied to cost optimization in cloud computing Year: (2016)
Ref_id:b53 Title: Open security controls assessment language Year: ()
Ref_id:b54 Title: Oscal-compass cncf sandbox project Year: (2024)
Ref_id:b55 Title: Resource usage cost optimization in cloud computing using machine learning Year: (2020)
Ref_id:b56 Title: Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies Year: (2023)
Ref_id:b57 Title: Towards Natural-Language Understanding and Automated Enforcement of Privacy Rules and Regulations in the Cloud: Survey and Bibliography Year: (2011)
Ref_id:b58 Title: Parliament and the Council of the European Union. Digital operational resilience act for the financial sector and amending regulations Year: (2024)
Ref_id:b59 Title: Recovery-Oriented Computing (ROC): Motivation, Definition, Techniques, and Case Studies Year: (2002-03)
Ref_id:b60 Title: Root cause analysis for microservice system based on causal inference: How far are we? Year: (2024)
Ref_id:b61 Title: Automated code generation for information technology tasks in yaml through large language models Year: (2023)
Ref_id:b62 Title: Intelligent container reallocation at microsoft 365 Year: (2021)
Ref_id:b63 Title: Autoscaling solutions for cloud applications under dynamic workloads Year: (2024)
Ref_id:b64 Title: Exploring llm-based agents for root cause analysis Year: (2024)
Ref_id:b65 Title: Evaluating business analytics agents through multi-step insight generation Year: (2024)
Ref_id:b66 Title: Pyrca: A python machine learning library for root cause analysis Year: ()
Ref_id:b67 Title: Reflexion: Language agents with verbal reinforcement learning Year: (2023)
Ref_id:b68 Title: Dapper, a large-scale distributed systems tracing infrastructure Year: (2010)
Ref_id:b69 Title:  Year: (2023)
Ref_id:b70 Title: Testing Configuration Changes in Context to Prevent Production Failures Year: (2020-11)
Ref_id:b71 Title: {NetBouncer}: Active device and link failure localization in data center networks Year: (2019)
Ref_id:b72 Title: Holistic Configuration Management at Facebook Year: (2015-10)
Ref_id:b73 Title: Twine: A Unified Cluster Management System for Shared Infrastructure Year: (2020-11)
Ref_id:b74 Title: Fail through the Cracks: Cross-System Interaction Failures in Modern Cloud Systems Year: (2023-05)
Ref_id:b75 Title: Optimisation methods for ranking functions with multiple parameters Year: (2006)
Ref_id:b76 Title: State of FinOps: 2025 report Year: (2025)
Ref_id:b77 Title: AI-Assisted Controls Change Management for Cybersecurity in the Cloud Year: ()
Ref_id:b78 Title: Pretrained AI models: Performativity, mobility, and change Year: (2019-09)
Ref_id:b79 Title: Maelstrom: Mitigating Datacenter-level Disasters by Draining Interdependent Traffic Safely and Efficiently Year: (2018-10)
Ref_id:b80 Title: Fault injection based interventional causal learning for distributed applications Year: (2023)
Ref_id:b81 Title: Travelplanner: A benchmark for real-world planning with language agents Year: (2024)
Ref_id:b82 Title: Cloud atlas: Efficient fault localization for cloud systems using language models and causal insight Year: (2024)
Ref_id:b83 Title: Rewoo: Decoupling reasoning from observations for efficient augmented language models Year: (2023)
Ref_id:b84 Title: Do Not Blame Users for Misconfigurations Year: (2013-11)
Ref_id:b85 Title: Swe-agent: Agentcomputer interfaces enable automated software engineering Year: (2024)
Ref_id:b86 Title: Optimizing it finops and sustainability through unsupervised workload characterization Year: (2024)
Ref_id:b87 Title: React: Synergizing reasoning and acting in language models Year: (2023)
Ref_id:b88 Title: Chain-of-event: Interpretable root cause analysis for microservices through automatically learning weighted event causal graph Year: (2024)
Ref_id:b89 Title: Cco -cloud cost optimizer Year: (2023)
Ref_id:b90 Title: Text Data Management and Analysis: A Practical Introduction to Information Retrieval and Text Mining Year: (2016)
Ref_id:b91 Title: Flash: A workflow automation agent for diagnosing recurring incidents Year: (2024-10)
Ref_id:b92 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b93 Title: Fault analysis and debugging of microservice systems: Industrial survey, benchmark system, and empirical study Year: (2018)
Ref_id:b94 Title: Hemirca: Fine-grained root cause analysis for microservices with heterogeneous data sources Year: (2024)
