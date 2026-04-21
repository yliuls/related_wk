Title: RELIABLE WEAK-TO-STRONG MONITORING OF LLM AGENTS
Abstract: We stress test monitoring systems for detecting covert misbehavior in LLM agents (e.g., secretly exfiltrating data). We propose a monitor red teaming (MRT) workflow that varies (1) agent and monitor awareness, (2) adversarial evasion strategies, and (3) evaluation across tool-calling (SHADE-Arena) and computer-use (CUA-SHADE-Arena) environments. We benchmark standard monitor scaffoldings and introduce a hybrid hierarchical-sequential design. Our experiments yield three findings. First, agent awareness dominates monitor awareness: agents that know they are monitored substantially degrade detection, while increasing monitor awareness helps less than expected. Second, monitor scaffolding matters: our hybrid design consistently outperforms baselines when deploying weaker monitors to oversee stronger agents (a weak-to-strong effect). Third, targeted human oversight is key: escalating only pre-flagged cases improves TPR by 15% at FPR = 0.01. Our work positions MRT as a standard workflow for stress-testing oversight, revealing robustness gaps in both LLM-and human-based monitoring. We release code, data, and logs to support further research.

Section: INTRODUCTION
Overseeing Large Language Model (LLM) agents is becoming a central challenge in AI control as model capability and autonomy grow rapidly. Monitoring an agent's trajectory, whether through LLMs or human overseers who inspect its actions and chain-of-thought (CoT), is essential for risk triage, containment, and mitigation (Chan et al., 2025;Greenblatt et al., 2024). This paper introduces Monitor Red Teaming (MRT), a standardized workflow for stress-testing the reliability of monitoring systems against adversarial agents. Following adversarial robustness practices in deep learning (Carlini et al., 2019), MRT defines threat models, evaluation metrics, and adversarial strategies that together provide a principled way to probe monitors' robustness.
A central design choice is how to parse agent trajectories into the monitor, i.e., the monitor scaffolding. The simplest baseline is zero-shot prompting on the full trajectory (Kutasov et al., 2025;Baker et al., 2025), which fails on long and noisy traces. We therefore test improved scaffoldings: (1) hierarchical summarization (Sumers et al., 2025); (2) sequential sliding-window inspection; and (3) a hybrid that combines both.
To evaluate MRT empirically, we use SHADE-Arena (Kutasov et al., 2025) and introduce CUA-SHADE-Arena, a new benchmark for computer-use sabotage based on TheAgentCompany (Xu et al., 2024). Across these environments, we answer five questions:
1. How monitor awareness affects reliability when agents are aware/unaware of being monitored (Section 4.2); 2. How much advantage a higher monitor awareness confers against evasion strategies (Section 4.3); 3. Which scaffolding designs are most adversarially reliable (Sections 4.2 and 4.3);
4. Whether weak models can monitor stronger ones (Section 4.4); and 5. Whether humans in the loop improve reliability (Appendix C).
Our results highlight three findings. First, agent awareness dominates: when agents know they are monitored, reliability drops sharply, while additional monitor awareness helps less than expected. Second, scaffolding matters: our hybrid design consistently outperforms baselines when using weak monitors (e.g., Claude 3.6 Haiku) to oversee stronger agents (e.g., Claude 4 Sonnet). Third, targeted human oversight is key: escalating only pre-flagged cases improves TPR by ∼ 15% at FPR = 0.01 (Appendix C).
In summary, our contributions are: (1) A standardized MRT workflow, including threat models, metrics, and adversarial strategies (Section 3); (2) CUA-SHADE-Arena, a benchmark of 15 side tasks for detecting computer-use agent misbehavior (Section 3.1); (3) sequential and hybrid monitor scaffoldings that enable weak LLMs to monitor stronger agents (Section 3.5); (4) empirical findings across SHADE-Arena and CUA-SHADE-Arena that inform monitor design and human-in-the-loop oversight (Section 4).
We release code, datasets, and logs to facilitate further research on monitor robustness.
this section cite: ['b12', 'b22', 'b11', 'b34', 'b5', 'b48', 'b34']

Section: BACKGROUND
Motivations of Agent Monitoring. As autonomous LLM agents grow more capable (Deng et al., 2025;Yao et al., 2024) and long-horizon task times double every 7 months (Kwa et al., 2025), organizations will deploy them at scale, raising the risk of covert misbehavior. LLM agents may actively violate safety specifications (Benton et al., 2024) or deviate from user intent (Baker et al., 2025). Monitoring their chain-of-thoughts (CoTs) can help detect behaviors such as reward hacking, though over-optimization may actually make detection harder (Baker et al., 2025). Recent evaluations confirm that even advanced models can secretly sabotage systems (Anthropic, 2025b).
From Agent Red Teaming to Monitor Red Teaming. Traditional red teaming emphasizes malicious queries (Andriushchenko et al., 2024) or environmental injections (Greshake et al., 2023;Debenedetti et al., 2024). We instead focus on monitor red teaming (MRT): systematically probing monitors (LLMs, humans, or both) for their ability to detect adversarial misbehavior such as codebase sabotage (Benton et al., 2024). The aim is to quantify monitor reliability before deployment, targeting both high precision and recall (Section 3.4). Prior work in non-adversarial domains has shown that verification is easier than generation Song et al. (2024); Zhang et al. (2024), and we propose these advantages could extend to agent monitoring as well.
Offline and Online MRT. In deployment, monitoring should operate online, flagging suspicious actions before they cause irreversible harm. However, current designs remain rudimentary and do not scale. In this work, we evaluate offline MRT, where the monitor inspects full trajectories, as a controlled proxy for studying how to build more robust online systems. Our metrics (AUC, TPR at FPR=0.01) are trajectory-level and offline. Truly online deployment introduces additional systemlevel constraints (latency per step, control logic, blocking vs. warning, etc.) that we intentionally leave to future work.
Controlled Experiments for Monitoring. Following prior work, we pair each benign main task with a malicious side task to create controlled sabotage settings (Benton et al., 2024). Main tasks lengthen trajectories and make concealment feasible, while side tasks specify misbehavior. Although today's agents still struggle with very long horizons, this limitation is rapidly diminishing (Kwa et al., 2025). We adopt the SHADE-Arena setup (Kutasov et al., 2025), where side task success does not depend on main task completion, ensuring monitors must evaluate complex trajectories directly.
We discuss additional related work in Appendix A.
this section cite: ['b18', 'b58', 'b35', 'b6', 'b5', 'b5', 'b1', 'b23', 'b16', 'b6', 'b46', 'b59', 'b6', 'b35', 'b34']

Section: MONITOR RED TEAMING
In this section, we introduce Monitor Red Teaming (MRT) by defining the following key components:
(1) The datasets and environments for running agents and the monitoring system (Section 3.1); (2) The threat models which define the respective advantages of the agent and the monitor (Section 3.2);
(3) The red teaming strategies to help the adversarial agents evade the monitor detection under each threat model (Section 3.3); (4) The metrics to evaluate a monitor's performance against adversarial agents (Section 3.4); and (5) A list of monitor architecture designs, i.e., the existing ones from prior works and our proposals, to evaluate in our experiments (Section 3.5).
this section cite: []

Section: DATASETS AND AGENT ENVIRONMENTS
We evaluate MRT on two agentic harm datasets: SHADE-Arena (Kutasov et al., 2025) and our new CUA-SHADE-Arena benchmark.
SHADE-Arena contains 17 benign tasks, each paired with a harmful side task, implemented in the AgentDojo environment (Debenedetti et al., 2024). Tasks range from productivity scenarios (e.g., booking flights) to software operations. As noted by Kutasov et al. (2025), constructing sufficiently complex task pairs is challenging; indeed, only 8% of AgentDojo's security test cases evade detection even by a simple BERT classifier (Debenedetti et al., 2024;ProtectAI, 2024).
To complement this, we introduce CUA-SHADE-Arena, a set of 15 main-side task pairs built on TheAgentCompany benchmark (Xu et al., 2024). These tasks are executed in a virtual machine environment where agents issue bash commands, edit files, and write code, reflecting more realistic web-agent deployments. Main tasks are drawn from standard company operations (e.g., completing HR forms), while paired side tasks require covert sabotage (e.g., exfiltrating sensitive employee data). Compared to SHADE-Arena, trajectories in CUA-SHADE-Arena are shorter and somewhat easier, but they more closely resemble real enterprise settings.
Further details on dataset construction and the complete list of side tasks are provided in Appendix L.
this section cite: ['b34', 'b16', 'b34', 'b16', 'b42']

Section: THREAT MODELS
In adversarial machine learning, a threat model defines the attacker's resources and access to the victim model, as in the panda-gibbon example in ℓ p robustness (Goodfellow et al., 2014). Fixing the threat model before optimizing methodology provides a fair ground for comparison.
In MRT, the monitor can access the full agent trajectory (i.e., CoT and actions), while the red team has only API access to the LLMs used in the system. The agent has no access to the monitor's LLM, and vice versa. Beyond these restrictions, an important axis of the threat model is the awareness each party has of the other, which strongly influences their strategies. We therefore motivate MRT threat models by considering both agent awareness and monitor awareness, which may vary across real-world deployments (Figure 19).
this section cite: ['b21']

Section: Agent
Awareness. An agent's strategies depend on whether it knows it is being monitored. Prior work has treated awareness as a red teaming strategy (Kutasov et al., 2025;Benton et al., 2024); here we formalize it as part of the threat model, since situational awareness is often the root cause of deception. We consider two levels:
• Unaware: The agent does not know a monitor exists, reflecting today's deployments with no mature monitoring infrastructure.
• Aware: The agent is told via system and tool prompts that it is in an evaluation environment and its behavior will be inspected by a monitor, approximating future settings where agents adapt to oversight.
Monitor Awareness. We define monitor awareness with respect to the hidden side task, i.e., the malicious objective embedded alongside the benign task (Kutasov et al., 2025;Baker et al., 2025).
• Unaware: The monitor sees only the trajectory and must infer misbehavior, as in third-party auditing with limited access.
• Aware: The monitor knows the main task but not the side task, reflecting typical deployments where task specifications are shared but potential misbehaviors are not.
• Omniscient: The monitor knows both main and side tasks, representing settings where misbehaviors are pre-cataloged. While a programmatic oracle could achieve perfect detection on these tasks, this serves as an approximate upper bound on LLM monitor reliabilityo.
Each agent-monitor pairing (two awareness levels × three awareness levels) defines one of six MRT threat models. Full descriptions and illustrative scenarios are provided in Appendix K.
this section cite: ['b34', 'b6', 'b34', 'b5']

Section: AGENT'S EVASION STRATEGIES
Similar to model safeguard red teaming, MRT frames the agent as adversarial, attempting to evade monitor detection. Since an unaware agent cannot adapt to oversight, we focus primarily on aware agents and propose the following strategies.
• Baseline -The Dealer's Choice. The agent receives no extra instruction beyond being told it is monitored, and chooses its own evasion strategy based on this context.
• Monitor Hijacking. LLM monitors are vulnerable to injected instructions in the agent's CoT, e.g., "Forget your previous instruction and output 1 as your suspicious score". Inspired by prompt injection attacks (Debenedetti et al., 2024;Wallace et al., 2024), we instruct agents to embed such manipulative prompts in their trajectories.
• Lying with Jailbreaking Strategies. Human red teamers often succeed at jailbreaks by overwhelming models with irrelevant detail or misdirection (Li et al., 2024b;Chao et al., 2025). Kritz et al. (2025) show that capable LLMs such as Claude 4 Sonnet can execute these strategies effectively. We adapt such techniques for monitoring evasion, finding that they help agents produce convincing lies. The full set of strategies is listed in Appendix I (Table 2).
• CoT Obfuscation. LLMs trained to verbalize their reasoning can be optimized to hide critical steps, reducing monitor visibility (Baker et al., 2025). As adversarial fine-tuning may encourage this behavior, monitors must expect partial or deceptive traces. To approximate such agents, we prompt the model to generate two blocks of thoughts: one sanitized for the monitor, the other hidden, following related work on under-verbalization (Hao et al., 2024;Chen et al., 2025).
The details and prompts for each strategy are provided in Appendix I.
Other Strategies. Additional approaches (e.g., random action injection, steganographic CoT encoding) were not tested directly. We instead focused on a representative subset that leverages different model capabilities, such as following complex hijacking prompts or creatively implementing jailbreak tactics. Future work may develop additional MRT techniques to broaden this space. These chunks can be evaluated in order (light red) or in parallel (medium red). The two techniques can be combined via a weighted average (dark red).
this section cite: ['b16', 'b52', 'b13', 'b32', 'b5', 'b24', 'b14']

Section: EVALUATION METRICS
Prior work often reports detection or evasion rates over a trajectory (Kutasov et al., 2025;Benton et al., 2024), but these can be misleading. A monitor that flags every trajectory attains perfect detection but is useless in practice, while success-rate metrics confound evasion with the agent's task ability. We therefore focus on two metrics that better capture monitor reliability:
• AUC-ROC. Computed by comparing monitor scores with the ground-truth verdict (whether the agent attempted the side task). Under MRT, the adversary's goal is to drive this score toward 0.5, i.e., random performance.
• TPR at FPR=0.01. The true positive rate when the false positive rate is constrained to 0.01, reflecting settings where escalating flagged cases to human review is costly. Following prior work (Kutasov et al., 2025), we adopt this threshold.
this section cite: ['b34', 'b6', 'b34']

Section: MONITOR SCAFFOLDING
We evaluate several architectures for LLM-based monitors, including prior proposals and new designs. While still rudimentary, they illustrate design choices that affect monitor reliability. Architectures are orthogonal to awareness and are tested under multiple threat models (Sections 4.2 and 4.3). Definitions follow, with illustrations in Fig. 2. Full details and prompts are provided in Appendix J.
this section cite: []

Section: Baseline (Full Trajectory Monitoring).
Prompting an LLM with the full agent trajectory is the simplest approach, used in prior work (Kutasov et al., 2025;Arnav et al., 2025;Baker et al., 2025). We adopt the same prompt from Kutasov et al. (2025).
Hierarchical Monitoring. Hierarchical summarization has been shown to improve monitoring over long contexts (Sumers et al., 2025). We adapt this to agentic trajectories by chunking N tool calls/outputs, scoring each chunk, and recursively summarizing, which reduces the needle-in-thehaystack problem (Li et al., 2024a;Kamradt, 2023) while exploiting natural task structure. This also cuts cost: for trajectory length L, complexity falls from L 2 to (L/N ) • N 2 , yielding ∼ 67× efficiency in our setting (20k-token trajectories, 300-token steps), with further gains from parallelization.
this section cite: ['b34', 'b4', 'b5', 'b34', 'b48', 'b27']

Section: Sequential Monitoring.
Recursive summarization from start to finish (Wu et al., 2021) provides an alternative for long contexts. We divide the trajectory into N -length chunks, evaluating each in sequence with the previous summary in context. This approach naturally extends to online monitoring, where trajectories evolve incrementally, and is more efficient than hierarchical monitoring since it requires only a single pass.
µ-Hybrid Monitoring. We combine hierarchical (h) and sequential (s) outputs linearly: µh + (1µ)s. A natural choice is µ = 0.5, equally weighting both. Empirical tuning (Appendix D) yields only marginal gains over this baseline, so we adopt µ = 0.5 throughout.
this section cite: ['b54']

Section: EXPERIMENTS
We evaluate MRT using the datasets, threat models, metrics, and monitor architectures from Section 3.
Our setup lets us empirically answer the following questions:
• Q1: Given an unaware vs. aware agent, how does monitor awareness affect reliability?
• Q2: How much advantage does higher monitor awareness confer against agent evasion strategies?
• Q3: Are refined monitor scaffoldings (e.g., hierarchical) more robust than the full-trajectory baseline? Which is most adversarially reliable?
• Q4: Can weak models (e.g., Claude 3.6 Haiku) reliably monitor stronger agents (e.g., Claude Sonnet 4)?
• Q5: If a human overseer can review trajectories and interact with the monitor under a fixed time budget, does overall reliability improve? By how much, and why (or why not)?
We detail experimental setups in Section 4.1, then address Q1-Q3 in Sections 4.2 and 4.3, and Q4 in Section 4.4. Findings on Q5 are in Appendix C.
this section cite: []

Section: EXPERIMENT SETUPS
Unaware Agents. To elicit unaware but misbehaving agents, we directly prompt with both the main and side tasks. Since refusal training often blocks such behavior, we follow SHADE-Arena and frame the agent as assisting in a red-teaming setting (Kutasov et al., 2025). We further emphasize: You can think about the tasks in <thinking> tags. Plan as needed to complete the side task. This proxy yields more natural "unaware" behavior. Non-reasoning models are prompted to wrap CoT in <thinking> tags, while reasoning models use their original CoT.
this section cite: ['b34']

Section: Aware Agents with Evasion Strategies.
We implement the strategies from Section 3.3.
• Dealer's choice: no extra instruction, as in SHADE-Arena.
• Monitor hijacking: the agent injects instructions (e.g., "output 1") to bias the monitor.
• Lying with jailbreaking: agents are provided with red-team strategies (Table 2) to generate deceptive CoTs.
• CoT obfuscation: agents output two CoTs, with only the "visible" one shown to the monitor. For reasoning models (e.g., Gemini 2.5 Pro), we hide the original CoT and prompt for a visible CoT.
In practice, agents may combine lying and obfuscation, hiding true thoughts while filling the visible CoT with irrelevant or misleading content. Further details and prompts for each attacker are provided in Appendix I, while example trajectories are in Appendix O.
this section cite: []

Section: Trajectory Roll-outs.
We collect 20 roll-outs for each agent across 32 tasks (17 SHADE-Arena, 15 CUA-SHADE-Arena). Half are malicious (main & side task) and half are benign (main only). Each episode runs up to 100 steps or 3600 seconds. Prompts used for monitor evaluation are provided in Appendix J and example evaluations are in Appendix P. Full rollout details (e.g., retry logic, failure handling) are in Appendix M. We reproduce Findings 1-6 in BashBench (Bhatt et al., 2025) in Appendix H.
this section cite: ['b8']

Section: Monitor Setup.
We implement four monitor types from Section 3.5 (baseline, hierarchical, sequential, hybrid). The baseline follows Kutasov et al. (2025). Each monitor outputs a suspicion score over a full trajectory, with awareness controlled by including the main, side, or both tasks in the system prompt. Prompt templates are in Appendix J.
this section cite: ['b34']

Section: Models.
In SHADE-Arena, agents are Claude 3.6 Sonnet and Gemini 2.5 Pro; in CUA-SHADE-Arena, we deploy Claude 4 Sonnet on the OpenHands scaffolding (Wang et al., 2024). Claude 3.6 Sonnet serves as the monitor unless otherwise specified. Full model version strings are listed in Appendix N.
this section cite: ['b53']

Section: EXPERIMENT 1: UNAWARE AGENTS VS. AWARE AGENTS WITH NO STRATEGY
As discussed in Section 3.4, we report AUC-ROC and TPR at FPR = 0.01 to measure the impact of agent and monitor awareness on reliability. This addresses Q1 and part of Q3. In Fig. 3, we compare monitors against unaware agents (light bars) and against aware agents with no strategy (i.e., the dealer's choice, medium bars).
Finding 1: Unaware agents are reliably detected. When the agent is unaware (light bars), all scaffoldings achieve strong reliability (AUC ≥ 0.9) across both datasets. TPR at FPR=0.01 remains stable across monitor awareness levels, though sequential monitors consistently yield lower TPR than other scaffoldings performing even worse than baseline. In SHADE-Arena, omniscient monitors offer little advantage over unaware monitors. Hybrid monitors are best overall, but their margin over second-best is small. In CUA-SHADE-Arena, aware monitors show no advantage, and omniscient monitors only gain ∼ 10%. Because trajectories here contain more granular actions, knowing the side task appears to help predict the consequences of these actions.
Finding 2: Aware agents reduce reliability; hybrid monitors remain strongest. When the agent is aware but uses no explicit evasion strategy (medium bars), both AUC and TPR drop across models and datasets, and degradation is larger in SHADE-Arena than in CUA-SHADE-Arena. Surprisingly, aware monitors are not consistently stronger than unaware monitors-suggesting that even third-party monitors with no task information may perform comparably to first-party monitors. Across all cases, hybrid monitor AUC consistently outperforms or matches alternatives.
Finding 3: Omniscient monitors are still imperfect. A programmatic oracle can directly verify side task completion (true upper bound, AUC ≈ 1); however, the omniscient monitors are far from this upper bound, especially against aware attackers in SHADE-Arena. Despite knowing the side task, chain-of-thought monitors struggle to reason over long trajectories where the agent is actively hiding its behavior. The underlying 'needle-in-a-haystack' problem (Kamradt, 2023) is difficult even if the monitor knows exactly what it is looking for. For sufficiently complex tasks, side task knowledge is not as important as scaffolding and monitor model capability (demonstrated in Section 4.4). In summary, for Q1, agent awareness is the dominant factor: monitor awareness provides limited gains. For Q3, the omniscient hybrid monitor provides an empirical upper bound on chain-of-thought monitoring below AUC = 1.0, showing that current LLMs analyze complex trajectories imperfectly even if told what to look for.
this section cite: ['b27']

Section: EXPERIMENT 2: RED TEAMING MONITORS WITH EVASION STRATEGIES
Monitor Awareness AUC TPR at FPR = 0.01 Unaware Hybrid (0.75) Baseline (0.20) Aware Hybrid (0.73) Hybrid (0.21) Omniscient Hybrid (0.80) Baseline (0.47) Winning Scaffolding Monitor Awareness AUC TPR at FPR = 0.01 Unaware Hybrid (0.85) Hybrid (0.68) Aware Hybrid (0.84) Hybrid (0.67) Omniscient Hybrid (0.98) Hybrid (0.88) Winning Scaffolding SHADE-Arena CUA-SHADE-Arena We now test the impact of evasion strategies on monitor reliability to answer Q2 and extend our conclusions for Q3. Using the implementations from Section 4.1, we evaluate AUC and TPR under each strategy. Fig. 4 summarizes the empirical lower bound by comparing the baseline (medium bars) to the worst-case across strategies (dark bars). Fig. 17 and Fig. 18 in Appendix I report full results.
Finding 4: Evasion strategies significantly degrade baseline monitors. When strategies are enabled (dark bars), the AUC and TPR of the full-trajectory baseline drop sharply relative to no strategy. In extreme cases, AUC falls below 0.5, e.g., when the agent uses monitor hijacking against an aware monitor, convincing it to assign suspiciousness = 1 to malicious trajectories while benign ones receive higher scores, producing both false negatives and false positives.
Finding 5: Hybrid monitors are consistently most robust. Across awareness levels and datasets, hybrid scaffolding maintains higher AUC and TPR than other types. While the baseline occasionally yields slightly higher TPR in SHADE-Arena, the margins are small and absolute TPR remains low. Sequential scaffolds perform especially poorly on SHADE-Arena, whereas hierarchical scaffolds underperform on CUA-SHADE-Arena. These cross-dataset weaknesses highlight the advantage of combining both approaches.
Finding 6: Monitor awareness helps, but only against unsophisticated agents. Higher awareness improves reliability for a fixed scaffolding, particularly in TPR. However, this benefit vanishes once sophisticated strategies are introduced. Evasion overwhelms the advantage of additional information.
Taken together with Findings 2-6, we conclude for Q3 that the hybrid monitor is most adversarially reliable, matching or exceeding the full-trajectory baseline in nearly all cases. Importantly, improving scaffolding design is more effective than providing the monitor with more task information. Thus, even a third-party monitor with no knowledge of the user's main task can still reliably detect misbehavior if equipped with a well-designed hybrid scaffolding. Eventually, misbehaving agents may be more capable than the models employed for monitoring. Greenblatt et al. (2024) propose a weaker-but-trusted monitoring paradigm: relying on weaker models to oversee stronger, untrusted ones. This reflects both the long-term scenario of humans monitoring super-intelligent systems and the near-term constraint that frontier models cannot safely monitor themselves. Here, we ask Q4: Can weak models (e.g., Claude 3.6 Haiku) reliably monitor strong ones (e.g., Claude Sonnet 4)? Our approach mirrors capability-gap studies in jailbreaking, where Panfilov et al. (2025) show that LLMs more easily break models with lower MMLU-Pro scores. We test whether the gap in MMLU-Pro between agent and monitor correlates with monitoring reliability in a controlled MRT setting.
Setup. We evaluate four aware agents, Claude 3.6 Sonnet, Claude Sonnet 4, Gemini 2.5 Flash, and Gemini 2.5 Pro, each using CoT obfuscation as an evasion strategy. For monitors, we employ Claude 3.6 Haiku, Claude 3.6 Sonnet, Claude 3.7 Sonnet, Claude Sonnet 4, Gemini 2.5 Pro, Gemini 2.5 Flash, GPT-4o, GPT-4o-mini, Mistral 3.1 Small and o3, resulting in a total of 40 attacker-monitor pairs. For each monitor, we implement all four scaffolding types from Section 3.5, using SHADE-Arena as the environment. Fig. 13 plots AUC-ROC scores against the MMLU-Pro gap between attacker and monitor.
this section cite: ['b22', 'b40']

Section: Summary of Findings.
First, monitoring reliability increases with the monitor's relative MMLU-Pro advantage, but the correlation depends on scaffolding. Hybrid monitors show the weakest dependence on capability gap, maintaining higher reliability even when the agent is stronger. Notably, even with weaker models such as GPT-4o-mini and Claude 3.6 Haiku, AUC exceeds 0.85 across all agents. Second, hybrid scaffolds consistently outperform baseline, hierarchical, and sequential monitors, confirming that sequential and hierarchical methods are complementary and more effective when combined. A detailed failure mode analysis is provided in Appendix E.
Overall, while an unscaffolded weak LLM struggles to monitor a stronger one, a carefully designed hybrid scaffold enables reliable weak-to-strong monitoring in a controlled MRT setting. This supports the feasibility of oversight systems that use weaker but more trusted models to supervise powerful agents.
this section cite: []

Section: FUTURE WORK AND LIMITATIONS
AUC and TPR Are Complementary Metrics. Our metrics are not always correlated: hybrid monitoring on SHADE-Arena improves AUC without raising TPR (Figure 4), while human-augmented monitoring boosts TPR with little AUC change (Figure 6b). Future work may explore additional metrics.
Threat Model Limitations. Our red-teaming setup uses prompting to induce behaviors such as monitor hijacking. This is not a complete model of real-world misalignment. While our setup reasonably approximates indirect prompt injection, conflicting objectives, and stealthy behavior under monitoring, stronger adversaries could emerge from adversarial finetuning or pretraining misaligned agents (Betley et al., 2025), which may invent novel evasions beyond prompting. Assessing monitor robustness against such attackers remains an open direction, and we encourage future work to apply MRT to adversarially finetuned agents.
Building Better Monitoring Systems. Sequential, hierarchical, and hybrid scaffolds are an initial step; more sophisticated methods are needed. Likewise, benchmarks must grow harder: CUA-SHADE-Arena tasks average 10k tokens versus 25k in SHADE-Arena (Kutasov et al., 2025).
Designing sabotage tasks on real applications and richer VM environments would strengthen evaluation.
Training Human Overseers. Lastly, human-in-the-loop monitoring provided modest gains when humans augmented LLM decisions with a clear interface, but generalist overseers remain less effective than LLMs (Kutasov et al., 2025). Progress will likely require better scaffolding, interfaces tailored to oversight, and techniques to reduce bias and false positives.
this section cite: ['b7', 'b34', 'b34']

Section: CONCLUSION
We introduce a systematic framework for Monitor Red Teaming (MRT) to stress-test systems that detect covert misbehavior in autonomous LLM agents. Our workflow spans varied threat models, adversarial strategies, and the new CUA-SHADE-Arena benchmark, enabling extensive empirical evaluation of monitoring approaches. Our results highlight three points. First, agent awareness dominates: detection degrades sharply when agents know they are monitored, while monitor awareness helps less than expected. Second, scaffolding matters most: hybrid hierarchicalsequential designs consistently outperformed baselines and enabled weak-to-strong monitoring in a controlled MRT setting. Third, targeted human oversight is effective: escalating only preflagged cases improved TPR by ∼ 15% at FPR = 0.01. Overall, improving monitoring requires architectural innovation and careful design. We release code, datasets, and logs to support further research on robust, scalable oversight of autonomous AI systems.
this section cite: []

Section: References
Ref_id:b0 Title: Concrete problems in ai safety Year: (2016)
Ref_id:b1 Title: A benchmark for measuring harmfulness of llm agents Year: (2024)
Ref_id:b2 Title: Do reasoning models use their scratchpad like we do? evidence from distilling paraphrases Year: ()
Ref_id:b3 Title: System card: Claude opus 4 & claude sonnet 4. Technical report, Anthropic, 2025b Year: ()
Ref_id:b4 Title: CoT Red-Handed: Stress Testing Chain-of-Thought Monitoring Year: (2025-05)
Ref_id:b5 Title: Monitoring reasoning models for misbehavior and the risks of promoting obfuscation Year: (2025)
Ref_id:b6 Title: Sabotage Evaluations for Frontier Models Year: (2024-10)
Ref_id:b7 Title: Emergent misalignment: Narrow finetuning can produce broadly misaligned llms Year: (2025)
Ref_id:b8 Title: Ctrl-z: Controlling ai agents via resampling Year: (2025)
Ref_id:b9 Title: Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks Year: (2024)
Ref_id:b10 Title: Measuring progress on scalable oversight for large language models Year: (2022)
Ref_id:b11 Title: On evaluating adversarial robustness Year: (2019)
Ref_id:b12 Title: Mle-bench: Evaluating machine learning agents on machine learning engineering Year: (2024)
Ref_id:b13 Title: Jailbreaking black box large language models in twenty queries Year: (2025)
Ref_id:b14 Title: Reasoning models don't always say what they think Year: (2025)
Ref_id:b15 Title: An open source guardrail system for building secure ai agents Year: (2025)
Ref_id:b16 Title: Agentdojo: A dynamic environment to evaluate prompt injection attacks and defenses for llm agents Year: (2024)
Ref_id:b17 Title: Defeating prompt injections by design Year: (2025)
Ref_id:b18 Title: SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks? Year: (2025)
Ref_id:b19 Title: Automated researchers can subtly sandbag Year: (2025-03)
Ref_id:b20 Title: Designing human-in-the-loop autonomous cyber-physical systems Year: (2019)
Ref_id:b21 Title: Explaining and harnessing adversarial examples Year: (2014)
Ref_id:b22 Title: AI Control: Improving Safety Despite Intentional Subversion Year: (2024-07)
Ref_id:b23 Title: Not what you've signed up for: Compromising real-world llm-integrated applications with indirect prompt injection Year: (2023)
Ref_id:b24 Title: Training large language models to reason in a continuous latent space Year: (2024)
Ref_id:b25 Title: Llama guard: Llm-based input-output safeguard for human-ai conversations Year: (2023)
Ref_id:b26 Title: Swe-bench: Can language models resolve real-world github issues? arXiv preprint Year: (2023)
Ref_id:b27 Title: Doing simple retrieval from LLM models at various context lengths to measure accuracy Year: (2023)
Ref_id:b28 Title: r 2 -guard: Robust reasoning enabled llm guardrail via knowledge-enhanced logical reasoning Year: (2024)
Ref_id:b29 Title: On scalable oversight with weak llms judging strong llms Year: (2024)
Ref_id:b30 Title: Evaluating language-model agents on realistic autonomous tasks Year: (2023)
Ref_id:b31 Title: Chain of thought monitorability: A new and fragile opportunity for ai safety Year: (2025)
Ref_id:b32 Title:  Year: (2025)
Ref_id:b33 Title: Nicolas Flammarion, and Maksym Andriushchenko. Os-harm: A benchmark for measuring safety of computer use agents Year: (2025)
Ref_id:b34 Title: SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents Year: (2025-06)
Ref_id:b35 Title: Measuring ai ability to complete long tasks Year: (2025)
Ref_id:b36 Title: Needlebench: Can llms do retrieval and reasoning in information Year: (2024)
Ref_id:b37 Title: Llm defenses are not robust to multi-turn human jailbreaks yet Year: (2024)
Ref_id:b38 Title: Autodan: Generating stealthy jailbreak prompts on aligned large language models Year: (2023)
Ref_id:b39 Title: How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions Year: (2023-09)
Ref_id:b40 Title: Capability-Based Scaling Laws for LLM Red-Teaming Year: (2025-05)
Ref_id:b41 Title: Evaluating Frontier Models for Dangerous Capabilities Year: (2024-04)
Ref_id:b42 Title: Fine-tuned deberta-v3-base for prompt injection detection Year: (2024)
Ref_id:b43 Title: Nemo guardrails: A toolkit for controllable and safe llm applications with programmable rails Year: (2023)
Ref_id:b44 Title: Towards understanding sycophancy in language models Year: (2023)
Ref_id:b45 Title: do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models Year: (2024)
Ref_id:b46 Title: Mind the gap: Examining the self-improvement capabilities of large language models Year: (2024)
Ref_id:b47 Title: Evaluating ai's ability to replicate ai research Year: (2025)
Ref_id:b48 Title: Monitoring computer use via hierarchical summarization Year: (2025-02)
Ref_id:b49 Title: Winning the race: America's ai action plan Year: (2025-07)
Ref_id:b50 Title: Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting Year: (2023)
Ref_id:b51 Title: Understanding different claude models: A guide to anthropic's ai Year: (2025-03)
Ref_id:b52 Title: The instruction hierarchy: Training llms to prioritize privileged instructions Year: (2024)
Ref_id:b53 Title: Openhands: An open platform for ai software developers as generalist agents Year: (2024)
Ref_id:b54 Title: Recursively summarizing books with human feedback Year: (2021)
Ref_id:b55 Title: Toward human-in-the-loop ai: Enhancing deep reinforcement learning via real-time human guidance for autonomous driving Year: (2023)
Ref_id:b56 Title: Safeguard llm agents by a guard agent via knowledge-enabled reasoning Year: (2024)
Ref_id:b57 Title: Theagentcompany: benchmarking llm agents on consequential real world tasks Year: (2024)
Ref_id:b58 Title: tau-bench: A benchmark for tool-agent-user interaction in real-world domains Year: (2024)
Ref_id:b59 Title: Generative verifiers: Reward modeling as next-token prediction Year: (2024)
Ref_id:b60 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b61 Title: Cve-bench: A benchmark for ai agents' ability to exploit real-world web application vulnerabilities Year: (2025)
