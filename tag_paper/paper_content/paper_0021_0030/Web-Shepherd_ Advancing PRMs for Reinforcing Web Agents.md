Title: WEB-SHEPHERD: Advancing PRMs for Reinforcing Web Agents
Abstract: Web navigation is a unique domain that can automate many repetitive real-life tasks and is challenging as it requires long-horizon sequential decision making beyond typical multimodal large language model (MLLM) tasks. Yet, specialized reward models for web navigation that can be utilized during both training and test-time have been absent until now. Despite the importance of speed and costeffectiveness, prior works have utilized MLLMs as reward models, which poses significant constraints for real-world deployment. To address this, in this work, we propose the first process reward model (PRM) called WEB-SHEPHERD which could assess web navigation trajectories in a step-level. To achieve this, we first construct the WEBPRM COLLECTION, a large-scale dataset with 40K step-level preference pairs and annotated checklists spanning diverse domains and difficulty levels. Next, we also introduce the WEBREWARDBENCH, the first meta-evaluation benchmark for evaluating PRMs. In our experiments, we observe that our WEB-SHEPHERD achieves about 30 points better accuracy compared to using GPT-4o on WEBREWARDBENCH. Furthermore, when testing on WebArena-lite by using GPT-4o-mini as the policy and WEB-SHEPHERD as the verifier, we achieve 10.9 points better performance, in 10× less cost compared to using GPT-4o-mini as the verifier. Our model, dataset, and code are publicly available at LINK.

Section: GPT-4o
GPT-4o-mini Claude-3.7-Sonnet Qwen-2.5-VL-72B Gemini-2.
this section cite: []

Section: Introduction
Web browsers serve as a common interface for countless digital tasks, making automation in this space a natural focus for recent advances in intelligent agents. Recent advances in multimodal large language models (MLLMs) have enabled agents to handle basic web interactions, such as retrieve addresses from map services or navigating simple webpages [1,2]. However, current agents remain highly unreliable, often exhibiting brittle behaviors such as repeatedly entering the same query when encountering minor issues, eventually failing the task [3][4][5]. This unreliability primarily stems from the long-horizon nature of web navigation, requiring agents to reason across multiple steps and maintain goal-directed planning, which MLLMs often find challenging [6]. Hence, to create a better performing web agent, there is a need for better learning methods and inference-time algorithms.
One effective method that allowed large language models (LLMs) to perform well across various tasks is using a reward model to perform search at test-time (e.g., Best-of-n), or using it for Reinforcement Learning (e.g., RLHF). However, specially trained reward models have been under-explored in the web navigation domain. Prior works such as Pan et al. [7] and Koh et al. [8] do not train separate reward models, but instead employ MLLMs as evaluators in inference-time algorithms, which has fundamental problems. First, using the evaluation from prompted MLLMs becomes a significant constraint in web navigation where speed and cost are crucially important. For example, using only GPT-4o for tree search on WebArena (consisting of 812 queries) requires approximately $14,000, and running inference on one A100 takes 40 hours, which is a major obstacle to deploying MLLMs as web navigation agents in real-world scenarios. Additionally, throughout our experiments, we confirm that prompting MLLMs performs worse than trained reward models. In summary, considering speed, cost, and performance, specially designed reward models for web navigation are absolutely necessary.
To address these challenges, we present WEB-SHEPHERD, which is, to the best of our knowledge, the first reward model trained specifically for evaluating trajectories of web navigation. In particular, WEB-SHEPHERD is designed as a process reward model (PRM) rather than an outcome reward model (ORM), because unlike other domains, ORM cannot be integrated into test-time algorithms in web navigation. For example, in mathematics, an LLM can write multiple solutions and the ORM can choose one, but in web navigation, if an LLM makes eight attempts to book a plane ticket, the airplane ticket cannot be refunded, so decisions about which action to take must be made at the process level. Furthermore, even during training-time, PRM can provide more fine-grained reward signals, making it more reliable than ORM [9,10]. WEB-SHEPHERD employs a structured checklist that explicitly decomposes high-level user instructions into clear, interpretable subgoals. By referencing this checklist as evaluation criteria, WEB-SHEPHERD accurately assesses step-level progress, enabling precise and robust guidance throughout agent trajectories.
The key contribution of this paper is that we also provide a suite of training data and benchmark to test PRMs for web navigation. First, we release the WEBPRM COLLECTION, which contains human-crafted instructions that covers diverse tasks across multiple difficulty levels. The notable feature of the WEBPRM COLLECTION is that it contains 40K step-level annotations for which action an agent should take and that each instruction contains an annotated checklist-structured sequences of subgoals that enable WEB-SHEPHERD to make accurate judgments. Second, we release the WEBREWARDBENCH, the first meta-evaluation benchmark to assess PRMs in web navigation. The WEBREWARDBENCH allows practitioners to test newly proposed PRMs without running resourceintensive web navigation agents, enabling efficient testing of different design choices and conducting ablation experiments. WEB-SHEPHERD achieves 85.0% performance on the WEBREWARDBENCH (WebArena set), significantly outperforming GPT-4o-mini with prompting at 5.0%. Furthermore, when using WEB-SHEPHERD's reward as guidance in tree search on GPT-4o-mini policy, it achieves 34.55% success rate on WebArena-lite [1,11], confirming it is 10.9 points better in performance, and 10× more cost-effective than using GPT-4o-mini as the evaluator.
2 Related Work MLLM-based web agents. Multimodal large language models (MLLMs) have emerged as powerful foundation models for web agents due to their strong generalization capabilities and adaptability to diverse interface. Previous work has leveraged MLLMs to complete web tasks via carefully designed instructions, often augmented with external tools (e.g., grounding module or verification) [12][13][14][15] or workflow [4]. Moreover, other approaches have trained MLLM-based agents to imitate expert trajec-tories using next-token prediction objective [16][17][18]. While these models perform well in-distribution, they often fail to generalize to unseen environments. To overcome these challenges, recent research has increasingly focused on inference-time scaling [7,19] or reinforcement learning (RL) [20][21][22], which enables agents to improve decision-making through reward feedback.
Inference-time scaling for web agents. Inference-time scaling has emerged as a crucial approach for multi-turn interactions in web environment. Recent studies have explored techniques such as tree search [23,19], long chain-of-thought (CoT) [24,25], and incorporating verifiers or judges to enhance agent performance with natural language feedback [26,27]. For example, Pan et al. [27] use a prompting-based evaluator to assess whether a trajectory is successful; if not, they apply Reflexion [26] to retry based on the generated feedback. Extending this direction, other work [8,5] investigates an interesting direction that tries to search the optimal browsing path with a prompted value function and A*-like algorithm and world model.
Rewards for web navigation. Prior works rely on binary rewards (success or failure) [21,22] from rule-based evaluations that require human annotation and lacks scalability in dynamic web environments [1,3]. To address these, recent studies have explored leveraging LLMs via prompting [14,7] or training outcome reward models (ORMs) [20]. However, binary reward offers limited guidance for credit assignment, especially in long-horizon tasks. To enable more informative feedback, the reasoning literature has introduced process reward models (PRMs), which assign step-level reward [9,10].
Building on this idea, recent work has explored using LLMs to estimate state-action values by prompting [19,28,29]. Nevertheless, the reliability and efficiency of MLLMs as process-level reward models remain underexplored. In this work, we aim to develop a PRM for web agents to support effective learning and cost-efficient inference-time guidance.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b0', 'b10', 'b11', 'b12', 'b13', 'b14', 'b3', 'b15', 'b16', 'b17', 'b6', 'b18', 'b19', 'b20', 'b21', 'b22', 'b18', 'b23', 'b24', 'b25', 'b26', 'b26', 'b25', 'b7', 'b4', 'b20', 'b21', 'b0', 'b2', 'b13', 'b6', 'b19', 'b8', 'b9', 'b18', 'b27', 'b28']

Section: Preliminaries

this section cite: []

Section: User Instruction
Buy Sony WH-1000XM4 in Amazon.
[142] link 'Amazon', url='https://www.amazon.com/ref=nav_logo' [149] button 'Deliver to Pittsburgh 15213\u200c' [234] searchbox 'Search Amazon', autocomplete='list', ...
[239] button 'Go' [245] link 'Choose a language for shopping in Amazon United .. We formulate the web navigation problem as a partially observable Markov decision process (POMDP) defined by the tuple (S, A, O, T, R), where S is the set of environment states, A is the set of agent actions, O is the set of observations, T (s ′ | s, a) is the transition function, R(s, a) is the reward function. At each time step t, the agent receives a browser-rendered observation o t ∈ O that only partially reflects the true underlying state s t ∈ S. In the context of web environments, o t consists of two modalities: (1) an accessibility tree o txt t , a text sequence of intractable elements that captures the hierarchical and semantic structure of the webpage elements [1,30], and (2) a rendered screenshot image o img t depicting the visual appearance of the browser [3]. Given these observations, the agent selects an action a t ∈ A from a discrete set of browser-level commands, including operations such as click(i), scroll(d), and type("text"), where i is the index of a DOM or accessibility node, and d denotes a scroll direction or offset. The agent's goal is to select actions that maximize the expected reward over a trajectory τ = (o 1 , a 1 , . . . , o T ).
this section cite: ['b0', 'b29', 'b2']

Section: WEBPRM COLLECTION
The major challenge of building a PRM in web navigation is the lack of a training dataset. To address this, we collect WEBPRM COLLECTION, the first dataset for training PRMs for web agents. Our goal is to collect a dataset D that contains (I, O, C, A + , A -), where A + is a sequence of chosen actions (a + 1 , a + 2 , ..., a + n ), i.e., an expert trajectory, and A -is a sequence of rejected actions (a - 1 , a - 2 , ..., a - n ) along with the checklist C, observations O = (o 1 , o 2 , ..., o n ), and user instruction I.
this section cite: []

Section: Collecting User Instruction and Expert Trajectory
From human experts, we collect user instructions I and the chosen actions A + . We select websites that permit access via playwright from the pool of sites used in Mind2Web [16]. Prior to annotation,
Real-world Websites (70+) tvguide tvguide sixt sixt steam steam amazon amazon soundcloud soundcloud apple apple expedia expedia imdb imdb mta mta Human Annotator Browser Instruction Construct checklist and annotate feedback 4OK Dataset (851 Instructions) Can you check the warranty period for the Apple MacBook Pro 14" MW2U3LL/A (Late 2024) 14.2" Laptop Computer in Space Black? Observation [131] link 'please check back regularly for restocks', url='https://www.microcenter.com/search/...' [137] link 'Micro Center', url='https://www.microcenter.com/' [146] textbox 'Search for product' [147] button 'Begin Product Search' [155] link 'Shippable Items', url='https://  www.microcenter.com/' THOUGHT: The goal is to check the warranty period for the Apple ... ACTION: fill('146', 'Apple MacBook Pro 14" MW2U3LL/A') Checklist 1: Search for Product -The agent has filled the search bar with the exact product name which ...
this section cite: ['b15']

Section: Checklist
Feedback & Judgement all annotators participated in a three-hour training session designed to familiarize them with our annotation toolkit and to clarify the differences between human and agent browsing behaviors. Following annotation, all collected data were reviewed by a panel of 10 human evaluators to ensure quality and consistency. During this process, we filtered out invalid trajectories that could not be reproduced, as well as vague instructions prone to misinterpretation. Annotators were instructed to craft instructions I spanning three difficulty levels: easy, medium, and hard.
this section cite: []

Section: Annotating Checklist and Rejected Action
Checklist. To mitigate bias toward specific websites and reduce sensitivity to action orderings, we construct coarse-grained checklists that emphasize meaningful task progress over exact execution steps. For example, fine-grained actions such as filter A and filter B are abstracted into a higher-level subgoal like filtering. This abstraction enables the model to generalize across semantically equivalent strategies. Given an instruction I and an expert trajectory A + , we use GPT-4o to generate subgoal analysis and corresponding checklists.
this section cite: []

Section: Rejected actions.
To collect rejected actions a - t , we sample 5 candidate actions from diverse policies and select those that differ from the expert action a + t . However, some of these alternatives may correspond to valid but different actions toward task completion (e.g., fill(423, "Sony Camera") vs. click(search_box)), rather than being truly suboptimal or incorrect. To minimize such cases, we apply rule-based filtering and collect up to five rejected actions a - t per expert action a + t . More details about dataset construction are provided in Appendix B.
this section cite: []

Section: Dataset Statistics
As shown in Figure 4, we analyze two key aspects across difficulty levels: the length of agent trajectories and the number of checklist subgoals. The left violin plot illustrates that trajectory length increases with difficulty. Easy tasks generally require fewer steps (median ≈ 5), whereas medium tasks show more variability (median ≈ 9), and hard tasks involve significantly longer trajectories (median ≈ 20), with some exceeding 40 steps. This indicates that our difficulty annotation effectively reflects the complexity and required interaction depth. The right violin plot shows that the number of checklist items also grows with task difficulty, though the range is more concentrated. Easy tasks typically involve 3-4 checklist items, while medium and hard tasks consistently require 4-5 subgoals.
this section cite: []

Section: WEB-SHEPHERD
In this section, we introduce WEB-SHEPHERD, a process reward model designed to provide dense and reliable supervision to web agents and enable more informative credit assignment. We train WEB-SHEPHERD on the WEBPRM COLLECTION to support two key functionalities: (1) generating task-specific checklists, and (2) assigning rewards based on checklist completion.
this section cite: []

Section: Step 1: Checklist Generation
As illustrated in Figure 5, WEB-SHEPHERD first generates a checklist that outlines key intermediate milestones for achieving the user's goal. Given an instruction I, it produces a checklist C comprising a sequence of natural language subgoals (g 1 , g 2 , • • • , g k ). This checklist then serves as the foundation for reward prediction, enabling WEB-SHEPHERD to track progress toward the goal. We further investigate the impact of checklist quality in Section 7.1.
this section cite: []

Section: Step 2: Reward Modeling with Checklist
Reward modeling as next-token prediction. To leverage the internal reasoning capabilities of MLLMs, we choose next-token prediction as our learning objective [31]. We optimize the language modeling loss over targets formed by concatenating the feedback F and the judgment J, treating the full sequence as a coherent response. For example, given an input consisting of a checklist C, an observation o, and an answer a, the model is trained to generate the corresponding feedback and judgment in an auto-regressive manner. The loss is defined as:
L NTP = - t log P θ (y t | y <t , C, o, a),(1)
where y = [F ; J] denotes the concatenated feedback and judgment tokens. This objective encourages the model to learn to evaluate the trajectories based on the checklist with reasoning and provide valuable feedback that explains the evaluation.
Scoring process reward. Since the reward is predicted via token generation, the output resides in a discrete space. To obtain a continuous reward signal, several mapping strategies can be employed. One approach is to sample multiple output sequences and compute the average reward. Alternatively, we employ a verbalizer [32] to estimate soft probabilities over label tokens (e.g., "Yes", "No", and "In Progress") using the logits from the LM head. At inference time, WEB-SHEPHERD generates the feedback F ∼ P (•|I, C, o, a) and compute the reward for each checklist item using the probabilities of "Yes" and "In Progress" tokens follow: r k (o, a) = 1 L L l P ("Yes"|I, C, o, a, F ) + 0.5 × P ("In Progress"|I, C, o, a, F ),
where L denotes the number of checklist and r k is the score assigned to the k th response. The final reward is computed as the average: r(o, a) = K k=1 r k (o, a). We provide an empirical comparison of different scoring strategies in Appendix E.3.
this section cite: ['b30', 'b31']

Section: Experiments
To evaluate the effectiveness of PRMs for web navigation, we conduct comprehensive experiments in assigning process-level reward for web agents, focusing on both the accuracy of reward assignment and the utility of those rewards in improving agent performance.
this section cite: []

Section: WEBREWARDBENCH
In developing PRMs, a reliable benchmark (e.g., RewardBench [33]) is essential for evaluating their performance. However, there does not yet exist a benchmark specifically designed to evaluate how accurately models assign process rewards to web agents' trajectories. To address this, we introduce WEBREWARDBENCH, a benchmark that directly measures the accuracy of predicted rewards.
this section cite: ['b32']

Section: Setup
Benchmark construction. We use two data sources, Mind2Web and WebArena, to obtain user instructions for web navigation tasks. For Mind2Web, we utilize the expert demonstrations provided in the dataset. In contrast, since expert trajectories are unavailable in WebArena, we manually annotate them. As a result, we obtain 69 instances from WebArena and 707 instances from Mind2Web. To construct a reliable benchmark for evaluating PRMs, we follow the setup of Kim et al. [34] and collect preference pairs (o t , a + t , {a - (t,i) } 4 i=1 }), where each observation o t is paired with one chosen action and four rejected actions. Additionally, we provide reference checklists for each tasks to ensure fair and consistent evaluation. Further details on benchmark construction are provided in Appendix D.1.
Metrics. We evaluate process reward prediction using the following three metrics: (1) Mean Reciprocal Rank (MRR): The average of the reciprocal ranks of the preferred action in the list of all candidate actions sorted by predicted reward. A higher MRR indicates that the model consistently ranks the preferred action closer to the top. Baselines. Prior work has leveraged prompted MLLMs to obtain process-level rewards by exploiting their reasoning and image understanding capabilities [5,8]. Following this approach, we construct baselines using representative MLLMs from both open-source and closed-source categories. For open-source models, we use GPT-4o-mini and GPT-4o; for closed-source models, we adopt Qwen-2.5-VL-72B, which are widely used in recent literature.
this section cite: ['b33', 'b4', 'b7']

Section: Implementation of WEB-SHEPHERD.
We train WEB-SHEPHERD on our dataset using the following base models: for text-only settings, we use Qwen2.5-3B [35] and Qwen3-8B [36]; for multimodal settings, we use Qwen2.5-VL-3B [37]. All models are trained for 3 epochs using LoRA [38].
this section cite: ['b34', 'b35', 'b36', 'b37']

Section: Results
MLLMs struggle with assigning correct process rewards. We evaluate the ability of models to accurately assign process rewards on WEBREWARDBENCH under different input types (text only vs. text and image) and with or without using the checklist. As shown in Table 1, state-of-the-art MLLMs struggle to provide reliable rewards for web navigation tasks. 3 This limitation is particularly Checklist allows reliable reward assignment. Table 1 demonstrates that both baseline and our models benefit significantly from the checklist in assigning rewards. Checklists lead to more accurate and consistent reward assignments, as evidenced by improvements in trajectory accuracy across all baselines. These results suggests that checklists serve as valuable guidance, helping models maintain coherence in predicting the process reward. Furthermore, as shown in Figure 6, when we conduct ablation studies with models that are trained to
0 20 40 60 80 100 WebRewardBench Score feedback + w/o checklist Web-Shepherd (ours) w/o feedback + checklist either assign rewards without checklists or use checklists without feedback, we observe a substantial performance drop. These findings underscore the importance of both checklists and feedback for assigning reliable rewards.
Multimodal input does not always improve performance. Contrary to our expectations, incorporating multimodal input does not always lead to performance gains; in some cases, using multimodal input even degrades the performance. For example, when using GPT-4o as the reward model, we observe a notable improvement in trajectory accuracy only on the cross-website of Mind2Web subset. This observation is consistent with the findings of Xue et al. [6], which suggest that processing inputs from multiple modalities can introduce ambiguity and act as a source of noise, ultimately hindering the model performance.
this section cite: ['b5']

Section: Reward-Guided Trajectory Search
Reward-guided search using Best-of-n (BoN) sampling offers a practical proxy for evaluating the capability of a reward model to guide policies [10,39,40]. Notably, it allows us to assess the potential for reward overoptimization without relying on reinforcement learning. In addition, it provides an effective approach to adapting an MLLM policy without fine-tuning [8,5,41].
Setup. We evaluate our approach on WebArena-lite and WorkArena [30] in an online setting. WebArena-lite [11] is a subset of WebArena [1], comprising 165 instructions with error-corrected judge code from the earlier version. WorArena is a remote-hosted benchmark of 33 tasks based on the widely-used ServiceNow platform. Among 5 action candidates sampled from the policy, the action that is assigned the highest reward is executed. For the policy, we use GPT-4o-mini, and compare Main results. We present the results in Table 2. Interestingly, when using GPT-4o-mini as the reward model, we observe a slight improvement in the GPT-4o-mini policy. However, overall performance degrades when GPT-4o is used as the policy model, dropping from 31.52 to 26.67. In contrast, applying WEB-SHEPHERD leads to substantial performance gains for both the GPT-4o-mini and GPT-4o policies across nearly all domains. Notably, WEB-SHEPHERD boosts the GPT-4o-mini's browsing performance from 23.64 to 34.55, which is about 3 points higher than GPT-4o without trajectory search. These results suggest that WEB-SHEPHERD remains effective in the online setting, even when paired with a stronger policy model.
this section cite: ['b9', 'b38', 'b39', 'b7', 'b4', 'b40', 'b29', 'b10', 'b0']

Section: Results on WorkArena.
To assess the robustness across domains, we also evaluate our models on WorkArena, a benchmark completely out-of-domain for WEB-SHEPHERD. As shown in Table 3, trajectory search guided by the PRM improves the success rate in WorkArena, where the Total score increases from 9.39 to 12.42 when comparing the baseline without trajectory search. Moreover, our model consistently outperforms GPT-4o-mini across all domains except for Menu. We attribute the relatively low performance in the Menu domain to the complexity of its multi-level dropdowns and embedded search boxes, which cause the policy model to produce unreliable action candidates. Can WEB-SHEPHERD provide useful feedback? To evaluate the effectiveness of the feedback generated by WEB-SHEPHERD, we conduct experiments in which the agent performs step-wise refinement using our feedback, similar to the Self-Refine [42]. Specifically, the agent refine current action with the feedback when its current reward is lower than the previous reward assigned by WEB-SHEPHERD. Interestingly, contrary to previous findings by Chae et al. [5] suggesting that step-wise feedback from models is not helpful and may even be detrimental, we observe notable improvements when incorporating model feedback during refinement. A possible explanation is that WEB-SHEPHERD not only learns the impact of actions but also identifies patterns that characterize suboptimal behavior.
this section cite: ['b41', 'b4']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Webarena: A realistic web environment for building autonomous agents Year: (2024)
Ref_id:b1 Title: Benchmarking web agents in online environments Year: (2024)
Ref_id:b2 Title: Visualwebarena: Evaluating multimodal agents on realistic visual web tasks Year: (2024)
Ref_id:b3 Title: Agent workflow memory Year: (2024)
Ref_id:b4 Title: Web agents with world models: Learning and leveraging environment dynamics in web navigation Year: (2025)
Ref_id:b5 Title: An illusion of progress? assessing the current state of web agents Year: (2025)
Ref_id:b6 Title: Autonomous evaluation and refinement of digital agents Year: (2024)
Ref_id:b7 Title: Tree search for language model agents Year: (2024)
Ref_id:b8 Title: Let's verify step by step Year: (2023)
Ref_id:b9 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2024)
Ref_id:b10 Title: Visualagentbench: Towards large multimodal models as visual foundation agents Year: (2025)
Ref_id:b11 Title: Gpt-4v (ision) is a generalist web agent, if grounded Year: (2024)
Ref_id:b12 Title: Synapse: Trajectory-as-exemplar prompting with memory for computer control Year: (2023)
Ref_id:b13 Title: Building an end-to-end web agent with large multimodal models Year: (2024)
Ref_id:b14 Title: Webwise: Web interface control and sequential exploration with large language models Year: (2023)
Ref_id:b15 Title: Mind2web: towards a generalist agent for the web Year: (2023)
Ref_id:b16 Title: Seeclick: Harnessing gui grounding for advanced visual gui agents Year: (2024)
Ref_id:b17 Title: Dualview visual contextualization for web navigation Year: (2024)
Ref_id:b18 Title: Tree search for language model agents Year: (2024)
Ref_id:b19 Title: Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning Year: (2025)
Ref_id:b20 Title: Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning Year: (2024)
Ref_id:b21 Title: Digi-q: Learning q-value functions for training device-control agents Year: (2025)
Ref_id:b22 Title: Alphazero-like tree-search can guide large language model decoding and training Year: (2023)
Ref_id:b23 Title: Openai o1 system card Year: (2024)
Ref_id:b24 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b25 Title: Reflexion: Language agents with verbal reinforcement learning Year: (2024)
Ref_id:b26 Title: Autonomous evaluation and refinement of digital agents Year: (2024)
Ref_id:b27 Title: Agent q: Advanced reasoning and learning for autonomous ai agents Year: (2024)
Ref_id:b28 Title: Advancing mobile gui agents: A verifier-driven approach to practical deployment Year: (2025)
Ref_id:b29 Title: How capable are web agents at solving common knowledge work tasks? arXiv preprint Year: (2024)
Ref_id:b30 Title: Chelsea Finn, and Alon Albalak. Generative reward models Year: (2024)
Ref_id:b31 Title: Knowledgeable prompt-tuning: Incorporating knowledge into prompt verbalizer for text classification Year: (2022)
Ref_id:b32 Title: Evaluating reward models for language modeling Year: (2024)
Ref_id:b33 Title: Evaluating robustness of reward models for mathematical reasoning Year: (2024)
Ref_id:b34 Title:  Year: (2024)
Ref_id:b35 Title:  Year: (2025)
Ref_id:b36 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b37 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b38 Title: Pairwise rm: Perform best-of-n sampling with knockout tournament Year: (2025)
Ref_id:b39 Title: Inference-time scaling for generalist reward modeling Year: (2025)
Ref_id:b40 Title: Is your llm secretly a world model of the internet? model-based planning for web agents Year: (2024)
Ref_id:b41 Title: Self-refine: Iterative refinement with self-feedback Year: (2024)
Ref_id:b42 Title: G-eval: Nlg evaluation using gpt-4 with better human alignment Year: (2023)
Ref_id:b43 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b44 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b45 Title: React: Synergizing reasoning and acting in language models Year: ()
Ref_id:b46 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b47 Title: Ligerkernel: Efficient triton kernels for llm training Year: (2024)
Ref_id:b48 Title: Generative verifiers: Reward modeling as next-token prediction Year: (2024)
Ref_id:b49 Title: Rim Assouel, et al. The browsergym ecosystem for web agent research Year: (2024)
