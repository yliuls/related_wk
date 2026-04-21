Title: Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search
Abstract: In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into "tree search" for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoM-CTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code is available at https://github.com/HJYao00/Mulberry.

Section: Introduction
"What I cannot create, I do not understand."
this section cite: []

Section: -Richard Feynman
Multimodal large language models (MLLMs) embody the essence of this dictum, which understand the world by learning to create expected responses to multimodal inputs such as images and text. While MLLMs have recently shown significant progress in straightforward tasks [1,2], they often experience obviously increased failures on complex tasks requiring in-depth reasoning [3]. Feynman's dictum might be the perfect metaphor of such failures of MLLMs, as we should only be able to work something out if we can create and have a firm understanding of each step of the reasoning involved. However, current MLLMs predominantly operate in a simple "direct prediction" mode [4], i.e., generating brief, final answers to questions with little explicit and well-defined intermediate reasoning steps.
In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. Recent advances in NLP, such as OpenAI o1 [5], have shown great potential in enabling LLM to learn to reason and tackle complex language tasks [6]. The core design of these advances lies in AlphaGo-like "tree search": they employ tree search methods, like MCTS [7], to bootstrap an LLM itself to build a tree 39th Conference on Neural Information Processing Systems (NeurIPS 2025). of intermediate thoughts, explore effective reasoning paths, and leverage these paths to teach model to reason step-by-step.
An intuitive idea is to directly apply these "tree search" methods to search effective reasoning paths for MLLMs, which, however, does not work well. As shown in Figure 1, we believe this is largely attributed to several search challenges for MLLMs. (1) Search Effectiveness: Traditional MCTS methods [7,8,9,10] generally work by self-bootstrapping while current MLLMs are typically trained with little explicit and well-defined intermediate reasoning steps, making these search methods often trapped in homogeneous low-quality nodes within the reasoning space of a single MLLM, ultimately leading to low search success rates. (2) Search Efficiency: Traditional MCTS methods typically expand and explore only one reasoning node per search iteration, which advance a single step each time and demand massive iterations, making them inefficient for computation-intensive MLLMs.
To tackle these challenges, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into "tree search" for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge to collaboratively conjecture, search and identify effective reasoning paths toward correct answers. Specifically, CoMCTS searches effective reasoning paths iteratively, and in each iteration, it leverages collective knowledge from multiple MLLMs to jointly (a) expand diverse and complementary candidate subsequent reasoning nodes till the end from a given start node, (b) simulate reasoning outcomes, position error candidate nodes and prune them along with their child nodes, (c) backpropagate to update the score and visit count of each reasoning node in a bottom-up manner, and (d) select the leaf reasoning node with the highest Upper Confidence Bound value as next start node.
In this way, our CoMCTS achieves effective and efficient reasoning search. (1) The joint expansion mechanism enables CoMCTS to concatenate reasoning trajectories from multiple MLLMs via iterative search, ultimately constructing an unified reasoning tree comprising diverse and complementary reasoning nodes. Thus, it allows reasoning-path search not only within the reasoning space of a given MLLM itself but also among those of others, benefiting from the synergy of multiple MLLMs while avoiding being trapped in homogeneous low-quality nodes within the reasoning space of a single MLLM itself. (2) The joint simulation and error positioning mechanism enables CoMCTS to, in each search iteration, skip multiple intermediate steps and select the last correct step as the next start node, largely reducing search time while maintaining search effectiveness. Here, collective knowledge is also crucial as it is often challenging for a model to recognize and position errors made by itself while relatively easy by using other models.
Furthermore, we extend our CoMCTS for reflective reasoning-path search. Based on the unified reasoning tree constructed by CoMCTS, which provides both positive and negative reasoning nodes , we identify and integrate negative sibling nodes into effective reasoning paths to build the reflective reasoning path that includes a transition from a negative reasoning node to a positive one. By learning from reflective reasoning paths, MLLMs can perform appropriate step-wise reflection, dynamically calibrating their reasoning trajectory from an erroneous node toward a correct one during long-chain reasoning. Here, collective knowledge facilitates reflective reasoning-path search by providing a rich set of diverse positive and negative reasoning nodes.
Using our CoMCTS, we search effective and reflective reasoning paths for a set of multimodal inputs, and construct Mulberry-260k, a Multimodal learning-to-Reason-and-Reflect dataset with a tree of rich, explicit and well-defined reasoning nodes for each question.
With Mulberry-260k, we perform collective supervised fine-tuning to train our model, Mulberry, a series of Multimodal LLMs with o1-like step-by-step Reasoning and Reflection capabilities. The main contributions of this work are fourfold. First, we introduce the concept of collective learning into MCTS, and propose CoMCTS which leverages collective knowledge to collaboratively conjecture, search and identify effective and reflective reasoning paths for MLLMs, significantly improving search effectiveness and efficiency. To the best of our knowledge, this is the first work that explores collective learning with MCTS for MLLMs. Second, we construct Mulberry-260k that provides a valuable resource for advancing research in step-by-step reasoning and reflection in MLLMs. Third, we develop Mulberry, a series of MLLMs with outstanding capabilities in stepby-step reasoning and reflection. Fourth, extensive experiments demonstrate the superiority of our proposed methods on various benchmarks.
2 Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b6', 'b7', 'b8', 'b9', 'b0']

Section: Multimodal Large Language Model
MLLMs [1,2,11,12,13,14,15,16] have made notable advancements in general vision-language understanding, enabling them to interpret visual semantics across various domains. Recent studies [17,3] explore MLLM reasoning and reveal that directly employing CoT prompt to derive the final answer may result in limited gains or even degradation. In addition, some studies [18,19] introduce planbased CoT prompting to guide models to generate intermediate information for predicting final answers. Recent advances [4] attempt structured reasoning with a planed flow of certain pre-defined stages, enhancing the CoT capabilities [15] of MLLMs. Differently, this paper, for the first time, introduces the concept of "tree search" into MLLM reasoning and proposes a novel CoMCTS technique to search effective and reflective reasoning paths to train our Mulberry, a series of MLLMs with outstanding capabilities in step-by-step reasoning and reflection.
this section cite: ['b0', 'b1', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b2', 'b17', 'b18', 'b3', 'b14']

Section: Large Language Model Reasoning
LLM reasoning methods can be broadly categorized into three types, i.e., prompt-based, plan-based and learning-based reasoning. Prompt-based methods, like Chain-of-Thought (CoT) [20], mimic human reasoning by providing a few hand-crafted, step-by-step solutions as references. Plan-based methods, such as Tree/Graph-of-thought [21,22], predict multiple reasoning paths in a tree or graph manner and take consistent units of thought for thoughtful decision-making. Learning-based reasoning methods, represented by GPTo1, Star [23], Iter-MCTS [6] and ReST-MCTS [24], first employ tree search approaches [25], like MCTS, to bootstrap an LLM itself to build a tree of intermediate thoughts, explore effective reasoning paths, and leverage these paths to train model to reason step-by-step.
this section cite: ['b19', 'b20', 'b21', 'b22', 'b5', 'b23', 'b24']

Section: Monte-Carlo Tree Search
Monte-Carlo Tree Search (MCTS) is a powerful search paradigm for complex decision making problems and has been extensively explored across diverse fields, including games [26,27], robotics [28,29], theorem proving [30], matrices multiplication [31], etc. For instance, AlphaGo [26] introduces deep learning into MCTS, achieving superhuman results in board and video games [26,27]. Besides, [32,33] explore MCTS for path finding and train timetabling problems, while [34] integrates MCTS into physics-informed planning networks for robot control. In this work, we propose CoMCTS that enables effective and reflective reasoning-path searching and learning on MLLMs.
this section cite: ['b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b25', 'b25', 'b26', 'b31', 'b32', 'b33']

Section: Collective Learning
Collective learning, also known as Co-training, aims to harness collective intelligence of multiple individuals to improve learning outcomes. This concept originates in early pioneering studies [35,36,37], which utilize collective knowledge to address data insufficiency issues in classification learning. Recent advances introduce collective learning into deep neural networks for efficient and effective deep learning. For example, [38,39] employ collective knowledge from multiple classifiers to predict more accurate pseudo-labels for semi-supervised classification; [40] utilizes collective knowledge from multiple discriminators to enhance image discrimination and generation; and [41] leverages the synergy of multiple models for reinforcement learning.
this section cite: ['b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40']

Section: Method
We first present CoMCTS that introduces collective learning into "tree search" for effective and efficient reasoning-path searching and learning. We then illustrate the extension of CoMCTS for reflective reasoning-path search, and describe data construction and model training using CoMCTS.
this section cite: []

Section: CoMCTS for effective reasoning
The core idea of CoMCTS is to leverage collective knowledge to collaboratively conjecture, search and identify effective reasoning nodes in an iterative manner, aiming to find effective reasoning paths leading to correct answers.
We denote a policy model as π, initialized by a pre-trained MLLM. We leverage collective knowledge from a group of MLLMs {π 1 , π 2 , ... (a) Expansion. The goal of this operation in CoMCTS is to expand the current leaf reasoning node (if it is not a terminal node) to integrate new subsequent candidate reasoning nodes. Given the current leaf node s k m (i.e., the node selected by Operation (d) Selection or the root node), CoMCTS utilizes collective knowledge from a group of MLLMs, {π 1 , π 2 , ..., π K }, to jointly expand a set of diverse and complementary candidate reasoning paths S candidate = ∪ K j=1 S j candidate in parallel till terminal node:
S j candidate ∼ π j (•|Q, Parent(s k m ), s k m ),(1)
where Parent(s k m ) returns all parent nodes of s k m and (Parent(s k m ), s k m ) denotes the current reasoning path from the root node to s k m . S j candidate = {s j i } stands for a potential reasoning path generated by model π j starting from s k m . (b) Simulation and Error Positioning. In this operation, CoMCTS utilizes collective knowledge from {π 1 , π 2 , ..., π K } to jointly estimate the potential value of child nodes s j i ∈ S candidate (added in Operation (a)), and considers low-score nodes as erroneous reasoning nodes, and positions and filters out them along with their child nodes: R(
s j i ) = 1 K K l=1 π l (•|prompt eval , Q, Parent(s j i ), s j i )(2)
S * candidate = {s j i ∈ S candidate |R(s j i ) >= t}(3)
where R(s j i ) denotes a reasoning node evaluation function that uses the prompt, prompt eval , to request a group of MLLMs, {π 1 , π 2 , ..., π K }, to jointly evaluate the candidate reasoning node s j i . t is a threshold and discontinued reasoning nodes in S * candidate are automatically removed following the error node removal in Eq.(3). CoSFT with CoMCTS Data including visit count N and node value V :
𝑆" " 𝑆" # 𝑆" " 𝑆" # 𝑆" " 𝑆" # 𝑆" " 𝑆" # 𝑆" ! 𝑆" ! 𝑆" ! 𝑆# ! 𝑆# " 𝑆# # 𝑆$ ! 𝑆$ " 𝑆$ # 𝑆% ! 𝑆% " 𝑆% # 𝑆& ! 𝑆' " 𝑆' # 𝑆# ! 𝑆# " 𝑆# # 𝑆$ ! 𝑆$ " 𝑆$ # 𝑆% ! 𝑆% " 𝑆% # 𝑆' ! 𝑆' " 𝑆' # 𝑆# ! 𝑆# " 𝑆# # 𝑆$ ! 𝑆$ # 𝑆% # 𝑆# ! 𝑆# " 𝑆# # 𝑆$ ! 𝑆$ # 𝑆% #
V (s) ← - N (s) • V (s) + s l ∈Child(s) R(s l ) N (s) + CountChild(S * candidate , s) ,(4)
N (s) ← -N (s) + CountChild(S * candidate , s),(5)
where Child(s) returns all the child nodes of s, and CountChild(S * candidate , s) is a child node counting function that calculates the number of child nodes of s in S * candidate . (d) Selection. Following Operations (a), (b) and (c), CoMCTS traverses the updated reasoning tree to select the next starting node. This selection is guided by the Upper Confidence Bound (UCB) value, which balances search exploration and exploitation. The UCB value of a node s is computed using the node reward value V (s) and the visit count N (s). Among the candidate nodes s ∈ S * candidate , the one with the highest UCB value is chosen as the starting node s k * m for next search iteration:
s k * m = arg max s∈S * candidate V (s) + c • log N (ŝ) 1 + N (s)(6)
where c stands for a constant which controls the level of exploration. ŝ denotes the parent node of s.
CoMCTS. These four operations, i.e., (a) Expansion, (b) Simulation and Error Positioning, (c) Backpropagation and (d) Selection, are repeated for a pre-defined number of iterations or until correct reasoning paths are found. This iterative process allows CoMCTS to construct a question-dependent reasoning tree S with the correct reasoning path Y , and ultimately form a multimodal learningto-reason data triplet {Q, Y, S}. By applying our CoMCTS to a set of multimodal questions, we can construct a collection of multimodal learning-to-reason data triplets, which provide a tree of rich, explicit and well-defined reasoning nodes toward the final answer for each question and enable MLLMs to learn to reason step-by-step.
this section cite: []

Section: CoMCTS for reflective reasoning
In this subsection, we extend CoMCTS for reflective reasoning-path search. Based on the unified reasoning tree constructed by CoMCTS, i.e., {Q, Y, S}, which provides both positive and negative reasoning nodes, we identify and integrate negative sibling nodes into effective reasoning paths to build the reflective reasoning path that includes a transition from a negative reasoning node to a positive one.
Identifying negative sibling node. Given the effective reasoning path Y , we identify the negative sibling reasoning node for s ∈ Y using UCB:
s neg = arg min s l ∈Sibling(s) UCB(s l ) -UCB(s), ∀s ∈ Y,(7)
where Sibling(s) returns all the sibling nodes of s, i.e., the nodes on the same hierarchical level under the same parent node of s. UCB(s) = V (s) + c • log N (ŝ) 1+N (s) as in Eq. 6. Constructing reflective reasoning path. Based on Eq. 7, we randomly sample a reasoning node s ∈ Y with its negative sibling node s neg , and concatenate them with a reflection prompt to form a reflection trajectory, i.e., (s neg , prompt reflect , s). We then use a function Replace(•) that replaces s ∈ Y with (s neg , prompt reflect , s) to convert Y into the reflective reasoning path Y reflect :
Y reflect = Replace(Y, s, (s neg , prompt reflect , s)),(8)
where prompt reflect denotes a reflection prompt, such as "The previous reasoning step is wrong and let's rethink it again." Then, we can integrate the reflective reasoning path Y reflect into our data as a quadruplet {Q, Y, Y reflect , S} ∈ D.
Collective Supervised Fine-Tuning (CoSFT). Given (Q, Y ) ∈ D, we apply standard SFT objective to train our MLLM to learn from D constructed by CoMCTS:
L CoSFT (π k ) = (Q,Y )∈D log π k (Y |Q),(9)
where Y = {s} denotes the effective reasoning path that includes a sequence of reasoning nodes collectively conjectured, searched and identified by a group of MLLMs.
CoSFT for reflective reasoning. Given a question and its reasoning tree (Q, S) ∈ D constructed by CoMCTS, we randomly sample a reflective reasoning path Y reflect from S as in Eqs.7-8, and conduct CoSFT for reflective reasoning:
L CoSFT-Re (π k ) = (Q,Yreflect)∈D log π k (Y reflect |Q),(10)
where Y reflect = {s} denotes the reflective reasoning path that includes an additional step-wise reflection trajectory.
this section cite: []

Section: Algorithm 1 Training Mulberry with CoMCTS
Input: a set of policy models {π 1 , π 2 , ..., π K } initialized by different MLLMs; a set of multimodal questions D Q for i = 1 to MaxEpoch do Reasoning Tree Search using CoMCTS:
for Q ∈ D Q do Collective Monte Carlo tree search: {Q, Y, S} = CoMCTS({π 1 , π 2 , ..., π K }; Q)
if found an effective reasoning path then Search and find Y reflect from S Add {Q, Y, Y reflect , S} into D Remove Q from D Q Model Training with CoMCTS Reasoning Trees: for k = 1 to K do for (Q, Y, Y reflect , S) ∈ D do Supervised Fine-Tuning: Optimize π k via L CoSFT (π k ) and
L CoSFT-Re (π k ) Output: Trained policy models {π 1 , π 2 , ..., π K }
The goal of L CoSFT and L CoSFT-Re is to maximize the log probability of effective and reflective reasoning path Y and Y reflect over a tree of reasoning nodes S generated by CoMCTS. In addition, L CoSFT-Re enables to leverage the negative information during CoMCTS search process by learning to calibrate negative reasoning nodes.
this section cite: []

Section: Training with Collective MCTS
Using CoMCTS, we search effective and reflective reasoning paths for a set of multimodal input questions, and construct Mulberry-260k, a multimodal learning-to-reason-and-reflect dataset with a tree of rich, explicit and welldefined reasoning nodes for each question, i.e., a set of quadruplets {Q, Y, Y reflect , S} ∈ D. To learn collective knowledge from Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of Multimodal LLMs with o1-like step-by-step Reasoning and Reflection capabilities.
this section cite: []

Section: Experiments
In this section, we first introduce our CoMCTS-generated dataset, Mulberry-260K, including its sources, construction, and analysis in Sec. 4.1, and provide implementation details in Sec. 4.2. We then present the main results in Sec. 4.3, demonstrating the effectiveness of the searched data (i.e., Mulberry-260K) and the trained models (i.e., Mulberry). In Sec. 4.4, we perform comprehensive ablation studies on the impact of effective and reflective reasoning data and the contributions of collective knowledge sources. Sec. 4.5 discusses the effectiveness and efficiency of CoMCTS with other tree search methods.
this section cite: []

Section: Dataset
The Sources of Raw Data. To construct a comprehensive and general-purpose tree-based reasoning dataset, we collect 260K raw multimodal input questions (i.e., a text task instruction with an image as an input question) from a wide range of domains, covering General Multimodal Understanding, Mathematics, Figure Understanding, Realworld Understanding, Science, Medical Image Understanding, etc. The specific data sources are provided in the Appendix I.
this section cite: []

Section: Reasoning Data Construction.
As detailed in Sec. 3 and Algorithm 1 and visually illustrated in Figures 2 and 3, we employ our CoMCTS to search effective and reflective reasoning paths for a set of raw multimodal input questions as collected from the mentioned "The Sources of Raw Data", ultimately constructing our dataset, Mulberry-260K. Note we only sample 15K data for reflective reasoning training to avoid overabundance of reflection data.
this section cite: []

Section: Implementation Detail
We implement collective learning in CoMCTS with four models, including GPT-4o, Qwen2-VL-7B, LLaMA-3.2-11B-Vision-Instruct, and Qwen2-VL-72B, to construct Mulberry-260K. In CoMCTS, we set maximum search iteration to 20 and threshold t in Eq. 3 to 0
this section cite: []

Section: Main Results
To examine the effectiveness of searched data (i.e., Mulberry-260K) and trained models (i.e., Mulberry), we conduct extensive experiments with four powerful baseline models, and comprehensively benchmark our Mulberry with various state-of-the-arts, including general and reasoning-based
this section cite: []

Section: Ablation Study Ablation Study on CoMCTS.
We conduct ablation studies with powerful GPT-4o as the baseline over 1K samples from Geo3K [52] and GeoQA-Plus [53], as shown in Tab. 2. As the core of CoMCTS, we examine how each model in the collective learning group contributes to the overall tree search performance. Tab. 2 reports the Search Success Rates, and baseline GPT-4o works not
this section cite: ['b51', 'b52']

Section: Conclusion
This paper presents CoMCTS, a new learning-to-reason approach for MLLMs, which introduces the concept of collective learning into "tree search" for effective and efficient reasoning-path searching and learning. Based on the CoMCTS, we search effective and reflective reasoning paths for a set of multimodal inputs, and construct Mulberry-260k, a multimodal learning-to-reason-and-reflect dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. Using Mulberry-260k, we train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Furthermore, extensive experiments, ablation studies and discussion demonstrate the superiority of our proposed methods on various benchmarks. We hope that CoMCTS along with Mulberry-260k and Mulberry will provides valuable resources and offer new insights for multimodal MCTS search and reasoning.
this section cite: []

Section: References
Ref_id:b0 Title: Visual instruction tuning Year: (2024)
Ref_id:b1 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b2 Title: Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning Year: (2024)
Ref_id:b3 Title: Llava-o1: Let vision language models reason step-by-step Year: (2024)
Ref_id:b4 Title:  Year: (2024)
Ref_id:b5 Title: Monte carlo tree search boosts reasoning via iterative preference learning Year: (2024)
Ref_id:b6 Title: Efficient selectivity and backup operators in monte-carlo tree search Year: (2006)
Ref_id:b7 Title: Mutual reasoning makes smaller llms stronger problem-solvers Year: (2024)
Ref_id:b8 Title: Improve mathematical reasoning in language models by automated process supervision Year: (2024)
Ref_id:b9 Title: Protinvtree: Deliberate protein inverse folding with reward-guided tree search Year: (2025)
Ref_id:b10 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b11 Title: Dense connector for mllms Year: (2024)
Ref_id:b12 Title: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization Year: (2025)
Ref_id:b13 Title: R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo Year: (2025)
Ref_id:b14 Title: Vision-language models for vision tasks: A survey Year: (2024)
Ref_id:b15 Title: A survey on agentic multimodal large language models Year: (2025)
Ref_id:b16 Title: Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark Year: (2024)
Ref_id:b17 Title: Compositional chain-of-thought prompting for large multimodal models Year: (2024)
Ref_id:b18 Title: Zoom in for enhanced multimodal text-rich image understanding Year: (2024)
Ref_id:b19 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b20 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
Ref_id:b21 Title: Graph of thoughts: Solving elaborate problems with large language models Year: (2024)
Ref_id:b22 Title: Star: Bootstrapping reasoning with reasoning Year: (2022)
Ref_id:b23 Title: Rest-mcts*: Llm self-training via process reward guided tree search Year: (2024)
Ref_id:b24 Title: Ensembling large language models with process reward-guided tree search for better complex reasoning Year: (2024)
Ref_id:b25 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b26 Title: Mastering atari games with limited data Year: (2021)
Ref_id:b27 Title: Dec-mcts: Decentralized planning for multi-robot active perception Year: (2019)
Ref_id:b28 Title: Monte-carlo robot path planning Year: (2022)
Ref_id:b29 Title: Hypertree proof search for neural theorem proving Year: (2022)
Ref_id:b30 Title: Discovering faster matrix multiplication algorithms with reinforcement learning Year: (2022)
Ref_id:b31 Title: Montecarlo tree search for multi-agent pathfinding: Preliminary results Year: (2023)
Ref_id:b32 Title: An integrated framework integrating monte carlo tree search and supervised learning for train timetabling problem Year: (2023)
Ref_id:b33 Title: Phyplan: Compositional and adaptive physical task reasoning with physics-informed skill networks for robot manipulators Year: (2024)
Ref_id:b34 Title: Combining labeled and unlabeled data with co-training Year: (1998)
Ref_id:b35 Title: Robust co-training Year: (2011)
Ref_id:b36 Title: Bayesian co-training Year: (2011)
Ref_id:b37 Title: Deep co-training for semi-supervised image recognition Year: (2018)
Ref_id:b38 Title: Maximum classifier discrepancy for unsupervised domain adaptation Year: (2018)
Ref_id:b39 Title: Generative co-training for generative adversarial networks with limited data Year: (2022)
Ref_id:b40 Title: Learning to communicate with deep multi-agent reinforcement learning Year: (2016)
Ref_id:b41 Title: Gpt-4o system card Year: (2024)
Ref_id:b42 Title:  Year: (2024)
Ref_id:b43 Title: Mm1. 5: Methods, analysis & insights from multimodal llm fine-tuning Year: (2024)
Ref_id:b44 Title: Building and better understanding vision-language models: insights and future directions Year: (2024)
Ref_id:b45 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b46 Title: Minicpm-v: A gpt-4v level mllm on your phone Year: (2024)
Ref_id:b47 Title: Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding Year: (2024)
Ref_id:b48 Title: Insight-v: Exploring long-chain visual reasoning with multimodal large language models Year: (2024)
Ref_id:b49 Title: Llava-next: Stronger llms supercharge multimodal capabilities in the wild Year: (2024-05)
Ref_id:b50 Title: The llama 3 herd of models Year: (2024)
Ref_id:b51 Title: Intergps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: ()
Ref_id:b52 Title: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning Year: (2021)
Ref_id:b53 Title: Bandit based monte-carlo planning Year: (2006)
Ref_id:b54 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b55 Title: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback Year: (2024)
Ref_id:b56 Title: Ovis: Structural embedding alignment for multimodal large language model Year: (2024)
Ref_id:b57 Title: Octopus: A multi-modal llm with parallel recognition and sequential understanding Year: (2024)
Ref_id:b58 Title:  Year: (2024)
Ref_id:b59 Title: What matters when building visionlanguage models? arXiv preprint Year: (2024)
Ref_id:b60 Title: Mini-gemini: Mining the potential of multi-modality vision language models Year: (2024)
Ref_id:b61 Title: Multimodal chain-of-thought reasoning in language models Year: (2023)
Ref_id:b62 Title: Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning Year: (2024)
Ref_id:b63 Title: G-llava: Solving geometric problem with multi-modal large language model Year: (2023)
Ref_id:b64 Title: Solving geometry problems: Combining text and diagram interpretation Year: (2015)
Ref_id:b65 Title: Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression Year: (2022)
Ref_id:b66 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b67 Title: Geomverse: A systematic evaluation of large models for geometric reasoning Year: (2023)
Ref_id:b68 Title: Math-llava: Bootstrapping mathematical reasoning for multimodal large language models Year: (2024)
Ref_id:b69 Title: Dvqa: Understanding data visualizations via question answering Year: (2018)
Ref_id:b70 Title: Docvqa: A dataset for vqa on document images Year: (2021)
Ref_id:b71 Title: Figureqa: An annotated figure dataset for visual reasoning Year: (2017)
Ref_id:b72 Title: Plotqa: Reasoning over scientific plots Year: (2020)
Ref_id:b73 Title: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b74 Title: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Year: (2022)
Ref_id:b75 Title: Numerical reasoning over multi hierarchical tabular and textual data Year: (2022)
Ref_id:b76 Title: Mitigating hallucination in large multi-modal models via robust instruction tuning Year: (2023)
Ref_id:b77 Title: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning Year: (2021)
Ref_id:b78 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: (2022)
Ref_id:b79 Title: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning Year: (2017)
Ref_id:b80 Title: Clevr-math: A dataset for compositional language, visual and mathematical reasoning Year: (2022)
Ref_id:b81 Title: Super-clevr: A virtual benchmark to diagnose domain robustness in visual reasoning Year: (2023)
Ref_id:b82 Title: A dataset of clinically generated visual questions and answers about radiology images Year: (2018)
Ref_id:b83 Title: Pmcvqa: Visual instruction tuning for medical visual question answering Year: (2023)
Ref_id:b84 Title: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension Year: (2017)
Ref_id:b85 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b86 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b87 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b88 Title: Aokvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b89 Title: Towards vqa models that can read Year: (2019)
Ref_id:b90 Title: Vizwiz grand challenge: Answering visual questions from blind people Year: (2018)
Ref_id:b91 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
