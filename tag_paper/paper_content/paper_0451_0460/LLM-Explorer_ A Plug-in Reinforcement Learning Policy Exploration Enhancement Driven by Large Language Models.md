Title: LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models
Abstract: Policy exploration is critical in reinforcement learning (RL), where existing approaches include ϵ-greedy, Gaussian process, etc. However, these approaches utilize preset stochastic processes and are indiscriminately applied in all kinds of RL tasks without considering task-specific features that influence policy exploration. Moreover, during RL training, the evolution of such stochastic processes is rigid, which typically only incorporates a decay in the variance, failing to adjust flexibly according to the agent's real-time learning status. Inspired by the analyzing and reasoning capability of large language models (LLMs), we design LLM-Explorer to adaptively generate task-specific exploration strategies with LLMs, enhancing the policy exploration in RL. In our design, we sample the learning trajectory of the agent during the RL training in a given task and prompt the LLM to analyze the agent's current policy learning status and then generate a probability distribution for future policy exploration. Updating the probability distribution periodically, we derive a stochastic process specialized for the particular task and dynamically adjusted to adapt to the learning process. Our design is a plug-in module compatible with various widely applied RL algorithms, including the DQN series, DDPG, TD3, and any possible variants developed based on them. Through extensive experiments on the Atari and MuJoCo benchmarks, we demonstrate LLM-Explorer's capability to enhance RL policy exploration, achieving an average performance improvement up to 37.27%. Our code is open-source at https://github.com/tsinghua-fib-lab/LLM-Explorer for reproducibility.

Section: Introduction
In recent decades, reinforcement learning (RL) has been proven to be a powerful tool for training smart agents in solving sequential decision-making problems [1,2]. The success of deep RL is especially noteworthy in tasks with high complexity, such as game playing [3,4,5,6], chip design [7], smart city governance [8,9,10,11,12,13,14], where deep RL agents now exhibit performance surpassing human professionals in more and more scenarios. In the training of RL agents, policy exploration plays an indispensable role, which allows the agents to sample a diverse range of actions and uncover better strategies that may not be immediately apparent. The explore-exploit trade-off is a critical aspect of reinforcement learning, where agents must balance exploring new possibilities to improve long-term rewards and exploiting known strategies to maximize immediate gains.
Various policy exploration approaches have been proposed in existing RL algorithms, including ϵ-greedy in DQN [15], Gaussian process noise in DDPG [16], etc. Despite their success, existing methods lack adaptability and flexibility. First, they are designed based on preset stochastic processes applied uniformly across all kinds of tasks without any environment-specific adaption, neglecting the unique characteristics of different environments that may influence policy exploration. Besides, the evolution of these stochastic processes during training tends to be simplistic, which typically merely involves a gradual decay in variance over time. As a result, these methods fail to flexibly adjust the policy exploration strategy based on the agent's real-time learning status, potentially reducing the effectiveness of policy exploration, especially in complex or non-stationary environments.
There exist several challenges in addressing these limitations. First, RL tasks span diverse environments, and the training process involves many action steps, during which the agent's learning status undergoes complex changes. Thus, relying on more fine-grained manual designs based on preset stochastic processes becomes increasingly impractical. Moreover, given its widespread success, there have been many well-established RL algorithms with proven performance, and how to incorporate improvements into these existing methods to enhance policy exploration while preserving their original strengths requires investigation.
Facing these challenges, we propose to enhance policy exploration in RL based on LLMs, namely LLM-Explorer. The emerging LLMs, with advanced analyzing and reasoning capabilities [17,18], demonstrate the potential to automatically analyze the environment characteristics and the agent's real-time learning status, thereby enabling more adaptive and flexible policy exploration. In LLM-Explorer, during the RL training process within a given environment, we periodically sample recent action-reward trajectories from the agent's experience and prompt the LLM to analyze the agent's current policy learning status based on the trajectories. The LLM then generates a tailored probability distribution that guides future policy exploration based on the agent's learning status and the specific characteristics of the environment. We update the probability distribution regularly, allowing it to dynamically adapt as the agent progresses through training and ensuring the exploration strategy evolves in response to changes in learning status. By doing so, we derive a specialized stochastic process from this dynamically updated distribution, which is uniquely suited to the environment. LLM-Explorer is designed to be a plug-in module that can be seamlessly integrated into existing RL algorithms by simply substituting the original preset stochastic process with the LLM-driven one without requiring any other architectural changes. Therefore, it is compatible with the DQN series [19,20,21,22], DDPG [16], TD3 [23], as well as any possible variants developed based on them, making it a versatile solution for various RL tasks, covering both discrete and continuous action spaces. We conduct extensive experiments on the Atari [24,25] and MuJoCo [26] benchmarks, and the results demonstrate the capability of LLM-Explorer.
In summary, the main contributions of this work include:
• We propose LLM-Explorer, a method that leverages LLMs to dynamically adjust the policy exploration during RL training in different tasks, which addresses the limitations of traditional policy exploration with preset stochastic processes.
• Our approach is designed as a plug-in module, allowing seamless integration with various widely applied RL algorithms, enabling enhanced exploration in both discrete and continuous action spaces without modifications to existing RL architectures.
• We conduct extensive experiments to evaluate our method in improving RL policy exploration across various tasks, attaining an average performance improvement up to 37.27%.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b15', 'b22', 'b23', 'b24', 'b25']

Section: Problem Formulation

this section cite: []

Section: Markov Decision Process (MDP)
Markov decision process (MDP) is the fundamental framework for reinforcement learning, where an agent solves the decision-making problems in interaction with a dynamic environment. Mathematically, an MDP is defined by a tuple (S, ρ, A, P, R) with S representing the state space, and ρ ∈ ∆(S) denoting the probability distribution of initial state, where ∆(S) is a collection of probability distribution over S. A denotes the action space, and when executing a specific action in a given state, P : S × A → ∆(S) and R : S × A → R are the state transition probability function and the single-step reward function, respectively. At time step t, the agent executes action a t ∈ A under the state of s t ∈ S, and then receives a reward of r t and experiences the state transition to s t+1 . The agent's goal in an MDP is to maximize its cumulative reward over time, which is the sum of discounted single-step rewards. This cumulative reward at time step t is formalized as G t = ∞ k=0 γ k r t+k , where γ is the discount factor that determines the importance of future rewards. To achieve this, the agent needs to balance exploiting known strategies and exploring unknown ones, where the former one means selecting the action with the largest estimated cumulative reward. In contrast, the latter requires trying other possibilities with randomness.
this section cite: []

Section: Large Language Models (LLMs)
Large language models are sophisticated neural networks with billions of parameters, which are mainly trained by predicting the probability of the next word in a sequence. Given {w 1 , w 2 , ..., w t-1 }, the models output w t to maximize the observation likelihood in the corpus as: T t=1 P (w t |w 1 , w 2 , ..., w t-1 ).
(1)
Over the past few years, LLMs have made significant progress, where notable examples include the GPT family [27,28,29], the Llama family [30,31], etc. These LLMs have exhibited strong capability across a wide range of natural language processing tasks, ranging from text generation and translation to summarization and question answering [17,32,33,34,35,36].
this section cite: ['b26', 'b27', 'b28', 'b29', 'b30', 'b16', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: Methods

this section cite: []

Section: Overview

this section cite: []

Section: LLM

this section cite: []

Section: Environment Agent
Reward Action (a) Learning status summary (b) Policy exploration strategy
In the last episode, the reward indicates that … the action strategy of moving while firing is effective… The agent should aim for optimal angles to fire…
this section cite: []

Section: Action Reward
Based on the analysis of the previous episode, the distribution of the action explorations for the next episode should reflect a balanced approach between offensive and defensive strategies… The action distribution is: {0:0.3,1:0.05,…}. In this paper, we propose to improve the policy exploration in RL based on LLMs, namely LLM-Explorer. As shown in Figure 1, our framework employs two LLMs that collaborate through natural language communication and guide the policy exploration through a structured process. First, we introduce the basic task description and sample action-reward trajectories of the agent from the previous episode, prompting the former LLM to summarize the learning status of the agent and recommend potential exploration strategies (Section 3.2). Then, we feed the obtained summary and suggestion to the second LLM, which subsequently generates a probability distribution for policy exploration in the next K episodes (Section 3.3). Here, K is the hyper-parameter representing the interval at which the probability distribution is updated. Our method is a plug-in design that simply modifies the probability distribution for policy exploration in existing RL algorithms with the LLM-driven one without any other architectural changes, making it compatible with a wide range of RL algorithms covering both discrete and continuous action spaces (Section 3.4).
this section cite: []

Section: Every step Every
𝐾 episode
this section cite: []

Section: Learning Status Summarizing
To effectively guide the policy exploration, we design the first LLM to summarize the learning status of the agent every K episode and provide suggestions on future exploration (Figure 1a). To achieve this, we first describe the basic elements of the task as {TaskDescription}, ensuring that the LLM is aware of the tasks' characteristics.
this section cite: []

Section: Task Description:
The task is a reinforcement learning problem where an agent {TaskDetails}. The action space is {ActionDetails}. The agent receives a reward of {RewardDetails}. The game ends when {EndConditions}. The goal is to {GoalDetails}.
Then, at each time of updating, we sample M actions uniformly from the latest episode, obtaining {ActionSequence}, where M stands for the sampling density. We also extract the total reward of the latest episode, obtaining {EpisodeReward}. Combining these, we design a tailored prompt for the first LLM, as formulated below:
Prompt 1: You are describing the last episode of the training process on a task. {TaskDescription}. In the last episode, the total reward is {EpisodeReward}, and the action sequence extracted at intervals is {ActionSequence}. Please analyze the data, generate a description, and provide possible strategy recommendations.
This prompt provides the necessary context for the LLM to summarize the information in the previous episode and extract insights into the agent's learning status. Additionally, it requires the LLM to offer potential strategy recommendations, aiming to provide more useful information for the upcoming policy exploration strategy generation.
It is worth mentioning that many RL tasks, such as the Atari benchmark, represent the environmental states by sequences of image frames, making it difficult for LLMs to process. In our design, we only sample the actions and rewards of the agent and exclude the states. The reason for this design is that our primary objective is to obtain an adaptive probability distribution for policy exploration based on the agent's learning status, rather than determining exact actions directly from the current state. Therefore, without requiring precise state information, LLMs can analyze what action patterns are most likely helpful for the current task from the task description and identify whether these action patterns have been adequately explored from the agent's historical behaviors and rewards. With such a design, our LLMs work with purely textual inputs, reducing computational consumption and ensuring compatibility with either multi-modal or text-only LLMs.
this section cite: []

Section: Policy Exploration Strategy Generation
To improve policy exploration, we design the second LLM in our framework to generate a probability distribution over the action space for future exploration (Figure 1b). This distribution is generated based on the first LLM's analysis regarding the learning status of the agent in the previous episode, as well as its suggestions for future policy exploration. We feed this information into the second LLM through the prompt structured as follows:
Prompt 2: You are determining the probability distribution for action exploration in reinforcement learning. {TaskDescription}. Here is a description of the situation in the previous episode: {Sum-mary&Suggestion}. Based on the above information, please analyze what kind of actions should be selected to better improve the task effectiveness. {OutputFormat}.
With this prompt, the LLM analyzes what actions should be explored more frequently and outputs a probability distribution for the next K episodes. This process enables the agent to prioritize actions that are more likely to improve the performance while also highlighting previously underexplored actions to discover new strategies. By periodically updating the strategy every K episode, we ensure the policy exploration evolves dynamically to adapt to the agent's learning progress.
this section cite: []

Section: Compatibility with Different RL Algorithms
In order to make the LLM-Explorer compatible with RL algorithms for both discrete action and continuous action spaces, we design different methods for generating probability distributions for each type of action space. For discrete action spaces, we directly generate a probability distribution for selecting each possible action, replacing the uniform distribution in the original algorithms. The {OutputFormat} for policy exploration strategy generation is:
Output Format (Discrete Action): Please output the distribution of the {ActionNum} action explorations for the next episode in decimal form. The format should be: {1: [probability], ...}.
For continuous action spaces, we generate a bias corresponding to the dimension of the action space and add it to the original symmetric Gaussian distribution centered around the origin, resulting in a biased probability distribution for action exploration with tendency. The {OutputFormat} is:
Output Format (Continuous Action): The approach is to add a Gaussian noise to each dimension of action, and you need to decide the bias of the Gaussian noise for each dimension. Please output the bias for each of the {ActionDim} dimensions of actions for action explorations in the next episode based on your analysis in decimal form. Your output format should be: {1: [bias], 2: ...}.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
We evaluate the performance of LLM-Explorer on the Atari [24,25] and MuJoCo [26] benchmarks, covering tasks with both discrete and continuous action spaces. In the main experiments, we use DQN [15] and DDPG [16] as the basis on Atari and MuJoCo, respectively, and plug our LLM-Explorer into them. We selected 15 out of 26 tasks from Atari and 5 out of 11 from MuJoCo, where raw DQN or DDPG algorithm can converge stably and obtain good rewards. In addition, we set the number of training steps to 100k-500k across different tasks based on how fast the reward increases when training the original DQN or DDPG algorithm. In the LLM-Explorer module, we use GPT-4o minifoot_3 as the core LLM and set the two key parameters in our design, namely action sampling density and exploration adjusting interval, as M = 100 and K = 1. In our deployment, we fix a set of hyper-parameters across all environments. For reproducibility, we provide specific values of all hyper-parameters in Appendix A and list detailed contents of the prompts in Appendix L.
this section cite: ['b23', 'b24', 'b25', 'b14', 'b15']

Section: Overall Performance
We train agents using the basis algorithm and algorithm with our LLM-Explorer module in the aforementioned environments, where in each environment, we repeat the training process with three different random seeds and average the results. We show the learning curves for each environment in Figure 2. On both the Atari and MuJoCo, LLM-Explorer improves the performance in most environments, verifying its ability to enhance the performance of the existing RL algorithm. Specifically, on the Atari tasks, LLM-Explorer reaches an increment of 37.27% and 13.84% on the mean and median human-normalized game score at the end of training [37,38], as summarized in Appendix B.
Also, we compare our method with two commonly used exploration methods without LLMs in Appendix C. The results illustrate the advantage of our LLM-Explorer design, highlighting the benefit of introducing LLMs to enhance policy exploration.
In our design, LLM-Explorer is a simple plug-in method that can be seamlessly integrated with a wide range of existing RL algorithms. To verify, besides the basic algorithms aforementioned, we selected another three widely applied variants of DQN (see Appendix D), including Double-Dueling DQN [19,20], Rainbow [21], and CURL [22]. We also include TD3 [23], a commonly used upgrade of DDPG. Respectively from the Atari and MuJoCo tasks, we selected three environments with relatively good training outcomes as representatives. In the selected tasks, we train agents with the original versions of the above RL algorithms, as well as the versions integrating our LLM-Explorer module. In each experiment, we repeat the training process with three different random seeds and average the results. We show the learning curves for the 12 experiments (4 algorithms×3 environments) in Figure 3. As the results illustrate, different RL algorithms exhibit diverse performance in different environments, while LLM-Explorer consistently improves their performance. This proves LLM-Explorer's compatibility with various RL algorithms, indicating its potential in tasks with either discrete or continuous action spaces.
this section cite: ['b36', 'b37', 'b18', 'b19', 'b20', 'b21', 'b22']

Section: Computational Overhead
In this section, we analyze the computational time and cost, which is crucial for assessing the practical application potential of our method. As shown in table 1, on average, each training run consumes approximately $1.3 in API calls and takes about 10 hours to complete. We consider this to be an acceptable and reasonable overhead. We also provided more detailed computational overhead comparison with baseline methods in Appendix F. In the framework work of LLM-Explorer, we utilize the LLMs with text-only prompts, leveraging their text-processing capability to derive smart policy exploration strategies. Instead of relying on some specific types or versions of LLMs, our design is a general framework that can work with various types of LLMs. To evaluate this, besides GPT-4o mini used above, we test several other LLMs that are most widely known, including GPT-4ofoot_4 , GPT-3.5foot_5 , Llama-3.1-405B, and Llama-3.1-70B 4 . We train agents with the original DQN algorithm and then integrate DQN with our LLM-Explorer method, where the latter is driven by each of these different LLMs. In each experiment, we repeat the training process with three different random seeds and average the results. We summarize the game scores obtained at the end of training in Table 2 and show the learning curves in these experiments in Appendix E. In the results, our method consistently improves the human-normalized score of the original algorithms despite the type of LLMs, indicating its strong compatibility with different LLMs.
this section cite: []

Section: Balance between Performance and Consumption
To facilitate the wide application of our method, it is important to understand the relationship between its performance and computational consumption. Since LLM-Explorer is a simple plug-in design that does not impact the original computational consumption in RL training, we mainly focus on its auxiliary consumption in utilizing LLMs.
this section cite: []

Section: Components in the LLM workflow.
To uncover the roles of key components in the LLM workflow, we conduct ablation experiments. In one experiment, we remove the summarize & suggestion mechanism and allow a single LLM to directly output a probability distribution for future policy exploration based on the {TaskDescription}, {ActionSequence}, and {EpisodeReward}. In another experiment, we retain the two-stage design of the LLM workflow but do not provide the {TaskDescription}, only informing the LLMs of the environment's name. In each experiment, we repeat the training process with three different random seeds and average the results. As shown in Table 3 and Appendix E, both ablations continue to improve the performance of the original DQN algorithm while significantly reducing the token consumption of LLM. However, the first ablation lacks sufficient analysis of the agent's learning status, making it less flexible for adjustment during the training process. The second ablation lacks sufficient environmental information, making it less adaptive to specific environments. As a result, neither of them performs as well as the full design.
Table 4: Performance of LLM-Explorer with different action sampling density M and exploration adjusting interval K. The human-norm scores (%) are recorded at the end of training and averaged across 3 random seeds. The underlines indicate improvements over the raw RL algorithm. The bold fonts are the best results. Task DQN DQN+LLM-Explorer M =100,K=1 M =50,K=1 M =100,K=2 M =200,K=1 Alien 0.26 0.59 0.51 0.38 0.83 Freeway 17.75 69.71 64.72 66.52 66.52 MsPacman 1.56 2.75 2.22 2.07 2.24
this section cite: []

Section: Setting of key parameters.
There are two key parameters within the workflow, namely action sampling density (M ) and exploration adjusting interval (K). By reducing sampling density, i.e., smaller M , or reducing the frequency of adjusting the exploration strategy, i.e., larger K, we can obviously reduce the token consumption of LLM. To evaluate the impact of these, we conduct experiments and show the results in Table 4 and Appendix E. As the results illustrate, LLM-Explorer with either smaller M or larger K keeps improving the performance of the original DQN algorithm. However, smaller M provides insufficient information about the agent's real-time learning status, and larger K limits adjustments to the exploration strategy. As a result, both of them are less capable of flexibly adapting the policy exploration to the training process, achieving worse performance than LLM-Explorer with the original settings of M and K. Moreover, we also analyze the impact of increasing the sampling density, i.e., larger M . As the results indicate, although increasing the token consumption of LLM, a larger M does not consistently improve the performance of LLM-Explorer. This may be because the original settings of M already provide sufficient information about the agent's real-time learning status. Therefore, further increasing the sampling density complicates the LLM's ability to analyze and summarize the data, which may hinder overall performance.
this section cite: []

Section: Size of LLMs.
As shown above, the proposed LLM-Explorer framework is compatible with different LLMs. To further explore the impact of LLM size, we test multiple Qwen2.5 models with varying sizes. As shown in Table 5, the performance degrades with smaller models, and models with ≤7B parameters generally fail to work. This suggests that, in general, larger and more capable models tend to perform better.
From these analyses, we demonstrate that the full design pipeline, properly configured values of M and K, and capable large LLMs are critical for achieving the best performance of LLM-Explorer. However, we also highlight the trade-offs between performance and computational consumption in LLM-Explorer. Therefore, for deployments with limited computational resources, it is possible to simplify the design of the LLM workflow or adjust M and K as above to reduce computational consumption while still maintaining certain performance improvements over the original RL algorithm. For deployments with sufficient computational resources, the full design with the original settings of M and K is the optimal choice. Also, fine-tuning a specialized LLM, for example, distilling the exploration strategy generation from a large LLM into a smaller one, would further reduce the minimum capable LLM size and save more computational cost.
this section cite: []

Section: Case Studies
To demonstrate the rationality in determining the policy exploration strategy with LLM-Explorer, we provide an intuitive case study in Figure 4 within the environment of the Freeway. In this environment, the goal is crossing the busy road safely, while the action space includes three items, namely no-ops, moving up, and moving down. In case 1, the previous action of the agent involves a large proportion of 'no ops', and the LLM in the stage of learning status summarizing points out its overly caution behavior that lacks clear direction. Subsequently, the latter LLM generates an exploration strategy that stresses moving up and down. In case 2, the previous action of the agent involves a large proportion of 'moving up', and the former LLM reveals that the current learning status of the agent is actively aiming to reach the other side of the highway. Based on this, the latter LLM generates an exploration strategy that further encourages 'moving up' to reach the goal while also adding a small proportion of 'moving down' to adjust position relative to traffic for safety. Such rational analyses enable our design to generate smart policy exploration strategies that are adaptive to specific environments and learning processes, enhancing the performance of various RL algorithms. 5 Related Works
this section cite: []

Section: Policy Exploration in RL
Plentiful approaches have been used in existing RL algorithms for policy exploration. One of the most basic methods is the ϵ-greedy strategy used in DQN [15], where with a probability of ϵ, the agent randomly samples an action from all possible actions rather than greedily exploiting the current best one. As an improvement of DQN, Noisy-DQN introduces noisy networks [39], which inject randomness directly into the action selection process, allowing for better policy exploration. Other methods utilize the randomness introduced by Gaussian distributions. For example, the actions are sampled from Gaussian distributions in PPO [40], and small Gaussian noises are added to the deterministic actions in DDPG [16]. Also, in some implementations of DDPG [41,42,43], the standard white Gaussian noise is replaced with an Ornstein-Uhlenbeck (OU) process with temporal correlation [44,45], leading to smoother and potentially more effective policy exploration. Moreover, extensive algorithms incorporate an entropy term in the reward function [46,47,48], encouraging more diverse action selections to enhance policy exploration. However, these methods are designed based on preset stochastic processes, which can neither adapt to specific environments nor be flexibly adjusted during the training process. In contrast, we design to dynamically generate a stochastic process by LLMs to guide policy exploration, which is adaptive and flexible.
this section cite: ['b14', 'b38', 'b39', 'b15', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47']

Section: Enhancing RL with LLMs
Many studies have explored the use of LLMs in enhancing the performance of RL [49]. First, a significant body of work focuses on leveraging LLMs to design reward functions based on the characteristics of the tasks, providing feedback for the agent's policy learning [50,51,52,53].
Additionally, other research investigates using LLMs to design state representation functions, offering more effective state inputs for the agents [54]. On a macro level, LLMs have been utilized to decompose complex tasks into sub-goals [50] or provide high-level instructions [55] to facilitate RL training. Moreover, LLMs are employed in human-AI coordination, enabling humans to specify the desired strategies for RL agents through natural language instructions [56]. Despite these works, it remains unknown how to leverage LLMs to enhance policy exploration in RL, where this paper manages to bridge such a knowledge gap. Furthermore, our plug-in module can integrate with a wide range of existing works using LLMs to enhance RL from various aspects, further benefiting their performance from the aspect of policy exploration.
this section cite: ['b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b49', 'b54', 'b55']

Section: Conclusions
In this paper, we propose a compatible plug-in design that utilizes LLMs to enhance policy exploration in RL algorithms. We design to use LLMs to analyze the agent's real-time learning status based on its action-reward trajectory and then periodically update the probability distribution for policy exploration. By doing so, we are able to adapt the policy exploration to any specific task and flexibly adjust it during the training process, only with the requirement of low-cost text-only prompts.
Through extensive experiments and in-depth analyses in various environments, we verify the validity of our design and illustrate its compatibility with a wide range of established RL algorithms, covering tasks with both discrete and continuous action spaces.
this section cite: []

Section: References
Ref_id:b0 Title: Reinforcement learning: An introduction Year: (2018)
Ref_id:b1 Title: An introduction to deep reinforcement learning Year: (2018)
Ref_id:b2 Title: Mastering the game of go without human knowledge Year: (2017)
Ref_id:b3 Title: Grandmaster level in starcraft ii using multi-agent reinforcement learning Year: (2019)
Ref_id:b4 Title: Dota 2 with large scale deep reinforcement learning Year: (2019)
Ref_id:b5 Title: Mastering atari games with limited data Year: (2021)
Ref_id:b6 Title: A graph placement methodology for fast chip design Year: (2021)
Ref_id:b7 Title: Hierarchical reinforcement learning for scarce medical resource allocation with imperfect information Year: (2021)
Ref_id:b8 Title: Reinforcement learning enhances the experts: Large-scale covid-19 vaccine allocation with multi-factor contact network Year: (2022)
Ref_id:b9 Title: Gat-mf: Graph attention mean field for very large scale multi-agent reinforcement learning Year: (2023)
Ref_id:b10 Title: Spatial planning of urban communities via deep reinforcement learning Year: (2023)
Ref_id:b11 Title: A survey of machine learning for urban decision making: Applications in planning, transportation, and healthcare Year: (2024)
Ref_id:b12 Title: Dyps: Dynamic parameter sharing in multi-agent reinforcement learning for spatio-temporal resource allocation Year: (2024)
Ref_id:b13 Title: Coopride: Cooperate all grids in city-scale ride-hailing dispatching with multi-agent reinforcement learning Year: (2025)
Ref_id:b14 Title: Human-level control through deep reinforcement learning Year: (2015)
Ref_id:b15 Title: Continuous control with deep reinforcement learning Year: (2016)
Ref_id:b16 Title: A survey of large language models Year: (2023)
Ref_id:b17 Title: Multimodal large language models: A survey Year: (2023)
Ref_id:b18 Title: Deep reinforcement learning with double q-learning Year: (2016)
Ref_id:b19 Title: Dueling network architectures for deep reinforcement learning Year: (2016)
Ref_id:b20 Title: Rainbow: Combining improvements in deep reinforcement learning Year: (2018)
Ref_id:b21 Title: Curl: Contrastive unsupervised representations for reinforcement learning Year: (2020)
Ref_id:b22 Title: Addressing function approximation error in actor-critic methods Year: (2018)
Ref_id:b23 Title: The arcade learning environment: An evaluation platform for general agents Year: (2013)
Ref_id:b24 Title: Modelbased reinforcement learning for atari Year: (2019)
Ref_id:b25 Title: Mujoco: A physics engine for model-based control Year: (2012)
Ref_id:b26 Title: Language models are few-shot learners Year: (2020)
Ref_id:b27 Title: A survey of gpt-3 family large language models including chatgpt and gpt-4 Year: (2023)
Ref_id:b28 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b29 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b30 Title: The llama 3 herd of models Year: (2024)
Ref_id:b31 Title: Stance detection with collaborative role-infused llm-based agents Year: (2024)
Ref_id:b32 Title: Hlm-cite: Hybrid language model workflow for text-based scientific citation prediction Year: (2024)
Ref_id:b33 Title: A survey on evaluation of large language models Year: (2024)
Ref_id:b34 Title: A survey on human-centric llms Year: (2024)
Ref_id:b35 Title: Towards large reasoning models: A survey of reinforced reasoning with large language models Year: (2025)
Ref_id:b36 Title: Barlowrl: Barlow twins for data-efficient reinforcement learning Year: (2024)
Ref_id:b37 Title: Image augmentation is all you need: Regularizing deep reinforcement learning from pixels Year: (2021)
Ref_id:b38 Title: Noisy networks for exploration Year: (2018)
Ref_id:b39 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b40 Title: Improved exploration through latent trajectory optimization in deep deterministic policy gradient Year: (2019)
Ref_id:b41 Title: Asynchronous episodic deep deterministic policy gradient: Toward continuous control in computationally complex environments Year: (2019)
Ref_id:b42 Title: Reinforcement learning based optimal control of batch processes using monte-carlo deep deterministic policy gradient with phase segmentation Year: (2021)
Ref_id:b43 Title: Exact numerical simulation of the ornstein-uhlenbeck process and its integral Year: (1996)
Ref_id:b44 Title: Ornstein-uhlenbeck processes and extensions. Handbook of financial time series Year: (2009)
Ref_id:b45 Title: Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b46 Title: Maximum entropy-regularized multi-goal reinforcement learning Year: (2019)
Ref_id:b47 Title: Maximum entropy gain exploration for long horizon multi-goal reinforcement learning Year: (2020)
Ref_id:b48 Title: Survey on large language model-enhanced reinforcement learning: Concept, taxonomy, and methods Year: (2024)
Ref_id:b49 Title: Augmenting autotelic agents with large language models Year: (2023)
Ref_id:b50 Title: Read and reap the rewards: Learning to play atari with the help of instruction manuals Year: (2024)
Ref_id:b51 Title: Self-refined large language model as automated reward function designer for deep reinforcement learning in robotics Year: (2023)
Ref_id:b52 Title: Text2reward: Reward shaping with language models for reinforcement learning Year: (2024)
Ref_id:b53 Title: Llm-empowered state representation for reinforcement learning Year: (2024)
Ref_id:b54 Title: Large language model as a policy teacher for training reinforcement learning agents Year: (2023)
Ref_id:b55 Title: Language instructed reinforcement learning for human-ai coordination Year: (2023)
Ref_id:b56 Title: Exploration by random network distillation Year: (2019-05-06)
Ref_id:b57 Title: Prioritized experience replay Year: (2016)
Ref_id:b58 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b59 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
