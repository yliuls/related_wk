Title: A Clean Slate for Offline RL
Abstract: Progress in offline reinforcement learning (RL) has been impeded by ambiguous problem definitions and entangled algorithmic designs, resulting in inconsistent implementations, insufficient ablations, and unfair evaluations. Although offline RL explicitly avoids environment interaction, prior methods frequently employ extensive, undocumented online evaluation for hyperparameter tuning, complicating method comparisons. Moreover, existing reference implementations differ significantly in boilerplate code, obscuring their core algorithmic contributions. We address these challenges by first introducing a rigorous taxonomy and a transparent evaluation procedure that explicitly quantifies online tuning budgets. To resolve opaque algorithmic design, we provide clean, minimalistic, single-file implementations of various model-free and model-based offline RL methods, significantly enhancing clarity and achieving substantial speed-ups. Leveraging these streamlined implementations, we propose Unifloral, a unified algorithm that encapsulates diverse prior approaches within a single, comprehensive hyperparameter space, enabling algorithm development in a shared hyperparameter space. Using Unifloral with our rigorous evaluation procedure, we develop two novel algorithms-TD3-AWR (model-free) and MoBRAC (model-based)-which substantially outperform established baselines. All code for this project can be found in our public codebase.

Section: Introduction
Offline reinforcement learning (RL)-the task of learning effective policies from pre-collected, static datasets-is critical for applying RL in real-world settings where online experimentation is expensive or risky. Despite significant interest [1][2][3][4][5], the field has struggled to converge on clear, actionable insights. Algorithms and methods proliferate rapidly but no broadly agreed-upon conclusions or standardized benchmarks have emerged [6]. This undermines both practical application and theoretical progress. In this work, we identify and address two primary problems that contribute to stagnation and confusion in offline RL research: an ambiguous problem setting and opaque algorithmic design.
Problem 1: Ambiguous Problem Setting Recent work in offline RL has lacked a rigorously articulated definition or standardized evaluation procedure. The broad mission statement, learning from a static dataset without direct environment interaction, is prone to misinterpretation that skews proposed methods towards impractical evaluation practices. Existing literature implicitly relaxes various definitions concerning critical details such as hyperparameter tuning allowances [4,7], the extent of post-deployment policy adaptation [8], and the specifics of evaluation procedures [9]. Consequently, comparisons between methods are confounded as each study might assume fundamentally different experimental conditions. While some approaches restrict tuning based on related dataset performance [10], most approaches extensively tune hyperparameters on the target environment [11,5,12]. Using the target environment to tune hyperparameters needs a large number of online evaluations, which is in conflict with the basic premise of offline RL.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b3', 'b6', 'b7', 'b8', 'b9', 'b10', 'b4', 'b11']

Section: Solution 1: A Novel Taxonomy and Evaluation Procedure
We first introduce a rigorous and explicit taxonomy of offline RL evaluation variants (Section 3.1) and specify the one we find to be implicitly adopted by most prior research. To facilitate consistent and transparent evaluation, we propose a rigorous procedure for this setting (Section 3.2) that evaluates algorithmic performance using a fixed hyperparameter range across multiple datasets. This procedure explicitly quantifies performance at various permissible levels of online hyperparameter tuning, i.e., interactions with the target environment, thus providing clarity about the practical deployment requirements of each method. To ensure ease of adoption and reproducibility, we release a straightforward software interface for performing this evaluation procedure, thereby empowering future work to evaluate offline RL algorithms robustly and transparently.
Problem 2: Opaque Algorithmic Design Offline RL methods are often presented as intricate bundles with intertwining algorithmic components, implementation-specific details, and unclear tuning procedures. Researchers compare proposed methods to baseline performance quoted directly from prior publications [6], inadvertently propagating these methodological issues. As a result, it is difficult to isolate the impact of individual methodological choices. Thus, the state-of-the-art remains ambiguous, with no method demonstrating uniformly strong performance across all datasets [13][14][15].
Solution 2: Consistent Reimplementations and a Unified Algorithm We first dissect the novel components of prior algorithms by defining a phylogenetic tree based on their compositional structure (Section 4.1). We use this representation to provide single-file reimplementations of a wide range of offline RL methods. These minimal implementations eliminate extraneous code differences and highlight fundamental components, as well as achieving average training speedups of 131.5× and 74.8× against OfflineRL-Kit [16] and CORL [17], two leading offline RL libraries. Furthermore, we propose a unified offline RL algorithm (Unifloral, Section 4.2) that integrates core components from various prior methods into one coherent framework. Crucially, Unifloral provides a single, unified hyperparameter space containing all of these algorithms.
Leveraging Unifloral with our evaluation procedure, we introduce two novel offline RL methods: a model-free approach (TD3-AWR, Section 5.1) and a model-based one (MoBRAC, Section 5.2). These methods demonstrate substantial performance improvements over established baselines, validating both our unified methodology and rigorous evaluation framework.
this section cite: ['b5', 'b12', 'b13', 'b14', 'b15', 'b16']

Section: Offline Training
Pre-deployment Post-deployment
this section cite: []

Section: Zero-shot
Example: Autonomous search and rescue
this section cite: []

Section: Preliminaries

this section cite: []

Section: Reinforcement Learning
We apply RL to a finite-horizon Markov Decision Process (MDP) defined by the tuple ⟨S 0 , S, A, D, R, T ⟩. Here S is the state space, A is the action space, and T is the horizon. D : S × A → ∆ (S) is the transition dynamics, defining how the state changes given a state and the action taken on that state. ∆ (S) is the set of all possible distributions over s. The scalar reward function is R : S × A → R. The environments in this paper are all fully observable as the Markov state is directly observed at each timestep.
A policy π maps a state in S to an action distribution over A. The policy is trained to maximize the expected return J π M for a given MDP M with trajectory length T : J π M := E a 0:T ∼π,s0∼S0,s 1:T ∼D T t=0 r t .
(1)
this section cite: []

Section: Offline Reinforcement Learning
Offline RL methods use a pre-collected dataset to optimize a target policy to maximize J π by, without online interactions in the environment. This dataset consists of transitions (s i , a i , r i , s i+1 , a i+1 ) for i = 1, . . . , N , where s i , s i+1 ∈ S, a i ∈ A, r i ∈ R are the current and next states, action, and reward, respectively. Here, initial states are drawn from the distribution s 0 ∼ S 0 and trajectories are gathered through a behaviour policy π b interacting with the environment. Since π b may be suboptimal, the resulting dataset might not contain sufficient coverage of the environment's state space to learn an effective policy.
An effective offline RL method must learn policies that generalize from this limited dataset to perform reliably when deployed in their environment. Typically, these methods require significant regularization to avoid overestimation bias. For model-free methods, this is commonly done with critic ensembles, where the minimum state value estimated by the ensemble is used for policy optimization. Model-based methods generalize by training a dynamics model D(s, a) to predict future states and rewards. This can be used to generate synthetic rollouts from the target policy, allowing for direct optimization of its performance.
this section cite: []

Section: Refining Evaluation in Offline RL
This section describes our taxonomy of offline RL, illustrated in Figure 1, which motivates our evaluation procedure in Figure 2. We also outline the procedure in detail and use it to analyze the performance of a set of model-free and model-based algorithms in multiple environments.
this section cite: []

Section: Variants of Offline RL
The goal of offline RL is to train an agent using solely offline data, with the objective of maximizing performance from deployment, i.e., the point where the agent is evaluated online. In this setting, deployment marks a strict separation between the offline training phase and the online evaluation phase. However, some methods may relax this strict separation in two ways. Firstly, pre-deployment interaction allows the agent to take limited interactions with the environment before deployment to improve post-deployment performance. For instance, to tune hyperparameters before selecting a policy for deployment. Secondly, post-deployment adaptation allows the agent to continue learning after deployment, and the performance metric includes all returns collected after deployment. Examples include dataset aggregation from multiple online episodes [18], selection from a set of policies trained offline [7,9], and fine-tuning a single policy [8], all of which can be performed both before and after deployment. While any combination of these is possible, we identify four key settings.
A Taxonomy of Offline RL
this section cite: ['b17', 'b6', 'b8', 'b7']

Section: ZERO-SHOT OFFLINE RL
• Train one policy offline, then deploy online with no further adaptation.
• No pre-deployment interaction, no post-deployment adaptation.
this section cite: []

Section: 2a. OFFLINE RL WITH PRE-DEPLOYMENT ONLINE POLICY SELECTION
• Train a set of policies offline, select the best policy based on N online evaluations before deployment.
• Limited pre-deployment interaction, no post-deployment adaptation.
this section cite: []

Section: 2b. OFFLINE RL WITH POST-DEPLOYMENT ONLINE POLICY SELECTION
• Train a set of policies offline, then deploy online, adaptively selecting a policy every episode based on online performance.
• No pre-deployment interaction, post-deployment adaptation via policy selection.
this section cite: []

Section: OFFLINE-TO-ONLINE RL
• Train one policy offline, then deploy online and fine-tune the policy on online data.
• Limited pre-deployment interaction and post-deployment adaptation via finetuning.
Many offline RL papers implicitly perform pre-deployment policy selection (Setting 2a), as they report final performance after extensive hyperparameter tuning involving online evaluation [11,5]. However, due to differences in the number of hyperparameters or computational resources, this tuning process varies in scope across studies. As a result, reported performances are often not directly comparable since they reflect not only algorithmic quality but also differences in tuning budgets. Furthermore, these procedures typically assume low-variance estimates of each policy's performance, determined by an indefinite number of online evaluations. This is rarely made explicit as hyperparameter tuning is often considered a technical detail and not part of the method, even though it can dramatically affect performance (Section 3.3).
this section cite: ['b10', 'b4']

Section: A Definition of Offline RL Methods
A method in offline RL consists of an algorithm and a fixed sampling range for each hyperparameter.
Finally, much prior work has blurred the line between algorithms and hyperparameters in offline RL, proposing different hyperparameter values or ranges for each task. This ambiguity enables the same "method" to have dramatically different behaviour across tasks, undermining the assumption of limited interactions by essentially proposing a different method for each task. To resolve this, we define an offline RL method to include a fixed hyperparameter range, which remains constant across datasets (see A Definition of Offline RL Methods).
this section cite: []

Section: Proposed Evaluation Procedure
We now propose a rigorous and practical evaluation procedure for offline RL with pre-deployment policy selection (Setting 2a), as it implicitly is the standard setting for evaluating offline RL methods (see Section 3.1). Our goal is to evaluate offline RL algorithms under a fixed budget of N predeployment environment interactions used for tuning. We measure this budget in terms of the number of evaluation episodes, reflecting practical deployment constraints where each online interaction can be costly. Whilst the tuning algorithm may be defined as part of the method, most research focuses on offline policy optimization prior to tuning. Therefore, we provide an upper confidence bound (UCB) bandit [19] in our implementation as the default tuning algorithm. Furthermore, to reflect real-world limitations, we assume that the expected return of each policy is not directly observable, with each pull from the bandit sampling a single episodic return from that policy's return distribution. This models the high-variance, sample-limited setting typical in real deployments, where evaluating a policy's performance requires interacting with the environment and yields only noisy, episodic feedback. The importance of this is demonstrated by the emergence of distractor policies, as discussed in Section 3.3.
In essence, our evaluation procedure repeatedly simulates hyperparameter tuning with a fixed online budget, using a bandit to select a single policy for final deployment. This procedure (Figure 2) has two steps: score collection and bandit evaluation. Step 1: Train Policies and Collect Scores Firstly, we collect a dataset of episodic evaluation scores from policies trained by the target algorithm. To do this, we sample P hyperparameter settings (with replacement and random seeds) from the range defined by the method, and then train P corresponding policies. These policies are evaluated online for a large number of episodes R and their episodic scores recorded. Following this, the policies may be discarded as only their episodic scores are required for bandit evaluation.
Step 2: Run Bootstrapped Tuning Bandit Using our collected episodic evaluation dataset, we then repeatedly simulate hyperparameter tuning to measure algorithm performance at different tuning budgets. This is performed by subsampling K policiesfoot_0 (i.e., their corresponding episodic scores) and running a multi-armed bandit over them. In this bandit, each arm corresponds to a policy, with each pull sampling one episode's return from the corresponding policy. At each number of pulls N , we evaluate the performance of the algorithm by selecting the policy estimated to have the highest return by the bandit, and taking its true average return. We repeat this process B times to obtain a bootstrapped estimate of algorithm performance.
Recommended Datasets It is essential to evaluate methods on a diverse distribution of tasks to ensure generality. Alarmingly, the majority of offline RL methods considered in this work were evaluated only on MuJoCo and Adroit tasks from the D4RL suite [20]. While computational budgets may be limited, we argue that they would be better spent considering a wider range of tasks and behaviour policies. In order to make environment selection consistent, we recommend starting with the following environments, where algorithms currently obtain non-trivial performance: hoppermedium, halfcheetah-medium-expert, and walker2d-medium-replay, as a representative subset of MuJoCo locomotion; pen-human, pen-cloned, and pen-expert, as algorithms often achieve zero or perfect performance on other Adroit environments; kitchen-mixed, maze2d-large, and antmaze-large-diverse, to provide diversity in the evaluated environments.
this section cite: ['b18', 'b19']

Section: Results
In Figure 3, we evaluate a range of prior algorithms (list in Appendix A). For this, we uniformly sample from the hyperparameter tuning ranges specified in each algorithm's original paper or the union of ranges when multiple are provided. Generally, an algorithm performs better if its curve is closer to the top left corner of a plot, representing strong performance after few online interactions. Prior work has typically reported performance after unlimited online tuning, which is the limit of the score with an increasing number of policy evaluations, i.e., the top right corner.
this section cite: []

Section: Inconsistent Algorithm Performance
No algorithm consistently performs well across all datasets. However, ReBRAC and IQL are competitive for the overall best performing algorithm, with ReBRAC achieving top performance at some number of evaluations on 5 out of 9 datasets and IQL on 4 out of 9 datasets. Even though both of these algorithms are worse than competing baselines on other datasets, we believe them to be the clearest baselines for future method development, as done in Section 5.1.
this section cite: []

Section: Overfit Model-Based Methods
The model-based algorithms we evaluate-MOPO, MOReL, and COMBO (Appendix A.2)-achieve notably poor performance on all non-locomotion datasets, ranking no higher than 6 th out of the 10 evaluated algorithms (and failing to beat BC) at any number of policy evaluations. While these results are surprising, we emphasize that our implementation successfully reproduces reference results with the specialized hyperparameters for each dataset (Appendix G). Instead, these results suggest that these methods are deeply overfit to the locomotion datasets they were originally evaluated on (Appendix C), providing a sobering reflection of the field.
0 20 40 60 80 100 120 Policy index 0 20 40 60 80 100 Score Distractor policies can achieve higher scores Distractor Policy Phenomenon While performance typically improves as more bandit arms are pulled, certain performance curves exhibit distinctive dips-temporary decreases in measured performance despite additional policy evaluations. To better understand this, we examine the ranked performance distribution of numerous ReBRAC policies trained on hopper-medium (Figure 4). This analysis reveals a notable cluster of policies that exhibit suboptimal average performance but possess a higher maximum performance compared to consistently better-performing policies. We refer to these anomalous policies as distractor policies.
To demonstrate their impact on evaluation, we simulate the initial phase of a bandit rollout over these policies, i.e., when the bandit enumerates all arms (Figure 9a). Over this phase, we observe a clear increase in the probability of preferring a distractor policy, explaining the initial decrease in evaluation performance. This phenomenon runs counter to the expectation that increasing policy evaluations would monotonically reduce estimator variance and underscores the need to directly consider environment interactions in evaluation, a crucial distinction from prior evaluation methodologies [9]. Further analysis of distractor policies is provided in Appendix D.
this section cite: ['b8']

Section: Elucidating Algorithm Design in Offline RL
In this section, we seek to simplify algorithm design in offline RL. Firstly, we present a genealogy of prior algorithms, using it to propose and implement a set of compositional reimplementations. Following this, we propose a unified algorithm, Unifloral, capable of expressing these methods-as well as any combination of their components-in a single hyperparameter space.
this section cite: []

Section: Disentangling Prior Methods

this section cite: []

Section: Figure 5:
Speed up from our JAX reimplementations -algorithms trained for 1M update steps on HalfCheetah-medium-expert using a single L40S GPU. Our library, Unifloral, is the fastest across the board. Full details can be found in Appendix F.
New offline RL methods are typically derived from preceding ones by adding or editing individual components of the agent's objective or architecture. Despite this, methods typically suffer from a range of unnecessary implementation differences, making it difficult for researchers to identify their contribution or fairly compare methods. Even in popular singlefile implementations, we observe significant code differences between "parent" and "child" algorithms, which should require only the individual components to be edited. This encourages researchers to compare entire algorithms rather than ablating components. We discuss this and how it informs our code philosophy in Appendix E.
As a solution, we provide single-file reimplementations of a range of existing model-free (BC, TD3-BC, ReBRAC, IQL, SAC-N, LB-SAC, EDAC, CQL, DT) and model-based (MOPO, MOReL, COMBO) methods. Our implementation has a number of advantages. Firstly, we focus on code clarity and minimal code edits between algorithms, leading to a dramatic reduction in code differences between algorithms. Secondly, we implement our algorithms in end-to-end compiled JAX, leading to major speed-ups against competing implementations (Figure 5). We believe these implementations will lead to better algorithm understanding and fairer evaluation, as well as enabling powerful experiments on low compute budgets. We verify the correctness of our reimplementations in Appendix G.
this section cite: []

Section: A Unified Hyperparameter Space for Offline RL
Implementation inconsistency and missing ablations are common flaws of offline RL research. The plethora of design decisions in each algorithm obfuscates evaluating how each feature contributes to the performance. To address this, we combine all components from a range of model-free and model-based algorithms (Appendix A) into a unified algorithm and single-file implementation, which we name Unifloral. We start by compiling a minimal subspace of components covering the model-free and model-based offline RL algorithms examined in this work (Appendix H). This has a range of hyperparameters in each of four broad design categories, which we identify from prior algorithms: model design, critic objective, actor objective, and dynamics modelling. A more detailed description of design category is in Appendix I.
this section cite: []

Section: Model Design
The choice of neural network architecture and optimizer is consistent across most offline RL research, with proposed algorithms commonly using multi-layer perceptrons and the Adam optimizer. However, the hyperparameters of these components commonly vary between algorithms. Regarding the model architecture, this includes the number of layers, layer width, and usage of observation and layer normalization. For optimization, this includes the learning rate (shared and actor-specific), learning rate schedule, discount factor, batch size, and Polyak averaging step size. The actor and critic networks can also have different structures, such the size of the critic ensemble and whether the policy stochasticity.
this section cite: []

Section: Critic Objective
The core contribution of offline RL research is often a novel critic objective [12,13]. However, many of the components in the proposed objectives are shared with prior work. We define the critic objective as the weighted sum of those components, or a selection between them if mutually exclusive, in order to include all referenced methods (except CQL, which we omit due to its substandard performance and high complexity). More detail in Appendix I.1.
Actor Objective We define the unified actor loss as the weighted sum of three terms:
L actor = β q • L q + β BC • L BC -β H • H(π(•|s t )).(2)
This consists of q loss L q , behaviour cloning loss L BC , and policy entropy H(•), with coefficients β q , β BC , β H ∈ R controlling the weight of these terms. More detail in Appendix I.2.
this section cite: ['b11', 'b12']

Section: Dynamics Modelling
We include optional dynamics model training and sampling, broadening Unifloral's coverage to include model-based methods. As is standard, we use an ensemble of dynamics models Dθ = { D1 θ , D2 θ , ..., DM θ }, where each Di θ is trained to predict state transitions and rewards. Following MOPO, we penalize the agent for going to states where the ensemble disagreement is high as measured by the standard deviation of the model's predictions. More in Appendix I.3.
this section cite: []

Section: Novel Methods Research with Unifloral
Our unified algorithm and hyperparameter space enable researchers to combine different components and search through algorithm designs by only modifying the configuration of the unified implementation. To demonstrate the avenues our work opens up and encourage further research, we provide two "mini-papers" completed entirely by specifying configurations of the unified implementation, without any code changes. We examine a model-free and a model-based improvement.
this section cite: []

Section: TD3 with Advantage Weighted Regression
Hypothesis In Section 3.3, we show that one of two methods consistently outperformed existing baselines: ReBRAC [5] and IQL [21]. ReBRAC is derived from TD3-BC, meaning it optimizes its actor using TD3 value loss in combination with a BC loss term for regularization. In contrast, IQL uses only a BC loss but performs advantage weighted regression (AWR) by weighting the BC loss of each action by its estimated advantage. We hypothesise that substituting the BC term in ReBRAC with AWR, a method we name TD3-AWR, would combine the strengths of these methods and lead to improved performance overall.
Evaluation We define TD3-AWR in Unifloral by using the AWR hyperparameters from IQL and the ReBRAC hyperparameters elsewhere. In Figure 6, we show that TD3-AWR's performance curve strictly dominates ReBRAC on 6 out of 9 datasets and is dominated by ReBRAC in only 1. Interestingly, TD3-AWR achieves superior performance to ReBRAC under few policy evaluationssuch as in halfcheetah-medium-expert and pen-expert-despite searching over a wider range of hyperparameters. Similarly, TD3-AWR strictly dominates IQL on 7 datasets, thereby outperforming both of its source algorithms.
this section cite: ['b4', 'b20']

Section: Improving Policy Optimization for Model-Based Offline RL Hypothesis
In Section 3.3, we demonstrate the poor performance of model-based methods on non-locomotion environments. Whilst this is partially due to overfit hyperparameters, the design space of policy optimizers in model-based methods is underexplored, with all considered methods using SAC-N or CQL (Figure 5). Given the performance improvements from recent methods, we posit that these methods would be more competitive with an alternative policy optimizer. We therefore propose using ReBRAC with synthetic rollouts generated from a MOPO world model, which we name Model-based Behaviour Regularized Actor-Critic, or MoBRAC.
Evaluation We implement MoBRAC in Unifloral, using the MOPO hyperparameters for dynamics model training and sampling, then using the ReBRAC hyperparameters elsewhere. Figure 7 shows how MoBRAC outperforms other model-based methods for all datasets, except for MOPO in maze2d-large-v1. Under a transparent evaluation budget, we find that MoBRAC outperforms the other model-based methods in 6 out of 9 datasets and is tied with MOPO for 3 others (Appendix K).
this section cite: []

Section: Related Work
Our work builds upon several foundational aspects of offline RL, including evaluation strategies, open-source implementations, and algorithmic unification. Existing evaluation regimes primarily address hyperparameter tuning either through limited online interactions [8,9] or by estimating policy performance offline [4,10,22]. In contrast, our approach introduces an evaluation procedure that requires neither reference policies nor additional hyperparameters, offering broader applicability across the entire D4RL benchmark suite. Furthermore, our single-file implementations draw inspiration from projects such as CORL [17,23] and CleanRL [24], whilst our unified algorithm, Unifloral, is informed by prior unification attempts [25,26,15,27]. For a comprehensive review, see Appendix L.
this section cite: ['b7', 'b8', 'b3', 'b9', 'b21', 'b16', 'b22', 'b23', 'b24', 'b25', 'b14', 'b26']

Section: Conclusion
In this work, we addressed critical challenges in problem formulation, evaluation, and algorithm unification in offline RL. We introduced a taxonomy that clearly distinguishes between offline RL variants-spanning zero-shot deployment to approaches with limited pre-deployment tuning or post-deployment adaptation. This categorization exposes the hidden online interactions, such as hyperparameter tuning, that have long confounded fair evaluation and reproducibility. To overcome these issues, we proposed a rigorous evaluation procedure that transparently quantifies the cost of online interactions via noisy, single-episode feedback. Additionally, by dissecting components of existing offline RL algorithms, we developed Unifloral, a novel unified offline RL algorithm that combines improvements of many previous methods, enabling seamless ablation of algorithmic components. We demonstrate this with two novel algorithms inside Unifloral, TD3-AWR and MoBRAC, which integrate the strengths of existing methods to achieve superior performance over a wide range of tasks. Collectively, our contributions set a new standard for addressing ambiguity in offline RL, promoting rigorous evaluation, and driving reproducible, impactful research in the field.
this section cite: []

Section: References
Ref_id:b0 Title: Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems Year: (2020-11)
Ref_id:b1 Title: MOPO: Model-based Offline Policy Optimization Year: (2020-11)
Ref_id:b2 Title: A Minimalist Approach to Offline Reinforcement Learning Year: (2021-12)
Ref_id:b3 Title: Hyperparameter Selection for Offline Reinforcement Learning Year: (2020-07)
Ref_id:b4 Title: Revisiting the Minimalist Approach to Offline Reinforcement Learning Year: (2023-10)
Ref_id:b5 Title: A survey on offline reinforcement learning: Taxonomy, review, and open problems Year: (2023)
Ref_id:b6 Title: Active offline policy selection Year: (2021)
Ref_id:b7 Title: Deployment-efficient reinforcement learning via model-based offline optimization Year: (2020)
Ref_id:b8 Title: Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters Year: (2022-06)
Ref_id:b9 Title: A strong baseline for batch imitation learning Year: (2023)
Ref_id:b10 Title: MOReL : Model-Based Offline Reinforcement Learning Year: (2021-03)
Ref_id:b11 Title: Conservative Q-Learning for Offline Reinforcement Learning Year: (2020-08)
Ref_id:b12 Title: Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble Year: (2021-10)
Ref_id:b13 Title: COMBO: Conservative Offline Model-Based Policy Optimization Year: (2022-01)
Ref_id:b14 Title: Revisiting Design Choices in Offline Model-Based Reinforcement Learning Year: (2022-03)
Ref_id:b15 Title: Offlinerl-kit: An elegant pytorch offline reinforcement learning library Year: (2023)
Ref_id:b16 Title: CORL: Research-oriented deep offline reinforcement learning library Year: (2022)
Ref_id:b17 Title: A reduction of imitation learning and structured prediction to no-regret online learning Year: (2011)
Ref_id:b18 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002)
Ref_id:b19 Title: D4rl: Datasets for deep data-driven reinforcement learning Year: (2020)
Ref_id:b20 Title: Offline Reinforcement Learning with Implicit Q-Learning Year: (2021-10)
Ref_id:b21 Title: A Workflow for Offline Model-Free Robotic Reinforcement Learning Year: (2021-09)
Ref_id:b22 Title: Jax-corl: Clean sigle-file implementations of offline rl algorithms in jax Year: (2024)
Ref_id:b23 Title: Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms Year: (2022)
Ref_id:b24 Title: Rainbow: Combining improvements in deep reinforcement learning Year: (2018)
Ref_id:b25 Title: Combining improvements in policy optimization Year: (2021)
Ref_id:b26 Title: Dual rl: Unification and new methods for reinforcement and imitation learning Year: (2023)
Ref_id:b27 Title: Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b28 Title: Addressing Function Approximation Error in Actor-Critic Methods Year: (2018-07)
Ref_id:b29 Title: Advantage-weighted regression: Simple and scalable off-policy reinforcement learning Year: (2019)
Ref_id:b30 Title: Alvinn: An autonomous land vehicle in a neural network Year: (1988)
Ref_id:b31 Title: Behavior regularized offline reinforcement learning Year: (2019)
Ref_id:b32 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b33 Title: The Generalization Gap in Offline Reinforcement Learning Year: (2024-03)
Ref_id:b34 Title: Discovered policy optimisation Year: (2022)
Ref_id:b35 Title: No more pesky hyperparameters: Offline hyperparameter tuning for rl Year: (2022)
Ref_id:b36 Title: Takuma Seno. d3rlpy: An offline deep reinforcement library Year: (2020)
Ref_id:b37 Title: Stable-baselines3: Reliable reinforcement learning implementations Year: (2021)
Ref_id:b38 Title: Spinning Up in Deep Reinforcement Learning Year: (2018)
Ref_id:b39 Title:  Year: (2024)
