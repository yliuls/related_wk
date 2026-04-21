Title: DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation
Abstract: Dexterous manipulation is critical for advancing robot capabilities in real-world applications, yet diverse and high-quality datasets remain scarce. Existing data collection methods either rely on human teleoperation or require significant human engineering, or generate data with limited diversity, which restricts their scalability and generalization. In this paper, we introduce DexFlyWheel, a scalable data generation framework that employs a self-improving cycle to continuously enrich data diversity. Starting from efficient seed demonstrations warmup, DexFlyWheel expands the dataset through iterative cycles. Each cycle follows a closed-loop pipeline that integrates Imitation Learning (IL), residual Reinforcement Learning (RL), rollout trajectory collection, and data augmentation. Specifically, IL extracts human-like behaviors from demonstrations, and residual RL enhances 0 † Corresponding author * This work was completed during an internship at PsiBot. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).policy generalization. The learned policy is then used to generate trajectories in simulation, which are further augmented across diverse environments and spatial configurations before being fed back into the next cycle. Over successive iterations, a self-improving data flywheel effect emerges, producing datasets that cover diverse scenarios and thereby scaling policy performance. Experimental results demonstrate that DexFlyWheel generates over 2,000 diverse demonstrations across four challenging tasks. Policies trained on our dataset achieve an average success rate of 81.9% on the challenge test sets and successfully transfer to the real world through digital twin, achieving a 78.3% success rate on dual-arm lift tasks.

Section: Introduction
Learning from Demonstration (LfD) [1] has become increasingly prevalent in robotics. Recent works has shown that training large models with extensive datasets can achieve more challenging tasks and better generalization [2][3][4][5][6][7][8]. In dexterous manipulation particularly [9][10][11][12], the higher degrees of freedom and richer contact interactions demand larger, more diverse and higher-quality datasets. However, collecting such datasets remains a considerable bottleneck. Human teleoperation approaches require significant human effort and typically constrain data collection to laboratory settings, which limits the scalability of data collection. While portable motion capture devices [13] can collect data in-the-wild, they still require substantial human involvement and suffer from cross-embodiment gap. Recently, simulation has emerged as a promising solution to address data challenges in robotics [14][15][16][17][18][19]. It offers numerous advantages: parallel data collection at scale, easy modification of robot embodiments and sensor configurations, and domain randomization for data augmentation. However, existing simulation-based approaches, such as optimization or heuristic planning methods [20], LLM-driven methods [19,[21][22][23], and purely RL-based methods [18,[24][25][26], struggle with the high-dimensional complexity of dexterous manipulation and often produce low-quality trajectories.
Given these simulation challenges, researchers have begun exploring the teleoperation with replaymechanism [14,27], where humans teleoperate simulated robots to collect training data and then use spatial transformations to synthesize new trajectories. Although this approach captures relatively high-quality data with simulation-based augmentation, it has several fundamental critical limitations:
(1) Inability to Explore Novel Manipulation Strategies. By replaying human demonstrations, these methods confine exploration to existing behaviors, restricting data to the scope of the original demonstrations and hindering generalization to novel scenarios. (2) Insufficient Data Diversity. Since these methods primarily apply spatial augmentations, the generated datasets often exhibit insufficient variability in object geometries and environments, inherently constraining the generalization of learned policies. These limitations motivate us to rethink the role of human demonstrations in data generation pipelines. We observe that manipulating different objects typically induces only minor changes in the manipulation trajectories. This suggests regarding human demonstrations not merely as replay data, but as strong behavioral priors that can guide exploration in novel scenarios.
Building on this insight, we propose DexFlyWheel, a scalable and self-improving data generation paradigm for dexterous manipulation. Our framework features two key design: IL + residual RL for Human-like and Diverse Data Generation. DexFlyWheel combines IL to learn human-like behaviors from demonstrations with residual RL to adapt these priors to novel scenarios, particularly when manipulating different objects, thereby generating diverse and human-like data. A Dexterous Manipulation Data Flywheel. Inspired by iterative self-improvement in LLMs [28,29], we design a data flywheel for dexterous manipulation. Specifically, IL and residual RL are combined with policy rollouts and data augmentation to form a self-improving cycle. At each iteration, the policy generates trajectories, which are then augmented in progressively more diverse scenarios and subsequently fed into the next iteration. This cycle produces a flywheel effect, progressively expanding data diversity, enhancing policy generalization, and evolving into a robust, generalizable data generation agent.
In summary, our main contributions include:
• We propose DexFlyWheel, a scalable and self-improving data generation framework for dexterous manipulation. By combining IL with residual RL and leveraging data augmentation within a self-improving flywheel mechanism, our framework efficiently produces diverse, high-quality demonstrations while preserving human-like behavior patterns. This alleviates the scarcity of dexterous manipulation data and provides a solid foundation for training generalizable policies.
• We demonstrate the effectiveness of our framework on four dexterous manipulation tasks. Starting from a single human demonstration per task, DexFlyWheel generates over 2,000 successful demonstrations across 500+ diverse scenarios. The flywheel effect of our framework progressively expands data diversity, enabling it to significantly outperform baseline data generation methods.
• We validate that policies trained on our generated data achieve an average success rate of 81.9% on challenging test sets, significantly outperforming policies trained with baselines. Furthermore, our policies transfer to a real-world dual-arm robot system via digital twin, achieving a 78.3% success rate on the dual-arm lift task and a 63.3% success rate on the dual-arm handover task.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b18', 'b20', 'b21', 'b22', 'b17', 'b23', 'b24', 'b25', 'b13', 'b26', 'b27', 'b28']

Section: Related Work
Dexterous Manipulation. Dexterous manipulation with multi-fingered robotic hands remains a significant challenge in robotics [30][31][32][33][34][35], largely constrained by high-quality demonstration data scarcity. While the prevailing approach employs reinforcement learning to develop manipulation skills, this method frequently encounters efficiency limitations and exploration challenges [36][37][38][39].
Researchers have explored human video demonstrations [40][41][42][43][44][45][46][47], but morphological differences between human and robotic hands create substantial transfer barriers. Human teleoperation has emerged as a promising alternative for collecting expert trajectories for imitation learning [14,27,[48][49][50][51], effectively capturing expert actions in native robot morphology. Nevertheless, existing approaches still struggle with data collection efficiency or require extensive human engineering, emphasizing the need for high-quality dexterous manipulation datasets.
this section cite: ['b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b13', 'b26', 'b47', 'b48', 'b49', 'b50']

Section: Robotic Data Generation in Simulation.
Current approaches for collecting robotic demonstrations in simulation face significant limitations when applied to dexterous manipulation. Motion planningbased methods, while effective for gripper-based systems [15,20,24,[52][53][54], struggle with the high-dimensional action space and complex contact dynamics of multi-fingered manipulation. LLMsdriven methods [19,[21][22][23] can generate high-level command, but they demonstrate limitations when confronted with high-degree-of-freedom dexterous hands, unable to provide the fine-grained guidance necessary for coordinated finger-level control. Other pipelines are designed specifically for grasping [52], and thus do not generalize well to more complex dexterous tasks. RL-based methods have also been widely adopted [36, 24-26, 18, 53, 55, 56], yet purely RL-trained policies often exhibit non-human-like behaviors, leading to less robust manipulation and increased sim-to-real transfer challenges. Moreover, RL faces exploration difficulties and relies heavily on reward engineering, which is particularly acute in dexterous manipulation. Replay-based methods [14,27,57] attempt to edit existing demonstrations to new scenarios but face fundamental scalability constraints, as they merely implement spatial transformations of recorded trajectories without the ability to explore novel manipulation strategies beyond the original demonstrations. For example, when an object's geometry changes significantly-e.g., from a sphere to a cuboid-replay-based methods struggle to adapt finger trajectories. These collective limitations underscore the critical need for more efficient and scalable methods to generate diverse, high-quality dexterous manipulation data in simulation.
this section cite: ['b14', 'b19', 'b23', 'b51', 'b52', 'b53', 'b18', 'b20', 'b21', 'b22', 'b51', 'b13', 'b26', 'b56']

Section: Task Formulation
To address the challenge of generating high-quality synthetic data for robotic manipulation tasks, we train policy models for each manipulation task in simulation environments and use these policies to collect demostrations. We formulate each manipulation task as a Markov Decision Process (MDP) M = (S S S, A A A, π, T , R, γ, ρ, G), where S S S is the state space, A A A is the action space, π is the agent's policy, T (s t+1 |s t , a t ) is the transition distribution, R is the reward function, γ is the discount factor, and ρ is the initial state distribution. The policy π conditions on the current state s t , and generates robot action distributions a t to maximize the likelihood between the future object states (s t+1 , s t+2 , . . . , s t+T ).
this section cite: []

Section: Method
In this section, we begin with an overview of the DexFlyWheel architecture (Section 4.1) and then detail the two-stage data generation pipeline (Sections 4.2 and 4.3). Together, these components enable scalable collection of diverse and high-quality dexterous manipulation demonstrations.
this section cite: []

Section: Iteration 1 Iteration n

this section cite: []

Section: Warm-up Stage

this section cite: []

Section: Dataset

this section cite: []

Section: Residual Policy Training

this section cite: []

Section: Data Filter
Env.
𝝅 𝒓𝒆𝒔 𝐢 𝝅 𝒃𝒂𝒔𝒆
this section cite: []

Section: 𝐢

this section cite: []

Section: Dataset Generation

this section cite: []

Section: 𝒜 𝐸𝑃

this section cite: []

Section: Trajectory Filter

this section cite: []

Section: Rollout Trajectory Collection
𝝅 𝒄𝒐𝒏𝒃𝒊𝒏𝒆𝒅 𝒊 = 𝜋 𝑏𝑎𝑠𝑒 𝑖 + 𝜋 𝑟𝑒𝑠 𝑖 Combined Policy Base Policy Residual Policy Data Augmentation Rollout Datasets 𝑑 𝑠𝑒𝑒𝑑
this section cite: []

Section: Data Collection via Teleoperation

this section cite: []

Section: Data Augmentation

this section cite: []

Section: 𝐷 1
Base Policy 𝝅 𝒃𝒂𝒔𝒆 𝒊 𝑫 𝒊 Dataset
this section cite: []

Section: Base Policy Training
𝜋 𝑏𝑎𝑠𝑒 i 𝐷 Self-improving Data FlyWheel Stage 𝒂 𝒕 𝒔 𝒕+𝟏 , 𝒓 𝒕+𝟏 Image Robot state Object state Warm-up Dataset
this section cite: []

Section: Overview
DexFlyWheel aims to generate diverse and high-quality data across various scenario configurations, providing broad coverage of objects, environments, and spatial variations, while only starting with minimal human demonstrations. Figure 2 illustrates the overall architecture of our framework.
this section cite: []

Section: Warm-up stage.
A single human demonstration d seed is collected via a VR-based teleoperation system. This seed demonstration is then processed using a multi-dimensional data augmentation module A EP . Given the human demonstration d seed , the augmentor generates new demonstrations with diverse environment and spatial variations to produce the initial dataset D 1 .
this section cite: []

Section: Self-improving Data FlyWheel stage.
We design a data flywheel mechanism to progressively enhance both data diversity and policy performance. This stage comprises multiple iterations i = {1, 2, . . . , n -1}, at each iteration i, the following steps are executed: (1) An imitation learning policy π i base is trained on dataset D i (with D 1 used when i = 1). (2) To improve generalization to novel objects, a residual reinforcement learning policy π i res is trained on top of the frozen π i base , yielding a combined policy π i combined = π i base +π i res .
(3) The combined policy is deployed in simulation to generate demonstrations under various object configurations, forming a high-quality rollout dataset D i O . (4) Finally, D i O is further augmented by A EP with environment and spatial variations to produce the dataset D i+1 used in the next iteration.
this section cite: []

Section: Warm-up Stage
The first stage aims to generate an initial dataset D 1 via data augmentation module A EP , starting from a single human demonstration d seed . This warm-up stage including two operations: Data Collection via Teleoperation. To bootstrap the framework with high-quality seed data, we design a VR-based teleoperation system implemented in simulation using Apple Vision Pro [50] to accurately track human hand, wrist, and head poses. Since large-scale data collection requires heavily human effort, we only need a single demonstration, denoted as d seed . This demonstration serves as the sole seed for subsequent data generation. This makes our method highly efficient in terms of human resources while maintaining the quality and diversity of the generated data.
this section cite: ['b49']

Section: Data Augmentation.
To scale and diversify our dataset, we introduce data augmentation module A EP , which builds upon the MimicGen framework [27] and extends it to support multi-dimensional data augmentation across various environments and spatial configurations. It can efficiently augment source dataset D through trajectory editing and simulation domain randomization. A EP takes D and augmented scenario configurations C aug as input, and outputs the augmented dataset D ′ . In the warmup phase, this process is applied to the seed demonstration: D 1 = A EP (d seed ; C 1 ). Through this warmup phase, we establish a foundation of diverse manipulation datasets that serves as the starting point for our iterative data flywheel mechanism.
this section cite: ['b26']

Section: Self-improving Data FlyWheel Stage
The second stage implements a closed-loop data flywheel mechanism that iteratively expands data diversity across objects, environments and spatial generalization dimensions. At each iteration i ∈ {1, 2, ..., n -1}, the data flywheel performs four key operations:
Base Policy Training. Given the dataset D i from the previous iteration, we employ diffusionbased policy [58] as the base policy π base to learning dexterous manipulation skill, obtaining a strong base policy for subsequent modules. At each step t, the base policy π i base takes the state s t = {s vis t , s obj t , s prop t } as inputs, where s vis t represents visual input from camera, s obj t contains object state information including 6D pose (position and orientation) and velocities, and s prop t includes robot proprioception data consisting of joint positions, velocities, and end-effector poses. The policy outputs a sequence of robots actions (a t , a t+1 , . . . , a t+H ), where H represents the prediction horizon, and each action a t consists of the end-effector 6D pose and target joint angles. Implementation details of the policy parameters are provided in Appendix A.1.
this section cite: ['b57']

Section: Residual Policy Training.
Generalizing to novel objects remains a key challenge in imitation learning for robotic manipulation, which often suffers from limited data. We observe that manipulating different objects induces only small changes in the manipulation trajectories 1 , suggesting that a well-initialized policy can only require fine-grained adjustments to adapt to new objects. Based on this observation, we propose a residual reinforcement learning framework that builds upon the base policy. Specifically, we train a residual policy π i res that takes object state s obj t and robot proprioception s prop t as inputs and generates correction actions △a = (△a t , . . . , △a t+H ). These corrections, scaled by α, are added to the base policy actions to form the combined policy π i combined = π i base + α • π i res , where ãt = a t + α • △a t at each timestep. This approach allows the residual policy start from a reasonable robot actions from π i base and focus on learning the fine-grained refinements to generalize objects. Implementation details are in Appendix A.2.
To stabilize exploration, we employ the progressive schedule from [59], defining the combined policy during training as:
π combined (s) = π base (s) + α • π res (s) with probability ϵ π base (s) with probability 1 -ϵ(1)
where ϵ serves as a mixing coefficient that linearly increases from 0 to 1 over T steps, gradually shifting control from the base to the residual policy.
this section cite: ['b58']

Section: Rollout Trajectory Collection.
In this module, we employ the frozen combined policy π i combined = π i base + α • π i res to perform rollouts in simulation under randomized object configurations:
D i O = {d j = {(s t , a t )} T -1 t=0 |d j ∼ π i combined } K j=1
, where we collect K high-quality trajectories by filtering based on task success. This rollout strategy achieves robust object generalization, while geometry-unaware trajectory editing methods fail to adapt [14].
this section cite: ['b13']

Section: Data Augmentation.
In this module, we employ the previously introduced data augmentation module A EP to efficiently augment data in various environment and spatial configurations. Taking the dataset D i O and augmented scenario configurations C i+1 aug as input, the module produces an expanded dataset: D i+1 = A EP (D i O ; C i+1 aug ). The expanded dataset D i+1 is used to train an improved base policy, which serves as the foundation for the next iteration.
this section cite: []

Section: Experiments
The experiments are designed to answer the following research questions: Q1: Data Flywheel Effect. How does DexFlyWheel exhibit a self-improving data flywheel in dexterous manipulation, continuously enhancing data diversity and policy generalization? (Section 5.2) Q2: Policy Performance and Data Generation Efficiency. How does DexFlyWheel compare with baselines in policy performance, data generation robustness, and time efficiency (Section 5.3)?
Q3: Component Contribution. How does each component of DexFlyWheel contribute quantitatively to the overall system performance? (Section 5.4) Q4: Real-World Deployment. How does DexFlyWheel enable the real-world deployment of bimanual dexterous robot systems? (Section 5.5)
this section cite: []

Section: Experimental Setup
Tasks and Robots. We evaluate our framework across four dexterous manipulation tasks on both single-arm and dual-arm robot settings: (1) Grasp (single-arm): The robot must grasp the target object and lift it to a height greater than 0.2 meters from the tabletop. (2) Pour (single-arm):
The robot manipulates a source container to transfer its contents into a target container, requiring controlled pouring of the contained objects from one receptacle to another. (3) Lift (dual-arm): The robot performs collaborative manipulation using its two arms to synchronously lift an object to a minimum height of 15 cm. (4) Handover (dual-arm): The robot performs an intra-agent handover by transferring an object from one hand to the other through a stable, coordinated motion.
For the single-arm Grasp and Pour tasks, we use a Franka Emika Panda robot arm equipped with an Inspire robotic hand. For the dual-arm Lift and Handover tasks, we use a 7-DoF Real-Man RM75-6F arm paired with a 6-DoF PsiBot G0-R hand. Dual-arm robot is equipped with a RealSense D435 camera mounted on its head, which provides a first-person perspective.
this section cite: []

Section: Data Collection and Environment.
For each manipulation task, we collected only a single demonstration trajectory using VR teleoperation as our minimal seed data. To ensure the generation of high-quality data, we employed OmniGibson [60] as our simulation platform, leveraging its realistic rendering to generates high-quality data. We prepared 80 distinct objects across various categories and 12 different environments with varying lighting conditions, tabletop appearances. we set the number of iterations to i = {1, 2, 3}. For each task, we generated 20, 100, and 500 trajectories in the three iterations, respectively. Simuation setup is visualized in Figure 3. More detailed data collection and environment setup are provided in Appendix A. 3 Evaluation Design. We evaluate our method using two criteria: (1) data diversity: the number of object variations O, environment variations E and spatial variations P in our generated dataset D i in each iteration, and the total number of scenarios (O × E × P ) that our pipeline can cover;
(2) generalization performance: the Success Rate (SR) of task execution when policies trained on our generated datasets D i . It is calculated as the ratio of successful task completion to the total attempts. Specifically, we construct two types of test sets. First, the multi-factor generalization test set (T OEP ) contains 40 unseen scenario configurations that simultaneously incorporate all three types of variations: object, environment, and spatial arrangements. Second, the object generalization test set (T O(i)) evaluates the success rate of a robot manipulating different objects when the scenario is fixed. This test set contains all the objects introduced during the data generation process of the i-th iteration. A higher value of this metric not only indicates better object generalization performance of the policy but also implies a better capability to enhance the diversity of objects in data generation. All success rates reported as mean values from 5 independent runs. Detailed compositions of all evaluation sets and success rate calculation method are provided in Appendix A.4.
Baselines. We compared our approach against the following methods: (1) Human Demo (Default): 20 human demonstrations per task in a fixed scenario; (2) Human Demo (Enhanced): 20 demonstrations collected across diverse scenarios; (3) DexMimicGen (Default) [14]: A representative method for dexterous data generation that synthesizes trajectories via replay and editing. For fair comparison, we provide it with the same initial dataset as DexFlyWheel-a single demonstration per task. and (4) DexMimicGen (Enhanced): To create a stronger baseline, we provided DexMimicGen with 10 diverse human demonstrations collected from different scenarios. This setup significantly enhances its ability to generalize more scenarios, giving it a 10× data advantage over our method. ( 5
this section cite: ['b59', 'b2', 'b13']

Section: Validating the Dexterous Data Flywheel Effect
This section empirically investigates Q1. We demonstrate how DexFlyWheel progressively expands dataset diversity and enhances policies performance trained on the generated data. As shown in the mid columns of Table 1, DexFlyWheel successfully expands data diversity with each iteration. In the final iteration (i=3), our method generates an average of 2,040 various scenario configurations spanning 20 different objects per task-all starting from just a single human demonstration per task. Furthermore, the object diversity results (shown in the O column of Table 1) demonstrate our framework's capacity for object-level data generation. As shown in the SR of π combined on T OEP column, we observe that as dataset diversity increases across iterations, the generalization capabilities of trained policies correspondingly improve, achieving an average success rate of 81.9% in iteration 3-a substantial improvement from the initial 16.5% in iteration 1. Additionally, as shown in the SR Boost with π res on T O (i) column, our residual policies consistently improve performance on object generalization by 32.1% on average. These results suggest promising potential for DexFlyWheel in enhancing the data diversity and policy performance across different dexterous tasks. More detail with extended iterations can be found in Appendix A.5.
this section cite: []

Section: Comparison of Policy Performance and Data Generation Efficiency
This section evaluates DexFlyWheel and baselines in both policy performance and data generation efficiency to address Q2. Policy Performance. We use identical diffusion-based policy architectures (Appendix A.1) and train them on datasets collected from DexFlyWheel and four baselines introduced in Section 5.1. As shown in Table 2, DexFlyWheel consistently achieves higher success rates than both human teleoperationbased data and replay-based methods such as DexMimicGen. This performance highlights the benefit of our iterative data flywheel mechanism, which progressively expands data diversity with policy improvement. Compared to the Human Demo baseline, which uses 20 teleoperated trajectories per task, DexFlyWheel achieves vastly superior performance-81.9% vs. 13.4% average success-while requiring only a single human demonstration per task, significantly reducing the human effort.
Data Generation Success Rate and Time Efficiency. As shown in Table 3, DexFlyWheel achieves high success across all tasks (avg. 89.8%). In contrast, DexMimicGen performs worse on dynamic
this section cite: []

Section: Ablation Study on DexFlyWheel Components
This section empirically investigates Q3. We conduct an ablation study across four manipulation tasks to isolate the impact of key modules. As shown in Figure 4, removing the residual policy leads to the most significant drop in task success rates, confirming its critical role in improving generalization and robustness. To further analysis DexFlyWheel's generalization ability, we compare the number of distinct objects successfully manipulated during data generation (Figure 5). This figure demonstrates that DexFlyWheel achieves superior object diversity compared to DexMimicGen. This performance can be attributed primarily to the residual reinforcement learning module, as evidenced by the significant drop in performance when this component is removed (w/o Res vs. ours: from 8.25 to 20 objects on average). In contrast, DexMimicGen typically operates effectively only on geometrically similar objects (same categories and shapes), which limits their generalization due to its lack of adaptability and inability to explore new strategies.   Grasp Pour Lift Handover 0 5 10 15 20 25 Number of Objects 1 1 1 1 10 9 6 2 12 9 8 4 22 12 26 20 Source Demo DexMimicGen w/o Res. DexFlywheel (Ours) Figure 5: Comparison of Object Diversity.
Our method successfully handles objects with diverse geometries, sizes, and categories.
this section cite: []

Section: Deployment on Dual-arm Real Robot System
To address Q4, we transfer our trained policy in simulation into real-world scenarios through digital twin. We set up identical hardware settings both in simulation and real-world as shown in Figure 3, which includes two Realman RM75-6F arms paired with two PsiBot G0-R hands. We employ an egocentric-view RealSense D455 camera to obtain the real-world object pose leveraging FoundationPose [61]. In particular, we train the policy using the dataset generated by the final iteration of DexFlyWheel and deploy it in the real world through the digital twin, ensuring consistency between simulation and physical environments. We evaluate the performance of our pipeline on the Dual-arm Lift and Handover Task. Experimental results show success rates of 78.3% (Dual-arm Lift) and 63.3% (Handover) in real-world deployment (20 trajectories per trial, 3 trials).
this section cite: ['b60']

Section: Limitations and Future Work
There are several limitations to our work. First, the reinforcement learning process currently relies on manually designed reward functions. Future research could investigate how LLM-driven reward generation methods can be efficiently integrated into DexFlyWheel. Second, our policies and simulations currently lack tactile feedback due to the immaturity of tactile sensing and simulation technologies. We plan to explore the potential of sensor-based tactile signals for contact-rich tasks.
this section cite: []

Section: Conclusion
We present DexFlyWheel, a scalable and self-improving framework for generating diverse, highquality dexterous manipulation data from minimal seed demonstrations. Our two-stage pipeline first leverages imitation learning to provide behavioral priors, then applies residual reinforcement learning to enhance generalization and robustness. This approach progressively expands the data distribution across diverse objects, environments, and spatial layouts. Experiments demonstrate that DexFlyWheel can generate up to 500× more trajectories and 214× more distinct scenarios per task. Policies trained on this data achieve an 81.9% success rate in challenging settings, outperforming baselines and successfully transferring to real-world robots.
this section cite: []

Section: References
Ref_id:b0 Title: Advances in neural information processing systems Year: (1996)
Ref_id:b1 Title: A vision-language-action flow model for general robot control Year: (2024)
Ref_id:b2 Title: Rt-2: Vision-language-action models transfer web knowledge to robotic control Year: (2023)
Ref_id:b3 Title: Rdt-1b: a diffusion foundation model for bimanual manipulation Year: (2024)
Ref_id:b4 Title: Scaling diffusion policy in transformer to 1 billion parameters for robotic manipulation Year: (2024)
Ref_id:b5 Title: Robobrain 2.0 technical report Year: (2025)
Ref_id:b6 Title: Tiger: Tool-integrated geometric reasoning in vision-language models for robotics Year: (2025)
Ref_id:b7 Title: Towards spatial referring with reasoning in vision-language models for robotics Year: (2025)
Ref_id:b8 Title: Dexterous manipulation using both palm and fingers Year: (2014)
Ref_id:b9 Title: Robot hands and the mechanics of manipulation Year: (1986)
Ref_id:b10 Title: Real-time behaviour synthesis for dynamic hand-manipulation Year: (2014)
Ref_id:b11 Title: Hermes: Human-to-robot embodied learning from multi-source motion data for mobile dexterous manipulation Year: (2025)
Ref_id:b12 Title: Dexcap: Scalable and portable mocap data collection system for dexterous manipulation Year: (2024)
Ref_id:b13 Title: Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning Year: (2024)
Ref_id:b14 Title: Rlbench: The robot learning benchmark & learning environment Year: (2019)
Ref_id:b15 Title: Mimicgen: A data generation system for scalable robot learning using human demonstrations Year: (2023)
Ref_id:b16 Title: Robocasa: Large-scale simulation of everyday tasks for generalist robots Year: (2024)
Ref_id:b17 Title: Robogen: Towards unleashing infinite data for automated robot learning via generative simulation Year: (2024)
Ref_id:b18 Title: Robotwin: Dual-arm robot benchmark with generative digital twins Year: (2024)
Ref_id:b19 Title: Graspnet-1billion: A largescale benchmark for general object grasping Year: (2020-06)
Ref_id:b20 Title: Scaling up and distilling down: Language-guided robot skill acquisition Year: (2023)
Ref_id:b21 Title: Rekep: Spatiotemporal reasoning of relational keypoint constraints for robotic manipulation Year: (2024)
Ref_id:b22 Title: Smart-llm: Smart multi-agent robot task planning using large language models Year: (2024)
Ref_id:b23 Title: Maniskill2: A unified benchmark for generalizable manipulation skills Year: (2023)
Ref_id:b24 Title: Bi-dexhands: Towards human-level bimanual dexterous manipulation Year: (2023)
Ref_id:b25 Title: Sequential dexterity: Chaining dexterous policies for long-horizon manipulation Year: (2023)
Ref_id:b26 Title: Mimicgen: A data generation system for scalable robot learning using human demonstrations Year: (2023)
Ref_id:b27 Title: Arena learning: Build data flywheel for llms post-training via simulated chatbot arena Year: (2024)
Ref_id:b28 Title: Star: Self-taught reasoner bootstrapping reasoning with reasoning Year: (2024)
Ref_id:b29 Title: Robot hands and the mechanics of manipulation Year: (1985)
Ref_id:b30 Title: Replay memory as an empirical MDP: Combining conservative estimation with experience replay Year: (2023)
Ref_id:b31 Title: Hands for dexterous manipulation and robust grasping: A difficult road toward simplicity Year: (2002)
Ref_id:b32 Title: Contact-invariant optimization for hand manipulation Year: (2012)
Ref_id:b33 Title: Real-time behaviour synthesis for dynamic hand-manipulation Year: (2014)
Ref_id:b34 Title: Dexterous manipulation using both palm and fingers Year: (2014)
Ref_id:b35 Title: Retrieval dexterity: Efficient object retrieval in clutters with dexterous hand Year: (2025)
Ref_id:b36 Title: Towards human-level bimanual dexterous manipulation with reinforcement learning Year: (2022)
Ref_id:b37 Title: Deep reinforcement learning for robotics: A survey of real-world successes Year: (2024)
Ref_id:b38 Title: A distance-based anomaly detection framework for deep reinforcement learning Year: (2024)
Ref_id:b39 Title: Human-to-robot imitation in the wild Year: (2022)
Ref_id:b40 Title: Dexvip: Learning dexterous grasping with human hand pose priors from video Year: (2022)
Ref_id:b41 Title: Affordances from human videos as a versatile representation for robotics Year: (2023)
Ref_id:b42 Title: Videodex: Learning dexterity from internet videos Year: (2023)
Ref_id:b43 Title: Roboclip: One demonstration is enough to learn robot policies Year: (2023)
Ref_id:b44 Title: Towards generalist robot learning from internet video: A survey Year: (2024)
Ref_id:b45 Title: Screwmimic: Bimanual imitation from human videos with screw space projection Year: (2024)
Ref_id:b46 Title: Learning generalizable 3d actions from in-the-wild 2d human videos for zero-shot robotic manipulation Year: (2025)
Ref_id:b47 Title: Wfh-vr: Teleoperating a robot arm to set a dining table across the globe via virtual reality Year: (2022)
Ref_id:b48 Title: Anyteleop: A general vision-based dexterous robot arm-hand teleoperation system Year: (2023)
Ref_id:b49 Title: Open-television: Teleoperation with immersive active visual feedback Year: (2024)
Ref_id:b50 Title: Bunny-visionpro: Real-time bimanual dexterous teleoperation for imitation learning Year: (2024)
Ref_id:b51 Title: Dexgraspnet 2.0: Learning generative dexterous grasping in large-scale synthetic cluttered scenes Year: (2024)
Ref_id:b52 Title: Gensim2: Scaling robot data generation with multi-modal and reasoning llms Year: (2024)
Ref_id:b53 Title: Code-as-monitor: Constraint-aware visual programming for reactive and proactive robotic failure detection Year: (2024)
Ref_id:b54 Title: Robot generating data for learning generalizable visual robotic manipulation Year: (2024)
Ref_id:b55 Title: Efficient preference-based reinforcement learning via aligned experience estimation Year: (2024)
Ref_id:b56 Title: Demogen: Synthetic demonstration generation for data-efficient visuomotor policy learning Year: (2025)
Ref_id:b57 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b58 Title: Policy decorator: Model-agnostic online refinement for large policy model Year: (2024)
Ref_id:b59 Title: Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation Year: (2023)
Ref_id:b60 Title: Unified 6d pose estimation and tracking of novel objects Year: (2023)
Ref_id:b61 Title: Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
