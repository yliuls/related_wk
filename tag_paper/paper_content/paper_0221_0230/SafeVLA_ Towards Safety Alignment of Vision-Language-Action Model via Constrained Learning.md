Title: SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning
Abstract: Vision-language-action models (VLAs) show potential as generalist robot policies. However, these models pose extreme safety challenges during real-world deployment, including the risk of harm to the environment, the robot itself, and humans. How can safety constraints be explicitly integrated into VLAs? We address this by exploring an integrated safety approach (ISA), systematically modeling safety requirements, then actively eliciting diverse unsafe behaviors, effectively constraining VLA policies via safe reinforcement learning, and rigorously assuring their safety through targeted evaluations. Leveraging the constrained Markov decision process (CMDP) paradigm, ISA optimizes VLAs from a min-max perspective against elicited safety risks. Thus, policies aligned through this comprehensive approach achieve the following key features: (I) effective safety-performance trade-offs, reducing the cumulative cost of safety violations by 83.58% compared to the state-of-the-art method, while also maintaining task success rate (+3.85%). (II) strong safety assurance, with the ability to mitigate long-tail risks and handle extreme failure scenarios. (III) robust generalization of learned safety behaviors to various out-of-distribution perturbations. The effectiveness is evaluated on long-horizon mobile manipulation tasks. Our data, models and newly proposed benchmark environment are available at https://pku-safevla.github.io. * Equal Contribution.

Section: Introduction
Embodied AI aims to develop a generalist policy that can perform perception, interaction, reasoning, and adaptation in the physical world [1]. Building on the emergence of large language models (LLMs) and vision-language models (VLMs), vision-language-action models (VLAs) [2,3,4,5] advance this field by enabling robots to follow vision-language instructions and perform tasks in real-world environments. As these models continue to evolve, they have the potential to become generalist robot policies [6,7], capable of executing previously unseen instructions and effectively generalizing behaviors across a diverse range of robot embodiments, scenes, skills, and objects [3]. Ensuring the alignment of these models with human values and safety has become more critical than ever [8,9], due to their increasing complexity and power [10,11,12,13,14]. While significant progress has been made in task performance, the explicit integration of safety mechanisms remains an open challenge.
How can safety constraints be explicitly integrated into VLAs without loss of performance?
The safety risks of LLMs and VLMs have been extensively studied, with existing methods such as data augmentation [15], content moderation [16,17], reinforcement learning from human feedback (RLHF) [18,19], Safe-RLHF [20], language feedback [21,22], and lightweight alignment [23,24]. However, these safety mechanisms cannot be directly applied to VLAs, as there is a substantial gap between the abstract safety concerns at the model intention level [25,26] and the unique safety M += 1 if N / M <= threshold and collided: return "UNSAFE" return "SAFE"
Figure 1: The Integrated Safety Approach (ISA) pipeline. Our proposed pipeline employs multifaceted framework for the systematic safety alignment of vision-language-action (VLA) models. challenges posed by the complex and unpredictable physical world [27]. Despite large-scale behavior cloning and careful alignment in existing VLAs [28,29], the most advanced models have yet to explicitly define and integrate safety as an integral aspect of their design [30,31,32,33,34,35]. This fundamental limitation motivates an urgent need to explore methodologies capable of explicitly embedding safety constraints into the VLAs [36,37].
To tackle this challenge, we make the first systematic explorations into VLA safety alignment. Our approach is grounded in the constrained Markov decision process (CMDP) framework [38,39], leveraging methodology from safe reinforcement learning (SafeRL) for optimization. We investigate an integrated safety approach (ISA), which systematically considers four key aspects: comprehensively modeling safety requirements within the CMDP setup, actively eliciting diverse unsafe behaviors to inform constraints, rigorously constraining VLA policies using CMDP-compliant SafeRL techniques, and thoroughly assuring safety through targeted evaluations. The core insight of such an approach is to explicitly trade off safety and task performance, prioritizing safety adherence. Our investigation addresses the significant engineering challenges in adapting and scaling these principles for VLAs, focusing on how to effectively model, elicit, and utilize safety signals.
To the best of our knowledge, this work is the first systematic explorations into explicitly integrating safety constraints into VLAs using principles from SafeRL. Our main contributions are:
• Integrated Safety Approach (ISA) Exploration: We conduct a comprehensive investigation into an ISA for VLA safety alignment. This involves systematically exploring and implementing methodologies for: (a) modeling intricate safety requirements and diverse scenarios; (b) eliciting a wide spectrum of latent unsafe behaviors; (c) constraining VLA policies using CMDP-based SafeRL, optimizing from a min-max perspective; and (d) establishing robust practices for assuring the safety of aligned policies through targeted evaluations and stress-testing. Our study details how these interconnected aspects contribute to a more holistic safety alignment.
• Environment: Addressing the gap in comprehensive VLA safety assessment, we introduce Safety-CHORES. This novel testbed is a direct result of the modeling and eliciting aspects of our ISA. To this end, the benchmark is designed with fine-grained safety constraints embedded within diverse, long-horizon tasks that integrate navigation and manipulation. By incorporating large-scale procedurally generated scenes and specifically targeting safety critical components, Safety-CHORES more effectively surfaces VLA vulnerabilities than conventional benchmarks.
• Empirical Validation and Key Findings: Our extensive experiments demonstrate that policies aligned through our ISA exploration achieve: (I) an effective trade-off between safety and task performance, evidenced by an average 83.58% safety improvement over state-of-the-art method, while maintaining task performance (+3.85%); (II) strong safety assurance, particularly in mitigating long-tail risks and handling extreme failure scenarios, as supported by the elimination of high-risk actions and a drastic reduction in unsafe incident severity; and (III) robust generalization of learned safety behaviors to out-of-distribution (OOD) perturbations. These findings underscore the potential of a comprehensive, multi-faceted approach to significantly advance VLA safety.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b2', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38']

Section: Related Work
Vision-Language-Action Models. Vision-language-action models (VLAs) [2,30,3,32,4,5,34] represent a significant step towards generalist robots capable of executing complex tasks based on multimodal instructions in diverse environments [6,7,40]. These models, often built upon powerful foundation models [41,35,42] and trained on large-scale trajectory datasets [3], demonstrate impressive task performance and generalization ability [43]. As these models advance, they exhibit a growing range of capabilities, including cross-embodiment generalization [44], dexterous manipulation [45], nuanced instruction following [46], long-horizon planning [47], reasoning [48,49], and spatial awareness [50]. However, their real-world deployment is hindered by safety concerns inherent to physical interaction [27,36]. While safety alignment is actively researched for LLMs and VLMs [18,51,52,23,53], methods focusing on mitigating abstract risks like harmful content generation [54,55] do not readily address the concrete physical hazards faced by embodied agents. Current VLA training, typically relying on imitation learning (IL) [32] or standard reinforcement learning (RL) fine-tuning [28,29], lacks mechanisms for explicitly integrating and enforcing safety constraints, leaving a critical gap for reliable deployment [37].
Safe Reinforcement Learning. Safe reinforcement learning (SafeRL) within the constrained Markov decision process (CMDP) framework [38,56], offers a principled paradigm to policy optimization where an agent learns to maximize task rewards while explicitly satisfying predefined safety constraints. This paradigm contrasts with heuristic methods like reward shaping, which indirectly encode safety preferences and lack formal guarantees. While SafeRL techniques have been explored for aligning foundation models (e.g., Safe-RLHF [20]), applying them to high-dimensional, multimodal VLAs operating in complex physical environments poses unique challenges [57]. VLAs are highly generalized agents capable of following open-ended instructions [58]. This is fundamentally different from training specialized agents from scratch for a single, fixed task. Model-free, first-order optimization methods compatible with the CMDP formulation, such as Lagrangian-based approaches [59,60], are promising for VLAs as they avoid restrictive assumptions about system dynamics or state structure, making them suitable for learning from raw perceptual inputs like RGB images [39].
Our work systematically explores the application of these principles to VLA safety alignment.
this section cite: ['b1', 'b29', 'b2', 'b31', 'b3', 'b4', 'b33', 'b5', 'b6', 'b39', 'b40', 'b34', 'b41', 'b2', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b26', 'b35', 'b17', 'b50', 'b51', 'b22', 'b52', 'b53', 'b54', 'b31', 'b27', 'b28', 'b36', 'b37', 'b55', 'b19', 'b56', 'b57', 'b58', 'b59', 'b38']

Section: Benchmarking Safety and VLA Alignment.
Evaluating VLA safety requires appropriate benchmarks capable of eliciting unsafe behaviors. Existing SafeRL benchmarks often involve simplified dynamics or non-photorealistic settings [61,62,63,64], while standard VLA benchmarks primarily focus on task success across manipulation [65,66,67] or navigation [68,69], lacking diverse and challenging scenarios with built-in safety constraints. Thus, we propose Safety-CHORES to comprehensively assess safety alongside task performance in complex, procedurally generated environments. While prior work like FLaRe [28] and GRAPE [29] employed RL fine-tuning for VLAs, their objective was primarily task performance improvement and generalization, without the explicit safety constraint satisfaction central to our SafeRL-based approach. Our approach utilizes the CMDP framework to formulate VLA alignment as a constrained optimization problem. This approach differs fundamentally from prior RL fine-tuning methods. Specifically, it allows for directly tackling the trade-off between safety and task performance to ensure adherence to predefined safety constraints.
this section cite: ['b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b27', 'b28']

Section: Problem Formulation
Constrained Markov Decision Process. The constrained Markov decision process (CMDP) [38] is commonly used to model dynamic decision-making under uncertainty when multiple objectives are present. In this framework, the policy aims to maximize one objective while satisfying constraints on the others. A CMDP is defined as a tuple (S, A, P, r, C, µ, γ), where S is state space, A is action space. P(s ′ |s, a) is probability of state transition from s to s is the initial state distribution and γ ∈ (0, 1). Let H t be the set of all possible trajectories (s 0 , a 0 , . . . , s t-2 , a t-2 , s t-1 ) of length t.
this section cite: ['b37']

Section: From CMDP to VLA Safety Alignment.
To address the safety constrained decision-making problem in VLAs, we formulate VLA safety alignment using an adapted CMDP framework, defined by the tuple (S, A, P, r, C, L, µ, γ), where L is the set of natural language instructions. The reward function r is conditioned on a natural language instruction l ∈ L, and is defined as r : S × S × A × L → R. Let π θ denote the vision-language-action model parameterized by θ, which maps an observation history h t = (o t+1-H , a t+1-H , . . . , o t ) to an action a t ∼ π θ (•|l, h t ), where H > 1 is the temporal horizon, l is the natural language instruction. Each observation o t = (v t , p t ) represents the multimodal perceptual input at time t, comprising visual input v t and proprioceptive input p t .
The reward-return is defined as J (π θ ) = E π θ ,L [ ∞ t=0 γ t r (s t+1 |s t , a t , l)]. The set of feasible policies is then defined as
Π C = π θ ∈ Π Θ | E π θ ∞ t=0 γ t c i (s t , a t ) ≤ b i , ∀i = 1, . . . , m .(1)
Formally, we aim to solve π * = arg max
π θ ∈Π C J (π θ ).(2)
this section cite: []

Section: Implementing the Integrated Safety Approach
We argue that VLA safety requires an integrated safety approach (ISA), rather than a single method. Specifically, an ISA address four interconnected aspects, as shown in Figure 1 (see Appendix B.9 for details): (i) modeling safety-critical aspects of tasks and environments; (ii) eliciting latent and diverse unsafe behaviors from existing policies; (iii) constraining the VLA's learning process to integrate these safety considerations; and (iv) assuring the resulting model's safety through rigorous and targeted evaluation. In this section, we present our methodologies into each of these aspects.
this section cite: []

Section: Modeling Safety: Scenes, Specifications, and Tasks
In our investigation, we focus on a mobile manipulation setting. The static part of each task t i ∈ T is defined as (e i , x i , q i , G, Φ, Ψ). Here, e i ∈ E is the scene, x i and q i are the randomly selected initial robot position and orientation, G is the set of object categories in e i , Φ is a set of state-action safety predicates, and Ψ is a set of trajectory-level safety predicates. A safety predicate serves as a compact representation for identifying unsafe behaviors. It can be expressed as either a state-action predicate ϕ : S × A → {0, 1} or a trajectory-level predicate ψ : H → {0, 1}.
Each state-action predicate is defined using compositional logic: ϕ(s, a) = 1 ⇐⇒ P s (s) ∧ P a (a) ∧ R(s, a), where P s and P a capture relevant conditions on states and actions, and R represents the risk-inducing relation. Similarly, trajectory predicates are defined as:
ψ(τ ) = 1 ⇐⇒ ∃t 0 , . . . , t k ∈ [0, len(τ )] s.t. k i=0 E i (s ti , a ti ) ∧ R temporal ({(t j , s tj , a tj )} k j=0 , τ ),
where each E i (s ti , a ti ) is an event predicate that evaluates to true if a specific condition holds for the state-action pair (s ti , a ti ) at time t i , and R temporal (•) is a predicate describing the temporal structure.
To instantiate t i , we dynamically augment the static components with a natural language instruction l. Specifically, we randomly select an object category g ∈ G as the goal, and then sample a natural language instruction l to specify g. Inspired by [32], we build three categories of tasks:
• Safety-ObjNav: The robot must navigate through multiple rooms to locate a designated object.
• Safety-PickUp: The robot begins in front of a surface and is instructed to pick up a specific object.
• Safety-Fetch: This task requires the robot first navigate to find the target object and then pick it up.
this section cite: ['b31']

Section: Eliciting Risks: Uncovering Latent Unsafe Behaviors
To ensure comprehensive risk elicitation and to prevent policies from overfitting to limited scenarios, maximizing the diversity of both environmental settings and interactable objects is critical. Therefore, we utilize a large-scale dataset of 150K diverse indoor scenes generated by ProcTHOR [70], alongside Objaverse [71], which provides an extensive library of 800K 3D assets. The simulation is conducted in the AI2THOR [72] simulator, which supports photo-realistic rendering quality, object state changes, arm-based manipulation, and causal interactions.
Building upon this foundation of diverse scenes and objects, to further systematize risk elicitation and ensure targeted coverage of known problematic scenarios, we identify and leverage several safety critical components. These are not separate entities but rather specific environmental features (e.g., narrow corners) or challenging object arrangements (e.g., fragile collections) that are instantiated or frequently occur within the aforementioned large-scale scenes. The safety critical components considered in our study include (see Appendix D for details):
• Corners (ϕ corner ): Situations where navigation into confined spaces like narrow corners leads to the robot becoming stuck or incurring repeated collisions.
• Blind Spots (ψ blind spot ): Collisions with previously seen but currently unobserved obstacles due to failures in maintaining short-term spatial awareness.
• Fragile Collections (ψ fragile collection ): Scenarios involving collateral damage to nearby fragile items during manipulation tasks, often due to object density or precarious placements.
• Critical Points (ψ critical point ): Incidents where robot actions, even indirect ones, destabilize precariously positioned objects (e.g., a knife on an edge), causing them to fall.
• Dangerous Equipment (ϕ dangerous equipment ): Prohibited interactions with intrinsically hazardous objects like active stovetops or exposed wiring, which demand strict avoidance.
By incorporating these diverse scenes, objects, and safety critical components, we propose Safety-CHORES to systematically elicit a wide spectrum of potential safety violations, thereby generating rich, safety-aware data. These complex tasks require VLAs to integrate natural language understanding, visual reasoning, and long-horizon planning, while adhering to the modeled safety constraints.
this section cite: ['b69', 'b70', 'b71']

Section: Constraining Policies: Safe Reinforcement Learning for Alignment
Once safety specifications are modeled and data of potential risks can be elicited, we leverage SafeRL techniques to effectively integrate these safety considerations into the VLA's policy learning process.
A preliminary step is translating safety predicates (ϕ, ψ) into cost signals for the cost-returns J ci (θ). State-action predicate (ϕ k ) violations incur a cost of 1 at the violating timestep t, otherwise 0. For trajectory-level predicates (ψ j ), a cost of 1 is attributed solely to the final step of the violating segment in this initial exploration. The credit assignment for ψ j remains an area for exploration in future work. The Lagrangian method is a general solution for SafeRL. By employing the Lagrangian relaxation  technique [73], Equation 2 is transformed into an unconstrained safe optimization problem:
min θ max λ≥0 [-J r (θ) + n i=0 λ i J ci (θ)],(3)
where λ i ≥ 0 is the Lagrange multiplier and n is the number of constraints.
Solving the min-max optimization in Equation 3 necessitates an iterative refinement process, where updates to the VLA model parameters θ are interleaved with those to the Lagrange multipliers λ. It optimizes for safety first, then maximizing task performance. This trade-off ensures that the VLA model adheres to safety requirements while maximizing task performance within these constraints.
this section cite: ['b72']

Section: Safety Assurance: Evaluating Aligned VLAs
The final aspect of ISA is the assurance of safety through comprehensive evaluation. Our assurance methodology systematically assesses the model's safety performance across several dimensions:
• Test-time Safety evaluates the model's adherence of safety constraints through performance on held-out test sets and out-of-distribution (OOD) perturbations. The primary goal is to quantify the learned safe behaviors in the training phase.
• Long-tail Safety considers the model's safety on statistically infrequent events. Ensuring that the model does not exhibit long-tail safety issues is crucial for robust safety in real-world deployments.
• Extreme Failure Safety focuses on the model's safety and behavior to catastrophic failures. This is particularly assessed in situations where task completion may be impossible.
this section cite: []

Section: Experiments
In this section, we aim to answer the following questions: (I) Can ISA outperform standard VLA fine-tuning methods? ( § 5.2.1); (II) How do ISA-aligned VLAs qualitatively handle risks and failures? ( § 5.2.2); (III) Which components within ISA critically impact its safety-performance balance? ( § 5.2.3) (IV) Do learned safety behaviors generalize to OOD scenarios and extreme failures? ( § 5.2.4)
this section cite: []

Section: Experimental Setup
Tasks, Environments and Training. Our primary experiments utilize Safety-CHORES. To contextualize the unique challenges posed by Safety-CHORES, we also conduct comparisons on other benchmarks [72,69,70] focusing on object navigation and generally lack the safety features of Safety-CHORES. The cost threshold b i is empirically set to 20% of the converged cost from the FLaRe baseline. This common SafeRL practice [74,63] avoids arbitrary absolute values. For simpler tasks like Safety-ObjNav and Safety-PickUp, we train for 15 million steps. For more complex tasks that require integrated capabilities, such as Safety-Fetch, we train for 25 million steps.
Baseline Methods. We compare ISA against a comprehensive set of baselines that represent various paradigms for VLA training and fine-tuning. IL-only: SPOC [32], which is a state-of-the-art imitation learning method. IL-only (Ground Truth): SPOC augmented with ground truth information. These models can thoroughly showcase the potential upper bound of IL methods. IL+RL (Standard): FLaRe [28], which fine-tunes pre-trained VLAs using reinforcement learning focused solely on task performance. IL+RL (Reward Shaping): FLaRe-RS, a variant of FLaRe where safety costs are directly used as penalties on reward, representing a common heuristic for addressing safety. RL-Only: Poliformer [75], an end-to-end RL approach for navigation tasks.
this section cite: ['b71', 'b68', 'b69', 'b73', 'b62', 'b31', 'b27', 'b74']

Section: Initial IL Model.
We begin our experiments with the SPOC-DINOv2 model. We select it as our initial model for two main reasons. First, SPOC is a state-of-the-art VLA trained solely on simulated data. Second, it demonstrates strong transferability to real-world deployment, making it suitable for safety-critical data collection. We also evaluate ISA on other VLA models (i.e., EmbCLIP [76], Embodied-Codebook [77] and their variants with different vision encoders).
this section cite: ['b75', 'b76']

Section: Evaluation Metrics.
Borrowing from safety considerations in robotics [78,79], our evaluation focuses on two metrics: the task success rate (SR) and the cumulative cost (CC). The CC is an aggregate measure of all safety violations throughout an episode. For a trajectory τ of length L and K distinct safety constraint types, it is computed as CC(τ ) = K k=1 L-1 t=0 c k (s t , a t ), where c k (•) is the cost incurred from violating the k-th safety constraint at step t.
this section cite: ['b77', 'b78']

Section: Main Results

this section cite: []

Section: Comparative Performance: ISA vs. Standard Methods
We first evaluate the effectiveness of ISA in enhancing VLA safety while preserving task performance. In   demonstrates substantial safety improvements, achieving an average reduction in CC of 83.58% compared to the strongest task-focused RL baseline, FLaRe. This significant decrease is consistent across all tasks, as illustrated by per-room safety improvements in Figure 10. Crucially, these safety enhancements are accompanied by maintained task performance. ISA achieves an average SR increase of 3.85% compared to FLaRe, outperforming IL-only baselines and matching or exceeding other RL-based methods. This indicates ISA effectively trades off the safety and task performance, in contrast to approaches that solely optimize for task performance.
this section cite: []

Section: Qualitative Insights: Risk Handling and Failure Modes
In Figure 3 (Left), we present the distribution of cumulative safety costs for ISA and FLaRe across all test trajectories. A key observation is that ISA eliminates trajectories with extremely high safety costs (cumulative cost >10). The upper bound of unsafe behavior severity in ISA is reduced to 1/35th of that in FLaRe, indicating a significant mitigation of catastrophic safety failures. This shift in distribution demonstrates ISA's effectiveness in mitigating long-tail risks, where a small number of trajectories could otherwise account for a disproportionate amount of unsafe behaviors.
Further analysis, shown in Figure 3 (Middle and Right), reveals a difference in how safety correlates with task success. For FLaRe, higher safety costs are more prevalent in task failures, suggesting that unsafe behaviors often contribute to or coincide with failure. Logistic regression and Pearson correlation tests (see Appendix A for more details) confirm a significant negative correlation between cost and success for FLaRe (p < 0.01). In contrast, ISA exhibits a more consistent cost distribution regardless of task outcome. The T-test rejects the correlation for ISA, indicating that the learned safety paradigm is largely decoupled from task success. Even when ISA fails a task, it tends to do so more safely, avoiding safety violations. This suggests a deeper integration of safety principles rather than superficial avoidance. For further cases and behavior analysis, please refer to Appendix B.1.
this section cite: []

Section: Ablation Studies: Impact of Key ISA Design Choices
To understand the contribution of specific design choices in ISA, we conduct several ablation studies. Importance of Risk Elicitation. The importance of risk elicitation is demonstrated by an ablation study in Figure 7 (Left). When the standard ISA training recipe was applied to simplified one-room scenes without safety critical components, safety performance degraded considerably. This ablated model yielded a CC nearly three times higher than the full ISA's (5.01 vs. 1.854) and even performed worse than the FLaRe-RS baseline, alongside a reduced SR (0.645 vs. 0.865). This significant decline, particularly in safety despite identical constraining mechanisms, underscores that rich elicitation environments are indispensable for achieving safety alignment superior to heuristic approaches. ISA Generalizability to Different VLA Models. In Figure 4, we validate the generalizability by applying ISA's alignment process to several distinct VLA base models. The results consistently show that ISA alignment leads to substantial improvements across these models, evidenced by significant reductions in CC alongside stable SR when evaluated on Safety-CHORES and other benchmarks.
this section cite: []

Section: Safety Challenges Posed by Safety-CHORES.
In Figure 5, we demonstrate the applicability of Safety-CHORES to various VLA models and observe a consistent trend: across various VLA models, the CC on Safety-CHORES (green segments) often more than 2 times that on benchmarks like iTHOR or ProcTHOR. This pronounced difference is observed under identical safety evaluation mechanisms applied to all benchmarks; however, standard benchmarks inherently lack the safetycritical environmental designs.
Importance of Lagrangian Multipliers. The Lagrangian dual formulation (Equation 3) uses dynamic multipliers λ to balance reward and cost objectives. We compare this against baselines using fixed penalty coefficients for safety costs, as shown in Figure 6. The results demonstrate that our approach with dynamic Lagrangian multipliers achieves a superior trade-off, adhering to the cost limit while attaining a higher success rate than any fixed-penalty baseline that meets the same cost constraint. This highlights the benefit of the adaptive constraining mechanism provided by the Lagrangian method for effectively balancing safety and task performance.
this section cite: []

Section: Impact of Cost Threshold b i .
The choice of the safety cost threshold b i in the CMDP formulation (Equation 1) directly influences the strictness of the safety constraints. In Figure 7 (Middle), we shows the performance on Safety-ObjNav when varying b i (e.g., 10%, 20%, 50% of FLaRe's converged cumulative cost 11.5982). As observed, stricter thresholds lead to lower realized safety costs, demonstrating effective constraint enforcement. However, excessively strict thresholds (e.g., 10%) might slightly impact SR. The chosen 20% threshold offers a balance.
this section cite: []

Section: Robustness: Generalization to OOD Scenarios and Extreme Failures

this section cite: []

Section: OOD Perturbation Results.
In Table 2, we presents the performance of ISA on Safety-CHORES tasks under four types of OOD perturbations: color, lighting, material, and all combined. The average changes reported at the bottom of
this section cite: []

Section: Safety Under Extreme Task Failure Conditions.
To further probe the robustness, particularly when task completion is unattainable, we curated a specialized set of environments. These scenarios incorporate novel goals and unfamiliar instructions to induce universal task failure (SR is nearly 0.0). Such extreme failure scenarios effectively isolate the models' inherent safety behaviors from any influence of task success. While task failure is universal, a pronounced difference in safety emerges. In Figure 7 (Right), we observe that baselines exhibit high safety violations. For instance, FLaRe incurs an average CC of 71.68, over 32 times higher than that of the ISA-aligned model ( 2.20). Similarly, SPOC accumulates a CC of 14.63, nearly 7 times greater. These excessive costs stem from their frequent engagement in risky behaviors, such as repeated collisions (see Appendix B.2 for more details), despite making no progress on the task. This pattern strongly indicates that their default behavior, when not guided by a successful task trajectory, remains inherently unsafe.
this section cite: []

Section: Empirical Study: Sim-to-Real Transfer
To validate the real-world applicability of our framework, we constructed the physical robot platform shown in Figure 8. On this platform, we successfully deployed the aligned policy for a Safety-PickUp task. Demonstration videos are available at our project website. The robot demonstrated effective obstacle avoidance that was consistent with its behavior in simulation. We identify and address two primary challenges: the input distribution shift from sensors and the dynamics mismatch between simulation and reality. We developed the following strategies to overcome them:
• Perception Strategy: To bridge the input shift, we leverage pre-trained models (e.g., Foundation-Pose [80]) to convert noisy images into robust, structured state representations (e.g., 6D poses), thus avoiding the need for extensive real-world image datasets.
• Dynamics Decoupling: To mitigate the dynamics mismatch, we decouple the high-level policy from low-level motor control via a shared semantic or Cartesian action space, making the policy robust to minor physical variations.
• Digital Twin Alignment: To further minimize the mismatch, we fine-tune simulator physics parameters (e.g., PID controllers, action cycles) to precisely mirror the real robot's motion characteristics.
• Data Pipeline Consistency: To reduce processing-related errors, we maintain an identical data transformation pipeline (e.g., pose estimator, IK solver) across both simulation and deployment.
The successful transfer validates that safety constraints can be learned in simulation and transfer effectively to the physical world. Our findings underscore the value of simulation as a tool for developing and testing safe robotic policies, similar to its application in autonomous driving [81,82].
this section cite: ['b79', 'b80', 'b81']

Section: Conclusion
In this work, we introduce an ISA to mitigate significant safety challenges of VLA. ISA systematically applies SafeRL principles via the CMDP framework, effectively aligning VLAs with safety requirements. Our research explored and systematically integrated novel modeling, eliciting (through our Safety-CHORES benchmark), policy constraining, and assurance techniques within this ISA. This comprehensive approach achieved an 83.58% safety improvement over the state-of-the-art method while maintaining task performance (+3.85%). Crucially, aligned policies showed robust safety assurance, mitigating long-tail risks and generalizing to out-of-distribution perturbations and extreme failures, marking a first systematic integration of explicit safety constraints into VLAs using SafeRL.
this section cite: []

Section: References
Ref_id:b0 Title: Aligning cyber space with physical world: A comprehensive survey on embodied ai Year: (2024)
Ref_id:b1 Title: Rt-1: Robotics transformer for real-world control at scale Year: (2022)
Ref_id:b2 Title: Open x-embodiment: Robotic learning datasets and rt-x models Year: (2023)
Ref_id:b3 Title: An open-source generalist robot policy Year: (2024)
Ref_id:b4 Title: An open-source vision-language-action model Year: (2024)
Ref_id:b5 Title: Jost Tobias Springenberg, et al. A generalist agent Year: (2022)
Ref_id:b6 Title: A survey on visionlanguage-action models for embodied ai Year: (2024)
Ref_id:b7 Title: Challenges and applications of large language models Year: (2023)
Ref_id:b8 Title: Ai alignment: A comprehensive survey Year: (2023)
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: Openai o1 system card Year: (2024)
Ref_id:b11 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b12 Title: Language models resist alignment: Evidence from data compression Year: (2025-07)
Ref_id:b13 Title: Shadows of intelligence: A comprehensive survey of ai deception Year: (2025-09-24)
Ref_id:b14 Title: Reinforced self-training (rest) for language modeling Year: (2023)
Ref_id:b15 Title: Llama guard: Llm-based input-output safeguard for human-ai conversations Year: (2023)
Ref_id:b16 Title: Llama guard 3 vision: Safeguarding human-ai image understanding conversations Year: (2024)
Ref_id:b17 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b18 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b19 Title: Safe rlhf: Safe reinforcement learning from human feedback Year: (2023)
Ref_id:b20 Title: Align anything: Training all-modality models to follow instructions with language feedback Year: (2024)
Ref_id:b21 Title: Sequence to sequence reward modeling: Improving rlhf by language feedback Year: (2025)
Ref_id:b22 Title: Aligner: Efficient alignment by learning to correct Year: (2024)
Ref_id:b23 Title: Med-aligner empowers llm medical applications for complex medical scenarios. The Innovation Year: (2025)
Ref_id:b24 Title: Generative rlhf-v: Learning principles from multi-modal human preference Year: (2025)
Ref_id:b25 Title: Multi-turn interleaved preference alignment with human feedback Year: (2025)
Ref_id:b26 Title: Safety-critical advanced robots: A survey Year: (2017)
Ref_id:b27 Title: Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning fine-tuning Year: (2024)
Ref_id:b28 Title: Generalizing robot policy via preference alignment Year: (2024)
Ref_id:b29 Title: Rt-2: Vision-languageaction models transfer web knowledge to robotic control Year: (2023)
Ref_id:b30 Title: Rt-trajectory: Robotic task generalization via hindsight trajectory sketches Year: (2023)
Ref_id:b31 Title: Imitating shortest paths in simulation enables effective navigation and manipulation in the real world Year: (2024)
Ref_id:b32 Title: Debidatta Dwibedi, and Dorsa Sadigh. Rt-h: Action hierarchies using language Year: (2024)
Ref_id:b33 Title: pi0 : A vision-language-action flow model for general robot control Year: (2024)
Ref_id:b34 Title: Rdt-1b: a diffusion foundation model for bimanual manipulation Year: (2024)
Ref_id:b35 Title: Antonios Gasteratos, and Ioannis Dokas. Safety bounds in human robot interaction: A survey Year: (2020)
Ref_id:b36 Title: Governing ai safety through independent audits Year: (2021)
Ref_id:b37 Title: Constrained Markov decision processes Year: (2021)
Ref_id:b38 Title: Omnisafe: An infrastructure for accelerating safe reinforcement learning research Year: (2024)
Ref_id:b39 Title: A survey on vision-languageaction models: An action tokenization perspective Year: (2025)
Ref_id:b40 Title: Robotic control via embodied chain-of-thought reasoning Year: (2024)
Ref_id:b41 Title: Fast: Efficient action tokenization for vision-language-action models Year: (2025)
Ref_id:b42 Title: Towards testing and evaluating vision-language-action models for robotic manipulation: An empirical study Year: (2024)
Ref_id:b43 Title: Learning to act anywhere with task-centric latent actions Year: (2025)
Ref_id:b44 Title: Dexgraspvla: A vision-language-action framework towards general dexterous grasping Year: (2025)
Ref_id:b45 Title: Gemini robotics: Bringing ai into the physical world Year: (2025)
Ref_id:b46 Title: Hi robot: Open-ended instruction following with hierarchical vision-language-action models Year: (2025)
Ref_id:b47 Title: Cot-vla: Visual chain-of-thought reasoning for vision-language-action models Year: (2025)
Ref_id:b48 Title: Vision-language-action model with open-world embodied reasoning from pretrained knowledge Year: (2025)
Ref_id:b49 Title: Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies Year: (2024)
Ref_id:b50 Title: Constitutional ai: Harmlessness from ai feedback Year: (2022)
Ref_id:b51 Title: Beavertails: Towards improved safety alignment of llm via a human-preference dataset Year: (2024)
Ref_id:b52 Title: Safe rlhf-v: Safe reinforcement learning from multi-modal human feedback Year: (2025)
Ref_id:b53 Title: Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned Year: (2022)
Ref_id:b54 Title: An overview of catastrophic ai risks Year: (2023)
Ref_id:b55 Title: A review of safe reinforcement learning: Methods, theory and applications Year: (2022)
Ref_id:b56 Title:  Year: (2024)
Ref_id:b57 Title: Action reasoning models that can reason in space Year: (2025)
Ref_id:b58 Title: Responsive safety in reinforcement learning by pid lagrangian methods Year: (2020)
Ref_id:b59 Title: Augmented proximal policy optimization for safe reinforcement learning Year: (2023)
Ref_id:b60 Title: Ai safety gridworlds Year: (2017)
Ref_id:b61 Title: Safe-control-gym: A unified benchmark suite for safe learningbased control and reinforcement learning in robotics Year: (2022)
Ref_id:b62 Title: Safety gymnasium: A unified safe reinforcement learning benchmark Year: (2023)
Ref_id:b63 Title: Hasard: A benchmark for vision-based safe reinforcement learning in embodied agents Year: (2025)
Ref_id:b64 Title: Rlbench: The robot learning benchmark & learning environment Year: (2020)
Ref_id:b65 Title: Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks Year: (2022)
Ref_id:b66 Title: Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks Year: (2024)
Ref_id:b67 Title: Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments Year: (2018)
Ref_id:b68 Title: Robothor: An open simulation-to-real embodied ai platform Year: (2020)
Ref_id:b69 Title: Procthor: Large-scale embodied ai using procedural generation Year: (2022)
Ref_id:b70 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b71 Title: An interactive 3d environment for visual ai Year: (2017)
Ref_id:b72 Title: Numerical Optimization Year: (2006)
Ref_id:b73 Title: First order constrained optimization in policy space Year: (2020)
Ref_id:b74 Title: Poliformer: Scaling on-policy rl with transformers results in masterful navigators Year: (2024)
Ref_id:b75 Title: Simple but effective: Clip embeddings for embodied ai Year: (2022)
Ref_id:b76 Title: Selective visual representations improve convergence and generalization for embodied ai Year: (2023)
Ref_id:b77 Title: A constraint-based method for solving sequential manipulation planning problems Year: (2014)
Ref_id:b78 Title: A real-time approach for chanceconstrained motion planning with dynamic obstacles Year: (2020)
Ref_id:b79 Title: Foundationpose: Unified 6d pose estimation and tracking of novel objects Year: (2024)
Ref_id:b80 Title: Av-fuzzer: Finding safety violations in autonomous driving systems Year: (2020)
Ref_id:b81 Title: Waymo simulated driving behavior in reconstructed fatal crashes within an autonomous vehicle operating domain Year: (2021)
Ref_id:b82 Title: Divscene: Benchmarking lvlms for object navigation with diverse scenes and objects Year: (2024)
Ref_id:b83 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b84 Title: Allenact: A framework for embodied ai research Year: (2020)
Ref_id:b85 Title: Safe learning in robotics: From learning-based control to safe reinforcement learning Year: (2022)
Ref_id:b86 Title: Provably safe and robust learning-based model predictive control Year: (2013)
Ref_id:b87 Title: Learning barrier functions for constrained motion planning with dynamical systems Year: (2019)
Ref_id:b88 Title: Manipulation planning on constraint manifolds Year: (2009)
Ref_id:b89 Title: Learning-based model predictive control: Toward safe learning in control Year: (2020)
Ref_id:b90 Title: Learning-based model predictive control for safe exploration Year: (2018)
Ref_id:b91 Title: Safe exploration in continuous action spaces Year: (2018)
Ref_id:b92 Title: Recovery rl: Safe reinforcement learning with learned recovery zones Year: (2021)
Ref_id:b93 Title: Safe reinforcement learning: A control barrier function optimization approach Year: (2021)
Ref_id:b94 Title: Uncertaintyaware reinforcement learning for collision avoidance Year: (2017)
Ref_id:b95 Title: Learning barrier certificates: Towards safe reinforcement learning with zero training-time violations Year: (2021)
