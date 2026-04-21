Title: WORLD-IN-WORLD: WORLD MODELS IN A CLOSED-LOOP WORLD
Abstract: Generative world models (WMs) can now simulate worlds with striking visual realism, which naturally raises the question of whether they can endow embodied agents with predictive perception for decision making. Progress on this question has been limited by fragmented evaluation: most existing benchmarks adopt openloop protocols that emphasize visual quality in isolation, leaving the core issue of embodied utility unresolved, i.e., do WMs actually help agents succeed at embodied tasks? To address this gap, we introduce World-In-World, the first open platform that benchmarks WMs in a closed-loop world that mirrors real agent-environment interactions. World-In-World provides a unified online planning strategy and a standardized action API, enabling heterogeneous WMs for decision making. We curate four closed-loop environments that rigorously evaluate diverse WMs, prioritize task success as the primary metric, and move beyond the common focus on visual quality; we also present the first data scaling law for world models in embodied settings. Our study uncovers three surprises: (1) visual quality alone does not guarantee task success-controllability matters more; (2) scaling posttraining with action-observation data is more effective than upgrading the pretrained video generators; and (3) allocating more inference-time compute allows WMs to substantially improve closed-loop performance. By centering evaluation on closedloop outcomes, World-In-World establishes a new benchmark for the systematic assessment of WMs.

Section: INTRODUCTION
Recent advances in visual generation have sparked interest in world generation, a field focused on the creation of diverse environments populated with varied scenes and entities, with applications in entertainment, gaming, simulation, and embodied AI. The rapid progress in video generation (Brooks et al., 2024;Yang et al., 2024b;Wan et al., 2025), 3D scene generation (Fridman et al., 2023;Chung et al., 2023;Yu et al., 2024;Koh et al., 2023;Ling et al., 2025), and 4D scene generation (Bahmani et al., 2024b;Xu et al., 2024;Bahmani et al., 2024a) has demonstrated high-quality individual scene generation, highlighting the potential of these models as world generation systems. In this work, we address this gap by proposing World-In-World, which wraps generative World models In a closed-loop World interface to measure their practical utility for embodied agents. Specifically, we present a unified strategy for closed-loop online planning and a standardized action API to seamlessly integrate diverse world models into closedloop tasks. The online planning strategy allows the agent to look ahead by anticipating environmental changes and task rewards before committing to an action. The standardized action API harmonizes input modalities expected by different world models, so that each model can be controlled consistently within the same evaluation protocol. In addition, we introduce a post-training protocol that fine-tunes pretrained video generators using a modest amount of action-observation data drawn from the same action space as the downstream tasks, which allows us to examine their adaptation potential and to characterize a data scaling law.
this section cite: ['b12', 'b76', 'b23', 'b17', 'b99', 'b46', 'b52', 'b88']

Section: Task Success Rate (%)
SVD
World-In-World offers a fair, closed-loop world interface to evaluate diverse WMs. We benchmark leading video generators (Wan et al., 2025;HaCohen et al., 2024;Kong et al., 2024) alongside task-focused world models (Bar et al., 2025;Koh et al., 2023;2021) in perception, navigation, and manipulation settings. Our findings reveal three consistent trends: (1) high visual quality does not necessarily translate into strong task success; (2) scaling post-training with action-observation data is more effective than upgrading the pretrained video generators; and (3) increasing inference-time compute via online planning substantially improves closed-loop performance. As shown in Figure 2, world models with strong visual scores do not necessarily bring high success rates, which underscores the need for closed-loop evaluation when judging WM practical value for embodied agents.
Our work makes three main contributions: The revision policy π revision (❹) evaluates all rollouts and commits to the best, yielding decision D ⋆ t . This decision is applied in the environment, closing the interaction loop.
this section cite: ['b76', 'b29', 'b47', 'b46']

Section: UNIFIED STRATEGY FOR CLOSED-LOOP ONLINE PLANNING
In Figure 3, we present a unified closed-loop strategy that uses visual world models for decisionmaking. It cycles through proposal, simulation, and revision. In proposal, the agent generates candidate plans; in simulation, each plan is rolled out by the world model to predict counterfactual futures; in revision, the agent scores rollouts and refines its plan. Finally, the agent executes the top-scoring plan in the environment, coupling model-based planning with real execution.
Let o t denote the agent's egocentric observation at time step t. 1 Define the agent's future potential action sequence of horizon L starting at time step t as Ât = ât+1 , ât+2 , . . . , ât+L , where each elementary action â is specified in either a continuous action space or a discrete action space, i.e., â ∈ V, with V denoting the set of action primitives available to the agent.
Our unified strategy can be formalized as a policy-guided beam search. The beam width corresponds to the number of candidate plans M drawn from the proposal policy π proposal . At time step t, given the current observation o t and the task goal g, the proposal policy π proposal samples M candidate action sequences that serve as future candidate plans:
Â(m) t ∼ π proposal A o t , g , m = 1, . . . , M.(1)
Each candidate plan Â(m) t is subsequently transformed by the unified action API C into the control inputs expected by the world model: I (2)
Then, the candidate plans and their simulated rollouts Â(m) t , Ô(m) t are evaluated and revised by the revision policy π revision , which assigns a score to each trajectory and selects the decision that maximizes the expected reward. In the most general form, we write
D ⋆ t = π revision { ( Â(m) t , Ô(m) t ) } M m=1 , o t , g .(3)
Here, D ⋆ t denotes the best decision according to π revision at time step t. Depending on the task, D ⋆ t may represent a high-level answer, a recognition result, or a refined sequence of low-level actions, which renders the framework more general than classical Model Predictive Control (MPC) (Morari & H. Lee, 1999), where optimization is typically restricted to sequences of actions.
A common instantiation implements π revision as a score-and-select operator S. When the decision is an action sequence, selection is performed over the M candidate plans produced at time step t:
D ⋆ t = Â(m ⋆ ) t , where m ⋆ = arg max m∈{1,...,M } S Â(m) t , Ô(m) t o t , g .(4)
Here, S(•) denotes a task-specific scoring function that estimates the expected reward or utility of a candidate plan based on its simulated outcomes. Alternatively, π revision may synthesize or update a new decision by aggregating information across the candidate set and their predicted consequences, rather than selecting one candidate verbatim.
Once the best decision D ⋆ t is executed in the environment, the agent acquires a new observation at time step t+1. The unified strategy then re-enters the proposal-simulation-revision loop, using the newly observed state to initiate the next round of proposal, simulation, and revision. In our framework, both π proposal and π revision can be instantiated flexibly: they may be pretrained modules, such as large-scale vision-language models or diffusion policies, or simple rule-based heuristics.
In our experiments, we explore multiple instantiations to systematically explore the flexibility and generality of our framework for different tasks.
this section cite: ['b59']

Section: UNIFIED ACTION API
In this section, we present a unified action API that transforms an action sequence A into control inputs I that guide the world model, i.e., I = C(A). The action API is designed to be flexible so that the same interface can serve a wide range of world models and tasks. It supports three principal types of control information: (1) text prompt, (2) camera trajectory/viewpoint, and (3) low-level actions, depending on the inputs expected by the chosen world model.
Text prompt. For image-and-text-to-video world models, the controller maps the intended action sequence into a descriptive text prompt. A predefined template converts each primitive action into a phrase, and concatenating these phrases yields the final prompt I text .
Camera trajectory / viewpoint. For models that consume explicit viewpoints, the controller translates A into a camera trajectory, e.g., each translation action moves the camera by 0.2 m, and each rotation action changes the azimuth by 22.5 • . The resulting trajectory is represented as a sequence (x k , y k , ϕ k ) K k=1 with (x k , y k ) ∈ R 2 and azimuth ϕ k ∈ R. Low-level actions. For world models that take discrete or continuous low-level actions as input, the controller maps the action sequence A to the world model's action vocabulary, yielding A world . This mapping A → A world applies the necessary transformations to maintain a unique and consistent correspondence between the agent's actions and the inputs expected by the world model.
this section cite: []

Section: COMPREHENSIVE EMBODIED TASKS
To evaluate the practical utility of visual world models in embodied tasks, we select a diverse set of tasks that span multiple domains and stress distinct capabilities. We focus on four representative tasks: Active Recognition (AR), Active Embodied Question Answering (A-EQA), Image-Goal Navigation (ImageNav), and Robotic Manipulation, as illustrated in Figure 4. Taken together, these tasks emphasize complementary aspects of embodied intelligence, including perception, navigation, and object-level manipulation, and thus provide a comprehensive testbed for assessing how effectively a visual world model supports online planning and decision-making. Below, we describe the tasks included in our benchmark, and more detailed settings are provided in Appendix B.
this section cite: []

Section: Image-Goal Navigation
What is the target object bounded by the red box? Step 1: <Front> view Step 2: <Front> view Navigate as needed and Identify the object marked by the red bbox. Navigate as needed and answer the user's <Query>. How many cushions are on the red sofa? Step 1: <Front> view <Goal Image> Step 1: <Front> view Step 1 Step 2 Active Recognition Active Embodied QA Navigate to the location from which the <Goal Image> was captured.
this section cite: []

Section: Image-Goal Navigation

this section cite: []

Section: Robotic Manipulation
Use the robotic arm to slide the red block onto the blue target. Active Recognition (AR) is closely related to amodal recognition (Aydemir et al., 2013;Liu et al., 2018;Yang et al., 2019;Fan et al., 2024;Bhattacharjee et al., 2025), in which the agent must identify a designated target that may be observed from extreme viewpoints or be heavily occluded.
In addition, AR allows the agent to acquire additional observations through active exploration. All AR experiments are conducted in the Habitat-Sim (Savva et al., 2019), encompassing 551 episodes across 29 scenes from the validation split of Matterport3D (Chang et al., 2017). Within AR, the visual world model assists two decision-making processes. For answering, synthetic views provide auxiliary evidence that helps the agent reason about occlusions and extreme viewpoints that impede recognition. For navigation, rollouts simulate the consequences of potential actions so that the agent can choose a path that is more likely to yield informative observations.
Image-Goal Navigation (ImageNav), also referred to as goal-conditioned visual navigation, requires an embodied agent to reach a target position in a scene given a single reference image that specifies the goal viewpoint. We construct 144 ImageNav episodes from 87 validation scenes of HM3D (Ramakrishnan et al., 2021). In this task, the visual world model exclusively supports navigation decisions. The agent simulates the outcomes of candidate action plans, selects the best option, executes the first segment of that plan, and then replans with the newly observed state in a closed-loop manner.
Active Embodied Question Answering (A-EQA) requires an agent to answer open-ended naturallanguage questions after actively exploring a 3D environment. Our evaluation set includes 184 questions across 54 indoor scenes from the official OpenEQA split (Majumdar et al., 2024) and the HM3D validation set (Ramakrishnan et al., 2021). As in AR, the visual world model supports both question answering and navigation. For answering, synthetic views generated by the world model provide complementary perspectives that help resolve references to occluded or distant objects. For navigation, the agent simulates high-level action plans using the world model's predictions to choose exploration strategies likely to reveal question-relevant information.
Robotic Manipulations are fundamental capabilities for embodied agents that must operate in realworld interaction settings. We study how visual world models contribute to closed-loop manipulation planning, evaluating performance on four RLBench (James et al., 2020) tasks with 50 episodes per task. In our setting, the visual world model supports the agent in assessing candidate 7-DoF gripper actions by providing visual evidence about anticipated object motions and interactions, which enables a comparison of alternative plans before execution. The predicted outcomes then guide the selection of actions that are more likely to achieve the specified objective, thereby linking visual prediction accuracy to improvements in manipulation performance.
this section cite: ['b4', 'b53', 'b89', 'b22', 'b9', 'b69', 'b15', 'b62', 'b57', 'b62', 'b37']

Section: EXPLOITING WORLD MODELS VIA POST-TRAINING
To evaluate the feasibility of adapting pretrained video generators for embodied tasks, we introduce a post-training procedure that aligns a pretrained model with the domain distribution and action space of target environments. We perform fine-tuning separately on data from two simulators, Habitat-Sim and CoppeliaSim, to match the corresponding task domains. For Habitat-Sim tasks (AR, A-EQA, ImageNav), we post-train on a panoramic action-observation dataset collected from the HM3D (Ramakrishnan et al., 2021) training split. For CoppeliaSim tasks (Robotic Manipulation), we post-train on task demonstrations generated with RLBench (James et al., 2020). To assess generalization rather than memorization, all Habitat-Sim data used for post-training are sourced from scenes that are disjoint from our evaluation scenes, so the scenes in our evaluation tasks remain unseen by the world models after post-training. Additional details regarding the training objective, dataset construction, and training configuration are provided in Appendices C and D.
this section cite: ['b37']

Section: EVALUATION RESULTS AND ANALYSIS
In this section, we report quantitative results and key observations on the four embodied tasks in Section 3.1, followed by ablation studies in Section 3.2. We evaluate visual world models
this section cite: []

Section: BENCHMARK RESULTS
World models can enhance the performance of the base proposal policy. Across AR, A-EQA, ImageNav, and Manipulation, adding a visual world model consistently improves the performance of the base proposal policy (e.g., a VLM policy, a heuristic policy, or a 3D diffusion policy), as shown in Tables 1 to 3. For example, in AR, the best proprietary model (Runway Gen4) attains an accuracy of 64.79% while reducing the mean steps per episode to 4.06, compared to the VLM base policy with an accuracy of 50.27% and mean steps 6.24. Similarly, in ImageNav, the best open-source model Wan2.1 † achieves a success rate of 45.14% with an average path length of 45.8, outperforming the VLM base policy at 35.42% SR and 47.5 average length. In A-EQA, the top post-trained model Wan2.2 † A14B reaches an answer score of 48.4 and SPL of 31.9, surpassing the VLM base policy at 45.7 answer score and 29.6 SPL. These results support the effectiveness of our World-In-World online planning framework with world models, in which the world model provides simulated future states that inform better decisions.
World models struggle to simulate precise motion and dynamics in manipulation. The gains are less pronounced for Robotic Manipulations (Table 3), likely because accurately modeling contact-rich interactions and robot kinematics is significantly more challenging than predicting purely view changes. For instance, the best post-trained model on manipulation (SVD †) reaches an SR of 46.5% with a mean trajectory length of 2.38, only modestly above the VLM baseline at 44.5% SR and 2.52 mean length. This gap suggests that while current visual world models can effectively guide perception and navigation, capturing fine-grained physical dynamics and action-conditioned object motion remains an open challenge. 400 4K 40K 80K Seen Examples During Training 52 53 54 55 56 57 58 59 60 61 62 63 64 Success Rate (%) 60.25% 61.52% 62.61% 63.34% 56.26% 56.44% 60.98%60.98% Wan2.2 Wan2.1 SVD 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 Avg Inference Count per Episode
52 53 54 55 56 57 58 59 60 61 62 63 64 Success Rate (%) 53.36% 56.44% 57.17% 60.98% 56.62% 58.26% 59.71% 62.61% Wan2.1 SVD Post-training substantially boosts world-model utility. Our post-training adaptation yields consistent improvements. Relative to off-the-shelf Wan2.1, Wan2.1 † raises AR accuracy from 58.26% to 62.61% and ImageNav SR from 38.19% to 45.14% (Table 1). Likewise, SVD † improves AR accuracy from 57.71% to 60.98% and ImageNav SR from 40.28% to 43.05%. In A-EQA, LTX-Video † increases the answer score from 46.6 to 48.6, and Wan2.1 † from 45.7 to 48.2. These gains show that aligning the generative model to the target domain and action space of the specific embodied tasks improves downstream decision-making.
this section cite: []

Section: ABLATION AND FINDINGS
Fine-grained controllability matters more than visuals for task success. Although recent off-theshelf video generators like Wan2.1 produce visually appealing clips, they are driven by text prompts with limited fine-grained low-level controls. Without adaptation, these models yield only small gains on downstream embodied tasks. We further study the relation between controllability and the success rate on AR. Here, controllability is defined as alignment between intended actions and the motions in the model's predictions. After action-conditioned post-training, alignment improves substantially and SR rises accordingly. Figure 5(b) shows a clearer positive correlation than Figure 5(a), which depicts SR versus generation quality (aesthetic and image-quality scores), and suggests that models that respond reliably to low-level controls achieve higher SR. These results indicate that precise control, not just visual quality, is critical for embodied world models to support effective decision-making.
Data-size scaling for post-trained models. We study how post-training data size affects WM performance (Wan2.2 †, Wan2.1 †, SVD †). Each WM is post-trained for one epoch on datasets from 400 to 80K instances. As shown in Figure 6, more post-training data consistently improves AR performance: Wan2.1 † rises from 60.25% to 63.34%, and SVD † from 56.80% to 60.98%. Wan2.2 † (A14B), despite substantially larger web-video pretraining, reaches nearly the same performance as Wan2.1 † after 40K post-training instances, suggesting that scaling action-conditioned post-training is more effective for embodied utility than upgrading the pretrained generator. Moreover, larger models (Wan2.1 †, 14B) benefit more and saturate less than smaller ones (SVD †, 1.5B), indicating greater capacity to absorb action-conditioned supervision.
Inference-time scaling for online planning with world models. Within our online planning framework, the number of world-model inferences (simulated potential futures per episode) directly affects task performance. As shown in Figure 7, increasing the average inferences per episode for AR yields a clear positive correlation with SR. For example, increasing the average inference count from 3 to 11 improves SR from 53.36% to 60.98% for SVD †. This suggests that allocating more inference-time computation to simulate potential futures lets the planner make more informed decisions, thereby improving overall performance. Table 5: Effect of world-model augmentation and revision policy on ImageNav. SR and SPL are higheris-better; mean trajectory length is lower-is-better. πproposal WM Type πrevision SR ↑ Mean Traj. ↓ SPL ↑ VLM None None 35.42 47.5 25.88 VLM SVD † VLM 43.05 46.0 30.96 VLM Wan2.1 † VLM 45.14 45.8 32.10 VLM SVD † LPIPS 47.92 41.3 39.82 VLM Wan2.1 † LPIPS 48.61 39.8 42.48 Effect of different revision policies. We study how the revision policy affects task performance by comparing a VLM-based revision policy with a simple LPIPS-based policy that selects the candidate whose predicted observation is closest to the goal image in perceptual feature space. From Table 5, we see that even a simple LPIPS-based revision policy could improve the performance significantly: SVD † obtains 47.92% SR and 39.82 SPL compared with 43.05% SR and 30.96 SPL using a VLM-based revision policy and 35.42% SR and 25.88 SPL without any WM augmentation. Augmenting the planner with action-conditioned WMs and applying a simple LPIPS-based revision can yield a higher SR and more efficient navigation.
Table 6: Cross-domain post-training: WMs post-trained on HSSD or HM3D and evaluated on HM3D/MP3D (val) for AR and ImageNav. WM Aug. Post-Train Env. AR ImageNav SR ↑ Mean Traj. ↓ SR ↑ SPL ↑ w/o WM None 50.27 6.24 35.42 25.88 +SVD † HSSD 58.98 5.24 38.89 27.60 +Wan2.1 † HSSD 62.98 4.78 42.36 31.18 +SVD † HM3D (train) 60.98 5.02 43.05 30.96 +Wan2.1 † HM3D (train) 62.61 4.73 45.14 32.10
this section cite: []

Section: Domain transfer across scene distributions.
We evaluate cross-domain generalization by posttraining WMs on the synthetic Habitat Synthetic Scenes Dataset (HSSD) and testing them on our AR and ImageNav suites built on the real-world scenes in HM3D/MP3D (Table 6). Despite the synthetic-toreal gap, HSSD-trained WMs still yield clear gains over the VLM-only baseline (e.g., SVD † improves AR SR from 50.27% to 58.98% and ImageNav SR from 35.42% to 38.89%). Performance remains below in-domain post-training on HM3D (SVD †: 60.98% AR SR, 43.05% ImageNav SR), as expected under a stronger distribution shift. These results indicate that post-training learns action-conditioned visual representations that transfer across scene distributions, consistent with prior work on adaptable world models (Gao et al., 2025).
Generalization capacity of world models is critical for practical use. Most video generators are pretrained on web videos. In unseen embodied environments, they may revert to training priors or ignore action controls, yielding plausible but physically or semantically inconsistent rollouts (see Figures 13 and 14). These deviations mislead planning and reduce success. Larger models or more pretraining data can partly help, but robust generalization remains central. Future work should prioritize strategies and action representations to improve transfer to novel environments, such as unified action representations (Gao et al., 2025;Wang et al., 2025f;Zhi et al., 2025;Wang et al., 2025e) and curriculum or domain-specific data collection (Zhao et al., 2025).
Long-horizon planning with world models remains challenging. In our experiments, visual world models simulate short-term changes but struggle on long horizons due to limited mechanisms for accumulating spatiotemporal history. We attempted to alleviate this issue by replacing front-view inputs with panoramas to provide global context, but gains were inconsistent across models and tasks. Future work should better encode and retrieve long-term dependencies, e.g., spatial memory (Zhou et al., 2025b;Xiao et al., 2025;Li et al., 2025d;Yu et al., 2025a;Ren et al., 2025;Wang et al., 2025c) and episode-level memory (Cai et al., 2025;Guo et al., 2025), to maintain scene-level context and enable coherent planning over extended horizons.
Precise modeling of interactions and dynamics remains difficult. For manipulation, capturing contact-rich interactions, compliance, friction, and state changes of articulated or deformable objects is essential. Current visual world models often miss these details, producing rollouts that violate physics and degrade planning and control-consistent with our observations and prior analyses (Kang et al., 2024;Li et al., 2025a). Promising directions include physics-guided motion generation (Wang et al., 2025a;Zhang et al., 2025b;Akkerman et al., 2025), inferring or generating physical properties to inform action-conditioned predictions (Cao et al., 2025;Gillman et al., 2025;Zhang et al., 2024b), and physics-aware reinforcement post-training (Wu et al., 2025;Liu et al., 2025). Integrating such signals into conditioning pathways may improve fidelity when precise dynamics are required.
Stronger proposal and revision policies set the performance floor. The agent's overall performance depends on both world-model fidelity and the strength of the proposal and revision policies that select and refine decisions. While simulated rollouts improve decision-making, base policies must be effective to provide a reliable starting point, and strengthening them raises the ceiling. Future work could explore stronger policies (Geng et al., 2025;Kim et al., 2025), and integration strategies that deepen synergy between world models and decision-making (Neary et al., 2025), such as more human-aligned reward models (Wang et al., 2024;Seneviratne et al., 2025;Rocamonde et al., 2023;Zhang et al., 2024a;Wang et al., 2025d;Wu et al., 2025).
Computational cost and efficiency remain practical concerns. Incorporating world models into model-based planning introduces additional computational overhead because multiple future rollouts must be simulated at each decision step. Although our experiments show that allocating more inference-time computation to the world model improves task performance, this extra cost may be impractical in settings with strict real-time constraints or limited hardware resources. Future work should therefore investigate more efficient world-model architectures (Yang et al., 2025b;Kodaira et al., 2025), training and inference strategies that enable near real-time rollouts (Huang et al., 2025;Cui et al., 2025), and distillation techniques (Wang et al., 2025b;Agarwal et al., 2025) that reduce computational demands while preserving the predictive fidelity of world models.
this section cite: ['b111', 'b109', 'b86', 'b13', 'b28', 'b39', 'b92', 'b92', 'b102', 'b2', 'b14', 'b27', 'b85', 'b54', 'b26', 'b42', 'b60', 'b84', 'b70', 'b65', 'b85', 'b44', 'b35', 'b18', 'b102', 'b0']

Section: CONCLUSION
We introduce World-In-World, a closed-loop world interface and benchmark that evaluates generative world models via embodied interaction rather than isolated visual metrics. By unifying heterogeneous controls, our action API enables any world model to serve as perception and planning utilities for an embodied agent. Coupled with a unified closed-loop planning strategy that proposes, simulates, and revises action plans, the benchmark measures agent performance on four demanding tasks. Our experiments reveal large gaps between visual metrics and task success, underscoring the need for closed-loop evaluation, and show that pretrained video generators improve with post-training data scaling and inference-time scaling. We expect World-In-World to guide world models toward not only striking visual realism but also reliable perception and planning in embodied scenarios.
this section cite: []

Section: References
Ref_id:b0 Title: Cosmos world foundation model platform for physical ai Year: (2025)
Ref_id:b1 Title: Aesthetic predictor v2 Year: (2024-05)
Ref_id:b2 Title: Interdyn: Controllable interactive dynamics with video diffusion models Year: (2025)
Ref_id:b3 Title: Diffusion for world modeling: Visual details matter in atari Year: ()
Ref_id:b4 Title: Active visual object search in unknown environments using uncertain semantics Year: (2013-08)
Ref_id:b5 Title: Tc4d: Trajectory-conditioned text-to-4d generation Year: (2024)
Ref_id:b6 Title: Text-to-4d generation using hybrid score distillation sampling Year: (2024)
Ref_id:b7 Title:  Year: (2025)
Ref_id:b8 Title: Navigation world models Year: ()
Ref_id:b9 Title: Believing is seeing: Unobserved object detection using generative models Year: (2025-03)
Ref_id:b10 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b11 Title: Align your latents: High-resolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b12 Title: Sora: Video generation models as world simulators Year: (2024)
Ref_id:b13 Title: Mixture of contexts for long video generation Year: (2025)
Ref_id:b14 Title: Physx-3d: Physical-grounded 3d asset generation Year: (2025)
Ref_id:b15 Title: Matterport3d: Learning from rgb-d data in indoor environments Year: (2017-10)
Ref_id:b16 Title: Yolo-world: Real-time open-vocabulary object detection Year: (2024)
Ref_id:b17 Title: Domain-free generation of 3d gaussian splatting scenes Year: (2023)
Ref_id:b18 Title: Self-forcing++: Towards minute-scale high-quality video generation Year: (2025)
Ref_id:b19 Title: Learning universal policies via text-guided video generation Year: (2023)
Ref_id:b20 Title: Video language planning Year: ()
Ref_id:b21 Title: Worldscore: A unified evaluation benchmark for world generation Year: (2025)
Ref_id:b22 Title: Evidential active recognition: Intelligent and prudent open-world embodied perception Year: (2024)
Ref_id:b23 Title: Scenescape: Text-driven consistent scene generation Year: (2023)
Ref_id:b24 Title: Vista: A generalizable driving world model with high fidelity and versatile controllability Year: (2024-11)
Ref_id:b25 Title: Learning adaptable world models with latent actions Year: ()
Ref_id:b26 Title: Roboverse: Towards a unified platform, dataset and benchmark for scalable and generalizable robot learning Year: (2025)
Ref_id:b27 Title: Force prompting: Video generation models can learn and generalize physics-based control signals Year: (2025)
Ref_id:b28 Title: Long context tuning for video generation Year: (2025)
Ref_id:b29 Title: Ltx-video: Realtime video latent diffusion Year: (2024)
Ref_id:b30 Title: Cameractrl: Enabling camera control for text-to-video generation Year: (2025)
Ref_id:b31 Title: Cameractrl ii: Dynamic scene exploration via camera-controlled video diffusion models Year: (2025)
Ref_id:b32 Title: Matrix-game 2.0: An open-source, real-time, and streaming interactive world model Year: (2025)
Ref_id:b33 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b34 Title: Gaia-1: A generative world model for autonomous driving Year: (2023-09)
Ref_id:b35 Title: Self forcing: Bridging the train-test gap in autoregressive video diffusion Year: (2025)
Ref_id:b36 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b37 Title: Rlbench: The robot learning benchmark & learning environment Year: (2020)
Ref_id:b38 Title: Rednet: Residual encoder-decoder network for indoor rgb-d semantic segmentation Year: (2018)
Ref_id:b39 Title: How far is video generation from world model: A physical law perspective Year: (2024)
Ref_id:b40 Title: Musiq: Multi-scale image quality transformer Year: (2021)
Ref_id:b41 Title: Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations Year: (2024)
Ref_id:b42 Title: Fine-tuning vision-language-action models: Optimizing speed and success Year: (2025)
Ref_id:b43 Title: Learning to act from actionless videos through dense correspondences Year: (2023)
Ref_id:b44 Title: Streamdit: Real-time streaming text-to-video generation Year: (2025)
Ref_id:b45 Title: Pathdreamer: A world model for indoor navigation Year: ()
Ref_id:b46 Title: Simple and effective synthesis of indoor 3d scenes Year: (2023-06)
Ref_id:b47 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b48 Title: Pisa experiments: Exploring physics post-training for video diffusion models by watching stuff drop Year: ()
Ref_id:b49 Title: Worldmodelbench: Judging video generation models as world models Year: ()
Ref_id:b50 Title: Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition Year: ()
Ref_id:b51 Title: Vmem: Consistent interactive video scene generation with surfel-indexed view memory Year: (2025)
Ref_id:b52 Title: Scenethesis: A language and vision agentic framework for 3d scene generation Year: (2025)
Ref_id:b53 Title: Extreme trust region policy optimization for active object recognition Year: (2018-06)
Ref_id:b54 Title: Improving video generation with human feedback Year: (2025)
Ref_id:b55 Title: A survey: Learning embodied intelligence from physical simulators and world models Year: (2025)
Ref_id:b56 Title: Generative world explorer Year: ()
Ref_id:b57 Title: Embodied question answering in the era of foundation models Year: (2024)
Ref_id:b58 Title: World models that know when they don't know: Controllable video generation with calibrated uncertainty Year: (2025)
Ref_id:b59 Title: Model predictive control: Past, present and future Year: (1999-05)
Ref_id:b60 Title: Improving pre-trained vision-language-action policies with model-based search Year: (2025)
Ref_id:b61 Title: Genie 3: A new frontier for world models Year: (2025-08)
Ref_id:b62 Title: ): 1000 large-scale 3d environments for embodied ai Year: (2021-08)
Ref_id:b63 Title: Segment anything in images and videos Year: (2024)
Ref_id:b64 Title: 3d-informed world-consistent video generation with precise camera control Year: (2025)
Ref_id:b65 Title: Vision-language models are zero-shot reward models for reinforcement learning Year: (2023)
Ref_id:b66 Title: High-resolution image synthesis with latent diffusion models Year: ()
Ref_id:b67 Title: Introducing runway gen Year: (2025-03)
Ref_id:b68 Title: Zeronvs: Zero-shot 360-degree view synthesis from a single image Year: (2024)
Ref_id:b69 Title: Habitat: A platform for embodied ai research Year: (2019)
Ref_id:b70 Title: Halo: Human preference aligned offline reward learning for robot navigation Year: (2025)
Ref_id:b71 Title: Genwarp: Single image to novel views with semantic-preserving generative warping Year: (2024-11)
Ref_id:b72 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b73 Title: Conformal prediction for uncertainty-aware planning with diffusion dynamics model Year: (2023)
Ref_id:b74 Title: A control-centric benchmark for video prediction Year: (2023)
Ref_id:b75 Title: SV3D: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion Year: ()
Ref_id:b76 Title: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b77 Title: Physctrl: Generative physics for controllable and physics-grounded video generation Year: (2025)
Ref_id:b78 Title: Dreamwalker: Mental planning for continuous vision-language navigation Year: ()
Ref_id:b79 Title: Videoscene: Distilling video diffusion model to generate 3d scenes in one step Year: ()
Ref_id:b80 Title: Evoworld: Evolving panoramic world generation with explicit 3d memory Year: (2025)
Ref_id:b81 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b82 Title: Latent policy steering with embodiment-agnostic pretrained world models Year: (2025)
Ref_id:b83 Title: Precise action-to-video generation through visual action prompts Year: (2025)
Ref_id:b84 Title: Rl-vlm-f: Reinforcement learning from vision language foundation model feedback Year: (2024)
Ref_id:b85 Title: Rewarddance: Reward scaling in visual generation Year: (2025)
Ref_id:b86 Title: Worldmem: Long-term consistent world simulation with memory Year: (2025-04)
Ref_id:b87 Title: Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency Year: (2024-07)
Ref_id:b88 Title: Comp4d: Llm-guided compositional 4d scene generation Year: (2024)
Ref_id:b89 Title: Embodied amodal recognition: Learning to move to perceive objects Year: (2019)
Ref_id:b90 Title: Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v Year: (2023)
Ref_id:b91 Title: Learning interactive real-world simulators Year: (2023)
Ref_id:b92 Title: Embodiedbench: Comprehensive benchmarking multi-modal large language models for vision-driven embodied agents Year: (2025-02)
Ref_id:b93 Title: Learning interactive real-world simulators Year: ()
Ref_id:b94 Title: Longlive: Real-time interactive long video generation Year: ()
Ref_id:b95 Title: Test-time scaling with world models for spatial reasoning Year: (2025)
Ref_id:b96 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b97 Title: Foundational interactive video generation Year: (2025)
Ref_id:b98 Title: Dragnuwa: Finegrained control in video generation by integrating text, image, and trajectory Year: (2023)
Ref_id:b99 Title: Interactive 3d scene generation from a single image Year: (2024)
Ref_id:b100 Title: Long-term photometric consistent novel view synthesis with diffusion models Year: (2023)
Ref_id:b101 Title: Context as memory: Scene-consistent interactive long video generation with memory retrieval Year: ()
Ref_id:b102 Title: Creating new games with generative interactive videos Year: (2025-01)
Ref_id:b103 Title: COMBO: Compositional world models for embodied multi-agent cooperation Year: ()
Ref_id:b104 Title: Candidate pseudolabel learning: enhancing vision-language models by prompt tuning with unlabeled data Year: (2024)
Ref_id:b105 Title: Think before you diffuse: Llms-guided physics-aware video generation Year: (2025)
Ref_id:b106 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b107 Title: Physdreamer: Physics-based interaction with 3d objects via video generation Year: ()
Ref_id:b108 Title: Improving generalizability and undetectability for targeted adversarial attacks on multimodal pre-trained models Year: (2025)
Ref_id:b109 Title: Synthetic video enhances physical fidelity in video synthesis Year: (2025)
Ref_id:b110 Title: Learning 4d embodied world models Year: (2025)
Ref_id:b111 Title: 3dflowaction: Learning cross-embodiment manipulation from 3d flow world model Year: (2025)
Ref_id:b112 Title: Stable virtual camera: Generative view synthesis with diffusion models Year: (2025-04)
Ref_id:b113 Title: Learning 3d persistent embodied world models Year: (2025)
Ref_id:b114 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
