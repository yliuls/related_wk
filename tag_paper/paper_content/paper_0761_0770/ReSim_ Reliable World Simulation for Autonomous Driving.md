Title: ReSim: Reliable World Simulation for Autonomous Driving
Abstract: How can we reliably simulate future driving scenarios under a wide range of ego driving behaviors? Recent driving world models, developed exclusively on real-world driving data with expert trajectories, struggle to represent hazardous or non-expert behaviors that are rare in training corpus. This limitation restricts their applicability to tasks such as policy evaluation. In this work, we address this challenge by enriching real-world human demonstrations with diverse non-expert data collected from a driving simulator (e.g., CARLA), and building a controllable world model trained on this heterogeneous corpus. Starting with a video generator featuring a diffusion transformer architecture, we devise several strategies to effectively integrate conditioning signals and improve prediction controllability and fidelity. The resulting model, ReSim, enables Reliable Simulation of diverse openworld driving scenarios under various actions, including hazardous non-expert ones. To close the gap between high-fidelity simulation and applications that require reward signals to judge different actions, we introduce a Video2Reward module that estimates a reward from ReSim's simulated future. Our ReSim paradigm achieves up to 44% higher visual fidelity, improves controllability for both expert and non-expert actions by over 50%, and boosts planning and policy selection performance on NAVSIM by 2% and 25%, respectively.

Section: Introduction
Learning a world model capable of predicting plausible future outcomes is now envisioned as a key milestone in achieving autonomy [1,2,3]. Over the past decade, researchers have leveraged visual world models to learn compact representations [4,5], guide test-time planning [6,7,8], and develop reinforcement learning agents [9,10] across various domains [11,12,13]. Unlike general-purpose video generators, which prioritize visual fidelity and generalization, world models simulate futures with precise control over ego actions.
In the autonomous driving domain, recent driving world models have also made rapid improvements in visual fidelity and generalization by scaling to massive driving datasets [14,15,16] and integrating frontier video generation techniques [17,18]. However, the ability to accurately follow actions, which is an essential requirement for precise reward estimation and effective planning [19,20], remains challenging [21]. As real-world data continues to grow, a critical question emerges: Is real-world human data alone sufficient to guarantee simulation reliability? A notable limitation of real-world data is that it predominantly consists of safe expert demonstrations, where the state-action space is inherently restricted by safety and regulation concerns [22,23]. Consequently, safety-critical or … 84.6 LAW 66.8 Avg. PDMS PDMS Figure 1: Overview of ReSim. (a) Heterogeneous driving data includes (i,ii) experts' safe driving logs, and (iii) potentially dangerous (non-expert) driving behaviors from simulations. (b)
Prior driving world models are trained on expert data solely, leading to consistently safe yet inaccurate imaginations; in ReSim, we leverage all sources of data to simulate reliable and realistic futures, and build a robust reward model that generalizes to open-world scenarios within the simulator. (c) The high-fidelity prediction, accurate action-following, and reward estimation abilities of ReSim facilitate driving applications related to both policy deployment and simulation.
hazardous events (e.g., collisions, off-road deviations) are significantly underrepresented [24,25,26]. This imbalance leads to severe hallucinations when the world model is exposed to unseen non-expert actions in certain states, undermining its robustness and reliability [27,28].
To address the problem, we present ReSim, a reliable driving world model that can be steered by various actions, including out-of-distribution ones, while achieving high-fidelity simulation results. Our approach first enriches real-world human driving logs with non-expert data gathered from a driving simulator [29], where agents can execute a broader spectrum of actions without safety concerns. The resulting training corpus illustrated in Fig. 1(a) covers a wide spectrum of scenarios and actions (including non-expert ones), and further supports the simulation reliability for our world model. Built upon a scalable text-to-video generator [30], ReSim applies a multi-stage training pipeline to integrate visual and action conditions. We devise an unbalanced noise sampling strategy along with a dynamics consistency loss to emphasize the learning of motion coherence, especially when being applied to non-expert actions with significant visual changes. As showcased in Fig. 1(b), prior works like Vista [16] fail to follow the specified steering action, while ReSim accurately simulates off-road behaviors. Moreover, making the world model beneficial for real-world driving often requires reward estimation [1,10,31,8], which judges the quality of different actions to guide decisions. Therefore, we develop a Video2Reward model to convert simulated video outputs from ReSim into scalar rewards in real-world scenarios.
Based on the above explorations, we further demonstrate applications for supporting real-world autonomous driving in various aspects, as depicted in Fig. 1(c). In scenarios where action conditioning is absent, the simulated future of ReSim can serve as a visual plan from which an executable ego trajectory can be derived. On the NAVSIM planning benchmark [23], with front-view sensory videos only, our video prediction-based policy achieves an improvement of +2.0 compared to state-of-the-art world model-based planners [32] and +2.6 compared to an end-to-end baseline [33] with supervised learning. Additionally, the integration of ReSim and the Video2Reward model offers a solution for selecting the trajectory with the highest estimated reward among those generated by candidate policies, thereby justifying and guiding the final decision. In our experiments, this policy selection process leads to a performance boost of 55.3% in comparison to the weak candidate policies. More intuitively, our system offers a synthetic environment where we can validate the behavior of a learned driving policy by running it within the imagination of ReSim in a closed-loop manner.
Contributions. (1) While prior works either use simulated or real-world data separately to develop driving world models, we demonstrate that integrating both sources can alleviate the shortage of unsafe driving behaviors in real-world data, and can improve the model's action controllability in real-world scenarios. (2) We present ReSim, a controllable world model that reliably simulates highfidelity future outcomes by precisely executing diverse action inputs, together with a comprehensive training recipe including an improved loss formulation and noise sampling strategy for incorporating condition inputs and capturing scenario dynamics. Rewards can be derived from the simulated futures via a Video2Reward model. (3) We applying ReSim to facilitate driving in real-world scenarios, and validate its effectiveness via benchmarking on a wide array of datasets and tasks, where it exhibits evident improvements over previous counterparts.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b15', 'b0', 'b9', 'b30', 'b7', 'b22', 'b31', 'b32']

Section: Reliable Driving World Simulation
We outline the ReSim framework as follows. In Sec. 2.1, we introduce the heterogeneous training data with a wide range of scenarios and actions. We instantiate ReSim on a diffusion transformer architecture with careful modifications to capture dynamic driving scenarios and enable accurate action conditioning in Sec. 2.2. We propose to derive rewards from the simulated results of ReSim via a Video2Reward model in Sec. 2.3. Furthermore, we demonstrate the applicability of our method in real-world driving applications in Sec. 2.4. More implementation details are in Appendix Sec. B.
this section cite: []

Section: Heterogeneous Data Compilation
Existing driving world models are typically developed on public autonomous driving datasets with expert trajectories [34,35] and web videos [14]. Similarly, in this work, we compile these two sources, specifically NAVSIM [23] and OpenDV [14], within our training data. NAVSIM contains rigorously labeled actions for action control learning, while the large-scale OpenDV dataset supports generalization of the world model. However, as shown in Fig. 1(a), both sources are dominated by human behaviors. The lack of non-expert actions limits prior world models' ability to emulate nonexpert behaviors and their corresponding outcomes, such as collisions. This issue further hinders the world models from effectively identifying an inferior driving policy and providing reliable rewards.
To address this limitation, we leverage a driving simulator, i.e., CARLA [29], to gather data in synthetic environments, enabling exploration without the costs and risks associated with the physical world. Notably, although simulated data has been adopted for world models in synthetic environments [2,10,36,37], such a source is overlooked in driving world models that operate in real scenarios [15,14,16,38]. World models trained in simulation alone struggle to generalize to real world due to the significant visual gap between two worlds. Our data collection is conducted in CARLA with randomly sampled routes from Bench2Drive settings [39]. Two types of agents are deployed within the environments. One uses a well-established driving policy, PDM-Lite [40,41], to collect expert executions, while the other adopts an exploration strategy whereby both the steering angle and the speed are randomly sampled from predefined sets to generate non-expert behavior data, which is underrepresented in human data yet a crucial component for training reliable world models. As a result, the numbers of video samples for each dataset are 4M for OpenDV, 85K for NAVSIM and 88K for CARLA. Each video sample is 4.9s long with a frequency of 10Hz, where the first 9 frames are visual context and the last 40 frames are prediction targets during training. See more data collection details in appendix B.1.
this section cite: ['b33', 'b34', 'b13', 'b22', 'b13', 'b28', 'b1', 'b9', 'b35', 'b36', 'b14', 'b13', 'b15', 'b37', 'b38', 'b39', 'b40']

Section: Controllable World Model
Basics. ReSim is built on CogVideoX [30], a high-capacity diffusion transformer originally conditioned only on text. To enable visual-context and trajectory conditioning, we replace the denoiser's historical latent inputs with their clean counterparts (following Vista [16]) and project future ego waypoints via a learnable encoder into the transformer's input space alongside video latents. The model is supervised by the following video diffusion loss:
L diffusion = E x,ϵ,t ∥x k: -D θ (x t ; t, h, c[, a]) k: ∥ 2 ,(1)
where x is the clean video latent, and x t is the noised video latent constructed by imposing a randomly sampled noise ϵ on x at a diffusion timestep t. The diffusion transformer D θ is conditioned on latents of historical frames h, a high-level text command c (e.g., "Turn left"), and a fine-grained action a which is a sequence of future ego waypoints. To focus on forecasting the future, the diffusion loss is applied to the latent from the k-th frame onward only, excluding the observed history.
this section cite: ['b29', 'b15']

Section: Dynamics Consistency Loss.
So far, the standard video diffusion loss (Eq. ( 1)) supervises each video frame independently, which overlooks temporal correlations in videos, resulting in inferior spatiotemporal coherence and realism [42,43,16]. To address this, we introduce a dynamics consistency loss to additionally supervise the "latent motion", the discrepancy of video latent across different timestep ranges. This loss forces predicted motion to match the ground truth. We compute this loss over multiple intervals to capture both short-term and long-term dynamics. The intuition behind this is that some agent behaviors, e.g., yielding, are hard to capture in the short term. To stabilize the magnitude of the loss value, we further normalize this loss by a factor of s, which is the average value of absolute motion disparity for each interval. This loss is formulated as:
L dynamics = E x,ϵ,t K j=1 N -j i=1 1 s ∥(d i+j -d i ) -(x i+j -x i )∥ 2 ,(2)
where x and d are the ground-truth and model-predicted video latent respectively, and i indexes the frame of the video latent. K is the maximum timestep intervals considered for latent motion, which is set to 4 in our experiments. N is the number of frames of video latent. The total loss for training the world model is the combination of video diffusion loss and dynamics consistency loss: L = L diffusion + λL dynamics , where λ is set to 0.1 empirically. Note that both L diffusion and L dynamics are applied to the video latent compressed by the video VAE [30]. Therefore, the indices and number of frames in Eq. ( 1) and Eq. ( 2) correspond to the video latent representation instead of the raw video.
this section cite: ['b41', 'b42', 'b15', 'b29']

Section: Unbalanced Noise Sampling.
The behavior of a diffusion model is largely influenced by how we sample noise during training [44,45], which controls how much noise is injected into the input data for the denoiser to recover [46,47]. When applying commonly-used uniform noise sampling as in [30], we empirically find that our world model underperforms on complex driving dynamics, especially when we consider rare and non-expert behaviors. The issue behind this is that uniform timestep sampling lets models take a "shortcut" on low-noise diffusion timesteps where the model can recover the injected video noise by simply averaging information in adjacent frames, instead of learning critical motion details, which degrades the dynamics fidelity in generated driving videos [48].
To force the model to capture complex agent-environment interactions, we bias sampling toward higher-noise steps. We increase the frequency of drawing timesteps in [500, 1000] from 1 /2 to 2 /3, thereby amplifying input corruption and compelling richer dynamics learning.
Progressive Multi-stage Learning. We adapt CogVideoX [30], originally pretrained with text-only conditioning, into an controllable driving world model via a three-stage curriculum. 1) We first endow it with the ability to predict futures that follow historical visual context and text commands, by training on OpenDV [14]. 2) Next, we incorporate NAVSIM [23] and CARLA [29] with annotated actions for joint training with OpenDV. Action conditions, i.e., future ego waypoints, are encoded through a learnable transformer. Notably, NAVSIM trajectories are randomly masked (p = 0.5) to support both action-conditioned and free prediction, while CARLA waypoints remain intact to guide hazardous maneuvers that cannot be directly inferred from visual context. To prioritize structural dynamics over high-frequency details while improving training efficiency, we downsample inputs to 256×448, freeze the diffusion backbone, and fine-tune only the trajectory encoder and a LoRA adapter [49]. 3) After the effective adaptation of action conditions, we finally resume the model training on 512×896 resolution with full fine-tuning, producing a model that generates 4s of 10Hz video conditioned on nine frames at 10Hz, an optional command, and a 4s, 2Hz waypoint sequence.
this section cite: ['b43', 'b44', 'b45', 'b46', 'b29', 'b47', 'b29', 'b13', 'b22', 'b28', 'b48']

Section: Reward Estimation from Video
To accomplish a feedback loop, world models need to estimate a reward to assess the predicted futures [1,10], which is largely overlooked in prior driving world models [38,14,15]. Among the few attempts, the lack of explicit goal states [19] in open-world driving and the complexity of outdoor scenarios make manual reward crafting challenging [31]. To overcome this, our key insight is to use the widely adopted simulator CARLA [29] as a rich source to learn rewards from via the unified video interface, as depicted in Fig. 2. Such a formulation offers several notable advantages. First, the driving simulator allows flexible exploration and can produce extensive data with environmental feedback to learn from. This includes not only successful driving experiences, but also non-expert mistakes and edge cases, covering a wide distribution of reward ranges. Second, contrary to constructing rewards manually with 3D perception models [31], the video interface does not require highly crafted 3D priors such as camera poses, and thus can benefit from a broad range of frontier vision models with strong cross-domain generalization [50,51]. Training Inference Rule-based Policy Learned Policy Infraction Score Figure 2: Video2Reward model (V2R). Top: V2R is supervised by infraction score of both safe and hazardous data from simulation, deriving the reward from a driving video. Bottom: In real-world inference, the predicted video of ReSim in reaction to a proposed action is fed into V2R to estimate the action's reward.
In detail, our Video2Reward model (V2R) is established on a frozen DINOv2 backbone [50] with an additional lightweight prediction head. Supervised by the CARLA infraction score [52,29] that comprehensively penalizes multiple factors such as collisions and excessively low speeds, V2R learns to estimate the reward from video sequences. During inference, we send a planned trajectory produced by any policy to ReSim to simulate future video, which is then sent to V2R to estimate the reward of that trajectory. Due to the highly generalizable visual features of the DINOv2 backbone and the realistic prediction of ReSim, V2R is readily applicable to real-world driving scenarios, effectively assessing the quality of diverse behaviors.
this section cite: ['b0', 'b9', 'b37', 'b13', 'b14', 'b18', 'b30', 'b28', 'b30', 'b49', 'b50', 'b49', 'b51', 'b28']

Section: Applications

this section cite: []

Section: Video Prediction-based Policy.
From the future prediction capability learned from massive human driving videos at scale, ReSim implicitly learns how the ego vehicle should behave and can be converted into a video prediction-based policy, akin to recent approaches in robotics [53,3,54]. As opposed to solely imitating the ego trajectory, predicting future observations allows for utilizing a broader source of unlabeled video data while leveraging richer supervision, including the intention of surrounding agents that are not captured in sparse trajectory-based outputs. To serve as a policy for deployment, ReSim takes historical visual observations and a high-level command as input and imagines the unseen future images, without conditioning on actions (which should be the output of this task). After visual imagination, the predicted future frames of ReSim are fed into an inverse dynamics model (IDM) that converts it into a future trajectory of the ego vehicle. Illustrative samples are shown in Fig. 3, where critical events for ego planning are highlighted in dashed boxes.   Reward-guided Policy Selection. Complex driving environments often necessitate the maintenance of multiple candidate proposals to ensure planning robustness across various scenarios [55]. While it is straightforward to obtain multiple trajectory proposals from different policies for the same scenario, it raises the question of how to reconcile these diverse outputs. To address this, we propose to apply our method to score each trajectory with a reward and select the one with the highest reward for execution. Concretely, each candidate trajectory is rendered into a short predictive video using ReSim, and the resulting video is then passed through Video2Reward model to obtain its reward. Guided by the estimated reward, the trajectory selection process results in a steered policy with significant improvement over individual policy candidates, by leveraging their advantages in different situations.
this section cite: ['b52', 'b2', 'b53', 'b54']

Section: Closed-loop Visual Simulation.
Vision-based driving agents are primarily evaluated in an openloop manner, either on static datasets against pre-recorded trajectories [34,35] or simulation-based benchmarks that consider local interactions [23]. Both these evaluation types confine agents to safe and human-driven scenarios. More seriously, they overlook error accumulations over extended rollouts and fail to reflect the closed-loop performance as in real-world driving, where agents would be continuously exposed to new states after taking actions. Owing to its precise action controllability and high visual fidelity, we can leverage ReSim to simulate visual states in a closed-loop manner. In each iteration, ReSim executes the predicted action of the driving agent to generate the next visual state, which is then input to the agent to make decisions for the next iteration.
this section cite: ['b33', 'b34', 'b22']

Section: Experiments
In this section, we first evaluate ReSim's simulation reliability, specifically relating to its action controllability, video prediction fidelity, and reasonableness of the reward formulation (Sec. 3.1). Next, we validate ReSim's applicability to real-world driving tasks (Sec. 3.2). Finally, we present ablation studies on data and methodological designs to verify their effectiveness (Sec. 3.3).
this section cite: []

Section: Results of Simulation Reliability
Results of Action Controllability. We verify the action controllability of ReSim on the unseen Waymo Open dataset [35]. For action-free and expert action conditioning, we follow the protocol of Vista [16] and use the Trajectory Difference metric to assess how closely the world model's predicted future aligns with the input trajectory. As reported in Tab. 1, ReSim improves the results by 80% and 54% for both conditioning modes compared to Vista. Moreover, removing the simulated data from training (ReSim w/o sim.) results in a performance decrease for both conditioning modes. This evaluation is conducted on a random subset of the Waymo validation set with 540 samples.
For non-expert action conditioning, we conduct a human preference study among samples generated by different methods conditioned on non-expert actions. As reported in Fig. 4, ReSim outperforms baselines by a large margin for both visual realism and trajectory following. We also make qualitative comparisons between different methods in Fig. 5, where ReSim yields more reliable and realistic results that align with the non-expert trajectory input. Moreover, the learned action controllability can be transferred to unseen datasets in a zero-shot manner, as showcased in Fig. 6.
this section cite: ['b34', 'b15']

Section: Comparison of Video Prediction Fidelity.
The fidelity of video prediction is a key indicator of a driving world model's ability to simulate realistic scenarios. As presented in Tab. 2, we evaluate the
ReSim Reliable & Realistic Scenario Inconsistent ReSim* Not Following Trajectory Vista Waymo nuScenes Expert Act. Non-expert Act. Non-expert Act. Expert Act. performance of various driving world models with FID [56] and FVD [57] metrics on nuScenes [34] validation set. The evaluation protocol follows Vista [16], using only context frames as conditions without imposing explicit action control. Notably, without training on any nuScenes data samples, ReSim yields significantly better results in a zero-shot manner compared to in-distribution models. We also provide qualitative comparisons for long-term future prediction in Appendix Sec. C, where Vista's prediction becomes oversaturated and loses semantics of the scene, while ReSim remains predicting visually rich future states in 30s.
this section cite: ['b55', 'b56', 'b33', 'b15']

Section: Results of Reward Estimation.
To evaluate the effectiveness of our reward formulations, we measure the ability of each reward model to distinguish "expert" from "non-expert" trajectories via a reward correlation metric. Specifically, for both CARLA [29] and NAVSIM [23], we randomly sample successful episodes with expert trajectories and accompany each with a randomly drawn trajectory from other samples that is potentially unsafe and assumed as non-expert. Evaluation is conducted with 250 pairs of com-  parative samples for the reward model to judge. Reward models are expected to assign higher scores to expert trajectories compared to non-expert ones for the same scenario. Results in Fig. 7 validate the advantage of our method, which surpasses its counterparts in both simulated and real-world datasets.
this section cite: ['b28', 'b22']

Section: Results of Applications
Video Prediction-based Policy. We evaluate the performance of our method on the navtest split of NAVSIM [23] benchmark. Specifically, we separately train an Inverse Dynamics Model (IDM) on the NAVSIM training set to convert the predicted video sequence of ReSim to an executable ego trajectory. As reported in Tab. 4, coupling ReSim and the lightweight IDM produces a video prediction-based policy that outperforms both end-to-end baselines (UniAD and Transfuser) and world model counterparts (DrivingGPT) by a non-trivial margin. Notably, our method only requires the history observations and a high-level command as input, without accessing multiple sensors, ego status, past trajectory, or extra annotations like other methods. The Visual Odometry (VO) planner shares the same architecture as our IDM yet performs poorly, underscoring ReSim's guidance.
this section cite: ['b22']

Section: Reward-guided Policy Selection.
We compare different strategies for selecting an action from two candidate policies, i.e., Transfuser and LTF [33]. The evaluation is conducted on a subset of NAVSIM, by selecting 300 challenging scenarios where one of the candidate policies fails while the other succeeds according to PDMS metric. As shown in Tab. 3, when applied separately, Transfuser and LTF achieve PDMS of 47.7 and 47.2, respectively. A uniform average ensemble lifts performance to 66.8, while the Vista reward only reaches 59.2. Instead, applying our reward strategy by composing ReSim and Video2Reward achieves a PDMS of 74.1, which is the closest score compared to the oracle selection according to ground-truth PDMS, and is higher than all baselines including our alternative (ours w/o sim.) that removes simulated data from the training of ReSim.
this section cite: ['b32']

Section: Closed-loop Visual Simulation.
As showcased in Fig. 8, we leverage ReSim to iteratively simulate visual feedback for a running policy starting from two NAVSIM [23] scenarios. At each iteration, ReSim simulates an entire 4s future simultaneously by executing the action (i.e., future trajectory for 4s) output by the policy. The newly generated frames are then fed into the policy for the subsequent Uniform Sampling Unbalanced Sampling Figure 9: Effect of unbalanced noise sampling. Training with unbalanced noise sampling yields improved motion and scenario consistency. W/O DCL W/ DCL, K=1 W/ DCL, K=4 Figure 10: Effect of dynamics consistency loss (DCL). Applying DCL with K = 4 (in Eq. ( 2)) works best.
decision. We opt for a lightweight Visual Odometry-based planner adopted from XVO [60] as the policy, since it only takes front-view video as input. Attributed to the generative rollout, ReSim position the policy into states that are never encountered in a pre-recorded dataset.
this section cite: ['b22', 'b59']

Section: Ablation Study
Effect of Simulated Data. Throughout our experiments, we demonstrate that training with simulated data improves results across multiple tasks. For action controllability, removing CARLA simulation data leads to inferior results for both expert (Tab. 1) and non-expert actions (Fig. 4). Without simulated data, the synthesized future may be inconsistent in the scenario's structure when conditioned on non-expert actions, as showcased in Fig. 5. Simulated data also contributes to more accurate reward estimates as shown in Fig. 7, which further benefits reward-guided policy selection (Tab. 3).
this section cite: []

Section: Effect of Unbalanced Noise Sampling.
As shown in Fig. 9, applying unbalanced noise sampling during training makes the predicted future more consistent in terms of agents' motion and scenario layout, compared to the baseline with uniform noise sampling.
this section cite: []

Section: Effect of Dynamics Consistency Loss.
We visualize the effect of applying our proposed dynamic consistency loss in Fig. 10. The qualitative results verify that incorporating the loss and extending the maximum interval K for latent motion extraction (in Eq. ( 2)) yield more coherent future predictions.
this section cite: []

Section: Conclusion and Outlook
In this paper, we present ReSim, a reliable driving world model that excels in simulating a diverse range of actions in open-world scenarios. We incorporate non-expert data with hazardous actions from an established driving simulator to enrich real-world human driving data that primarily consists of safe behaviors. We also integrate several new training strategies, including a dynamics consistency loss, unbalanced noise sampling, and multi-stage learning. To facilitate driving applications beyond visual simulation, a Video2Reward model is devised to estimate the reward from the simulated future.
Extensive experiments demonstrate the effectiveness and versatility of our ReSim system.
Limitation and Future Works. We envision our work as an early glimpse at open-world simulation with reward feedback, a cornerstone in establishing robust intelligence in the unstructured physical world. However, our system is still bottlenecked by inference efficiency due to iterative denoising, and how to train agents within the synthesized world produced by ReSim is yet to be discovered. Future work focused on enhancing the efficiency, developing reinforced agents with the world model, and constructing fair closed-loop planning benchmarks would propel us closer to this goal. A discussion of limitations and broader impact of our work is included in Appendix Sec. D.
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We will release all code and models.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: References
Ref_id:b0 Title: A path towards autonomous machine intelligence Year: (2022)
Ref_id:b1 Title: Recurrent world models facilitate policy evolution Year: (2018)
Ref_id:b2 Title: Video as the new language for real-world decision making Year: (2024)
Ref_id:b3 Title: Self-supervised learning from images with a joint-embedding predictive architecture Year: (2023)
Ref_id:b4 Title: Policy pre-training for autonomous driving via self-supervised geometric modeling Year: (2023)
Ref_id:b5 Title: Temporal difference learning for model predictive control Year: (2022)
Ref_id:b6 Title: World models on pre-trained visual features enable zero-shot planning Year: (2024)
Ref_id:b7 Title: Navigation world models Year: (2025)
Ref_id:b8 Title: Diffusion for world modeling: Visual details matter in atari Year: (2024)
Ref_id:b9 Title: Dream to Control: Learning behaviors by latent imagination Year: (2020)
Ref_id:b10 Title: Diffusion models are real-time game engines Year: (2024)
Ref_id:b11 Title: Pathdreamer: A world model for indoor navigation Year: ()
Ref_id:b12 Title: Learning interactive real-world simulators Year: (2024)
Ref_id:b13 Title: Generalized predictive model for autonomous driving Year: (2024)
Ref_id:b14 Title: GAIA-1: A generative world model for autonomous driving Year: (2023)
Ref_id:b15 Title: Vista: A generalizable driving world model with high fidelity and versatile controllability Year: (2024)
Ref_id:b16 Title: DriveDreamer-2: LLM-enhanced world models for diverse driving video generation Year: (2024)
Ref_id:b17 Title: GEM: A generalizable ego-vision multimodal world model for fine-grained ego-motion, object dynamics, and scene composition control Year: (2025)
Ref_id:b18 Title: A control-centric benchmark for video prediction Year: (2023)
Ref_id:b19 Title: Learning adaptable world models with latent actions Year: (2025)
Ref_id:b20 Title: ACT-Bench: Towards action controllable world models for autonomous driving Year: (2024)
Ref_id:b21 Title: Is ego status all you need for open-loop end-to-end autonomous driving Year: (2024)
Ref_id:b22 Title: NAVSIM: Data-driven non-reactive autonomous vehicle simulation and benchmarking Year: (2024)
Ref_id:b23 Title: Learning to drive from a world on rails Year: (2021)
Ref_id:b24 Title: Rates of motor vehicle crashes, injuries and deaths in relation to driver age Year: (2014)
Ref_id:b25 Title: ActiveAD: Planningoriented active learning for end-to-end autonomous driving Year: (2024)
Ref_id:b26 Title: How far is video generation from world model: A physical law perspective Year: (2024)
Ref_id:b27 Title: SimGen: Simulator-conditioned driving scene generation Year: (2024)
Ref_id:b28 Title: CARLA: An open urban driving simulator Year: (2017)
Ref_id:b29 Title: CogVideoX: Text-to-video diffusion models with an expert transformer Year: (2025)
Ref_id:b30 Title: Driving into the Future: Multiview visual forecasting and planning with world model for autonomous driving Year: (2024)
Ref_id:b31 Title: Enhancing end-to-end autonomous driving with latent world model Year: ()
Ref_id:b32 Title: Trans-Fuser: Imitation with transformer-based sensor fusion for autonomous driving Year: (2008)
Ref_id:b33 Title: nuScenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b34 Title: Scalability in perception for autonomous driving: Waymo open dataset Year: (2020)
Ref_id:b35 Title: Mastering atari with discrete world models Year: (2021)
Ref_id:b36 Title: Mastering diverse domains through world models Year: (2023)
Ref_id:b37 Title: DriveDreamer: Towards real-world-driven world models for autonomous driving Year: (2024)
Ref_id:b38 Title: Bench2Drive: Towards multi-ability benchmarking of closed-loop end-to-end autonomous driving Year: (2024)
Ref_id:b39 Title: DriveLM: Driving with graph visual question answering Year: (2024)
Ref_id:b40 Title: PDM-Lite: A rule-based planner for carla leaderboard 2 Year: (2024)
Ref_id:b41 Title: Track4Gen: Teaching video diffusion models to track points improves video generation Year: (2025)
Ref_id:b42 Title: MotiF: Making text count in image animation with motion focal loss Year: (2025)
Ref_id:b43 Title: Stable Video Diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b44 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b45 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b46 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b47 Title: Snap Video: Scaled spatiotemporal transformers for text-to-video synthesis Year: (2024)
Ref_id:b48 Title: LoRA: Low-rank adaptation of large language models Year: ()
Ref_id:b49 Title: DINOv2: Learning robust visual features without supervision Year: (2024)
Ref_id:b50 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b51 Title: CARLA autonomous driving leaderboard Year: (2022)
Ref_id:b52 Title: Learning universal policies via text-guided video generation Year: (2023)
Ref_id:b53 Title: GR-2: A generative video-language-action model with web-scale knowledge for robot manipulation Year: (2024)
Ref_id:b54 Title: PiP: Planning-informed trajectory prediction for autonomous driving Year: (2020)
Ref_id:b55 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b56 Title: Towards Accurate Generative Models of Videos: A new metric & challenges Year: (2018)
Ref_id:b57 Title: DriveGAN: Towards a controllable high-quality neural simulation Year: (2021)
Ref_id:b58 Title: WoVoGen: World volume-aware diffusion for controllable multi-camera driving scene generation Year: (2024)
Ref_id:b59 Title: XVO: Generalized visual odometry via cross-modal self-training Year: (2023)
Ref_id:b60 Title: Planning-oriented autonomous driving Year: (2023)
Ref_id:b61 Title: DrivingGPT: Unifying driving world modeling and planning with multi-modal autoregressive transformers Year: (2024)
Ref_id:b62 Title: Pre-training contextualized world models with in-the-wild videos for reinforcement learning Year: (2023)
Ref_id:b63 Title: Visual Foresight: Model-based deep reinforcement learning for vision-based robotic control Year: (2018)
Ref_id:b64 Title: Deep visual foresight for planning robot motion Year: (2017)
Ref_id:b65 Title: Learning latent dynamics for planning from pixels Year: (2019)
Ref_id:b66 Title: Generative interactive environments Year: (2024)
Ref_id:b67 Title: Structured world models from human videos Year: (2023)
Ref_id:b68 Title:  Year: (2024)
Ref_id:b69 Title: FlatFusion: Delving into details of sparse transformer-based camera-lidar fusion for autonomous driving Year: (2024)
Ref_id:b70 Title: Interleave-VLA: Enhancing robot manipulation with interleaved image-text instructions Year: (2025)
Ref_id:b71 Title: Learning from all vehicles Year: (2022)
Ref_id:b72 Title: Curse of rarity for autonomous vehicles Year: (2024)
Ref_id:b73 Title: Trajectory-guided control prediction for end-to-end autonomous driving: A simple yet strong baseline Year: (2022)
Ref_id:b74 Title: Think Twice before Driving: Towards scalable decoders for end-to-end autonomous driving Year: (2023)
Ref_id:b75 Title: DriveAdapter: Breaking the coupling barrier of perception and planning in end-to-end autonomous driving Year: (2023)
Ref_id:b76 Title: DriveTransformer: Unified transformer for scalable end-to-end autonomous driving Year: (2025)
Ref_id:b77 Title: The arcade learning environment: An evaluation platform for general agents Year: (2013)
Ref_id:b78 Title:  Year: (2020)
Ref_id:b79 Title: ViZDoom: A doom-based ai research platform for visual reinforcement learning Year: (2016)
Ref_id:b80 Title: IDE-Net: Interactive driving event and pattern extraction from human data Year: (2021)
Ref_id:b81 Title: Multi-agent trajectory prediction by combining egocentric and allocentric views Year: (2022)
Ref_id:b82 Title: Towards capturing the temporal dynamics for trajectory prediction: a coarse-to-fine approach Year: (2023)
Ref_id:b83 Title: HDGT: Heterogeneous driving graph transformer for multi-agent trajectory prediction via scene encoding Year: (2023)
Ref_id:b84 Title: AMP: Autoregressive motion prediction revisited with next token prediction for autonomous driving Year: (2024)
Ref_id:b85 Title: Model-based imitation learning for urban driving Year: (2022)
Ref_id:b86 Title: Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2) Year: (2024)
Ref_id:b87 Title: Raw2Drive: Reinforcement learning with aligned world models for end-to-end autonomous driving Year: (2025)
Ref_id:b88 Title: A multimodal world model for autonomous driving via unified bev latent space Year: (2024)
Ref_id:b89 Title: MUVO: A multimodal generative world model for autonomous driving with geometric representations Year: (2023)
Ref_id:b90 Title: Visual point cloud forecasting enables scalable autonomous driving Year: (2024)
Ref_id:b91 Title: Learning unsupervised world models for autonomous driving via discrete diffusion Year: (2024)
Ref_id:b92 Title: OccWorld: Learning a 3D occupancy world model for autonomous driving Year: (2024)
Ref_id:b93 Title: Gaussian world model for streaming 3D occupancy prediction Year: (2024)
Ref_id:b94 Title: Bench2Drive-R: Turning real world data into reactive closed-loop autonomous driving benchmark by generative model Year: (2024)
Ref_id:b95 Title: LLM4Drive: A survey of large language models for autonomous driving Year: (2023)
Ref_id:b96 Title: DriveMoE: Mixture-of-experts for vision-language-action model in end-to-end autonomous driving Year: (2025)
Ref_id:b97 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b98 Title: DrivingDojo Dataset: Advancing interactive and knowledge-enriched driving world model Year: (2024)
Ref_id:b99 Title: BEVFormer: Learning bird's-eye-view representation from multi-camera images via spatiotemporal transformers Year: (2022)
Ref_id:b100 Title: MapTR: Structured modeling and learning for online vectorized hd map construction Year: (2023)
Ref_id:b101 Title: 3D object detection from images for autonomous driving: a survey Year: (2023)
Ref_id:b102 Title: SDXL: Improving latent diffusion models for high-resolution image synthesis Year: (2024)
Ref_id:b103 Title: Align your latents: High-resolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b104 Title: AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2024)
Ref_id:b105 Title: Latent Video Diffusion Models for High-Fidelity Long Video Generation Year: (2022)
Ref_id:b106 Title: GenTron: Diffusion transformers for image and video generation Year: (2024)
Ref_id:b107 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b108 Title: Open source computer vision library Year: (2015)
Ref_id:b109 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b110 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b111 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b112 Title:  Year: (2017)
Ref_id:b113 Title: VideoGPT: Video generation using vq-vae and transformers Year: (2021)
Ref_id:b114 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b115 Title: ST-P3: End-to-end vision-based autonomous driving via spatial-temporal feature learning Year: (2022)
Ref_id:b116 Title: End-to-end autonomous driving: Challenges and frontiers Year: (2024)
Ref_id:b117 Title: Video prediction policy: A generalist robot policy with predictive visual representations Year: (2024)
Ref_id:b118 Title: Accelerating video diffusion models via distribution matching Year: (2024)
Ref_id:b119 Title: From slow bidirectional to fast causal video generators Year: (2024)
Ref_id:b120 Title: Learning to drive from a world model Year: (2025)
Ref_id:b121 Title: Robust autonomy emerges from self-play Year: (2025)
Ref_id:b122 Title: VAD: Vectorized scene representation for efficient autonomous driving Year: (2023)
Ref_id:b123 Title: nuPlan: A closed-loop ml-based planning benchmark for autonomous vehicles Year: (2021)
