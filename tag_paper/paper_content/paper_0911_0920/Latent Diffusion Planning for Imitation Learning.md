Title: Latent Diffusion Planning for Imitation Learning
Abstract: Recent progress in imitation learning has been enabled by policy architectures that scale to complex visuomotor tasks, multimodal distributions, and large datasets. However, these methods often rely on learning from large amount of expert demonstrations. To address these shortcomings, we propose Latent Diffusion Planning (LDP), a modular approach consisting of a planner which can leverage action-free demonstrations, and an inverse dynamics model which can leverage suboptimal data, that both operate over a learned latent space. First, we learn a compact latent space through a variational autoencoder, enabling effective forecasting of future states in image-based domains. Then, we train a planner and an inverse dynamics model with diffusion objectives. By separating planning from action prediction, LDP can benefit from the denser supervision signals of suboptimal and action-free data. On simulated visual robotic manipulation tasks, LDP outperforms state-of-the-art imitation learning approaches, as they cannot leverage such additional data. 1

Section: Introduction
Combining large-scale expert datasets and powerful imitation learning policies has been a promising direction for robot learning. Recent methods using transformer backbones or diffusion heads (Octo Model Team et al., 2024;Kim et al., 2024;Zhao et al., 2024;Chi et al., 2023) have capitalized on new robotics datasets pooled together from many institutions (Khazatsky et al., 2024;Open X-Embodiment Collaboration et al., 2023), showing potential for learning generalizable robot policies. However, this recipe is fundamentally limited by expert data, as robotics demonstration data can be challenging, time-consuming, and expensive to collect. While it is often easier to collect in-domain data that is suboptimal or action-free, these methods are not designed to use such data, as they rely on directly modeling optimal actions.
Prior works in offline RL, reward-conditioned policies, or imitation learning from suboptimal demonstrations attempt to leverage suboptimal trajectories, though they are still unable to utilize action-free data. Notably, these works often make restrictive assumptions like access to either reward labels (Kumar et al., 2020;Chen et al., 2021;Kumar et al., 2019a), the optimality of demonstrations (Beliaev et al., 2022), or similar metrics (Zhang et al., 2022), which can be impractical or noisy to label. Other works implicitly attempt to use unlabelled, suboptimal data via pretraining on such data and later fine-tuning the policy on the optimal data (Radosavovic et al., 2023;Wu et al., 2023b;Cui et al., 2024). While these approaches can potentially learn representations during pretraining, it does not necessarily improve planning capabilities of these methods.
Our key idea is to take a modular approach, where we separate learning a video planner from learning an inverse dynamics model. Each one of these two components can leverage different types of data. For instance, a planner can benefit from action-free data, while an IDM can leverage unlabelled suboptimal data. While using a modular approach has been proposed in recent prior work (Du et al., 2023a;Black et al., 2023), prior approaches focus on highlevel decision making by forecasting subgoals, limiting its capabilities for closed loop re-planning in robotics tasks. These works also operate across images, which are highdimensional and expensive to generate. To create an efficient modular approach that can benefit from all forms of data (suboptimal, action-free, and optimal), we propose learning the planner and inverse-dynamics model over a learned, compact latent space, allowing for closed-loop robot policies. Our imitation learning objective consists of forecasting a dense trajectory of latent states, scaling up gracefully to vision-based domains without the computational complexities of video generation.
We propose Latent Diffusion Planning (LDP), which learns a planner that can be trained on action-free data; and an inverse dynamics model (IDM) that can be trained on data that may be suboptimal. First, it trains a variational autoencoder with an image reconstruction loss, producing compressed latent embeddings that are used by the planner and inverse dynamics model. Then, it learns an imitation learning policy through two components: (1) a planner, which consumes demonstration state sequences, which may be action-free, and (2) an inverse dynamics model, trained on in-domain, possibly suboptimal, environment interactions. As diffusion objectives have proven to be effective for imitation learning in robotics tasks (Chi et al., 2023), we use diffusion for both forecasting plans (planner) and extracting actions (IDM), which enables competitive performance. Our method is closed-loop and reactive, as planning over latent space is much faster than generating visually and physically consistent video frames.
In summary, our main contributions are threefold:
• We propose a novel imitation learning algorithm, Latent Diffusion Planning, a simple, diffusion planning-based method comprised of a learned visual encoder, latent planner, and an inverse dynamics model.
• We show that Latent Diffusion Planning can be trained on suboptimal or action-free data, and improves from learning on such data in the regime where demonstration data is limited.
• We experimentally show that our method outperforms prior video planning-based work by leveraging temporally dense predictions in a latent space, which enables fast inference for closed-loop planning.
this section cite: ['b26', 'b63', 'b39', 'b25', 'b32', 'b7', 'b2', 'b61', 'b43', 'b10', 'b3', 'b39']

Section: Related Work
Imitation Learning in Robotics. One common approach to learning robot control policies is imitation learning, where policies are learned from expert-collected demonstration datasets. This is most commonly done via behavior cloning, which reduces policy learning to a supervised learning objective of mapping states to actions. Recently, Diffusion Policy (Chi et al., 2023) and Action Chunking with Transformers (Zhao et al., 2023) have shown successful results in complex manipulation tasks using action chunking and more expressive architectures. Diffusion models have also been successful in capturing multimodal human behavior (Pearce et al., 2023). Similarly, Behavior Transformer (Shafiullah et al., 2022) and VQ-BeT (Lee et al., 2024) improve the ability of policies to capture multimodal behaviors. In this work, we focus on forecasting a sequence of future states instead of actions, and use diffusion to capture multimodal trajectories.
this section cite: ['b39', 'b62', 'b41', 'b48', 'b33']

Section: Learning from Unlabelled Suboptimal and Action-Free Data.
Learning from suboptimal data has long been a goal of many robot learning methods, including reinforcement learning. A typical approach is offline reinforcement learning, which considers solving a Markov decision process from an offline dataset of states, actions, and reward (Levine et al., 2020;Kumar et al., 2020;Kostrikov et al., 2021;Hansen-Estruch et al., 2023;Yu et al., 2022). Particularly relevant are the approaches that use supervised learning conditioned on rewards (Schmidhuber, 2019;Kumar et al., 2019a;Chen et al., 2021). In this work, we want to leverage suboptimal, reward-free data, such as play data or failed trajectories. In addition, we would like to avoid the additional complexity of annotating the data with rewards or training a value function which the offline RL methods rely on.
Several works have also addressed learning from actionfree data, such as using inverse models (Torabi et al., 2018;Baker et al., 2022), latent action models (Edwards et al., 2019;Schmeckpeper et al., 2020;Bruce et al., 2024), or representation learning (Radosavovic et al., 2023;Wu et al., 2023b;Cui et al., 2024). In this work we focus on a simple recipe for robotic imitation learning that is naturally able to leverage action-free data through state forecasting.
Diffusion and Image Prediction in Robot Learning. Diffusion models, due to their expressivity and training and sampling stability, have been applied to robot learning tasks. Diffusion has been used in offline reinforcement learning (Hansen-Estruch et al., 2023) and imitation learning (Chi et al., 2023). Diffuser (Janner et al., 2022) learns a denoising diffusion model on trajectories, including both states and actions, in a model-based reinforcement learning setting. Decision Diffuser (Ajay et al., 2023) extends Diffuser by showing compositionality over skills, rewards, and constraints, and instead diffuses over states and uses an inverse dynamics model to extract actions from the plan. Due to the complexity of modeling image trajectories, Diffuser and Decision Diffuser restrict their applications to low-dimensional states.
To scale up to diffusing over higher-dimensional plans, UniPi (Du et al., 2023a;Ko et al., 2023) adapts video models for planning. Unlike works that rely on foundation models and video models for planning (Du et al., 2023b;Yang et al., 2024;Zhou et al., 2024), our method avoids computational and modeling complexities of generative video modeling by planning over latent embeddings instead.
Previous works have used world models to plan over images in a compact latent space (Hansen et al., 2024;Hafner et al., 2019;2020). In contrast with these works, we focus on single task imitation instead of reinforcement learning.
Many prior works argue that state forecasting objectives are uniquely suitable for robotics to improve planning quality with trajectory optimization or reinforcement learning (Finn & Levine, 2017;Yang et al., 2023), by using the model directly to plan future states (Du et al., 2023b;a), as well as representation learning (Wu et al., 2023a;Radosavovic et al., 2023). We follow this line of work by proposing a planningbased method competitive to state-of-the-art robotic imitation learning that can leverage heterogeneous data sources.
this section cite: ['b34', 'b32', 'b29', 'b20', 'b60', 'b47', 'b7', 'b51', 'b1', 'b14', 'b46', 'b6', 'b43', 'b10', 'b20', 'b39', 'b24', 'b0', 'b28', 'b58', 'b64', 'b19', 'b17', 'b15', 'b57', 'b43']

Section: Background
Diffusion Models Diffusion models, such as Denoising Diffusion Probabilistic Models (DDPMs), are likelihoodbased generative models that learn an iterative denoising process from a Gaussian prior to a data distribution (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2020). During training time, DDPMs are trained to reverse a single noising step. Then, at sampling time, to reverse the diffusion process, the model iteratively denoises a sample drawn from the known Gaussian prior.
Diffusion models may also be conditioned on additional context. For example, text-to-image generative models are conditioned on text, Diffusion Policy is conditioned on visual observations, and Decision Diffuser can be conditioned on reward, skills, and constraints.
Recent generative models have used Latent Diffusion Models, which trains a diffusion model in a learned, compressed latent space (Rombach et al., 2022;Peebles & Xie, 2023;Blattmann et al., 2023) to improve computational and memory efficiency. The latent space is typically learned via an autoencoder, with encoder E and decoder D trained to re-
construct x ≈ x ˆ= D(E(x)). Instead of diffusing over x, the diffusion model is trained on diffusing over z = E(x).
Imitation Learning In the imitation learning framework, we assume access to a dataset of expert demonstrations, D ≜ {(s 0 , x 0 , a 0 ), . . . , (s T , x T , a T )}, generated by π E , an expert policy. s i , x i , a i correspond to the state, image, and action at timestep i respectively. The imitation learning objective is to extract a policy π ˆ(a|s, x) that most closely imitates π E . In robotics, this is typically approached through behavior cloning, which learns the mapping between states and actions directly via supervised learning. We consider single-task imitation, where the dataset corresponds to a single task.
Diffusion Policy (Chi et al., 2023) is an instantiation of diffusion models for imitation learning that has shown success in simulated and real-world tasks. Diffusion Policy uses a DDPM objective to model the distribution of action sequences, conditioned on observations. The CNN instantiation uses a Conditional U-Net Architecture, based on the 1D Temporal CNN in (Janner et al., 2022), which encourages temporal consistency due to the inductive biases of convolutions. LDP's planner architecture is based on the CNN-based Diffusion Policy, though we forecast latent states instead of actions.
Datasets of expert demonstrations often do not provide sufficient state distribution coverage to effectively solve a given task with imitation learning. However, there often exists additional data in the form of action-free or suboptimal data, which may consist of failed policy rollouts, play data, or miscellaneous environment interactions. Unfortunately, behavior cloning assumes access to data annotated with optimal actions, so such additional data cannot be easily incorporated into training.
this section cite: ['b49', 'b22', 'b50', 'b45', 'b42', 'b4', 'b39', 'b24']

Section: Latent Diffusion Planning
Latent Diffusion Planning consists of three parts, as shown in Figure 1: (1) Training an image encoder via an image reconstruction loss, (2) learning an inverse dynamics model to extract actions a t from pairs of latent states z t , z t+1 , and
(3) learning a planner to forecast future latents z t .
Figure 2. After training the encoder, Latent Diffusion Planning trains two diffusion models. Top: We train a inverse dynamics model (IDM) with a diffusion objective to directly extract the actions that will be used for control from pairs of latent states.
Bottom: We train a powerful latent diffusion model to forecast a chunk of future latent states. The planner and the IDM are used together to produce an action chunk, similar to (Chi et al., 2023).
this section cite: ['b39']

Section: Learning the Latent Space
We circumvent planning over high-dimensional image observations by planning over a learned latent space. Similar to prior work in planning with world models (Watter et al., 2015;Ha & Schmidhuber, 2018;Hafner et al., 2020), we learn this latent space using an image reconstruction objective. Our planner thus becomes similar to video models that forecast image frames in a learned latent space (Yan et al., 2021;Hong et al., 2022;Blattmann et al., 2023).
In this work, we train a variational autoencoder (Kingma  Rezende et al., 2014) to obtain a latent encoder E and decoder D. Specifically, we optimize the β-VAE (Higgins et al., 2017) objective, where x is our original image, z is our learned latent representation of the image, θ are the parameters for our decoder, ϕ are the parameters for our encoder, and β is the weight for the KL regularization term:
L VAE (θ, ϕ; x, z, β) = E q ϕ (z | x) [log p θ (x | z)] -βD KL (q ϕ (z | x)||p(z))(1)
In practical scenarios, we may have a limited expert demonstration dataset, but much larger and diverse suboptimal or action-free datasets. In this phase of learning, we can make use of the visual information in such datasets for training a more robust latent encoder.
this section cite: ['b52', 'b16', 'b18', 'b56', 'b23', 'b4', 'b44', 'b21']

Section: Planner and Inverse Dynamics Model
Our policy consists of two separate modules: (1) a planner over latent embeddings z, and (2) an inverse dynamics model similarly operating over the latent embeddings. The planner and IDM are both parameterized as DDPM models, motivated by the expressivity that diffusion models offer.
The planner is conditioned on the current latent embedding, which consists of the concatenated latent image embedding and robot proprioception, and diffuses over a horizon of future embeddings. We use Diffusion Policy's Conditional U-Net architecture, with a CNN backbone. Concretely, we optimize the following objective:
L planner (ψ, z) = E t,ϵ [||ϵ ψ (z ˆk+1 , . . . , z ˆk+H ; z k , t) -ϵ|| 2 ] (2
)
where z k is the latent embedding at timestep k of the trajectory; z ˆk+1 , . . . , z ˆk+H is the noised latent embedding sequence, with corresponding noise ϵ; H is the maximum horizon of the forecasted latent plan; t is the diffusion noise timestep; and ψ are the parameters of the planner diffusion model.
Our inverse dynamics model is trained to reconstruct the action between a pair of states, conditioned on their associated latent embeddings. We use the MLPResNet architecture from IDQL (Hansen-Estruch et al., 2023) to diffuse actions, as it is more lightweight. We optimize the loss:
L IDM (ξ, z) = E t,ϵ [||ϵ ξ (a ˆk; z k , z k+1 , t) -ϵ|| 2 ] (3
)
where z k is the latent embedding at timestep k of the trajectory; a ˆk is the noised action, with corresponding noise ϵ; t is the diffusion noise timestep; and ξ are the parameters of the inverse dynamics diffusion model.
Because our latent embedding is frozen from the learned VAE, the planner and IDM do not share parameters and can be trained separately (Figure 2). Then, at inference time, the two modules are combined to extract action sequences. First, the planner forecasts a future horizon of states via DDPM sampling (Alg. 1 Lines 6-9). Then, we use the inverse dynamics model to extract actions from latent embedding pairs produced by the planner, also via DDPM Diffusion (Alg. 1 Lines 12-15). Like Diffusion Policy, we employ receding-horizon control (Mayne & Michalska, 1988), and execute for a shorter horizon than the full forecasted horizon (Alg. 1 Lines 17-19).
this section cite: ['b20', 'b37']

Section: Experiments
We seek to answer the following questions:
• Does Latent Diffusion Planning leverage action-free data for improved planning?
• Is Latent Diffusion Planning comparable to state-ofthe-art imitation learning algorithms that leverage suboptimal data?
• Can Latent Diffusion Planning be an effective imitation learning method in a real-world robotics system, where there may be suboptimal or action-free data?
this section cite: []

Section: Experimental Setup
Simulated Tasks We focus our experiments on 4 imagebased imitation learning tasks: (1) Robomimic Lift, (2) Robomimic Can, (3) Robomimic Square, and (4) ALOHA Sim Transfer Cube. Robomimic (Mandlekar et al., 2021) is a robotic manipulation and imitation benchmark, including the tasks Lift, Can, and Square. The Transfer Cube task is a simulated bimanual ALOHA task, in which one ViperX 6-DoF arm grabs a block and transfers it to the other arm (Zhao et al., 2023).
To demonstrate the effectiveness of Latent Diffusion Planning, we assume a low demonstration data regime, such that additional suboptimal or action-free data can improve performance. For Can and Square, we use 100 out of the 200 demonstrations in the Robomimic datasets; for Lift, we use 3 demonstrations out of the 200 total; and for Transfer Cube, we use 25 demonstrations. To further emphasize the importance of suboptimal data, these demonstrations cover a limited state space of the environment. Our suboptimal data consists of 500 failed trajectories from an undertrained behavior cloning agent. Our action-free data consists of 100 demonstrations for Lift, Can, and Square from the Robomimic dataset, and 25 demonstrations for Cube. We evaluate the success rate out of 50 trials, using the best checkpoint from the last 5 saved checkpoints, with 2 seeds.
Real World Task We create a real world implementation of the Robomimic Lift task, where the task is to pick up a red block from a randomly initialized position. We use a Frank Panda 7 degree of freedom robot arm, with a wrist-mounted Zed camera. We use the DROID setup (Khazatsky et al., 2024) and teleoperate via the Oculus Quest 2 headset. We use cartesian pose control.
Table 1. Leveraging Action-Free Data. LDP is able to leverage suboptimal and action-free data. We compare against DP baselines that leverage action-free data by using an IDM to relabel actions (DP-VPT) and video planning models that may use action-free data for the video planner (UniPi). We find LDP to perform better than both approaches, especially when combined with suboptimal data.
Method Lift Can Square ALOHA Cube Average DP 0.60 ± 0.00 0.63 ± 0.01 0.48 ± 0.00 0.32 ± 0.00 0.51 DP-VPT 0.69 ± 0.01 0.75 ± 0.01 0.48 ± 0.04 0.45 ± 0.03 0.59 UniPi-OL + Action-Free 0.09 ± 0.05 0.23 ± 0.03 0.07 ± 0.03 0.02 ± 0.00 0.11 UniPi-CL + Action-Free 0.14 ± 0.02 0.32 ± 0.04 0.09 ± 0.01 0.17 ± 0.03 0.18 LDP 0.69 ± 0.03 0.70 ± 0.02 0.46 ± 0.00 0.64 ± 0.04 0.65 LDP + Action-Free 0.67 ± 0.01 0.78 ± 0.04 0.47 ± 0.03 0.70 ± 0.02 0.66 LDP + Action-Free + Subopt 1.00 ± 0.00 0.98 ± 0.00 0.83 ± 0.01 0.97 ± 0.01
this section cite: ['b62', 'b25']

Section: 0.95
We collect 82 demonstrations, collect 84 suboptimal trajectories, and 12 action-free demonstrations.
Our suboptimal data consists of failed trajectories from policy evaluations. This is an effective way to reuse the data generated during iterations of training, that algorithms modeling actions, such as Diffusion Policy, cannot use. Action-Free data may consist of kinesthetic demonstrations, human videos, or handheld demonstrations (Chi et al., 2024). In our case, we collect teleoperated demonstrations with actions removed.
To evaluate our policies, we calculate the success rate across 45 evaluation trials. To thoroughly evaluate performance across the initial state space, we evaluate across a grid of 3x3 points, with 5 attempts per point. We evaluate 3 seeds per method.
Baselines We consider two main categories of baselines: (1) Imitation learning with suboptimal or action-free data (DP, RC-DP, DP+Repr, DP PT + FT, DP-VPT), and (2) Video planning (UniPi-OL, UniPi-CL).
• Diffusion Policy (DP) is a state-of-the-art imitation learning algorithm.
• Reward-Conditioned Diffusion Policy (RC-DP) utilizes suboptimal actions by conditioning the policy on a binary value indicating whether the action chunk comes from optimal demonstrations or not. This method is inspired by reward-conditioned approaches (Kumar et al., 2019b;Chen et al., 2021).
• Diffusion Policy with Representation Learning (DP+Repr) uses a VAE pretrained on demonstration, suboptimal, and action-free data as the observation encoder. This is representative of methods that leverage suboptimal data through representation learning.
• Diffusion Policy Pretrain + Finetune (DP PT + FT) pretrains on suboptimal trajectories and finetunes on demos. This is representative of methods that leverage suboptimal data through learning trajectory-level features.
• Diffusion Policy with Video PreTraining (DP-VPT) trains an inverse dynamics model to relabel actionfree data, inspired by VPT (Baker et al., 2022) and BCO (Torabi et al., 2018).
• Open-Loop UniPi (UniPi-OL) is based off of UniPi (Du et al., 2023a), a video planner for robot manipulation. UniPi-OL generates a single video trajectory, extracts actions, and executes the actions in an open-loop fashion. We use a goal-conditioned behavior cloning agent to reach generated subgoals (Wen et al., 2024).
• Closed-Loop UniPi (UniPi-CL) is a modification that allows UniPi to perform closed-loop replanning over image chunks. Like LDP, UniPi-CL generates dense plans instead of waypoints, though in image space. We learn an inverse dynamics model to extract actions.
this section cite: ['b7', 'b1', 'b51', 'b53']

Section: Imitation Learning with Action-Free Data
In Table 1, we examine how action-free data can be used to improve imitation learning policies. Imitation learning policies that model actions, such as DP, are unable to natively use action-free data, while planning-based approaches can benefit from this additional data.
One approach is to relabel action-free data using an inverse dynamics model. We find this to be effective for most tasks, showing that the reannotated actions are useful for policy improvement. However, we see that LDP is better able to leverage action-free data by directly using it for the planner, rather than generating possibly inaccurate actions to subsequently learn from.
Like LDP, video planning methods can directly use actionfree data for improving the planner, which for UniPi is the Qualitatively, for UniPi-OL, we find that while the goalconditioned agent is able to follow goals effectively, the policy still struggles with the difficult parts of the task, such as grasping the object. Forecasting goals does not provide the dense supervision for exactly how to grasp an object, and furthermore, UniPi-OL does not support replanning when a grasp is missed. UniPi-CL is able to address this by dense image forecasting, and consistently outperforms UniPi-OL. However, this closed-loop method is not only slow, but faces issues with video generation, such as regenerating static frames during parts of the task with less movement, leading the agent to be stuck in certain positions. This is especially noticeable for the ALOHA Cube task, where the agent is often stuck right before picking up the cube. Compared to UniPi-OL and UniPi-CL, LDP is able to circumvent many of these issues due to its latent planning and dense forecasting.
this section cite: []

Section: Imitation Learning with Suboptimal Data
In Table 2, we present imitation learning results with suboptimal data. First, LDP outperforms DP, which can only utilize data with optimal actions. We notice, especially, that LDP with suboptimal data typically improves further upon LDP, showing the potential of leveraging diverse data sources outside of the demonstration dataset.
Next, RC-DP, a conditional variant of DP that utilizes suboptimal data, outperforms DP. By learning from suboptimal data, RC-DP can learn priors of robot motions while distinguishing optimal action sequences. We hypothesize that for the Can, Square, and ALOHA Cube tasks, the primitive motions of reaching toward or grasping the object, which are partially covered by the suboptimal dataset, provides a useful visuomotor prior for the policy. ALOHA Cube sees significant improvement, possibly because the larger action space of bimanual control benefits from additional reward-labelled data.
Next, we explore using suboptimal data for feature-learning. DP + Repr uses suboptimal data for pretraining a visionencoder, and DP PT + FT for pretraining the visuomotor policy. DP + Repr only improves policy performance for the Lift task, implying that end-to-end training of the vision encoder learns stronger features for complex manipulation tasks. DP PT + FT is particularly successful for tasks that require more precise manipulation, such as Square and ALOHA Cube, implying that learned prior motions is a useful policy initialization. Both the success of RC-DP and DP PT + FT suggest that leveraging the suboptimal trajectories is a useful way to improve imitation learning results. LDP, which leverages suboptimal data for the latent encoder and IDM, has higher overall performance than these methods, averaging across the suite of simulated tasks.
Next, we compare against UniPi, which plans over image subgoals (OL) or image chunks (CL). Due to the low demonstration data regime, learning effective and accurate video policies is difficult, and LDP strongly outperforms UniPi-OL and UniPi-CL. In addition, we notice that UniPi-CL outperforms UniPi-OL for all tasks, implying that dense forecasting, even within the image domain, is more effective than goal-conditioned methods.
Finally, we find that LDP with action-free and suboptimal data leads to the strongest performance. We find a significant improvement in all of the simulated tasks, including compared to the variants of LDP that only use either actionfree or suboptimal data. This suggests that the recipe for combining these two data sources can lead to a stronger planner and a more robust inverse dynamics model, leading to the best performing model for these tasks.
this section cite: []

Section: Imitation Learning in the Real World
Real world data is more expensive and time-consuming to collect; hence, examining the effect of easier-to-collect suboptimal and action-free data provides insights for scalable learning. In Table 3, we provide results on a Franka Lift Cube task. In this task, we examine the performance of DP, which can only leverage action-labeled data, with our method, which can use suboptimal and action-free data. We find that LDP is able to consistently outperform DP, especially with the addition of action-free data. For real-world systems, this is promising, as collecting high-quality demos can be difficult and time intensive, whereas utilizing suboptimal trajectories, often a byproduct of evaluating policies, or collecting action-free trajectories more efficiently, such as in (Chi et al., 2024), can be a scalable direction.
this section cite: []

Section: Ablation: LDP Hierarchical
In an attempt to understand the effect of LDP's dense forecasting, we compare LDP with a hierarchical version of LDP (LDP Hierarchical). In this implementation, LDP Hierarchical plans over subgoals 4 steps apart, and extracts 4 actions between pairs of forecasted latent states. Thus, LDP Hierarchical is closed-loop, yet the planner operates at a slightly abstracted level compared to non-hierarchical LDP.
In Table 4, we find that LDP outperforms LDP Hierarchical across the 3 Robomimic tasks. This suggests that dense forecasting is an important part and contribution of LDP. This also reflects UniPi results in Table 1 and Table 2 that show closed-loop planning outperforms open-loop planning.
Table 4. Hierarchical Ablation. LDP's dense forecasting outperforms a hierarchical variant of LDP. Method Lift Can Square LDP 0.69 ± 0.03 0.70 ± 0.02 0.46 ± 0.00 LDP Hier. 0.53 ± 0.03 0.62 ± 0.04 0.31 ± 0.01 LDP + Subopt 0.84 ± 0.06 0.68 ± 0.02 0.55 ± 0.03 LDP Hier. + Subopt 0.65 ± 0.03 0.60 ± 0.02 0.43 ± 0.05
this section cite: []

Section: Discussion
We presented Latent Diffusion Planning, a simple planningbased method for imitation learning. We show that our design using powerful diffusion models for latent state forecasting enables competitive performance with state-of-theart imitation learning. We further show this latent state forecasting objective enables us to easily leverage heterogeneous data sources. In the low-demonstration data imitation regime, LDP outperforms prior imitation learning work that does not leverage such additional data as effectively.
Limitations. One limitation of the current approach is that the latent space for planning is simply learned with a variational autoencoder and might not learn the most useful features for control. Future work will explore different representation learning objectives. Further, our method requires diffusing over states, which incurs additional computational overhead as compared to diffusing actions. However, we expect continued improvements in hardware and inference speed will mitigate this drawback. Finally, we did not explore applying recent improvements in diffusion models (Peebles & Xie, 2023;Lipman et al., 2022), which may be important in large data regimes.
Future work. We have validated in simulation and real the hypothesis that latent state forecasting can leverage heterogeneous data sources. Future work can also evaluate whether this can be used to further improve more complex real-world tasks. One direction is to use a diverse dataset of human collected data, such as with handheld data collection tools (Young et al., 2021). Another approach would be to use autonomously collected robotic data (Bousmalis* et al., 2023). As these alternative data sources are easier to collect than demonstrations, they represent a different scaling paradigm that can outperform pure behavior cloning approaches. By presenting a method that can leverage such data, we believe this work makes a step toward more performant and general robot policies.
this section cite: ['b42', 'b35', 'b59', 'b5']

Section: References
Ref_id:b0 Title: Is conditional generative modeling all you need for decision-making? Year: (2023)
Ref_id:b1 Title: Video pretraining (vpt): Learning to act by watching unlabeled online videos Year: (2022)
Ref_id:b2 Title: Imitation learning by estimating expertise of demonstrators Year: (2022)
Ref_id:b3 Title: Zero-shot robotic manipulation with pretrained image-editing diffusion models Year: (2023)
Ref_id:b4 Title: Align your latents: Highresolution video synthesis with latent diffusion models Year: (2023)
Ref_id:b5 Title: Robocat: A self-improving foundation agent for robotic manipulation Year: (2023)
Ref_id:b6 Title: Genie: Generative interactive environments Year: (2024)
Ref_id:b7 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b8 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: ()
Ref_id:b9 Title: Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots Year: ()
Ref_id:b10 Title: -domain dynamics pretraining for visuo-motor control Year: (2024)
Ref_id:b11 Title: Vision transformers need registers Year: (2023)
Ref_id:b12 Title: Learning universal policies via text-guided video generation Year: (2023)
Ref_id:b13 Title:  Year: (2023)
Ref_id:b14 Title: Imitating latent policies from observation Year: (2019)
Ref_id:b15 Title: Deep visual foresight for planning robot motion Year: (2017)
Ref_id:b16 Title:  Year: (2018)
Ref_id:b17 Title: Learning latent dynamics for planning from pixels Year: (2019)
Ref_id:b18 Title: Dream to control: Learning behaviors by latent imagination Year: (2020)
Ref_id:b19 Title: Td-mpc2: Scalable, robust world models for continuous control Year: (2024)
Ref_id:b20 Title: Implicit q-learning as an actorcritic method with diffusion policies Year: (2023)
Ref_id:b21 Title: Learning basic visual concepts with a constrained variational framework Year: (2017)
Ref_id:b22 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b23 Title: Large-scale pretraining for text-to-video generation via transformers Year: (2022)
Ref_id:b24 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
Ref_id:b25 Title: A large-scale in-the-wild robot manipulation dataset Year: (2024)
Ref_id:b26 Title: Openvla: An open-source vision-language-action model Year: (2024)
Ref_id:b27 Title: Auto-encoding variational bayes Year: (2014)
Ref_id:b28 Title: Learning to Act from Actionless Videos through Dense Correspondences Year: (2023)
Ref_id:b29 Title: Offline reinforcement learning with implicit q-learning Year: (2021)
Ref_id:b30 Title: Reward-conditioned policies Year: (2019)
Ref_id:b31 Title: Reward-conditioned policies Year: (2019)
Ref_id:b32 Title: Conservative q-learning for offline reinforcement learning Year: (2020)
Ref_id:b33 Title: Behavior generation with latent actions Year: (2024)
Ref_id:b34 Title: Offline reinforcement learning: Tutorial, review, and perspectives on open problems Year: (2020)
Ref_id:b35 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b36 Title: What matters in learning from offline human demonstrations for robot manipulation Year: ()
Ref_id:b37 Title: Receding horizon control of nonlinear systems Year: (1988)
Ref_id:b38 Title: An open-source generalist robot policy Year: (2024)
Ref_id:b39 Title: Robotic learning datasets and RT-X models Year: (2023)
Ref_id:b40 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b41 Title: Imitating human behaviour with diffusion models Year: (2023)
Ref_id:b42 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b43 Title: Robot learning with sensorimotor pretraining Year: (2023)
Ref_id:b44 Title: Stochastic backpropagation and approximate inference in deep generative models Year: (2014)
Ref_id:b45 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b46 Title: Learning predictive models from observation and interaction Year: (2020)
Ref_id:b47 Title: Reinforcement learning upside down: Don't predict rewards-just map them to actions Year: (2019)
Ref_id:b48 Title: Behavior transformers: Cloning k modes with one stone Year: (2022)
Ref_id:b49 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b50 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b51 Title: Behavioral cloning from observation Year: (2018)
Ref_id:b52 Title: Embed to control: A locally linear latent dynamics model for control from raw images Year: (2015)
Ref_id:b53 Title: Any-point trajectory modeling for policy learning Year: (2024)
Ref_id:b54 Title: Daydreamer: World models for physical robot learning Year: (2023)
Ref_id:b55 Title: Masked trajectory models for prediction, representation, and control Year: (2023)
Ref_id:b56 Title: Videogpt: Video generation using vq-vae and transformers Year: (2021)
Ref_id:b57 Title: Learning interactive real-world simulators Year: (2023)
Ref_id:b58 Title: Video as the new language for real-world decision making Year: (2024)
Ref_id:b59 Title: Visual imitation made easy Year: (2021)
Ref_id:b60 Title: How to leverage unlabeled data in offline reinforcement learning Year: (2022-07)
Ref_id:b61 Title: Confidenceaware imitation learning from demonstrations with varying optimality Year: (2022)
Ref_id:b62 Title: Learning fine-grained bimanual manipulation with low-cost hardware Year: (2023)
Ref_id:b63 Title: ALOHA unleashed: A simple recipe for robot dexterity Year: (2024)
Ref_id:b64 Title: Robodreamer: Learning compositional world models for robot imagination Year: (2024)
