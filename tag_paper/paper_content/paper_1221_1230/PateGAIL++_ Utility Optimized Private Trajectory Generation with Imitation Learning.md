Title: PATEGAIL++: UTILITY OPTIMIZED PRIVATE TRAJEC-TORY GENERATION WITH IMITATION LEARNING
Abstract: Human mobility trajectory data supports a wide range of applications, including urban planning, intelligent transportation systems, and public safety monitoring. However, large-scale, high-quality mobility datasets are difficult to obtain due to privacy concerns. Raw trajectory data may reveal sensitive user information, such as home addresses, routines, or social relationships, making it crucial to develop privacy-preserving alternatives. Recent advances in deep generative modeling have enabled synthetic trajectory generation, but existing methods either lack formal privacy guarantees or suffer from reduced utility and scalability. Differential Privacy (DP) has emerged as a rigorous framework for data protection, and recent efforts such as PATE-GAN and PATEGAIL integrate DP with generative adversarial learning. While promising, these methods struggle to generalize across diverse trajectory patterns and often incur significant utility degradation. In this work, we propose a new framework that builds on PATEGAIL++ by introducing a sensitivity-aware noise injection module that dynamically adjusts privacy noise based on sample-level sensitivity. This design significantly improves trajectory fidelity, downstream task performance, and scalability under strong privacy guarantees. We further adapt our framework to the local differential privacy (LDP) setting, allowing individual-level protection without reliance on a trusted server. We evaluate our method on a real-world mobility dataset and demonstrate its superiority over state-of-the-art baselines in terms of privacy-utility trade-off.

Section: INTRODUCTION
Figure 1: Trajectory segments that strongly reflect user-specific behaviors pose higher privacy risks than those that resemble common or overlapping ones.
Human mobility data supports a wide range of critical applications, including urban planning (Ruan et al., 2020), intelligent transportation systems (Ma et al., 2013), human mobility analysis (Liang et al., 2021;Wang et al., 2021), and public safety monitoring (Gao et al., 2017). However, obtaining high-quality, large-scale mobility datasets remains challenging due to serious privacy concerns. Raw trajectory records can expose sensitive personal information, such as home addresses, daily routines, or social relationships, that may be exploited for malicious purposes if not properly protected.
Recent advances (Wang et al., 2023b;a;Roman et al., 2025;Zhang et al., 2025;Ma et al., 2024;Fan et al., 2024;Bouabba et al., 2024;Netzler & Lienkamp, 2024;Narayanan et al., 2024) in deep generative modeling have motivated the use of synthetic trajectory generation as a privacy-preserving alternative. While promising, many existing methods either lack formal privacy guarantees (Kim & Jang, 2022;Rao et al., 2020) or incur significant utility degradation and computational overhead (Du et al., 2023;Rao et al., 2020), limiting their practical applicability.
Differential Privacy (DP) has emerged as the gold standard for provable privacy protection (Dwork et al., 2006), and has been increasingly applied in data publishing (Asghar et al., 2020) and synthetic data generation (Jordon et al., 2018). One notable effort is PATE-GAN (Jordon et al., 2018), which combines generative adversarial networks with the Private Aggregation of Teacher Ensembles (PATE) (Papernot et al., 2017) framework to generate differentially private synthetic data. While promising, its utility remains limited, especially in complex spatial-temporal domains like mobility trajectories. On the other hand, PATEGAIL (Wang et al., 2023b) adapts DP mechanisms to the imitation learning setting, leveraging Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) while incorporating differential privacy into policy training. While PATEGAIL provides a promising direction for behavior modeling under DP, it suffers from reduced performance and challenges in generalizing to diverse trajectory patterns in real-world settings.
this section cite: ['b28', 'b21', 'b20', 'b34', 'b10', 'b27', 'b36', 'b22', 'b8', 'b2', 'b24', 'b23', 'b18', 'b26', 'b5', 'b26', 'b7', 'b0', 'b17', 'b17', 'b25', 'b15']

Section: Limitation of State-of-the-Art.
A critical limitation of current DP-based trajectory generation methods lies in their uniform treatment of all data points when applying privacy noise. These approaches typically inject the same amount of noise across all samples, regardless of their semantic or statistical sensitivity. However, not all trajectories pose the same privacy risk-some may contain highly identifying patterns, while others are inherently less sensitive. As shown in Fig. 1, trajectory segments that are behaviorally unique are more sensitive than those that overlap with many users' trajectories. Applying identical noise levels can therefore lead to unnecessary utility loss for lowrisk samples and insufficient protection for highly sensitive ones. This rigid noise assignment limits scalability and weakens the overall privacy-utility trade-off.
this section cite: []

Section: Our Approach.
We propose PATEGAIL++, a new differentially private imitation learning framework for trajectory generation. Our design introduces a sensitivity-aware noise injection module that dynamically adjusts noise levels according to the sensitivity of trajectory samples, offering stronger protection for high-risk segments while reducing unnecessary distortion in low-risk cases. We further extend this framework to the local differential privacy (LDP) setting, ensuring user-level protection without reliance on a trusted server. To stabilize training under adaptive noise, PATE-GAIL++also incorporates Wasserstein GAN with Gradient Penalty (WGAN-GP) (Gulrajani et al., 2017a), which mitigates instability in discrete trajectory learning and accelerates policy convergence. Our key contributions are as follows:
• We design a differentially private generative model tailored for mobility trajectory data that better balances privacy and utility.
• We introduce a novel framework PATEGAIL++with a customizable sensitivity module that enhances trajectory realism while maintaining strict privacy guarantee.
• We extend our framework to the local differential privacy setting, allowing individual privacy protection without reliance on a trusted server.
• We conduct extensive experiments on a real-world dataset, demonstrating superior performance over existing baselines in privacy and utility trade-off.
this section cite: []

Section: PRELIMINARIES

this section cite: []

Section: HUMAN MOBILITY DECISION-MAKING
We consider a scenario with multiple users, each equipped with a personal mobile device that locally stores their historical mobility trajectories. These trajectories consist of sequential decision-making records that capture human movement behaviors, such as daily commuting or shopping. Formally, for each user u ∈ U, the trajectory is represented as T u = {(l 1 , t 1 ), • • • , (l N , t N )}, where l i and t i denote the location and timestamp at step i, respectively. These sequences are the result of the user's latent decision-making strategy (i.e., policy) under spatial-temporal constraints.
Each trajectory describes a user's traversal through a sequence of spatial-temporal states, guided by a corresponding sequence of decisions (i.e., actions). We define the state s t as a user's mobility history up to time t, i.e., s t = {(l τ , t τ )} τ ≤t , and the action a t action space is defined based on the widely-adopted exploration and preferential return (EPR) model (Song et al., 2010;Jiang et al., 2016). Core to this modeling framework are two functions, i.e., policy and reward, that govern decision-making and need to be learned from trajectory data:
Definition 1: Policy function π(a|s) defines the probability of choosing action a given the current state s. This function governs how a user makes decisions at different situations.
Definition 2: Reward function R(s, a) quantifies the implicit benefit or desirability of taking action a at state s. This function explains the preferences that guide the user's behavior.
Human users are adapting policies for higher total reward in decision-making. Both the policy and reward functions are unobserved and need to be inferred from the user trajectories.
this section cite: ['b30', 'b16']

Section: HUMAN STRATEGY LEARNING VIA IMITATION LEARNING
Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016) is an imitation learning method that jointly learns the reward function and policy from expert demonstrations. It frames imitation as a minimax optimization problem, i.e.,
min π θ max D ϕ E (s,a)∼π E [log D ϕ (s, a)] + E (s,a)∼π θ [log(1 -D ϕ (s, a))],(1)
where π E denotes the expert (real) policy, and D ϕ (s, a) is a discriminator trained to distinguish expert from generated behavior. The learned policy π θ is trained using reinforcement learning to maximize expected rewards derived from the discriminator's output. The learned policy π θ is optimized to fool the discriminator, effectively mimicking expert trajectories.
this section cite: ['b15']

Section: DIFFERENTIAL PRIVACY IN DECISION-MAKING
Differential Privacy (DP) (Dwork et al., 2006) is a formal framework for quantifying privacy guarantees. Let D and D ′ be two neighboring datasets that differ in a single record. A randomized algorithm M satisfies (ε, δ)-differential privacy if, for all such neighboring datasets D, D ′ and for all measurable subsets E of the output space,
P[M(D) ∈ E] ≤ e ε • P[M(D ′ ) ∈ E] + δ.(2)
To ensure DP, noise is typically added using the Laplace mechanism:
Lap(x | λ) = 1 2λ exp - |x| λ , (3
)
where λ is calibrated based on the global sensitivity and the privacy budget ε. The privacy budget controls how much information about a user's data can leak; smaller ε means stronger privacy. Beyond noise addition, several fundamental properties of DP are critical when applying it to learning algorithms. Specifically:
Sequential Composition. The sequential composition states that combining multiple subroutines that satisfy DP for (ε 1 , δ 1 ), (ε 2 , δ 2 ), • • • results in a mechanism that satisfies (ε, δ)-DP for ε = ε i and δ = δ i .
this section cite: ['b7']

Section: Post-processing.
Given an (ε, δ)-DP algorithm M, releasing g(M(D)) for any g still satisfies (ε, δ)-DP. That is, post-processing an output of a differentially private algorithm does not incur any additional loss of privacy.
Zero-Concentrated Differential Privacy. While (ε, δ)-DP is the standard definition, Zero-Concentrated Differential Privacy (ρ-zCDP) (Dwork & Rothblum, 2016;Azize & Basu, 2024) provides a more refined privacy accounting framework. Its tighter composition guarantees make it particularly well-suited for iterative algorithms.
this section cite: ['b6', 'b1']

Section: PROBLEM DEFINITION

this section cite: []

Section: Differentially Private Trajectory Generation Problem.
Given a set of users U with private mobility trajectories {T u } u∈U , the goal is to learn a global trajectory generation policy π(a|s) such that: 1). The generated synthetic trajectories are utility-preserving, i.e., they statistically resemble the real trajectories and support downstream applications such as prediction and recommendation. 2).
The learning algorithm satisfies (ε, δ)-differential privacy, meaning the presence or absence of any individual user's trajectory does not significantly affect the output. Challenges. The proposed differentially private trajectory generation problem presents three key challenges: (C1) How can privacy budgets be allocated to reflect varying data sensitivity and ensure good model utility? (C2) How can we provide formal privacy guarantees under non-uniform noise, given that adaptive noise scaling complicates standard DP analysis? (C3) How can policy learning remain stable under adaptive noise injection in the federated learning setting?
Federated Learning Setting. In this work we adopt a federated data-access model following Wang et al. (2023b). Human mobility trajectories are sensitive, e.g., GPS traces can reveal home and work locations, daily routines, and social patterns, and are often subject to legal or institutional restrictions that prohibit centralizing raw data. Federated learning thus provides an appropriate deployment framework: each user retains their trajectory data locally, and only differentially private reward signals are shared with the server. Our work focuses on improving the utility-privacy tradeoff within this practically constrained federated setting.
this section cite: []

Section: METHOD
In this section, we address the differentially private trajectory generation problem by introducing the PATEGAIL++ framework as shown in Fig. 2. In this framework, each user device locally trains a discriminator on its private mobility data to evaluate the plausibility of synthetic trajectories, while the global policy is trained on the server using aggregated, differentially private reward signals derived from local discriminator outputs.
Three important components are proposed in PATEGAIL++: First, we introduce a sensitivity-aware module that identifies the privacy sensitivity levels of user trajectories in the training data, thereby addressing Challenge C1 (see Sec.3.1). Second, we provide a formal theoretical guarantee for (ε, δ)differential privacy, thereby addressing Challenge C2 (see Sec.3.2). Finally, to tackle Challenge C3, we design a reward function ensemble strategy that stabilizes policy learning under adaptively injected noise (see Sec.3.3).
this section cite: []

Section: SENSITIVITY-DRIVEN PRIVACY BUDGET ASSIGNMENT
In this framework, each user u ∈ U maintains a local discriminator D ϕu trained to distinguish real from generated trajectories on-device. For each state-action pair (s, a), the local reward estimate is R (u) (s, a) = D ϕu (s, a), and the server aggregates these signals via a Laplace mechanism,
R(s, a) = 1 |U| u∈U D ϕu (s, a) + Lap(0, λ).(4)
Here, the server sees all sample-wise rewards, and the Lap(0, λ) ensures differential privacy for global policy updates.
To account for reward variability across users, PATEGAIL introduces a reward dynamics compensation term based on the variance of local rewards, R(s, a) = R(s, a) -β • ξ(s, a), where ξ(s, a) = Var(D ϕu (s, a)) + Lap(0, λ c ).
This adjustment allows the global policy π(a|s) to maximize a high-probability lower bound of the users' expected cumulative rewards, while satisfying (ε, δ)-differential privacy.
Despite these contributions, PATEGAIL has important limitations. First, it applies a uniform privacy budget across all iterations, regardless of individual-wise as well as iteration-wise sensitivity. This neglects trajectory-specific privacy risks, potentially overprotecting benign data while underprotecting highly sensitive samples. Second, the added noise is not calibrated to local context or trajectory variability. As a result, the policy may suffer from degraded utility or lack robustness across heterogeneous mobility behaviors. These limitations motivate the need for a more fine-grained approach that is able to identify trajectory-level data sensitivity and adjust privacy budget accordingly.
this section cite: []

Section: Trajectory Sensitivity Module.
To improve the privacy-utility trade-off in federated imitation learning, we introduce a trajectory sensitivity module that dynamically calibrates the amount of noise injected into the reward aggregation process based on the privacy sensitivity of samples each iteration. This mechanism is to enable fine-grained differential privacy guarantees. We also extend this to individual-level differential privacy guarantees later.
We estimate the privacy sensitivity of each state-action pair (s, a) based on the output of local discriminators D ϕu . Intuitively, if a discriminator D ϕu assigns a high reward R (u) (s, a) = D ϕu (s, a) ≈ 1, it indicates that the generated sample is highly indistinguishable from real user trajectories. Such samples are more likely to resemble unique or rare behaviors, making them more vulnerable to inference attacks. Therefore, they are considered more privacy-sensitive and require stronger protection.
To reflect this, we define the privacy sensitivity of a sample inversely with respect to its confidence margin from 1, i.e.,
Sensitivity(s, a) ∝ 1 1 -R(s, a) + δ ′ ,
where δ ′ > 0 is a small constant to avoid division by zero. Based on this sensitivity score, we allocate a per-sample privacy budget ε(s, a) such that:
ε(s, a) = ε • w(s, a) (s ′ ,a ′ )∈D w(s ′ , a ′ )
, where w(s, a) = 1 -Rp (s, a) + δ ′ .
To avoid potential information leakage when adaptively allocating privacy budgets, the Rp (s, a) above is a differentially private pilot estimate of R(s, a). This ensures that samples with higher discriminator confidence (i.e., more sensitive) receive a smaller share of the privacy budget, leading to stronger noise injection. The total privacy expenditure is constrained by:
(s,a)∈D ε(s, a) = ε,
where D is the set of all state-action pairs queried during training.
This sensitivity-aware allocation enables stronger protection for high-risk samples while reducing unnecessary noise for low-risk ones, improving both privacy and utility under a fixed global budget.
this section cite: []

Section: Interpretation of Sensitivity.
Our sensitivity module reflects how closely a generated state-action resembles behavioral patterns that are distinctive to an individual user. When a user's local discriminator D ϕu assigns a high score, it indicates that the generated sample is highly similar to that user's real behavior. We therefore treat such samples as high-sensitivity and add stronger noise, preventing the server from learning or exploiting user-specific behavioral signatures. This design also preserves DP guarantees: incorporating external semantic cues (e.g., location types or home-work labels) into the sensitivity calculation would depend on non-private information and violate the DP accounting assumption. By basing sensitivity solely on internal model signals, our formulation maintains end-to-end DP protection.
this section cite: []

Section: SENSITIVITY-AWARE REWARD AGGREGATION
For each state-action pair (s, a), each user u computes a local reward R (u) (s, a) = D ϕu (s, a). The server aggregates these using a sensitivity-aware Laplace mechanism: R(s, a) = 1
N N u=1 R (u) (s, a) + Lap ∆f ε(s, a)(6)
where ∆f ε(s,a) equals to λ in Eq. ( 4) if the total privacy are distributed evenly per sample (s, a). To account for reward variability across users, we adopt the dynamics compensation mechanism introduced in PATEGAIL. Extending to LDP. While our current framework enables fine-grained privacy control at the sample level with strong theoretical guarantees, it assumes a trustworthy server that can access raw rewards. To relax this assumption, we extend our framework to the local differential privacy (LDP) setting, where each user perturbs their own rewards before transmitting them to the server. In this way, the server never observes raw rewards from users (i.e., the outputs of individual discriminators). We further introduce a per-user privacy budget ε (u) (s, a), such that these budgets across users aggregate to the per-sample budget ε(s, a). The aggregated reward is then computed as:
this section cite: []

Section: R(s,
R(s, a) = 1 N N u=1 R (u) (s, a) + Lap ∆f ε (u) (s, a)(8)
To allocate per-user budgets, we define:
ε (u) (s, a) = ε(s, a) • w (u) (s, a) N u=1 w (u) (s, a)
, where
w (u) (s, a) = 1 -R(u) (s, a) + δ ′ .
Note that the weights can be calculated and shared using homomorphic encryption (Das, 2018) among users to protect individual privacy.
This design enables fine-grained privacy control at both the sample and individual-user levels, while preserving strong theoretical guarantees. As a result, our method improves utility in generated trajectories and downstream tasks, and broadens the applicability of the framework under more practical privacy assumptions.
this section cite: ['b4']

Section: WGAN-GP FOR STABLE POLICY LEARNING
The original PATEGAIL employs a cross-entropy-based discriminator, which often produces vanishing gradients and unstable updates when applied to discrete trajectory data. These issues are further amplified under DP-induced perturbations. To stabilize training, PATEGAIL++ adopts a Wasserstein GAN with Gradient Penalty (Gulrajani et al., 2017b), which minimizes the Wasserstein-1 distance between expert and generated distributions:
min π θ max D ϕ E (s,a)∼π E [D ϕ (s, a)] -E (s,a)∼π θ [D ϕ (s, a)] -λ GP E x (∥∇ xD ϕ (x)∥ 2 -1) 2 , (9
)
where x denotes samples interpolated between expert and generated state-action pairs, and λ GP controls the penalty strength.
Following the central reward design in Sec. 3.2, the critic output of each local discriminator is directly treated as the per-user reward, i.e., R (u) (s, a) = D (u) ϕ (s, a). These local rewards are then aggregated into R(s, a) through the sensitivity-aware Laplace mechanism. Compared to the log-based reward in GAIL, the Wasserstein critic provides smoother gradients and avoids reward saturation, leading to more stable aggregation and policy updates under adaptive noise injection.
We further apply spectral normalization as an optional regularizer to enforce Lipschitz continuity alongside the gradient penalty. This combination enhances the robustness of local discriminators, improves the fidelity of aggregated rewards, and ultimately accelerates policy convergence in PATE-GAIL++.
this section cite: []

Section: PRIVACY RISKS ANALYSIS
Following PATEGAIL (Wang et al., 2023b), we measure the privacy risks in PATEGAIL++ using white-box membership inference attack (MIA), where the attack has full access to the target model. Specifically, let T A denote the set of trajectories selected as attack targets. For each T u ∈ T A , the adversary extracts the reward corresponding to every state-action pair (s, a) and constructs a feature vector v f from these values. This feature vector v f serves as input to a Random Forest classifier, whose objective is to determine whether T u originates from the training set of the trajectory generation model. To train the classifier, an 70% of trajectories are drawn from the training set and 30% from outside the training set are used as positive and negative instances, respectively.
this section cite: []

Section: EVALUATION
Dataset. We evaluate PATEGAIL++ on two datasets: the processed Geolife dataset (Wang et al., 2023b) and the Telecom Shanghai dataset (Li et al., 2021;Guo et al., 2020;Wang et al., 2019). The Geolife dataset was previously used in PATEGAIL, and contains mobility trajectories of 83 users collected between April 2007 and October 2011. The Telecom Shanghai dataset contains more than 7.2 million records of accessing the Interent through 3,233 base stations from 9,481 mobile phones for six months. For both datasets, each record is obtained from GPS logs on users' mobile phones and includes latitude, longitude, and timestamp information.
Implementation We implemented PATEGAIL++ in Python 3.10 using PyTorch 2.1 with CUDA support, and conducted all experiments on an NVIDIA A100 GPU with 256 GB RAM. Training is based on Proximal Policy Optimization (PPO) (Schulman et al., 2017), with entropy regularization and gradient clipping for stability. Our system is fully configurable via YAML and supports automated logging, evaluation, and model checkpointing.
this section cite: ['b19', 'b14', 'b33', 'b29']

Section: Metrics.
To evaluate the quality of generated mobility trajectories, we adopt five statistical metrics at both trajectory and record levels, following (Wang et al., 2023b). At the trajectory level, Radius (of gyration) quantifies the dispersion of a trajectory around its center of mass, while DailyLoc measures the number of distinct locations visited. At the record level, Distance captures travel between consecutive trajectory points, G-rank reflects the normalized frequency of visits to globally popular locations across users, and I-rank measures user-specific visit frequencies to top locations averaged across individuals. Each metric is represented as a probability distribution over trajectories or records, and we employ the Jensen-Shannon Divergence (JSD) to quantify the discrepancy between real and synthetic distributions.
this section cite: []

Section: COMPARISON RESULTS WITH BASELINES
In this section, we evaluate PATEGAIL++'s utility performance using five key metrics: Radius, Dai-lyLoc, Distance, G-rank, and I-rank. We assess robustness under varying noise levels. For clarity, all reported noise levels correspond to the uniform noise setting used in PATEGAIL; in contrast, PATE-GAIL++ employs dynamic noise allocation, but we ensure that the total privacy budget is matched across methods. In addition to the privacy budget ε, we further evaluate privacy leakage through Membership Inference Attacks (MIA).
Baselines vs. Federated/DP Setting. The baselines (GAN (Goodfellow et al., 2020), SeqGAN (Yu et al., 2017), Time-Geo (Jiang et al., 2016), MoveSim (Feng et al., 2020), DiffTraj (Zhu et al., 2023)) are trained and evaluated in a centralized, non-federated, non-DP setting, whereas PATEGAIL and PATEGAIL++ operate in a federated setup with DP-style perturbations during reward aggregation. Therefore, the baseline numbers in Tab. 1 are reported to demonstrate compatibility of our method under stronger constraints, rather than to claim a strict apples-to-apples SOTA victory. See detailed description of the baseline methods in Appendix A.2 Despite these constraints, PATEGAIL++ attains competitive utility across metrics and strong semantic fidelity: for example, PATEGAIL++ achieves G-Rank = 0.0256 (better than MoveSim's 0.0387), with I-Rank comparable to MoveSim (0.0176 vs. 0.0173). In contrast, standard GAN suffers large errors (e.g., DailyLoc = 0.5795, G-Rank = 1.0000). While SeqGAN and MoveSim yield smaller values on geometry-oriented metrics such as Radius/Distance, they do not consistently preserve ranking-based semantics; PATEGAIL++ better maintains high-level trajectory realism under federated/DP constraints.
this section cite: ['b11', 'b35', 'b16', 'b9', 'b37']

Section: STUDY ON OTHER KEY HYPERPARAMETERS
We conduct an ablation study to evaluate the sensitivity of PATEGAIL++ to key hyperparameters and design choices. Specifically, we focus on the following gradient penalty coefficient introduced in Sec.3.3 and user subset ratio.
this section cite: []

Section: Gradient Penalty Coefficient (λ GP ).
To evaluate the effect of WGAN-GP regularization, we vary the gradient penalty coefficient across five settings: λ GP ∈ {1, 5, 10, 15, 20}. As shown in Tab. 6, incorporating WGAN-GP consistently improves performance, with PATEGAIL++ outperforming PATEGAIL on nearly all metrics. For each noise level and metric, we also selected the best PATE-GAIL++ configuration over λ GP ∈ {1, 5, 10, 15, 20} and compared it to PATEGAIL (See Fig. 3 in Appendix A.3.2). User Subset Ratio. Since our approach trains one discriminator per user, we analyze the effect of varying the number of users involved during training. Specifically, we fix λ GP = 20 and compare performance when using all, 80% and 40% of the available users. Due to space limit, we defer the results of user sebset ration in Appendix A.4.
this section cite: []

Section: EXTENDING TO LOCAL DIFFERENTIAL PRIVACY
In this section, we extend our framework to the local differential privacy (LDP) setting and evaluate its performance with and without sensitivity-aware aggregation at the individual level. We report results across the same five key metrics: Radius, DailyLoc, Distance, G-rank, and I-rank, as shown in Tab. 7. We denote the LDP variant of PATEGAIL++ with sensitivity-aware aggregation as PATEGAIL++ + and the variant without sensitivity-aware aggregation as PATEGAIL++ -. Overall,   For each noise level and metric, we select the best PATEGAIL++ configuration over five different gradient penalty coefficients λ GP ∈ {1, 5, 10, 15, 20} and compare it to PATEGAIL. Fig. 3 shows that PATEGAIL++ consistently outperforms PATEGAIL across all noise regimes and evaluation metrics. By adaptively allocating noise according to sample sensitivity and stabilizing training with WGAN-GP, PATEGAIL++ maintains high trajectory fidelity even under strong differential privacy constraints, whereas PATEGAIL deteriorates noticeably as noise increases. Across all metrics where lower is better (Radius, DailyLoc, Distance, G-Rank, and I-Rank), PATEGAIL++ either matches or surpasses PATEGAIL, often by a substantial margin in the moderate-noise setting where DP pressure is highest. Overall, PATEGAIL++ produces more stable, realistic, and consistent trajectories under differential privacy.
A.4 USER SUBSET RATIO
The results in Tab.10 show that both PATEGAIL++ and PATEGAIL remain robust when user participation is reduced from 100% to 80%, showing only minor performance shifts. However, with a user participation rate of 40%, PATEGAIL++ generally yields lower differences in metric val-
this section cite: []

Section: A APPENDIX

this section cite: []

Section: A.1 ETHICS AND REPRODUCIBILITY STATEMENTS
Ethics Statement This research adheres to the ICLR Code of Ethics. All experiments are conducted on publicly available benchmark datasets and environments that do not involve human subjects or personally identifiable information. No new data collection was performed, and no sensitive or private information is included. The contributions of this work are methodological, aiming to advance machine learning. While reinforcement learning and related methods may be applied in safety-critical or socially sensitive domains, this paper does not directly address or deploy in such contexts. We have taken care to report results honestly, acknowledge limitations, and follow best practices for research integrity. No conflicts of interest or ethical concerns beyond standard research conduct arise from this work.
this section cite: []

Section: Reproducibility Statement
We have made every effort to ensure the reproducibility of our results. Detailed descriptions of the proposed algorithms, theoretical assumptions, and derivations are provided in the main text and appendices. Hyperparameter settings, model architectures, and training configurations are reported in full. Data preprocessing procedures and evaluation protocols are documented, and all datasets used are publicly available. Our source code and instructions for reproducing experiments are available at https://github.com/yingjie1234/  PATEGAIL-PlusPlus-Utility-Optimized-Private-Trajectory-Generation.
Together, these resources ensure that independent researchers can reproduce and verify our findings.
this section cite: []

Section: A.2 BASELINE METHODS
We compare PATEGAIL++ against five baseline methods: GAN (Goodfellow et al., 2020), Seq-GAN (Yu et al., 2017), Time-Geo (Jiang et al., 2016), MoveSim (Feng et al., 2020), and Diff-Traj (Zhu et al., 2023). We describe each baseline in detail below.
• GAN (Goodfellow et al., 2020) directly generates full trajectories using a vanilla GAN architecture, training a generator and discriminator adversarially without explicit temporal modeling.
• SeqGAN (Yu et al., 2017) extends GANs to sequence generation by training the generator step-by-step via policy gradient, allowing it to produce trajectories one transition at a time.
• TimeGeo (Jiang et al., 2016) is a rule-based probabilistic mobility model grounded in the classical Exploration and Preferential Return (EPR) framework. It characterizes human movement using hand-crafted behavioral rules rather than learned representations.
• MoveSim (Feng et al., 2020) is a GAN-based trajectory generator that incorporates domain knowledge of human mobility regularities to enhance realism and statistical fidelity.
• DiffTraj (Zhu et al., 2023) is a diffusion-based trajectory generation model that learns continuous denoising dynamics over spatial-temporal embeddings. It achieves high-quality trajectory synthesis under a centralized, non-private training setting.
this section cite: ['b11', 'b35', 'b16', 'b9', 'b37', 'b11', 'b35', 'b16', 'b9', 'b37']

Section: A.3 MORE RESULTS

this section cite: []

Section: A.3.1 PRIVACY LEAKAGE
As a complement to the Geolife results discussed in the main text, we further evaluate privacy leakage on the Telecom Shanghai dataset to confirm that the same trends persist across a different mobility distribution. The following tables report both white-box MIA performance and LiRA-based black-box performance under varying noise levels.
The Telecom Shanghai results show a similar overall pattern as in Geolife. Under white-box MIAs, PATEGAIL continues to leak membership information at moderate noise levels, reflected by elevated AUCs between 0.60 and 0.73, while PATEGAIL++ remains more stable and private in terms of AUCs. Moreover, PATEGAIL's attack performance fluctuates substantially across noise levels, whereas PATEGAIL++ exhibits more stable behavior with values consistently near chance. LiRA
this section cite: []

Section: References
Ref_id:b0 Title: Differentially private release of datasets using gaussian copula Year: (2020)
Ref_id:b1 Title: Concentrated differential privacy for bandits Year: (2024)
Ref_id:b2 Title: Federated timegan for privacy preserving synthetic trajectory generation Year: (2024)
Ref_id:b3 Title: Membership inference attacks from first principles Year: (2022)
Ref_id:b4 Title: Secure cloud computing algorithm using homomorphic encryption and multi-party computation Year: (2018)
Ref_id:b5 Title: Ldptrace: Locally differentially private trajectory synthesis Year: (2023-04)
Ref_id:b6 Title:  Year: (2016)
Ref_id:b7 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b8 Title: Tlpp: Deep-learning-based two-layer privacy preserving mechanism for protecting vehicle trajectory data Year: (2024)
Ref_id:b9 Title: Learning to simulate human mobility Year: (2020)
Ref_id:b10 Title: Identifying human mobility via trajectory embeddings Year: (2017)
Ref_id:b11 Title: Generative adversarial networks Year: (2020)
Ref_id:b12 Title: Improved training of wasserstein gans Year: (2017)
Ref_id:b13 Title: Improved training of wasserstein gans Year: (2017)
Ref_id:b14 Title: User allocation-aware edge cloud placement in mobile edge computing. Software: Practice and Experience Year: (2020)
Ref_id:b15 Title: Generative adversarial imitation learning Year: (2016)
Ref_id:b16 Title: The timegeo modeling framework for urban mobility without travel surveys Year: (2016)
Ref_id:b17 Title: Pate-gan: Generating synthetic data with differential privacy guarantees Year: (2018)
Ref_id:b18 Title: Deep learning-based privacy-preserving framework for synthetic trajectory generation Year: (2022)
Ref_id:b19 Title: Profit-aware edge server placement Year: (2021)
Ref_id:b20 Title: Modeling trajectories with neural ordinary differential equations Year: (2021)
Ref_id:b21 Title: T-share: A large-scale dynamic taxi ridesharing service Year: (2013)
Ref_id:b22 Title: St-trajgan: A synthetic trajectory generation algorithm for privacy preservation Year: (2024)
Ref_id:b23 Title: Geogrcnn: Synthetic trajectory generation for location privacy protection Year: (2024)
Ref_id:b24 Title: Privacy preserving human mobility generation using grid based data and graph autoencoders Year: (2024)
Ref_id:b25 Title: Semisupervised knowledge transfer for deep learning from private training data Year: (2017)
Ref_id:b26 Title: Lstm-trajgan: A deep learning approach to trajectory privacy protection Year: (2020)
Ref_id:b27 Title: Apu-trajgen+: Gru-based adaptive privacy and utility preserving trajectory generation Year: (2025)
Ref_id:b28 Title: Learning to generate maps from trajectories Year: (2020)
Ref_id:b29 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b30 Title: Limits of predictability in human mobility Year: (2010)
Ref_id:b31 Title: Privtrace: Differentially private trajectory synthesis by adaptive markov model Year: (2023)
Ref_id:b32 Title: Pategail: A privacy-preserving mobility trajectory generator with imitation learning Year: (2023)
Ref_id:b33 Title: Delay-aware microservice coordination in mobile edge computing: A reinforcement learning approach Year: (2019)
Ref_id:b34 Title: Trajectory simplification with reinforcement learning Year: (2021)
Ref_id:b35 Title: Seqgan: Sequence generative adversarial nets with policy gradient Year: (2017)
Ref_id:b36 Title: Dp-ltgan: Differentially private trajectory publishing via locally-aware transformer-based gan Year: (2025)
Ref_id:b37 Title: Difftraj: Generating gps trajectory with diffusion probabilistic model Year: (2023)
