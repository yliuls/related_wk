Title: EXPLORATORY DIFFUSION MODEL FOR UNSUPERVISED REINFORCEMENT LEARNING
Abstract: Unsupervised reinforcement learning (URL) pre-trains agents by exploring diverse states in reward-free environments, aiming to enable efficient adaptation to various downstream tasks. Without extrinsic rewards, prior methods rely on intrinsic objectives, but heterogeneous exploration data demand strong modeling capacity for both intrinsic reward design and policy learning. We introduce the Exploratory Diffusion Model (ExDM), which leverages the expressive power of diffusion models to fit diverse replay-buffer distributions, thus providing accurate density estimates and a score-based intrinsic reward that drives exploration into under-visited regions. This mechanism substantially broadens state coverage and yields robust pre-trained policies. Beyond exploration, ExDM offers theoretical guarantees and practical algorithms for fine-tuning diffusion policies under limited interactions, overcoming instability and computational overhead from multi-step sampling. Extensive experiments on Maze2d and URLB show that ExDM achieves superior exploration and faster downstream adaptation, establishing new state-of-the-art results, particularly in environments with complex structure or cross-embodiment settings. The source code is provided at https://github.com/yingchengyang/ExDM.

Section: INTRODUCTION
Developing agents that generalize across diverse tasks remains a central challenge in reinforcement learning (RL). Unsupervised RL (URL) (Eysenbach et al., 2018;Laskin et al., 2021) aims to address this by pre-training in reward-free environments to acquire diverse skills or transferable representations. In the absence of extrinsic rewards, agents often rely on intrinsic objectives that are frequently hand-crafted, myopic, and weakly aligned with downstream tasks. The data collected through exploration are highly heterogeneous, demanding representations that are expressive yet stable against collapse or spurious correlations. In addition, policies trained in fixed reward-free settings often fail to transfer under shifts in dynamics, embodiment, or semantics.
A central obstacle in URL is the demand for strong modeling capacity during both pre-training and fine-tuning. Effective exploration in reward-free environments hinges on intrinsic rewards derived from accurate estimates of the underlying state distribution, which is typically heterogeneous and difficult to capture. Existing methods can collect diverse trajectories but often rely on simple pretrained policies-such as Gaussian (Pathak et al., 2017;Mazzaglia et al., 2022) or discrete skillbased policies (Eysenbach et al., 2018;Laskin et al., 2022)-chosen for their ease of training and sampling. Such policies fail to capture the full diversity of explored data in the replay buffer, limiting both unsupervised exploration and downstream adaptation. This calls for more powerful modeling approaches, where diffusion models stand out for their stability and strong density estimation ability.
To address these challenges, we propose the Exploratory Diffusion Model (ExDM), which leverages diffusion-based density estimation to address the exploration bottleneck in unsupervised RL while providing a reusable prior for downstream adaptation. At its core, ExDM trains a diffusion model on the heterogeneous and nonstationary state distribution in the replay buffer, and uses the resulting Beyond enhancing unsupervised exploration, ExDM also provides a strong initialization for downstream tasks. In addition to fine-tuning the Gaussian behavior policy with standard RL algorithms, ExDM allows the diffusion model itself to be adapted for downstream control. This adaptation is particularly challenging in URL, where fine-tuning must succeed with limited online interaction.
To this end, we analyze the fine-tuning objective and derive an alternating optimization procedure whose convergence and optimality are formally established in Theorem 4.2.
We evaluate ExDM on both unsupervised exploration and downstream adaptation across standard benchmarks, including Maze2d (Campos et al., 2020) and continuous control in URLB (Laskin et al., 2022). In Maze2d, ExDM consistently achieves substantially higher state coverage than all baselines. On the most challenging mazes with many branching paths and decision points, ExDM attains up to 51% higher coverage and reaches comparable performance using only 37% of timesteps, demonstrating its ability to efficiently explore diverse regions under strict interaction budgets, while baselines often stall near corners and fail to cover the maze. Beyond exploration, URLB experiments, including single-embodiment and cross-embodiment settings, further show that ExDM adapts rapidly to diverse downstream tasks and outperforms SOTA URL and diffusion fine-tuning baselines by large margins, highlighting its effectiveness as a general framework for exploration and transfer.
In summary, the main contributions are as follows:
• To the best of our knowledge, this is the first work to introduce diffusion models into unsupervised RL, enabling accurate modeling of heterogeneous state distributions and defining a score-based intrinsic reward that substantially improves exploration.
• Beyond exploration, ExDM develops an efficient decoupled training scheme and a finetuning algorithm for adapting pre-trained diffusion components to downstream tasks under limited interaction, with theoretical guarantees of convergence and optimality.
• Extensive experiments on Maze2d and URLB benchmarks demonstrate that ExDM achieves broader state coverage and faster adaptation than prior methods, establishing new state-of-the-art performance in both exploration and transfer.
this section cite: ['b16', 'b52', 'b46', 'b16', 'b32', 'b6', 'b32']

Section: RELATED WORK
Unsupervised Pre-training in RL. For achieving zero-shot generalization in RL, there are lots of attempts like in-context RL (Rakelly et al., 2019;Zintgraf et al., 2021) or forward-backward repre-sentations (Touati & Ollivier, 2021;Tirinzoni et al., 2025). However, these methods require sampling from tasks during pre-training or reward signals from the offline datasets. Differently, URL pre-trains agents in reward-free environments to acquire knowledge for fast fine-tuning downstream tasks. Existing methods mainly rely on intrinsic rewards to guide agents to explore the environment, falling into two categories: exploration and skill discovery. Exploration methods typically explore diverse states by maximizing intrinsic rewards designed to estimate either uncertainty (Pathak et al., 2017;Burda et al., 2018;Pathak et al., 2019;Raileanu & Rocktäschel, 2020;Mazzaglia et al., 2022;Yuan et al., 2023;Ying et al., 2024) or state entropy (Lee et al., 2019;Liu & Abbeel, 2021;Seo et al., 2021;Mutti et al., 2021). Skill-discovery methods hope to collect diverse skills by maximizing the mutual information between skills and states (Eysenbach et al., 2018;Lee et al., 2019;Campos et al., 2020;Kim et al., 2021;Park et al., 2022;Laskin et al., 2022;Yuan et al., 2022;Zhao et al., 2022;Yang et al., 2023b;Park et al., 2023;Bai et al., 2024;Wilcoxson et al., 2024). Although exploring diverse states, existing methods always neglect the expression ability of pre-trained policies and choose simple Gaussian policies (Pathak et al., 2017;Mazzaglia et al., 2022) or skill-based policies (Eysenbach et al., 2018;Yang et al., 2023b), which fail to capture the diversity present in the explored data. Consequently, applying generative models with strong expressive ability for improving the diversity of pre-trained policies is still less studied.
this section cite: ['b57', 'b81', 'b66', 'b65', 'b52', 'b5', 'b53', 'b56', 'b46', 'b76', 'b75', 'b33', 'b39', 'b60', 'b47', 'b16', 'b33', 'b6', 'b29', 'b50', 'b32', 'b77', 'b78', 'b51', 'b4', 'b71', 'b52', 'b46', 'b16']

Section: RL with Diffusion Models.
Recent advancements have shown that high-fidelity diffusion models can benefit RL from different perspectives (Zhu et al., 2023). In offline RL, diffusion policies (Wang et al., 2023;Chen et al., 2023;Lu et al., 2023;Chi et al., 2023;Hansen-Estruch et al., 2023;Kang et al., 2024) excel at modeling multimodal behaviors, outperforming previous policies such as Gaussians. Besides policies, diffusion planners (Janner et al., 2022;Ajay et al., 2023;He et al., 2023;Liang et al., 2023;Nuti et al., 2023;Chen et al., 2024a) have demonstrated the potential in long-term sequence prediction and test-time planning. Some works have also investigated online training diffusion policies to improve performance (Psenka et al., 2023;Li et al., 2024;Ren et al., 2024;Mark et al., 2024;Celik et al., 2025;Ma et al., 2025;Ishfaq et al., 2025). However, the computational cost of multi-step sampling remains the efficiency bottleneck. In addition to behavior modeling, diffusion models have also been employed as world models (Alonso et al., 2024;Ding et al., 2024), augmented replay buffer (Lu et al., 2024;Wang et al., 2024a), hierarchical RL (Li et al., 2023;Chen et al., 2024b), and so on. To the best of our knowledge, this work represents the first attempt to leverage the strong modeling capabilities for heterogeneous distribution of diffusion models for unsupervised exploration.
this section cite: ['b79', 'b68', 'b42', 'b42', 'b11', 'b22', 'b28', 'b27', 'b1', 'b23', 'b36', 'b49', 'b55', 'b35', 'b58', 'b45', 'b7', 'b44', 'b26', 'b2', 'b14', 'b43', 'b34']

Section: BACKGROUND
3.1 UNSUPERVISED REINFORCEMENT LEARNING RL considers Markov decision processes (MDP) M = (S, A, P, R, ρ 0 , γ). Here S and A denote the state and action spaces, respectively. For ∀(s, a) ∈ S × A, P(•|s, a) is a distribution on S, representing the dynamic of M, and R(s, a) is the extrinsic task reward function. ρ 0 is the initial state distribution and γ is the discount factor. For a given policy π : S → ∆(A), we define the discount state distribution of π at state s as
d π (s) = (1 -γ) ∞ t=0 [γ t P(s t = s)].
The objective of RL is to maximize the expected cumulative return of π over the task R:
J(π) ≜ E τ ∼M,π [R(τ )] = 1 1 -γ E s∼dπ,a∼π [R(s, a)] .(1)
To boost agents' generalization, unsupervised RL (URL) typically includes two stages: unsupervised pre-training and few-shot fine-tuning. During pre-training, agents explore the reward-free environment M c , i.e., M without the reward function R. Thus, URL requires designing intrinsic rewards R int to guide policies to maximize the state entropy H(d π (•)). During fine-tuning, agents adapt pre-trained policies to handle downstream tasks represented by extrinsic task-specific rewards R, through limited interactions (like one-tenth of pre-training steps, the formulation is in Eq. 9).
this section cite: []

Section: DIFFUSION MODELS IN REINFORCEMENT LEARNING
Recent studies have demonstrated that diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020) excel at accurately representing heterogeneous behaviors in continuous control, particularly through the use of diffusion policies (Wang et al., 2023;Chi et al., 2023). Given state-action pairs (s, a) sampled from some unknown policy µ(a|s), diffusion policies consider the forward diffusion process that gradually injects standard Gaussian noise ϵ into actions:
a t = α t a + σ t ϵ, t ∈ [0, 1],(2)
here α t , σ t are pre-defined hyperparameters satisfying that when t = 0, we have a t = a, and when t = 1, we have a t ≈ ϵ. For ∀t ∈ [0, 1], we can define the marginal distribution of a t as p t (a t |s, t) = N (a t |α t a, σ 2 t I)µ(a|s)da.
Then we train a conditional "noise predictor" ϵ θ (a t |s, t) to predict the added noise of each timestep:
min θ E t,ϵ,s,a [∥ϵ θ (a t |s, t) -ϵ∥ 2 ].(4)
The learned ϵ θ can estimate the score function ∇ at log p t (a t |s t , t). We can discretize diffusion ODEs of the reverse process (Song et al., 2021b) and sample actions with numerical solvers (Song et al., 2021a;Lu et al., 2022) in around 5 ∼ 15 steps, to approximate the original policy µ(a|s).
However, this multi-step sampling affects the training efficiency, especially in online settings.
this section cite: ['b61', 'b25', 'b68', 'b11', 'b41']

Section: METHODOLOGY
Below, we introduce the Exploratory Diffusion Model (ExDM) to capture diverse data to boost unsupervised exploration (Sec. 4.1) and obtain powerful initialization for fast fine-tuning (Sec. 4.2).
this section cite: []

Section: EXPLORATORY DIFFUSION MODEL FOR UNSUPERVISED PRE-TRAINING
The major challenge and objective during unsupervised pre-training is to explore diverse states in reward-free environments. Consequently, a natural pathway is to pre-train the policy to maximize the entropy of the state (Liu & Abbeel, 2021), i.e., H(d π (•)) = s -d π (s) log d π (s)ds. Although the optimal policy of fully-observable single-agent RL is a simple deterministic policy, we prove that, even if the environment is discrete, policies with the maximum state entropy are still complicated and not deterministic with a high probability, requiring much stronger modeling abilities. Theorem 4.1 (Policy with maximal state entropy). When S, A are discrete spaces, i.e., |S| = S, |A| = A, there are M ≜ A S deterministic policies. Set π = arg max π H(d π (•)), under some mild assumptions, we have
P (π is not deterministic policy and H(d π ) = log |S|) ≥1 -M S v(S) M ,(5)
will fast converge to 1 with the increasing of A, and here v(S) is a constant only related to S and satisfies 0 < v(S) < 1.
Details and proof are in Appendix B.1 (we also discuss continuous situations there). This theorem demonstrates that maximizing state entropy requires policies with strong expression abilities, rather than simple deterministic policies. Despite previous work mainly considering simple Gaussian policies or skill-based policies, in practice, explored replay buffer is always diverse and heterogeneous, as the policy continuously changes to visit new states during pre-training. Consequently, URL requires capturing the heterogeneous distribution of collected data and obtaining policies with high diversity. These challenges pose the requirement of strong density estimation and fitting abilities, while maintaining training stability and efficiency. Inspired by the recent great success of diffusion models in modeling diverse image distributions (Dhariwal & Nichol, 2021) and behaviors (Chi et al., 2023;Janner et al., 2022), ExDM proposes to utilize the diffusion models ϵ θ ′ and ϵ θ to model the distribution of states and state-action pairs in the replay buffer D collected before:
min E s,a∼D [E t,ϵ ∥ϵ θ ′ (s t |t) -ϵ∥ 2 + E t,ϵ ∥ϵ θ (a t |s, t) -ϵ∥ 2 ].(6)
To maximize the entropy of the state distribution, we can use log p θ ′ (s) to measure the frequency of states in the replay buffer. Consequently, we design -log p θ ′ (s) as the intrinsic reward to encourage the agent to explore these regions. Although estimating the log-probability of the diffusion model
this section cite: ['b39', 'b13', 'b11', 'b27']

Section: Algorithm 1 Pre-training of ExDM
Require: Reward-free environment M c , replay buffer D, Gaussian behavior policy π g , diffusion policy π d parameterized with the score model ϵ θ , state diffusion model ϵ θ ′ . 1: for sample step = 1, 2, ..., S do 2:
for update step = 1, 2, ..., U do 3:
Sample s-a pairs {(s m , a m )} M m=1 from D.
4:
Update ϵ θ and ϵ θ ′ via optimizing with Eq. ( 6) with sampled data.
5:
Calculate score-based intrinsic rewards r m via Eq. ( 8) for each sampled pair (s m , a m ).
6:
Train π g with (s m , a m , r m ) by any off-policy RL algorithm.
7:
end for 8:
Utilize the behavior policy π g to interact with M c and store state-action pairs into D. 9: end for is challenging, it is well known that -log p θ ′ (s) can be bounded by the following evidence lower bound (ELBO) (Ho et al., 2020):
-log p θ ′ (s) ≤ E ϵ,t [w t ∥ϵ θ ′ (s t |t) -ϵ∥ 2 ] + C,(7)
here C is a constant independent of θ ′ , and w t are parameters related to α t , σ t , which are typically ignored (Ho et al., 2020). Consequently, we propose our score-based intrinsic rewards as:
R score (s) = E ϵ,t [∥ϵ θ ′ (s|t) -ϵ∥ 2 ].(8)
Intuitively, our score-based intrinsic rewards can measure the fitting quality of the diffusion model to the explored data, thereby encouraging the agent to explore regions that are poorly fitted or unexplored (more analyses between R score and -log p θ ′ are in Appendix C.1). By maximizing these intrinsic rewards, ExDM trains agents to discover unseen regions effectively. However, directly using diffusion policies to interact with reward-free environments during pre-training is inefficient and unstable due to the requirement of multi-step sampling. To address this limitation, ExDM incorporates a Gaussian behavior policy π g for efficient action sampling. Gaussian behavior policy π g can be trained using any off-policy RL algorithm, guided by score-based intrinsic rewards R score (s). This encourages the exploration of regions where the diffusion model either fits poorly or has not yet been exposed. The pseudo code of the unsupervised exploration stage of ExDM is in Algorithm 1.
this section cite: ['b25', 'b25']

Section: EFFICIENT ONLINE FINE-TUNING TO DOWNSTREAM TASKS
When adapting pre-trained policies to downstream tasks with limited timesteps, existing URL methods always directly apply online RL algorithms like DDPG (Lillicrap, 2015) or PPO (Schulman et al., 2017) for fine-tuning. The behavior policy π g in ExDM can also be fine-tuned to handle the downstream task with the same online RL algorithms, performing fair comparison of exploration efficiency between ExDM and baselines (detailed experimental results are in Sec. 5.3).
Besides π g , ExDM has also pre-trained the diffusion policy π d , which can better capture the heterogeneous explored trajectories for adapting to downstream tasks. Unfortunately, it is challenging to online fine-tune diffusion policies due to the instability caused by the multi-step sampling and the lack of closed-form probability calculation (Ren et al., 2024). To address these challenges, we first analyze the online fine-tuning objective for URL. Given the limited fine-tuning timesteps, the objective can be formulated as the combination of maximizing the cumulative return and keeping close to the pre-trained policy over all s (Eysenbach et al., 2021) (more analyses are in Appendix C.2):
max π J f (π) ≜J(π) - β (1 -γ) E s∼dπ [D KL (π(•|s)∥π d (•|s))] = 1 1 -γ E s∼dπ,a∼π [R(s, a) -βD KL (π(•|s)∥π d (•|s))] = 1 1 -γ E s∼dπ,a∼π R(s, a) -β log π(a|s) π d (a|s) ,(9)
here β > 0 is an unknown trade-off parameter that is related to fine-tuning steps. J f (π) can be interpreted as penalizing the probability offset of the policy in (s, a) over π and π d . More specifically, it
this section cite: ['b37', 'b59', 'b58', 'b17']

Section: Algorithm 2 Diffusion Policy Fine-tuning of ExDM
Require: Environment M with rewards R, replay buffer D, pre-trained diffusion policy π d parameterized with the score model ϵ θ , fine-tuned diffusion policy ϵ ψ . 1: for update iteration n = 1, 2, ..., N do 2:
Sample s-a-r pairs {(s m , a m , r m )} M m=1 from D.
3: Update Q function with IQL and update Guidance f ϕn-1 with CEP. 4:
Optimize ψ by score distillation with Eq. ( 14).
5:
for interaction step = 1, 2, ..., S do 6:
Interact with M by ϵ ψ and store state-action-reward pairs into D.
7:
end for 8: end for aims to maximize a surrogate reward of the form R(s, a) -β log π(a|s) π d (a|s) . However, this surrogate reward depends on the policy π, and we cannot directly apply the classical RL analyses. Inspired by soft RL (Haarnoja et al., 2017) and offline RL (Peng et al., 2019), we define our Q functions as:
Q π (s, a) =E R(s, a) + ∞ i=1 γ i R(s i , a i ) -β log π(a i |s i ) π d (a i |s i ) .(10)
Based on this Q function, we can simplify J f as
J f (π) =E s∼ρ0,a∼π [Q π (s, a) -βD KL (π(•|s)∥π d (•|s))] .(11)
To optimize J f , ExDM decouples optimizing Q functions and diffusion policies. In detail, we initial
π 0 = π d , Q 0 = Q π0 , then for n = 1, 2, ..., we set π n (•|s) ≜ arg max π E a∼π Q πn-1 (s, a) -βD KL (π(•|s)∥π d (•|s)) = π d (a|s)e Qn-1(s,a)/β Z(s) , Q n ≜Q πn ,(12)
here Z(s) = π d (a|s)e Qn-1(s,a)/β da. Building on soft RL analysis (Haarnoja et al., 2017;2018), we show the policy improvement of each iteration and the optimality of the alternating optimization: Theorem 4.2 (Optimality of ExDM, Proof in Appendix B.2). ExDM can achieve policy improvement, i.e., J f (π n ) ≥ J f (π n-1 ) for ∀n ≥ 1. And π n will converge to the optimal policy of J f .
Compared with offline RL, in which Q functions are related to offline datasets, Q functions here are related to current policies, which introduces extra challenges as Q functions change correspondingly during fine-tuning. Below, we introduce the practical diffusion policy fine-tuning method of ExDM for both updating Q functions and diffusion policies, respectively (pseudo-code in Algorithm 2).
this section cite: ['b20', 'b54', 'b20']

Section: Q function optimization.
Our principle for updating Q functions is to penalize actions with large log probability ratios between π and π d . Thus, we apply implicit Q-learning (IQL) (Kostrikov et al., 2022), which leverages expectile regression to penalize out-of-distribution actions (Appendix C.3).
this section cite: ['b30']

Section: Diffusion policy distillation.
At each iteration n with Q function Q n-1 , calculating π n by Eq. ( 12) is difficult as Z(s) is a complicated integral. However, sampling from π n can be regarded as sampling from π d with energy guidance Q n-1 , i.e., guided sampling (Janner et al., 2022). Especially, we employ contrastive energy prediction (CEP) (Lu et al., 2023) to sample from ∝ π d e Qn-1/β and parameterize f ϕn-1 (s, a t , t) to represent timestep t's energy guidance, which can be optimizated as:
min ϕn-1 E t,s E a 1 ,...,a K ∼π d (•|s) - K i=1 e Qn-1(s,a i )/β K j=1 e Qn-1(s,a j )/β log f ϕn-1 (s, a i t , t) K j=1 f ϕn-1 (s, a j t , t) .(13)
Then ExDM fine-tunes diffusion policies by distilling the score of π n parameterized by ϵ ψ (a t |s, t):
min ψ E s,a,t ∥ϵ ψ (a t |s, t) -ϵ θ (a t |s, t) -f ϕn-1 (s, a t , t)∥ 2 .(14)
Finally, we can directly sample from ϵ ψ to generate action of π n (details are in Appendix C.4). Domains Square-a Square-b Square-c Square-d Square-tree Square-bottleneck Square-large ICM 0.58 ± 0.04 0.53 ± 0.06 0.47 ± 0.07 0.49 ± 0.06 0.49 ± 0.05 0.32 ± 0.07 0.25 ± 0.04 RND 0.50 ± 0.14 0.39 ± 0.08 0.52 ± 0.16 0.32 ± 0.05 0.28 ± 0.06 0.33 ± 0.06 0.33 ± 0.08 Disagreement 0.38 ± 0.10 0.30 ± 0.10 0.41 ± 0.19 0.29 ± 0.11 0.32 ± 0.11 0.28 ± 0.04 0.21 ± 0.06 LBS 0.32 ± 0.04 0.29 ± 0.09 0.27 ± 0.05 0.25 ± 0.03 0.22 ± 0.03 0.21 ± 0.02 0.19 ± 0.06 RE3 0.85 ± 0.09 0.72 ± 0.22 0.73 ± 0.16 0.74 ± 0.01 0.73 ± 0.04 0.62 ± 0.01 0.46 ± 0.03 MEPOL 0.98 ± 0.03 0.99 ± 0.02 0.96 ± 0.07 0.77 ± 0.01 0.89 ± 0.06 0.62 ± 0.01 0.59 ± 0.04 DIAYN 0.41 ± 0.06 0.44 ± 0.04 0.42 ± 0.04 0.37 ± 0.03 0.38 ± 0.06 0.29 ± 0.04 0.30 ± 0.04 SMM 0.47 ± 0.13 0.45 ± 0.20 0.36 ± 0.08 0.28 ± 0.04 0.25 ± 0.02 0.41 ± 0.13 0.34 ± 0.10 LSD 0.45 ± 0.03 0.38 ± 0.05 0.36 ± 0.03 0.35 ± 0.03 0.28 ± 0.03 0.34 ± 0.03 0.32 ± 0.03 CIC 0.94 ± 0.02 0.98 ± 0.01 0.86 ± 0.03 0.74 ± 0.01 0.89 ± 0.01 0.58 ± 0.05 0.47 ± 0.01 BeCL 0.50 ± 0.08 0.48 ± 0.11 0.42 ± 0.10 0.37 ± 0.03 0.36 ± 0.06 0.29 ± 0.06 0.25 ± 0.05 CeSD 0.70 ± 0.04 0.79 ± 0.04 0.67 ± 0.06 0.46 ± 0.06 0.37 ± 0.06 0.46 ± 0.03 0.40 ± 0.01 ExDM (Ours) 0.99 ± 0.02 0.99 ± 0.01 0.98 ± 0.02 0.78 ± 0.01 0.91 ± 0.01 0.75 ± 0.15 0.71 ± 0.07 Table 1: State coverage in Maze. We report the mean and std of 10 seeds for each algorithm.
this section cite: ['b27', 'b42']

Section: EXPERIMENTS
In this section, we present extensive empirical results to mainly address the following questions:
• Can ExDM boost the unsupervised exploration efficiency, especially in complicated mazes with numerous branching paths and decision points? (Sec. 5.2)
• What about the adaptation efficiency of the pre-trained Gaussian policies of ExDM compared to other URL baselines? (Sec. 5.3)
• As for fast fine-tuning pre-trained diffusion policies to downstream tasks, how does the performance of ExDM compare to existing baselines? (Sec. 5.4)
this section cite: []

Section: EXPERIMENTAL SETUP
Maze2d. We first evaluate the exploration diversity during the unsupervised stage in widely used maze2d environments (Campos et al., 2020;Yang et al., 2023b): Square-a, Square-b, Square-c, Square-d, Square-tree, Square-bottleneck, and Square-large. Observations and actions here belong to R 2 . When interacting with mazes, agents will be blocked when they contact walls.
this section cite: ['b6']

Section: Continuous Control.
To evaluate the performance of fine-tuning in downstream tasks, we choose 4 continuous control settings in URLB (Laskin et al., 2021): Walker, Quadruped, Jaco, and Hopper. Each domain contains four downstream tasks. More details are in Appendix D.1.
Baselines. In Maze2d experiments, we take 6 exploration baselines: ICM (Pathak et al., 2017), RND (Burda et al., 2018), Disagreement (Pathak et al., 2019), RE3 (Seo et al., 2021), MEPOL (Mutti et al., 2021), and LBS (Mazzaglia et al., 2022); as well as 6 skill discovery baselines: DIAYN (Eysenbach et al., 2018), SMM (Lee et al., 2019), LSD (Park et al., 2022), CIC (Laskin et al., 2022), BeCL (Yang et al., 2023b), and CeSD (Bai et al., 2024), which are standard and SOTA. As for fine-tuning in URLB, we consider three settings: (a) for a fair comparison, we directly utilize
0.2 0.4 0.6 0.8 ExDM (Ours) CeSD BeCL CIC RND Disagreement MEPOL RE3 LBS ICM SMM DIAYN LSD Median ( ) 0.2 0.4 0.6 0.8 IQM ( ) 0.2 0.4 0.6 0.8 Mean ( ) 0.4 0.6 0.8 Optimality Gap ( ) Expert Normalized Score (a) URLB fine-tuned by DDPG 0.45 0.60 0.75 ExDM (Ours) PEAC BeCL CIC RND LBS SMM DIAYN Disagreement ICM Median ( ) 0.45 0.60 0.75 IQM ( ) 0.45 0.60 0.75 Mean ( ) 0.30 0.45 0.60 Optimality Gap ( ) Expert Normalized Score (b) Cross-embodiment URLB fine-tuned by DDPG 0.2 0.4 0.6 ExDM (Ours) DIPO IDQL DQL QSM Median ( ) 0.2 0.4 0.6 IQM ( ) 0.2 0.4 0.6 Mean ( ) 0.4 0.6 0.8 Optimality Gap ( ) Expert Normalized Score (c) URLB for fine-tuning diffusion policies DDPG (Sohl-Dickstein et al., 2015) to fine-tune the pre-trained behavior Gaussian policy in ExDM, compared to existing URL baselines, including ICM, RND, Disagreement, RE3, MEPOL, LBS, DIAYN, SMM, LSD, CIC, BeCL, and CeSD (all baselines fine-tuned by DDPG, the standard RL backbone in URLB, except CeSD fine-tuned by ensembled DDPG); (b) We consider complicated cross-embodiment URL (Ying et al., 2024), comparing ExDM with ICM, Disagreement, RND, LBS, DIAYN, SMM, CIC, BeCL, and PEAC (Ying et al., 2024); (c) ExDM also fine-tunes pre-trained diffusion policies, compared to diffusion policy fine-tuned baselines, like DQL (Wang et al., 2023), IDQL (Hansen-Estruch et al., 2023), QSM (Psenka et al., 2023), and DIPO (Yang et al., 2023a).
Metrics. In Maze2d, we pre-train agents in reward-free environments with 100k steps and visualize all collected trajectories. Moreover, to quantitatively compare the exploration efficiency, we evaluate the state coverage ratios, which are measured as the proportion of 0.01 × 0.01 square bins visited. As for URLB, following standard settings, we pre-train agents in reward-free environments for 2M steps and fine-tune pre-trained policies to adapt each downstream task within extrinsic rewards for 100K steps. All settings are run for 10 seeds to mitigate the effectiveness of randomness.
this section cite: ['b52', 'b5', 'b53', 'b60', 'b47', 'b46', 'b33', 'b50', 'b32', 'b4']

Section: UNSUPERVISED PRE-TRAINING FOR EXPLORATION
In Fig. 2, we visualize the heatmap of collected trajectories during pre-training in complicated Square-bottleneck and Square-large (results of all 7 mazes and 13 baselines are in Appendix D.4).
To quantitatively evaluate the exploration efficiency of each algorithm, we further report the state coverages in Table 1 (training curves are in Fig. 8 of Appendix D.4). In both qualitative visualization and quantitative metrics, ExDM outperforms baselines by large margins. Especially, in complicated mazes like Square-bottleneck and Square-large with many different branching points (Fig. 2), all baselines will struggle at some wall corner and cannot explore the entire maze. In contrast, ExDM
N N 0 0 3UHWUDLQLQJ6WHSV ([SHUW1RUPDOL]HG6FRUH /6' ',$<1 600 ,&0 /%6 'LVDJUHHPHQW 51' &,& %H&/ &H6' (['0 (a) Pre-training timesteps 6WDQG :DON 5XQ -XPS ([SHUW1RUPDOL]HG6FRUH 4XDGUXSHG (['0ZR,4/ (['0 (b) Q Learning choices 'LIIXVLRQ3ROLF\6DPSOLQJ6WHS ([SHUW1RUPDOL]HG6FRUH 4XDGUXSHG 6WDQG :DON 5XQ -XPS (c) Diffusion steps 6WDQG :DON 5XQ -XPS ([SHUW1RUPDOL]HG6FRUH 4XDGUXSHG (['0 (['0 (['0 (d) β for Fine-tuning successfully explores almost the whole maze, demonstrating that our R score , leveraging the accurate data estimation ability of diffusion models, can effectively guide agents to explore diverse states.
this section cite: []

Section: FINE-TUNING THE GAUSSIAN POLICY TO DOWNSTREAM TASKS
We verify the ability of ExDM to fine-tune downstream tasks in both single-embodiment and crossembodiment URLB. As existing URL methods directly fine-tune policies with DDPG, for a fair comparison, we also use DDPG to fine-tune pre-trained Gaussian policies π g in ExDM. Following previous settings, we train DDPG agents for each downstream task with 2M steps to obtain the expert return and calculate the expert-normalized score for each algorithm. In Fig. 3(a)-3(b), we compare all methods with four metrics: mean, median, interquartile mean (IQM), and optimality gap (OG), along with stratified bootstrap confidence intervals. ExDM significantly outperforms all existing SOTA methods, demonstrating that introducing diffusion models can lead to more efficient generalization in downstream tasks. Details of each downstream task are in Appendix D.5.
this section cite: []

Section: FINE-TUNING THE DIFFUSION POLICY TO DOWNSTREAM TASKS
Moreover, ExDM has pre-trained diffusion policies that can capture the diversity of explored trajectories and adapt to downstream tasks. Consequently, in Fig. 3(c), ExDM substantially outperforms existing diffusion online fine-tuning baselines, demonstrating the efficiency of its alternating optimization. However, diffusion policy fine-tuning performance is still lower than Gaussian policy performance, which may be due to limited interaction timesteps during fine-tuning. It is an interesting future direction to design more efficient diffusion online fine-tuning methods.
this section cite: []

Section: ABLATION STUDIES
Pre-training Steps. We first do ablation studies on pre-trained steps (100k, 500k, 1M, and 2M) to evaluate fine-tuned performance (100k fine-tuned steps). As shown in Fig. 4(a), ExDM markedly exceeds all baselines from 500k steps, indicating that the diffusion model enhances fine-tuning. Moreover, ExDM substantially improves with increasing pre-training timesteps, showing that the unsupervised exploration benefits downstream tasks. Additional results are in Appendix D.6.
this section cite: []

Section: Q function optimization.
We conduct ablation studies to evaluate the impact of Q learning methods during fine-tuning. In detail, we introduce ExDM w/o IQL, which utilizes In-support Softmax Q-Learning (Lu et al., 2023) for optimizing Q functions.
this section cite: ['b42']

Section: Sampling steps of diffusion policies.
During fine-tuning diffusion policies, ExDM requires sampling actions from diffusion policies for both trajectory generation and final evaluation. We adopt DPM-Solver (Lu et al., 2022) to accelerate sampling. For trajectory collection, we set the diffusion step to 15, following previous works (Lu et al., 2023). Then we conduct ablation studies between the diffusion sampling steps used during inference and the fine-tuned performance. Fig. 4(c) shows that performance improves as diffusion steps increase and gradually stabilizes when the step exceeds 5.
Ablation study on β. We consider the objective J f with the behavior regularization term because the fine-tuning step is limited (more analyses are in Appendix C.2). The parameter β implicitly relies on the fine-tuning steps. If the fine-tuning steps are infinite, the optimal β should be 0, and J f degrades into J. We set β = 1/3.0 in experiments (following previous work (Lu et al., 2023)) and do ablations with different β in Fig. 4(d), showing that ExDM performs relatively stably of β.
Time cost of ExDM. One of the major concerns for diffusion models is their time cost due to multi-step sampling. This problem may be more severe in online RL, as each collected trajectory requires sampling from diffusion policies. To address it, ExDM decouples modeling from acting, i.e., utilizing Gaussian behavior policies π g for sampling. Thus, ExDM exhibits high training efficiency for both timesteps and training time. For example, in Square-large, RND achieves the state coverage of 0.33 with 100k timesteps and 1000s of time. ExDM can achieve 0.71 state coverage within 100k timesteps, and achieve 0.4 state coverage with the same time cost (1000s) and only 16.6k timesteps.
this section cite: ['b41', 'b42', 'b42']

Section: CONCLUSION
Unsupervised exploration is one of the major problems in RL for improving task generalization, as it relies on accurate intrinsic rewards to guide exploration of unseen regions. In this work, we address the challenge of limited policy expressivity in previous exploration methods by leveraging the powerful expressive ability of diffusion policies. In detail, our Exploratory Diffusion Model (ExDM) improves exploration efficiency during pre-training while generating policies with high behavioral diversity. We also provide a theoretical analysis of diffusion policy fine-tuning, along with practical alternating optimization methods. Experiments in various settings demonstrate that ExDM can effectively benefit both pre-training exploration and fine-tuning performance. We hope this work can inspire further research in developing high-fidelity generative models to improve unsupervised exploration, particularly in large-scale pre-trained agents or real-world control applications.
this section cite: []

Section: References
Ref_id:b0 Title: Deep reinforcement learning at the edge of the statistical precipice Year: (2021)
Ref_id:b1 Title: Is conditional generative modeling all you need for decision making? Year: (2023)
Ref_id:b2 Title: Diffusion for world modeling: Visual details matter in atari Year: (2024)
Ref_id:b3 Title: Random polytopes, convex bodies, and approximation. Stochastic Geometry: Lectures given at the CIME Summer School Year: (2004)
Ref_id:b4 Title: Constrained ensemble exploration for unsupervised skill discovery Year: (2024)
Ref_id:b5 Title: Exploration by random network distillation Year: (2018)
Ref_id:b6 Title: Explore, discover and learn: Unsupervised discovery of state-covering skills Year: (2020)
Ref_id:b7 Title: Dime: Diffusion-based maximum entropy reinforcement learning Year: (2025)
Ref_id:b8 Title: Diffusion forcing: Next-token prediction meets full-sequence diffusion Year: (2024)
Ref_id:b9 Title: Simple hierarchical planning with diffusion Year: (2024)
Ref_id:b10 Title: Offline reinforcement learning via high-fidelity generative behavior modeling Year: (2023)
Ref_id:b11 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b12 Title: Model-based reinforcement learning via meta-policy optimization Year: (2018)
Ref_id:b13 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b14 Title: Diffusion world model Year: (2024)
Ref_id:b15 Title: Rl 2: Fast reinforcement learning via slow reinforcement learning Year: (2016)
Ref_id:b16 Title: Diversity is all you need: Learning skills without a reward function Year: (2018)
Ref_id:b17 Title: The information geometry of unsupervised reinforcement learning Year: (2021)
Ref_id:b18 Title: Model-agnostic meta-learning for fast adaptation of deep networks Year: (2017)
Ref_id:b19 Title: Bootstrapped meta-learning Year: (2021)
Ref_id:b20 Title: Reinforcement learning with deep energy-based policies Year: (2017)
Ref_id:b21 Title: Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b22 Title: Idql: Implicit q-learning as an actor-critic method with diffusion policies Year: (2023)
Ref_id:b23 Title: Diffusion model is an effective planner and data synthesizer for multi-task reinforcement learning Year: (2023)
Ref_id:b24 Title: Meta-model-based meta-policy optimization Year: (2021)
Ref_id:b25 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b26 Title: Langevin soft actor-critic: Efficient exploration through uncertainty-driven critic learning Year: (2025)
Ref_id:b27 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
Ref_id:b28 Title: Efficient diffusion policies for offline reinforcement learning Year: (2024)
Ref_id:b29 Title: Unsupervised skill discovery with bottleneck option learning Year: (2021)
Ref_id:b30 Title: Offline reinforcement learning with implicit qlearning Year: (2022)
Ref_id:b31 Title: Urlb: Unsupervised reinforcement learning benchmark Year: ()
Ref_id:b32 Title: Xue Bin Peng, Denis Yarats, Aravind Rajeswaran, and Pieter Abbeel. Unsupervised reinforcement learning with contrastive intrinsic control Year: (2022)
Ref_id:b33 Title: Efficient exploration via state marginal matching Year: (2019)
Ref_id:b34 Title: Hierarchical diffusion for offline decision making Year: (2023)
Ref_id:b35 Title: Learning multimodal behaviors from scratch with diffusion policy gradient Year: (2024)
Ref_id:b36 Title: Adaptdiffuser: Diffusion models as adaptive self-evolving planners Year: (2023)
Ref_id:b37 Title: Continuous control with deep reinforcement learning Year: (2015)
Ref_id:b38 Title: A theoretical understanding of gradient bias in meta-reinforcement learning Year: (2022)
Ref_id:b39 Title: Behavior from the void: Unsupervised active pre-training Year: (2021)
Ref_id:b40 Title: Energy-guided diffusion sampling for offline-to-online reinforcement learning Year: (2024)
Ref_id:b41 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b42 Title: Contrastive energy prediction for exact energy-guided diffusion sampling in offline reinforcement learning Year: (2023)
Ref_id:b43 Title: Synthetic experience replay. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b44 Title: Soft diffusion actor-critic: Efficient online reinforcement learning for diffusion policy Year: (2025)
Ref_id:b45 Title: Policy agnostic rl: Offline rl and online rl fine-tuning of any class and backbone Year: (2024)
Ref_id:b46 Title: Curiosity-driven exploration via latent bayesian surprise Year: (2022)
Ref_id:b47 Title: Task-agnostic exploration via policy gradient of a non-parametric state entropy estimate Year: (2021)
Ref_id:b48 Title: Learning to adapt in dynamic, real-world environments through metareinforcement learning Year: (2018)
Ref_id:b49 Title: Extracting reward functions from diffusion models Year: (2023)
Ref_id:b50 Title: Lipschitzconstrained unsupervised skill discovery Year: (2022)
Ref_id:b51 Title: Scalable unsupervised rl with metric-aware abstraction Year: (2023)
Ref_id:b52 Title: Curiosity-driven exploration by self-supervised prediction Year: (2017)
Ref_id:b53 Title: Self-supervised exploration via disagreement Year: (2019)
Ref_id:b54 Title: Advantage-weighted regression: Simple and scalable off-policy reinforcement learning Year: (2019)
Ref_id:b55 Title: Learning a diffusion model policy from rewards via q-score matching Year: (2023)
Ref_id:b56 Title: Ride: Rewarding impact-driven exploration for procedurally-generated environments Year: (2020)
Ref_id:b57 Title: Efficient off-policy meta-reinforcement learning via probabilistic context variables Year: (2019)
Ref_id:b58 Title: Diffusion policy policy optimization Year: (2024)
Ref_id:b59 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b60 Title: State entropy maximization with random encoders for efficient exploration Year: (2021)
Ref_id:b61 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b62 Title: Denoising diffusion implicit models Year: ()
Ref_id:b63 Title: Score-based generative modeling through stochastic differential equations Year: ()
Ref_id:b64 Title: Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite Year: (2018)
Ref_id:b65 Title: Zero-shot whole-body humanoid control via behavioral foundation models Year: (2025)
Ref_id:b66 Title: Learning one representation to optimize all rewards Year: (2021)
Ref_id:b67 Title: Prioritized generative replay Year: (2024)
Ref_id:b68 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2023)
Ref_id:b69 Title: One-step diffusion policy: Fast visuomotor policies via diffusion distillation Year: (2024)
Ref_id:b70 Title: A problem in geometric probability Year: (1962)
Ref_id:b71 Title: Leveraging skills from unlabeled prior data for efficient online exploration Year: (2024)
Ref_id:b72 Title: Policy representation via diffusion probability model for reinforcement learning Year: (2023)
Ref_id:b73 Title: Behavior contrastive learning for unsupervised skill discovery Year: (2023)
Ref_id:b74 Title: Towards safe reinforcement learning via constraining conditional value-at-risk Year: (2022)
Ref_id:b75 Title: Peac: Unsupervised pre-training for cross-embodiment reinforcement learning Year: (2024)
Ref_id:b76 Title: Automatic intrinsic reward shaping for exploration in deep reinforcement learning Year: (2023)
Ref_id:b77 Title: Towards efficient unsupervised reinforcement learning with multi-choice dynamics model Year: (2022)
Ref_id:b78 Title: A mixture of surprises for unsupervised reinforcement learning Year: (2022)
Ref_id:b79 Title: Diffusion models for reinforcement learning: A survey Year: (2023)
Ref_id:b80 Title: A very good method for bayes-adaptive deep rl via metalearning Year: (2019)
Ref_id:b81 Title: Varibad: Variational bayes-adaptive deep rl via meta-learning Year: (2021)
