Title: Hyperspherical Normalization for Scalable Deep Reinforcement Learning
Abstract: Scaling up the model size and computation has brought consistent performance improvements in supervised learning. However, this lesson often fails to apply to reinforcement learning (RL) because training the model on non-stationary data easily leads to overfitting and unstable optimization. In response, we introduce SimbaV2, a novel RL architecture designed to stabilize optimization by (i) constraining the growth of weight and feature norm by hyperspherical normalization; and (ii) using a distributional value estimation with reward scaling to maintain stable gradients under varying reward magnitudes. Using the soft actor-critic as a base algorithm, SimbaV2 scales up effectively with larger models and greater compute, achieving state-of-the-art performance on 57 continuous control tasks across 4 domains. The code is available at dojeon-ai.github.io/SimbaV2.

Section: Introduction
Over the past decade, a scaling law has emerged as the cornerstone of supervised learning (SL), suggesting that increasing model size, compute, and data consistently improve performance (Kaplan et al., 2020;Dehghani et al., 2023). This paradigm has driven significant breakthroughs, from large language models (Gemini et al., 2023;Achiam et al., 2023) to diffusion models (Ramesh et al., 2021;Rombach et al., 2022), where bigger models reliably translate to better performance.
In contrast, scaling laws often fail to apply in reinforcement learning (RL) (Song et al., 2019;Li et al., 2023). Unlike SL's static data distributions, RL agents must contend with continuously evolving data distributions and shifting objectives throughout their training process (Sutton & Barto, 2018). This fundamental non-stationarity creates a scaling paradox: increasing model capacity or computational re-Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). sources frequently leads to overfitting to earlier experiences and reduced adaptability to new tasks (Lyle et al., 2022;Dohare et al., 2023).
One of the root causes of RL's scaling challenges lies in uncontrolled norm growth during training, which destabilizes optimization in multiple ways. Feature norms grow uncontrollably due to the implicit bias of TD loss (Kumar et al., 2022), where dominant dimensions cause overfitting and loss of plasticity (Lyle et al., 2022;Ma et al., 2023). Parameter norms grow unbounded, reducing effective learning rates (gradient-to-parameter ratio) and making weight updates increasingly difficult (Dohare et al., 2023;Lyle et al., 2024). Gradient norms fluctuate due to varying reward scales and outliers, further disrupting optimization. These instabilities are compounded with an increased model size or update frequency, making RL harder to scale than SL.
Previous work has addressed these norm instabilities through separate, isolated approaches. Normalization layers such as ℓ 2 -normalization (Bjorck et al., 2021;Hussing et al., 2024), layer normalization (Lei Ba et al., 2016;Lyle et al., 2023), and RL-specific variants (Bhatt et al., 2024;Lee et al., 2024c) control the growth of feature norm. Weight decay (Farebrother et al., 2018) manages the growth of param- eter norm. Reward scaling and cross-entropy loss (Schaul et al., 2021;Farebrother et al., 2024) were adopted to control gradient norm fluctuations. However, these techniques are applied individually without a unified framework, making coordination and scaling difficult. Periodic weight reinitialization (Nikishin et al., 2022;D'Oro et al., 2023;Schwarzer et al., 2023) offers an alternative by completely retraining networks periodically. While effective, this approach requires additional training time and causes sharp performance drops, making it impractical for safety-critical applications.
In response, we present SimbaV2, a novel RL architecture that addresses these challenges by simultaneously stabilizing weight, feature, and gradient norms within a unified framework. Building on the Simba architecture (Lee et al., 2024c), which uses pre-layernorm residual blocks (Xiong et al., 2020) and weight decay (Krogh & Hertz, 1991), Sim-baV2 introduces three key modifications:
• Hyperspherical Feature Normalization: We replace all layer normalization with hyperspherical normalization (ℓ 2 -normalization).
• Hyperspherical Weight Normalization: We remove weight decay and instead project weights onto the unitnorm hypersphere after each gradient update (Loshchilov et al., 2024). Combined with hyperspherical feature normalization, this ensures consistent effective learning rates across layers and eliminates the need for weight regularization tuning.
• Distributional Value Estimation with Reward Scaling: To address unstable gradient norms caused by varying reward scales and outliers, we integrate a distributional critic (Bellemare et al., 2017) and apply reward scaling to maintain unit variance of the target throughout training.
Using Soft Actor-Critic (Haarnoja et al., 2018) as our base algorithm, SimbaV2 effectively stabilizes all three types of norms while maintaining consistent effective learning rates throughout training (Section 5.2 and Figure 4). We evaluated SimbaV2 on four standard online RL benchmarks: MuJoCo (Todorov et al., 2012), DMC Suite (Tassa et al., 2018), MyoSuite (Caggiano et al., 2022), and Humanoid-Bench (Sferrazza et al., 2024); as well as the D4RL MuJoCo benchmark (Fu et al., 2020) for offline RL. As shown in Figures 1 and 2, SimbaV2 achieves state-of-the-art performance without requiring algorithmic modifications or hyperparameter tuning, and scales effectively with increased model size and computation without using periodic reinitialization.
this section cite: ['b43', 'b15', 'b32', 'b1', 'b76', 'b61', 'b79', 'b67', 'b17', 'b53', 'b67', 'b17', 'b69', 'b6', 'b40', 'b60', 'b68', 'b5', 'b21', 'b71', 'b22', 'b18', 'b19', 'b73', 'b92', 'b51', 'b66', 'b35', 'b82', 'b10', 'b74', 'b24']

Section: Related Work

this section cite: []

Section: Regularization in Deep Reinforcement Learning
Deep RL is particularly susceptible to overfitting due to its inherently non-stationary optimization process (Song et al., 2019). To address overfitting, researchers have adapted regularization techniques from SL, including weight decay (Farebrother et al., 2018), dropout (Hiraoka et al., 2021), various normalization layers (Gogianu et al., 2021;Bjorck et al., 2021;Lyle et al., 2023;Gallici et al., 2024;Bhatt et al., 2024;Lee et al., 2024c;Elsayed et al., 2024;Palenicek et al., 2025), and mixture of expert (Obando-Ceron et al., 2024;Willi et al., 2024). However, these methods often prove insufficient when scaling RL models, as larger computational resources and increased model sizes can easily exacerbate overfitting (Li et al., 2023;Nauman et al., 2024a).
To further scale computations and model sizes in RL, recent studies have explored periodic weight reinitialization strategies to rejuvenate learning and escape local minima (D'Oro et al., 2023;Nauman et al., 2024b). These strategies include reinitializing weights to their initial distributions (Nikishin et al., 2022), interpolating between random and current weights (Xu et al., 2023;Schwarzer et al., 2023), utilizing momentum networks (Lee et al., 2024b), and selectively reinitializing dormant weights (Sokar et al., 2023). While promising, reinitialization has a notable limitation: it can lead to the loss of useful information and incur significant computational overhead as model size increases.
To address these limitations, we introduce SimbaV2, an architecture that explicitly constrains parameter, feature, and gradient norms throughout training. By constraining norms through hyperspherical normalization, SimbaV2 stabilizes an optimization process and eliminates the need for weight decay or periodic weight reinitialization.
this section cite: ['b76', 'b21', 'b39', 'b33', 'b6', 'b68', 'b30', 'b5', 'b20', 'b91', 'b91', 'b61', 'b19', 'b18', 'b93', 'b73', 'b75']

Section: Hyperspherical Representations in Deep Learning
Hyperspherical representations are widely used in deep learning across image classification (Salimans & Kingma, 2016;Liu et al., 2017b), face recognition (Wang et al., 2017;Liu et al., 2017a), variational autoencoders (Xu & Durrett, 2018), and contrastive learning (Chen et al., 2020). Using spherical embeddings is known to enhance feature separability (Wang & Isola, 2020), improving performance in tasks requiring precise discrimination. Recently, researchers have applied the hyperspherical normalization to intermediate features and weights to stabilize training in large-scale models such as diffusion models (Karras et al., 2024) and transformers (Loshchilov et al., 2024).
In this work, we apply hyperspherical normalization to RL. Unlike previous studies that focus on training the network on stationary data distributions with discrete inputs and outputs, we demonstrate their effectiveness on non-stationary data distributions with continuous inputs and outputs.
this section cite: ['b86', 'b94', 'b13', 'b87', 'b44', 'b66']

Section: Preliminaries
As background, we briefly explain the Soft Actor-Critic (SAC) algorithm (Haarnoja et al., 2018) and the Simba architecture (Lee et al., 2024c).
this section cite: ['b35']

Section: Soft Actor Critic
SAC is a prominent off-policy algorithm for continuous control. It aims to maximize both expected cumulative reward and policy entropy, where τ = (o, a, r, o ′ ) represents a transition tuple. SAC comprises a stochastic policy π θ (a|o), a Q-function Q ϕ (o, a), and an entropy coefficient α that balances reward and entropy.
The policy network is optimized to maximize the expected return while encouraging entropy, which is formalized as:
L π = E ā∼π θ [α log π θ (ā|o) -Q ϕ (o, ā)] .(1)
The Q-function Q ϕ (o, a) is trained to minimize the Bellman residual loss:
L Q = (Q ϕ (o, a) -r + γQ φ(o ′ , a ′ ) -α log π θ (a ′ |o ′ ) ) 2 ,(2)
where a ′ ∼ π θ (•|o ′ ), γ ∈ [0, 1) is the discount factor, and Q φ represents the target Q-network updated via an exponential moving average of ϕ.
this section cite: []

Section: Simba Architecture
Simba (Lee et al., 2024c) is an RL architecture with normalization layers composed of the following stages:
Input Embedding. Given an input observation o t ∈ R |O| , Simba applies Running Statistics Normalization (RSNorm) to normalize each dimension to zero mean and unit variance.
At each timestep t, the running mean µ t ∈ R |O| and variance σ 2 ∈ R |O| are updated recursively as:
µ t = µ t-1 + 1 t δ t , σ 2 t = σ 2 t-1 + 1 t (δ 2 t -σ 2 t-1 ) (3
)
where δ t = o t -µ t-1 .
Given running statistics, the observation is normalized as:
ōt = RSNorm(o t ) = o t -µ t σ 2 t + ϵ .(4)
Then, the normalized observation, ōt , is embedded with a linear layer W 0 h ∈ R |O|×d h defined as:
h 0 t = W 0 h ōt .(5)
Latent Encoding. Next, the embedding h 0 t is encoded by a stack of L residual blocks with pre-layer normalization. For l ∈ {1, . . . , L}, each of the l-th block is defined as:
h l t = h l-1 t + MLP(LayerNorm(h l-1 t ))(6)
After the final block, the output is normalized again to obtain the latent feature:
z t = LayerNorm(h L t ).(7)
Output Prediction: Finally, to predict the policy or Qvalue, a linear layer W o ∈ R d h ×do maps z t to:
p t = W o z t .(8)
this section cite: []

Section: SimbaV2
SimbaV2 builds on Simba by adding constraints on weights, features, and gradients to enhance training stability, particularly when scaling to larger models and more computation.
The modifications include:
×N Scaler Scaler ReLU Scaler Linear Linear LERP Linear Linear L2 Norm Input L2 Norm Linear L2 Norm Output L2 Norm RSNorm Shift LERP 𝛂 𝟏 -𝛂 𝑥 !!" 𝑥 #$%&'()* Shift & Norm
this section cite: []

Section: L2 Normalization
Add a New Axis
this section cite: []

Section: New Axis

this section cite: []

Section: L2 Normalization
𝑥 +#,- Shift Upwards LERP & Norm +c Figure 3. SimbaV2 architecture.
The input observation is first normalized using running statistics, then shifted along a new axis with a constant cshift to preserve magnitude information before being projected onto the unit hypersphere. The projected observation is passed through a linear layer, followed by a series of non-linear blocks and refined with LERP, serving as a residual connection. A final linear layer predicts the policy or value function.
• LayerNorm → ℓ 2 -Norm: Layer normalization is replaced with ℓ 2 -normalization, constraining intermediate features to have unit norm.
• Linear → Linear + Scaler: Standard linear layer is decoupled into a linear layer with weights constrained to a unit norm hypersphere, without a bias, and a learnable scaling vector that performs element-wise scaling.
• Residual Connection → LERP: Residual connection is replaced with a learnable linear interpolation (LERP), which combines raw and transformed features via a learnable interpolation vector.
• Weight Decay → Weight Projection: Weight decay is replaced with direct weight projection onto the unit hypersphere after each gradient update.
• MSE Loss → KL-divergence Loss: MSE-based Bellman loss is replaced with KL-divergence loss, using a categorical critic (Bellemare et al., 2017).
• No Reward Scaling → Reward Scaling: Rewards are normalized with running statistics to stabilize the scale of both actor loss (Equation.1) and critic loss (Equation .2).
In the following subsections, we describe these modifications in detail.
this section cite: []

Section: Input Embedding
Following Simba, SimbaV2 first standardize the raw observations o t ∈ R |O| using RSNorm, yielding ōt . To further stabilize training, we map ōt onto the unit hypersphere before applying a linear layer.
Shift + ℓ 2 -Norm. Direct ℓ 2 -normalization can discard magnitude information (e.g., ōt = [1, 0] and [2, 0] both map to [1,0]). To retain magnitude information, we embed ōt into an (|O| + 1)-dimensional vector by concatenating a positive constant c shift > 0, then apply ℓ 2 -normalization:
o t = ℓ 2 -Norm( ōt ; c shift ).(9)
As illustrated in Figure 3, this additional coordinate encodes the original norm of ōt , preserving magnitude information.
Linear + Scaler. We then embed õt using a linear layer W 0 h ∈ R (|O|+1)×d h and a scaling vector s 0 h ∈ R d h as:
h 0 t = ℓ 2 -Norm(s 0 h ⊙ (W 0 h Norm(õ t )).(10)
where the ℓ 2 -normalization projects back to the hypersphere.
this section cite: []

Section: Feature Encoding
Starting from the initial hyperspherical embedding h 0 t , we apply L consecutive blocks of non-linear transformations. Each l-th block transforms h l t into h l+1 t as follows:
MLP + ℓ 2 -Norm. Each block uses an inverted bottleneck MLP (Vaswani, 2017) followed by ℓ 2 -normalization to project the output back onto the unit hypersphere.
hl t = ℓ 2 -Norm(W l h,2 ReLU (W l h,1 h l t ) ⊙ s l h ). (11
)
where W l h,1 ∈ R 4d h ×d h and W l h,2 ∈ R d h ×4d h are weight matrices, and s l h ∈ R 4d h is a learnable scaling vector. LERP + ℓ 2 -Norm. We then linearly interpolate between the original input h l t and its non-linearly transformed output hl t , followed by another ℓ 2 -normalization:
h l+1 t = ℓ 2 -Norm((1 -α l ) ⊙ h l t + α l ⊙ hl t ).(12)
where 1 ∈ R d h and α l ∈ R d h are one vector and a learnable interpolation vector, respectively.
LERP acts analogous to a learnable residual connection but can also be viewed as a first-order approximation of a Riemannian retraction on the hypersphere (Absil et al., 2008). Please refer to Appendix A.1 for further discussion.
this section cite: ['b84', 'b0']

Section: Output Prediction
We use a linear layer to parameterize both the policy distribution and Q-value. Because Simba's single Q-value estimate with an MSE-based Bellman loss is susceptible to outliers, we adopt a categorical critic with KL-divergence loss (Bellemare et al., 2017), which provides smoother gradients and more stable optimization (Imani & White, 2018).
this section cite: ['b42']

Section: Distributional Critic.
We represent the Q-value as a categorical distribution over a discrete set of returns:
δ i = G min + (i -1) G max -G min n atom -1 i = 1, ..., n atom ,(13)
where G min and G max denote the minimum and maximum possible returns, and n atom is the number of discrete atoms.
Given the encoded representation h L t , we compute unnormalized logits z t ∈ R |A|×natom for all actions as follows:
z t = W o,2 ( W o,1 h L t ⊙ s o ),(14)
where
W o,1 ∈ R d h ×d h , W o,2 ∈ R |A|×natom×d h , and s o ∈ R d h are trainable parameters.
For each action a ∈ A, the categorical probability is represented by applying the softmax function to z t,a ∈ R natom : p t,a = softmax z t,a .
The resulting Q-value is the expected return under p t,a :
Q(o t , a) = natom i=1 δ i p t,a,i .(16)
Reward Bounding and Scaling. To use a categorical critic, we first bound the target returns within [G min , G max ] and then scale the reward to maintain unit variance, ensuring stable gradients for both the actor and the critic. Unlike previous work (Schaul et al., 2021), which scaled the critic loss, we scale the reward itself, affecting both components simultaneously. Moreover, unlike observation normalization, we do not center the reward, as shifting the reward can alter the optimal policy in episodic tasks (Naik et al., 2024).
Given a reward r t at time t and a discounted factor γ, we track a running discounted return:
G t ← γG t-1 + r t (17
)
where G t is re-initialized to 0 at the start of each episode.
Then, we track the running variance of G t , denoted as σ 2 t,G
and maintain a running maximum:
G t,max ← max(G t,max , G t ).(18)
We then scale the reward as follows:
rt ← r t max( σ 2 t,G + ϵ, G t,max /G max ) .(19)
This formula stabilizes gradients for both high-variance and low-variance returns, while thresholding with G t,max /G max ensures target returns remain within [G min , G max ].
this section cite: ['b71']

Section: Initialization and Update
In this subsection, we outline how weight matrix W , scaler s, and interpolation vector α are initialized and updated.
Weight. All weight vectors are initialized orthogonally and then projected onto the unit hypersphere which forms an orthonormal basis. At each gradient step, we re-project them onto the unit sphere to maintain unit norm.
Formally, let W be the weight matrix before the update, and let L denote the loss function. The update rule is defined as:
W ← ℓ 2 -Norm(W -η ∂L ∂W )(20)
where η > 0 is a learning rate and ℓ 2 -Norm is the ℓ 2normalization operator along the embedding axis.
Scaler. Following Loshchilov et al. (2024), we decouple the initialization scale of s from its learning dynamics by using two scalars, s init and s scale . Although s is initialized to s scale , it behaves as if it was initialized to s init during the forward pass by:
s ← s scale ⊙ (s init ⊘ s scale )(21)
where ⊙ and ⊘ are element-wise product and division, respectively. This formulation lets s scale control the learning rate of s independently from the global learning rate η.
When both the feature vector h ∈ R d h and the randomly orthonormal initialized weight matrix W ∈ R d h ×d h lie on the unit hypersphere, each component of W h ∈ R
d h can be approximated by cos(θ) with E θ [cos 2 (θ)] = 1/2. Therefore, we set s init = s scale = ( 2/d h ) 1 to maintain unit norm after scaling at initialization. A detailed derivation is in Appendix A.2. Interpolation vector. Analogous to the scaler, the interpolation vector, α, also has α init and α scale . Following Loshchilov et al. (2024), we initialize α init = 1/(L + 1) and α scale = 1/ √ d h , to preserve residual feature and gradually integrate non-linear features. 0 0.2 0.4 0.6 0.8 1.0 0.00 0.25 0.50 0.75 1.00 DMC-Hard (a) Average Return 0 0.2 0.4 0.6 0.8 1.0 1 60 120 180 240 (b) Feature Norm 0 0.2 0.4 0.6 0.8 1.0 0 20 40 60 80 (c) Parameter Norm 0 0.2 0.4 0.6 0.8 1.0 10 4 0.5 1.0 1.5 2.0 (d) Gradient Norm 0 0.2 0.4 0.6 0.8 1.0 0.01 0.25 0.50 0.75 1.00 ×10 2 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0.00 0.25 0.50 0.75 1.00 HBench-Hard 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 1 25 50 75 100 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 30 60 90 120 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 10 4 2 4 6 8 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0.01 0.25 0.50 0.75 1.00 ×10 1 0.01 15.00 30.00 45.00 60.00 ×10 2 (e) Effective LR 0.01 2.50 5.00 7.50 10.00 ×10 1 SimbaV2 Simba SimbaV2 Encoder SimbaV2 Predictor Simba Encoder Simba Predictor Figure 4. SimbaV2 vs. Simba Training Dynamics. We track 4 metrics during training to understand the learning dynamics of SimbaV2: (a) Average normalized return across tasks. (b) Weighted sum of ℓ2-norms of all intermediate features in critic. (c) Weighted sum of ℓ2-norms of all critic parameters (d) Weighted sum of ℓ2-norms of all gradients in critic (e) Effective learning rate (ELR) of the critic. On both environments, SimbaV2 maintains stable norms and ELR, while Simba exhibits divergent fluctuations.
this section cite: ['b66']

Section: Experiments
We now present a series of experiments designed to evaluate SimbaV2. Our investigation centers on four main setups:
• Optinmization Analysis (Section 5.2). Investigate whether SimbaV2 stabilizes the optimization process.
• Scaling Analysis (Section 5.3). Investigate whether Sim-baV2 allows scaling model capacity and computation.
• Comparisons (Sections 5.4). Compare SimbaV2 against state-of-the-art RL algorithms.
• Design Study (Section 5.5.) Conducts ablation studies on individual architectural components of SimbaV2.
this section cite: []

Section: Experimental Setup
Environment. A total of 57 continuous-control tasks are considered across 4 domains: MuJoCo (Todorov et al., 2012), DMC Suite (Tassa et al., 2018), MyoSuite (Caggiano et al., 2022), and HumanoidBench (Sferrazza et al., 2024). Also, two challenging subsets are defined for an empirical analysis: DMC-Hard (7 tasks involving dog and humanoid embodiments) and HBench-Hard (5 tasks: run, balance-simple, sit-hard, stair, walk).
Baselines. Comparisons include a broad range of deep RL algorithms, PPO (Schulman et al., 2017), SAC (Haarnoja et al., 2018), TD3 (Fujimoto et al., 2018) TD3+OFE (Ota et al., 2020), TQC (Kuznetsov et al., 2020), DreamerV3 (Hafner et al., 2023), TD7 (Fujimoto et al., 2023), TD-MPC2 (Hansen et al., 2023), Cross-Q (Bhatt et al., 2024), BRO (Nauman et al., 2024b), MAD-TD (Voelcker et al., 2024), MR.Q (Fujimoto et al., 2025), and Simba (Lee et al., 2024c). Whenever available, we report the results from the original paper; otherwise, we run the authors' official code. In addition, to further compare performance before and after scaling, we evaluate BRO, Simba, and SimbaV2 under both low UTD ratios (≤ 2) and high UTD ratios (≤ 8). Additional details are described in Appendix E.
this section cite: ['b82', 'b10', 'b74', 'b72', 'b35', 'b36', 'b28', 'b37', 'b5', 'b85', 'b29']

Section: Metrics.
To aggregate performance across diverse domains, each environment's return is normalized to a near [0, 1) range. Specifically, MuJoCo performance is normalized by TD3 (Fujimoto et al., 2018); DMC returns are divided by 1000; MyoSuite scores use success rates; and Humanoid-Bench scores are normalized by their success score.
Training. If possible, we tried to closely follow Simba's training configuration aiming to provide an apples-to-apples comparison. Unless otherwise specified, the actor and critic have hidden dimensions of 128 and 512, respectively (approximately 5M parameters). The model is trained for 1M environment steps, using a UTD ratio 2. We used an Adam (Kingma & Ba, 2014) optimizer without weight decay and set the batch size to 256. The learning rate is linearly decayed from 1 × 10 -4 to 3 × 10 -5 . Full hyperparameter configurations are provided in Appendix C.
this section cite: ['b46']

Section: Optimization Analysis
To understand the optimizatiom dynamics of SimbaV2, we measure the feature norm, weight norm, gradient norm, and the effective learning rate (ELR), defined as the ratio of the gradient norm to the weight norm (Kodryan et al., 2022;Lyle et al., 2024) (See Appendix G for details). We weighted average each metric across layers where weights correspond to each layer's fraction of total parameters. Additionally, we divide the layers into encoder layers (all layers before the output prediction) and predictor layers (those after) to analyze their respective dynamics. 128 256 512 1024 Hidden Dim of Critic (d h ) 0.25 0.38 0.50 0.62 0.75 Total Reward (K) DMC-Hard 128 256 512 1024 Hidden Dim of Critic (d h ) 0.30 0.45 0.60 0.75 0.90 Sucess-Norm Score HBench-Hard SimbaV2 Simba Figure 5. Width Scaling. We scale the number of model parameters by increasing the width of the critic network. On DMC-Hard, both Simba and SimbaV2 benefit from increased model size. On HBench-Hard, however, Simba plateaus at larger model sizes, whereas SimbaV2 continues to improve. large, often divergent fluctuations in feature, weight, and gradient norms between the encoder and predictor. Consequently, Figure 4.(e) shows that the encoder's ELR trending upward while the predictor's ELR declines.
In contrast, SimbaV2 enforces tighter constraints, stabilizing norms and ELRs throughout training. Although certain parameters (e.g., scalers or interpolation vectors) can exceed the unit norm, the majority of parameters remain on the hypersphere, resulting in more robust optimization. A standalone visualization of SimbaV2 is in Appendix G.
this section cite: ['b48', 'b69']

Section: Scaling Analysis
For this experiment, we investigate whether SimbaV2's stable training dynamics enable better scaling performance as model parameters or computational resources increase, while reducing overfitting compared to existing methods.
this section cite: []

Section: Experimental Setup.
We conduct two types of scaling experiments. For parameter scaling, we focus on scaling the critic network, as prior studies indicate that scaling the actor provides limited benefits (Nauman et al., 2024b;Lee et al., 2024c). We test two scaling approaches: width scaling by varying the critic's hidden dimension across {128, 256, 512, 1024}, increasing parameters from 0.3M to 17.8M ; and depth scaling by varying the number of critic blocks L across {1, 2, 4, 8}, growing parameters from 2.2M to 17.8M .
For compute scaling experiments, we vary the update-todata (UTD) ratio across {1, 2, 4, 8}. We compare results both with and without periodic weight reinitialization, since prior work suggests that compute scaling requires periodic reinitialization to avoid overfitting (D'Oro et al., 2022).
Following Nauman et al. (2024b), we apply reinitialization every 500,000 update steps when used.
Parameter Scaling. However, on the more challenging HBench-Hard benchmark, while both methods achieve comparable performance at the smallest scale (d h = 128), their scaling behavior diverges significantly. Simba plateaus at larger scales with peak performance at d h = 1024, while SimbaV2 continues to improve with increased width. This demonstrates that SimbaV2's stabilized training dynamics effectively leverages larger model capacity.
Figure 6 presents depth scaling results. In HBench-Hard, SimbaV2 shows consistent performance improvements as the depth of the critic L increases, successfully solving the five complex tasks in L = 8. On DMC-Hard, SimbaV2's performance also improves with depth but begins saturating around L = 4, likely due to task complexity limitations rather than architectural constraints. In contrast, Simba's performance either plateaus around L = 2 or slightly decreases after initial improvement. This clear difference demonstrates SimbaV2's superior depth scalability, which we attribute to its effective regularization mechanisms that enable stable training of deeper networks.
Compute Scaling. We next explore compute scaling through increased UTD ratios, a key factor in improving sample efficiency in deep RL (Li et al., 2023). While higher UTD ratios can enhance sample efficiency, they also increase the risk of overfitting. Previous approaches address this through ensembling (Chen et al., 2021b), periodic reinitialization (Lee et al., 2024a;D'Oro et al., 2023;Nauman et al., 2024b), or both (Kim et al., 2023). We investigate whether SimbaV2's stable dynamics enable effective scaling without these additional mechanisms.
Figure 7 shows the effect of varying the UTD ratio on DMC-Hard (left) and HBench-Hard (right), comparing Simba and SimbaV2 with and without reinitialization (solid lines: no reinitialization; dashed lines: reinitialization). In Simba, performance plateaus at a UTD ratio of 2 on DMC-Hard and 1 on HBench-Hard. When combined with reinitialization, but further improves with reinitialization, consistent with Table 1. Online RL. Average final performance after 1M environment steps, where ± captures a 95% confidence interval (CI) computed over all raw benchmark samples. For algorithms with only average scores for each task available, we approximate the CI using these averages ( †). Note that this estimation may be inaccurate. The highest performance is highlighted. Any performance that is not statistically worse than the highest performance (according to Welch's t-test with significance level 0.05) is highlighted.
this section cite: ['b18', 'b61', 'b19', 'b45']

Section: Mujoco (5) DMC-Easy (21) DMC-Hard (7)
MyoSuite ( 10) HBench ( 14) All (57) Method TD3.Norm Return (1k) Return (1k) Success Rate Success.Norm -(a) Low UTD (≤ 2) PPO (Schulman et al., 2017) 0.447 ± 0.270 † 0.327 ± 0.128 † 0.033 ± 0.030 † ---SAC (Haarnoja et al., 2018) 1.092 ± 0.081 0.762 ± 0.094 † 0.136 ± 0.04 0.607 ± 0.088 0.279 ± 0.050 0.554 ± 0.057 TD3 (Fujimoto et al., 2018) 1.000 ± 0.000 † -----TD3+OFE ( Ota et al., 2020)1.322 ± 0.263 † -----TQC (Kuznetsov et al., 2020) 1.137 ± 0.125 † -----TD7 (Fujimoto et al., 2023) 1.570
± 0.030 0.689 ± 0.134 † 0.182 ± 0.137 † 0.356 ± 0.126 0.289 ± 0.083 0.617 ± 0.358 † TD-MPC2 (Hansen et al., 2023) 1.040 ± 0.115 0.889 ± 0.064 † 0.465 ± 0.139 † 0.650 ± 0.148 0.710 ± 0.149 0.749 ± 0.168 † CrossQ (Bhatt et al., 2024) 1.475 ± 0.141 -----MR.Q (Fujimoto et al., 2025) 1.448 ± 0.156 0.868 ± 0.026 0.723 ± 0.061 ---BRO (Nauman et al., 2024b) 1.101 ± 0.182 0.861 ± 0.036 0.693 ± 0.066 0.714 ± 0.076 0.468 ± 0.107 0.731 ± 0.039 Simba (Lee et al., 2024c) 1.147 ± 0.077 0.864 ± 0.024 0.706 ± 0.05 0.743 ± 0.079 0.606 ± 0.073 0.780 ± 0.028 SimbaV2 (ours) 1.617 ± 0.103 0.874 ± 0.025 0.729 ± 0.065 0.847 ± 0.066 0.776 ± 0.064 0.892 ± 0.032 (b) High UTD (≥ 8) REDQ (Chen et al., 2021b) 1.160 ± 0.071 -----DroQ (Hiraoka et al., 2021) 1.134 ± 0.070 -----DreamerV3 (Hafner et al., 2023) 0.760 ± 0.095 0.714 ± 0.124 † 0.009 ± 0.006 † 0.482 ± 0.166 0.022 ± 0.023 0.397 ± 0.289 † MAD-TD (Voelcker et al., 2024) --0.708 ± 0.065 ---BRO (Nauman et al., 2024b) 1.150 ± 0.202 0.871 ± 0.034 0.767 ± 0.059 0.814 ± 0.066 0.619 ± 0.117 0.807 ± 0.037 Simba (Lee et al., 2024c) 1.175 ± 0.136 0.866 ± 0.036 0.720 ± 0.087 0.834 ± 0.098 0.657 ± 0.099 0.818 ± 0.043 SimbaV2 (ours) 1.598 ± 0.176 0.876 ± 0.035 0.769 ± 0.089 0.866 ± 0.090 0.822 ± 0.099 0.911 ± 0.044 1 2 4 8 UTD Ratio 0.60 0.65 0.70 0.75 0.80 Total Reward (K) DMC-Hard 1 2 4 8 UTD Ratio 0.40 0.55 0.70 0.85 1.00 Sucess-Norm Score HBench-Hard SimbaV2 SimbaV2+Reset Simba Simba+Reset Figure 7. Compute Scaling.. We scale compute by increasing the UTD ratio. We compare Simba and SimbaV2, both with and without periodic reset. Simba saturates at lower ratios without reset, but improves with reset. In contrast, SimbaV2 scales smoothly even without reset, where using reset slightly degrades its performance.
D' Oro et al. (2023). In contrast, SimbaV2 scales consistently as the UTD ratio increases, even without reinitialization. Notably, reinitialization slightly degrades SimbaV2's performance, as it disrupts training and adds time to recover.
To verify the importance of hyperspherical weight and feature normalization for UTD scaling, we test a variant in Appendix H.1 that includes distributional critics and reward scaling on Simba. This variant fails to scale at higher UTD ratios, confirming the critical role of hyperspherical normalization for effective scaling.
this section cite: ['b72', 'b35', 'b28']

Section: Online RL
Having observed SimbaV2's scalability, we now compare it against standard model-free and model-based RL.
Table 1.(a) presents results at a UTD ratio below 2. Sim-baV2 with UTD=2 attains an average normalized score of 0.892, exceeding the previous best of 0.780. Only except for DMC-Easy suite, SimbaV2 outperforms leading model-free (CrossQ (Bhatt et al., 2024), BRO (Nauman et al., 2024b), Simba (Lee et al., 2024c)) and model-based (TD-MPC2 (Hansen et al., 2023), MR.Q (Fujimoto et al., 2025)) baselines, demonstrating superior sample efficiency. Table 1.(b) evaluates higher UTD settings. Increasing Sim-baV2's UTD from
2 to 8 further elevates its average score from 0.892 to 0.911. SimbaV2 also surpasses BRO with UTD=10, which utilizes periodic reinitialization to avoid overfitting at high update rates. These consistent gains at larger UTD ratios underscore the efficacy of hyperspherical normalization in stabilizing training.
For offline RL, we simply add a behavioral cloning loss during training with using identical configurations to the online RL. Despite minimal changes, SimbaV2 performs competitively with existing baselines (Appendix D).
this section cite: []

Section: Design Study
Table 2 presents the results from ablation studies isolating the contributions of various architectural choices.
this section cite: []

Section: Input Projection.
Projecting observations onto a hypersphere before passing them through the linear layer is crucial for performance (Table 2.(a)), where omitting this step leads to a significant performance drop. Equally important design is preserving the original magnitude during projection (Table 2.(b)). We also explore an alternative "resize" projection, where inputs are first divided by c shift √ d h before being projected onto an (n + 1)-dimensional hypersphere. The resize projection yields comparable performance as it can also retain magnitude information (Table 2
this section cite: []

Section: .(d)).
Output Projection. Incorporating a distributional critic and reward scaling improves performance, especially in environments with high reward variance like MuJoCo (Table 2.(e)-(f)). Bounding target returns proves essential for easier tasks (Table 2.(g)), such as cartpole in the DMC-Easy suite (Table 25). Without bounding, consistent high returns can diminish return variance, and scaling returns push target values beyond the range of the distributional critic, leading to collapse in the TD loss.
this section cite: []

Section: Initialization & Update.
Gradually decaying the learning rate is critical. Without decay, the model may struggle to refine its predictions during later training stages, as SimbaV2 maintains an effective constant learning rate throughout training (Table 2.(i)). Tuning initial scaler values has minimal impact on performance where the architecture remains stable by these changes (Table 2.(j)-(m)).
this section cite: []

Section: Lessons and Opportunities
Lessons. Historically, RL research has relied on complex regularizations to address overfitting and scalability issues (Klein et al., 2024). Our findings suggest that suitably chosen constraints, exemplified by SimbaV2, can simplify these design complexities while retaining strong performance.
Opportunities. Future opportunities include deploying SimbaV2 in real-world robotics (Hwangbo et al., 2019), where sample efficiency is crucial, and extending it to model-based (Hansen et al., 2023) or visual RL (Kostrikov et al., 2020). Furthermore, with increasing interest in RL for training large language models (Ouyang et al., 2022;Guo et al., 2025), the potential benefits of using stricter normalization for large models remain an exciting open question.
this section cite: ['b47', 'b41', 'b37', 'b49', 'b34']

Section: Impact Statement
This paper presents work aimed at advancing the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. Ma, G., Li, L., Zhang, S., Liu, Z., Wang, Z., Chen, Y., Shen, L., Wang, X., and Tao, D. Revisiting plasticity in visual reinforcement learning: Data, modules and training stages. arXiv preprint arXiv:2310.07418, 2023. (Cited on page 1) Naik, A., Wan, Y., Tomar, M., and Sutton, R. S. Reward centering. arXiv preprint arXiv:2405.09999, 2024. (Cited on page 5) Nauman, M., Bortkiewicz, M., Ostaszewski, M., Miłoś, P., Trzciński, T., and Cygan, M. Overestimation, overfitting, and plasticity in actor-critic: the bitter lesson of reinforcement learning. arXiv preprint arXiv:2403.00514, 2024a. (Cited on page 2) Nauman, M., Ostaszewski, M., Jankowski, K.
, Miłoś, P., and Cygan, M. Bigger, regularized, optimistic: scaling for compute and sample-efficient continuous control. arXiv preprint arXiv:2405.16158, 2024b. (Cited on page 3, 6, 7, 8, 23) Nikishin, E., Schwarzer, M., D'Oro, P., Bacon, P.-L., and Courville, A. The primacy bias in deep reinforcement learning. Proc. the International Conference on Machine Learning (ICML), 2022. (Cited on page 2, 3) Obando-Ceron, J., Sokar, G., Willi, T., Lyle, C., Farebrother, J., Foerster, J., Dziugaite, G. K., Precup, D., and Castro, P. S. Mixtures of experts unlock parameter scaling for deep rl. arXiv preprint arXiv:2402.08609, 2024. (Cited on page 2) Ota, K., Oiki, T., Jha, D., Mariyama, T., and Nikovski, D. Can increasing input dimensionality improve deep reinforcement learning? In International conference on machine learning, pp. 7424-7433. PMLR, 2020. (Cited on page 6, 8, 22) Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022. (Cited on page 9) Palenicek, D., Vogt, F., and Peters, J. Scaling off-policy reinforcement learning with batch and weight normalization. arXiv preprint arXiv:2502.07523, 2025. (Cited on page 2) Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. In International conference on machine learning, pp. 8821-8831. Pmlr, 2021. (Cited on page 1) Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684-10695, 2022. (Cited on page 1) Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems, 29, 2016. (Cited on page 3)
During the feature encoding stage in SimbaV2, the input h and its non-linearly transformed output h are linearly interpolated using a learnable interpolation vector α ∈ R d h :
h ← ℓ 2 -Norm((1 -α) ⊙ h + α ⊙ h),(22)
followed by ℓ 2 -normalization.
Intuitively, this can be interpreted as a first-order (retraction-based) approximation of the Riemannian update formula on the hypersphere. This section provides a brief introduction to the differential geometry concepts that underpin the Riemannian optimization perspective of α. For brevity, we omit the mathematical definitions, derivations, and proofs here.
The comprehensive introduction to differential geometry and Riemannian optimization can be found in Spivak (1970), Do Carmo & Flaherty Francis (1992), and Boumal (2023).
Let S n-1 denote the n-dimensional hypersphere embedded in R n , i.e., S n-1 = {h ∈ R n | ∥h∥ 2 = 1}.
Manifold. A manifold M of dimension n is a space that can locally be approximated by a Euclidean space R n . The simplest examples of a manifold include the open ball U = {x ∈ R n | ∥x∥ 2 < r} for r ∈ R >0 , and the hypersphere S n-1 is also a manifold in R n .
Tangent Spaces. At each point x ∈ M, the tangent space T x M is an n-dimensional vector space that locally approximates M near x. Tangent vectors generalize the concept of directional derivatives. For the hypersphere S n-1 , the tangent space at a point p consists of all vectors orthogonal to p:
T p S n-1 = {h ∈ R n | ⟨p, h⟩ = 0}(23)
where ⟨•, •⟩ denotes the Euclidean inner product.
Riemannian Metrics and Manifolds. The tangent space T x M is not inherently equipped with an inner product. A Riemannian metric ρ provides a collection of inner products ρ x (•, •) : T x M × T x M → R on the tangent spaces, ρ := (ρ x ) x∈M , which locally define the geometry of M. A Riemannian manifold (M, ρ) is a smooth manifold M equipped with such a metric. This enables us to define geometric notions such as distance, angle, length, volume, and curvature of manifold. For a detailed explanation of geometrics on Riemannian manifolds, refer to Lee (2006).
Exponential Mapping and Retraction. Under some conditions (Do Carmo & Flaherty Francis, 1992), the exponential map exp x : T x M → M can be defined at a point x ∈ M. exp x (v) maps a tangent vector v ∈ T x M to a point on the manifold along the geodesic from x in the direction of v. Therefore, for small t ∈ R, exp x (tv) represents the shortest path on M starting at x with initial direction v. In Euclidean space (R n , I n ), the exponential map exp x (v) = x + v is simply defined as a straight path. In practice, for computational efficiency (e.g., the mappings do not have closed-form), we often approximate the exponential map exp x by a retraction (Absil et al., 2008
) R x : Definition A.1 (Retraction). A retraction R on a manifold M is a smooth map: R : x∈M T x M → M (x, v) → R x (v)
with the following properties: R x (0) = x and (dR x ) 0 = id where R x denotes the restriction of R to T x M, (dR x ) 0 denotes the differential of R x at 0, and id is the identity map.
Intuitively, a retraction R x (v) provides a first-order approximation of the exponential map exp x (v) (Boumal, 2023).
Figure 8 illustrates the difference between the exponential map and retraction on S 2 . For the hypersphere S n-1 , the retraction of a tangent vector ξ ∈ T h S n-1 onto S n-1 is given by (Absil et al., 2008): Riemannian Optimization. On Riemannian manifolds, gradient updates ideally follow the curved geodesics, rather than straight lines as in Euclidean space. To this end, Bonnabel (2013) introduce Riemannian SGD that generalizes SGD to Riemannian manifolds using exponential map:
R h (ξ) = ℓ 2 -Norm(h + ξ) = h + ξ ∥h + ξ∥ 2 (24
)
h ← exp h (-αg) (25
)
where α > 0 is the global learning rate and g ∈ T h M denotes the Riemannian gradient.
In our case, -( hh) can be viewed as the gradient g in the Euclidean space. Then, we project the gradient onto the tangent space T h S n-1 :
g proj = g -⟨g, h⟩h(26)
= -( h -h) -⟨-h + h, h⟩h(27)
= -h + ⟨ h, h⟩h(28)
Applying the retraction exp h (-αg proj ) ≈ R h (-αg proj ):
h ← ℓ 2 -Norm h + α( h -⟨ h, h⟩h) (29
) = ℓ 2 -Norm (1 -α⟨ h, h⟩)h + α h(30)
Thus, the LERP operation in SimbaV2 can be interpreted as a retraction-based approximation of the Riemannian update rule on the hypersphere, where the learning rate α is replaced by a learnable vector α and the inner product ⟨ h, h⟩ term is neglected. Also, Loshchilov et al. (2024) empirically show that neglecting the inner product term has no significant impact on performance.
this section cite: ['b77', 'b16', 'b8', 'b59', 'b16', 'b0', 'b8', 'b0', 'b7', 'b66']

Section: A.2. Scaler Initialization
In our algorithm, the scaler s ∈ R d h is a learnable vector that element-wise scales the output z of the linear layer:
z = s ⊙ W h ∈ R d h(31)
where W ∈ R d h ×n is the weight matrix of the linear layer, and h ∈ R n is the input vector. To ensure that z (approximately) maintains unit norm at initialization, we initialize s as s =
2 d h • 1.
The following section provides the derivation for this initialization.
We assume that each normalized embedding w l ∈ R n of W , and a random n-dimensional normalized vector h ∈ R n , are uniformly distributed on the n-dimensional hypersphere S n-1 . Furthermore, we assume that the vectors w l and h are mutually independent (Feller, 1991). We denote the angle between w l and h by θ l , such that cos θ
l = w l • h since ∥w l ∥ 2 = ∥h∥ 2 = 1.
Distribution of the Cosine of the Angle. For simplicity, assume that w l is fixed. Since h is uniformly distributed on the hypersphere, the distribution of the angle θ l depends on the solid angle (Weisstein, 2005) subtended by h with respect to w l . The surface area A n-2 of an (n -1)-dimensional hyperspherical cap (Li, 2010) leads to the probability density function f (θ l ) (Cai et al., 2013):
f (θ l ) = A n-1 S n-1 = 2π (n-1)/2 Γ( n-1 2 ) 2π n/2 Γ( n 2 ) sin n-2 (θ l ) = Γ( n 2 ) √ πΓ( n-1 2 ) sin n-2 (θ l )(32)
where θ l ∈ [0, π], Γ is the gamma function and S n-1 is the surface area of (Weisstein, 2002). Norm of Output Vector. Let z = s ⊙ W h ∈ R d h be the output of the linear layer. Each element of z and s, denoted by z l and s l , respectively, corresponds to the scaled cosine of the angle θ l between w l and h:
S n-1 = 2π n/2 Γ( n 2 )
z = s ⊙ W h =      s 1 (w 1 • h) s 2 (w 2 • h) . . . s d h (w d h • h)      =      s 1 cos θ 1 s 2 cos θ 2 . . . s d h cos θ d h      (33
)
The expected squared norm of z is then given by:
E[∥z∥ 2 2 ] = d h l=1 s 2 l E[cos 2 θ l ](34)
Using the trigonometric identity cos 2 (θ) = 1+cos(2θ) 2 and the following integrals:
π 0 sin n-2 (θ) dθ = Γ( n-1 2 )Γ( 1 2 ) Γ( n 2 ) = √ πΓ( n-1 2 ) Γ( n 2 ) (35
) π 0 cos(2θ) sin n-2 (θ) dθ = 0 (36
)
where the second integral vanishes due to the symmetry of cos(2θ) about θ = π 2 , we compute the expectation:
E[cos 2 (θ l )] = π 0 cos 2 (θ l ) Γ( n 2 ) √ πΓ( n-1 2 ) sin n-2 (θ l ) f (θ l ) dθ l (37
) = Γ( n 2 ) √ πΓ( n-1 2 ) π 0 cos 2 (θ l ) sin n-2 (θ l )dθ (38
) = Γ( n 2 ) 2 √ πΓ( n-1 2 ) × √ πΓ( n-1 2 ) Γ( n 2 ) = 1 2(39)
Thus, by setting s l = 2 d h for all ℓ ∈ {1, • • • , d h }, we expect that the expected norm E[∥z∥ 2  2 ] is 1 at initialization.
this section cite: ['b90', 'b62', 'b11', 'b89']

Section: B. Implementation Details
Listings 1, 2 and 3 provide the Google JAX implementation of scaling vector (Section 4.4), input embedding (Section 4.1), and MLP block (Section 4.2), respectively.
import flax.linen as nn class Scaler(nn.Module): dim: int init: float scale: float def setup(self): self.scaler = self.param( nn.initializers.constant(1.0 * self.scale), self.dim, ) self.forward_scaler = self.init / self.scale def __call__(self, x: jnp.ndarray) -> jnp.ndarray: return self.scaler * self.forward_scaler * x Listing 1. A JAX implementation of Scaler (Section 4.4) import jax.numpy as jnp import flax.linen as nn class InputEmbedding(nn.Module): observation_dim: int hidden_dim: int shift_const: float input_scaler_init: float input_scaler_scale: float def setup(self): self.obs_rms = RunningMeanStd( shape=self.observation_dim ) self.w0 = nn.Dense( features=self.hidden_dim, use_bias=False ) self.input_scaler = Scaler( dim=self.observation_dim, init=input_scaler_init, scale=input_scaler_scale ) def __call__(self, observation: jnp.ndarray) -> jnp.ndarray: # RSNorm o = (observations -self.obs_rms.mean) / jnp.sqrt( self.obs_rms.var + self.epsilon ) # Shift + l2-Norm new_axis = jnp.ones((o.shape[:-1] + (1,))) * self.shift_const o = jnp.concatenate([o, new_axis], axis=-1) o = l2normalize(o, axis=-1) # Linear + Scaler h = self.w0(o) h = self.input_scaler(h) h = l2normalize(h, axis=-1) return h Listing 2. A JAX implementation of Input Embedding (Section 4.1). SimbaV2: Hyperspherical Normalization for Scalable Deep Reinforcement Learning import flax.linen as nn class SimbaV2Block(nn.Module): hidden_dim: int ffn_scaler_init: float ffn_scaler_scale: float alpha_scaler_init: float alpha_scaler_scale: float def setup(self): self.w1 = nn.Dense( features=4 * self.hidden_dim, use_bias=False )(x) self.mlp_scaler = Scaler( dim=4 * self.hidden_dim, init=ffn_scaler_init, scale=ffn_scaler_scale ) self.w2 = nn.Dense( features=self.hidden_dim, use_bias=False ) self.alpha = Scaler( dim=self.hidden_dim, init=alpha_scaler_init, scale=alpha_scaler_scale ) def __call__(self, x: jnp.ndarray) -> jnp.ndarray: residual = x # MLP + l2-Norm x = self.w1(x) x = self.mlp_scaler(x) x = nn.relu(x) x = self.w2(x) x = l2normalize(x, axis=-1) # LERP + l2-Norm x = l2normalize(residual + self.alpha(x -residual), axis=-1) return x Listing 3. A JAX implementation of MLP block (Section 4.2).
this section cite: []

Section: C. Hyperparameters
For all experiments, we use consistent hyperparameters across benchmarks. The default settings are listed in Table 3.
Table 3. Hyperparameters Table . The hyperparameters listed below are used consistently across all tasks using SimbaV2, unless stated otherwise. For the discount factor γ, we set it automatically using heuristics used by TD-MPC2 (Hansen et al., 2023).
Hyperparameter Notation Value Input Shift constant cshift 3.0 Output Number of return bins natoms 101 Support of return [Gmin, Gmax] [-5, 5] Reward scaler epsilon ϵ 1e-8 Training Input scaler
(s 0 h,init , s 0 h,scale ) ( √ 2/ √ d h , √ 2/ √ d h ) MLP scaler (s l h,init , s l h,scale ) ( √ 2/ √ 4d h , √ 2/ √ 4d h ) Output scaler (so,init, so,scale) ( √ 2/ √ d h , √ 2/ √ d h ) LERP vector (αinit, αscale) (1/(L + 1), 1/ √ d h ) Behavior
cloning weight λ Online: 0.0 Offline: 0.1 Common Discount factor γ Heuristic (Hansen et al., 2023) Replay buffer capacity -1M Buffer sampling -Uniform Batch size -256 Update-to-data (UTD) ratio -2 TD steps k 1 Actor Number of blocks L 1 Hidden dimension d h 128 Initial temperature α0 1e-2 Target entropy H * |A|/2 Critic Number of blocks L 2 Hidden dimension d h 512 Number of atoms natoms 101 Target critic momentum τ 5e-3 Clipped double Q -Has Failure Termination (Mujoco, HBench): True No Failure Termination (DMC, MyoSuite): False Optimizer Optimizer -Adam Optimizer momentum (β1, β2) (0.9, 0.999) Weight Decay -0.0 Learning rate init η 1e-4 Learning rate final -3e-4
this section cite: ['b37']

Section: D. Offline RL
In this section, we assess whether the SimbaV2 architecture also provide benefits in offline RL, training from a stationary distribution. We adopt the minimalist offline RL method from (Fujimoto & Gu, 2021), where the behavioral cloning loss is integrated into the reinforcement learning objective. The objective is defined as:
π ≈ arg max π E (s,a)∼D Q(s, π(s)) -λ |E s∼D [Q(s, π(s))]| • (π(s) -a) 2(40)
where we used λ = 0.1, as in (Fujimoto et al., 2023), and no parameter tuning is performed.
this section cite: ['b25', 'b28']

Section: D.1. Experimental Setup
Environment. We use 9 MuJoCo tasks from the D4RL (Fu et al., 2020) benchmark, covering 3 environments (HalfCheetah, Hopper, Walker2d) and 3 difficulty levels (Medium, Medium-Replay, Medium-Expert).
Baselines. We compare SimbaV2 against standard offline RL methods: Percentile BC, Decision Transformer (DT, (Chen et al., 2021a)), Diffusion Q-Learning (DQL, (Wang et al., 2022)), Implicit Diffusion Q-Learning (IDQL, (Hansen-Estruch et al., 2023)), Conservative Q-Learning (CQL, (Chen et al., 2021a)), TD3+BC (Fujimoto & Gu, 2021), Implicit Q-Learning (IQL, (Kostrikov et al., 2021)), Extreme Q-Learning (X -QL, (Garg et al., 2023)), and TD7+BC (Fujimoto et al., 2023).
The results for Percentile BC, DT, DQL, and IDQL is from (Hansen-Estruch et al., 2023), while CQL, TD3+BC, IQL, X -QL, and TD7 results come from (Fujimoto et al., 2023).
Metrics. Following the standard offline RL protocol (Fu et al., 2020), we normalize the score of each environment based on the expert trajectory in the dataset.
Training. We use the same training configuration as in online RL (Appendix C), with a learning rate decaying linearly from 1 × 10 -4 to 1 × 10 -5 over 100 epochs, and include an additional behavioral cloning loss.
this section cite: ['b24', 'b88', 'b38', 'b25', 'b50', 'b31', 'b28', 'b38', 'b28', 'b24']

Section: D.2. Results
Table 4 reports the performance of SimbaV2 + BC, averaged over 10 random seeds.  (Chen et al., 2021a) 48.4 40.6 92.9 56.9 75.9 110.9 75.0 62.5 109.0 74.0 DT (Chen et al., 2021a) 42.6 36.6 86.8 67.6 82.7 110.9 74.0 66.6 108.1 74.7 TD3+BC (Fujimoto & Gu, 2021) 48.1±0.1 44.6±0.4 93.7±0.9 59.1±3.0 52.0±10.6 98.1±10.7 84.3±0.8 81.0±3.4 110.5±0.4 74.6±1.7 IQL (Kostrikov et al., 2021) 47.4±0.
2 43.9±1.3 89.6±3.5 63.9±4.9 93.4±7.8 64.2±32.0 84.2±1.6 71.2±8.3 108.9±1.4 74.1±3.8 DQL (Wang et al., 2022) 50.6 45.8 93.3 75.2 94.5 102.1 83.4 86.7 109.6 82.4 X -QL (Garg et al., 2023) 47.4±0.1 44.2±0.7 90.2±2.7 67.7±3.6 82.0±14.9 92.0±10.0 79.2±4.0 61.8±7.7 110.3±0.2 75.0±2.3 IDQL (Hansen-Estruch et al., 2023) 49.7 45.1 94.4 63.1 82.4 105.3 80.2 79.8 111.6 79.1 TD7+BC (Fujimoto et al., 2023) 58.0±0.4 53.8±0.8 104.6±1.6 76.1±5.1 91.1±8.0 108.2±4.8 91.1±7.8 89.7±4.7 111.8±0.6 87.2±1.6 SimbaV2+BC (ours) 54.8±0.5 48.6±0.8 92.2±1.4 98.1±2.4 99.9±0.6 106.2±1.5 82.7±10.3 87.7±2.1 110.6±0.6 86.7±1.6
With minimal changes, SimbaV2 performs highly competitively with existing offline RL algorithms, with statistically significantly better performance on Hopper. Again, this experimental results reinforces the importance of architectural design over complex algorithmic modifications. We believe our architectural approach offers exciting future potential for bridging offline and online RL (Ball et al., 2023;Zhou et al., 2024).
this section cite: ['b25', 'b50', 'b2', 'b95']

Section: E. Baselines
PPO (Lillicrap, 2015). Proximal Policy Optimization (PPO) is an on-policy policy gradient method that constrains updated policies to remain proximal to the old policies to circumvent performance collapse. Results for Gym -MuJoCo and DMC were obtained from Fujimoto et al. (2025), which are averaged over 10 seeds.
SAC (Haarnoja et al., 2018). Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm in which the actor simultaneously maximizes expected return and entropy, encouraging both stability and exploration. For the MuJoCo tasks, results averaged over 10 random seeds were obtained directly from the Bhatt et al. (2024) authors, with the update-to-data (UTD) ratio set to 1. For DMC, MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository for 10 random seeds, with the update-to-data (UTD) ratio set to 2.
TD3 (Fujimoto et al., 2018). Twin Delayed DDPG (TD3) is an off-policy actor-critic algorithm that mitigates Qoverestimation bias via three key techniques: (i) clipped double Q-learning, (ii) delayed policy updates, (iii) target policy smoothing. Results for Gym-MuJoCo were obtained from Table 1 of Fujimoto et al. (2023). These scores are averaged over 10 random seeds.
TD3+OFE (Ota et al., 2020). By replacing the encoder with an Online Feature Extractor (OFE)-trained via a dynamics prediction task to produce high-dimensional representations of observation-action pairs-TD3+OFE outperforms the original TD3 without requiring any hyperparameter adjustments. Results for Gym-MuJoCo were obtained from Table 1 of Fujimoto et al. (2023). These scores are averaged over 10 random seeds. We attach these results into Table 1 by TD3-normalizing the scores as outlined in Appendix F.1.
TQC (Kuznetsov et al., 2020). Truncated Quantile Critic (TQC) proposes to truncate the return distribution of the distributional critics to flexibly balance between under-and overestimation bias of Q-value. Results for Gym-MuJoCo were taken directly from Table 1 of Fujimoto et al. (2023). We attach these results into Table 1 by TD3-normalizing the scores as described in Appendix F.1.
REDQ (Chen et al., 2021b). Randomized Ensembled Double Q-Learning (REDQ) expands clipped double Q-learning from two Q-networks to an ensemble of ten to control estimation bias and variance, and enhance training stability. For the MuJoCo tasks, results averaged over 10 random seeds were obtained directly from the Bhatt et al. ( 2024) authors, with the update-to-data (UTD) ratio set to 20.
DroQ (Chen et al., 2021b). Dropout Q-Function (DroQ) reduces the computational burden of REDQ by using a smaller ensemble of Q functions while employing Dropout and Layer Normalization to stabilize training against Dropout-induced noise. For the MuJoCo tasks, results averaged over 10 random seeds were obtained directly from the Bhatt et al. (2024) authors, with the update-to-data (UTD) ratio set to 20.
DreamerV3 (Hafner et al., 2023). DreamerV3 encodes sensory inputs into categorical representations to build a learned world model, enabling long-horizon behavior learning in its compact latent space. Results for Gym-MuJoCo and DMC were obtained from Fujimoto et al. (2025), which are averaged over 10 seeds. For MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository (https://github.com/  SonyResearch/simba) over 3 random seeds.
TD7 (Fujimoto et al., 2023). TD7 improves TD3 by combining TD3 with four key improvements: (i) state-action representation learning (SALE), (ii) prioritized experience replay, (iii) policy checkpoints, and (iv) additional behavior cloning loss for offline RL. Results for Gym-MuJoCo and DMC were obtained from Fujimoto et al. (2025), which are averaged over 10 seeds. For MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository (https://github.com/SonyResearch/simba) over 5 random seeds.
TD-MPC2 (Hansen et al., 2023). TD-MPC2 is a model-based algorithm that learns an implicit (decoder-free) world model through multiple dynamics prediction tasks and performs local trajectory optimization within the learned latent space.
Results for Gym-MuJoCo and DMC were obtained from Fujimoto et al. (2025), which are averaged over 10 seeds. For MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository (https://github.com/SonyResearch/simba) over 3 random seeds.
CrossQ (Bhatt et al., 2024). CrossQ achieves superior performance and sample efficiency with low replay ratio, by removing target networks and employing careful batch normalization. Results for Gym-MuJoCo were obtained by running the official repository (https://github.com/adityab/CrossQ) for 10 random seeds, iQRL (Scannell et al., 2024). Implicitly Quantized Reinforcement Learning (iQRL) is a representation learning technique of model-free RL that prevents representation collapse and improve sample-efficiency via latent quantization. For the DMC hard tasks, results averaged over 3 random seeds were obtained directly from the authors. BRO (Nauman et al., 2024b). Bigger, Regularized, Optimistic (BRO) scales the critic network of SAC by integrating distributional Q-learning, optimistic exploration, and periodic resets. Results for Gym-MuJoCo and DMC Easy were obtained by running the official repository (https://github.com/naumix/BiggerRegularizedOptimistic) for 5 random seeds. For DMC hard, MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository (https://github.com/SonyResearch/simba) over 5 random seeds for HBench tasks and 10 random seeds for DMC hard and MyoSuite tasks. Unless stated otherwise, we set update-to-data (UTD) ratio to be 2.
MAD-TD (Voelcker et al., 2024). Model-Augmented Data for Temporal Difference learning (MAD-TD) aims to stabilize high UTD training by mixing a small fraction α of model-generated on-policy data with real off-policy replay data. For the DMC hard tasks, results averaged over 10 random seeds were obtained directly from the authors using the best algorithm setting (UTD = 8, α = 0.05).
MR.Q (Fujimoto et al., 2025). Model-based Representations for Q-learning (MR.Q) is a model-free algorithm that uses model-based objectives, such as dynamics and reward prediction, to obtain rich representation for actor-critic agent. We use the results for Gym-MuJoCo and DMC from Fujimoto et al. (2025) which were obtained by running the official repository (https://github.com/facebookresearch/MRQ) over 10 random seeds.
Simba (Lee et al., 2024c). SimBa is an architecture designed to scale up parameters in deep reinforcement learning by injecting a simplicity bias with observation normalizer, residual blocks, and layer normalizations. For Gym-MuJoCo, DMC, MyoSuite, and HBench tasks, we use the results from Lee et al. (2024c) which were obtained by running the official repository (https://github.com/SonyResearch/simba) over 15 random seeds for DMC hard tasks and 10 random seeds otherwise. Unless stated otherwise, we set update-to-data (UTD) ratio to be 2.
this section cite: ['b63', 'b29', 'b35', 'b5', 'b28', 'b28', 'b28', 'b5', 'b36', 'b29', 'b28', 'b29', 'b37', 'b29', 'b5', 'b70', 'b85', 'b29', 'b29']

Section: F. Environment Details
This section outlines the benchmark environments used in our evaluation. A complete list of all tasks from each benchmark, including their observation and action dimensions, is provided at the end of this section. Additionally, Table 5 outlines the episode length, action repeat, total number of environment steps, and performance metrics for each task domain.
this section cite: []

Section: F.1. Gym -MuJoCo
Gym (Brockman, 2016;Towers et al., 2024) is a suite of benchmark environments spanning finite MDPs to Multi-Joint dynamics with Contact (Todorov et al., 2012, MuJoCo) simulations. It offers a diverse range of tasks, including classic Atari games, small-scale tasks such as Toy Text and classic controls, as well as physics-based continuous robot control. For our experiments, we focus on 5 locomotion tasks within MuJoCo environments, which simulate complex physical interactions involving multi-body dynamics and contact forces. A complete list of these tasks is provided in Table 6. Note that we use the v4 version.
For comparison across different score scales of each task, all MuJoCo scores are normalized using TD3 and the random score for each task, as provided in TD7 (Fujimoto et al., 2023).
TD3-Normalized(x) := xrandom score TD3 scorerandom score Task Random TD3 Ant-v4 -70.288 3942 HalfCheetah-v4 -289.415 10574 Hopper-v4 18.791 3226 Humanoid-v4 120.423 5165 Walker2d-v4 2.791 3946 F.2. DeepMind Control Suite
DeepMind Control Suite (Tassa et al., 2018, DMC) is a standard continuous control benchmarks, encompassing a variety of locomotion and manipulation tasks with varying levels of complexity. These tasks range from simple low-dimensional settings (O ∈ R 3 , A ∈ R 1 ) to highly complex scenarios (O ∈ R 223 , A ∈ R 38 ). Our evaluation includes 27 DMC tasks, divided into two categories: DMC-Easy&Medium and DMC-Hard. All Humanoid and Dog tasks are grouped as DMC-Hard, while the rest are fall under DMC-Easy&Medium. Comprehensive lists of DMC-Easy&Medium and DMC-Hard are available in Tables 7 and 8, respectively.
this section cite: ['b9', 'b83', 'b28']

Section: F.3. MyoSuite
MyoSuite (Caggiano et al., 2022) models human motor control using musculoskeletal simulations of the human elbow, wrist, and hand, focusing on physiologically accurate movements. It provides benchmarks for intricate real-world object manipulation, ranging from simple posing tasks to the simultaneous manipulation of two Baoding balls. Our evaluation focuses on 10 MyoSuite tasks involving the hand. As defined by the authors, each task is categorized as hard when the goal is randomized; otherwise the goal is fixed. The full list of MyoSuite tasks is presented in Table 9.
this section cite: ['b10']

Section: F.4. HumanoidBench
HumanoidBench (Sferrazza et al., 2024) serves as a high-dimensional simulated robot learning benchmark, leveraging the Unitree H1 humanoid robot equipped with dexterous hands. It encompasses a diverse set of whole-body control tasks, spanning from fundamental locomotion to complex human-like activities that require refined manipulation. In our experiments, we concentrate on 14 locomotion tasks. A comprehensive list of tasks is provided in Table 10.
Note that the locomotion tasks do not necessitate hand dexterity. Therefore, to reduce the complexity arising from high degrees of freedom (DoF) and complex dynamics, we streamline the environments setup by excluding the hands of humanoid. For example, in case of walk, this drastically declines the dimension of the observation and action spaces by approximately 66%.
walk Without hand With 2 hand Observation dim |O| 51 151 Action dim |A| 19 61 DoF (body) 25 25 DoF (two hands) 0 50 For comparison across different score scales of each task, all HumanoidBench scores are normalized using each task's target success score provided by the authors and random score. Random scores are measured by the average undiscounted returns over 10 episodes of random agent. Each measurement is repeated over 10 seeds. Success-Normalized(x) := xrandom score Target success scorerandom score Task Random Target Success h1-balance-simple 9.391 800 h1-balance-hard 9.044 800 h1-crawl 272.658 700 h1-hurdle 2.214 700 h1-maze 106.441 1200 h1-pole 20.09 700 h1-reach 260.302 12000 h1-run 2.02 700 h1-sit-simple 9.393 750 h1-sit-hard 2.448 750 h1-slide 3.191 700 h1-stair 3.112 700 h1-stand 10.545 800 h1-walk
this section cite: ['b74']

Section: 700
Table 6. Gym-MuJoCo. We evaluate a total of 5 continuous control tasks from the Gym-MuJoCo benchmark. Below, we provide a list of all the tasks considered. The baseline performance for each task is reported at 1M environment steps.  Task Observation dim |O| Action dim |A| dog-run 223 38 dog-trot 223 38 dog-stand 223 38 dog-walk 223 38 humanoid-run 67 24 humanoid-stand 67 24 humanoid-walk 67 24
Table 9. MyoSuite Complete List. We evaluate a total of 10 continuous control tasks from the MyoSuite benchmark including both fixed-goal and randomized-goal (hard) settings. Below, we provide a list of all the tasks considered. The baseline performance for each task is reported at 1M environment steps.
this section cite: []

Section: Task Observation dim |O| Action dim |A|
myo-key-turn 93 39 myo-key-turn-hard 93 39 myo-obj-hold 91 39 myo-obj-hold-hard 91 39 myo-pen-twirl 83 39 myo-pen-twirl-hard 83 39 myo-pose 108 39 myo-pose-hard 108 39 myo-reach 115 39 myo-reach-hard 115 39
Table 10. HumanoidBench Complete List. We evaluate a total of 14 continuous control locomotion tasks from the HumanoidBench benchmark that simulates the UniTree H1 humanoid robot. Below, we provide a list of all the tasks considered. The baseline performance for each task is reported at 1M environment steps.
Task Observation dim |O| Action dim |A| h1-balance-hard 77 19 h1-balance-simple 64 19 h1-crawl 51 19 h1-hurdle 51 19 h1-maze 51 19 h1-pole 51 19 h1-reach 57 19 h1-run 51 19 h1-sit-simple 51 19 h1-sit-hard 64 19 h1-slide 51 19 h1-stair 51 19 h1-stand 51 19 h1-walk 51 19
this section cite: []

Section: G. Training Stability
In Section 5.2, we investigated the training dynamics of SimbaV2 on DMC-Hard and HBench-Hard via four metrics: feature norm, parameter norm, gradient norm, and effective learning rate (ELR) of neural networks. This section presents these standalone metrics for SimbaV2 to highlight its stable behavior throughout training.
Effective Learning Rate. We base our notion of ELR on the effective step size of Kodryan et al. (2022), omitting the global learning rate η and using dimension-based weighting
w i = |θi| N j=1 |θj | instead of squared-parameter-norm weighting w i = ∥θi∥ 2 N j=1 ∥θj ∥ 2 . Definition G.1 (Effective Learning Rate). Let θ = {θ i } N
i=1 be the parameter set of a neural network, and g i be the back-propagated gradient associated with θ i . The (total) effective learning rate ELR of the network is defined as:
ELR ≜ N i=1 w i ∥g i ∥ 2 ∥θ i ∥ 2(41)
where
w i = |θi| N j=1 |θj | .
Intuitively, our ELR measures the "effective" gradient step-per parameter dimension-before scaled by the global learning rate.
Metrics. To reflect dimensional contributions across layers, we also apply the same weighting w i when computing the feature norm, parameter norm, and gradient norm. For instance, our gradient norm is defined as:
∥g i ∥ 2 ≜ N i=1 w i ∥g i ∥ 2 2 (42)
where ∥ • ∥ 2 is the standard ℓ 2 -norm (Frobenius norm ∥ • ∥ F in case of matrices). Analogous expressions are applied for feature and parameter norms. We separate encoder layers (all layers preceding the output) from predictor layers (all layers after) to capture their distinct roles in the network. Average returns are normalized by maximum score 1000 for DMC-Hard, by success and random scores for HBench-Hard (Appendix F.4).
this section cite: ['b48']

Section: Results.
Figure 9 shows the tracked metrics over 1 million training steps. Certain features (e.g., logits) and parameters (e.g., scalers and interpolation vectors) may occasionally exceed unit norm, the overall parameter norms are tightly controlled (Figure 9.(b)-(c)), and gradient magnitudes are consistently balanced across modules (Figure 9.(d)). This leads to the consistent trend and scales of their ELRs over time. We hypothesize that this stable behavior contributes to SIMBAV2's improved performance and scalability. SimbaV2: Hyperspherical Normalization for Scalable Deep Reinforcement Learning
this section cite: []

Section: H. Additional Experiments
This section complements Section 5.2 by presenting further experiments probing the properties and robustness of SimbaV2:
• Scalability Effect of Hyperspherical Normalization (Section H.1). Investigate the necessity of hyperspherical normalization for achieving SimbaV2's scalability.
• Effectiveness beyond SAC (Section H.2.) Assess SimbaV2's broader applicability by substituting SAC with DDPG.
this section cite: []

Section: H.1. Scalability Effect of Hyperspherical Normalization
In Section 5.3, we observe that SimbaV2 consistently scales with an increasing update-to-data (UTD) ratio, even without reinitialization, while Simba saturates at a ratio of 2. However, this raises the question of whether hyperspherical normalization is critical for UTD scaling. This section investigates the effectiveness of hyperspherical normalization in scalability.
Experimental Setup. In this experiment, we examine a "Simba-like" variant, named Simba+, which incorporates distributional critic and reward scaling but only excludes the hyperspherical normalization. In other words, Simba+ is identical to SimbaV2 except that it excludes hyperspherical normalization. On DMC-Hard tasks, we compare SimbaV2, Simba, and Simba+ under varying model sizes and UTD ratios to determine the role of hyperspherical normalization in scaling performance.
Result. Figure 10 shows the scaling results. In Figure 10 (left), all three methods benefit from increased model capacity, but Simba+ slightly underperforms at larger parameter counts. More critically, Simba and Simba+ both plateau when the UTD ratio surpasses 2 in Figure 10 (right), while SimbaV2 continues to improve. These results confirm that hyperspherical normalization is truly indispensable for UTD scaling.
this section cite: []

Section: H.2. Effectiveness beyond SAC
In Section 5.2, we observe that replacing the neural network of SAC with SimbaV2 consistently improves the performance of a wide range of domains. To assess the broader applicability and robustness of SimbaV2's architectural advantages beyond a single algorithm, we conducted additional experiments using Deep Deterministic Policy Gradient (Lillicrap, 2015, DDPG), another widely adopted off-policy algorithm for continuous control.
Experimental Setup. We evaluated SimbaV2 against the original Simba (Lee et al., 2024c) and a standard MLP baseline on two challenging continuous control benchmark suites: DMC-Hard and HBench-Hard. All methods utilized the DDPG algorithm as their underlying learning framework. The MLP architecture we adopted consists of a sequence of linear layers followed by ReLU non-linearities.
this section cite: []

Section: Result.
Comparative results are presented in Figure 11.
On the DMC-Hard benchmark, SimbaV2 achieved performance competitive with the original Simba, with both significantly outperforming the MLP baseline. More notably, on the more complex HBench-Hard benchmark, SimbaV2 demonstrated a clear improvement over Simba. These results indicate that SimbaV2 not only generalizes to the DDPG algorithm but also exhibits enhanced stability and generalization capabilities in more demanding environments, likely attributable to its refined architecture and regularization mechanisms. 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0.00 0.18 0.36 0.54 0.72 Total Reward (K) DMC-Hard 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0.00 0.18 0.36 0.54 0.72 Sucess-Norm Score HBench-Hard SimbaV2 + DDPG Simba + DDPG MLP + DDPG  (Lee et al., 2024c), and the MLP baseline on DMC-Hard and HBench-Hard benchmarks using DDPG (Lillicrap, 2015). SimbaV2 performs competitively with Simba on DMC-Hard, both significantly outperforming the MLP baseline. In the more challenging HBench-Hard, SimbaV2 shows clear improvements over Simba, indicating enhanced stability and generalization beyond SAC.
this section cite: ['b63']

Section: I. Complete UTD Scaling Results

this section cite: []

Section: I.1. Gym -MuJoCo

this section cite: []

Section: I.2. Deepmind Control Suite -Easy

this section cite: []

Section: I.3. Deepmind Control Suite -Hard
Table 13. DMC-Hard UTD Scaling Results. Final average performance at 1M environment steps for each of the 7 tasks of the DMC-Hard benchmark. The number of evaluated random seeds for each update-to-data (UTD) ratio is provided 5. The values in [brackets] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are reported in units of 1k.
Task UTD = 1 UTD = 2 UTD = 4 UTD = 8 dog-run 477 [429, 525] 562 [516, 608] 655 [620, 691] 555 [523, 587] dog-stand 967 [959, 974] 981 [977, 985] 967 [960, 974] 972 [967, 976] dog-trot 850 [810, 890] 861 [772, 950] 846 [782, 910] 898 [888, 909] dog-walk 921 [912, 930] 935 [927, 944] 923 [905, 941] 949 [945, 953] humanoid-run 183 [164, 203] 194 [182, 207] 272 [230, 313] 253 [228, 278] humanoid-stand 660 [585, 734] 916 [886, 945] 928 [926, 930] 933 [924, 941] humanoid-walk 568 [533, 603] 651 [590, 713] 818 [751, 885] 819 [762, 877] IQM 0.713 [0.598, 0.809] 0.808 [0.725, 0.88] 0.851 [0.755, 0.916] 0.849 [0.727, 0.924] Median 0.666 [0.563, 0.774] 0.729 [0.655, 0.81] 0.771 [0.678, 0.868] 0.767 [0.652, 0.861] Mean 0.669 [0.581, 0.753] 0.729 [0.663, 0.791] 0.769 [0.687, 0.845] 0.759 [0.67, 0.84] I.4. MyoSuite  I.5. Humanoid Bench
this section cite: []

Section: J.2. Deepmind Control Suite -Easy
Table 17. DMC Easy. Final average performance at 1M environment steps for each of the 21 tasks of the DMC Easy benchmark. The number of evaluated random seeds for each algorithm is provided in Appendix E. The values in [brackets] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are reported in units of 1k.
Task DreamerV3 TD7 TD-MPC2 MR.Q BRO Simba SimbaV2 acrobot-swingup 230 [193, 266] 58 [38, 75] 584 [551, 615] 567 [523, 616] 529 [504, 555] 431 [379, 482] 436 [391, 482] ball-in-cup-catch 968 [965, 973] 984 [982, 986] 983 [981, 985] 981 [979, 984] 982 [981, 984] 981 [978, 983] 982 [980, 984] cartpole-balance 998 [997, 1000] 999 [998, 1000] 996 [995, 998] 999 [999, 1000] 999 [998, 999] 998 [998, 999] 999 [999, 999] cartpole-balance-sparse 1000 [1000, 1000] 999 [1000, 1000] IQM 0.021 [0.003, 0.069] † 0.069 [0.042, 0.114] 0.694 [0.528, 0.805] 0.772 [0.662, 0.854] 0.787 [0.691, 0.865] Median 0.026 [0.001, 0.040] † 0.159 [0.08, 0.183] 0.64 [0.516, 0.766] 0.694 [0.615, 0.774] 0.707 [0.634, 0.786] Mean 0.033 [0.009, 0.068] † 0.136 [0.098, 0.175] 0.642 [0.531, 0.747] 0.693 [0.625, 0.757] 0.708 [0.642, 0.771] J.4. MyoSuite
this section cite: []

Section: 
841 [824, 858] cheetah-run 917 [913, 920] 920 [918, 922] 902 [868, 937] 916 [912, 920] finger-spin 940 [895, 985] 891 [810, 972] 762 [608, 915] 910 [790, 1030] finger-turn-easy 951 [916, 987] 953 [925, 980] 954 [917, 992] 936 [857, 1014] finger-turn-hard 928 [885, 972] 951 [925, 977] 902 [866, 939] 950 [910, 990] fish-swim 818 [779, 856] 826 [806, 846] 815 [780, 850] 807 [778, 836] hopper-hop 379 [224, 535] 290 [233, 348] 326 [243, 410] 317 [230, 404] hopper-stand 845 [704, 986] 944 [926, 962] 781 [449, 1112] 932 [898, 967] pendulum-swingup 817 [776, 858] 827 [805, 849] 820 [781, 859] 821 [784, 859] quadruped-run 931 [922, 940] 935 [928, 943] 943 [936, 949] 935 [930, 940] quadruped-walk 962 [955, 970] 962 [955, 969] 964 [958, 971] 965 [958, 972] reacher-easy 963 [927, 1000] 983 [979, 986] 975 [958, 992] 983 [981, 985] reacher-hard 975 [971, 980] 967 [946, 987] 976 [972, 980] 974 [970, 978] walker-run 920 [918, 922] 821 [642, 913] finger-spin 666 [577, 763] 335 [99, 596] 986 [986, 988] 937 [917, 956] 988 [987, 989] 849 [758, 939] 891 [810, 972] finger-turn-easy 906 [883, 927] 912 [774, 983] 979 [975, 983] 953 [931, 974] 957 [923, 992] 935 [903, 968] 953 [925, 980] finger-turn-hard 864 [812, 900] 470 [199, 727] 947 [916, 977] 950 [910, 974] 957 [920, 993] 915 [859, 972] 951 [925, 977] fish-swim 813 [808, 819] 86 [64, 120] 659 [615, 706] 792 [773, 810] 618 [523, 713] 823 [799, 846] 826 [806, 846] hopper-hop 116 [66
this section cite: []

Section: J. Complete Main Results
This section provides learning curves and final performance for each online RL task across the evaluated algorithms.
Learning Curve. For visibility of learning curve, we focus on DreamerV3 (Hafner et al., 2023), TD7 (Fujimoto et al., 2023), TD-MPC2 (Hansen et al., 2023), MR.Q (Fujimoto et al., 2025), and Simba (Lee et al., 2024c) as main baselines, selected for their strong performance and community adoption. We omit curves for algorithms with unavailable raw samples at each task.
this section cite: ['b36', 'b28', 'b37', 'b29']

Section: Confidence Interval.
The light-colored area in the figures and the gray-shaded, bracketed terms in the tables represent 95% bootstrap confidence intervals. For each task evaluated over n random seeds, the 95% bootstrap confidence interval CI is computed as:
CI = µ -1.96 × σ √ n , µ + 1.96 × σ √ n
where µ and σ are the sample mean and standard deviation (with Bessel's correction) of the evaluation, respectively. For aggregated scores (mean, median, and interquartile mean), confidence intervals are computed over all n × T raw samples, where n and T are the number of evaluated random seeds and tasks in the benchmark, respectively. For algorithms with only average scores for each task available, we approximate the CI of aggregated scores using these averages (denoted with gray-colored †). We caution that this estimation may be inaccurate. 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Average Return Success h1-walk-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-stand-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-run-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 2500 5000 7500 10000 12500 15000 Success h1-reach-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Average Return Success h1-hurdle-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-crawl-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 1250 1500 Success h1-maze-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-sit-simple-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Average Return Success h1-sit-hard-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-balance-simple-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-balance-hard-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-stair-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Average Return Success h1-slide-v0 0 0.2 0.4 0.6 0.8 1.0 Env steps (M) 0 250 500 750 1000 Success h1-pole-v0 DreamerV3 TD7 TD-MPC2 Simba SimbaV2
this section cite: []

Section: J.1. Gym -MuJoCo

this section cite: []

Section: K. Complete Ablation Results
This section presents a per-environment analysis of the design variations discussed in Section 5.5. Each table includes raw scores for individual environments, with [bracketed values] indicating 95% bootstrap confidence intervals. The aggregate mean, median, and interquartile mean (IQM) are calculated based on the differences in normalized scores. To illustrate the magnitude of these differences, we use the following highlight scale:
• (≥ 0.1)
• [0.05, 0.1)
• [0.02, 0.05)
• [-0.02, -0.05)
• [-0.05, -0.1) Table 26. DMC-Easy (Training Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
• (≤ -0.05) K.1. Gym -MuJoCo
Task SimbaV2 No LR Decay sinit : 1 sscale : 1 αinit : 0.5 αscale : 1 IQM 0.933 [0.918, 0.948] 0.923 [0.894, 0.947] 0.932 [0.916, 0.947] 0.934 [0.918, 0.949] 0.927 [0.901, 0.949] 0.922 [0.892, 0.947] Median 0.875 [0.847, 0.904] 0.858 [0.811, 0.902] 0.878 [0.847, 0.906] 0.871 [0.842, 0.902] 0.863 [0.818, 0.911] 0.859 [0.809, 0.902] Mean 0.874 [0.849, 0.897] 0.858 [0.813, 0.896] 0.873 [0.848, 0.897] 0.87 [0.843, 0.894] 0.866 [0.819, 0.905] 0.856 [0.812, 0.895] IQM 0.808 [0.726, 0.88] 0.789 [0.65, 0.868] 0.805 [0.667, 0.893] 0.783 [0.663, 0.882] 0.795 [0.659, 0.894] Median 0.729 [0.655, 0.808] 0.732 [0.595, 0.816] 0.73 [0.619, 0.826] 0.717 [0.606, 0.823] 0.724 [0.61, 0.825] Mean 0.729 [0.665, 0.79] 0.7 [0.6, 0.795] 0.724 [0.629, 0.814] 0.718 [0.619, 0.809] 0.72 [0.619, 0.811]
Table 28. DMC-Hard (Output Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 MSE Loss No Reward Scaling No Return Bounding Hard Target dog-run 562 [516, 608] 545 [450, 639] 478 [402, 554] 617 [537, 696] 676 [654, 699] dog-stand 981 [977, 985] 976 [959, 993] 967 [956, 978] 969 [958, 981] 980 [971, 989] dog-trot 861 [772, 950] 841 [796, 886] 737 [614, 859] 884 [803, 964] 848 [763, 934] dog-walk 935 [927, 944] 905 [883, 928] 925 [915, 935] 922 [899, 945] 928 [891, 964] humanoid-run 194 [182, 207] 173 [146, 200] 237 [181, 293] 182 [154, 209] 209 [159, 260] humanoid-stand 916 [886, 945] 786 [612, 960] 879 [821, 936] 851 [744, 958] 928 [920, 937] humanoid-walk 651 [590, 713] 729 [577, 880] 754 [643, 865] 706 [563, 849] 645 [602, 689] IQM 0.808 [0.728, 0.881] 0.78 [0.62, 0.877] 0.778 [0.655, 0.871] 0.812 [0.69, 0.901] 0.814 [0.706, 0.902] Median 0.729 [0.655, 0.809] 0.705 [0.57, 0.836] 0.715 [0.61, 0.817] 0.731 [0.621, 0.84] 0.746 [0.641, 0.847] Mean 0.729 [0.665, 0.79] 0.708 [0.586, 0.813] 0.712 [0.626, 0.792] 0.733 [0.632, 0.824] 0.746 [0.648, 0.833]
Table 29. DMC-Hard (Training Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 No LR Decay sinit : 1 sscale : 1 αinit : 0.5 αscale : 1 IQM 0.808 [0.725, 0.879] 0.798 [0.658, 0.895] 0.783 [0.708, 0.851] 0.766 [0.687, 0.838] 0.819 [0.685, 0.892] 0.781 [0.642, 0.891] Median 0.729 [0.655, 0.808] 0.716 [0.616, 0.822] 0.719 [0.646, 0.794] 0.711 [0.634, 0.782] 0.724 [0.62, 0.833] 0.715 [0.603, 0.823] Mean 0.729 [0.664, 0.791] 0.719 [0.623, 0.809] 0.718 [0.656, 0.777] 0.706 [0.644, 0.767] 0.728 [0.627, 0.819] 0.714 [0.61, 0.81] SimbaV2: Hyperspherical Normalization for Scalable Deep Reinforcement Learning K.4. Myosuite
Table 30. Myosuite (Input Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 No L2 Normalize No Shifting cshift : 1 Resize Projection myo-key-turn 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-key-turn-hard 0.62 [0.427, 0.813] 0.325 [-0.009, 0.659] 0.25 [-0.044, 0.544] 0.8 [0.661, 0.939] 0.85 [0.752, 0.948] myo-obj-hold 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-obj-hold-hard 0.98 [0.954, 1.006] 0.975 [0.926, 1.024] 0.975 [0.926, 1.024] 1.0 [1.0, 1.0] 0.975 [0.926, 1.024] myo-pen-twirl 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pen-twirl-hard 0.93 [0.888, 0.972] 0.9 [0.82, 0.98] 0.875 [0.689, 1.061] 0.975 [0.926, 1.024] 0.8 [0.604, 0.996] myo-pose
1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pose-hard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] myo-reach 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-reach-hard0
.94 [0.873, 1.007] 0.95 [0.893, 1.007] 0.9 [0.82, 0.98] 0.925 [0.831, 1.019] 0.9 [0.82, 0.98] IQM 0.99 [0.968, 1.0] 0.98 [0.885, 1.0] 0.98 [0.86, 1.0] 1.0 [0.955, 1.0] 0.975 [0.925, 1.0] Median 0.845 [0.78, 0.925] 0.815 [0.695, 0.935] 0.805 [0.68, 0.92] 0.875 [0.765, 0.975] 0.86 [0.75, 0.955] Mean 0.847 [0.782, 0.906] 0.815 [0.702, 0.915] 0.8 [0.682, 0.905] 0.87 [0.77, 0.953] 0.852 [0.75, 0.938]
Table 31. Myosuite (Output Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 MSE Loss No Reward Scaling No Return Bounding Hard Target myo-key-turn 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-key-turn-hard 0.62 [0.427, 0.813] 0.2 [-0.192, 0.592] 0.76 [0.66, 0.86] 0.225 [-0.153, 0.603] 0.675 [0.417, 0.933] myo-obj-hold 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-obj-hold-hard 0.98 [0.954, 1.006] 0.933 [0.868, 0.999] 0.98 [0.941, 1.019] 1.0 [1.0, 1.0] 0.975 [0.926, 1.024] myo-pen-twirl 1.0 [1.0, 1.0] 0.667 [0.013, 1.32] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pen-twirl-hard 0.93 [0.888, 0.972] 0.867 [0.605, 1.128] 0.98 [0.941, 1.019] 0.9 [0.82, 0.98] 0.9 [0.82, 0.98] myo-pose 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 0.8 [0.408, 1.192] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pose-hard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] myo-reach 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-reach-hard 0.94 [0.873, 1.007] 0.9 [0.787, 1.013] 0.88 [0.766, 0.994] 0.925 [0.831, 1.019] 0.925 [0.831, 1.019] IQM 0.99 [0.968, 1.0] 0.944 [0.712, 1.0] 0.985 [0.931, 1.0] 0.985 [0.87, 1.0] 0.98 [0.93, 1.0] Median 0.845 [0.78, 0.93] 0.77 [0.58, 0.94] 0.85 [0.74, 0.96] 0.79 [0.68, 0.93] 0.845 [0.74, 0.955] Mean 0.847 [0.783, 0.906] 0.757 [0.613, 0.887] 0.84 [0.746, 0.922] 0.805 [0.685, 0.912] 0.848 [0.745, 0.938]
Table 32. Myosuite (Training Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 No LR Decay sinit : 1 sscale : 1 αinit : 0.5 αscale : 1 myo-key-turn 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 0.9 [0.704, 1.096] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-key-turn-hard 0.62 [0.427, 0.813] 0.65 [0.345, 0.955] 0.74 [0.585, 0.895] 0.69 [0.502, 0.878] 0.675 [0.581, 0.769] 0.85 [0.662, 1.038] myo-obj-hold 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-obj-hold-hard 0.98 [0.954, 1.006] 1.0 [1.0, 1.0] 0.95 [0.897, 1.003] 0.98 [0.954, 1.006] 0.95 [0.893, 1.007] 1.0 [1.0, 1.0] myo-pen-twirl 1.0 [1.0, 1.0] 0.75 [0.26, 1.24] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pen-twirl-hard 0.93 [0.888, 0.972] 0.775 [0.451, 1.099] 0.89 [0.81, 0.97] 0.88 [0.816, 0.944] 0.925 [0.831, 1.019] 1.0 [1.0, 1.0
] myo-pose 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-pose-hard 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] myo-reach 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] 1.0 [1.0, 1.0] myo-reach-hard0
.94 [0.873, 1.007] 0.925 [0.831, 1.019] 0.97 [0.928, 1.012] 0.91 [0.848, 0.972] 0.875 [0.752, 0.998] 0.9 [0.82, 0.98] IQM 0.99 [0.968, 1.0] 0.985 [0.875, 1.0] 0.99 [0.964, 1.0] 0.982 [0.948, 1.0] 0.975 [0.905, 1.0] 1.0 [0.97, 1.0] Median 0.845 [0.78, 0.925] 0.84 [0.7, 0.94] 0.865 [0.785, 0.935] 0.84 [0.77, 0.92] 0.85 [0.735, 0.945] 0.875 [0.77, 0.98] Mean 0.847 [0.782, 0.906] 0.81 [0.698, 0.908] 0.855 [0.792, 0.913] 0.836 [0.77, 0.896] 0.842 [0.742, 0.928] 0.875 [0.772, 0.96] IQM 0.799 [0.684, 0.907] 0.709 [0.525, 0.881] 0.708 [0.509, 0.908] 0.833 [0.635, 0.997] 0.808 [0.61, 0.977] Median 0.781 [0.69, 0.862] 0.698 [0.566, 0.851] 0.698 [0.552, 0.85] 0.801 [0.639, 0.944] 0.801 [0.631, 0.934] Mean 0.776 [0.704, 0.847] 0.711 [0.589, 0.833] 0.7 [0.578, 0.825] 0.791 [0.66, 0.921] 0.779 [0.653, 0.905] IQM 0.799 [0.684, 0.907] 0.778 [0.594, 0.95] 0.745 [0.584, 0.896] 0.819 [0.623, 0.977] 0.778 [0.572, 0.965] Median 0.781 [0.692, 0.863] 0.762 [0.614, 0.917] 0.722 [0.605, 0.864] 0.776 [0.636, 0.933] 0.772 [0.614, 0.923] Mean 0.776 [0.704, 0.848] 0.767 [0.637, 0.894] 0.735 [0.629, 0.842] 0.787 [0.658, 0.915] 0.77 [0.64, 0.897]
Table 35. HumanoidBench (Training Design). Final performance at 1M env steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward.
Task SimbaV2 No LR Decay sinit : 1 sscale : 1 αinit : 0.5 αscale : 1 IQM 0.799 [0.683, 0.908] 0.764 [0.578, 0.93] 0.796 [0.688, 0.901] 0.823 [0.709, 0.927] 0.725 [0.532, 0.917] 0.817 [0.615, 0.978] Median 0.781 [0.69, 0.862] 0.769 [0.599, 0.912] 0.776 [0.7, 0.866] 0.792 [0.705, 0.876] 0.746 [0.595, 0.893] 0.802 [0.64, 0.938] Mean 0.776 [0.704, 0.847] 0.754 [0.625, 0.883] 0.781 [0.711, 0.852] 0.789 [0.719, 0.861] 0.745 [0.619, 0.872] 0.792 [0.659, 0.916]
this section cite: []

Section: References
Ref_id:b0 Title: Optimization algorithms on matrix manifolds Year: (2008)
Ref_id:b1 Title: Gpt-4 technical report Year: (2023)
Ref_id:b2 Title: Efficient online reinforcement learning with offline data Year: (2023)
Ref_id:b3 Title: A distributional perspective on reinforcement learning Year: ()
Ref_id:b4 Title:  Year: (2017)
Ref_id:b5 Title: Batch normalization in deep reinforcement learning for greater sample efficiency and simplicity Year: (2024)
Ref_id:b6 Title: Towards deeper deep reinforcement learning with spectral normalization Year: (2021)
Ref_id:b7 Title: Stochastic gradient descent on riemannian manifolds Year: (2013-09)
Ref_id:b8 Title: An introduction to optimization on smooth manifolds Year: (2023)
Ref_id:b9 Title: Openai gym Year: (2016)
Ref_id:b10 Title: Myosuite-a contact-rich simulation suite for musculoskeletal motor control Year: (2022)
Ref_id:b11 Title: Distributions of angles in random packing on spheres Year: (2013)
Ref_id:b12 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b13 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b14 Title: Randomized ensembled double q-learning Year: (2021)
Ref_id:b15 Title: Scaling vision transformers to 22 billion parameters Year: (2023)
Ref_id:b16 Title:  Year: (1992)
Ref_id:b17 Title: Maintaining plasticity in deep continual learning Year: (2023)
Ref_id:b18 Title: Sample-efficient reinforcement learning by breaking the replay ratio barrier Year: (2022)
Ref_id:b19 Title: Sample-efficient reinforcement learning by breaking the replay ratio barrier Year: (2023)
Ref_id:b20 Title: Streaming deep reinforcement learning finally works Year: (2024)
Ref_id:b21 Title: Generalization and regularization in dqn Year: (2018)
Ref_id:b22 Title: Stop regressing: Training value functions via classification for scalable deep rl Year: (2024)
Ref_id:b23 Title: SimbaV2: Hyperspherical Normalization for Scalable Deep Reinforcement Learning Feller, W. An introduction to probability theory and its applications Year: (1991)
Ref_id:b24 Title: D4rl: Datasets for deep data-driven reinforcement learning Year: (2020)
Ref_id:b25 Title: A minimalist approach to offline reinforcement learning Year: (2021)
Ref_id:b26 Title: Addressing function approximation error in actor-critic methods Year: ()
Ref_id:b27 Title:  Year: (2018)
Ref_id:b28 Title: For sale: State-action representation learning for deep reinforcement learning Year: (2023)
Ref_id:b29 Title: Towards general-purpose model-free reinforcement learning Year: (2025)
Ref_id:b30 Title: Simplifying deep temporal difference learning Year: (2024)
Ref_id:b31 Title: Maxent rl without entropy Year: (2023)
Ref_id:b32 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b33 Title: Spectral normalisation for deep reinforcement learning: an optimisation perspective Year: (2021)
Ref_id:b34 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b35 Title: Soft actorcritic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor Year: (2018)
Ref_id:b36 Title: Mastering diverse domains through world models Year: (2023)
Ref_id:b37 Title: Td-mpc2: Scalable, robust world models for continuous control Year: (2023)
Ref_id:b38 Title: Implicit q-learning as an actorcritic method with diffusion policies Year: (2023)
Ref_id:b39 Title: Dropout q-functions for doubly efficient reinforcement learning Year: (2021)
Ref_id:b40 Title: Dissecting deep rl with high update ratios: Combatting value divergence Year: (2024)
Ref_id:b41 Title: Learning agile and dynamic motor skills for legged robots Year: (2019)
Ref_id:b42 Title: Improving regression performance with distributional losses Year: (2018)
Ref_id:b43 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b44 Title: Analyzing and improving the training dynamics of diffusion models Year: (2024)
Ref_id:b45 Title: Sampleefficient and safe deep reinforcement learning via reset deep ensemble agents Year: (2023)
Ref_id:b46 Title: A method for stochastic optimization Year: (2014)
Ref_id:b47 Title: Plasticity loss in deep reinforcement learning: A survey Year: (2024)
Ref_id:b48 Title: Training scale-invariant neural networks on the sphere can happen in three regimes Year: (2022)
Ref_id:b49 Title: Image augmentation is all you need: Regularizing deep reinforcement learning from pixels Year: (2020)
Ref_id:b50 Title: Offline reinforcement learning with implicit q-learning Year: (2021)
Ref_id:b51 Title: A simple weight decay can improve generalization Year: (1991)
Ref_id:b52 Title: Conservative q-learning for offline reinforcement learning Year: (2020)
Ref_id:b53 Title: DR3: Value-based deep reinforcement learning requires explicit regularization Year: (2022)
Ref_id:b54 Title: Controlling overestimation bias with truncated mixture of continuous distributional quantile critics Year: ()
Ref_id:b55 Title:  Year: (2020)
Ref_id:b56 Title: Plastic: Improving input and label plasticity for sample efficient reinforcement learning Year: (2024)
Ref_id:b57 Title: Slow and steady wins the race: Maintaining plasticity with hare and tortoise networks Year: (2024)
Ref_id:b58 Title: Simplicity bias for scaling up parameters in deep reinforcement learning Year: (2024)
Ref_id:b59 Title: Riemannian manifolds: an introduction to curvature Year: (2006)
Ref_id:b60 Title: Layer normalization Year: (2016)
Ref_id:b61 Title: Efficient deep reinforcement learning requires regulating overfitting Year: (2023)
Ref_id:b62 Title: Concise formulas for the area and volume of a hyperspherical cap Year: (2010)
Ref_id:b63 Title: Continuous control with deep reinforcement learning Year: (2015)
Ref_id:b64 Title: Deep hypersphere embedding for face recognition Year: (2017)
Ref_id:b65 Title: Deep hyperspherical learning Year: (2017)
Ref_id:b66 Title: Normalized transformer with representation learning on the hypersphere Year: (2024)
Ref_id:b67 Title: Understanding and preventing capacity loss in reinforcement learning Year: (2022)
Ref_id:b68 Title: Understanding plasticity in neural networks Year: (2023)
Ref_id:b69 Title: Normalization and effective learning rates in reinforcement learning Year: (2024)
Ref_id:b70 Title: iqrl-implicitly quantized representations for sample-efficient reinforcement learning Year: (2024)
Ref_id:b71 Title: Returnbased scaling: Yet another normalisation trick for deep rl Year: (2021)
Ref_id:b72 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b73 Title: Bigger, better, faster: Human-level atari with human-level efficiency Year: (2023)
Ref_id:b74 Title: Simulated humanoid benchmark for whole-body locomotion and manipulation Year: (2024)
Ref_id:b75 Title: The dormant neuron phenomenon in deep reinforcement learning Year: (2023)
Ref_id:b76 Title: Observational overfitting in reinforcement learning Year: (2019)
Ref_id:b77 Title: A comprehensive introduction to differential geometry. (No Title) Year: (1970)
Ref_id:b78 Title: Riemannian gradient descent for spherical area-preserving mappings Year: (2024)
Ref_id:b79 Title: Reinforcement learning: An introduction Year: (2018)
Ref_id:b80 Title:  Year: ()
Ref_id:b81 Title: Deepmind control suite Year: (2018)
Ref_id:b82 Title: Mujoco: A physics engine for model-based control Year: (2012)
Ref_id:b83 Title: Gymnasium: A standard interface for reinforcement learning environments Year: (2024)
Ref_id:b84 Title: Attention is all you need Year: (2017)
Ref_id:b85 Title: Mad-td: Model-augmented data stabilizes high update ratio rl Year: (2024)
Ref_id:b86 Title: Normface: L2 hypersphere embedding for face verification Year: (2017)
Ref_id:b87 Title: Understanding contrastive representation learning through alignment and uniformity on the hypersphere Year: (2020)
Ref_id:b88 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2022)
Ref_id:b89 Title:  Year: (2002)
Ref_id:b90 Title:  Year: (2005)
Ref_id:b91 Title: Mixture of experts in a mixture of rl settings Year: (2024)
Ref_id:b92 Title: On layer normalization in the transformer architecture Year: (2020)
Ref_id:b93 Title: Mastering visual reinforcement learning through dormant ratio minimization Year: (2023)
Ref_id:b94 Title: Spherical latent spaces for stable variational autoencoders Year: (2018)
Ref_id:b95 Title: Efficient online reinforcement learning fine-tuning need not retain offline data Year: (2024)
Ref_id:b96 Title:  Year: ()
Ref_id:b97 Title: Deepmind Control Suite -Hard Table 18. DMC Hard. Final average performance at 1M environment steps for each of the 7 tasks of the DMC Hard benchmark. The number of evaluated random seeds for each algorithm is provided in Appendix E. The values in [brackets] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are reported in units of 1k. Task DreamerV3 TD7 TD-MPC2 MR.Q Simba SimbaV2 dog-run 4 Year: ()
Ref_id:b98 Title:  Year: ()
Ref_id:b99 Title: The values in [brackets] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the success normalized score as described in Appendix F.4. Task DreamerV3 TD7 TD-MPC2 Simba SimbaV2 h1-pole-v0 41 Year: ()
Ref_id:b100 Title:  Year: ()
Ref_id:b101 Title: Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the TD3-normalized score as described in Appendix F.1. Task SimbaV2 MSE Loss No Reward Scaling No Return Bounding Hard Target Ant Year: ()
Ref_id:b102 Title:  Year: ()
Ref_id:b103 Title: Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean (IQM) are computed over the TD3-normalized score as described in Appendix F.1. Task SimbaV2 No LR Decay sinit : 1 sscale Year: ()
Ref_id:b104 Title:  Year: ()
Ref_id:b105 Title: SimbaV2: Hyperspherical Normalization for Scalable Deep Reinforcement Learning K.5. Humanoid Bench Table 33. HumanoidBench (Input Design). Final performance at 1M environment steps averaged over 3 seeds. The [bracketed values] represent a 95% bootstrap confidence interval. The aggregate mean, median and interquartile mean are computed over the default reward. Task SimbaV2 No L2 Normalize No Shifting cshift : 1 Resize Projection h1-balance-hard-v0 143 Year: ()
