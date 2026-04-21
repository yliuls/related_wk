Title: Quantization-Free Autoregressive Action Transformer
Abstract: Current transformer-based imitation learning approaches introduce discrete action representations and train an autoregressive transformer decoder on the resulting latent code. However, the initial quantization breaks the continuous structure of the action space thereby limiting the capabilities of the generative model. We propose a quantization-free method instead that leverages Generative Infinite-Vocabulary Transformers (GIVT) as a direct, continuous policy parametrization for autoregressive transformers. This simplifies the imitation learning pipeline while achieving state-of-the-art performance on a variety of popular simulated robotics tasks. We enhance our policy roll-outs by carefully studying sampling algorithms, further improving the results.

Section: Introduction
Generative modeling lies at the heart of modern machine learning, enabling the production of outputs that adhere to user-specified objectives and constraints by navigating high-dimensional spaces of potential outcomes. Fundamentally, this process can be understood as a form of search where naive or exhaustive strategies are computationally intractable due to the size of the space. We consider imitation learning for robotic systems as an extreme example, where the search space has uncountable cardinality and a complex structure arising from nonlinear dynamics, interactions with the environment and the fundamental multimodality in human or robot behavior. In many real-world robotics applications, the control signals involved are continuous in nature [33,59,28]. Current state-of-the-art methods for modeling policies in continuous control tasks broadly fall into two categories: autoregressive transformer models [26,45] and diffusion models [7,54,40], which will be discussed in the next two paragraphs.
Existing autoregressive policies, on the one hand, ignore the challenge of learning in a continuous domain by discretizing actions [26,45]. This discretization can introduce several drawbacks: It discards the inherent structure of the continuous space, increases complexity by adding a separate quantization step, and may limit expressiveness or accuracy when fine-grained control is required. Furthermore, the popular approach of learning a VQ-VAE [52] of the action space involves nondifferentiable operations and therefore requires advanced tricks during representation learning to sidestep optimization difficulties and instabilities [24,20,18,31]. Hence, our goal is to design autoregressive transformers that preserve the continuous nature of the action space, leveraging the smooth and potentially multimodal behavior patterns that experts exhibit.
Diffusion policies, on the other hand, are adept at capturing continuous action distributions, but face two critical challenges in control tasks. First, real-time applications like robotics demand fast, efficient inference, yet diffusion models rely on iterative sampling, requiring dozens of forward passes    to generate actions. This makes them orders-of-magnitude slower than autoregressive or feedforward policies [26]. Second, the downstream task of imitation learning is often reinforcement learning [43,39,58,17], which typically requires exact action likelihoods for tasks like exploration and off-policy corrections [49,1,27,56]. Diffusion models, however, generate samples by integrating (stochastic) differential equations [48,47], making likelihood estimation computationally expensive.
To address these limitations, we propose the use of a Generative Infinite Vocabulary Transformer (GIVT) [50] that models the policy as a Gaussian Mixture Model (GMM) on top of a decoder-only transformer architecture [53,41]. This approach enables modeling continuous actions as "tokens" by directly parameterizing the components of the GMM in each time step, avoiding the need for discretization and thus preserving the inherent geometry of the action space. This choice allows us to exploit the benefits of large-scale sequence modeling, such as autoregressive policy generation and access to explicit likelihood estimates, while avoiding the pitfalls of artificially segmenting continuous control signals into discrete bins. In addition, we discuss two different sampling strategies that reduce variability in the generated trajectories.
In summary, our work:
• Eliminates the need for action quantization in autoregressive policy parameterization, simplifying behavioral cloning pipelines while retaining the ability to directly evaluate action likelihoods.
• Proposes two sampling algorithms to reduce the variance of the generated trajectories, including a global mode-finding algorithm.
• Demonstrates state-of-the-art performance on standard benchmarks, both for conditional and unconditional policy generation using proprioceptive and image-based states, confirming the effectiveness of our method for multimodal policy modeling.
The implementation is available at https://github.com/ziyadsheeba/qfat.
this section cite: ['b32', 'b58', 'b27', 'b25', 'b44', 'b6', 'b53', 'b39', 'b25', 'b44', 'b51', 'b23', 'b19', 'b17', 'b30', 'b25', 'b42', 'b38', 'b57', 'b16', 'b48', 'b0', 'b26', 'b55', 'b47', 'b46', 'b49', 'b52', 'b40']

Section: Sequence to Action Prediction
We consider the following standard setting for imitation learning of simulated robotics tasks. An agent equipped with continuous controls A evolves in a continuous state space S in discrete time. In behavioral cloning, we assume that a so-called behavior policy π b collected a dataset D = {τ i } N -1 i=0 of N trajectories in the environment, where each trajectory τ i = {s i 0 , a i 0 , . . . , a i n i -1 , s i n i } is a sequence of states s i t ∈ S and actions a i t ∈ A, with n i denoting the length of trajectory i. Note that we do not impose a Markovian assumption on the behavior policy. In general a i t might depend on a history of states {s i j } t j=t-h , where h is a fixed window. Our objective is to learn a sequence-to-action predictor, or imitating policy π θ . We treat the objective as a supervised learning problem, where the parameter θ of our policy is optimized to minimize the negative log-likelihood with respect to π b , E τ ∼π b [-log(π θ )(τ )]. To do so, we minimize:
L(θ) = - 1 N N -1 i=0 n i -1 t=0 log π θ (a i t | s i t-h:t ) .(1)
Note that a standard result in variational inference [15] shows that this objective is equivalent to minimizing the Kullback-Leibler (KL) divergence between the behavioral policy π b and the learned policy π θ , implying that the learned policy π θ should match, or clone, the behavior of π b . However, the choice of parametrization is crucial to guarantee a good approximation of the behavior policy.
this section cite: ['b14']

Section: The Q-FAT Algorithm
Current transformer-based policy parametrization approaches [26,45,32] consist of first discretizing the action space, e.g. using a VQ-VAE [52,57], to comply with the standard transformer architecture requiring a discrete vocabulary. Inspired by the success of GIVT [50] in the image generation domain, our model Quantization-Free Autoregressive Action Transformer (Q-FAT) adapts GIVT to the continuous action behavioral cloning setting, modeling the action distribution at each step by predicting the parameters of a multivariate Gaussian mixture model (GMM). Namely, Q-FAT models the probability of an action a t ∈ R m at step t as
π θ a t s t-1 , . . . , s t-h s ) = k i=1 π t i N a t µ t i , σ t i ,(2)
where π t i , µ t i , and σ t i are produced by the transformer decoder [53,41] at step t, conditioned on a history of previous states {s i } t-1 i=t-h s with s i ∈ R d . Using a state history of length h s instead of only using the most recent state accounts for the potential non-Markovian structure in the demonstration dataset. Rather than predicting the full covariance Σ t i per mixture component, we assume a diagonal Gaussian distribution and only predict the diagonal entries of the covariance matrix σ t i . While this simplifying assumption enables efficient likelihood computations, it does not impose independence across action dimensions in general. Q-FAT is trained using standard teacher forcing with a causal attention mask. We highlight that we only use state feedback and do not condition on the predicted actions. We discuss the important architecture decisions in Appendix D.
this section cite: ['b25', 'b44', 'b31', 'b51', 'b56', 'b49', 'b52', 'b40']

Section: Goal Conditioning
For some tasks such as tracking, a near-future goal can be appended to the input to condition the resulting policy. The conditioning is formalized by defining a set of goal states G ⊆ S. We then learn a goal-conditioned policy π θ a t s t-1 , . . . , s t-h s , g where g ⊆ G. In practice, we define g as a sequence of future states to be reached, i.e., g = {s i } t+h g i=t , where the goal states are typically generated from a high-level planner that plans instrumental goal states as done in Gupta et al. [11]. In our experiments, we simulate goal states from the demonstration dataset.
this section cite: ['b10']

Section: Action Sampling from a k-GMM
QFAT samples the next action from the conditional distribution, a t ∼ π θ (•|s t-1 , . . . , s t-h s , g), which is a GMM with k components. In practice, the model can be trained to output a short sequence of actions that are then executed in an open-loop fashion. This is done to ensure action consistency and avoid over-fitting to idle actions [7]. For simplicity, we describe only the next-action generation setting.
One can directly sample from the GMM that parametrizes the policy. For the vanilla GMM sampling, the procedure is:
1. Sample a mixture index k from the categorical distribution defined by the probabilities {π i } k i=1 . 2. Sample an action a from the corresponding Gaussian component N µ i , σ 2 i .
While this approach directly samples from the learned policy distribution, it can degrade performance in practice when the dataset is noisy. In that case, the GMM may overfit the data distribution and produce noisy actions, resulting in jitter or unstable trajectories. We discuss below how Q-FAT can be easily equipped with stabilizing sampling techniques that directly exploit the fitted GMM distribution.
this section cite: ['b6']

Section: Stabilizing Output Trajectories
When controlling real-world robotics systems, it is important to generate action sequences without jitter, as high frequency inputs may damage actuators and excite undesirable structural modes [2].
Thanks to the richness of the GMM representation of the policy, we can directly design stabilization patches.
this section cite: ['b1']

Section: Down-scaling the Mixture Variances
A common heuristic to mitigate noisy samples is to adjust the sampling temperature, which in our GMM parametrization is equivalent to down-scaling the variance in each Gaussian component [29,16,50,51]. Concretely, one replaces each variance σ 2 i by α σ 2 i , for some small α ∈ (0, 1)-sometimes by several orders of magnitude (α ≪ 1). This compresses each component, forcing sampled actions to concentrate more tightly around each component's mean. However, merely shrinking the variances does not address the large-scale spread when mixture components are far apart. The inter-component variance remains, and noisy samples may still arise from small but non-negligible transitions among disparate components. This phenomenon is exacerbated when the choice of the number of components k is overspecified (see Figure 2 and further discussion in Appendix A). Furthermore, it has been shown by Carreira-Perpinan [5] that GMMs in dimension d ≥ 2 may have a mode that does not directly align with the mean of any component. In such a situation, the down-scaling technique is likely to result in an insidious loss of the extra mode (see Figure 2).
this section cite: ['b28', 'b15', 'b49', 'b50', 'b4']

Section: Mode-tracking via the Mean-Shift Algorithm
We explored a principled way to reduce the variance of sampling from multimodal distributions by explicitly identifying the major modes and then sampling from those modes with an appropriate probability while controlling variance. An effective way to find the modes in a GMM is via the mean-shift algorithm [5], which is an efficient fixed-point iteration algorithm (see Appendix B for further details). We propose a multinomial distribution over the modesfoot_0 {m 1 , ...m J } that respects the shape of the density:
Pr(m j ) = w j j ′ w j ′ ,(3)
where
w j = p GMM m j • -H j -1 2(4)
and
H j = ∇ 2 log p GMM (m j ) .(5)
Note that each H j admits a closed-form expression and can be computed in O(km 2 ) operations. One can then further inject noise to the sampled mode using a Gaussian noise with a fixed variance or use the Hessian around the mode to account for the curvature of the distribution to determine the variance (see more details on this process in Appendix B). We emphasize that this mode-tracking algorithm can be efficiently implemented on GPUs using common deep learning libraries and introduces a minor overhead compared to the transformer forward pass.
this section cite: ['b4']

Section: Experiments
We evaluate the performance of Q-FAT through extensive experiments across 9 different tasks using five representative simulated robotics environments: PushT [7], Kitchen [11], UR3 BlockPush, BlockPush [10] and Multimodal Ant [26]. These environments cover the most common control signals in practice, namely position and velocity of robot's end effector and robot joint angles. A detailed description of each environment is provided in Appendix C. We further ran preliminary experiments on an autonomous driving dataset (nuScene [4]) and we add the experimental results in Appendix E. Our experiments are designed to address the following key research questions:
1. Is Q-FAT competitive with state-of-the-art methods in both conditional and unconditional behavior imitation? 2. Can Q-FAT effectively capture the diversity of demonstration datasets while avoiding mode collapse? 3. What is the effect of different sampling techniques on the performance of Q-FAT? 4. How sensitive is Q-FAT's performance to the choice of the mixture components k in the learned policy? 5. How does the inference speed of Q-FAT compare to VQ-BeT?
We find that Q-FAT achieves state-of-the-art performance across the majority of conditional and unconditional tasks, achieving state-of-the-art balance between the quality and diversity of the generated actions. We report the results of Q-FAT using the variance reduced sampling (Section 3.2) with a factor of 10 -6 and we discuss the implications of this choice in Section 4.3. Further training details are provided in Appendix D. A summary of our results is presented in Table 1.
this section cite: ['b6', 'b10', 'b9', 'b25', 'b3']

Section: Baselines and Performance Metrics
Baselines We compare Q-FAT against a range of state-of-the-art baselines. We restrict our focus to comparing against diffusion and transformer-based policy learning methods. For unconditional tasks, the baselines include: (1) BeT [45], which performs action discretization using k-means clustering and models the action distribution with a transformer decoder; (2) VQ-BeT [26], an extension of BeT that leverages vector quantization via a hierarchical Variational Autoencoder (VQ-VAE) [57] for action discretization; and (3) two variants of Diffusion Policy [7], namely a convolutional-based implementation and a transformer-based implementation. For goal-conditional tasks, we extend the comparison to include conditional versions of BeT and VQ-BeT, as well as two additional baselines: BESO [42], a denoising diffusion-based method with a conditional variant (C-BESO) and a classifier-free guided variant (CFG-BESO).
this section cite: ['b44', 'b25', 'b56', 'b6', 'b41']

Section: Quality and Diversity of Q-FAT's Actions
Evaluation Metrics We assess performance using task success rate and induced behavioral entropy. Task success is measured differently per environment: in Kitchen, Multimodal Ant and UR3 Block-Table 1: Performance comparison between unconditional and conditional behaviors on the task success rates. We report the performance of Q-FAT with variance down-scaling with a factor of 10 -6 .
-Unconditional Tasks -Environment Metric BeT DiffPolicy-C DiffPolicy-T VQ-BeT Q-FAT PushT Final IoU (•/1) 0.39 0.73 0.74 0.78 0.80 Image PushT 0.01 0.66 0.45 0.68 0.69 Kitchen Goals (•/4) 3.07 2.62 3.44 3.66 3.84 Image Kitchen 2.48 3.11 3.01 2.98 3.15 Multimodal Ant 2.73 3.12 2.90 3.22 3.30 UR3 BlockPush Goals (•/2) 1.59 1.83 1.82 1.84 1.99 BlockPush 1.67 0.47 1.93 1.79 1.77 -Conditional Tasks -Environment Metric C-BeT C-BESO CFG-BESO VQ-BeT Q-FAT Cond. Kitchen Goals (•/4) 0.15 3.75 3.47 3.78 3.78 Cond. UR3 BlockPush Goals (•/2) 0.00 1.14 0.92 1.94 1.96 2.6 2.8 3.0 3.2 3.4 3.6 3.8 4.0 4.2 p4-Entropy 3.07 2.62 3.44 3.66 4.06 Kitchen 2.50 2.75 3.00 3.25 3.50 3.75 4.00 4.25 4.50 p4-Entropy 2.73 3.12 2.90 3.22 4.33 Multimodal Ant 0.90 0.92 0.94 0.96 0.98 1.00 p2-Entropy 0.99 0.91 0.98 0.99 0.99 UR3 BlockPush 2.4 2.6 2.8 3.0 3.2 3.4 3.6 p4-Entropy 2.48 3.11 3.01 2.98 3.44 Image Kitchen 1.90 1.92 1.94 1.96 1.98 2.00 2.02 p2-Entropy 1.95 1.94 1.95 1.99 1.99 BlockPush BeT DiffusionPolicy-C DiffusionPolicy-T VQ-BeT Q-FAT (Us) Push, the task success rate corresponds to the number of goals achieved per roll-out, while in PushT, task success is quantified using the final Intersection over Union (IoU) between the object's position and the target area. Behavioral diversity is measured by computing task completion sequence entropy (length 4 for Kitchen and Multimodal Ant, length 2 for UR3 BlockPush), reflecting variability in task completion. Baseline performance follows metrics from Lee et al. [26], and results are reported as the average success rate of the best checkpoint over 1000 episodes, including behavioral entropy where applicable.
Comparison to Baselines Q-FAT achieves state-of-the-art performance on the success metrics, as shown in Table 1. The performance gap observed with BeT can be attributed to the limitations of k-means clustering in high-dimensional spaces, which may hinder its ability to effectively capture complex action distributions. This is particularly exacerbated due to BeT attempting to discretize the whole unconditional action space, and not the conditional one. In the case of VQ-BeT, the sequential nature of first predicting a discrete action token followed by an action correction step introduces two potential sources of error: inaccuracies in token prediction and errors in the subsequent correction mechanism. This places significant reliance on the vector quantization network, which is challenging to optimize due to the need to discretize the entire action space and the required two-stage training process -first for the quantization network and then for the policy. For diffusion-based policies, the absence of an explicit likelihood function makes it difficult to control the variance of sampled actions, particularly in the presence of noisy data. This can be particularly problematic in fine-grained control tasks, where precise action execution is crucial for success. Consequently, while these approaches demonstrate strong performance in certain scenarios, they may struggle in tasks requiring high accuracy and reliability under noisy data. Q-FAT largely outperforms or matches the behavioral entropy of all baselines (Figure 3), which assesses diversity across the generated motion.
this section cite: ['b25']

Section: Effects of Sampling Techniques
We assessed the effect of the different sampling techniques discussed in Section 3.2. We found that down-scaling the component variances had a 7% and a 6.5% increase in the success performance metrics in the Kitchen environment and PushT, respectively, compared to the GMM sampling, while only seeing a marginal 2% drop in the diversity of the generated behaviors in the Kitchen environment.
The modes sampling matched the performance of the variance down-scaling on the evaluation metrics. This could be attributed to the fact that the learned mixture components are sufficiently distant in the environments we explored, thus the effects discussed in Section 3.2 are not pronounced.
Despite the quantitative metrics not showing significant differences between the variance downscaling method and the mode sampling, we visualize the trajectories produced by both sampling techniques on a toy multi-route environment introduced by Shafiullah et al. [45] and see the effects of the irreducible variance in Figure 4. We further display a similar effect in the PushT environment in Figure 5.
this section cite: ['b44']

Section: Effect of the Number of Mixtures
We systematically evaluated the effect of varying the number of mixture components k in Q-FAT, where k controls the number of modes in the conditional action distribution for a given state context.
Specifically, we considered k ∈ {2, 4, 8, 16} in two benchmark domains, the Kitchen and UR3 BlockPush environments. Our results indicate that Q-FAT's success performance is highly robust across these values of k, with negligible degradation in performance. In the Kitchen environment, we observed an 18% increase in behavioral entropy when k was increased from 2 to 4, after which entropy remained approximately constant as k continued to increase. In contrast, in the UR3 BlockPush environment, the behavioral entropy remained stable regardless of k. The key to achieving this robustness is careful initialization.
In particular, we found it important to initialize each mixture component's mean to be distant from the others. Concretely, we placed these means on a hyper-cube spanning the range [-1, 1] per dimension, maximizing their mutual separation, and assigned each component a unit variance. The range [-1, 1] is a reasonable choice, as actions are typically bounded in practice and normalized in our experiments.
To investigate how the mixture components were utilized during inference, we tracked the average number of active components over the course of training (see Figure 6, left). We observed that the model effectively prunes redundant mixture components as training progresses, with the pruning effect becoming more pronounced when more components are initially employed. Interestingly, in the Kitchen environment, which has a nine dimensional action space, less pruning occurs as k increases compared to the UR3 environment, whose action space is only two dimensional. A plausible explanation for this discrepancy lies in the dimensionality of the action space.
In higher dimensions, each Gaussian component covers a proportionally smaller volume of the action space. Consequently, the model requires more components to adequately represent the conditional action distribution, and therefore prunes fewer mixture components. In lower-dimensional environments, by contrast, each component can cover a larger fraction of the data, which makes aggressive pruning more viable.
We further plotted the distribution over the number of active modes for the Kitchen and the UR3 BlockPush environments. We observed that approximately 70% of the course of a trajectory is uni-modal (Figure 6, right). This shows that multi-modality is required only a minority of the time during a rollout. This observation could be useful for future work to exploit this uni-modality computationally.
this section cite: []

Section: Inference Time Analysis
To compare computational efficiency, we evaluated the inference times of Q-FAT and VQ-BeT on a 16 GB MacBook Pro CPU. The analysis utilized models trained in the Kitchen environment, with hyperparameters for Q-FAT specified in Table 3 and for VQ-BeT as described in [26]. Q-FAT is approximately 2x faster on the tested hardware due to its efficient action decoding head. The breakdown of inference time per model component is detailed in Table 2. The primary performance difference arises from the action decoding head. While both models use an identical backbone architecture, VQ-BeT's reliance on two separate 2-layer MLPs for action decoding creates a computational bottleneck. In contrast, Q-FAT's use of a single, efficient linear action decoding layer results in a significant speedup.
this section cite: ['b25']

Section: Related Work
Offline Learning for Decision Making Learning to act from offline data has received significant attention over the years due to its vast potential in practical applications. This line of research can be divided into two categories: offline reinforcement learning [25,27,22,6] and imitation learning [34,37,38,13]. In offline reinforcement learning, on the one hand, the goal is to learn a policy from demonstrations with suboptimal behaviors, which necessitates the access to rewards in the demonstration dataset. Imitation learning, on the other hand, assumes that the trajectories are collected from near optimal behaviors, avoiding access to reward signals in the collected datasets. This increases the attractiveness of imitation learning in practical setups, since reward signals are hard to define [9,23]. Q-FAT falls under the framework of behavioral cloning, which is a form of imitation learning where the policy learning is treated as a supervised learning problem.
Generative Models for Cloning Behavior Complex generative models have been employed to capture the full data diversity due to the complex and multimodal behavior in human and robot datasets. Earlier work has explored the use of Gaussian processes [46], energy-based models [10] and generative adversarial networks [21] as policy parametrizations. More recently, more attention has been directed towards more powerful models, such as transformers [45,26,32,55] and diffusion models [7,54,40,36]. Unlike other transformer-based policies, Q-FAT uses a continuous parametrization of the output by predicting a GMM. Note that this idea has been explored before using LSTM backbones [29,45], however, it has been shown to produce poor performance [45,26], which we attribute to the instability of training LSTM networks and their limited capacity to model long sequences [14].
this section cite: ['b24', 'b26', 'b21', 'b5', 'b33', 'b36', 'b37', 'b12', 'b8', 'b22', 'b45', 'b9', 'b20', 'b44', 'b25', 'b31', 'b54', 'b6', 'b53', 'b39', 'b35', 'b28', 'b44', 'b44', 'b25', 'b13']

Section: Discussion and Future Work
In this work, we introduced Q-FAT, a quantization-free action transformer that achieves state-of-theart performance across various simulated robotics environments, opening new research directions in both behavioral modeling and reinforcement learning. Compared to prior work [26], Q-FAT overcomes the challenging non-differentiable step in action discretization while maintaining or improving performance.
By integrating Q-FAT into end-to-end policy learning pipelines, future research could evaluate its influence on sample efficiency and final task performance. Another promising area involves incorporating Bayesian priors into Q-FAT's action distribution estimates, which could facilitate active exploration in complex, high-dimensional settings. For fine manipulation tasks, researchers could further investigate coarse-to-fine sampling strategies based on Gaussian mixture model representations, potentially improving both exploration breadth and control. Finally, extending Q-FAT to non-Euclidean action spaces, such as those with Riemannian geometry, may enable more accurate and natural representations for tasks like legged locomotion or dexterous manipulation.
this section cite: ['b25']

Section: Limitations
While we have demonstrated that Q-FAT is effective in learning complex action distributions, the underlying GMM loss function used to train the policy assumes a Euclidean action space. In some environments (e.g., humanoid robots), the action space has a Riemannian geometry, which could potentially result in difficulty during learning. This can be counteracted by mapping actions into a latent Euclidean space using an action autoencoder [19] and perform the GMM loss in the latent space. However, unlike previous work that uses discrete action encoders [26,45], Q-FAT allows the flexibility of using continuous latent spaces, allowing end-to-end differentiability of the joint encoder and policy models. While this is an exciting research direction, we leave this for future work.
this section cite: ['b18', 'b25', 'b44']

Section: Impact Statement
Our work tackles simulated continuous control tasks whose downstream consequences are important for robotics. The probabilistic nature of our method may impact the safety of the generated behavior and should be further studied and tested. However, unlike prior work, our approach enables uncertainty estimates directly in the action space, which have the potential to improve safety by providing uncertainty quantification.
this section cite: []

Section: A GMM Properties

this section cite: []

Section: GMM Moments
Let X be a random variable from a k-mixture GMM distribution in R d . The probability density function can then be written as:
p(x) = k i=1 π i N (x | µ i , Σ i ) ,(6)
where N (x | µ i , Σ i ) is the multivariate Gaussian distribution for the i-th component. The mean and the covariance of X are given by:
E[X] = k i=1 π i µ i ,(7)
Cov[X] = k i=1 π i Σ i + (µ i -E[X])(µ i -E[X]) ⊤ ,(8)
where π i are the mixing coefficients, and µ i , Σ i are the mean and covariance parameters for the i-th Gaussian component. Note that while the mean of the GMM is a weighted sum of the component means, the covariance is not. The total covariance reflects both the inherent variability within each component and the variability introduced by the differences in component means.
this section cite: []

Section: B GMM Mode Finding and Sampling

this section cite: []

Section: Mode Finding
To find the modes of the GMM, we compute the gradient of the probability density function with respect to x and equate it to zero:
∇p(x) = k i=1 π i ∇N (x | µ i , Σ i ) = 0 ,(9)
where the gradient of the multivariate Gaussian distribution is given by
∇N (x | µ i , Σ i ) = N (x | µ i , Σ i )Σ -1 i (µ i -x) .(10)
We obtain therefore:
k i=1 π i N (x | µ i , Σ i )Σ -1 i (µ i -x) = 0.(11)
We identify the modes of the GMM by computing all critical points x * that satisfy (11). Moreover, to ensure that a critical point x * is indeed a mode, the Hessian of the density function at that point must be negative definite. The Hessian of the GMM density is given by
∇ 2 p(x) = k i=1 π i ∇ 2 N (x | µ i , Σ i ),(12)
where the Hessian of the multivariate Gaussian distribution is:
∇ 2 N (x | µ i , Σ i ) = N (x | µ i , Σ i ) Σ -1 i (µ i -x)(µ i -x) ⊤ Σ -1 i -Σ -1 i .(13)
Therefore, the Hessian of the GMM density becomes
∇ 2 p(x) = k i=1 π i N (x | µ i , Σ i ) Σ -1 i (µ i -x)(µ i -x) ⊤ Σ -1 i -Σ -1 i ,(14)
and a critical point x * is a mode if: ∇p(x * ) = 0 and ∇ 2 p(x * ) is negative definite.
While GMMs are often parameterized by a finite number of components, the resulting distribution can exhibit a richer structure, particularly in higher-dimensional spaces. Specifically, in dimensions greater than two (d > 2), a GMM can possess more modes (local maxima) than the number of its constituent components [5]. This phenomenon arises due to the interplay between the covariance structures and the relative positions of the Gaussian components. In higher dimensions, the overlapping regions of different components can create multiple peaks in the probability density function that cannot be attributed to individual components.
this section cite: ['b10', 'b4']

Section: Mean-Shift
We start by solving (11) for x as follows:
x = k i=1 π i N x µ i , Σ i Σ -1 i -1 k i=1 π i N x µ i , Σ i Σ -1 i µ i .(15)
To obtain the mean-shift update in practice, one can treat the solution to ( 15) as a fixed-point iteration.
More precisely, we define the operator T : R d → R d by
T (x) = k i=1 π i N x µ i , Σ i Σ -1 i -1 k i=1 π i N x µ i , Σ i Σ -1 i µ i .(16)
Then, starting from an initial guess x (0) , the mean-shift procedure updates the estimate of x via t) .
x (t+1) = T x (
The fixed-point iteration has the following interpretation: each iteration shifts the current iterate after t iterations closer to the modes (high-density regions) of the underlying density defined by the mixture of Gaussians. In practice, it has been shown that initializing x (0) with the component means µ i results in fast convergence [5], which we also observe in our experiments. To account for the fact that the number of modes could exceed the number mixture components, we add extra initialization points from the convex-hull of the component centroids capturing the most important modes.
this section cite: ['b4']

Section: Mean-Shift with Diagonal-Covariance Components
An important special case arises when the covariance matrices Σ i are diagonal. In that scenario, let
Σ i = diag σ 2 i,1 , . . . , σ 2 i,d .
Since each Σ -1 i is also diagonal (with entries 1/σ 2 i,j along the diagonal), the vector and matrix operations inside T (x) decouple across dimensions. Hence the inverse of
k i=1 π i N (x | µ i , Σ i ) Σ -1 i
reduces to a diagonal matrix whose components can be computed efficiently. As a result, each coordinate of the updated point can be determined independently, making the mean-shift iteration particularly fast on modern hardware.
this section cite: []

Section: Modes Sampling
To sample the modes proportional to their coverage of the data support, we approximate the integral of the GMM density in the local neighborhood of each mode using a Laplace approximation and use the value of the integral around each mode to compute the relative weight of the mode. We summarize the full mode-finding and sampling procedure in Algorithm 1.
this section cite: []

Section: C Environment Details
In our experiments, we evaluated Q-FAT in three distinct environments: Kitchen, PushT, and UR3 BlockPush. In the following, we provide a brief description of each environment and its variants.
• Franka Kitchen: This environment involves a Franka Panda robotic arm with a nine dimensional action space, designed for multi-task manipulation in a simulated kitchen 22: Return: x j setting [11]. The dataset consists of 566 human-collected demonstrations, where each trajectory completes a subset of four out of seven possible tasks in varying orders. For the image-based version of the environment, we rendered the collected trajectories into 112x112 images.
• PushT: The PushT environment focuses on pushing a T-shaped block to a target position on a table using two dimensional end-effector velocity control [7]. The dataset includes 206 human-collected demonstrations.
• UR3 BlockPush: In this task, a UR3 robotic arm is tasked with moving two blocks to designated goal circles on a table [10]. The dataset exhibits multimodality, as the blocks can be moved in either order. In the unconditional setting, we evaluate whether both blocks reach their respective goals. In the conditional setting, the model is provided with the target positions of the blocks, and performance is assessed based on the order in which the blocks reach their goals.
• BlockPush: The goal of this task is moving two (red and blue) blocks two targets [10]. The blocks can be moved in either order and can be put into either of the targets. While this environment is similar to UR3 BlockPush, it exhibits more stochasticity due to the random initialization of the block positions, and the targets being randomly translated and rotated. The dataset contained 1000 demonstrations.
• Multimodal Ant: The Multimodal Ant environment [26] is a modification of the Ant environment introduced by Schulman et al. [44] where the goal of the ant is to visit four distinct locations placed on vertices of a square. The multimodality in the dataset arises from the different order the goals are visited. The dataset contains 600 human-collected trajectories
this section cite: ['b10', 'b6', 'b9', 'b9', 'b25', 'b43']

Section: D Training Details
For our experiments, we used the minGPT [3] backbone as the decoder-only transformer implementation. The hyperparameters used for each of the environments are detailed in Table 3.
this section cite: ['b2']

Section: D.1 Architectural Design Choices
The policy was trained using teacher forcing with a causal attention mask, as this provided a stronger learning signal compared to computing the loss only on the last action given a context of states. We found that not using teacher forcing and employing a full attention mask significantly degraded performance. We further experimented with feeding back the sampled actions into the model's input by concatenating states and actions into a single feature vector. This approach degraded policy performance in simulated environments, likely due to the model overfitting to the highly correlated previous actions in the context while ignoring state information. Injecting Gaussian noise into the previous actions was necessary to achieve reasonable performance in this setting; however, this provided no discernible benefit compared to a policy without action feedback in our experiments.
this section cite: []

Section: D.2 History Masking
During training, we observed instances where both validation and training losses decreased, yet the environment rewards declined. We attribute this phenomenon to causal confusion [8], where the model overfits the recent sequence of states. Instead of learning true causal relationships, the model relies on temporally correlated but non-causal patterns to predict future actions, leading to suboptimal performance. To mitigate this, we introduced history masking, randomly masking the context (excluding the current state) with a certain probability during training. Applying this technique in the Kitchen, Multimodal Ant, and BlockPush environments (with masking probabilities of 0.7, 0.3, and 0.3, respectively) led to an improved correlation between environment rewards and validation loss.
this section cite: ['b7']

Section: D.3 Image-based Environments
For Image PushT and Image Kitchen, we fine-tuned a ResNet18 encoder [12], extracting feature maps from the first four layers to construct low-level features [35]. These features were then spatially average-pooled and fed into the transformer with a dimensionality of 1024. For Image PushT, we applied data augmentation (color jitter, random cropping, and random gray-scaling with probability 0.5), which proved crucial for preserving the invariance of the pre-trained ResNet and preventing overfitting to our dataset. Conversely, for Image Kitchen, we found data augmentation deteriorated performance.
this section cite: ['b11', 'b34']

Section: D.4 Goal Conditioning
In goal-conditional tasks, we conditioned the model on a sequence of future states equal in length to the state history. During inference, we randomly sampled a reference trajectory from the dataset. The agent's reward was computed only if it achieved the goal sequence in the same order as the reference trajectory.
this section cite: []

Section: D.5 Evaluation
All training datasets were normalized using min-max scaling to ensure state and action features lie within the range [-1, 1]. Following Lee et al. [26], Shafiullah et al. [45], we split the data into 95% for training and 5% for validation. The policy was evaluated periodically during training using a small number of environment roll-outs (20 to 50). The models achieving the lowest validation loss and highest success metrics were selected for reporting.
this section cite: ['b25', 'b44']

Section: D.6 Model Size
For a fair comparison with VQ-BeT [26], we ensured our models have a comparable (or smaller) number of trainable parameters, on the order of 10 5 -10 6 . We slightly deviated from their transformer encoder hyperparameters to account for the additional capacity utilized by their hierarchical vector quantization autoencoder for action discretization. Training hyperparameters, such as batch size and learning rates, were adjusted to accommodate the differences in loss functions between the methods. Further details are provided in Table 3 and Table 13 in Lee et al. [26].
this section cite: ['b25', 'b25']

Section: D.7 Hardware
Experiments were conducted on a heterogeneous cluster, making precise hardware control infeasible. However, all experiments were run on a single desktop-grade GPU with at most 32 GB of memory. Training a single model typically took 4-8 hours, depending on dataset size and the frequency of environment evaluations for validation.
this section cite: []

Section: E nuScenes Experiment
To evaluate our model's applicability beyond robotic manipulation, we use the nuScenes [4] dataset, a large-scale benchmark for autonomous driving. We utilize the object-centric, preprocessed version of the dataset from Mao et al. [30], and follow the exact preprocessing and tokenization steps for driving data as detailed in the VQ-BeT paper [26]. The model's input is a set of tokens representing the driving mission (e.g., turn left), the ego-vehicle's current state (velocity, acceleration), its recent trajectory history, and the state of surrounding objects (position, class). The task is to predict the ego-vehicle's trajectory over the next six timesteps. Success is evaluated using two primary metrics: the average L2 distance between the predicted and ground-truth trajectories to measure accuracy, and the collision rate to assess the safety of the generated path.
For this sequential prediction task, we found it necessary to adapt our model's architecture. A naive approach of predicting a single high-dimensional vector concatenating all future waypoints resulted in degraded performance. This is likely because the model is forced to learn a direct mapping to a highly complex and multimodal joint distribution of future states. Such a method struggles to enforce temporal consistency, often producing kinematically implausible trajectories. Consequently, we adopted an autoregressive prediction scheme, where the model forecasts one waypoint at a time, conditioned on its previous predictions. This approach simplifies the learning problem by factorizing the joint distribution into a sequence of more manageable conditional distributions. By doing so, the model implicitly learns the transition dynamics of the environment, ensuring that the generated trajectory is temporally coherent and physically plausible. Furthermore, instead of predicting the absolute coordinates of each waypoint, our model predicts the delta, or displacement, from the previous waypoint. This encourages the model to learn a policy that is invariant to the absolute frame of reference, which has been shown to improve generalization.
The results from these initial experiments are presented in Table 4. These findings show that Q-FAT achieves performance competitive with the VQ-BeT baseline, demonstrating that our method is effective in this challenging domain. The training hyperparameters can be found in Table 3.
this section cite: ['b3', 'b29', 'b25']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We introduce Q-FAT and evaluate it extensively on common bench marks, demonstrating SOTA performance.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: See Section 7.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: We do not present any theoretical claims.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: See Appendix D. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Our code and datasets will be made publicly available.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: See Appendix D. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: It was computationally expensive to train multiple models for each environment to assess the modeling uncertainty. As for the aleatoric uncertainty in the environments, it was negligible since we ran evaluations on 1000 episodes and averaged the results. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: See Appendix D. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: We will open source our code with public licenses. We further clearly stated the societal impact in Section 8. Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: See Section 8. Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11.
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: We train specialized models for simulated robotics tasks, which does not require safeguards.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: We ensured that all the data and code that was used was made publicly available for research use.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: We will open source our code.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: Our work does not include human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: Our work does not include human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: References
Ref_id:b0 Title: Maximum a posteriori policy optimisation Year: (2018)
Ref_id:b1 Title: PID controllers: theory, design, and tuning Year: (1995)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b4 Title: Mode-finding for mixtures of Gaussian distributions Year: (2000)
Ref_id:b5 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b6 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b7 Title: Causal confusion in imitation learning Year: (2019)
Ref_id:b8 Title: Explicable reward design for reinforcement learning agents Year: (2021)
Ref_id:b9 Title: Implicit behavioral cloning Year: (2022)
Ref_id:b10 Title: Relay policy learning: Solving long-horizon tasks via imitation and reinforcement learning Year: (2019)
Ref_id:b11 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b12 Title: Generative adversarial imitation learning Year: (2016)
Ref_id:b13 Title: Long short-term memory Year: (1997)
Ref_id:b14 Title: Stochastic variational inference Year: (2013)
Ref_id:b15 Title: A research framework for distributed reinforcement learning Year: (2020)
Ref_id:b16 Title: Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning fine-tuning Year: (2024)
Ref_id:b17 Title: Straightening out the straightthrough estimator: Overcoming optimization challenges in vector quantized networks Year: (2023)
Ref_id:b18 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b19 Title: Uvim: A unified modeling approach for vision with learned guiding codes Year: (2022)
Ref_id:b20 Title: Imitating driver behavior with generative adversarial networks Year: (2017)
Ref_id:b21 Title: Conservative q-learning for offline reinforcement learning Year: (2020)
Ref_id:b22 Title: When should we prefer offline reinforcement learning over behavioral cloning Year: (2022)
Ref_id:b23 Title: Robust training of vector quantized bottleneck models Year: (2020)
Ref_id:b24 Title: Batch reinforcement learning Year: (2012)
Ref_id:b25 Title: Nur Muhammad Mahi Shafiullah, and Lerrel Pinto. Behavior generation with latent actions Year: (2024)
Ref_id:b26 Title: Offline reinforcement learning: Tutorial, review, and perspectives on open problems Year: (2020)
Ref_id:b27 Title: Reinforcement learning with model-based feedforward inputs for robotic table tennis Year: (2023)
Ref_id:b28 Title: What matters in learning from offline human demonstrations for robot manipulation Year: (2021)
Ref_id:b29 Title: Gpt-driver: Learning to drive with gpt Year: (2023)
Ref_id:b30 Title: Finite scalar quantization: VQ-VAE made simple Year: (2024)
Ref_id:b31 Title: Quest: Selfsupervised skill abstractions for learning continuous control Year: (2024)
Ref_id:b32 Title: Nonlinear analysis and control of a reactionwheel-based 3-d inverted pendulum Year: (2017)
Ref_id:b33 Title: The surprising effectiveness of representation learning for visual imitation Year: (2021)
Ref_id:b34 Title: The unsurprising effectiveness of pre-trained vision models for control Year: (2022)
Ref_id:b35 Title: Imitating human behaviour with diffusion models Year: (2023)
Ref_id:b36 Title: Deepmimic: Exampleguided deep reinforcement learning of physics-based character skills Year: (2018)
Ref_id:b37 Title: Amp: Adversarial motion priors for stylized physics-based character control Year: (2021)
Ref_id:b38 Title: Reinforcement learning of motor skills with policy gradients Year: (2008)
Ref_id:b39 Title: Learning a diffusion model policy from rewards via q-score matching Year: (2023)
Ref_id:b40 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b41 Title: Goal-conditioned imitation learning using score-based diffusion policies Year: (2023)
Ref_id:b42 Title: Learning from demonstration Year: (1996)
Ref_id:b43 Title: Highdimensional continuous control using generalized advantage estimation Year: (2015)
Ref_id:b44 Title: Behavior transformers: Cloning k modes with one stone Year: (2022)
Ref_id:b45 Title: Robotic imitation from human motion capture using Gaussian processes Year: (2005)
Ref_id:b46 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b47 Title:  Year: (2023)
Ref_id:b48 Title: Policy gradient methods for reinforcement learning with function approximation Year: (1999)
Ref_id:b49 Title: Givt: Generative infinite-vocabulary transformers Year: (2024)
Ref_id:b50 Title: Jetformer: An autoregressive generative model of raw images and text Year: (2025)
Ref_id:b51 Title: Neural discrete representation learning Year: (2017)
Ref_id:b52 Title: Attention is all you need Year: (2017)
Ref_id:b53 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2022)
Ref_id:b54 Title: Unleashing large-scale video generative pre-training for visual robot manipulation Year: (2023)
Ref_id:b55 Title: Behavior regularized offline reinforcement learning Year: (2019)
Ref_id:b56 Title: Soundstream: An end-to-end neural audio codec Year: (2021)
Ref_id:b57 Title: Reinforcement and imitation learning for diverse visuomotor skills Year: (2018)
Ref_id:b58 Title: Dynamic electromagnetic navigation Year: (2025)
