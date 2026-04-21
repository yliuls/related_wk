Title: Trust Region Constrained Measure Transport in Path Space for Stochastic Optimal Control and Inference
Abstract: Solving stochastic optimal control problems with quadratic control costs can be viewed as approximating a target path space measure, e.g. via gradient-based optimization. In practice, however, this optimization is challenging in particular if the target measure differs substantially from the prior. In this work, we therefore approach the problem by iteratively solving constrained problems incorporating trust regions that aim for approaching the target measure gradually in a systematic way. It turns out that this trust region based strategy can be understood as a geometric annealing from the prior to the target measure, where, however, the incorporated trust regions lead to a principled and educated way of choosing the time steps in the annealing path. We demonstrate in multiple optimal control applications that our novel method can improve performance significantly, including tasks in diffusion-based sampling, transition path sampling, and fine-tuning of diffusion models.

Section: Introduction
Even though the theory of stochastic optimal control (SOC) dates back several decades [12,49], it has recently attracted renewed interest within the machine learning community. Building on novel formulations that are well-suited for gradient-based optimization (see [40] for an overview) and drawing connections to diffusion models [15,36,92], recent work has led to significant progress in the numerical approximation of high-dimensional control problems using neural networks [42,87]. Related problems are crucial in many practical applications, ranging from sampling problems (e.g., in statistical physics [48,68], Bayesian statistics [54,85], and reinforcement learning [24]) to finetuning of diffusion models [38,41,132]. In this work, we aim to further advance SOC approximation methods by taking inspiration from trust region methods used in optimization [1,88,93,110,123], resulting in a principled framework from the perspective of measure transport in path space. Stochastic optimal control. SOC problems (with quadratic control costs) describe optimization problems of the form
where one optimizes the control u of the stochastic differential equation (SDE). Since the law of the SDE solution X u induces a so-called path measure P u on the space of continuous trajectories (specifying how likely a certain trajectory is), finding the optimal control is equivalent to finding an optimal target path space measure Q. From the SOC literature it is known that the likelihood of Q w.r.t. P u can be expressed in closed-form (see [34] and (3) below), which allows to minimize divergences 2 D(P u , Q) via gradient-based optimization (also termed iterative diffusion optimization).
this section cite: ['b11', 'b48', 'b39', 'b14', 'b35', 'b91', 'b41', 'b86', 'b47', 'b67', 'b53', 'b84', 'b23', 'b37', 'b40', 'b131', 'b0', 'b87', 'b92', 'b109', 'b122', 'b33']

Section: Trust region methods.
However, if the target Q is rather different from the initialization P u0
(typically the uncontrolled process with u 0 = 0), many algorithms face challenges with high variances or mode discovery when directly minimizing D(P u , Q), especially in high dimensions.
To this end, we propose to approach the target measure gradually by a sequence (P ui ) i , where in the i-th step we add the constraint D KL (P u |P ui-1 ) ≤ ε to the cost functional (1), with u i-1 being the approximated optimal control from the previous iteration and ε > 0 a chosen trust region bound. We prove that the intermediate measures P ui define a geometric annealing between the prior P u0 and target measure Q, where the annealing step-sizes are chosen optimally, in the sense of having an approximately constant change in Fisher-Rao distance (Props. 2.2 and 2.3). Finding an optimal annealing schedule is paramount for the convergence speed of many measure transport and sampling methods [119], and understanding physical processes [30,106]. While the direct computation of Fisher-Rao distances can be challenging, we show that trust region methods lead to a simple way of obtaining equidistant steps in an information-geometric sense. Moreover, we show that the Lagrangian of the constrained problem can be written as another SOC problem and that the optimal Lagrangian multiplier can be obtained via a dual optimization problem without additional computational overhead (Sec. 2.1). Finally, we adapt successful approaches based on SOC matching [41,42] and log-variance divergences [87] to the constrained SOC problem to get a practical algorithm (Sec. 2.2).
Applications. The resulting trust region stochastic optimal control method can be viewed as an extension of various existing algorithms, yielding significant improvements on a range of applications (Sec. 3). In particular, we consider (i) deep learning approaches to classical SOC problems (extending [42,87]) enabling the usage of cross-entropy losses in high dimensions, (ii) diffusionbased sampling from unnormalized densities (extending [97,129]) enabling efficient sampling from high-dimensional, multimodal densities with substantially fewer target evaluations, (iii) transition path sampling in molecular dynamics (extending [72,113]) yielding notably higher transition hit rates, and (iv) reward fine-tuning of text-to-image models (extending [41]) achieving comparable performance while requiring significantly fewer simulations.
this section cite: ['b118', 'b29', 'b105', 'b40', 'b41', 'b86', 'b41', 'b86', 'b96', 'b128', 'b71', 'b112', 'b40']

Section: Contributions.
Our contributions can be summarized as follows:
• We develop a general framework for solving measure transport with trust regions and apply it to SOC problems using iterative diffusion optimization.
• We prove that our framework leads to a sequence of SOC problems whose solutions define an equispaced annealing between initialization and optimum w.r.t. the Fisher-Rao distance.
• Relying on different loss functionals, we propose two practical instantiations of our framework and demonstrate state-of-the-art performance on a series of applications, ranging from sampling from unnormalized densities to transition path sampling and reward fine-tuning of text-to-image models.
Notation. We denote by U ⊂ C(R d × [0, T ]; R d ) the set of admissible controls and by P the set of all probability measures on C([0, T ], R d ). We define the path space measure P ∈ P as the law of a R d -valued stochastic process X = (X t ) t∈[0,T ] and we denote by P s the marginal distribution at time s. We refer to App. A for further details on our notation and assumptions.
2 Trust region constrained measure transport for optimal control
The idea of iterative diffusion optimization in optimal control based on path space measures is to consider loss functionals of the form
L(u) = D(P u , Q)(2)
and minimize them with gradient-descent algorithms [87]. The loss functional (2) yields implementable algorithms for SOC problems since the optimal path measure Q of (1) can be stated explicitly via the Radon-Nikodym derivative dQ dP (X) = e -W(X,0) Z(X0) with W(X, t) = T t f (Xs, s) ds + g(XT ),
where Z := E e -W(X,0) |X 0 and P is the path measure of the uncontrolled process X = X 0 ; see App. D. In this work, we extend this attempt by using trust regions that shall make sure that the optimization is conducted in a more "regulated" fashion, where the essential idea is to divide the global problem into smaller (reasonably chosen) chunks. We quantify this in Prop. 2.3 below. To this end, we consider the iterative optimization scheme defined by
ui+1 = arg min u∈U DKL (P u |Q) s.t. DKL(P u |P u i ) ≤ ε,(4)
for any i ∈ N, where ε > 0 defines a trust region w.r.t. to the previous control iterate and where we often set u 0 = 0 (and thus P u0 = P). This corresponds to dividing the overall optimization problem into parts according to their distance measured in the KL divergence between the respective preceding and succeeding path measures. Due to the convexity of the KL divergence, we can show that in all but the last step we actually have an equality constraint in (4); see App. E.1. Thus, there exists an I ∈ N such that u I = u * is the optimal control of the global control problem defined in (1).
Remark 2.1 (Controlling the variance of importance weights). The constraint D KL (P u |P ui ) ≤ ε can be motivated by the goal to control the variance of importance weights Var P u i (dP ui+1 /dP ui ), which can be explained by the inequality Var P
u i (dP ui+1 /dP ui ) ≥ e DKL(P u i+1 |P u i ) -1,
see, e.g., [60]. For small ε (which is a common choice in practice) we typically observe
Var P u i (dP ui+1 /dP ui ) ≈ 2ε
(see App. I.foot_2), which can be explained by a Taylor expansion and assuming that dP ui+1 /dP ui ≈ 1. Low variance of importance weights is directly related to efficiency of many measure transport methods and too high variance makes it practically impossible to obtain reliable results. Note also that the reverse KL divergence allows for explicit expressions for the resulting constrained problem (see Sec. 2.1) and we leave alternative divergences for future research. In practice, under suitable regularity assumptions, we can approach the above constrained optimization problem using a relaxed Lagrangian formalism. To this end, we consider the loss functionals
L (i) TR (u, λ) = DKL (P u |Q) + λ (DKL(P u |P u i ) -ε) , (5
)
where λ ≥ 0 is a Lagrange multiplier, and solve the saddle point problems
max λ≥0 min u∈U L (i) TR (u, λ).(6)
We note that L (i) TR is convex in u by convexity of the KL divergence (see App. E.1) and concave in λ since it can be expressed as the pointwise minimum min u L (i) TR (u, λ) among a family of linear functions of λ. Thus, (6) has unique optima which we denote by u i+1 and λ i , respectively. We can now show the following evolution of the optimal measures. Proposition 2.2 (Optimal change of measure as geometric annealing). Let Q be the optimal path measure defined in (3). The intermediate optimal path measures corresponding to (4) then satisfy 3
dP u i+1 dP u i ∝ dQ dP u i 1 1+λ i (7
)
and the optimal change of measure w.r.t. the base measure P is given byfoot_3
dP u i dP (X) ∝ dQ dP (X) β i dP u 0 dP (X) 1-β i with βi = 1 - i-1 j=0 λ j 1+λ j .(8)
Proof. The first statement follows by the definition of the Lagrangian and the second follows by induction; see App. B.
Note that the sequence (β i ) i is monotonically increasing with values in [0, 1], where we have β 0 = 0 and β I = 1 (as λ I-1 = 0 due to optimality). Thus, the formula in (8) can be seen as a geometric annealing from the prior to the target measure. Note that when u 0 = 0, the second factor vanishes. Importantly, the step-size of the annealing is automatically chosen such that we obtain a well-behaved sequence of distributions; see also
this section cite: ['b86', 'b59']

Section: Remark 2.4 (Trust regions for general measures).
The observant reader has likely noticed that so far all our arguments do not rely on the fact that we consider path space measures, but work for general probability measures. We could therefore as well write our trust region method stated in (4) as
Pi+1 = arg min P∈P DKL (P|Q) s.t. DKL(P|Pi) ≤ ε.(9)
We refer to App. H for a treatment when the measures admit densities on R d , which can, e.g., be considered for variational inference with normalizing flows.
this section cite: []

Section: Constrained stochastic optimal control
While the above formulation in principle works for arbitrary measures, in this work we focus on path space measures corresponding to optimal control problems. In this setting we can compute some of the objectives more explicitly and recover helpful relations.
Lagrangian as SOC problem. First, note that, using the Girsanov theorem (see App. A.2), it turns out that, for a fixed Lagrange multiplier λ, the Lagrangian in (5) defines another SOC problem, i.e.,
L (i) TR (u, λ) = L (i) TRC (u, λ) -λε,(10)
wherefoot_4
L (i) TRC (u, λ) = E T 0 1+λ 2 ∥u -λ 1+λ ui∥ 2 + λ 2(1+λ) ∥ui∥ 2 + f (X u s , s) ds + g(X u T ) + log Z(X0)(11)
and X u is still defined as in (1); see App. E.4 for details. Note that this cost functional is more general than the one stated in (1), which one recovers when setting λ = 0. We can show that the corresponding SOC problem satisfies the following optimality conditions. Proposition 2.5 (Optimality for trust region SOC problems). For fixed λ, let us define by
V λ i+1 (x, t) := inf u∈U E T t 1+λ 2 ∥u -λ 1+λ ui∥ 2 + λ 2(1+λ) ∥ui∥ 2 + f (X u s , s) ds + g(X u T ) Xt = x
the value function of the SOC problem inf u∈U L (i) TRC (u, λ) corresponding to (11) and by u λ i+1 its solution. Then it holds
(i) (Estimator for value function) V λ i+1 (x, t) = -(1 + λ) log E e -1 1+λ Wi(X u i ,t) X ui t = x , where Wi(X u i , t) = T t 1 2 ∥ui(X u i s , s)∥ 2 ds + T t ui(X u i s , s) • dWs + W(X u i , t).
this section cite: ['b0', 'b10']

Section: (ii) (Connection between solution and value function) It holds
u λ i+1 = λ 1+λ u i -1 1+λ σ ⊤ ∇V λ i+1 .
Proof. The statements can be proven using the verification theorem; see App. E.4 for details.
We note that Prop. 2.2, the Girsanov theorem, and (3) relate the functional W i in Prop. 2.5 to the importance weights
dQ dP u i (X u i ) ∝ e -W i (X u i ,0) and dP u i+1 dP u i (X u i ) ∝ e - 1 1+λ i W i (X u i ,0) . (12
)
Dual problem for Lagrange multiplier. Next, we will outline how to find the optimal Lagrange multiplier λ in (6) in the SOC setting. Plugging the optimal control u λ i+1 in the Lagrangian (10) yields the dual function
L (i) Dual ∈ C(R, R) given by L (i) Dual (λ) := L (i) TR (u λ i+1 , λ) = L (i) TRC (u λ i+1 ) -λε. (13
)
We note that evaluating the SOC problem in (11) at the optimal control can be expressed via the value function given in Prop. 2.5, which yields
L (i) Dual (λ) = E V λ i+1 (X u i 0 , 0) -λε = -(1 + λ)E log E e -1 1+λ W i (X u i ,0) X u i 0 -λε,(14)
Algorithm 1 Trust Region SOC with buffer (see App. E.2 for details) Require: Initial path measure P u 0 , target path measure Q, divergence D, termination threshold δ for i = 0, 1, . . . do Sample trajectories X ∼ P u i by integrating the SDE in (1) with Brownian motion W and control ui Compute importance weights w = dQ dP u i (X u i ) ∝ exp(-Wi(X u i , 0)) as in (12) Initialize buffer B = W, X, w Compute multiplier λi = arg max λ∈R + L (i) Dual (λ) as in (14) using B and a 1-dim. non-linear solver Compute u i+1 = arg min u D(P u , P u i+1 ) using B and dP u i+1 dP u ∝ w 1 1+λ i dP u i dP u as in Sec. 2.2 if λi ≤ δ then return control ui+1 with P u i+1 ≈ Q
where we note that the expression in the expectation is proportional to the importance weights in (12). Note that we can obtain a Monte Carlo estimate of the dual function using only simulations X ui from the previous iterations. As it turns out, these simulations are in most cases already required when learning the control u i+1 and we can thus store them in a buffer. We can then obtain λ i = arg max λ∈R + L (i) Dual (λ) using any non-linear solver with minimal computational overhead. In theory, we can define u i+1 = u λi i+1 using the representations in Prop. 2.5 and proceed with the next iteration of our trust region method in (4). However, computing the optimal control u i+1 using the representations in Prop. 2.5 requires gradients and Monte Carlo estimators of the value functions. This is problematic since it relies on a large amount of samples for each state x due to the (typically) very high variance of the estimator; see App. C for details. Thus, we propose versions of iterative diffusion optimization to learn parametrized approximations to u i+1 in the next section.
this section cite: ['b11']

Section: Learning the constrained optimal control
In this section we propose strategies to learn the optimal control for each iteration. As before, the general idea is to minimize loss functionals based on divergences between path space measures, namely L(u) = D(P u , P ui+1 ). Such divergences often rely on the Radon-Nikodym derivative
dP u i+1 dP u (X u i ) = dP u i+1 dP u i (X u i ) dP u i dP u (X u i ) ∝ exp T 0 ∥u i -u∥ 2 2 (X u i s , s)ds + T 0 ui -u (X u i s , s) • dWs -W i (X u i ,0) 1+λ i ,(15)
where we used Girsanov's theorem and (12). Note that the Radon-Nikodym derivative in (15) depends only on samples of the process with the already learned u i . Let us now suggest two concrete divergences. Those divergences are desirable for high-dimensional problems since both do not rely on computing derivatives of the stochastic process and can be optimized "off-policy" using trajectories X ui with the control u i of the previous iteration, which can be stored in a buffer; see Algorithm 1.
Log-variance divergence. This divergence can be considered w.r.t. an arbitrary reference measure, where we choose P ui for convenience [87,98]. We can then define the loss functional
LLV(u) := Var log dP u i+1 dP u (X u i ) ,(16)
where the Radon-Nikodym derivative can be explicitly computed as in (15). Note that for λ i = 0, this loss reduces to the on-policy log-variance loss typically used in the literature [97]. While this loss has beneficial theoretical properties [87], it requires to keep the full trajectory in memory for the gradient computation.
Cross-entropy divergence and SOC matching. Alternatively, we can consider the cross-entropy loss (i.e., the forward KL divergence computed using reweighting)
LCE(u) := DKL(P u i+1 |P u ) = E log dP u i+1 dP u (X u i ) dP u i+1 dP u i (X u i ) ,(17)
where the Radon-Nikodym derivative is again given by (15). Contrary to the log-variance loss, the reweighting dP u i+1 dP u i in (12) induces exponential terms. Our trust region constraint makes sure, however, that the variance of those weights stays bounded, see Remark 2.1.
RE CE LV AM SOCM TR-SOCM TR-LV 2 10 25 50 100 150 200 10 -4 10 -3 10 -2 10 -1 10 0 10 1 10 2 d control L 2 error 2 10 25 50 100 150 200 0 1 2 3 d |∆ log Z| 2 10 25 50 100 150 200 0 200 400 600 800 1,000 d sinkhorn distance 2 10 25 50 100 150 200 0 0.2 0.4 0.6 0.8 d total variation distance
Figure 2: Performance criteria for a Gaussian mixture target density with varying dimension d, averaged across four seeds. We show the errors of estimating the optimal control, the log-normalizing constant, as well as the Sinkhorn and total variation distances over different dimensions (from left to right). We observe that our trust region methods (TR-SOCM and TR-LV) are the only methods that perform well in high dimensions.
To efficiently compute this loss, we define the so-called (lean 6 ) adjoint state a as in [41]
via d ds ai+1(Xs, s) = -(∇b(Xs, s) ⊤ ai+1(Xs, s) + βi+1∇f (Xs, s)(18)
with a i+1 (X T , T ) = β i+1 ∇g(X T ), satisfying a i+1 (X s , s) = ∇ Xs β i+1 W(X, s); see [41,Lemma 5] and observe that it differs from the standard lean adjoint by the factor β i defined in Prop. 2.2. Similar to [42], we can use the expression for the optimal control in Prop. 2.5 and the Girsanov theorem to arrive at the SOC matching loss 7 , a simple regression objective given by
LSOCM(u) := E 1 2 T 0 ∥σ ⊤ ai+1(X u i s , s) -u(X u i s , s)∥ 2 ds dP u i+1 dP u i (X u i ) ,(19)
see App. G.5 for details. Contrary to the log-variance divergence above, this objective does not require to keep the whole trajectory X ui in memory for backpropagation but can be computed at times t ∼ Unif([0, T ]) using a Monte Carlo approximation. We summarize our algorithm in (1) and compare the different losses against existing approaches for SOC problems in the next section.
this section cite: ['b11', 'b14', 'b86', 'b97', 'b14', 'b96', 'b86', 'b14', 'b40', 'b40', 'b41']

Section: Applications
In this section, we explore several applications of SOC, comparing our novel trust-region-based optimization algorithm against existing methods. Specifically, we consider the three tasks sampling from unnormalized densities, transition path sampling, and fine-tuning text-to-image models. For background information, detailed experimental setups, and additional results, we refer to Apps. I to K, respectively. We also include further experiments on classical SOC problems in App. L.
this section cite: []

Section: Diffusion-based sampling
Using (3), we can show that sampling problems can be reformulated as SOC problems. To this end, we leverage the following corollary showing that the terminal distributions Q T and P T of the optimally controlled and uncontrolled processes differ by a tilting. Corollary 3.1 (Sampling from tilted distributions). Let us set f = 0 and assume that the terminal distribution of the uncontrolled process X is independent of p 0 and admits a density denoted by P T . Then it holds that Q T ∝ P T e -g .
Proof. Using (3) it holds that dQ dP (X) = e -g(X T ) Z(X0) with Z(X 0 ) = E e -g(X T ) |X 0 . The results
follows from the independence of X T and X 0 ; see [41] and App. I for details.
Cor. 3.1 shows that the optimally controlled process X u * samples from a given unnormalized density ρ target when using an uncontrolled process with known terminal distribution P T and setting g = log P T ρtarget ; see [33, 95, 97, 125, 129-131, 144, 149] and App. I for details. Such sampling problems are of immense practical interest, with numerous applications in the natural sciences [109,151], in Bayesian statistics [54], and reinforcement learning [24].
Numerical experiments. Here, we compare existing methods for solving SOC problems with our trust region method on challenging multimodal sampling problems. We use the Denoising Diffusion Sampler (DDS) [129] method, which leverages an ergodic Ornstein-Uhlenbeck process initialized at its equilibrium measure as uncontrolled process X. We consider five baselines, specifically, reverse and (importance weighted) forward KL, also known as relative entropy (RE) and cross entropy (CE) method, respectively. Additionally, we consider the log-variance loss [98], adjoint matching (AM) [41], and stochastic optimal control matching (SOCM) [42], for the unconstrained problem in (2); see [40] for a comprehensive overview of SOC losses. In all experiments, we deliberately avoid using gradient guidance from the target density in the diffusion process, often referred to as Langevin preconditioning (LP) [66]. Prior work has shown that LP is essential for preventing mode collapse in neural samplers [18,66]. However, LP is computationally expensive, as it requires querying the target distribution at every discretization step, making such approaches impractical for many problems where evaluating the target gradient is costly.
First, we consider a Gaussian Mixture Model (GMM) comprising 10 components and randomized mixing weights. GMMs are particularly compelling as they admit an analytical solution for the optimal control, which enables direct computation of the L 2 error between the learned and optimal controls, a reliable metric for detecting mode collapse. In addition, we assess the Sinkhorn distance [31] between samples from the target and the model, and the absolute error in estimating the lognormalizing constant, denoted |∆ log Z|. Finally, we evaluate the total variation distance between the true mixing weights and the model's estimated weights. The results, shown in Fig. 2, indicate that for d = 2, all methods closely approximate the optimal control. However, for dimensions beyond d = 10, most methods suffer from mode collapse, as reflected by increased control errors, except for those employing trust region updates. Trust region methods maintain robustness across a wide range of dimensions and only begin to show signs of mode collapse in high dimensions (d ≥ 150).
We additionally evaluate our method on the Many Well target [135] with 32 modes. For quantitative analysis, we report the log-normalization error |∆ log Z|, as other ground-truth quantities are unavailable. Additionally, for the high-dimensional case d = 200, we visualize pairs of marginal distributions in App. I. The results, presented in Fig. 3, demonstrate that our method significantly outperforms competing approaches in estimating the normalizing constant. Furthermore, the visualizations in App. I illustrate that trust region updates effectively prevent mode collapse, even in high dimensions. In contrast, baseline methods either suffer from mode collapse or fail to converge.
Finally, we perform an ablation study on the GMM target, analyzing key components of our proposed method. Specifically, we investigate the effects of incorporating a replay buffer and applying trust region optimization. To this end, we compare a variant using a fixed Lagrangian multiplier λ, selected via hyperparameter tuning, with one in which λ is dynamically optimized using our trust region approach. Additionally, we evaluate the log-variance loss both with and without using a replay buffer. Moreover, we compare our method to LV with Langevin preconditioning on the GMM target with dimensionality d = 100. The results, shown in Figure 3, demonstrate that trust region optimization significantly reduces control error and decreases the number of target evaluations by several orders of magnitude.
this section cite: ['b40', 'b108', 'b150', 'b53', 'b23', 'b128', 'b97', 'b40', 'b41', 'b39', 'b65', 'b17', 'b65', 'b30', 'b134']

Section: Transition path sampling
Transition path sampling is of great importance for studying phase transitions and chemical reactions.
The key challenge comes from the energy barrier that connects two sets A and B along the energy landscape, which makes direct sampling of transition paths extremely unlikely. These problems can also be formulated as SOC problems [59,62,115]. Specifically, we set b = -∇U , where U : R N ×3 → R is the potential function, and g =log 1 B as well as p 0 ∝ 1 A , which constraints the initial and target states in the sets A and B. As in (3), it holds that dQ dP = 1 B (X T ) Z(X0) . Recent work  All results are averaged over three random seeds, with both the mean and standard deviation reported. Our method identifies transition paths more consistently and robustly, as evidenced by higher THP values and lower standard deviations.
has leveraged neural networks to parameterize a bias force to solve the corresponding SOC problem, employing objectives such as the KL [44,72,141], or log-variance divergence [113].
Numerical experiments. We evaluate the performance of the trust-region-based log-variance loss (TR-LV) on two transition path sampling problems: Alanine Dipeptide isomerization and Chignolin folding, with 22 and 138 atoms, respectively.
Our evaluation includes three metrics: Kabsch-aligned root mean squared distance (RMSD) between the final states of the sampled paths and the target state, transition hit percentage (THP) measuring the proportion of final states hitting within the target region, and energy of transition state (ETS) identifying the highest energy values along paths that reach the target.
We compare our method to standard molecular dynamics (MD) with increased temperature (UMD), steered MD (SMD) [75] with force applied to collective variables, and PIPS [72] which uses the crossentropy loss. We also include TPS-DPS [113] as a key baseline, which employs an (unconstrained) logvariance loss to formulate TPS as a stochastic optimal control (SOC) problem. Further experimental details are provided in App. J.
Table 1 shows that TR-LV achieves superior target state RMSD and transition hit percentage compared to the standard log-variance objective (TPS-DPS) for both molecular systems. Notably, SMD performs well due to its use of collective variables with biased force guiding the sampling process. Figure 4 illustrates that the trust region constraint leads to significantly more robust training compared to TPS-DPS as indicated by low standard deviations across different seeds. Moreover, on Alanine Dipeptide, the trust region constraint initially regularizes optimization and accelerates convergence thereafter. Across both systems, the trust region constraint significantly enhances training stability and performance.
this section cite: ['b58', 'b61', 'b114', 'b43', 'b71', 'b140', 'b112', 'b74', 'b71', 'b112']

Section: Fine-tuning of diffusion models
Interpreting -g as a reward and the uncontrolled process X as a pretrained diffusion model (i.e., b includes the pretrained neural network), Cor. 3.1 shows that we can perform reward fine-tuning by solving the SOC problem in (1); see also [38,41,132]. Reward fine-tuning has recently shown impressive results, e.g., in image [28,41] and molecule generation [38], and SOC provides a principled framework. A special case is given by posterior sampling [38]. Setting g =log p(y|x),
where p(y|x) is the likelihood and we interpret P T as a learned (diffusion) prior p(x), Bayes' theorem shows that the optimally controlled process samples from the posterior p(x|y).
this section cite: ['b0', 'b37', 'b40', 'b131', 'b27', 'b40', 'b37', 'b37']

Section: Numerical experiments.
We perform reward fine-tuning on Stable Diffusion 1.5 [102], using ImageReward [140], which is a reward model designed to capture prompt alignment and image quality according to human preferences. We take the adjoint matching (AM) method as baseline and
AM, η = 0 AM, η = 1 TR-SOCM, η = 0 TR-SOCM, η = 1 0 0.2 0.4 0.6 0.8 1 0.2 0.4 0.6 0.8 1 number of trajectories ×10 5 ImageReward 0 0.2 0.4 0.6 0.8 1 0.272 0.274 0.276 0.278 0.28 0.282 0.284 number of trajectories ×10 5 CLIP-Score 0 0.2 0.4 0.6 0.8 1 0.26 0.27 0.28 number of trajectories ×10 5 HPS 0 0.2 0.4 0.6 0.8 1 0.3 0.32 0.34 0.36 0.38 0.4 number of trajectories ×10 5 DreamSim Variance masterpiece, best quality, realistic photograph, 8k, high detailed vintage motorcycle parked on a wet cobblestone street at dusk, neon reflections, shallow depth of field close up photo of anthropomorphic fox animal dressed in white shirt, fox animal, glasses compare it against our TR-SOCM loss (19), keeping all other hyperparameters fixed. Our TR-SOCM allows the principled use of buffers, and we perform three passes on each buffer of size 500, leading to three times fewer trajectories for a fixed number of model updates. For faster convergence, we use a modified version of TR-SOCM with annealing factor β i = 1. For each algorithm, we evaluate 5 checkpoints during fine-tuning (with ODE and SDE inference) on ImageReward and three additional metrics: CLIP-Score [69], which measures prompt alignment, Human Preference Score [137], which measures human-perceived image quality, and Dreamsim diversity [51], which measures per-prompt diversity. We observe that TR-SOCM achieves similar performance metrics to AM at a fraction of the cost, as sampling the trajectories and solving the lean adjoint ODE, which dominates the computational costs, is amortized over the buffer passes; see Figs. 5 and 6 as well as App. K for more details.
this section cite: ['b101', 'b139', 'b18', 'b68', 'b136', 'b50']

Section: Related works
In this section, we discuss the most related works, comparing our approach to existing methods for solving SOC problems. We provide a more extensive comparison in App. C.
this section cite: []

Section: Iterative diffusion optimization.
Many recently developed methods approach SOC problems by simulating the (diffusion) process X u , computing a suitable cost function, and optimizing the parameters of the control function u using variants of stochastic gradient methods. These techniques are collectively referred to as iterative diffusion optimization (IDO) methods [87]. While the underlying theory dates back to [33,91], combinations with deep learning in the context of SOC have been explored by [15,87,95,97,129,131,149,152]. One can derive most of the related objectives starting from the Radon-Nikodym derivative dP u dQ (X u ) as in (12) (with u = u i ). One can then minimize a loss based on a suitable divergence as in (2). Previous works have, e.g., proposed the log-variance divergence [97,113] or the forward KL divergence (corresponding to the cross-entropy loss [61,72,76,104,150]), for which we develop corresponding trust region versions in (16) and (17). The SOC matching loss [42], which we extended to trust regions in (19), is equal to the cross entropy loss in expectation but exhibits lower variance empirically. We refer to [41] for more IDO losses. However, all existing methods have either directly tackled the target measure Q or relied on a form of hand-tuned annealing.
this section cite: ['b86', 'b32', 'b90', 'b14', 'b86', 'b94', 'b96', 'b128', 'b130', 'b148', 'b151', 'b96', 'b112', 'b60', 'b71', 'b75', 'b103', 'b149', 'b15', 'b16', 'b41', 'b18', 'b40']

Section: Trust region methods.
We show how IDO methods can generally be extended to trust region methods, enabling (1) automatic control on the variance of the importance weights and (2) principled usage of buffers, leading to faster and more stable convergence, in particular avoiding mode collapse in high dimensions. Trust region methods have a long history as robust optimization algorithms that iteratively minimize an objective within an adaptively sized "trust region"; see [29] for an overview. These methods have also been extended to optimize over spaces of probability distributions, particularly in reinforcement learning [2-4, 7, 83, 88, 90, 93, 110, 111, 138, 139, 142], black-box optimization [1,118,134], variational inference [9,10] and path integral control [123]. To the best of our knowledge, these methods have not yet been extended to path measures or inference problems. Moreover, the connection between trust-region iterates and geometric annealing has not previously been established.
this section cite: ['b28', 'b0', 'b117', 'b133', 'b8', 'b9', 'b122']

Section: Conclusion
In this work, we develop a novel framework for solving SOC problems using deep learning. Our framework builds on the fact that we can reformulate specific problems as finding an optimal path space measure induced by a controlled SDE. Instead of finding this optimal measure at once, we divide the unconstrained problem into a sequence of constrained optimization problems by bounding the KL divergence to the measure from the previous iteration. We show that this defines a wellbehaved geometric annealing between the prior and the target path measure, resulting in equidistant steps on the Fisher-Rao information manifold. Crucially, each intermediate problem turns out to be an altered SOC problem that can be efficiently solved without simulations by using a buffer of trajectories with the control from the previous iteration. In our experiments, we show that our method significantly improves the learning of the optimal control, including applications in diffusion-based sampling and transition path sampling in molecular dynamics. Further, we show that our method can be scaled to improve the efficiency of reward fine-tuning for text-to-image diffusion models. In the future, we expect our framework to improve even more applications of SOC, potentially including the use of divergences other than the KL divergence for the trust region constraint. Finally, our results for general measures motivate the use of trust region methods for other learned measure transports, e.g., normalizing flows.
this section cite: []

Section: References
Ref_id:b0 Title: Modelbased relative entropy stochastic search Year: (2015)
Ref_id:b1 Title: Relative entropy regularized policy iteration Year: (2018)
Ref_id:b2 Title: Maximum a posteriori policy optimisation Year: (2018)
Ref_id:b3 Title: Constrained policy optimization Year: (2017)
Ref_id:b4 Title: Progressive inference-time annealing of diffusion models for sampling from Boltzmann densities Year: (2025)
Ref_id:b5 Title: Iterated denoising energy matching for sampling from Boltzmann densities Year: (2024)
Ref_id:b6 Title: Projections for approximate policy iteration algorithms Year: (2019)
Ref_id:b7 Title: NETS: A non-equilibrium transport sampler Year: (2024)
Ref_id:b8 Title: A unified perspective on natural gradient variational inference with Gaussian mixture models Year: (2022)
Ref_id:b9 Title: Trust-region variational inference with Gaussian mixture models Year: (2020)
Ref_id:b10 Title: Solving the Kolmogorov PDE by means of deep learning Year: (2021)
Ref_id:b11 Title: Dynamic programming Year: (1957)
Ref_id:b12 Title: Numerically solving parametric families of highdimensional Kolmogorov partial differential equations via deep learning Year: (2020)
Ref_id:b13 Title: From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training Year: (2025)
Ref_id:b14 Title: An optimal control perspective on diffusion-based generative modeling Year: (2024)
Ref_id:b15 Title: Training diffusion models with reinforcement learning Year: (2024)
Ref_id:b16 Title: Underdamped diffusion bridges with applications to sampling Year: (2025)
Ref_id:b17 Title: Beyond ELBOs: A large-scale evaluation of variational methods for sampling Year: (2024)
Ref_id:b18 Title: End-to-end learning of Gaussian mixture priors for diffusion sampler Year: (2025)
Ref_id:b19 Title: Transition path sampling: Throwing ropes over rough mountain passes, in the dark Year: (2002)
Ref_id:b20 Title: Jax: Autograd and xla Year: (2021)
Ref_id:b21 Title: All in the exponential family: Bregman duality in thermodynamic variational inference Year: (2020)
Ref_id:b22 Title: An algorithm with guaranteed convergence for finding a zero of a function Year: (1971)
Ref_id:b23 Title: Diffusion-based maximum entropy reinforcement learning Year: (2025)
Ref_id:b24 Title: Sequential controlled Langevin diffusions Year: (2025)
Ref_id:b25 Title: Variational and optimal control representations of conditioned and driven processes Year: (2015)
Ref_id:b26 Title: Non-equilibrium annealed adjoint sampler Year: (2025)
Ref_id:b27 Title: Directly fine-tuning diffusion models on differentiable rewards Year: (2024)
Ref_id:b28 Title: Trust region methods Year: (2000)
Ref_id:b29 Title: Measuring thermodynamic length Year: (2007)
Ref_id:b30 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b31 Title: Optimal transport tools (OTT): A jax toolbox for all things Wasserstein Year: (2022)
Ref_id:b32 Title: A stochastic control approach to reciprocal diffusion processes Year: (1991)
Ref_id:b33 Title: Connections between stochastic control and dynamic games Year: (1996)
Ref_id:b34 Title: Reinforcement learning of rare diffusive dynamics Year: ()
Ref_id:b35 Title: Diffusion Schrödinger bridge with applications to score-based generative modeling Year: (2021)
Ref_id:b36 Title: Efficient transition path sampling: Application to lennard-jones cluster rearrangements Year: (1998)
Ref_id:b37 Title: A framework for conditional diffusion modelling with applications in motif scaffolding for protein design Year: (2023)
Ref_id:b38 Title: Sampling via Föllmer flow Year: (2023)
Ref_id:b39 Title: A taxonomy of loss functions for stochastic optimal control Year: (2024)
Ref_id:b40 Title: Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control Year: (2025)
Ref_id:b41 Title: Stochastic optimal control matching Year: (2024)
Ref_id:b42 Title: Score-based diffusion meets annealed importance sampling Year: (2022)
Ref_id:b43 Title: Doob's lagrangian: A sample-efficient variational approach to transition path sampling Year: ()
Ref_id:b44 Title: Doob's lagrangian: A sample-efficient variational approach to transition path sampling Year: (2024)
Ref_id:b45 Title: Continuously tempered diffusion samplers Year: (2025)
Ref_id:b46 Title: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models Year: (2023)
Ref_id:b47 Title: Sampling algorithms in statistical physics: a guide for statistics and machine learning Year: (2024)
Ref_id:b48 Title: Deterministic and Stochastic Optimal Control. Applications of mathematics Year: (1975)
Ref_id:b49 Title: Controlled Markov processes and viscosity solutions Year: (2006)
Ref_id:b50 Title: Learning new dimensions of human visual similarity using synthetic data Year: (2023)
Ref_id:b51 Title: MCMC variational inference via uncorrected hamiltonian annealing Year: (2021)
Ref_id:b52 Title: Langevin diffusion variational inference Year: (2023)
Ref_id:b53 Title: Bayesian Data Analysis Year: (2013)
Ref_id:b54 Title: Improving the evaluation of samplers on multi-modal targets Year: (2025)
Ref_id:b55 Title: Adaptive destruction processes for diffusion samplers Year: (2025)
Ref_id:b56 Title: Complexity analysis of normalizing constant estimation: from Jarzynski equality to annealed importance sampling and beyond Year: (2025)
Ref_id:b57 Title: Solving high-dimensional partial differential equations using deep learning Year: (2018)
Ref_id:b58 Title: Variational approach to rare event simulation using least-squares regression Year: (2019)
Ref_id:b59 Title: Nonasymptotic bounds for suboptimal importance sampling Year: (2024)
Ref_id:b60 Title: Variational characterization of free energy: Theory and algorithms Year: (2017)
Ref_id:b61 Title: Efficient rare event simulation by optimal nonequilibrium forcing Year: (2012)
Ref_id:b62 Title: Adjoint sampling: Highly scalable diffusion samplers via adjoint matching Year: (2025)
Ref_id:b63 Title: Training neural samplers with reverse diffusive kl divergence Year: (2024)
Ref_id:b64 Title: FEAT: Free energy estimators with adaptive transport Year: (2025)
Ref_id:b65 Title: No trick, no treat: Pursuits and challenges towards simulation-free training of neural samplers Year: (2025)
Ref_id:b66 Title: Gaussian error linear units (gelus) Year: (2016)
Ref_id:b67 Title: Enhanced sampling methods for molecular dynamics simulations Year: (2022)
Ref_id:b68 Title: Clipscore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b69 Title: Leaps: A discrete neural sampler via locally equivariant networks Year: (2025)
Ref_id:b70 Title: Stochastic optimal control for collective variable free sampling of molecular transition paths Year: (2023)
Ref_id:b71 Title: Path integral stochastic optimal control for sampling transition paths Year: (2022)
Ref_id:b72 Title: Schrödinger-Föllmer sampler: sampling without ergodicity Year: (2021)
Ref_id:b73 Title: Monte Carlo sampling without isoperimetry: A reverse diffusion approach Year: (2023)
Ref_id:b74 Title: Steered molecular dynamics Year: (1997)
Ref_id:b75 Title: Adaptive importance sampling for control and inference Year: (2016)
Ref_id:b76 Title: Adaptive teachers for amortized samplers Year: (2024)
Ref_id:b77 Title: On scalable and efficient training of diffusion samplers Year: (2025)
Ref_id:b78 Title: A method for stochastic optimization Year: (2014)
Ref_id:b79 Title: Adjoint schrödinger bridge sampler Year: (2025)
Ref_id:b80 Title: Flow-grpo: Training flow matching models via online rl Year: (2025)
Ref_id:b81 Title: Efficient diversity-preserving diffusion alignment via gradient-informed GFlowNets Year: (2025)
Ref_id:b82 Title: An off-policy trust region policy optimization method with monotonic improvement guarantee for deep reinforcement learning Year: (2021)
Ref_id:b83 Title: Flow annealed importance sampling bootstrap Year: (2022)
Ref_id:b84 Title: Probabilistic inference using Markov chain Monte Carlo methods Year: (1993)
Ref_id:b85 Title: Learned reference-based diffusion sampling for multi-modal distributions Year: (2024)
Ref_id:b86 Title: Solving high-dimensional Hamilton-Jacobi-Bellman PDEs using neural networks: perspectives from the theory of controlled diffusions and measures on path space Year: (2021)
Ref_id:b87 Title: Differentiable trust region layers for deep reinforcement learning Year: (2021)
Ref_id:b88 Title: BNEM: A Boltzmann sampler based on bootstrapped noised energy matching Year: (2024)
Ref_id:b89 Title: Compatible natural gradient policy search Year: (2019)
Ref_id:b90 Title: Stochastic control and nonequilibrium thermodynamical systems Year: (1989)
Ref_id:b91 Title: On local entropy, stochastic control and deep neural networks Year: (2022)
Ref_id:b92 Title: Relative entropy policy search Year: (2010)
Ref_id:b93 Title: Continuous-time Stochastic Control and Optimization with Financial Applications. Stochastic Modelling and Applied Probability Year: (2009)
Ref_id:b94 Title: Solving high-dimensional PDEs, approximation of path space measures and importance sampling of diffusions Year: (2021)
Ref_id:b95 Title: Robust SDE-based variational formulations for solving linear PDEs via deep learning Year: (2022)
Ref_id:b96 Title: Improved sampling via learned diffusions Year: (2024)
Ref_id:b97 Title: VarGrad: A low-variance gradient estimator for variational inference Year: (2020)
Ref_id:b98 Title: Solving high-dimensional parabolic PDEs using the tensor train format Year: (2021)
Ref_id:b99 Title: From continuous-time formulations to discretization schemes: tensor trains and robust regression for bsdes and parabolic pdes Year: (2024)
Ref_id:b100 Title: Progressive tempering sampler with diffusion Year: (2025)
Ref_id:b101 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b102 Title: A reinforcement learning approach to rare trajectory sampling Year: (2021)
Ref_id:b103 Title: The cross-entropy method: a unified approach to combinatorial optimization, Monte-Carlo simulation and machine learning Year: (2013)
Ref_id:b104 Title: Unbiased deep solvers for linear parametric PDEs Year: (2021)
Ref_id:b105 Title: Thermodynamic length and dissipated availability Year: (1983)
Ref_id:b106 Title: Scalable discrete diffusion samplers: Combinatorial optimization and statistical physics Year: (2025)
Ref_id:b107 Title: Rethinking losses for diffusion bridge samplers Year: (2025)
Ref_id:b108 Title: Temperature-annealed boltzmann generators Year: (2025)
Ref_id:b109 Title: Trust region policy optimization Year: (2015)
Ref_id:b110 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b111 Title: Improved off-policy training of diffusion samplers Year: (2024)
Ref_id:b112 Title: Transition path sampling with improved off-policy training of diffusion path samplers Year: (2024)
Ref_id:b113 Title: Diffusion-PINN sampler Year: (2024)
Ref_id:b114 Title: Variational path sampling of rare dynamical events Year: (2025)
Ref_id:b115 Title: Variational deep learning of equilibrium transition path ensembles Year: ()
Ref_id:b116 Title: Dynamical measure transport and neural PDE solvers for sampling Year: (2024)
Ref_id:b117 Title: Efficient natural evolution strategies Year: (2009)
Ref_id:b118 Title: Optimised annealed sequential Monte Carlo samplers Year: (2024)
Ref_id:b119 Title: Scalable equilibrium sampling with sequential Boltzmann generators Year: (2025)
Ref_id:b120 Title: Noise-free sampling algorithms via regularized Wasserstein proximals Year: (2023)
Ref_id:b121 Title: Fourier features let networks learn high frequency functions in low dimensional domains Year: (2020)
Ref_id:b122 Title: Adaptive smoothing for path integral control Year: (2020)
Ref_id:b123 Title: Monte Carlo variational auto-encoders Year: (2021)
Ref_id:b124 Title: Theoretical guarantees for sampling and inference in generative models with latent diffusions Year: (2019)
Ref_id:b125 Title: Fine-tuning of continuous-time diffusion models as entropyregularized control Year: (2024)
Ref_id:b126 Title: Stochastic calculus, filtering, and stochastic control. Course notes Year: (2007)
Ref_id:b127 Title: Transition-path theory and path-finding algorithms for the study of rare events Year: (2010)
Ref_id:b128 Title:  Year: (2023)
Ref_id:b129 Title: Bayesian learning via neural Schrödinger-Föllmer flows Year: (2023)
Ref_id:b130 Title: Transport meets variational inference: Controlled Monte Carlo diffusions Year: (2024)
Ref_id:b131 Title: Amortizing intractable inference in diffusion models for vision, language, and control Year: (2024)
Ref_id:b132 Title: Importance weighted score matching for diffusion samplers with enhanced mode coverage Year: (2025)
Ref_id:b133 Title: Natural evolution strategies Year: (2014)
Ref_id:b134 Title: Stochastic normalizing flows Year: (2020)
Ref_id:b135 Title: Reverse diffusion Sequential Monte Carlo samplers Year: (2025)
Ref_id:b136 Title: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis Year: (2023)
Ref_id:b137 Title: Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation Year: (2017)
Ref_id:b138 Title: Trust region policy optimization via entropy regularization for kullback-leibler divergence constraint Year: (2024)
Ref_id:b139 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b140 Title: Learning nonequilibrium control forces to characterize dynamical phase transitions Year: (2022)
Ref_id:b141 Title: Projection-based constrained policy optimization Year: (2020)
Ref_id:b142 Title: Value gradient sampler: Sampling as sequential decision making Year: (2025)
Ref_id:b143 Title: Diffusion generative flow samplers: Improving learning signals through partial trajectory optimization Year: (2024)
Ref_id:b144 Title: Improving GFlowNets for text-to-image diffusion alignment Year: (2024)
Ref_id:b145 Title: Differentiable annealed importance sampling and the perils of gradient noise Year: (2021)
Ref_id:b146 Title: Generalised parallel tempering: Flexible replica exchange via flows and diffusions Year: (2025)
Ref_id:b147 Title: Accelerated parallel tempering via neural transports Year: (2025)
Ref_id:b148 Title: Path Integral Sampler: a stochastic control approach for sampling Year: (2022)
Ref_id:b149 Title: Applications of the cross-entropy method to importance sampling and optimal control of diffusions Year: (2014)
Ref_id:b150 Title: Artificial intelligence for science in quantum, atomistic, and continuum systems Year: (2023)
Ref_id:b151 Title: Actor-critic method for high dimensional static Hamilton-Jacobi-Bellman partial differential equations based on neural networks Year: (2021)
Ref_id:b152 Title: Mdns: Masked diffusion neural sampler via stochastic optimal control Year: (2025)
