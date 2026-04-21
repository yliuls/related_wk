Title: Dynamical Decoupling of Generalization and Overfitting in Large Two-Layer Networks
Abstract: Understanding the inductive bias and generalization properties of large overparametrized machine learning models requires to characterize the dynamics of the training algorithm. We study the learning dynamics of large two-layer neural networks via dynamical mean field theory, a well established technique of nonequilibrium statistical physics. We show that, for large network width m, and large number of samples per input dimension n/d, the training dynamics exhibits a separation of timescales which implies: (i) The emergence of a slow time scale associated with the growth in Gaussian/Rademacher complexity of the network;(ii) Inductive bias towards small complexity if the initialization has small enough complexity; (iii) A dynamical decoupling between feature learning and overfitting regimes; (iv) A non-monotone behavior of the test error, associated 'feature unlearning' regime at large times.

Section: Introduction
Machine learning (ML) models are trained using stochastic gradient descent (SGD), or one of its variants to minimize the error on training data (empirical risk function). Classically, their good behavior on unseen test data is explained by the fact that model complexity is kept small by regularization techniques: these models do not 'overfit.' Traditional ML theory decouples the analysis of the model from the optimization algorithm, which is assumed to converge to an approximate global minimizer [47].
In contrast, in modern ML, the empirical risk is highly non-convex, the number of parameters is comparable with the number of training samples, and the model complexity is only weakly controlled. As a consequence, there can be many assignments of the model parameters (many global empirical risk minimizers) that perfectly interpolate the data -even when these are noisy. While all of these interpolators are indistinguishable on the training data, they behave very differently (and some of them very poorly) on test data. It has been hypothesized that models trained by SGD generalize well to test data because the algorithm selects a near global minimizer with low complexity, although a mechanistic understanding of this process is lacking. For this reason, the generalization properties cannot be decoupled from the training dynamics.
Several striking consequences of this lack of decoupling are documented in the literature (and have long been familiar to practitioners): (i) Test error after training is observed to depend strongly on the initial weights distribution [28]; (ii) Test error depends strongly on the optimization algorithm (SGD, RMSProp, ADAM, to name a few), even when these algorithms achieve the same train error [55]; (iii) Careful choice of the hyperparameters in the optimization algorithm is crucial [34,59], and the optimal choice is often different from the one that minimizes train error; (iv) Models learned by training for a shorter time have smaller complexity and can generalize better [44,11]. Training data comprises n points in d dimensions distributed according to a single index model. We assume n, m, d all large with n/md = α (here α = 0.3). Blue: test error. Purple: train error. Red: ℓ 1 norm of second-layer weights (a proxy for model complexity).
These observations have motivated a broad effort to encapsulate the effect of the dynamics as 'implicit regularization' [48,3,15,56]: the algorithm selects an empirical risk minimizer that also minimizes a specific notion of model complexity. While this implicit regularization hypothesis has been fruitful, it can only be validated if we can precisely understand the training dynamics.
In this work we leverage tools from theoretical physics to directly analyze the training dynamics and derive quantitative predictions on the implicit bias of neural network training, in a simple setting. This allows us to capture feature learning and lazy/overfitting regimes within the same unified picture. We discover a time-scale separation in the training dynamics, between an early stage in which the model learns the relevant features representation of the data, and a late stage of training that is characterized by overfitting, feature 'unlearning,' and hence test error that increases with training. While the regularizing effect of early stopping has been an important object of study (for simpler models) in the past [44,11,61,57], our work is the first to point out a time-scale separation between feature learning (on a faster timescale) and overfitting (on a slower time scale), thus reconciling the feature learning and neural tangent theories of learning.
We study two-layer fully connected neural networks f ( • ; θ) : R d → R, i.e.
f (x; θ) = 1 m m i=1 a i σ(⟨w i , x⟩) ,(1.1)
where θ = (a, W ), where W = (w 1 , . . . , w m ) ∈ R d×m and a = (a 1 , . . . , a m ) ∈ R m are, respectively, first-and second-layer weights. For convenience, we fix the normalization ∥w i ∥ = 1, and assume that σ does not depend on m. We apply model (1.1) to a supervised learning task. We are given i.i.d. data (y i , x i ), i ≤ n, with y i ∈ R a response variable and x i ∈ R d a feature vector, and try to learn a model f ( • ; θ) to predict the response y new corresponding to a new input x new . We use gradient flow (GF) to minimize the empirical risk under square loss, namely
θ(t) = - n d P θ ∇ R n (θ(t)) , R n (θ) := 1 2n n i=1 y i -f (x i ; θ) 2 . (1.2)
Here P θ is a projection matrix that guarantees that w i (t) ∈ S d-1 at all times. The factor n/d is introduced for convenience and simply amounts to a rescaling of time. We will typically initialize the training by setting (w i ) i≤m ∼ iid Unif(S d-1 ), and a i = a 0 for all i ≤ m, and study the dependence of the training dynamics on three key parameters: Network width: m, Overparametrization ratio: α := n md , Initialization scale: a 0 .
Alongside the train error, we will be interested in the test error at time t, i.e. R(θ(t)) := E{(y newf (x new ; θ(t))) 2 }/2, and the generalization error R(θ(t)) -R n (θ(t)).
Model (1.1) is much simpler than state-of-the-art architectures [52], but is rich enough to investigate several general questions, which we summarize below:
When the network is sufficiently overparametrized (α small) and a 0 is large, neural tangent kernel (NTK) theory predicts that GF converges to an interpolator [30,22,16] .
this section cite: ['b46', 'b27', 'b54', 'b33', 'b58', 'b43', 'b10', 'b47', 'b2', 'b14', 'b55', 'b43', 'b10', 'b60', 'b56', 'b51', 'b29', 'b21', 'b15']

Section: Q1.
For which region of α, a 0 does convergence take place, beyond NTK theory?
Q2. Does the selected model provide good generalization or not [27,37]?
In contrast, when a 0 is small, gradient-based algorithms can learn non-linear low-dimensional representation of the data [5,21,1,6]. In these results, the difference between train and test error (generalization error) is negligible: the model does not overfit.
Q3. Can we reconcile this feature-learning/no-overfitting behavior with the lazytraining/overfitting regime described previously?
In the early phase of training, the generalization error vanishes. However, training longer times can be beneficial, despite leading to overfitting.
this section cite: ['b26', 'b36', 'b4', 'b20', 'b0', 'b5']

Section: Q4.
When does the test error start increasing with training time? When should we stop training?
Finally, scaling with the network size is crucial:
Q5. How does the generalization error depend on network size and number of iterations?
Q6. Does overfitting start earlier for larger networks or later?
In Section 2, we will present our analysis using theoretical physics techniques. Section 3 presents rigorous results confirming the picture emerging from this analysis. Finally, in Section 4 we discuss how our results address the above questions.
this section cite: []

Section: Main results: Dynamical mean field theory
We study the dynamics of model (1.1) under the simplest data distribution in which genuine non-linear learning is required to efficiently learn a good prediction rule, the so called k-index model. Namely, we assume x i ∼ N(0, I d ) and y i that depends on a low-dimensional projection U T x i :
y i = φ(U T x i ) + ε i , ε i ∼ N(0, τ 2 ) ,(2.1)
where the noise ε i is independent of x i , U ∈ R d×k is an orthogonal matrix (U T U = I k ) and φ : R k → R is a nonlinear function, E{φ(g) 2 } < ∞ for g standard Gaussian.
An important aspect of this data distribution is that (for large d) it presents the largest possible gap between linear/kernel learning, which requires sample size to be superpolynomial in d [27,58], and nonlinear/neural network learning which only requires n = O(d) (generically, for constant k).
When the dimension d becomes large, discovering the latent features U T x is crucial for learning and requires nonlinear processing of the labels y i [5,21,1,6].
Our main focus will be on the simplest case, namely k = 1, with φ a generic function (in particular E{φ(G)G} ̸ = 0 for G ∼ N(0, 1), which corresponds information exponent equal to one according to the classification of [4].). Some of our results apply to k-index models for general fixed k (in particular, the rigorous results of Section 3). We defer to future work a more complete analysis of the DMFT for k ≥ 2.
We discover a separation of time scales at large m (or large n/d), for sufficiently small initialization a 0 : feature learning takes place on a fast time scale, followed by overfitting/reversal to kernel learning. This scenario is summarized in Figure 1, which plots numerical evaluations of our theoretical predictions at k = 1, τ > 0 data distribution, in the limit n, d, m → ∞ at overparametrization ratio α = 0.3.
More precisely, we observe three regimes (below W 2nd := a/m is the vector of second-layer weights in model (1.1)):
(i) Mean field feature learning. t = O(1). The network learns the low-dimensional features U T x; the train error and test error decrease while their difference (generalization error) is negligible; the second layer weights remain small ∥W 2nd ∥ 1 = O(1). (ii) Extended feature learning. 1 ≪ t ≪ m. The train error decreases slowly; the generalization error increases is small, i.e. R(θ(t)) -R n (θ(t)) = o(1); the test error can evolve non-monotonically, but remains approximately constant. Second-layer weights become large 1 ≪ ∥W 2nd ∥ 1 ≪ √ m.
(iii) Overfitting and feature unlearning. t ≳ m. Train error and test error diverge significantly, i.e. R(θ(t)) -R n (θ(t)) becomes of order one. At the end of this regime, the train error converges to 0, i.e. the neural network interpolates the noisy data. The test error instead grows, and its limit value is the one of a (data independent) kernel method: in other words, the model unlearns the low-dimensional structure. Finally, the second weights grow to ∥W 2nd ∥ 1 ≍ √ m, which indeed is the scale required for interpolation.
In this section we outline our results based on 'dynamical mean field theory' (DMFT). The next section will present rigorous results that are proven independently.
this section cite: ['b26', 'b57', 'b4', 'b20', 'b0', 'b5', 'b3']

Section: Technique
Our DMFT analysis is based on the following two steps:
Step 1: We leverage techniques from theoretical physics to derive an approximate asymptotic characterization of the gradient flow dynamics (1.2) in the limit n, d → ∞, with n/d → α. This characterization consists of a set of integral-differential equations for the following asymptotic quantities (here p-lim denotes limit in probability, and we use the superscripts n to emphasize the dependence of the right-hand side on n, d) C ij (t 1 , t 2 ) := p-lim n,d→∞ ⟨w n i (t 1 ), w n j (t 2 )⟩ , v i (t) := p-lim n,d→∞ U T w n i (t) , a i (t) := p-lim n,d→∞ a n i (t) .
this section cite: []

Section: (2.2)
A rigorous derivation of the DMFT in a setting that includes two-layer networks is given in [13].
However, the asymptotically exact DMFT characterization of [13] is rather complex to integrate numerically or to study analytically. In order to circumvent this problem, we use a DMFT that is is asymptotically exact for a well-defined Gaussian version of the original model. Namely, we observe that the empirical risk of Eq. ( 1.2) takes the form
R n (θ) = 1 2n F (θ) 2 ,(2.3)
where F : (S d-1 ) m ×R m → R n is s stochastic process with i.i.d. components F i (θ) = y i -f (x i ; θ). We replace these by Gaussian processes with matching mean and covariance, and study the DMFT for gradient flow with respect to the associated risk R g n (θ). The Gaussian approximation comes with an error which we show analytically is vanishing on time scales of order one ( indeed on these time scales we correctly recover the mean field theory of [38,14]) and we demonstrate empirically to be small on larger time scales ( see for instance example Fig. 4.) The curves in Fig. 1 were obtained by solving numerically the DMFT equations, see Appendix C for details.
this section cite: ['b12', 'b12', 'b37', 'b13']

Section: Step 2:
We study this DMFT, with special attention to the large network limit m → ∞, and large sample size α → ∞, with α = α/m fixed, for a generic single index model (k = 1). We obtain a separation of time scales in the dynamics, corresponding to distinct learning regimes. Here we use mean field initialization, h(z) = (9/10)z + (1/6)z 3 , α = 0.4 and τ = 0.6. Symbols: SGD results on actual 2-layer networks with d = 200, n = αmd (averaged over 10 simulations). Continuous viridis lines: Numerical solution of the DMFT equations. Note that the second layer weights are given in terms of a scalar quantity as the result of the statistically symmetric initialization.
The analysis of the DMFT equations in the double limit m, t → ∞ is an example of singular perturbation theory [9,29]. Making this type of analysis rigorous is notoriously challenging and we proceed by a combination of numerical solutions and analytical derivations.
In the following, we will first consider the simplest possible setting, pure noise data, and subsequently consider the single-index model. The structure of the activation function and target nonlinearity will be encoded in the functions
h(q) := E{σ(G 1 )σ(G q )}, φ(q) := E{φ(G 1 )σ(G q )} ,
where G 1 , G q are standard jointly Gaussian with E{G 1 G q } = q. The relation between σ, φ and h, φ is conveniently expressed in terms of the expansions in Hermite polynomials σ(x) = k≥0 s k He k (x), φ(x) = k≥0 f k He k (x), which corresponds to the analytic expansion h(q) = k≥0 s 2 k q k , φ(q) = k≥0 s k f k q k . As mentioned above, we assume throughout n, d → ∞, with n/d → α ∈ (0, ∞), with the limit m, α → ∞ taken afterwards. To further simplify our analysis, we assume a symmetric initialization whereby a i (0) = a 0 is independent of i ≤ m and (w i (0) : i ≤ m) ∼ iid Unif(S d-1 ). Throughout, we use 'with high probability' for 'with probability converging to one as n, d → ∞.'
In Section 3 we present rigorous results that do not require either of these simplifying assumptions.
this section cite: ['b8', 'b28']

Section: Training on pure noise
We begin by the case in which the data is pure noise: y i = ε i ∼ N(0, τ 2 ). A by-now-classic experiment [60] showed that deep learning models have sufficient capacity to achieve vanishing training error even when actual labels are replaced by random ones: they 'interpolate pure noise. ' The ability of a model F Θ = (f ( • ; θ) : θ ∈ Θ) to interpolate pure noise is intimately connected to its Gaussian complexity G(F Θ ; n) := E sup θ∈Θ ⟨g, f (X; θ)⟩/n [53] (where g ∼ N(0, I n ) is independent of f (X, ; θ) = (f (x i ; θ) : i ≤ n). Indeed, interpolation is impossible unless G(F Θ ; n) ≥ τ . Viceversa, G(F Θ ; n) ≪ τ ensures good generalization.
By a theorem of [7] for the network (1.1), G(F Θ ; n) ≤ L σ ∥a/m∥ 1 d/n (with L σ depending uniquely on σ). This means that, in order to interpolate noise, the average magnitude of second layer weights must be ∥a/m∥
1 ≥ L -1 σ τ n/d = (L -1 σ α 1/2 )τ √ m.
However, complexity bounds do not have implications on the convergence of GF to an interpolator.
Figure 2 compares the DMFT predictions to simulations using SGD to train an actual two layer networks. In this figure we initialize a(0) = 1, and let a(t) evolve with GF alongside the first layer weigths. We observe that the theory describes well the empirical results, despite the Gaussian approximation in our DMFT and the difference between SGD and GF. We also observe that secondlayer weights remain roughly constant until a large time t # (m), which appears to increase with m. Roughly at the same time, train error starts to decrease and converges to zero.
In Section G.1 of the appendix, we will make precise the above picture of the evolution of a(t). Here, we consider a simplified setting in which a(t) = γ √ m with γ independent of m, not evolving with training. Note that G(F Θ ; n) ≍ γ/ √ α and hence such a network can interpolate pure noise if γ is larger than threshold depending on α. Our DMFT predicts a sharp phase transition. For α ∈ (0, 1), GF converges to vanishing train error with high probability if γ > γ GF (α, m)τ , and converges to a strictly positive training error if γ < γ GF (α, m)τ . The threshold γ GF (α, m) converges to a limit γ * GF (α) ∈ (0, 1) as m → ∞.
A rephrasing of the same phenomenon states that lim n,d→∞ R g n (θ(t)) = e tr (t; m, γ), and
lim t→∞ lim m→∞ e tr (t; m, γ 0 ) = e * (γ) > 0 for γ < γ * GF (α)τ , 0 for γ ≥ γ * GF (α)τ .
(2.4) Informally γ * GF (α) is the minimum complexity γ for a very large network to interpolate noise via gradient flow. The functions γ * GF (α), e * (γ) will play an important role below. We will next consider training on data from a single-index model. The initial scale of secondlayer weights ∥a(0)/m∥ 1 plays a crucial role and we will separately analyze lazy and mean field initializations.
this section cite: ['b59', 'b52', 'b6']

Section: Training on data with latent structure: lazy initialization
We initialize a(0) = γ 0 √ m, and let a(t) evolve according to GF alongside first-layer weights. DMFT predicts the emergence of three dynamical regimes for large m and large α (with n/d → α). For an illustration, we refer to Fig. 3.
First dynamical regime: t = O(1/m). Second layer weights do not change significantly γ(t) = γ 0 + o m (1), while first layer-weights move by ∥w i (t)w i (0)∥ = Θ(1/ √ m). Because the weights a i (t) are of order √ m, even an O(1/ √ m) change in the w i leads to a significant decrease in test error and train error.
Train and test error are close to each other. Namely, the following limits are well defined
lim n,d→∞ R g n (θ(t)) = e tr (t; φ, γ 0 , m, α) , lim n,d→∞ R g (θ(t)) = e ts (t; φ, γ 0 , m, α) .
(2.5)
with lim m→∞ e tr ( t/m; φ, γ 0 , m, α) = lim m→∞ e ts ( t/m; φ, γ 0 , m, α) =: e lz1 ( t; φ, γ 0 , α).
For large scaled time t, the error e lz1 ( t; φ, γ 0 , α) converges to the error of the best linear approximation to f * . This dynamical regime follows the qualitative predictions of NTK theory, and is essentially linear in the weights w i , but the time is too short for the model to overfit the data.
Second dynamical regime: t = Θ(1). Second layer weights do not change significantly: γ(t) = γ 0 + o m (1), while first layer weights change significantly ∥w i (t)w i (0)∥ = Θ(1). However they change orthogonally to the latent subspace U and hence the test error does not change: no actual learning takes place in this regime, but the model starts to overfit the data.
More formally, train and test error have well defined limits as the network width diverges: e lz2 tr (t; φ, γ 0 , α) := lim m→∞ e tr (t; φ, γ 0 , m, α) , e lz2 ts (t; φ, γ 0 , α) := lim m→∞ e ts (t; φ, γ 0 , m, α) . (2.6) However, the scaling function e lz2 ts (t; φ, γ 0 , α) for the test error is constant in time and equal to the value achieved at the end of the first dynamical regime. Namely
e lz2 ts (t; φ, γ 0 , α) = lim t→∞ e lz1 ( t; φ, γ 0 , α) = 1 2 τ 2 + ∥φ∥ 2 - ∥∇ φ(0)∥ 2 h ′ (0) + γ 2 0 (h(1) -h ′ (0)) . (2.7)
Since the w i 's move orthogonally to the latent space, their dynamics is equivalent (for large m) to the one in the pure noise setting, modulo a redefinition of h. The right plot in Fig. 3 illustrates this.
Third dynamical regime: t = Θ(m). The qualitative properties of this regime depend whether or not γ 0 is larger than an interpolation threshold γ * GF (α, φ, τ ), which generalizes the threshold γ * GF (α) = γ * GF (α, 0, 1) introduced in the pure noise case. Because the dynamics of weights w i in the subspace orthogonal to U is equivalent to dynamics in pure noise, we expect the interpolation threshold γ * GF (α, φ, τ ) to be given in terms of pure noise threshold γ * GF (α) as follows:
γ * GF (α, φ, τ ) = τ 2 + ∥φ∥ 2 - ∥∇ φ(0)∥ 2 h ′ (0) 1/2 γ * GF (α) . (2.8)
For γ 0 > γ * GF (α, φ, τ ), interpolation is achieved during the second dynamical regime, no further evolution takes place.
For γ 0 < γ * GF (α, φ, τ ), a non-trivial evolution takes place for t = Θ(m). Introducing the rescaled time z ∈ (0, ∞), we obtain, as m → ∞,
γ(mz) = γ lz3 (z) + o m (1
), e tr (mz) = e lz3 tr (z) + o m (1), e ts (mz) = e lz3 ts (z) + o m (1) . (2.9) Further, for large values of the rescaled time z → ∞, γ lz3 (z) grows to γ * GF (α, φ, τ ) ≈ γ * GF (α, φ, τ ), while e lz3 tr (z) decreases to 0. In other words, interpolation is achieved on this third regime. Further the test error e lz3 ts (z) increases from e lz2 ts (t; φ, γ 0 , α) to e lz2 ts (t; φ, γ * GF , α), with γ * GF = γ * GF (α, φ, τ ) whereby e lz2 ts (• • • ) is given by Eq. (2.7).
this section cite: []

Section: Training on data with latent structure: mean field initialization
We initialize a(0) = a 0 , independent of m and let second layer weights evolve. Note that at initialization the network's Rademacher complexity is small, namely of order a 0 d/n = a 0 / √ αm.
Our DMFT analyisis predicts two dynamical regimes for large m. We will refer to them as 'first' and 'third regime' for consistency with other settings ( see Sec.G.2 of the appendix). For an illustration, we refer to Figs. 4 and 5.
First dynamical regime: t = O(1). Both first and second layer weights change by order one: a(t) = a 0 + Θ(1) and ∥w i (t)w i (0)∥ = Θ(1). and as a consequence test and train error decrease significantly. In this regime, the two errors remain close to each other and their evolution is well captured by the mean field theory of [38,14], as specialized to the case of spherically invariant distributions [10,2].
Namely, lim m→∞ a(t) = a mf1 (t), lim m→∞ v(t) = v mf1 (t), and DMFT reduces to a system of k + 1 ordinary differential equations for the k + 1 scalar variables (
a mf1 (t), v mf1 (t)) ∂ t v mf1 (t) = αa mf1 (t)Q v mf1 (t) ∇ φ(v mf1 (t)) -a mf1 (t)h ′ (∥v mf1 (t)∥ 2 )v mf1 (t) , ∂ t a mf1 (t) = α φ(v mf1 (t)) -αa mf1 (t)h(∥v mf1 (t)∥ 2 ) ,(2.10)
where Q v := I kvv T . As mentioned above, train and test error coincide in the large width limit 10 0 10 1 10 2 10 3 t 10 0 10 1 a m = 2 5 m = 2 6 m = 2 7 m = 2 8 m = 2 9 10 0 10 1 10 2 10 3 t 0.0 0.2 0.4 0.6 0.8 Train/Test m = 2 5 m = 2 6 m = 2 7 m = 2 8 m = 2 9 An explicit formula for e mf1 (t) is given in Appendix G.2.1. In the case k = 1 and φ(z) = h(z), we have that a mf1 = 1, v mf1 = 1 is a fixed point of Eq. (2.10), and indeed the only fixed point with v mf1 > 0. If h ′ (0) > 0, then, we have (a mf1 (t), v mf1 (t)) → (1, 1) as t → ∞, and therefore test and train error converge to the Bayes error e mf1 (t) → τ 2 /2. This is significantly smaller than the test error achieved with lazy initialization. The separation between lazy and mean-field initialization is expected because feature learning takes place in the mean field regime.
Third dynamical regime: t = Ω(m). Computing the local stability of DMFT solutions around the mean field asymptotics (see Appendix G.2.2) suggests that the latter breaks down for t = Θ(m). For t ≳ m, we observe that the second layer weights grow to achieve a(t) ≍ √ m, the projection onto
where γ mf3 (z), e mf3 tr (z), e mf3 ts (z) are scaling functions describing the dynamics on this timescale. We expect t 0 (m; c) = t * (c)m + o(m), and w(m) ≲ t 0 (m; c), but our numerical solutions are not sufficient to determine the precise scaling. On the other hand, it appears that at large times, the complexity converges close the interpolation threshold:
lim z→∞ γ mf3 (z) = γ * GF (α, φ, τ ) ≈ γ * GF (α, φ, τ ) . (2.12)
Finally, the evolution of train and test error for a(t) ≍ √ m appears to match the behavior at fixed second-layer weights. Namely, we define two functions ε mf tr/ts (γ) := lim m→∞ e tr/ts (t 0 (m; γ), m) .
(2.13)
We observe that the limit curves (γ, ε mf tr (γ)), (γ, ε mf ts (γ)), match closely asymptotic train and test error obtained by fixing a(t) = γ √ m, and not letting second-layer weight evolve. This confirms the hypothesis that γ(t) is a slow variable, while others converge as if γ was fixed.
this section cite: ['b37', 'b13', 'b9', 'b1']

Section: Lower bounding the overfitting timescale
In this section we rigorously establish two results that confirm elements of the scenario outlined in the previous sections. We emphasize that the result presented here are non-asymptotic, i.e. hold at finite
0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 10 -3 10 -2 10 -1 1 10 10 2 a(t)/ √ m t/m m = 2 3 m = 2 4 m = 2 5 m = 2 6 m = 2 7 m = 2 8 m = 2 9 m = 2 10 m = 2 11 m = 2 12 m = 2 13 m = 2 14 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0.001 0.01 0.1 1 10 100 v(t) t/m m = 2 3 m = 2 4 m = 2 5 m = 2 6 m = 2 7 m = 2 8 m = 2 9 m = 2 10 m = 2 11 m = 2 12 m = 2 13 m = 2 14 GF and the curves appear to converge to that limit. Center: the projection of the first layer weights on the latent space in the single index model as a function of time on timescales of order m. Right: difference between test and train error as a function of the second layer weights on the scale √ m. The finite m curve are approaching a scaling curve which coincides with the one obtained by evaluating the same quantity but with a lazy initialization and fixed second layer weights.
n, m, d modulo unspecified absolute constants. Further, we do not assume a symmetric initialization of the weights. Throughout this section setting, it is more convenient to rescale time defining t = tα.
Hence. instead of the flow (1.2), we study
θ( t) = -mP θ ∇ R n (θ( t)) .(3.1)
For α = Θ(1) the parametrizations t and t are equivalent.
The first result of this section implies that (under mean field initialization) overfitting cannot take place on times of order one. Theorem 1. Under the GF dynamics (1.2), and the data distribution in the introduction (with k arbitrary), further assume ∥σ∥ Lip , ∥σ∥ ∞ ≤ L, |φ(0)|, ∥φ∥ Lip ≤ L, ∥a(0)∥ ∞ ≤ a 0 , for some a 0 ≥ 1 and that the w i (0), i ≤ m are independent of the data {(y i , x i ) : i ≤ n}. Finally assume n ≥ d ∨ m.
Then, there exist universal constants C 0 , C 1 , and the following holds for all t ≥ 0,
∥a( t)∥ ∞ ≤ a 0 + a 1 t , a 1 := C 0 L(τ + √ k + a 0 L) , (3.2) R(a( t), W ( t)) -R n (a( t), W ( t)) ≤ C 1 (L 2 (a 0 + a 1 t) 2 + τ 2 ) • d n . (3.3)
Under mean field initialization, a 0 is a fixed constant and hence a 1 is also bounded, whence the generalization error in Eq. ( 3.3) is small as long as t = o((n/d) 1/4 ) (equivalently, for α fixed, t = o(m 1/4 )).
By itself, this result implies a separation of timescales between learning and overfitting, thus confirming the picture developed within DMFT, but falls short of characterizing the overfitting timescale.
The second result implies that, up to time-scale of order one, the dynamics is closely tracked by the mean field equations (2.10). Since the a i (0) at initialization are not necessarily all equal, these are generalized as
∂ tv mf1 i ( t) = a mf1 i ( t)Q v mf1 i ( t) ∇ φ(v mf1 i ( t)) - 1 m m j=1 a mf1 j ( t)h ′ (⟨v mf1 i ( t), v mf1 j ( t)⟩)v mf1 j ( t) , ∂ ta mf1 i ( t) = φ(v mf1 i ( t)) - 1 m m j=1 a mf1 j ( t)h(⟨v mf1 i ( t), v mf1 j ( t)⟩) .
(3.4)
The mean field prediction for test error is the same as for training error and given by
e ts ( t) = 1 2 ∥φ∥ 2 L 2 - 1 m m j=1 a mf1 j ( t) φ(v mf1 j ( t)) + 1 2m 2 m j=1 a mf1 i ( t)a mf1 j ( t) h(⟨v mf1 i ( t), v mf1 j ( t)⟩)
Theorem 2. Under the the GF dynamics (1.2), and the data distribution in the introduction (with k arbitrary), further assume that ∥φ∥ ∞ ,
∥φ ′ ∥ ∞ , ∥φ ′ ∥ Lip ≤ L, ∥σ∥ ∞ , ∥σ ′ ∥ ∞ , ∥σ ′ ∥ Lip ≤ L. Further assume |a i (0)| ≤ L for all i ≤ m, (w i (0)) i≤m ∼ iid Unif(S d-1
). Then for any δ > 0 there exist constants c 0 c 1 , C depending on L, τ, δ, k such that, letting T lb = c 0 (log m) 1/3 ∧ (log n/d) 1/3 , the following happens with probability at least 1 -2 exp(-c 1 d),
sup t≤T lb 1 m m i=1 |a i ( t) -a mf1 i ( t)| + ∥v i ( t) -v mf1 i ( t)∥ ≤ C 1 m ∨ 1 d ∨ d n 1/2-δ ,(3.5)
sup t≤T lb R(a( t), W ( t)) -e ts ( t) ≤ C 1 m ∨ 1 d ∨ d n 1/2-δ .
(3.6) Remark 3.1. While the analysis in the previous section requires m → ∞ after n, d → ∞, neither Theorem 3.1 nor Theorem 3.2 make the assumption. In particular, Eq. ( 3.3) implies that the generalization error is small for t = o((n/d) 1/4 ) irrespective of m.
Similarly, Eqs. (3.5), (3.6) imply that the mean field theory of [38,14,45] captures well the evolution of the system for times t = o((log m) 1/3 ∧ (log n/d) 1/3 ).
this section cite: ['b37', 'b13', 'b44']

Section: Discussion
We conclude by highlighting a few qualitative conclusions of our work, and how they address questions raised in Section 1. In the following remarks, we consider α = n/md as constant.
Interpolation mechanism. In the current setting, the neural model complexity is proportional to ∥a(t)∥ 1 / √ m = γ(t)+o n (1). We observe two alternative scenarios. If the complexity at initialization is large enough γ 0 > γ * GF (α)τ , then the gradient flow rapidly converges to a near interpolator without significant change in γ(t). If instead, γ 0 < γ * GF (α)τ , then γ(t) grows to reach the interpolation threshold at which point the training error converges to 0.
this section cite: []

Section: References
Ref_id:b0 Title: The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks Year: (2022)
Ref_id:b1 Title: From highdimensional and mean-field dynamics to dimensionless odes: A unifying approach to sgd in two-layers networks Year: (2023)
Ref_id:b2 Title: Implicit regularization in deep matrix factorization Year: (2019)
Ref_id:b3 Title: Online stochastic gradient descent on non-convex losses from high-dimensional inference Year: (2021)
Ref_id:b4 Title: Highdimensional asymptotics of feature learning: How one gradient step improves the representation Year: (2022)
Ref_id:b5 Title: Hidden progress in deep learning: SGD learns parities near the computational limit Year: (2022)
Ref_id:b6 Title: For valid generalization the size of the weights is more important than the size of the network Year: (1996)
Ref_id:b7 Title: Cugliandolo-kurchan equations for dynamics of spin-glasses. Probability theory and related fields Year: (2006)
Ref_id:b8 Title: Perturbation theory of dynamical systems Year: (2001)
Ref_id:b9 Title: Learning time-scales in two-layers neural networks Year: (2024)
Ref_id:b10 Title: Regularization and complexity control in feed-forward networks Year: (1995)
Ref_id:b11 Title: Self-consistent dynamical field theory of kernel evolution in wide neural networks Year: (2022)
Ref_id:b12 Title: The high-dimensional asymptotics of first order methods with random data Year: (2021)
Ref_id:b13 Title: On the global convergence of gradient descent for overparameterized models using optimal transport Year: (2018)
Ref_id:b14 Title: Implicit bias of gradient descent for wide two-layer neural networks trained with the logistic loss Year: (2020)
Ref_id:b15 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b16 Title: The spherical p-spin interaction spin-glass model: the dynamics Year: (1993)
Ref_id:b17 Title: Recent applications of dynamical mean-field methods Year: (2023)
Ref_id:b18 Title: Full dynamical solution for a spherical spin-glass model Year: (1995)
Ref_id:b19 Title: Analytical solution of the off-equilibrium dynamics of a long-range spin-glass model Year: (1993)
Ref_id:b20 Title: Neural networks can learn representations with gradient descent Year: (2022)
Ref_id:b21 Title: Gradient descent finds global minima of deep neural networks Year: (2019)
Ref_id:b22 Title: Rethinking mean-field glassy dynamics and its relation with the energy landscape: The surprising case of the spherical mixed p-spin model Year: (2020)
Ref_id:b23 Title: Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors) Year: (2000)
Ref_id:b24 Title: A spin glass model for reconstructing nonlinearly encrypted signals corrupted by noise Year: (2019)
Ref_id:b25 Title: Optimization landscape in the simplest constrained random least-square problem Year: (2022)
Ref_id:b26 Title: Linearized two-layers neural networks in high dimension Year: ()
Ref_id:b27 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b28 Title: Introduction to Perturbation Methods Year: (2013)
Ref_id:b29 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b30 Title: Dynamical mean field theory for models of confluent tissues and beyond Year: (2023)
Ref_id:b31 Title: Stochastic gradient descent outperforms gradient descent in recovering a high-dimensional signal in a glassy energy landscape Year: (2023)
Ref_id:b32 Title: On the topology of solutions to random continuous constraint satisfaction problems Year: (2024)
Ref_id:b33 Title: Towards explaining the regularization effect of initial large learning rate in training neural networks Year: (2019)
Ref_id:b34 Title: Passed & spurious: Descent algorithms and local minima in spiked matrix-tensor models Year: (2019)
Ref_id:b35 Title: A vector-contraction inequality for Rademacher complexities Year: (2016)
Ref_id:b36 Title: Generalization error of random feature and kernel methods: hypercontractivity and kernel matrix concentration Year: (2022)
Ref_id:b37 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b38 Title: Spin glass theory and beyond Year: (1987)
Ref_id:b39 Title: Dynamical mean-field theory for stochastic gradient descent in gaussian mixture classification Year: (2020)
Ref_id:b40 Title: The effective noise of stochastic gradient descent Year: (2022)
Ref_id:b41 Title: Solving overparametrized systems of random equations: I. model and algorithms for approximate solutions Year: (2023)
Ref_id:b42 Title: On Smale's 17th problem over the reals Year: (2024)
Ref_id:b43 Title: Generalization and parameter estimation in feedforward nets: Some experiments Year: (1989)
Ref_id:b44 Title: Trainability and accuracy of artificial neural networks: An interacting particle system approach Year: (2022)
Ref_id:b45 Title: The threshold energy of low temperature Langevin dynamics for pure spherical spin glasses Year: (2024)
Ref_id:b46 Title: Understanding machine learning: From theory to algorithms Year: (2014)
Ref_id:b47 Title: The implicit bias of gradient descent on separable data Year: (2018)
Ref_id:b48 Title: Concentration for the zero set of random polynomial systems Year: (2023)
Ref_id:b49 Title: Mean field models for spin glasses: Volume I: Basic examples Year: (2010)
Ref_id:b50 Title: A continuous constraint satisfaction problem for the rigidity transition in confluent tissues Year: (2023)
Ref_id:b51 Title: Attention is all you need Year: (2017)
Ref_id:b52 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b53 Title: Limitations of the ntk for understanding generalization in deep learning Year: (2022)
Ref_id:b54 Title: The marginal value of adaptive gradient methods in machine learning. Advances in neural information processing systems Year: (2017)
Ref_id:b55 Title: Kernel and rich regimes in overparametrized models Year: (2020)
Ref_id:b56 Title: On early stopping in gradient descent learning Year: (2007)
Ref_id:b57 Title: On the power and limitations of random features for understanding neural networks Year: (2019)
Ref_id:b58 Title: How does learning rate decay help modern neural networks Year: (2019)
Ref_id:b59 Title: Understanding deep learning (still) requires rethinking generalization Year: (2021)
Ref_id:b60 Title: Boosting with early stopping: Convergence and consistency Year: (2005)
Ref_id:b61 Title: Quantum field theory and critical phenomena Year: (2021)
