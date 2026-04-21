Title: Temperature is All You Need for Generalization in Langevin Dynamics and other Markov Processes
Abstract: We analyze the generalization gap (gap between the training and test errors) when training a potentially over-parametrized model using a Markovian stochastic training algorithm, initialized from some distribution θ 0 ∼ p 0 . We focus on Langevin dynamics with a positive temperature β -1 , i.e. gradient descent on a training loss L with infinitesimal step size, perturbed with β -1 -variances Gaussian noise, and lightly regularized or bounded. There, we bound the generalization gap, at any time during training, by (βEL(θ 0 ) + ln(1/δ))/N with probability 1 -δ over the dataset, where N is the sample size, and EL(θ 0 ) = O(1) with standard initialization scaling. In contrast to previous guarantees, we have no dependence on either training time or reliance on mixing, nor a dependence on dimensionality, gradient norms, or any other properties of the loss or model. This guarantee follows from a general analysis of any Markov process-based training that has a Gibbs-style stationary distribution. The proof is surprisingly simple, once we observe that the marginal distribution divergence from initialization remains bounded, as implied by a generalized second law of thermodynamics.

Section: Introduction
One main goal of contemporary machine learning theory is to predict a model's behavior before training occurs. A commonly desired metric is the generalization of overparameterized models, such as neural networks (NN). For these models, such a predictive theory of generalization is still lacking, despite great empirical success [71,23]. In particular, a significant line of work aimed to explain the role of optimization in generalization (e.g. [23,64,40,66]), and specifically the effect of stochasticity (e.g. [59,49,10,8]). Data-dependent Markov processes are a common optimization approach. These include stochastic gradient descent (SGD), as well as other stochastic gradient methods either studied theoretically [30,59], or used in practice such as SGD with momentum [52], ADAM [34], and many more. Of particular interest are continuous Langevin dynamics (CLD) and discrete analogues of it, which have been studied extensively as models for SGD (see Section 4.1).
In Section 2 we develop, for the first time, a generalization bound applicable to any data-dependent Markov process with a Gibbs-type stationary distribution (i.e. whose finite density exists and is nonzero w.r.t. some data-independent base measure). An important feature of our analysis is that it is entirely independent of the training time t, both in that we do not rely on training for only a small number of steps, nor that we rely on mixing -the guarantees are valid at any time, with no dependence at all on t. Furthermore, it is also completely trajectory independent.
In Section 3 we apply these general results to the particular case where training is done with CLD with loss L and inverse temperature β, deriving a particularly simple generalization bound for CLD, which we compare to previous generalization bounds for CLD in Section 4, as well as discussing other related work. Finally, we address limitations and future work in Section 5.
To prove these results, we first show in Section 2 how, for the marginal distribution at time t, p t , its divergence (either KL or the Rényi infinity divergence) from initialization is bounded due to its monotonicity, i.e. a generalized second law of thermodynamics [11,46]. This surprisingly simple derivation 2 leads to our key technical result (Corollary 2.5). Standard PAC-Bayes generalization bounds [43] then yield our generalization bounds (Theorem 2.7 and Corollary 3.1).
this section cite: ['b70', 'b22', 'b22', 'b63', 'b39', 'b65', 'b58', 'b48', 'b9', 'b7', 'b29', 'b58', 'b51', 'b33', 'b10', 'b45', 'b42']

Section: Generalization Bounds for General Markov Process
In this Section, we consider general data-dependent Markov processes over predictors and obtain a bound on their generalization gap. Importantly, although the bound only depends on the initialization distribution and a stationary distribution, it will apply to predictors at any time t ≥ 0 along the Markov process. Our main goal is to apply these bounds to stochastic training methods, such as Langevin dynamics, where the iterates form a data-dependent Markov process. But to emphasize the broad generality of the results, in this section we consider a generic stochastic optimization framework and general data-dependent Markov processes.
We obtain generalization guarantees by bounding the KL-divergence (or, for high probability bounds, the Rényi infinity divergence, see Definition 2.1) between the data-dependent marginal distribution p t of the predictors at time t, and some data-independent base measure ν (the PAC-Bayes "prior"). The crux of the analysis is therefore bounding the divergence between p t and ν, based only on assumptions on the initial distribution p 0 (specifically, the divergence between p 0 and ν) and a stationary distribution p ∞ (specifically, requiring that p ∞ can be expressed as a Gibbs distribution with bounded potential or expected potential, see Definition 2.2) -we do this in Section 2.1. Then, in Section 2.2 we plug these bounds on the divergence between p t and ν into standard PAC-Bayes bounds to obtain the desired generalization guarantees.
Detailed proofs of all the results in this section can be found in Appendix B.
this section cite: []

Section: Bounding the Divergence of a Markov Process
In this subsection, we consider a general time-invariant Markov processfoot_0 h t ∈ H over a state space H. The Markov process can be either in discrete or continuous time, i.e. we can think of t as either an integer or a real index. We denote by p t the marginal distribution at time t, i.e. h t ∼ p t . We do not assume that the Markov process is ergodic, and all our results will rely on the existence of some stationary distribution p ∞ . The main goal of this subsection is to bound the divergence D (p t ∥ ν) between the marginal distribution at time t and some reference distribution ν. We can think of a bound on the divergence as ensuring high entropy relative to ν, or in other words that p t does not concentrate too much relative to ν, i.e. does not have too much probability mass in a small ν-region. We present all bounds for both the KL-divergence KL (p ∥ q) and the Rényi infinity divergence D ∞ (p ∥ q), defined below.
Divergences and Gibbs distributions. We recall the definitions of our two divergences, and also relate them to the Gibbs distribution. It will also be convenient for us to introduce "relative" versions of divergences.
2 e.g. to bound the KL divergence of a Markov process having a stationary distribution with potential Ψ ∈ [0, ∞), i.e. dp∞/ dp0 ∝ e -Ψ (e.g., Ψ = βL for CLD), the second law implies the first inequality below
KL(pt||p0) = pt ln pt p0 = pt ln pt p∞ + pt ln p∞ p0 ≤ p0 ln p0 p∞ + pt ln p∞ p0 = Ep 0 Ψ -Ep t Ψ ≤ Ep 0 Ψ.
Definition 2.1 (Divergences 4 ). For probability distributions p, q and µ:
1. The µ-weighted Kullback-Leibler (KL) divergence (a.k.a. relative cross-entropy) is 5KL µ (p ∥ q) = dµ ln dp dq , and the KL-divergence is then KL (p ∥ q) = KL p (p ∥ q).
2. The Rényi infinity divergence isfoot_3 D µ ∞ (p ∥ q) = ess sup µ ln dp dq , with D ∞ (p ∥ q) = D p ∞ (p ∥ q). Definition 2.2 (Gibbs distribution). A distribution p is Gibbs w.r.t. a base distribution q with potential Ψ : H → R if Z = e -Ψ dq < ∞ and dp = Z -1 e -Ψ dq .
Claim 2.3. If p, q, µ, ν are probability measures, and p is Gibbs w.r.t. q with potential Ψ < ∞, then
1. KL µ (p ∥ q) + KL ν (q ∥ p) = E ν Ψ -E µ Ψ, 2. D µ ∞ (p ∥ q) + D ν ∞ (q ∥ p) = ess sup ν Ψ -ess inf µ Ψ.
So, KL (p ∥ q) + KL (q ∥ p) = E q Ψ -E p Ψ, and D ∞ (p ∥ q) + D ∞ (q ∥ p) = ess sup q Ψ -ess inf p Ψ.
That is, the potential of a Gibbs distribution p allows us to bound the divergence in both directions between p and the base measure q. A generalized converse of Claim 2.3 also holds, and we have that bounding on the symmetrized divergences (but not just on one direction!) is also sufficient for p being Gibbs with a bounded potential. 7Second Law of Thermodynamics. Central to our analysis is the following monotonicity property on the divergence between the marginal distribution of a Markov process and any stationary distribution. Claim 2.4 (Cover's Second Law of Thermodynamics). Let p t be the marginal distribution of a time-invariant Markov process, and p ∞ a stationary distribution for the transitions of the Markov process (the process need not be ergodic, and p t need not converge to p ∞ ). Then for any t ≥ 0
KL (p t ∥ p ∞ ) ≤ KL (p 0 ∥ p ∞ ) and D ∞ (p t ∥ p ∞ ) ≤ D ∞ (p 0 ∥ p ∞ ) .
When the stationary distribution is uniform (thus having maximal entropy), the KL-form of Claim 2.4 recovers the familiar second law of thermodynamics, i.e. that the entropy is monotonically nondecreasing. The more general form, as in Claim 2.4, is a direct consequence of the data processing inequality, as pointed out by Theorem 4 of Cover [11] (see also [12,46] and the generalization to Rényi divergences in [65,Theorem 9 and Example 2] -for completeness we provide a proof in Appendix A.2).
In our case, the stationary distribution p ∞ will not be uniform, but rather will be very data-dependent (we are interested mostly in processes that aim to optimize some data-dependent quantity, such as Langevin dynamics). Nevertheless, we do want to use Claim 2.4 to control the entropy of p t relative to some benign data-independent base distribution ν (which we can informally think of as "uniform").
To do so, we can use the chain rule and plug in Claim 2.4 to obtain that for any distribution ν and at any time t we have (see Lemma B.1 in Appendix B for the full derivation):
KL (p t ∥ ν) = KL (p t ∥ p ∞ ) + KL pt (p ∞ ∥ ν) ≤ KL (p 0 ∥ p ∞ ) + KL pt (p ∞ ∥ ν) = KL (p 0 ∥ ν) + KL p0 (ν ∥ p ∞ ) + KL pt (p ∞ ∥ ν) , (1
)
and similarly,
D ∞ (p t ∥ ν) ≤ D ∞ (p 0 ∥ ν) + D p0 ∞ (ν ∥ p ∞ ) + D pt ∞ (p ∞ ∥ ν) .(2)
Bounding the last two terms in (1) and (2) using Claim 2.3 we obtain the main result of this subsection:
Corollary 2.5. For any distribution ν and any time-invariant Markov process, and any stationary distribution p ∞ that is Gibbs w.r.t. ν with potential Ψ ≥ 0 (the Markov chain need not be ergodic, and need not converge to p ∞ ), at any time t ≥ 0:
KL (p t ∥ ν) ≤ KL (p 0 ∥ ν) + E p0 Ψ -E pt Ψ ≤ KL (p 0 ∥ ν) + E p0 Ψ (3
) D ∞ (p t ∥ ν) ≤ D ∞ (p 0 ∥ ν) + ess sup p0 Ψ (4
)
The important feature of Corollary 2.5 is that it bounds the divergence at any time t, in terms of a right-hand side that depends only on the initial distribution p 0 and a stationary distribution p ∞ . Interpreting the divergence D (p t ∥ ν) as a measure of concentration, the Corollary ensures that at no point during its run, and regardless of mixing, does the Markov process concentrate too much, and it always maintains high entropy (relative to the base measure ν). Remark 2.6. In order to bound the divergence D (p t ∥ ν) at finite time t, it is not enough to rely only on the divergences D (p 0 ∥ ν) and D (p ∞ ∥ ν) from the initial and stationary distributions, and it is necessary to rely also on the reverse divergence D (ν ∥ p ∞ ) -see Appendix C.
this section cite: ['b10', 'b11', 'b45', 'b64']

Section: From Divergences to Generalization
Corollary 2.5 can be directly used to obtain PAC-Bayes type generalization guarantees. Specifically, we consider a generic stochastic optimization setting specified by a bounded instantaneous objective f : H×Z → [0, 1] over a class H, which we will refer to as the "predictor" class, and instance domain Z. For example, in supervised learning Z = X × Y, H ⊆ Y X and f (h, (x, y)) = I {h(x) ̸ = y} measures the error of predicting h(x) when the correct label is y. For a source distribution D over Z and data S ∼ D N of size N we would like to relate the population and empirical objectives
E D (h) = E z∼D [f (h, z)] E S (h) = 1 N z∈S f (h, z).(5)
In our case, we are interested in predictors generated by a data-dependent Markov process h t .
That is, conditioned on the data S, {h t } t≥0 is a time-invariant Markov process, specified by some (possibly data-dependent) initial distribution p 0 (h 0 ; S), and a transition distribution that would also depend on the data S, and specifies a (randomized) rule for generating the next iterate h t+1 (if in discrete time) from the current iterate h t and the data S (as in, e.g., stochastic gradient descent or stochastic gradient Langevin dynamics; SGLD).
We present two types of generalization guarantees: guarantees that hold in expectation over a draw from the Markov process ((6) below) and guarantees that hold with high probability over a single draw from the Markov process (as in (7), e.g. a single run of CLD). In both cases, the guarantees hold with high probability over the training set.
Theorem 2.7. Consider any distribution D over Z, function f : H × Z → [0, 1], sample size N ≥ 8, and any distribution ν over H. Let {h t ∈ H} t≥0 be a discrete or continuous time process (i.e. t ∈ Z + or t ∈ R + ) that is time-invariant Markov conditioned on S, that starts from an initial distribution p 0 (•; S) (that may depend on S), and admits a stationary distribution conditioned on S, p ∞ (•; S). Let Ψ S (h) ≥ 0 be a non-negative potential function and assume that p ∞ (•; S) is Gibbs w.r.t. ν with potential Ψ S . Then: 1. with probability
1 -δ over S ∼ D N , E [E D (h t ) -E S (h t )|S] ≤ KL (p 0 (•; S) ∥ ν) + E [Ψ S (h 0 )|S] + ln N /δ 2N ,(6)
2. with probability 1 -δ over S ∼ D N and over h t :
E D (h t ) -E S (h t ) ≤ D ∞ (p 0 (•; S) ∥ ν) + ess sup h∼p0(•;S) Ψ S (h) + ln N /δ 2N .(7)
Proof. The Theorem follows immediately by plugging the divergence bounds of Corollary 2.5 into standard PAC-Bayes guarantees, which we do in Appendix B.
Remark 2.8. A simplified variant of Theorem 2.7 can be stated when the initial distribution p 0 is data-independent and always equal to ν. In this case the divergence between p 0 and ν vanishes, and ( 6) and ( 7) become
E [E D (h t ) -E S (h t )|S] ≤ E p0 [Ψ S | S] + ln N /δ 2N , E D (h t )-E S (h t ) ≤ ess sup p0 Ψ S + ln N /δ 2N .(8)
But allowing p 0 ̸ = ν is more general, as it both allows using a data-dependent initialization (recall that ν must be data independent) and it allows initializing to a distribution where D (p ∞ ∥ p 0 ) is infinite -e.g., we can allow initializing to a degenerate initial distribution p 0 whose support is a strict subset of the support of p ∞ (in which case p ∞ will definitely not be Gibbs w.r.t. p 0 ), as long as the ν-mass of the support of p 0 is not too small. Remark 2.9. In Theorem 2.7, the Markov process need not be ergodic, and need not converge to p ∞ , or converge at all. If there are multiple stationary distributions, the theorem holds for all of them, and so we can take p ∞ to be any stationary distribution we want. And in any case, there is no mixing requirement, and the theorem holds at any time t. Remark 2.10. Our data-dependent Markov process of interest, and in particular CLD and SGD, might aim to minimize E S (h t ), and the potential Ψ might also be related to it (as in, e.g., CLD). This is allowed, but is in no way required in Theorem 2.7. Even for CLD, these might be related but not the same, as we might be minimizing a surrogate loss, such as a logistic loss, but are interested in bounding the generalization gap for a zero-one error. In stating Theorem 2.7 we intentionally refer to an arbitrary stochastic optimization problem and an arbitrary data-dependent Markov process, that are allowed to be related or dependent in arbitrary ways. Remark 2.11. In Appendix C we show that in order to ensure generalization at every intermediate t, it is not sufficient to only bound KL (p ∞ ∥ ν) or D ∞ (p ∞ ∥ ν), and we do need the stronger symmetric bound ensured by the Gibbs potential and Claim 2.3; and that it is also necessary to relate both p 0 and p ∞ to the same data independent distribution ν, as relating them to different data-independent distributions ensures generalization at the beginning and at the end, but not the middle of training. Remark 2.12. In Theorem 2.7 we plugged Corollary 2.5 into a simplified PAC-Bayes bound that allows for easy interpretation and comparison with other results. But once we have the divergence bounds of Corollary 2.5, we can just as easily plug them into tighter PAC-Bayes bounds -see Appendix B. For example, when E S (h t ) ≈ 0, these yield a rate of O (1/N ).
this section cite: ['b6']

Section: Special Case: Continuous Langevin Dynamics
Clearly, given Theorem 2.7 all we need to do in order to derive explicit generalization bounds for any Markovian training procedure, is to find a stationary distribution, and bound its potential (or its expectation at p 0 ). In this section, we will exemplify our results in a few special cases of continuous-time Langevin dynamics (CLD), a commonly studied approximation for NN training with "infinitesimal learning rate" (e.g. [41], see Section 4.1 for additional references), which have a normalized stationary distribution that we can write analytically.
this section cite: ['b40']

Section: Additional notation.
In the following, it will be convenient to consider a parametric model. Specifically, we assume that there exists some parameter space Θ ⊆ R d that parameterizes a hypothesis class H ⊆ Y X via a mapping Θ ∋ θ → h θ ∈ H, and assume Markovian dynamics in parameter space, instead of in the hypothesis space (note that Markov processes in parameter space may not be Markovian in hypothesis space, but the same generalization results apply ). We shall also use, with some abuse of notation, φ (θ) = φ (h θ ) for any data-dependent or data-independent function φ over hypotheses, e.g. a training loss/objective L S w.r.t a training set S. Finally, we use C 2 to denote the space of twice continuously differentiable functions on Θ.
this section cite: []

Section: CLD in a bounded domain.
Let Θ be a box in R d , and suppose that training is modeled with CLD in a bounded domain, i.e. that the parameters evolve according to the stochastic differential equation with reflection at the boundary (SDER)
dθ t = -∇L S (θ t ) dt + 2β -1 dw t + dr t ,(9)
where L S ≥ 0 is twice continuously differentiable, w t is a standard Brownian motion, and r t is a reflection process that constrains θ t within Θ. Such weight clipping is quite common in practical scenarios such as NN training. For simplicity, we assume that r t has normal reflection, meaning that the reflection is perpendicular to the boundary. An established result in the analysis of SDERs states that under these assumptions (9) has a stationary distribution p ∞ (θ) ∝ e -βL S (θ) I Θ {θ} (see Appendix H.2). Thus, when p 0 = Uniform (Θ), we have p 0 = ν.
Regularized CLD in R d . Suppose that the parameters evolve according to the stochastic differential equation (SDE) with weight decay (i.e. ℓ 2 regularization)
dθ t = -∇L S (θ t ) dt -λβ -1 θ t dt + 2β -1 dw t ,(10)
where L S ≥ 0 is twice continuously differentiable, w t is a standard Brownian motion. Such weight decay is also quite common in practical scenarios such as NN training. Similar to the previous case, with the regularization and twice continuous differentiability of L S this process has a unique stationary distribution p ∞ (θ) ∝ e -βL S (θ) ϕ λ (θ), where ϕ λ is the density of the multivariate Gaussian N 0, λ -1 I d . Thus, when p 0 = N 0, λ -1 I d , we also have p 0 = ν.
We can now formulate a generalization bound for both cases. Corollary 3.1. Assume that the parameters evolve according to either (9) with p 0 = Uniform (Θ), or (10) with p 0 = N 0, λ -1 I d . Then for any time t ≥ 0, and δ ∈ (0, 1),
1. w.p. 1 -δ over S ∼ D N , E θt∼pt [E D (θ t ) -E S (θ t ) | S] ≤ βE θ∼p0 [L S (θ) | S] + ln (N/δ) 2N .(11)
2. w.p. 1 -δ over S ∼ D N and θ t ∼ p t E D (θ t ) -E S (θ t ) ≤ β ess sup p0 L S (θ) + ln (N/δ) 2N . (12
)
The proof is simple -by assumption, in both cases p 0 = ν so D ∞ (p 0 ∥ ν) = 0. The rest is a direct substitution into Theorem 2.7, and in particular, using βL S as potential Ψ S .
this section cite: []

Section: Interpreting Corollary 3.1
Corollary 3.1 raises questions on the relevance of this setting, which we address below: (1) How large is E p0 L S (θ) in practically relevant cases? (2) Can we attribute the generalization to the regularization (either with the ℓ 2 regularization term, or the bounded domain)? (3) Can models successfully train in the presence of noise with a variance large enough to make the bounds non-vacuous?
Magnitude of the initial loss. Commonly, the dependence on E p0 L S (θ) with realistic p 0 and L S is relatively mild. For example, using standard initialization schemes, Gaussian process approximations [50,42,35,25] imply that the output of an infinitely wide fully connected neural network converges to a Gaussian with mean 0 and O(1) variance at initialization. So in many cases E p0 L S (θ) = O(1), such as for the scalar square and logistic losses. In the multi-output case, E p0 L S (θ) may also depend on the number of outputs (e.g., logarithmically so in softmax-cross-entropy). A more difficult question is concerned with the case that ess sup p0 L S = ∞, which is common when p 0 has infinite support. This can be mitigated by clipping the loss, which is standard in practice (e.g. in reinforcement learning [48,62]) and in the theory of optimization [37,33]. Moreover, this clipping can be done in a differentiable way (e.g. using either softmin, tanh (e.g. c • tanh(L/c)), etc) and at values only slightly higher than the typical loss at the initialization (since the loss is roughly monotonically decreasing in CLD with small noise, the optimization process would typically operate below the clipping and will not be affected by it).
this section cite: ['b49', 'b41', 'b34', 'b24', 'b47', 'b61', 'b36', 'b32']

Section: Magnitude of regularization.
In the above result we must use regularization (or a bounded domain) that matches the initialization p 0 (this can be somewhat relaxed, see Section 3.2). The same assumption, that the regularization matches the initialization, was also made in other theoretical works on CLD [49,38,19]. Note that, NN models regularized this way remain highly expressive, both empirically (Appendix F) and theoretically (Appendix G), and therefore we cannot use this regularization alone, together with classical uniform convergence approaches to show generalization. Intuitively, this is because the regularization term can be tiny, for example, in (10) the regularization term is divided by β. Therefore, when β = O (N ) (which is sufficient for a non-vacuous result), p 0 = ν, and we use a standard deep nets initialization distribution p 0 (e.g., [21,28], where λ ∝ layer width), the regularization coefficient is O layer width N that is rather small in realistic cases. Therefore, we found (empirically) that it does not seem to have a large effect at practical timescales. In addition, one can always increase the regularization by modifying the loss L S ← L S + c ∥θ∥ 2 in (10). Under standard initializations, this changes the loss in the bound by an O(c d) factor, where d is the depth of the neural network and so c d is small, for common values of c and d. Therefore, combining these observations, we do not see the magnitude of the regularization as a significant practical issue.
Magnitude of noise: theoretical perspective. In the above result we must use β = O (N ) to obtain a non-vacuous bound. This requirement is standard in many theoretical works. For example, as we will discuss below in Section 4.1, all previous generalization bounds for CLD and SGLD also required, to generalize well, β = O (N ) and potentially much worse (lower β). In addition, other theoretical works on noisy training also typically had β = O (N ) or worse. For example, when considering the ability of noisy gradient descent to escape saddle points, Jin et al. [30] uses noise sampled uniformly from a ball with a radius that depends on the dimensionality and smoothness of the problem, and thus cannot decay with N . Moreover, it is known that the Gibbs posterior 8 generalizes well with β = O √ N (e.g. see Theorem 2.8 in 1), which is significantly smaller than β = O (N ). Lastly, in Appendix E we examine the impact of β in the simple model of linear regression with i.i.d. standard Gaussian input, labels produced by a constant-magnitude teacher label noise, trained using regularized CLD as in (10), with λ ∝ d to match standard initialization. We find there that whenever d ≪ β ≪ N , the added noise does not significantly affect the training or population losses, and our bound is useful, i.e., it implies a vanishing generalization gap (since β ≪ N and E p0 L = O(1)). Note that d ≪ N is not a major constraint, since d ≪ N is required to obtain low population loss in this setting, even if we did not add noise to the training process (i.e. β = ∞).
this section cite: ['b48', 'b37', 'b18', 'b20', 'b27', 'b9', 'b29', 'b9']

Section: Magnitude of noise: empirical perspective.
An inverse temperature of β = O (N ) is also relevant in many practical settings. For example, in Bayesian settings, when we wish to (approximately) sample from the posterior, it is quite common to use variants of SGLD; then inverse temperatures of order β = O (N ) are commonly used to achieve good generalization [69], which matches our results. In the standard practical training settings, the inverse temperature is a hyperparameter tuned to best fit a given problem. Empirically, in Appendix F we find that β = O (N ) can be tuned to obtain nonvacuous generalization bounds for overparameterized NNs in a few small binary classification datasets (binary MNIST, Fashion MNIST, SVHN, and a parity problem), i.e. the sum of the generalization gap bound and the training error is smaller than 0.5. Importantly, these non-vacuous bounds do not use any trajectory-dependent quantities as other non-vacuous bounds (e.g. [15,39]), which can make them arguably more useful as they can be calculated before training. The bounds are still not very tight (at noise levels that allow for non-vacuous bounds), but we believe there is still much room for improvement in future work.
this section cite: ['b68', 'b14', 'b38']

Section: Extensions and Modifications
State dependent diffusion coefficient. Consider a state-dependent diffusion coefficient
dθ t = -∇L S (θ t ) dt + 2β -1 σ 2 (θ t )dw t + dr t ,
where σ 2 ∈ C 2 . For example, in Appendix D.1 we derive the explicit form of stationary distributions when σ 2 (θ) = (L S (θ) + α) k or σ 2 (θ) = e αL S (θ) , for some k ∈ N and α > 0. In both cases, the analytic form of the stationary potential Ψ can be used directly with Theorem 2.7 to derive generalization bounds.
Restricted initialization. In Appendix D.2 we present generalizations of Corollary 3.1 to cases where p 0 and ν are different. Specifically, for the bounded case we consider p 0 that is uniform in a subset Θ 0 ⊂ Θ of the domain, and for the regularized case we consider general diagonal Gaussian initialization and regularization. In particular, this means that some of the parameters can be more Table 1: Comparison of generalization bounds for CLD. We compare the main bounds in settings similar to the CLD setting considered here. All the bounds here consider different functions for training and evaluation, as was done in this paper with L S and E S , E D , respectively. For simplicity, we assume that E S , E D are bounded in [0, 1], and are therefore 1/2-subGaussian via Hoeffding's inequality. We use g t to denote trajectory-dependent statistics of the gradients, K for the Lipschitz constant, and C for a bound on the loss, or the expected loss at initialization, when they are required. For compactness, low-order terms are omitted, time-dependent quantities are simplified to an approximate asymptotic value, and trajectory dependent integrals are solved by considering the statistics g t constant w.r.t. the variable of integration. Finally, all bounds assume a Gaussian initialization N 0, λ -1 I d and regularization term λ 2β ∥θ t ∥ 2 , both with the same λ.
this section cite: []

Section: Paper Trajectory dependent dimension dependence Bound (big O)
Mou et al. [49] ✓ through gradients
β N • 1 λ g 2 t Li et al. [38] ✗ through K e 4βC √ β N • 2K √ λ
Futami and Fujisawa [19] ✓ through gradients
β N e 8βC • 1 λ g 2 t Ours (11) ✗ ✗ β N • √ C
loosely regularized/bounded at a cost proportional to their number. For example, in a deep NN, if only a single layer is loosely regularized/bounded, the KL-divergence cost will be proportional only to the number of parameters in that layer, not the entire d.
this section cite: ['b48', 'b18']

Section: Related Work
Information theoretic guarantees and PAC-Bayes theory. A common type of generalization bounds consists of a measure of the dependence between the learned model and the dataset used to train it, such as the mutual information between the data and algorithm [58,70,61] or the KLdivergence between the predictor's distribution and any data-independent distribution [44,9,1]. In particular, recent works were able to estimate such dependence measures from trained models to derive non-vacuous generalization bounds, even for deep overparameterized models. For example, Dziugaite et al. [17] used held-out data to bound the KL-divergence in a PAC-Bayes bound with a datadependent prior. Other works used some property of the trained model to estimate the information content, adding valuable insight to the mechanisms facilitating the successful generalization, such as the size of the compressed model after training, due to noise stability [3], and data structure [39].
this section cite: ['b57', 'b69', 'b60', 'b43', 'b8', 'b0', 'b16', 'b2', 'b38']

Section: Generalization of the Gibbs posterior.
One classic result in the PAC-Bayesian theory of generalization is that the Gibbs posterior with properly tuned temperature minimizes the PAC-Bayes bound of McAllester [44], i.e. the KL-regularized expected loss. Raginsky et al. [59] used uniform stability [7] to derive a different generalization bound for sampling from the Gibbs distribution. Due to these known generalization capabilities, some works relied on it to derive bounds for related algorithms.
this section cite: ['b43', 'b58', 'b6']

Section: Explicit Comparison for CLD
Many previous works [59,49,38,18,19,14] derived generalization bounds specifically for CLD, under different assumptions. Our bound offers some improvements over previous ones:
• It is trajectory independent, and does not require gradient statistics [49,19].
• It does not require very large time scales to make sure we have already converged near Gibbs [59], nor does it deteriorate with time, as is common for stability-based bounds [49,14].
• It does not depend on the dimension of the parameters, neither explicitly through constants [18], nor implicitly, e.g. through the Lipschitz constant or the norms of the gradients [49,38,19]. In particular, as previously discussed, using standard initialization, our in-expectation bound in (11) is dimension independent. However, our high-probability bound (12) relies on the effective supremum at t = 0, and may also depend on the dimension if the loss is not bounded.
• The dependence on the inverse temperature β and loss' (or expected loss) bound C is polynomial ( √ βC) instead of exponential [38,18,19].
• The bounded expectation assumption in (11) is weaker than a uniform bound on the loss [38,19].
• Theorem 2.7 and Corollary 3.1 demonstrate that our results hold for general initializationregularization pairs, beyond Gaussian initialization with matching ℓ 2 regularization.
In Table 1 we compare in more detail Corollary 3.1 to other bounds that remain bounded as t → ∞.
Finally, Dupuis et al. [14] recently derived bounds on the generalization gap for all intermediate times 0 ≤ s ≤ t simultaneously. Naturally, as avoiding parameters with large generalization gap is increasingly less likely as the process mixes, their bounds grow with time. Therefore, Dupuis et al. [14]'s bounds are qualitatively different, and higher than most other bounds, including ours.
this section cite: ['b58', 'b48', 'b37', 'b17', 'b18', 'b13', 'b48', 'b18', 'b58', 'b48', 'b13', 'b17', 'b48', 'b37', 'b18', 'b37', 'b17', 'b18', 'b10', 'b37', 'b18', 'b13', 'b13']

Section: Technical Novelty
As a representative example, we first focus on Raginsky et al. [59], which provided a bound for CLD (as an intermediate step for deriving a generalization bound for SGLD, a discretized version of CLD). Using spectral methods [e.g. 5], they bound the distance between the process' distribution to the Gibbs posterior, which, when combined with the generalization bound for the Gibbs distribution, results in generalization bounds for intermediate times. Our Corollary 2.5 and the preceding arguments are similar to the proof of Lemma 3.4 of Raginsky et al. [59] that bounds the divergence between the initialization and the Gibbs distribution, where their dissipativity coefficient m corresponds to our explicit ℓ 2 regularization coefficient λ. We use some significant observations that make the bound simpler, and time/dimension/Lipschitz/smoothness independent.
• Instead of a bound on the convergence of intermediate time distributions to Gibbs, which restricts the result to very large times and introduces exponential dependence on dimensionality through the spectral gap, we only require the monotonic convergence to it. As a result, we do not use a spectral gap, but a complexity term for the initial distribution. This also enables us to generalize the result to any Markov process, relying on E p0 Ψ as a complexity term for the Gibbs posterior, which is also included in Lemma 3.4 of Raginsky et al. [59] along other quantities.
• By using a symmetric version of the divergence (e.g. by summing KL (p ∥ q) and KL (q ∥ p)) we were able to completely remove the partition function from the analysis, avoiding the complications arising from it.
• By separating the regularization from the loss we were able to disentangle their effects. This approach also sidesteps the main difficulties encountered by other works, e.g., using stabilitybased bounds [49,38,19] which either diverge with training time or have dimension dependence.
this section cite: ['b58', 'b58', 'b58', 'b48', 'b37', 'b18']

Section: Generalization Guarantees Applicable for Neural Networks
Many additional lines of work established generalization guarantees applicable for NNs, but are less directly related to our work. These results have some limitations that do not exist in ours. For example, NTK analysis [29] can imply generalization guarantees in certain settings, but they do not allow for feature learning; Mean-field results [45] require non-standard initialization and specific architectures; Algorithmic stability analysis Bousquet and Elisseeff [7], Hardt et al. [26], Richards and Rabbat [60], Lei et al. [36], Wang et al. [67] only apply when the number of iterations is sufficiently small; Norm-based generalization bounds [6,22] ignore optimization aspects and depend exponentially on the network's depth; And bounds for random interpolators [8] involve impractical training procedures.
A closely related setting to the one studied here is SGLD, i.e. a discretized version of CLD. There is an extensive line of work bounding the generalization gap of such models (see [59,49,55,51,18,19,13] for a partial list). These results typically have a significant dependence on hyperparameter stemming from the discretization such as the learning rate and batch size, or suffer from constraints similar to the ones discussed in Section 4.1, such as dependence on trajectory or dimensionality (e.g. via smoothness, parameter norms, log-Sobolev or spectral gap constants). potential or expected potential. For CLD with regularization/boundedness constraint matching the initial distribution, we proved that the model generalizes well when the inverse temperature is of order β = O (N ). There are several interesting directions to extend this result.
Non-isotropic noise. We can consider a more general model for training, such as
dθ t = -∇L S (θ t ) dt + Σ (θ t ) dw t + dr t ,
where Σ is a matrix-valued dispersion coefficient, and r t is some regularization process, such as ℓ 2 regularization or a reflection process in a bounded domain. In contrast, in this paper, to derive concrete generalization bounds, we focused on CLD with isotropic noise, i.e. such that Σ is a scalar multiple of the identity matrix. The reason for this was that our bound (Corollary 3.1) relies on explicit analytical expressions or bounds on stationary distributions, which are difficult to find in the general case. In addition, in typical overparameterized settings, the noise induced by the randomness of SGD may not only be non-isotropic, but also singular with low-rank. The analysis of such processes poses various challenges beyond the ability to derive an analytic form for their stationary distribution. For example, they may concentrate on low-dimensional manifolds, possibly making the KL-divergence term infinite, or making some of the assumptions unrealistic (e.g. the choice of initial distribution).
this section cite: ['b28', 'b44', 'b6', 'b25', 'b59', 'b35', 'b66', 'b5', 'b21', 'b7', 'b58', 'b48', 'b54', 'b50', 'b17', 'b18', 'b12']

Section: No regularization.
In this work, we only considered processes that have stationary probability measures. For this reason, in the examples in Section 3 we used either a bounded domain or regularization. This seems essential for generalization at t → ∞, unless there are other architectural constraints. For example, consider training a model for the classification of randomly labeled data. Without regularization, a sufficiently expressive model is likely to arrive (at some point) at high training accuracy, yet it cannot generalize in this setting. Nonetheless, it might be possible to ensure generalization as a function of time, but here we focus on time-independent bounds.
this section cite: []

Section: Discrete time steps.
The behavior of SGD with a large step size may be qualitatively different than that of the continuous process considered here. Specifically, Azizian et al. [4] showed that while the asymptotic distribution of SGD resembles the Gibbs posterior, it is influenced by the step size and geometry of the loss surface. While an extension of our analysis to this setting is straightforward given a stationary distribution, such stationary distributions are typically hard to find explicitly (except in simple cases, such as quadratic potentials), and the error terms coming from their approximations are typically detrimental to finding non-vacuous generalization bounds, as they may depend on the dimension of the parameters through the model's Lipschitz or smoothness coefficients, etc. (49,38,19,14). Hence, a direct application of our approach to such algorithms requires additional considerations. An alternative approach is to incorporate a Metropolis-Hastings type rejection [47,27], ensuring that the stationary distribution is indeed the Gibbs posterior.
this section cite: ['b3', 'b48', 'b37', 'b18', 'b13', 'b46', 'b26']

Section: Can noise be useful for generalization?
There is a long line of work in the literature (e.g. see [20] and references therein), debating the effect of noise on generalization. Our work does not imply that higher noise improves the test error, only that it decreases the gap between training and testing. Since higher noise could hurt the training error, the overall effect depends on the specific situation. Even if introducing noise does not improve test performance, there could still be an advantage to introducing noise, based on our results, in that it reduces the gap and thus could increase the training error to match the test error in cases we cannot hope to learn (i.e. to get a small test error). This is a good thing since it prevents being mislead by overfitting, hopefully without hurting the test error when we can generalize well (i.e. in learnable regimes, both training and test errors are low, perhaps also without noise, but in non-learnable regimes, where the test error is necessarily high, noise forces the training error to be high as well, so that the gap is small). Indeed, in our small-scale experiments in Appendix F, we noticed that a small amount of noise can decrease the generalization gap, without significantly harming the test error (e.g. see the bottom half of Tables 2 to 4). Further analysis is necessary in order to establish general conditions under which test performance is not significantly hurt by noise, while ensuring a small gap. This, in particular, requires studying the effect of noise on the training loss, and what noise level still ensures obtaining a small training loss in learnable regimes.
this section cite: ['b19']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: See Sections 2 and 3.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: See Section 5.
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
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: See Sections 2 and 3 and appendices.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: See Appendix F. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [No] Justification: The experiments conducted use standard models and datasets, and are described in a manner that allows for simple reproducibility.
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

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes]
Justification: The required details appear in Appendix F.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [Yes] Justification: See Appendix F.
Guidelines:
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [No] Justification: The experiments are not computationally demanding and can be reproduced with basic resources.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). 9.
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: This work does not deviate from the code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction). 10.
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: The research conducted in this work is theoretical, and foundational in nature.
Thus, there are no broader impacts we feel must be specifically highlighted.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA]
Justification: The research conducted in this work is theoretical, and foundational in nature.
Thus, there are no risks we feel must be specifically safeguarded.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We properly reference previous theoretical work throughout the paper, and the datasets used for experimentation in Appendix F.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
• We denote stationary distributions of Markov processes by p ∞ .
• In the context of PAC-Bayesian theory, we denote prior distributions by ρ, and data dependent posteriors by ρ = ρS .
• In case some stationary distribution is also data-dependent, we use p ∞ .
• We also use p, q for generic distributions, or modify the pervious notation.
this section cite: []

Section: A.2 General Lemmas: Data processing inequality and generalized second laws of thermodynamics
For completeness, we start by proving some well known results in probability and the theory of Markov processes. Lemma A.1 (Data processing inequality). Let p (x, y) and q (x, y) be the densities of two joint distributions over a product measure space X × Y. Denote by p X (x) , q X (x) the marginal densities, e.g. Proof. By definition of the KL divergence
KL (p ∥ q) = X ×Y p (x, y) ln p (x, y) q (x, y) dxdy = X ×Y p (x, y) ln p (y | x) p X (x) q (y | x) q X (x) dxdy = X ×Y p (x, y) ln p X (x) q X (x) dxdy + X ×Y p (x, y) ln p (y | x) q (y | x) dxdy = X ×Y p (y | x) p X (x) ln p X (x) q X (x) dxdy + X ×Y p X (x) p (y | x) ln p (y | x) q (y | x) dxdy [Fubini] = X p X (x) ln p X (x) q X (x) dx + E X∼p X Y p (y | X) ln p (y | X) q (y | X) dy = KL (p X ∥ q X ) + E X∼p X KL (p (• | X) ∥ q (• | X)) .
The KL divergence is non-negative and therefore the expectation in the last line is non-negative as well, and we conclude that
KL (p ∥ q) ≥ KL (p X ∥ q X ) . Let X n = {X n } ∞ n=0
be a discrete-time Markov chain on Ω ⊂ R d , with transition kernel P (y | x) such that for all n ∈ N 0 ,
p n+1 (y) = Ω P (y | x) p n (x) dx .
In addition, assume that the there exists an invariant distribution p ∞ such that
p ∞ (y) = Ω P (y | x) p ∞ (x) dx .
We proceed to present a generalized form of the second law of thermodynamics, regarding the monotonicity of the (relative) entropy of Markov processes with possibly non-uniform stationary distributions [11,12].
Lemma A.2 (Generalized second law of thermodynamics). For all n ≥ 0,
KL (p n+1 ∥ p ∞ ) ≤ KL (p n ∥ p ∞ ) .
Proof. First, note that we can assume that KL (p n ∥ p ∞ ) < ∞ , since otherwise the claim holds trivially. Let q (x, y) = p n (x) P (y | x) be the joint densities of (X n , X n+1 ) where X n ∼ p n , and let r (x, y) = p ∞ (x) P (y | x) be the joint distribution under X n ∼ p ∞ . By definition of p n+1 , q Y (y) = p n+1 (y) , and by definition of the stationary distribution, r Y (y) = p ∞ (y) .
Therefore according to Lemma A.1,
KL (p n+1 ∥ p ∞ ) ≤ KL (q ∥ r) .
In addition,
KL (q ∥ r) = Ω×Ω q (x, y) ln q (x, y) r (x, y) dxdy = Ω×Ω q (x, y) ln p n (x) P (y | x) p ∞ (x) P (y | x) dxdy = Ω×Ω q (x, y) ln p n (x) p ∞ (x) dxdy = Ω×Ω p n (x) P (y | x) ln p n (x) p ∞ (x) dxdy [Fubini] = Ω p n (x) ln p n (x) p ∞ (x) dx = KL (p n ∥ p ∞ ) ,
and overall
KL (p n+1 ∥ p ∞ ) ≤ KL (p n ∥ p ∞ ) .
A similar result can be obtained form D ∞ (• ∥ •).
this section cite: ['b10', 'b11']

Section: Lemma A.3 (The Pointwise Second Law).
For all n > 0 :
D ∞ (p n+1 ∥ p ∞ ) ≤ D ∞ (p n ∥ p ∞ ) .
Proof. Let p, q be some probability measures such that dp dq exists. By definition,
D ∞ (p ∥ q) = ess sup q ln dp dq = inf c ∈ R | q x | ln dp dq > c = 0 .
Let C ∈ R and suppose that for all measurable A ⊂ X , p (A) ≤ e C q (A). Assume by way of contradiction that D ∞ (p ∥ q) > C, that is, that there exists c > C such that
q x | ln dp dq > c > 0 . Denote A = x | ln dp dq > c ,then
p (A) = A dp dq dq > e c q (A) > e C q (A) ,
in contradiction to the assumption. Therefore, for all C such that p (A) ≤ e C q (A) for all measurable A, C ≥ D ∞ (p ∥ q). We can now show the claim.
Let P (dy | x) be the processes' transition kernel (in measure form). We can assume D ∞ (p n ∥ p ∞ ) < ∞, since otherwise the claim holds trivially. Let A be measurable, then by definition,
p n+1 (A) = P (A | x) dp n (x) = P (A | x) dp n dp ∞ (x) dp ∞ (x) ≤ e D∞(pn ∥ p∞) P (A | x) dp ∞ (x) = e D∞(pn ∥ p∞) p ∞ (A) , so D ∞ (p n+1 ∥ p ∞ ) ≤ D ∞ (p n ∥ p ∞ ).
We can now state the relevant results for continuous-time processes. Corollary A.4. Let X t be a Markov process with marginals p t and stationary distribution p ∞ . Then, for all t > 0 :
KL (p t ∥ p ∞ ) ≤ KL (p 0 ∥ p ∞ ) or D ∞ (p t ∥ p ∞ ) ≤ D ∞ (p 0 ∥ p ∞ )
Proof. Let 0 < t and let ∆t > 0 such that t ∈ ∆t • N. Define Y n = X n∆t , then Y n is a discrete time Markov chain with marginals p n•∆t and stationary distribution p ∞ , so Lemma A.2 and Lemma A.3 imply the results.
this section cite: []

Section: B Proof of Theorem 2.7 and its Related Claims in Section 2
In this section, we present the proof of Theorem 2.7, the claims leading to it, and some of its generalizations.
this section cite: []

Section: B.1 Derivation of Corollary 2.5
Recall Claim 2.3. If p, q, µ, ν are probability measures, and p is Gibbs w.r.t q with potential Ψ < ∞, then
1. KL µ (p ∥ q) + KL ν (q ∥ p) = E ν Ψ -E µ Ψ, 2. D µ ∞ (p ∥ q) + D ν ∞ (q ∥ p) = ess sup ν Ψ -ess inf µ Ψ. In particular, KL (p ∥ q) + KL (q ∥ p) = E q Ψ -E p Ψ, and D ∞ (p ∥ q) + D ∞ (q ∥ p) = ess sup q Ψ - ess inf p Ψ.
Proof. By definition dp dq = Z -1 e -Ψ where Z < ∞ is the appropriate partition function. Then we have
KL µ (p ∥ q) + KL ν (q ∥ p) = dµ ln dp dq + dν ln dq dp = (-Ψ -ln Z) dµ + (Ψ + ln Z) dν = E ν Ψ -E µ Ψ .
Also,
D µ ∞ (p ∥ q) + D ν ∞ (q ∥ p) = ln ess sup µ dp dq + ln ess sup ν dq dp = ess sup µ (-Ψ -ln Z) + ess sup ν (Ψ + ln Z) = ess sup ν Ψ -ess inf µ Ψ ,
where in the last equality we used the fact that ess sup (-Ψ) = -ess inf Ψ, and that Z is a constant.
Using the Chain Rule and Claim 2.4, we derive the bounds of ( 1) and ( 2), as re-stated and established in the following lemma. Lemma B.1. If p t is the marginal distribution of a Markov process with initial distribution p 0 at time t, p ∞ is a stationary distribution, and ν is a probability measure, then
KL (p t ∥ ν) ≤ KL (p 0 ∥ ν) + KL p0 (ν ∥ p ∞ ) + KL pt (p ∞ ∥ ν) ,
and similarly,
D ∞ (p t ∥ ν) ≤ D ∞ (p 0 ∥ ν) + D p0 ∞ (ν ∥ p ∞ ) + D pt ∞ (p ∞ ∥ ν) .
Proof. This is a simple application of the chain rule,
KL (p t ∥ ν) = dp t ln dp t dν = dp t ln dp t dp ∞ dp ∞ dν = KL (p t ∥ p ∞ ) + KL pt (p ∞ ∥ ν) ≤ KL (p 0 ∥ p ∞ ) + KL pt (p ∞ ∥ ν) = KL (p 0 ∥ ν) + KL p0 (ν ∥ p ∞ ) + KL pt (p ∞ ∥ ν) ,
where in the first inequality we used Claim 2.4. Similarly,
D ∞ (p t ∥ ν) = ess sup pt ln dp t dν = ess sup pt ln dp t dp ∞ dp ∞ dν ≤ D ∞ (p t ∥ p ∞ ) + D pt ∞ (p ∞ ∥ ν) ≤ D ∞ (p 0 ∥ p ∞ ) + D pt ∞ (p ∞ ∥ ν) = D ∞ (p 0 ∥ ν) + D p0 ∞ (ν ∥ p ∞ ) + D pt ∞ (p ∞ ∥ ν) .
Corollary 2.5 now follows from plugging in Claim 2.3 into Lemma B.1.
Given these bounds on the divergences, All that remains in order to prove Theorem 2.7 is plugging Corollary 2.5 into a PAC-Bayes bound.
this section cite: []

Section: B.2 In-Expectation PAC-Bayes Bounds
Theorem B.2 (Theorem 5 from Maurer [43]). For any δ ∈ (0, 1) and any N ≥ 8, for any dataindependent prior distribution ρ:
P S∼D N ∀ ρ kl (E h∼ ρE S (h) ∥ E h∼ ρE D (h)) ≤ KL (ρ ∥ ρ) + ln 2 √ N δ N ≥ 1 -δ ,
where kl (a ∥ b) = a ln a b + (1 -a) ln 1-a 1-b for 0 ≤ a, b ≤ 1 is the KL divergence for a Bernoulli random variable, and ρ denotes a posterior distribution.
this section cite: ['b42']

Section: B.3 Single-Sample PAC-Bayes Bounds
Theorem B.2 can be viewed as a bound in expectation over the draw from the posterior, which corresponds to the traditional PAC-Bayes view of considering the expected error of a randomized predictor. But it is actually possible to get guarantees for a single draw from this predictor, which is more appropriate when we view the randomness as part of the training algorithm, that then outputs a single deterministic predictor (chosen at random). High probability guarantees for a single draw from the posterior were shown by Alquier et al. [1] based on Catoni [9] and also discussed by Dziugaite and Roy [16]. Here we present a tight version based on a simple modification to Maurer's proof [43]. Theorem B.3. For any δ ∈ (0, 1) and N ≥ 8, for any data independent prior ρ, and any learning rule specified by a conditional probability h|S ∼ ρS such that ρ ≪ ρS S-a.s.,
P S∼D N ,h∼ ρS   kl (E S (h) ∥ E D (h)) ≤ ln d ρS dρ (h) + ln 2 √ N δ N   ≥ 1 -δ ,
and so, by the definition of D ∞ (ρ S ∥ ρ),
P S∼D N ,h∼ ρS kl (E S (h) ∥ E D (h)) ≤ D ∞ (ρ S ∥ ρ) + ln 2 √ N δ N ≥ 1 -δ .
Proof. Following and modifying the proof of Theorem 5 of Maurer [43], we start with the inequality Theorem 1], which holds for any h, and so also in expectation over h w.r.t. ρ:
E S e N kl(E S (h) ∥ E D (h)) ≤ 2 √ N [43,
2 √ N ≥ E h∼ρ E S [exp (N kl (E S (h) ∥ E D (h)))] = E S E h∼ρ [exp (N kl (E S (h) ∥ E D (h)))]
with a change of measure from ρ to ρS ,
= E S E h∼ ρS exp (N kl (E S (h) ∥ E D (h))) dρ dρ S (h) (13
) = E S,h∼ ρS exp N kl (E S (h) ∥ E D (h)) -ln dρ S dρ (h) (14
)
Now applying Markov's inequality, we get:
P S,h∼ ρS exp N kl (E S (h) ∥ E D (h)) -ln dρ S dρ (h) ≤ 2 √ N δ ≥ 1 -δ . (15
)
Rearranging terms, we get the desired bound.
this section cite: ['b0', 'b8', 'b15', 'b42', 'b42']

Section: B.4 Arriving at Theorem 2.7
Theorem B.4. Consider any distribution D over Z, function f : H × Z → [0, 1], and sample size N ≥ 8, any distribution ν over H, and any discrete or continuous time process {h t ∈ H} t≥0 (i.e. t ∈ Z + or t ∈ R + ) that is time-invariant Markov conditioned on S. Denote p 0 (•; S) the initial distribution of the Markov process (that may depend on S). Let p ∞ (•; S) be any stationary distribution of the process conditioned on S, and Ψ S (h) ≥ 0 a non-negative potential function that can depend arbitrarily on S, such that p ∞ (•; S) is Gibbs w.r.t. ν with potential Ψ S . Then:
1. With probability 1 -δ over S ∼ D N , kl (E [E S (h t )|S] ∥ E [E D (h t )|S]) ≤ KL (p 0 (•; S) ∥ ν) + E [Ψ S (h 0 )|S] + ln 2 √ N/δ N (16
)
and so
E [E D (h t ) -E S (h t )|S] ≤ 2E [E S (h t ) | S] KL (p 0 (•; S) ∥ ν) + E [Ψ S (h 0 )|S] + ln 2 √ N/δ N + 2 KL (p 0 (•; S) ∥ ν) + E [Ψ S (h 0 )|S] + ln 2 √ N/δ N (17
)
2. With probability 1 -δ over S ∼ D N and over h t :
kl (E S (h t ) ∥ E D (h t )) ≤ D ∞ (p 0 (•; S) ∥ ν) + ess sup p0 Ψ S (h 0 ) + ln 2 √ N/δ N (18
)
and so, when
E S (h t ) < E D (h t ) E D (h t ) -E S (h t ) ≤ 2E S (h t ) D ∞ (p 0 (•; S) ∥ ν) + ess sup p0 Ψ S (h 0 ) + ln 2 √ N/δ N + 2 D ∞ (p 0 (•; S) ∥ ν) + ess sup p0 Ψ S (h 0 ) + ln 2 √ N/δ N (19
)
Lemma B.5. Let a, b ∈ [0, 1]. Then b ≤ a + 2akl (a ∥ b) + 2kl (a ∥ b) . (20
)
Proof. The KL divergence is non-negative, so it suffices to consider the case that b ≥ a. Defining φ : [0, 1 -a] → R as
φ (u) = u 2 2 (a + u) ,
it can be readily checked by differentiation that for all u ∈ [0, 1 -a], kl (a ∥ a + u) ≥ φ (u) . In particular, for u = b -a ∈ [0, 1 -a],
kl (a ∥ b) ≥ (b -a) 2 2b . (21
)
Next, we consider the following inequality
2u 2 + √ 2au + a -b ≥ 0 , u ≥ 0 .(22)
Solving for u, it turns out that the inequality holds when
u ≥ √ 8b -6a - √ 2a 4 .(23)
In addition, under the assumption that b ≥ a,
√ 8b -6a - √ 2a 4 ≤ (b -a) 2 2b .(24)
Combining ( 21), ( 23), and (24), u = kl (a ∥ b) solves ( 22) implying (20).
Proof. The inequalities ( 16) and ( 18) follow by plugging Corollary 2.5 into Theorems B.2 and B.3. For inequalities ( 17) and ( 19), we use (20). For (17), we use a = E D (h t ) and b = E S (h t ), which yields:
E D (h t ) ≤ E S (h t ) + 2E S (h t )kl (E S (h t ) ∥ E D (h t )) + 2kl (E S (h t ) ∥ E D (h t )) ≤ E S (h t ) + 2E S (h t ) KL (p 0 (•; S) ∥ ν) + E p0 Ψ S (h 0 ) + ln 2 √ N/δ N + 2 KL (p 0 (•; S) ∥ ν) + E p0 Ψ S (h 0 ) + ln 2 √ N/δ N ,
and similarly for (19). Proof. The first direction follows directly from Claim 2.3, so we only need to prove the converse. Assume that either KL (p ∥ q) + KL (q ∥ p) ≤ β, or D ∞ (p ∥ q) + D ∞ (q ∥ p) ≤ β. In these cases, both dp/ dq and dq/ dp exist, and for any measurable event B, p (B) = 0 ⇐⇒ q (B) = 0, or equivalently, p (B) > 0 ⇐⇒ q (B) > 0. Therefore, supp (p) = supp (q), and dp/ dq > 0 on supp (p). Denote Ψ = -ln dp/ dq, then p is Gibbs w.r.t. q with potential Ψ. The same derivation as in the proof of Claim 2.3 results in the bounds E q Ψ -E p Ψ ≤ β and ess sup q Ψ -ess inf p Ψ ≤ β. In particular, if the latter holds then Ψ can be shifted such that essentially 0 ≤ Ψ ≤ β.
this section cite: ['b19', 'b19', 'b18']

Section: C Tightness and Necessity of the Divergence Conditions
If we are only interested in ensuring generalization at time t → ∞, and when we converge to the stationary distribution p ∞ , then it is enough to bound the divergence D (p ∞ ∥ ν). If we are interested in bounding D (p t ∥ ν) (and consequently, the generalization gap) at all times t, then we need also to limit p 0 's dependence on S, since p 0 (as well as p t for small t) can be completely different from a stationary p ∞ , and just bounding D (p ∞ ∥ ν) does not say anything about it. Bounding D (p 0 ∥ µ), for some data-independent distribution µ, ensures generalization at p 0 . This leaves the following questions regarding the proof of Theorem 2.7:
• Why do we need to bound the divergences D (p ∞ ∥ ν) and D (p 0 ∥ ν) from the same distribution ν? That is, we do we need to require µ = ν? Bounding the divergences of p 0 and p ∞ to two different divergences µ ̸ = ν is sufficient to get generalization at the beginning (i.e. initialization) and end (i.e. after mixing)-is it sufficient for generalization in the middle (i.e. at any t)? • Why do we need to also bound the reverse divergence D (ν ∥ p ∞ )? I.e., why do we need to require p ∞ is Gibbs w.r.t. ν with a bounded potential, instead of just controlling the divergence D (p ∞ ∥ ν), which is a weaker requirement and sufficient for generalization after mixing?
As we now show, both are necesairy, and without requiring both, i.e. if we drop either one of these, we cannot ensure generalization at intermediate times t ≥ 0.
Construction. Consider a supervised learning problem with Z = X × Y, X = [0, 1], Y = {0, 1}, H = all measurable functions from X to Y, and the zero-one loss f (h, (x, y)) = I {h(x) ̸ = y}, with D being the uniform distribution over X , and y being Bernoulli( 1 2 ) independent of x. For all h, E D (h) = 0.5. Let p 0 be the constant zero function with probability 1  2 and the constant one function with probability 1  2 . Consider the following deterministic S-dependent transition function over h: if h t is the constant zero function, then h t+1 = h S which memorizes S, i.e. h S (x) = y for (x, y) ∈ S, and h S (x) = 1 otherwise. If h t is not the constant zero function, then h t+1 is the constant ones function. We have that p ∞ is deterministic at the constant one function, and KL (p ∞ ∥ p 0 ) = ln 2, and in fact p t = p ∞ for t > 1. But with probability half, h 1 = h S , for which for any sample size
N > 0, E S (h S ) = 0 while E D (h S ) = 1 2 .
How does this show it is not enough to bound D (p 0 ∥ ν) and D (p ∞ ∥ ν), but that we also need the reverse D (ν ∥ p ∞ )? Since p 0 is data independent, we can take ν = p 0 , in which case
KL (p 0 ∥ ν) = D ∞ (p 0 ∥ ν) = 0 and KL (p ∞ ∥ ν) = D ∞ (p ∞ ∥ ν) = ln 2,
but even as N → ∞, the gap for h 1 does not diminish. Indeed, D (ν ∥ p ∞ ) = ∞, and so p ∞ is not Gibbs w.r.t. ν and Theorem 2.7 does not apply.
How does this show it is not enough to bound D (p ∞ ∥ ν) + D (ν ∥ p ∞ ) and D (p 0 ∥ µ) for µ ̸ = ν? Since in this example p ∞ is also data independent, we can take ν = p ∞ and µ = p 0 , in which case
D (p 0 ∥ µ) = 0 and D (p ∞ ∥ ν) + D (ν ∥ p ∞ ) = 0.
We are indeed ensured a small gap for h 0 and h ∞ , but not for h 1 .
this section cite: []

Section: D Generalized Version of Corollary 3.1
We start by characterizing the stationary distributions of SDERs in a box with different noise scales σ 2 . The stationary distributions for Gaussian initialization can be found similarly. Then, we extend Corollary 3.1 to scenarios where p 0 ̸ = ν, as an immediate consequence of Theorem 2.7.
this section cite: []

Section: D.1 Stationary distributions of CLD
We first derive the stationary distribution of SDERs of the form
dx t = -∇L (x t ) dt + 2β -1 σ 2 (x t )dw t + dr t ,(25)
with normal reflection in a box domain (for a full definition see ( 45)-( 47) in Appendix H.2), where L ≥ 0 is some C 2 loss function, β > 0 is an inverse temperature parameter, and σ 2 is a diffusion coefficient. First, we present a well known characterization of the stationary distribution of (25).
Lemma D.1. If L, σ 2 ∈ C 2 , σ 2 (•) > 0 is uniformly bounded away from 0 in Ω, Z = Ω 1 σ 2 (x) exp -β ∇L (x) σ 2 (x) dx < ∞ ,
the integrals exist, and the field ∇L/σ 2 is conservative (curl-free), then
p ∞ (x) = 1 Z 1 σ 2 (x) exp -β ∇L (x) σ 2 (x) dx (26
)
is a stationary distribution of (25).
For completeness, the proof is presented in Appendix H.2.1, following additional results and definitions in Appendix H. We can now calculate explicit stationary distributions for some choices of σ 2 . Specifically, we focus on cases where σ 2 (x) = g (L (x)) for some scalar function g, as it guarantees the curl-free condition, and is convenient to integrate. Example D.2 (Uniform noise scale). Assuming that σ 2 (x) ≡ 1, the stationary distribution becomes the well-known Gibbs distribution
p ∞ (x) = 1 Z e -βL(x) , (27
) so Ψ uniform (x) = βL (x) .(28)
Example D.3 (Linear noise scale). Let α > 0, and suppose that σ 2 (x) = (L (x) + α). Then
∇L (x) σ 2 (x) = ∇ ln (L (x) + α)
so the stationary distribution is
p ∞ (x) ∝ 1 L (x) + α exp (-β ln (L (x) + α)) = 1 L (x) + α (L (x) + α) -β = (L (x) + α) -β-1 ,(29)
which is integrable in a bounded domain. Recall that we want to represent p ∞ using a potential Ψ with inf Ψ ≥ 0. In this case, we can start from Ψ (x) = (β + 1) ln (L (x) + α). Since L ≥ 0 it clearly holds that Ψ ≥ (β + 1) ln (α), so we can use the shifted version
Ψ linear (x) = (β + 1) (ln (L (x) + α) -ln (α)) = (β + 1) ln L (x) α + 1 .(30)
Example D.4 (Polynomial noise scale). Let α > 0, and k > 1. Suppose that
σ 2 (x) = (L (x) + α) k . Then ∇L (x) σ 2 (x) = ∇L (x) (L (x) + α) -k = 1 1 -k ∇ (L (x) + α) 1-k so p ∞ (x) ∝ (L (x) + α) -k exp β k -1 (L (x) + α) 1-k .
As before, the potential is monotonically increasing with L (x), so we can make a shift
Ψ poly = k ln L (x) α + 1 + β k -1 α 1-k -(L (x) + α) 1-k .
Example D.5 (Exponential noise scale). Let α > 0 and suppose that σ 2 (x) = e αL(x) . Then
∇L (x) σ 2 (x) = - 1 α ∇ e -αL(x) so p ∞ (x) ∝ e -αL(x) exp β α e -αL(x) = exp β α e -αL(x) -αL (x) . Denote ψ (τ ) = ατ -β α e -ατ , then ψ ′ (τ ) = α + βe -ατ ≥ 0. Therefore, min τ ≥0 ψ (τ ) = ψ (0) = -β
α , and we can take
Ψ exp (x) = αL (x) - β α e -αL(x) + β α = αL (x) + β α 1 -e -αL(x)(31)
this section cite: ['b24', 'b24']

Section: D.2 Generalization bounds
Bounded domain with uniform initialization. Assume that training follows a CLD in a bounded domain as described in (25) with uniform initialization p 0 = Uniform (Θ 0 ), where Θ 0 ⊆ Θ. For simplicity we take σ 2 ≡ 1. In that case Theorem 2.7 implies the following. Lemma D.6. Assume that the parameters evolve according to (25) with σ 2 ≡ 1 and uniform initialization p 0 = Uniform (Θ 0 ), where Θ 0 ⊆ Θ. Then for any time t ≥ 0, and δ ∈ (0, 1),
1. w.p. 1 -δ over S ∼ D N , E θt∼pt [E D (θ t ) -E S (θ t ) | S] ≤ βE p0 [L S (θ) | S] + ln |Θ|/|Θ 0 | + ln (N/δ) 2N . (32
)
2. w.p. 1 -δ over S ∼ D N and θ t ∼ p t E D (θ t ) -E S (θ t ) ≤ β ess sup p0 L S (θ) + ln |Θ|/|Θ 0 | + ln (N/δ) 2N .(33)
Proof. This is a direct corollary of Theorem 2.7 with KL (p
0 ∥ ν) = ln |Θ|/|Θ 0 |.
ℓ 2 regularization with Gaussian initialization. Let λ ∈ R d >0 be regularization terms, and consider the unconstrained SDE
dθ t = -∇L (θ t ) dt -β -1 diag (λ) θ t dt + 2β -1 σ 2 (θ t )dw t .(34)
Notice that -β -1 diag (λ) θ t dt corresponds to an additive regularization of the form 1 2β θ ⊤ t diag (λ) θ t , so each parameter can have a different regularization coefficient. We shall denote by ϕ λ a multivariate Gaussian distribution with mean 0 and covariance matrix diag λ -1 , where
λ -1 = λ -1 1 , . . . , λ -1 d .
For simplicity, we present the results with σ 2 ≡ 1.
Lemma D.7. Let λ 0 , λ 1 > 0, and let θ t evolve according to (34) with σ 2 ≡ 1 and λ = λ 1 , and start from a Gaussian initialization p 0 = ϕ λ0 . Then for any time t ≥ 0, and δ ∈ (0, 1),
1. w.p. 1 -δ over S ∼ D N , E θt∼pt [E D (θ t ) -E S (θ t ) | S] ≤ βE p0 [L S (θ) | S] + KL (ϕ λ0 ∥ ϕ λ1 ) + ln (N/δ) 2N .(35
) 2. w.p. 1 -δ over S ∼ D N and θ t ∼ p t E D (θ t ) -E S (θ t ) ≤ β ess sup p0 L S (θ) + KL (ϕ λ0 ∥ ϕ λ1 ) + ln (N/δ) 2N ,(36)
where
KL (ϕ λ0 ∥ ϕ λ1 ) = 1 2 d i=1 ln λ1,i λ0,i -1 + λ0,i λ1,i . 9
Proof. This is a direct corollary of Theorem 2.7 with the explicit expression for the KL divergence between two Gaussians.
Remark D.8 (Dependence on the parameters' dimension). While the bound in Lemma D.7 depends on the dimension of the parameters d, this can be mitigated in practice. For example, by matching the regularization coefficient and initialization variance, the KL-divergence term vanishes and we lose the dependence on dimension. Furthermore, we can control each parameter separately by using parameter specific initialization variances and regularization coefficients. Then, the KL-divergence can have different dependencies, if any, on the dimension d.
this section cite: ['b24']

Section: E Linear Regression with CLD
Theorem 2.7 and Corollary 3.1 only bound the gap between the population and training errors, yet this does not necessarily bound the population error itself. One way to do this is by separately bounding the training error and showing that in the regime in which the generalization gap is small, the training error can be small as well. In Appendix F we show empirically that deep NNs can reach low training error when trained with SGLD in the regime in which Corollary 3.1 is not vacuous. Here, we look at the particular case of the asymptotic behavior of ridge regression with CLD training with Gaussian i.i.d. data, for which we can analytically study the training and population losses.
Setup. Let θ ⋆ ∈ R d , y = x ⊤ θ ⋆ + ε with ∥θ ⋆ ∥ = 1 and ε ∼ N 0, σ 2 independent of x. We assume that x has i.i.d. entries with Ex = 0 and covariance E xx ⊤ = I. Let X ∈ R N ×d be the data (design) matrix, y ∈ R N the training targets, ε ∈ R N the pointwise perturbations, and θ ∈ R d the parameters in a linear regression problem. In what follows, we focus on the overdetermined case N > d, where X has full column rank with probability 1, so the empirical covariance A = 1 N X ⊤ X ≻ 0 a.s. In addition, we denote θ LS = 1 N A -1 X ⊤ y, and θ = θθ LS . The training objective is then the minimization of the regularized empirical loss
L S (θ) + λ 2β ∥θ∥ 2 = 1 2N ∥Xθ -y∥ 2 + λ 2β ∥θ∥ 2 = 1 2 θ⊤ A θ + C S + λ 2β ∥θ∥ 2 , where C S = L S (θ LS ) = 1 2N ∥y∥ 2 -1 2 θ LS Aθ LS = 1 2N ∥y∥ 2 -1 2N y ⊤ X X ⊤ X -1 X ⊤ y, is the empirical irreducible error.
this section cite: []

Section: CLD training.
Assume that training is performed by CLD with inverse temperature β > 0, which, because L S is quadratic, takes the form
dθ t = -A (θ t -θ LS ) dt -λβ -1 θ t dt + 2 β dw t .(37)
Since A ≻ 0 and λ > 0, the Gibbs distribution
p ∞ (θ) ∝ exp - 1 2 (θ -θ LS ) ⊤ βA (θ -θ LS ) + λθ ⊤ θ
is the unique stationary distribution, and furthermore, it is the asymptotic distribution of (37). We can simplify this to a Gaussian. Denote α = λ/β and
Σ = 1 β (A + αI) -1 and θ = βΣAθ LS = 1 N (A + αI) -1 X ⊤ y , then θ -θ ⊤ Σ -1 θ -θ = βθ ⊤ (A + αI) θ -2θ ⊤ Σ -1 θ + θ⊤ Σ -1 θ = βθ ⊤ (A + αI) θ -2βθ ⊤ Σ -1 ΣAθ LS + β 2 θ ⊤ LS AΣΣ -1 ΣAθ LS = βθ ⊤ (A + αI) θ -2βθ ⊤ Aθ LS + β 2 θ ⊤ LS AΣAθ LS . Since the last term is constant w.r.t. θ, we deduce that p ∞ (θ) ∝ exp - 1 2 θ -θ ⊤ Σ -1 θ -θ ,
i.e. the stationary distribution is a Gaussian N θ, Σ . We can now calculate the expected training and population losses.
this section cite: ['b36']

Section: Goal.
In the rest of this section, our final aim is to calculate the expected training and population losses in the setup described above, in the case when the data is sampled i.i.d. from standard Gaussian distribution, σ is a fixed constant, λ ∝ d (to match standard initialization), 10 N, β and d are large, but β ≪ N , so our generalization bound is small (since E p0 L is a fixed constant in this case). We will find (in Remark E.2 and Remark E.4) that if also d ≪ β then the training and expected population loss are not significantly degraded. This is not a major constraint, since we need d ≪ N to get good population loss anyway, even without noise (i.e. β → ∞). This shows that in this regime d ≪ β ≪ N , the randomness required by our generalization bound (the KL bounds in Corollary 3.1) does not significantly harm the training loss or the expected population loss. Claim E.1. With some abuse of notation, denote
L S (θ ∞ ) = E θ∼p∞ L S (θ). Then E [L S (θ ∞ ) | X] = 1 2β Tr A (A + αI) -1 + α 2 2 θ ⋆⊤ (A + αI) -2 Aθ ⋆ + σ 2 α 2 2N Tr (A + αI) -2 + σ 2 2 1 - d N .
Proof. From Petersen and Pedersen [56] (equation 318)
L S (θ ∞ ) = 1 2 E (θ -θ LS ) ⊤ A (θ -θ LS ) + C S = 1 2 Tr (AΣ) + 1 2 θ -θ LS ⊤ A θ -θ LS + C S .
For the second term, notice that θ -
θ LS = (βΣA -I) θ LS = (βΣA + λΣ -λΣ -I) θ LS =   Σβ (A + αI) =Σ -1 -λΣ -I    θ LS = -λΣθ LS = -α (A + αI) -1 θ LS .
A and Σ are simultaneously diagonalizable. To see this, let A = QΛQ ⊤ be a spectral decomposition of A, then A + αI = Q (Λ + αI) Q ⊤ , so Σ = β -1 Q (Λ + αI) -1 Q ⊤ . This means that A, Σ, and their inverses all multiplicatively commute. Therefore,
L S (θ ∞ ) = 1 2 Tr (AΣ) + α 2 2 θ ⊤ LS (A + αI) -1 A (A + αI) -1 θ LS + C S = 1 2 Tr (AΣ) + α 2 2N 2 y ⊤ XA -1 (A + αI) -1 A (A + αI) -1 A -1 X ⊤ y + C S = 1 2β Tr A (A + αI) -1 + α 2 2N 2 y ⊤ X (A + αI) -2 A -1 X ⊤ y + C S ,
Conditioned on X, standard results about the residuals in linear regression imply that,
E [C S | X] = σ 2 2 1 - d N .
In addition, for any symmetric matrix M we have
E ε y ⊤ My = E ε (Xθ ⋆ + ε) ⊤ M (Xθ ⋆ + ε) = (Xθ ⋆ ) ⊤ MXθ ⋆ + E ε ε ⊤ Mε = (Xθ ⋆ ) ⊤ MXθ ⋆ + σ 2 Tr (M) .
In particular,
E y ⊤ X (A + αI) -2 A -1 X ⊤ y | X = θ ⋆⊤ X ⊤ X (A + αI) -2 A -1 X ⊤ Xθ ⋆ + σ 2 Tr X (A + αI) -2 A -1 X ⊤ = θ ⋆⊤ N A (A + αI) -2 A -1 N Aθ ⋆ + σ 2 Tr X ⊤ X (A + αI) -2 A -1 = N 2 θ ⋆⊤ (A + αI) -2 Aθ ⋆ + N σ 2 Tr (A + αI) -2 ,
where we used the definition of A, the joint diagonalizability of A and Σ, and the cyclicality of the trace. In total, the expected training loss, conditioned on the data is
E ε L S (θ ∞ ) = 1 2β Tr A (A + αI) -1 + α 2 2 θ ⋆⊤ (A + αI) -2 Aθ ⋆ + σ 2 α 2 2N Tr (A + αI) -2 + σ 2 2 1 - d N .
Remark E.2. We intuitively derive the asymptotic behavior of Claim E.1. Let λ be constant, and let β grow (so α shrinks). We can decompose (A + αI) -1 as
(A + αI) -1 = A -1 -αA -2 + α 2 A -2 (A + αI) -1 .
This can be readily verified as
A -1 -αA -2 + α 2 A -2 (A + αI) -1 = A -2 (A + αI) -1 A (A + αI) -α (A + αI) + α 2 I = A -2 (A + αI) -1 A 2 + αA -αA -α 2 I + α 2 I = A -2 (A + αI) -1 A 2 = (A + αI) -1 ,
where we used the multiplicative commutativity, as before. Notice that since
A ≻ 0, A + αI ≻ A, so (A + αI) -k ≺ A -k for any k ∈ N. Denote R 2 (α) = α 2 A -2 (A + αI) -1 , then ∥R 2 (α)∥ 2 ≤ α 2
λmin(A) 3 , where λ min (A) is the minimal eigenvalue of A. As the elements of X are i.i.d. with mean 0 and variance 1, the limiting distribution of the spectrum of A as N, d → ∞ with d/N → γ ∈ (0, 1) is the Marchenko-Pastur distribution, which is supported on
1 - √ γ 2 , 1 + √ γ 2 . In particular, as N, d → ∞, λ min (A) ≥ 1 -d/N 2 , so for ε > 0, ∥R 2 (α)∥ 2 ≤ α 2 1 -d/N -ε 6
with high probability. Therefore, in the following we shall treat the remainder as R 2 (α) = O α 2 , even when taking the expectation over X.
Since α = λ/β and λ ∝ d, then for d ≤ β, we have α/β = O α 2 , and we conclude that
E [L S (θ ∞ ) | X] = d 2 1 β + σ 2 1 d - 1 N + O α 2 .
Therefore, the added noise does not significantly hurt the training loss when 1 β ⪅ σ 2 1 d -1 N , or equivalently, β ⪆ N d (N -d)σ 2 . In particular, this holds when d ≪ β ≪ N , which is a regime where our generalization bound Corollary 3.1 also becomes small (since β ≪ N ). This shows that the randomness required by Corollary 3.1 can allow for successful optimization of the training loss.
Moving on to the population loss, we define L D in the usual way
L D (θ t ) = 1 2 E x,ε x ⊤ θ t -y 2 = 1 2 E x ⊤ θ t -x ⊤ θ ⋆ -ε 2 .
Due to the independence between x and ε,
L D (θ) = 1 2 E x ⊤ (θ -θ ⋆ ) 2 + σ 2 2 = 1 2 ∥θ -θ ⋆ ∥ 2 + σ 2 2 . Claim E.3. With some abuse of notation, denote L D (θ ∞ ) = E θ∼p∞ L D (θ). Then E [L D (θ ∞ ) | X] = 1 2β Tr (A + αI) -1 + 1 2 θ ⋆⊤ A 2 (A + αI) -2 θ ⋆ + σ 2 2N Tr A (A + αI) -2 -θ ⋆⊤ A (A + αI) -1 θ ⋆ + 1 2 ∥θ ⋆ ∥ 2 + σ 2 2 .
Proof. Taking the expectation w.r.t θ ∼ N θ, Σ we get from Petersen and Pedersen [56]
L D (θ ∞ ) = 1 2 Tr (Σ) + 1 2 θ -θ ⋆ 2 + σ 2 2 = 1 2β Tr (A + αI) -1 + 1 2 θ⊤ θ - θ⊤ θ ⋆ + 1 2 ∥θ ⋆ ∥ 2 + σ 2 2 .
We can simplify some of the terms when taking the expectation conditioned on X.
E ε θ⊤ θ = 1 N 2 E y ⊤ X (A + αI) -1 (A + αI) -1 X ⊤ y = 1 N 2 E (Xθ ⋆ + ε) ⊤ X (A + αI) -2 X ⊤ (Xθ ⋆ + ε) = 1 N 2 θ ⋆⊤ X ⊤ X (A + αI) -2 X ⊤ Xθ ⋆ + 1 N 2 E ε ε ⊤ X (A + αI) -2 X ⊤ ε = θ ⋆⊤ A 2 (A + αI) -2 θ ⋆ + σ 2 N 2 Tr X (A + αI) -2 X ⊤ = θ ⋆⊤ A 2 (A + αI) -2 θ ⋆ + σ 2 N Tr A (A + αI) -2 .
In addition,
E ε θ⊤ θ ⋆ = 1 N E ε (Xθ ⋆ + ε) ⊤ X (A + αI) -1 θ ⋆ = 1 N θ ⋆⊤ X ⊤ X (A + αI) -1 θ ⋆ + 1 N E ε ε ⊤ X (A + αI) -1 θ ⋆ = θ ⋆⊤ A (A + αI) -1 θ ⋆ .
Combining these we get the desired result.
Remark E.4. As we have done for the training loss in Remark E.2, we can estimate the expected population loss in some asymptotic regimes. Let λ be constant, and let β grow (so α shrinks). As in Remark E.2, we use the approximation (A + αI) -1 = A -1 -αA -2 + O α 2 I , which also implies (A + αI) -2 = A -2 -2αA -3 + O α 2 I , and treat the remainders as O α 2 even when taking the expectation w.r.t. X. Then,
E [L D (θ ∞ ) | X] = 1 2β Tr A -1 -αA -2 + O α 2 I + 1 2 θ ⋆⊤ A 2 A -2 -2αA -3 + O α 2 I θ ⋆ + σ 2 2N Tr A A -2 -2αA -3 + O α 2 I -θ ⋆⊤ A A -1 -αA -2 + O α 2 I θ ⋆ + 1 2 ∥θ ⋆ ∥ 2 + σ 2 2 = 1 2 1 β + σ 2 N Tr A -1 + σ 2 2 - α 2β Tr A -2 + O (αI) -αθ ⋆⊤ A -1 + O (αI) θ ⋆ - σ 2 α N Tr A -2 + O (αI) + αθ ⋆⊤ A -1 + O (αI) θ ⋆ .
Simplifying, we arrive at
E [L D (θ ∞ ) | X] = 1 2 1 β + σ 2 N Tr A -1 + σ 2 2 -α 1 2β + σ 2 N Tr A -2 + O α 2 .
Assuming that x are i.i.d. N (0, I), N • A ∼ W d (N, I), i.e. has a Wishart distribution. According to Theorem 3.3.16 of [24], if N > d + 3 then
EA -1 = N N -d -1 I , EA -2 = N 2 • Tr (I) I (N -d) (N -d -1) (N -d -3) + N 2 • I (N -d) (N -d -3) = N 2 d + N 2 (N -d -1) (N -d) (N -d -1) (N -d -3) I .
Then, the expectation over X and if σ 2 N ⪅ α (which is true for λ ∝ d and β ≪ N like we assume here),
EL D (θ ∞ ) = 1 2 1 β + σ 2 1 N + N -d -1 N d • N d N -d -1 + O α 2 = 1 2 1 β + σ 2 • N -1 N d • N d N -d -1 + O α 2 .
This result is similar to the one in Remark E.2 -for the expected population loss not to be significantly hurt by the added noise, it must hold that β ⪆ N d (N -1)σ 2 . In particular, this holds when d ≪ β ≪ N , which is a regime where our generalization bound Corollary 3.1 also becomes small (since β ≪ N ). This shows that the randomness required by Corollary 3.1 does not harm the expected population loss.
this section cite: ['b55', 'b55', 'b23']

Section: F Numerical Experiments

this section cite: []

Section: F.1 Experimental results
The following are results of training with SGLD (a discretized version of the CLD in (10)) on a few benchmark datasets. Notice we use the regularized version where regularization coefficient is λ • β -1 and the λ hyperparameter is dictated by the initialization from the normal distribution p 0 = N 0, λ -1 I d . We used a common initialization of N 0, 1 din , i.e. λ = d in . We use several different values of β relative to N (the number of training samples). For simplicity, we focused on binary classification cases. In all datasets with more than 2 classes, we constructed a binary classification task by partitioning the original label set into 2 disjoint sets of the same size.
The results demonstrate that learning with SGLD is possible with various values of β. In fact, in several instances, the injected noise appears to improve the generalization gap, e.g, in SVHN [53], in all the tested β values between 0.4 • N and 2 • N the average test error remained almost the same while the training error decreased as β increased (i.e. the generalization gap increased). Notably, we also observe that for sufficiently large levels of noise, the generalization bounds are non-vacuous.
this section cite: ['b52']

Section: SVHN.
The network was trained with a convolutional neural network with 5 convolutional layers, lr = 0.01, for 80 epochs. The complete architecture:
• Two convolutional layers (3×3 kernel, padding 1) with 32 channels, followed by ReLU activations and a 2×2 max pooling.
• Two convolutional layers (3×3 kernel, padding 1) with 64 channels, followed by ReLU activations and a 2×2 max pooling.
• A 3×3 convolution with 128 channels, ReLU, and 2×2 max pooling.
• 2 A linear layer 2048 → 512, followed by ReLU and another 512 → 1 linear layer
this section cite: []

Section: Parity.
In this experiment, we consider a synthetic binary classification task where each input is a binary vector of length 70 and the target label is defined as the parity of 3 randomly selected input dimensions. We train a neural network using SGLD with varying values of the inverse temperature parameter β and different sample sizes.
The network was trained with a fully connected network with 4 hidden layers of sizes [512, 1028, 2064, 512] and ReLU activation, lr = 0.05, for 100 epochs.
The results show that injecting noise can improve the generalization gap: specifically, the case of β ≥ N 2 leads to overfitting, while smaller values of β (e.g., 1.5 • N to 12 • N ) yield better generalization. Moreover, as well as in the benchmark datasets, in this setting, our generalization bound is non-vacuous in several cases.
this section cite: []

Section: G Mild Overparametrization Prevents Uniform Convergence
In this section, we consider fully-connected ReLU networks, where the weights are bounded, such that for each layer j the absolute values of all weights are bounded by 1 √ dj-1
, where d j-1 is the width of layer j -1. Moreover, we assume that the input x is such that each coordinate x i is bounded in [-1, 1]. We show that m training examples do not suffice for learning constant depth networks with O(m) parameters. Thus, even a mild overparameterization prevents uniform convergence in our setting.
Our result follows by bounding the fat-shattering dimension, defined as follows:
Definition G.1. Let F be a class of real-valued functions from an input domain X . We say that F shatters m points {x i } m i=1 ⊆ X with margin ϵ > 0 if there are r 1 , . . . , r m ∈ R such that for all y 1 , . . . , y m ∈ {0, 1} there exists f ∈ F such that
∀i ∈ [m], f (x i ) ≤ r i -ϵ if y i = 0 and f (x i ) ≥ r i + ϵ if y i = 1 .
The fat-shattering dimension of F with margin ϵ is the maximum cardinality m of a set of points in X for which the above holds.
The fat-shattering dimension of F with margin ϵ lower bounds the number of samples needed to learn F within accuracy ϵ in the distribution-free setting (see, e.g., [2, Part III]). Hence, to lower bound the sample complexity by some m it suffices to show that we can shatter a set of m points with a constant margin. Theorem G.2. We can shatter m points {x i } m i=1 where ∥x i ∥ ∞ ≤ 1, with margin 1, using ReLU networks of constant depth and O(m) parameters, such that for each layer j the absolute values of all weights are bounded by 1  √ dj-1
, where d j-1 is the width of layer j -1.
Proof. Consider input dimension d 0 = 1. For 1 ≤ i ≤ m, consider the points x i = i m , and let {y i } m i=1 ⊆ {0, 1}. Consider the following one-hidden-layer ReLU network N , which satisfies N (x i ) = yi m for all i. First, the network N includes a neuron with weight 0 and bias y1 m , i.e., [0 • x + y1 m ] + . Now, for each i such that y i = 0 and y i+1 = 1 we add two neurons: [x -y i ] + -[x -y i+1 ] + , and for i such that y i = 1 and y i+1 = 0 we add -[x -y i ] + + [x -y i+1 ] + . It is easy to verify that this construction has width at most 2m -1 and allows us to shatter m points with margin 1 2m . However, the output weights of the neurons are ±1, and thus it does not satisfy the theorem's requirement. Consider the network N ′ (x) = N (x) • 1 √ 2m-1 obtained from N by modifying the output weights. The network N ′ satisfies the theorem's requirement on the weight magnitudes, and allows for shattering with margin 1 2m √ 2m-1 . We will now show how to increase this margin to 1 using a constant number of additional layers.
Let Ñ be a network obtained from N ′ as follows. First, we add a ReLU activation to the output neuron of N ′ . Since for every x i we have N ′ (x i ) ≥ 0, it does not affect these outputs. Next, we add L = 8 additional layers (layers 3, . . . , 3 + L -1) of width √ m and without bias terms, where the incoming weights to layer 3 are all 1 and the weights in layers 4, . . . , 3 + L -1 are 1 m 1/4 . Finally, we add an output neuron (layer 3 + L) with incoming weights 1 m 1/4 . The network Ñ satisfies the theorem's requirements on the weight magnitudes, and it has depth 3 + L = 11 and O(m) parameters. Now, suppose that all neurons in a layer 3 ≤ j ≤ 3 + L -1 have values (i.e., activations) z ≥ 0, then the values of all neurons in layer j + 1 are z
• 1 m 1/4 • √ m = z • m 1/4 .
this section cite: []

Section: H Background on Stochastic Differential Equations with Reflection
We supply an introduction to the theory of stochastic differential equations with reflection (SDERs), then proceed to characterize the stationary distribution of a family of SDERs in a box. The background of standard (non-reflective) SDEs is similar and more common, and is therefore not included here. See for example [54] for more.
this section cite: ['b53']

Section: H.1 SDEs with reflection
One of the main analytical tools of this work is the characterization of stationary distributions of SDER in bounded domains (see 57, 63, for an introduction).
The purpose of this section is to present more rigorously the setting of the paper, and supply the relevant definitions and results required to arrive at Lemma D.1. As Lemma D.1 is considered a well-known result, this section is mainly intended for completeness. Specifically, in the following we present some relevant definitions and results by Kang and Ramanan [31,32], and specifically, ones that relate solutions to SDERs (Definition 2.4 in [32]), to solutions to sub-martingale problems (Definition 2.9 in [32]), and that characterize the stationary distributions of such solutions. For simplicity, we sometimes do not state the results in full generality.
this section cite: ['b30', 'b31', 'b31', 'b31']

Section: Setting.
Let Ω ⊂ R d be a domain (non-empty, connected, and open). Let the drift term b : R d → R d and dispersion coefficient Σ : R d → R d×d be measurable and locally bounded. We also denote the diffusion coefficient by
A (•) = Σ (•) Σ (•) ⊤ = (a ij (•)) d i,j=1
, and denote its columns by a i (•). We say that the diffusion coefficient is uniformly elliptic if there exists σ > 0 such that
∀v ∈ R d , ∀x ∈ Ω v ⊤ A (x) v > σ ∥v∥ .(38)
Let η be a set valued mapping of allowed reflection directions defined on Ω such that η (x) = {0} for x ∈ Ω, and η (x) is a non-empty, closed and convex cone in R d such that {0} ⊆ η (x) for x ∈ ∂Ω, and furthermore assume that the set (x, v) : x ∈ Ω, v ∈ η (x) is closed in R 2d . In addition, for x ∈ ∂Ω let n (x) be the set of inwards normals to Ω at x,
n (x) = r>0 nr (x) , nr (x) = n ∈ R d | ∥n∥ = 1, B r (x -rn) ∩ Ω = ∅ .
Then, denote the set of boundary points with inward pointing cones U ≜ {x ∈ ∂Ω | ∃n ∈ n (x) : ∀η ∈ η (x) ⟨n, η⟩ > 0} , and let V ≜ ∂Ω \ U. For example, if Ω is a convex polyhedron and η (x) is the cone defined by the positive span of n (x) we get that V = ∅.
Throughout this section and the rest of the paper, the stochastic differential equation with reflection (SDER) in (Ω, η)
dx t = b (x t ) dt + Σ (x t ) dw t + dr t ,(39)
where w t is a Wiener process, and r t is a reflection process with respect to some filtration, is understood as in Definition 2.4 of [32], and the submartingale problem associated with (Ω, η), V, b and Σ, refers to Definition 2.9 of [32]. In addition, we use the following definition.
Definition H.1 (Piecewise C 2 with continuous reflection; Definition 2.11 in [32]). The pair (Ω, η) is said to be piecewise C 2 with continuous reflection if it satisfies the following properties:
1. Ω is a non-empty domain in R d with representation Ω = i∈I Ω i , where I is a finite set and for each i ∈ I, Ω i is a non-empty domain with C 2 boundary in the sense that for each x ∈ ∂Ω, there exist a neighborhood N (x) of x, and functions
φ i x ∈ C 2 R d , i ∈ I (x) = i ∈ I | x ∈ ∂Ω i , such that N (x) ∩ Ω i = z ∈ N (x) | φ i x (z) > 0 , N (x) ∩ ∂Ω i = z ∈ N (x) | φ i x (z) = 0 ,
and ∇φ i x ̸ = 0 on N (x). For each x ∈ ∂Ω i and i ∈ I (x), let
n i (x) = ∇φ i x ∥∇φ i
x ∥ denote the unit inward normal vector to ∂Ω i at x.
2. The (set-valued) direction "vector field" η : Ω → R d is given by
η (x) = {0} x ∈ Ω , i∈I(x) α i η i (x) | α i ≥ 0 , i ∈ I (x) x ∈ ∂Ω ,(40)
where for each i ∈ I, η i (•) is a continuous unit vector field defined on ∂Ω i that satisfies for all x ∈ ∂Ω i n i (x) , η i (x) > 0 .
If η i (•) is constant for every i ∈ I, the the pair (Ω, η) is said to be piecewise C 2 with constant reflection. If, in addition, n i (•) is constant for every i ∈ I, then the pair (Ω, η) is said to be polyhedral with piecewise constant reflection.
In addition, let S denote the smooth parts of ∂Ω. Remark H.2. It is clear from the definition that if Ω is polyhedral, i.e. if all Ω i 's are half-spaces, and η consists of inward normal reflections, then (Ω, η) is polyhedral with piecewise constant reflection.
Theorem H.3 (Theorem 3 in [31], simplified). Suppose that the pair (Ω, η) is piecewise C 2 with continuous reflection, for all i ∈ I and x ∈ ∂Ω i , n i (x) , η i (x) = 1, V = ∅, b (•) ∈ C 1 Ω and A ∈ C 2 Ω (elementwise), and the submartingale problem associated with (Ω, η) and V is well posed. Furthermore, suppose there exists a nonnegative function p ∈ C 2 Ω with Z p = Ω p (x) dx < ∞ that solves the PDE defined by the following three relations:
1. For x ∈ Ω:
0 = 1 2 d i,j=1 ∂ 2 ∂x i ∂x j (a ij (x) p (x)) - d i=1 ∂ ∂x i (b i (x) p (x)) .(41)
2. For each i ∈ I and x ∈ ∂Ω ∩ S,
0 = -2p (x) n i (x) , b (x) + n i (x) ⊤ A (x) ∇p (x) -∇ • p (x) q i (x) + p (x) K i (x) ,(42)
where
q i (x) ≜ n i (x) ⊤ A (x) n i (x) η i (x) -A (x) n i (x)
and
K i (x) ≜ n i (x) , ∇ • A (x) = d k=1 n i (x) k d j=1
∂a kj ∂x j (x) .
3. For each i, j ∈ I, i ̸ = j, and x ∈ ∂Ω i ∩ ∂Ω j ∩ ∂Ω, p (x) q i (x) , n j (x) + q j (x) , n i (x) = 0 .
Then the probability measure on Ω defined by
p ∞ (A) ≜ 1 Z p A p (x) dx , A ∈ B Ω ,(44)
is a stationary distribution for the well-posed submartingale problem.
We are now ready to state a characterization of stationary distributions of (39). Note that for simplicity, we do not maintain full generality. Corollary H.4 (Stationary distribution of weak solutions to SDERs). Suppose that, Ω is convex and bounded, b ∈ C 1 Ω and A ∈ C 2 Ω , (Ω, η) is piecewise C 2 with continuous reflection, A is uniformly elliptic (see (38)), and V = ∅. Then p ∈ C 2 satisfying the conditions in Theorem H.3 defines a stationary distribution for (39).
Proof. Assumptions compactness of the domain, and continuous differentiability of the drift and dispersion coefficient imply that they are Lipschitz, hence Exercise 2.5.1 and Theorem 2.5.4 of [57] imply that there exists a unique strong solution to the SDER (39). Then, piecewise C 2 with continuous reflection, the uniform ellipticity assumption, Theorems 1 and 3 of [32], and Theorem H.3 imply that if there exists p ∈ C 2 satisfying ( 41)-(43), then (44) is a stationary distributions of (39).
In the next subsection we use this to derive explicit expressions for the stationary distribution in the setting of this paper.
this section cite: ['b31', 'b31', 'b31', 'b30', 'b38', 'b37', 'b38', 'b56', 'b38', 'b31', 'b38']

Section: H.2 SDER with isotropic diffusion in a box
We proceed to assume that the diffusion term is a scalar matrix of the form A (x) = 2σ 2 (x) I d , and that Ω is a bounded box in R d , i.e. there exist {m i < M i } d i=1 such that
Ω = d i=1 (m i , M i ) = d i=1 Ω i m ∩ Ω i M ,(45)
where
Ω i m ≜ x ∈ R d | x i > m i , Ω i M ≜ x ∈ R d | x i < M i ,(46)
and that the reflecting field is normal to the boundary, i.e. given by (40) with
η i m ≡ n i m ≡ e i ,and
η i M ≡ n i M ≡ -e i(47)
for i = 1, . . . , d. In this setting, we can considerably simplify the conditions in Theorem H.3, as done in the following corollary. Lemma H.5 (Stationarity condition for SDER in a box with normal reflection). Let b (•) ∈ C 1 , and let σ (•) ∈ C 2 be uniformly bounded away from 0, i.e. there exists σ 2 > 0 such that for all x ∈ Ω, σ 2 (x) > σ 2 . If there exists p ∈ C 2 such that
0 = ∇ • ∇ σ 2 (x) p (x) -b (x) p (x) x ∈ Ω , 0 = ∇ σ 2 (x) p (x) -b (x) p (x) , n (x) x ∈ ∂Ω ,(48)
and Ω p (x) dx = 1, then p is a stationary distribution of
dx t = b (x t ) dt + 2σ 2 (x t )dw t + dr t (49
)
in Ω. Remark H.6. ( 48) is exactly the stationarity condition derived from the Fokker-Planck equation with Neumann boundary conditions ensuring conservation of mass.
Proof. Under the assumptions we see that the conditions of Corollary H.4 are satisfied, and we can use ( 41)-( 43) to find stationary distributions of (49). First, notice that (41) simplifies to
0 = 1 2 d i,j=1 ∂ 2 ∂x i ∂x j (a ij (x) p (x)) - d i=1 ∂ ∂x i (b i (x) p (x)) = 1 2 d i=1 ∂ 2 ∂x 2 i 2σ 2 (x) p (x) - d i=1 ∂ ∂x i (b i (x) p (x)) = ∇ • ∇ σ 2 (x) p (x) -b (x) p (x) .
Next, we can considerably simplify the boundary conditions. First, notice that S consists of the interior of the domain's faces so for x ∈ ∂Ω∩S, the set of active boundary regions I (x) is a singleton I (x) = {(i, s)}, for some i = 1, . . . , d and s ∈ {m, M }. We focus on the lower boundaries (m), as the conditions for the upper boundaries are symmetric.
For i = 1, . . . , d and x ∈ ∂Ω ∩ S, η i m (x) = n i m (x) = e i so
q i m (x) = n i m (x) ⊤ A (x) n i m (x) η i m (x) -A (x) n i m (x) = σ 2 (x)
K i m (x) = ∇ • a i (x) = ∂ ∂x i σ 2 (x) ,
so (42) becomes, for all i = 1, . . . , d,
0 = -2p (x) n i m (x) , b (x) + n i m (x) ⊤ A (x) ∇p (x) -∇ • p (x) q i m (x) + p (x) K i m (x) 0 = -2p (x) b i (x) + a ⊤ i ∇p (x) + p (x) ∂ ∂x i σ 2 (x) 0 = -2p (x) b i (x) + σ 2 (x) ∂ ∂x i p (x) + p (x) ∂ ∂x i σ 2 (x) , which is 0 = -p (x) b i (x) + 1 2 ∂ ∂x i p (x) σ 2 (x) = ∇ σ 2 (x) p (x) -b (x) p (x) , n (x) .
this section cite: ['b39', 'b48']

Section: H.2.1 Reflected Langevin dynamics in a box
In this section, we derive some useful properties of the SDER
dx t = -∇L (x t ) + 2β -1 σ 2 (x t )dw t + dr t ,(50)
in a box domain as defined in ( 45)-( 47), where L ≥ 0 is some (loss/potential) function, and β > 0 is an inverse temperature parameter. First, we characterize the stationary distribution of this process.
this section cite: []

Section: Recall Lemma D.1.
If L, σ 2 ∈ C 2 , σ 2 (•) > 0 is uniformly bounded away from 0 in Ω,
Z = Ω 1 σ 2 (x) exp -β ∇L (x) σ 2 (x) dx < ∞ ,
the integrals exist, and the field ∇L/σ 2 is conservative (curl-free), then
p ∞ (x) = 1 Z 1 σ 2 (x) exp -β ∇L (x) σ 2 (x) dx (51
)
is a stationary distribution of (50).
Proof. The drift term in this setting is b = -β∇L. Therefore, from Lemma H.5, we get that any distribution that satisfies
0 = ∇ σ 2 (x) p ∞ (x) + βp ∞ (x) ∇L (x)
on Ω, is a stationary distribution. We can solve this PDE as
0 = β∇L (x) p ∞ (x) + p ∞ (x) ∇σ 2 (x) + σ 2 (x) ∇p ∞ (x) = p ∞ (x) β∇L (x) + ∇σ 2 (x) + σ 2 (x) ∇p ∞ (x)
this section cite: ['b49']

Section: References
Ref_id:b0 Title: User-friendly introduction to pac-bayes bounds Year: (2024)
Ref_id:b1 Title: Neural network learning: Theoretical foundations Year: (2009)
Ref_id:b2 Title: Stronger generalization bounds for deep nets via a compression approach Year: (2018)
Ref_id:b3 Title: What is the long-run distribution of stochastic gradient descent? a large deviations analysis Year: (2024)
Ref_id:b4 Title: Diffusions hypercontractives Year: (1985)
Ref_id:b5 Title: Spectrally-normalized margin bounds for neural networks Year: (2017)
Ref_id:b6 Title: Stability and generalization Year: (2002-03)
Ref_id:b7 Title: How uniform random weights induce non-uniform bias: Typical interpolating neural networks generalize with narrow teachers Year: (2024-07)
Ref_id:b8 Title: Pac-bayesian supervised classification: the thermodynamics of statistical learning Year: (2007)
Ref_id:b9 Title: Loss landscapes are all you need: Neural network generalization can be explained without the implicit bias of gradient descent Year: (2022)
Ref_id:b10 Title: Which processes satisfy the second law? Year: (1994)
Ref_id:b11 Title: Entropy, Relative Entropy and Mutual Information, chapter 2 Year: (2001)
Ref_id:b12 Title: Generalization of noisy SGD in unbounded nonconvex settings Year: (2025)
Ref_id:b13 Title: Uniform generalization bounds on data-dependent hypothesis sets via pac-bayesian theory on random sets Year: (2024)
Ref_id:b14 Title: Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data Year: (2017)
Ref_id:b15 Title: The size of teachers as a measure of data complexity: Pac-bayes excess risk bounds and scaling laws Year: (2025-05)
Ref_id:b16 Title: On the role of data in pac-bayes bounds Year: (2021)
Ref_id:b17 Title: Time-independent generalization bounds for sgld in non-convex settings Year: (2021)
Ref_id:b18 Title: Time-independent information-theoretic generalization bounds for sgld Year: (2023)
Ref_id:b19 Title: Stochastic training is not necessary for generalization Year: (2022)
Ref_id:b20 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010-05)
Ref_id:b21 Title: Size-independent sample complexity of neural networks Year: (2018)
Ref_id:b22 Title: Implicit regularization in matrix factorization Year: (2017)
Ref_id:b23 Title: Matrix Variate Distributions. Monographs and Surveys in Pure and Applied Mathematics Year: (1999)
Ref_id:b24 Title: Random neural networks in the infinite width limit as gaussian processes Year: (2023)
Ref_id:b25 Title: Train faster, generalize better: Stability of stochastic gradient descent Year: (2016)
Ref_id:b26 Title: Monte carlo sampling methods using markov chains and their applications Year: (1970)
Ref_id:b27 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b28 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b29 Title: How to escape saddle points efficiently Year: (2017)
Ref_id:b30 Title: Characterization of stationary distributions of reflected diffusions Year: (2014)
Ref_id:b31 Title: On the submartingale problem for reflected diffusions in domains with piecewise smooth boundaries Year: (2017)
Ref_id:b32 Title: High probability bounds for a class of nonconvex algorithms with adagrad stepsize Year: (2022)
Ref_id:b33 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b34 Title: Deep neural networks as gaussian processes Year: (2018)
Ref_id:b35 Title: Stability and generalization analysis of gradient methods for shallow neural networks Year: (2022)
Ref_id:b36 Title: STORM+: Fully adaptive SGD with recursive momentum for nonconvex optimization Year: (2021)
Ref_id:b37 Title: On generalization error bounds of noisy gradient methods for non-convex learning Year: (2020)
Ref_id:b38 Title: Pac-bayes compression bounds so tight that they can explain generalization Year: (2022)
Ref_id:b39 Title: Gradient descent maximizes the margin of homogeneous neural networks Year: (2020)
Ref_id:b40 Title: Stochastic gradient descent as approximate bayesian inference Year: (2017)
Ref_id:b41 Title: Gaussian process behaviour in wide deep neural networks Year: (2018)
Ref_id:b42 Title: A note on the pac bayesian theorem Year: (2004)
Ref_id:b43 Title: Some pac-bayesian theorems Year: (1998)
Ref_id:b44 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b45 Title: Data processing theorems and the second law of thermodynamics Year: (2011)
Ref_id:b46 Title: Equation of state calculations by fast computing machines Year: ()
Ref_id:b47 Title: Human-level control through deep reinforcement learning Year: (2015)
Ref_id:b48 Title: Generalization bounds of sgld for non-convex learning: Two theoretical viewpoints Year: (2018)
Ref_id:b49 Title: Priors for Infinite Networks Year: (1996)
Ref_id:b50 Title: Information-theoretic generalization bounds for sgld via data-dependent estimates Year: (2019)
Ref_id:b51 Title: A method for solving the convex programming problem with convergence rate Year: (1983)
Ref_id:b52 Title: Reading digits in natural images with unsupervised feature learning Year: (2011)
Ref_id:b53 Title: Stochastic Differential Equations Year: (2003)
Ref_id:b54 Title: Generalization error bounds for noisy, iterative algorithms Year: (2018)
Ref_id:b55 Title: The matrix cookbook Year: (2012-11)
Ref_id:b56 Title: An introduction to stochastic differential equations with reflection Year: ()
Ref_id:b57 Title: Informationtheoretic analysis of stability and bias of learning algorithms Year: (2016)
Ref_id:b58 Title: Non-convex learning via stochastic gradient langevin dynamics: a nonasymptotic analysis Year: (2017)
Ref_id:b59 Title: Learning with gradient descent and weakly convex losses Year: (1990)
Ref_id:b60 Title: How much does your data exploration overfit? controlling bias via information usage Year: (2020)
Ref_id:b61 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b62 Title: Euler's Scheme and Wiener's Measure Year: (2013)
Ref_id:b63 Title: The implicit bias of gradient descent on separable data Year: (2018)
Ref_id:b64 Title: Rényi divergence and kullback-leibler divergence Year: (2014)
Ref_id:b65 Title: On the implicit bias in deep-learning algorithms Year: (2023)
Ref_id:b66 Title: Generalization guarantees of gradient descent for shallow neural networks Year: (2025)
Ref_id:b67 Title: Variational deep learning via implicit regularization Year: (2025)
Ref_id:b68 Title: How good is the bayes posterior in deep neural networks really Year: (2020)
Ref_id:b69 Title: Information-theoretic analysis of generalization capability of learning algorithms Year: (2017)
Ref_id:b70 Title: Understanding deep learning requires rethinking generalization Year: (2017)
