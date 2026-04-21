Title: Temporal Difference Flows
Abstract: Predictive models of the future are fundamental for an agent's ability to reason and plan. A common strategy learns a world model and unrolls it step-by-step at inference, where small errors can rapidly compound. Geometric Horizon Models (GHMs) offer a compelling alternative by directly making predictions of future states, avoiding cumulative inference errors. While GHMs can be conveniently learned by a generative analog to temporal difference (TD) learning, existing methods are negatively affected by bootstrapping predictions at train time and struggle to generate high-quality predictions at long horizons. This paper introduces Temporal Difference Flows (TD-Flow), which leverages the structure of a novel Bellman equation on probability paths alongside flow-matching techniques to learn accurate GHMs at over 5× the horizon length of prior methods. Theoretically, we establish a new convergence result and primarily attribute TD-Flow's efficacy to reduced gradient variance during training. We further show that similar arguments can be extended to diffusion-based methods. Empirically, we validate TD-Flow across a diverse set of domains on both generative metrics and downstream tasks, including policy evaluation. Moreover, integrating TD-Flow with recent behavior foundation models for planning over policies demonstrates substantial performance gains, underscoring its promise for long-horizon decision-making.

Section: Introduction
Predictive modeling lies at the heart of intelligent decisionmaking, enabling agents to reason and plan in complex environments. In Reinforcement Learning (RL), this pre-dictive capability has traditionally been achieved through world models that capture the transition structure of the environment. These models have enabled significant advances across numerous domains -from robotics manipulation employing model-predictive control (Sikchi et al., 2021;Hafner et al., 2023;Hansen et al., 2022;2024), to sampleefficient exploration strategies (Schmidhuber, 1991;Stadie et al., 2016;Pathak et al., 2017), and sophisticated planning algorithms (Silver et al., 2016;2017;Schrittwieser et al., 2020). However, while world models have demonstrated impressive results, they face fundamental limitations when deployed for long-horizon reasoning. The standard approach of unrolling predictions step-by-step leads to compounding errors, as small inaccuracies in each prediction accumulate and propagate forward in time (Talvitie, 2014;Jafferjee et al., 2020;Lambert et al., 2022). This "curse of horizon" presents a significant challenge for applications requiring reliable long-range predictions.
An alternative approach is to learn a generative model of future states directly, avoiding compounding errors during inference. These models, usually referred to as Geometric Horizon Models (GHM; Thakoor et al., 2022) or γ-models (Janner et al., 2020), are learned by leveraging the temporal difference structure of the successor measure (Blier et al., 2021). However, their reliance on bootstrapped predictions during training can lead to instability and growing inaccuracy over long horizons. As a result, current methods struggle to make accurate predictions beyond 20-50 steps, also limiting their utility for long-term decisionmaking. In this paper, we show that while state-of-the-art generative methods like flow matching (Lipman et al., 2023) and denoising diffusion (Ho et al., 2020) cannot be directly applied to learn long-horizon GHMs, their iterative nature can be leveraged to better exploit the temporal difference structure of the problem. This insight yields a new class of methods that provably converges to the successor measure while reducing the variance of their sample-based gradient estimates, enabling stable long-horizon predictions. Empirically, our approach produces significantly more accurate GHMs at all horizons, consistently outperforming state-ofthe-art algorithms across domains and metrics, including prediction accuracy, value function estimation, and generalized policy improvement.
this section cite: ['b23', 'b50', 'b29', 'b36', 'b32', 'b6', 'b27']

Section: Background
In the following, we use capital letters to denote random variables, sans-serif fonts for sets, and P(A) to denote the space of probability measures over a measurable set A.
this section cite: []

Section: Markov Decision Process
We consider a reward-free discounted Markov decision process M = (S, A, P, γ), which characterizes the dynamics of a sequential decision-making problem. At each step, the agent selects an action a ∈ A in state s ∈ S according to its policy π : S → A. This action influences the transition to the next state s ′ ∈ S, governed by the transition kernel P : S × A → P(S), which defines a probability measure over successor states. The discount factor γ ∈ [0, 1) can be interpreted as implying a process that either continues with probability γ or terminates with probability 1 -γ. This interpretation naturally defines a geometric distribution of future states the agent will occupy, where states reached after k steps are discounted by γ k .
this section cite: []

Section: Successor Measure
The normalized successor measure (Dayan, 1993;Blier et al., 2021) of a policy π describes the discounted distribution of future states visited by π starting from an initial state-action pair (s, a). For the measurable subset X ⊆ S the successor measure m π (X | s, a) represents the probability that future states fall within X, geometrically discounted by γ according to the time of visitation. Formally, it is defined as:
m π ( X | s, a ) = (1
) (1 -γ) ∞ k=0 γ k Pr(S k+1 ∈ X | S 0 = s, A 0 = a, π),
where Pr(• | S 0 , A 0 , π) denotes the probability of stateaction sequences (S k , A k ) k≥0 generated from (S 0 , A 0 ) following S k ∼ P (• | S k-1 , A k-1 ) and A k = π(S k ). The successor measure encapsulates the long-term dynamics of π, enabling value estimation for any reward function r : S → R. Specifically, the value of taking action a ∈ A in state s ∈ S is the expected reward under states visited by π amplified by the effective horizon (1 -γ) -1 :
Q π (s, a) = (1 -γ) -1 E X∼m π (•|s,a) [r(X)] . (2)
Moreover, m π is the fixed point of the Bellman operator T π : P(S) S×A → P(S) S×A (Thakoor et al., 2022): Geometric Horizon Model A Geometric Horizon Model (GHM; Thakoor et al., 2022) or γ-model (Janner et al., 2020) is a generative model of the normalized successor measure. To learn the parametric model m(• • • ; θ) ≈ m π we can minimize a Monte-Carlo cross-entropy objective over source states from the empirical distribution ρ as,
m π (• | s, a) = (T π m π ) (• | s, a)(3)
E s∼ρ, X∼m π (•|S,π(A)) [ -log m(X | S, A; θ)) ] .
In order to sample from m π we deploy policy π for t ∼ Geom(1 -γ) steps resulting in state X = S t . Similar to other Monte Carlo methods in RL, this approach is problematic when learning from off-policy data, often resulting in high-variance estimators that rely on importance sampling.
Alternatively, we can leverage the Bellman equation ( 3) to construct an off-policy iterative method for estimating m π . Given initial weights θ (0) , each iteration updates θ by minimizing the following temporal-difference cross-entropy objective over transitions that need not come from policy π,
E (S,A)∼ρ,X∼(T π m (n) )(•|S,A) [-log m(X | S, A; θ)]. (4)
In the equation above and throughout the paper, we adopt the shorthand m
(n) = m(• • • ; θ (n) ). To generate samples X ∼ T π m (n) (• | S, A)
we first draw a successor state S ′ ∼ P (• | S, A); then with probability 1 -γ, we return S ′ ; otherwise, with probability γ, we return a bootstrapped sample drawn from m (n)
(• | S ′ , π(S ′ )).
Several probabilistic models have been applied to this problem, including generative adversarial networks (e.g., Janner et al., 2020;Wiltzer et al., 2024b), normalizing flows (e.g., Janner et al., 2020), and variational auto-encoders (e.g., Thakoor et al., 2022;Tomar et al., 2024). We now turn our attention to a class of generative models based on the flowmatching framework specifically designed to leverage the underlying structure of the Bellman equation (3), enabling more effective generative models of the successor measure.
this section cite: ['b11', 'b6', 'b32', 'b32', 'b32']

Section: Temporal Difference Flows
Flow Matching (FM; Lipman et al., 2023;2024;Liu et al., 2023;Albergo & Vanden-Eijnden, 2023) constructs a timedependent probability path m t : S × A → P(S) for t ∈ [0, 1] that evolves smoothly from the source distribution m 0 = p 0 ∈ P(S) to the target distribution m 1 ≈ m π . This evolution is governed by a vector field v t : S × S × A → S, which dictates the instantaneous movement of samples along m t . The relationship between v t and the resulting probability path m t is established through a time-dependent flow ψ t : S × S × A → S, defined by the following ODE: We say that v t generates m t if its flow ψ t satisfies X t := ψ t (X 0 | S, A) ∼ m t (• | S, A) for X 0 ∼ m 0 . In words, the flow ψ t pushes samples forward through time, ensuring they are distributed according to m t at time t. To learn this transformation, we can minimize the squared L 2 distance between a parameterized vector field ṽt (• • • ; θ) and the true vector field v t over t ∼ U([0, 1]), yielding the Monte-Carlo Flow Matching (MC-FM) loss ℓ MC-FM (θ):
E ρ,t,Xt ṽt (X t | S, A; θ) -v t (X t | S, A) 2 ,
where X t ∼ m t (• | S, A) . (MC-FM; 5)
Despite its conceptual simplicity, direct optimization of the flow matching objective above proves challenging due to the inaccessibility of the true probability path m t and its associated vector field v t .
Alternatively, Lipman et al. (2023) shows that we can sidestep this problem entirely by introducing additional conditioning information. Instead of directly modeling the probability path m t we can introduce a random variable Z and define a conditional path on Z as p t|Z : S × Z → P(S) (Lipman et al., 2024;Tong et al., 2024). The conditional velocity field u t|Z : S × Z → S that generates p t|Z can now be computed in closed form for many simple choices of Z and p t|Z . One such choice is taking Z = X 1 and performing a linear Gaussian interpolation from
X 0 → X 1 resulting in p t|1 (• | X 1 ) = N (• | tX 1 , (1 -t) 2 I)
with the corresponding vector field given by u
t|1 (x | X 1 ) = (X 1 -x)/(1 -t).
Armed with the ability to sample from p t|1 and to compute u t|1 , we can directly learn ṽt by optimizing the Monte-Carlo Conditional Flow Matching (MC-CFM) objective ℓ MC-CFM (θ):
E ρ,t,Z,Xt ṽt (X t | S, A; θ) -u t|Z (X t | Z) 2 , where Z = X 1 ∼ m π (• | S, A) , X t ∼ p t|Z (• | Z) . (MC-CFM; 6)
Remarkably, both (MC-FM; 5) and (MC-CFM; 6) share the same gradient and converge to the same solution.
Proposition 1 (Lipman et al. 2024). Given a conditional probability path p t|Z and vector field u t|Z with their associated marginal counterparts p t (x) and v t (x), we have
∇ θ ℓ MC-FM (θ) = ∇ θ ℓ MC-CFM (θ).
TD-CFM While (MC-CFM; 6) requires direct access to samples from the target distribution m π , we can instead learn from an offline dataset ρ containing only one-step transitions (S, A, S ′ ) through an iterative process similar to ( 4). Starting with initial parameters θ (0) , at each iteration, we minimize the TD-Conditional Flow Matching (TD-CFM) loss ℓ TD-CFM -an extension of (MC-CFM; 6) that differs only in its sampling procedure:
X 0 ∼ p 0 Z = X 1 ∼ (1 -γ) δ S ′ + γ δ ψ (n) 1 (X0 | S ′ ,π(S ′ )) .
(TD-CFM; 7) In this procedure, with probability 1 -γ, we return the successor state S ′ . Otherwise, with probability γ we sample from the neural ordinary differential equation (Chen et al., 2018)
ψ (n) t with corresponding vector field ṽ(n) t (X t | S ′ , π(S ′ )) from X 0 ∼ p 0 to produce a sample X 1 ∼ m (n) (• | S ′ , π(S ′ )).
Coupled TD-CFM Although (TD-CFM; 7) offers a principled way of learning the flow from noise to data, an increasingly popular strategy to improve flow matching methods is to correlate noise and data whenever a "natural" coupling is available (e.g., Liu et al., 2023;Shi et al., 2023;Pooladian et al., 2023;Tong et al., 2024;De Bortoli et al., 2024). Motivated by this idea, we observe that the process used to generate X 1 described above already provides a direct coupling between X 0 and X 1 . We can leverage this coupling by conditioning the probability path p t|Z on both endpoints, i.e., Z = (X 0 , X 1 ), rather than just conditioning on Z = X 1 as in TD-CFM. As illustrated in Figure 1, this coupling helps align X t with the path generated by ψ (n) t , potentially simplifying the regression problem. This procedure gives rise to the Coupled TD-Conditional Flow Matching (TD-CFM(C)) loss ℓ TD-CFM(C) which now extends ℓ TD-CFM , again, differing only in its sampling procedure:
X 0 ∼ p 0 X 1 ∼ (1 -γ) δ S ′ + γ δ ψ (n) 1 (X0|S ′ ,π(S ′ )) Z = (X 0 , X 1 ) .
(TD-CFM(C); 8)
A convenient approach to specifying the conditional path p t|Z is to define X t = ϕ t (X 0 , X 1 ) = α t X 1 + β t X 0 as the affine interpolant between X 0 and X 1 , with the interpolation coefficients satisfying the boundary conditions α 0 = β 1 = 0, α 1 = β 0 = 1, and monotonicity constraints αt > 0, -βt > 0, where the over-dot denotes the time derivative. From this definition, the conditional vector field arises as the time derivative of this interpolant defined as bergo et al., 2023). A simple choice of the interpolation coefficients that yields a linear (straight-line) conditional path is given by β t = 1 -α t = 1 -t.
u t|0,1 (X t | X 0 , X 1 ) = φt (X 0 , X 1 ) = αt X 1 + βt X 0 (Al
TDfoot_1 -CFM While (TD-CFM(C); 8) improves upon (TD-CFM; 7) by accounting for the coupling between bootstrapped samples and their generating noise, both methods rely upon fitting an ad-hoc conditional vector field u t|Z that generates the surrogate conditional path p t|Z . To formulate a more structured approach, we exploit the linearity of the Bellman equation, as detailed in the following result.
Lemma 1. Let → p t be a probability path for P generated by vector field
→ v t and ↷ p (n) t be a probability path for P π m (n) 1 generated by ↷ v (n) t such that → p 0 = ↷ p (n) 0 = m 0 . For any t ∈ [0, 1] and (s, a) let v (n+1) t (• | s, a) be the solution of 1 arg min v : R d →R d (1 -γ)E → Xt∼ → pt(•|s,a) v( → X t ) - → v t ( → X t | s, a) 2 + γE ↷ Xt∼ ↷ p (n) t (•|s,a) v( ↷ X t ) - ↷ v (n) t ( ↷ X t | s, a) 2 . Then v (n+1) t induces a probability path m (n+1) t such that m (n+1) 0 = m 0 and m (n+1) 1 = T π m (n) 1 .
This result shows that it is possible to use two independent probability paths for the two terms in the sampling process induced by the Bellman operator. For the first term, we can use a standard CFM approach for Z = X 1 with conditional path
v (n) t (x|s, a) = v (n) t (x|s ′ , a ′ ) m (n) t (x|s ′ , a ′ )P (ds ′ |s, a) ↷ p (n) t (x|s, a) , where ↷ p (n) t (x | s, a) = m (n) t (x | s ′ , a ′ )P (ds ′ | s, a)
, and a ′ = π(s ′ ). This shows that m (n) t plays the role of a conditional probability path for the bootstrapped term and v (n) t is its associated conditional vector field. We can then use the equivalence between FM and CFM in Proposition 1 to replace the marginal probability paths and vector fields in Lemma 1 with their conditional counterparts to obtain the loss:
→ ℓ(θ) = E ρ,t,Z, → Xt ṽt ( → X t | S, A; θ) - → u t|Z ( → X t | Z) 2 , where Z = X 1 ∼ P (• | S, A), → X t ∼ → p t|Z (• | Z) , ↷ ℓ(θ) = E ρ,t, ↷ Xt ṽt ( ↷ X t | S, A; θ) - ṽ(n) t ( ↷ X t | S ′ , π(S ′ ) 2 ,
where
X 0 ∼ p 0 , S ′ ∼ P (• | S, A), ↷ X t = ψ (n) t (X 0 | S ′ , π(S ′ )) , ℓ TD 2 -CFM (θ) = (1 -γ) → ℓ(θ) + γ ↷ ℓ(θ) . (TD 2 -CFM; 9)
Since we now bootstrap the previous estimate not only in the sampling process but also in the objective function, we refer to this method as TD 2 -Conditional Flow Matching (TD 2 -CFM). The right panel of Figure 1 depicts the process of obtaining the bootstrapped vector field ṽ(n) t for TD 2 -CFM. We provide further implementation details and pseudo-code for all TD-Flow methods in Appendix C.3.1. Next, we extend our TD 2 result to the class of denoising diffusion models.
this section cite: ['b1', 'b41', 'b41', 'b10']

Section: Extension to Diffusion Models
Denoising Diffusion models (Sohl-Dickstein et al., 2015;Ho et al., 2020) build a diffusion process starting from a data sample X 0 ∼ q 0 = m π (• | S, A) 2 and corrupting it via a stochastic differential equation (SDE),
dX t = f (t) X t dt + g(t) dW t ,(10)
where t ∈ [0, T ] for some time horizon T , f, g : [0, T ] → R is drift and diffusion term, and W t ∈ R d is a standard Brownian motion. The forward process of the linear SDE (10) has an analytic Gaussian kernel q
t|0 (• | X 0 ) = N (• | α t X 0 , σ 2 t I
), where α t and σ t can be computed in closed form. To sample from the target data distribution q 0 , we can solve the reverse SDE (Song & Ermon, 2019) from time T to 0:
dX t = f (t) X t -g(t) ∇ Xt log q t (X t | S, A) dt+g(t) dW t(11)
where W t is the reverse-time Brownian motion and q t is the marginal distribution of both the forward (16) and reverse (17) process.
To simulate (11), we can train a parametrized score function st (x | s, a; θ) to approximate ∇ xt log q t (x t | s, a) using the denoising diffusion / score matching objective (Vincent, 2011) ℓ DD (θ):
E ρ,t,X0,Xt st (X t | S, A; θ) -∇ Xt log q t|0 (X t | X 0 ) 2 ,
where
X 0 ∼ m π (• | S, A), X t ∼ q t|0 (• | X 0 ) . (DD; 12)
Temporal Difference Diffusion Following the blueprint in §3, we define an iterative process starting from s(0
) = s(• • • ; θ (0)
) and minimize at each iteration the Temporal-Difference Denoising Diffusion (TD-DD) loss ℓ TD-DD (θ):
E ρ,t,X0,Xt s(X t | S, A; θ) -∇ x log q t|0 (X t | X 0 ) 2 ,
where
X 0 ∼ T π m (n) 0|T (• | S, A), X t ∼ q t|0 (• | X 0 ) . (TD-DD; 13) Once again, to sample X 0 ∼ T π m (n) 0|T (• | S, A)
, we proceed as follows: with probability 1 -γ, we draw a successor state S ′ ∼ P (• | S, A); conversely, with probability γ, we sample from the bootstrapped model by solving the reverse SDE with score function s(n) , initiated from X T . Following an approach analogous to Lemma 1, we demonstrate in Appendix B that we can employ two distinct diffusion processes for the two terms involved in the Bellman operator, which consequently leads to the TD 2 -DD objective:
→ ℓ(θ) = E ρ,t, → Xt st ( → X t | S, A; θ) -∇ → Xt q t|0 ( → X t | S ′ ) 2 , where → X t ∼ q t|0 (• | S ′ ) , ↷ ℓ(θ) = E ρ,t, ↷ Xt st ( ↷ X t | S, A; θ) - s(n) t ( ↷ X t | S ′ , π(S ′ ) 2 , where X T ∼ q T , ↷ X t ∼ q (n) t|T (• | S ′ , π(S ′ )) , ℓ TD 2 -DD (θ) = (1 -γ) → ℓ(θ) + γ ↷ ℓ(θ) . (TD 2 -DD; 14)
this section cite: ['b27']

Section: Theoretical Analysis
We now study the learning dynamics of an idealized version of the TD-Flow methods, assuming that the flow-matching loss is minimized exactly at each iteration. Under this assumption, at each iteration we compute a probability path m
(n) t such that m (n) 1 = T π m (n-1) 1
, which implies that m (n) 1 → m π by the contraction property of T π . The following result shows that the overall probability paths m (n) t follow a similar process. Proofs are deferred to Appendix E.
Theorem 1. For any n ≥ 1, the probability paths generated by TD-CFM, TD-CFM(C), or TD 2 -CFM satisfy
m (n+1) t (x | s, a) = B π t m (n) t (x | s, a), ∀ t ∈ [0, 1]
where B π t m := (1 -γ)P t + γP π m and P t (x|s, a) := p t|1 (x | x 1 )P (x 1 |s, a)dx 1 . For any t ∈ [0, 1], the operator B π t is a γ-contraction in 1-Wasserstein distance, that is, for any couple of probability paths p t , q t , sup s,a
W 1 ((B π t p t ) (• | s, a), (B π t q t ) (• | s, a)) ≤ γ sup s,a W 1 (p t (• | s, a), q t (• | s, a)) .
Theorem 1 shows that all TD-flow methods fundamentally implement the same update where the probability path at t ∈ [0, 1] is obtained by applying a Bellman-like operator B t to the previous iteration. This operator is a γ-contraction as T π , directly implying the following result.
Corollary 1. Let {m (n)
t } n≥0 be the sequence of probability paths produced by TD-CFM, TD-CFM(C), or TD 2 -CFM starting from an arbitrary vector field v (0) t . Then,
lim n→∞ m (n) t = m t = B t m t ,
where m t is the unique fixed point of B t , and m t = m MC t , where
m MC t (• | s, a) = p t|1 (• | x 1 ) m π (x 1 | s, a)
is the probability path of the Monte-Carlo approach (MC-CFM; 6). This corollary shows that the fixed point of B t coincides with the probability path generated in Monte-Carlo Conditional Flow Matching (MC-CFM; 6), which assumes direct access to samples of m π . An important subtlety in Theorem 1 is that all algorithms apply the same operator for n ≥ 1, but the result holds for n = 0 only for TD 2 -CFM. This means that even starting from the same θ (0) , the three algorithms may generate different sequences {m (n) t } n≥0 , while still converging to m t . In Theorems 5 and 6 , we show we can reconcile TD-CFM(C) and TD-CFM with TD 2 -CFM under a mild assumption on the form of the initial vector field.
While Theorem 1 analyzes an idealized version of the algorithms, in practice gradients are estimated from samples and the following analysis reveals important differences in their variance. We introduce the (unbiased) sample-based gradients for each of the algorithms,
E g TD-CFM (Y TD-CFM ) = ∇ θ ℓ TD-CFM (θ), E g TD-CFM(C) (Y TD-CFM(C) ) = ∇ θ ℓ TD-CFM(C) (θ) E g TD 2 -CFM (Y TD 2 -CFM ) = ∇ θ ℓ TD 2 -CFM (θ),
where Y summarizes the random variables involved in the loss definitions in (TD-CFM; 7), (TD-CFM(C); 8), and (TD 2 -CFM; 9) (see Appendix E.6 for a formal definition of the gradients). We want to compare the total variance of the gradient estimates σ 2 = Tr Cov Y [ g(Y ) ] , where Tr denotes the trace.
Theorem 2. For any n ≥ 1 and t ∈ [0, 1], assume that m
(n) t (x | s, a) = p t|1 (x | x 1 )m (n) 1 (x 1 | s, a)dx 1 , then σ 2 TD-CFM = σ 2 TD 2 -CFM + γ 2 E Tr Cov X1|s,a,Xt ∇ θ v t (X t |s, a; θ) ⊤ u t|1 (X t |X 1 ) .
Theorem 3. For any n ≥ 1 and t ∈ [0, 1], assume that m
(n) t (x | s, a) = p t|0,1 (x | x 0 , x 1 )m (n) 0,1 (x 0 , x 1 | s, a)dx 0 dx 1 3 , then we obtain σ 2 TD-CFM(C) = σ 2 TD 2 -CFM + γ 2 E Tr Cov Z|S,A,Xt ∇ θ v t (X t |S, A; θ) ⊤ u t|Z (X t |Z) ,
where Z = (X 0 , X 1 ). Furthermore, if we use straight conditional paths, i.e., X t = tX 1 + (1 -t)X 0 , and the linear interpolant X t does not intersect for any s, a, s ′ , then σ 2 TD-CFM(C) = σ 2 TD 2 -CFM .
In both results, the probability path m
(n) t from the previous iteration must be identical for the algorithms being compared. The analysis reveals that TD-CFM and TD-CFM(C) suffer from a larger variance compared to TD 2 -CFM, which uses the vector field v (n) both to sample X t and as a target for the regression problem. This variance gap is "discounted" by γ 2 , which suggests that the performance of these algorithms would be similar for problems with small horizons but would increase as γ → 1. The extra variance in both cases stems from samples generated by the algorithm (i.e., they do not depend on the transitions available in the dataset). In this sense, we can refer to it as computational variance, and in principle, it could be reduced by increasing the number of samples X 0 , X 1 , and X t used in gradient computation. While the variance of TD-CFM and TD-CFM(C) cannot be directly compared, we expect that constructing X t from X 0 and X 1 (instead of X 1 only) will tend to reduce its variance. Specifically, when X t is obtained by linear interpolation between X 0 and X 1 , and it does not generate crossing paths, the variance of TD-CFM(C) reduces to the one of TD 2 -CFM.
this section cite: []

Section: Experiments
We now present a series of experiments to assess the efficacy of our TD-based flow and diffusion approaches with baselines employing Generative Adversarial Networks (Goodfellow et al., 2014) and β-Variational Auto-Encoders (Higgins et al., 2017).
m (n) 0,1 (x0, x1|s, a) = m0(x0)δ ψ (n) 1 (x 0 |s,a) (x1)
is the joint distribution of (X0, X1), i.e the endpoints of the ODE.
For a single policy, we evaluate how well each method models its i) successor measure and ii) value function. While lower errors in estimating the successor measure are expected to lead to better value estimation, this is not always the case since modeling errors may disproportionally affect states with negligible rewards. Additionally, motivated by our theoretical results, we explore how the probability path's design affects our proposed methods' relative performance.
Finally, we examine the scalability of our approach by learning a generative model of the successor measure across a class of parameterized policies derived from the Forward-Backward (FB) representation (Touati & Ollivier, 2021;Touati et al., 2023), a non-generative model of the successor measure. We conclude by demonstrating how TD 2 enables more effective planning for task-relevant policies when performing Generalized Policy Improvement (GPI; Barreto et al., 2017), far surpassing the capabilities of FB alone.
this section cite: ['b21', 'b26', 'b4']

Section: Empirical Evaluation of Geometric Horizon Models
Before benchmarking, we must first obtain a policy to evaluate. We follow the approach taken in Thakoor et al. (2022) and pre-train a set of deterministic policies -one for each task -using TD3 (Fujimoto et al., 2018). The final policy obtained from this pre-training phase is now fixed for the remainder of our experiments. GHM training proceeds in an off-policy manner where we learn the successor measure of a TD3 policy using transition data from the ExoRL dataset (Yarats et al., 2022); specifically, we use a dataset of 10M transitions collected by a random network distillation policy (Burda et al., 2019). All GHM methods are trained for 3M gradient steps using the AdamW optimizer (Loshchilov & Hutter, 2019) with a batch size of 1024 and weight decay of 0.001. We maintain a target network using an exponential moving average of the training parameters with a step size of 0.001. Special care was taken to match the capacity of the neural networks between methods with a UNet-style architecture employed for all flow and diffusion methods, while the GAN and VAE baselines use an MLP with residual connections for all their respective networks. Full details for the training methodology, network architecture, and hyperparameters can be found in Appendix C. We implement all conditional flow matching methods (TD-CFM, TD-CFM(C), TD 2 -CFM) with the Optimal Transport Gaussian conditional path from Lipman et al. (2023). When constructing our bootstrap targets, we sample from the neural ODE using the Midpoint solver with a constant step size of t/10 for a maximum of 10 steps. For TD 2 -CFM, we sample t ∼ U([0, 1]); otherwise, we integrate to t = 1 and construct X t using the conditional path. For Denoising Diffusion methods (TD-DD, TD 2 -DD), we train a DDPM (Ho et al., 2020) by discretizing β ∈ (0.1, 20) using T = 1, 000 steps. We construct diffusion bootstrapped targets using
5 10 20 50 100 Effective Horizon 10 4 10 2 10 0 10 2 10 4 Value Function MSE TD-CFM TD²-CFM TD-CFM(C) TD-DD TD²-DD TD-GAN TD-VAE
this section cite: ['b18', 'b8', 'b43', 'b27']

Section: Scaling Effective Horizon
Figure 2. Value-Function prediction error as a function of the effective horizon (1 -γ) -1 for γ ∈ {0.8, 0.9, 0.95, 0.98, 0.99} on the POINTMASS loop task. TD 2 methods show impressive robustness to increasingly long-horizon predictions.
20 steps of the DDIM (Song et al., 2021a) sampler. For TD-DD, we solve to t = 0 and regress towards the noise that re-corrupted our sample. Alternatively, TD 2 -DD directly regresses towards the noise prediction from the target network at a randomly selected noise level. The first baseline we consider is a GHM instantiated as a Generative Adversarial Network (Goodfellow et al., 2014) similar to the one found in Janner et al. (2020). We follow the best practices from Huang et al. (2024) with the primary modification being a relativistic discriminator (Jolicoeur-Martineau, 2019) equipped with a zero-centered gradient penalty on both real and fake samples. For our second baseline, we implement a β-VAE (Higgins et al., 2017) following the practices outlined in Thakoor et al. (2022).
To evaluate the quality of our models, we first generate samples from the ground truth successor measure m π according to the following procedure. We first randomly sample 64 source states S 0 from the initial state distribution and execute policy π for 1, 000 steps. Along each trajectory, we resample 2048 states with replacement according to the stopping time t ∼ Geometric(1 -γ). For the same 64 source states, we generate a matching set of 2048 samples from each GHM. Now in possession of these two sets of samples, we evaluate the: 1) log-likelihood of the true samples for models with tractable densities (i.e., diffusion and flow methods); 2) Earth Mover's Distance (EMD; Rubner et al., 2000), which quantifies the minimal transport cost between the two empirical distributions; and 3) mean-squared error of a Monte-Carlo estimate of the true value function Q π and the value function derived from GHM samples using (2). Full details can be found in Appendix C.1.
Having established our training framework, baselines, and evaluation protocol, we proceed to investigate a key prediction from our theoretical analysis. Our variance analysis suggests that our TD-Flow framework should enable more stable training across extended temporal horizons. To validate this hypothesis, we train each GHM for 3 seeds on the loop task in the Maze domain while varying the effective horizon (1 -γ) -1 across five values: {5, 10, 20, 50, 100}.
Figure 2 illustrates the relationship between value function MSE and the effective horizon. The results demonstrate that TD 2 -based methods maintain consistent performance even as the effective horizon increases, while alternative approaches show significant performance degradation. Notably, at an effective horizon of 100, TD 2 -based methods maintain their accuracy and achieve performance improvements of nearly four orders of magnitude compared to their naive implementations. These results empirically support for our initial hypothesis, with the stability of TD 2 methods aligning with our predictions.
In the following, we shift our attention to a more in-depth analysis of the largest horizon of 100 (γ = 0.99). For each algorithm, we train a GHM for 3 independent seeds for all domains and tasks. Table 1 reports aggregate performance across our full suite of metrics. For each domain and metric, we highlight results in a 1% range with respect to the bestperforming method. The results demonstrate a clear pattern of superior performance for TD 2 -based algorithms: TD 2 -CFM achieves significant improvements over TD-CFM with a 10× reduction in value-function MSE, 1.5× reduction in EMD, and 3× reduction in log-likelihood, averaged across all four domains. In line with our theoretical predictions, the coupled variant of TD-CFM performs comparably to TD 2 -CFM, given straight conditional paths. While a comparison between flow matching and diffusion is not at the core of this paper, in our experiments, flow matching generally outperforms diffusion across all metrics. We posit this is primarily due to noise in the diffusion process adversely impacting an already noisy prediction problem for large horizons.
Given the comparable performance between TD-CFM(C) and TD 2 -CFM with straight conditional paths, we next examine how these methods behave with alternative path geometries. Our theoretical analysis suggests an important distinction: TD 2 -CFM should maintain its effectiveness with non-straight paths, while the performance of TD-CFM(C) should degrade. To test this prediction, we maintain the methodology above while replacing conditional path in (TD 2 -CFM; 9) with the following curved path p
t|1 (• | X 1 ) = N (• | α t X 1 , β 2 t )
with coefficients α t = sin π 2 t and β t = cos π 2 t . The corresponding conditional vector field is now given by u t|1 (X t |X 1 ) = αt -αt βt X 1 + βt βt X t . Additionally, for TD-CFM(C) we condition the curved path above on X 0 and X 1 resulting in the conditional vector field
u t|0,1 (X t | X 0 , X 1 ) = π 2 β t X 1 -α t X 0 .
Table 2 illustrates the performance difference relative to the straight path results (Table 1) averaged across all domains and tasks. The results strongly support our theoretical prediction: TD 2 -CFM not only maintained but surprisingly improved performance compared to the linear path. In contrast, TD-CFM(C) showed significant performance degradation, confirming our hypothesis about its limitations with non-straight paths.
this section cite: ['b21', 'b32', 'b26', 'b58']

Section: Planning via Generalized Policy Improvement
We now turn our attention towards training policyconditioned GHMs which can be utilized for test-time planning. To accomplish this, we first pre-train a Forward Backward (FB;Touati & Ollivier, 2021;Touati et al., 2023) representation using the same dataset of 10M transitions as described in §5.1. This pre-training yields a class of wconditioned policies π w , where each w ∈ W = S d-1 ( √ d) represents an embedding of a reward function situated on a d-dimensional hypersphere with radius √ d. We then train the GHM m πw conditioned on the policy by incorporating the embedding w directly into the model's input. All GHM methods are trained for 8M gradient steps, maintaining the same parameters used in §5.1, with the exception of a higher weight decay coefficient of 0.01. For additional insights into the accuracy of the policy-conditioned GHMs, we direct the reader to Appendix D. Overall, we observed similar trends to those seen in our single-policy experiments.
Given that both FB and w-conditioned GHM models enable estimation of a policy's value function Q πw , we can utilize this information to perform Generalized Policy Improvement (GPI; Barreto et al., 2017) during evaluation. Specifi-cally, at each time step t, we choose an action a t = π wt (s t ), where w t is derived as follows:
w t ∈ arg max w∼D(W) (1 -γ) -1 E X∼m πw (•|st,πw(st))) [r(X) ] Q πw (st,πw(st))
.
(15) Here D(W) is a sampling distribution over W. We consider three such distributions: i) Random: uniform distribution over W; ii) Local Perturbation: we perturb the embedding w r of the task reward r by the uniform distribution; iii) Train Distribution: we sample w from the training distribution used by FB. To approximate (15), we sample 255 embeddings from D(W) and explicitly include the task embedding w r , resulting in a maximization over 256 policies. To estimate Q πw , we average the reward over 128 states sampled from m πw . Performance is measured by averaging returns over 100 episodes, each lasting 1000 steps.
Figure 3 illustrates the average percentage of improvement for each algorithm and w-sampling strategy relative to the performance of the FB policy π wr for the task reward r. We refer to Appendix D for a more detailed view of these results. All TD-based GHM approaches lead to a significant improvement over the base FB policy, with TD-CFM(C) and TD 2 -CFM providing ≈ 30%+ improvement with all sampling approaches. TD 2 -DD also leads to significant performance gains but is still dominated by the flow matching methods. Notably, FB-based GPI not only fails to improve performance but actually deteriorates it on average with significant degradation observed in three out of four domains (detailed results available in Appendix D). When comparing different distributions D(W), we observe that while FB-GPI's performance fluctuates considerably, GHM methods maintain their robustness across distributions, showing only minor variation. These results underscore the ability of our improved GHMs to make long-term predictions enabling powerful planning capabilities.
this section cite: ['b4']

Section: Discussion
In this paper, we introduced temporal difference flows, a novel generative modeling approach that significantly advances long-horizon predictive models of state. By leveraging the successor measure's temporal difference structure both in its sampling procedure and learning objective, TD 2 -CFM and TD 2 -DD effectively address challenges associated with modeling long-range state dynamics. The methods developed in this paper provide a robust theoretical and empirical foundation that demonstrates the advantages of our framework across a range of tasks, metrics, and domains. We envision numerous exciting applications emerging from this work, particularly around imitation learning (Wu et al., 2025;Jain et al., 2025), planning (Sutton, 1991;Thakoor et al., 2022;Zhu et al., 2024), and off-policy evaluation (Precup et al., 2000;2001;Nachum et al., 2019;Fujimoto et al., 2021). Furthermore, recent work on consistency models (Song et al., 2023;Yang et al., 2024) and self-distillation (Frans et al., 2025) suggests promising avenues for tackling the computational burden of sampling -a limitation common to the family of iterative generative models that our approach builds upon.
this section cite: ['b54', 'b55', 'b48']

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: ()
Ref_id:b1 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b2 Title: Reverse-time diffusion equation models Year: (1982)
Ref_id:b3 Title: Layer normalization Year: (2016)
Ref_id:b4 Title: Successor features for transfer in reinforcement learning Year: (2017)
Ref_id:b5 Title: Fast reinforcement learning with generalized policy updates Year: (2020)
Ref_id:b6 Title: Learning successor states and goal-dependent values: A mathematical viewpoint Year: (2021)
Ref_id:b7 Title: Universal successor features approximators Year: (2019)
Ref_id:b8 Title: Exploration by random network distillation Year: (2019)
Ref_id:b9 Title: Finer behavioral foundation models via auto-regressive features and advantage weighting Year: (2024)
Ref_id:b10 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b11 Title: Improving generalization for temporal difference learning: The successor representation Year: (1993)
Ref_id:b12 Title: Schrödinger bridge flow for unpaired data translation Year: ()
Ref_id:b13 Title: NICE: Non-linear independent components estimation Year: (2015)
Ref_id:b14 Title: Density estimation using real nvp Year: (2017)
Ref_id:b15 Title: Protovalue networks: Scaling representation learning with auxiliary tasks Year: ()
Ref_id:b16 Title: Pot: Python optimal transport Year: (2021)
Ref_id:b17 Title: One step diffusion via shortcut models Year: ()
Ref_id:b18 Title: Addressing function approximation error in actor-critic methods Year: (2018)
Ref_id:b19 Title: A deep reinforcement learning approach to marginalized importance sampling with the successor representation Year: ()
Ref_id:b20 Title: Reinforcement learning from passive data via latent intentions Year: ()
Ref_id:b21 Title: Generative adversarial nets Year: (2014)
Ref_id:b22 Title: FFJORD: free-form continuous dynamics for scalable reversible generative models Year: (2019)
Ref_id:b23 Title: Mastering diverse domains through world models Year: (2023)
Ref_id:b24 Title: Temporal difference learning for model predictive control Year: ()
Ref_id:b25 Title: TD-MPC2: Scalable, robust world models for continuous control Year: ()
Ref_id:b26 Title: betavae: Learning basic visual concepts with a constrained variational framework Year: (2017)
Ref_id:b27 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b28 Title: The gan is dead; long live the gan! a modern gan baseline Year: ()
Ref_id:b29 Title: Hallucinating value: A pitfall of dyna-style planning with imperfect environment models Year: (2020)
Ref_id:b30 Title: Maximum state entropy exploration using predecessor and successor representations Year: ()
Ref_id:b31 Title: Non-adversarial inverse reinforcement learning via successor feature matching Year: ()
Ref_id:b32 Title: Gamma-models: Generative temporal difference learning for infinitehorizon prediction Year: (2020)
Ref_id:b33 Title: The relativistic discriminator: a key element missing from standard gan Year: (2019)
Ref_id:b34 Title: A method for stochastic optimization Year: (2015)
Ref_id:b35 Title: Auto-encoding variational bayes Year: (2014)
Ref_id:b36 Title: Investigating compounding prediction errors in learned dynamics models Year: (2022)
Ref_id:b37 Title: On the generalization of representations in reinforcement learning Year: ()
Ref_id:b38 Title: A novel stochastic gradient descent algorithm for learning principal subspaces Year: ()
Ref_id:b39 Title: Bootstrapped representations in reinforcement learning Year: (2023)
Ref_id:b40 Title: Flow matching for generative modeling Year: ()
Ref_id:b41 Title: Flow matching guide and code Year: (2024)
Ref_id:b42 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: ()
Ref_id:b43 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b44 Title: Eigenoption discovery through the deep successor representation Year: (2018)
Ref_id:b45 Title: Countbased exploration with the successor representation Year: (2020)
Ref_id:b46 Title: Temporal abstraction in reinforcement learning with the successor representation Year: (2023)
Ref_id:b47 Title: Mish: A self regularized non-monotonic neural activation function Year: (2019)
Ref_id:b48 Title: Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections Year: (2019)
Ref_id:b49 Title: Foundation policies with hilbert representations Year: ()
Ref_id:b50 Title: Curiosity-driven exploration by self-supervised prediction Year: (2017)
Ref_id:b51 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b52 Title: Fast imitation via behavior foundation models Year: ()
Ref_id:b53 Title: Multisample flow matching: Straightening flows with minibatch couplings Year: ()
Ref_id:b54 Title: Eligibility traces for off-policy policy evaluation Year: (2000)
Ref_id:b55 Title: Off-policy temporal difference learning with function approximation Year: (2001)
Ref_id:b56 Title: Variational inference with normalizing flows Year: (2015)
Ref_id:b57 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b58 Title: The earth mover's distance as a metric for image retrieval Year: (2000)
