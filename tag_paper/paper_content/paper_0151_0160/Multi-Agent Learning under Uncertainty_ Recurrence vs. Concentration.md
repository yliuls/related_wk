Title: Multi-Agent Learning under Uncertainty: Recurrence vs. Concentration
Abstract: In this paper, we examine the convergence landscape of multi-agent learning under uncertainty. Specifically, we analyze two stochastic models of regularized learning in continuous games-one in continuous and one in discrete time-with the aim of characterizing the long-run behavior of the induced sequence of play. In stark contrast to deterministic, full-information models of learning (or models with a vanishing learning rate), we show that the resulting dynamics do not converge in general. In lieu of this, we ask instead which actions are played more often in the long run, and by how much. We show that, in strongly monotone games, the dynamics of regularized learning may wander away from equilibrium infinitely often, but they always return to its vicinity in finite time (which we estimate), and their long-run distribution is sharply concentrated around a neighborhood thereof. We quantify the degree of this concentration, and we show that these favorable properties may all break down if the underlying game is not strongly monotone-underscoring in this way the limits of regularized learning in the presence of persistent randomness and uncertainty.

Section: Introduction
In its most abstract form, the standard model for online learning in games unfolds as follows: (i ) at each stage of the process, every participating agent selects an action; (ii ) the agents receive a reward determined by their chosen actions and their individual payoff functions; (iii) the agents update their actions, and the process repeats. In this general context, the agents have to contend with various-and varying-degrees of uncertainty: (a) uncertainty about the game, the strategic interests of other players, and/or who else is involved in the game; (b) uncertainty about the outcomes of their actions, and which update directions may lead to better outcomes; and (c) uncertainty stemming from the environment, manifesting as random shocks to the players' payoffs and / or other disturbances. In this regard, uncertainty could be either endogenous or exogenous; but, in either case, it leads to players having to take decisions with very limited information at their disposal.
Our goal in this paper is to quantify the impact of uncertainty on multi-agent learning-and, more precisely, to understand the differences that arise in the players' long-run behavior when such uncertainty is present versus when it is not. A natural framework for exploring this question is within the greater setting of no-regret learning and, in particular, the family of "follow-the-regularizedleader" (FTRL) algorithms and dynamics [38,64,65]. This class contains several mainstay learning methods-like online gradient descent (or, in our case, ascent) [74], the exponential / multiplicative weights (EW) algorithm and its variants (HEDGE, EXP3, etc.) [1,2,40,69], and many others-so it has become practically synonymous with the notion of online learning in games. Accordingly, we seek to answer the following questions:
What is the long-run distribution of regularized learning under uncertainty?
Which actions are played more often, and by how much? Do the dynamics concentrate-and, if so, where?
Our contributions in the context of related work. Needless to say, the interpretation of these questions is context-specific, and it depends on the particular learning setting at hand. In this paper, motivated by applications to machine learning, signal processing and data science (which typically involve continuous action spaces and rewards), we focus on continuous games, and we consider two models of regularized learning, one in continuous time, and one in discrete time.
In continuous time, we model the dynamics of FTRL in the presence of uncertainty as a stochastic differential equation (SDE) perturbed by a general Itô diffusion process, i.e., a continuous-time martingale with possibly colored and/or correlated components. In the context of finite games, models of this type have been studied by, among others, Bravo & Mertikopoulos [10], Foster & Young [20], Fudenberg & Harris [21] and Mertikopoulos & Moustakas [43,44], the first two in an evolutionary setting, the latter as a continuous-time model of the EW algorithm in the presence of random disturbances. Follow-up works in this direction include [10-12, 18, 25, 28, 49] on finite games, while [33][34][35]47] considered a regularized learning model in convex minimization problems. The model which is closest to our own is that of [46,48], who study the regret properties and guarantees of a stochastic version of the dual averaging dynamics of Nesterov [57].
At a high level, our findings reveal a crisp dichotomy between games that are null-monotone (like bilinear min-max games or zero-sum bimatrix games), and strongly monotone games (like Kelly auctions, Cournot competitions, joint signal covariance optimization problems, etc.). Specifically:
1. In null-monotone games, uncertainty induces a persistent drift away from equilibrium: the dynamics reach greater distances from equilibrium in finite time (which we estimate) and they require, on average, infinite time to return. In particular, if the game admits an interior equilibrium, the dynamics diffuse away-escaping in the mean toward infinity or to the boundary of the game's action space-and they exhibit no concentration in any region of interior actions.
2. In strongly monotone games, uncertainty still induces a persistent outward drift, but this is now partially countered by the dynamics' deterministic component. Thus, in stark contrast to the null-monotone case, the players' learning trajectories end up in a near-equilibrium region whose size scales with the level of uncertainty, and we estimate both the size of this region and the time required to reach it. Somewhat paradoxically, the dynamics return with probability 1 arbitrarily close to where they started, infinitely often, in a way reminiscent of Poincaré recurrence in bimatrix min-max games [52,59]; however, these returns can be exceedingly far apart, so there is no antinomy.
In discrete time, we consider a standard implementation of FTRL with a constant learning rate and stochastic first-order oracle feedback. Variants with a vanishing learning rate have been studied extensively in the stochastic approximation literature, and they are known to exhibit favorable convergence guarantees in, among others, strongly monotone games, cf. [50,54] and references therein. At the same time however, these properties typically come at the expense of the algorithm slowing down to a crawl; for this reason, owing to their simplicity, robustness, and superior empirical performance, constant / non-vanishing learning rate schedules are much more common in practice.
On the downside, the long-run behavior of FTRL is much less understood in this case. To the best of our knowledge, the most relevant results come from recent works by Loizou et al. [41] and Huang & Zhang [27], who established upper bounds on the mean distance to equilibrium for stochastic gradient descent / ascent in strongly monotone games, and Vlatakis et al. [68], who studied the ergodic properties of constant step-size variants of the stochastic extragradient and stochastic gradient descent-ascent algorithms for weakly quasi-strongly monotone variational inequalities. Dually to this, in the null-monotone regime, Cauvin et al. [12] showed that FTRL exhibits a similar tendency to escape from interior equilibria in finite min-max and harmonic games; our continuous-time analysis is, in this view, an extension of the corresponding result of [12].
One reason that results about the statistics of the long-run behavior of FTRL are particularly scarce in the literature is that, in discrete time, even the most basic tools of stochastic analysis are often inapplicable; for an illustration of the difficulties involved, see e.g., Azizian et al. [3,4] and references therein. Nevertheless, based in no small part on the insights gained by our continuous-time analysis, we manage to establish the following version of the strong-null dichotomy in discrete time:
1. In null-monotone games with an unbounded action space, the sequence of play under FTRL drifts away to infinity on average (though not necessarily with probability 1). 2. In strongly monotone games, we show that the mean time required to reach a given distance from the game's equilibrium is finite, and we provide an explicit estimate thereof. If the game's equilibrium is interior, we also show that FTRL converges strongly to a unique invariant measure, which is concentrated in a certain region around the game's equilibrium, which we also estimate.
We find these results particularly appealing as they provide the first glimpse into the distributional properties of multi-agent regularized learning under uncertainty.
this section cite: ['b38', 'b64', 'b65', 'b74', 'b0', 'b1', 'b40', 'b69', 'b9', 'b20', 'b21', 'b43', 'b44', 'b33', 'b34', 'b35', 'b47', 'b46', 'b48', 'b57', 'b52', 'b59', 'b50', 'b54', 'b41', 'b27', 'b68', 'b11', 'b11', 'b2', 'b3']

Section: Preliminaries

this section cite: []

Section: Continuous games.
Throughout the sequel, we consider games with a finite number of players and a continuum of actions per player. Formally, players will be indexed by 𝑖 ∈ N = {1, . . . , 𝑁 } and, during play, each player will be selecting an action 𝑥 𝑖 from a closed convex subset X 𝑖 of some 𝑑 𝑖 -dimensional normed space V 𝑖 . Aggregating over all players, we will write X = 𝑖 X 𝑖 for the space of the players' joint action profiles 𝑥 = (𝑥 1 , . . . , 𝑥 𝑁 ) and 𝑑 = 𝑖 𝑑 𝑖 for the dimension of the ambient space V = 𝑖 V 𝑖 . Finally, we will use the shorthand 𝑥 = (𝑥 𝑖 ; 𝑥 -𝑖 ) when we want to highlight the action of player 𝑖 ∈ N against the action profile 𝑥 -𝑖 = (𝑥 𝑗 ) 𝑗≠𝑖 of all other players-and, in similar notation, X -𝑖 = 𝑗≠𝑖 X 𝑗 for the space thereof.
The reward of each player 𝑖 ∈ N in a given action profile will be determined by an associated payoff function 𝑢 𝑖 : X → ℝ, assumed here to be individually concave in the sense that 𝑢 𝑖 (𝑥 𝑖 ; 𝑥 -𝑖 ) is concave in 𝑥 𝑖 for all 𝑥 -𝑖 ∈ X -𝑖 . We will further assume that each 𝑢 𝑖 is 𝛽-Lipschitz smooth, and we will write respectively 𝑣 𝑖 (𝑥) = ∇ 𝑥 𝑖 𝑢 𝑖 (𝑥 𝑖 ; 𝑥 -𝑖 ) and 𝑣(𝑥) = (𝑣 1 (𝑥), . . . , 𝑣 𝑁 (𝑥))
for the individual gradient field of each player and the ensemble thereof. 1The tuple G ≡ G (N , X , 𝑢) will be referred to as a concave game [62]. Mainstay examples of such games include (mixed extensions of) finite games, resource allocation problems, Kelly auctions, Cournot competitions, etc.; for completeness, we detail some of these applications in Appendix A.
this section cite: ['b62']

Section: Nash equilibrium.
The leading solution concept in game theory is that of a Nash equilibrium, defined here as an action profile 𝑥 * ∈ X which discourages unilateral deviations, i.e., 𝑢 𝑖 (𝑥 * ) ≥ 𝑢 𝑖 (𝑥 𝑖 ; 𝑥 * -𝑖 ) for all 𝑥 𝑖 ∈ X 𝑖 and all 𝑖 ∈ N .
A concave game always admits a Nash equilibrium if X is compact, and it admits a unique equilibrium if the game is strongly monotone in the sense of Definition 1 below:
Definition 1. A game G ≡ G (N , X , 𝑢) is called 𝛼-monotone if there exists some 𝛼 ≥ 0 such that ⟨𝑣(𝑥 ′ ) -𝑣(𝑥), 𝑥 ′ -𝑥⟩ ≤ -𝛼∥𝑥 ′ -𝑥∥ 2 for all 𝑥, 𝑥 ′ ∈ X .(Mon)
If (Mon) holds for some 𝛼 > 0, the game will be called strongly monotone; otherwise, if (Mon) only holds for 𝛼 = 0, G will be called merely monotone (or simply monotone when the distinction is not important). Finally, if (Mon) binds for 𝛼 = 0 and all 𝑥, 𝑥 ′ ∈ X -that is, ⟨𝑣(𝑥 ′ ) -𝑣(𝑥)), 𝑥 ′ -𝑥⟩ = 0 for all 𝑥, 𝑥 ′ ∈ X -the game will be called null-monotone. ❦ Remark 1. Merely monotone games could be viewed as a "hybrid" between null and strictly monotone games: generically, at any given action profile of a merely monotone game, there would be directions of motion where the (symmetrized) Jacobian of the players' gradient field has a zero eigenvalue, and directions with positive eigenvalues; either set (but not both) could be empty, the former corresponding to the "null-monotone" directions, the latter corresponding to the "strongly monotone" ones. ❦ Remark 2. A weighted variant of (Mon) is sometimes called diagonal (strict / strong) concavity, in reference to the work of Rosen [62]; for a pointed version of these conditions known as variational stability [26,50,54] or coherence [53,73]. These variants will not be important for our purposes. ❦
this section cite: ['b62', 'b26', 'b50', 'b54', 'b53', 'b73']

Section: Regularized learning.
In the rest of our paper, we will consider a family of online learning schemes adhering to the following model of "regularized learning": players aggregate gradient feedback on their payoff functions over time and, at each instance of play, they choose the action which is most closely aligned to this aggregate. We provide a detailed description of this model in Sections 3 and 4-in continuous and discrete time respectively-and only describe here the core idea.
At a high level, the common denominator of these schemes is the way that players choose their actions based on the accumulation of payoff gradients over time. Formally, we will treat payoff gradients as dual vectors and we will write Y 𝑖 := V * 𝑖 for the dual space of V 𝑖 and Y = 𝑖 Y 𝑖 = V * for the ensemble thereof. Then, given an aggregate of gradient steps 𝑦 𝑖 ∈ Y 𝑖 , we will assume that the 𝑖-th player chooses an action via a "generalized projection"-or mirror-map 𝑄 𝑖 :
Y 𝑖 → X 𝑖 of the general form 𝑄 𝑖 (𝑦 𝑖 ) = arg max 𝑥 𝑖 {⟨𝑦 𝑖 , 𝑥 𝑖 ⟩ -ℎ 𝑖 (𝑥 𝑖 )} for all 𝑦 𝑖 ∈ Y 𝑖 .
(2) In the above ℎ 𝑖 : X 𝑖 → ℝ is a continuous 𝐾 𝑖 -strongly convex function, that is,
ℎ 𝑖 (𝜆𝑥 𝑖 + (1 -𝜆)𝑥 ′ 𝑖 ) ≤ 𝜆ℎ 𝑖 (𝑥 𝑖 ) + (1 -𝜆)ℎ 𝑖 (𝑥 ′ 𝑖 ) -1 2 𝐾 𝑖 𝜆(1 -𝜆) ∥𝑥 ′ 𝑖 -𝑥 𝑖 ∥ 2(3)
for all 𝑥 𝑖 , 𝑥 ′ 𝑖 ∈ X 𝑖 and all 𝜆 ∈ [0, 1]. This function is known as the regularizer of the method and it acts as a penalty term that smooths out the "hard" arg max correspondence 𝑦 𝑖 ↦ → arg max 𝑖 ⟨𝑦 𝑖 , 𝑥 𝑖 ⟩. This regularization scheme has a very long and rich history in game theory and optimization, where 𝑄 is often referred to as a "quantal" or "regularized" best response operator, cf. [38,42,50,64,67] and references therein. For concreteness, we describe below the two leading examples of this regularization setup (suppressing in both cases the player index 𝑖 ∈ N for notational clarity
): Example 1 (Euclidean regularization). Let ℎ(𝑥) = 1 2 ∥𝑥∥ 2 2 . Then (B.12) boils down to the Euclidean projection map 𝑄(𝑦) = Π X (𝑦) ≡ arg max 𝑥 ∈X ∥𝑦 -𝑥∥ 2 . (4) Thus, in particular, if X = V, we readily recover the identity map 𝑄(𝑦) = 𝑦. ❦ Example 2 (Entropic regularization). Let X = {𝑥 ∈ ℝ 𝑑 + : 𝑑 𝑘=1 𝑥 𝑘 = 1} be the unit simplex of ℝ 𝑑 , and let ℎ(𝑥) = 𝑑 𝑘=1 𝑥 𝑘 log 𝑥 𝑘 denote the (negative) entropy on X . Then (B.12) yields the logit map 𝑄(𝑦) = Λ(𝑦) ≡ (exp(𝑦 1 ), . . . , exp(𝑦 𝑑 )) exp(𝑦 1 ) + • • • + exp(𝑦 𝑑 ) .
This map forms the basis of the seminal HEDGE and EXP3 algorithms in online learning, cf. [1,2,13,38,40,64] and references therein.
this section cite: ['b38', 'b42', 'b50', 'b64', 'b67', 'b0', 'b1', 'b12', 'b38', 'b40', 'b64']

Section: ❦
To ease notation in the sequel, we will write ℎ(𝑥) := 𝑖 ℎ 𝑖 (𝑥 𝑖 ) for the players' aggregate regularizer, 𝐾 := min 𝑖 𝐾 𝑖 for the strong convexity modulus of ℎ, and 𝑄 := 𝑖 𝑄 𝑖 : Y → X for the resulting ensemble mirror map. In the next sections, we describe in detail how this regularization setup is used in a learning context.
this section cite: []

Section: Learning under uncertainty in continuous time
To set the stage for the sequel, we begin with two simple games that will serve as "minimal working examples" for the more general model and results presented in the sections to come. We focus for the moment on continuous-time interactions; the discrete-time setting is presented in Section 4.
this section cite: []

Section: A gentle start.
Consider the following 2-player, convex-concave min-max games:
(𝑎) Bilinear saddle:
𝑢 1 (𝑥 1 , 𝑥 2 ) = -𝑢 2 (𝑥 1 , 𝑥 2 ) = -𝑥 1 𝑥 2 for 𝑥 1 , 𝑥 2 ∈ ℝ. (6a
) (𝑏) Quadratic saddle: 𝑢 1 (𝑥 1 , 𝑥 2 ) = -𝑢 2 (𝑥 1 , 𝑥 2 ) = 𝑥 2 2 /2 -𝑥 2 1 /2 for 𝑥 1 , 𝑥 2 ∈ ℝ.(6b)
Both games are monotone and they admit a unique Nash equilibrium at the origin. Their gradient fields are 𝑣(𝑥 1 , 𝑥 2 ) = (-𝑥 2 , 𝑥 1 ) and 𝑣(𝑥 1 , 𝑥 2 ) = -(𝑥 1 , 𝑥 2 ) respectively, so the first game is nullmonotone and the second one is 1-strongly monotone. Accordingly, if each player follows their individual payoff gradient to increase their rewards, we obtain the gradient descent / ascent dynamics (𝑎) 𝑥(𝑡) = (-𝑥 2 (𝑡), 𝑥 1 (𝑡)) and (𝑏)
𝑥(𝑡) = -(𝑥 1 (𝑡), 𝑥 2 (𝑡))(GDA)
for the bilinear and quadratic games (6a) and (6b) respectively. It is then trivial to see that, in the bilinear case, (GDA) cycles periodically at a constant distance from the game's equilibrium, whereas, in the quadratic case, the dynamics converge to the game's equilibrium at a geometric rate.
To model uncertainty in this setting, we will consider the stochastic gradient dynamics
𝑑𝑋 (𝑡) = 𝑣(𝑋 (𝑡)) 𝑑𝑡 + 𝜎 𝑑𝑊 (𝑡) (S-GDA)
where 𝑊 (𝑡) = (𝑊 1 (𝑡), 𝑊 2 (𝑡)) is a Brownian motion in ℝfoot_1 and 𝜎 > 0 is the magnitude of the noise entering the process. Intuitively, this SDE should be viewed as a rigorous formulation of the informal model 𝑥(𝑡) = 𝑣(𝑥 (𝑡)) + "noise", with the Brownian term 𝑊 (𝑡) capturing all sources of randomness and uncertainty in the players' environment. 2 Consequently, to understand the impact of uncertainty in each case of (GDA), we will examine the following quantities:
1. The distance ∥ 𝑋 (𝑡) ∥ 2 2 of 𝑋 (𝑡) from the game's equilibrium (that is, the origin of ℝ 2 ). 2. The time 𝜏 𝑟 = inf{𝑡 > 0 : ∥ 𝑋 (𝑡) ∥ 2 ≤ 𝑟 } at which 𝑋 (𝑡) gets within 𝑟 of the game's equilibrium. 3. The density P (𝑥, 𝑡) of 𝑋 (𝑡)-and, if it exists, its long-run limit P ∞ (𝑥) := lim 𝑡→∞ P (𝑥, 𝑡).
When it exists, P ∞ is known as the stationary-or invariant-distribution of 𝑋, and it is closely related to the occupation measure 𝜇 𝑡 of the process, defined here as
𝜇 𝑡 (B) = 1 𝑡 ∫ 𝑡 0 1{𝑋 (𝑠) ∈ B} 𝑑𝑠 for every Borel B ⊆ X .(7)
Under mild ergodicity conditions [30, Cor. 25.9], we have lim 𝑡→∞ 𝜇 𝑡 (B) = ∫ B P ∞ so, concretely, P ∞ measures the fraction of time that 𝑋 (𝑡) spends in a given subset of X in the long run.
Taken together, these metrics provide a fairly complete picture of the statistics of 𝑋 (𝑡) so, in the rest of this section, we analyze them in the context of (S-GDA) applied to the games (6a) and (6b).
This suggests that, on average, ∥ 𝑋 (𝑡) ∥ 2 2 increases as Θ(𝜎 2 𝑡). Building on this observation, we show in Appendix D that the dynamics (S-GDA) for the bilinear game (6a) enjoy the following properties: Proposition 1. Suppose that (S-GDA) is run on the game (6a) with initial condition 𝑥 0 ∈ ℝ 2 . Then: 2  2 = ∞, i.e., 𝑋 (𝑡) escapes to infinity in mean square. 2. 𝔼 𝑥 0 [𝜏 𝑟 ] = ∞ if 𝑟 < ∥𝑥 0 ∥, i.e., 𝑋 (𝑡) takes infinite time on average to get closer to equilibrium.
1. lim 𝑡→∞ 𝔼 𝑥 0 ∥ 𝑋 (𝑡) ∥
3. The limit P ∞ (𝑥) = lim 𝑡→∞ P (𝑥, 𝑡) does not exist, i.e., 𝑋 does not admit an invariant distribution.
Proposition 1 shows that, in the presence of uncertainty, the periodicity of the deterministic dynamics (GDA) is completely destroyed. In fact, despite random fluctuations that occasionally bring 𝑋 (𝑡) closer to equilibrium, (S-GDA) exhibits a consistent drift away from equilibrium, escaping any compact set in finite time and requiring infinite time to return on average. As a result, 𝑋 (𝑡) becomes infinitely spread out in the long run, exhibiting no measurable concentration in any region of ℝ 2 . For a partial illustration of this behavior-which we view as antithetical to convergence-cf. Fig. 1. ❦ Case 2: Quadratic saddles. We now proceed to examine the behavior of (S-GDA) in the quadratic min-max problem (6b), where (S-GDA) gives
𝑑𝑋 (𝑡) = -𝑋 (𝑡) 𝑑𝑡 + 𝜎 𝑑𝑊 (𝑡) . (9
)
As is well known [36, Chap. 7.4], this SDE describes the 2-dimensional Ornstein-Uhlenbeck (OU) process
𝑋 (𝑡) = 𝑋 (0)𝑒 -𝑡 + 𝜎 ∫ 𝑡 0 𝑒 -(𝑡 -𝑠) 𝑑𝑊 (𝑠) .
(OU) Hence, by unfolding the stochastic integral in (OU), we can draw the following conclusions: Proposition 2. Suppose that (S-GDA) is run on the game (6b) with initial condition 𝑥 0 ∈ ℝ 2 . Then:
1. lim 𝑡→∞ 𝔼 𝑥 0 ∥ 𝑋 (𝑡) ∥ 2 2 = 𝜎 2 , i.e., the dynamics fluctuate at mean distance 𝜎 from equilibrium. 2. The mean time required to get within distance 𝑟 of the game's equilibrium is bounded as
𝔼 𝑥 0 [𝜏 𝑟 ] ≤ 1 2 ∥𝑥 0 ∥ 2 2 -𝑟 2 𝑟 2 -𝜎 2 for all 𝜎 < 𝑟 < ∥𝑥 0 ∥ 2 .(10)
3. The density of 𝑋 (𝑡) is P (𝑥, 𝑡) = [𝜋𝜎 2 (1 -𝑒 -2𝑡 )] -1 exp -∥ 𝑥-𝑒 -𝑡 𝑥 0 ∥ 2 2 (1-𝑒 -2𝑡 ) 𝜎 2 . In particular, 𝑋 (𝑡) converges in distribution to a Gaussian random variable centered at 0, viz.
P ∞ (𝑥) ≡ lim 𝑡→∞ P (𝑥, 𝑡) = 1/(𝜋𝜎 2 ) • 𝑒 -∥ 𝑥 ∥ 2 2 /𝜎 2 . (11
)
Proposition 2 shows that the geometric convergence properties of the deterministic dynamics (GDA) are again destroyed in the presence of uncertainty. However, in stark contrast to Proposition 1 for the bilinear case, 𝑋 (𝑡) now exhibits a consistent drift toward equilibrium, and it ends up being sharply concentrated at a distance of O(𝜎 2 ) from equilibrium. This interplay between recurrence and concentration will play a crucial role in the sequel, and our aim in the rest of this section will be to quantify the extent to which it holds in a more general setting.
this section cite: []

Section: Learning in continuous time.
We now proceed to describe our general model for multi-agent learning under uncertainty, hinging on the stochastic "follow-the-regularized-leader" template
𝑑𝑌 𝑖 (𝑡) = 𝑣 𝑖 (𝑋 (𝑡)) 𝑑𝑡 + 𝑑𝑀 𝑖 (𝑡) 𝑋 𝑖 (𝑡) = 𝑄 𝑖 (𝑌 𝑖 (𝑡)) . (S-FTRL)
In the above, (i) 𝑌 𝑖 (𝑡) ∈ Y 𝑖 is a "score" variable that tracks the aggregation of individual payoff gradients in Y 𝑖 ; (ii) 𝑀 𝑖 (𝑡) ∈ Y 𝑖 is a continuous square-integrable martingale acting as a catch-all, "colored noise" disturbance term; and (iii) 𝑄 𝑖 : Y 𝑖 → X 𝑖 is the regularized mirror map of player 𝑖 ∈ N , as per (B.12). In this regard, (S-FTRL) represents a noisy "stimulus-response" mechanism, where each player 𝑖 ∈ N tracks the aggregation of payoff gradients under uncertainty-the "stimulus"-and "responds" to this aggregate via their individual regularized mirror map 𝑄 𝑖 .
Remark 3. The terminology "follow-the-regularized-leader" is due to [64,65], who first studied this scheme in the context of online convex optimization in discrete time. This family of algorithms and dynamics has been widely studied in the literature; we provide more details on this in Appendix B. ❦ For concreteness, we will assume that the noise term 𝑀(𝑡) = (𝑀 𝑖 (𝑡)) 𝑖∈N in (S-FTRL) is of the form
𝑑𝑀(𝑡) = 𝜎(𝑋 (𝑡)) • 𝑑𝑊 (𝑡) or, more explicitly 𝑑𝑀 𝑖 (𝑡) = 𝜎 𝑖 (𝑋 (𝑡)) • 𝑑𝑊 (𝑡)(12)
where 𝑊 (𝑡) = (𝑊 1 (𝑡), . . . , 𝑊 𝑚 (𝑡)) is a standard Brownian motion in ℝ 𝑚 , and 𝜎(𝑥) = (𝜎 𝑖 (𝑥)) 𝑖 ∈N is an ensemble of state-dependent diffusion matrices 𝜎 𝑖 : X 𝑖 → ℝ 𝑑 𝑖 ×𝑚 , 𝑖 ∈ N . 3 Importantly, the model (12) allows for correlated uncertainty between different components of the process-e.g., accounting for random disturbances on shared road segments in a congestion game-so it will be the base model for our analysis. Our only standing assumption will be that 𝜎 : X → ℝ 𝑑×𝑚 is bounded and Lipschitz continuous, which ensures that (S-FTRL) is well-posed, i.e., it admits a unique strong solution that exists for all time and for every initial condition 𝑌(0) ← 𝑦 ∈ Y (cf. Appendix C).
this section cite: ['b64', 'b65', 'b11']

Section: Analysis and results.
We now proceed to describe our main results for the stochastic dynamics (S-FTRL)-which, as we show shortly, reflect the dichotomy between bilinear and quadratic saddlepoint problems that we noted in Section 3.1. To state them, it will be convenient to introduce a "primaldual"generalization of the Euclidean distance that is more closely aligned with the regularization setup underlying the players' response scheme. Deferring the details to Appendix B, we define here the Fenchel coupling induced by the regularizer ℎ 𝑖 of player 𝑖 ∈ N as
𝐹 𝑖 ( 𝑝 𝑖 , 𝑦 𝑖 ) = ℎ 𝑖 ( 𝑝 𝑖 ) + ℎ * 𝑖 (𝑦 𝑖 ) -⟨𝑦 𝑖 , 𝑥 𝑖 ⟩ for all 𝑖 ∈ N , and all 𝑝 𝑖 ∈ X 𝑖 , 𝑦 𝑖 ∈ Y 𝑖 (13
)
where ℎ * 𝑖 (𝑦 𝑖 ) := max 𝑥 𝑖 ∈X 𝑖 {⟨𝑦 𝑖 , 𝑥 𝑖 ⟩ℎ 𝑖 (𝑥 𝑖 )} denotes the convex conjugate of ℎ 𝑖 . For example, in the unconstrained Euclidean case (Example 1), we recover the Euclidean distance squared, viz. 𝐹 𝑖 ( 𝑝 𝑖 , 𝑦 𝑖 ) = 1  2 ∥𝑄 𝑖 (𝑦 𝑖 ) -𝑝 𝑖 ∥ 2 ; by comparison, under entropic regularization on the simplex (Example 2), we get a "dualized" version of the Kullback-Leibler divergence, cf. Appendix B. In all cases, 𝐹 𝑖 is positive-semidefinite in the sense that 𝐹 𝑖 ( 𝑝 𝑖 , 𝑦 𝑖 ) ≥ 0 for all 𝑦 𝑖 ∈ Y 𝑖 , with equality if and ony if 𝑄 𝑖 (𝑦 𝑖 ) = 𝑝 𝑖 . In view of this, the total coupling 𝐹(𝑥, 𝑦) := 𝑖 𝐹 𝑖 ( 𝑝 𝑖 , 𝑦 𝑖 ) is a valid measure of "divergence" between 𝑝 ∈ X and 𝑦 ∈ Y, and we will use it freely in the sequel as such.
The last ingredient that we will need is two measures of the amount of randomness in (S-FTRL), viz. 𝜎 2 min := min 𝑥 ∈X 𝜆 min (Σ(𝑥)) and 𝜎 2 max := max 𝑥 ∈X 𝜆 max (Σ(𝑥))
where Σ ≡ 𝜎𝜎 ⊤ denotes the quadratic covariation matrix of the martingale 𝑀(𝑡), and 𝜆 min (resp. 𝜆 max ) denotes the minimum (resp. maximum) eigenvalue thereof.
With all this in hand, we will focus on two broad classes of games, null-monotone and strongly monotone, of which the bilinear and quadratic examples of Section 3.1 are archetypal examples. To state our results, we will assume that (S-FTRL) is initialized at 𝑥 0 ← 𝑄(𝑦 0 ) ∈ ri X for some 𝑦 0 ∈ Y, and we will write 𝐹 𝑡 ≡ 𝐹 (𝑥 * , 𝑌(𝑡)) where 𝑥 * is an equilibrium of the game. We then have: Theorem 1 (Null-monotone games). Suppose that (S-FTRL) is run with a smooth mirror map 𝑄 in a null-monotone game G. Suppose further that the game admits an interior equilibrium 𝑥 * , and consider the hitting times 𝜏 - 𝜀 := inf{𝑡 > 0 : 𝐹 𝑡 ≤ 𝐹 0 -𝜀} and 𝜏 + 𝜀 := inf{𝑡 > 0 : 𝐹 𝑡 ≥ 𝐹 0 + 𝜀}. If 𝜎 2 min > 0 and 𝜀 > 0 is small enough, then
𝔼 𝑥 0 [𝜏 - 𝜀 ] = ∞ and 𝔼 𝑥 0 [𝜏 + 𝜀 ] ≤ 2𝜀 𝜅 𝜎 2 min (15
)
for some constant 𝜅 ≡ 𝜅 𝜀 > 0; in addition, 𝑋 (𝑡) does not admit a limiting distribution in this case.
Theorem 2 (Strongly monotone games). Suppose that (S-FTRL) is run in an 𝛼-strongly monotone game G, and consider the hitting time
𝜏 𝑟 := inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )} (16
)
where 𝔹 𝑟 (𝑥 * ) = {𝑥 : ∥𝑥 -𝑥 * ∥ ≤ 𝑟 } is a ball of radius 𝑟 centered on the (necessarily unique) equilibrium 𝑥 * of G. Then:
𝔼 𝑥 0 [𝜏 𝑟 ] ≤ (𝐹 0 /𝛼) (𝑟 2 -𝑟 2 𝜎 ) for all 𝑟 > 𝑟 𝜎 ,(17)
where 𝑟 𝜎 := 𝜎 max / √ 2𝐾𝛼. If, in addition, 𝜎 min > 0 and 𝑥 * is interior, 𝑋 (𝑡) admits an invariant distribution concentrated in a ball of radius O(𝜎 max ) around 𝑥 * , and we have
lim 𝑡→∞ 𝜇 𝑡 (𝔹 𝑟 (𝑥 * )) ≥ 1 -𝑟 2 𝜎 /𝑟 2 for all 𝑟 > 𝑟 𝜎 .(18)
Remark 5. The bounds depend implicitly on the regularizer through its strong convexity modulus 𝐾 and they indicate a trade-off between the degree of concentration of the process around the radius beyond which the noise dominates the drift, and the time required to hit this region. ❦ Remark 6. The result of Theorem 2 holds for radius of concentration not sharper than O(𝜎). This coincides with the special case of Proposition 2, indicating that our bound is tight in this regard. ❦ Conceptually, Theorems 1 and 2 reflect the dichotomy between the bilinear and quadratic examples studied in detail in Section 3.1. Indeed, we see that:
1. In null-monotone games, the stochastic dynamics (S-FTRL) exhibit a consistent drift away from equilibrium, moving to greater distances in finite time, and requiring infinite time to return. As a result, if the game has an interior equilibrium, 𝑋 (𝑡) becomes infinitely spread out in the long run, exhibiting no concentration in any region of X other than, possibly, its boundary (if X is constrained). 2. In strongly monotone games, the dynamics drift toward equilibrium, and they end up being concentrated around the game's (necessarily unique) equilibrium. However, the players' learning trajectories continue to fluctuate at a distance which scales as O(𝜎 max ) and, with probability 1, they return arbitrarily close to where they started, infinitely often.
These properties paint a sharp separation between null-and strongly monotone games, with uncertainty carrying drastically different consequences in each case; for an illustration, see Fig. 1.
The proof of Theorems 1 and 2 is detailed in Appendix D. From a technical standpoint, our analysis hinges on the use of the Fenchel coupling (13) as a "mean" energy function for the dynamics. In the null-monotone case, the hitting time estimates (15) rely on an application of Dynkin's formula [58, Chap. 7.4], coupled with an eigenvalue estimation for the growth of 𝐹. Then, by descending to a specific quotient of Y that compactifies the sublevel sets of 𝐹, we are able to leverage the fact that 𝔼 𝑥 0 [𝜏 𝑟 ] = ∞ for 𝑟 < 𝐹 0 to show that the dynamics are not positively recurrent-and hence, they do not admit an invariant distribution. The analysis for the strongly monotone case has the same starting point, but it then branches out almost immediately: the hitting time estimate (17) is again obtained via Dynkin's stopping time formula, but positive recurrence can no longer be established in X , because the infinitesimal generator of 𝑋 (𝑡) is not uniformly elliptic (that is, its eigenvalues are not bounded away from zero). Instead, we work directly with the infinetisimal generator of the score process 𝑌(𝑡) whose generator is uniformly elliptic after taking a specific quotient in Y. This allows us to deduce positive recurrence in Y, which we then push forward to X via 𝑄, and leverage the convergence of the occupation measures to the invariant distribution of the process to derive the concentration bound (18). We detail these steps in a series of technical lemmas in Appendix D. Remark 7. If the game is neither null-nor strongly monotone, our analysis suggests that (S-FTRL) would tend to "wander around" the null-monotone directions, and be carried along the strongly monotone directions toward the game's set of equilibria. However, obtaining a precise version of such a result is quite involved, so we defer it to future work. ❦
this section cite: ['b18']

Section: Learning under uncertainty in discrete time
We now turn to the discrete-time setting, which is of more direct algorithmic relevance. Compared to Section 3, the analysis here is considerably more involved due to the lack of closed-form solutions and the limited applicability of diffusion-based methods. Nevertheless, as we shall see later in this section, the structural insights gained from the continuous-time analysis remain highly valuable as they form the foundation of the tools and techniques developed here.
this section cite: []

Section: Learning in discrete time.
In discrete time, the most widely used implementation of the FTRL template unfolds for 𝑡 = 0, 1, . . . as
𝑌 𝑖,𝑡+1 = 𝑌 𝑖,𝑡 + 𝛾 v𝑖,𝑡 𝑋 𝑖,𝑡+1 = 𝑄 𝑖 (𝑌 𝑖,𝑡+1 ) .
(FTRL) In addition to the notions already introduced and discussed in Section 3.2, (i) v𝑖,𝑡 denotes here a stochastic estimate of the player's payoff gradient vector at 𝑋 𝑖,𝑡 ; and (ii) 𝛾 > 0 is a step-size parameter, interchangeably referred to as the learning rate of the process. We discuss these two new elements below.
The feedback process. In terms of feedback, we assume that, at every round 𝑡 = 0, 1, . . . , each player 𝑖 ∈ N receives stochastic gradient feedback of the form 𝑥 is jointly continuous in 𝑥 and 𝑦, and it satisfies inf 𝑥 ∈K 𝑝 𝑥 (𝑦) > 0 for every compact set K ⊆ X and all 𝑦 ∈ Y. This last assumption is relatively mild and ensures that the noise retains a non-degenerate, smooth component across X , much like the assumption 𝜎 min > 0 for the diffusion matrix of (S-FTRL) in Section 3. 𝑡 < ∞. In many cases, the use of a vanishing step-size enables convergence of the algorithm because it dampens the impact of the noise over time [50]; at the same time however, in many applied settings, algorithms are implemented with a constant-or, at the very least, non-vanishing-step-size. This choice is largely driven by practical considerations: constant step-size schedules are easier to calibrate and maintain, particularly in large-scale systems where adaptivity and simplicity are critical. Moreover, vanishing step-size schedules often exhibit prolonged transient phases and converge slowly toward equilibrium neighborhoods; by contrast, constant step-size methods tend to reach near-stationary regions much faster, even within 0.1% accuracy or lower [16]. This behavior underlies their widespread use in modern machine learning pipelines, where learning rates are kept effectively constant throughout training, even for models trained over billions of samples and/or hundreds of billions of tokens [15].
v𝑖,𝑡 = V 𝑖 (𝑋 𝑡 ; 𝜔 𝑡 ) or, aggregating over all players v𝑡 = V(𝑋 𝑡 ; 𝜔 𝑡 ) (19
) where v𝑡 = (v 𝑖,𝑡 ) 𝑖 ∈N and V(𝑥; 𝜔) = (V 𝑖 (𝑥; 𝜔)) 𝑖 ∈N is a stochastic first-order oracle for 𝑣(𝑥), viz. V(𝑥; 𝜔) = 𝑣(𝑥) + U(𝑥; 𝜔) .(
this section cite: ['b50', 'b15', 'b14']

Section: Analysis and results.
We now have the necessary machinery in place to present our results for (FTRL). Before doing so, we should only stress that the discrete-time analysis is, by necessity, more qualitative than the more explicit, continuous-time results presented in Section 3. This gap is difficult to avoid: in continuous time, the rules of stochastic calculus comprise a very sharp set of tools with which to obtain closed-form estimates for the processes involved; on the other hand, in discrete time, even the most basic tools of stochastic analysis-like Dynkin's formula-are dulled down because of measurability and subsampling issues.
As before, we split our focus between null-and strongly monotone games.
this section cite: []

Section: The null-monotone regime.
A key take-away from the analysis of Section 3 is that, in nullmonotone games, uncertainty causes the dynamics of regularized learning to spread out, diverging to infinity on average, without concentrating at any region of X other than its boundary. Our first result below shows that a version of this tenet continues to hold in discrete time: Theorem 3 (Null-monotone games). Suppose that (FTRL) is run in a null-monotone game G, and let 𝑥 * be an equilibrium of G. Suppose further that ℎ * is strongly convex, and let 𝐹 𝑡 = 𝐹 (𝑥 * , 𝑌 𝑡 ), where 𝐹 is the induced Fenchel coupling (B.22). Then lim 𝑡→∞ 𝔼[𝐹 𝑡 ] = ∞.
This result shows that (FTRL) drifts away to infinity on average-though, of course, as in the continuous-time case, this does not mean that this occurs with probability 1. What is missing from Theorem 3 relative to Theorem 1 is a bound on the mean time required for 𝐹 𝑡 to increase or decrease by 𝜀. In the absence of a consistent drift component, our continuous-time estimates were only made possible through the use of stochastic calculus. In discrete time however, 𝑋 𝑡 evolves in discrete, driftless jumps, introducing overshoots and upcrossings that render this question significantly harder. used in both optimization and reinforcement learning to promote sufficient exploration and avoid degeneracy issues and saddle-points [6,22,39,70]. In such cases, the density of the absolutely continuous component is strictly positive everywhere and independent of 𝑥 ∈ X , so the uniform lower bound condition holds trivially.
We conjecture that similar bounds do hold in discrete time, but we leave this open as a conjecture.
[We only note here for completeness that a similar result holds for any decaying step-size sequence 𝛾 𝑡 with 𝑡 𝛾 2 𝑡 = ∞.]
The strongly monotone regime. We now turn to the long-run behavior of (FTRL) in strongly monotone games. Based in no small part on the continuous-time analysis of the previous section, our goal will be to understand the distributional properties of the dynamics, with a particular focus on (a) the existence and uniqueness of an invariant measure; and (b) the extent to which this measure is concentrated around the game's equilibrium-which, in turn, quantifies the long-run proximity of the iterates of (FTRL) to equilibrium. With all this in mind, our results can be stated as follows: Theorem 4 (Strongly monotone games). Suppose that (FTRL) is run in an 𝛼-strongly monotone game G, and consider the hitting time
𝜏 𝑟 := inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )} (22
)
where 𝔹 𝑟 (𝑥 * ) = {𝑥 : ∥𝑥 -𝑥 * ∥ ≤ 𝑟 } is a ball of radius 𝑟 centered on the (necessarily unique) equilibrium 𝑥 * of G. Then, for all 𝑟 > 𝑟 𝜎 := √︁ 𝛾(𝜎 2 + 𝛽 2 )/(𝛼𝐾), we have
𝔼[𝜏 𝑟 ] ≤1
𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) × 𝐹 0 if 𝑋 0 ∉ 𝔹 𝑟 (𝑥 * ), 𝐹 0 + 𝛼𝛾𝑟 2 if 𝑋 0 ∈ 𝔹 𝑟 (𝑥 * ),(23)
where 𝐹 0 = 𝐹 (𝑥 * , 𝑌 0 ). If, in addition, 𝑥 * is interior, 𝑋 𝑡 admits a unique invariant distribution to which it converges in total variation, and we have
lim 𝑡→∞ 1 𝑡 𝔼 𝑡 ∑︁ 𝑠=0 1{𝑋 𝑡 ∈ 𝔹 𝑟 (𝑥 * )} ≥ 1 -𝑟 2 𝜎 𝑟 2(24)
for all 𝑟 > 𝑟 𝜎 such that 𝔹 𝑟 (𝑥 * ) ⊆ ri X . Remark. Unlike the continuous-time setting of Section 3, we must treat the cases 𝑋 0 ∈ 𝔹 𝑟 (𝑥 * ) and 𝑋 0 ∉ 𝔹 𝑟 (𝑥 * ) separately. This distinction arises only in discrete time, because the iterates may exhibit large jumps-so, returning to 𝔹 𝑟 (𝑥 * ) is not guaranteed, even if the process is initialized within. ❦
We prove Theorem 4 in Appendix E following the strategy outlined below. First, shadowing the continuous-time analysis of Section 3, we reduce the dynamics to a suitable quotient space of Y, eliminating redundant directions and ensuring that the process evolves in a minimal, non-degenerate domain. Building on this, we then show that the induced dynamics are Lebesgue-irreducible, i.e., every measurable set with positive Lebesgue measure is reachable with positive probability under the transition kernel of the process. Moreover, invoking (23), we further deduce that ℙ(𝜏 𝑟 < ∞) = 1 for any initial condition, implying that 𝔹 𝑟 (𝑥 * ) is visited infinitely often. Finally, we also show that 𝔹 𝑟 (𝑥 * ) satisfies a minorization condition, meaning that the transition kernel from any point in the ball dominates a fixed reference measure. In turn, this implies that, upon returning to 𝔹 𝑟 (𝑥 * ), the process has a nonzero chance of "forgetting" its past, allowing us to construct a regeneration structure via a coupling argument. Then, leveraging the continuity of 𝐹 (𝑥 * , 𝑦), we obtain a uniform bound on the expected return times 𝔼[𝜏 𝑟 ] over any initialization in 𝔹 𝑟 (𝑥 * ), which allows us to conclude that the process 𝑌 𝑡 is positive Harris recurrent. As a result, it can be shown that the iterates of (FTRL) converge to a unique invariant measure, and we obtain quantitative control over their long-run concentration by means of our previous estimates.
this section cite: ['b5', 'b22', 'b39', 'b70']

Section: Concluding remarks
Our aim in this paper was to quantify the impact of noise and uncertainty on the dynamics of multi-agent regularized learning. Our findings reveal a sharp separation between games that are null-monotone (like bilinear min-max games), and strongly monotone games (like Kelly auctions or Cournot competitions). In the former case, the quasi-periodic profile of the deterministic dynamics is destroyed, and learning under uncertainty drifts away on average toward extreme points (or escapes to infinity); in the latter, the sharp convergence guarantees of the deterministic dynamics are diluted by noise, and the resulting dynamics end up concentrated in a region around the game's equilibrium (which we estimate). This paves the way for further explorations of the long-run statistics of regularized learning in games-especially pertaining to the invariant measure of the process-a topic which we find particularly promising for advancing our understanding of the field.
this section cite: []

Section: A Examples
In this appendix, we present several examples of games satisfying the standing assumptions we outlined in Section 2. Overall, these assumptions are quite standard in the study of online learning and games with continuous action spaces, and most of the positive results in the literature hinge on precisely these assumptions or close variants thereof.
To provide some context, the monotonicity assumption (cf. Definition 1) provides an amenable "convex structure", which is essential for establishing the existence and characterization of equilibria via first-order variational condtions. Without a structural characteriztion of this type, even defining a meaningful solution concept becomes unclear, and global convergence cannot be expected-at least in general. In a sense, these assumptions parallel the convex/non-convex separation in optimization-but with the added challenge that, in non-convex games, equilibria may fail to exist altogether, unlike minimizers (either local or global) in non-convex minimization problems.
Our regularity assumptions-closed action sets, Lipschitz smoothness, etc.-are largely technical, standard in practice and, as such, nearly universal in the literature. They could be relaxed, for instance, by assuming local or relative smoothness, Hölder continuity, or something of the sort-though the resulting analysis would be considerably more involved. Relaxing monotonicity, however, is considerably trickier: some of our results would go through as long as the game's equilibrium admits a global variational characterization, e.g., in the spirit of variational stability or a Minty-type condition, cf. [50,53,73] and references therein. If, however, the game admits distinct components of equilibria, all bets are off: in that case, FTRL could transit in perpetuity between the game's different equilibrium components, and characterizing the mean sojourn and transition times of the process would only be possible in very special cases.
All in all, the set of assumptions that we consider represents a certain "sweet spot" between theoretical tractability and practical relevance, which explains their prevalence in the literature. The examples below illustrate the range of settings where these assumptions arise naturally.
Example A.1 (Zero-sum bimatrix games). A bimatrix game consists of two players, each with a finite set of actions A 𝑖 , 𝑖 = 1, 2, and a min-max objective function 𝐿 : A 1 × A 2 → ℝ, typically encoded in a matrix 𝑀 ∈ ℝ A 1 ×A 2 with 𝑀 𝛼𝛽 = 𝐿 (𝛼, 𝛽) for all 𝛼 ∈ A 1 , 𝛽 ∈ A 2 . The first player is cast in the role of the minimizer and the second player in that of the maximizer, so their corresponding payoff functions are defined as 𝑢 1 = -𝐿 = -𝑢 2 .
In the mixed extension of the game, each player can mix their actions by selecting a probability distribution-a mixed strategy-over A 𝑖 , that is, an element 𝑥 𝑖 of the probability simplex
X 𝑖 ≡ Δ(A 𝑖 ) = {𝑥 𝑖 ∈ ℝ A 𝑖 + : ∥𝑥 𝑖 ∥ 1 = 1}.
Accordingly, in matrix notation, the players' corresponding mixed payoffs are given by
𝑢 1 (𝑥 1 , 𝑥 2 ) = -𝑥 ⊤ 1 𝑀𝑥 2 = -𝑢 2 (𝑥 1 , 𝑥 2 ) (A.1)
so their individual gradient fields can be expressed as
𝑣 1 (𝑥 1 , 𝑥 2 ) = -𝑀𝑥 2 and 𝑣 2 (𝑥 1 , 𝑥 2 ) = 𝑀 ⊤ 𝑥 1 (A.2)
for all 𝑥 1 ∈ X 1 and all 𝑥 2 ∈ X 2 .
By definition, a mixed-strategy Nash equilibrium of a bimatrix zero-sum game satisfies
𝐿 (𝑥 * 1 , 𝑥 2 ) ≤ 𝐿 (𝑥 * 1 , 𝑥 * 2 ) ≤ 𝐿 (𝑥 1 , 𝑥 * 2 ) for all 𝑥 1 ∈ X 1 , 𝑥 2 ∈ X 2 . (A.3)
If, in addition, 𝑥 * 1 , 𝑥 * 2 both have full support-that is, 𝑥 * 1 ∈ ri X 1 and 𝑥 * 2 ∈ ri X 2 -we also have the "equalizing payoffs" condition
𝐿 (𝑥 * 1 , 𝑥 2 ) = 𝐿 (𝑥 1 , 𝑥 * 2 ) for all 𝑥 1 ∈ X 1 , 𝑥 2 ∈ X 2 (A.4)
which means that (A.3) binds identically. In this case, we readily get
⟨𝑣 1 (𝑥 1 , 𝑥 2 ), 𝑥 1 -𝑥 * 1 ⟩ + ⟨𝑣 2 (𝑥 1 , 𝑥 2 ), 𝑥 2 -𝑥 * 2 ⟩ = 𝑢 1 (𝑥 1 , 𝑥 2 ) -𝑢 1 (𝑥 * 1 , 𝑥 2 ) + 𝑢 2 (𝑥 1 , 𝑥 2 ) -𝑢 2 (𝑥 1 , 𝑥 * 2 ) = 0 (A.5)
for all 𝑥 1 ∈ X 1 , 𝑥 2 ∈ X 2 , i.e., the game is null-monotone in the sense of Definition 1. ❦ Example A.2 (Cournot competition). In the standard Cournot competition model, there is a finite set of firms, indexed by 𝑖 ∈ N = {1, . . . , 𝑁 }, each providing the market with a quantity 𝑥 𝑖 ∈ [0, 𝐵 𝑖 ] of some good (or service) up to the firm's production budget 𝐵 𝑖 . Following the law of supply and demand, this good is priced following the simple linear model 𝑃(𝑥) = 𝑎 -𝑏 𝑖 𝑥 𝑖 , i.e., as a linearly decreasing function of the total supply. Accordingly, in this model, the utility of firm 𝑖 is given by
𝑢 𝑖 (𝑥) = 𝑥 𝑖 𝑃(𝑥) -𝑐 𝑖 𝑥 𝑖 = 𝑎 -𝑏 ∑︁ 𝑗 ∈N 𝑥 𝑗 -𝑐 𝑖 𝑥 𝑖 , (A.6)
where 𝑐 𝑖 represents the marginal production cost of firm 𝑖.
By a straightforward derivation, the players' individual payoff gradients are given by
𝑣 𝑖 (𝑥) = 𝜕𝑢 𝑖 𝜕𝑥 𝑖 = 𝑎 -𝑏 ∑︁ 𝑗 ∈N 𝑥 𝑗 -𝑐 𝑖 -𝑏𝑥 𝑖 (A.7)
and hence, the Hessian matrix of the game will be
𝐻 𝑖 𝑗 (𝑥) := 1 2 𝜕 2 𝑢 𝑖 𝜕𝑥 𝑗 𝜕𝑥 𝑖 + 1 2 𝜕 2 𝑢 𝑗 𝜕𝑥 𝑖 𝜕𝑥 𝑗 = -𝑏 -𝑏𝛿 𝑖 𝑗 (A.8)
where 𝛿 𝑖 𝑗 is the standard Kronecker delta. Since 𝐻 is circulant, standard linear algebra considerations show that its eigenvalues are -𝑏 and -(𝑁 + 1)𝑏 (with multiplicity 𝑁 -1 and 1 respectively), so it follows by a well-known second-order criterion that the Cournot competition game is 𝑏-strongly monotone [50,62]. ❦ Example A.3 (Signal covariance optimization). Consider a vector Gaussian channel of the form
y = ∑︁ 𝑖 ∈N H 𝑖 x 𝑖 + z (A.9)
where x 𝑖 ∈ ℂ 𝑚 𝑖 is the (complex-valued) signal transmitted by the 𝑖-th user of the channel, H ∈ ℂ 𝑛×𝑚 𝑖 is the transfer matrix of the channel, z ∈ ℂ 𝑛 is the noise in the channel (assumed zero-mean Gaussian and, without loss of generality, with unit covariance), and y ∈ ℂ 𝑛 is the aggregate signal output of the channel [72]. In this context, each user 𝑖 ∈ N controls the covariance matrix X 𝑖 = 𝔼[x 𝑖 x † 𝑖 ] subject to the power constraint tr(X 𝑖 ) = 𝔼[∥x 𝑖 ∥ 2 ] ≤ 𝑃 𝑖 , where 𝑃 𝑖 denotes the user's maximum transmit power. In this case, by the celebrated Shannon-Telatar formula [66], and assuming a single-user decoding scheme at the receiver, the achievable rate of the 𝑖-th user is
𝑢 𝑖 (X 𝑖 ; X -𝑖 ) = log det I + ∑︁ 𝑗 H 𝑗 X 𝑗 H † 𝑗 -log det I + ∑︁ 𝑗≠𝑖 H 𝑗 X 𝑗 H † 𝑗 . (A.10)
Putting everything together, this defines a continuous game with players 𝑖 ∈ N = {1, . . . , 𝑁 }, spectrahedral action sets of the form
Q 𝑖 = {X 𝑖 ∈ ℂ 𝑚 𝑖 ×𝑚 𝑖 : X 𝑖 ≽ 0 and tr X 𝑖 ≤ 𝑃 𝑖 } (A.11)
for all 𝑖 ∈ N , and payoff functions given by (A.10). By a calculation of Belmega et al. [5], it is known that this game is concave and monotone-and, in fact, strongly monotone if the linear mapping
(X 1 , . . . , X 𝑁 ) ↦ → 𝑖 H 𝑖 X 𝑖 H † 𝑖 is not rank-deficient. ❦
Examples that are closer to signal processing and data science include distributed metric learning, multimedia classification, etc. For a range of applications along these lines, we refer the reader to [51,63] and references therein.
this section cite: ['b50', 'b53', 'b73', 'b50', 'b62', 'b72', 'b66', 'b4', 'b51', 'b63']

Section: B Mirror maps and regularization
In this appendix, we collect some background material, properties and examples regarding the regularization machinery underlying (FTRL) and (S-FTRL). To lighten notation-especially with respect to the player index 𝑖 ∈ N -we base everything in this appendix on an abstract closed convex subset of some 𝑑-dimensional vector space, which could either be X 𝑖 or X , depending on the context.
The results presented below (or a version thereof) are known in the literature; nevertheless, we provide detailed proofs for completeness and to resolve any conflicts or ambiguities with different conventions in the literature.
this section cite: []

Section: B.1. Preliminaries.
Let V be a 𝑑-dimensional normed space, let Y := V * denote the (algebraic) dual of V, and let ⟨𝑦, 𝑥⟩ denote the canonical bilinear pairing between 𝑥 ∈ V and 𝑦 ∈ V * . If ∥•∥ is a norm on V will also write ∥𝑦∥ * = max{⟨𝑦, 𝑥⟩ : ∥𝑥∥ ≤ 1} (B.1) for the induced dual norm on Y, so |⟨𝑦, 𝑥⟩| ≤ ∥𝑥∥ ∥𝑦∥ * for all 𝑥 ∈ V and all 𝑦 ∈ Y by construction.
Given a closed convex subset C of V, we also define:
1. The tangent cone to C at 𝑝 ∈ C as TC( 𝑝) = cl{𝑧 ∈ V : 𝑝 + 𝑡𝑧 ∈ C for some 𝑡 > 0} (B.2)
i.e., as the closure of the set of rays emanating from 𝑝 and meeting C in at least one other point. 2. The dual cone to C at 𝑝 ∈ C as
TC * ( 𝑝) = {𝑤 ∈ Y : ⟨𝑤, 𝑧⟩ ≥ 0 for all 𝑧 ∈ TC( 𝑝)} (B.3)
3. The polar cone to C at 𝑝 ∈ C as PC( 𝑝) = {𝑤 ∈ Y : ⟨𝑤, 𝑧⟩ ≤ 0 for all 𝑧 ∈ TC( 𝑝)} (B.4) Following standard conventions in the field [60], convex functions will be allowed to take values in the extended real line ℝ ∪ {∞}, and we will denote the effective domain of a convex function 𝑓 : V → ℝ ∪ {∞} as dom 𝑓 :={𝑥 ∈ V : 𝑓 (𝑥) < ∞} . (B.5) When there is no danger of confusion, we will identify a convex function 𝑓 : V → ℝ with its restriction on dom 𝑓 ; in other words, we will treat 𝑓 interchangeably as a function on dom 𝑓 with values in ℝ, or as a function on V with values in ℝ ∪ {∞} (and finite on dom 𝑓 ).
Throughout the sequel, we will assume that all functions under study are proper, that is, dom 𝑓 ≠ ∅. Then, given a proper function 𝑓 : V → ℝ ∪ {∞}, the subdifferential of 𝑓 at 𝑥 ∈ dom 𝑓 is defined as
𝜕 𝑓 (𝑥) :={𝑦 ∈ Y : 𝑓 (𝑥 ′ ) ≥ 𝑓 (𝑥) + ⟨𝑦, 𝑥 ′ -𝑥⟩ for all 𝑥 ′ ∈ V } (B.6)
and we denote the domain of subdifferentiability of 𝑓 as
dom 𝜕 𝑓 = {𝑥 ∈ V : 𝜕 𝑓 (𝑥) ≠ ∅} . (B.7)
With all this in hand, a regularizer on a closed convex subset C of V is a continuous function ℎ : C → ℝ which is strongly convex, i.e., there exists some 𝐾 > 0 such that
ℎ(𝜆𝑥 + (1 -𝜆)𝑥 ′ ) ≤ 𝑡ℎ(𝑥) + (1 -𝜆)ℎ(𝑥 ′ ) - 𝐾 2 𝜆(1 -𝜆) ∥𝑥 ′ -𝑥∥ 2 (B.8)
for all 𝑥, 𝑥 ′ ∈ C and for all 𝜆 ∈ [0, 1]. By standard arguments [7,61], this immediately implies that
ℎ(𝑥 ′ ) ≥ ℎ(𝑥) + 𝜕ℎ(𝑥; 𝑥 ′ -𝑥) + 𝐾 2 ∥𝑥 ′ -𝑥∥ 2 for all 𝑥, 𝑥 ′ ∈ X , (B.9) where 𝜕ℎ(𝑥; 𝑥 ′ -𝑥) = lim 𝜃→0 + [ℎ(𝑥 + 𝜃 (𝑥 ′ -𝑥)) -ℎ(𝑥)]/𝜃 (B.10)
denotes the one-sided directional derivative of ℎ at 𝑥 along the direction of 𝑥 ′ -𝑥. In addition, we also define the following objects associated to ℎ:
1. The prox-domain of ℎ: Proof. For the most part, these properties are well known in the literature (except possibly the last one), so we only provide a pointer or a short sketch for most of them.
(a) This readily follows from the fact that ℎ is strongly convex, so the arg max in (B.12) is attained and is unique for all 𝑦 ∈ Y.
(b) By Fermat's rule [60, Chap. 26], we readily see that 𝑥 solves (B.12) if and only if 𝑦 -𝜕ℎ(𝑥) ∋ 0, that is, if and only if 𝑦 ∈ 𝜕ℎ(𝑥). Since this implies that 𝜕ℎ, our claim follows.
(c) By (B.14), we readily get im 𝑄 = C ℎ . As for the second part of our claim, it follows from basic properties of the subdifferential, cf. Rockafellar Hence, by rearranging and taking the limit 𝜃 → 0 + , we conclude that
𝜕ℎ(𝑥; 𝑥 ′ -𝑥) = lim 𝜃→0 + ℎ(𝑥 + 𝜃 (𝑥 ′ -𝑥)) -ℎ(𝑥) 𝜃 ≥ ⟨𝑦, 𝑥 ′ -𝑥⟩ (B.20)
as claimed. 5(g) By (B.14) it suffices to show that 𝑦 + 𝑤 ∈ 𝜕ℎ(𝑥) for all 𝑤 ∈ PC(𝑥). However, if 𝑤 ∈ PC(𝑥), we also have ⟨𝑤, 𝑥 ′ -𝑥⟩ ≤ 0 for all 𝑥 ′ ∈ X , and hence, with 𝑦 ∈ 𝜕ℎ(𝑥), we readily get
ℎ(𝑥 ′ ) ≥ ℎ(𝑥) + ⟨𝑦, 𝑥 ′ -𝑥⟩ ≥ ℎ(𝑥) + ⟨𝑦 + 𝑤, 𝑥 ′ -𝑥⟩ for all 𝑥 ′ ∈ X . (B.21)
This shows that 𝑦 + 𝑤 ∈ 𝜕ℎ(𝑥) and completes our proof.
this section cite: ['b60', 'b6', 'b61']

Section: ■
Following [45,50], we also define the Fenchel coupling associated to ℎ as
𝐹 ( 𝑝, 𝑦) = ℎ( 𝑝) + ℎ * (𝑦) -⟨𝑦, 𝑝⟩ for all 𝑝 ∈ X , 𝑦 ∈ Y. (B.22
)
The next proposition shows that the Fenchel coupling can be seen as a "primal-dual" measure of divergence between 𝑝 ∈ C and 𝑦 ∈ Y: Proposition B.2. Let ℎ be a 𝐾-strongly convex regularizer on C. Then, for all 𝑝 ∈ X and all 𝑦 ∈ Y, we have: Proof. These properties are known in the literature, but we provide a quick proof for completeness.
(a) By the Fenchel-Young inequality, we have ℎ( 𝑝) + ℎ * (𝑦) ≥ ⟨𝑦, 𝑝⟩ for all 𝑝 ∈ X , 𝑦 ∈ Y, with equality if and only if 𝑦 ∈ 𝜕ℎ( 𝑝). Our claim then follows from (B.14).
(b) Let 𝑥 = 𝑄(𝑦) so 𝑦 ∈ 𝜕ℎ(𝑥) by (B.14). Then, by the definition of 𝐹, we have
𝐹 ( 𝑝, 𝑦) = ℎ( 𝑝) + ℎ * (𝑦) -⟨𝑦, 𝑝⟩ = ℎ( 𝑝) + ⟨𝑦, 𝑥⟩ -ℎ(𝑥) -⟨𝑦, 𝑝⟩ # since 𝑦 ∈ 𝜕ℎ(𝑥) ≥ ℎ( 𝑝) -ℎ(𝑥) -𝜕ℎ(𝑥; 𝑝 -𝑥) # by Proposition B.1 ≥ 1 2 𝐾 ∥𝑥 -𝑝∥ 2 # by (B.8)
so our proof is complete.
this section cite: ['b45', 'b50']

Section: ■
Our last result at this point is a useful differentiation formula for the Fenchel coupling: Lemma B.1. For all 𝑝 ∈ X and all 𝑦 ∈ Y, we have:
∇ 𝑦 𝐹 ( 𝑝, 𝑦) = 𝑄(𝑦) -𝑝 .(
B.24) Proof. The proof follows immediately from Danskin's theorem, cf. Eq. (B.16) of Proposition B.1. ■ B.2. Update lemmas. Moving forward, we note that the basic update step of (FTRL) can be written as 𝑦 + = 𝑦 + 𝑤 and 𝑥 + = 𝑄(𝑦 + ) (B.25)
for some 𝑦, 𝑤 ∈ Y. With this in mind, we state below a series of identities and estimates for the Fenchel coupling before and after an update of the form (B.25).
The first is a primal-dual version of the so-called "three-point identity" for Bregman functions [14]: Lemma B.2. Fix some 𝑝 ∈ X , 𝑦 ∈ Y, and let 𝑥 = 𝑄(𝑦). Then, for all 𝑦 + ∈ Y, we have: Our assertion then follows by recalling that 𝑥 = 𝑄(𝑦), so ℎ(𝑥) + ℎ * (𝑦) = ⟨𝑦, 𝑥⟩.
this section cite: ['b13']

Section: ■
The next result we present concerns the Fenchel coupling before and after a direct update step; similar results exist in the literature, but we again provide a proof for completeness. Lemma B.3. Fix some 𝑝 ∈ X and 𝑦, 𝑤 ∈ Y. Then, letting 𝑥 = 𝑄(𝑦), 𝑦 + = 𝑦 + 𝑤, and 𝑥 + = 𝑄(𝑦 + ) as per (B.25), we have:
𝐹 ( 𝑝, 𝑦 + ) =
𝐹 ( 𝑝, 𝑦) + ⟨𝑤, 𝑥 + -𝑝⟩ -𝐹 (𝑥 + , 𝑦) (B.29a) ≤ 𝐹 ( 𝑝, 𝑥) + ⟨𝑤, 𝑥 -𝑝⟩ + 1 2𝐾 ∥𝑤∥foot_7 * . (B.29b) Proof. By the three-point identity (B.26), we have 𝐹 (𝑥, 𝑦) = 𝐹 (𝑥, 𝑦 + ) + 𝐹 (𝑥 + , 𝑥) + ⟨𝑦 -𝑦 + , 𝑥 + -𝑝⟩ (B.30) so our first claim is immediate. For our second claim, rearranging terms and employing the Fenchel-Young inequality gives 𝐹 ( 𝑝, 𝑦) + ⟨𝑤, 𝑥 + -𝑝⟩ -𝐹 (𝑥 + , 𝑦) = 𝐹 ( 𝑝, 𝑦) + ⟨𝑤, 𝑥 -𝑝⟩ + ⟨𝑤, 𝑥 + -𝑥⟩ -𝐹 ( 𝑝, 𝑦) ≤ 𝐹 ( 𝑝, 𝑦) + ⟨𝑤, 𝑥 -𝑝⟩ + 1 2𝐾 ∥𝑤∥ 2 * + 𝐾 2 ∥𝑥 -𝑝∥ 2 -𝐹 ( 𝑝, 𝑦) (B.31) so our claim follows from Proposition B.2. ■
this section cite: []

Section: C A short primer on stochastic analysis
In this appendix, we collect some standard results from stochastic analysis in order to provide a degree of self-completeness to the main text. For an introduction to stochastic analysis and the theory of SDEs, we refer the reader to the masterful accounts of Øksendal [58] and Kuo [36].
The main focus of the theory is the study of ordinary differential equations (ODEs) perturbed by noise, modeled informally after the Langevin equation
𝑑𝑍 𝑑𝑡 = 𝑏(𝑍 (𝑡)) + 𝜂(𝑡)(LE)
where 𝑍 (𝑡) is a stochastic process in ℝ, 𝑏 : ℝ → ℝ is the drift of the process, and 𝜂(𝑡) is the "noise" perturbing the deterministic ODE 𝑧 = 𝑏(𝑧). Unfortunately, albeit natural, the problem with (LE) is that any reasonable continuous-time model of noise would lead to trajectories that are almost nowhere differentiable, so the meaning of "𝑑𝑍/𝑑𝑡" in (LE) is rather precarious. 6   In lieu of this, to give formal meaning to (LE), we consider instead the stochastic differential equation
∫ 𝑡 0 𝜎(𝑍 (𝑠)) 𝑑𝑊 (𝑠) ≈ ⌈𝑡/ 𝛿𝑡 ⌉ ∑︁ 𝑘=1 𝜎(𝑍 (𝑡 𝑘 )) [𝑊 (𝑡 𝑘+1 ) -𝑊 (𝑡 𝑘 )] (C.2)
where 𝑊 (𝑡) is some stochastic process that satisfies what one would expect from a "white noise" process (zero-mean, with independent increments), but is still "regular enough" to possess a reasonable behavior in the limit 𝛿𝑡 → 0. These considerations lead to the formal definition of a Brownian motion-or, more precisely, the Wiener process-which is characterized by the following properties:
The existence of a process with the above properties is by no means a trivial affair, but it can constructed e.g., as the scaling limit of a random walk, or some other discrete-time stochastic processes with stationary independent increments.
Providing a more detailed account of the definition of 𝑊 (𝑡) and the associated stochastic integral which appears in (SDE) is well beyond the scope of our paper; for an accessible introduction, we refer the reader to Øksendal [58, Chap. 2]. What is more important for our purposes is that, albeit non-differentiable, the solution 𝑍 (𝑡) still satisfies a certain version of the chain rule, known as Itô's formula [29]. Specifically, for any 𝐶 2 function 𝑓 : ℝ → ℝ, we have 𝑑𝑓 (𝑍 (𝑡)) = 𝑓 ′ (𝑍 (𝑡))𝑏(𝑍 (𝑡)) 𝑑𝑡 + 1 2 𝑓 ′′ (𝑍 (𝑡))𝜎 2 (𝑡) 𝑑𝑡 + 𝑓 ′ (𝑍 (𝑡)) 𝜎(𝑍 (𝑡)) 𝑑𝑊 (𝑡) (C.3) or, more compactly:
𝑑𝑓 (𝑍 (𝑡)) = 𝑓 ′ (𝑍 (𝑡)) 𝑑𝑍 (𝑡) + 1 2 𝑓 ′′ (𝑍 (𝑡)) 𝑑𝑍 (𝑡) • 𝑑𝑍 (𝑡)
(C.4) where the product 𝑑𝑍 • 𝑑𝑍 is computed according to the rules of stochastic calculus [58]:
𝑑𝑡 • 𝑑𝑡 = 0 𝑑𝑡 • 𝑑𝑊 (𝑡) = 0 and 𝑑𝑊 (𝑡) • 𝑑𝑊 (𝑡) = 𝑑𝑡 .
(C.5) Thanks to Itô's formula, we can still do calculus with stochastic processes satisfying (SDE); the resulting set of differentiation rules is known as Itô-or stochastic-calculus.
For our purposes, we will consider multi-dimensional analogues of (SDE) where, mutatis mutandis, In fact, this simple expression admits a far-reaching generalization known as Dynkin's formula [58, Chap. 7.4]: Proposition C.1 (Dynkin's formula). Suppose that 𝑍 (𝑡) is initialized at 𝑍 (0) ← 𝑧 ∈ ℝ 𝑛 . Then, for every bounded stopping time 𝜏 and every 𝐶 2 -smooth function 𝑓 : ℝ 𝑛 → ℝ, we have 𝔼 𝑧 [ 𝑓 (𝑍 (𝜏))] = 𝑓 (𝑧) + 𝔼 𝑧 ∫ 𝜏 0 L 𝑓 (𝑍 (𝑠)) 𝑑𝑠 . (C.12) Moving forward, the matrix 𝐴(𝑧) = 𝜎(𝑧)𝜎(𝑧) ⊤ (C.13) or, in components, 𝐴 𝑖 𝑗 (𝑧) = 𝑚 ∑︁ 𝑘=1 𝜎 𝑖𝑘 (𝑧)𝜎 𝑗 𝑘 (𝑧) 𝑖, 𝑗 = 1, . . . , 𝑛, (C.14) is known as the principal symbol of L, and we say that L is uniformly elliptic if there exists some 𝑐 > 0 such that 𝑢 ⊤ 𝐴(𝑧)𝑢 ≥ 𝑐∥𝑢∥ 2 for all 𝑧, 𝑢 ∈ ℝ 𝑛 (that is, if the eigenvalues of 𝐴(𝑧) are positive and uniformly bounded away from 0). If this is the case, the noise in (SDE) is "uniformly exciting" in the sense that it does not vanish along any direction at any point of the state space of the process. Concretely, by standard results-see e.g., [56, Sec. 3.3.6.1] and references therein-this implies that every region of ℝ 𝑛 is visited by 𝑍 (𝑡) with positive probability, viz. ℙ 𝑧 (𝑍 (𝑡) = 𝑧 ′ for some 𝑡 > 0) > 0 for all 𝑧, 𝑧 ′ ∈ ℝ 𝑛 . (C.15) If (SDE) is uniformly elliptic-i.e., if the infinitesimal generator thereof is uniformly elliptic-the behavior of 𝑍 (𝑡) can be further classified as transient or recurrent. Formally, these two fundamental notions are defined as follows: Definition C.1. Suppose that (SDE) is initialized at some 𝑧 ∈ ℝ 𝑛 . Then: 1. 𝑍 (𝑡) is transient from 𝑧 ∈ ℝ 𝑛 if it escapes every compact subset K of ℝ 𝑛 in finite time, i.e., there exists some (possibly random) 𝑇 K < ∞ such that ℙ 𝑧 (𝑍 (𝑡) ∉ K for all 𝑡 ≥ 𝑇 K ) = 1 . (C.16) 2. 𝑍 (𝑡) is recurrent relative to a compact subset K of ℝ 𝑛 if the hitting time
𝜏 K = inf{𝑡 > 0 : 𝑍 (𝑡) ∈ K} (C.17)
is finite (a.s.). If, in addition, 𝔼[𝜏 K ] < ∞, we will say that 𝑍 (𝑡) is positive recurrent; otherwise, 𝑍 (𝑡) will be called null recurrent.
If (SDE) is uniformly elliptic, we have the following fundamental dichotomy: Theorem C.1 (Transience / recurrence dichotomy). Suppose that (SDE) is uniformly elliptic. Then:
1. If (SDE) is positive recurrent (resp. null recurrent) for some initial condition 𝑧 ∈ ℝ 𝑛 and some compact subset K of ℝ 𝑛 , then it is positive recurrent (resp. null recurrent) for every initial condition and every compact subset of ℝ 𝑛 .
this section cite: ['b58', 'b36', 'b58']

Section: If (SDE)
is transient from some initial condition 𝑧 ∈ ℝ 𝑛 , it is transient from every initial condition.
For a more detailed version of Theorem C.1, we refer the reader to Bhattacharya [8, Proposition 3.1] who, to the best of our knowledge, was the first to state and prove this criterion. In words, Theorem C.1 simply states that, as long as (SDE) is uniformly elliptic, then it is either transient or recurrent; and if it is recurrent, it is either positive or null recurrent; no other outcome is possible. The choice of initialization or compact set in Definition C.1 does not matter (so, in particular, 𝑍 cannot be transient from some region of ℝ 𝑛 and recurrent from another). This crisp separation of regimes will play a major role in our analysis, and we will refer to it as the transience / recurrence dichotomy.
An important consequence of positive recurrence is that, under uniform ellipticity, 𝑍 (𝑡) admits a unique invariant measure, that is, a probability measure 𝜈 on ℝ 𝑛 such that 𝑍 (𝑡) ∼ 𝜈 for all 𝑡 ≥ 0 whenever 𝑍 (0) ∼ 𝜈. Importantly, the proviso that 𝜈 is a probability measure implies that 𝜈(ℝ 𝑛 ) < ∞; if the process is null-recurrent, the semigroup of flows of (SDE) still admits an invariant meassure in the sense of Khasminskii [32], but this measure is no longer finite, i.e., 𝜈(ℝ 𝑛 ) = ∞. Finally, if the process is transient, (SDE) does not admit such a measure.
this section cite: ['b32']

Section: D Analysis and results in continuous time
We now proceed to prove the continuous-time results for (S-FTRL) that we presented in Section 3.
this section cite: []

Section: D.1. Proofs omitted from Section 3.1.
We begin with the "gentle start" results of Section 3.1, which we restate below for convenience. Proposition 1. Suppose that (S-GDA) is run on the game (6a) with initial condition 𝑥 0 ∈ ℝ 2 . Then:
1. lim 𝑡→∞ 𝔼 𝑥 0 ∥ 𝑋 (𝑡) ∥ 2 2 = ∞, i.e., 𝑋 (𝑡) escapes to infinity in mean square. 2. 𝔼 𝑥 0 [𝜏 𝑟 ] = ∞ if 𝑟 < ∥𝑥 0 ∥, i.e., 𝑋 (𝑡) takes infinite time on average to get closer to equilibrium.
3. The limit P ∞ (𝑥) = lim 𝑡→∞ P (𝑥, 𝑡) does not exist, i.e., 𝑋 does not admit an invariant distribution. Proposition 2. Suppose that (S-GDA) is run on the game (6b) with initial condition 𝑥 0 ∈ ℝ 2 . Then:
1. lim 𝑡→∞ 𝔼 𝑥 0 ∥ 𝑋 (𝑡) ∥ 2 2 = 𝜎 2 , i.e., the dynamics fluctuate at mean distance 𝜎 from equilibrium. 2. The mean time required to get within distance 𝑟 of the game's equilibrium is bounded as
𝔼 𝑥 0 [𝜏 𝑟 ] ≤ 1 2 ∥𝑥 0 ∥ 2 2 -𝑟 2 𝑟 2 -𝜎 2 for all 𝜎 < 𝑟 < ∥𝑥 0 ∥ 2 . (10
)
3. The density of 𝑋 (𝑡) is P (𝑥, 𝑡) = [𝜋𝜎 2 (1 -𝑒 -2𝑡 )] -1 exp - ∥ 𝑥-𝑒 -𝑡 𝑥 0 ∥ 2 2 (1-𝑒 -2𝑡 ) 𝜎 2 .
In particular, 𝑋 (𝑡) converges in distribution to a Gaussian random variable centered at 0, viz.
P ∞ (𝑥) ≡ lim 𝑡→∞ P (𝑥, 𝑡) = 1/(𝜋𝜎 2 ) • 𝑒 -∥ 𝑥 ∥ 2 2 /𝜎 2 . (11
)
Proof of Proposition 1. For our first claim, note that Itô's formula (C.6) applied to the function 𝑓 (𝑥) = ∥𝑥∥ 2 2 under the dynamics (S-GDA) for the game (6a) readily yields the expression
𝑑 ∥ 𝑋 (𝑡) ∥ 2 2 = 2𝑋 (𝑡) • 𝑑𝑋 (𝑡) + 𝑑𝑋 (𝑡) • 𝑑𝑋 (𝑡) = 2𝜎 2 𝑑𝑡 + 𝜎 𝑋 (𝑡) • 𝑑𝑊 (𝑡) . (D.1)
Hence, by (C.11), we get 𝔼 𝑥 0 [∥ 𝑋 (𝑡) ∥ 2 2 ] = 2𝜎 2 𝑡 , (D.2) which proves our claim.
For our second claim, consider the hitting time 𝜏 = inf{𝑡 > 0 : ∥ 𝑋 (𝑡) ∥ 2 ≤ 𝑟 } with 𝑟 < ∥𝑥 0 ∥ 2 , and assume that 𝔼[𝜏] < ∞. Then, by Dynkin's formula (Proposition C.1) applied to 𝑓 (𝑥) = ∥𝑥∥ 2  2 and 𝜏, we readily get
𝔼 𝑥 0 [ 𝑓 (𝑋 (𝜏))] = 𝑓 (𝑥 0 ) + 𝔼 𝑥 0 ∫ 𝜏 0 2𝜎 2 𝑑𝑠 = 𝑓 (𝑥 0 ) + 2𝜎 2 𝔼 𝑥 0 [𝜏] ≥ ∥𝑥 0 ∥ 2 2 . (D.3)
However, since 𝑓 (𝑋 (𝜏)) = 𝑟 2 by construction, we readily get 𝑟 2 ≥ ∥𝑥 0 ∥ 2 2 , a contradiction. This shows that 𝔼 𝑥 0 [𝜏] = ∞, as asserted.
Finally, for our third claim, it is easy to check that (S-GDA) is uniformly elliptic under the stated assumptions. Thus, by Theorem C.1 and the fact that 𝔼 𝑥 0 [𝜏] = ∞, it follows that 𝑋 (𝑡) cannot be positive recurrent. By the discussion following Theorem C.1, this implies that 𝑋 (𝑡) does not admit an inveriant measure, so the density P (𝑥, 𝑡) of 𝑋 (𝑡) does not converge to a limit either. In turn, by [36,Theorem 7.4.7], this implies that the transition probability kernel of 𝑋 𝑖 (𝑡) is given by
P 𝑖 (𝑥 𝑖 , 𝑡) =1
𝜎 √︁ 𝜋(1 -𝑒 -2𝑡 ) exp - (𝑥 𝑖 -𝑒 -𝑡 𝑥 𝑖,0 ) 2 (1 -𝑒 -2𝑡 )𝜎 2 for 𝑖 = 1, 2, (D.6)
that is, 𝑋 𝑖 (𝑡) follows a Gaussian distribution with mean 𝔼 𝑥 𝑖,0 [𝑋 𝑖 (𝑡)] = 𝑥 𝑖,0 𝑒 -𝑡 and variance
𝔼[𝑋 2 𝑖 (𝑡)] = 𝜎 2 2 1 -𝑒 -2𝑡 . (D.7)
Our first and third claims then follow immediately.
For our second claim, note that the infinitesimal generator of 𝑋 (𝑡) is now given by
L 𝑓 (𝑥) = -⟨∇ 𝑓 (𝑥), 𝑥⟩ + 1 2 𝜎 2 Δ 𝑓 (𝑥) , (D.8)
where Δ 𝑓 ≡ tr ∇ 2 𝑓 denotes the Laplacian of 𝑓 . Then, Dynkin's formula (Proposition C.1) applied to 𝑓 (𝑥) = ∥𝑥∥ 2 2 at the truncated hitting time 𝜏 𝑟 ∧ 𝑡 ≡ min{𝜏 𝑟 , 𝑡}, 𝑡 > 0, readily yields
𝔼 𝑥 0 [∥ 𝑋 (𝜏 𝑟 ∧ 𝑡) ∥ 2 2 ] = ∥𝑥 0 ∥ 2 2 + 𝔼 𝑥 0 ∫ 𝜏 𝑟 ∧𝑡 0 2[𝜎 2 -∥ 𝑋 (𝑠) ∥ 2 2 ] 𝑑𝑠 ≤ ∥𝑥 0 ∥ 2 2 + 𝔼 𝑥 0 ∫ 𝜏 𝑟 ∧𝑡 0 2(𝜎 2 -𝑟 2 ) 𝑑𝑠 = ∥𝑥 0 ∥ 2 2 + 2(𝜎 2 -𝑟 2 ) 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] . (D.9) Since ∥ 𝑋 (𝜏 𝑟 ∧ 𝑟) ∥ 2 2 ≥ 𝑟 2 by construction (recall that ∥𝑥 0 ∥ 2 > 𝑟), we get 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] ≤ ∥𝑥 0 ∥ 2 2 -𝑟 2 2(𝑟 2 -𝜎 2 )
for all 𝑡 > 0. (D.10)
Since 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] is uniformly bounded, our claim follows by taking the limit 𝑡 → ∞ (so 𝜏 𝑟 ∧ 𝑡 → 𝜏 𝑟 pointwise), and invoking the dominated convergence theorem. ■ D.2. General properties of the dynamics (S-FTRL). We now proceed to establish the properties of the stochastic dynamics (S-FTRL) in the general case, for null-and strongly monotone games respectively. Before doing so, we begin with a result of a book-keeping nature (which is, however, necessary to ensure that the ensuing questions are meaningful). Proposition D.1. Suppose that 𝜎 is Lipschitz continuous. Then, for every initial condition 𝑋 (0) ← 𝑥 0 = 𝑄(𝑦 0 ) ∈ X , the dynamics (S-FTRL) admit a unique strong solution that exists for all time.
Proof. Note that the dynamics (S-FTRL) can be recast in fully autonomous form as 1 2 tr[𝜎 ⊤ (𝑋 (𝑡)) ∇ 2 𝐸 (𝑌(𝑡)) 𝜎(𝑋 (𝑡))] 𝑑𝑡 = ⟨𝑣(𝑋 (𝑡)), 𝑋 (𝑡) -𝑝⟩ 𝑑𝑡 + 1 2 tr[Σ(𝑋 (𝑡)) Jac 𝑄(𝑌(𝑡))] 𝑑𝑡 + (𝑋 (𝑡) -𝑝) ⊤ 𝜎(𝑋 (𝑡)) 𝑑𝑊 (𝑡) (D.16) so (D.14) follows. Now, if 𝑄 is not smooth, Proposition B.1 shows that it is still (1/𝐾)-Lipschitz continuous, which, equivalently, means that ℎ * is (1/𝐾)-Lipschitz smooth. Thus, (D.13) follows by the weak Itô formula for Lipschitz smooth functions (C.7) applied to ℎ * , and noting that tr[𝜎(𝑥)𝜎(𝑥) ⊤ ] ≤ 𝑑𝜎 2 max . ■ D.3. The null-monotone case. We begin our analysis proper with our result for null-monotone games, which we restate below for convenience. Theorem 1 (Null-monotone games). Suppose that (S-FTRL) is run with a smooth mirror map 𝑄 in a null-monotone game G. Suppose further that the game admits an interior equilibrium 𝑥 * , and consider the hitting times 𝜏 - 𝜀 := inf{𝑡 > 0 : 𝐹 𝑡 ≤ 𝐹 0 -𝜀} and 𝜏 + 𝜀 := inf{𝑡 > 0 : 𝐹 𝑡 ≥ 𝐹 0 + 𝜀}. If 𝜎 2 min > 0 and 𝜀 > 0 is small enough, then 𝔼 𝑥 0 [𝜏 - 𝜀 ] = ∞ and 𝔼 𝑥 0 [𝜏 + 𝜀 ] ≤ 2𝜀 𝜅 𝜎 2 min (15) for some constant 𝜅 ≡ 𝜅 𝜀 > 0; in addition, 𝑋 (𝑡) does not admit a limiting distribution in this case. Proof. We start with the decreasing case, where we argue by contradiction. Specifically, let 𝑥 * be an equilibrium of G, and assume that 𝔼 𝑥 0 [𝜏 - 𝜀 ] < ∞. Then, by applying Dynkin's formula to the energy function 𝐸 (𝑦) at 𝜏 - 𝜀 for 𝑝 ← 𝑥 * (cf. Propositions C.1 and D.2), we readily get 𝔼 𝑥 0 [𝐸 (𝑌(𝜏 - 𝜀 ))] = 𝐸 (𝑦 0 ) + 𝔼 𝑥 0 ∫ 𝜏 - 𝜀 0 ⟨𝑣(𝑋 (𝑠)), 𝑋 (𝑠) -𝑥 * ⟩ + 1 2 tr[Σ(𝑋 (𝑠)) Jac 𝑄(𝑌(𝑠))] 𝑑𝑠 = 𝐹 0 + 1 2 𝔼 𝑥 0 ∫ 𝜏 - 𝜀 0 tr[Σ(𝑋 (𝑠)) Jac 𝑄(𝑌(𝑠))] 𝑑𝑠 # by null monotonicity ≥ 𝐹 0 (D.17)
where the last line follows from the fact that Σ and Jac 𝑄 are both positive-semidefinite. However, since 𝔼 𝑥 0 [𝐸 (𝑌(𝜏 - 𝜀 ))] = 𝐹 0 -𝜀 by the definition of 𝜏 - 𝜀 , we get 𝐹 0 -𝜀 ≥ 𝐹 0 , a contradiction which establishes our claim.
Since 𝜎 min > 0, we further conclude that 𝑌(𝑡) is uniformly elliptic. Thus, for any compact set K ⊆ {𝑦 ∈ Y : 𝐹 (𝑥 * , 𝑦) ≤ 𝐹 0 -𝜀}, the hitting time 𝜏 K = inf{𝑡 > 0 : 𝑌(𝑡) ∈ K} will be infinite on average (because
𝔼 𝑥 0 [𝜏 K ] ≥ 𝔼 𝑥 0 [𝜏 -
𝜀 ] = ∞), so, by Theorem C.1, 𝑌(𝑡) cannot be positive recurrent. In turn, this implies that 𝑌(𝑡) does not admit an invariant measure on Y, which proves our claim. Finally, for the second part of (15), applying Dynkin's formula to the energy function 𝐸 (𝑦) for 𝑝 ← 𝑥 * at the truncated hitting times 𝜏 + 𝜀 ∧ 𝑡, 𝑡 > 0, we get:
𝔼 𝑥 0 [𝐸 (𝑌(𝜏 + 𝜀 ) ∧ 𝑡)] = 𝐸 (𝑦 0 ) + 𝔼 𝑥 0 ∫ 𝜏 + 𝜀 ∧𝑡 0 ⟨𝑣(𝑋 (𝑠)), 𝑋 (𝑠) -𝑥 * ⟩ + 1 2 tr[Σ(𝑋 (𝑠)) Jac 𝑄(𝑌(𝑠))] 𝑑𝑠 = 𝐹 0 + 1 2 𝔼 𝑥 0 ∫ 𝜏 + 𝜀 ∧𝑡 0 tr[Σ(𝑋 (𝑠)) Jac
𝑄(𝑌(𝑠))] 𝑑𝑠 # by null monotonicity ≥ 𝐹 0 + 𝜎 2 min 2 𝔼 𝑥 0 ∫ 𝜏 + 𝜀 ∧𝑡 0 tr[Jac 𝑄(𝑌(𝑠))] 𝑑𝑠 (D.18) where the last line follows from the estimate tr[Σ Jac 𝑄] = tr[(Jac 𝑄) 1/2 Σ(Jac 𝑄) 1/2 ] = (1, . . . , 1) • (Jac 𝑄) 1/2 Σ(Jac 𝑄) 1/2 • (1, . . . , 1) ⊤ ≥ 𝜎 2 min (1, . . . , 1) • (Jac 𝑄) 1/2 • (Jac 𝑄) 1/2 • (1, . . . , 1) ⊤ = 𝜎 2 min tr[Jac 𝑄] . (D.19)
By the assumptions of the theorem (smooth 𝑄 and interior initialization), it follows that the (necessarily compact) set
D 𝜀 :={𝑥 = 𝑄(𝑦) : 𝐹 (𝑥 * , 𝑦) ≤ 𝐹 0 + 𝜀} is contained in the relative interior ri X of X .
In turn, this implies that 𝜅 ≡ 𝜅 𝜀 := min{tr[Jac 𝑄(𝑦)] : 𝐹 (𝑥 * , 𝑦) ≤ 𝐹 0 + 𝜀} > 0, so (D.18) becomes
𝐹 0 + 𝜀 ≥ 𝐹 0 + 1 2 𝜅𝜎 2 min 𝔼 𝑥 0 [𝜏 + 𝜀 ∧ 𝑡] (D.20)
and hence
𝔼 𝑥 0 [𝜏 + 𝜀 ∧ 𝑡] ≤ 2𝜀 𝜅𝜎 2 min for all 𝑡 ≥ 0. (D.21)
This shows that 𝔼 𝑥 0 [𝜏 + 𝜀 ∧ 𝑡] is uniformly bounded, so the upper bound in (15) follows by letting 𝑡 → ∞ (which implies 𝜏 + 𝜀 ∧𝑡 → 𝜏 + 𝜀 pointwise), and invoking the dominated convergence theorem. ■ D.4. The strongly monotone case. We now turn to our main result for strongly monotone games. Our proof strategy draws on methods related to the analysis of (S-FTRL) in the context of convex minimization, as explored by [47], and incorporating ideas that can be traced back to [28].
For convenience, we begin by restating Theorem 2. Theorem 2 (Strongly monotone games). Suppose that (S-FTRL) is run in an 𝛼-strongly monotone game G, and consider the hitting time
𝜏 𝑟 := inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )}(16)
where 𝔹 𝑟 (𝑥 * ) = {𝑥 : ∥𝑥 -𝑥 * ∥ ≤ 𝑟 } is a ball of radius 𝑟 centered on the (necessarily unique) equilibrium 𝑥 * of G. Then:
𝔼 𝑥 0 [𝜏 𝑟 ] ≤ (𝐹 0 /𝛼) (𝑟 2 -𝑟 2 𝜎 ) for all 𝑟 > 𝑟 𝜎 ,(17)
where 𝑟 𝜎 := 𝜎 max / √ 2𝐾𝛼. If, in addition, 𝜎 min > 0 and 𝑥 * is interior, 𝑋 (𝑡) admits an invariant distribution concentrated in a ball of radius O(𝜎 max ) around 𝑥 * , and we have
lim 𝑡→∞ 𝜇 𝑡 (𝔹 𝑟 (𝑥 * )) ≥ 1 -𝑟 2 𝜎 /𝑟 2 for all 𝑟 > 𝑟 𝜎 .(18)
Proof. Our proof proceeds along the following basic steps:
this section cite: ['b36', 'b47', 'b28']

Section: Step 1. Deriving an estimate for the mean hitting time
𝔼 𝑥 0 [𝜏 𝑟 ].
this section cite: []

Section: Step 2.
Descending to a restricted process Ỹ(𝑡) where any redundant degrees of freedom in 𝑌(𝑡) have been "modded out".
this section cite: []

Section: Step 3.
Showing that the restricted process is positive recurrent.
this section cite: []

Section: Step 4. Estimating the resulting invariant distribution and pushing the result forward to 𝑋 (𝑡).
In what follows, we go through the steps outlined above, one at a time.
this section cite: []

Section: Step 1: Estimating the hitting time.
We begin with the hitting time estimate (17). To that end, setting 𝑝 ← 𝑥 * in Proposition D.2, we get
𝐸 (𝑌(𝜏)) -𝐸 (𝑦 0 ) = ∫ 𝜏 0 ⟨𝑣(𝑋 (𝑠)), 𝑋 (𝑠) -𝑥 * ⟩ 𝑑𝑠 + 1 2𝐾 ∫ 𝜏 0 tr[Σ(𝑋 (𝑠))] 𝑑𝑠 + ∫ 𝜏 0 (𝑋 (𝑠) -𝑥 * ) ⊤ 𝜎(𝑋 (𝑠)) 𝑑𝑊 (𝑠) ≤ -𝛼 ∫ 𝜏 0 ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑑𝑠 + 𝜎 2 max 2𝐾 𝜏 + 𝑀(𝜏) . (D.22)
where we set
𝑀(𝑡) = ∫ 𝑡 0 (𝑋 (𝑠) -𝑥 * ) ⊤ 𝜎(𝑋 (𝑠)) 𝑑𝑊 (𝑠) . (D.23)
Thus, by a quick rearrangement, we obtain
𝛼 ∫ 𝜏 0 ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑑𝑠 ≤ 𝐸 (𝑦 0 ) -𝐸 (𝑌(𝜏)) + 𝜎 2 max 𝜏 2𝐾 + 𝑀(𝜏) (D.24)
and hence, with 𝐸 ≥ 0:
∫ 𝜏 0 ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑑𝑠 ≤ 𝐹 0 𝛼 + 𝜎 2 max 𝜏 2𝛼𝐾 + 𝑀(𝜏) 𝛼 . (D.25)
Thus, applying the above to the truncated hitting time 𝜏 ← 𝜏 𝑟 ∧ 𝑡 ≡ min{𝜏 𝑟 , 𝑡}, 𝑡 > 0, we get
𝑟 2 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] ≤ 𝔼 𝑥 0 ∫ 𝜏 𝑟 ∧𝑡 0 ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑑𝑠 # b/c ∥ 𝑋 (𝑠) -𝑥 * ∥ ≥ 𝑟 for 𝑠 ≤ 𝜏 𝑟 ∧ 𝑡 ≤ 𝐹 0 𝛼 + 𝑟 2 𝜎 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] + 1 𝛼 𝔼 𝑥 0 [𝑀(𝜏 𝑟 ∧ 𝑡)] . (D.26)
Since 𝜏 𝑟 ∧ 𝑡 ≤ 𝑡 is uniformly bounded, we will have 𝔼 𝑥 0 [𝑀(𝜏 𝑟 ∧ 𝑡)] = 𝔼[𝑀(0)] = 0 by the optional sampling theorem for continuous-time martingales [31,Theorem 3.22]. Thus, a simple rearrangement gives
𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] ≤ 𝐹 0 /𝛼 𝑟 2 -𝑟 2 𝜎 for all 𝑡 ≥ 0. (D.27)
This shows that 𝔼 𝑥 0 [𝜏 𝑟 ∧ 𝑡] is uniformly bounded, so the bound (17) follows by letting 𝑡 → ∞ (which implies 𝜏 𝑟 ∧ 𝑡 → 𝜏 𝑟 pointwise), and invoking the dominated convergence theorem.
this section cite: ['b16', 'b31']

Section: Step 2: Descending to the restricted process.
We now proceed to examine the recurrence properties of 𝑋 (𝑡). To that end, note first that the assumption 𝜎 min > 0 directly implies that (S-FTRL) is uniformly elliptic in the sense discussed in Appendix C. As such, consider the set
D 𝑟 := 𝑄 -1 (𝔹 𝑟 (𝑥 * )) = {𝑦 ∈ Y : ∥𝑄(𝑦) -𝑥 * ∥ ≤ 𝑟 } . (D.28)
and note that 𝜏 𝑟 = inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )} = inf{𝑡 > 0 : 𝑌(𝑡) ∈ D 𝑟 } (D.29) so 𝑌(𝑡) is positive recurrent relative to D 𝑟 . Thus, if D 𝑟 is compact, Theorem C.1 immediately shows that 𝑌(𝑡) is positive recurrent, and hence admits an invariant measure 𝜈 on Y. In general however, D 𝑟 need not be compact, so we cannot conclude that 𝑌(𝑡) is recurrent from the fact that it hits D 𝑟 in finite time on average.
To circumvent this difficulty, we will consider a "restricted" process which is positive recurrent, while retaining all information present in 𝑌(𝑡). The main idea here will be to "collapse" the fibers of 𝑄, that is, those directions in Y which map to the same point in X under 𝑄: since the dynamics (S-FTRL) factor through 𝑋 (𝑡) = 𝑄(𝑌(𝑡)), these directions carry no relevant information, so they can be effectively discarded.
To carry all this out, let Ṽ denote the "tangent hull" of X in V, viz.
Ṽ := aff (X -X ) = {𝑧 ∈ V : 𝑥 + 𝑡𝑧 ∈ X for all sufficiently small 𝑡 > 0 and all 𝑥 ∈ ri X } . (D. 30) In words, Ṽ is the smallest subspace of V which contains X when the latter is translated to the origin so, by construction, X is full-dimensional when viewed as a subset of Ṽ. 7 In this sense, Ṽ contains all the "essential" directions of motion of the problem.
Dually to the above, we also consider the corresponding dual space Ỹ ≡ Ṽ * of Ṽ; this is not a subspace of Y, but there exists a canonical surjection Π : Y ↠ Ỹ defined by restricting the action of 𝑦 ∈ Y to Ṽ, that is, ⟨Π(𝑦), 𝑧⟩ = ⟨𝑦, 𝑧⟩ for all 𝑧 ∈ Ṽ. (D.31) The kernel of Π is precisely the annihilator Ann( Ṽ) of Ṽ, i.e.,
ker Π = Ann( Ṽ) ≡ {𝑤 ∈ Y : ⟨𝑤, 𝑧⟩ = 0 for all 𝑧 ∈ Ṽ } (D.32)
so, by the first isomorphism theorem, we get a canonical identification (V/ Ṽ) * ker Π.
The main reason for descending from Y to Ỹ is the following: in the original space Y, we have 𝑄(𝑦 + 𝑤) = 𝑄(𝑦) whenever 𝑤 annihilates Ṽ, cf. Proposition B.1. As a result, the inverse image of any compact subset of X under 𝑄 will always contain a copy of Ann( Ṽ), so it can never be compact itself. By contrast, by "modding out" Ann( Ṽ) and descending to the restricted space Ỹ, this is no longer the case.
To move forward, consider the restricted mirror map Q : Ỹ → X given by Q( ỹ) = 𝑄(𝑦) whenever Π(𝑦) = ỹ. (D.33)
By the last item of Proposition B.1 we have 𝑄(𝑦) = 𝑄(𝑦 + 𝑤) whenever 𝑤 ∈ Ann( Ṽ); this means that the choice of representative in (D.33) does not matter, so Q is well-defined. Accordingly, letting
Ỹ(𝑡) = Π(𝑌(𝑡)) (D.34)
and applying Π to (S-FTRL) yields the "restricted" dynamics
𝑑 Ỹ(𝑡) = 𝑑 (Π • 𝑌(𝑡)) = Π • 𝑣(𝑋 (𝑡)) 𝑑𝑡 + Π • 𝜎(𝑋 (𝑡)) • 𝑑𝑊 (𝑡) (D.35)
where
𝑋 (𝑡) = 𝑄(𝑌(𝑡)) = Q( Ỹ(𝑡))
and, in a slight abuse of notation, we are overloading the symbol Π to denote both the linear map Π : Y → Ỹ and its representation as a matrix. These dynamics represent a time-homogeneous SDE in terms of Ỹ , and they will be our main object of study in the rest of our proof.
this section cite: ['b30']

Section: Step 3: Positive recurrence of the restricted process.
With all this in hand, positive recurrence for the restricted process Ỹ(𝑡) boils down to the following: a) verifying that the infinitesimal generator of Ỹ is uniformly elliptic; and b) showing that the mean time required for Ỹ(𝑡) to reach some compact set of Ỹ is finite.
We begin by establishing uniform ellipticity. In view of (D.35), the principal symbol (C.13) of the infinitesimal generator of Ỹ(𝑡) is
𝐴 = (Π • 𝜎) • (Π • 𝜎) ⊤ = Π𝜎𝜎 ⊤ Π ⊤ = ΠΣΠ ⊤ . (D.36)
Since Σ ≽ 𝜎 2 min 𝐼, we readily get
𝐴 ≽ 𝜎 2 min ΠΠ ⊤ ≽ 𝜎 2 min 𝜋 2 min 𝐼 (D.37)
with 𝜎 min > 0 (by assumption) and 𝜋 min := 𝜆 min (ΠΠ ⊤ ) > 0 (because Π is surjective, so it has full rank). This shows that the principal symbol ΠΣΠ ⊤ of the generator of Ỹ is uniformly positive-definite, so Ỹ is itself uniformly elliptic.
For the second component of our proof of positive recurrence, recall that 𝑥 * ∈ ri X , so there exists some sufficiently small 𝑟 > 0 such that the (compact convex) set
K 𝑟 := 𝔹 𝑟 ∩ X = {𝑥 ∈ X : ∥𝑥 -𝑥 * ∥ ≤ 𝑟 } (D.38)
lies in its entirety within ri X . We then claim that the inverse image
D𝑟 := Q-1 (K 𝑟 ) = { ỹ ∈ Ỹ : ∥ Q( ỹ) -𝑥 * ∥ ≤ 𝑟 } (D.39)
of K 𝑟 under the restricted mirror map Q is compact. To see this, note first that D𝑟 = 𝜕ℎ(K 𝑟 ) by Proposition B.1. 8 Thus, given that K 𝑟 is a convex body in Ṽ that is entirely contained in the (relative) interior of the prox-domain
X ℎ of ℎ (because ri X ⊆ dom 𝜕ℎ ≡ X ℎ ), it follows that 𝜕ℎ(K 𝑟 ) is itself compact by the upper hemicontinuity of 𝜕ℎ [24, Remark 6.2.3].
To conclude, note that
inf{𝑡 > 0 : Ỹ(𝑡) ∈ D𝑟 } = inf{𝑡 > 0 : ∥ Q( Ỹ(𝑡)) -𝑥 * ∥ ≤ 𝑟 } = inf{𝑡 > 0 : ∥𝑄(𝑌(𝑡)) -𝑥 * ∥ ≤ 𝑟 } = inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )} = 𝜏 𝑟 (D.40)
so it follows from (17) that Ỹ(𝑡) hits D𝑟 in finite time on average. Since Ỹ(𝑡) is uniformly elliptic, Theorem C.1 shows that it is positive recurrent, as claimed.
this section cite: []

Section: Step 4: Estimating the long-run occupation measure.
Since the restricted process Ỹ(𝑡) is a positive recurrent Itô diffusion, standard results show that it admits an invariant distribution ν on Ỹ which satisfies the law of large numbers 41) for every ν-integrable test function 𝑓 on Ỹ. Thus, letting 𝜈 = Q * ν ≡ ν • Q-1 denote the corresponding push-forward measure on X , we get
lim 𝑡→∞ 1 𝑡 ∫ 𝑡 0 𝑓 ( Ỹ(𝑠)) 𝑑𝑠 = ∫ Ỹ 𝑓 𝑑 ν (D.
lim 𝑡→∞ 𝜇 𝑡 (𝔹 𝑟 ) = lim 𝑡→∞ 1 𝑡 ∫ 𝑡 0 1{𝑋 (𝑠) ∈ 𝔹 𝑟 } 𝑑𝑠 = lim 𝑡→∞ 1 𝑡 ∫ 𝑡 0 1{ Q( Ỹ(𝑠)) ∈ 𝔹 𝑟 } 𝑑𝑠 = lim 𝑡→∞ 1 𝑡 ∫ 𝑡 0 1{ Ỹ(𝑠) ∈ D𝑟 } 𝑑𝑠 = ∫ Ỹ 1{ ỹ ∈ D𝑟 } 𝑑 ν( ỹ) = ν( D𝑟 ) .
(D.42)
In a similar manner, we also get
1 -ν( D𝑟 ) = lim 𝑡→∞ 1 𝑡 𝔼 ∫ 𝑡 0 1{𝑋 (𝑠) ∉ 𝔹 𝑟 } 𝑑𝑠 # b/c lim 𝑡→∞ 𝜇 𝑡 is deterministic ≤ lim 𝑡→∞ 1 𝑡 𝔼 ∫ 𝑡 0 ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑟 2 𝑑𝑠 # b/c ∥ 𝑋 (𝑠) -𝑥 * ∥ 2 𝑟 2 ≥ 1 outside 𝔹 𝑟 ≤ lim 𝑡→∞ 1 𝑟 2 𝐹 0 𝛼𝑡 + 𝜎 2 max 2𝛼𝐾 # by (D.25) = 𝑟 2 𝜎 𝑟 2 .
(D.43)
Our claim then follows by combining Eqs. (D.42) and (D. 43).
this section cite: ['b41', 'b43']

Section: ■

this section cite: []

Section: E Analysis and results in discrete time
In this appendix, we proceed to prove the discrete-time results presented in Section 4.
this section cite: []

Section: E.1. The null-monotone case.
We begin with our analysis for for null-monotone games. For convenience, we restate Theorem 3 below:
Theorem 3 (Null-monotone games). Suppose that (FTRL) is run in a null-monotone game G, and let 𝑥 * be an equilibrium of G. Suppose further that ℎ * is strongly convex, and let 𝐹 𝑡 = 𝐹 (𝑥 * , 𝑌 𝑡 ), where 𝐹 is the induced Fenchel coupling (B.22). Then lim 𝑡→∞ 𝔼[𝐹 𝑡 ] = ∞.
Proof. By a second-order Taylor expansion with Lagrange remainder, there exists 𝑤 𝑡 ∈ [𝑌 𝑡 , 𝑌 𝑡+1 ] such that:
𝐹 𝑡+1 = 𝐹 𝑡 + 𝛾⟨v 𝑡 , 𝑋 𝑡 -𝑥 * ⟩ + 𝛾 2 2 v𝑡 ∇ 2 ℎ * (𝑤 𝑡 ) v𝑡 . (E.1)
Since G is null-monotone, we have ⟨𝑣(𝑋 𝑡 ), 𝑋 𝑡 -𝑥 * ⟩ = 0 by assumption, and thus
𝐹 𝑡+1 = 𝐹 𝑡 + 𝛾⟨𝑈 𝑡 , 𝑋 𝑡 -𝑥 * ⟩ + 𝛾 2 2 v𝑡 ∇ 2 ℎ * (𝑤 𝑡 ) v𝑡(
E.2) ≥ 𝐹 𝑡 + 𝛾⟨𝑈 𝑡 , 𝑋 𝑡 -𝑥 * ⟩ + 𝑚 2 𝛾 2 ∥ v𝑡 ∥ 2 * (E.3) where 𝑚 denotes here the strong convexity modulus of ℎ * . Moving forward, note that (i) 𝔼[⟨𝑈 𝑡 , 𝑋 𝑡 -𝑥 * ⟩] = 𝔼[⟨𝔼[𝑈 𝑡 | F 𝑡 ], 𝑋 𝑡 -𝑥 * ⟩] = 0; and (ii) inf 𝑡 𝔼 ∥ v𝑡 ∥ 2 * ≥ inf 𝔼[∥V(𝑥; 𝜔) ∥ 2 * ] > 0, so there exists some 𝑉 * > 0 such that 𝔼[∥ v𝑡 ∥ 2 * ] ≥ 𝑉 2 * for all 𝑡. We thus get 𝔼[𝐹 𝑡+1 ] ≥ 𝔼[𝐹 𝑡 ] + 𝑚 2 𝛾 2 𝔼 ∥ v𝑡 ∥ 2 * ≥ 𝔼[𝐹 𝑡 ] + 𝑚𝛾 2 𝑉 2 * /2 ≥ 𝐹 0 + 𝑚𝛾 2 𝑉 2 * 𝑡/2 (E.4) Our result then follows by taking the limit 𝑡 → ∞. ■ E.2. The strongly monotone case. We now turn to our main result for strongly monotone games, which we restate below for convenience. Theorem 4 (Strongly monotone games). Suppose that (FTRL) is run in an 𝛼-strongly monotone game G, and consider the hitting time
𝜏 𝑟 := inf{𝑡 > 0 : 𝑋 (𝑡) ∈ 𝔹 𝑟 (𝑥 * )} (22
)
where 𝔹 𝑟 (𝑥 * ) = {𝑥 : ∥𝑥 -𝑥 * ∥ ≤ 𝑟 } is a ball of radius 𝑟 centered on the (necessarily unique) equilibrium 𝑥 * of G. Then, for all 𝑟 > 𝑟 𝜎 := √︁ 𝛾(𝜎 2 + 𝛽 2 )/(𝛼𝐾), we have
𝔼[𝜏 𝑟 ] ≤1
𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) × 𝐹 0 if 𝑋 0 ∉ 𝔹 𝑟 (𝑥 * ), 𝐹 0 + 𝛼𝛾𝑟 2 if 𝑋 0 ∈ 𝔹 𝑟 (𝑥 * ),(23)
where 𝐹 0 = 𝐹 (𝑥 * , 𝑌 0 ). If, in addition, 𝑥 * is interior, 𝑋 𝑡 admits a unique invariant distribution to which it converges in total variation, and we have
lim 𝑡→∞ 1 𝑡 𝔼 𝑡 ∑︁ 𝑠=0 1{𝑋 𝑡 ∈ 𝔹 𝑟 (𝑥 * )} ≥ 1 -𝑟 2 𝜎 𝑟 2(24)
for all 𝑟 > 𝑟 𝜎 such that 𝔹 𝑟 (𝑥 * ) ⊆ ri X .
Proof. The main theme of our proof shadows the continuous-time analysis, but it requires distinct tools and techniques to address the specific challenges that arise in the discrete-time Markov chain setting (where, among others, the main tools of stochastic calculus cannot be applied). In a nutshell, this proceeds along the following sequence of steps. First, we derive an upper bound on the expected hitting time of the process to a neighborhood of the equilibrium. Subsequently, we reduce the dynamics to a "reduced space" (formally an affine quotient of the dual space), removing redundant directions and ensuring the process evolves within a minimal and non-degenerate domain. Within this reduced space, we show that the induced Markov process satisfies several crucial probabilistic properties. Specifically, we prove:
• Irreducibility: any open set in the state space can be reached with positive probability.
• Minorization: after entering certain regions of the space, the process mixes sufficiently to allow for probabilistic regeneration.
• Uniform control of return times: the expected time to revisit a neighborhood of equilibrium remains bounded regardless of the starting point within that neighborhood.
These properties collectively enable the construction of a regeneration structure, a probabilistic framework that ensures the process repeatedly returns to a well-behaved region of the state space with sufficient mixing. In turn, this enables us to establish positive Harris recurrence of the learning dynamics, a key property which ensures the existence and uniqueness of a stationary invariant distribution.
To streamline our presentation, we follow a step-by-step approach, as outlined below.
this section cite: []

Section: Step 1: Deriving a hitting estimate.
Due to measurability issues, we cannot apply Dynkin's lemma directly in the discrete-time setting, which makes the proof more involved. Moreover, unlike in the continuous-time regime, we need to distinguish between different initializations. Specifically, we consider two cases depending on whether the initial state 𝑋 0 lies within the ball 𝔹 𝑟 (𝑥 * ) or not.
• Case 1: 𝑋 0 ∉ 𝔹 𝑟 (𝑥 * ).
Letting 𝐹 𝑡 := 𝐹 (𝑥 * , 𝑌 𝑡 ) and unfolding (B.29b), we readily obtain:
𝐹 𝑡 ≤ 𝐹 0 + 𝛾 𝑡 -1 ∑︁ 𝑠=0 ⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 𝑡 -1 ∑︁ 𝑠=0 ∥ v𝑠 ∥ 2 * (E.5)
and, setting 𝑡 ← 𝜏 𝑟 ∧ 𝑡, we get:
𝐹 𝜏 𝑟 ∧𝑡 ≤ 𝐹 0 + 𝛾 ( 𝜏 𝑟 ∧𝑡 ) -1 ∑︁ 𝑠=0 ⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 ( 𝜏 𝑟 ∧𝑡 ) -1 ∑︁ 𝑠=0 ∥ v𝑠 ∥ 2 * (E.6)
Thus, taking expectation conditional on the initial state 𝑌 0 = 𝑦, we have
𝔼 𝐹 𝜏 𝑟 ∧𝑡 ≤ 𝐹 0 + 𝛾 𝔼 ( 𝜏 𝑟 ∧𝑡 ) -1 ∑︁ 𝑠=0 ⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 𝔼 ( 𝜏 𝑟 ∧𝑡 ) -1 ∑︁ 𝑠=0 ∥ v𝑠 ∥ 2 * ≤ 𝐹 0 + 𝑡 -1 ∑︁ 𝑠=0 𝔼 𝛾⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 ∥ v𝑠 ∥ 2 * 1(𝜏 𝑟 ≥ 𝑠 + 1) (E.7)
For notational convenience, we denote each summand above per
𝐷 𝑠 := 𝔼 𝛾⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 ∥ v𝑠 ∥ 2 * 1(𝜏 𝑟 ≥ 𝑠 + 1)
and noting that the random variable 1(𝜏 𝑟 ≥ 𝑠 + 1) is F 𝑠 -measurable, we get
𝐷 𝑠 = 𝔼 𝔼 𝛾⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 ∥ v𝑠 ∥ 2 * 1(𝜏 𝑟 ≥ 𝑠 + 1) F 𝑠 = 𝔼 1(𝜏 𝑟 ≥ 𝑠 + 1) 𝔼 𝛾⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 ∥ v𝑠 ∥ 2 * F 𝑠 = 𝔼 1(𝜏 𝑟 ≥ 𝑠 + 1) 𝛾⟨𝑣(𝑋 𝑠 ), 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 𝔼 ∥ v𝑠 ∥ 2 * F 𝑠 = 𝔼
1(𝜏 𝑟 ≥ 𝑠 + 1) -𝛾𝛼∥ 𝑋 𝑠 -𝑥 * ∥ 2 + 𝛾 2 2𝐾 𝔼 ∥ v𝑠 ∥ 2 * F 𝑠 (E.8) where we used that 𝔼[𝑈 (𝑋 𝑠 , 𝜔 𝑠 ) | F 𝑠 ] = 0. At this point, we note that 𝔼 ∥ v𝑠 ∥ 2 * | F 𝑠 ≤ 2 𝔼 ∥𝑣(𝑋 𝑠 ) ∥ 2 * + ∥𝑈 (𝑋 𝑠 , 𝜔 𝑠 ) ∥ 2 * | F 𝑠 ≤ 2(𝛽 2 + 𝜎 2 ), and since 𝑟 2 𝜎 ≡ 𝛾(𝛽 2 + 𝜎 2 )/(𝛼𝐾), we get
𝐷 𝑠 ≤ 𝔼 -𝛾𝛼∥ 𝑋 𝑠 -𝑥 * ∥ 2 + 𝛼𝛾𝑟 2 𝜎 1(𝜏 𝑟 ≥ 𝑠 + 1) ≤ 𝔼 -𝛾𝛼𝑟 2 + 𝛼𝛾𝑟 2 𝜎 1(𝜏 𝑟 ≥ 𝑠 + 1) (E.9)
where in the last step we used that ∥ 𝑋 𝑠 -𝑥 * ∥ ≥ 𝑟 2 on {𝜏 𝑟 ≥ 𝑠 + 1}. Thus, plugging the above bound into (E.7), we obtain
𝔼 𝐹 𝜏 𝑟 ∧𝑡 ≤ 𝐹 0 + 𝑡 -1 ∑︁ 𝑠=0 𝐷 𝑠 ≤ 𝐹 0 + 𝑡 -1 ∑︁ 𝑠=0 𝔼 -𝛾𝛼𝑟 2 + 𝛼𝛾𝑟 2 𝜎 1(𝜏 𝑟 ≥ 𝑠 + 1) = 𝐹 0 -𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) 𝔼 ( 𝜏 𝑟 ∧𝑡 ) -1 ∑︁ 𝑠=0 1 = 𝐹 0 -𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) 𝔼[(𝜏 𝑟 ∧ 𝑡)](
E.10) As 𝐹 is nonnegative, we readily obtain that 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) 𝔼[(𝜏 𝑟 ∧ 𝑡)] ≤ 𝐹 0 (E.11) and, since 𝑟 𝜎 < 𝑟, we get: 𝔼[(𝜏 𝑟 ∧ 𝑡)] ≤ 1 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) 𝐹 0 (E.12) Finally, taking 𝑡 → ∞, and invoking the monotone convergence theorem [19], we get 𝔼[𝜏 𝑟 ] ≤ 1 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) 𝐹 0 (E.13) • Case 2: 𝑋 0 ∈ 𝔹 𝑟 (𝑥 * ). In this case, we have: 𝔼[𝜏 𝑟 ] = 𝔼[1(𝑄(𝑌 1 ) ∈ 𝔹 𝑟 (𝑥 * )) + 1(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * ))𝜏 𝑟 ] = ℙ(𝑄(𝑌 1 ) ∈ 𝔹 𝑟 (𝑥 * )) + 𝔼 1(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * )) (1 + 𝔼 𝑌 1 [𝜏 𝑟 ]) = ℙ(𝑄(𝑌 1 ) ∈ 𝔹 𝑟 (𝑥 * )) + ℙ(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * )) + 𝔼 1(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * )) 𝔼 𝑌 1 [𝜏 𝑟 ] = 1 + 𝔼 1(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * )) 𝔼 𝑌 1 [𝜏 𝑟 ] ≤ 1 + 𝔼 1(𝑄(𝑌 1 ) ∉ 𝔹 𝑟 (𝑥 * )) 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1 𝐹 (𝑥 * , 𝑌 1 ) ≤ 1 + 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1 𝔼[𝐹 (𝑥 * , 𝑌 1 )] ≤ 1 + 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1
𝔼 𝐹 0 + 𝛾⟨v 0 , 𝑋 0 -𝑥 * ⟩ + 𝛾 2 2𝐾 ∥ v0 ∥ 2 * ≤ 1 + 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1 𝐹 0 + 𝛾⟨𝑣(𝑋 0 ), 𝑋 0 -𝑥 * ⟩ + 𝛼𝛾𝑟 2 𝜎 ≤ 1 + 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1 𝐹 0 -𝛾𝛼∥ 𝑋 0 -𝑥 * ∥ 2 + 𝛼𝛾𝑟 2 𝜎 ≤ 1 + 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) -1 𝐹 0 + 𝛼𝛾𝑟 2 𝜎 = 𝐹 0 + 𝛼𝛾𝑟 2 𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) (E.14)
Thus, collectively, we get: and applying Π to (FTRL) yields the "restricted" process
𝔼[𝜏 𝑟 ] ≤1
𝛼𝛾(𝑟 2 -𝑟 2 𝜎 ) × 𝐹 0 if 𝑋 0 ∉ 𝔹 𝑟 (𝑥 * ), 𝐹 0 + 𝛼𝛾𝑟 2 if 𝑋 0 ∈ 𝔹 𝑟 (𝑥 * ), (E.15)
Ỹ𝑡+1 = Π • 𝑌 𝑡+1 = Π • 𝑌 𝑡 + 𝛾(Π • 𝑣(𝑋 𝑡 ) + Π • 𝑈 𝑡 ) = Ỹ𝑡 + 𝛾(ṽ(𝑋 𝑡 ) + Ũ𝑡 ) (E.20)
where 𝑋 𝑡 = 𝑄(𝑌 𝑡 ) = Q( Ỹ𝑡 ) and, in a slight abuse of notation, we are overloading the symbol Π to denote both the linear map Π : Y → Ỹ and its representation as a matrix. Finally, writing Ỹ as
Ỹ𝑡+1 = Ỹ𝑡 + 𝛾 ṽ Q( Ỹ𝑡 ) + 𝛾U Q( Ỹ𝑡 ), 𝜔 𝑡 (E.21)
we conclude that it is a time-homogeneous Markov process, and we denote its kernel by q, where for any ỹ ∈ Ỹ and Borel set A ⊆ Ỹ, we have q( ỹ, A) = ℙ Ỹ𝑡+1 ∈ A Ỹ𝑡 = ỹ .
this section cite: []

Section: Step 3: Recurrence of the restricted process.
To establish the recurrence of the restricted process, we first need to understand the effect of Π on the distribution of U(𝑥). As stated in the assumptions in Section 4, the probability distribution 𝜈 𝑥 of U(𝑥) decomposes as 𝜈 𝑥 = 𝜈 𝑐 𝑥 + 𝜈 ⊥ 𝑥 . Noting the push-forward measure is linear, we readily obtain that Π * 𝜈 𝑥 = Π * 𝜈 𝑐 𝑥 + Π * 𝜈 ⊥ 𝑥 , where Π * 𝜈 𝑥 denotes the push-forward measure A ↦ → (𝜈 𝑥 • Π -1 ) (A). For notational convenience, we denote Y ≡ Ann( Ṽ) and 𝑝(𝑥, 𝑦) ≡ 𝑝 𝑥 (𝑦). Then, each 𝑦 ∈ Y can be decomposed as 𝑦 = ỹ + ŷ, and since Π has full column-rank, the measure Π * 𝜈 𝑐 𝑥 has density with respect to the Lebesgue measure 𝜆 Ỹ on Ỹ, given by
p(𝑥, ỹ) = ∫ Y 𝑝(𝑥, ỹ, ŷ)𝑑𝜆 Y ( ŷ) (E.22)
where 𝜆 Y is the Lebesgue measure on Y. Importantly, the density p satisfies the following properties, which will be crucial for establishing the recurrence of the process. We formalize these in the proposition below, whose proof is deferred until after the theorem.
Proposition E.1. Let the function p as defined in (E.22). Then:
(i) For any compact set K ⊆ X and every ỹ ∈ Ỹ, it holds inf 𝑥 ∈K p(𝑥, ỹ) > 0.
(ii) The function p is (jointly) lower semi-continuous.
this section cite: []

Section: Lebesgue irreducibility.
We now show that the restricted process Ỹ𝑡 is Lebesgue irreducible; that is, starting from any point in its domain, the process has a positive probability of reaching any open set with nonzero Lebesgue measure. This property is crucial for establishing recurrence, as it ensures that the process does not avoid regions of the space indefinitely.
For this, let a Borel measurable set A ⊆ Ỹ with 𝜆 Ỹ (A) > 0. We will show that q( ỹ, A) > 0 for all ỹ ∈ Ỹ, which implies that A can be reached from any state ỹ in one step with positive probability.
q( ỹ, A) = ℙ Ỹ𝑡+1 ∈ A Ỹ𝑡 = ỹ = ℙ ỹ + 𝛾 ṽ( Q( ỹ)) + 𝛾 Ũ( Q( ỹ), 𝜔) ∈ A = ℙ Ũ( Q( ỹ), 𝜔) ∈ 𝛾 -1 A -𝛾 -1 ỹ -ṽ( Q( ỹ)) = ℙ Ũ( Q( ỹ), 𝜔) ∈ A ỹ ≥ ∫ A ỹ p( Q( ỹ), 𝑧)𝑑𝜆 Ỹ (𝑧) (E.23)
where
A ỹ ≡ 𝛾 -1 A -𝛾 -1 ỹ -ṽ( Q( ỹ)) with 𝜆 Ỹ (A ỹ) = 𝛾 -𝑑 𝜆 Ỹ (A) > 0.
Finally, since p( Q( ỹ), •) strictly positive, we conclude that q( ỹ, A) > 0, which shows that Ỹ𝑡 induced by (E.20) is Lebesgueirreducible.
Harris recurrence. Our next step is to show that Ỹ𝑡 is Harris recurrent. This means that the process returns to every set of positive Lebesgue measure infinitely often with probability one. Establishing Harris recurrence is a key step toward proving ergodicity, as it ensures that the process does not drift away or get trapped. For this, we will show that D𝑟 = { ỹ ∈ Ỹ : ∥𝑄( ỹ) -𝑥 * ∥ ≤ 𝑟 } is a recurrent set from which we can go "everywhere" with positive probability. Importantly, the set D𝑟 is compact as shown in
Step 3 of Theorem 2. The first part to prove Harris recurrence is immediate from Step 1 of our proof; namely, since 𝔼 ỹ [𝜏 𝑟 ] < ∞ for any initial condition ỹ ∈ Ỹ, we readily get that ℙ ỹ (𝜏 𝑟 < ∞) = 1. For the second part, we will prove the so-called minorization property; that is, there exists a nontrivial measure 𝜇 and a constant 𝛼 > 0 such that q( ỹ, A) ≥ 𝛼𝜇(A) for all ỹ ∈ D𝑟 and Borel sets A ⊆ Ỹ. (E.24) This condition implies that, from any point in D𝑟 the process has a uniformly lower-bounded probability of reaching any set A in one step according to the reference measure 𝜇. To establish the minorization condition (E.24), we define for notational convenience the function 𝑓 : Ỹ × Ỹ → X ℎ × Ỹ as 𝑓 ( ỹ, 𝑧) = Q( ỹ), 𝛾 -1 (𝑧ỹ)ṽ( Q( ỹ)) (E.25) which is continuous as a composition of continuous functions. With this definition in hand, we perform the change of variables in (E.23), and we have: q( ỹ, A) ≥ 𝛾 -𝑑 ∫ A p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) ≥ 𝛾 -𝑑 ∫ A inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) (E.26) To finally construct the measure 𝜇, we need to ensure that 0 < ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) < ∞ (E.27) To this end, we state the following proposition, whose proof is deferred until after the theorem to maintain the flow. Proposition E.2. The density p satisfies: 0 < ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) < ∞ (E.28) With Proposition E.1 in hand, we define the measure 𝜇 as 𝜇(A) := ∫ A inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) for all Borel A ⊆ Ỹ. (E.29) Therefore, (E.26) becomes: q( ỹ, A) ≥ 𝛾 -𝑑 ∫ A inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ = 𝛾 -𝑑 ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) • 𝜇(A) (E.30) Thus, setting 𝛼 ≡ 𝛾 -𝑑 ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧), we conclude the minorization condition (E.24). Therefore, the set D𝑟 is recurrent and 𝜇( D𝑟 ) > 0 (since 𝜆 Ỹ ( D𝑟 ) > 0), and thus by [17, Proposition 11.2.1] the Markov process Ỹ𝑡 admits an invariant measure. In addition, based on the equivalence (D.40) the expected return time 𝔼[𝜏 𝑟 ] to D𝑟 is uniformly bounded for all initial conditions ỹ on D𝑟 , due to the continuity of the Fenchel coupling 𝐹. Therefore, invoking [55, Theorem 13.0.1],
we conclude that the process Ỹ𝑡 admits a unique invariant probability measure ν, and the law of Ỹ𝑡 converges to ν in total variation for every initial condition ỹ ∈ Ỹ.
this section cite: []

Section: Step 4: Estimating the long-run occupation measure.
Finally, for the last part, letting 𝐹 𝑡 := 𝐹 (𝑥 * , 𝑌 𝑡 ) and unfolding (B.29b), we obtain:
𝐹 𝑡 ≤ 𝐹 0 + 𝛾 𝑡 -1 ∑︁ 𝑠=0 ⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 𝑡 -1 ∑︁ 𝑠=0 ∥ v𝑠 ∥ 2 * (E.31)
Taking expectations in both sides, we readily get
0 ≤ 𝔼[𝐹 𝑡 ] ≤ 𝔼 𝐹 0 + 𝛾 𝑡 -1 ∑︁ 𝑠=0 ⟨v 𝑠 , 𝑋 𝑠 -𝑥 * ⟩ + 𝛾 2 2𝐾 𝑡 -1 ∑︁ 𝑠=0 ∥ v𝑠 ∥ 2 * ≤ 𝐹 0 -𝛼𝛾 𝔼 𝑡 -1 ∑︁ 𝑠=0 ∥ 𝑋 𝑠 -𝑥 * ∥ 2 + 𝑡𝛼𝛾𝑟 2 𝜎 (E.32)
Therefore, by rearranging terms and dividing both sides by 𝑡, we have:
1 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 ∥ 𝑋 𝑠 -𝑥 * ∥ 2 ≤ 1 𝛼𝛾𝑡 𝐹 0 + 𝑟 2 𝜎 (E.33)
Moreover, we have:
1 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 1{𝑋 𝑠 ∉ 𝔹 𝑟 (𝑥 * )} ≤ 1 𝑟 2 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 ∥ 𝑋 𝑠 -𝑥 * ∥ 2 ≤ 1 𝛼𝛾𝑡𝑟 2 𝐹 0 + 𝑟 2 𝜎 𝑟 2 (E.
34) Now, note that {𝑋 𝑠 ∉ 𝔹 𝑟 (𝑥 * )} ≡ { Ỹ𝑡 ∉ D𝑟 } by construction, and thus 1 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 1{𝑋 𝑠 ∉ 𝔹 𝑟 (𝑥 * )} = 1 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 1{ Ỹ𝑡 ∉ D𝑟 } (E.35) Taking 𝑡 → ∞, and invoking Birkhoff's individual ergodic theorem [23, Theorem 2.3.4], we readily get that the mean occupation measure A ↦ → 𝑡 -1 𝔼 𝑡 -1 𝑠=0 1{ Ỹ𝑡 ∈ A} converges strongly to the invariant measure ν, and therefore lim 𝑡→∞ 1 𝑡 𝔼 𝑡 -1
∑︁ 𝑠=0 1{𝑋 𝑠 ∉ 𝔹 𝑟 (𝑥 * )} = lim 𝑡→∞ 1 𝑡 𝔼 𝑡 -1 ∑︁ 𝑠=0 1{ Ỹ𝑡 ∉ D𝑟 } = 1 -ν( D𝑟 ) (E.36)
and, using (E.34), we have:
ν( D𝑟 ) ≥ 1 - 𝑟 2 𝜎 𝑟 2 (E.37)
and our proof is complete.
this section cite: []

Section: ■
To keep the presentation self-contained, we restate and prove Proposition E.1 and Proposition E.2 below. Proposition E.1. Let the function p as defined in (E.22). Then:
(i) For any compact set K ⊆ X and every ỹ ∈ Ỹ, it holds inf 𝑥 ∈K p(𝑥, ỹ) > 0.
(ii) The function p is (jointly) lower semi-continuous.
Proof.
(i) For the first part, let ỹ ∈ Ỹ. Then
inf 𝑥 ∈K p(𝑥, ỹ) = inf 𝑥 ∈K ∫ Y 𝑝(𝑥, ỹ, ŷ)𝑑𝜆 Y ( ŷ) ≥ ∫ Y inf 𝑥 ∈K 𝑝(𝑥, ỹ, ŷ)𝑑𝜆 Y ( ŷ) > 0 (E.38)
(ii) For the second part, let (𝑥, ỹ) ∈ X × Ỹ, and let a sequence {(𝑥 𝑡 , ỹ𝑡 )} 𝑡 ∈ℕ with lim 𝑡→∞ (𝑥 𝑡 , ỹ𝑡 ) = (𝑥, ỹ). Since 𝑝 is jointly continuous, applying Fatou's lemma [19], we get
p(𝑥, ỹ) = ∫ Y 𝑝(𝑥, ỹ, ŷ)𝑑𝜆 Y ( ŷ) = ∫ Y lim inf 𝑡→∞ 𝑝(𝑥 𝑡 , ỹ𝑡 , ŷ)𝑑𝜆 Y ( ŷ) ≤ lim inf 𝑡→∞ ∫ Y 𝑝(𝑥 𝑡 , ỹ𝑡 , ŷ)𝑑𝜆 Y ( ŷ) = lim inf 𝑡→∞ p(𝑥 𝑡 , ỹ𝑡 ) (E.
39) i.e., p(𝑥, ỹ) ≤ lim inf 𝑡→∞ p(𝑥 𝑡 , ỹ𝑡 ) (E.40) and the result follows. ■ Proposition E.2. The density p satisfies: 0 < ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) < ∞ (E.28) Proof. The upper bound is trivial since p is a probability density and ∫ Ỹ inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) ≤ ∫ Ỹ p( 𝑓 ( ỹ, 𝑧))𝑑𝜆 Ỹ (𝑧) ≤ 1 (E.41) For the lower bound, we will show that inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧)) > 0 for all 𝑧 ∈ Ỹ. (E.42) Suppose not, i.e., there exists 𝑧 0 ∈ Ỹ such that inf ỹ∈ D𝑟 p( 𝑓 ( ỹ, 𝑧 0 )) = 0. Since D𝑟 is compact and p • 𝑓 is lower semi-continuous, the infimum over D𝑟 is realized, meaning that there exists ỹ0 ∈ D𝑟 such that p( 𝑓 ( ỹ0 , 𝑧 0 )) = 0, or, equivalently,
p Q( ỹ0 ), 𝛾 -1 (𝑧 0 -ỹ0 ) -ṽ( Q( ỹ0 )) = 0 (E.43)
This contradicts Proposition E.1 for K ← K 𝑟 . Finally, since we integrating over a set with positive measure, our result follows.
this section cite: ['b19']

Section: ■

this section cite: []

Section: F Further numerical results and details
In this section, we present some additional numerical simulations to illustrate and validate our theoretical findings. To this end, we consider two simple yet representative examples: (i ) a strongly monotone two-player min-max game on the unit square; and (ii) a finite zero-sum game (as an example of a null-monotone game).
this section cite: []

Section: Strongly monotone games.
We consider the strongly monotone two-player min-max game defined by 𝑓 : [0, 1] × [0, 1] → ℝ with 𝑓 (𝑥 1 , 𝑥 2 ) = -(𝑥 1 -0.5) 2 + 0.5𝑥 1 𝑥 2 + 2(𝑥 2 -0.5) 2 (F.1) and entropic regularization. To be more precise, the payoff functions of the two players are given by 𝑢 1 (𝑥 1 , 𝑥 2 ) = 𝑓 (𝑥 1 , 𝑥 2 ) = -𝑢 2 (𝑥 1 , 𝑥 2 ), and 𝑥 * = (20/33, 14/33) is the unique Nash equilibrium point. Fig. 2 demonstrates the behavior of (FTRL) under varying step sizes and noise levels for the min-max game defined by the function 𝑓 (𝑥 1 , 𝑥 2 ). Specifically, we consider step-sizes 𝛾 ∈ {0.1, 0.5}, and stochastic feedback of the form v𝑡 = 𝑣(𝑋 𝑡 ) + 𝜎𝜔 𝑡 , where 𝜔 ∼ 𝑁 (0, 𝐼 2 ) for 𝜎 ∈ {0.5, 1}. For each (𝛾, 𝜎) configuration, we perform 10 5 independent trials, each running for 10 2 steps. The initial state 𝑌 0 for each trial was drawn uniformly at random from [0, 1] 2 . Each surface represents the empirical density of the final (FTRL) iterates, while the color overlay visualizes their distribution across the 10 5 independent trials. Warmer (red) regions indicate higher concentration of final iterates, whereas cooler (blue) regions correspond to lower probability of ending in those regions, as indicated by the colorbar on the side. We observe that smaller step sizes and lower noise levels lead to a tighter concentration of the final iterates around the Nash equilibrium. In contrast, increasing either the step size or the noise variance results in a more dispersed distribution. This behavior aligns with both intuition and our theoretical findings: higher noise introduces greater stochastic variability, while larger step sizes amplify this effect by inducing more aggressive updates that are prone to overshooting, ultimately increasing the spread of the iterates.
To further explore the behavior of (FTRL) under different noise levels and step sizes, we conduct an additional set of experiments summarized in Figs. 3 and 4. These figures illustrate the distance from 𝑥 * of the final iterate and the hitting time in a neighborhood of 𝑥 * with varying radii. Specifically, we consider step sizes 𝛾 ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.5} and stochastic feedback of the form v𝑡 = 𝑣(𝑋 𝑡 ) + 𝜎𝜔 𝑡 for noise levels 𝜎 ∈ {0.01, 0.05, 0.1, 0.5, 1}. For each (𝛾, 𝜎) configuration, we perform 100 independent runs, each consisting of 10,000 iterations. The initial state 𝑌 0 in each run is drawn uniformly at random from [0, 1] 2 . The first plot reports the average final distance of the iterates from the equilibrium, averaged across the 100 runs, while the subsequent plots show the hitting time required for the iterates to enter a neighborhood of the equilibrium of radius 𝑟 ∈ {0.005, 0.01, 0.05, 0.1}. Null-monotone games. Fig. 5 shows the empirical distribution of the final iterates under the (FTRL) dynamics in the classic matching pennies game with entropic regularization, played over the probability simplex with payoff matrix 𝑃 = (+1, -1) (-1, +1) (-1, +1) (+1, -1) .
The unique Nash equilibrium of the game is the mixed strategy (0.5, 0.5) for both players. As before, we consider stochastic feedback of the form v𝑡 = 𝑣(𝑋 𝑡 ) + 𝜎𝜔 𝑡 , where 𝜔 ∼ 𝑁 (0, 𝐼 2 ) for 𝛾 ∈ {0.1, 0.2} and 𝜎 ∈ {1, 2}. For each (𝛾, 𝜎) configuration, we perform 10 5 independent trials, each running for 10 2 steps. Each surface plot corresponds to a different combination of step-size and noise variance,  with the empirical density of the final iterates represented through both height and color over the simplex domain. We see that across all configurations, the iterates tend to concentrate near the corners of the simplex, reflecting the instability of the interior equilibrium in the presence of noise. This consistent shift toward extreme points highlights the system's inherent tendency to escape the central equilibrium under stochastic perturbations.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: It can be found in Appendix F. Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: No statistical significance applicable.
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
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: It can be found in Section 3, Section 4 and the appendix. Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: It can be found in Section 1.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: It can be found in Section 3, Section 4 and the appendix. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: It can be found in Appendix F, and the code is included in the supplemental material. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: 5.
Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code is included in the supplemental material. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/guides/  CodeSubmissionPolicy) for more details.
Answer: [Yes] Justification: It can be found in Appendix F. Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper). Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [NA] Justification: There is no societal impact. Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: There are no such risks. Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: The paper does not use existing assets. Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets. Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Gambling in a rigged casino: The adversarial multi-armed bandit problem Year: (1995)
Ref_id:b1 Title: The nonstochastic multiarmed bandit problem Year: (2002)
Ref_id:b2 Title: What is the long-run distribution of stochastic gradient descent? A large deviations analysis Year: (2024)
Ref_id:b3 Title: The global convergence time of stochastic gradient descent in non-convex landscapes: Sharp bounds via large deviations Year: (2025)
Ref_id:b4 Title: Power allocation games for MIMO multiple access channels with coordination Year: (2009-06)
Ref_id:b5 Title: Dynamics of stochastic approximation algorithms Year: (1999)
Ref_id:b6 Title: Convex optimization algorithms Year: (2015)
Ref_id:b7 Title: Criteria for recurrence and existence of invariant measures for multidimensional diffusions Year: (1978)
Ref_id:b8 Title: Stochastic Approximation: A Dynamical Systems Viewpoint Year: (2008)
Ref_id:b9 Title: On the robustness of learning in games with stochastically perturbed payoff observations Year: (2017-05)
Ref_id:b10 Title: Stochastic replicator dynamics Year: (2000-05)
Ref_id:b11 Title: The impact of uncertainty on regularized learning in games Year: (2025)
Ref_id:b12 Title: Prediction, Learning, and Games Year: (2006)
Ref_id:b13 Title: Convergence analysis of a proximal-like minimization algorithm using Bregman functions Year: (1993-08)
Ref_id:b14 Title: DeepSeek-AI and many other authors that you can see in the arxiv link below Year: (2025)
Ref_id:b15 Title: Bridging the gap between constant step size stochastic gradient descent and markov chains Year: (2017)
Ref_id:b16 Title: Markov chains. Operation research and financial engineering Year: ()
Ref_id:b17 Title:  Year: (2018)
Ref_id:b18 Title: A stochastic variant of replicator dynamics in zero-sum games and its invariant measures Year: (2023-12)
Ref_id:b19 Title:  Year: (1999)
Ref_id:b20 Title: Stochastic evolutionary game dynamics Year: (1990)
Ref_id:b21 Title: Evolutionary dynamics with aggregate shocks Year: (1992-08)
Ref_id:b22 Title: Escaping from saddle points -Online stochastic gradient for tensor decomposition Year: (2015)
Ref_id:b23 Title:  Year: (2003)
Ref_id:b24 Title: Fundamentals of Convex Analysis Year: (2001)
Ref_id:b25 Title: Time averages, recurrence and transience in the stochastic replicator dynamics Year: (2009)
Ref_id:b26 Title: Adaptive learning in continuous games: Optimal regret bounds and convergence to Nash equilibrium Year: (2021)
Ref_id:b27 Title: New first-order algorithms for stochastic variational inequalities Year: (2022)
Ref_id:b28 Title: The long-run behavior of the stochastic replicator dynamics Year: (2005)
Ref_id:b29 Title: Stochastic integral Year: (1944)
Ref_id:b30 Title: Foundations of modern probability. Probability and its Applications Year: (2002)
Ref_id:b31 Title: Brownian Motion and Stochastic Calculus Year: (1998)
Ref_id:b32 Title: Stochastic Stability of Differential Equations. Number 66 in Stochastic Modelling and Applied Probability Year: (2012)
Ref_id:b33 Title: Continuous and discrete dynamics for online learning and convex optimization Year: (2016)
Ref_id:b34 Title: Acceleration and averaging in stochastic descent dynamics Year: (2017)
Ref_id:b35 Title: Online learning of Nash equilibria in congestion games Year: (2015)
Ref_id:b36 Title: Introduction to Stochastic Integration Year: (2006)
Ref_id:b37 Title: Stochastic Approximation Methods for Constrained and Unconstrained Systems Year: (1978)
Ref_id:b38 Title: Bandit Algorithms Year: (2020)
Ref_id:b39 Title: Continuous control with deep reinforcement learning Year: (2016-05-02)
Ref_id:b40 Title: The weighted majority algorithm Year: (1994)
Ref_id:b41 Title: Stochastic gradient descent-ascent and consensus optimization for smooth games: Convergence analysis under expected co-coercivity Year: (2021)
Ref_id:b42 Title: Quantal response equilibria for normal form games Year: (1995)
Ref_id:b43 Title: Learning in the presence of noise Year: (2009)
Ref_id:b44 Title: The emergence of rational behavior in the presence of stochastic perturbations Year: (2010-07)
Ref_id:b45 Title: Learning in games via reinforcement and regularization Year: (2016-11)
Ref_id:b46 Title: Convergence to Nash equilibrium in continuous games with noisy first-order feedback Year: (2017)
Ref_id:b47 Title: On the convergence of gradient-like flows with noisy gradient input Year: (2018-01)
Ref_id:b48 Title: Stochastic mirror descent dynamics and their convergence in monotone variational inequalities Year: (2018-12)
Ref_id:b49 Title: Imitation dynamics with payoff shocks Year: (2016-03)
Ref_id:b50 Title: Learning in games with continuous action sets and unknown payoff functions Year: (2019-01)
Ref_id:b51 Title: Distributed stochastic optimization via matrix exponential learning Year: (2017-05)
Ref_id:b52 Title: Cycles in adversarial regularized learning Year: (2018)
Ref_id:b53 Title: Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile Year: (2019)
Ref_id:b54 Title: A unified stochastic approximation framework for learning in games Year: (2024-01)
Ref_id:b55 Title: Markov Chains and Stochastic Stability. Cambridge Mathematical Library Year: (2009)
Ref_id:b56 Title: An introduction to Malliavin calculus and some of its applications Year: (1990)
Ref_id:b57 Title: Primal-dual subgradient methods for convex problems Year: (2009)
Ref_id:b58 Title: Stochastic Differential Equations Year: (2013)
Ref_id:b59 Title: Optimization despite chaos: Convex relaxations to complex limit sets via Poincaré recurrence Year: (2014)
Ref_id:b60 Title: Convex Analysis Year: (1970)
Ref_id:b61 Title: Variational Analysis Year: (1998)
Ref_id:b62 Title: Existence and uniqueness of equilibrium points for concave 𝑁-person games Year: (1965)
Ref_id:b63 Title: Convex optimization, game theory, and variational inequality theory in multiuser communication systems Year: (2010-05)
Ref_id:b64 Title: Online learning and online convex optimization Year: (2011)
Ref_id:b65 Title: Convex repeated games and Fenchel duality Year: (2006)
Ref_id:b66 Title: Capacity of multi-antenna Gaussian channels Year: (1999)
Ref_id:b67 Title: Stability and perfection of Nash equilibria Year: (1987)
Ref_id:b68 Title: Stochastic methods in variational inequalities: Ergodicity, bias and refinements Year: (2024-05)
Ref_id:b69 Title: Aggregating strategies Year: (1990)
Ref_id:b70 Title: Bayesian learning via stochastic gradient langevin dynamics Year: (2011)
Ref_id:b71 Title: An analysis of constant step size sgd in the non-convex regime: asymptotic normality and bias. NIPS '21 Year: (2021)
Ref_id:b72 Title: Iterative water-filling for Gaussian vector multiple-access channels Year: (2004)
Ref_id:b73 Title: On the convergence of mirror descent beyond stochastic convex programming Year: (2020)
Ref_id:b74 Title: Online convex programming and generalized infinitesimal gradient ascent Year: (2003)
