Title: Transfer Faster, Price Smarter: Minimax Dynamic Pricing under Cross-Market Preference Shift
Abstract: We study contextual dynamic pricing when a target market can leverage K auxiliary markets-offline logs or concurrent streams-whose mean utilities differ by a structured preference shift. We propose Cross-Market Transfer Dynamic Pricing (CM-TDP), the first algorithm that provably handles such model-shift transfer and delivers minimax-optimal regret for both linear and nonparametric utility models. For linear utilities of dimension d, where the difference between source-and targettask coefficients is s 0 -sparse, CM-TDP attains regret O (dK -1 + s 0 ) log T . For nonlinear demand residing in a reproducing kernel Hilbert space with effective dimension α, complexity β and task-similarity parameter H, the regret becomes 2α+1) , matching informationtheoretic lower bounds up to logarithmic factors. The RKHS bound is the first of its kind for transfer pricing and is of independent interest. Extensive simulations show up to 50% lower cumulative regret and 5× faster learning relative to single-market pricing baselines. By bridging transfer learning, robust aggregation, and revenue optimization, CM-TDP moves toward pricing systems that transfer faster, price smarter.

Section: Introduction
Dynamic pricing is now a core operational tool for ride-sharing platforms, airlines, and large ecommerce retailers. State-of-the-art single-market algorithms learn a demand model from scratch and achieve minimax regret when sufficient data accumulate [19,28,9]. In practice, however, many markets launch with only dozens of transactions per day, while mature markets of the same firm collect data at orders-of-magnitude higher rates. Transferring information from data-rich to data-poor markets is therefore essential for fast revenue convergence and early-stage pricing accuracy. Industry practice gives rise to two distinct transfer regimes. First, in the Offline-to-Online (O2O off ) setting, the firm holds a fixed log of source-market data gathered before the target market opens, and this static information is used once the target goes live. Second, in the Online-to-Online (O2O on ) setting, the source and target markets operate concurrently; streaming data from large markets must be incorporated into the pricing decisions of small markets in real time.
Existing transfer approaches do not fully address these settings. Meta-dynamic pricing [4] learns a shared Bayesian prior but requires directly observed linear demands and only exploits offline data. TLDP [30] handles covariate (domain) shift from a single offline source but assumes that the reward model is identical across markets. Bandit-transfer methods focus on either covariate shift [5] or sparse parameter heterogeneity [32,18], yet they do not incorporate revenue-maximising price choice.
this section cite: ['b18', 'b27', 'b8', 'b3', 'b29', 'b4', 'b31', 'b17']

Section: This paper.
We propose CM-TDP (Cross-Market Transfer Dynamic Pricing), a unified framework that (i) operates in both O2O off and O2O on regimes, (ii) accommodates linear and RKHS-smooth nonparametric utilities, and (iii) allows multiple source markets whose mean utilities differ from the target by a structured utility model shift. CM-TDP alternates a bias-corrected aggregation step with an optimistic pricing rule, thereby transferring knowledge while balancing exploration and exploitation.
To the best of our knowledge, this work establishes the first rigorous regret analysis for transfer learning under general utility discrepancies between markets. A primary difficulty that arose during our analysis was maintaining tight control of error propagation: conventional techniques would accumulate slack and inflate constant-order terms into non-negligible O(T c ) factors, rendering the bounds both theoretically and practically uninformative. Our main contributions are as follows:
(C1) Unified transfer pricing framework under utility shifts. CM-TDP is the first dynamic pricing framework that allows multiple source markets whose utilities differ from the target by a structured shift, working in both O2O off and O2O on regimes.
(C2) Minimax-optimal guarantees for two utility classes. We prove (i) O d K log T + s 0 log T regret under linear mean utilities and (ii) the first transfer-pricing bound O K
-2αβ 2αβ+1 T 1 2αβ+1 + H 2 2α+1 T 1 2α+1
for RKHS-smooth utilities-matching known lower bounds.
(C3) Bias-corrected aggregation architecture. Our two-step aggregate → debias pipeline cleanly connects meta-learning (prior pooling), robust statistics (trimmed debiasing), and exploration-driven bandits, and can plug in MLE, Lasso, or kernel ridge as well as black boxes.
(C4) Large empirical gains. Simulations show up to 50 % lower cumulative regret, 28 % lower standard error and 5× faster learning relative to single-market pricing [19], with the largest gains in data-scarce targets under O2O on transfer.
Organization. Section 3 introduces the multi-market dynamic pricing problem with transfer learning under random utility models. Sections 4-6 present CM-TDP and its theoretical analysis. Section 7 reports empirical results, and Section 8 outlines future work.
this section cite: ['b18']

Section: Related Work
Single-market contextual pricing. Early algorithms assume a deterministic valuation map, typically linear, and achieve sub-linear regret [2,13,20], with nonparametric variants studied in [22]. The modern benchmark is the random utility model in which valuation equals a covariate-dependent mean plus i.i.d. noise. When the noise distribution is known, [19] establish the first regret bounds; subsequent work removes that knowledge via doubly robust or moment-matching estimators while retaining linearity [31,21,16,33]. [28] close the gap to the information-theoretic optimum for linear utilities and extend the analysis to Hölder-smooth demand curves, whereas [9] give fully nonparametric guarantees. All of these methods relearn from scratch in every market, degrading performance when target data are limited.
Transfer and meta-learning for pricing. Meta Dynamic Pricing pools directly observed linear demands across products and learns a shared Gaussian prior [4]. TLDP transfers under pure covariate (domain) shift but only from a single offline source [30]. Our Cross-Market Transfer Dynamic Pricing (CM-TDP) differs by coping with utility-model shift, supporting multiple online/offline sources, and providing guarantees for both linear and RKHS utilities.
Multitask contextual bandits and reinforcement learning. Domain-shift transfer for bandits is analysed in [5], whereas sparse heterogeneity is addressed via trimmed-mean/LASSO debiasing [32] or weighted-median MOLAR [18]. Causal-transport ideas reveal negative-transfer risks [15]. Rewardand transition-level transfer and meta learning in RL is explored by [12,10,7,8,34]. We adapt these bias-correction techniques to revenue maximisation under shifting utilities.
Fully online meta-learning without task boundaries. FOML [23] and Online-within-Online metalearning [14] operate on a single stream of data without explicit task resets. CM-TDP follows the same streaming paradigm but must balance exploration and exploitation through posted prices rather than prediction losses.
Positioning of this work. CM-TDP is the first dynamic pricing framework that (i) transfers across multiple auxiliary markets under utility-model shift, (ii) achieves minimax regret for both linear and RKHS utilities, and (iii) unifies bias-corrected aggregation with revenue-maximising price selection, thereby bridging single-market pricing [28], offline meta-priors [4,11], and multitask bandits [18].
Key distinctions from prior work. Unlike Meta-DP [4] and TLDP [30], CM-TDP (i) handles concurrent source streams, (ii) tolerates utility shift rather than merely covariate shift, and (iii) supplies the first nonparametric (RKHS) transfer-pricing regret bound. Multitask-bandit methods such as MOLAR [18] focus on prediction error and linear bandits, do not optimize posted prices, and therefore cannot exploit revenue structure. Consequently, existing approaches cannot deliver the minimax-optimal guarantees or empirical gains demonstrated by CM-TDP.
this section cite: ['b1', 'b12', 'b19', 'b21', 'b18', 'b30', 'b20', 'b15', 'b32', 'b27', 'b8', 'b3', 'b29', 'b4', 'b31', 'b17', 'b14', 'b11', 'b9', 'b6', 'b7', 'b33', 'b22', 'b13', 'b27', 'b3', 'b10', 'b17', 'b3', 'b29', 'b17']

Section: Problem Formulation
We consider a pricing model for the target market where products are sold one at a time, and only a binary response indicating success or failure of a sale is observed. For each decision point t ∈ [T ], the market value of the product at time t depends on the observed contextual information x (0) t . A general random utility model for the market value of the product is given by
v (0) t = g(0) (x (0) t ) + ε t ,(1)
where g(0) (•) ∈ G is the unknown function of the mean utility in the target market, and ε t are i.i.d. noises following an known distribution F (•) with E[ε t ] = 0 and support
S ε := [-B ε , B ε ].
Given a posted price of p t for the product at time t, we observe y
Therefore, given a posted price p t , the expected revenue from the target market at time t conditioned on x (0) t is rev
t (p t ) := p t • 1 -F (p t -g(0) (x (0) t )) . The oracle optimal offered price p * t is defined by p * (0) t = arg max
pt≥0 p t [1 -F (p t -g(0) (x (0) t ))],(3)
and hence, under Assumption 1, we have p * (0) t = h • g(0) (x t ), where h(u) = u + φ -1 (-u) and φ(u) = u -1 -F (u) F (u) . (4)
this section cite: []

Section: Assumption 1 (Regularity condition).
There exists positive constants L φ and B u such that φ ≥ L φ for all u ∈ [-B u , B u ], and inf |u|≤Bu φ (u) ≥ 1.
Assumption 1 guarantees the uniqueness of the optimal solution of (3). The restriction of L φ ≥ 1 is commonly used in dynamic pricing study [19].
this section cite: ['b18']

Section: Optimal policy and regret.
For a policy π that sets price p t at t, its regret over the time horizon of T is defined as
Regret(T ; π) = T t=1 E [p * t 1(v t ≥ p * t ) -p t 1(v t ≥ p t )] ≡ T t=1
[rev t (p * t ) -rev t (p t )] .
The goal of a decision maker is to design a pricing policy that minimizes Regret(T ; π), or equivalently, maximize the collected expected revenue T t=1 rev t (p t ).
this section cite: []

Section: Cross-Market Transfer Learning.
In the context of cross-market transfer learning, we observe additional samples from K sources markets indexed by superscript (k) for k ∈ t } t≥1 are drawn i.i.d. from a fixed, but a priori unknown, distribution P x , supported on a bounded set X ⊂ R d . Let Σ = E[x t x t ] denote the second moment matrix of P x . We assume:
(1) Eigenvalue boundedness: The minimum and maximum eigenvalues of Σ, denoted C min and C max , satisfy 0 < C min ≤ C max < ∞.
(2) Non-degeneracy: The distribution P x has a density bounded away from zero in a neighborhood of the origin, ensuring Σ is positive definite. Remark 1. Assumption 2 isolates market differences to utility model shifts, which is reasonable when source and target markets have similar populations but differ in preferences. The key conditions ensure: (i) Stability: bounded eigenvalues and support guarantee well-behaved estimators. (ii) Identifiability: a density bounded away from zero near the origin ensures Σ is positive definite. These hold in many practical settings e.g., truncated uniform or Gaussian distributions.
We will study the estimation and decision for the target model (1) leveraging the data from the target markets as well as the data from K auxiliary source markets.
this section cite: []

Section: Similarity Characterization.
Transfer is effective only when source and target markets are sufficiently alike; we formalize this by assuming their mean-utility functions lie in a common hypothesis class G and differ only through a structured utility shift. For any candidate mean-utility function g ∈ G we distinguish two notions of similarity, corresponding to (i) linear (parametric) and (ii) RKHS (nonparametric) utility models:
• Parametric classes. When every g is indexed by a finite-dimensional parameter vector, similarity is expressed as a bound on the parameter gap between source and target.
• Nonparametric classes. For infinite-dimensional G we impose a bound on the functional discrepancy between source and target under a suitable function-space metric.
Later assumptions specialize these high-level conditions (e.g. sparse parameter differences in Assumption 4 for linear utilities and smooth residuals for RKHS utilities in Assumption 8 for nonparametric utilities). This formulation places every source task in a recoverable neighbourhood of the target, ensuring its data are informative for transfer across both linear and non-linear utility models.
this section cite: []

Section: Cross-Market Transfer Dynamic Pricing Algorithms
We address both practical data scenarios introduced in Section 1. For O2O off (Offline-to-Online), a large, static source log is available prior to launch. The algorithm transfers that log during the early episodes-those for which the cumulative target sample size is below a theory-driven threshold τ . Once |T m | ≥ τ (source data no longer dominate the information budget) the procedure switches automatically to pure single-market learning. Hence transfer is phased, not one-shot: it is used exactly while it provably reduces estimation error and is dropped thereafter. Full details and guarantees are given in Appendix B.
For O2O on (Online-to-Online), source and target markets operate concurrently; transfer is repeated at the start of every episode, ensuring that incoming source data continuously guide the target-market prices. Let the time horizon be partitioned into episodes m = 1, 2, . . . , M with lengths ℓ m = 2 m-1 (so that M m=1 ℓ m ≈ T and M = log 2 T ). At the start of each episode the algorithm (i) fits or debiases a demand estimator using all data collected in the preceding episode, and (ii) fixes the resulting pricing rule for the next ℓ m periods. Because episode length doubles, parameter updates occur only O(log T ) times, yet the cumulative sample size entering each update grows geometrically, guaranteeing progressively tighter confidence bounds and the desired O(polylog T ) regret. Both algorithms share a common two-step bias-corrected aggregation pipeline and are instantiated for (i) linear utilities, with maximum-likelihood estimation (MLE), and (ii) RKHS utilities, with kernel logistic regression (KLR). Pseudocode is given in Algorithms 4 (O2O off ) and 1 (O2O on ); estimation details appear in Algorithms 2 and 3.
this section cite: []

Section: Comparison between O2O off and O2O on .
When source streams remain active, each episode m recomputes an aggregate estimate g(ag) m from the preceding episode's source data and debiases it using the matching target observations to form g(0) m . This persistent adaptation leads to provably faster regret decay compared to O2O off . In contrast, O2O off employs phased transfer only during initial episodes, resulting in asymptotic regret growth rates that eventually match single-market learning, though with improved constants during the transfer period.
Empirically, O2O on achieves flatter regret trajectories with consistently lower cumulative regret (Figures 1 and 2). O2O off shows parallel regret growth to single-market baseline in later stages (Figures 3 and 4), confirming our theoretical analysis. However, we still observe a jump-start benefit in O2O off during early stages, ultimately translating to significantly lower overall regret compared to the single-market baseline.
this section cite: []

Section: Algorithm 1: CM-TDP-O2O on
Input: Streaming source data{(p
(k) t , x (k) t , y (k) t )} t≥1 for k ∈ [K]; streaming target contexts {x (0) t } t≥1 1 Initialisation: ℓ 1 ← 1, T 1 = {1}, g(0) 0 = 0. 2 for m = 1, 2, . . . do // episodes 3 Compute ℓ m = 2 m-1 , T m = {ℓ m , . . . , ℓ m+1 -1}. // (i) aggregate previous episode's source data 4 g(ag) m ← MLE_or_KRR {(p (k) t , x (k) t , y (k) t )} t∈Tm-1,k∈[K] .
// (ii) debias with previous episode's target data
5 δ m ← Debias g(ag) m , {(p (0) t , x(0)
t , y
t )} t∈Tm-1 .
// For functions MLE_or_KRR and Debias, call Algorithm 2 for linear utility (or Algorithm 3 for nonparametric utility)
6 Set g(0) m ← g(ag) m + δ m . 7 for t ∈ T m do // pricing 8 Post price p (0) t = h g(0) m (x (0) t ) ; observe y (0) t and store data.
this section cite: []

Section: Parametric Utility Models: Similarity, Transfer, and Guarantee
We start with the linear setting that allows us to isolate and rigorously characterize the transfer mechanism itself, before introducing the additional complexity of nonlinear effects.
this section cite: []

Section: Linear Utility Models.
Consider a linear model for the mean utility:
v (0) t = x (0) t • β (0) + ε t ,(5)
where β (0) ∈ R d denotes the coefficient vector. For source market data, we have for k ∈
[K], v (k) t = x (k) t • β (k) + ε t .
To simplify the presentation, we impose the following assumption on parameter space.
Assumption 3 (Parameter Boundedness). We assume that ||x t || ∞ ≤ 1, ∀x t ∈ X , and
||β (k) || 1 ≤ W for a known constant W ≥ 1, ∀k ∈ 0 ∪ [K].
We denote by Ω the set of feasible parameters, i.e.,
Ω = β ∈ R d+1 : ||β|| 1 ≤ W , β (k) ∈ Ω, ∀k ∈ 0 ∪ [K].
We formalize the notion of similarity between source and target markets using the sparsity of the difference between coefficients.
Assumption 4 (Task Similarity in Linear Model). The maximum l 0 -norm of the difference between target and source coefficients is bounded:
max k∈[K] β (0) -β (k) 0 ≤ s 0 .
While our linear utility model itself is not necessarily sparse, Assumption 4 specifically constrains the cross-market parameter differences to be sparse, implying that at most s 0 covariates have significantly different effects across markets, mirroring the economic intuition that only certain latent features drive market variations.
Bias-corrected Aggregation for Linear Utility. Algorithm 2 serves as the dual-mode estimator in Algorithms 1 and 4, switching between: (i) source data aggregation (no prior input), or (ii) ℓ 1 -regularized debiasing (given aggregate estimate).
this section cite: []

Section: Algorithm 2: Maximum Likelihood Estimation for Linear Utility Model
Input: Data {(p t , x t , y t )} t∈[n] , aggregate estimate β (ag)
1 if β (ag)
is None then // compute aggregate term
2 β = argmin b 1 n n t=1 L(b; p t , x t , y t )) 3 else // compute debiasing term 4 β = argmin b 1 n n t=1 L(b + β(ag)
; p t , x t , y t )) + λ tf b 1 5 where the function
L(b; p, x, y) := -1(y = 1) log(1 -F (p -b • x)) + 1(y = 0) log(F (p -b • x)) . (6
)
Output: β
In Algorithm 2, the aggregation step employs unregularized MLE. This choice is motivated by the fact that source market data are typically abundant, so the aggregate estimate can be reliably learned without imposing sparsity or other high-dimensional penalties. By contrast, the debiasing step operates on target market samples, which are relatively scarce. Here, we incorporate regularization to stabilize estimation and exploit structured similarities across markets.
this section cite: []

Section: Theoretical Guarantee for CM-TDP-O2O on under Linear Utility
The following theorem bounds the regret of our O2O on Policy under linear utility model. Theorem 5 (Regret Upper Bound for O2O on under Linear Utility). Consider linear utility model (5) with Assumptions 1 (revenue regularity), 2 (covariate property), 3 (parameter space), and 4 (market similarity) holding true, the cumulative regret of Algorithm 1 admits the following bound:
Regret(T ; π) = O d K log d log T + s 0 log d log T .(7)
where K and T denote the number of source markets and time horizon, respectively.
We defer the complete proof to Appendix E. Theorem 5 reveals crucial insights about the role of source market quantity K in two distinct operational regimes. First, in the source-constrained regime (K d/s 0 ), the first term dominates, showing that each additional source market provides linear reduction in regret. Notably, the logarithmic dependence on T is consistent with classical linear bandit results [1], though our bound strictly improves their O(d log T ) through transfer. Second, in the source-saturation regime (K d/s 0 ), the second term becomes pivotal, quantifying the price for cross-market heterogeneity.
As illustrated in Figure 1, the empirical scaling behavior of regret w.r.t. both the number of source markets K and time horizon T precisely matches the theoretical predictions derived from Theorem 5.
While our theoretical guarantee is established in asymptotic regimes, the finite-horizon empirical performance robustly validates the practical effectiveness. This alignment between theory and practice confirms that our asymptotic analysis yields operationally meaningful insights for real-world applications. Theorem 6 (Regret Lower Bound under Linear Utility). Consider the linear utility model in (5), for any pricing policy π, the worst-case target-market regret over horizon T satisfies
inf π sup Regret(T ; π) ≥ c 1 d K log T + c 2 s 0 log d s 0 log T,(8)
where the two constants c 1 , c 2 depends only on noise distribution F , second moment matrix Σ, and parameter space W .
In particular, (8) matches the upper bound (7) up to polylogarithmic factors in d, hence CM-TDP is minimax-type optimal in its T -and K-scaling.
6 Nonparametric Utility Models: Similarity, Transfer, and Guarantee
In this section, we model market utilities in an RKHS [3], leveraging (i) the kernel trick for efficient nonlinear computation [24], and (ii) its universal approximation power to capture rich market responses [26].
this section cite: ['b0', 'b6', 'b2', 'b23', 'b25']

Section: Nonparametric Utility Models.
Let H k be an RKHS induced by a symmetric, positive and semidefinite kernel function K : X × X → R, and we define its equipped norm as • 2 K = •, • K with the endowed inner product •, • K . We also define K x := K(x, •) ∈ H k . An important property in H k is called the reproducing property, stating that g, K x K = g(x). The utility function now follows:
v (k) t = g (k) (x (k) t ) + ε t , g (k) ∈ H k , k ∈ 0 ∪ [K],(9)
where g (k) is the unknown target function and {ε
t } t≥1 are i.i.d. noise with known distribution F . To start with, we place the following regularity assumptions on H k , which requires the model to be well-specified, and the kernel to be bounded.
this section cite: []

Section: Assumption 7 (Regularity Condition). Assume that g (0)
H k ≤ R, for some R > 0, and there exists a positive constant κ, such that the feature map φ(x) = K(x, •) satisfies φ(x) H k ≤ κ, ∀x ∈ X . Assumption 8 (Task Similarity in RKHS). For all k ∈ [K], the discrepancy between the target task g (0) and the k-th source task g (k) in the RKHS norm is uniformly bounded as
max k∈[K] g (0) -g (k) K ≤ H.
Remark 2. Assumption 8 characterizes the similarity between the target and source tasks through the bound H. A smaller value of H indicates that the source tasks are more similar to the target task, which enables more effective knowledge transfer and potentially improves the estimation accuracy by leveraging information from related sources.
this section cite: []

Section: Assumption 9 (Complexity).
Define the effective dimension as N (λ) := Tr[Σ(Σ+λI) -1 ], a variant of what is typically used to characterize the complexity of RKHS [6]. We assume:
(i) There exist some constants α > 1/2 such that N (λ) = Tr(Σ(Σ + λI) -1 ) λ -1/(2α) .
(ii) There exist some constants β ∈ (0, 1] such that for each k ∈ 0 ∪ [K], there holds g (k) = Σ β ρ (k) , for some ρ (k) ∈ L 2 (X , P x ), where Σ, P x are defined in Assumption 2. Remark 3. (i) controls the complexity of the considered H k . Smaller α means slower eigenvalue decay and higher intrinsic dimensionality. (ii) is a regularity condition on the source and target functions, and is also commonly assumed in literature [6,25,17]. Here β > 0 controls the degree of smoothness, and larger β means g (k) is smoother and easier to estimate.
this section cite: ['b5', 'b5', 'b24', 'b16']

Section: Bias-corrected Aggregation for Nonparametric Utility.
Algorithm 3 presents the aggregation and debiasing operations using regularized kernel regression.
this section cite: []

Section: Theoretical Guarantee for CM-TDP-O2O on under Nonparametric Utility
The following theorem bounds the regret of our O2O on Policy under RKHS utility model.
this section cite: []

Section: Algorithm 3: Kernel Regression for RKHS Utility Model
Input: Data {(p t , x t , y t )} t∈[n] , Kernel K(x, x ) = φ(x), φ(x ) , Aggregated estimator g (ag) ∈ H k . 1 if g (ag) is None then // compute aggregation term ag) ; p t , x t , y t ) + λ tf g 2 H k 5 where the function L(g; p, x, y) := -1(y = 1) log(1 -F (p -g(x))) + 1(y = 0) log(F (p -g(x))) .
2 g = argmin g∈H k 1 n n t=1 L(g; p t , x t , y t ) + λ ag g 2 H k 3 else // compute debiasing term 4 g = argmin g∈H k n t=1 L(g + g (
this section cite: []

Section: Output: g
Theorem 10 (Regret Upper Bound for O2O on under RKHS Utility). Consider RKHS utility model (9) with Assumptions 1 (revenue regularity), 2 (covariate property), 7 (parameter space), 8 (market similarity), and 9 (complexity) holding true, the cumulative regret of Algorithm 1 admits the following bound:
Regret(T ; π) = O K -2αβ 2αβ+1 T 1 2αβ+1 + H 2 2α+1 T 1 2α+1 (10
)
where K and T denote the number of source markets and time horizon, respectively.
Theorem 10 again highlights the benefit of transfer learning, and finds clear empirical support in the experiments shown in Figure 2. The first term reflects the learning complexity of the target function, where β quantifies its intrinsic complexity with larger β indicating simpler functions and enabling faster transfer. The α parameter, as the kernel's effective dimension, modulates how β impacts the exponent. The second term encodes cross-market disparity through H, which directly measures the worst-case RKHS distance between source and target markets. The α-dependence shows that high-dimensional RKHS amplify heterogeneity costs. Compared to Theorem 5, we observe polynomial rather than logarithmic T -dependence, reflecting the fundamental difficulty shift from parametric to nonparametric estimation.
this section cite: ['b8']

Section: Special Cases.
The following boundary cases demonstrate the degradation properties of our theoretical results, showing how the general bound naturally adapts to different simpler scenarios.
Perfect Task Similarity. When source and target domains are identical (H = 0), the bound becomes:
Regret(T ; π) = O K -2αβ 2αβ+1 T 1 2αβ+1
The regret depends on the total sample size from all sources. The rate improves with number of source market K.
No Transfer Learning (K eff = 1, H = 0) In the absence of transfer learning, i.e., when no source domain data is available (K = 0, H = 0), we evaluate the bound (10) with K eff = max{K, 1} = 1 (equivalently, a "self-aggregation" that ignores external sources). Our general framework reduces to conventional dynamic pricing with RKHS utility functions. The regret bound simplifies to:
Regret(T ; π) = O T 1 2αβ+1 .
While existing literature has not explicitly analyzed regret bounds for dynamic pricing with RKHS utility functions, we can establish an important connection to the well-studied linear case. By setting α → 1/2 + and β = 1, which corresponds to linear utility functions, the general bound (10) reduces to O( √ T ), matching the often-seen regret for online decision-making problems with linear structures [1,19]. This O( √ T ) rate should be contrasted with the O(log T ) regret we established for the linear transfer setting in (7). The difference stems from the underlying estimation complexity: in the parametric case, the generalized linear model is finite-dimensional, and abundant source data together with MLE-based updates enable nearly logarithmic regret growth. In contrast, the RKHS formulation must estimate an infinite-dimensional function under binary feedback, where nonparametric learning is intrinsically harder. Thus, the gap reflects fundamental differences between parametric and nonparametric estimation, rather than looseness in the analysis. Theorem 11 (Regret Lower Bound under RKHS Utility). Consider the RKHS utility model in (9). For any pricing policy π, then there exists a constant c > 0 depending only on (F, P x , K) via (m rev , m h , κ) and the Bernoulli KL smoothness constant (defined in Lemma 28) such that for all horizons T ≥ 1,
inf π sup Regret(T ; π) ≥ c K -2αβ 2αβ+1 T 1 2αβ+1 + H 2 2α+1 T 1 2α+1 .(11)
In particular, (11) matches the upper bound (10) up to polylogarithmic factors, hence CM-TDP is minimax-type optimal in its T -and K-scaling.
this section cite: ['b9', 'b0', 'b18', 'b6', 'b8', 'b9']

Section: Numerical Experiments
We evaluate CM-TDP through extensive simulations covering multiple market scenarios and dimensionalities:
(1) Identical Markets: an ideal baseline where
β (0) ≡ β (k) or g (0) ≡ g (k)
for all source markets;
(2) Sparse difference Markets: for linear utility, we implement Assumption 4 with β (0) -β (k) 0 ≤ 0.3 * d; for RKHS utility, we implement Assumption 8 with g (0) -g (k)
H k ≤ 0.3.
(3) Dense difference Markets: for linear utility, we implement Assumption 4 with β (0) -β (k) 0 ≤ 0.5 * d; for RKHS utility, we implement Assumption 8 with g (0) -g (k)
H k ≤ 0.5. In each market scenario, we test T = 2000 periods with dimensions d ∈ {10, 15, 20, 100} and K ∈ {1, 3, 5, 10} source markets. RKHS function (RBF kernel with γ = 0.5) and kernel parameters (κ = 0.5, R = 1.0) remain consistent across all experiments. We evaluate O2O on Policy against no transfer baseline [19], a standard dynamic pricing using only target market data. Market noise follows logistic distribution on R. For numerical stability, we clip simulated valuations to [-B ε , B ε ] with B ε = 1. The code is available at https://github.com/CS-SAIL/transfer_pricing_neurips2025.
this section cite: ['b18']

Section: Simulation Results.
Figures 1 and 2 present the empirical cumulative regret for linear and RKHS utility models with d = 10 and 100, respectively, demonstrating three key findings (For conciseness, results for d = 15, 20 are deferred to Appendix D.1).
(1) Universal effectiveness over non-transfer scheme. CM-TDP-O2O on consistently outperforms single-market baseline in all scenarios. Significant improvements emerge even with minimal source markets (K = 1), with larger K values enhance robustness in divergent market conditions. Against single-market learning, CM-TDP reduces cumulative regret by roughly 43-55% on average (peaking at 75%), reduces standard error by about 24-31% (up to 39 %), and attains the same estimation error level as much as 9 × sooner (Table 2 in Appendix D.2.1).
(2) Adaptivity. The transfer mechanism automatically handles both identical markets and sparsedifference markets, without requiring manual adjustments, with identical market condition achieving faster convergence than sparse difference cases and dense difference cases.
(3) Scalability: Higher dimensions maintain stable performance, confirming the method's scalability.
These results collectively validate our framework as a versatile solution for real-world dynamic pricing, particularly in environments with varying market similarities.
this section cite: []

Section: Conclusion and Future Work
We introduce Cross-Market Transfer Dynamic Pricing (CM-TDP), the first framework that provably accelerates revenue learning by pooling information from multiple auxiliary markets in both Offlineto-Online and Online-to-Online regimes. CM-TDP achieves minimax-optimal regret for both linear and RKHS utilities, matching information-theoretic lower bounds, and delivers up to an average of 50% lower cumulative regret and 5× faster learning in extensive simulations. These results bridge single-market pricing, meta-learning, and multitask bandits, laying the groundwork for pricing systems that "transfer faster, price smarter".
(a) Identical, d = 10 (b) Sparse, d = 10 (c) Dense, d = 10 (d) Identical, d = 100 (e) Sparse, d = 100 (f) Dense, d = 100  Several extensions of CM-TDP provide fertile ground for future research. First, relaxing the assumption of homogeneous covariate distributions would allow the framework to handle domain shift, e.g., via reweighting, importance sampling, or domain-invariant representation learning. Second, while we focus on ℓ 0 -sparsity for interpretability, the aggregation framework naturally extends to richer similarity notions such as ℓ q -sparsity (q ∈ [0, 1]), smoothness metrics, or distributional divergences. Third, CM-TDP currently lacks mechanisms to down-weight or exclude adversarial source markets with large parameter gaps, and developing robust similarity detection and market-selection strategies remains an important direction.
this section cite: []

Section: References
Ref_id:b0 Title: Online-to-confidence-set conversions and application to sparse stochastic bandits Year: (2012-04)
Ref_id:b1 Title: Repeated contextual auctions with strategic buyers Year: (2014)
Ref_id:b2 Title: Theory of reproducing kernels Year: (1950)
Ref_id:b3 Title: Meta dynamic pricing: Transfer learning across experiments Year: (2022)
Ref_id:b4 Title: Transfer learning for contextual multi-armed bandits Year: (2024)
Ref_id:b5 Title: Optimal rates for the regularized least-squares algorithm Year: (2007-07)
Ref_id:b6 Title: Deep transfer Q-learning for offline non-stationary reinforcement learning Year: (2025)
Ref_id:b7 Title: Transfer Q-learning with composite MDP structures Year: ()
Ref_id:b8 Title: Dynamic contextual pricing with doubly non-parametric random utility models Year: (2024)
Ref_id:b9 Title: Data-driven knowledge transfer in batch Q learning Year: (2024)
Ref_id:b10 Title: Distributed tensor principal component analysis with data heterogeneity Year: (2025)
Ref_id:b11 Title: Transfer Q-learning for finite-horizon markov decision processes Year: (2025)
Ref_id:b12 Title: Feature-based dynamic pricing Year: (2020)
Ref_id:b13 Title: Online-within-online meta-learning Year: (2019)
Ref_id:b14 Title: Transfer learning in latent contextual bandits with covariate shift through causal transportability Year: (2025)
Ref_id:b15 Title: Policy optimization using semiparametric models for dynamic pricing Year: (2024)
Ref_id:b16 Title: Learning theory of distributed spectral algorithms Year: (2017)
Ref_id:b17 Title: Optimal multitask linear regression and contextual bandits under sparse heterogeneity Year: (2025)
Ref_id:b18 Title: Dynamic pricing in high-dimensions Year: (2019)
Ref_id:b19 Title: Optimal contextual pricing and extensions Year: (2021)
Ref_id:b20 Title: Contextual dynamic pricing with unknown noise: Explore-then-ucb strategy and improved regrets Year: (2022)
Ref_id:b21 Title: Contextual pricing for lipschitz buyers Year: (2018)
Ref_id:b22 Title: Fully online meta-learning without task boundaries Year: (2022)
Ref_id:b23 Title: Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond Year: (2001)
Ref_id:b24 Title: Learning theory estimates via integral operators and their approximations Year: (2007)
Ref_id:b25 Title: On the influence of the kernel on the consistency of support vector machines Year: (2001)
Ref_id:b26 Title: Nonparametric estimators Year: (2008)
Ref_id:b27 Title: Improved algorithms for contextual dynamic pricing Year: (2024)
Ref_id:b28 Title: Introduction to the non-asymptotic analysis of random matrices Year: (2011)
Ref_id:b29 Title: Transfer learning for nonparametric contextual dynamic pricing Year: (2025)
Ref_id:b30 Title: Towards agnostic feature-based dynamic pricing: Linear policies vs linear valuation with unknown noise Year: (2022)
Ref_id:b31 Title: Multitask learning and bandits via robust statistics Year: (2025)
Ref_id:b32 Title: Maximal extractable value in batch auctions Year: (2025)
Ref_id:b33 Title: Prior-aligned meta-rl: Thompson sampling with learned priors and guarantees in finite-horizon mdps Year: (2025)
