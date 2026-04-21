Title: Stronger Neyman Regret Guarantees for Adaptive Experimental Design
Abstract: We study the design of adaptive, sequential experiments for unbiased average treatment effect (ATE) estimation in the design-based potential outcomes setting. Our goal is to develop adaptive designs offering sublinear Neyman regret, meaning their efficiency must approach that of the hindsightoptimal nonadaptive design. Recent work (Dai et al., 2023) introduced ClipOGD, the first method achieving O( √ T ) expected Neyman regret under mild conditions. In this work, we propose adaptive designs with substantially stronger Neyman regret guarantees. In particular, we modify Cli-pOGD to obtain anytime O(log T ) Neyman regret under natural boundedness assumptions. Further, in the setting where experimental units have pretreatment covariates, we introduce and study a class of contextual "multigroup" Neyman regret guarantees: Given any set of possibly overlapping groups based on the covariates, the adaptive design outperforms each group's best non-adaptive designs. In particular, we develop a contextual adaptive design with O( √ T ) anytime multigroup Neyman regret. We empirically validate the proposed designs through an array of experiments.

Section: Introduction
Randomized control trials (RCTs) play a central role in a variety of settings where causal effects need to be accurately measured, spanning healthcare and epidemiology, policymaking, the social sciences, econometrics, e-commerce, and beyond. In the classic potential outcomes framework (Neyman, 1923;Rubin, 1974), a central estimand is the average treatment effect (ATE) -the average individual causal effect across experimental units. To obtain precise estimates of the ATE, we generally seek estimators that are unbiased and have low variance.
In many cases, RCTs are run sequentially: Experimental units arrive one by one, and each unit is assigned to treatment or control adaptively, based on previous outcomes or auxiliary information. The data-driven nature and flexibility of these experiments suggest that such adaptive trials can achieve substantial efficiency gains over standard fixed designs, as shown in domains ranging from political science (Offer-Westort et al., 2021;Blackwell et al., 2022) to medicine (Chow & Chang, 2008;Villar et al., 2015;FDA, 2019). However, so far adaptive experiments have received limited attention (Hu & Rosenberger, 2006) and have been rarely used in practice due to concerns that adaptivity could invalidate standard statistical guarantees (van der Laan, 2008). Indeed, classic solutions for improving estimator efficiency in the batch setting, such as Neyman allocation (Neyman, 1992), can be nontrivial to extend to the sequential setting.
Recently, a growing body of work (Hahn et al., 2011;Kato et al., 2020;Li & Owen, 2024;Dai et al., 2023;Cook et al., 2023) has made progress on this front by introducing multistage adaptive designs that estimate the ATE via inverseprobability weighting (IPW)-type estimators with adaptively adjusted propensity scores. 1 Our work contributes to this literature by developing novel adaptive sequential designs for IPW-based ATE estimation with efficiency guarantees. Crucially, our methods -unlike most existing work-are developed within the finite-population setting (Wager, 2024), where the ATE is defined as a deterministic function of the observed population rather than a superpopulation parameter. This distinction ensures robustness to treatment effect heterogeneity and temporal data drift, challenges that can undermine conventional superpopulation-based designs.
Our Contributions We focus on the design of adaptive RCTs to estimate the ATE as efficiently as the best-inhindsight IPW design from some benchmark class, up to Georgy Noarov conducted part of this work as an intern at Amazon Web Services. 1 In parallel, studies that fall into the multi-armed bandits literature have developed adaptive designs for finding rewardmaximizing treatments (arms) or policies, which is a distinct, and conflicting, objective than estimation efficiency (Zhang et al., 2020;2021;Hadad et al., 2021;Xu et al., 2016;2024). error terms. Specifically, we aim to minimize the Neyman regret (Kato et al., 2020;Dai et al., 2023) -a measure comparing the variance of our adaptive estimator to that of the variance-minimizing nonadaptive Bernoulli trial where units are treated with some fixed probability. Currently, to our knowledge Dai et al. (2023)'s ClipOGD method is the only adaptive design achieving sublinear Neyman regret in the finite-population setting. This method guarantees O( √ T ) expected regret for any T -unit trial under momentbounded potential outcomes. However, two important questions arise: I. Can we develop designs with better regret rates? Dai et al. (2023) conjectured that O( √ T ) is the minimax Neyman rate.
II. Can we develop context-aware designs that use pretreatment covariates to improve efficiency?
In this work, we answer both these questions affirmatively:
Contribution I: Exponentially Improved Noncontextual Neyman Regret Bound. We show that, under a natural strengthening of Dai et al. (2023)'s assumptions on the outcomes, we can modify ClipOGD to attain an anytimevalid Neyman regret bound of O(log T ). 2 To achieve this speedup, we leverage the strong convexity of the Neyman objective under our stricter lower-bounding assumption on the outcomes, which as we show leads to near-logarithmic regret via techniques introduced by (Hazan et al., 2007). Moreover, it can be shown that even under the weaker outcome lower bound assumption of Dai et al. (2023), our adaptive design can be tweaked to have the asymptotic efficiency of (1 + ϵ) V * + O log T T for any ϵ > 0, where V * denotes the optimal nonadaptive design variance; the interpretation is that any (1 + ϵ)-multiplicative approximation to the optimal variance can be attained at this fast rate. We validate the greater efficiency of our proposed design against that of ClipOGD through a suite of experiments on synthetic and real-world data.
Contribution II: Adaptive Designs with Contextual Neyman Regret Guarantees. We next develop a novel adaptive design MGATE (Multi-Group ATE) that leverages pretreatment covariates to improve efficiency relative to the non-contextual setting. In a nutshell, given an arbitrary predefined finite collection G ⊆ 2 X of contextual groups defined by the covariates (e.g., demographics), we propose a no G-multigroup-Neyman-regret adaptive design that obtains sublinear regret simultaneously on all subsequences of experimental units corresponding to the groups in G. Critically, we also allow for overlapping groups, i.e., units can simultaneously belong to multiple groups. A key challenge here is to balance the treatment probabilities in a way that balances the efficiency of the ATEs estimates across groups. Our proposed design leverages a variation of the "sleeping experts" approach (Blum & Lykouris, 2020;Acharya et al., 2024) used in the online learning literature (Lee et al., 2022;Deng et al., 2024), that deals with the limited feedback and the fact that the observed objective values do not live in an a-priori bounded range. The method achieves O( √ T ) multigroup Neyman regret. We also empirically validate its performance.
Our multigroup guarantees can be interpreted through the lens of group ATE (GATE) estimation (Chernozhukov et al., 2017;Semenova & Chernozhukov, 2021;Zimmert & Lechner, 2019). GATE occupies a middle ground between ATE, which measures the average effect over the entire sequence, and CATE (conditional ATE), which measures the ATE conditionally on each covariate vector. Existing works on GATE, however, are mainly focused on learning data-driven disjoint groups to improve overall ATE estimation. In contrast, our objective is to simultaneously ensure efficient GATE inference for any family of arbitrarily overlapping groups. This is related in motivation (though distinct in technique) to the recent work of (Kern et al., 2024) who use "multiaccuracy" to make CATE inference robust to certain kinds of distribution shift.
We expect that such multigroup efficiency guarantees can be broadly useful, and hope future work will study multigroup adaptive designs beyond the sequential finite-population setting that we focus on in this paper.
For an additional discussion of related work, including relevant independent work in the superpopulation setting, please see Appendix A.
this section cite: ['b30', 'b41', 'b33', 'b1', 'b4', 'b47', 'b9', 'b19', 'b46', 'b31', 'b13', 'b22', 'b25', 'b7', 'b6', 'b48', 'b53', 'b12', 'b50', 'b22', 'b7', 'b7', 'b7', 'b7', 'b15', 'b7', 'b2', 'b0', 'b24', 'b8', 'b3', 'b42', 'b23']

Section: Organization
In Section 2, we introduce our general setting and objectives. In Section 3, we focus on the (vanilla) non-contextual setting, and present and analyze our adaptive design ClipOGD SC , which achieves near-logarithmic Neyman regret. We prove the main regret bound in Theorem 3.2 and demonstrate further guarantees on the adaptive design.
In Section 4, we introduce the notion of multigroup Neyman regret, and present our multigroup adaptive design MGATE (Algorithm 2), which achieves O( √ T ) multigroup Neyman regret as shown in Theorem 4.2. Furthermore, in Appendix D we provide a general multigroup design (Algorithm 7) that significantly generalizes MGATE. In Section 5, we compare the empirical performance of our adaptive designs to the Dai et al. (2023) ClipOGD design on an array of real-world and synthetic sequential experimental design tasks.
this section cite: ['b7']

Section: Preliminaries
Setting We work in the design-based, sequential variant of the potential outcomes setting (Neyman, 1923;Rubin, 1974;Imbens & Rubin, 2015). A finite number of experimental units in the population arrive one by one at rounds t ∈ N + . Each unit has two associated fixed potential outcomes, only one of which can be observed: treatment outcome y t (1) ∈ R and control outcome y t (0) ∈ R.
In the basic setting, the observed outcome is the only information the experimenter receives about the units. A richer setting is one where before choosing treatment or control for unit t, the Experimenter is given access to pre-treatment covariate x t ∈ X , where X is a feature space of arbitrary nature (e.g. X may be a finite-dimensional vector space). In this paper, we will study both settings: the noncontextual setting in Section 3 and the contextual one in Section 4.
this section cite: ['b30', 'b41', 'b20']

Section: Adaptive Design
In a randomized controlled trial (RCT), the experimenter (randomly) decides whether to apply treatment or control to each unit, and observes the corresponding outcome but not the counterfactual. These randomized decisions for all units constitute the experimental design. We study adaptive experimental designs, described as follows.
this section cite: []

Section: T -round Adaptive Design Protocol
Potential outcomes {(y t (1), y t (0))} t∈[T ] are generated upfront (but not shown to Experimenter). Then, sequentially for each unit t = 1 . . . T :
1. (Contextual setting only) Experimenter observes pre-treatment covariate x t ∈ X .
2. Experimenter sets treatment probability p t .
3. Experimenter flips bias-p t coin to obtain realized treatment decision: Z t ∼ Bernoulli(p t ).
this section cite: []

Section: Experimenter observes outcome Y t = y t (Z t ).
By contrast, the standard nonadaptive (Bernoulli) trial fixes upfront the same treatment probability p t = p for all units t, and uses it throughout the experiment without any adjustments.
Our estimand of interest is the average treatment effect (ATE), which corresponds to the difference between the average outcomes of treatment and control units in the population. We provide the formal definition below.
Definition 2.1 (ATE). The average treatment effect for potential outcomes {(y t (1), y t (0))} T t=1 is:
τ T = 1 T T t=1 y t (1) -y t (0).
A classical estimator of the ATE is the adaptive IPW estimator (Horvitz & Thompson, 1952), which employs inverse probability weighting. We define it next.
Definition 2.2 (Adaptive IPW Estimator). The adaptive IPW estimator of the ATE τ T is:
τT = 1 T t Y t Z t p t - 1 -Z t 1 -p t .
This estimator is unbiased, meaning that for any outcomes {(y t (0), y t (1)} T t=1 and any adaptive design (p t ) T t=1 with all p t ∈ (0, 1), we have E[τT ] = τ T . Thus, no matter what adaptive design Experimenter employs, the induced adaptive IPW estimator will always be unbiased. However, the estimator's variance will vary based on the design, making some designs more efficient than others.
Objective: Minimize Variance of ATE Estimator Our main goal will be to construct adaptive designs that asymptotically approach the variance of the best-in-hindsight experimental design in some benchmark class. A basic class of designs is that of nonadaptive designs, parameterized by the choice of fixed propensity p ∈ (0, 1). Formally, we measure the Neyman regret (Kato et al., 2020;Dai et al., 2023) of any proposed adaptive design as the (time-rescaled) difference between its IPW estimator variance and the variance of same estimator under the most efficient nonadaptive design.
To define Neyman regret, note (see Proposition 2.2 of Dai et al. (2023)) that Var[τ T ] = T t=1 E [f t (p t )] /T 2 -k ATE ,
where f t (p) := y t (1) 2 /p + y t (0) 2 /(1 -p) is the variance of the propensity-p IPW estimator at unit t, and k ATE = T t=1 (y t (1) -y t (0)) 2 /T 2 is a design-independent term. We are now ready to provide the formal definition.
Definition 2.3 (Neyman Regret (Kato et al., 2020;Dai et al., 2023)). The Neyman regret of adaptive design (p t ) T t=1 on a potential outcomes sequence {(y t (1), y t (0))} T t=1 is:foot_3
RegVar T = max p * T ∈(0,1) T t=1 f t (p t ) -f t (p * T ).
Thus the variance of the IPW estimator for a design (p t )
T t=1 differs from that of the best nonadaptive design by exactly RegVar T /T 2 , justifying the Neyman regret definition. Our goal will be to develop adaptive designs with sublinear expected Neyman regret: E [RegVar T ] = o(T ), or equivalently with vanishing average expected Neyman regret: E [RegVar T /T ] = o(1). We call any design that satisfies this a no-regret design.
this section cite: ['b18', 'b22', 'b7', 'b7']

Section: Efficient Non-Contextual ATE Estimation
We now present our first contribution: An adaptive design that achieves O(log T ) Neyman regret under natural assumptions on the outcomes. We begin by discussing the O( √ T )-Neyman regret design ClipOGD of Dai et al. (2023), and then modifying it to better exploit the strongly convex structure of the Neyman objective. Next, we discuss further guarantees on our method's performance.
this section cite: ['b7']

Section: Adaptive Design with Logarithmic Neyman Regret
Meta-Design: ClipOGD The first finite-population design that achieves sublinear Neyman regret, ClipOGD, was introduced by Dai et al. (2023). Leveraging the fact that the per-round Neyman objectives f t (p) are convex in p, it performs a modified version of online gradient descent (OGD) on f t to adaptively modify the treatment probabilities p t .
The complicating factor is that the gradients of f t diverge when p is close to 0 or 1: standard OGD analyses typically require explicit or implicit bounds on the gradients of the objective (Hazan et al., 2016), so vanilla projected OGD on the entire interval [0, 1] will not work without modification. ClipOGD solves this problem by clipping the OGD iterates {p t } t∈N+ to be within a nested family {[δ t , 1 -δ t ]} t∈N+ of subintervals of (0, 1), which gradually expand to cover the whole interval in the infinite time limit (i.e., lim t→∞ δ t = 0). The expansion is needed to handle cases when p *
T is close to the boundary. In view of this, we let δ t = 1/h(t) for all t ∈ N + , where h : N + → R >0 is some strictly increasing function with lim t→∞ h(t) = ∞. We call δ t the clipping rate, h the clipping function, and refer to any adaptive design (p t ) t∈N+ that satisfies 1/h(t) ≤ p t ≤ 1 -1/h(t) for all t as h-clipped. Algorithm 1 gives the pseudocode for ClipOGD. Here, ΠS(x) denotes the projection of x onto interval S ⊂ (0, 1).
Algorithm 1 ClipOGD (Dai et al., 2023) Initialize p 0 ← 0.5 and g 0 ← 0 for units t = 1, 2, . . . do Set step size η t > 0 and clipping rate δ t ∈ (0, 0.5) Set treatment probability p t ← Π
[δt,1-δt] Dai et al. (2023) analyzed and provided guarantees for a specific instantiation of ClipOGD, where η t = 1/T and δ t = 0.5• t -1/α where α = √ 5 log T for all t = 1, . . . , T . For clarity, we call this design ClipOGD 0 . Their main result proves that ClipOGD 0 has O( √ T ) Neyman regret under a moment assumption on the outcomes: 0 < c ≤ ( 1 T T t=1 y i (t) 2 ) 1/2 and ( 1 T T t=1 y i (t) 4 ) 1/4 ≤ C for i ∈ {0, 1} and some c ≤ C. However, the learning rate of ClipOGD 0 has several drawbacks. First, it is too conservative, precluding improvement in Neyman regret beyond O( √ T ). Second, it is horizon-dependent, making it necessary to know (or commit to) T upfront. Finally, it is constant rather than decreasing, so the design probabilities will jump around (rather than gradually converge) during any given run of ClipOGD 0 .
(p t-1 -η t • g t-1 ) Set treatment decision Z t ∼ Bernoulli(p t ) Observe outcome Y t ← y t (Z t ) Set gradient estimate: g t ← Y 2 t -Zt p 3 t + 1-Zt (1-pt) 3 end for ClipOGD 0 : A O( √ T ) Regret Design In their paper,
ClipOGD SC : Our O(log T ) Regret Design We now present an adaptive design called ClipOGD SC that addresses these issues: It uses the learning rate η t ∼ 1/t that, under Assumption 3.1, (1) achieves an exponentially improved Neyman regret bound, (2) is anytime, i.e., does not require advance knowledge of the time horizon T , and (3) its propensities converge in L 2 to the hindsight-best propensity. Our Neyman regret bound relies on a stricter assumption than the one made by Dai et al. (2023)'s, which we detail below.
this section cite: ['b7', 'b16', 'b7', 'b7', 'b7']

Section: Assumption 3.1 (Bounds on Potential Outcomes).
There exist positive constants c, C such that outcomes {(y t (0), y t (1))} t≥1 satisfy for all time horizons T :
max t≥1 {|y t (0)|, |y t (1)|} ≤ C, c ≤ min t≥1 y t (0) 2 + y t (1) 2 1/2 c ≤ min i∈{0,1} 1 T T t=1 y t (i) 2 1/2 .
Next, let h inv be the inverse function of h, defined via the identity h inv • h = h • h inv = Id. Our main result is the following Neyman regret bound in terms of T , h, and h inv .
Theorem 3.2 (Stronger Neyman Regret Bound). Suppose Assumption 3.1 is satisfied with C, c the corresponding constants. Let h : N + → R >0 be strictly increasing. Let ClipOGD SC be the adaptive design that instantiates Algorithm 1 with learning rate η t = 1/(2c 2 t) and clipping rate δ t = 1/h(t). Then, ClipOGD SC attains the following anytime-valid Neyman regret bound:
E[RegVar T ] = O (h(T )) 5 •log(T )+(h inv (1+C/c)) 2 .
(1)
Since h can be chosen to grow arbitrarily slowly, we can get: E[RegVar T ] = O(log T ).
The proof is contained in Appendix B. It exploits the strong convexity of the Neyman objectives f t enabled by Assumption 3.1 (hence the 'SC' in ClipOGD SC ), by applying the techniques for analyzing strongly convex gradient descent (Hazan et al., 2007;Rakhlin et al., 2012).
Compared to the analysis in Dai et al. (2023), we make explicit the dependence of the regret of ClipOGD on the clipping rate. Note that the choice of h is flexible in the sense that any h(t) = o(t 0.2-ε ) for any ε > 0 will result in a regret bound that is sublinear in T . From a practical standpoint, however, picking h may be a nontrivial affair, as a slower-growing h will have a faster-growing inverse mapping h inv . While the h inv -dependent term in the regret bound is constant in T , it can still be large in the constants of the problem. Intuitively, if C/c is large, the optimal propensity p * T may be near the boundary and convergence may be slow. We hope future work will further explore the 'well-conditioning' properties of Neyman regret.
this section cite: ['b15', 'b38', 'b7']

Section: Convergence of Adaptive Treatment Probabilities
We now investigate the trajectory of treatment probabilities (p t ) t≥1 produced by ClipOGD SC . Ideally, these propensities would converge to the optimal probabilities (p * T ) T ≥1 as T grows large. By tweaking the arguments used in establishing our Neyman regret bounds of Theorem 3.2, we can obtain convergence in squared means (and hence in probability). The next claims formalize this result. In particular, we first establish a quantitative bound on the L 2 convergence of our propensities to the benchmark ones. (See Appendix B for the derivation.) Lemma 3.3 (L 2 -Deviation from Benchmark Design). The deviation of the design probabilities of ClipOGD SC from the best nonadaptive design probabilities is L 2 -bounded for all T as:
E (pT -p * T ) 2 ≤ -Θ E[RegVar T ] T + O (h(T )) 2 log T T .
This implies the following L 2 -convergence result, subject to an assumption on the Neyman regret of ClipOGD SC which asks for it to not consistently outperform the optimal nonadaptive design.
Corollary 3.4 (L 2 -Convergence to Benchmark Design). Assume ClipOGD SC has asymptotically nonnegative Neyman regret: lim inf T →∞ E[RegVar T ] T ≥ 0. Then, its propensities (p t ) t≥1 will converge to the benchmark nonadaptive propensities
(p * T ) T ≥1 in squared means: E (p T -p * T ) 2 → 0 as T → ∞.
In the special case of sequences of potential outcomes that are (i.i.d.) samples from a superpopulation, the regret nonnegativity holds automatically, implying that our adaptive design will necessarily converge to the best nonadaptive design without further assumptions.
Corollary 3.5 (Convergence in the Superpopulation Setting). Suppose that the outcomes are drawn i.i.d. from a superpopulation: (y t (0), y t (1)) ∼ D for all t ≥ 1 and any fixed distribution D. Then, ClipOGD SC guarantees that E (p T -p * ) 2 → 0 at the rate O(log T /T ), and thus in particular that p T → p * in probability.
Proof. In the superpopulation setting, any adaptive design will have nonnegative Neyman regret:
f t (p) = f (p) = E[y(1) 2 ]/p + E[y(0) 2 ]/(1 -p) has the same optimum p * = 1 + E[(yt(0)) 2 ]/ E[(yt(1)) 2 ] -1 for all units t, so E[RegVar T ] = E T t=1 (f (p t ) -f (p * )) ≥ 0.
this section cite: []

Section: Valid CIs for the Adaptive IPW Estimator
We now turn to the issue of endowing the IPW estimator τT induced by our adaptive design with asymptotically valid confidence intervals (CIs). In general, the existence and construction of valid CIs for τT delicately depends on the choice of the design. However, we will now see that a construction of Dai et al. (2023) lends conservative CIs to all h-clipped adaptive designs with vanishing regret.
To formalize this result, we make a standard assumption: that the outcome sequences are not perfectly anticorrelated. To state it, define "empirical second raw moments" of the two outcome populations as: S T (i) 2 := 1 T T t=1 (y t (i)) 2 for i ∈ {0, 1}. Assumption 3.6 (Correlation of Outcome Populations (Dai et al., 2023)). For a constant c ρ > 0 and all T ≥ 1, the running correlation ρ T of the sequences {(y t (0), y t (1))} t≥1 satisfies:
ρ T ≥ -1 + c ρ , where ρ T := 1 T T t=1 y t (1)y t (0) S T (1)S T (0) .
Theorem 3.7 (CIs for Clipped Adaptive Designs). Suppose the potential outcomes satisfy Assumption 3.1 and Assumption 3.6. Consider any h-clipped adaptive design (p t ) t≥1 with vanishing Neyman regret: lim T →∞ RegVar T = 0. Let VB = 4 T S T (1)S T (0) be a conservative upper bound on the hindsight-best nonadaptive variance. Then, letting (Z t ) t≥1 be the treatment decisions, the estimator of Dai et al. (2023) given by:
VB = 4 T 1 T T t=1 (y t (1)) 2 Z t p t 1 T T t=1 (y t (0)) 2 1 -Z t 1 -p t converges to VB in probability at rate O p h(T )/T .
Consequently, VB can be used to construct asymptotically valid Chebyshev-type confidence intervals for the adaptive IPW estimator τT under any adaptive design satisfying the above conditions. Specifically, for any confidence level α ∈ (0, 1]:
lim inf T →∞ Pr τ T ∈ τT ± α -1/2 VB ≥ 1 -α.
The proof for Theorem 3.7 is outlined in Appendix C.
this section cite: ['b7', 'b7', 'b7']

Section: Efficient Multigroup ATE Estimation
The Contextual Setting Section 3 covers non-contextual adaptive designs that only observe outcomes. A contextual adaptive design, however, also observes pre-treatment covariates x t ∈ X at the start of each round, which can help predict potential outcomes (y t (0), y t (1)). We can leverage this extra information to improve treatment assignments and outcome estimation.
this section cite: []

Section: A Multigroup Formulation
We frame the contextual setting in a multigroup way. Before the experiment, we have a finite set of context-defined groups
G = {G 1 , G 2 , . . .}, each G ⊆ X ,
where X is the feature space. Any covariate vector x t can belong to none, one, or more groups. The group definition is dependent on the specifics of the task, e.g., in a medical application the features x t could represent a patient's health history.
Our objective in a multigroup setting, informally, is to design an adaptive scheme that offers ATE estimation efficiency guarantees (such as Neyman regret guarantees) not only on average over the entire sequence of units but also on each subsequence that results from conditioning on units belonging to a group G, simultaneously for all groups G ∈ G.
this section cite: []

Section: A New Metric: Multigroup Neyman Regret
We introduce multigroup Neyman regret as a strengthening of (vanilla) Neyman regret. Specifically, given any contextual group collection G, G-multigroup Neyman regret is the maximum Neyman regret that an adaptive design achieves over any group G in the collection. We formalize it next.
Definition 4.1 (G-Multigroup Neyman Regret). Given any group collection G ⊆ 2 X , the group-conditional Neyman regret of an adaptive design A on any group G ∈ G is defined as:
RegVar T (A; G) := E max p * ∈(0,1) T t=1 1[x t ∈ G] (f t (p t ) -f t (p * )) .
The G-multigroup Neyman regret of A is then defined as its maximum group-conditional Neyman regret over all groups G ∈ G:
RegVarMG T (A; G) := max G∈G RegVar T (A; G).
this section cite: []

Section: Achieving O( √ T ) Multigroup Neyman Regret
We now present in Algorithm 2 an adaptive design which we call MGATE (for Multi-Group ATE) and achieves the
O( √ T ) multigroup Neyman regret bound. Algorithm 2 A M GAT E : Multigroup Adaptive Design Receive clipping function h : N + → R >0 Receive number of groups d = |G| Set group counts n 0 ← 0 d Initialize p 1 ← 0.5 • 1 d // At round t, p t = (p t,G ) G∈G will contain group propensities Initialize w ′ 1 ← 1 d , L 0 ← 0 d , q 0 ← 0 // Parameters used to update group weights for t = 1, 2, . . . do Receive covariate vector x t ∈ X , determine the set of active groups G t = {G : x t ∈ G, G ∈ G} Cast G t as indicator vector a t ∈ {0, 1} d (a t,G = 1 ⇐⇒ G ∈ G t ). Set group counts: n t ← n t-1 + a t Normalize group weights: w t,eff ← at⊙w ′ t ⟨at,w ′ t ⟩ // Set inactive group weights to 0 Set effective treatment probability: p t,eff ← ⟨w t,eff , p t ⟩ // Aggregate group propensities Set treatment decision: Z t ∼ Bernoulli(p t,eff ) Receive realized outcome: Y t ← y t (Z t ) for active groups G ∈ G t do / * Update group propensities using group-specific ClipOGD SC -type update * / Set estimated Neyman gradient as: g t,G ← Y 2 t Zt p t,eff + 1-Zt 1-p t,eff -Zt p 2 t,G + 1-Zt (1-p t,G ) 2 Update p t+1,G ← Π [δ t,G ,1-δ t,G ] (p t,G -η t,G • g t,G ), where η t,G ← 1 2c 2 •n t,G and δ t,G ← 1 h(n t,G ) / * Get losses used to update group weights * / Set estimated Neyman loss as: ℓ t,G ← Y 2 t Zt p t,eff + 1-Zt 1-p t,eff Zt p t,G + 1-Zt 1-p t,G end for for inactive groups G ̸ ∈ G t do Set p t+1,G ← p t,G and ℓ t,G ← 0 // Inactive groups are not updated end for / * Update group weights: Higher cumulative group losses → larger weights * / Set surrogate loss: ℓ t ← a t ⊙ ℓ t -⟨ ℓ t , w t,eff ⟩ Set L t ← L t-1 + ℓ t and q t ← q t-1 + ∥ℓ t ∥ 2 2 Update group weights: w ′ t+1 ← max 0 d , -1 √ qt L t end for
Additional Notation: We use ⊙ to denote elementwise vector multiplication, and let 1 d , 0 d be d-dimensional allones and all-zeros vectors. Also note that the update of w ′
t+1 takes an elementwise maximum of the vectors, and assumes that 0/0 = 0 to account for the corner case q t = 0.
Algorithm Description: Given a collection G of d groups, in each round MGATE reads off the currently active groups G t ⊆ G, i.e., those groups that contain x t (G ∋ x t ), and then proceeds to determine the new treatment probability by aggregating the "best-guess" probabilities for all active groups G ∈ G t determined based on the past performance of those groups. To do so, MGATE maintains group weights w ′ t,G and group-specific propensities p t,G . It comes up with a single effective treatment probability: p t,eff ∼ G∈Gt w ′ t,G p t,G in each round by reweighing the group specific propensities of the active groups. This effective treatment probability should simultaneously satisfy the interests of all active groups. The treatment decision Z t is then generated according to p t,eff . After the outcome is revealed, MGATE updates all group weights, as well as the propensities of groups that were active.
We can show that MGATE achieves the following multigroup Neyman regret guarantee. We note that MGATE is anytime valid, meaning that just like our noncontextual design ClipOGD SC , it does not require advance knowledge of the time horizon T .
Theorem 4.2 (Guarantees for Algorithm 2). Fix any context space X and finite group family G ⊆ 2 X . Supposefoot_4 Assumption 3.1 holds with lower bound constant c > 0. Then, for any clipping function h, the expected multigroup regret of Algorithm 2 will be bounded as:
RegVarMG T (A; G) = O |G| • (h(T )) 5 • √ T .
this section cite: []

Section: Technical Overview
The full analysis of Algorithm 2 is contained in Appendix D. It builds on several tools recently developed in the online learning literature, which are formally introduced in Appendix D.1, and we briefly survey them here. The central tool is the sleeping experts algorithmic framework (Blum & Lykouris, 2020), which has recently been shown to be able to combine the wisdom of multiple sub-learners (or experts) into a meta-algorithm with performance on par with each of the sub-learners. The key difference from typical online aggregation schemes is that each sub-learner is allowed to be inactive (asleep) on some rounds, on which it does not give advice to the meta-algorithm. At a high level, to obtain multigroup Neyman regret, we would thus like to use a sleeping experts algorithm to aggregate propensities suggested by |G| = d copies of ClipOGD SC that are respectively active on all groups G ∈ G; the aggregated design would then perform comparably to each copy of ClipOGD SC on its group G. Then, since that copy of ClipOGD SC will have no regret on group G, neither will the aggregated design.
this section cite: ['b2']

Section: Challenges and Solutions
Past work on sleeping experts does not fully address the combination of difficulties present in our setting: (1) stochastic (realized outcome) feedback rather than full-information (both outcomes) feedback; (2) the need to perform clipping of the iterates (propensities) to explicitly restrict them from approaching the feasible set's boundary too fast; and (3) the fact that the gradient feedback magnitude grows unboundedly as T → ∞, even with clipping.
While there are a limited number of "sleeping bandits" algorithms in the literature (e.g., see Nguyen & Mehta (2024)) that address the stochastic feedback, they don't naturally extend to cover both of the latter two issues. Therefore, we design from scratch a new sleeping experts algorithm tailored to all of these challenges. It employs scale-free updates of the group weights w ′ t so as to control the loss and gradient feedback magnitudes; we achieve this by deploying an instance of the seminal scale-free SOLO FTRL algorithm of Orabona & Pál (2018) and endowing it with sleeping experts regret guarantees via a recent reduction of Orabona (2024). To clip the effective probability magnitudes, our algorithm aggregates over the suggested per-group probabilities via convex combinations rather than via sampling from their mixture. Finally, to ensure that the per-group propensity updates remain valid under stochastic gradient feedback and despite the aggregator using a different propensity than the suggested per-group one, MGATE uses a combination of unbiased first-order ( g t,G ) and zeroth-order ( ℓ t,G ) per-group feedback estimators, which depend on both p t,eff and p t,G .
A Generalized Meta-Design Our analysis in Appendix D generalizes beyond MGATE (Algorithm 2). Indeed, our approach more generally allows the use of any scale-free sleeping experts algorithm to update group weights, and any ClipOGD-style (see Appendix D.3) no-regret adaptive designs to update the groupwise treatment probabilities. Thus, we more generally provide a meta-design that reduces multigroup designs to a broad class of non-contextual, noregret designs. This generalized meta-design is given as Algorithm 7 in Appendix D.4, and Theorem D.6 contains its regret bound, of which Theorem 4.2 above is a corollary.
this section cite: ['b32', 'b35', 'b34']

Section: Experimental Results
We first present the results for the non-contextual setting and then turn to the analysis of the performance for the contextual algorithm. Our code is available at the following link: https://github.com/amazon-science/adaptive-abtester.
this section cite: []

Section: Non-Contextual Experiments
Tasks We compare our method ClipOGD SC with ClipOGD 0 (Dai et al., 2023) on multiple tasks. Below, we show two key datasets (one synthetic and one realworld) used in our experiments, with full details in Appendix E. The first is a synthetic dataset is generated as follows: y t (i) iid ∼ N (µ i , σ 2 ) for t = 1, . . . , T and i = 0, 1 with µ 0 = 1 and µ 1 = 2. We vary σ i ∈ R + to showcase where our method succeeds and where it struggles. The second dataset comes from Egypt's largest microfinance organization (Groh & McKenzie, 2016), covering 2,961 clients. Here, the treatment is a new insurance product, and the outcome is how much individuals invest in machinery. Following Dai et al. (2023), we fill missing values with Gaussian noise and resample each unit five times to increase the population size. We also present experiments on the ASOS Digital Experiments Dataset (Liu et al., 2021), and on question-answering tasks for large language models, including BigBench (Srivastava et al., 2023), in the Appendix.
this section cite: ['b7', 'b11', 'b7', 'b27', 'b44']

Section: Experimental Setup
In our simulation, each unit is randomly assigned to treatment or control using the treatment probability from our method or ClipOGD 0 . We repeat this process 10,000 times, generating many different treatment-control paths. We then measure the Neyman regret by averaging the regret across these probabilities obtained at each time step.
Hyperparameter Choices Throughout the experiments, we use the following hyperparameters. For our method, we set η t = 2/t and δ t = 1/h(t), where the clipping function is h(t) = exp (log(t + 2)) 1/4 . For ClipOGD 0 , we follow Dai et al. (2023) with a constant learning rate η t = 1/ √ T and clipping rate δ t = 0.5 • t -1/ √ 5 log T .
this section cite: ['b7']

Section: Results
We analyze three synthetic data settings where we vary σ as {0.1, 1, 10}. As σ increases, the ratio C/c also grows, so by Equation (1), we expect slower convergence of our algorithm. We set T = 50,000. Figure 1 shows the Neyman regret across these settings, matching our theoretical expectations: when σ = 0.1, the regret of ClipOGD SC drops to 0 quickly, but for larger σ, the regret remains high and converges later. The regret of ClipOGD 0 instead keeps increasing with time. Nonetheless, in line with Corollary 3.4, Figure 1 also shows that our method's adaptively chosen propensities ultimately converge to the Neyman optimal probability in all three cases. By contrast, the propensities of ClipOGD 0 only converge when σ = 10, which happens to match the initial probability of 0.5. Next, we turn to examine the results on the microfinance data. Figure 2 illustrates the treatment probabilities and Neyman regret for both algorithms. On average, each design assigns probabilities near the Neyman probability. However, those of ClipOGD 0 exhibit higher variance compared to ClipOGD SC . This translates into greater Neyman regret in Group = 0 Group = 1 Group = 2 0 5000 10000 15000 0 3000 6000 9000 0 3000 6000 9000 12000 0.01 0.02 0.000 0.005 0.010 0.015 0.020 0.00 0.01 0.02 Round Neyman regret Method MGATE CLIPOGD SC CLIPOGD 0 Figure 3. Group-conditional Neyman regret of ClipOGD and MGATE on microfinance data. MGATE produces the lowest G-multigroup Neyman regret as desired, and in this case dominates the non-contextual ClipOGD variants for each group, including the noncontextual group G0 = X .
later rounds, which never converges to 0. The probabilities assigned by our method, instead, converge to the Neyman probability, yielding vanishing average Neyman regret.
this section cite: []

Section: Contextual Experiments
Here we present our contextual results using Algorithm 2 over the previously-described datasets. To standardize the contextual groups in each experiment, we design simple, synthetic post-hoc groups by scoring each sample as s t = 1/ 1 + yt(0) 2 yt(1) 2 +ϵ (the optimal Neyman sampling probability for the single sample). Our groups are computed by checking whether sample t belongs to some predetermined quantile of the score function
G 0 = X , G 1 = 1 F -1 (s t ) ≤ 2 3 , G 2 = 1 1 3 ≤ F -1 (s t )
. We note that these groups are overlapping and informative since G 1 is guaranteed to have lower or equal optimal sampling probability than G 2 .
We stress that these groups are included for illustrative purposes and rely on information that would be unobservable in a real ATE experiment, but nonetheless showcase the potential for high-quality contextual information for multi-group ATE. Figure 3 shows the Neyman regret for ClipOGD 0 , ClipOGD SC , and MGATE on the microfinance dataset on each group; our MGATE method achieves the lowest group-conditional regret out of all the methods, effectively minimizing the G-multigroup Neyman regret, and thereby validating our theoretical results. Additional contextual experiments are provided in the Appendix.
this section cite: []

Section: Conclusion
In this paper, we have studied adaptive designs for unbiased ATE estimation with finite-population guarantees. We introduced a modification of the ClipOGD algorithm that provably yields vanishing Neyman regret, achieving an anytimevalid O(log T ) Neyman regret, improving upon previous O( √ T ) guarantees. We also extend our framework to incorporate contextual information by introducing a multigroup formulation. Our proposed multigroup adaptive design ensures O( √ T ) regret for each predefined group, enabling efficiency improvements for subgroup ATE estimation. Experimental results corroborate these findings.
Overall, these results suggest that adaptive experimentation can achieve strong finite-population efficiency guarantees, offering practical advantages for a wide range of applications. Future work could explore extensions to other experimental designs and further reductions in regret rates.
Zimmert, M. and Lechner, M. Nonparametric estimation of causal heterogeneity under high-dimensional confounding. arXiv preprint arXiv:1908.08779, 2019.
this section cite: []

Section: References
Ref_id:b0 Title: Oracle efficient algorithms for groupwise regret Year: (2024)
Ref_id:b1 Title: Batch adaptive designs to improve efficiency in social science experiments Year: (2022)
Ref_id:b2 Title: Advancing subgroup fairness via sleeping experts Year: (2020)
Ref_id:b3 Title: Fisher-schultz lecture: Generic machine learning inference on heterogenous treatment effects in randomized experiments, with an application to immunization in india Year: (2017)
Ref_id:b4 Title: Adaptive design methods in clinical trials-a review Year: (2008)
Ref_id:b5 Title: Evaluating cross-lingual sentence representations Year: (2018)
Ref_id:b6 Title: Semiparametric efficient inference in adaptive experiments Year: (2023)
Ref_id:b7 Title: Clip-ogd: An experimental design for adaptive neyman allocation in sequential experiments Year: (2023)
Ref_id:b8 Title: Group-wise oracle-efficient algorithms for online multi-group learning Year: (2024)
Ref_id:b9 Title: Adaptive designs for clinical trials of drugs and biologics: Guidance for industry Year: (2019)
Ref_id:b10 Title: Precise model benchmarking with only a few observations Year: (2024)
Ref_id:b11 Title: Macroinsurance for microenterprises: A randomized experiment in post-revolution egypt Year: (2016)
Ref_id:b12 Title: Confidence intervals for policy evaluation in adaptive experiments Year: (2021)
Ref_id:b13 Title: Adaptive experimental design using the propensity score Year: (2011)
Ref_id:b14 Title: Balancing covariates in randomized experiments with the gram-schmidt walk design Year: (2024)
Ref_id:b15 Title: Logarithmic regret algorithms for online convex optimization Year: (2007)
Ref_id:b16 Title: Introduction to online convex optimization Year: (2016)
Ref_id:b17 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b18 Title: A generalization of sampling without replacement from a finite universe Year: (1952)
Ref_id:b19 Title: The theory of responseadaptive randomization in clinical trials Year: (2006)
Ref_id:b20 Title: Causal inference in statistics, social, and biomedical sciences Year: (2015)
Ref_id:b21 Title: Mistral 7b Year: (2023)
Ref_id:b22 Title: Efficient adaptive experimental design for average treatment effect estimation Year: (2020)
Ref_id:b23 Title: Multi-accurate cate is robust to unknown covariate shifts Year: (2024)
Ref_id:b24 Title: Online minimax multiobjective optimization: Multicalibeating and other applications Year: (2022)
Ref_id:b25 Title: Double machine learning and design in batch adaptive experiments Year: (2024)
Ref_id:b26 Title: Optimal adaptive experimental design for estimating treatment effect Year: (2024)
Ref_id:b27 Title: Datasets for online controlled experiments. NeurIPS 2021 Datasets and Benchmarks Track Year: (2021)
Ref_id:b28 Title: Logarithmic neyman regret for adaptive estimation of the average treatment effect Year: (2024)
Ref_id:b29 Title: Optimistic algorithms for adaptive estimation of the average treatment effect Year: (2025)
Ref_id:b30 Title: Sur les applications de la théorie des probabilités aux experiences agricoles: Essai des principes Year: (1923)
Ref_id:b31 Title: On the two different aspects of the representative method: the method of stratified sampling and the method of purposive selection Year: (1992)
Ref_id:b32 Title: Near-optimal per-action regret bounds for sleeping bandits Year: (2024)
Ref_id:b33 Title: Adaptive experimental design: Prospects and applications in political science Year: (2021)
Ref_id:b34 Title: Black-box reductions: Sleeping experts Year: (2024-10-20)
Ref_id:b35 Title: Scale-free online learning Year: (2018)
Ref_id:b36 Title: A large-scale multi-subject multi-choice dataset for medical domain question answering Year: (2022)
Ref_id:b37 Title: Xcopa: A multilingual dataset for causal commonsense reasoning Year: (2020)
Ref_id:b38 Title: Making gradient descent optimal for strongly convex stochastic optimization Year: (2012)
Ref_id:b39 Title: On distributional discrepancy for experimental design with general assignment probabilities Year: (2024)
Ref_id:b40 Title: Some aspects of the sequential design of experiments Year: (1952)
Ref_id:b41 Title: Estimating causal effects of treatments in randomized and nonrandomized studies Year: (1974)
Ref_id:b42 Title: Debiased machine learning of conditional average treatment effects and other causal functions Year: (2021)
Ref_id:b43 Title: Optimal design of sampling from finite populations: A critical review and indication of new research areas Year: (1970)
Ref_id:b44 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2023)
Ref_id:b45 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b46 Title: The construction and analysis of adaptive group sequential designs. U.C. Berkeley Division of Year: (2008)
Ref_id:b47 Title: Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges Year: (2015)
Ref_id:b48 Title: Causal inference: A statistical learning approach Year: (2024)
Ref_id:b49 Title: Sequential tests of statistical hypotheses Year: (1992)
Ref_id:b50 Title: Subgroup-based adaptive (suba) designs for multi-arm biomarker trials Year: (2016)
Ref_id:b51 Title: The fallacy of minimizing local regret in the sequential task setting Year: (2024)
Ref_id:b52 Title: Can a machine really finish your sentence? Year: (2019)
Ref_id:b53 Title: Inference for batched bandits Year: (2020)
Ref_id:b54 Title: Statistical inference with m-estimators on adaptively collected data. Advances in Neural Information Processing Systems Year: (2021)
