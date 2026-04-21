Title: On the sample complexity of semi-supervised multi-objective learning
Abstract: In multi-objective learning (MOL), several possibly competing prediction tasks must be solved jointly by a single model. Achieving good trade-offs may require a model class G with larger capacity than what is necessary for solving the individual tasks. This, in turn, increases the statistical cost, as reflected in known MOL bounds that depend on the complexity of G. We show that this cost is unavoidable for some losses, even in an idealized semi-supervised setting, where the learner has access to the Bayes-optimal solutions for the individual tasks as well as the marginal distributions over the covariates. On the other hand, for objectives defined with Bregman losses, we prove that the complexity of G may come into play only in terms of unlabeled data. Concretely, we establish sample complexity upper bounds, showing precisely when and how unlabeled data can significantly alleviate the need for labeled data. This is achieved by a simple pseudo-labeling algorithm.

Section: Introduction
The multi-objective learning (MOL) paradigm has recently emerged to extend the classical problem of risk minimization from statistical learning to settings with multiple notions of risk [32,19,59,27]. Multi-objective learning problems are ubiquitous in practice, as it often matters how our models behave with respect to multiple metrics and across different populations. For example, consider designing a policy for a self-driving car: the risks could measure different notions of safety (e.g., safety of passengers or pedestrians), or safety under various conditions (e.g., different locations).
More formally, we study the MOL setting with K population risk functionals R 1 , . . . , R K , each quantifying an average, possibly different, loss ℓ k incurred by a prediction model g over the data distribution P k . The aim is to learn models from a class G that minimize all K excess risks E k (g) := R k (g)inf R k jointly, using only finite-sample access to the distributions. Here, inf R k is the Bayes risk of the kth task, which is the smallest achievable risk over all measurable functions. Specifically, we study the sample complexity of learning the set of Pareto optimal models in G. Recall that a model is Pareto optimal in G if any alternative model in G that reduces one risk necessarily increases another (see Definition 1); we often simply say that such a model makes an optimal trade-off. Under mild conditions, the set of Pareto optimal models can be recovered by minimizing a family of scalarized objectives T s that we call the s-trade-offs: 1 min g∈G T s (g) := s E 1 (g), . . . , E K (g) the s-trade-off achieved by g , s ∈ S
where the map s : R K → R is from some family S of scalarization functions that aggregates the excess risks into a single statistic. Notice that if the excess risks were known, the problem in Eq. ( 1) would reduce to a family of classical (multi-objective) optimization problems [44,54,38]. However, because the objectives in Eq. ( 1) depend on distributional quantities that are unknown, we need to learn the solutions from data. Specifically, we study the sample complexity of achieving Eq. ( 1) up to errors ε s > 0, a problem we call S-multi-objective learning, S-MOL for short (see Definition 2).
Two main lines of work have studied the sample complexity of MOL, both predominantly in the supervised framework. In the multi-distribution learning (MDL) literature [27,4,51,73], the goal of the learner is to recover a solution to Eq. ( 1) for one specific s-trade-off induced by the scalarization s(v) = max k∈[K] v k . This yields the familiar min-max formulation of MOL. 2 And in the literature for the general S-MOL setting [19,59], the learner aims to solve Eq. ( 1) for multiple scalarizations. Both lines establish sample complexity bounds in terms of capacity measures of G, which can be shown to be tight in the worst case. However, solutions with good trade-offs may only be found in a complex model class G even when individual tasks are easy to solve in smaller classes H k . In such cases, previous worst-case results do not address whether it is really necessary to pay the full, supervised statistical cost of S-MOL over G. This motivates our consideration of semi-supervised multi-objective learning, in which the learner has access to both labeled and (cheaper) unlabeled data for each of the K tasks. In the single-objective setting, it is well-known that access to unlabeled data can, at times, significantly reduce the amount of labeled data required [17,70]. But for multiobjective learning, the sample complexity in a semi-supervised setting is largely unexplored, with only a few exceptions [5,65] that rely on additional assumptions for unlabeled data to be helpful (see Appendix A for a discussion of related works). Thus, the question we aim to address in this paper is
Given that each task k ∈ [K] is solvable in a hypothesis class H k , how much labeled and unlabeled data is needed to achieve trade-offs available in a larger function class G?
In this paper, we give a holistic characterization of the conditions when unlabeled data can help and by how much. In terms of sample complexity upper bounds, we show that for a large class of losses, the capacity of G comes into play only in the amount of unlabeled data required, while the amount of labeled data merely depends on that of H 1 , . . . , H K . Moreover, we show that these rates are achieved by a simple, pseudo-labeling-based algorithm. Concretely, our contributions are as follows:
• We first show hardness of S-MOL under uninformative losses via a minimax sample complexity lower bound that holds even when the learner knows the Bayes-optimal models for each task and has access to the marginal distributions over unlabeled data, i.e., infinite unlabeled data (Section 3.1).
• We then prove that risks induced by Bregman divergence losses-which include the square and cross-entropy losses-effectively disentangle the multi-objective learning problem. For Bregman losses, information about individual risk minimizers can significantly reduce labeled sample complexity in the semi-supervised setting via a simple pseudo-labeling algorithm (Section 3.2).
• Specifically, for S-MOL with Bregman losses, we first provide a uniform bound over the excess s-trade-offs of the pseudo-labeling algorithm for bounded, Lipschitz losses via uniform convergence (Section 4.1). Our major technical contribution then lies in proving localized rates that are distribution-specific under stronger assumptions (Section 4.2). Crucially, the labeled sample complexity in both bounds only depends on the classes {H k } K k=1 , while G only appears in the unlabeled sample complexity.
Our analysis reveals an interesting insight: even though the pseudo-labeling algorithm is reminiscent of single-objective semi-supervised learning procedures, the reason behind the benefits of unlabeled data turns out to be fundamentally different. In single-objective learning, unlabeled data can only help if, roughly speaking, the marginal carries information about the labels [13,26,76]. Our results, in contrast, hold without any such assumptions. In multi-objective learning, unlabeled data helps the learner determine the relative importance of each test instance to each task: if the likelihood of an input is higher under one task than another, a model can accordingly prioritize the more relevant risk to achieve better trade-offs. This may be completely independent of the labels assigned by each task.
this section cite: ['b31', 'b18', 'b58', 'b26', 'b43', 'b53', 'b37', 'b26', 'b3', 'b50', 'b72', 'b18', 'b58', 'b16', 'b69', 'b4', 'b64', 'b12', 'b25', 'b75']

Section: Semi-supervised multi-objective learning
In this section, we formally introduce the semi-supervised S-multi-objective learning problem. For ease of reference, an overview of notation is provided in Table 2 of Appendix F.
this section cite: []

Section: Preliminaries and the individual tasks
Let X be the feature space, and Y ⊆ R q the label space. We are interested in K prediction tasks, indexed by k ∈ [K] := {1, . . . , K}, over joint distributions P k of (X k , Y k ) on the product space X × Y. We denote the underlying joint probability measure by P. From each task, we observe n k i.i.d. labeled samples {(X k i , Y k i )} n k i=1 from P k , and N k i.i.d. unlabeled samples { X k i } N k i=1 from the marginal of P k on X , denoted P k X . Let D denote the combined dataset of both labeled and unlabeled data. For each task k ∈ [K], we define the population and empirical risks of a function f : X → Y as
R k (f ) := E ℓ k (Y k , f (X k ))
and R k (f ) := 1 n k
n k i=1 ℓ k (Y k i , f (X k i )),(2)
where ℓ k : Y × Y → R is a (not necessarily symmetric) loss function with ℓ k (y, y) being the loss incurred by predicting y when the true label is y. Further, we write F all for the set of all functions f : X → Y for which all integrals in this paper are well-defined.For each k ∈ [K], we assume access to a function class H k ⊆ F all that contains a population risk minimizer of R k , that is,
∃f ⋆ ∈ F all such that f ⋆ k ∈ arg min f ∈F all R k (f ) and f ⋆ k ∈ H k .(3)
The risk that any model f : X → Y incurs is at least R k (f ⋆ k ), so we focus our attention on achieving small excess risk with respect to the Bayes optimal predictor, defined as E k (f ) := R k (f ) -R k (f ⋆ k ).
this section cite: []

Section: Pareto optimality and scalarization
In multi-objective learning, our aim is to learn models g from some function class G that, ideally, achieve low excess risk on all tasks simultaneously. Since, by assumption, the individual tasks are optimally solved in H k , we only consider hypothesis classes G ⊂ F all that satisfy G ⊇ k∈[K] H k . But even if G is very large, minimizing all excess risks may not be possible. In particular, in this work we do not assume that there exists one f : X → Y that performs well across objectives (as opposed to the collaborative learning framework [14] or the setting in [5]). Instead, the aim is to recover the set of Pareto optimal solutions in the class G for the K objectives, formally defined as follows.
Definition 1 (Pareto optimality). Let E 1 , . . . , E K be a collection of excess risk functionals. We say that a function g ∈ G is Pareto optimal in G if there is no other g ′ ∈ G such that
∃k ∈ [K] s.t. E k (g ′ ) < E k (g) and ∀j ∈ [K], E j (g ′ ) ≤ E j (g).
The subset of G containing all Pareto optimal functions is called the Pareto set. The subset of R K containing the excess risk vectors of the Pareto set is called the Pareto front, defined as
F(G) := E 1 (g), . . . , E K (g) : g is in the Pareto set of G ⊆ R K .
In words, any model in G that reduces one risk over a Pareto optimal model must increase another risk. Every Pareto-optimal model corresponds to a distinct "preference" or "trade-off", all of which are equally valid from a decision-theoretic perspective [29]. We can quantify such trade-offs using scalarization functions s : R K → R that, for all f ∈ F all , map the excess risks into a scalar objective
T s (f ) = s E 1 (f ), . . . , E K (f ) with g s ∈ arg min g∈G T s (g),(4)
see also Eq. ( 1). We call T s (f ) the s-trade-off achieved by f . It has a natural interpretation: recall that E k (g s ) is the cost incurred by the k-th task to make this particular type of trade-off over myopically optimizing E k . Then, T s (g s ) aggregates these costs (see Fig. 1). This interpretation also further motivates scalarizing the excess risks instead of the risks: if one task were to have much higher Bayes risk than another, scalarizing the risks would not aggregate the additional cost, cf. Fig. 1a and [1].  Two popular examples of scalarization families are Tchebycheff and linear scalarizations, defined as
R 1 R 2 λ λ R 1 ( f ⋆ 1 ) R 2 ( f ⋆ 2 ) s m a x λ ( R(
S max = s max λ (v) = max k∈[K] λ k v k | λ ∈ ∆ K-1 , S lin = s lin λ (v) = k∈[K] λ k v k | λ ∈ ∆ K-1 ,(5)
where ∆ K-1 is the (K -1)-probability simplex. They represent the worst-case and averaged notions of excess risks, respectively (see Fig. 1a for a visualization of the Tchebycheff scalarization). Minimizing these families of scalarizations recovers the Pareto set under some conditions (e.g., convexity for linear scalarization), see the detailed discussions in [47,23,44]. But of course, other scalarizations also exist [23]. Our most general result (Section 4.1) holds for monotonic scalarizations that satisfy the reverse triangle inequality and positive homogeneity, defined as ∀v, w ∈ [0, ∞) K :
(∀k ∈
Both the linear and Tchebycheff scalarizations from Eq. ( 5) satisfy the properties in Eq. ( 6).
this section cite: ['b13', 'b4', 'b28', 'b0', 'b46', 'b22', 'b43', 'b22']

Section: Multi-objective learning
Because inf g∈G T s (g) may be arbitrarily large, we evaluate our empirical estimates of g s using the excess s-trade-off, defined, for f ∈ F all , as T s (f )inf g∈G T s (g). The S-MOL problem is then to achieve small excess s-trade-off across scalarizations with high probability.
Definition 2 (S-MOL). Let S be a family of scalarizations s : R K → R, (ε s ) s∈S a family of positive real numbers, and δ ∈ (0, 1). Let A be an algorithm that, provided with a dataset D and the function classes {H k } K k=1 and G, returns a family of functions { g s : s ∈ S} ⊂ G. Then A solves the S-multi-objective learning (S-MOL) problem with parameters ((ε s ) s∈S , δ), if
P ∀s ∈ S : T s ( g s ) -inf g∈G T s (g) ≤ ε s ≥ 1 -δ, (S-MOL)
where the probability is taken with respect to draws of the training dataset D.
From the population-level optimization perspective in Eq. ( 1), better trade-offs become possible as the class G grows. This is visualized in Fig. 1b, showing the Pareto fronts achieved by the function classes F all , G, H 1 , H 2 in a two-objective setting. The separation between the Pareto front F(G) and the theoretical optimum F(F all ) can be seen as the "multi-objective bias" incurred in S-MOL due to a conservative choice of G. For two Tchebycheff scalarizations, the red bi-directed arrows in Fig. 1b reflect this point-wise "bias". However, because the Pareto front needs to be learned from finite samples, we would also expect from classical learning theory that as G grows, so does the "multi-objective variance" of an empirical Pareto front F(G). Fig. 1c illustrates this by the gap between F(G) and F(G), and for the same two Tchebycheff scalarizations, the red bi-directed arrows reflect the excess s-trade-off. In the next section we first address how much excess trade-off any algorithm necessarily incurs when learning F(G) from data.
this section cite: []

Section: Motivating Bregman losses: A hardness result
To answer this in the context of a semi-supervised setting, we now argue that for the unlabeled data to help solve S-MOL, the structure of the loss functions is key.
this section cite: []

Section: A sample complexity lower bound for ideal semi-supervised S-MOL
Let us consider the class of PAC-learners for S-MOL, which are learners that achieve S-MOL for all distributions over X × Y. For concreteness, consider multi-objective binary classification with zero-one loss, where S is the entire family of linear scalarizations S lin : Definition 3 (Binary classification). Let G be a hypothesis class with VC dimension
d G ∈ N on a data domain X × Y where Y = {0, 1}. For each task k ∈ [K], define ℓ k (y, y) = 1{y ̸ = y}.
For supervised S-MOL with ε s ≡ ε for all s ∈ S lin , prior works achieve a sample complexity upper bound of O(Kd G /ε 2 ), up to logarithmic terms, see [19,59] and Corollary A.1. In fact, a matching lower bound of Ω(Kd G /ε 2 ) holds as well: after all, the set of s-trade-offs {T s : s ∈ S lin } contains the individual excess risk functionals E k , and hence solving S-MOL requires the learner to solve the K original tasks as well. The lower bound then follows from standard agnostic PAC-learning [57, Theorem 6.8]. In short, previous upper bounds are tight and the sample complexity of supervised S-MOL is Θ(Kd G /ε 2 ), which also coincides with the sample complexity of MDL under non-adaptive sampling [73]. In the semi-supervised S-MOL setting, the question now becomes: can the unlabeled data reduce the label complexity of this problem? Perhaps surprisingly, we now show that the same lower bound holds, even if the learner has additional access to Bayes optimal classifiers f ⋆ k and the marginal distributions P k X . Proposition 1 (Hardness of semi-supervised multi-objective binary classification). Fix any K > 1 and any ε ∈ (0, 1/12). For a given tuple (P 1 , . . . , P K ), denote by S k a labeled dataset consisting of i.i.d. samples from P k , let f ⋆ k be a Bayes optimal classifier of P k , and let P k X be the marginal distribution on X . Denote by A any algorithm that, given {S k , f ⋆ k , P k X } K k=1 , returns a set of classifiers { g s ∈ G : s ∈ S lin }. If A achieves (S-MOL) with ε s ≡ ε for all linear scalarizations s ∈ S lin , δ ≤ 1/6, and for all distributions (P 1 , . . . , P K ) in the multi-objective binary classification setting (Definition 3), then the total number of labeled samples it requires is at least
|S 1 | + • • • + |S K | ≥ CKd G /ε 2 where C > 0 is a universal constant.
See Appendix D.1 for the proof. Proposition 1 shows that the label sample complexity lower bounds for supervised S-MOL cannot be improved for the problem in Definition 3-even in an idealized semi-supervised S-MOL setting where the learner has infinite unlabeled data and can perfectly solve the individual learning tasks. This effect is due to the zero-one loss not being a proper scoring rule, which is necessary for weighing the risks of two different tasks against each other. And indeed, other losses, such as the hinge or absolute deviation loss, suffer from the same problem. See also [62] for a discussion of calibration in multi-objective learning. In our main results, we show that this lower bound can be circumvented in learning settings where the loss functions are proper in this sense.
this section cite: ['b18', 'b58', 'b72', 'b61']

Section: Bregman divergence losses and a pseudo-labeling algorithm
In this section, we introduce Bregman losses and their key property that allows us to leverage unlabeled data and alleviate labeled sample complexity via a pseudo-labeling algorithm. Definition 4 (Bregman loss). Let Y be convex. A loss ℓ : Y × Y → [0, ∞] is called a Bregman loss if there is a strictly convex and differentiable potential ϕ : Y → R such that ℓ(y, y) = ϕ(y)ϕ( y) -⟨∇ϕ( y), y -y⟩.
Many standard prediction losses are Bregman losses. For example, the squared loss can be obtained by setting ϕ(y) = ∥y∥ 2 2 , the logistic loss by choosing ϕ(y) = y log y + (1y) log(1y), and the Kullback-Leibler divergence using ϕ(y) = q k=1 y j log y j . As we now show, an important fact that we will leverage about learning with a Bregman loss is that the associated excess risk functional can be expressed in terms of its minimizer. To state it precisely, we introduce the notions of population and empirical risk discrepancies of a function f ∈ F all from some h ∈ H k , defined as:
d k (f ; h) := E ℓ k (h(X k ), f (X k )) , d k (f ; h) := 1 N k N k i=1 ℓ k (h( X k i ), f ( X k i )).(7)
We further define for some h 1 ∈ H 1 , . . . , h K ∈ H K and h = (h 1 , . . . , h K ) the population and empirical scalarized risk discrepancies of a function f ∈ F all from h as
d s (f ; h) = s(d 1 (f ; h 1 ), . . . , d K (f ; h K )), d s (f ; h) = s( d 1 (f ; h 1 ), ..., d K (f ; h K )). (8
)
We are now ready to state Lemma 1, proved in Appendix D.2. Lemma 1 (Properties of Bregman losses, based on [7]). For each k ∈ [K], let ℓ k be a Bregman loss with potential ϕ k . If both E[Y k ] and E[ϕ k (Y k )] are finite, then up to almost sure equivalence,
f ⋆ k (•) := arg min f ∈F all R k (f ) = E Y k |X k = • and ∀f ∈ F all , E k (f ) = d k (f ; f ⋆ k ).
Algorithm 1 Pseudo-labeling (PL-MOL)
1: for k ∈ [K] do 2: Compute h k = arg min h∈H k R k (h) 3: end for 4: for s ∈ S do 5: Compute g s = arg min g∈G d s (g; h) 6: end for 7: Return { g s : s ∈ S}.
Along with Eqs. (4) and ( 8), Lemma 1 implies
T s (f ) = d s (f ; f ⋆ ) for f ⋆ = (f ⋆ 1 , . . . , f ⋆ K ).
Note that the second part of Lemma 1 decomposes the risk into a task-specific intrinsic noise and a discrepancy term;
R k (f ) = R k (f ⋆ k ) + d k (f ; f ⋆ k ).
It turns out that Bregman divergences are, up to transformation of the label space, the only loss functions that enjoy such a decomposition (see [28] and Appendix B). This decomposition helps justify the following pseudo-labeling multi-objective learning algorithm (Algorithm 1).
First, we minimize the individual empirical risks R k over H k to obtain h = ( h 1 , . . . , h K ), the set of empirical risk minimizers; thus, we estimate the task-wise Bayes-optimal models f ⋆ k (Line 2). Given Lemma 1, we can then approximate the excess risks E k via the empirical risk discrepancies d k (•; h k ) using unlabeled data. And so, the s-trade-off T s (•) can accordingly be approximated by d s (•; h). The empirical estimate of the Pareto set in G is then given as the minimizer of d s (•; h) in G (Line 5). Note that reusing the covariates of the labeled data in this second step would yield at most a constant gain.
Notice that the second step (Line 5) is equivalent to first pseudo-labeling all unlabeled data using the ERMs, and then passing it to the supervised S-MOL algorithm from [59] ("ERM-MOL", Algorithm 2 discussed in Appendix A.1). Finally, note that from a computational perspective, even if S is not finite, Algorithm 1 can be implemented, e.g., using hypernetworks, see Appendix A.2.
this section cite: ['b6', 'b27', 'b58']

Section: Characterization of models with optimal trade-offs: A variational inequality
The pseudo-labeling method illustrates how one can estimate the s-trade-off solutions g s ∈ G from both labeled and unlabeled data when the losses are Bregman divergences. We now show that Bregman losses also enable characterizing the minimizers g s via a variational inequality in some cases. From this inequality, in turn, we can derive conditions for g s to have a particularly simple representation which sheds some light on why unlabeled data can help. Specifically, under linear scalarizations and convexity, we can show the following result. Lemma 2 (Variational characterization of minimizers). For each k ∈ [K], let ℓ k be a Bregman loss with potential ϕ k . Suppose s = s lin λ is linear with weights λ, and µ s := K k=1 λ k P k X . Denote by ⟨•, •⟩ s the inner product defined as ⟨f, f ′ ⟩ s = ⟨f (x), f ′ (x)⟩ dµ s (x). Then for every non-empty, convex and closed set G ⊆ F all and convex g → T s (g),
T s (g) = inf g ′ ∈G T s (g ′ ) if and only if ∀g ′ ∈ G : K k=1 λ k dP k X dµ s ∇ 2 ϕ k (g)(g -f ⋆ k ), g ′ -g s ≥ 0
where
∇ 2 ϕ k (g) denotes the function x → ∇ 2 ϕ k (g(x)). If G is bounded, such a g ∈ G exists.
Lemma 2 is a direct consequence of Lemma D.6 and Theorem 46 in [69]. From this lemma, we can derive the s-trade-off solutions g s analytically in the special case where G = F all and all potentials are shared ϕ k = ϕ. In that case, since the set of feasible models is unconstrained, the variational inequality in Lemma 2 holds with equality. In particular, the first argument of ⟨•, •⟩ s must vanish, up to µ s -equivalence. Thus, we can deduce that the s-trade-off solution is µ s -a.s. of the form
g s (x) = k∈[K] w k (x)f ⋆ k (x), where w k (x) = λ k dP k X dµ s (x)(9)
so that x → w k (x) is non-negative and k∈[K] w k (x) ≡ 1. In short, the optimal prediction with respect to the s-trade-off on the instance x ∈ X is a convex combination of the individual Bayes optimal labels, cf. [40]. Additionally, if the marginals are shared P k X = P X , then each dP k X /dµ s = 1, so the weights w k are independent of x. However, these are specific settings, and g s does not generally need to take this form. We will later make use of this specific form in Section 4.2.
this section cite: ['b68', 'b39']

Section: Sample complexity upper bounds for pseudo-labeling
We now present uniform and localized upper bounds for Algorithm 1 for Bregman losses in terms of Rademacher complexities. Specifically, we use the coordinate-wise Rademacher complexity of a Y-valued function class H ⊆ F all under distribution P k X with n samples, which is defined as
R k n (H) := E X k 1 ,...,X k n ∼P k X σ11,σ12,...,σnq∼Rad   sup h∈H 1 n n i=1 q j=1 σ ij h j (X k i )   .
We discuss the choice and properties of this Rademacher complexity in Appendix E.2.
this section cite: []

Section: A uniform learning bound
We start with some assumptions on the loss functions ℓ k that we require for our bounds. Assumption 1 (Regularity of the losses). For each k ∈ [K], let ℓ k be a Bregman loss satisfying: 3  • Its associated potential function ϕ k is µ k -strongly convex in Y with respect to ℓ 2 -norm, so that for all y, y ′ ∈ Y, it holds that ℓ k (y,
y ′ ) = ϕ k (y) -ϕ k (y ′ ) -⟨∇ϕ k (y ′ ), y -y ′ ⟩ ≥ µ k 2 ∥y -y ′ ∥ 2 2 .
• The loss is L k -Lipschitz continuous in both arguments with ℓ 2 -norm in R q , that is, for all y, y ′ , y ′′ ∈ Y it holds that |ℓ(y,
y ′ ) -ℓ(y, y ′′ )| ≤ L k ∥y ′ -y ′′ ∥ 2 and |ℓ(y ′ , y) -ℓ(y ′′ , y)| ≤ L k ∥y ′ -y ′′ ∥ 2 .
• The loss is bounded by some constant B k < ∞ as ℓ k ≤ B k .
The boundedness enables the concentration bounds used in our results and is a common assumption, and the strong convexity and Lipschitz continuity enable using a vector contraction inequality from [41], as well as establishing a uniform approximation of the excess risks. Most Bregman losses satisfy Assumption 1 on bounded domains Y, while some (like the logistic loss) require careful treatment of Lipschitz continuity if the gradient is unbounded at the boundary of Y (cf. Corollary C.1 and Lemma E.1). We now state our first main result. Theorem 1. Suppose that Assumption 1 holds. Let S be any class of monotone scalarizations that satisfy the reverse triangle inequality and positive homogeneity in Eq. (6), and let { g s : s ∈ S} be the class of solutions returned by Algorithm 1. Then (S-MOL) holds for any δ ∈ (0, 1) and ε s = s(ε 1 , . . . , ε K ), where each ε k is bounded by
ε k ≤ C k   R k N k (G) + log(K/δ) N k 1/2 + R k n k (H k ) + log(K/δ) n k 1/2   ,(10)
with 3 The norm of the strong convexity and Lipschitz continuity in the first argument can be replaced by an arbitrary norm in Theorem 1. Replacing the norm of the Lipschitz continuity in the second argument in Theorem 1 entails using other vector Rademacher complexities, cf. Appendix E.2. They cannot be replaced in Theorem 2. Moreover, for our results it is sufficient for the Lipschitz property to hold on the range of G.
C k = max{4L k , √ 2B k , L k 24L k /µ k , L k 6B k /µ k }.
Theorem 1 is proved in Appendix D.3. Using VC bounds on the Rademacher complexity (see Lemma E.6), Theorem 1 implies that for VC (subgraph) classes G and H k = H with VC dimensions d G , d H , only O(Kd H /ε 4 ) labeled and O(Kd G /ε 2 ) unlabeled samples are necessary to achieve ε-excess s-trade-off uniformly for all scalarizations. Comparing this with the sample complexity Θ(Kd G /ε 2 ) from Proposition 1, it is apparent that for Bregman losses, Algorithm 1 can alleviate the label complexity of S-MOL significantly when d G ≫ d H , and completely eradicates its dependence on G. It shows that a large complexity of G can be compensated by a large amount of unlabeled data N k , as long as R k N k (G) → 0 for N k → ∞. Also notice that, under Assumption 1, the map g → E k (g) or its domain G can be non-convex, in which case non-linear scalarizations are necessary to reach the entire Pareto front. Theorem 1 applies to many such scalarizations, and in particular, the Tchebycheff scalarizations from Eq. ( 5).
this section cite: ['b40']

Section: A localized learning bound
The analysis in Theorem 1 is crude: it estimates the excess risks on all of H k and G, which is why the global Rademacher complexities appear in the bound and the unusual extra square-root appears. Such an analysis can be overly conservative, and a localized bound can provide much tighter statistical guarantees [9,36]. To facilitate a localized analysis for Algorithm 1, we require some additional assumptions. First of all, we only consider linear scalarizations, that is, S ⊆ S lin from Eq. ( 5), mostly for the following norms to be Hilbert norms: for all k ∈ [K], s ∈ S lin , and f ∈ F all , define ∥f ∥ We also require the following shape, strong convexity and smoothness assumptions. Assumption 2 (Shape, strong convexity and smoothness). Recall that f ⋆ k ∈ arg min f ∈F all R k (f ).
• For all k ∈ [K], the function classes H kf ⋆ k are star-shaped around the origin; for all α ∈ [0, 1], if h ∈ H kf ⋆ k , then αh ∈ H kf ⋆ k . Moreover, the function class G is convex and closed.
• For γ > 0 and all s ∈ S, h ∈ H
1 × • • • × H K , the map g → d s (g; h) -γ ∥g∥ 2 s is convex on G.
• For some ν ∈ (0, ∞), the second and third derivatives of the potentials ϕ k are bounded on Y as sup y∈Y ∇ 2 ϕ k (y) 2 ≤ ν and sup y∈Y ∇ 3 ϕ k (y) 2 ≤ ν in the ℓ 2 -operator norms.
The shape constraints are commonly used in local Rademacher complexity proofs [9], and the strong convexity acts as a "multi-objective Bernstein condition" [9,35]. Moreover, the convexity of G, strong convexity, and smoothness also enable a variational argument that is integral to the bound based on Lemma 2. To operationalize the smoothness, we assume a well-specified setting:
Assumption 3. The minimizer of f → d s (f ; f ⋆ ) over F all is contained in G for all s ∈ S.
Finally, for a refined version of our result, we also require the following norm-equivalence assumption that allows relating errors in ∥•∥ s -norm to errors in ∥•∥ k -norm. Assumption 4 (Norm equivalence). All covariate distributions P k X are absolutely continuous with respect to the mixture distributions K k=1 λ k P k X for all s lin λ ∈ S, and there is a constant η so that ∀k ∈ [K], s lin λ ∈ S : ess sup
dP k X d K k=1 λ k P k X ≤ η 2 < ∞.
Specifically, as proved in Lemma D.7, Assumption 4 is equivalent to imposing ∥•∥ k ≤ η ∥•∥ s for all k ∈ [K] and s ∈ S. Sufficient conditions for Assumption 4 are that all weights of the scalarizations in S are bounded away from zero, or P k X ≪ P j X and ess sup dP k X /dP j X ≤ η 2 for all k, j ∈ [K]. We are now ready to state the localized bound. Recall that f ⋆ k is the Bayes model for the kth task, cf. Eq. (3 (4). The result depends on the Rademacher complexities of the following sets of functions, defined using the balls B ∥•∥ k = {f ∈ F all : ∥f ∥ k ≤ 1} as
this section cite: ['b8', 'b35', 'b8', 'b8', 'b34']

Section: ), and for any
h = (h 1 , . . . , h K ) ∈ H 1 × • • • × H K , define g h s := arg min g∈G d s (g; h), so that g s = g f ⋆ s , cf. Eq.
H k (r) := (H k -f ⋆ k ) ∩ rB ∥•∥ k and G k (r; h) := s∈S (G -g h s ) ∩ rB ∥•∥ k .(11)
The excess s-trade-off is bounded in terms of the following critical radii, defined for each k ∈ [K] as
l k = inf r ≥ 0 : r 2 ≥ R k n k (H k (r)) and u k = inf r ≥ 0 : r 2 ≥ R k N k (G k (r; f ⋆ )) . (12
)
Critical radii like these are the key quantities of localized generalization bounds [9,36]. They can be bounded using VC dimension (Lemma E.6) or with (generic) chaining [22,61]. We define the worstcase critical radius in G by replacing f ⋆ in the definition of u k with a supremum over ground-truth functions h, ūk := sup h∈H1×•••×H K inf r ≥ 0 : r 2 ≥ R k N k (G k (r; h)) , and then clearly u k ≤ ūk . Theorem 2. Let S ⊆ S lin be a set of linear scalarizations, and let Assumptions 1 to 3 hold. Then, if δ > 0 is sufficiently small, the output { g s : s ∈ S} from Algorithm 1 satisfies (S-MOL) with probability 1δ and ε s = s(ε 1 , . . . , ε K ), where
ε k ≲ C k ū2 k + l 2 k + N -1 k + n -1 k log(4K/δ)(13)
and 4 If additionally Assumption 4 holds, then for l 2 S = sup s∈S s(l 2 1 , . . . , l 2 K ) and n S = (sup s∈S s(1/n 1 , . . . , 1/n K )) -1 we have
C k = ( ν 3 (1+diam ∥•∥ 2 (Y)) /γ 2 ) max{ L 2 k/γ 2 + B k/γ, L 2 k/µ 2 k + B k/µ k }.
ε k ≲ C k u 2 k + l 2 S + (N -1 k + n -1 S ) log(4K/δ) ,(14)
with
C k = C k • ( ην /γ) 2 max k∈[K] B k/µ k + L 2 k/µ 2 k .
The proof of Theorem 2 can be found in Appendix D.4. By comparing Eq. ( 10) with Eq. ( 13), we can see that, under the additional assumptions, Theorem 2 yields much better rates than Theorem 1, whenever the critical radii are (much) smaller than the global Rademacher complexities. Effectively, Theorem 1 provides a "slow rate" analysis, while Theorem 2 provides a "fast rate" analysis. Additionally, Theorem 2 avoids the "doubly slow rate" R k n k (H k ) 1/2 that appears in Theorem 1, and hence can potentially yield a speed-up of power 4 over Theorem 1; e.g., if H has VC (subgraph) dimension d H , the label complexity reduces to order O(Kd H /ε) compared to the O(Kd H /ε 4 ) from Theorem 1.
In the setting where the algorithm has access to the marginals {P k X } K k=1 , called the ideal semisupervised setting [70], the proof of Theorem 2 also yields a slightly tighter bound than (13) (by combining Eqs. (32) and (36)). Under Assumptions 1 and 2, we obtain
ε s = s(ε 1 , . . . , ε K ) with ε k ≲ C k l 2 k + n -1 k log(2K/δ) , where C k = ν 3 γ 2 1 + diam ∥•∥ 2 (Y) ( B k/µ k + L 2 k/µ 2 k ).
Adaptivity and weakening Assumption 4. While Eq. ( 13) depends on the worst location of the true Pareto set g s in G (through ūk ), Eq. ( 14) refines this bound by also showing the adaptivity of the algorithm to the specific location of the true Pareto set g s in G. Depending on the geometry of G, this set may lie in a "low complexity region" of G. If that is the case, then the radii u k can be smaller than ūk , and the bound adapts to this low complexity. But this comes at a cost: to prove Eq. ( 14), we require the norm equivalence from Assumption 4, and have to replace l k by l S . Intuitively, the distance of g s to g s can only be controlled in the norm ∥•∥ s ; in particular, if λ k = 0, then there is no reason that g s should be close to g s in the norm ∥•∥ k . But u k , defined through ∥•∥ k , has to bound the kth coordinate for all scalarizations s ∈ S, making the norm equivalence from Assumption 4 necessary. For finite sets of scalarizations, on the other hand, this can be avoided (but replaced by a union bound), see Corollary A.2. Hence, there seems to be an inherent tension between controlling the error for all scalarizations simultaneously and proper adaptivity to the local complexity of the problem. It is interesting to explore this tension further.
this section cite: ['b8', 'b35', 'b21', 'b60', 'b69', 'b31', 'b35']

Section: Example: non-parametric regression with Lipschitz functions
We now exemplify the benefit of Theorem 2 in an example where the localized rates are much faster than unlocalized ones. More examples are presented in Appendix C.
Let X = [0, 1], Y = [0, 1] and let ℓ k be the square loss. Define for 0 < L H < L G the function classes
H = {h : [0, 1] → [0, 1] : h is L H -Lipschitz} and G = {g : [0, 1] → [0, 1] : g is L G -Lipschitz}.
Furthermore, let K = 2 and P k X have a density p k on [0, 1] with respect to the Lebesgue measure. For Eq. ( 3) to hold, assume that there exist two functions f ⋆ 1 , f ⋆ 2 ∈ H for which E[Y k |X k = x] = f ⋆ k (x) for all x ∈ [0, 1]. We now apply Theorem 2 to obtain upper bounds for S-MOL in this setting. Corollary 1. Let S ⊆ S lin be a set of linear scalarizations and assume the functions from Eq. ( 9) are L G -Lipschitz. Then the output { g s : s ∈ S} of Algorithm 1 satisfies (S-MOL) with probability 0.99 and ε s = s(ε 1 , . . . , ε K ) where ε k ≲ ( L H /n k ) 2/3 + ( L G /N k ) 2/3 for all s ∈ S. The proof of Corollary 1 can be found in Appendix C.3. Note that we recover the familiar minimax rate n -2/3 of Lipschitz regression. In comparison, the crude, unlocalized bound from Theorem 1 would yield the potentially much slower rates L
1/4 H n -1/4 k + L 1/2 G N -1/2 k .
We illustrate Corollary 1 in Fig. 2 on the following example: Let H be a set of almost constant functions (that is, L H = 0.2), and let f ⋆ 1 ≡ a and f ⋆ 2 ≡ b for two constants a, b ∈ [0, 1]. Minimizing T s (h) for the weights λ = (1/2, 1/2) over H yields the solution h s ≈ (a + b)/2 while for large enough L G , the solution in G becomes g s = (p 1 a + p 2 b)/(p 1 + p 2 ). On the left of Fig. 2, we show one data instance and the resulting models from Algorithms 1 and 2 when the densities are p 1 (x) = 0.7 sin(20x) + 1 and p 2 = 2p 1 . In the center and on the right, we show the excess s-trade-off in this setting as a function of sample size. We can see the rates predicted by Corollary 1: when we fix the unlabeled sample sizes as large enough (N 1 = N 2 = 2 12 ), PL-MOL achieves a small excess s-trade-off already for small labeled sample sizes. Meanwhile, ERM-MOL requires a labeled sample size to be of the same order 2 12 before it achieves a similar excess s-trade-off. At the same time, if we fix the labeled sample size sufficiently large to learn the almost constant functions in H, only PL-MOL improves with an increasing number of unlabeled data. In both cases, the familiar n -2/3 -rate from Lipschitz regression is observable, as also predicted by Corollary 1. Finally, on the right of Fig. 2, we see that if we keep all sample sizes fixed-except for N 1 -, then the rates are eventually bottlenecked by the harder task for all scalarizations; the risks stagnate at λ 2 N -2/3 2 ≍ λ 2 .
this section cite: []

Section: Discussion
This work studies when it is possible to mitigate the statistical cost of multi-objective learning, in which we illuminate the roles of unlabeled data and of the loss functions. This need arises because the function classes that contain models achieving good trade-offs may need to be much larger than those that are well-suited for any one task. We show that for general losses, the label complexity of learning multiple trade-offs simultaneously in a class G is determined solely by the complexity of G, even when the learner has full access to marginal distributions and the Bayes optimal models for each task (Proposition 1). But for Bregman losses, a simple pseudo-labeling algorithm can significantly reduce the label complexity (Theorem 1), where unlabeled data can fully absorb the statistical cost of the expressive model class. Our analysis with local Rademacher complexities further refines these bounds (Theorem 2) and shows adaptivity of the algorithm under some conditions.
The key property that the pseudo-labeling algorithm exploits is the risk decomposition from Lemma 1, which is unique to (generalized) Bregman losses [28]. Nevertheless, it is interesting to determine for exactly which losses the semi-supervised setting can improve upon the supervised one beyond Bregman losses. Under stronger assumptions, we provide a first result of this kind in Appendix B.
Future work may also investigate the tension between controlling the errors of all scalarizations and adaptive rates, and in this context, whether Assumption 4 is really necessary (see discussion in Section 4.2). Moreover, it would be interesting to relax structural assumptions in Theorem 2, e.g., by generalizing it to non-linear scalarizations, and to apply our framework to generative models.
this section cite: ['b27']

Section: References
Ref_id:b0 Title: Minimax regret optimization for robust machine learning under distribution shift Year: (2022)
Ref_id:b1 Title: Infinite dimensional analysis: A hitchhiker's guide Year: (2006)
Ref_id:b2 Title: A framework for learning predictive structures from multiple tasks and unlabeled data Year: (2005)
Ref_id:b3 Title: Open Problem: The Sample Complexity of Multi-Distribution Learning for VC Classes Year: (2023)
Ref_id:b4 Title: Semi-supervised group DRO: Combating sparsity with unlabeled data Year: (2024)
Ref_id:b5 Title: A discriminative model for semi-supervised learning Year: (2010)
Ref_id:b6 Title: On the optimality of conditional expectation as a Bregman predictor Year: (2005)
Ref_id:b7 Title: Clustering with Bregman divergences Year: (2005)
Ref_id:b8 Title: Local Rademacher complexities Year: (2005)
Ref_id:b9 Title: Rademacher and Gaussian complexities: Risk bounds and structural results Year: (2002)
Ref_id:b10 Title: Empirical minimization. Probability theory and related fields Year: (2006)
Ref_id:b11 Title: Convex Analysis and Monotone Operator Theory in Hilbert Spaces Year: (2017)
Ref_id:b12 Title: Does Unlabeled Data Provably Help? Worst-case Analysis of the Sample Complexity of Semi-Supervised Learning Year: (2008)
Ref_id:b13 Title: Collaborative PAC learning Year: (2017)
Ref_id:b14 Title: Bias/Variance is not the same as Approximation/Estimation Year: ()
Ref_id:b15 Title: A short note on an inequality between KL and TV Year: (2022)
Ref_id:b16 Title: Semi-Supervised Learning Year: (2006)
Ref_id:b17 Title: Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance Year: (2023)
Ref_id:b18 Title: Agnostic learning with multiple objectives Year: (2020)
Ref_id:b19 Title: A Probabilistic Theory of Pattern Recognition Year: (1996)
Ref_id:b20 Title: After VaR: The theory, estimation, and insurance applications of quantile-based risk measures Year: (2006)
Ref_id:b21 Title: The sizes of compact subsets of Hilbert space and continuity of Gaussian processes Year: (1967)
Ref_id:b22 Title: Multicriteria Optimization Year: (2005)
Ref_id:b23 Title: ℓ∞ Vector Contraction for Rademacher Complexity Year: (2019)
Ref_id:b24 Title: A decision-theoretic generalization of on-line learning and an application to boosting Year: (1997)
Ref_id:b25 Title: When can unlabeled data improve the learning rate? Year: (2019)
Ref_id:b26 Title: On-demand sampling: Learning optimally from multiple distributions Year: (2022)
Ref_id:b27 Title: Bias-variance decompositions: The exclusive privilege of Bregman divergences Year: (2025)
Ref_id:b28 Title: Multiple objective decision making-methods and applications: A state-of-the-art survey Year: (2012)
Ref_id:b29 Title: Analysis in Banach spaces Year: (2016)
Ref_id:b30 Title: Bias free multiobjective active learning for materials design and discovery Year: (2021)
Ref_id:b31 Title: Multi-objective machine learning Year: (2007)
Ref_id:b32 Title: Multi-objective optimization: Methods and applications Year: (2022)
Ref_id:b33 Title: The statistical analysis of failure time data Year: (2002)
Ref_id:b34 Title: Exponential tail local Rademacher complexity risk bounds without the Bernstein condition Year: (2024)
Ref_id:b35 Title: Local Rademacher Complexities and Oracle Inequalities in Risk Minimization Year: (2006)
Ref_id:b36 Title: Learning bounds for risk-sensitive learning Year: (2020)
Ref_id:b37 Title: Pareto set learning for expensive multi-objective optimization Year: (2022)
Ref_id:b38 Title: Semi-supervised multitask learning Year: (2007)
Ref_id:b39 Title: Domain adaptation with multiple sources Year: (2008)
Ref_id:b40 Title: Bounds for linear multi-task learning Year: (2006)
Ref_id:b41 Title: A vector-contraction inequality for Rademacher complexities Year: (2016)
Ref_id:b42 Title: On the method of bounded differences Year: (1989)
Ref_id:b43 Title: Nonlinear multiobjective optimization Year: (1999)
Ref_id:b44 Title: A comprehensive survey of mixture-of-experts: Algorithms, theory, and applications Year: (2025)
Ref_id:b45 Title: Stacked generalization: An introduction to super learning Year: (2018)
Ref_id:b46 Title: Sequential Approximate Multiobjective Optimization using Computational Intelligence Year: (2009)
Ref_id:b47 Title: Learning the Pareto Front with Hypernetworks Year: ()
Ref_id:b48 Title: Characterizations of inner product spaces by strongly convex functions Year: (2011)
Ref_id:b49 Title: Towards empirical process theory for vector-valued functions: Metric entropy of smooth function classes Year: (2023)
Ref_id:b50 Title: The sample complexity of multi-distribution learning Year: (2024)
Ref_id:b51 Title: Generalization error bounds in semi-supervised classification under the cluster assumption Year: (2007)
Ref_id:b52 Title: Optimization on Pareto sets: On a theory of multi-objective optimization Year: (2023)
Ref_id:b53 Title: Scalable Pareto front approximation for deep multi-objective learning Year: (2021)
Ref_id:b54 Title: Learning with labeled and unlabeled data Year: (2001)
Ref_id:b55 Title: A gentle introduction to empirical process theory and applications Year: (2018)
Ref_id:b56 Title: Understanding machine learning: From theory to algorithms Year: (2014)
Ref_id:b57 Title: Quantile risk control: A flexible framework for bounding the probability of high-loss predictions Year: ()
Ref_id:b58 Title: Generalization in Multi-Objective Machine Learning Year: (2024)
Ref_id:b59 Title: Sharper bounds for Gaussian and empirical processes Year: (1994)
Ref_id:b60 Title: The generic chaining: Upper and lower bounds of stochastic processes Year: (2005)
Ref_id:b61 Title: On calibration in multi-distribution learning Year: (2025)
Ref_id:b62 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b63 Title: Conditional language policy: A general framework for steerable multi-objective finetuning Year: (2024)
Ref_id:b64 Title: Learning Pareto manifolds in high dimensions: How can regularization help? Year: ()
Ref_id:b65 Title: Stacked generalization Year: (1992)
Ref_id:b66 Title: Local Rademacher complexitybased learning guarantees for multi-task learning Year: (2018)
Ref_id:b67 Title: Festschrift for Lucien Le Cam: research papers in probability and statistics Year: (1997)
Ref_id:b68 Title: Nonlinear Functional Analysis and its Applications Year: (1985)
Ref_id:b69 Title: Semi-supervised inference: General theory and estimation of means Year: (2019)
Ref_id:b70 Title: Random hypervolume scalarizations for provable multi-objective black box optimization Year: (2020)
Ref_id:b71 Title: An overview of multi-task learning Year: (2018)
Ref_id:b72 Title: Optimal multi-distribution learning Year: (2024)
Ref_id:b73 Title: e-PAL: An active learning approach to the multi-objective optimization problem Year: (2016)
Ref_id:b74 Title: Active learning for multi-objective optimization Year: (2013)
Ref_id:b75 Title: Can semi-supervised learning use all the data effectively? A lower bound perspective Year: ()
