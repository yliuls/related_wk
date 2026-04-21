Title: A Principled Approach to Randomized Selection under Uncertainty: Applications to Peer Review and Grant Funding
Abstract: Many decision-making processes involve evaluating and selecting items, including scientific peer review, job hiring, school admissions, and investment decisions. These domains feature error-prone evaluations and uncertainty about outcomes, which undermine deterministic selection rules. Consequently, randomized selection mechanisms are gaining traction. However, current randomized approaches are ad hoc and, as we prove, inappropriate for their purported objectives. We propose a principled framework for randomized decision-making based on interval estimates of item quality. We introduce MERIT (Maximin Efficient Randomized Interval Top-k), which maximizes the worst-case expected number of top candidates selected under uncertainty represented by overlapping intervals. MERIT provides optimal resource allocation under an interpretable robustness notion. We develop a polynomial-time, practically efficient algorithm and prove our approach satisfies desirable axiomatic properties not guaranteed by existing methods. Experiments on synthetic peer review data from grant funding and conferences demonstrate that MERIT matches existing algorithms' expected utility under fully probabilistic models while outperforming them under our worst-case formulation.

Section: Introduction
In many applications like scientific funding, job hiring, school admissions, and startup investment, decision makers evaluate and select items based on imperfect assessments. Recently, there has been growing interest in introducing randomization into selection processes to address uncertainty in evaluations, reduce reviewer burden, encourage high-risk proposals, and combat reviewer partiality. In fact, many funding agencies have already adopted partial lotteries to allocate grant money, starting with the New Zealand Health Research Council in 2013 [31], followed by the Swiss NSF in 2019 [1], and recently expanding to numerous agencies worldwide [14,50,58]. Proposals for randomization have also emerged in college admissions [1,25], job screening [40,8] and startup investment [35].
In current deployments, decision makers collect peer-review assessments and run a lottery where selection probabilities derive from review scores. However, current procedures are ad hoc. We initiate a principled approach to randomizing decisions from evaluations, focusing on the question: Given imperfect evaluations of candidate quality, what is a suitable probability distribution over applicants for random selection?
We take the perspective of a funder selecting the highest quality grant proposals. A key motivation for randomization is uncertainty about relative quality. Many existing deployments describe peer review lotteries as random "tie-breaking" between proposals of equal quality [36,53]. Similar to previous work [21], we assume the funder estimates numeric quality intervals for each proposal. These intervals may capture uncertainty due to evaluation errors, miscalibration, subjectivity, or aleatoric uncertainty about future success. Crucially, we capture settings with "Knightian uncertainty," where a funder cannot assign a probability measure over possible outcomes [26]-a model widely applied in policy-making [52,45,7], financial investment [13,33,37], and R&D investment [3].
This assumption is particularly apt for peer review, where probabilistic models of reviewer errors have performed poorly in real deployments [27, 51, Section 'Miscalibration'], decision makers lack ground truth data to evaluate model appropriateness, and future success is inherently difficult to predict [49,12,60,29]. Without a reliable probabilistic model, we assume decision makers describe uncertainty through quality intervals rather than point estimates. The funder draws conclusions only about relative ordering: overlapping intervals indicate insufficient evidence to conclude one proposal is better, while non-overlapping intervals indicate dominance. We develop theory and a practical algorithm for making funding decisions from interval quality estimates, with the following key contributions:
1. Modeling uncertainty as Knightian uncertainty intervals: Prior work assumes known probabilistic relationships between quality and scores [21]. We show that in such fully Bayesian settings, deterministic selection always maximizes expected utility, making randomization unnecessary.
Our model captures the motivation for randomizing using "Knightian" uncertainty intervals. 2. A principled approach to randomization: We formalize two key principles: ex ante optimality-maximizing worst-case utility over all rankings consistent with intervals-and ex post validity-respecting strict dominance relationships. Prior heuristic rules fail to satisfy both principles. 3. Efficient algorithm: We develop a polynomial time algorithm solving the maximin optimization problem, despite related graph problems being NP-hard [15,42,61,2]. Our Maximin Efficient Randomized Interval Top-k (MERIT) algorithm runs in under 5 minutes on 10,000+ candidates on a standard laptop. Implementation available at github.com/akgoldberg/lottery. 4. Axiomatic comparison: We initiate an axiomatic approach to comparing randomized mechanisms, identifying desirable properties of "monotonicity in budget", "stability", and "reversal symmetry."
We prove MERIT prevents "maximal instability" and respects "reversal symmetry" while existing mechanisms do not. 5. Empirical comparison: We evaluate MERIT against existing methods using synthetic data based on real peer review data from conferences (NeurIPS 2024, ICLR 2025) and grant agencies (Swiss NSF 2020). MERIT performs comparably to existing methods in expected utility under a linear reviewer error model used by the Swiss NSF [21] and many other prior works [16,5,43,44]. However, under our worst-case objective, our algorithm significantly outperforms deterministic selection and the Swiss NSF's randomized approach.
For clarity of presentation, all formal proofs are deferred to Appendix A.
this section cite: ['b30', 'b0', 'b13', 'b49', 'b57', 'b0', 'b24', 'b39', 'b7', 'b34', 'b35', 'b52', 'b20', 'b25', 'b51', 'b44', 'b6', 'b12', 'b32', 'b36', 'b2', 'b48', 'b11', 'b59', 'b28', 'b20', 'b14', 'b41', 'b60', 'b1', 'b20', 'b15', 'b4', 'b42', 'b43']

Section: Background and Approach
We begin by describing current deployments of randomized decisions in scientific funding and then motivate our approach.
this section cite: []

Section: Existing Deployments
In recent years, there have been many deployments of "peer review lotteries" in scientific funding decisions. Most deployments use an approach of "randomize-above-threshold." Under this approach, the funder chooses a minimum acceptable quality threshold and samples uniformly at random among all proposals that are above this threshold. This approach has been adopted by numerous funding agencies [14,55,58,50,31] Council, the British Academy [55] and as a means of allocating oral presentations at the USENIX Security Conference [57]. However, as we describe in Section 3, randomize-above-threshold may violate a desired principle of "ex post validity", which says that if one proposal clearly dominates another, the stronger proposal should be funded if the weaker proposal is funded.
Taking a different approach, the Swiss National Science Foundation (NSF) [21,56] pioneered a method that explicitly accounts for uncertainty about the quality of each proposal. They assume that each proposal has a latent true quality and review scores are generated based on these quality scores and reviewer-specific noise parameters. They assume priors on the model parameters and develop methods to obtain point estimates and confidence intervals for the true quality of each proposal. As described in Algorithm 8, the Swiss NSF then samples k proposals by setting a "provisional funding line" as the k-th highest point estimate. All proposals with intervals strictly above the funding line are selected and all proposals strictly below the funding line are rejected. The remaining budget is allocated uniformly at random among proposals with intervals that overlap the funding line. We provide pseudocode for the Swiss NSF method and randomize-above-threshold in Appendix E.
The intuition provided by the Swiss NSF for their approach is that confidence intervals capture the funder's uncertainty in estimating the proposal quality and randomizing decisions accounts for this uncertainty, thereby leading to better decisions. In contrast, as we prove in Theorem A.1, when the funder assumes that review data is generated from a fully specified Bayesian model, there exists a deterministic selection of proposals that maximizes the funder's expected utility for any utility function. Informally, if the funder knows the model that generates their data, then they do not need to randomize in order to maximize their expected utility. We present the formal proposition and proof in Appendix A.1. This result suggests one drawback of the Swiss NSF's method-in their model setting there may be a utility cost to randomization compared to choosing deterministically in an optimal manner. A second drawback is that the Swiss NSF's algorithm for selecting proposals from intervals violates natural axioms for a selection rule, like monotonicity in the budget k and stability, as we show in Section 5.
this section cite: ['b13', 'b54', 'b57', 'b49', 'b30', 'b54', 'b56', 'b20', 'b55']

Section: Our Approach
In our work, we propose a model that captures the motivation for randomizing due to uncertainty about the relative quality of the proposals. We show that if the funder cares about their worst case utility they must randomize decisions in order to robustly optimize their utility.
Specifically, we consider a funder who estimates intervals for each proposal based on data. These intervals need not come from any one particular model, but they should capture the funder's inherent uncertainty about the relative quality of proposals.
Our interpretation of the quality intervals stems from the intuitive argument that if the intervals for two proposals overlap then the funder does not have enough evidence to distinguish between them. On the other hand, if the interval of proposal A dominates the interval of proposal B, then the funder has sufficient evidence to believe that A is better than B. Hence, the intervals define a partial ordering of proposals that represents a set of conclusions by the funder regarding the relative quality of different proposals. This ordering is the canonical "interval order" for a set of intervals. It captures the spirit of how the Swiss NSF interprets confidence intervals in their setup citeheyard2022rethinking.
In practice, such intervals can arise from a variety of sources, including missing-data imputation, model ensembling, expert input, multi-criteria aggregation, or robustness to model misspecification for confidence intervals. We provide detailed explanations of such intervals in Appendix F.
this section cite: []

Section: Problem Formulation
Our method applies to settings like admissions, scientific peer review, job screening, and financial investment, where decision makers estimate quality intervals and select top candidates based on these intervals. For concreteness, throughout our exposition, we will describe a funder choosing proposals.
Consider a funder who receives n proposals. From these, the funder wishes to select the k highest quality proposals. Note that n could be as large as thousands of proposals and k a fixed fraction of the total and can also be in the hundreds or thousands. For each proposal i ∈ [n]foot_0 , the funder estimates an interval [ℓ i , u i ] ⊆ R representing a range of quality scores that the proposal could possibly take. A higher score indicates higher quality. The funder wishes to design a randomized selection mechanism to choose k proposals given the intervals. In order to design such a mechanism, we adopt two primary principles which we term as ex ante optimality and ex post validity, described below.
Ex ante optimality The funder's utility is the expected number of the true top-k proposals that they select, ranked by their quality. Formally, let σ : [n] → [n] denote a ranking of the proposals where for each proposal i ∈ [n], the rank of the proposal is denoted by σ(i) ∈ [n]. If the funder samples proposals with marginal probabilities p ∈ [0, 1] n and the true ranking of proposals is σ, their expected utility is n i=1 p i 1{σ(i) ≤ k}. Clearly, if the funder knew the true ranking σ, then they would optimize utility by choosing deterministically, i.e., by setting p i = 1 for i with σ(i) ≤ k, and 0 otherwise.
However, recall that the funder is uncertain about relative qualities of the proposals, as captured by overlaps in intervals. Any ordering that is consistent with overlapping intervals could be the true ranking. Hence, the intervals define a set of feasible rankings:
Σ n = {σ permutation of [n] | ∀i, j ∈ [n], ℓ i > u j =⇒ σ(i) < σ(j) }
In other words, if proposal i has quality strictly above proposal j, then i is ranked higher than j in all σ ∈ Σ n . As an example of an extreme case, if all n intervals overlap each other, then Σ n consists of all possible permutations of the proposals.
For ex ante optimality, the funder optimizes their worst case utility over feasible rankings Σ n :
min σ∈Σn n i=1 p i 1{σ(i) ≤ k}(1)
The funder maximizes their worst-case expected utility by choosing the optimal marginal probabilities p solving the maximin optimization problem:
max p∈[0,1] n : ∥p∥1=k min σ∈Σn n i=1 p i 1{σ(i) ≤ k}.(2)
Finally, the funder randomly chooses n proposals with marginal probabilities corresponding to p.
The ex ante optimization problem has a game theoretic interpretation that motivates the need for randomization. Our model corresponds to a zero-sum Stackelberg game where the funder is the "leader" who selects k proposals. The funder faces an adversarial "follower" who chooses a ranking of proposals. The leader's utility is the number of top k proposals selected based on the adversary's ranking, while the adversary's utility is the negation of the leader's. The Strong Stackelberg Equilibrium (SSE) is exactly the solution to Objective 2. It is well known that in an SSE, the leader may need to commit to a randomized (or mixed) strategy.
this section cite: []

Section: Ex post validity
The ex post validity criterion requires that for any pair of proposals a and b, if b's quality interval lies strictly below a's interval and if b is selected, then a must also be selected.
Formally, a selection rule that takes as input a set of quality intervals I and outputs a set of selected proposals S satisfies ex post validity, if for all pairs of intervals a, b ∈ I with ℓ a > u b , and all outputs S selected with non-zero probability, b ∈ S =⇒ a ∈ S.
The ex post validity criteria ensures that the actual selected set of proposals is legitimate to stakeholders. In particular, if the funder rejects a proposal that dominates an accepted proposal, that would be unacceptable to the funder and to applicants.
While the ex post condition seems natural, the simple randomize-above-threshold mechanism can violate it: Suppose proposals a and b both lie above the threshold, but a dominates b. Because a and b are entered into a uniform lottery, a may be rejected at random, while b is accepted at random.
this section cite: []

Section: Efficient Algorithm
The ex ante optimization problem in (2) is equivalent to solving the following linear program (LP):
max p∈R n ,v∈R v (3
) subject to v ≤ n i=1 p i 1{σ(i) ≤ k}, ∀σ ∈ Σ n , n i=1 p i = k and 0 ≤ p i ≤ 1, ∀i ∈ [n]
This LP has exponentially many constraints ( n k ). The minimum weight k-ideal problem on arbitrary partial orders is NP-hard [15], and several related interval order problems are also NP-hard [2,42,61]. Despite this, we show the problem is solvable in polynomial time using the ellipsoid method with a separation oracle (Section 4.1). We develop a practical cutting plane algorithm (Section 4.2) and show how to ensure ex post validity efficiently (Section 4.3). We call this end-to-end algorithm Maximin Efficient Randomized Interval Top-k (MERIT).
this section cite: ['b14', 'b1', 'b41', 'b60']

Section: Polynomial Time Algorithm
We now develop a polynomial-time algorithm to solve linear program (3). Our approach solves the problem using a polynomial time "separation oracle" [18] with the ellipsoid algorithm. A separation oracle checks whether a proposed solution satisfies all the constraints. If the solution is feasible, the oracle confirms it. If not, it identifies (at least one) specific constraint that the solution violates. The separation oracle may be used to solve the LP without enumerating all (exponentially many) constraints by starting with a limited set of constraints and iteratively shrinking the possible feasible region of the LP through calls to the separation oracle. Our main result proves that this method yields a polynomial time algorithm: Theorem 4.1 (Polynomial time solution). The linear program (3) can be solved within accuracy ϵ of the optimal solution in polynomial time with respect to n and log(1/ϵ) using the ellipsoid algorithm with Algorithm 1 as a separation oracle.
this section cite: ['b17']

Section: Algorithm 1 Polynomial-time Separation Oracle
Input: Candidate solution: (p, v) ∈ [0, 1] n × R with ∥p∥ 1 = k, number selected k, set of intervals {ℓ i , u i } i∈[n] sorted in decreasing order of lower bound ℓ i Output: A set of violated constraints (∅ if (p, v) is feasible) 1: Z ← ∅ 2: for i = 1 to k + 1 do 3: S i ← {j ∈ (i, n] : intervals j and i overlap} 4:
if |S i | ≥ (k -(i -1)) then 5:
Obtain Si by sorting S i by p and keeping only the k -(i -1) smallest values 6:
if v > i-1 j=1 p j + j∈ Si p j † 7: Z ← Z ∪ " v ≤ i-1 j=1 p j + j∈ Si p j " then 8: return Z
† By convention, we take the empty sum from j = 1 to 0 to be 0.
The primary technical difficulty is the design of an efficient separation oracle. We present our separation oracle in Algorithm 1. Given a candidate solution (p, v), the separation oracle checks whether p achieves a worst-case objective value of at least v. If the worst-case objective value under p is greater than v, then the solution is feasible, if not the oracle returns a set of violated constraints. At a high level, the oracle works by constructing worst-case possible sets of top-k proposals. For each of the k + 1 intervals with the largest lower bounds, the algorithm constructs the worst-case set of top-k proposals that includes intervals 1 to (i -1) and excludes interval i in the top-k. Excluding an interval with a large lower bound constrains the set of intervals that must be in the top-k, since all intervals strictly below the interval with the large lower bound must be excluded from the top-k.
In considering all such sets of intervals, the algorithm enumerates possible worst-case permutations with respect to p in time O(nk). If the separation oracle finds a permutation that gives objective values smaller than v, it returns this permutation, which corresponds to a violated constraint in the LP. If it does not find any such permutation, then (p, v) is feasible. This separation oracle is used as a sub-routine in the ellipsoid algorithm to compute the optimal solution in polynomial time in Theorem 4.1.
this section cite: []

Section: Lemma 4.2 (Polynomial-time separation oracle).
For any candidate solution (p, v) to the linear program (3), Algorithm 1 returns ∅ only if the candidate solution is feasible and returns a non-empty set of violated constraints otherwise. Moreover, Algorithm 1 runs in time O(n max{k, log n}).
this section cite: []

Section: Practical Algorithm
The cutting plane algorithm is described in full in Algorithm 2. The algorithm starts by solving a relaxation of the LP without any of the worst-case value constraints to find an initial (potentially infeasible) candidate solution (p, v). Then, the algorithm repeatedly calls the separation oracle to check the feasibility of the current candidate solution. If the candidate solution is feasible, it is an optimal solution to the LP, since it is optimal for a relaxation of the full LP. If the candidate solution is infeasible, the algorithm adds the constraints returned by the separation oracle and re-solves the LP.
The cutting plane algorithm converges to a feasible optimal solution quickly in practice because it is initialized with a useful set of constraints on the feasible region of the problem. These constraints prune the problem and impose monotonicity and symmetry constraints on the marginal probabilities p, based on the number of intervals above and below each proposal, which we define below.
this section cite: []

Section: Definition 4.3 (Number above (A) and below (B)).
For each proposal i: A(i) = |{r : ℓ r > u i }| and
B(i) = |{r : ℓ i > u r }|. Definition 4.4 (Monotonically ordered subset). Subset M ⊆ [n] is monotonically ordered if ∀i ∈ [|M | -1], A(M [i]) ≤ A(M [i + 1]) and B(M [i]) ≥ B(M [i + 1]).
Full analysis is in Appendix B.
this section cite: []

Section: Algorithm 2 Cutting Plane Algorithm
Input: Number of proposals k, intervals I = {ℓ i , u i } i∈[n] , max iterations T Output: Ex ante optimal vector p # Prune Intervals 1: For intervals strictly below ≥ k others, set p i = 0 and remove. 2: For intervals strictly above ≥ n -k others, set p i = 1 and remove. 3: Let a = # accepted intervals. Update k ← k -a.
# Initialize Linear Program 4: Compute A(i) = # proposals strictly above i, B(i) = # proposals strictly below i. 5: Using A, B, partition intervals into w monotone subsets M 1 , . . . , M w (Alg. 5). 6: Solve LP to obtain initial p, v:
min v,p v s.t. n i=1 p i = k, p i ∈ [0, 1] ∀i, v ≤ k j=1 p j , p M [i] ≥ p M [i+1] ∀i ∈ [|M | -1], M ∈ {M 1 , . . . , M w } # Add Cuts 7: for T iterations do 8: C ← SeparationOracle((p, v), k, I) 9: if C = ∅ then return p ▷ Feasible 10:
else Add constraints from C to LP and resolve for new (p, v) ▷ Infeasible 11: return Failure
this section cite: []

Section: Enforcing Ex Post Validity
A solution to the ex ante optimality LP (3) returned by the Cutting Plane Algorithm (Algorithm 2) or the Ellipsoid Algorithm, is not guaranteed to output a vector of marginal probabilities, such that sampling proposals with these marginals always guarantees ex post validity. However, we prove that we can post-process any solution to the ex ante optimization problem, and then sample with marginal probabilities p to guarantee ex ante optimality and ex post validity simultaneously. This stands in contrast to the commonly used "randomize-above-threshold" approach to randomization, which does not guarantee ex post validity as described in Section 3. Theorem 4.5 (Post-processing for ex post validity). Given any ex ante optimal p, Algorithm 3 enables the funder to sample k proposals while satisfying both ex ante and ex post conditions and is computable in time O(n 2 ).
Theorem 4.5 applies the post-processing algorithm given in Algorithm 3 to a solution from the Cutting Plane Algorithm. For any a, b ∈ [n] with ℓ a > u b , Algorithm 3 terminates with p a = 1 or p b = 0. Moreover, Algorithm 3 never decreases the objective value of p. Hence, applying post-processing to an ex ante optimal solution is without loss of optimality and ensures that any sampling method that selects proposals with marginal probabilities p satisfies ex post validity.
We then implement the sampling step using systematic sampling [32] which has runtime O(n) (described in Appendix C).
this section cite: ['b31']

Section: Algorithm 3 Post-Processing of p for Ex Post Validity
Input: Vector of marginal probabilities p, sequence of intervals {[ℓ i , u i ]} i∈[n] Output: Vector of marginal probabilities p 1: Order the intervals by increasing u. if ℓ a > u b and p a < 1 then 7:
d ← min{p b , 1 -p a } 8: p b ← p b -d 9: p a ← p a + d
this section cite: []

Section: Full Algorithm
The complete MERIT algorithm solves the ex ante optimization with post-processing for ex post validity (Algorithm 3) followed by sampling. The algorithm is provably polynomial time using the ellipsoid method (Theorem 4.1). In practice, we use the cutting plane algorithm (Algorithm 2), which is efficient albeit with non-polynomial theoretical convergence.
this section cite: []

Section: Algorithm 4 MERIT Algorithm
Input: Number of proposals to select k, set of intervals I = {ℓ i , u i } i∈[n] Output: Selection of k proposals 1: Compute an ex ante optimal vector of marginal probabilities p using Algorithm 2. 2: Apply ex post validity post-processing to p (Algorithm 3). 3: Sample k proposals from [n] with marginal probabilities of inclusion given by p (Algorithm 6).
this section cite: []

Section: Axiomatic Comparison
In applications like scientific funding or college admissions, there is no agreed upon ground-truth measurement of selection quality. Hence, it is unclear how to empirically measure whether one algorithm performs better than another. Therefore, inspired by social choice theory [10], we initiate an axiomatic comparison of MERIT with alternative methods from Section 2.1-deterministic top-k selection, randomize above threshold, and the Swiss NSF method. We analyze the behavior of MERIT and alternatives with respect to natural axioms.
this section cite: ['b9']

Section: Defining Axioms
We propose three natural desiderata for algorithms selecting proposals from quality assessments. We begin by defining a generic "randomized selection rule."
this section cite: []

Section: Definition 5.1 (Selection rule).
A selection rule receives as input n quality estimates I = {(ℓ i , e i , u i )} i∈[n] where ℓ i , u i are lower and upper limits on item i's quality and e i ∈ [ℓ i , u i ] is a point estimate. Given budget k ∈ {1, . . . , n}, it outputs a subset of [n] of size k. Let p(I, k) ∈ [0, 1] n denote the marginal selection probabilities. This captures methods that do not use intervals (deterministic selection), use only intervals (MERIT), and use both intervals and point estimates (Swiss NSF). Next, we define three axioms.
First, "monotonicity in budget" requires that increasing k should not decrease any proposal's selection probability: Definition 5.2 (Monotonicity in budget). A selection rule respects monotonicity in budget if for any input I and all budgets k ∈ [n -1], p(I, k + 1) i ≥ p(I, k
) i ∀i ∈ [n].
Second, a selection rule should be "stable"-changing one interval should not drastically change algorithm behavior. We define an undesirable form of instability. An algorithm exhibits "maximal instability" if changing a single interval by an arbitrarily small amount can switch behavior between deterministic selection (minimum entropy) and uniform random sampling (maximum entropy): Definition 5.3 (Maximum instability). A selection rule is maximally unstable if there exist inputs I and J differing by arbitrarily small ϵ > 0 in one proposal's quality estimate and budget k ∈ {2, . . . , n -2} such that p(I, k) = k n 1 n (uniform random sampling on I) whereas p(J, k) ∈ {0, 1} n (deterministic on J).
A selection algorithm should avoid maximum instability. We restrict budget to {2, . . . , k -2} since stability with respect to changing a single proposal is not meaningful when choosing only one proposal to accept or reject.
Finally, inspired by "reversal symmetry" from social choice theory [46], when selecting 1 of 2 proposals, if the quality scale is reversed (all intervals flipped), the selection rule should flip the selection probabilities: Definition 5.4 (Reversal symmetry). For input
I = {(ℓ i , e i , u i )} i∈[n] where ℓ i , e i , u i ∈ [0, 1]∀i, let I (R) = {(1 -u i , 1 -e i , 1 -ℓ i )} i∈[n]
be the reversed input. A selection rule selecting k = 1 of n = 2 proposals respects reversal symmetry if for any flipped inputs I and I (R) , p(I, 1) = (p 1 , p 2 ) and p(I (R) , 1) = (p 2 , p 1 ).
this section cite: ['b45']

Section: Theoretical Analysis
Deterministic top-k selection meets these criteria but does not account for uncertainty (violating our ex ante requirement). We characterize randomized mechanisms in Theorem 5.5: Theorem 5.5 (Axiomatic analysis). Existing randomized algorithms have the following properties:
(a) Swiss NSF and randomize-above-threshold both exhibit maximum instability, while MERIT is never maximally unstable.
(b) Swiss NSF, randomize-above-threshold and MERIT all violate monotonicity in budget.
(c) It is not possible to simultaneously satisfy ex ante optimality and monotonicity in budget.
(d) Swiss NSF and randomize-above threshold violate reversal symmetry, while MERIT satisfies reversal symmetry.
We formally prove Theorem 5.5 in Appendix 5.5. While all randomized selection rules considered vioalte monotonicity, a funder could enforce monotonicity in budget for MERIT by solving a sequence of optimization problems from 1 to k, but this may come at loss of ex ante optimality, as we describe in Appendix D.
this section cite: []

Section: Experimental Comparison of Methods
We evaluate MERIT using real peer review data from the Swiss NSF grant reviews [21], NeurIPS 2024 conference papers, and ICLR 2025 submissions. We compare performance under both (1) expected utility in a probabilistic model of reviewer behavior and (2) our worst-case utility objective. We provide additional ablations and qualitative insights into different lottery outcomes in Appendix G.   Expected Utility under Linear Miscalibration We evaluate expected utility under the Swiss NSF's widely-adopted linear miscalibration model [21,16,5], where reviewer scores are linear combinations of true quality, reviewer bias, and Gaussian noise. We simulate two settings: (1) Swiss NSF grants (350 proposals, 10 reviewers, 80 proposals/reviewer) and (2) CS conference review (1000 papers, 1000 reviewers, 5 papers/reviewer), selecting the top one-third in both cases. We generate 50 synthetic datasets with parameters σ θ = 2, σ b = 1, σ ϵ = 0.5 matching prior work [54], and estimate 50% confidence intervals following Swiss NSF methodology.
Worst-Case Utility Our worst-case objective (1) maximizes the minimum expected fraction of true top-k proposals over all rankings consistent with score intervals. We evaluate on three real datasets: Swiss NSF 2020 grants (n = 353), NeurIPS 2024 accepted papers (n = 4035), and ICLR 2025 submissions (n = 11520). For Swiss NSF, we use their linear model intervals. For NeurIPS and ICLR, we generate intervals using three methods: (1) leave-one-out (LOO) ranges, (2) Bayesian credibility intervals under a Gaussian model, and (3) subjectivity-adjusted intervals based on reviewer emphasis on different criteria [28,38]. Full details are in Appendix F.
this section cite: ['b20', 'b20', 'b15', 'b4', 'b53', 'b27', 'b37']

Section: Results
Figure 1 shows our main results. Under the Swiss NSF's Bayesian model, MERIT performs comparably to Swiss NSF's randomized method and deterministic model-based selection in expectation, while deterministic mean-based selection performs poorly. In the worst-case setting, MERIT substantially outperforms all baselines across all datasets, achieving up to 0.19 higher utility than Swiss NSF. The optimality gap is particularly large when intervals are wide (e.g., NeurIPS LOO, NeurIPS Gaussian), where deterministic selection and even Swiss NSF's randomized approach perform poorly. These results suggest MERIT maintains expected utility under probabilistic models while providing superior worst-case robustness. We provide additional ablations and experiments in Appendix G with similar results.
this section cite: []

Section: Computational Efficiency
All experiments run on a standard 2019 MacBook Pro. Using Gurobi 12.0.1 [20] to solve LPs, MERIT completes in under five minutes even for 11,000+ proposals. Detailed runtime analysis is in Appendix H.
this section cite: ['b19']

Section: Related Work
Designing Better Evaluation Processes Many works diagnose flaws in peer review-such as miscalibration and arbitrariness in opinions of reviewers-and propose improvements [51]. Many models assume linear miscalibration [16,5,43,44,48,12], including those used in the Swiss NSF [21], but these often perform poorly in practice [27], likely due to the complexity of realworld miscalibration [11]. A recent approach [59] addresses arbitrary miscalibration and shows that randomized estimators can improve ranking accuracy. Related work in hiring proposes algorithmic solutions to prejudice and to uncertainty about unknown options [24,30]. Both approaches make stronger assumptions on a known model of errors in hiring process than our work.
this section cite: ['b50', 'b15', 'b4', 'b42', 'b43', 'b47', 'b11', 'b20', 'b26', 'b10', 'b58', 'b23', 'b29']

Section: Selection Under Uncertainty and Robust Optimization
Our method falls under the umbrella of distributionally robust optimization (DRO) [41]. Our main contribution is a formulation specific to scenarios where decision makers want to select k high quality items under ambiguity sets defined by intervals. Though similar in spirit to robust portfolio optimization [62], our assumptions differ as investors typically have probabilistic models and seek risk-averse diversification. Recent work on conformal prediction for selection [22] focuses on bounding false positives above a fixed threshold, whereas we select the top-k items without assuming a fixed quality threshold or reliable probabilities.
Randomized Social Choice Our work connects to the literature on randomized social choice, which examines how incorporating randomness into voting rules can enhance desired properties [10,9]. Unlike our setting, most social choice models assume no underlying ground-truth quality. A related line of work in social choice theory on "distortion" interprets voters' rankings as noisy reflections of latent utilities and seeks aggregation rules that perform well under this uncertainty [4]. Our approach is conceptually similar: we consider rankings consistent with intervals of quality and design a randomized mechanism that maximizes the worst-case expected utility of the selected top-k items.
this section cite: ['b40', 'b61', 'b21', 'b9', 'b8', 'b3']

Section: Discussion
We introduce MERIT, a computationally efficient lottery for top-k selection under uncertainty. By relying solely on intervals rather than fully specified generative models, MERIT respects real-world funder constraints while providing a principled robust optimization solution. Our case studies demonstrate scalability to tens of thousands of candidates. An additional benefit of MERIT is that it can handle additional constraints on the form of the lottery. For example, in some cases, funders may prefer to implement a uniform lottery, where all candidates subject to randomization are selected with equal probabilities. The optimization problem used in MERIT can be modified to constrain the lottery to uniform sampling, thereby implementing the ex ante optimal uniform lottery that respects ex post validity constraint (discussed in Appendix I).
Limitations. MERIT exploits only interval ordering, appropriate when credible probabilistic models are unavailable but potentially sub-optimal with well-calibrated predictive models. We assume fixed proposal costs; variable budgets require accounting for both quality and budget allocation. Additional pairwise quality information (e.g., from common reviewers) is not currently exploited but could improve performance. Finally, MERIT may be less interpretable than threshold-based methods like Swiss NSF.
Future work. Several extensions merit investigation. First, variable-cost candidates: allowing partial funding or incorporating cost-quality trade-offs in budget-constrained settings, with attention to incentive compatibility. Second, additional ordering information: developing efficient heuristics for general partial orders beyond interval orders, perhaps via additional monotonicity constraints. Third, richer utility functions: extending beyond 0-1 utility to positional scoring rules (e.g., Borda count) while maintaining tractability. Fourth, cost-quality trade-offs: analyzing efficient trade-offs between reviewer resources and decision quality to guide when and how to randomize. Finally, equilibrium effects: understanding how randomization alters applicant and reviewer incentives and strategic behavior. Progress on these questions can inform both theoretical understanding and practical implementation of randomized selection mechanisms.
Progress on these questions can inform both theoretical understanding and practical implementation of randomized selection mechanisms.
this section cite: []

Section: References
Ref_id:b0 Title: Science funders gamble on grant lotteries Year: (2019)
Ref_id:b1 Title: Complexity of maximum cut on interval graphs Year: (2023)
Ref_id:b2 Title: R&d profitability: the role of risk and knightian uncertainty Year: (2017)
Ref_id:b3 Title: Distortion in social choice problems: The first 15 years and beyond Year: (2021)
Ref_id:b4 Title: Statistical quality estimation for general crowdsourcing tasks Year: (2013)
Ref_id:b5 Title: Predictive inference with the jackknife+ Year: (2021)
Ref_id:b6 Title: Decision making in times of knightian uncertainty: An info-gap perspective Year: (2016)
Ref_id:b7 Title: Focal random selection closes the gender gap in competitiveness Year: (2020)
Ref_id:b8 Title: Consistent probabilistic social choice Year: (2016)
Ref_id:b9 Title: Handbook of Computational Social Choice Year: (2016)
Ref_id:b10 Title: Modeling patterns of probability calibration with random support theory: Diagnosing case-based judgment Year: (2005)
Ref_id:b11 Title: Inconsistency in conference peer review: Revisiting the 2014 NeurIPS experiment Year: (2021)
Ref_id:b12 Title: Intertemporal asset pricing under knightian uncertainty Year: (2004)
Ref_id:b13 Title: Peer review -not perfect but still the best system that we have Year: (2023-11)
Ref_id:b14 Title: Computational complexity of some maximum average weight problems with precedence constraints Year: (1994)
Ref_id:b15 Title: Novel tools to streamline the conference review process: Experiences from SIGKDD'09 Year: (2010)
Ref_id:b16 Title: Algorithmic Graph Theory and Perfect Graphs Year: (1980)
Ref_id:b17 Title: The ellipsoid method and its consequences in combinatorial optimization Year: (1981)
Ref_id:b18 Title: Geometric Algorithms and Combinatorial Optimization Year: (1988)
Ref_id:b19 Title: LLC. Gurobi Optimizer Reference Manual Year: (2025)
Ref_id:b20 Title: Rethinking the funding line at the swiss national science foundation: Bayesian ranking and lottery Year: (2022)
Ref_id:b21 Title: Selection by prediction with conformal p-values Year: (2023)
Ref_id:b22 Title: A polynomial algorithm in linear programming Year: (1979)
Ref_id:b23 Title: Selection problems in the presence of implicit bias Year: (2018)
Ref_id:b24 Title: More of a declaration than a constitution. The Daily Pennsylvanian Year: (2024-02)
Ref_id:b25 Title: Risk, Uncertainty and Profit Year: (1921)
Ref_id:b26 Title: ICML acceptance statistics Year: (2012-05)
Ref_id:b27 Title: Commensuration bias in peer review Year: (2015)
Ref_id:b28 Title: Algorithms for college admissions decision support: Impacts of policy change and inherent variability Year: (2024)
Ref_id:b29 Title: Hiring as exploration Year: (2020)
Ref_id:b30 Title: The acceptability of using a lottery to allocate research funding: A survey of applicants Year: (2020)
Ref_id:b31 Title: On the theory of systematic sampling, II Year: (1949)
Ref_id:b32 Title: Asymmetric information and security design under knightian uncertainty Year: (2020)
Ref_id:b33 Title: Nonparametric bounds on treatment effects Year: (1990)
Ref_id:b34 Title: Identifying and spurring high-growth entrepreneurship: Experimental evidence from a business plan competition Year: (2017)
Ref_id:b35 Title: The case for lotteries as a tiebreaker of quality in research funding Year: (2022-09)
Ref_id:b36 Title: Irreversible investment and knightian uncertainty Year: (2007)
Ref_id:b37 Title: Loss functions, axioms, and peer review Year: (2021)
Ref_id:b38 Title: Computing electricity spot price prediction intervals using quantile regression and forecast averaging Year: (2015)
Ref_id:b39 Title:  Year: (2019)
Ref_id:b40 Title: Frameworks and results in distributionally robust optimization Year: (2022-07)
Ref_id:b41 Title: Ordering problems approximated: Singleprocessor scheduling and interval graph completion Year: (1991)
Ref_id:b42 Title: How to calibrate the scores of biased reviewers by quadratic programming Year: (2011)
Ref_id:b43 Title: A statistical approach to calibrating the scores of biased reviewers: The linear vs. the nonlinear model Year: (2012)
Ref_id:b44 Title: Some implications of knightian uncertainty for finance and regulation Year: (2014-04)
Ref_id:b45 Title: of Studies in Economic Theory Year: (2012)
Ref_id:b46 Title: Counterfactual evaluation of peer-review assignment policies Year: (2023)
Ref_id:b47 Title: Consensual affine transformations for partial valuation aggregation Year: (2019)
Ref_id:b48 Title: Evaluation of editors' abilities to predict the citation potential of research manuscripts submitted to the bmj: a cohort study Year: (2022)
Ref_id:b49 Title: Science Foundation Ireland. Innovate for ireland programme Year: (2024-08)
Ref_id:b50 Title: An overview of challenges, experiments, and computational solutions in peer review Year: (2022-06)
Ref_id:b51 Title: Knightian uncertainty. Available at SSRN 4662711 Year: (2023)
Ref_id:b52 Title: Drawing lots as a tie-breaker Year: (2021-03)
Ref_id:b53 Title: Least square calibration for peer reviews Year: (2021)
Ref_id:b54 Title: Partial randomisation trial extended after diversity of applicants and award holders increases Year: (2025-04)
Ref_id:b55 Title: Using lotteries to allocate research funding: Perspectives from switzerland Year: (2021-09)
Ref_id:b56 Title: USENIX Security 2025 Program Committee. USENIX security 2025 conference format Year: (2001)
Ref_id:b57 Title: Partially randomized procedure -lottery and peer review Year: (2025-05)
Ref_id:b58 Title: Your 2 is my 1 Year: (2018)
Ref_id:b59 Title: How predictive is peer review for gauging impact? the association between reviewer rating scores, publication status, and article impact measured by citations in a pain subspecialty journal Year: (2024)
Ref_id:b60 Title: On the approximability of average completion time scheduling under precedence constraints Year: (2003)
Ref_id:b61 Title: Robust portfolio optimization: A categorized bibliographic review Year: (2020)
