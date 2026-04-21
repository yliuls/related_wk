Title: On Traceability in ℓ p Stochastic Convex Optimization
Abstract: In this paper, we investigate the necessity of traceability for accurate learning in stochastic convex optimization (SCO) under ℓ p geometries. Informally, we say a learning algorithm is m-traceable if, by analyzing its output, it is possible to identify at least m of its training samples. Our main results uncover a fundamental tradeoff between traceability and excess risk in SCO. For every p ∈ [1, ∞), we establish the existence of an excess risk threshold below which every sample-efficient learner is traceable with the number of samples which is a constant fraction of its training sample. For p ∈ [1, 2], this threshold coincides with the best excess risk of differentially private (DP) algorithms, i.e., above this threshold, there exist algorithms that are not traceable, which corresponds to a sharp phase transition. For p ∈ (2, ∞), this threshold instead gives novel lower bounds for DP learning, partially closing an open problem in this setup. En route to establishing these results, we prove a sparse variant of the fingerprinting lemma, which is of independent interest to the community.* SV and MH are equal contribution authors.

Section: Introduction
Tracing or membership inference informally asks whether it is possible, using only the output of a learning algorithm, to distinguish samples in the training set from held-out samples. The existence of a tracer that identifies training examples reveals that the model has memorized specific examples rather than purely captured the underlying distribution [SSSS17;CCNSTT22]. In particular, understanding tracing has an important role in generalization theory, where an algorithm that is not traceable is known to generalize well beyond its training data [SZ20]. Tracing is also an important technical tool in, e.g., differential privacy (DP), where tracing attacks are the workhorse behind tight lower bounds for the risk-privacy trade-offs [BUV14]. From a privacy standpoint, even the leakage of a single example is viewed as catastrophic. However, from a generalization theory standpoint, we want to understand the exact relationship between an algorithm's generalization performance and the number of traceable examples.
To reason rigorously about tracing, following [DSSUV15], we define the problem of tracing as follows. Let A n be a learning algorithm that, given a training set S n = (Z 1 , . . . , Z n ) of n i.i.d. samples from some underlying distribution D, outputs a learned model θ. Then, a tracer T is a hypothesis tester that, given the model θ and a candidate point Z, outputs In if it believes Z was in S n , or Out otherwise. Formally, for some small soundness parameter ξ ∈ (0, 1) and m ≤ n, we require T to satisfy:
Pr Sn∼D ⊗n Z∼Unif(Sn) [T ( θ, Z) = IN] ≥ m n Pr Sn∼D ⊗n Z∼D⊥ ⊥Sn [T ( θ, Z) = OUT] ≥ 1 -ξ
When such a tracer exists, we say that A n is (ξ, m)-traceable. Equivalently, m is the expected number of samples in the training set S n for which the tracer outputs IN, and we refer to m as recall. (See Definitions 2.3 and 2.4.)
A fundamental problem in learning theory is investigating how an algorithm's generalization ability interacts with the information it retains about the training samples (including, in our language, its traceability). The common wisdom is that any information about the training set in a learned model is in tension with generalization [XR17; BMNSY18; SZ20]. On the other hand, non-traceable algorithms, such as differentially private algorithms, are often unable to reach optimal excess risk. The central question we study in this paper is: what is the exact tradeoff between the number of traceable examples and achievable excess risk?
This question was considered, first, in the context of mean estimation of a d-dimensional vector, in the seminal work of [BUV14;DSSUV15]. It was also studied in the context of Stochastic Convex Optimization (SCO) [SSSS09] in the work of [ADHLR24]. The work of [DSSUV15] studied the tradeoff between excess risk and the number of traceable examples, when a mechanism publishes an estimate of the mean that is accurate in every coordinate (i.e., the output of the algorithm has error of α with respect to ℓ ∞ norm to the true mean). At a high level, they showed that for every algorithm that has accuracy better than that achievable by a private algorithm, Ω(1/α 2 ) examples are traceable on some hard instance. Notice that for the task of mean estimation in ℓ ∞ norm, the statistical sample complexity is Θ(log(d)/α 2 ). Thus, for every algorithm, the preceding result only shows that it is possible to trace out a 1/ log(d) fraction of the input samples. In contrast, [ADHLR24] exhibited an SCO problem in ℓ 2 geometry for which a constant fraction of the training samples are traceable. An important open problem, then, is to further explore and understand in which setups we expect a constant fraction of the training sample to be traceable.
In this work, we investigate this question, of traceability, in the fundamental learning setup of Stochastic Convex Optimization for general ℓ p geometries. We show that, in this general learning setup, when private learning is not possible, there is no meaningful gap between sample complexity and traceability. That is, in every geometry, there exist a hard problem for which every (sample-efficient) algorithm is traceable with a recall which is a constant fraction of its sample size. Due to connections between SCO and mean estimation problems, our results also extend to the latter settings; in particular, we close the log(d) gap in the setting of [DSSUV15] and show that optimal traceability is dimension-dependent.
SCO is an ideal testbed for this problem: (1) as in modern machine learning practices, first-order methods are known to achieve optimal sample-complexity rates in this setting [Fel16; HRS16; AKL21], and (2) within this framework, we can design provable methods that mitigate tracing, such as DP algorithms [CMS11; BST14; BFTG19]. Therefore, by studying the problem of traceability in SCO, we also deepen our understanding of the interaction between privacy risks and sample-optimal learning.
To present our results, we recall the basic setup of SCO. An SCO problem is characterized via a triple P = (Z, Θ, f ), where Z is the data space, Θ ⊂ R d is the parameter space, which must be convex, and f : Θ × Z → R is a loss function such that f (•, z) is convex for all z ∈ Z.
In SCO, data points are drawn from an underlying distribution D over Z, unknown to the learner. The objective of the learner is to minimize the expected risk based on observed samples. Then, a learning algorithm A n : Z n → Θ receives a sample S n = (Z 1 , . . . , Z n ) of n data points from Z n and returns a (perhaps randomized) output in Θ. Then, for D ∈ M 1 (Z), expected risk is defined as F D (θ) := E Z∼D [f (Z, θ)] . For an SCO problem to be learnable, one often assumes that the loss function f is Lipschitz and the diameter of the space Θ is bounded, both of which can be measured w.r.t. different norms. These bounds govern the behavior of learnability, but they can be measured in different geometries. A canonical class of SCO problems is induced by the ℓ p norms, in which case we assume that Θ has bounded ℓ p -diameter and f (•, z) is ℓ p -Lipschitz, for a fixed p ∈ [1, ∞].
this section cite: ['b44', 'b17', 'b47', 'b15', 'b21', 'b15', 'b21', 'b43', 'b5', 'b21', 'b5', 'b21']

Section: Contributions
In this paper, we establish a fundamental tradeoff between traceability and excess risk for algorithms in the context of SCO in general geometries. Some settings in which tracing is not possible are already well-understood: in the excess risk regime where DP [DMNS06] is possible, no samples can be traced. Due to this observation, the problem of traceability is only meaningful outside the DP risk regime. More formally, let us define minimax statistical and DP excess risks in ℓ p SCO. Specifically, for a family of ℓ p Lipschitz problems L d p , we let α stat (p, n) = max
P∈L d p min An max D E [F D (A n (S n ))] -inf θ∈Θ F D (θ)
and ( 1)
α DP (p, n) = max P∈L d p min (ε,δ)-DP-An max D E [F D (A n (S n ))] -inf θ∈Θ F D (θ) ,(2)
where, for concreteness, we take ε = 0.01 and δ = 1/n 2 in the above.
this section cite: ['b20']

Section: Main Contribution:
We show that every sample-efficient algorithm that achieves an excess risk outside the DP regime (that is, α = o(α DP )) is traceable with recall proportional to the number of samples. The precise statement of our main contribution varies based on the geometry of SCO:
Tracing when p = 1. For the case p = 1, we show that any learner whose excess risk is better, by a small polynomial factor, than the best risk attainable by a DP algorithm with constant ε and δ = 1/n 2 must be traceable. Moreover, we give an essentially optimal lower bound on the number of samples that can be traced. In more detail, we show that there exists an ℓ 1 -SCO problem such that, if an algorithm achieves risk of
α ≲ α DP d 0.01 log 2 (n) ,
then Ω(log(d)/α 2 )) of the training samples can be traced (see Theorem 2.6). We note that the choice of the constant 0.01 above is arbitrary. It is instructive to compare our results to [DSSUV15]. While the settings of mean estimation and SCO are generally different, our lower bound for ℓ 1 geometry also extends to mean estimation in ℓ ∞ norm (see Corollary 2.9 for a formal argument). In both settings, the sample complexity scales like log(d)/α 2 , however, [DSSUV15] showed traceability of only 1/ log(d) fraction of the samples. On the other hand, in our work we show that there is no meaningful gap between sample complexity and traceability, and every sample-efficient algorithm outside the DP regime must memorize a constant fraction of its sample. Notably, our results also imply that traceability is dimension-dependent in this setup.
Tracing when p ∈ (2, ∞). For ℓ p SCO with p ≥ 2, in Theorem 2.7, we show that for
α ≲ α DP log(n) ,
where α DP is set as α DP = Θ d/n 2 1/p we can construct an SCO problem such that, if a learner achieves a risk of α, then Ω(1/α p ) of its samples are traceable. Note that, the non-private sample complexity of learning for p > 2 is precisely Θ(1/α p ) in the relevant parameter regime, * * i.e., the number of traced out samples is of the order of the sample complexity. Note that, the optimal DP risk in this setup constitutes an open problem, and the quantity α DP above need not be the optimal DP risk in this setting. Nevertheless, this quantity can be shown to be a lower bound on the optimal DP risk (in the regime ε ∈ Θ(1)).
We extend this result to other regimes of ε, and another important contribution of our work is proving such DP lower bounds.
p Recall Range of α Sample complexity Minimax DP rate Refs. 1 log(d) α 2 log(d) n , d 0.49 n √ log(1/ξ) log(d) α 2 log(d) n + √ d εn Thm. 2.6 (1, 2] In particular, we provide an improved lower bound on DP-SCO under ℓ p geometries for p > 2 in the high dimensional regime, i.e., d ≥ εn, which is arguably the most interesting regime as it is more relevant for the modern ML applications. Specifically, we show, in Theorem 2.8, that for all ε < 1 and small δ we can construct a problem such that for every (ε, δ)-DP algorithm, A n , there exists a data distribution such that:
1 α 2 1 n , √ d n √ log(1/ξ) 1 α 2 1 √ n + √ d εn Thm. 2.5 [2, ∞) 1 α p min 1 n 1/p , d 1/2-1/p √ n , d n 2 log(1/ξ) 1/
E Sn∼D ⊗n , θ∼An(Sn) F D ( θ) -inf θ∈Θ F D (θ) ≳ d n 2 ε 2 1/p .
In particular, the above implies that when d ≥ εn, the risk due to privacy dominates the statistical risk. This result improves upon all previous best bounds in the literature when d ≥ εn [ABGMU22; LLL24]. In particular, Theorem 3.1 of [LLL24] gives a lower bound d/n 2 ε 2 , which is weaker than our lower bound for every p > 2. Corollary 4 of [ABGMU22] gives a lower bound of min
1 εn 1 p , d 1-1/p εn
which is weaker than our lower bound for d ≥ εn.
Tracing when p ∈ (1, 2]. For each p ∈ (1, 2], we show that there exists an ℓ p SCO problem such that, if an algorithm achieves excess risk of α ≲ α DP log 2 (n) , then Ω(1/α 2 ) of its samples can be traced (see Theorem 2.5). This result uncovers a fundamental dichotomy between traceability and privacy in ℓ p SCO. It is known that p ∈ (1, 2], Θ(1/α 2 ) is precisely the sample complexity of learning ℓ p -Lipschitz problems [AWBR09]. We note that α DP for p = 2 is known due to [BST14]. However, as we discuss in Appendix B.1, combining [BST14] with the tracing results of [DSSUV15] does not give the optimal tracing of Θ(1/α 2 ) samples.
this section cite: ['b21', 'b21', 'b36', 'b3', 'b0', 'b9', 'b9', 'b21']

Section: Traceability beyond SCO: PAC Learning
A natural question is whether a similar phenomenon holds true for other learning setups. Consider the setting of binary classification PAC learning. We show that, for every class with VC dimension bounded by d vc , the recall of every tracer is in O(d vc log 2 (n)), i.e., it is at most a small fraction of the training sample provided n ≫ d vc . Since many such classes, including the class of thresholds, are not privately learnable [BNSV15; ALMM19; BLM20], the sharp transitions between privacy and traceability does not hold in PAC classification. We also point out that for the class of thresholds, we can remove the log 2 (n) factor from the recall upper bound. See Appendix H.
this section cite: []

Section: Technical contributions
Our technical contributions are elaborated on in Section 2.3. In essence, our technical novelties are twofold. First, we present a novel sparse fingerprinting lemma that, intuitively, shows that learners over sparse domains must be correlated to their samples. The key novelty of this result is that the correlation is inversely proportional to the sparsity parameter. This feature is not present in prior work, since fingerprinting lemmas are most often applied for learners/estimators over a hypercube domain.
Second, armed with this new fingerprinting result, we present a generic conversion result using a notion of a subgaussian trace value, which converts any lower bound on correlation with the samples into a number of samples that can be traced. While it is well-appreciated by prior work that, conceptually, a fingerprinting lower bound implies a traceability lower bound, proving results for our setting of SCO involves complicated sparse domains embedded into ℓ p balls. This makes it more technically challenging to prove the necessary concentration phenomena holds for a tracer over the corresponding domain, which motivates us to restrict our attention to tracers that induce a subgaussian process over the domain.
this section cite: []

Section: Related Work
Our work is most similar in spirit to [DSSUV15;ADHLR24]. Our work builds on top of these results on a number of fronts. A key distinct aspect of our approach is the difference in the structure of hard problems and the new sparse fingerprinting lemma. Also, our generic traceability theory of subgussian trace value (Section 2.3) provides an abstract treatment of the approach in [DSSUV15]. Our approach allows to seamlessly convert fingerprinting lemmas into traceability results and even non-private sample complexity lower bound.
Our work also makes progress towards closing the gap regarding the optimal excess error for ℓ p DP-SCO for p > 2. The best known upper bounds for DP-SCO in ℓ p geometry for p > 2 are due to [BGN21; GLLST23], and Theorem 2.8 is the best lower bound.
To put our sparse fingerprinting lemma into the context of prior work, it can be seen to generalize the results of [SU17] to sparse sets. Another "sparse fingerprinting lemma" in the literature is given by [CWZ23]. Our results are distinct by the way sparsity enters the lemmas: in [CWZ23] the mean vector is sparse (and data is dense), and in our case, the mean is dense and the data vectors are sparse. The proof techniques also differ substantially. Our sparse fingerprinting lemma is also an example of a fingerprinting lemma for the setting where the coordinates of the data vector are not independent, similar to [KMS22;LT24]. Additional related work is discussed in Appendix B.
this section cite: ['b21', 'b5', 'b21', 'b46', 'b16', 'b16', 'b35', 'b39']

Section: Problem Setup and Main Results
We begin by some definitions. For a (measurable) space R, M 1 (R) denotes the set of all probability measures on R. In SCO, an α-learner is defined to be a learner whose expected excess risk is bounded by α. A formal definition is given below. Definition 2.1 (α-learner). Fix α > 0, n ∈ N and SCO problem (Θ, Z, f ). We say A n : Z n → M 1 (Θ) is an α-learner for (Θ, Z, f ) iff for every D ∈ M 1 (Z), we have
E Sn∼D ⊗n , θ∼An(Sn) F D ( θ) -inf θ∈Θ F D (θ) ≤ α.
In our work, we focus on learning Lipschitz-bounded families of problems, which are defined below. For every p ∈ [1, ∞], let B p (r) = {θ ∈ R d : ∥θ∥ p ≤ r} be the unit ball in ℓ p norm.
Definition 2.2 (Lipschitz-bounded problems). Fix p ∈ [1, ∞], and let d < ∞ be a natural number. We let L d p denote the set of all ℓ p -Lipschitz-bounded SCO problems in d dimensions. Namely, P = (Θ, Z, f ) ∈ L d p iff (i) Θ ⊂ B p (1), and (ii) for every θ 1 , θ 2 ∈ Θ and z ∈ Z, we have |f (z, θ 1 ) -f (z, θ 2 )| ≤ ∥θ 1 -θ 2 ∥ p .
this section cite: []

Section: Tracing
The key notion we study here is tracing, and we next introduce our framework for traceability. We consider families of tracers that assign each candidate point a real-valued score capturing how likely it is to have been seen during training. Then, the tracer converts these scores into binary In or Out decisions by thresholding the score. Intuitively the score corresponds to the likelihood of the event that the learner saw a data point during training. Definition 2.3 (Tracer). Fix data space Z and parameter space Θ. A tracer's strategy is a tuple of T = (ϕ, D) where ϕ : Θ × Z → R and D ∈ M 1 (Z). Definition 2.4 ((ξ, m)-traceability). Let n ∈ N, ξ ∈ (0, 1), and m ∈ N. We say a learning algorithm A n is (ξ, m)-traceable if there exists a tracer (ϕ, D) and λ ∈ R such that, if (Z 0 , Z 1 , . . . , Z n ) ∼ D ⊗(n+1) and θ ∼ A n (Z 1 , . . . , Z n ), we have (i) Soundness: Pr ϕ( θ, Z 0 ) ≥ λ ≤ ξ, and (ii) Recall: E {i ∈ [n] : ϕ( θ, Z i ) ≥ λ} ≥ m.
this section cite: []

Section: Main Results

this section cite: []

Section: Traceability of α-Learners
In this section, we discuss our traceability results for accurate learners in ℓ p geometries. First, we will state a result that applies to p ∈ [1, 2), and then present its slight refinement for p = 1. We will then present our result for p ≥ 2. See Appendices F.1 to F.3 for proofs.
Theorem 2.5. There exists a universal constant c > 0 such that, for all p ∈ [1, 2), if d, n, ξ ∈ (0, 1/e), and α > 0 are such that
c √ n ≤ α ≤ min c • d n 2 log(1/ξ) , 1 6 , (3
)
then there exist an ℓ p SCO problem that every α-learner is (ξ, m)-traceable with m ∈ Ω α -2 .
Note that the upper bound on α in Equation (3) is precisely the optimal DP excess risk for ε ∈ Θ(1) and p ∈ [1, 2] [AFKT21; BGN21], and the lower bound is precisely the optimal non-private risk (except p = 1; see Theorem A.1). Moreover, for p ∈ (1, 2], the lower bound on m exactly matches the statistical sample complexity.
As mentioned above, for p = 1, the lower bound on recall in Theorem 2.5 is less than sample complexity by a factor of log(d). This prompts us to establish the following refinement
Theorem 2.6. There exists a universal constant c > 0 such that, if d is large enough and n, ξ ∈ (0, 1/e), and α > 0 are such that
c • log(d) n ≤ α ≤ min c • d 0.49 n log(1/ξ) , 1 8 , then(4)
there exists a ℓ 1 SCO problem that every α-learner is (ξ, m)-traceable with m ∈ Ω log(d)/α 2 .
Note that the upper bound in Equation (4) is slightly stronger than in Equation (3); however, the lower bound on recall now matches the sample complexity of learning in ℓ 1 geometry. We now present a result for p ≥ 2.
Theorem 2.7. There exists a universal constant c > 0 such that, for all p ∈ [2, ∞), if d, n, ξ ∈ (0, 1/e), and α > 0 are such that
1 6 • min 1 n 1/p , d 1 2 -1 p √ n ≤ α ≤ min c • d n 2 log(1/ξ) 1/p , 1 6 , then(5)
there exist an ℓ p SCO problem such that every α-learner is (ξ, m)-traceable with m ∈ Ω (1/(6α) p ) .
For p ∈ (2, ∞), our results have a different implication, showing that all sufficiently accurate learners need to memorize a number of samples on the order of the sample complexity. However, in this case, the upper bound in Equation (5) need not be the optimal DP risk.
Instead, it provides a lower bound on the optimal DP risk, as we will see next.
this section cite: []

Section: Improved DP-SCO Lower Bound for p > 2
Theorem 2.8. Let p ∈ [2, ∞). There exist a universal constant c > 0 and an ℓ p SCO problem P = (Θ, Z, f ) such that every (ε, δ)-DP learner of P with ε ≤ 1 and δ ≤ c/n satisfies,
α ≥ c • min d ε 2 n 2 1 p , d 1-1/p εn , 1 .
this section cite: []

Section: Consequences for mean estimation
Consider the setting of mean estimation in ℓ ∞ norm as in [DSSUV15]. Our results in Theorem 2.6 extend almost verbatim to this setting.
Corollary 2.9. Let Z = {±1} d , and suppose an estimator is given such that, given access to i.i.d. samples Z 1 , . . . , Z n ∈ Z, outputs μ with E ∥μ -E[Z 1 ]∥ ∞ ≤ α/2. Then, there exists a universal constant c > 0 such that, if d is large enough and n, ξ ∈ (0, 1/e), and α > 0 satisfy Equation (4), then the estimator μ is (ξ, m)-traceable with m ∈ Ω log(d)/α 2 .
this section cite: ['b21']

Section: Roadmap of the proof
Our proofs rely on introducing two key technical elements that allow us to generalize tracing techniques to general ℓ p setups. The first element is a generic conversion result involving a complexity notion which we term the subgaussian trace value of a problem. As we show in the proof of Theorem 2.8, we can use the subgaussian trace value to prove traceability results over general domains, establish DP sample complexity lower bounds, and, even to recover non-private sample complexity lower bounds. While the connection between the first two aspects is well-known [FS17], we find the ability of trace value to recover non-private lower bounds surprising.
Our second technical contribution concerns techniques for lower bounding the subgaussian trace value, which we accomplish through several novel fingerprinting lemmas. Previous works used the standard fingerprinting lemma, where the learner observes points on a hypercube, to lower bound DP and traceability in ℓ 2 geometry. However, when moving to general ℓ p geometries, this setup no longer captures the hardest settings to learn. For instance, for p > 2, canonical instances of hard problems involve data drawn from sparse sets [AWBR09]. We thus prove new fingerprinting lemmas that enable us to leverage our framework in such settings. These fingerprinting lemmas are then applied to carefully constructed instances of hard problems, and we show that every accurate learner of these problems is traceable.
this section cite: ['b25', 'b0']

Section: General framework: subgaussian trace value
We next describe more formally the framework of subgaussian tracers. For a random variable X, the subgaussian norm of X is the quantity ∥X∥ ψ2 := inf{t : E exp(X 2 /t 2 ) ≤ 2} [Ver18].
We use the following definition of a subgaussian process:
Definition 3.1 (Subgaussian process). We call an indexed collection of random variables {X θ } a σ-subgaussian process w.r.t a metric space (Θ, ∥•∥) if for every θ, θ ′ ∈ Θ, we have (i)
∥X θ -X θ ′ ∥ ψ2 ≤ σ ∥θ -θ ′ ∥ , and (ii) ∥X θ ∥ ψ2 ≤ σ diam ∥•∥ (Θ).
For origin symmetric convex body Θ, let ∥•∥ Θ denote the Minkowski norm w.r.t. Θ, that is ∥x∥ Θ := inf {λ > 0 : x ∈ λΘ} . If Θ is not convex or not origin symmetric, we let ∥•∥ Θ be the Minkowski norm w.r.t. convex hull of (Θ ∪ -Θ). Note that ∥•∥ Θ is the minimal norm to contain Θ in its unit ball.
Definition 3.2 (Subgaussian tracer). Fix κ ∈ R to be a constant, and let Θ be a convex body. We let T κ be the class of subgaussian tracers at scale κ > 0, that is, a tracer (ϕ, D) ∈ T κ iff (i) {ϕ(θ, Z)} θ∈Θ where Z ∼ D is a 1-subgaussian process w.r.t. (Θ, ∥•∥ Θ ).
(ii) |ϕ(θ, z)| ≤ κ for all θ ∈ Θ and z ∈ Z.
Definition 3.3 (Subgaussian trace value). Fix n ∈ N, α ∈ [0, 1], and κ ∈ R. Consider an arbitrary SCO problem P = (Θ, Z, f ). Let T κ be as in Definition 3.2. Then, we define the subgaussian trace value of problem P by
Tr κ (P; n, α) = inf α-learnerAn sup T =(ϕ,D)∈Tκ E Sn=(Z1,...,Zn)∼D ⊗n , θ∼An(Sn) 1 n i∈[n] ϕ( θ, Z i ) .
where the inf is taken over all A n that achieve excess risk ≤ α on P with n samples.
this section cite: ['b49']

Section: Traceability via subgaussian trace value.
The subgaussian trace value characterizes the average score the pair (ϕ, D) assigns to the data points in the training set. However, the definition of recall in Definition 2.3 requires characterizing the number of samples in the training set that takes a large value. The former can be converted into the latter, provided the sum of squared scores of samples is not too large. A formal statement, which is a consequence of Paley-Zygmund inequality, can be found in Lemma A.11. In the next lemma, we show how to control the sum of squares of the ϕ( θ, Z i ) using the subgaussian assumption.
Lemma 3.4. Fix n, d ∈ N. Suppose Θ ⊂ R d is a subset of a unit ball in some norm ∥•∥. Let ϕ : Θ × Z → R and D ∈ M 1 (Z) be such that, as Z ∼ D, {ϕ(θ, Z)} is a σ-subgaussian process w.r.t. (Θ, ∥•∥). Let (Z 1 , . . . , Z n ) ∼ D ⊗n . Then, there is a constant C > 0, such that
Pr   sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + √ d + t   ≥ 1 -4 exp(-t 2 ), ∀t ≥ 0.
Equipped with this lemma, in the next theorem, we show that if, the subgaussian trace value of a problem is large, then every α-learner is traceable. Theorem 3.5. Fix n ∈ N, d ∈ N, κ > 0 and α ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ). Let T = Tr κ (P; n, α) be the subgaussian trace value of P. Then, for some constant c > 0, every α-learner A n is (ξ, m)-traceable with
ξ = exp(-cT 2 ), m = c n 2 T 2 n + d - 16κ 2 n exp(n + d) .
Privacy lower bounds via subgaussian trace value. In the next theorem, we show that the notion of subgaussian trace value directly lower bounds the best privacy parameters achievable by a DP algorithm. The proof is based on [FS17].
Theorem 3.6. There exists a universal constant c > 0, such that the following holds. Fix p ∈ [1, ∞), n ∈ N, d ∈ N, α ∈ [0, 1], κ > 0 ε > 0, and δ ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ) in R d . Let T = Tr κ (P; n, α) be the subgaussian trace value of problem P. Then, for every (ε, δ)-DP α-learner A n , we have exp(ε) -1 ≥ c (T -2δκ) .
this section cite: ['b25']

Section: Non-private sample complexity via subgaussian trace value.
Surprisingly, if we directly use subgaussian trace value, we can recover optimal sample complexity bounds for all p ∈ [1, ∞) and all regimes of (d, α), thus unifying traceability with private and non-private sample complexity lower bounds. While we detail the argument formally in Appendix G, we consider here a helpful example of ℓ 2 geometry. First, it can be shown that we always have Tr(P; n, α) ≲ d/n for arbitrary problem P (see Proposition G.1). Also, we will later show that, for every α > 0, there exist an ℓ 2 problem P with Tr(P; n, α) ≳ √ d/nα (see Theorem 5.1). Combining these two inequalities gives n ≳ 1/α 2 , which is optimal.
this section cite: []

Section: The sparse fingerprinting lemma
By introducing the notion of subgaussian trace value, we have reduced the problems of traceability and privacy lower bounds to the question of lower bounding the subgaussian trace value. Now, we discuss the techniques to lower bound subgaussian trace value. The proofs can be found in Appendix D. Due to space limitation, we only discuss the details for the case of p > 1 and present the details of p = 1 in Appendix E.2.
For ℓ 2 geometry, one can lower bound subgaussian trace value using the classical fingerprinting lemma in [DSSUV15]. While this strategy leads to traceability results in ℓ 2 geometry, examples of hard problems for ℓ p geometry with p > 2 are those with sparse sets Z (e.g., as in [AWBR09]). This motivates us to prove the following sparse fingerprinting lemma, which is another important contribution of our work. For a vector x ∈ R d , let supp(x) be the set of its non-zero coordinates and denote
∥x∥ 0 = | supp(x)|.
Definition 4.1 (Sparse distributions family). Fix d ∈ N, k ∈ [d] and µ ∈ [-k/d, k/d] d . Consider the mixture distribution on Z k = {z ∈ {0, ±1} d : ∥z∥ 0 = k} given by, for all z ∈ Z k ,
D µ,k (z) = E J∼unif(( [d]
k )) [P µ,k,J (z)] , where
P µ,k,J (z) = 1(supp(z) = J) • j∈J 1 + (d/k) • µ j z j 2 .
Note that, in particular, E Z∼D µ,k [Z] = µ. Intuitively, one can think of sampling from D µ,k using the following procedure: (i) sample the support coordinates J ∼ unif [d]  k , (ii) for each j ∈ J, sample Z j from {±1} with mean d k µ j independently, (iii) for each j ̸ ∈ J, set Z j = 0.
With this distribution family at hand, we may state the sparse fingerprinting lemma. For x, y ∈ R d and a subset R ⊆ [d] of coordinates, we use ⟨•, •⟩ S to denote the inner product ⟨x, y⟩ R := i∈R x i y i . Also, for α, β, γ > 0, let s-beta [-γ,γ] (α, β) be the symmetric betadistribution, i.e., beta distribution with parameters α, β scaled and shifted to have support [-γ, γ] (see Definition A.13).
Lemma 4.2 (Sparse fingerprinting). Fix d, n ∈ N and let k ∈ [d]. For each µ ∈ [-k/d, k/d] d , let Z k and D µ,k be as in Definition 4.1 . Let π = s-beta [-k/d,k/d] (β, β) ⊗d be a prior and set
ϕ µ (θ, Z) := θ, Z - d k µ supp(Z)
.
Then, for every learning algorithm
A n : Z n → M 1 (R d ) with sample S n = (Z 1 , . . . , Z n ), E µ∼π E Sn∼D ⊗n µ,k , θ∼An(Sn) n i=1 ϕ µ ( θ, Z i ) = 2βd k E µ∼π µ, E Sn∼D ⊗n µ,k , θ∼An(Sn) [ θ] .
The key novelty of this lemma is that it provides a way to study the correlation between a learner's output and training samples on sparse sets Z k . An important and distinctive feature of this result is that the right-hand side scales by a factor of d/k, highlighting the fact that sparse problems correspond to greater subgaussian trace values. Intuitively, this stems from the fact that each coordinate is seen fewer times by the learning algorithm, meaning it must retain more information from each training sample in order to learn accurately. Additionally, for the special case k = d, the result precisely recovers the fingerprinting lemma from [SU17].
this section cite: ['b21', 'b0', 'b46']

Section: Final steps: bounding the subgaussian trace value for hard problems
Finally, we go over the construction of hard problems. To illustrate the difficulty of problem constructions, we give an example of a problem that requires many samples to learn but nevertheless is not traceable. Consider learning over ℓ 1 ball with linear loss. Let
Θ = B 1 (1), Z = {±1} d , f (θ, Z) = -⟨θ, Z⟩. (6
)
Consider a difficult set of distributions {D i } d i=1 where D i is a product distribution on Z and has mean α on coordinate i and mean zero on all other coordinates. It can be shown this problem requires Θ(log(d)/α 2 ) samples to learn up to risk of α/3, and ERM is an optimal learner. However, after seeing Θ(log(d)/α 2 ) samples from D i , the ERM takes the value θ = e i w.h.p., which is also the population risk minimizer. In other words, it becomes impossible to trace out any specific samples on which θ was trained.
Generic construction for p ∈ (1, ∞). As mentioned above, to obtain optimal results for p > 2, problems constructed need to be sparse, and the main subtlety in our constructions is choosing the sparsity parameter. For some k ∈ [d] to be chosen later, consider the following
ℓ p -Lipschitz problem P k,p . Θ = B ∞ (d -1/p ), Z = {z ∈ {0, ±1} d : ∥z∥ 0 = k}, f (θ, z) = -k -1/q ⟨θ, z⟩.(7)
Here, the parameter space Θ is the largest ℓ ∞ ball inscribed into the unit ℓ p ball, and q is the Hölder conjugate of p, i.e., 1 p + 1 q = 1. The next step is to show that α-learners for the above problem must be correlated with the mean of the unknown data distribution. Let D be a distribution with mean µ, and suppose A n is an α-learner for Equation (7). Then,
E Sn∼D ⊗n , θ∼An(Sn) µ, θ ≥ sup θ∈Θ ⟨µ, θ⟩ -k 1/q • α = d -1/p ∥µ∥ 1 -k 1/q α.
Now, we apply the sparse fingerprinting lemma (Lemma 4.2). A key step is choosing the scale β ≥ 1 of the beta-prior. On the one hand, β should be small enough to guarantee Ed -1/p ∥µ∥ 1 > k 1/q α, so that the above lower bound is non-vacuous. On the other hand, taking β too small decreases the sample complexity of learning the problem, thus, disallowing the desired level of recall. The optimal choice is β ∝ α -2 • (k/d) 1/p , as long as this quantity is ≥ 1. This choice yields
Tr κ (P k,p ; n, α) ≥ E Sn∼D ⊗n , θ∼An(Sn) 1 n n i=1 ϕ( θ, Z i ) ≳ d 1-1/p k 1/2-1/p nα ,
where κ ∈ Θ(1), and, for some universal constant c > 0, we let
ϕ(θ, Z) := cd 1/p √ k θ, Z - d k µ supp(Z)
.
Note that the d 1/p / √ k scaling ensures ϕ induces a 1-subgaussian process. Finally, it remains to choose a suitable value for k, for each pair (p, α). Recall the definition of P k,p from Equation (7).
Theorem 5.1. Let P k,p be the family of problems described in Equation (7). There exist universal constants c 1 , c 2 > 0 such that, for all α ∈ (0, 1/6] and d ∈ N, the following subgaussian trace value lower bounds hold for all p ∈ [1, ∞) and κ ≤ c 1 √ d:
(i) For p ≤ 2 and k = d, we have Tr κ (P k,p ; n, α) ≥ c 2
√ d nα .
(ii) For p ≥ 2 and k = (6α) p d ∨ 1, we have Tr κ (P k,p ; n, α) ≥ c 2
√ d n(6α) p/2 ∧ d 1-1/p nα .
Using the reduction Theorem 3.5, the above establishes Theorems 2.5 and 2.7.
Refinement for p = 1. While the above construction also yields a traceability result for p = 1, it is suboptimal for the following simple reason: for k = d, the problem in Equation (7) only requires Θ(1/α 2 ) samples to learn, thus, it is impossible to trace out Ω(log(d)/α 2 ) samples. On the other hand, the problem in Equation (6) requires Θ(log(d)/α 2 ) samples to learn but is not traceable. The intuition we follow here is to modify the construction in Equation (7) to make Θ "look" more like an ℓ 1 -ball to drive up the sample complexity while still avoiding the counterexample with an ERM learner from the beginning of the section. In particular, we consider the following ℓ 1 -problem,
Θ = B 1 (1) ∩ B ∞ (1/s), Z = {±1} d , f (θ, z) = -⟨z, θ⟩, (8
)
for a suitably chosen s ∈ [d]. Note that, if we choose s ≫ 1, Θ above is a polytope with much more vertices (2 s d s ) than an ℓ 1 ball (2d), which would intuitively force a learner like an ERM to reveal more information about the training sample. On a technical level, selecting large s improves the subgaussian constant of a tracer; however, selecting s that is too large shrinks the diameter of the set, and thus, the problem becomes easier to learn. We must trade off these two aspects, and carefully set the value of s. As it turns out, the optimal choice is s ∝ d 1-c for any small c > 0 in order to establish Theorem 2.6. The remainder of the proof is rather technical and hence is deferred to Appendix F.2.
this section cite: []

Section: Limitations
We conclude by stating an intriguing open problem. We conjecture Theorem 2.8 is tight, and the dichotomy between traceability and SCO also holds for p > 2. In particular, we conjecture that the optimal DP-SCO excess risk for ℓ p with p > 2 scales as
min d 1/2-1/p √ n , 1 n 1/p + min d 1-1/p nε , d ε 2 n 2 1/p ,
ignoring log(1/δ) factors. If the conjecture is true, we have a complete understanding of traceability in SCO. If it is false, it reveals that there is something fundamentally different about settings with p > 2, which would also significantly enrich our understanding of DP-SCO.
this section cite: []

Section: A Additional preliminaries

this section cite: []

Section: A.1 Background on SCO
The next proposition summarizes the known minimax rates for learning SCO problems in general geometries. A proof can be found in [NY83; AWBR09; ST10].
Theorem A.1. Fix p ∈ [1, ∞], d ∈ N, and n ∈ N. Let α stat (L d p , n) be the minimax excess risk rate of learning ℓ p -Lipschitz-bounded problems, as defined in Equation (1). Then, 1. For p = 1, we have
α stat (L d p , n) ∈ Θ log(d) n .
2. For 1 < p ≤ 2, we have
α stat (L d p , n) ∈ Θ log(d) n ∧ 1 (p -1) √ n 3. For 2 ≤ p < ∞, we have α stat (L d p , n) ∈ Θ d 1/2-1/p √ n ∧ 1 n 1/p 4. For p = ∞, we have α stat (L d p , n) ∈ Θ d n .
Remark A.2. Notice that, in the overparameterized regime (d ≥ n), the minimax excess risk for p ≥ 2 is Θ 1 n 1/p which is dimension-independent. This shows that for d ≥ n, in all geometries except p = {1, ∞}, the minimax excess risk is dimension-free. ◁ This proposition implies the following corollary on the minimum number of samples required for α-learners.
Corollary A.3. Fix p ∈ [1, ∞], d ∈ N, and α ∈ (0, 1]. Let N stat (L d p , n) be the sample complexity of learning problems L d p up to excess risk α, i.e.,
N stat (L d p , n) = min n : α stat (L d p , n) ≤ α . Then, 1. For p = 1, we have N stat (L d p , n) ∈ Θ log(d) α 2 . 2. For 1 < p ≤ 2, we have N stat (L d p , n) ∈ Θ log(d) α 2 ∧ 1 ((p -1)α) 2 . 3. For 2 ≤ p < ∞, we have N stat (L d p , n) ∈ Θ d 1-2/p α 2 ∧ 1 α p .
4. For p = 1, we have
N stat (L d p , n) ∈ Θ d α 2 .
this section cite: []

Section: A.2 Differential Privacy
Definition A.4. Let ε > 0 and δ ∈ [0, 1). A randomized mechanism A n : Z n → M 1 (Θ) is (ε, δ)-DP, iff, for every two neighboring datasets S n ∈ Z n and S ′ n ∈ Z n (that is, S n , S ′ n differ in one element), and for every measurable subset M ⊆ Θ, it holds
Pr θ∼An(Sn) θ ∈ M ≤ e ε • Pr θ∼An(S ′ n ) θ ∈ M + δ.
Algorithms that satisfy DP are not traceable in the sense of Definition 2.4 [KOV17]. The following simple proposition formalizes this observation.
Proposition A.5. Fix n ∈ N and ε, δ > 0. Let A n be an (ε, δ)-DP algorithm. Then, if A n is (ξ, m)-traceable, it holds that m ≤ n exp(ε)ξ + nδ.
this section cite: ['b32']

Section: A.3 Concentration inequalities
First, we collect lemmata on the subgaussian norm, introduced in Section 3, which we use to derive concentration inequalities. The following is Equation (2.14) in [Ver18], and shows that a bound on subgaussian norm immediately leads to concentration inequalities.
this section cite: ['b49']

Section: Lemma A.6 (Subgaussian concentration).
There exists a universal constant C such that the following holds for every random variable X with ∥X∥ ψ2 < ∞: for every t ≥ 0,
Pr [|X| ≥ t] ≤ 2 exp - ct 2 ∥X∥ 2 ψ2
The subgaussian norm behaves nicely under the summation of independent random variables. The following is Proposition 2.6.1 in [Ver18].
Lemma A.7 (Sum of subgaussian variables). Let C > 0 be a universal constant. Let X 1 , . . . , X n be a collection of arbitrary independent real random variables. Then,
n i=1 X i 2 ψ2 ≤ C n i=1 ∥X i ∥ 2 ψ2 .
Subgaussian norm also behaves nicely under mixtures. In particular, we have the following proposition.
Proposition A.8 (Subgaussian mixtures). Let {X α } α∈A be σ-subgaussian random variables, and let π be a distribution over the index set A. Then, a mixture of {X α } α∈A under α ∼ π is also σ-subgaussian.
Proof. Let Y be such mixture. Then, for every t > 0, we have
E[exp(Y 2 /t 2 )] = E α∼π E[exp(X 2 α /t 2 )]
. Plugging in t = σ into above, and using that ∥X α ∥ ψ2 ≤ σ for all α, we have
E[exp(Y 2 /σ 2 )] = E α∼π E[exp(X 2 α /t 2 )] ≤ 2, i.e., ∥Y ∥ ψ2 ≤ σ, as desired.
It is well-known that bounded random variables are subgaussian (Equation (2.17) of [Ver18]). Proof. Fix an arbitrary θ 0 ∈ Θ. Using Theorem 8.1.6 [Ver18], we obtain the following bound for the increment of the subgaussian process {X θ },
Pr sup θ∈Θ |X θ -X θ0 | ≤ Cσ ∞ 0 log N (Θ, ∥•∥ , ε)dε + t ≥ 1 -2 exp(-t 2 ).
First, note that for ε ≥ 1, N (Θ, ∥•∥ , ε) = 1, since Θ lies in the unit ball of ∥•∥. Thus,
Pr sup θ∈Θ |X θ -X θ0 | ≤ Cσ 1 0 log N (Θ, ∥•∥ , ε)dε + t ≥ 1 -2 exp(-t 2 ). (9
)
Note that, by triangle inequality, we have
sup θ∈Θ |X θ -X θ0 | ≥ sup θ∈Θ |X θ | -X θ0 . (10
)
Since {X θ } θ∈Θ satisfies Definition 3.1, we have
∥X θ0 ∥ ψ2 ≤ 2σ.
From Lemma A.6, we then have The following lemma is an anti-concentration inequality based on Paley-Zygmund inequality. It shows that if the sum of variables is large, one can conclude that many of them are large given an appropriate control over their sum of squares. It is given as Lemma A.4 in [ADHLR24], and it is also similar to Lemma 25 in [DSSUV15].
Lemma A.11. Fix n ∈ N and (a 1 , . . . , a n ) ∈ R n . Let A 1 := i∈[n] a i and A 2 := i∈[n] (a i ) 2 . Then, for every β ∈ R, {i ∈ [n] :
a i ≥ β/n} ≥ (max{A1-β,0}) 2 A2 .
this section cite: ['b49', 'b49', 'b49', 'b5', 'b21']

Section: A.4 Beta distributions
Next definitions are the versions of beta distributions that we use in this paper. Recall that, classically, beta distribution is supported on [0, 1]. However, in our results, it is convenient to consider the rescaled and centered variants.
Definition A.12. Fix β > 0. A (symmetric) beta distribution denoted by s-beta (β, β) is a continuous distribution, such that, if X ∼ s-beta (β, β), then, for every a ∈ [-1, 1], we have
Pr (X ≤ a) = a -1 1 -x 2 β-1 B(β) dx,
where B(β) = 2 2β-1 Γ(β) 2 /Γ(2β).
Definition A.13. Fix β > 0 and γ ∈ (0, 1]. We define rescaled (symmetric) beta distribution, denoted by s-beta [-γ,γ] (β, β), where for a ∈ [-γ, γ], its distribution is given by Pr (X ≤ a) = 1 γB(β
) a -γ 1 - x γ 2 β-1 dx,
where B(β) = 2 2β-1 Γ(β) 2 /Γ(2β).
We have the following result on the first moment of the beta distribution.
Lemma A.14. Fix β > 0. Let X ∼ s-beta (β, β) where β ≥ 1. Then,
E|X| ≥ 1 3 √ β .
Proof. Let B(β) = 2 2β-1 Γ(β) 2 /Γ(2β) be the normalization constant. We have
E|X| = 1 B(β) 1 -1 |x|(1 -x 2 ) β-1 dx = 1 B(β) 1 0 2x(1 -x 2 ) β-1 dx = 1 β • B(β) .
It remains to upper bound B(β). It follows from Theorem 1.5 of [Bat08] that, for every x ≥ 1, we have
a x -1/2 e x-1/2 ≤ Γ(x) ≤ b x -1/2 e x-1/2 ,
where a = √ 2e and b = √ 2π are absolute constants. Thus,
B(β) = 2 2β-1 Γ(β) 2 Γ(2β) ≤ 2 2β-1 b 2 β-1/2 e 2β-1 a 2β-1/2 e 2β-1/2 = b 2 √ e a (2β -1/2) -1/2 2β -1 2β -1/2 2β-1 ≤ b 2 √ e a (2β -1/2) -1/2 = 2π √ e √ 2e (2β -1/2) -1/2 = π (β -1/4) -1/2 ≤ π 3 4β 1/2 ,
where in the last line we used β -1/4 ≥ 3 4 β which holds as β ≥ 1. Thus,
E|X| ≥ 1 π(3/4) 1/2 1 √ β ≥ 1 3 √ β ,
as desired.
Since the density of the rescaled beta distribution is homogeneous w.r.t. γ, we have the following result.
Corollary A.15. Fix β ≥ 1 and γ ∈ (0, 1]. Let X ∼ s-beta [-γ,γ] (β, β). Then,
E|X| ≥ γ 3 √ β .
this section cite: ['b10']

Section: B Additional Related Work

this section cite: []

Section: Necessity of memorization in learning.
A parallel line of work investigated memorization using the notion of label memorization in supervised setups. As per this definition, a learner is said to memorize its training samples if it "overfits" at these points. Feldman [Fel20] showed that, in some classification tasks, if the underlying distribution is long-tailed, then a learner is forced to memorize many training labels. Cheng, Duchi, and Kuditipudi [CDK22] showed this phenomenon also occurs in the setting of linear regression. While this framework is suitable to study memorization in supervised tasks, the notion of "labels" in SCO in not well-defined and thus calls for alternative definitions.
Another line of work studied memorization through the lens of information theoretic measures. Brown, Bun, Feldman, Smith, and Talwar [BBFST21] used input-output mutual information (IOMI) as a memorization metric and showed that IOMI can scale linearly with the training sample's entropy, indicating that a constant fraction of bits is memorized. In the context of SCO in ℓ 2 geometry, lower bounds on IOMI have been studied in [HRTSMK23;Liv24]. Specifically [Liv24] demonstrated that, for every accurate algorithm, its IOMI must scale with dimension d. Our approach to the study of memorization is conceptually different since we focus on the number of samples memorized as opposed to the number of bits. Nevertheless, it can be shown using Lemma H.3 and [HNKRD20, Thm. 2.1] that the recall lower bounds IOMI of an algorithm (provided that soundness parameter ξ is small enough, e.g., ξ = 1/n 2 ). However, because of the Lipschitzness of loss functions in ℓ p SCO, we can use discretization of Θ and design algorithms with IOMI that is significantly smaller that the entropy of the training set, thus, memorization in the sense of [BBFST21] does not arise here.
this section cite: ['b23', 'b19', 'b12', 'b29', 'b38', 'b38', 'b12']

Section: Membership inference.
Membership inference is an important practical problem [HSRDTMPSNC08; SSSS17; CCNSTT22]. In these works, the focus is on devising strategies for the tracer in modern machine learning settings, particularly neural networks. Our work takes a more fundamental perspective, aiming to determine whether membership inference is inherently unavoidable or simply a byproduct of specific training algorithms. An interesting aspect of our results is that, for 1 < p ≤ 2, the optimal strategy for tracing depends only on the loss function, which is in line with empirical studies [SDSOJ19].
Private Stochastic Convex Optimization. DP-SCO has been extensively studied in ℓ 2 geometry (see, for instance, [CMS11; BST14; BFTG19; FKT20]). For ℓ p with p ∈ [1, 2), the optimal DP excess risk was established in [AFKT21; BGN21]. The best known upper bounds for DP-SCO in ℓ p geometry for p > 2 are due to [BGN21;GLLST23]. In this setting, there is a long-standing gap between upper and lower bounds, and the best known lower bounds are due to [ABGMU22; LLL24], which our paper improves on.
this section cite: ['b42', 'b7', 'b26']

Section: B.1 Detailed comparison with [DSSUV15; BST14].
One might hope that existing traceability results (such as [DSSUV15]) and a clever reduction to mean estimation (such as [BST14, Section 5.1]) might yield optimal results for SCO. Here, we will demonstrate rigorously that merely combining results and techniques of [DSSUV15; BST14] yields suboptimal results for the setup of SCO, even in the simple setting of ℓ 2 geometry.
[BST14] considers the following ℓ 2 problem:
Θ = B 2 (1), Z = ± 1 √ d d , f (θ, Z) = -⟨θ, Z⟩.
To apply fingerprinting to establish traceability, we first need to posit a prior distribution over the unknown distribution. [DSSUV15] does so by considering product distributions over Z, and placing a uniform prior over the mean µ ∈ [-1/ √ d, 1/ √ d] d . We now show that this (Bayesian) problem requires only O(1/α) samples to learn, and thus, tracing Ω(1/α 2 ) samples is clearly impossible. Consider the ERM learner θ. It is easy to see that θ can be written as:
θ = μ ∥μ∥ 2 ,
where μ is the empirical mean of the dataset, that is, μ = 1 n n i=1 Z i . Similarly, the population risk minimizer θ ⋆ is µ/ ∥µ∥ 2 . The expected excess risk of θ is then:
E µ ∥µ∥ 2 , µ - μ ∥μ∥ 2 , µ = E ∥µ∥ 2 ∥μ∥ 2 -⟨μ, µ⟩ ∥μ∥ 2 ≤ (a) E 1 2 ∥µ∥ 2 2 + 1 2 ∥μ∥ 2 2 -⟨μ, µ⟩ ∥μ∥ 2 ∧ 2 = 1 2 E ∥µ -μ∥ 2 2 ∥μ∥ 2 ∧ 4 ,
where in (a) we used the AM-GM inequality, and the fact that the expression on the preceding line is always bounded by 2. The intuition behind the rest of the argument is that, due to the uniform prior on µ, we have ∥µ∥ , ∥μ∥ ∈ Ω(1) with high probability. At the same time
E ∥µ -μ∥ 2 = E 1 dn n i=1 2µ i (1 -µ i ) ≤ 1 2n ,
thus, expected risk will be on the order of O(1/n). To formalize this, note that E ∥µ∥ 2 2 = 1/3, and ∥µ∥ 2 is a sum of d independent random variables bounded by 1/ √ d in absolute value. Hoeffding's inequality then yields that we have ∥µ∥ 2 2 ≥ 1/6 with very high probability. Similarly, we can obtain ∥µ -μ∥ 2 2 ≤ 1/(2n) + 1/36 ≤ 1/12 for large enough n, with high probability. Then, with high probability, event
E := ∥μ∥ ≥ 1 √ 6 - 1 √ 12 ≥ 0.1
holds Then, the excess risk is upper bounded by
E µ ∥µ∥ 2 , µ - μ ∥μ∥ 2 , µ ≤ 1 2 E ∥µ -μ∥ 2 2 ∥μ∥ 2 ∧ 4 = 1 2 E 1(E) ∥µ -μ∥ 2 2 ∥μ∥ 2 ∧ 4 + 1 2 E 1(E c ) ∥µ -μ∥ 2 2 ∥μ∥ 2 ∧ 4 ≤ 1 2 E 10 ∥µ -μ∥ 2 2 + 2 Pr(E c ) ∈ O(1/n), as desired.
this section cite: ['b21', 'b21']

Section: C Proofs from Section 3

this section cite: []

Section: C.1 Proof of Lemma 3.4
We first prove a slightly more general concentration statement to bound the supremum in Lemma 3.4, which will be useful to reuse in other proofs. Let N (Θ, ∥•∥ , ε) denote the size of the minimal cover of Θ in norm ∥•∥ at scale ε > 0. Then, the more general statement is given below.
Lemma C.1. Fix n, d ∈ N. Suppose Θ ⊂ R d is a subset of a unit ball in some norm ∥•∥. Let ϕ : Θ × Z → R and D ∈ M 1 (Z) be such that, as Z ∼ D, {ϕ(θ, Z)} is a σ-subgaussian process w.r.t. (Θ, ∥•∥) and for every θ ∈ Θ, E[ϕ(θ, Z)] = 0. Let (Z 1 , . . . , Z n ) ∼ D ⊗n . Then, there exist a universal constant C > 0, such that for every t ≥ 0,
Pr   sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + 1 0 log N (Θ; ∥•∥ , ε)dε + t   ≥ 1 -4 exp(-t 2 ).
Proof. Let Φ θ denote the following random vector
Φ θ =    ϕ(θ, Z 1 ) . . . ϕ(θ, Z n )    .
Then, observe that, the desired quantity is equal to
sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 = sup θ∈Θ ∥Φ θ ∥ 2 = sup θ∈Θ,x∈S n-1 ⟨x, Φ θ ⟩ .
Then, ⟨x, Φ θ ⟩ can be seen to be a random process parameterized by a pair (x, θ). We will show that it is, in fact, a subgaussian process. Indeed, note that, by triangle inequality,
∥⟨x, Φ θ ⟩ -⟨x ′ , Φ θ ′ ⟩∥ ψ2 ≤ ∥⟨x -x ′ , Φ θ ⟩∥ ψ2 + ∥⟨x ′ , Φ θ -Φ θ ′ ⟩∥ ψ2 . (11
)
Since Φ i θ is σ-subgaussian for each i, we have by Lemma A.7, ∥⟨x -
x ′ , Φ θ ⟩∥ ψ2 ≤ Cσ ∥x -x ′ ∥ 2 ,
for some universal constant C > 0. Now, for every i,
(Φ θ -Φ θ ′ ) i is σ ∥θ -θ ′ ∥-subgaussian.
Therefore, by Lemma A.7, we have
∥⟨x ′ , Φ θ -Φ θ ′ ⟩∥ ψ2 = n i=1 (x ′ ) i (Φ i θ -Φ i θ ′ ) ψ2 ≤ Cσ ∥θ -θ ′ ∥ .
Combining the two inequalities, we get
∥⟨x, Φ θ ⟩ -⟨x ′ , Φ θ ′ ⟩∥ ψ2 ≤ Cσ ∥θ -θ ′ ∥ + Cσ ∥x -x ′ ∥ 2 = 2Cσ • 1 2 [∥θ -θ ′ ∥ + ∥x -x ′ ∥ 2 ] .
Thus, ⟨x, Φ θ ⟩ is (2Cσ)-subgaussian process w.r.t the norm γ, defined as γ((x, θ)) := 1 2 [∥x∥ 2 + ∥θ∥] .
Moreover, we can see that Θ × S n-1 is a subset of a unit ball in γ. By definition of γ, we have
N S n-1 × Θ; γ, ε ≤ N S n-1 ; ∥•∥ 2 , ε • N (Θ; ∥•∥ , ε) .(12)
Then, using Proposition A.10, we have, for some constant K > 0, that with probability 1 -4 exp(-t 2 )
sup θ ∥Φ θ ∥ 2 ≤ Kσ 1 0 log N (Θ × S n-1 ; γ, ε)dε + t ≤ Kσ 1 0 log N (S n-1 ; ∥•∥ 2 , ε) + log N (Θ; ∥•∥ , ε)dε + t ≤ (a) Kσ 1 0 n log 1 + 4 ε dε + 1 0 log N (Θ; ∥•∥ , ε)dε + t ≤ K ′ σ √ n + 1 0 log N (Θ; ∥•∥ , ε)dε + t ,
as desired, where in (a) we used Example 5.8 from [Wai19], and K ′ > 0 is some other universal constant.
Using Example 5.8 from [Wai19] once again to upper bound log N (Θ; ∥•∥ , ε), we have the proof of Lemma 3.4.
Lemma 3.4. Fix n, d ∈ N. Suppose Θ ⊂ R d is a subset of a unit ball in some norm ∥•∥. Let ϕ : Θ × Z → R and D ∈ M 1 (Z) be such that, as Z ∼ D, {ϕ(θ, Z)} is a σ-subgaussian process w.r.t. (Θ, ∥•∥). Let (Z 1 , . . . , Z n ) ∼ D ⊗n . Then, there is a constant C > 0, such that
Pr   sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + √ d + t   ≥ 1 -4 exp(-t 2 ), ∀t ≥ 0.
Proof. From Example 5.8 in [Wai19], we have log N (Θ; ∥•∥ , ε) ≤ d log 1 + 2 ε .
Plugging this into the result of Lemma C.1, with probability at least 1 -4 exp(-t 2 ), we have
sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + 1 0 log N (Θ; ∥•∥ , ε)dε + t ≤ Cσ √ n + 1 0 d log 1 + 2 ε dε + t ≤ C ′ σ √ n + √ d + t ,
for some other universal constant C ′ > 0.
this section cite: ['b50', 'b50', 'b50']

Section: C.2 Proof of Theorem 3.5
Theorem 3.5. Fix n ∈ N, d ∈ N, κ > 0 and α ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ). Let T = Tr κ (P; n, α) be the subgaussian trace value of P. Then, for some constant c > 0, every α-learner A n is (ξ, m)-traceable with ξ = exp(-cT 2 ), m = c n 2 T 2 n + d -16κ 2 n exp(n + d) .
Proof. We set λ := T 2 First, we show that the soundness condition holds. Since Z and θ are independent, and using the subgaussian nature of ϕ( θ, Z), we have by Lemma A.6
Pr Z∼D ϕ( θ, Z) ≥ λ ≤ exp -cλ 2 ≤ exp -cT 2 /4 ,
where c > 0 is some constant. For recall, let's define the set I as follows
I = {i ∈ [n] : ϕ( θ, Z i ) ≥ λ}.
Using Lemma A.11, we have
E [|I|] = E {i ∈ [n] : ϕ( θ, Z i ) ≥ λ} ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + n i=1 ϕ( θ, Z i ) 2    ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + sup θ ∥{ϕ(θ, Z i )} n i=1 ∥ 2 2    ,
where for every x ∈ R, we define (x) + = max{x, 0}. Then, Lemma 3.4 tells us that, for t := √ n + d, we have with probability 1 -4 exp(-t 2 ), sup θ∈Θ ϕ( θ, Z 1 ), . . . , ϕ( θ, Z n ) ⊤ 2 2 ≤ C (n + d) ,
for some constant C > 0. Thus,
Pr E := sup θ∈Θ ϕ( θ, Z 1 ), . . . , ϕ( θ, Z n ) ⊤ 2 2 ≤ C (n + d) ≥ 1 -4 exp(-t 2 ).
This implies,
E|I| ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + sup θ ∥{ϕ(θ, Z i )} n i=1 ∥ 2 2    ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + sup θ ∥{ϕ(θ, Z i )} n i=1 ∥ 2 2 1(E)    = E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d) 1(E)    ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d)    -E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d) 1(E c )    .
We know that, almost surely,
ϕ( θ, Z) 2 ≤ κ 2 .
Thus, almost surely,
n i=1 ϕ( θ, Z i ) -nλ 2 + ≤ n i=1 ϕ( θ, Z i ) -λ + 2 ≤ n n i=1 ϕ( θ, Z i ) 2 ≤ κ 2 n 2 .
Thus,
E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d) 1(E c )    ≤ Pr[E c ] • κ 2 n 2 C(n + d) ≤ 4κ 2 n C exp(n + d) .
Hence,
E [|I|] ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d)    -E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d) 1(E c )    ≥ E    n i=1 ϕ( θ, Z i ) -nλ 2 + C (n + d)    - 4κ 2 n C exp(n + d) ≥ (a) n i=1 Eϕ( θ, Z i ) -nλ 2 + C (n + d) - 4κ 2 n C exp(n + d) ≥ n 2 T 2 /4 C (n + d) - 4κ 2 n C exp(n + d) = c n 2 T 2 n + d - 16κ 2 n exp(n + d) ,
where (a) follows by Jensen's inequality and c = 1/4C.
this section cite: []

Section: C.3 Proof of Theorem 3.6
Theorem 3.6. There exists a universal constant c > 0, such that the following holds. Fix p ∈ [1, ∞), n ∈ N, d ∈ N, α ∈ [0, 1], κ > 0 ε > 0, and δ ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ) in R d . Let T = Tr κ (P; n, α) be the subgaussian trace value of problem P. Then, for every (ε, δ)-DP α-learner A n , we have exp(ε) -1 ≥ c (T -2δκ) .
Proof. Consider an arbitrary distribution D and a function ϕ s.
t. {ϕ(θ, Z)} θ∈Θ is a 1subgaussian process w.r.t. (Θ, ∥•∥ Θ ) and |ϕ| ≤ κ almost surely. Consider a sample S n = (Z 1 , . . . , Z n ) and let Z 0 be a freshly sampled point; let S (i)
n be a sample with Z i substituted by Z 0 . Let θ be a learner trained on S n and θ(i) be a learner trained on S (i) n . Then, since θ is (ε, δ)-DP and noting that ϕ(θ, Z) is supported on [-κ, κ], we may apply Lemma A.1 of [FS17] and get
Eϕ( θ, Z i ) -Eϕ( θ(i) , Z i ) ≤ E ϕ( θ(i) , Z i ) (exp(ε) -1) + 2δκ.
By independence of θ and Z i , we conclude that ϕ( θ, Z i ) is 1-subgaussian random variable. It is well-known that E|X| ≤ Cσ if X is σ-subgaussian for some constant C (see part (ii) of Proposition 2.5.2 of [Ver18] for p = 1), thus the above gives Eϕ( θ, Z i ) ≤ C(exp(ε) -1) + 2δκ. Then, for every D and ϕ we get that,
E 1 n n i=1 ϕ( θ, Z i ) ≤ C(exp(ε) -1) + 2δκ.
Thus, T ≤ C(exp(ε) -1) + 2δκ, which, after rearranging, implies the desired result.
this section cite: ['b25', 'b49']

Section: D Proofs of fingerprinting lemmas (Section 4)

this section cite: []

Section: D.1 Proof of Lemma 4.2
Lemma 4.2 (Sparse fingerprinting). Fix d, n ∈ N and let k ∈ [d]. For each µ ∈ [-k/d, k/d] d , let Z k and D µ,k be as in Definition 4.1 . Let π = s-beta [-k/d,k/d] (β, β)
⊗d be a prior and set
ϕ µ (θ, Z) := θ, Z - d k µ supp(Z) .
Then, for every learning algorithm A n : Z n → M 1 (R d ) with sample S n = (Z 1 , . . . , Z n ),
E µ∼π E Sn∼D ⊗n µ,k , θ∼An(Sn) n i=1 ϕ µ ( θ, Z i ) = 2βd k E µ∼π µ, E Sn∼D ⊗n µ,k , θ∼An(Sn) [ θ] .
Proof. For each j ∈ [d], let I j := {i ∈ [n] : Z j i ̸ = 0} as the index of the training points such that their j-th coordinate is non-zero. Then, we have
E n i=1 ϕ µ ( θ, Z i ) = E   d j=1 i∈Ij ( θ) j Z j i - d k µ j   (13
)
= d j=1 E   i∈Ij ( θ) j Z j i - d k µ j   . (14
)
Then, define the following function
g j (µ j ) := E ( θ) j {I r } r∈[d] , {Z m i } m̸ =j,i∈[n] .
We claim
E   i∈Ij ( θ) j • Z j i - d k µ j {I r } r∈[d] , {Z m i } m̸ =j,i∈[n]   = k d 1 - d k µ j 2 d dµ j g j (µ j ). (15
) The proof is based on the following two observations: 1) conditioned on {Z m i } m̸ =j,i∈[n] , θ is a function of {Z j i } i∈[n] , 2) conditioned on {I r } r∈[d] the non-zero elements in {Z j i } i∈[n] are sampled i.i.d from {±1} with mean d k µ j . Then, based on these observations Equation (15) follows as an straightforward application of [Ste16, Lemma 4.3.7].
Recall the definition of π and notice that π is a product measure. Let π j be the distribution on the j-th coordinate. By the definition of the prior distribution, we can write
E µ j ∼π j k d 1 - d k µ j 2 d dµ j g j (µ j ) = 1 C + k d -k d k d 1 - d k v 2 d dv g j (v) 1 - d k v 2 β-1 dv = 1 C + k d -k d k d d dµ j g j (v) 1 - d k v 2 β dv = 2 C + k d -k d d k βv 1 - d k v 2 β-1 g j (v)dv = 2β d k E µ j ∼π j g j (µ j )µ j . (16
)
Therefore, we have
E µ∼π E Sn∼D ⊗n µ,k , θ∼An(Sn) n i=1 ϕ µ ( θ, Z i ) = d j=1 E   i∈Ij ( θ) j Z j i - d k µ j   = d j=1 E   E   i∈Ij ( θ) j Z j i - d k µ j {I r } r∈[d] , {Z m i } m̸ =j,i∈[n]     = 2β d k d j=1 E g j (µ j ) • µ j ,
where the last step follow from Equations ( 15) and ( 16). Then, notice that
E g j (µ j ) • µ j = E E ( θ) j • µ j {I r } r∈[d] , {Z m i } m̸ =j,i∈[n] = E ( θ) j • µ j .
Therefore, by the definition of inner product in R d , we have
2β d k d j=1 E g j (µ j ) • µ j = 2β d k E θ, µ ,
as was to be shown.
this section cite: []

Section: D.2 Fingerprinting for ℓ 1 setup.
Additionally, to prove Theorem 2.6, we will need the following fingerprinting lemma. It can be seen as a generalization of beta-fingerprinting lemma in [SU17] using the scaling matrix technique of [KLSU18].
Lemma D.1 (Fingerprinting lemma with a scaling matrix). Fix d ∈ N. Let Z = {±1} d and let β > 0 be arbitrary. Consider arbitrary 0 < γ ≤ 1. For every µ ∈ [-γ, γ] d , let D µ be the product distribution on Z with mean µ, i.e., for every z ∈ Z, we have
D µ = d k=1 1+z k µ k 2 let
Λ µ be a diagonal matrix of size d where the i-th diagonal element is given by Λ ii µ = 1-(µ i /γ) 2 1-(µ i ) 2 , and let ϕ µ (θ, z) = ⟨θ, Λ µ (z -µ)⟩. Let π = s-beta [-γ,γ] (β, β) ⊗d be a prior. Then, for every algorithm A n : Z n → M 1 (R d ), we have
E µ∼π E Sn∼D ⊗n µ , θ∼An(Sn) Z∈Sn ϕ µ ( θ, Z) = 2β γ 2 E µ∼π µ, E Sn∼D ⊗n µ , θ∼An(Sn) [ θ] .
This fingerprinting lemma is handy for the following reason. To ensure the problem is hard to learn, entries of µ typically need to inversely scale with α. To achieve this, one can select small γ in the above to shrink the beta-prior to a smaller scale, while simultaneously having the freedom to set β to any value. In particular, this allows us to choose β ∈ Θ(log(d)) in the proof of Theorem 2.6 to leverage the anti-concentration result of [SU17, Prop. 5].
Before we proceed with the proof, we state the necessary lemmata. Throughout this section, for a real number p ∈ [-1, 1], we will write X ∼ p to denote the fact that X is a random variable on {±1} with mean p. The following is a classical fingerprinting result.
Lemma D.2 (Lemma 5 of [DSSUV15]). Let f : {±1} n → R be arbitrary.
Define g : [-1, 1] → R by g(p) = E X∼p ⊗n [f (X)].
Then,
E X∼p ⊗n   f (X) i∈[n] (X i -p)   = (1 -p 2 )g ′ (p).
Armed with the above result, we proceed to the proof of Lemma D.1. We will first prove a per-coordinate version of Lemma D.1. We make a note that the proofs combine techniques for beta-fingerprinting results of [SU17] and the scaling matrix technique of [KLSU19; ADHLR24].
this section cite: ['b46', 'b33', 'b21', 'b46']

Section: Lemma D.3 (Per-coordinate version of Lemma D.1).
Let f : {±1} n → R be arbitrary. Let π = s-beta [-γ,γ] (β, β) be a prior distribution. Then,
E p∼π E X∼p ⊗n 1 -(p/γ) 2 1 -p 2 f (X) n i=1 (X i -p) = 2β γ 2 E p∼π p • E X∼p ⊗n f (X) . Proof. Let g(p) = E X∼p ⊗n [f (X)]
. Then, by Lemma D.2, we have for every p ∈ [-1, 1],
E X∼p ⊗n   1 -(p/γ) 2 1 -p 2 f (X) i∈[n] (X i -p)   = 1 - p γ 2 g ′ (p).
Recalling the definition of scaled symmetric beta distribution from Definition A.13, we have
E p∼π E X∼p ⊗n   1 -(p/γ) 2 1 -p 2 f (X) i∈[n] (X i -p)   = E p∼π 1 - p γ 2 g ′ (p) = 1 γB(β) γ -γ 1 - p γ 2 β-1 • 1 - p γ 2 g ′ (p)dp = 1 γB(β) γ -γ 1 - p γ 2 β g ′ (p)dp = (a) 1 γB(β)   1 - p γ 2 β g(p) γ -γ - γ -γ   1 - p γ 2 β   ′ g(p)dp   = 1 γB(β)   γ -γ 1 - p γ 2 β-1 • 2βp γ 2 g(p)dp   = 2β γ 2 E p∼π [p • g(p)] ,
where in (a) we used integration by parts. This concludes the proof.
Applying the above results to each coordinate and summing the equalities gives Lemma D.1.
Lemma D.1 (Fingerprinting lemma with a scaling matrix). Fix d ∈ N. Let Z = {±1} d and let β > 0 be arbitrary. Consider arbitrary 0 < γ ≤ 1. For every µ ∈ [-γ, γ] d , let D µ be the product distribution on Z with mean µ, i.e., for every z ∈ Z, we have
D µ = d k=1 1+z k µ k 2 let
Λ µ be a diagonal matrix of size d where the i-th diagonal element is given by Λ ii µ = 1-(µ i /γ) 2 1-(µ i ) 2 , and let ϕ µ (θ, z) = ⟨θ, Λ µ (z -µ)⟩. Let π = s-beta [-γ,γ] (β, β) ⊗d be a prior. Then, for every algorithm A n : Z n → M 1 (R d ), we have
E µ∼π E Sn∼D ⊗n µ , θ∼An(Sn) Z∈Sn ϕ µ ( θ, Z) = 2β γ 2 E µ∼π µ, E Sn∼D ⊗n µ , θ∼An(Sn) [ θ] .
Proof. For a sample S n = (Z 1 , . . . , Z n ) and j ∈ [j], we will use S j n ∈ R n to denote a vector (Z j 1 , . . . , Z j n ) of j th coordinates. For each coordinate j ∈ [d], let f j :
{±1} n → R the function such that f j (S j n ) = E µ∼π E Sn∼D ⊗n µ , θ∼An(Sn) θi | S j n
In other words, f j (X) is the expected value of θ j , given that j th coordinates of samples in S n are given by X. Applying the result of Lemma D.1 to f j , we have
E µ j ∼π j E S j n ∼(µ j ) ⊗n Λ jj µ • f j (S j n ) n i=1 (Z j i -µ j ) = E µ j ∼π j E S j n ∼(µ j ) ⊗n 1 -(µ j /γ) 2 1 -(µ j ) 2 f j (S j n ) n i=1 (Z j i -µ j ) = 2β γ 2 E µ j ∼π j µ j • E S j n ∼(µ j ) ⊗n f j (S j n )
. By the law of total expectation, we get
E µ∼π E Sn∼D ⊗n µ , θ∼An(Sn) Λ jj µ • θj n i=1 (Z j i -µ j ) = 2β γ 2 E µ∼π µ j • E Sn∼D ⊗n µ , θ∼An(Sn) θj .
Finally, summing the above over all coordinates j ∈ [d], we obtain
E µ∼π E Sn∼D ⊗n µ , θ∼An(Sn) Λ µ θ, n i=1 (Z i -µ) = 2β γ 2 E µ∼π µ, E Sn∼D ⊗n µ , θ∼An(Sn)
θ , as desired.
this section cite: []

Section: E Hard problem constructions and proofs of subgaussian trace value lower bounds

this section cite: []

Section: E.1 Proofs for ℓ p -geometries (Theorem 5.1)
First, recall here the construction of the hard problems
P k,p in Equation (7), parameterized by k ∈ [d] Θ = B ∞ (d -1/p ), Z = {z ∈ {0, ±1} d : ∥z∥ 0 = k}, f (θ, z) = -k -1/q ⟨θ, z⟩.
(P k,p ) First, we show in the simple proposition below that α-learners for linear problems must agree with the distribution mean. Proposition E.1. Let A n be an α-learner for P k,p . Let D ∈ M 1 (Z) be a distribution with mean µ = E Z∼D [Z]. Then, we have
E Sn∼D ⊗n , θ∼An(Sn) µ, θ ≥ d -1/p ∥µ∥ 1 -k 1/q α. Proof. Since A n is an α-learner, we have α ≥ E F D ( θ) -inf θ∈Θ F D (θ) = E Sn∼D ⊗n , θ∼An(Sn) E Z∼D f ( θ, Z) -inf θ∈Θ E Z∼D f (θ, Z) = k -1/q sup θ∈Θ E Z∼D ⟨θ, Z⟩ -E Sn∼D ⊗n , θ∼An(Sn) E Z∼D θ, Z = k -1/q sup θ∈Θ ⟨θ, µ⟩ -E Sn∼D ⊗n , θ∼An(Sn) θ, µ , which, after rearranging, becomes E Sn∼D ⊗n , θ∼An(Sn) µ, θ ≥ sup θ∈Θ ⟨µ, θ⟩ -k 1/q • α = d -1/p ∥µ∥ 1 -k 1/q α,
where in the last transition we used duality of ℓ ∞ and ℓ 1 norms and the fact that Θ = B ∞ (d -1/p ). This concludes the proof.
In the next lemma, we show that every α-learner for P k,p needs to have a large correlation with the training samples in order to achieve small excess risk. The proof is an application of Lemma 4.2 combined with Proposition E.1. Lemma E.2. Let α ≤ 1/6, and suppose k ∈ [d] is such that k ≥ (6α) p d. Then, for every α-learner A n for P k,p , there exists µ ∈ [-k/d, k/d] d and distribution D ∈ M 1 (Z k ) with mean µ such that the following holds: let
ϕ(θ, Z) := d 1/p √ k θ, Z - d k µ supp(Z) ,
then,
E Sn∼D ⊗n , θ∼An(Sn) n i=1 ϕ( θ, Z i ) ≥ d 1-1/p 18k 1/2-1/p α Proof. Let β = k 1/p 6d 1/p α 2 ≥ 1,
and π = s-beta [-k/d,k/d] (β, β). Then, using Corollary A.15, we have
E µ∼π [∥µ∥ 1 ] = dE µ∼π |µ 1 | ≥ d • k/d 3 √ β = k 3(k 1/p /6d 1/p α) = 2αd 1/p k 1-1/p = 2αd 1/p k 1/q .
Then, using Lemma 4.2 we have
E µ∼π E Sn∼D ⊗n µ,k n i=1 θ, Z i - d k µ supp(Zi) = 2dβ k E µ∼π E Sn∼D ⊗n µ,k , θ∼An(Sn) µ, θ ≥ (a) 2dβ k E µ∼π d -1/p ∥µ∥ 1 -k 1/q α ≥ (b) 2dβ k d -1/p • 2αd 1/p k 1/q -k 1/q α = 2d k • k 1/p 6d 1/p α 2 • k 1/q α = 2d k • k 1/p 6d 1/p α 2 • k 1/q α = d 1-2/p k 1/p 18α .
Since the above holds in expectation over draws of µ, there exists at least one value of µ for which the above holds; let D = D k,µ . Then, letting
ϕ(θ, Z) := d 1/p √ k θ, Z - d k µ supp(Z) ,
we obtain
E Sn∼D ⊗n , θ∼An(Sn) n i=1 ϕ( θ, Z i ) = E µ∼π E Sn∼D ⊗n µ,k n i=1 d 1/p √ k θ, Z i - d k µ supp(Zi) ≥ d 1-1/p 18k 1/2-1/p α , as desired.
We now argue that the pair (ϕ, D) from the lemma above (with ϕ scaled by some constant) constitutes a valid subgaussian tracer. In particular, the lemma below shows that {ϕ(θ, Z)} θ∈Θ induces a O(1
)-subgaussian process w.r.t. (Θ, ∥•∥ Θ ) norm. Lemma E.3. Fix d ∈ N. Let µ ∈ [-k/d, k/d] d be arbitrary. Let ϕ : Θ × Z k → R
be as in Lemma E.2. Let D µ,k be the data distribution from Definition 4.1 for some µ, and consider Z ∼ D µ,k . Then, {ϕ(θ, Z)} θ∈Θ is a C-subg process w.r.t. to (Θ, ∥•∥ Θ ) for some universal constant C > 0.
Proof. Let J ∈ [d]  k be an arbitrary coordinate subset of size k, and, recalling Definition 4.1, let Z J be a random variable with PMF given by P µ,k,J . Then, Z is a uniform mixture of {Z J } J∈( [d] k ) . Fix J ∈ [d]  k , and let θ 1 , θ 2 ∈ Θ be two arbitrary points. First, we upper bound a subgaussian norm of ϕ(θ 1 , Z J ) -ϕ(θ 2 , Z J ). We have
∥ϕ(θ 1 , Z J ) -ϕ(θ 2 , Z J )∥ ψ2 = d 1/p √ k ∥⟨θ 1 -θ 2 , Z J ⟩ J ∥ ψ2 = d 1/p √ k j∈J (θ j 1 -θ j 2 )Z j J ψ2 ≤ (a) d 1/p √ k C 1 j∈J (θ j 1 -θ j 2 )Z j J 2 ψ2 ≤ (b) d 1/p √ k C 1 j∈J C 2 |θ j 1 -θ j 2 | 2 = d 1/p √ k • C 1 C 2 √ k ∥θ 1 -θ 2 ∥ ∞ = (c) C 1 C 2 ∥θ 1 -θ 2 ∥ Θ ,
where C 1,2 > 0 are universal constants, in (a) we apply Lemma A.7, in (b) we apply Proposition A.9, and in (c) we use that, since
Θ = B ∞ (d -1/p ), we have ∥•∥ Θ = d 1/p ∥•∥ ∞ . Thus, letting C = √ C 1 C 2 , we have ∥ϕ(θ 1 , Z J ) -ϕ(θ 2 , Z J )∥ ψ2 ≤ C ∥θ 1 -θ 2 ∥ Θ .
Now, note that ϕ(θ 1 , Z) -ϕ(θ 2 , Z) has the same distribution as a uniform mixture of {ϕ(θ 1 , Z J ) -ϕ(θ 2 , Z J )} J∈( [d] k ) . Then, by Proposition A.8, we also have ∥ϕ(θ
1 , Z) -ϕ(θ 2 , Z)∥ ψ2 ≤ C ∥θ 1 -θ 2 ∥ Θ ,
which satisfies the first condition in Definition 3.1. Finally, by plugging θ 2 = 0 into the above, we have ∥ϕ(θ 1 , Z)∥ ψ2 ≤ C ∥θ 1 ∥ Θ ≤ C, which satisfies the second condition in Definition 3.1. Thus, {ϕ(θ, Z)} θ∈Θ is a C-subgaussian process w.r.t. (Θ, ∥•∥ Θ ), as desired.
Finally, we lower bound the subgaussian trace value of P k,p .
Lemma E.4. Let α ≤ 1/6 and d ∈ N be arbitrary, and let k ∈ [d] be such that k ≤ (6α) p d. Let P k,p be as in Equation (7). Then, the following subgaussian trace value lower bounds hold for every p ∈ [1, ∞) and some κ ≤ c 1 √ d,
Tr κ (P k,p ; n, α) ≥ c 2 d 1-1/p k 1/2-1/p nα where c 1,2 > 0 are universal constants.
Proof. Let C > 0 be the constant from Lemma E.3, and ϕ be as in Lemma E.2. Then, {ϕ(θ, Z)/C} θ∈Θ is a 1-subgaussian process w.r.t. (Θ, ∥•∥ Θ ). Moreover, ϕ(θ, Z)/C ≤ √ k/C ≤ √ d/C. Then, letting κ = √ d/C and using Lemma E.2, the subgaussian trace value of P k,p can be lower bounded by
Tr κ (P k,p ; n, α) ≥ 1 C E Sn∼D ⊗n , θ∼An(Sn) 1 n n i=1 ϕ( θ, Z i ) ≥ d 1-1/p 18C • k 1/2-1/p nα , as desired.
Now we are ready to prove Theorem 5.1. Theorem 5.1. Let P k,p be the family of problems described in Equation (7). There exist universal constants c 1 , c 2 > 0 such that, for all α ∈ (0, 1/6] and d ∈ N, the following subgaussian trace value lower bounds hold for all p ∈ [1, ∞) and κ ≤ c 1 √ d:
(i) For p ≤ 2 and k = d, we have Tr κ (P k,p ; n, α) ≥ c 2
√ d nα .
(ii) For p ≥ 2 and k = (6α
) p d ∨ 1, we have Tr κ (P k,p ; n, α) ≥ c 2 √ d n(6α) p/2 ∧ d 1-1/p nα .
Proof. The theorem is a direct consequence of Lemma E.4. For p ≤ 2, plug in k = d into the statement of Lemma E.4. We obtain
Tr κ (P k,p ; n, α) ≥ c 2 d 1-1/p k 1/2-1/p nα = c 2 √ d nα .
For p ≥ 2, plug in k = (6α) p ∨ 1. We obtain,
Tr κ (P k,p ; n, α) ≥ c 2 d 1-1/p k 1/2-1/p nα = c 2 d 1-1/p nα ∧ √ d (6α) p(1/2-1/p) nα = c 2 d 1-1/p nα ∧ √ d (6α) p/2 n , as desired. E.2 Proofs for ℓ 1 -geometry E.2.1 Intuition Refinement for p = 1.
While the above construction also yields a traceability result for p = 1, it is suboptimal for the following simple reason: for k = d, the problem in Equation (7) only requires Θ(1/α 2 ) samples to learn, thus, it is impossible to trace out Ω(log(d)/α 2 ) samples. On the other hand, the problem in Equation (6) requires Θ(log(d)/α 2 ) samples to learn but is not traceable. The intuition we follow here is to modify the construction in Equation (7) to make Θ "look" more like an ℓ 1 -ball to drive up the sample complexity while still avoiding the counterexample with an ERM learner from the beginning of the section. In particular, we consider the following ℓ 1 -problem,
Θ = B 1 (1) ∩ B ∞ (1/s), Z = {±1} d , f (θ, z) = -⟨z, θ⟩,(17)
for a suitably chosen s ∈ [d]. Note that, if we choose s ≫ 1, Θ above is a polytope with much more vertices (2 s d s ) than an ℓ 1 ball (2d), which would intuitively force a learner like an ERM to reveal more information about the training sample. On a technical level, selecting large s improves the subgaussian constant of a tracer; however, selecting s that is too large shrinks the diameter of the set, and thus, the problem becomes easier to learn. We must trade off these two aspects, and carefully set the value of s. As it turns out, the optimal choice is s ∝ d 1-c for an arbitrary small c > 0 in order to establish Theorem 2.6. The remainder of the proof is rather technical and hence is deferred to Appendix F.2.
this section cite: []

Section: E.2.2 Formal Proof
For technical reasons, we will need the following refinement of Lemma 3.4 for the special case when the function ϕ is convex and Θ is a polytope. In the proof, we use Lemma C.1.
Lemma E.5. Fix n, d ∈ N. Suppose Θ ⊂ R d is (i) a subset of a unit ball in some norm ∥•∥, and (ii) Θ is a polytope with N vertices. Let ϕ : Θ × Z → R be a measurable function that is convex in its first argument. Let D ∈ M 1 (Z) be such that ϕ(θ, Z) is a σ-subgaussian process w.r.t. (Θ, ∥•∥).Let (Z 1 , . . . , Z n ) ∼ D ⊗n . Then, for every t ≥ 0,
Pr   sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + log(N ) + t   ≥ 1 -2 exp(-t 2 )
where C > 0 is some universal constant.
Proof. Similarly to the proof of Lemma C.1, let Φ θ denote the following random vector
Φ θ =    ϕ(θ, Z 1 ) . . . ϕ(θ, Z n )    .
Then, observe that, the desired quantity is equal to
sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 = sup θ∈Θ ∥Φ θ ∥ 2 = sup θ∈Θ,x∈S n-1 ⟨x, Φ θ ⟩ .
Let V be the set of vertices of Θ with |V | ≤ N . Since ϕ is convex in its first argument, the supremum above is attained in one of the vertices of Θ. Thus,
sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 = sup θ∈V,x∈S n-1 ⟨x, Φ θ ⟩ = sup θ∈V n i=1 [ϕ(θ, Z i )] 2 .
Thus, we may apply Lemma C.1 to V instead of Θ. Trivially, we have
N (V, ∥•∥ , ε) ≤ |V | = N.
Then, we have with probability 1 -2 exp(-t 2 ),
sup θ∈Θ n i=1 [ϕ(θ, Z i )] 2 ≤ Cσ √ n + 1 0 log N (Θ; ∥•∥ , ε)dε + t ≤ Cσ √ n + log(N ) + t , as desired.
Intuitively, in the special case when Θ is a polytope, the log-number of vertices becomes "effective dimension" instead of d, due to the fact that ϕ satisfies the convexity requirement.
In some cases, we can have d ≫ log(N ), in which the above gives a tighter concentration. In particular, this is a case in our construction for ℓ 1 geometry in Equation (17). With the above result, we can also establish the following refinement of Theorem 3.5.
Theorem E.6. Fix n ∈ N, d ∈ N, κ > 0 and α ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ), and suppose Θ is a polytope with N > 0 vertices. Let T be defined as, T := Tr κ (P; n, α).
Then, for some constant c > 0, every α-learner A n is (ξ, m)-traceable with ξ = exp(-cT 2 ), m = c n 2 T 2 n + log(N ) -16κ 2 n exp(n + log(N )) .
Proof. The proof is identical to Theorem 3.5, but using Lemma E.5 instead of Lemma 3.4, and thus, replacing d with log(N ) everywhere. We omit the details.
Now, recall the construction from Equation (17),
Z = {±1} d , Θ = B 1 (1) ∩ B ∞ (1/s), f (θ, Z) = -⟨θ, Z⟩ . (18
)
It is easy to see that f (•, Z) above is 1-Lipschitz w.r.t. ℓ 1 as Z ⊂ B ∞ (1). We have the following claim.
Lemma E.7. Let P s be as in (18), n ∈ N and 1/8 > α > 0. Then,
Tr κ (P s ; n, α) ≥ c √ s log d 16(s∨14) nα ,
where κ ≤ c ′ √ s for some constants c, c ′ > 0
Proof. We aim to use Lemma D.1 to characterize the subgaussian trace value. Consider the construction of the prior in Lemma D.1 with the following parameters: γ = 8α ≤ 1 and
β = 1 + 1 2 log d 16(s∨14)
. Then, by combining Lemma D.1 with and Proposition E.1, there exist a prior π and a family {Λ µ } of diagonal matrices with non-negative diagonal entries bounded by 1 from above, such that
E µ∼π E Z∼µ ⊗n θ, n i=1 Λ µ (Z i -µ) = 2β γ 2 E µ∼π sup θ ⟨θ, µ⟩ -α ≥ 2β γ 2 E µ∼π    1 s sup I⊂[d] |I|=s i∈I |µ i | -α    ≥ (a) β γ 2 γ 2 -2α ≥ log d 16(s∨14) 32α 2 • 2α ≥ log d 16(s∨14) 16α ,
where in (a) we used Proposition 5 of [SU17]. Since this holds in expectation over µ, it holds for at least one choice of µ. Let µ be that value. Now, let ϕ(θ, Z) = C -1/2 √ s θ, Λ(Z -µ) , where C is the absolute constant from Lemma A.7. For all θ, θ ′ ∈ Θ, we have
∥ϕ(θ, Z) -ϕ(θ ′ , Z)∥ ψ2 ≤ C -1/2 √ s ∥⟨θ ′ -θ, Λ (Z -µ)⟩∥ ψ2 ≤ (a) C -1/2 √ s ∥θ ′ -θ∥ 2 • C 1/2 max i Λ i Z i -µ i ψ2 ≤ √ s ∥θ ′ -θ∥ 2 ≤ (b) ∥θ ′ -θ∥ Θ ,
where in (a) we apply Lemma A.7, and in (b) we use that for every θ ∈ Θ, we have ∥θ∥ 2 ≤ 1/ √ s, thus, √ s ∥•∥ 2 ≤ ∥•∥ Θ . Plugging θ ′ = 0 gives ∥ϕ(θ, Z)∥ ψ2 ≤ ∥θ∥ Θ ≤ 1.
Thus, {ϕ(θ, Z)} θ∈Θ is a 1-subg process w.r.t. (Θ, ∥•∥ Θ ). Finally, we have
|ϕ(θ, Z)| ≤ C -1/2 √ s
Therefore, setting κ = C -1/2 √ s and noting that ϕ is linear (and therefore convex) in its first argument, we have
Tr κ (P s ; n, α) ≥ E µ∼π E Z∼µ ⊗n n i=1 1 n ϕ(θ, Z i ) ≥ C -1/2 √ s log d 16(s∨14) nα ,
as desired.
this section cite: ['b46']

Section: F Proofs of the main results (Section 2.2)

this section cite: []

Section: F.1 Proof of Theorem 2.5
Theorem 2.5. There exists a universal constant c > 0 such that, for all p ∈ [1, 2), if d, n, ξ ∈ (0, 1/e), and α > 0 are such that
c √ n ≤ α ≤ min c • d n 2 log(1/ξ) , 1 6 , (3
)
then there exist an ℓ p SCO problem that every α-learner is (ξ, m)-traceable with m ∈ Ω α -2 .
Proof. We begin by noting that the interval for α in Equation (3) is non-empty only if
c √ n ≤ min c • d n 2 log(1/ξ) , 1 6 ≤ c • d n 2 ,
where we used ξ < 1/e in the last transition. Via straightforward algebra, the above implies d ≥ n, thus, we may without loss of generality assume d ≥ n in the remainder of the proof.
Let P k,p be as in Equation (7), and set k as in Theorem 5.1. Then, Theorem 5.1 gives the following lower bound on the subgaussian trace value in this case:
T := Tr κ (P k,p ; n, α) ≥ c 2 √ d nα ≥ c 2 c log 1 ξ ,
for some κ ≤ c 1 √ d. Thus, provided c is small enough (c ≤ c 2 ), by Theorem 3.5, every α-learner is (ξ, m)-traceable with m satisfying, for some universal constant c ′ > 0,
m ≥ c ′ n 2 T 2 n + d - 16κ 2 n exp(n + d) ≥ c ′ c 2 2 d n + d • 1 α 2 - 16c 2 1 dn exp(n + d) ≥ c ′ c 2 2 2 • 1 α 2 - 16c 2 1 d 2 exp(d) , (19
)
where we used d ≥ n in the last inequality. Then, we have m ∈ Ω 1 α 2 , as desired.
this section cite: []

Section: F.2 Proof of Theorem 2.6
Then, the proof of Theorem 2.6 follows.
Theorem 2.6. There exists a universal constant c > 0 such that, if d is large enough and n, ξ ∈ (0, 1/e), and α > 0 are such that
c • log(d) n ≤ α ≤ min c • d 0.49 n log(1/ξ) , 1 8 , then (4
)
there exists a ℓ 1 SCO problem that every α-learner is (ξ, m)-traceable with m ∈ Ω log(d)/α 2 .
Proof. We begin by noting that the interval for α in Equation ( 4) is non-empty only if c • log(d) n ≤ c • d 0.49 n log(1/ξ) ≤ c • d 0.49 n where we used ξ < 1/e and c < 1/6 in the last transition. Via straightforward algebra, the above implies d 0.98 / log(d) ≥ n, thus, we may without loss of generality assume d 0.98 / log(d) ≥ n in the remainder of the proof. Let s = d 0.98 . Note that, letting V be the set of vertices of the polytope Θ = B 1 (1)∩B ∞ (1/s), we have V = z ∈ 0, ± 1 s : ∥z∥ 0 = s . The proof of this fact is straightforward and it is based on showing that every point in Θ can be written as a convex combination of the points in V . Thus, log |V | = log 2 s d s ≤ s log(de/s) + s = s log(de 2 /s). By the choice of s and using Lemma E.7, we have, for some κ ≤ c 1 √ s, T := Tr κ (P s ; n, α) ≥ c 2 d 0.49 log d 16(s∨14) nα ≥ (a) c 2 d 0.49 log d 16s nα = c 2 d 0.49 log d 0.02 16 nα = c 2 d 0.49 (0.02 • log (d) -log(16)) nα ≥ (b) c 2 d 0.49 • 0.01 • log (d) nα ≥ (c) log(1/ξ), where (a) and (b) hold provided d (and thus s) is large enough, and (c) holds provided c > 0 in Equation (4) is small enough. By Theorem E.6, every α-learner is (ξ, m)-traceable, where
m ≥ c ′ n 2 T 2 n + log |V | - 16κ 2 n exp(n + log |V |) ≥ c ′ 0.01 2 c 2 2 • d 0.98 log(d) n + log |V | • log(d) α 2 - 16c 2 1 • sn exp(n + log |V |)
.
Recall that d 0.98 ≥ n log(d) ≥ n (for d ≥ 3). Moreover, log |V | ≤ s log(de 2 /s) = d 0.98 log(e 2 d 0.02 ) ≤ Cd 0.98 log(d), for some universal C > 0. Thus, for d large enough, m ∈ Ω log(d) α 2 , as desired.
this section cite: []

Section: F.3 Proof of Theorem 2.7
Theorem 2.7. There exists a universal constant c > 0 such that, for all p ∈ [2, ∞), if d, n, ξ ∈ (0, 1/e), and α > 0 are such that
1 6 • min 1 n 1/p , d 1 2 -1 p √ n ≤ α ≤ min c • d n 2 log(1/ξ) 1/p , 1 6 , then(5)
there exist an ℓ p SCO problem such that every α-learner is (ξ, m)-traceable with m ∈ Ω (1/(6α) p ) .
Proof. Throughout, we assume c is a sufficiently small constant. Assume c < 1/6. Then, we begin by noting that the interval for α in Equation ( 5) is non-empty only if
1 6 • 1 n 1/p ≤ c • d n 2 log(1/ξ) 1/p ≤ 1 6 • d n 2 1/p ,
where we used ξ < 1/e and c < 1/6 in the last transition. Via straightforward algebra, the above implies d ≥ n, thus, we may without loss of generality assume d ≥ n in the remainder of the proof.
Let P k,p be as in Equation ( 7), and set k as in Theorem 5.1. Then, Theorem 5.1 gives the following lower bound on the subgaussian trace value
T := Tr κ (P k,p ; n, α) ≥ c 2 d 1-1/p nα ∧ √ d (6α) p/2 n (20
)
for some κ ≤ c 1 √ d. Note that, from (5), we have
α ≥ 1 6 • 1 n 1/p ≥ 1 6 • 1 d 1/p
Now, note that, the minimum in Equation ( 20) is achieved in the second term iff α ≥ d -1/p /6. Then, the lower bound on the subgaussian trace value becomes
T ≥ c 2 √ d (6α) p/2 n ≥ c 2 (6c) p/2 • log(1/ξ) ≥ log(1/ξ)
where the second transition follows from Equation (5) and the third transition holds whenever c > 0 is small enough (e.g., when c ≤ c 2/p 2 /6). Then, by Theorem 3.5 every α-learner is (ξ, m)-traceable with m satisfying, for some universal constant c ′ > 0,
m ≥ c ′ n 2 T 2 n + d - 16κ 2 n exp(n + d) ≥ c ′ c 2 2 d n + d • 1 (6α) p - 16c 2 1 dn exp(n + d) ≥ c ′ c 2 2 2 • 1 (6α) p - 16c 2 1 d 2 exp(d) ,(21)
where we used d ≥ n in the last transition. Thus, m ∈ Ω 1 (6α) p , as desired.
this section cite: []

Section: F.4 Proof of Theorem 2.8
Theorem 2.8. Let p ∈ [2, ∞). There exist a universal constant c > 0 and an ℓ p SCO problem P = (Θ, Z, f ) such that every (ε, δ)-DP learner of P with ε ≤ 1 and δ ≤ c/n satisfies,
α ≥ c • min d ε 2 n 2 1 p , d 1-1/p εn , 1 .
Proof. By Theorem 5.1, for some problem P, we have
T := Tr κ (P; n, α) ≥ c ′ d 1-1/p nα ∧ √ d nα p/2 Then Theorem 3.6 implies exp(ε) -1 ≥ c ′′ [T -2δκ] , Note that for all ε ≤ 1, we have 2ε ≥ exp(ε) -1. Thus, 2ε ≥ c ′ [T -2δκ] . which implies, 2(ε/c ′′ + δκ) ≥ T ≥ c ′ d 1-1/p nα ∧ √ d nα p/2 . Then, for some C > 0,
C(ε ∨ δκ) ≥ d 1-1/p nα ∧ √ d nα p/2 Rearranging gives α ≥ d 1-1/p Cn(ε ∨ δκ) ∧ √ d Cn(ε ∨ δκ) 2/p
Note that if ε ≥ δκ, the desired bound is immediate. For δκ ≥ ε, we have, since δ ≤ c/n and κ ≤ c ′ √ d,
α ≥ d 1-1/p C ′ √ d ∧ √ d C ′ √ d 2/p ≥ (C ′ ) -2/p ,
for some C ′ > 0, as desired.
this section cite: []

Section: F.5 Proof of Corollary 2.9
Corollary 2.9. Let Z = {±1} d , and suppose an estimator is given such that, given access to i.i.d. samples Z 1 , . . . , Z n ∈ Z, outputs μ with E ∥μ -E[Z 1 ]∥ ∞ ≤ α/2. Then, there exists a universal constant c > 0 such that, if d is large enough and n, ξ ∈ (0, 1/e), and α > 0 satisfy Equation (4), then the estimator μ is (ξ, m)-traceable with m ∈ Ω log(d)/α 2 .
Proof. We will first show that we can use the mean estimation algorithm to solve the corresponding hard problem for ℓ 1 -SCO. Specifically, consider the SCO problem as in Equation (17), and define the following learning algorithm based on the mean estimation. Let μ be the output of mean estimator based on the samples Z 1 , . . . , Z n , and let θ := arg max θ∈Θ ⟨θ, μ⟩ .
Let θ ⋆ be the population risk minimizer, and let µ be the true mean, that is, µ = E[Z 1 ]. Then, the excess risk of θ can be upper bounded as:
⟨θ ⋆ , µ⟩ -θ, µ = θ ⋆ -θ, µ = θ ⋆ -θ, µ -μ + θ ⋆ -θ, μ ≤ θ ⋆ -θ 1 ∥µ -μ∥ ∞ + θ ⋆ -θ, μ ≤ 2 ∥µ -μ∥ ∞ + θ ⋆ -θ, μ ,
where the last transition follows since θ, θ ⋆ both lie inside Θ ⊂ B 1 . Now, by the choice of θ, the second term is non-positive. Thus,
⟨θ ⋆ , µ⟩ -θ, µ ≤ 2 ∥µ -μ∥ ∞ .
Taking expectations on both sides, we get
E ⟨θ ⋆ , µ⟩ -θ, µ ≤ 2E ∥µ -μ∥ ∞ ≤ α.
Applying the result of Theorem 2.6 to θ, we conclude the proof.
this section cite: []

Section: G Connection between subgaussian trace value and non-private sample complexity
From the main part of the paper, we observe that the subgaussian trace value is typically inversely proportional to α. We start by proving two innocuous results (Propositions G.1 and G.3) that establish an absolute upper bound on subgaussian trace value. It will then allow us to extract lower bounds on α by plugging in our lower bounds on subgaussian trace value (Theorems G.2 and G.4). We start with the p ∈ (1, ∞) case, and then consider p = 1. Proof. Let (ϕ, D) be an arbitrary subgaussian tracer. Consider the process {X θ } θ∈Θ defined as
this section cite: []

Section: G
X θ := 1 n n i=1 ϕ(θ, Z i ),
where
S n = (Z 1 , . . . , Z n ) ∼ D ⊗n . We will argue that {X θ } θ∈Θ is O(1/ √ n)-subgaussian process w.r.t. (Θ, ∥•∥ Θ ). First, consider arbitrary θ 1 , θ 2 ∈ Θ. We have ∥X θ1 -X θ2 ∥ ψ2 = 1 n n i=1 (ϕ(Z i , θ 1 ) -ϕ(Z i , θ 2 )) ψ2 ≤ (a) C n n i=1 ∥ϕ(Z i , θ 1 ) -ϕ(Z i , θ 2 )∥ 2 ψ2 ≤ (b) C n n i=1 ∥θ 1 -θ 2 ∥ Θ = C √ n ∥θ 1 -θ 2 ∥ Θ ,
where in (a) we applied Lemma A.7, and in (b) we used the fact that {ϕ(θ, Z i )} θ∈Θ is 1-subgaussian process w.r.t. (Θ, ∥•∥ Θ ) for every i ∈ [n]. Moreover, for every θ ∈ Θ, we similarly have
∥X θ ∥ ψ2 = 1 n n i=1 ϕ(Z i , θ) ψ2 ≤ (a) C n n i=1 ∥ϕ(Z i , θ)∥ 2 ψ2 ≤ (b) C n n i=1 ∥θ∥ Θ = C √ n ∥θ∥ Θ ,
where in (a) we applied Lemma A.7, and in (b) we used the fact that {ϕ(θ, Z i )} θ∈Θ is 1-subgaussian process w.r.
t. (Θ, ∥•∥ Θ ) for every i ∈ [n]. Thus, {X θ } θ∈Θ is C/ √ n-subgaussian process w.r.t. (Θ, ∥•∥) as per Definition 3.1. Therefore, by Proposition A.10, we have, with probability at least 1 -4 exp(-t 2 ) sup θ∈Θ
1 n n i=1 ϕ(θ, Z i ) ≤ Ct √ n + C √ n 1 0 log N (Θ; ∥•∥ Θ , ε)dε ≤ Ct √ n + C √ n 1 0 d log 1 + 2 ε dε ≤ Ct √ n + C √ d √ n = C √ n √ d + t ,(22)
where in second inequality we use [Wai19, Example 5.8]. Hence,
E sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) = ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > u du ≤ C d n + ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > C d n + u du ≤ (a) C d n + C √ n ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > C d n + Ct √ n dt ≤ (b) C d n + C √ n ∞ 0 4 exp(-t 2 )dt ≤ C d n + C ′ √ n ≤ 2(C ∨ C ′ ) d n ,
where C ′ > 0 is some universal constant, (a) follows by a change of variables u = Ct/ √ n, and (b) follows from Equation (22). By Definition 2.3, we can write
Tr κ (P; n, α) = inf α-learnerAn sup T ∈Tκ E Sn=(Z1,...,Zn)∼D ⊗n , θ∼An(Sn)   1 n i∈[n] ϕ( θ, Z i )   ≤ sup T ∈Tκ E Sn=(Z1,...,Zn)∼D ⊗n   sup θ∈Θ 1 n i∈[n] ϕ(θ, Z i )   ≤ 2(C ∨ C ′ ) d n .
By rearranging the terms we obtain the desired result.
We now show that Theorem 5.1 implies lower bounds on the sample complexity of learning ℓ p -Lipshitz-bounded problems for every p ∈ [1, ∞). In particular, we show that the problems considered in Theorem 5.1 require many samples to learn (equivalently, we show a lower bound on optimal excess risk α).
Theorem G.2. Let α > 0, p ∈ [1, ∞) and n ∈ N be arbitrary. Let P k,p be as in Equation (7), and set k as in Theorem 5.1. Suppose there exist an α-learner for P k,p . Then, (i) for p ∈ [1, 2], we have α ≥ c √ n , (ii) for p ∈ (2, ∞), we have
α ≥ c 1 n 1/p ∧ d 1/2-1/p √ n ,
for some universal constant c > 0.
Proof. First, consider an arbitrary k ∈ [d]. We apply Proposition G.1 to the result of Lemma E.4. We then have the following double inequality
c 2 d 1-1/p k 1/2-1/p nα ≤ Tr κ (P k,p ; n, α) ≤ C d n .
Solving for α in the above, we have
α ≥ c 2 C • (d/k) 1/2-1/p √ n .
First, consider the case p ∈ [1, 2]. Then k = d, and we have
α ≥ c 2 C • 1 √ n ,
as desired. Now, consider the case p ∈ (2, ∞). Then k = (6α) p d ∨ 1, and we have
α ≥ c 2 C • (d/k) 1/2-1/p √ n = c 2 C (1/6α) p/2-1 √ n ∧ d 1/2-1/p √ n .
Solving for α, we have
α ≥ c 2 C6 p/2-1 2/p • 1 √ n ∧ c 2 C • d 1/2-1/p √ n .
Note that, since 2/p ≤ 1, we have
c 2 C6 p/2-1 2/p = 1 6 6c 2 C 2/p ≥ 1 6 1 ∧ 6c 2 C .
Thus,
α ≥ 1 6 ∧ c 2 C • 1 √ n ∧ d 1/2-1/p √ n ,as desired.
G.2 Lower bounds for p = 1. Now, consider the case p = 1. For p = 1, we consider the problem as in Equation ( 17). We will need the following refinement of Proposition G.1 in a special case when Θ is a polytope with few vertices. Intuitively, d in the statement Proposition G.1 can be replaced by log N where N is the number of vertices of Θ.
Proposition G.3. Fix n ∈ N, d ∈ N, and α ∈ [0, 1]. Consider an arbitrary SCO problem P = (Θ, Z, f ) in R d , where Θ is a polytope with N vertices. Let Tr κ (P; n, α) be the subgaussian trace value of problem P. Then, we have Tr κ (P; n, α) • √ n ≤ C • log N , for some universal constant C > 0.
Proof. Let (ϕ, D) be an arbitrary subgaussian tracer. Similarly to the proof of Proposition G.1, consider the process {X θ } θ∈Θ defined as
X θ := 1 n n i=1 ϕ(θ, Z i ),
where S n = (Z 1 , . . . , Z n ) ∼ D ⊗n . As in the proof of Proposition G.1, {X θ } θ∈Θ is a C/ √ nsubgaussian process w.r.t. (Θ, ∥•∥ Θ ) for some universal constant C > 0.
Let V be the set of vertices of Θ; then |V | = N , as per the proposition statement. Since ϕ is convex in its first argument, the mapping θ → X θ is also convex (almost surely). Then,
sup θ∈Θ |X θ | = sup θ∈V |X θ |.
Therefore, by Proposition A.10, we have, with probability at least 1 -4 exp(-t 2 )
sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) ≤ Ct √ n + C √ n 1 0 log N (V ; ∥•∥ Θ , ε)dε ≤ Ct √ n + C √ n 1 0 log N dε ≤ Ct √ n + C √ log N √ n .
Hence,
E sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) = ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > u du ≤ C log N n + ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > C log N n + u du ≤ (a) C log N n + C √ n ∞ 0 Pr sup θ∈Θ 1 n n i=1 ϕ(θ, Z i ) > C log N n + Ct √ n dt ≤ (b) C log N n + C √ n ∞ 0 4 exp(-t 2 )dt ≤ C log N n + C ′ √ n ≤
E Sn=(Z1,...,Zn)∼D ⊗n , θ∼An(Sn)   1 n i∈[n] ϕ( θ, Z i )   ≤ sup T ∈Tκ E Sn=(Z1,...,Zn)∼D ⊗n   sup θ∈Θ 1 n i∈[n] ϕ(θ, Z i )   ≤ 2(C ∨ C ′ ) log N n .
By rearranging the terms we obtain the desired result.
Then, sample complexity lower bounds for ℓ 1 geometry follow.
Theorem G.4. Let α > 0 and n ∈ N be arbitrary. Let P s be as in Equation (17) and set s = d 0.99 . Suppose there exists an α-learner for P s . Then, for large enough d, we have
α
≥ c log(d) n , for some universal constant c > 0. Proof. Lemma E.7 gives the following lower bound on the subgaussian trace value of P s , Tr κ (P s ; n, α) ≥ c √ s log d 16(s∨14) nα . At the same time, noting that Θ is a polytope with vertices given by V = z ∈ 0, ± 1 s d : ∥z∥ 0 = s , which has cardinality |V | = d s ≤ de s s , we have by Proposition G.3 Tr κ (P s ; n, α) ≤ C s log(de/s) n , for some universal C > 0. Combining this with the lower bound on subgaussian trace value, we have c √ s log d 16(s∨14) nα ≤ C s log(de/s) n , which gives α ≥ c C log d 16(s∨14) log(de/s)n .
Recall that s = d 0.99 . For large enough d, we have s ∨ 14 = s. Also, note that log(d/s) ≥ log(d)/100. Then, for for large enough d, we have α ≥ c ′ log(d) n , for some universal c ′ > 0, as desired.
this section cite: []

Section: H Traceability of VC classes (Section 1.1.1)
First, we state the main result. Theorem H.1. Fix n ∈ N and ξ < 0.1. Let H be an arbitrary VC concept class with VC dimension d vc . Then, there exists an optimal algorithm in terms of number of samples such that it is (ξ/(n log(n), m)-traceable with m ≤ O d vc log 2 (n) . Moreover, when H is the class of thresholds, we have m ∈ O(1).
To prove it we use an information-theoretic notion that controls the difficulty of tracing from [SZ20].
Definition H.2. Fix n ∈ N. Let D be a data distribution, and A n be a learning algorithm. For every n ∈ N, let Z = (Z j,i ) j∈{0,1},i∈[n] be an array of i.i.d. samples drawn from D, and U = (U 1 , . . . , U n ) ∼ Ber 1 2 ⊗n , where U and Z are independent. Define training set
S n = (Z Ui,i ) i∈[n] . Then, define CMI D (A n ) := I A n (S n ); U Z .
In the next theorem, we show that the existence of a tracer for a learning algorithm provides a lower bound on the CMI of the algorithm. A similar observation is made in [ADHLR24]. Lemma H.3. Fix n ∈ N such that n ≥ 2 and ξ < 1/2. Let A n be an arbitrary learning algorithm that is (ξ/(n log(n), m)-traceable. Then, it holds sup D CMI D (A n ) ≥ m -3ξ.
The following two results from [SZ20] and [HDMR21] provide upper bounds on the CMI of sample compression schemes. We skip the formal definitions of sample compression schemes and refer the reader to
Pr (G i = 0) = E Pr G i = 0 U = Pr Sn∼D ⊗n ,Z∼D, θ∼An(Sn) ϕ( θ, Z) ≥ λ ≤ ξ/(n. log(n)), Therefore, CMI D (A n ) ≥ n i=1 Pr (Y i = 1) - ξ log(n) -nH b ξ n log(n)
By the recall condition from Definition 2.3 and the definition of CMI in Definition H.2, we have
m = E Sn=(Z1,...,Zn)∼D ⊗n , θ∼An(Sn) i ∈ [n] : ϕ( θ, Z i ) ≥ λ = n i=1 Pr ϕ( θ, Z i ) ≥ λ ≤ n i=1 Pr ϕ( θ, Z Ui,i ) ≥ λ ∧ G i + Pr (G c i ) = n i=1 Pr (Y i = 1) + n i=1 Pr (G c i ) ≤ n i=1 Pr (Y i = 1) + ξ log(n) .
We also use the following well-known inequality, H b (x) ≤ -x log(x) + x for x ∈ [0, 1]. As a result, we obtain
CMI D (A n ) ≥ n i=1 Pr (Y i = 1) - ξ log(n) -nH b ξ n log(n) ≥ m - 2ξ log(n) -n ξ n. log(n) - ξ (n. log(n)) log ξ (n. log(n)) ≥ m -ξ 1 e + 3 log(n) ,
where the last step follows because -x log(x) ≤ 1/e for x ∈ [0, 1].
this section cite: ['b47', 'b5', 'b47', 'b27']

Section: References
Ref_id:b0 Title: Information-theoretic lower bounds on the oracle complexity of convex optimization Year: (2009)
Ref_id:b1 Title: Private PAC learning implies finite Littlestone dimension Year: (2019)
Ref_id:b2 Title: SGD generalizes better than GD (and regularization doesn't help) Year: ()
Ref_id:b3 Title: Differentially private generalized linear models revisited Year: (2022)
Ref_id:b4 Title: Private stochastic convex optimization: Optimal rates in l1 geometry Year: ()
Ref_id:b5 Title: Information Complexity of Stochastic Convex Optimization: Applications to Generalization and Memorization Year: (2024)
Ref_id:b6 Title: Private stochastic convex optimization with optimal rates Year: (2019)
Ref_id:b7 Title: Non-Euclidean differentially private stochastic convex optimization Year: ()
Ref_id:b8 Title: Learners that Use Little Information Year: (2018)
Ref_id:b9 Title: Private empirical risk minimization: Efficient algorithms and tight error bounds Year: (2014)
Ref_id:b10 Title: Inequalities for the gamma function Year: (2008)
Ref_id:b11 Title: Proper learning, Helly number, and an optimal SVM bound Year: ()
Ref_id:b12 Title: When is memorization of irrelevant training data necessary for high-accuracy learning? Year: (2021)
Ref_id:b13 Title: An equivalence between private classification and online prediction Year: (2020)
Ref_id:b14 Title: Differentially private release and learning of threshold functions Year: (2015)
Ref_id:b15 Title: Fingerprinting codes and the price of approximate differential privacy Year: (2014)
Ref_id:b16 Title: Score attack: A lower bound technique for optimal differentially private learning Year: (2023)
Ref_id:b17 Title: Membership inference attacks from first principles Year: (2022)
Ref_id:b18 Title: Differentially private empirical risk minimization Year: (2011)
Ref_id:b19 Title: Memorize to generalize: on the necessity of interpolation in high dimensional linear regression Year: ()
Ref_id:b20 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b21 Title: Robust traceability from trace amounts Year: (2015)
Ref_id:b22 Title: Generalization of ERM in Stochastic Convex Optimization: The Dimension Strikes Back Year: (2016)
Ref_id:b23 Title: Does learning require memorization? a short tale about a long tail Year: (2020)
Ref_id:b24 Title: Private stochastic convex optimization: optimal rates in linear time Year: (2020)
Ref_id:b25 Title: Generalization for adaptively-chosen estimators via stable median Year: ()
Ref_id:b26 Title: Private convex optimization in general norms Year: (2023)
Ref_id:b27 Title: Towards a unified information-theoretic framework for generalization Year: (2021)
Ref_id:b28 Title: Sharpened generalization bounds based on conditional mutual information and an application to noisy, iterative algorithms Year: (2020)
Ref_id:b29 Title: Limitations of Information-Theoretic Generalization Bounds for Gradient Descent Methods in Stochastic Convex Optimization Year: (2023)
Ref_id:b30 Title: Train faster, generalize better: Stability of stochastic gradient descent Year: ()
Ref_id:b31 Title: Resolving individuals contributing trace amounts of DNA to highly complex mixtures using high-density SNP genotyping microarrays Year: (2008)
Ref_id:b32 Title: The Composition Theorem for Differential Privacy Year: (2017)
Ref_id:b33 Title: Privately Learning High-Dimensional Distributions Year: (2018)
Ref_id:b34 Title: Privately learning high-dimensional distributions Year: (2019)
Ref_id:b35 Title: New lower bounds for private estimation and a generalized fingerprinting lemma Year: (2022)
Ref_id:b36 Title: The Power of Sampling: Dimension-free Risk Bounds in Private ERM Year: (2024)
Ref_id:b37 Title: Relating data compression and learnability Year: (1986)
Ref_id:b38 Title: Information theoretic lower bounds for information theoretic upper bounds Year: (2024)
Ref_id:b39 Title: Fingerprinting Codes Meet Geometry: Improved Lower Bounds for Private Query Release and Adaptive Data Analysis Year: (2024)
Ref_id:b40 Title: Sample compression schemes for VC classes Year: (2016)
Ref_id:b41 Title: Problem complexity and method efficiency in optimization Year: (1983)
Ref_id:b42 Title: White-box vs black-box: Bayes optimal strategies for membership inference Year: ()
Ref_id:b43 Title: Stochastic Convex Optimization Year: (2009)
Ref_id:b44 Title: Membership Inference Attacks against Machine Learning Models Year: (2017)
Ref_id:b45 Title: Convex Games in Banach Spaces Year: (2010)
Ref_id:b46 Title: Tight lower bounds for differentially private selection Year: (2017)
Ref_id:b47 Title: Reasoning about generalization via conditional mutual information Year: ()
Ref_id:b48 Title: Upper and lower bounds for privacy and adaptivity in algorithmic data analysis Year: (2016)
Ref_id:b49 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b50 Title: High-dimensional statistics: A non-asymptotic viewpoint Year: (2019)
Ref_id:b51 Title: Information-theoretic analysis of generalization capability of learning algorithms Year: (2017)
