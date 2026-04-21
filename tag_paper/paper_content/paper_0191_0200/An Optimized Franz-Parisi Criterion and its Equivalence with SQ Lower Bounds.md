Title: An Optimized Franz-Parisi Criterion and its Equivalence with SQ Lower Bounds
Abstract: Bandeira et al. (2022) introduced the Franz-Parisi (FP) criterion for characterizing the computational hard phases in statistical detection problems. The FP criterion, based on an annealed version of the celebrated Franz-Parisi potential from statistical physics, was shown to be equivalent to low-degree polynomial (LDP) lower bounds for Gaussian additive models, thereby connecting two distinct approaches to understanding the computational hardness in statistical inference. In this paper, we propose a refined FP criterion that aims to better capture the geometric "overlap" structure of statistical models. Our main result establishes that this optimized FP criterion is equivalent to Statistical Query (SQ) lower bounds-another foundational framework in computational complexity of statistical inference. Crucially, this equivalence holds under a mild, verifiable assumption satisfied by a broad class of statistical models, including Gaussian additive models, planted sparse models, as well as non-Gaussian component analysis (NGCA), single-index (SI) models, and convex truncation detection settings. For instance, in the case of convex truncation tasks, the assumption is equivalent with the Gaussian correlation inequality (Royen, 2014) from convex geometry. In addition to the above, our equivalence not only unifies and simplifies the derivation of several known SQ lower bounds-such as for the NGCA model ( Diakonikolas et al., 2017) and the SI model (Damian et al., 2024)-but also yields new SQ lower bounds of independent interest, including for the computational gaps in mixed sparse linear regression (Arpino et al., 2023) and convex truncation (De et al., 2023).

Section: Introduction
Over the past decades, a central focus in statistical inference has been to understand the transition from computationally easy to hard regimes-that is, to characterize when a statistical task can be solved by polynomial-time algorithms. A key insight from this line of work is the emergence of computational-statistical tradeoffs: in many models, there exist broad parameter regimes where information-theoretic recovery is possible, yet no known polynomial-time algorithm succeeds.
Evidence for such tradeoffs spans multiple disciplines with varying levels of mathematical rigor. In particular, the statistical physics community has played an instrumental role by leveraging nonrigorous but highly predictive techniques to study average-case hardness. Their approach typically analyzes the geometry of solution spaces and identifies structural properties that correlate with algorithmic intractability (see [40] for a survey). Remarkably, for many statistical models, the predictions from statistical physics have been in striking agreement with the performance of the best-known polynomial-time algorithms.
Alongside these heuristic predictions, rigorous frameworks from statistics and theoretical computer science have been developed to analyze the limitations of efficient algorithms. While ruling out all polynomial-time algorithms would require resolving P ̸ = N P, substantial progress has been made by studying broad, expressive classes of polynomial-time algorithms. Two frameworks have emerged as particularly influential: low-degree (LD) polynomial lower bounds [31] and statistical query (SQ) lower bounds [20]. For many "nice enough" detection problems, the lower bounds derived from these frameworks align closely with the performance of the best-known polynomial-time algorithms 1 . This striking consistency has motivated the formulation of the so-called low-degree conjecture by Hopkins [26], which posits that for sufficiently "symmetric and noisy" models, the failure of degree-O(log n) polynomials is indicative of the failure of all polynomial-time algorithms [31].
Given this context, a natural question arises: can one formally connect these two seemingly distinct approaches? At first glance, the answer appears negative, due to a fundamental mismatch in scope. Statistical physics techniques are primarily geared toward estimation problems, where the goal is to recover a hidden signal, while the rigorous frameworks discussed above-such as LD and SQ lower bounds-are focused on detection or hypothesis testing, where the task is to distinguish between the presence or absence of a signal in a noisy environment 2 . Nevertheless, a major step towards bridging this gap was taken by Bandeira et al. (2022) [6], who introduced the Franz-Parisi (FP) criterion for computational hardness in detection tasks. Inspired by the seminal work of Franz and Parisi in spin glass theory [21], the FP criterion provides a geometric perspective on computational hardness rooted in overlap structures. Crucially, Bandeira et al. showed that for Gaussian additive models, the FP criterion is mathematically equivalent to the low-degree (LD) lower bounds, thereby establishing a rigorous link between statistical physics heuristics and formal algorithmic barriers.
Specifically, consider the following general detection problem between two distributions P and Q supported on a subset of R n , which in what follows we refer to as a "P versus Q" task. Under the planted distribution P = E u P u , a signal u is drawn from a prior distribution π supported on Θ ⊆ S N -1 , and one observes m independent samples Y 1 , . . . , Y m ∼ P u . Under the null distribution Q, the samples are drawn independently from Y 1 , . . . , Y m ∼ Q. The goal in the detection task 3 is to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. Note that the computational question then is whether such a successful test statistic exists that also terminates in polynomial-in-mn time. To characterize the hardness of detection problems from the statistical physics perspective, Bandeira et al. in [6] introduced the following notion of Franz-Parisi (FP) hardness: Definition 1 (FP hardness). For D, m ∈ N, ε > 0, we say that a P versus Q detection task is (q, m, ε)-FP hard if
FP: E ⟨L ⊗m u , L ⊗m v ⟩ • 1(|⟨u, v⟩| ≤ δ(q)) ≤ 1 + ε, where(1)
δ(q) = sup{δ > 0 : π 2 (|⟨u, v⟩| ≥ δ) ≥ q -2 }.(2)
In the definition we denoted as customary L u = dPu dQ , u ∈ Θ, and for f, g ∈ L 2 (R n ), the Hilbert space L 2 (Q) of (square integrable) functions from R n to R, we use ⟨f,
g⟩ Q = E Y ∼Q f (Y )g(Y ).
We elaborate in Section A.1 on the statistical physics motivations behind this criterion, and only briefly highlight its core intuition here. The left-hand side of the FP condition integrates the function F ann (t) := E [⟨L ⊗m u , L ⊗m v ⟩ • 1(⟨u, v⟩ = t)] , over a (1 -q -2 )-typical region of the overlap variable t, corresponding to the constraint |⟨u, v⟩| ≤ δ(q). This function F ann (t) is an annealed proxy for the Franz-Parisi potential, a central object in statistical physics that has long served as a predictor of algorithmic hardness [40]. Intuitively, the Franz-Parisi potential captures the energy landscape experienced by local algorithms-such as Langevin or Glauber dynamics-whose performance is constrained by the geometry of the underlying signal space. The overlap ⟨u, v⟩ naturally quantifies a local "geometric" similarity between signals, making it a meaningful argument for F ann (t) and explaining its role within the FP criterion.
Returning to the definition of FP hardness, the parameter m corresponds to the sample size, and one should interpret q as a proxy for the required runtime. In this light, Bandeira et al. (2022) proved that, for Gaussian additive models, FP-hardness is equivalent to the failure of degree-D = log q polynomials to solve the detection task with m samples-i.e., roughly the authors of [6] showed that the problem is (q, m, O(1))-FP hard if and only if it is "hard" for degree-log q polynomials to solve the detection task 4 . Hence, based on the current belief in the literature of low-degree lower bounds that a D-degree lower bound implies that the detection task requires at least e D runtime to be solved, e.g., see [18], proving a task is ((mn) ω(1) , m, O(1))-FP hard for a Gaussian additive model provides rigorous evidence for polynomial-time hardness for the task.
Despite this success, the connection between the FP potential and other rigorous notions of algorithmic hardness remains limited. [6] only established a formal equivalence for Gaussian additive models and an one-sided implication for planted sparse models between the FP criterion and "low-degree" lower bounds. They further presented counterexamples where the equivalence fails entirely. In this work, our aim is to extend the Franz-Parisi criterion to rigorously characterize hardness beyond Gaussian additive models, and to clarify the scope and limitations of this framework across a broader class of statistical models.
this section cite: ['b39', 'b30', 'b19', 'b25', 'b30', 'b5', 'b20', 'b5', 'b39', 'b5', 'b17', 'b5']

Section: Main Contributions
Our main contribution is to propose a slight modification of the FP-hardness criterion from [6], motivated by the observation that sticking to the Euclidean geometry assumption (and hence the "overlap" ⟨u, v⟩) may fail to capture the "true" hardness of some statistical models. We remark that this is an arguably natural modification, as (1) there are many statistical models for which the Euclidean geometry appears unnatural for navigating their parameter space (see Section 5 for a simple such construction), and (2) even in statistical physics settings, the Franz-Parisi potential is often considered under a more general notion of overlap [22]. Motivated by these considerations, we propose optimizing the "overlap" event inside the FP-hardness definition, subject only to a mild symmetry assumption for technical reasons. This leads to the following new criterion of FP-hardness: Definition 2 (Generalized Franz Parisi (GFP) hardness under symmetry G). Fix q, m ∈ N, ε > 0 and a group G of finite order acting on the parameter space of the signal. We say a "P versus Q" problem is (q, m, ε)-GFP G hard if
GFP G : inf A:π ⊗2 (A)≥1-q -2 A is G 2 -invariant E L ⊗m u , L ⊗m v Q 1(A) ≤ 1 + ε.(3)
As in the original FP-hardness framework of [6], one should interpret q as a proxy for runtime, and therefore ((mn) ω(1) , m, O(1))-GFP hardness should be providing evidence of polynomial-time hardness with m samples in this framework. We highlight that the assumption on the invariance of the optimizing event under group G is made for technical reasons to enhance the applicability of our hardness criterion. We point the reader to Section 3.1 for further discussion of this assumption.
The main result of this work is that the "optimized" notion of GFP-hardness is fundamentally connected with the well-established framework of Statistical Query (SQ) hardness. The SQ framework was initially proposed by Kearns in [30] to capture the power of noise-tolerant algorithms. The notion of a statistical dimension proposed by [20] allowed for achieving powerful lower bounds against SQ methods, which we refer to from now on as SQ-hardness results. We employ here a slight strengthening of the notion of SQ-hardness from [20], introduced in [8].
Definition 3 (SQ hardness). Fix q, m ∈ N. We say a "P versus Q" detection problem is (q, m)-SQ hard if SQ:
sup A:π 2 (A)≥q -2 E ⟨L u , L v Q -1 | A ≤ 1 m .(4)
Roughly, a detection problem is (q, m)-SQ hard if any Statistical Query method succeeding at solving it with m samples requires q queries, which should be interpreted as requiring runtime q (see [8, Appendix A] for more details and motivation). Hence, proving a task is ((mn) ω(1) , m)-SQ hard provides evidence for polynomial-time hardness for the task.
Our main result is informally described as follows.
Theorem 1. (Informal, GFP and SQ equivalence) Consider any P versus Q detection task which we assume (1) it satisfies a mild assumption with respect to a group G of finite order acting on the parameter space (namely Assumption 1 below), and (2) it is information-theoretically impossible to be solved with m IT samples. Then the following holds for any samples size m and proxy runtime q = m Ω(1) .
• If the task is (q, m)-SQ hard, then it is also (Θ(q), Θ(m), O(1))-GFP G -hard.
• If the task is (q, m, O(1))-GFP G -hard, then it is also (m Θ(mIT) , m 1-o(1) )-SQ hard.
Note that often in statistical tasks of interest m IT = ω(log n) (in fact, more often than not m IT = poly(n)). Under this condition, Theorem 1 implies that a task is ((mn) ω( 1) , m 1-o(1) , O(1))-GFP hard if and only if it is ((mn) ω( 1) , m 1-o(1) )-SQ hardness, matching the two criteria for hardness.
On top of that, as we mentioned above and discuss in Section 3.1, the required Assumption 1 on the detection task is rather mild. In fact, it turns out that it is satisfied for several models of recent interest in the community, making a strong case of how the Generalized Franz-Parisi criterion now correctly predicts the hardness phase for them. Importantly, these models include the Gaussian additive models and also greatly extend beyond them, significantly extending the key message from [6] about connecting the physics-based forms of hardness to more rigorous frameworks. We list now some of the tasks that satisfy Assumption 1.
1. All Gaussian additive models (GAMs), under any symmetric prior, satisfy Assumption 1 with G = Z 2 that flips the sign of the signal. Moreover, in that case the Generalized Franz Parisi criterion is equivalent to the Franz-Parisi criterion, that is the optimizing event A in (3) is of the form {|⟨u, v⟩| ≤ δ(q)}. Hence, Theorem 1 allows us to extend the result of [6] which proved the equivalence of FP-hardness to Low-degree hardness for GAMs, to also proving FP-hardness equivalent with SQ-hardness for these settings 5 .
this section cite: ['b5', 'b0', 'b21', 'b5', 'b29', 'b19', 'b19', 'b7', 'b5', 'b5']

Section: 2.
All Planted Sparse Models satisfy Assumption 1 for the trivial group G = {id}. In particular, using Theorem 1 we can prove that GFP-hardness is equivalent to SQ-hardness for multiple well-studied models such as sparse phase retrieval [5], sparse regression [24,6], (multisample) sparse PCA [7], and Bernoulli group testing [11]. As a corollary of this connection, we present a straightforward argument to obtain an SQ lower bound for the mixed sparse linear regression problem [5]. We remark that in [6] it has been proven that FP-hardness implies low-degree hardness for all Planted Sparse Models, but no result was presented for the other direction.
3. All Non-Gaussian component analysis (NGCA) models and all single-index models (under any symmetric prior) satisfy Assumption 1 with G = Z 2 , Therefore, via Theorem 1 GFPhardness is again equivalent with SQ-hardness for these tasks.
this section cite: ['b4', 'b23', 'b5', 'b6', 'b10', 'b4', 'b5']

Section: 4.
All Gaussian convex truncation models satisfy Assumption 1 for G = {id}. In particular, interestingly Assumption 1 for these models is exactly equivalent to the celebrated Gaussian correlation inequality for convex bodies in probability theory, which was a multi-decade open problem posed in 1972 in [25] that was finally proven by Royen in 2014 [37]. Leveraging the equivalence between GFP-hardness and SQ-hardness in Theorem 1, we establish an SQlower bound for the convex truncation detection task. This allows us to provide, to the best of our knowledge, the first formal evidence that the current state-of-the-art polynomial-time detection method for convex truncation proposed in [15] has optimal sample complexity.
We also complement our results, with a simple example satisfying Assumption 1 where FP-hardness does not coincide with GFP-hardness, which we interpret as a model where the Euclidean geometry is not appropriate. We finally conclude the paper with a discussion.
For completeness, we prove in Appendix B the equivalence between GFP-hardness and low-degree (LD) polynomial hardness for noise-robust models. This result follows by combining our GFP-SQ equivalence with the equivalence between SQ-hardness and LD-hardness under noise robustness shown by Brennan et al. [8]. In particular, our GFP-hardness results for the examples presented in this paper immediately imply low-degree lower bounds in all those settings. This substantially extends the equivalence established in [6]. For clarity and readers' convenience, we also include succinct proofs of the SQ-to-LD equivalence, adapted from [8].
this section cite: ['b24', 'b36', 'b14', 'b7', 'b5', 'b7']

Section: Setting and Definitions
We first recall the definition of a "P versus Q" task mentioned in the Introduction. Under the planted distribution P = E u P u , a signal u is drawn from a prior distribution π supported on Θ ⊆ S N -1 , and one observes m independent samples Y 1 , . . . , Y m ∼ P u . Under the null distribution Q, the samples are drawn independently from Y 1 , . . . , Y m ∼ Q. The goal in the detection task 6 is the so-called strong detection task to distinguish between these two hypotheses based on the observed data, that is to find a test statistics with vanishing Type I and Type II errors, as n grows. We will also be interested in the weak detection task, which is that the sum of type I and type II errors is at most 1 -ε for some fixed ε > 0 (not depending on n). In other words, strong detection means the test succeeds with high probability, while weak detection means the test has some non-trivial advantage over random guessing.
Throughout, we will work in the Hilbert space L 2 (Q) of (square integrable) functions R N → R with inner product ⟨f, g⟩ Q := E Y ∼Q [f (Y )g(Y )] and corresponding norm ∥f ∥ Q := ⟨f, f ⟩ 1/2 Q . We will assume that P u is absolutely continuous with respect to Q for all u ∈ supp(π), use L u := dPu dQ to denote the likelihood ratio, and assume that L u ∈ L 2 (Q) for all u ∈ supp(π). The likelihood ratio between P and Q is denoted by L := dP dQ = E u∼µ L u . Observe that for m samples, we denote by L m = E u∼µ L u the m-sample likelihood ratio. Finally, for a function f : R N → R and integer D ∈ N, we let f ≤D denote the orthogonal (w.r.t. ⟨•, •⟩ Q ) projection of f onto the subspace of polynomials of degree at most D.
An important identity between the (squared) norm of the likelihood ratio with m samples and the chi-squared divergence χ 2 (P ⊗m ∥ Q ⊗m ) is
∥L∥ 2 Q = ∥E u∼µ L u ∥ 2 Q = χ 2 (P ∥ Q) + 1 ≥ 1 .
This quantity has the following standard implications for information-theoretic impossibility of testing, in the asymptotic regime n → ∞. The proofs can be found in e.g. [34,Lemma 2].
• If ∥L∥ 2 Q = O(1) then strong detection is impossible.
• If ∥L∥ 2 Q = 1 + o(1) then weak detection is impossible.
this section cite: ['b33']

Section: Main Results
In this section, we formally present our equivalence between GFP-hardness and SQ-hardness.
this section cite: []

Section: The Assumption
As mentioned in the Introduction, all our results operate under a crucial assumption on the "P versus Q" detection task. The assumption is as follows. Assumption 1. Given any "P versus Q" task, there exists a π-preserving finite group G acting on the parameter space Θ, i.e., for all g ∈ G, g(v)
= v for v ∼ π, such that for any sample size m for any u, v ∈ Θ, the following "correlation" inequality holds for any k ∈ N
E g,g ′ ∼Unif(G) (⟨L g(u) , L g ′ (v) ⟩ Q -1) k ≥ 0.(5)
We first remark that (5) is a natural condition even if G is the trivial group, G = {id}. Indeed in that case (5) asks that for all u, v ∈ Θ,
⟨L u , L v ⟩ Q ≥ 1.(6)
Recall that if one averages over all (u, v) ∼ π ⊗2 , we have by standard identities
E⟨L u , L v ⟩ Q = E Q ∥E u L u ∥ 2 2 = 1 + χ 2 (P, Q) ≥ 1.
Thus, (6) should be understood as a pointwise condition that is guaranteed to hold in expectation over the product measure π ⊗2 for any P, Q. While this pointwise condition turns out to be vanilla satisfied in many models (such as Planted Sparse Models or Convex Truncation settings), a slight modification of it-leading to (5)-applies more broadly. Specifically, this modified condition requires (6) to hold for a pair u, v only after performing a "small" averaging over the a group orbit that preserves the prior π. For instance, if the prior is symmetric around 0 and the group G is Z 2 , which acts by flipping the sign of the signal u, then for k = 1, condition (5) reduces to demonstrating that, for all u, v,
1 4 (E⟨L u , L v ⟩ Q + E⟨L -u , L v ⟩ Q + E⟨L u , L -v ⟩ Q + E⟨L -u , L -v ⟩ Q ) ≥ 1,
which is significantly less restrictive than the original pointwise condition (6). This averaging approach allows for much greater generality, making it applicable to various settings, including Gaussian additive models, single-index models, and Non-Gaussian component analysis settings.
Remark 3.1. We finally make a trivial remark that will be useful in verifying (5) in our examples in Section 4 with symmetric prior. In all of them by symmetry we have for all u, v
⟨L u , L -v ⟩ Q = ⟨L -u , L v ⟩ Q and ⟨L u , L v ⟩ Q = ⟨L -u , L -v ⟩ Q .
Using that and the trivial fact that for all x, y ∈ R, if x + y ≥ 0 then x k + y k ≥ 0 for all k ∈ N, we conclude that if G is either the trivial group or Z 2 (which will be the case in all examples of Section 4) it suffices to check the case k = 1 in (5), and then it automatically holds for all k ∈ N.
Remark 3.2. As mentioned in the previous remark, we highlight that in all our examples in Section 4 of our GFP-SQ equivalence theorem below, we either use G to be the trivial group or Z 2 . The reason we state our assumption Assumption 1 for a general finite group G is for potential further applications of our work.
this section cite: ['b4', 'b4']

Section: 3.2
The GFP-SQ equivalence
this section cite: []

Section: Simplifying GFP-hardness
We present our equivalence theorem in two steps. First, we identify an approximate optimal "overlap" event A in the definition of GFP-hardness, which simplifies GFP-hardness significantly and makes GFP-hardness easier to establish in applications. Then, we prove the equivalence between this simplified version and SQ-hardness.
Given a group G acting on the parameter space of the signal, it turns out that the approximately optimal "overlap" event A takes the form {ρ G (u, v) ≤ r} for the following notion of "overlap" between u, v,
ρ G (u, v) = max g,g ′ ∈G {|⟨L g(u) , L g ′ (v) ⟩ Q -1|}.
In particular, focusing only on such type of events we define the following version of FP-hardness.
Definition 4 (ρ G -FP hardness). Fix q, m ∈ N, ε > 0 and a finite group G acting on the parameter space of the signal. We say a "P versus Q" problem is (q, m, ε)-ρ G -FP hard if
ρ G -FP : E ⟨L ⊗m u , L ⊗m v ⟩ Q • 1(ρ G (u, v) < r(q)) ≤ 1 + ε, where(7)
r(q) = sup{r :
π 2 (ρ G (u, v) ≥ r) ≥ q -2 },(8)
We prove that GFP G -hardness is equivalent to ρ G -FP hardness under Assumption 1.
Theorem 2. Consider any "P versus Q" task that satisfies Assumption 1 for a group G. Suppose m, q ∈ N and ε > 0. Then the following statements hold.
1. If the task is (q, m, ε)-ρ G -FP hard, then the task is also (q, m, ε)-GFP G hard.
2. Assume there exists an r > 0 such that π 2 (ρ G (u, v) < r) = 1 -q -2 and that m is even. Then, if the task is (q, m, ε)-GFP G hard, then it is (q, m, 3
|G| (1 + ε) + m • χ 2 (P, Q))-ρ G -FP hard. In particular, if mχ 2 (P, Q) = O(1), the task is (q, m, O(1 + ε))-ρ G -FP hard.
The proof of this theorem can be found in Appendix C.1.
this section cite: []

Section: Remark 3.3.
While the first implication is immediate to grasp, the second implication has some additional conditions we now elaborate upon. First, both the requirements of the existence of r with the desired probability mass and the parity of m are for technical convenience, and both can be easily remove with some tedious work. Second, any potential "blow-up" in the ε-term for ρ G -FP hard depends only on |G|, which should be treated as constant, and the term m • χ 2 (P, Q), which is an easy to compute quantity (usually n = 1 and χ 2 (P, Q) is an one-dimensional integral). Moreover, it is almost always of order O(1) for detection tasks that are conjecturally hard with m samples. Indeed, the mathematical reason behind this is exactly that it is equal to the squared L 2 -norm of the projection of the likelihood onto the degree-1 polynomial space, i.e., on the span of linear functions. On top of that, if the detection task is (q, m)-SQ hard for any q then it holds directly mχ 2 (P, Q) = O(1) as well. We elaborate more on this in Remark B.1 in Section B.
this section cite: []

Section: The equivalence
As we have already proven an equivalence between GFP-hardness and ρ G -FP hardness, it suffices to connect the latter with SQ-hardness. This is the topic of the next theorem.
Theorem 3 (SQ and ρ G -FP Equivalence). Suppose a "P versus Q" task satisfies Assumption 1 for a group G.
1. If the task is (q, m)-SQ hard for some q, m with q > 2 then, it is also (q ′ , m ′ , e|G| -1 m ′ /m)ρ G -FP hard for any integers q ′ < q/ √ 2 and m ′ ≤ m/2.
2. Suppose the task is (q, m, ε)-ρ G -FP hard for some q, m integers. Assume that there exists an r = r(q) > 0 such that π 2 (ρ G (u, v) < r) = 1 -q -2 and m is even. Then, the model is also (q ′ , m ′ )-SQ hard for any even integer t with t ≤ log q/ log m and any integer q ′ > 0, where
m ′ = m (t(1 + ε) 1/t + χ 2 (P ⊗4t ∥ Q ⊗4t ))(q ′ ) 2/t .
In particular, if for some sample size m IT , we have (a) (Bounded χ 2 for m IT samples)
χ 2 (P ⊗mIT ∥ Q ⊗mIT ) = O(1). (b) (Large enough q) q ≥ m mIT
then the model is (m δmIT , Θ( m 1-O(δ)  mIT(1+ε) ))-SQ hard for any δ > 0.
The proof of this theorem can be found in Appendix C.2. Similar to Theorem 2, the conditions on r, m of part 2 in Theorem 3 are for technical convenience and can be easily removed. As we discussed in the Introduction the assumption that there exists some sufficiently growing m IT (e.g., growing super-logarithmically in n) is natural for multiple commonly studied models. We remark that the condition on the information theory threshold m IT to be growing with n is also necessary, by constructing a variant of the planted clique problem which satisfies Assumption 1, it is not SQ-hard and is GFP-hard. Lastly, we also note that our introduced Assumption 1 is also necessary for the equivalence. In Section A, we discuss a counterexample not satisfying Assumption 1 that is GFP-hard, but not SQ-hard. Remark 3.4. We note that while our bounds in the equivalence of Theorem 2 deteriorate when |G| becomes large, a slightly more general equivalence between GFP and SQ, using a variant of ρ G , can also be proven for infinite groups G under an "hypercontractivity" assumption on ⟨L u , L v ⟩ Q with respect to the pair (u, v). We omit this generalization as in all relevant examples in this work a small group action using either the trivial or 2-cyclic group suffices.
this section cite: []

Section: Examples
In this section, we discuss two popular classes of detection tasks that satisfy Assumption 1 and hence fall under our GFP-SQ equivalence. Further examples are deferred to Appendix D.
this section cite: []

Section: Gaussian Additive Models
A P versus Q task is a Gaussian additive model (GAM) if it satisfies:
1. Under the null model, Q = N (0, I n ).
2. Under the planted model P u (for u ∈ S n-1 ), for some signal-to-noise ratio (SNR) λ > 0 we set Y = λu + Z, for some Z ∼ Q.
GAMs includes multiple well-studied models in the literature, with the predominant examples being (multisample variants) of tensor PCA [36] and sparse PCA [2]. For such models, it can be straightforwardly checked (see [6,Proposition 2.3]) that for all u, v,
⟨L u , L v ⟩ Q = e λ 2 ⟨u,v⟩ .
So for instance, in the case of non-negative sparse PCA where u, v are binary k-sparse vectors in (see e.g., [4,10]) we always have ⟨u, v⟩ ≥ 0, and therefore Assumption 1 is always satisfied for the trivial group G. On top of that, Assumption 1 remains true for any prior which is symmetric around 0; this time Assumption 1 is also always satisfied by choosing the action of G = Z 2 which flips the sign of u. We remark that symmetric priors encompass most commonly used priors for GAMs, e.g., for tensor PCA where u = vec(x ⊗r ), x ∼ Unif(S d-1 ). Lemma 1. Consider any GAM with symmetric π, i.e.
, v = -v, v ∼ π. For any u, v ∈ support(π), 1 4 (⟨L u , L v ⟩ Q + ⟨L -u , L v ⟩ Q + ⟨L u , L -v ⟩ Q + ⟨L -u , L -v ⟩ Q ) ≥ 1.
Moreover, any GAM satisfies Assumption 1 for G = Z 2 acting by flipping the sign of u.
Proof. Notice 1 4 (⟨L u , L v ⟩ Q +⟨L -u , L v ⟩ Q +⟨L u , L -v ⟩ Q +⟨L -u , L -v ⟩ Q ) = 1 2 (exp λ 2 ⟨u, v⟩ +exp -λ 2 ⟨u, v⟩ ) ≥ 1.
Hence, given Remark 3.1, the conclusion follows.
Given the above lemma, we conclude the (almost) equivalence between GFP-hardness and SQhardness from Theorem 3. Remark 4.1. We remark that in the symmetric prior case for a GAM and G = Z 2 acting by flipping the sign of u, ρ G (u, v) = exp λ 2 |⟨u, v⟩| is an increasing function of |⟨u, v⟩|. Hence, for such GAMs we conclude via Theorem 2 that FP-hardness is equivalent to GFP-hardness, and therefore also to SQ-hardness. This is in agreement with the results of [6] establishing that FP-hardness is equivalent to LD-hardness; in fact, our approach can offer an alternative proof of their result via the LD-SQ equivalence [8] and the noise robustness of GAMs (see Theorem Theorem 6).
this section cite: ['b35', 'b1', 'b5', 'b3', 'b9', 'b5', 'b7']

Section: Planted Sparse Models
In [6], the authors introduced the family of planted sparse models (PSM) and proved that FP-hardness for a PSM implies it's also low-degree hard. We start with the definition.
A P versus Q task is a planted sparse model (PSM) if it satisfies:
1. Under the null model, the one sample is given by Y = (Y 1 , . . . , Y n ) ∼ Q, where each entry
Y i , i = 1, . . . , n is drawn independently from some distribution Q i , i = 1, . . . , n on R.
2. Under the planted model P u , we associate u with a set of planted entries Φ u ⊂ [n]. Then on sample is generated as follows. For the entries i / ∈ Φ u , we draw Y i independently from Q i (which is identical as in the Q measure). For the entries in Φ u we draw from an arbitrary joint distribution P u | Φu with the following symmetry condition: for any subset S ⊆ Φ u , the marginal distribution
P u | ϕu (S) does not depend on u but only on S, i.e. P u | S = P S .
Multiple well-known detection models satisfy this definition, such as, a well studied model of sparse regression [24,6], Bernoulli group testing [1,11], sparse phase retrieval [5], as well as multi-sample variants [8] of planted clique [29] and sparse (Wigner) PCA [2].
Satisfyingly, all planted sparse models directly satisfy Assumption 1 for the trivial group. In fact this result has already been established for a different use in [6, Proposition 3.6], proving that any u, v we have ⟨L u , L v ⟩ Q ≥ 1. We state here for completeness. Lemma 2. Consider any PSM. For any u, v ∈ support(π), ⟨L u , L v ⟩ Q ≥ 1. This is to say, any PSM satisfies Assumption 1 for the trivial group.
The proof follows from [6, Proposition 3.6] for D = 0. Using this, one can apply our main equivalence Theorem 3 to multiple interesting planted sparse models and obtain old and new SQhardness results in a rather streamlined fashion. As an instantiation of this, in Appendix D.1 we prove the GFP-hardness for the mixed sparse linear regression setting studied in [5] in its conjecturally hard regime. We then use our equivalence theorem to translate it into an SQ-hardness result in the same regime. Our results complement the existing low-degree lower bound [5], providing further evidence for the hard phase of the problem.
this section cite: ['b5', 'b23', 'b5', 'b0', 'b10', 'b4', 'b7', 'b28', 'b1', 'b4', 'b4']

Section: Other examples
Due to space constraint, we defer the following examples to Appendix D:
• Non-Gaussian Component Analysis (NGCA): Assumption 1 holds with G = Z 2 for any symmetric prior. We recover the SQ-hardness result of [16] for the uniform prior via its equivalence with GFP-hardness, and establish a new SQ lower bound under a sparse prior.
• Single-Index Models (SIM): Again, Assumption 1 holds with G = Z 2 . We rederive the SQ-hardness result of [12] for the uniform prior through the GFP-hardness equivalence, and prove a new SQ lower bound for sparse priors.
• Convex truncation detection: Here Assumption 1 holds with the trivial group. In fact this assumption is precisely equivalent to the celebrated Gaussian Correlation Inequality on convex bodies [37,32]. Using the GFP-hardness correspondence, we derive a new SQ lower bound that matches the current state-of-the-art polynomial-time algorithm of [15].
this section cite: ['b15', 'b11', 'b36', 'b31', 'b14']

Section: GFP-hardness is not always equal to FP-hardness
Recall that by definition, FP-hardness implies GFP-hardness. In this section, we show that the converse does not necessarily hold: we construct a P versus Q detection task that satisfies Assumption 1 and is easy under the FP criterion but hard under the GFP criterion. In particular, by using Theorem 2 and Theorem 3, the problem is also SQ-hard. Thus, while the FP criterion fails to capture the SQ-hardness in this case, our optimized GFP criterion correctly predicts it. As our initial departure from FP-hardness was that in many models the Euclidean overlap ⟨u, v⟩ might not be the "correct" choice, our example is carefully creating a model where the natural "overlap" ρ G (u, v) (based on Theorem 3) is not a function of the Euclidean dot product.
The P versus Q problem is defined as follows. The null model is Q = Rad 1 2 ⊗(n+1) , i.e., each coordinate is an independent Rademacher random variable. For a signal u ∈ {0, 1} n+1 , the sample x ∼ P u is generated by drawing each coordinate independently according to
x i = +1, w.p. 1 2 + r • 1-(1-α)•ui 2 , -1, w.p. 1 2 -r • 1-(1-α)•ui 2 , (9
)
where α, r ∈ (0, 1) are fixed constants to be chosen later. The following holds.
Lemma 3. Let u, v ∈ {0, 1} n+1 . For any u, v ∈ {0, 1} n+1 , ⟨L u , L v ⟩ Q = n i=0 1 + r 2 • α ui+vi .
Notice that our construction importantly ensures that the likelihood ratio inner product ⟨L u , L v ⟩ is not solely a function of ⟨u, v⟩, but instead has a more intricate dependence on u and v. It is exactly this reason that leads to the discrepancy between GFP and FP hardness stated below. Theorem 4. There exist a two-point prior π on u such that, for r = n -1/2 , α = n -1+2ε , m = n 1-ε and D = n ε , where ε > 0 is any small constant, the following hold. The m-sample hypothesis testing problem
E u∼π P ⊗m u versus Q ⊗m is (e D/2 , m, Θ(n -ε ))-GFP hard but not (n -1 , m, exp (Θ(n ε )))-FP hard. Moreover, via our equivalence theorem the model is (e n Θ(ε) , n 1-Θ(ε) )-SQ hard.
The proof of this Theorem and the above Lemma can be found in Appendix E.
this section cite: []

Section: Conclusion
In this work, we generalize the Franz-Parisi (FP) criterion introduced by [6], motivated by the observation that the Euclidean dot product may not be the most natural geometry for all statistical task-a point partially illustrated by our example in Section 5. Our main result shows that optimizing the overlap event in the FP definition of [6] leads to a Generalized Franz-Parisi (GFP) hardness criterion, which is equivalent to SQ-hardness for models satisfying the mild Assumption 1. This assumption holds in a broad range of well-studied problems, including Gaussian additive models, planted sparse models, single-index models, and convex truncation. Our work significantly strengthens the theoretical foundation behind the (annealed) FP potential's predictions from statistical physics, but also opens several questions:
1. (Algorithmic implications) Does the optimal overlap function ρ G (u, v)-as characterized in
Theorem 2-yield meaningful algorithmic insights, particularly for local search or geometric methods?
2. (The annealed potential) Can similar equivalences be established for the original (also known as quenched) FP potential, or is the choice of the annealed version fundamental?
3. (Interpretation of FP Area) Why does the area under the FP curve appear to govern detection hardness? Is there some physical/algorithmic interpretation of this phenomenon?
4. (Generalization to estimation) Can our techniques be extended from detection to estimation tasks, for which the Franz-Parisi potential was originally introduced?
We believe these questions point toward promising future directions, with the potential to unify different approaches on the computational complexity of statistical inference.
this section cite: ['b5', 'b5']

Section: References
Ref_id:b0 Title: Group testing: an information theory perspective Year: (2019)
Ref_id:b1 Title: High-dimensional analysis of semidefinite relaxations for sparse principal components Year: (2008)
Ref_id:b2 Title: Algorithmic thresholds for tensor pca Year: (2020)
Ref_id:b3 Title: Free energy wells and overlap gap property in sparse pca Year: (2023)
Ref_id:b4 Title: Statistical-computational tradeoffs in mixed sparse linear regression Year: (2023)
Ref_id:b5 Title: The franz-parisi criterion and computational trade-offs in high dimensional statistics Year: (2022)
Ref_id:b6 Title: Universality of computational lower bounds for submatrix detection Year: (2019)
Ref_id:b7 Title: Statistical query algorithms and low degree tests are almost equivalent Year: (2021)
Ref_id:b8 Title: Can neural networks achieve optimal computational-statistical tradeoff? an analysis on single-index model Year: ()
Ref_id:b9 Title: On the low-temperature mcmc threshold: the cases of sparse tensor pca, sparse regression, and a geometric rule Year: (2024)
Ref_id:b10 Title: Statistical and computational phase transitions in group testing Year: (2022)
Ref_id:b11 Title: Computational-statistical gaps in gaussian single-index models Year: (2024)
Ref_id:b12 Title: Efficient statistics, in high dimensions, from truncated samples Year: (2018)
Ref_id:b13 Title: Computationally and statistically efficient truncated regression Year: (2019)
Ref_id:b14 Title: Testing convex truncation Year: (2023)
Ref_id:b15 Title: Statistical query lower bounds for robust estimation of high-dimensional gaussians and gaussian mixtures Year: (2017)
Ref_id:b16 Title: Efficient algorithms and lower bounds for robust linear regression Year: (2019)
Ref_id:b17 Title: Subexponential-time algorithms for sparse pca Year: (2024)
Ref_id:b18 Title: Curse of heterogeneity: Computational barriers in sparse mixture models and phase retrieval Year: (2018)
Ref_id:b19 Title: Statistical algorithms and a lower bound for detecting planted cliques Year: (2017)
Ref_id:b20 Title: Recipes for metastable states in spin glasses Year: (1995)
Ref_id:b21 Title: Effective potential in glassy systems: theory and simulations Year: (1998)
Ref_id:b22 Title: An examination into the registered speeds of american trotting horses, with remarks on their value as hereditary data Year: (1898)
Ref_id:b23 Title: Sparse high-dimensional linear regression. estimating squared error and a phase transition Year: (2022)
Ref_id:b24 Title: Inequalities on the probability content of convex regions for elliptically contoured distributions Year: (1970)
Ref_id:b25 Title: Statistical Inference and the Sum of Squares Method Year: (2018)
Ref_id:b26 Title: Near-optimal statistical query lower bounds for agnostically learning intersections of halfspaces with gaussian marginals Year: (2022)
Ref_id:b27 Title: Semiparametric least squares (sls) and weighted sls estimation of singleindex models Year: (1993)
Ref_id:b28 Title: Large cliques elude the metropolis process Year: (1992)
Ref_id:b29 Title: Efficient noise-tolerant learning from statistical queries Year: (1998)
Ref_id:b30 Title: Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio Year: (2019)
Ref_id:b31 Title: Royen's proof of the gaussian correlation inequality Year: (2017)
Ref_id:b32 Title: Generalized linear models Year: (2019)
Ref_id:b33 Title: On the limitation of spectral methods: From the gaussian hidden clique problem to rank-one perturbations of gaussian tensors Year: (2015)
Ref_id:b34 Title: On the systematic fitting of curves to observations and measurements Year: (1902)
Ref_id:b35 Title: A statistical model for tensor pca Year: (2014)
Ref_id:b36 Title: A simple proof of the gaussian correlation conjecture extended to multivariate gamma distributions Year: (2014)
Ref_id:b37 Title: Computational barriers to estimation from low-degree polynomials Year: (2022)
Ref_id:b38 Title: Lattice-based methods surpass sum-of-squares in clustering Year: (2022)
Ref_id:b39 Title: Statistical physics of inference: Thresholds and algorithms Year: (2016)
