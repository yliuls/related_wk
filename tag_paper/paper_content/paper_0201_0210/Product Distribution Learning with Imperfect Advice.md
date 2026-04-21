Title: Product distribution learning with imperfect advice
Abstract: Given i.i.d. samples from an unknown distribution P , the goal of distribution learning is to recover the parameters of a distribution that is close to P . When P belongs to the class of product distributions on the Boolean hypercube {0, 1} d , it is known that Ω(d/ε 2 ) samples are necessary to learn P within total variation (TV) distance ϵ. We revisit this problem when the learner is also given as advice the parameters of a product distribution Q. We show that there is an efficient algorithm to learn P within TV distance ε that has sample complexity Õ(d 1-η /ε 2 ), if ∥p -q∥ 1 < εd 0.5-Ω(η) . Here, p and q are the mean vectors of P and Q respectively, and no bound on ∥p -q∥ 1 is known to the algorithm a priori.

Section: Introduction
Science fundamentally relies on the ability to learn models from data. In many real-world settings, the majority of available datasets consist of unlabeled examples -sample points drawn without corresponding labels, outputs or classifications. These unlabeled datasets are often modeled as samples from a joint probability distribution on a large domain. The goal of distribution learning is to output the description of a distribution that approximates the underlying distribution that generated the observed samples. See [Dia16] for a comprehensive survey.
In practice, distribution learning rarely occurs in isolation. While a given dataset may be new, one often has access to previously learned models from related datasets. Alternatively, the data may arise from an evolving process, motivating the reuse of information from past learning to guide current inference. This prior information can be viewed as a form of "advice" or "prediction" of some kind. In the framework of algorithms with predictions, the objective is to integrate such advice in a way that improves performance when the advice is accurate, while ensuring robustness: performance should not degrade beyond that an advice-free baseline algorithm, even when the predictions are inaccurate. Most previous works in this setting are in the context of online algorithms, e.g. for the ski-rental problem [GP19, WLW20, ADJ + 20], non-clairvoyant scheduling [PSK18], scheduling [LLMV20,BMRS20,AJS22], augmenting classical data structures with predictions (e.g. indexing [KBC + 18] and Bloom filters [Mit18]), online selection and matching problems [AGKK20, DLPLV21, CGLB24, CJS25], online TSP [BLMS + 22, GLS23], and a more general framework of online primal-dual algorithms [BMS20]. However, there have been some recent applications to other areas, e.g. graph algorithms [CSVZ22, DIL + 21], causal learning [CGB23], mechanism design [GKST22, ABG + 22], and most relevantly to us, distribution learning [BCGG25].
In this work, we study the problem of learning product distributions over the d-dimensional Boolean hypercube {0, 1} d , arguably one of the most fundamental classes of discrete high-dimensional distributions. A product distribution P is fully specified by its mean vector p ∈ [0, 1] d , where the i-th coordinate p i represents the expectation of the i-th marginal of P , or equivalently, the probability that the i-th coordinate of a sample from P is 1. It is well-known that Θ(d/ε 2 ) samples from a product distribution P are both necessary and sufficient to learn a distribution P such that d TV (P, P ) ≤ ε with probability at least 2/3, where d TV denotes the total variation distance. This optimal sample complexity is achieved by a simple, natural and efficient algorithm: computing the empirical mean of each coordinate. Motivated by the framework of algorithms with predictions, we investigate whether this sample complexity can be improved when, in addition to samples from P , the learner is given an advice mean vector q ∈ [0, 1] d . Importantly, we make no assumption that q is close to the true mean p. However, if we can detect that q is accurate -i.e., that ∥q -p∥ is small in an appropriate normcan this information be leveraged to constrain the search space and improve sample or computational efficiency? Our goal is to design algorithms that adapt to the quality of the advice: performing better when q is accurate, while remaining robust when it is not.
Our main result establishes that this is indeed possible. Specifically, we show that if ∥q-p∥ 1 ≪ ε √ d, then there exists a polynomial-time algorithm with sample complexity sublinear in d that outputs a distribution P such that d TV (P, P ) ≤ ε with probability at least 2/3. More precisely, under the regularity condition that no coordinate of P is too close to deterministic (i.e., bounded away from 0 and 1), we show that the sample complexity is:
Õ d ε 2 d -η + min 1, ∥p -q∥ 2 1 d 1-4η ε 2
for any small enough constant η. In particular, when ∥p -q∥ 1 is small, the dependence on d becomes sublinear. We also prove that the non-determinism assumption is necessary: if coordinates of P can be arbitrarily close to 0 or 1, then sample complexity that is sublinear in d is impossible, even when ∥q -p∥ 1 = O(1). Furthermore, we show that when ∥q -p∥ 1 ≫ ε √ d, no algorithm with sublinear sample complexity exists.
this section cite: ['b15', 'b29', 'b27', 'b7', 'b4', 'b28', 'b23', 'b8', 'b11', 'b5']

Section: Technical Overview
We call a product distribution P balanced if no marginal of P is too biased. That is, each coordinate of p is bounded away from 0 and 1. It is known that, for balanced distributions, learning P in TV distance is equivalent to learning the mean vector p with ℓ 2 -error. Hence, in this overview, we focus on the latter task.
To build intuition about how an advice vector q can be exploited, consider the following two situations: 1. Exact advice: Suppose q = p. Then, it suffices to verify that ∥q -p∥ 2 ≤ ε and a learning algorithm can simply return q. This is the classic identity testing problem, which has been extensively studied (see [Can20] for a detailed survey). For product distributions, Daskalakis and Pan [DP17] and Canonne, Diakonikolas, Kane and Stewart [CDKS17] independently showed that identity testing requires Θ( √ d/ε 2 ) samples. This demonstrates that sublinear sample complexity is achievable when q = p. Morever, Ω(
√ d/ε 2
) is a fundamental lower bound that applies even when q = p. 2. Sparse disagreement: Suppose q differs from p in at most t coordinates, i.e. ∥q -p∥ 0 ≤ t.
In this case, one only needs to estimate p on those t coordinates, so the information-theoretic sample complexity should scale as ∼ log d t . Unfortunately, since t is unknown a priori, we cannot directly exploit this sparsity. However, from the compressive sensing literature, it is known that closeness in ℓ 2 norm to a t-sparse vector can be certified by a small ℓ 1 norm (e.g., Theorem 2.5 of [FR13]). Motivated by the above, we take the quality of the advice q to be governed by the ℓ 1 -distance ∥p -q∥ 1 , and aim for an algorithm whose sample complexity improves as this distance decreases. Suppose we can certify that ∥p -q∥ 1 ≤ λ. Then, we may restrict attention to the ℓ 1 -ball of radius λ centered at q, and cover it with N ℓ 2 -balls of radius ε, where the covering number N grows polynomially as d O(λ 2 /ε 2 ) . It is known (e.g. see Chapter 4 of [DL01]) that using the Scheffé tournament method, the sample complexity of learning p up to ℓ 2 -norm ε scales as (log N )/ε 2 , which yields a bound of O( λ 2 ε 4 log d). While the Scheffé tournament is computationally inefficient, the same sample complexity guarantee can be achieved efficiently by solving a constrained least squares problem. More precisely, given samples x 1 , . . . , x n from P , we consider the estimator:
argmin b∈R d :∥b-q∥1≤λ 1 n n i=1 ∥x i -b∥ 2 2
For n = O(λ 2 ε -4 log d), this estimates achieves ℓ 2 -error at most ε.
The key challenge that remains is then to approximate λ ≈ ∥p -q∥ 1 using a sublinear number of samples from P . To this end, we devise a new identity testing algorithm that, using O( √ d/ε 2 ) samples, either (i) 2-approximates ∥p -q∥ 2 , or (ii) certifies that ∥p -q∥ 2 ≤ ε, in which we simply return q. If we are in case (i), we can upper bound ∥p -q∥ 1 with λ = ∥p -q∥ 2 • √ d.
However, this would make the sample complexity of the learning algorithm to be
O λ 2 ε 4 log d = O d log(d)•∥p-q∥ 2 2 ε 4 ≫ d ε 2 , i.e.
, exceeding the standard O(d/ε 2 ) bound and defeating the purpose. To improve upon this, we can partition the d coordinates into d/k blocks of size k each. Then, within each block, the ratio between the ℓ 1 and ℓ 2 norms improves from √ d to √ k. By appropriately choosing k, we can obtain a non-trivial reduction in overall sample complexity.
The structure of our algorithm described above and its analysis parallels the recent work by [BCGG25], which addresses the problem of learning Gaussian distributions with imperfect advice. However, our setting differs in several important ways:
• [BCGG25] used a well-known algorithm to approximate the ℓ 2 norm of a Gaussian's mean vector. In contrast, our ℓ 2 -approximation algorithm for product distribution is new, to the best of our knowledge.
• We critically rely on the balancedness assumption to relate total variation distance and ℓ 2 error of the mean vector. No such assumption is needed in the Gaussian setting. In fact, we show that for product distributions, balancedness is essential: without it, no sublinear-sample algorithm exists, even when the advice vector is O(1)-close to the truth in ℓ 1 distance. We find this somewhat surprising since O(
√ d/ε 2
) samples suffice without any balancedness assumptions in identity testing [DP17,CDKS17].
this section cite: ['b9', 'b20', 'b10', 'b21', 'b18', 'b5', 'b5', 'b20', 'b10']

Section: Preliminaries
A distribution P on {0, 1} d is said to be a product distribution if there exist distributions P 1 , . . . ,
P d on {0, 1} such that P (x) = P 1 (x 1 ) • P 2 (x 2 ) • • • P d (x d ) for every x ∈ {0, 1} d . In this case, we can write P = P 1 ⊗ P 2 • • • ⊗ P d . Definition 2.1 (Mean vectors). The mean vector of a distribution P is p ≜ E x∼P [x]. In particular, if P = P 1 ⊗ • • • ⊗ P d is a product distribution, p = [p 1 • • • p d ], where p i = P i (1).
For a vector p ∈ [0, 1] d , we denote by Ber(p) the product distribution with mean vector p. We next define the notion of balancedness. Definition 2.2. For τ ∈ [0, 1/2], a product distribution P on {0, 1} d is said to be τ -balanced if for every i ∈ [d], the marginal P i satisfies τ ≤ P i (1) ≤ 1 -τ . Proposition 2.3 (e.g., [CDKS17], Lemma 1). Suppose P and Q are τ -balanced product distributions on {0, 1} d with mean vectors p and q respectively. Then their KL divergence d KL (P ∥Q) satisfies:
2∥p -q∥ 2 2 ≤ d KL (P ∥Q) ≤ 2 τ ∥p -q∥ 2 2 .
Proposition 2.4. Suppose P and Q are τ -balanced product distributions on {0, 1} d with mean vectors p and q respectively. Then, for some constant c < 0.2, the TV distance d TV (P, Q) satisfies:
c • min{1, ∥p -q∥ 2 } ≤ d TV (P, Q) ≤ 1 √ τ ∥p -q∥ 2 .
Proof. The first inquality is the main result of [Kon25]. The second inequality follows from applying Pinsker to the upper bound on d KL in Proposition 2.3.
Note that the dependence on τ above is necessary (up to constant factors). For example, suppose P = Ber(0) and Q = Ber(u) where 0 is the all-zero vector and u = [
1 d , 1 d , . . . , 1 d ]. Then, ∥0 -u∥ 2 = 1/ √ d, while d TV (P, Q) ≥ P (0) -Q(0) = 1 -(1 -1/d) d ≈ 1 -1/e.
this section cite: ['b10', 'b26']

Section: Algorithm
The goal of this section is to establish the following result.
Theorem 3.1. There exists algorithm TESTANDOPTIMIZEMEAN that for any given ε, δ, τ ∈ (0, 1), η ≥ 0, and q ∈ [0, 1] d , and sample access to a τ -balanced product distribution Ber(p) on {0,
1} d , it draws n = Õ d ε 2 • (d -η + min{1, f (p, q, d, η, ε)}) i.i.d.
samples from Ber(p), where:
f (p, q, d, η, ε) = ∥p -q∥ 2 1 d 1-4η τ 6 ε 2 .
The algorithm produces as output p in poly(n, d) time such that d TV (Ber(p), Ber( p)) ≤ ε with success probability at least 1 -δ.
A basic component of the algorithm is a test to determine how close the advice q is to the true p in ℓ 2 norm. Lemma 3.2 (Tolerant mean tester). Given ε > 0, δ ∈ (0, 1), d sufficiently large integer, and q ∈ [0, 1] d , there is a tolerant tester TMT that uses O
√ d ε 2 log 1 δ i.i.d.
samples from Ber(p) and satisfies both conditions below with probability at least 1 -δ:
1. If ∥p -q∥ 2 ≤ ε, then the tester outputs Accept 2. If ∥p -q∥ 2 ≥ 2ε, then the tester outputs Reject Proof. Notice that if p i = q i = 1, we could interpret p 1 , . . . , p d and q 1 , . . . , q d as distributions p and q on [d] that sample i ∈ [d] with probability p i and q i respectively. Diakonikolas and Kane [DK16] showed that using O(∥p∥ 2 /ε 2 ) samples from p, one can test whether ∥p -q∥ 2 ≤ ε or ∥p -q∥ 2 ≥ 2ε. Inspired by this observation, we mimic the analysis of [DK16] to devise a tolerant tester for general product distributions.
Assume the desired failure probability to be 1/3; we can reduce to any δ by repeating the test O(log 1/δ) times and taking the majority vote. Set m = c √ d/ε 2 for a large enough constant c, and let m i be sampled independently from Poi(m) for each i ∈ [d]. Note that max i m i ≤ 2em with high probability; we condition on this event and set the desired failure probability to 1/4. Therefore, using 2em samples from Ber(p), for each i, we can obtain m i samples from the ith coordinate, and let X i be the number of times the i'th coordinate is sampled to be 1. Note that by standard properties of the Poisson distribution, the X 1 , . . . , X d are independent and each X i is sampled from Poi(mp i ).
Define the statistic Z = d i=1 Z i , where:
Z i = (X i -mq i ) 2 -X i .
Using similar calculations as in [DK16], we can show that:
E[Z i ] = m 2 (p i -q i ) 2
and E[Z] = m 2 ∥p -q∥ 2 2 . Also, we can calculate the variance to be:
Var[Z] = 4m 3 d i=1 p i (p i -q i ) 2 + 2m 2 d i=1 p 2 i ≤ 4m 3 ∥p∥ 2 ∥p -q∥ 2 4 + 2m 2 ∥p∥ 2 2 ,
where the inequality is by Cauchy-Schwarz.
If ∥p -q∥ 2 ≤ ε, then E[Z] ≤ c 2 d/ε 2 , and Var[Z] ≤ (4c 3 + 2c 2 )d 2 /ε 4 . On the other hand, it always holds that E[Z] ≥ c 2 d∥p -q∥ 2 2 /ε 4 , and Var[Z] ≤ (4c 3 d 1.5 ∥p∥ 2 ∥p -q∥ 2 2 /ε 2 + 2c 2 d∥p∥ 2 2 )/ε 4 ≤ (4c 3 ∥p -q∥ 2 2 /ε 2 + 2c 2 )d 2 /ε 4 , since ∥p∥ 2 ≤ √ d. Using Chebyshev's inequality, if c is large enough, when ∥p -q∥ 2 ≤ ε, Z is at most 2c 2 d/ε 2 with probability 3/4, but when ∥p -q∥ 2 ≥ 2ε, Z is at least 3c 2 d/ε 2 with probability 3/4. Algorithm 1 The APPROXL1 algorithm. Proof. We begin by stating some properties of o 1 , . . . , o w . Fix an arbitrary index j ∈ {1, . . . , w} and suppose o j is not a Fail, i.e. the tolerant tester of Lemma 3.2 outputs Accept for some i * ∈ {1, 2, . . . , ⌈log 2 ζ/α⌉}. Note that APPROXL1 sets o j = ℓ i * and the tester outputs Reject for all smaller indices i ∈ {1, . . . , i * -1}. Since the tester outputs Accept for i * , we have that ∥p Bjq Bj ∥ 2 ≤ 2ℓ i * = 2o j . Meanwhile, if i * > 1, then ∥p Bj -q Bj ∥ 2 > ℓ i * -1 = ℓ i * /2 = o j /2 since the tester outputs Reject for i * -1. Thus, we see that
• When o j is not Fail, we have ∥p Bj -q Bj ∥ 2 ≤ 2o j .
• When ∥p Bj -q Bj ∥ 2 ≤ 2α, we have i * = 1 and o j = ℓ 1 = α.
• When ∥p Bj -q Bj ∥ 2 > 2α = 2ℓ 1 , we have i * > 1 and so o j < 2∥p Bj -q Bj ∥ 2 .
this section cite: ['b17', 'b17', 'b17']

Section: Success probability.
Fix an arbitrary index i ∈ {1, 2, . . . , ⌈log 2 ζ/α⌉} with ℓ i = 2 i-1 α, where ℓ i ≤ ℓ 1 = α for any i. We invoke the tolerant tester with ε = ℓ i in the i th invocation, so the i th invocation uses at most m 1 (k, ε, δ) ≜ n k,ε • r δ i.i.d. samples to succeed with probability at least 1 -δ, where n k,ε ≜ ⌈ 16 √ k 3ε 2 ⌉ and r δ ≜ 1 + ⌈log(12/δ)⌉. So, with at most m(k, α, δ) ≜ m 1 (k, α, δ ′ ) = n k,α • r δ ′ samples, any call to the tolerant tester succeeds with probability at least 1 -δ ′ , where δ ′ ≜ δ w•⌈log 2 ζ/α⌉ . By construction, there will be at most w • ⌈log 2 ζ/α⌉ calls to the tolerant tester. Therefore, by union bound, all calls to the tolerant tester jointly succeed with probability at least 1 -δ.
Proof of Property 1. When APPROXL1 outputs Fail, there exists a Fail amongst {o 1 , . . . , o w }. For any fixed index j ∈ {1, . . . , w}, this can only happen when all calls to the tolerant tester outputs Reject. This means that ∥x Bj ∥ 2 > ε 1 = ℓ i = 2 i-1 • α for all i ∈ {1, 2, . . . , ⌈log 2 ζ/α⌉}. In particular, this means that ∥x Bj ∥ 2 > ζ/2.
Proof of Property 2. When APPROXL1 outputs λ = 2 w j=1 |B j | • o j ∈ R, we can lower bound λ as follows:
λ = 2 w j=1 |B j | • o j ≥ 2 w j=1 |B j | • ∥p Bj -q Bj ∥ 2 2 (since ∥p Bj -q Bj ∥ 2 ≤ 2o j ) ≥ w j=1 ∥p Bj -q Bj ∥ 1 (since ∥p Bj -q Bj ∥ 1 ≤ |B j | • ∥p Bj -q Bj ∥ 2 ) = ∥p -q∥ 1 (since w j=1 ∥p Bj -q Bj ∥ 1 = ∥p Bj -q Bj ∥ 1 )
That is, λ ≥ ∥p -q∥ 1 . Meanwhile, we can also upper bound λ as follows:
λ = 2 w j=1 |B j | • o j ≤ 2 √ k w j=1 o j (since |B j | ≤ k) = 2 √ k •      w j=1 ∥p B j -q B j ∥2≤2α o j + w j=1 ∥p B j -q B j ∥2>2α o j      (partitioning the blocks based on ∥p Bj -q Bj ∥ 2 versus 2α) = 2 √ k •      w j=1 ∥p B j -q B j ∥2≤2α α + w j=1 ∥p B j -q B j ∥2>2α o j      (since ∥p Bj -q Bj ∥ 2 ≤ 2α implies o j = α) ≤ 2 √ k •      w j=1 ∥p B j -q B j ∥2≤2α α + w j=1 ∥p B j -q B j ∥2>2α 2∥p Bj -q Bj ∥ 2      (since ∥p Bj -q Bj ∥ 2 > 2α implies o j ≤ 2∥p Bj -q Bj ∥ 2 ) ≤ 2 √ k •      w j=1 ∥p B j -q B j ∥2≤2α α + 2 w j=1 ∥p B j -q B j ∥2>2α ∥p Bj -q Bj ∥ 1      (since ∥p Bj -q Bj ∥ 2 ≤ ∥p Bj -q Bj ∥ 1 ) ≤ 2 √ k •      ⌈d/k⌉ • α + 2 w j=1 ∥p B j -q B j ∥2>2α ∥p Bj -q Bj ∥ 1      (since |{j ∈ [w] : p Bj ∥ 2 ≤ 2α}| ≤ w) ≤ 2 √ k • (⌈d/k⌉ • α + 2∥p -q∥ 1 ) (since w j=1 ∥p B j -q B j ∥2>2α ∥p Bj -q Bj ∥ 1 ≤ w j=1 ∥p Bj -q Bj ∥ 1 = ∥p Bj -q Bj ∥ 1 ) That is, λ ≤ 2 √ k • (⌈d/k⌉ • α + 2∥p -q∥ 1 ).
The property follows by putting together both bounds. Now, suppose APPROXL1 tells us that ∥p -q∥ 1 ≤ r. We can then perform a constrained LASSO to search for a candidate p ∈ [0, 1] d using O( r 2 ε 4 log d δ ) samples from Ber(p). Lemma 3.4. Fix d ≥ 1, r ≥ 0, ε, δ > 0, and q ∈ [0, 1] d . Given O( r 2 ε 4 log d δ ) samples from Ber(p) for some unknown p ∈ [0, 1] d with ∥p -q∥ 1 ≤ r, one can produce an estimate p ∈ [0, 1] d in poly(n, d) time such that ∥ p -p∥ 2 ≤ ε with success probability at least 1 -δ.
Proof. Suppose we get n samples y 1 , . . . , y n ∼ Ber(p). For i ∈ [n], we can re-express each y i as y i = p + z i for some z i distributed as Ber(p) -p. Let us define p ∈ [0, 1] d as follows:
p = argmin ∥b-q∥1≤r 1 n n i=1 ∥y i -b∥ 2 2
(1) By optimality of p in Equation (1), we have
1 n n i=1 ∥y i -p∥ 2 2 ≤ 1 n n i=1 ∥y i -p∥ 2 2
(2) By expanding and rearranging Equation (2), one can show:
∥ p -p∥ 2 2 ≤ 2 n n i=1 z i , p -p(3)
Meanwhile, a standard Chernoff bound shows that Pr
∥ n i=1 z i ∥ ∞ ≥ 2n log 2d δ ≤ δ.
Therefore, using Hölder's inequality and triangle inequality with the above, we see that, with probability at least 1 -δ,
∥ p -p∥ 2 2 ≤ 2 n ⟨ n i=1 z i , p -p⟩ ≤ 2 n • n i=1 z i ∞ • ∥ p -p∥ 1 ≤ 2 n • n i=1 z i ∞ • (∥ p -q∥ 1 + ∥p -q∥ 1 ) ≤ 4r • 2 log 2d δ n
Finally, it is known that LASSO runs in poly(n, d) time.
Using Lemma 3.4, we now ready to prove Theorem 3.1.
Proof of Theorem 3.1. Correctness of p output. TESTANDOPTIMIZEMEAN (Algorithm 2) has two possible outputs for p: Case 1: p = argmin ∥b-q∥1≤λ 1 n n i=1 ∥y i -b∥ 2 2 , which can only happen when Outcome is λ ∈ R and λ < ε
√ d Case 2: p = 1 n n i=1 y i
Conditioned on APPROXL1 succeeding, with probability at least 1 -δ, we will show that d TV (Ber(p), Ber( p)) ≤ ε and failure probability at most δ in each of these cases, which implies the theorem statement.
Case 1: Using r = λ as the upper bound, Lemma 3.4 tells us that ∥ p -p∥ 2 ≤ ε τ (1 -τ )/2 with failure probability at most δ when O(λ 2 /τ 2 ε 4 ) i.i.d. samples are used. Using Proposition 2.4, d TV (Ber(p), Ber( p)) ≤ ε.
Case 2: With O(d/ε 2 ) samples, it is known that the empirical mean p achieves d TV (Ber(p), Ber( p)) ≤ ε with failure probability at most δ.
this section cite: []

Section: Algorithm 2
The TESTANDOPTIMIZEMEAN algorithm. 1: Input: Error rate ε > 0, failure rate δ ∈ (0, 1), parameter η ∈ [0, 1 4 ], parameter τ ∈ [0, 1 2 ], and sample access to Ber(p) 2: Output:
p ∈ R d 3: Define k = min(⌈d 4η /τ 4 ⌉, d), α = εd (3η-1)/2 /τ , ζ = 4ε • √ d, and δ ′ = δ ⌈d/k⌉•⌈log 2 ζ/α⌉ 4: Draw O( √ k log(1/δ ′ )/α 2 ) i.i.d.
samples from Ber(p) and store it into a set S 5: Let Outcome be the output of the APPROXL1 algorithm given k, α, ζ, and S as inputs 6: if Outcome is λ ∈ R and λ < ε
√ d then 7: Draw n ∈ O(λ 2 /ε 4 ) i.i.d. samples y 1 , . . . , y n ∈ {0, 1} d 8: return p = argmin ∥b-q∥1≤λ 1 n n i=1 ∥y i -b∥ 2 2 9: else 10: Draw n ∈ O(d/ε 2 ) i.i.d. samples y 1 , . . . , y n ∈ {0, 1} d 11: return p = 1 n n i=1 y i {Empirical mean} 12: end if Sample complexity used. APPROXL1 uses |S| = m(k, α, δ ′ ) ∈ O( √ k/α 2
) samples to produce Outcome. Then, APPROXL1 further uses O(λ 2 /τ 2 ε 4 ) samples or O(d/ε 2 ) samples depending on whether λ < ε √ d. So, TESTANDOPTIMIZEMEAN has a total sample complexity of
O √ k α 2 + min λ 2 τ 2 ε 4 , d ε 2 . Meanwhile, Lemma 3.3 states that ∥p -q∥ 1 ≤ λ ≤ 2 √ k • (⌈d/k⌉ • α + 2∥p -q∥ 1 ) whenever Outcome is λ ∈ R. Since (a + b) 2 ≤ 2a 2 + 2b 2
for any two real numbers a, b ∈ R, we see that
λ 2 τ 2 ε 4 ∈ O k τ 2 ε 4 • d 2 α 2 k 2 + ∥p -q∥ 2 1 ⊆ O d ε 2 • 1 τ 2 dα 2 ε 2 k + k•∥p-q∥ 2 1 dε 2
. Putting together the above observations, we see that the total sample complexity is
O √ k α 2 + d ε 2 • min 1, dα 2 ε 2 τ 2 k + k • ∥p -q∥ 2 1 dτ 2 ε 2 .
Recalling that TESTANDOPTIMIZEMEAN sets k = min(⌈d 4η τ -4 ⌉, d) and α = εd (3η-1)/2 τ -1 , the above expression simplifies to
O d ε 2 • d -η + min 1, ∥p-p∥ 2 1 d 1-4η τ 6 ε 2 ..
this section cite: []

Section: Lower Bounds
For proving of our lower bounds, we use the following corollary of Fano's inequality. Lemma 4.1 (Lemma 6.1 of [ABDH + 20]). Let κ : R → R be a function and let F be a class of distributions such that, for all ε > 0, there exist distributions f 1 , . . . , f M ∈ F such that
d KL (f i , f j ) ≤ κ(ε) and d TV (f i , f j ) > 2ε ∀i ̸ = j ∈ [M ]
Then any method that learns F to within total variation distance ε with probability ≥ 2/3 has sample complexity Ω log M κ(ε) log(1/ε) . Lemma 4.2 (Learning unbalanced distributions requires linear samples). Suppose ε is sufficiently small, and we are given sample access to a product distribution Ber(p) on {0, 1} d with mean vector p having entries which are O(1/d), along with an advice mean vector q such that ∥p -q∥ 1 ≤ O(ε). Even in this case, learning p such that d TV (Ber(p), Ber( p)) ≤ ε requires Ω d ε samples
Proof. Suppose that the advice distribution Ber(q
) has mean vector q ≜ ε d • • • ε d . If S ⊆ [d], define p S ∈ [0, 1] d with p S [i] = 2ε
d if i ∈ S and = ε d otherwise. Then, we have
∥p S -q∥ 1 = |S| ε d for all S ⊆ [d]. Also, for all S, T ⊆ [d] we have d TV (Ber(p S ), Ber(p T )) ≥ Pr x∼Ber(p S ) (x S\T = 0) -Pr x∼Ber(p T ) (x S\T = 0) = 1 -2ε d |S\T | -1 -ε d |S\T | ≥ 1 - |S\T |ε d + 1 1+2 |S\T |ε d
; this is using the inequalities (1 -x) r ≥ 1 -rx for r ∈ {0} ∪ [1, ∞),
x ≤ 1, and (1 -x) r ≤ 1 1+rx for r ≥ 0, x ∈ (-1/r, 1]. Using the same argument with the set T \ S, we get d TV (Ber(p S ), Ber(p T )) ≥ 1 -
|T \S|ε d + 1 1+2 |T \S|ε d . Thus, we have d TV (Ber(p S ), Ber(p T )) ≥ max H∈{S\T,T \S} ξ≜|H|ε/d 1 -ξ + 1 1+2ξ . Note that, by calculation, we can show that 1 -ξ + 1 1+2ξ ≥ ξ/2 -ξ 2 for ξ ≥ 0, which is Ω(ξ) for ξ ∈ (0, 1/4). Similarly, we have d KL (Ber(p S )∥Ber(p T )) = i∈[d] kl([p S ] i , [p T ] i ) = i∈S\T kl( 2ε d , ε d ) + i∈T \S kl( ε d , 2ε d ) (where kl(p, q) ≜ d KL (Ber(p)∥Ber(q))). We can see by simple calculations along with the logarithmic inequality ln(1+x) ≤ x for x > -1, that kl( ε d , 2ε d ) ≤ ε d 1 -ln(2) + ε d-2ε ≤ 0.5ε d (for d ≥ 10 and ε ≤ 1), and kl
2ε d , ε d ≤ ε d 2 ln(2) -1 + ε d-ε ≤ 0.5 d (for d ≥ 10 and ε ≤ 1). Thus d KL (Ber(p S )∥Ber(p T )) ≤ ε 2d (|S \ T | + |T \ S|) = |S⊕T |ε 2d . Viewing sets S ⊆ [d] as vectors in F d
2 and using the Gilbert-Varshamov bound, we can say that, for any constant c ∈ (0, 1) and sufficiently large d, there exists a family of sets {S 1 , . . . , S M } ⊆ 2 [d]  with M ≥ 2 Ω(cd) such that |S i | = cd and |S i ⊕ S j | ≥ cd 4 for all i, j ∈ [M ]. We use this family to instantiate distributions f i ≜ Ber(p S i ) for each i ∈ ≤ cε for all i, j ∈ [M ]. Since |S i ⊕ S j | ≥ cd/4, we will have at least one of H ∈ {S \ T, T \ S} with |H| ∈ cd 8 , cd . Thus, we will have |H|ε d ∈ cε 8 , cε = Θ(ε) (for constant c > 0), and d TV (f i , f j ) ≥ Ω(ε) (supposing cε < 1/4).
By appropriately scaling ε and applying Lemma 4.1, we can show that we need Ω d ε samples to learn a product distribution f * = Ber(p S * ) ∈ {f 1 , . . . , f M } to within ε in TV distance, even when given advice q with ∥p S * -q∥ 1 < ε, if the distribution mean vector p S * are allowed to be unbalanced (specifically, with entries ≤ O(1/d)).
We also prove a sample complexity lower bound for learning product distributions balanced case given advice, which adapts the sample complexity lower bound in ([BCGG25], Lemma 32) for learning multivariate isotropic gaussians N (µ * , I d ) given an advice vector which is close to the true mean vector in ℓ 1 distance. Lemma 4.3. Let ε > 0 be sufficiently small. Suppose that we are given sample access to a distribution Ber(p) where p is 1 4 -balanced, and also an advice vector q with ∥p -q∥ 1 = λ ≥ 100ε. Then, any algorithm that learns Ber(p) up to distance ε in total variation with constant failure probability requires Ω min ∥p -q∥ 2 1 /ε 4 , d/ε 2 samples. In particular, when ∥p -q∥ 1 ≥ ε √ d, we need Ω(d/ε 2 ) samples.
Proof. Suppose we want ∥p -q∥ 1 = λ for λ sufficiently small. Fix q = 1 2 • • • 1 2 and suppose p = p S for some S ⊆ As in ([BCGG25], Lemma 32), we consider {S 1 , . . . , S M } and take p i ≜ p Si for a family of ksubsets with M ≥ 2 Ω(k) and |S i ⊕ S j | ≥ k/4 for all i ̸ = j, known to exist via the Gilbert-Varshamov bound. We can do this as long as, e.g. k ≥ 10. This gives d KL (Ber(p i )∥Ber(p j )) ≤ 16λ 2 k (where λ is a function of ε) and d TV (Ber(p i ), Ber(p j )) ≥ c λ 2 √ k as long as λ 2 √ k < 1.
If we choose k = ⌈ λ 2 ε 2 ⌉ ≥ 100 such that λ = ε √ k < 2 √ k, we will get pairwise TV ≥ cε/2 and pairwise KL ≤ 16ε 2 k k ≤ O(ε 2 ). Finally, scaling ε before applying Lemma 4.1 gives the result.
this section cite: []

Section: Conclusion
This work introduces an efficient algorithm for learning product distributions on the Boolean hypercube when provided with an imperfect advice distribution. The sample complexity of this algorithm is O(d 1-η /ε 2 ) under specific conditions on the advice quality and a "balancedness" assumption on the true distribution. Note that the algorithm's sample complexity becomes sublinear in the dimension d if the advice is sufficiently accurate, and it remains robust even with poor advice. Key to this is a novel tolerant mean tester and techniques for approximating the ℓ 1 -distance between the true and advice distributions. Future research could extend this learning-with-advice framework to other complex models like Bayesian networks and Ising models, aiming to understand how structural properties of these models interact with advice quality. It would also be interesting to investigate if advice can improve the sample complexity of learning an unstructured distribution over a discrete domain [n] compared to the classical upper bound of O( n ε 2 ) samples.
this section cite: []

Section: References
Ref_id:b0 Title: Near-optimal sample complexity bounds for robust learning of gaussian mixtures via compression schemes Year: (2020-10)
Ref_id:b1 Title: Learning-augmented mechanism design: Leveraging predictions for facility location Year: (2022)
Ref_id:b2 Title: Online Computation with Untrusted Advice Year: (2020)
Ref_id:b3 Title: Secretary and online matching problems with machine learned advice Year: (2020)
Ref_id:b4 Title: A Novel Prediction Setup for Online Speed-Scaling Year: (2022)
Ref_id:b5 Title: Learning multivariate gaussians with imperfect advice Year: ()
Ref_id:b6 Title: A Universal Error Measure for Input Predictions Applied to Online Graph Problems Year: (2022)
Ref_id:b7 Title: Learning Augmented Energy Minimization via Speed Scaling Year: (2020)
Ref_id:b8 Title: The Primal-Dual method for Learning Augmented Algorithms Year: (2020)
Ref_id:b9 Title: A survey on distribution testing: Your data is big. but is it blue? Theory of Computing Year: (2020)
Ref_id:b10 Title: Testing bayesian networks Year: (2017)
Ref_id:b11 Title: Active causal structure learning with advice Year: (2023)
Ref_id:b12 Title: Online bipartite matching with imperfect advice Year: (2024)
Ref_id:b13 Title: Learning-augmented online bipartite fractional matching Year: ()
Ref_id:b14 Title: Faster fundamental graph algorithms via learned predictions Year: (2022)
Ref_id:b15 Title: Learning structured distributions. Handbook of Big Data Year: (2016)
Ref_id:b16 Title: Faster matchings via learned duals Year: (2021)
Ref_id:b17 Title: A new approach for testing properties of discrete distributions Year: (2016)
Ref_id:b18 Title: Combinatorial methods in density estimation Year: (2001)
Ref_id:b19 Title: Secretaries with Advice Year: (2021)
Ref_id:b20 Title: Square hellinger subadditivity for bayesian networks and its applications to identity testing Year: (2017)
Ref_id:b21 Title:  Year: (2013)
Ref_id:b22 Title: Improved price of anarchy via predictions Year: (2022)
Ref_id:b23 Title: Learning-Augmented Algorithms for Online TSP on the Line Year: (2023)
Ref_id:b24 Title: Online Algorithms for Rent-or-Buy with Expert Advice Year: (2019)
Ref_id:b25 Title: The Case for Learned Index Structures Year: (2018)
Ref_id:b26 Title: On the tensorization of the variational distance Year: (2025)
Ref_id:b27 Title: Online Scheduling via Learned Weights Year: (2020)
Ref_id:b28 Title: A Model for Learned Bloom Filters, and Optimizing by Sandwiching Year: (2018)
Ref_id:b29 Title: Improving Online Algorithms via ML Predictions Year: (2018)
Ref_id:b30 Title: Online Algorithms for Multi-shop Ski Rental with Machine Learned Advice Year: (2020)
Ref_id:b31 Title: For initial submissions, do not include any information that would break anonymity (if applicable) Year: ()
Ref_id:b32 Title: usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: LLMs were not used in the development of this research. Guidelines: • The answer NA means that the core method development Year: ()
Ref_id:b33 Title: ) for what should or should not be described Year: ()
