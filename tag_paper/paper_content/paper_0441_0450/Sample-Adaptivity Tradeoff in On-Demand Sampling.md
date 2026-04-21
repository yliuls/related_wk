Title: Sample-Adaptivity Tradeoff in On-Demand Sampling
Abstract: We study the tradeoff between sample complexity and round complexity in ondemand sampling, where the learning algorithm adaptively samples from k distributions over a limited number of rounds. In the realizable setting of Multi-Distribution Learning (MDL), we show that the optimal sample complexity of an r-round algorithm scales approximately as dk Θ(1/r) /ε. For the general agnostic case, we present an algorithm that achieves near-optimal sample complexity of O((d + k)/ε 2 ) within O( √ k) rounds. Of independent interest, we introduce a new framework, Optimization via On-Demand Sampling (OODS), which abstracts the sample-adaptivity tradeoff and captures most existing MDL algorithms. We establish nearly tight bounds on the round complexity in the OODS setting. The upper bounds directly yield the O( √ k)-round algorithm for agnostic MDL, while the lower bounds imply that achieving sub-polynomial round complexity would require fundamentally new techniques that bypass the inherent hardness of OODS.

Section: Introduction
Modern machine learning pipelines increasingly treat the training set as a mutable resource-adapting the data collection process in response to intermediate learning signals in order to focus effort where it matters most. This adaptivity arises in a range of settings. In multi-distribution learning (see e.g. [BHPQ17,HJZ22]), the on-demand sampling framework allows algorithms to adaptively select domains from which to sample to minimize the worst-case loss. Similarly in multi-armed bandit problems, adaptively selecting which arm to pull next is an important aspect of algorithm design. In practice, pipelines for training models adaptively decide how to reweigh or augment their datasets to improve downstream accuracy [XPD + 23, SRC24]. While these paradigms demonstrate-both theoretically and empirically-that adaptive data collection can significantly improve performance and sample efficiency, adaptivity is often an undesirable feature. It requires the ML practitioner to collect data sequentially, slowing down the end-to-end training process and limiting opportunities for parallelism and scalability. This tension raises a central question:
to what extent is adaptive sample collection necessary to achieve the observed gains in learning performance? And what are the quantitative tradeoffs between the number of adaptive rounds and sample complexity?
We study this question in the context of on-demand sampling within the framework of multidistribution learning [HJZ22]. Multi-distribution learning extends the classical agnostic learning setting by giving the learner sampling access to k distributions D 1 , . . . , D k , with the goal of learning a single predictor that minimizes the worst-case error across all distributions. This framework has emerged as a central model for studying algorithmic dataset selection, offering both a method of allocating a fixed sampling budget across heterogeneous data sources and a unifying perspective on several recent advances in federated learning, multi-task learning, domain adaptation, and fair and robust machine learning [KNRW18, MSS19, SKHL20, RY21, TH22, HJZ23, ZZC + 24].
Prior work on multi-distribution learning has established optimal on-demand sample complexities of O((d + k)/ε) in the realizable case [BHPQ17,CZZ18,NZ18] and O((log(|H|) + k)/ε 2 ) in the agnostic case [HJZ22], where d is the VC dimension of hypothesis class H. The latter was recently extended to O((d + k)/ε 2 ) for infinite hypothesis classes [ZZC + 24, Pen24]. However, these algorithms rely on a large number of adaptive rounds-often polynomial in 1/ε and the complexity of H and with a mild sublinear dependence on k (see Table 1 for details). In some cases, these algorithms collect a single sample per round, resulting in a number of rounds that is as large as the sample complexity itself! In contrast, the best known fully non-adaptive algorithms incur significantly higher sample complexities of O(dk/ε) and O(dk/ε 2 ) in the realizable and agnostic settings, respectively.
Despite this gap, the complexity landscape between the two extremes-full adaptivity and full nonadaptivity-remains largely unexplored. In particular, it is unknown whether a constant number of adaptive rounds, or even one that is merely independent of the accuracy level ε, could suffice to recover the optimal sample complexities achieved in the fully adaptive setting.
this section cite: ['b2', 'b15', 'b15', 'b2', 'b8', 'b25', 'b15']

Section: Our Contributions and Results
In this work, we formalize the problem of studying the tradeoffs between adaptivity and sample complexity of on-demand sampling algorithms. Specifically, we aim for achieving the optimal sample complexity with a number of adaptive rounds that is nearly independent of ε and d and with only sublinear dependence on k. We refer to the number of adaptive rounds as the round complexity of an algorithm.
In the realizable case, we provide a tight characterization of the sample-adaptivity tradeoff. In particular, we prove that a round complexity r allows for a sample complexity of dk Θ(1/r) /ε, yielding a smooth tradeoff between round complexity and sample complexity. In addition to confirming that log k rounds are necessary for achieving the optimal sample complexity in the realizable setting, this also indicates that a small constant number of rounds-say, 3 rounds (!)-are sufficient to achieve an O(d √ k/ε) sample complexity, which is a significant improvement over fully non-adaptive approaches. In the agnostic case, we show that O( √ k) rounds of adaptivity is sufficient to achieve the optimal sample complexity of O((d + k)/ε 2 ).
From a technical perspective, we establish the tradeoff between adaptivity and sample complexity through two approaches. In the realizable case, our algorithms are based on a novel application of a variant of the AdaBoost algorithm with a particular notion of margin. In the agnostic setting, we introduce a general and abstract optimization problem called Optimization via On-Demand Sampling (OODS). In this framework, the goal is to optimize a concave function f over [0, 1] k , representing weights over k distributions. There is no notion of sample complexity in this setting; instead, the algorithm can only access value and gradient information about f within a restricted trust region, which the algorithm can expand in every round. At a high level, the extent of the trust region serves as a proxy for sample complexity: the more a distribution is sampled, the better we can estimate the performance of predictors on it. The number of times the trust region is expanded before finding the optimum of f corresponds to the round complexity. We establish both upper and lower bounds on the round complexity in the OODS setting.
A strength of the OODS framework is that algorithms developed for OODS naturally transfer to the agnostic multi-distribution learning problem, forming the foundation for our performance guarantees. Additionally, the optimization formulation gives rise to more natural algorithm-independent lower bounds on adaptivity. In particular, we prove poly(k) lower bounds on the round complexity of the OODS problem. These lower bounds shed light on the challenges of achieving the optimal sample complexity in agnostic multi-distribution learning using a sub-polynomial number of rounds.
this section cite: []

Section: Related Work
Multi-Distribution Learning Blum, Haghtalab, Procaccia and Qiao [BHPQ17] introduced the realizable setting of multi-distribution learning, for which several O(log k)-round algorithms with near-optimal sample complexity of O((d + k)/ε) were given [BHPQ17,CZZ18,NZ18]. On the other Setting Sample Complexity Round Complexity Reference Realizable O((d + k)/ε) O(log k) [BHPQ17, CZZ18, NZ18] Agnostic O((log(|H|) + k)/ε 2 ) O((log(|H|) + k)/ε 2 ) [HJZ22] Agnostic O(d/ε 4 + k/ε 2 ) O(log(k)/ε 2 ) [AHZ23] Agnostic O((d + k)/ε 2 ) • (log k) O(log(1/ε)) (log k) O(log(1/ε)) [Pen24] Agnostic O((d + k)/ε 2 ) O(log(k)/ε 2 ) [ZZC + 24] Realizable O((k 2/r • d + k)/ε) r Theorem 1
Realizable Ω(k 1/r • d/r) r Theorem 2 Agnostic O((d + k)/ε 2 ) min{ O( √ k), O(k log k)} Propositions 2 and 3
Table 1: An overview of sample-adaptivity tradeoff in multi-distribution learning. k is the number of distributions. H is the hypothesis class and d is its VC dimension. r denotes a tunable round complexity between 1 and O(log k). O(•) and Ω(•) suppress polylog(d, k, 1/ε, 1/δ) factors.
hand, it is folklore that the sample complexity is Ω(dk/ε) without adaptive sampling. For the more challenging agnostic setting where a "perfect" predictor may not exist, the optimal sample complexity was shown to be O((d + k)/ε 2 ) in a series of recent work [HJZ22, AHZ23, ZZC + 24, Pen24]. Interestingly, all these algorithms have a round complexity of at least poly(1/ε) (see Table 1 for details). Other variants of the problem, where some data sources might be adversarial or differently labeled, have also been studied [Qia18,DQ24].
Power of Adaptivity in Learning and Beyond Agarwal, Agarwal, Assadi and Khanna [AAAK17] systematically formulated the tradeoff between adaptivity and sample complexity in several learning problems, including a batched setting of multi-armed bandits that was previously introduced by Jun, Jamieson, Nowak and Zhu [JJNZ16] and subsequently studied in [GHRZ19, JYT + 24, JZZ25]. Chen, Papadimitriou and Peng [CPP22] proposed a PAC learning framework of continual learning, and quantified the tradeoff between the number of sequential passes and the memory usage of the learning algorithm. Another recent line of work focused on the adaptivity-query tradeoff in submodular optimization [BS18, FMZ19, EN19, CQ19b, BRS19, CQ19a, ENV19, LLV20].
this section cite: ['b2', 'b2', 'b8', 'b25', 'b27', 'b9', 'b0', 'b17', 'b5']

Section: Preliminaries
Multi-Distribution Learning (MDL) We follow the formulation of MDL in [HJZ22]. Let X be the instance space and Y = {0, 1} be the binary label space. Let H ⊆ Y X be a hypothesis class and d be its VC dimension. There are k unknown data distributions D 1 , D 2 , . . . , D k over X × Y. In each round, the algorithm draws samples from the k distributions. The number of samples may differ on the k distributions, and may be chosen adaptively based on samples drawn in previous rounds. The sample complexity is the total number of samples drawn from all distributions in all rounds. An r-round algorithm draws r rounds of samples and has a round complexity of r.
The goal is to learn a predictor ĥ : X → Y that performs well on all k distributions D 1 , . . . , D k . Formally, letting err( ĥ, D) := Pr (x,y)∼D ĥ(x) ̸ = y denote the population error of predictor ĥ on distribution D, an MDL algorithm is (ε, δ)-PAC (Probably Approximately Correct) if max i∈ [k] err( ĥ, D i ) ≤ OPT + ε where OPT := min h∈H max i∈ [k] err(h, D i )
holds with probability at least 1 -δ over the randomness in both the algorithm and the samples.
In the realizable setting, the data distributions are promised to satisfy OPT = 0, i.e., there exists a perfect predictor h ⋆ ∈ H such that err(h ⋆ , D i ) = 0 for every i ∈ [k]. We also refer to the general MDL setting-where OPT can be non-zero-as the agnostic setting.
this section cite: ['b15']

Section: MDL via Game Dynamics
Most previous agnostic MDL algorithms (e.g., [HJZ22, ZZC + 24, Pen24]) view the learning problem as a zero-sum game, in which the "min player" chooses a hypothesis (or a mixture of multiple hypotheses) and the "max player" chooses a mixture of the k data distributions. These algorithms solve MDL by simulating the game dynamics when the two players follow certain strategies, e.g., best response or a no-regret online learning algorithm. The analysis then boils down to finding the sample size that suffices for simulating the game dynamics accurately. For instance, simulating a "min player" that best-responds to the "max player" is equivalent to finding a hypothesis that approximately minimizes the error on a given mixture of the k distributions.
this section cite: []

Section: Sample-Adaptivity Tradeoff for Realizable MDL

this section cite: []

Section: Overview of Upper Bound
For the realizable setting of MDL, we present an algorithm that establishes a tradeoff between sample complexity and round complexity.
this section cite: []

Section: Theorem 1 (Informal version of Theorem 10).
Algorithm 1 is an r-round (ε, δ)-PAC algorithm for realizable MDL with sample complexity O(k 2/r log k
• d ε + k log(k) log(k/δ) ε ).
We highlight two special cases of the general tradeoff above. First, when r = log k, Algorithm 1 is most sample-efficient and recovers the near-optimal sample complexity bound of O((d + k)/ε) for realizable MDL [BHPQ17,CZZ18,NZ18]. Second, with a small constant number of adaptive rounds (e.g., r = 4), Algorithm 1 has a sample complexity of O((d
√ k + k)/ε).
Thus, our result demonstrates that even in the limited-adaptivity regime of r = O(1), we can improve on the Ω(dk/ε) sample complexity required in the fully non-adaptive case of r = 1. Remark 1. Using more sophisticated variants of boosting such as boost-by-majority [SF12, Chapter 13] or recursive boosting [Sch90], it is possible to achieve a sample complexity of O((d √ k+k)/ε) using exactly 3 rounds. Formal claims are deferred to Appendix A.
this section cite: ['b2', 'b8', 'b25', 'b29']

Section: Additional Notations
To state our algorithm succinctly, we introduce a few more notations. For a distribution D, we write D ⊗m as its m-fold product distribution, and S ∼ D ⊗m is a shorthand for drawing a size-m sample S from D. At each iteration t of the algorithm, we maintain weights q t (1), q t (2), . . . , q t (k) ≥ 0 that sum up to 1. We also abuse the notation and write q t as the mixture distribution k i=1 q t (i)•D i . While distributions D 1 , D 2 , . . . , D k are unknown, the algorithm may still sample from the mixture q t easily: it suffices to first draw a random index i such that Pr [i = j] = q t (j) for each j ∈ [k] and then sample from D i .
this section cite: []

Section: Algorithm 1: Trade-off Multi-Distribution Learning
Input: Sample access to k unknown distributions D 1 , . . . , D k , an optimal PAC learner A for class H, number of rounds r, target error ε, and failure probability δ.
1 Set margin θ = r 2 log(k) , p = 1 2 • 4k 2 /r -1 /(1-θ) , τ = ε 1+1/θ , and α = 1 2 ln 1-p p . 2 Set ε A = τ p 4 , δ A = δ 2r , and m = O d+log(1/δ A ) ε A .
3 Initialize q 1 (j) = 1 k for each j ∈ {1, . . . , k}, and let q 1 = k j=1 q 1 (j)D j . 4 for t = 1, 2, . . . , r do 5 Call learner A on a sample S t ∼ q ⊗m t , and let h t be the returned predictor.
this section cite: []

Section: 6
for j = 1, 2, . . . , k do 7 Draw a sample S j,t ∼ D ⊗n j , where n = 12 τ log(2rk/δ).
this section cite: []

Section: 8
Update:
q t+1 (j) = q t (j) Z t × e -α if err(h t , S j,t ) ≤ τ 2 e α if err(h t , S j,t ) > τ 2
where Z t is a normalization constant that ensures
k j=1 q t+1 (j) = 1. Output: Majority-Vote Predictor F : x → 1 1 r r t=1 h t (x) ≥ 1/2 .
this section cite: []

Section: Technical Overview
We explain here the main ideas behind Algorithm 1; the full proof and analysis is deferred to Appendix A. We run a variant of the classical AdaBoost algorithm [SF12] to maximize a particular notion of "margin" that is defined on distributions D 1 , . . . , D k . Specifically, in each round 1 ≤ t ≤ r, Algorithm 1 calls learner A to learn a predictor h t that has a low error on q t :
k j=1 q t (j) • err(h t , D j ) = err(h t , q t ) ≤ τ p.
Then, by Markov's inequality, h t also minimizes the fraction of distributions (as weighted by q t ) that have error more than τ :
k j=1 q t (j)1 [err(h t , D j ) > τ ] ≤ p.
Subsequently, Algorithm 1 updates the weighted mixture q t over the k distributions based on the thresholded loss function 1[err(h t , D j ) > τ ] (as is done in AdaBoost). After r rounds, the marginmaximization property of AdaBoost guarantees that
1 k k j=1 1 1 r r t=1 1[err(h t , D j ) > τ ] > 1 2 - θ 2 ≤ r t=1 2 (1 -p) 1+θ p 1-θ .
By choosing the margin parameter θ and p such that r t=1 2 (1 -p) 1+θ p 1-θ < 1/k, we are guaranteed that, on each of the k distributions, at least 1/2 + θ/2 fraction of the r predictors have error at most τ . Formally, it holds for every j ∈ [k] that
1 r r t=1 1[err(h t , D j ) > τ ] ≤ 1 2 - θ 2 .
Finally, with this margin property, invoking Lemma 13 implies that the majority-vote predictor will have error at most (1 + 1/θ)τ = ε on all k distributions.
this section cite: ['b30']

Section: Overview of Lower Bound
The following theorem complements Theorem 1 by showing that a k Ω(1/r) overhead is unavoidable.
this section cite: []

Section: Theorem 2 (Informal version of Theorem 16).
For every r = O(log k) and sufficiently large d, every r-round algorithm for realizable MDL has an Ω dk 1/r r log 2 k sample complexity.
This sample complexity lower bound nearly matches the dk 2/r log k term in Theorem 1, up to a poly(r, log k) factor and a factor of 2 in the exponent of k Θ(1/r) .
We briefly sketch the proof of the r = 2 case, i.e., an Ω(d √ k) lower bound against two-round algorithms. We consider the class of linear functions over X = F d 2 , which has a VC dimension of d. The ground truth classifier h ⋆ is drawn uniformly at random from all the 2 d linear functions. To construct the k data distributions, we choose k difficulty levels diff 1 , diff 2 , . . . , diff k as a uniformly random permutation of: (1) 1 copy of Θ(d); (2)
√ k copies of Θ(d/ √ k); (3) k - √ k -1 copies of Θ(d/k). Note that k i=1 diff i ≤ d. Each data distribution D i is the uniform distribution over a randomly chosen diff i -dimensional subspace V i ⊆ F d 2 . Furthermore, the subspaces V 1 , V 2 , . . . , V k are chosen such that they are linearly independent, i.e., dim(Span(V 1 ∪ V 2 ∪ • • • ∪ V k )) = k i=1 diff i .
Intuitively, diff i measures the "effective sample complexity" for learning D i : Θ(diff i ) samples are sufficient and necessary to learn an accurate classifier for D i . In addition, since h ⋆ is randomly chosen and the subspaces V 1 , V 2 , . . . , V k are independent, samples collected from one distribution D i provide no information about the value of h ⋆ on V j (except for the zero vector) for every j ̸ = i. Furthermore, if diff i ∈ {Θ(d/ √ k), Θ(d)} and m ≪ d/ √ k samples have been drawn from D i , it holds with high probability that the m vectors in these samples are linearly independent. Then, the learner gains no information for distinguishing whether
diff i = Θ(d/ √ k) or diff i = Θ(d).
A three-round learner has a simple strategy: (1) In Round 1, draw Θ(d/k) samples from each distribution, thereby identifying the distributions with diff i = Θ(d/k) as well as learning the value
of h ⋆ on each V i ; (2) In Round 2, draw Θ(d/ √ k)
samples from each of the √ k + 1 remaining distributions, which is sufficient for all distributions except the one with diff i = Θ(d); (3) In Round 3, learn the only remaining distribution using Θ(d) samples. The resulting sample complexity is O(d).
In contrast, a two-round learner must "skip" one of the three steps. For example, in Round 2 where there are still √ k + 1 "suspects" among which one distribution has difficulty level Θ(d), the learner could draw Θ(d) samples from each of them. Alternatively, the learner could draw Θ(d/ √ k) samples from each distribution in Round 1, so that the distribution with diff i = Θ(d) can be identified and then learned in Round 2. However, both strategies would have an Ω(d √ k) sample complexity.
The formal proof (in Appendix B) extends the hard instance construction to all r = O(log k) by using r + 1 different difficulty levels separated by a k 1/r factor. We then formalize the intuition that every r-round MDL algorithm must "skip" a step and thus incur a k 1/r overhead in the sample complexity.
this section cite: []

Section: Sample-Adaptivity Tradeoff for Agnostic MDL
For the agnostic setting, we show that the near-optimal sample complexity of O((d + k)/ε 2 ) can be achieved by a poly(k)-round algorithm.
this section cite: []

Section: Proposition 1 (Corollaries 21 and 22).
There is a min{ O( √ k), O(k log k)}-round MDL algorithm with sample complexity O((d + k)/ε 2 ).
The MDL Algorithm of [ZZC + 24] Our starting point is the approach of [ZZC + 24, Algorithm 1], which we briefly describe below. For brevity, we use O(•) and Θ(•) to suppress polylog(k, d, 1/ε, 1/δ) factors, and let err(h, S) := 1
|S| (x,y)∈S 1 [h(x) ̸ = y] denote the empir- ical error of hypothesis h : X → Y on dataset S ⊆ X × Y.
The algorithm maintains k datasets S 1 , S 2 , . . . , S k , where each S i contains training examples drawn from D i . The algorithm runs the Hedge algorithm for T = Θ((log k)/ε 2 ) iterations starting at w (1) = (1/k, 1/k, . . . , 1/k). Each iteration t ∈ [T ] consists of the following two steps:
• ERM step: For each i ∈ [k], draw additional samples from D i and add them to
S i until |S i | ≥ w (t) i • Θ((d + k)/ε 2 ).
Then, find a hypothesis h (t) ∈ H that minimizes the empirical error L(h) := k i=1 w
(t) i • err(h, S i ), which is an estimate of the error of h on k i=1 w (t) t) and r (t) via a Hedge update.
i D i . • Hedge update step: For each i ∈ [k], draw w (t) i • Θ(k) fresh samples from D i to obtain an estimate r (t) i ≈ err(h (t) , D i ). Compute w (t+1) from w (
The crux of the analysis of [ZZC + 24] is to show that the dataset sizes in the two steps above are sufficiently large, so that h (t) approximately minimizes the error on mixture k i=1 w (t) i D i , and the reward vector r (t) is accurate enough for the Hedge update.
A straightforward implementation of the algorithm needs T = Θ((log k)/ε 2 ) rounds of sampling. In comparison, the O(
√ k) round complexity in Proposition 1 is lower when ε ≪ 1/k 1/4 .
Hedge with Lazy Updates We prove Proposition 1 by modifying the algorithm of [ZZC + 24] so that it draws samples more lazily. The resulting algorithm is termed LazyHedge and formally defined in Algorithm 2. There are two versions of the algorithm-the "box" version and the "ellipsoid" version-that give the O(k log k) and O( √ k) round complexity bounds, respectively.
Similar to the Hedge algorithm, LazyHedge maintains a weight vector w (t) at each iteration t.
In addition, it maintains a cap vector w (t) i as a proxy for the size of dataset S i at time t. At the start of iteration t, it checks whether w (t) is "observable" under cap w (t-1) in the sense that w (t) ∈ O(w (t-1) ). If the condition holds, no additional samples are drawn and the cap vector is left unchanged. Otherwise, the cap w (t) is updated to C times the entrywise maximum of all weight vectors so far, and additional samples are drawn so that both |S i | and |S i,t | match w (t) i . Finally, LazyHedge computes the next weight vector w (t+1) from w (t) using the Hedge update rule.
Algorithm 2: LazyHedge: Hedge with Lazy Updates Input: Number of distributions k, number of iterations T = Θ((log k)/ε 2 ), step size η = Θ(ε), margin parameter C > 1. 1 Box version: Define O(w) := {w ∈ ∆ k-1 : w i ≤ w i , ∀i ∈ [k]}. 2 Ellipsoid version: Define O(w) := {w ∈ ∆ k-1 : k i=1 w 2 i /w i ≤ 1}. 3 Set w (1) = (1/k, 1/k, . . . , 1/k) and w (0) = (0, 0, . . . , 0). 4 Set S i = ∅ for i ∈ [k] and S i,t = ∅ for i ∈ [k] and t ∈ [T ]. 5 for t = 1, 2, . . . , T do 6 if w (t) ∈ O(w (t-1) ) then 7 Set w (t) = w (t-1) .
this section cite: []

Section: else 9
Set
w (t) i = C • max{w (1) i , w (2) i , . . . , w (t) i } for every i ∈ [k]. 10 Add samples from D i to S i until |S i | ≥ w (t) i • Θ((d + k)/ε 2 ) for every i ∈ [k]. 11 Add samples from D i to S i,t ′ until |S i,t ′ | ≥ w (t) i • Θ(k) for every i ∈ [k] and t ≤ t ′ ≤ T . 12 ERM step: Set ĥ(t) ∈ argmin h∈H k i=1 w (t) i • err(h, S i ). 13 Hedge update step: Set r (t) i = err( ĥ(t) , S i,t ). Compute w (t+1) ∈ ∆ k-1 such that w (t+1) i = w (t) i •e ηr (t) i k j=1 w (t) j •e ηr (t) j for every i ∈ [k].
Output: Randomized classifier uniformly distributed over { ĥ(1) , ĥ(2) , . . . , ĥ(T ) }.
The correctness and sample complexity of LazyHedge follow from the analysis of [ZZC + 24]. At a high level, either version of LazyHedge ensures that using S i in the ERM step and using S i,t in the Hedge update step lead to low-variance estimates, which allow the analysis of [ZZC + 24] to go through. We provide a more detailed analysis in Appendix C.4. It remains to upper bound the round complexity of LazyHedge, namely, the number of times the cap vector is updated in Line 9. For the box version, we have an O(k log k) upper bound. Proposition 2. The box version of LazyHedge takes at most O(k log k) rounds. Proof sketch. If the cap vector is updated in the t-th iteration, there exists i ∈ [k] such that w (t) i > w (t-1) i
. We call such index i the culprit of this cap update. For index i to be the culprit, the historical high of w (t) i must have increased by a factor of C since the last cap update. As this historical high is non-decreasing and in [1/k, 1], each index i can be the culprit at most O(log C k) times. Thus, the round complexity is at most k • O(log C k) = O(k log k) for any constant C > 1.
For the ellipsoid version, a more involved analysis gives an O( √ k) round complexity bound.
Proposition 3. The ellipsoid version of LazyHedge takes at most O( √ k) rounds.
this section cite: []

Section: The analysis applies the following technical lemma shown by [ZZC + 24].
Lemma 3 (Lemma 3 of [ZZC + 24]). For some choice of T = Θ((log k)/ε 2 ) and η = Θ(ε) in LazyHedge, it holds with probability 1 -δ that
k i=1 max 1≤t≤T w (t) i ≤ O(log 8 (k/(εδ))) = O(1).
Proof sketch of Proposition 3. We classify the cap updates into two types: A "Type I" update is when some coordinate w i reaches a historical high of > 1/ √ k, and a "Type II" update is one without a significant increase in any coordinate. We show that either type of cap updates happen O( √ k) times.
The upper bound for Type I updates follows from Lemma 3, which implies that there are at most O(1) • √ k Type I updates where the coordinate reaches ≈ 1/ √ k, at most O(1) • √ k/2 Type I updates where the coordinate reaches ≈ 2/ √ k, and so on. These upper bounds sum up to O( √ k).
The analysis for Type II updates is more involved. Roughly speaking, we say that a coordinate i ∈ [k] gains a potential of a 2 /b when the historical high of w i increases from b to a through the Hedge dynamics. The O( √ k) bound follows from two technical claims: (1) Each Type II update may happen only if a total potential of Ω(1) is accrued over all k coordinates; (2) The total potential that the k coordinates may contribute to Type II updates is at most O( √ k).
5 A General Framework: Optimization via On-Demand Sampling
this section cite: []

Section: Problem Setup
In Optimization via On-Demand Sampling (OODS), the goal is to maximize a concave function f over the probability simplex ∆ k-1 := {w ∈ R k : k i=1 w i = 1, w i ≥ 0 ∀i ∈ [k]} by "sampling" from the k coordinates. The algorithm does not have full access to f ; instead, it maintains a cap vector w ∈ [0, 1] k that specifies the observable region of the simplex. We focus on two concrete settings of the problem, where the observable region is either a box or an ellipsoid defined by w.
Definition 1 (Optimization via On-Demand Sampling). f : ∆ k-1 → [0, 1] is an unknown concave function. In each round t = 1, 2, . . . , r, the algorithm chooses cap w (t) ∈ [0, 1] k that is lower bounded by w (t-1) entry-wise (if t > 1). Then, the algorithm makes arbitrarily many queries to a first-order oracle of f -which returns the value and a supergradient-at any w ∈ O(w (t) ), where O(w) := {w ∈ ∆ k-1 : w i ≤ w i , ∀i ∈ [k]} in the box setting, and O(w) := {w ∈ ∆ k-1 :
k i=1 w 2 i /w i ≤ 1} in the ellipsoid setting. The goal is to find ŵ ∈ ∆ k-1 such that f ( ŵ) ≥ max w∈∆ k-1 f (w) -ε while minimizing the sample overhead k i=1 w (r)
i and the round complexity r.
To see how OODS connects to MDL and on-demand sampling in general, we view the k coordinates as distributions D 1 , D 2 , . . . , D k from which the algorithm may sample. Maximizing f (w) can then be viewed as optimizing the mixing weights in mixture k i=1 w i D i . The cap w i is a proxy for and proportional to the number of samples that have already been drawn from D i . In light of this analogy, the sample overhead
k i=1 w (r)
i is simply a proxy for the total number of samples that the algorithm draws, while the round complexity r is the number of rounds of on-demand sampling.
The observable region O(w) represents the mixing weights w ∈ ∆ k-1 on which f (w) can be accurately estimated using the current dataset specified by w. In the box setting, the algorithm is only allowed to query f (w) if w i ≤ w i holds for every i ∈ [k], which can be viewed as a sufficient condition for the algorithm to obtain an accurate estimate for mixture k i=1 w i D i using the Θ(w i ) samples collected from each D i . In the ellipsoid setting, we use the more refined condition k i=1 w 2 i /w i ≤ 1, where the summation is a proxy for the variance in estimating the mixture k i=1 w i D i using the datasets. More details on how the box and ellipsoid settings connect to MDL can be found in Appendix C.4.
this section cite: []

Section: Overview of Upper Bounds
For the OODS problem, we give a simple algorithm with a poly(k) round complexity. The algorithm is also termed LazyHedge, as it is almost identical to the agnostic MDL algorithm in Section 4. We formally define the algorithm (Algorithm 3) in Appendix C for completeness. To guarantee a sample overhead of O(s), the algorithm takes O(k/s) rounds in the box setting and O( k/s) rounds in the ellipsoid setting. Here, the O(•) notation hides polylog(k/ε) factors, where ε is the accuracy parameter in OODS.
this section cite: []

Section: Theorem 4 (Informal version of Theorems 8 and 20).
There is an OODS algorithm with sample overhead O(s) that takes O(k/s) rounds in box setting and O( k/s) rounds in ellipsoid setting.
Hedge with Lazy Updates We apply the same LazyHedge strategy as in Algorithm 2 for MDL. LazyHedge maintains a weight vector w (t) at each iteration t. At the start of iteration t, it checks whether w (t) is still in the observable region O(w (t-1) ) specified by the previous cap w (t-1) . If so, the cap is left unchanged; otherwise, the cap w (t) is set to C times the entrywise maximum of all weight vectors so far. Then, LazyHedge queries the first-order oracle to obtain a supergradient r (t) at w (t) , and computes the next weight vector w (t+1) using the Hedge update. While LazyHedge takes T = Θ((log k)/ε 2 ) iterations, its round complexity is the number of times the cap is updated, which can be much lower than T .
this section cite: []

Section: Analysis for the Box Setting
We sketch the analysis for the box setting, and defer the ellipsoid setting to Appendix C. The standard regret analysis of Hedge shows that LazyHedge finds an O(ε)approximate maximum. We prove the following lemma in Appendix C.1. Lemma 5. LazyHedge outputs ŵ ∈ ∆ k-1 such that f ( ŵ) ≥ max w∈∆ k-1 f (w) -O(ε).
Next, we show that the sample overhead of LazyHedge is low. Recall that C > 1 is the margin parameter used in LazyHedge for the cap updates.
Lemma 6. LazyHedge has an O(C log 8 (k/ε)) sample overhead.
Proof. LazyHedge guarantees that w
(T ) i ≤ C • max 1≤t≤T w (t) i for every i ∈ [k]. By Lemma 3, the sample overhead is k i=1 w (T ) i ≤ C • k i=1 max 1≤t≤T w (t) i = O(C • log 8 (k/ε)).
It remains to upper bound the round complexity of LazyHedge in the box setting. The proof of the following lemma resembles and extends that of Proposition 2 in the MDL setting.
Lemma 7. LazyHedge takes min k, O((k/C) • log 8 (k/ε) } • O(log C k) rounds in the box setting. Proof sketch. If the cap is updated in the t-th iteration of LazyHedge, there exists i ∈ [k] such that w (t) i > w (t-1) i
. We call such index i the culprit of this cap update. By the same argument as in Proposition 2, each index i can be the culprit of at most O(log C k) cap updates. It remains to bound the number of indices that become the culprit of at least one cap update. This number is trivially at most k. Furthermore, since LazyHedge sets w (1) = (C/k, C/k, . . . , C/k) in the first iteration, for index i to become the culprit of a later cap update, w (t) i must reach C/k for some t. By Lemma 3, at most O(1)/(C/k) = O(k/C) indices can satisfy this. Thus, the round complexity is at most min{k, O(k/C)} • O(log C k).
Combining Lemmas 5, 6 and 7 immediately gives the first part of Theorem 4. Theorem 8. For any C ∈ [2, k], in the box setting, LazyHedge finds an O(ε)-approximate maximum with an O(C log 8 (k/ε)) sample overhead in min{O(k log k), O((k/C) • log 9 (k/ε))} rounds.
this section cite: []

Section: Overview of Lower Bounds
The following theorem shows that the poly(k/s) round complexity in Theorem 4 cannot be avoided when ε ≤ 1/ poly(k). For exponentially small ε, the exponents on k/s also match Theorem 4. Theorem 9 (Informal version of Theorems 25 and 28). If ε ≤ O(1/k), every OODS algorithm with sample overhead s must take Ω( k/s) rounds in the box setting and Ω((k/s) 1/4 ) rounds in the ellipsoid setting. If ε ≤ e -Ω(k) , every OODS algorithm with sample overhead s must take Ω(k/s) rounds in the box setting and Ω( k/s) rounds in the ellipsoid setting.
As a corollary, if ε ≤ O(1/k), every OODS algorithm has either a poly(k) round complexity or a sample overhead that is almost linear in k. While these lower bounds do not directly imply lower bounds for agnostic MDL, they show that further improving the round complexity in Proposition 1 requires a substantially different approach. Roughly speaking, the MDL algorithm of [ZZC + 24] fits into the OODS framework because: (1) It uses samples in a restricted way: finding an ERM ĥ on mixture k i=1 w i D i for some weight vector w in an "observable region", and estimating the error of ĥ on every D i ;
(2) By solving the MDL instance, it finds a "hard" mixture k i=1 ŵi D i on which the best hypothesis in H has an error close to the minimax value. The first property ensures that the MDL algorithm requires no more information than what the first-order oracle provides in OODS. The second ensures that the MDL algorithm implicitly solves the OODS problem. Theorem 9 then suggests that every algorithm with the two properties faces an inherent obstacle in achieving a sub-polynomial round complexity.
We sketch the proof of Theorem 9 in the box setting and the ε ≤ O(1/k) regime; the formal proofs are deferred to Appendix D. We consider the objective function f (w) := min j∈[m] {w i ⋆ j + j/m 2 }, where i ⋆ 1 , i ⋆ 2 , . . . , i ⋆ m ∈ [k] are m ≤ k different critical indices sampled uniformly at random. The lower bound builds on two observations on f : (1) (Lemma 23) If ε ≤ O(1/k), every ε-approximate maximum must put an Ω(1/m) weight on at least half of the critical indices i ⋆ 1 , . . . , i ⋆ m ; (2) (Lemma 24) Unless we put a weight of > 1/m 2 on each of i ⋆ 1 , i ⋆ 2 , . . . , i ⋆ j , the value of f (w) is determined by the first j terms in the minimum. Intuitively, unless we already "know" the first j critical indices, we cannot learn the values of i ⋆ j+1 , . . . , i ⋆ m from the first-order oracle. There is a natural m-round algorithm that solves the instance above. In the first round, we query the uniform weight vector w = (1/k, 1/k, . . . , 1/k) to learn the value of i ⋆ 1 . In the second round, we query f on some w with w i ⋆ 1 ≫ 1/m 2 , thereby learning the value of i ⋆ 2 . Repeating this m times recovers all the critical indices. One might hope to be "more clever" and learn many critical indices in a round. For example, the algorithm might put a cap of ≫ 1/m 2 on several coordinates in the first round, in the hope of hitting more than one indices in i ⋆ 1 , i ⋆ 2 , . . .. However, if the algorithm has a sample overhead of s, only O(m 2 s) such guesses can be made. In particular, assuming m ≪ k/s, the O(m 2 s) ≪ k guesses only cover a tiny fraction of the indices. Thus, over the uniform randomness in i ⋆ , the algorithm learns only O(1) critical indices within each round in expectation.
this section cite: []

Section: Discussion
In this work, we formalized the trade-off between sample and round complexities in multi-distribution learning (MDL). For the realizable case, we obtained a nearly tight characterization: when the learner is allowed r rounds of sampling, the optimal sample complexity is proportional to k Θ(1/r) . In particular, a constant number of rounds suffice to achieve a sublinear dependence on k, whereas nearly log k rounds are necessary to reach near-optimal sample complexity.
For the more general agnostic setting, we introduced the optimization via on-demand sampling (OODS) problem as an abstraction of the common approach shared by many recent MDL algorithms. We then leveraged the intuition behind the OODS algorithms to obtain an improved round complexity of O( √ k). On the negative side, any MDL algorithm based on the OODS approach must take poly(k) rounds to match the near-optimal sample complexity of O((d + k)/ε 2 ).
this section cite: []

Section: References
Ref_id:b0 Title: Learning with limited rounds of adaptivity: Coin tossing, multi-armed bandits, and ranking from pairwise comparisons Year: (2017-07-10)
Ref_id:b1 Title: Open problem: The sample complexity of multi-distribution learning for VC classes Year: (2023-07-15)
Ref_id:b2 Title: Collaborative PAC learning Year: (2017-12-04)
Ref_id:b3 Title: An optimal approximation for submodular maximization under a matroid constraint in the adaptive complexity model Year: (2019-06-23)
Ref_id:b4 Title: The adaptive complexity of maximizing a submodular function Year: (2018)
Ref_id:b5 Title: Memory bounds for continual learning Year: (2022-11-03)
Ref_id:b6 Title: Parallelizing greedy for submodular set function maximization in matroids and beyond Year: (2019-06-23)
Ref_id:b7 Title: Submodular function maximization in parallel via the multilinear relaxation Year: (2019)
Ref_id:b8 Title: Tight bounds for collaborative PAC learning via multiplicative weights Year: (2018-12-03)
Ref_id:b9 Title: Collaborative learning with different labeling functions Year: (2024)
Ref_id:b10 Title: Submodular maximization with nearly-optimal approximation and adaptivity in nearly-linear time Year: (2019)
Ref_id:b11 Title: Submodular maximization with matroid and packing constraints in parallel Year: (2019-06-23)
Ref_id:b12 Title: Submodular maximization with nearly optimal approximation, adaptivity and query complexity Year: (2019)
Ref_id:b13 Title: Batched multi-armed bandits problem Year: (2019-12-08)
Ref_id:b14 Title: Refined error bounds for several learning algorithms Year: (2016)
Ref_id:b15 Title: On-demand sampling: Learning optimally from multiple distributions Year: (2022-11-28)
Ref_id:b16 Title: A unifying perspective on multicalibration: Game dynamics for multi-objective learning Year: (2023-12-10)
Ref_id:b17 Title: Top arm identification in multi-armed bandits with batch arm pulls Year: (2016-05-09)
Ref_id:b18 Title: Optimal batched best arm identification Year: (2024-12-10)
Ref_id:b19 Title: Breaking the log(1/∆ 2 ) barrier: Better batched best arm identification with adaptive grids Year: ()
Ref_id:b20 Title: Preventing fairness gerrymandering: Auditing and learning for subgroup fairness Year: (2018)
Ref_id:b21 Title: Bagging is an optimal PAC learner Year: (2023-07-15)
Ref_id:b22 Title: A polynomial lower bound on adaptive complexity of submodular maximization Year: (2020)
Ref_id:b23 Title: Foundations of Machine Learning, second edition. Adaptive Computation and Machine Learning series Year: (2018)
Ref_id:b24 Title: Agnostic federated learning Year: (2019-06)
Ref_id:b25 Title: Improved algorithms for collaborative PAC learning Year: (2018-12-03)
Ref_id:b26 Title: The sample complexity of multi-distribution learning Year: (2023-07-03)
Ref_id:b27 Title: Do outliers ruin collaboration Year: (2018)
Ref_id:b28 Title: Multi-group agnostic PAC learnability Year: (2021-07-24)
Ref_id:b29 Title: The strength of weak learnability Year: (1990)
Ref_id:b30 Title: Boosting: Foundations and Algorithms Year: (2012)
Ref_id:b31 Title: Distributionally robust neural networks Year: (2020)
Ref_id:b32 Title: The data addition dilemma Year: (2024)
Ref_id:b33 Title: Simple and near-optimal algorithms for hidden stratification and multi-group learning Year: (2022-07-23)
Ref_id:b34 Title: Doremi: Optimizing data mixtures speeds up language model pretraining Year: (2023)
Ref_id:b35 Title: Optimal multi-distribution learning Year: (2023-07-03)
