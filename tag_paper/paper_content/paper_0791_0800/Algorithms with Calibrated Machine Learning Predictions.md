Title: Algorithms with Calibrated Machine Learning Predictions
Abstract: The field of algorithms with predictions incorporates machine learning advice in the design of online algorithms to improve real-world performance. A central consideration is the extent to which predictions can be trusted-while existing approaches often require users to specify an aggregate trust level, modern machine learning models can provide estimates of prediction-level uncertainty. In this paper, we propose calibration as a principled and practical tool to bridge this gap, demonstrating the benefits of calibrated advice through two case studies: the ski rental and online job scheduling problems. For ski rental, we design an algorithm that achieves near-optimal prediction-dependent performance and prove that, in high-variance settings, calibrated advice offers more effective guidance than alternative methods for uncertainty quantification. For job scheduling, we demonstrate that using a calibrated predictor leads to significant performance improvements over existing methods. Evaluations on real-world data validate our theoretical findings, highlighting the practical impact of calibration for algorithms with predictions.

Section: Introduction
In recent years, advances in machine learning (ML) models have inspired researchers to revisit the design of classic online algorithms, incorporating insights from ML-based advice to improve decision-making in real-world environments. This research area, termed algorithms with predictions, seeks to design algorithms that are both robust to worst-case inputs and achieve performance that improves with prediction accuracy (a desideratum termed consistency) (Lykouris & Vassilvitskii, 2018). Many learning-augmented algorithms have been developed for online decision-making tasks ranging from rent-or-buy problems like ski rental (Purohit et al., 2018;Anand et al., 2020;Sun et al., 2024) to sequencing problems like job scheduling (Cho et al., 2022).
This framework often produces a family of algorithms indexed by a single parameter intended to reflect the global reliability of the ML advice. Extreme settings of this parameter yield algorithms that make decisions as if the predictions are either all perfect or all uninformative (e.g., Mahdian et al., 2007;Lykouris & Vassilvitskii, 2018;Purohit et al., 2018;Rohatgi, 2020;Wei & Zhang, 2020;Antoniadis et al., 2020). In contrast, ML models often produce local, prediction-specific uncertainty estimates, exposing a disconnect between theory and practice. For instance, many neural networks provide calibrated probabilities or confidence intervals for each data point.
In this paper, we demonstrate that calibration can serve as a powerful tool to bridge this gap. An ML predictor is said to be calibrated if the probabilities it assigns to events match their observed frequencies; when the model outputs a high probability, the event is indeed likely, and when it assigns a low probability, the event rarely occurs. Calibrated predictors convey their uncertainty on each prediction, allowing decision-makers to safely rely on the model's advice, and eliminating the need for ad-hoc reliability estimates. Moreover, calibrating an ML model can easily be accomplished using popular methods (e.g. Platt Scaling (Platt et al., 1999) or Histogram Binning (Zadrozny & Elkan, 2001)) that reduce overconfidence (Vasilev & D'yakonov, 2023).
Although we are the first to study calibration for algorithms with predictions, Sun et al. (2024) proposed using conformal prediction in this setting-a common tool in uncertainty quantification (Vovk et al., 2005;Shafer & Vovk, 2008). Conformal predictions provide instance-specific confidence intervals that cover the target with high probability. While these approaches are orthogonal, we prove that calibration can offer key advantages over conformal prediction, especially when the predicted quantities have high variance. In extreme cases, conformal intervals can become too wide to be informative: for binary predictions, a conformal approach returns {0, 1} unless the true label is nearly certain to be 0 or 1. In contrast, calibration still conveys information that aids decision-making.
this section cite: ['b22', 'b28', 'b0', 'b5', 'b23', 'b22', 'b28', 'b30', 'b37', 'b3', 'b27', 'b38', 'b34', 'b35', 'b31']

Section: Our contributions
We demonstrate the benefit of using calibrated predictors through two case studies: the ski rental and online job scheduling problems. Theoretically, we develop and give performance guarantees for algorithms that incorporate calibrated predictions. We validate our theoretical findings with strong empirical results on real-world data, highlighting the practical benefits of our approach.
this section cite: []

Section: Ski rental.
The ski rental problem serves as a prototypical example of a broad family of online rent-or-buy problems, where one must choose between an inexpensive, short-term option (renting) and a more costly, long-term option (buying). In this problem, a skier will ski for an unknown number of days and, each day, must decide to either rent skis or pay a one-time cost to buy them. Generalizations of the ski rental problem have informed a broad array of practical applications in networking (Karlin et al., 2001), caching (Karlin et al., 1988), and cloud computing (Khanafer et al., 2013).
We design an online algorithm for ski rental that incorporates predictions from a calibrated predictor. We prove that our algorithm achieves optimal expected prediction-level performance for general distributions over instances and calibrated predictors. At a distribution level, its performance degrades smoothly as a function of the mean-squared error and calibration error of the predictor. Moreover, we demonstrate that calibrated predictions can be more informative than the conformal predictions of Sun et al. (2024) when the distribution over instances has high variance that is not explained by features, leading to better performance.
Scheduling. We next study online scheduling in a setting where each job has an urgency level, but only a machinelearned estimate of that urgency is available. This framework is motivated by scenarios such as medical diagnostics, where machine-learning tools can flag potentially urgent cases but cannot fully replace human experts.
We demonstrate that using a calibrated predictor provides significantly better guarantees than prior work (Cho et al., 2022), which approached this problem by ordering jobs based on the outputs of a binary predictor. We identify that this method implicitly relies on a crude form of calibration that assigns only two distinct values, resulting in many ties that must be broken randomly. In contrast, we prove that a properly calibrated predictor with finer-grained confidence levels provides a more nuanced job ordering, rigorously quantifying the resulting performance gains.
this section cite: ['b16', 'b14', 'b17', 'b5']

Section: Related work Algorithms with predictions.
There has been significant recent interest in integrating ML advice into the design of online algorithms (see, e.g., Mitzenmacher & Vassilvitskii (2022) for a survey). Much of the research provides a parameterized family of algorithms with no assumption on the reliability of predictions (e.g., Lykouris & Vassilvitskii, 2018;Purohit et al., 2018;Wei & Zhang, 2020). Subsequent work has studied more practical settings, such as assuming access to ML predictors learned from samples (Anand et al., 2020), with probabilistic correctness guarantees (Gupta et al., 2022), with a known confusion matrix (Cho et al., 2022), or that provide distributional predictions (Dinitz et al., 2024;Angelopoulos et al., 2024;Lin et al., 2022;Diakonikolas et al., 2021). While conceptually related, these papers do not study uncertainty quantification.
Recently, Sun et al. (2024) proposed a framework for quantifying prediction-level uncertainty based on conformal prediction. We show that calibration can offer key advantages over conformal prediction in this context, particularly when predicted quantities exhibit high variance.
this section cite: ['b24', 'b22', 'b28', 'b37', 'b0', 'b5', 'b7']

Section: Calibration for decision-making.
A recent line of work examines calibration as a tool for downstream decisionmaking. Gopalan et al. (2023) show that a multi-calibrated predictor can be used to optimize any convex, Lipschitz loss function of an action and binary label. Zhao et al. (2021) adapt the required calibration guarantees to specific offline decision-making tasks, while Noarov et al. (2023) extend this algorithmic framework to the online adversarial setting. Though closely related to our work, these results do not extend to the (often unwieldy) loss functions encountered in competitive analysis.
this section cite: []

Section: Preliminaries
For clarity, we follow the convention that capital letters (e.g., X) denote random variables and lowercase letters denote realizations of random variables (e.g., the event f (X) = v).
this section cite: []

Section: Learning-augmented algorithm design.
With each algorithmic task, we associate a set I of possible instances, a set X of features for those instances, and a joint distribution D over X × I. Given a target function T : I → Y that provides information about each instance, we assume access to a predictor f : X → Z ⊇ Y that has been trained to predict the target over D. Let R(f ) denote the range of f . If A(v, i) is the cost incurred by algorithm A with prediction f (X) = v on instance i ∈ I, and OPT(i) is that of the offline optimal solution, the goal is to minimize either the expected competitive ratio (CR)
E (X,I)∼D A(f (X), I)
this section cite: []

Section: OPT(I)
or the expected additive regret E [A(f (X), I) -OPT(I)], depending on context. Both measure the performance of A relative to OPT over D. The former is consistent with prior work on training predictors from samples for algorithms with predictions (Anand et al., 2020), while the latter is commonly used to quantify suboptimality in learningaugmented scheduling (Lindermayr & Megow, 2022;Im et al., 2023). When D and f are clear from context, we refer to these quantities as E[CR(A)] and E[R(A)], respectively.
Calibration. An ML model is said to be calibrated if its predictions are, on average, correct. Formally, Definition 2.1. A predictor f : X → Z with target T :
I → Y is calibrated over D if E (X,I)∼D [T (I) | f (X)] = f (X). When Y = {0, 1}, the equivalent condition Pr[T (I) = 1 | f (X)] = f (X) requires that f (X) is a reliable probabilistic estimate of the event {T (I) = 1}.
A classic result from the literature on probabilistic forecasting states that calibrated predictions are the global minimizers of proper loss functions (DeGroot & Fienberg, 1983). However, achieving perfect calibration is difficult in practice.
As a result, post-hoc calibration methods aim to minimize calibration error, such as the max calibration error, which measures the largest deviation from perfect calibration for any prediction.
Definition 2.2. The max calibration error of a predictor f : X → Z with target T :
I → Y over D is max v∈R(f ) |v -E[T (I) | f (X) = v]| .
Given any black box ML model and sufficient data, these methods yield a new predictor with a desired level of calibration error with high probability.
this section cite: ['b0', 'b21', 'b6']

Section: Ski Rental
In this section, we analyze calibration as a tool for uncertainty quantification in the classic online ski rental problem. All omitted proofs in this section are in Appendix A.
this section cite: []

Section: Setup
Problem. A skier plans to ski for an unknown number of days Z ∈ N and has two options: buy skis at a one-time cost of b ∈ N dollars or rent them for 1 dollar per day. The goal is to determine how many days to rent before buying, minimizing the total cost. If Z = z were known a priori, the optimal policy would rent for b days when z < b and buy immediately otherwise, costing min{z, b}. Without knowledge of z, competitive ratios of 2 (Karlin et al., 1988) and e e-1 (Karlin et al., 1994) are tight for deterministic and random strategies, respectively. For convenience, we study a continuous variant of this problem where Z, b, k ∈ R ≥0 as in prior work (Anand et al., 2020;Sun et al., 2024).
Algorithm 1 A k * input: prediction f (X) = v, max calibration error α if v ≤ 4+3α 5
then Rent for b days before buying. else
Rent for b 1-v+α v+α days before buying. end if Predictions. Let X be a set of skier features, I = R ≥0 be the set of possible days skied, and D be an unknown distribution over feature/duration pairs X × R ≥0 . Motivated by the form of the optimal offline algorithm, we analyze a calibrated predictor f : X → [0, 1] for the target T (z) = 1 {z>b} , indicating if the skier will ski for more than b days.
For (X, Z) ∼ D, a prediction of f (X) ≈ 1 (respectively, f (X) ≈ 0) means Z > b (respectively, Z ≤ b) with high certainty.
Learning-augmented ski rental. A deterministic learning-augmented algorithm A k for ski rental takes as input a prediction f (X) = v and returns a recommendation: "rent skis for k(v) days before buying." The cost of following this policy when skiing for z days is
A k (v, z) = k(v) + b if z > k(v) z if z ≤ k(v) .
We aim to select k : [0, 1] → R + to minimize E[CR(Ak)].
this section cite: ['b14', 'b15', 'b0']

Section: Ski rental with calibrated predictions
In Algorithm 1, we introduce a deterministic policy for ski rental based on calibrated predictions. To avoid following bad advice, the algorithm defaults to a worst-case strategy of renting for b days unless sufficiently confident that the skier will ski for at least b days. In this second case, the algorithm smoothly interpolates between a strategy that rents for b (1 -α)/α days and one that rents for b α/(1 + α) days, where α ∈ [0, 1] is a bound on local calibration error that hedges against greedily following predictions.
Theorem 3.1. Given a predictor f with mean-squared error η and max calibration error α, Algorithm 1 achieves
E[CR(Ak * )] ≤ 1 + 2α + min E[f (X)] + α, 2 √ η + 3α .
As the predictor becomes more accurate (i.e., both η and α decrease), the algorithm's expected CR approaches 1. The rest of this subsection will build to a proof of Theorem 3.1.
Prediction-level analysis. We begin by upper bounding
E[CR(Ak) | f (X) = v]. Let B v = {f (X)
= v} be the event that f predicts v ∈ R(f ) and C = {Z > b} be the
(i) z ≤ min{k(v), b} z z (ii) k(v) < z ≤ b z k(v) + b (iii) b < z ≤ k(v) b z (iv) z > max{k(v), b} b k(v) + b
event that the number of days skied is more than b. Then
E[CR(Ak) | B v ] = E[CR(Ak) | B v , C] • Pr[C | B v ] (1) + E[CR(Ak) | B v , C c ] • Pr[C c | B v ].
Lemma 3.2 bounds each of the quantities from Equation (1). Lemma 3.2. Given a predictor f with max calibration error α, for all v ∈ R(f ),
1. Pr[C | f (X) = v] ≤ v + α 2. Pr[C c | f (X) = v] ≤ 1 -v + α 3. E[CR(Ak) | B v , C] ≤ 1 + k(v) b 4. E[CR(Ak) | B v , C c ] ≤ 1 + b•1 {k(v)<b} k(v)
.
Proof sketch. (1) and ( 2) follow from the fact that f predicts 1 C with max calibration error α. Under C = {Z ≥ b}, one of conditions (iii) or (iv) from Table 3 hold. In either case, A k (v, Z)/OPT(Z) ≤ 1 + k(v) b . Under C c , one of conditions (i) or (ii) hold. CR(A k ) = 1 for (i). For (ii),
A k (v, Z) OPT(Z) ≤ k(v) + b k(v) = 1 + b • 1 {k(v)<b} k(v) .
Applying all four bounds to Equation ( 1) yields
E[CR(Ak) | f (X) = v] ≤ (2) 1 + 2α + (v + α)k(v) b + 1 {k(v)<b} • (1 -v + α)b k(v) .
The renting strategy k * (v) from Algorithm 1 is the minimizer of the upper bound in Equation ( 2). Theorem 3.3. Given a predictor f with max calibration error α, for any prediction v ∈ R(f ), Algorithm 1 achieves
E[CR(Ak * ) | f (X) = v] ≤ 1 + 2α + min v + α, 2 (v + α)(1 -v + α) .
Proof sketch. Given a prediction f (X) = v, Algorithm 1 rents for k * (v) days where
k * (v) = b if 0 ≤ v ≤ 4+3α 5 b 1-v+α v+α if 4+3α 5 < v ≤ 1.
Evaluating the right-hand-side of Equation (2) at k * (v) gives
1 + 2α + (v + α) if 0 ≤ v ≤ 4+3α 5 1 + 2α + 2 (v + α)(1 -v + α) if 4+3α 5 < v ≤ 1.
The fact that v + α ≤ 2 (v + α)(1 -v + α) for v ∈ [0, 4+3α 5 ] and v + α > 2 (v + α)(1 -v + α) for v ∈ ( 4+3α 5 , 1] completes the proof.
Moreover, no deterministic learning-augmented algorithm for ski rental can outperform Algorithm 1 for general distributions D and calibrated predictors f . The construction is non-trivial, so we refer the reader to the proof in Appendix A.
Theorem 3.4. For all renting strategies k : [0, 1] → R + , predictions v ∈ [0, 1] and ϵ > 0, there exists a distribution D ϵ v and a calibrated predictor f such that
E[CR(Ak) | f (X) = v] ≥ 1 + min v, 2 v(1 -v) -ϵ.
this section cite: []

Section: Global analysis.
In extracting a global bound from the conditional guarantee in Theorem 3.3, we encounter a term (f (X) + α)(1 -f (X) + α) that is an upper bound on the variance of the conditional distribution 1 {Z≥b} | f (X).
Lemma 3.5 relates this quantity to error statistics of f . Lemma 3.5. If f : X → [0, 1] has mean-squared error η and max calibration error α, then
E[f (X)(1 -f (X))] ≤ η + α.
Finally, we prove this section's main theorem.
this section cite: []

Section: Proof of Theorem 3.1. By the tower property of conditional expectation, E[CR(Ak
* )] = E E[CR(Ak * ) | f (X)] . Ap- plying Theorem 3.3 yields E[CR(Ak * )] ≤ 1 + 2α + E min f (X) + α, 2 (f (X) + α)(1 -f (X) + α) . Recall that E[min(X, Y )] ≤ min(E[X], E[Y ]) for ran- dom variables X, Y . Furthermore, the function h(y) = (y + α)(1 -y + α) is concave over the unit interval, so by Jensen's inequality E min f (X) + α, 2 (f (X) + α)(1 -f (X) + α) ≤ min E[f (X)] + α, 2 E[(f (X) + α)(1 -f (X) + α)] .
Finally, observe that
(f (X) + α)(1 -f (X) + α) ≤ f (X)(1 -f (X)) + 2α. We apply Lemma 3.5 to bound E[f (X)(1 -f (X))].
this section cite: []

Section: Comparison to previous work
Consistency and robustness. It is well known that for λ ∈ (0, 1), any (1 + λ)-consistent algorithm for deterministic ski rental must be at least (1 + 1 λ )-robust (Wei & Zhang, 2020;Angelopoulos et al., 2020;Gollapudi & Panigrahi, 2019). While Algorithm 1 is subject to this trade-off in the worst case, calibration provides sufficient information to hedge against adversarial inputs in expectation, leading to substantial improvements in average-case performance. Indeed, it can be seen from the bound in Theorem 3.3 that Algorithm 1 is 1-consistent and always satisfies E[CR(Ak * )] ≤ 1.8 when advice is calibrated (α = 0). An analysis similar to that of Theorem 15 in Anand et al. (2020) shows that Algorithm 1 is g(α)-robust, where
g(α) = 1 + 1+α α if α < 1/3 2 if α ≥ 1/3
is a decreasing function of α. This is because Algorithm 1 executes a worst-case 2-competitive strategy when α ≥ 1/3 and never buys skis before day b α 1+α otherwise.
We note that one can run the same algorithm using an artificial upper bound α ′ > α on max calibration error to achieve an improved robustness level g(α ′ ). As seen from the bounds in Theorem 3.3 and Theorem 3.1, this adjustment will come at the cost of expected performance, highlighting the tradeoff between average and worst-case performance.
this section cite: ['b37', 'b1', 'b9', 'b0']

Section: Uncertainty quantification.
We are not the first to explore uncertainty quantified predictions for ski rental. Sun et al. (2024) take an orthogonal approach based on conformal prediction. Their method, Algorithm 2, assumes access to a probabilistic interval predictor PIP δ : X → P([0, 1]). PIP δ outputs an interval [ℓ, u] = PIP δ (X) containing the true number of days skied Z ∈ [ℓ, u] with probability at least 1 -δ. Interval predictions are especially useful when the uncertainty δ and interval width u -ℓ are both small. However, as features become less informative, the width of prediction intervals must increase to maintain the same confidence level. This can result in intervals that are too wide to provide meaningful insight into the true number of days skied. Lemma 3.6 and Theorem 3.7 demonstrate that there are infinite families of distributions for which calibrated predictions are more informative than conformal predictions for ski rental. Lemma 3.6. For all a ∈ [0, 1/2], there exists an infinite family of input distributions for which Algorithm 2 defaults to a worst-case break-even strategy for all interval predictors PIP δ with uncertainty δ < a.
Proof sketch. The construction places mass 1 -a on some day z 1 ≤ b 2 and mass a on z 2 ≥ 2b. Any PIP δ with δ < a Algorithm 2 (Sun et al., 2024) Optimal ski rental with conformal predictions
input: interval prediction [ℓ, u] = PIP δ (X) if ℓ ≤ u < b then Rent for b days else if b < ℓ ≤ u then Rent for b • min{ δ/1 -δ, 1} days else if ζ(δ, ℓ) ≥ 2 and δ + u b ≥ 2 then Rent for b days else if ζ(δ, ℓ) ≤ δ + u b then Rent for ℓ • min{ bδ/ℓ(1 -δ), 1} days else Rent for u days end if end if ζ(δ, ℓ) := δ + (1-δ)b ℓ + 2 δ(1-δ)b ℓ if δ ∈ [0, ℓ ℓ+b ) 1 + b ℓ if δ ∈ [ ℓ ℓ+b , 1]
must output an interval [ℓ, u] containing both z 1 and z 2 . Moreover, ζ(δ, ℓ) ≥ 2 and δ + u b ≥ 2 by construction.
Theorem 3.7. For all a ∈ [0, 1/2], all instantiations A of Algorithm 2 using PIPs with uncertainty δ < a, and all distributions from Lemma 3.6, if f is a predictor with meansquared error η and max calibration error α satisfying
2α + 2 √ η + 3α < a, then E[CR(Ak * )] < E[CR(A)].
Proof sketch. For the distributions in Lemma 3.6, the number of days skied is greater than b with probability a. Thus, the expected competitive ratio of the break-even strategy is
E[CR(A)] = a • 2 + (1 -a) • 1 = 1 + a.
The result follows from the bound on E[CR(Ak * )] given in Theorem 3.1.
this section cite: []

Section: Online Job Scheduling
In this section, we explore the role of calibration in a model for scheduling with predictions first proposed by Cho et al. (2022) to direct human review of ML-flagged abnormalities in diagnostic radiology. Omitted proofs from this section can be found in Appendix B.
this section cite: ['b5']

Section: Setup
Problem. There is a single machine (lab tech) that needs to process n jobs (diagnostic images), each requiring one unit of processing time. Job i has some unknown priority y i ∈ {0, 1} that is independently high (y i = 1) with probability ρ and low (y i = 0) with probability 1 -ρ. Although job priorities are unknown a priori, the priority y i is revealed after completing some fixed fraction θ ∈ (0, 1) of job i. Upon learning y i , a scheduling algorithm can choose to complete job i, or switch to a new job and "store" job i for completion at a later time. The goal is to schedule the n jobs in a way that minimizes the weighted sum of completion times n i=1 C i • ω yi where C i is the completion time of job i, and ω 1 > ω 0 > 0 are costs associated with delaying a job of each priority for one unit of time. In hindsight, it is optimal to schedule jobs in decreasing order of priority.
this section cite: []

Section: ML predictions.
Based on the assumption that the n jobs to be scheduled are iid, let X = X n 0 be a set of job features, I = {0, 1} n be the set of possible priorities, and D = D n 0 be an unknown joint distribution over feature/priority pairs. The prediction task for this problem involves training a predictor f whose target is the true priority of each job T (⃗ y) = ⃗ y. This amounts to training a 1-dimensional predictor f : X 0 → Z that acts on the n jobs independently:
f ( ⃗ X) := (f ( ⃗ X 1 ), . . . , f ( ⃗ X n )).
Learning-augmented scheduling. Cho et al. (2022) introduce a threshold-based scheduling rule informed by probabilities p i that job i is high priority based on identifying features (Algorithm 3). Their algorithm switches between two extremes-a preemptive policy that starts a new job whenever the current job is revealed to be low priority, and a non-preemptive policy that completes any job once it is begun-based on the threshold parameter
β := θ 1 -θ • ω 1 ω 1 -ω 0 .
In detail, jobs are opened in decreasing order of p i . Jobs with p i > β are processed preemptively, and the remaining jobs are processed non-preemptively.
A learning-augmented algorithm A for job scheduling determines the probabilities p i from ML advice. Cho et al. (2022) assume access to a binary predictor f b : X 0 → {0, 1} of job priority and study the case where
p i = Pr[ ⃗ Y i = 1 | f b ( ⃗ X i )]
. These probabilities can be computed using Bayes' rule, and because f b is binary, this procedure effectively assigns each job one of two probabilities. Although not explicitly discussed by Cho et al. (2022), this amounts to a basic form of post-hoc calibration. In contrast, our results extend to arbitrary calibrated predictors f : X 0 → [0, 1]-a more general framework that calls for new mathematical techniquesallowing us to significantly improve upon their results. In this setting, A takes the predictions f ( ⃗ X) = ⃗ v as input and executes Algorithm 3 with probabilities p i = ⃗ v i .
To quantify the optimality gap of A, Cho et al. (2022) note that compared to OPT, Algorithm 3 incurs (1) a cost of θω 1 for each inversion, or pair of jobs whose true priorities y i are out of order, and (2) a cost of θω 0 for each pair of low priority jobs encountered when acting preemptively. When acting non-preemptively, Algorithm 3 incurs (3) a cost of ω 1 -ω 0 for each inversion. Thus, for fixed predictions
Algorithm 3 β-threshold rule input: Probabilities {p i } n i=1 that each job is high-priority Define n 1 = |{i : p i > β}| Order probabilities p (1) ≥ • • • ≥ p (n)
Run jobs j (1) , . . . , j (n1) preemptively, in order Complete remaining jobs non-preemptively, in order f ( ⃗ X) = ⃗ v and true job priorities ⃗ y,
A(⃗ v, ⃗ y) -OPT(⃗ y) (3) = θω 1 L(⃗ v, ⃗ y) + θω 0 M (⃗ v, ⃗ y) + (ω 1 -ω 0 )N (⃗ v, ⃗ y),
where L(⃗ v, ⃗ y), M (⃗ v, ⃗ y), and N (⃗ v, ⃗ y) count occurrences of (1), (2), and (3), respectively (see Table 2 for details).
this section cite: ['b5', 'b5', 'b5', 'b5']

Section: Scheduling with calibrated predictions
Calibration and job sequencing. To build intuition for why finer-grained calibrated predictors sequence jobs more accurately, we begin by observing that Algorithm 3 orders jobs with the same probability p i randomly. Given a calibrated predictor f , consider the coarse calibrated predictor
f ′ (x) = E[f (X) | f (X) > β] if f (x) > β E[f (X) | f (X) ≤ β] if f (x) ≤ β
obtained by averaging the predictions of f above and below the threshold β. Whereas |R(f )| may be large, f ′ is only capable of outputting |R(f ′ )| = 2 values. As a result, when ordering jobs with features X 1 , . . . , X n according to predictions from f ′ , all jobs with f (X) > β will be sequenced before jobs with f (X) ≤ β, but the ordering of jobs within these bins will be random. In contrast, predictions from f provide a more informative ordering of jobs (Figure 1). Note, however, that f = f ′ when f has no variance in its predictions above or below the threshold β. We demonstrate in Theorem 4.3 that this intuition holds in general: improvements scale with the granularity of predictions.
f (X) f ′ (X) 0 β 1 × × × × × × × × 6 5 4 3 2 1 4 5 6 2 3 1 Figure 1.
Job sequencing under fine-grained (above) and coarse (below) calibrated predictors. For six example jobs, predicted probabilities pi are marked with ×, and numbered boxes give the order of jobs according to each predictor.
Table 2. Quantities of interest in learning-augmented scheduling for fixed predictions f ( ⃗ X) = ⃗ v and job priorities ⃗ y.
this section cite: []

Section: Quantity Description
Relevant setting
n 1 = |{i : ⃗ v i > β}|
Number of jobs likely to be high priority.
-
L(⃗ v, ⃗ y) = n1 i=1 n1 j=i+1 1 {⃗ y (i) =0∧⃗ y (j) =1}
Number of inversions among jobs likely to be high priority.
Preemptive
M (⃗ v, ⃗ y) = n1 i=1 n1 j=i+1 1 {⃗ y (i) =0∧⃗ y (j) =0}
Number of low-priority job pairs among jobs likely to be high priority.
Preemptive
N (⃗ v, ⃗ y) = n i=1 n j=i+1 1 {⃗ y (i) =0∧⃗ y (j) =1} -L(⃗ v, ⃗ y) Number
of inversions among job pairs where at least one is likely to be low priority. Non-preemptive Performance analysis. Building off of Equation (3), we bound the expected competitive ratio E[CR(A)] by bounding each of E[
L(f ( ⃗ X), ⃗ Y )], E[M (f ( ⃗ X), ⃗ Y )],and
E[N (f ( ⃗ X), ⃗ Y )].
The dependence on the ordering of predictions from f in these random counts means our analysis heavily involves functions of order statistics. For example, considering the shared summand of L(•) and N (•),
E 1 { ⃗ Y (i) =0} • 1 { ⃗ Y (j) =1} | f ( ⃗ X) = Pr[ ⃗ Y (i) = 0 | f ( ⃗ X (i) )] • Pr[ ⃗ Y (j) = 0 | f ( ⃗ X (j) )] = (1 -f ( ⃗ X (i) ))f ( ⃗ X (j) ) = g(f ( ⃗ X (i) ), f ( ⃗ X (j) ))
for the function g(x, y) = (1 -x)y. Similarly, the analysis for the summand of M (•) yields g(f ( ⃗ X (i) ), f ( ⃗ X (j) )) for g(x, y) = (1 -x)(1 -y). Based on this, our high-level strategy is to relate "ordered" expectations of the form
E   n i=1 n j=i+1 g f ( ⃗ X (i) ), f ( ⃗ X (j) )   to their "unordered" counterparts E   n i=1 n j=i+1 g f ( ⃗ X i ), f ( ⃗ X j ) ,  
which are simple to compute. Lemma 4.1 shows that the ordered and unordered expectations are, in fact, equivalent when the function g satisfies g(x, y) = g(y, x).
Lemma 4.1. Let X 1 , . . . , X n be iid random variables with
order statistics X (1) ≥ • • • ≥ X (n) . For any symmetric function g : R × R → R, n i=1 n j=i+1 g(X (i) , X (j) ) = n i=1 n j=i+1 g(X i , X j ).
This result is sufficient to compute the expectation of M (•) exactly. For the other counts, the analysis is more technical as g(x, y) = (1 -x)y is not symmetric. Lemma 4.2 characterizes the relationship between the ordered and unordered expectations for the function g(x, y) = (1 -x)y. Lemma 4.2. Let X 1 , . . . , X n be iid samples from a distribution over the unit interval [0, 1] with order statistics
X (1) ≥ • • • ≥ X (n) . Then, E   n i=1 n j=i+1 (1 -X (i) ) • X (j)   ≤ E   n i=1 n j=i+1 (1 -X i ) • X j   - n 2 • Var(X 1 ).
Proof sketch. By Lemma 4.1 with g(x, y) = xy,
n i=1 n j=i+1 X (i) • X (j) = n i=1 n j=i+1 X i • X j
can be removed from both sides. Then, we apply Lemma 4.1 with g(x, y) = min(x, y) to simplify the left-hand-side.
n i=1 n j=i+1 X (j) = n i=1 n j=i+1 min{X (i) , X (j) } = n i=1 n j=i+1 min{X i , X j }. Finally, we show that E[X1 -min{X 1 , X 2 }] ≥ Var(X 1 ). Note that E[X1] -E[min{X1, X 2 }] = 1 2 E |X 1 -X 2 | since X 1 -min{X 1 , X 2 } = 0 if X 1 ≤ X 2 |X 1 -X 2 | if X 1 > X 2 . Finally, E |X 1 -X 2 | ≥ E |X 1 -X 2 | 2 = 2Var(X 1 ).
With careful conditioning to deal with random summation bounds, we apply Lemma 4.2 to bound the expectations of L(•) and N (•), giving this section's main theorem. Of note, Theorem 4.3 says that the expected number of inversions of high and low priority jobs decreases with predictor granularity, measured by κ 1 and κ 2 . For the method from Cho et al. (2022), κ 1 = κ 2 = 0 and the inequalities hold with equality. An analogous result holds under the weaker assumption that f monotonically calibrated. That is, the empirical frequencies Pr[Y = 1 | f (X)] are non-decreasing in the prediction f (X). This property holds trivially for calibrated predictors, but zero calibration error is not required. In fact, many calibration approaches used in practice (e.g. Platt scaling (Platt et al., 1999) and isotonic regression (Zadrozny & Elkan, 2001)) produce a monotonically calibrated predictor with non-zero calibration error. See Appendix B for details.
Theorem 4.3. Let f be calibrated, with Pr[f (X) > β | Y = 0] = ϵ 0 , Pr[f (X) ≤ β | Y = 1] = ϵ 1 , κ 1 = Pr[f (X) > β] 2 • Var(f (X) | f (X) > β),and
κ 2 = Pr[f (X) ≤ β] 2 • Var(f (X) | f (X) ≤ β). Then 1. E[L(f ( ⃗ X), ⃗ Y )] ≤ n 2 ρ(1 -ρ)(1 + ϵ 0 )ϵ 1 -κ 1 2. E[M (f ( ⃗ X), ⃗ Y )] = n 2 (1 -ρ) 2 ϵ 2 0 3. E[N (f ( ⃗ X), ⃗ Y )] ≤ n 2 ρ(1 -ρ)ϵ 0 (1 -ϵ 1 ) -κ 2 Remark 4.4. A(f ( ⃗ X), •) -OPT(•) = 0 when ϵ 0 = ϵ 1 = 0,
this section cite: ['b5', 'b27', 'b38']

Section: Experiments
We now evaluate our algorithms on two real-world datasets, demonstrating the utility of using calibrated predictions. See Appendix C for additional details about our datasets and model training, as well a broader collection of results for different ML models and parameter settings. 1
this section cite: []

Section: Ski rental: Citi Bike rentals
To model the rent-or-buy scenario in the ski rental problem, we use publicly available Citi Bike usage data. 2 . This dataset has been used for forecasting (Wang, 2016), system balancing (O'Mahony & Shmoys, 2015), and transportation policy (Lei & Ozbay, 2021), but to the best of our knowledge, this is its first use for ski rental. In this context, a Citi Bike user can choose one of two options: pay by ride duration (rent) or purchase a day pass (buy).
If the user plans 1 Code and data available here: https://github.com/  heyyjudes/algs-cali-pred 2 Monthly usage data is publicly available at https://  citibikenyc.com/system-data. 4 6 8 10 12 14 16 Breakeven point (minutes) 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 [ALG/OPT] ALG Conformal Binary Breakeven Calibrated Figure 2. Comparison of E[ALG/OPT] for algorithms aided by predictions from a small MLP with two hidden layers of size 8 and 2. Algorithm 1 (CALIBRATED) performs best on average.
to ride for longer than the break-even point of b minutes, it is cheaper to buy a day pass than to pay by trip duration. 3We use single-ride durations to approximate the rent vs. buy trade-off for a spectrum of break-even points b. The distribution over ride durations can be seen in Appendix C.
We analyze the impact of advice from multiple predictor families, including XGBoost, logistic regression, and small multi-layer perceptrons (MLP). Each predictor has access to available ride features: start time, start location, user age, user gender, user membership, and approximate end station latitude. While these features are not extremely informative, most predictor families are able to achieve AUC and accuracy above 0.8 for b > 6. Figure 2 summarizes the expected competitive ratios achieved by our method from Algorithm 1 (CALIBRATED) and baselines from previous work when given advice from a small neural network. Baselines include the worst-case optimal deterministic algorithm that rents for b minutes (Karlin et al., 1988) (BREAKEVEN), the black-box binary predictor ski-rental algorithm by Anand et al. (2020) (BINARY), and the PIP algorithm described in Algorithm 2 (Sun et al., 2024) (CONFORMAL). Though each algorithm is aided by predictors from the same family, the actual advice may differ. For example, CONFORMAL assumes access to a regressor that predicts ride duration directly. While performance is distribution-dependent, we see that our calibration-based approach often leads to the most cost-effective rent/buy policy in this scenario.
this section cite: ['b36', 'b19', 'b14']

Section: Scheduling: sepsis triage
We use a real-world dataset for sepsis prediction to validate our theory results for scheduling with calibrated predictions. Sepsis is a life-threatening response to infection that typically appears after hospital admission (Singer et al., 2016).
2 3 4 5 6 7 8 9 1 / 0 ratio 0 1 2 3 4 5 6 7 8 [ALG OPT] Many works have studied using machine learning to predict the onset of sepsis, as every hour of delayed treatment is associated with a 4-8% increase in mortality (Kumar et al., 2006;Reyna et al., 2020); existing works aim to better predict sepsis to treat high-priority patients earlier. Replicating results from Chicco & Jurman (2020) we train a binary predictor for sepsis onset using logistic regression on a dataset of 110,204 hospital admissions. The base predictor achieves an AUC of 0.86 using age, sex, and septic episodes as features. We then calibrate this predictor using both the naive method from Cho et al. (2022) (BINARY) and more nuanced histogram calibration (Zadrozny & Elkan, 2001) (CALIBRATED). Figure 3 shows the expected competitive ratio (normalized by the number of jobs n = 100) achieved by Algorithm 3 when provided advice from each of these predictors for varying delay costs ω 1 , ω 0 and information barrier θ. We see that the more nuanced predictions consistently result in schedules with smaller delay costs.
this section cite: ['b32', 'b18', 'b29', 'b38']

Section: Conclusion
In this paper, we demonstrated that calibration is a powerful tool for algorithms with predictions in settings where performance is measured over a distribution and probabilistic estimates of a binary target enable good decisions. In particular, calibration bridges the gap between traditional theoretical approaches-which treat all predictions as equally reliable-and modern ML methodologies that offer finegrained, instance-specific uncertainty quantification. We focused on the ski rental and online scheduling problems, developing online algorithms that exploit calibration guarantees to achieve strong average-case performance. For both problems, we highlighted settings where our algorithms outperform existing approaches and supported these findings with empirical evidence on real-world datasets.
This work exposes a number of directions for future research.
For ski rental, deriving performance guarantees in terms of binary cross entropy and focusing on less rigid calibration measures (e.g. expected calibration error) offer to further close the gap between theory and practice. More broadly, we believe calibration-based approaches offer broad potential for designing online decision-making algorithms beyond these two case studies, particularly in scenarios that require balancing worst-case robustness with reliable per-instance predictions.
this section cite: []

Section: References
Ref_id:b0 Title: Customizing ML predictions for online algorithms Year: (2020)
Ref_id:b1 Title: Online computation with untrusted advice Year: (2020)
Ref_id:b2 Title: Contract scheduling with distributional and multiple advice Year: ()
Ref_id:b3 Title: Secretary and online matching problems with machine learned advice Year: (2020)
Ref_id:b4 Title: Survival prediction of patients with sepsis from age, sex, and septic episode number alone Year: (2020)
Ref_id:b5 Title: Scheduling with predictions Year: (2022)
Ref_id:b6 Title: The comparison and evaluation of forecasters Year: (1983)
Ref_id:b7 Title: Learning online algorithms with distributional advice Year: (2021)
Ref_id:b8 Title: Binary search with distributional predictions Year: ()
Ref_id:b9 Title: Online algorithms for rentor-buy with expert advice Year: (2019)
Ref_id:b10 Title: Loss minimization through the lens of outcome indistinguishability Year: ()
Ref_id:b11 Title: Augmenting online algorithms with ϵ-accurate predictions Year: ()
Ref_id:b12 Title: Distribution-free calibration guarantees for histogram binning without sample splitting Year: ()
Ref_id:b13 Title: Nonclairvoyant scheduling with predictions Year: ()
Ref_id:b14 Title:  Year: (1988)
Ref_id:b15 Title: Competitive randomized algorithms for nonuniform problems Year: (1994)
Ref_id:b16 Title: Dynamic TCP acknowledgement and other stories about e/(e-1) Year: (2001)
Ref_id:b17 Title: The constrained ski-rental problem and its application to online cloud cost optimization Year: (2013)
Ref_id:b18 Title: Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock Year: (2006)
Ref_id:b19 Title: A robust analysis of the impacts of the stay-at-home policy on taxi and Citi Bike usage: A case study of Manhattan Year: (2021)
Ref_id:b20 Title: Learning augmented binary search trees Year: ()
Ref_id:b21 Title: Permutation predictions for non-clairvoyant scheduling Year: (2022)
Ref_id:b22 Title: Competitive caching with machine learned advice Year: (2018)
Ref_id:b23 Title: Allocating online advertisement space with unreliable estimates Year: (2007)
Ref_id:b24 Title: Algorithms with predictions Year: (2022)
Ref_id:b25 Title: Highdimensional prediction for sequential decision making Year: ()
Ref_id:b26 Title: Data analysis and optimization for (Citi) bike sharing Year: (2015)
Ref_id:b27 Title: Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods Year: (1999)
Ref_id:b28 Title: Improving online algorithms via ML predictions Year: (2018)
Ref_id:b29 Title: Early prediction of sepsis from clinical data: the physionet/computing in cardiology challenge Year: (2019)
Ref_id:b30 Title: Near-optimal bounds for online caching with machine learned advice Year: (2020)
Ref_id:b31 Title: A tutorial on conformal prediction Year: (2008)
Ref_id:b32 Title: The third international consensus definitions for sepsis and septic shock (sepsis-3) Year: (2016)
Ref_id:b33 Title: Online algorithms with uncertainty-quantified predictions Year: ()
Ref_id:b34 Title: Calibration of neural networks Year: (2023)
Ref_id:b35 Title: Algorithmic learning in a random world Year: (2005)
Ref_id:b36 Title: Forecasting Bike Rental Demand Using New York Citi Bike Data Year: (2016)
Ref_id:b37 Title: Optimal robustness-consistency trade-offs for learning-augmented online algorithms Year: (2020)
Ref_id:b38 Title: Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers Year: (2001)
Ref_id:b39 Title: Calibrating predictions to decisions: A novel approach to multi-class calibration Year: ()
