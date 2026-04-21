Title: All-Purpose Mean Estimation over R: Optimal Sub-Gaussianity with Outlier Robustness and Low Moments Performance
Abstract: We consider the basic statistical challenge of designing an "all-purpose" mean estimation algorithm that is recommendable across a variety of settings and models. Recent work by Lee & Valiant (2022) introduced the first 1-d mean estimator whose error in the standard finite-variance+i.i.d. setting is optimal even in its constant factors; experimental demonstration of its good performance was shown by Gobet et al. (2022). Yet, unlike for classic (but not necessarily practical) estimators such as median-of-means and trimmed mean, this new algorithm lacked proven robustness guarantees in other settings, including the settings of adversarial data corruption and heavy-tailed distributions with infinite variance. Such robustness is important for practical use cases. This raises a research question: is it possible to have a mean estimator that is robust, without sacrificing provably optimal performance in the standard i.i.d. setting? In this work, we show that Lee and Valiant's estimator is in fact an "all-purpose" mean estimator by proving:(A) It is robust to an η-fraction of data corruption, even in the strong contamination model; it has optimal estimation error O(σ √ η) for distributions with variance σ 2 . (B) For distributions with finite z th moment, for z ∈ (1, 2), it has optimal estimation error, matching the lower bounds of Devroye et al. (2016) up to constants.We further show (C) that outlier robustness for 1-d mean estimators in fact implies neighborhood optimality, a notion of beyond worst-case and distribution-dependent optimality recently introduced by Dang et al. (2023). Previously, such an

Section: Introduction
The past decade has seen reinvigorated interest in understanding fundamental statistical problems, and in particular, the foundational problem of mean estimation: given n samples from an unknown distribution, and an allowed failure probability δ, what is the most accurate estimate of the distribution mean? The sample mean is the traditional estimate, yet, its sensitivity to extreme values makes it an unreliable estimator in practice. Other classic estimators, for example the well-known median-of-means estimator (Alon et al., 1999;Nemirovsky & Yudin, 1983;Jerrum et al., 1986), also suffer from poor empirical performance.
Recent work by Lee & Valiant (2022), sitting in a line of research initiated by the seminal paper of Catoni (2012), gave the first 1-dimensional mean estimator whose estimation error is optimal (i.e. sub-Gaussian error) and tight even in the leading constant. Gobet et al. (2022) further demonstrated its good empirical performance against other competing estimators.
Fact 1.1 (Lee & Valiant 2022). Given any distribution D with mean µ and variance σ 2 (both unknown), parameters n, δ > 0, let X be a set of n independent samples from D. Then, with probability at least 1 -δ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ n  
Here, the o(1) term tends to 0 as log 1 δ n , δ → (0, 0) and, crucially, is independent of D.
Estimator 1 Mean Estimator of Lee and Valiant Inputs: n i.i.d. samples {x i }; Confidence parameter δ.
1. Compute an initial estimate κ using any sub-Gaussian, scale-and-translation-equivariant and robust mean estimator (cf. Facts A.18 and A.23). One possible choice is the median-of-means estimate: evenly partition the data into log 1 δ groups and let κ be the median of the set of means of the groups; 2. Find the solution α to the monotonic, piecewise-linear equation i min(α(x i -κ) 2 , 1) = 1 3 log 1 δ 3. Output:
μ = κ + 1 n i (x i -κ)(1 -min(α(x i -κ) 2 , 1))
The constant " √ 2" in Fact 1.1 is optimal, with matching lower bounds. Essentially, Estimator 1 uses Step 1 to find a preliminary estimate (e.g. using median-of-means, but it can be other choices as well), which is then refined in Steps 2 and 3 into a better estimate-whose accuracy is optimal even in the leading multiplicative constant. Even though the form of the estimator is somewhat complicated-including a function inversion in Step 2-the estimator can in fact be computed easily in quasilinear time via a sorting operation, or even in linear time using a more subtle algorithm. See (Lee & Valiant, 2022) for more discussion of the design and analysis of Estimator 1.
While Fact 1.1 is a strong recommendation for Estimator 1, getting optimal performance in this regime is not the only practical desideratum for an estimator. Practitioners have long observed that some real-world distributions are very heavy-tailed, with variances that are gigantic (thus rendering guarantees like Fact 1.1 effectively useless), but whose lower moments-between the first and second momentsmight be much more bounded. Further, real-world data can be subject to random or even adversarial corruption. Performance guarantees in these more demanding settings are therefore critical for practice. However, while there are relatively straightforward folklore analyses of classic algorithms in these settings-including, notably, median-of-means and trimmed mean-Lee and Valiant's estimator has not been studied in these contexts. The breadth of guarantees shown for these classic estimators might therefore lead researchers and practitioners to choose either of them over Lee and Valiant's estimator, despite the latter being a more accurate estimator in standard finite variance i.i.d. sample settings.
The natural and pressing research question, then, is Is it possible for a mean estimator to satisfy the important robustness guarantees of classic estimators, without losing the strong and optimal-constant performance that the Lee and Valiant estimator enjoys in the standard setting?
In this paper, we show 4 strong positive results demonstrating the robustness and performance of Estimator 1: (A) In the presence of η-fraction of arbitrary adversarial corruption, the error of Estimator 1 degrades by only O(σ √ η)
from the sub-Gaussian error guaranteed in Fact 1.1. This extra error term is optimal under the assumption that the distribution has (finite) variance σ 2 . (B) When samples are drawn from a distribution with finite z th moment, for z ∈ (1, 2), Estimator 1 achieves error matching the lower bound of Devroye et al. (2016) up to constants.
(C) Estimator 1 enjoys a fine-grained beyond-worst-case notion of optimality called neighborhood optimality, recently introduced by Dang et al. (2023). Previously, only medianof-means was known to be neighborhood optimal, and our result answers the question raised by Dang et al. on whether more modern estimators are also neighborhood optimal.
(D) Under the standard finite variance and i.i.d. setting, Lee and Valiant's estimator is asymptotically normal and efficient.
We emphasize that these new results hold without any changes at all to either the structure or the parameters of Estimator 1. That is, the estimator does not need to know that it is being expected to perform in these challenging settings. The guarantees we show also smoothly revert to Fact 1.1 as the amount of corruption decreases to 0, or as the z th -moment assumption tends to z → 2.
These results, in aggregate, demonstrate that Lee and Valiant's estimator is in fact "all-purpose", enjoying the same breadth of guarantees as either median-of-means or trimmed mean, while also having what is essentially the smallest possible estimation error (even in the leading constant) in the standard setting. Thus, Lee and Valiant's estimator should be used in practice over existing estimators.
this section cite: ['b0', 'b19', 'b14', 'b16', 'b4', 'b10', 'b16', 'b16', 'b7', 'b5']

Section: Our Results at a Glance
We now formally state our technical contributions.
this section cite: []

Section: Outlier robustness
Lee and Valiant's estimator is robust against data corruption from the strong contamination model, the most adversarial data corruption model in the literature.
Definition 2.1 (Strong Contamination Model). Given a corruption parameter η and a distribution D on the uncorrupted data, an algorithm gets a set of n η-corrupted samples from D as follows. The algorithm specifies n and nature draws n i.i.d. samples from D. Then, an arbitrarily powerful adversary can inspect the n samples from D, and arbitrarily replace ⌈ηn⌉ of them before giving the (new) set of samples to the algorithm, with no indication of which samples were corrupted.
In the setting of η-corruption, when we only assume that D has a finite variance σ 2 , it is well-known (Diakonikolas & Kane, 2023) that the minimum estimation error is Ω(σ √ η)
even if there are infinitely many samples. Our analysis of Lee and Valiant's estimator shows that it achieves this estimation error bound, as long as the amount of corruption is O(log 1 δ /n)-where δ, the desired failure probability, is a parameter under the user's control. Put differently, the δ parameter for Estimator 1 can be viewed as "dual use", parameterizing not just the allowed failure probability of the estimator, but also its desired robustness, expressing the maximum tolerated level of corruption.
We emphasize that in the following theorem, Estimator 1 does not need to know the precise value of η, the fraction of corruption. It only makes the assumption that η ≤ O(log 1 δ /n), or in other words, Estimator 1 adapts to the level of corruption.
Theorem 2.2. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D. Suppose both log 1 δ n and δ are bounded by some small universal constant, and suppose η ≤ 1 24n log 1 δ . Then, with probability at least 1 -δ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ n + 222 √ η  
In the above theorem, Estimator 1 gets δ as its input parameter, and fails with probability δ. Accordingly, the estimator error is the sum of the sub-Gaussian error term from Fact 1.1, and a new "robustness" term of O(σ √ η). On the other hand, the standard robustness analysis and guarantee of trimmed mean takes a different form (see Oliveira et al. (2025))-if the desired failure probability is δ ′ and the corruption parameter is η, then the number of trimmed samples is chosen as log 1 δ = Θ(log 1 δ ′ + ηn). We give an analogous theorem for Estimator 1, where again we preserve the optimal constant of √ 2 in the sub-Gaussian error term. Comparing Theorem 2.2 with Theorem 2.3, the latter asks for a failure probability δ ′ > δ, but correspondingly has a smaller sub-Gaussian error term (since log 1 δ ′ < log 1 δ ). Theorem 2.3. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D. Suppose both log 1 δ n and δ are bounded by some small universal constant, and suppose η ≤ 1 9n log 1 δ . Let δ ′ be so that 1 3 log 1 δ = 1 3 log 1 δ ′ + 3ηn. Then, with probability at least 1 -δ ′ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ ′ n + (135 + o(1)) √ η  
We emphasize again that, similar to Theorem 2.2, Theorem 2.3 says that Estimator 1 adapts to the value of η without knowing what it is precisely. Namely, Theorem 2.3 simultaneously holds for all values of δ ′ even though the algorithm is only given a fixed δ.
We sketch the proof of the above two theorems in Section 4 and give full details in Appendix A. While the √ 2 constant in the sub-Gaussian terms is tight, we expect that the constants in the robust error terms (222 and 135) might be significantly improved with a different proof strategy.
In Appendix B, we also show that Estimator 1 is robust in the Huber and Total Variation Contamination Models, in addition to the Strong Contamination Model above.
Median-of-means also has guarantees analogous to the above, but they are folklore. See the proofs of Theorem 2.2 and Theorem 2.3 in Appendix A.4 and Appendix A.3 respectively for the formal statements and proofs of these folklore guarantees. Trimmed mean also gives analogous robustness guarantees, but current analysis (see Oliveira et al. (2025)) requires precise knowledge of η in order to match the optimal rate. In contrast, although Estimator 1 does require an upper bound on η (from the necessary assumption that η ≤ 1 24n log 1 δ ), the accuracy will gracefully improve if the actual corruption rate is less than the pessimistic upper bound.
We further stress that, while median-of-means and trimmed mean both have relatively straightforward and folklore proofs of robustness, the analysis required for the Lee and Valiant estimator is much more intricate-this is due to the fact that Lee and Valiant's estimator is (a lot) more complicated, in order to yield optimal constants in the i.i.d. setting.
this section cite: ['b8', 'b20', 'b20']

Section: Low moment performance
Next, we study the performance of Lee and Valiant's estimator when given data drawn from a distribution that only has finite low moments, specifically, moments between 1 and 2.
Theorem 2.4. Given any distribution D with mean µ and z th moment M z for some z ∈ (1, 2), let X be a set of n i.i.d. samples from D. Then, with probability at least 1 -δ over the randomness of X, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ (M z ) 1 z • (1 + o(1)) c z log 1 δ n 1-1 z
where c z = 2(5.6) 1 z-1 -1 . Here, the o(1) term tends to 0 as log 1 δ n , δ → (0, 0), in a manner independent of D and independent of z.
The above result matches the lower bounds shown by Devroye et al. (2016) up to constants. As z → 2, the guarantee converges to Fact 1.1. Analogous results (up to constants) were shown for median-of-means in Bubeck et al. (2012), but the multiplicative constant we achieve for Estimator 1, (2(5.6)
1 z-1 -1 ) is better than that of median-of- means (8 √ 3(12
) 1 z-1 -1 ) across all values of z ∈ (1, 2].
this section cite: ['b7', 'b3']

Section: Neighborhood optimality
Going beyond the worst-case analysis of Estimator 1, we give a finer-grained analysis of its performance. We provide finite-sample error bounds that optimally adapt to the underlying distribution on an instance-by-instance basis, which is far stronger than just having optimal dependence on the variance.
Recent work by Dang et al. (2023) gave a first study of the fine-grained optimality of 1-dimensional mean estimators, providing upper and lower bounds that match up to constants. At a high level, they showed that sub-Gaussian error bounds are essentially all one can hope for, for any distribution with a finite mean. More specifically, they define an error rate function ϵ n,δ (D) over distributions D, and prove that i) median-of-means attains this error bound; and ii) for any distribution D, there exists a "reasonable" counterpart distribution D ′ such that no algorithm can distinguish between the distributions, and thus no estimator can simultaneously get error ≪ ϵ n,δ (D) using samples from D, while also getting error ≪ ϵ n,δ (D ′ ) using samples from D ′ . We define ϵ n,δ (D) below. The combination of lower and upper bounds of i) and ii) is extended into a new optimality notion called neighborhood optimality by Dang et al.
Intuitively, these are bounds that are optimal in an instanceby-instance basis, because property (ii) shows that no algorithm can get error better than ϵ n,δ (D) on samples from distribution D-even an algorithm customized specifically for distribution D-without getting unacceptably bad error on a designated nearby distribution D ′ .
For concreteness, we define ϵ n,δ (D) below.
Definition 2.5 (Dang et al. 2023). Given a (continuous) distribution D with mean µ and a real number t ∈ [0, 1], define the t-trimming operation on D as follows: select a radius r such that the probability mass in [µ -r, µ + r] equals 1 -t; then, return the distribution D conditioned on lying in [µ -r, µ + r].
Given n and δ, define the trimmed distribution D * n,δ to be the 0.45 n log 1 δ -trimmed version of D. When δ is implicit, we may denote this as D * n . Now define the error function
ϵ n,δ (D) = |µ -µ * n | + σ * n log 1 δ n
, where µ * n and σ * n are the mean and standard deviation of D * n respectively. Dang et alshow that median-of-means achieves error O(ϵ n,δ (D)), and raises the question of whether more modern estimators such as Lee and Valiant's (Estimator 1) also achieve this error bound and are hence neighborhood optimal. We show a more general result: in fact, every sub-Gaussian and robust estimator (satisfying a slight variant of Theorem 2.2) achieves this error bound. Proposition 2.6. Let μ be an arbitrary estimator that, when given δ > 0 and a set of n η-corrupted samples from any distribution D with mean µ and variance σ 2 , outputs a mean estimate satisfying
|μ -µ| ≤ O   σ   log 1 δ n + √ η    
with probability at least 1 -δ 2 over the randomness of the (uncorrupted) samples.
Then, the same estimator μ, on input n i.i.d. samples drawn from a distribution D with finite mean, will output an estimate with error upper bounded by O(ϵ n,δ (D)) (as defined in Definition 2.5) with probability at least 1 -δ.
We formally prove Proposition 2.6 in Appendix C. The precondition of Proposition 2.6 is implied by a mild variant of Theorem 2.2 (decreasing the failure probability from δ to δ/2), which holds also for Estimator 1. As a consequence: Corollary 2.7 (Informal). Estimator 1 is neighborhood optimal, in the sense of Dang et al. (2023).
In Section 5, we state the formal definition of neighborhood optimality and discuss the intuition on Proposition 2.6.
this section cite: ['b5', 'b5', 'b5']

Section: Asymptotic normality and efficiency
Lastly, we show that Lee and Valiant's estimator is asymptotically normal and efficient, under the standard finite variance and i.i.d. sample assumption. In particular, we show that, if the δ parameter in Estimator 1 is fixed, and the number of samples n → ∞, then Estimator 1 converges in probability to the sample mean at the appropriate scale. The Central Limit Theorem for the sample mean then implies the asymptotic optimality of Lee and Valiant's estimator as a corollary. Theorem 2.8. Let D be a distribution with mean µ and variance σ 2 .
Let μ denote Estimator 1 on input parameter δ and n i.i.d. samples from D. Also let Xn denote the sample mean. Then, fixing δ and D and taking n → ∞, we have
√ nμ p → √ n Xn that is, | √ nμ- √ n Xn | p → 0, that √ nμ converges to √ n Xn in probability.
As a corollary, by the Central Limit Theorem, we have
√ n(μ -µ) d → N 0, σ 2
That is, μ is asymptotically normal and efficient.
The above theorem contrasts with the asymptotic behavior of median-of-means, whose error-scaled by √ n-converges to N (0, (π/2)σ 2 ) (Minsker, 2023); median-of-means thus has asymptotic variance a π/2-factor larger than desired.
this section cite: ['b18']

Section: Related Work
Mean estimation in 1 dimension. Mean estimation, even in 1-dimension, has been studied algorithmically since the 1980s. The classic median-of-means estimator was the first big-O optimal sub-Gaussian mean estimator proposed in the literature, independently invented by different groups of authors (Alon et al., 1999;Nemirovsky & Yudin, 1983;Jerrum et al., 1986). Catoni's influential work (2012) gave the first sub-Gaussian mean estimator that yields the tight multiplicative constant in its error, but under strong assumptions that either the variance is known (to extremely high accuracy) or the distribution kurtosis (normalized 4 th moment) is bounded. Followup work by Devroye et al. (2016) studied "multiple-δ" estimators, also with sharp error constants, in the same setting. More recently, Lee & Valiant (2022) constructed a sub-Gaussian mean estimator with tight constants, under the bare minimum assumption that the variance exists, and absent any extra knowledge or moment assumptionsthis estimator is the subject of study in the current work.
See the survey of Lugosi & Mendelson (2021) on mean estimation results prior to 2019.
In low moment settings where the underlying distribution might have infinite variance, Bubeck et al. (2012) studied the performance of median-of-means. Devroye et al. (2016) then showed lower bounds that match up to constants. Our work shows that Lee and Valiant's estimator achieves analogous results as median-of-means in these regimes (Theorem 2.4), with sharper dependence on the z th moment, for every z ∈ (1, 2].
"Beyond worst-case analysis" of 1-d mean estimators is a new research topic of recent interest in the community. In the standard i.i.d. setting, Dang et al. (2023) characterized the optimal distribution-specific error rates up to constants, showing that median-of-means achieve such rates. Our work shows that in fact all estimators that simultaneously achieve (big-O) optimal sub-Gaussian and robust estimation error must also achieve the distribution-specific optimal error rates (Proposition 2.6). In addition to the standard i.i.d. setting, a different line of work has also studied distributionspecific mean estimation error rates for various differential privacy settings (Asi & Duchi, 2020a;b;Huang et al., 2021).
Robust mean estimation. Robust statistics, the setting where part of the input data can be corrupted by an adversary, has been an active area of research in the statistics community since the 1960s (Huber, 1992;Tukey, 1960). However, it was only in the past decade that polynomial-time algorithms for these statistical problems were found. See the textbook of Diakonikolas & Kane (2023) for a detailed introduction to these recent advances. Most directly relevant to our present work are results that give simultaneously sub-Gaussian and robust mean estimators, even in arbitrary high dimensions (Diakonikolas et al., 2020;Depersin & Lecué, 2022;Hopkins et al., 2020). Median-of-means is also known to be such a robust and sub-Gaussian estimator in 1-dimension -this is a folklore result, but see Laforgue et al. (2021) for more details on the analysis in the robust setting. Similarly, trimmed mean also has a folklore analysis for its robustness, and see Oliveira et al. (2025) for more results on the robustness and additional properties of trimmed mean.
this section cite: ['b0', 'b19', 'b14', 'b7', 'b16', 'b17', 'b3', 'b7', 'b5', 'b12', 'b13', 'b21', 'b8', 'b9', 'b6', 'b11', 'b15', 'b20']

Section: Outlier Robustness
In this section, we outline the proof of Theorem 2.2 (restated below for the reader's convenience), which says that Estimator 1 is robust against adversarial data contamination from the strong contamination model of Definition 2.1. The proof of Theorem 2.3 has analogous structure.
Theorem 2.2. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D. Suppose both log 1 δ n and δ are bounded by some small universal constant, and suppose η ≤ 1 24n log 1 δ . Then, with probability at least 1 -δ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ n + 222 √ η  
The proof strategy is to bound the difference between the estimates returned by Estimator 1 on (corrupted) samples X versus its behavior on uncorrupted samples X, fixing the confidence parameter δ.
Changing the input from uncorrupted samples to corrupted samples has two effects on the resulting estimate:
1. The α "influence parameter" (as computed in Step 2 of Estimator 1) may change. However, we show that in a certain sense, when the fraction of corruption η is small compared to log 1 δ /n, this corruption will not change the computed α value by much (Lemma 4.2). We further show that artificially changing the α value by a small amount will not change the mean estimate of Step 3 by much, with high probability, when given uncorrupted samples.
2. For a fixed influence parameter α, corrupting the samples from X to X changes the returned mean estimate. However, we show a (high probability) lower bound on the value α computed by the algorithm on input X (Lemma 4.3); and this lower bound on α gives us a natural upper bound on how much any corrupted input value can affect the final mean estimate.
We state here the two key structural lemmas (and a corresponding prerequisite definition) for the α value computed from corrupted samples X.
Definition 4.1. Let X = {x i } be a set of clean samples, and let X = {x i } be the corresponding set of η-corrupted samples. Denote by α ρ the "influence parameter" computed from the clean samples so as to satisfy a version of Step 2 but with a modified right hand side ( 1 3 log 1 δ + ρn instead of 1 3 log 1 δ ): i min(α ρ x 2 i , 1) = 1 3 log 1 δ + ρn and denote by αρ the corresponding "influence parameter" computed instead from the corrupted samples: i min(α ρ x2 i , 1) = 1 3 log 1 δ + ρn Lemma 4.2. Consider an arbitrary set of samples X and a new sample set X η-corrupted from X. Consider also an arbitrary input parameter δ. Using α to denote the influence parameter of Estimator 1 on inputs (δ, X), i.e. α0 in Definition 4.1, we have α -2η ≤ α ≤ α 2η Considering the right hand side of the condition in Step 2 of Estimator 1 as expressing the level of "desired robustness": Lemma 4.2 states that the modified influence parameter from η-corruption is always sandwiched between the uncorrupted influence parameters, but at slightly different levels of desired robustness. We point out that Lemma 4.2 is a deterministic lemma, that always holds, regardless of the sampling over X. Lemma 4.3. In the setting of Lemma 4.2, suppose both log 1 δ n
and δ are bounded by some small universal constant, and suppose η ≤ 1 24n log 1 δ . With probability at least 1 -4δ/11 over the sampling of n samples X from a distribution D with variance σ 2 , we have α ≥ 0.0008496η.
These lemmas let us bound α even when given corrupted data, and relate it to the uncorrupted α; these bounds are the crucial tools needed to bound the mean estimation error in both Theorem 2.2 and Theorem 2.3. Lemma 4.3 is used in the proof of Theorem 2.2, and shown inside Proposition A.22 in Appendix A.4. The proof of Theorem 2.3 has an analogous lemma with slightly different parameters.
The full analysis of the outlier robustness of Estimator 1 is given in Appendix A.
this section cite: []

Section: Neighborhood Optimality
Neighborhood optimality is a new notion of fine-grained distribution-dependent optimality recently proposed by Dang et al. (2023). While sub-Gaussian bounds are worstcase optimal for the class of finite variance distributions, neighborhood optimality captures the extent to which estimators can beneficially adapt to the non-Gaussianity of the underlying distribution and outperform the sub-Gaussian bound.
Before we formally state the definition of neighborhood optimality, let us give some preliminary definitions.
Let P 1 be the entire set of all distributions with a finite first moment over R. We say that N is a neighborhood function (defined over P 1 ) if N maps a distribution D ∈ P 1 to a set of distributions N (D) ⊆ P 1 . Intuitively, the neighborhood N (D) of D is a set of distributions that we expect an estimator to perform similarly well on (and we typically consider neighborhoods where D ∈ N (D) ). Similarly, an error function ϵ maps distributions to non-negative numbers, like the function introduced in Definition 2.5. In the later definitions, we use the notations N n,δ and ϵ n,δ to indicate their dependence on the sample complexity n and failure probability δ.
Given these two notions, we can now define the notion of a neighborhood Pareto bound, as a property that an error function satisfies. Essentially, the definition imposes admissibility/Pareto efficiency structure within the local neighborhood N n,δ (D) of every distribution D ∈ P 1 .
Definition 5.1 (Neighborhood Pareto Bounds (Dang et al., 2023)). Let n be the number of samples and δ be the failure probability. Given a neighborhood function N n,δ : P 1 → 2 P1 , we say that the error function ϵ n,δ (D) : P 1 → R + 0 is a neighborhood Pareto bound for P 1 with respect to N n,δ if for all distributions D ∈ P 1 , no estimator μ taking n i.i.d. samples can simultaneously achieve the following two conditions:
• For all D ′ ∈ N n,δ (D), with probability 1 -δ over the n i.i.d. samples from D ′ , we have |μ -µ D ′ | ≤ ϵ n,δ (D ′ ).
• With probability 1 -δ over the n i.i.d. samples from D, |μ -µ D | < ϵ n,δ (D).
Neighborhood Pareto ounds essentially play the role of "lower bounds" in an optimality definition, and the strength of the result depends crucially on the choice of the neighborhood function N under consideration. As a basic observation, the strength of this definition is monotonic in the size of the neighborhoods returned by N : if an error function ϵ is a neighborhood Pareto bound for a neighborhood function N , then for any neighborhood function N ′ such that N (D) ⊆ N ′ (D) for every D ∈ P 1 , ϵ is also a neighborhood Pareto bound for N ′ . Thus, the smaller each neighborhood is, the stronger the neighborhood Pareto bound is.
Finally, we define neighborhood optimality.
Definition 5.2 ((κ, τ )-Neighborhood Optimal Estimators (Dang et al., 2023)). Let κ > 1 be a multiplicative loss factor in estimation error, and τ > 1 be a multiplicative loss factor in sample complexity.
Given the parameters κ, τ > 1, sample complexity n, failure probability δ and neighborhood function N n,δ , a mean estimator μ is (κ, τ )-neighborhood optimal with respect to N n,δ if there exists an error function ϵ n,δ (D) such that min(ϵ n/τ,δ (D), ϵ n,δ (D)) is a neighborhood Pareto bound 1 , and μ gives estimation error at most κ • ϵ n,δ (D) with probability at least 1 -δ when taking n i.i.d. samples from any distribution D ∈ P 1 . Dang et al. (2023) showed that the error function ϵ n,δ from Definition 2.5 yields a neighborhood Pareto bound 1 κ min(ϵ n/τ,δ (D), ϵ n,δ (D)) for an appropriate choice of neighborhood function, for some constants κ, τ > 1. Their choice of neighborhood function N n,δ is technical; we state it in Appendix C, and refer the reader to their paper for the justification. Based on this result, they also showed that median-of-means indeed achieves error O(ϵ n,δ ) from Definition 2.5 and hence is neighborhood optimal by Definition 5.2. Dang et al. (2023) further raised the immediate question of whether other more-modern estimators, such as Lee and Valiant's Estimator 1, can also achieve such estimation error.
We show a general affirmative answer, that in fact, all estimators that are (up to constants) simultaneously sub-Gaussian 1 While it is intuitive to expect that an error function decreases in n, it might not be true in general. Indeed, the function used by Dang et al. (2023) (Definition 2.5) is not necessarily monotonic. This is the reason for the "min" in the neighborhood Pareto bound requirement.
and optimally robust to corruption must achieve the error rate from Definition 2.5 (stated formally as Proposition 2.6), and are thus neighborhood optimal as a corollary.
Proposition 2.6. Let μ be an arbitrary estimator that, when given δ > 0 and a set of n η-corrupted samples from any distribution D with mean µ and variance σ 2 , outputs a mean estimate satisfying
|μ -µ| ≤ O   σ   log 1 δ n + √ η    
with probability at least 1 -δ 2 over the randomness of the (uncorrupted) samples.
Then, the same estimator μ, on input n i.i.d. samples drawn from a distribution D with finite mean, will output an estimate with error upper bounded by O(ϵ n,δ (D)) (as defined in Definition 2.5) with probability at least 1 -δ.
The precondition of Proposition 2.6 is satisfied by a variant of Theorem 2.2-with slightly smaller failure probabilitywhich holds for Estimator 1. Thus the neighborhood optimality of Estimator 1 follows as a corollary of Proposition 2.6.
To see the intuition behind Proposition 2.6, recall the definition of the error rate function ϵ n,δ (D) from Definition 2.5. The definition constructs a distribution D * n,δ from D, by removing the tails of D with probability mass O(log 1 δ /n), and the error function
ϵ n,δ (D) = σ * n log 1 δ n + |µ -µ * n | is the sub-Gaussian error for distribution D *
n,δ plus the mean difference between D and D * n,δ . Thus, when given samples from D, one could view them as corrupted samples from D * n,δ where roughly O(log 1 δ /n) fraction of the samples are corrupted. A sub-Gaussian and robust estimator would thus achieve good error with respect to the mean of D * n,δ , and by triangle inequality, also with respect to the mean of D.
We present proofs of the above statements in Appendix C. For completeness, we also provide a summary of Dang et al. (2023)'s results. We again refer the reader to their paper for a more in-depth discussion on the intricacies of the neighborhood optimality notion.
this section cite: ['b5', 'b5', 'b5', 'b5', 'b5', 'b5']

Section: Infinite Variance Distributions
In this section, we extend Lee and Valiant's analysis of Estimator 1 to more heavy-tailed distributions. Instead of Fact 1.1, where the performance of the estimator is characterized in terms of the variance of the distribution D, we instead ask if we can characterize the performance of the estimator on distributions that may not have a finite variance but instead only have finite z th moment for some 1 < z ≤ 2.
We restate our main theorem for this section, which matches the lower bound of Devroye et al. (2016) up to constants.
Theorem 2.4. Given any distribution D with mean µ and z th moment M z for some z ∈ (1, 2), let X be a set of n i.i.d. samples from D. Then, with probability at least 1 -δ over the randomness of X, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ (M z ) 1 z • (1 + o(1)) c z log 1 δ n 1-1 z where c z = 2(5.6) 1 z-1 -1 .
Here, the o(1) term tends to 0 as log 1 δ n , δ → (0, 0), in a manner independent of D and independent of z.
At a high level, our analysis is a generalization of Lee and Valiant's analysis to the low-moment setting, which allows us to prove a guarantee that gracefully reduces to their main result (Fact 1.1, with the sharp constant of √ 2) as z → 2. Furthermore, our value of c z is smaller than the corresponding multiplicative constant in the analysis of median-of-means by Bubeck et al. (2012), across all values of z ∈ (1, 2].
Here we give an overview of our analysis.
Without loss of generality, from the shift-and-scale equivariance of Estimator 1, we assume the underlying distribution has mean 0 and z th moment M z = 1. The goal is to prove tailored Chernoff bounds for this estimator to show its concentration. Lee and Valiant's analysis in the finite variance setting provides two useful techniques to address obstacles described in the following subsections.
this section cite: ['b7', 'b3']

Section: Estimator 1 is a sum of dependent terms
Estimator 1 is a sum of dependent terms, due to the influence parameter α computed in Step 2 involving all the samples. This makes proving Chernoff bounds tricky, given that moment generating functions multiply only for sums of independent terms. Lee and Valiant's approach is to reduce (via a Lipschitz argument) to analyzing the case where the preliminary estimate κ from Step 1 is taken to be exactly equal to the true mean µ = 0, and crucially reformulate Estimator 1 as a "2-parameter ψ-estimator".
Definition 6.1 (Lee & Valiant 2022). Consider Estimator 1 but with Step 1 replaced with "κ = 0". The estimator can be equivalently expressed as follows:
1. Input: Failure probability δ, independent samples X =
x 1 , . . . , x n 2. Solve for the (unique) pair (μ, α) satisfying ψ µ = 0 and ψ α = 0, where the functions ψ µ , ψ α are defined as follows:
ψ µ (X, μ, α) = n i=1 (μ -x i (1 -min(αx 2 i , 1))) ; ψ α (X, μ, α) = n i=1 min(αx 2 i , 1) - 1 3n log 1 δ 3. Output: μ from the previous step.
This reformulation has the advantage that, for any fixed pair (μ, α), any linear combination of the ψ µ and ψ α functions is a sum of independent terms. The concentration of Estimator 1 is then reduced to proving Chernoff bounds for these linear combinations.
This reformulation and reduction is independent of the finite variance assumption, and therefore also applicable to the low moment setting that our work analyzes.
this section cite: ['b16']

Section: Proving Chernoff bounds over large distribution classes
Even in the finite variance setting, proving a Chernoff bound that applies for all distributions D with mean 0 and variance 1 is daunting, given how large the distribution class is compared to standard concentration bounds. Lee and Valiant showed that the worst-case Chernoff bound (for linear combinations of the ψ-equations from Definition 6.1) can in fact be viewed as a max-min linear programming game.
For simplicity, let us illustrate this by sketching the analysis of the Chernoff bound of a hypothetical linear estimator that is a sum of independent terms:
f ({x 1 , . . . , x n }) = 1 n i f (x i ), for some fixed function f : R → R.
Proving a Chernoff bound is equivalent to upper bounding the moment generating function of the estimator f , and choosing the "Chernoff parameter" t accordingly. Thus, it suffices to upper bound the objective of the following max-min game, where the max player chooses any mean-0 variance-1 distribution D-represented by variables {p x } denoting the probability mass at x (ignoring probability formalism issues with non-discrete distributions)-and the min player chooses the Chernoff parameter t. Using the moment generating function as the objective function, we have
max {px} min t x p x e t•f (x) such that x p x = 1 x p x • x = 0 x p x • x 2 = 1 where p x ≥ 0(1)
By using minimax duality and linear programming duality, the game can then be rewritten into a pure minimization program with the same optimum, where the dual variables U, M, V correspond to the 3 respective constraints in the program in (1).
min t min U,M,V U + V such that for all x ∈ R, V x 2 + M x + U ≥ e t•f (x)
(2) It thus suffices to choose dual variables U, M, V and an appropriate Chernoff parameter t to certify an upper bound on the optimum.
We modify this approach from Lee & Valiant (2022) by relying on a z th moment bound instead of a variance bound.
The key observation is that the z th moment bound may be expressed as a linear constraint in the above program, replacing x p x x 2 = 1 with x p x |x| z = 1. The technical challenge from here is to provide a feasible dual solution and choose Chernoff parameter t so as to satisfy the desired bounds, including that the guarantees converge to Fact 1.1 as z → 2. We show our complete proof in Appendix D.
this section cite: ['b16']

Section: Asymptotic Normality
In this section, we show that under the standard finite variance assumption, the estimator of Lee and Valiant is asymptotically normal and efficient. Specifically, we prove that if we fix the input parameter δ and take the number of samples n → ∞, the estimator converges to the sample mean in probability, which by the Central Limit Theorem implies asymptotic normality and efficiency. This result contrasts with median-of-means, which, under the slightly stronger 2 + ι moment assumption for any ι > 0, is asymptotically normal yet inefficient (Minsker, 2023)the asymptotic distribution of √ nμ MoM is N (µ, (π/2)σ 2 ) instead of the desired N (µ, σ 2 ).
Theorem 2.8. Let D be a distribution with mean µ and variance σ 2 .
Let μ denote Estimator 1 on input parameter δ and n i.i.d. samples from D. Also let Xn denote the sample mean. Then, fixing δ and D and taking n → ∞, we have
√ nμ p → √ n Xn that is, | √ nμ- √ n Xn | p → 0, that √ nμ converges to √ n Xn in probability.
As a corollary, by the Central Limit Theorem, we have
√ n(μ -µ) d → N 0, σ 2
That is, μ is asymptotically normal and efficient.
The proof is relatively straightforward. The key idea is that Estimator 1 differs from the sample mean by removing Θ(log 1 δ ) weighted samples, so we might as well bound the difference by Θ(log 1 δ ) times the maximum sample (and symmetrically the minimum sample), multiplied by a factor of √ n since that is the scale that the Central Limit Theorem holds at. Under the finite variance assumption, we can use a (slightly refined) Chebyshev's inequality and a standard Chernoff bound to upper bound the magnitude of the maximum sample with high probability. See the complete calculations in Appendix E.
this section cite: []

Section: A. Remaining Proofs of Section 4
In Section 4, we discussed the intuition behind our main results, Theorem 2.2 and 2.3. At a high level, our proof strategy for both main theorems uses the triangle inequality to bound estimation error introduced by adversarial corruption as the sum of two parts, one from changing the influence parameter from α to α, and one from the adversary arbitrarily corrupting the samples. We provide formal proofs for relevant lemmas and propositions in this section, and restate the main theorems for completeness:
Theorem 2.2. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D. Suppose both log 1 δ n and δ are bounded by some small universal constant, and suppose η ≤ 1 24n log 1 δ . Then, with probability at least 1 -δ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ n + 222 √ η   Theorem 2.3.
Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D.
Suppose both
log 1 δ n
and δ are bounded by some small universal constant, and suppose η
≤ 1 9n log 1 δ . Let δ ′ be so that 1 3 log 1 δ = 1 3 log 1 δ ′ + 3ηn.
Then, with probability at least 1 -δ ′ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ ′ n + (135 + o(1)) √ η  
Throughout the proofs, we will compare and make use of different values of "α", computed from either corrupted or uncorrupted data, and computed from different choices of parameters in the equation defining α. Here, we give a more general definition of notation (generalizing Definition 4.1) that we will be using.
Definition A.1. Let X = {x i } be a set of clean samples, and let X = {x i } be the corresponding set of η-corrupted samples. Denote by α δ,ρ the "influence parameter" solved from the corresponding condition involving the clean samples, so as to satisfy a version of Step 2 of Estimator 1 but with a modified right hand side
( 1 3 log 1 δ + ρn instead of 1 3 log 1 δ ): i min(α δ,ρ x 2 i , 1) = 1 3 log 1 δ + ρn
and denote by αδ,ρ the "influence parameter" solved instead from the corresponding condition involving the corrupted samples:
i min(α δ,ρ x2 i , 1) = 1 3 log 1 δ + ρn
Theorem 2.3 refers to failure probability δ ′ , so parts of the analysis will involve the notations α δ ′ ,η and αδ ′ ,η , for example.
We start with showing the crucial preliminary Lemma A.2, which bounds the value of α, allowing us to approximate it using different influence parameters on the clean data, with no assumption of the corrupted data:
α δ,-2η ≤ α ≤ α δ,2η
Proof. Let the clean samples be X = {x i } and the corrupted samples be X = {x i }.
To prove the first inequality, suppose for the sake of contradiction that α < α δ,-2η . Then, i min(αx 2 i , 1)
≤ i min(αx 2 i , 1) + i : xi corrupted min(αx 2 i , 1) ≤ 1 3 log 1 δ -2ηn + i : xi corrupted min(αx 2 i , 1) (since α < α δ,-2η ) ≤ 1 3 log 1 δ -ηn(
since the sum has ηn elements, each at most 1)
< 1 3 log 1 δ
which is a contradiction.
The second inequality follows similarly. Suppose for the sake of contradiction that α > α δ,2η . Then,
i min(αx 2 i , 1) ≥ i min(αx 2 i , 1) - i : xi corrupted | min(αx 2 i , 1) -min(αx 2 i , 1)| ≥ 1 3 log 1 δ + 2ηn - i : xi corrupted | min(αx 2 i , 1) -min(αx 2 i , 1)| (since α > α δ,2η ) ≥ 1 3 log 1 δ + ηn (since the sum has ηn elements, each at most 1) > 1 3 log 1 δ
which is a contradiction.
We in fact generalize the above lemma slightly for use in the proof of Theorem 2.3, which can be proven from Lemma A.2 by reparameterizing δ. Corollary A.3. For any set of clean samples X and the corresponding η-corrupted samples X, and for any constant c > 2, we have α δ,(c-2)η ≤ αδ,cη ≤ α δ,(c+2)η .
The proof structure of Theorem 2.2 and 2.3 are essentially identical. We present the proof of Theorem 2.3 first.
For the rest of the appendix, we will assume that the uncorrupted data distribution has mean 0 and variance 1 without loss of generality, due to the shift-and-scale equivariance of Estimator 1.
this section cite: []

Section: A.1. Bounding the Error due to Changing the Influence Parameter
We present the following proposition upper bounding the error incurred on Estimator 1 by using influence parameter α := αδ ′ ,3η instead of α := α δ ′ ,0 on the clean samples. That is, we compute α on the corrupted samples, but analyze its effect on the clean samples.
For the following section, recall our assumption that the underlying distribution has mean 0 and variance 1.
Proposition A.4. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Let α be the influence parameter computed from the clean samples with robustness level 1 3 log 1 δ ′ , namely α := α δ ′ ,0 . Let α be the influence parameter computed from the corrupted samples with robustness level 1 3 log 1 δ ′ + 3ηn, namely, α := αδ ′ ,3η . Then with probability at least 1 -6 8 δ ′ , the mean estimate using α on the clean samples differs from the mean estimate using α on the clean samples by at most 125.5 √ η, i.e.,
i x i min(αx 2 i , 1) - i x i min(αx 2 i , 1) ≤ 125.5n √ η(3)
To provide some intuition towards our proof strategy for Proposition A.4, first notice that we can bound the left hand side via Cauchy-Schwarz as
i x 2 i i (min(αx 2 i , 1) -min(αx 2 i , 1)) 2
This turns out to be insufficient; we instead bound (3) by defining the set S of indices where min(αx 2 i , 1) ̸ = min(αx 2 i , 1), and restrict the range of both sums in (3) to the range i ∈ S, since doing so only discards zero terms and does not change the sum. Thus we instead have the Cauchy-Schwarz bound i∈S
x 2 i i∈S (min(αx 2 i , 1) -min(αx 2 i , 1)) 2
The bound on the second parenthetical makes crucial use of the comparison of α and α provided by Corollary A.3. The first parenthetical is an empirical variance, but the restriction i ∈ S means that |x i | cannot be too large; we thus use Bernstein's inequality to bound this S-truncated empirical second moment, in terms of a lower bound on α, which we prove next.
To show our lower bound on α, we first calculate the following straightforward relations between the empirical and population quantiles.
Throughout this section, we denote by Q q (D) the q (true) quantile of D, i.e., P[D ≤ Q q (D)] = q.
Lemma A.5. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Denote c 1 = 0.277 < 1 3 , and
κ := c 1 ( 1 n log 1 δ ′ ).
Let constant c 2 := 102.907. Then the 1 -κ empirical quantile of x i is at most Q 1-κ/c2 (D) with probability at least 1 -1 8 δ ′ .
Proof. For the 1 -κ empirical quantile of x i to be greater than Q 1-κ/c2 (D), there has to be more than κn samples greater than Q 1-κ/c2 (D). Thus, it suffices to prove that |{i ∈ [n] :
x i ≥ Q 1-κ/c2 (D)}| ≥ κn with probability at most 1 8 δ ′ . Denote Z i := 1 xi≥Q 1-κ/c 2 (D) . Then Z := |{i ∈ [n] : x i ≥ Q 1-κ/c2 (D)}| = n i=1 Z i . We denote by p = κ
c2 the probability that an individual i is in this set; and thus E[Z] = pn. Since each Z i is a coin flip of probability p, we further have that V ar[Z i ] = p(1 -p).
this section cite: []

Section: By multiplicative Chernoff,
P[Z ≥ c 2 pn] ≤ e c2-1 c c2 2 pn = exp ((c 2 -1 -c 2 log c 2 )pn) = exp c 1 (c 2 -1 -c 2 log c 2 ) c 2 log 1 δ ≤ exp (-1.01)log 1 δ ′ by choice of c 1 and c 2 ≤ exp -log 8 δ ′ = 1 8 δ ′ since 1.01log 1 δ ′ ≥ log 8 δ ′ for suff. small δ ′
as desired.
By symmetry, we have the following corollary as well:
Corollary A.6. The κ empirical quantile of x i is at least Q κ/c2 (D) with probability at least 1 -1 8 δ ′ . Lemma A.7. Let D trimmed denote a "trimmed" version of D: namely, D conditioned on lying in [Q κ/c2 (D), Q 1-κ/c2 (D)].
Then E[D 2 trimmed ] ≤ c 2 2 (c2-2κ) 2 .
Proof. Denote by 1 untrimmed (x) the indicator that returns 1 if x ∈ [Q κ/c2 (D) , Q 1-κ/c2 (D)] and 0 otherwise. Then
observe that D trimmed = c2 c2-2κ (D • 1 untrimmed ). Thus E[D 2 trimmed ] = E c 2 c 2 -2κ (D • 1 untrimmed ) 2 = c 2 2 (c 2 -2κ) 2 E[D 2 1 untrimmed ] ≤ c 2 2 (c 2 -2κ) 2 E[D 2 ] = c 2 2 (c 2 -2κ) 2
as desired.
Lemma A.8. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Let (x trimmed ) i denote the sample Proof. First notice that if we replace any trimmed x i with a random sample according to D trimmed , then the sum i (x trimmed ) 2 i can only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from D trimmed . With an abuse of notation, we let {(x trimmed ) i } i≤n denote a set of such samples.
x i after trimming, namely: let (x trimmed ) i = x i if x i ∈ [Q κ/c2 (D), Q 1-κ/c2 (D)],
Also notice that by our choice of c 3 , we have, crucially,
3c1c 2 3 (6+2c3)c2 ≥ 1.01.
We start by bounding Q κ/c2 (D) and Q 1-κ/c2 (D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality,
P |D| ≥ c2 κ ≤ κ c2 , which implies that Q κ/c2 ≥ -c2 κ and Q 1-κ/c2 ≤ c2 κ . As a result, (x trimmed ) 2 i ≤ c2 κ for all i.
Then,
Var[(x trimmed ) 2 i ] ≤ E ((x trimmed ) 2 i ) 2 ≤ c 2 κ E[(x trimmed ) 2 i ] ≤ c 3 2 κ(c 2 -2κ) 2 by Lemma A.7
Thus, by Bernstein's inequality,
P i ((x trimmed ) i ) 2 ≥ (c 3 + 1)c 2 2 n (c 2 -2κ) 2 = P 1 n i ((x trimmed ) i ) 2 - c 2 2 (c 2 -2κ) 2 ≥ c 3 c 2 2 (c 2 -2κ) 2 ≤ P 1 n i ((x trimmed ) i ) 2 -E[((x trimmed ) i ) 2 ] ≥ c 3 c 2 2 (c 2 -2κ) 2 ≤ 2 exp   - c 2 3 n c 4 2 (c2-2κ) 4 2 Var[(x trimmed ) 2 i ] + 2c3c 3 2 3κ(c2-2κ) 2   ≤ 2 exp   - c 2 3 n c 4 2 (c2-2κ) 4 2c 3 2 κ(c2-2κ) 2 + 2c3c 3 2 3κ(c2-2κ) 2   = 2 exp - 3c 2 3 c 2 nκ (6 + 2c 3 )(c 2 -2κ) 2 ≤ exp - 3c 2 3 nκ (6 + 2c 3 )c 2 = 2 exp(- 3c 1 c 2 3 (6 + 2c 3 )c 2 log 1 δ ′ ) by definition of κ ≤ 2 exp(-1.01log 1 δ ′ ) by choice of c 3 ≤ 2 exp(-log 8 δ ′ ) since 1.01log 1 δ ≥ log 8 δ ′ for suff. small δ ′ = 2 8 δ ′
as desired.
Combining Lemma A.5, Corollary A.6, and Lemma A.8, we have the following corollary:
Corollary A.9. Suppose there is a sufficiently small constant that upper bounds δ ′ . Let S <κ denote the set of indices i s.t.
x 2 i is not in the top κ (empirical) quantile. Then i∈S<κ x 2 i ≤ (c3+1)c 2 2 (c2-2κ) 2 n with probability at least 1 -4 8 δ ′ .
We are now ready to present a lower bound on α := α δ ′ ,0 , before proving Proposition A.4.
Lemma A.10. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Then α := α δ ′ ,0 ≥ 0.000214 1 n log 1 δ ′ with probability at least 1 -4 8 δ ′ .
Proof. Recall that by definition of α,
1 3 log 1 δ ′ = i min(αx 2 i ,1
) = i:x 2 i not in the top κ quantile min(αx 2 i , 1) + i:x 2 i in the top κ quantile min(αx 2 i , 1) ≤ i:x 2 i not in the top κ quantile αx 2 i + i:x 2 i in the top κ quantile 1 ≤ i:x 2 i not in the top κ quantile αx 2 i + κn = i:x 2 i not in the top κ quantile
αx 2 i + c 1 log 1 δ ′
Rearranging, this is equivalent to
1 3 -c 1 log 1 δ ′ ≤ i:x 2 i not in the top κ quantile αx 2 i
By Corollary A.9, with probability at least 1 -4 8 δ ′ , i:x 2 i not in the top κ quantile
x 2 i ≤ (c 3 + 1)c 2 2 (c 2 -2κ) 2 n ≤ (c 3 + 1)c 2 2 (c 2 -2) 2 n Which implies that α ≥ 1 3 -c 1 (c2-2) 2 (c3+1)c 2 2 1 n log 1 δ ′ ≥ 0.000214 1 n log 1 δ ′ with probability at least 1 -4 8 δ ′ .
Proof of Proposition A.4. First, notice that for all i such that |x i | ≥ 1 α ≥ 1 α , the corresponding term in the sum in the left hand side becomes 0. Thus, using the notation ≤ to denote summing over elements |x i | ≤ 1 α , the left hand side in the guarantee of Proposition A.4 is equal to
≤ x i min(αx 2 i , 1) - ≤ x i min(αx 2 i , 1)
Then, rearranging the sums, we have
≤ x i min(αx 2 i , 1) - ≤ x i min(αx 2 i , 1) ≤ ≤ x i min(αx 2 i , 1) -min(αx 2 i , 1) ≤   ≤ x 2 i     ≤ (min(αx 2 i , 1) -min(αx 2 i ,1
)) 2   by Cauchy-Schwarz for which we can bound the two terms separately. To bound the first term, since we sum over only those terms where |x i | ≤ 1 α for all i, and by Lemma A.10 α ≥ 0.000214 n log 1 δ ′ with probability 1 -4 8 δ ′ , we have that x 2 i ≤ 4672n log 1 δ ′ for all i. Since X has mean 0 and variance 1, we know that E[x 2 i ] ≤ E[X 2 ] = 1, and Var[x
2 i ] ≤ E[x 4 i ] ≤ 4672n log 1 δ ′ E[x 2 i ] ≤ 4672n log 1 δ ′
. Thus, by Bernstein's inequality,
P   ≤ x 2 i ≥ 3150n   = P   1 n ≤ x 2 i -1 ≥ 3149   ≤ P   1 n ≤ x 2 i -E[x 2 i ] ≥ 3149   ≤ 2 exp - 3149 2 n 9344 n log 1 δ ′ + 9344 • 3149 n log 1 δ ′ /3 ≤ 2 exp -1.01log 1 δ ′ ≤ 2 exp -log 8 δ ′ since 1.01log 1 δ ′ ≥ log 8 δ ′ for suff. small δ ′ = 2 8 δ ′
In other words, conditioning on Lemma A.10 holding, with probability at least 1
-2 8 δ ′ , ≤ x 2 i ≤ 3150n. To bound the second term, note that ≤ min(αx 2 i , 1) -min(αx 2 i , 1) 2 ≤ i min(αx 2 i , 1) -min(αx 2 i , 1) 2 ≤ i min(αx 2 i , 1) -min(αx 2 i , 1)
since α := αδ ′ ,3η ≥ α δ ′ ,η ≥ α := α δ ′ ,0 by Corollary A.3, and "α" is monotonic in the η argument, and thus 0 ≤ min(αx 2 i , 1) -min(αx 2 i , 1) ≤ 1 for all i. To further upper bound the last quantity, we have
i min(αx 2 i , 1) -min(αx 2 i , 1) = i min(αx 2 i , 1) - i min(αx 2 i , 1) ≤ i min(α δ ′ ,5η x 2 i , 1) - i min(α δ ′ ,0 x 2 i , 1) by Corollary A.3 and by definition of α = 1 3 log 1 δ ′ + 5ηn - 1 3 log 1 δ ′ = 5ηn
Finally, summarizing, we have that with probability at least 1 -6 8 δ ′ :
  ≤ x 2 i     ≤ (min(αx 2 i , 1) -min(αx 2 i , 1)) 2   ≤ 3150n • 5ηn = n 15750η ≤ 125.5n √ η
as desired.
this section cite: []

Section: A.2. Bounding the Error due to Corrupting the Samples
We now present the following proposition upper bounding the error incurred on Estimator 1 by the arbitrary corruption of the adversary on the clean samples, while fixing α := αδ ′ ,3η as the influence parameter.
We again assume that the uncorrupted distribution has mean 0 and variance 1.
Proposition A.11. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Let α be the influence parameter computed from the corrupted samples with robustness level 1 3 log 1 δ ′ + 3ηn, namely, α := αδ ′ ,3η . Then with probability at least 1 -1 8 δ ′ , the mean estimate using α on the clean samples differs from the mean estimate using α on the corrupted samples by at most 8.586 √ η, i.e.,
i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 8.586n √ η
To provide some intuition towards our proof strategy for Proposition A.11, consider the adversary's arbitrary corruption, which can be interpreted piece-wise as moving each clean sample that the adversary wishes to corrupt to a new location. Since the influence parameter controls the contribution of each sample to the mean estimate, based on how far from the mean it is, moving a sample too far from the mean or moving the sample too close to the mean will both incur very little error. The question then, is, what is the maximum estimation error the adversary can incur by corrupting a single sample?
Fixing the value of α as in the statement of Proposition A.11, we can upper bound the maximum magnitude of the expression
1 n i x i (1 -min(αx 2 i , 1)) by calculus, which is O(1/ √ α).
We will show a lower bound of α, specifically that α ≥ Ω(η), and then conclude that the maximum total error possible by corrupting ηn samples is at most
1 n • ηn • Θ(1/ √ η) = Θ( √ η),
as desired.
We use a similar strategy as in the proof of Proposition A.4, using quantile statistics as well as Corollary A.3 to obtain our desired lower bound on α, before proving Proposition A.11 in Appendix A.
Lemma A.12. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Denote c 1 = 1 3 , and κ := c 1 ( 1 n log 1 δ ′ ). Let c 2 := 55.252. Then the 1 -κ empirical quantile of x i is at most Q 1-κ/c2 (D) with probability at least 1 -1 32 δ ′ .
Proof. For the 1 -κ empirical quantile of x i to be greater than Q 1-κ/c2 (D), there has to be more than κn samples greater than Q 1-κ/c2 (D). Thus, it suffices to prove that |{i ∈ [n] :
x i ≥ Q 1-κ/c2 (D)}| ≥ κn with probability at most 1 32 δ ′ . Denote Z i := 1 xi≥Q 1-κ/c 2 (D) . Then Z := |{i ∈ [n] : x i ≥ Q 1-κ/c2 (D)}| = n i=1 Z i . Obviously E[Z] = κn/c 2 . Denote p := E[Z i ] = κ/c 2 . Then V ar[Z i ] = p(1 -p).
this section cite: []

Section: By multiplicative Chernoff,
P[Z ≥ c 2 pn] ≤ e c2-1 c c2 2 pn = exp ((c 2 -1 -c 2 log c 2 )pn) = exp c 1 (c 2 -1 -c 2 log c 2 ) c 2 (log 1 δ ′ ) ≤ exp (-1.01)(log 1 δ ′ ) by choice of c 1 , c 2 ≤ exp -log 32 δ ′ = 1 32 δ ′ since 1.01log 1 δ ′ ≥ log 32 δ ′ for suff. small δ ′
as desired.
By symmetry, we have the following corollary as well:
Corollary A.13. The κ empirical quantile of x i is at least Q κ/c2 (D) with probability at least 1
-1 32 δ ′ . Lemma A.14. Let D trimmed denote the trimmed version of D conditioned on lying in [Q κ/c2 (D), Q 1-κ/c2 (D)]. Then E[D 2 trimmed ] ≤ c 2 2 (c2-2κ) 2 . Proof. Denote by 1 untrimmed (x) the indicator measure that maps x to 1 Q κ/c 2 (D)≤x≤Q 1-κ/c 2 (D) , the indicator of whether x is untrimmed. Then observe that D trimmed = c2 c2-2κ (D • 1 untrimmed ). Thus E[D 2 trimmed ] = E c 2 c 2 -2κ (D • 1 untrimmed ) 2 = c 2 2 (c 2 -2κ) 2 E[D 2 1 untrimmed ] ≤ c 2 2 (c 2 -2κ) 2 E[D 2 ] = c 2 2 (c 2 -2κ) 2
as desired.
Lemma A.15. Suppose both
log 1 δ ′ n
and δ ′ are bounded by some small universal constant. Let (x trimmed ) i denote the sample x i after trimming, such that (x trimmed Proof. First notice that if we replace any trimmed x i with a random sample according to D trimmed , then the sum i (x trimmed ) 2 i can only increase. Thus, to prove the claim, it suffices to bound the sum of n i.i.d. samples from D trimmed . With an abuse of notation, we let {(x trimmed ) i } i≤n denote a set of such samples.
) i = x i if x i ∈ [Q κ/c2 (D), Q 1-κ/c2 (D)],
Also notice that by our choice of c 3 , we have, crucially,
3c 2 3 c1(
6+2c3)c2 ≥ 1.01. We start by bounding Q κ/c2 (D) and Q 1-κ/c2 (D). Since we assume D has mean 0 and variance 1, by Chebyshev's inequality, P |D| ≥ c2 κ ≤ κ c2 , which implies Q κ/c2 ≥ -c2 κ and Q 1-κ/c2 ≤ c2 κ . As a result, (x trimmed ) 2 i ≤ c2 κ for all i. Then, Var[(x trimmed ) 2 i ] ≤ E ((x trimmed ) 2 i ) 2 ≤ c 2 κ E[(x trimmed ) 2 i ] ≤ c 3 2 κ(c 2 -2κ) 2 by Lemma A.14 Thus, by Bernstein's inequality,
P i ((x trimmed ) i ) 2 ≥ (c 3 + 1)c 2 2 n (c 2 -2κ) 2 = P 1 n i ((x trimmed ) i ) 2 - c 2 2 (c 2 -2κ) 2 ≥ c 3 c 2 2 (c 2 -2κ) 2 ≤ P 1 n i ((x trimmed ) i ) 2 -E[((x trimmed ) i ) 2 ] ≥ c 3 c 2 2 (c 2 -2κ) 2 ≤ 2 exp   - c 2 3 n c 4 2 (c2-2κ) 4 2 Var[(x trimmed ) 2 i ] + 2c3c 3 2 3κ(c2-2κ) 2   ≤ 2 exp   - c 2 3 n c 4 2 (c2-2κ) 4 2c 3 2 κ(c2-2κ) 2 + 2c3c 3 2 3κ(c2-2κ) 2   = 2 exp - 3c 2 3 c 2 nκ (6 + 2c 3 )(c 2 -2κ) 2 ≤ 2 exp - 3c 2 3 nκ (6 + 2c 3 )c 2 = 2 exp(- 3c 2 3 c 1 (6 + 2c 3 )c 2 (log 1 δ ′ ))by
definition of κ ≤ 2 exp(-1.01(log 1 δ ′ )) by choice of c 3 ≤ 2 exp(-(log 32 δ ′ )) since 1.01log 1 δ ′ ≥ log 32 δ ′ for suff. small δ ′ = 2 32 δ ′ as desired. Combining Lemma A.12, Corollary A.13, and Lemma A.15, we have the following corollary: Corollary A.16. Suppose both log 1 δ ′ n and δ ′ are bounded by some small universal constant. Let S <κ denote the set of indices i s.t. x 2 i is not in the top κ (empirical) quantile. Then i∈S<κ x 2 i ≤ (c3+1)c 2 2 (c2-2κ) 2 n with probability at least 1 -1 8 δ ′ .
Proof of Proposition A.11. We start by consider a single sample xi which is corrupted such that x i ̸ = xi . Fixing all other samples, we solve for the maximum (signed) error, or equivalently mean shift, that the adversary can incur by arbitrarily corrupting this sample x i only.
Recall that the estimator (fixing influence parameter α := αδ ′ ,3η ) is defined as μ = i x i (1 -min(αx 2 i , 1)). Taking the derivative with respect to x i , restricted to the region in which min(αx 2 i , 1) = αx 2 i , we have
∂ ∂x i μ = 1 -3αx 2 i = 0 ⇒ x = ± 1 √ 3 α
Since α ≥ 0, a local maximum occurs at x = 1 √ 3 α . The corresponding maximum contribution of this single term is 2 3 √ 3 α . It is easy to verify that this local maximum is in fact the global maximum based on taking the minimization between αx 2 i and 1. Symmetrically, the minimal contribution of a single term is -2 3 √ 3 α .
Thus, the maximum error the adversary can incur by corrupting a single term is 4 3 √ 3 α . The maximum error the adversary can incur by corrupting ηn terms is 4ηn 3 √ 3 α .
To upper bound 4ηn 3 √ 3 α , we need to lower bound α. Towards this, notice that α ≥ α δ ′ ,η by Corollary A.3, which by definition satisfies:
ηn + 1 3 log 1 δ ′ = i min(α δ ′ ,η x 2 i ,1
) ≤ i min(αx 2 i , 1) by Corollary A.3 ≤ i:x 2 i not in the top κ quantile min(αx 2 i , 1) + i:x 2 i in the top κ quantile min(αx 2 i , 1) ≤ i:x 2 i not in the top κ quantile αx 2 i + i:x 2 i in the top κ quantile 1 = i:x 2 i not in the top κ quantile αx 2 i + κn = i:x 2 i not in the top κ quantile αx 2 i + 1 3 log 1 δ ′ Rearranging, this is equivalent to ηn ≤ i:x 2 i not in the top κ quantile αx 2 i By Corollary A.16, with probability at least 1 -1 8 δ ′ , i:x 2 i not in the top κ quantile x 2 i ≤ (c 3 + 1)c 2 2 (c 2 -2κ) 2 n ≤ (c 3 + 1)c 2 2 (c 2 -2) 2 n Thus, ηn ≤ i:x 2 i not in the top κ quantile
αx 2 i ≤ (c 3 + 1)c 2 2 (c 2 -2) 2 nα which is equivalent to α ≥ (c 2 -2) 2 (c 3 + 1)c 2 2 η ≥ 0.00804η
Thus, we have
i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 4ηn 3 √ 3α ≤ 4ηn 3 √ 0.02412η ≤ 8.586n √ η
as desired.
this section cite: []

Section: A.3. Proof of Theorem 2.3
Equipped with Propositions A.4 and A.11, we are ready to formally prove Theorem 2.3.
Proof of Theorem 2.3. We will start by assuming that the underlying distribution has mean 0 and variance 1 (by the shiftand-scale equivariance of Estimator 1), and furthermore, that the initial estimate κ computed in Step 1 is exactly the true mean of 0. At the end of this proof, we will show that the κ = 0 assumption introduces only a negligible amount of mean estimation error.
Fixing the initial estimate κ = 0, then, Estimator 1 on input the clean samples X and confidence parameter δ ′ computes the influence parameter α := α δ ′ ,0 , and returns an estimate μclean,α := 1 n i x i (1 -min(αx 2 i , 1)). As a part of their analysis, Lee & Valiant (2022) showed that
|μ clean,α | ≤ ( √ 2 + o(1)) log 1 δ ′
n with probability at least 1 -δ ′ . We note that the above guarantee can in fact be slightly strengthened, so that the failure probability becomes δ ′ /16, at the expense of a slightly increased o(1) by no more than a constant multiplicative factor. Now we consider changing the "α" value from α to α computed from the corrupted samples, but applying these α influence parameters on the same clean sample set, and bound the difference in the resulting mean estimates. Specifically, consider α := αδ ′ ,3η , which is the influence parameter computed with robustness level 1 3 log 1 δ ′ + 3ηn from the corrupted samples. Let the mean estimate of using α on the clean samples be μclean,α := 1 n i x i (1 -min(αx 2 i , 1)). Then, by Proposition A.4, with probability at least 1 -6 8 δ ′ we have
|μ clean, α -μclean,α | = 1 n i x i min(αx 2 i , 1) - i x i min(αx 2 i , 1) ≤ 125.5 √ η
Next we consider the effect of replacing the effect of corruption, but after fixing the influence parameter to be α. Specifically, define μcorrupt,α to be the mean estimate of the corrupted samples using α, namely
1 n i xi (1 -min(αx 2 i ,1
)). Then, by Proposition A.11, with probability at least 1 -1 8 δ ′ , we have
|μ corrupt, α -μclean,α | = 1 n i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 8.586 √ η
By union bound and triangle inequality, summing over all three error terms, we have that with probability at least 1 -15 16 δ ′ , μcorrupt,α satisfies
|μ corrupt, α| ≤ ( √ 2 + o(1)) log 1 δ ′ n + 134.086 √ η ≤ ( √ 2 + o(1)) log 1 δ ′ n + 135 √ η
Finally, observe that on a mean-0-variance-1 distribution, and with η-corruption, the only difference between μcorrupt,α and the output μ of Estimator 1 (on input as stated in the theorem statement) lies in μcorrupt,α assuming that the initial mean estimate κ was the true mean of 0. On the other hand, μ (Estimator 1) uses an estimator (for example, median-of-means) to compute κ.
We use the following fact shown in Lee & Valiant (2022) about the structural properties of Estimator 1, to analyze the effect of the κ assumption:
Fact A.17 (Lee & Valiant 2022). Let X be a fixed set of samples of size n, and let δ > 0 be a confidence parameter. Let μ = μ(X, δ) denote the output of Estimator 1 of LV22. Then Estimator 1 is affine invariant, i.e.:
μ(aX + b, δ) = aμ(X, δ) + b
Additionally, let μκ (X, δ) denote the output of Estimator 1 but where Step 1 is omitted, and the initial estimate κ is instead considered as an input. Then:
∂ μκ (X, δ) ∂κ = O   log 1 δ n  
We also use the following fact/assumption on the robustness of the initial estimate κ against adversarial corruption: it is a folklore result that median-of-means satisfies this. A proof of Fact A.18 is given in Appendix F for completeness, since we are unaware of literature explicitly writing down this proof. As discussed in Estimator 1, we are free to use other estimators to compute κ as long as Fact A.18 holds analogously.
Fact A.18 (Folklore). For any distribution D with mean µ and standard deviation σ, let X be a set of n η-corrupted samples from D. The median-of-means estimate κ from grouping samples into O(log 1 δ ′ + ηn) buckets, on input X, satisfies
P   |κ -µ| ≥ O   σ log 1 δ ′ n + η     ≤ 1 16 δ ′ By Fact A.18, with probability at least 1 -1 16 δ ′ , |κ| ≤ O log 1 δ ′ n
+ η , and by the Lipschitz bound of Fact A.17, we can bound the (absolute) difference between μ and μcorrupt,α by
|μ -μcorrupt,α | ≤ O   |κ| log 1 δ ′ n   ≤ O   log 1 δ ′ n 2 + η log 1 δ ′ n   ≤ o   log 1 δ ′ n   + o( √ η)
Thus, on a mean-0-variance 1 distribution, with probability at least 1 -δ ′ over the clean samples (and with an arbitrary η-corruption), Estimator 1 outputs
|μ| ≤ ( √ 2 + o(1)) log 1 δ ′ n + (135 + o(1)) √ η
Combined with the affine invariance of Estimator 1 as stated in Fact A.17, if the underlying distribution instead had mean µ and variance σ 2 , we have the mean estimation guarantee
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ ′ n + (135 + o(1)) √ η   as desired.
We note that the o(1) √ η term in the estimation error is solely incurred by the estimation error from the initial estimate κ.
this section cite: ['b16', 'b16', 'b16']

Section: A.4. Proof of Theorem 2.2
In this section, we present the proof of Theorem 2.2. While the high level idea of our proof is similar to that of Theorem 2.3, there are some important distinctions, most importantly in what we require the failure probability of our component lemmas to be.
Recall that in Theorem 2.3, while we gave the parameter δ to Estimator 1, we relaxed the failure probability of the estimator to δ ′ ≥ δ. On the other hand, Theorem 2.2 actually analyzes Estimator 1 at failure probability δ, which in turn requires the prerequisite lemmas to have failure probability ≤ δ.
Towards that end, notice that given the desired failure probability δ, if we let δ ′ be such that 1 3 log 1 δ = 1 3 log 1 δ ′ + 8ηn, then δ = δ ′ e -24ηn . Thus, the goal of this section is to devise analogs of Proposition A.4 and A.11 with 1 -δ = 1 -δ ′ e -24ηn as failure probability instead. We stress that this reparameterization is purely for analytical purposes; Neither the estimator nor the user know anything about the corruption parameter η or the analytical assumption we use to define δ ′ . Our choice of the constant 8 in front of the ηn term is different from the previous section, and stems from a somewhat arbitrary numerical choice to prevent constants in the theorem from blowing up, while also implicitly posing constraints on η.
We present the counterpart of Proposition A.4 first, bounding the error coming from the change in α value on uncorrupted samples, with a smaller failure probability.
Proposition A.19. Suppose both log 1 δ n and δ are bounded by some small universal constant. Let α be the influence parameter computed from the clean samples with robustness level 1 3 log 1 δ ′ + 8ηn, namely, α := α δ ′ ,8η . Let α be the influence parameter computed from the corrupted samples with robustness level 1 3 log 1 δ ′ + 8ηn, namely, α := αδ ′ ,8η . Then with probability at least 1 -10 11 δ ′ e -24ηn , the mean estimate using α on the clean samples differs from the mean estimate using α on the clean samples by at most 195.065 √ η, i.e.,
i x i min(αx 2 i , 1) - i x i min(αx 2 i , 1) ≤ 195.065n √ η
this section cite: []

Section: We begin proving Proposition A.19 by stating an alternative version of Lemma A.10:
Lemma A.20. Suppose both
log 1 δ n
and δ are bounded by some small universal constant. Then α := α δ ′ ,8η ≥ 0.000214 1 n (log 1 δ ′ + 24ηn) with probability at least 1
-4 11 δ ′ • e -24ηn .
The proof is identical to that of Lemma A.10, with the appropriate log 1 δ and δ terms replaced with log 1 δ ′ + 24ηn and δ ′ • e -24ηn instead.
Now that we have lower bounded α (with high probability), analogously to the proof of Proposition A.4, we now wish to also lower bound α. The proof of this lower bound will deviate from that in Proposition A.4-in Proposition A.4 we were analyzing and comparing α := αδ ′ ,3η and α := α δ ′ ,0 ; here in Proposition A.19, we are comparing α := αδ ′ ,8η against α := α δ ′ ,8η .
For Proposition A.4, the "η-subscripts" between the two "α" values differ by 3η, so we could apply Corollary A.3 and the monotonicity of α δ ′ ,η in the "η-subscript" to yield α := αδ ′ ,3η ≥ α δ ′ ,η ≥ α := α δ ′ ,0 . Here, we need a new argument to lower bound α := αδ ′ ,8η , shown in the following lemma.
Lemma A.21. Suppose both
log 1 δ n and δ are bounded by some small universal constant. Then α := αδ ′ ,8η ≥ 0.0000354 1 n (log 1 δ ′ + 24ηn) with probability at least 1 -4 11 δ ′ • e -24ηn . Proof. Notice that α := αδ ′ ,8η ≥ α δ ′ ,6η by Corollary A.3. Letting κ := c 1 1 n (log 1 δ ′ + 24ηn) for some constant c 1 < 1 4 , we have 1 4 (24ηn + log 1 δ ′ ) = 6ηn + 1 4 log 1 δ ′ ≤ 6ηn + 1 3 log 1 δ ′ = i min(α δ ′ ,6η x 2 i , 1) ≤ i min(αx 2 i , 1) by Corollary A.3 as above ≤ i:x 2 i not in the top κ quantile min(αx 2 i , 1) + i:x 2 i in the top κ quantile min(αx 2 i , 1) ≤ i:x 2 i not in the top κ quantile αx 2 i + i:x 2 i in the top κ quantile 1 = i:x 2 i not in the top κ quantile αx 2 i + κn = i:x 2 i not in the top κ quantile αx 2 i + c 1 (24ηn + log 1 δ ′ ) This implies that i:x 2 i not in the top κ quantile αx 2 i ≥ ( 1 4 -c 1 )(log 1 δ ′ + 24ηn) With a slightly modified version of Corollary A.16, bounding i:x 2 i not in the top κ quantile x 2 i by (c3+1)c 2 2 (c2-2) 2 n with probability at least 1 -4 11 δ ′ • e -24ηn , we can choose c 1 = 0.202, c 2 = 398.432, c 3 = 1328.46, and obtain α ≥ 0.0000354(log 1 δ ′ + 24ηn) as desired. Proof of Proposition A.19. First, notice that for all i such that |x i | ≥ max( 1 α , 1 α ) = 1 min(α, α) , the corresponding term in the left hand side becomes 0. Thus, using the notation ≤ to denote summing over elements |x i | ≤ 1 min(α, α) , the left hand side in the guarantee of Proposition A.19 is equal to
≤ x i min(αx 2 i , 1) - ≤ x i min(αx 2 i , 1)
Then, rearranging the sums, we have
≤ x i min(αx 2 i , 1) - ≤ x i min(αx 2 i , 1) = ≤ x i min(αx 2 i , 1) -min(αx 2 i , 1) ≤   ≤ x 2 i     ≤ (min(αx 2 i , 1) -min(αx 2 i ,1
)) 2   by Cauchy-Schwarz for which we can bound the two terms separately. To bound the first term, since we sum over only those terms where |x i | ≤ 1 min(α δ ′ ,8η , α) for all i, and by Lemma A.20 and A.21, min(α δ ′ ,8η , α) ≥ 0.0000354 n (log 1 δ ′ + 24ηn) with probability 1 -8 11 δ ′ • e -24ηn , we have that x 2 i ≤ 28249n log 1 δ ′ +24ηn for all i. Since X has mean 0 and variance 1, we know that E[x 2 i ] ≤ E[X 2 ] = 1, and Var[x 2 i ] ≤ E[x 4 i ] ≤ 28249n log 1 δ ′ +24ηn E[x 2 i ] ≤ 28249n log 1 δ ′ +24ηn . Thus, by Bernstein's inequality,
P   ≤ x 2 i ≥ 19025n   ≤ P   1 n ≤ x 2 i -1 ≥ 19024   ≤ P   1 n ≤ x 2 i -E[x 2 i ] ≥ 19024   ≤ 2 exp - 19024 2 n 56498 n log 1 δ ′ +24ηn + 56498 • 19024 n log 1 δ ′ +24ηn /3 ≤ 2 exp -1.1(log 1 δ ′ + 24ηn) ≤ 2 exp -(log 11 δ ′ + 24ηn) for suff. small δ = 2 11 δ ′ • e -24ηn
In other words, conditioning on Lemma A.20 and A.21 both holding, with probability at least 1
-2 11 δ ′ • e -24ηn , ≤ x 2 i ≤ 19025n.
To bound the second term, note that depending on the order between α := α δ ′ ,8η and α := αδ ′ ,8η , either min(αx 2 i , 1)min(αx 2 i , 1) ≥ 0 for all i, or min(αx 2 i , 1) -min(αx 2 i , 1) ≥ 0 for all i holds. Without loss of generality, we assume that α ≥ α, and that min(αx 2 i , 1) -min(αx 2 i , 1) ≥ 0 for all i. Then, we have
≤ min(αx 2 i , 1) -min(αx 2 i , 1) 2 ≤ i min(αx 2 i , 1) -min(αx 2 i , 1) 2 ≤ i min(αx 2 i , 1) -min(αx 2 i , 1)
since α ≥ α, and that min(αx 2 i , 1) -min(αx 2 i , 1) ≥ 0 for all i. To further upper bound the last quantity, we have
i min(αx 2 i , 1) -min(αx 2 i , 1) = i min(αx 2 i , 1) - i min(αx 2 i , 1) ≤ i min(α δ ′ ,10η x 2 i , 1) - i min(α δ ′ ,8η x 2 i , 1
) by Corollary A.3 and by the definition of α
= 1 3 log 1 δ ′ + 10ηn - 1 3 log 1 δ ′ -8ηn = 2ηn
Finally, summarizing, we have that with high probability at least 1
-10 11 δ ′ • e -24ηn : i x 2 i i (min(αx 2 i , 1) -min(αx 2 i , 1)) 2 ≤ 19025n • 2ηn = n 38050η ≤ 195.065n √ η as desired.
We now present the counterpart of Proposition A.11, bounding the error due to the adversarial corruption with tighter failure probability.
Proposition A.22. Suppose both log 1 δ n and δ are bounded by some small universal constant. Let α be the influence parameter computed from the corrupted samples with robustness level 1 3 log 1 δ ′ + 8ηn, namely, α := αδ ′ ,8η . Then with probability at least 1 -4 11 δ ′ e -24ηn , the mean estimate using α on the clean samples differ from the mean estimate using α on the corrupted samples by at most 26.411 √ η, i.e.,
i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 26.411n √ η
Proof. With arguments identical to that in the proof of Proposition A.11, the maximum error the adversary can incur by corrupting ηn terms is 4ηn 3 √ 3 α . To arrive at a similar lower bound of α in terms of η, notice that by Lemma A.21, we have that with probability at least 1
-4 11 δ ′ • e -24ηn , α ≥ 0.0000354 1 n log 1 δ ′ + 24ηn ≥ 0.0000354 1 n (24ηn) ≥ 0.0008496η
Thus, we have that
i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 4ηn 3 √ 3α ≤ 4ηn 3 √ 3 • 0.0008496η ≤ 26.411n √ η
Equipped with Propositions A.19 and A.22, we are ready to formally prove Theorem 2.2.
Proof of Theorem 2.2. We will start by assuming that the underlying distribution has mean 0 and variance 1 (by the shiftand-scale equivariance of Estimator 1), and furthermore, that the initial estimate κ computed in Step 1 is exactly the true mean of 0. At the end of this proof, we will show that the κ = 0 assumption introduces only a negligible amount of mean estimation error.
Fixing the initial estimate κ = 0, then, Estimator 1 on input the clean samples X and confidence parameter δ computes the influence parameter α := α δ,0 , and returns an estimate μclean,α :=
1 n i x i (1 -min(αx 2 i , 1)). Note that α δ,0 = α δ ′ ,8η , since 1 3 log 1 δ = 1 3 log 1 δ ′ + 8ηn by definition of δ ′ .
As a part of their analysis, Lee & Valiant (2022) showed that
|μ clean,α | ≤ ( √ 2 + o(1)) log 1 δ n
with probability at least 1 -δ. We note that the above guarantee can in fact be slightly strengthened, so that the failure probability becomes δ/22, at the expense of a slightly increased o(1) by no more than a constant multiplicative factor. Now we consider changing the "α" value from α to α computed from the corrupted samples, but applying these α influence parameters on the same clean sample set, and bound the difference in the resulting mean estimates. Specifically, consider α := αδ ′ ,8η , which is the influence parameter computed with robustness level 1 3 log 1 δ ′ + 8ηn from the corrupted samples. Let the mean estimate of using α on the clean samples be μclean,α := 1 n i x i (1-min(αx 2 i , 1)). Then, by Proposition A.19, with probability at least 1 -10 11 δ ′ • e -24ηn = 1 -10 11 δ we have
|μ clean, α -μclean,α | = 1 n i x i min(αx 2 i , 1) - i x i min(αx 2 i , 1) ≤ 195.065 √ η
Next we consider the effect of replacing the effect of corruption, but after fixing the influence parameter to be α. Specifically, define μcorrupt,α to be the mean estimate of the corrupted samples using α, namely
1 n i xi (1 -min(αx 2 i ,1
)). Then, by Proposition A.22, with probability 1 conditioned on Proposition A.19 and thus Lemma A.21 holding, we have
|μ corrupt, α -μclean,α | = 1 n i xi (1 -min(αx 2 i , 1)) - i x i (1 -min(αx 2 i , 1)) ≤ 26.411 √ η
By union bound and triangle inequality, summing over all three error terms, we have that with probability at least 1 -21 22 δ, μcorrupt,α satisfies
|μ corrupt, α| ≤ ( √ 2 + o(1)) log 1 δ n + 221.476 √ η ≤ ( √ 2 + o(1)) log 1 δ n + 222 √ η
Finally, observe that on a mean-0-variance-1 distribution, and with η-corruption, the only difference between μcorrupt,α and the output μ of Estimator 1 (on input as stated in the theorem statement) lies in μcorrupt,α assuming that the initial mean estimate κ was the true mean of 0. On the other hand, μ (Estimator 1) computes κ from data.
We use Fact A.17 as well as the following fact/assumption about the robustness of the initial estimate κ against adversarial corruption to analyze the effect of the κ assumption. Fact A.23 is known to hold true for median-of-means as a folklore result, which we show in Appendix F for completeness. As mentioned in Estimator 1, we can choose to use other estimators to compute the initial estimate κ as long as the estimator satisfies Fact A.23.
Fact A.23 (Folklore). For any distribution D with mean µ and standard deviation σ, let X be a set of n η-corrupted samples from D. Assuming that η ≤ 1 24n log 1 δ , the median-of-means estimate κ from grouping samples into O(log 1 δ ) buckets, on input X, satisfies
P   |κ -µ| ≥ O   σ log 1 δ n     ≤ 1 22 δ By Fact A.23, with probability at least 1 -1 22 δ, |κ| ≤ O log 1 δ n
, and by the Lipschitz bound of Fact A.17, we can bound the (absolute) difference between μ and μcorrupt,α by
|μ -μcorrupt,α | ≤ O   |κ| log 1 δ n   ≤ O   log 1 δ n 2   ≤ o   log 1 δ n  
Thus, on a mean-0-variance 1 distribution, with probability at least 1 -δ over the clean samples (and with an arbitrary η-corruption), Estimator 1 outputs
|μ| ≤ ( √ 2 + o(1)) log 1 δ n + 222 √ η
Combined with the affine invariance of Estimator 1 as stated in Fact A.17, if the underlying distribution instead had mean µ and variance σfoot_0 , we have the mean estimation guarantee
|μ -µ| ≤ σ •   ( √ 2 + o(1)) log 1 δ n + 222 √ η   as desired.
this section cite: ['b16']

Section: B. Robustness of Estimator 1 in Weaker Models
As corollaries to our main results, stating that Estimator 1 is robust against adversarially corrupted data, we also show that Estimator 1 is robust against two (slightly weaker) contamination models, namely Huber contamination and TV contamination. For simplicity, we present and prove direct corollaries of Theorem 2.2 only. Corollaries of Theorem 2.3 follow similarly.
Definition B.1 (Huber contamination (Huber, 1992)). Given a corruption parameter η and a distribution D on the uncorrupted data, we say that a set of n samples is an η-Huber-contaminated sample from D if it is drawn i.i.d. from some distribution (1 -η)D + ηE for an arbitrary distribution E.
The Huber contamination model (Huber, 1992) can be regarded as being weaker than the strong contamination model, because the corruption is always drawn randomly and obliviously from a fixed distribution E chosen by the adversary; on the other hand, the adversary in the strong contamination model gets to choose the corruptions adaptively after inspecting all the samples. We point out however that, due to the random nature of the number of corrupted samples in the Huber model (and TV contamination model later in Definition B.5), these models are not strictly weaker than strong contamination, despite being "weaker in expectation". Nonetheless, by Chernoff bounds, the number of corrupted samples will concentrate to O(ηn) except with exp(-Ω(ηn)) probability.
Later on, in Appendix C, we aim to show the neighborhood optimality of Estimator 1 as a corollary of its robustness against Huber contamination. In those results, we aim for a failure probability at most δ when the algorithm is given δ as input, as opposed to a failure probability that is (slightly) larger than δ. Consequently, we write our theorem below (Theorem B.4) for Huber-contamination robustness with a failure probability δ/2 + exp(-Ω(ηn)), so that it is upper bounded by δ when η is sufficiently large. 2   We will show Theorem B.4 as a corollary of Corollary B.2, which is a variant of the strong-contamination robustness result in Theorem 2.2, with failure probability δ/2 instead of δ. Then, recalling that the probability of having too many corruptions is at most exp(-Ω(ηn)), a union bound gives the target failure probability in Theorem B.4.
Corollary B.2. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, let X be a set of n η-corrupted samples from D.
Theorem B.6. Suppose both log 1 δ n and δ are bounded by some small universal constant. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, and a set X of n η-TV-contaminated samples from D, for some η ≤ 1 24en log 1 δ . Estimator 1 of LV22, when given access to n, δ, and X only, will, with probability at least 1 -( 1 2 δ + exp(-80ηn)) over the sampling process, yield an estimate μ with error at most
|μ -µ| ≤ σ •   (1 + o(1)) 2 log 1 δ n + 222 33η   Proof. Let D ′ be any distribution such that D T V (D, D ′ ) ≤ η.
There exists a coupling between D and D ′ such that for any sample index i, the probability that the sample x i from D and the sample x ′ i from D ′ differs is at most η. Let z i denote the indicator variable for the event that x i differs from x ′ i . By multiplicative chernoff:
P[ n i=1 z i ≥ 33ηn] ≤ e 33-1 33 33 ηn ≤ exp(-80ηn)
Thus with probability at least 1 -exp(-80ηn), the TV-contaminated sample set has at most 33ηn samples corrupted.
Conditioned on this happening, Corollary B.2 applies to the set of contaminated samples X with an adversary capable of arbitrarily corrupting 33ηn samples, with probability at least 1 -1 2 δ. Our theorem thus follows from a union bound.
this section cite: ['b13']

Section: C. Neighborhood Optimality of Estimator 1
Neighborhood optimality is a notion of instance-by-instance optimality beyond the worst-case defined in Dang et al. (2023).
In this section, we show that Estimator 1 is also asymptotically neighborhood optimal as a corollary to its robustness against TV contamination. We point out that our result matches that of Dang et al. (2023), which states that the median-of-means estimator is asymptotically neighborhood optimal, with the same choice of neighborhood structure and asymptotes.
For mean estimation in R, Estimator 1 achieves the optimal sub-Gaussian rate even up to the constants. Their optimality is only in the worst-case regime, i.e., there exists an instance of mean estimation on which no estimator can perform better. A natural question to this observation is: Can we beat the sub-Gaussian rate for some distributions? Dang et al. (2023) presented the following new universal estimation lower bound function:
Definition 2.5 (Dang et al. 2023). Given a (continuous) distribution D with mean µ and a real number t ∈ [0, 1], define the t-trimming operation on D as follows: select a radius r such that the probability mass in [µ -r, µ + r] equals 1 -t; then, return the distribution D conditioned on lying in [µ -r, µ + r].
Given n and δ, define the trimmed distribution D * n,δ to be the 0.45 n log 1 δ -trimmed version of D. When δ is implicit, we may denote this as D * n . Now define the error function
ϵ n,δ (D) = |µ -µ * n | + σ * n log 1 δ n
, where µ * n and σ * n are the mean and standard deviation of D * n respectively.
The ground truth error function ϵ n,δ in Definition 2.5 applies simultaneously for all distributions with a finite mean, not just for the worst-case distribution, in the sense that for any distribution p we can construct a neighboring distribution q that is indistinguishable with probability at least 1 -δ with n samples, but the mean of p and q are well-separated by O(ϵ n,δ (p)) in distance. Thus, no estimator can beat the error function on p and q simultaneously, since any estimator that can will be a distinguisher between p and q.
To formalize this notion of lower bound and optimality, Dang et al. (2023) presented the following pair of definitions:
Definition 5.1 (Neighborhood Pareto Bounds (Dang et al., 2023)). Let n be the number of samples and δ be the failure probability. Given a neighborhood function N n,δ : P 1 → 2 P1 , we say that the error function ϵ n,δ (D) : P 1 → R + 0 is a neighborhood Pareto bound for P 1 with respect to N n,δ if for all distributions D ∈ P 1 , no estimator μ taking n i.i.d. samples can simultaneously achieve the following two conditions:
• For all D ′ ∈ N n,δ (D), with probability 1 -δ over the n i.i.d. samples from D ′ , we have |μ -µ D ′ | ≤ ϵ n,δ (D ′ ).
• With probability 1 -δ over the n i.i.d. samples from D, |μ -µ D | < ϵ n,δ (D).
Definition 5.2 ((κ, τ )-Neighborhood Optimal Estimators (Dang et al., 2023)). Let κ > 1 be a multiplicative loss factor in estimation error, and τ > 1 be a multiplicative loss factor in sample complexity. Given the parameters κ, τ > 1, sample complexity n, failure probability δ and neighborhood function N n,δ , a mean estimator μ is (κ, τ )-neighborhood optimal with respect to N n,δ if there exists an error function ϵ n,δ (D) such that min(ϵ n/τ,δ (D), ϵ n,δ (D)) is a neighborhood Pareto boundfoot_1 , and μ gives estimation error at most κ • ϵ n,δ (D) with probability at least 1 -δ when taking n i.i.d. samples from any distribution D ∈ P 1 .
Definition 5.1 enforces that it is impossible to "beat" the neighborhood Pareto bound locally, performing as good over the local neighborhood, while strictly better on the center p. It essentially enforces admissibility over every such local neighborhood, forming a smooth interpolation between instance optimality and admissibility, which are both classical definitions of optimality beyond the worst-case that fails in context of mean estimation in R. Dang et al. (2023) showed that there exists a neighborhood function N n,δ for which ϵ n,δ as defined in Definition 2.5 is a neighborhood Pareto bound, and for which median-of-means is (κ, 3)-neighborhood optimal for some sufficiently large constant κ. We capture the necessary components of their analysis and state them without proof in the following fact: Fact C.1. There exists a neighborhood function N n,δ : P 1 → 2 P1 for which ϵ n,δ as defined in Definition 2.5 is a neighborhood Pareto bound. Any estimator that obtains estimation error O(ϵ n,δ (p)) for all distributions p ∈ P 1 is (κ, 3)-neighborhood optimal with respect to N n,δ for some sufficiently large constant κ.
Thus, to show that Estimator 1 is neighborhood optimal, it suffices to show that it asymptotically matches the performance of the ground truth error bound ϵ n,δ . We present a reanalysis of Estimator 1 using its robustness against TV contamination that proves its neighborhood optimality in such a way.
Theorem C.2. Suppose both log 1 δ n and δ are bounded by some small universal constant. Denote by N n,δ the neighborhood function whose existence is implied by Fact C.1. Estimator 1 is (κ, 3)-neighborhood optimal with respect to N n,δ for some sufficiently large constant κ.
Proof. Let ϵ n,δ be the ground truth error function as defined in Definition 2.5. By Fact C.1, it suffices to show that Estimator 1 achieves an error rate of O(ϵ n,δ (p)) for all distributions p.
Let p be any distribution with mean µ p . Let c be the constant such that c n log 2 δ = 1 80n log 1 δ ≤ 1 24en log 1 δ . Let p * n be the c n log 2 δ -trimmed version of p as defined in Definition 2.5, and let µ p * n and σ p * n be the mean and standard deviation of p * n respectively. Let ϵ n,δ be the error function as defined in Definition 2.5.
Observe that p * n satisfies that D T V (p * n , p) = c n log 2 δ . Thus, by Theorem B.6 with D = p * n and η = c n log 2 δ , we have that with probability at least 1 -
( 1 2 δ + exp(-80ηn)) = 1 -δ, |μ -µ p * n | ≤ σ p * n •   (1 + o(1)) 2log 1 δ n + 222 33c log 2 δ n   = O   σ p * n log 1 δ n  
Thus by the triangle inequality, we have that with probability at least 1 -δ,
|μ -µ p | ≤ |µ p -µ p * n | + |μ -µ p * n | ≤ |µ p -µ p * n | + O   σ p * n log 1 δ n   = O(ϵ n,δ )
Combined with Fact C.1, this implies that Estimator 1 is indeed (κ, 3)-neighborhood optimal with respect to N n,δ for some sufficiently large constant κ, as desired.
We point out that the proof of Theorem B.6 and Theorem C.2 does not depend on the specific characteristics of Estimator 1, and instead only relies on Theorem 2.2 holding. Thus, we can obtain a similar neighborhood optimality result for any mean estimator that is sub-Gaussian and robust against adversarial corruption, and enjoys asymptotically matching bounds as in Theorem 2.2:
Corollary C.3. Any estimator that, when given δ > 0 and a set of n η-corrupted sample from distribution D with mean µ and variance σ 2 , yields a mean estimate μ satisfying
|μ -µ| ≤ O   σ   log 1 δ n + √ η    
is (κ, 3)-neighborhood optimal with respect to N n,δ for some sufficiently large constant κ.
this section cite: ['b5', 'b5', 'b5', 'b5', 'b5', 'b5', 'b5', 'b5']

Section: D. Remaining Proofs of Section 6
In Section 6, we outlined the proof strategy of our main theorem for low moment performances of Estimator 1, which follows that of Lee & Valiant (2022), and provided the statement of the reformulation of Estimator 1 as a 2-parameter ψ-estimator. In this section, we present the full proof of Theorem 2.4, following the structure and organization of Lee & Valiant (2022). We present each component lemma along with the intuition behind it, and refer the reader to Lee & Valiant (2022) for more detailed motivations and discussions. For completeness, we restate our main theorem:
Theorem 2.4. Given any distribution D with mean µ and z th moment M z for some z ∈ (1, 2), let X be a set of n i.i.d. samples from D. Then, with probability at least 1 -δ over the randomness of X, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ (M z ) 1 z • (1 + o(1)) c z log 1 δ n 1-1 z
where c z = 2(5.6) 1 z-1 -1 . Here, the o(1) term tends to 0 as log 1 δ n , δ → (0, 0), in a manner independent of D and independent of z.
From the structural properties of Estimator 1 stated explicitly in Fact A.17, we make the simplifying assumption that the underlying distribution D has mean 0 and z th moment M z = 1, and that the initial estimate κ in Step 1 of Estimator 1 is replaced by 0. For the rest of the appendix, we will refer to the error bound (c z log 1 δ /n) 1-1/z as ϵ (omitting the 1 + o(1) factor).
Definition 6.1 reduced proving Chernoff bounds for Estimator 1, which is a sum of dependent terms, to proving Chernoff bounds for the sums of independent terms of the 2-parameter ψ-estimator. Thus, it suffices to show that with high probability, the ψ-estimator in Definition 6.1 returns an estimate μ that is close to 0-or equivalently, every pair (μ ′ , α′ ) with μ′ far away from 0 must satisfy ψ(X, μ′ , α′ ) ̸ = 0. We turn to analyze the following proposition, capturing this reduction:
Proposition D.1. There exists a universal constant c > 0 such that, for all 1 < z ≤ 2, fixing ϵ ′ = 1 + c log log 1 δ log 1 δ ϵ where ϵ = c z log 1 δ n
1-1 z and c z = 2(5.6) 1 z-1 -1 , we have that for all distributions D with mean 0 and z th moment 1, with probability at least 1 -δ 2 over the set of samples X, for all μ, α where |μ| > ϵ ′ and α > 0, the vector ψ(X, μ, α) ̸ = 0.
We stress that the universal constant c in fact does not depend on z.
Towards proving Proposition D.1, we extend Lee & Valiant (2022)'s strategy, analyzing the function ψ(X, μ, α) on a finite bounded mesh of values of μ, α covering the most delicate range for our analysis, and show via standard ϵ-net arguments that the proposition holds for any choice of |μ| > ϵ ′ , α. Specifically, we show that ψ(X, μ, α), on the finite mesh, is far from the origin in some direction, and is Lipschitz within the mesh. We then show how to reduce the analysis outside of the mesh to that inside the mesh.
To find a mesh to cover the relevant range of α in the right "scale-invariant" way, we reparameterize α by ŵ := log 2 (1/δ) 3 αn 2 ϵ 2 , where ϵ in the denominator (as defined in Proposition D.1) encodes the desired dependence on z. We choose an evenly-spaced mesh over ŵ ∈ [0.05, 555]. This is analogous to the mesh on "v" in Lee & Valiant (2022)'s analysis.
Having shown the stronger version of the claim for the restriction μ = ϵ ′ and ŵ ∈ [0.05, 555] we now extend to the entire domain via three monotonicity arguments. Explicitly, assume the set of samples X satisfies the dot product inequality above with the vector function d( ŵ), where d( ŵ) satisfies the boundary conditions at ŵ = 0.05 and 555 specified in Lemma D.2.
From this assumption, we will show that ψ ̸ = 0 for any positive ŵ = log 2 1 δ 3n 2 ϵ 2 α , and for any μ ≥ ϵ ′ . First consider ŵ > 555, still fixing μ = ϵ ′ . The function ψ α = n i=1 (min(αx 2 i , 1) -1 3n log 1 δ ) is an increasing function of α, and thus a decreasing function of ŵ = log 2 1 δ 3n 2 ϵ 2 α . Since for ŵ = 555, the dot product d • ψ > 0 with d µ = 0, d α < 0, the dot product will thus remain positive for this same choice of d as we increase ŵ from 555.
Next, for ŵ < 0.05, again fixing μ = ϵ ′ , we analogously show that the dot product of ψ(X, ϵ ′ , α) with the fixed vector d(0.05) will increase as we decrease ŵ. The i-th term in the sums defining ψ µ or ψ α depends on α (and thus ŵ) only in the factor min(αx i , 1). Further, there is no dependence unless the first term attains the min, namely
|x i | ≤ 1/α, which in turn is upper bounded by √ 0.15 nϵ log 1 δ
because of our assumption that ŵ < 0.05. Thus, the only i-th terms in the dot product which have α dependency are simply equal to d µ αx
3 i + d α αx 2 i = αx 2 i (d α + x i d µ ). By our choice of d µ (0.05) = √ 3.75 log 1 δ nϵ and d α = √ 3 from Lemma D.2, the expression (d α + x i d µ ) ≥ √ 3 - √ 0.15 √ 3.
75 is thus always non-negative, and thus the overall dot product cannot decrease as we send ŵ to 0 as desired.
We have thus shown that, for all non-negative α = log 2 1 δ 3n 2 ϵ 2 ŵ , there is a vector d with d µ ≥ 0 whose dot product with ψ(X, ϵ ′ , α) is greater than 0. We complete the proof by noting that the only dependence on μ in ψ is that ψ µ is (trivially) increasing in μ. Since d µ ≥ 0, increasing μ from ϵ ′ will only increase the dot product, and thus the dot product remains strictly greater than 0, implying that ψ(X, μ, α) ̸ = 0, as desired.
To close out this section, we formally show that Proposition D.1 implies our desired main theorem, Theorem 2.4 for completeness, using the following fact about the performance of the median-of-means estimator for the initial estimate κ: Fact D.4 (Bubeck et al. 2012). For any distribution D with mean µ and z th moment σ z , the median-of-means estimate κ from grouping samples into O(log 1 δ ) buckets, on input n samples, satisfies
P   |κ -µ| > O   (σ z ) 1 z log 1 δ n 1-1 z     ≤ 1 2 δ
Here, the big-O notation hides a universal constant that is crucially also independent of z.
Again, we are free to choose to use other estimators for κ in Step 1 of Estimator 1 as long as the above fact holds true for the estimator being used.
Proof of Theorem 2.4. We start by making the simplifying assumption that µ = 0 and M z = 1, and reformulate Estimator 1 with Step 1 replaced with κ = 0 as a 2-parameter ψ-estimator, that takes in n independent samples X = x 1 , • • • , x n from D, and solves for the (unique) pair μ, α satisfying ψ µ = 0 and ψ α = 0, where the functions are defined as follows:
ψ µ (X, μ, α) = n i=1 (μ -x i (1 -min(αx 2 i , 1))) ψ α (X, μ, α) = n i=1 min(αx 2 i , 1) - 1 3n log 1 δ
and outputs μ in the solution. We denote by ψ the 2-element vector (ψ µ , ψ α ).
this section cite: ['b16', 'b16', 'b16', 'b16', 'b3']

Section: By Proposition
D.1, with probability at least 1 -δ 2 , any μ′ , α′ with |μ ′ | > (1 + o(1)) c z log 1 δ n 1-1 z and α′ > 0 satisfies that ψ(X, μ′ , α′ ) ̸ = 0, and thus the solution μ found by Estimator 1 by solving ψ = 0 satisfies |μ| ≤ (1 + o(1)) c z log 1 δ n 1-1 z . We now remove the simplifying assumptions using the structural properties of Estimator 1. Let κ be the initial estimate in Step 1 of Estimator 1, say, computed by median-of-means. By Fact D.4, with probability at least 1 -1 2 δ, |κ| ≤ O   log 1 δ n 1-1 z   and by the Lipschitz bound of Fact A.17 with Lipschitz constant O log 1 δ n
, this incurs an error on Estimator 1 of at most
O   log 1 δ n 1-1 z • log 1 δ n   ≤ o   log 1 δ n 1-1 z   Since c 1-1 z z
is lower bounded by a universal constant, the above expression is also o c z
log 1 δ n 1-1 z .
Now consider any distribution with mean µ and z th moment M z . By the affine invariance of Estimator 1 of Fact A.17, Estimator 1 suffers a multiplicative factor of (M z ) 1 z in the estimation error.
Thus with probability at least 1 -δ, we have
|μ(X) -µ| ≤ (M z ) 1 z • (1 + o(1)) c z log 1 δ n 1-1 z as desired.
this section cite: []

Section: D.1. Proof of Lemma D.2 and D.3
In this section, we present the proof of Lemma D.2, motivated by the discussion in Section 6.2 and consequently Lee & Valiant (2022), modeling the worst-case Chernoff bound as a max-min linear programming game. We later present the short proof of Lemma D.3.
Lemma D.2. Let D be an arbitrary distribution with mean 0 and z th moment 1 for some 1 < z ≤ 2. There exists a universal constant c independent of z where the following is true. Fixing μ = 1 +
P X←D n d( ŵ) • ψ X, μ = ϵ ′ , α = log 2 (1/δ) 3 ŵn 2 ϵ 2 > 1 log 1 δ ≥ 1 - δ log 4 1 δ Furthermore, for ŵ = 0.05 we have d µ = √ 3.75 log 1 δ nϵ , d α = √
3; and for ŵ = 555 we have d µ = 0, d α < 0.
Proof. We instead bound the contrapositive statement, namely,
P X←D n d( ŵ) • ψ X, μ = ϵ ′ , α = log 2 (1/δ) 3 ŵn 2 ϵ 2 ≤ 1 log 1 δ ≤ δ log 4 1 δ
We start by applying a standard Chernoff bound analysis
P X←D n d( ŵ) • ψ (X, μ, α) ≤ 1 log 1 δ = P X←D n exp (-d( ŵ) • ψ (X, μ, α)) ≥ exp - 1 log 1 δ ≤ 2 E X←D n (exp (-d( ŵ) • ψ (X, μ, α))) by Markov's and exp - 1 log 1 δ ≤ 2 for suff. small δ = 2 E x←D (exp (-d( ŵ) • ψ (x, μ, α))) n by independence = 2 exp -d µ μ + d α 1 3n log 1 δ E x←D exp d µ x(1 -min(αx 2 , 1)) -d α min(αx 2 , 1) n
Motivated by our discussion in Section 6.2, we state the following technical claim, Lemma D.5, reminiscent of the constraint in (2). We formally prove Lemma D.5 in Appendix D.2.
Lemma D.5. For all ŵ ∈ [0.05, 555], there exists a > 0, b such that for all 1 < z ≤ 2 and y ∈ R, letting c z = 2(5.6) 1 z-1 -1 , the following holds true:
ay(1 -min(y 2 , 1)) -b min(y 2 , 1) ≤ log 1 + ay + |y| z (3 ŵ) z 2 c z-1 z -1 + a √ 3 ŵ - b 3(4)
where a and b are bounded by constants. Further, for ŵ = 0.05, the pair a = 0.75, b = √ 3 works.
For ŵ ∈ [0.05, 555), we use Lemma D.5, substituting y = √ αx, to choose
d µ = √ αx = log 1 δ nϵ √ 3 ŵ a, d α = b.
In particular, for ŵ = 0.05, we have d µ = √ 3.75 log 1 δ nϵ and d α = √ 3. Then, the failure probability is bounded by
2   exp -d µ μ + d α 1 3n log 1 δ E x←D y=x √ α 1 + ay + |y| z (3 ŵ) z 2 c z-1 z -1 + a √ 3 ŵ - b 3   n = 2 exp -d µ μ + d α 1 3n log 1 δ 1 + log 1 δ nϵ √ 3 ŵ z (3 ŵ) z 2 c z-1 z -1 + d µ ϵn log 1 δ - d α 3 n since D has mean 0, z th moment 1 = 2 exp -d µ μ + d α 1 3n log 1 δ 1 + log 1 δ n -1 + d µ ϵn log 1 δ - d v 3 n ≤ 2 exp -d µ μ + d α 1 3n log 1 δ + log 1 δ n -1 + d µ ϵn log 1 δ - d α 3 n since 1 + x ≤ e x for any x = 2 exp -d µ μ + d α 1 3n log 1 δ - 1 n log 1 δ + d µ ϵ -d α 1 3n log 1 δ n = 2 exp -d µ n(μ -ϵ) -log 1 δ = 2 exp - a √ 3 ŵ c log log 1 δ -log 1 δ
by choice of d µ and μ ≤ δ log 4 1 δ as desired for large enough c, since a √ 3 ŵ is greater than some positive constant. For ŵ = 555, we use the following technical claim: Claim D.6. For all 1 < z ≤ 2, for all values of y, the following holds true: 4 min(y 2 , 1) ≤ log(1 + 54|y| z ) which is easily verifiable by two subclaims in the form of 4y 2 ≤ log(1 + 54y 2 ) for -1 ≤ y ≤ 1 and 4 ≤ log(1 + 54|y|) for |y| > 1.
We choose d µ = 0 and d α = -4.
Substituting yields 2 exp -4 3n log 1 δ E x←D exp 4 min(αx 2 , 1) n ≤ 2δ 4/3 E x←D (1 + 54 √ α z |x| z ) n by Claim D.6 = 2δ 4/3 (1 + 54 log 1 δ nϵ √ 3 ŵ z ) n since D has z th moment 1 ≤ 2δ 4/3 exp n • 54 log 1 δ nϵ √ 3 • 555 z since 1 + x ≤ e x = 2δ 4/3 exp   n • 54   1 2 1-1 z 5.6 2 z -1 √ 1665 log 1 δ n 1 z   z   by definition of ϵ = 2δ 4/3 exp 54 2 z-1 5.6 2-z √ 1665 z log 1 δ ≤ 2δ 4/3 δ -0.237 for all values of 1 < z ≤ 2 ≤ 2δ 1.096 ≤ δ log 4 1 δ for suff. small δ as desired.
We have thus proven one of two components necessary to prove Proposition D.1. We restate and prove the remaining component, which is a Lipschitz bound over the region covered by our finite mesh over ŵ ∈ [0.05, 555]: Lemma D.3. Consider an arbitrary set of n samples X. Consider the expressions ψ µ (X, μ, α), ψ α (X, α), reparameterized in terms of ŵ = log 2 1 δ 3 αn 2 ϵ 2 in place of α. Suppose the equation ψ α (X, α) = 0 has a solution in the range ŵ ∈ [0.05, 555]. Proof. Consider the derivative with respect to ŵ of ψ α (X, μ, α) = n i=1 min
this section cite: ['b16']

Section: Then the functions
log 2 1 δ 3n 2 ϵ 2 ŵ x 2 i , 1 -1 3n log 1 δ . The ŵ derivative of min log 2 1 δ 3n 2 ϵ 2 ŵ x 2 i , 1 is either - log 2 1 δ 3n 2 ϵ 2 ŵ2 x 2 i = -1 ŵ αx 2
i or 0, depending on which term in the min is the smallest, and in either case has magnitude at most 1 ŵ min(αx 2 i , 1). Thus, the overall ŵ derivative of ψ α (X, μ, α) has magnitude at most 1 ŵ i min(αx 2 i , 1). Since by assumption i min(αx 2 i , 1) = 1 3 log 1 δ for some ŵ ∈ [0.05, 555], the derivative with respect to ŵ must be within a constant factor of log 1 δ across the entire range, as desired. Similarly, consider the derivative with respect to ŵ of ψ µ (X, μ, α) = n i=1 (μ
-x i (1 -min(αx 2 i , 1))). The ŵ derivative of (μ -x i (1 -min(αx 2 i , 1))) is either -1 ŵ αx 3 i or 0, depending on whether x i ≤ 1/α,
and thus the magnitude of the entire derivative is bounded by 1 ŵ√ α i min(αx 2 i , 1). Since i min(αx 2 i , 1) is bounded by a constant times log 1 δ , and 1 ŵ√ α is bounded by a constant times 1 √ ŵ α = √ 3nϵ log 1 δ since ŵ ∈ [0.05, 555], the magnitude of the derivative of log 1 δ nϵ ψ µ (X, μ, α) is bounded by a constant times log 1 δ , as desired.
this section cite: []

Section: D.2. Proof of Lemma D.5
In this section, we prove the technical Lemma D.5.
Lemma D.5. For all ŵ ∈ [0.05, 555], there exists a > 0, b such that for all 1 < z ≤ 2 and y ∈ R, letting c z = 2(5.6) 1 z-1 -1 , the following holds true:
ay(1 -min(y 2 , 1)) -b min(y 2 , 1) ≤ log 1 + ay + |y| z (3 ŵ) z 2 c z-1 z -1 + a √ 3 ŵ - b 3(4)
For negative y, the left hand side is convex and the right hand side is concave, so the left hand side minus the right hand side is convex, and its maximum must be attained at one of its two endpoints, y = -1 or y = 0. Numerically checking both cases confirms the inequality for y ≤ 0. For positive y, as in the previous case, we take derivatives of the left and right hand side and set them equal to each other, leaving us with a cubic equation, which thus has closed form solutions. We thus check that the desired inequality is true at all the positive roots of the cubic, along with the extreme points y = 0, y = 1.
Case z = 1 and w ∈ (0.05, 5.5): In this case we have chosen a = -
√ 6+ √ 6+96 ŵ 2 √ 2 ŵ
-the positive root of the equation √ 2 ŵ(a 2 -12) + √ 6a = 0-and b = 3 -a 2 2 . Since a is an increasing function of ŵ it is easy to check that, for ŵ in our range (0.05, 5.5) we have a ∈ (0, 3.12]. Because of these relations between a, b, ŵ, we can simplify part of the expression in (4) inside the logarithm as
√ 3 ŵ -1 + a √ 3 ŵ - b 3 = 1 2 a
Using this relation, and substituting b = 3 -a 2 2 into the left hand side of ( 4) simplifies (4) to:
ay(1 -min(y 2 , 1)) -(3 -a 2 2 ) min(y 2 , 1) ≤ log (1 + ay + 2.8a|y|)
As above, we point out that the left hand side is constant for y outside the range [-1, 1] and the right hand side is monotonic away from this interval, so it suffices to prove the inequality for y ∈ [-1, 1] in which case it simplifies to
ay(1 -y 2 ) -(3 - a 2 2 )y 2 ≤ log (1 + ay + 2.8a|y|)(5)
We first show the y ≥ 0 case. Since a > 0 we have ay ≥ 0. Substituting ay → x above yields the following inequality, which we show for x ≥ 0 and a ∈ (0, 3.12]:
x + x 2 2 - 1 a 2 (x 3 + 3x 2 ) ≤ log (1 + 3.8x)
The term x 3 + 3x 2 is nonnegative for x ≥ 0, and thus the left hand side is increasing in a. Thus it suffices to prove the inequality for the maximum value of a = 3.12, and all x ≥ 0.
As above, we prove this by pointing out that both sides are smooth functions for x ≥ 0, so we take their derivatives and set them equal to each other, which yields a cubic equation in x. Our inequality takes its extreme values at either a positive root of the cubic, or at the boundary value x = 0; we thus confirm the inequality in these cases to prove it in general.
We now show the y ≤ 0 case.
We point out that the left hand side of Equation 5, ay(1 -y 2 ) -(3 -a 2 2 )y 2 , is increasing in y at y = 0, convex for sufficiently negative y, and can only transition from convex to concave once. Meanwhile, the right hand side, log (1 -1.8ay), is decreasing everywhere for y ≤ 0, and concave everywhere.
Let c be the location where the cubic function ay(1 -y 2 ) -(3 -a 2 2 )y 2 transitions from convex to concave. For those y in the (possibly empty) interval [c, 0], the cubic is concave-since it is also increasing at y = 0, it must be increasing on this entire interval. Recalling that the right hand side of Equation 5 is decreasing for y ≤ 0, the difference between the left and right hand sides attains its maximum in the interval y ∈ [c, 0] at y = 0, which is just 0, satisfying the inequality.
And for those y < c, where the left hand side is convex, then the difference between the left and right hand sides is convex, and thus its maximum must occur either at the left extreme, y = -1, or the right extreme, y = c; however the difference at y = c we already showed was at most the difference at y = 0, so overall, the maximum difference between the left and right hand sides must occur at either y = -1 or y = 0. As above, the y = 0 case is trivial, and it remains to prove the y = -1 case. For the y = -1 case Equation 5 becomes:
a 2 2 -3 ≤ log (1 + 1.8a)
The left hand side is convex and the right hand side is concave; so we prove the inequality by numerically checking both endpoints: a = 0 and a = 3.12.
this section cite: []

Section: 
Suppose both log 1 δ n and δ are bounded by some small universal constant, and suppose η ≤ 1 24n log 1 δ . Then, with probability at least 1 -1 2 δ over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   (1 + o(1)) 2log 1 δ n + 222 √ η  
The proof of Corollary B.2 is almost identical to that of Theorem 2.2. The main difference is, instead of using Fact 1.1 to guarantee the performance of Estimator 1 on i.i.d. uncorrupted samples, we use a stronger variant which is Fact B.3 below. Specifically, Fact B.3 inputs the parameter δ to Estimator 1 but asks for a failure probability of δ/2 instead of δ, at the expense of a slightly larger "o(1)" term. Given the above target parameters, Fact B.3 is not a black-box corollary of Fact 1.1. Nonetheless, it follows directly from the analysis in Lee and Valiant's original paper (2022), so we state it without proof below.
Fact B.3 (Lee & Valiant 2022). Given any distribution D with mean µ and variance σ 2 , parameters n, δ > 0, let X be a set of n independent samples from D. Then, with probability at least 1 -δ/2 over the sampling process, Estimator 1 on input δ and X will output an estimate μ with error at most
|μ -µ| ≤ σ •   (1 + o(1)) 2log 1 δ n  
Here, the o(1) term tends to 0 as log 1 δ n , δ → (0, 0) and, crucially, is independent of D.
We can now state and prove the robustness of Estimator 1 against Huber contamination.
Theorem B.4. Suppose both log 1 δ n and δ are bounded by some small universal constant. Given any distribution D with mean µ and variance σ 2 , parameters n, δ, η > 0, and a set X of n η-Huber-contaminated samples from D, for some η ≤ 1 24en log 1 δ . Estimator 1 of LV22, when given access to n, δ, and X only, will, with probability at least 1 -( 1 2 δ + exp(-80ηn)) over the sampling process, yield an estimate μ with error at most
|μ -µ| ≤ σ •   (1 + o(1)) 2 log 1 δ n + 222 33η  
Proof. Note that sampling from distribution (1 -η)D + ηE is exactly identical to sampling from D with probability 1 -η and from E with η. We bound the probability that such a sampling process samples more than 33ηn samples from E.
Let z i denote the indicator variable for the event that sample x i is sampled from E. By multiplicative Chernoff:
P[ n i=1 z i ≥ 33ηn] ≤ e 33-1 33 33 ηn ≤ exp(-80ηn)
Thus with probability at least 1 -exp(-80ηn), the Huber-contaminated sample set has at most 33ηn samples corrupted.
Conditioned on this happening, Corollary B.2 applies to the set of contaminated samples X with an adversary capable of arbitrarily corrupting 33ηn samples, with probability at least 1 -1 2 δ. Our theorem thus follows from a union bound.
Generalizing from Huber contamination slightly is the TV contamination model, which draws samples from a distribution D ′ that is within η TV distance from the genuine underlying distribution D.
Definition B.5 (TV contamination (Diakonikolas & Kane, 2023)). Given a corruption parameter η and a distribution D on the uncorrupted data, we say that a set of n samples is an η-TV-contaminated sample from D if it is drawn i.i.d. from some distribution D ′ such that D TV (D, D ′ ) ≤ η.
And correspondingly, we have the following theorem showing that Estimator 1 is robust against TV contamination.
The main technical component of our (and Lee & Valiant (2022)'s) proof strategy is the proof that for each μ, ŵ = log 2 (1/δ) 3 αn 2 ϵ 2
from the finite mesh we analyze, ψ(X, μ, α) is far away from 0 in some direction. We linearize this claim and prove a stronger result, that there exists a specific direction d( ŵ) such that with high probability, ψ(X, μ, α) is more than 1 log(1/δ) distance away from the origin in direction d. More formally, we prove the following lemma: Lemma D.2. Let D be an arbitrary distribution with mean 0 and z th moment 1 for some 1 < z ≤ 2. There exists a universal constant c independent of z where the following is true. Fixing μ = 1 +
P X←D n d( ŵ) • ψ X, μ = ϵ ′ , α = log 2 (1/δ) 3 ŵn 2 ϵ 2 > 1 log 1 δ ≥ 1 - δ log 4 1 δ Furthermore, for ŵ = 0.05 we have d µ = √ 3.75 log 1 δ nϵ , d α = √
3; and for ŵ = 555 we have d µ = 0, d α < 0.
From here, Proposition D.1 follows from a union bound over pairs of (μ, α) on the finite mesh, the monotonicity of ψ(X, μ, α) beyond the mesh granted by the boundary conditions specified in Lemma D.2, and the Lipschitzness of ψ(X, μ, α) within the region covered by the mesh, formalized in the following lemma: With these components, we now prove Proposition D.1, and subsequently formally prove our main theorem, Theorem 2.4, before returning to prove Lemma D.2 and D.3 in Appendix D.1.
Lemma D.3. Consider an arbitrary set of n samples X. Consider the expressions ψ µ (X, μ, α), ψ α (X, α), reparameterized in terms of ŵ = log 2 1 δ 3 αn 2 ϵ 2 in place of α. Suppose the equation ψ α (X, α) = 0 has
Proof of Proposition D.1. By symmetry, instead of considering positive and negative μ, it suffices to consider the case μ > ϵ ′ and show that this case succeeds with probability at least 1 -δ 4 . To prove the claim, we first prove a stronger statement on a restricted domain, that with probability at least 1 -δ 4 over the randomness of the sample set X, for each ŵ ∈ [0.05, 555] there exists a vector d = (d µ , d α ) such that d • ψ(X, ϵ ′ , α) > 0, with d µ > 0 throughout, and , for ŵ = 0.05 we have d µ = √ 3.75
log 1 δ nϵ , d α = √
3; and for ŵ = 555 we have d µ = 0, d α < 0.
We will first apply Lemma D.2 to each ŵ in a discrete mesh: let M consist of evenly spaced points between 0.05 and 555 with spacing 1/ log 3 1 δ (thus with Θ(log 3 1 δ ) many points). By Lemma D.2 and a union bound over these Θ(log 3 1 δ ) points, we have that with probability at least 1 -
δ Θ(log 1 δ ) (which is at least 1 -δ
4 for δ smaller than some universal constant) over the set of n samples X, for all ŵ ∈ M , there exists a vector
d( ŵ) such that d( ŵ) • ψ(X, μ = ϵ ′ , α) > 1/log 1
δ , where d further satisfies the desired positivity and boundary conditions, and where both nϵ log 1 δ |d µ | and |d α | are bounded by a universal constant. For the rest of the proof, we will only consider sets of samples X satisfying the above condition. Now consider an arbitrary ŵ′ ∈ [0.05, 555] \ M and consider the vector ψ evaluated at α′ = log 2 1 δ 3n 2 ϵ 2 ŵ′ . We wish to extend the dot product inequality to hold also for ŵ′ . If ψ α ̸ = 0 then there is nothing to prove: set d µ = 0 and d α = sign(ψ α ); otherwise, ψ α = 0 means we may apply Lemma D.3 to conclude that both log 1 δ nϵ ψ µ (X, μ, α′ ) and ψ α (X, μ, α′ ) are Lipschitz with respect to ŵ′ on the interval ŵ′ ∈ [0.05, 555] with Lipschitz constant clog 1 δ for some universal constant c. Consider the closest ŵ ∈ M to ŵ′ , which by definition of M is at most 1/ log 3 1 δ away. By assumption on X, there exists a vector d such that d • ψ(X, μ = ϵ ′ , α) > 1/log 1 δ , with d µ > 0 and both nϵ log 1 δ |d µ | and |d α | are bounded by a universal constant. Because of the Lipschitz bounds on ψ, combined with the bounds on the size of d µ and d α , we conclude that the Lipschitz constant of the dot product (treating the vector d as fixed) is O(log 1 δ ). Thus, the large positive dot product at ŵ implies at least a positive dot product nearby at ŵ′ :
d • ψ(X, μ = ϵ ′ , ŵ′ ) > 1 log 1 δ -O(log 1 δ ) 1 log 3 1 δ
> 0, for sufficiently small δ as given in the proposition statement.
where a and b are bounded by constants. Further, for ŵ = 0.05, the pair a = 0.75, b = √ 3 works.
Proof. We remark that Lemma D.5 is more nuanced than its counterpart in Lee & Valiant (2022), in part due to the introduction of a new variable z in the exponent. However, notice that taking the derivative with respect to z of the right hand side of (4), omitting the outer logarithm-which is a monotone function-and omitting terms independent of z, yields
∂ ∂z |y| z (3 ŵ) z 2 c z-1 z = (|y| √ 3 ŵ) z • 2 z-1 • 5.6 2-z • log 2|y| √ 3 ŵ 5.6
whose sign is determined solely by the sign of log 2|y| √ 3 ŵ 5.6
, and thus is independent of z. This implies that the right hand side of (4) is monotone in z, and thus it suffices to show that (4) holds for the boundary cases z → 1 and z = 2. Since
lim z→1 c z-1 z = lim z→1
(2(5.6) 1 z-1 -1 ) z-1 = 5.6, we evaluate the limit z → 1 of (4) by simply substituting z = 1 into the equation and replacing the term c z-1 z with 5.6.
We now choose the values of the parameters a, b. We follow the choices of Lee & Valiant (2022) for w < 5.5: for w = 0.05 we set a = 0.75, b = √ 3 as promised in the lemma statement; and for w ∈ (0.05, 5.5] we choose a = -
√ 6+ √ 6+96 ŵ 2 √ 2 ŵ , i.e.,
the positive root of the equation √ 2 ŵ(a 2 -12) + √ 6a = 0, and b = 3 -a 2 2 . Note that these choices set a > 0, as promised in the lemma. For w ≥ 5.5 we differ from Lee & Valiant (2022) and instead choose the constants a = 4.2 and b = -3 (independent of ŵ).
Case z = 2 and w ∈ [0.05, 5.5): In this case our lemma is identical to its counterpart in Lee & Valiant (2022) and thus needs no proof.
Case z = 2 and w ≥ 5.5: In this case we choose a = 4.2 and b = -3, and thus Equation 4 Further, the logarithm term is the only term that depends on ŵ, and the logarithm monotonically increases with ŵ, so thus it is sufficient to prove the inequality for the smallest value of ŵ, namely ŵ = 5.5. We thus show:
4.2y(1 -y 2 ) + 3y 2 ≤ log 1 + 4.2y + y 2 √ 3 • 5.5 • 2 • 4.2
To prove this, we note that both sides are smooth for y ∈ [-1, 1]; thus we take derivatives of both sides and set them equal to each other. The derivative of the left hand side is a quadratic polynomial; and the derivative of the right hand side is the multiplicative inverse of a quadratic polynomial. So thus the solutions to this equation are the solutions to a quartic. And it is straightforward to find these four solutions for y with a computer algebra package, and confirm that, at all of these four possible extrema and the points y = -1, y = 1, the desired inequality is true.
Case z = 1 and w ≥ 5.5 Next, we turn to the case of z = 1 and w ≥ 5.5, where we set a = 4.2 and b = -3. Equation 4 now simplifies to a condition independent of ŵ:
4.2y(1 -min(y 2 , 1)) + 3 min(y 2 , 1) ≤ log (1 + 4.2y + |y| • 5.6 • 4.2) As above, the expression inside the logarithm on the right hand side is increasing outside the range y ∈ [-1, 1] and the left hand side is constant outside this range, so it is sufficient to show this inequality for y ∈ [-1, 1] in which case it simplifies to 4.2y(1 -y 2 ) + 3y 2 ≤ log (1 + 4.2y + |y| • 5.6 • 4.2) Case z = 1 and ŵ = 0.05: In this case we choose a = 0.75 and b = √ 3 and Equation 4 becomes 0.75y(1 -min(y 2 , 1)) -√ 3 min(y 2 , 1) ≤ log (1 + 0.75y + 2.086|y|) where 2.086 is a lower bound on 15 √ 3 ŵ -1 + a √ 3 ŵ -b 3 As usual, for y outside [-1, 1] we point out that the left hand side is constant while the right hand side monotonically increases, so it suffices to show the inequality for y ∈ [-1, 1]. For y ∈ [-1, 1], we trivially lower bound the right hand side by log(1 + 0.75y) (dropping the 2.086|y| term), and lower bound this logarithm expression with the quadratic 0.75y -0.75y 2 . The difference between the left hand side and this polynomial lower bound on the right hand side is thus -.75y 3 + (.75 -√ 3)y 2 , which is negative for y > 1 -4 √ 3 , which is below -1, and hence our inequality holds on the entire interval y ∈ [-1, 1]. Thus we have shown all cases of the desired inequality.
this section cite: ['b16', 'b8', 'b16', 'b16', 'b16', 'b16', 'b16']

Section: E. Proof of Theorem 2.8
In this section, we present the proof of Theorem 2.8. We restate the theorem for completeness: Theorem 2.8. Let D be a distribution with mean µ and variance σ 2 .
Let μ denote Estimator 1 on input parameter δ and n i.i.d. samples from D. Also let Xn denote the sample mean. Then, fixing δ and D and taking n → ∞, we have
√ nμ p → √ n Xn that is, | √ nμ - √ n Xn | p → 0, that √ nμ converges to √ n Xn in probability.
As a corollary, by the Central Limit Theorem, we have
√ n(μ -µ) d → N 0, σ 2
That is, μ is asymptotically normal and efficient.
Proof. Without loss of generality, by the shift-and-scale equivariance of both Lee and Valiant's estimator and the sample mean, we assume µ = 0 for notational simplicity, and that the variance of D is 1.
The second part of the theorem follows from the straightforward reasoning that, since Estimator 1 converges to the sample mean (as claimed in the first part of the theorem statement), and since the sample mean converges to a Gaussian (from the Central Limit Theorem), then Estimator 1 also converges to the same Gaussian. Formally, this uses Slutsky's theorem and the fact that convergence in probability implies convergence in distribution. Hence it only remains to show the first part of the theorem, that Lee and Valiant's estimator converges to the sample mean in probability, for fixed δ and as n → ∞.
The claim that √ nμ p → √ n Xn is equivalent by definition to the statement that, for any fixed ϵ > 0,
lim n→∞ P( √ n|μ -Xn | > ϵ) = 0
First, recall that μ differs from the sample mean by removing a total of 1 3 log 1 δ weighted samples before taking the average. Thus,
√ n|μ -Xn | is upper bounded by Θ log 1 δ |x max |/ √
n where x max is the largest sample in magnitude (recalling that we assumed µ = 0):
√ n κ + 1 n i (x i -κ)(1 -min(α(x i -κ) 2 , 1)) - 1 n i x i = 1 √ n i (x i -κ) min(α(x i -κ) 2 , 1) ≤ 2|x max | √ n i min(α(x i -κ) 2 , 1) = 2 3 log 1 δ |x max | √ n Thus, the event √ n|μ -Xn | > ϵ implies |x max | > ϵ √ n Θ(log 1 δ )
, which we now show is an event that occurs with probability → 0 as n → ∞.
We show the following claim, that the probability a single sample from D is larger than ϵ √ n Θ(log 1 δ ) is at most o(1/n). Claim E.1. Fix any ϵ > 0 and δ ∈ (0, 1). Suppose we draw a single sample X from a distribution D with mean 0 and variance 1. Then,
P |X| > ϵ √ n Θ log 1 δ = o 1 n
Here, the o(•) is in the limit where ϵ, δ and D are fixed, and n → ∞.
We prove Claim E.1 at the end. Claim E.1 implies that the expected number of samples (among the n drawn) with magnitude exceeding ϵ √ n Θ(log 1 δ ) is o(1). Thus, by Markov's inequality, the probability that at least one sample exceeds that threshold is also o(1), showing Theorem 2.8. We finish with the short proof of Claim E.1. Here, for a generic non-negative random variable Y , we use a refined version of Markov's inequality that
P(Y > a) ≤ E[Y 1[Y ≥ a]] a Applying this to X 2 in Claim E.1, we have P X 2 > ϵ 2 n Θ log 2 1 δ ≤ E X 2 1 X 2 ≥ ϵ 2 n Θ log 2 1 δ • Θ log 2 1 δ ϵ 2 n
Since E[X 2 ] = 1 and ϵ, δ and D are fixed, we have
E X 2 1 X 2 ≥ ϵ 2 n Θ log 2 1 δ = o(1) as n → ∞.
Thus, the right hand side in the above inequality is o(1/n), showing Claim E.1. This completes the proof that μ converges to Xn in probability, showing the theorem.
this section cite: []

Section: F. Proofs of the Folklore Robustness of Median-of-Means
For completeness, we provide formal proofs of Facts A.18 and A.23, two folklore facts about the robustness of the median-of-means estimator against adversarial corruption, which we use in our proofs of Theorems 2.2 and 2.3.
Fact A.18 (Folklore). For any distribution D with mean µ and standard deviation σ, let X be a set of n η-corrupted samples from D. The median-of-means estimate κ from grouping samples into O(log 1 δ ′ + ηn) buckets, on input X, satisfies
P   |κ -µ| ≥ O   σ log 1 δ ′ n + η     ≤ 1 16 δ ′
Proof. Let k denote the chosen amount of buckets. Choose k = 16(log 1 δ ′ + ηn). Let µ i denote the mean of the i-th bucket before any adversarial corruption.
Then each µ i is independent with variance σ 16(log 1 δ ′ +ηn) n . By Chebyshev's inequality, P[|µ i -µ| ≥ 4σ 16(log 1 δ ′ +ηn) n ] ≤ 1 16 . Let d i denote the indicator variable for the event |µ i -µ| ≤ 4σ 16(log 1 δ ′ +ηn) n , then E(d i ) ≥ 15 16 . By Hoeffding's inequality, P[ i d i ≤ 9(log 1 δ ′ + ηn)] ≤ e -4.5(log 1 δ ′ +ηn) ≤ e -(log 1 δ ′ +ηn) , which is at most e -log 16 δ ′ = 1 16 δ ′ for large enough n. Note that for |κ -µ| ≥ 4σ 16(log 1 δ ′ +ηn) n , at most k/2 = 8(log 1 δ ′ + ηn) buckets can have |µ i -µ| ≤ 4σ 16(log 1 δ ′ +ηn) n . Accounting for the adversarial corruption, which can affect at most ηn buckets, at most 9(log 1 δ ′ + ηn) buckets can have |µ i -µ| ≤ 4σ 16(log 1 δ ′ +ηn) n before adversarial corruption, which happens with probability at most 1 16 δ ′ as desired.
Fact A.23 (Folklore). For any distribution D with mean µ and standard deviation σ, let X be a set of n η-corrupted samples from D. Assuming that η ≤ 1 24n log 1 δ , the median-of-means estimate κ from grouping samples into O(log 1 δ ) buckets, on input X, satisfies
P   |κ -µ| ≥ O   σ log 1 δ n     ≤ 1 22 δ
Proof. Let k denote the chosen amount of buckets. Choose k = 16log 1 δ . Let µ i denote the mean of the i-th bucket before any adversarial corruption.
Then each µ i is independent with variance σ 16log 1 δ n . By Chebyshev's inequality, P[|µ i -µ| ≥ 4σ 16log 1 δ n ] ≤ 1 16 . Let d i denote the indicator variable for the event |µ i -µ| ≤ 4σ 16log 1 δ n , then E(d i ) ≥ 15 16 . By Hoeffding's inequality, P[ i d i ≤ 9log 1 δ ] ≤ e -4.5log 1 δ , which is at most e -log 22 δ = 1 22 δ for sufficiently small δ. Note that for |κ -µ| ≥ 4σ 16log 1 δ n , at most k/2 = 8log 1 δ buckets can have |µ i -µ| ≤ 4σ 16(log 1 δ ′ +ηn) n . Accounting for the adversarial corruption, which can affect at most ηn ≤ log 1 δ buckets, at most 9log 1 δ buckets can have |µ i -µ| ≤ 4σ 16log 1 δ n before adversarial corruption, which happens with probability at most 1 22 δ as desired.
this section cite: []

Section: References
Ref_id:b0 Title: The space complexity of approximating the frequency moments Year: (1999)
Ref_id:b1 Title: Near instance-optimality in differential privacy Year: (2020)
Ref_id:b2 Title: Instance-optimality in differential privacy via approximate inverse sensitivity mechanisms Year: (2020)
Ref_id:b3 Title: Bandits with heavy tail Year: (2012)
Ref_id:b4 Title: Challenging the empirical mean and empirical variance: a deviation study Year: (2012)
Ref_id:b5 Title: Optimality in mean estimation: Beyond worst-case, beyond subgaussian, and beyond 1 + α moments Year: (2023)
Ref_id:b6 Title: Robust sub-gaussian estimation of a mean vector in nearly linear time Year: (2022)
Ref_id:b7 Title: Sub-Gaussian mean estimators Year: (2016)
Ref_id:b8 Title: Algorithmic highdimensional robust statistics Year: (2023)
Ref_id:b9 Title: Outlier robust mean estimation with subgaussian rates via stability Year: (2020)
Ref_id:b10 Title: Mean estimation for randomized quasi monte carlo method Year: (2022)
Ref_id:b11 Title: Robust and heavy-tailed mean estimation made simple, via regret minimization Year: (2020)
Ref_id:b12 Title: Instance-optimal mean estimation under differential privacy Year: (2021)
Ref_id:b13 Title: Robust estimation of a location parameter Year: (1992)
Ref_id:b14 Title: Random generation of combinatorial structures from a uniform distribution Year: (1986)
Ref_id:b15 Title: Generalization bounds in the presence of outliers: a median-ofmeans study Year: (2021)
Ref_id:b16 Title: Optimal sub-Gaussian mean estimation in R Year: (2022)
Ref_id:b17 Title: Robust multivariate mean estimation: The optimality of trimmed mean Year: (2021)
Ref_id:b18 Title: U-statistics of growing order and sub-gaussian mean estimators with sharp constants Year: (2023)
Ref_id:b19 Title: Problem Complexity and Method Efficiency in Optimization Year: (1983)
Ref_id:b20 Title: Finitesample properties of the trimmed mean Year: (2025)
Ref_id:b21 Title: A survey of sampling from contaminated distributions. Contributions to probability and statistics Year: (1960)
