Title: Private Set Union with Multiple Contributions
Abstract: In the private set union problem each user owns a bag of at most k items (from some large universe of items), and we are interested in computing the union of the items in the bags of all of the users. This is trivial without privacy, but a differentially private algorithm must be careful about reporting items contained in only a small number of bags. We consider differentially private algorithms that always report a subset of the union, and define the utility of an algorithm to be the expected size of the subset that it reports. Because the achievable utility varies significantly with the dataset, we introduce the utility ratio, which normalizes utility by a dataset-specific upper bound and characterizes a mechanism by its lowest normalized utility across all datasets. We then develop algorithms with guaranteed utility ratios and complement them with bounds on the best possible utility ratio. Prior work has shown that a single algorithm can be simultaneously optimal for all datasets when k = 1, but we show that instance-optimal algorithms do not exist when k > 1, and characterize how performance degrades as k grows. At the same time, we design a private algorithm that achieves the maximum possible utility, regardless of k, when the item histogram matches a prior prediction (for instance, from a previous data release) and degrades gracefully with the ℓ ∞ distance between the prediction and the actual histogram when the prediction is imperfect.

Section: Introduction
Consider a dataset where each entry is a set of items donated by a different user. The set union problem is to output the union of all of the sets. This simple problem arises in many practical scenarios, and when the items have the potential to be sensitive we may want privacy guarantees to ensure that the result does not reveal personal data. For example, private set union can be used for discovering n-grams in a corpus [Gopi et al., 2020], releasing keys in SQL queries [Wilson et al., 2020], and in general for determining the domain of private aggregate statistics [Amin et al., 2022].
Since the number of conceivable items (e.g., all possible n-grams) can be very large, it is often necessary for the algorithm to restrict its output to a subset of the true union [Gopi et al., 2020, Desfontaines et al., 2022]. Motivated by this, Cohen et al. [2021], Desfontaines et al. [2022] proposed an optimal (ε, δ)-differentially private algorithm when each user contributes exactly one item. However, in many realistic settings users can contribute multiple items. This prompts a natural question: can we design an optimal (ε, δ)-differentially private algorithm when each user contributes up to k items?
We begin with the definition of differential privacy.  0 2 4 6 8 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Utility Ratio 1/2 1/3 1/4 k = 2 k = 3 k = 4
Figure 2: Optimal utility ratios over datasets with three users for δ = 0.01 and various settings of ε and k. For intermediate values of ε the ratio can be nearly as low as 1/k.
Definition 1.1 (Differential privacy [Dwork et al., 2006]). A randomized algorithm M satisfies (ε, δ)-differential privacy if for any two neighboring datasets D, D ′ and for any subset of the output space S, it holds that Pr[M (D) ∈ S] ≤ e ε • Pr[M (D ′ ) ∈ S] + δ.
Two datasets D and D ′ are neighboring if and only if
d ham (D, D ′ ) ≜ |D \ D ′ | + |D ′ \ D| = 1.
We can now provide a formal definition of the differentially private set union problem. Definition 1.2 (Differentially private set union). Fix a universe X of items and a contribution bound k. Let D be a dataset consisting of bags B i ⊆ X , |B i | ≤ k for i ∈ [n]. A differentially private set union algorithm M has to output a subset of ∪ i∈[n] B i , and its goal is to output a subset which is as large as possible. We denote by UNION k (ε, δ) the set of all (ε, δ)-differentially private set union algorithms.
We define the utility of algorithm M ∈ UNION k (ε, δ) on dataset D to be the expected cardinality of its output set, E [|M (D)|]. In the best-case scenario, the utility is equal to the cardinality of the full union, i.e., E[|M (D)|] = | ∪ i∈[n] B i |. However, the best-case utility is typically not achievable. For example, consider a dataset in which each item is contained in the bag of a single user: any algorithm in UNION k (ε, δ) cannot report more than a δ fraction of the items in expectation, since for each item there is a neighboring dataset in which it does not exist and hence is reported with probability zero.
In general, the achievable fraction of the best-case utility is highly dependent on the item frequencies, making it difficult to compare algorithms across datasets when the goal is a naive maximization of
E[|M (D)|].
The work of Cohen et al. [2021], Desfontaines et al. [2022] suggests an appropriate adjustment. Let the number of times item x appears in dataset D be denoted by c(x, D) (or simply c(x) when the underlying dataset is clear). They showed that the utility of any algorithm
M ∈ UNION k (ε, δ) satisfies E[|M (D)|] ≤ Π(D, ε, δ) := x∈X π(c(x); ε, δ),
where π is a sigmoid-like function given by
π(c(x); ε, δ) =      e c(x)ε -1 e ε -1 • δ if c(x) ≤ c ℓ (1 -e -(c(x)-c ℓ )ε ) 1 + δ e ε -1 + e -(c(x)-c ℓ )ε π(c ℓ ) if c ℓ < c(x) ≤ c h 1
otherwise and c ℓ and c h are constants given by
c ℓ = 1 + 1 ε ln e ε + 2δ -1 (e ε + 1)δ c h = c ℓ + 1 ε ln 1 + e ε -1 δ (1 -π(c ℓ , ε, δ)) .(1)
In the special case ε = 0, we have π(c(x); 0, δ) = min(c(x)δ, 1). See Figure 1 for an illustration of π.
Cohen et al. [2021], Desfontaines et al. [2022] showed that the upper bound of Π(D, ε, δ) is achievable, simultaneously for all datasets, when k = 1. We will see later that this is not possible when k > 1 (except under certain extreme values of ε and δ). However, Π(D, ε, δ) is achievable, regardless of k, for any single dataset (see Theorem 1.3 below). This is not trivial since algorithms in UNION k (ε, δ) can only return items appearing in their input dataset, which rules out constant algorithms that ignore their input and return a fixed result. (Such algorithms are differentially private and, in other settings, can be used to trivially obtain optimality for any single dataset.)
The achievability of Π(D, ε, δ) when k = 1 motivates its use as a normalizer for the utility E[|M (D)|].
We introduce the following target measure, which we use to establish bounds on the performance of algorithms in UNION k (ε, δ).
this section cite: ['b15', 'b22', 'b15', 'b8', 'b7', 'b8', 'b10', 'b7', 'b8', 'b8']

Section: Definition 1.3 (Utility ratio).
The utility ratio of an algorithm M ∈ UNION k (ε, δ) is u k (M ) := min
D∈D k E M [|M (D)|] Π(D, ε, δ) ,
where D k is the collection of all nonempty datasets where each user contributes at most k items.
That u k (M ) is generally less than one when k > 1 is easily demonstrated numerically using a linear program that finds the optimal mechanism for a finite collection of datasets. Figure 2 shows that the resulting behavior is complex, even considering only very small datasets, and the worst-case utility ratio appears to be close to 1/k. Our aim is to characterize this behavior theoretically.
this section cite: []

Section: Main results
Theorem 1.1 (Informal impossibility results). Let δ = O ε (1/k 2 ), where the subscript denotes an unstated dependence on ε. Then for any algorithm M in UNION k (ε, δ) we have
u k (M ) = O 1 k 1 + ln k ε . (Theorem 2.2)
In addition, even if D is restricted to "easy" datasets where Π(D, ε, δ) = Ω(|X |), we still have
u k (M ) = Õn 1 k 1/4 ,
this section cite: []

Section: (Theorem 2.3)
where Õ hides logarithmic terms.
Theorem 1.1 shows that instance-optimal algorithms are not generally possible when k > 1, with the bounds roughly matching the minimums in Figure 2. However, we can still construct algorithms with meaningful utility guarantees.
this section cite: []

Section: Theorem 1.2 (Informal achievability results).
There exists an algorithm M in UNION k (ε, δ) such that for every dataset D,
E[|M (D)|] = 1 k • Π(D, ε ′ , δ ′ ),
where ε ′ = Ω(ε) and δ ′ = Ω(δ/e ε ) (see Theorem 3.2). In addition, when ε = 0 or ε → ∞ (holding δ constant), there exists an M such that u k (M ) = 1 (see Lemmas 3.5 and 3.6).
The results above raise an important question: can we ever do better than a utility ratio of O(1/k)? Theorem 1.3 shows that, if we can predict the histogram of items in the dataset in advance, then there exists a private set union algorithm achieving the optimal utility regardless of k.
Theorem 1.3 (Informal achievability with predictions). For any nonempty histogram over X and privacy parameters ε and δ, there exists an algorithm M ∈ UNION k (ε, δ) such that
E [|M (D)|] = Π(D, ε, δ)
for any D matching the predicted histogram, regardless of the contribution bound k (Theorem 4.1).
The private algorithm M satisfying Theorem 1.3 also performs well on datasets that are "similar" to the target dataset D, making it an appropriate algorithm for settings where some public prediction regarding the union is available (see Section 4 for more details).
A note about k: In some settings we may not have any a priori contribution bound k, in which case we need to choose one and enforce it. This introduces a natural tradeoff: larger k retain more data, but reduce the utility ratio (as indicated by our results). Similar contribution-bounding tradeoffs have been explored in prior work [Amin et al., 2019, Epasto et al., 2020, Amin et al., 2022]. Although it is not our main focus here, in Appendix A we show one way that k can be selected privately when no contribution bound is known in advance.
this section cite: ['b0', 'b13']

Section: Related work
The differentially private set union problem was implicitly introduced by Korolova et al. [2009] in the early days of differential privacy. Subsequent work by Gopi et al. [2020], Carvalho et al. [2022] improved utility by processing users sequentially and choosing contributions in a clever way, minimizing waste on heavy items while maintaining low sensitivity. Swanberg et al. [2023], Chen et al. [2024] proposed multi-round mechanisms with careful budget-splitting across the rounds; this allowed them to process users in parallel while retaining good utility. However, none of these works provide worst case utility guarantees, and they primarily compare different approaches empirically.
The optimal reporting probabilities when k = 1 were introduced by Desfontaines et al. [2022], Cohen et al. [2021]. [Knop and Steinke, 2023] studied the related problem of estimating the size of the union rather than the union itself. More distantly related work on privately finding the k most frequent items in a database was published by Bhaskar et al. [2010], Durfee and Rogers [2019], McKenna and Sheldon [2020], Gillenwater et al. [2022].
this section cite: ['b18', 'b15', 'b21', 'b5', 'b8', 'b7', 'b17', 'b2', 'b9', 'b19', 'b14']

Section: Impossibility results
In this section we derive upper bounds on the utility ratio of any (ε, δ)-differentially private set union mechanism when applied to datasets with contribution bound k. Our upper bounds all diminish with k, showing that for datasets with large contribution bounds, no differentially private mechanisms can meaningfully compete with the optimal utility simultaneously for all datasets. All omitted proofs are given in Appendix B.
Our first result shows that there exist regimes for ε and δ such that every set union mechanism has utility ratio O(1/k). In particular, as the contribution bound k grows, no mechanism is competitive with Π(D) on every dataset D, despite the fact that Theorem 1.3 establishes a mechanism matching Π(D) for any single D.
Theorem 2.1 (Warm-up). Let k ≥ 2, ε ≥ 0, δ ≤ 1 e ε +2
, and let M be any (ε, δ)-differentially private set union mechanism. Then there exists a dataset D with contribution bound k such that
E |M (D)| Π(D, ε, δ) ≤ 1 k + k e ε + k .
In particular, for ε = 2 ln(k) and δ
≤ 1 k 2 +2 , we have E[|M (D)|] Π(D,ε,δ) ≤ 2 k .
Proof sketch. Let the item universe X contain k items, and let D be the dataset with a single user that contributes every item, i.e., B 1 = X . For each item x ∈ X , construct a dataset D x by adding a second user to D that contributes only item x, i.e., B 2 = {x}. The privacy parameters ε and δ are chosen to ensure that π(2) is much larger than π(1), so a mechanism M is only competitive on D x if it outputs item x with probability close to π(2). However, since D x neighbors a dataset containing only item x (after removing user 1), the total probability mass of outputting any set containing an item other than x must be at most δ.
Thus, Pr x ∈ M (D x ) ≤ Pr M (D x ) = {x} + δ.
And, since M is DP, we further have Pr x ∈ M (D x ) ≤ e ε Pr M (D) = {x}) + 2δ. So, for M to compete with Π(D x ), we require that M outputs the singleton set {x} with non-trivial probability when run on the single-user dataset D. On the other hand, D neighbors the empty dataset, so the total probability mass it assigns to non-empty outputs is at most δ, implying that there exists an item y such that Pr(M (D) = {y}) ≤ δ/k. Therefore, Pr y ∈ M (D y ) ≤ e ε δ/k + 2δ. By contrast, in the specified parameter regime we have π(2) = δe ε + δ. When ε is sufficiently large that the e ε δ terms dominates, the mechanism M is only able to output y with probability approximately π(2)/k.
Intuitively, the only way for M to compete with Π(D x ) is for M to have an output distribution on the single-user dataset D that prioritizes x, and it is not possible for a single mechanism M to do this simultaneously for all x ∈ X .
A weakness of Theorem 2.1 is that the utility ratio bound of 2/k holds only when ε ≥ 2 ln k. For large k, this is an extremely low privacy regime. The next result extends the argument of Theorem 2.1 and establishes a bound of O( ln k kε ) on the utility ratio of any mechanism that holds in almost any privacy regime. In particular, it holds for any ε as long as δ decays like 1/k 2 .
Theorem 2.2. Let k ≥ 4, ε ≥ 0, δ ≤ 1 k 2 • 1 e ε/2 • e ε -1 e ε +1
and M be any (ε, δ)-differentially private set union mechanism. Then there exists a dataset D with contribution bound k such that
E[|M (D)|] Π(D, ε, δ) ≤ 12 k -1 1 + 1 ε log k .
Proof sketch. The proof follows a similar argument to the one for Theorem 2.1, but instead of adding a single user to D, we add O(ln(k)) users, each contributing a constant fraction of the previous user's items. The key advantage of this iterative construction is that the suboptimality incurred by the mechanism is determined by its inability to output items with sufficiently large probability across a range of item counts from 1 to ln(k). In particular, rather than requiring ε to be O(ln(k)) to ensure that π(2) is much larger than π(1), here we allow for constant ε and drive suboptimality from the ratio between π(ln(k)) and π(1).
The datasets that witness the utility ratio upper bound in Theorem 2.2 have the property that Π(D, ε, δ)/| ∪ i∈[n] B i | tends to zero as the contribution bound k grows. In other words, even the optimal mechanisms for datasets D established by Theorem 1.3 are only able to return a vanishing fraction of the items contained in D as the contribution bound k grows. Our final impossibility result shows that even on a class of datasets where Π(D) ≥ | ∪ i∈[n] B i |/2, the utility ratio achievable by any mechanism diminishes with k, albeit at a slower rate than for the previous results.
Theorem 2.3. Let k ≥ 2, n > 2 • c h (so that π(n/2) = 1), ε ≥ 1/n, and δ < 1/(40e ε n 1/2 k 1/4 ).
Let M be any (ε, δ)-differentially private set union mechanism. Then there exists a dataset D with contribution bound k such that
E[|M (D)|] Π(D, ε, δ) = O n 2 log(nk) k 1/4 .
Proof sketch. The key idea is a reduction showing that a private set union mechanism M can be used to construct a mechanism for estimating matrix marginals whose performance is related to the utility ratio u k (M ). Combined with an impossibility result for privately estimating matrix marginals based on robust fingerprinting codes (modified from the work of Steinke and Ullman [2015]), this yields a bound on the utility ratio.
The marginal problem we reduce from is the following: given a binary matrix C ∈ {0, 1} n×k , the mechanism aims to output a vector in {0, 1} k such that whenever a column of C is entirely 0 or 1, the corresponding component of the output vector is also equal to 0 or 1 (respectively). On mixed columns of C, the mechanism can output either 0 or 1. This is an easier problem than computing the column marginals of C (i.e., the fraction of 1s per column), since the mechanism is only required to identify "pure" columns. We are interested in mechanisms that are at most (β, γ)-inaccurate, which requires that with probability at least γ their output is correct on all but at most βk columns. Steinke and Ullman [2015] upper bound β for differentially private mechanisms.
The reduction works as follows: view row i of the matrix C as the indicator vector for user u i 's bag of items from a universe of size k. Given a mechanism M for private set union, we obtain a set Û approximating the union of the contributed items. We then output the vector m ∈ {0, 1} k where mj = 1 if j ∈ Û and mj ∼ Bernoulli(1/2) if j ̸ ∈ Û . Because M ∈ UNION k (ε, δ), every index j ∈ Û must be a column of C that contains at least one 1, so we never make mistakes on those columns. And for each pure column j ̸ ∈ Û , we have a 1/2 chance of correctly guessing whether the column was all 0s or all 1s. It follows that the expected number of mistakes made by the reduction mechanism is at most (k -u k (M ) • Π(D))/2, where D is the set union instance encoded by the rows of C. To finish the proof, we construct C to ensure that Π(D) ≥ k/2, which ensures the expected fraction of marginal mistakes is bounded in terms of u k (M ). Then we convert this to a high probability bound that contradicts the impossibility result for the marginal problem when u k (M ) is too large.
this section cite: ['b20', 'b20']

Section: Algorithms with utility guarantees

this section cite: []

Section: A simple budget splitting algorithm
A straightforward approach when k > 1 is to divide the budget by k and apply the optimal k = 1 algorithm, including each item x ∈ X in the output independently with probability π(c(x, D); ε/k, δ/k).
Clearly the utility of this mechanism is Π(D, ε/k, δ/k). The following lemma argues that it is private. The proof is straightforward and included for completeness in Appendix C. Lemma 3.1. Let M split (D; ε, δ, k) be the mechanism that works as follows: for each item x ∈ X , include x in the output with probability π(c(x, D); ε/k, δ/k). Then M split is a (ε, δ)-differentially private set union mechanism when users contribute at most k items.
this section cite: []

Section: Bicriteria Approximation
The simple budget splitting algorithm achieves the Π(D, ε, δ) bound of Theorem 1.3 for every dataset D but with privacy parameters smaller by a factor of k than the target parameters. In the following theorem we compete with this bound for a larger value of ε, smaller than the "real" ε by only a factor of ln(1/δ). This gain comes with a multiplicative loss of 1/k over the Π(D, ε, δ) bound. Theorem 3.2. Let ε, δ < 1 be small enough constants. There exists an (ε, δ)-DP algorithm whose expected number of identified items is
1 k • Π D, Ω ε ln(1/δ)
, Ω δ ln(1/δ)e ε .
We refer to this result as "bicriteria" because our (ε, δ)-DP algorithm incurs a multiplicative loss of 1 k when compared not with the optimal reporting probabilities for parameters (ε, δ), but rather with those for the relaxed parameters ε ln(1/δ) , δ ln(1/δ)e ε . Our bicriteria algorithm, called Bicrit, is given below. We present an alternative construction of a bicriteria algorithm in Appendix C.1.
Algorithm 1 Bicrit Notation: Let k denote the contribution bound, let X be a domain of items, and let ∆ X ,k = {B ⊆ X : |B| ≤ k} denote the set of all possible bags of size at most k from X . Input: Dataset D ∈ (∆ X ,k ) n containing n bags, privacy parameters ε, δ > 0.
1. Denote ε = ε 4 ln(2/δ) and δ = δ 8 log(2/δ)e ε 2. For each x ∈ X : (a) Let b x ← Bernoulli 1 k (b) If b x = 1
then report x with probability π(c(x); ε, δ)
Note that Algorithm Bicrit does not need to explicitly traverse all x ∈ X ; we can skip items to which no user contributes since π(0; ε, δ) = 0.
The next lemma captures the privacy guarantee of Algorithm Bicrit. Lemma 3.3. Algorithm Bicrit is (ε, δ)-DP.
Proof. Fix two neighboring datasets D 0 and
D 1 = D 0 ∪ {B} for B = {x 1 , x 2 , . . . , x z } where z ≤ k. Let ℓ = |{x ∈ B : b x = 1}|
be the random variable denoting the number of elements from B that are sampled in Step 2a. Let E denote the event that ℓ ≤ ℓ 0 := 4 ln(2/δ), and Ē its complement. By the Chernoff bound we have Pr Ē ≤ δ/2. Now, by composition (and by our choice of ε and δ in Step 1), for any outcome event F we have that
Pr[Bicrit(D 0 ) ∈ F ] = Pr[E] • Pr[Bicrit(D 0 ) ∈ F |E] + Pr Ē • Pr Bicrit(D 0 ) ∈ F | Ē ≤ Pr[E] e εℓ0 • Pr[Bicrit(D 1 ) ∈ F |E] + ℓ 0 e (ℓ0-1)ε δ + δ 2 ≤ Pr[E] e ε • Pr[Bicrit(D 1 ) ∈ F |E] + δ 2 + δ 2 ≤ e ε • Pr[Bicrit(D 1 ) ∈ F ] + δ.
The utility analysis of the bicriteria algorithm is straightforward:
Lemma 3.4. The expected number of identified items in Algorithm Bicrit is
1 k • Π D, Ω ε ln(1/δ)
, Ω δ ln(1/δ)e ε .
Proof. For any dataset D we have
E[|Bicrit(D)|] = x∈X 1 k • π(c(x); ε, δ) = 1 k • Π D, ε, δ = 1 k • Π D, ε 4 ln(2/δ)
, δ 8 ln(2/δ)e ε .
this section cite: []

Section: Optimal Mechanisms in Extreme Privacy Regimes
Finally, we describe some mechanisms that behave optimally when the privacy parameter ε is extremely large or small. Small ε regime. When ε = 0 the π function takes a particularly simple form: π(c; 0, δ) = min(cδ, 1). The following lemma gives a mechanism M u that matches these output probabilities as long as δ < 1/n, where n is the number of users. Its proof is straightforward and omitted.
Lemma 3.5. Let n be the number of users and assume that δ < 1/n. Let M 0 (D; δ) be a mechanism that with probability nδ picks a user i uniformly at random and outputs the set of items in B i and with probability 1 -nδ outputs the empty set. Then M 0 is (0, δ) differentially private, and it outputs each item x with probability c(x, D) • δ = π(c(x, D); 0, δ).
Large ε regime. We describe a mechanism that achieves the optimal utility Π(D, ε, δ) as ε → ∞. The mechanism composes the budget splitting mechanism of Section 3.1 with a simple mechanism M all that outputs the full union with probability δ and otherwise outputs the empty set. Importantly, M all outputs items that appear exactly once with the maximum possible probability π(1; ε, δ) = δ. Lemma C.7 in Appendix C.2 shows that M all is a (0, δ)-differentially private.
The intuition underlying the combination of M all and M split is as follows. For any dataset D, M split outputs each item with probability π(c(x, D); ε/k, δ/k) which is smaller than π(c(x, D); ε, δ). However, for all items that appear at least twice, both probabilities converge to 1 in the limit as ε → ∞. The only catch is that M split outputs items appearing exactly once with probability δ/k instead of δ (regardless of ε). To fix this, we compose M all and M split , spending most of our δ budget on M all to get the maximum output probabilities for items that appear once, and relying on the fact that for any nonzero δ and count c ≥ 2, we have lim ε→∞ π(c; ε, δ) = 1. The final mechanism and its properties are summarized in the following lemma, which we prove in Appendix C.2. Lemma 3.6. Let M large (D; ε, δ, k) be the following mechanism: let δ ′ = δ -min(δ, 1/ε) and output the union of M all (D; δ ′ ) and M split (D; ε, δ -δ ′ , k). Then M large is an (ε, δ)-differentially private set union mechanism. Furthermore, for any contribution bound k, dataset D with contributions bounded by k, and privacy parameter δ, we have that
lim ε→∞ E[|M large (D; ε, δ, k)|] Π(D; ε, δ) = 1.
this section cite: []

Section: Leveraging a prediction
Finally, in this section, we study whether predicted information about the underlying dataset D, e.g., based on historical runs, can improve the utility for private set union algorithms. In particular, we consider the case where a predicted histogram H for the item counts is available. Our goal is to perform well on datasets whose histogram is close to H.
The requirement in Definition 1.2 that the algorithm must output a subset of the input dataset excludes the trivially successful algorithm that always outputs the union {x | H(x) > 0}. However, somewhat surprisingly, we show that if the predicted histogram H is correct, it is possible to design a private set union algorithm M H that achieves the best possible expected utility Π(D, ε, δ), regardless of the contribution bound k. For any dataset D, let H D be its histogram where ∀x, H D (x) = c(D, x). Note that the optimal utility bound Π(D, ε, δ) only depends on H D . We abuse notation and define
Π(H, ε, δ) = x∈X π(H(x); ε, δ),
and we have ∀D, Π(D, ε, δ) = Π(H D , ε, δ). Moreover, for any d > 0 and histogram H, we define
Π -d (H, ε, δ) := x∈X π(H(x) -d; ε, δ)
to be the Π bound when all item counts have been reduced by d. The result is stated below. Theorem 4.1. Let H be a predicted histogram. Then there exists an (ε, δ)-private set union mechanism M H such that
E[|M H (D)|] = Π -ℓ∞(H D ,H) (H, ε, δ), where in particular we have E[|M H (D)|] = Π(D, ε, δ) if H D = H.
Proof. Given a predicted histogram H, we construct the mechanism M H as follows. Compute
d = ℓ ∞ (H D , H) = max x |H D (x) -H(x)| and sample p ∼ U (0, 1). Then M H outputs the set M H (D) = {x | π(H(x) -d; ε, δ) > p}.
The utility guarantee follows by noting that
E[|M (D)|] = x∈X Pr p < π(H(x) -d; ε, δ) = x∈X π(H(x) -d; ε, δ) = Π -ℓ∞(H D ,H) (H, ε, δ).
It remains to prove that the algorithm is private. Since π(c -d; ε, δ) is a monotonically increasing function of c, the output of the algorithm is determined by c D , defined as the smallest c such that π(c -d; ε, δ) ≥ p. To see this, note that we can get M H (D) by post-processing c D and outputting the set {x | H(x) > c D }. Hence it is sufficient to prove that c D is a private statistic of D.
Note that c D only depends on D through d = ℓ ∞ (H D , H), and by the definition of c D , we have Pr(c D = m) = π(m -d) -π(m -d -1).
We denote the distribution of c D when d = ℓ ∞ (H D , H) as P d . For all neighboring datasets D and D ′ , by the reverse triangle inequality we have
|ℓ ∞ (H D , H) -ℓ ∞ (H D ′ , H)| ≤ ℓ ∞ (H D , H D ′ ) ≤ 1.
Hence it is sufficient to prove that ∀d ≥ 0, P d and P d+1 are (ε, δ)-indistinguishable. More precisely, we want to prove that for d ≥ 0, we have
Pr m∼P d (P d (m) ≤ e ε P d+1 (m)) ≥ 1 -δ(2)
and
Pr m∼P d+1 (P d+1 (m) ≤ e ε P d (m)) ≥ 1 -δ.(3)
By the definition of P d , for all m ′ ≥ 0, we have
P d+m ′ (m + m ′ ) = P d (m).
This implies
Pr m∼P d (P d (m) ≤ e ε P d+1 (m)) = Pr m∼P0 (P 0 (m -d) ≤ e ε P 1 (m -d)) and Pr m∼P1 (P 1 (m -d) ≤ e ε P 0 (m -d)) = Pr m∼P1 (P 1 (m -d) ≤ e ε P 0 (m -d)) .
Hence it is sufficient to prove Equation (2) and Equation (3) for d = 0.
By [Desfontaines et al., 2022, Lemma 1], we have that there exist c ℓ and c h such that
π(c +1, ε, δ) =        0, if c ≤ 0 e ε π(c, ε, δ) + δ, if 0 < c ≤ c ℓ , 1 -e -ε (1 -π(c, ε, δ) -δ), if c ℓ < c ≤ c h , 1, if c > c h .
Moreover, the above implies π(1, ε, δ) = δ, π(c h , ε, δ) ∈ [1 -δ, 1).
We start by proving Equation (2). We show that
∀m ≥ 2, P 0 (m) ≤ e ε P 1 (m).(4)
Since, in addition, Pr m∼P0 (m ≤ 1) = P 0 (1) = π(1, ε, δ) -π(0, ε, δ) = δ, Equation (2) holds.
To see Equation ( 4), when 0 ≤ m -2 ≤ c ℓ , we have
P 0 (m) = π(m, ε, δ) -π(m -1, ε, δ) = π(m, ε, δ) -(e ε π(m -2, ε, δ) + δ) ≤ (e ε π(m -1, ε, δ) + δ) -(e ε π(m -2, ε, δ) + δ) (5) ≤ e ε (π(m -1, ε, δ) -π(m -2, ε, δ)) = e ε P 1 (m),
where Equation ( 5) is due to the (ε, δ)-DP guarantee of π.
If c h ≥ m -2 > c ℓ , we have
P 0 (m) = π(m, ε, δ) -π(m -1, ε, δ) = π(m, ε, δ) -(1 -e -ε (1 -π(m -2, ε, δ) -δ)) ≤ (1 -e -ε (1 -π(m -1, ε, δ) -δ)) -(1 -e -ε (1 -π(m -2, ε, δ) -δ))(6)
≤ e -ε (π(m -1, ε, δ) -π(m -2, ε, δ)) = e -ε P 1 (m),
where Equation (6) follows since, by the (ε, δ)-DP guarantee of π, we have
1 -π(m -1, ε, δ) ≤ e ε (1 -π(m -1, ε, δ)) + δ. For m -2 > c h , we have P 0 (m) = π(m, ε, δ) -π(m -1, ε, δ) = 0.
Combining the three cases completes the proof of Equation (2).
To prove Equation (3), we similarly need to show that
∀m ≤ c h +1, P 1 (m) ≤ e ε P 0 (m),(7)
and then since Pr
m∼P1 (m ≥ c h +2) = P 1 (c h +2) = π(c h +1, ε, δ) -π(c h , ε, δ) ≤ 1 -(1 -δ) = δ, Equation(
3) will follow. The proof of Equation (7) follows the proof of Equation (4) and is omitted here.
NeurIPS Paper Checklist 1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The abstract and introduction provide accurate summaries of the major claims made in the paper.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper. 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The limitations of the results are captured in the formal theorem statements, and we emphasize the differences between various analyses and the settings they apply to in the text of the paper.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: ['b8']

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All nontrivial proofs are provided in the appendices.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [NA] Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [NA]
Justification: The paper does not include experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The paper and its underlying research conform to the code of ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA]
Justification: The paper is not expected to have any immediate societal impact.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA]
Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets ( if applicable). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing or research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Bounding user contributions: A bias-variance trade-off in differential privacy Year: (2019)
Ref_id:b1 Title:  Year: ()
Ref_id:b2 Title: Discovering frequent patterns in sensitive data Year: (2010)
Ref_id:b3 Title: Make up your mind: The price of online queries in differential privacy Year: (2017)
Ref_id:b4 Title: Incorporating item frequency for differentially private set union Year: (2022)
Ref_id:b5 Title: Scalable private set union beyond uniform weighting Year: (2024)
Ref_id:b6 Title: The target-charging technique for privacy analysis across interactive computations Year: (2023)
Ref_id:b7 Title: Differentially private weighted sampling Year: (2021)
Ref_id:b8 Title: Differentially private partition selection Year: (2022)
Ref_id:b9 Title: Practical differentially private top-k selection with pay-what-youget composition Year: (2019)
Ref_id:b10 Title: Our data, ourselves: Privacy via distributed noise generation Year: (2006)
Ref_id:b11 Title: On the complexity of differentially private data release: efficient algorithms and hardness results Year: (2009)
Ref_id:b12 Title: The algorithmic foundations of differential privacy Year: (2014)
Ref_id:b13 Title: Smoothly bounding user contributions in differential privacy Year: (2020)
Ref_id:b14 Title: A joint exponential mechanism for differentially private top-k Year: (2022)
Ref_id:b15 Title: Differentially private set union Year: (2020)
Ref_id:b16 Title: The sparse vector technique, revisited Year: (2021)
Ref_id:b17 Title: Counting distinct elements under person-level differential privacy Year: (2023)
Ref_id:b18 Title: Releasing search queries and clicks privately Year: (2009)
Ref_id:b19 Title: Permute-and-flip: a new mechanism for differentially private selection Year: (2020)
Ref_id:b20 Title: Interactive fingerprinting codes and the hardness of preventing false discovery Year: (2015)
Ref_id:b21 Title: DP-SIPS: A simpler, more scalable mechanism for differentially private partition selection Year: (2023)
Ref_id:b22 Title: Differentially private SQL with bounded user contribution Year: (2020)
