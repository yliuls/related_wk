Title: MISSING MASS FOR DIFFERENTIALLY PRIVATE DOMAIN DISCOVERY
Abstract: We study several problems in differentially private domain discovery, where each user holds a subset of items from a shared but unknown domain, and the goal is to output an informative subset of items. For set union, we show that the simple baseline Weighted Gaussian Mechanism (WGM) has a near-optimal ℓ 1 missing mass guarantee on Zipfian data as well as a distribution-free ℓ ∞ missing mass guarantee. We then apply the WGM as a domain-discovery precursor for existing known-domain algorithms for private top-k and k-hitting set and obtain new utility guarantees for their unknown domain variants. Finally, experiments demonstrate that all of our WGM-based methods are competitive with or outperform existing baselines for all three problems.

Section: INTRODUCTION
Modern data analysis often requires working in data domains like queries, reviews, and purchase histories that are a priori unknown or impractically large (e.g., the set of all strings up to a fixed length). For these datasets, domain discovery is a critical first step for efficient downstream applications. Differential privacy (Dwork et al., 2006) (DP) enables privacy-preserving analysis of sensitive data, but it complicates domain discovery.
In the basic problem of set union, for example, each user has a set of items, and the goal is simply to output as many of these items as possible. This is a necessary step before further analysis, so set union (also known as key selection or partition selection) is a core component of several industrial (Wilson et al., 2020;Rogers et al., 2021;Amin et al., 2023) and open source (OpenDP, 2025) DP frameworks. For similar reasons, there are by now many DP set union algorithms in the literature. However, there are almost no provable utility guarantees (see Section 1.1). This makes it difficult to understand how well existing algorithms work, or how much they can be improved.
this section cite: ['b9', 'b26', 'b22']

Section: Our Contributions.
We prove utility guarantees for several problems in DP domain discovery. First, by reframing DP set union in terms of mass instead of cardinality (i.e., the fraction of all items recovered, rather than the number of unique items), we prove utility guarantees for the simple and scalable Weighted Gaussian Mechanism (Gopi et al., 2020) (WGM). We first show that the WGM has near-optimal ℓ 1 missing mass on Zipfian data (Theorem 3.3). We then prove a similar but distribution-free ℓ ∞ missing mass guarantee (Theorem 3.6).
Next, we build on these results by considering unknown domain variants of top-k and k-hitting set and obtain further utility guarantees for simple algorithms that run WGM to compute a baseline domain and then run a standard known-domain algorithm afterward. Relying on Theorem 3.6 enables us to prove utility guarantees for both top-k (Theorem 4.3) and k-hitting set (Theorem 4.5).
Finally, we evaluate our algorithms against the existing state of the art on six real-world datasets from varied domains (Section 5). These experiments demonstrate that, in addition to their theoretical guarantees, our WGM-based methods obtain strong empirical utility.
this section cite: ['b12']

Section: RELATED WORK

this section cite: []

Section: DP Set Union.
Early work by Korolova et al. (2009) introduced the core idea of collecting a bounded number of items per user, constructing a histogram of item counts, and releasing items whose noisy counts exceed a carefully chosen threshold. Desfontaines et al. (2022) developed an optimal algorithm for the restricted setting where each user contributes a single item. Gopi et al. (2020) adapted noisy thresholding in the Weighted Gaussian Mechanism (WGM) by scaling contributions to unit ℓ 2 norm. Swanberg et al. (2023) investigated a repeated version of WGM, and Chen et al. (2025) further built on these ideas by incorporating adaptive weighting to determine user contributions, and proved that the resulting algorithm dominates the WGM (albeit by a small margin, empirically). A separate line of work has studied sequential algorithms that attempt to choose user contributions adaptively, obtaining better empirical utility at the cost of scalability (Gopi et al., 2020;Carvalho et al., 2022).
We note that, as the utility results of Desfontaines et al. (2022) and Chen et al. (2025) are stated relative to other algorithms, our work is, to the best of our knowledge, the first to prove absolute utility guarantees for DP set union.
DP Top-k. While several algorithms have been proposed for retrieving a dataset's k most frequent items given a known domain (Bhaskar et al., 2010;McKenna & Sheldon, 2020;Qiao et al., 2021;Gillenwater et al., 2022), to the best of our knowledge only Durfee & Rogers (2019) provide an algorithm for the unknown domain setting (see discussion in Section 5.2). They also provide a utility guarantee in terms of what Gillenwater et al. (2022) call k-relative error, which bounds the gap between the smallest-count output item and the k th highest-count item. In contrast, we prove a utility result for the more stringent notion of missing mass (see discussion in Section 4.1).
DP k-Hitting Set. In k-hitting set, the objective is to output a set of k items that maximizes the number of users whose subset intersects with it. This problem can also be viewed as an instance of cardinality-constrained submodular maximization. Previous works on private submodular maximization by Mitrovic et al. (2017) and Chaturvedi et al. (2021) establish approximation guarantees in the known-domain setting. However, they are not directly applicable when the domain is unknown.
this section cite: ['b15', 'b7', 'b12', 'b24', 'b12', 'b7', 'b3', 'b16', 'b21', 'b8', 'b17']

Section: PRELIMINARIES
2.1 NOTATION Let X denote a countable universe of items. A dataset W of size n is a collection of subsets {W i } i∈ [n] where W i ⊂ X and |W i | < ∞. We will use N = i |W i | to denote the total number of items across all users in the dataset and M := | i W i | to denote the number of unique items across W . For an element x ∈ i W i , we let N (x) := i 1{x ∈ W i } denote its frequency. For a number r ∈ [M ], we use N (r) to denote the r'th largest frequency after sorting {N (x)} x∈ i Wi in decreasing order. We will use N (0, σ 2 ) to denote a mean-zero Gaussian distribution with standard deviation σ. Finally, we use the notation Õk (•) to suppress poly-logarithmic factors in k and likewise for Ωk and Θk .
this section cite: []

Section: DIFFERENTIAL PRIVACY
We say that a pair of datasets W, W ′ are neighboring if W ′ is the result of adding or removing a single user from W . In this work, we consider randomized algorithms A : (2 X ) ⋆ → 2 X which map a dataset W to a random subset S ⊆ X . We say that A is (ϵ, δ)-differentially private if its distribution over outputs for two neighboring datasets are "close."
Definition 2.1 (Dwork et al. (2006)). A randomized algorithm A is (ϵ, δ)-differentially private if for all pairs of neighboring datasets W, W ′ , and all events Y ⊆ 2 X ,
P S∼A(W ) [S ∈ Y ] ≤ e ϵ P S ′ ∼A(W ′ ) [S ′ ∈ Y ] + δ.
We only consider approximate differential privacy (δ > 0) since we will often require that A(W ) ⊆ i W i , which precludes pure differential privacy (δ = 0).
this section cite: ['b9']

Section: PRIVATE DOMAIN DISCOVERY AND MISSING MASS
In private domain discovery, we are given a dataset W of n users, each of which holds a subset of items W i ⊆ X such that |W i | < ∞ and X is unknown. Given W , our is goal is to extract an informative subset S ⊆ i W i that captures the "domain" of W while preserving differential privacy. In this paper, we often measure quality in terms of the missing mass of S. Definition 2.2. Given dataset W and output set S, the missing mass of S with respect to W is MM(W, S) :=
x∈ i Wi\S N (x) N .
Smaller values of MM(W, S) indicate the that S better captures the high-frequency items in i W i .
A useful perspective to the MM is that it is the ℓ 1 norm of the vector (N (x)/N ) x∈ i Wi\S . This view yields a generalization of the MM objective by taking the p'th norm of the vector of missing frequencies. That is, for p ≥ 0, define
MM p (W, S) := N (x) N x∈ i Wi\S p(1)
where || • || p : R ⋆ → R denotes the ℓ p norm. The usual missing mass objective corresponds to setting p = 1. However, it is also meaningful to set p ̸ = 1. For example, when p = ∞, the objective corresponds to minimizing the maximum missing mass. When p = 0, we recover the cardinality-based objective studied by existing work (see Related Work).
this section cite: []

Section: PRIVATE SET UNION
In general, we consider algorithms that satisfy the following "soundness" property: an item only appears in the output if it also appears in the input dataset. Assumption 1. For every algorithm A : (2 X ) ⋆ → 2 X and dataset W , we require A(W
) ⊆ i W i .
Assumption 1 is standard across works in the unknown domain setting. However, even with this assumption, it is difficult to obtain a meaningful trade-off between privacy and missing mass without assumptions on W .
To see why, fix some n ∈ N, and consider the singleton dataset W where each user has a single, unique item, such that W i = {x i } and x i ̸ = x j for i ̸ = j. Fix j ∈ [n] and consider the neighboring dataset W ′ obtained by removing W j from W. Since we require that A(W ′ ) ⊆ i W ′ i , we have that P [x j ∈ A(W ′ )] = 0. Since A is (ϵ, δ)-differentially private, we know that P [x j ∈ A(W )] ≤ δ. Since j ∈ [n] was picked arbitrarily, we know that this is true for all j ∈ [n] and hence, E S∼A(W ) [MM(A, S)] ≥ 1 -δ. As δ is usually picked to be o 1 n , it is not possible to significantly minimize MM for these datasets.
Fortunately, in practice, these sorts of pathological datasets are rare. Instead, datasets often exhibit what is known as Zipf's or Power law (Zipf, 1949;Gabaix, 1999;Adamic & Huberman, 2002;Piantadosi, 2014). This means that the frequency of items in a dataset exhibit a polynomial decay. Hence, one natural way of measuring the complexity of W is by how "Zipfian" it is.
Definition 3.1 ((C, s)-Zipfian). Let C ≥ 1 and s ≥ 0. A dataset W is (C, s)-Zipfian if N (r) N ≤ C r s
for all r ∈ [M ], where N (r) is the r'th largest frequency and N is the total number of items in W .
In light of the hardness above, we first restrict our attention to datasets that are (C, s)-Zipfian for s > 1. When s ≤ 1, the hard dataset previously outlined becomes a valid Zipfian dataset. We note that this restriction only impacts how we define the utility guarantee of the algorithm, and not its privacy guarantee; differential privacy is still measured with respect to the worst-case pair of neighboring datasets.
As s increases, the empirical mass gets concentrated more and more at the highest frequency item. Accordingly, any upper bound on missing mass should ideally decay as s increases. Another important property of (C, s)-Zipfian datasets W is that they restrict the size of any individual set W i . Lemma 3.1, whose proof is in Appendix C.1, makes this precise. The rest of this section uses these two properties of Zipfian-datasets to obtain high-probability upper bounds on the missing mass. Our main focus will be on a simple mechanism used in practice known as the Weighted Gaussian Mechanism (WGM) (Gopi et al., 2020).
this section cite: ['b27', 'b10', 'b0', 'b20', 'b12']

Section: THE WEIGHTED GAUSSIAN MECHANISM
The WGM is parameterized by a noise-level σ > 0, threshold T ≥ 1, and user contribution bound ∆ 0 ≥ 1. Given a dataset W , the WGM operates in three stages. In the first stage, the WGM constructs a random dataset by subsampling without replacement from each user's itemset to ensure that each user has at most ∆ 0 items. In the second, stage the WGM constructs a weighted histogram over the items in the random dataset. In the third stage, the WGM computes a noisy weighted histogram by adding mean-zero Gaussian noise with standard deviation σ to each weighted count. Finally, the WGM returns those items whose noisy weighted counts are above the threshold T . Pseudocode appears in Algorithm 1.
Algorithm 1 Weighted Gaussian Mechanism Input: Dataset W , noise level σ, threshold T , and user contribution bound ∆ 0 .
1 Construct random dataset W such that for every i ∈ [n], W i ⊆ W i is a random sample (without replacement) of size min{∆ 0 , |W i |} from W i . 2 Compute weighted histogram H : i W i → R such that, for each x ∈ i W i , H(x) = n i=1 1 | W i | 1/2 1{x ∈ W i }.
3 For each x ∈ i W i , sample Z x ∼ N (0, σ 2 ) and compute noisy H ′ (x) := H(x) + Z x .
4 Keep items with large noisy weighted counts S = x ∈ i W i : H ′ (x) ≥ T .
this section cite: []

Section: Output: S
The following theorem from Gopi et al. (2020) verifies the approximate DP guarantee for the WGM. Theorem 3.2 (Theorem 5.1 (Gopi et al., 2020)). For every ∆ 0 ≥ 1, ϵ > 0 and δ ∈ (0, 1), if σ, T > 0 are chosen such that
Φ 1 2σ -ϵσ -e ϵ Φ - 1 2σ -ϵσ ≤ δ 2 and T ≥ max 1≤t≤∆0 1 √ t + σΦ -1 1 - δ 2 1 t
then the WGM run with (σ, T ) and input ∆ 0 is (ϵ, δ)-differentially private.
In Appendix C.2.1, we prove that the smallest choice of σ and T to satisfy the constraints in Theorem 3.2 gives that σ = Θ 1 ϵ log(1/δ) and T = Θ max σ log ∆0 δ , 1 = Θδ,∆0 (max{σ, 1}). This result will be useful for deriving asymptotic utility guarantees involving the WGM.
this section cite: ['b12', 'b12']

Section: UPPER BOUNDS ON MISSING MASS
Our main result in this section is Theorem 3.3, which provides a high-probability upper bound on the missing mass for the WGM in terms of the Zipfian parameters of the input dataset. Theorem 3.3. For every s > 1, C ≥ 1 and (C, s)-Zipfian dataset W , if the WGM is run with noise parameter σ > 0, threshold T ≥ 1, and user contribution bound ∆ 0 ≥ 1, then with probability at least 1 -β over S ∼ WGM(W, ∆ 0 ), we have that
MM(W, S) = Õβ,C,N C 1 s s -1 max i |W i | N √ q ⋆ s-1 s (T + σ) s-1 s .
where
q ⋆ := min{max i |W i |, ∆ 0 }.
Note that in Theorem 3.3 the missing mass decays as the total number of items N grows. Moreover, as C decreases or s increases, the upper bound on missing mass decreases when N is sufficiently large compared to σ and T . This matches our intuition, as decreasing C and increasing s results in datasets that exhibit faster decays in item frequencies so relatively more of the mass is contained in high-mass items.
The proof of Theorem 3.3 relies on three helper lemmas. Lemma C.2 provides an upper bound on the missing mass due to the subsampling stage. Lemma C.3 guarantees that a high-frequency item in the original dataset will remain high-frequency in the subsampled dataset. Finally, Lemma C.4 provides a high-probability upper bound on the frequency of items that are missed by the WGM during the thresholding step. We provide the full proof in Appendix C.2.2.
Theorem 3.3 bounds the overall missing mass of the WGM mechanism. As corollary, note that if ∆ 0 ≥ max i |W i | then the missing mass contributed by the subsampling step vanishes. By Theorem 3.2 and Lemma C.1, for every user contribution bound ∆ 0 ≥ 1, we need to pick σ = Θ 1 ϵ log(1/δ) and T = Θ∆0,δ (max{σ, 1}) to to achieve (ϵ, δ)-differential privacy. Substituting these values into Theorem 3.3 gives the following corollary. Corollary 3.4. In the setting of Theorem 3.3, if we choose the minimum σ and T to ensure (ϵ, δ)-DP, then with probability at least 1 -β, we have that Theorem 3.5, whose proof is in Appendix D.1, shows that the dependence of ϵ and N in our upper bound from Corollary 3.4 can be tight. Theorem 3.5. Let A be any (ϵ, δ)-differentially private algorithm satisfying Assumption 1. For every s > 1, C ≥ 1, there exists a (C, s)-Zipfian dataset W ⋆ such that
MM(W, S) ≤ Õβ,δ,∆0,C,N C 1 s s -1 max i |W i | ϵ N √ q ⋆ s-1 s , where q ⋆ = min{∆ 0 , max i |W i |}.
E S∼A(W ⋆ ) [MM(W ⋆ , S)] = Ω C 1/s s -1 1 ϵN (s-1)/s ln 1 + e ϵ -1 2δ (s-1)/s .
The proof of Theorem 3.5 exploits Assumption 1 by showing that any private algorithm that satisfies Assumption 1 cannot output low-frequency items with high-probability. We end this section by noting that our proof technique in Theorem 3.3 can also give us bounds on the ℓ ∞ missing mass (see Equation 1). Note that unlike Theorem 3.3, Theorem 3.6, whose proof is in Appendix C.2.3, does not require the dataset to be Zipfian. Theorem 3.6. Let W be any dataset. For every ϵ > 0, δ ∈ (0, 1), and user contribution bound
∆ 0 ≥ 1, picking σ = Θ 1 ϵ log(1/δ)
and T = Θ∆0,δ (max{σ, 1}) gives that the WGM is (ϵ, δ)-differentially private and with probability at least 1 -β over S ∼ WGM(W, ∆ 0 ), we have
MM ∞ (W, S) ≤ Õ∆0,δ,β max i |W i | ϵ N √ q ⋆ , where q ⋆ = min{∆ 0 , max i |W i |}.
Upper bounds on the ℓ ∞ norm missing mass will be useful for deriving guarantees for the top-k selection (Section 4.1) and k-hitting set (Section 4.2) problems.
this section cite: []

Section: APPLYING THE WEIGHTED GAUSSIAN MECHANISM
This section applies WGM to construct unknown domain algorithms for top-k and k-hitting set. For both problems, we spend half of the overall privacy budget running WGM to obtain a domain D, and then spend the other half of the privacy budget running a known-domain private algorithm, using domain D, for the problem in question. By basic composition, the overall mechanism satisfies the desired privacy budget. Pseudocode for this approach is given in Algorithm 2.
Algorithm 2 Meta Algorithm Input: Dataset W , noise-level and threshold (σ, T ), output size k, user contribution bound ∆ 0 ≥ 1, known-domain mechanism B 5 Let D ← WGM(W, ∆ 0 ) be the output of WGM with noise-level and threshold (σ, T ) and input ∆ 0 6 Let S ← B(W, D, k) be the output of B on input W and domain D
this section cite: []

Section: Output: S
In the next two subsections, we introduce the top-k selection and k-hitting set problems, summarize existing known-domain algorithms, and provide the specification of all algorithmic parameters. An important difference between the results in this section and that of Section 3, is that by using MM ∞ bounds, we no longer require our dataset to be Zipfian in order to get meaningful guarantees.
this section cite: []

Section: PRIVATE TOP-k SELECTION
In the DP top-k selection problem, we are given some k ∈ N and our goal is to output, in decreasing order, the k largest frequency items in a dataset W . Various loss objective have been considered for this problem, but we focus on missing mass. Definition 4.1. For a dataset W , k ∈ N, q ≤ k and ordered sequence of domain elements S = (x 1 , ..., x q ), we denote the top-k missing mass by
MM k (W, S) = k i=1 N (i) - q i=1 N (x i ) N .
We let the sequence S have length q ≤ k because we will allow our mechanisms to output less than k items, which will be crucial for obtaining differential privacy when the domain
X is unknown. Note that MM(W, S) = MM k (W, S) if one takes k = | i W i |.
As before, our objective is to design an approximate DP mechanism B which outputs a sequence S ⊆ i W i of size at most k that minimizes MM k (W, S) with high probability.
To adapt Algorithm 2 to top-k, we need to specify a known-domain private top-k algorithm. We use the peeling exponential mechanism (see Algorithm 3) for its simplicity, efficiency, and tight privacy composition. Its privacy and utility guarantees appear in Lemmas 4.1 and 4.2 respectively.
Algorithm 3 Peeling Exponential Mechanism Input: Dataset W , domain D, noise-level λ, output size k ≤ |D| 1 Let N (x) = n i=1 1{x ∈ W i } for x ∈ D. 2 Let Ñ (x) = N (x) + Z x for x ∈ D where Z x ∼ Gumbel(λ). Output: Ordered sequence (x 1 , . . . , x k ) such that Ñ (x i ) = Ñ(i) for all i ∈ [k].
Lemma 4.1 (Corollary 4.1 (Durfee & Rogers, 2019)). For every ϵ > 0, δ ∈ (0, 1) and k ≥ 1, if
λ = Õδ √ k ϵ
, then Algorithm 3 is (ϵ, δ)-differentially private.
Lemma 4.2. For every dataset W , domain D, λ ≥ 1 and k ≤ |D|, if Algorithm 3 is run with noise-level λ, then with probability 1 -β over its output S, we have that
1 N   x∈T k (W,D) N (x) - x∈S N (x)   ≤ O kλ N log |D| β ,
where T k (W, D) ⊆ D is the true set of top-k most frequent items in D.
We provide the exact λ to achieve (ϵ, δ)-differential privacy for Lemma 4.1 in Lemma B.1. The proof of Lemma 4.2 is standard, relies on Gumbel concentration inequalities, and appears in Appendix C.3.Similar upper bounds for other performance metrics have also been derived (Bafna & Ullman, 2017). With Lemmas 4.2 and 4.1 in hand, using the same choice of (σ, T ) as in Theorem 3.2 for WGM yields our main result. The proof of Theorem 4.3 can be found in Appendix C.3. Theorem 4.3. Fix ϵ > 0, δ ∈ (0, 1), and user contribution bound ∆ 0 ≥ 1. For every dataset W and k ≥ 1, if one picks σ = Θ 1 ϵ log(1/δ) , T = Θ∆0,δ/2 (max{σ, 1}) from Theorem 3.2, and
λ = Θδ/2 √ k ϵ
from Lemma 4.1, then Algorithm 2, run with Algorithm 3, is (ϵ, δ)-differentially private and with probability 1 -β, its output S satisfies
MM k (W, S) ≤ Õβ,δ,∆0 k N max i |W i | ϵ √ q ⋆ + √ k log(M ) ϵ ,
where
q ⋆ := min{∆ 0 , max i |W i |}.
We end this section by proving that a linear dependence on k ϵ on the top-k missing mass is unavoidable for algorithms satisfying Assumption 1 when ϵ ≤ 1. Corollary 4.4. Let ϵ ≤ 1 and δ ∈ (0, 1). Let A be any (ϵ, δ)-differentially private algorithm satisfying Assumption 1. Then, for every k ≥ 1, there exists a dataset W such that
E S∼A(W,k) MM k (W, S) ≥ Ωδ k ϵN .
The proof of Corollary 4.4 is in Appendix D.2 and is largely a consequence of Lemma D.1, which was used to prove the lower bound for set union (Theorem 3.5).
this section cite: ['b8', 'b2']

Section: PRIVATE k-HITTING SET
In the k-hitting set problem, our goal is to output a set S of items of size at most k which intersects as many user subsets as possible, which is useful for data summarization and feature selection (Mitrovic et al., 2017). More precisely, our objective is to design an approximate DP mechanism which maximizes the number of hits Hits(W, S) := n i=1 1{S ∩ W i ̸ = ∅}. Since this problem is also NP-hard without privacy concerns (Karp, 1972), we will measure performance relative to the optimal solution, i.e., show that with high probability, our algorithm output S satisfies
Hits(W, S) ≥ γ • Opt(W, k) -err(ϵ, δ, k)
where Opt(W, k) := arg max S⊆X ,|S|≤k Hits(W, S) is the optimal value, err(ϵ, δ, k) is an additive error term that depends on problem specific parameters, and γ ∈ (0, 1) is the approximation factor. Like our algorithm for top-k selection, our mechanism for the k-hitting problem will follow the general structure of Algorithm 2. We will take the known-domain algorithm B to be the privatized version of the greedy algorithm for submodular maximization, as in Algorithm 1 from Mitrovic et al. (2017). This mechanism repeatedly runs the exponential mechanism (equivalently the Gumbel mechanism) to pick an item that hits a large number of users. After each iteration, we remove all users who contain the item output in the previous round and continue until we either have output k items, run out of items, or run out of users, and return the overall set of items. We call this algorithm the User Peeling Mechanism and its pseudo-code is given in Algorithm 4 in Appendix C.4.
By combining this with the same WGM choice of (σ, T ) as in Theorem 3.2 for the first step of Algorithm 2, we get the main result of this section. Theorem 4.5. Fix ϵ > 0 and δ ∈ (0, 1). For every dataset W , k ≥ 1, and user contribution bound ∆ 0 , if one picks σ = Θ 1 ϵ log(1/δ) , T = Θ∆0,δ/2 (max{σ, 1}) from Theorem 3.2, and
λ = Θδ/2 1 ϵ √
k from Lemma 4.1, then Algorithm 2, run with Algorithm 4, is (ϵ, δ)-differentially private and with probability 1 -β, its output S satisfies
Hits(W, S) ≥ 1 - 1 e Opt(W, k) -Õβ,δ,∆0 k • max i |W i | ϵ √ q ⋆ + k 3/2 ϵ log (M k) ,
where
q ⋆ := min{∆ 0 , max i |W i |} and M = | i W i |.
Theorem 4.5, proved in Appendix C.4, gives that if k is not very large (i.e., ln(M k) ln(M ) ≤ max i |W i |), then with high probability, the additive sub-optimality gap is on the order of Õ∆0,δ,β,k
k 3/2 • max i |W i | • log(M ) ϵ √ q ⋆ .
When |X | ≫ M , this provides an improvement over Theorem 1 in Mitrovic et al. (2017) whose guarantee is in terms log(|X |) and not log(M ).
As in the lower bound proof for top-k selection, we again rely on the work behind Theorem 3.5 to show that one must lose k ϵ from the optimal value by restricting the algorithm A to output a subset of i W i .
Corollary 4.6. Let ϵ ≤ 1, δ ∈ (0, 1) and A be any (ϵ, δ)-differentially private algorithm satisfying Assumption 1. Then, for every k ≥ 1, there exists a dataset W such that
E S∼A(W,k) [Hits(W, S)] ≥ Opt(W, k) -Ωδ k ϵ .
this section cite: ['b17', 'b14', 'b17', 'b17']

Section: EXPERIMENTS
We empirically evaluate our methods on six real-life datasets spanning diverse settings. Informally, Reddit (Gopi et al., 2020), Amazon Games (Ni et al., 2019), and Movie Reviews (Harper & Konstan, 2015) are "large", while Steam Games (Steam, 2025), Amazon Magazine (Ni et al., 2019), and Amazon Pantry (Ni et al., 2019) are "small" (see Appendix E for details). All experiments use a total privacy budget of (1, 10 -5 )-DP; additional experiments using (0.1, 10 -5 )-DP appear in Appendix F, but are not significantly qualitatively different. Dataset processing and experiment code can be found in the Supplement.
this section cite: ['b12', 'b18', 'b13', 'b18', 'b18']

Section: SET UNION
Datasets. We evaluate the WGM and baselines on all six datasets, relegating experiments on the small datasets to the Appendix F.1.1 for space.
Baselines. The baselines are the Policy Gaussian mechanism from Gopi et al. (2020) and the Policy Greedy mechanism from Carvalho et al. (2022), as these have obtained the strongest (though least scalable) performance in past work. As suggested in those papers, we set the policy hyperparameter α = 3 throughout.
Results. Figure 1 plots the average MM across 5 trials, for all three mechanisms as a function of ℓ 0 bound ∆ 0 ∈ {1, 50, 100, 150, 200, 300}. Across datasets, we find that the WGM obtains MM within 5% of that of the policy mechanisms, in spite of their significantly more intensive computation. This contrasts with previous empirical results for cardinality, where sequential methods often output ≈ 2X more items (see, e.g., Table 2 in Swanberg et al. (2023)). Plots for the small datasets (Appendix F.1.1) show a similar trend.
this section cite: ['b12', 'b24']

Section: TOP-k
Datasets. All methods achieve near 0 top-k missing mass across all values of number of selected items k ∈ {5, 10, 20, 50, 100, 200} on the three large datasets, as most mass is concentrated in a small number of heavy items. We therefore focus on the three small datasets.
Baselines. We compare our WGM-then-top-k mechanism to the limited-domain top-k mechanism from Durfee & Rogers (2019). Unlike our algorithm, the limited-domain mechanism has a hyperparameter k. As such, for each k ∈ {5, 10, 20, 50, 100, 200}, we take as baselines the limited-domain algorithm with k ∈ {k, 5k, 10k, ∞}. When k < ∞, we set ∆ 0 = ∞ for the limited-domain algorithm. Otherwise, when k = ∞, we set ∆ 0 = 100 for the limited-domain algorithm, as recommended in Section 3 of Durfee & Rogers (2019).
Results. Figure 2 compares our method against the limited-domain method across different choices for k. Note that each line for the limited-domain method uses a different k. We find that across all datasets, our method consistently obtains smaller top-k MM than all limited-domain baselines, and its advantage grows with k. Plots in Appendix F.2 demonstrate similar trends for a more stringent ℓ 1 loss.
this section cite: ['b8', 'b8']

Section: k-HITTING SET
Datasets. We use the same datasets as in the top-k experiments, for the same reason: a small number of items covers nearly all users in the large datasets.
Baselines. To the best of our knowledge, there are no existing private algorithm for the k-hitting set problem for unknown domains. Hence, we consider the following baselines: the non-private greedy algorithm and the private non-domain algorithm from Mitrovic et al. (2017) after taking i W i to be a public known-domain. Note that the latter baseline is not a valid private algorithm in the unknown domain setting since, in reality, i W i is private.
Results. Figure 3 plots the average number of users hit, along with its standard error across 5 trials, as a function of k ∈ {5, 10, 20, 50, 100, 200}, fixing ∆ 0 = 100. We find that our method performs comparably with both baseline methods, neither of which is fully private. In particular, for the Steam Games and Amazon Magazine datasets, our method outperforms the known-domain private greedy algorithm that assumes public knowledge of i W i . This is because our method's application of WGM for domain discovery produces a domain that is smaller than i W i while still containing high-quality items. This makes an easier problem for the peeling mechanism in the second step.
this section cite: ['b17']

Section: FUTURE DIRECTIONS
We conclude with some possible future research directions. First, our upper and lower bounds for top-k and k-hitting set do not match, so closing these gaps is a natural problem. Second, all of our methods enforce ℓ 0 bounds by uniform subsampling without replacement from each user's item set. Recent work by Chen et al. (2025) employs more involved and data-dependent subsampling strategies to obtain higher cardinality answers. Extending similar techniques to missing mass may be useful. Proof. Consider a single X i , and recall that the CDF of a Gumbel distribution with parameter λ is F (x) = exp(-exp(-x/λ)). Then
P [|X i | > T ] = P [X i > T ] + P [X i < -T ] = 1 -exp(-exp(-T /λ)) + exp(-exp(T /λ)) ≤ exp(-T /λ) + exp(-exp(T /λ))
where the inequality uses 1 -e -x ≤ x. Substituting in T = λ log(2n/δ) yields
P [|X i | > t] ≤ δ 2n + exp(-2n/δ) ≤ δ n
since for n ≥ 1 and δ ∈ (0, 1), 2n δ ≥ ln 2n δ . Union bounding over the n samples completes the result. ■
this section cite: []

Section: B PRIVACY ANALYSIS OF PEELING EXPONENTIAL MECHANISM
Lemma B.1 (Lemma 4.2 in Gillenwater et al. ( 2022)). For every ϵ > 0, δ ∈ (0, 1) and k ≥ 1, if λ = 1 ϵ0 , where
ϵ 0 := max ϵ k , 8 log 1 δ + 8ϵ k - 8 log 1 δ k ,
then Algorithm 3 is (ϵ, δ)-differentially private. Proof. Starting with σ, it suffices to find the smallest σ such that
Φ 1 2σ -ϵσ ≤ δ 2 .
By monotonicity of Φ -1 (•), we have that
Φ 1 2σ -ϵσ ≤ δ 2 ⇐⇒ 1 2σ -ϵσ ≤ Φ -1 δ 2 .
Hence, it suffices to find the smallest σ that satisfies
2ϵσ 2 + 2Φ -1 δ 2 σ -1 ≥ 0.
Using the quadratic formula we can deduce that we need to take
σ ≥ -Φ -1 δ 2 ϵ = Φ -1 1 -δ 2 ϵ = Ω   log 1 δ ϵ   ,
where the last inequality follows from the fact that Φ -1 (p) ≤ 2 log 1 1-p for p > 1 2 . Now for T , we have
1 + σΦ -1 1 - δ 2 1 ∆ 0 ≥ max 1≤t≤∆0 1 √ t + σΦ -1 1 - δ 2 1 t .
Hence, it suffices to upper bound
1 + σΦ -1 1 - δ 2 1 ∆ 0 .
By Bernoulli's inequality and monotonicity of Φ -1 (•), we have that
Φ -1 1 - δ 2 1 ∆ 0 ≤ Φ -1 1 - δ 2∆ 0 .
Since δ ≤ 1 and ∆ 0 ≥ 1, we have that
Φ -1 1 - δ 2∆ 0 ≤ 2 log 2∆ 0 δ .
Hence, it suffices to take
T = 1 + σ 2 log 2∆ 0 δ = Θ max σ log ∆ 0 δ , 1 ,
This completes the proof. ■
this section cite: []

Section: C.2.2 PROOF OF THEOREM 3.3
Before we prove Theorem 3.3, we present three helper lemmas, Lemma C.2, C.3, and C.4, which correspond to three different "good" events. Lemma C.2 provides an upper bound on the missing mass due to the subsampling stage. Lemma C.3 guarantees that a high-frequency item in the original dataset will remain high-frequency in the subsampled dataset. Lemma C.4 provides a high-probability upper bound on the frequency of items that are missed by the WGM during the thresholding step. The proof of Theorem 3.3 will then follow by combining Lemmas C.2, C.3, and C.4. Lemma C.2. Let W be a (C, s)-Zipfian dataset for C ≥ 1 and s > 1. Fix a user contribution bound ∆ 0 ≥ 1. Let W be the random dataset such that W i ⊆ W i is a random sample without replacement of size min{∆ 0 , W i }. Then, for every β ∈ (0, 1), with probability 1 -β, we have that
MM(W, ∪ i W i ) ≤ C 1/s s -1 1 p ⋆ N log (CN ) 1/s β s-1 s
where p ⋆ := min 1, ∆0 maxi |Wi| .
Proof. Let p i := min 1, ∆0 |Wi| and p ⋆ := min i p i . Fix an item x ∈ i W i . Then
P x / ∈ i W i = i:x∈Wi (1 -p i ) ≤ exp - i:x∈Wi p i ≤ e -p ⋆ N (x) ,
where N (x) = i 1{x ∈ W i }. Fix β ∈ (0, 1) and consider the threshold
Q := 1 p ⋆ log (CN ) 1/s β .
Published as a conference paper at ICLR 2026 Note that Q ⋆ ≥ 1 by definition of p ⋆ . Since W is (C, s)-Zipfian, for any r ∈ [M ] with N (r) > Q, it must be the case that r ≤ CN Q 1/s by r s ≤ CN N (r) . Hence, there are at most CN Q 1/s "heavy" items whose frequencies are above Q. By the union bound, we get that
P ∃x ∈ i W i \ W i and N (x) > Q ≤ CN Q 1/s e -p ⋆ Q ≤ β
so with probability at least 1 -β, we have that N (x) > Q =⇒ x ∈ i W i for all x ∈ i W i .
Under this event, we have that
MM(W, ∪ i W i ) ≤ 1 N x:N (x)≤Q N (x) ≤ 1 N r≥r0 N (r)
where r 0 = max CN Q 1/s , 1 . Using the fact that N (r) ≤ CN r -s and s > 1 (by assumption), we get that
1 N r≥r0 N (r) ≤ C r≥r0 r -s ≤ C ∞ r0-1 x -s dx = C s -1 (r 0 -1) 1-s ≤ C s -1 r 1-s 0 . Since r 0 ≥ CN Q 1/s , we get MM(W, ∪ i W i ) ≤ C 1/s s-1 Q N s-1 s . Using Q = 1 p ⋆ log (CN ) 1/s β
completes the proof. ■ Lemma C.3. In the same setting as Lemma C.2, for every β ∈ (0, 1), we have that with probability 1 -β, for every
x ∈ i W i , N (x) ≥ τ 2 =⇒ N (x) ≥ 1 2 p ⋆ N (x),
where p ⋆ := min 1, ∆0 maxi |Wi| , τ 2 := 8 p ⋆ log (CN ) 1/s β , and N (x) = i 1{x ∈ W i }.
Proof. Fix some x ∈ i W i such that N (x) ≥ τ 2 . Then, N (x) is the sum of independent Bernoulli random variables with success probability at least p ⋆ . Thus, we have that E N (x) ≥ p ⋆ N (x) and multiplicative Chernoff's inequality gives
P N (x) ≤ 1 2 p ⋆ N (x) ≤ exp - 1 8 p ⋆ N (x) ≤ exp - 1 8 p ⋆ τ 2 .
Now, since W is (C, s)-Zipfian, we have that N (x) ≤ CN r s for all x ∈ i W i , so there can be at most
CN τ2 1/s elements x ∈ i W i with N (x) ≥ τ 2 . A union bound yields P ∃x ∈ i W i : N (x) ≥ τ 2 , N (x) < 1 2 p ⋆ N (x) ≤ β (CN ) 1/s • CN τ 2 1/s ≤ β,
which completes the proof.
this section cite: []

Section: ■
Lemma C.4. For every dataset W , if the WGM is run with noise parameter σ > 0, threshold T ≥ 1, and user contribution bound ∆ 0 ≥ 1, then for every β ∈ (0, 1), with probability at least 1-β over S ∼ WGM(W, ∆ 0 ), we have that H(x) ≤ T 0 , ∀x ∈ M , where T 0 := T +σ 2 log 2N β and M := i W i \ S.
Proof. By Line 4 in Algorithm 1, we have that for all x ∈ M , its noisy weighted count is H ′ (x) < T . Hence, by standard Gaussian concentration bounds (see Appendix A), with probability at least 1 -β over just the sampling of Gaussian noise in Line 3, we have H(x) ≤ T 0 for all x ∈ M . ■
We are now ready to prove Theorem 3.3.
Proof. (of Theorem 3.3) Let W be the random dataset obtained by sampling a set W i of elements of size min{∆ 0 , |W i |} without replacement from each W i , and let S be the overall output of the WGM.
Let N (x) = n i=1 1{x ∈ W i } be the frequency of item x in the subsampled dataset and note that N (x) ≤ √ q ⋆ H(x) for all x ∈ i W i , where H is the weighted histogram of item frequencies from W . Let M = i W i \ S be the random variable denoting the set of items in i W i but not in the algorithm's output S. Finally, define τ 1 := 2 √ q ⋆ T0 p ⋆
and τ 2 := 8 p ⋆ log 3(CN ) 1/s β , where
T 0 = T + σ 2 log 2N β .
Let E 1 , E 2 , and E 3 be the events of Lemma C.2, C.3, and C.4 respectively, setting the failure probability for each event to be β 3 . Then, by the union bound, E 1 ∩ E 2 ∩ E 3 occurs with probability 1 -β. It suffices to show that E 1 ∩ E 2 ∩ E 3 implies the stated upper bound on MM(W, S). We can decompose MM(W, S) into two parts, mass missed by subsampling and mass missed by noisy thresholding:
MM(W, S) = MM(W, ∪ i W i ) + 1 N x∈ M N (x).(2)
Under E 1 , we have that
MM(W, ∪ i W i ) ≤ C 1/s s -1 1 p ⋆ N log 3(CN ) 1/s β s-1 s ,
hence for the remainder of the proof, we will focus on bounding 1 N x∈ M N (x). First, we claim that under E 2 and E 3 , we have that N (x) ≤ max{τ 1 , τ 2 } =: τ for all x ∈ M . This is because, by event E 3 , we have that for every x ∈ M , N (x) ≤ √ q ⋆ T 0 . Thus, if there exists an x ∈ M such that N (x) ≥ τ 2 , then by event E 2 , it must be the case that
p ⋆ 2 N (x) ≤ N (x) ≤ √ q ⋆ T 0 , which implies that N (x) ≤ τ 1 . Now, define r 0 := max CN τ 1/s , 1 . If r ≥ r 0 , then CN r s ≤ τ.
Since N (x) ≤ τ for every x ∈ M , every such item has rank greater than r 0 . Hence,
1 N x∈ M N (x) ≤ 1 N r≥r0 N (r) ≤ r≥r0 C r s ≤ C ∞ r0-1 t -s dt ≤ Cr 1-s 0 s -1 .
Substituting in the definition of r 0 and continuing yields
1 N x∈ M N (x) ≤ C 1/s s -1 τ N s-1 s ≤ C 1/s s -1   max 2 √ q ⋆ T0 p ⋆ , 8 p ⋆ log 3(CN ) 1/s β N   s-1 s .
Now, we are ready to complete the proof. Using the decomposition of MM(W, S) in Equation 2along with
E 1 ∩ E 2 ∩ E 3 implies that MM(W, S) ≤ C 1/s s -1 1 p ⋆ N log 3(CN ) 1/s β s-1 s + C 1/s s -1   max 2 √ q ⋆ T 0 , 8 log 3(CN ) 1/s β p ⋆ N   s-1 s ≤ C 1/s s -1 1 p ⋆ N s-1 s 9 max √ q ⋆ T 0 , log 3(CN ) 1/s β s-1 s = C 1/s s -1 9 p ⋆ N s-1 s max √ q ⋆ T + σ 2 log 6N β , log 3(CN ) 1/s β s-1 s .
The proof is complete after noting that
√ q ⋆ p ⋆ = maxi |Wi| √ q ⋆ . ■
this section cite: []

Section: C.2.3 PROOF OF THEOREM 3.6
As in the proof of Theorem 3.3, we start with the following lemma which bounds the maximum missing mass due to the subsampling step.
Lemma C.5. Let W be any dataset. Fix a user contribution bound ∆ 0 ≥ 1. Let W be the random dataset such that W i ⊆ W i is a random sample without replacement of size min{∆ 0 , W i }. Then, for every β ∈ (0, 1), with probability 1 -β, we have that
MM ∞ (W, ∪ i W i ) ≤ log N β p ⋆ N .
where p ⋆ := min 1, ∆0 maxi |Wi| .
Proof. Let W be any dataset, ∆ 0 ≥ 1 and β ∈ (0, 1). We will follow the same proof strategy as in the proof of Lemma C.2. Let p i := min 1, ∆0 |Wi| and p ⋆ := min i p i . Fix an item x ∈ i W i . Then,
P x / ∈ i W i = i:x∈Wi (1 -p i ) ≤ exp - i:x∈Wi p i ≤ e -p ⋆ N (x) ,
where N (x) = i 1{x ∈ W i }. Fix β ∈ (0, 1) and consider the threshold
Q := 1 p ⋆ log N β ≥ 1.
Since x∈ i Wi N (x) = N , we have that that there are at most N Q items such that N (x) > Q. Hence, by the union bound, we get that
P ∃x ∈ i W i \ W i and N (x) > Q ≤ N Q e -p ⋆ Q ≤ β.
Hence, with probability at least 1 -β, we have that if x / ∈ i W i , then N (x) ≤ Q, giving that
MM ∞ (W, ∪ i W i ) ≤ log N β p ⋆ N ,
which completes the proof. ■ Now, we use Lemma C.5 to complete the proof of Theorem 3.6. Since the proof follows almost identically, we only provide a sketch here.
Proof. (sketch of Theorem 3.6) As in the proof of Theorem 3.3, define
q ⋆ = min{max i |W i |, ∆ 0 } and T 0 = T + σ 2 log 6N β . Keep τ 1 := 2 √ q ⋆ T0 p ⋆
but take
τ 2 := 8 p ⋆ log 3N β .
Let E 2 be defined identically in terms of T 0 . That is, E 2 is the event that H(x) ≤ T 0 for all x ∈ M , where
T 0 = T + σ 2 log 6N β and M = i W i \ S. Likewise, define E 3 in terms of τ 2 analogous to that in the proof of Theorem 3.3. That is, E 3 is the event that for all x ∈ i W i , either N (x) < τ 2 or N (x) ≥ p ⋆ 2 • N (x).
The fact that E 2 occurs with probability at least 1 -β 3 follows identitically from the proof of Theorem 3.3.. As for event E 3 , note that we have
P ∃x ∈ i W i : N (x) ≥ τ 2 , N (x) < 1 2 p ⋆ N (x) ≤ N τ 2 • e -p ⋆ τ 2 8 ≤ β 3 ,
which follows similarly by using multiplicative Chernoff's, the union bound, and the fact that there can be at most N τ2 items with frequency at least τ 2 . Hence, E 3 occurs with probability at least 1 -β 3 .
Then, by the union bound we have that with probability 1 -2β 3 , both E 2 and E 3 occur. When this happens, we have that N (x) ≤ max{τ 1 , τ 2 } for all x ∈ M because either N (x) ≤ τ 2 , or otherwise
1 2 p ⋆ N (x) ≤ N (x) ≤ √ q ⋆ T 0 , implying that N (x) ≤ τ 1 . Consequently, under E 2 and E 3 , we have that max x∈∪i Wi\S N (x) N ≤ max{τ 1 , τ 2 } N By Lemma C.5, the event MM ∞ (W, ∪ i W i ) ≤ log 3N β p ⋆ N .
occurs with probability 1
-β 3 . Hence, under E 1 ∩ E 2 ∩ E 3 , we have that MM ∞ (W, S) ≤ max MM ∞ (W, ∪ i W i ), max x∈ W \S N (x) N ≤ 8 p ⋆ N max log 3N β , √ q ⋆ T + σ 2 log 6N β , which occurs with probability 1 -β. Finally, plugging in σ = Θ √ ln(1/δ) ϵ
and T = Θ∆0,δ (σ) completes the claim. ■ C.3 PROOF OF THEOREM 4.3
Since the privacy guarantee follows by basic composition, we only focus on proving the utility guarantee in Theorem 4.3. First, we provide the proof of Lemma 4.2.
Proof. (of Lemma 4.2) Let I = T k (W, D) \ S and O = S \ T k (W, D). Then, |I| = |O| ≤ k and
x∈T k (W,D) N (x) - x∈S N (x) = x∈I N (x) - x∈O N (x).
Since |I| = |O|, there exists a one-to-one mapping π : I → O that pairs each item in I with an item in O. Thus, we can write
x∈I N (x) - x∈O N (x) = x∈I (N (x) -N (π(x))).
By definition of I and O, we have that for every x ∈ I and y ∈ O, Ñ (y) ≥ Ñ (x). Hence, we have that
N (x) -N (y) ≤ Z y -Z x and x∈I (N (x) -N (π(x))) ≤ x∈I (Z π(x) -Z x ). Define R := x∈I (Z π(x) -Z x ).
Our goal is to get a high-probability upper bound on R via concentration. By Gumbel concentration (Lemma A.2), with probability 1 -β, we have that
max x∈D |Z x | ≤ λ • log(2|D|/β). Hence, under this event, we get that R ≤ 2kλ • log(2|D|/β).
Altogether, with probability 1 -β, we have
x∈T k (W,D) N (x) - x∈S N (x) ≤ R ≤ 2kλ • log(2|D|/β).
this section cite: []

Section: Dividing by N completes the proof.
■ Published as a conference paper at ICLR 2026 Combining Lemmas 4.2 and 4.1 then gives the following corollary. Corollary C.6. For every dataset W , domain D, k ≤ |D|, ϵ > 0, and δ ∈ (0, 1), if Algorithm 3 is run with λ = Θδ √ k ϵ from Lemma 4.1, then Algorithm 3 is (ϵ, δ)-differentially private and with probability at least 1 -β over its output S, we have that
1 N   x∈T k (W,D) N (x) - x∈S N (x)   ≤ Õδ,β k 3/2 log |D| ϵN .
With Corollary C.6 in hand, we are now ready to prove Theorem 4.3 after picking the same choice of (σ, T ) as in Theorem 3.2.
Proof. (of Theorem 4.3) Recall that by Theorem 3.6, if we set σ = Θ √ ln(1/δ) ϵ and T = Θ∆0,δ/2 (max{σ, 1}), then the WGM is (ϵ/2, δ/2)-differentially private and with probability at least
1 -β/2 over D ∼ WGM(W, ∆ 0 ), we have that MM ∞ (W, D) ≤ Õ∆0,δ/2,β/2 max i |W i | ϵN √ q ⋆ .(3)
Let T k (W ) be the true set of top-k elements and T k (W, D) be the set of top-k elements within the (random) domain D. Then, under this event, Equation 3gives that
1 N   x∈T k (W ) N (x) - x∈T k (W,D) N (x)   ≤ k • MM ∞ (W, D) ≤ Õ∆0,δ/2,β/2 k • max i |W i | ϵN √ q ⋆ .
(4) By Corollary C.6, we know that running Algorithm 3 on input W , domain D and λ = Õδ/2
√ k ϵ
gives (ϵ/2, δ/2)-differentially privacy and that with probability at least 1-β/2, its output S satisfies
1 N   x∈T k (W,D) N (x) - x∈S N (x)   ≤ Õδ/2,β/2 k 3/2 log |D| ϵN .(5)
Adding Inequalities 4 and 5 together and taking |D| ≤ | i W i | =: M gives that with probability 1 -β, the output S of Algorithm 2 satisfies
MM k (W, S) ≤ Õβ,δ,∆0 k N max i |W i | ϵ √ q ⋆ + √ k log(M ) ϵ ,
which completes the proof. ■ C.4 PROOF OF THEOREM 4.5
Before we prove Theorem 4.5, we first present the pseudo-code (Algorithm 4) for the user peeling mechanism described in Section 4.2 along with its privacy and utility guarantees.
The following lemma gives the utility and privacy guarantee of Algorithm 4. Lemma C.7. For every dataset W , domain D, and k ≤ |D|, if Algorithm 4 is run with noise parameter λ > 0, then with probability 1 -β over its output S, we have that
Hits(W, S) ≥ 1 - 1 e Opt(W, D, k) -2kλ log 2|D|k β .
where Opt(W, D, k) := arg max S⊆D,|S|≤k Hits(W, S). If one picks λ = Θδ √ k ϵ from Lemma 4.1, then Algorithm 4 is (ϵ, δ) differentially private and with probability 1 -β over its output S, we have
Hits(W, S) ≥ 1 - 1 e Opt(W, D, k) -Õδ,β k 3/2 log (|D|k) ϵ .
Algorithm 4 User Peeling Mechanism Input: Dataset W , domain D, number of elements k, noise-level λ 1 Initialize W 1 ← W , D 1 ← D, and output set S 0 ← ∅ 2 for j = 1, . . . , k do 3 Compute histogram H j (x) = i 1{x ∈ W j i } for all x ∈ D j .
this section cite: []

Section: 4
Compute noisy histogram Hj (x) = H j (x) + Z j x for all x ∈ D j where Z j x ∼ Gumbel(λ).
5 Let x j ∈ arg max x∈Dj Hj (x) 6 Update S j ← S j-1 ∪ {x j }, D j+1 ← D j \ {x j }, and W j+1 ← {W i ∈ W j : x j / ∈ W i }. 7 end
Output: S k Since Algorithm 4 also uses the peeling exponential mechanism, the privacy guarantee in Lemma C.7 follows exactly from Lemma 4.1, and so we omit the proof here. As for the utility guarantee, the proof is similar to the proof of Theorem 7 in Mitrovic et al. (2017). For the sake of completeness, we provide a self-contained analysis below.
Proof. Note that there exists at most |D|k random variables Z j x . Let E be the event that |Z j x | ≤ α for all x ∈ D and j ∈ [k], where α = λ ln 2|D|k β . Then, by Gumbel concentration (Lemma A.2), we have that P(E) ≥ 1 -β. For the rest of this proof, we will operate under the assumption that event E happens.
Define the function f : 2 D → N as f (S) := n i=1 1{W i ∩ S ̸ = ∅} = Hits(W, S). Then, f is a monotonic, non-negative submodular function. For x ∈ D and S ⊆ D let ∆(x, S) := f (S ∪ {x}) -f (S) ≥ 0. Let S ⋆ ∈ arg max S⊆D,|S|≤k f (S) be the optimal subset of D and denote Opt := f (S ⋆ ). First, we claim that for every S ⊆ D,
max x∈S ⋆ \S ∆(x, S) ≥ Opt -f (S) k . (6) This is because Opt = f (S ⋆ ) ≤ f (S ⋆ ∪ S) ≤ f (S) + x∈S ⋆ \S ∆(x, S) ≤ f (S) + k max x∈S ⋆ \S ∆(x, S),
where the first two inequalities follows from monotonicity and submodularity respectively. Now, let x 1 , . . . , x k be the (random) items selected by the algorithm, and S j = {x 1 , . . . , x j } with S 0 = ∅. At round j ∈ [k], define the current domain D j := D \ S j-1 . Let x ⋆ j ∈ arg max
x∈S ⋆ \Sj-1 ∆(x, S j-1 ).
Then, by Equation 6, we have ∆(x ⋆ j , S j-1 ) ≥ Opt -f (Sj-1) k
. This implies that
∆(x j , S j-1 ) ≥ Opt -f (S j-1 ) k -(∆(x ⋆ j , S j-1 ) -∆(x j , S j-1 )) = Opt -f (S j-1 ) k -(H j (x ⋆ j ) -H j (x j ))
where the last equality stems from the fact that for every x ∈ D j , ∆(x, S j-1 ) = H j (x).
Recall that Hj (x) := H j (x) + Z j x for all x ∈ D. We can upper bound
H j (x ⋆ j ) -H j (x j ) as H j (x ⋆ j ) -H j (x j ) = ( Hj (x ⋆ j ) -Hj (x j )) + (Z j xj -Z j x ⋆ j ) ≤ Z j xj -Z j x ⋆ j ,
where last inequality is because Hj (x ⋆ j ) -Hj (x j ) ≤ 0 by the choice of x j ∈ arg max x∈Dj H j (x)
and the fact that x ⋆ j ∈ S ⋆ \ S j-1 ⊆ D j . Therefore,
∆(x j , S j-1 ) ≥ Opt -f (S j-1 ) k -Z j xj -Z j x ⋆ j .(7)
Let G j := Opt -f (S j ). Using f (S j ) = f (S j-1 ) + ∆(x j , S j-1 ), we rearrange Equation 7 to get
G j ≤ 1 -1 k G j-1 + (Z j xj -Z j x ⋆ j
).
On the event E, we have that
Z j xj -Z j x ⋆ j ≤ |Z j xj | + |Z j x ⋆ j | ≤ 2α, hence G j ≤ 1 -1 k G j-1 + 2α.
Recursing for j = 1, . . . , k and using the fact that G 0 = Opt and (1
-1 k ) k ≤ e -1 , gives G k ≤ 1 e Opt +2αk.
Substituting in the definition of G k and α = λ ln 2|D|k β gives
f (S k ) ≥ 1 -1 e Opt -2kλ log 2|D|k β .
this section cite: ['b17']

Section: ■
The guarantee in Lemma C.7 is with respect to the optimal set of k elements within the domain D. Since D ⊆ i W i , we have that Opt(W, D, k) ≤ Opt(W, k).
The following simple lemma shows that when the domain D contains high-frequency items from W , Opt(W, D, k) is not too far away from Opt(W, k).
Lemma C.8. Fix a dataset W , τ ≥ 0, and let D = {x ∈ i W i : N (x) ≥ τ }. Then,
Opt(W, D, k) ≥ Opt(W, k) -kτ.
Proof. Recall that f (S) := Hits(W, S) is a monotone, non-negative submodular function. Let S 1 ⊆ X be the subset of X that achieves f (S 1 ) = Opt(W, k) and S 2 be the subset of D that achieves f (S 2 ) = Opt(W, D, k). By submodularity and the definition of X \ D we have that
f (S 1 ) ≤ f (S 1 ∩ D) + f (S 1 ∩ X \ D) ≤ f (S 2 ) + kτ,
which completes the proof. ■ Lemma C.8 allows us to use the MM ∞ upper bound obtained by the WGM in Theorem 3.6 to upgrade the guarantee provided by Lemma C.7 to be in terms of Opt(W, k) instead of Opt(W, D, k).
By using the same choice of (σ, T ) as in Theorem 3.2, we get the main result of this section. As before, since the privacy guarantee follows by basic composition, we only focus on proving the utility guarantee.
Proof. (of Theorem 4.5) Recall that by Theorem 3.6 that if we set σ = Θ √ ln(1/δ) ϵ and T = Θ∆0,δ/2 (max{σ, 1}), then the WGM is (ϵ/2, δ/2)-differentially private and with probability at least 1 -β/2 over D ∼ WGM(W, ∆ 0 ), we have that
MM ∞ (W, D) ≤ Õ∆0,δ/2,β/2 max i |W i | ϵN √ q ⋆ .
By Lemma C.8, under this event we have that
Opt(W, D, k) ≥ Opt(W, k) -Õ∆0,δ/2,β/2 k • max i |W i | ϵ √ q ⋆ .
Now, by Lemma C.7, we know that running Algorithm 4 on input W , domain D and λ = Õδ/2 √ k ϵ
gives (ϵ/2, δ/2)-differentially privacy and that with probability at least 1 -β/2, its output S satisfies
Hits(W, S) ≥ 1 - 1 e Opt(W, D, k) -Õδ/2,β/2 k 3/2 ϵ log (|D|k)
Hence, by the union bound, with probability at least 1 -β, both events occur and we have that
Hits(W, S) ≥ 1 - 1 e Opt(W, D) -Õβ,δ,∆0 k • max i |W i | ϵ √ q ⋆ + k 3/2 ϵ log (M k) ,
where we use the fact that |D| ≤ M . ■
this section cite: []

Section: D LOWER BOUNDS

this section cite: []

Section: D.1 LOWER BOUNDS FOR MISSING MASS
In this section, we prove Theorem 3.5. The following lemma will be useful.
Lemma D.1. Let A be any (ϵ, δ)-differentially private algorithm satisfying Assumption 1. Then, for every dataset W and item
x ∈ i W i , if N (x) ≤ 1 ϵ ln 1 + e ϵ -1 2δ , then P S∼A(W ) [x ∈ S] ≤ 1 2 .
Proof. Fix some dataset W . Let A be any randomized algorithm with privacy parameters ϵ ≤ 1 and δ ∈ (0, 1) satisfying Assumption 1. This means that for any item x ∈ i W i such that N W (x) = 1, we have that P S∼A(W ) [x ∈ S] ≤ δ, where for a dataset W , we define N W (x) := i 1{x ∈ W i }. Now, suppose x ∈ i W i is an item such that N W (x) = 2. We can always construct a neighboring dataset W ′ by removing a user such that N W ′ (x) = 1. Thus, we have that
P S∼A(W ) [x ∈ S] ≤ e ϵ P S∼A(W ′ ) [x ∈ S] + δ ≤ δe ϵ + δ = δ(e ϵ + 1).
More generally, by unraveling the recurrence, we have that for any x ∈ i W i with N W (x) = b,
P S∼A(W ) [x ∈ S] ≤ δ b-1 i=0 (e ϵ ) i = δ e ϵb -1 e ϵ -1 .
Since W was arbitrary, for any dataset W and any x ∈ i W i with N W (x) = b, we have that
P S∼A(W ) [x / ∈ S] ≥ 1 -δ e ϵb -1 e ϵ -1 .
Now, consider the exclusion probability p = 1/2. Our goal is to compute the largest b ⋆ such that for any dataset W , any ∈ S] ≥ 1 2 . Consider a dataset W ⋆ of size n, taking n sufficiently large, where
x ∈ i W i with N W (x) ≤ b ⋆ has P S∼A(W ) [x / ∈ S] ≥ 1 2 . It suffices to solve for b in the inequality 1 -δ e ϵb -1 e ϵ -1 ≥ 1/2, which yields b ≤ 1 ϵ ln 1 + e ϵ -1 2δ =: b ⋆ . ■ Lemma D.
N (r) N = Θ C r s for all ranks r ∈ [M ⋆ ], where M ⋆ = | i W ⋆
i | (note that such a dataset is possible if, for example, one restricts each user to contribute exactly a single item). We can lower bound the missing mass of A on W ⋆ as
E S∼A(W ⋆ ) [MM(W ⋆ , S)] ≥ 1 N x∈ i W ⋆ i ,N (x)≤b ⋆ P S∼A(W ⋆ ) [x / ∈ S] N (x) ≥ 1 2N x∈ i W ⋆ i ,N (x)≤b ⋆ N (x).
Our next goal will be to find the smallest rank r ⋆ such that N (r ⋆ ) ≤ b ⋆ . It suffices to solve the inequality CN r s ≤ b ⋆ for r. Doing so gives that r ≥ CN b ⋆ 1 s =: r ⋆ . Hence, for this W ⋆ , we have that
E S∼A(W ⋆ ) [MM(W ⋆ , S)] ≥ 1 2N x∈ i W ⋆ i ,N (x)≤b ⋆ N (x) = 1 2N M ⋆ r=r ⋆ N (r) = Ω 1 N M ⋆ r=r ⋆ CN r s .
Since s > 1, we can take M ⋆ (and hence n) to be large enough so that
Ω 1 N M ⋆ r=r ⋆ CN r s = Ω 1 N ∞ r ⋆ CN r s dr .
Thus,
E S∼A(W ⋆ ) [MM(W ⋆ , S)] = Ω 1 N ∞ r ⋆ CN r s dr = Ω C s -1 (r ⋆ ) 1-s = Ω C 1/s N (1-s)/s s -1 (b ⋆ ) (s-1)/s = Ω C 1/s N -(s-1)/s s -1 1 ϵ ln 1 + e ϵ -1 2δ (s-1)/s = Ω C 1/s s -1 • 1 ϵN (s-1)/s • ln 1 + e ϵ -1 2δ (s-1)/s , which completes the proof. ■ D.2 LOWER BOUNDS FOR TOP-k SELECTION Proof. (of Corollary 4.4) Define b ⋆ = 1 ϵ ln 1 + e ϵ -1 2δ
like in Lemma D.1. Recall from Lemma D.1, that for any dataset W and any item x ∈ i W i such that N (x) ≤ b ⋆ , we have that
P S∼A(W,k) [x / ∈ S] ≥ 1 2 . Consider any dataset W ⋆ such that | i W ⋆ i | = k and N (i) = b ⋆ for all i ∈ [k]. Then k i=1 N (i) = kb ⋆ . But, we also have that E S∼A(W,k) x∈S N (x) ≤ kb ⋆ 2 . Hence, E S∼A(W ⋆ ,k) MM k (W ⋆ , S) ≥ kb ⋆ 2N = Ωδ k ϵN , completing the proof. ■ D.3 LOWER BOUNDS FOR k-HITTING SET. Proof. (of Corollary 4.6) Define b ⋆ = 1 ϵ ln 1 + e ϵ -1 2δ
as in Lemma D.1. Recall from Lemma D.1, that for any dataset W and any item x ∈ i W i such that N (x) ≤ b ⋆ , we have that
P S∼A(W,k) [x / ∈ S] ≥ 1 2 .
This is due to our restriction that A(W, k) ⊆ i W i . Consider any dataset W ⋆ consisting of k unique items and n = kb ⋆ users such that each item hits a disjoint set of b ⋆ users. Since the frequency of each of the k items is at most b ⋆ , A can output each item with probability at most 1/2. Hence, in expectation, A outputs at most k/2 distinct items, hitting at most kb ⋆ 2 users, while the optimal set of items includes all k items and hits all users.
■
this section cite: []

Section: E EXPERIMENT DETAILS
In this section, we provide details about the datasets used in Section 5. Table 1 provides statistics for the 6 datasets we consider. Reddit (Gopi et al., 2020) is a text-data dataset of posts from r/askreddit. For this dataset, each user corresponds to a set of documents, and following prior methodology, we take a user's item set to be the set of tokens used across all documents. Movie Reviews (Harper & Konstan, 2015) is a dataset containing movie reviews from the MovieLens website. Here, we group movie reviews by user-id, and take a user's itemset to be the set of movies they reviewed. Amazon Games, Pantry, and Magazine (Ni et al., 2019) are review datasets for video games, prime pantry, and magazine subscriptions respectively. Like for Movie Reviews, we group rows by user-id and take a user's item set to the set of items they reviewed. Finally, Steam Games (Steam, 2025) is a dataset of 200k user interactions (purchases/play) on the Steam PC Gaming Hub. As before, we group the rows by user-id and take a user's itemset to the be the set of games they purchased/played.  (a) Steam Games (b) Amazon Magazine (c) Amazon Pantry Figure 5: Log-log plot of frequency vs. rank for small datasets E.2 USER ITEM SET SIZE DISTRIBUTIONS Figures 6 and 7 plot the Empirical CDF (ECDF) of user item set sizes for the large and small datasets respectively. (a) Reddit (b) Amazon Games (c) Movie Reviews Figure 6: ECDFs of user item set sizes for large datasets.  (a) Missing Mass (b) Missing Mass (c) Missing Mass Figure 8: MM as a function of ∆ 0 ∈ {1, 50, 100, 150, 200, 300} for the small datasets. F.1.2 RESULTS FOR ϵ = 0.10 Figures 9 and 10 plot the MM for the large and small datasets respectively when ϵ = 0.1 and δ = 10 -5 . (a) Missing Mass (b) Missing Mass (c) Missing Mass Figure 9: MM as a function of ∆ 0 ∈ {1, 50, 100, 150, 200, 300} for large datasets when ϵ = 0.1 and δ = 10 -5 . (a) Missing Mass (b) Missing Mass (c) Missing Mass Figure 10: MM as a function of ∆ 0 ∈ {1, 50, 100, 150, 200, 300} for small datasets when ϵ = 0.1 and δ = 10 -5 .
this section cite: ['b12', 'b13', 'b18']

Section: F.2 TOP-k SELECTION
The top-k ℓ 1 loss is defined as
ℓ k 1 (W, S) = min{|S|,k}i=1
|N (i) -N (S i )| + k i=min{|S|,k} N (i) ,
where S is any ordered sequence of items. Unlike the top-k MM, the top-k ℓ 1 loss cares about the order of items output, and hence is a more stringent measure of utility.
Figure 11 plots the top-k ℓ 1 MM for ϵ = 1.0, δ = 10 -5 , and ∆ 0 = 100. Similar to Figure 2, we observe that our method (purple) achieves significantly less Top-k ℓ 1 loss compared to all baselines.   F.3 k-HITTING SET Figure 13 plots the Number of missed users for the small datasets when ϵ = 0.1, δ = 10 -5 , and ∆ 0 = 100. We observe that our method (blue) performs comparably and sometimes outperforms the case where the domain i W i is public. (a) Steam Games (b) Amazon Magazine (c) Amazon Pantry Figure 13: Number of missed users vs. k for k ∈ {5, 10, 20, 50, 100, 200} with ϵ = 0.1, δ = 10 -5 , and ∆ 0 = 100.
this section cite: []

Section: A USEFUL CONCENTRATION INEQUALITIES
In this section, we review some basic concentration inequalities that we use in the main text. The first is the following Gaussian concentration equality.
Lemma A.1 (Gaussian concentration (Vershynin, 2018)). Let X 1 , . . . , X n be an iid sequence of mean-zero Gaussian random variables with variance σ 2 . Then, for every δ ∈ (0, 1), with probability at least 1 -δ, we have that
max i |X i | ≤ σ 2 log 2n δ .
The second is for the concentration of a sequence of Gumbel random variables. While this result is likely folklore, we provide a proof for completeness.
Lemma A.2. Let X 1 , . . . , X n be an iid sequence of Gumbel random variables with parameter λ. Then, for every δ ∈ (0, 1), with probability at least 1 -δ, we have that
max i |X i | ≤ λ • ln 2n δ .
this section cite: ['b25']

Section: References
Ref_id:b0 Title: Zipf's law and the Internet Year: (2002)
Ref_id:b1 Title: Plume: differential privacy at scale Year: ()
Ref_id:b2 Title: The price of selection in differential privacy Year: (2017)
Ref_id:b3 Title: Discovering frequent patterns in sensitive data Year: (2010)
Ref_id:b4 Title: Incorporating item frequency for differentially private set union Year: ()
Ref_id:b5 Title: Differentially private decomposable submodular maximization Year: ()
Ref_id:b6 Title: Scalable Private Partition Selection via Adaptive Weighting Year: ()
Ref_id:b7 Title: Differentially private partition selection Year: (2022)
Ref_id:b8 Title: Practical differentially private top-k selection with pay-whatyou-get composition Year: (2019)
Ref_id:b9 Title: Calibrating noise to sensitivity in private data analysis Year: (2006)
Ref_id:b10 Title: Zipf's law for cities: an explanation Year: (1999)
Ref_id:b11 Title: A Joint Exponential Mechanism For Differentially Private Top-k Year: ()
Ref_id:b12 Title: Differentially private set union Year: (2020)
Ref_id:b13 Title: The MovieLens Datasets: History and Context Year: (2015)
Ref_id:b14 Title: Reducibility among combinatorial problems Year: (1972)
Ref_id:b15 Title: Releasing search queries and clicks privately Year: (2009)
Ref_id:b16 Title: Permute-and-flip: A new mechanism for differentially private selection Year: (2020)
Ref_id:b17 Title: Differentially private submodular maximization: Data summarization in disguise Year: (2017)
Ref_id:b18 Title: Justifying recommendations using distantly-labeled reviews and fine-grained aspects Year: (2019)
Ref_id:b19 Title:  Year: (2025)
Ref_id:b20 Title: Zipf's word frequency law in natural language: A critical review and future directions Year: (2014)
Ref_id:b21 Title: Oneshot differentially private top-k selection Year: (2021)
Ref_id:b22 Title: LinkedIn's Audience Engagements API: A Privacy Preserving Data Analytics System at Scale Year: (2021)
Ref_id:b23 Title: Steam. Steam games dataset Year: (2025)
Ref_id:b24 Title: DP-SIPS: A simpler, more scalable mechanism for differentially private partition selection Year: (2023)
Ref_id:b25 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b26 Title: Differentially Private SQL with Bounded User Contribution Year: (2020)
Ref_id:b27 Title: Human behavior and the principle of least effort: An introduction to human ecology Year: (1949)
