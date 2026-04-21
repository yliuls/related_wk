Title: Low-degree evidence for computational transition of recovery rate in stochastic block model
Abstract: We investigate implications of the (extended) low-degree conjecture (recently formalized in [MW23]) in the context of the symmetric stochastic block model. Assuming the conjecture holds, we establish that no polynomial-time algorithm can weakly recover community labels below the Kesten-Stigum (KS) threshold. In particular, we rule out polynomial-time estimators that, with constant probability, achieve 𝑛 -0.49 correlation with the true communities. Whereas, above the KS threshold, polynomial-time algorithms are known to achieve constant correlation with the true communities with high probability [Mas14, AS15]. To our knowledge, we provide the first rigorous evidence for such a sharp transition in recovery rate for polynomial-time algorithms at the KS threshold. Notably, under a stronger version of the low-degree conjecture, our lower bound remains valid even when the number of blocks diverges. Furthermore, our results provide evidence of a computational-to-statistical gap in learning the parameters of stochastic block models. In contrast, prior work either (i) rules out polynomial-time algorithms with 1 -𝑜(1) success probability [Hop18, BBK + 21a] under the low-degree conjecture, or (ii) degree-poly(𝑘) polynomials for learning the stochastic block model [LG24]. For this, we design a hypothesis test which succeeeds with constant probability under symmetric stochastic block model, and 1 -𝑜(1) probability under the distribution of Erdős Rényi random graphs. Our proof combines low-degree lower bounds from [Hop18, BBK + 21a] with graph splitting and cross-validation techniques. In order to rule out general recovery algorithms, we employ the correlation preserving projection method developed in [HS17].

Section: Introduction
The stochastic block model (SBM) is among the most fundamental models in (social) network analysis and information theory, and has been intensively studied for decades [HLL83, MNS12, ABH15, KMM + 13, Abb18]. A fascinating phenomenon in the SBM is the sharp computational threshold for weak recovery of its hidden community structure: efficient algorithms are known for achieving constant correlation with the hidden signal when the signal-to-noise ratio is above a certain threshold [CO10, DKMZ11, Mas14, AS15], while no polynomial-time algorithms have been discovered below this threshold despite significant research effort. This computational threshold is known as the Kesten-Stigum threshold (KS threshold) in the statistical physics literature [DKMZ11], and it is an important topic in both probability theory and theoretical computer science.
this section cite: ['b17']

Section: Kesten-Stigum threshold in the symmetric stochastic block model.
For simplicity, we focus on the following special case of the stochastic block model.
Definition 1.1 (Symmetric 𝑘-stochastic block model SSBM(𝑛, 𝑑  𝑛 , 𝜀, 𝑘)). Let 𝑘 ∈ ℕ + be the number of communities, 𝑑 ∈ ℕ + be the average degree of the graph, 𝜀 ∈ [0, 1] be the bias parameter, and 𝑛 ∈ ℕ + be a multiple of 𝑘. A graph 𝑌 ∈ {0, 1} 𝑛×𝑛 1 follows the symmetric 𝑘-stochastic block model distribution SSBM(𝑛, 𝑑  𝑛 , 𝜀, 𝑘) if it is sampled in the following way: assign each vertex a label uniformly at random from [𝑘], then independently add edges with probability (1 + 𝑘-1 𝑘 𝜀) 𝑑 𝑛 between vertices with the same label and with probability (1 -𝜀 𝑘 ) 𝑑 𝑛 between vertices with different labels.
Note that, when the bias parameter 𝜀 = 0, the model reduces to Erdős-Rényi random graphs with average degree 𝑑, denoted by 𝔾(𝑛, 𝑑 𝑛 ). Given a graph sampled from SSBM(𝑛, 𝑑  𝑛 , 𝜀, 𝑘), the most fundamental problem is to recover the hidden community labels of the vertices, or equivalently to recover the community membership matrix 𝑀 • given by
𝑀 • 𝑖,𝑗 := 1 𝑥 • 𝑖 =𝑥 • 𝑗 - 1 𝑘 (𝑖, 𝑗 ∈ [𝑛]),
where 𝑥 • 𝑖 ∈ [𝑘] is the label of the 𝑖-th vertex. In this work, we consider weak recovery of 𝑀 • , that is to find a matrix 𝑀 which correlates with 𝑀 • in the following sense.
Definition 1.2 (Recovery rate and weak recovery in the SBM). For any 𝛿 ∈ [0, 1], an algorithm achieves recovery rate 𝛿 in the 𝑘-stochastic block model, if given a random graph sampled from SSBM(𝑛, 𝑑  𝑛 , 𝜀, 𝑘), it outputs a nonzero matrix 𝑀 ∈ ℝ 𝑛×𝑛 such that with constant probability,
⟨𝑀, 𝑀 • ⟩ ⩾ 𝛿∥𝑀 ∥ F ∥𝑀 • ∥ F .
If the recovery rate 𝛿 satisfies 𝛿 ⩾ Ω(1), then the algorithm is said to achieve weak recovery.
The difficulty of achieving weak recovery in the SBM appears to be closely related to the choice of parameters 𝜀, 𝑑, 𝑘. In particular, [Mas14,MS15] give polynomial-time algorithms for weak recovery above the KS threshold 𝜀 2 𝑑 ⩾ 𝑘 2 . On the other hand, while it is known that exponential-time algorithms can achieve weak recovery below this threshold (when 𝑘 ⩾ 5) [BMNN16], it is widely believed that no polynomial-time algorithms exist that achieve weak recovery when 𝜀 2 𝑑 < 𝑘 2 .
this section cite: ['b31', 'b34', 'b14']

Section: Rigorous evidence for average case complexity.
To provide rigorous evidence for the innate hardness of weak recovery below the Kesten-Stigum threshold, one could follow two general approaches. The first is to construct a reduction from problems widely believed to be hard (such as planted clique [BBH19,BB20] or learning with errors (LWE) [BRST20,GVV22,Tie24]). However, this approach is unlikely to be successful for our problem as no such reductions are known for (other) average-case problems with constant sharp statistical threshold. The second approach is to prove unconditional lower bounds that rule out certain classes of algorithms. For example, those based on sums of squares [BHK + 19, KMOW17, JPR + 22, 1For ease of notation, we will use the adjacency matrix and the graph interchangeably. MRX20], statistical queries [BKW03, Fel17, BBH + 20], or low-degree polynomials [HKP + 17, HS17, Hop18, KWB19]. As it appears that significant technical barriers have to be overcome to prove lower bounds against the former two classes for average-case problems with sharp statistical threshold, we focus on the latter.
The low-degree method for hypothesis testing. In recent years, the low-degree method has emerged as a standard tool for providing rigorous evidence for computational hardness in average-case problems [Hop18,KWB19]. Inspired by the fact that thresholding the likelihood ratio function provides optimal algorithms for hypothesis testing, the low-degree method provides a heuristic for average-case computational hardness by proving lower bounds against the low-degree projection of the likelihood ratio between two distributions from the hypothesis class. In fact, previous works[Hop18, BBK + 21a] has provided low-degree hardness evidence for the related hypothesis testing problems on distinguishing the stochastic block model and Erdős-Rényi distribution with probability 1 -𝑜(1). However, significant barrier needs to be overcome for extending their computational lower bound to weak recovery below KS threshold. The reason is that we want to rule out recovery algorithms which only need to succeed with constant probability, while below the Kesten-Stigum threshold, the two distributions considered in [Hop18, BBK + 21a] can be distinguished with probability strictly larger than 1/2 by counting triangles in the graph [BM17]. 2 Low-degree recovery lower bound. In this paper, we focus on the implications of the following conjecture of [MW23] in the context of the SBM3.
this section cite: ['b4', 'b3', 'b15', 'b20', 'b38', 'b33', 'b22', 'b28', 'b13', 'b35']

Section: Conjecture 1.3 (Low-degree conjecture).
Let 𝑃 be a distribution from the 𝑘-stochastic block model and 𝑄 be a distribution of Erdős-Rényi random graphs. For (randomized) functions 𝑓 : ℝ 𝑛×𝑛 → ℝ, consider the parameter
𝑅 𝑃,𝑄 ( 𝑓 ) 𝔼 𝑌∼𝑃 𝑓 (𝑌) 𝔼 𝑌∼𝑄 ( 𝑓 (𝑌)) 2 .
(1.1) Suppose that for any arbirary constant 𝛿 ∈ (0, 1], and every polynomial 𝑓 (•) of degree at most 𝑛 𝛿 , we have 𝑅 𝑃,𝑄 ( 𝑓 ) ⩽ 𝑂(1).
Then, for any function 𝑓 (•) computable in time exp(𝑛 0.99𝛿 ) taking values in [0, 1] satisfying 𝔼 𝑌∼𝑃 𝑓 (𝑌) ⩾ Ω(1) , we have 𝑅 𝑃,𝑄 ( 𝑓 ) ⩽ 𝑂(1) . The parameter 𝑅 𝑃,𝑄 (•) in (1.1) is motivated by Le Cam's method: if the maximum of 𝑅 𝑃,𝑄 ( 𝑓 ) over all computable functions 𝑓 is bounded by 𝑂(1), no algorithm can distinguish between the distributions 𝑃 and 𝑄 with high probability. Intuitively, Conjecture 1.3 tells us that, if the maximum of 𝑅 𝑃,𝑄 ( 𝑓 ) is bounded over all low-degree polynomials, it is in fact bounded over all efficiently computable functions. Assuming this conjecture for 𝑃 given by the symmetric SBM and 𝑄 given by the Erdős-Rényi graph distribution, we aim to clarify the relation between the upper bounds for the low-degree likelihood ratio (proved in [Hop18, BBK + 21a]) and lower bounds for computationally efficient algorithms for the SBM. Specifically, we address the following question: Question 1.4. Assuming Conjecture 1.3, can we rule out polynomial-time algorithms achieving weak recovery (or even non-trivial error rate) in the stochastic block model below the Kesten-Stigum threshold, when the number of communities is a universal constant?
Implications for learning stochastic block model. A potential computational-statistical gap similar to weak recovery can also be observed in the closely related problem of learning the parameters of the stochastic block model [Xu17]. Although [LG24] established a lowdegree recovery lower bound for learning the probability matrix in the symmetric SBM, their result does not imply a lower bound for learning the parameters 𝑑, 𝜀, as their hard instance is given by a symmetric SBM with fixed parameters. Moreover, when 𝑘 ≲ log(𝑛) (rather than constant), their lower bound cannot rule out polynomial-time algorithms for achieving the minimax error rate.
As such, we address the following question in this paper:
Question 1.5. Assuming Conjecture 1.3, can we provide rigorous evidence for a computational-statistical gap in the error rate of learning the parameters of the stochastic block model? 2 Main result 2.1 Computational lower bound for weak recovery in stochastic block model We provide the first rigorous evidence that no polynomial-time algorithms can achieve recovery rate 𝑛 -0.49 with constant probability below the Kesten-Stigum threshold, assuming Conjecture 1.3. Theorem 2.1 (Computational lower bound below the KS threshold, see Theorem B.1 for the full statement). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑂(1), 𝑑 ⩽ 𝑂(1). Assume that for any 𝑑 ′ ∈ ℕ + such that 0.999𝑑 ⩽ 𝑑 ′ ⩽ 𝑑, Conjecture 1.3 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 ′ 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by the Erdős-Rényi graph model 𝔾(𝑛, 𝑑 ′ 𝑛 ). Then, no exp 𝑛 0.99 time algorithm can achieve recovery rate 𝑛 -0.49 in the 𝑘-stochastic block model when 𝜀 2 𝑑 ⩽ 0.99𝑘 2 . Note that the algorithm which outputs a matrix M reflecting a random partition of the vertices into 𝑘 communities only achieves vanishing recovery rate 𝛿 ≲ 1/ √ 𝑛. In contrast, recall that above the KS threshold (when 𝜀 2 𝑑/𝑘 2 > 1), polynomial-time algorithms can achieve a recovery rate in Ω(1) (i.e., weak recovery). To our knowledge, this is the first result showing such a sharp transition in the recovery rate that can be achieved by efficient algorithms above and below the KS threshold4. SNR 𝑑𝜀 2 /𝑘 2 𝜆 𝐼𝑇 𝜆 𝐴𝑙𝑔 Impossible Regime (𝜆 < 𝜆 𝐼𝑇 ) No algorithm achieves constant correlation Hard Regime (𝜆 𝐼𝑇 < 𝜆 < 𝜆 𝐴𝑙𝑔 ) Sub-exponential time algorithms cannot achieve 𝑛 -0.49 correlation. Easy Regime (𝜆 > 𝜆 𝐴𝑙𝑔 ) Polynomial time algorithms achieve Ω(1) correlation. Under a strengthened low-degree conjecture (see Conjecture A.2 below), our lower bound extends to the regime where the number of communities can be as large as 𝑛 𝑜(1) . Theorem 2.2 (Computational lower bound for diverging number of blocks). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑛 𝑜(1) , 𝑑 ⩽ 𝑛 𝑜(1) . Assume that for any 𝑑 ′ ∈ ℕ + such that 0.999𝑑 ⩽ 𝑑 ′ ⩽ 𝑑, Conjecture A.2 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 ′ 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by the Erdős-Rényi graph model 𝔾(𝑛, 𝑑 ′ 𝑛 ). Then no exp 𝑛 0.99 time algorithm can achieve recovery rate 𝑛 -0.49 in the 𝑘-stochastic block model when 𝜀 2 𝑑 ⩽ 0.99𝑘 2 . Concurrent work. A concurrent work [SW25] also provides rigorous evidence for computational lower bound below KS threshold based on low-degree polynomials. They give the first unconditional low-degree recovery lower bound below the KS threshold in the stochastic block model. In our setting, for the task of weak recovery (which is to say, achieving constant recovery rate), their lower bound directly rules out estimators based on degree 𝑛 𝛿 polynomials, which captures many natural candidates of algorithms. Such a lower bound is beyond reach with techniques from [Hop18, BBK + 21a]. For this, they introduce significantly new techniques in analyzing the low-degree polynomials. In comparison, we give evidence that sub-exponential time algorithms cannot achieve recovery rate 𝑛 -0.49 with constant probability below the KS threshold, while polynomial time algorithms are known to achieve weak recovery above the threshold. Our techniques are based on relating the rate of recovery to the low-degree conjecture formulated in [MW23]. Notably, we did not prove new low-degree lower bounds for our main results, but exploited the existing results from [Hop18, BBK + 21a].
In a concurrent work, [Li25] established computational lower bounds for random graph matching below sharp thresholds, also based on a strengthened version of the low-degree conjecture. However, their polynomial-time reduction between hypothesis testing with lopsided success probability and recovery differs significantly from ours.
As we discuss next, our result also has implications for other learning tasks in the stochastic block model.
this section cite: ['b39', 'b29', 'b30']

Section: Computational lower bound for learning stochastic block model
We give computational lower bounds for learning the stochastic block model under two different error metrics: learning the edge connection probability matrix and the block graphon function.
Definition 2.3 (Edge connection probability matrix for the SBM).
In the symmetric stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), the edge connection probability matrix 𝜃 • ∈ [0, 1] 𝑛×𝑛 has entries 𝜃 • 𝑖,𝑗 = (1 + (𝑘-1)𝜀 𝑘 ) 𝑑 𝑛 if 𝑖, 𝑗 belong to the same community and 𝜃 • 𝑖,𝑗 = (1 -𝜀 𝑘 ) 𝑑 𝑛 if 𝑖, 𝑗 belong to different communities. Given a random graph sampled from distribution SSBM(𝜀, 𝑑, 𝑘, 𝑛), the simple polynomialtime algorithm based on 𝑘-SVD outputs a matrix 𝜃 ∈ [0, 1] 𝑛×𝑛 such that 𝔼∥𝜃-𝜃 • ∥ 2 𝐹 ⩽ 𝑂(2𝑘•𝑑) [Xu17, LG24]. On the other hand, exponential-time algorithms based on maximum-likelihood can give an estimator which achieves the optimal error rate 𝔼∥𝜃 -𝜃 • ∥ 2 F ⩽ 𝑂 log(𝑘) • 𝑑 + 𝑘 2 . We give rigorous evidence for the hardness of learning the edge connection probability matrix of symmetric SBMs, by proving the following computational lower bound. Theorem 2.4 (Computational lower bound for learning the SBM). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑛 𝑜(1) , 𝑑 ⩽ 𝑜(𝑛). Assume that for any 𝑑 ′ ∈ ℕ + such that 0.999𝑑 ⩽ 𝑑 ′ ⩽ 𝑑, Conjecture A.2 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 ′ 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by Erdős-Rényi graph model 𝔾(𝑛, 𝑑 ′ 𝑛 ). Then given graph 𝐺 ∼ SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), no exp 𝑛 0.99 time algorithm can output 𝜃 ∈ [0, 1] 𝑛×𝑛 achieving error rate ∥𝜃 -𝜃 • ∥ 2 F ⩽ 0.99𝑘𝑑/4 with constant probability, where 𝜃 • is the sampled edge connection probability matrix.
Our computational lower bound matches the guarantees of known efficient algorithms in [Xu17] up to constant factors.5 In comparison, [LG24] show that degree-ℓ polynomials cannot give error rate better than 𝑂(𝑘𝑑/ℓ 4 ). For standard low-degree conjectures [Hop18, KWB19, SW22], to give evidence of hardness for polynomial-time algorithms, the polynomial degree ℓ needs to be taken as large as log(𝑛). Therefore, their lower bound on the error rate can only match the guarantees of existing algorithm up to logarithmic factors. As a result, they cannot give evidence of a computational-statistical gap for the error rate when 𝑘 = 𝑂(log(𝑛)).6
Another error metric considered in [KTM + 15, BCS15, BCSZ18, CDD + 24] is learning the graphon function. In the context of the symmetric SBM, the graphon function is block-wise constant and given by:
5For completeness, we state this algorithmic result in Appendix F.2. 6We also obtain an unconditional low-degree recovery lower bound for learning the 𝑘-stochastic block model when 𝑘 ⩽ 𝑛 0.001 in Appendix D.
Definition 2.5 (Graphon in the symmetric SBM). Let 𝑑, 𝑘 ∈ ℕ + and 𝜀 ∈ [0, 1]. Consider the symmetric stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘). Let 𝐵 • ∈ [0, 1] 𝑘×𝑘 be the community connection probability matrix with diagonal entries given by (1 -𝜀
(𝑘-1) 𝑘 ) 𝑑 𝑛 and non-diagonal entries given by (1 + 𝜀 𝑘 ) 𝑑 𝑛 . Let 𝛾 : [0, 1] → [𝑘] be a mapping such that 𝛾(𝑥) = ⌈𝑘𝑥⌉. Then a function 𝑊 • : [0, 1] × [0, 1] → [0, 1] is a graphon generating distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) if 𝑊 • (𝑥, 𝑦) = 𝐵 • 𝛾(𝑥),𝛾(𝑦) . We note that in contrast with the edge connection probability matrix, the graphon function only depends on the parameters of the distribution 𝜀, 𝑑, 𝑘. Previous works [BCS15, BCSZ18, KTM + 15, CDD + 24] consider the following distance metric between graphons: Definition 2.6 (Gromov-Wasserstein distance between graphons). Let functions 𝑊 1 , 𝑊 2 : [0, 1] × [0, 1] → [0, 1]. We consider the Gromov-Wasserstein distance metric GW(𝑊 1 , 𝑊 2 ) min 𝜙 ∫ 1 0 ∫ 1 0 𝑊 1 (𝜙(𝑥), 𝜙(𝑦)) -𝑊 2 (𝑥, 𝑦) 2 𝑑𝑥𝑑𝑦 where the minimum is taken over all measure-preserving bĳective mappings. Given a graph sampled from SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), our goal is to output a graphon Ŵ minimizing GW( Ŵ , 𝑊 • ). [KTM + 15] obtains the minimax error rate GW( Ŵ , 𝑊 • ) ≲ 𝑑 2 𝑛 2 • 𝑘 2 𝑛𝑑 + log(𝑘) 𝑑 + 𝑘 𝑛 . However, existing polynomial-time algorithms [Xu17, CDD + 24] can only achieve error rate GW( Ŵ , 𝑊 • ) ≲ 𝑑 2 𝑛 2 • 𝑘 𝑑 + 𝑘 𝑛 . Although there is a lower bound showing that the error term 𝑑 𝑛 (𝑘/𝑛) 1/4 is informationtheoretically necessary[BCS15], it is not clear whether polynomial time algorithms can achieve better error rate than 𝑑 𝑛 𝑘/𝑑, especially if the graph is sparse. In this paper, we give the first rigorous evidence for a computational-statistical gap in learning the graphon function when the number of blocks 𝑘 is a sufficiently large constant. Theorem 2.7 (Computational lower bound for learning block graphon function). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑂(1), 𝑑 ⩽ 𝑜(𝑛). Assume that Conjecture 1.3 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by Erdős-Rényi graph model 𝔾(𝑛, 𝑑 𝑛 ). Then no exp 𝑛 0.99 time algorithm can output a poly(𝑛)-block graphon function Ŵ :
[0, 1] × [0, 1] → [0, 1] such that GW( Ŵ , 𝑊 • ) ⩽ 𝑑 3𝑛 𝑘
𝑑 with 1 -𝑜(1) probability under distribution 𝑃 and distribution 𝑄.
In comparison, [LG24] do not provide any lower bound for learning the graphon function since their hard instance is a symmetric SBM with fixed distribution parameters 𝜀, 𝑑, 𝑘.
this section cite: ['b39', 'b29', 'b29']

Section: Techniques

this section cite: []

Section: Lower bounds for weak recovery
In this section, we give an overview of the techniques we use to prove lower bounds for weak recovery in the stochastic block model with constant number of blocks 𝑘 and constant average degree 𝑑. That is, an overview of our proof of (a special case of) Theorem 2.1. Suppose for a contradiction that we have a polynomial-time recovery algorithm for SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), 𝜀 2 𝑑 ⩽ 0.99𝑘 2 , that achieves recovery rate 𝛿 ⩾ Ω(𝑛 -0.49 ), in the sense of Definition 1.2. Using this algorithm, we will construct a function 𝑓 (•), which can be evaluated in polynomial time, such that 𝑅 𝑃,𝑄 ( 𝑓 ) ⩾ 𝑛 Ω(1) . Here, 𝑅 𝑃,𝑄 (•) is the parameter (1.1) for the distributions 𝑃 = SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and 𝑄 = 𝔾(𝑑, 𝑛). Assuming the low-degree conjecture (Conjecture 1.3) for this 𝑃 and 𝑄, this implies that there exists a low-degree polynomial 𝑓 ′ with 𝑅 𝑃,𝑄 ( 𝑓 ′ ) ⩾ 𝜔(1). But, since 𝜀 2 𝑑 ⩽ 0.99𝑘 2 is below the KS threshold, this leads to a contradiction with the low-degree lower bound of [Hop18] (Theorem A.1).
In the remainder of this section, we show how to construct the function 𝑓 (•), and give a sketch of the proof that 𝑅 𝑃,𝑄 ( 𝑓 ) ⩾ 𝑛 Ω(1) .
this section cite: ['b22']

Section: Regularization via correlation-preserving projection.
We begin with the following tool, which allows us to regularize the estimators of the community membership matrix 𝑀 • provided by the recovery algorithm. Suppose that M0 is a matrix achieving correlation 𝛿 with 𝑀 • , i.e., satisfying ⟨ M0 , 𝑀 • ⟩ ⩾ 𝛿∥ M0 ∥ F ∥𝑀 • ∥ F . We show that M0 can be projected into a (small) convex set 𝒦 ⊆ ℝ 𝑛×𝑛 containing 𝑀 • , while preserving the correlation (up to a constant). Concretely, the convex set 𝒦 here is given by
𝒦 := 𝑀 ∈ [-1/𝛿, 1/𝛿] 𝑛×𝑛 : 𝑀 + 1 𝑘𝛿 1 1 ⊤ ⪰ 0 , Tr(𝑀 + 1 𝑘𝛿 1 1 ⊤ ) ⩽ 𝑛/𝛿 .
(3.1)
In particular, elements of 𝒦 have bounded entries and bounded nuclear norm, which will be crucial in later steps of the proof where we apply a Bernstein inequality. To achieve this, we make use of the correlation preserving projection from [HS17] (see Theorem F.1), which projects M0 onto a matrix M ∈ 𝒦 satisfying
⟨ M, 𝑀 • ⟩ ⩾ Ω(1) • 𝛿∥ M ∥ F ∥𝑀 • ∥ F ,
In addition, Theorem F.1 promises that ∥ M ∥ F = Θ(∥𝑀 • ∥ F ). Thus, we find that
⟨ M, 𝑀 • ⟩ ⩾ Ω(1) • 𝛿∥ M ∥ F ∥𝑀 • ∥ F ⩾ Ω(1) • 𝛿∥𝑀 • ∥ 2 F ⩾ 𝛿 • Ω(𝑛 2 ).
(3.2) Importantly, the correlation preserving projection can be implemented in polynomial time via semidefinite programming. See Lemma B.5 for details.
this section cite: ['b23']

Section: Testing statistics via cross validation.
The basic idea is to construct 𝑓 (•) via cross validation. Given a random graph 𝐺, we construct a subgraph 𝐺 1 with the same vertex set by subsampling each edge in 𝐺 independently with probability 1-𝜂, where 𝜂 > 0 is a small constant. Note that if 𝐺 is drawn from SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), then 𝐺 1 is distributed according to SSBM(𝑛, (1-𝜂) 𝑑 𝑛 , 𝜀, 𝑘). If 𝐺 is drawn from 𝔾(𝑛, 𝑑 𝑛 ), then 𝐺 1 is distributed according to 𝔾(𝑛, (1 -𝜂) 𝑑 𝑛 ). We run the polynomial-time recovery algorithm on 𝐺 1 to obtain an estimate M ∈ ℝ 𝑛×𝑛 of the community membership matrix 𝑀 • , which we regularize using the correlation preserving projection discussed above. Let 𝑌 2 denote the adjacency matrix of 𝐺 2 := 𝐺 \ 𝐸(𝐺 1 ). Our function 𝑓 (•) is then defined as Output: Test function 𝑓 (𝑌) ∈ {0, 1}.
𝑓 (𝑌) := 1, if ⟨ M, 𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ ⟩ ⩾ 𝑛 0.51 ,
this section cite: []

Section: Algorithm:
1. Obtain a subgraph 𝐺 1 of 𝐺 by subsampling each edge with probability 1 -𝜂.
2. Obtain an estimator M0 by running a recovery algorithm on the graph 𝐺 1 .
3. Obtain M by projecting M0 onto the set 𝒦 defined in (3.1) using the correlation preserving projection.
4. Return 𝑓 (𝑌) = 𝟙{⟨ M, 𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ ⟩ ⩾ 𝑛 0.51 },
where 𝑌 2 is the adjacency matrix of 𝐺 \ 𝐸(𝐺 1 ).
Lower bound on the expectation under the SBM.. First, we give a lower bound on the expectation of 𝑓 under the distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), i.e, on
𝔼 𝑌∼𝑃 𝑓 (𝑌) = ℙ 𝑌∼𝑃 ⟨ M, 𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ ⟩ ⩾ 𝑛 0.51 .
To do so, note that we may decompose 𝑌 2 -
𝜂𝑑 𝑛 𝟙𝟙 ⊤ = 𝜀𝜂𝑑 𝑛 𝑀 • + 𝑊 2 ,
where 𝑊 2 is a random matrix whose entries are independent with mean zero. Then, we have
⟨𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ = 𝜀𝜂𝑑 𝑛 ⟨𝑀 • , M⟩ + ⟨𝑊 2 , M⟩ .
The first term on the RHS above is large with constant probability by (3.2). We would like to apply a Bernstein inequality to the second term, but the matrices 𝑊 2 and M are not independent. However, as we show in Lemma E.3, they are approximately independent in the sense that there exists a zero-mean symmetric matrix W2 with independent entries, independent of M, so that each entry in W2 -𝑊 2 has variance bounded by 𝑂(𝑑 2 /𝑛 2 ).
For ease of presentation, we ignore the difference between W2 and 𝑊 2 for now, and assume that 𝑊 2 and M are independent. In this case, we note that ⟨𝑊 2 , M⟩ can be written as the summation of independent zero-mean random variables, namely
⟨𝑊 2 , M⟩ = 𝑖,𝑗 𝑊 2 (𝑖, 𝑗) M(𝑖, 𝑗) ,
where 𝑊 2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩽ 𝑂(1/𝛿) for each 𝑖, 𝑗 ∈ [𝑛]. (Here, we have used that M ∈ 𝒦 ).
Moreover, since ∥ M ∥ 2 F = Θ(∥𝑀 • ∥ 2 F ) = Θ(𝑛 2 ), we have 𝑖,𝑗 M(𝑖, 𝑗) 2 𝔼 𝑊 2 (𝑖, 𝑗) 2 ≲ 𝑑 𝑛 𝑖,𝑗 M(𝑖, 𝑗) 2 ⩽ 𝑂(𝑛 2 • 𝑑 𝑛 ) = 𝑂(𝑛𝑑) .
By the Bernstein inequality, and using the fact that 𝛿 ⩾ Ω(𝑛 -0.49 ), we then have
ℙ       𝑖,𝑗 𝑊 2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩾ 𝑛 0.501       ⩽ exp -𝑛 0.001 .
As result, when 𝑑 ⩽ 𝑂(1), 𝑘 ⩽ 𝑂(1), 𝜂 = Θ(1), with constant probability, we have
⟨𝑌 2 - 𝜂𝑑 𝑛 , M⟩ ⩾ 𝜂𝜀𝑑 𝑛 ⟨𝑀 • , M⟩ -𝑛 0.501 ≳ 𝛿𝑛 -𝑛 0.501 ≳ 𝑛 0.51 -𝑛 0.501 ⩾ Ω(𝑛 0.51 ) .
Therefore, we have 𝔼 𝑌∼𝑃 𝑓 (𝑌) ⩾ Ω(1).
Upper bound under the null distribution. Next, we give an upper bound on the expectation of 𝑓 under the Erdős-Rényi distribution 𝔾(𝑛, 𝑑 𝑛 ). Our proof shares many ingredients with the proof of the lower bound in the previous section. We show that with high probability under the distribution 𝔾(𝑑, 𝑛), we have 𝑔(𝑌) := ⟨𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ = 𝑜(𝑛 0.51 ) .
To do so, we again apply the argument that 𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ and M are approximately independent.
In particular, let 𝑊 2 = 𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ , for some i.i.d. zero-mean symmetric matrix W2 , independent of M, so that each entry in W2 -𝑊 2 has variance bounded by 𝑑 2 /𝑛 2 . By the triangle inequality, we have
⟨𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ ⩽ ⟨𝑊 2 -W2 , M⟩ + ⟨ W2 , M⟩ .
For simplicity, we again ignore the difference between W2 and 𝑊 2 here. By the same reasoning as above, and again relying on the properties of M guaranteed by the correlation preserving projection, we have the Bernstein inequality
ℙ ⟨ W2 , M⟩ ⩾ 𝑛 0.501 ⩽ exp -𝑛 0.01 .
As result, when 𝑑 ⩽ 𝑂(1) and 𝑘 ⩽ 𝑂(1), we have 𝑔(𝑌) ⩽ 𝑜(𝑛 0.501 ) with probability at least 1 -exp(-𝑛 0.01 ) and thus 𝔼 𝑌∼𝑄 𝑓 (𝑌) ⩽ exp(-𝑛 0.01 ).
this section cite: []

Section: Finishing the proof.
Using the lower and upper bound established above, and the fact that 𝑓 (•) ∈ {0, 1}, we get that
𝔼 𝑌∼𝑃 𝑓 (𝑌) -𝔼 𝑌∼𝑄 𝑓 (𝑌) Var 𝑌∼𝑄 ( 𝑓 (𝑌))
⩾ Ω(1) exp(-𝑛 0.01 ) ⩾ exp(𝑛 0.01 ) ⩾ 𝜔(1).
this section cite: []

Section: Lower bound for learning the stochastic block model
In this section, we give an overview of the techniques used to prove our results on learning the stochastic block model, stated in Section 2.2.
this section cite: []

Section: Lower bound for learning edge connection probability matrix.
We sketch the proof of Theorem 2.4. We show that if an 𝑂(exp 𝑛 0.99 )-time algorithm can learn the edge connection probability matrix 𝜃 • such that with constant probability, the error rate ∥ θ -𝜃 • ∥ 2 F ⩽ 0.99𝑘𝑑, then an algorithm with running time exp 𝑛 0.99 can achieve weak recovery when 𝜀 2 𝑑 ⩾ 0.99𝑘 2 .
The key observation is that, for the symmetric stochastic block model, the edge connection probability matrix is given by 𝜃
• = (1-𝜂)𝜀𝑑 𝑛 𝑀 • + (1-𝜂)𝑑
𝑛 , where 𝑀 • ∈ {1 -1/𝑘, -1/𝑘} 𝑛×𝑛 is the community membership matrix. Therefore, when the estimation error is smaller than 0.99 √ 𝑘𝑑, the estimator θ -𝑑 𝑛 achieves weak recovery under the distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), which contradicts the extended low-degree conjecture (Conjecture A.2).
this section cite: []

Section: Lower bound for learning graphon function.
We sketch the proof of Theorem 2.7. Let 𝑊 0 be the graphon function underlying the distribution 𝔾(𝑛, 𝑑 𝑛 ) and 𝑊 1 be the graphon function underlying the distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘). We then have GW(𝑊 0 , 𝑊 1 ) ⩾ 𝑑 𝑛 0.99𝑘 𝑑 when
𝜀 2 𝑑 ⩾ 0.99𝑘 2 .
Now suppose there is a polynomial-time algorithm which, given a random graph 𝐺 sampled from an arbitrary symmetric 𝑘-stochastic block model, outputs an 𝑛-block graphon function Ŵ : [0, 1] × [0, 1] → [0, 1] achieving error 𝑑 3𝑛 𝑘 𝑑 with probability 1 -𝑜(1). Then one can construct a testing statistic by taking
𝑓 (𝑌) = 1, if GW( Ŵ , 𝑊 0 ) ⩽ 3𝑑 𝑛 𝑘 𝑑 , 0, otherwise.
We have 𝑓 (𝑌) = 1 with probability 1 -𝑜(1) under the distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and 𝑓 (𝑌) = 0 with probability 1 -𝑜(1) under the distribution 𝔾(𝑛, 𝑑 𝑛 ). Therefore, we have
𝑅 𝑃,𝑄 ( 𝑓 ) ⩾ 𝜔(1). Since the function 𝑓 (•) can be evaluated in polynomial time, this contradicts the low-degree lower bound (Theorem A.1), assuming Conjecture 1.3. A Preliminaries A.1 Low-degree framework Low-degree likelihood ratio lower bound in the SBM. The low-degree likelihood ratio lower bound is a standard framework to provide evidence of hardness for hypothesis testing problems. [Hop18, BBK + 21a] prove the following theorem7 on the low-degree lower bound for the stochastic block model: Theorem A.1 (Low-degree lower bound for SBM, Thm. 2.20 in [BBK + 21a]). Let 𝑑 = 𝑜(𝑛), 𝑘 = 𝑛 𝑜(1) and 𝜀 ∈ [0, 1]. Let 𝜇 : {0, 1} 𝑛×𝑛 → ℝ be the relative density of SSBM(𝑛, 𝑑, 𝜀, 𝑘) with respect to 𝐺 𝑛, 𝑑 𝑛 . Let 𝜇 ⩽ℓ be the projection of 𝜇 to the degree-ℓ polynomials with respect to the norm induced by 𝐺 𝑛, 𝑑 𝑛 For any constant 𝛿 > 0,
𝜇 ⩽ℓ is        ⩾ 𝑛 Ω(1) , if 𝜀 2 𝑑 > (1 + 𝛿)𝑘 2 , ℓ ⩾ 𝑂(log 𝑛) ⩽ 𝑂 𝛿 exp(𝑘 2 ) , if 𝜀 2 𝑑 < (1 -𝛿)𝑘 2 , ℓ < 𝑛 0.99
Assuming the low-degree conjectures in [Hop18,KWB19], this gives rigorous hardness evidence for distinguishing
• planted distribution 𝑃: symmetric 𝑘-stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), • null distribution 𝑄: Erdős-Rényi random graph 𝔾(𝑛, 𝑑 𝑛 ), with probability 1 -𝑜(1) when 𝑘 is a universal constant and 𝜀 2 𝑑 ⩽ 0.99𝑘 2 . However, even assuming the low-degree conjectures for hypothesis testing from [Hop18, KWB19], these works do not rule out polynomial-time weak recovery algorithms under our definition (i.e., algorithms that achieve constant correlation with constant probability). Extended low-degree hypothesis. To show our lower bounds in the regime where 𝑘 is polylogarithmic, Conjecture 1.3 is not sufficient. Instead, we rely on the following stronger low-degree hypothesis from [MW23]. Conjecture A.2 (Extended low-degree conjecture). Let 𝑃 be a distribution from the 𝑘-stochastic block model and 𝑄 be a distribution of Erdős-Rényi random graphs. Consider the hypothesis testing problem between 𝑌 ∼ 𝑃 and 𝑌 ∼ 𝑄 for distribution 𝑃 and 𝑄. Let 𝑅 𝑃,𝑄 ( 𝑓 ) 𝔼 𝑌∼𝑃 𝑓 (𝑌) √ 𝔼 𝑌∼𝑄 ( 𝑓 (𝑌)) 2 . Let 𝛿 ∈ (0, 1]. For any (randomized) function 𝑓 (•) computable in time exp(𝑛 0.99𝛿 ) taking values in [0, 1] and satisfying 𝔼 𝑃 𝑓 (𝑌) ⩾ Ω(1), we have
𝑅 𝑃,𝑄 ( 𝑓 ) ≲ max deg( 𝑓 )⩽𝑛 𝛿 𝑅 𝑃,𝑄 ( 𝑓 ) .
When max deg( 𝑓 )⩽𝑛 𝛿 𝑅 𝑃,𝑄 ( 𝑓 ) ⩽ 𝑂(1), the conjecture is reduced to Conjecture 1.3. The extended low-degree hypothesis is closely related to the low-degree lower bound for random optimization problems (see [GJW24]).
this section cite: ['b22', 'b28', 'b19']

Section: A.2 Organization
The rest of the paper is organized as follows. We present our main proof ideas in Section 3. In Appendix B, we give the formal proof of our computational lower bounds for recovery algorithms conditional on the low-degree conjectures (i.e., the proof of Theorem 2.1). In 7Although the original theorem statement is for constant 𝑘, 𝑑, it is easy to see in their analysis that the lower bound holds when 𝑑 = 𝑜(𝑛). Also a weaker version of the theorem is stated in Thm. 8.6.1 of [Hop18].
Appendix C, we give proofs of our computational lower bounds for parameter learning algorithms conditional on the low-degree conjectures (i.e., the proof of Theorem 2.4 and Theorem 2.7). In Appendix E, we introduce some facts from probability theory used in our paper. In Appendix F, we introduce some existing algorithms from the literature that are used in our paper. In Appendix G, we clarify the upper bound on the low-degree likelihood ratio when the number of blocks 𝑘 diverges, which is implicitly obtained in [BBK + 21a].
this section cite: ['b22']

Section: B Computational lower bound for recovery
In this section, we prove Theorem 2.1 by showing that there exists an efficient algorithm that reduces testing to weak recovery in SBM. We will show that there exists a efficiently computable testing function (shown in Algorithm B.2) that is large with constant probability if the input is sampled from SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and is small with high probability if the input is sampled from 𝔾(𝑛, 𝑑/𝑛). This will lead to a contradiction with low-degree lower bounds of testing if we assume Conjecture 1.3.
Before describing the algorithm, we restate Theorem 2.1 here for completeness. Theorem B.1 (Full version of Theorem 2.1). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑂(1), 𝑑 ⩽ 𝑛 𝑜(1) . Assume that for any 𝑑 ′ ∈ ℕ + such that 0.999𝑑 ⩽ 𝑑 ′ ⩽ 𝑑, Conjecture 1.3 holds for distribution 𝑃 = SSBM(𝑛, 𝑑 ′ 𝑛 , 𝜀, 𝑘) and distribution 𝑄 = 𝔾(𝑛, 𝑑 ′ 𝑛 ). Then for any small constants 𝛿 1 , 𝛿 2 , no exp 𝑛 0.99 time algorithm can achieve recovery rate 𝑛 -0.5+𝛿 1 in the 𝑘-stochastic block model when
𝜀 2 𝑑 ⩽ (1 -𝛿 2 )𝑘 2 .
The reduction that we consider is the following.
this section cite: []

Section: Algorithm B.2 (Reduction from testing to weak recovery).
Input: A random graph 𝐺 with equal probability sampled from Erdős-Rényi model or stochastic block model, and target recovery rate 𝛿, parameters 𝜀, 𝑘, 𝑑. Output: Testing statistics 𝑔(𝑌) ∈ ℝ, where 𝑌 is the adjacency matrix.
this section cite: []

Section: Algorithm:
1. Let 𝜂 = 0.001𝛿 2 , where 𝛿 2 = 1 -𝜀 2 𝑑/𝑘 2 . Obtain subgraph 𝐺 1 by subsampling each edge with probability 1 -𝜂, and let 𝐺 2 = 𝐺 \ 𝐺 1 .
2. Obtain estimator M0 by running weak recovery algorithm on graph 𝐺 1 .
3. Obtain M by applying correlation preserving projection (see Theorem F.1) on M0 to the set
𝒦 = 𝑀 ∈ [-1/𝛿, 1/𝛿] 𝑛×𝑛 : 𝑀 + 1 𝑘𝛿 1 1 ⊤ ⪰ 0 , Tr(𝑀 + 1 𝑘𝛿 1 1 ⊤ ) ⩽ 𝑛/𝛿 . 4. Return testing statistics 𝑔(𝑌) = ⟨ M, 𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ ⟩,
where 𝑌 2 is the adjacency matrix for the graph 𝐺 2 .
To prove Theorem B.1, we will show that the testing statistics 𝑔(𝑌) from Algorithm B.2 satisfies the following two lemmas.
Lemma B.3. Let 𝑌 be the adjacency matrix of the graph sampled from the symmetric 𝑘-stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and 𝑀 • ∈ {-1/𝑘, 1 -1/𝑘} 𝑛×𝑛 be the corresponding community membership matrix. Suppose that ⟨ M0 , 𝑀 • ⟩ ⩾ 𝑛 -0.5+𝛿 1 ∥ M0 ∥ F ∥𝑀 • ∥ F and ∥ M ∥ F = Θ(∥𝑀 • ∥ F ).
Then Algorithm B.2 outputs testing statistics 𝑔(𝑌) ∈ ℝ such that 𝑔(𝑌) ⩾ Ω 𝑛 0.
5(1+𝛿 1 ) . Lemma B.4. Let 𝑌 be the adjacency matrix of the graph sampled from Erdős-Rényi random graph 𝔾(𝑛, 𝑑/𝑛). With probability at least 1 -exp(-𝑛 0.001𝛿 1 ), Algorithm B.2 outputs 𝑔(𝑌) ⩽ 𝑂(𝑛 0.5+𝛿 1 /3 ) in polynomial time. Combining Lemma B.3 and Lemma B.4, Theorem 2.1 follows as a corollary. Proof of Theorem 2.1. Suppose that there is a exp 𝑛 0.99 time algorithm which outputs estimator M0 such that ⟨ M0 , 𝑀 • ⟩ ⩾ 𝑛 -0.5+𝛿 1 ∥ M0 ∥ F ∥𝑀 • ∥ F . Let 𝑓 (𝑌) = 1 𝑔(𝑌)⩾0.001𝑛 0.5+𝛿 1 /2 . When 𝜀 2 𝑑 ⩾ Ω(𝑘 2 ), combining Lemma B.3 and Lemma B.4, we have 𝔼 𝑃 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌)) ⩾ exp(𝑛 0.001𝛿 1 ) . By the low-degree likelihood ratio upper bound Theorem A.1, when 𝜀 2 𝑑 ⩽ (1 -𝛿 2 )𝑘 2 , we have max deg( 𝑓 )⩽𝑛 0.01 𝔼 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌))
⩽ exp(𝑘 2 ) .
Since 𝑓 (𝑌) can be evaluated in 𝑂(exp 𝑛 0.99 ) time, assuming Conjecture 1.3, we then have
𝔼 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌)) ≲ max deg( 𝑓 )⩽𝑛 0.01 𝔼 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌))
⩽ 𝑂(1) , which leads to a contradiction. As a result, assuming Conjecture 1.3, we cannot achieve weak recovery in exp 𝑛 0.99 time when 𝜀 2 𝑑 ⩽ (1 -𝛿 2 )𝑘 2 . □
this section cite: []

Section: B.1 Correlation preserving projection
In this part, we prove that we can project the estimator into the set of matrices with bounded entries and bounded nuclear norm, while preserving correlation.
Lemma B.5. Let 𝑀 • ∈ {-1/𝑘, 1 -1/𝑘} 𝑛×𝑛 be a symmetric matrix with rank-(𝑘 + 1). For any 𝛿 ⩽ 𝑂(1), given matrix M0 such that ⟨ M0 , 𝑀 • ⟩ ⩾ 𝛿∥ M0 ∥ F ∥𝑀 • ∥ F , there is a polynomial time algorithm which outputs M ∈ 𝒦 such that ⟨ M, 𝑀 • ⟩ ⩾ Ω(1) • 𝛿∥ M ∥ F ∥𝑀 • ∥ F and ∥ M ∥ F ⩾ Ω(∥𝑀 • ∥ F ), where
𝒦 = 𝑀 ∈ [-1/𝛿, 1/𝛿] 𝑛×𝑛 : 𝑀 + 1 𝑘𝛿 1 1 ⊤ ⪰ 0 , Tr(𝑀 + 1 𝑘𝛿 1 1 ⊤ ) ⩽ 𝑛/𝛿 .
Proof. We apply the correlation preserving projection from [HS17] (restated in Theorem F.1).
this section cite: ['b23']

Section: By definition, 𝑀
• = 𝑋 • (𝑋 • ) ⊤ -1 𝑘 1 1 ⊤ is in 𝒦 .
Let 𝑁 be the matrix that minimizes ∥𝑁 ∥ F subject to 𝑁 ∈ 𝒦 ′ and ⟨𝑁 , M0 ⟩ ⩾ 𝛿∥𝑀 • ∥ F ∥ M0 ∥ F , where
𝒦 ′ = 𝑀 ∈ [-1, 1] 𝑛×𝑛 : 𝑀 + 1 𝑘 1 1 ⊤ ⪰ 0 , Tr(𝑀 + 1 𝑘 1 1 ⊤ ) ⩽ 𝑛 .
Using ellipsoid method, this semidefinite program can be solved in polynomial time. By Theorem F.1, we have ⟨𝑁 , 𝑀
• ⟩ ⩾ Ω(1) • 𝛿∥𝑁 ∥ F ∥𝑀 • ∥ F and ∥𝑁 ∥ F ⩾ 𝛿∥𝑀 • ∥ F . We let M = ∥𝑀 • ∥ F ∥𝑁 ∥ F •𝑁. Then it follows that M ∈ 𝒦 , ∥ M ∥ F = ∥𝑀 • ∥ F and ⟨ M, 𝑀 • ⟩ ⩾ Ω(𝛿)∥ M ∥ F • ∥𝑀 • ∥ F . □
this section cite: []

Section: B.2 Proof of Lemma B.3
In this section, we prove Lemma B.3.
Proof of Lemma B.3. We consider the decomposition that 𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ = 𝜀𝜂𝑑 𝑛 𝑀 • + 𝑊 2 where 𝑊 2 is a symmetric random matrix with independent and zero mean entries. By Lemma E.3, there exists an i.i.d zero mean symmetric matrix W2 that is independent with 𝑌 1 , and satisfies that the entries in W2 -𝑊 2 are independent with zero mean and have variance bounded by 𝑂(𝑑 3 /𝑛 3 ), conditioning on the subsampled graph 𝑌 1 and community matrix 𝑀 • . As result, we have
⟨𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ = ⟨ 𝜀𝜂𝑑 𝑛 𝑀 • , M⟩ + ⟨𝑊 2 -W2 , M⟩ + ⟨ W2 , M⟩ .
For the first term ⟨ 𝜀𝜂𝑑 𝑛 𝑀 • , M⟩, it follows from Lemma B.5 that
⟨𝑀 • , M⟩ ⩾ Ω 𝜀𝜂𝑑 𝑛 𝛿∥ M ∥ F ∥𝑀 • ∥ F ⩾ Ω 𝛿𝜀𝜂𝑑 𝑛 ∥𝑀 • ∥ 2 F .
As with probability at least 1 -exp(-𝑛 0.001 ), we have ∥𝑀 • ∥ 2 F ⩾ Ω(𝑛 2 ), and as result ⟨𝑀 • , M⟩ ⩾ Ω(𝑛𝛿𝜀𝑑).
For bounding the second term ⟨𝑊 2 -W2 , M⟩, we condition on the subsampled graph 𝑌 1 and the community matrix 𝑀 • . With probability at least 1 -exp(-𝑛 𝛿 1 ), we have
|⟨𝑊 2 -W2 , M⟩| ⩽ ∥𝑊 2 -W2 ∥ F • ∥ M ∥ F ≲ 𝑑 3 𝑛 3 • 𝑛 2+𝛿 1 • 𝑛 2 = √ 𝑛 1+𝛿 1 𝑑 3 .
For the third term ⟨ W2 , M⟩, we again conditional on the subsampled graph 𝑌 1 and the community matrix 𝑀 • . We note that it can be written as the summation of independent zero-mean random variables ⟨ W2 , M⟩ = 𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) .
where W2 (𝑖, 𝑗) M(𝑖, 𝑗) are independent zero mean variables bounded by 𝑂(1/𝛿) for all 𝑖 ⩽ 𝑗.
Moreover, we have
𝑖,𝑗 M(𝑖, 𝑗) 2 𝔼 W2 (𝑖, 𝑗) 2 ≲ 𝑑 𝑛 𝑖,𝑗 M(𝑖, 𝑗) 2 ⩽ 𝑂(𝑛 2 • 𝑑 𝑛 ) = 𝑂(𝑛𝑑) .
By Bernstein inequality, we have
ℙ       𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩾ 100𝑡       ⩽ exp -𝑡 2 /(𝑛𝑑 + 𝑡/𝛿) .
Taking 𝑡 = 𝑛 (1+𝛿
1 )/2 √ 𝑑 and 𝛿 ⩾ 𝑛 -0.5+𝛿 1 , we have ℙ       𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩾ 𝑛 0.5(1+𝛿 1 ) √ 𝑑       ⩽ exp -𝑛 𝛿 1 /2 . As a result, when 𝑑 ⩽ 𝑛 𝑜(1) , 𝑘 ⩽ 𝑛 𝑜(1) , 𝛿 ⩾ 𝑛 -0.5+𝛿 1 , 𝜀 = Θ(1/ √ 𝑑), with constant probability, we have ⟨𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ ⩾ Ω(𝑛 0.5+𝛿 1 √ 𝑑) . □ B.3 Proof of Lemma B.4 In this section, we prove Lemma B.4. Proof of Lemma B.4. We will use the fact that 𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ and M are approximately independent. More precisely, let 𝑊 2 = 𝑌 2 -𝜂𝑑 𝑛 𝟙𝟙 ⊤ , by Lemma E.3, there exists symmetric zero mean matrix W2 with independent entries such that each entry in W2 -𝑊 2 has zero mean variance bounded by 𝑂(𝑑 3 /𝑛 3 ) conditioning on M. By triangle inequality, we have
𝑔(𝑌) = ⟨𝑌 2 - 𝜂𝑑 𝑛 𝟙𝟙 ⊤ , M⟩ ⩽ ⟨𝑊 2 -W2 , M⟩ + ⟨ W2 , M⟩ .
For bounding the first term ⟨𝑊 2 -W2 , M⟩, we condition on the subsampled graph 𝑌 1 . With probability at least 1 -exp(-𝑛 𝛿 1 /3 ), we have
|⟨𝑊 2 -W2 , M⟩| ⩽ ∥𝑊 2 -W2 ∥ F • ∥ M ∥ F ≲ 𝑑 3 𝑛 3 • 𝑛 2+𝛿 1 /3 • 𝑛 2 = 𝑑 3 𝑛 1+𝛿 1 /3
. For the second term, we note that ⟨ W2 , M⟩ can be written as the summation of independent zero-mean random variables ⟨ W2 , M⟩ = 𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) .
where W2 (𝑖, 𝑗) M(𝑖, 𝑗) are independent zero mean variables bounded by 𝑂(1/𝛿) for 𝑖 ⩽ 𝑗.
this section cite: []

Section: Moreover, we have
𝑖,𝑗 M(𝑖, 𝑗) 2 𝔼 W2 (𝑖, 𝑗) 2 ≲ 𝑑 𝑛 𝑖,𝑗 M(𝑖, 𝑗) 2 ⩽ 𝑂(𝑛 2 • 𝑑 𝑛 ) = 𝑂(𝑛𝑑) .
By Bernstein inequality, we have
ℙ       𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩾ 100𝑡       ⩾ exp -𝑡 2 /(𝑛𝑑 + 𝑡/𝛿) .
Taking 𝑡 = 𝑛 0.5+𝛿 1 /3 √ 𝑑 and 𝛿 ⩾ 𝑛 -0.5+𝛿 1 , we have
ℙ       𝑖,𝑗 W2 (𝑖, 𝑗) M(𝑖, 𝑗) ⩾ 𝑛 0.5+𝛿 1 /3       ⩽ exp -𝑛 0.001𝛿 1 .
this section cite: []

Section: □

this section cite: []

Section: B.4 Proof of Theorem 2.2
In this part, we give the proof of Theorem 2.2, which is the same as the proof of Theorem 2.1 except that we assume stronger low-degree conjecture.
Proof of Theorem 2.2. Suppose that there is a polynomial time algorithm which outputs
estimator M0 such that ⟨ M0 , 𝑀 • ⟩ ⩾ 𝑛 -0.5+𝛿 1 ∥ M0 ∥ F ∥𝑀 • ∥ F . Let 𝑓 (𝑌) = 1 𝑔(𝑌)⩾0.001𝑛 0.5+𝛿 1 /2 . When 0.001𝑘 2 ⩽ 𝜀 2 𝑑 ⩽ (1 -𝛿 2 )𝑘 2 ,
combining Lemma B.3 and Lemma B.4, we have 𝔼 𝑃 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌)) ⩾ exp(𝑛 0.001 ) . Since 𝑓 (𝑌) can be evaluated in 𝑂(exp(𝑛 0.001 )) time, assuming Conjecture 1.3, by [Hop18](stated in Theorem A.1), we have 𝔼 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌)) ≲ max deg( 𝑓 )⩽𝑛 0.99 𝔼 𝑓 (𝑌) Var 𝑄 ( 𝑓 (𝑌)) ⩽ exp(𝑘 2 ) . When 𝑘 2 ⩽ 𝑛 0.001 , this leads to a contradiction. As a result, assuming Conjecture A.2, we cannot achieve recovery rate 𝑛 -0.5+𝛿 1 in polynomial time when 𝜀 2 𝑑 ⩽ (1 -Ω(1))𝑘 2 . □ C Computational lower bound for learning stochastic block model C.1 Computational lower bound for learning the edge connection probability matrix In this section, we prove Theorem 2.4 by showing that there exists an efficient algorithm that reduces testing to learning in SBM. The reduction of algorithm Algorithm C.2 is similar to that of Algorithm B.2. The proof of Theorem 2.4 is also a similar proof by contradiction to the proof of Theorem 2.1. Before describing the algorithm, we restate Theorem 2.4 here for completeness. Theorem C.1 (Restatement of Theorem 2.4). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑛 𝑜(1) , 𝑑 ⩽ 𝑜(𝑛). Assume that for any 𝑑 ′ ∈ ℕ + such that 0.999𝑑 ⩽ 𝑑 ′ ⩽ 𝑑, Conjecture A.2 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 ′ 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by Erdős-Rényi graph model 𝔾(𝑛, 𝑑 ′ 𝑛 ). Then given graph 𝐺 ∼ SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), no exp 𝑛 0.99 time algorithm can output 𝜃 ∈ [0, 1] 𝑛×𝑛 achieving error rate ∥𝜃 -𝜃 • ∥ 2 F ⩽ 0.99𝑘𝑑/4 with constant probability, where 𝜃 • is the ground truth sampled edge connection probability matrix.
The reduction that we consider is the following.
this section cite: []

Section: Algorithm C.2 (Reduction from testing to learning).
Input: A random graph 𝐺 with equal probability sampled from Erdős-Rényi model or stochastic block model. Output: Testing statistics 𝑔(𝑌) ∈ ℝ, where 𝑌 is the centered adjacency matrix Algorithm:
1. Obtain subgraph 𝐺 1 by subsampling each edge with probability 1 -𝜂 = 0.999, and let 𝐺 2 = 𝐺 \ 𝐺 1 .
2. Run learning algorithm on 𝐺 1 , and obtain estimator θ ∈ ℝ 𝑛×𝑛
3. Obtain M by running correlation preserving projection on θ -𝑑 𝑛 1 1 ⊤ to the set
𝒦 = 𝑀 ∈ [-1, 1] 𝑛×𝑛 : 𝑀 + 1 𝑘 1 1 ⊤ ⪰ 0 , Tr(𝑀 + 1 𝑘 1 1 ⊤ ) ⩽ 𝑛 .
⟨ θ - 𝑑 𝑛 1 1 ⊤ , 𝑀 • ⟩ = ⟨ θ -𝜃 • , 𝑀 • ⟩ + ⟨𝜃 • - 𝑑 𝑛 1 1 ⊤ , 𝑀 • ⟩ = ⟨ θ -𝜃 • , 𝑀 • ⟩ + ⟨ 𝜀𝑑 𝑛 𝑀 • , 𝑀 • ⟩ .
For the first term, since with constant probability, ∥ θ -𝜃 • ∥ F ⩽ √ 0.99𝑘𝑑, we have
⟨ θ -𝜃 • , 𝑀 • ⟩ ⩽ ∥𝑀 • ∥ F ∥ θ -𝜃 • ∥ F ⩽ ∥𝑀 • ∥ F √ 0.99𝑘𝑑 .
For the second term, since with overwhelming high probability,
∥𝑀 • ∥ F ⩾ 𝑛 √ 𝑘 (1 -1 𝑘 ), we have ⟨ 𝜀𝑑 𝑛 𝑀 • , 𝑀 • ⟩ = 𝜀𝑑 𝑛 ∥𝑀 • ∥ 2 F ⩾ 𝜀𝑑 2 √ 𝑘 ∥𝑀 • ∥ F .
Therefore, when 𝜀 2 𝑑 > 0.999𝑘 2 , we have
⟨ θ - 𝑑 𝑛 1 1 ⊤ , 𝑀 • ⟩ ⩾ 𝜀𝑑 2 √ 𝑘 ∥𝑀 • ∥ F -∥𝑀 • ∥ F √ 0.99𝑘𝑑 2 ⩾ Ω 𝜀𝑑∥𝑀 • ∥ F √ 𝑘 .
On the other hand, by triangle inequality
θ - 𝑑 𝑛 1 1 ⊤ F ⩽ θ -𝜃 • F + 𝜃 • - 𝑑 𝑛 1 1 ⊤ F ⩽ 𝑂( √ 𝑘𝑑 + 𝜀𝑑 √ 𝑘 ) ⩽ 𝑂 𝜀𝑑/ √ 𝑘 ,
Therefore we have
⟨ θ - 𝑑 𝑛 1 1 ⊤ , 𝑀 • ⟩ ⩾ Ω(∥𝑀 • ∥ F • ∥ θ - 𝑑 𝑛 1 1 ⊤ ∥ F ) .
We thus conclude that with constant probability, θ -𝑑 𝑛 1 1 ⊤ achieves weak recovery when
𝜀 2 𝑑 ⩾ 0.99𝑘 2 . □
With Lemma C.3, the proof of lower bound for learning the edge connection probability matrix of stochastic block model follows as a corollary.
Proof of Theorem 2.4. By Lemma C.3, suppose an exp 𝑛 0.99 time algorithm achieves error rate less than 0.99 √ 𝑘𝑑 in estimating the edge connection probability matrix, then in Algorithm C.2, θ -𝑑 𝑛 achieves weak recovery when 𝜀 2 𝑑 = 0.99𝑘 2 . We let 𝑓 (𝑌) = 1 𝑔(𝑌)⩾0.001𝜀 2 𝑑 2 /𝑘 . We show that with constant probability under 𝑃, we have 𝑓 (𝑌) = 1. We essentially follow the proof of Lemma B.3 with 𝛿 taken as a constant, except that we take a different strategy for bounding ⟨𝑊 2 -W2 , M⟩. By Lemma E.2, we have, with probability at least 1 -𝑜(1), the following spectral radius bounds on the symmetric random matrices
∥𝑊 2 -W2 ∥ op ⩽ 𝑂 𝑑 log(𝑛) • 𝑑 𝑛 .
Therefore, by Trace inequality, we have
|⟨𝑊 2 -W2 , M⟩| = |⟨𝑊 2 -W2 , M + 1 𝑘𝛿 1 1 ⊤ ⟩ -⟨𝑊 2 -W2 , 1 𝑘𝛿 1 1 ⊤ ⟩| ⩽ |⟨𝑊 2 -W2 , M + 1 𝑘𝛿 1 1 ⊤ ⟩| + |⟨𝑊 2 -W2 , 1 𝑘𝛿 1 1 ⊤ ⟩| ⩽ ∥𝑊 2 -W2 ∥ op Tr( M + 1 𝑘𝛿 1 1 ⊤ ) + ∥𝑊 2 -W2 ∥ op Tr(1
𝑘𝛿 1 1 ⊤ ) ⩽ 𝑂 𝑑 log(𝑛) • 𝑑 𝑛 (1 + 1 𝑘 ) 𝑛 𝛿 = 𝑂 (𝑑 + 𝑑 𝑘 ) 𝑛 log(𝑛) 𝛿 .
With the same reasoning, by Lemma B.4, with probability at least 1 -exp(-𝑛 0.001 ) under distribution 𝑄, we have 𝑓 (𝑌) = 0. Therefore, we have 𝑅 𝑃,𝑄 ( 𝑓 ) ⩾ exp(𝑛 0.001 ). Since 𝑓 (𝐴) can be evaluated in 𝑂 exp 𝑛 0.99 time, assuming conjecture 1.3 we have
𝑅 𝑃,𝑄 ( 𝑓 ) 𝔼 𝑓 (𝐴) Var 𝑄 ( 𝑓 (𝐴)) ≲ max deg( 𝑓 )⩽𝑛 0.99 𝔼 𝑓 (𝐴) Var 𝑄 ( 𝑓 (𝐴))
.
On the other hand, by low-degree lower bound stated in Theorem A.1, we have
max deg( 𝑓 )⩽𝑛 0.99 𝔼 𝑓 (𝐴) Var 𝑄 ( 𝑓 (𝐴))
⩽ exp(𝑘 2 ) .
Since we have exp(𝑛 0.001 ) ≫ exp(𝑘 2 ) when 𝑘 ⩽ 𝑛 𝑜(1) , this leads to a contradiction. □
this section cite: []

Section: C.2 Computational lower bound for learning graphon
In this part, we give formal proof of Theorem 2.7.
Theorem C.4 (Restatement of Theorem 2.7). Let 𝑘, 𝑑 ∈ ℕ + be such that 𝑘 ⩽ 𝑂(1), 𝑑 ⩽ 𝑜(𝑛).
Assume that Conjecture 1.3 holds with distribution 𝑃 given by SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘) and distribution 𝑄 given by Erdős-Rényi graph model 𝔾(𝑛, 𝑑 𝑛 ). Then no exp 𝑛 0.99 time algorithm can output a
poly(𝑛)-block graphon function Ŵ : [0, 1] × [0, 1] → [0, 1] such that GW( Ŵ , 𝑊 • ) ⩽ 𝑑 3𝑛 𝑘
𝑑 with 1 -𝑜(1) probability under distribution 𝑃 and distribution 𝑄(where 𝑊 • is the underlying graphon of the corresponding distribution).
Proof. Let 𝑊 0 be the graphon function underlying the distribution 𝔾(𝑛, 𝑑 𝑛 ) and 𝑊 1 be the graphon function underlying the distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), we have GW(
𝑊 0 , 𝑊 1 ) ⩾ 𝑑 𝑛 0.99𝑘 𝑑 when 𝜀 2 𝑑 ⩾ 0.99𝑘 2 .
Now suppose there is a polynomial time algorithm, which given random graph 𝐺 sampled from an arbitrary symmetric 𝑘-stochastic block model, outputs an 𝑛-block graphon function
Ŵ : [0, 1] × [0, 1] → [0, 1] achieving error 𝑑 3𝑛 𝑘 𝑑
with probability 1 -𝑜(1). Then one can construct the testing statistics by taking
𝑓 (𝑌) = 1, if GW( Ŵ , 𝑊 0 ) ⩽ 𝑑 3𝑛 𝑘 𝑑 0, otherwise
We have 𝑓 (𝑌) = 1 with probability 1 -𝑜(1) under the distribution of symmetric stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘). By triangle inequality, we have 𝑓 (𝑌) = 0 with probability 1 -𝑜(1) under the distribution 𝔾(𝑛, 𝑑 𝑛 ). Therefore we have 𝑅 𝑃,𝑄 ( 𝑓 ) ⩾ 𝜔(1). Now since the function Ŵ can be represented as a symmetric matrix with poly(𝑛) number of rows and columns, and moreove since 𝑊 0 is a constant function,
GW( Ŵ , 𝑊 0 ) = ∫ 1 0 ∫ 1 0 ( Ŵ(𝑥, 𝑦) -𝑊 0 (𝑥, 𝑦)) 2 𝑑𝑥𝑑𝑦 .
Therefore, the function 𝑓 (•) can be evaluated in polynomial time. This contradicts the low-degree lower bound (Theorem A.1) assuming Conjecture 1.3.
this section cite: []

Section: □

this section cite: []

Section: D Low-degree recovery lower bound for learning dense stochastic block model
In this part, we give unconditional lower bound against low-degree polynomial estimators for the edge connection probability matrix in stochastic block model, via implementing reduction from hypothesis testing to weak recovery using low-degree polynomials. For simplicity, we focus on the dense graph.
Theorem D.1 (Low-degree lower bound for learning). Let 𝑛 ∈ ℕ + and ℓ ⩽ 𝑛 0.001 . Let 𝑑 = Θ(𝑛).
Let ℱ 𝑛,ℓ be the set of degree-ℓ polynomials mapping from 𝑛 × 𝑛 symmetric matrices to 𝑛 × 𝑛 symmetric matrices. Suppose 𝜃 • ∈ [0, 1] 𝑛×𝑛 , 𝑌 ∈ {0, 1} are edge connection probability matrix and adjacency matrix sampled from symmetric stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘). Then for 𝑘 ⩽ 𝑛 0.001 , we have min
𝑓 ∈ℱ 𝑛,ℓ max 𝜀∈[0,1] 𝔼 (𝑌,𝜃 • )∼SSBM(𝑛, 𝑑 𝑛 ,𝜀,𝑘) ∥ 𝑓 (𝑌) -𝜃 • ∥ 2 F ⩾ Ω(𝑘 • 𝑛) .
this section cite: []

Section: D.1 Construction of the low-degree polynomial
For simplicity, we define the community matrix of symmetric stochastic block model.
Definition D.2 (Community matrix for stochastic block model). Under symmetric stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), we define the community matrix 𝑋 • ∈ {±1} as following: 𝑋 • (𝑖, 𝑗) = 1 if vertex 𝑖, 𝑗 have the same community label and 𝑋 • (𝑖, 𝑗) = 0 otherwise.
Given the polynomial function 𝑓 : ℝ 𝑛×𝑛 → ℝ 𝑛×𝑛 . We consider a graph with 2𝑛 nodes and randomly partition the nodes into two equal-sized sets 𝑆 1 and 𝑆 2 . Let 𝑋 = 𝑛 𝜀𝑑 𝑓 (𝑌 1 ) -𝑑 𝑛 where 𝑌 1 is the subgraph induced by vertices in 𝑆 1 . We construct the polynomial function 𝑔 : ℝ 𝑛×𝑛 → ℝ as following: 𝑔(𝑌) = 𝑌 12 -𝑑 𝑛 𝑋 𝑌 12 -𝑑 𝑛 , 𝑌 2 -𝑑 𝑛 , (D.1) where 𝑌 12 ∈ ℝ 𝑛×𝑛 is the adjacency matrix of the bipartite graph between vertices in 𝑆 1 and 𝑆 2 , and 𝑌 2 is the adjacency matrix of the induced subgraph supported on 𝑆 2 . We show the lower bound of this polynomial under the symmetric stochastic block model, and the upper bound of this polynomial under the Erdős-Rényi graph model. Lemma D.3. Let 𝜃 • , 𝑌 be the edge connection probability matrix and adjacency matrix sampled from the planted distribution SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘). Let 𝑋 • 1 be the community matrix of the subgraph induced by vertices in 𝑆 1 . Suppose in Eq. (D.1), 𝔼∥𝑋 -𝑋 • ∥ 2 F ⩽ 𝑜(𝑛 2 ), then we have 𝔼 𝑔(𝑌) ⩾ 𝜀𝑑 𝑛 3 𝑛 4 . Lemma D.4. When the graph is sampled from the null distribution 𝔾(𝑛, 𝑑 𝑛 ), we have 𝔼 𝑔(𝑌) = 0 and Var(𝑔(𝑌)) ⩽ 𝑑 3/2 • 𝑛 1-Ω(1) . Combining Lemma D.3 and Lemma D.4, Theorem D.1 follows as a corollary: Proof of Theorem D.1. Suppose there is a degree-𝑛 0.001 polynomial 𝑓 : ℝ 𝑛×𝑛 → ℝ 𝑛×𝑛 which gives error rate 𝑜(𝑛 • 𝑘). Let 𝑋 = 𝑛 𝜀𝑑 𝑓 (𝑌) -𝑑 𝑛 . Then we have
∥𝑋 -𝑋 • ∥ F = 𝑛 2 𝜀 2 𝑑 2 ∥ 𝑓 (𝑌) -𝜃 • ∥ 2 F ⩽ 𝑜 𝑛 2 𝜀 2 𝑑 2 𝑘𝑛 ⩽ 𝑜 𝑘 𝜀 2 𝑑 • 𝑛 𝑑 • 𝑛 2 .
When 𝜀 2 𝑑 ⩾ 0.001𝑘 2 and 𝑑 = Θ(𝑛), we have 𝔼∥𝑋 -𝑋
• ∥ 2 F ⩽ 𝑜(𝑛 2 ). combining Lemma D.3 and Lemma D.4, we have 𝔼 𝑔(𝑌) Var(𝑔(𝑌)) ⩾ 𝑛 0.001 . Since 𝑔(𝑌) is a degree-ℓ polynomial with ℓ ⩽ 𝑛 0.01 , by [Hop18], we have 𝔼 𝑔(𝑌) Var(𝑔(𝑌)) ⩽ exp(𝑘 2 ) . When exp(𝑘 2 ) ⩽ 𝑛 0.001 , this leads to a contradiction. As result, we conclude that no degree-𝑛 0.001 polynomial can achieve error rate 𝑜(𝑛 𝑘). □ D.2 Proof of Lemma D.3 In this section, we analyze the property of the polynomial in Eq. (D.1) under the 𝑘-symmetric stochastic block model, and give a proof for Lemma D.3. Proof of Lemma D.3. Let 𝑋 • 12 ∈ {±1} 𝑛×𝑛 be the community matrix for the bipartite graph between 𝑆 1 and 𝑆 2 , i.e for 𝑖 ∈ 𝑆 1 and 𝑗 ∈ 𝑆 2 , we have 𝑋 • 12 (𝑖, 𝑗) = 1 if 𝑖, 𝑗 belongs to the same community and 𝑋 • 12 (𝑖, 𝑗) = -1 if 𝑖, 𝑗 belongs to different communities. Moreover, we let 𝑋 • 1 be the community matrix for the induced subgraph on 𝑆 1 and let 𝑋 • 2 be the community matrix for the induced subgraph on 𝑆 2 . Then we have 𝑌 12 = 𝜀𝑑 𝑛 𝑋 • 12 + 𝑊 12 , 𝑌 1 = 𝜀𝑑 𝑛 𝑋 • 1 + 𝑊 1 and 𝑌 2 = 𝜀𝑑 𝑛 𝑋 • 2 + 𝑊 2 , where • 𝑊 12 , 𝑊 1 , 𝑊 2 are independent, • (𝑊 12 , 𝑊 1 , 𝑊 2 ) is independent with (𝑋 • 12 , 𝑋 • 1 , 𝑋 • 2 ), • every entry in 𝑊 12 , 𝑊 1 , 𝑊 2 has zero mean. Then we have 𝔼 𝑔(𝑌) = 𝔼 (𝑌 12 -𝑑 𝑛 )𝑋(𝑌 12 -𝑑 𝑛 ), 𝑌 2 -𝑑 𝑛 = 𝜀𝑑 𝑛 3 𝔼 𝑋 • 12 𝑋𝑋 • 12 , 𝑋 • 2 + 𝔼⟨𝑊 12 𝑋𝑊 12 , 𝑊 2 ⟩ + 2𝜀𝑑 𝑛 𝔼 𝑊 12 𝑋𝑋 • 12 , 𝑊 2 = 𝜀𝑑 𝑛 3 𝔼 𝑋 • 12 𝑋𝑋 • 12 , 𝑋 • 2 . Since 𝔼 𝑋 • 12 𝑋 • 1 𝑋 • 12 , 𝑋 • 2 ⩾ Ω(𝑛 4 ) and 𝔼 𝑌 • 12 (𝑋 1 -𝑋 • 1 )𝑋 • 12 , 𝑋 • 2 ⩽ 𝔼∥𝑋 1 -𝑋 • 1 ∥ 2 F • 𝔼∥𝑋 • 12 𝑋 • 2 𝑋 • 12 ∥ 2 F ⩽ 𝑜(𝑛 4 ) . Therefore, we have 𝔼 𝑋 • 12 𝑋𝑋 • 12 , 𝑋 • 2 ⩾ Ω(𝑛 4 ). and the claim follows. □ D.3 Proof of Lemma D.4 In this section, we analyze the property of the polynomial defined in Eq. (D.1), under the Erdős-Rényi graph distribution, and give a proof for Lemma D.4. Proof of Lemma D.4. Under the Erdős-Rényi graph distribution, the entries in 𝑌 2 -𝑑 𝑛 are i.i.d zero mean random variables, independent with 𝑌 12 and 𝑌 2 (i.e the rest of the graph). As result, we have 𝔼 𝑔(𝑌) = 0. It remains to bound the variance of the polynomial under the Erdős-Rényi graph distribution, which is to say, we bound 𝔼 𝑔(𝑌) 2 = 𝔼 𝑌 12 -𝑑 𝑛 𝑋 𝑌 12 -𝑑 𝑛 , 𝑌 2 -𝑑 𝑛 2 . Let 𝑊 12 = 𝑌 12 -𝑑 𝑛 and 𝑊 2 = 𝑌 2 -𝑑 𝑛 . The main observation is that 𝑋 , 𝑊 12 , 𝑊 2 are all independent. As result, let 𝑍 = 𝑊 12 𝑋𝑊 12 , we have
𝔼⟨𝑊 12 𝑋𝑊 12 , 𝑊 2 ⟩ 2 = 𝑖𝑗 (𝔼 𝑊 2 (𝑖, 𝑗)𝑍(𝑖, 𝑗)) 2 = 𝑖𝑗 𝔼 𝑊 2 2 (𝑖, 𝑗) 𝔼 𝑍(𝑖, 𝑗) 2 = 𝑑 𝑛 𝔼∥𝑍∥ 2 F .
As ∥𝑋 ∥ F ⩽ 𝑂(𝑛) without loss of generality, and ∥𝑊 12 ∥ ⩽ √ 𝑑 log(𝑛) with overwhelming high probability, we have
𝔼∥𝑍∥ 2 F ⩽ 𝑂 𝑛 2 𝑑 2 log 4 (𝑛) .
Therefore we have
𝔼⟨𝑊 12 𝑋𝑊 12 , 𝑊 2 ⟩ 2 ⩽ 𝑂 𝑛𝑑 3 log 4 (𝑛) .
By taking the square root, we conclude the proof.
this section cite: []

Section: □

this section cite: []

Section: E Probability theory facts
In this section, we provide probability tools that we will need in the paper.
this section cite: []

Section: E.1 Concentration of spectral radii of random matrices
The following concentration inequality for the spectral norm of the centered adjacency matrix of stochastic block model will be useful for our proofs.
Theorem E.1 (Spectral norm bound for random matrices, theorem 2.7 in [BGBK20]). Let 𝐻 ∈ ℝ 𝑛×𝑛 be a symmetric matrix whose upper triangular entries are independent zero mean random variables. Moreover, suppose that there exist 𝑞 > 0 and 𝜅 ⩾ 1 such that
max 𝑖 𝑗 𝔼|𝐻 𝑖𝑗 | 2 ⩽ 1 , max 𝑖,𝑗 𝔼|𝐻 𝑖𝑗 | 2 ⩽ 𝜅/𝑛 , max 𝑖,𝑗 |𝐻 𝑖𝑗 | ⩽ 1/𝑞 .
Then we have
𝔼∥𝐻 ∥ ⩽ 2 + 𝐶 log(𝑛) 𝑞 .
Moreover, we have
ℙ[|∥𝐻 ∥ -𝔼∥𝐻 ∥| ⩾ 𝑡] ⩽ 2 exp(-𝑐𝑞 2 𝑡 2 ) .
As corollary, for stochastic block model, we have the following concentration inequality:
ℙ[∥𝐻 ∥ ⩾ 𝔼∥𝐻 ∥ + 𝑡] ⩽ 2 exp(-𝑐 ′ 𝑡 2 ) .
where 𝑐 ′ is a universal constant. Taking 𝑡 ⩾ 1000 log(𝑛), we have the claim. □
this section cite: ['b10']

Section: E.2 Decoupling edge partition
In this section, we give a lemma that describes the approximate independence between edge sets of the subsampling process.
Lemma E.3. Let 𝑋 ∼ Ber(𝑝), and let 𝑋 1 be obtained from 𝑋 by subsampling with probability 1-𝜂, i.e 𝑋 1 = 𝑋𝜉, where 𝜉 ∼ Ber(1-𝜂) is independent of 𝑋. Let 𝑋 2 = 𝑋 -𝑋 1 , and X2 = 𝑋 2 -𝔼[𝑋 2 |𝑋 1 ]+𝜂𝑝. Then we have 𝔼[ X2 |𝑋 1 ] = 𝔼[𝑋 2 ] = 𝜂𝑝 and 𝔼[( X2 -𝑋 2 ) 2 |𝑋 1 ] ⩽ 𝑂(𝑝 3 ). Moreover, we have X2 -𝑋 2 ⩽ 𝜂𝑝. Proof. We first note that 𝔼[ X2 |𝑋 1 ] = 𝜂𝑝 = 𝔼[𝑋 2 ] . Next we note that 𝔼[𝑋 2 |𝑋 1 ] = 𝜂𝑝 1 -(1 -𝜂)𝑝 (1 -𝑋 1 ) , Therefore, we have 𝔼[( X2 -𝑋 2 ) 2 |𝑋 1 ] = 𝔼 𝜂𝑝 -𝜂𝑝(1 -𝑋 1 ) (1 -(1 -𝜂)𝑝) 2 ⩽ 𝑂(𝜂 2 𝑝 3 )
this section cite: []

Section: □
Corollary E.4. Let 𝑌 be the adjacency matrix of a random graph with each edge (𝑖, 𝑗) sampled with probability 𝑝(𝑖, 𝑗). Suppose that 𝑝(𝑖, 𝑗) ⩽ 𝑝 for each 𝑖, 𝑗 ∈ [𝑛]. Let 𝑌 1 be the adjacency matrix of the graph obtained by subsampling each edge in 𝑌 with probability 1 -𝜂. Let 𝑌 2 = 𝑌 -𝑌 1 . Then there is a matrix Ỹ2 such that • for every 𝑡 ⩾ log(𝑛), ∥ Ỹ2 -𝑌 2 ∥ 2 F ⩽ 𝑡𝜂𝑝 3 𝑛 2 with probability at least 1 -exp(-𝑡),
• for every 𝑖, 𝑗 ∈ [𝑛], 𝔼 Ỹ2 (𝑖, 𝑗) = 𝔼 𝑌 2 (𝑖, 𝑗)
• moreover Ỹ2 and 𝑌 1 are independent,
• and finally the entries in Ỹ2 are independent.
Proof. We construct the matrix Ỹ2 in the following way. Furthermore Ỹ2 (𝑖, 𝑗) and 𝑌 1 are independent.
Finally since the upper triangular entries in 𝑌 are independent, we have the upper triangular entries in Ỹ2 are independent. By Hoeffding bound, with probability at least 1 -exp(-𝑡), we have ∥ Ỹ2 -𝑌 2 ∥ 2 F ⩽ 𝑡𝜂𝑝 3 𝑛 2 log(𝑛). □
this section cite: []

Section: F Useful algorithmic results
In this section, we provide two algorithmic results from previous work that will be useful in our paper.
this section cite: []

Section: F.1 Correlation preserving projection
Given a vector 𝑃 that has constant correlation with an unknown vector 𝑌, [HS17] shows that one can project the vector 𝑃 into a convex set containing 𝑌, and preserve the constant correlation with 𝑌.
Theorem F.1 (Correlation preserving projection, theorem 2.3 in [HS17]). Let 𝛿 ∈ ℝ + Let 𝒞 be a convex set and 𝑌 ∈ 𝒞. Let 𝑃 be a vector with ⟨𝑃, 𝑌⟩ ⩾ 𝛿 • ∥𝑃∥ • ∥𝑌∥. Then, if we let 𝑄 be the vector that minimizes ∥𝑄 ∥ subject to 𝑄 ∈ 𝒞 and ⟨𝑃, 𝑄⟩ ⩾ 𝛿 • ∥𝑃∥ • ∥𝑌∥, we have ⟨𝑄, 𝑌⟩ ⩾ 𝛿/2 • ∥𝑄 ∥ • ∥𝑌∥ .
(F.1) Furthermore, 𝑄 satisfies ∥𝑄 ∥ ⩾ 𝛿∥𝑌∥.
We include their proof here for completeness.
Proof. By construction, 𝑄 is the Euclidean project of 0 into the set {𝑄 ∈ 𝒞|⟨𝑃, 𝑄⟩ ⩾ 𝛿∥𝑃∥ ∥𝑌∥}.
this section cite: ['b23', 'b23']

Section: By
Pythagorean inequality, the Euclidean projection into a set decreases distances to points into the set. Therefore, ∥𝑌 -𝑄 ∥ 2 ⩽ ∥𝑌 -0∥ 2 , which implies that ⟨𝑌, 𝑄⟩ ⩾ ∥𝑄 ∥ 2 /2. Moreover, ⟨𝑃, 𝑄⟩ ⩾ 𝛿∥𝑃 ∥ ∥𝑌∥, which implies ∥𝑄 ∥ ⩾ 𝛿∥𝑌∥ by Cauchy-Schwartz. Thus, we can conclude that ⟨𝑌, 𝑄⟩ ⩾ 𝛿/2 • ∥𝑌∥ • ∥𝑄 ∥. □ F.2 Learning edge connection probability matrix via SVD Theorem F.2. When 𝑑 ⩾ log(𝑛), there is a polynomial time algorithm which given the adjacency matrix of a graph sampled from symmetric 𝑘-stochastic block model SSBM(𝑛, 𝑑 𝑛 , 𝜀, 𝑘), returns an estimator θ ∈ [0, 1] 𝑛×𝑛 such that ∥𝜃 • -𝜃∥ 2 F ⩽ 𝑘𝑑 with high probability. Proof. We take θ as the best rank-𝑘 approximation for the adjacency matrix. Then since ∥𝐴 -𝜃 • ∥ op ⩽ √ 𝑘𝑑 with high probability, we have ∥ θ -𝐴∥ op ⩽ √ 𝑘𝑑 with high probability. By triangle inequality, we have ∥ θ -𝜃 • ∥ op ⩽ 2 √ 𝑑. As result, we have ∥ θ -𝜃 • ∥ F ⩽ 2 √ 𝑘𝑑 . □ G Low-degree lower bound beyond constant number of blocks By extending the result of [BBK + 21b], we can get a more general bound (with respect to 𝑘) on low degree likelihood ratio of k-SBM. The proof of the extended result follows trivially from the proof of Theorem 2.20 of [BBK + 21b]. Therefore, we only provide a proof sketch by pointing out the simple modifications that we need from the original proof. Theorem G.1 (Restatement of Theorem A.1). Let 𝑑 = 𝑜(𝑛), 𝑘 = 𝑛 𝑜(1) and 𝜀 ∈ [0, 1]. Let 𝜇 : {0, 1} 𝑛×𝑛 → ℝ be the relative density of SBM (𝑛, 𝑑, 𝜀, 𝑘) with respect to 𝐺 𝑛, 𝑑 𝑛 . Let 𝜇 ⩽ℓ be the projection of 𝜇 to the degree-ℓ polynomials with respect to the norm induced by 𝐺 𝑛, 𝑑 𝑛 For any constant 𝛿 > 0, 𝜇 ⩽ℓ is        ⩾ 𝑛 Ω(1) , if 𝜀 2 𝑑 > (1 + 𝛿)𝑘 2 , ℓ ⩾ 𝑂(log 𝑛) ⩽ 𝑂 𝛿 exp(𝑘 2 ) , if 𝜀 2 𝑑 < (1 -𝛿)𝑘 2 , ℓ < 𝑛 0.99 Proof. In this proof, we stick to the notations of [BBK + 21b]. The only modification we need is that the size of the 𝛿-net of the unit sphere in ℝ 𝑘 , denoted by 𝐶(𝛿, 𝑘), is equal to exp(𝑂 𝛿 (𝑘)). The size of the 𝛿-net 𝐶(𝛿, 𝑘) is crucial in Proposition 6.4 and Proposition 6.5 of [BBK + 21b] and is treated as constant in the proof of Theorem 2.16 and Theorem 2.20 of [BBK + 21b]. By plugging 𝐶(𝛿, 𝑘) = 𝑂 𝛿 (exp(𝑘)) into the upper bound of the small deviation term 𝐿 1 in the proof of Theorem 2.16 of [BBK + 21b], it follows that we have 𝐿 ⩽𝐷 = 𝑂 𝛿 (exp(𝑘)) for the likelihood ratio 𝐿 and 𝐷 ⩽ 𝑜(𝑛/log(𝑛)). Then, the bound on low-degree likelihood ratio of k-SBM follows from the same reduction as in proof of Theorem 2.20 of [BBK + 21b], and we get 𝜀 2 𝑑/(1 -𝑑/𝑛) ⩽ 1. As 𝑑 = 𝑜(𝑛), we get the claimed bound on low-degree likelihood ratio.
this section cite: []

Section: □

this section cite: []

Section: H Conclusion and future directions
Based on low-degree heuristics, our paper gives rigorous evidence for a computational phase transition of recovery at the Kesten-Stigum threshold. We view our work as a first step in studying this phenomenon, leaving open many interesting questions:
• Below the Kesten-Stigum threshold, suppose we are given an initalization achieving recovery rate 𝑛 -0.49 , could we boost the accuracy in polynomial time to get a weak recovery algorithm?
• Can we show a computational-statistical gap for learning the graphon function when the number of blocks satisfies 𝑘 ⩽ √ 𝑛 (as in [LG24])?
this section cite: ['b29']

Section: References
Ref_id:b0 Title: Community detection and stochastic block models: Recent developments Year: (2018)
Ref_id:b1 Title: Exact recovery in the stochastic block model Year: (2015)
Ref_id:b2 Title: Community detection in general stochastic block models: Fundamental limits and efficient algorithms for recovery Year: (2015)
Ref_id:b3 Title: Reducibility and statistical-computational gaps from secret leakage Year: (2020)
Ref_id:b4 Title: Reducibility and computational lower bounds for problems with planted sparse structure Year: (2019)
Ref_id:b5 Title: Statistical query algorithms and low-degree tests are almost equivalent Year: (2020)
Ref_id:b6 Title: Spectral planting and the hardness of refuting cuts, colorability, and communities in random graphs, Conference on Learning Theory Year: (2021)
Ref_id:b7 Title: Spectral planting and the hardness of refuting cuts, colorability, and communities in random graphs, Proceedings of Thirty Fourth Conference on Learning Theory Year: (2021-08-19)
Ref_id:b8 Title: Private graphon estimation for sparse graphs Year: (2015)
Ref_id:b9 Title: Revealing network structure, confidentially: Improved rates for node-private graphon estimation Year: (2018)
Ref_id:b10 Title: Spectral radii of sparse random matrices Year: ()
Ref_id:b11 Title: A nearly tight sum-of-squares lower bound for the planted clique problem Year: (2019)
Ref_id:b12 Title: Noise-tolerant learning, the parity problem, and the statistical query model Year: (2003)
Ref_id:b13 Title: Optimal hypothesis testing for stochastic block models with growing degrees Year: (2017)
Ref_id:b14 Title: Information-theoretic thresholds for community detection in sparse networks Year: (2016)
Ref_id:b15 Title: Private graphon estimation via sum-of-squares Year: (2020)
Ref_id:b16 Title: Graph partitioning via adaptive spectral techniques, Combinatorics Year: (2010)
Ref_id:b17 Title: Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications Year: (2011)
Ref_id:b18 Title: A general characterization of the statistical query complexity, Conference on learning theory Year: (2017)
Ref_id:b19 Title: Hardness of random optimization problems for boolean circuits, low-degree polynomials, and langevin dynamics Year: (2024)
Ref_id:b20 Title: Continuous lwe is as hard as lwe & applications to learning gaussian mixtures Year: (2017)
Ref_id:b21 Title: Stochastic blockmodels: First steps Year: (1983)
Ref_id:b22 Title: Statistical inference and the sum of squares method Year: (2018)
Ref_id:b23 Title: Efficient bayesian estimation from few samples: Community detection and related problems Year: (2017)
Ref_id:b24 Title: Planted clique conjectures are equivalent Year: (2024)
Ref_id:b25 Title: Sum-of-squares lower bounds for sparse independent set Year: (2013)
Ref_id:b26 Title: Sum of squares lower bounds for refuting any csp Year: (2017)
Ref_id:b27 Title: Nicolas Verzelen MODAL'X, Crest, and Mistea, Oracle inequalities for network models and sparse graphon estimation, arXiv: Statistics Theory Year: (2015)
Ref_id:b28 Title: Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio Year: (2019)
Ref_id:b29 Title: Computational lower bounds for graphon estimation via low-degree polynomials Year: (2006)
Ref_id:b30 Title: Algorithmic contiguity from low-degree conjecture and applications in correlated random graphs Year: (2025)
Ref_id:b31 Title: Community detection thresholds and the weak ramanujan property Year: (2014)
Ref_id:b32 Title: Stochastic block models and reconstruction Year: (2012)
Ref_id:b33 Title: Lifting sum-of-squares lower bounds: degree-2 to degree-4 Year: (2020)
Ref_id:b34 Title: Semidefinite Programs on Sparse Random Graphs and their Application to Community Detection Year: (2015)
Ref_id:b35 Title: Precise error rates for computationally efficient testing Year: (2023)
Ref_id:b36 Title: Computational barriers to estimation from low-degree polynomials Year: (2022)
Ref_id:b37 Title: Sharp phase transitions in estimation with low-degree polynomials Year: (2025)
Ref_id:b38 Title: Improved hardness results for learning intersections of halfspaces Year: (2024)
Ref_id:b39 Title: Rates of convergence of spectral methods for graphon estimation Year: (2006)
