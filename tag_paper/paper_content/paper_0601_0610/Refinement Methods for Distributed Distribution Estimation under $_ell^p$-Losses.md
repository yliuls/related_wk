Title: Refinement Methods for Distributed Distribution Estimation under ℓ p -Losses
Abstract: Consider the communication-constrained estimation of discrete distributions under ℓ p losses, where each distributed terminal holds multiple independent samples and uses limited number of bits to describe the samples. We obtain the minimax optimal rates of the problem for most parameter regimes. As a result, an elbow effect of the optimal rates at p = 2 is clearly identified. In order to achieve the optimal rates for different parameter regimes, we introduce refinement methods and develop additional customized techniques in the estimation protocols. The general idea of the refinement methods is to first generate rough estimate by partial information and then establish refined estimate in subsequent steps guided by the rough estimate. Then customized techniques such as successive refinement, sample compression, thresholding and random hashing are leveraged to achieve the optimal rates in different parameter regimes. The optimality of the estimation protocols is shown by deriving compatible minimax lower bounds.

Section: Introduction
Motivated by applications in areas such as federated learning [1][2][3], distributed statistical estimation problems have recently received wide attention. In this setting, multiple distributed agents cooperate to train a model, while each of them can only access to a subset of training data. These agents can exchange messages but their communication budgets are constrained. The performance of the system is often limited by the communication constraints.
One fundamental learning task is to estimate the underlying discrete distribution of the data. Under communication constraints, the minimax optimal rates for the estimation error were studied in [4][5][6][7][8][9][10][11]. Another important constraint is the differential privacy, and the corresponding problem was similarly considered in [5,6,12,13]. In these works, only one sample was accessed by each distributed terminal and the most common ℓ 1 and ℓ 2 losses were used to measure the estimation error. However, this is an oversimplification of the practical case, where general ℓ p losses may be necessary and each terminal can access to n > 1 samples. revealed in the n = 1 case. Even though [14] presented an optimal protocol for n > 1 and the ℓ 1 loss, it still does not apply to ℓ p losses since its optimality depends heavily on several special properties of the ℓ 1 loss.
In this work, we consider the distributed estimation of discrete distributions under communication constraints. The range of the problem is expanded in two directions, letting each terminal hold n > 1 samples and imposing general ℓ p losses simultaneously. We design interactive protocols to achieve optimal rates in this technically more challenging setting. The difficulty lies in the communication budget allocation strategy, namely how to assign multiple terminals and their communication budgets to the tasks of estimating different distribution entries. The naive uniform allocation strategy that treats all the entries equally fails to achieve the optimal convergence rate under the general ℓ p loss for p > 1. To achieve the optimal rate, communication budgets should be invested based on the distribution to be estimated. As a result, existing protocols cannot handle the general problem under the ℓ p loss. Instead, we develop refinement methods in the estimation protocol, which first establishes rough estimate based on partial information obtained by a portion of budgets, and then uses it to allocate the remaining budgets for refining the estimate. The refined estimate can achieve the optimal error rate, since the remaining budgets are allocated most effectively.
We introduce additional auxiliary estimation techniques to customize the refinement methods for different parameter regimes. The induced estimation protocols shows upper bounds for the optimal rates. We also derive compatible lower bounds for most parameter regimes. Hence the optimality of the protocols is shown and the optimal rates are obtained in these regimes.
1. We exploit the classic divide-and-conquer technique and design a successive refinement estimation protocol equipped with an adaptive budget allocation strategy. The distribution is divided into blocks. The estimation task is achieved by first estimating the block distribution and then conditional distribution over each block. The block distribution has a lower dimension, and the divide-and-conquer procedure is not stopped until it is more efficient to estimate each entry directly. This induces a successive refinement protocol where the rough estimate for the block distribution is refined by further estimating the conditional distributions over blocks. More importantly, in the refinement step we introduce an adaptive budget allocation strategy. Specifically, terminals are assigned to estimating different conditional distributions based on the block distribution estimated by the former phase, which achieves faster convergence rate for p > 1 than the uniform allocation strategy by previous works [14]. Hence the successive refinement protocol achieves the optimal rates up to logarithmic factors for most parameter regimes with 1 ≤ p ≤ 2. Moreover, by using multiple successive refinement steps rather than only one step, our protocol for p = 1 achieves the optimal rates for a larger range of regimes than that in [14].
2. For p > 2, we develop auxiliary sample compression techniques, so that refinement methods can be adopted in the estimation protocol. Different from 1 ≤ p ≤ 2, the protocol in this regime obtains a rough estimate of the distribution itself (rather than an estimate of the block distribution) first by uniform allocation of budgets. It then refines the estimate by allocating the remaining budgets according to the rough estimate. In the refinement stage of the protocol, we further develop sample compression techniques, which compress the description for samples and reduce the communication budget, allowing more samples to be transmitted. The resulting protocols can achieve the optimal rates for relatively large n.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b4', 'b5', 'b11', 'b12', 'b13', 'b13', 'b13']

Section: 3.
In the very special regime where the total communication budget is extremely tight, we incorporate a thresholding technique into the estimation protocol to achieve the optimal rate.
The key observation is that under the extremely tight communication budget, if an entry of the distribution is too small then approximating it simply by 0 induces a lower variance than trying to estimate them. For p > 2, the thresholding technique are combined with the sample compression to yield the optimality protocol. To the best of our knowledge, the regime has not been discussed in any previous work.
4. For the special case n = 1, we design an optimal non-interactive protocol by exploiting random hash functions, rather than the sample splitting trick or the simulate and infer protocol used in previous works [7,8,18]. To show the optimality, we further establish a compatible lower bound that is strictly better than that in [16,17] for p > 2. This proves the optimal rates under general ℓ p losses, especially that for p > 2 left open by previous works [16,17].
The expression of the optimal rates under ℓ p losses reveals an elbow effect at p = 2, providing more insights into the distributed estimation problem. It is interesting to compare our results with the elbow effect discovered in the nonparamentric density estimation problem [19,20]. It is not a coincidence since in both problems there are constraints for the estimated object (namely the normalization constraint for the distribution estimation problem and the Sobolev regularity constraint for the nonparametric density estimation problem), and the loss functions can vary with a parameter. The similarity sheds light on how the optimal rates are affected by the relation between the imposed loss function and the constraints on the estimated object.
The remaining part of this work is organized as follows. First, the problem is formulated in Section 2. Then we present our main results for 1 ≤ p ≤ 2 and p > 2 in Sections 3 and 4 respectively.
In Section 5, the special case with n = 1 is discussed and the non-interactive protocol is presented.
Finally, the optimal rates are summarized in Section 6 and a few further remarks are given in Section 7.
Detailed estimation protocols and complete proofs of both upper and lower bounds can be found in the technical appendix.
this section cite: ['b6', 'b7', 'b17', 'b15', 'b16', 'b15', 'b16', 'b18', 'b19']

Section: Problem Formulation
Denote a discrete random variable by a capital letter and its finite alphabet by the corresponding calligraphic letter, e.g., W ∈ W. We use the superscript n to denote an n-sequence, e.g., W n = (W i ) n i=1 . For a finite set W of size k = |W|, let ∆ W be the set of all the probability measures over W, i.e. ∆ W ≜ {p(•) : p(w) ∈ [0, 1], ∀w ∈ W, w p(w) = 1}. Let ∆ ′ W be the set of subprobability measures, i.e. ∆ ′ W ≜ {p(•) : p(w) ∈ [0, 1], ∀w ∈ W, w p(w) ≤ 1}. Suppose that we want to estimate the finite-dimensional distribution p W ∈ ∆ W with dimension k, and the samples are generated at random. To be precise, let W ij ∼ p W (w), i = 1, 2, • • • , m, j = 1, 2, • • • , n be i.i.d. random variables distributed over W. The total sample size is mn.
Consider the distributed minimax parametric distribution estimation problem with communication constraints depicted in Fig. 1, which is a theoretical model of federated learning systems. There are m encoders and one decoder, and common randomness is shared among them. The i-th encoder observes the samples W n i = (W ij ) n j=1 and transmits an encoded message B i of length l to the decoder, i = 1, ..., m. Upon receiving messages B m = (B i ) m i=1 , the decoder needs to establish a reconstruction pW ∈ ∆ ′ W of p W . An (m, n, k, l)-protocol P is defined by a series of random encoding functions
Enc i : W n × {0, 1} (i-1)l → {0, 1} l , ∀i = 1, ..., m,
this section cite: []

Section: and a random decoding function
Dec : {0, 1} ml → ∆ ′ W . The i-th encoder is aware of the messages sent by the previous i -1 encoders (which can be achieved by interacting with other encoders and/or the decoder), and it generates a binary sequence B i = Enc i (W n , B 1:i-1 ) of length l. The reconstruction of the distribution is pP W = Dec(B 1 , B 2 , ..., B m ). For p ≥ 1, we use the ℓ p loss to measure the estimation error. We are interested in the minimal error of all the estimation protocols in the worst case, as the true distribution p W varies in the probability simplex ∆ W . To be specific, our goal is to characterize the order of the the following minimax convergence rate R(m, n, k, l, p) = inf
(m, n, k, l)-protocol P sup p W ∈∆ W E[∥ pP W -p W ∥ p p ].
Remark 1. The (m, n, k, l)-protocol P defined in this work is usually called the (sequentially) interactive protocol in the literature. The protocol is called non-interactive, if for each i = 1, ..., m, the i-th encoder is ignorant of all the messages B 1:i-1 sent by previous encoders and the encoding function Enc i (W n ) is a function of the samples only. In most cases we design interactive protocols since it is too hard to construct a non-interactive protocol. For some simple special cases, noninteractive protocol achieving the optimal rates can be constructed, which will be indicated.
We further define some necessary notations. For any positive functions a(m, n, k, l, p) and b(m, n, k, l, p), we say a ⪯ b if a ≤ c • b for some positive constant c > 0 independent of parameters (m, n, k, l). The notation ⪰ is defined similarly. Then we denote by a ≍ b if both a ⪯ b and a ⪰ b hold. Denote by a ∧ b the minimum of two real numbers a and b, and a ∨ b the maximum.
W n 1 Encoder 1 B 1 (B 1 , W n 2 ) Encoder 2 B 2 ... (B 1:m-1 , W n m ) Encoder m B m
Decoder pW 3 Optimal Rates for 1 ≤ p ≤ 2
First assume that 1 ≤ p ≤ 2. We present the upper bound in the following theorem. Theorem 1. Let 1 ≤ p ≤ 2, then we have R(m, n, k, l, p)
⪯                                k (mnl) p 2 ∨ k 1-p 2 (mn) p 2 , n ≥ k, m(l ∧ k) > 1000k log(mn) log n, k 1-p 2 log p 2 ( k n + 1) (ml) p 2 ∨ k 1-p 2 (mn) p 2 , k 2 l ≤ n < k, m(l ∧ n) > 2000n log(mn) log n, k (mn2 l ) p 2 , n < k 2 l , m(l ∧ n) > 4000n log(mn) log n, log p 2 k (ml) p-1 , log k < l < n, ml < k. (1a
) (1b
) (1c
) (1d
)
Proof. The case (1a) is by Proposition 1 in Appendix A, cases (1b) and (1c) are by Proposition 2 in Appendix B, and the case (1d) is by Proposition 5 in Appendix E. We sketch the proof here and details can be found in the appendix.
Successive refinement protocol with adaptive budget allocation For the first three cases (1a), (1b) and (1c), the estimation protocol can be sketched as the following inductive procedure.
At each step, choose some l 0 and construct a division W = ∪ t s=1 W s with |W s | ≤ 2 l0 -1, l 0 ≤ l and t = ⌈ k 2 l 0 -1 ⌉. First suppose that the distribution p B of blocks has been estimated to some accuracy by some pB . Then each encoder can use its l 0 -bit message to only describe its samples on a predetermined block W s . Based on these messages, the decoder then estimates the conditional distribution p s ∈ ∆ Ws on the s-th block, where p s (w) ≜ p(w|W s ). Based on the message, the decoder constructs ps as an estimate of p s . Combining pB and ps for each block s, an estimate of p W can be immediately obtained by letting p W (w) = pB (s)p s (w) for w ∈ W s . Note that p B ∈ ∆ [1:t] always has a lower dimension t than the dimension k for p W . Fewer encoders are needed for the smaller problem. Hence pW can be refined from these layered block distributions successively.
For the base case k < n where the length k of the distribution p W is sufficiently small, it is optimal to estimate p W (w) directly for each w ∈ W using the one-bit protocol in [14]. Although the error analysis is only shown for the ℓ 1 loss in [14], it can be adapted to prove the optimality of the above procedure for ℓ p losses with 1 ≤ p ≤ 2. See Appendix A for details.
Then consider the successive refinement subroutine for estimating all the p s , s = 1, ..., t given an estimate pB for p B . Through detailed error analysis (see Lemma 5 and its discussions), to achieve the optimality, the budget for estimating each p s should be proportional to pB (s). Since the decoder have obtained the rough estimate pB (s), it can allocate remaining budget by interactions with encoders. Such an allocation plan is in contrast to the estimation problem under the TV loss discussed in Section 3.1 and [14], where a uniform budget allocation among all the p s , s = 1, ..., t is optimal.
Given the successive refinement subroutine and the estimation protocol for the base case, it remains to consider the choice of l 0 as well as the budget allocation between these successive steps. They depend on the parameter regime. For the case (1b) where the length l is relatively large, each message can be divided to describe multiple samples. In order to directly exploit the protocol for the base case in estimating the block distribution p B , we choose t ∼ n and l 0 ∼ log k n . In contrast, for the case (1c) where l is relatively small, we let l 0 = l and then use multiple successive steps until the dimension of the distribution is reduced to n • 2 l . The complete estimation protocol is constructed in Appendices B.2.2 and B.2.3 respectively.
this section cite: ['b13', 'b13', 'b13']

Section: Refinement protocol with thresholding techniques
For the final case (1d), the refinement protocol is designed with the help of thresholding techniques. The idea is that under the extremely tight communication budget, roughly ∼ ml samples can be transmitted by the protocol. Then approximating those p W (w) ⪯ 1 ml simply by 0 is better than estimating them. In the refinement step, the remaining budget can be used for generating another independent estimate for those p W (w) ⪰ 1 ml , whose number ∼ ml is limited. Detailed protocol can be found in Appendix E.1.
Lower bounds under the ℓ p loss can be derived in the following lemma, which provides a baseline.
Lemma 1. For 1 ≤ p ≤ 2, we have R(m, n, k, l, p) ⪰                              k (mnl) p 2 ∨ k 1-p 2 (mn) p 2 , n ≥ k log k, m > k l 2 k 1-p 2 (ml log k) p 2 ∨ k 1-p 2 (mn) p 2 , k 2 l ≤ n < k log k, m > k l 2 , k (mn2 l ) p 2 , n < k 2 l , mn2 l > k 2 , 1 (ml) p-1 ∨ k 1-p 2 (mn) p 2 , ml < k 2 .
Proof. Lower bounds for the first three cases under the ℓ p loss can be derived from existing results in [14] under the ℓ 1 loss. For the last case, we use an algebraic technique to first note R(m, n, k, l, p) ≥ R(m, n, 2ml, l, p) and then bound the latter, which slightly strengthens the usual bound obtained by the data processing inequality. This induces compatible lower bound with the upper bound in (1d). The detailed proof can be found in Appendix G.
Combining Theorem 1 and lemma 1, the optimal rates for the following cases can be roughly characterized by R(m, n, k, l, p)
≍                              k (mnl) p 2 ∨ k 1-p 2 (mn) p 2 , n ≥ k, ml ⪰ k, k 1-p 2 (ml) p 2 ∨ k 1-p 2 (mn) p 2 , k 2 l ≤ n < k, ml ⪰ k, k (mn2 l ) p 2 , n < k 2 l , mn2 l ⪰ k 2 , 1 (ml) p-1 ∨ k 1-p 2 (mn) p 2 , ml ⪯ k.(2)
Remark 2. We add a few explanations concerning the boundaries in (2). The regularity condition m > ( k l ) 2 in the lower bound is induced mainly by technical reasons and the boundary ml > k is more essential. Similarly, the conditions m(l ∧ k) > 1000k log(mn) log n and m(l ∧ n) > 2000n log(mn) log n in the upper bound can be relaxed by finer analysis and the true boundaries seem to be around ml > k and ml > n, respectively. Under these observations, in the third case the conditions mn2 l ≥ k 2 and n < k 2 l imply that m > k and hence ml > k > n is fullfilled.
this section cite: ['b13']

Section: Special Cases: Optimal Rates for p = 1 and p = 2
In this subsection, we specialize our results and characterize the optimal rates under the most commonly used total variation (TV) and squared losses, i.e. ℓ 1 and ℓ 2 losses. For the TV loss, the successive refinement protocol can be made non-interactive. See Appendix C for details.
Theorem 2. The following upper bound can be achieved by a non-interactive protocol. R(m, n, k, l, 1)
⪯                    k 2 mnl ∨ k mn , n ≥ k, m(l ∧ k) > 1000k log m log n, k log( k n + 1) ml ∨ k mn , k 2 l ≤ n < k, m(l ∧ n) > 2000n log m log n, k 2 mn2 l , n < k 2 l , m(l ∧ n) > 4000n log m log n.
For the TV loss, we have the following characterization of the optimal rates. R(m, n, k, l, p = 1)
≍                        k 2 mnl ∨ k mn , n ≥ k, ml ⪰ k, k ml ∨ k mn , k 2 l ≤ n < k, ml ⪰ k, k 2 mn2 l ∧ 1, n < k 2 l , 1, ml ⪯ k.(3)
Remark 3. In [14], a non-iteractive protocol for the same problem in Section 2 under the TV loss is also constructed. However, corresponding to the third case in Theorem 2, in [14] a stronger restriction m > 100 k 2 l log m log n is imposed (cf. Theorem 1.1 in [14] and note that the notations m and n are interchanged therein). The restriction is induced by using the first bit of each encoder to estimate the block probability p B with the protocol for the first case. The conditional probability in each block B is then estimated. Combining it with the estimate for p B , an estimate for p W is obtained. In fact, it is a one-step reduction. We note that the step that estimates the conditional probability can be abstracted and summarized as a separate protocol, and it has an inductive nature. Instead of using it only once, we iteratively use the protocol, which is inspired by the classic divide-and-conquer technique. Thus our successive refinement protocol relaxes the restriction in [14] and achieve an upper bound for a wider parametric range.
The squared loss is the most important loss, in both theoretical analysis and algorithm research. By directly specializing Theorem 1, we have the following upper bounds under the squared loss.
Corollary 1. For the squared loss, we have R(m, n, k, l, p = 2)
⪯                        k mnl ∨ 1 mn , n ≥ k, m(l ∧ k) > 1000k log(mn) log n, log( k n + 1) ml ∨ 1 mn , k 2 l ≤ n < k, m(l ∧ n) > 2000n log(mn) log n, k mn2 l , n < k 2 l , m(l ∧ n) > 4000n log(mn) log n, log k ml , log k < l < n, ml < k.
Lemma 1 can be specialized to obtain the lower bounds as well. Then we have a more complete characterization of the order of R(m, n, k, l, p = 2). R(m, n, k, l, p = 2)
≍              k mnl ∨ 1 mn , n ≥ k, ml ⪰ k, 1 ml ∨ 1 mn , k 2 l ≤ n < k or n ≥ k, ml ⪯ k, k mn2 l , n < k 2 l , mn2 l ⪰ k 2 .(4)
4 Optimal Rates for p > 2
For p > 2, we first present the upper bound in the following. Theorem 3. Let p > 2, then we have R(m, n, k, l, p
) ⪯                                    k (mnl) p 2 ∨ 1 (mn) p 2 , n ≥ k, m(l ∧ k 2 p ) > 1000k log(mn) log n, log p 2 k (ml) p 2 n p 2 -1 ∨ 1 (mn) p 2 , k (2 l ) p 2 ≤ n < k, m(l ∧ n 2 p ) ≥ 1000n log(mn) log k, l > log k, k mn2 l p 2 , n < k (2 l ) p 2 , m(l ∧ n) > 4000n log(mn) log n, log p k ∨ log 4p (mn) (ml) p-1 ∨ 1 (mn) p 2 , log k < l < n, ml < n. (5a
) (5b
) (5c
) (5d
) Proof. The case (5a) is by Proposition 1 in Appendix A, the case (5b) is by Proposition 4 in Appendix D, the case (5c) is by Proposition 2 in Appendix B, and the case (5d) is by Proposition 5 in Appendix E. We present a sketch of the proof for these cases here.
Refinement protocol For the case (5a), the first step of the protocol for p > 2 is the same as that for 1 ≤ p ≤ 2. That is, a rough estimate pW is established by assigning the first half of all encoders uniformly to estimating each entry p W (w) using the one-bit protocol in [14]. But it is not enough, since for p > 2 the estimation error for the big entry p W (w) decays significantly slower than that for the small entry, which is different from the case 1 ≤ p ≤ 2. To overcome the difficulty, a refinement method is necessary, where a portion of roughly pW (w) remaining budget is allocated to estimate p W (w). The spirit of the allocation strategy is similar to that designed for the pointwise estimation problem [11] with n = 1. Details can be found in Appendix A.
this section cite: ['b13', 'b13', 'b13', 'b13', 'b13', 'b10']

Section: Refinement protocol with sample compression techniques
For the case (5b), sample compression techniques are further incorporated. The starting point is also the refinement method, but in this case the length of the distribution is too long, namely k > n. Hence the optimal estimation method for the encoder is not to summarize its samples and describe each p W (w), but to describe samples it observes directly. This makes how to do the refinement step obscure.
this section cite: []

Section: Sample compression techniques are designed to customize the refinement methods in this regime.
Note that the number of the elements w with p W (w) ⪰ 1 n (denote the set containing those elements w by W ′ ) is about n. Samples are first compressed by projecting them to W ′ , which saves the communication budget for describing them. Hence those p W (w) ⪰ 1 n are refined by invoking the protocol for the case (5a). See Appendix D for details.
The remaining two cases The bound in (5c) is a corollary of the successive refinement protocol in Appendix B. For the case (5d), the bound is achieved by a refinement protocol exploiting both sample compression and thresholding techniques, and details can be found in Appendix E.2. Similar to Section 3, we present the lower bound as a baseline in the following lemma. Lemma 2. For p > 2, we have R(m, n, k, l, p)
⪰                              k (mnl) p 2 ∨ 1 (mn) p 2 , n ≥ k log k, m > k l 2 1 (ml) p 2 n p 2 -1 log n ∨ 1 (mn) p 2 , k (2 l ) p 2 ≤ n < k log k, m > n/ log n l 2 , k (mn2 l ) p 2 , n < k (2 l ) p 2 , mn2 l > k 2 , 1 (ml) p-1 ∨ 1 (mn) p 2 , ml < k 2 .
Proof. Most of the lower bounds can be derived from that under the ℓ 1 loss in [14] using Hölder's Inequality. Two of the bounds, namely 1 (ml) p 2 n p 2 -1 log n in the second case and 1 (ml) p-1 in the last case require the additional algebraic technique in the proof of Lemma 1. The last exception, the centralized bound 1 (mn) p 2 without communication constraints is little-known but easy to show. Moreover, we think its proof uncovers the major differences of the estimation problem for p > 2 compared with that for p ≤ 2. Hence it is sketched as follows. Detailed proof for all the lower bounds can be found in Appendix G.
The centralized bound without communication constraints For p > 2, the key observation is that distributions most difficult to estimate have only a few large entries of constant order. It is in contrast to the case 1 ≤ p ≤ 2 where such distributions are close to uniform and each entry is roughly ∼ 1 k . In light of this, the bound can be proved by a simple way of reduction to a binary hypothesis testing. It is elaborated in the proof of Lemma 11 in Appendix G.
Remark 4. We summarize our technical contributions in the lower bounds in Lemmas 1 and 2 here. From a technical perspective, the overall proof of the lower bounds depend on four different ways of reduction to hypothesis testing problems. Most of lower bounds under the ℓ p loss are derived from that under the ℓ 1 loss in [14]. Typically, the proof in [14] uses the reduction to a hypothesis testing problem of roughly 2 k 2 hypotheses. However, the derived bounds are not tight, especially for the case p > 2. In this work, one of the major finding is that the optimal bounds are different for p ≤ 2 and p > 2. To show that, we introduce two major techniques, which rely on three ways of reduction to hypothesis testing. First, the centralized bound 1 (mn) p 2 for p > 2 is proved by the reduction to a binary hypothesis testing. Second, the algebraic technique is exploited for the communicate-constrained bounds
1 (ml) p 2 n p 2 -1 log n
and 1 (ml) p-1 . In its spirit, the technique used in these two cases is equivalent to two different ways of reduction hypothesis testing problems, with roughly 2 n and 2 ml hypotheses respectively. The latter three ways of reduction used in this work improve bounds derived from the first way in [14], so that the overall lower bound is tight. These four reductions together complete the proof of lower bounds.
Combining Theorem 3 and lemma 2, the optimal rates can be characterized in the following, except for the third case where our lower and upper bounds do not coincide. We conjecture that the lower bound
k (mn2 l ) p 2 is tight, which is partially verified for the case n = 1 in the next section. R(m, n, k, l, p) ≍                            k (mnl) p 2 ∨ 1 (mn) p 2 , n ≥ k, ml ⪰ n 1 (ml) p 2 n p 2 -1 ∨ 1 (mn) p 2 , k (2 l ) p 2 ≤ n < k, ml ⪰ n, k (mn2 l ) p 2 , n < k (2 l ) p 2 , mn2 l ⪰ k 2 , 1 (ml) p-1 ∨ 1 (mn) p 2 , ml ⪯ k, k < n or ml ⪯ n, k > n.(6)
5 Optimal Rates for n = 1, p ≥ 2 and the Non-interactive Estimation Protocol
For n = 1 and p ≥ 2, the lower bound can be derived by specializing Lemma 2, and the compatible upper bound is achieved by a non-interactive protocol, shown in the following thoerem.
Theorem 4. Let n = 1, p ≥ 2 and m(2
l ∧ k 2 p ) ≥ k 2 .
We can design a non-interactive protocol that achieves the optimal rate R(m, 1, k, l, p)
≍ k (m2 l ) p 2 ∨ 1 m p 2 .
Proof. The lower bound is implied by Lemma 2. The upper bound is by Proposition 6 in Appendix F, for which we present the proof sketch here.
Non-interactive protocol with random hashing For each encoder, a hash function h i : W → [1 : 2 l ] is randomly generated. Then the encoder can compress its sample W i to the message h i (W i ) using its l bits. Upon receiving all the messages, the decoder can directly obtain the estimate by constructing and rescaling the histogram. Further discussions can be found in the proof of Proposition 6 and Appendix F.1. Remark 5. Note that the centralized bound 1 m p 2 without the communication constraints is neglected by previous works [16,17] (see Theorem 6 in [16] and Corollary 3.2 in [17]). Hence the lower bounds in both works are clearly not tight (for p > 2). The work [17] further claimed that the lower bound
k (m2 l ) p 2 ∨ k 1-p 2 m p 2
is optimal (see Lemma 3.3 therein), but the sketch given there is not sufficient to describe a protocol that achieves the bound. In fact, given that the lower bound in [17] can be strictly improved, it is impossible to show its optimality. Moreover, constructing the protocol that achieves the optimal rates for p > 2 is not straightforward and needs additional ideas. We use random hashing technique to resolve the difficulty in this work, and there may be other solutions. Remark 6. We give some intuitive explanations about why our random hashing protocol achieves the optimal rate in Theorem 4, while existing methods like the simulate-and-infer protocol [7,8,18] fail to do so. As discussed in Section 1 and the proof sketch of Lemma 2, for p > 2 relatively larger entries p W (w) are typically more difficult to estimate, and communication budgets should be invested more into estimating them. In this sense, the problem resembles a sparse distribution estimation. The simulate-and-infer protocol uses too much communication budget to estimate the smaller entries, while fails to simulate enough samples for estimating the larger entries. In contrast, random hashing reduces estimation errors for the larger entries, despite increasing the error for the smaller entries. Therefore, it achieves an optimal communication budget allocation strategy, as well as the optimal rate.
this section cite: ['b13', 'b13', 'b13', 'b13', 'b15', 'b16', 'b15', 'b16', 'b16', 'b16', 'b6', 'b7', 'b17']

Section: Summary of the Optimal Rates
In Table 1, we summarize the characterizations of the optimal rate obtained in Equations ( 2) to ( 4) and (6) and Theorem 4, where fundamentally different regimes lead to different rates. The essential bounds originally proved in this work are highlighted in red, while those established in previous works [7,8,14,16,17] are shown in blue. All the other bounds are corollaries of them. The optimal rates (up to logarithmic factors) are obtained for most cases, except the case p > 2, n < k (2 l ) p 2 and mn2 l ≥ k 2 , where our lower and upper bounds do not coincide. Though a good news is that for its special case n = 1, the optimal rates can be obtained. We conjecture that the lower bound
k (mn2 l ) p 2
is tight, which is partially verified in the case n = 1. We find several interesting phenomena of the optimal rates. 1. There is an elbow effect in the parameter p between the regimes 1 ≤ p < 2 and p ≥ 2.
The difference is clearly reflected in the centralized bound without any communication
constraints, i.e. l = ∞. The bound is k 1-p 2 (mn) p 2 for 1 ≤ p < 2, while for p ≥ 2 it is 1 (mn) p 2
and independent of the dimension k of the distribution. The other sharp difference is that, for a medium n, i.e. k (2 l ) p 2 ∨1 ≤ n < k, the optimal rate is independent of k (up to logarithmic factors) for p ≥ 2, which is not the case for 1 ≤ p < 2.
2. Second, the minimum transmitted bits required for recovering the same rates in the centralized case without any communication constraints are interesting for p > 2. It is roughly k 2 p for k < n, ml ≥ k and n 2 p for k ≥ n, ml ≥ n, which is out of expectation. It shows a shrinkage compared to the required number of bits k and n for the case 1 ≤ p < 2. Similarly, for n = 1 and m2 l ≥ k 2 , the required number of bits is roughly 2  p log k instead of log k. 3. The last observation is that if the total communication budget is extremely tight (ml ≪ k),
then the optimal rate is dependent only on the total budget and independent of the parameters k and n. This parameter regime has not been studied in previous work to our best knowledge.
Table 1: Bounds of R(m, n, k, l, p) for Different Cases
Parameter Regimes p = 1 1 ≤ p ≤ 2 p = 2 p ≥ 2 l = ∞ R ≍ k mn R ≍ k 1-p 2 (mn) p 2 R ≍ 1 mn R ≍ 1 (mn) p 2 (Lemma 11) n ≥ k, R ≍ k √ mnl R ≍ k (mnl) p 2 R ≍ k mnl R ≍ k (mnl) p 2 l p 2 ∨1 ≤ k, ml ≥ k (Proposition 1) k (2 l ) p 2 ∨1 ≤ n < k, R ≍ k ml R ≍ k 1-p 2 (ml) p 2 R ≍ 1 ml R ≍ 1 (ml) p 2 n p 2 -1 l p 2 ∨1 ≤ n, ml ≥ k (p ≤ 2), ml ≥ n (p > 2)
(Propositions 2 and 4)
ml < k (p ≤ 2 R ≍ 1 R ≍ 1 (ml) p-1 R ≍ 1 ml R ≍ 1 (ml) p-1 or p > 2, k ≤ n), ml < n (p > 2, k > n), l > log k (Proposition 5) (Proposition 5) n < k (2 l ) p 2 ∨1 , R ≍ k √ mn2 l R ≍ k (mn2 l ) p 2 R ≍ k mn2 l R ⪯ k mn2 l p 2 (Proposition 2) mn2 l ≥ k 2 R ⪰ k (mn2 l ) p 2 n = 1, R ≍ k √ m2 l R ≍ k (m2 l ) p 2 R ≍ k m2 l R ≍ k (m2 l ) p 2 (2 l ) p 2 ∨1 < k, m2 l ≥ k 2 (Proposition 6)
this section cite: ['b6', 'b7', 'b13', 'b15', 'b16']

Section: Discussions and Future Works
In this work, we focused on the minimax optimal rates of distribution estimation over the whole probability simplex, without imposing any additional assumptions on the structure of the distribution to be estimated. In contrast, many previous works studied the structured distribution estimation problems [4,[9][10][11]17], such as the point-wise distribution estimation problem [11,17] and the sparse distribution estimation problems [9,10]. These problems are also of both theoretical and practical importance. However, existing works limited their scope to n = 1, leaving problems with n > 1 and ℓ p losses unexplored. We hope our methods can help with determining optimal rates for these problems.
Moreover, the methods in this work are not restricted to the discrete distribution estimation problem. The analysis of statistical learning problems in various other settings under ℓ p losses can also benefit from our methods. The methods deal with the difficulty induced by the normalization constraint of the distribution in the distribution estimation setting, which also shows a potential direction for solving problems with similar implicit constraints. A more challenging problem is whether we can construct non-interactive protocols, instead of interactive protocols in this work, to achieve the minimax optimal rates with n > 1 samples per terminal and under ℓ p losses. Determining the privacy-constrained optimal rates for n > 1 and ℓ p losses is also an interesting direction for future work.
Finally, our protocol for estimating a discrete distribution (especially for the squared loss) can be used as a subroutine of the protocol achieving the optimal rates in the nonparametric density estimation and regression problems. See the works [20,21] for details.
this section cite: ['b3', 'b8', 'b9', 'b10', 'b16', 'b10', 'b16', 'b8', 'b9', 'b19', 'b20']

Section: NeurIPS Paper Checklist
1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The paper's scope is clearly defined, and our contributions are accurately summarized in the abstract and introduction.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Range of applicability of our results is accurately described in the introduction and theorems. Potential further directions not covered in this work are provided in the final discussion section.
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
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All the assumptions and proofs are presented in the main paper or the supplemental material. In the case where the complete proof appears in the supplemental material, a short sketch is provided in the main body to provide some intuition. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [NA] Justification: The paper does not contain experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [NA] Justification: The paper does not include experiments. Guidelines:
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
Justification: The research conducted in the paper conform with the NeurIPS Code of Ethics in every respect.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA]
Justification: The paper focuses on foundational theoretical results not tied to particular application. Although some further applications may have societal impacts, the paper is too far from those impacts and should not be responsible for them.
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
Answer: [NA]
Justification: The paper focuses on foundational theoretical results not tied to particular application. It does not release data or practical models.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [NA] Justification: The paper does not use existing assets.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
this section cite: []

Section: New assets
Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA]
Justification: The paper does not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: Refinement of the estimate
The second step is to let the next m 2 encoders and the decoder jointly generate a refined estimate p2 W . Let m(w) = ⌊ ml( p1 W (w)+ 1 k ) 4 ⌋ ∧ m 2 . Each encoder can concurrently run l one-bit protocols in Lemma 3 using its l bits, for estimating some p W (w). At the same time, a proper allocation plan can ensure that for each w ∈ W, there are m(w) encoders 1 running the protocol for estimating p W (w). The decoder then constructs the refined estimate p2 W following the protocol.
this section cite: []

Section: Remark 8 (Necessity of the Refinement Methods). It is easy to analyze the error of the rough estimate p1
W . By Lemma 3 and the assumption ml > 1000k log(mn) log n, for any w ∈ W we have
E |p W (w) -p1 W (w)| p = O kp W (w) mnl p 2 + k mnl p-1 p W (w) + 1 mnl p 2 . (8)
However, simply taking the summation can only get the total error bound O(( k mnl )
p 2 ), which is not tight for p > 2. To obtain the tight bound, our protocol uses the rough estimate p1 W for directing the budget allocation in the second step.
Then the refined estimate in the second step can achieve the desired upper bound, i.e. E[∥ p2 W -p W ∥ p p ] = O k (mnl) p 2 , which completes the proof of Proposition 1. See Appendix A.2 for details.
this section cite: []

Section: A.2 Proof of Proposition 1: Error Analysis for the Protocol in Appendix A.1
We first show the following preliminary error bound concerning the rough estimate.
Lemma 4. If p W (w) ≥ 1 k for some w ∈ W, then P p W (w) p1 W (w) ≥ 2 ≤ O 1 (np W (w)) p 2 .
Proof. By (8) and p W (w) ≥ 1 k , we have
E |p W (w) -p1 W (w)| p = O kp W (w) mnl p 2 . (9
)
By the Markov inequality, we can obtain that
P p W (w) p1 W (w) ≥ 2 = P p1 W (w) p W (w) ≤ 1 2 ≤P p1 W (w) -p W (w) ≥ 1 2 p W (w) ≤ 2 p E[ p1 W (w) -p W (w) p ] p W (w) p .
Then by ( 9) and the assumption that ml > 1000k log(mn) log n, we complete the proof.
Now we return to the proof of Proposition 1. Note that it suffices to show that for each w ∈ W,
E |p W (w) -p2 W (w)| p = O 1 (mnl) p 2 + k mnl p-1 p W (w) + p W (w) p 2 (mn
) p 2 , (10
)
then taking the summation and using mnl ≥ k 2 can complete the proof. 1 One may worry that the estimate p1 W may not be normalized. But it does not affect the subsequent steps of using p1 W for directing the budget allocation. This can be seen by the following analysis. By the proof of Theorem A.2 in [14] and n ≥ k, for a constant C > 1,
P[∥ p1 W ∥1 ≥ C] ≤ w P[|p 1 W (w) -pW (w)| ≥ (C -1)( 1 n ∨ p W (w) n )] ≤ k log n • e - m ′
240 log n , which is sufficiently small if ml ≫ k log n log(mn). In the case that p1
W is used as a ratio for budget allocation, we can simply divide it by the constant C and then the error analysis is still true. Hence, for simplicity we assume that p1 W is normalized and do not point out this minor obstacle in similar cases where p1
W is generated by the protocol in Lemma 3.
By Lemma 3, we have
E |p W (w) -p2 W (w)| p =O E p W (w) mnl(p 1 W (w) + 1 k ) p 2 + k mnl p-1 p W (w) + 1 mnl p 2 + p W (w) mn p 2 .
It suffices to bound the first term. Define the event F w = p W (w) p1 W (w) ≥ 2 . Then by Lemma 4 and n ≥ k, we have
E p W (w) mnl(p 1 W (w) + 1 k ) p 2 =E 1 Fw p W (w) mnl(p 1 W (w) + 1 k ) p 2 + E 1 F ∁ w p W (w) mnl(p 1 W (w) + 1 k ) p 2 ≤P [F w ] • kp W (w) mnl p 2 + O 1 mnl p 2 =1 {p W (w)< 1 k } •O 1 mnl p 2 +1 {p W (w)≥ 1 k } • O 1 np W (w) • kp W (w) mnl p 2 +O 1 mnl p 2 =O 1 mnl p 2
, which completes the proof.
this section cite: ['b13']

Section: B The Protocol for Cases (1b), (1c) and (5c) and Its Analysis
In this section, we design a successive refinement protocol with adaptive budget allocation that achieves the optimal rates for cases (1b), (1c) and (5c). Similar to the discussion in Remark 7, it suffices to show the following proposition for p ≥ 2.
Proposition 2. Let p ≥ 2. Then for the problem in Section 2, there exists an interactive protocol SSR(m, n, k, l, p) such that for any p W ∈ ∆ W , the protocol outputs an estimate pW satisfying,
1. if k ≤ n, m(l ∧ k) > 1000k log(mn) log n, then E[∥ pW -p W ∥ p p ] = O k mnl p 2 ∨ 1 (mn) p 2 ; 2. if n < k ≤ (2 l -1) • n, l ≥ 2 and m(l ∧ n) > 2000n log(mn) log n, then E[∥ pW -p W ∥ p p ] = O log( k n +1) ml p 2 ∨ 1 (mn
) p 2 ; 3. if k > (2 l -1) • n, l ≥ 4 and m(l ∧ n) > 4000n log(mn) log n, then E[∥ pW -p W ∥ p p ] = O k 2 l mn p 2 . Remark 9.
Although the bound in Proposition 2 is not always tight for p > 2, it is indeed tight (up to logarithmic factors) for p = 2 and can imply tight bound for 1 ≤ p < 2. The advantage of using the successive refinement protocol for 1 ≤ p < 2 is that the protocol can apply for a lager parameter regime. In comparison, the protocol in Appendix D can be used for 1 ≤ p < 2 and k > n but it requires that l > log k. Hence it fails to handle the case 2 for log( k n + 1) < l ≤ log k and the case 3 in Proposition 2.
We design the successive refinement protocol SSR(m, n, k, l, p) in Proposition 2 inductively, which turns out to be a successive refinement procedure. The protocol for each case in Proposition 2 relies on that for the preceding case. The goal is to estimate a distribution p W ∈ ∆ W . If the communication budget l for each encoder is too tight, then it is difficult to describe all the entries of p W . Instead, we can perform a a divide-and-conquer technique.
At each step, choose some l 0 and construct a division W = ∪ t s=1 W s with |W s | ≤ 2 l0 -1, l 0 ≤ l and t = ⌈ k 2 l 0 -1 ⌉. Then each encoder is assigned several W s and ordered to describe the conditional distribution p s ∈ ∆ Ws for the assigned W s , where p s (w) ≜ p(w|W s ). Based on the message, the decoder constructs ps as an estimate of p s . Let the block distribution be p B , where p B (s) = w∈Ws p(w). As long as an estimate pB of the distribution p B can be obtained, we can obtain an estimate p W (w) = pB (s)p s (w) for w ∈ W s .
The above procedure can be repeated for the estimation of p B . Note that p B ∈ ∆ [1:t] always has a lower dimension t than the dimension k for p W , the inductive procedure will finally terminate. Hence the estimate pB can be obtained, as well as pW .
The error of each one-step procedure is bounded by the following lemma, proved in Appendix B.3.
Lemma 5. For p ≥ 2, we have
E[∥ pW -p W ∥ p p ] ≤ 2 p-1 E[∥ pB -p B ∥ p p ] + t s=1 E[p B (s) p 2 pB (s) p 2 ∥ ps -p p ∥ p p ] .(11)
Remark 10. For the TV distance (p = 1), it is easy to obtain that (cf. Lemma 3.1 in [14])
E[∥ pW -p W ∥ TV ] ≤ E[∥ pB -p B ∥ TV ] + t s=1 p B (s)E[∥ ps -p s ∥ TV ].(12)
Now consider the subroutine for estimating all the p s , s = 1, ..., t given an estimate pB for p B . By (11), it is intuitive that the budgets for estimating each p s should be based on the multiplicative weight pB (s) p 2 p B (s) p 2 of the estimation error ∥ ps -p s ∥ p p . It turns out that the number of encoders for estimating p s can be proportional to pB (s). Since the quantity pB (s) can be obtained by the decoder, the allocation of encoders can be based on it by interaction between the decoder and encoders. Such an allocation plan is in contrast to the estimation problem under the TV loss discussed in Appendix C. The difference is characterized by the error bound (12), where the weight is simply p B (s) and a uniform budget allocation plan among all the p s , s = 1, ..., t is optimal.
The detailed subroutine is presented in the following subsection.
this section cite: ['b13', 'b10', 'b11']

Section: B.1 Successive Refinement Subroutines
Suppose that there are m ′ encoders and each of them observes i.i.d. samples W n . Fix l 0 ≤ l and let n 0 = ⌊ l l0 ⌋ ∧ n. Then we design the successive refinement subroutine SSRSub(m ′ , n, k, l, l 0 , p) as follows. It receives an estimate pB of the block distribution p B of dimension t, and outputs an estimate pW of the original distribution p W .
Allocating frames to blocks Divide the l-bit message for each encoder into multiple l 0 -bit frames. Then each encoder holds at least n 0 such frames and all encoders hold m ′ n 0 frames in total. Each l 0 -bit frame is sufficient to transmit a sample, given that the sample is from a fixed block s of size no more than
2 l -1. Simply let r(s) = pB (s). (13
)
Then r is a block distribution. And we allocate all m ′ n 0 frames held by m ′ encoders to encoding samples in different W s , such that (i) for each block s, N s = ⌊m ′ n 0 r(s)⌋ frames are allocated;
(ii) for each encoder, there are at most ⌈n 0 r(s)⌉ frames allocated to transmitting samples in W s .
Encoding For each block s, each encoder divides all its n samples into ⌈n 0 r(s)⌉ parts, and each part has ⌊ n ⌈n0r(s)⌉ ⌋ samples (ignoring the remaining n -⌈n 0 r(s)⌉ • ⌊ n ⌈n0r(s)⌉ ⌋). Each frame that is held by the encoder and allocated for transmitting samples in block s is then mapped to one of these parts injectively. If in that part, there are samples falling into the block s, then the encoder uses the corresponding frame to encode the first such sample. If not, the frame is encoded as 0.
Decoding and estimating For each block s, the decoder extracts frames in messages which are allocated to the block. For b = 1, ..., N s , let W s b = ∅ if the b-th such frame is 0 and let W s b be the sample encoded by the frame if it is not 0. The decoder computes
N ′ s = Ns b=1 1 W s b ̸ =∅ . Then it computes ps (w) = Ns b=1 1 W s b =w N ′ s (14) if N ′
s ̸ = 0, and it computes ps (w) = 1 |Ws| otherwise. Finally, for each s = 1, ..., t and each w ∈ W s , it computes pW (w) = pB (s)p s (w).
The complete successive refinement subroutine SSRSub(m ′ , n, k, l, l 0 , p) is summarized in Algorithm 1. The estimation error induced by the subroutine is described in the following lemma, proved in Appendix B.4.
Lemma 6. For p ≥ 2, we have
t s=1 E[p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p ] = O   1 ∨ t n p 2 • l0 l ∨ 1 n p 2 m ′ p 2   .(15)
this section cite: []

Section: B.2 Construction of the Complete Protocol SSR
By inductively using the subroutine, the complete protocol SSR(m, n, k, l, p) for the three cases in Proposition 2 can be constructed as follows. Then the error bounds are derived accordingly from Lemmas 5 and 6 in Appendix B.5 and B.6.
this section cite: []

Section: B.2.1 The Protocol for Case 1
Invoke the first step of the protocol IR(m, n, k, l ∧ k, p) in Appendix A and then output the rough estimate p1 W . By the analysis in Remark 8, we have
E[∥ pW -p W ∥ p p ] = O k mnl p 2 ∨ 1 (mn) p 2 .
this section cite: []

Section: B.2.2 The Protocol for Case 2
Let l 0 = ⌈log( k n + 1)⌉ ≤ l and divide the set W into t = ⌈ k 2 l 0 -1 ⌉ ∈ [ n 2 , n] blocks. Let the first m 2 encoders and the decoder estimate the reduced distribution of dimension t ≤ n. By the assumptions m(l ∧ n) > 2000n log(mn) log n, they can invoke the protocol SSR( m 2 , n, t, l, p) in Appendix B.2.1.
Then let the second m 2 encoders and the decoder invoke the subroutine SSRSub( m 2 , n, k, l, l 0 , p) and compute the estimate of the original distribution p W .
this section cite: []

Section: B.2.3 The Protocol for Case 3
It suffices to design the protocol for m ≥ 8k n2 l , since the upper bound is vacuous otherwise. Let l 0 = l and then compute the integer a as follows. Let k 1 = k, then iteratively compute k u+1 = ⌈ ku 2 l -1 ⌉ for u = 1, ..., a. Let a be the minimal number satisfying k a+1 ≤ n • (2 l -1), then k a+1 > n.
this section cite: []

Section: Let the first m
2 encoders invoke the protocol SSR( m 2 , n, k, l, p) defined in Appendix B.2.2 to estimate the last reduced block distribution of dimension k a+1 .
Divide the second m 2 encoders into a parts, such that the u-th part has m u = ⌊ m 2 u+1 ⌋ encoders. By the choice of a, we have a ≤ 2 log(
k n(2 l -1) ) l . Then we have 2 a ≤ 2 k n(2 l -1) 2 l ≤ m 2 for l ≥ 4, Hence m u ≥ m 2 a+1 ≥ 1. For u = 1, .
.., a, the decoder iteratively invokes SSRSub(m u , n, k u , l, l 0 , p) with encoders in the u-th part successively. Then compute the estimate of the original distribution p W .
Algorithm 1 Successive Refinement Subroutine SSRSub(m ′ , n, k, l, l 0 , p) Input: Parameters (m ′ , n, k, l, l 0 , p), an estimate pB of the block distribution p B (at all encoder and decoder sides). Output: An estimate pW of the original distribution p W .
this section cite: []

Section: Allocating frames to blocks:
1: n 0 ← ⌊ l l0 ⌋ ∧ n. 2: Divide each l-bit message into n 0 frames of length l 0 . 3: for s = 1 : t do 4: r(s) ← pB (s). Divide all n samples into ⌈n 0 r(s)⌉ parts, each with ⌊ n ⌈n0r(s)⌉ ⌋ samples.
10:
Find frames allocated to W s .
11:
for b = 1 : ⌈n 0 r(s)⌉ do 12:
if all such frames have been encoded then 13:
Break.
this section cite: []

Section: 14:
else if ∃W i ∈ W s for some W i in the b-th part then 15:
The b-th frame ← the first such W i .
16:
else 17:
The b-th frame ← 0.
this section cite: []

Section: 18:
end if 19:
end for 20: end for Decoding and estimating at the decoder side: 21: for s = 1 : t do end for 30:
N ′ s ← Ns b=1 1 W s b ̸ =∅ . 31: for w ∈ W s do 32: if N ′ s ̸ = 0 then 33: ps (w) ← Ns b=1 1 W s b =w N ′ s 34: else 35: ps (w) ← 1 |Ws| .
36:
end if 37:
pW (w) ← pB (s)p s (w).
this section cite: []

Section: 38:
end for 39: end for 40: return pW .
this section cite: []

Section: B.3 Proof of Lemma 5
Note that
(p B (s)p s (w) -pB (s)p s (w)) 2 ≤(p B (s)p s (w) -pB (s)p s (w)) 2 + (p B (s)p s (w) -pB (s)p s (w)) 2 =(p s (w) 2 + ps (w) 2 )(p B (s) -pB (s)) 2 + 2p B (s)p B (s)(p s (w) -ps (w)) 2 .
Then by the Hölder's inequality, we have
(p B (s)p s (w) -pB (s)p s (w)) p ≤2 p 2 -1 p s (w) 2 + ps (w) 2 p 2 (p B (s) -pB (s)) p + 2 p 2 p B (s) p 2 pB (s) p 2 (p s (w) -ps (w)) p ≤2 p-1 1 2 (p s (w) + ps (w)) (p B (s) -pB (s)) p + p B (s) p 2 pB (s) p 2 (p s (w) -ps (w)) p .
where the last inequality is since p ≥ 2 and p s (w), ps (w) ∈ [0, 1]. Take the summation, and then we have
∥ pW -p W ∥ p p ≤2 p-1 t s=1 (p B (s) -pB (s)) p + p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p .
Then (11) is obtained by taking the expectation. We complete the proof.
this section cite: []

Section: B.4 Proof of Lemma 6
If
m ′ n 0 r(s) = m ′ n 0 pB (s) ≤ 4, since ∥ ps -p s ∥ p p ≤ 2, then p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p ≤ 2p B (s) p 2 pB (s) p 2 ≤ 2 p+1 p B (s) m ′ n 0 p 2 . Otherwise, we have m ′ n 0 r(s) = m ′ n 0 pB (s) > 4, hence N s = Θ (m ′ n 0 r(s)) = Θ (m ′ n 0 pB (s)).
Given pB , then W s u for u = 1, ..., N s are i.i.d. random variables with
q s ≜P[ W s u ̸ = ∅| pB ] = 1 -(1 -p B (s)) ⌊ n ⌈n 0 r(s)⌉ ⌋ =Θ p B (s) n ⌈n 0 r(s)⌉ ∧ 1 = Θ p B (s) n ⌈n 0 pB (s)⌉ ∧ 1 .(16)
In this case, we can establish the bound shown in the following lemma.
Lemma 7. E[p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p | pB ] ≤ CE pB (s) m ′ n ∨ 1 m ′ nn0 ∨ p B (s) m ′ n0 p 2 pB for some C > 0.
Proof. By the Chernoff bound, we have
P N ′ s ≥ N s q s 2 pB ≤ exp - N s q s 8 .(17)
And conditional on the event { W s u ̸ = ∅}, the distribution of W s u is p s . Hence for each w ∈ W s , it is folklore that (cf. Theorem 4 in [22] or Rosenthal's inequality [23]),
E[|p s (w) -p s (w)| p |N ′ s , pB ] = O p s (w) N ′ s p 2 + p s (w) N ′p-1 s .
Take the summation, since p ≥ 2 and p s (w) ∈ [0, 1] we have
E ∥ ps -p s ∥ p p N ′ s ≥ N s q s 2 , pB = O 1 N p 2 s q p 2 s .
Since ∥ psp s ∥ 2 ≤ 2, we have
E[∥ ps -p s ∥ 2 | pB ] ≤ 2 exp - N s q s 8 + O 1 N p 2 s q p 2 s = O 1 N p 2 s q p 2 s .
Since n 0 ≤ n, we have ⌈n 0 pB (s)⌉ ≤ n and n ⌈n0 pB (s)⌉ ≥ 1. Hence there exists some C > 0, such that
E[p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p | pB ] ≤CE p B (s)p B (s) m ′ n 0 pB (s)q s p 2 pB =CE      p B (s) m ′ n 0 p B (s)⌊ n ⌈n0 pB (s)⌉ ⌋ ∧ 1   p 2 pB    =CE      1 m ′ n 0 n ⌈n0 pB (s)⌉ ∨ p B (s) m ′ n 0   p 2 pB    =CE n 0 pB (s) ∨ 1 m ′ nn 0 ∨ p B (s) m ′ n 0 p 2 pB ,
completing the proof.
In both cases, we can take the expectation and obtain that
E[p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p ] ≤ C ′ E pB (s) m ′ n ∨ 1 m ′ nn 0 ∨ p B (s) m ′ n 0 p 2 , for some C ′ > 0.
Finally, take the sum over s and note that p ≥ 2, then
t s=1 E[p B (s) p 2 pB (s) p 2 ∥ ps -p s ∥ p p ] ≤C ′ t s=1 E pB (s) m ′ n ∨ 1 m ′ nn 0 ∨ p B (s) m ′ n 0 p 2 =O 1 m ′ n 0 p 2 ∨ t (m ′ nn 0 ) p 2 =O   1 ∨ t n p 2 • l0 l ∨ 1 n p 2 m ′ p 2   ,
which completes the proof.
this section cite: ['b21', 'b22']

Section: B.5 Proof of Proposition 2: Analysis of The Protocol for Case 2
By the case 1, the estimation error for the reduced block distribution is bounded by
C 3 • t mnl p 2 ∨ 1 (mn) p 2
for some C 3 > 0.
By Lemma 6, the estimation error for the conditional distribution induced by the invoking of the subroutine SSRSub( m 2 , n, k, l, l 0 , p) is bounded by
C 4 • l0 l ∨ 1 n m 2 p 2 = C 4 2 l0 l ∨ 1 n m p 2 =C 4 2l 0 ml p 2 ∨ 2 p 2 (mn) p 2 ≤C 4   4 log( k n + 1) ml p 2 ∨ 2 p 2 (mn) p 2   ,
for some C 4 > 0.
Then by Lemma 5, the total error is bounded by
2 p-1    C 4 •   4 log( k n + 1) ml p 2 ∨ 2 p 2 (mn) p 2   + C 3 • t mnl p 2 ∨ 1 (mn) p 2    =O   log k n + 1 ml p 2 ∨ 1 (mn) p 2   .
this section cite: []

Section: B.6 Proof of Proposition 2: Analysis of The Protocol for Case 3
By the analysis in Appendix B.2.2, the estimation error for the reduced block distribution induced by the invocation of SSR( m 2 , n, k a+1 , l, p) is bounded by
C 5 •      log ka+1 n + 1 ml   p 2 ∨ 1 (mn) p 2    ≤ C 5 m p 2 ,
for some C 5 > 0.
We have k u+1 ≥ k a+1 > n and l0 l = 1 > 1 n . Then by Lemma 6, the estimation error for the conditional distribution induced by the u-th invocation of the subroutine SSRSub(m u , n, k u , l, l 0 , p) is bounded by
C 6 • k u+1 m u n p 2 ≤ C 6 k 2 u(l-1) m 2 u+2 n p 2 = C 6 2 u+2 k (2 l-1 ) u mn p 2 ,(18)
for some C 6 > 0.
Then by Lemma 5 and l ≥ 4, the total error is bounded by
2 a(p-1) • C 5 m p 2 + C 6 a u=1 2 u(p-1) • 2 u+2 k (2 l-1 ) u mn p 2 ≤2 k n(2 l -1) 2(p-1) l • C 5 m p 2 + 2 3p C 6 k 2 l mn p 2 =O k 2 l mn p 2 .
this section cite: []

Section: C The Non-interactive Protocol for the TV Loss and Its Analysis
Consider the estimation problem under the TV loss, i.e. p = 1. In this section, we show that a uniform budget allocation plan is sufficient in this case, thanks to the error bound (12). The advantage of the uniform allocation plan is obvious, since there is no need for the decoder to send any message to the encoders. Hence a non-interactive protocol is immediate induced, only by changing (13) to r(s
) = 1 t (19
)
in the successive refinement subroutine SSRSub(m ′ , n, k, l, l 0 , 1) in Appendix B.1.
For simplicity, we slightly abuse the notations SSRSub(m ′ , n, k, l, l 0 , 1) and SSR(m, n, k, l, 1) to still denote the resulting non-interactive protocols. The non-interactive successive refinement subroutine SSRSub(m ′ , n, k, l, l 0 , 1) is presented in Algorithm 2 for completeness, where differences with Algorithm 1 are underlined.
To show Theorem 2, it remains to show the error bound in the following proposition. Proposition 3. For any p W ∈ ∆ W , the non-interactive protocol SSR(m, n, k, l, 1) outputs an estimate pW satisfying, Algorithm 2 Non-Interactive Successive Refinement Subroutine SSRSub(m ′ , n, k, l, l 0 , 1) Input: Parameters (m ′ , n, k, l, l 0 ), an estimate pB of the block distribution p B (only at the decoder side). Output: An estimate pW of the original distribution p W . Allocating frames to blocks:
1: n 0 ← ⌊ l l0 ⌋ ∧ n. 2: Divide each l-bit message into n 0 frames of length l 0 . 3: for s = 1 : t do 4: r(s) ← 1/t.
this section cite: ['b11']

Section: 5:
N s ← ⌊m ′ n 0 r(s)⌋.
this section cite: []

Section: 6:
Allocate N s frames to W s , s.t. at most ⌈n 0 r(s)⌉ frames are at the same encoder side. 7: end for Proceed as that in Algorithm 1.
1. if k ≤ n, m(l ∧ k) > 1000k log m log n, then E[∥ pW -p W ∥ p p ] = O k 2 mnl ∨ k mn ; 2. if n < k ≤ (2 l -1) • n, l ≥ 2 and m(l ∧ n) > 2000n log m log n, then E[∥ pW -p W ∥ p p ] = O k log( k n +1) ml ∨ k mn ; 3. if k > (2 l -1) • n, l ≥ 4 and m(l ∧ n) > 4000n log m log n, then E[∥ pW -p W ∥ p p ] = O k 2 2 l mn .
this section cite: []

Section: C.1 Error Analysis of the Subroutine for p = 1
First, the estimation error induced by the subroutine SSRSub(m ′ , n, k, l, l 0 , 1) is described in the following lemma. Lemma 8. We have
t s=1 E[p B (s)∥ ps -p s ∥ TV ] = O t m ′ 1 ∨ t n • l 0 l ∨ 1 n . (20
) Proof. If m ′ n 0 r(s) = m ′ n0 t ≤ 4, since ∥ ps -p s ∥ TV ≤ 2, then p B (s)∥ ps -p s ∥ TV ≤ 2p B (s) ≤ 4 p B (s) 2 t m ′ n 0 ≤ 4 p B (s) 2 k m ′ n 0 .
Otherwise, we have
m ′ n 0 r(s) = m ′ n0 t > 4, hence N s = Θ (m ′ n 0 r(s)) = Θ m ′ n0 t .
Then W s u for u = 1, ..., N s are i.i.d. random variables with
q s ≜ P[ W s u ̸ = ∅| pB ] = Θ p B (s) n ⌈n 0 r(s)⌉ ∧ 1 = Θ p B (s) n ⌈n 0 /t⌉ ∧ 1 . (21
)
Then we can establish the following lemma.
Lemma 9. E[p B (s)∥ ps -p s ∥ TV ] ≤ CE p B (s)k m ′ nt ∨ p B (s)k m ′ nn0 ∨ p B (s) 2 k m ′ n0
for some C > 0.
Proof. By the Chernoff bound, we have
P N ′ s ≥ N s q s 2 pB ≤ exp - N s q s 8 . (22
)
And conditional on the event { W s u ̸ = ∅}, the distribution of W s u is p s . By the Cauchy-Schwarz inequality and p s (w) ∈ [0, 1],
E ∥ ps -p s ∥ TV N ′ s ≥ N s q s 2 ≤ |W s | • E ∥ ps -p s ∥ 2 2 N ′ s ≥ N s q s 2 = O |W s | N s q s .
Since ∥ psp s ∥ 2 ≤ 2, we have
E[∥ ps -p s ∥ 2 | pB ] ≤ 2 exp - N s q s 8 + O |W s | N s q s = O |W s | N s q s .
Since n 0 ≤ n, we have ⌈ n0 t ⌉ ≤ n and n ⌈n0/t⌉ ≥ 1. Hence there exists some C > 0, such that
E[p B (s)∥ ps -p s ∥ TV ] ≤ CE p B (s) 2 k t m ′ n0 t q s =CE    p B (s) 2 k m ′ n 0 p B (s)⌊ n ⌈n0/t⌉ ⌋ ∧ 1    =CE    p B (s)k m ′ n 0 n ⌈n0/t⌉ ∨ p B (s) 2 k m ′ n 0    =CE   p B (s)k m ′ nt ∨ p B (s)k m ′ nn 0 ∨ p B (s) 2 k m ′ n 0   ,
completing the proof.
In both cases, we can take the expectation and obtain that
E[p B (s)∥ ps -p s ∥ TV ] ≤ C ′ E   p B (s)k m ′ nt ∨ p B (s)k m ′ nn 0 ∨ p B (s) 2 k m ′ n 0   , for some C ′ > 0.
Finally, take the sum over s and use the Cauchy-Schwarz inequality, then
t s=1 E[p B (s)∥ ps -p s ∥ TV ] ≤C ′ t s=1 E   p B (s)k m ′ nt ∨ p B (s)k m ′ nn 0 ∨ p B (s) 2 k m ′ n 0   =O k m ′ n 0 ∨ kt m ′ nn 0 =O k m ′ 1 ∨ t n • l 0 l ∨ 1 n ,
which completes the proof of Proposition 3.
this section cite: []

Section: C.2 Error Analysis of the Non-Interactive Protocol
We complete the proof of Proposition 3 in this subsection.
this section cite: []

Section: C.2.1 Error Analysis for the Base Case 1
Since the protocol for p = 1 is the same as that for p = 2, then by the Cauchy-Schwarz inequality and the analysis in Appendix A we have
E[∥ pW -p W ∥ TV ] ≤ kE[∥ pW -p W ∥ 2 2 ] ⪯ k 2 mnl ∨ k mn .
this section cite: []

Section: C.2.2 Error Analysis for Case 2
By the analysis in Appendix C.2.1, the estimation error for the reduced block distribution is bounded by
C 3 • t 2 mnl ∨ t mn ,
for some C 3 > 0.
By Lemma 8, the estimation error for the conditional distribution induced by the invoking of the subroutine SSRSub( m 2 , n, k, l, l 0 , 1) is bounded by
C 4 k l0 l ∨ 1 n m 2 = C 4 2k l0 l ∨ 1 n m =C 4 2l 0 k ml ∨ 2k mn ≤C 4 •   4k log( k n + 1) ml ∨ 2k mn   ,
for some C 4 > 0.
Then by (12), the total error is bounded by
C 3 • t 2 mnl ∨ t mn + C 4 •   4k log( k n + 1) ml ∨ k mn   =O   k log k n + 1 ml ∨ k mn   .
this section cite: ['b11']

Section: C.2.3 Error Analysis for Case 3
By the analysis in Appendix C.2.2, the estimation error for the reduced block distribution induced by the invocation of SSR( m 2 , n, k a+1 , l, 1) is bounded by
C 5 •     k a+1 log ka+1 n + 1 ml ∨ k a+1 mn     ≤ C 5 • k a+1 m ,
for some C 5 > 0.
We have k u+1 ≥ k a+1 > n and l0 l = 1 > 1 n . Then by Lemma 8, the estimation error for the conditional distribution induced by the u-th invocation of the subroutine SSRSub(m u , n, k u , l, l 0 , 1) is bounded by
C 6 • k u+1 • k u m u n ≤ C 6 k 2 u(l-1) • k m 2 u+2 n = C 6 2 u+2 k 2 (2 l-1 ) u mn ,(23)
for some C 6 > 0.
Then by (12) and l ≥ 4, the total error is bounded by
C 5 • k a+1 m + C 6 a u=1 2 u+2 k 2 (2 l-1 ) u mn ≤ C 5 • k m + 8C 6 k 2 2 l mn = O k 2 2 l mn .
this section cite: ['b11']

Section: D The Protocol for the Case (5b) and Its Analysis
In this section, we design a refinement protocol with sample compression that achieves the optimal rates for the case (5b), summarized in the following proposition.
Proposition 4. Let p ≥ 2, k > n, ml ≥ 1000n log(mn) log k and ⌈log k⌉ ≤ l ≤ n 2 p . Then for the problem in Section 2, there exists an interactive protocol such that for any p W ∈ ∆ W , the protocol
outputs an estimate pW satisfying E[∥ pW -p W ∥ p p ] = O log p 2 k (ml) p 2 n p 2 -1 .
Note that the communication budget l ≥ ⌈log k⌉ is sufficient to encode more than one sample. A naive idea is to let each terminal transmit their i.i.d. samples directly, so that the decoder can infer the distribution based on the samples.
To achieve higher accuracy, a subset W ′ containing w with relatively larger p W (w) is identified and those p W (w) needs to be refined. A sample compression technique projects each sample to the subset W ′ , which makes the encoding of the samples efficient. The protocol designed in Appendix A is then used to refine the distribution on W ′ . We present the details as follows.
this section cite: []

Section: D.1 The Refinement Protocol with Sample Compression
Transmit multiple samples Let n 0 = ⌊ l ⌈log k⌉ ⌋ ≤ n. Each of the first m 3 encoders divides its l-bit message into n 0 frames, and each frame has ⌈log k⌉ bits. Then encode each of its first n 0 samples by one of these n 0 frames. Send the message to the decoder.
Receiving the message, the decoder can access M 1 ≜ mn
0 i.i.d. random samples (W 1 l ) M1 l=1 . Then for each w ∈ W, let p1 W (w) = M1 l=1 1 W 1 l =w M 1
and output the estimate p1 W .
this section cite: []

Section: Refinement with sample compression
Based on the estimate p1 W , the decoder computes
W ′ = w ∈ W : p1 W (w) > 2 n ,
where it is immediate that |W ′ | ≤ n -1 since p1 W is normalized. All the remaining 2m 3 encoders are informed of W ′ .
this section cite: []

Section: Let the second m
3 encoders and the decoder repeat the protocol in the first step, so that an estimate p2 W (w) is obtained by the decoder. Finally, consider the last m 3 encoders. For the i-th encoder among them, it computes W ′ ij = h(W ij ) for j = 1, ..., n, where (W ij ) n j=1 are its observed samples and
h(w) = w, w ∈ W ′ , ∅, w / ∈ W ′ .
Let W ′ = h(W ) and p W ′ be its distribution of dimension no more than n. Then each encoder holds n i.i.d. samples (W ′ ij ) n j=1 and W ′ ij ∼ p W ′ . Let these encoders and the decoder invoke the protocol IR( m 2 , n, |W ′ | + 1, l, p) defined in Appendix A (which is possible since |W ′ | + 1 ≤ n and ml ≥ 1000(|W ′ | + 1) log(mn) log n). The decoder can obtain the estimate p3 W ′ for p W ′ . Finally, for each w ∈ W, the decoder computes
p3 W (w) = p3 W ′ (w), w ∈ W ′ , p2 W (w), w / ∈ W ′ ,
and outputs the estimate p3 W .
this section cite: []

Section: D.2 Proof of Proposition 4: Error Analysis for the Protocol in Appendix D.1
It is easy to analyze the error for the rough estimate p1 W . For each w ∈ W, it is folklore that for p ≥ 1 (cf. Theorem 4 in [22] or Rosenthal's inequality [23]),
E[|p 1 W (w) -p W (w)| p ] = O p W (w) mn 0 p 2 + 1 p≥2 • p W (w) (mn 0 ) p-1 .(24)
Remark 11 (Necessity of the refinement method). For 1 ≤ p ≤ 2, taking the summation and using the Hölder's Inequality imply that
E[∥ p1 W -p W ∥ p p ] ≤ O k 1-p 2 (mn 0 ) p 2 = O k 1-p 2 log p 2 k (ml) p 2 .
The bound is tight up to logarithm factors for 1 ≤ p ≤ 2. However, for p > 2 we can only get the total error bound O log p 2 k (ml) p 2 , which is not tight. In contrast, the refined estimate p3
W can achieve a better upper bound and we show
E[∥ p3 W -p W ∥ p p ] = O log p 2 k (ml) p 2 n p 2 -1
in the following.
To complete the proof of Proposition 4, it suffices to show that
E[∥ p3 W -p W ∥ p p ] = O 1 (mn0) p 2 n p 2 -1 .
We can obtain the following preliminary results, characterizing the estimation errors for the first and the second step. The proof is derived from (24) and similar to the proof of Lemma 4: for p W (w) > 4 n ,
P p1 W (w) ≤ p W (w) 2 = O 1 (mn 0 p W (w)) p 2 .
(
By (10) in the proof of Proposition 1, we have
E |p W (w) -p3 W ′ (w)| p |w ∈ W ′ = O 1 (mnl) p 2 + n mnl p-1 p W (w) + p W (w) p 2 (mn) p 2 . (26
)
Note that
E[∥ p3 W -p W ∥ p p ] ≤ w:p W (w)≤ 4 n E |p W (w) -p3 W (w)| p + w:p W (w)> 4 n E |p W (w) -p3 W (w)| p .
It suffices to bound the above two terms separately.
If p W (w) ≤ 4 n , then by the error bounds (24) (applied to p2 W ) and (26), we have
E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p3 W ′ (w)| p + E 1 w / ∈W ′ |p W (w) -p2 W (w)| p ≤P[w ∈ W ′ ]E |p W (w) -p3 W ′ (w)| p |w ∈ W ′ + E |p W (w) -p2 W (w)| p ≤O P[w ∈ W ′ ] (mnl
) p 2 + 1 ml p-1 p W (w) + p W (w) p 2 (mn) p 2 + O p W (w) mn 0 p 2 + p W (w) (mn 0 ) p-1 =O P[w ∈ W ′ ] (mnl) p 2 + p W (w) (mn 0 ) p 2 n p 2 -1 .
Take the summation and note that |W ′ | ≤ n, then
w:p W (w)≤ 4 n E |p W (w) -p3 W (w)| p ≤O   w:p W (w)≤ 4 n P[w ∈ W ′ ] (mnl) p 2 + p W (w) (mn 0 ) p 2 n p 2 -1   ≤O E[|W ′ |] (mnl) p 2 + 1 (mn 0 ) p 2 n p 2 -1 = O 1 (mn 0 ) p 2 n p 2 -1 . (27) If p W (w) > 4 n , then P[w / ∈ W ′ ] ≤ P p1 W (w) ≤ p W (w) 2
. By (24) (applied to p2 W ), ( 25) and (26), we have
E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p3 W ′ (w)| p + E 1 w / ∈W ′ |p W (w) -p2 W (w)| p ≤E |p W (w) -p3 W ′ (w)| p |w ∈ W ′ + P[w / ∈ W ′ ] • E |p W (w) -p2 W (w)| p ≤O 1 (mnl) p 2 + 1 ml p-1 p W (w) + p W (w) p 2 (mn) p 2 +O 1 (mn 0 p W (w)) p 2 • p W (w) mn 0 p 2 + p W (w) (mn 0 ) p-1 =O 1 (mnn 0 ) p 2 + 1 ml p-1 p W (w) + p W (w) p 2 (mn) p 2 ,
where the last step is since p W (w) > 4 n and mn 0 ≥ ml 4 log k > 1000n. Take the summation and note that |{w : p W (w) > 4 n }| ≤ n, we have
w:p W (w)> 4 n E |p W (w) -p3 W (w)| p ≤O   w:p W (w)> 4 n 1 (mnn 0 ) p 2 + 1 ml p-1 p W (w) + p W (w) p 2 (mn) p 2   =O 1 (mn 0 ) p 2 n p 2 -1 ,(28)
where the last step is since n 0 = ⌊ l ⌈log k⌉ ⌋ ≤ n
this section cite: ['b21', 'b22', 'b9']

Section: E The Protocol for Cases (1d) and (5d) and Its Analysis
In this section, we design a refinement protocol with thresholding that achieves the optimal rates for cases (1d) and (5d). It suffices to prove the following proposition in this section. Proposition 5. For the problem in Section 2 and each of the following cases, there exists an interactive protocol such that for any p W ∈ ∆ W , the protocol outputs an estimate pW satisfying 1. If 1 ≤ p ≤ 2, ⌈log k⌉ ≤ l ≤ n and ml < k, then E[∥ pWp W ∥ p p ] = O log p 2 k (ml) p-1 .
2. If p > 2, ⌈log k⌉ ≤ l ≤ n and ml < n, then
E[∥ pW -p W ∥ p p ] = O log p-1 k∨log 2p-1 (mn) log 2p-1 n (ml) p-1 ∨ 1 (mn) p 2 .
To overcome the difficulty induced by the extremely tight total communication budget, huge "preys" and little "flies" among all p W (w) to be estimated should be classified and dealt with differently. The thresholding level is naturally 1  ml , since roughly ∼ ml samples can be transmitted by the first step of the protocol in Appendix D.1. For those little "flies" p W (w) ⪯ 1 ml , it is better to overlooking them than trying to estimate them. The remaining budgets should be used for refining huge "preys" p W (w) ⪰ 1 ml whose number ∼ ml is limited, by generating another independent estimate. For p > 2, sample compression strategies and the protocol in Appendix A are applied to refine the estimate similar to the refinement step of the protocol in Appendix D.1. With the help of thresholding, the resulting estimation protocol can catch the rough landscape of the distribution p W and achieve the optimal error rate under the communication constraints.
We present the protocols for two cases respectively in the following subsections and detailed error analysis can be found in Appendices E.3 and E.4.
this section cite: []

Section: E.1 Thresholding Methods for Case 1
Rough estimation Let n 0 = ⌊ l ⌈log k⌉ ⌋ ≤ n. Let the first m 2 encoders and the decoder invoke the first step (namely the "transmit multiple sample" step) of the protocol presented in Appendix D.1, so that the decoder can obtain an estimate p1 W .
this section cite: []

Section: Thresholding technique
Based on that, the decoder computes
W ′ = w ∈ W : p1 W (w) > 2 ml ,
where it is immediate that |W ′ | ≤ ml since p1 W is normalized. Let the second m 2 encoders and the decoder repeat the first step of the protocol in Appendix D.1, so that an estimate p2 W (w) is obtained by the decoder. Then for each w ∈ W, the decoder computes
p3 W (w) = p2 W (w), w ∈ W ′ , 0, w / ∈ W ′ ,
and outputs the estimate p3 W .
this section cite: []

Section: E.2 Combining Thresholding Methods and Refinement for Case 2
Rough estimation Let k ′ = ml 2000 log(mn) log n , then k ′ < ml < n and ml > 1000k ′ log(mn) log n. Let the first m 2 encoders and the decoder invoke the protocol presented in the first step of Appendix D.1. Then the decoder can obtain an estimate p1 W .
this section cite: []

Section: The mixed thresholding and refinement technique
Based on that, the decoder computes
W ′ = w ∈ W : p1 W (w) > 2 k ′ ,
where it is immediate that |W ′ | ≤ k ′ -1 since p1 W is normalized. All the remaining m 2 encoders are informed of W ′ . Then consider the second m 2 encoders. For the i-th encoder among them, it computes W ′ ij = h(W ij ) for j = 1, ..., n, where (W ij ) n j=1 are its observed samples and h(w) = w, w ∈ W ′ , ∅, w / ∈ W ′ .
Let W ′ = h(W ) and p W ′ be its distribution of dimension no more than n.
Then each encoder holds n i.i.d. samples (W ′ ij ) n j=1 and W ′ ij ∼ p W ′ . Let these encoders and the decoder invoke the protocol IR( m 2 , n, |W ′ | + 1, l, p) defined in Appendix A (which is possible since |W ′ | + 1 ≤ k ′ < n and ml ≥ 1000(|W ′ | + 1) log(mn) log n). The decoder can obtain the estimate p2 W ′ for p W ′ . Then for each w ∈ W, it computes p3 W (w) = p2 W ′ (w), w ∈ W ′ , 0, w / ∈ W ′ , and outputs the estimate p3 W . E.3 Error Analysis for the Protocol in Appendix E.1 It suffices to show that E[∥ p3 W -p W ∥ p p ] = O 1 (mn0) p 2 (ml) p 2 -1
.
We first give the following preliminary results, characterizing the estimation error for the first step. The proof is derived from (24), similar to the proof of Lemma 4 but simpler.
P p1 W (w) ≤ p W (w) 2 ≤ O 1 (mn 0 p W (w)) p 2 .(29)
Note that
E[∥ p3 W -p W ∥ p p ] ≤ w:p W (w)≤ 4 ml E |p W (w) -p3 W (w)| p + w:p W (w)> 4 ml E |p W (w) -p3 W (w)| p .
It suffices to bound the two terms separately. If p W (w) ≤ 4 ml , then by (24) (applied to p2
W ′ ), E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p2 W (w)| p + E [1 w / ∈W ′ p W (w) p ] ≤P[w ∈ W ′ ] • E |p W (w) -p2 W (w)| p + p W (w) p =O P[w ∈ W ′ ] • p W (w) mn 0 p 2 + p W (w) p =O P[w ∈ W ′ ] • 1 m 2 n 0 l p 2 + p W (w) (ml) p-1 .
Take the summation and note that |W ′ | ≤ ml, then
w:p W (w)≤ 4 ml E |p W (w) -p3 W (w)| p ≤O   w:p W (w)≤ 4 ml P[w ∈ W ′ ] • 1 m 2 n 0 l p 2 + p W (w) (ml) p-1   ≤O E[|W ′ |] (m 2 n 0 l) p 2 + 1 (ml) p-1 = O 1 (mn 0 ) p 2 (ml) p 2 -1 .(30
) If p W (w) > 4 ml , then P[w / ∈ W ′ ] ≤ P p1 W (w) ≤ p W (w) 2 . By (24) (applied to p2 W ) and (29), we have E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p2 W (w)| p + E [1 w / ∈W ′ p W (w) p ] ≤E |p W (w) -p2 W (w)| p + P[w / ∈ W ′ ] • p W (w) p ≤O p W (w) mn 0 p 2
+ p W (w) p • O 1 (mn 0 p W (w)) p 2 =O p W (w) mn 0 p 2 .
Taking the summation and noting that |{w : p W (w) > 4 ml }| ≤ ml, by the Hölder's inequality we have
w:p W (w)> 4 ml E |p W (w) -p3 W (w)| p ≤ O   w:p W (w)> 4 ml p W (w) mn 0 p 2   = O 1 (mn 0 ) p 2 (ml) p 2 -1 .
(31) Combining (30) and (31), we complete the proof.
this section cite: []

Section: E.4 Error Analysis for the Protocol in Appendix E.2

this section cite: []

Section: It remains to show that E[∥ p3
W -p W ∥ p p ] = O 1 k ′p-1 ∨ (ml) p (mn0) 2p-1 . We first give the following preliminary results, characterizing the estimation error for the first step. The proof is derived from (24), similar to the proof of Lemma 4 (where p in Lemma 4 is replaced by 2p). For p W (w) > 4 k ′ ,
P p1 W (w) ≤ p W (w) 2 ≤ O (ml) p-1 (mn 0 ) 2p-1 (p W (w)) p .(32)
By (10) in the proof of Proposition 1, we have
E |p W (w) -p2 W ′ (w)| p |w ∈ W ′ = O 1 (mnl) p 2 + k ′ mnl p-1 p W (w) + p W (w) p 2(mn) p 2 . (33)
Note that
E[∥ p3 W -p W ∥ p p ] ≤ w:p W (w)≤ 4 k ′ E |p W (w) -p3 W (w)| p + w:p W (w)> 4 k ′ E |p W (w) -p3 W (w)| p .
It suffices to bound the two terms separately. If p W (w) ≤ 4 k ′ , then by (33) (applied to p2 W ), we have
E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p2 W ′ (w)| p + E [1 w / ∈W ′ p W (w) p ] ≤P[w ∈ W ′ ]E |p W (w) -p2 W ′ (w)| p |w ∈ W ′ + p W (w) p ≤O P[w ∈ W ′ ] • 1 (mnl) p 2 + k ′ mnl p-1 p W (w) + p W (w) p 2(mn)
p 2 + O p W (w) k ′p-1 =O P[w ∈ W ′ ] (mnl
) p 2 + p W (w) k ′p-1 .
Take the summation and note that |W ′ | ≤ k ′ , then
w:p W (w)≤ 4 k ′ E |p W (w) -p3 W (w)| p ≤O   w:p W (w)≤ 4 k ′ P[w ∈ W ′ ] (mnl
) p 2 + p W (w) k ′p-1   ≤O E[|W ′ |] (mnl
) p 2 + 1 k ′p-1 = O 1 k ′p-1 . (34) If p W (w) > 4 k ′ , then P[w / ∈ W ′ ] ≤ P p1 W (w) ≤ p W (w) 2
. By (32) and (33), we have
E |p W (w) -p3 W (w)| p =E 1 w∈W ′ |p W (w) -p2 W ′ (w)| p + E [1 w / ∈W ′ p W (w) p ] ≤E |p W (w) -p2 W ′ (w)| p |w ∈ W ′ + P[w / ∈ W ′ ] • p W (w) p ≤O 1 (mnl) p 2 + k ′ mnl p-1 p W (w) + p W (w) p 2(mn)
p 2 + O (ml) p-1 (mn 0 ) 2p-1 (p W (w)) p • p W (w) p =O (ml) p-1 (mn 0 ) 2p-1 + k ′ mnl p-1 p W (w) + p W (w) p 2(mn) p 2
, where the last step is since mn 0 = m⌊ l ⌈log k⌉ ⌋ < ml < n. Take the summation and note that |{w : p W (w) > 4 k ′ }| ≤ k ′ < ml, we have
w:p W (w)> 4 k ′ E |p W (w) -p3 W (w)| p ≤O   w:p W (w)> 4 k ′ (ml) p-1 (mn 0 ) 2p-1 + k ′ mnl p-1 p W (w) + p W (w) p 2 (mn) p 2   =O (ml) p (mn 0 ) 2p-1 ∨ 1 (mn) p 2 . (35
)
Combining (34) and (35), we complete the proof.
this section cite: ['b9']

Section: G.1.1 Choose a prior distribution and lower bound the minimax risk by the Bayes risk
We can assume that W = [1 : k] without loss of generality. Let
p 1 W = 1 + ϵ 2 , 1 -ϵ 2 , 0, ..., 0 , p 2 W = 1 -ϵ 2 , 1 + ϵ 2 , 0, ..., 0 .(37)
Let Z ∼ Bern( 1 2 ) and define the prior distribution to be p Z W . Let P be an (m, n, l)-protocol defined in Section 2, then we have
sup p W ∈∆ W E[∥ pP W -p W ∥ p p ] ≥ 1 2 E[∥ pP W -p 1 W ∥ p p ] + E[∥ pP W -p 2 W ∥ p p ] =E[∥ pP W -p Z W ∥ p p ].
this section cite: []

Section: G.1.2 Convert the estimation problem into a testing problem
Let
Ẑ = arg min z∈{0,1} ∥p z W -pP W ∥ p .
Then we have
∥p Ẑ W -p Z W ∥ p ≤∥ pP W -p Ẑ W ∥ p + ∥ pP W -p Z W ∥ p ≤2∥ pP W -p Z W ∥ p .
Hence we have
E[∥ pP W -p Z W ∥ p p ] ≥ 1 2 p E[∥p Ẑ W -p Z W ∥ p p ] = 1 2 p-1 ϵ p P[ Ẑ ̸ = Z].(38)
Since Z -W mn -B m -Ẑ is a Markov chain, then by the Fano's inequality, we have
I(Z; B m ) ≥ 1 -h P[ Ẑ ̸ = Z] ,(39)
where h(p) = -p log 2 p -(1 -p) log 2 (1 -p) is the binary entropy function. If we can show that for a suitably chosen ϵ,
I(Z; B m ) ≤ 1 2 ,(40)
then by (38) and (39) we have
P[ Ẑ ̸ = Z] ≥ 1 10 , thus E[∥ pP W -p Z W ∥ 2 2 ] ⪰ ϵ p .
Then we have R(m, n, l, r) ⪰ ϵ p .
this section cite: []

Section: References
Ref_id:b0 Title: Communication-efficient learning of deep networks from decentralized data Year: (2017-04)
Ref_id:b1 Title: Federated learning: Challenges, methods, and future directions Year: (2020-05)
Ref_id:b2 Title: Advances and open problems in federated learning Year: (2021-06)
Ref_id:b3 Title: Communicationefficient distributed learning of discrete distributions Year: (2017)
Ref_id:b4 Title: Hadamard response: Estimating distributions privately, efficiently, and with little communication Year: (2019-04)
Ref_id:b5 Title: Breaking the communication-privacy-accuracy trilemma Year: (2020-12)
Ref_id:b6 Title: Lower bounds for learning distributions under communication constraints via fisher information Year: (2020-02)
Ref_id:b7 Title: Geometric lower bounds for distributed parameter estimation under communication constraints Year: (2021-12)
Ref_id:b8 Title: Estimating sparse discrete distributions under privacy and communication constraints Year: (2021-03)
Ref_id:b9 Title: Breaking the dimension dependence in sparse distribution estimation under communication constraints Year: (2021-08)
Ref_id:b10 Title: Pointwise bounds for distribution estimation under communication constraints Year: (2021-12)
Ref_id:b11 Title: Discrete distribution estimation under local privacy Year: (2016-06)
Ref_id:b12 Title: Optimal schemes for discrete distribution estimation under locally differential privacy Year: (2018-08)
Ref_id:b13 Title: Distributed estimation with multiple samples per user: Sharp rates and phase transition Year: (2021-12)
Ref_id:b14 Title: Discrete distribution estimation under user-level local differential privacy Year: (2023-04)
Ref_id:b15 Title: Unified lower bounds for interactive highdimensional estimation under information constraints Year: (2023-12)
Ref_id:b16 Title: Lq lower bounds on distributed estimation via fisher information Year: (2024-07)
Ref_id:b17 Title: Inference under information constraints II: Communication constraints and shared randomness Year: (2020-12)
Ref_id:b18 Title: Local differential privacy: Elbow effect in optimal density estimation and adaptation over Besov ellipsoids Year: (2020-08)
Ref_id:b19 Title: Optimal rates for nonparametric density estimation under communication constraints Year: (2024-03)
Ref_id:b20 Title: Distributed nonparametric estimation: from sparse to dense samples per terminal Year: (2025)
Ref_id:b21 Title: Handy formulas for binomial moments Year: (2020)
Ref_id:b22 Title: On the subspaces of L p (p > 2) spanned by sequences of independent random variables Year: (1970-09)
