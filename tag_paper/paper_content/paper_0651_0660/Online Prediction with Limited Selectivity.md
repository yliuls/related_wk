Title: Online Prediction with Limited Selectivity
Abstract: Selective prediction [Dru13, QV19] models the scenario where a forecaster freely decides on the prediction window that their forecast spans. Many data statistics can be predicted to a non-trivial error rate without any distributional assumptions or expert advice, yet these results rely on that the forecaster may predict at any time. We introduce a model of Prediction with Limited Selectivity (PLS) where the forecaster can start the prediction only on a subset of the time horizon. We study the optimal prediction error both on an instance-by-instance basis and via an average-case analysis. We introduce a complexity measure that gives instancedependent bounds on the optimal error. For a randomly-generated PLS instance, these bounds match with high probability.

Section: Introduction
In selective prediction [Dru13,QV19], a forecaster observes n numbers in [0, 1] one by one. At any time t, having observed the first t numbers, the forecaster may predict the average of the next w ≤ n -t unseen numbers. Both the stopping time t and the window length w are freely chosen by the forecaster, and only one such prediction needs to be made. The goal is to minimize the expected prediction error-the expected squared difference between the forecast and the actual average. How small can this error be? Surprisingly, Drucker [Dru13] showed that the forecaster can guarantee an error that vanishes as n → +∞, even though the sequence might be arbitrarily and adversarially chosen. Specifically, this result holds without any distributional assumption on the sequence (other than boundedness), and is thus robust to any misspecification, non-stationarity, or adversarial corruption in the data. Moreover, it directly addresses the prediction error, rather than a regret with respect to a class of experts.
In this paper, we study a variant of the selective prediction model where the forecaster only has limited selectivity. Concretely, the forecaster is given a subset of the time horizon, which specifies the timesteps on which they are allowed to make a prediction. This model captures many natural scenarios where predictions are either infeasible or unnecessary during certain time periods. For example, people tend to care about weather forecasts primarily when they have plans for outdoor activities. An investor may be restricted from trading specific commodities during particular seasons, making market predictions less relevant at those times. Similarly, epidemic forecasts are most critical before and during pandemics.
The main contributions of this work are summarized as follows:
• We introduce a theoretical model of Prediction with Limited Selectivity (PLS), which generalizes the selective prediction models of [Dru13,QV19].
• We define a complexity measure termed "approximate uniformity", which gives instance-dependent bounds on the optimal error that a forecasting algorithm can achieve on a PLS instance.
• For PLS instances that are randomly generated according to a k-monotone sequence (Definition 4), we show that the instance-dependent bounds match up to a constant factor with high probability.
this section cite: ['b3', 'b11', 'b3', 'b3', 'b11']

Section: Problem Formulation
Definition 1 (Prediction with limited selectivity). The forecaster is given n and a stopping time set T ⊆ {0, 1, 2, . . . , n -1}, and the nature secretly chooses a sequence x ∈ [0, 1] n . At each timestep t = 1, 2, . . . , n, the forecaster observes x t . At any timestep t ∈ T , after seeing x 1 , . . . , x t , the forecaster may optionally choose a window length w ∈ {1, 2, . . . , n -t} and make a prediction μ on the average µ = 1 w w i=1 x t+i . Once a prediction is made, the game ends and the forecaster incurs an error of (μ -µ) 2 . The forecaster must make one prediction before all n numbers are revealed.
An instance of PLS consists of a sequence length n and a stopping time set T . When the forecaster is fully-selective (namely, T = {0, 1, 2, . . . , n -1}), PLS recovers the selective mean prediction setup of [Dru13,QV19]. For simplicity, we focus on the special case of mean prediction, which already captures most of the interesting aspects of PLS. Nevertheless, the problem setup and our results can be easily extended to the more general prediction settings in [QV19]; see Section 5 for more details.
Following prior work on selective prediction, we measure the performance of a forecaster using its worst-case error over all possible choices of the sequence x ∈ [0, 1] n . Definition 2. The worst-case error of algorithm A is error worst (A) := sup x∈[0,1] n error(A, x), where error(A, x) denotes the expected squared error that A incurs on sequence x ∈ [0, 1] n . We will frequently use the following equivalent yet more convenient representation for a PLS instance. Definition 3. The block representation of a PLS instance with sequence length n and stopping time set T is L = (l 1 , l 2 , . . . , l m ) = (t 2 -t 1 , t 3 -t 2 , . . . , t m -t m-1 , n -t m ), where m = |T | and t 1 < t 2 < • • • < t m are the elements of T in ascending order.
In the rest of the paper, we will use the stopping time set T and the block representation L interchangeably to represent a PLS instance. Intuitively, each block length l i corresponds to l i consecutive timesteps t i + 1, t i + 2, . . . , t i + l i = t i+1 between adjacent stopping times in T . During these timesteps, the forecaster observes new data but cannot make predictions.
The fully-selective setup of [Dru13,QV19] corresponds to the block representation L that is an all-one sequence. When L consists of block lengths of various magnitudes, we naturally expect that the PLS instance becomes harder in the sense that the optimal forecaster has a higher worst-case error. We will formalize this intuition and derive instance-dependent error bounds for every PLS instance L in terms of a simple and combinatorial complexity measure of L.
In addition to the instance-dependent analysis, we will also study settings where the stopping time set T is randomly generated. Concretely, we consider a setup where each timestep between 0 and n -1 gets included in the stopping time set T independently, possibly with different probabilities. Definition 4 (Random stopping time set). For integer n ≥ 1 and (p ⋆ 0 , p ⋆ 1 , . . . , p ⋆ n-1 ) ∈ [0, 1] n , a p ⋆ -random stopping time set T is a random subset of {0, 1, . . . , n -1} obtained by independently including each element t with probability p ⋆ t .
this section cite: ['b3', 'b11', 'b11', 'b3', 'b11']

Section: Our Results
We prove both upper and lower bounds on the optimal worst-case error in PLS, both on an instanceby-instance basis and via an average-case analysis.
Approximate uniformity. We introduce a complexity measure, termed approximate uniformity, that captures the hardness of a PLS instance.
Definition 5. The approximate uniformity of PLS instance
L = (l 1 , l 2 , . . . , l m ) is U (L) := max 1≤i≤j≤m l i + l i+1 + • • • + l j max{l i , l i+1 , . . . , l j } .
For the fully-selective case that L = (1, 1, . . . , 1), we have U (L) = m. More generally, U (L) captures the "effective horizon length" in instance L. As we show in the following, the approximate uniformity roughly characterizes the lowest worst-case error that a forecasting algorithm can achieve.
this section cite: []

Section: Instance-dependent error bounds.
Our main algorithmic contribution is a forecasting algorithm with a worst-case error upper bounded in terms of U (L). This generalizes the O(1/ log n) error bound of [Dru13] for the fully-selective case. Theorem 1. For every PLS instance L, there is a forecasting algorithm with a worst-case error of O(1/ log U (L)).
We prove Theorem 1 in two steps. First, we establish an O(1/ log m) upper bound for the special case that the m block lengths are approximately uniform in the sense of being within a constant factor. Then, we reduce a general PLS instance L to the special case by merging the blocks into longer blocks with approximately uniform lengths. We show that at least Ω( U (L)) longer blocks can be obtained in this way, so the result for the special case implies the O(1/ log U (L)) upper bound.
Complementary to Theorem 1, we give two lower bounds on the worst-case error.
Theorem 2. For every PLS instance L = (l 1 , l 2 , . . . , l m ), every forecasting algorithm has a worstcase error of Ω(max{1/[ U (L)] 2 , 1/ log m}).
While the first lower bound of Ω(1/ U 2 ) does not match the upper bound in Theorem 1, it already has some interesting applications. The following corollary (proved in Appendix A) presents concrete examples where both the sequence length n and the number of blocks m tend to infinity, yet the worst-case error remains lower bounded by a constant. In the first example, the block lengths are geometrically increasing, so m is at most logarithmic in the sequence length n. The second example, inspired by the Cantor set, shows that an Ω(1) error is still unavoidable even if m is polynomial in n. Corollary 3. For every m ≥ 1, on the PLS instance L m = (2 0 , 2 1 , 2 2 , . . . , 2 m-1 ) with sequence length n = 2 m -1 and m blocks, every forecasting algorithm has an Ω(1) worst-case error. Furthermore, for every k ≥ 1, there is a PLS instance L ′ k with sequence length n = 3 k and m = 2 k+1 -1 blocks, on which every forecasting algorithm has an Ω(1) worst-case error.
The second lower bound of Ω(1/ log m) shows that an m-block PLS instance is the easiest when all blocks have the same length: An O(1/ log m) worst-case error can be achieved when l 1 = l 2 = • • • = l m , while no algorithm can achieve a worst-case error ≪ 1/ log m on any instance with m blocks. While this result might sound intuitive, our proof of the Ω(1/ log m) bound is non-trivial. The proof involves a novel hierarchical decomposition of the m blocks into a ternary tree and the design of a random process on the resulting tree. This extends a construction of [QV19] for m blocks of equal lengths, which is based on representing the m blocks as a full binary tree of depth log m.
this section cite: ['b3', 'b11']

Section: An average-case analysis.
While the instance-dependent upper and lower bounds do not match on every PLS instance, they do match with high probability when the instance is randomly generated. A sequence is k-monotone if it can be partitioned into at most k contiguous monotone subsequences. Our next result states that, if T is a p ⋆ -random stopping time set (Definition 4) for a k-monotone sequence p ⋆ , both |T | and U (T ) can be bounded in terms of ∥p ⋆ ∥ 1 with high probability. Theorem 4. Suppose that p ⋆ ∈ [0, 1] n is k-monotone and T is a p ⋆ -random stopping time set. Let m 0 := n-1 t=0 p ⋆ t . The following two hold simultaneously with probability
1 -e -m0/3 -1/n over the randomness in T : (1) |T | = O (m 0 ); (2) U (T ) = Ω(m 0 /(k log 2 n)).
As a direct corollary, our bounds on the optimal worst-case error are tight up to a constant factor with high probability. We prove this corollary in Appendix A.
Corollary 5. If k-monotone sequence p ⋆ ∈ [0, 1] n satisfies n-1 t=0 p ⋆ t ≥ Ω((k log 2 n) 1+c
) for some constant c > 0, with high probability over the randomness in the p ⋆ -random stopping time set T , the forecasting algorithm from Theorem 1 has a worst-case error that is optimal up to a constant factor.
Concretely, if k = O(1) is a constant, Corollary 5 applies whenever the expected number of stopping times,
n-1 t=0 p ⋆ t , is at least polylog(n). If k = O(n α )
is polynomial in n for some α ∈ (0, 1), we have nearly-tight bounds as long as n-1 t=0 p ⋆ t = Ω(n β ) for some β ∈ (α, 1].
this section cite: []

Section: Related Work
Most closely related to our study is the prior work on selective prediction. Drucker [Dru13] introduced the problem of selective mean prediction under the name of "density prediction game" and proved an O(1/ log n) upper bound on the expected error. Qiao and Valiant [QV19] proved a matching lower bound, and extended the positive result of [Dru13] to the setting of predicting more general functions.
Chen, Valiant, and Valiant [CVV20] introduced a more general framework that encompasses selective prediction as well as other data-collection procedures including importance sampling and snowball sampling. Brown-Cohen [BC21] subsequently obtained a faster algorithm under this framework. Qiao and Valiant [QV21] studied a "learning" variant of selective prediction, in which the learner observes multiple sequences and aims to identify the sequence with the highest average inside a prediction window of their choice.
All the positive results above are based on the Ramsey-theoretic observation that a sufficiently long, bounded sequence must be "predictable" or "repetitive" at some scale. This allows a selective forecaster to randomly select a timescale and achieve a vanishing error as the sequence length goes to infinity. Similar observations have been made in different contexts [Fei15,FKT17,MHO25].
More broadly, our setting is related to the recent work on online prediction with abstention, where the forecasting algorithm is allowed to occasionally abstain from making predictions at an additional cost [ZC16, CDG + 18, NZ20, GKCS21, GHMS23, PRT + 24].
2 Instance-Dependent Upper Bounds
this section cite: ['b3', 'b11', 'b3', 'b2', 'b0', 'b12', 'b4', 'b5', 'b8']

Section: Special Case: Approximately Uniform Block Lengths
Recall from Definition 3 that a PLS instance can be represented by a list of block lengths L = (l 1 , l 2 , . . . , l m ). Towards proving Theorem 1, we start with the case that the block lengths do not vary drastically and prove an O(1/ log m) upper bound on the worst-case error. Our algorithm is defined in Algorithm 2. It calls RandomSelect (Algorithm 1) to obtain a randomized prediction position i and length j. The algorithm then reads the first i -1 blocks of the sequence, and uses the mean of the last j blocks (namely, blocks i -j, i -j + 1, . . . , i -1) to predict the mean of the next j blocks (i through i + j -1).
Algorithm 1 (RandomSelect(s, k)) is a recursive procedure that prescribes a prediction position and a window length within 2 k consecutive blocks (with indices s to s + 2 k -1). With probability 1/k, it outputs (s + 2 k-1 , 2 k-1 ), i.e., predicting the average of the last 2 k-1 blocks using that of the first 2 k-1 . Otherwise, it computes p, the proportion of the total length of the first 2 k-1 blocks within the 2 k blocks. It then recurses on one of the two halves with probability p and 1 -p, respectively.
Algorithm 1: RandomSelect(s, k) Input: Integers s, k ≥ 1. Output: A pair (i, j) such that s ≤ i -j and i + j ≤ s + 2 k . 1 With probability 1/k, return (s + 2 k-1 , 2 k-1 );
2 p ← 2 k-1 -1 i=0 l s+i / 2 k -1 i=0 l s+i ;
3 return RandomSelect(s, k -1) with probability p, and RandomSelect(s + 2 k-1 , k -1) with the remaining probability 1 -p; Proposition 6. On PLS instance L = (l 1 , l 2 , . . . , l m ) that satisfies max{l1,l2,...,lm} min{l1,l2,...,lm} ≤ C, Algorithm 2 has a worst-case error of O(C/ log m).
This extends the result of [Dru13] for C = 1, i.e., all blocks have the same length. The proof is similar to those in prior work and deferred to Appendix B. We provide a brief proof sketch below.
Proof sketch. For k ≥ 1 and µ ∈ [0, 1], let L(k, µ) be the maximum squared error that Algorithm 2 incurs on a sequence of 2 k blocks with average µ. We prove by induction that L(k, µ) ≤ O(C/k) • µ(1 -µ). The proposition then follows from C/k = O(C/ log m) and µ(1 -µ) = O(1).
this section cite: ['b3']

Section: Algorithm 2: Prediction with Limited Selectivity on Approximately Uniform Blocks
Input: Instance L = (l 1 , l 2 , . . . , l m ). Sequential access to sequence x ∈ [0, 1] n of length
n = l 1 + l 2 + • • • + l m . 1 k ← ⌊log 2 m⌋; 2 (i, j) ← RandomSelect(1, k); 3 t ← l 1 + l 2 + • • • + l i-1 ; 4 Read x 1 , x 2 , . . . , x t ; 5 w 0 ← l i-j + l i-j+1 + • • • + l i-1 ; 6 μ ← 1 w0 (x t + x t-1 + • • • + x t-w0+1 ); 7 w ← l i + l i+1 + • • • + l i+j-1 ;
8 Predict the mean of x t+1 , . . . , x t+w as μ;
this section cite: []

Section: The General Case
To prove Theorem 1, we merge the m blocks of possibly different lengths into ≈ U (L) blocks with lengths within a constant factor, so that Proposition 6 can be applied to obtain an O(1/ log U (L)) error bound. Formally, we say that a PLS instance L ′ is a merge of another instance L, if L ′ can be obtained from L by merging consecutive blocks and taking a contiguous subsequence. Definition 6.
L ′ = (l ′ 1 , l ′ 2 , . . . , l ′ m ′ ) is a merge of L = (l 1 , l 2 , . . . , l m ) if there are 1 ≤ i 1 < i 2 < • • • < i m ′ < i m ′ +1 ≤ m + 1 such that l ′ j = l ij + l ij +1 + • • • + l ij+1-1 for every j ∈ [m ′ ].
For instance, (5, 9) is a merge of (1, 2, 3, 4, 5, 6) witnessed by (i 1 , i 2 , i 3 ) = (2, 4, 6): we merge consecutive elements to obtain (1, 5, 9, 6) and take the contiguous subsequence (5, 9). Naturally, if the merge of a PLS instance can be solved with a low worst-case error, so can the original instance. We prove the following lemma in Appendix B. Lemma 7. If L ′ is a merge of L, the minimum worst-case error that can be obtained on L is smaller than or equal to that on L ′ .
The next lemma states that any instance L has a merge of length Ω( U (L)) that consists of elements within a constant factor. Lemma 8. For every C > 1, every PLS instance L has a merge
L ′ = (l ′ 1 , l ′ 2 , . . . , l ′ m ′ ) such that: (1) m ′ ≥ ⌊(1 -1/C) • U (L)⌋; (2) max{l ′ 1 , l ′ 2 , . . . , l ′ m ′ }/ min{l ′ 1 , l ′ 2 , . . . , l ′ m ′ } ≤ C.
Proof. By definition of U , there exist indices
1 ≤ i 0 ≤ j 0 ≤ m such that li 0 +li 0 +1+•••+lj 0 max{li 0 ,li 0 +1,...,lj 0 } = U (L).
Using the shorthands L := l i0 + l i0+1 + • • • + l j0 and M := max{l i0 , l i0+1 , . . . , l j0 }, we have L = U (L) • M . Then, we construct L ′ by merging the block lengths l i0 , l i0+1 , . . . , l j0 . We set T := M/(C -1) and greedily form longer blocks of length ≈ T . Formally, we run the following procedure:
1. Start with i 1 = i 0 and counter cnt = 1.
2. Check whether l icnt + l icnt+1 + • • • + l j0 < T . If so, set m ′ = cnt -1, L ′ = (l ′ 1 , l ′ 2 , . . . , l ′ m ′ )
, and end the procedure.
this section cite: []

Section: Otherwise, find the smallest
k ∈ {i cnt , i cnt + 1, . . . , j 0 } such that l icnt + l icnt+1 + • • • + l k ≥ T . 4. Set l ′ cnt = l icnt + l icnt+1 + • • • + l k and i cnt+1 = k + 1. Increment cnt by 1 and return to Step 2.
Clearly, the resulting
L ′ = (l ′ 1 , l ′ 2 , . . . , l ′ m ′ ) is a merge of L witnessed by indices i 1 , i 2 , . . . , i m ′ +1 . The construction ensures that l ′ j ∈ [T, T + M ) for every j ∈ [m ′ ]. Therefore, max{l ′ 1 , l ′ 2 , . . . , l ′ m ′ } min{l ′ 1 , l ′ 2 , . . . , l ′ m ′ } < T + M T = M/(C -1) + M M/(C -1) = C.
The size of the merge is at least
L T +M = U (L)•M M/(C-1)+M = (1 -1/C) • U (L) .
Theorem 1 directly follows from Proposition 6, Lemma 7, and Lemma 8.
Proof of Theorem 1. Applying Lemma 8 with C = 2 shows that L has a merge
L ′ = (l ′ 1 , l ′ 2 , . . . , l ′ m ′ ) such that m ′ ≥ ⌊ U (L)/2⌋ and max{l ′ 1 , l ′ 2 , . . . , l ′ m ′ }/ min{l ′ 1 , l ′ 2 , . . . , l ′ m ′ } ≤ 2.
By Proposition 6, there is a forecasting algorithm for L ′ with a worst-case error of O(1/ log m ′ ) = O(1/ log U (L)). Then, Lemma 7 gives a forecasting algorithm for L with the same error bound.
i := {l 1 + l 2 + • • • + l i-1 + j : j ∈ [l i ]}.
We prove Theorem 9 using the following observation: Regardless of the prediction window [t + 1, t + w] chosen by the forecasting algorithm, an Ω(1/ U (L)) fraction of the timesteps within the window must come from the same unseen block.
Lemma 10. Let L = (l 1 , l 2 , . . . , l m ) be a PLS instance with sequence length n and stopping time set T . Then, for every
i 0 ∈ [m], t = l 1 + l 2 + • • • + l i0-1 ∈ T and w ∈ [n -t], there exists i ∈ {i 0 , i 0 + 1, . . . , m} such that |B i ∩ [t + 1, t + w]| ≥ w 2 U (L) .
The proof is deferred to Appendix C. Next, we show how Theorem 9 follows from Lemma 10.
Proof of Theorem 9 assuming Lemma 10. Consider the random sequence x ∈ {0, 1} n constructed by setting all entries within each block to the same bit chosen independently and uniformly at random. Formally, we draw µ 1 , µ 2 , . . . , µ m ∼ Bernoulli(1/2) independently. Then, for each i ∈ [m] and j ∈ B i , we set x j = µ i .
Fix a forecasting algorithm A. For t ∈ T and w ∈ [n -t], let E t,w denote the event that A makes a prediction at time t on X t,w :=foot_0 w w i=1 x t+i . We will show that, conditioning on event E t,w and any observation x 1:t = (x 1 , x 2 , . . . , x t ), the conditional variance of X t,w is at least Ω(1/[ U (L)] 2 ). This would imply that the conditional expectation of the squared error incurred by A is lower bounded by Ω(1/[ U (L)] 2 ), and the theorem would then follow from the law of total expectation. Recall that t ∈ T must be of form l 1 + l 2 + • • • + l i0-1 for some i 0 ∈ [m]. Then, X t,w can be equivalently written as X t,w = m i=i0 α i • µ i , where α i := 1 w |B i ∩ [t + 1, t + w]| denotes the fraction of timesteps in [t + 1, t + w] that fall into block B i . Since we sample µ 1 , . . . , µ m independently, conditioning on event E t,w and the observations x 1:t -both of which are solely determined by the randomness in µ 1 , µ 2 , . . . , µ i0-1 and A-each of µ i0 , µ i0+1 , . . . , µ m still follows Bernoulli(1/2) independently and has a variance of 1/4. Therefore, the conditional variance of X t,w is given by
Var [X t,w | E t,w , x 1:t ] = Var m i=i0 α i • µ i | E t,w , x 1:t = 1 4 m i=i0 α 2 i .
By Lemma 10, there exists i ∈ {i
0 , i 0 + 1, . . . , m} such that α i ≥ 1 2 U (L) , so Var [X t,w | E t,w , x 1:t ] is at least 1 4 • 1 2 U (L) 2 = 1 16[ U (L)] 2 .
this section cite: []

Section: Hard Instance via Tree Construction
Theorem 11 (Second part of Theorem 2). For every PLS instance L = (l 1 , l 2 , . . . , l m ), every forecasting algorithm A has a worst-case error of error worst (A) ≥ Ω (1/ log m).
Similar to the proof of Theorem 9, we randomly generate a sequence x by first choosing µ ∈ [0, 1] m , and setting every entry within the i-th block to µ i . The key difference is that, instead of drawing each µ i independently, we carefully design the correlation between different entries of µ. This correlation structure is specified by a tree construction similar to [QV19]: We build a tree whose leaves correspond to the m entries µ 1 , . . . , µ m . We assign a noise to each edge of the tree, and the value of a leaf is set to the sum of noises on the root-to-leaf path. Again, we will argue that for every possible prediction window [t + 1, t + w], even after seeing the first t entries, the conditional variance in the average of x t+1 , . . . , x t+w is still sufficiently high.
A key difference between our construction and that of [QV19] is that they focused on the special case where l 1 = l 2 = • • • = l m = 1, so that a full binary tree of depth log 2 m suffices. In contrast, to handle the general case, our construction involves a ternary tree in which every internal node has either two or three children, depending on whether the current subtree contains a long block whose length dominates the total block length.
We formally introduce our tree construction as follows.
Definition 7 (Tree construction). Given a PLS instance (l 1 , l 2 , . . . , l m ), we construct a tree using the following recursive procedure:
• If m = 1, return a tree with a single leaf node that corresponds to l 1 .
•
Let S := l 1 + l 2 + • • • + l m . If there exists i ⋆ ∈ [m] such that l i ⋆ > S/2, such index i ⋆ must be
unique, and we construct a ternary tree where: (1) The left subtree of the root node is the tree construction for (l 1 , . . . , l i ⋆ -1 );
(2) The middle subtree is a single leaf node corresponding to the i ⋆ -th block;
(3) The right subtree is the tree construction for (l i ⋆ +1 , . . . , l m ).
• Otherwise, every l i is at most S/2. We choose the cutoff i ⋆ ∈ [m] as the smallest index such that
l 1 + l 2 + • • • + l i ⋆ ≥ S/4. Note that we must have l 1 + l 2 + • • • + l i ⋆ = (l 1 + l 2 + • • • + l i ⋆ -1 ) + l i ⋆ ≤ S/4 + S/2 = 3S/4.
We recursively build two trees for (l 1 , l 2 , . . . , l i ⋆ ) and (l i ⋆ +1 , l i ⋆ +2 , . . . , l m ), and return the tree obtained from joining these two subtrees.
By construction, the tree has m leaf nodes, each of which corresponds to one of the m blocks. For each node v in the tree, let I(v) denote the set of block indices that correspond to one of the leaves in the subtree rooted at v. It is clear that every I(v) is of form {i, i + 1, . . . , j} for some 1 ≤ i ≤ j ≤ m. We write size(v) := |I(v)| as the number of leaves in the subtree rooted at v.
Next, we assign a noise magnitude to each node in the tree.
Definition 8. The noise magnitude at node v is set to σ(v) := 1 -ln(size(v)) ln m .
The noise magnitude is always in [0, 1]: it takes value 0 at the root node and takes value 1 at every leaf node. Then, we assign random values in [0, 1] to the nodes in the tree construction as follows.
Definition 9 (Node value). The root node r is assigned value µ r = 1/2 deterministically. Then, for every edge (u, v) in the tree, after µ u is determined, we independently draw µ v such that:
µ v ∈ 1 -σ(v) 2 , 1 + σ(v) 2 and E [µ v | µ u ] = µ u .
Note that the above is well-defined: By Definition 8, it always holds that σ(u) < σ(v). So, regardless
of whether µ u equals 1-σ(u) 2 or 1+σ(u) 2 , we always have µ u ∈ 1-σ(v) 2 , 1+σ(v) 2
. Therefore, there exists a unique distribution over 1-σ(v) 2 , 1+σ(v) 2 with an expectation of µ u .
Finally, we construct the hard instance by setting the value of each block to the corresponding node value in the tree construction.
Definition 10 (Hard instance). Given a PLS instance L = (l 1 , l 2 , . . . , l m ), we construct a tree following Definition 7 and assign values to its nodes following Definitions 8 and 9. For each i ∈ [m], let µ i denote the value of the leaf node that corresponds to the i-th block. Finally, the sequence x consists of l 1 copies of µ 1 , l 2 copies of µ 2 , . . ., l m copies of µ m in order.
this section cite: ['b11', 'b11']

Section: Structural Properties
Towards proving Theorem 11 using the hard instance from Definition 10, we make some observations on the tree structure as well as the node values. We first note that, for each edge (u, v) in the tree, the conditional variance of µ v given µ u takes a simple form. Indeed, this is the main motivation behind the choices of the noise magnitudes and node values. Lemma 12. For every edge (u, v) in the tree and conditioning on any realization of µ u , it holds that
Var [µ v | µ u ] = ln(size(u))-ln(size(v)) 4 ln m .
Proof. Since adding a constant does not change the variance,
Var [µ v | µ u ] = Var [µ v -1/2 | µ u ] = E (µ v -1/2) 2 | µ u -[E [µ v -1/2 | µ u ]] 2 . By Definition 9, we have E [µ v | µ u ] = µ u , so the second term [E [µ v -1/2 | µ u ]] 2 reduces to (µ u -1/2) 2 . Then, we note that |µ u -1/2| = σ(u)/2 and |µ v -1/2| = σ(v)/2 always hold, which further simplifies Var [µ v | µ u ] into [σ(v)] 2 /4 - [σ(u)] 2 /4.
Finally, the lemma follows from the choices of of σ(u) and σ(v) in Definition 8.
The next technical lemma (which we prove in Appendix C) states that, for any 1 ≤ i ≤ j ≤ m, there exists an edge (u, v) in the tree such that: (1) Observing the first i -1 blocks does not reveal the value of µ v ; (2) The remaining variance of µ v has a significant contribution to the average of blocks i, i + 1, . . . , j. To state the lemma succinctly, we define the "total length" of a set S. We will mostly use this notation for S = I(v) (where v is a node in the tree) or S = [i, j] (where
1 ≤ i ≤ j ≤ m). Definition 11. The total length of set S is totlen(S) := m i=1 l i • 1 [i ∈ S].
Lemma 13. For any 1 ≤ i ≤ j ≤ m, there exists an edge (u, v) in the tree such that: (1) I(v) ∩ {1, 2, . . . , i -1} = ∅; (2) totlen(I(v)) ≥ Ω(1) • totlen([i, j]); (3) size(v) ≤ size(u)/2.
this section cite: []

Section: Lower Bound in Terms of Number of Blocks
We prove Theorem 11 by putting together Lemmas 12 and 13. As in the proof of Theorem 9, we can decompose any prediction window [t + 1, t + w] into several complete blocks i 0 , i 0 + 1, . . . , j 0 -1 and a possibly incomplete block j 0 . If the complete blocks constitute at least half of the window, we apply Lemma 13 to identify an edge (u, v) in the tree, such that the noise in µ v | µ u contributes an Ω(1/ log m) variance to the average to be predicted. Otherwise, we lower bound the variance directly by considering the edge above the leaf that corresponds to block j 0 .
Proof of Theorem 11. We consider the random sequence x ∈ {0, 1} n constructed in Definition 10 and fix a forecasting algorithm A. For t ∈ T and w ∈ [n -t], let E t,w denote the event that A makes a prediction at time t on X t,w := 1 w w i=1 x t+i . We will show that, conditioning on any event E t,w as well as the observations x 1:t = (x 1 , x 2 , . . . , x t ), the conditional variance of X t,w is at least Ω(1/ log m). This would lower bound the conditional expectation of the squared error incurred by A by Ω(1/ log m), and the theorem would then follow from the law of total expectation.
Recall that t ∈ T must be of form
l 1 + l 2 + • • • + l i0-1 for some i 0 ∈ [m]. Let j 0 ∈ [m] be the smallest number such that l i0 + l i0+1 + • • • + l j0 ≥ w. Let δ := w -(l i0 + l i0+1 + • • • + l j0-1 ).
Consider the following two cases, depending on whether δ exceeds half of the window length w:
• Case 1: δ ≥ w/2. Let v be the leaf that corresponds to the j 0 -th block in the tree construction, and u be the parent of v. Note that size(v) = 1 and size(u) ≥ 2. By Lemma 12, Var [µ v | µ u ] is given by ln(size(u))-ln(size(v)) 4 ln m ≥ ln 2 4 ln m = Ω 1 log m . Furthermore, since event E t,w and x 1:t only depend on the realization of µ 1 , µ 2 , . . . , µ i0-1 , the value of µ j0 (namely, µ v ) still has an Ω(1/ log m) variance conditioning on E t,w and x 1:t . Then, since δ ≥ w/2 entries among x t+1 , x t+2 , . . . , x t+w are set to µ j0 , Var [X t,w | E t,w , x 1:t ] is at least 1 4 • Var [µ j0 | E t,w , x 1:t ] ≥ Ω(1/ log m).
• Case 2: δ < w/2. In this case, we have totlen(
[i 0 , j 0 -1]) = l i0 + l i0+1 + • • • + l j0-1 = w -δ ≥ w/2.
Applying Lemma 13 with i = i 0 and j = j 0 -1 gives an edge (u, v) such that: (1) I(v) ∩ {1, 2, . . . , i 0 -1} = ∅; (2) totlen(I(v)) ≥ Ω(1) • totlen([i 0 , j 0 -1]); (3) size(v) ≤ size(u)/2.
this section cite: []

Section: By Lemma 12, Var [µ
v | µ u ] is equal to ln(size(u))-ln(size(v))
4 ln m ≥ ln 2 4 ln m = Ω(1/ log m). Since event E t,w and the observation of x 1:t only depend on the realization of µ 1 , µ 2 , . . . , µ i0-1 , and none of these i 0 -1 leaves is inside the subtree rooted at v, µ v still has an Ω(1/ log m) conditional variance. Furthermore, within the length-w prediction window, the number of entries that are affected by µ v is exactly totlen(I(v)) ≥ Ω(1) • totlen([i 0 , j 0 -1]) ≥ Ω(1) • w. Therefore, the conditional variance of X t,w is at least Ω(1) • Var [µ v | E t,w , x 1:t ] ≥ Ω(1/ log m).
this section cite: []

Section: An Average-Case Analysis
As a warm-up towards proving Theorem 4, we analyze the special case that p ⋆ consists of n identical entries. The full proof is presented in Appendix D. Proposition 14. Let T be a p ⋆ -random stopping time set for p ⋆ = (p, p, . . . , p) ∈ [0, 1] n . With probability at least 1 -e -np/3 -1/n over the randomness in T : (1
) |T | ≤ 2np = O(np); (2) U (T ) ≥ n/⌈(2 ln n)/p⌉ -1 = Ω(np/ log n).
Proof. We first note that |T | follows Binomial(n, p), so a multiplicative Chernoff bound gives Pr [|T | > 2np] ≤ e -np/3 . Towards lower bounding U (T ), let L = (l 1 , l 2 , . . . , l m ) be the block representation of T and let L 0 := ⌈(2 ln n)/p⌉. By definition, U (T ) is at least l1+l2+•••+lm max{l1,l2,...,lm} = n-min T max{l1,l2,...,lm} . If we additionally have max{l 1 , l 2 , . . . , l m } ≤ L 0 and min T ≤ L 0 , we would have U (T ) ≥ n-L0 L0 = n/⌈(2 ln n)/p⌉ -1. Thus, it remains to show that, with probability at least 1 -1/n, both max{l 1 , l 2 , . . . , l m } ≤ L 0 and min T ≤ L 0 hold.
Consider the complementary event: for either max{l 1 , l 2 , . . . , l m } > L 0 or min T > L 0 to hold, there must be some t ∈ {0, 1, . . . , n -L 0 -1} such that T ∩ [t + 1, t + L 0 ] = ∅. For each fixed t, t+1, t+2, . . . , t+L 0 get included in T independently with probability p. Thus, T ∩[t+1, t+L 0 ] = ∅ holds with probability (1-p) L0 . By the union bound, Pr [max{l 1 , l 2 , . . . , l m } > L 0 ∨ min T > L 0 ] is at most (n -L 0 ) • (1 -p) L0 , which is further upper bounded by ne -pL0 ≤ ne -2 ln n = 1/n using 1 -p ≤ e -p and L 0 ≥ (2 ln n)/p. This completes the proof.
Our proof of Proposition 14 implies that, when p ⋆ consists of n copies of the same value p ∈ [0, 1], the resulting value of U is at least ≈ np/ log n with high probability. Furthermore, the proof of this lower bound still holds if each entry of p ⋆ is lower bounded by p instead of exactly equal to p. To prove Theorem 4, the addition step is then to show that, within each k-monotone sequence p ⋆ , we can always find a consecutive subsequence of length n ′ such that each entry is at least p ′ , and n ′ p ′ is at least ≈ n t=1 p ⋆ t /(k log n). This is done by identifying a monotone subsequence in p ⋆ that contributes at least a (1/k)-fraction of the sum, and then finding an appropriate prefix or suffix of that subsequence.
this section cite: []

Section: Discussion
An obvious open problem is to tighten the instance-dependent error bounds (Theorems 1 and 2). Since the optimal error is Θ(1/ log n) for the full-selectivity case [Dru13,QV19], one might hope to strengthen the 1/[ U (L)] 2 lower bound to 1/ log U (L), or at least 1/polylog( U (L)). Unfortunately, as we show in Proposition 16 (Appendix E), this is not possible: there exists a family of instances (L k ) +∞  k=1 such that U (L k ) → +∞ as k → +∞, but a worst-case error of O(1/ U (L k )) can be achieved on each L k .
Therefore, to obtain tighter instance-dependent bounds, we need to identify a complexity measure that characterizes the hardness of PLS more exactly. A concrete starting point is to examine the instance family from Proposition 16, which is based on a construction that resembles the Cantor set. Roughly speaking, these instances suggest that a sharper complexity measure should account for the number of approximately uniform blocks that can be obtained via not only merging consecutive blocks, but also "skipping" some shorter blocks at the cost of an additional term in the prediction error.
While we focus on predicting the average of a number sequence, our results can be easily extended to the setting of predicting more general functions (including smooth and concatenation-concave functions studied by [QV19]). In particular, they imply an O(1/ log 1/2 U (L)) upper bound (on the worst-case absolute error) for predicting smooth functions and an O(1/ log U (L)) bound (on the squared error) for concatenation-concave functions. Since the average function is both smooth and concatenation-concave, the lower bounds in Theorem 2 also apply to these broader function classes.
Yet another natural direction is to revisit other classic online learning settings (such as the experts problem) from the perspective of limited selectivity, i.e., when the learner is only allowed to change its prediction or action on some given timesteps. The approximate uniformity measure as well as some of our proof techniques would be natural first steps towards understanding these models.
this section cite: ['b3', 'b11', 'b11']

Section: References
Ref_id:b0 Title: Faster algorithms and constant lower bounds for the worst-case expected error Year: (2021)
Ref_id:b1 Title: Online learning with abstention Year: (2018)
Ref_id:b2 Title: Worst-case analysis for randomly collected data Year: (2020)
Ref_id:b3 Title: High-confidence predictions under adversarial uncertainty Year: (2013)
Ref_id:b4 Title: Why are images smooth? Year: (2015)
Ref_id:b5 Title: Chasing ghosts: competing with stateful policies Year: (2017)
Ref_id:b6 Title: Adversarial resilience in sequential prediction via abstention Year: (2023)
Ref_id:b7 Title: Online selective classification with limited feedback Year: (2021)
Ref_id:b8 Title: Expected correlation in time-series analysis Year: (2025)
Ref_id:b9 Title: Fast rates for online prediction with abstention Year: (2020)
Ref_id:b10 Title: Bandits with abstention under expert advice Year: (2024)
Ref_id:b11 Title: A theory of selective prediction Year: (2019)
Ref_id:b12 Title: Exponential weights algorithms for selective learning Year: (2021)
Ref_id:b13 Title: The extended littlestone's dimension for learning with mistakes and abstentions Year: (2016)
