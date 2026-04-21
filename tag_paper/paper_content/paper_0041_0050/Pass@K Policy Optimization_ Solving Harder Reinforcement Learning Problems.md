Title: Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems
Abstract: Reinforcement Learning algorithms commonly sample multiple (n > 1) solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes individual sample performance over the diversity and collective utility of a set of samples. Such algorithms under-utilize the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a multivariate transformation on batches of rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that feature a large maximum reward when considered jointly. Our primary contribution is to derive novel low variance unbiased estimators for the pass@k and its gradient, in both the binary and continuous reward settings. We show that optimizing with these estimators reduces to reinforcement learning with (batches of) rewards that have been jointly transformed by a function that is stable and efficient to compute.While previous efforts propose transformations for k = n, our transformations are the first to enable robust optimization of the pass@k for any arbitrary k ≤ n. Rather than simply trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 performance alongside significant pass@k gains.We validate our transformations on illustrative toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source models GEMMA2 and LLAMA3.1. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k. Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely by improving exploration through the prioritization of joint utility over the utility of individual samples

Section: Introduction
Recent years have seen the rapid rise of large language models (LLMs) trained with internet-scale pretraining data [RNS + 18] with post training using both supervised fine-tuning [WBZ + 21] and reinforcement learning (RL) [AAA + 23, Tea23, Ant, GYZ + 25]. The seminal paradigm of RL with human feedback [CLB + 17] is limited by the human-derived data it is based on and the reward hacking issues that arise from the use of subjective signals more generally [ABC + 21]. To enable progress toward superhuman capabilities, current work is focusing on grounded reward signals that are free of fine-grained human input as in code generation [SJTR23, LWG + 22, DLJ + 24, YTC + 23, GZC + 24] and mathematics [LCC + 22, AT, CTO + 25, YSG + 23].
0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 x and 0.0 0.2 0.4 0.6 0.8 1.0 maxg@k (a function of ) k = 1 k = 2 k = 4 k = 8 k = 16 0.0 0.2 0.4 0.6 0.8 1.0 raw reward g (a function of x) maxg@k for x N( , 0.1) and various k g(x)
(a) g(x) and maxg@k.
0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 x and 7 6 5 4 3 2 1 0 1 2 maxg@k (a function of ) k = 2 k = 4 k = 8 k = 16 0.0 0.2 0.4 0.6 0.8 1.0 raw reward g (a function of x) maxg@k for x N( , 0.1) and various k g(x)
(b) g(x) and ∇ θ maxg@k.
Figure 1: The effect of k on the optimal policy for a one-dimensional toy problem. The policy is normal with mean parameter θ and fixed standard deviation 0.1. For the max g @k objective (left, defined in Equation ( 11)) the optimal θ corresponds to the horizontal position with maximum max g @k. For the derivative (right, the estimation of which is the focus of this paper), the optimal θ corresponds to the location of the zero crossing. For larger k the optimal θ is more risk tolerant, allowing more samples to exceed one (getting zero reward) in order to increase the chance of obtaining at least one sample close to, but less than one (getting a large reward). See Section 5.1 for more details.
The policy gradients family of RL methods [Wil92] has proven effective in language model training [GYZ + 25]. To scale to new capabilities, model training needs to tackle challenging RL task sets with no known solutions but for which correctness may be verified, as in formal mathematics environments [AT]. In such settings, the RL training loop both updates model parameters and searches for solutions to problems at the continuously advancing frontier of model capabilities.
The specific search algorithm introduces a coupling between inference and model updates, which means that naively optimizing the expected single sample reward, or pass@1 may be suboptimal. While various inference-time search methods are possible [HYM + 24, KZA + 24, LKB + 23, WSL + 24], simply taking multiple independent samples from the model has proven rather effective [OIW + 23]. Our contribution is to couple this simple search method with model parameter updates by enabling robust optimization of the pass@k objective, which is the expected maximum reward over k independent samples.
this section cite: ['b32']

Section: Related Literature
The pass@k was championed by [CTJ + 21a] who gave a popular unbiased estimator of the metric which we derive from a new perspective (to set up our gradient estimators) as Theorem 1, generalize to continuous rewards in Theorem 3, and provide additional characterisations in Corollaries 2 and 3. Concurrently with our work [TZSM25] offered an elegant variance reduction method for the gradient of the pass@k which corresponds to the special case n = k of our Equation (33). Interpreting pass@k in terms of a partial sort, [CTV19] and [XDC + 20] present elegant approximations that are rather general but less efficient in our setting. Others have provided variational approximations for handling the closely related Best-of-N [CTG + 24, AVAC24] and other more general [BSB + 24] inference-time algorithms. The contrasting idea of training a model to approximate the Best-of-N prediction with a single sample was addressed by [SDH + 24].
Our contribution can be interpreted as a generalization and variance reduction of [TZSM25]. For a general discussion of gradient estimation, variance reduction, and Monte Carlo, we recommend [MRFM20,Owe13].
Overview and Contributions Our theme is constructing robust estimators of the pass@k and its gradient given n ≥ k samples by averaging (over all n k subsets of size k) simple estimators that are functions of k samples. This is straightforward for binary rewards (Section 2), using the counting proof of Theorem 4. We generalize to continuous rewards using the key trick of assuming without loss of generality that the rewards are sorted, as in Section 3. Finally, we give baselining methods that require more involved derivations due to averaging over all subsets that do not include a given element (to retain unbiasedness) but which boil down to the same easy-to-apply results in Section 4, yielding our Pass-at-k Policy Optimization (PKPO). We present toy experiments in Section 5.1 which demonstrate the variance reduction afforded by our estimators. Finally, Section 5.2 demonstrates that using our reward transformation solves more tasks and selectively optimizes pass@k through RL experiments on GEMMA2 [TRP + 24] and LLAMA3.1 [GDJ + 24], showcasing real-world impact.
How to Apply this Method It is easy to adapt any policy gradient algorithm to use our results. Assume a vector (g(x 1 ), g(x 2 ), . . . , g(x n )) of per-sample rewards for a given task. For example, the x i could be model samples of source code addressing a specific task (which should be the same for all n samples), and g could provide a numeric score that measures how many tests the code passes, or an overall binary pass indicator, or some combination with additional stylistic or brevity terms, etc. Then in order to optimize the pass@k of Equation (1) (or the continuous analog max g @k of Equation ( 11)) we simply transform the vector of rewards using either the sloo or the sloo minus one function of Listing 1, which map R n → R n . 1
this section cite: ['b30', 'b10', 'b30', 'b20', 'b22']

Section: Binary Rewards
Given a binary reward function f : X → {0, 1} on the action space X , the pass@k for the model p(x|θ) is the probability that at least one of k samples drawn i.i.d. is correct:
pass@k = P k i=1 [f (x i ) = 1] (1) = E 1 - k i=1 (1 -f (x i )) ,(2)
where the expectation is over i.i.d.
x 1 , x 2 , • • • , x k ∼ p(x|θ).
this section cite: []

Section: An Unbiased pass@k Estimator
An estimator for the pass@k was given in [CTJ + 21a]: given n ≥ k i.i.d. samples of which c are correct, the estimator is
ρ(n, c, k) ≡ 1 - n-c k n k . (3
)
The following was proven in [CTJ + 21a]; we give a different proof that sets up our gradient estimator.
Theorem 1. ρ(n, c, k) is an unbiased estimator of the pass@k.
Proof. Let x 1 , x 2 , • • • , x n ∼ p(x|θ), f i = f (x i )
, and I be a set of k elements sampled uniformly without replacement from {1, 2, . . . , n}. Then
pass@k = E x1,x2,...,xn E I 1 - i∈I (1 -f i ) .(4)
Averaging over all subsets of size k recovers ρ:
1 n k |I|=k I⊆{1,2,...,n} 1 - i∈I (1 -f i ) = 1 - 1 n k |I|=k I⊆{1,2,...,n} i∈I (1 -f i ) (5
) = 1 - n-c k n k (6) ≡ ρ(n, c, k),(7)
where ( 6) holds because the sum on the r.h.s. of ( 5) is the number of subsets of size k of the (n -c) incorrect elements. Since averaging in this way retains unbiasedness, this completes the proof. We show in Corollary 2 that no such unbiased estimator exists for n < k, and in Corollary 3 that the asymptotic variance of this estimator decreases at a rate of 1/n.
0 1 2 3 4 5 6 7 ascending sort index i 0.0 0.2 0.4 0.6 0.8 1.0
i (normalized) Element Weight for Various k k = 1 k = 2 k = 4 k = 8
Figure 2: The effect k has on the effective weight µ i / n k of (12) for a mini-batch of size n = 8. This is the weight of the contribution of each sample assuming that the samples have been sorted in ascending order from left to right. The horizontal axis is the sort index. For k = n = 8 only the largest sample is included; for k = 1 all samples are weighted equally. Intermediate values interpolate these extremes in a precise manner that gives rise to unbiased gradient estimation.
this section cite: []

Section: An Unbiased pass@k Gradient Estimator
Given a mini-batch of n i.i.d. samples x 1 , x 2 , . . . , x n from p(x|θ) with corresponding correctness labels f i ∈ {0, 1}, we want to optimize the pass@k w.r.t. the model parameters θ. Letting c = n i=1 f i be the number of correct samples, we will demonstrate unbiasedness of the estimator
∇ = n i=1 r i ∇ θ log p(x i |θ), where r i = k n if f i = 1 k n • ρ(n -1, c, k -1) if f i = 0,(8)
that assigns more weight to correct samples, while also assigning some reward to incorrect samples to encourage exploration. The following well-known results will be used to show that (8) is unbiased.
Lemma 1 (Policy Gradients). For any absolutely continuous distribution p(x|θ)
E x∼p(x|θ) [r(x)∇ θ log p(x|θ)] = ∇ θ E x∼p(x|θ) [r(x)] .(9)
Corollary 1. If c is constant w.r.t. both θ and x then E p(x|θ) [c∇ θ log p(x|θ)] = 0. Proof. By Lemma 1, E p(x|θ) [c∇ θ log p(x|θ)] = ∇ θ E [c] = ∇ θ c = 0.
We can now give our first main result:
Theorem 2. ∇ is an unbiased estimator of the gradient of the pass@k:
E x1,x2,...,xn∼p(x|θ) ∇ = ∇ θ pass@k.(10)
See Appendix A.3 for a proof.
this section cite: []

Section: Continuous Rewards
We generalize the pass@k to non-binary rewards g : X → R as
max g @k ≡ E max {g(x i )} k i=1 .(11)
this section cite: []

Section: An Unbiased max g @k Estimator
The following estimator for the max g @k is a direct analog of ρ: given n ≥ k i.i.d. samples, assuming w.l.o.g. that the rewards
g i = g(x i ) are sorted, so that g 1 ≤ g 2 ≤ • • • ≤ g n the estimator is ρ (g) (n, c, k) ≡ 1 n k n i=k µ i g i ,(12)
where
µ i = i -1 k -1 .(13)
To compute this stably we cancel factors in the binomial coefficients to getfoot_1
ρ (g) (n, c, k) ≡ k n -k + 1 n i=k g i k-1 j=1 i -j n -j + 1 .(14)
Theorem 3. ρ (g) (n, c, k) is an unbiased estimator of the max g @k.
Proof. The proof is similar to Theorem 1. Here we exploit the assumption that the g i are sorted, so
1 n k |I|=k I⊆{1,2,...,n} max i∈I g i = 1 n k |I|=k I⊆{1,2,...,n} g max i∈I (15
) = 1 n k n i=k µ i g i (16
) ≡ ρ (g) (n, c, k),(17)
since µ i is the number of subsets of 1, 2, . . . , i -1 of size k -1, which equals i-1 k-1 . The sum starts at k because all subsets of size k include elements that are greater than or equal to g k . See Line 17 of Listing 1 for an implementation of ρ (g) .
this section cite: []

Section: An Unbiased max g @k Gradient Estimator
We propose the gradient estimator
∇ (g) = n i=1 s i ∇ θ log p(x i |θ),(18)
where if we assume w.l.o.g. that the g i are sorted, the s i are a weighted combination of them,
s i = 1 n k n j=i m ij g j ,(19)
where the diagonals are
m ii = i-1 k-1 if i ≥ k -1 0 otherwise,(20)
and the off-diagonals are
m ij = j-2 k-2 if (j > i) ∧ (j ≥ k) ∧ (k ≥ 2) 0 otherwise.(21)
Theorem 4. ∇ (g) is an unbiased estimator of the gradient of the max g @k:
E x1,x2,...,xn∼p(x|θ) ∇ (g) = ∇ θ max g @k.(22)
Proof. The proof is analogous to that of Theorem 2. Here we have
∇ ≡ ρ (g) (n, c, k)∇ θ n i=1 log p(x i |θ) (23
) = 1 n k |I|=k I⊆{1,2,...,n} max j∈I g j n i=1 ∇ θ log p(x i |θ) (24
) E ≡ 1 n k n i=1 ∇ θ log p(x i |θ) n j=1 m ij g j ,(25)
By assumption the g i are sorted, so max j∈I g j = g max j∈I . Therefore m ij is the number of subsets I of {1, 2, . . . , n} that 1. are of size k, 2. have j ≥ i as the largest element (so that we can factor out g j ),
3. include i (so that (25) holds in expectation by Corollary 1).
Due to the second condition, the form of m ij depends on whether i = j.
The diagonals m ii are zero if i < k since the largest element of any subset of size k is at least k. If i ≥ k then we fix i and are left with i -1 elements from which to choose k -1 which we can do
i-1 k-1 ways in line with (20).
The m ij for i = j are obtained by fixing i and j leaving j -2 elements 1, 2, . . . , i -1, . . . , i + 1, . . . , j -1 from which to choose k -2 which we can do j-2 k-2 ways in line with ( 21). Theorem 5. s 1 , s 2 , . . . , s n can be computed in total time O(k + n log n).
See Appendix A.4 for the proof and Line 36 of Listing 1 for an implementation based on it.
this section cite: []

Section: Variance Reduction

this section cite: []

Section: Leave-One-Out Baseline for the Simple Case
A popular variance reduction method [MRFM20, Owe13, GYZ + 25] for point-wise rewards g(x) subtracts the mean of the leave one out (LOO) rewards within each mini-batch x 1 , x 2 , . . . , x n :
g (loo) (x i ) = g(x i ) - 1 n -1 n j=1 j =i g(x j ).(26)
Since the subtracted part does not depend on x i , by Corollary 1 this retains unbiasedness.
this section cite: []

Section: Leave-One-Out Baseline for max g @k
Baselining the s i of ( 19) in this way introduces bias, however, as each s i depends on all x 1 , . . . , x n . We instead apply LOO to the following form of s i that follows from Theorem 4 and the proof thereof:
s i = 1 n k |I|=k i∈I I⊆{1,2,...,n} max j∈I g j (27
) ≡ S(i, k, {1, 2, . . . , n}),(28)
We can then retain unbiasedness by excluding i from the baseline, by defining
s (loo) i ≡ S(i, k, {1, 2, . . . , n}) - 1 n -1 n j=1 j =i S(j, k, {1, 2, . . . , n} \ i). (29
) Theorem 6. s (loo) 1 , s (loo) 2 , . . . , s (loo) n can be computed in total time O(k + n log n).
Proof. Given (5) it is sufficient to consider computing, for i = 1, 2, . . . , n, b
(k) i ≡ n j=1 j =i S(j, k, {1, 2, . . . , n} \ i).(30)
By assuming w.l.o.g. an ascending ordering of g(x i ), excluding the first index does not change the ordering of the remaining indices. The first term is therefore
2 b (k) 1 = N i=2 s i = 1 n-1 k N i=2 m ii + m i-1,i (i -2) g(x i ),(31)
where ( 31) follows from (49). From (49) we obtain for
1 ≤ i < n the left to right recursion b (k) i+1 = b (k) i + 1 n-1 k g(x i ) -g(x i+1 ) m ii + m i-1,i (i -2) .(32)
Similar arguments to the proof of Theorem 5 therefore imply the same time complexity.
Line 52 of Listing 1 implements s (loo) i using the recursion in the above proof.
4.3 max g @(k -1) Leave-One-Out Baseline for max g @k
The baseline b
i is an average of the max g @k estimates over sets of size k. For a number of samples n equal to k, there are no such subsets to construct the baseline. [TZSM25] recently overcame this issue for the specific case n = k by using max g @(k -1) as the baseline statistic. We generalize their approach to k < n and to averaging over all subsets by defining similarly to Equation ( 27)
s (loo-1) i = 1 n k |I|=k i∈I I⊆{1,2,...,n} max j∈I g j -max b∈I\i g b .(33)
Averaging smaller but more numerous subsets in the baseline reduces variance but introduces bias (in the baseline, not s Proof. By the linearity of the expectation we can split the two terms in the parentheses of Equation (33) into two separate sums. The first summation is by definition simply s i of Equation ( 19). The (negation of the) second summation can be computed efficiently using
1 n k |J |=k J ⊆{1,2,...,n} max b∈J \i g b = 1 n k |B|=k-1 B⊆{1,2,...,n}\i max b∈B g b = k n(k -1) b (k-1) i ,(34)
where the final equality follows with a little algebra from Equation ( 28) and Equation ( 30). Listing 1 implements s
(loo-1) i using (34); Figure 5 compares s i , s (loo) i , and s (loo-1) i .
5 Experiments
this section cite: ['b30']

Section: One-Dimensional Toy Example
We start with a policy that is Gaussian with a fixed standard deviation and mean parameter θ we wish to learn, so that x ∼ N (θ, 0.1). We set the raw reward to be
g(x) = x 2 0 ≤ x ≤ 1 0 otherwise.(35)
The optimal policy under the max g @k reward varies with k (see Figure 1). The variance of our estimators is compared in Figure 4 where s (loo-1) is the strongest. For GEMMA2-2B we use a v5litepod-128 [Goo] which needs around 4 hours per 1000 training steps. Each RL training run [SWD + 17] involves sampling a fixed n number of completions {x i } n i=1 for a given prompt at a given training step. For our experiments, we set n = 16. The rewards are computed for every completion using a reward function g(•). We transform these rewards {g(x i )} n i=1 using our unbiased estimator s (loo-1) of ( 33), which we favour due to Figure 4, and which we refer to as PKPO. We repeat the training for a selection of k opt , thus optimizing a different pass@k opt each time. Since k opt = 1 leads to no reward transformation, this is our baseline (al- 12.00 ± 04.33 02.00 ± 01.69 08.18 ± 04.00 k opt=4 82.33 ± 04.14 22.00 ± 02.00 38.18 ± 04.67 k opt=8
84.14 ± 04.67 26.67 ± 02.50 44.50 ± 04.33 [TZSM25] 22.00 ± 04.44 06.00 ± 02.67 10.16 ± 04.57 EntropyReg 24.67 ± 04.50 04.00 ± 02.33 08.89 ± 04.89 22.00 ± 04.18 03.33 ± 02.00 08.00 ± 04.50 k opt=4 87.17 ± 04.14 24.33 ± 02.33 42.00 ± 04.16 k opt=8
88.89 ± 04.33 29.67 ± 02.67 43.13 ± 04.67 [TZSM25] 36.00 ± 02.50 08.00 ± 04.00 18.00 ± 04.89 EntropyReg 28.00 ± 04.44 08.00 ± 02.50 14.67 ± 04.44 though we use basic LOO mean centering of Equation ( 26), without which the training diverges).
For each run, we measure pass@k eval for every k eval ∈ {1, 2, 4, 8, 12, 16} at each step. Additionally, we also track model entropy and cumulative solve rate during training. The latter is defined as the fraction of tasks from the task-set for which the model has sampled a correct solution at least once; this is a critical metric that reflects the success of the model's exploration and measures its ability to find novel solutions.
this section cite: ['b30', 'b30']

Section: Entropy regularization baseline
In addition to our PKPO and the special case thereof of [TZSM25], we also add the entropy regularization baseline, which is PPO with an additional entropy term in the objective. We give this baseline an arguably unfair advantage by performing a small sweep over the values 0.001, 0.005, 0.01, 0.05, 0.1 for the entropy coefficient for each (model, benchmark) pair and only report the best result as EntropyReg.
5.2.1 Choosing k opt selectively optimizes pass@k eval and solves more tasks
We use the training split of Hendrycks MATH [HBK + 21] which contains 12,000 problems as our task set. Figure 6a shows that a higher k opt in our transformation leads to a consistently higher cumulative solve rate throughout training, as well as a higher entropy. By optimizing pass@k instead of pass@1, the model appears to better utilize the exploration budget thus finding more solutions.
In Figure 7, we compare pass@k eval across our runs (k opt ∈ {1, 4, 8}) for various k eval . We find the best pass@k eval when k opt = k eval (or k opt is closest to k eval among available k opt ). Nontransformed rewards optimize pass@1, leading to sub-optimal pass@k eval for k eval = 1, and the deficit worsens as k eval increases. Thus, our experiments also demonstrate that setting k opt := k eval in our transformation suffices to optimize pass@k eval for a k eval ≤ n. This generalizes the already powerful result of [TZSM25] by alleviating the coupling that restricts to optimizing either pass@n or pass@1. In other words, since RL training of LLMs typically samples a large batch (n 1), failing to use our transformation results in sub-optimal pass@k performance, especially for modest values of k.
As k opt -→ n, the variance of our estimator increases as there are fewer subsets in (33) (see Figure 4). We presume this is why 1) gains of k opt = 8 over k opt = 4 are more prominent when k eval ∈ {12, 16} than when k eval = 8. That is, when k eval is further away from k opt ∈ {4, 8} than when it is closer, and 2) the special case n = k opt of [TZSM25] struggles to optimize the pass@n.
this section cite: ['b30', 'b30', 'b30']

Section: PKPO robustly improves pass@k on held out evaluations
Tables 1-4 above  in the appendix) present performance on held-out sets for two tasks. We report the mean and standard error based on three runs with different random seeds. For math, we train on the train split and evaluate on the test split of Hendrycks MATH [HBK + 21]. To evaluate coding, we use MBPP [AON + 21] for training and evaluate on HUMANEVAL [CTJ + 21b].
MBPP has multiple unit tests per problem and hence we use this not only as a proxy for additional benchmarks but also to showcase our handling of a continuous reward function (% unit tests passed).
this section cite: []

Section: Improving pass@k without sacrificing pass@1
Figure 3 demonstrates that as PKPO can use any arbitrary k opt ≤ n, this allows varying k opt over the course of training to good effect. We show a simple annealing procedure which starts training with a high k opt = 8 and reduces it to k opt = 1 after 1500 steps. This trains the model to initially prioritize exploration (optimize pass@k) and then consolidate the single-sample policy (optimize pass@1). This switch is apparent in Figure 3a, at step 1500 where the slope of k annealed changes. While traditional methods like [TZSM25] suffer from a trade-off between pass@k and pass@1, we get a final model which has higher pass@k eval for all k eval > 1 with no sacrifice in pass@1.
this section cite: ['b30']

Section: PKPO is essential for learning on hard problems
Figure 8 shows the limitation of traditional pass@1 optimization through RL on an especially challenging task-set. We use the easy subset of ARC-AGI-1 [CKKL25]. We observe that conventional pass@1 optimization stalls. However, our pass@k approach unblocks learning, and results in higher pass@k eval across all k eval including k eval = 1. Furthermore, we see higher k opt leads to more effective and faster learning. This is likely because the benefits of prioritizing joint utility over individual sample utility are more prominent on a harder task-set.
Tables 5 and 6 show more extensive experiments on ARC-AGI-1. We make an 80:20 train:test split of the same easy subset as before and report the cumulative solve rate on the train set and pass@k rate on the test set. We train to saturation (no change in cumulative rate for 1k steps), and again use three random restarts to provide standard errors. By encouraging exploration in a direct and stable manner, our method unblocks learning unlike other methods. Entropy Regularization does indeed sacrifice pass@1 and slightly improves pass@k by promoting exploration, but it is hard to tune, and is significantly outperformed by our method. Moreover, it has no explicit way to optimize for a specific k eval. [TZSM25] targets the same objective as PKPO, but couples the minibatch size to k and thereby incurs higher variance than PKPO with k < n.
this section cite: ['b6', 'b30']

Section: Conclusions and Outlook
In RL training with multiple independent samples per task, optimizing the pass@k maximizes the expectation of the best reward in the set of samples, rather than the average one. This preserves model output diversity, which leads to solving more problems and ultimately yields stronger policies. We provide drop-in replacements for more traditional RL reward transformations that robustly and efficiently optimize the pass@k. This work can be extended in various ways, such as to other inference-time search algorithms, and to more sophisticated baseline techniques.
ing with retrieval-augmented language models. Advances in Neural Information Processing Systems, 36:21573-21612, 2023.
[YTC + 23] Zishun Yu, Yunzhe Tao, Liyu Chen, Tao Sun, and Hongxia Yang. B-coder: Value-based deep reinforcement learning for program synthesis. arXiv preprint arXiv:2310.03173, 2023.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: A general language assistant as a laboratory for alignment Year: (2021)
Ref_id:b2 Title: Program synthesis with large language models Year: (2021)
Ref_id:b3 Title: AI achieves silver-medal standard solving international mathematical olympiad problems Year: ()
Ref_id:b4 Title:  Year: (2024)
Ref_id:b5 Title: Chirag Nagpal, Flavien Prost, Aradhana Sinha, et al. Infalign: Inference-aware language model alignment Year: (2024)
Ref_id:b6 Title: Arc prize Year: (2024)
Ref_id:b7 Title: Deep reinforcement learning from human preferences Year: (2017)
Ref_id:b8 Title: Inference-aware fine-tuning for Best-of-N sampling in large language models Year: (2021)
Ref_id:b9 Title: Goldmedalist performance in solving olympiad geometry with alphageometry2 Year: (2025)
Ref_id:b10 Title: Differentiable ranking and sorting using optimal transport Year: (2019)
Ref_id:b11 Title: Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy Year: (2024)
Ref_id:b12 Title: Deepseek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning Year: (2025)
Ref_id:b13 Title: Rlef: Grounding code llms in execution feedback with reinforcement learning Year: (2024)
Ref_id:b14 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b15 Title: A class of statistics with asymptotically normal distribution Year: (1948)
Ref_id:b16 Title: V-star: Training verifiers for self-taught reasoners Year: (2024)
Ref_id:b17 Title: Unbiased estimates. Izvestiya Akademii Nauk SSSR Year: (1950)
Ref_id:b18 Title: Training language models to self-correct via reinforcement learning Year: (2022)
Ref_id:b19 Title: Coderl: Mastering code generation through pretrained models and deep reinforcement learning Year: (1998)
Ref_id:b20 Title: Monte carlo gradient estimation in machine learning Year: (2020)
Ref_id:b21 Title: Is self-repair a silver bullet for code generation? arXiv preprint Year: (2023)
Ref_id:b22 Title: Monte Carlo theory, methods and examples Year: (2013)
Ref_id:b23 Title: Marvlqa: A benchmark for mathematical reasoning over visual landscapes Year: (2025)
Ref_id:b24 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b25 Title: BOND: aligning LLMs with Best-of-N distillation Year: (2024)
Ref_id:b26 Title: Executionbased code generation using deep reinforcement learning Year: (2023)
Ref_id:b27 Title: Proximal policy optimization algorithms Year: (2017)
Ref_id:b28 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b29 Title:  Year: (2024)
Ref_id:b30 Title: Optimizing language models for inference time objectives using reinforcement learning Year: (2025)
Ref_id:b31 Title: Finetuned language models are zero-shot learners Year: (2021)
Ref_id:b32 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b33 Title: Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models Year: (2024)
Ref_id:b34 Title: Differentiable top-k with optimal transport Year: (2020)
Ref_id:b35 Title:  Year: ()
