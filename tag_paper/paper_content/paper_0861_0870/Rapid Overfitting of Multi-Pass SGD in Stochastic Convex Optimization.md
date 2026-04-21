Title: Rapid Overfitting of Multi-Pass SGD in Stochastic Convex Optimization
Abstract: We study the out-of-sample performance of multipass stochastic gradient descent (SGD) in the fundamental stochastic convex optimization (SCO) model. While one-pass SGD is known to achieve an optimal Θ(1/ √ 𝑛) excess population loss given a sample of size 𝑛, much less is understood about the multi-pass version of the algorithm which is widely used in practice. Somewhat surprisingly, we show that in the general non-smooth case of SCO, just a few epochs of SGD can already hurt its out-of-sample performance significantly and lead to overfitting. In particular, using a step size 𝜂 = Θ(1/ √ 𝑛), which gives the optimal rate after one pass, can lead to population loss as large as Ω(1) after just one additional pass. More generally, we show that the population loss from the second pass onward is of the order Θ(1/(𝜂𝑇) +𝜂 √ 𝑇), where 𝑇 is the total number of steps. These results reveal a certain phase-transition in the outof-sample behavior of SGD after the first epoch, as well as a sharp separation between the rates of overfitting in the smooth and non-smooth cases of SCO. Additionally, we extend our results to withreplacement SGD, proving that the same asymptotic bounds hold after 𝑂 (𝑛 log 𝑛) steps. Finally, we also prove a lower bound of Ω(𝜂 √ 𝑛) on the generalization gap of one-pass SGD in dimension 𝑑 = 𝑂 (𝑛), improving on recent results of Koren et al. (2022) and Schliserman et al. (2024).

Section: Introduction
Stochastic gradient descent (SGD) is one of the most fundamental algorithms in machine learning and optimization, widely used for training large-scale models. Theoretical analysis of SGD has traditionally focused on its behavior in the one-pass (single-epoch) setting, particularly in the context of convex optimization, where it is well understood that SGD achieves an optimal excess population loss of order Θ(1/ √ 𝑛) when trained on a sample of size 𝑛. This rate is known to be minimax optimal under standard assumptions (Nemirovskiȋ & Yudin, 1983).
However, in practice, it is common to run multiple passes over the training data, a scheme often referred to as multipass SGD, where, in each pass, training examples are sampled without-replacement, with or without reshuffling between passes. Despite its empirical success, the theoretical implications of performing multiple passes are far less understood, particularly with respect to generalization and outof-sample performance. The standard intuition, based on empirical observations, suggests that more training should lead to better empirical performance, yet, more steps and passes might eventually lead to overfitting as the model becomes too closely tailored to the training data being processed repeatedly. Indeed, a substantial body of work has studied the convergence of multi-pass SGD in finite-sum problems (e.g., Recht & Re, 2012;Hardt et al., 2016;Nagaraj et al., 2019;Rajput et al., 2020;Safran & Shamir, 2020;2021;Koren et al., 2022;Cha et al., 2023). However, these studies primarily focus on optimization convergence in terms of empirical risk and how it is influenced by without-replacement sampling, as opposed to with-replacement which is significantly easier to analyze theoretically. Far less attention has been given to convergence in terms of population risk, which quantifies out-of-sample performance and is arguably the true goal in the context of machine learning.
In fact, to the best of our knowledge, the following basic question still remains, rather surprisingly, unanswered:
How does the population risk performance of SGD, tuned to attain the minimax optimal rate after a single pass, deteriorate (if at all) after making just a few additional passes?
While there are several existing upper bounds on the population risk in the multi-pass scenario (e.g., Hardt et al., 2016;Bassily et al., 2020), none of them give a meaningful, nontrivial answer to this question in the general stochas-tic convex setting in which the Θ(1/ √ 𝑛) minimax rate of Nemirovskiȋ & Yudin (1983) applies.
Our investigation in this work reveals a rather surprising answer to this basic question: we show that even in the fundamental, well-studied convex setting, multiple passes of SGD can lead to rapid overfitting and reach a trivially-large population loss after just one additional pass. In particular, using the canonical stepsize of 𝜂 = Θ(1/ √ 𝑛) that leads to the minimax rate after a single pass, results in population excess risk as high as Ω(1) after merely two passes. More generally, we establish tight population excess risk bounds for multi-pass SGD using any stepsize 𝜂 and number of steps 𝑇. Our results cover without-replacement, single-shuffle and multi-shuffle, multi-pass SGD as well as with-replacement SGD, demonstrating similar rapid overfitting in all cases. An illustration of the population risk bounds we obtain across different epochs and stepsizes can be seen in Figure 1.
this section cite: ['b13', 'b16', 'b6', 'b12', 'b15', 'b17', 'b7', 'b3', 'b6', 'b1', 'b13']

Section: Our Contributions
In some more detail, we examine the out-of-sample behavior of multi-pass SGD in the fundamental setting of Stochastic Convex Optimization (SCO). In this setting, we operate with a convex and Lipschitz (yet not necessarily smooth) loss function over a bounded convex domain.
In this setting, we make the following contributions:
• We establish a tight bound of Θ(1/(𝜂𝑇) + 𝜂 √ 𝑇) on the population excess risk of multi-pass (without replacement) SGD with stepsize 𝜂 over 𝑇 steps (see Theorems 3.1 and 3.3 in Section 3). This result applies to both the single-shuffle and multi-shuffle variants, but also more generally to any multi-pass scheme that processes examples according to an arbitrary sequence of permutations.
• We also prove a similar tight Θ(1/(𝜂𝑇) +𝜂 √ 𝑇) population excess risk bound for with-replacement SGD, that holds after 𝑂 (𝑛 log 𝑛) steps (see Theorems 3.2 and 3.4).
• Finally, we also provide a new Ω(𝜂 √ 𝑛) lower bound on the empirical risk (and therefore also on the generalization gap) of single-pass SGD. Our result holds in dimension 𝑂 (𝑛) thus improving upon previous results by Koren et al. (2022); Schliserman et al. (2024) that were established in dimension at least quadratic in 𝑛.
We note that all of our lower bound constructions apply in an overparameterized regime, where the dimension scales linearly in the sample size 𝑛. Our approach builds on recent techniques developed for analyzing the sample complexity of (approximate) empirical risk minimization in stochastic convex optimization (Feldman, 2016;Amir et al., 2021;Koren et al., 2022;Schliserman et al., 2024;Livni, 2024). In particular, we leverage methods from Livni (2024), who refined these constructions to achieve minimal dimension dependence of 𝑂 (𝑛).
this section cite: ['b7', 'b19', 'b5', 'b0', 'b7', 'b19', 'b11', 'b11']

Section: Discussion and Open Questions
It is insightful to contrast our lower bounds for multi-pass SGD with existing lower bounds for gradient methods in SCO. Most prior work on population risk lower bounds has focused on algorithms that approximately minimize the empirical risk, such as full-batch gradient descent (Amir et al., 2021;Schliserman et al., 2024;Livni, 2024). A key observation in these works is that after just a single step of gradient descent, the entire training set has been "touched" and so can be effectively memorized in the optimization iterate. Once this occurs, subsequent gradient steps can be steered toward an overfitting solution with respect to the particular training set at hand. Our results in the present paper reveal a similar memorization effect for multi-pass SGD: after a single full pass, SGD is also capable of effectively memorizing the training set and driving the optimization towards overfitting. However, a crucial (and quite remarkable) difference is that this effect cannot be manifested before completing the first pass, as doing so would contradict the classical Nemirovskiȋ & Yudin (1983) stochastic approximation upper bounds.
Our result regarding the empirical risk of one-pass SGD complement earlier work by Koren et al. (2022); Schliserman et al. (2024) and challenges the classic learning paradigm of minimizing empirical risk and generalization gap in relation to SGD. It reveals that the generalization gap is inadequate in explaining the minimax optimal generalization behavior of one-pass SGD, which constitute one of the fundamental cornerstones of convex optimization. Together with our lower bounds for multi-pass SGD, this result highlights a sharp phase transition between the first and later epochs: generalization bounds, such as those implied by algorithmic stability (Bousquet & Elisseeff, 2002;Hardt et al., 2016;Bassily et al., 2020), are ineffective during the first pass but become tight after the second pass, whereas the optimal performance in the first pass is only explained by online-to-batch (aka stochastic approximation) arguments which collapse immediately after the first pass.
Open questions. Our findings raise several intriguing open questions for further investigation:
• First, our construction leverages techniques for lower bounding the generalization gap in SCO in order to control the population risk of multi-pass SGD. However, in principle, the population risk performance of multi-pass SGD should be unrelated to its generalization gap (e.g., as in the case of one-pass SGD). In particular, it is an interesting question whether our results could be reproduced in settings where uniform convergence holds (e.g., low-norm linear predictors with a Lipschitz loss), and the generalization gap is necessarily small.
• Another interesting problem is to precisely characterize the rate at which overfitting develops during the second pass. Indeed, our results are only effective starting from the end of the second epoch and onward. As we discussed earlier, generalization bounds remain vacuous at the start of the second pass, and regret analysis fails due to its reliance on independent samples at each iteration. Thus, it appears that more refined techniques will be required to analyze SGD dynamics immediately after the first pass.
• Finally, our work focuses on the general SCO setting, which allows for non-smooth convex loss functions. Establishing population risk bound for multi-pass projected SGD in the analogous smooth setting remains an interesting open question that is likely to require significantly different techniques than those used in the non-smooth case. We remark that for unconstrained optimization, it is known that variants of smooth GD and SGD can overfit, even in dimension one (Nikolakakis et al., 2023;Zhang et al., 2023). However, these results rely on unbounded losses over unbounded domains, where learnability itself is not guaranteed. 1 In contrast, our focus is on constrained optimization-a regime where learning is possible yet ERM-based methods can fail-and the open question pertains to this case as well.
this section cite: ['b0', 'b19', 'b11', 'b13', 'b7', 'b19', 'b2', 'b6', 'b1', 'b14', 'b23']

Section: Additional Related Work
Convergence bounds for multi-pass SGD. Numerous studies have explored multi-pass SGD from an optimization perspective (e.g., Rajput et al., 2020;Safran & Shamir, 2020;Cha et al., 2023). Notably, focusing on empirical performance in finite-sum problems, Nagaraj et al. (2019) derived upper bounds for smooth functions, and Koren et al. (2022) extended these results to non-smooth func-1 Indeed, with an unbounded range of function values, even evaluating the loss of a single model, let alone learning, becomes intractable. (E.g., without additional assumptions, standard concentration bounds scale linearly with the range of the random variables.) tions. Some works specifically study the differences in optimization convergence rates between with-replacement and without-replacement sampling (e.g., Recht & Re, 2012;Yun et al., 2021;Lai & Lim, 2020;De Sa, 2020;Safran & Shamir, 2021). Other works examine the population performance of multi-pass SGD. For example, Sekhari et al. (2021) focused on establishing upper bounds for a variant of multi-pass SGD that uses a validation set, and in a slightly non-standard formulation of SCO that allows for non-convex individual functions. In contrast, our main focus is on lower bounds for standard multi-pass SGD in the classical SCO model (Shalev-Shwartz et al., 2010). Lei et al. (2021) derive upper bounds using decreasing step sizes, without assuming Lipschitz continuity but requiring gradient norms to be bounded linearly by function values. In contrast, we establish tight bounds for fixed stepsize SGD under the standard Lipschitz assumption. Other upper bounds on population loss for the closely related with-replacement SGD can be found in Hardt et al. (2016); Lei & Ying (2020).
Lower bounds for SGD. In terms of lower bounds, Zhang et al. (2023) and Nikolakakis et al. (2023) analyzed SGD with replacement and multi-pass SGD, respectively, but both focused specifically on the unconstrained (smooth) case and allowed for unbounded function values, which in general, render the problem unlearnable, as discussed before. We, on the other hand, focus exclusively on bounded domains and functions values bounded by a constant, where the classical Nemirovskiȋ & Yudin (1983) results apply. More recently, Koren et al. (2022) established lower bounds for the generalization gap of one-pass SGD. They demonstrate that even though SGD achieves optimal population loss rates after a single pass, its empirical risk can be as high as Ω(𝜂 √ 𝑇) after 𝑇 steps. Their construction was in dimension exponential in the sample size, which was later improved to quadratic by Schliserman et al. (2024). Here, we further improve the dimensionality dependence to linear, which, as argued by Feldman (2016); Livni (2024), is optimal.
this section cite: ['b15', 'b17', 'b3', 'b12', 'b7', 'b16', 'b22', 'b8', 'b4', 'b18', 'b20', 'b21', 'b10', 'b6', 'b9', 'b23', 'b14', 'b13', 'b7', 'b19', 'b5', 'b11']

Section: Stochastic Convex Optimization (SCO).
Our work belongs to the study of sample complexity of learning algorithms in the fundamental SCO model. Previous research has shown that in SCO some algorithms overfit when the sample size depends on the dimension (Shalev-Shwartz et al., 2010;Feldman, 2016), while others like online-tobatch algorithms achieve dimension independent rates (Nemirovskiȋ & Yudin, 1983). This makes SCO a valuable model for testing the generalization and sample complexity of learning algorithms. Amir et al. (2021) were the first to establish a dimension dependent lower bound for the sample complexity of gradient descent (GD), they showed a population loss lower bound of Ω(𝜂 √ 𝑇) after 𝑇 steps using a step size 𝜂 in a construction with dimension exponential in the sample size 𝑛. Later, Schliserman et al. (2024) extended this result to functions with dimensions proportional to the square of the sample size, and Livni (2024) further reduced it to linear size. This shows that to prevent GD from overfitting, a sample of size Ω(𝑑) must be considered. While we build on ideas from these works, our focus is on SGDrather than GD-which does generalize in its first epoch. Specifically, multi-pass SGD circumvents previous hardness constructions during its first pass and begins the second epoch with optimal generalization performance. This behavior requires a careful adaptation of existing constructions to account for the rapid deterioration of generalization after the first epoch.
Algorithmic stability. Our work is also related to the study of stability of learning algorithms as means to establish generalization bounds (Shalev-Shwartz et al., 2010;Bousquet & Elisseeff, 2002). The crucial advantage of stability in the multi-epoch (without replacement) setting is the fact it does not require individual examples to be sampled i.i.d.-an assumption that only holds during the first pass. Notably, Hardt et al. (2016) show bounds on uniform stability of multi-pass SGD for smooth functions and Bassily et al. (2020) show tight stability bounds for non-smooth functions. Their results demonstrate that in the non-smooth case, an additional Θ(𝜂 √ 𝑇) term must be added, that make such bounds vacuous (at least for the purpose of establishing risk bounds) unless the stepsize is extremely small as a function of 𝑇. Importantly, these bounds apply uniformly across epochs and do not explain why SGD generalizes well in its first pass and overfitting starts only from the second pass. This suggests a gap between instability and generalization behavior, which calls for a more refined analysis that goes beyond plain stability arguments.
this section cite: ['b21', 'b5', 'b13', 'b0', 'b19', 'b21', 'b2', 'b6', 'b1']

Section: Problem Setup Stochastic Convex Optimization (SCO).
We study outof-sample performance of multi-pass SGD in the context of the fundamental stochastic convex optimization (SCO) framework. A learning problem in SCO is defined by a convex and 𝐺-Lipschitz loss function 𝑓 : 𝑊 × 𝑍 → ℝ, where 𝑍 is a sample space endowed with a population distribution Z, and 𝑊 is a convex, compact domain with diameter bounded by 𝐷. A learning algorithm receives a training set 𝑆 = {𝑧 1 , . . . , 𝑧 𝑛 } of 𝑛 i.i.d. samples from Z and outputs a model 𝑤 ∈ 𝑊, with the goal of minimizing the population loss:
𝐹 (𝑤) = 𝔼 𝑧∼Z [ 𝑓 (𝑤, 𝑧)].
Let 𝑤 ★ ∈ arg min 𝑤 ∈𝑊 𝐹 (𝑤) denote a minimizer of the objective. The empirical loss on the sample 𝑆 is given by:
𝐹 𝑆 (𝑤) = 1 𝑛 𝑛 ∑︁ 𝑖=1 𝑓 (𝑤, 𝑧 𝑖 ),
and we denote by 𝑤 ★ 𝑆 ∈ arg min 𝑤 ∈𝑊 𝐹 𝑆 (𝑤) its minimizer.
One-pass Stochastic Gradient Descent (SGD). One-Pass SGD receives a training set 𝑆 = {𝑧 1 , . . . , 𝑧 𝑛 }, a step size 𝜂 > 0, a suffix averaging parameter 𝜏 ∈ [𝑛 + 1] and a first-order oracle 𝑂 𝑧 (𝑤) ∈ 𝜕 𝑓 (𝑤, 𝑧) and proceeds as:
Initialize:
𝑤 0 = 0 For 𝑡 = 1, . . . , 𝑛 : 𝑤 𝑡+1 = Π 𝑊 𝑤 𝑡 -𝜂𝑂 𝑧 𝑡 (𝑤 𝑡 ) Output: 𝑤 𝑛, 𝜏 = 1 𝜏 𝑛 ∑︁ 𝑡=𝑛-𝜏+1 𝑤 𝑡 ,
where Π 𝑊 : ℝ 𝑑 → 𝑊 denotes the projections onto 𝑊. Note that we consider general 𝜏-suffix averaging as an output.
Multi-pass SGD. In contrast to single-pass SGD, multipass SGD does several passes on permutations of the same training set. It receives a training set
𝑆 = {𝑧 1 , . . . , 𝑧 𝑛 }, a number of epochs 𝐾, a step size 𝜂 > 0, 𝐾 permutations {𝜋 𝑘 : [𝑛] → [𝑛]} 𝑘 ∈ [𝐾 ]
, a suffix averaging parameter 𝜏 ∈ [𝑛𝐾 + 1] and a first-order oracle 𝑂 𝑧 (𝑤) ∈ 𝜕 𝑓 (𝑤, 𝑧), and proceeds as follows:
Initialize:
𝑤 0 = 0 For 𝑡 = 1, . . . , 𝑛𝐾 : 𝑖 𝑡 = 𝜋 ⌊𝑡/𝑛⌋ (𝑡 mod 𝑛) 𝑤 𝑡+1 = Π 𝑊 𝑤 𝑡 -𝜂𝑂 𝑧 𝑖 𝑡 (𝑤 𝑡 ) Output: 𝑤 𝑇 , 𝜏 = 1 𝜏 𝑇 ∑︁ 𝑡=𝑇 -𝜏+1 𝑤 𝑡 ,
where 𝑇 = 𝑛𝐾 is the total number of steps. We note two important special cases: the single-shuffle and multi-shuffle.
In the single-shuffle case, 𝜋 1 is chosen uniformly at random, and 𝜋 1 = 𝜋 𝑘 for all 𝑘 ∈ [𝐾]; in the multi-shuffle case, the {𝜋 𝑘 } 𝑘 ∈𝐾 are chosen uniformly and independently.
With-replacement SGD. With replacement SGD receives a training set 𝑆 = {𝑧 1 , . . . , 𝑧 𝑛 }, a number of steps 𝑇, a step size 𝜂 > 0, a suffix averaging parameter 𝜏 ∈ [𝑛 + 1] and a first-order oracle 𝑂 𝑧 (𝑤) ∈ 𝜕 𝑓 (𝑤, 𝑧). It then proceeds as follows:
Initialize:
𝑤 0 = 0 For 𝑡 = 1, . . . , 𝑇 : 𝑖 𝑡 ∼ Unif( [𝑛]) 𝑤 𝑡+1 = Π 𝑊 𝑤 𝑡 -𝜂𝑂 𝑧 𝑖 𝑡 (𝑤 𝑡 ) Output: 𝑤 𝑛, 𝜏 = 1 𝜏 𝑛 ∑︁ 𝑡=𝑛-𝜏+1 𝑤 𝑡 .
Note that this is not a multi-pass algorithm per-se, as the stochastic gradients used are independent regardless of the total number of steps 𝑇.
this section cite: []

Section: Population Risk Bounds
We present a lower bound and a matching upper bound for the population loss of multi-pass SGD for 𝐾 ≤ 𝑛 epochs.
this section cite: []

Section: Lower Bound for Multi-Pass SGD
We begin with stating the lower bound, which constitutes our main contribution: Theorem 3.1. For every 𝑛 ≥ 24, 2 ≤ 𝐾 ≤ 𝑛 2 , 𝑇 = 𝑛𝐾, 𝑑 = 256𝑛, 𝜂 > 0 and 𝜏 ∈ [𝑇 + 1], let 𝑊 = {𝑤 ∈ ℝ 2𝑑+1 : ∥𝑤∥ 2 ≤ 1} then there are a finite sample space 𝑍, a distribution Z over 𝑍, a 4-Lipschitz convex function 𝑓 (𝑤, 𝑧) in ℝ 2𝑑+1 such that for any first order oracle of 𝑓 (𝑤, 𝑧) with probability 1 2 if we run without replacement SGD for 𝑇 steps with stepsize 𝜂 and with any sequence of permutations:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (𝑤 ★ ) = Ω min 𝜂 √ 𝑇 + 1 𝜂𝑇 , 1 .
We note that this result applies to single, multi shuffle and more generally for any sequence of permutations. In particular, for 𝜂 = Θ(1/ √ 𝑛) that yields minimax optimal result for one-pass, can lead up to Ω(1) population loss after just one additional pass.
We provide here a proof sketch, the full proof is presented in Section 5.
Proof sketch. To obtain the lower bound, we rely on the, recently introduced, notion of a sample-dependent oracle introduced by Livni (2024), and the reduction from sampledependent oracles to the standard setting of stochastic convex optimization. Livni (2024) observed that if the gradients of an algorithm such as Gradient Descent (GD) are dependent on the whole sample, then one can utilize that to construct stochastic functions for which Gradient Descent overfits. It also turns out that one can reduce the standard setup of stochastic optimization to this weaker setting of a sample dependent oracle, for GD. These observations allowed the construction of distributions that cause GD to overfit at certain regimes. SGD, though, differ from GD that observes the whole sample at the first iteration. In particular, it is impossible for the trajectory of SGD to depend on future seen examples. In turn, the reduction becomes more subtle. We note that this is not just an artifact of the proof technique, but is demonstrated by the fact that, indeed, SGD does minimize the population loss at the first epoch, and SGD circumvent the hard construction of Livni (2024). Nevertheless, we build on a similar approach, and we use a weaker sample dependent first order oracle, that cannot depend on the whole sample but may depend on past observations. As it turns out, such an oracle is enough to construct lower bounds, and also allows a reduction for the case of SGD.
Following Livni (2024), we rely on a loss function with the following structure: 𝑓 (𝑤, 𝑉) = 𝑔(𝑤, 𝑉) + 𝛼ℎ(𝑤), where 𝑔(𝑤, 𝑉) is a variant of Feldman's function (Feldman, 2016) that has the property that it has spurious empirical risk minimizers. Namely, using this function and a specific distribution, we can guarantee the high-probability existence of a "bad" vector, which for this bound will have high population loss. The second term, ℎ(𝑤), is used to guide the iterates toward this bad vector, enabling us to achieve the lower bound. The constant 𝛼 prevents projections from disrupting this process.
Feldman's function and existence of a spurious empirical minimizer. Feldman's function, 𝑔, relies on the existence of a set 𝑈 ⊂ ℝ 𝑑 with |𝑈| ≥ 2 Ω(𝑑) such that for every 𝑢 ≠ 𝑣 ∈ 𝑈, we have
⟨𝑢, 𝑣⟩ ≤ 𝑐 1 < 𝑐 2 ≤ ∥𝑣∥ 2 .
We consider the following function:
𝑔 1 (𝑤, 𝑉) = max 𝑣 ∈𝑉 {𝑐 1 , 𝑣 • 𝑤} .
The existence of such 𝑈 follows from a standard packing argument. Next, we let 𝑍 = 𝑃(𝑈), the powerset of 𝑈, and treat examples in 𝑍 as subsets of 𝑈. Let Z be a distribution over 𝑍 such that for 𝑉 ∼ Z every 𝑢 ∈ 𝑈 is in 𝑉 with probability 𝛿.
Notice that if 𝑤 ∈ 𝑉, then 𝑔 1 (𝑤, 𝑉) ≥ 𝑐 2 , and otherwise, 𝑔 1 (𝑤, 𝑉) = 𝑐 1 . Let 𝑆 = {𝑉 1 , . . . , 𝑉 𝑛 }. For this bound we will define a bad vector as an ERM with high population loss. Notice that any vector
𝑢 0 ∉ ∪ 𝑛 𝑖=1 𝑉 𝑖 is an ERM since 𝐹 𝑆 (𝑢 0 ) -𝐹 𝑆 (𝑤 ★ 𝑆 ) = 𝑐 1 -𝑐 1 = 0.
If, furher, 𝑢 0 ∈ 𝑈, then its population loss is given by 𝐹 (𝑢 0 ) ≥ 𝛿(𝑐 2 -𝑐 1 ). By setting 𝛿 = 1 2 , we can show then that any such 𝑢 0 is a bad ERM. By further setting 𝑑 = Ω(𝑛), we can show that with high probability such a 𝑢 0 ∈ 𝑈 exists.
Reaching spurious empirical risk minimizer. Feldman's function demonstrated that there exists a "bad vector" in the sense that there are minimizers of the empirical risk that yield large population loss. However, there is no guarantee that if we run SGD on Feldman's function we will reach such a minimzer (in fact, notice that SGD applied on 𝑔 will remain at zero). Therefore, our strategy is to add a further term ℎ that will cause SGD to converge towards 𝑢 0 . For simplicity of the overview we assume here that 𝑢 0 ∈ {0, 1} 𝑑 .
For 𝑇 = 𝑂 (𝑑), we use the following function as ℎ:
ℎ 1 (𝑤) = max 𝑖 ∈ [𝑑 ] {0, -𝑤(𝑖)}.
Assume we ran SGD for 𝑛 iterations. Notice that 0 ∈ 𝜕ℎ 1 (0), therefore, in our setup we are allowed to return a 0 subgradient for 𝑛 iterations. After 𝑛 iterations, we start using ℎ after identifying a specific bad vector 𝑢 0 . Here, we exploit the fact that our oracle is sample dependent, and we use ℎ 1 to take one step on each of the positive coordinates of 𝑢 0 to reach 𝜂𝑢 0 in at most 𝑑 steps.
When 𝑇 ≫ 𝑑, ℎ 1 won't suffice since after 𝑑 such steps, we can't further increase the desired loss. To achieve the 𝑑 3 term, we use the following function instead:
ℎ 2 (𝑤) = max 𝑖 ∈ [𝑑 ] {0, -𝑤(𝑖), -(𝑤(𝑖) -𝑤(𝑖 + 1))} .
With the following oracle:
1. If there is an index 𝑖 such that 𝑤(𝑖) = 𝑤(𝑖 + 1) ≥ 𝜂, then output: 𝑒 𝑖+1 -𝑒 𝑖 .
2. If there is no such index and if 𝑖 is the minimal index such that 𝑤(𝑖) = 0, output: -𝑒 𝑖 .
3. If neither of the above conditions holds, output: 0.
By applying these gradient steps only for the positive indices of 𝑢 0 , this will allow us to achieve Ω(𝜂 √ 𝑇) when 𝑇 ≈ 𝑑 3 .
For the case when 𝑑 ≪ 𝑇 ≪ 𝑑 3 , we will use ℎ 2 , but instead of considering one index at a time, we consider blocks of indices of size 𝐵 ≥ 1 to achieve the same effect.
Putting it together. To compute and reach the bad vector we use the oracle's sample dependence and the fact that 0 ∈ 𝜕 𝑓 (0, 𝑉) for all 𝑉, allowing us to stay at 0. During the first epoch, we remain at 0 to observe the examples. At the start of the second epoch we have memorized the entire training set 𝑆, we compute 𝑢 0 ∉ ∪ 𝑛 𝑖=1 𝑉 𝑖 and take gradient steps to reach it using ℎ.
Reduction. The construction we described assumes a sample-dependent oracle. The key idea behind the reduction to this case is encoding information about past examples into the iterates. This allows the gradient to use knowledge of previous examples by examining only the current iterate and sample. □
this section cite: ['b11', 'b11', 'b11', 'b5']

Section: Lower Bound for With-Replacement SGD
Using the same technique, we also prove a lower bound for with-replacement SGD. The proof is given in Section 5 and here we give a short sketch of the proof.
Theorem 3.2. For every 𝑛 ≥ 24, 2 ≤ 𝐾 ≤ 𝑛, 𝑇 = 𝐾𝑛 log 𝑛, 𝑑 = 256𝑛, 𝜂 > 0 and 𝜏 ∈ [𝑇 + 1], let 𝑊 = {𝑤 ∈ ℝ 2𝑑+1 : ∥𝑤∥ ≤ 1} then there are a finite sample space 𝑍, a distribution Z over 𝑍, a 4-Lipschitz convex function 𝑓 (𝑤, 𝑧) in ℝ 2𝑑+1 such that for any first order oracle of 𝑓 (𝑤, 𝑧) with probability 1 4 if we run with-replacement SGD for 𝑇 steps with stepsize 𝜂:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (𝑤 ★ ) = Ω min 𝜂 √ 𝑇 + 1 𝜂𝑇 , 1 .
Proof sketch. Noticing that the key idea in the construction of Theorem 3.1 is to memorize the entire training set and then step in a "bad direction" uninterrupted, we can also apply the same approach here. In the case of with-replacement sampling, from folklore analysis of the coupon collector's problem we know that with probability 1 2 we have seen the entire training set after 𝑂 (𝑛 log 𝑛) steps, and from that point we can proceed similarly. □
this section cite: []

Section: Matching Upper Bounds
Finally, we give matching upper bounds to the two lower bounds presented above. Both bounds are straightforward consequences of standard techniques in convex optimization and algorithmic stability analysis; we only state the bounds here and defer proofs to Appendix A.
Theorem 3.3. Let 𝑓 be a convex, non-smooth, and 𝐺-Lipschitz function, and let 𝑆 be a dataset of 𝑛 samples. If we run multi-pass SGD (either multi-shuffle or single-shuffle) with step size 𝜂 > 0 for 𝐾 ≥ 1 epochs, for a total of 𝑇 = 𝑛𝐾 steps. Then, for any suffix average iterate 𝜏 = Ω(𝑇), we have the following guarantee:
𝔼 𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (𝑤 ★ ) = 𝑂 𝜂 √ 𝑇 + 1 𝜂𝑇 + 𝜂𝑇 𝑛 .
We will note that while the upper bound holds for the multishuffle and single-shuffle cases, our lower bound holds for every set of 𝐾 permutations. A similar result can be derived for with-replacement SGD:
Theorem 3.4. Let 𝑓 be a convex, non-smooth, and 𝐺-Lipschitz function, and let 𝑆 be a dataset of 𝑛 samples. If we run with-replacement SGD with step size 𝜂 > 0 for 𝐾 ≥ 1 epochs, for a total of 𝑇 = 𝑛𝐾 steps. Then, for any suffix average iterate 𝜏 = Ω(𝑇), we have the following guarantee:
𝔼 𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (𝑤 ★ ) = 𝑂 𝜂 √ 𝑇 + 1 𝜂𝑇 + 𝜂𝑇 𝑛 .
this section cite: []

Section: Empirical Risk Bounds
We further prove a lower bound of Ω(𝜂 √ 𝑇) for the empirical risk (equivalently, the generalization gap) of one-pass SGD using a construction in dimension 𝑑 = 𝑂 (𝑛), improving on results from Koren et al. (2022); Schliserman et al. (2024).
Theorem 4.1. For every 𝑛 ≥ 17, 𝑑 = 712𝑛 log 𝑛, 𝜂 > 0, and 𝜏 ∈ [𝑛 + 1] let 𝑊 = {𝑤 ∈ ℝ 2𝑑+1 : ∥𝑤∥ ≤ 1} then there are a datapoint set 𝑍 and a distribution Z over 𝑍, a 4-Lipschitz convex function 𝑓 (𝑤, 𝑧) in ℝ 2𝑑+1 such that for any first order oracle of 𝑓 (𝑤, 𝑧) with probability 1 2 if we run one-pass SGD with 𝜂 as a learning rate for 𝑛 steps then:
𝐹 𝑆 ( 𝑤 𝑛, 𝜏 ) -𝐹 𝑆 (𝑤 ★ 𝑆 ) = Ω min 𝜂 √ 𝑛, 1 .
The proof is deferred to Appendix C. This result strengthens the previous results that one-pass SGD cannot be fully explained by the classical learning framework of minimizing empirical loss and generalization gap. Specifically, while SGD ensures low population loss with 𝜂 = Θ(1/ √ 𝑛), it can still cause a generalization gap of up to Ω(1). Moreover, the construction is done in nearly linear dimension (up to a logarithmic factor), which is the lower bound for the dimension for such results, as uniform convergence must not hold.
Proof sketch. We use the same kind of construction as we did for the other proofs. Dinstinctively, though, to achieve high empirical risk, a "bad vector" is a vector that appears frequently in the training set, but is infrequent on the population. To guarantee the existence of such a vector we change the distribution Z so that 𝑢 0 ∈ 𝑉 with probability 𝛿 = 𝑂 (1/𝑛 2 ).
A similar argument (but reversed) as before demonstrates that there has to be a "bad vector" that is frequent at the first 1/16𝑛 of the examples, observed by the algorithm, but does not appear at the tail of the training set.
The appearance at a constant fraction of the training set ensures high empirical risk. On the other hand, absence from the tail allows uninterrupted advancement there in a similar technique as before. Our strategy then is similar to before, to compute the bad vector 𝑢 0 , we also stay at 0, this time for 𝑛/16 steps. Then if ∩ 𝑛/16 𝑖=1 𝑉 𝑖 ≠ ∅, we select the minimal vector 𝑢 0 from this set and take gradient steps to reach it using ℎ. □
this section cite: ['b7', 'b19']

Section: Proofs of Theorems 3.1 and 3.2
We first prove the results using a weaker notion of a sampledependent oracle and later relax this dependence. Formally, a sample-dependent oracle is a gradient oracle that, at each step, has access to both the current and all previous samples. We denote it as 𝑂 𝑺 , where 𝑺 = (𝑆 1 , . . . , 𝑆 𝑇 ) is the ordered sequence of sample sets. At iteration 𝑡, the algorithm receives 𝑆 𝑡 , which may be a set of samples rather than a single sample. This general setting captures a broad class of algorithms, including variants of SGD, as well as GD and batch methods. The oracle is defined as:
𝑂 𝑺 (𝑆 1:𝑡 -1 ; 𝑆 𝑡 , 𝑤 𝑡 ) = 1 |𝑆 𝑡 | ∑︁ 𝑧 ∈𝑆 𝑡 𝑂 𝑧 (𝑆 1:𝑡 -1 ; 𝑤 𝑡 ),
when 𝑆 1:0 = ∅, 𝑆 1:𝑡 -1 = (𝑆 1 , . . . , 𝑆 𝑡 -1 ) and 𝑂 𝑧 (𝑆 1:𝑡 -1 ; 𝑤) ∈ 𝜕 𝑓 (𝑤, 𝑧). The trajectory induced by 𝑂 𝑺 which is initialized at 𝑤 0 = 0 is specified by the following equation:
𝑤 𝑡+1 = 𝑤 𝑡 -𝜂𝑂 𝑺 (𝑆 1:𝑡 -1 ; 𝑤 𝑡 , 𝑆 𝑡 ).
We will state the following general Lemma for an upper bound of Ω(𝜂 √ 𝑇) and later show how it proves Theorems 3.1 and 3.2. This Lemma shows that once an algorithm memorizes the entire training set we can quickly lead it to overfit.
Proof. We will prove here the Lemma for the case 𝐾 ≥ 34.
For 2 ≤ 𝐾 ≤ 34 the proof is deferred to Appendix B. First we notice that there exists 𝑑 ≤ 𝑑 ′ < 2𝑑 such that 𝑑 ′ = 2 𝑚 for some 𝑚 ∈ ℕ. Our construction will be over
ℝ 𝑑 ′ which of course implies that such a construction can be done in ℝ 2𝑑 using only a subspace of dimension 𝑑 ′ . From the proof of Lemma 8 in Livni (2024) there exists 𝑈 ⊂ {0, 1} 𝑑 ′ such that: ∀𝑢 ≠ 𝑣 ∈ 𝑈, 𝑢 • 𝑣 ≤ 5 16 𝑑 ′ and ∥𝑢∥ 2 = 7 16 𝑑 ′ , and |𝑈| > 𝑒 𝑑 ′ /258 > 𝑒 𝑑/258 ≥ 2 𝑑/256 = 2 𝑛 .
We will let 𝑍 = 𝑃(𝑈) and define a distribution Z over 𝑍 such that for a random sample 𝑉 ∼ Z, each 𝑢 ∈ 𝑈 lies in 𝑉 with probability 1 2 . Let 𝛼 = min 1 𝜂 √ 2𝑇
, 1 . First we will assume 𝑇 ≤ 𝑑 ′3 and we will consider the case where 𝑇 > 𝑑 ′3 at the end. Let 𝐵 ∈ ℕ be such that:
𝑑 ′ ≤ 𝐵 𝑇 34 1/3 < 2𝑑 ′ .
One can show that without loss of generality we can assume 𝐵 is also a power of 2, in particular 𝑑 ′ is divisible by 𝐵 for a large enough 𝑇. This will imply two useful facts:
𝑑 ′3 𝐵 3 ≤ 𝑇 34 =⇒ 𝜏 epoch + 𝑑 ′3 𝐵 3 ≤ 𝑇 34 + 𝑇 34 = 1 17 𝑇, (1) since 𝑇 ≥ 34𝜏 epoch and 𝑇 ≤ 272𝑑 ′3 𝐵 3 ,(2)
which we will use later. We consider blocks of indices so we will present the following notation: for two sets of indices 𝐼, 𝐽 ⊂ [𝑑 ′ ] denote 𝐼 ≺ 𝐽 if max{𝑖 ∈ 𝐼} < min{ 𝑗 ∈ 𝐽} and denote:
𝑒 𝐼 = 1 √︁ |𝐼 | ∑︁ 𝑖 ∈ 𝐼 𝑒 𝑖 and 𝑤(𝐼) = 1 √︁ |𝐼 | ∑︁ 𝑖 ∈ 𝐼
this section cite: []

Section: 𝑤(𝑖).
We will define the functions:
ℎ(𝑤) = max 0, max 𝐼 ⊂ [𝑑 ′ ], | 𝐼 |=𝐵 {-𝑤(𝐼)}, max 𝐼 ≺ 𝐽 ⊂ [𝑑 ′ ], | 𝐼 |=| 𝐽 |=𝐵 {-(𝑤(𝐼) -𝑤(𝐽))} 𝑔(𝑤, 𝑉) = 1 √ 𝑑 ′ max 𝑣 ∈𝑉 45𝜂𝛼𝑑 ′2 2 • 16 2 𝐵 1.5 , 𝑤 • 𝑣 .
Finally our loss function will be:
𝑓 (𝑤, 𝑉) = 𝑔(𝑤, 𝑉) + 𝛼ℎ(𝑤).(3)
Notice that 𝑓 is convex and 3-Lipschitz. Next we define a sample dependent oracle 𝑂 𝑺 . Given 𝑤 and all examples seen so far 𝑺 1:𝑡 = {𝑆 1 , . . . , 𝑆 𝑡 }:
1. If |𝑺 1:𝑡 | ≤ 𝜏 epoch output 0.
2. Otherwise check in lexicographic order if there exists
𝑢 0 ∈ 𝑈 such that 𝑢 0 ∉ (∪ 𝜏 epoch 𝑡 ′ =1 ∪ 𝑉 ∈𝑆 𝑡 ′ 𝑉).
• If there doesn't exist such 𝑢 0 or 𝑢 0 ∈ (∪ 𝑡 𝑡 ′ =𝜏 epoch +1 ∪ 𝑉 ∈𝑆 𝑡 ′ 𝑉) output an arbitrary subgradient.
• Otherwise we will compute the
7𝑑 ′ 16 indices of 𝑢 0 that hold 𝑢 0 = 1 denote them 𝐽 = { 𝑗 1 , . . . , 𝑗 7𝑑 ′ /16 }. Such a set exists since ∥𝑢 0 ∥ 2 = 7𝑑 ′
16 and 𝑢 0 ∈ {0, 1} 𝑑 ′ . We will divide the indices in 𝐽 to blocks of size 𝐵, for 𝑚 = 1, . . . , 7𝑑 ′ 16𝐵 :
𝐼 𝑚 = { 𝑗 (𝑚-1) •𝐵+1 , . . . , 𝑗 𝑚•𝐵 }.
We have three scenarios: (a) If there is a block 𝐼 𝑗 such that 𝑤(𝐼 𝑗 ) = 𝑤(𝐼 𝑗+1 ) > 0 then output: 𝛼(𝑒 𝐼 𝑗+1 -𝑒 𝐼 𝑗 ). (b) If there is no such block and if 𝐼 𝑗 is the minimal block such that 𝑤(𝐼 𝑗 ) = 0 output: -𝛼𝑒 𝐼 𝑗 . (c) If both of the conditions stated above do not hold output: 0.
We will denote the following event:
E = ∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆 and ∃𝑢 0 ∈ 𝑈 : 𝑢 0 ∉ ∪ 𝜏 epoch 𝑡=1 ∪ 𝑉 ∈𝑆 𝑡 𝑉 .
(4) We will prove 𝑂 𝑺 is a valid oracle in the following lemma whose proof is deferred to Appendix B.1. Lemma 5.2. 𝑂 𝑺 as stated above is a valid sample dependent first order oracle of 𝑓 as defined in Equation (3). Furthermore if E holds it will induce a trajectory such that we never leave the unit ball and no projection takes place.
We will now assume that E holds. Since |𝑈| ≥ 2 𝑛 from Lemma E.2 and from the definition of 𝜏 epoch ,
Pr[E] = Pr[∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆] • Pr[∃𝑢 0 ∈ 𝑈 : 𝑢 0 ∉ ∪ 𝑉 ∈𝑆 𝑉] ≥ 1 2 𝑝.
Next we will assume that E holds. Then after at most
𝑇 ′ = 𝜏 epoch + 1 + 7𝑑 ′ 16𝐵 𝑡=1 𝑡 𝑡 ′ =1 𝑡 ′ steps we will have that for every 𝑖 ∈ 𝐼 𝑡 , 𝑤 𝑇 ′ (𝑖) = 𝛼𝜂 √ 𝐵 7𝑑 ′ 16𝐵 + 1 -𝑡 , and for every 𝑡 ′ ≥ 𝑇 ′ , 𝑤 𝑡 ′ = 𝑤 𝑇 ′ which implies: 𝑤 𝑡 ′ • 𝑢 0 = 𝑤 𝑇 ′ • 𝑢 0 = √ 𝐵𝜂 𝛼 7𝑑 ′ 16𝐵 ∑︁ 𝑡=1 7𝑑 ′ 16𝐵 + 1 -𝑡 = √ 𝐵𝜂𝛼 7𝑑 ′ 16𝐵 ∑︁ 𝑡=1 𝑡 ≥ √ 𝐵𝜂𝛼 2 • 7𝑑 ′ 16𝐵 2 = √ 𝐵𝜂𝛼 49𝑑 ′2 2 • (16𝐵) 2 . Since 𝑇 ′ ≤ 𝜏 epoch + 𝑑 ′3
𝐵 3 ≤ 1 17 𝑇 (see Equation ( 1)), for every suffix averaging 𝜏:
𝑤 𝑇 , 𝜏 • 𝑢 0 ≥ 16 17 𝑤 𝑇 ′ • 𝑢 0 ≥ √ 𝐵𝜂𝛼 46𝑑 ′2 2 • (16𝐵) 2 .
Also because we assumed 𝑇 ≤ 272𝑑 ′3 𝐵 3 (see Equation ( 2)), and with probability at least 1 2 𝑢 0 will appear in a fresh sample:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (0) ≥ 1 2 46 √ 𝐵𝛼𝜂𝑑 ′1.5 2 • (16𝐵) 2 - 45 √ 𝐵𝛼𝜂𝑑 ′1.5 2 • (16𝐵) 2 ≥ 𝛼𝜂 4 • 16 2 𝑑 ′ 𝐵 1.5 ≥ 1 4 • √ 2 • √ 272 • 16 2 min{𝜂 √ 𝑇, 1}.
This concludes the proof for the case 𝑇 ≤ 𝑑 ′3 , when 𝑇 > 𝑑 ′3 if we take the construction with 𝑇 = 𝑑 ′3 then after 𝑇 steps our oracle will keep giving 0, hence we can use the above construction for any 𝑇 ′ > 𝑇. So for 𝑇 > 𝑑 ′3 we have with probability 1 2 𝑝:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (0) ≥ 1 4 √ 2 • √ 272 • 16 2 min{𝜂 √︁ 𝑑 ′3 , 1} ≥ 1 4 √ 2 • √ 272 • 16 2 min{𝜂 √︁ (256𝑛) 3 , 1}.
Overall in both cases with probability 1 2 𝑝:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (0) = Ω min 1, 𝜂 √︁ min{𝑛 3 , 𝑇 } . □
We will now show how the lemma gives the desired results.
Proof of Theorem 3.1. The Ω(1/(𝜂𝑇) + 𝜂) term is given in Lemma E.1. For the Ω(𝜂 √ 𝑇) term we will use Lemma 5.1. Let 𝑺 = {𝑧 𝑖 1 }, . . . , {𝑧 𝑖 𝑇 } be the ordered sequence of samples given to multi-pass SGD during a run of 𝑇 steps with training set 𝑆 of 𝑛 samples. After the first epoch we are sure to have observed the entire training set, so for 𝜏 epoch = 𝑛 and 𝑝 = 1 from Lemma 5.1 we get the result using a sampledependent oracle. We can relax this dependence using the reduction in Lemma D.2 and conclude the proof.
this section cite: []

Section: □
Proof of Theorem 3.2. The Ω(1/(𝜂𝑇) + 𝜂) term is given in Lemma E.1. For the Ω(𝜂 √ 𝑇) term we will use Lemma 5.1. Let 𝑺 = {𝑧 𝑖 1 }, . . . , {𝑧 𝑖 𝑇 } be the ordered sequence of samples given to with-replacement SGD during a run of 𝑇 steps with training set 𝑆 with 𝑛 samples. From folklore analysis of the coupon collector's problem it is known that with probability 1 2 after at most 𝑛 log 𝑛 iterations we have memorized the entire training set, so for 𝜏 epoch = 𝑛 log 𝑛 and 𝑝 = 1 2 from Lemma 5.1 we get the result using a sample-dependent oracle. We can relax this dependence using the reduction in Lemma D.2 and conclude the proof. □
this section cite: []

Section: A. Proofs of Upper Bounds
To establish the upper bounds, we will use stability arguments, and state the following definition for uniform stability:
Definition A.1. A randomized algorithm 𝐴 is 𝜖-uniformly stable if for all data sets 𝑆, 𝑆 ′ ∈ 𝑍 𝑛 such that 𝑆 and 𝑆 ′ differ in at most one example, we have: sup 𝑧 𝔼 [ 𝑓 ( 𝐴(𝑆), 𝑧) -𝑓 ( 𝐴(𝑆), 𝑧 ′ )] ≤ 𝜖 .
We will denote by 𝜖 stab the infimum over such 𝜖.
We recall the important Lemma stating that uniform stability implies generalization in expectation: Bousquet & Elisseeff (2002)). Let 𝐴 be an algorithm that is 𝜖 uniformly stable, then:
Lemma A.2 (Lemma 7 in
𝔼 𝑆, 𝐴 [𝐹 ( 𝐴(𝑆)) -𝐹 𝑆 ( 𝐴(𝑆))] ≤ 𝜖 .
In particular using our notation:
𝔼 𝑆, 𝐴 [𝐹 ( 𝐴(𝑆)) -𝐹 𝑆 ( 𝐴(𝑆))] ≤ 𝜖 stab .
Our upper bounds will be established using the following standard inequality:
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) = 𝔼 [𝐹 ( 𝑤) -𝐹 𝑆 ( 𝑤)] + 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) + ≤0 𝔼 𝐹 𝑆 (𝑤 ★ 𝑆 ) -𝐹 𝑆 (𝑤 ★ ) + =0 𝔼 𝐹 𝑆 (𝑤 ★ ) -𝐹 (𝑤 ★ ) ≤ 𝔼 [𝐹 ( 𝑤) -𝐹 𝑆 ( 𝑤)] + 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 )
Using Lemma A.2 yeilds:
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) ≤ 𝜖 stab + 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 )(5)
We will now continue to prove the upper bounds.
Proof of Theorem 3.3. We will prove the theorem for the average of the iterates. The proof for any suffix average 𝜏 = Ω(𝑇) can be derived using similar arguments. To bound the stability we will use the following Lemma which was originally given for multi-pass SGD with single-shuffle but can easily be extended to the multi shuffle case using similar arguments.
Lemma A.3 (Theorem 3.4 in Bassily et al. (2020)). Let 𝑓 : 𝑊 × 𝑍 → ℝ be a convex 𝐺-Lipschitz function. Then the uniform stability of multi-pass SGD with multi-shuffling or single-shuffling is bounded as:
𝜖 stab ≤ 2𝐺 2 𝜂 √ 𝑇 + 4𝐺 2 𝜂𝑇 𝑛 .
To bound the optimization error we will use the following: Koren et al., 2022). Let 𝑓 : 𝑊 × 𝑍 → ℝ be a convex and 𝐺-Lipschitz function. And let 𝑆 = {𝑧 1 , . . . , 𝑧 𝑛 } be some training set. Consider running 𝐾 ≥ 1 epochs of without replacement SGD over 𝑓 and 𝑆. Then, we have the following guarantee for 𝑤 = 1 𝑛𝐾 𝑇+1 𝑡=1 𝑤 𝑡 :
Lemma A.4 (Theorem 6 in
• For the multi-shuffle case:
𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 2𝜂𝐺 2 √ 𝑛 + 𝐷 2 2𝜂𝑛𝐾 + 1 2 𝜂𝐺 2 .
• For the single-shuffle case:
𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 8𝐺 2 𝜂 √ 𝑛𝐾 + 8𝐺 2 𝜂𝐾 + 𝐷 2 2𝜂𝑛𝐾 + 𝜂𝐺 2 2 . So for both cases 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 8𝐺 2 𝜂 √ 𝑛𝐾 + 8𝐺 2 𝜂𝐾 + 𝐷 2 2𝜂𝑛𝐾 + 𝜂𝐺 2 2 .
Using Equation ( 5):
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) ≤ 𝜖 stab + 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 2𝐺 2 𝜂 √ 𝑛𝐾 + 2𝜂𝐾 + 8𝐺 2 𝜂 √ 𝑛𝐾 + 8𝐺 2 𝜂𝐾 + 𝐷 2 2𝜂𝑛𝐾 + 𝜂𝐺 2 2 = 𝐷 2 2𝑛𝐾𝜂 + 10𝐺 2 𝜂 √ 𝑛𝐾 + 12𝐺 2 𝜂𝐾 + 1 2 𝐺 2 𝜂.
So we have:
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) = 𝑂 1 𝑛𝐾𝜂 + 𝜂 √ 𝑛𝐾 + 𝜂𝐾 = 𝑂 1 𝜂𝑇 + 𝜂 √ 𝑇 + 𝜂𝑇 𝑛 .
this concludes the proof. □ Proof of Theorem 3.4. We will prove the theorem for the average of the iterates. The proof for any suffix average 𝜏 = Ω(𝑇) can be derived using similar arguments. To bound the stability we will use the following Lemma
Lemma A.5 (Theorem 3.3 in Bassily et al. (2020)). Let 𝑓 : 𝑊 × 𝑍 → ℝ be a convex 𝐺-Lipschitz function.
Then the uniform stability of SGD with-replacement is bounded as:
𝜖 stab ≤ 4𝐺 2 𝜂 √ 𝑇 + 4𝐺 2 𝜂𝑇 𝑛 .
To bound the optimization error we will use the following classical result from Nemirovskiȋ & Yudin (1983).
Lemma A.6. Assume we run SGD with stepsize 𝜂 > 0 on a convex function 𝐹 ′ = 𝔼 Z [ 𝑓 ′ (𝑤, 𝑧)], for some distribution Z, assume further that 𝑓 ′ is 𝐺 Lipschitz and ∥𝑤 0 -𝑤 ★ ∥ ≤ 𝐷. Let 𝑤 denote the average of the 𝑇 iterates of the algorithm. Then we have:
𝐹 ′ ( 𝑤) -𝐹 ′ (𝑤 ★ ) ≤ 𝐷 2 2𝜂𝑇 + 1 2 𝐺 2 𝜂.
Applying Lemma A.6 on the empirical loss 𝐹 ′ = 𝐹 𝑆 with Z being the uniform distribution over 𝑆 we have:
𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 𝐷 2 2𝑛𝐾𝜂 + 1 2 𝐺 2 𝜂.
Using Equation ( 5) and putting everything together:
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) ≤ 𝜖 stab + 𝔼 𝐹 𝑆 ( 𝑤) -𝐹 𝑆 (𝑤 ★ 𝑆 ) ≤ 𝐷 2 2𝑛𝐾𝜂 + 1 2 𝐺 2 𝜂 + 4𝐺 2 𝜂 √ 𝑛𝐾 + 𝜂𝐾 ≤ 𝐷 2 2𝑛𝐾 • 1 𝜂 + 2𝐺 2 𝜂( √ 𝑛𝐾 + 𝐾).
So we have:
𝔼 𝐹 ( 𝑤) -𝐹 (𝑤 ★ ) = 𝑂 1 𝑛𝐾𝜂 + 𝜂 √ 𝑛𝐾 + 𝜂𝐾 = 𝑂 1 𝜂𝑇 + 𝜂 √ 𝑇 + 𝜂𝑇 𝑛 .
this concludes the proof. □ B. Proof of Lemma 5.1
We will prove now the Lemma for 2 ≤ 𝐾 ≤ 34. The proof for the 𝐾 ≥ 34 case is given in Section 5. From the proof of Lemma 8 in Livni (2024) there exists 𝑈 ⊂ {0, 1} 𝑑 such that for every 𝑢 ≠ 𝑣 ∈ 𝑈:
⟨𝑢, 𝑣⟩ ≤ 5𝑑 16 ≤ 7𝑑 16 = ∥𝑣∥ 2
and, |𝑈| ≥ 𝑒 𝑑/258 ≥ 2 𝑛 .
We will take the power set of 𝑈 as the sample space: 𝑍 = 𝑃(𝑈), identifying samples as subsets of 𝑈. Define a distribution Z over 𝑍 such that for a random sample 𝑉 ∼ Z, each 𝑢 ∈ 𝑈 lies in 𝑉 with probability
1 2 . Let 𝛼 = min 1, 1 𝜂 √ 𝑇 , denote 𝑤 (1) = 𝑤 [1 : 𝑑] and 𝑤 (2) = 𝑤 [𝑑 + 1 : 2𝑑].
We will introduce the following notation for a block of indices 𝐼 ⊂ [𝑑]:
𝑒 𝐼 = 1 √︁ |𝐼 | ∑︁ 𝑖 ∈ 𝐼 𝑒 𝑖 and 𝑤(𝐼) = 1 √︁ |𝐼 | ∑︁ 𝑖 ∈ 𝐼 𝑤(𝑖).
For 𝐵 = 3𝑑 𝜏 epoch consider the following functions:
𝑔(𝑤, 𝑉) = 1 √ 𝑑 max 𝑣 ∈𝑉 5𝛼 16 √ 𝐵 𝜂𝑑, 𝑤 (1) + 𝑤 (2) + 5𝜂𝛼 7 √ 𝐵 ì 1 • 𝑣 ,(6)
ℎ(𝑤) = 5 7 • max 𝐼 ⊂ [𝑑 ]:| 𝐼 | ≤ 𝐵 0, 𝑤 (1) (𝐼) + 2 7 • max 𝐼 ⊂ [𝑑 ]:| 𝐼 | ≤ 𝐵 0, -𝑤 (2) (𝐼) .(7)
Finally, our loss function will be:
𝑓 (𝑤, 𝑉) = 𝑔(𝑤, 𝑉) + 𝛼ℎ(𝑤).(8)
It holds that 𝑓 is convex and 3-Lipschitz. Next we define a sample dependent oracle 𝑂 𝑺 . Given 𝑤 𝑡 and all the examples seen so far 𝑺 1:𝑡 = {𝑆 1 , . . . , 𝑆 𝑡 }:
1. If |𝑺 1:𝑡 | ≤ 𝜏 epoch output 0.
2. Otherwise we can check if there exists 𝑢 0 ∈ 𝑈 such that 𝑢 0 ∉ (
𝜏 epoch 𝑡 ′ =1 𝑉 ∈𝑆 𝑡 ′ 𝑉). We will check this in lexicographic order. (a) If there doesn't exist such 𝑢 0 , or 𝑢 0 ∈ ( 𝑡 𝑡 ′ =𝜏 epoch +1 𝑉 ∈𝑆 𝑡 ′ 𝑉)
, output an arbitrary sub-gradient. (b) Otherwise we compute the following set:
𝐽 0 𝑡 = 𝑖 ∈ [𝑑] : 𝑢 0 (𝑖) = 0 and 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂𝛼 7 √ 𝐵 ì 1 (𝑖) > 0 if |𝐽 0 𝑡 | > 0 choose min{𝐵, |𝐽 0 𝑡 |} indices out of 𝐽 0 𝑡 denote them 𝑍 𝑡 . If |𝑍 𝑡 | < |𝐽 0 𝑡 | let 𝑂 𝑡 = ∅.
Otherwise we will let 𝑂 𝑡 be at most 𝐵 elements of the following set:
𝐽 1 𝑡 = 2𝑖 : 𝑖 ∈ [𝑑] and 𝑢 0 (𝑖) = 1 and 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂𝛼 7 √ 𝐵 ì 1 (𝑖) < 𝜂𝛼 √ 𝐵 .
Finally, output 𝛼 5 7 𝑒(𝑍 𝑡 ) -2 7 𝑒(𝑂 𝑡 ) .
We will denote the following event:
E = ∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆 and ∃𝑢 0 ∈ 𝑈 : 𝑢 0 ∉ (∪ 𝜏 epoch 𝑡=1 ∪ 𝑉 ∈𝑆 𝑡 𝑉)(9)
We will prove 𝑂 𝑺 is a valid oracle and that under the event of E no projections take place in the following Lemma whose proof is deferred to the end of the proof.
Lemma B.1. 𝑂 𝑺 stated above is a valid sample dependent first order oracle of 𝑓 defined in Equation (8). Furthermore it will induce a trajectory such that if E holds then we never leave the unit ball and no projections take place.
Since |𝑈| ≥ 2 𝑛 from Lemma E.2 and the definition of 𝜏 epoch :
Pr[E] = Pr[∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆] • Pr[∃𝑢 0 ∈ 𝑈 : 𝑢 0 ∉ (∪ 𝜏 epoch 𝑡=1 ∪ 𝑉 ∈𝑆 𝑡 𝑉)| ∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆] = Pr[∪ 𝜏 epoch 𝑡=1 𝑆 𝑡 = 𝑆] • Pr[∃𝑢 0 ∈ 𝑈 : 𝑢 0 ∉ ∪ 𝑉 ∈𝑆 𝑉] ≥ 1 2
𝑝 Next we will assume that E holds. According to 𝑂 𝑺 in every step we change at least 𝐵 indices unless we output 0. So after at most 𝑇 ′ = 𝜏 epoch + ⌈ 𝑑 𝐵 ⌉ steps we will have
|𝐽 0 𝑡 | = |𝐽 1 𝑡 | = 0 meaning that 𝑤 (1) 𝑇 ′ + 𝑤 (2) 𝑇 ′ + 5𝜂 𝛼 7 √ 𝐵 ì 1 = 𝜂 𝛼 √ 𝐵
𝑢 0 . Note that:
𝑇 ′ = 𝜏 epoch + ⌈ 𝑑 𝐵 ⌉ ≤ 𝜏 epoch + ⌈ 𝜏 epoch 3 ⌉ ≤ 𝜏 epoch + 𝜏 epoch 3 + 1 ≤ 𝜏 epoch + 𝜏 epoch 3 + 𝜏 epoch 24 𝜏 epoch ≥ 24 ≤ 3𝜏 epoch 2 .
From the way we defined 𝑂 𝑺 for every 𝑡 ′ ≥ 𝑇 ′ : 𝑤 𝑡 ′ = 𝑤 𝑇 ′ . For every 𝑖 ∈ [𝑑] such that 𝑢 0 (𝑖) = 1:
𝑤 (1) 𝑇 , 𝜏 + 𝑤 (2) 𝑇 , 𝜏 + 5𝜂𝛼 7 √ 𝐵 ì 1 (𝑖) = 1 𝜏 𝑇 ∑︁ 𝑡=𝑇 -𝜏+1 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂𝛼 7 √ 𝐵 ì 1 (𝑖) ≥ 1 𝑇 𝑇 ∑︁ 𝑡=1 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂𝛼 7 √ 𝐵 ì 1 (𝑖) = 𝜂𝛼 𝑇 √ 𝐵 5 7 • 𝑇 ′ + (𝑇 -𝑇 ′ ) ≥ 𝜂𝛼 √ 𝐵 5 7 • 3 2 + 1 2 = 11𝜂𝛼 14 √ 𝐵 Which implies: 𝑤 (1) 𝑇 , 𝜏 + 𝑤 (2) 𝑇 , 𝜏 + 5𝜂𝛼 7 √ 𝐵 ì 1 • 𝑢 0 = 7𝑑 16 • 11𝜂𝛼 14 √ 𝐵 = 11𝜂𝛼𝑑 32 √ 𝐵
From the definition of Z, with probability 1 2 , 𝑢 0 will appear in a new random sample so for every suffix averaging 𝜏:
𝐹 (𝑤 𝑇 , 𝜏 ) -𝐹 (0) ≥ 1 2 1 √ 𝑑 𝑤 (1) 𝑇 , 𝜏 + 𝑤 (2) 𝑇 , 𝜏 + 5𝜂𝛼 7 √ 𝐵 ì 1 • 𝑢 0 - 5𝛼𝜂 16 • √ 𝐵 √ 𝑑 = 1 2 11𝜂𝛼 √ 𝑑 32 √ 𝐵 - 5𝛼𝜂 16 • 2 √ 𝑐 √ 𝑑 = 𝛼𝜂 √ 𝑑 2 • 32 • √ 𝐵 .
Overall we have shown that with probability 1 2 for every suffix averaging 𝜏:
𝐹 ( 𝑤 𝑇 , 𝜏 ) -𝐹 (0) ≥ 𝛼𝜂 2 • 32 • √︂ 𝑑 𝐵 = 𝛼𝜂 2 • 32 • √︂ 𝜏 epoch 3 = Ω min 1, 𝜂 √ 𝑇 ,
since 𝑇 = 𝑂 (𝜏 epoch ), the proof is complete. □
this section cite: ['b2', 'b1', 'b7', 'b13']

Section: B.1. Proofs for Auxiliary Lemmas
We will prove here Lemmas 5.2 and B.1 that concern gradient oracles that were given in the proofs of Lemma 5.1.
Proof of Lemma B.1. First we note that for every 𝑢 ∈ 𝑈 :
5𝜂𝛼 7 √ 𝐵 ì 1 • 𝑣 = 7 16 𝑑 • 5𝜂𝛼 7 √ 𝐵 = 5𝜂𝛼𝑑 16 √ 𝐵 (10
)
So 0 ∈ 𝜕 𝑓 (0, 𝑉) for every 𝑉 ∈ 𝑃(𝑈). For that reason we can stay at 0 for as long as we want. Note that it suffices to show that if E holds the oracle is valid since for every other scenario we output either 0 or an arbitrary subgradient. If E holds we are clearly outputting a subgradient of 𝛼ℎ(𝑤) in Item 2b, so it is left to prove that for every 𝑉 ∈ 𝑆 we have that 0 ∈ 𝜕𝑔(𝑤 𝑡 , 𝑉). Indeed,
• If for all 𝑖 ∈ [𝑑]:
𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂 𝛼 7 √ 𝐵 ì 1 (𝑖) ≤ 5𝜂 𝛼 7 √ 𝐵
then as we saw in Equation ( 10) this holds.
• If there exists 𝑖 ∈ [𝑑] such that:
𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂 𝛼 7 √ 𝐵 ì 1 (𝑖) > 5𝜂 𝛼 7 √ 𝐵
this can happen only after we have zeroed all the entries that 𝑢 0 has zeros on. So we are left only with positive coordinates of 𝑢 0 and the proof is completed by noticing that since 𝑢 0 ∉ ∪ 𝑉 ∈𝑆 𝑉, for every 𝑣 ∈ ∪ 𝑉 ∈𝑆 𝑉:
𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 + 5𝜂𝛼 7 √ 𝐵 ì 1 • 𝑣 ≤ 𝜂𝛼 √ 𝐵 𝑢 0 • 𝑣 = 5𝜂𝛼𝑑 16 √ 𝐵 .
This completes the proof that 𝑂 𝑺 is valid. To see that projections don't take place according to this oracle for every 𝑡 ∈ [𝑇] and 𝑖 ∈ [2𝑑] we have that: 𝑤 𝑡 (𝑖) ≤ 𝜂 𝛼 √ 𝐵 which implies:
∥𝑤 𝑡 ∥ 2 ≤ 𝜂𝛼 • √︂ 2𝑑 𝐵 ≤ 𝜂𝛼 √︂ 2𝜏 epoch 3 ≤ 𝜂𝛼 √ 𝑇 ≤ 1, since 𝛼 = min 1, 1 𝜂 √ 𝑇 . □
Proof of Lemma 5.2. For the first part of the oracle note that 0 ∈ 𝜕 𝑓 (0, 𝑉) so we can stay at 0 for as long as we like, in particular for 𝜏 epoch steps. For the second part of the oracle, if there doesn't exists such 𝑢 0 or 𝑢 0 ∈ ( 𝑡 𝑡 ′ =𝜏 epoch +1 𝑉 ∈𝑆 𝑡 ′ 𝑉) we output a valid sub-gradient by definition. Otherwise, we are clearly taking gradient steps for 𝛼ℎ(𝑤). It is left to show that in this case 0 ∈ 𝜕𝑔(𝑤 𝑡 , 𝑉). Indeed, this event ensures that up to this point the oracle only output 0 or executes Items 2a to 2c, so our only nonzero coordinates are positive coordinates of 𝑢 0 . Also, for every 𝑣 ∈ ( 𝑉 ∈𝑆 𝑡 𝑉), we have that 𝑣 ≠ 𝑢 0 so 𝑣 • 𝑢 0 ≤ 5𝑑 ′ 16 which implies:
𝑤 𝑡 • 𝑣 ≤ 𝑑 ′ ∑︁ 𝑖=1 𝑤 𝑡 (𝑖)1{𝑣(𝑖) = 𝑢 0 (𝑖) = 1} ≤ 5𝑑 ′ 16𝐵 ∑︁ 𝑖=1 √ 𝐵𝜂𝛼 7𝑑 ′ 16𝐵 + 1 -𝑡 ≤ 5𝑑 ′ 16𝐵 ∑︁ 𝑖=0 √ 𝐵𝜂𝛼 7𝑑 ′ 16𝐵 -𝑡 ≤ ≤ √ 𝐵𝜂𝛼 5𝑑 ′ 16𝐵 • 7𝑑 ′ 16𝐵 - 1 2 5𝑑 ′ 16𝐵 2 ≤ √ 𝐵𝜂𝛼 45𝑑 ′2 2 • (16𝐵) 2 .
This concludes the proof that 𝑂 𝑺 is well defined. To prove we never leave the unit ball if E as depicted in Equation (4) holds, notice that this is exactly the event where we either output 0 or execute one of Items 2a to 2c in the oracle. We will show by induction that for all 𝑡 ∈ [𝑇]:
∥𝑤 𝑡+1 ∥ 2 = ∥𝑤 𝑡 -𝜂𝑂 𝑺 (𝑆 1:𝑡 , 𝑤 𝑡 , 𝑉 𝑡 ) ∥ 2 ≤ 2𝜂 2 𝛼 2 (𝑡 + 1).
For the base case ∥𝑤 0 ∥ = 0 ≤ 2𝜂 2 𝛼 2 . Now assume it holds for 𝑡 and we will prove for 𝑡 + 1. Consider the case where the first type of update is performed with 𝑤 𝑡 (𝐼 𝑗 ) = 𝑤 𝑡 (𝐼 𝑗+1 ):
∥𝑤 𝑡+1 ∥ 2 = ∥𝑤 𝑡 -𝜂𝛼𝑒 𝐼 𝑗+1 + 𝜂𝛼𝑒 𝐼 𝑗 ∥ 2 = 𝑗 -1 ∑︁ 𝑖=1 (𝑤 𝑡 (𝐼 𝑠 )) 2 + (𝑤 𝑡 (𝐼 𝑗 ) + 𝜂𝛼) 2 + (𝑤 𝑡 (𝐼 𝑗+1 ) -𝜂𝛼) 2 + 7𝑑 ′ 16𝐵 ∑︁ 𝑠= 𝑗+2 (𝑤 𝑡 (𝐼 𝑠 )) 2 = 7𝑑 ′ 16𝐵 ∑︁ 𝑠=1 (𝑤 𝑡 (𝐼 𝑠 )) 2 + 2𝜂𝛼(𝑤 𝑡 (𝐼 𝑗 ) -𝑤 𝑡 (𝐼 𝑗+1 )) + 2𝜂 2 𝛼 2 = 7𝑑 ′ 16𝐵 ∑︁ 𝑖=1 (𝑤 𝑡 (𝐼 𝑠 )) 2 + 2𝜂 2 𝛼 2 ≤ 2𝜂 2 𝛼 2 • 𝑡 + 𝜂 2 𝛼 2 = 𝜂 2 𝛼 2 (1 + 𝑡).
And if the second type of update occurs:
∥𝑤 𝑡+1 ∥ 2 = ∥𝑤 𝑡 + 𝜂𝛼𝑒 𝐼 𝑗 ∥ 2 = ∑︁ 𝑠≠ 𝑗 (𝑤 𝑡 (𝐼 𝑠 )) 2 + 𝜂 2 𝛼 2 ≤ 2𝜂 2 𝛼 2 𝑡 + 𝜂 2 𝛼 2 ≤ 2𝜂 2 𝛼 2 . Since 𝛼 = min 1, 1 𝜂 √ 2𝑇
, this shows that for all 𝑡 ∈ [𝑇]:
∥𝑤 𝑡 ∥ ≤ 2𝜂 2 𝛼 2 𝑇 = 2𝜂 2 𝑇 • 1, 1 2𝜂 2 𝑇 ≤ 1. □
this section cite: []

Section: C. Proof of Theorem 4.1
From the reduction in Lemma D.2 it suffices to prove the result for a sample-dependent oracle as defined in Appendix D. This is established in the following Lemma:
Lemma C.1. For every 𝑛 ≥ 17, 𝑑 = 712𝑛 log 𝑛 and 𝜂 > 0, there are a finite datapoint set 𝑍 and a distribution Z over 𝑍, a 3-Lipschitz convex function 𝑓 (𝑤, 𝑧) in ℝ 2𝑑 and sample-dependent gradient oracle 𝑂 𝑺 such that with probability 1 2 if we run one-pass SGD with 𝜂 as a learning rate for 𝑛 steps then for every suffix averaging 𝜏 ∈ [𝑛 + 1]:
𝐹 ( 𝑤 𝑛, 𝜏 ) -𝐹 (0) = Ω min 𝜂 √ 𝑛, 1 . Proof of
Lemma C.1. Since 𝑑 ≥ 256, from Lemma 1 in Schliserman et al. (2024), there exists 𝑈 ⊂ 1 √ 𝑑 , -1 √ 𝑑 𝑑 such that |𝑈| ≥ 2 𝑑 178 ≥ 2 𝑛 and for all 𝑢 ≠ 𝑣 ∈ 𝑈: |⟨𝑢, 𝑣⟩| ≤ 1 8 . This implies that: ∀𝑢 ≠ 𝑣 ∈ 𝑈 : |{𝑖 ∈ [𝑑] : 𝑢(𝑖) = 𝑣(𝑖)}| ≤ 9 16 𝑑.
Let 𝑍 = 𝑃(𝑈), and let Z be a distribution over 𝑍 such just for a new sample 𝑉 ∼ Z every 𝑢 ∈ 𝑈 will be in 𝑉 with probability
𝛿 = 1 4𝑛 2 . Let 𝛼 = min{1, 1 𝜂 √ 𝑛 }. Let 𝑊 {𝑥 ∈ ℝ 2𝑑 : ∥𝑥∥ ≤ 1}. Denote 𝑤 (1) = 𝑤 [1 : 𝑑], 𝑤 (2) = 𝑤 [𝑑 + 1 : 2𝑑].
For a subset of indices 𝐼 ⊂ [𝑑] denote the following:
𝑒(𝐼) = 1 |𝐼 | ∑︁ 𝑖 ∈𝐼 𝑒 𝑖 and 𝑤(𝐼) = 1 |𝐼 | ∑︁ 𝑖 ∈ 𝐼
this section cite: []

Section: 𝑤(𝑖).
We will consider the following function with blocks of indices of size at most 8 log 𝑛:
𝑓 (𝑤, 𝑉) = max 𝑣 ∈𝑉 9𝛼 16 • 2 √ 2 𝜂 √ 𝑛, 𝑤 (1) + 𝑤 (2) • 𝑣(12)
+ max 𝐼 ⊂𝑑:| 𝐼 | ≤8 log 𝑛 0, 𝑤 (1) (𝐼) + max 𝐼 ⊂𝑑:| 𝐼 | ≤8 log 𝑛 0, -𝑤 (2) (𝐼) .
It holds that 𝑓 is convex and 3-Lipschitz. Now assume we have an order over the vectors in 𝑈 which is lexicographic. Denote by 𝑢 (𝑟 ) the 𝑟-th vector in such an order. We will be interested in the following event:
E = ∩ ⌈ 𝑛 16 ⌉ 𝑖=1 𝑉 𝑖 ≠ ∅ and min 𝑟 ∈ [ |𝑈 | ] {𝑢 (𝑟 ) : 𝑢 (𝑟 ) ∈ ∩ ⌈ 𝑛 16 ⌉ 𝑖=1 𝑉 𝑖 } ∈ ∩ 𝑛 𝑖=⌈ 𝑛 16 +1⌉ 𝑉 𝑖 .(13)
From Lemma E.3, E occurs with probability at least 1 2 . From now on we will assume E occurs. Let 𝑆 = {𝑉 1 , . . . , 𝑉 𝑛 } be some training set. Next we define a sample dependent oracle 𝑂 𝑺 . Given 𝑤 𝑡 and at all past samples 𝑺 1:𝑡 = {𝑉 1 , . . . , 𝑉 𝑡 }:
1. If |𝑺 1:𝑡 | ≤ ⌈ 𝑛 16 ⌉ output 0.
2. Otherwise check if the exists 𝑢 0 ∈ ∩ 𝑛 𝑖=1 𝑉 𝑖 . We will check this in lexicographic order over vectors in 𝑈, so we will find the minimal such vector.
(a) If ∩ ⌈ 𝑛 16 ⌉ 𝑖=1 𝑉 𝑖 = ∅ or 𝑢 0 ∈ ∪ 𝑡 𝑡 ′ =⌈ 𝑛 16 ⌉+1 𝑉 𝑖 𝑡 ′ ,
output an arbitrary sub-gradient. (b) Otherwise compute the following sets: 𝐽 𝑝 𝑡 = 2𝑖 : 𝑖 ∈ [𝑑] and 𝑢 0 (𝑖) > 0 and 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 (𝑖) = 0 𝐽 𝑛 𝑡 = 𝑖 ∈ [𝑑] : 𝑢 0 (𝑖) < 0 and 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 (𝑖) = 0 . We will use 𝑤 (1) to take steps in the negative coordinates of 𝑢 0 -𝐽 𝑛 𝑡 -and 𝑤 (2) to take steps in the positive coordinates of 𝑢 0 -𝑗 𝑝 𝑡 . Choose min{8 log 𝑛, |𝐽 𝑝 𝑡 |} indices out of 𝐽 𝑝 𝑡 denote them 𝑃 𝑡 and min{8 log 𝑛, |𝐽 𝑛 𝑡 |} out of 𝐽 𝑛 𝑡 denote them 𝑁 𝑡 . Output 𝛼 (𝑒(𝑁 𝑡 ) -𝑒(𝑃 𝑡 )) .
We will prove this oracle is valid and when E holds not projections take place. Its proof is deferred to Appendix C.1.
Lemma C.2. 𝑂 𝑺 stated above is a valid sample dependent first order oracle of 𝑓 defined in Equation (12). Furthermore if E holds it will induce a trajectory such that we never leave the unit ball and no projections take place.
Assuming E occurs, for 𝑇 ′ = ⌈ 𝑛 16 ⌉ + ⌈ 𝑑 8 log 𝑛 ⌉ ≤ 3𝑛 16 + 2 we have that
∀𝑡 ≥ 𝑇 ′ : 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 = 𝛼𝜂 √ 𝑑 √︁ 8 log 𝑛 𝑢 0 = 𝛼𝜂 √ 𝑛 2 √ 2 .
So for every suffix averaging 𝜏:
𝑤 (1) 𝑛, 𝜏 + 𝑤 (2) 𝑛, 𝜏 • 𝑢 0 ≥ 1 - 𝑇 ′ 𝑛 𝑤 (1) 𝑇 + 𝑤 (2) 𝑇 ′ • 𝑢 0 = 1 - 3𝑛 16 + 2 𝑛 • 𝛼𝜂 √ 𝑛 2 √ 2 ∥𝑢 0 ∥ 2 ≥ 1 - 3𝑛 16 + 𝑛 8 𝑛 • 𝛼𝜂 √ 𝑛 2 √ 2 𝑛 ≥ 16 = 11 16 • 𝛼𝜂 √ 𝑛 2 √ 2 .
We now have that with probability 1 2 :
𝐹 𝑆 (𝑤 𝑛, 𝜏 ) -𝐹 𝑆 (0) ≥ 1 𝑛 𝑛 16 • 11𝛼𝜂 √ 𝑛 16 • 2 √ 2 + 7𝑛 8 • 9𝛼𝜂 √ 𝑛 16 • 2 √ 2 - 9𝛼𝜂 √ 𝑛 16 • 2 √ 2 ≥ 𝛼𝜂 √ 𝑛 365 = 1 365 • min{1, 𝜂 √ 𝑛}. □
this section cite: []

Section: C.1. Proofs for Auxiliary Lemmas
Proof of Lemma C.2. For the first part of the oracle note that 0 ∈ 𝜕 𝑓 (0, 𝑉) so we can stay at 0 for as long as we like, in particular for ⌈ 𝑛 16 ⌉ steps.
For the second part if ∩ ⌈ 𝑛 16 ⌉ 𝑖=1 𝑉 𝑖 = ∅ we keep outputting 0 which is allowed as we saw before. If ∩ ⌈ 𝑛 16 ⌉ 𝑖=1 𝑉 𝑖 ≠ ∅ we choose some 𝑢 0 . If 𝑢 0 ∈ ∪ 𝑡 𝑖=⌈ 𝑛 16 ⌉+1
𝑉 𝑖 we output a valid sub gradient by definition, if it is not we are clearly a taking gradient step for 𝛼ℎ(𝑤). It is thus left to show that in this case 0 ∈ 𝜕𝑔(𝑤 𝑡 , 𝑉). Indeed, in this event until this point the gradient only output 0 or executed Item 2b in the oracle so the nonzero coordinates in 𝑤 𝑡 have the same sign as 𝑢 0 . Also for all 𝑣 ∈ 𝑉 𝑖 𝑡 we have that 𝑣 ≠ 𝑢 0 which suggests using Equation ( 11): 𝑤 (1) 𝑡 + 𝑤 (2) 𝑡 • 𝑣 ≤ ∑︁ 𝑖:𝑣 (𝑖)=𝑢 0 (𝑖) 𝑤 (1) 𝑡 (𝑖) + 𝑤 (2) 𝑡 (𝑖) • 𝑣(𝑖) ≤ 𝛼𝜂 √︁ 8 log 𝑛 √ 𝑑 • ∑︁ 𝑖:𝑣 (𝑖)=𝑢 0 (𝑖) 𝑢 0 • 𝑣(𝑖) ≤ 9𝛼𝜂 16 • 2 √ 2 √ 𝑛. This concludes the proof that 𝑂 𝑺 is well defined. To see no projections take place if E as depicted in Equation (13) holds, note that this is exactly the event where the oracle either outputs 0 or executes Item 2b in the oracle. In this event: ∀𝑡 ∈ [𝑛], 𝑖 ∈ [𝑑] : |𝑤 𝑡 (𝑖)| ≤ 𝛼𝜂 √︁ 8 log 𝑛 =⇒ ∥𝑤 𝑡 ∥ 2 ≤ 𝛼𝜂 √ 2𝑑 √︁ 8 log 𝑛 = 1 2 min{𝜂 √ 𝑛, 1} ≤ 1 2 . □ D. Reduction to Sample-Dependent Oracle when 𝑆 1:0 = ∅, 𝑆 1:𝑡 = (𝑆 1 , . . . , 𝑆 𝑡 ). Finally we denote the trajectory induced by 𝑂 𝑺 which is initialized at 𝑤 0 = 0 and is specified by the following equation:
𝑤 𝑺 𝑡+1 = 𝑤 𝑺 𝑡 -𝜂𝑂 𝑺 (𝑆 1:𝑡 -1 ; 𝑤 𝑡 , 𝑆 𝑡 ).(14)
The following Lemma from Livni (2024) will prove the reduction to the sample dependent oracle case: Livni (2024)). Suppose 𝑞 ∈ ℝ 𝑇 , ∥𝑞∥ ∞ ≤ 1 and 𝑍 is finite. And suppose that 𝑓 (𝑤, 𝑧) is a convex, L-Lipschitz function over 𝑤 ∈ ℝ 𝑑 , let 𝜂 > 0, let 𝑂 𝑺 be a sample dependent first order oracle, and for every sequence of samples 𝑺 = (𝑆 1 , . . . , 𝑆 𝑇 ) define the sequence {𝑤 𝑺 𝑡 } 𝑇+1 𝑡=1 as in Equation ( 14). Then, for every 𝜖 > 0 there exists an 𝐿 + 1-Lipschitz convex function f ((𝑤, 𝑥), 𝑧) over ℝ 𝑑+1 (that depends on 𝑞, 𝑓 , 𝑇, 𝜂, 𝑛, 𝑂 𝑺 , 𝜖) such that for any oracle 𝑂 𝑧 for f (𝑧, •), define 𝑢 0 = 0 ∈ ℝ 𝑑 and 𝑥 0 = 0 ∈ ℝ and
Lemma D.1 (Lemma 9 in
(𝑢 𝑡 , 𝑥 𝑡 ) = (𝑢 𝑡 -1 , 𝑥 𝑡 -1 ) - 𝜂 |𝑆 𝑡 | ∑︁ 𝑧 ∈𝑆 𝑡 𝑂 𝑧 ((𝑢 𝑡 , 𝑥 𝑡 ))
then if we define The following Lemma easily follows:
this section cite: ['b11', 'b11']

Section: 
Lemma D.2. Suppose 𝑍 is finite, 𝑓 (𝑤, 𝑧) is a convex, L-Lipschitz function over 𝑤 ∈ ℝ 𝑑 , let 𝜂 > 0, let 𝑂 𝑺 be a sample dependent first order oracle, and for every sequence of samples 𝑺 = (𝑆 1 , . . . , 𝑆 𝑇 ) define the sequence {𝑤 𝑺 𝑡 } 𝑇+1 𝑡=1 as in Equation (14). Suppose also that for some number of steps 𝑇 and 𝜏 ∈ [𝑇 + 1] we have with probability 𝑝: 𝐹 ( 𝑤 𝑺 𝑇 , 𝜏 ) -𝐹 (0) ≥ ℓ Where ℓ > 0. Then there exists an 𝐿 +1-Lipschitz convex function f ((𝑤, 𝑥), 𝑧) over ℝ 𝑑+1 (that depends on 𝜏, 𝑓 , 𝑇, 𝜂, 𝑛, 𝑂 𝑺 , 𝜖) such that for any oracle 𝑂 𝑧 for f (𝑧, •) if we define 𝑣 0 = 0 ∈ ℝ 𝑑+1 and
𝑣 𝑡+1 = 𝑣 𝑡 - 𝜂 |𝑆 𝑡 | ∑︁ 𝑧 ∈𝑆 𝑡 𝑂 𝑧 (𝑣 𝑡 ),
we will have:
F ( 𝑣 𝑇 , 𝜏 ) -F (0) ≥ ℓ 2 .
Proof. Let 𝜖 = ℓ 4 > 0, define 𝑞 ∈ ℝ 𝑑 as follows:
𝑞(𝑡) = 1 𝑇 -𝜏+2 𝜏 ≤ 𝑡 ≤ 𝑇 + 1 0 otherwise .
For this 𝑞, 𝜖 let f be the function whose existence follows from Lemma D.1. It is easy to see that 𝑤 𝑺 𝜏 = 𝑤 𝑺 𝑞 and 𝑣 𝜏 = (𝑢 𝑞 , 𝑥 𝑞 ). Then with probability 𝑝 we have: Koren et al. (2022)). For any step-size 𝜂 > 0, 𝑇 ∈ ℕ and 𝑑 = ⌈16𝜂 2 𝑇 2 ⌉ there exists a deterministic convex optimization problem ℎ : 𝑊 → ℝ where 𝑊 ⊂ ℝ 𝑑+1 is of constant diameter such that:
F ( 𝑣 𝑇 , 𝜏 ) -F (0) = F (𝑢 𝑞 , 𝑥 𝑞 ) -F (0) ≥ 𝐹 (𝑤 𝑺 𝑞 ) -𝐹 (0) -2𝜖 ≥ ℓ -2 • ℓ 4 = ℓ 2 . □ E. Additional Lemmas Lemma E.1 (Lemma 14 in
ℎ( 𝑤) -min 𝑤 ∈𝑊 ℎ(𝑤) ≥ 1 8 min 1 𝜂𝑇 + 𝜂, 1 .
Lemma E.2. Let 𝑈 be a subspace such that |𝑈| ≥ 2 𝑛 . Let 𝑍 = 𝑃(𝑈) and let Z be a distribution of 𝑍 such that for a random sample 𝑉 ∼ Z every 𝑢 ∈ 𝑈 is in 𝑉 with probability 1 2 . Then for a sample 𝑆 = {𝑉 1 , . . . , 𝑉 𝑛 } ∼ Z 𝑛 with probability at least 1 2 it holds that: Schliserman et al. (2024)). For 𝑑 = 712𝑛 log 𝑛 and 𝑈 𝑑 as depicted in Lemma 1 from Schliserman et al. (2024), let 𝑍 = 𝑃(𝑈) and let a distribution Z over 𝑍 be such that for 𝑉 ∼ Z every 𝑢 ∈ 𝑈 𝑑 ′ is in 𝑉 with probability Then if 𝑇 = 𝑛, Pr[E] ≥ 1 2 . In particular with probability at least 1 2 :
∪ 𝑛 𝑖=1 𝑉 𝑖 ≠ 𝑈 Proof. First it holds that: Pr 𝑢 ∈ ∪ 𝑛 𝑖=1 𝑉 𝑖 = 1 -Pr 𝑢 ∉ ∪ 𝑛 𝑖=1 𝑉 𝑖 = 1 -2 -𝑛 . This implies: Pr ∪ 𝑑 𝑖=1 𝑉 𝑖 = 𝑈 = Pr ∀𝑢 ∈ 𝑈, 𝑢 ∈ ∪ 𝑛 𝑖=1 𝑉 𝑖 = (1 -2 -𝑛 ) |𝑈 | ≤ (1 -2 -𝑛 ) 2 𝑛 ≤ 1 𝑒 < 1 2 . □ Lemma E.3 (Lemma 9 in
𝑃 ⌈𝑛/16⌉ ≠ ∅ and 𝐽 ⌈𝑛/16⌉ ∈ 𝑆 ⌈𝑛/16⌉ .
.
this section cite: ['b7', 'b19', 'b19']

Section: References
Ref_id:b0 Title: SGD Generalizes Better than GD (and Regularization Doesn't Help) Year: (2021)
Ref_id:b1 Title: Stability of Stochastic Gradient Descent on Nonsmooth Convex Losses Year: (2020)
Ref_id:b2 Title: Stability and Generalization Year: (2002)
Ref_id:b3 Title: Tighter Lower Bounds for Shuffling SGD: Random Permutations and Beyond Year: (2023)
Ref_id:b4 Title: Random Reshuffling is Not Always Better Year: (2020)
Ref_id:b5 Title: Generalization of ERM in Stochastic Convex Optimization: The Dimension Strikes Back Year: (2016)
Ref_id:b6 Title: Generalize Better: Stability of Stochastic Gradient Descent Year: (2016)
Ref_id:b7 Title: Benign Underfitting of Stochastic Gradient Descent Year: (2022)
Ref_id:b8 Title: Recht-ré noncommutative arithmeticgeometric mean conjecture is false Year: (2020)
Ref_id:b9 Title: Fine-Grained Analysis of Stability and Generalization for Stochastic Gradient Descent Year: (2020)
Ref_id:b10 Title: Generalization Performance of Multi-Pass Stochastic Gradient Descent with Convex Loss Functions Year: (2021)
Ref_id:b11 Title: The Sample Complexity of Gradient Descent in Stochastic Convex Optimization Year: (2024)
Ref_id:b12 Title: SGD Without Replacement: Sharper Rates for Ggeneral Smooth Convex Functions Year: (2019)
Ref_id:b13 Title: Problem Complexity and Method Efficiency in Optimization Year: (1983)
Ref_id:b14 Title: Select without Fear: Almost All Mini-Batch Schedules Generalize Optimally Year: (2023)
Ref_id:b15 Title: Closing the Convergence Gap of SGD Without Replacement Year: (2020)
Ref_id:b16 Title: Beneath the valley of the noncommutative arithmetic-geometric mean inequality: conjectures, case-studies Year: (2012)
Ref_id:b17 Title: How Good is SGD with Random Shuffling? Year: (2020-07)
Ref_id:b18 Title: Random Shuffling Beats SGD Only After Many Epochs on Ill-Conditioned Problems Year: (2021)
Ref_id:b19 Title: The Dimension Strikes Back with Gradients: Generalization of Gradient Methods in Stochastic Convex Optimization Year: (2024)
Ref_id:b20 Title: The Role of Implicit Regularization, Batch-Size and Multiple-Epochs Year: (2021)
Ref_id:b21 Title: Stability and Uniform Convergence Year: (2010)
Ref_id:b22 Title:  Year: (2021)
Ref_id:b23 Title: Lower Generalization Bounds for GD and SGD in Smooth Stochastic Convex Optimization Year: (2023)
