Title: Computational Efficiency under Covariate Shift in Kernel Ridge Regression
Abstract: This paper addresses the covariate shift problem in the context of nonparametric regression within reproducing kernel Hilbert spaces (RKHSs). Covariate shift arises in supervised learning when the input distributions of the training and test data differ, presenting additional challenges for learning. Although kernel methods have optimal statistical properties, their high computational demands in terms of time and, particularly, memory, limit their scalability to large datasets. To address this limitation, the main focus of this paper is to explore the trade-off between computational efficiency and statistical accuracy under covariate shift. We investigate the use of random projections where the hypothesis space consists of a random subspace within a given RKHS. Our results show that, even in the presence of covariate shift, significant computational savings can be achieved without compromising learning performance. 39th Conference on Neural Information Processing Systems (NeurIPS 2025).importance weighting (IW) function. This function corresponds to the Radon-Nikodym derivative of the test marginal distribution with respect to the training marginal distribution Huang et al. (2006); Sugiyama et al. (2012); Fang et al. (2020). Shimodaira (2000) was the first to demonstrate the consistency of the importance-weighted maximum likelihood estimator, while in Cortes et al. (2010) the authors derived suboptimal finite-sample bounds for restricted function classes with finite pseudodimension. In the context of nonparametric regression in reproducing kernel Hilbert spaces (RKHSs) and under the assumption that the regression function belongs to the RKHS (well-specified case), Gizewski et al. (2022) provided optimal excess risk convergence results for the minimizer of the reweighted empirical risk when the weight function is known and uniformly bounded. In Ma et al. 2023, authors recently showed that the standard unweighted kernel ridge regression estimator is still minimax optimal under an appropriate choice of the regularization parameter, provided the IW function is uniformly bounded or its second moment is bounded. Similar results for nonparametric classification were obtained in Kpotufe & Martinet (2021). We also mention Schmidt-Hieber & Zamolodtchikov ( 2024); Pathak et al. (2022) in the context of nonparametric regression for classes of functions different from RKHSs, and Wen et al. (2014); Lei et al. (2021); Yamazaki et al. (2007) for parametric models. Building on Ma et al. (2023), in Gogolashvili et al. ( 2023) the authors extended the analysis to (simplified) misspecified case, where the regression function is not assumed to lie in the RKHS itself, but its projection is. In this setting, the authors showed that, under covariate shift, the unweighted classic KRR predictor is not a consistent estimator of the projection of the regression function. In such cases, IW correction is necessary.Kernel methods provide a robust framework for nonparametric learning, but their scalability is limited by high computational and memory costs-challenges that clearly do not disappear under covariate shift. To address this, researchers have developed more efficient strategies, ranging from improved optimization (

Section: Introduction
Classical supervised learning assumes that the training and test data distributions are identical Vapnik (1999). However, in practice, it is often the case that there is a significant mismatch between the two Quiñonero-Candela et al. (2022). This discrepancy can arise from various factors, such as inconsistencies in measurement equipment, differences in data collection domains, or variations in subject populations Koh et al. (2021). Among these scenarios, one particularly interesting and common case is known as covariate shift Sugiyama & Kawanabe (2012). Covariate shift occurs when the marginal distributions of the input covariates differ between the training and test data, while the conditional distribution of the output label given the input covariates remains unchanged Shimodaira (2000); Cortes et al. (2008). This phenomenon is observed in a variety of well-studied learning problems, including among all domain adaptation Ben-David et al. (2006); Mansour et al. (2009); Cortes & Mohri (2014); Zhang et al. (2012), active learning Wiens (2000); Kanamori & Shimodaira (2003); Sugiyama & Ridgeway (2006), natural language processing Jiang & Zhai (2007), and medical image analysis Guan & Liu (2021).
Despite its practical significance, covariate shift remains relatively underexplored in theoretical frameworks compared to the classical setting where no distribution mismatch is assumed. Recently, several studies have attempted to bridge this gap. A widely adopted approach to addressing covariate shift involves correcting the learning objective by reweighting the loss function using the so-called the applicability of the Nyström method with ALS sampling in the unbounded case. Assumption 2 ensures that the projection of the regression function onto the hypothesis space H exists and lies within H, rather than on its boundary. This is slightly more general than the standard well-specified setting, but conceptually similar. We do not consider the fully misspecified case here, which introduces significant additional complexity (see interpolation spaces in Steinwart & Christmann (2008) for example). Finally, as mentioned in Section 8, a more refined understanding of the role of the constants in the learning rate bounds could explain why, in practice, a misalignment between training and test distributions can sometimes be benign or, conversely, highly adversarial, depending on its relation with the source condition. These directions are left for future work.
Organization. The paper is organized as follows. In Section 2, we introduce the key definitions and notations used throughout the paper, while refreshing KRR in the classical setting. Section 3 provides an overview of KRR in the covariate shift setting. In Section 4, we present empirical risk minimization on random subspaces using Nyström. Section 5 explores the interplay between computational efficiency and statistical accuracy. Section 6 extends our analysis to scenarios where the true IW function is unknown. Section 7 presents a series of simulations and real-world experiments.
this section cite: ['b60', 'b42', 'b29', 'b54', 'b49', 'b11', 'b3', 'b37', 'b10', 'b63', 'b28', 'b55', 'b26', 'b23', 'b52']

Section: Background
We start defining some key quantities we will need in the rest of the paper, see e.g. Caponnetto & De Vito (2007); Smale & Zhou (2007). Given a measurable space X , a probability distribution µ on X , a space of square-integrable functions L 2 µ with respect to measure µ and a Reproducing Kernel Hilbert Space (RKHS) H of (bounded) kernel K : X × X → R, with K x (•) = K(x, •) ∈ H, define
S µ : H → L 2 µ , (S µ f ) (x) = ⟨f, K x ⟩ H = f (x) µ -a.s. and S * µ g = K x g(x)dµ.
for f ∈ H, g ∈ L 2 µ . Covariance operator Σ µ : H → H is defined as
Σ µ := S * µ S µ = E µ [K x ⊗ K x ].
Define the sampling operator S : H → R n associated with set {x 1 , . . . , x n } ∈ X n , for f ∈ H, as ( Sf ) i := f (x i ) = ⟨f, K xi ⟩ H , i ∈ [n], and S * ( y) := 1 n n i=1 y i K xi , y ∈ R n .
this section cite: ['b8', 'b50']

Section: Classical Setting
Classical nonparametric regression aims to predict a real-valued output Y = R given a vector of covariates X ∈ X . More formally, let X × R be a probability space with distribution ρ, where X and R are the input and output spaces, respectively. Let ρ X denote the marginal distribution of ρ on X and ρ(•|x) the conditional distribution on R given x ∈ X . For any fixed x ∈ X , the optimal estimator in a mean-squared sense is given by the regression function g * (x) := ydρ(y|x), i.e. g * = arg min
g∈R X R(g) = E[(Y -g(X)) 2 ].
The minimization problem is generally unsolvable since the distribution ρ is unknown. In practice, we only have access to a dataset of n input-output pairs (x i , y i ) n i=1 ∈ (X × R) sampled independently and identically (i.i.d.) from the joint distribution ρ, i.e. (x 1 , y 1 ), . . . , (x n , y n ) ∼ ρ(x, y).
A common approach to derive an approximation of g * involves restricting the hypothesis space from all measurable functions to a real separable Hilbert space H, and replacing the expected risk R with the (regularized) empirical risk R λ : H → [0, ∞) defined as R λ (f ) := 1 n n i=1 (y i -f (x i )) 2 + λ∥f ∥ 2 H , f ∈ H, λ > 0.
(1)
The (Regularized) Empirical Risk Minimization (ERM) algorithm then solves:
f λ := arg min f ∈H R λ .(2)
When H is an RKHS, as in the rest of this paper, the resulting estimator is known as the kernel ridge regression (KRR) estimator. Our primary goal is to upper bound the excess risk E of f λ , i.e.
E( f λ ) := R( f λ ) -R(g * ) = ∥ f λ -g * ∥ 2 ρ X .
Here, the equality follows from a standard result, see for example Caponnetto & De Vito (2007).
this section cite: ['b8']

Section: Covariate Shift Setting
The covariate shift setting introduces an additional complexity: training and test distributions may differ, but only through their marginals, while sharing the same conditional distribution: x). Since the regression function g * only depends on the conditional distribution, which is identical for both ρ te and ρ tr , it is unique. As in the standard setting, we are provided with n input-output pairs (x i , y i ) n i=1 ∈ (X × R) sampled i.i.d. from ρ tr , i.e. (x 1 , y 1 ), . . . , (x n , y n ) ∼ ρ tr (x, y). The challenge consists in the fact that we train our model using samples from the ρ tr , but we aim to evaluate its performance on new data drawn from ρ te . Then, we want to upper bound the excess risk
ρ te (x, y) = ρ(y|x)ρ te X (x), ρ tr (x, y) = ρ(y|x)ρ tr X(
E( f λ ) = ∥ f λ -g * ∥ 2 ρ te X .
Note that the empirical risk computed from ρ tr samples is a biased estimate of the expected risk under ρ te . As a result, minimizing it may not yield a predictor that performs well on the test distribution.
this section cite: []

Section: Importance-Weighting (IW) Correction
The goal of importance-weighting (IW) correction is to construct an unbiased estimator of the risk with respect to test distribution ρ te , while using data sampled from ρ tr . The idea is to reweight ERM samples based on their relevance to the test distribution, ensuring good performance under ρ te . If ρ te X ≪ ρ tr X , we can define the IW function w representing the weight assigned to a point x ∈ X as the Radon-Nikodym derivative of ρ te X with respect to ρ tr X :
w(x) := dρ te X dρ tr X (x).(3)
Points that are likely to be encountered during testing (ρ te X is large) while are rare to be sampled at training time (ρ tr X is small) are considered particularly relevant and will receive higher weights. In the rest of the paper, we will focus on applying this framework in the context of kernel methods. Assumption 1. H is an RKHS with scalar product ⟨•, •⟩ H and associated kernel K : X × X → R.
We define the regularized importance-weighted empirical risk, for all f ∈ H, as
R w λ (f ) := 1 n n i=1 w(x i )(y i -⟨K xi , f ⟩) 2 + λ∥f ∥ 2 H ,(4)
where H ∋ K x (•) = K(x, •). For the square loss, the minimizer f w λ of eq. ( 4) is given by:
f w λ (x) = ( S * M w S + λI) -1 S * M w y,(5)
where M w is the diagonal matrix with i-th entry w(x i ). In case weights are all positive, we have
f w λ (x) = n i=1 c w i K xi (x) ∈ span{K x1 , . . . , K xn }, c w = ( K + nλ M 1/w ) -1 y ∈ R n , (6
)
where K is the kernel Gram matrix, and M 1/w is the diagonal matrix with i-th entry 1/w(x i ).
Since the regression function g * may not generally belong to H (i.e., the model may be misspecified), we introduce the best approximation f H ∈ H of g * with respect to the L 2 ρ te X distance.
Assumption 2. There exists an f H ∈ H such that
E(f H ) = min f ∈H E(f ) = min f ∈H ∥f -g * ∥ 2 ρ te X .(7)
Note that, while the minimizer might not be unique, we select f H as the unique minimizer with minimal norm (De Vito et al., 2021). In the following, we will evaluate the performance of our estimator relative to the best estimator in H, i.e. f H .
Computations A significant limitation of the procedure outlined above is the computational cost associated with the n × n matrix inversion required to compute the estimator, see ( 6). This operation has a complexity of O(n 3 ) in time and O(n 2 ) in memory, making it impractical when n > 10 5 .
this section cite: []

Section: ERM on Random Subspaces and the Nyström Method
In this paper, we consider an efficient approximation of the above procedure based on considering a subspace B ⊂ H and solving the corresponding importance-weighted regularized ERM problem
min β∈B R w λ (β),(8)
with β w λ the unique minimizer. As clear from ( 6), choosing B = H n = span{K x1 , . . . , K xn } is equivalent to considering the full space H and yields the same solution as in (5). However, a natural alternative is to consider a smaller subspace:
B = H m = span{K x1 , . . . , K xm },(9)
where { x 1 , . . . , x m } ⊂ {x 1 , . . . , x n } is a random subset of the input points and m ≤ n. This is equivalent to Nyström approximation (Williams & Seeger, 2000). A basic approach is to select these points uniformly at random from the training dataset. Alternatively, we can employ more refined sampling techniques, such as using leverage scores (Drineas et al., 2012)
l i (α) = ( K( K + αnI) -1 ) ii , i = 1, . . . , n.(10)
Since in practice computing leverage scores directly can be computationally expensive, approximations ( li (α)) n i=1 have been considered (Drineas et al., 2012;Cohen et al., 2015;Alaoui & Mahoney, 2015). In particular, we consider the following one. Definition 1 (T -approximate leverage scores). Let (l i (α)) n i=1 be the leverage scores associated to the training set for a given α. Let δ > 0, t 0 > 0 and T ⩾ 1. We say that ( li (α)) n i=1 are T -approximate leverage scores with confidence δ, when with probability at least 1 -δ,
1 T l i (α) ⩽ li (α) ⩽ T l i (α), ∀i ∈ {1, . . . , n}, α ⩾ t 0 . (11
)
Given the T -approximate leverage scores for α ⩾ t 0 , ALS sampling proceeds by independently drawing samples x1 , . . . , xm from the training set with replacement, where each point x i is selected with probability Q α (i) = li (α)/ j lj (α). All the results in the next sections are obtained under ALS sampling.
We can now define the Nyström W-KRR problem as follows:
f w λ,m := arg min β∈Hm 1 n M 1/2 w ( y -Sβ) 2 2 + λ ∥β∥ 2 H = arg min f ∈H 1 n M 1/2 w ( y -SP m f ) 2 2 + λ ∥f ∥ 2 H ,(12)
where P m is the orthogonal projection operator onto H m , given by
P m = V V * , see Appendix A.
Taking the derivative and using the first order condition (see Appendix A), the Nyström estimator can be expressed as:
f w λ,m = V (V * S * M w SV + λI) -1 V * S * M w y.(13)
Alternatively, using some linear algebra, it can also be written as:
f w λ (x) = m i=1 c w i K xi (x), c w = ( K T nm M w K nm + nλ K mm ) -1 K T nm M w y,(14)
where
f w λ (x) ∈ span{K x1 , . . . , K xm }, c w ∈ R m , K nm ∈ R n×m , ( K nm ) ij = K(x i , x j ) and K mm ∈ R m×m , ( K mm ) ij = K( x i , x j ) (see derivation in Appendix A).
Computations From eq. ( 14), it is clear that the Nyström method can offer significant computational benefits. Unlike in eq. ( 13), computing our projected estimator only requires O(m 3 + m 2 n) time and O(mn) memory, compared to the previous O(n 3 ) and O(n 2 ). When m ≪ n the difference between the two can be big and efficient implementations such as in Rudi et al. (2017); Meanti et al. (2020) can drastically reduce computational requirements.
this section cite: ['b64', 'b18', 'b18', 'b9', 'b0', 'b39']

Section: Statistical Guarantees
In this section, we aim to derive excess risk bounds for the Nyström estimator presented in eq. ( 13). We begin by introducing the technical assumptions required for the subsequent analysis.
this section cite: []

Section: Further Assumptions
The following assumption, inspired by Gogolashvili et al. (2023), ensures the boundedness of the importance-weighting (IW) function or of its moments. Assumption 3. Let w = dρ te X /dρ tr X be the IW function. There exist constants q ∈ [0, 1], W > 0 and σ > 0 such that ∀p ∈ N, p ⩾ 2
X w(x) p-1 q dρ te X q ⩽ 1 2 p!W p-2 σ 2 , (15
)
where the left-hand side for q = 0 is defined as w p-1 ∞,ρ te X , the ess sup with respect to ρ te X .
Considering the uniformly bounded case ∥w∥ ∞ < ∞, Assumption 3 is satisfied for q = 0. When w is not uniformly bounded, Assumption 3 can still be satisfied for q ∈ (0, 1] if the moments of w are bounded. For example, it is satisfied for q ∈ (0, 1] if W ⩾ 1, σ 2 ⩾ 1 and Gogolashvili et al. (2023) for the detailed result). Equivalently, Assumption 3 can be stated as a condition on the Rényi divergence between ρ te X and ρ tr X (Mansour et al., 2009;Cortes et al., 2010). It follows that this assumption can be interpreted as a requirement for the test distribution ρ te X not to deviate significantly from the train distribution ρ tr X , with q ∈ [0, 1] quantifying the extent of the deviation. Introducing the parameter q is useful for presenting results such as Theorem 1 in a unified form, which can then be specialized to the two cases considered in Corollary 1: uniformly bounded (q = 0) and possibly unbounded (q ̸ = 0) weights. Note that in Ma et al. (2023), a weaker assumption requiring only the second moment to be bounded is considered. The stronger condition in eq. ( 15) is anyway crucial in our case to apply the Nyström method in the unbounded setting (see Appendix B and how the proof differentiates from Rudi et al. (2015)). Assumption 4. The range of the output Y ∈ R is upper bounded, i.e. Y ∈ [-B, B], B < ∞.
2ρ te X x ∈ X : dρ te X dρ tr X (x) ⩾ t ⩽ σ 2 exp -W -1 t 1/q for all t > 0 (see Appendix A in
Furthermore, we make the following regularity assumption, usually known as source condition.
this section cite: ['b22', 'b22', 'b37', 'b12', 'b35', 'b44']

Section: Assumption 5 (Source condition).
There exist 1/2 ⩽ r ⩽ 1 and g ∈ L 2 (X, ρ te X ) with ∥g∥ ρ te ⩽ R for some R > 0 such that f H = L r g, where L := S ρ te S * ρ te is the integral operator.
Assumption 5 and its equivalent formulations (e.g., Assumption 4 in Rudi et al. (2015)) are common in literature (Smale & Zhou, 2007;Caponnetto & De Vito, 2007). r quantifies the smoothness of the target function f H and the extent to which it can be well approximated by functions in H. For r = 1/2, the assumption is always satisfied. Intuitively, a larger r implies f H being smoother.
Finally, we impose an assumption on the capacity of our RKHS, which roughly measures the number of eigenvalues of Σ greater than λ (Zhang, 2005;Caponnetto & De Vito, 2007).
Definition 2 (Effective dimension). For λ > 0, define the random variable
N x (λ) = K x , (Σ + λI) -1 K x H with x ∈ X distributed according to ρ, then N ρ (λ) = E ρ N x (λ) is called effective dimension. Assumption 6 (Capacity condition). ∀λ > 0, there exists 0 ⩽ γ ⩽ 1, Q > 0 s.t. N ρ te X (λ) < Qλ -γ .
It is known that the condition in Assumption 6 is ensured if the eigenvalues (η i ) i of the covariance operator Σ satisfy a polynomial decaying condition η i ∼ i -1/γ (see Appendix C).
this section cite: ['b44', 'b50', 'b8', 'b8']

Section: Excess Risk Bounds
In this section, we present our main theoretical results. We will derive excess risk bounds for the Nyström predictor defined in eq. ( 13) and we will show that Nyström approximation does not affect the state-of-art rates of convergence while instead reducing both time and memory requirements.
We present now the main result of the paper. Theorem 1. Under assumptions 1,2,3,4,5,6, for ALS sampling, let δ > 0, 256(W +σ 2 ) log 2 (4/δ) n 1 γ(1-q)+1+q ⩽ λ ⩽ ∥Σ∥ op , and m ⩾ 144T 2 Qλ -γ log 8n δ , with probability greater or equal than
1 -δ R( f w λ,m ) -R(f H ) 1/2 ⩽ 64B W n √ λ + σ 2 nλ γ(1-q)+q log 8 δ + 43Rλ r .
The detailed proof of Theorem 1 can be found in Appendix B.
If the weighting function w is bounded, i.e. ∥w∥ ∞ < ∞, then Assumption 3 is satisfied for q = 0.
We specify Theorem 1 for this setting against the unbounded one. Corollary 1. Under the assumptions and conditions as in Theorem 1, (a) with Assumption 3 satisfied by q = 0 and choosing λ ≍ (∥w∥
∞ /n) 1 2r+γ , with m ≳ (n/∥w∥ ∞ ) γ 2r+γ log n, with high probability E(f w λ,m ) = ∥ f w λ,m -f H ∥ 2 ρ te X ≲ ∥w∥ ∞ n 2r 2r+γ ,(16)
(b) with Assumption 3 satisfied by q = 1 and choosing λ ≍ n/(W + σ 2 )
-1 2r+1 , with m ≳ (n/(W + σ 2 )) γ 2r+1 log n, with high probability E(f w λ,m ) = ∥ f w λ,m -f H ∥ 2 ρ te X ≲ 1 n 2r 2r+1 .(17)
The rate in eq. ( 16) matches the optimal convergence rate of standard kernel ridge regression (KRR) established in Caponnetto & De Vito (2007). However, it explicitly depends on ∥w∥ ∞ , which can become arbitrarily large as the training and test distributions diverge. This result also recovers Theorem 2.1 from Myleiko & Solodky (2024) for Tikhonov regularization in the special case γ = 1. Compared to their work, we consider ALS sampling, which enables fast rates under a suitable capacity condition. Furthermore, we extend the analysis to the case of unbounded importance weights. Specifically, eq. ( 17) shows that when the weighting function is unbounded (i.e., q = 1 in Assumption 3), the convergence rate deteriorates to O(n -2r 2r+1 ). This slower rate, which does not depend on the capacity assumption 6), is always worse than the rate in eq. ( 16). These findings are consistent with the results reported in Gogolashvili et al. (2023) for the full (non-projected) model. Note that the rate in eq. ( 17) can possibly be improved, even under our Nyström approximation, by clipping the unbounded IW function w at a threshold that depends on n (see Section 5.2.2 in Gogolashvili et al. (2023)). This idea follows Corollary 2 of Ma et al. (2023), although the result is not directly comparable due to differing assumptions. Example 1. To illustrate the benefits of the Nyström approach, consider for example the common setting q = 0, r = 1/2 and γ = 1. From eq. ( 16), we achieve the optimal rate of O(n -1/2 ) (Caponnetto & De Vito, 2007)
, with m = O( √ n log(n)). This results in computational costs of O(m 3 + m 2 n) = O(n √ n + n 2 ) in time and O(mn) = O(n √ n) in memory.
These are significantly lower than the costs of the non-approximated method, respectively O(n 3 ) and O(n 2 ).
this section cite: ['b8', 'b40', 'b22', 'b22', 'b35', 'b8']

Section: Unknown Weights
Clearly, when dealing with real data, assuming knowledge of the exact weighting function w is unrealistic. Instead, if we have access to some samples from both distributions, we can attempt to estimate an approximate weighting function v ≈ w and control the error resulting from this approximation. Let us choose a function v such that there exists a probability distribution ρ v X with
v(x) = dρ v X (x) dρ tr X (x) ,(18)
i.e. v(x) is the Radon-Nikodym derivative of ρ v X with respect to ρ tr X . Note that, in general, ρ v X ̸ = ρ te X . Similarly to eq. ( 7) in Assumption 2, we assume that
f v H = arg min f ∈H E v (f ) = ∥f -g * ∥ 2 ρ v X (19)
exists (again we consider the unique with minimal norm in case of multiple minimizers). Furthermore, we assume that v satisfies Assumptions 3, 5, 6 with constants V, η, r, γ, Q, respectively. Since v is chosen, weights v(x) are known for all x ∈ X , allowing us to compute the estimator
f v λ,m = V (V * S * M v SV + λI) -1 V * S * M v y,(20)
where M v is the diagonal matrix with diagonal entries v(x 1 ), . . . , v(x n ). Still, we are interested in the excess risk of this estimator with respect to the original ρ te distribution. Denoting Σ = Σ ρ te X and
Σ v = Σ ρ v X , we decompose the excess risk of f v λ,m as: E(f v λ,m ) = ∥ f v λ,m -f H ∥ 2 ρ te X ⩽ ∥ΣΣ -1 λv ∥ op ∥ f v λ,m -f v H ∥ 2 ρ v X + ∥f v H -f H ∥ 2 ρ te X . (21
)
Well Specified Case If g * ∈ H, then clearly g * = f H = f v H and eq. ( 21) simplifies to:
∥ f v λ,m -f H ∥ 2 ρ te X ⩽ ∥ΣΣ -1 λv ∥ op ∥ f v λ,m -f H ∥ 2 ρ v X .(22)
The second term corresponds to what we have already analyzed in Theorem 1, with the distribution ρ te X and its associated weighting function w replaced by ρ v X and v. The first term quantifies the additional cost incurred due to the mismatch between ρ te X and ρ v X . Using Proposition 5 from Appendix C, we can show that if we set
v(x) ≡ 1, i.e. ρ v X ≡ ρ tr X , then ∥ΣΣ -1 λv ∥ op ⩽ ∥w∥ ∞ .
This indicates that, when w is bounded, we have a finite control on the appearing term involving the two covariance operators.
By setting λ ≍ (∥w∥ ∞ n) 1 2r+γ
, and ensuring m ≳ n γ 2r+γ log n, then, with high probability
∥ f v≡1 λ,m -f H ∥ 2 ρ te X ≲ ∥w∥ ∞ 1 n 2r 2r+γ ,(23)
where we used the fact that ∥v∥ ∞ = 1. It is important to note that, when compared with the bound in eq. ( 16), which assumes knowledge of the true (typically unknown) weights, we achieve the same rate in n. This means that, when the model is well-specified, the classical Nyström-ERM algorithm gives the same rate as the importance-weighted variant, despite the covariate shift between train and test distributions. This result agrees with findings in Ma et al. (2023); Gogolashvili et al. (2023), where anyway random projection approximations are not involved. More than that, we emphasize that although the rate remains the same, the dependence on ∥w∥ ∞ , which is assumed finite but can be arbitrarily large, is worse than in eq. ( 16), where the true w is employed.
this section cite: ['b35', 'b22']

Section: Misspecified Case
In the misspecified case, the situation is more complex since in general g 21) is not zero, and its magnitude can become arbitrarily large, depending on the severity of the mismatch between ρ te X and ρ v X .
* ̸ = f H ̸ = f v H . In particular, the term ∥f v H -f H ∥ 2 ρ te X in eq. (
this section cite: []

Section: Simulations and real data experiments
As emphasized in the introduction, the main goal of this study is to show that the Nyström method can deliver significant computational savings under covariate shift without compromising accuracy.
this section cite: []

Section: Simulations
We start by reproducing the experimental setting in Gogolashvili et al. (2023). We want to solve a regression problem using KRR with RBF kernel in the context of distribution shift, assuming
ρ tr X ∼ N (µ tr , Σ tr ), ρ te X ∼ N (µ te , Σ te ) and µ tr ̸ = µ te , Σ tr ̸ = Σ te . The regression function is g * (x) := c 1 e - c 2 ∥x∥ 2k 2
, k ∈ N, c ∈ N, where the parameter k controls the level of misspecification. Note that, in fact, when k increases, the regression function becomes essentially piece-wise constant, and neither constant nor discontinuous functions belong to the RKHS of the Gaussian kernel. Data samples are generated following
y tr i = g * (x tr i ) + ξ i , y te i = g * (x te i ), with x tr i ∼ ρ tr X , x te i ∼ ρ te X and ξ i ∼ N (0, ε 2 ).
Figure 1 shows the results in this setting for k = 50. In the plot on the left, we observe that the two weighted models, namely KRR with IW correction (W-KRR) and its Nyström-approximated version (Nyström W-KRR), perform similarly. As expected, the simple (unweighted) KRR model shows a performance gap. On the right, despite reaching the same error of the weighted KRR model, Nyström approximation can lead to important computational savings allowing for choosing a number of Nyström centers m ≪ n.
this section cite: ['b22']

Section: Experiments on Benchmark Datasets
As regards real-world applications, we conduct experiments on commonly used benchmark datasets in the domain adaptation field (Wang & Sun, 2024;Wilson et al., 2020;He et al., 2023;Dinu et al., 2023). The size of the original datasets is reduced in case of memory issues with KRR and W-KRR when building the full Gram matrix K (see memory bottlenecks in Section 4). In these experiments, weights are estimated using RuLSIF method (Yamada et al., 2013;Liu et al., 2013). We consider 4 real-world datasets: HHAR (Stisen et al., 2015), WISDM (Kwapisz et al., 2011), HAR70+ (Ustad et al., 2023) and HARChildren (Tørring et al., 2024). These datasets consist of data collected from multiple users using wearable sensors, such as accelerometers and gyroscopes. To simulate covariate shift, we train each model on data collected from one user and evaluate it on data from a different user. We employ an RBF kernel with length-scale parameter γ and regularization parameter λ, both selected via cross-validation. The results reported in Table 1 are obtained using ALS sampling (see Definition 1), specifically through the BLESS fast implementation described in Rudi et al. (2018). For comparison, Table 2 in Appendix D presents analogous results obtained with uniform sampling of the Nyström centers. Additional details on the datasets and experimental setup are provided in Appendix D. KRR 10 ± 1 1694 ± 2 15.0 ± 0.5 26.6 ± 0.9 762 ± 12 10.2 ± 0.4 3.7 ± 0.3 876 ± 6 10.5 ± 0.9 7.8 ± 0.1 3280 ± 48 38 ± 5 W-KRR 5.0 ± 0.2 1785 ± 2 15.1 ± 0.3 13.5 ± 0.8 809 ± 26 9.0 ± 0.1 1.8 ± 0.3 1034 ± 93 9.9 ± 0.1 4.8 ± 0.2 3364 ± 30 33 ± 2 Ny W-KRR 5.1 ± 0.2 89 ± 19 1.0 ± 0.1 13.2 ± 0.6 8.0 ± 0.2 1.6 ± 0.1 1.8 ± 0.1 6.5 ± 0.4 1.1 ± 0.1 4.7 ± 0.3 9.9 ± 0.4 1.4 ± 0.1
The above Table 1 shows that the two methods using IW correction achieve the best and essentially equal performance. However, in terms of computational efficiency, our Nyström W-KRR method, offers significant time and memory savings. The number of Nyström points m required by Nyström W-KRR is 1100, 1800, 1400, and 1550, for HAR70+, HARChildren, HHAR, and WISDM, respectively.
this section cite: ['b61', 'b66', 'b24', 'b16', 'b34', 'b34', 'b53', 'b32', 'b59', 'b58', 'b46']

Section: Conclusions and Future Work
In this paper, we showed that even under covariate shift, random projection techniques -particularly the Nyström method-can significantly enhance computational efficiency without any loss in learning performance. We provide new statistical bounds for our compressed Nyström algorithm, showing that it matches the optimal statistical guarantees of the full W-KRR model. Leveraging results from random projection theory, we developed novel technical proofs to account for the mismatch between training and test distributions and the potential unboundedness of the IW function. We evaluated the effectiveness of our approach through simulations and experiments on real-world datasets. However, several questions remain open for future investigation. Although optimal rates are achieved in the well-specified case, the misalignment between the training and test distributions relative to the target function (see the source condition in Assumption 5) appears to play a critical role empirically, as it can make covariate shift either benign or severely adversarial. A deeper understanding of this phenomenon may come from a more detailed analysis of the constants in the learning bounds and their influence on the overall rate (see eq. ( 21) and the interaction between the covariance operators of source and target distributions).
Woodruff, D. P. Sketching as a tool for numerical linear algebra. arXiv preprint arXiv:1411.4357, 2014.
Yamada, M., Suzuki, T., Kanamori, T., Hachiya, H., and Sugiyama, M. Relative density-ratio estimation for robust distribution comparison. Neural computation, 25(5):1324-1370, 2013.
Yamazaki, K., Kawanabe, M., Watanabe, S., Sugiyama, M., and Müller, K.-R. Asymptotic bayesian generalization error when training and test distributions are different. In Proceedings of the 24th international conference on Machine learning, pp. 1079-1086, 2007.
Zhang, C., Zhang, L., and Ye, J. Generalization bounds for domain adaptation. Advances in neural information processing systems, 25, 2012.
Zhang, T. Learning bounds for kernel regression using effective data dimensionality. Neural Computation, 17(9):2077-2098, 2005.
NeurIPS Paper Checklist
this section cite: []

Section: References
Ref_id:b0 Title: Fast randomized kernel ridge regression with statistical guarantees Year: (2015)
Ref_id:b1 Title: Sharp analysis of low-rank kernel matrix approximations Year: (2013)
Ref_id:b2 Title: On the equivalence between kernel quadrature rules and random feature expansions Year: (2017)
Ref_id:b3 Title: Analysis of representations for domain adaptation Year: (2006)
Ref_id:b4 Title: Generalized inverses: theory and applications Year: (2006)
Ref_id:b5 Title: The tradeoffs of large scale learning Year: (2008)
Ref_id:b6 Title: Concentration Inequalities: A Nonasymptotic Theory of Independence Year: (2013)
Ref_id:b7 Title: Distributed adaptive sampling for kernel matrix approximation Year: (2017)
Ref_id:b8 Title: Optimal rates for the regularized least-squares algorithm Year: (2007)
Ref_id:b9 Title: Uniform sampling for matrix approximation Year: (2015)
Ref_id:b10 Title: Domain adaptation and sample bias correction theory and algorithm for regression Year: (2014)
Ref_id:b11 Title: Sample selection bias correction theory Year: (2008)
Ref_id:b12 Title: Learning bounds for importance weighting Year: (2010)
Ref_id:b13 Title: Regularization: From inverse problems to large-scale machine learning Year: (2021)
Ref_id:b14 Title: Regularized erm on random subspaces Year: (2021)
Ref_id:b15 Title: The nyström method for convex loss functions Year: (2024)
Ref_id:b16 Title: Addressing parameter choice issues in unsupervised domain adaptation by aggregation Year: (2023)
Ref_id:b17 Title: On the nyström method for approximating a gram matrix for improved kernel-based learning Year: (2005-12)
Ref_id:b18 Title: Fast approximation of matrix coherence and statistical leverage Year: (2012)
Ref_id:b19 Title: Rethinking importance weighting for deep learning under distribution shift Year: (2020)
Ref_id:b20 Title: Norm inequalities equivalent to heinz inequality Year: (1993)
Ref_id:b21 Title: On a regularization of unsupervised domain adaptation in rkhs Year: (2022)
Ref_id:b22 Title: When is importance weighting correction needed for covariate shift adaptation? arXiv preprint Year: (2023)
Ref_id:b23 Title: Domain adaptation for medical image analysis: a survey Year: (2021)
Ref_id:b24 Title: Domain adaptation for time series under feature and label shifts Year: (2023)
Ref_id:b25 Title: Correcting sample selection bias by unlabeled data Year: (2006)
Ref_id:b26 Title: Instance weighting for domain adaptation in nlp Year: (2007)
Ref_id:b27 Title: Accelerating stochastic gradient descent using predictive variance reduction Year: (2013)
Ref_id:b28 Title: Active learning algorithm using the maximum weighted loglikelihood estimator Year: (2003)
Ref_id:b29 Title: Wilds: A benchmark of in-the-wild distribution shifts Year: (2021)
Ref_id:b30 Title: Marginal singularity and the benefits of labels in covariate-shift Year: (2021)
Ref_id:b31 Title: Kernel sketching yields kernel jl Year: (2019)
Ref_id:b32 Title: Activity recognition using cell phone accelerometers Year: (2011)
Ref_id:b33 Title: Near-optimal linear regression under distribution shift Year: (2021)
Ref_id:b34 Title: Change-point detection in time-series data by relative density-ratio estimation Year: (2013)
Ref_id:b35 Title: Optimally tackling covariate shift in rkhs-based nonparametric regression Year: (2023)
Ref_id:b36 Title: Randomized algorithms for matrices and data Year: (2011)
Ref_id:b37 Title: Domain adaptation: Learning bounds and algorithms Year: (2009)
Ref_id:b38 Title: Beyond least-squares: Fast rates for regularized empirical risk minimization through self-concordance Year: (2019)
Ref_id:b39 Title: Kernel methods through the roof: handling billions of points efficiently Year: (2020)
Ref_id:b40 Title: Regularized nyström subsampling in covariate shift domain adaptation problems Year: (2024)
Ref_id:b41 Title: A new similarity measure for covariate shift with applications to nonparametric regression Year: (2022)
Ref_id:b42 Title: Dataset shift in machine learning Year: (2022)
Ref_id:b43 Title: Generalization properties of learning with random features Year: (2017)
Ref_id:b44 Title: Less is more: Nyström computational regularization Year: (2015)
Ref_id:b45 Title: Falkon: An optimal large scale kernel method Year: (2017)
Ref_id:b46 Title: On fast leverage score sampling and optimal learning Year: (2018)
Ref_id:b47 Title: Minimizing finite sums with the stochastic average gradient Year: (2017)
Ref_id:b48 Title: Local convergence rates of the nonparametric least squares estimator with applications to transfer learning Year: (2024)
Ref_id:b49 Title: Improving predictive inference under covariate shift by weighting the log-likelihood function Year: (2000)
Ref_id:b50 Title: Learning theory estimates via integral operators and their approximations Year: (2007)
Ref_id:b51 Title: Sparse greedy matrix approximation for machine learning Year: (2000)
Ref_id:b52 Title: Support vector machines Year: (2008)
Ref_id:b53 Title: Smart devices are different: Assessing and mitigatingmobile sensing heterogeneities for activity recognition Year: (2015)
Ref_id:b54 Title: Machine learning in non-stationary environments: Introduction to covariate shift adaptation Year: (2012)
Ref_id:b55 Title: Active learning in approximately linear regression based on conditional expectation of generalization error Year: (2006)
Ref_id:b56 Title: Density ratio estimation in machine learning Year: (2012)
Ref_id:b57 Title: But how does it work in theory? linear svm with random features Year: (2018)
Ref_id:b58 Title: Validation of two novel human activity recognition models for typically developing children and children with cerebral palsy Year: (2024)
Ref_id:b59 Title: Validation of an activity type recognition model classifying daily physical behavior in older adults: the har70+ model Year: (2023)
Ref_id:b60 Title: The Nature of Statistical Learning Theory Year: (1999)
Ref_id:b61 Title: Domain adaptation of time series classification Year: (2024)
Ref_id:b62 Title: Robust learning under uncertain test distributions: Relating covariate shift to model misspecification Year: (2014)
Ref_id:b63 Title: Robust weights and designs for biased regression models: Least squares and generalized m-estimation Year: (2000)
Ref_id:b64 Title: Using the nyström method to speed up kernel machines Year: (2000)
Ref_id:b65 Title: Using the nyström method to speed up kernel machines Year: (2001)
Ref_id:b66 Title: Multi-source deep domain adaptation with weak supervision for time-series sensor data Year: (2020)
