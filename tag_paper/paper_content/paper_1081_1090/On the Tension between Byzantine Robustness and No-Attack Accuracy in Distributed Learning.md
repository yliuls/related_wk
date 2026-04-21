Title: On the Tension between Byzantine Robustness and No-Attack Accuracy in Distributed Learning
Abstract: Byzantine-robust distributed learning (BRDL), which refers to distributed learning that can work with potential faulty or malicious workers (also known as Byzantine workers), has recently attracted much research attention. Robust aggregators are widely used in existing BRDL methods to obtain robustness against Byzantine workers. However, Byzantine workers do not always exist in applications. As far as we know, there is almost no existing work theoretically investigating the effect of using robust aggregators when there are no Byzantine workers. To bridge this knowledge gap, we theoretically analyze the aggregation error for robust aggregators when there are no Byzantine workers. Specifically, we show that the worst-case aggregation error without Byzantine workers increases with the increase of the number of Byzantine workers that a robust aggregator can tolerate. The theoretical result reveals the tension between Byzantine robustness and noattack accuracy, which refers to accuracy without faulty workers and malicious workers in this paper. Furthermore, we provide lower bounds for the convergence rate of gradient descent with robust aggregators for non-convex objective functions and objective functions that satisfy the Polyak-Łojasiewicz (PL) condition, respectively. We also prove the tightness of the lower bounds. The lower bounds for convergence rate reveal similar tension between Byzantine robustness and noattack accuracy. Empirical results further support our theoretical findings.

Section: Introduction
Distributed learning has been a hot research topic for years (Haddadpour et al., 2019;Jaggi et al., 2014;Lee et al., 2017;Lian et al., 2017;Ma et al., 2015;Shamir et al., 2014;Sun et al., 2018;Yang, 2013;Yu et al., 2019a;b;Zhao et al., 2017;2018;Zhou et al., 2018;Zinkevich et al., 2010). Traditional distributed learning typically assumes no attack or failure. However, in some real-world scenarios such as federated learning (McMahan & Ramage, 2017), the probability of attacks or failure greatly increases due to the server's weak control on workers (Baruch et al., 2019;Kairouz et al., 2021;Xie et al., 2020a). The workers under attack or failure are also called Byzantine workers (Lamport et al., 1982). Byzantine-robust distributed learning (BRDL), which refers to distributed learning that can work with potential Byzantine workers, has recently attracted much research attention (Bernstein et al., 2019;Bulusu et al., 2021;Chen et al., 2018;Damaskinos et al., 2018;Diakonikolas et al., 2017;Diakonikolas & Kane, 2019;Konstantinidis & Ramamoorthy, 2021;Rajput et al., 2019;Sohn et al., 2020;Wu et al., 2020;Yang & Li, 2021;Yang et al., 2024;2020;Yin et al., 2019).
Robust aggregation is a widely used technique to obtain Byzantine robustness in distributed learning. By replacing the vanilla mean aggregator on the server with some robust aggregators, the distributed learning method can be robust against a certain number of Byzantine workers. There are many robust aggregators proposed in existing works (Blanchard et al., 2017;Chen et al., 2017;Guerraoui et al., 2018;Karimireddy et al., 2021;Yin et al., 2018). Meanwhile, using robust aggregation will also inevitably introduce an aggregation error, which is the square distance between the aggregation result and the true mean value. Large aggregation error will lead to a decrease of model accuracy.
Existing analyses (Allouah et al., 2023;Karimireddy et al., 2021;2022) for the aggregation error of robust aggregators typically assume that the number (or fraction) of Byzantine workers is known in advance, while in real-world applications, the number of Byzantine workers is usually unavailable. It is advised in existing works (Karimireddy et al., 2022;Yang & Li, 2023) to determine the maximum number of Byzantine workers that the BRDL method can tolerate in advance and then set the robust aggregator accordingly. However, Byzantine workers do not always exist in applications. When there are actually no Byzantine workers, the aggregation error introduced by robust aggregators could have a negative effect on the convergence of distributed learning methods and degrade the model accuracy. As far as we know, there is almost no existing work theoretically investigating the performance of robust aggregators when there are no Byzantine workers. To bridge this knowledge gap, we mainly focus on the following question:
How large could the aggregation error be when there are actually no Byzantine workers?
The main contributions of our work are listed as follows:
• To the best of our knowledge, this is the first work that theoretically investigates the tension between Byzantine robustness and no-attack accuracy in distributed learning.
• We theoretically prove that the worst-case aggregation error without Byzantine workers increases with the increase of f , which is the number of Byzantine workers that a robust aggregator can tolerate. The theoretical result reveals the tension between Byzantine robustness and no-attack accuracy.
• Moreover, we provide lower bounds for the convergence rate of gradient descent with robust aggregators for non-convex objective functions and objective functions that satisfy the Polyak-Łojasiewicz (PL) condition, respectively. We further show that the lower bounds are tight and increase with the increase of f .
• We also provide empirical results, which further support our theoretical findings.
In addition, we would like to point out that existing works on robust aggregators mainly explore the performance when there exist Byzantine workers while we focus on the case without Byzantine workers in this work.
this section cite: ['b12', 'b14', 'b21', 'b22', 'b24', 'b27', 'b29', 'b36', 'b45', 'b47', 'b48', 'b25', 'b1', 'b15', 'b20', 'b2', 'b4', 'b5', 'b7', 'b9', 'b8', 'b18', 'b26', 'b28', 'b32', 'b37', 'b39', 'b42', 'b3', 'b6', 'b11', 'b16', 'b41', 'b0', 'b16', 'b17', 'b38']

Section: Preliminary
Many distributed learning problems can be formulated as:
min w∈R d F (w) = 1 n n i=1 F i (w),(1)
where w is the model parameter, n is the number of workers and F i (w) is the loss function associated with the training data on the i-th worker. Moreover, we mainly focus on the Parameter Server (PS) framework, where there is an extra server responsible for coordination. A widely used method
Algorithm 1 Byzantine-Robust Gradient Descent (ByzGD)
Input: iteration number T , learning rates {η t } T -1 t=0 , robust aggregator Agg(•); Initialization: model parameter w 0 ; for t = 0 to T -1 do Broadcast w t to all workers; on worker i ∈ {1, . . . , n} in parallel do Compute local gradient g i = ∇F i (w t ); Send g i to the server; end on worker Compute:
w t+1 = w t -η t • Agg(g 1 , . . . , g n ); end for Output: model parameter w T .
to solve the problem (1) is gradient descent as presented below:
w t+1 = w t -η t • 1 n n i=1 g i .
Specifically, at the t-th iteration, all the workers compute local gradients g 1 , . . . , g n in parallel and send them to the server. Then, the server aggregates the gradients with vanilla averaging and updates the model parameter with the aggregated gradient. However, the averaging value is sensitive to the outliers and not robust against Byzantine workers. Therefore, robust aggregators such as Krum (Blanchard et al., 2017), geometric median (Chen et al., 2017) and coordinate-wise trimmed mean (Yin et al., 2018) are proposed to replace vanilla averaging in BRDL. By replacing the vanilla averaging with some robust aggregators, we can obtain Byzantine-robust gradient descent (ByzGD). In ByzGD, the model parameters are updated by:
w t+1 = w t -η t • Agg(g 1 , . . . , g n ),
where Agg(•) denotes a general robust aggregator. More details about ByzGD are presented in Algorithm 1.
In existing works, there are several definitions of robust aggregators that are used to analyze the aggregation error. The definition of (δ max , c)-ARAgg (Karimireddy et al., 2022) is proposed based on the expectation of the distances among the vectors while the definition of (f, λ)-resilient averaging (Farhadkhani et al., 2022) is proposed based on the maximum distances among the vectors. In (Allouah et al., 2023), the definition of (f, κ)-robust aggregator is further proposed, which can unify many existing definitions of robust aggregators including (δ max , c)-ARAgg and (f, λ)resilient averaging. Due to this reason, we mainly follow the definition of (f, κ)-robust aggregator in this work. For simplicity, we use the notation ∥ • ∥ to represent the L 2 -norm in this paper.
Definition 2.1 ((f, κ)-robustness (Allouah et al., 2023)). Let f < n 2 and κ ≥ 0. An aggregator Agg : R d×n → R d is said to be (f, κ)-robust if for any n vectors x 1 , . . . , x n ∈ R d and for any set S ⊆ {1, . . . , n} satisfying |S| = n -f , we have
∥Agg(x 1 , . . . , x n ) -xS ∥ 2 ≤ κ • 1 |S| i∈S ∥x i -xS ∥ 2 , where xS = 1 |S| i∈S x i .
For an (f, κ)-robust aggregator, the square distance between the aggregated result Agg(x 1 , . . . , x n ) and the mean value of non-Byzantine workers xS is bounded. ∥Agg(x 1 , . . . , x n ) -xS ∥ 2 is also known as the aggregation error. Since the server cannot identify Byzantine workers, the inequality should be satisfied for any set S with |S| = n -f , where f is the number of Byzantine workers that the aggregator can tolerate. Specifically, when there are actually no Byzantine workers, the aggregation error is ∥Agg(x 1 , . . . ,
x n ) -1 n n i=1 x i ∥ 2 .
Thus, we propose the definition of ϵ-accuracy in Definition 2.2, which measures the aggregation error when there are no Byzantine workers.
Definition 2.2 (ϵ-accuracy). Let ϵ ≥ 0. An aggregator Agg : R d×n → R d is said to be ϵ-accurate if for any n vectors x 1 , . . . , x n ∈ R d , we have ∥Agg(x 1 , . . . , x n ) -x∥ 2 ≤ ϵ • 1 n n i=1 ∥x i -x∥ 2 , where x = 1 n n i=1 x i .
Comparing Definition 2.1 and 2.2, we can find that ϵaccuracy is equivalent to (f, κ)-robustness when f = 0 and κ = ϵ. However, it is meaningless to say the robustness of an aggregator that can tolerate at most 0 Byzantine worker (f = 0). Thus, for better readability, we present the definition of ϵ-accuracy alone, which is about the performance of an aggregator in the case without Byzantine workers.
this section cite: ['b3', 'b6', 'b41', 'b17', 'b10', 'b0', 'b0']

Section: Analysis of Aggregation Error
In this section, we theoretically analyze the aggregation error of (f, κ)-robust aggregators in the case without Byzantine workers. We only present the main results and proof sketches in this section. Proof details are deferred to Appendix A in the supplementary material due to the limited space.
To avoid confusion, we would like to clarify that f is the number of Byzantine workers that the aggregator can tolerate and there are actually no Byzantine workers in the case that we consider. Specifically, we will focus on how large ϵ (please refer to Definition 2.2) could be in the case without Byzantine workers.
this section cite: []

Section: General Lower Bound
Firstly, we provide a lower bound of ϵ for (f, κ)-robust aggregators.
Theorem 3.1 (Lower Bound). If an (f, κ)-robust aggregator is ϵ-accurate, we have ϵ ≥ f n-f .
The lower bound f n-f in Theorem 3.1 is between 0 and 1 since f < n 2 . Moreover, when the total number of workers n is fixed, the lower bound increases as f increases. In other words, making the aggregator robust to more Byzantine workers will inevitably introduce a larger worst-case aggregation error. We will discuss more about this after proving the tightness of the lower bound in Theorem 3.1.
this section cite: []

Section: General Upper Bound
Then we provide a general upper bound of ϵ for all (f, κ)robust aggregators in Theorem 3.2.
Theorem 3.2. Any (f, κ)-robust aggregator is κ-accurate.
The constant κ is dependent on f and differs for different robust aggregators (Allouah et al., 2023). We present the values of κ for three common (f, κ)-robust aggregators, which are known as coordinate-wise trimmed mean (TM), coordinate-wise median (CM) and geometric median (GM), respectively, and the lower bound of κ in Table 1.
As we can see, the values of κ are much larger than the lower bound f n-f of ϵ. Specifically, for GM and CM, the value of κ is larger than 4 while the lower bound f n-f of ϵ is always smaller than 1 when f < n 2 . Moreover, even the lower bound f n-2f of κ will be much larger than f n-f when f is close to n 2 since the denominator is close to 0. To obtain tighter results, then we separately analyze the aggregation error of each aggregator.
this section cite: ['b0']

Section: Analysis for Specific Aggregators
In this section, we analyze the value of ϵ for TM, CM, and GM. Firstly, we present the definitions of these three common (f, κ)-robust aggregators.
Definition 3.3 (coordinate-wise trimmed mean (Yin et al., 2018)). Let vectors x 1 , . . . , x n ∈ R d . The coordinate-wise trimmed mean TM f /n : R d×n → R d is defined as:
[TM f /n (x 1 , . . . , x n )] j = 1 n -2f x∈Xj x,
where [•] j denotes the j-th coordinate of a vector and the set X j is obtained by removing the f largest and the f smallest values of {[x 1 ] j , . . . , [x n ] j }.
Definition 3.4 (coordinate-wise median (Yin et al., 2018)). Let x 1 , . . . , x n ∈ R d . The coordinate-wise median CM :
ϵ f n-f ⌊ n-1 2 ⌋ n-⌊ n-1 2 ⌋ 1 f n-f R d×n → R d is defined as: [CM(x 1 , . . . , x n )] j = Median([x 1 ] j , . . . , [x n ] j ),
where Median(•) is the scalar median operator.
Definition 3.5 (geometric median). Let x 1 , . . . , x n ∈ R d . The geometric median GM : R d×n → R d is defined as:
GM(x 1 , . . . , x n ) ∈ arg min x∈R d n i=1 ∥x -x i ∥.
(
In other words, the geometric median is the vector that minimizes the sum of L 2 -distances. When the vectors x 1 , . . . , x n are not collinear, the optimization problem in the right-hand side of (2) has a unique solution (Vardit & Zhang, 2000).
The values of ϵ for the three (f, κ)-robust aggregators are summarized in Table 2. Due to the limited space, the analysis details for TM, CM and GM are deferred to Appendix A.3, Appendix A.4 and Appendix A.5, respectively. Moreover, please note that the value of ϵ for the coordinatewise trimmed mean TM f /n exactly meets the general lower bound in Theorem 3.1. It indicates that the value of ϵ for TM f /n and the general lower bound are both tight.
this section cite: ['b41', 'b41', 'b30']

Section: Conditions of the Worst Case
We have obtained that the value of ϵ can be up to f n-f in the worst case. Then we would like to focus on the following question:
Under what conditions is the aggregation error largest?
To answer the question above, we would like to first present the theoretical result and a proof sketch for the aggregation error of TM in Theorem 3.6 below. The conditions of the worst case can be derived based on the proof of Theorem 3.6.
Theorem 3.6. When f < n 2 , the coordinate-wise trimmed mean TM f /n is f n-f -accurate.
this section cite: []

Section: PROOF SKETCH OF THEOREM 3.6
We first consider the case where the dimension d = 1. Let x 1 , . . . , x n be any n real numbers. Without loss of generality, we assume that x 1 ≤ x 2 ≤ . . . ≤ x n since the order does not affect the trimmed mean of the n values. Furthermore, we let sets I -= {1, 2, . . . , f }, I 0 = {f + 1, . . . , n -f } and I + = {n -f + 1, . . . , n}, and define
xI-= 1 |I -| i∈I- x i = 1 f f i=1 x i ,(3)
xI0 = 1 |I 0 | i∈I0 x i = 1 n -2f n-f i=f +1 x i ,(4)
xI+ = 1 |I + | i∈I+ x i = 1 f n i=n-f +1 x i .(5)
According to the definition,
x = f n xI-+ n -2f n xI0 + f n xI+ , TM f /n (x 1 , . . . , x n ) = xI0 .
Thus, we have
TM f /n (x 1 , . . . , x n ) - x = f n (x I0 -xI-) + f n (x I0 -xI+ ) = f n ∆ -- f n ∆ + , (6
)
where
∆ -= xI0 -xI-≥ 0,and
∆ + = xI+ -xI0 ≥ 0.
Moreover, we define the within-group variances
s 2 I-= 1 f i∈I- (x i -xI-) 2 , s 2 I0 = 1 f i∈I0 (x i -xI0 ) 2 , s 2 I+ = 1 f i∈I+ (x i -xI+ ) 2 ,
and we have the following equation
1 n n i=1 (x i -x) 2 = n -f f TM f /n (x 1 , . . . , x n ) -x 2 + 2f n (∆ -∆ + ) + f n • s 2 I-+ n -2f n • s 2 I0 + f n • s 2 I+ .(7)
For space saving, the proof details of ( 7) are deferred to Appendix A.3. Noticing that
s 2 I-≥ 0, s 2 I0 ≥ 0, s 2 I+ ≥ 0, ∆ -≥ 0 and ∆ + ≥ 0, we have that 1 n n i=1 (x i -x) 2 ≥ n-f f TM f /n (x 1 , . . . , x n ) -x 2 , or, equivalently, TM f /n (x 1 , . . . , x n ) -x 2 ≤ f n -f • 1 n n i=1 (x i -x) 2 . (8) For general d-dimension cases, ∥TM f /n (x 1 , . . . , x n )-x∥ 2
and ∥x i -x∥ 2 can be written as summations of square errors on each dimension. Recursively using inequality (8) for 1dimension case above, we finally obtain that
∥TM f /n (x 1 , . . . , x n )-x∥ 2 ≤ f n -f • 1 n n i=1 ∥x i -x∥ 2 . (9) It implies that TM f /n is f n-f -accurate.
this section cite: []

Section: DISCUSSION ON THE WORST CASE
By comparing Equation ( 7) and Inequality (8), we can find that the equation in ( 8) holds if and only if
2f n (∆ -∆ + ) + f n • s 2 I-+ n -2f n • s 2 I0 + f n • s 2 I+ = 0,
which is equivalent to that s 2 I-= s 2 I0 = s 2 I+ = 0 and ∆ -∆ + = 0. Without loss of generality, we assume that ∆ + = 0. Recall that s 2 I-, s 2 I0 and s 2 I+ are the within-group variances of {x i } i∈I-, {x i } i∈I0 and {x i } i∈I+ , respectively. Therefore, we have
x 1 = x 2 = . . . = x f and x f +1 = . . . = x n . (10)
In summary, for 1-dimension case, in the case described by Equation (10), the aggregation error is the largest. For the general d-dimension case, the aggregation error is the largest when Equation (10) holds for each dimension.
Moreover, please note that the condition described by Equation (10) indicates a large skewness among {x 1 , . . . , x n }. Thus, our theoretical results show that a large data skewness will lead to a large error of robust aggregation even if there is actually no Byzantine workers.
this section cite: []

Section: More Discussion

this section cite: []

Section: ABOUT THE TIGHTNESS OF THE BOUNDS
As we can see, the value of ϵ for TM f /n exactly meets the lower bound f n-f . Thus, both the upper bound for TM f /n and the lower bound are tight. Moreover, since coordinatewise median (CM) is robust against f = ⌊ n-1 2 ⌋ Byzantine workers, the value of ϵ for CM also meets the lower bound.
Then we show the tightness of the value of ϵ for GM. Consider the case where n = 2n ′ is an even number and the
dimension d = 1. Moreover, x 1 = x 2 = . . . = x n ′ = 0 and x n ′ +1 = x n ′ +2 = . . . = x 2n ′ = 1. In this case, x = 1 2 and 1 n n i=1 ∥x -x i ∥ 2 = 1 n n i=1 1 4 = 1
4 . The geometric median x GM can be any value in [0, 1]. When x GM = 0 or 1, we have that ∥x GM -x∥ 2 = 1 4 . Therefore, in this case,
∥x GM -x∥ 2 = 1 4 = 1 • 1 n n i=1 ∥x -x i ∥ 2 .
Therefore, the value ϵ = 1 for GM in Table 2 is tight. Moreover, the value of ϵ for GM asymptotically meets the lower bound since GM is robust against f = ⌊ n-1 2 ⌋ Byzantine workers and
lim n→∞ ⌊ n-1 2 ⌋/(n -⌊ n-1 2 ⌋) = 1.
this section cite: []

Section: ABOUT DATA HETEROGENEITY
Due to the tightness of the lower bound, the aggregation error in the worst case is proportional to 1 n n i=1 ∥x i -x∥ 2 , which could be large when the probability distributions of x i 's are heterogeneous. Therefore, the aggregation error could be larger when data distribution are more heterogeneous. There are some existing techniques to reduce the gradient (or momentum) heterogeneity in BRDL such as bucketing (Karimireddy et al., 2022) and nearest neighbour mixing (NNM) (Allouah et al., 2023). However, both of the two techniques requires the prior knowledge of the maximum number of Byzantine workers to mix the input vectors. Less Byzantine workers can be tolerated if we want to make the vectors more homogeneous after mixing.
From another perspective, an aggregator combined with NNM (or bucketing) can be considered as a new aggregator that can resist fewer Byzantine workers but have less aggregation error. Therefore, when using NNM or bucketing, we actually use the prior knowledge of the Byzantine worker number to make a better trade-off between robustness and accuracy. Please see Section 5 for the empirical results, which will further support the theoretical findings in this section.
this section cite: ['b17', 'b0']

Section: SUMMARY OF THEORETICAL RESULTS
In this section, we theoretically show that for any (f, κ)robust aggregator, when there are no Byzantine workers, it is inevitably to have an aggregation error in the order of O( f n-f ) in the worst case. Moreover, the value of f n-f increases as f increases. It shows that to obtain robustness against more Byzantine workers, the worst-case performance in no-attack cases is inevitably degraded, which we deem as the tension between Byzantine robustness and no-attack accuracy.
this section cite: []

Section: Convergence
In this section, we theoretically analyze the convergence of ByzGD with (f, κ)-robust aggregators when there are actually no Byzantine workers. Firstly, we list the assumptions below.
Assumption 4.1 (Bounded loss). ∃F * ∈ R such that ∀w ∈ R d , F (w) ≥ F * . Assumption 4.2 (L-smoothness). ∀w, w ′ ∈ R d , ∥∇F (w) -∇F (w ′ )∥ ≤ L∥w -w ′ ∥. Assumption 4.3 (Bounded heterogeneity). There exists G ≥ 0 such that ∀w ∈ R d , 1 n n i=1 ∥∇F i (w) -∇F (w)∥ 2 ≤ G 2 . Assumption 4.4 (Polyak-Łojasiewicz (PL) condition). Let F * denote the lower bound of F (w). ∃µ > 0 such that ∀w ∈ R d , F (w) -F * ≤ 1 2µ ∥∇F (w)∥ 2 .
The four assumptions above are common in distributed learning (Allouah et al., 2023;Blanchard et al., 2017;Farhadkhani et al., 2022;Karimireddy et al., 2021;2022;Xie et al., 2019;2020b). Specifically, the value of G measures the heterogeneity of local gradients on workers. A larger G typically makes the distributed learning task more challenging. Under the assumptions, we have the following convergence result for ByzGD. Theorem 4.5 (Lower bound). For ByzGD (Algorithm 1) with any (f, κ)-robust aggregator, there exist n loss functions F 1 (w), . . . , F n (w) satisfying Assumptions 4.1, 4.2, 4.3 and 4.4, and satisfying the following condition: For any initial point w 0 , any positive constant learning rate η t = η > 0, any constant C 1 < f n-f G 2 and any positive integer K > 0, there exists an integer T > K such that
1 T T -1 t=0 ∥∇F (w t )∥ 2 > C 1 ,and
F (w T ) -F * > C 1 2µ .
Due to the limited space, we only provide a proof sketch of Theorem 4.5 here. Consider the 1-dimension case where
F 1 (w) = . . . = F f (w) = nG 2 f (n -f ) w 2
and
F f +1 (w) = . . . = F n (w) = nG 2 f (n -f ) (w 2 -2w).
It can be verified that F (w) = 1 n n i=1 F i (w) has the unique global mimimum point w * = n-f n and satisfies Assumptions 4.1, 4.2, 4.3 and 4.4. Moreover, the updating rule of ByzGD for this case can be written as:
w t+1 -1 = 1 - nGη f (n -f ) (w t -1).
By separately analyzing the three cases where (i
) 0 < η < 2 √ f (n-f ) nG ; (ii) η = 2 √ f (n-f ) nG
; and (iii) η > In general cases where ∇F i (w)'s are not all the same, we have G > 0. Theorem 4.5 indicates that ByzGD with a (f, κ)-robust aggregator cannot guarantee the convergence to 0. Moreover, the term f n-f increases with the increase of f , where f is the number of Byzantine workers that can be tolerated. Therefore, the convergence lower bound will be larger if more Byzantine workers can be tolerated, which we consider as the tension between Byzantine robustness and no-attack accuracy. Then we show that the constant terms L and Agg(•) is ϵ-accurate, we have that
1 T T -1 t=0 ∥∇F (w t )∥ 2 ≤ 2L[F * -F (w T )] T + ϵG 2 .
Furthermore, when F (w) also satisfies the PL condition (Assumption 4.4), we have that
F (w T ) -F * ≤ 1 - µ L T [F (w 0 ) -F * ] + ϵG 2 2µ .
The terms 2L[F * -F (w T )]
T and (1 -µ L ) T [F (w 0 ) -F * ] in Theorem 4.6 approach 0 when T → ∞, while the constant terms ϵG 2 and ϵG 2 2µ remain. Please note that ϵ ≥ f n-f (Theorem 3.1) and that TM f /n is both (f, κ)-robust and ϵ-accurate with ϵ = f n-f . It indicates that the lower bound f n-f G 2 for C 1 in Theorem 4.5 cannot be improved. Therefore, the constant terms in Theorem 4.5 and those in Theorem 4.6 are both tight.
In this section, we mainly analyze the convergence of ByzGD. In some large-scale problems, the computation of local gradients is time-consuming. Some variants such as Byzantine-robust stochastic gradient descent (ByzSGD) has a much lower computation cost of each iteration and is more widely used. We would like to point out that the lower bound in Theorem 4.5 can be directly applied to ByzSGD since ByzGD can be considered as a special case of ByzSGD with a zero variance. Local momentum (Allouah et al., 2023;Farhadkhani et al., 2022;Karimireddy et al., 2021) is also proposed to reduce the variance of stochastic gradients in ByzSGD. However, the constant terms ϵG 2 and ϵG 2 2µ are related to the aggregation accuracy ϵ and the heterogeneity degree G while using stochastic gradients or local momentums does not help to reduce ϵ or G. Stochastic gradients can reduce the computation cost of each iteration, but do not have a better worst-case convergence guarantee. Therefore, for these variants, there are similar tensions between Byzantine robustness and no-attack accuracy, which we will verify by empirical results.
In real-world applications, the number of Byzantine workers is usually unknown, and it is typically required to determine in advance the number f of Byzantine workers that the distributed learning system can tolerate. A too small f will make the BRDL method easy to be foiled by Byzantine workers. However, a too large f may also lead to a large aggregation error, and thus degrade the model accuracy, even if there are actually no Byzantine workers. Thus, f should be carefully estimated before designing the distributed learning system in real-world applications.
this section cite: ['b0', 'b3', 'b10', 'b16', 'b33', 'b0', 'b10', 'b16']

Section: Experiment
In this section, we will empirically test the effect of using robust aggregator when there are no Byzantine workers. Specifically, we use ByzSGD with various robust aggregators to train a ResNet-20 (He et al., 2016) deep learning model on the CIFAR-10 dataset (Krizhevsky et al., 2009) for 160 epochs without attacks. All the experiments are conducted on a distributed platform with 16 Docker containers serving as workers and an extra Docker container as the server. Each Docker container is bound to an NVIDIA TITAN Xp GPU. We test the performance of each method when the training instances are randomly distributed to the workers according to the Dirichlet distribution with hyperparameters α = 0.1, 1.0 and 10.0, respectively. A smaller α will lead to a more heterogeneous data distribution. Moreover, the batch normalization (BN) layers in the ResNet-20 model are replaced with group normalization layers since BN layers have a poor performance with heterogeneous data across workers (Wu & He, 2018). All algorithms are implemented with PyTorch 1.3.
We use cross-entropy as the loss function, set the batch size on each worker to 16, and use the cosine annealing learning rates (Loshchilov & Hutter, 2017). Specifically, the learning rate at the p-th epoch is η p = 1+cos(pπ/160) 2 η 0 for p = 0, 1, . . . , 159. The initial learning rate η 0 is selected from {0.1, 0.2, 0.5, 1.0}, and the best final top-1 test accuracy is used as the final metrics. Local momentum is used with momentum hyper-parameter set to 0.9. We first test the performance of using multi-Krum (Blanchard et al., 2017) and coordinate-wise trimmed mean (Yin et al., 2018) when hyper-parameter f , which is the number of the Byzantine workers that can be tolerated, ranges from 0 to 7. It takes about 1.5 hours to run each method for 160 epochs.
As the results in Table 3 and Table 4 show, for both multi-Krum and coordinate-wise trimmed mean, the final top-1 test accuracy decreases as f increases. In other words, to make the BRDL method robust to more Byzantine workers, the model accuracy under no attack will decrease. Furthermore, the final top-1 test accuracy decreases more rapidly with a smaller α. Please note that α is the hyper-parameter related to the data distribution. A smaller α indicates a more heterogeneous data distribution and a larger G (please refer to Assumption 4.3). Thus, the final top-1 test accuracy decreases more rapidly when G is larger, which is consistent with the theoretical results in Theorem 4.5 and Theorem 4.6.
We also test the performance of multi-Krum and coordinate-wise trimmed mean with nearest neighbour mixing (NNM) (Allouah et al., 2023). The NNM technique can reduce the heterogeneity of vectors. Specifically, we fix f = 7 (the maximum number of Byzantine workers that can be tolerated) for both of the two robust aggregators and let the hyper-parameter f NNM range from 1 to 7 for NNM. A smaller f NNM makes the vectors after mixing more homogeneous but will also decrease the number of Byzantine workers that the whole BRDL method can tolerate. Specifically, the whole BRDL method is robust to min{f, f NNM } Byzantine workers. When f NNM = 0, the output vectors of NNM will be all the same and equal the mean of the input vectors. Thus, when f NNM = 0, robust aggregator combined with NNM is equivalent to vanilla mean aggregator.
As the results in Table 5 and Table 6 show, using NNM alleviates the degrade of final top-1 test accuracy for both multi-Krum and coordinate-wise trimmed mean. Moreover, using a smaller f NNM can lead to a higher final top-1 test accuracy, but will also make the distributed learning method robust to less Byzantine workers. Therefore, there is also a tension between Byzantine robustness and no-attack accuracy when using the NNM technique.
this section cite: ['b13', 'b19', 'b31', 'b23', 'b3', 'b41', 'b0']

Section: Conclusion
To the best of our knowledge, this is the first work that theoretically investigates the tension between Byzantine robustness and no-attack accuracy in distributed learning. We theoretically analyze the aggregation error of robust aggregators and the convergence rate of using gradient descent with robust aggregators. The theoretical results show that making the distributed learning method robust to more Byzantine workers will degrade the worst-case performance under no attack. Our theoretical findings are further supported by the empirical results.
this section cite: []

Section: References
Ref_id:b0 Title: Fixing by mixing: A recipe for optimal byzantine ml under heterogeneity Year: (2023)
Ref_id:b1 Title: A little is enough: Circumventing defenses for distributed learning Year: (2019)
Ref_id:b2 Title: signSGD with majority vote is communication efficient and fault tolerant Year: (2019)
Ref_id:b3 Title: Machine learning with adversaries: Byzantine tolerant gradient descent Year: (2017)
Ref_id:b4 Title: Byzantine resilient non-convex scsg with distributed batch gradient computations Year: (2021)
Ref_id:b5 Title: Byzantine-resilient distributed training via redundant gradients Year: (2018)
Ref_id:b6 Title: Distributed statistical machine learning in adversarial settings: Byzantine gradient descent Year: (2017)
Ref_id:b7 Title: Asynchronous Byzantine machine learning (the case of SGD) Year: (2018)
Ref_id:b8 Title: Recent advances in algorithmic high-dimensional robust statistics Year: (2019)
Ref_id:b9 Title: Being robust (in high dimensions) can be practical Year: (2017)
Ref_id:b10 Title: Byzantine machine learning made easy by resilient averaging of momentums Year: (2022)
Ref_id:b11 Title: The hidden vulnerability of distributed learning in byzantium Year: (2018)
Ref_id:b12 Title: Trading redundancy for communication: Speeding up distributed SGD for non-convex optimization Year: (2019)
Ref_id:b13 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b14 Title: Communication-efficient distributed dual coordinate ascent Year: (2014)
Ref_id:b15 Title: Advances and open problems in federated learning Year: (2021)
Ref_id:b16 Title: Learning from history for Byzantine robust optimization Year: (2021)
Ref_id:b17 Title: Byzantine-robust learning on heterogeneous datasets via bucketing Year: (2022)
Ref_id:b18 Title: Byzshield: An efficient and robust system for distributed training Year: (2021)
Ref_id:b19 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b20 Title: The byzantine generals problem Year: (1982)
Ref_id:b21 Title: Distributed stochastic variance reduced gradient methods by sampling extra data with replacement Year: (2017)
Ref_id:b22 Title: Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent Year: (2017)
Ref_id:b23 Title: Sgdr: Stochastic gradient descent with warm restarts Year: (2017)
Ref_id:b24 Title: Adding vs. averaging in distributed primaldual optimization Year: (2015)
Ref_id:b25 Title: Federated learning: Collaborative machine learning without centralized training data Year: (2017)
Ref_id:b26 Title: Detox: A redundancy-based framework for faster and more robust gradient aggregation Year: (2019)
Ref_id:b27 Title: Communicationefficient distributed optimization using an approximate newton-type method Year: (2014)
Ref_id:b28 Title: Election coding for distributed learning: Protecting signsgd against Byzantine attacks Year: (2020)
Ref_id:b29 Title: Slimdp: a multi-agent system for communication-efficient distributed deep learning Year: (2018)
Ref_id:b30 Title: The multivariate l1-median and associated data depth Year: (2000)
Ref_id:b31 Title: Group normalization Year: (2018)
Ref_id:b32 Title: Federated variance-reduced stochastic gradient descent with robustness to Byzantine attacks Year: (2020)
Ref_id:b33 Title: Distributed stochastic gradient descent with suspicion-based fault-tolerance Year: (2019)
Ref_id:b34 Title: Fall of empires: Breaking Byzantine-tolerant sgd by inner product manipulation Year: (2020)
Ref_id:b35 Title: Robust fully asynchronous SGD Year: (2020)
Ref_id:b36 Title: Trading computation for communication: Distributed stochastic dual coordinate ascent Year: (2013)
Ref_id:b37 Title: Buffered asynchronous SGD for Byzantine learning Year: (2021)
Ref_id:b38 Title: Buffered asynchronous SGD for Byzantine learning Year: (2023)
Ref_id:b39 Title: On the effect of batch size in byzantine-robust distributed learning Year: (2024)
Ref_id:b40 Title: Adversary-resilient distributed and decentralized statistical inference and machine learning: An overview of recent advances under the Byzantine threat model Year: (2020)
Ref_id:b41 Title: Byzantinerobust distributed learning: Towards optimal statistical rates Year: (2018)
Ref_id:b42 Title: Defending against saddle point attack in Byzantine-robust distributed learning Year: (2019)
Ref_id:b43 Title: On the linear speedup analysis of communication efficient momentum SGD for distributed non-convex optimization Year: (2019)
Ref_id:b44 Title: Parallel restarted SGD with faster convergence and less communication: Demystifying why model averaging works for deep learning Year: (2019)
Ref_id:b45 Title: SCOPE: scalable composite optimization for learning on spark Year: (2017)
Ref_id:b46 Title: Proximal SCOPE for distributed sparse learning Year: (2018)
Ref_id:b47 Title: Distributed proximal gradient algorithm for partially asynchronous computer clusters Year: (2018)
Ref_id:b48 Title: Parallelized stochastic gradient descent Year: (2010)
