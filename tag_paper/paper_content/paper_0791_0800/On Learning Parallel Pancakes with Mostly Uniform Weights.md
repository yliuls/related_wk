Title: On Learning Parallel Pancakes with Mostly Uniform Weights
Abstract: We study the complexity of learning k-mixtures of Gaussians (k-GMMs) on R d . This task is known to have complexity d Ω(k) in full generality. To circumvent this exponential lower bound on the number of components, research has focused on learning families of GMMs satisfying additional structural properties. A natural assumption posits that the component weights are not exponentially small and that the components have the same unknown covariance. Recent work gave a d O(log(1/wmin)) -time algorithm for this class of GMMs, where w min is the minimum weight. Our first main result is a Statistical Query (SQ) lower bound showing that this quasi-polynomial upper bound is essentially best possible, even for the special case of uniform weights. Specifically, we show that it is SQ-hard to distinguish between such a mixture and the standard Gaussian. We further explore how the distribution of weights affects the complexity of this task. Our second main result is a quasi-polynomial upper bound for the aforementioned testing task when most of the weights are uniform while a small fraction of the weights are potentially arbitrary.

Section: Introduction
Learning mixture models in high dimensions is a classic and fundamental task with applications in a plethora of domains, such as bioinformatics, astrophysics, and marketing (Lindsay, 1995;García-Escudero et al., 2010); see Titterington et al. (1985) for an extensive list of applications. The prototypical case is that of Gaussian Mixture Models (GMMs) which is one of the most studied problems in statistics and machine learning, with a large body of research over the past few decades, e.g., Vempala & Wang (2002); Kannan et al. (2005); Achlioptas & McSherry (2005) -see Appendix A for a detailed literature review.
The setup is as follows: the learning algorithm observes i.i.d. samples from a k-component GMM model (k-GMM) in R d , P = k i=1 w i N (µ i , Σ i ), and the goal is to either learn the mixture in total variation distance, learn its parameters, or cluster samples from the GMM correctly. The first task is known to be information-theoretically feasible with poly(d, k) samples, as are the second and third, provided the components are sufficiently well-separated, however known algorithms often require more. While the particular case of spherical mixtures (i.e., Σ i = I) can be learned in poly(d, k) time and samples, (Liu & Li, 2022;Diakonikolas & Kane, 2024), the best-known algorithms for learning arbitrary GMMs (i.e. with arbitrary weights, and arbitrary and different component covariances) have sample complexity that scales with d O(k)  (Bakshi et al., 2022). In this paper, we are concerned with an intermediate regime between the two extremes, where the components share an unknown but common covariance matrix. Diakonikolas et al. (2017) showed that for such mixtures, any sub-exponential time algorithm in the Statistical Query (SQ) model requires a sample complexity of d Ω(k) . The SQ model consists of algorithms that, instead of drawing samples from the data distribution, make queries to approximate expectations of bounded functions (formally defined in Definition 1.2).
The hard instances they proposed are "parallel pancakes" GMMs-mixtures of pairwise-separated Gaussians whose component means are collinear along an unknown direction v, with arbitrary variance in the v-direction and identity covariance in the orthogonal subspace. This will be formally defined in Problem 1.1. Bruna et al. (2021); Gupte et al. (2022) further extended the hardness result to general algorithms but under cryptographic assumptions; similar hardness results were also shown for sum-of-squares algorithms (Diakonikolas et al., 2024).
While these results together might suggest that k-GMM learning is fully understood algorithmically, the current theory remains unsatisfactory, in the following sense: the hard instances developed in Diakonikolas et al. (2017) have rather ill-conditioned mixing weights-some of the mixing weights are 1/ poly(k), but others can be as small as 2 -k . A natural question then is: is it possible to improve the complexity of learning algorithms when all weights are more naturally conditioned, i.e., w i ≥ 1/ poly(k) for all i?
This question was considered in Buhai & Steurer (2023); Anderson et al. (2024), which study GMMs that have a minimum mixing weight w min ≥ 1/ poly(k) and unknown but common covariance across components. Under the assumption that the mixture components are separated in total variation distance, they provide an algorithm that can correctly cluster 99% of the points, using time and sample complexity d log(1/wmin) ≤ d O(log k) . In particular, their results apply to parallel pancake instances, showing that it is possible to circumvent the d Ω(k) (SQ) lower bound under mixing weight assumptions.
These prior results on learning mixtures with restricted mixing weights serve as motivation and the starting point of the present work. In particular, the first question we study is: Is it possible to substantially improve the algorithm of Anderson et al. (2024) to a poly(d, k) time algorithm for parallel pancakes when each w i ≥ 1/ poly(k)?
Our first main result rules out this possibility for SQ algorithms. Specifically, we show in Theorem 1.3 that even when the mixing weights are uniform, any SQ algorithm for such instances requires d Ω(log k) complexity. In fact, the lower bound holds even for the more basic task of distinguishing between a k-GMM from that family and N (0, I).
Our second question stems from the fact that the algorithm in Anderson et al. (2024) has complexity d log (1/wmin) , meaning that a single point with arbitrarily small weight (e.g., 2 -k ) can result in d k complexity.
What is the correct complexity dependence on w min for learning k-component parallel pancakes?
Specifically, we consider again the testing problem of distinguishing between a k-parallel-pancake GMM and N (0, I), but where k ′ ≤ k components can have arbitrary weights while the remaining k -k ′ points must have uniform weights. We show that this mixing weight restriction implies that the testing problem can be solved with time and sample (kd) O(k ′ +log k) + (log k)/w min -an inverse-linear dependence on w min instead of quasi-polynomial as suggested by the Anderson et al. (2024) result. While this testing upper bound does not imply a general learning algorithm k-GMM, it serves as a first step in understanding the nuances of the computational landscape of GMMs with respect to the assumptions on the mixing weights.
The technical core for both our main results is to deter-mine the maximum number m of moments that k-parallelpancake GMMs can match with N (0, I). Our SQ lower bound (Theorem 1.3) comes from showing that m = Ω(log k), by employing a result from design theory. Our second, algorithmic result (Theorem 1.4) critically builds on an impossibility-of-moment-matching argument (Proposition 4.1), showing that if there are k ′ ≤ k arbitrary weights in the k-GMM, then m must be O(log(k) + k ′ ). We show this through a novel proof strategy that bounds the ratio of expectations of appropriately chosen non-negative polynomials that vanish on the points with arbitrary weights.
this section cite: ['b34', 'b24', 'b44', 'b45', 'b29', 'b0', 'b35', 'b15', 'b16', 'b25', 'b16', 'b11', 'b1', 'b1', 'b1', 'b1']

Section: Our Results
We first formally state the hypothesis testing problem which requires the algorithm to distinguish between a k-parallel pancake and the standard Gaussian N (0, I). Problem 1.1 (Parallel Pancakes Testing Problem). One has (i.i.d. sample or SQ) access to a distribution D where either:
• (Null Hypothesis) D = N (0, I).
• (Alternative Hypothesis) D is a Gaussian mixture of the form i∈[k] w i N (vµ i , I -δvv ⊤ ), for some unit vector v ∈ S d-1 , centers µ i ∈ R, and weights w i ≥ 0 for i ∈ [k], with i∈[k] w i = 1. That is, D is a k-GMM with collinear centers and variance 1 -δ along the direction of the centers and 1 in every orthogonal direction.
The goal is to distinguish between the two cases.
Before presenting our first main result, we recall the definition of SQ algorithms. These algorithms, instead of directly accessing samples, query expectations of bounded functions of the distribution. The SQ model, introduced in (Kearns, 1998), has since been extensively studied in various contexts (Feldman, 2016). Many supervised learning algorithms, and several known machine learning techniques are implementable using SQs (Feldman et al., 2017a;b).
Definition 1.2 (STAT Oracle). Let D be a distribu- tion on R d . A statistical query is a bounded function f : R d →[-1, 1]. Given f and an accuracy parameter τ > 0, STAT(τ ) returns a v ∈ R such that |v -E x∼D [f (x)]| ≤ τ .
Since a call to STAT(τ ) can be simulated in the standard PAC model by averaging 1/τ 2 samples, τ serves as the SQ model's analog to sample complexity. An informationcomputation tradeoff in the SQ model states that any SQ algorithm for a given problem must either make a large number of queries or at least one query with very fine accuracy (which informally implies a tradeoff between sample complexity and runtime in the standard PAC model). We are now ready to state our first main result. Theorem 1.3 (SQ Lower Bound for Uniform Weights). Let C be a sufficiently large absolute constant, k > C and d ≥ (log k log d) 2 be integers. If we further restrict the alternative hypothesis in Problem 1.1 to have w i = 1/k for all i ∈ [k], any SQ algorithm requires either 2 d Ω(1) queries or at least one query of accuracy d -Ω(log k) .
Remarks Buhai & Steurer (2023); Anderson et al. (2024) presented an algorithm for solving Problem 1.1 using d O(log k) time and samples (e.g., Theorem 1.1 in the first paper, which was the first to achieve this). Our Theorem 1.3 shows that this complexity is best possible. Notably their work requires the components to be statistically separated, but this is something that we can also ensure by taking δ sufficiently small (since δ does not affect the complexity lower bound).
We now move to our second main result.
this section cite: ['b30', 'b21', 'b34', 'b11', 'b1']

Section: Theorem 1.4 (Testing Algorithm for Parallel Pancakes).
Consider the version of the parallel pancakes hypothesis testing problem (Problem 1.1), where k ′ ≤ k of the weights w i in the Gaussian mixture are unconstrained and the remaining k -k ′ are assumed to be equal to each other.
There is an algorithm for that problem which draws n = O (kd/δ) O(k ′ +log(k)) + log(k)/w min samples (where δ is as in Problem 1.1 and w min = min i∈[k] w i is the smallest weight), has runtime polynomial in n, d, and it outputs the correct hypothesis with probability at least 0.99.
The algorithm is based on estimating the first O(k ′ + log k) moment tensors through the empirical tensors, and thus it is also naturally expressible in the SQ model.
Remarks A single component with arbitrarily small weight can make the complexity in (Buhai & Steurer, 2023;Anderson et al., 2024) blow up quasipolynomially. By contrast, our algorithm can handle any number of such points, and the complexity interpolates smoothly between the alluniform and the fully general weights cases.
this section cite: ['b11', 'b1']

Section: Overview of Techniques
For Theorem 1.3, it suffices to show existence of a onedimensional distribution (corresponding to the projection along the hidden direction v in the parallel pancakes mixture in Problem 1.1) that matches a lot of moments with N (0, 1) and is thus hard to distinguish. Concretely, the goal is to show the existence of a set
S ⊂ R of size k such that E x∼S [x i ] = E x∼N (0,1) [x i
] for all i = 1, . . . , t, where t = Ω(log k) and x ∼ S denotes the uniform distribution on S. Once established, the theorem follows from standard SQ theory: convolving this discrete distribution with a narrow Gaussian yields a k-GMM B that still matches the first t moments with N (0, 1). A standard result from (Diakonikolas et al., 2023) then shows that hiding B along an unknown direction is hard to distinguish from N (0, I).
Fortunately, the desired moment-matching construction, known as a t-design, has been well-studied. Kane (2015) shows that designs of small size to match the moments of a distribution Q exist when the support of Q is "pathconnected". The design's size is upper bounded by the number K, which is defined to be the supremum of the ratio
sup x∈X p(x)
| inf x∈X p(x)| taken over all degree-t zero-mean polynomials p. Thus, it suffices to show K = 2 O(t) to prove Theorem 1.3. However, since the Gaussian distribution has unbounded support, there are (many) polynomials p where sup x∈R p(x) is infinite while the infimum is clearly finite.
To address this, we can instead consider another distribution Q supported on an interval I of length O( √ t), which also matches the first t moments with N (0, 1) (Lemma 3. 3  from Diakonikolas et al. (2017)). Thus, by creating a design to match Q we only need to bound K with X = I, which is now possible (Lemma 3.5). Specifically, by expressing p in the Hermite basis, we can show that t) . Additionally, by Gaussian anti-concentration, we can show that any zero-mean polynomial p of degree t has a 2 -O(t) probability of being less than -2 t) .
sup x∈I p(x) ≤ E x∼N (0,1) [|p(x)|]2 O(
-O(t) E x∼N (0,1) [|p(x)|]. This shows that |inf x∈I p(x)| ≥ E x∼N (0,1) [|p(x)|]2 -O(
We can also show that this bound is tight: if A is the uniform distribution on k points, it cannot match more than O(log k) moments with N (0, 1). The argument is that for any non-negative function
f , E x∼A [f 2 (x)]
E x∼A [f (x)] 2 ≤ k. If A matches the first 4t moments with N (0, 1), setting f (x) = x 2t makes the ratio 2 Ω(t) , implying t = O(log k).
In fact, we can extend the result to non-uniform distributions where all but k ′ points in the support have equal weight, showing that such distributions cannot match more than O(log(k)+k ′ ) moments with N (0, 1) (Proposition 4.1). As we will explain later, this will lead to the testing algorithm in Theorem 1.4.
The first step towards Proposition 4.1 is to show that, if all but k ′ points have weight at least w 0 , then it is impossible to match more than O(log(1/w 0 ) + k ′ ) moments with N (0, 1) (Proposition 4.2). The proof relies on extending the idea of the previous paragraph that any non-negative function f which vanishes (i.e. gives value zero) on the k ′ points in question satisfies
E x∼A [f 2 (x)] E x∼A [f (x)] 2 ≤ 1/w 2 0 .
We specifically use f (x) = x t p(x), where p(x) = (x -µ 1 ) 2 . . . (x -µ k ′ ) 2 and µ 1 , . . . , µ k ′ are the points in the support of A with unrestricted weights. The goal then is to show that the ratio r :=
E x∼A [f 2 (x)] E x∼A [f (x)] 2 is at least 2 Ω(
t-k ′ ) -combining this with our earlier lower bound r ≤ 1/w 2 0 implies t = O(log(1/w 0 )+k ′ )). To bound r, we assume A matches Θ(t + k ′ ) moments, allowing us to replace E x∼A [•] with E x∼N (0,1) [•] in the definition of r. Since p(x) has degree 2k ′ , we show p(x) ≥ 2 -O(k ′ ) E x∼N (0,1) [p 2 (x)] 1/2 near x = √ 2t (Corollary 4.5). The contribution to E x∼N (0,1) [f 2 (x)] from x ∈ [0.9 √ 2t, 1.1 √ 2t] will then be at least (3.6t) t E x∼N (0,1) [p 2 (x)]2 -O(k ′ ) (see Equations (7) and (8) for the full calculations). Meanwhile, by Hölder's inequality, E x∼N (0,1) [f (x)] ≤ E x∼N (0,1) [x 3t/2 ] 2/3 E x∼N (0,1) [p 3 (x)] 1/3 , which by hypercontractivity is at most ( 3t 2e ) t/2 E x∼N (0,1) [p(x) 2 ] 1/2 2 O(k ′ ) . Combining these bounds establishes r ≥ 2 Ω(t-k ′ ) and proves Proposition 4.2.
We can also argue that the previous paragraph's result can always be used with w 0 ≥ 2 -O(k ′ ) /k (Proposition 4.1). By considering the same polynomial p which vanishes at the points with unconstrained weights, we can combine the hypercontractivity of p with the Cauchy-Schwarz inequality to derive a lower bound on the total weight of the equalweight points:
i≥k ′ +1 w i ≥ 2 -O(k ′ ) . This immediately implies w 0 ≥ 2 -O(k ′ ) /k.
We have shown that any discrete distribution with k ′ arbitrary weights and k -k ′ equal weights cannot match more than O(log(k) + k ′ ) moments with N (0, 1). This result extends to approximate moment matching within error 2 -O(t) , and holds even after convolving the distribution with a Gaussian (cf. Lemma D.3). For the parallel pancakes testing problem, this implies that for some i ≤ O(log(k) + k ′ ) the i-th order tensor of the GMM in the alternative hypothesis differs significantly from that of N (0, I). This gap can be detected by estimating the tensor via averaging samples (an operation that has complexity d Θ(m) ), leading to the testing algorithm of Theorem 1.4.
this section cite: ['b19', 'b28']

Section: Preliminaries
We present only the essential preliminaries here; see Appendix B for a full version.
Notation We use Z + for positive integers, R + 0 for nonnegative reals, and [n] def = {1, . . . , n}. We use x ⊗ y for the tensor product of two vectors. For a random variable x following distribution D, we write x ∼ D and
E[x] for its expectation. The Gaussian distribution with mean µ and covariance Σ is N (µ, Σ), and Pr(E) denotes the probability of event E. The indicator function of E is 1(E). The L p norm of an R-valued random variable x is ∥x∥ p = E[|x| p ] 1/p , and for a function f : R d
→ R, it is ∥f ∥ p = E x∼N (0,I) [|f (x)| p ] 1/p . We use a ≲ b to indicate a ≤ Cb
for an absolute constant C > 0 independent of a and b.
this section cite: []

Section: Hermite Analysis
In this paper, we use the normalized probabilist's Hermite polynomials, which form an orthonormal basis of
L 2 := {f : E x∼N (0,1) [f 2 (x)] < ∞} with respect to the Gaussian measure, i.e., R h k (x)h m (x)e -x 2 /2 dx = √ 2π1(k = m). Every function f ∈ L 2 can be uniquely expressed as f (x) = ∞ i=0 a i h i (x)
. Probability Facts The first fact below follows from direct calculations, the second from the Carbery-Wright inequality, and the last from Hölder's inequality combined with Fact 2.3.
Fact 2.1 (Gaussian Moments). E x∼N (0,1) [x t ] ≲ (t/e) t/2 ∀t≥0.
Fact 2.2. For every polynomial of degree r and every ϵ > 0,
Pr x∼N (0,1) (|p(x)| ≤ ϵ∥p∥ 1 ) ≤ O(rϵ 1/r ). Fact 2.3 (Gaussian Hypercontractivity). If p is a degree r polynomial and k > 2, then ∥p∥ k ≤ (k -1) r/2 ∥p∥ 2 .
Fact 2.4. For any polynomial p of degree r, ∥p∥1 ∥p∥2 ≥ 3 -r .
this section cite: []

Section: Arithmetic Mean-Geometric Mean Inequality
We record the following continuous analog of the Arithmetic Mean-Geometric Mean (AM-GM) inequality. We refer to Appendix B for a more detailed discussion. Fact 2.5 (Continuous AM-GM Inequality). Let f : R → R + 0 be a function, and let I ⊆ R be a finite interval. If f (x) and ln f (x) are integrable on I, then the following holds:
1 |I| I f (x)dx ≥ exp 1 |I| I ln f (x)dx .
this section cite: []

Section: Non-Gaussian Component Analysis
The parallel pancakes Problem 1.1 is a special case of the following problem. Problem 2.6 (Non-Gaussian Component Analysis (NGCA)). Let B be a distribution on R.
For a unit vector v, we denote by P B,v the distribution with the density P B,v (x) := B(v ⊤ x)ϕ ⊥v (x), where ϕ ⊥v (x) = exp -∥x -(v ⊤ x)v∥ 2 2 /2 /(2π) (d-1)/2 , i.e., the distribution that coincides with B on the direction v and is standard Gaussian in every orthogonal direction. We define the following hypothesis testing problem:
• H 0 : The data distribution is N (0, I d ).
• H 1 : The data distribution is P B,v , for some vector v ∈ S d-1 in the unit sphere.
It is known that solving Problem 2.6 when B matches the first m moments with N (0, 1) requires at least d Ω(m) complexity in the statistical query model (Proposition B.8).
this section cite: []

Section: The Uniform Weights Case
In this section we prove the following proposition, which is sufficient for showing our first result, Theorem 1.3. Proposition 3.1. For each k that is larger than a sufficiently large absolute constant, there exists a set S of k points in R such that the uniform distribution over S matches the first Ω(log k) moments with N (0, 1).
Given the above, Theorem 1.3 follows directly from standard SQ theory. The details are provided in Appendices B.5 and C, but the steps are summarized as follows: Let A be the uniform distribution on the set S from Proposition 3.1. We can define the distribution B to be what one obtains by first drawing a sample from A, rescaling it by 1/ √ δ and adding Gaussian noise N (0, 1 -δ). This operation preserves moment matching and makes B a GMM. The NGCA Problem 2.6 with that B then becomes equivalent to the parallel pancakes Problem 1.1. Since B matches m = Ω(log k) moments with N (0, 1), its standard SQ hardness state that its complexity is d Ω(m) = d Ω(log k) (Proposition B.8). We refer to Appendix C for the details of this paragraph.
In the remainder, we focus on proving Proposition 3.1 by leveraging a result on designs theory from Kane (2015). The original result in Kane (2015) is highly general and applies to a wide range of topological, path-connected design problems. However, as we will only use the theorem for intervals, we present here a specialized version tailored to this case.
Fact 3.2 (see Theorem 4 in Kane (2015)). Let t ∈ Z + be an integer, I ⊂ R be an interval and Q be a distribution on I. Let W t be the vector space of all polynomials of degree at most t, and V t be the vector space of polynomials p of degree at most t with
E x∼Q [p(x)] = 0. Define K t = sup p∈V \{0} sup x∈I p(x) | inf x∈I p(x)| . Then for every integer n > (t -1)(K t + 1) there exists a set S ⊂ I of n points such that 1 |S| x∈S p(x) = E x∼Q [p(x)] for all p ∈ W t .
Our goal is to show that K t = 2 O(t) for Q = N (0, 1), which would directly imply Theorem 1.3. However, as noted in Section 1.2, K t may be infinite when I = R. To address this, we use a distribution Q supported on a bounded interval of R that matches the first t moments of N (0, 1). Applying Fact 3.2 with this Q also suffices for establish Theorem 1.3.
this section cite: ['b28', 'b28']

Section: Lemma 3.3 (Gaussian Quadrature (Lemma 4.3 in Diakonikolas et al. (2017))).
There is a discrete distribution Q on the real line, supported on t points, that agrees with N (0, 1) on the first 2t -1 moments. All points x in the support of Q have |x| = O( √ t).
We start with an anti-concentration property of Gaussian polynomials that will be useful for bounding the numerator in the definition of K t .
Lemma 3.4. Let C be a sufficiently large constant. For every polynomial p : R → R of degree at most t with E x∼N (0,1) [p(x)] = 0 and for every ϵ > 0 it holds
Pr x∼N (0,1) (p(x) > ϵ∥p∥ 1 ) ≥ 1 2 ∥p∥ 1 ∥p∥ 2 (1 -Ctϵ 1+1/t ) 2 .
Proof. Denote by ϕ(x) the pdf of N (0, 1). We have the following (each step is explained below):
∥p∥ 1 = p(x)>0 p(x)ϕ(x)dx - p(x)≤0 p(x)ϕ(x)dx = 2 p(x)>0 p(x)ϕ(x)dx = 2 p(x)≥ϵ∥p∥1 p(x)ϕ(x) dx + 0≤p(x)<ϵ∥p∥1 p(x)ϕ(x)dx ≤ 2∥p∥ 2 Pr x∼N (0,1) (p(x) ≥ ϵ∥p∥ 1 ) 1/2 + 2ϵ∥p∥ 1 Pr x∼N (0,1) (|p(x)| ≤ ϵ∥p∥ 1 ) ≤ 2∥p∥ 2 Pr x∼N (0,1) (p(x) ≥ ϵ∥p∥ 1 ) 1/2 + 2ϵ∥p∥ 1 Ctϵ 1/t ,
where the first line used that E x∼N (0,1) [p(x)] = 0, the penultimate inequality used the Cauchy-Schwarz inequality for the first term and the bound p(x) ≤ ϵ∥p∥ 1 for the second term, and the last line used the Carbery-Wright inequality (Fact 2.2). Rearranging, we obtain
Pr x∼N (0,1) (p(x) > ϵ∥p∥ 1 ) ≥ 1 2 ∥p∥1 ∥p∥2 (1 -2Ctϵ 1+1/t
). We rename the constant 2C to C.
We now bound K t from Fact 3.2 with I=[-C √ t,C √ t].
Lemma 3.5. Let t > C be an integer, where C is a sufficiently large constant, and define I = [-C √ t, +C √ t]. For every polynomial p of degree at most t with
E x∼N (0,1) [p(x)] = 0 it holds sup x∈I p(x) | inf x∈I p(x)| ≤ 2 O(t) .
Proof. It suffices to upper bound the numerator by 2 O(t) ∥p∥ 1 and lower bound the denominator by 2 -O(t) ∥p∥ 1 .
Upper bound on numerator We require the following: Fact 3.6 (Krasikov (2004)). For the k-th normalized probabilist's Hermite polynomial h k , we have
sup x∈R h 2 k (x)e -x 2 /2 = O(k -1/6 ).
Consider a polynomial p which has degree at most t and satisfies E x∼N (0,1) [p(x)] = 0. We first expand the polynomial in the Hermite basis: p(x) = t k=1 a k h k (x), where the summation starts from k = 1 because a 0 = E x∼N (0,1) [p(x)] = 0. For any x ∈ I we have (the first step uses Cauchy-Schwarz inequality): t) .
|p(x)| = t k=1 a k h k (x) ≤ t k=1 a 2 k t k=1 h 2 k (x) ≲ ∥p∥ 2 t k=1 e x 2 /2 k -1/6 (by Fact 3.6) ≤ ∥p∥ 2 2 O(t) t k=1 k -1/6 (using |x| = O( √ k)) ≤ ∥p∥ 2 2 O(t) t O(1) (using t k=1 k -1/6 = t O(1) ) ≤ ∥p∥ 2 2 O(t) ≤ ∥p∥ 1 2 O(
(using Fact 2.4)
Lower bound on the denominator From Lemma 3.4 with -p in place of p and ϵ = 2 -t , and Fact 2.4 we get that
Pr x∼N (0,1) (p(x) < -2 -t ∥p∥ 1 ) ≥ 1 2 ∥p∥ 1 ∥p∥ 2 (1-Ct2 -t-1 ) 2 ≥ 1 2 3 -t (1 -Ct2 -t-1 ) 2 > 2 -4t .
where we used that t is big enough so that C t 2 t <0.5. Then,
Pr x∼N (0,1) (p(x) < -2 -t ∥p∥ 1 , x ∈ I) ≥ Pr x∼N (0,1) (p(x) < -2 -t ∥p∥ 1 ) -Pr x∼N (0,1) (x ̸ ∈ I) ≥ 2 -4t -2 -100t > 0 ,
where in the last line we used that I = [-C √ t, +C √ t] for a large constant C. We have thus shown that inf x∈I p(x) ≤ -2 -t ∥p∥ 1 and therefore | inf x∈I p(x)| > 2 -t ∥p∥ 1 .
this section cite: ['b33']

Section: The Mostly Equal Weights Case
This section focuses on Theorem 1.4 and is organized as follows. The key structural result is the following impossibility of moment matching: if A is a distribution on k points, with k ′ points having unconstrained weights and the remaining k -k ′ equal, then A cannot match more than O(log k + k ′ ) moments with the standard Gaussian.
Proposition 4.1. Let k ′ < k be positive integers, and let A be a discrete distribution on k points in R. Suppose k -k ′ of the points have equal probability masses, while the remaining k ′ points have unrestricted probability masses. Denote by m the highest degree for which every degree-
m ′ ≤ m polynomial g satisfies E x∼A [g(x)] -E x∼N (0,1) [g(x)] ≤ 2 -C•m ∥g∥ 2 , then m must satisfy m ≤ O(log k) + O(k ′ ).
Section 4.1 explains how Proposition 4.1 leads to a testing algorithm (the full proof appears in Appendix D.1). Section 4.2 provides the proof of Proposition 4.1.
this section cite: []

Section: Proof Sketch of Theorem 1.4
Consider the parallel pancakes problem from Theorem 1.4, which is equivalent to the NGCA problem (Problem 2.6) with the 1
-d GMM B = i∈[k] w i N (µ i , 1-δ).
If B approximately matches m moments of N (0, 1), we aim to show that m≤O(log k+k ′ ), enabling a testing algorithm that detects significant deviations in moment tensors.
Specifically, suppose every polynomial p of degree m ′ ≤m with ∥p∥ 2 =1 satisfies E x∼B [p(x)]-E x∼N (0,1) [p(x)] ≤(δ/2) Cm for some large constant C ≫ 1. Now, consider the discrete distribution A, which assigns weight w i to each center µ i / √ δ. By Lemma D.3, A also approximately matches the moments of N (0, 1), but with an error of 2 -O(m) instead of (δ/2) O(m) . Then Proposition 4.1 yields m ≤ O(log k+k ′ ), as desired.
We just showed that there is a polynomial p of degree at most m = O(log(k) + k ′ ), where the expectations under B and N (0, 1) differ significantly: λ := E x∼B [p(x)] -E x∼N (0,1) [p(x)] > (δ/2) Cm . An averaging argument further implies that a gap holds even for some monomial x i . Lifting this to the d-dimensional parallel pancakes, we have the moment tensor gap
E x∼P B,v [x ⊗i ] - E x∼N (0,I) [x ⊗i ] = ±λv ⊗i .
The Frobenius norm of the gap is λ > (δ/2) Cm , implying that between the (expected) moment tensors, at least one entry differs by at least ϵ := λ/d m = (d/δ) (C-1)m . We will test by searching for such an entry in the empirical tensor.
Algorithm 1 Testing Algorithm (simplified)
1: Input: k, n. Output: Ĥ ∈ {H 0 , H 1 }. 2: For i = 1, 2, 3, . . . , C • (log(k) + k ′ ) do 3: Draw x 1 , . . . , x n ∼ D. 4: Define M ← 1 n n i=1 x ⊗i . 5: Define M ′ := E x∼N (0,I) [x ⊗i ]. 6: If ∃a=(i 1 , . . . , j i ) such that |M a -M ′ a |>d -Cm λ m 7:
then Output H 1 and terminate. 8: Return H 0 .
The tester above is a simplified version. However, it is not fully correct, as we must ensure the concentration of the empirical tensor to bound the sample complexity. The Gaussian N (0, I)'s empirical tensor is well-concentrated. While the parallel pancake's tensor might not concentrate well, this happens only when there is a Gaussian component much farther than
√
d from the origin -otherwise every sample from the parallel pancake is within O( √ d) in norm in high probability, and the empirical tensor is entrywise well-concentrated (e.g. by Hoeffding). This is also a testable condition: with ≫ log(k)/w min samples, we will be able to check if every component is centered at most O( √ d) from the origin. The full version of the algorithm with this additional preliminary check, along with its correctness proof, are provided in Appendix D.1.
this section cite: []

Section: Proof of Proposition 4.1
We now show Proposition 4.1. We will actually show a slightly different version below, where k ′ of the points have arbitrary weights and the rest have weight at least w 0 .
Proposition 4.2. Let C be a sufficiently large constant, and let k ′ < k be positive integers. Let A be a discrete distribution on k points in R with probability masses w 1 , . . . , w k , where w i ≥ w 0 for i = k ′ + 1, . . . , k (i.e., the last k -k ′ weights are lower bounded by w 0 , while the first k ′ weights are unrestricted). Let m be the largest degree such that every polynomial g of degree m ′ ≤ m satisfies
E x∼A [g(x)] - E x∼N (0,1) [g(x)] ≤ w 0 2 -C•m ∥g∥ 2 . (1)
Then m ≤ O(log(1/w 0 )) + O(k ′ ).
Proposition 4.1 can be derived from this via the following observation (shown in Appendix D.2): in the setting of Proposition 4.2, let p(x) = k ′ i=1 (x -µ i ), where µ 1 , . . . , µ k ′ are the points in the support of A with the unconstrained weights. Then, the weights of the k -k ′ points with uniform weights is always
k i=k ′ +1 w i ≥ E x∼A [p(x)] 2 E x∼A [p 2 (x)] ≳ ∥p∥ 2 1 ∥p∥ 2 2 ≥ 3 -2k ′
, where the first step uses Cauchy-Schwarz inequality, the second uses the (approximate) moment matching, and the third is a consequence of Gaussian hypercontractivity (Fact 2.4). This means that every such weight is w i ≥ 3 -2k ′ /k, which when plugged into Proposition 4.2 gives Proposition 4.1.
We now focus on showing Proposition 4.2. We will follow a top-down presentation, starting with the proof strategy and concluding with a derivation of the necessary bounds.
We will use a reparameterization m = 2t + 4k ′ with t even. The goal is to show that if A is assumed to approximately match the first m = 2t + 4k ′ moments with N (0, 1) (in the sense of Equation ( 1)), then t must be at most O(log(1/w 0 ) + k ′ ). Let µ 1 , . . . , µ k be the points on which A is supported, where the first k ′ points are the ones with the unrestricted weights, and consider the polynomial f (x) = x t p(x), where p( x
) = (x -µ 1 ) 2 (x -µ 2 ) 2 • • • (x -µ k ′ ) 2 .
The proof strategy is the following: if the expectation of f under A approximately matches that of N (0, 1), then the value of f on every point µ i cannot be too large, which will cause the expectations of f 2 to deviate.
Because of Equation (1) with g(x) = f 2 (x), we have:
k i=k ′ +1 w i µ t i p(µ i ) = E x∼A x t p(x)(2)
≤ E x∼N (0,1) x t p(x) + w 0 ∥x t p(x)∥ 2 2 C(2t+4k ′ ) .(3)
This, together with the lower bound w i ≥ w 0 for the points i = k ′ + 1, . . . , k and the fact that t is even, implies that for all i = k ′ + 1, . . . , k it holds
µ t i p(µ i ) ≤ 1 w 0 E x∼N (0,1) x t p(x) + ∥x t p(x)∥ 2 2 C(2t+4k ′ ) . (4
)
We now examine the expectations of the square of f (x).
Because of Equation (1) with g(x) = f 2 (x), we have
E x∼N (0,1) x 2t p 2 (x) ≤ E x∼A x 2t p 2 (x) + ∥x 2t p 2 (x)∥ 2 2 C(2t+4k ′ ) = k i=k ′ +1 w i µ t i p(µ i ) 2 + ∥x 2t p 2 (x)∥ 2 2 C(2t+4k ′ ) .
Next, we can combine this with Equation ( 4), divide both sides by E x∼N (0,1) [x t p(x)] 2 (and use k i=k ′ +1 w i ≤ 1) to obtain the following, where λ := 2 -C(2t+4k ′ ) :
E x∼N (0,1) x 2t p 2 (x) E x∼N (0,1) [x t p(x)] 2 ≤ 1 w 0 2 + λ 2 E x∼N (0,1) x 2t p 2 (x) E x∼N (0,1) [x t p(x)] 2 + λ E x∼N (0,1) x 4t p 4 (x) 1 2 E x∼N (0,1) [x t p(x)] 2 .
Let us simplify this inequality. Let r be the ratio on the LHS. The second term on the RHS is λ 2 • r. The third term is at most 3 t+2k ′ λr, by applying Gaussian hypercontractivity (Fact 2.3) to the polynomial x t p(x). Thus, the inequality becomes r(1 -λ 2 -λ3 t+2k ′ ) ≲ 1/w 2 0 . Since λ = 2 -C(2t+4k ′ ) for large constant C, the expression inside the parentheses is greater than 0.5. Therefore, the inequality implies that r ≲ 1/w 2 0 . The next step is to establish a lower bound for r, specifically to show that r ≥ 2 Ω(t) /2 O(k ′ ) . If this can be done, combining the two bounds 2 Ω(t) /2 O(k ′ ) ≤ 1/w 2 0 and taking logarithms yields t = O(log(1/w 0 )) + O(k ′ ), completing the proof of Proposition 4.2.
this section cite: []

Section: LOWER BOUNDING THE RATIO r
We want to establish the following, which was the missing piece in the proof of Proposition 4.2 above.
Lemma 4.3. Let p : R → R + 0 be a polynomial of the form p(
x) = (x -µ 1 ) 2 (x -µ 2 ) 2 • • • (x -µ k ′ ) 2 . Then E x∼N (0,1) x 2t p 2 (x) E x∼N (0,1) [x t p(x)] 2 ≳ 2 Ω(t) 2 O(k ′ ) .
The most difficult part involves lower bounding the numerator. To this end, we will show the following bound:
Lemma 4.4. Let p : R → R be a polynomial of the form p(
x) = (x-µ 1 )(x-µ 2 ) • • • (x-µ k ′ )
where µ 1 , . . . , µ k ′ ∈ R, and define I := [0.9
√ 2t, 1.1 √ 2t].
For every t > 0 and for every µ 1 , . . . , µ k ′ ∈ R, the following holds:
exp 1 |I| x∈I ln |p(x)|dx ≥ max y∈R:|y|≤ √ t |p(y)| 2 O(k ′ ) . (5
)
We will actually apply Lemma 4.4 after taking expectations of both sides. This version is presented below, and its proof follows by taking expectations and performing some manipulations (see Appendix D.3 for a detailed proof).
Corollary 4.5. Let p : R → R be a polynomial of the form p(x) = (x-µ 1 )(x-µ 2 ) • • • (x-µ k ′ ) where µ 1 , . . . , µ k ′ ∈ R are arbitrary parameters. Define I = [0.9
√ 2t, 1.1 √ 2t]. For all t ≥ 1 we have exp 1 |I| x∈I ln |p(x)|dx ≥ ∥p∥2 2 O(k ′ ) .
To see why the above bound is needed to prove Lemma 4.3, we will first prove Lemma 4.3 assuming Corollary 4.5. Then, we will prove Lemma 4.4.
Proof of Lemma 4.3. First, for the denominator, we have the following:
E x∼N (0,1) [x t p(x)] ≤ E x∼N (0,1) [p 3 (x)] 1/3 E x∼N (0,1) [x 3t/2 ] 2/3 ≲ ∥p∥ 3 3t 2e t/2 ≲ 2 k ′ ∥p∥ 2 3t 2e t/2 ,(6)
where the first step uses Hölder's inequality, the second step uses the Gaussian moments bound (Fact 2.1), and the final step uses Gaussian hypercontractivity (Fact 2.3).
We now lower bound the numerator. Define I := [0.9 √ 2t, 1.1 √ 2t]. We have the following (see below for explanations of each step):
E x∼N (0,1) x 2t p 2 (x) ≳ +∞ -∞ x 2t e -x 2 /2 p 2 (x) dx ≥ x∈I x 2t e -x 2 /2 p 2 (x) dx ≥ (1.62t) t e -0.81t x∈I p 2 (x) dx = (1.62t) t e -0.81t |I| 1 |I| x∈I p 2 (x) dx ≳ (1.62t) t e -0.81t 1 |I| x∈I p 2 (x) dx , (7
)
where the third inequality uses that min x∈I x 2t e -x 2 /2 ≥ (1.62t) t e -0.81t , and the final inequality uses that |I| = 0.2 √ 2t = Ω(1). We now focus on the root mean square term 1
|I| x∈I p 2 (x) dx, which we will bound using the AM-GM inequality (Fact 2.5) and the geometric mean bound from Lemma 4.4. The first step below applies Fact 2.5 with f (x) := p 2 (x), and the next step uses Corollary 4.5.
1 |I| x∈I p 2 (x) dx ≥ exp 1 |I| x∈I ln |p(x)| dx 2 ≥ ∥p∥ 2 2 2 O(k ′ ) .
Combining with Equation ( 7), we obtain the following bound for the numerator:
E x∼N (0,1) x 2t p 2 (x) ≳ (1.62t) t e -0.81t ∥p∥ 2 2 2 O(k ′ ) . (8)
Combining Equation (6) and Equation (8), we conclude
E x∼N (0,1) x 2t p 2 (x) E x∼N (0,1) [x t p(x)] 2 ≳ (1.62) t ( 1.5 e ) t e 0.81t 2 O(k ′ ) ≥ (1.3) t 2 O(k ′ ) .
We conclude this section by proving Lemma 4.4.
Proof of Lemma 4.4. Fix an arbitrary y ∈ R with |y| ≤ √ t. First, note that by the property of logarithms and sums, we can write the left hand side as
exp   k ′ i=1 1 |I| x∈I ln |x -µ i |dx   .
In order to show Equation ( 5), it suffices to work with each term and show the following for each i ∈ [k ′ ]:
1 |I| x∈I ln |x -µ i | ≥ ln |y -µ i | -O(1) .
Equivalently, it suffices to show that Equation ( 5) holds for every linear polynomial of the form p(x) = x -a. Therefore, the goal for the rest of this proof is to show that
exp 1 |I| x∈I ln |x -a|dx ≥ |y -a|/O(1) , (9
)
holds for every a ∈ R and y ∈ R with |y| ≤ √ t. We will examine two cases.
Case 1 The first case is when the root a of the polynomial is outside the interval I. In this case, we can show that |x -a|/|y -a| = Θ(1), which implies ln |x -a| ≥ ln |y -a|-O(1), and the desired conclusion (Equation ( 9)) follows by integrating both sides and applying the exp(•) function.
To show the earlier claim that |x -a|/|y -a| = Θ(1), we can consider the following sub-cases:
1. Case a ≥ 1.1 √ 2t (i.e., a is to the right of I): Suppose a = 1.1 √ 2t+u for some non-negative u. Then, a-x = (1.1 √ 2t -x) + u = Θ( √ t) + u and a -y = (1.1 √ 2ty)+u = Θ( √ t)+u. Therefore, for any u ≥ 0, the ratio |x -a|/|y -a| = (Θ( √ t) + u)/(Θ( √ t) + u) = Θ(1).
2. The cases a < -√ t and a ∈ [ √ t, 0.9 √ 2t] can be shown in a similar manner.
Case 2 Suppose that the root a of the polynomial p lies within the interval I. In that case, we can show via derivative analysis that f (a) := 1|I| x∈I ln |x-a| dx for a ∈ I is minimized at the midpoint of I, i.e., at a = √ 2t, and confirm that f ( √ 2t) ≥ √ t/20 = Ω(|y -a|). These calculations are provided in Appendix D.3.
this section cite: []

Section: Conclusions and Future Work
Our work makes progress in understanding the complexity of learning parallel pancake GMMs, in terms of both lower and upper bounds. We establish the tightness of existing algorithms for uniform weights and provide an improved testing algorithm for uneven weights. A number of interesting open problems remain:
• Can we extend our testing algorithm to learning the unknown direction of the parallel pancakes? More broadly, can we characterize the complexity of learning GMMs with common covariance and not necessarily collinear means as a function of the weights distribution?
• Can we obtain an algorithm with quasi-polynomial (i.e., d O(log(1/wmin) ) complexity for GMMs whose components have unknown (and potentially different) covariances?
this section cite: []

Section: References
Ref_id:b0 Title: On spectral learning of mixtures of distributions Year: (2005)
Ref_id:b1 Title: Dimension reduction via sum-of-squares and improved clustering algorithms for non-spherical mixtures Year: (2024)
Ref_id:b2 Title: Special Functions Year: (1999)
Ref_id:b3 Title: Learning mixtures of arbitrary Gaussians Year: (2001)
Ref_id:b4 Title: Outlier-robust clustering of non-spherical mixtures Year: (2020)
Ref_id:b5 Title: Outlier-robust clustering of gaussians and other non-spherical mixtures Year: (2020)
Ref_id:b6 Title: Robustly learning mixtures of k arbitrary gaussians Year: ()
Ref_id:b7 Title: Polynomial learning of distribution families Year: (2015)
Ref_id:b8 Title:  Year: (1998)
Ref_id:b9 Title: Isotropic pca and affineinvariant clustering Year: (2008)
Ref_id:b10 Title: Symposium on Theory of Computing (STOC) Year: ()
Ref_id:b11 Title: Beyond parallel pancakes: Quasipolynomial time guarantees for non-spherical gaussian mixtures Year: (2023)
Ref_id:b12 Title: Distributional and l q norm inequalities for polynomials over convex bodies in R n Year: (2001)
Ref_id:b13 Title: Learning mixtures of gaussians Year: (1999)
Ref_id:b14 Title: Faster and sample nearoptimal algorithms for proper learning mixtures of gaussians Year: (2014)
Ref_id:b15 Title: Implicit high-order moment tensor estimation and learning latent variable models Year: (2024)
Ref_id:b16 Title: Statistical query lower bounds for robust estimation of highdimensional gaussians and gaussian mixtures Year: (2017)
Ref_id:b17 Title: Robustly learning any clusterable mixture of gaussians Year: (2020)
Ref_id:b18 Title: Clustering mixture models in almost-linear time via list-decodable mean estimation Year: ()
Ref_id:b19 Title: Sq lower bounds for non-gaussian component analysis with weaker assumptions Year: (2023)
Ref_id:b20 Title: Sum-of-squares lower bounds for non-gaussian component analysis Year: (2024)
Ref_id:b21 Title: Statistical query learning Year: (2016)
Ref_id:b22 Title: Statistical algorithms and a lower bound for detecting planted cliques Year: ()
Ref_id:b23 Title: Statistical query algorithms for mean vector estimation and stochastic convex optimization Year: (2017)
Ref_id:b24 Title: A review of robust clustering methods Year: (2010)
Ref_id:b25 Title: Continuous LWE is as hard as LWE & applications to learning gaussian mixtures Year: (2022)
Ref_id:b26 Title: Tight bounds for learning a mixture of two gaussians Year: (2015)
Ref_id:b27 Title: Mixture models, robustness, and sum of squares proofs Year: (2018)
Ref_id:b28 Title: Small designs for path-connected spaces and path-connected homogeneous spaces Year: (2015)
Ref_id:b29 Title: The spectral method for general mixture models Year: (2005)
Ref_id:b30 Title: Efficient noise-tolerant learning from statistical queries Year: (1998)
Ref_id:b31 Title: Outlier-robust momentestimation via sum-of-squares Year: (2017)
Ref_id:b32 Title: Robust moment estimation and improved clustering via sum of squares Year: (2018)
Ref_id:b33 Title: New bounds on the Hermite polynomials Year: (2004)
Ref_id:b34 Title: Mixture models: theory, geometry and applications Year: (1995)
Ref_id:b35 Title: Clustering mixtures with almost optimal separation in polynomial time Year: (2022)
Ref_id:b36 Title: Settling the robust learnability of mixtures of gaussians Year: (2021)
Ref_id:b37 Title: Learning gmms with nearly optimal robustness guarantees Year: ()
Ref_id:b38 Title: Settling the polynomial learnability of mixtures of gaussians Year: (2010)
Ref_id:b39 Title: The free markoff field Year: (1973)
Ref_id:b40 Title: Analysis of Boolean Functions Year: (2014)
Ref_id:b41 Title: Contributions to the mathematical theory of evolution Year: (1894)
Ref_id:b42 Title: Near-optimal-sample estimators for spherical gaussian mixtures Year: (2014)
Ref_id:b43 Title:  Year: (1989)
Ref_id:b44 Title: Statistical Analysis of Finite Mixture Distributions Year: (1985)
Ref_id:b45 Title: A spectral algorithm for learning mixtures of distributions Year: (2002)
