Title: On the Benefits of Active Data Collection in Operator Learning
Abstract: We study active data collection strategies for operator learning when the target operator is linear and the input functions are drawn from a mean-zero stochastic process with continuous covariance kernels. With an active data collection strategy, we establish an error convergence rate in terms of the decay rate of the eigenvalues of the covariance kernel. We can achieve arbitrarily fast error convergence rates with sufficiently rapid eigenvalue decay of the covariance kernels. This contrasts with the passive (i.i.d.) data collection strategies, where the convergence rate is never faster than linear decay (∼ n -1 ). In fact, for our setting, we show a non-vanishing lower bound for any passive data collection strategy, regardless of the eigenvalues decay rate of the covariance kernel. Overall, our results show the benefit of active data collection strategies in operator learning over their passive counterparts.

Section: Introduction
There is an increasing interest in using data-driven methods to estimate solution operators of partial differential equations (PDEs) encountered in scientific applications. To set up the problem, consider X ⊆ R d and a linear PDE of the form Lu = f , subject to the boundary condition u(x) = 0 for all x ∈ boundary(X ). The goal, given a function f , is to find the corresponding solution u that satisfies the PDE.
Traditionally, numerical PDE solvers are used to compute u from f . In contrast, operator learning focuses on approximating the solution operator of the PDE (Lu et al., 2021;Kovachki et al., 2023). Under typical conditions, this PDE has a linear solution operator F such that u = F(f ) for all f in an appropriate function space. Given a set of training samples (f 1 , u 1 ), . . . , (f n , u n ), the aim is to learn an operator F n that closely approximates the true solution operator F under a suitable metric. The hope is that for a new input function f , evaluating u = F n (f ) will be considerably faster than solving for u through traditional methods, with minimal loss in accuracy.
In this work, we study the sample complexity of operator learning. Specifically, given an operator F, how many inputoutput pairs {(f j , F(f j ))} n j=1 are necessary to estimate an operator F n such that F n (f ) ≈ F(f ) for all relevant f ? This question has been studied in several specific contexts, such as for linear operators with fixed singular value decomposition by de Hoop et al. (2023) and Subedi & Tewari (2024), for Lipschitz operators by Liu et al. (2024), and for the random feature model by Nelsen & Stuart (2021). These are just a few representative works, and we refer the reader to (Kovachki et al., 2024b, Section 5) for a more comprehensive review of such sample complexity results.
A common theme of these sample complexity analyses is that they are conducted within the framework of traditional statistical learning (Kovachki et al., 2023, Section 2.2). In the statistical setting, the learner has access to training samples {(f j , F(f j ))} n j=1 , where f j ∼ iid µ from some probability measure µ, and the objective is to produce an estimator F n such that F n (f ) ≈ F(f ) on average over test samples f ∼ µ. This scenario is also referred to as the passive learning setting. Under reasonable non-trivial assumptions, the best achievable rate of error convergence in this setting is ∼ 1/n, when F n is evaluated under an appropriate metric, say the p-th power of the L p µ -Bochner norm.
this section cite: ['b21', 'b11', 'b8', 'b25', 'b19', 'b23', 'b11']

Section: Our Contribution
In this work, we go beyond the passive statistical setting and study operator learning where the learner is not restricted to iid samples from a source distribution and can use active data collection strategies. We focus on the case where the operator of interest F is a bounded linear operator and µ is a distribution with zero mean and the covariance structure defined by a continuous kernel K. Such distributions include Gaussian processes with common covariance kernels. For a given covariance kernel K, our main result provides an active data collection strategy, an estimation rule, and establishes an error bound for the proposed estimator in terms of the eigenvalue decay of the integral operator of K. Formally, if λ 1 ≥ λ 2 ≥ . . . are the eigenvalues of the integral operator of K, there exists an active data collection strategy and estimation rule such that the estimator obtained using n actively collected input-output pairs achieves the following error bound:
ε 2 n i=1 λ i + ∥F∥ 2 op ∞ i=n+1 λ i .
The ε 2 captures the error of approximation oracle O for F that the learner has access to. For example, O could be the PDE solver used to generate training data for operator learning. Generally, ε > 0 is the irreducible error of the bound. The second term O( i>n λ i ) is a reducible error, which goes to 0 as n → ∞ for continuous kernels K under the bounded domain. For example, for the covariance operator α(-∇ 2 + βI) -γ used by Li et al. (2021) and Kovachki et al. (2023), we show that the reducible error vanish at the rate ≲ n -( 2γ d -1) . Taking 2γ ≫ d, one can achieve any polynomial rate of decay. In fact, given any rate R n → 0 as n → ∞, one can always construct a continuous kernel K such that the reducible error decays faster than R n . Thus, arbitrarily fast rates can be obtained using active data collection strategies. Our main result is formalized in Theorem 3.1, and the proof is based on the celebrated Karhunen-Loève decomposition for functions drawn from µ with covariance kernel K.
Furthermore, in Theorem 4.2, we show that, irrespective of the decay rate of the eigenvalues of the covariance kernel K, there always exists a bounded linear operator F and a distribution µ with covariance kernel K such that the minimax estimation error fails to converge to 0 under any passive (i.i.d.) data collection strategy. In particular, for every n, even when ε = 0, we establish the minimax lower bound of ∥F∥ 2 op λ 1 under any passive data collection strategy. That is, the lower bound does not vanish even as n → ∞. Collectively, Theorems 3.1 and 4.2 establish a clear advantage of active data collection strategy for operator learning.
this section cite: ['b18', 'b11']

Section: Related Works
Recent work by Musekamp et al. (2024) considers active methods for operator learning. However, in contrast to our approach of using linearity of the solution operator and the distributional family of interest, their methods rely on estimating uncertainty and identifying coreset. Additionally, their study is purely empirical and lacks theoretical guarantees. In a similar spirit, Li et al. (2024) study using active learning to select input functions from multiresolution datasets to lower the data cost.
On the theoretical side, a closely related work is by Kovachki et al. (2024a), who allow for active data collection strategies. However, the upper bound in (Kovachki et al., 2024a, Theorem 3.3) is derived assuming that input functions v 1 , . . . , v n are drawn i.i.d. from µ. Their proof, based on standard empirical risk minimization (ERM) analysis, achieves a convergence rate that, at best, matches the Monte Carlo rate of n -1/2 . Moreover, their lower bounds apply to both active and passive data collection strategies, suggesting that, for the nonparametric operator classes considered by Kovachki et al. (2024a), active learning provides no clear advantage over passive approaches. Exploring whether an adaptive data collection strategy, informed by the covariance of µ and targeting smaller subclasses within these broad nonparametric classes, could yield faster convergence rates remains an interesting direction for future research.
Additionally, Boullé et al. (2023) shares our objective of achieving faster convergence rates for PDEs with linear solution operators, but there are notable differences between their results and ours. First, their approach requires stronger control over the Hilbert-Schmidt norm of the operator F, whereas we only require control over the operator norm. Notably, the Hilbert-Schmidt norm can be arbitrarily larger than the operator norm. Second, their estimator uses the specific structure of F, particularly the Green's function, while we rely on black-box access to F via an ε-approximate oracle. Lastly, although both approaches introduces a term measuring the quality of training data (ε in our bound and Γ ε in theirs), their definition of Γ ε is more technical and less intuitive. However, their guarantee is stronger, as their upper bound applies uniformly to any L 2 -integrable input function, whereas our guarantees hold in expectation for inputs drawn from the distribution µ.
Our work considers the setting where µ is defined by a stochastic process with a specific covariance structure. Such a µ was taken to be a Gaussian process with mean zero and covariance given by α(-∇ 2 + βI) -γ in (Bhattacharya et al., 2021;Li et al., 2021;Kovachki et al., 2023). The use of Karhunen-Loève decomposition for generating input functions is also discussed by Boullé & Townsend (2023, Section 4.1). Our upper bound also share conceptual similarities with results in (Lanthaler et al., 2022;Lanthaler, 2023), who established approximation error bounds, rather than estimation, in terms of s eigenvalues of covariance operator.
Finally, we highlight the ICML 2024 tutorial by Azizzadenesheli (2024), who mentions active data collection as an important future direction for operator learning. We also acknowledge the extensive literature on the learning-theoretic foundations of active learning (Settles, 2009). The active learning framework we adopt is known as the membership query model, which has a rich history in learning theory (Angluin, 1988). A more detailed discussion of various active learning models within the learning theory literature is deferred to Section 3.4.
this section cite: ['b22', 'b17', 'b5', 'b18', 'b11', 'b15', 'b14', 'b3', 'b24', 'b0']

Section: Preliminaries

this section cite: []

Section: Notation
Let R, C denote the set of real and complex numbers respectively. The set N and Z denote the natural numbers and integers. Define N 0 := N ∪ {0}. For any x ∈ R d , we use |x| p to denote the ℓ p norm of x. Given a set X ⊆ R d , we use L 2 (X ) to denote the space of squared integrable realvalued functions on X under some base measure ν. For any u ∈ L 2 (X ), we define ∥u∥ 2 L 2 := X |u(x)| 2 dν(x). The notation ν is reserved for the base measure on X , whereas µ will be used to denote the probability distribution over L 2 (X ). For a linear operator F : L 2 (X ) → L 2 (X ), we define ∥F∥ op := sup{∥Fv∥ L 2 : ∥v∥ L 2 = 1}. We use GP to denote Gaussian Process.
this section cite: []

Section: Distribution Over Function Space
Let X ⊆ R d be any compact set, B(X ) denote the Borel sigma-algebra, and ν denote some finite measure on X (that is, ν(X ) < ∞). Generally, we will take ν to be Lebesgue measure on X but sometimes it may be useful to take a weighted measure such as ∝ e -α 2 |x| 2 dx. Denote L 2 (X , B(X ), ν) to be the set of all squared integrable functions on X . From here on, we will drop the dependence on B(X ) and ν, and just write L 2 (X ). Let (Ω, Σ, P) denote a probability space. We will consider a sequence of real-valued random variables {h x : x ∈ X } defined over the probability space (Ω, Σ, P) that is centered, squared integrable, and has continous covariance kernel K : X × X → R. Recall that covariance kernels are symmetric and positive definite. More precisely, for any x, y ∈ X , the random variables h x satisfies
E[h x ] = Ω h x (ω) dP(ω) = 0 E[h 2 x ] = Ω |h x (ω)| 2 dP(ω) < ∞ E[h x h y ] = Ω h x (ω) h y (ω) dP(ω) = K(y, x).
Next, we use this process to define a probability distribution over L 2 (X ). To that end, it will be more convenient to write the process as a function h : X × Ω → R. By definition, h(x, •) is Σ-measurable for every x ∈ X . However, this is not enough to argue that h is a random element of L 2 (X ). Thus, to ensure measurability, we will only consider stochastic processes h that satisfy the following: (i) The process h is measurable with respect to product sigma algebra B(X ) × Σ and (ii) For every ω ∈ Ω, the sample path h(•, ω) : X → R is an element of L 2 (X ). Conditions (i) and (ii) ensure that ω → h(•, ω) is a measurable function from Ω to L 2 (X ) (Hsing & Eubank, 2015, Theorem 7.4.1).
In other words, h is a L 2 (X ) valued random variable. We can now meaningfully talk about probability distribution over L 2 (X ) induced by the stochastic process h.
Accordingly, given a continuous covariance kernel K, let P(K) denote the set of all centered and squared-integrable stochastic processes with covariance kernel K indexed by X that satisfies conditions (i) and (ii) above. With a slight abuse of notation, we will also use P(K) to denote the set of all distributions over L 2 (X ) induced by these stochastic processes. Each element µ ∈ P(K) is now a probability distribution over L 2 (X ).
this section cite: []

Section: Problem Setting and Goal
Let F : L 2 (X ) → L 2 (X ) denote the operator of interest.
One should think of F as the solution operator of the PDE. The goal is to estimate a surrogate F n using n input/output functions such that
sup µ∈P(K) E v∼µ F n (v) -F(v) 2 L 2(1)
is small. In the absence of additional knowledge about the image space of the solution operator F, minimizing this objective is the most natural choice. Accordingly, for a fixed µ, the L p µ -Bochner norm has been a standard error metric in the operator learning literature (see (Kovachki et al., 2023, Section 2.2), (Liu et al., 2024)). The p = 2 case, in particular, is of practical significance, as its empirical counterpart is the widely used mean squared loss. Regarding the family of probability distributions, our proposed family P(K) aims to unify and generalize marginal distributions on input functions commonly used in practice (Li et al., 2021;Lu et al., 2021). This family also aligns with the recommendation of Boullé & Townsend (2023). Other families of probability distributions, such as the set of all compactly supported measures on a Hilbert space, have been used in theoretical analyses (e.g., (Liu et al., 2024)). Extending our result to include other distribution families of theoretical or applied interest is left for future work.
Throughout this work, we will assume that the learner knows the covariance kernel K. Assumption 2.1. The learner knows the kernel K.
Although not always explicitly stated, this has been a standard assumption in the operator learning literature. For example, Li et al. (2021) and Kovachki et al. (2023) generate their input functions, both during training and testing, from a Gaussian process with the covariance kernel K such that its associated integral operator is α(-∇ 2 + βI) -γ for some constants α, β, γ > 0. Thus, all the empirical performances observed in these works are in a setup similar to those described above. Additionally, Boullé & Townsend (2023, Section 4.1.1) also suggests generating source terms (input functions) from Gaussian processes with standard covariance kernels such as RBF, Mattern, etc.
Additionally, from a learning-theoretic perspective, assuming knowledge of the kernel K is arguably without loss of generality. In active learning, it is common to assume access to an unlimited pool of unlabeled samples v 1 , . . . , v m ∼ iid µ, where µ ∈ P(K), and focus on minimizing label complexity-the number of labeled samples requested (Hanneke, 2013). This aligns with our setting, where labeling (e.g., solving a PDE) is the primary cost. Given such unlabeled samples, one can estimate the covariance operator as (Hsing & Eubank, 2015) guarantees that Σ m → Σ almost surely in Hilbert-Schmidt norm, where Σ is the integral operator associated with K. While our work assumes Σ has a finite trace norm, this is not required to recover its eigenfunctions: convergence in Hilbert-Schmidt norm suffices for accurate spectral approximation. Thus, assuming access to the eigenfunctions of K is reasonable in theory, even if it may be computationally demanding in practice.
Σ m = 1 m -1 m i=1 (v i -vm ) ⊗ (v i -vm ) where vm = 1 m m i=1 v i . Since E[∥v i ∥ 2 ] < ∞, Theorem 8.1.2 of
Once the input functions are generated, the learner has to use numerical solvers to PDE numerically in order to generate the solution function. In this work, we will make the following assumption about learner's access to the PDE solver. Assumption 2.2. The learner only has black-box access to F through an ε-approximate oracle O that satisfies
sup v∈L 2 (X ) ∥O(v) -F(v)∥ 2 L 2 ≤ ε 2 .
From an implementation standpoint, it might seem unnatural to consider O(v) for a function v ∈ L 2 (X ), especially since most PDE solvers usually only take function values over a discrete grid as an input. Nevertheless, the oracle is an abstract object, and the grid can be integrated into its definition. For example, given any function v, the oracle first extracts the values of v on a grid {x 1 , . . . , x m } and produces output values on the same or a different grid. On the output side, the oracle may then construct an actual function, either through trigonometric interpolation or simply by setting the function values to zero outside the grid points. Thus, we do not specify these implementation details of the oracle and instead characterize it solely by accuracy parameter ε.
In general, ε primarily reflects the discretization error for finite-difference type methods and truncation error for spectral methods, but it may also include measurement errors or errors resulting from the early stopping of some iterative routine. Therefore, ε can be broadly viewed as quantifying the quality of the training data. From this perspective, ε represents the irreducible error in (1). Specifically, there exists a function g : [0, ∞) → [0, ∞) such that (1) is bounded below by g(ε), even as n → ∞. There is extensive literature that attempts to quantify ε for various oracles (PDE solvers), and we can use these results readily to establish bounds on the irreducible error in our context. For example, for spectral solvers truncated to the first N basis functions where the input and output functions are s-times continuously differentiable, we typically have ε ∼ N -s/d . Here, d is the dimension of the domain Ω.
this section cite: ['b11', 'b19', 'b18', 'b21', 'b19', 'b18', 'b11', 'b9', 'b10']

Section: Upper Bounds Under Active Data Collection
In Section 2.3, we discussed the problem setting and the goal. Next, we specify how the learner can collect the training data (v 1 , w 1 ), . . . , (v n , w n ). In a departure from the standard statistical learning setting, where the training data is obtained as iid samples from the distribution under which the learner is evaluated, we investigate active data collection strategies. In active data collection strategies, the learner can pick any source terms v 1 , . . . , v n and use the oracle to obtain w i = O(v i ). Since the goal is to provide guarantees under samples from the distribution µ ∈ P(K), the learner can use the knowledge of K to pick source terms. For a given oracle with accuracy ε, covariance kernel K, and the desired accuracy η > 0, the goal of the learner is to develop an active data collection strategy for the source terms and an estimation rule to produce F such that the accuracy of η can be obtained with the fewest number of oracle calls. Or equivalently, achieve an optimal decay in the upperbound of (1) for n ∈ N number of oracle calls. Under this model, we provide an upperbound on (1) when F is a bounded linear operator. Theorem 3.1 (Upper Bound). Suppose F is a bounded linear operator. There exists a deterministic data collection strategy and a deterministic estimation rule such that the estimate F n produced after n calls to oracle O satisfies
sup µ∈P(K) E v∼µ F n (v) -F(v) 2 L 2 ≤ ε 2 n i=1 λ i +∥F∥ 2 op i>n λ i .
Here, λ 1 ≥ λ 2 ≥ . . . are the eigenvalues of the integral operator of K defined as
(I K v)(•) = X K(•, x) v(x) dν(x).
The first term above is the irreducible error, which depends on the quality of the training data. For the second term, Hsing & Eubank (2015, Theorem 4.6.7) implies that
∞ i=1 λ i = X K(x, x) dν(x) ≤ sup x |K(x, x)| ν(X ) < ∞.
This is finite because ν is a finite measure on X , and K(x, x) is a continuous function on a compact domain, making it bounded. As a result, the second term in the upper bound of Theorem 3.1 vanishes as n → ∞. In Section 3.3, we apply Theorem 3.1 to derive precise rates for several common covariance kernels.
this section cite: []

Section: Data Collection Strategy and The Estimator
Here, we specify the data collection strategy and the estimator that achieves the claimed guarantee in Theorem 3.1. Let {λ j , φ j } ∞ j=1 be the sequence of eigenpairs of K defined by solving the Feldholm integral equation
X K(y, x) φ j (x) dν(x) = λ j φ j (y), y, x ∈ X .
Given the Oracle call budget of n, the input functions that the learner selects are φ 1 , φ 2 , . . . , φ n as source terms. For each i ∈ [n], the learner makes an oracle call and obtains w i = O(φ i ). Then, we consider the estimator
F n := n i=1 w i ⊗ φ i .
More precisely, this estimation rule yields an operator F n such that
F n v = n i=1 w i ⟨φ i , v⟩ L 2 for any v ∈ L 2 (X ). Appendix A.
1 provides an overview of the process for deriving this estimator starting from a least-squares estimation rule. Furthermore, Appendix C discusses methods for approximating the eigenfunctions φ i when the Fredholm integral equation cannot be solved exactly.
this section cite: []

Section: Sketch of a Proof of Theorem 3.1
We now provide a high-level, non-rigorous sketch of a proof of Theorem 3.1, and defer a full proof to Appendix A.
To bound the risk of the estimator specified above, we first rewrite the risk using Karhunen-Loève Theorem. Pick any v ∼ µ. Since v is defined using a centered and squaredintegrable stochastic process with continuous covariance kernel K, the celebrated Karhunen-Loève Theorem (Hsing & Eubank, 2015, Theorem 7
.3.5) states that v(•) = ∞ j=1 λ j ξ j φ j (•),
where ξ j 's are random variables defined as
ξ j := 1 √ λj X v(x) φ j (x) dν(x).
It turns out that ξ j 's are uncorrelated random variables with mean 0 and variance
1. That is, E[ξ j ] = 0 and E[ξ i ξ j ] = 1[i = j].
This decomposition allows us to rewrite expectation over µ in terms of expectation over the randomness of the sequence (ξ j ) j≥1 , which is more tractable. For simplicity, assume that ε = 0. Then, using Karhunen-Loève expansion, we can show that
E v∼µ F n (v) -F(v) 2 L 2 ≤ E ξ F j>n λ j ξ j φ j 2 L 2 ,
which can then be further upper bounded by ∥F∥ 2 op j>n λ j using properties of ξ i 's.
There are two primary challenges in completing this argument in a fully rigorous manner. First, we must address the fact that the oracle O is only ε-approximate for any ε > 0. Second, the convergence statement for
∞ j=1 λ j ξ j φ j (•)
is quite specific, requiring careful attention when applying this result.
this section cite: []

Section: Examples of Covariance Kernels
To make the upperbound in Theorem 3.1 more concrete, let us consider a few specific covariance kernels K. While not all claims are rigorously proven in this subsection, a detailed and formal treatment of the material can be found in Appendix B. Li et al. (2021); Kovachki et al. (2023) generated input functions from GP(0, α(-∇ 2 + βI) -γ ) for some constants α, β, γ > 0. Here, ∇ 2 is the Laplacian operator defined as
this section cite: ['b18', 'b11']

Section: FRACTIONAL INVERSE OF SHIFTED LAPLACIAN
∇ 2 v = d j=1 ∂ 2 v ∂x 2 j .
In this section, we will consider X to be a d-dimensional periodic torus T d and the base measure ν is Lebesgue. We identify T d by [0, 1] d with periodic boundary conditions. Let us define a function φ m :
T d → C as φ m (x) = e 2π i m•x for every m ∈ Z d . Recall that φ m is the eigenfunction of ∇ 2 with eigenvalue -4π 2 |m| 2 2 . In particular, ∇ 2 e 2π i m•x = d j=1 ∂ 2 ∂x 2 j e 2π i m•x = -4π 2 |m| 2 2 e 2π i m•x .
Since {φ m : m ∈ Z d } forms a complete orthonormal system in L 2 (T d ), there are no other eigenfunctions of ∇ 2 . A simple algebra shows that φ m 's are also the eigenfunctions of (-∇ 2 + βI) -γ with eigenvalues being (β + 4π 2 |m| 2 2 ) -γ . Using this fact in Theorem 3.1 yields the upper bound
≤ ε 2 αβ -γ + α + α 2γ -d + α ∥F∥ 2 op 2γ -d 1 n 2γ d -1 .
When 2γ/d -1 > 0, the reducible error above goes to 0 when n → ∞. Again, as an example, Li et al. (2021) uses α = 7 3/2 , β = 49 and γ = 2.5 in their experiment for 2d-Navier Stokes. In this case, 2γ/d = 2.5, yielding the convergence rate of n -1.5 for the reducible error. Note that this rate is faster than the usual passive statistical rate of 1/n. However, for any value τ , one can take γ = d(τ + 1)/2 to get the rate of n -τ . Thus, every polynomial rate is possible depending on the choice of γ.
this section cite: ['b18']

Section: RBF KERNEL
Let X = R and K(x, y) = exp -1 2ℓ 2 |x -y| 2 for all x, y ∈ R and ℓ > 0. For now, let ν is a Gaussian measure with mean 0 and variance σ 2 on R. Using the known results on eigenfunctions of RBF kernel in terms of Hermite polynomials (Williams & Rasmussen, 2006, Section 4.3.1), we show that there exists γ ∈ (0, 1) such that the upper bound in Theorem 3.1 is
≤ 1 (1 -γ) ε 2 + ∥F∥ 2 op γ n .
That is, the reducible error vanishes exponentially fast as n → ∞. In Appendix B, we also show that a rate faster than any polynomial rate can be achieved for RBF kernel on R d .
this section cite: ['b26']

Section: BROWNIAN MOTION
Let us consider the case where X = [0, 1], the base measure ν is Lebegsue, and the stochastic process in Section 2.2 is Brownian motion. Recall that the Brownian motion is a Gaussian process with covariance kernel K(s, t) = min(s, t) for all s, t ∈ [0, 1]. It is well-known (Hsing & Eubank, 2015, Example 4.6.3) that the eigenfunctions of K can be written in terms of sine waves. A simple analysis can then be used to establish an upper bound of
≤ ε 2 2 + ∥F∥ 2 op 1 π 2 2 2n -1 .
Therefore, the reducible error vanishes at rate ∼ n -1 .
this section cite: ['b10']

Section: Comparison to Traditional Active Learning
The active learning framework we adopt in this work is referred to as the membership query model, which has a longstanding history in the learning theory literature (Angluin, 1988;2001). However, in traditional learning settings, the membership query model-where the learner can request labels for any unlabeled instance-is generally unrealistic. For example, in the context of human data, it may not be feasible to generate a label for an individual with an arbitrary feature vector, as such a person may not exist in reality.
As a result, other active learning frameworks, such as the stream-based sampling model (Atlas et al., 1989) and the pool-based model (Lewis & Gale, 1994;Hanneke, 2013), have gained prominence in the recent literature. These models restrict the learner to requesting labels for instances sampled from a specific distribution, making them more practical for many real-world applications. For a comprehensive review of active learning models, their history, and key results, we refer readers to (Settles, 2009). That said, we believe that the membership query model is the right model for developing surrogates for solution operators of PDEs. This is because a PDE solver can provide a solution to any query of an input function within an appropriate function space.
this section cite: ['b0', 'b1', 'b2', 'b16', 'b9', 'b24']

Section: Lower Bounds on Passive Learning
In this section, we establish a lower bound on (1) for any passive data collection strategy. As usual, the kernel K is known to the learner. Nature selects a distribution µ ⋆ ∈ P(K), and the learner receives n i.i.d. samples v 1 , v 2 , . . . , v n ∼ µ ⋆ . For each i ∈ [n], the learner queries the oracle O to produce w i = O(v i ). It is important to emphasize that the learner can only make oracle calls for the i.i.d. samples v 1 , v 2 , . . . , v n . If the learner were allowed to make oracle calls for other input functions, the learner could simply disregard these i.i.d. samples and implement the "active strategy" from Section 3.1. Such restriction on oracle calls still includes most passive learning rules of interest, such as arbitrary empirical risk minimization (ERM), regularized least-squares estimators, and parametric operators trained with stochastic gradient descent.
Using these n training points {(v i , w i )} i≤n , the learner then constructs an operator F n . Since the learner only has access to samples from µ ⋆ , it is unrealistic to expect a uniform guarantee over the entire family P(K) as established in Theorem 3.1. Therefore, in this section, the learner will be evaluated solely under the distribution µ ⋆ . The objective is to minimize the expected loss under µ ⋆ , defined as
E v1:n∼µ n ⋆ E v∼µ⋆ F n (v) -F(v) 2 L 2 .
Moreover, establishing any meaningful lower bound on this risk requires imposing some restriction on the oracle O. To understand why, consider the case where F is a finite-rank operator that only maps to the span of {ψ 1 , ψ 2 , . . . , ψ N } for some orthonormal sequence ψ 1 , . . . , ψ N in L 2 (X ). Now, consider an oracle O such that for any v ∈ L 2 (X ), it outputs
O(v) = F(v) + χ ψ N +1 ,
where χ ∈ R and
ψ N +1 is a unit norm function in L 2 (X ) that is orthogonal to all ψ j for 1 ≤ j ≤ N . If |χ| ≤ ε, it is easy to see that sup v∈L 2 (X ) ∥O(v) -F(v)∥ 2 L 2 = ∥χ ψ N +1 ∥ 2 L 2 = |χ| 2 ≤ ε 2 .
Thus, O is a valid oracle according to Assumption 2.2. However, in principle, it is possible to encode the entire identity of F in a real number χ. Thus, the learner could determine the identity of F with just a single call to O, making any attempt at establishing a lower bound futile.
This problem may still persist even when ε = 0. Consider the case where ν is the Lebesgue measure, and the oracle is of the form
O(v) = F(v) + χ1[x = x 0 ]
for some x 0 ∈ X . Then, for any v ∈ L 2 (X ), we have
∥O(v) -F(v)∥ 2 L 2 = ∥χ1{x = x 0 }∥ 2 L 2 = 0 as ν({x 0 }) = 0.
This shows that the oracle can still reveal the identity of F in regions of the domain that have zero measure under ν. Therefore, to avoid these pathological edge cases, we will assume that the oracle is perfect.
Definition 4.1 (Perfect Oracle). O is a perfect oracle for F if, for every v ∈ L 2 (X ), we have
O(v) (x) = F(v) (x) ∀x ∈ X .
In other words, the perfect oracle O produces exactly the same function that F does-nothing more, nothing less. With this assumption, we are in the usual realizable setting often considered in statistical learning theory. That is, the learner has access to n samples {(v i , F(v i ))} n i=1 , where v 1 , . . . , v n are drawn iid from some distribution µ.
Theorem 4.2 provides a lower bound on the risk of any estimator under such passive data collection strategy.
Theorem 4.2 (Lowerbound). Fix any continuous covariance kernel K with eigenvalues λ 1 ≥ λ 2 ≥ . . .. Then, there exists a solution operator F, accessible to the learner through a perfect oracle O, such that the following holds: for every n ∈ N, there exists a distribution µ ∈ P(K) such that, under any estimation rule within a passive data collection strategy, the risk of the resulting estimator F n is
E v1:n∼µ n E v∼µ F n (v) -F(v) 2 L 2 ≥ ∥F∥ 2 op 2 m j=1 λ j
for every fixed m ∈ N.
Specifically, for m = 1, we obtain a lower bound of 1 2 ∥F∥ 2 op λ 1 . This provides a non-vanishing lower bound for any non-trivial operator F and covariance kernel K.
Our lower bound is constructive: we explicitly define a difficult distribution for the learner. We construct a distribution µ over input functions such that, along each eigenfunction direction φ j , it places mass 0 with probability 1 -p, and ±1/ √ p with equal probability p/2. This yields a sparse distribution with rare but large spikes. A careful argument shows that this construction defines a valid distribution in P(K) for any p > 0. When p is small, the learner observes mostly zero inputs during training with probability at least 1/2, yet the expected squared error along each direction is (1/ √ p) 2 • p = 1, leading to a non-vanishing error. The full proof is provided in Appendix D.
this section cite: []

Section: Experiments
In this section, we conduct numerical studies comparing our active data collection strategy with passive data collection (random sampling) for learning solution operators for the Poisson and Heat Equations. For the actively collected data, we implement the linear estimator defined in Section 3.1. On the other hand, for passively collected data, we use a least-squares estimator, where the pseudoinverse is computed numerically. Recall that, given inputoutput functions {v i , w i } n i=1 , the least-squares estimator has a form L = (
n i=1 w i ⊗ v i ) ( n i=1 v i ⊗ v i ) † .
For the actively collected data in Section 3.1, the v i 's are orthogonal, which yielded a simple and natural pseudoinverse (see Appendix A.1). However, for the passively collected data, the v i 's may not be orthogonal anymore and the pseudoinverse does not have a nice closed form. Thus, we use standard numerical techniques to compute the pseudo-inverse
( n i=1 v i ⊗ v i ) † .
However, in practice, one rarely uses linear estimators for passively collected data. Thus, we also compare our method against the Fourier Neural Operator (Li et al., 2021), the most popular architecture for operator learning. Our code is available at https://github.com/  unique-subedi/active-operator-learning.
this section cite: ['b18']

Section: Poisson Equation
Let X = [0, 1] 2 . Consider Poisson equation with Dirichlet boundary conditions:
-∇ 2 u = f, u(x) = 0 ∀x ∈ boundary(X ),
where ∇ 2 is the Laplace operator. The objective is to learn the solution operator that maps the source function f to the solution u. This solution operator is the inverse of the Laplacian, which is a compact linear operator since X is bounded.
For the passive data collection strategy, the input functions f are independently sampled as f ∼ GP(0, 50 2 (-∇ 2 +I) -2 ), where GP denotes Gaussian Process.
The solution u is computed using the finite-difference method. Both linear estimators and Fourier Neural Operators (FNO) are trained on n such independently sampled pairs (f, u). For testing, 100 additional source functions f ∼ GP(0, 50 2 (-∇ 2 + I) -2 ) are generated, and their corresponding solutions u are also obtained via the finitedifference method. Both active and passive estimators are evaluated on this test set, with the performance measured using the mean-squared relative error:
Error = 1 n test ntest i=1 u true i -u predicted i 2 L 2 ∥u true i ∥ 2 L 2 .
We report the relative error instead of the absolute error to normalize for potential arbitrary scaling due to the norms   While the convergence guarantee of our estimator is formally established only for the covariance operator 50 2 (-∇ 2 + I) -γ with γ > 1 as d = 2, we observe that the estimator demonstrates robust convergence even when γ ≤ 1 in the context of Poisson equation.
this section cite: []

Section: Heat Equation
Consider the heat equation
∂u ∂t = τ ∇ 2 u,
where u : [0, 1] 2 → R vanishes on the boundary. The solution operator for this equation is given by exp(τ t∇ 2 ), and the solution at time t ≥ 0 can be expressed as u t = exp(τ t∇ 2 )u 0 . Fixing t = 1, our objective is to learn the solution operator exp(τ ∇ 2 ). This operator is defined as
exp(τ ∇ 2 ) = ∞ k=0 (τ ∇ 2 ) k k! ,
which is a bounded linear operator. As in the previous case, we sample n initial conditions u 0 ∼ GP(0, (-∇ 2 + I) -1.5 ).
For each initial condition, we use the finite difference method with forward-time discretization to compute the solution u 1 at t = 1. This is done using 1000 time discretization steps on a 64 × 64 grid. For our experiments, we set τ = 10 -2 . As τ is the step size in the forward Euler method, choosing a larger τ would result in instability in the numerical PDE solver.
All estimators are evaluated on a test set of size 100, drawn from the same distribution as the training data. Figure (4) presents the relative testing errors. Furthermore, the error plot for the Fourier Neural Operator (FNO) trained on actively collected data is shown in Figure Our experimental results verify the theoretical advantage of active data collection strategies over passive sampling, as established in Theorem 3.1. These findings highlight the practical utility of active learning frameworks in improving data efficiency for operator learning tasks.
this section cite: []

Section: Discussion and Future Work
In this work, we show that arbitrarily fast rates can be achieved with an active data collection strategy when the operator of interest is a bounded linear operator and the input functions are drawn from centered distributions with continuous covariance kernels. A natural extension of these results would involve non-linear operators. Specifically, one might ask whether there exists a natural class of non-linear operators that permits such fast rates when input functions are drawn from centered distributions with continuous covariance kernels. A natural starting point might be to consider the RKHS of operators. Additionally, given that functional PCA is the estimation of truncated Karhunen-Loève decomposition, it would be interesting to explore whether a variant of a PCANet-based architecture could achieve fast rates with active data collection.
this section cite: []

Section: Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning for scientific applications. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.
this section cite: []

Section: A. Proof of Theorem 3.1 A.1. Specifying Data Collection Strategy and The Estimator
We first specify the estimator that achieves the claimed guarantee in Theorem 3.1. Let {λ j , φ j } ∞ j=1 be the sequence of eigenpairs of K defined by solving the Feldholm integral equation X K(y, x) φ j (x) dν(x) = λ j φ j (y), y, x ∈ X .
Given the Oracle call budget of n, the input functions that the learner selects are φ 1 , φ 2 , . . . , φ n as source terms. For each i ∈ [n], the learner makes an oracle call and obtain
w i = O(φ i ).
Consider the estimation rule
arg min L is linear n i=1 ∥Lφ i -w i ∥ 2 L 2 .
Solving this optimization problem boils down to solving the linear equation
n i=1 w i ⊗ φ i = L • n i=1 φ i ⊗ φ i .
It is clear that this system is ill-posed and has infinitely many solutions. The family of solutions can be written as
L = n i=1 w i ⊗ φ i n i=1 φ i ⊗ φ i † ,
where † indicates the pseudoinverse. Each particular choice of pseudoinverse yields a distinct solution. Since φ i 's are orthonormal, a natural one is
n i=1 φ i ⊗ φ i † = n i=1 φ i ⊗ φ i .
This choice of pseudoinverse yields the estimator
F n := n i=1 w i ⊗ φ i ,
which will be our estimator interest.
this section cite: []

Section: A.2. Rewriting Risk using Karhunen-Loève Theorem
Next, we bound the risk of this estimator. Pick any v ∼ µ. Since v is defined using a centered and squared-integrable stochastic process with continuous covariance kernel K, the celebrated Karhunen-Loève Theorem (Hsing & Eubank, 2015, Theorem 7.3.5) states that
v(•) = ∞ j=1 λ j ξ j φ j (•),
where ξ j 's are random variables defined as
ξ j := 1 λ j X v x (ω) φ j (x) dν(x).
Here, v x (ω) is simply just v(x), but we write the dependence on ω explicitly to highlight the fact that v is generated by stochastic process on the probability space (Ω, Σ, P).
It turns out that ξ j 's are uncorrelated random variables with mean 0 and variance 1. In particular, we have
E[ξ j ] = 0 and E[ξ i ξ j ] = 1[i = j].
The precise convergence statement is
lim m→∞ sup x∈X E    v(x) - m j=1 λ j ξ j φ j (x) 2    = 0.(2)
We refer the reader to standard texts (Hsing & Eubank, 2015, Theorem 7.3.5) or (Lord et al., 2014, Theorem 7.52) for the full proof of Karhunen-Loève Theorem.
Fix m ∈ N such that m > n and define Π m to be a projection operator onto the first m eigenfunctions of K. That is, for each v with Karhunen-Loève decomposition v(
•) = ∞ j=1 λ j ξ j φ j (•), we define Π m (v) := m j=1 λ j ξ j φ j (•).
Since both F n and F are linear operators, we can write
E v∼µ F n (v) -F(v) 2 L 2 = E v∼µ F n Π m (v) -F Π m (v) + F n -F v -Π m (v) 2 L 2 ≤ E v∼µ F n Π m (v) -F Π m (v) 2 L 2 + 2 E F n Π m (v) -F Π m (v) L 2 F n -F v -Π m (v) L 2 + E F n -F v -Π m (v) 2 L 2
The inequality follows upon using triangle inequality and expanding the square. For the cross term, Cauchy-Schwarz inequality yields
E F n Π m (v) -F Π m (v) L 2 F n -F v -Π m (v) L 2 ≤ E F n Π m (v) -F Π m (v) 2 L 2 E F n -F v -Π m (v) 2 L 2 .
Thus, we can write
E v∼µ F n (v) -F(v) 2 L 2 ≤ (I) + 2 (I) (II) + (II),
where we define
(I) := E v∼µ F n Π m (v) -F Π m (v) 2 L 2 (II) := E F n -F v -Π m (v) 2 L 2 .
Next, we will bound (I) and (II) separately.
this section cite: []

Section: A.3. Bounding (I).
Pick any v ∼ µ. Then, we know that there exists
{ξ j } j∈N such that v = ∞ j=1 λ j ξ j φ j . So, Π m (v) = m j=1
λ j ξ j φ j , which subsequently implies that
F n Π m (v) = F n   m j=1 λ j ξ j φ j   = m j=1 λ j ξ j F n (φ j ) = n j=1 λ j ξ j w j ,
where the final equality uses the fact that n < m and F n (φ j ) = 0 for all j > n. Defining δ i := O(φ i ) -F(φ i ), we obtain w i = F(φ i ) + δ i . This allows us to write
F n Π m (v) = n i=1 λ i ξ i (F(φ i ) + δ i ) = F n i=1 λ i ξ i φ i + n i=1 λ i ξ i δ i = F (Π m (v)) -F   m j=n+1 λ j ξ j φ j   + n i=1 λ i ξ i δ i .
So, we can rewrite (I) as
E v∼µ F n Π m (v) -F Π m (v) 2 L 2 = E ξ    n i=1 λ i ξ i δ i -F   m j=n+1 λ j ξ j φ j   2 L 2    = E ξ   n i=1 λ i ξ i δ i 2 L 2   -2 E ξ   n i=1 λ i ξ i δ i , F   m j=n+1 λ j ξ j φ j     + E ξ    F   m j=n+1 λ j ξ j φ j   2 L 2    .
The cross-term vanishes upon swapping sum and integral as ξ i 's are zero mean and uncorrelated. For the first term, note that
E ξ   n i=1 λ i ξ i δ i 2 L 2   = E ξ n i=1 λ i ξ i δ i , n i=1 λ i ξ i δ i L 2 = n i=1 λ i E[ξ 2 i ] ∥δ i ∥ L 2 + 2 1≤i<j≤n λ i λ j E[ξ i ξ j ] ⟨δ i , δ j ⟩ L 2 = n i=1 λ i ∥δ i ∥ 2 L 2 + 0 ≤ ε 2 n i=1 λ i .
The final inequality uses the fact that O is ε-approximate for F. For the third term, similar arguments show that
E ξ    F   m j=n+1 λ j ξ j φ j   2 L 2    ≤ ∥F∥ 2 op E ξ    m j=n+1 λ j ξ j φ j 2 L 2    = ∥F∥ 2 op m j=n+1 λ j ∥φ j ∥ 2 L 2 = ∥F∥ 2 op m j=n+1 λ j .
Thus, we have established that
(I) ≤ ε 2 n i=1 λ i + ∥F∥ 2 op m j=n+1 λ j .
this section cite: []

Section: A.4. Bounding (II)
For any v ∼ µ, we have
E v∼µ F n -F v -Π m (v) 2 L 2 ≤ F n -F 2 op E ∥v -Π m (v)∥ 2 L 2 Let v = j≥1 λ j ξ j φ j . Then, E ∥v -Π m (v)∥ 2 L 2 = E    v - m j=1 λ j ξ j φ j 2 L 2    = E    X   v(x) - m j=1 λ j ξ j φ j (x)   2 dν(x)    = X E      v(x) - m j=1 λ j ξ j φ j (x)   2    dν(x) ≤ ν(X ) • sup x∈X E      v(x) - m j=1 λ j ξ j φ j (x)   2    .
The third equality uses joint measurability, finiteness of ν, and Tonelli's theorem to exchange the order or integration. Therefore, we have established that
(II) ≤ F n -F 2 op • ν(X ) • sup x∈X E      v(x) - m j=1 λ j ξ j φ j (x)   2    .
this section cite: []

Section: A.5. Combining (I) and (II)
For each m > n, we have established
E v∼µ F n (v) -F(v) 2 L 2 ≤ (I) + 2 (I) (II) + (II),where
(I) ≤ ε 2 n i=1 λ i + ∥F∥ 2 op m j=n+1 λ j (II) ≤ F n -F 2 op • ν(X ) • sup x∈X E      v(x) - m j=1 λ j ξ j φ j (x)   2    .
It now remains to choose m such that the upperbound is minimized. To that end, we will take m → ∞. Since w 1 , . . . , w n ∈ L 2 (X ), we must have F n op < ∞. As F is also a bounded linear operator, for any n ∈ N, we must have
F n -F 2 op < ∞.
Importantly, the norm of F n -F may grow with n, but is independent of m and does not grow as m → ∞. Moreover, as ν is a finite measure, we must have ν(X ) < ∞. Therefore, Karhunen-Loève Theorem (Hsing & Eubank, 2015, Theorem 7.3.5) (also see Equation ( 2)) implies that
(II) ≤ F n -F 2 op • ν(X ) • sup x∈X E      v(x) - m j=1 λ j ξ j φ j (x)   2    m→∞ ----→ 0.
On the other hand,
(I) m→∞ ----→ ε 2 n i=1 λ i + ∥F∥ 2 op ∞ j=n+1 λ j .
Overall, we have shown that
E v∼µ F n (v) -F(v) 2 L 2 ≤ ε 2 n i=1 λ i + ∥F∥ 2 op ∞ j=n+1 λ j .
This completes our proof of Theorem 3.1.
this section cite: []

Section: B. Examples of Covariance Kernels
In this section, we build upon and present a more rigorous analysis of the material discussed in Section 3.3 of the main text. Li et al. (2021) and Kovachki et al. (2023) generated input functions from GP(0, α(-∇ 2 + βI) -γ ) for some constants α, β, γ > 0. Here, ∇ 2 is the Laplacian operator defined as
this section cite: ['b18', 'b11']

Section: B.1. Fractional Inverse of Shifted Laplacian
∇ 2 v = d j=1 ∂ 2 v ∂x 2 j .
In this section, we will consider X to be a d-dimensional periodic torus T d and the base measure ν is Lebesgue. We identify T d by [0, 1] d with periodic boundary conditions.
Let us define a function φ m : T d → C as φ m (x) = e 2π i m•x for every m ∈ Z d . Recall that φ m is the eigenfunction of ∇ 2 with eigenvalue -4π 2 |m| 2 2 . In particular,
∇ 2 e 2π i m•x = d j=1 ∂ 2 ∂x 2 j e 2π i m•x = d j=1 (2π i m j ) 2 e 2π i m•x = -4π 2 |m| 2 2 e 2π i m•x .
Since {φ m : m ∈ Z d } forms a complete orthonormal system in L 2 (T d ), there are no other eigenfunctions of ∇ 2 . A simple algebra shows that φ m 's are also he eigenfunctions of shifted Laplacian -∇ 2 + βI with eigenvalues being (β + 4π 2 |m| 2 2 ). Finally, the spectral mapping theorem implies that {(λ m , φ m ) : m ∈ Z d } is the sequence of eigenpairs of α(-∇ 2 + βI) -γ , where the eigenvalues are
λ m = α β + 4π 2 |m| 2 2 -γ .
Next, we need to show that these eigenvalues are summable to use Theorem 3.1. Note that
m∈Z d λ m = m∈Z d α β + 4π 2 |m| 2 2 -γ ≤ αβ -γ + α (2π) 2γ m∈Z d \{0} |m| -2γ ∞ .
It is easy to see that
m∈Z d \{0} |m| -2γ ∞ ≤ ∞ j=1 j -2γ (2j + 1) d ≤ 3 d ∞ j=1 j -2γ+d < ∞
as long as 2γ > d. The first inequality holds because |{m ∈ Z d \{0} : |m| ∞ = j}| ≤ 2(2j + 1) d-1 . This is true because at least one of the entries has to be ±j and other d -1 entries could be anything in {0, ±1, . . . , ±j}. So we have
m∈Z d |λ m | < ∞, implying that the operator α(-∇ 2 + βI) -γ is in trace class as long as 2γ > d.
Finally, it is easy to see that the operator α(-∇ 2 + βI) -γ is integral operator associated with the kernel
K(y, x) = m∈Z d λ m φ m (y) φ m (x).
Upon writing the Fourier series of v ∈ L 2 (T), it is obvious that α(-
∇ 2 + βI) -γ v (y) = X K(y, x)v(x) dx for all y ∈ X . As m∈Z |λ m | < ∞,
the convergence is absolute and uniform. Since K is a uniform limit of the sum of continuous functions, K must also be continuous. Moreover, as λ m = λ -m , the kernel K must be real-valued. In particular, we have
K(y, x) = m∈Z d λ m φ m (y) φ m (x) = m∈Z d λ m 2 φ m (y) φ m (x) + φ -m (y) φ -m (x) = m∈Z d λ m cos 2πm • (y -x) = m∈Z d λ m cos(2πm • y) cos(2πm • x) + sin(2πm • y) sin(2πm • x)
This is a generalization of the cosine covariance kernel often considered in computational PDE literature (see (Lord et al., 2014, Example 5.20)). Since cos(θ) = cos(-θ), it is obvious that K is symmetric. As K is a continuous and real-valued covariance kernel defined on a bounded domain T d , we can use Theorem 3.1 for such K. In principle, we could use the Fourier modes φ m 's as source terms to define the estimator discussed in Section 3.1. However, the proof of Theorem 3.1 assumes that the eigenfunctions of the kernel are real-valued. So, we will first show that we can write the eigenfunctions of K solely using sine and cosine functions without having to use complex exponentials. This allows us to use these sine and cosine functions to define the estimator discussed in Section 3.1, and invoke results of Theorem 3.1.
this section cite: []

Section: B.1.1. CASE d = 1
When d = 1, it is easy to see that {1} ∪ { √ 2 cos(2πjx), √ 2 sin(2πjx) : j ∈ N} are the eigenfunctions of K. Writing the expansion of K and using Fubini's to switch the sum and the integral, we get
T K(y, x) √ 2 cos(2πjx) dx = λ j √ 2 cos(2πjy) 1 2 + λ -j √ 2 cos(-2πjy) 1 2 = λ j √ 2 cos(2πjy).
Note that the first equality holds because cos(2πjx) is orthogonal to all other cosine and sine functions except for cos(2πjx) and cos(-2πjx). The final equality holds because λ j = λ -j and cos(θ) = cos(-θ). A similar calculation shows that
T K(y, x) √ 2 sin(2πjx) dx = λ j √ 2 sin(2πjy) 1 2 + λ -j √ 2 sin(-2πjy) -1 2 = λ j √ 2 sin(2πjy).
Finally, we have T K(y, x) 1 dx = λ 0 1. Thus, λ j for j ∈ N are the eigenvalues for sine/cosine functions and λ 0 for 1. Since {1} ∪ { √ 2 cos(2πjx), √ 2 sin(2πjx) : j ∈ N} forms a complete orthonormal system of L 2 (T, R), there cannot be any more eigenfunctions of K. Next, we will plug in the values of λ j 's in Theorem 3.1 to get the precise rates.
Pick an odd n ∈ N and suppose the n input terms used to construct the estimator in Section 3.1 are {1} ∪ { √ 2 cos(2πjx), √ 2 sin(2πjx) : j ≤ (n -1)/2}. Then, the upperbound is
ε 2   λ 0 + (n-1)/2 j=1 2λ j   + ∥F∥ 2 op ∞ j=(n+1)/2 2λ j ≤ ε 2 αβ -γ + ε 2 2 (n-1)/2 j=1 α β + 4π 2 j 2 -γ + 2 ∥F∥ 2 op ∞ j=(n+1)/2 α β + 4π 2 j 2 -γ ≤ ε 2 αβ -γ + ε 2 α 2 (4π 2 ) γ (n-1)/2 j=1 1 j 2γ + 2α (4π 2 ) γ ∥F∥ 2 op ∞ j=(n+1)/2 1 j 2γ ≤ ε 2 αβ -γ + ε 2 α 2 (4π 2 ) γ + ε 2 α 2 (4π 2 ) γ (n-1)/2 1 t -2γ dt + 2α (4π 2 ) γ ∥F∥ 2 op ∞ (n-1)/2 t -2γ dt ≤ ε 2 αβ -γ + ε 2 α 2 (4π 2 ) γ + 2 (4π 2 ) γ ε 2 α 2γ -1 + 2α (4π 2 ) γ ∥F∥ 2 op 1 2γ -1 2 2γ-1 (n -1) 2γ-1 , ∀γ > 1 2 . Since 2 • 2 2γ-1 ≤ (4π 2 ) γ and 2 2γ-1 ≤ (4π 2 ) γ , the overall error is at most ε 2 αβ -γ + α + α 2γ -1 + α ∥F∥ 2 op 2γ -1 1 (n -1) 2γ-1 for all γ > 1 2 .
Since γ > 1/2, the reducible error goes to 0 as n → ∞. As an example, (Li et al., 2021) uses α = 625, β = 25 and γ = 2 in their experiment for 1d-Burger's equation. In this case, we get the convergence rate of n -3 for the reducible error. Note that this rate of cubic order is faster than the usual passive statistical rate of 1/n. In fact, for any value τ , one can take γ = (τ + 1)/2 to get the rate of n -τ . Thus, every polynomial rate is possible depending on the choice of γ.
this section cite: ['b18']

Section: B.1.2. CASE d > 1
Recall that {1} ∪ { √ 2 cos(2πjx), √ 2 sin(2πjx) : j ∈ N} are the eigenvalues of K for d = 1 with eigenvalues λ j := α β + 4π 2 j 2 -γ . Define a set of functions
E = d i=1 {1} ∪ { √ 2 cos(2πjx i ), √ 2 sin(2πjx i ) : j ∈ N}.
For each element e ∈ E , there exists a tuple j := (j 1 , . .
. , j d ) ∈ N d 0 such that e(x) = ψ j1 (x 1 ) . . . ψ j d-1 (x d-1 ) • ψ j d (x d ), where ψ ji (x i ) ∈ { √ 2 cos(2πj i x i ), √ 2 sin(2πj i x i )} for j i > 0 and √ 2 for j i = 0.
Let us denote the collection of all such functions by E j . Then, we have E = ∪ j∈N d 0 E j . We prove the following result on the eigenpairs of K.
Proposition B.1. For each j ∈ N d 0 , define λ j = α β + 4π 2 |j| 2 2 -γ . Then, j∈N d 0 e∈Ej {(λ j , e)}
is the set of eigenpairs of K on T d .
We defer the full proof of Proposition B.1 to the end of this subsection. First, we use Proposition B.1 and Theorem 3.1 to get the precise rate for kernel K. Pick r such that the source terms used to construct the estimator defined in Section 3.1 are
j∈N d 0 : |j|∞≤r E j Note that |E 0 | = 1 and |E j | ≤ 2 d for all |j| ∞ > 0.
Thus, there are n ≤ (r + 1) d 2 d source terms. Then, the upperbound is
≤ ε 2 λ 0 + 0<|j|∞≤r 2 d λ j + ∥F∥ 2 op |j|∞>r 2 d λ j = ε 2 αβ -γ + ε 2 2 d 0<|j|∞≤r α β + 4π 2 |j| 2 2 -γ + 2 d ∥F∥ 2 op |j|∞>r α β + 4π 2 |j| 2 2 -γ = ε 2 αβ -γ + ε 2 2 d 0<|j|∞≤r α β + 4π 2 |j| 2 ∞ -γ + 2 d ∥F∥ 2 op |j|∞>r α β + 4π 2 |j| 2 ∞ -γ ≤ ε 2 αβ -γ + ε 2 2 d r k=1 α (β + 4π 2 k 2 ) -γ (k + 1) d-1 + 2 d ∥F∥ 2 op ∞ k=r+1 α β + 4π 2 k 2 -γ (k + 1) d-1 ≤ ε 2 αβ -γ + ε 2 2 d α2 d (4π 2 ) γ r k=1 k d-1-2γ + 2 d ∥F∥ 2 op α 2 d (4π 2 ) γ ∞ k=r+1 k d-1-2γ ≤ ε 2 αβ -γ + ε 2 α2 2d (4π 2 ) γ + ε 2 α2 2d (4π 2 ) γ r 1 t d-1-2γ dt + ∥F∥ 2 op α 2 2d (4π 2 ) γ ∞ r t d-1-2γ dt ≤ ε 2 αβ -γ + ε 2 α2 2d (4π 2 ) γ + ε 2 α2 2d (4π 2 ) γ 1 2γ -d + ∥F∥ 2 op α 2 2d (4π 2 ) γ 1 2γ -d 1 r 2γ-d , for all 2γ > d. Recall that n ≤ (2r + 2) d . So, we have n 1/d /2 -1 ≤ r. For n 1/d ≥ 4, we have r ≥ n 1/d /4. Thus, 1 r 2γ-d ≤ 4 2γ-d n 2γ d -1 Note that (4π 2 ) γ = (2π) 2γ ≥ 2 2d 4 2γ-d . Moreover, as 2γ > d, we also have (4π 2 ) γ ≥ 2 2d . Therefore, our upper bound is at most ε 2 αβ -γ + α + α 2γ -d + α ∥F∥ 2 op 2γ -d 1 n 2γ d -1 .
Since 2γ/d -1 > 0, the reducible error above goes to 0 when n → ∞. Again, as an example, (Li et al., 2021) uses α = 7 3/2 , β = 49 and γ = 2.5 in their experiment for 2d-Navier Stokes. In this case, 2γ/d = 2.5, yielding the convergence rate of n -1.5 for the reducible error. Note that this rate is faster than the usual passive statistical rate of 1/n. However, as usual, for any value τ , one can take γ = d(τ + 1)/2 to get the rate of n -τ . Thus, every polynomial rate is possible depending on the choice of γ.
We now end this section by providing the proof of Theorem B.1.
Proof of Proposition B.1. Since ∪ j∈N d 0 ∪ e∈Ej {e} forms an orthonormal basis of L 2 (T d , R), there cannot be anymore eigenfunctions of K. Thus, it suffices to show that (λ j , e) is an eigenpair for any e ∈ E j and j ∈ N d 0 . To prove this, we will establish that
T d m∈Z d 1{|m i | = j i ∀i ∈ [d]} cos 2πm • (y -x) e j (x) dx = e j (y),(3)
where e j is an arbitrary element of E j . Recall that
T d cos 2πm • (y -x) e j (x) dx = 0 if ∃i such that |m i | ̸ = j i . This is true because if ∃i such that |m i | ̸ = j i , then we can write cos 2πm • (y -x) = cos 2π ℓ̸ =i m ℓ (y ℓ - x ℓ ) cos(2πm i (y i -x i )) -sin 2π ℓ̸ =i m ℓ (y ℓ -x ℓ ) sin(2πm i (y i -x i )). Moreover, e j (x) = ψ j1 (x 1 ) . . . ψ j d-1 (x d-1 ) • ψ j d (x d
), where ψ j ℓ 's are either sine, cosine, or a constant function. Our claim follows upon noting that ψ ji (x i ) is orthogonal to both sin(2πm i (y i -x i )) and cos(2πm i (y i -x i )).
Thus, Equation (3) together with the fact that λ m = λ j for all m ∈ {k ∈ Z d : |k i | = j i ∀i ∈ [d]} implies that (λ j , e j ) is the eigenpair of K. As j ∈ N d 0 and e j ∈ E j are arbitrary, this completes our proof. Now, it remains to prove Equation ( 3). We will proceed by induction on d. For the base case, take d = 1. If j = 0, e j = 1 and our claim follows trivially. Suppose j ̸ = 0. Since cos(θ) = cos(-θ), we have m∈Z 1{|m| = j} cos 2πm(y -x) = 2 cos(2πj(y -x))
= 2 cos(2πjy) cos(2πjx) + 2 sin(2πjy) sin(2πjx).
If e j (x) = √ 2 cos(2πjx), then
T (2 cos(2πjy) cos(2πjx) + 2 sin(2πjy) sin(2πjx)) √ 2 cos(2πjx) dx = √ 2 cos(2πjy). If e j (x) = √ 2 sin(2πjx), a similar calculation shows that T (2 cos(2πjy) cos(2πjx) + 2 sin(2πjy) sin(2πjx)) √ 2 sin(2πjx) dx = √ 2 sin(2πjy).
This completes our proof of the base case.
Suppose (3) is true for d -1. We will now prove it for d. Note that
cos 2πm • (y -x) = cos 2π d i=1 m i (y i -x i ) = cos 2π d-1 i=1 m i (y i -x i ) cos (2πm d (y d -x d )) -sin 2π d-1 i=1 m i (y i -x i ) sin (2πm d (y d -x d )) .
First, observe that when summed over all m ∈ Z d such that |m i | = j i for all i ∈ [d], the sine term vanishes. That is,
m∈Z d 1 |m i | = j i ∀i ∈ [d] sin 2π d-1 i=1 m i (y i -x i ) sin (2πm d (y d -x d )) =   m∈Z d-1 1 |m i | = j i ∀i ∈ [d -1] sin 2π d-1 i=1 m i (y i -x i )   m d ∈Z 1{|m d | = j d } sin (2πm d (y d -x d )) = 0.
The final step follows here because the term in the second parenthesis above is always 0. There are two cases to consider. If j d = 0, the summand only has one term and our claim holds as sin(0) = 0. On the other hand, if j d ̸ = 0, then we are have sin(θ) + sin(-θ) = 0.
Therefore, we obtain
m∈Z d 1{|m i | = j i ∀i ∈ [d]} cos 2πm • (y -x) =   m∈Z d-1 1{|m i | = j i ∀i ∈ [d -1]} cos 2π d-1 i=1 m i (y i -x i )   m d ∈Z 1{|m d | = j d } cos (2πm d (y d -x d ))
A similar factorization can be done for e j to write
e j (x) = ψ j1 (x 1 ) . . . ψ j d (x d ),
where ψ ji 's are either sine, cosine, or a constant function.
However, ψ j d is some e j d defined on T. Thus, using the base case, we have
T m d ∈Z 1{|m d | = j d } cos (2πm d (y d -x d )) ψ j d (x d ) dx d = ψ j d (y d ).
Similarly, using the induction hypothesis, we have
T d-1   m∈Z d-1 1{|m i | = j i ∀i ∈ [d -1]} cos 2π d-1 i=1 m i (y i -x i )   d-1 i=1 ψ ji (x i ) d(x 1 , . . . , x d-1 ) = d-1 i=1 ψ ji (y i ).
Combining everything, we obtain
T d m∈Z d 1{|m i | = j i ∀i ∈ [d]} cos 2πm • (y -x) d i=1 ψ ji (x i ) dx = d i=1 ψ ji (y i ).
The final step requires using the factorization of cosine and writing integral over T d as product of integral over T d-1 and T. This completes our induction step, and thus the proof.
this section cite: ['b18']

Section: B.2. RBF Kernel on R.
Let K be the RBF kernel on R. That is, K(x, y) = exp -1 2ℓ 2 |x -y| 2 for all x, y ∈ R. For now, let ν is a Gaussian measure with mean 0 and variance σ 2 on R. Then, it is known (Williams & Rasmussen, 2006, Section 4.3.1) that K(x, y) = ∞ j=0 λ j φ j (x) φ j (y), where
λ j := 2a a + b + c b a + b + c j φ j (x) := exp(-(c -a)x 2 ) H j ( √ 2cx).
Here, a = (4σ 2 ) -1 , b = (2ℓ 2 ) -1 , c = √ a 2 + 2ab, and H j (•) is the Hermite polynomial of order j defined as
H j (x) = (-1) j exp(x 2 ) d j dx j exp(-x 2 ).
Note that this is the eigenpairs of K(y, x) over the entire R, whereas we need eigenpairs over some compact domain X ⊆ R.
The eigenpairs of K(y, x) are generally not available in closed form for arbitrary X . However, the variance of the Gaussian measure σ 2 can be tuned appropriately to localize the domain R to appropriate X of interest. For example, let X = [-1, 1]. Then,
1 -1 K(y, x)φ j (x) dν(x) = R K(y, x) φ j (x) dν(x) - |x|>1 K(y, x) φ j (x) dν(x). Since R K(y, x) φ j (x) dν(x) = λ j φ j (y), we have 1 -1 K(y, x)φ j (x) dν(x) -λ j φ j (y) ≤ |x|>1 |K(y, x)| |φ j (x)| dν(x) ≤ |x|>1 |K(y, x)| 2 dν(x) |x|>1 |φ j (x)| 2 dν(x) ≤ |x|>1 exp - |x -y| 2 ℓ 2 dν(x),
where the second term is upper bounded by 1 as φ 2 j integrates to 1 over the whole domain R. Note that exp -|x-y| 2 ℓ 2 ≤ 1 and ν([-1, 1] c ) ≤ 3.9 × 10 -12 when σ = 0.1. So, σ can be appropriately tuned such that (λ j , φ j ) j≥1 is a good approximation of the eigenpair of K for our domain X of interest. Next, we use these eigenvalues to study how the upper bound in Theorem 3.1 decays as n → ∞.
Let γ := b/(a + b + c). It is clear that γ ∈ (0, 1). Since c = √ a 2 + 2ab ≥ a, we also have 2a a+b+c ≤ 1. Thus, we obtain λ j ≤ γ j . Plugging this estimate in the upperbound of Theorem 3.1, we obtain
ε 2 n-1 i=0 λ i + ∥F∥ 2 op ∞ i=n λ i ≤ ε 2 n-1 i=0 γ j + ∥F∥ 2 op ∞ i=n γ j = 1 -γ n 1 -γ ε 2 + ∥F∥ 2 op γ n 1 -γ ≤ 1 (1 -γ) ε 2 + ∥F∥ 2 op γ n .
Therefore, the reducible error vanishes exponentially fast as n → ∞.
this section cite: ['b26']

Section: B.3. RBF Kernel on R d
Let K(y, x) = exp(-|x -y| 2 2 /(2ℓ 2 )), where x, y ∈ R d . Then, it is clear that
K(y, x) = d i=1 exp(-|x i -y i | 2 /(2ℓ 2 )) =: d i=1 K i (y i , x i ).
If (λ ij , φ ij ) j∈N are the eigenpairs of K i under the weighted measure standard Gaussian measure on R, then
d i=1 λ iji , d i=1 φ iji (j 1 , j 2 , . . . , j d ) ∈ N d 0
are the eigenpairs of K when ν is multivariate Gaussian with mean 0 and covariance σ 2 I. This follows immediately upon noting that
R d K(y, x) d i=1 φ iji (x i ) dν(x) = d i=1 R K i (y i , x i ) φ iji (x i ) dν(x i ) = d i=1 λ iji φ iji (x i ).
Finally, these are the only eigenpairs because the product functions d i=1 φ iji for all possible j 1 , . . . , j d ∈ N 0 form a complete orthonormal system of L 2 (R d ) under the base measure ν.
Pick m such that m > d, and suppose the n source terms in Theorem 3.1 are {φ ij : i ∈ [d] and 0 ≤ j ≤ m -1}. That is, we have n = m d source terms. So, the upperbound is
ε 2 m-1 j1=0 . . . m-1 j d =0 d i=1 λ iji + ∥F∥ 2 op (j1,...,j d )∈N d 0 max{j1,...,j d }≥m d i=1 λ iji
The first summation is
m-1 j1=0 . . . m-1 j d =0 d i=1 λ iji = d i=1 m-1 ji=0 λ iji ≤ d i=1 m-1 ji=0 γ ji ≤ 1 -γ m 1 -γ d ≤ 1 (1 -γ) d . On the other hand, (j1,...,j d )∈N d 0 max{j1,...,j d }≥m d i=1 λ iji ≤ (j1,...,j d )∈N d 0 max{j1,...,j d }≥m γ j1+...+j d ≤ ∞ r=m r d γ r ≤ ∞ m-1 r d γ r dr.
The second inequality follows because the number of tuple (j 1 , . . . , j d ) that sum to r is ≤ r d . It is easy to see that the integral converges faster than 1/n t for every t ≥ 1. To see this, pick t ≥ 1. Then, there exists c > 0 such that γ r ≤ c r -dt-1-d . Note that c may depend on γ, d, and t, but it does not depend on r. Thus, we obtain
∞ m-1 r d γ r dr ≤ c ∞ m-1 r -dt-1 dr = c (m -1) dt . Since m = n 1/d , this rate is c ′ /n t for some c ′ . That is, our overall upper bound is ε 2 1 (1 -γ) d + ∥F∥ 2 op c ′ n t .
for some c ′ for every t ≥ 1. Therefore, the reducible error vanishes at a rate faster than every polynomial function of 1/n.
this section cite: []

Section: B.4. Brownian Motion
Let us consider the case where X = [0, 1], the base measure ν is Lebegsue, and the stochastic process in Section 2.2 is Brownian motion. Recall that the Brownian motion is a Gaussian process with covariance kernel K(s, t) = min(s, t) s, t ∈ [0, 1].
It is well-known (Hsing & Eubank, 2015, Example 4.6.3) that the eigenpairs of K is given by
λ j := 1 j -1 2 2 π 2 and φ j (t) := √ 2 sin j - 1 2 πt ∀j ∈ N.
Plugging this in the upperbound of Theorem 3.1 yields the bound
ε 2 n j=1 1 j -1 2 2 π 2 + ∥F∥ 2 op ∞ j=n+1 1 j -1 2 2 π 2 = ε 2 π 2 2 1 π 2 + ∥F∥ 2 op ∞ j=n+1 1 j -1 2 2 π 2 ≤ ε 2 2 + ∥F∥ 2 op 1 π 2 ∞ n 1 (t -1/2) 2 dt = ε 2 2 + ∥F∥ 2 op 1 π 2 2 2n -1 .
Therefore, the reducible error vanishes at rate ∼ 1 n .
this section cite: ['b10']

Section: C. Numerical Approximation of Eigenfunctions
In Section 3.3, we provided analytic expressions for the eigenfunctions of certain covariance kernels. However, for some kernels of interest, closed-form expressions for the eigenfunctions are generally not available. In such cases, numerical approximation is necessary. Here, we will briefly mention some key concepts behind the numerical approximation of eigenfunctions of kernels. The material presented here is based on (Williams & Rasmussen, 2006, Section 4.3.2), so we refer the reader to that text for a more detailed discussion and relevant references.
Let dν(x) ∝ p(x) dx for some density function p. For example, if ν is Lebesgue measure on [-1, 1] × [-1, 1], then p(x) = 1/4. Then, the solution of Feldolm integral
X K(y, x) φ j (x) dν(x) = λ j φ(y)
is approximated using the equation 1 N
N i=1 K(y, x i ) φ j (x i ) = λ j φ j (y).
Here, x 1 , x 2 , . . . , x N are iid samples from p. Taking y = x 1 , . . . , x N , we obtain a matrix eigenvalue equation
Ku j = γ j u j ,
where K is a N × N matrix such that [K] = K(x i , x j ). The sequence (γ j , u j ) j≥1 is the eigenpair of K. Then, the estimator for eigenfunctions φ j 's and eigenvalues λ j 's are
φ j (x i ) ∼ √ N [u j ] i λ j ∼ γ j N .
The √ N normalization for eigenfunction is to ensure that the squared integral of φ j on the observed samples is 1. That is,
X φ j (x) φ j (x) dν(x) ≈ 1 N N i=1 φ j (x i ) φ j (x i ) = 1 N i=1 √ N [u j ] i • √ N [u j ] i = u ⊺ j u j = 1.
As for the eigenvalues, the proposed estimator is consistent. That is, γ j /N → λ j when N → ∞ (Baker & Taylor, 1979, Theorem 3.4).
The estimator for eigenfunction only allows evaluation on points x 1 , . . . , x N used to solve the matrix eigenvalue equation.
To evaluate the eigenfunction on arbitrary input, one can use a generalized Nyström-type estimator, defined as
φ j (y) ∼ √ N γ j N i=1 K(y, x i ) [u j ] i .
this section cite: ['b26']

Section: D. Proof of Lower Bound
Proof. Let {φ j } j∈N be the eigenfunctions of K. That is,
X K(y, x) φ i (x) dx = λ i φ i (y) ∀i ∈ N.
We now construct a hard distribution for the learner. Fix some p ∈ (0, 1) and let ξ 1 , ξ 2 , . . . denote the sequence of pairwise independent random variables such that
ξ j =     
-1/p with probability p 2 0 with probability 1 -p 1/p with probability p 2 .
this section cite: []

Section: Given such sequence, define a function
v such that v(•) = ∞ j=1 λ j ξ j φ j (•). Note that ∥v∥ L 2 = ∞ j=1 λ j ξ 2 j < ∞ as sup j∈N |ξ j | 2 ≤ 1/p and ∞ j=1 λ j < ∞. Thus, v is a random element in L 2 (X ).
Let µ denote the probability measure over L 2 (X ) induced by the random sequence {ξ j } j∈N . It is easy to see that E[v(x)] = 0 for each x ∈ X . Moreover, for every x, y ∈ X , we have
E[v(x) v(y)] = E     ∞ j=1 λ j ξ j φ j (x)     ∞ j=1 λ j ξ j φ j (y)     = E   ∞ j=1 λ j ξ 2 j φ j (x) φ j (y) + 2 i<j λ i λ j ξ i ξ j φ i (x) φ j (y)   = ∞ j=1 λ j E[ξ 2 j ] φ j (x) φ j (y) = ∞ j=1 λ j φ j (x) φ j (y) = K(y, x),
where the final equality holds due to Mercer's theorem and the convergence is uniform over x, y ∈ X . Therefore, we have shown that µ ∈ P(K). Let σ := {σ j } j≥1 be a sequence of iid random variables such that σ j ∼ Uniform({-1, 1}). Fix c > 0 and for each such σ ∈ {-1, 1} N , define
F σ := c ∞ j=1 σ j φ j ⊗ φ j .
For each m ∈ N, we will show that
E σ E v1:n∼µ n E v∼µ F n (v) -F σ (v) 2 L 2 ≥ c 2 2 m j=1 λ j .
Since this holds in expectation, using the probabilistic method, there must be a σ ⋆ such that
E v1:n∼µ n E v∼µ F n (v) -F σ ⋆ (v) 2 L 2 ≥ c 2 2 m j=1 λ j .
Noting that ∥F σ ⋆ ∥ op = c completes our proof. The rest of the proof will establish this inequality.
Since {φ j } j∈N is the orthonormal bases of L 2 (X ), Parseval's identity implies that
F n (v) -F σ (v) 2 L 2 = ∞ j=1 F n (v) -F σ (v), φ j 2 = ∞ j=1 F n (v), φ j -⟨F σ (v), φ j ⟩ 2 .
Recall that F ⋆ (φ j ) = cσ j φ j , where F ⋆ is the adjoint of F. Thus, for any v ∼ µ, we have
⟨F(v), φ j ⟩ = ⟨v, F ⋆ (φ j )⟩ = ⟨v, c σ j φ j ⟩ = c σ j λ j ξ j , which subsequently implies F n (v) -F σ (v) 2 L 2 = ∞ j=1 F n (v), φ j -c σ j λ j ξ j 2 .
Using this fact, we can write
E σ E v1:n∼µ n E v∼µ F n (v) -F σ (v) 2 L 2 = E σ   E v1:n∼µ n   E v∼µ   ∞ j=1 F n (v), φ j -c σ j λ j ξ j 2       = E v1:n∼µ n   E v∼µ   E σ   ∞ j=1 F n (v), φ j -c σ j λ j ξ j 2       .
In the final step, we changed the order of integration. Note that drawing n samples of v 1 , . . . , v n and drawing σ can be done in any order, as they are interchangeable. Finally, the draw of v ∼ µ occurs during the test phase, independent of the previously drawn samples v 1:n and σ.
Next, let E n,m denote the event such that ⟨v i , φ j ⟩ = 0 ∀1 ≤ i ≤ n and 1 ≤ j ≤ m.
Then, we will lowerbound
E v∼µ   E σ   ∞ j=1 F n (v), φ j -c σ j λ j ξ j 2     conditioned on the event E n,m . First, note that E v∼µ   E σ   ∞ j=1 F n (v), φ j -c σ j λ j ξ j 2     ≥ E v∼µ   m j=1 E σ F n (v), φ j -c σ j λ j ξ j 2   ≥ E v∼µ   m j=1 E σ F n (v), φ j -c σ j λ j ξ j 2   ,
where the final step uses Jensen's inequality. Next, we use the fact that when the event E n,m occurs, the learner has no information about σ 1 , . . . , σ m . This is because the input data shows no variation along the directions spanned by φ 1 , . . . , φ m . Given that O is the perfect oracle for F σ , any information provided by the oracle O must be independent of how F σ operates on the subspace spanned by φ 1 , . . . , φ m . Specifically, for every 1 ≤ i ≤ n and 1 ≤ j ≤ m, the output of the oracle O(v i ) must be independent of σ j . If this condition holds, then the estimator F n must also be independent of σ 1 , . . . , σ m . Thus, conditioned on the event E n,m , for any 1 ≤ j ≤ m, we have
E σ F n (v), φ j -cσ j λ j ξ j = E E σj F n (v), φ j -c σ j λ j ξ j σ\{σ j } = E 1 2 F n (v), φ j -c λ j ξ j + F n (v), φ j + c λ j ξ j ≥ 1 2 2 c λ j ξ j = |c λ j ξ j |.
The first equality uses the fact that conditioned on σ\{σ j }, the function F n (v) is independent of σ j . Thus, conditioned on the event E n,m , we have shown that
E v∼µ   E σ   ∞ j=1 F(v), φ j -c σ j λ j ξ j 2     ≥ E v∼µ   m j=1 c 2 λ j ξ 2 j   = c 2 m j=1 λ j E[ξ 2 j ] = c 2 m j=1 λ j .
Therefore, our overall lowerbound is
E v1:n∼µ n   E v∼µ   E σ   ∞ j=1 F(v), φ j -c σ j λ j ξ j 2       ≥ c 2   m j=1 λ j   P [E n,m ] = c 2 (1 -p) n•m m j=1 λ j .
The final step uses the fact that P[E n,m ] = (1 -p) n•m . It now remains to pick p to obtain the claimed lowerbound. Let us pick p = 1 2mn . Then, we have (1 -p) mn ≥ 1/2 as long as n ≥ 1, yielding the lowerbound of
c 2 2 m j=1 λ j .
Since m ∈ N is arbitrary, our lowerbound holds for every fixed m. Noting that ∥F σ ∥ op = c for every σ completes our proof.
this section cite: []

Section: E. Experiments
This section presents additional experimental results using the same setup as described in Section 5. The results show that the Fourier Neural Operator (FNO) performs poorly with actively collected data. This is likely because the training data are not i.i.d. samples from the test distribution, requiring FNO to generalize out of distribution when trained on actively collected data.
this section cite: []

Section: E.1. Poisson Equation

this section cite: []

Section: E.2. Heat Equation

this section cite: []

Section: References
Ref_id:b0 Title: Queries and concept learning Year: (1988)
Ref_id:b1 Title: Queries revisited Year: (2001)
Ref_id:b2 Title: Training connectionist networks with queries and selective sampling Year: (1989)
Ref_id:b3 Title: Neural operator learning Year: (2024)
Ref_id:b4 Title: The numerical treatment of integral equations Year: (1979)
Ref_id:b5 Title: Model reduction and neural networks for parametric pdes Year: (2021)
Ref_id:b6 Title: A mathematical guide to operator learning Year: (2023)
Ref_id:b7 Title: Elliptic pde learning is provably data-efficient Year: (2023)
Ref_id:b8 Title: Convergence rates for learning linear operators from noisy data Year: (2023)
Ref_id:b9 Title: A statistical theory of active learning Year: (2013)
Ref_id:b10 Title: Theoretical foundations of functional data analysis Year: (2015)
Ref_id:b11 Title: Neural operator: Learning maps between function spaces with applications to pdes Year: (2023)
Ref_id:b12 Title: Data complexity estimates for operator learning Year: (2024)
Ref_id:b13 Title: Operator learning: Algorithms and analysis Year: (2024)
Ref_id:b14 Title: Operator learning with pca-net: upper and lower complexity bounds Year: (2023)
Ref_id:b15 Title: Error estimates for deeponets: A deep learning framework in infinite dimensions Year: (2022)
Ref_id:b16 Title: A sequential algorithmfor training text classifiers Year: (1994)
Ref_id:b17 Title: Multi-resolution active learning of fourier neural operators Year: (2024)
Ref_id:b18 Title: Fourier neural operator for parametric partial differential equations Year: (2021)
Ref_id:b19 Title: Deep nonparametric estimation of operators between infinite dimensional spaces Year: (2024)
Ref_id:b20 Title: An introduction to computational stochastic PDEs Year: (2014)
Ref_id:b21 Title: Learning nonlinear operators via deeponet based on the universal approximation theorem of operators Year: (2021)
Ref_id:b22 Title: Active learning for neural pde solvers Year: (2024)
Ref_id:b23 Title: The random feature model for input-output maps between banach spaces Year: (2021)
Ref_id:b24 Title: Active learning literature survey Year: (2009)
Ref_id:b25 Title: Error bounds for learning fourier linear operators Year: (2024)
Ref_id:b26 Title: Gaussian processes for machine learning Year: (2006)
