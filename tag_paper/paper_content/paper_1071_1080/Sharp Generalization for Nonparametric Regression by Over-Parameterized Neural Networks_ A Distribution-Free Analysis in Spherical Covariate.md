Title: Sharp Generalization for Nonparametric Regression by Over-Parameterized Neural Networks: A Distribution-Free Analysis in Spherical Covariate
Abstract: Sharp generalization bound for neural networks trained by gradient descent (GD) is of central interest in statistical learning theory and deep learning. In this paper, we consider nonpara-

Section: 
metric regression by an over-parameterized twolayer NN trained by GD. We show that, if the neural network is trained by GD with early stopping, then the trained network renders a sharp rate of the nonparametric regression risk of O(ε 2 n ), which is the same rate as that for the classical kernel regression trained by GD with early stopping, where ε n is the critical population rate of the Neural Tangent Kernel (NTK) associated with the network and n is the size of the training data. It is remarked that our result does not require distributional assumptions on the covariate as long as the covariate lies on the unit sphere, in a strong contrast with many existing results which rely on specific distributions such as the spherical uniform data distribution or distributions satisfying certain restrictive conditions. As a special case of our general result, when the eigenvalues of the associated NTK decay at a rate of λ j ≍ j -d d-foot_0 for j ≥ 1 which happens under certain distributional assumption such as the training features follow the spherical uniform distribution, we immediately obtain the minimax optimal rate of O(n -d 2d-1 ), which is the major results of several existing works in this direction. The neural network width in our general result is lower bounded by a function of only d and ε n , and such width does not depend on the minimum eigenvalue of the empirical NTK matrix whose lower bound usually requires additional assumptions on the training data. Our Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). results are built upon two significant technical results which are of independent interest. First, uniform convergence to the NTK is established during the training process by GD, so that we can have a nice decomposition of the neural network function at any step of the GD into a function in the Reproducing Kernel Hilbert Space associated with the NTK and an error function with a small L ∞ -norm. Second, local Rademacher complexity is employed to tightly bound the Rademacher complexity of the function class comprising all the possible neural network functions obtained by GD. Our result formally fills the gap between training a classical kernel regression model and training an over-parameterized but finite-width neural network by GD for nonparametric regression without distributional assumptions about the spherical covariate.
this section cite: []

Section: Introduction
With the stunning success of deep learning in various areas of machine learning (LeCun et al., 2015), generalization analysis for neural networks is of central interest for statistical learning learning and deep learning. Considerable efforts have been made to analyze the optimization of deep neural networks showing that gradient descent (GD) and stochastic gradient descent (SGD) provably achieve vanishing training loss (Du et al., 2019b;Allen-Zhu et al., 2019b;Du et al., 2019a;Arora et al., 2019;Zou & Gu, 2019;Su & Yang, 2019). There are also extensive efforts devoted to generalization analysis of deep neural networks (DNNs) with algorithmic guarantees, that is, the generalization bounds for neural networks trained by gradient descent or its variants. It has been shown that with sufficient over-parameterization, that is, with enough number of neurons in hidden layers, the training dynamics of deep neural networks (DNNs) can be approximated by that of a kernel method with the kernel induced by the neural network architecture, termed the Neural Tangent Kernel (NTK), while other studies such as (Yang & Hu, 2021) show that infinitewidth neural networks can still learn features. The key idea of NTK based generalization analysis is that, for highly over-parameterized networks, the network weights almost remain around their random initialization. As a result, one can use the first-order Taylor expansion around initialization to approximate the neural network functions and analyze their generalization capability (Cao & Gu, 2019;Arora et al., 2019;Ghorbani et al., 2021).
Many existing works in generalization analysis of neural networks focus on clean data, but it is a central problem in statistical learning that how neural networks can obtain sharp convergence rates for the risk of nonparametric regression where the observed data are corrupted by noise. Considerable research has been conducted in this direction which shows that various types of DNNs achieve optimal convergence rates for smooth (Yarotsky, 2017;Bauer & Kohler, 2019;Schmidt-Hieber, 2020;Jiao et al., 2023;Zhang & Wang, 2023) or non-smooth (Imaizumi & Fukumizu, 2019) target functions for nonparametric regression. However, most of these works do not have algorithmic guarantees, that is, the DNNs in these works are constructed specially to achieve optimal rates with no guarantees that an optimization algorithm, such as GD or its variants, can obtain such constructed DNNs. To this end, efforts have been made in the literature to study the minimax optimal risk rates for nonparametric regression with over-parameterized neural networks trained by GD with either early stopping (Li et al., 2024) or ℓ 2 -regularization (Hu et al., 2021;Suh et al., 2022). However, most existing works either require spherical uniform data distribution on the unit sphere (Hu et al., 2021;Suh et al., 2022) or certain restrictive conditions on the data distribution.
It remains an interesting and important question for the statistical learning and theoretical deep learning literature that if an over-parameterized neural network trained by GD can achieve sharp risk rates for nonparametric regression with milder assumptions or restrictions on the distribution of the covariate, so that theoretical guarantees can be obtained for data in more practical scenarios. In this paper, we give a confirmative answer to this question. We present sharp risk rate for nonparametric regression with an over-parameterized two-layer NN trained by GD with early stopping, which is distribution-free in spherical covariate. Throughout this paper, distribution-free in spherical covariate means that there are no distributional assumptions about the covariate as long as the covariate lies on the unit sphere. Furthermore, our results give confirmative answers to certain open questions or address particular concerns in the literature of training over-parameterized neural networks by GD with early stopping for nonparametric regression with minimax optimal rates, such as the characterization of the stopping time in the early-stopping mechanism, the lower bound for the network width, and the constant learning rate used in GD. Benefiting from our analysis which is distribution-free in spherical covariate, our answers to these open questions or concerns do not require distributional assumptions about spherical covariate. Section 3 summarizes our main results with their significance and comparison to existing works.
We organize this paper as follows. We first introduce the necessary notations in the remainder of this section. We then introduce in Section 2 the problem setup for nonparametric regression. Our main results are summarized in Section 3 and detailed in Section 5. The training algorithm for the over-parameterized two-layer neural network is introduced in Section 4. The roadmap of proofs, the summary of the technical approaches and the novel results in the proofs, and the novel proof strategy of this work are presented in Section 6. The detailed proofs are deferred to Section A-Section C of the appendix, and Section D of the appendix presents the simulation results.
Notations. We use bold letters for matrices and vectors, and regular lower letter for scalars throughout this paper. The bold letter with a single superscript indicates the corresponding column of a matrix, e.g., A (i) is the i-th column of matrix A, and the bold letter with subscripts indicates the corresponding rows or elements of a matrix or a vector. We put an arrow on top of a letter with subscript if it denotes a vector, e.g.,
this section cite: ['b21', 'b2', 'b30', 'b34', 'b9', 'b2', 'b13', 'b37', 'b5', 'b28', 'b17', 'b39', 'b16', 'b23', 'b15', 'b31', 'b15', 'b31']

Section: ⇀
x i denotes the i-th training feature. ∥•∥ F and ∥•∥ p denote the Frobenius norm and the vector ℓ p -norm or the matrix p-norm. [m : n] denotes all the natural numbers between m and n inclusively, and [1 : n] is also written as [n]. Var [•] denotes the variance of a random variable. I n is a n × n identity matrix. 1I {E} is an indicator function which takes the value of 1 if event E happens, or 0 otherwise. The complement of a set A is denoted by A c , and |A| is the cardinality of the set A. vec (•) denotes the vectorization of a matrix or a set of vectors, and tr (•) is the trace of a matrix. We denote the unit sphere in d-dimensional Euclidean space by S d-1 := {x : x ∈ R d , ∥x∥ 2 = 1}. Let L 2 (S d-1 , µ) denote the space of square-integrable functions on S d-1 with probability measure µ, and the inner product ⟨•, •⟩ µ and ∥•∥ 2 µ are defined as ⟨f, Throughout this paper we let the input space be X = S d-1 , and Unif (X ) denotes the uniform distribution on X . The constants defined throughout this paper may change from line to line. For a Reproducing Kernel Hilbert Space H, H(µ 0 ) denotes the ball centered at the origin with radius µ 0 in H. We use E P [•] to denote the expectation with respect to the distribution P .
g⟩ L 2 := S d-1 f (x)g(x)dµ(x) and ∥f ∥ 2 L 2 := S d-1 f 2 (x)dµ(x) < ∞. B (x; r) is the Euclidean closed ball centered at x with radius r. Given a function g : S d-1 → R, its L ∞ -norm is denoted by ∥g∥ ∞ := sup x∈S d-1 |g(x)|. L ∞ is
this section cite: []

Section: Problem Setup
We introduce the problem setups for nonparametric regression in this section.
this section cite: []

Section: Two-Layer Neural Network
We are given the training data (
⇀ x i , y i ) n i=1
where each data point is a tuple of feature vector ⇀ x i ∈ X and its response y i ∈ R. Throughout this paper we assume that no two training features coincide, that is,
⇀ x i ̸ = ⇀
x j for all i, j ∈ [n] and i ̸ = j. We denote the training feature vec-
tors by S = ⇀ x i n i=1
, and denote by P n the empirical distribution over S. All the responses are stacked as a vector y = [y 1 , . . . , y n ] ⊤ ∈ R n . The response y i is given by y
i = f * ( ⇀ x i ) + w i for i ∈ [n]
, where {w i } n i=1 are i.i.d. sub-Gaussian random noise with mean 0 and variance proxy σ 2 0 , that is, E [exp(λw i )] ≤ exp(λ 2 σ 2 0 /2) for any λ ∈ R. f * is the target function to be detailed later. We define y := [y 1 , . . . , y n ], w := [w 1 , . . . , w n ]
⊤ , and use
f * (S) := f * ( ⇀ x 1 ), . . . , f * ( ⇀ x n ) ⊤
to denote the clean target labels. The feature vectors in S are drawn i.i.d. according to an underlying unknown continuous data distribution P with µ being the probability measure for P . We consider a two-layer NN (NN) in this paper whose mapping function is
f (W, x) = 1 √ m m r=1 a r σ ⇀ w r ⊤ x ,(1)
where
x ∈ X is the input, σ(•) = max {•, 0} is the ReLU activation function, W = ⇀ w r m r=1 with ⇀ w r ∈ R d for r ∈ [m]
denotes the weighting vectors in the first layer and m is the number of neurons. a = [a 1 , . . . , a m ] ∈ R m denotes the weights of the second layer. Throughout this paper we also write W as W S so as to indicate that the weighting vectors in W are trained on the training features S.
this section cite: []

Section: Kernel and Kernel Regression for Nonparametric Regression
We define the kernel function
K(u, v) := ⟨u, v⟩ 2π (π -arccos ⟨u, v⟩) , ∀ u, v ∈ X ,(2)
which is in fact the NTK associated with the two-layer NN (1) when only the first layer is trained, and K is a positivedefinite (PD) kernel. Let the gram matrix of K over the training data S be K ∈ R n×n , K ij = K(
⇀ x i , ⇀
x j ) for i, j ∈ [n], and K n := K/n is the empirical NTK matrix. Let the eigendecomposition of K n be K n = UΣU ⊤ where U is a n × n orthogonal matrix, and Σ is a diagonal matrix with its diagonal elements λ i n i=1 being eigenvalues of K n and sorted in a non-increasing order. It is proved in existing works, such as (Du et al., 2019b), that K n is non-singular, and it can be verified that λ 1 ∈ (0, 1/2). Let H K be the Reproducing Kernel Hilbert Space (RKHS) associated with K. Because K is continuous on the compact set X × X , the integral operator
T K : L 2 (X , µ) → L 2 (X , µ), (T K f ) (x) := X K(x, x ′ )f (x ′ )dµ(x ′
) is a positive, self-adjoint, and compact operator on L 2 (X , µ). By the spectral theorem, there is a countable orthonormal basis {e j } j≥1 ⊆ L 2 (X , µ) and {λ j } j≥1 with 1 2 ≥ λ 1 ≥ λ 2 ≥ . . . > 0 such that e j is the eigenfunction of T K with λ j being the corresponding eigenvalue. That is, T K e j = λ j e j , j ≥ 1. Let {µ ℓ } ℓ≥1 be the distinct eigenvalues associated with T K , and let m ℓ be the be the sum of multiplicity of the eigenvalue {µ
ℓ ′ } ℓ ℓ ′ =1 . That is, m ℓ ′ -m ℓ ′ -1 is the mul- tiplicity of µ ℓ ′ . It is well known that v j = λ j e j j ≥1 is an orthonormal basis of H K . For a positive con- stant µ 0 , we define H K (µ 0 ) := {f ∈ H K : ∥f ∥ H ≤ µ 0 } as the closed ball in H K centered at 0 with radius µ 0 . We note that H K (µ 0 ) is also specified by H K (µ 0 ) = f ∈ L 2 (X , µ) : f = ∞ j=1 β j e j , ∞ j=1 β 2 j /λ j ≤ µ 2 0 .
The Task of Nonparametric Regression. With f * ∈ H K (µ 0 ), the task of the analysis for nonparametric regression is to find an estimator f from the training data
( ⇀ x i , y i ) n i=1
so that the risk E P f -f * 2 can converge to 0 with a fast rate. In this work, we aim to establish a sharp rate of the risk where the over-parameterized neural network (1) trained by GD with early stopping serves as the estimator f .
this section cite: []

Section: Sharp rate of the risk of nonparametric regression using classical kernel regression.
The statistical learning literature has established rich results in the sharp convergence rates for the risk of nonparametric kernel regression (Stone, 1985;Yang & Barron, 1999;Raskutti et al., 2014;Yuan & Zhou, 2016), with one representative result in (Raskutti et al., 2014) about kernel regression trained by GD with early stopping. Let ε n be the critical population rate of the PD kernel K, which is also referred to as the critical radius (Wainwright, 2019) of K. (Raskutti et al., 2014, Theorem 2) shows the following sharp bound for the nonparametric regression risk of a kernel regression model trained by GD with early stopping when f * ∈ H K (µ 0 ). That is, with probability at least 1 -Θ exp(-Θ(nε 2 n ) ,
E P f T -f * 2 ≲ ε 2 n ,(3)
where T is the stopping time whose formal definition is deferred to Section 5.1, and f T is the kernel regressor at the T -th step of GD for the optimization problem of kernel regression. The risk bound (3) is rather sharp, since it is minimax optimal in several popular learning setups, such as the setup where the eigenvalues {λ i } i≥1 exhibit a certain polynomial decay. Such risk bound (3) also holds for a general PD kernel rather than the NTK (2), and the risk bound (3) is also minimax optimal when the PD kernel is low rank. It is also remarked that the risk bound (3) is distribution-free in the bounded covariate, that is, there are no distributional assumptions about the covariate when it is in a bounded input space. Interested readers are referred to (Raskutti et al., 2014) for more details.
The main result of this paper is that the over-parameterized two-layer NN (1) trained by GD with early stopping achieves the same order of risk rate as that in (3) with arbitrary continuous distribution of the spherical covariate, which are summarized in the next section.
this section cite: ['b29', 'b35', 'b27', 'b38', 'b27', 'b32', 'b27', 'b27']

Section: Summary of Main Results.
Our main results are summarized in this section. Throughout this paper, we consider fixed dimension d ≥ 4.
First, Theorem 5.1 in Section 5.2 shows that the neural network (1) trained by GD with early stopping using Algorithm 1 enjoys a sharp rate of the nonparametric regression risk, O ε 2 n , which is the same as that for the classical kernel regression in (3). Such rate of nonparametric regression risk in Theorem 5.1 is distribution-free in spherical covariate, and it immediately leads to minimax optimal rates for certain special cases. For example, when the eigenvalues of the integral operator associated with K has a particular polynomial eigenvalue decay rate (EDR), that is, λ j ≍ j
-d d-1 for j ≥ 1, then in this case ε 2 n ≍ n -d 2d-1
according to (Raskutti et al., 2014, Corollary 3), and Theorem 5.1 renders the rate of the nonparametric regression risk of O(n -d 2d-1 ) which is minimax optimal for this special case (Stone, 1985;Yang & Barron, 1999;Yuan & Zhou, 2016). We refer to such EDR the polynomial EDR in the sequel. It is shown in (Bietti & Mairal, 2019;Bietti & Bach, 2021;Li et al., 2024) that the polynomial EDR holds for our NTK in (2) if P = Unif (X ), or P satisfies the distributional assumption for (Li et al., 2024, Proposition 13) in Table 1.
We remark that such a minimax optimal rate O(n -d 2d-1 ) is derived from Theorem 5.1 under the special case of polynomial EDR, and this minimax optimal rate is also the major result of a series of existing works in nonparametric regression by training over-parameterized neural networks (Hu et al., 2021;Suh et al., 2022;Li et al., 2024) when the target function f * belongs to H K , the RKHS associated with the NTK K of the network in each particular existing work. We note that K is the NTK of the network considered in a particular existing work which may not be the same as our NTK in (2). We also note that one needs to set s = 1 in (Li et al., 2024, Proposition 13) so that f * ∈ H K , and in this case the risk rate for nonparametric regression in (Li et al., 2024, Proposition 13) is O(n -d 2d-1 ). To the best of our knowledge, Theorem 5.1 presents the first sharp risk rate for nonparametric regression which is distribution-free in spherical covariate, which is closer to practical scenarios. In contrast, the minimax rates in (Hu et al., 2021;Suh et al., 2022) require spherical uniform data distribution on X . The recent work (Ko & Huo, 2024) also requires certain distributional assumptions for the results about regression convergence rates which does not have algorithmic guarantees. Although the minimax rate in another recent work (Li et al., 2024) does not need the spherical uniform distribution, it still requires a restrictive condition on the data distributions detailed in Table 1, and such condition is met by sub-Gaussian distributions. It is under this condition that (Li et al., 2024) derives the polynomial EDR. Table 1 compares our work to existing works for nonparametric regression with a common setup, that is, f * ∈ H K and the responses {y i } n i=1 are corrupted by i.i.d. Gaussian noise. We further note that although the result in (Kuzborskij & Szepesvári, 2021, Theorem 2) does not require distributional assumptions about the covariate, its risk rate under this common setup is not minimax optimal due to the term σ 2 0 in the risk bound. Furthermore, the other term O(n
-2 2+d
) in its risk bound suffers from the curse of dimension with a slow rate to 0 for high-dimensional data. We also note that (Kuzborskij & Szepesvári, 2021, Theorem 1) shows the minimax optimal rate of O(n -2 2+d ), however, this rate is derived for the noiseless case where the responses are not corrupted by noise.
Second, our results provide confirmative answers to several outstanding open questions or address particular concerns in the existing literature about training over-parameterized neural networks for nonparametric regression by GD with early stopping and sharp risk rates, which are detailed below.
this section cite: ['b29', 'b35', 'b38', 'b7', 'b6', 'b23', 'b15', 'b31', 'b23', 'b15', 'b31', 'b18', 'b23', 'b23', 'b20', 'b20']

Section: Stopping time in the early-stopping mechanism.
An open question raised in (Kuzborskij & Szepesvári, 2021;Hu et al., 2021) is how to characterize the stopping time in the early-stopping mechanism when training the overparameterized network by GD. Let T be the stopping time, (Li et al., 2024, Proposition 13) shows that the stopping time should satisfy T ≍ n d 2d-1 under the distributional as-
λ j ≍ j -d d-1 minimax optimal, O(n -d 2d-1 )
Our Result (Theorem 5.1) No distributional assumption about P as long as X = S d-1 No requirement for EDR O ε 2 n , which leads to the minimax optimal rate O(n (Hu et al., 2021;Suh et al., 2022) and (Li et al., 2024) as special cases.
-d 2d-1 ) claimed in
sumption in Table 1. In contrast, Theorem 5.1 provides a characterization of T showing that T ≍ ε -2 n , which is distribution-free in spherical covariate. Theorem 5.1 further suggests that for each neural network function f t obtained at the t-th step of GD with t ≍ ε -2 n , the sharp risk rate of O ε 2 n is obtained. Lower bound for the network width m. Our main result, Theorem 5.1, requires that the network width m, which is the number of neurons in the first layer of the network, satisfies m ≳ d 8 3 /ε 80 3 n . Such lower bound for m solely depends on d and ε n . Under the polynomial EDR, Corollary 5.2, which is a direct consequence of Theorem 5.1, shows that m should satisfy m ≳ n 80α 3(2α+1) d 8 3 with α = d/(2(d -1)) (see ( 11)) so that GD with early stopping leads to the minimax rate of O(n -d 2d-1 ). We remark that this is the first time that the lower bound for the network width m is specified only in terms of n and d under the polynomial EDR with a minimax optimal risk rate for nonparametric regression, which can be easily estimated from the training data. In contrast, under the same polynomial EDR, all the existing works (Hu et al., 2021;Suh et al., 2022;Li et al., 2024) require m ≳ poly(n, 1/ λ n ). The problem here is that one needs additional assumptions on the training data (Bartlett et al., 2021;Nguyen et al., 2021) to find the lower bound for λ n , which is the minimal eigenvalue of the empirical NTK matrix K n , to further estimate the lower bound for m using the training data.
Corollary 5.2 also gives a competitive and smaller lower bound for the network width m than some existing works which give explicit orders of the lower bound for m. For example, under the assumption of uniform spherical distribution, (Suh et al., 2022, Theorem 3.11) requires that m/ log 3 m ≳ L 20 n 24 where L is the number of layers of the DNN used in that work, and m/ log 3 m ≳ 2 20 n 24 even with L = 2 for the two-layer network (1) used in our work. Furthermore, the proof of (Li et al., 2024, Propo-sition 13) suggests that m ≳ n 24 (log m) 12 . Both lower bounds for m in (Suh et al., 2022, Theorem 3.11) and (Li et al., 2024, Proposition 13) are much larger than our lower bound for m, n 80α 3(2α+1) d 8 3 , when n → ∞ and d is fixed, which is the setup considered in (Li et al., 2024). It is worthwhile to mention that (Suh et al., 2022;Li et al., 2024) use DNNs with multiple layers for nonparametric regression. As shown in Table 1, through our careful analysis, a shallow two-layer NN (1) exhibits the same minimax risk rate as its deeper counterpart under the same assumptions with much smaller network width. This observation further support the claim in (Bietti & Bach, 2021) that a shallow over-parameterized neural networks with ReLU activations exhibit the same approximation properties as its deeper counterpart, in our nonparametric regression setup.
Training the network with learning rate η = Θ(1). It is also worthwhile to mention that our main result, Theorem 5.1, suggests that a constant learning rate η = Θ(1) can be used for GD when training the two-layer NN (1), which could lead to better empirical optimization performance in practice. Some existing works in fact require an infinitesimal η. For example, (Li et al., 2024, Proposition 13) is obtained by gradient flow where η → 0 instead of the practical GD. Furthermore, (Hu et al., 2021, Theorem 5.2) requires the learning rates for both the squared loss and the ℓ 2 -regularization term to have the order of o(n -3d-1 2d-1 ) → 0 as n → ∞. We note that (Nitanda & Suzuki, 2021) also employs constant learning rate in SGD to train neural networks.
More discussion about this work and the relevant literature. We herein provide more discussion about the results of this work and comparison to the existing relevant works with sharp rates for nonparametric regression. While this paper establishes sharp rate which is distribution-free in spherical covariate, such rate still depends on bounded input space (X = S d-1 ) and the condition that the target function f * ∈ H K (µ 0 ). Some other existing works consider target function f * not belonging to the RKHS ball centered at the origin with constant or low radius, such as (Haas et al., 2023;Bordelon et al., 2024). However, the target functions in (Haas et al., 2023;Bordelon et al., 2024) escape the finite norm or low-norm regime of RKHS at the cost of either restriction condition on the density function of the covariate distribution or the training process. In particular, (Haas et al., 2023, Theorem G.5) requires the condition for bounded density function (in its condition (D3) ) of the distribution P , which is not required by our result. Moreover, the training process of the model in (Bordelon et al., 2024) requires information about the target function (in its Eq. ( 4)) and certain distribution P which admits certain polynomial EDR, that is, λ j ≍ j -α with α > 1, which happens under certain restrictive conditions on P .
We also note that in this work, only the first layer of an over-parameterized two-layer neural network is trained, while the weights of the second layer are randomly initialized and then fixed in the training process. In existing works such as (Hu et al., 2021;Suh et al., 2022;Allen-Zhu et al., 2019a), all the layers of a deep neural networks with more than two layers are trained by GD or its variants. However, this work shows that only training the first layer still leads to sharp rate for nonparametric regression, which supports the claim in (Bietti & Bach, 2021) that a shallow over-parameterized neural networks with ReLU activations exhibit the same approximation properties as its deeper counterpart.
this section cite: ['b20', 'b15', 'b15', 'b31', 'b23', 'b15', 'b31', 'b23', 'b4', 'b25', 'b23', 'b31', 'b23', 'b6', 'b26', 'b14', 'b8', 'b14', 'b8', 'b8', 'b15', 'b31', 'b6']

Section: Training by Gradient Descent and Preconditioned Gradient Descent
In the training process of our network (1), only W is optimized with a randomly initialized to ±1 and then fixed.
The following quadratic loss function is minimized during the training process:
L(W) := 1 2n n i=1 f (W, ⇀ x i ) -y i 2 .(4)
In the (t + 1)-th step of GD with t ≥ 0, the weights of the neural network, W S , are updated by one-step of GD through
vec (W S (t + 1)) -vec (W S (t)) = - η n Z S (t)( y(t) -y),(5)
where
y i = y i , y(t) ∈ R n with [ y(t)] i = f (W(t), ⇀ x i ).
The notations with the subscripts S indicate the dependence on the training features S. We also denote f (W(t), •) as f t (•) as the neural network function with weighting vectors W(t) obtained after the t-th step of GD.
We define Z S (t) ∈ R md×n which is computed by
[Z S (t)] [(r-1)d+1:rd]i = 1 √ m 1I ⇀ wr(t) ⊤ ⇀ x i≥0 ⇀ x i a r (6)
for all i ∈ [n] and r ∈ [m]. where [Z S (t)] [(r-1)d+1:rd]i ∈ Algorithm 1 Training the Two-Layer NN by GD 1: W(T ) ← Training-by-GD(T, W(0)) 2: input: T, W(0) 3: for t = 1, . . . , T do 4:
Perform the t-th step of GD by (5) 5: end for 6: return W(T ) R d is a vector with elements in the i-th column of Z S (t) with indices in [(r -1)d + 1 : rd]. We employ the following particular symmetric random initialization so that y(0) = 0, which has been used in existing works such as (Chizat et al., 2019;Zhang et al., 2020). In our two-layer NN, m is even,
⇀ w 2r ′ (0) m/2 r ′ =1
and {a 2r ′ } m/2 r ′ =1 are initialized randomly and independently according to
⇀ w 2r ′ (0) ∼ N (0, κ 2 I d ), a 2r ′ ∼ unif ({-1, 1}) , ∀r ′ ∈ [m/2],
where N (µ, Σ) denotes a Gaussian distribution with mean µ and covariance Σ, unif ({-1, 1}) denotes a uniform distribution over {1, -1}, 0 < κ ≤ 1 controls the magnitude of initialization, and κ ≍ 1. We set
⇀ w 2r ′ -1 (0) = ⇀ w 2r ′ (0)
and a 2r ′ -1 = -a 2r for all r ′ ∈ [m/2]. It then can be verified that y(0) = 0. Once randomly initialized, a is fixed during the training. We use W(0) to denote the set of all the random weighting vectors at initialization, that
is, W(0) = ⇀ w r (0) m r=1
. We run Algorithm 1 to train the two-layer NN by GD for T steps. Early stopping is enforced in Algorithm 1 through a bounded T via T ≤ T .
this section cite: ['b10', 'b40']

Section: Main Results
We present the definition of kernel complexity in this section, and then introduce the main results for nonparametric regression of this paper.
this section cite: []

Section: Kernel Complexity
The local kernel complexity has been studied by (Bartlett et al., 2005;Koltchinskii, 2006;Mendelson, 2002). For the PD kernel K, we define the empirical kernel complexity R K and the population kernel complexity R K as
R K (ε) := 1 n n i=1 min λ i , ε 2 , R K (ε) := 1 n ∞ i=1 min {λ i , ε 2 }.(7)
It can be verified that both σ 0 R K (ε) and σ 0 R K (ε) are subroot functions (Bartlett et al., 2005) in terms of ε 2 . The formal definition of sub-root functions is deferred to Definition A.2 in the appendix. For a given noise with variance proxy σ 2 0 , the critical empirical radius ε n > 0 is the smallest positive solution to the inequality R K (ε) ≤ ε 2 /σ 0 , where ε 2 n is the also the fixed point of
σ 0 R K (ε) as a func- tion of ε 2 : σ 0 R K ( ε n ) = ε 2 n .
Similarly, the critical population rate ε n is defined to be the smallest positive solution to the inequality R K (ε) ≤ ε 2 /σ 0 , where
ε 2 n is the fixed point of σ 0 R K (ε) as a function of ε 2 : σ 0 R K (ε n ) = ε 2 n .
In this paper we consider the case that nε 2 n → ∞ as n → ∞, which is also used in standard analysis of nonparametric regression with minimax rates by kernel regression (Raskutti et al., 2014).
Let η t := ηt for all t ≥ 0, we then define the stopping time T as
T := min t : R K ( 1/η t ) > (σ 0 η t ) -1 -1. (8)
The stopping time in fact limit the number of steps T in for Algorithm 1 as to be shown in Section 5.2, which in turn enforces the early stopping mechanism.
this section cite: ['b3', 'b19', 'b24', 'b3', 'b27']

Section: Results
Theorem 5.1. Let c T , c t ∈ (0, 1] be arbitrary positive constants, and c T T ≤ T ≤ T . Suppose f * ∈ H K (µ 0 ), and m satisfies
m ≳ d 8 3 /ε 80 3 n ,(9)
and the neural network f (W(t), •) is trained by GD using Algorithm 1 with the learning rate η ∈ [1, 2) and T ≤ T . Then for every t ∈ [c t T : T ], with probability at least 1exp (-Θ(n)) -7 exp -Θ(nε 2 n ) -2/n over the random noise w, the random training features S and the random initialization W(0), the stopping time satisfies T ≍ ε -2 n , and f (W(t), •) = f t satisfies
E P (f t -f * ) 2 ≲ ε 2 n .(10)
Significance of Theorem 5.1 and comparison to existing works. To the best of our knowledge, Theorem 5.1 is the first theoretical result which proves that over-parameterized neural network trained by gradient descent with early stopping achieves sharp rate of O(ε 2 n ), without distributional assumption on the covariate as long as the input space X is S d-1 . More discussions about the significance with comparison to existing works are detailed in Section 3.
When the polynomial EDR holds, we can apply Theorem 5.1 to obtain the following corollary. Corollary 5.2 (Applying Theorem 5.1 to the special case of polynomial EDR). Suppose λ j ≍ j -2α for j ≥ 1 and α > 1/2. Let c T , c t ∈ (0, 1] be positive constants, and
c T T ≤ T ≤ T . Suppose m satisfies m ≳ n 80α 3(2α+1) d 8 3 ,(11)
and the neural network f (W(t), •) is trained by GD using Algorithm 1 with the learning rate η ∈ [1, 2) and T ≤ T . Then for every t ∈ [c t T : T ], with probability at least 1exp (-Θ(n)) -7 exp -Θ(nε 2 n ) -2/n over the random noise w, the random training features S and the random initialization W(0), the stopping time satisfies T ≍ n
d 2d-1 , E P (f t -f * ) 2 ≲ 1 n 2α 2α+1 . (12
)
The significance of Corollary 5.2 is also detailed in Section 3. Section D of the appendix shows the simulation results with the empirical early stopping time and the theoretically predicted early stopping time, 1/ ε 2 n ≍ n d/(2d-1) , for a neural network trained by Algoirthm 1.
this section cite: []

Section: Roadmap of Proofs
We present the roadmap of our theoretical results which lead to the main result, Theorem 5.1 in Section 5. We first present in the next subsection our results about the uniform convergence to the NTK (2) and more, which are crucial in the analysis of training dynamics by GD.
this section cite: []

Section: Uniform Convergence to the NTK and More
We define functions
h(w, x, y) := x ⊤ y1I {w ⊤ x≥0} 1I {w ⊤ y≥0} , h(W, x, y) := 1 m m r=1 h( ⇀ w r , x, y), (13
) v R (w, x) := 1I {|w ⊤ x|≤R} , v R (W, x) := 1 m m r=1 v R ( ⇀ w r , x).(14)
Then we have the following theorem stating the uniform convergence of h(W(0), •, •) to K(•, •) and uniform con-
vergence of v R (W(0), x) to 2R √ 2πκ for a positive num- ber R ≲ ηT / √ m.
While existing works such as (Li et al., 2024) also has uniform convergence results for overparameterized neural network, our result does not depend on the Hölder continuity of the NTK. Theorem 6.1. The following results hold with η ≲ 1, m ≳ max n 2/d , T 8 5 , and m/ log 8 5 m ≥ d.
(1) With probability at least 1 -1/n over the random ini-
tialization W(0) = ⇀ w r (0) m r=1 , sup x∈X , y∈X K(x, y) -h(W(0), x, y) ≤ C 1 (m/2, d, 1/n) ≲ d log m m .(15)
(2) With probability at least 1 -1/n over the random ini-
tialization W(0) = ⇀ w r (0) m r=1 , sup x∈X | v R (W(0), x)| ≤ 2R √ 2πκ + C 2 (m/2, d, 1/n) ≲ √ dm -3 16 T 1 2 ,(16)
where C 1 (m/2, d, 1/n), C 2 (m/2, d, 1/n) are two positive numbers depending on (m, d, n), with their formal definitions deferred to ( 39) and ( 42) in Section C.2 of the appendix.
Proof. This theorem follows from Theorem C.1 and Theorem C.2 in Section C.2 of the appendix. Note that h(W,
x, y) = 1 m m r=1 h( ⇀ w r , x, y) = 1 m/2 m/2 r=1 h( ⇀ w 2r (0),
x, y), then part (1) directly follows from Theorem C.1. Similarly, part (2) directly follows from Theorem C.2.
this section cite: ['b23']

Section: Roadmap of Proofs
Because our main result, Theorem 5.1, is proved by Theorem C.10 and Theorem C.11 deferred to Section C.2, we illustrate in Figure 1, deferred to the appendix, the roadmap containing the intermediate theoretical results which lead to our main result, Theorem 5.1.
Summary of the technical approaches and novel results in the proofs. Theorem C.8 is the first novel result in the proofs of this work, showing that with high probability, the neural network function f (W(t), •) at step t of GD can be decomposed into two functions by f (W(t), •) = f t = h + e, where h ∈ H K is a function in the RKHS associated with K with bounded H K -norm. The error function e has a small L ∞ -norm, that is, ∥e∥ ∞ ≤ w with w being a small number controlled by the network width m, that is, larger m leads to smaller w. Theorem C.10 is the second novel result in the proofs, where we derive sharp and novel bound for the nonparametric regression risk of the neural network function f (W(t), •) in Theorem C.10, that is,
E P (f t -f * ) 2 -2E Pn (f t -f * ) 2 ≲ ε 2 n + w.
To the best of our knowledge, Theorem C.10 is among the first in the literature to employ local Rademacher complexity so as to obtain sharp rate for the risk of nonparametric regression which is distribution-free in spherical covariate, and local Rademacher complexity is employed to tightly bound the Rademacher complexity of the function class comprising all the possible neural network functions obtained by GD.
Novel proof strategy of this work. We remark that the proof strategy of our main result, Theorem 5.1, is signifi-cantly novel and different from the existing works in training over-parameterized neural networks for nonparametric regression with minimax rates (Hu et al., 2021;Suh et al., 2022;Li et al., 2024). In particular, the common proof strategy in these works uses the decomposition f t -f * = (f t -f (NTK) t ) + ( f (NTK) t -f * ) and then show that both f t -f (NTK) t L 2 and f (NTK) t -f * L 2 are bounded by certain minimax optimal rate, where f (NTK) t is the kernel regressor obtained by either kernel ridge regression (Hu et al., 2021;Suh et al., 2022) or GD with early stopping (Li et al., 2024). The remark after Theorem C.8 details a formulation of
f (NTK) t . f (NTK) t -f * L 2
is bounded by the minimax optimal rate under certain distributional assumptions in the covariate, and this is one reason for the distributional assumptions about the covariate in existing works such as (Hu et al., 2021;Suh et al., 2022;Li et al., 2024). In a strong contrast, our analysis does not rely on such decomposition of f t -f * . Instead of approximating f t by f (NTK) t , we have a new decomposition of f t by f t = h t + e t where f t is approximated by h t with e t being the approximation error. As suggested by the remark after Theorem C.8, we have h
t = f (NTK) t + e 2 (•, t) so that f t = f (NTK) t + e 2 (•, t) + e t .
Our analysis only requires the network width m to be suitably large so that the H K -norm of e 2 (•, t) is bounded by a positive constant and ∥e t ∥ ∞ ≤ w, while the common proof strategy in (Hu et al., 2021;Suh et al., 2022;Li et al., 2024) needs m to be sufficiently large so that both ∥ e 2 (•, t)∥ ∞ and ∥e t ∥ ∞ are bounded by an infinitesimal number (a minimax optimal rate such as O(n -d 2d-1 ) and then f t -f (NTK)
t L 2
is bounded by such minimax optimal rate. Detailed in Section 3, such novel proof strategy leads to our sharp analysis, rendering a smaller lower bound for m in our main result compared to some existing works.
this section cite: ['b15', 'b31', 'b23', 'b15', 'b31', 'b23', 'b15', 'b31', 'b23', 'b15', 'b31', 'b23']

Section: Conclusion
In this paper, we show that an over-parameterized two-layer neural network trained by gradient descent (GD) with early stopping renders a sharp rate of the nonparametric regression risk with the order of Θ(ε 2 n ) with ε n being the critical population rate or the critical radius of the NTK, which is distribution-free in spherical covariate. We compare our results to the current state-of-the-art with a detailed roadmap of our technical approaches and results in our proofs.
this section cite: []

Section: References
Ref_id:b0 Title: Learning and generalization in overparameterized neural networks, going beyond two layers Year: (2019-12-08)
Ref_id:b1 Title: A convergence theory for deep learning via over-parameterization Year: (2019)
Ref_id:b2 Title: Finegrained analysis of optimization and generalization for overparameterized two-layer neural networks Year: (2019)
Ref_id:b3 Title: Local rademacher complexities Year: (2005-08)
Ref_id:b4 Title: Deep learning: a statistical viewpoint Year: (2021)
Ref_id:b5 Title: On deep learning as a remedy for the curse of dimensionality in nonparametric regression Year: (2019)
Ref_id:b6 Title: Deep equals shallow for relu networks in kernel regimes Year: (2021)
Ref_id:b7 Title: On the inductive bias of neural tangent kernels Year: (2019)
Ref_id:b8 Title: How feature learning can improve neural scaling laws Year: (2024)
Ref_id:b9 Title: Generalization bounds of stochastic gradient descent for wide and deep neural networks Year: (2019)
Ref_id:b10 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b11 Title: Gradient descent finds global minima of deep neural networks Year: (2019)
Ref_id:b12 Title: Gradient descent provably optimizes over-parameterized neural networks Year: (2019)
Ref_id:b13 Title: Linearized two-layers neural networks in high dimension Year: (2021)
Ref_id:b14 Title: Mind the spikes: Benign overfitting of kernels and neural networks in fixed dimension Year: (2023)
Ref_id:b15 Title: Regularization matters: A nonparametric perspective on overparametrized neural network Year: (2021)
Ref_id:b16 Title: Deep neural networks learn non-smooth functions effectively Year: (2019)
Ref_id:b17 Title: Deep nonparametric regression on approximate manifolds: Nonasymptotic error bounds with polynomial prefactors Year: (2023)
Ref_id:b18 Title: Universal consistency of wide and deep relu neural networks and minimax optimal convergence rates for kolmogorov-donoho optimal function classes Year: (2024)
Ref_id:b19 Title: Local rademacher complexities and oracle inequalities in risk minimization Year: (2006)
Ref_id:b20 Title: Nonparametric regression with shallow overparameterized neural networks trained by GD with early stopping Year: (2021-08-19)
Ref_id:b21 Title: Deep learning Year: (2015)
Ref_id:b22 Title: Isoperimetry and Processes Year: (1991)
Ref_id:b23 Title: On the eigenvalue decay rates of a class of neural-network related kernel functions defined on general domains Year: (2024)
Ref_id:b24 Title: Geometric parameters of kernel machines Year: (2002)
Ref_id:b25 Title: Tight bounds on the smallest eigenvalue of the neural tangent kernel for deep relu networks Year: (2021)
Ref_id:b26 Title: Optimal rates for averaged stochastic gradient descent under neural tangent kernel regime Year: (2021)
Ref_id:b27 Title: Early stopping and non-parametric regression: an optimal datadependent stopping rule Year: (2014)
Ref_id:b28 Title: Nonparametric regression using deep neural networks with ReLU activation function Year: (2020)
Ref_id:b29 Title: Additive Regression and Other Nonparametric Models Year: (1985)
Ref_id:b30 Title: On learning over-parameterized neural networks: A functional approximation perspective Year: (2019)
Ref_id:b31 Title: A non-parametric regression viewpoint : Generalization of overparametrized deep RELU network under noisy observations Year: (2022)
Ref_id:b32 Title: High-Dimensional Statistics: A Non-Asymptotic Viewpoint Year: (2019)
Ref_id:b33 Title: A Bound on Tail Probabilities for Quadratic Forms in Independent Random Variables Whose Distributions are not Necessarily Symmetric Year: (1973)
Ref_id:b34 Title: Tensor programs IV: feature learning in infinite-width neural networks Year: (2021-07-24)
Ref_id:b35 Title: Information-theoretic determination of minimax rates of convergence Year: (1999)
Ref_id:b36 Title: Gradient descent finds overparameterized neural networks with sharp generalization for nonparametric regression Year: (2025)
Ref_id:b37 Title: Error bounds for approximations with deep relu networks Year: (2017)
Ref_id:b38 Title: Minimax optimal rates of estimation in high dimensional additive models Year: (2016)
Ref_id:b39 Title: Deep learning meets nonparametric regression: Are weight-decayed dnns locally adaptive? Year: (2023)
Ref_id:b40 Title: A type of generalization error induced by initialization in deep neural networks Year: (2020-07)
