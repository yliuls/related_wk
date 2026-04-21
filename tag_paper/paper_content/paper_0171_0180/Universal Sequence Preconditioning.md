Title: Universal Sequence Preconditioning
Abstract: We study the problem of preconditioning in sequential prediction. From the theoretical lens of linear dynamical systems, we show that convolving the target sequence corresponds to applying a polynomial to the hidden transition matrix. Building on this insight, we propose a universal preconditioning method that convolves the target with coefficients from orthogonal polynomials such as Chebyshev or Legendre. We prove that this approach reduces regret for two distinct prediction algorithms and yields the first ever sublinear and hidden-dimension-independent regret bounds (up to logarithmic factors) that hold for systems with marginally stable and asymmetric transition matrices. Finally, extensive synthetic and realworld experiments show that this simple preconditioning strategy improves the performance of a diverse range of algorithms, including recurrent neural networks, and generalizes to signals beyond linear dynamical systems.

Section: Introduction
In sequence prediction the goal of the learner is to predict the next token accurately according to a specified loss function, such as the mean square error or cross-entropy. This fundamental problem in machine learning has gained increased importance with the rise of large language models, which perform sequence prediction on tokens using cross entropy. The focus of this paper is preconditioning, i.e. modifying the target sequence to make it easier to learn. A classic example is differencing, introduced by Box and Jenkins in the 1970s [13], which transforms observations y 1 , y 2 , . . . into successive differences, y 1 → y 0 , y 2 → y 1 , ..., y t → y t↑1 , ...
It is widely acknowledged that learning this sequence can be "easier" than learning the original sequence for a large number of modalities. In this work we seek a more general framework for sequence preconditioning that captures the same intuition behind differencing and extends it to a broader class of transformations. The question we ask is What is the general form of sequence preconditioning that enables provably accurate learning?
We address this question by introducing a preconditioning method which takes in n fixed coefficients c 0 , . . . , c n and converts the sequence of observations y 1 , . . . , y t , . . . to the sequence of convolved observations 1 c 0 y 0 , c 0 y 1 + c 1 y 0 , . . . , n i=0 c i y t↑i , . . .
From an information-theoretic perspective, approaches of this kind seem futile-predicting y t or i c i y t↑i seems equally hard in an adversarial setting. Yet we show that when the data arises from a linear dynamical system (LDS), there exists a universal form of preconditioning that provably improves learnability, independent of the specific system. In the LDS setting, we show that preconditioning significantly strengthens existing prediction methods, leading to new regret bounds. Here, preconditioning has an elegant effect: the preconditioning filter forms coefficients of an n degree polynomial, and the hidden system transition matrix is evaluated on this polynomial-potentially shrinking the domain. In this setting, shrinking the learnable domain is akin to making the problem "easier to learn", a relationship that is formalized by [26]. This allows us to prove the first dimensionindependent sublinear regret bounds for asymmetric linear dynamical systems that are marginally stable.
this section cite: ['b12', 'b25']

Section: Our results
Our main contribution is Universal Sequence Preconditioning, a novel method of sequence preconditioning which convolves the target sequence with the coefficients of the n-th monic Chebyshev polynomial. We give a more general form of preconditioning, allowing arbitrary user-specified coefficients, in Algorithm 1 and an online version in Algorithm 4 (Appendix C). We analyze the effect of Universal Sequence Preconditioning on two canonical sequence prediction algorithms in the online setting: (1) convex regression and (2) spectral filtering. In either case, the results are impressive-yielding the first known sublinear regret bounds as compared to the optimal ground-truth predictor that are simultaneously (1) applicable to marginally stable systems, (2) independent of the hidden dimension (up to logarithmic factors), and (3) applicable to systems whose transition matrix is asymmetricfoot_0 (see Table 1 ).
Algorithm 1 General Sequence Preconditioning (Offline Version) 1: Training 2: Input: training data (u 1:N 1:T , y 1:N 1:T ) where (u i t , y i t ) is the t-th input/output pair in the i-th sequence; coefficients c 0:n ; prediction algorithm A. 3: Assert c 0 = 1. 4: for i = 1 to N do 5: y preconditioned,i 1:T ↑ convolution(y i 1:T , c 0:n ) ¡ y preconditioned,i t = y i t + n j=1 c j y i t↑j 6: end for 7: Train A on preconditioned data u 1:N 1:T , y preconditioned,1:N 1:T . 8: Test Time 9: for t = 1 to T do 10: Receive u t . 11: Predict ŷt ↑ A u 1:t , y 1:(t↑1) → n i=1 c i y t↑i . 12:
Receive y t . 13: end for First, applying USP to standard convex regression results in regret Õ(T ↑2/13 ), which holds simultaneously across the three settings above and remains dimension-independent. For comparison, a naive analysis of regression yields a vacuous regret bound of O(T 5/2 ) on marginally stable systems. Second, combining USP with a variant of spectral filtering [27] that uses novel filters, the algorithm is able learn a broader class of linear dynamical systems-in particular systems whose hidden transition matrix may be asymmetric. The enhanced method achieves regret Õ(T ↑3/13 ), the best known rate under the joint conditions of (1)-(3) discussed above: marginal stability, dimension independence, and asymmetry. Both results require that the transition matrix eigenvalues have imaginary parts bounded by O(1/ log T )-a near-tight condition for achieving dimension-free regret. Further discussion on this appears in Appendix B.
Empirical results in Section 4 demonstrate that USP consistently improves performance across diverse algorithms-including regression, spectral filtering, and neural networks-and across data types extending beyond linear dynamical systems.
this section cite: ['b26']

Section: Intuition for Universal Sequence Preconditioning
We now give some brief intuition for the result. Linear dynamical systems (LDS) are perhaps the most basic and well studied dynamical systems in engineering and control science. Given input vectors u 1 , . . . , u T ↗ C din , the system generates a sequence of output vectors y 1 , . . . y T ↗ C dout according to the law
x t+1 = Ax t + Bu t , y t = Cx t + Du t ,(1)
where x 0 , . . . , x T ↗ C dhidden is a sequence of hidden states and (A, B, C, D) are matrices which parameterize the LDS. We assume w.l.o.g. that D = 0. We can factor out the hidden state x t so that the observation at time t is
y t = t s=1 CA t↑s Bu s . Given coefficients c 0:n = (c 0 , . . . , c n ) let p c n (x) def = n i=0 c i x n↑i .(2)
Consider a "preconditioned" target at time t to be a linear combination of y t:t↑n with coefficients c 0:n . A key insight is the following identity, If we take c 0 = 1 (i.e. a monic polynomial), we can re-write y t as
y t = → n i=1 c i y t↑i ↓0 + n↑1 s=0 s i=0 c i CA s↑i Bu t↑s ↓1 + t↑n↑1 s=0 Cp c n (A)A s Bu t↑n↑s ↓2 .(3)
This expression highlights our approach as a balance of three terms:
↘ 0 The universal preconditioning term: it depends only on the coefficients c 0:n and not on any learning algorithm. ↘ 1 A term learnable via convex relaxation and regression, for example by denoting
Q learned s = s i=0 c i CA s↑i B.
The diameter of the coefficient Q s depends on the magnitude of the coefficients c 0:n . ↘ 2 The residual term with polynomial p c n (A). By a careful choice of coefficients c 0:n , we can force this term to be very small. The main insight we derive from this expression is the inherent tension between two terms ↘ 1 , ↘ 2 . The polynomial p c n (x) and its coefficients c 0 , . . . , c n control two competing effects:
1. The preconditioning coefficients grow larger with the degree n of the polynomial and the magnitude of the coefficients c i . A higher degree polynomial and larger coefficients increase the diameter of the search space over the preconditioning coefficients, and therefore increase the regret bound stemming from the ↘ 1 component learning. 2. On the other hand, a larger search space can allow a broader class of polynomials p n (•) which can better control of the magnitude of p n (A), and therefore reduce the search space of the ↘ 2 component.
What choice of polynomial is best? This work considers the Chebyshev polynomial. The reason is the following property of the n-th monic Chebyshev polynomial: n↑1)  .
max ω↔[↑1,1] |p n (ω)| ≃ 2 ↑(
As an example, consider any LDS whose hidden transition matrix A is diagonalizable and has eigenvalues in [→1, 1]foot_1 . By the above property, observe that ⇐p n (A)⇐ ↗ ≃ 2 • 2 ↑n , therefore shrinking the ↘ 2 term at a rate exponential with the number of preconditioning coefficients. We pause to remark on the universality of this choice of polynomial. Indeed, one could instead have chosen the preconditioning coefficients to depend on A so that p c n (•) is the characteristic polynomial of A. By the Cayley-Hamilton theorem, p n (A) = 0. This means that ↘ 2 term is canceled out completely. However this would have required knowledge of the spectrum of A. The Chebyshev polynomial, on the other hand, is agnostic to the particular hidden transition matrix. Moreover, even if the spectrum of A were known, choosing the preconditioning coefficients to form the characteristic polynomial would result in an algorithm which must learn hidden dimension many parameters, which is prohibitive. Instead, the degree of the Chebyshev polynomial must only grow logarithmically with the hidden dimension.
this section cite: []

Section: Related work
Our manuscript is technically involved and incorporates linear dynamical systems, spectral filtering, complex Chebyshev and Legendre polynomials, Hankel and Toeplitz matrix eigendecay, Gaussian quadrature and other techniques. The related work is thus expansive, and due to space limitations we give a detailed treatment in Appendix A. Preconditioning in the context of time series analysis has roots in the classical work of Box and Jenkins [13]. In their foundational text they propose differencing as a method for making the time series stationary, and thus amenable to statistical learning techniques such as ARMA (auto-regressive moving average) [9]. The differencing operator can be applied numerous times, and for different lags, giving rise to the ARIMA family of forecasting models. Identifying the order of an ARIMA model, and in particular the types of differencing needed to make a series stationary, is a hard problem. This is a special case of the problem we consider: differencing corresponds to certain coefficients of preconditioning the time series, whereas we consider arbitrary coefficients. For a thorough introduction to modern control theory and exposition on open loop / closed loop predictors, learning via regression, and spectral filtering, see [26]. The fundamental problem of learning in linear dynamical systems has been studied for many decades, and we highlight several key approaches below:
1. System identification refers to the method of recovering A, B, C from the data. This is a non-convex problem and while many methods have been considered in this setting, they depend polynomially on the hidden dimension.
2. The (auto) regression method predicts according to ŷt = h i=1 M i u t↑i . The coefficients M i can be learned using convex regression. The downside of this approach is that if the spectral radius of A is 1 → ε, it can be seen that ↔ 1 ε terms are needed. 3. The regression method can be further enhanced with "closed loop" components, that regress on prior observations y t↑1:1 . It can be shown using the Cayley-Hamilton theorem that using this method, d h components are needed to learn the system, where d h is the hidden dimension of A. 4. Filtering involves recovering the state x t from observations. While Kalman filtering is optimal under specific noise conditions, it generally fails in the presence of marginal stability and adversarial noise.
5. Finally, spectral filtering combines the advantages of all methods above. It is an efficient method, its complexity does not depend on the hidden dimension, and works for marginally stable systems. However, spectral filtering requires A to be symmetric, or diagonalizable under the real numbers.
this section cite: ['b12', 'b8', 'b25']

Section: Main Results
In this section we formally state our main algorithms and theorems. We show that the Universal Sequence Preconditioning method provides significantly improved regret bounds for learning linear dynamical systems than previously known when used in conjunction with two distinct methods. The first method is simple convex regression, and the second is spectral filtering. Both algorithms allow for learning in the case of marginally stable linear dynamical systems and allow for certain asymmetric transition matrices of arbitrary high hidden dimension. The regret bounds are free of the hidden dimension (up to logarithmic factors) -which significantly extends the state of the art.
this section cite: []

Section: Universal Sequence Preconditioning Applied to Regression
Algorithm 2 is an instantiation of Algorithm 4 for the method of convex regression. We set the preconditioning coefficients to be the coefficients of the n-th degree (monic) Chebyshev polynomial.
Algorithm 2 Universal Sequence Preconditioning for Regression 1: Input: initial parameter Q 0 ; preconditioning coefficients c 0:n from the n-th degree (monic) Chebyshev polynomial; convex constraints
K = {(Q 0 , . . . , Q n↑1 ) s.t. ⇐Q j ⇐ ≃ C domain ⇐c⇐ 1 } 2: Assert that c 0 = 1. 3: for t = 1 to T do 4: Receive u t . 5: Predict ŷt (Q t ) = → n i=1 c i y t↑i + n j=0 Q t j u t↑j . 6:
Observe true output y t and suffer loss ϑ t (Q t ) = ⇐ŷ t (Q t ) → y t ⇐ 1 .
7:
Update and project:
Q t+1 ↑ proj K Q t → ϖ t ⇒ Q ϑ t (Q t ) .
8: end for Theorem 2.1 shows that vanishing loss compared to the optimal ground-truth predictor, at a rate that is independent of the hidden dimension of the system.
Theorem 2.1. Let {u t } T t=1 ↗ C din be any sequence of inputs which satisfy ⇐u t ⇐ 2 ≃ 1 and let {y t } T t=1 ↗ C dout be the corresponding output coming from some linear dynamical system (A, B, C) as defined per Eq. 1. Let P diagonalize A (note P exists w.l.o.g.) and let ϱ = ⇐P⇐⇐P ↑1 ⇐. Assume that ⇐B⇐⇐C⇐ϱ ≃ C domain . Let ω 1 , . . . ,
ω d h denote the spectrum of A. If max j↔[d h ] | arg(ω j )| ≃ 1/(32 log 2 (2T 3 /d out )) 2
then the predictions ŷ1 , . . . , ŷT from Algorithm 2 where the preconditioning coefficients c 0:n are chosen to be the coefficients of the n-th monic Chebyshev polynomial satisfy
1 T T t=1 ⇐ŷ t → y t ⇐ 1 ≃ Õ ⇐B⇐⇐C⇐ϱ ⇑ d out T 2/13
, where Õ(•) hides polylogarithmic factors in T .
The proof of Theorem 2.1 is in Appendix D. For a simple baseline comparison, the regret achieved (via the same proof technique) by the vanilla regression algorithm without preconditioning is
O C domain ⇑ d out T 5/2 which is not sublinear in T .
this section cite: []

Section: Universal Sequence Preconditioning Applied to Spectral Filtering
Our second main result is the application of Universal Sequence Preconditioning to the spectral filtering algorithm [27]. Our results are more general and apply to any choice of polynomial, not just Chebyshev. In addition to applying USP to spectral filtering, we also propose a novel spectral filtering basis. Both changes to the vanilla spectral filtering algorithm are necessary to extend its sublinear regret bounds to the case of underlying systems with asymmetric hidden transition matrices. First we define the spectral domain
C ϑ = {z ↗ C | |z| ≃ 1, | arg(z)| ≃ ς} .
Given horizon T and φ ↗ C ϑ let μT (φ
) def = (1 → φ 2 ) 1 φ ... φ T ↑1  ↘ ,(4)
and
Z T def =  ϖ↔C ω μT (φ)μ T (φ) ↘ dφ,(5)
where φ ↗ C denotes the complex conjugate. The novel spectral filters are the eigenvectors of Z T ↑n↑1 , which we denote as ↼ 1 , . . . , ↼ T ↑n↑1 . Note that in the standard spectral filtering literature, the spectral filtering matrix is an integral over the real line and does not involve the complex conjugate.
Our new matrix has an entirely different structure and although it looks quite similar, it surprisingly upends the proof techniques to ensure exponential spectral decay, a critical property for the method. Future work examines this matrix more thoroughly, but in this paper we simply provide a standard bound on its eigenvalues.
Algorithm 3 Universal Sequence Preconditioning for Spectral Filtering x). Let c0 , . . . , cn+2 be the coefficients of pc n (x). 3: Let ↼ 1 , ..., ↼ n be the top n eigenvectors of Z T ↑n↑1 . 4: Assert c0 = 1. 5: for t = 1 to T do 6:
1: Input: initial Q 1 1:n , M 1 1:k , horizon T , convex constraints K = {(Q 0 , . . . , Q n↑1 , M 1 , . . . , M k s.t. ⇐Q j ⇐ ≃ R Q and ⇐M j ⇐ ≃ R M } , parameter n, coefficients c 1:n . 2: Let p c n (x) = c 0 x n + c 1 x n↑1 + • • • + c n and pc n (x) = (1 → x 2 )p c n(
Let ũt↑n↑1:1 be u t↑n↑1:1 padded with zeros so it has dimension T → n → 1 ↓ d in .
7: Predict ŷt (Q t , M t ) = → n+2 i=1 ci y t↑i + n j=0 Q t j u t↑j + 1 ≃ T k j=1 M t j ↼ ↘ j ũt↑n↑1:1 . 8:
Observe true y t , define loss ϑ t (ŷ t ) = ⇐ŷ t (Q t , M t ) → y t ⇐ 1 .
9:
Update and project: (Q t+1 , M t+1 ) = proj K (Q t , M t ) → ϖ t ⇒ϑ t (Q t , M t )) 10: end for Theorem 2.2. Let {u t } T t=1 ↗ R din be any sequence of inputs which satisfy ⇐u t ⇐ 2 ≃ 1 and let {y t } T t=1 be the corresponding output coming from some linear dynamical system (A, B, C) as defined per Eq. 1. Let P diagonalize A (note P exists w.l.o.g.) and let ϱ = ⇐P⇐⇐P ↑1 ⇐. Suppose the radius parameters of Algorithm 3 satisfy R Q ⇓ ⇐C⇐⇐B⇐⇐c⇐ 1 and
R M ⇓ 2⇐C⇐⇐B⇐ϱ log(T ) max j↔[d h ] | arg(ω j )| 4/3 T 7/6 max ϖ↔C ω |p c n (φ)| .
Further suppose that the eigenvalues of A have bounded argument:
max j↔[d h ] | arg(ω j )| ≃ T ↑1/3 .
Then the predictions ŷ1 , . . . , ŷT from Algorithm 3 where the preconditioning coefficients c 0:n are chosen to be the coefficients of the n-th monic Chebyshev polynomial satisfy
1 T T t=1 ⇐ŷ t → y t ⇐ 1 ≃ Õ ⇐C⇐⇐B⇐ϱ ⇑ d out T 1/39 .
We remark that the result of Theorem 2.2 is rather weak. Although the complex eigenvalues of A are not trivially bounded (trivial would be a bound of 1/T ), they still must be polynomially small in T . Moreover we note that the proof technique for the result does not make use of the critical properties of spectral filtering and relies much more on the power of preconditioning. The proof of Theorem 2.2 is in Section E.
this section cite: ['b26']

Section: Proof Overview
In this section we give a high level overview of the proofs for Theorem 2.1 and Theorem 2.2. We start by recalling the intuition for Universal Sequence Preconditioning developed in Section 1.2 which shows that if {y t } T t=1 evolves as a linear dynamical system parameterized by matrices (A, B, C) with inputs {u t } Recall that ↘ 0 is the universal preconditioning component, ↘ 1 is the term that can easily be learned by convex relaxation and regression, and ↘ 2 is the critical term that contains p c n (A). Both Theorem 2.1 and Theorem 2.2 use the standard result from online convex optimization (Theorem 3.1 from [24]) that online gradient descent over convex domain K achieves regret 3  2 GD ⇑ T as compared to the best point in K, where D denotes the diameter of K and G denotes the maximum gradient norm.
Regression: Proof of Theorem 2.1 In the case of regression, the domain is chosen so that ↘ 2 may be learned and the proof proceeds by bounding the diameter of such a domain and its corresponding maximum gradient norm to get regret Cn 2 ⇑ d out ⇐c⇐ 1 ⇑ T for a universal constant C > 0 which depends on the norms of matrices B and C from the underlying system. Then ↘ 3 is treated as an un-learnable error term. Let ω(A) denote the set of eigenvalues of A. By the simple magnitude bound of
⇐ t↑n↑1 s=0 Cp n (A)A s Bu t↑n↑s ⇐ ≃ max ω↔ω(A) |p n (ω)| • T • ⇐C⇐ • ⇐B⇐,
the error of ignoring this term can be very small if max ω(A) |p n (ω(A))| is small. In the proof of Theorem 2.1 in Appendix D we show that the regret for a generic polynomial p c n defined by coefficients c 0:n is
T t=1 ⇐y t → ŷt ⇐ 1 ≃ Cn 2  d out ⇐c⇐ 1 ⇑ T Regret from learning ↓2 + C max ω↔D |p c n (ω)|T 2
this section cite: ['b23']

Section: Unlearnable Error Term
, where D is the region where A is allowed to have eigenvalues (see Theorem D.1). Therefore, to get sublinear regret, we must choose a polynomial which has bounded ϑ 1 norm of its coefficients, while also exhibits very small infinity norm on the domain of A's eigenvalues.
this section cite: []

Section: Spectral Filtering: Proof of Theorem 2.2
In the case of spectral filtering, the domain is chosen so that both ↘ 2 and ↘ 3 may be learned. Because spectral filtering learns ↘ 3 , it is able to accumulate less error and hence achieves a better regret bound of O(T ↑3/13 ) as compared to regression's O(T ↑2/13 ). At a high level, the proof proceeds by exploiting the fact that p n (A) shrinks the size of the learnable domain. However this is not enough, in order to extend the result to systems where A may have complex eigenvalues, the spectral filters must be eigenvalues of a new matrix, defined in Eq. 5, whose domain of integration includes the possibly complex eigenvalues of A. To get the dimensionindependent regret bounds enjoyed by spectral filtering in this new setting where complex eigenvalues may occur, the exponential decay of Z T from Eq. 5 must be established. This is nontrivial and requires several novel techniques inspired by [11]. The details are in Appendix E.3.1. Theorem 2.2 gives the main guarantee for the spectral filtering algorithm, which states that Algorithm 3 instantiated with some choice of polynomial p c n (•) achieves regret
Õ n⇐c⇐ 1 + T 7/6 max ϖ↔C ω |p c n (φ)| (n + k)  d out ⇑ T .
Both this theorem, as well as our new guarantee for convex regression, leads us to the following question: Is there a universal choice of polynomial p n (x), where n is independent of hidden dimension, which guarantees sublinear regret?
this section cite: ['b10']

Section: Using the Chebyshev Polynomial over the Complex Plane
For the real line, the answer to this question is known to be positive using the Chebyshev polynomials of the first kind. In general, the n th (monic n↑1) . However, we are interested in a more general question over the complex plane. Since we care about linear dynamical systems that evolve according to a general asymetric matrix, we need to extending our analysis to C ϑ . This is a nontrivial extension since, in general, functions that are bounded on the real line can grow exponentially on the complex plane. Indeed, 2 n↑1 M n (x) = cos(n arccos(x)) and while cos(x) is bounded within [→1, 1] for any x ↗ R, over the complex numbers we have cos(z) = 1  2 (e iz + e ↑iz ), which is unbounded. Thus, we analyze the Chebyshev polynomial on the complex plane and provide the following bound. Lemma 3.1. Let z ↗ C be some complex number with magnitude
) Chebyshev polynomial M n (x) satisfies max x↔[↑1,1] |M n (x)| ≃ 2 ↑(
|φ| ≃ 1. Let M n (•) denote the n-th monic Chebyshev polynomial. If | arg(z)| ≃ 1/64n 2 , then |M n (z)| ≃ 1/2 n↑2 .
We provide the proof in Appendix B. We also must analyze the magnitude of the coefficients of the Chebyshev polynomial, which can grow exponentially with n. We provide the following result.
Lemma 3.2. Let M n (•) have coefficients c 0 , . . . , c n . Then max k=0,...,n |c k | ≃ 2 0.3n .
The proof of Lemma 3.2 is in Appendix B. Together, these two lemmas are the fundamental building block for universal sequence preconditioning and for obtaining our new regret bounds.
this section cite: []

Section: Experimental Evaluation
We empirically validate that convolutional preconditioning with Chebyshev or Legendre coefficients yields significant online regret improvements across various learning algorithms and data types. Below we summarize our data generation, algorithm variants, hyperparameter tuning, and evaluation metrics.
this section cite: []

Section: Synthetic Data Generation
We generate N = 200 sequences of length T = 2000 via three mechanisms: (i) a noisy linear dynamical system, (ii) a noisy nonlinear dynamical system, and (iii) a noisy deep RNN. Inputs u 1:T ↔ N (0, I). (A, B, C) with A ↗ R 300⇐300 having eigenvalues {z j } drawn uniformly in the complex plane subject to Im(z j ) ≃ ↽ thresh and L ≃ |z j | ≃ U , and B, C ↗ R 300 . Then
this section cite: []

Section: Linear Dynamical System. Sample
x t = A x t↑1 + B u t , y t = C x t + ⇀ t , ⇀ t ↔ N (0, ⇁ 2 I). Nonlinear Dynamical System. Similarly sample (A 1 , B 1 , C) and (A 2 , B 2 ) with A i ↗ R 10⇐10 , B i , C ↗ R 10 . Then x (0) t = A 1 x t↑1 + B 1 u t , x(1)
t = ⇁ x (0) t , x t = A 2 x(1)
t + B 2 u t , y t = Cx t + ⇀ t .
Deep RNN. We randomly initialize a sparse 10-layer stack of LSTMs with hidden dimension 100 and ReLU nonlinear activations. Given u 1:T we use this network to generate y 1:T .
this section cite: []

Section: Algorithms and Preconditioning Variants
We evaluate the following methods: (1) Regression (Alg. 2) , (2) Spectral Filtering (Alg. 3) , (3) DNN Predictor: n-layer LSTM with dims [d 1 , . . . , d n ], ReLU. Each method is applied with one of:
1. Baseline: no preconditioning 2. Chebyshev: c 0:n are the coefficients for the nth-Chebyshev polynomial. Note that when n = 2 we have c 0 = 1 and c 1 = →1 and therefore this is the method of differencing discussed in the introduction.
this section cite: []

Section: 3.
Legendre: c 0:n are the coefficients for the nth-Legendre polynomial 4. Learned: c 0:n is a parameter learned jointly with the model parameters
We test polynomial degrees n ↗ {2, 5, 10, 20}. This choice of degrees shows a rough picture of the impact of n.
Hyperparameter Tuning. To ensure fair comparison, for each algorithm and conditioning c variant we perform a grid search over learning rates ϖ ↗ {10 ↑3 , 10 ↑2 , 10 ↑1 }, selecting the one minimizing average regret across the N sequences. In the case of the learned coefficients, we sweep over the 9 pairs of learning rates (ϖ model , ϖ coefficients ) ↗ {10 ↑3
, 10 ↑2 , 10 ↑1 } ↓ {10 ↑3 , 10 ↑2 , 10 ↑1 }.
this section cite: []

Section: Results
Tables 2-4 report the mean ± std of the absolute error over the final 200 predictions, averaged across 200 runs. In the linear and nonlinear cases we train a 2-layer DNN (dims (64, 128)); for RNN-generated data we match the 10-layer (100-dim) generator.
this section cite: []

Section: Key observations:
• Preconditioning drastically reduces baseline errors for all algorithms and data types.
• Chebyshev and Legendre yield nearly identical gains.
• For Chebyshev and Legendre, once the degree is higher than 5→10 the performance degrades since ⇐c⇐ 1 gets very large (see our Lemma 3.2 which shows that these coefficients grow exponentially fast).
• Improvements decay as the complex threshold ↽ thresh increases, consistent with our theoretical results which must bound Im(z j ).
• Learned coefficients excel with regression and spectral filtering but destabilize the DNN on nonlinear and RNN-generated data.
Setting Baseline Chebyshev Legendre Learned Deg. 2 Deg. 5 Deg. 10 Deg. 2 Deg. 5 Deg. 10 Deg. 2 Deg. 5 Deg. 10 Deg. 20 Regression ↽thresh = 0.01 0.74 ± 0.28 0.25 ± 0.09 0.15 ± 0.07 0.77 ± 0.31 0.36 ± 0.13 0.14 ± 0.06 0.64 ± 0.26 0.52 ± 0.19 0.27 ± 0.11 0.17 ± 0.07 0.24 ± 0.09 ↽thresh = 0.1 1.92 ± 0.81 0.84 ± 0.27 0.66 ± 0.18 1.90 ± 0.67 1.10 ± 0.40 0.63 ± 0.17 1.66 ± 0.58 1.34 ± 0.43 0.57 ± 0.14 0.55 ± 0.14 0.56 ± 0.14 s ↽thresh = 0.9 2.47 ± 0.89 1.59 ± 0.56 2.18 ± 0.79 2.68 ± 0.48 1.64 ± 0.58 1.94 ± 0.70 2.63 ± 0.45 1.73 ± 0.59 0.83 ± 0.25 0.68 ± 0.27 0.63 ± 0.26 Spectral Filtering ↽thresh = 0.01 5.94 ± 3.37 1.72 ± 0.95 0.69 ± 0.38 3.25 ± 1.79 2.78 ± 1.56 0.66± 0.36 2.74 ± 1.51 1.99 ± 0.94 0.54± 0.29 0.55 ± 0.25 0.61 ± 0.25 ↽thresh = 0.1 0.89 ± 0.34 0.42 ± 0.11 0.34 ± 0.07 0.86 ± 0.28 0.54 ± 0.17 0.33 ± 0.07 0.76 ± 0.24 0.69 ± 0.28 0.31 ± 0.06 0.37 ± 0.08 0.45 ± 0.28 ↽thresh = 0.9 10.17 ± 8.80 9.87 ± 8.90 12.66 ± 8.18 32.90 ± 22.02 9.42 ± 8.39 11.53 ± 7.46 28.83 ± 19.24 7.93 ± 4.42 6.31 ± 4.19 5.73 ± 3.80 5.86 ± 4.02 2-layer DNN ↽thresh = 0.01 4.49 ± 2.02 2.31 ± 1.18 2.62 ± 1.52 10.36± 6.05 2.79 ± 1.34 2.35 ± 1.35 8.92 ± 5.20 2.89 ± 1.24 1.56 ± 0.64 0.79 ± 0.25 0.40 ± 0.16 ↽thresh = 0.1 9.41 ± 7.34 2.66 ± 1.76 1.52 ± 0.72 4.65 ± 3.03 4.24± 3.06 1.44 ± 0.71 4.03 ± 2.64 6.54 ± 4.32 3.22 ± 1.93 1.59 ± 0.95 0.80± 0.48 ↽thresh = 0.9 2.45 ± 1.31 2.24 ± 1.34 3.49 ± 2.27 11.48 ± 8.15 2.17 ± 1.25 3.10 ± 2.00 9.97 ± 7.05 1.29 ± 0.66 0.71 ± 0.34 0.43 ± 0.18 0.24 ± 0.11 Table 4: Performance (average absolute error of the last 200 predictions) of a 10-layer DNN (detailed in Sec. 4.2) on data generated from the same model (detailed in Sec. 4.1).
this section cite: []

Section: ETTh1 Dataset
To evaluate whether our proposed preconditioning approach generalizes to real-world time series, we conduct experiments on the well-established ETTh1 dataset from the Electricity Transformer Temperature (ETT) benchmark [37]. The ETTh1 dataset consists of continuous hourly measurements of load and oil temperature collected from electricity transformers and has been used in several recent works [37,35,32,22,23,36,31]. We study the effect of preconditioning on a 10-layer LSTM with hidden dimension 100 per layer using the Adam optimizer. We set the horizon to be T = 5000 and we sweep over a broader range of learning rates ϖ ↗ {10 ↑j } j=0,1,2,3,4,5 . As before we consider (i) no preconditioning (baseline), (ii) fixed Chebyshev coefficients, (iii) fixed Legendre coefficients, and (iv) coefficients learned jointly with model parameters. As seen in Figure 1, preconditioning with Chebyshev and Legendre for degree 5 the best performance after only the first 1000 iterations, while the performance of jointly learning the coefficients is worse at this stage. The performance of all three preconditioning methods are roughly on par with each other by 2500 iterations and by the full horizon T = 5000, jointly learning the coefficients results in the best average prediction error.
this section cite: ['b36', 'b36', 'b34', 'b31', 'b21', 'b22', 'b35', 'b30']

Section: Discussion
There are many settings in machine learning where universal, rather than learned, rules have proven very efficient. For example, physical laws of motion can be learned directly from observation data. However, Newton's laws of motion succinctly crystallize very general phenomenon, and have proven very useful for large scale physics simulation engines. Similarly, in the theory of mathematical optimization, adaptive gradient methods have revolutionized deep learning. Their derivation as a consequence of regularization in online regret minimization is particularly simple [19], and thousands of research papers have not dramatically improved the initial basic ideas. These optimizers are, at the very least, a great way to initialize learned optimizers [34].
By analogy, our thesis in this paper is that universal preconditioning based on the solid theory of dynamical systems can be applicable to many domains or, at the very least, an initialization for other learning methods.
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: [NA]
this section cite: ['b18', 'b33']

Section: References
Ref_id:b0 Title: Tracking adversarial targets Year: (2014)
Ref_id:b1 Title: Regret bounds for the adaptive control of linear quadratic systems Year: (2011)
Ref_id:b2 Title: Handbook of mathematical functions with formulas, graphs, and mathematical tables Year: (1948)
Ref_id:b3 Title: Online control with adversarial disturbances Year: (2019)
Ref_id:b4 Title: Futurefill: Fast generation from convolutional sequence models Year: (2024)
Ref_id:b5 Title: Logarithmic regret for online control Year: (2019)
Ref_id:b6 Title: Spectral state space models Year: (2023)
Ref_id:b7 Title: Online learning for adversaries with memory: price of past mistakes Year: (2015)
Ref_id:b8 Title: Online learning for time series prediction Year: (2013)
Ref_id:b9 Title: A new approach to learning linear dynamical systems Year: (2023)
Ref_id:b10 Title: On the singular values of matrices with displacement structure Year: (2017)
Ref_id:b11 Title: Dynamic Programming and Optimal Control Year: (2007)
Ref_id:b12 Title:  Year: (1976)
Ref_id:b13 Title: Prediction, learning, and games Year: (2006)
Ref_id:b14 Title: Black-box control for linear dynamical systems Year: (2021)
Ref_id:b15 Title: Online linear quadratic control Year: (2018)
Ref_id:b16 Title: Learning linear-quadratic regulators efficiently with only ⇑ T regret Year: (2019)
Ref_id:b17 Title: Regret bounds for robust adaptive control of the linear quadratic regulator Year: (2018)
Ref_id:b18 Title: Adaptive subgradient methods for online learning and stochastic optimization Year: (2011)
Ref_id:b19 Title: Global convergence of policy gradient methods for the linear quadratic regulator Year: (2018)
Ref_id:b20 Title: No-regret prediction in marginally stable systems Year: (2020)
Ref_id:b21 Title: Efficiently modeling long sequences with structured state spaces Year: (2022)
Ref_id:b22 Title: Simplifying long-range modeling with structured state space models Year: (2022)
Ref_id:b23 Title: Introduction to online convex optimization Year: (2016)
Ref_id:b24 Title: Spectral filtering for general linear dynamical systems Year: (2018)
Ref_id:b25 Title: Introduction to online nonstochastic control Year: (2022)
Ref_id:b26 Title: Learning linear dynamical systems via spectral filtering Year: (2017)
Ref_id:b27 Title: A new approach to linear filtering and prediction problems Year: (1960)
Ref_id:b28 Title: On-line learning of linear dynamical systems: Exponential forgetting in kalman filters Year: (2019)
Ref_id:b29 Title: Certainty equivalence is efficient for linear quadratic control Year: (2019)
Ref_id:b30 Title: Preconditioning state-space models for stable long-horizon forecasting Year: (2024)
Ref_id:b31 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: (2022)
Ref_id:b32 Title: Improper learning for non-stochastic control Year: (2020)
Ref_id:b33 Title: Learned optimizers that scale and generalize Year: (2017)
Ref_id:b34 Title: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting Year: (2021)
Ref_id:b35 Title: Are transformers effective for time series forecasting Year: (2023)
Ref_id:b36 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
Ref_id:b37 Title: Robust and Optimal Control Year: (1996)
