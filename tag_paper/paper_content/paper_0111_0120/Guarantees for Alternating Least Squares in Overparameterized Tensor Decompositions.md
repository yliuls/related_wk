Title: Guarantees for Alternating Least Squares in Overparameterized Tensor Decomposition
Abstract: Tensor decomposition is a canonical non-convex optimization problem that is computationally challenging, and yet important due to applications in factor analysis and parameter estimation of latent variable models. In practice, scalable iterative methods, particularly Alternating Least Squares (ALS), remain the workhorse for tensor decomposition despite the lack of global convergence guarantees. A popular approach to tackle challenging non-convex optimization problems is overparameterization-on input an n × n × n tensor of rank r, the algorithm can output a decomposition of potentially rank k (potentially larger than r). On the theoretical side, overparameterization for iterative methods is challenging to reason about and requires new techniques. The work of Wang et al., (NeurIPS 2020) makes progress by showing that a variant of gradient descent globally converges when overparameterized to k = O(r 7.5 log n). Our main result shows that overparameterization provably enables global convergence of ALS: on input a third order n × n × n tensor with a decomposition of rank r ≪ n, ALS overparameterized with rank k = O(r 2 ) achieves global convergence with high probability under random initialization. Moreover our analysis also gives guarantees for the more general low-rank approximation problem. The analysis introduces new techniques for understanding iterative methods in the overparameterized regime based on new matrix anticoncentration arguments.

Section: Introduction
Iterative heuristics like alternating least squares (ALS), alternate minimization, and gradient descent are the workhorse for many computational tasks in machine learning and high-dimensional data analysis. Their simplicity, scalability and empirical success have led to their widespread use, even for highly non-convex problems. Yet rigorous guarantees have been hard to establish due to the non-convex nature of these problems.
Tensor decomposition is a prime example of a well-studied non-convex problem where there is a disconnect between the practical performance of iterative heuristics and known theoretical guarantees. We are given a third-order tensor T ∈ R n×n×n , and the goal is to decompose the tensor as a sum of 39th Conference on Neural Information Processing Systems (NeurIPS 2025). a few rank-1 tensors when possible, i.e.,
T = k i=1 x i ⊗ y i ⊗ z i ,
where the {x i ⊗ y i ⊗ z i : i ∈ [k]} are the rank-1 terms of the decomposition, and the vectors {x i , y i , z i : i ∈ [k]} ∈ R n are called the factors of the decomposition. This is sometimes called the CP decomposition of the tensor, and is an important tool in factor analysis and parameter estimation of many latent variable models in machine learning [see e.g., KB09,Moi18,JGKA19]. Finding the smallest r for which a rank-r decomposition of T exists is NP-hard in the worst case [Hås90,HL13] . Algorithmic guarantees are known under additional genericity or smoothed analysis assumptions for more sophisticated but less scalable algorithms like simultaneous diagonalization and other spectral methods [Har72,LRA93], algebraic methods [DLCC07] and the sum-of-squares hierarchy [BKS15] (see [Vij20] for more detailed comparisons).
The most popular algorithms in practice are iterative methods for the optimization problem given by the least squares objective min {(xi,yi,zi):i∈[k]} T -k i=1
x i ⊗ y i ⊗ z i 2 F .
(1)
In particular, the Alternating Least Squares (ALS) algorithm is an iterative algorithm that alternately updates one set of variables, say x 1 , . . . , x k ∈ R n , while keeping the rest of them (y 1 , . . . , y k and z 1 , . . . , z k ) fixed. Note that each update step is a least squares problem (e.g., in the variables x 1 , . . . , x k ) when the remaining variables are fixed (see Section 2 for a more detailed description of the algorithm). The optimization landscape of (1) is highly non-convex, and such iterative algorithms can potentially converge to bad local optima. This has even inspired new variants of ALS like the Orthogonalized ALS algorithm of [SV17], that come with rigorous guarantees under strong assumptions like orthogonality or incoherence of the factors. Yet, the ALS algorithm remains the most popular method for tensor decompositions in practice, despite our poor understanding of when ALS succeeds [KB09,BK25].
Overparameterization has recently emerged as a powerful approach to mitigating non-convexity. Introducing more parameters than those of the ground-truth model often improves optimization dynamics in practice, even in complex settings like training deep neural networks. In our setting, the given tensor T has a rank r decomposition of the form T = r i=1 a i ⊗ b i ⊗ c i , and the goal is to find a decomposition of potentially larger rank k.
It is challenging to reason about overparameterization with iterative methods for tensor decomposition. Existing approaches based on lazy training and standard mean-field analysis requires an overparameterization of rank that depends polynomially or even exponentially on the ambient dimension. Surprisingly, the work of [WWL + 20] makes progress by showing global convergence for a variant of gradient descent with moderate overparameterization of k = O(r 7.5 log n) that is nearly independent of the ambient dimension n. The main question we address in this paper is: Question. Does ALS admit a polynomial time global convergence guarantee with moderate overparameterization (a function of r and not n)?
Our main result answers this question in the affirmative. Concretely, our main contributions are the following:
• We prove that the ALS algorithm on a tensor with a mildly conditioned rank-r decomposition 1 when overparameterized with k = O(r 2 ) and with random initialization of {(x i , y i , z i ) : i ∈ [r]}, converges with high probability to a global minimum (i.e., objective value 0).
• We also provide rigorous guarantees for ALS under overparameterization for the more general low-rank approximation problem OPT r = min {(xi,yi,zi):i∈[r]} T -r j=1
x i ⊗ y i ⊗ z i 2 F .
(2) 1 By midly conditioned we mean that the condition numbers of factor matrices are bounded by poly(r)
We prove that ALS when overparameterized with k = O(rfoot_0 ) and initialized randomly, finds a solution whose objective value is competitive with the OPT r up to a multiplicative factor that is polynomial in only r (and independent of n).
For both our results, a moderate overparameterization of k = O(r 2 ) suffices. We suspect it may be challenging to improve the amount of overparameterization necessary. We remark that even for more computationally-intensive algorithms based on spectral methods, the best polynomial time guarantees require an overparameterization of k = O(r 2 ) [BCV14,SWZ19]. We leave it as a direction for future work to investigate whether one can prove better upper or lower bounds on the amount of overparameterization necessary to recover a provable guarantee for ALS.
Recent work has developed a few different techniques for analyzing iterative methods with overparameterization. Techniques based on the lazy training approach argue that when the model is sufficiently overparameterized, the optimization problem is locally convex and the method will converge to a good solution near the initialization [COB19]. Lazy training analyses incur overparameterization that is polynomial in the ambient dimension n, which can be very large compared to the rank r.
The work of [WWL + 20] makes progress by instead adopting the framework of mean field analysis.
While previous work that introduced this technique analyzes problems in the case of very large, or even infinite overparameterization [MMN18], [WWL + 20] was able to show global convergence for a variant of gradient descent with only moderate overparameterization of k = O(r 7.5 log n) 2 , achieving an exponentially better dependence on n than lazy training analyses for small r.
Our work develops an analysis that is significantly different from previous approaches, based on new matrix anticoncentration statements. At a high level, we observe that if the iterates X, Y, Z are sufficiently random and in the span of the components of the true tensor, then due to overparameterization they will form a basis for an appropriate space related to the components. If this occurs, then the next iteration of ALS will find a near-exact solution and converge. At initialization, X, Y, Z are independently fully random. However, they are random in the ambient space, and not restricted to the subspace of interest. The first iteration of ALS should in fact update X, Y, Z to be within the correct space. However, the updated X, Y, Z now exhibit significant dependencies on each other. Our analysis shows that despite this, the updated X, Y, Z are still random enough to form the appropriate basis. Thus, the crux of our argument is in showing that this iteration preserves enough randomness from the initialization. Quantitatively, this requires arguing about the least singular value of various structured random matrices that arise in the algorithm, along with careful matrix perturbation analyses. Our techniques are applied to a version of ALS that updates all the factor matrices in parallel. Similar guarantees, based on these techniques, for the standard ALS will appear in the arxiv version of the paper.
this section cite: ['b21', 'b29', 'b18', 'b14', 'b16', 'b13', 'b26', 'b11', 'b7', 'b35', 'b31', 'b21', 'b6', 'b3', 'b32', 'b9', 'b28']

Section: Related Work
We now describe related work on tensor decompositions and overparameterization, and place our work in the context of these prior works.
Tensor decomposition has a rich history going back to at least [Har70,Har72]. This decomposition into a sum of rank-one tensors is also referred to as CP decomposition or PARAFAC decomposition. See also [KB09] for other decomposition notions for tensors including Tucker decompositions. There are several iterative algorithms that are popular in practice like alternating least squares, alternate minimization, gradient descent and tensor power method [see KB09, AGH + 14, JGKA19, for more details]. In particular, the ALS algorithm was first introduced by [Har70,CC70], and has been the workhorse algorithm for tensor decomposition in practice [BK25]. While ALS is popular for its efficiency and steps towards understanding convergence of the iterates have been made [Usc12, WC14], we do not have a good understanding of when it converges to a global optimum solution.
The two results on tensor decompositions that is most relevant to our work are the work of Sharan and Valiant [SV17] who introduced and gave guarantees for an orthogonalized version of ALS, and the work of Wang, Wu, Lee, Ma and Ge [WWL + 20] on analyzing gradient descent in the overparameterized regime.
this section cite: ['b12', 'b13', 'b21', 'b12', 'b8', 'b6', 'b31']

Section: Comparison to the Orthogonalized ALS algorithm.
The work of Sharan and Valiant [SV17] introduced a variant of the ALS algorithm that orthogonalizes the factors in each step, in addition to the ALS update. As described in [SV17] this allows the ALS algorithm to avoid issues where multiple components of the decomposition capture the same factor of the tensor, when the rank-1 terms have different magnitudes. Their work also proves guarantees under the assumption that the factors {a i : i ∈ [r]} of the decomposition are orthogonal or incoherent. The decomposition computed by the algorithm is not overparameterized i.e., k = r. However, the algorithm is more suited for settings where the target decomposition has near-orthogonal factors. Our analysis is for the ALS algorithm in the overparameterized regime. We do not make incoherence or orthogonality assumptions on the factors; we just need mild conditioning of the factor matrix (condition number that is polynomial in r). One can interpret our results as proving that in the overparameterized setting, ALS does not face some fo the earlier issues pointed out in [SV17].
Overparameterized tensor decompositions using iterative algorithms. The work of [WWL + 20] analyzed a variant of gradient descent in the overparameterized regime of tensor decompositions. Their techniques were able to go beyond lazy-training analyses, and the standard mean-field analysis bounds that require overparameterization of polynomial or even exponential in the ambient dimension n. They were surprisingly able to provide guarantees for overparameterized rank k that is almost independent of the ambient dimension. Concretely, their guarantees hold for third-order tensors when the overparameterization is k = O(r 7.5 log n); for general order-ℓ tensors they need k = O(r 2.5ℓ log n). In this work, we instead analyze ALS in the overparameterized regime. We can get guarantees for smaller overparameterized rank k = O(r 2 ). Moreover, we also approximation guarantees for overparameterized ALS even when the tensor is not exactly of rank r. We get guarantees for the more general low-rank approximation problem that incurs a loss that is within a multiplicative factor (depending polynomially only on r, and independent of k) of the optimum value.
To the best of our knowledge, we are unaware of any such guarantees for gradient descent and other iterative algorithms.
this section cite: ['b31', 'b31', 'b31']

Section: Analysis of other iterative algorithms for tensor decompositions.
There are several other works that try to provide guarantees for iterative methods including alternating minimization, the tensor power method, and other gradient descent based algorithms. The work of [AGH + 14] analyzes the tensor power method, and provides guarantees that are specialized to the setting when the factors are orthogonal or near orthogonal [AGH + 14, AGJ14, AGJ17]. The works of [JO14,JGKA19] also analyze a variant of the alternating minimization algorithm, and provides convergence guarantees under nearby initialization. These works are not in the overparameterized regime (k = r). They find components one at a time but either require stronger assumptions, or provide local convergence guarantees. Finally, it is known that for certain matrix factorization problems and special settings of tensor decomposition (e.g., orthogonal factors), the non-convex optimization landscape is benign i.e., it does not have any local optima that are not globally optimal [Ma21]. However, the general tensor decomposition is highly non-convex with bad local minima as shown in [WWL + 20].
Other tensor decomposition algorithms. Theoretical guarantees have been established for the simultaneuous diagonalization algorithm [Har72, LRA93, Moi18] and its variants, algebraic methods [DLCC07, JLV23, Koi24, KMW24], sum-of-squares algorithms [BKS15,HSSS16,MSS16]. In the overparameterized setting, spectral methods and algorithms based on subspace embeddings can also find decompositions of rank k = r 2 in polynomial time even for the more general low-rank approximation problem with an error that is constant factor competitive with the best rank-r decomposition [SWZ19,BCV14]. The focus of our paper is to prove rigorous guarantees for the ALS algorithm, which is the most popular algorithm in practice.
this section cite: ['b20', 'b18', 'b27', 'b7', 'b17', 'b30', 'b32', 'b3']

Section: Algorithm, Results, and Preliminaries
The Alternating Least Squares (ALS) algorithm for tensor decomposition has many variants. The version that we analyze is given in Algorithm 1 3 . Given a tensor T , the algorithm randomly initializes the three modes X, Y, Z ∈ R n×k of a decomposition, corresponding to a model tensor
T = k i=1 x i ⊗ y i ⊗ z i .
On each iteration, the algorithm updates each mode individually in parallel to minimize the least squares objective: min ∥T -T ∥ 2 F . Since T is multilinear in X, Y, Z, this is a least squares problem with respect to each mode. The least squares problem could have multiple optima, in fact due to the overparameterization, we expect this to be the case. Typically, ALS is implemented using a linear system solver for each of these subproblems [BK25]; this is also what we will analyze. The updates (Algorithm 1, Lines 8, 12, 16) hence correspond to
X (t+1) = flatten(T, mode X, modes Y ⊗ Z)(Y (t) ⊙ Z (t) ) ⊤ † , Y (t+1) = flatten(T, mode Y, modes X ⊗ Z)(X (t) ⊙ Z (t) ) ⊤ † , Z (t+1) = flatten(T, mode Z, modes X ⊗ Y )(X (t) ⊙ Y (t) ) ⊤ † .
Here we use the shorthand flatten(T, mode A, modes B ⊗ C) to mean that the order-3 tensor T is reshaped into a matrix, by taking each n × n slice in the B, C modes and vectorizing it into an n 2 -dimensional row of the flattened matrix. There will be n such rows corresponding to mode A.
(This is explained in more detail in Section 2.1.) Also, M † refers to the Moore-Penrose pseudoinverse of the matrix M .
this section cite: ['b6']

Section: Algorithm 1 Alternating Least Squares (ALS) for order-3 tensor decomposition
Require: Tensor T ∈ R n×n×n , rank r of T , error tolerance ε 1: k ← Θ(r 2 ) // rank of overparameterized model 2: X (0) ∈ R n×k ← N (0, 1) n×k , Y (0) ∈ R n×k ← N (0, 1) n×k , Z (0) ∈ R n×k ← N (0, 1) n×k 3: // randomly initialize model 4: t ← 0 5: while true do 6: // X, Y, Z updates can be evaluated in parallel 7:
// X update 8: err X , X (t+1) ← min X∈R n×k and arg min X∈R n×k of T - k i=1 X i ⊗ Y (t) i ⊗ Z (t) i 2 F 9: if err X ≤ ε then 10: return X (t+1) , Y (t) , Z (t) 11: // Y update 12: err Y , Y (t+1) ← min Y ∈R n×k and arg min Y ∈R n×k of T - k i=1 X (t) i ⊗ Y i ⊗ Z (t) i 2 F 13: if err Y ≤ ε then 14: return X (t) , Y (t+1) , Z (t) 15: // Z update 16: err Z , Z (t+1) ← min Z∈R n×k and arg min Z∈R n×k of T - k i=1 X (t) i ⊗ Y (t) i ⊗ Z i 2 F 17: if err Z ≤ ε then 18: return X (t) , Y (t) , Z (t+1) 19: t ← t + 1
We give the following guarantee for Algorithm 1. In what follows, we will assume that ALS uses a sub-routine for solving the linear system in polynomial time; concretely, it computes the pseudo-inverse solution up to arbitrary precision ε > 0 in Frobenius norm in time polynomial in n, log(1/ε).
Theorem 2.1 (Guarantee for overparameterized ALS). For any constant c 0 > 0, there exists constants c = c(c 0 ) ≥ 1 and γ 0 ∈ (0, 1), such that the following holds. Let A, B, C ∈ R n×r be the factor matrices of the decomposition of a rank-r tensor T ,
T = r i=1 a i ⊗ b i ⊗ c i ,
and suppose the condition numbers κ(A), κ(B), κ(C) ≤ r c0 . Then, given T , an error parameter ε, and a k ∈ N satisfying cr 2 ≤ k ≤ n γ0 , with probability at least 1 -o(1), Algorithm 1 runs in polynomial time and in O(1) steps finds a rank-k decomposition X, Y, Z ∈ R n×k of T . That is, X, Y, Z satisfy
∥T - k i=1 x i ⊗ y i ⊗ z i ∥ 2 F ≤ ε.
The above theorem shows that ALS succeeds from random initialization with overparameterized rank k = O(r 2 ). For the theorem, we analyze standard Gaussian initializiation, the scale of the random initialization does not matter much. For the above theorem, we assume that the factor matrices A, B and C have condition numbers upper bounded by some large polynomial in r. This assumption on the condition numbers is quite mild: for example, it is satisfied w.h.p. for a natural smoothed analysis model. 4 It is weaker than incoherence or orthogonality assumptions, as the vectors in our setting can be quite correlated. Moreover, we believe the assumption to be an artifact of our analysis, and may not be necessary.
Finally, our analysis also implies approximation guarantees for overparameterized ALS with k = O(r 2 ) under random initialization, in the more general low-rank approximation problem, where
T = r i=1 a i ⊗ b i ⊗ c i + E,
where ∥E∥ F is the error. Theorem 2.2 (Low-rank tensor approximation using overparameterized ALS). For any constant c 0 > 0, there exists constants c = c(c 0 ) ≥ 1 and γ 0 ∈ (0, 1), such that the following holds. Let A, B, C ∈ R n×r be the decomposition of a rank-r tensor T ,
T = r i=1 a i ⊗ b i ⊗ c i + E,
and suppose the condition numbers κ(A), κ(B), κ(C) ≤ r c0 . Then, given T , r, and an error parameter ε, for cr 2 ≤ k ≤ n γ0 , with probability at least 1 -o(1), Algorithm 1 runs in polynomial time and in O(1) steps finds a rank-k decomposition X, Y, Z ∈ R n×k of T . That is, X, Y, Z satisfy
T - k i=1 x i ⊗ y i ⊗ z i 2 F ≤ poly(r)∥E∥ F + ε.
The above theorem gives an ALS guarantee under overparameterization for the optimization problem in (2) in the general setting when OP T r > 0, and generalizes Theorem 2.1 (special case when OP T r = 0). The multiplicative factor loss in the objective compared to OP T r is polynomial in r and independent of the ambient dimension n. To the best of our knowledge, such an approximation guarantee was not known previously for ALS or other iterative algorithms like gradient descent.
this section cite: []

Section: Notation and Preliminaries
We now introduce some notation and preliminaries that will be used in the rest of the paper. We refer to a tensor T by its decomposition into factor matrices. That is, we associate T with the matrices X, Y, Z ∈ R n×r , where r is the rank of T and
T = r i=1 x i ⊗ y i ⊗ z i ,
where x i , y i , z i refer to columns i of X, Y, Z respectively, and ⊗ when applied to vectors is the outer product. That is, each component x i ⊗ y i ⊗ z i is an n × n × n tensor. Since each component is an outer product of 3 vectors, this is an order-3 tensor. Each direction of the outer product is referred to as a mode, and we will sometimes refer to the modes of the tensor by the corresponding factor matrix, i.e., the X mode, Y mode, and Z mode. (These are the analogues of the rows and columns of a matrix/order-2 tensor.) The squared Frobenius norm of a tensor T is the sum of the squares of its entries.
It will also be useful to interact with flattened forms of the tensors we analyze. We use ⊙ to refer to the Khatri-Rao product of two matrices Y, Z ∈ R n×r , which has columns given by
(Y ⊙ Z) i = vec(y i ⊗ z i ),
where vec(•) reformats an n × n matrix as an n 2 -dimensional vector. This is useful to flatten tensors into matrices. In particular, we have
∥ r i=1 x i ⊗ y i ⊗ z i ∥ 2 F = ∥ r i=1 x i ⊗ vec(y i ⊗ z i )∥ 2 F = X(Y ⊙ Z) ⊤ 2 F .
This reshaping into a matrix is exactly what arises in the least squares problem in Algorithm 1 (Lines 8, 12, 16). In Section 2 we describe the flattening operation for a tensor T with factor matrices X, Y, Z. The flattening is just a reformatting of the entries of T , so computing it does not require explicit access to the factor matrices X, Y, Z. However, it is indeed the case that flatten(T, mode X, modes Y ⊗ Z) = X(Y ⊙ Z) ⊤ , which will be useful for our analysis.
Another useful tensor product on matrices is the Kronecker product which we refer to as ⊗. 5 The Kronecker product of two matrices X ∈ R n×r , Y ∈ R m×k is an nm × rk matrix that satisfies
(X ⊗ Y )vec(a ⊗ b) = vec(Xa ⊗ Y b), ∀a ∈ R r , b ∈ R k .
(The entries can also be written explicitly as
(X ⊗ Y ) n(i1-1)+i2,k(j1-1)+j2 = X i1,j1 Y i2,j2 .
) Since the columns of a Khatri-Rao product are flattenings of rank-1 matrices, this gives the following identity. For A ∈ R n×r , B ∈ R m×k , and X ∈ R r×ℓ , Y ∈ R k×q , we have
(A ⊗ B)(X ⊙ Y ) = (AX ⊙ BY ).(3)
We will use the Moore-Penrose pseudoinverse M † ∈ R m×n of a matrix M ∈ R n×m . This is defined as a matrix such that
∀x ⊥ Null(M ), M x = y ⇒ M † y = x, ∀y / ∈ Im(M ), M † y = 0.
where the nullspace of M , Null(M ), is the subspace of vectors mapped to 0 by M : ⟨{x | M x = 0}⟩, and the image of M , Im(M ) is the subspace of vectors that can be realized by the linear transformation M : ⟨{y | ∃x, M x = y}⟩. Here and elsewhere we use ⟨•⟩ to denote the linear span. There are also a few direct expressions for the pseudoinverse that will be useful to us. For M ∈ R n×r , where r ≤ n, if M is rank r (M is full column rank), we have
M † = (M ⊤ M ) -1 M ⊤ .
For M ∈ R r×n , where r ≤ n, if M is rank r (M is full row rank), we have
M † = M ⊤ (M M ⊤ ) -1 .
this section cite: []

Section: Analysis of the Algorithm
Fix a tensor T = r i=1 a i ⊗ b i ⊗ c i for some rank-r decomposition A, B, C ∈ R n×r . Consider iteration t of ALS on T . Without loss of generality, we focus on the update to mode Z (Algorithm 1, Line 8). ALS will converge on this step if X (t) , Y (t) , Z (t+1) fits T , i.e., the error between the model tensor and the true tensor is below ε. For the purposes of the overview, we will ignore the ε error term, and refer to this as perfectly fitting the tensor.
The least-squares problem for Z at time t + 1 is to reconstruct the slices of T as linear combinations of the
X (t) i ⊗ Y (t)
i . That is, slice j of T is given by T :,:,j = r i=1 C i,j (a i ⊗ b i ).
The least-squares problem along mode Z is then
min Z T - k i=1 x (t) i ⊗ y (t) i ⊗ z i 2 F = n j=1 min Z:,j T :,:,j - r i=1 Z i,j x (t) i ⊗ y (t) i 2 F
, which is n independent least-squares problems, one for each slice of T or row of Z. Thus, ALS will fit T and converge if, for every slice j, T :,:,j is realizable as a linear combination of the {x
(t) i ⊗ y (t) i : i ∈ [k]}. Since every slice j of T is a linear combination of {a i ⊗ b i : i ∈ [r]}, a sufficient condition for convergence is that ⟨{a i ⊗ b i : i ∈ [r]}⟩ ⊆ x (t) i ⊗ y (t) i : i ∈ [k] i.e., colspan(A ⊙ B) ⊆ colspan(X (t) ⊙ Y (t) ).(4)
The two lines are reshapings of the same statement. Now suppose for a moment that the columns of Y (t) and X (t) were each drawn independently and randomly from colspan(A) and colspan(B) respectively (for example, from standard Gaussians over the r-dimensional spaces). Then we would have that since k ∈ Ω(r 2 ), with high probability
colspan(A ⊗ B) ⊆ colspan(X (t) ⊙ Y (t) ),(5)
where A ⊗ B denotes the Kronecker product of A with itself. Since colspan(A ⊙ B) ⊆ colspan(A ⊗ B), because the Khatri-Rao product is a subset of the columns of the Kronecker product, (5) is sufficient to ensure convergence.
Of course, the columns of X and Y are not initialized randomly in the span of A and B, instead they are random in the whole n-dimensional space. This means that at initialization, each column of X (Y ) can be thought of as the sum of a random vector in the span of A (B), and a component orthogonal to A (B). Components orthogonal to the span of A (B) only make the Frobenius error of the decomposition higher, since they contribute terms that are orthogonal to T . That is, denote X = X + X ⊥ , where the columns of X are in the column span of A, and the columns of X ⊥ are orthogonal to the column span of A.
Then
T - k i=1 x i ⊗ y i ⊗ z i 2 F = T - k i=1 x i ⊗ y i ⊗ z i 2 F + k i=1 x ⊥ i ⊗ y i ⊗ z i 2 F .
The first step of ALS (Algorithm 1, Line 8) will set X so that the second term is 0, which since Y and Z are randomly initialized means setting X ⊥ = 0. If the first step of ALS only set X ⊥ , Y ⊥ , Z ⊥ = 0 and did not modify X, Y , Z, then on the second step (4) would hold, and ALS would converge. This is however not the case as ALS updates X as the minimizer of the least squares objective. Thus
X (1) = X (1) no longer has independent random columns. Instead it is a function of X (0) , Y (0) , Z (0) ,
and A, B, C. Our main technical insight is that, despite X (1) , Y (1) , Z (1) having this complex dependence on each other and A, B and C, (4) holds with high probability. It is straightforward to show that after the first iteration, each of the factor matrices will be in the span of A, B and C respectively, which means that X (1) ⊙ Y (1) will be in the span of A ⊗ B. Thus, proving that
X (1) ⊙ Y (1) has rank r 2 implies that colspan X (1) ⊙ Y (1) = colspan (A ⊗ B), then condition (4) follows since colspan(A ⊙ B) ⊆ colspan(A ⊗ B)
. This is captured by Theorem 4.1, which is the main technical component of our proof, and the focus of the next section.
this section cite: []

Section: Least Singular Value Bound through Anti-Concentration
For the proofs, we will refer to X (1) , Y (1) , Z (1) as X, Ŷ , Ẑ. In this section we give an overview of the proof of our claim that X ⊙ Ŷ spans A ⊗ A when X, Y and Z are initialized randomly with their entries being i.i.d. standard Gaussians and, in particular, show that this statement is true in a robust sense. Formal statements of this claim as well as detailed proofs can be found in Appendix C. We give an inverse polynomial in n lower bound on the least singular value of the matrix. Our main technical contribution is proving the following theorem. Theorem 4.1. Under the assumptions of Theorem 2.1 with probability at least 1 -o(1) we have that:
σ r 2 ( X ⊙ Ŷ ) ≥ 1 n 4 poly(k)
Assuming that A and B are mildly conditioned we can in fact turn our attention to showing least singular value bounds for the following matrix:
(B ⊙ C) ⊤ (Y ⊙ Z) † ⊤ ⊙ (A ⊙ C) ⊤ (X ⊙ Z) † ⊤ ∈ R r 2 ×k
One main challenge is that the entries of the matrices (Y ⊙ Z) † and (X ⊙ Z)
† are random but highly dependent. While there have been powerful techniques developed recently for proving least singular value bounds of matrices with polynomial entries [BESV24], the random matrix in our setting does not exhibit such structure and is therefore difficult to reason about directly.
We proceed by first arguing about the matrix pseudoinverse by showing in Lemma C.1 that with probability at least 1 -o(1),
(Y ⊙ Z) † ⊤ = 1 n 2 (Y ⊙ Z)(I + E 1 ),
where
∥E 1 ∥ ≤ O log(k) n + k log(k) n
(and similarly for (X ⊙ Z) † ⊤ ).
The high level intuition is that if the columns of the matrix that we are taking the pseudoinverse of were Gaussian vectors then they would be mostly orthogonal, meaning that the pseudoinverse would be close to the transpose of the matrix. Our analysis shows that the same intuition translates to matrices whose columns are tensor products of Gaussian vectors.
Using that, it suffices, up to poly(n) factors, to analyze the least singular value of the matrix:
L = (B ⊙ C) ⊤ (Y ⊙ Z)(I + E 1 ) ⊙ (A ⊙ C) ⊤ (X ⊙ Z)(I + E 2 )
Furthermore, let:
L = (B ⊙ C) ⊤ (Y ⊙ Z) ⊙ (A ⊙ C) ⊤ (X ⊙ Z)
Assume for now that we have a guarantee stating that σ r 2 ( L) ≥ 1 poly(k) . As we will discuss later, this is challenging, and much of our technical work is devoted to proving this. We can now use this to prove the existence of a matrix M such that:
LM = (B ⊙ C) ⊤ (Y ⊙ Z) ⊗ (A ⊙ C) ⊤ (X ⊙ Z)
Matrix M expresses the columns of the Kronecker product of matrices (B ⊙ C) ⊤ (Y ⊙ Z) and (A ⊙ C) ⊤ (X ⊙ Z) as linear combinations of the columns of their Khatri-Rao product. Such a matrix is guaranteed to exist only because we have assumed that L spans R r 2 . Using (3) we can express L as:
L = (B ⊙ C) ⊤ (Y ⊙ Z)(I + E 1 ) ⊙ (A ⊙ C) ⊤ (X ⊙ Z)(I + E 2 ) = (B ⊙ C) ⊤ (Y ⊙ Z) ⊗ (A ⊙ C) ⊤ (X ⊙ Z) (I ⊙ I + E)
where
E = I ⊙ E 2 + E 1 ⊙ I + E 1 ⊙ E 2 ; further by Lemma C.8, ∥E∥ ≤ O log(k) n + log(k)k n .
We can now leverage the existence of matrix M to get that:
L = L(I + M E)
where we have crucially used that, by definition of M , M (I ⊙ I) = I. In Lemma C.3, we use that σ r 2 ( L) ≥ 1/poly(k) to prove that the spectral norm of M is also bounded by a polynomial in k.
Hence
∥M E∥ ≤ ∥M ∥ • ∥E∥ ≤ poly(k) n = o(1),
when n is a sufficiently large compared to k. We now have that:
σ r 2 (L) = σ r 2 L(I + M E) ≥ σ r 2 ( L) • σ min (I + M E) ≥ (1 + o(1)) poly(k)
thus establishing the main least singular value claim.
In the above proof overview, we assumed a lower bound on the least singular value of L. The proof of this claim is quite technical and involves a careful net argument along with anti-concentration of low-degree polynomials of independent random variables [CW01]. We first express the columns of the matrix in a convenient way that factors the dependency of having Z on both sides of the Khatri-Rao product. We then argue by applying an ε-net argument and showing that for every fixed vector in R r 2 , the probability that the inner products between the fixed vector and all the columns of L is negligible is exponentially small in k. The formal statement with a detailed proof of it can be found in Lemma C.2.
this section cite: ['b4', 'b10']

Section: Conclusion
Our work proves rigorous polynomial-time global convergence guarantees of the popular ALS method for tensor decomposition with moderate overparameterization of O(r 2 ). It has been challenging to establish rigorous guarantees for iterative heuristics that are the state-of-the-art in practice. Our analysis is based on new matrix anticoncentration techniques to argue about the iterates, that differs significantly from previous approaches. Our theoretical results on overparameterization are also supported by empirical evaluations in Appendix D. It would be compelling to use these techniques to analyze gradient descent, or prove global convergence guarantees for other non-convex optimization heuristics.
this section cite: []

Section: References
Ref_id:b0 Title: Tensor decompositions for learning latent variable models Year: (2014)
Ref_id:b1 Title: Guaranteed non-orthogonal tensor decomposition via alternating rank-1 updates Year: (2014)
Ref_id:b2 Title: Analyzing tensor power method dynamics in overcomplete regime Year: (2017)
Ref_id:b3 Title: Uniqueness of tensor decompositions with applications to polynomial identifiability Year: (2014)
Ref_id:b4 Title: New tools for smoothed analysis: Least singular value bounds for random matrices with dependent entries Year: (2024)
Ref_id:b5 Title:  Year: (1997)
Ref_id:b6 Title: Tensor Decompositions for Data Science Year: (2025)
Ref_id:b7 Title: Dictionary learning and tensor decomposition via the sum-of-squares method Year: (2015)
Ref_id:b8 Title: Analysis of individual differences in multidimensional scaling via an n-way generalization of "eckart-young" decomposition Year: (1970)
Ref_id:b9 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b10 Title: Distributional and L q norm inequalities for polynomials over convex bodies in R n Year: (2001)
Ref_id:b11 Title: Fourth-order cumulant-based blind identification of underdetermined mixtures Year: (2007)
Ref_id:b12 Title: Foundations of the parafac procedure: Models and conditions for an "explanatory" multimodal factor analysis. UCLA Working Papers in Phonetics Year: (1970)
Ref_id:b13 Title: Determination and proof of minimum uniqueness conditions for PARAFAC1 Year: (1972)
Ref_id:b14 Title: Tensor rank is np-complete Year: (1990)
Ref_id:b15 Title: Matrix analysis Year: (2012)
Ref_id:b16 Title: Most tensor problems are np-hard Year: (2013)
Ref_id:b17 Title: Fast spectral algorithms from sum-of-squares proofs: Tensor decomposition and planted sparse vectors Year: (2016)
Ref_id:b18 Title: Spectral learning on matrices and tensors Year: (2019)
Ref_id:b19 Title: Computing linear sections of varieties: quantum entanglement, tensor decompositions and beyond Year: (2023-11)
Ref_id:b20 Title: Provable tensor factorization with missing data Year: (2014)
Ref_id:b21 Title: Tensor decompositions and applications Year: (2009)
Ref_id:b22 Title: Overcomplete tensor decomposition via koszul-young flattenings Year: (2024)
Ref_id:b23 Title: On the uniqueness and computation of commuting extensions Year: (2024)
Ref_id:b24 Title: Tensorly: Tensor learning in python Year: (2019)
Ref_id:b25 Title: An elementary proof of anti-concentration of polynomials in gaussian variables Year: (2010)
Ref_id:b26 Title: A decomposition for three-way arrays Year: (1993)
Ref_id:b27 Title: Why Do Local Methods Solve Nonconvex Problems? Year: (2021)
Ref_id:b28 Title: A mean field view of the landscape of two-layer neural networks Year: (2018)
Ref_id:b29 Title: Algorithmic Aspects of Machine Learning Year: (2018)
Ref_id:b30 Title: Polynomial-time tensor decompositions with sum-of-squares Year: (2016)
Ref_id:b31 Title: Orthogonalized ALS: A theoretically principled tensor decomposition algorithm for practical use Year: (2017-08)
Ref_id:b32 Title: Relative error tensor low rank approximation Year: (2019)
Ref_id:b33 Title: Local convergence of the alternating least squares algorithm for canonical tensor approximation Year: (2012)
Ref_id:b34 Title: High-dimensional probability: An introduction with applications in data science Year: (2018)
Ref_id:b35 Title: Beyond the Worst-Case Analysis of Algorithms, chapter 19 Efficient Tensor Decomposition Year: (2020)
Ref_id:b36 Title: On the global convergence of the alternating least squares method for rank-one approximation to generic tensors Year: (2014)
Ref_id:b37 Title: Beyond lazy training for over-parameterized tensor decomposition Year: (2020)
