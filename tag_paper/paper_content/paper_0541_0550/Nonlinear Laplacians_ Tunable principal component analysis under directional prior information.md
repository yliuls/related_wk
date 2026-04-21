Title: Nonlinear Laplacians: Tunable principal component analysis under directional prior information
Abstract: We introduce a new family of algorithms for detecting and estimating a rank-one signal from a noisy observation under prior information about that signal's direction, focusing on examples where the signal is known to have entries biased to be positive. Given a matrix observation Y , our algorithms construct a nonlinear Laplacian, another matrix of the form Y + diag(σ(Y 1)) for a nonlinear σ : R → R, and examine the top eigenvalue and eigenvector of this matrix. When Y is the (suitably normalized) adjacency matrix of a graph, our approach gives a class of algorithms that search for unusually dense subgraphs by computing a spectrum of the graph "deformed" by the degree profile Y 1. We study the performance of such algorithms compared to direct spectral algorithms (the case σ = 0) on models of sparse principal component analysis with biased signals, including the Gaussian planted submatrix problem. For such models, we rigorously characterize the strength of rank-one signal, as a function of the nonlinearity σ, required for an outlier eigenvalue to appear in the spectrum of a nonlinear Laplacian matrix. While identifying the σ that minimizes the required signal strength in closed form seems intractable, we explore three approaches to design σ numerically: exhaustively searching over simple classes of σ, learning σ from datasets of problem instances, and tuning σ using black-box optimization of the critical signal strength. We find both theoretically and empirically that, if σ is chosen appropriately, then nonlinear Laplacian spectral algorithms substantially outperform direct spectral algorithms, while retaining the conceptual simplicity of spectral methods compared to broader classes of computations like approximate message passing or general first order methods.

Section: Introduction
Principal component analysis (PCA) is one of the most ubiquitous computational tasks in statistics and data science, seeking to extract informative low-rank structures from noisy observations organized into matrices (see, e.g., [AW10, JC16, JP18, GGH + 22] for a few of the huge number of available surveys). We will study a family of mathematical models of such problems involving detecting or recovering these low-rank structures. These problems are specified by a family of probability measures P n,β on n × n symmetric matrices, where β is a signal-to-noise parameter. The observed matrix is biased in the direction of a low-rank signal: there is a latent unobserved unit vector x ∈ S n-1 ⊂ R n such that, when Y ∼ P n,β , then
E [Y | x] ≈ β √ n • xx ⊤ .
For this class of problems, we consider two computational tasks:
1. Detection: Determine whether the observed data is uniformly random (β = 0) or contains a signal (β > 0). In particular, we say that a sequence of functions f n : R n×n sym → {0, 1} (usually encoding a single algorithm allowing for various n) achieves strong detection if lim n→∞ P n,β (f n (Y ) = 1) = lim n→∞ P n,0 (f n (Y ) = 0) = 1.
2. Recovery: When β > 0, estimate the hidden signal x. 1 We say that a sequence of functions
x = x n : R n×n sym → S n-1 achieves weak recovery if, for some δ > 0, lim inf n→∞ P n,β (|⟨ x n (Y ), x⟩| ≥ δ) > 0, and achieves strong recovery if |⟨ x n (Y ), x⟩| → 1 in probability as n → ∞.
A common approach to such problems that we detail in Section 3.1 is to "perform PCA" on Y directly, meaning in this context to look for an unusually large eigenvalue to test whether β = 0 or β > 0, and to estimate x by the eigenvector associated to such an eigenvalue. We call this a direct spectral algorithm. This approach is effective, but is agnostic to any information we might have in advance about the hidden x. In this paper, we propose a new framework for improving direct spectral algorithms when we have some knowledge about x. In particular, our approach will be sensible when we have directional information about x, say that it lies in a given cone, a class of problem proposed by [DMR14]. Our approach is probably not well-suited to other kinds of structural prior information that other works have sought to exploit, like sparsity [ZHT06, AW08, JL09, DM14] or the opposite assumption of a perfectly flat signal from the hypercube, x ∈ {±1/ √ n} n [DAM15,FMM18].
To be concrete, we will consider the following class of models where ⟨x, 1⟩ is somewhat large with high probability.
Definition 1.1 (Sparse Biased PCA). Let p = p(n) ∈ [0, 1] satisfy ω log n n ≤ p(n) ≤ o(1).
Let η be a probability measure with positive mean (E z∼η z > 0) and finite third absolute moment (E z∼η |z| 3 < ∞). Let x = y/∥y∥ with y ∈ R n having entries y i = ε i z i , where z i i.i.d.
∼ η and ε i ∈ {0, 1} are drawn in one of the following ways:
• Random Subset: Sample S ⊆ [n] uniformly at random among subsets of size |S| = pn, and set ε i := 1{i ∈ S}.
• Independent Entries: Draw ε i i.i.d.
this section cite: ['b28', 'b23', 'b32']

Section: ∼ Ber (p).
We call this choice the sparsity model. Finally, we draw Y ∼ P n,β from this model as
Y = β √ n • xx ⊤ + W
for W a symmetric matrix drawn from the Gaussian orthogonal ensemble (GOE), i.e., having 2 We call (η, p(n)) the model parameters and β the signal strength of the model.
W ij = W ji ∼ N (0, 1 + 1{i = j}) independently.
In these models, clearly we have arranged to have E[Y | x] = β √ n • xx ⊤ exactly. Our models are in the spirit of Non-Negative PCA [MR15], where the stricter entrywise condition x i ≥ 0 is imposed. On the other hand, for technical reasons we focus on sparse x, though our algorithms seem sensible for dense x as well (see Appendix D.3).
For sparsity p(n) = o(1), it is believed that optimal algorithms for such problems have a tradeoff between runtime and performance, in the sense that one may spend more time computing and in return identify weaker signals (with a smaller value of β). In contrast, for p(n) a constant, there is a single critical β * such that a polynomial-time algorithm can identify signals of strength β > β * , while doing so for β < β * is believed to require nearly exponential time. In the sparse case see, e.g., [AKS98] for the case of the planted clique problem discussed below, or [DKWB23] for Gaussian models as above. Our goal here is to study a particular aspect of the former, more algorithmically flexible situation. Namely, we will ask: how weak of a signal can one detect without straying too far from the direct spectral algorithm? To give a concrete example, the following is one special case of our model that has been widely studied [MRZ15, MW15, HWX17, CX16, BMV + 18, BBH19, LM19, GJS21].
Example 1.2 (Gaussian Planted Submatrix). Let η := δ 1 be the probability measure concentrated on the constant 1, let p(n) := β/ √ n for the same β as the signal strength, and use the Random Subset sparsity model. The result is that Y is the Gaussian random matrix W , where a random principal submatrix of dimension β √ n has had the means of its entries elevated from 0 to 1.
While we focus on the specific case of x correlated with 1, in Appendix F.3 we discuss possible extensions to other forms of directional prior information.
this section cite: ['b60', 'b1', 'b25']

Section: Summary of contributions
In this paper, we first propose a new PCA algorithm that seeks to incorporate our prior information about x by deforming the matrix Y before computing its top eigenpair. Namely, we add to (a normalized version of) Y a diagonal matrix D with entries given by a bounded nonlinear function σ of (a normalized version of) the entries of Y 1, for 1 the all-ones vector. The idea behind these nonlinear Laplacian matrices is that, since xx ⊤ is rank-one and x is positively correlated with 1, both the spectrum of Y and the vector Y 1 carry information about x. Forming a diagonal matrix from the latter and attenuating its largest entries by applying σ lets these two sources of information cooperate, leading to a more effective spectral algorithm for detecting and estimating the signal.
We then give a complete description of the appearance of unusually large "outlier" eigenvalues in the spectrum of such matrices built from Y drawn from models as in Definition 1.1. The appearance of such outliers corresponds to when, for instance, an algorithm thresholding the largest eigenvalue can detect the presence of a signal. For a large class of σ and η, we identify the β * = β * (σ) such that there is an outlier in the σ-Laplacian if and only if β > β * (σ) (Theorem 3.3 and its subsequent discussion). This analysis applies sophisticated tools developed in prior work on random matrix theory and free probability, and gives β * (σ) via a sequence of integral equations involving σ and η. As a consequence we learn that, for instance for the concrete example of the Gaussian Planted Submatrix problem, nonlinear Laplacian algorithms considerably outperform direct spectral algorithms (Theorem 3.4 and Figure 1).
Because the description of β * (σ) is rather complex and seems unlikely to admit a closed form, we also explore several heuristics for identifying an effective σ for a given problem. We find that our algorithms are quite robust to this choice, and a good σ can equally well be found by hand, learned from data, or tuned by black-box optimization (Figure 2). Further, nonlinear Laplacian algorithms appear to be robust across the models described in Definition 1.1; a σ that we optimize for one such model is quite effective for others (Appendix F.2). Based on these findings, we argue that nonlinear Laplacian algorithms give a simple, robust, and substantial improvement over direct spectral algorithms, using a bare minimum of extra information about the input matrix beyond its spectrum.
this section cite: []

Section: Nonlinear Laplacian spectral algorithms
We will work with the following normalization of Y in the setting of Definition 1.1:
Y := Y / √ n = βxx ⊤ + W
for W := W / √ n. As we will describe, in our models this ensures that the extreme eigenvalues of Y are of constant order. We will construct algorithms for PCA using the following class of matrix.
Definition 2.1 (σ-Laplacian matrix). Given the observed matrix Y and a scalar function σ : R → R, we define the σ-Laplacian as:
L = L σ ( Y ) := Y + diag(σ( Y 1)) =:Dσ( Y )=D
where σ applies entrywise to the vector Y 1 ∈ R n .
Definition 2.2 (σ-Laplacian spectral algorithms). Given σ : R → R and τ ∈ R, the associated σ-Laplacian spectral algorithm for detection is f : R n×n sym → {0, 1} that outputs f (Y ) := 1{λ 1 (L σ ( Y )) ≥ τ }, and the associated σ-Laplacian spectral algorithm for recovery is v : R n×n sym → S n-1 that outputs v(Y ) := v 1 (L σ ( Y )). Here (λ 1 (•), v 1 (•)) denote the top eigenpair of a matrix.
As we discuss in Section 3.1, the direct spectral algorithms that previous work has focused on are special cases of the above with σ = 0.
For technical reasons (see Section 3.4) it is easier to work with the following variant of the σ-Laplacian.
Definition 2.3 (Compressed σ-Laplacian matrix). For each n ≥ 1, fix V ∈ R n×(n-1) with columns an orthonormal basis of the orthogonal complement of the span of the all-ones vector 1. Given Y ∈ R n×n sym and σ as before, define the compressed σ-Laplacian as L σ ( Y ) := V ⊤ L σ ( Y )V ∈ R (n-1)×(n-1) . And, define the compressed σ-Laplacian spectral algorithm for detection as in Definition 2.2, only with L replaced by L. For recovery, use V v 1 ( L σ ( Y )).
We expect all results given below for compressed σ-Laplacians to hold as well for the original σ-Laplacians; this change is almost certainly merely a theoretical convenience. The simple idea behind these algorithms is that, if x is biased in the 1 direction, then Y 1 = β⟨x, 1⟩x + W 1 will be somewhat correlated with x. For the models of Definition 1.1, W 1 will further be a standard Gaussian random vector (up to a negligible adjustment). In particular, if σ is monotone, then D will become larger entrywise as β increases, and will have larger diagonal entries in the coordinates where x is larger.
While it is tempting to dispense with σ entirely, we will see below that we have ∥ Y ∥ = O(1), while standard asymptotics about the maximum of independent Gaussian random variables max n i=1 |( W 1) i | = Ω( √ log n). So, we must "tame" the largest entries of D by applying σ so that it can "cooperate" with Y in determining the largest eigenvalue of L rather than dominating Y .
For further intuition about the D term, note that if Y is a normalized adjacency matrix of a graph, then the vector Y 1 contains normalized and centered degrees of each vertex in the graph. If a random graph is deformed to have a planted clique (see Appendix D.2) or an unusually dense subgraph, then this degree vector carries some information about which vertices belong to this planted structure. As we discuss in Appendix A, both spectral and degree-based algorithms have been studied before for such problems, and the σ-Laplacians describe a simple and tunable family of "hybrid" algorithms involving both kinds of information. This example is also why we call L a "Laplacian," since its definition resembles that of the graph Laplacian, the difference of the (unnormalized) diagonal degree and adjacency matrices of a graph.
Our original motivation, which we discuss in greater detail in Appendix F.4 (see also Appendix A for general discussion of related work), was to study a broad class of spectral algorithms where one performs PCA on M (Y ) for some function M : R n×n sym → R n×n sym . It is reasonable to parametrize such M (Y ) as a neural network, alternating linear maps and entrywise nonlinearities. Subject to the natural criteria of equivariance and dimension generalization of M that we explain in the Appendix, such M (Y ) are closely related to the much-studied graph neural networks, and the space of permissible linear maps to use is actually quite small, including the function Y → diag(Y 1) that appears in nonlinear Laplacians. To perform random matrix analysis at the level of detail that we do here, one must be careful to make sure that the spectrum of M (Y ) remains on a fixed scale as one applies these transformations. This can easily break down if one allows others of the available linear functions in M (Y ), such as functions with rank-one outputs like Y → 1(Y 1) ⊤ . We have found this issue to be quite delicate, so, as a first step, we have focused on nonlinear Laplacians as one special case of the above general class of flexible spectral algorithms, which do allow for sufficient control of the spectrum to study limiting empirical spectral distributions and outlier eigenvalues in detail.
this section cite: []

Section: Characterization of outlier eigenvalues
Our main results characterize when a σ-Laplacian matrix has an outlier eigenvalue. Let us first quickly recall the corresponding results for the case σ = 0.
this section cite: []

Section: Prior work: σ = 0 and direct spectral algorithms
In this case, when β = 0, Y = W is just a normalized Wigner matrix,foot_2 and it is a classical result of random matrix theory that its eigenvalues follow Wigner's semicircle law µ sc , with high probability lying in the interval [-2o(1), 2 + o(1)] (Theorem B.12 in Appendix). When β > 0, Y consists of a rank-one perturbation of a Wigner matrix. Such random matrix distributions are commonly referred to as spiked matrix models and have been studied extensively in high-dimensional statistics and random matrix theory [Joh01, BBAP05, Pau07, FP07, CDMF09, PWBM18, EAKJ20, LM19]. The bulk eigenvalues remain stable under this rank-one perturbation (Proposition B.17 in Appendix) and still obey the semicircle law. However, when the signal-to-noise ratio β exceeds a certain threshold, the rank-one perturbation induces a single outlier eigenvalue outside of the bulk. This sharp phase transition in the behavior of the largest eigenvalue of Y is known as a Baik-Ben Arous-Péché (BBP) transition, named after the work of [BBAP05]. In this setting, it takes the following form:
Theorem 3.1 ([FP07]). Consider a symmetric random matrix Y = W + β √ n • xx ⊤ ∈ R n×n
sym as above, where β > 0, x is a unit vector and W is a GOE random matrix independent of x. Then the following hold for the largest eigenvalue λ 1 ( Y ) and the corresponding unit eigenvector
• If β ≤ 1, then λ 1 ( Y ) (p) --→ 2 and |⟨v 1 ( Y ), x⟩| (p)
--→ 0 (the arrows denoting convergence in probability).
•
If β > 1, then λ 1 ( Y ) (p) --→ β + 1/β > 2 and |⟨v 1 ( Y ), x⟩| 2 (p) --→ 1 -1/β 2 > 0.
For models as in Definition 1.1, this result implies the following analysis of direct spectral algorithms:
Corollary 3.2. In a model of Sparse Biased PCA as in Definition 1.1, a direct spectral algorithm (the algorithm of Definition 2.2 with σ = 0) achieves strong detection and weak recovery if and only if β > β * (0) := 1.
this section cite: ['b4']

Section: Our contribution: σ ̸ = 0 and nonlinear Laplacian spectral algorithms
We now present our results, generalizing part of Theorem 3.1 to (compressed) nonlinear Laplacian matrices. We always make the following assumptions on the nonlinearity σ without further mention.
Assumption 1 (Properties of σ). We assume that:
1. σ is monotonically non-decreasing.
2. σ is bounded: |σ(x)| ≤ K for some K > 0 and all x ∈ R.
3. σ is ℓ-Lipschitz for some ℓ > 0.
We write edge + (σ) := sup x∈R σ(x), which is finite by the second assumption, and σ(R) for the image of σ, which is an interval (open or closed on either side) of R of finite length by the first two assumptions.
For now we give just the final result of our analysis, and describe the idea of the derivation below. Our main result is as follows:
Theorem 3.3. For a model of Sparse Biased PCA as in Definition 1.1, define
m 1 := E x∼η x > 0, m 2 := E x∼η x 2 .
Given σ, define θ = θ σ (β) to solve the equation
E y∼η g∼N (0,1) y 2 θ -σ( m1 m2 βy + g) = m 2 β
if such θ > edge + (σ) exists, and θ = edge + (σ) otherwise. The following hold almost surely for the sequence of compressed σ-Laplacians L = L (n) :
• If Eg∼N (0,1) 1 (θσ(β)-σ(g)) 2 ≥ 1, then λ 1 ( L (n) ) → edge + (µ sc ⊞ σ(N (0, 1))),
the right boundary point of the support of the probability measure µ sc ⊞ σ(N (0, 1)). Here µ sc is Wigner's semicircle law, ⊞ is the additive free convolution operation presented in Appendix B.4, and σ(N (0, 1)) is the pushforward of the standard Gaussian by σ. Moreover,
|⟨x, V v 1 ( L (n) )⟩| → 0.
• If Eg∼N (0,1)
1 (θσ(β)-σ(g)) 2 < 1, then λ 1 ( L (n) ) → θ σ (β) + E g∼N (0,1) 1 θ σ (β) -σ(g) > edge + (µ sc ⊞ σ(N (0, 1))), |⟨x, V v 1 ( L (n) )⟩| 2 → m 2 β 2   E y∼η g∼N (0,1) y 2 θ σ (β) -σ m1 m2 βy + g 2   -1 1 - E g∼N (0,1) 1 θ σ (β) -σ(g) 2 > 0.
In the special case where the entrywise condition x i ≥ 0 holds almost surely (i.e., η in Definition 1.1 is a probability measure on R ≥0 ), there is a unique
β * = β * (σ) > 0 that solves E g∼N (0,1) 1 (θ σ (β * ) -σ(g)) 2 = 1,
and the conditions of the two cases above are equivalent to β ≤ β * and β > β * , respectively. In that case, this result precisely identifies the critical signal strength β * (σ) mentioned earlier, the threshold beyond which the σ-Laplacian has an outlier eigenvalue. 4 One may also check that, setting σ = 0 and using that in this case edge + (µ sc ⊞ σ(N (0, 1))) = edge + (µ sc ) = 2, this result is indeed compatible with Theorem 3.1, giving β * (0) = 1.
this section cite: []

Section: Example: Gaussian Planted Submatrix and Planted Clique models
We demonstrate the concrete consequence of Theorem 3.3 for the model proposed in Example 1.2 above. This is conditional on the accuracy of the numerical evaluation of the Gaussian expectations appearing in the Theorem. These involve only low-dimensional function and integral evaluations and we are confident that our numerical solutions are accurate, but we mark the following result with (n)  to indicate its mild conditional nature.
this section cite: []

Section: Theorem 3.4 (Gaussian Planted Submatrix (n)
). There exist σ : R → R and τ ∈ R such that the following holds for the choices of Example 1.2 substituted into the setting of Definition 1.1. If β > 0.76 and Y ∼ P n,β , then the compressed σ-Laplacian L σ ( Y ) with high probability has a single outlier eigenvalue (see Figure 1 for an illustration and Appendix B.3 for precise definitions), and |⟨x, V v 1 ( L σ ( Y ))⟩| converges in probability to a strictly positive deterministic number. In particular, if β > 0.76, then the compressed σ-Laplacian spectral algorithm with threshold τ succeeds in strong detection and weak recovery in the Gaussian Planted Submatrix model.
Underlying this result is a choice of σ for which β * (σ) < 0.76; we discuss below in Section 4 various ways one can find σ achieving the above, and illustrate one such σ in Figure 1. There are two reasonable benchmarks with which to compare this performance. On the one hand, β * (0) = 1 is the corresponding threshold for the direct spectral algorithm, per Theorem 3.1. In words, our result says that only 76% as strong of a signal is required by a suitable nonlinear Laplacian to achieve strong detection. On the other hand, the work of [HWX17] shows that a belief propagation (BP) algorithm achieves weak recovery provided β > 1/ √ e ≈ 0.61. In this setting of dense input data, BP is likely also to behave similarly to approximate message passing (AMP), an approximation that is more efficient to compute. Thus the performance of our nonlinear Laplacian algorithm lies between that of the direct spectral algorithm and BP/AMP, while our algorithm enjoys the advantages of being conceptually simpler than BP/AMP and only making a small modification to the direct spectral algorithm. (See Appendix A for a more detailed comparison with BP/AMP.)
In the same vein, we may also better understand the individual power of the two components of any σ-Laplacian, and show that they must be combined in order to achieve the above performance: neither the eigenvalues of Y nor the values of Y 1 alone can achieve strong detection for any β < 1.
Theorem 3.5. The following hold in the Gaussian Planted Submatrix model:
1. If β < 1, then there is no function of the vector (λ 1 ( Y ), . . . , λ n ( Y )) that achieves strong detection. (This result is due to prior work of [MRZ15].) 2. For any β ≥ 0 (not depending on n), there is no function of the vector Y 1 that achieves strong detection. (This result is our contribution, which we prove in greater generality than just the Gaussian Planted Submatrix model; see Theorem C.10.)
It is maybe surprising that the information contained in Y 1, which by itself is useless for detection in this regime, is enough to "boost" the performance of a spectral algorithm substantially. The question of how effective "purely spectral" algorithms can be for (weak) recovery is raised by [HWX17] (their Section 1.3), asking whether the direct spectral algorithm's β * = 1 threshold is optimal in this regard.
Our results suggest that only a small step beyond algorithms using only the eigenvalues of Y is enough to improve on this.
Finally, we offer a more speculative extension. As we discuss in Appendix D.2, from the point of view of the random matrix theory of σ-Laplacians, the much-studied Planted Clique problem looks nearly identical to the Gaussian Planted Submatrix problem. We define this problem formally in Definition D.1, but, in words, it is given by taking Y to be a centered adjacency matrix of an Erdős-Rényi random graph with each edge present independently with probability 1/2, with a clique (complete subgraph) inserted on a random subset of β √ n vertices. Replacing the Gaussian structure with discrete structure creates technical challenges that we have not been able to surmount. We are quite confident, but leave as an open problem to show, that the above results apply directly to the Planted Clique problem.
Conjecture 3.6. The results of Theorem 3.4 hold verbatim if the Gaussian Planted Submatrix problem is replaced by the Planted Clique problem.
See Appendix D.3 for discussion of further examples and extensions.
this section cite: ['b38', 'b61', 'b38']

Section: Proof techniques
We now sketch the analysis leading to Theorem 3.3. Full proofs are given in Appendix C. Recall that we are interested Y = W + βxx ⊤ , where W is a Wigner random matrix with entrywise variance 1/n and x is a unit vector. Consequently, the matrix L can be expressed as
L = W + βxx ⊤ + diag(σ( W 1 + β⟨x, 1⟩x)) =:X ,(1)
which we interpret as a perturbation of the Wigner noise W by a matrix X.
If σ = 0, the perturbation term is simply X = βxx ⊤ , and in particular is low-rank. In that case, the bulk eigenvalue distribution of L is always the same as that of W , obeying the semicircle law. The effect of X in such models is limited to creating potential outlier eigenvalues, leaving the bulk spectrum unchanged. Our setting of σ ̸ = 0 presents a key difference, stemming from the fact that our X is (usually) full-rank, even when β = 0. Therefore, even when β = 0, the spectrum of L undergoes a non-trivial deformation from that of W , which is described by free probability theory (specifically, by the operation of additive free convolution appearing in Theorem 3.3). When β > 0, the bulk eigenvalues will resemble this same deformation, and may have a further outlier eigenvalue generated by a corresponding outlier eigenvalue of X. Such results have been obtained by [CDMFF11,Cap17,BG24], which our analysis applies.
Those results, roughly speaking, give a recipe for deducing the behavior of the eigenvalues of L from those of X; in particular, outlier eigenvalues in L arise from sufficiently extreme eigenvalues in X. So, we proceed by characterizing the eigenvalues of X. Notably, X itself resembles a spiked matrix model, although one where the "noise term" is a diagonal matrix, making the analysis different than that for conventional spiked matrix models. The eigenvalues of X are as follows:
Lemma 3.7. In the setting of Theorem 3.3, the following hold almost surely for the sequence of X = X (n) :
1. The empirical spectral distribution satisfies
1 n i δ λi(X (n) )(w)
--→ σ(N (0, 1)), where the arrow denotes weak convergence (Definition B.1 in Appendix).
this section cite: ['b19', 'b17', 'b6']

Section: The largest eigenvalue of X
(n) satisfies λ 1 (X (n) ) → θ σ (β) for the function θ σ described in Theorem 3.3. 3. All other eigenvalues λ 2 (X (n) ), . . . , λ n (X (n)
) lie in σ(R), where the bar denotes the closure.
With this understanding, the eigenvalues of L can be effectively described using the above tools, provided that we make the adjustment from Definition 2.3. The reason for this is that W is weakly dependent on X, as X depends on W 1, while standard analysis from random matrix theory assumes these signal and noise matrices to be independent. When W is drawn from the GOE, we can circumvent this issue by instead analyzing the spectrum of the compressed σ-Laplacian
L = L (n) = V ⊤ LV .
By the rotational symmetry of the GOE, the noise term of L remains a (n -1) × (n -1) GOE matrix, up to a negligible rescaling. And, this noise term has had the 1 direction "projected away," whereby, it is now independent of the projected signal term, making the model compatible with existing results.
To analyze the top eigenvector of L (n) , we use a simple trick: if we replace the term βxx ⊤ in the underlying L with (β + t)xx ⊤ for another parameter t, then one may show that ⟨x,
V v 1 ( L (n) )⟩ 2 is precisely the derivative of λ 1 ( L (n)
) with respect to t at t = 0. We argue that one may exchange this derivative with the limit n → ∞, and thus ⟨x, v 1 (L)⟩ 2 is obtained as a derivative of a closely related formula to that for lim n→∞ λ 1 ( L (n) ).
As an aside, in addition to the analysis in Theorem 3.3 of the largest eigenvalue, we obtain the following result on the empirical spectral distribution of the σ-Laplacian, which is sensible given the meaning of the additive free convolution operation (see Definition B.20 in Appendix) and explains its appearance in Theorem 3.3: Lemma 3.8. For a model of Sparse Biased PCA as in Definition 1.1, for any σ and β ≥ 0, almost surely the empirical spectral distribution satisfies
1 n i δ λi( L (n) ) (w) --→ µ sc ⊞ σ(N (0, 1)).
This statement should be read as describing the bulk eigenvalues of L; recall that weak convergence does not give any guarantees about the behavior of extreme or outlier eigenvalues, so Theorem 3.3 indeed gives additional further information.
this section cite: []

Section: Numerical optimization of nonlinearities
Let us comment briefly on how we actually find the σ and the number 0.76 in Theorem 3.4. First, note that, given σ, in principle the above results determine β * (σ), albeit via an integral equation involving an expectation over g ∼ N (0, 1). Further, that equation is only given in terms of the function θ σ (β), which itself is only given in terms of the solution of another integral equation involving an expectation over g ∼ N (0, 1) and y ∼ η. This is why we point out above that our results only determine β * assuming the fidelity of the numerical calculations of these integrals (which are, however, only at most two-dimensional and thus not computationally challenging).
The question remains of how to find σ that minimizes β * (σ). We have not been able to identify even a contrived construction of σ ̸ = 0 satisfying Assumption 1 for which we can find β * (σ) in closed form. So, we resort to heuristically identifying good σ and then estimating β * (σ) for such σ numerically. We have studied three approaches to this task, which all seem more or less equally effective:
1. Pick a simple class of σ given by a small number of parameters, such as σ(x) = a tanh(bx) for a, b ∈ R or σ(x) = min{c, max{d, ax + b}} for a, b, c, d ∈ R and optimize β * (σ) over these few parameters by manual inspection of numerical results or exhaustive grid search. 2. Fix a multi-layer perceptron (MLP) structure for σ and optimize it by training λ 1 (L σ (Y ))
to classify a training dataset of synthetic Y drawn from the null model (β = 0) and the structured alternative model (β > 0). 3. Fix a simple structure for σ such as a step functionfoot_4 over a fixed grid and directly optimize the (complicated) objective function β * (σ) via gradient-free black-box optimization methods such as the Nelder-Mead or differential evolution algorithms.
We discuss the implementation details of these choices further in Appendix D.1.2. We conclude from these explorations that σ-Laplacian algorithms are rather robust to the choice of σ-just a few degrees of freedom in σ appear to suffice to achieve optimal performance. We illustrate this in Figure 2, which gives the σ obtained by each of the above methods. On the other hand, mathematically understanding the behavior of the equations determining β * (σ) seems quite challenging, and we leave this as an interesting problem for future work.
this section cite: []

Section: Conclusion
Above, we have introduced the new class of nonlinear Laplacian spectral algorithms, given a complete analysis of their performance for the task of strong detection in Sparse Biased PCA models, demonstrated as a consequence that such algorithms substantially outperform direct spectral algorithms for the Gaussian Planted Submatrix problem, and verified those findings empirically (see also Figures 3 and 4 in the Appendix for further experimental results). Limitations We mention three limitations of our work. First, as mentioned, algorithms like BP and AMP can perform better than nonlinear Laplacian algorithms for specific problems, for example as shown for the Gaussian Planted Submatrix problem by [HWX17]. We claim that nonlinear Laplacian algorithms, however, are conceptually simpler than BP/AMP and also easier to tune: our results indicate that one can either tune them mechanically from data or merely "eyeball" a reasonable nonlinearity to use. This is far from the case for AMP, where the iteration rules (including the subtle "Onsager correction") must be chosen carefully to yield a sensible limiting behavior. Second, we have made the assumption that our models involve only additive Gaussian noise (Definition 1.1), but we believe that the same analysis should apply to more general models (see Conjecture 3.6 and Appendix D.2). Finally, we leave open several challenging technical questions, including those of analyzing the algorithm's performance on the Planted Clique problem (Conjecture 3.6) and of understanding analytically the behavior of the threshold value β * (σ) and the structure of the optimal nonlinearity σ for a given problem.
Future directions It is tempting to consider designing more complex diagonal matrices D = D( Y ) with which to augment spectral algorithms. One may, for instance, use a graph neural network to map the matrix Y to a vector (to set as the diagonal of D) in an equivariant way and optimize this over a dataset (as in our treatment of building σ with a multi-layer perceptron). See Section F.4 for more discussion of such an approach; here we only mention a few salient considerations. Firstly, if D can depend on Y more strongly than merely through Y 1, then the device of compression that we have used to decouple D from W may break down and the analysis could become more challenging; for sufficiently complex D, the entire apparatus of free probability (which plays a crucial role in the results of [CDMFF11] that we use) might no longer apply, which would leave us with random matrices requiring a fundamentally different toolkit to analyze. Also, allowing very complex D might allow one to build a complex algorithm solving the underlying statistical problem into this diagonal matrix alone, making the first term of the nonlinear Laplacian L = Y + D superfluous. As we have argued, the merit of nonlinear Laplacians is in their balance of simplicity and strong performance, and we believe it would be valuable to understand more precisely the tradeoff between these properties as one blends more and more complex side information into spectral algorithms.
this section cite: ['b38', 'b19']

Section: References
Ref_id:b0 Title: Community detection and stochastic block models: recent developments Year: (2017)
Ref_id:b1 Title: Finding a large hidden clique in a random graph Year: (1998)
Ref_id:b2 Title: High-dimensional analysis of semidefinite relaxations for sparse principal components Year: (2008)
Ref_id:b3 Title: Principal component analysis Year: (2010)
Ref_id:b4 Title: Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. The Annals of Probability Year: (2005)
Ref_id:b5 Title: Universality of computational lower bounds for submatrix detection Year: (2019)
Ref_id:b6 Title: Large deviations for the smallest eigenvalue of a deformed GOE with an outlier Year: (2024)
Ref_id:b7 Title: On the free convolution with a semi-circular distribution Year: (1997)
Ref_id:b8 Title: Processes with free increments Year: (1998)
Ref_id:b9 Title: On the geometry of the set of symmetric matrices with repeated eigenvalues Year: (2018)
Ref_id:b10 Title: Average-case integrality gap for non-negative principal component analysis Year: (2022)
Ref_id:b11 Title: Non-backtracking spectrum of random graphs: community detection and non-regular Ramanujan graphs Year: (2015)
Ref_id:b12 Title: Community detection with the z-Laplacian Year: (2014)
Ref_id:b13 Title: Informationtheoretic bounds and phase transitions in clustering, sparse PCA, and submatrix localization Year: (2018)
Ref_id:b14 Title: Notes on computational-to-statistical gaps: predictions using statistical physics Year: (2018)
Ref_id:b15 Title: Necessary and sufficient conditions for almost sure convergence of the largest eigenvalue of a Wigner matrix Year: (1988)
Ref_id:b16 Title: Exact separation phenomenon for the eigenvalues of large information-plusnoise type matrices, and an application to spiked models Year: (2014)
Ref_id:b17 Title: Deformed ensembles, polynomials in random matrices and free probability theory Year: (2017)
Ref_id:b18 Title: The largest eigenvalues of finite rank deformation of large Wigner matrices: convergence and nonuniversality of the fluctuations Year: (2009)
Ref_id:b19 Title: Free convolution with a semi-circular distribution and eigenvalues of spiked deformations of Wigner matrices Year: (2011-09)
Ref_id:b20 Title: The estimation error of general first order methods Year: (2020)
Ref_id:b21 Title: Statistical-computational tradeoffs in planted problems and submatrix localization with a growing number of clusters and submatrices Year: (2016)
Ref_id:b22 Title: On convergence of approximate message passing Year: (2014)
Ref_id:b23 Title: Asymptotic mutual information for the two-groups stochastic block model Year: (2015)
Ref_id:b24 Title: Approximation of the stability number of a graph via copositive programming Year: (2002)
Ref_id:b25 Title: Subexponential-time algorithms for sparse PCA Year: (2023)
Ref_id:b26 Title: Sparse PCA via covariance thresholding Year: (2014)
Ref_id:b27 Title: Finding hidden cliques of size N/e in nearly linear time Year: (2015)
Ref_id:b28 Title: Cone-constrained principal component analysis Year: (2014)
Ref_id:b29 Title: On the empirical distribution of eigenvalues of large dimensional information-plus-noise-type matrices Year: (2007)
Ref_id:b30 Title: Fundamental limits of detection in the spiked Wigner model Year: (2020)
Ref_id:b31 Title: The eigenvalues of random symmetric matrices Year: (1981)
Ref_id:b32 Title: TAP free energy, spin glasses, and variational inference Year: (2018)
Ref_id:b33 Title: The largest eigenvalue of rank one deformation of large Wigner matrices Year: (2007)
Ref_id:b34 Title: A unifying tutorial on approximate message passing Year: (2021)
Ref_id:b35 Title: Principal component analysis Year: (2022)
Ref_id:b36 Title: The overlap gap property in principal submatrix recovery. Probability Theory and Related Fields Year: (2021)
Ref_id:b37 Title: The semicircle law, free random variables and entropy Year: (2000)
Ref_id:b38 Title: Submatrix localization via message passing Year: (2017)
Ref_id:b39 Title: Semidefinite programs simulate approximate message passing robustly Year: (2024)
Ref_id:b40 Title: Fast, robust approximate message passing Year: (2025)
Ref_id:b41 Title: Principal component analysis: a review and recent developments Year: (2016)
Ref_id:b42 Title: On consistency and sparsity for principal components analysis in high dimensions Year: (2009)
Ref_id:b43 Title: On the distribution of the largest eigenvalue in principal components analysis Year: (2001)
Ref_id:b44 Title: PCA in high dimensions: An orientation Year: (2018)
Ref_id:b45 Title: Spectral redemption in clustering sparse networks Year: (2013)
Ref_id:b46 Title: Universal invariant and equivariant graph neural networks Year: (2019)
Ref_id:b47 Title: Expected complexity of graph partitioning problems Year: (1995)
Ref_id:b48 Title: Notes on computational hardness of hypothesis testing: Predictions using the low-degree likelihood ratio Year: (2019)
Ref_id:b49 Title: Optimal spectral initialization for signal recovery with applications to phase retrieval Year: (2019)
Ref_id:b50 Title: Any-dimensional equivariant neural networks Year: (2024)
Ref_id:b51 Title: MMSE of probabilistic low-rank matrix estimation: Universality with respect to the output channel Year: (2015)
Ref_id:b52 Title: Phase transitions of spectral initialization for high-dimensional non-convex estimation. Information and Inference: A Year: (2020)
Ref_id:b53 Title: Fundamental limits of symmetric low-rank matrix estimation Year: (2019)
Ref_id:b54 Title: On differentiating eigenvalues and eigenvectors Year: (1985)
Ref_id:b55 Title: Invariant and equivariant graph networks Year: (2018)
Ref_id:b56 Title: Large deviations for extreme eigenvalues of deformed Wigner random matrices Year: (2021)
Ref_id:b57 Title: On the universality of invariant networks Year: (2019)
Ref_id:b58 Title: Construction of optimal spectral methods in phase retrieval Year: (2022)
Ref_id:b59 Title: The computer science and physics of community detection: landscapes, phase transitions, and hardness Year: (2017)
Ref_id:b60 Title: Non-negative principal component analysis: message passing algorithms and sharp asymptotics Year: (2015)
Ref_id:b61 Title: On the limitation of spectral methods: From the gaussian hidden clique problem to rank-one perturbations of gaussian tensors Year: (2015)
Ref_id:b62 Title: Free probability and random matrices Year: (2017)
Ref_id:b63 Title: Optimal combination of linear and spectral estimators for generalized linear models Year: (2022)
Ref_id:b64 Title: Computational barriers in minimax submatrix detection Year: (2006)
Ref_id:b65 Title: Asymptotics of sample eigenstructure for a large dimensional spiked covariance model Year: (2007)
Ref_id:b66 Title: Equivariant polynomials for graph neural networks Year: (2023)
Ref_id:b67 Title: Statistical limits of spiked tensor models Year: (2020)
Ref_id:b68 Title: Optimality and suboptimality of PCA I: Spiked random matrix models Year: (2018)
Ref_id:b69 Title: Vector approximate message passing Year: (2019)
Ref_id:b70 Title: Community detection in graphs using singular value decomposition Year: (2011)
Ref_id:b71 Title: Topics in random matrix theory Year: (2012)
Ref_id:b72 Title: Free random variables. Number 1 Year: (1992)
Ref_id:b73 Title: High-dimensional probability Year: (2009)
Ref_id:b74 Title: Optimal transport: old and new Year: (2009)
Ref_id:b75 Title: The analogues of entropy and of Fisher's information measure in free probability theory, I. Communications in mathematical physics Year: (1993)
Ref_id:b76 Title: Collective dynamics of 'small-world' networks Year: (1998)
Ref_id:b77 Title: Sparse principal component analysis Year: (2006)
