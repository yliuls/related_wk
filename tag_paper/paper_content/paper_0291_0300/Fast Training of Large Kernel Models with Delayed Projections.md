Title: Fast Training of Large Kernel Models with Delayed Projections
Abstract: Classical kernel machines have historically faced significant challenges in scaling to large datasets and model sizes-a key ingredient that has driven the success of neural networks. In this paper, we present a new methodology for building kernel machines that can scale efficiently with both data size and model size. Our algorithm introduces delayed projections to Preconditioned Stochastic Gradient Descent (PSGD) allowing the training of much larger models than was previously feasible. We validate our algorithm, EigenPro 4, across multiple datasets, demonstrating drastic training speedups without compromising the performance. Our implementation is publicly available at: https://github.com/EigenPro/EigenPro.

Section: Introduction
Kernel methods have strong theoretical foundations and broad applicability. They have also served as the foundation for understanding many significant phenomena in modern machine learning [9,4,3,28]. Despite these advantages, the scalability of kernel methods has remained a persistent challenge, particularly when applied to large datasets. Addressing this limitation is critical for expanding the utility of kernel-based techniques in modern machine learning applications.
A naive approach for training kernel machines is to directly solve the equivalent kernel matrix inversion problem. In general, the computational complexity of this approach is O(n 3 ), where n is the number of training samples. Thus, computational cost grows rapidly with the size of the dataset, making it computationally intractable for datasets with more than ∼ 10 5 data points.
To address this challenge, various methods employing iterative algorithms and approximations have been proposed. Among these, Gradient Descent (GD)-based algorithms like Pegasos [21] and EigenPro 1.0,2.0 [13,14] have significantly reduced the computational complexity to O(n 2 ). These methods, adaptable for stochastic settings, offer more efficient implementations. Nevertheless, the scalability of kernel machines remains constrained by the inherent linkage between the model size and the training set. Furthermore, the Nyström methods have emerged as a favored approach for scaling kernel machines, with seminal works with [26] paving the way. Methods such as NYTRO [5], Falkon [20] and ASkotch [19] leverage the Nyström Approximation (NA) in combination with other strategies to enhance performance. NYTRO merges NA with gradient descent to improve condition number, ASkotch combines it with block coordinate descent, whereas Falkon combines it with the Conjugate Gradient method, facilitating the handling of large training sets. However, these strategies are limited by model size due to memory restrictions, exhibiting quadratic scaling in relation to the size of the Algorithm FLOPS Memory setup per batch EigenPro 4 O(1) O(p) O(p) EigenPro 3 O(1) O(p 2 ) O(p) Falkon O(p 3 ) O(p) O(p 2 ) 64 128 256 512 1000 Model size (×1000) 1 5 10 50 100 500 Hours 87.99% 88.25% 88.43% 88.58% 88.74% 88.33% 88.42% 88.61% 88.58% 88.7% 86.1% 86.55% 86.73% 86.71% OOM Time per epoch EP4 EP3 Falkon model. For instance, scaling Falkon [16] method to a model size of 512, 000 necessitates over 1TB of RAM, surpassing the capacity of most high-end servers available today.
Other lines of work in the Gaussian Processes literature, e.g., [22,27,7,15], use so-called inducing points to control model complexity. However, these methods face similar scaling issues as they require quadratic memory in terms of the number of inducing points, preventing large models.
Recently, EigenPro 3.0 was introduced in [1]. Unlike previous versions, EigenPro 3.0 distangles the model from the training set, similar to Falkon, but with the added advantage that its memory requirements scales linearly with the model size. This advancement makes it feasible to tackle kernel models of sizes previously deemed unattainable. However, its per iteration time complexity remains quadratic relative to the model size, significantly slowing its practical application.
In this paper, we build upon EigenPro 3.0 and introduce EigenPro 4.0. This new algorithm retains the advantageous features of EigenPro 3.0, such as decoupling the model from the training set and linear scaling in memory complexity. Moreover, it significantly improves upon the time complexity, achieving amortized linear scaling per iteration with respect to model size. Empirically we observe that the proposed algorithm converges in fewer epochs, without compromising generalization performance.
this section cite: ['b8', 'b3', 'b2', 'b27', 'b20', 'b12', 'b13', 'b25', 'b4', 'b19', 'b18', 'b15', 'b21', 'b26', 'b6', 'b14', 'b0']

Section: Main contribution
Our method for kernel machine problems achieves three key advantages: (1) linear amortized time complexity per iteration, (2) linear memory scaling with model size, (3) comparable or superior performance compare to existing methods while demonstrating up to 600× speedup in our experiments. Figure 1 demonstrates these benefits on the CIFAR5M data set.
this section cite: []

Section: Organization of the Paper
The remainder of this paper is organized as follows. Section 2 provides the necessary background and preliminaries. In Section 3, we derive the complete algorithm, and introduce the key insights and techniques that enable its dramatic improvement in computational efficiency. Finally, Section 4 presents extensive experimental results across multiple datasets and model sizes.
this section cite: []

Section: Notation and Background
In what follows, functions are lowercase letters a, sets are uppercase letters A, vectors are lowercase bold letters a, matrices are uppercase bold letters A, operators are calligraphic letters A, spaces and sub-spaces are boldface calligraphic letters A.
this section cite: []

Section: General kernel models.
Following EigenPro 3.0 [1] notations, given training data (X, y) = x i ∈ R d , y i ∈ R n i=1 , General kernel models are models of the form,
f (x) = p i=1 α i K(x, z i ).
Here, K : R d ×R d → R is a positive semi-definite symmetric kernel function and Z = {z i ∈ R d } p i=1 are centers, which are not necessarily in the training set X. We will refer to p as the model size. We define H as the unique reproducing kernel Hilbert space (RKHS) corresponding to K.
Loss function. Our goal will be to find the solution to the following infinite-dimensional Mean Squared Error (MSE) problem for general kernel models,
minimize f ∈H L(f ) = 1 2 n i=1 (f (x i ) -y i ) 2 , subject to f ∈ Z := span {K(•, z j )} p j=1 . (1
)
Evaluations and kernel matrices. The vector of evaluations of a function f over a set
X = {x i } n i=1
is denoted f (X) := (f (x i )) ∈ R n . For sets X and Z, with |X| = n and |Z| = p, we denote the kernel matrix K(X, Z) ∈ R n×p , while K(Z, X) = K(X, Z) ⊤ . Similarly, K(•, X) ∈ H n is a vector of n functions, and we use K(•, X)α := n i=1 K(•, x i )α i ∈ H, to denote their linear combination. Finally, for an operator A, a function a, and a set A = {a i } k i=1 , we denote the vector of evaluations,
A {a} (A) := (b(a i )) ∈ R k where b = A (a) .(2)
Fréchet derivative. Given a function J : H → R, the Fréchet derivative of J with respect to f is a linear functional, denoted ∇ f J, such that for h ∈ H
lim ∥h∥ H →0 |J(f + h) -J(f ) -∇ f J(h)| ∥h∥ H = 0.(3)
Since ∇ f J is a linear functional, it lies in the dual space H * . Since H is a Hilbert space, it is self-dual, whereby H * = H. If f is a general kernel model, and L is the square loss for a given dataset (X, y), i.e., L(f ) := 1 2 n i=1 (f (x i ) -y i ) 2 we can apply the chain rule, and using reproducing property of H, and the fact that ∇ f ⟨f, g⟩ H = g, we get, that the Fréchet derivative of L, at f = f 0 is,
∇ f L(f 0 ) = n i=1 (f 0 (x i ) -y i )∇ f f (x i ) = K(•, X)(f 0 (X) -y).(4)
Hessian operator. The Hessian operator ∇ 2 f L : H → H for the square loss is given by
K := n i=1 K(•, x i ) ⊗ K(•, x i ), K {f } (z) = n i=1 K(z, x i )f (x i ) = K(z, X)f (X), (5
)
where ⊗ denotes the outer product between functions in the RKHS, defined as (a⊗b
)(•) = a(•) ⟨b, •⟩ H .
The operator K has non-negative eigenvalues, which we order as λ 1 ≥ λ 2 ≥ • • • ≥ λ n ≥ 0. Hence, we can write its eigen-decomposition as K = n i=1 λ i ψ i ⊗ ψ i . Combining Equations 4 and 5, we can rewrite the Fréchet derivative of the loss function as
∇ f L(f 0 )(z) = K {f 0 (X) -y} (z).(6)
Exact minimum norm solution. The closed-form minimum ∥•∥ H norm solution to the problem defined in equation ( 1) is given by:
f := K(•, Z)K † (Z, X)y,(7)
where † is the pseudoinverse or Moore-Penrose inverse. In the case of X = Z it simplifies to f := K(•, X)K -1 (X, X)y.
this section cite: ['b0']

Section: Gradient Descent (GD).
If we apply GD on the optimization problem in 1, with learning rate η, in the H functional space, the update is as following,
f t+1 = f t -η • ∇ f L(f t ) = f t -ηK(•, X) (f t (X) -y) .(8)
The first point to note is that the derivative lies in X := span {K(•, x j )} n j=1 rather than in Z. Therefore, when X ̸ = Z, SGD cannot be applied in this form. We will revisit this issue later. The second point is that in the case of X = Z, traditional kernel regression problem, the convergence of SGD depends on the condition number of K. Simply put, this is the ratio of the largest to the smallest non-zero singular value of the Hessian operator defined in 5. It is known that for general kernel models this condition number is ill conditioned and converges slow, see [2] for more details on this.
EigenPro . Prior work, EigenPro , by [13], addresses the slow convergence of SGD by introducing a preconditioned stochastic gradient descent mechanism in Hilbert spaces. The update rule is the same as Equation 9but with an additional preconditioner P : H → H applied to the gradient,
f t+1 = f t -η • P∇ f L(f t ).(9)
In short, the role of the preconditioner P is to suppress the top eigenvalues of the Hessian operator K to improve the condition number. We next explicitly define P.
Definition 1 (Top-q Eigensystem). Let λ 1 > λ 2 > . . . > λ n be the eigenvalues of a Hermitian matrix A ∈ R n×n , where for standard unit vector e i , we have Ae i = λ i e i . We define the tuple (Λ q , E q , λ q+1 ) as the top-q eigensystem, where:
Λ q := diag(λ 1 , λ 2 , . . . , λ q ) ∈ R q×q , E q := [e 1 , e 2 , . . . , e q ] ∈ R n×q .(10)
Preconditioner. Using Definition 1, let (Λ q , E q , λ q+1 ) be the top-q eigensystem of K(X, X), the preconditioner P : H → H can be explicitly written as, P := I -
q i=1 1 - λq+1 λq ψ i ⊗ ψ i .
Nyström approximate preconditioner. EigenPro 2, introduced by [14], implements a stochastic approximation for P based on the Nyström extension, thereby reducing the time and memory complexity compared to EigenPro . The first step is to approximate the Hessian operator using the Nyström extension as follows,
K s := s k=1 K(•, x i k ) ⊗ K(•, x i k ) = s i=1 λ s i • ψ s i ⊗ ψ s i .(11)
This is a Nyström approximation of K using s uniformly random samples from X, referred to as X s , where (Λ s q , E s q , λ s q+1 ) represents the corresponding top-q eigensystem of K(X s , X s ). Using this approximation, we can define the approximated preconditioner as follows, P s := I -
q i=1 1 - λ s q+1 λ s q ψ s i ⊗ψ s i .
For more details on the performance of this preconditioner compared to the case of s = n, see [2], who showed that choosing s ≳ log 4 n is sufficient.
Of particular importance is the action of this preconditioner on any function of the form K(•, A)u.
P s K(•, A)u = K(•, A)u - q i=1 1 - λ s q+1 λ s q ψ i ψ i (A) ⊤ u (12a
) = K(•, A)u - q i=1 1 - λ s q+1 λ s q K(•, X s )e i √ λ i e ⊤ i K(X s , A) √ λ i u (12b) = K(•, A)u -K(•, X s )E s q D q E s ⊤ q K(X s , A)u (12c
)
where D q := Λ -1 q -λ q+1 Λ -2 q . EigenPro 3. The primary limitation of EigenPro 2 was its inability to handle cases where Z ̸ = X, a necessary condition for disentangling the model and the training set. EigenPro 3 overcomes this limitation by recognizing that although the gradients in (4) may not lie within Z, it is possible to project them back to Z. Consequently, EigenPro 3 can be summarized as follows:
f t+1 = proj Z f t -ηP s { ∇ f L(f t )} ,(13)
where proj Z (u) := argmin f ∈Z ∥u -f ∥ 2 H for any u ∈ H. As shown in [1, Section 4.2 ], the exact projection is, proj Z (u) = K(•, Z)K -1 (Z, Z)u(Z).
This projection can be interpreted as solving a kernel in Z and can be approximated using EigenPro 2, as done in [1], with a time complexity that scales quadratically with model size. However, since this projection must be performed after each stochastic step, it becomes the most computationally expensive part of the EigenPro 3 algorithm.
this section cite: ['b1', 'b12', 'b13', 'b1', 'b0']

Section: EigenPro 4: Algorithm Design, Derivation, and Optimization
In this section, we provide a high-level overview and illustrations to highlight the key components of EigenPro 4 and how it significantly reduces training time. We present the EigenPro 4 algorithm in three parts. First, we introduce the algorithm's main components: the pre-projection and projection steps. Then, we detail each of these steps in the following two subsections. Finally, we describe a computational optimization that reduces the runtime of EigenPro 4 by half.
this section cite: []

Section: Scaling Challenge: High Projection Overhead.
The key development of EigenPro 3 over its contemporaries was that it could train general kernel models in O(p) memory. This was a huge improvement over the prior methods which required O(p 2 ) memory [20,19]. However, EigenPro 3 has a high cost O(mp + p 2 ) per batch of data processed, as summarized in the table in Figure 1. This is especially expensive when m ≪ p, i.e., when the batch size m is small compared to the model size p.
this section cite: ['b19', 'b18']

Section: Main Idea: Delayed Projection.
To address computational complexity challenges, EP4 amortizes projection costs by delaying them for T iterations. An effective method for selecting T , along with an illustration of the delayed projection mechanism for T = 4 (Figure 4), is provided in Appendix B.
In fact, EigenPro 3 can be viewed as a special case of EigenPro 4 when the parameter T is set to 1.
Figure 5 in Appendix B illustrates this relationship. Furthermore, as shown in equation ( 26) in the same appendix, the total training time is minimized when T is proportional to p m , where m denotes the mini-batch size used in SGD. Under this setting, the per-batch training cost becomes O(p).
this section cite: []

Section: Derivation of the EigenPro 4 Algorithm
As mentioned previously T is a crucial hyperparameter that determines the frequency of projection back to Z after every T steps. Before the projection step T , at every step when a new batch (X m , y m ) is fetched, it is added to a set defined as the "temporary centers" set, denoted by Z tmp . Starting with an empty set, Z tmp = ∅, we continuously add temporary centers to Z tmp until the step count reaches T .
Formally, the prediction function prior to the projection is no longer fixed and is now expanding. The model can be described as follows:
f (x) = original model z∈Z α z K(x, z) + temporary centers z∈Ztmp β z K(x, z) (14
)
where α z refers to the weights corresponding to the original model center z and β z refers to the weights corresponding to the temporary model centers. The full EigenPro 4 algorithm has been illustrated in Figure 2 and mathematically can be summarized as following,
f t = proj Z f t-1 -ηP s { ∇ f L(f t-1 )} , t ≡ 0 mod T, f t-1 -ηP s { ∇ f L(f t )}, otherwise. (15
)
where ∇L is a stochastic gradient of the loss function computed over a mini-batch of data, and P s is the preconditioner.
this section cite: []

Section: Steps before the Projection Step: Update for t < T
Based on Equation ( 15), suppose (X 1 , y 1 ), . . . , (X T , y T ) are the minibatches of size m, and the initial model is f 0 = K(•, Z)α. After t < T step, the following holds,
f t = f t-1 -ηP s K(•, X t )(f t-1 (X t ) -y t ) = f 0 -ηP s t i=1 K(•, X i )g i g i := f i-1 (X i ) -y i
Replacing P s using equation (11), setting f 0 = K(•, Z)α, and following the notation introduced in section 2, let (Λ q , E s q , λ q+1 ) denote the top-q eigensystem of K(X s , X s ), where E s q ∈ R s×q and D q := Λ -1 q -λ q+1 Λ -2 q . Then, we can simplify the update above as follows: This update rule implies that after t steps, the weights corresponding to the original centers Z remain unchanged, the weights for the temporary centers X i are set once to -ηg i after they are added, and do not change thereafter, and finally, the weights associated with the Nyström samples X s are η
f t = K(•, Z)α 0 -η t i=1 K(•, X i ) -K(•, X s )E s q D q E s ⊤ q K(X s , X i ) g i (16
)
t i=1 E s q D q E s ⊤ q K(X s , X i )g i which can
be updated after each batch via an additive update. This is how we update the weights before projection at step T .
this section cite: ['b10']

Section: Projection Step: Update for t = T
Once, we reach step T we need to project f T into Z, or formally
f T = proj Z f T -1 -ηP s { ∇ f L(f T -1 )} (17
)
= proj Z f T -1 -η K(•, X T )g T -ηK(•, X s )E s q DE s ⊤ q K(X s , X T )g T ,
Applying Proposition 2 from [1], the solution to this projection problem is as follows,
f T = K(•, Z)K -1 (Z, Z) f T -1 (Z) -ηK(Z, X T )g T + ηK(Z, X s )E s q DE s ⊤ q K(X s , X T )g T (18
)
Algorithm 1 EigenPro 4
Require: Data (X, y), centers Z, batch size m, Nyström size s, preconditioner level q, projection period T 1: Fetch subsample X s ⊆ X of size s 2: (Λ q , E s q , λ q+1 ) ← top-q eigensystem of K(X s , X s ) and define D q := (Λ -1 q -λ q+1 Λ -2 q ) ∈ R q×q 3: while Stopping criterion is not reached do 4:
Z tmp ← ∅, α tmp ← ∅, α s ← 0 s , h ← 0 p 5: for t = {1, 2, . . . , T } do 6: Fetch minibatch (X m , y m ) of m samples 7: g m ← K(X m , Z)α + K(X m , Z tmp )α tmp + K(X m , X s )α s -y m 8:
Z tmp .append(X m ) and α tmp .append(-η • g m ) 9:
α s = α s + η • E s q DE s ⊤ q K(X s , X m )g m 10: h ← h + K(Z, X m )g m -K(Z, X s )E s q DE s ⊤ q K(X s , X m )g m 11:
end for 12:
α ← α -η • proj Z (h)
{Approximate projection implemented using EigenPro 2} 13: end while
this section cite: ['b0', 'b8']

Section: Improving Computational Efficiency
Upon careful examination of the derivations in (16), we observe that f T -1 (Z) have already been computed previously. This allows us to efficiently reuse f T -1 (Z) as follows,
f T -1 (Z) = K(Z, Z)α 0 -η T -1 i=1 K(Z, X i )g i -K(Z, X s )E s q DE s ⊤ q K(X s , X i )g i (19
)
plugging this in (18) we obtain,
f T = K(•, Z) α 0 -ηK -1 (Z, Z)h , (20a
) h := T i=1 K(Z, X i )g i -K(Z, X s )E s q DE s ⊤ q T i=1 K(X s , X i )g i (20b
)
this section cite: []

Section: Final algorithm
The final EigenPro 4 can be found in Algorithm 1. Note that we follow the same inexact projection scheme used in [1] to approximate the exact projection in the last step of the algorithm.
The benefit of this approximation is that we don't need to solve the problem exactly in X , nor do we need to project back to Z after each iteration. This approach offers the best of both worlds. In the next section, we demonstrate the effectiveness of this approach compared to prior state-of-the-art methods.
this section cite: ['b0']

Section: Convergence Analysis
In Appendix A, we provide a convergence analysis for the special case of Algorithm 1 where T = ∞ and exact projection is used in the projection step. However, a more rigorous theoretical analysis that accounts for the approximations introduced for scability-namely, finite T and inexact projections-is left for future work and is beyond the scope of this paper.
this section cite: []

Section: Numerical experiments
In this section, we demonstrate that our approach achieves orders-of-magnitude speedups over stateof-the-art kernel methods while maintaining comparable or superior generalization performance.
We evaluate several kernel methods on the following datasets: (1) CIFAR5M, (2) CIFAR5M 2 [17], (3) ImageNet 1 [6], (4) WebVision 2 [12], and (5) LibriSpeech [18]. Dataset details are provided in Appendix C. While our method is compatible with any kernel function, we primarily use the Laplace kernel due to its simplicity and strong empirical performance. For completeness, we also report results using the Gaussian and NTK kernels in Appendix C.1.
For multi-class classification, we adopt a one-vs-all decomposition strategy, training a separate binary regressor for each class with targets in 0, 1. The final prediction is obtained by selecting the class corresponding to the highest predicted value across all binary regressors.
Substantial Reduction in Per-Epoch Training Time. EigenPro 4 has substantially reduced the per-epoch training time, making it the most efficient kernel method on modern machine learning hardware. In contrast to performing projection every mini-batch iteration as in EigenPro 3, EigenPro 4 schedules one projection every few iterations such that its amortized cost is comparable to that of the standard iterations. This results in an ideal per-sample complexity O(p), a remarkable improvement over the O(p 2 ) complexity from EigenPro 3.
In Table 1, we evaluate the performance and computational timing for a single epoch of our proposed model against established kernel regression methods. As noted earlier, Falkon exhibits limitations due to its quadratic memory complexity. For the CIFAR5M * dataset, training with a model size of 512,000 required 1.3TB of RAM, while scaling to 1M model size necessitated over 5TB. Resource constraints limited our Falkon benchmarks to model sizes of 128,000 and 64,000 for the remaining datasets. While EigenPro 3 addresses these memory constraints, it demonstrates significant computational overhead, particularly evident in the Librispeech dataset where our method, EigenPro 4, achieves a 411× speedup. Notably, EigenPro 4 maintains comparable or superior performance across all evaluated datasets relative to both baseline methods.
this section cite: ['b16', 'b5', 'b11', 'b17']

Section: Linear Scaling with Model Size.
The total training time and memory usage of EigenPro 4 scales linearly with the model size. In comparison, the time required for a single EigenPro 3 iteration grows quadratically with the model size, while the preprocessing time for Falkon grows cubically. Furthermore, the memory demand of Falkon increases quadratically with the model size. In practice, we are unable to run it with large model sizes, e.g., 128,000 centers for ImageNet data. We summarize all empirical results in Figure 3 and demonstrate that our method achieves both linear memory complexity and linear time complexity (empirically verified) with respect to model size, offering the best of both worlds. For the ImageNet dataset, we trained all methods until convergence. While EigenPro 3 does not have the quadratic memory scaling problem, Figure 3 shows that even for a relatively small dataset like ImageNet with 1M data points, training a model size of 512,000 centers requires approximately 43 days on a single GPU to reach convergence (about 100 epochs). In contrast, our proposed model achieves convergence in approximately 3 hours, requiring only 15 epochs, with each epoch being significantly more efficient than EigenPro 3 (see Table 1).
Faster Convergence with EigenPro 4. EigenPro 4 generally demonstrates the fastest convergence among all tested methods. In certain cases, such as ImageNet with 1.2 million model centers, Eigen-Pro 4 converges in less than 10% of the epochs needed by other methods, while also delivering superior model performance. Figure 3 compares EigenPro 4 and EigenPro 3 across multiple training epochs, following the experimental setup established in [1]. Despite EigenPro 4's linear time complexity per iteration (compared to EigenPro 3's quadratic complexity), it demonstrates faster convergence with fewer epochs. This efficiency gain is particularly pronounced for larger model sizes, where EigenPro 4 maintains or exceeds EigenPro 3's accuracy while requiring significantly fewer epochs. These results empirically validate that EigenPro 4's algorithmic improvements translate to practical benefits: not only does each iteration run faster, but fewer iterations are needed to achieve optimal performance across diverse datasets. This empirically shows that our model has a linear time complexity with respect to the model size.
this section cite: ['b0']

Section: Conclusion
In this work, we introduced EigenPro 4, an advancement in training large kernel models that achieves linear time complexity per iteration and linear memory scaling with model size. By implementing a delayed projection strategy, we addressed the high computational overhead previously associated with frequent projections, achieving significant time and memory efficiency improvements over EigenPro 3 and Falkon. Our empirical results on diverse datasets highlight EigenPro 4 ability to match or exceed the performance of prior methods with vastly reduced computational resources. Specifically, the algorithm demonstrates both faster convergence and superior scalability, enabling training with model sizes and datasets that were previously infeasible due to memory and time constraints.
Furthermore, EigenPro 4 design opens up new possibilities for parallelization, as it is well-suited for multi-GPU and distributed architectures. Future work will explore these aspects, further expanding its potential in real-world applications requiring efficient, scalable kernel methods for massive data volumes.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Toward large kernel models Year: (2023)
Ref_id:b1 Title: On the nyström approximation for preconditioning in kernel machines Year: (2024)
Ref_id:b2 Title: Reconciling modern machinelearning practice and the classical bias-variance trade-off Year: (2019)
Ref_id:b3 Title: To understand deep learning we need to understand kernel learning Year: (2018)
Ref_id:b4 Title: When subsampling meets early stopping Year: (2016)
Ref_id:b5 Title: Imagenet: A largescale hierarchical image database Year: (2009)
Ref_id:b6 Title: Product kernel interpolation for scalable gaussian processes Year: (2018)
Ref_id:b7 Title: Evaluation of neural architectures trained with square loss vs cross-entropy in classification tasks Year: (2021)
Ref_id:b8 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b9 Title: Speech & language processing Year: (2000)
Ref_id:b10 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b11 Title: Webvision database: Visual learning and understanding from web data Year: (2017)
Ref_id:b12 Title: Diving into the shallows: a computational perspective on large-scale shallow learning Year: (2017)
Ref_id:b13 Title: Kernel machines that adapt to gpus for effective large batch training Year: (2019)
Ref_id:b14 Title: GPflow: A Gaussian process library using TensorFlow Year: (2017-04)
Ref_id:b15 Title: Kernel methods through the roof: handling billions of points efficiently Year: (2020)
Ref_id:b16 Title: The deep bootstrap framework: Good online learners are good offline generalizers Year: (2021)
Ref_id:b17 Title: Librispeech: an asr corpus based on public domain audio books Year: (2015)
Ref_id:b18 Title: Have askotch: A neat solution for large-scale kernel ridge regression Year: (2025)
Ref_id:b19 Title: Falkon: An optimal large scale kernel method Year: (2017)
Ref_id:b20 Title: Pegasos: Primal estimated sub-gradient solver for svm Year: (2007)
Ref_id:b21 Title: Variational learning of inducing variables in sparse gaussian processes Year: (2009)
Ref_id:b22 Title:  Year: (2014-09)
Ref_id:b23 Title: ESPnet: End-to-end speech processing toolkit Year: (2018)
Ref_id:b24 Title: Pytorch image models Year: (2019)
Ref_id:b25 Title: Using the nyström method to speed up kernel machines Year: (2000)
Ref_id:b26 Title: Kernel interpolation for scalable structured gaussian processes (kiss-gp) Year: (2015)
Ref_id:b27 Title: Understanding deep learning (still) requires rethinking generalization Year: (2021)
