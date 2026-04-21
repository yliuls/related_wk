Title: Training Dynamics of In-Context Learning in Linear Attention
Abstract: While attention-based models have demonstrated the remarkable ability of in-context learning (ICL), the theoretical understanding of how these models acquired this ability through gradient descent training is still preliminary. Towards answering this question, we study the gradient descent dynamics of multi-head linear self-attention trained for in-context linear regression. We examine two parametrizations of linear self-attention: one with the key and query weights merged as a single matrix (common in theoretical studies), and one with separate key and query matrices (closer to practical settings). For the merged parametrization, we show that the training dynamics has two fixed points and the loss trajectory exhibits a single, abrupt drop. We derive an analytical time-course solution for a certain class of datasets and initialization. For the separate parametrization, we show that the training dynamics has exponentially many fixed points and the loss exhibits saddle-to-saddle dynamics, which we reduce to scalar ordinary differential equations. During training, the model implements principal component regression in context with the number of principal components increasing over training time. Overall, we provide a theoretical description of how ICL abilities evolve during gradient descent training of linear attention, revealing abrupt acquisition or progressive improvements depending on how the key and query are parametrized.

Section: Introduction
Self-attention-based models, such as transformers (Vaswani et al., 2017), exhibit a remarkable ability known as incontext learning (Brown et al., 2020). That is, these models Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
can solve unseen tasks based on exemplars in the context of an input prompt. In-context learning (ICL) is critical to the flexibility of large language models, allowing them to solve tasks not explicitly included in their training data. However, it remains unclear how architectures like self-attention acquire this ability through gradient descent training.
Seminal work by Olsson et al. (2022) identified an intriguing trait in the training dynamics of ICL: the ICL ability often emerges abruptly, coinciding with an abrupt drop in loss during training. This abrupt learning phase can reflect the formation of an induction head in the ICL setting (Olsson et al., 2022;Reddy, 2024;Singh et al., 2024;Edelman et al., 2024), and can also occur more broadly in transformer training dynamics (Nanda et al., 2023;Chen et al., 2024a;Hoffmann et al., 2024;Gopalani et al., 2024). Furthermore, Singh et al. (2023) found that ICL may often be a transient ability that the transformers acquire and then lose over the course of long training time, a phenomenon that has since been reproduced in many settings (He et al., 2024;Anand et al., 2025;Chan et al., 2025;Nguyen & Reddy, 2025;Park et al., 2025;Singh et al., 2025). These findings underscore the importance of understanding not only the ICL ability in trained models, but its full training dynamics. This work aims to provide a theoretical description of how the ICL ability evolves in gradient descent training. To do so, we consider the increasingly common setup of linear attention 1 (Von Oswald et al., 2023) trained on an in-context linear regression task (Garg et al., 2022). The in-context linear regression task, in which the model needs to perform linear regression on the data in context, is a canonical instantiation of ICL (Garg et al., 2022;Akyürek et al., 2023;Von Oswald et al., 2023;Ahn et al., 2023;Bai et al., 2023). The linear attention model, which has been used in many prior studies (Schlag et al., 2021;Von Oswald et al., 2023;Ahn et al., 2023;Zhang et al., 2024a;Wu et al., 2024;Fu et al., 2024;Mahankali et al., 2024;Duraisamy, 2024;Li et al., 2024;Yau et al., 2024;Lu et al., 2025;Frei & Vardi, 2025), reproduces key optimization properties of practical transformers (Ahn et al., 2024) and is more amenable to theoretical analysis. Importantly, despite its name, linear attention is a nonlinear model, as it removes the softmax operation but is still a nonlinear function of the input.
We study two common parametrizations of multi-head linear attention: (i) ATTN M , linear attention where the key and query matrices in each head are merged into a single matrix, a reparametrization procedure widely used in theoretical studies on transformers (Ahn et al., 2023;Tian et al., 2023;Ataee Tarzanagh et al., 2023;Zhang et al., 2024a;b;Chen et al., 2024b;Wu et al., 2024;Kim & Suzuki, 2024;Huang et al., 2024b;Wang et al., 2024b;Ildiz et al., 2024;Ren et al., 2024;Tarzanagh et al., 2024;Yau et al., 2024;Julistiono et al., 2024;Anwar et al., 2024;Vasudeva et al., 2025;Lu et al., 2025;Chen & Li, 2025;Huang et al., 2025a); (ii) ATTN S , linear attention with separate key and query matrices, which is closer to the implementation of attention in real-world transformers (Vaswani et al., 2017). We specify the fixed points in the loss landscapes, as well as how gradient descent training dynamics traverses the landscape. Our findings are summarized as follows.
• We find two fixed points in the training dynamics of ATTN M , and exponentially many fixed points in that of ATTN S .
• We show a single, abrupt loss drop in training ATTN M from small initialization and derive an analytical timecourse solution when the input token covariance is white. We show saddle-to-saddle training dynamics in training ATTN S from small initialization and reduce the highdimensional training dynamics to scalar ordinary differential equations through an ansatz. We demonstrate the rank of the separate key and query weights affects the dynamics by shortening the duration of certain plateaus.
• We identify the in-context algorithm of the converged and early stopped models. When ATTN M and ATTN S are trained to convergence, they approximately implement least squares linear regression in context. When the training of ATTN S early stops during the (m + 1)-th loss plateau, it approximately implements principal component regression in context with the first m principal components.
• As a tool for our analysis, we show that when trained on in-context linear regression tasks, ATTN M is equivalent to a two-layer fully-connected linear network with a cubic feature map as input, and ATTN S is equivalent to a sum of three-layer convolutional linear networks with the same cubic feature map as input.
• We empirically demonstrate that the single and multiple loss drops also occur in softmax ATTN M and ATTN S , respectively.
Comparing the two models, we find that the ICL ability evolves differently in them: ATTN M acquires the in-context linear regression ability through one abrupt loss drop, while ATTN S acquires this ability by progressively improving on in-context principal component regression. This makes a theoretical case for the progressive improvements of ICL in gradient descent training. Our results also reveal how parametrization, such as merged versus separate key and query and the rank of the separate key and query weights, influences the loss landscape and training dynamics. This motivates future research to take the parametrization factor into account when studying the landscape and dynamics of attention models.
this section cite: ['b72', 'b13', 'b53', 'b53', 'b56', 'b65', 'b24', 'b50', 'b33', 'b30', 'b64', 'b32', 'b4', 'b14', 'b51', 'b55', 'b66', 'b74', 'b28', 'b28', 'b3', 'b74', 'b1', 'b9', 'b61', 'b74', 'b1', 'b78', 'b26', 'b48', 'b23', 'b47', 'b80', 'b47', 'b25', 'b2', 'b1', 'b69', 'b7', 'b78', 'b43', 'b37', 'b57', 'b68', 'b80', 'b42', 'b5', 'b71', 'b47', 'b17', 'b72']

Section: Preliminaries
Notation. Non-bold small and capital symbols are scalars. Bold small symbols are column vectors. Bold capital symbols are matrices. ∥ • ∥ denotes the ℓ 2 norm of a vector or the Frobenius norm of a matrix. vec(•) represents flattening a matrix to a column vector by stacking its columns.
For example, vec 1 3 2 4 = 1 2 3 4 ⊤ . We use i = 1, • • • , H to denote the index of an attention head, µ = 1, • • • , P to denote the index of a training sample, and n = 1, • • • , N to denote the index of a token in a sample.
this section cite: []

Section: In-Context Linear Regression Task
We study a standard ICL task of predicting the next token.
The input is a sequence
{x 1 , y 1 , x 2 , y 2 , • • • , x N , y N , x q }
and the desired output is y q . We refer to x q as the query token, {x 1 , y 1 , x 2 , y 2 , • • • , x N , y N } as the context, and N as the context length. By convention (Ahn et al., 2023;Zhang et al., 2024a;b;Chen et al., 2024b;Huang et al., 2024b), the input sequence is presented to the model as a matrix X, defined as
X = x 1 x 2 • • • x N x q y 1 y 2 • • • y N 0 ∈ R (D+1)×(N +1) , (1
)
where
x 1 , • • • , x N , x q ∈ R D and y 1 , • • • , y N ∈ R.
We are given a training dataset {X µ , y µ,q } P µ=1 consisting of P samples. All x tokens are independently sampled from a D-dimensional zero-mean normal distribution with covariance Λ,
x µ,n , x µ,q ∼ N (0, Λ), n = 1, • • • , N, µ = 1, • • • , P. (2)
We consider the in-context linear regression task, where the y n in context and the target output y q are generated as a linear map of the corresponding x n and x q (Garg et al., 2022). For each sequence X µ , we independently sample a task vector w µ from a D-dimensional standard normal distribution, w µ ∼ N (0, I), and generate
y µ,n = w ⊤ µ x µ,n , y µ,q = w ⊤ µ x µ,q , n = 1, • • • , N, µ = 1, • • • , P .
Note that the task vector w µ is fixed for all tokens in one sample sequence but varies across different samples, and is independent of the tokens x µ,1 , • • • , x µ,N , x µ,q .
this section cite: ['b1', 'b28']

Section: Multi-Head Self-Attention
A standard multi-head softmax self-attention layer (Vaswani et al., 2017) takes the matrix X as input and returns a matrix of the same size,
ATTN(X) = X + H i=1 W V i Xsmax X ⊤ W K i ⊤ W Q i X ρ
where H is the number of heads, ρ is a scaling factor, and W V i , W K i , W Q i are the trainable value, key, and query matrices in the i-th head. The prediction for y q is the bottom right entry of the output matrix:
ŷq = ATTN(X) D+1,N +1 .(3)
In this work, we consider multi-head linear self-attention, where we remove the softmax operation and take ρ = N . Specifically, we study two common parametrizations of linear attention: (i) linear attention with merged key and query introduced in Section 2.3 and analyzed in Section 3;
(ii) linear attention with separate key and query introduced in Section 2.4 and analyzed in Sections 4 and 5.
this section cite: ['b72']

Section: Linear Attention with Merged Key and Query
The multi-head linear attention ATTN M with the key and query matrices in each head merged as a single matrix
W K i ⊤ W Q i = W KQ i computes ATTN M (X) = X + H i=1 1 N W V i XX ⊤ W KQ i X,
where the terms can be written in block form,
XX ⊤ = x q x ⊤ q + N n=1 x n x ⊤ n N n=1 x n y n N n=1 y n x ⊤ n N n=1 y 2 n ,and
W V i = * * v ⊤ i v i , W KQ i = U i * u ⊤ i * .
this section cite: []

Section: The blocks have dimensionalities
v i , u i ∈ R D , v i ∈ R, U i ∈ R D×D .
v i , u i = 0, the prediction for y q , which is the bottom right entry of ATTN M (X), is ATTN M (X) D+1,N +1 = H i=1 v i β ⊤ U i x q , (M
)
where β is the correlation between x n and y n in context,
β ≡ 1 N N n=1 y n x n .(4)
this section cite: []

Section: Linear Attention with Separate Key and Query
In multi-head attention with separate key and query, we follow the standard practice (Vaswani et al., 2017) of using low-rank key and query matrices where the rank R ≤ D and RH ≥ D. In practice, usually RH = D. The multi-head linear attention ATTN S with separate rank-R key and query matrices computes
ATTN S (X) = X + H i=1 1 N W V i XX ⊤ W K i ⊤ W Q i X.
We can write the value, key, and query weights in block form,
W V i = * * v ⊤ i v i , W K i =    k ⊤ i,1 k i,1 . . . . . . k ⊤ i,R k i,R    , W Q i =    q ⊤ i,1 * . . . . . . q ⊤ i,R *    .
The blocks have dimensionalities v i , k i,r ∈ R and
v i , k i,r , q i,r ∈ R D (r = 1, • • • , R).
Similarly to the case with merged key and query, we initialize v i = 0, k i,r = 0; they will remain zero throughout training (see Appendix F.1). With v i = 0 and k i,r = 0, the multi-head linear attention with separate rank-one key and query matrices computes
ATTN S (X) D+1,N +1 = H i=1 R r=1 v i β ⊤ k i,r q ⊤ i,r x q , (S)
where β is the input-output correlation in context defined in Equation ( 4). The expression of Equation (S) already reveals interesting insight. It implies that linear attention with H heads and rank-R key and query differs from linear attention with RH heads and rank-one key and query only in the sharing of certain value weights.
this section cite: ['b72']

Section: Gradient Flow Training Dynamics
We train the linear attention model using gradient descent on squared loss of the query tokenfoot_1 , that is L = E(y q -ŷq ) 2 . We analyze the gradient flow dynamics on the loss, given by
τ dW dt = - 1 2 ∂L ∂W = E (y q -ŷq ) ∂ ŷq ∂W ,(5)
where τ is the time constant. The gradient flow dynamics captures the behavior of gradient descent in the limit of a small learning rate.
this section cite: []

Section: Linear Attention with Merged Key and Query
We first study multi-head linear attention with the key and query matrices merged as a single matrix, as described by Equation (M).  Von Oswald et al., 2023) and softmax attention (Singh et al., 2024). Here D = 4, N = 31, H = 8.
this section cite: ['b74', 'b65']

Section: Connection to A Fully-Connected Linear Network
The H-head linear attention with input sequence X defined in Equation (M) can be viewed as a two-layer width-H fully-connected linear network with a cubic feature z(X) as input,
ATTN M (X) D+1,N +1 = H i=1 v i β ⊤ U i x q = H i=1 v i vec(U i ) ⊤ vec βx ⊤ q = w ⊤ 2 W 1 z = MLP(z),(6)
where
w 2 =      v 1 v 2 . . . v H      , W 1 =      vec(U 1 ) ⊤ vec(U 2 ) ⊤ . . . vec(U H ) ⊤      , z(X) = vec βx ⊤ q .(7)
The feature z ∈ R D 2 , whose entries are cubic functions of the entries in the original sequence X, is the input to the equivalent two-layer fully-connected linear network. The stacked value weights correspond to the second-layer weights w 2 ∈ R H of the fully-connected linear network. The stacked merged key-query weights correspond to the first-layer weights W 1 ∈ R H×D 2 of the fully-connected linear network. A schematic of this equivalence is given in Figure 1.
this section cite: []

Section: Loss Landscape: Two Fixed Points
The gradient flow training dynamics of the linear attention or the equivalent two-layer fully-connected linear network given in Equation ( 6) is
τ Ẇ1 = w 2 E y q z ⊤ -w ⊤ 2 W 1 E zz ⊤ ,(8a)
τ ẇ2 = W 1 E y q z ⊤ -w ⊤ 2 W 1 E zz ⊤ ⊤ . (8b
)
There are two manifolds of fixed points in this dynamical system: one is the unstable fixed point at zero, denoted M 0 , and the other is a manifold of stable fixed points at the global minimum, denoted M * ,
M 0 = {w 2 = 0, W 1 = 0}(9a)
M * = w 2 , W 1 w ⊤ 2 W 1 = E y q z ⊤ E zz ⊤ -1(9b)
this section cite: []

Section: Training Dynamics: An Abrupt Drop in the Loss
We have shown the linear attention defined in Equation (M) is equivalent to a fully-connected linear network with cubic feature input. Since this equivalence holds at the level of the computation of the model, the equivalence applies to the training dynamics with any initialization and optimizer.
Here we discuss the training dynamics from small initialization, commonly referred to as the rich learning regime (Woodworth et al., 2020).
With small initialization, the network is initially near the unstable fixed point, M 0 , at zero. As training progresses, the network escapes from the unstable fixed point, and subsequently converges to a stable fixed point on the global minimum manifold, M * . The time it takes to escape from the unstable fixed point is approximately τ ∥Λ 2 ∥ ln 1 winit , where the initialization scale w init is the initial ℓ 2 norm of a layer (see Appendix D.6.1). Because the time to escape from the unstable fixed point starting from small initialization is long, the loss exhibits an initial plateau followed by an abrupt drop, as validated by simulations in Figure 1. In particular, when the input token covariance is white Λ = I and the initialization is infinitesimally small, we exploit the equivalence between linear attention and linear networks to derive an analytical time-course solution (see Appendix D.5) and obtain
ATTN M (X; t) D+1,N +1 = σ(t)β ⊤ x q , where σ(t) = e 2 √ D t τ 1 + 1+D N e 2 √ D t τ -1 + √ D w 2 init . (10)
Since σ(t) is a rescaled and shifted sigmoid function, the weights and the loss trajectories have sigmoidal shapes, characterized by a plateau followed by a rapid drop.
this section cite: ['b77']

Section: ICL Algorithm: Least Squares Regression
When the linear attention model converges to the global minimum manifold M * at the end of training, the model implements
ATTN M (X) D+1,N +1 = E y q z ⊤ E zz ⊤ -1 z = β ⊤ Λ + Λ + tr(Λ)I N -1 x q ,(11)
where the first equality follows directly from Equations ( 6) and (9b) and the second equality is proved in Appendix D.4. Equation ( 11) reveals an intriguing duality: the linear regression solution in the cubic feature space of z is the in-context linear regression solution in the original space of the x n , y n token pairs in a sequence X. The first line of Equation ( 11) is the linear regression solution of fitting y µ,q with z µ for all training sequences µ = 1, • • • , P . The second line of Equation ( 11) is approximately the in-context linear regression solution, which fits y µ,n with x µ,n (n = 1, • • • , N ) for each sequence X µ . When the sequence length N is large, the model recovers the inverse of the true covariance matrix,
lim N →∞ β ⊤ Λ + Λ + tr(Λ)I N -1 x q = β ⊤ Λ -1 x q .
Here β is the x n , y n correlation in a sequence X, and Λ is the covariance of all x n tokens in all training sequences, which approximates the covariance of x n in each individual sequence.
this section cite: []

Section: Linear Attention with Separate Rank-One Key and Query
We now study multi-head linear attention with separate lowrank key and query matrices. Because the rank-one case captures most of the behaviors of the general rank-R case, we focus on the rank-one case in this section and defer the rank-R case to Section 5. When R = 1, the model definition in Equation (S) simplifies to
ATTN S (X) D+1,N +1 = H i=1 v i β ⊤ k i q ⊤ i x q . (12
)
this section cite: []

Section: Connection to Convolutional Linear Networks
The H-head linear attention with separate rank-one key and query can be viewed as a sum of H three-layer convolutional linear network with the cubic feature z defined in Equation ( 7) as input. Specifically, Equation ( 12) can be
𝒌 𝐻 𝒌 1 ⋯ ⋯ Figure 2.
Multi-head linear attention with separate rank-one key and query ATTNS(X)D+1,N+1 is a sum of H (number of heads) three-layer convolutional linear networks with the cubic feature z as input. Here we take D = 3 to avoid clutter. Entries in the vectors are denoted as
xq = x 1 q , x 2 q , x 3 q ⊤ , β = β 1 , β 2 , β 3 ⊤ .
rewritten as
ATTN S (X) D+1,N +1 = H i=1 v i q ⊤ i K i z,
where
K i =      k ⊤ i 0 ⊤ D . . . 0 ⊤ D 0 ⊤ D k ⊤ i . . . 0 ⊤ D . . . . . . . . . . . . 0 ⊤ D 0 ⊤ D . . . k ⊤ i      ∈ R D×D 2 .(13)
The matrix K i is a convolutional matrix with kernel size D and stride D. A schematic of the three-layer convolutional linear network is given in Figure 2.
When the number of heads satisfies H ≥ D, the linear attention with separate rank-one key and query, ATTN S (X), can express any linear map of z(X) and has the same expressivity as linear attention with merged key and query, ATTN M (X). However, the two models correspond to multilayer linear networks with different connectivity and depths, resulting in different loss landscape (Kohn et al., 2022;2024) and training dynamics (Saxe et al., 2014;2019).
this section cite: ['b44', 'b59', 'b62']

Section: Loss Landscape: Exponentially Many Fixed Points
The gradient flow training dynamics of linear attention with separate rank-one key and query, derived in Appendix E.2, is given by
τ vi = k ⊤ i Λ 2 -E Λ2 H i ′ =1 v i ′ k i ′ q ⊤ i ′ Λ q i , (14a
) τ ki = v i Λ 2 -E Λ2 H i ′ =1 v i ′ k i ′ q ⊤ i ′ Λ q i ,(14b)
τ qi = v i Λ 2 -Λ H i ′ =1 v i ′ k i ′ q ⊤ i ′ E Λ2 k i ,(14c)
0 5000 10000 Training steps t 0.0 0.2 0.4 0.6 0.8 1.0 Loss Simulations Theory where we denote the in-context covariance of x n tokens as
L(M 0 ) L(M 1 ) L(M 2 ) L(M 3 ) L(M 4 )(
Λ = N n=1 x n x ⊤ n /N and the expectation of Λ2 is E Λ2 = Λ 2 + Λ + tr(Λ)I N Λ (15
)
This dynamical system contains 2 D fixed points in the function space of ATTN S (X) D+1,N +1 . We specify the fixed points below and prove their validity in Appendix E.3.
Let λ 1 , • • • , λ D be the eigenvalues of the covariance matrix Λ arranged in descending order, and e 1 , • • • , e D be the corresponding normalized eigenvectors. We use M(S m ) to denote a set of fixed points that correspond to learning m
(m = 0, 1, • • • , D) out of the D eigenvectors, M(S m ) = (v, k, q) 1:H conditions (C1)-(C3) , (16
)
where the set S m specifies the indices of the learned eigenvectors,
S m ⊆ {1, 2, • • • , D}, |S m | = m.(17)
The three conditions for Equation ( 16) are:
(C1) The heads sum to fit the eigenvectors with indices in the set
S m H i=1 v i k i q ⊤ i = d∈Sm λ -1 d 1 + 1 + tr(Λ)/λ d N -1 e d e ⊤ d .(18)
(C2) For heads with a nonzero value weight, v i ̸ = 0, both k i and q i lie in the span of {e d } d∈Sm .
(C3) For heads with a zero value weight, v i = 0, at least one of k i or q i lies in the span of {e d } d∈Sm .
Since there are D m possible ways of choosing m out of D indices to define S m in Equation ( 17), the total number of possible choices summed over 18) and thus a different function, ATTN S (X) D+1,N +1 . Hence, the gradient flow dynamics in Equation ( 14) has 2 D fixed points in the function space. 3The two fixed points of ATTN M (Section 3.2) are contained in the 2 D fixed points of ATTN S : the zero fixed point in Equation (9a) corresponds to M(S 0 ), i.e., learning no eigenvector; the global minimum fixed point in Equation (9b) corresponds to M(S D ), i.e., learning all D eigenvectors.
m = 0, • • • , D is D m=0 D m = 2 D . Each choice corresponds to a different condition (C1) in Equation (
this section cite: []

Section: Training Dynamics: Saddle-to-Saddle Dynamics
Building on the exponentially many fixed points we have identified, we now analyze which fixed points are actually visited in gradient flow training and in what order. We find that starting from small initialization, the model visits (D + 1) out of the 2 D fixed points.
With small initialization, the model is initially near the unstable zero fixed point, M 0 = M(∅). As training progresses, the model sequentially visits the fixed points
in M 1 , M 2 , • • • , M D , where M m = M({1, 2, • • • , m}).
That is, the model trained from small initialization sequentially learns to fit the first eigenvector (the eigenvector of Λ with the largest eigenvalue), the second eigenvector, and so on. As shown in Figure 3a, the loss goes through D abrupt drops in training, each corresponding to the transition from one fixed point to the next. The abrupt drops of loss are separated by plateaus, during which the model lingers near an unstable fixed point. Because the time required for a head to learn the eigenvector e m from small initialization scales with λ -2 m (see Appendix E.6), eigenvectors associated with larger eigenvalues are learned faster. This explains why the model learns to fit the eigenvectors sequentially in descending order of the eigenvalues, as well as why we empirically see the later plateaus last longer in Figure 3a.
When the model is at a fixed point in M m , we compute the loss in Appendix E.4 and obtain
L(M m ) = tr(Λ) - m d=1 λ d 1 + 1 + tr(Λ)/λ d N -1 .(19)
Equation ( 19) is highly interpretable in the limit of a large sequence length N . The loss, L(M m ), is the sum of the eigenvalues associated with the remaining unlearned eigenvectors
lim N →∞ L(M m ) = tr(Λ) - m d=1 λ d = D d=m+1 λ d .
Thus, the loss decreases by approximately λ m during the m-th abrupt loss drop. We plot Equation ( 19) as dashed gray lines in Figure 3a and find they match the plateaus of simulated loss trajectories well.
When the model reaches M m from small initialization, its weights take on a highly structured form, which is a specific instance of the general definition in Equation ( 16). As shown in Figure 3c, the key and query weights in a head grow in scale and align with a new eigenvector of the input token covariance Λ during each abrupt loss drop. Based on simulations in Figure 3 and derivations in Appendices E.5 and E.6, we propose an ansatz that during the (m + 1)-th plateau (0 ≤ m < D) and the subsequent abrupt drop of loss, the weights are approximately given byfoot_3
k i = q i = v i e i , v i = λ -1 3 i 1 + 1 + tr(Λ)/λ i N -1 3 , 1 ≤ i ≤ m,(20a)
k i = q i = v i (t)e m+1 , i = m + 1,(20b)
k i = q i = 0, v i = 0, m + 2 ≤ i ≤ H,(20c)
where v m+1 (t) is small during the (m + 1)-th loss plateau and grows during the (m + 1)-th abrupt loss drop. Equation (20) implies that the ℓ 2 norms of v i , k i , q i in a head are equal, which is a consequence of small initialization and the conservation law in Appendix E.8. With this ansatz, the high-dimensional training dynamics during the (m + 1)-th plateau and the subsequent abrupt drop of loss reduces to an ordinary differential equation about v i (t), i = m + 1:
τ vi = λ 2 m+1 v 2 i -λ 3 m+1 1 + 1 + tr(Λ)/λ m+1 N v 5 i .(21)
Equation ( 21) is a separable differential equation but does not admit a general analytical solution of v m+1 (t) in terms of t (see Equation ( 71)). Nonetheless, it greatly simplifies the high-dimensional dynamics in Equation ( 14) and provides a good approximation of the true dynamics: during each plateau and the subsequent abrupt loss drop, weights in one of the heads grow in scale with the key and query weights aligning with the next eigenvector, while the rest of the heads remain approximately unchanged. In Figure 3b, we compare the numerical solution of Equation ( 21) with the value weights trajectories in the simulation and find excellent agreement.
In summary, the loss trajectory of linear attention with separate rank-one key and query trained from small initialization exhibits D abrupt drops, each followed by a plateau. The amount of the m-th abrupt loss drop (1 ≤ m ≤ D) is approximately the eigenvalue λ m , during which the key and query weights in an attention head grow in scale and align with the eigenvector e m .
this section cite: []

Section: ICL Algorithm: Principal Component Regression
When the linear attention model is at a fixed point in M m , based on Equation ( 18), the model implements
ATTN S (X) D+1,N +1 =β ⊤ m d=1 λ -1 d 1 + 1 + tr(Λ)/λ d N -1 e d e ⊤ d x q . (22
)
In the limit of a large sequence length N , Equation ( 22) simplifies and can be interpreted as principal component regression in context with m principal components
lim N →∞ ATTN S (X) D+1,N +1 = w ⊤ m d=1 e d e ⊤ d x q .
Here w is the task vector for the sequence X, and m d=1 e d e ⊤ d x q is query input x q projected onto the first m principal components. Hence, if training stops during the (m + 1)-th plateau, the linear attention approximately implements the principal component regression algorithm in context with m principal components.
After the model has undergone D plateaus, it converges to the global minimum fixed point, M D , and approximately implements principal component regression in context with all D components, which is least square regression. Thus, the linear attention model with either merged or separate key and query undergoes different training dynamics but converges to the same global minimum solution.
this section cite: []

Section: Linear Attention with Separate Low-Rank Key and Query
The linear attention model with separate rank-R key and query shares many behaviors with its rank-one counterpart.
For loss landscape, linear attention with rank-R key and query has the same 2 D fixed points in the function space as its rank-one counterpart, corresponding to the model implementing in-context principal component regression with a subset of all D principal components (see Appendix F.3).
For training dynamics, the loss trajectories differ slightly, depending on the rank R. We plot the loss trajectories with input token dimension D = 8 and different ranks R = 1, 2, 4, 8 in Figure 4 (see Figure 12 for R = 3, 5, 6, 7). For R = 1, the loss exhibits plateaus at eight values L(M m ) (m = 0, 1, • • • , 7). For R = 2, the loss exhibits plateaus at four values L(M m ) (m = 0, 2, 4, 6), and either brief plateaus or no plateau at the other four values. For R = 4, the loss exhibits conspicuous plateaus at only two values L(M m ) (m = 0, 4). To summarize, with rank-R key and query, the loss trajectory exhibits conspicuous plateaus at value L(M m ) for m that divides R.
The difference in the loss trajectories arises from the structure of the model defined in Equation (S). Each attention head has a single value weight v i that is associated with all R pairs of key and query weights in that head,
k i,r , q i,r (r = 1, • • • , R).
During a conspicuous plateau, a new value weight escapes from the unstable zero fixed point and grows in scale. Once the value weight has grown, it leads to larger gradient updates for all the key and query weights in that head, speeding up their escape from the zero fixed point. Hence, in the rank-R case, a conspicuous plateau occurs when m divides R, corresponding to learning a new head from small initialization. Brief or no plateau occurs when m does not divide R, corresponding to learning a new pair of key and query weights in a head whose value weight has already grown, as shown in Figure 11. See Appendix F.4 for further details.
this section cite: []

Section: Related Work
Recent theoretical research on linear attention has investigated its expressivity (Vladymyrov et al., 2024;Gatmiry et al., 2024), learnability (Yau et al., 2024), loss landscape (Mahankali et al., 2024;Li et al., 2024), convergence (Zhang et al., 2024a;b;Ren et al., 2024;Fu et al., 2024), and generalization (Wu et al., 2024;Mahankali et al., 2024;Duraisamy, 2024;Abedsoltan et al., 2024;Lu et al., 2025;Frei & Vardi, 2025).
The seminal work by Zhang et al. (2024a) analyzed the gradient flow training dynamics of linear attention to prove convergence guarantees, showing what the model converges to at the end of training. Our work also analyzes the gradient flow training dynamics but goes beyond existing convergence results to describe the entire training dynamics. Moreover, we study multi-head attention with merged or separate key and query weights, while Zhang et al. (2024a) focused on single-head attention with merged key and query.
Another line of recent research on the training dynamics of softmax attention models has shown stage-wise dynamics. Due to the intractability of softmax attention training dynamics in general, many of these studies made strong assumptions to enable theoretical analyses, including a simplified layer-wise training algorithm in place of standard gradient descent (Tian et al., 2023;Nichani et al., 2024;Chen et al., 2024c;Wang et al., 2024a), restricted weights (Boix-Adsera et al., 2023;Chen et al., 2024b;Rende et al., 2024;Edelman et al., 2024), and specifically chosen datasets (Huang et al., 2024b). In comparison, our work leverages the linear attention model without the softmax operation, enabling us to study in fine detail the dynamics of standard gradient descent training without restrictions on weights. Namely, we derive an analytical time-course solution and reduce the high-dimensional dynamics to one-dimensional ordinary differential equations for the two models we study, respectively. Furthermore, we characterize how parametrization (i.e., merged or separate key and query, and rank of the separate key and query weights) affects the loss landscape and training dynamics, an aspect not previously examined.
this section cite: ['b73', 'b28', 'b80', 'b48', 'b47', 'b57', 'b26', 'b78', 'b48', 'b23', 'b0', 'b47', 'b25', 'b69', 'b52', 'b10', 'b58', 'b24']

Section: Discussion
We studied the gradient flow training dynamics of multihead linear attention and demonstrated how it acquires ICL abilities in training. We begin with a simple setting of linear attention with merged key and query trained for in-context linear regression, following the setting in seminal works (Von Oswald et al., 2023;Ahn et al., 2023;Zhang et al., 2024a). We show an abrupt loss drop in training and give an analytical time-course solution in the case of a white input token covariance and small initialization. However, a single abrupt loss drop does not fully capture the evolution of ICL in training practical transformers, where the abilities continue to develop throughout training (Xia et al., 2023;Park et al., 2025). We thus extend our analysis to a parametrization closer to the attention in practical transformers: attention with separate key and query. In the separate case, we find that the loss exhibits saddle-to-saddle dynamics with multiple abrupt drops. The ICL ability evolves progressively, manifesting as implementing principal component regression in context, with the number of principal components increasing over training time. We thus characterize how the linear attention model develops increasingly sophisticated ICL abilities in gradient descent training.
this section cite: ['b74', 'b1', 'b79', 'b55']

Section: References
Ref_id:b0 Title: Context-scaling versus task-scaling in in-context learning Year: (2024)
Ref_id:b1 Title: Transformers learn to implement preconditioned gradient descent for in-context learning Year: (2023)
Ref_id:b2 Title: Linear attention is (maybe) all you need (to understand transformer optimization) Year: (2024)
Ref_id:b3 Title: What learning algorithm is in-context learning? investigations with linear models Year: (2023)
Ref_id:b4 Title: Dual process learning: Controlling use of in-context vs. inweights strategies with weight forgetting Year: (2025)
Ref_id:b5 Title: Adversarial robustness of in-context learning in transformers for linear regression Year: (2024)
Ref_id:b6 Title: A convergence analysis of gradient descent for deep linear neural networks Year: (2019)
Ref_id:b7 Title: Max-margin token selection in attention mechanism Year: (2023)
Ref_id:b8 Title: Neural networks as kernel learners: The silent alignment effect Year: (2022)
Ref_id:b9 Title: Transformers as statisticians: Provable in-context learning with in-context algorithm selection Year: (2023)
Ref_id:b10 Title: Transformers learn through gradual rank increase Year: (2023)
Ref_id:b11 Title: When can transformers reason with abstract symbols? Year: (2024)
Ref_id:b12 Title: Infinite limits of multi-head transformer dynamics Year: (2024)
Ref_id:b13 Title: Language models are few-shot learners Year: (2020)
Ref_id:b14 Title: Toward understanding in-context vs. in-weight learning Year: (2025)
Ref_id:b15 Title: Data distributional properties drive emergent in-context learning in transformers Year: (2022)
Ref_id:b16 Title: Sudden drops in the loss: Syntax acquisition, phase transitions, and simplicity bias in MLMs Year: (2024)
Ref_id:b17 Title: Provably learning a multi-head attention layer Year: (2025)
Ref_id:b18 Title:  Year: ()
Ref_id:b19 Title: Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality Year: (2024-07-03)
Ref_id:b20 Title: Unveiling induction heads: Provable training dynamics and feature learning in transformers Year: ()
Ref_id:b21 Title: On lazy training in differentiable programming Year: (2019)
Ref_id:b22 Title: Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced Year: (2018)
Ref_id:b23 Title: Finite sample analysis and bounds of generalization error of gradient descent in in-context linear regression Year: (2024)
Ref_id:b24 Title: The evolution of statistical induction heads: In-context learning markov chains Year: (2024)
Ref_id:b25 Title: Trained transformer classifiers generalize and exhibit benign overfitting in-context Year: (2025)
Ref_id:b26 Title: Transformers learn to achieve second-order convergence rates for in-context linear regression Year: (2024)
Ref_id:b27 Title: Effect of batch learning in multilayer neural networks Year: (1998)
Ref_id:b28 Title: What can transformers learn in-context? a case study of simple function classes Year: (2022)
Ref_id:b29 Title: Dynamic metastability in the self-attention model Year: (2024)
Ref_id:b30 Title: Abrupt learning in transformers: A case study on matrix completion Year: (2024)
Ref_id:b31 Title: In-context linear regression demystified: Training dynamics and mechanistic interpretability of multi-head softmax attention Year: (2025)
Ref_id:b32 Title: Learning to grok: Emergence of in-context learning and skill composition in modular arithmetic tasks Year: (2024)
Ref_id:b33 Title: Eureka-moments in transformers: Multi-step tasks reveal softmax induced optimization problems Year: (2024-07)
Ref_id:b34 Title: Transformers learn to implement multi-step gradient descent with chain of thought Year: ()
Ref_id:b35 Title: Non-asymptotic convergence of training transformers for next-token prediction Year: (2024-07)
Ref_id:b36 Title: A theoretical analysis of self-supervised learning for vision transformers Year: ()
Ref_id:b37 Title: From self-attention to Markov models: Unveiling the dynamics of generative transformers Year: (2024-07)
Ref_id:b38 Title: LoRA training in the NTK regime has no spurious local minima Year: (2024-07)
Ref_id:b39 Title: Vision transformers provably learn spatial structure Year: (2022)
Ref_id:b40 Title: Gradient descent aligns the layers of deep linear networks Year: (2019)
Ref_id:b41 Title: Unveil benign overfitting for transformer in vision: Training dynamics, convergence, and generalization Year: (2024)
Ref_id:b42 Title: Optimizing attention with mirror descent: Generalized maxmargin token selection Year: (2024)
Ref_id:b43 Title: Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape Year: (2024-07)
Ref_id:b44 Title: Geometry of linear convolutional networks Year: (2022)
Ref_id:b45 Title: Function space and critical points of linear convolutional networks Year: (2024)
Ref_id:b46 Title: Is attention required for ICL? exploring the relationship between model architecture and in-context learning ability Year: (2024)
Ref_id:b47 Title: Fine-grained analysis of incontext linear estimation: Data, architecture, and beyond Year: (2024)
Ref_id:b48 Title: One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention Year: (2024)
Ref_id:b49 Title: Local to global: Learning dynamics and effect of initialization for transformers Year: (2024)
Ref_id:b50 Title: Progress measures for grokking via mechanistic interpretability Year: (2023)
Ref_id:b51 Title: Differential learning kinetics govern the transition from memorization to generalization during in-context learning Year: (2025)
Ref_id:b52 Title: How transformers learn causal structure with gradient descent Year: (2024-07)
Ref_id:b53 Title: -context learning and induction heads Year: (2022)
Ref_id:b54 Title: Slow motion of gradient flows Year: (2007)
Ref_id:b55 Title: Algorithmic phases of in-context learning Year: (2025)
Ref_id:b56 Title: The mechanistic basis of data dependence and abrupt learning in an in-context classification task Year: (2024)
Ref_id:b57 Title: Learning and transferring sparse contextual bigrams with linear transformers Year: (2024)
Ref_id:b58 Title: A distributional simplicity bias in the learning dynamics of transformers Year: (2024)
Ref_id:b59 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2014)
Ref_id:b60 Title: A mathematical theory of semantic development in deep neural networks Year: (2019)
Ref_id:b61 Title: Linear transformers are secretly fast weight programmers Year: (2021-07)
Ref_id:b62 Title: Exponential convergence time of gradient descent for one-dimensional deep linear neural networks Year: (2019-06-28)
Ref_id:b63 Title: Implicit regularization of gradient flow on one-layer softmax attention Year: (2024)
Ref_id:b64 Title: The transient nature of emergent in-context learning in transformers Year: (2023)
Ref_id:b65 Title: What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation Year: (2024-07)
Ref_id:b66 Title: Strategy coopetition explains the emergence and transience of in-context learning Year: (2025)
Ref_id:b67 Title: Unraveling the gradient descent dynamics of transformers Year: (2024)
Ref_id:b68 Title: Transformers as support vector machines Year: (2024)
Ref_id:b69 Title: Scan and snap: Understanding training dynamics and token composition in 1-layer transformer Year: (2023)
Ref_id:b70 Title: MLPs learn in-context on regression and classification tasks Year: (2025)
Ref_id:b71 Title: Implicit bias and fast convergence rates for self-attention Year: (2025)
Ref_id:b72 Title: Attention is all you need Year: (2017)
Ref_id:b73 Title: Linear transformers are versatile in-context learners Year: (2024)
Ref_id:b74 Title: Transformers learn in-context by gradient descent Year: (2023-07)
Ref_id:b75 Title: How transformers implement induction heads: Approximation and optimization analysis Year: (2024)
Ref_id:b76 Title: Transformers provably learn sparse token selection while fullyconnected nets cannot Year: (2024-07)
Ref_id:b77 Title: Kernel and rich regimes in overparametrized models Year: (2020-07)
Ref_id:b78 Title: How many pretraining tasks are needed for in-context learning of linear regression? Year: (2024)
Ref_id:b79 Title: Training trajectories of language models across scales Year: (2023-07)
Ref_id:b80 Title: Learning linear attention in polynomial time Year: (2024)
Ref_id:b81 Title: Trained transformers learn linear models in-context Year: (2024)
Ref_id:b82 Title: In-context learning of a linear transformer block: Benefits of the mlp component and one-step gd initialization Year: (2024)
