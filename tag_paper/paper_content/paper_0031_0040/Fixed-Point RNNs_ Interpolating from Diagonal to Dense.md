Title: Fixed-Point RNNs: Interpolating from Diagonal to Dense
Abstract: Linear recurrent neural networks (RNNs) and state-space models (SSMs) such as Mamba have become promising alternatives to softmax-attention as sequence mixing layers in Transformer architectures. Current models, however, do not exhibit the full state-tracking expressivity of RNNs because they rely on channel-wise (i.e. diagonal) sequence mixing. In this paper, we investigate parameterizations of a large class of dense linear RNNs as fixed-points of parallelizable diagonal linear RNNs. The resulting models can naturally trade expressivity for efficiency at a fixed number of parameters and achieve state-of-the-art results on the state-tracking benchmarks A 5 and S 5 , while matching performance on copying and other tasks.

Section: 
State-space models (SSMs) and other new efficient recurrent token mixers are becoming a popular alternative to softmax attention in language modeling (Gu & Dao, 2024) as well as in other applications such as vision (Liu et al., 2024) and DNA processing (Nguyen et al., 2024). Inspired by linear input-controlled filtering, these models can be expressed as carefully parametrized linear recurrent neural networks (RNNs) with input-dependent, diagonal state transition:
h t = diag(a t )h t-1 + B t x t(1)
Compared to classical RNNs such as LSTMs (Hochreiter & Schmidhuber, 1997), in Eq. ( 1) the relation between the previous hidden state h t-1 and the current h t is linear and its coefficient a t does not depend on the hidden states. These choices allow SSMs such as Mamba (Gu & Dao, 2024) to be computed through efficient parallel methods during training. Furthermore, they are easier to optimize than classical RNNs, thanks to stable and efficient reparametrizations available for diagonal transitions (Orvieto et al., 2023;Zucchet & Orvieto, 2024) -techniques that are significantly more difficult to apply effectively in the classical setting (Arjovsky et al., 2016;Helfrich et al., 2018). At test time, they are faster than classical Transformers on long sequences due to their recurrent nature.
Though modern linear RNNs have shown promise in practice, recent theoretical studies suggest that using dense, input-dependent transition matrices (i.e. replacing diag(a t ) with a dense A t ) could present an opportunity to improve expressivity and unlock performance on challenging tasks. In particular, Cirone et al. (2024b) prove that dense selective SSMs are endowed with the theoretical expressivity of classical non-linear RNNs such as LSTMs. As shown by Merrill et al. (2024) and Sarrof et al. (2024), such gained expressivity proves to be particularly useful in state-tracking applications where models are expected to maintain and extrapolate a complex state of the world. Since state-tracking is naturally expressed by non-linear RNNs but provably unavailable to channelwise sequence mixers such as SSMs or Transformers, Merrill & Sabharwal (2023) speculate on a fundamental tradeoff between parallelism and expressivity. This discussion sparked interest in non-diagonal recurrences and parallelizable architectures capable of state-tracking (Grazzi et al., 2024;Terzic et al., 2025;Schöne et al., 2025;Peng et al., 2025;Siems et al., 2025).
When designing new architectures involving dense selective yet linear state transitions of the form h t = A t h t-1 + B t x t , two fundamental concerns arise:
1. What should the parametric form for A t , as a function of the input be? How can we guarantee this parametrization induces a stable recurrence, like in standardfoot_0 SSMs?
2. How does a parametrization balance between expressivity and parallelism? Which assumptions on the structure of A t enable efficient computation, and how do they interact with expressivity?
Perhaps the first approach tackling the above questions was DeltaNet (Schlag et al., 2021a;Yang et al., 2024b) with a block-diagonal and orthogonal therefore, stable state transition structure, where each block is parametrized by a Householder matrix. The parallelizable algorithm, was then extended to include negative eigenvalues (Grazzi et al., 2024), gates (Yang et al., 2025), and most recently products of Householders (Siems et al., 2025). Such choices, leading to increased expressivity as exemplified by their state-tracking and length generalization capabilities, are motivated mainly by hardware considerations: Householder-based mixing can be implemented efficiently on GPUs as linear attention via WY-representations and the UT transform (Yang et al., 2024b).  [5,50]. Our single layer FP-Mamba-H with mixer reflections r ∈ {1, 2, 4} is compared to baselines of increasing depth ∈ {1, 2, 4, 6, 8}. FP-Mamba-H is the only model capable of solving both the state-tracking and the copy task.
While the works above offer exciting practical strategies for boosting capabilities at a relatively low additional computational cost, they fall short in exploring the sea of intriguing options for dense transitions and hence, in thoroughly answering questions (1) and (2) above.
Unfortunately, this is not an easy task: although linear recurrences are theoretically parallelizable across sequence length (Martin & Cundy, 2018), parallelizing dense RNNs efficiently is not trivial due to increased memory I/O. These thoughts inspired us to change our viewpoint: instead of designing an algorithm which adds a fraction of non-diagonal processing to a model, here, we look for a strategy to navigate the parallelism tradeoff towards a truly dense object.
Motivated by the idea of designing a parallelizable general-purpose method to implement new dense RNN variations, in this paper we devise a new adaptive computation strategy which allows to interpolate between fast recurrent diagonal RNNs and dense recurrences with arbitrary preselected structure. Instead of parametrizing the dense RNN layer as an explicit function h = F θ (x), we build on the literature of equilibrium/implicit models (Bai et al., 2019;Ghaoui et al., 2021) to parametrize it implicitly as a solution h * to a fixed-point equation h = f θ (x, h) involving only a diagonal RNN.
As described in Fig. 3a, we solve for h * using a fixed-point iteration of diagonal RNN evaluations f θ .
A fundamental question some readers might rightfully ask, is the following: "what is the advantage of iterating a single layer in depth compared to depth-stacking multiple SSM, e.g. Mamba layers?"
We claim one advantage comes from having access to the limiting dense object. As showcased by Fig. 2, this allows to adaptively provide the required expressivity for a fixed set of parameters without any a priori choice on the network size.
this section cite: ['b36', 'b42', 'b27', 'b43', 'b70', 'b1', 'b26', 'b40', 'b50', 'b39', 'b22', 'b61', 'b55', 'b47', 'b57', 'b22', 'b69', 'b57', 'b38', 'b3', 'b17']

Section: Summary.
In this work, we propose a recipe to design a general class of dense linear RNNs as fixed points of corresponding diagonal linear RNNs. Our contributions are:
1. We develop the framework of Fixed-Point RNNs to adaptively trade parallelism for expressivity using the number of fixed-point iterations (Fig. 1). 2. We achieve a stable parametrization of a dense RNN via a carefully designed diagonal RNN. 3. The framework allows for easy integration of both non-linear hidden state dependence and linear attention based matrix-valued formulations. This way, our FP-Mamba unites previously isolated capabilities of recurrent computation and memory (Fig. 2).
this section cite: []

Section: Background
Since their introduction (Rumelhart et al., 1986;Elman, 1990), RNNs have significantly contributed to the evolution of machine learning methods for sequential data (Hochreiter & Schmidhuber, 1997;Jaeger, 2001). But despite their theoretical promise of Turing-completeness (Siegelmann & Sontag, 1992), recurrent models fell out of fashion due to two significant challenges: they are inherently sequential, and notoriously difficult to train (Hochreiter et al., 2001;Pascanu et al., 2013). The recent advancements of linear RNNs (Gu & Dao, 2024) suggest a way forward to combine the scalability of Transformers (Vaswani et al., 2017) with the expressivity of classical RNNs (Cirone et al., 2024b).
The key challenge here is the stable and efficient parametrization of a linear RNN layer with a time-varying recurrent transition matrix. In this paper we are exploring first steps towards this goal.
this section cite: ['b49', 'b15', 'b27', 'b30', 'b56', 'b28', 'b44', 'b63']

Section: Dense Selective RNN.
Traditionally, RNNs are parametrized as either time-invariant, non-linear, or element-wise system. To the best of our knowledge, a time-variant, dense, and linear RNN parametrization has been of mild interest at best. To understand why, consider the general form
F θ : x → h, h t = A t h t-1 + B t x t ,(2)
where A t ∈ R d×d corresponds to the time-varying state transition matrix, B t ∈ R d×d is the input transformation matrix, h t ∈ R d denotes the hidden state, and x t ∈ R d is the input for t < T steps.
For a given sequence of A t , the complexity of a forward pass is O(T d 2 ) in memory and O(T ) sequential steps. Although such a linear RNN could also be computed in O(log T ) sequential steps using a parallel scan algorithm (Martin & Cundy, 2018), this would require materializing matrixmatrix multiplications at cost O(d 3 ). An issue in both scenarios, however, is the parametrization of A t as time-varying, i.e. input-or even hidden state-dependent matrices. In general, this requires a map M : d → d 2 with potentially d × d 2 parameters, and O(T d 3 ) time complexity. While structured dense matrix representations for A t could potentially present a remedy, they come with additional challenges: (1) In order to guarantee expressivity, the A t cannot be co-diagonalizable such as for example Toeplitz matrices (Cirone et al., 2024b).
(2) In order to guarantee stability of the dynamical system, the spectral radius ρ(A t ) needs to be less than, but still close to 1 for long-range interactions (Orvieto et al., 2023).
(3) The matrix structure needs to be closed under multiplications to enable parallel scans without having to materialize dense representations at O(T d 2 ) memory cost.
this section cite: ['b38', 'b43']

Section: Related Works.
Improving the trainability of classical non-linear RNNs has a long history. For example, Arjovsky et al. (2016) and Helfrich et al. (2018) investigate parameterizations to stabilize their spectral radius with structured matrix representations, while Lim et al. (2024) and Gonzalez et al. (2024) propose iterative methods to parallelize their computation. In this work, however, we focus on stabilizing and parallelizing a time-variant, dense, linear RNN. Improving the limited expressivity of existing diagonal linear RNNs is the focus of a few recent works, e.g. by Grazzi et al. (2024) and Siems et al. (2025). In contrast, we investigate a wide class of structured parameterizations for dense RNNs where the additional cost is adaptively chosen depending on the task. In concurrent work, Schöne et al. (2025) propose an iterative method similar to ours, but as opposed to our carefully designed implicit dense RNN layer, they focus on scaling implicit causal models of existing multi-layer architectures on language. For a more extensive literature review, we refer the reader to App. A.
this section cite: ['b1', 'b26', 'b35', 'b19', 'b22', 'b57', 'b55']

Section: Fixed-Points as an RNN Layer
In this section, we introduce an implicit parameterization for a family of dense RNNs F θ (x) which describes its output by a solution h * ∈ R T ×d to the fixed-point equation h = f θ (x, h) (Sec. 3.1). Then, we discuss how to find the solution h * using fixed-point iterations (Sec. 3.2) and the algorithmic implications (Sec. 3.4) of the FP-RNN framework in light of the challenges outlined in Sec. 2. Finally, we briefly touch on how to train an implicitly dense model F θ (x) with gradient descent (Sec. 3.5).
this section cite: []

Section: From Explicit to Implicit Parameterization
We start by designing a diagonal RNN f θ (x, h) such that the solution h * to its fixed-point equation h = f θ (x, h) implicitly represents a dense RNN h * = F θ (x). Consider the factorized parametrization of A t similar to the one introduced by Helfrich et al. (2018) for non-linear and time-invariant RNN:
F θ : x → h * , h * t = Q -1 t Λ t h * t-1 + B t x t .(3)
Separating A t into a diagonal matrix Λ t ∈ R d×d and a non-diagonal invertible mixing matrix Q t ∈ R d×d allows to describe h * by only a diagonal transition Λ t by reformulating Eq. 3 to
h * t = Λ t h * t-1 + Q t B t x t + (I -Q t )h * t .(4)
This means that the states h * = F θ (x) of the dense linear RNN can be implicitly described by the fixed-point h * = f θ (x, h * ) of a corresponding diagonal linear RNN of the following form:
f θ : (x, h) → h ′ , h ′ t = Λ t h ′ t-1 + Q t B t x t + (I -Q t )h t .(5)
In other words, if we could find the fixed-point h * = f θ (x, h * ) ∈ R T ×d for the diagonal RNN defined in Eq. 5, then h * would describe the states of a corresponding dense RNN h * = F θ (x). Motivated by this insight, in Sec. 3.2 we carefully parametrize the diagonal RNN f θ (x, h) and its channel mixer Q t such that a computable fixed-point exists.
this section cite: ['b26']

Section: The Fixed-Point Iteration
Solving fixed-point equations such as h = f θ (x, h), is perhaps one of the most well-studied problems in mathematics (Granas et al., 2003). In the context of deep learning, the literature on Neural ODEs (Chen et al., 2018) and Deep Equilibrium Models (Bai et al., 2019;Ghaoui et al., 2021) investigates fixed-point methods for implicit parametrizations of neural networks. A straightforward, yet effective method computes the forward pass by simply rolling out the fixed-point iteration. In the context of solving h * = f θ (x, h * ), this corresponds to introducing an iteration in depth h ℓ = f θ (x, h ℓ-1 ).
Denoting ℓ as the current iteration in depth (i.e., over the layer dimension), and t as the current iteration in time (i.e., over the sequence dimension), the iteration starts at h 0 t = 0 and proceeds with
h ℓ t = Λ t h ℓ t-1 + Q t B t x t + (I -Q t )h ℓ-1 t .(6)
Intuitively, this iteration mixes information with interleaved channel mixing (with Q t ) and sequence mixing (with Λ t ) until convergence towards the hidden states of an implicit dense RNN F θ (cf. 3a).
The difficulty with such an iteration in depth and time is that the recurrent dynamics could explode without proper stabilization. While the recurrence in time can be stabilized with RNN techniques (Zucchet & Orvieto, 2024) such as an input gate I -Λ t , the recurrence in depth, however, could still diverge if f θ (x, h) does not have an attracting fixed-point (Granas et al., 2003). In order to design a diagonal linear RNN f θ (x, h) which is guaranteed to have an attracting fixed-point, we make use of Banach (1922)'s theorem. In our context, the theorem states that f θ (x, h) converges to a fixed-point from any initialization h 0 if it has a Lipschitz constant < 1 in h. For a fixed-point RNNs with input gate I -Λ, we present the following theorem:
Theorem 3.1. Let f θ (x, h) be the diagonal linear RNN with input-independent Λ and Q
f θ : (x, h) → h ′ , h ′ t = Λh ′ t-1 + (I -Λ) (QB t x t + (I -Q) h t ) .(7)
If ||Λ|| 2 < 1 and ||I -Q|| 2 < 1, then f θ (x, h) has a Lipschitz constant < 1 in h. Proof in App. B.1.
Intuitively, Thm. 3.1 states two conditions for stable parametrization of an implicitly dense RNN F θ : (1) the recurrence in time needs to be coupled with input normalization and contractive (i.e. ∥Λ∥ 2 < 1).
(2) The recurrence in depth acting on h, i.e. (I -Q t ), needs to be contractive. Together, this guarantees that all sequences h ℓ up to h * throughout the fixed-point iteration do not explode without any explicit assumptions on the spectral radius on A (Arjovsky et al., 2016).
this section cite: ['b20', 'b7', 'b3', 'b17', 'b70', 'b20', 'b5', 'b1']

Section: Parametrization of Q t and Λ t
To satisfy the assumptions required for expressivity in (Cirone et al., 2024b), the implicit transition matrix A t and therefore Λ t and Q t need to be input-controlled (i.e. selective), which could be realized through a linear mapping of the input, i.e. Q t = M(x t ) := reshape(W Q x t ). However, this presents two challenges: how can stability be guaranteed (c.f. Thm. 3.1) and excessive computational cost due to the O(d 3 ) parameters of W Q be avoided? A straight-forward solution lies in structured matrix representations for both the diagonal transition matrix Λ t and the channel mixer Q t .
Inspired by Helfrich et al. (2018), we aim for Q t to be approximately norm-preserving and Λ t to control the eigenvalue scale using a parametrization akin to Mamba or Griffin (Gu & Dao, 2024;De et al., 2024) and normalization (I -Λ t ). For the channel mixers Q t , we consider the structures:
• Diagonal Plus Low Rank (DPLR): Q t = M(x t ) := Ir i=1 α it • ūit ū⊤ it , for rank r.
• Householder Reflections (H): Q t = M(x t ) := r i=1 I -α it • ūit ū⊤ it , for r reflections.
• Kronecker (K): Q t = M(x t ) := I -( K1 t ⊗ K2 t ), where ⊗ denotes the Kronecker product.
This allows to reduce the size of the input-dependent parameters α it , ūit , and Ki t to O(d), and consequently reduce the size of the linear map W Q to O(d 2 r) and O(d 2 ). In order to guarantee stability, the condition ||I -Q|| 2 < 1 can be enforced by scaling α it , ūit , and Ki t appropriately. For more details about the channel mixer variants, please refer to App. C. Fig. 3b, we compare different channel mixer variants and observe that the Kronecker structure seems to be most appropriate the state-tracking task A 5 .
this section cite: ['b26', 'b13']

Section: Algorithmic Implications
Recall from Sec. 2 that an explicitly parametrized dense selective RNN can only be parallelized under strict assumptions on its structure and runs otherwise in O(T ) sequential steps. However, a parallelizable structure is given by the element-wise, diagonal transition Λ t of a diagonal RNN (Martin & Cundy, 2018). Since such a diagonal RNN is called ℓ * -times as a subroutine of the fixed-point iteration in Eq. 6, a fixed-point RNN runs in O(ℓ * • log T ) sequential steps. This means that the implicit parametrization -as opposed to explicit or non-linear parametrizations-allows to decouple the number of sequential steps ℓ * from the sequence length T itself, and trade parallelism for expressivity. This insight suggests an opportunity to introduce a non-linear computation for every sequential step, like in classical RNNs. Concretely, we investigate channel mixers M(x t + h ℓ-1 t-1 ) which are a function of both the input x t and the hidden state h ℓ-1 t-1 from the previous iteration (in both time and depth) without degrading parallelizability. In Fig. 3b, we compare channel mixers with and without hidden state dependence and observe that this indeed improves sequence length generalization.
Summarizing the results so far, we arrive at an updated recurrence with hidden state dependence:
h ℓ t = λ ℓ t ⊙ h ℓ t-1 + (1 -λ ℓ t ) ⊙ (Q ℓ t B ℓ t x t + (I -Q ℓ t )h ℓ-1 t ),(8)
where we use ⊙ to highlight the parallelizability of the element-wise product. We would like to note that due to the normalization (I -Λ t ), the corresponding dense RNN F θ is not explicitly representable anymore as discussed in App. B.2. Furthermore, for the time-varying parametrization in Eq. 8, the convergence guarantees may be weaker and solutions h * could be non-unique due to the hidden state dependence. In practice, we iterate until ||h ℓ -h ℓ-1 ||∞ ||h ℓ ||∞ < 0.1 and observe that the conditions of Thm. 3.1 are strong enough to reach convergence within a finite number of iterations ℓ * as evidenced by Fig. 3c. Interestingly, the model navigates the parallelism tradeoff (Merrill & Sabharwal, 2023) and adaptively increases its sequential computation for harder tasks.
this section cite: ['b38', 'b39']

Section: Optimizing Fixed-Point RNNs
One advantage of converging to a fixed-point as opposed to general layer looping lies in model training. Since the gradient with respect to h 0 is not needed, implicit differentiation can be used to avoid storing and backpropagating through the computational graph of the fixed-point iteration, as discussed by Liao et al. (2018), Bai et al. (2019), and in App. B.3. In practice, truncated backpropagation of the last k iterations suffices to approximate the gradient through the full iteration
J * x ≈ J x (h ℓ * -k ) • . . . • J x (h ℓ * ).
For Fixed-Point RNNs we observe that computing the gradient only at the fixed-point (k = 0), is enough to stabilize training. This means that compared to a single diagonal RNN layer, Fixed-Point RNNs incur no memory overhead and only sequential overhead in the forward pass but not in the backward pass.
We hypothesize that this is possible because f θ (x, h) is a mostly linear object as opposed to multilayer implicit models such as (Schöne et al., 2025). Furthermore, we observe that hidden state dependence M(x t + h ℓ-1 t-1 ) particularly helps with gradient-based optimization. We credit this to the symmetry between the gradients w.r.t. x and h, and formalize this in the following theorem:  In the previous section we introduced the FP-RNN framework on a small RNN with vector hidden state. Now, we extend it to modern matrix state RNNs in Sec. 4.1 and parametrize a dense variant of Mamba (Gu & Dao, 2024) in Sec. 4.2. A detailed description of the architecture is available in App. C.2. We compare the architecture to the baselines Mamba (Gu & Dao, 2024), Mamba-2 (Dao & Gu, 2024), Gated DeltaNet (Yang et al., 2025), and LSTM (Hochreiter & Schmidhuber, 1997) on the copy task introduced by Jelassi et al. (2024) in Sec. 4.3 and state-tracking introduced by Merrill & Sabharwal (2023) in Sec. 4.4. In order to keep the number of layers at the same order of magnitude, we use two layers for the diagonal linear RNN baselines and one layer for FP-Mamba and LSTM. Finally, we discuss the required number of fixed-point iterations in the context of state-tracking and language modeling in Sec. 4.5.
(c) S5 -FP-Mamba Mixer Qt H -r = 1 H -r = 2 H -r = 4 DPLR -r = 1 DPLR -r = 2 DPLR -r = 4 K 0 20 40 Sequence Length (d) S5 -Baselines Mamba Mamba-2 GatedDeltaNet DeltaProduct LSTM
this section cite: ['b34', 'b3', 'b55', 'b69', 'b27', 'b31', 'b39']

Section: Introducing Matrix States
Memory capacity is an important consideration in RNNs. In preliminary experiments, we notice a clear gap between the performance of a Fixed-Point RNNs and Mamba in terms of copying ability. We attribute this difference in performance to Mamba's state-expansion which endows it with matrix hidden states similar to linear attention, DeltaNet, or mLSTM (Katharopoulos et al., 2020;Schlag et al., 2021a;Beck et al., 2024). In simple terms, these models use an outer product of an inputdependent vector b t ∈ R dstate (i.e. the key) and the input vector x t ∈ R dinner (i.e. the value) as an input to a matrix-valued recurrence with hidden state and transition gate H t , λ t ∈ R dstate×dinner . The hidden state is then contracted with another input-dependent vector c t ∈ R dstate (i.e. the query) to get the output
y ⊤ t = c ⊤ t H t ∈ R dinner : H t = λ t ⊙ H t-1 + b t x ⊤ t ,(9)
This matrix-valued recurrence introduces some challenges to our fixed-point framework. Specifically, in order to mix all the channels over the entirety of the state elements, the mixer has to be a fourth-order
tensor Q t ∈ R dstate×dinner×dstate×dinner in H ℓ t = λ t ⊙ H ℓ t-1 + Q t • b t x ⊤ t + (I -Q t ) • H ℓ-1 t ,(10)
where • denotes the tensor contraction einsum(klij, ij → kl) with fourth-order identity tensor I of the same shape as Q t . Certainly, computing the fixed-point introduced in Eq. 10 is very challenging both in terms of computation and memory. As we will confirm in Sec. 4.2, one solution is to pass the contracted output y t between fixed-point iterations
H ℓ t = λ t ⊙ H ℓ t-1 + b t (Q t x t ) ⊤ + b t (I -Q t ) y ℓ-1 t ⊤ .
(11) This implicitly factorizes the tensor mixer Q t into separately mixing along dimension d inner which is used for better expressivity, and dimension d state which is used for better memory capacity.
this section cite: ['b32', 'b6']

Section: FP-Mamba Iteration
Let us apply the the fixed-point RNN framework to the Mamba parametrization. We represent the hidden state as H ℓ t , where t is the token index (i.e., indexing over the sequence dimension), and ℓ is the fixed-point iteration index (i.e., indexing over the depth dimension). The same notation is used for other variables to emphasize when they depend on the input and hidden state of the current iteration. We propose the following iteration to adapt Mamba with notation from App. C.1 to the fixed-point mechanism for matrix state RNNs in Eq. 11:
H ℓ t = λ t ⊙ H ℓ t-1 + bℓ t ∆ t Q ℓ t x t ⊤ + bℓ t ∆ t I -Q ℓ t y ℓ-1 t ⊤ , y ℓ t ⊤ = (c ℓ t ) ⊤ H ℓ t .(12)
L2-normalizing bℓ t and cℓ t allows to limit the Lipschitz constant according to Theorem 3.1. Furthermore, we replace the normalization term (1 -λ t ) with Mamba's normalization term ∆ t . Expanding y ℓ-1 t yields the recurrence on the matrix state
H ℓ t = λ t ⊙ H ℓ t-1 + bℓ t (∆ t Q ℓ t x t ) ⊤ + bℓ t (c ℓ-1 t ) ⊤ H ℓ-1 t (I -Q ℓ t ) ⊤ ∆ t ,(13)
where the last term nicely illustrates the two components which mix the channels of the hidden states: the low-rank matrix bℓ t (c ℓ-1 t ) ⊤ mixes over the dimension d state , while (I -Q ℓ t ) ⊤ mixes over the dimension d inner . This factorization significantly simplifies the fourth-order tensor mixer formulation introduced in Eq. 10, remains expressive as discussed in App. F, and performs well in practice.
Finally, Eq. 12 can be computed as Mamba with an adjusted input xℓ
t = Q ℓ t x t -y ℓ-1 t + y ℓ-1 t , H ℓ t = λ t ⊙ H ℓ t-1 + bℓ t ∆ t xℓ t ⊤ . (14
)
In other words, one fixed-point step consists of a channel mixing using Q t , followed by a sequence mixing using Mamba. This separation of concerns allows to speed up the parallel recurrence in time using the Mamba implementation. To find a fixed-point, the two phases are repeated until
∥y ℓ -y ℓ-1 ∥∞ ∥y ℓ ∥∞ < 0.1 is satisfied.
After these ℓ * iterations, required for the model to converge to a fixed-point, H * t and y * t present the hidden state and output of the dense matrix-valued RNN F θ . Similar to Mamba, we apply a gated linear unit g t ∈ R dinner to the output, which we observe to provide a slight improvement in performance when present within the fixed-point loop: ỹℓ t = g t ⊙ y ℓ-1 t .
Dependence on y ℓ
H -r = 1 H -r = 2 H -r = 4 DPLR -r = 1 DPLR -r = 2 DPLR -r =
this section cite: []

Section: Shifted Hidden State Dependence y ℓ-1 t-1
In preliminary experiments, we observe that even the Fixed-Point RNN with input-dependent parameters and matrix state akin to Mamba-1 is outperformed by Mamba-2 or DeltaNet (Dao & Gu, 2024;Yang et al., 2024b) on a copy task. Inspired by the short convolution in Mamba, we investigate the effect of augmenting the input-dependence of parameters λ ℓ t , b ℓ t , c ℓ t , and Q ℓ t at iteration ℓ with a shifted hidden state dependence. In practice, this means that these are linear functions of x t as well as the shifted previous iterate in depth y ℓ-1 t-1 . We refer the reader to App. C.2 for the exact formulation of the dependency.
In Tab. 1, we ablate the hidden state dependence for various combinations of λ t , b t , c t , and a Householder Q t . Observe that the dependence of b t and c t is crucial to enable the model to copy. In App. C.4, we discuss why this dependence of b t and c t could be important for copying. If additionally λ t and Q t depend on y ℓ-1 t-1 , the copy task is essentially solvable at ×2 length generalization. We therefore adopt the hidden state dependence for all components in FP-Mamba.
In Fig. 5, we evaluate length generalization on the copying task. While the best-performing baseline Gated DeltaNet is specifically designed for associative recall tasks (Yang et al., 2025), both Mamba 1 and 2 struggle with ×2 generalization. FP-Mamba closes this gap and proves the effectiveness of our proposed modifications for better memory. We would like to highlight that the number of fixed-point iterations ℓ * (gray vertical line) in FP-Mamba is well below the maximum sequence length.
this section cite: ['b69']

Section: State-Tracking
200 300 400 500 600 700 800 900 Train Time 0 10 20 30 40 50
Max Test Seq Len (Acc In Fig. 4, we evaluate the state-tracking capabilities of FP-Mamba with Kronecker, Householder, and DPLR channel mixers of r ∈ {1, 2, 4} reflections or ranks, respectively. In particular, we compare our FP-Mamba to the baselines with regards to their length generalization beyond the training sequence length 16. As expected, LSTM solves A 5 and S 5 , while Mamba and Mamba-2 are not able to learn it even at the training sequence length. Similar to Fig. 3b, the Kronecker structure seems to be the most suitable for the task. But FP-Mamba based on Householders also improves in terms of sequence length generalization presumably due to its improved memory. A comparison to the recent DeltaProduct (Siems et al., 2025) on training sequence length 128 is available in App. E.2. A fixed-point iteration in the forward pass inevitably introduces sequential overhead to the computation of a model. While this might be acceptable for sequential generation at test time, reduced parallelism can be inhibiting at training time. In Fig. 1, we therefore evaluate FP-Mamba-H on A 5 with limited number of fixed-point iterations at training time ℓ max ∈ {2, 4, 8, 16}. We observe that the performance decreases once ℓ max is lower than the training sequence length of 16. In Fig. 6, we confirm that the resulting longer training times are indeed required for good length generalization. However, as opposed to baselines of increasing depth ∈ {1, 2, 4, 6, 8}, fixed-point iterations gain from the additional training time. Furthermore, there is room to improve efficiency, as suggested by a simple randomization scheme (gray stars) where ℓ max ∼ Γ(4, 1) is sampled from a Gamma distribution with mean 4 for every batch. But most importantly, the effective number of fixed-point iterations depends on the difficulty of the task. Indeed, Fig. 7 shows that the model automatically adapts to using less fixed-point iterations on language pretraining at context length 2048. Similarly, on copying (Fig. 5) and and modular arithmetic (Fig. 10), we observe that the required number of fixed-point iterations ℓ * is well below the sequence length T . This suggests that the model adapts to O(T ) complexity on simpler tasks when the full state-tracking expressivity is not required.
> 90%) r = 4 r = 1 r = 2 r = 1 r = 2 r = 4 FP-Mamba-H FP-Mamba-H -(4, 1) Mamba Mamba2 GatedDeltaNet
this section cite: ['b57']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Language model pretraining in pytorch Year: (2024)
Ref_id:b1 Title: Unitary evolution recurrent neural networks Year: (2016)
Ref_id:b2 Title: Simple linear attention language models balance the recall-throughput tradeoff Year: (2024)
Ref_id:b3 Title: Deep equilibrium models Year: (2019)
Ref_id:b4 Title: Stabilizing equilibrium models by jacobian regularization Year: (2021)
Ref_id:b5 Title: Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales Year: (1922)
Ref_id:b6 Title: Extended long short-term memory Year: (2024)
Ref_id:b7 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b8 Title: Skyformer: Remodel self-attention with Gaussian kernel and Nystrom method Year: (2021)
Ref_id:b9 Title: Rethinking attention with performers Year: (2020)
Ref_id:b10 Title: Genus expansion for non-linear random matrix ensembles with applications to neural networks Year: (2024)
Ref_id:b11 Title: Theoretical foundations of deep selective state-space models Year: (2024)
Ref_id:b12 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b13 Title: Mixing gated linear recurrences with local attention for efficient language models Year: (2024)
Ref_id:b14 Title: Universal transformers Year: (2019)
Ref_id:b15 Title: Finding structure in time Year: (1990)
Ref_id:b16 Title: Scaling up test-time compute with latent reasoning: A recurrent depth approach Year: (2025)
Ref_id:b17 Title: Implicit deep learning Year: (2021)
Ref_id:b18 Title: Looped transformers as programmable computers Year: (2023)
Ref_id:b19 Title: Towards scalable and stable parallelization of nonlinear RNNs Year: (2024)
Ref_id:b20 Title: Fixed point theory Year: (2003)
Ref_id:b21 Title: Adaptive computation time for recurrent neural networks Year: (2016)
Ref_id:b22 Title: Unlocking state-tracking in linear RNNs through negative eigenvalues Year: (2024)
Ref_id:b23 Title: Linear-time sequence modeling with selective state spaces Year: (2024)
Ref_id:b24 Title: Efficiently modeling long sequences with structured state spaces Year: (2022)
Ref_id:b25 Title: Universal simulation of stable dynamical systems by recurrent neural nets Year: (2020)
Ref_id:b26 Title: Orthogonal recurrent neural networks with scaled Cayley transform Year: (2018)
Ref_id:b27 Title: Long short-term memory Year: (1997)
Ref_id:b28 Title: Gradient flow in recurrent nets: the difficulty of learning long-term dependencies. A Field Guide to Dynamical Recurrent Neural Networks Year: (2001)
Ref_id:b29 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b30 Title: The "echo state" approach to analysing and training recurrent neural networks-with an erratum note Year: (2001)
Ref_id:b31 Title: Repeat after me: Transformers are better than state space models at copying Year: (2024)
Ref_id:b32 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b33 Title: On the computational power of RNNs Year: (2019)
Ref_id:b34 Title: Reviving and improving recurrent back-propagation Year: (2018)
Ref_id:b35 Title: Parallelizing non-linear sequential models over the sequence length Year: (2024)
Ref_id:b36 Title: Vmamba: Visual state space model Year: (2024)
Ref_id:b37 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b38 Title: Parallelizing linear recurrent neural nets over sequence length Year: (2018)
Ref_id:b39 Title: The parallelism tradeoff: Limitations of log-precision transformers Year: (2023)
Ref_id:b40 Title: The illusion of state in state-space models Year: (2024)
Ref_id:b41 Title: Artificial kuramoto oscillatory neurons Year: (2025)
Ref_id:b42 Title: Sequence modeling and design from molecular to genome scale with Evo Year: (2024)
Ref_id:b43 Title: Resurrecting recurrent neural networks for long sequences Year: (2023)
Ref_id:b44 Title: On the difficulty of training recurrent neural networks Year: (2013)
Ref_id:b45 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b46 Title: Eagle and Finch: RWKV with matrix-valued states and dynamic recurrence Year: (2024)
Ref_id:b47 Title: Rwkv-7 "goose" with expressive dynamic state evolution Year: (2025)
Ref_id:b48 Title: Gated linear RNNs with state expansion Year: (2024)
Ref_id:b49 Title: Sequential thought processes in pdp models Year: (1986)
Ref_id:b50 Title: The expressive capacity of state space models: A formal language perspective Year: (2024)
Ref_id:b51 Title: Reasoning with latent thoughts: On the power of looped transformers Year: (2025)
Ref_id:b52 Title: Linear transformers are secretly fast weight programmers Year: ()
Ref_id:b53 Title: Learning associative inference using fast weight memory Year: ()
Ref_id:b54 Title: Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks Year: (2021)
Ref_id:b55 Title: Implicit language models are rnns: Balancing parallelization and expressivity Year: (2025)
Ref_id:b56 Title: On the computational power of neural nets Year: (1992)
Ref_id:b57 Title: Increasing the expressivity of deltanet through products of householders Year: (2025)
Ref_id:b58 Title: Simplified state space layers for sequence modeling Year: (2023)
Ref_id:b59 Title: Retentive network: A successor to transformer for large language models Year: (2023)
Ref_id:b60 Title: Long range arena: A benchmark for efficient transformers Year: (2020)
Ref_id:b61 Title: On the expressiveness and length generalization of selective state space models on regular languages Year: (2025)
Ref_id:b62 Title: Mimetic initialization helps state space models learn to recall Year: (2024)
Ref_id:b63 Title: Attention is all you need Year: (2017)
Ref_id:b64 Title: An empirical study of Mamba-based language models Year: (2024)
Ref_id:b65 Title: Linformer: Self-attention with linear complexity Year: (2020)
Ref_id:b66 Title: Towards AI-complete question answering: A set of prerequisite toy tasks Year: (2015)
Ref_id:b67 Title: Gated linear attention transformers with hardware-efficient training Year: (2024)
Ref_id:b68 Title: Parallelizing linear transformers with the delta rule over sequence length Year: ()
Ref_id:b69 Title: Gated delta networks: Improving mamba2 with delta rule Year: (2025)
Ref_id:b70 Title: Recurrent neural networks: vanishing and exploding gradients are not the end of the story Year: (2024)
