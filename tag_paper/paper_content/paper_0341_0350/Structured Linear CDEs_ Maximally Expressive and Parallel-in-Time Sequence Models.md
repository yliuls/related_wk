Title: Structured Linear CDEs: Maximally Expressive and Parallel-in-Time Sequence Models
Abstract: This work introduces Structured Linear Controlled Differential Equations (SLiCEs), a unifying framework for sequence models with structured, input-dependent statetransition matrices that retain the maximal expressivity of dense matrices whilst being cheaper to compute. The framework encompasses existing architectures, such as input-dependent block-diagonal linear recurrent neural networks and DeltaNet's diagonal-plus-low-rank structure, as well as two novel variants based on sparsity and the Walsh-Hadamard transform. We prove that, unlike the diagonal statetransition matrices of S4D and Mamba, SLiCEs employing block-diagonal, sparse, or Walsh-Hadamard matrices match the maximal expressivity of dense matrices. Empirically, SLiCEs solve the A 5 state-tracking benchmark with a single layer, achieve best-in-class length generalisation on regular language tasks among parallelin-time models, and match the performance of log neural controlled differential equations on six multivariate time-series classification datasets while cutting the average time per training step by a factor of twenty.

Section: Introduction
Parallel-in-time architectures, such as Transformers and Structured State-Space Models (SSMs), have allowed language models to scale to billions of parameters [96,39]. However, theory and practice agree that these models do not generalise to longer sequences on state-tracking problems, a task that classical Recurrent Neural Networks (RNNs) handle with ease [65,66,60]. Linear Neural Controlled Differential Equations (LNCDEs) are a continuous-time sequence model where the state-transition matrix, or vector field, depends linearly on the input path. This allows for the multiplicative interactions between the hidden state and the input path necessary for gating. Reframing SSMs as LNCDEs, it becomes clear that using a diagonal state-transition matrix severely restricts expressivity [22]. Replacing it with a dense matrix restores maximal expressivity and the ability to state-track, but increases the number of parameters and computational cost from O(d 2 h ) to O(d 3 h ), where d h is the hidden dimension [66,22].
Structured alternatives seek the best of both worlds. Block-diagonal Linear RNNs (LRNNs) [33] and the Diagonal-Plus-Low-Rank (DPLR) structure of DeltaNet [85,101,88] reduce computational cost while preserving some expressivity, with the latter recently shown to be maximally expressive [72]. We generalise and extend these ideas with Structured Linear Controlled Differential Equations (SLiCEs), a unifying framework for structured, input-dependent state-transition matrices. SLiCEs replace the dense state-transition matrix of an LNCDE with an efficient structured variant, such as block-diagonal, sparse, Walsh-Hadamard or DPLR, which maintain the maximal expressivity of dense matrices whilst reducing both the parameter count and computational cost. Contributions 1. Structured Linear Controlled Differential Equations (SLiCEs) are introduced as a common framework for models with structured, input-dependent state-transition matrices. SLiCEs incorporate SSMs such as Mamba [39], LRNNs such as DeltaNet [85] and input-dependent block-diagonal LRNN [33], LNCDEs [22], and two novel structures based on sparse matrices and the Walsh-Hadamard transform. 2. Block-diagonal, sparse, and Walsh-Hadamard SLiCEs are proven to achieve maximal probabilistic expressivity in Theorems 4.1, 4.2, and 4.3, respectively. Previously, such expressivity had only been shown for dense and DPLR matrices [22,72]. 3. A comprehensive empirical evaluation showing that the structure of the state-transition matrix significantly impacts length generalisation on state-tracking problems. The blockdiagonal structure emerges as a promising option, due to its strong empirical results and favourable parallelisation. Furthermore, on six real-world multivariate time-series classification datasets, a block-diagonal SLiCE is shown to match the predictive accuracy of Log-NCDEs, whilst reducing the per-step training time by a factor of twenty. 4. Open-source implementations of SLiCEs in both PyTorch and JAX, along with code to fully reproduce all experiments from this paper. These are available at https://github.com/  Benjamin-Walker/structured-linear-cdes (PyTorch) and https://github.com/  Benjamin-Walker/log-neural-cdes (JAX).
this section cite: ['b95', 'b38', 'b64', 'b65', 'b59', 'b21', 'b65', 'b21', 'b32', 'b84', 'b100', 'b87', 'b71', 'b38', 'b84', 'b32', 'b21', 'b21', 'b71']

Section: Related work
Increasing the expressivity of parallel-in-time sequence models while retaining their computational efficiency is of significant interest, as it would facilitate training large, performant models. One approach parallelises non-linear RNNs by rewriting them as fixed-point problems and applying parallel Newton or quasi-Newton methods to calculate their output [59,36]. However, reported wall-clock gains remain limited; parallel autoregressive generation can be up to twice as slow as sequential baselines [36]. There have also been a large number of input-dependent LRNN architectures proposed, including input-dependent block-diagonal LRNN [33], DeltaNet [100], DeltaProduct [88], Gated DeltaNet [101], Mamba [39], Mamba-2 [28], RWKV-7 [76], HGRN-2 [79], mLSTM [6], Gated Linear Attention [99], Gated Random Feature Attention [77], Gated Slot Attention [102], TTT-Linear [92], and Titans [7]. These models use either diagonal or DPLR state-transition matrices. Table 2 in [100] presents a comparison of the architectures for a number of these models.
Utilising structured matrices to reduce the computational burden of neural networks extends beyond sequence models. The lottery-ticket hypothesis [35] argues that dense networks contain sparse sub-networks that, when trained in isolation, can match the accuracy of the full model. Such sub-networks have been uncovered by pruning before [94], during [91], and after training [43]. SLiCEs differ from pruning by imposing structured sparsity at initialisation and training the resulting sparse model directly. Other structured-matrix approaches include sparse Transformers [16], 2:4 sparsity in linear layers [67], and Monarch layers, which factorise weight matrices into two block-diagonal components [29].
2 Background
this section cite: ['b58', 'b35', 'b35', 'b32', 'b99', 'b87', 'b100', 'b38', 'b27', 'b75', 'b78', 'b5', 'b98', 'b76', 'b101', 'b91', 'b6', 'b99', 'b34', 'b93', 'b90', 'b42', 'b15', 'b66', 'b28']

Section: Linear controlled differential equations
Let ω : [0, T ] → R dω be a path with bounded-variation, where ω s denotes the value of the path at time s ∈ [0, T ]. A linear Controlled Differential Equation (CDE) takes the form
dh s = dω i=1 A i h s dω i s ,(1)
where A i ∈ R d h ×d h is the linear vector field for each channel i = 1, . . . , d ω , and h : [0, T ] → R d h is the solution path. Approximating ω s with linear interpolation on the grid 0 = t 0 < • • • < t n = T yields htj+1 = exp dω i=1 (ω i tj+1 -ω i tj )A i htj .
(
The outputs htj are computable in O(log(n)) parallel steps by composing the flow on each interval using an associative parallel scan [9]. This approach has been used to parallelise linear RNNs [64] and SSMs [89].
On each interval [t j , t j+1 ], (2) uses only the increments of ω, providing a first-order approximation. The Log-ODE method extends this to higher orders by combining the iterated Lie brackets of the vector field with the log-signature of ω [12]. See Cass and Salvi [11, Section 3.2.2] for a summary description of the algorithm and Appendix C for a description of applying the algorithm to an LNCDE.
this section cite: ['b8', 'b63', 'b88', 'b11']

Section: Linear neural controlled differential equations
Let {(t i , x i )} n i=0 denote a set of observations from a multivariate time-series and X : [t 0 , t n ] → R d X be a continuous interpolation, such that X ti = (t i , x i ). NCDEs are defined as
h t0 = ξ ϕ (t 0 , x 0 ), h t = h t0 + t t0 g θ (h s )dX s , z t = l ψ (h t ),(3)
where ξ ϕ : R d X → R d h and g θ : R d h → R d h ×d X are neural networks, and l ψ : R d h → R dz is a linear map [53]. NCDEs have a number of desirable properties, including maximal expressivity and robustness to irregular sampling rates. Building on the work of Neural Rough Differential Equations [70], Log-NCDEs [98] demonstrate that combining NCDEs with the Log-ODE method during training leads to state-of-the-art performance on a range of multivariate time-series modelling benchmarks with up to 50,000 observations.
LNCDEs take the form
h t = h t0 + t t0 dω i=1 A i θ h s dω X,i s = h t0 + t t0 dω i=1 A i θ dω X,i s h s ,(4)
where ω X : [t 0 , t n ] → R dω is a path which depends on the input X and the A i θ are trainable matrices. As will be discussed further in Section 3.2, LNCDEs are maximally expressive [53,22]. Therefore, there exists a maximally expressive sequence model whose recurrence can be calculated parallel-in-time using the approach outlined in Section 2.1. However, the number of parameters and computational cost makes this approach infeasible in large models. Independently of LNCDEs, Merrill et al. [66] proposed IDS4, a modification of the S4 layer designed to allow state-tracking, which has the same form as (4). Appendix A provides a more detailed introduction to LNCDEs by comparing and contrasting them to SSMs and LRNNs. Additionally, Appendix A contains a toy example demonstrating how the structure of the matrices A i θ affects model expressivity, a discussion of how to extend LNCDEs to matrix-valued hidden states, and a pseudo-code implementation.
this section cite: ['b52', 'b69', 'b97', 'b52', 'b21', 'b65']

Section: Expressivity

this section cite: []

Section: Introduction
Expressivity characterises the set of functions a model can approximate, and maximal expressivity (or universal approximation) guarantees that, with suitable parameters, any continuous function on a compact set can be approximated arbitrarily closely.
this section cite: []

Section: Definition 3.1 (Maximal expressivity).
Let X be a topological space, and let F = {f θ : X → R | θ ∈ Θ} be a class of real-valued functions on X , parametrised by some set Θ. We say that F is maximally expressive (or universal) if, for every compact set K ⊂ X and every real-valued continuous function f : K → R, the following property holds:
∀ϵ > 0, ∃θ ∈ Θ s.t. sup x∈K f (x) -f θ (x) < ϵ.(5)
A classical result is the Universal Approximation Theorem, which states that for X = R d , singlehidden-layer feed-forward networks with a suitable activation function are maximally expressive [27,47].
this section cite: ['b26', 'b46']

Section: Maximally expressive models on paths
Let X denote the space of continuous paths of bounded variation on the interval [0, T ] that start at the same point and contain time as a channel (time-augmented). We endow this space with the 1-variation topology. Let F be the class of LNCDEs defined in (4) with a linear readout layer l θ2 , such that f θ : X → R is defined by
ω → l θ2 (h tn ) = l θ2 h t0 + tn t0 dω i=1 A i θ1 h s dω i s ,(6)
for ω ∈ X . In this setting, F is maximally expressive [51]. Furthermore, LNCDEs with diagonal matrices A i θ1 , such as S4D [40] and Mamba [39], are not maximally expressive [22]. An alternative to maximal expressivity is the following probabilistic property.
this section cite: ['b50', 'b39', 'b38', 'b21']

Section: Definition 3.2 (Maximal Probabilistic Expressivity).
Let X be a topological space, N ∈ N, and
F N = {f N θ : X → R | θ ∈ Θ N } be a class of real-valued functions on X defined by f N θ (ω) = l θ2 ( f N θ1 (ω)),(7)
for ω ∈ X , where
f N θ1 : X → R N , l θ2 ∈ R N is a linear readout, θ 1 ∈ Θ N 1 , θ 2 ∈ Θ N 2 , and Θ N = Θ N 1 ∪ Θ N 2 .
Given a sequence of probability measures P N on Θ N 1 with θ 1 ∼ P N , F has maximal probabilistic expressivity if, for every compact set K ⊂ X and every real-valued continuous function f : K → R, the following property holds:
∀ϵ > 0, lim N →∞ P N ∃l θ2 s.t. sup ω∈K f (ω) -f N θ (ω) < ϵ = 1.(8)
In the context of machine learning, maximal probabilistic expressivity may be considered a more promising property than maximal expressivity, as for large enough N , it implies there exists a significant abundance of parameters θ 1 that are capable of achieving uniformly bounded and arbitrarily low error rates with a linear readout layer. This suggests the parameters should be readily discoverable through standard optimisation methods.
In the case of LNCDEs, N = d h ,
f d h θ1 (ω) = h t0 + tn t0 dω i=1 A i θ1 h s dω i s ,(9)
and
P d h on Θ d h 1 is a collection of probabilities on matrices P i d h with A i θ1 ∼ P i d h .
Achieving maximal probabilistic expressivity depends crucially on the choice of P i d h . Building on the work of Cuchiero et al. [26], Cirone et al. [22,Theorem B.13] showed that choosing the A i θ to be dense Gaussian matrices with independent entries achieves maximal probabilistic expressivity. Unfortunately, using dense matrices is infeasible in practice due to computational constraints, as discussed in Section 2.2.
this section cite: ['b25', 'b21']

Section: Diagonal-plus-low-rank SLiCEs
SSMs with diagonal state-transition matrices, such as S4D [40] and Mamba [39], are examples of SLiCEs with diagonal matrices, A i θ = D i θ . Hence, they are not maximally expressive and underperform on state-tracking benchmarks [22,66]. This limited expressivity motivates the use of alternative structured state-transition matrices.
DeltaNet, DeltaProduct, and Gated DeltaNet use specific versions of DPLR state-transition matrices [85,100,88,101]. The general form of a DPLR-SLiCE is
A i θ = D i θ + r j=1 u i,j θ (v i,j θ ) ⊤ , (10
)
where r is the rank. This parameterisation reduces the number of trainable parameters and computational cost of calculating a hidden state update from O(d ω d 2 h ) to O(d ω rd h ). Furthermore, [72, Proposition D.2] shows that if r → ∞ as d h → ∞, then DPLR-SLiCEs have maximal probabilistic expressivity.
this section cite: ['b39', 'b38', 'b21', 'b65', 'b84', 'b99', 'b87', 'b100']

Section: Block-Diagonal SLiCEs
Block-diagonal state-transition matrices were first explored in LRNNs to improve performance on regular-language tasks [33]. Block-diagonal SLiCEs (BD-SLiCEs) use the same structure, but make the dependence on the input path linear,
A i θ = BlockDiag(B i θ,1 , B i θ,2 , . . . , B i θ,k ),(11)
where each B i θ,j ∈ R bj ×bj is a trainable dense block, k is the number of blocks, and b j are the blocksizes, with d h = k j=1 b j . This parameterisation reduces the number of trainable parameters and computational cost of calculating a hidden state-update from O(d
ω d 2 h ) to O(d ω k j=1 b 2 j )
, providing a substantial speed-up when each b j ≪ d h . Furthermore, this does not restrict the expressivity. Theorem 4.1. If max j b j → ∞ as d h → ∞, then block-diagonal SLiCEs have maximal probabilistic expressivity.
Hence, the non-linear dependence of the input-dependent block-diagonal LRNN is not necessary for theoretical expressivity. Because the hidden state factorises into k independent parts, BD-SLiCEs can be viewed as a multi-head dense LNCDE (DE-LNCDE) of head sizes b j . For a fixed d h , choosing smaller blocks yields greater speed; choosing larger blocks yields greater expressivity. Under a fixed compute budget, there is a trade-off between expressivity and hidden dimension, and this is explored empirically in Appendix D.2.
this section cite: ['b32']

Section: Sparse SLiCEs
Let 0 < ϵ < 1. A sparse SLiCE (S-SLiCE) takes each A i θ to be a sparse matrix with O(d 1+ϵ h ) non-zero entries, selected at random according to a Bernoulli distribution. This reduces the parameter count and computational cost of calculating a hidden state update from O(d ω d 2 h ) to O(d ω d 1+ϵ h ). Furthermore, it does not restrict the expressivity. Theorem 4.2. Sparse SLiCEs have maximal probabilistic expressivity.
In theory, S-SLiCEs have faster training and inference times than DE-LNCDEs. In practice, current deep-learning frameworks (e.g. JAX [10], PyTorch [75]) are not optimised for unstructured sparsity, so practical speed-ups are not observed in our implementations. Nonetheless, we anticipate that ongoing work on sparse matrices will enable future gains in efficiency.
this section cite: ['b9', 'b74']

Section: Walsh-Hadamard SLiCEs
A Hadamard matrix of order n is an n × n matrix H n with entries ±1 whose rows (and columns) are mutually orthogonal, H n H ⊤ n = nI n , where I n is the n × n identity matrix. When n = 2 m for m ∈ N, we can construct these matrices iteratively using the Sylvester construction [93]. Commonly, these matrices are applied via the Walsh-Hadamard transform (WHT), which admits an O(n log n) algorithm [86]. Many scientific computing libraries include efficient CPU and GPU kernels for performing the Walsh-Hadamard transform [97,1]. In practice, a normalisation factor of n -1/2 can be applied to ensure the matrix is orthonormal [95].
this section cite: ['b92', 'b85', 'b96', 'b0', 'b94']

Section: Walsh-Hadamard SLiCEs (WH-SLiCEs) replace each dense matrix A i θ by the product
A i θ = HD i θ ,(12)
where H is a normalised Hadamard matrix, and D i θ is a diagonal matrix. This parameterisation reduces the number of trainable parameters from O(d ω d 2 h ) to O(d ω d h ). Summing the diagonal matrices across the channels and then applying the fast Walsh-Hadamard transform, the computational cost is O(max(d ω d h , d h log(d h )). This is substantially cheaper than the O(d ω d 2 h ) for dense LNCDEs. Furthermore, this modification does not restrict the expressivity. Theorem 4.3. Walsh-Hadamard SLiCEs have maximal probabilistic expressivity.
this section cite: []

Section: Parallel computation
The recurrent cost of a SLiCE is based solely on the cost of calculating a single hidden state update, whereas the calculation when using an associative scan is repeatedly composing the flow
exp dω i=1 (ω i tj+1 -ω i tj )A i θ ≈ I + dω i=1 (ω i tj+1 -ω i tj )A i θ ,(13)
where the first-order approximation of the exponential is sometimes used in practice. When the A i θ are diagonal or block-diagonal, the composition of (13) preserves the structure, as these classes of matrices are closed under multiplication. Therefore, using a parallel associative scan reduces the scan depth from n to log(n), whilst having a computational cost per composition of O(d h ) or O(d h j b 2 j ), respectively. However, for DPLR, sparse, and WH SLiCEs, the structured matrices are not closed under multiplication, which means that the limiting computational cost per composition is the same as a DE-LNCDE, O(d 3 h ). Table 1 summarises the differences in parameter count, computational cost, existence of an efficient implementation, and expressivity of all the SLiCEs considered in this paper, where for simplicity we have taken d ω = d h .
For large models, parallel associative scans result in high I/O costs, as each state-transition matrix must be materialised in GPU memory [100]. A possible approach to mitigating I/O costs for SLiCEs is combining them with the Log-ODE method. By approximating the solution over intervals, this method avoids explicitly materialising intermediate state-transition matrices. However, it does require computing the log-signature of the input path and iterated Lie brackets of the vector fields [98]. A detailed description of this approach is given in Appendix C and Table 1 quantifies the impact of the Log-ODE method on computational cost. In Section 5.3, we implement a hybrid strategy: the Log-ODE method is applied to small intervals, and the resulting outputs are then processed using a parallel associative scan. Yang et al. [100] introduced an alternative approach for DeltaNet, where a chunk-wise algorithm specifically tailored for diagonal-plus-rank-one state-transition matrices is used to bypass the need to materialise every intermediate matrix, significantly cutting down I/O costs [100]. Independently, Cirone and Salvi [18] and Siems et al. [88] extended this approach to higher rank matrices. These approaches can also be applied to diagonal state-transition matrices. Therefore, a block-diagonal SLiCE with a large diagonal portion (b i = 1 for i = 1, . . . , k -1) followed by a small dense block emerges as an attractive solution. The large diagonal section can efficiently utilise the chunk-wise algorithm and the smaller dense section can be processed using parallel associative scans without incurring significant I/O costs. We refer to this structure as diagonal-dense SLiCE (D-DE-SLiCE).
this section cite: ['b99', 'b97', 'b99', 'b99', 'b17', 'b87']

Section: Experiments

this section cite: []

Section: The A 5 benchmark
The A 5 benchmark tests models on their ability to state-track [66]. Each sequence in the dataset consists of a series of permutations from the group of even permutations on five elements, denoted A 5 . The task is to compose the permutations, which requires state-tracking. Following Merrill et al.
Table 1: Comparison of SLiCEs on parameter count, computational cost, the existence of an efficient implementation, and expressivity. Here, d h is the hidden dimension, n is the sequence length, b j are BD's block-sizes, r is DPLR's rank, ϵ is S's sparsity, and for simplicity we have taken d ω = d h . Parallel cost is measured as O(scan depth, cost per composition) when applying a parallel associative scan. Log-X-SLiCE corresponds to applying the Log-ODE method with fixed-size intervals containing s samples and a truncation depth of N , where X is a specific SLiCE structure with O(P X ) parameters, O(R X ) recurrent cost, and O(C X ) cost per composition.
this section cite: ['b65']

Section: Model
Parameters Recurrent Cost Parallel Cost Efficient Impl.
Maximally Expressive
DE-LNCDEs O(d 3 h ) O(nd 3 h ) O(log(n), d 3 h ) Yes Yes D-SLiCEs O(d 2 h ) O(nd 2 h ) O(log(n), d 2 h ) Yes No DPLR-SLiCEs O(rd 2 h ) O(nrd 2 h ) O(log(n), d 3 h ) Yes Yes S-SLiCEs O(d 2+ϵ h ) O(nd 2+ϵ h ) O(log(n), d 3 h ) No Yes WH-SLiCEs O(d 2 h ) O(nd 2 h ) O(log(n), d 3 h ) Yes Yes BD-SLiCEs O d h j b 2 j O nd h j b 2 j O log(n), d h j b 2 j Yes Yes Log-X-SLiCEs O(PX ) O Rx s d N -1 h O log n s , CX d N -1 h - -
[66], we train and evaluate models on sequences ranging from length 3 to 20 and determine how many stacked layers each model needs to achieve a validation accuracy greater than 90%.
This benchmark serves as an empirical validation of our theoretical results; D-SLiCEs are less expressive than DPLR, sparse, WH, and BD SLiCEs. In addition to the SLiCEs, we consider Mamba [39], LSTM [44], gated DeltaProduct with negative eigenvalues [101,88], and the two components of xLSTM [6] (mLSTM and sLSTM) on this benchmark. All baselines use a hidden dimension of 1024 and all SLiCEs use 1024 parameters per state-transition matrix. Full experimental details are given in Appendix D.1.
Figure 1a shows that the diagonal state-transition matrices of Mamba, mLSTM, and D-SLiCE mean that an increasing number of stacked layers are needed as the sequence length grows. Interestingly, Gated DeltaProduct with negative eigenvalues, which uses a DPLR structure, and the D-DE-SLiCE also need a growing number of stacked layers. However, DPLR and BD SLiCE both need one layer for all sequence lengths, suggesting this is not an inherent limitation of theses structures. Similarly, sparse, Walsh-Hadamard, and dense SLiCEs, as well as the two recurrent baselines LSTM and sLSTM, all need only one layer for all sequence lengths.
To assess length generalisation, we select the models that achieve at least 90% validation accuracy on sequences of length 20 and retrain them on sequences ranging from 3 to 40. Early stopping is performed using a validation set with sequence lengths from 40 to 128. The mLSTM is excluded because it requires fixed-length inputs. Figure 1b reports test accuracy for lengths from 20 to 5120. The recurrent LSTM and sLSTM generalise well, maintaining high test accuracy beyond both the training and validation ranges. Among the parallel-in-time models, three patterns emerge: (i) WH-SLiCE and Mamba do not attain high accuracy even at training lengths; (ii) DeltaProduct and D-DE-SLiCE generalise to approximately 2× the training length but not beyond the validation range; and (iii) DE-LNCDE, DPLR-SLiCE, S-SLiCE, and BD-SLiCE sustain high accuracy on sequences at least 8× the training length, exceeding the maximum validation length.
this section cite: ['b38', 'b43', 'b100', 'b87', 'b5']

Section: Regular language tasks
The formal language benchmark is a collection of language style tasks split into categories using the Chomsky hierarchy [17,31]. Here, we use the regular tasks, which can be solved by processing inputs sequentially with a fixed set of internal states and no external memory, i.e. state-tracking. On this benchmark, the models are challenged to generalise to longer sequences, by training on sequences from length 3 to 40 and evaluating on sequences from length 40 to 256. Details on the individual tasks can be found in Appendix D.2. A wide range of existing sequence model architectures are used as baselines, including LSTM [44], xLSTM and its two components mLSTM and sLSTM [6], four variations of DeltaNet [85,99,101,88,38], RWKV-7 [76], a Transformer [96], S4D [40], and Mamba [39]. All models use two stacked layers. For each dataset and baseline model, we 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20Sequence Length   selected the hidden dimension that yields a higher validation accuracy from two choices. The choices were 128 and 512 for all models aside from Mamba, and 256 and 512 for Mamba, as it does not support a hidden size of 128. All SLiCEs use two stacked layers and 512 non-zero parameters per state-transition matrix, except for the diagonal and Walsh-Hadamard, which also consider 128, with the better performing choice reported for each dataset. For the DPLR, block-diagonal, and diagonal-dense SLiCEs, we consider multiple choices of rank and block-size, respectively, and present the best performing models. A thorough investigation of the effect of block-size and rank is given in Appendix D.2. We do not consider S-SLiCE on this benchmark, due to the lack of an efficient implementation.
Table 2 presents the results. As expected, the recurrent LSTM generalises almost perfectly on all four tasks. Amongst the parallel models, DeltaNet with negative eigenvalues and Gated DeltaProduct with negative eigenvalues are the best performing baselines, aligning with the expectation that increased complexity in the state-transition matrix improves state-tracking performance. Similarly, D-SLiCE outperforms Mamba, aligning with the results of Grazzi et al. [38] that expanding the eigenvalue range of the state-transition matrix improves state-tracking performance. Similarly to the A 5 length Table 2: Results for formal language tasks. Average and standard deviation of validation accuracy over five runs for a range of recurrent and parallel models.
this section cite: ['b16', 'b30', 'b43', 'b5', 'b84', 'b98', 'b100', 'b87', 'b37', 'b75', 'b95', 'b39', 'b38', 'b37']

Section: Model Cycle Nav.
Even Pairs Mod Arith. No Brack. Parity Average Recurrent LSTM 100.0 ± 0.0 100.0 ± 0.0 99.9 ± 0.1 100.0 ± 0.0 100 sLSTM 32.5 ± 0.4 100.0 ± 0.0 27.7 ± 0.6 100.0 ± 0.0 65.1 xLSTM[1:1] 53.5 ± 5.6 99.0 ± 1.9 29.
3 ± 1.6 100.0 ± 0.0 70.5 Parallel DeltaNet 49.8 ± 4.7 100.0 ± 0.0 42.2 ± 4.8 57.8 ± 0.8 62.5 DeltaNet[-1, 1] 46.7 ± 6.1 100.0 ± 0.0 66.4 ± 8.8 97.7 ± 2.0 77.7 Gated DeltaNet 53.8 ± 8.8 100.0 ± 0.0 42.8 ± 8.2 56.5 ± 1.9 63.3 Gated DeltaProduct[-1,1] 46.3 ± 6.6 100.0 ± 0.0 78.4 ± 10.9 98.0 ± 1.4 80.7 RWKV-7 37.8 ± 5.0 88.1 ± 14.2 39.5 ± 6.1 51.1 ± 0.3 54.1 mLSTM 52.4 ± 10.5 99.9 ± 0.1 28.8 ± 3.1 53.0 ± 2.1 58.5 Transformer 24.4 ± 0.5 90.4 ± 10.4 23.6 ± 0.7 52.2 ± 0.4 47.7 Mamba 48.4 ± 2.2 100.0 ± 0.0 33.1 ± 6.6 54.2 ± 2.1 58.9 S4D 23.7 ± 1.1 68.7 ± 4.7 21.7 ± 0.4 51.2 ± 1.0 41.3 D-SLiCE 69.5 ± 6.3 100.0 ± 0.0 20.9 ± 0.1 100.0 ± 0.0 72.6 WH-SLiCE 69.7 ± 8.8 93.1 ± 13.9 23.8 ± 1.1 71.4 ± 12.9 64.5 BD-SLiCE d h =128,b=4 99.8 ± 0.2 85.9 ± 11.3 54.0 ± 12.5 95.3 ± 3.9 83.8 D-DE-SLiCE d h =272,b=16 73.3 ± 29.4 84.8 ± 8.5 98.4 ± 0.7 83.8 ± 11.3 85.1 DPLR-SLiCE d h =57,r=4 81.1 ± 16.6 100.0 ± 0.0 68.3 ± 19.3 91.0 ± 18.0 85.1 Random 20.0 50.0 20.0 50.0 35.0 generalisation task, the WH-SLiCE underperforms other SLiCE structures. However, unlike in the A 5 length generalisation task, the D-DE-SLiCE achieves the joint highest average validation accuracy among the parallelisable models, alongside DPLR-SLiCE.
this section cite: []

Section: UEA multivariate time-series classification archive
Since SLiCEs are descendants of NCDEs, they inherit a number of the desirable properties which arise from having a natural continuous-time formulation. These include robustness to irregular sampling rates and decoupling the number of recurrent steps from the number of observations in the time-series [53,98]. Furthermore, SLiCEs have the same theoretical expressivity as NCDEs, whilst being parallel-in-time, making them an attractive alternative for real-world time-series modelling.
As a demonstration of the practical benefits, we consider six datasets from the UEA Multivariate Time-Series Classification Archive (UEA-MTSCA), a collection of time-series classification tasks, ranging from classifying worms into species based on movement to classifying alcohol by concentration using vibrational spectroscopy [4]. Walker et al. [98] showed that Log-NCDEs outperform the linear recurrent unit (LRU) [73], S5 [89], S6 [39], and Mamba [39] on average test accuracy over the six longest datasets with at least 200 observations [98]. However, the per-step training time is significantly higher for Log-NCDEs than the baseline methods.
Keeping all other hyperparameters the same, Table 3 presents the impact of replacing the non-linear vector field of a Log-NCDE with a structured linear vector field. The GPU memory and time per 1000 training steps were recalculated for all models on an NVIDIA H100. BD-SLiCE achieves similar performance to the Log-NCDE, whilst reducing the average per-step training time by a factor of nearly 20 and increasing the average GPU memory usage by only 8%. Appendix D.3 presents the results for individual datasets and analyses the impact of the Log-ODE method and parallel associative scan on run-time and GPU memory.
this section cite: ['b52', 'b97', 'b3', 'b97', 'b72', 'b88', 'b38', 'b38', 'b97']

Section: Limitations and Future Work
To reduce the computational burden of SLiCEs, our implementation approximates the matrix exponential when computing the flow via (13). However, even with this adjustment, scaling SLiCEs to the multi-billion parameter regime remains challenging. A key technical goal is the development of efficient GPU kernels for the matrix exponential and parallel associative scans, particularly when handling many small independent systems, such as for BD-SLiCEs. Alternatively, building on the work of Yang et al. [100] and Cirone and Salvi [18], fast chunk-wise methods for a broader class of structured matrices may offer a viable path forward.
Alternative SLiCE architectures may achieve maximal expressivity and improved empirical performance. A theoretical characterisation of the conditions that a SLiCE's structured matrix needs to satisfy to achieve maximal probabilistic expressivity would aid the search for additional structures. Moreover, although establishing maximal probabilistic expressivity is a significant step towards a deeper theoretical understanding of structured state-transition matrices, expressivity at finite hidden dimensions remains an open challenge.
Finally, unlike NCDEs, SLiCEs are sequence-to-sequence models that update their state with each input sample. Therefore, similarly to other discrete sequence models, SLiCEs are susceptible to over-sampled data. Combining SLiCEs with the Log-ODE method enables path-based inputs by operating with flows over intervals, rather than individual samples. However, the Log-SLiCE outputs a sequence whose elements correspond to the boundary values of the output path for each interval the Log-ODE method was applied to. Therefore, you cannot stack two Log-SLiCEs, as the first level has produced a sequence, whereas the second level consumes a path. A natural direction for future work is developing a true path-to-path model.
this section cite: ['b99', 'b17']

Section: Conclusion
This paper introduced SLiCEs, a unifying framework for sequence-to-sequence layers that are maximally expressive, computationally efficient, and allow for parallel-in-time computation. We explored four specific instances, diagonal-plus-low-rank, sparse, Walsh-Hadamard, and blockdiagonal, analysing their theoretical properties and empirical performance. Theorems 4.1, 4.2, and 4.3 established that block-diagonal, sparse, and Walsh-Hadamard SLiCEs achieve maximal probabilistic expressivity. Furthermore, all SLiCE structures demonstrated single-layer state-tracking on the A 5 benchmark, unlike the other parallelisable layers considered: diagonal SLiCEs, mLSTM, Mamba, and DeltaProduct. Among the SLiCEs, block-diagonal stands out as the only maximally expressive variant that strictly reduces parameter count, recurrent cost, and parallel cost compared to dense LNCDEs. Additionally, a variant of the block-diagonal SLiCE achieved the joint highest average validation accuracy among parallel models on the regular language tasks from the formal language benchmark. Finally, practical speed-ups for real-world time series modelling were demonstrated on six multivariate time-series classification datasets, where replacing the non-linear vector field of a Log-NCDE with a block-diagonal linear vector field reduced the average time per training step by a factor of twenty, without impacting the model's overall performance.
this section cite: []

Section: References
Ref_id:b0 Title: Tensor core accelerated hadamard transform kernel Year: (2024)
Ref_id:b1 Title: Sig-SDEs model for quantitative finance Year: (2020)
Ref_id:b2 Title:  Year: (2016)
Ref_id:b3 Title: The UEA multivariate time series classification archive Year: (2018)
Ref_id:b4 Title: Sigdiffusions: Score-based diffusion models for long time series via log-signature embeddings Year: ()
Ref_id:b5 Title: xLSTM: Extended long short-term memory Year: ()
Ref_id:b6 Title: Learning to memorize at test time Year: (2024)
Ref_id:b7 Title: Permutation equivariant neural controlled differential equations for dynamic graph representation learning Year: (2025)
Ref_id:b8 Title: Prefix sums and their applications. (CMU-CS-90-190) Year: (1990)
Ref_id:b9 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b10 Title: Lecture notes on rough paths and applications to machine learning Year: (2024)
Ref_id:b11 Title: An efficient approximation method for stochastic differential equations by means of the exponential Lie series Year: (1995)
Ref_id:b12 Title: Iterated integrals and exponential homomorphisms Year: (1954)
Ref_id:b13 Title: Integration of paths, geometric invariants and a generalized Baker-Hausdorff formula Year: (1957)
Ref_id:b14 Title: A primer on the signature method in machine learning Year: (2025)
Ref_id:b15 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b16 Title: Three models for the description of language Year: (1956)
Ref_id:b17 Title: ParallelFlow: Parallelizing linear transformers via flow discretization Year: (2025)
Ref_id:b18 Title: Rough kernel hedging Year: (2025)
Ref_id:b19 Title: Neural signature kernels as infinite-width-depth limits of controlled resnets Year: ()
Ref_id:b20 Title: Graph expansions of deep neural networks and their universal scaling limits Year: (2024)
Ref_id:b21 Title: Theoretical foundations of deep selective state-space models Year: (2024)
Ref_id:b22 Title: SK-Tree: a systematic malware detection algorithm on streaming trees via the signature kernel Year: (2021)
Ref_id:b23 Title: Nowcasting with signature methods Year: (2023)
Ref_id:b24 Title: Subtle variations in sepsis-III definitions markedly affect predictive performance Year: (2024)
Ref_id:b25 Title: Expressive power of randomized signature Year: (2021)
Ref_id:b26 Title: Approximation by superpositions of a sigmoidal function Year: (1989)
Ref_id:b27 Title: Transformers are SSMs: generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b28 Title: Monarch: Expressive structured matrices for efficient and accurate training Year: (2022)
Ref_id:b29 Title: Language modeling with gated convolutional networks Year: (2017)
Ref_id:b30 Title: Neural networks and the Chomsky hierarchy Year: (2023)
Ref_id:b31 Title: CoRoPa computational rough paths (software library) Year: (2010)
Ref_id:b32 Title: Advancing regular language reasoning in linear recurrent neural networks Year: (2024-06)
Ref_id:b33 Title: New directions in the applications of rough path theory Year: (2023)
Ref_id:b34 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2019)
Ref_id:b35 Title: Towards scalable and stable parallelization of nonlinear RNNs Year: ()
Ref_id:b36 Title: Sparse arrays of signatures for online character recognition Year: (2013)
Ref_id:b37 Title: Unlocking state-tracking in linear RNNs through negative eigenvalues Year: (2024)
Ref_id:b38 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b39 Title: On the parameterization and initialization of diagonal state space models Year: (2022)
Ref_id:b40 Title: A basis for free Lie rings and higher commutators in free groups Year: (1950)
Ref_id:b41 Title: Uniqueness for the signature of a path of bounded variation and the reduced path group Year: (2010)
Ref_id:b42 Title: Channel pruning for accelerating very deep neural networks Year: (2017-10)
Ref_id:b43 Title: Long short-term memory Year: (1997-11)
Ref_id:b44 Title: A neural RDE approach for continuoustime non-Markovian stochastic control problems Year: (2023)
Ref_id:b45 Title: Exact gradients for stochastic spiking neural networks driven by rough signals Year: (2024)
Ref_id:b46 Title: Approximation capabilities of multilayer feedforward networks Year: (1991)
Ref_id:b47 Title: Optimal stopping via distribution regression: a higher rank signature approach Year: (2023)
Ref_id:b48 Title: Non-adversarial training of neural SDEs with signature kernel scores Year: (2024)
Ref_id:b49 Title: Transformers are RNNs: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b50 Title: On neural differential equations Year: (2022)
Ref_id:b51 Title: Deep signature transforms Year: (2019)
Ref_id:b52 Title: Neural controlled differential equations for irregular time series Year: (2020)
Ref_id:b53 Title: A method for stochastic optimization Year: (2017)
Ref_id:b54 Title: Kernels for sequentially ordered data Year: (2019)
Ref_id:b55 Title: SigGPDE: Scaling sparse gaussian processes on sequential data Year: (2021)
Ref_id:b56 Title: Distribution regression for sequential data Year: (2021)
Ref_id:b57 Title: Learning from the past, predicting the statistics for the future, learning an evolving system Year: (2016)
Ref_id:b58 Title: Parallelizing nonlinear sequential models over the sequence length Year: (2024)
Ref_id:b59 Title: Transformers learn shortcuts to automata Year: (2023)
Ref_id:b60 Title: Differential equations driven by rough signals Year: (1998)
Ref_id:b61 Title: Differential Equations Driven by Rough Paths: École D'été de Probabilités de Saint-Flour XXXIV-2004 Year: (2007)
Ref_id:b62 Title: Signature kernel conditional independence tests in causal discovery for stochastic processes Year: (2024)
Ref_id:b63 Title: Parallelizing linear recurrent neural nets over sequence length Year: (2018)
Ref_id:b64 Title: The parallelism tradeoff: Limitations of log-precision transformers Year: (2023)
Ref_id:b65 Title: The illusion of state in state-space models Year: ()
Ref_id:b66 Title: Accelerating sparse deep neural networks Year: (2021)
Ref_id:b67 Title: Proceedings of the 23rd Python in Science Conference Year: (2024)
Ref_id:b68 Title: The signature-based model for early detection of sepsis from electronic health records in the intensive care unit Year: (2019)
Ref_id:b69 Title: Neural rough differential equations for long time series Year: ()
Ref_id:b70 Title: On the choice of interpolation scheme for neural CDEs Year: (2022)
Ref_id:b71 Title: Fixed-point RNNs: From diagonal to dense in a few iterations Year: (2025)
Ref_id:b72 Title: Resurrecting recurrent neural networks for long sequences Year: ()
Ref_id:b73 Title: A path-dependent PDE solver based on signature kernels Year: (2024)
Ref_id:b74 Title: PyTorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b75 Title: RWKV-7 "Goose" with expressive dynamic state evolution Year: (2025)
Ref_id:b76 Title: Random feature attention Year: ()
Ref_id:b77 Title: Learning dynamic graph embeddings with neural controlled differential equations Year: (2025-10)
Ref_id:b78 Title: Gated linear RNNs with state expansion Year: (2024)
Ref_id:b79 Title: Free Lie Algebras. LMS monographs Year: (1993)
Ref_id:b80 Title: Rough paths, kernels, differential equations and an algebra of functions on streams Year: (2021)
Ref_id:b81 Title: The signature kernel is the solution of a Goursat PDE Year: (2021)
Ref_id:b82 Title: Higher order kernel mean embeddings to capture filtrations of stochastic processes Year: (2021)
Ref_id:b83 Title: A structure theorem for streamed information Year: (2023)
Ref_id:b84 Title: Linear transformers are secretly fast weight programmers Year: ()
Ref_id:b85 Title: Computation of the fast Walsh-Fourier transform Year: (1969)
Ref_id:b86 Title: Sparse signature coefficient recovery via kernels Year: (2024)
Ref_id:b87 Title: DeltaProduct: Improving state-tracking in linear RNNs via Householder products Year: (2025)
Ref_id:b88 Title: Simplified state space layers for sequence modeling Year: (2023)
Ref_id:b89 Title: Dropout: A simple way to prevent neural networks from overfitting Year: (2014)
Ref_id:b90 Title: Sparse connection and pruning in large dynamic artificial neural networks Year: (1997)
Ref_id:b91 Title: RNNs with expressive hidden states Year: (2025)
Ref_id:b92 Title: Thoughts on inverse orthogonal matrices, simultaneous signsuccessions, and tessellated pavements in two or more colours, with applications to Newton's rule, ornamental tile-work, and the theory of numbers Year: (1867)
Ref_id:b93 Title: Pruning neural networks without any data by iteratively conserving synaptic flow Year: (2020)
Ref_id:b94 Title: Quantum process tomography with unsupervised learning and tensor networks Year: (2023)
Ref_id:b95 Title: Attention is all you need Year: (2017)
Ref_id:b96 Title: Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python Year: (2020)
Ref_id:b97 Title: Log neural controlled differential equations: The Lie brackets make a difference. International Conference on Machine Learning Year: (2024)
Ref_id:b98 Title: Gated linear attention transformers with hardware-efficient training Year: (2024)
Ref_id:b99 Title: Parallelizing linear transformers with the delta rule over sequence length Year: (2024)
Ref_id:b100 Title: Gated delta networks: Improving Mamba2 with Delta Rule Year: (2025)
Ref_id:b101 Title: Gated slot attention for efficient linear-time sequence modeling Year: ()
