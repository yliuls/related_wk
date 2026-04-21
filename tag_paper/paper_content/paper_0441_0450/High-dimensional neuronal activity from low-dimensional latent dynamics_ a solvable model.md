Title: High-dimensional neuronal activity from low-dimensional latent dynamics: a solvable model
Abstract: Computation in recurrent networks of neurons has been hypothesized to occur at the level of low-dimensional latent dynamics, both in artificial systems and in the brain. This hypothesis seems at odds with evidence from large-scale neuronal recordings in mice showing that neuronal population activity is high-dimensional. To demonstrate that low-dimensional latent dynamics and high-dimensional activity can be two sides of the same coin, we present an analytically solvable recurrent neural network (RNN) model whose dynamics can be exactly reduced to a lowdimensional dynamical system, but generates an activity manifold that has a high linear embedding dimension. This raises the question: Do low-dimensional latents explain the high-dimensional activity observed in mouse visual cortex? Spectral theory tells us that the covariance eigenspectrum alone does not allow us to recover the dimensionality of the latents, which can be low or high, when neurons are nonlinear. To address this indeterminacy, we develop Neural Cross-Encoder (NCE), an interpretable, nonlinear latent variable modeling method for neuronal recordings, and find that high-dimensional neuronal responses to drifting gratings and spontaneous activity in visual cortex can be reduced to low-dimensional latents, while the responses to natural images cannot. We conclude that the high-dimensional activity measured in certain conditions, such as in the absence of a stimulus, is explained by low-dimensional latents that are nonlinearly processed by individual neurons.

Section: Introduction
The mammalian cortex comprises a large number of neurons, which, in principle, should allow it to use a high-dimensional neural code to represent sensory, motor, and cognitive information. Nevertheless, multi-neuronal recordings in nonhuman primates [1][2][3][4] have suggested that cortical populations perform computations by approximating low-dimensional dynamical systems [5,6], with neuronal firing rates lying on a low-dimensional "neural manifold" [7]. In support of this hypothesis, low-dimensional dynamics have been inferred from multi-neuronal recordings through a wide variety of methods [8][9][10][11][12][13][14][15][16][17][18][19][20][21][22][23]; they spontaneously emerge in recurrent neural networks (RNNs) trained to solve behavioral tasks [2,[24][25][26][27][28][29][30][31][32][33][34]; and they appear in several theoretical models of noise-robust neuronal population dynamics [35][36][37][38]. A result that might at first sight challenge the low-dimensional dynamical systems hypothesis is that visual cortical population activity in mice has high linear dimension [39,40] with shared neuronal covariance having a heavy-tailed eigenspectrum (see also [41] and [42] for recordings in cerebellum and across cortex, respectively). In particular, the shared covariance eigenspectrum has a power-law tail with an exponent close to 1 (α ≈ 1.04) [39] for responses to natural images and an exponent of α ≈ 1.14 for spontaneous activity [40]. Are these two views on the dimensionality of population activity compatible? Namely, can a low-dimensional dynamical system produce a neural manifold that has a high linear embedding dimension?
Here, we first construct a solvable RNN model that reconciles the low-and high-dimensional perspectives on population activity by carefully disambiguating the linear dimension of the system before and after the neurons' nonlinearity, which we refer to as the pre-and post-activation dimension, respectively. This dichotomy refines the usual distinction between linear and "intrinsic" dimension [39,43,44], since the intrinsic dimension of a system is the same before and after any continuous, injective nonlinearity. Using the notions of pre-and post-activation linear dimensions, we show that our RNN can be exactly reduced to a low-dimensional dynamical system in the space of preactivations, making the pre-activations low-dimensional. Then, we show that these latent dynamics generate high-dimensional post-activation activity that has a power-law covariance eigenspectrum. (In this work, dimension will always refer to linear dimension, unless stated otherwise.) Before analyzing experimental recordings, we revisit the spectral theory of infinite-width neural networks (random feature kernels) [45][46][47] to quantitatively relate the pre-activation dimension, the neuronal activation function, and the post-activation covariance eigenspectrum. This three-way relationship tells us that high-dimensional activity is consistent with both low-and high-dimensional pre-activations. To uncover the pre-activation dimension of high-dimensional activity in visual cortex, we perform two-photon calcium recordings of tens of thousands of neurons from mouse visual cortex, and infer the pre-activation dimension using the Neural Cross-Encoder (NCE), an interpretable, nonlinear latent variable modeling method which models the activity of each neuron as a simple linear-nonlinear readout of low-dimensional latents. NCE reveals that both the responses to drifting gratings and spontaneous activity can be well approximated by low-dimensional pre-activations, but that responses to natural images cannot. This suggests that the encoding of natural images in visual cortex is already high-dimensional in the space of pre-activations.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b1', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b38', 'b39', 'b38', 'b42', 'b43', 'b44', 'b45', 'b46']

Section: Solvable RNN Model
To demonstrate how high-dimensional post-activations can arise from low-dimensional pre-activation dynamics, we first present a solvable RNN model whose autonomous dynamics is low-dimensional in the space of pre-activations, but high-dimensional in the space of post-activations, with the post-activations producing a power-law covariance eigenspectrum.
We consider an RNN consisting of N rate-units (neurons). The pre-activation x i of neuron i evolves according to
ẋi = -x i + 1 N N j=1 W ij ϕ(x j ),(1)
where W ij denotes the synaptic weight from neuron j to neuron i, and ϕ : R → R ≥0 is a nonlinear activation function converting the pre-activations into post-activations (firing rates). To define the weights W ij , we randomly place neurons on a ring [48][49][50] by assigning to each neuron i an independent and uniformly distributed angle θ i ∈ [0, 2π) (Fig. 1A). The weights W ij are then given by the following shifted cosine function:
W ij := J cos(θ i -θ j -∆).(2)
The shift ∆ in Eq. ( 2) makes the weights asymmetric, with neurons sending their strongest excitatory output to neurons located at an angle ∆ counter-clockwise (Fig. 1B). To make the model solvable, we assume that the activation function ϕ is the Heaviside step function Θ, i.e., ϕ(x) = Θ(x) = 1, if x ≥ 0, 0, if x < 0.  (3). Shown in D and F are theoretical values given by Eqs. (5) and (8), respectively, which closely match simulations of large networks (see Appendix C).
this section cite: ['b47', 'b48', 'b49', 'b4', 'b7']

Section: Low-dimensional pre-activation dynamics
The RNN model defined above has low-dimensional pre-activation dynamics because the weight matrix W has rank 2. Indeed, using an elementary trigonometric identity,foot_0 W can be factorized as the outer product W = UV T of two N × 2-matrices (Fig. 1C), with
U :=    cos(θ 1 ) sin(θ 1 ) . . . . . . cos(θ N ) sin(θ N )    and V := J    cos(θ 1 + ∆) sin(θ 1 + ∆) . . . . . . cos(θ N + ∆) sin(θ N + ∆)    .
Then, following Beiran et al. [51], we can reduce the N -dimensional system, Eq. ( 1), to a 2dimensional system describing the dynamics of the latent variables κ := U † x, where † denotes the pseudoinverse and x the N -dimensional vector of pre-activations (x 1 , . . . , x N ) T . The dynamics of the latent variables follows
κ = -κ + 1 N V T ϕ(Uκ),
where ϕ is applied element-wise to the N -dimensional vector of pre-activations Uκ = x. In the equation above, the vector ϕ(Uκ) = ϕ(x) represents the joint post-activations (firing rates) of the N neurons.
Taking the number of neurons N → ∞ yields a neural field limit [52] where the sum over neurons becomes an integral over the ring,
κ = -κ + 2π θ=0 v(θ)ϕ (u(θ) • κ) dθ 2π ,(4)
with u(θ) := (cos(θ), sin(θ)) T and v(θ) := J(cos(θ + ∆), sin(θ + ∆)) T . Equation ( 4) describes the dynamics of the latent variables κ as the solution to a 2-dimensional dynamical system whose vector field involves an integral over the "circuit structure" [52] (the ring).
Since ϕ is the step function, the integral over the ring in Eq. ( 4) can be solved, and we obtain the solvable 2-dimensional dynamical system,
κ = -κ + 1 -1 1 1 κ ∥κ∥ ,(5)
when J := π √ 2 and ∆ := π/4 (derivation presented in Appendix A). Equation (5) generates a stable limit cycle over the unit circle (Fig. 1D), which implies that the latent variables κ will eventually rotate on the unit circle indefinitely. In Appendix E, we provide other examples of low-rank RNNs for which the latent dynamics can be expressed in a tractable form similar to Eq. ( 5).
Neuronal activity, modeled here as the post-activations ϕ(Uκ), are simple linear-nonlinear readouts of latent variables κ (Fig. 1E). Hence, we have effectively reduced the dynamics of the RNN, Eq. ( 1), to a 2-dimensional latent dynamical system. We will say that neuronal activity is a linearnonlinear function of latent variables if it is given by the composition of a linear mapping (U) and an element-wise, nondecreasing nonlinear mapping (ϕ).
this section cite: ['b50', 'b51', 'b51']

Section: Post-activations produce a power-law eigenspectrum
Since the dynamics of the latent variables is solvable in the large-network limit, and rotates on the unit circle, we can compute the correlation between the post-activations of two neurons. For any pair of neurons i and j, with positions θ i and θ j on the ring, respectively, the correlation of their post-activations C ij is, in the long-recording limit, given by
C ij = 2 π (π -|θ i -θ j |) -1,(6)
where |θ i -θ j | := cos -1 (cos(θ i -θ j )) is the absolute angle difference θ i and θ j (derivation presented in Appendix B).
We can find the eigenvalue spectrum of post-activations by noting that as the number of recorded neurons M → ∞, the eigenvalues of the M × M correlation matrix C (M ) defined by Eq. ( 6) converge to the eigenvalues of an integral operator. Since the angles θ i are independently and uniformly sampled on circle [0, 2π), the correlation matrix C (M ) is a so-called Euclidean random matrix [53], that is, a matrix whose entries are given by the pairwise distances between randomly sampled points in a given space. Writing
λ (M ) 1 ≥ λ (M ) 2 ≥ • • • ≥ λ (M ) N
for the ranked eigenvalues of the matrix C (M ) , random matrix theory [54,55] tells us that, as M → ∞, the scaled eigenvalues {λ (M ) n /M } M n=1 , converge (in a ℓ 2 sense) to the eigenvalues of the integral operator,
f (θ) → 2π θ ′ =0 2 π (π -|θ -θ ′ |) 0-th arc-cosine kernel -1 f (θ ′ ) dθ ′ 2π . (7
)
The eigenvalues of this integral operator can be computed analytically. In Eq. ( 7), we have highlighted the presence of the 0-th arc-cosine kernel k 0 (θ, θ ′ ) := π -|θ -θ ′ | of Cho and Saul [45], which is well-known in machine learning, and whose eigenvalues have been computed in [46]. In short, by the rotational invariance of k 0 , we have that, for any positive integer m, the functions θ → cos(mθ) and θ → sin(mθ) are orthogonal eigenfunctions of the operator Eq. ( 7) sharing the same eigenvalue,
2 π 2 π θ=0 (π -θ) cos(mθ)dθ = 2 π 2 1 -(-1) m m 2 .
Using this result, we obtain that the ranked eigenvalues λ 1 ≥ λ 2 ≥ . . . of the operator Eq. ( 7) are given by
λ n = 4 π 2 (2⌊(n -1)/2⌋ + 1) -2 , ∀n ∈ N * ,(8)
that is, eigenvalues come in identical pairs that decay exactly as a power law with decay exponent α = 2 (Fig. 1F). Hence, post-activations are high-dimensional in the sense that their covariance eigenspectrum has a heavy tail [39]. Although the model presented assumed, for simplicity, a cosine lateral connectivity on the ring, Eq. ( 2), similar results can be derived for more general lateral connectivity; see Appendix D for an example.
In summary, this solvable model shows that low-dimensional dynamics in the space of pre-activations can generate high-dimensional post-activations. The heavy tail of the covariance eigenspectrum implies that post-activations are not confined to any finite-dimensional linear subspace. Formally, the smallest vector space containing the post-activations generated by our model has the same size as the infinite-dimensional reproducing kernel Hilbert space associated with the kernel k 0 . We stress that, in this model, the heavy tail of the post-activation eigenspectrum is not due to noise, since we used a deterministic, non-chaotic RNN. Also, all the results presented above remain exact if the rate-units in Eq. ( 1) are replaced by linear-nonlinear-Poisson neurons, as spike noise cancels out in the limits we consider [38,52].
this section cite: ['b52', 'b53', 'b54', 'b44', 'b45', 'b38', 'b37', 'b51']

Section: Post-activation eigenspectrum depends on pre-activation dimension and activation function
To shed light on the relationship between the post-activation eigenspectrum, pre-activation dimension, and the activation function ϕ, we now turn to a more general setup, which allows us to relax some of the strong assumptions of the solvable model (Fig. 1E). First, we allow the number of latent variables d to be greater than 2, assuming that the latent variables, henceforth denoted by z (instead of κ), are uniformly distributed on the unit sphere S d-1 in R d . We assume that the pre-activations of the network are determined by passing the latent activity through a N × d feedforward weight matrix U with i.i.d. standard normal entries. In this setup (Fig. 2A), we call d the pre-activation dimension, as it sets the linear dimensionality of the pre-activations. In the solvable model of Sec. 2.2, for example, the pre-activation dimension was d = 2 (Fig. 1E). Finally, we replace the step function, Eq. ( 3), by the general rectified power activation function ϕ p,c (x) := [max(0, x + c)] p ,
where the activation parameter p ∈ R ≥0 is a nonnegative real value and the bias c ∈ R. (By convention, ϕ 0,c (x) := Θ(x + c).)
This setup can be analyzed within the framework of random feature kernels (see [56,Sec. 9.5]). Denoting µ d-1 the uniform probability measure on the sphere S d-1 , let us take T independent latent variable samples z 1 , . . . , z T from µ d-1 , and define the N × T post-activation matrix A (N,T ) := (ϕ p,c (Uz 1 ), . . . , ϕ p,c (Uz T )). In the limits N → ∞ and T → ∞ taken successively, the covariance eigenspectrum of A (N,T ) converges (when properly scaled) to the eigenvalue spectrum of the integral operator
f (z) → S d-1 K p,c,d (z, z ′ )f (z ′ )dµ d-1 (z ′ ),(10)
where K p,c,d :
S d-1 × S d-1 → R is the random feature kernel K p,c,d (z, z ′ ) := E ξ∼N (0,I d ) [ϕ p,c (ξ • z)ϕ p,c (ξ • z ′ )](11)
(see [46,56] or Appendix F for more details).
Drawing intuition from Fourier analysis, the smoothness of a function (here the kernel) should be related to the decay rate of its Fourier transform (here the eigenspectrum)-the smoother the function, the faster the decay rate of its Fourier transform. Known results on the eigenvalues of random feature kernels for the cases p = 0 and p = 1, with c = 0, confirm this intuition and show how it extends to general integers d [46]. Extrapolating those results to any nonnegative p and any real c, we get the following conjecture. Conjecture 1. For any p ∈ R ≥0 , c ∈ R, and any integer d ≥ 2, the ranked eigenvalues λ 1 ≥ λ 2 ≥ . . . of the integral operator (10) obey the following power-law decay:
λ n ≍ n -α with α = 1 + 2p + 1 d -1 ,(12)
where a n ≍ b n means lim n→+∞ a n /b n = C ∈ (0, +∞).
To the best of our knowledge, Conjecture 1 is not a straightforward consequence of any existing result in theoretical machine learning [47,57,58] or harmonic analysis [59][60][61][62], hence our presentation of Eq. ( 12) as a conjecture. Note that when the activation parameter p is an integer, ϕ p,c is p-times weakly differentiable, that is, the first p weak derivatives 3 of ϕ p,c are all locally integrable. This, and the fact that the bias c does not affect the decay rate, suggest a further extension of the conjecture to more general activation functions, with p replaced by the weak differentiability of the activation function.
We tested Conjecture 1 numerically by performing PCA on large post-activation matrices A (N,T ) . The linear and continuous dependence of the decay exponent α on the activation parameter p predicted by Eq. ( 12) was confirmed in simulations (Fig. 2B). Simulations also confirmed that the bias c of the activation function does not affect the decay rate (Fig. 2C), a fact already mentioned in [63,64].
To summarize, the spectral theory of random feature kernels suggests a three-way relationship between the power-law tail exponent of the post-activation eigenspectrum, the pre-activation dimension, and the activation function. This relationship should hold when we can consider the neurons as linearnonlinear functions of the latent vector, with weights that vary randomly and independently between neurons. The relationship suggests that, when high-dimensional neuronal activity (modeled here as post-activations) is observed [39,40], two scenarios are possible: high-dimensional activity could arise from nonlinear transformation of low-dimensional latent states, or it could reflect pre-activations that are already high-dimensional. To distinguish these two scenarios, we propose, in what follows, a method for inferring the pre-activation dimension of neuronal activity in experimental recordings.
this section cite: ['b55', 'b45', 'b55', 'b45', 'b46', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b38', 'b39']

Section: Latent Variable Modeling of Neuronal Recordings
To estimate the input dimensionality of neuronal activity in mouse visual cortex (as defined in Sec.
3) we developed the Neural Cross-Encoder (NCE), a nonlinear generalization of Reduced Rank Regression. Using NCE, we show that high-dimensional neuronal responses to drifting gratings 3 The k-th weak derivative of a function f : R → R is the defined as the function g ∈ L 1 loc that satisfies R φ(x)g(x)dx = (-1) k R φ (k) (x)f (x)dx, for all φ ∈ C ∞ c (R).
are well-approximated by a linear-nonlinear readout of a low dimensional latent variable, whereas responses to natural images are not. Finally, we apply NCE to high-dimensional spontaneous dynamics in the cortex and find that they are well-approximated by a linear-nonlinear readout of low-dimensional latents.
this section cite: []

Section: Experimental Data
We conducted large-scale volumetric two-photon microscopy on awake, adult mice during visual stimulation and spontaneous activity. We targeted primary and higher visual cortices with a Light Beads Microscope [65], and extracted deconvolved activity traces for 19,223 ± 2,948 neurons using Suite3D [66] as described in Appendix H. Recordings were performed in three stimulus conditions:
(1) responses to 320 full-field drifting grating stimuli with 2-14 repeats each; (2) responses to 1866 natural images with 2 repeats each; (3) spontaneous activity in the absence of stimuli for 10-15 minutes.
this section cite: ['b64', 'b65']

Section: Neural Cross-Encoder (NCE)
The Neural Cross-Encoder (NCE) divides neurons randomly into two sets: a source set and a target set. It predicts the activity b t of the target set from the source set a t via a non-linear readout of a set of latents, z t (Fig. 3A). NCE uses a multi-layer feedforward encoder E that ends in a bottleneck layer whose activity z t = E(a t ) represents a low-dimensional latent state estimated from the source neurons. The reason to use this rather than an autoencoder, which predicts one set of neurons from themselves, is to discard variability that is not shared across neurons. The NCE we used here has a single power-ReLU output layer, matching the setup of section 3, and a 3-layer encoder allowing flexible estimation of latent variables, so that the number of latent variables can be readily interpreted as the pre-activation dimension. We train NCE with stochastic gradient descent on source-target activity pairs as described in Appendix I. When all nonlinearities are removed, NCE becomes equivalent to Reduced Rank Regression [67]. When predicting stimulus-driven activity, we pair the activity of source and target neurons on different repeats of the same stimulus, to also discard shared variability that is not related to the stimulus [39].
Linear-nonlinear readout. The recorded activity of a set of target neurons at time t, bt ∈ R B , is modeled as a weighted sum of the latent variables, z t ∈ R d , passed through a nondecreasing nonlinearity:
b t = ϕ p,c (Uz t ) + r.(13)
Here, ϕ p,c is the rectified power activation function, defined in Eq. 9, with a power parameter p that is constant across neurons, a pre-activation bias that varies across neurons encoded by an N -dimensional vector c, and a post-activation added bias r to account for non-zero baseline firing rates. The decoder parameters {p, c, U, r} are learned alongside the encoder parameters of E. The fact that the decoder, Eq. ( 13), has a single-layer is crucial as it allows us to interpret the latents (z t ) as linear factors of the observed neurons' pre-activations (Uz t ). It is this constrained decoder that allows us to infer the pre-activation dimension of neuronal activity; in comparison, a multi-layer decoder as used in [13,23] would infer something closer to the intrinsic dimension of neuronal activity, which is not our goal.
this section cite: ['b66', 'b38', 'b12', 'b22']

Section: Results

this section cite: []

Section: NCE identifies the latent dimensionality of simulated data.
To validate that NCE can identify the pre-activation latent variables, we test it on simulated data generated from the toy model in Sec. 2 with a ReLU readout (p = 1) and d = 2. NCE recovers the true latents up to a scaling and a shift (Fig 3B). Moreover, NCE can explain all of the variance in the population with only two pre-activation dimensions, while the corresponding linear model (Reduced Rank Regression) requires more dimensions (Fig 3C).
Pre-activation dimension is low for grating responses, high for natural image responses. We next consider the pre-activation dimensional of visual stimulus responses of visual cortex neurons. To ensure that the NCE focused on the stimulus responses, and not correlated ongoing activity such as spontaneous activity or encoding of movements, the activity of the target cells and the source cells were taken from different repeats of the same stimuli. In the case of drifting gratings, for which we know there is a low-dimensional latent variable (the grating orientation), an NCE model with low-dimensional pre-activations accurately predicts neuronal responses (Fig. 4A,B). NCE requires fewer dimensions (5.5 ± 1.2, mean ± std) than the corresponding linear model (13.9 ± 3.0) to predict 95% of the explainable variance (defined as the maximum variance explained across all d in both models). On the other hand, NCE models with low pre-activation dimension are not sufficient to predict responses to natural images (Fig. 4C,D), requiring 93.9 ± 6.0 dimensions to reach the threshold, suggesting that natural images produce high-dimensional representations in the space of pre-activations (but see Limitations below). Linear models only account for a smaller fraction of the total variance (Fig. 4D), and therefore underestimate the dimensionality of natural image responses (48.0 ± 13.1).
this section cite: []

Section: Spontaneous activity has low pre-activation dimension.
Spontaneous activity is well predicted by NCE with low pre-activation dimension (Fig. 4E,F). Across all recordings, spontaneous activity of 1000 target neurons has an estimated pre-activation dimension d of 7.0 ± 1.0 (mean ± std.), somewhat larger than grating responses but substantially lower than natural images responses (Fig. 4G,H). The linear model finds a similar dimension (7.5 ± 2.0), though its performance deteriorates at high preactivation dimensions due to overfitting, while the NCE performance remains consistent (Fig. 4F).
These results indicate that visual cortex activity can be modeled as a linear-nonlinear transformation of a latent vector, which is low-dimensional for grating responses and spontaneous activity, but high-dimensional for natural image responses. In the case of grating responses, we find latents that resemble the sine and cosine of the stimulus angle (Fig. 4A)-this is what one would expect to find if neurons follow the canonical model of simple cells in visual cortex [68]. On the other hand, during spontaneous activity, the dynamics of the latent variables are correlated with the running speed of the mouse, and are perhaps related to its arousal state (see Appendix I.4).
this section cite: ['b67']

Section: Summary of Technical Contributions and Previous Works
Latent dynamics of low-rank RNNs Low-rank RNNs are tractable models of how the brain can perform computations through low-dimensional population dynamics [36,51,[69][70][71][72]. In particular, the dynamics of certain low-rank RNNs, in the large-network limit, reduce to that of "effective circuits", i.e., dynamical systems describing the evolution of the latent variables [51,52,73]. A limitation of these effective dynamical systems is that the expression of the vector field involves an integral over the distribution of weights (the "circuit structure" [52]), making them somewhat opaque and costly to solve numerically in general. In this work (Sec. 2.1 and Appendix E), we prove that, in several special cases that are beyond the case treated in [74], the integral over the weight distribution can be solved, yielding simple exact equations for the latent dynamics.
Eigenvalue decay of random feature kernels In the infinite-width limit, two-layer neural networks with random input weights behave like random feature kernels that depend on the distribution of input weights and the activation function of the neurons in the hidden layer [56, Sec. 9.5]. This functional perspective can be generalized to deep networks [45,75] and constitute the basis of the Neural Tangent Kernel formalism for studying learning dynamics [76]. When the activation function is the ReLU, the decay rate of the eigenvalues has been proven to be polynomial [46,47,58,64,77,78], even when inputs are not assumed to be uniformly distributed on the sphere [58]; for general results on dot-product kernels, see [57,61,63,79,80]. In this work, we propose a simple formula that links the power-law exponent of the eigenvalue decay rate, the power of the rectified-power activation function, and the input dimensionality. This formula, which goes beyond known results [46,47,64], is presented as a conjecture that we test in simulations.
Latent variable modeling of neuronal activity While most latent variable models of neuronal activity were originally developed for electrophysiological recordings [8][9][10][11][12][13][14][15][16][17][18][19][20][21][22][23], some are tailored for calcium recordings [81][82][83]. These models vary in their mechanistic interpretability: The inferred latents are either abstract variables, for example when the model's mapping from latents to neuronal activity involves a multi-layer neural network [13,23], or they can be interpreted as linear factors of the neurons' pre-activations, as in [9,10]. With nonlinear dimensionality reduction methods such as CEBRA [84] or Rastermap [85], latent variables are also abstract as there is no explicit mapping going from the latents to neuronal activity. In this work (Sec. 4), we developed NCE, a latent variable model for calcium recordings that models neuronal activity as an interpretable linear-nonlinear readout of latent variables. NCE also uses a cross-encoding scheme, which allows it to discard variability not shared across neurons. We demonstrate that NCE is capable of identifying a low-dimensional pre-activation space even when the recorded neuronal activity has high linear dimension.
this section cite: ['b35', 'b50', 'b68', 'b69', 'b70', 'b71', 'b50', 'b51', 'b72', 'b51', 'b73', 'b44', 'b74', 'b75', 'b45', 'b46', 'b57', 'b63', 'b76', 'b77', 'b57', 'b56', 'b60', 'b62', 'b78', 'b79', 'b45', 'b46', 'b63', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b80', 'b81', 'b82', 'b12', 'b22', 'b8', 'b9', 'b83', 'b84']

Section: Discussion

this section cite: []

Section: References
Ref_id:b0 Title: Neural population dynamics during reaching Year: (2012)
Ref_id:b1 Title: Context-dependent computation by recurrent dynamics in prefrontal cortex Year: (2013)
Ref_id:b2 Title: Motor cortex embeds muscle-like commands in an untangled population response Year: (2018)
Ref_id:b3 Title: Flexible sensorimotor computations through rapid reconfiguration of cortical dynamics Year: (2018)
Ref_id:b4 Title: Cortical control of arm movements: a dynamical systems perspective Year: (2013)
Ref_id:b5 Title: Computation through neural population dynamics Year: (2020)
Ref_id:b6 Title: Neural manifolds for the control of movement Year: (2017)
Ref_id:b7 Title: Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population activity Year: (2009)
Ref_id:b8 Title: Empirical models of spiking in neural populations Year: (2011)
Ref_id:b9 Title: Inferring single-trial neural population dynamics using sequential auto-encoders Year: (2018)
Ref_id:b10 Title: Learning interpretable continuous-time models of latent stochastic dynamical systems Year: (2019)
Ref_id:b11 Title: Recurrent switching dynamical systems models for multiple interacting neural populations Year: (2020)
Ref_id:b12 Title: Learning identifiable and interpretable latent models of highdimensional neural activity using pi-vae Year: (2020)
Ref_id:b13 Title: Inferring latent dynamics underlying neural population activity via neural differential equations Year: (2021)
Ref_id:b14 Title: iLQR-VAE : control-based learning of input-driven dynamics with applications to neural data Year: (2022)
Ref_id:b15 Title: A large-scale neural network training framework for generalized estimation of single-trial population dynamics Year: (2022)
Ref_id:b16 Title: Extracting computational mechanisms from neural data using low-rank RNNs Year: (2022)
Ref_id:b17 Title: Mesoscopic modeling of hidden spiking neurons Year: (2022)
Ref_id:b18 Title: The dynamics and geometry of choice in the premotor cortex Year: (2025)
Ref_id:b19 Title: Flow-field inference from neural data using deep recurrent networks Year: (2025)
Ref_id:b20 Title: Disentangling the roles of distinct cell classes with cell-type dynamical systems Year: (2024)
Ref_id:b21 Title: Latent circuit inference from heterogeneous neural responses during cognitive tasks Year: (2025)
Ref_id:b22 Title: MARBLE: interpretable representations of neural population dynamics using geometric deep learning Year: (2025)
Ref_id:b23 Title: Opening the black box: low-dimensional dynamics in highdimensional recurrent neural networks Year: (2013)
Ref_id:b24 Title: A neural network that finds a naturalistic solution for the production of muscle activity Year: (2015)
Ref_id:b25 Title: Optimal control of transient dynamics in balanced networks supports generation of complex movements Year: (2014)
Ref_id:b26 Title: Dynamic control of response criterion in premotor cortex during perceptual detection under temporal uncertainty Year: (2015)
Ref_id:b27 Title: Neural population dynamics during reaching are better explained by a dynamical system than representational tuning Year: (2016)
Ref_id:b28 Title: Local dynamics in trained recurrent neural networks Year: (2017)
Ref_id:b29 Title: Early selection of task-relevant features through population gating Year: (2023)
Ref_id:b30 Title: A unifying perspective on neural manifolds and circuits for cognition Year: (2023)
Ref_id:b31 Title: Reconstructing computational system dynamics from neural data with recurrent neural networks Year: (2023)
Ref_id:b32 Title: Flexible multitask computation in recurrent networks utilizes shared dynamical motifs Year: (2024)
Ref_id:b33 Title: Feedback control of recurrent dynamics constrains learning timescales during motor adaptation Year: (2024)
Ref_id:b34 Title: Slow dynamics and high variability in balanced cortical networks with clustered connections Year: (2012)
Ref_id:b35 Title: Linking connectivity, dynamics, and computations in low-rank recurrent neural networks Year: (2018)
Ref_id:b36 Title: The centrality of population-level factors to network computation is demonstrated by a versatile approach for training spiking networks Year: (2023)
Ref_id:b37 Title: Emergent rate-based dynamics in duplicate-free populations of spiking neurons Year: (2025)
Ref_id:b38 Title: High-dimensional geometry of population responses in visual cortex Year: (2019)
Ref_id:b39 Title: Spontaneous behaviors drive multidimensional, brainwide activity Year: (2019)
Ref_id:b40 Title: Cerebellar granule cell axons support high-dimensional representations Year: (2021)
Ref_id:b41 Title: Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number Year: (2024)
Ref_id:b42 Title: Interpreting neural computations by examining intrinsic and embedding dimensionality of neural activity Year: (2021)
Ref_id:b43 Title: Strong and weak principles of neural dimension reduction. Neurons, Behavior, Data analysis, and Theory Year: (2021)
Ref_id:b44 Title: Kernel methods for deep learning Year: (2009)
Ref_id:b45 Title: Breaking the curse of dimensionality with convex neural networks Year: (2017)
Ref_id:b46 Title: Deep equals shallow for ReLU networks in kernel regimes Year: (2021)
Ref_id:b47 Title: A model of the neural basis of the rat's sense of direction Year: (1994)
Ref_id:b48 Title: Theory of orientation tuning in visual cortex Year: (1995)
Ref_id:b49 Title: Representation of spatial orientation by the intrinsic dynamics of the headdirection cell ensemble: a theory Year: (1996)
Ref_id:b50 Title: Shaping dynamics with multiple populations in low-rank recurrent networks Year: (2021)
Ref_id:b51 Title: Linking neural manifolds to circuit structure in recurrent networks Year: (2024)
Ref_id:b52 Title: Spectra of euclidean random matrices Year: (1999)
Ref_id:b53 Title: Random matrix approximation of spectra of integral operators Year: (2000)
Ref_id:b54 Title: Eigenvalues of euclidean random matrices. Random Structures & Algorithms Year: (2008)
Ref_id:b55 Title: Learning theory from first principles Year: (2024)
Ref_id:b56 Title: A spectral analysis of dot-product kernels Year: (2021)
Ref_id:b57 Title: On the eigenvalue decay rates of a class of neural-network related kernel functions defined on general domains Year: (2024)
Ref_id:b58 Title: Eigenvalues of integral operators with smooth positive definite kernels Year: (1987)
Ref_id:b59 Title: Eigenvalue decay of positive integral operators on the sphere Year: (2012)
Ref_id:b60 Title: Sharp estimates for eigenvalues of integral operators generated by dot product kernels on the sphere Year: (2014)
Ref_id:b61 Title: Estimates for fourier sums and eigenvalues of integral operators via multipliers on the sphere Year: (2016)
Ref_id:b62 Title: Population codes enable learning from few examples by shaping inductive bias Year: (2022)
Ref_id:b63 Title: Taskdependent optimal representations for cerebellar learning Year: (2023)
Ref_id:b64 Title: High-speed, cortex-wide volumetric recording of neuroactivity at cellular resolution using light beads microscopy Year: (2021)
Ref_id:b65 Title: Suite3D: Volumetric cell detection for two-photon microscopy Year: (2025)
Ref_id:b66 Title: Reduced-rank regression for the multivariate linear model Year: (1975)
Ref_id:b67 Title: What simple and complex cells compute Year: (2006-12)
Ref_id:b68 Title: Dynamics of random recurrent networks with correlated low-rank structure Year: (2020)
Ref_id:b69 Title: The role of population structure in computations through neural dynamics Year: (2022)
Ref_id:b70 Title: High-dimensional dynamics in low-dimensional networks Year: (2025)
Ref_id:b71 Title: Stochastic activity in low-rank recurrent neural networks Year: (2025)
Ref_id:b72 Title: Local/global analysis of the stationary solutions of some neural field equations Year: (2010)
Ref_id:b73 Title: Trained recurrent neural networks develop phase-locked limit cycles in a working memory task Year: (2024)
Ref_id:b74 Title: Deep neural networks as gaussian processes Year: (2018)
Ref_id:b75 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b76 Title: On the similarity between the laplace and neural tangent kernels Year: (2020)
Ref_id:b77 Title: On the inductive bias of neural tangent kernels Year: (2019)
Ref_id:b78 Title: Regularization with dot-product kernels Year: (2000)
Ref_id:b79 Title: Mercer's theorem, feature maps, and smoothing Year: (2006)
Ref_id:b80 Title: Parallel inference of hierarchical latent dynamics in two-photon calcium imaging of neuronal populations Year: (2021)
Ref_id:b81 Title: A deep learning framework for inference of single-trial neural population dynamics from calcium imaging with subframe temporal resolution Year: (2022)
Ref_id:b82 Title: Dimensionality reduction of calcium-imaged neuronal population activity Year: (2023)
Ref_id:b83 Title: Learnable latent embeddings for joint behavioural and neural analysis Year: (2023-05)
Ref_id:b84 Title: Rastermap: a discovery method for neural population recordings Year: (2025)
Ref_id:b85 Title: Minute-scale oscillatory sequences in medial entorhinal cortex Year: (2024)
Ref_id:b86 Title: Priors for infinite networks. Bayesian learning for neural networks Year: (1996)
Ref_id:b87 Title: Computing with infinite networks Year: (1996)
Ref_id:b88 Title: Dimension of activity in random neural networks Year: (2023)
Ref_id:b89 Title: Neuronal firing rate diversity lowers the dimension of population covariability Year: (2024)
Ref_id:b90 Title: Faced 2.0 enables large-scale voltage and calcium imaging in vivo Year: (2025)
Ref_id:b91 Title: Revisiting the high-dimensional geometry of population responses in visual cortex Year: ()
Ref_id:b92 Title: A critical initialization for biological neural networks Year: (2025)
Ref_id:b93 Title: Ultrasensitive fluorescent proteins for imaging neuronal activity Year: (2013)
Ref_id:b94 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b95 Title: Adam: A method for stochastic optimization. arXiv Year: ()
Ref_id:b96 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
