Title: An Analytical Theory of Spectral Bias in the Learning Dynamics of Diffusion Models
Abstract: We develop an analytical framework for understanding how the generated distribution evolves during diffusion model training. Leveraging a Gaussian-equivalence principle, we solve the full-batch gradient-flow dynamics of linear and convolutional denoisers and integrate the resulting probability-flow ODE, yielding analytic expressions for the generated distribution. The theory exposes a universal inverse-variance spectral law: the time for an eigen-or Fourier mode to match its target variance scales as τ ∝ λ -1 , so high-variance (coarse) structure is mastered orders of magnitude sooner than low-variance (fine) detail. Extending the analysis to deep linear networks and circulant full-width convolutions shows that weight sharing merely multiplies learning rates-accelerating but not eliminating the bias-whereas local convolution introduces a qualitatively different bias. Experiments on Gaussian and natural-image datasets confirm the spectral law persists in deep MLP-based UNet. Convolutional U-Nets, however, display rapid near-simultaneous emergence of many modes, implicating local convolution in reshaping learning dynamics. These results underscore how data covariance governs the order and speed with which diffusion models learn, and they call for deeper investigation of the unique inductive biases introduced by local convolution.4 Learning in Diffusion Models with a Linear Denoiser Problem set-up. Throughout the paper, we assume the denoiser at each noise scale is linear (affine) and independent across scales:Since the parameters {W σ , b σ } are decoupled across noise scales, each σ can be analysed independently. Through further parametrization, this umbrella form captures linear residual nets, deep linear nets, and linear convolutional nets (see Sec. 5).We train on an arbitrary distribution p 0 with mean µ and covariance Σ by gradient flow on the full-batch DSM loss, i.e. the exact expectation over data and noise (2). (In practice, one cannot sample all z values, but the full-batch limit yields clean closed-form dynamics.)This setting lets us dissect analytically the role of data spectrum, model architecture (W σ parametrisation), and loss variant in shaping diffusion learning. Diffusion learning as ridge regressionGaussian equivalence. For any joint distribution p(X, Y ) the quadratic lossdepends on p only through the first two moments of (X, Y ); see App. C.1.1 for proof. Hence a linear denoiser trained on arbitrary p 0 interacts with the data solely via its mean µ and covariance Σ.Instance for diffusion. Under EDM loss (2), the noisy input-target pair is X = x 0 + σz, Y = x 0 , giving Σ XX = Σ + σ 2 I, Σ Y X = Σ.

Section: 
1 Introduction Diffusion models create rich data by gradually transforming Gaussian noise into signal, a paradigm that now drives state-of-the-art generation in vision, audio, and molecular design [1,2,3]. Yet two basic questions remain open. (i) Which parts of the data distribution do these models learn first, and which linger unlearned-risking artefacts under early stopping? (ii) How does architectural inductive bias shape this learning trajectory? Addressing both questions demands that we track the evolution of the full generated distribution during training and relate it to the network's parameterization.
We tackle the learning puzzle through the simplest tractable setting-linear denoisers-where datasets become equivalent to a Gaussian with matched mean and covariance. In this regime we solve, in closed form, the nested dynamics of gradient-flow of the weights and the probability-flow ODE that carries noise into data, leading to an analytical characterization of the evolution of the generated distribution. The analysis exposes an inverse-variance spectral law: the time required for an eigen-mode to match target variance scales like τ k ∝ λ -α k , so high-variance directions corresponding to global structure are mastered orders of magnitude sooner than low-variance, fine-detail directions. Extending the analysis to deep linear and linear convolutional nets, we show how convolutional architecture redirect this bias to Fourier or patch space, and accelerate convergence via weight sharing.
this section cite: ['b0', 'b1', 'b2']

Section: Main contributions 1.
Closed-form distribution dynamics. We derive exact weight and distributional trajectories for one-layer, two-layer linear, and convolutional denoisers under full-batch DSM training. 2. Inverse-variance spectral bias. The theory reveals and quantifies a spectral-law ordering of mode convergence, offering one mechanistic explanation for early-stop errors. 3. Empirical validation in nonlinear neural nets. Experiments on Gaussian and natural-image datasets confirm the spectral-law in deep MLP-based diffusion. 4. Convolutional architectural shape learning dynamics. Experiments on convolutional UNet, showing rapid patch-first learning dynamics different from fully-connected architectures.
2 Related Work and Motivation: Spectral Bias in Distribution Learning Spectral structure of natural data Many natural signals have interesting spectral structures (e.g. image [4], sound [5], video [6]). For natural images, their covariance eigenvalues decay as a power law, and the corresponding eigenvectors can align with semantically meaningful patterns [4]. For faces, for instance, leading eigenmodes capture coarse, low-frequency shape variations, whereas tail modes encode fine-grained textures [7,8]. Analyzing spectral effect on diffusion learning can therefore show which type of features the model acquires first and which remain slow to learn.
this section cite: ['b3', 'b4', 'b5', 'b3', 'b6', 'b7']

Section: Hidden Gaussian Structure in Diffusion Model
Recent work has shown, for most diffusion times, the learned neural score is closely approximated by the linear score of a Gaussian fit to the data, which is usually the best linear approximation [9,10]. Crucially, this Gaussian linear score admits a closed-form solution to the probability-flow ODE, which can be exploited to accelerate sampling and improve its quality [11]. Moreover, this same linear structure has been linked to the generalization-memorization transition in diffusion models [10]. In sum, across many noise levels, the Gaussian linear approximation is a predominant structure in the learned score. Thus, we hypothesize it will have a significant effect on the learning dynamics of score approximator. From this perspective, our contribution is to elucidate the learning process of this linear structure.
Learning theory for regression and deep linear networks Gradient dynamics in regression are well-studied, with spectral bias and implicit regularisation emerging as central themes [12,13,14]. In Sec. 4.1, we show that the loss of a linear diffusion model reduces to ridge regression, letting us import those results directly. Our analysis also builds on learning theory of deep linear networks (including linear-convolutional and denoising autoencoders) [15,16,17,18]. We extend these insights to modern diffusion-based generative models, offering closed-form description of how the generated distribution itself evolves during training.
this section cite: ['b8', 'b9', 'b10', 'b9', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17']

Section: Diffusion learning theory
Several recent theory studies address diffusion models from a spectral perspective but tackle different questions. [19,20,21,22] document spectral bias in the sampling process after training; our focus is on how that bias arises during training. [23] study stochastic sampling assuming an optimal score, orthogonal to our analysis of training dynamics. Sharing our interest in training, [24] analyze learning of mixtures of spherical Gaussians to recover component means, whereas we tackle anisotropic covariances and track reconstruction of the full covariance. [25] characterises optimal score and distribution under constraints; results from our convolutional setup can be viewed through that lens.
this section cite: ['b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24']

Section: Background

this section cite: []

Section: Score-based Diffusion Models
Let p 0 (x) be the data distribution of interest, and for each noise level σ > 0 define p(x; σ) = p 0 * N (0, σ 2 I) (x) = p 0 (y) N (x | y, σ 2 I) dy. The associated score function is ∇ x log p(x; σ), i.e. the gradient of the log-density at noise scale σ. In the EDM framework [26], one shows that the "probability flow" ODE dx dσ = -σ ∇ x log p(x; σ)
exactly transports samples from p( • ; σ T ) to p( • ; σ) as σ decreases. In particular, integrating from σ T down to σ = 0 recovers clean data samples from p 0 . We adopt the EDM parametrization for its notational simplicity; other common diffusion formalisms are equivalent up to simple rescalings of space and time [26]. To learn the score of a data distribution p 0 (x), we minimize the denoising score matching (DSM) objective [27] with a function approximator. We reparametrize the score function with the 'denoiser' s θ (x, σ) = (D θ (x, σ) -x)/σ 2 , then at noise level σ the DSM objective reads
Lσ = E x 0 ∼p 0 , z∼N (0,I) D θ (x0 + σz; σ) -x0 2 2 . (2
)
To balance the loss and importance of different noise scales, practical diffusion models all adopt certain weighting functions in their overall loss L = σ dσ w(σ) L σ .
this section cite: ['b25', 'b25', 'b26']

Section: Gaussian Data and Optimal Denoiser
To motivate our linear score approximator set up, it is useful to consider the optimal score and the denoiser of a Gaussian distribution. For Gaussian data x 0 ∼ N (µ, Σ), x 0 ∈ R d and Σ is a positive semi-definite matrix. When noising x 0 by Gaussian noise at scale σ, the corrupted x satisfies x ∼ N (µ, Σ + σ 2 I), for which the Bayes-optimal denoiser is an affine function of x.
D * (x; σ) = µ + (Σ + σ 2 I) -1 Σ(x -µ)(3)
For Gaussian data, minimizing (2) yields D * . This solution has an intuitive interpretation, i.e. the difference of the state x and distribution mean was projected onto the eigenbasis and shrinked mode-by-mode by λ k /(λ k + σ 2 ). Thus, according to the variance λ k along target axis, modes with variance significantly higher than noise λ k ≫ σ 2 will be retained; modes with variance much smaller than noise will be "shrinked" out. Effectively σ 2 defines a threshold of signal and noise, and modes below which will be removed. This intuition similar to Ridge regression is made exact in Sec. 4.1.
Gradient and optimum. Differentiating and setting gradients to zero yields
∇ Wσ L σ = -2Σ + 2W σ (Σ + σ 2 I) + ∇ bσ L σ µ ⊤ , ∇ bσ L σ =2 b σ -(I -W σ )µ , (5
)
W * σ = Σ(Σ + σ 2 I) -1 , b * σ =(I -W * σ )µ,(6)
min L σ = σ 2 Tr Σ(Σ + σ 2 I) -1 .
Thus the optimal linear denoiser reproduces the denoiser for the Gaussian approximation of data (3) , and its best achievable loss is set purely by the data spectrum.
Other objectives. While the main text focuses on the EDM loss (2), we have worked out the gradients, optima, and learning dynamics for several popular variants used in diffusion and flow-matching [28] literature; these results are summarised in Tab. C.4 (derivations in App. C.4).
this section cite: ['b27']

Section: Ridge viewpoint.
Because
L σ = E x∼p0 Wx + b -x 2 + σ 2 ∥W∥ 2 F
, full-batch diffusion at noise scale σ is simply auto-encoding with ridge regularisation of strength σ 2 (App. C.2.1; cf. [29]). We will exploit classic ridge-regression results when analyzing learning dynamics in the following sections.
this section cite: ['b28']

Section: Weight Learning Dynamics of a Linear Denoiser
With the gradient structure in hand, we solve the full-batch gradient-flow ODE,
dW σ dτ = -η∇ Wσ L σ , db σ dτ = -η∇ bσ L σ ,(GF)
where τ is training time and η the learning-rate.
Zero-mean data (µ = 0): Exponential convergence mode-by-mode Because the gradients to W, b decouple (5), the dynamics is simplified on the eigenbasis of the covariance. We diagonalize the covariance, Σ = Interpretation. Each eigenmode projection of the weight W σ u k converges to the optimal value W * σ u k exponentially with rate (σ 2 + λ k ); hence (i) the weights at larger noise σ generally converge faster;
(ii) at a fixed σ, high-variance λ k modes converge first, while modes buried beneath the noise floor (λ k ≪ σ 2 ) share the same slower timescale. Fig. 2A illustrates this spectrum-ordered convergence, with high-variance modes reaching their optima before the low-variance ones (see also 5A).
Non-centred data (µ ̸ = 0): Interaction of mean and covariance learning. A non-zero mean introduces a rank-one coupling between W and b (matrix M in Prop D.1). Eigenmodes of weights overlapping with the mean (u ⊤ k µ ̸ = 0) now interact with b, producing transient overshoots and other non-monotonic effects; orthogonal modes retain the exponential convergnece above. App. D.2 gives the full linear-system analysis and two-dimensional visualisations (Fig. 27).
this section cite: []

Section: Sampling Dynamics during Training
For diffusion models, our goal is the generated distribution, obtained by integrating the probability-flow ODE (PF-ODE) backwards from a large σ T to a σ min ≈ 0, dx dσ = -σ -1 (W σ -I)x + b σ ,
initialized with Gaussian noise x T ∼ N (0, σ 2 T I). For linear denoiser, the PF-ODE is an inhomogeneous affine system, so its solution x σ is necessarily an affine function of the initial state x T [30], x(σ 0 ) = A(σ 0 ; σ T ) x(σ T ) + c(σ 0 ; σ T ). Since the map is affine, the distribution of x(σ 0 ) remains Gaussian, with covariance σ 2  T A(σ 0 ; σ T )A ⊤ (σ 0 ; σ T ). However, in general, the state-transition matrix A(σ 0 ; σ T ) is hard to evaluate, as it involves time-ordered matrix exponential, and the weight matrices at different noise scales W σ may not commute. The analysis below-and our closed-form results-hinges on situations where commutativity is maintained by gradient flow or architectural bias, thus removing the time-ordering operator. Interpretation. For each common eigenvector u k , the term (u ⊤ k W σ u k -1)/σ is the instantaneous expansion (or contraction) rate of the sample variance along u k ; the final variance is obtained by integrating this rate over noise scales σ (see App. C.5).
When does commutativity hold? This arises in three common settings. (i) At convergence, this is satisfied by the optimal weights W * σ (6), which jointly diagonalize on eigenbasis of Σ. In such case, we recover the the closed-form solution to PF-ODE for Gaussian data, as found by [9,31]. (ii) During training of linear denoisers, if weights are initialized to be aligned with eigenbasis of Σ, then gradient flow keeps them aligned, preserving commutativity (iii) For linear convolutional denoisers, circulant weights share the Fourier basis and commute by construction (see Sec. 5.2). In these cases, the sampling process can be understood mode-by-mode. Here we show the explicit solution for one layer linear denoiser. Proposition 4.2 (Dynamics of generated distribution in one layer case). Assume (i) zero-mean data, (ii) aligned initialization W σ (0) = k Q k u k u ⊤ k , and (iii) gradient flow, full-batch training with learning rate η. Then, while training the one-layer linear denoiser, the generated distribution at time τ is N (μ, Σ) with Σ = k λk (τ )u k u ⊤ k and
λk (τ ) = σ 2 T Φ 2 k (σ0, τ ) Φ 2 k (σT , τ ) , Φ k (σ, τ ) = λ k + σ 2 exp 1-Q k 2 Ei -2ητ σ 2 e -2ητ λ k -1 2 Ei -2ητ (σ 2 +λ k )
where Ei is the exponential-integral function. (derivation in App. D.3) Spectral bias. Figure 2B traces the variance trajectory λk (τ ) for each eigen-mode. All modes begin with the same initialization-induced level, then follow sigmoidal curves to their targets, but in descending order of λ k We define the first-passage time τ * k as the training time at which λk (τ ) reaches the geometric (or harmonic) mean of its initial and target values. We find the first-passage time obeys an inverse law τ * k ∝ λ -α k , α ≈ 1, (Fig. 2C), which implies that learning a mode with variance 1/10 smaller takes roughly 10 times longer to converge. With larger weight initialization (larger Q k ), the initial variance is closer to the target variance of some modes, then the inverse law splits into separate branches for modes with rising vs. decaying variance (Fig. 5B, Fig. 6).
Practical implication. This suggests when training stops earlier, the distribution in higher variance PC spaces have already converged, while low-variance ones-often the perceptual finer points such as letter strokes or finger joints-are under-trained. This could be an explanation for the familiar "wrong detail" artefacts in diffusion samples.
this section cite: ['b29', 'b8', 'b30']

Section: Deep and Convolutional Extensions
After analyzing the simplest linear denoiser, we set out to examine the effect of architectures via different parametrizations of the weights, specifically deeper linear models and linear convolutional networks. In the following, we will assume µ = 0 and focus on learning of covariance.
this section cite: []

Section: Deeper linear network
Consider a depth-L linear denoiser D(x, σ) = W L • • • W 1 x , where-for notational clarity-we suppress the explicit σ-dependence of weights. We assume aligned initialization, where for singular decomposition of each matrix, W ℓ (0) = U ℓ Λ ℓ V ⊤ ℓ , the right basis of each layer matching the left basis of the next, V ℓ+1 = U ℓ , ∀ℓ = 1, . . . , L -1, and with U L = V 1 = U where U diagonalizes data covariance Σ. Then the total weight at initialization is
W tot (0) = L ℓ=1 W ℓ (0) = U ( L ℓ=1 Λ ℓ )U ⊤ ,
With aligned initialization, every eigenmode learns independently-mirroring classical results [15,32]. In our case, this also implies that the total weight l W l shares the eigenbasis U across training and noise scales, thus commute, making sampling tractable.
One especially illuminating case is the two-layer symmetric network, where D(x, σ) = P σ P ⊤ σ x. Proposition 5.1 (Dynamics of weight and distribution in two layer linear model). Assume (i) centered data µ = 0; (ii) the weight matrix is initialized aligned, i.e. P σ (0)P σ (0
) ⊤ = k Q k u k u ⊤
k , then the gradient flow ODE admits a closed-form solution (derivation in App. E.1)
Wσ(τ ) = Pσ(τ )Pσ(τ ) ⊺ = k λ k σ 2 + λ k u k u ⊺ k Q k ( λ k σ 2 +λ k -Q k )e -8ηλ k τ + Q k (8
)
The generated distribution at time τ is N (μ, Σ) with Σ = k λk (τ )u k u ⊤ k and λk (τ ) = σ 2
T Φ 2 k (σ0) Φ 2 k (σ T ) Φ k (σ) = (σ) (1-Q k )e -8ητ λ k Q k +(1-Q k )e -8ητ λ k λ k e -8ητ λ k + Q k 1 -e -8ητ λ k λ k + σ 2 Q k 2Q k +2(1-Q k )e -8ητ λ k
Interpretation. The learning dynamics of weights and variance along different principal components are visualized in Fig. 2 D-F. Compared to one-layer case, here, the weight converges along the PCs via sigmoidal dynamics, with the emergence time (reaching harmonic mean of initial and final value) τ * k = ln 2/(8η λ k ). As for generated distribution, we find similar relationship between the target variance and emergence time
τ * k ∝ λ -α k , α ≈ 1.
For the more general non-aligned initialization, we show the non-aligned parts of weight will follow non-monotonic rise-and-fall dynamics (App. E.1.2). Extensions to non-symmetric two layer model and deeper model were studied in App. F, which have similar bias but lack clean expressions.
this section cite: ['b14', 'b31']

Section: Linear convolutional network
We consider a linear denoiser with convolutional architecture, D(x, σ) = w σ * x where samples x ∈ R N have 1d spatial structure, and a width K convolution filter w σ operates on it. The analysis could be easily generalized to 2d convolution. With circular boundary condition, w σ defines a circulant weight matrix W σ ∈ R N ×N , where w σ * x = W σ x. One favorable property of circulant matrices is that they are diagonalized by discrete Fourier transform F [33].
Wσ = F ΓσF * F mk := 1 √ N exp -2πi mk N (9
)
Thus all weights W σ commutes, which allows us to leverage Lemma 4.1, and solve the sampling dynamics mode-by-mode on the Fourier basis, leading to following result. Proposition 5.2. Linear convolutional denoisers with circular boundary can only model stationary Gaussian processes (GP), with independent Fourier modes, proof in App.G.2.
Learning dynamics of full-width filter K = N When convolution filter w σ is as large as the signal, the gradient flow is diagonal and unconstrained in the Fourier domain. Thus, the analyses in Sec. 4 re-emerge with variance of Fourier mode Σkk taking the place of λ k . Proposition 5.3 (Full-width circular convolution learning dynamics). Let D(x, σ) = w σ * x, with full-width filter K = N , and train w σ by full-batch gradient flow at rate η. Then the weights at noise σ and its spectral representation γ evolves as
wσ(τ ) = 1 √ N F * γ(τ, σ) ; γ k (τ, σ) = γ * k (σ) + γ k (τ, σ) -γ * k (σ) e -2N η(σ 2 + Σkk )τ (10
)
where γ * k (σ) = Σkk /(σ 2 + Σkk ) and Σkk = [F * ΣF ] kk is the variance of Fourier mode. The generated distribution has diagonal covariance in the Fourier basis and follows exactly Prop. 4.2 after the replacement λ k → Σkk , η → N η, U → F . (derivation in App. G.3) Interpretation. The weight and distribution dynamics mirror the fully-connected case, with spectral bias towards higher variance Fourier modes; convolutional weight sharing simply multiplies every rate by N , accelerating convergence without altering the inverse-variance law.
Notably, the learned distribution is asymptotically equivalent to the Gaussian approximation to the original training data with all possible spatial shifts as augmentations (proof in App. G.3.2). This is one case where equivariant architectural constraints facilitates creativity as discussed in [25].
Similarly, two-layer linear conv net with full-width filter can be treated as in Sec.5.1.
Learning dynamics of local filter K < N When the convolution filter has a limited bandwidth K ̸ = N , the Fourier domain dynamics get constrained, so it is easier to work with the filter weights.
Let r be the half-width of the kernel (K = 2r + 1). Define the circular patch extractor P r (x) = x i-r: i+r N i=1 ∈ R K×N , and the patch covariance Σ patch = 1 N E x P r (x) P r (x) ⊤ ∈ R K×K . Proposition 5.4 (Patch-convolution learning dynamics). For the circular convolutional denoiser, D(x, σ) = w σ * x trained by full-batch gradient flow with step size η. Let e 0 ∈ R K be the one-hot vector with a single 1 at the center position r + 1 (1-indexed). (derivation in App. G.4)
wσ(τ ) = w * σ + exp -2N ητ (σ 2 I + Σpatch) wσ(0) -w * σ , w * σ = (σ 2 I + Σpatch) -1 Σpatche0.
Interpretation. Training with a narrow convolutional filter reduces to ridge regression in patch space. Under gradient flow, filter converges along eigenmodes of patch Σ patch : modes with larger variance converges sooner, those with smaller variance later, preserving the inverse-variance law. It also enjoys the N times speed up given by weight sharing, accelerating progress without altering the ordering. The sampling ODE remains diagonal in Fourier space, so the generated distribution will be a stationary Gaussian process with local covariance structure shaped by the learned patch denoiser, though its exact form needs numerical integration to spell out. This setting is similar to the equivariant and local score machine described in [25], but with the additional linear constraint.
this section cite: ['b32', 'b24', 'b24']

Section: Simulation.
We numerically simulated the dynamics of the sample distribution for linear patchconvolution denoisers using FFHQ dataset (details in App. B.1.2). The spectral scaling exponents depend systematically on the convolutional patch size P : smaller kernels produced shallower, and in some cases even inverted, scaling relations (Tab. 2), potentially due to stronger coupling between more Fourier modes.
this section cite: []

Section: Empirical Validation of the Theory in Practical Diffusion Model Training
General Approach To test our theoretical predictions about the evolution of generated distribution (esp. covariance), we resort to the following method: 1) we fix a training dataset {x i } and compute its empirical mean µ and covariance Σ. We then perform an eigen-decomposition of Σ, obtaining eigenvalues λ k and eigenvectors u k . 2) Next, we train a diffusion model on this dataset by optimizing the DSM objective with a neural network denoiser D θ (x, σ). 3) During training, at certain steps τ , we generate samples {x τ i } from the diffusion model by integrating the PF-ODE (1). We then estimate the sample mean μτ and sample covariance Στ . Finally, we compute the variance of the generated samples along the eigenbasis of training data, λτ k = u ⊺ k Στ u k . To stress test our theory and maximize its relevance, we'd keep most of the training hyperparameters as practical ones.
this section cite: []

Section: Multi-Layer Perceptron (MLP)
To test our theory about linear and deep linear network (Prop.4.2,5.1), we used a Multi-Layer Perceptron (MLP) inspired by the SongUnet in EDM [26,34] (details in App. I.2). We found
1 41 2 17 32 39 08 11 28 0 27 15 2 48 91 2 10 00 00 Training Step 0 500 1000 1500 2000 2500 3000 Eigenmode index Evolution of eigen projected variance -6 -4 -2 0 2 4 log(Variance) C.
10 0 10 1 10 2 10 3 10 4 10 5 Training step 10 0 10 1 Variance Eig5 = 29.38 Eig15 = 6.81 Eig25 = 3.68 Eig35 = 2.41 Eig45 = 1.74 Eig55 = 1.35 Eig65 = 1.07 Eig75 = 0.85 Eig85 = 0.74 Eig95 = 0.63 Variance of learned patches (32x32, stride=1) on true eigenbas FFHQ32_UNet_MLP_EDM_8L_3072D_lr1e-4 A.
this section cite: ['b25', 'b33']

Section: D.
Variance along data cov eigenbasis this architecture effective in learning distribution like point cloud data (Fig. 29). We kept the preconditioning, loss weighting and initialization the same as in [26].
Experiment 1: Zero-mean Gaussian Data x MLP We first consider a zero mean Gaussian N (0, Σ) in d dimension as training distribution, with covariance defined as a randomly rotated diagonal matrix with log normal spectrum (details in App. I.4). During training, the generated variance of each eigenmode follows a sigmoidal trajectory toward its target value λ k ; modes with larger λ k cross the plateau sooner (Fig. 10A). We mark the emergence time τ * as the step at which the variance reaches the geometric mean of its initial and asymptotic values (Fig. 10B). Across both high-and low-variance modes, τ * obeys an inverse power-law, τ * ∝ λ -α k . With higher-dimensional Gaussians the exponent is estimated more precisely and remains close to 1: for d = 256, α = 1.08; for d = 512, α incr = 1.05 and α decr = 1.13 (Fig. 10C). The scaling breaks down only for modes whose initial variance is already near λ k ; in that regime the trajectory is less sigmoidal and τ * becomes ill-defined. This result shows that despite many non-idealistic conditions e.g. deeper network, nonlinear activation function, residual connections, normal weights initialization, shared parametrization of denoisers at different noise level, the prediction from the linear network theory is still quantitatively correct. Experiment 2: Natural Image Datasets x MLP Next, we validated our theory on natural image datasets. We flattened the images as a vectors, and trained a deeper and wider MLP-UNet to learn the distribution. Using FFHQ as our running example, monitoring the generated samples throughout training (Fig. 3A), despite heavy noise early on, the coarse facial contours-corresponding to the mean and top principal components of human face distribution [7]-emerge quickly, whereas high-frequency details (lower PCs) only appear later. We note that this spectral ordering effect of training dynamics is reminiscent and similar to that in the sampling dynamics after training [19,22].
Quantitatively, the sample covariance Στ rapidly aligns with and becomes close to diagonal in the data eigenbasis U (Fig. 11). The top eigenmodes' variances, λk (τ ), follow sigmoidal trajectories converging to their targets, and their "emergence times" τ * k increase down the spectrum (Fig. 3B,C). We exclude a central band of modes whose initial variances lie within 0.5-2× the target, since their undulating learning dynamics make first-passage time estimates unreliable. After this exclusion, modes with increasing and decreasing variance each exhibit a clear power-law scaling between emergence step τ * and target variance λ k , with exponents -0.48 (R 2 = 0.97, N = 57) and -0.35 (R 2 = 0.92, N = 2, 914), respectively (Fig. 3D). Although the observed spectral bias is slightly attenuated relative to the Gaussian case and linear theory prediction, it remains robust and consistent across datasets (MNIST, CIFAR-10, FFHQ32 and AFHQ32) (App. B.2.2). This shows that even with natural image data, the distributional learning dynamics of MLP-based diffusion still suffers from slower convergence speed for lower eigenmodes.
this section cite: ['b25', 'b6', 'b18', 'b21']

Section: Convolutional Neural Networks (CNNs)
Next we turn to the convolutional U-Net-the work-horse of image-diffusion models [34,35]. For a full-width linear convolutional network our analysis predicts an inverse-variance law in Fourier space (Prop. 5.3). The patch-convolution variant lacks a clear forecast on distribution, so the following experiments probe empirically whether-and how-its learning dynamics is affected by spectral bias.
this section cite: ['b33', 'b34']

Section: Experiment 3: Natural Image Datasets x CNN UNet
Training on the same FFHQ dataset, the distributional learning trajectory of CNN-UNet is markedly different from the MLPs: early in training, we do not see contour of face, but locally coherent patches, reminiscent of Ising models (Fig. 4A.). Visually and variance-wise, the CNN-based UNet converge much faster and better than the MLP-based UNet, matching the N-fold speed-up from weight sharing (Prop. 5.4; Fig. 4B). When projecting onto the data eigenbasis, all eigenmodes with increasing variance rise simultaneously, while eigenmodes with decreasing variance co-decay at a later time, giving an effective power-law exponent α ≈ 0; Thus, spectral bias is essentially absent (Fig. 13C.D.).
Why is spectral bias absent? On the theory side, the likely cause is locality: local convolutional filters couple neighbouring pixels, binding many Fourier modes into one learning unit. Because sampling remains diagonal in Fourier space, a broad band of modes is amplified simultaneously, attenuating the spectral ordering, as we observed numerically in App. B.1.2. In line with this, early in training, the CNN denoiser is indeed well-approximated by a local linear filter (Fig. 15).
On the empirical side, the key factor appears to be network width. We systematically varied the channel number and depth of deep convolutional denoisers (App. B.2.4), and found that-regardless of depth-narrower networks (e.g. ch = 4) exhibit slower convergence and a stronger spectral ordering, consistent with the patch-convolution theory (Fig. 22,23). In contrast, wide networks with many channels (ch = 128) learn spectral modes almost instantaneously, similar to our observations in practical UNet. In hindsight, the analytic theory effectively assumes a convolution with the same number of channels as the input (e.g., RGB = 3), so the ratio between the network's channel and the input channel likely governs the deviation from theoretical predictions.
A complete analytic treatment of convolutional U-Net training dynamics is left for future work.
this section cite: []

Section: Discussion
In summary, we presented closed-form solutions for training denoisers with linear, deep linear or linear convolutional architecture, under the DSM objective on arbitrary data. This setup allows for a precise mode-wise understanding of the gradient flow dynamics of the denoiser and the evolution of the learned distribution: covariance eigenmode for deep linear network and Fourier mode for convolutional networks. For both the weights and the distribution, we showed analytical evidence of spectral bias, i.e. weights converge faster along the eigenmodes or Fourier modes with high variance, and the learned distribution recovers the true variance first along the top eigenmodes. These theoretical results are summarized in Tab. 1.
We hope these results can serve as a solvable model for spectral bias in the diffusion models through the nested training and sampling dynamics. Furthermore, our analysis is not limited to the diffusion and the DSM loss, in App. H, we showed a similar derivation for the spectral bias in flow matching models [28,36].
Relevance of our theoretical assumptions We found, for the purpose of analytical tractability, we made many idealistic assumptions about neural network training, 1) linear neural network, 2) small or orthogonal weight initialization, 3) "full-batch" gradient flow, 4) independent evolution of weights at each noise scale. In our MLP experiments, we found even when all of these assumptions were somewhat violated, the general theoretical prediction is correct, with modified power coefficients. This shows most of these assumptions could be relaxed in real life, and the spectrum of data indeed have a large effect on the learning dynamics, esp. for fully connected networks.
Inductive bias of the local convolution In our CNN experiments, however, the theoretical predictions from linear models deviate: the spectral bias in learning speed does not directly apply to the distribution of full images. Although our theory predicts that filter-weight learning dynamics are governed by the patch covariance, the ultimate image distribution is shaped by the convolution of those filters. To date, many learning-theory analyses for diffusion models assume MLP-like architectures [24]. For future theoretical work on the learning dynamics of practical diffusion models, a rigorous treatment of the local convolutional structure-and its frequency-coupling effects-will likely be essential, rather than relying on full-width convolution analyses [37].
Implications for high channel inputs Our ablation suggests that the ratio between network width and input channel count may underlie the observed deviations from theoretical predictions-specifically, the absent of spectral bias and the near-simultaneous convergence of eigenmodes. While most image and latent representations traditionally have few channels (e.g., 3 for RGB, 4 for latent diffusion [38]), recent architectures employ much higher channel counts-such as DC-AEs with 64-128 channels [39] or encoder-based diffusion models with ch = 768 [40]. In these regimes, where input channel becomes comparable to that of the UNet or DiT, the theory's predictions may become increasingly relevant for understanding, regularizing, and stabilizing training dynamics in high-channel-dimensional diffusion models.
Broader Impact Although our work is primarily theoretical, the inverse scaling law could offer valuable insights into how to improve the training of large-scale diffusion or flow generative models.
this section cite: ['b27', 'b35', 'b23', 'b36', 'b37', 'b38', 'b39']

Section: References
Ref_id:b0 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b1 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b2 Title: Diffusion models beat gans on image synthesis Year: ()
Ref_id:b3 Title: Statistics of natural image categories Year: (2003)
Ref_id:b4 Title: Temporal low-order statistics of natural sounds Year: (1996)
Ref_id:b5 Title: Statistics of natural time-varying images. Network: computation in neural systems Year: (1995)
Ref_id:b6 Title: Eigenfaces for recognition Year: (1991)
Ref_id:b7 Title: A geometric analysis of deep generative image models and its applications Year: (2021)
Ref_id:b8 Title: The Hidden Linear Structure in Score-Based Models and its Application. arXiv e-prints, art Year: (2023-11)
Ref_id:b9 Title: Understanding generalizability of diffusion models requires rethinking the hidden gaussian structure Year: (2024)
Ref_id:b10 Title: The unreasonable effectiveness of gaussian score approximation for diffusion models and its applications Year: (2024-12)
Ref_id:b11 Title: On the spectral bias of neural networks Year: (2019)
Ref_id:b12 Title: Spectrum dependent learning curves in kernel regression and wide neural networks Year: (2020-07)
Ref_id:b13 Title: Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks Year: (2021-05)
Ref_id:b14 Title: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks Year: (2013)
Ref_id:b15 Title: Implicit bias of gradient descent on linear convolutional networks Year: (2018)
Ref_id:b16 Title: Learning dynamics of linear denoising autoencoders Year: (2018)
Ref_id:b17 Title: A unifying view on implicit bias in training linear neural networks Year: (2020)
Ref_id:b18 Title: Diffusion is spectral autoregression Year: (2024)
Ref_id:b19 Title: Generative modelling with inverse heat dissipation Year: (2022)
Ref_id:b20 Title: Wavelet score-based generative modeling Year: (2022)
Ref_id:b21 Title: Diffusion models generate images like painters: an analytical theory of outline first, details later Year: (2023)
Ref_id:b22 Title: Dynamical regimes of diffusion models Year: (2024)
Ref_id:b23 Title: Learning mixtures of gaussians using the ddpm objective Year: (2023)
Ref_id:b24 Title: An analytic theory of creativity in convolutional diffusion models Year: (2024)
Ref_id:b25 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b26 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b27 Title: Flow matching guide and code Year: (2024)
Ref_id:b28 Title: Training with noise is equivalent to tikhonov regularization Year: (1995)
Ref_id:b29 Title: Ordinary differential equations Year: (2002)
Ref_id:b30 Title: Diffusion models for gaussian distributions: Exact solutions and wasserstein errors Year: (2024)
Ref_id:b31 Title: Effect of batch learning in multilayer neural networks Year: (1998)
Ref_id:b32 Title: Toeplitz and circulant matrices: A review Year: (2006)
Ref_id:b33 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b34 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b35 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b36 Title: Implicit bias of gradient descent on linear convolutional networks Year: (2018)
Ref_id:b37 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b38 Title: Deep compression autoencoder for efficient high-resolution diffusion models Year: (2024)
Ref_id:b39 Title: Diffusion transformers with representation autoencoders Year: (2025)
Ref_id:b40 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b41 Title: Blink of an eye: a simple theory for feature localization in generative models Year: (2025)
Ref_id:b42 Title: Pulkit Agrawal, and Phillip Isola. The low-rank simplicity bias in deep networks Year: (2021)
Ref_id:b43 Title: Implicit rank-minimizing autoencoder Year: (2020)
Ref_id:b44 Title: Expandnets: Linear overparameterization to train compact convolutional networks Year: (2020)
Ref_id:b45 Title: Inductive bias of multi-channel linear convolutional networks with bounded weight norm Year: (2022)
Ref_id:b46 Title: Geometry of linear convolutional networks Year: (2022)
Ref_id:b47 Title: Function space and critical points of linear convolutional networks Year: (2024)
Ref_id:b48 Title: Deep image prior Year: (2018)
Ref_id:b49 Title: The spectral bias of the deep image prior Year: (2019)
Ref_id:b50 Title: A bayesian perspective on the deep image prior Year: (2019)
Ref_id:b51 Title: The convergence rate of neural networks for learned functions of different frequencies Year: (2019)
Ref_id:b52 Title: Frequency bias in neural networks for input of non-uniform density Year: (2020)
Ref_id:b53 Title: Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks Year: (2020)
Ref_id:b54 Title: Denoising and regularization via exploiting the structural bias of convolutional generators Year: (2019)
Ref_id:b55 Title: Rank-one modification of the symmetric eigenproblem Year: (1978)
Ref_id:b56 Title: A stable and efficient algorithm for the rank-one modification of the symmetric eigenproblem Year: (1994)
Ref_id:b57 Title: Some modified matrix eigenvalue problems Year: (1973)
Ref_id:b58 Title: A proposal for Toeplitz matrix calculations Year: (1986)
Ref_id:b59 Title: On the solution of circulant linear system Year: (1985)
Ref_id:b60 Title: Presentation slides (mastronardi.pdf Year: (2007)
Ref_id:b61 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
