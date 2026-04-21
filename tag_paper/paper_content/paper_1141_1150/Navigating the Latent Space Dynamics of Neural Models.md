Title: NAVIGATING THE LATENT SPACE DYNAMICS OF NEURAL MODELS
Abstract: Neural networks transform high-dimensional data into compact, structured representations, often modeled as elements of a lower dimensional latent space. In this paper, we present an alternative interpretation of neural models as dynamical systems acting on the latent manifold. Specifically, we show that autoencoder models implicitly define a latent vector field on the manifold, derived by iteratively applying the encoding-decoding map, without any additional training. We observe that standard training procedures introduce inductive biases that lead to the emergence of attractor points within this vector field. Drawing on this insight, we propose to leverage the vector field as a representation for the network, providing a novel tool to analyze the properties of the model and the data. This representation enables to: (i) analyze the generalization and memorization regimes of neural models, even throughout training; (ii) extract prior knowledge encoded in the network's parameters from the attractors, without requiring any input data; (iii) identify out-of-distribution samples from their trajectories in the vector field. We further validate our approach on vision foundation models, showcasing the applicability and effectiveness of our method in real-world scenarios.

Section: INTRODUCTION
Figure 1: Latent dynamics of AEs. Latent vector fields induced by autoencoders with bottleneck k = 2, trained on MNIST, with z 0 ∼ U [-8, 8]. Models with different initializations are shown. Colors (viridis colormap) represent vector norms ranging from violet (low) to yellow (high). The shape of the latent manifold identifies with the encoder's support. White regions indicate where the vector field vanishes, revealing attractors aligned with high-density areas of the data distribution.
Neural networks are powerful function approximators, capable of solving complex tasks across a wide range of domains (Jumper et al., 2021;Radford et al., 2021). A core component of their success lies in their ability to transform high-dimensional inputs into compact, structured representations (Bengio et al., 2013), typically modeled as points in a lower-dimensional latent space.
In this work, we propose a novel perspective: for any given autoencoder (AE) architecture, there exists an associated latent vector field (see Figure 1) that characterizes the model's behavior in 2 METHOD Notation and background We consider neural models F as compositions of encoder-decoder modules, defined as
F Θ = D θ2 • E θ1 parametrized by Θ = [θ 1 , θ 2 ].
The encoder E θ1 maps inputs x ∼ p(x) supported on X ⊂ R m to a typically lower-dimensional space Z ⊂ R k , and the decoder D θ2 reconstructs the input. The model is trained to minimize the mean squared error L M SE :
L M SE (x) = x∈X ∥x -F Θ (x)∥ 2 2 + λR(Θ) , (1
)
where R is either an explicit or implicit regularization term encouraging contractive behavior in F .
We posit that minimizing this objective leads to a reduction in the spectral norm of the Jacobian, ∥J F (x)∥ σ for x ∼ p(x). Examples of regularization include weight decay, where R = ∥Θ∥ F 2 , or data augmentation by sampling transformations T ∼ p(T ) and minimizing:
L M SE (x) = x∈X ∥x -F (x)∥ 2 2 + T ∈p(T ) ∥x -F (T x)∥ 2 2 .(2)
Such augmentations include additive Gaussian noise in denoising AEs (Vincent et al., 2008) and input masking in masked AEs (He et al., 2022). Another form of contractive pressure is the bottleneck dimension k = dim(Z), which places a hard upper bound on the rank of the encoder Jacobian, rank(J E ) <= k and, by the chain rule, constrains the full model Jacobian J F (•) = J E (•) ⊙ J D (•).
Table 3 (Appendix) lists many AE variants, including denoising AEs (DAEs) (Vincent et al., 2008), sparse AEs (SAEs) (Ng et al., 2011), variational AEs (VAEs) (Kingma et al., 2013), and others (Rifai et al., 2011;Alain and Bengio, 2014;Gao et al., 2024), highlighting how their objectives promote local contractive behavior around training data.
Given a possibly pretrained AE model, we define the map f (z) = E • D(z) and study its repeated application f (. . . f (f (z))), which can be modeled as a differential equation.
this section cite: ['b26', 'b45', 'b5', 'b59', 'b20', 'b59', 'b40', 'b29', 'b48', 'b1', 'b15']

Section: THE LATENT DYNAMICS OF NEURAL MODELS
Given a sample z ∈ Z ⊆ R k , we study the effect of repeatedly applying the map f , i.e., f •f •. . . f (z). By introducing a discrete time parameter t, this iterative process defines the discrete ODE:
z t+1 = f (z t ) z 0 = z(3)
which discretizes the following continuous differential equation:
∂z ∂t = f (z) -z(4)
In Eq. 3, the map f defines the pushforward of a latent vector field V : R k → R k , tracing nonlinear trajectories in latent space (Fig. 1). A natural question is whether the ODE has well-defined and unique solutions.
By the Banach fixed-point theorem, this holds if and only if f is Lipschitz-continuous with Lipschitz constant C<1. Definition 1. A function f : Z → Z is Lipschitz-continuous if there exists a constant C s.t. for every pair of points z 1 , z 2 :
d(f (z 1 ), f (z 2 )) Z ≤ C d(z 1 , z 2 ) Z .(5)
When C < 1, f is called contractive, for a given metric d on Z.
For any contractive map, Eq. 3 admits fixed-point solutions, i.e., repeatedly applying the map f will converge to a unique solution z * satisfying z * = f (z * ). The fixed points z * can act as attractors, capturing and summarizing the system's long-term dynamics.
Definition 2. A fixed point z = f (z) is an attractor of a differentiable map f if all eigenvalues of the Jacobian J of f at z are strictly less than one in absolute value.
When f is nonlinear, the ODE in Eq. 3 can have multiple solutions depending on the initial conditions z 0 . The previous definition allows for a definition of Lipschitz continuity, which is inherently local: Definition 3. Let f : Z → Z be differentiable and C-Lipschitz continuous. Then the Lipschitz constant C is given by: C = sup z∈Z ∥J f (z)∥ σ , where J f (z) is the Jacobian of f evaluated at z and ∥ • ∥ σ is the spectral norm.
The set of initial conditions z 0 leading to the same attractor z * , is denoted as basin of attraction.
Why are neural mappings contractive in practice? We argue that mappings learned by neural AEs and their variants tend to be locally contractive, i.e.,their Jacobians have small eigenvalues near training examples. This behavior emerges naturally from several explicit and implicit inductive biases present in modern training pipelines. Below, we outline the main factors promoting contractive behavior:
• Initialization bias. Standard initialization schemes (LeCun et al., 2002;Glorot and Bengio, 2010;He et al., 2015) are designed to preserve activation variance at the start of training to avoid vanishing or exploding gradients. Theoretical arguments in support assume i.i.d. inputs and weights, and are often tied to specific activation functions (e.g., ReLU in (He et al., 2015)). However, real-world training data is typically correlated, and architectural features like residual connections (He et al., 2016) break weight independence. As a result, networks often exhibit a bias toward mappings that are globally expansive or contractive, empirically skewed toward the latter (Poole et al., 2016). We illustrate this empirically in Figure 6 (Appendix), showing that various vision backbones exhibit contractive behavior at initialization. Figure 3a (left) shows a 2D latent vector field which is globally contractive at initialization towards a single attractor at the origin. This contractive behavior also holds in higher dimensions (see the number of attractors at epoch 0 in Figure 3c).
• Explicit regularization. Common regularization methods like weight decay (D'Angelo et al., 2024) encourage contraction by penalizing the norm of model parameters. In the linear case, minimizing parameter norms directly reduces the Jacobian's spectral norm, making the map contractive. While this link is less direct for nonlinear models, the effect due to weight decay persists in practice.
Regularization is often integrated into optimizers used in large-scale models (Loshchilov, 2017). Additional architectural constraints, such as small bottleneck dimensions, also limit the rank of J F (z). Soft constraints include KL divergence in VAEs or sparsity penalties in SAEs (see Table 3 in the Appendix for a complete overview).
• Implicit regularization. Data augmentations introduce local perturbations around training examples, effectively defining a neighborhood structure in input space. For instance, Gaussian noise in DAEs or masking in MAEs perturbs inputs along specific directions. These augmentations implicitly regularize the Jacobian J f (x) by penalizing sensitivity to those perturbations. Unlike parametric regularization, this effect is inherently local and nonparametric.
this section cite: ['b32', 'b17', 'b18', 'b18', 'b19', 'b43', 'b9', 'b35']

Section: THEORETICAL REMARKS
In this section, we study the behavior of the latent vector field, focusing on its trajectories (Section 3.1) as well as its fixed points and attractors (Section 3.2), when they exist. Formal statements and proofs of all theorems and propositions are provided in Appendix A.
this section cite: []

Section: TRAJECTORIES OF THE LATENT VECTOR FIELD
Prior work (Miyasawa et al., 1961;Robbins, 1992;Alain and Bengio, 2014) has shown that, under ideal conditions, the residual of a denoising AE trained with noise variance σ 2 approximates the score function, i.e., the gradient of the log-density of the data, as σ → 0. Interestingly, Alain and Bengio (2014) shows a connection between denoising and contractive AEs, where the Jacobian norm of the input-output mapping is explicitly penalized.
We build on this by observing a more general phenomenon: when an autoencoding map is locally contractive relative to a chosen neighborhood structure (e.g., Gaussian noise, masking) and sufficiently approximates the input distribution p(x), the induced latent vector field pushes points in the direction of the score of the corresponding prior in latent space.
Informally, this implies that the vector field acts to nonlinearly project samples toward regions of high probability on the data manifold.
Theorem 1 (informal, proof in Appendix A.1). Let F be a trained autoencoder and let q(z) = p(x)q(z|x)dx be the marginal distribution induced in latent space. Assume q(z) is smooth and that there exists an open neighborhood Ω ⊇ supp q and a constant L < 1 such that sup z∈Ω J f (z) σ ≤ L. Then, latent dynamics f (z) -z in Ω is locally proportional to the score function ∇ z log q(z).
This result establishes a general link between the vector field's trajectories and the score function, under the assumption of local contractivity. In practice, neural networks often strike a balance between the reconstruction loss (Eq. 1) and regularization on the Jacobian, which contributes to the emergence of attractors in the latent space.
An important implication of Theorem 1 is that integrating the vector field f (z)-z effectively estimates the log-density, i.e., z ∇ log q(z)dz = log q(z) + C in unbounded domains. If the Jacobian J f is symmetric (e.g., in AEs with tied weights (Alain and Olivier, 2013)), then the latent vector field is conservative and corresponds to the gradient of a potential V f = ∇E. In this case, the AE defines an energy-based model with q(z) = e -E(z) (Zhai et al., 2016;Song and Kingma, 2021). In the general (non-conservative) case, we show empirically in Section 4.2 that the trajectories still reflect the learned prior distribution.
However, attractors are generally not accessible solving the fixed-point equation with first order methods, unless initialized very close to them. The following result formalizes this: Proposition 3.1 (informal, proof in Appendix A.2). Iterations of the map f in Eq. 3 correspond to gradient descent on a potential L(z) (i.e., f ≈ z -αL(z)) if and only if f is an isometry (its eigenvalues are near 1), or the dynamics occur near attractors, where J f vanishes.
This underscores the role of higher-order dynamics in the vector field: repeated applications of f trace complex, nonlinear trajectories that can escape spurious or unstable fixed points.
this section cite: ['b37', 'b49', 'b1', 'b1', 'b0', 'b64', 'b53']

Section: CHARACTERIZING THE ATTRACTORS OF THE LATENT VECTOR FIELD
In this section, we aim to characterize what the fixed point solutions of Eq. 3 represent. In Appendix A.4 we consider the two cases of linear and homogeneous networks as examples, to build intuition on the characterization of attractors and latent vector field.
Between memorization and generalization. Our goal is to characterize the properties of the latent vector field and the learned attractors in the general setting. We argue that different models can fit training points reaching similar low loss, but interpolate differently outside the training support, depending on the regularization strength and the amount of overparametrization.
In this context, a key question arises: what information do the attractors z * encode? We argue that neural models lie in a spectrum between memorization and generalization, depending on the strength of the regularization term in Eq. 1. We hypothesize that attractors fully characterize where a model falls on this spectrum. To do so, we first connect attractors to the notion of generalization via the following proposition:
Proposition 3.2 (Informal, proof in Appendix A.3). Let Z * be a dictionary of attractors of f = E •D in a neighborood Ω of the latent space, and let Π(z) denote the projection onto the nearest attractor to a latent code z = E(x). If D is L D -Lipschitz on Ω, then for any test point x:
∥x -F (x)∥ 2 2 ≤ ∥x -D(Π(E(x)))∥ 2 2 error to prototype + L 2 D ∥E(x) -Π(E(x)))∥ 2 2 coverage error .
In short, attractors define a dictionary for the data: when they coincide with training points, prototype error on these points vanishes but coverage is narrow (memorization regime), whereas generalization requires attractors that both cover the latent space and serve as good prototypes for unseen data.
We consider attractors as representations summarizing the information stored in the weights of the network. This interpretation is in line with the convergence of paths in the vector field to modes of the learned distribution of Theorem 1, and can be easily seen in the case of memorization.
Extreme overparametrization case. When the capacity of a network exceeds the number of training examples by far, AEs enter an overfitting regime leading to data memorization (Zhang et al., 2019;Radhakrishnan et al., 2020;Jiang and Pehlevan, 2020;Kadkhodaie et al., 2023). The network in this case learns a constant function or an approximation thereof, which can be retrieved through the iterations of Eq. 3.
this section cite: ['b65', 'b46', 'b25', 'b27']

Section: THE ROLE OF REGULARIZATION
We empirically demonstrate that the balance between memorization and generalization is governed by the strength of regularization. In Figure 2, we show how increasing regularization on the Jacobian J f (z * ) drives the model from a memorization regime, where many training examples are stored as attractors, toward generalization. We remark that this memorization arises in an over-regularized regime, as opposed to the overfitting regime of extremely overparametrized networks.
Setting. We trained 30 convolutional AEs on the CIFAR, MNIST, and FashionMNIST datasets, varying the bottleneck dimension k from 2 to 512. This acts as a hard regularizer on J f (z * ), by constraining its rank to be ≤ k. We compute attractors Z * from elements of the training set X train by iterating f till convergence. We measure the degree of memorization by defining a memorization coefficient, which is given by the cosine similarity of each decoded attractor x * = D(z * ) to its closest point in the training set, mem(z * ) = min x∈Xtrain cos(D(z * ), x) and we report the mean and standard deviation over attractors. We measure generalization by simply reporting the error on the test set. In Figure 13 in the Appendix, we also report the rank explaining 90% of the variance of the matrix of decoded attractors X * , showing that attractors corresponding to good generalization models span more directions in the input space. For additional information on the model, hyperparameters, and settings, we refer to Appendix D. Attractors computed from training and from gaussian noise converge during training (green), while the separability of the trajectories (measured as FPR95, the lower the better) increase (purple).
this section cite: []

Section: Analysis of results.
In Figure 2 on the left, we observe that the more regularized networks (k from 2 to 16) tend to memorize data, trading off generalization performance (middle plot). We remark that this kind of memorization is not due to overfitting, as was shown in (Kadkhodaie et al., 2023) for diffusion models, but it happens in the underfit regime, due to the strong regularization constraint on J f (x). We show in Figure 14 in the Appendix results in the overfitting regime, by training the models with different sample sizes and observing a similar pattern.
Takeaway. Attractors capture the interplay between generalization and memorization of neural models, which corresponds to the trade-off between the reconstruction performance and regularization term of the AE model.
this section cite: ['b27']

Section: MEMORIZATION AND GENERALIZATION ACROSS TRAINING
In the experiment in Figure 3 we show that a similar transition from a memorization regime to generalization occurs across training.
Setting. We monitor the latent vector field and attractors statistics across the training dynamics of a convolutional AE trained on MNIST with bottleneck dimension k = 128 (bottom row plots in the Figure ). To show qualitatively the evolution of a latent vector field, we also plot it across training epochs for AEs with bottleneck dimension k = 2 in Figure 3a.
this section cite: []

Section: Analysis of results.
In Figure 3b we show the transition from memorization, occurring at the first epochs of training, to generalization, by plotting the memorization coefficient and the test error across training, observing a trade-off between the two. In the center plot, we plot the fraction of distinct attractors, computed from 5000 random elements respectively from the P train , P test and N (0, I), where we consider two attractors equal if cos(z * 1 , z * 2 ) > 0.99. Initially, the model converges to a single attractor (as also seen in the 2D example in Figure 3a). Over time, the number of attractors increases, stabilizing for training and test data in tandem with the test loss, while attractors from noise inputs converge more slowly. The right plot shows the Chamfer symmetric similarity between attractors from the training and noise distributions, which increases over training as the two distributions of attractors match. Importantly, while attractors derived from noise and training data become increasingly similar over the course of training, the trajectories leading to them differ significantly. To quantify this, we compute the False Positive Rate at 95% True Positive Rate (FPR95) scores for distinguishing trajectories originating from noise versus those from training data: FPR95 measures what percentage of noise latent trajectories are falsely classified as training trajectories, setting a threshold such that 95% of training trajectories are classified correctly. We define the trajectory score as score(z) = Takeaway. As the latent vector field evolves during training, the model transitions from memorization to generalization, forming similar attractors from different input distributions, while retaining information of the source distribution in the latent trajectories.
this section cite: []

Section: EXPERIMENTS ON VISION FOUNDATION MODELS
In this section we demonstrate the existence and applicability of the latent dynamics on vision foundation models, including the AE backbones of the Stable Diffusion model (Rombach et al., 2022) and vision transformer masked AEs (He et al., 2022), showing; (i) how information stored in the weights of a pretrained model can be recovered by computing the attractors from noise (ii) how trajectories in the latent dynamics are informative to characterize the learned distribution and detect distribution shifts.
this section cite: ['b50', 'b20']

Section: DATA FREE WEIGHT PROBING

this section cite: []

Section: Setting.
In this experiment, we investigate how much information stored in a neural network's weights can be recovered purely from attractors computed on Gaussian noise, without access to any input data. We focus on AE component of Stable Diffusion (Rombach et al., 2022) pretrained on the large scale Laion2B dataset (Schuhmann et al., 2022).
We sample Z n ∼ N (0, I) and compute the corresponding attractors as solutions to f (Z * n ) = Z * n . We generate N = 4096 such points, matching the latent dimensionality k of the model. We evaluate the reconstruction performance on 6 diverse datasets Laion2B (Schuhmann et al., 2022), Imagenet (Deng et al., 2009), EuroSAT (Helber et al., 2019), CIFAR100 (Krizhevsky et al., 2009), PatchCamelyon (Litjens et al., 2018), and Places365 (Zhou and Paffenroth, 2017). These datasets span general, medical, and satellite image domains.
For each dataset, we randomly sample 500 test examples and reconstruct them in latent space using Orthogonal Matching Pursuit (OMP) (Mallat and Zhang, 1993), with varying sparsity levels (i.e.,
SUN397 Places365 Texture iNaturalist Method FPR ↓ AUROC ↑ FPR ↓ AUROC ↑ FPR ↓ AUROC ↑ FPR ↓ AUROC ↑ d(Attractors) 29.60 91.20 29.95 90.99 25.85 92.63 29.85 91.29 KNN 100.00 42.59 100.00 32.36 34.50 89.41 86.35 68.60 (a) FPR and AUROC scores (b) KNN baseline (c) Latent trajectories number of atoms used). Reconstructions are decoded via D, and performance is compared against a baseline reconstruction using the initial orthogonal samples Z * 0 , which, by design, fully span the latent space Z.
this section cite: ['b50', 'b51', 'b51', 'b11', 'b21', 'b30', 'b34', 'b68', 'b36']

Section: Analysis of results.
In Figure 4, we plot the error as a function of the number of elements (atoms) chosen to reconstruct the test samples using OMP. We compare by building a random orthogonal basis sampled in the latent space. On all considered datasets, noise attractors recover test samples with lower reconstruction error, representing a better dictionary of signals. In Figure 4a, we show qualitative reconstruction using only 5 % of the atoms. In Figures 7, 8, 9 in the Appendix, we include additional qualitative evidence of this phenomenon, visualizing reconstructions of random samples of the datasets as a function of the number of atoms used. In Appendix B, we report additional results on variants of the AE of different sizes.
Takeaway. Attractors of foundation models computed from noise can serve as a dictionary of signals to represent diverse datasets, demonstrating that it is possible to probe the information stored in the weights of foundation models in a black box way, without requiring any input data.
this section cite: []

Section: LATENT TRAJECTORIES CHARACTERIZE THE LEARNED DISTRIBUTION
In the following experiment, we test the hypothesis that trajectories are informative on the distribution learned by the model, testing how well Theorem 1 holds in practice. Our goal is to evaluate whether these trajectories can be used to detect distribution shifts in the input data. To classify a sample as out-of-distribution (OOD), we focus on two key questions: (i) does the sample trajectory converge to one of the attractors of the training data, i.e., does it share the same basin of attraction? (ii) If so, how fast does it converge?
Notably, an OOD sample may still lie within a shared attractor basin. To capture both scenarios, we track the distance from each point along a test sample's trajectory to the nearest attractor from the training set. In the former case, we expect that an OOD sample converges faster, while in the latter case, the distance term to the attractors will dominate the score.
this section cite: []

Section: Setting.
We test this hypothesis using the ViT-MAE (He et al., 2022) architecture pretrained on ImageNet. We sample 2000 training images and compute their attractors, stopping when convergence reaches a tolerance of 10 -5 or a maximum number of iterations. We test on samples from SUN397 (Xiao et al., 2016), Places365 (Zhou and Paffenroth, 2017), Texture (Cimpoi et al., 2014), and iNaturalist (Van Horn et al., 2018), standard benchmarks for OOD detection (Yang et al., 2024). We report two metrics: FPR95, and Area Under the Receiver Operating Characteristic Curve (AUROC). FPR95 here measures what percentage of OOD data we falsely classify as ID, setting a threshold s.t. 95% of ID data is correctly classified.
this section cite: ['b20', 'b61', 'b68', 'b8', 'b57', 'b63']

Section: Analysis of results.
In Figure 5 we report OOD detection performance by using the following score function: for a test sample z test we compute its trajectory π ztest = [z 0 , ..., z * N ] towards attractors and compute the distance of π Ztest to the set of training attractors Z * train , i.e. score(z) = d(π ztest , Z * train ). As a distance function, we employ Euclidean distance, and we aggre-gate the score by computing the mean distance over the trajectory score(z) = 1 N i d(z i , Z * train ). We compare with a K-Nearest neighbor baseline (adapted from Sun et al. (2022)), where the score for a test sample is obtained by taking the mean distance over the K-NN on the training dataset, where K = 2000 in the experiments. The score proposed demonstrates how informative the latent vector field is on the training distribution. In Figure 5 on the right, we show histograms of scores for the distance to attractors and the KNN, showing again that the former method is able to tell apart in-distribution and out-of-distribution data correctly.
Takeaway. Trajectories in the latent vector field characterize the source distribution and are informative to detect distribution shifts.
this section cite: ['b55']

Section: RELATED WORK
Memorization and generalization in neural networks (NNs). NNs exhibit a rich spectrum of behaviors between memorization and generalization, depending on model capacity, regularization, and data availability (Arpit et al., 2017;Zhang et al., 2021;Power et al., 2022). In the case of extreme overparametrization, namely networks trained on few data points, it has been shown experimentally in (Radhakrishnan et al., 2020;Zhang et al., 2019) and theoretically for sigmoid shallow AEs in (Jiang and Pehlevan, 2020) that AEs can memorize examples and implement associative memory mechanisms. (Alain and Bengio, 2014;Vincent, 2011), A similar phenomenon has been observed in diffusion models in (Somepalli et al., 2023;Dar et al., 2023) and analyzed in (Kadkhodaie et al., 2023) Similarly, non gradient-based approaches such as Hopfield networks and their modern variants (Hopfield, 1982;Ramsauer et al., 2020) extend classical attractor dynamics to neural systems that interpolate between memory-based and generalizing regimes. In our work, we show that AEs fall in general in the spectrum between memorization and generalization, depending on inductive biases that enforce contraction.
Contractive neural models. Different approaches have been proposed to regularize NNs in order to make them smoother and more robust to input perturbations and less prone to overfitting. Many of these regularization techniques either implicitly or explicitly promote contractive mappings in AEs: for example, sparse AEs (Ng et al., 2011;Gao et al., 2024), their denoising variant (Vincent et al., 2008), and contractive AEs (Rifai et al., 2011;Alain and Bengio, 2014) losses enforce learned maps to be contractive through the loss. Regularization strategies such as weight decay (Krogh and Hertz, 1991) favor as well contractive solutions for neural models, and they are incorporated in standardized optimizers such as AdamW (Loshchilov, 2017). All these approaches enforce either directly or indirectly the existence of fixed points and attractors in the proposed latent vector field representation.
Neural networks as dynamical systems. Distinct lines of work have interpreted neural networks as dynamical systems. Neural ODEs (Chen et al., 2018) view depth as continuous time and model hidden states via differential equations, while deep equilibrium models (Bai et al., 2019) characterize predictions as fixed points of implicit dynamics. Closer to our work, Radhakrishnan et al. (2020) interpret overparameterized autoencoders as dynamical systems acting in the input space, implementing associative memory. In contrast, we show that any autoencoder induces a latent vector field, and we link its properties to generalization, memorization, and the characterization of the learned distribution.
Nonlinear operators spectral analysis. In the context of image processing and 3D graphics, previous work has inspected generalization of spectral decompositions to nonlinear operators (Bungert et al., 2021;Gilboa et al., 2016;Fumero et al., 2020), focusing on one homogeneous operator. In our work, fixed points of Eq. 3 can be interpreted as the decomposition of the NN into a dictionary of signals.
this section cite: ['b3', 'b66', 'b44', 'b46', 'b65', 'b25', 'b1', 'b58', 'b52', 'b10', 'b27', 'b22', 'b47', 'b40', 'b15', 'b59', 'b48', 'b1', 'b31', 'b35', 'b4', 'b46', 'b6', 'b16', 'b12']

Section: CONCLUSIONS AND DISCUSSION
In this work we proposed to represent neural AEs as vectors fields, implicitly defined by iterating the autoencoding map in the latent space. We showed that (i) attractors in the latent vectors field exists in practice due to inductive biases in the training regime which enforce local contractions; (ii) they retain key properties of the model and the data, linking to memorization and generalization regimes of the model; (iii) knowledge stored in the weights can be retrieved without access to input data in vision foundation models; (iv) paths in the vector field inform on the learned distribution and its shifts.
this section cite: []

Section: LIMITATIONS AND FUTURE WORKS
Generalizing to arbitrary models. Eq. 3 cannot be directly generalized to model trained with discriminative objectives such as a deep classifiers, or self supervised models, as the network is not invertible. However we note that (i) our theory holds in general for any self map which is locally contractive (ii) the vector field is still defined in the output space, and can be simply obtained by computing the residual F (x)-y and in the neighborhood of an attractor, the relation in proposition 3.1 can still hold for different objectives. One idea is to train a decoder on top of the frozen encoder-only model. We give preliminary evidence of the existence of latent vector fields in self-supervised models and in next token predictors (e.g. LLMs) in Appendix B.8, showing that extending our framework to arbitrary models holds promise. An alternative intriguing idea is the one to train a surrogate AE model in the latent space of the model of interest, which would be agnostic from the pretraining objective. Sparse AEs for mechanistic interpretability of large language models (LLMs) (Gao et al., 2024) fit in this category. Analyzing the associated latent vector field can shed light on features learned by SAEs and biases stored in their weights.
Learning dynamics. Characterizing how attractors forms during training, under which conditions noise attractors converge to the training attractors, holds promise to use the proposed representation to study the learning dynamics of neural models to inspect finetuning of AE modules, such as low-rank adapters (Hu et al., 2022) and double descent (Nakkiran et al., 2021).
Alignment of latent vector fields. Finally, following recent findings in representation alignment (Moschella et al., 2022;Fumero et al., 2024;Huh et al., 2024) inspecting how latent vector fields of networks trained are related is an open question for future works, to possibly use this representations to compare different neural models.
this section cite: ['b15', 'b23', 'b39', 'b38', 'b13', 'b24']

Section: ETHICS STATEMENT
This work is primarily methodological and theoretical, focusing on the dynamics of autoencoder networks. We do not foresee any direct negative societal impacts. All datasets used are publicly available benchmark datasets (e.g., MNIST, CIFAR, Imagenet, Laion2B) or openly released via HuggingFace (Lhoest et al., 2021). No sensitive, private, or personally identifiable information is involved. Our experiments are designed for scientific analysis only, and no human subjects or protected groups are included. While the methodology in Section 4.1 provides tools to probe information about trained data in autoencoders foundation models, it does not directly improve fairness, robustness, or safety. In principle, insights into memorization could be misused to develop algorithms to recover information from pretrained models in unintended ways. However, our study is limited to publicly available models and datasets, and we emphasize that the proposed techniques should be applied responsibly in research contexts. We hope our findings contribute to a better understanding of neural representations and their generalization/memorization properties, which may inform the design of more transparent and robust models. We provide details and hyperparameters for each experiment for reproducibility and formal statements and proof of our theoretical results in the Appendix.
this section cite: ['b33']

Section: REPRODUCIBILITY STATEMENT
All formal statements and proofs of theorems are reported in Appendix A. We implement all our experiments using PyTorch (Paszke et al., 2019) and HuggingFace libraries Lhoest et al. (2021); Wolf et al. (2020). We experiment solely with openly available models and datasets available on HuggingFace Hub. We will open-source our codebase for all the experiments upon acceptance. Unless otherwise noted in the experiment-specific sections, all hyperparameters are listed in Section D.
this section cite: ['b42', 'b33', 'b60']

Section: A PROOFS AND DERIVATION

this section cite: []

Section: A.1 THEOREM 1
We report here a detailed formulation of Theorem 1 alongside its proof. Assumption A.1 (Latent marginal). Let p data (x) be the data distribution and q ϕ (z | x) an encoder mapping inputs x ∈ R n to latent codes z ∈ R d . The latent marginal density is defined as:
q(z) = p data (x) q ϕ (z | x) dx.
We assume that q(z) is continuously differentiable, and that its gradient ∇ log q(z) and Hessian ∇ 2 log q(z) are well-defined and continuous on an open domain containing Ω ⊆ R d . Assumption A.2 (Fixed-point manifold). Let E : R n → R d be an encoder and D : R d → R n a decoder, both continuously differentiable. Define the composite map
f (z) = E(D(z)) ∈ R d .
We define the fixed-point manifold of f as
M = z ∈ R d : f (z) = z .
this section cite: []

Section: Assumption A.3 (Local contraction).
There exists an open, convex set Ω ⊆ R d , with M ⊆ Ω, and a constant 0 < L < 1, such that:
sup z∈Ω ∥J f (z)∥ σ ≤ L,
where J f (z) ∈ R d×d denotes the Jacobian of f at z, and ∥ • ∥ σ denotes the spectral norm (i.e., largest singular value). Therefore, f is a contraction mapping on Ω. Assumption A.4 (Training optimality with Jacobian regularization). Let f (z) = E(D(z)) be as above, with E, D trained to minimize the objective:
E x∼p data ∥D(E(x)) -x∥ 2 + λ∥J f (E(x))∥ 2 F for λ > 0.
For example, assume the encoder is deterministic, i.e., q ϕ (z|x) = δ(z -E(x)), so the latent marginal satisfies:
q(z) = p data (x) δ(z -E(x)) dx.
Then q(z) is supported and concentrated on the fixed-point manifold M, and f is locally contractive around M. Lemma A.5 (Directional ascent of the residual field). Under the above assumptions, for every z ∈ Ω \ M, define the residual field:
v(z) := f (z) -z.
Then the directional derivative of log q in direction v(z) is strictly positive:
⟨∇ log q(z), v(z)⟩ > 0.
Proof. Let z ∈ Ω \ M be arbitrary. Then f (z) ̸ = z, so v(z) = f (z) -z ̸ = 0. By contractiveness and training optimality, f (z) is closer to the fixed-point set M than z is. Since q(z) is concentrated on M, we have:
q(f (z)) > q(z) ⇒ log q(f (z)) > log q(z).
We now Taylor-expand log q at z in direction v(z): z), for some ξ between z and f (z). Since f is contractive, ∥v(z)∥ is small and R = o(∥v(z)∥). Therefore:
log q(f (z)) = log q(z) + ⟨∇ log q(z), v(z)⟩ + R, where R = 1 2 v ⊤ (z)∇ 2 log q(ξ)v(
⟨∇ log q(z), v(z)⟩ > -R ⇒ ⟨∇ log q(z), v(z)⟩ > 0.
1. The Jacobian J f (z) ≈ Id (i.e., f is locally an isometry).
2. z is near an attractor point z * such that f (z * ) = z * and J f (z * ) ≈ 0.
Proof. We begin by expanding the loss:
L(z) = ∥f (z) -z∥ 2 = ⟨f (z) -z, f (z) -z⟩. Let v(z) := f (z) -z. Then, L(z) = ∥v(z)∥ 2 .
Now compute the gradient of L:
∇ z L(z) = ∇ z v(z) ⊤ v(z) = 2J v (z) ⊤ v(z), where J v (z) = J f (z) -Id is the Jacobian of the residual field. So we obtain: ∇ z L(z) = 2(J f (z) -Id) ⊤ v(z).
Therefore, if we ask whether v(z) ∝ -∇ z L(z), we need:
v(z) ∝ -(J f (z) -Id) ⊤ v(z), which simplifies to: [(J f (z) -Id) ⊤ + αId]v(z) = 0 for some α > 0.
This holds approximately in the two special cases:
• Isometry: If J f (z) ≈ Id, then ∇L(z) ≈ 0, so v(z) is nearly stationary-i.e., we are at or near a local minimum.
• Attractor: If z ≈ z * with f (z * ) = z * and J f (z * ) ≈ 0, then: ∇L(z * ) ≈ -2v(z * ) = 0, and thus z * is a fixed point and local minimum of the loss.
In these two cases cases, the dynamics of z t+1 = f (z t ) follow the direction of steepest descent of L, up to scaling. Higher order terms dominate influence the dynamics in the general case.
this section cite: []

Section: A.3 ATTRACTORS CONNECTS TO GENERALIZATION ERROR
We report here a detailed formulation of Proposition 3.2 alongside its proof. Proposition A.8 (Attractors as prototypes for generalization). Let Z ⋆ ⊂ Ω ⊂ R d be a (finite) dictionary of attractors of f = E • D in a neighborhood Ω of latent space, and let Π : Ω → Z ⋆ denote a (measurable) nearest-attractor map Π(z) ∈ arg min u∈Z ⋆ ∥z -u∥ 2 (with any fixed tiebreaking rule). If the decoder D is L D -Lipschitz on Ω, then for any test point x with z = E(x) ∈ Ω:
x -F (x) 2 2 ≤ x -D(Π(z)) 2 2 prototype error + L 2 D z -Π(z) 2 2 coverage error .
Moreover, if Z ⋆ is an ε-cover of supp q (the latent marginal), then for all x with E(x) ∈ supp q,
x -F (x) 2 2 ≤ x -D(Π(E(x))) 2 2 + L 2 D ε 2
, and the same bound holds in expectation over x ∼ p test whenever E(x) ∈ supp q almost surely.
We make the following mild assumptions:
(A1) Z * ⊂ Ω is a (finite) set of attracting fixed points of f = E • D contained in an open neighborhood Ω ⊂ R d . (A2) D is L D -Lipschitz on Ω, i.e., ∥D(z) -D(u)∥ ≤ L D ∥z -u∥ 2 for all z, u ∈ Ω. (A3) Π : Ω → Z ⋆ is any measurable selection of nearest points, e.g. Π(z) ∈ arg min u∈Z ⋆ ∥z - u∥ 2 .
Proof of Proposition A.8. Let x be any test point with z = E(x) ∈ Ω. Add and subtract D(Π(z)):
x -D(E(x)) = x -D(Π(z)) prototype error + D(Π(z)) -D(E(x)) decoder distortion .
Taking norms and using Cauchy-Schwarz inequality gives:
∥x -D(E(x))∥ 2 2 ≤ ∥x -D(Π(z))∥ 2 2 + ∥D(Π(z)) -D(E(x))∥ 2 2 . By (A2), ∥D(Π(z)) -D(E(x))∥ ≤ L D ∥Π(z) -z∥ 2 , which yields: ∥x -F (x)∥ 2 2 ≤ ∥x -D(Π(z))∥ 2 2 + L 2 D ∥z -Π(z)∥ 2 2 .
This proves the first inequality. If Z ⋆ is an ε-cover of supp q, then for any z ∈ supp q we have
∥z -Π(z)∥ ≤ ε, hence ∥x -F (x)∥ 2 2 ≤ ∥x -D(Π(E(x)))∥ 2 2 + L 2 D ε 2 , for all x with E(x) ∈ supp q.
Taking expectations (when E(x) ∈ supp q almost surely) gives the stated bound. □ Remarks. (i) The bound decomposes test error into a prototype term, ∥x -D(Π(E(x)))∥ 2 2 , and a coverage term, L 2 D ∥z -Π(z)∥ 2 2 . (ii) Under the assumptions of Theorem 1, Z * are attractors with basins; then Π aligns with the destination of latent dynamics, tying the algebraic bound to the memorization-generalization picture.
this section cite: []

Section: A.4 ATTRACTORS IN SIMPLE NETWORKS
In this section we characterize attractors in linear and homogeneous networks, showing that in this simple setting is possible to prove converge speed to attractors of the iteration in Eq 3.
this section cite: []

Section: Linear maps.
In the linear case, the encoding map E and decoding D are parametrized by matrices W 1 ∈ R N ×k and W 2 ∈ R k×N and the only fixed point of the map corresponds to the origin if the network is bias free or to a shift of it. The rate of convergence of the iteration in Eq 3 is established by the spectrum of W T 2 W 1 , and the iteration is equivalent to shrinking the input in the direction of the eigenvectors with associated eigenvalue λ < 1. In case the eigenvectors of the trained AEs are aligned with the optimal solution given by the Principal Component Analysis (PCA) decomposition of the data X = ΦΛΦ * , then the latent vector field vanished as the mapping is isometric and no contraction occurs.
Homogeneous maps. In the case of homogeneous neural networks, the network satisfies F (cx) = c α F (x) with c ∈ R for some α. For example, this holds for ReLU networks without biases with α = 1, which learn a piecewise linear mapping. The input-output mapping can be rewritten as:
F (x) = J F (x)x(6)
A similar observation was made in for denoising networks in (Kadkhodaie et al., 2023), we remark here its generality.
This equality implies that we can rewrite Equation 3 as :
z t+1 = J f (z t )z t = i λ i ϕ i z t (7
) Where i λ i ϕ i is the eigendecomposition of J f (z).
Since in the proximity of an attractor maxλ(J F (z * ) <= 1, the iterations shrink directions corresponding to the eigenvectors of the Jacobian by the corresponding eigenvalue. This allows us to derive the following result on the speed of convergence. Proposition A.9 (informal, proof in Appendix A.5). The error e t = ∥z t -z * ∥ converge exponentially to an ϵ depending on the spectral norm ∥J f (z * )∥ σ , according to the formula
log( ϵ ∥e 0 ∥ ) log(∥J f (z * )∥σ)
that provides an estimate for the number of iterations T to converge to the attractor.
In Figure 11 in the Appendix, we measure how well the convergence formula predicts the measured number of iterations to converge to an attractor, on a fully non linear AE with biases, where the assumptions of the current section are less likely to hold. The error in the estimate will be higher, when the initial condition z 0 is far from the attractor, as higher order terms dominate the dynamics, and the first order Taylor approximation accumulates more error.
this section cite: ['b27']

Section: A.5 PROOF OF PROPOSITION A.9
We report here a detailed formulation of Proposition A.9 alongside its proof. Proposition A.10. Let f : R n → R n be a differentiable function with a fixed point z * ∈ R n such that f (z * ) = z * . Assume the Jacobian J f (z * ) has spectral norm ∥J f (z * )∥ σ = ρ < 1. Then, for initial point z 0 sufficiently close to z * , the sequence {z t } defined by the iteration z t+1 = f (z t ) satisfies:
∥z t -z * ∥ ≤ ∥e 0 ∥ • ρ t ,
and the number of iterations T needed to reach an error ∥z T -z * ∥ ≤ ϵ is bounded by:
T ≥ log(ϵ/∥e 0 ∥) log(ρ) .
Proof. Let e t = z t -z * denote the error at iteration t. We are given that f is differentiable around z * , and that f (z * ) = z * . Applying the first-order Taylor expansion of f around z * , we obtain:
f (z t ) = f (z * ) + J f (z * )(z t -z * ) + R(z t ),
where R(z t ) is the Taylor remainder satisfying ∥R(z t )∥ = o(∥z t -z * ∥). Since f (z * ) = z * , this becomes:
z t+1 = f (z t ) = z * + J f (z * )(z t -z * ) + R(z t ).
Thus, the error evolves as:
e t+1 = z t+1 -z * = J f (z * )e t + R(z t ).
For z t sufficiently close to z * , we can neglect the higher-order term R(z t ) in comparison to the leading linear term. Hence, the error evolves approximately as:
∥e t+1 ∥ ≤ ∥J f (z * )∥ σ • ∥e t ∥ + o(∥e t ∥).
By continuity of f , there exists a neighborhood U of z * where ∥J f (z)∥ σ ≤ ρ + δ < 1 for some δ > 0. Choosing z 0 ∈ U and letting R(z t ) = 0 (first-order approximation), the error satisfies:
∥e t ∥ ≤ ∥e 0 ∥ • ρ t .
We now ask: for which T does ∥e T ∥ ≤ ϵ? We solve:
∥e 0 ∥ • ρ T ≤ ϵ.
Dividing both sides by ∥e 0 ∥ and taking logarithms yields:
ρ T ≤ ϵ ∥e 0 ∥ ⇒ T • log(ρ) ≤ log ϵ ∥e 0 ∥ .
Since log(ρ) < 0, we reverse the inequality when dividing:
T ≥ log(ϵ/∥e 0 ∥) log(ρ) .
This proves that the error converges exponentially and that the number of iterations required to reach error ϵ is lower bounded by the expression above.
this section cite: []

Section: B ADDITIONAL EXPERIMENTS

this section cite: []

Section: B.1 BIAS AT INITIALIZATION
As stated in the method section of the main paper, To empirically measure to what extent model are contractive at initialization we sample N = 1000 samples from N (0, I), and we map them trough 12 different vision backbones, initialized randomly across 10 seeds. We measure the ratio between the input variance σ in = 1 and output variance σ out and we report in Figure 6 . (a) 256 atoms (b) 1024 atoms (c) 2048 atoms Figure 7: Visualization of reconstructions from the data-free sample recovery experiment of Figure 4: Visualizing on Laion2B reconstructions of five random samples. First row: input samples; second row: reconstructions from orthogonal basis; third row reconstructions from attractors of noise. (a) 256 atoms (b) 1024 atoms (c) 2048 atoms Figure 8: Visualization of reconstructions from the data-free sample recovery experiment of Figure 4: Visualizing on Imagenet1k reconstructions of five random samples. First row: input samples; second row: reconstructions from orthogonal basis; third row reconstructions from attractors of noise. (a) 256 atoms (b) 1024 atoms (c) 2048 atoms We report additional qualitative results from the data free weight probing experiment in section 4.1, in Figures 7,8,9, observing that the reconstructions from attractors as a functions of the number of atoms used, are superior with respect to the orthogonal basis baseline.
this section cite: []

Section: B.3 ADDITIONAL RESULTS ON LATENT TRAJECTORIES
In Table 1 we extend results of the experiment in Fig. 5, adding two additional baselines as OOD detection score: Mahanabolis distance to the on in-distribution features and reconstruction error. In both cases statistics using the latent trajectories are more discriminative than using scores based on features.   B.4 ANALYSIS SPEED OF CONVERGENCE
In figure 11 we report an empirical validation of proposition A.9, for a 2 dimensional bottleneck convolution autoencoder trained on MNIST, by sampling points in the training set and measuring their convergence speed in terms of number of iteration. We remark that assumptions for the proposition don't hold for this model, as the farer an initial condition is from an attractor the more higher order terms will dominate in the dynamics, making the evaluation of the spectrum of the Jacobian at the attractor insufficient. We observe nevertheless, that the convergence speed bounds still correlates, providing a looser estimate for the number of iteration needed to converge towards an attractors with error ϵ.
Sparsity 0.0 0.1 0.2 0.3 0.4 MSE Laion2B Ortho Attractors 4096 2048 1024 512 256 128 64 32 Sparsity MSE EuroSAT Ortho Attractors 4096 2048 1024 512 256 128 64 32 Sparsity MSE Places365 Ortho Attractors 4096 2048 1024 512 256 128 64 32 Sparsity 0.0 0.1 0.2 0.3 0.4 MSE Imagenet-1k Ortho Attractors 4096 2048 1024 512 256 128 64 32 Sparsity MSE PatchCamelyon Ortho Attractors 4096 2048 1024 512 256 128 64 32 Sparsity MSE Cifar100 Ortho Attractors Figure 15: Data-free weight probing Stable diffusion XL AE : the results on the larger version of the table diffusion autoencoder confirm that attractors from noise form an informative dictionary of signals to reconstruct different datasets. To assess how the latent vector field behaves, we sample 500 samples from the Imagenet1k training dataset, and compute 300 iterations of Eq 3. We monitor (a) the norm of the residuals in the output space (i.e. reconstruction error with respect to previous iteration) (b) the norm of the residuals in the latent space (i.e. the elements vector field at current time step) (c) the average norm of the embeddings at every step. We show the results in Figure 16 observe that the reconstruction residuals and the latent space residuals both monotonically decrease at the beginning, with the former approaching 0, demonstrating a contraction behavior and then stabilize. This together with the observation that the norm of the embeddings is bounded and does not increase across iterations makes us conclude that the latent vector field exists, does not diverge, and is well behaved. The latent space residuals stabilizing around iteration 150 highlight that the eigenvalues of J f (z t ) are approaching 1, indicating either that the iteration has reached an attractor region (as opposed to a point) or that the the map is still contractive but with a constant very close to 1.
this section cite: []

Section: B.8.2 LATENT VECTORS FIELDS IN LLMS
We now consider the case of next token predictors models, in particular case light weight large language models. In this case we consider the input-output mapping in representation space, by iterating representation computed after the embedding layer to the final layer before the softmax, therefore f corresponds to the entire residual stream of the model.
As models we consider Qwen3-0.6 B Yang et al. (2025) and Smol-LM2 Allal et al. (2025). To test whether the latent vectors field are we sample 1000 sample from the Pile dataset Gao et al. (2020) and iterate Eq. 3 for 300 iteration monitoring residuals in the latent space (final layer) (a) and the norm of the final layer embeddings (b). In Figure 17 we plot the results, observing that both models show contractive behavior (a) ithout diverging, given the bounded norms (b).
Taken together these results give us preliminary evidence that latent vector fields exist and are well defined beyond the case of autoencoder model, but we stress that further analysis should be performed in order to study them, and analyze their attractor modes whenever they exists. Also the impact of different architectural elements, for example for LLMs the role of positional embeddings, (e.g. rotary position embeddings Su et al. (2024) as opposed to absolute ones) pose questions on the resulting dynamics. We believe that studying properties of the latent vector field induced by models beyond autoencoders hold promise to understand mechanistically the properties of the models, and we find promising to study recent phenomenon such as latent space reasoning Zhu et al. (2025). B.9 QUALITATIVE VISUALIZATION OF ATTRACTORS In this section we provide more visualizations from decoded attractors other than the ones in Figure 2 (c). In Figure 18 we report examples of decoded attractors from the latent space of the Stable Diffusion autoencoder. Attractors don't show semantically meaningful patterns but rather geometrical and texture patterns,resembling elements of a basis, similarly to the last rows of Figure 2 (c).
Input C × H × W image C × H × W Conv2D C → d, 3 × 3, stride 2, pad 1 d × H 2 × W 2 Conv2D d → 2d, 3 × 3, stride 2, pad 1 2d × H 4 × W 4 Conv2D 2d → 4d, 3 × 3, stride 2, pad 1 4d × H 8 × W 8 Conv2D 4d → 8d, 3 × 3, stride 2, pad 1 8d × H 16 × W 16 Flatten - 8d • H 16 • W 16 Bottleneck Linear layer or projection z (latent code) Decoder Unflatten z → 8d × H 16 × W 16 8d × H 16 × W 16 ConvTranspose2D 8d → 4d, 3 × 3, stride 2, pad 1, output pad 1 4d × H 8 × W 8 ConvTranspose2D 4d → 2d, 3 × 3, stride 2, pad 1, output pad 1 2d × H 4 × W 4 ConvTranspose2D 2d → d, 3 × 3, stride 2, pad 1, output pad 1 d × H 2 × W 2 ConvTranspose2D d → C, 3 × 3, stride 2, pad 1, output pad 1 C × H × W Output: Reconstructed Image
For the experiments in Figures 2, 3, 14 we considered convolutional autoencoders with architecture specified in Table 4. We train the models with Adam (Kingma, 2014) with learning rate 5e -4, linear step learning rate scheduler, and weight decay 1e -4 for 500 epochs. For the experiements in section 4.1, we use the pretrained autoencoder of (Rombachet al., 2022). We report in Table 2 the architectural details and we refer to the paper for training details. For the experiments in section 4.2, we use the pretrained autoencoder of (Heet al., 2022), considering the base model. We refer to the paper for training details. For computing attractors in experiments in Figures 2,3, 14, we compute the iterations z t+1 = f (z t ) until ∥f (z t+1 ) -f (z t )∥ 2 2 < 1e -6 or reaching t = 3000. Similarly in experiments in Sections 4.1, 4.2, we compute the iterations z t+1 = f (z t ) until ∥f (z t+1 )-f (z t )∥ 2 2 < 1e -5 or reaching t = 500. All experiments are performed on a GPU NVIDIA 3080TI in Python code, using the PyTorch library (Paszke et al., 2019).
this section cite: ['b54', 'b69', 'b42']

Section: 
Theorem A.6 (Convergence to latent-space modes). Let Assumptions 1-4 hold. Let z 0 ∈ Ω, and define the iterative sequence: z t+1 := f (z t ) = E(D(z t )) for all t ≥ 0.
Then:
1. The sequence {z t } converges exponentially fast to a unique fixed point z * ∈ M, satisfying f (z * ) = z * .
2. At the limit point z * , we have ∇ log q(z * ) = 0.
3. Moreover, z * is a local maximum of the density q: the Hessian satisfies
∇ 2 log q(z * ) ≺ 0,
meaning that the Hessian is negative definite.
Proof.
Step 1: Contraction and convergence. By Assumption 3, f : Ω → Ω is a contraction mapping with contraction constant L < 1. Since Ω is convex and hence complete under ∥ • ∥, Banach's fixed-point theorem applies. Therefore:
• There exists a unique fixed point z * ∈ Ω such that f (z * ) = z * .
• For any initial z 0 ∈ Ω, the iterates satisfy:
∥z t -z * ∥ ≤ L t ∥z 0 -z * ∥ → 0 as t → ∞.
Step 2: Stationarity of the limit. At the fixed point z * , we have:
v(z * ) = f (z * ) -z * = 0.
Assume for contradiction that ∇ log q(z * ) ̸ = 0. By continuity of ∇ log q, there exists a neighborhood U ∋ z * where ∇ log q(z) stays close to ∇ log q(z * ). Then for nearby z, Lemma A.5 implies:
⟨∇ log q(z), v(z)⟩ > 0.
But continuity of f and the residual v(z) → 0 implies the ascent direction vanishes at z * , which contradicts the assumption that ∇ log q(z * ) ̸ = 0. Hence, we conclude:
∇ log q(z * ) = 0.
Step 3: Local maximality. By Lemma A.5 and Assumption 4, the sequence q(z t ) is strictly increasing and converges to q(z * ). If z * were a saddle point or local minimum, there would exist a direction u ∈ R d such that the second-order Taylor expansion yields:
d 2 dt 2 log q(z * + tu) t=0 = u ⊤ ∇ 2 log q(z * )u ≥ 0,
which contradicts the fact that z t → z * through ascent directions. Therefore, all second directional derivatives must be negative, implying:
∇ 2 log q(z * ) ≺ 0.
That is, z * is a strict local maximum of q.
this section cite: []

Section: A.2 PROPOSITION 3.1
We report here a detailed formulation of
Proposition 3.1 alongside its proof. Proposition A.7. Let f = E • D : R d → R d be the composition of an autoencoder's decoder D and encoder E, and define the residual vector field v(z) := f (z) -z. Consider the reconstruction loss L(z) := ∥f (z) -z∥ 2 . Then, the iteration z t+1 = f (z t ) corresponds locally to gradient descent on L (i.e., v(z) ∝ -∇ z L(z)) if either of the following conditions holds:
this section cite: []

Section: B.5 RANK OF ATTRACTOR MATRIX
In Figure 13 we report an alternative measure of generalization of the attractors computed in the experiment in Figure 2 and in Figure 3. Specifically, we consider the matrix X * = D(Z) * , and we compute its singular value decomposition U * diag(s * )V * = X * . We define the generalization entropy as the number of eigenvectors needed to explain 90% of the variance of the matrix. We report this measure for both the models employed in the experiment in Figure2 and in Figure 3, showing that attractors are more expressive as the model generalized better, as well during training.
this section cite: []

Section: B.6 MEMORIZATION OVERFITTING REGIME
In this section, we test the memorization capabilities of models in different (strong) overparametrization regimes, similarly as tested in (Kadkhodaie et al., 2023) for diffusion models and in (Radhakrishnan et al., 2020;Zhang et al., 2019) for the extremely overparametrized case (network trained on few samples). We remark that this case corresponds to an overfitting regime of the network, as opposed Transitioning from memorization to generalization as underfitting (over-regularized) regime showed in the main paper. We train a convolutional autoencoder on subsamples of the CIFAR,MNIST,FashionMNIST datasets, with bottleneck dimension k = 128 and weight decay 1e -4. In Figure 14 we plot the memorization coefficient as a function of the dataset size, showing that networks trained on less data( strong overparametrization) tend to memorize more the training data. We replicate the experiment in 4.1, on the larger XL version of the Stable diffusion architecture (Rombach et al., 2022), observing that attractors from noise still form an informative dictionary of signals which scales better w.r.t, to a random orthogonal basis.
this section cite: ['b27', 'b46', 'b65', 'b50']

Section: B.8 BEYOND AUTOENCODERS MODEL
In this section we preliminary explore the extent to which latent vector exist beyond pretrained autoencoder models.
this section cite: []

Section: B.8.1 LATENT VECTOR FIELDS IN SELF SUPERVISED MODELS
We first consider the case discriminative self-supervised (SSL) models such as DINOv2 Oquab et al. (2023) andSigLIP2 Tschannen et al. (2025). A straightforward idea to extend our framework to encoder-only models, is the one of training a decoder on top of the frozen encoder model. To this end we employ the decoders trained in Zheng et al. (2025), which provide pretrained decoders models for DINOv2 and SigLIP2.
this section cite: ['b41', 'b67']

Section: C REGULARIZED AUTOENCODER ENFORCE CONTRACTIVENESS
In Table 3, we list many AE variants including denoising AEs (DAEs) (Vincent et al., 2008), sparse AEs (SAEs) (Ng et al., 2011), variational AEs (VAEs) (Kingma et al., 2013) and other variants (Rifai et al., 2011;Alain and Bengio, 2014;Gao et al., 2024) and show how their objectives enforce local contractive solutions around training points.
We provide below a precise connection for the case of weight decay and feed-forward networks.
Example: weight decay Consider an L-layer feed-forward autoencoder F θ = D θ • E θ , where each layer is defined as h ℓ = ϕ ℓ (W ℓ h ℓ-1 + b ℓ ) and the activations ϕ ℓ are 1-Lipschitz (e.g., ReLU, GELU, SiLU, tanh). The overall map can be written as
F θ (x) = W L ϕ L-1 W L-1 • • • ϕ 1 (W 1 x) . Let D ℓ (x) = diag(ϕ ′ ℓ (W ℓ h ℓ-1 (x)
)) be the diagonal Jacobian of the activation at layer ℓ. By the chain rule,
J θ (x) = W L D L-1 (x)W L-1 • • • D 1 (x)W 1 .
Since each activation is 1-Lipschitz, ∥D ℓ (x)∥ 2 ≤ 1 for all ℓ, and therefore
∥J θ (x)∥ 2 ≤ L ℓ=1 ∥W ℓ ∥ 2 .
A sufficient condition for local contraction is thus
L ℓ=1 ∥W ℓ ∥ 2 < 1 ⇒ ∥J θ (x)∥ 2 < 1.
Training with weight decay minimizes L(θ) + λ L ℓ=1 ∥W ℓ ∥ 2 F , and since ∥W ℓ ∥ 2 ≤ ∥W ℓ ∥ F , this suppresses the product of spectral norms appearing above:
L ℓ=1 ∥W ℓ ∥ 2 ≤ L ℓ=1 ∥W ℓ ∥ F ,
biasing the solution toward ∥J θ (x)∥ 2 < 1 even without an explicit Jacobian penalty.
this section cite: ['b59', 'b40', 'b29', 'b48', 'b1', 'b15']

Section: E USE OF LARGE LANGUAGE MODELS (LLMS)
In this work, we used LLMs for occasionally polishing and improving the readability of the paper. All substantive research contributions, analysis, and interpretations were carried out by the authors.
this section cite: []

Section: References
Ref_id:b0 Title: Gated autoencoders with tied input weights Year: (2013)
Ref_id:b1 Title: What regularized auto-encoders learn from the data-generating distribution Year: (2014)
Ref_id:b2 Title: Smollm2: When smol goes big-data-centric training of a small language model Year: (2025)
Ref_id:b3 Title: A closer look at memorization in deep networks Year: (2017)
Ref_id:b4 Title: Deep equilibrium models Year: (2019)
Ref_id:b5 Title: Representation learning: A review and new perspectives Year: (2013)
Ref_id:b6 Title: Nonlinear power method for computing eigenvectors of proximal operators and neural networks Year: (2021)
Ref_id:b7 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b8 Title: Describing textures in the wild Year: (2014)
Ref_id:b9 Title: Why do we need weight decay in modern deep learning? Year: (2024)
Ref_id:b10 Title: Investigating data memorization in 3d latent diffusion models for medical image synthesis Year: (2023)
Ref_id:b11 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b12 Title: Nonlinear spectral geometry processing via the tv transform Year: (2020)
Ref_id:b13 Title: Latent functional maps: a spectral framework for representation alignment Year: (2024)
Ref_id:b14 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b15 Title: Scaling and evaluating sparse autoencoders Year: (2024)
Ref_id:b16 Title: Nonlinear spectral analysis via one-homogeneous functionals: overview and future prospects Year: (2016)
Ref_id:b17 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b18 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b19 Title: Deep residual learning for image recognition Year: (2016-06)
Ref_id:b20 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b21 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b22 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b23 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b24 Title: The platonic representation hypothesis Year: (2024)
Ref_id:b25 Title: Associative memory in iterated overparameterized sigmoid autoencoders Year: (2020)
Ref_id:b26 Title: Highly accurate protein structure prediction with alphafold Year: (2021)
Ref_id:b27 Title: Generalization in diffusion models arises from geometry-adaptive harmonic representations Year: (2023)
Ref_id:b28 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b29 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b30 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b31 Title: A simple weight decay can improve generalization Year: (1991)
Ref_id:b32 Title: Neural networks: Tricks of the trade Year: (2002)
Ref_id:b33 Title: Datasets: A community library for natural language processing Year: (2021-11)
Ref_id:b34 Title: Rob Van de Loo, Rob Vogels, et al. 1399 h&e-stained sentinel lymph node sections of breast cancer patients: the camelyon dataset Year: (2018)
Ref_id:b35 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b36 Title: Matching pursuits with time-frequency dictionaries Year: (1993)
Ref_id:b37 Title: An empirical bayes estimator of the mean of a normal population Year: (1961)
Ref_id:b38 Title: Relative representations enable zero-shot latent space communication Year: (2022)
Ref_id:b39 Title: Deep double descent: Where bigger models and more data hurt Year: (2021)
Ref_id:b40 Title: Sparse autoencoder. CS294A Lecture notes Year: (2011)
Ref_id:b41 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b42 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b43 Title: Exponential expressivity in deep neural networks through transient chaos Year: (2016)
Ref_id:b44 Title: Generalization beyond overfitting on small algorithmic datasets Year: (2022)
Ref_id:b45 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b46 Title: Overparameterized neural networks implement associative memory Year: (2020)
Ref_id:b47 Title: Hopfield networks is all you need Year: (2020)
Ref_id:b48 Title: Contractive autoencoders: Explicit invariance during feature extraction Year: (2011)
Ref_id:b49 Title: An empirical bayes approach to statistics Year: (1992)
Ref_id:b50 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b51 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b52 Title: Diffusion art or digital forgery? investigating data replication in diffusion models Year: (2023)
Ref_id:b53 Title: How to train your energy-based models Year: (2021)
Ref_id:b54 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b55 Title: Out-of-distribution detection with deep nearest neighbors Year: (2022)
Ref_id:b56 Title: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features Year: (2025)
Ref_id:b57 Title: The inaturalist species classification and detection dataset Year: (2018)
Ref_id:b58 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b59 Title: Extracting and composing robust features with denoising autoencoders Year: (2008)
Ref_id:b60 Title: Transformers: State-of-the-art natural language processing Year: (2020-10)
Ref_id:b61 Title: Sun database: Exploring a large collection of scene categories Year: (2016)
Ref_id:b62 Title: Qwen3 technical report Year: (2025)
Ref_id:b63 Title: Generalized out-of-distribution detection: A survey Year: (2024)
Ref_id:b64 Title: Deep structured energy based models for anomaly detection Year: (2016)
Ref_id:b65 Title: Identity crisis: Memorization and generalization under extreme overparameterization Year: (2019)
Ref_id:b66 Title: Understanding deep learning (still) requires rethinking generalization Year: (2021)
Ref_id:b67 Title: Diffusion transformers with representation autoencoders Year: (2025)
Ref_id:b68 Title: Anomaly detection with robust deep autoencoders Year: (2017)
Ref_id:b69 Title: A survey on latent reasoning Year: (2025)
