Title: In-Context Denoising with One-Layer Transformers: Connections between Attention and Associative Memory Retrieval
Abstract: We introduce in-context denoising, a task that refines the connection between attention-based architectures and dense associative memory (DAM) networks, also known as modern Hopfield networks. Using a Bayesian framework, we show theoretically and empirically that certain restricted denoising problems can be solved optimally even by a single-layer transformer. We demonstrate that a trained attention layer processes each denoising prompt by performing a single gradient descent update on a context-aware DAM energy landscape, where context tokens serve as associative memories and the query token acts as an initial state. This one-step update yields better solutions than exact retrieval of either a context token or a spurious local minimum, providing a concrete example of DAM networks extending beyond the standard retrieval paradigm. Overall, this work solidifies the link between associative memory and attention mechanisms first identified by Ramsauer et al., and demonstrates the relevance of associative memory models in the study of in-context learning.

Section: Introduction
The transformer architecture (Vaswani et al., 2017) has achieved remarkable success across diverse domains, from natural language processing (Devlin et al., 2019;Brown et al., 2020;Touvron et al., 2023) to computer vision (Dosovitskiy et al., 2021). Despite their practical success, un-derstanding the mechanisms behind transformer-based networks remains an open challenge. This challenge is exacerbated by the growing scale and complexity of modern large networks. Toward addressing this, researchers studying simplified architectures have identified connections between the attention operation that is central to transformers and associative memory models (Ramsauer et al., 2021), providing not only an avenue for understanding how such architectures encode and retrieve information but also potentially ways to improve them further.
The most celebrated model for associative memories in systems neuroscience is the so-called Hopfield model (Amari, 1972;Nakano, 1972;Little, 1974;Hopfield, 1982). This model has a capacity to store "memories" (stable fixed points of a recurrent update rule) proportional to the number of nodes (Hopfield, 1982;Amit et al., 1985). In the last decade, new energy functions (Krotov & Hopfield, 2016;Demircigil et al., 2017) were proposed for dense associative memories with much higher capacities. These energy functions are often referred to as modern Hopfield models. Ramsauer et al. (2021) pointed out the similarity between the one-step update rule of a certain modern Hopfield network (Demircigil et al., 2017) and the softmax attention layer of transformers, generating interest in the statistical physics and systems neuroscience communities (Krotov & Hopfield, 2021;Krotov, 2023;Lucibello & Mézard, 2024;Millidge et al., 2022). Recent work has extended this concept to improve retrieval by incorporating sparsity (Hu et al., 2023;Wu et al., 2024b;Santos et al., 2024;Wu et al., 2024a), while others have leveraged associative memory principles to design new energy-based transformer architectures (Hoover et al., 2023). However, these extensions and the foundational construction in Ramsauer et al. (2021) primarily focus on the specific task of exact retrieval (converging to a fixed point), while in practice transformers may tackle many other tasks.
To explore this connection beyond retrieval, we introduce in-context denoising, a task that bridges the behavior of trained transformers and associative memory networks through the lens of in-context learning (ICL). In standard ICL, a sequence model is trained to infer an unknown function g from contextual examples, predicting g(X L+1 ) given a sequence of input-output pairs E = ((X 1 , g(X 1 )), ..., (X L , g(X L )), (X L+1 , -)). Crucially, g is implied solely through the context and differs across prompts -performant models are therefore said to "learn g(x) in context". While ICL has been extensively studied in supervised settings (Garg et al., 2022;Zhang et al., 2024;Akyürek et al., 2023;Reddy, 2024), recent work suggests that transformers may internally emulate gradient descent over a context-specific loss function during inference (Von Oswald et al., 2023;Dai et al., 2023;Ahn et al., 2023). This general perspective aligns with our findings.
In this work, we generalize ICL to an unsupervised setting where the prompt consists of L samples from a random distribution and the query is a noise-corrupted sample from the same distribution. This shift allows us to probe how trained transformers internally approximate Bayes optimal inference, while deepening the connection to associative memory models which are prototypical denoisers. By setting up this problem in this way, we also attempt to answer a few questions. One concerns the memorization-generalization dilemma in denoising: a Hopfield model's success is usually measured by successful memory recovery, while in-context learning may have to solve a completely new problem. Another question has to do with the number of iterations of the corresponding Hopfield model: why does the Ramsauer et al. (2021) correspondence involve only one iteration of Hopfield energy minimization and not many?
In summary, our contributions are as follows: In Section 2, we introduce in-context denoising as a framework for understanding how transformers perform implicit inference beyond memory retrieval. In Section 3, we establish that single-layer transformers with one attention head are expressive enough to optimally solve certain denoising problems. We then empirically demonstrate that standard training from random weights can recover the Bayes optimal predictors. The trained attention layers are mapped back to dense associative memory networks in Section 4. Our results refine the general connection pointed out in previous work, offer new mechanistic insights into attention, and provide a concrete example of dense associative memory networks extending beyond the standard memory retrieval paradigm to solve a novel in-context learning task.
this section cite: ['b42', 'b12', 'b7', 'b40', 'b13', 'b34', 'b3', 'b32', 'b28', 'b22', 'b22', 'b5', 'b26', 'b11', 'b34', 'b11', 'b27', 'b25', 'b30', 'b31', 'b23', 'b37', 'b19', 'b34', 'b15', 'b46', 'b1', 'b35', 'b43', 'b10', 'b0', 'b34']

Section: Problem formulation: In-context denoising
In this section, we describe our general setup. Recurring common notation is described in Appendix A.1.
this section cite: []

Section: Setup
Each task corresponds to a distribution D over the probability distribution of data:
p X ∼ D. Let X 1 , • • • , X L+1
iid ∼ p X , define the sampling of the tokens. Let the noise cor-ruption be defined by X ∼ p noise (•|X L+1 ). The random sequence E = (X 1 , X 2 , ..., X L , X) are given as "context" (input) to a sequence model F (•; θ) which outputs an estimate XL+1 of the original (L + 1)-th token . The task is to minimize the expected loss E[l( XL+1 , X L+1 )] for some loss function l(•, •). Namely, our problem is to find
min θ E p X ∼D,X 1:L+1 ∼p L+1 X , X∼pnoise(•|X L+1 ) [l(F (E, θ), X L+1 )].(1)
In practice, we choose X = X L+1 + Z, a pure token corrupted by the addition of isotropic Gaussian noise Z ∼ N (0, σ 2 Z I n ), and our objective function to minimize is the mean squared error
(MSE) E[|| XL+1 -X L+1 || 2 ].
In the following subsection, we explain the pure token distributions for three specific tasks. These tasks are of course structured so that a one-layer transformer has the expressivity to capture a solution, which, as L → ∞, provides an optimal solution, in some sense. To that end, we derive Bayes optimal estimators for each of the three tasks, under the assumption that we know the original distribution p X of pure tokens. In Section 3, we use these estimators as baselines to evaluate the performance of the denoiser f (E, θ) based on a one-layer transformer trained on finite datasets.
this section cite: []

Section: Task-specific token distributions
We consider three elementary in-context denoising tasks, where the data (vectors in R n ) comes from:
1. Linear manifolds (d-dimensional subspaces) 2. Nonlinear manifolds (d-spheres) 3. Small noise Gaussian mixtures (clusters) where the component means have fixed norm Below we describe the task-specific distributions p X and the process for sampling tokens {x t }. The same corruption process applies to all cases: X = X L+1 +Z, Z ∼ N (0, σ 2 Z I n ).
this section cite: []

Section: CASE 1 -LINEAR MANIFOLDS
A given training prompt consists of pure tokens sampled from a random d-dimensional subspace S of R n .
• Let P be the orthogonal projection operator to a random d-dim subspace S of R n , sampled according to the uniform measure, induced by the Haar measure on the coset space O(n)/O(n -d) × O(d), on the Grassmanian G(d, n), the manifold of all d-dimensional subspaces of R n .
• Let Y ∼ N (0, σ 2 0 I n ) and define X = P Y ; we use this procedure to construct the starting sequences (X 1 , ..., X L+1 ) of L + 1 independent tokens. We thus have p X = N (0, σ 2 0 P ), with the Haar distribution of P characterizing the task ensemble associated with D.
this section cite: []

Section: CASE 2 -NONLINEAR MANIFOLDS
We focus on the case of d-dimensional spheres of fixed radius R centered at the origin in R n .
• Choose a random d+1-dimensional subspace V of R n , sampled according to the uniform measure, as before, on the Grassmanian G(d + 1, n). The choice of this random subspace generates the distribution of tasks D.
• Inside V , sample uniformly from the radius R sphere (once more, a Haar induced measure on a coset space O(d + 1)/O(d)). We use this procedure to construct input sequences X 1:L+1 = (x 1 , ..., x L+1 ) of L + 1 independent tokens.
In practice, we uniformly sample points with fixed norm in R d and embed them in R n by concatenating zeros. We then rotate the points by selecting a random orthogonal matrix Q ∈ R n×n .
this section cite: []

Section: CASE 3 -GAUSSIAN MIXTURES (CLUSTERING)
Pure tokens are sampled from a weighted mixture of isotropic Gaussians in n-dimensions, {w a , (µ a , σ 2 a )} K a=1 . The density is
p X (x) = K a=1 w a C a e -∥x-µa∥ 2 /2σ 2 a ,
where C a = (2πσ 2 a ) -n/2 are normalizing constants. The µ a are independently chosen from a uniform distribution on the radius R sphere of dimension n -1, centered around zero. The distribution of tasks D, is decided by the choice of {µ a } K a=1 . For our ideal case, we will consider the limit that the variances go to zero. In that case, the density is simply
p X0 (x) = K a=1 w a δ(x -µ a ).
this section cite: []

Section: Bayes optimal denoising baselines for each case
The first L tokens in E are "pure samples" from p that should provide information about the distribution for our denoising task. Our performance is expected to be no better than that of the best method, in the case that the token distribution and also the corrupting process are exactly known. This is where the Bayesian optimal baseline comes in. As is well-known, the Bayes optimal predictor of a quantity is given by the posterior mean. We use that fact to compute the Bayes optimal loss.
In particular, we seek a function f : R n → R n such that
E X, X ∥X -f ( X)∥ 2 is minimized. Since the perturba- tion Z is Gaussian, the posterior distribution of X, given X is p X| X (x | x) = C(x)p X (x)e -∥x-x∥ 2 /2σ 2 Z ,
where C(x) is a normalizing factor (see Appendix A.2 for more explanation). The following proposition sets up a baseline to which we expect to compare our results as L → ∞. The proof is in Appendix B.1.
Proposition 1. For each task, specified by the input distribution p X , and the noise model p X|X ,
E X, X ∥X -f ( X)∥ 2 ≥ E X Tr Cov(X | X) . (2) This lower bound is met when f ( X) = E[X | X].
Thus, the Bayes optimal denoiser is the posterior expectation for X given X. The expected loss is found by computing the posterior sum of variances.
These optimal denoisers can be computed analytically for both the linear and nonlinear manifold cases (given the variances and dimensionalities). In the Gaussian mixture (clustering) case, it depends on the choice of the centroids which then needs to be averaged over.
Linear case. For the linear denoising task, pure samples X are drawn from an isotropic Gaussian in a restricted subspace. The following result provides the Bayes optimal predictor in this case, the proof of which is in Appendix C.1. Proposition 2. For p X corresponding to Subsection 2.2.1, the Bayes optimal answer is
f opt ( X) = E[X| X] = σ 2 0 σ 2 0 + σ 2 Z P X,(3)
and the expected loss is
E ∥P X -X L+1 ∥ 2 = dσ 2 0 σ 2 Z /(σ 2 0 + σ 2 Z ).(4)
Projection Projection (shrunk)
Figure 2. Baseline estimators for the case of random linear manifolds with projection operator P (i) .
this section cite: []

Section: Manifold case.
In the nonlinear manifold denoising problem, we focus on the case of lower dimensional spheres S (e.g. the circle S 1 ⊂ R 2 ). For such manifolds, the Bayes optimal answer is given by the following proposition. Proposition 3. For p X defined as in Subsection 2.2.2, with P being the orthogonal projection operator to V , the d + 1 dimensional linear subspace, with R being the radius of sphere S, the Bayes optimal answer is
f opt ( X) = E[X | X] = e ⟨x, X∥⟩/σ 2 Z x dS x e ⟨x, X∥⟩/σ 2 Z dS x (5
) = I d+1 2 R ∥ X∥∥ σ 2 Z I d-1 2 R ∥ X∥∥ σ 2 Z R X∥ ∥ X∥∥ ,(6)
where X∥ = P X and I ν is the modified Bessel function of the first kind.
Clustering case. For clustering with isotropic Gaussian mixtures {w a , (µ a , σ 2 a )} p a=1 , the Bayes optimal predictors for some important special cases are as follows. See Appendix C.3 for the general case.
Proposition 4. For general isotropic Gaussian model with σ a = σ 0 , ||µ a || = R for all a = 1, . . . , K.
f opt ( X) = E[X| X] = σ 2 0 σ 2 0 + σ 2 Z X + σ 2 Z σ 2 0 + σ 2 Z a w a e ⟨µa, X⟩/(σ 2 0 +σ 2 Z ) µ a a w a e ⟨µa, X⟩/(σ 2 0 +σ 2 Z )
.
(7)
If σ 0 → 0, f opt ( X) = E[X | X] = a w a e ⟨µa, X⟩/σ 2 Z µ a a w a e ⟨µa, X⟩/σ 2 Z .(8)
In all three cases, we notice similarities between the form of the Bayes optimal predictor, and attention operations in transformers, a connection which we explore below.
this section cite: []

Section: In-context denoising with one-layer transformers -Empirical results
In this section, we provide simple constructions of one-layer transformers that approximate (and under certain conditions, exactly match) the Bayes optimal predictors above.
Input: Let p (1) X , . . . , p (N ) X
iid ∼ D, be distributions sampled for one of the tasks. For each distribution p
(i) X , we sample E (i) := (X (i) 1 , . . . , X (i) L , X(i) ) taking value in R n×(L+1)
be an input to a sequence model. We also retain the true (L + 1)-th token X (i) L+1 for each i. Objective: Given an input sequence E (i) , return the uncorrupted final token X (i) L+1 . We consider the meansquared error loss over a collection of N training pairs,
{E (i) , X (i) L+1 } N i=1 , C(θ) = N i=1 ∥F (E (i) , θ) -x (i) L+1 ∥ 2 , (9
)
where F (E (i) , θ) denotes the parametrized function predicting the target final token based on input sequence E (i) .
this section cite: []

Section: One-layer transformer and the attention between the query and pure tokens
To motivate our choice of architecture, let us start by discussing the linear case.
There we have f opt ( X) = σ 2 0 σ 2 0 +σ 2 Z P X. Note that, by the strong law of large numbers, P = 1
σ 2 0 L L t=1 X t X T
t is a random matrix that almost surely converges component-bycomponent to the orthogonal projection P as L → ∞, since, for each t, X t X T t has the expectation σ 2 0 P and that X t is a Gaussian random variable with zero mean and a finite covariance matrix. So we could propose
f ( X) = σ 2 0 σ 2 0 + σ 2 Z P X = 1 (σ 2 0 + σ 2 Z )L L t=1 X t ⟨X t , X⟩.(10)
We now consider a simplified one-layer linear transformer (see Appendices D.1 and D.2 for more detailed discussions) which still has sufficient expressive power to capture our finite sample approximation to the Bayes optimal answer. We define
X = F Lin (E, θ) := 1 L W P V X 1:L X T 1:L W KQ X(11)
taking values in R n , where
X 1:L := [X 1 , . . . , X L ] taking values in R n×L , with learnable weights W KQ , W P V ∈ R n×n abbreviated by θ. Note that, when W P V = αI n , W KQ = βI n , and αβ = 1 σ 2 0 +σ 2 Z , F (E, θ) should ap- proximate the Bayes optimal answer f opt ( X) as L → ∞.
For a detailed discussion of the convergence rate, see Appendix E, in general, and Proposition 5, in particular.
Similarly, we could argue that the second two problems, the d-dimesional spheres and the σ 0 → 0 zero limit of the Gaussian mixtures could be addressed by softmax attention
X = F (E, θ) := W P V X 1:L softmax(X T 1:L W KQ X) (12)
taking values in R n . The function softmax(z) := 1 n i=1 e z i (e z1 , . . . , e zn ) T ∈ R n is applied column-wise. For both problems, namely the spheres and the σ 0 → 0 Gaussian mixtures, we could have
W P V = αI n , W KQ = βI n with α = 1, β = 1/σ 2
Z providing Bayes optimal answers as L → ∞.
In fact, we could make a more general statement about distributions p X where the norm of X is fixed. Theorem 3.1. If we have a task distribution D so that the support of each p X is the subset of some sphere, centered around the origin, with a p X -dependent radius R, then the function
F (({X t } L t=1 , x), θ * ) = L t=1 X t e ⟨Xt,x⟩/σ 2 Z L t=1 e ⟨Xt,x⟩/σ 2 Z (13
)
converges almost surely to the Bayes optimal answer f opt (x) for all x ∈ R n , as L → ∞. The optimal parameter θ * refers to
W P V = I n , W KQ = 1 σ 2 Z I n .
The proof of the theorem is in Appendix D.3. See Appendix E, particularly Proposition 6, for consideration of convergence rates. Note that the condition of p X being supported on a sphere is not artificial as, in many practical transformers, pre-norm with RMSNorm gives you inputs on the sphere, up to learned diagonal multipliers.
Note that the natural form of attention that is suggested by our formulation of in-context denoising would involve Gaussian kernels:
X = F G (E, θ) := t W P V X t e -1 2 ||W K Xt-W Q X|| 2 t e -1 2 ||W K Xt-W Q X|| 2 .
(14) The relation between softmax attention and the Gaussian kernel has been noted in (Choromanski et al., 2021;Ambrogioni, 2024) and a Gaussian kernel-based attention is implemented in (Chen et al., 2021). A related Hopfield energy, with W K , W Q , and W P V proportional to identity matrices, is proposed in (Hoover et al., 2024a).
For the linear case, we use linear attention, but that may not be essential. Informally speaking, the softmax attention model has the capacity to subsume the linear attention model. Proposition 3.2. As ϵ → 0,
F E, 1 ϵ W P V , ϵW KQ = 1 ϵ W P V X + 1 L W P V L t=1 X t (X t -X) T W KQ X + O(ϵ),(15)
where X = 1 L L t=1 X t is the empirical mean.
See Appendix F for the details of small W KQ expansion and Appendix F.1 for the proof of Proposition 3.2.
For case 1, note that E[X t ] = 0 and covariance of X t is finite, E[ X] = 0, and E[|| X|| 2 ] = O( 1 L ), allowing us to drop X as L → ∞. If, in addition, ϵ is small, only the second term survives. Thus, F E, ( 1 ϵ W P V , ϵW KQ ) starts to approximate F Lin E, (W P V , W KQ ) when L is large and ϵ is small, with ϵ √ L large. We therefore could use the softmax model for all three cases.
this section cite: ['b9', 'b4', 'b8']

Section: Case 1 -Linear manifolds
The Bayes optimal predictor for the linear denoising task from Section 2.3 suggests that the linear attention weights should be scaled identity matrices with their product satisfying αβ = 1 σ 2 0 +σ 2 Z . Fig. 3 shows that a one-layer network of size n = 16 trained on tasks with σ 2 Z = 1, σ 2 0 = 2, d = 8, L = 500 indeed achieves this bound, training to nearly diagonal weights with the appropriate scale ⟨w  Fig. 4(a) displays how this bound is approached as the context length L of training samples is increased. In Fig. 4(b) we study how the performance of a model trained to denoise random subspaces of dimension d = 8 is affected by shifts in the subspace dimension at inference time. We find that when provided sufficient context, such models can adapt with mild performance loss to solve more challenging tasks not present in the training set.
It is evident from Fig. 3(a) that the softmax network performs similarly to the linear one for this task. We can understand this through the small argument expansion of the softmax function mentioned above. The learned weights displayed in Fig. 3(b) indicate that β softmax ≈ 0.194 becomes small (note it decreases by a factor ϵ ≈ 0.344 relative to β linear ), while the value scale α softmax ≈ 1.607 becomes larger by a similar factor ∼ 1/ϵ to compensate. Thus, although the optimal denoiser for this case is intuitively expressed through linear self-attention, it can also be achieved with softmax self-attention in the appropriate limit.
Moreover, we find that when the entire prompt undergoes a global invertible transformation A ̸ = I, the optimal attention weights are no longer scaled identity matrices but acquire a structured form determined by A. Both linear and softmax attention layers are able to recover this structure through training; see Appendix H for details and empirical verification.
this section cite: []

Section: Case 2 -Nonlinear manifolds
Fig. 3 (case 2) shows networks of size n = 16 trained to denoise subspheres of dimension d = 8 and radius R = 1, with corruption σ 2 Z = 0.1 and context length L = 500. Once again, the network trains to have scaled identity weights.
We note that although the network nearly achieves the optimal MSE on the test set, the weights appear at first glance to deviate slightly from the Bayes optimal predictor of Section 2.3, which indicated W P V = αI, W KQ = βI with α = 1, β = 1/σ 2 Z . To better understand this, we consider a coarse-grained MSE loss landscape by scanning over α and β. See Fig. 6(a) in Appendix G. We find that the 2D loss landscape has roughly hyperbolic level sets which is suggestive of the linear attention limit, where the weight scales become constrained by their product αβ. Reflecting the symmetry of the problem, we also note mirrored negative solutions (i.e. one could also identify α = -1, β = -1/σ 2 Z from the analysis in Section 2.3). Importantly, the plot shows that the trained network lies in the same valley of the loss landscape as the optimal predictor, in agreement with Fig. 3. Moreover, the shape of the loss landscape suggested that linear attention might also be applicable to this case, which we demonstrate and discuss further in Appendix G. train test L=50 L=30 L=500 Performance maintained away from d=8 Train n=16 model: d=8, L=500 Loss of trained network linear projection Effect of context length L on training Shifting the subspace dimension at inference time (a) in-context learning subspace provided only via context subspace dimension can vary (b) Predict Mean diagonal weights of trained network weight scaling Figure 4. (a) Trained linear attention network converges to Bayes optimal estimator as context length increases (n = 16, d = 8, σ 2 0 = 2, σ 2 z = 1). (b) A network trained to denoise subspaces of dimension d = 8 can accurately denoise subspaces of different dimensions presented at inference time, given sufficient context.
this section cite: []

Section: Case 3 -Gaussian mixtures
nents that have isotropic variance σ 2 0 = 0.02 and centers randomly placed on the unit sphere in R n . The corruption magnitude is σ 2 Z = 0.1 and context length is L = 500. The baselines show the zero predictor (dashed grey line) as well as the optimum from Proposition (4) (pink) and its σ 2 0 → 0 approximation Eq. ( 8) (grey).
The trained weights qualitatively approach the optimal estimator for the zero-variance limit but with a slightly different scaling: while the scale of
W P V is α ≈ 1, the W KQ scale is β ≈ 5.127 < 1/σ 2 Z .
To study this, we provide a corresponding plot of the 2D loss landscape in Fig. 6(a) in Appendix G. While the symmetry of the previous case has been broken (the context cluster centers {µ a } will not satisfy ⟨µ⟩ = 0), we again find that the trained network lies in the anticipated global valley of the MSE loss landscape.
this section cite: []

Section: Connection to dense associative memory networks
In each of the denoising problems studied above, we have shown analytically and empirically that the optimal weights of the one-layer transformer are scaled identity matrices W P V ≈ αI, W KQ ≈ βI. In the softmax case, the trained denoiser can be concisely expressed as
x = g(X 1:L , x) := αX 1:L softmax(βX T 1:L x),
re-written such that X ∈ R n×L stores pure context tokens.
We now demonstrate that such denoising corresponds to one-step gradient descent (with specific step sizes) of energy models related to dense associative memory networks, also known as modern Hopfield networks (Ramsauer et al., 2021;Demircigil et al., 2017;Krotov & Hopfield, 2016).
Consider the energy function:
E(X 1:L , s) = 1 2α ∥s∥ 2 - 1 β log L t=1 e βX
T t s , (16) which mirrors the Ramsauer et al. (2021) construction but with a Lagrange multiplier added to the first term. Figure 5 illustrates this energy landscape for the spherical manifold case. query target prediction trajectories Num. steps: 1 Num. steps: 50 context tokens Z .
An operation inherent to the associative memory perspective is the recurrent application of a denoising update. Gradient descent iteration s(t + 1) = s(t) -γ ∇ s E X 1:L , s(t) yields
s(t + 1) = 1 - γ α s(t) + γX 1:L softmax βX T 1:L s(t) .(17)
It is now clear that initializing the state to the query s(0) = x and taking a single step with size γ = α recovers the behavior of the trained attention model (Fig. 5). The attention mechanism here is thus mechanistically interpretable: the context tokens X 1:L induce a context-dependent associative memory landscape, while the query acts as an initial condition for inference-time gradient descent. One could naturally consider alternative step sizes and recurrent iteration. However, Fig. 5 demonstrates that naive iteration of Eq. ( 17) has the potential to degrade performance.
Additional details are provided in Appendix I. In particular, the energy model for linear attention is discussed in Appendix I.1.
this section cite: ['b34', 'b11', 'b26']

Section: Discussion
Motivated by the connection between attention mechanisms and dense associative memories, here we have introduced incontext denoising, a task that distills their relationship. We first analyze the general problem, deriving Bayes optimal predictors for certain restricted tasks. We identify that onelayer transformers using either softmax or linearized selfattention are expressive enough to describe these predictors. We then empirically demonstrate that standard training of attention layers from random initial weights will readily converge to scaled identity weights with scales that approach the derived optima given sufficient context. Accordingly, the rather minimal transformers studied here can perform optimal denoising of novel tasks provided at inference time via self-contained prompts. This work therefore sheds light on other in-context learning phenomena, a point we return to below.
While practical transformers differ in various ways from the minimal models studied here, we note several key connections. Intriguingly, the self-attention heads of trained transformers sometimes exhibit weights W KQ , W P V that resemble scaled identity matrices, i.e. cI + ϵ with small fluctuations ϵ ij ∼ N (0, σ 2 ), an observation noted in Trockman & Kolter (2023). This phenomenon motivated their proposal of "mimetic" weight initialization schemes mirroring this learned structure. Relatedly, connections to associative memory concepts have been explored in other architectures (Smart & Zilman, 2021), which enabled data-dependent weight initialization strategies to be identified and leveraged.
More broadly, our study suggests that trained attention layers can readily adopt structures that facilitate context-aware associative retrieval. We have also noted preliminary connections between our work and other architectural features of modern transformers, namely layer normalization and residual streams, which warrant further study.
In-context denoising and generative modeling both involve learning about an underlying distribution, suggesting potential relationships between these two tasks. Recently, Pham et al. (2024) invoked spurious states of the Hopfield model as a way of understanding how one can move away from retrieving individual memorized patterns towards generalization via appropriate mixtures of multiple similar "memories". In our work, one-step updates do not have to land in a spurious minimum, but we often operate under circumstances where there are such states (see, for example, the energy landscape in Fig. 5). More generally, analogies between energy-based associative memory and diffusion models have recently been noted (Ambrogioni, 2024;Hoover et al., 2024b). Lastly, Bayes optimal denoisers play an important role in the analysis (Ghio et al., 2024) of a very related generative model that is based on stochastic interpolants (Albergo & Vanden-Eijnden, 2023). Although this work focuses on the case where it is possible to sample enough tokens from the relevant distributions for certain functions to converge, generative models become important when the distribution is in a prohibitively high-dimensional space making direct sampling difficult. Nonetheless, investigating the precise relationship between our work and different generative modeling approaches would be an interesting direction to pursue.
Overall, this work refines the connection between dense associative memories and attention layers first identified in (Ramsauer et al., 2021). While we show that one energy minimization step of a particular DAM (associated with a trained attention layer) is optimal for the denoising tasks studied here, it remains an open question whether multilayer architectures with varying or tied weights could extend these results to more complex tasks by effectively performing multiple iterative steps. This aligns with recent studies on in-context learning, which have considered whether transformers with multiple layers emulate gradient descent updates on a context-specific objective (Von Oswald et al., 2023;Shen et al., 2024;Dai et al., 2023;Ahn et al., 2023), and may provide a bridge to work on emerging architectures guided by associative memory principles (Hoover et al., 2023). Investigating when and how multilayer attention architectures perform such gradient descent iterations in a manner that is both context-dependent and informed by a large training set represents an exciting direction for future research at the intersection of transformer mechanisms, associative memory retrieval, and in-context learning.
this section cite: ['b41', 'b39', 'b33', 'b4', 'b16', 'b2', 'b34', 'b43', 'b38', 'b10', 'b0', 'b19']

Section: References
Ref_id:b0 Title: Transformers learn to implement preconditioned gradient descent for in-context learning Year: (2023)
Ref_id:b1 Title: What learning algorithm is in-context learning? investigations with linear models Year: (2023)
Ref_id:b2 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b3 Title: Learning patterns and pattern sequences by selforganizing nets of threshold elements Year: (1972)
Ref_id:b4 Title: In search of dispersed memories: Generative diffusion models are associative memory networks Year: (2024)
Ref_id:b5 Title: Spinglass models of neural networks Year: (1985)
Ref_id:b6 Title: A spherical hopfield model Year: (2003)
Ref_id:b7 Title: Language models are few-shot learners Year: (2020)
Ref_id:b8 Title: Skyformer: Remodel self-attention with gaussian kernel and nyström method Year: (2021)
Ref_id:b9 Title: Rethinking attention with performers Year: (2021)
Ref_id:b10 Title: Why can gpt learn in-context? language models implicitly perform gradient descent as meta-optimizers Year: (2023)
Ref_id:b11 Title: On a model of associative memory with huge storage capacity Year: (2017)
Ref_id:b12 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b13 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b14 Title:  Year: (1993)
Ref_id:b15 Title: What can transformers learn in-context? a case study of simple function classes Year: (2022)
Ref_id:b16 Title: Sampling with flows, diffusion, and autoregressive neural networks from a spin-glass perspective Year: (2024)
Ref_id:b17 Title: Table of Integrals, Series, and Products Year: (2007)
Ref_id:b18 Title: Probability inequalities for sums of bounded random variables. The collected works of Wassily Hoeffding Year: (1994)
Ref_id:b19 Title: Energy transformer Year: (2023)
Ref_id:b20 Title: Dense associative memory through the lens of random features Year: (2024)
Ref_id:b21 Title: Memory in plain sight: Surveying the uncanny resemblances of associative memories and diffusion models Year: (2024)
Ref_id:b22 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b23 Title: On sparse modern hopfield model Year: (2023)
Ref_id:b24 Title: Transformers are rnns: fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b25 Title: A new frontier for hopfield networks Year: (2023)
Ref_id:b26 Title: Dense associative memory for pattern recognition Year: (2016)
Ref_id:b27 Title: Large associative memory problem in neurobiology and machine learning Year: (2021)
Ref_id:b28 Title: The existence of persistent states in the brain Year: (1974)
Ref_id:b29 Title: Probability theory i. Graduate Texts in Mathematics Year: (1977)
Ref_id:b30 Title: Exponential capacity of dense associative memories Year: (2024-02)
Ref_id:b31 Title: Universal hopfield networks: A general framework for single-shot associative memory models Year: (2022)
Ref_id:b32 Title: Associatron-a model of associative memory Year: (1972)
Ref_id:b33 Title: Memorization to generalization: The emergence of diffusion models from associative memory Year: (2024)
Ref_id:b34 Title: Hopfield networks is all you need Year: (2021)
Ref_id:b35 Title: The mechanistic basis of data dependence and abrupt learning in an in-context classification task Year: (2024)
Ref_id:b36 Title: High-dimensional statistics Year: (2023)
Ref_id:b37 Title: Sparse and structured hopfield networks Year: (2024-07)
Ref_id:b38 Title: Position: Do pretrained transformers learn in-context by gradient descent? Year: (2024-07)
Ref_id:b39 Title: On the mapping between hopfield networks and restricted boltzmann machines Year: (2021)
Ref_id:b40 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b41 Title: Mimetic initialization of selfattention layers Year: (2023)
Ref_id:b42 Title: Attention is all you need Year: (2017-12)
Ref_id:b43 Title: Transformers learn in-context by gradient descent Year: (2023)
Ref_id:b44 Title: Uniform memory retrieval with larger capacity for modern hopfield models Year: ()
Ref_id:b45 Title: Sparse tandem hopfield model for memoryenhanced time series prediction Year: ()
Ref_id:b46 Title: Trained transformers learn linear models in-context Year: (2024)
