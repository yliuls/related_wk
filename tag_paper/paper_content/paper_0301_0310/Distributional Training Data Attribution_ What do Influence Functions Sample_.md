Title: Distributional Training Data Attribution: What do Influence Functions Sample?
Abstract: Randomness is an unavoidable part of training deep learning models, yet something that traditional training data attribution algorithms fail to rigorously account for. They ignore the fact that, due to stochasticity in the initialisation and batching, training on the same dataset can yield different models. In this paper, we address this shortcoming through introducing distributional training data attribution (d-TDA), the goal of which is to predict how the distribution of model outputs (over training runs) depends upon the dataset. Intriguingly, we find that influence functions (IFs), a popular data attribution tool, are 'secretly distributional': they emerge from our framework as the limit to unrolled differentiation, without requiring restrictive convexity assumptions. This provides a new perspective on the effectiveness of IFs in deep learning. We demonstrate the practical utility of d-TDA in experiments, including improving data pruning for vision transformers and identifying influential examples with diffusion models.

Section: Introduction
Training data attribution (TDA) techniques are of fundamental interest in machine learning, shedding light on the relationship between a model's properties and its training data. TDA is typically framed as a counterfactual prediction problem: estimating how a model's behaviour would change upon removal of particular examples from the training dataset [1,2] . This invites the concept of influence. Training examples are deemed 'influential' if the model's behaviour would change significantly upon their exclusion. The practical utility of TDA has been demonstrated in applications including interpreting, debugging and improving models [2,3], dataset curation [4], and data valuation [1,5].
this section cite: ['b0', 'b1', 'b1', 'b2', 'b3', 'b0', 'b4']

Section: Influence Functions.
It is typically prohibitively expensive to compute influence by retraining with different datapoints removed. This has motivated a number of TDA methods designed to approximate influence, but without actually retraining. Amongst such TDA methods, a leading example is influence functions (IFs) [2,6]. This classical technique from robust statistics uses the implicit function theorem to estimate the optimal model parameters' sensitivity to downweighting a training datapoint. IFs have been deployed to investigate the generalisation patterns of 52 billion parameter large language models [3], and for data attribution of diffusion models [7]. Separately, researchers have proposed an alternative TDA method called unrolled differentiation [8,9,10]. Here, one differ-Figure 1: Distributional training data attribution. Classical data attribution methods like influence functions are typically motivated using convex loss functions, predicting a deterministic shift to the unique optimal model weights (left). In contrast, this paper advocates for a distributional perspective, approximating the new probability distribution over model parameters/outputs after removal of training examples (interpreted as perturbation of the training loss). This includes for non-convex loss functions (right).
entiates through a particular training trajectory to directly obtain the sensitivity of the final model parameters to the weighting of a particular example in the loss function. Unrolled differentiation tends to work better than IFs in experiments, but it is more expensive to compute.
this section cite: ['b1', 'b5', 'b2', 'b6', 'b7', 'b8', 'b9']

Section: Randomness in training.
The success of IFs in deep learning is perhaps surprising because the classical foundations of both TDA and IFs fail to account for a core property of modern training: stochasticity [11]. Given the randomness inherent in weight initialisation and mini-batching, training can be understood as sampling from a distribution over final models. Each training run corresponds to drawing a single sample from this distribution. Yet classical TDA is only defined for deterministic training algorithms, and IFs are primarily understood for convex objectives (or by finding convex proxies [11]). Stochasticity is usually dismissed as a nuisance for TDA methods, glossed over in method derivations [2]. At best, it is sometimes heuristically managed by ensembling or averaging [12]. In practice, stochasticity makes it difficult to diagnose which changes to model behaviour are attributable to changes in the training dataset, and which are due to sampling randomness.
Introducing distributional training data attribution. In this paper, we argue that the randomness in model training is not a nuisance. Conversely, it ought to play a central role in our understanding of influence, and deserves a proper mathematical treatment. Viewing training as sampling from a distribution over final model weights (or outputs), the goal of TDA should be to efficiently predict changes to this distribution under modifications to the training dataset: a novel perspective that we coin distributional training data attribution (d-TDA). Figure 1 provides a visual schematic.
Influence functions are distributional. We show that unrolled differentiation is natively a d-TDA method (Section 3.1). Subsequently, in Section 4.1, we rigorously show that IFs approximate unrolled differentiation for long enough training times, and hence IFs are already inherently distributional.
Core contributions. (1) We introduce distributional training data attribution (d-TDA), a framework for studying data attribution in stochastic deep learning settings (Section 3). (2) We show that influence functions (IFs) are 'secretly distributional', solving special limiting cases of a d-TDA task (Section 4). This may help explain the effectiveness of IFs in deep learning, far from the convex setting in which they were originally proposed. (3) We propose distributional influence, which quantifies the importance of examples by how much their inclusion/exclusion affects the distribution over model weights and outputs. We show that distributional influence captures interesting information missing from its regular predecessor, and leads to more effective data pruning (Section 5).
this section cite: ['b10', 'b10', 'b1', 'b11', 'b0', 'b1', 'b2']

Section: Background 'Classical' Training Data Attribution (TDA).
Consider the space 𝔇 ≔ ∪ ∞ 𝑁=1 𝒵 𝑁 of possible finite training datasets 𝒟 ≔ (𝑧 𝑖 ) 𝑁  𝑖=1 . In classical TDA, one is concerned with deterministic training algorithms 𝜽 * : 𝔇 → ℝ 𝑑 param , which take a dataset as their input and return 'trained' model parameters 𝜽 * (𝒟). The goal of TDA is to predict how the output of the training algorithm 𝜽 * would change if it were run using a perturbed training dataset 𝒟 ′ , with some examples removed. Concretely, given some trained model 𝜽 * (𝒟), TDA methods θ * (𝒟 ′ ) aim to approximate 𝜽 * (𝒟 ′ ) ≈ θ * (𝒟 ′ ) without actually retraining the model. Of course, in practice, one is typically interested in the change in some measurement function 𝑚 : ℝ 𝑑 param → ℝ 𝑑 m when the dataset is modified -for instance, the loss on a particular test example. Therefore, TDA methods θ * (⋅) are typically evaluated on their ability to approximate 𝑚(𝜽 * (𝒟 ′ )) ≈ 𝑚( θ * (𝒟 ′ )).
'Classical' influence. The discussion above invites the concept of influence. The influence of an example is the change in the measurement 𝑚 ⚬ 𝜽 * when the example is removed from the training dataset. Influential samples change the measurement by a large amount. The influence of a training datapoint 𝑧 𝑘 with respect to a measurement function 𝑚 is given by:
Inf(𝑧 𝑘 ) ≔ 𝑚(𝜽 * (𝒟)) -𝑚(𝜽 * (𝒟 \ 𝑧 𝑘 )).
(
This is extended to groups of examples (𝑧 𝑖 ) 𝑁 𝑘 𝑘=1 ⊂ 𝒟 in the obvious way. To approximate Inf(𝑧 𝑘 ) without actually retraining, one uses a TDA method to approximate 𝜽 * (𝒟 \ 𝑧 𝑘 ) .
this section cite: []

Section: Response.
A practical difficulty posed by the formulation of influence in Eq. ( 1) is that 𝔇, the domain of the training algorithm 𝜽 * (⋅), is discontinuous. Datapoints 𝑧 𝑘 are either included or not included. The binary nature of this choice makes it difficult to analyse Inf(𝑧 𝑘 ) directly using gradient-based methods. Hence, it is typical to instead consider a continuous relaxation to the training algorithm.
Let us introduce a scalar 𝜀 ∈ [0, 1  𝑁 ] which controls the weighting of a particular example in the training algorithm. Suppose 𝜀 = 0 corresponds to inclusion and 𝜀 = 1  𝑁 corresponds to exclusion, with intermediate values meaning the example is still present but downweighted. The precise setup will depend on the training algorithm of interest. Let 𝜽 * 𝒟→𝒟\𝑧 𝑘 (𝜀) denote the (assumed deterministic) outcome of the training algorithm with loss ℒ 𝒟→𝒟\𝑧 𝑘 (𝜀). Provided 𝜽 * 𝒟→𝒟\𝑧 𝑘 (𝜀) is continuous and twice-differentiable at 𝜀= 0, we have that
𝜽 * 𝒟→𝒟\𝑧 𝑘 (𝜀) = 𝜽 * (𝒟) + 𝜀 d𝜽 * 𝒟→𝒟\𝑧 𝑘 (𝜀) d𝜀 | 𝜀=0 + 𝑂(𝜀 2 ) as 𝜀 → 0. Response 𝒓(𝑧 𝑘 ) ≔ (2) Hence, we define the response 𝒓(𝑧 𝑘 ) ≔ d𝜽 * 𝒟→𝒟\𝑧 𝑘 (𝜀) d𝜀 | 𝜀=0 , such that 𝑚(𝜽 * (𝒟 \ 𝑧 𝑘 )) = 𝑚(𝜽 * (𝒟)) + 𝜀∇𝑚 ⊤ 𝒓(𝑧 𝑘 ) + 𝒪(𝜀 2
). Intuitively, response measures the sensitivity of the training algorithm output with respect the weighting 𝜀 of the example 𝑧 𝑘 ∈ 𝒟. To make this more explicit, we will now give two concrete examples: influence functions and unrolled differentiation.
this section cite: []

Section: Influence functions.
Many classical algorithms only depend on the data through a loss function ℒ 𝒟 (𝜽)≔ 1  𝑁 ∑ 𝑁 𝑛=1 ℓ 𝑛 (𝜽) with ℓ 𝑛 : ℝ 𝑑 param → ℝ some per-example loss. One natural way to codify downweighting in that case is to define an interpolated loss ℒ 𝒟→𝒟\𝑧 𝑘 (𝜀)≔ℒ 𝒟 -𝜀ℓ 𝑘 . Suppose that the loss function ℒ 𝒟 has a single unique minimum and that the training algorithm successfully locates it. Mathematically, this can be written as 𝜽 * (𝒟) = argmin 𝜽∈ℝ 𝑑 ℒ 𝒟 (𝜽). Minimising the interpolated loss ℒ 𝒟→𝒟\𝑧 𝑘 (𝜀) and applying the implicit function theorem, it is straightforward to prove that in this special case:
𝒓(𝑧 𝑘 ) = ∇ 2 ℒ 𝒟 (𝜽 * (𝒟)) -1 ∇ℓ 𝑘 (𝜽 * (𝒟)) ≕ 𝒓 IF . (3
)
We derive this result in detail in Section B. 𝒓 IF is referred to as an influence function (IF) -a popular TDA tool. The effectiveness of IFs for deep learning is perhaps surprising given the unrealistic assumptions made during their derivation.
this section cite: []

Section: Unrolled differentiation.
Suppose instead that the model is trained using stochastic gradient descent (SGD).
Consider the weight update rule 𝜽 𝑡+1 = 𝜽 𝑡 -1 𝐵 ∑ 𝑁 𝑛=1 𝛿 𝑡 𝑛 ∇ℓ 𝑛 (𝜽 𝑡 ), where (𝜽 𝑡 ) 𝑡∈ℕ denotes the trajectory of model parameters and 𝜽 0 is some random initialisation. 𝜹 𝑡 with 𝑡 ∈ ℕ are independently and identically distributed batching variables in {0, 1} 𝑁 , with mean 𝔼[𝛿 𝑡 𝑛 ] = 𝐵 𝑁 . In close analogy to the interpolated loss ℒ 𝒟→𝒟\𝑧 𝑘 (𝜀), consider the interpolated update step:¹ 𝜽 𝑡+1 (𝜀) = 𝜽 𝑡 (𝜀) -𝜂 𝑡 𝐵 ∑ 𝑁 𝑛=1 𝛿 𝑡 𝑛 ∇ℓ 𝑛 (𝜽 𝑡 )(1 -𝜀𝟙 𝑛=𝑘 ),
where 𝟙 𝑛=𝑘 is the indicator function. If we train for 𝑇 timesteps, one can directly differentiate through the training trajectory to obtain the sensitivity of the final model weights 𝜽 𝑇 with respect to the weighting 𝜀. Applying the chain rule, one obtains a rather cumbersome expression (Eq. ( 57) in Section D). In this setting, we call 𝒓 UD ≔ d𝜽 𝑇 d𝜀 | 𝜀=0 the unrolled differentiation response. 𝒓 UD can be used as a classical TDA method if we consider all sources of randomness to be fixed. This algorithm tends to work better than IFs in experiments, but the repeated computation and caching of Hessians makes its naive implementation expensive for long training runs. This has prompted work on approximate unrolled differentiation [9].
this section cite: ['b8']

Section: Distributional Training Data Attribution
An obvious problem with the classical TDA formulation described in Section 2 is that in reality training is stochastic: the randomness in model initialisation and SGD precludes defining a deterministic map 𝜽 * : 𝒟 → ℝ 𝑑 param . Even retraining with an identical dataset will in general give a different model; 𝜽 * (𝒟) is better thought of as a random variable. Previous work has dealt with this randomness heuristically by averaging over training ensembles [2,9]. In contrast, in this paper we advocate for a more rigorous distributional perspective. Taking the model initialisation 𝜽 0 and the batch selections (𝜹 𝑡 ) 𝑡∈ℕ to be random variables on some probability space (Ω, ℱ, ℙ), we frame distributional training data attribution as follows.
this section cite: ['b1', 'b8']

Section: Distributional training data attribution (d-TDA).
Let 𝜽 * (𝒟) be the outcome of (stochastic) training with some dataset 𝒟 ∈ 𝔇. Let 𝜇 𝒟 (𝐴) ≔ ℙ[𝜽 * (𝒟) ∈ 𝐴] (for 𝐴 ∈ ℱ) denote its probability distribution. Let 𝑚 # 𝜇 𝒟 be the distribution of some measurement function 𝑚 : ℝ 𝑑 param → ℝ 𝑑 m of the trained model. The goal of distributional TDA is to reason about the behaviour of 𝜇 𝒟 and 𝑚 # 𝜇 𝒟 with respect to changing 𝒟 -especially, removing examples by taking 𝒟 → 𝒟 \ 𝑧 𝑘 .
Rather than considering randomness to be a nuisance, d-TDA acknowledges that the training dataset determines the distribution over trained models. Effective d-TDA methods answer questions like:
1. Given samples from 𝜇 𝒟 , how can I approximately sample from 𝜇 𝒟\𝑧 𝑘 ? 2. If removed from the training dataset, which example 𝑧 𝑘 ∈ 𝒟 would most drastically change 𝜇 𝒟 ? 3. Which examples should I remove to change the variance of 𝑚 # 𝜇 𝒟 upon retraining?
The fact that d-TDA predicts changes in distributions over measurements leads us to reevaluate the notion of influence. In particular, removing influential samples ought to substantially modify 𝑚 # 𝜇 𝒟 . With this in mind, we define distributional influence (c.f. Eq. ( 3)) as follows:
this section cite: []

Section: Definition 1. (Distributional influence).
The distributional influence of a training example 𝑧 𝑘 ∈ 𝒟 with respect to a measurement function 𝑚 : ℝ 𝑑 param → ℝ 𝑑 m is given by:
DistInf(𝑧 𝑘 ) ≔ Δ(𝑚 # 𝜇 𝒟 ‖𝑚 # 𝜇 𝒟\𝑧 𝑘 ),(5)
where Δ(𝜇 1 ‖ 𝜇 2 ) is some 'difference function' between 𝜇 1 and 𝜇 2 .
There exist many possible instantiations of distributional influence, depending on the choice of Δ.
Letting 𝑋 ∼ 𝜇 1 , 𝑌 ∼ 𝜇 2 denote the final measurement random variables, one could consider:
this section cite: []

Section: Mean influence Variance increase influence Wasserstein influence
Δ(𝜇 1 ‖𝜇 2 ) ≔ 𝔼(𝑋) -𝔼(𝑌 ) Var(𝑌 ) -Var(𝑋) 𝒲 2 (𝜇 1 , 𝜇 2 )
this section cite: []

Section: Distributional influence with unrolled differentiation
To compute DistInf(𝑧 𝑘 ), we need to (approximately) sample from 𝜇 𝒟\𝑧 𝑘 without retraining the model. This can be achieved using unrolled differentiation, described by the pseudocode below.
¹This can be roughly thought of as SGD updates with the interpolated loss function ℒ 𝒟→𝒟\𝑧 𝑘 (𝜀).
this section cite: []

Section: Alg. 1. Unrolled differentiation for d-TDA.
1. Sample 𝜽 * (𝒟) ≔ 𝜽 𝑇 from 𝜇 𝒟 by training the model with stochastic updates (Eq. ( 4)).
2. Obtain approximate samples from 𝜇 𝒟\𝑧 𝑘 without retraining by taking 𝜽 * (𝒟 \ 𝑧 𝑘 ) ≈ 𝜽 * (𝒟) + 1 𝑁 𝒓 UD , with 𝒓 UD ≔ d𝜽 𝑇 d𝜀 | 𝜀=0 the unrolled differentiation response. If interested in the distribution over some measurement, compute 𝑚(𝜽 * (𝒟 \ 𝑧 𝑘 )) ≈ 𝑚(𝜽 * (𝒟)) + 1 𝑁 ∇𝑚(𝜽 * (𝒟)) ⊤ 𝒓 UD . 3. Using these two sets of (correlated) samples, compute the difference function Δ between the empirical distributions to efficiently approximate DistInf(𝑧 𝑘 ).
When computing 𝒓 UD , the following observation simplifies differentiating through long training trajectories.
this section cite: []

Section: Remark 1. (Unrolled differentiation is a Markov chain).
Applying the chain rule of differentiation to Eq. ( 4) gives the following recursive formula for (𝜽 𝑡 , 𝒓 𝒕 ) ≔ (𝜽 𝑡 ,
d𝜽 𝑡 d𝜀 | 𝜀=0 ): ( 𝜽 𝑡+1 𝒓 𝑡+1 ) = ( ( ( 𝜽 𝑡 - 𝜂 𝑡 𝐵 ∑ 𝑁 𝑛=1 𝛿 𝑡 𝑛 ∇ℓ 𝑛 (𝜽 𝑡 ) (𝐼 - 𝜂 𝑡 𝐵 ∑ 𝑁 𝑛=1 𝛿 𝑡 𝑛 ∇ 2 ℓ 𝑛 (𝜽 𝑡 ))𝒓 𝑡 + 𝜂 𝑡 𝐵 𝛿 𝑡 𝑘 ∇ℓ 𝑘 (𝜽 𝑡 ) ) ) ) .(6)
Intuitively, Eq. ( 6) shows that the response 𝒓 𝑡+1 depends on the response at the previous timestep 𝒓 𝑡 , modulated by the loss function curvature. If datapoint 𝑧 𝑘 is present in the batch sampled at timestep 𝑡, 𝒓 𝑡+1 also depends on the corresponding loss gradient ∇ℓ 𝑘 (𝜽 𝑡 ). Practically, Eq. ( 6) permits us to compute the final response 𝒓 𝑇 at linear time and constant space complexity with respect to training duration, without caching or explicitly computing the batch Hessians (c.f. Eq. ( 57)). This is akin to forward-mode automatic differentiation for meta-learning [13]. To the best of our knowledge, this is the first application of such techniques to efficient computation of the response. Crucially, if the batch selection 𝛿 𝑡 𝑛 is i.i.d., Eq. ( 6) defines a Markov Chain -an observation that unlocks well-studied mathematical machinery and invites us to analyse its limiting distribution (see Section 4.1).
this section cite: ['b12']

Section: Empirical demonstration.
Figure 2 showcases the application of unrolled differentiation as a d-TDA method, successfully predicting changes in the distribution of measurements when a select subset of the dataset is removed.
Figure 2: d-TDA demo for a neural network trained on UCI Concrete. d-TDA (using unrolled differentiation) gives approximate samples from the distribution of models re-trained on some fixed subset 𝒟 ′ ⊂ 𝒟 (left). Actual samples from 𝜇 𝒟 ′ (obtained by expensive retraining) are closer to predicted samples from 𝜇 𝒟 ′ (obtained by efficient d-TDA methods) than they are to samples from the original model 𝜇 𝒟 , both in terms of their means (centre) and Wasserstein distance (right). The distributions are over measurements on query samples for different stochastic training runs.
Why not use regular TDA with a fixed seed? A natural question raised by the challenge of stochasticity is: instead of treating the outcome of training as a random variable, why not simply fix all sources of randomness? Why not just stratify by the random choices like initialisation and data ordering? Naively, this seems to recover a deterministic training algorithm, to which one may apply regular (non-distributional) TDA methods. We refer to this as 'fixed-seed TDA'. Distributional TDA is often preferable to fixed-seed TDA because many methods, like IFs, more accurately perform the d-TDA task, even when failing at the fixed-seed one. For instance, IFs and unrolled differentiation find local perturbations around the current optimum to predict outcomes of counterfactual retraining. However, even fixing all randomness, chaotic training dynamics can push training into a completely different region of the parameter space. Moreover, when using a fixed batch-size, even tiny changes to the training set size can offset at what iteration each datum appears. This means even fixed-seed trajectories can converge to widely different 'optima' under small dataset perturbations, rendering the shift in the original local optimum inadequate. Later in Section 5, we demonstrate IFs perform better on downstream tasks as a distributional TDA method compared to as a fixed-seed TDA method. This suggests the distributional perspective provides a more accurate picture of how influence functions work.
this section cite: []

Section: What do influence functions sample?
In Section 3, we introduced distributional TDA, adopting a rigorous mathematical perspective that accounts for stochasticity in training. Here, we demonstrate how d-TDA relates to classical influence functions (IFs; Eq. ( 3)). From two complementary perspectives, we find that IFs are actually 'secretly distributional', appearing as asymptotic samples in specific d-TDA settings. In stark contrast to usual derivations of IFs, which rely on assumptions that are unrealistic for deep learning [2,14], we place only mild constraints on ℒ(𝜽). All proofs are in Section A.
this section cite: ['b1', 'b13']

Section: Perspective 1: unrolled differentiation converges a.s. to influence functions
Adopting a distributional perspective, a natural question is: what is the limiting distribution of the random variable (𝜽 𝑡 , 𝒓 𝑡 ), updated according to Eq. ( 6)? Begin by considering the model weights (𝜽 𝑡 ), which are updated by SGD with i.i.d. batch selection. We make the following assumptions.
this section cite: []

Section: A1.
∇ 2 ℒ and ∇ℓ 𝑘 are Lipschitz continuous and bounded. A2. The step sizes (𝜂 𝑡 ) ∞ 𝑡=0 are positive scalars satisfying ∑ 𝑡 𝜂 𝑡 = ∞ and ∑ 𝑡 𝜂 2 𝑡 < ∞. A3. The iterates of Eq. (12) remain bounded a.s., i.e. sup 𝑡 ‖(𝜽 𝑡 , 𝒓 𝑡 )‖ < ∞ a.s.
Standard results due to e.g. H. J. Kushner and G. G. Yin [15] give us the following result: Theorem 1 demonstrates that, with a suitably decaying learning rate, SGD converges to critical points -namely, saddle points or local minima. Define the set of local minima as follows:
𝒮 𝑚 ℒ ≔ {𝜽 : -∇ℒ(𝜽) = 0, -∇ 2 ℒ(𝜽) ⪯ 0} ⊆ 𝒮 ℒ .(7)
If the weights converge to a saddle point in 𝒮 ℒ \ 𝒮 𝑚 ℒ , it is intuitive that the response 𝒓 𝑡 ≔ d𝜽 𝑡 d𝜀 | 𝜀=0 will diverge. This is because the final model parameters will become sensitive to any infinitesimal perturbation of the loss function.² Conversely, if the weights converge to a local minimum, the limiting behaviour of 𝒓 𝑡 becomes tractable. Consider the following additional assumptions.
A4. ∇ℓ 𝑘 (𝜽) ∈ Span(∇ 2 ℒ(𝜽)) for all 𝜽 ∈ 𝒮 𝑚 ℒ . A5. The nonzero eigenvalues of ∇ 2 ℒ(𝜽) for 𝜽 ∈ 𝒮 𝑚 ℒ are uniformly bounded away from 0. A6. There exists some compact neighborhood 𝒩(𝒮 𝑚 ℒ ) around 𝒮 𝑚 ℒ such that gradient flow trajectories 𝜽(𝑡) initialised therein converge uniformly over initialisations to points in 𝒮 𝑚 ℒ . Moreover, their lengths are bounded a.s., so that: sup 𝜽(0)∈𝒩(𝒮 𝑚 ℒ ) ∫ ∞ 𝑠=0 ‖𝜽(𝑠) -lim 𝑠 ′ →∞ 𝜽(𝑠 ′ )‖ d𝑠 ′ < ∞. Theorem 2. (Unrolled differentiation converges to IFs). Suppose that A1-A6 hold, and consider an SGD trajectory in the set that converges to 𝒮 𝑚 ℒ (c.f. 𝒮 ℒ \ 𝒮 𝑚 ℒ ). The sequence of iterates ((𝜽 𝑡 , 𝒓 𝑡 )) ∞ 𝑡=0 generated by Eq. (6) converges almost surely to the set ℛ * ≔ {(𝜽 * , 𝒓 IF (𝜽 * ) + 𝒓 NS (𝜽 * )) : 𝜽 * ∈ 𝒮 𝑚 ℒ , 𝒓 IF (𝜽) ≔ ∇ 2 ℒ(𝜽) + ∇ℓ 𝑘 (𝜃), 𝒓 NS (𝜽) ∈ Null(∇ 2 ℒ(𝜽))} -that is, pointwise IFs, plus a component in the Hessian nullspace. (⋅) + is the pseudoinverse. ²This could be interpreted as a limitation of the conventional notions of response and influence.
Proof sketch. We consider the ODE which is the continuous time relaxation of Eq. ( 6). We prove that the solution to this ODE 𝒓(𝑡) converges to an influence function, plus a component in the flat directions of a minimum manifold. Under the learning rate assumptions above, the SGD updates asymptotically track this ODE, which allows us to prove the final result. ∎ Commentary on Theorem 2. The response iterate 𝒓 𝑡 either diverges (in the case that the weights 𝜽 𝑡 converge to a saddle), or converges to ℛ * . Hence, at late times, one can approximately sample from 𝜇 𝒟\𝑧 𝑘 by sampling from 𝜇 𝒟 and offsetting by 1  𝑁 𝒓 IF . Using 𝒓 IF instead of 𝒓 UD means that 1) we assume we have trained for long enough, and 2) we neglect components of response in the Hessian nullspace. The latter may not converge and will in general depend on the history of SGD iterates 𝜽 𝑡 .
Assumptions. A1-A3 are standard assumptions, needed to ensure that SGD converges. A4 guarantees the perturbation ℓ 𝑘 doesn't have a component in the flat directions of the minimum manifold, or else unrolled differentiation diverges. Note that A4 automatically holds for any symmetries shared by ℒ and ℓ 𝑘 , e.g. due to neural network parameterisation. A5 ensures that IFs remain bounded; Hessian eigenvalues on 𝒮 𝑚 ℒ can be zero, but not nonzero and arbitrarily small. The most technically meaningful assumption is A6, which assumes gradient flow converges sufficiently fast to local minima. We stress that it is much less restrictive than the requirements usually cited for IFs to apply, such as strong convexity [2,9,14].
this section cite: ['b14', 'b1', 'b8', 'b13']

Section: Remark 2.
For Generalised Linear Models (GLMs), the component of the unrolled response 𝒓 𝑡 in the nullspace of the Hessian is 0 throughout training. Hence, in this setting, Theorem 2 gives exact convergence of the unrolled response to the influence functions formula.
Figure 3: Validating Theorem 2. Top: Correlation between changes in measurement predicted by unrolled differentiation and changes predicted by IFs, plotted against training time. The coefficient becomes high as the Markov chain converges. The correlation is stronger for small 𝜂 where SGD is closer to gradient flow [15]. Bottom: Norm of the measurement gradient component in the span of the Hessian divided by norm of the measurement gradient. The nullspace component of ∇𝑚 remains tiny.
this section cite: ['b14']

Section: Empirical validation.
In Figure 3, we test our theoretical results on a regression task with UCI Concrete. As predicted by Theorem 2, the strength of correlation becomes very high (90%) at late times as the unrolled differentiation Markov Chain converges to IFs. As expected, the correlation is better for lower step sizes, for which the SGD iterates (normalised by learning rate) track gradient flow more closely.
We also experimentally confirm that the component of the unrolled response 𝒓 𝑡 in the null space of the Hessian -that is, the error term that IFs cannot capture -is insignificant for the practical tasks we test. In the lower panel of Figure 3, we see that the measurement gradient ∇𝑚 only has a tiny component in the nullspace of the Hessian, living almost entirely in the column space. Recalling that 𝑚(𝜽 * (𝒟 \ 𝑧 𝑘 )) ≈ 𝑚(𝜽 * (𝒟)) + 1  𝑁 ∇𝑚 ⊤ 𝒓 UD , this means it barely contributes to predicted changes in 𝑚. As such, the fact that IFs do not capture this part of the limiting distribution of 𝒓 𝑡 does not appear to be of substantial concern for downstream tasks.
this section cite: []

Section: Perspective 2: transport maps between Boltzmann distributions
Departing from unrolled differentiation, we now instead model the final weights by a Boltzmann distribution. This is motivated by the fact that it is the limiting distribution of Stochastic Gradient Langevin Dynamics [16,17], which closely resembles SGD. Given an energy function ℒ(𝜽) -𝜀ℓ 𝑘 (𝜽) and an inverse temperature parameter 𝛽 ∈ ℝ + , the Boltzmann distribution is:
𝑝 𝛽 𝜀 (𝜽) = 𝑒 -𝛽(ℒ(𝜽)-𝜀ℓ 𝑘 (𝜽)) 𝑍(𝛽, 𝜀) 𝑍(𝛽, 𝜀) = ∫ 𝑒 -𝛽(ℒ(𝜽)-𝜀ℓ 𝑘 (𝜃)) d𝜽. (8
)
Let
this section cite: ['b15', 'b16']

Section: Theorem 3. (Asymptotic optimality of IFs with Boltzmann distributions).
Let ℛ ⊂ 𝐶 1 (ℝ 𝑑 , ℝ 𝑑 ) denote the class of bounded vector fields 𝒓 such that 𝑇 𝜀 (𝜽) ≔ 𝜽 + 𝜀𝒓(𝜽) is a 𝐶 1 diffeomorphism for all sufficiently small 𝜀 > 0. Define the functional
ℱ(𝒓, 𝜀) ≔ lim 𝛽→∞ 1 𝛽 𝐷 KL (𝑇 𝜀# 𝑃 𝛽 0 |𝑃 𝛽 𝜀 ),(9)
equal to the asymptotic KL divergence between the transformed base measure 𝑇 𝜀# 𝑃 𝛽 0 and the true perturbed measure 𝑃 𝛽 𝜀 . Consider the subset of maps ℛ IF ≔ {𝒓 ∈ ℛ : 𝒓(𝜽) = 𝒓 IF (𝜽) for 𝜽 ∈ 𝒮 𝑔 ℒ } ⊂ ℛ, for which the map is equal to influence functions on the minimum manifold. Then, given any 𝒓 ∈ ℛ IF and any 𝒓 ′ ∈ ℛ \ ℛ IF , there exists some 𝑎 ∈ ℝ + such that
ℱ(𝒓, 𝜀) ≤ ℱ(𝒓 ′ , 𝜀) ∀ |𝜀| ≤ 𝑎. (10
)
Moreover, the set ℛ IF is non-empty, so such diffeomorphisms do indeed exist.
Proof sketch. We start by showing that, for small enough 𝜀, there do indeed exist continuously differentiable bijections in the class 𝑇 𝜀 (𝜽) such that 𝒓(𝜽) = 𝒓 IF (𝜽) when 𝜽 ∈ 𝒮 𝑔 ℒ . At low temperatures, only the behaviour at 𝒮 𝑔 ℒ matters because the probability mass concentrates where the loss is minimised. Taking 𝛽 → ∞ and using the Laplace approximation, we analyse the low-temperature KL divergence between 𝑇 𝜀# 𝑃 𝛽 0 (the transformed measure, without loss perturbation) and 𝑃 𝛽 𝜀 (the measure with loss perturbation). Among the class 𝑇 𝜀 (𝜽), this is minimised at 𝒪(𝜀 2 ) terms by IFs. ∎ Commentary on Theorem 3. For Boltzmann distributions, IFs provide exactly the transport map in 𝑇 𝜀 (𝜽) required to transform the low-temperature (weak limit) Boltzmann distribution with loss ℒ(𝜽) onto the Boltzmann distribution with a perturbed loss ℒ(𝜽) -𝜀ℓ 𝑘 (𝜽), up to 𝒪(𝜀 2 ) terms. This provides a very explicit distributional motivation for IFs: they map samples from 𝑃 ∞ 0 (read: 𝜇 𝒟 ) onto approximate samples from 𝑃 ∞ 𝜀 (read: 𝜇 𝒟\𝑧 𝑘 ), and do so approximately optimally in the KL sense. We also remark that, since the KL divergence is invariant under parameter transformation, this notion of optimality does not depend on the specific choice of coordinate system. Minimal assumptions are made on ℒ(𝜽) throughout for Theorem 3 to hold.
this section cite: []

Section: Key takeaways from Section 4.
IFs are implicitly distributional. Supposing 𝜽 * (𝒟) ∼ 𝜇 𝒟 , then the sample 𝜽 * (𝒟) + 1  𝑁 𝒓 IF is approximately distributed according to 𝜇 𝒟\𝑧 𝑘 in two precise mathematical senses: (1) as an asymptotic limit of unrolled differentiation, and (2) minimising a KL divergence if the final weights follow low-temperature Boltzmann distributions. This means that we can use 𝒓 IF instead of 𝒓 UD in Alg. 1 as a cheaper yet principled proxy, unlocking d-TDA at scale. It may also help explain why IFs are effective in deep learning, far from the convexity assumptions relied upon during typical derivations from robust statistics.
this section cite: []

Section: Distributional Training Data Attribution in Practice
Having demonstrated how d-TDA can be operationalised using IFs (Section 4), we now discuss its practical utility. We begin by demonstrating that distributional influence captures interesting information missing from its classical counterpart.
this section cite: []

Section: Rethinking influence.
Previous papers have heuristically considered what amounts to mean influence [12] -if removed from the training dataset, which example would change a model measurement most on average? In a synthetic 1D regression task shown in Figure 5, this criterion identifies 𝑥 30 as the most influential datapoint. As discussed above, we could quantify the difference in distributions after retraining in a different way, e.g. with Wasserstein influence. Here, in contrast, 𝑥 31 is deemed the most influential. Note that 𝑥 31 is not very influential by conventional measures since its mean shift is modest, yet its removal drastically changes the behaviour after training, sharply increasing uncertainty. This demonstrates that different notions of distributional influence can capture meaningful information about the training data missed by e.g. heuristic ensembling. As a second demonstration, in Figure 11 (App. C.2.3) we use d-TDA to identify MNIST examples which lead to a large change  Test accuracy improvements on CIFAR-10 with a SWIN Vision Transformer from IF data pruning. We compare two approaches to subset selection: 1) traditional TDA with a fixed seed, where for each random seed we remove 5000 datapoints that are predicted to decrease the validation loss the most for the model trained with that specific fixed seed; and 2) distributional-TDA, where for each model we remove 5000 datapoints predicted to decrease the validation loss the most on average. Both methods lead to test accuracy improvements upon the baseline trained with all data. However, d-TDA leads to greater overall improvements on average. Black dots show accuracies for individual models (with seeds indicated in gray), whereas coloured diamonds ◆ indicate the average result for each method.
in variance but a small change in mean. The right panel of the figure confirms that retraining does actually lead to the changes our methods predict.
this section cite: ['b11']

Section: Distributional influence on diffusion models.
To illustrate this concept at scale, we apply distributional influence to identify the most influential training examples for a latent diffusion model [7]. Figure 7 ranks the most and least influential examples on ArtBench, comparing Wasserstein influence, mean influence, and classical fixed-seed influence. The identified examples vary in each case.
The ability to predict how different training examples impact the distribution over training runs may be practically useful. For example, one could identify examples to add or remove to most reduce variance, hence reducing model (epistemic) uncertainty. Further, d-TDA methods could also be used to operationalise criteria like information gain [18,19] or marginal likelihood [20,21,22,23]. The "most influential" examples are those that change the DDPM loss (a proxy for the log-likelihood) of the generated sample the most, following [7].
Data pruning with d-TDA. Next, we apply distributional (mean) influence to a data pruning task, where the goal is to remove datapoints from the training set to improve the performance of the final trained model. We consider a SWIN transformer [24] trained on the full CIFAR-10 dataset (see for details). For the baseline, we remove 5000 datapoints deemed to be most influential -that is, estimated to decrease the validation loss the most when ablated -using regular TDA on a single model. For d-TDA , we remove 5000 datapoints estimated to decrease the validation loss the most on average, for 10 models trained using different random seeds. We then compare the final accuracies and losses for individual models trained with those examples ablated, using the same random seeds. Figure 6 shows the results. The distributional variant unlocks accuracy gains c.f. fixed-seed TDA.
this section cite: ['b6', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b6', 'b23']

Section: Evaluating TDA methods.
The observations above also invite us to rethink how we evaluate data attribution methods for stochastic training algorithms: d-TDA methods ought to be effective at identifying examples responsible for large changes in distribution. These changes are often missed when one only looks at the change in mean. Note that the Linear Datamodelling Score (LDS) [12], a common evaluation metric, can already be interpreted as a d-TDA evaluation metric. It measures how accurately attribution methods rank training datapoints by mean influence:
LDS = spearman[(DistInf 𝜽 * (𝒟 ′ 𝑖 )) 𝑀 𝑖=1 ; (DistInf θ * (𝒟 ′ 𝑖 )) 𝑀 𝑖=1 ], (11
)
where spearman denotes the Spearman rank correlation, DistInf 𝜽 * , DistInf θ * are distributional (mean) influence scores computed using exact retraining and a d-TDA method respectively, and 𝒟 ′ 𝑖 are randomly subsampled subsets of the training data. In light of our discussion, it is natural to generalise Eq. ( 11) using other notions of distributional influence. We term such metrics distributional LDS, of which regular LDS is a special case. We show a preliminary benchmark in Figure 10 (App. C.2), showcasing that distributional LDS can better flesh out differences between d-TDA methods. Distributional LDS (e.g. Wasserstein) can be computed at virtually no additional cost over standard LDS, and we argue should become the default for benchmarking data attribution in deep learning.
Leave-one-out is not broken, just noisy. Prior works have reported that TDA methods such as IFs are incapable of accurately predicting the outcome of leave one out (LOO) retraining, often obtaining near 0% correlation to groundtruth measurements after actual retraining [14]. Adopting a distributional perspective, we view this differently. For big datasets, removing a training example 𝑧 𝑘 only leads to a tiny change in distribution 𝜇 𝒟 → 𝜇 𝒟\𝑧 𝑘 . We have seen that IFs allow us to approximately sample from 𝜇 𝒟\𝑧 𝑘 , but we may need many empirical samples to detect such a minor distributional shift empirically. In other words, realworld training is noisy; TDA methods struggle with LOO primarily because of a low signal-to-noise ratio, rather than any fundamental incompatibility. In Figure 8, we verify that common attribution methods are capable of accurately approximating the LOO distribution with enough samples -the means of the predicted and ground-truth distributions correlate extremely well.
this section cite: ['b11', 'b13']

Section: Conclusion
This paper introduced distributional training data attribution (d-TDA): a new paradigm for data attribution when training algorithms are stochastic. To demonstrate its utility, we used d-TDA to more effectively identify training examples whose removal improves test loss and accuracy, and proposed novel ways to evaluate d-TDA methods. Rigorously tackling distributional questions also yielded new mathematical motivations for influence functions for deep learning. There is clear correlation between the means of the true and predicted measurements. See App. C.1 for full details.
this section cite: []

Section: References
Ref_id:b0 Title: Data Shapley: Equitable valuation of data for machine learning Year: (2019)
Ref_id:b1 Title: Understanding black-box predictions via influence functions Year: (2017)
Ref_id:b2 Title: Studying large language model generalization with influence functions Year: (2023)
Ref_id:b3 Title: Influence selection for active learning Year: (2021)
Ref_id:b4 Title: Towards efficient data valuation based on the shapley value Year: (2019)
Ref_id:b5 Title: The influence curve and its role in robust estimation Year: (1974)
Ref_id:b6 Title: Influence Functions for Scalable Data Attribution in Diffusion Models Year: ()
Ref_id:b7 Title: Data Cleansing for Models Trained with SGD Year: ()
Ref_id:b8 Title: Training data attribution via approximate unrolled differentiation Year: (2024)
Ref_id:b9 Title: MAGIC: Near-Optimal Data Attribution for Deep Learning Year: (2025)
Ref_id:b10 Title: If influence functions are the answer, then what is the question? Year: (2022)
Ref_id:b11 Title: Trak: Attributing model behavior at scale Year: (2023)
Ref_id:b12 Title: Forward and Reverse Gradient-Based Hyperparameter Optimization Year: (2017)
Ref_id:b13 Title: Influence functions in deep learning are fragile Year: (2020)
Ref_id:b14 Title: Stochastic approximation and recursive algorithm and applications Year: (1997)
Ref_id:b15 Title: Bayesian learning via stochastic gradient Langevin dynamics Year: (2011)
Ref_id:b16 Title: Stochastic gradient descent as approximate bayesian inference Year: (2017)
Ref_id:b17 Title: On a measure of the information provided by an experiment Year: (1956)
Ref_id:b18 Title: Prediction-oriented Bayesian active learning Year: (2023)
Ref_id:b19 Title: Information theory, inference and learning algorithms Year: (2003)
Ref_id:b20 Title: On the marginal likelihood and cross-validation Year: (2020)
Ref_id:b21 Title: Hyperparameter Optimization through Neural Network Partitioning Year: (2023)
Ref_id:b22 Title: Scalable marginal likelihood estimation for model selection in deep learning Year: (2021)
Ref_id:b23 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b24 Title: Stochastic approximation: a dynamical systems viewpoint Year: (2008)
Ref_id:b25 Title: The Implicit Function Theorem Year: (2003)
Ref_id:b26 Title: A Daleckii-Krein formula for the Frechet derivative of a generalized matrix function Year: (2016)
Ref_id:b27 Title: Integration and differentiation of functions of Hermitian operators and applications to the theory of perturbations Year: (1965)
Ref_id:b28 Title: On the accuracy of influence functions for measuring group effects Year: (2019)
Ref_id:b29 Title: On second-order group influence functions for black-box predictions Year: (2020)
Ref_id:b30 Title: Relatif: Identifying explanatory training samples via relative influence Year: (2020)
Ref_id:b31 Title: Optimizing neural networks with kronecker-factored approximate curvature Year: (2015)
Ref_id:b32 Title: Kronecker-factored approximate curvature for modern neural network architectures Year: (2023)
Ref_id:b33 Title: Fast approximate natural gradient descent in a kronecker factored eigenbasis Year: (2018)
Ref_id:b34 Title: Concrete Compressive Strength Year: (1998)
Ref_id:b35 Title: The MNIST Database of Handwritten Digit Images for Machine Learning Research Year: (2012)
Ref_id:b36 Title: Position: Curvature Matrices Should Be Democratized via Linear Operators Year: ()
Ref_id:b37 Title: Eigenvalue corrected noisy natural gradient Year: (2018)
