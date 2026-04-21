Title: Multitask Learning with Stochastic Interpolants
Abstract: We propose a framework for learning maps between probability distributions that broadly generalizes the time dynamics of flow and diffusion models. To enable this, we generalize stochastic interpolants by replacing the scalar time variable with vectors, matrices, or linear operators, allowing us to bridge probability distributions across multiple dimensional spaces. This approach enables the construction of versatile generative models capable of fulfilling multiple tasks without task-specific training. Our operator-based interpolants not only provide a unifying theoretical perspective for existing generative models but also extend their capabilities. Through numerical experiments, we demonstrate the zero-shot efficacy of our method on conditional generation and inpainting, fine-tuning and posterior sampling, and multiscale modeling, suggesting its potential as a generic task-agnostic alternative to specialized models.

Section: Introduction
Recent years have witnessed remarkable advances in generative modeling, with transport-based approaches such as normalizing flows (Lipman et al., 2022;Albergo and Vanden-Eijnden, 2022;Liu et al., 2022) and diffusion models (Ho et al., 2020;Song and Ermon, 2020;De Bortoli et al., 2021;Albergo et al., 2023a) emerging as state-of-the-art techniques across various application domains (Rombach et al., 2022;Mazé and Ahmed, 2023;Alverson et al., 2024). These methods have revolutionized our ability to generate high-quality images, text, and other complex data types by viewing these data as samples from an unknown target distribution and learning to transform simple (e.g., noise) distributions into this target. This transformation is effectively achieved via transport of the samples by a flow or diffusion process with a drift (or score) parameterized by neural networks and estimated via simulation-free quadratic regression, enabling highly efficient training.
Despite their impressive performance, these generative frameworks face a fundamental limitation: they are typically designed and trained for specific, predetermined tasks, with the generative objective specified before training. For example, a diffusion model trained to generate images cannot easily be repurposed to perform inpainting or other editing tasks without substantial modification or retraining.
While some flexibility can be achieved through conditioning variables or prompting, these approaches remain constrained within narrowly defined operational boundaries established beforehand. Recent attempts at multitask generation using approximated guidance strategies (Chung et al., 2023;Song et al., 2022;Wang et al., 2024) have shown promise, but rely on uncontrolled approximations that limit their theoretical guarantees and can lead to unpredictable results. These methods typically operate within a predefined space of capabilities and lack the flexibility to adapt to novel tasks without retraining, often requiring domain-specific architectural modifications or specialized training procedures that further limit their versatility.
In this paper, we introduce a novel framework for training truly multi-task generative models based on a generalized formulation of stochastic interpolants. Our key insight is to replace the scalar time variable traditionally used in transport-based models with a linear operator. These operator-based interpolants enable interpolation between random variables across multiple dimensional planes or setups, providing a unified mathematical formulation that treats various generative tasks as different ways of traversing the same underlying space, rather than as separate problems requiring distinct models. This dramatically expands the space of possible tasks that a single model can perform.
Our main contributions include theoretical advances that establish a framework for multiple generative applications:
• We extend traditional scalar interpolation in dynamical generative models to higher-dimensional structures, developing a unified mathematical formulation of operator-based interpolants that treats various generative tasks as different ways of traversing the same underlying space. This opens up fundamentally new ways of seeing how generative models can be structured to handle multiple objectives simultaneously.
• We show how this framework enables generative models with continual self-supervision over a wide purview of generative tasks, making possible: universal inpainting models that work with arbitrary masks, multichannel data denoisers with operators in the Fourier domain, posterior sampling with quadratic rewards, and test-time dynamical optimization with rewards and interactive user feedback, all with one pretrained model.
• We demonstrate these various tools on image-infilling, data de-corruption, statistical physics simulation, and dynamical robotic planning tasks across a number of datasets, showing that our method matches or surpasses existing approaches without being specifically tied to any single generative objective.
All common augmentations like text-conditioning and guidance can still be used just as before in our setup. While our approach increases the complexity of the initial learning problem, we provide arguments that this additional complexity can be addressed through scale. In essence, the "pretraining" phase becomes more challenging, but the resulting model gains substantial flexibility and versatility that compensates for the pretraining costs. That is, our approach can be seen as a way of amortizing learning over a variety of tasks. This points toward a more general paradigm of universal generative models that can be trained once and then applied to a variety of objectives, potentially reducing computational and environmental costs associated with training separate models for each task.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b5', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12']

Section: Related works
Flow matching and diffusion models. Our approach extends the theoretical groundwork established in flow matching with stochastic interpolant and rectified flows (Lipman et al., 2022;Albergo and Vanden-Eijnden, 2022;Albergo et al., 2023a;Liu et al., 2022) as well as the probability flow formulations in diffusion models (Ho et al., 2020;Song et al., 2020). which established techniques for handling multiple target distributions simultaneously. Unlike data-dependent coupling approaches (Albergo et al., 2023b) that require constructing specific couplings for tasks like inpainting, our method learns a general operator space that naturally accommodates such tasks without additional coupling design. By introducing operator-valued interpolants, we enable a richer space of transformations between distributions, unlocking a flexible framework for multiple generative tasks.
Inverse problems and inpainting Our framework offers a unified approach to inverse problems, contrasting with traditional methods that require problem-specific variational optimization procedures Pereyra et al. (2015).  2022) remain fundamentally task-specific. Our approach encodes solution paths within the interpolant operator structure itself, enabling multiple inverse problems to be addressed through appropriate operator path selection during inference-without additional training. Our work can also be seen as a way to formalize methods that seek to clean data corrupted in various ways Bansal et al. (2022) Multiscale and any-order generation Recent approaches to generative modeling have explored hierarchical strategies through progressive refinement. Visual Autoregressive Modeling (Tian et al., 2024) employs next-scale prediction, while Fractal Generative Models (Li et al., 2025) utilize selfsimilar structures for multiscale representations. These methods typically constrain generation to fixed paths established during training. In contrast, our framework decouples the training process from generation trajectories, allowing flexible path selection at inference time. This is conceptually related to optimizing generation order in discrete diffusion (Shi et al., 2025) and token ordering studies (Kim et al., 2025), but provides greater flexibility by enabling post-training optimization and dynamic, self-guided generation strategies that adapt based on intermediate results.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b15', 'b23', 'b24', 'b25', 'b26', 'b27']

Section: Theoretical framework
Imagine we want to create a single generative model capable of multiple tasks -sampling new data, inpainting, denoising, and more. To achieve this, we need to expand beyond the traditional "single path" between noise and data. In this section, we develop the theoretical foundations of operator-based interpolants, which allow flexible navigation through a richer multidimensional space and will enable multitask generative capabilities discussed in Section 3.
this section cite: []

Section: Operator-based interpolants
Suppose that we are given a couple of random variables (x 0 , x 1 ) both taking values in a Hilbert space H (for example R d ) and drawn from a joint distribution µ(dx 0 , dx 1 ). Our aim is to design a transport between a broad class of distributions supported on H involving mixtures of x 0 and x 1 . We will do so by generalizing the framework of stochastic interpolant.
Definition 2.1 (Operator-based interpolants). Let B(H) be a connected set of bounded linear operators on H and let S ⊆ B(H) × B(H), also connected. Given any pair of linear operators (α, β) ∈ S, the operator-based interpolant I(α, β) is the stochastic process given by
I(α, β) = αx 0 + βx 1 , (α, β) ∈ S,(1)
where (x 0 , x 1 ) ∼ µ. We will denote by µ α,β the probability distribution of I(α, β).
If we picked e.g. α = (1 -t) Id and β = t Id with t ∈ [0, 1], respectively, we would go back to a standard stochastic interpolant, but we stress that the operator-based interpolants from Definition 2.1 are much more general objects. For example, if H = R d , we could take B(H) to be a set of d × d matrices with real entries -choices of α, β tailored to several multitask generation will be discussed in Section 3. The main objective of this paper is to show how to exploit this flexibility of design. Specifically we will show that we can learn a model that can be used to transport samples of I(α, β) along any paths of (α, β) in S without having to choose any such path during training. This will mean that transport problems in a broad class associated to a variety of tasks, including inpainting and block generation of any order, fine-tuning, etc. will be pretrained into this model.
this section cite: []

Section: Multipurpose drifts and score
To proceed, we introduce two drifts, which are functions of S × H taking value in H: Definition 2.2 (Multipurpose drift). The drifts η 0 , η 1 : S × H → H are given by
η 0 (α, β, x) = E[x 0 |I(α, β) = x], η 1 (α, β, x) = E[x 1 |I(α, β) = x],(2)
where E[ • |I(α, β) = x] denotes expectation over the coupling (x 0 , x 1 ) ∼ µ conditioned on the event I(α, β) = x.
Using the L 2 characterization of the conditional expectation, these drifts can be estimated via solution of a tractable optimization problem with an objective function involving an expectation: Lemma 2.3 (Drift objective). Let ν(dα, dβ) be a probability distribution whose support is S. Then the drifts η 0,1 (α, β, x) in Definition 2.2 can be characterized globally for all (α, β) ∈ S and all x ∈ supp(µ α,β ) via solution of the optimization problems
η 0 = argmin η0 E (α,β)∼ν (x0,x1)∼µ ∥η 0 (α, β, I(α, β)) -x 0 ∥ 2 ,(3)
η 1 = argmin η1 E (α,β)∼ν (x0,x1)∼µ ∥η 1 (α, β, I(α, β)) -x 1 ∥ 2 ,(4)
where ∥ • ∥ denotes the norm in H.
This lemma is proven in Appendix A. Below we will use (3) and (4) to learn η 0,1 over a rich parametric class made of deep neural networks. Note that the drifts η 0 and η 1 are not linearly independent since the definition of I(α, β) in ( 9) together with the equality x = E[I(α, β)|I(α, β) = x] imply that x = αη 0 (α, β, x) + βη 1 (α, β, x).
(5) Therefore we can obtain η 0 from η 1 if α is invertible and η 1 from η 0 if β is invertible. Note also that, when x 0 is Gaussian and x 0 ⊥ x 1 , η 0 is related to the score of the distribution of the stochastic interpolant: Lemma 2.4 (Score). Assume that H = R d and that the probability distribution µ α,β of the stochastic interpolant I(α, β) is absolutely continuous with respect to the Lebesgue measure with density ρ α,β (x). Assume also that x 0 ∼ N (0, Id) and
x 0 ⊥ x 1 . Then the score s α,β (x) = ∇ log ρ α,β (x) is related to the drift η 0 (α, β, x) via η 0 (α, β, x) = -αs α,β (x).(6)
This lemma is proven in Appendix A.
this section cite: []

Section: Transport with flows and diffusions
We can now state the main theoretical results of this paper: if we are able to sample the stochastic interpolant I(α 0 , β 0 ) at a specific value (α 0 , β 0 ) ∈ S, then we can produce sample of I(α t , β t ) along any curve (α t , β t ) ∈ S by solving either a probability flow ODE or an SDE, assuming we have estimated the drifts η 0,1 (α, β, x) from Definition 2.2 along this curve: Proposition 2.5 (Probability flow). Let (α t , β t ) t∈[0,1] be any one-parameter family of operators (α t , β t ) ∈ S. Assume that α t , β t are differentiable for all t ∈ [0, 1]. Then, for all t ∈ [0, 1], the law of I(α t , β t ) is the same as the law of the solution X t to
Ẋt = αt η 0 (α t , β t , X t ) + βt η 1 (α t , β t , X t ), X 0 d = I(α 0 , β 0 ).(7)
This proposition is proven in Appendix A. Similarly, for generation with an SDE, we have: Proposition 2.6 (Diffusion). Assume that H = R d and the probability distribution µ α,β of the stochastic interpolant I(α, β) is absolutely continuous with respect to the Lebesgue measure. Assume also that, in I(α, β), x 0 is Gaussian and x 0 ⊥ x 1 . Then, under the same conditions as in Proposition 2.5, if α t is invertible, for all t ∈ [0, 1] and any ϵ t ⩾ 0, the law of I(α t , β t ) is the same as the law of the solution X ϵ t to
dX ϵ t = αt -ϵ t α -1 t η 0 (α t , β t , X ϵ t )dt + βt η 1 (α t , β t , X ϵ t )dt + √ 2ϵ t dW t , X ϵ 0 d = I(α 0 , β 0 ). (8)
where W t is a Wiener process in R d .
this section cite: []

Section: This proposition is also proven in
Appendix A. Note that the SDE (8) reduces to the ODE (7) if we set ϵ t = 0. Note that if α t is positive-definite we can use εt = ϵ t α -1 t as new diffusion coefficient, which set the noise term in (8) to √ 2ϵ t α 1/2 t dW t ; this allows to extend this SDE to paths along which we can have α t = 0. May 15, 2025 1 AR Morton In-paint
this section cite: []

Section: Multitask generation
In this section we discuss how to use the theoretical framework introduced in Section 2 to perform multiple generative tasks without having to doing any retraining.
this section cite: []

Section: Self-supervised generation and inpainting
Inpainting-the task of filling in missing parts of an image-traditionally requires specialized training for each possible mask configuration. Our operator-based framework enables a fundamentally different approach: a single model that can perform inpainting with any arbitrary mask, chosen at inference time, or can generate samples from scratch in an arbitrary ordering of the generation. This includes standard generation of all dimensions at once, autoregressive generation dimension by dimension, blockwise fractal generation, and so forth. In particular, we may want to construct a generative model that fills in missing entries from a sample x 1 ∈ R d drawn from a data distribution µ 1 . We would like this model to be universal, in the sense that it can be used regardless of which entries are missing; their number and position can be arbitrary and changed post-training, allowing for flexible inpainting and editing (see Figure 1 for an illustration). This approach creates a natural self-supervision mechanism, as the model continuously tracks which parts have been generated and which remain to be filled.
To perform this task, assume that x 1 is drawn from the data distribution µ 1 of interest and x 0 drawn independently from N (0, Id), so that µ = N (0, Id) × µ 1 , set β = 1 -α in the operator interpolant (1), and assume that α is a diagonal matrix. With a slight abuse of notations we can then identify the diagonal elements of the matrix α with a vector α ∈ R d and write (1) as
I(α) = α ⊙ x 0 + (1 -α) ⊙ x 1 ,(9)
where ⊙ denotes the Hadamard (i.e. entrywise) product. The drift to learn in this case is
η(α, x) = E x 0 -x 1 |α ⊙ x 0 + (1 -α) ⊙ x 1 = x (10
)
for α ∈ [0, 1] d , and this learning can be done via solution of
min η E α∼U ([0,1] d ) x0∼N (0,Id) x1∼µ1 ∥η(α, I(α ⊙ x 0 + (1 -α) ⊙ x 1 ) + x 1 -x 0 )∥ 2 , (11
) Denoting x 1 = (x 1 1 , x 2 1 , . . . , x d 1 )
, suppose that we observe x i 1 for the entries with i ∈ σ ⊂ {1, . . . , d} and would like to infer the missing entries with i ∈ σ c = {1, . . . , d} \ σ. To perform this inpainting we can use the probability flow ODE (7) with a path α t such that α i t = 0 if i ∈ σ and α i t = 1 -t if i ∈ σ c . Note that this can be done for any choice of σ without retraining.
this section cite: []

Section: Multichannel denoising
Suppose that B 1 , B 2 , . . . B n are deterministic corruption operators that can be applied to the data. For example B 1 could be a high-pass filter, B 2 a motion blur, etc. and they could be defined primarily in the Fourier representation of the data. Similarly, if x 0 ∼ N (0, Id), let A 1 , A 2 , . . . A m be operators that give some structure to this noise (e.g. some spatial correlation over the domain of the data). Set A 0 = B 0 = Id and take
α = m i=0 a i A i , β = n i=0 b i B i (12
)
where a i , b i are nonegative scalars each taking values in some range that includes 0 and b 0 = 1. With this choice we can find path (α t , β t ) t∈[0,1] that bridges data x 1 ∼ µ corrupted in any possible channel as
m i=1 a i A i x 0 + n i=1 b i B i x 1 for some choice of (a 1 , a 2 , . . . , b 1 , b 2 , . . .)
back to the clean data via a path that bridges this choice of parameters to b 0 = 1 and a 0 = a 1 . . . = b 1 = . . . = 0. See Figure 2 for an illustration of two possible corruption schemes.
this section cite: []

Section: Fine-tuning and posterior sampling
Suppose that we are given data from a distribution µ 1 (dx), and that we would like to generate samples from µ r 1 (dx) = Z -1 e r(x) µ 1 (dx) where r : H → R is a reward function and Z = H e r(x) µ 1 (dx) is a normalization function, which is unknown to us but we assume finite -in the context of Bayesian inference, µ 1 plays the role of prior distribution, r is the likelihood, and µ r 1 is the posterior distribution. We will assume that the reward r is a quadratic function, i.e.
r(x) = 1 2 ⟨x, Ax⟩ + ⟨b, x⟩ (13
)
where A is a definite negative bilinear operator on H, b ∈ H, and ⟨•, •⟩ denotes the inner product on H. For simplicity we also assume that H = R d : the general case can be treated similarly.
In this context, assume that we have learned the drifts η 0,1 associated with the interpolant
I(α, β) = αx 0 + βx 1 , x 0 ∼ N (0, Id), x 1 ∼ µ 1 , x 0 ⊥ x 1 (14
)
so that we can generate samples from the prior distribution. Our next result shows that this gives us access to the drifts η r 0,1 associated with the interpolant
I r (α, β) = αx 0 + βx r 1 , x 0 ∼ N (0, Id), x r 1 ∼ µ r 1 , x 0 ⊥ x r 1 (15
) involving data x r 1 from the posterior distribution. Proposition 3.1. Let η 0 (α, β, x) = E x 0 |I(α, β) = x , η 1 (α, β, x) = E x 1 |I(α, β) = x ,(16)
be the drifts associated with the interpolant (14) and
η r 0 (α, β, x) = E x 0 |I r (α, β) = x , η r 1 (α, β, x) = E x r 1 |I r (α, β) = x ,(17)
be the drifts associated with the interpolant (15). If α and β are invertible, then
η r 0 (α, β, x) = α -1 ββ -1 r α r η 0 (α r , β r , x r ) + α -1 (x -ββ -1 r x r ) (18) η r 1 (α, β, x) = η 1 (α r , β r , x r ) (19
)
as long as we can find a pair (α r , β r ) that satisfies
β T r α -T r α -1 r β r = β T α -T α -1 β -A(20)
and x r is given by
x r = α r α T r β -T r β T α -T α -1 x + b .(21)
This proposition is proven in Appendix A as a corollary of Proposition A.1 that relates the probability distribution of I r to that of I. Proposition 3.1 offers a way to sample the posterior distribution without retraining, by using the drifts ( 18) and ( 19) in the ODE (7) or the SDE (8).
this section cite: []

Section: Inference adaption
Suppose that we have learned the drifts η 0,1 in Definition 2.2 and wish to transport samples along a path (α t , β t ) with fixed end points. We can leverage the flexibility of our formulation to perform inference adaptation, that is, optimize the path (α t , β t ) used during generation to achieve specific objectives, such as minimizing computational cost, maximizing sample quality, or satisfying user constraints. This can be done in two ways: (1) offline optimization, where we pre-compute optimal paths for different scenarios using objectives like Wasserstein length minimization, and (2) online adaptation, where paths are dynamically adjusted during generation based on intermediate results or user feedback.
In the case of offline optimization, we could for example optimize the Wasserstein length of the path.
That is, if we want to bridge the distributions µ α0,β0 of I(α 0 , β 0 ) and µ α1,β1 of I(α 1 , β 1 ) via µ αt,βt with (α t , β t ) ∈ S for all t ∈ [0, 1] then the path that minimizes the Wasserstein length of the bridge distribution µ αt,βt solves
min (αt,βt)t 1 0 E ∥ αt η 0 (α t , β t , I(α t , β t )) + βt η 1 (α t , β t , I(α t , β t ))∥ 2 dt (22
)
where the minimization is performed over paths (α t , β t ) t ≡ (α t , β t ) t∈[0,1] such that (α t , β t ) ∈ S for all t ∈ [0, 1] with their end points (α 0 , β 0 ) and (α 1 , β 1 ) prescribed and fixed.
Algorithm 1: Multitask learner input: Samples (x 0 , x 1 ) ∼ µ; choice of distribution ν(dα, dβ) and associated sampler. repeat Draw batch (
x i 0 , x 1 i , α i , β i ) M i=1 ∼ µ × ν. Compute I i = α i x i 0 + β i x i 1 . Compute L = 1 M M i=1 ∥η 0 (α i , β i , I i ) -x i 0 ∥ 2 + ∥η 1 (α i , β i , I i ) -x i 1 ∥ 2
. Take a gradient step on L to update η0 and η1 . until converged; output: Drifts η0 and η1 .
Algorithm 2: Multitask generator input: Drifts η0 , η1 ; choice of path (α t β t ) t∈[0,1] tailored to the generation task; data
I(α 0 , β 0 ) = α 0 x 0 + β 0 x 1 ; diffusion coefficient ϵ t ⩾ 0; time step h = 1/K with K ∈ N. initialize: Xϵ 0 = I(α 0 , β 0 ); for k = 0, . . . , K -1 do set ηk 0 = η0 (α kh , β kh , Xϵ kh ), ηk 1 = η1 (α kk , β kh , Xϵ kh ), and z k ∼ N (0, Id) update Xϵ k+1k = Xϵ kh + h αkk -ϵ kh α -1 kk ηk 0 + h βkh ηk 1 + √ 2ϵ kh h z k , end output: Xϵ 1 d = I(α 1 , β 1 ) (approximately)
this section cite: []

Section: Algorithmic aspects
The algorithmic aspects of our framework can be summarized in a few key steps. First, we define a connected set S of (α, β) such that the ensemble of different tasks we will want to perform correspond to getting samples of I(α, β) at some value of (α 0 , β 0 ) ∈ S and generating from them new data at another value of (α 1 , β 1 ) ∈ S. Second, we specify a measure ν on S for the learning of the drifts η 0 and η 1 defined in (2). Third, we learn these drifts via minimization of the objectives in (3) and (4), using the procedure outlined in Algorithm 1. Note that we can possibly simplify this algorithm, learning only one of the two drifts and obtaining the other through the relation (5). Finally, given any pairs (α 0 , β 0 ), (α 1 , β 1 ) ∈ S, we use a path (α t , β t ) t∈[0,1] with α t , β t ∈ S for all t ∈ [0, 1] and integrate the SDE (8) (or possibly the ODE (7) if we set ϵ t = 0) to perform the generation, as outlined in Algorithm 2. Note that this path could also be adapted on-the-fly during inference, using some feedback about the solution of the SDE.
this section cite: []

Section: Numerical experiments
Below we provide numerical realization of some of the various objectives that can be fulfilled with the multitask objective. Details of the experimental setup can be found in Appendix B.
this section cite: []

Section: Multitask inpainting and sequential generation
We evaluate our method on three datasets: MNIST, with images of size 28 × 28, CelebA, resized to 128 × 128, and of Animal FacesHQ focused on cat class, with images resized to 256 × 256. Details of the experimental setup are standard and can be found in Appendix B. In these experiments, we use the Hadamard interpolant (9).
this section cite: []

Section: MNIST.
We demonstrate the versatility of our operator-based interpolant framework through inpainting and sequential generation tasks on MNIST. The results are shown Figure 3 where all the generated images come from the same model without any retraining.
For inpainting (left panels), we replace masked regions with Gaussian noise (shown as pink for clarity), then generate only these regions while preserving unmasked pixels. This is achieved by setting the entries of α to 1 -t for masked pixels and 0 for unmasked ones. To preserve unmasked pixels, we apply a secondary mask setting η(α, x) to zero at these positions. Sequential generation (right panels) reformulates image creation as progressive inpainting. Starting with pure Gaussian noise, we generate the image block-by-block by successively updating the operator masks. Unlike single-pass inpainting, this requires multiple forward passes-one per block. For each pass, we apply α = 1 -t only to pixels in the current generation block, maintaining appropriate values for previously generated and remaining noise regions.
CelebA and AFHQ-Cat. We present benchmark results for all methods across various image restoration tasks, evaluating the average peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) on 100 test images from each dataset: AFHQ-Cat (256 × 256) and CelebA (128 × 128). To assess the performance of our methodology, we employed two types of masking: square masks of sizes 40 × 40 and 80 × 80 with added Gaussian noise of standard deviation 0.05, and random masks covering 70% of image pixels with Gaussian noise of standard deviation 0.01. We benchmark our method against four state- As shown in Table 1, our method consistently ranks either first or second in both reconstruction metrics across all tasks and datasets (with all values except the last row taken from Martin et al. (2025)). Regarding visual quality (Fig. 4), our method generates realistic, artifact-free images, albeit with slight over-smoothing at times.
this section cite: ['b17']

Section: Posterior sampling in the ϕ 4 -model
We apply our approach in the context of the ϕ 4 model in d = 2 spacetime dimensions, a statistical lattice field theory where field configurations ϕ ∈ R L×L represent the lattice state (L denotes spatiotemporal extent)-for details see Appendix B.2. This model poses sampling challenges due to its phase transition from disorder to full order, during which neighboring sites develop strong correlations in sign and magnitude Vierhaus (2010); Albergo et al. (2019).  The ϕ 4 model is specified by the following probability distribution
µ(dϕ) = Z -1 e -E(ϕ) dϕ (23
)
where ϕ) dϕ is a normalization constant and E(ϕ) is an energy function defined as
Z = R L×L e -E(
E(ϕ) = 1 2 χ a∼b |ϕ(a) -ϕ(b)| 2 + 1 2 κ a |ϕ(a)| 2 + 1 4 γ a |ϕ(a)| 4 , (24
)
where a, b ∈ [0, . . . , L -1] 2 denote the discrete positions on a 2-dimensional lattice of size L × L, a ∼ b denotes neighboring sites on the lattice, and we assume periodic boundary conditions; χ > 0 , κ ∈ R and γ > 0 are parameters. We perform MCMC simulations to generate configuration in a parameter range close to the phase transition. We use these data to learn a stochastic interpolant of the form ( 9 May 15, 2025 1 Prior ( ) h = 0 Posterior ( ) h = 0.02 True (MCMC) Generated (interpolant) Histogram of magnetization M(φ) = (1/N ) ∑ a φ(a) Figure 5: Simulating a lattice ϕ 4 theory. Top left: L = 32 × L = 32 lattice configurations at the phase transition. Bottom left: lattice examples with drift parameter h = 0.02. Top middle: Generated lattice examples at phase transition. Bottom middle: generated lattice examples with field h = 0.02. Right: magnetization of 2000 lattice configurations.
The additional term plays the role of a reward. The results of the generation based on Proposition 3.1 are shown on Figure 5, which indicate that our approach permits to valid sample configurations (as verified by their magnetization) of the posterior without retraining.
this section cite: ['b31', 'b32']

Section: Planning and decision making in a maze
This section applies our framework to shortest path planning in maze environments, drawing from Jan For training, we use paths of length 300 randomly extracted from the trajectory of length 2,000,000 from Chen et al. (2024). For simplicity, we subsample these paths every six points, creating sparse paths of length 50, from which we can recover paths of length 300 through linear interpolation between consecutive points. At inference, we perform zero-shot generation between any two points in the maze by enforcing that the trajectory passes through these points: the length of the path between these locations can be varied by pinning the first point by setting α 1 = 0, and the second point by setting α i = 0 with a value of i ∈ [2, 50] that can be adjusted (see Appendix B.3 for details). Typical results are shown in Fig. 6. In terms of quality assessment, we check that the generated trajectories remain within allowed maze regions: all the 10,000 paths we generated between randomly chosen point pairs avoided the forbidden areas, demonstrating robust performance. More numerical experiments in Appendix B.3 demonstrate that with a similar strategy, one can impose the pathway to take detour at will, even if it implies generating a longer path. Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. URL https:  //arxiv.org/abs/2407.01392. David Williams. Probability with Martingales. Cambridge Mathematical Textbooks. Cambridge University Press, Cambridge, 1991. ISBN 9780521406055. doi: 10.1017/CBO9780511813658. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/1412.6980. Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. CoRR, abs/1505.04597, 2015. URL http://arxiv.org/abs/1505.  04597.
this section cite: []

Section: A Proofs
Definition 2.2 (Multipurpose drift). The drifts η 0 , η 1 : S × H → H are given by
η 0 (α, β, x) = E[x 0 |I(α, β) = x], η 1 (α, β, x) = E[x 1 |I(α, β) = x],(2)
where E[ • |I(α, β) = x] denotes expectation over the coupling (x 0 , x 1 ) ∼ µ conditioned on the event I(α, β) = x. Lemma 2.3 (Drift objective). Let ν(dα, dβ) be a probability distribution whose support is S. Then the drifts η 0,1 (α, β, x) in Definition 2.2 can be characterized globally for all (α, β) ∈ S and all x ∈ supp(µ α,β ) via solution of the optimization problems
η 0 = argmin η0 E (α,β)∼ν (x0,x1)∼µ ∥η 0 (α, β, I(α, β)) -x 0 ∥ 2 ,(3)
η 1 = argmin η1 E (α,β)∼ν (x0,x1)∼µ ∥η 1 (α, β, I(α, β)) -x 1 ∥ 2 ,(4)
where ∥ • ∥ denotes the norm in H.
Proof. The lemma is a simple consequence of the L 2 characterization of the conditional expectation as least-squares-best predictor, see e.g. Section 9.3 in Williams (1991).
Lemma 2.4 (Score). Assume that H = R d and that the probability distribution µ α,β of the stochastic interpolant I(α, β) is absolutely continuous with respect to the Lebesgue measure with density ρ α,β (x). Assume also that x 0 ∼ N (0, Id) and
x 0 ⊥ x 1 . Then the score s α,β (x) = ∇ log ρ α,β (x) is related to the drift η 0 (α, β, x) via η 0 (α, β, x) = -αs α,β (x).(6)
Proof. The lemma follows from Stein's lemma (aka Gaussian integration by parts formula) that asserts that
E[x 0 |I(α, β, x) = x] = -αs α,β (x)(26)
as well as the definition of η 0 (α, β, x) in ( 16).
Proposition 2.5 (Probability flow). Let (α t , β t ) t∈[0,1] be any one-parameter family of operators (α t , β t ) ∈ S. Assume that α t , β t are differentiable for all t ∈ [0, 1]. Then, for all t ∈ [0, 1], the law of I(α t , β t ) is the same as the law of the solution X t to
Ẋt = αt η 0 (α t , β t , X t ) + βt η 1 (α t , β t , X t ), X 0 d = I(α 0 , β 0 ).(7)
Proof. From the framework of standard stochastic interpolants Albergo and Vanden-Eijnden (2022); Albergo et al. (2023a), we know that the law of I t = I(α t , β t ) is the same for all t ∈ [0, 1] as the law of X t , i.e. the solution to the probability flow ODE
Ẋt = b t (X t ), X t=0 d = I t=0 , (27
) where b t (x) = E[ İt |I t = x].(28)
By the chain rule İt = αt
x 0 + βt x 1 so that b t (x) = αt E[x 0 |I t = x] + αt E[x 1 |I t = x] ≡ αt η 0 (α t , β t , x) + βt η 1 (α t , β t , x).(29)
where η 0,1 are the drifts defined in ( 16). This means that ( 27) is ( 7).
Proposition 2.6 (Diffusion). Assume that H = R d and the probability distribution µ α,β of the stochastic interpolant I(α, β) is absolutely continuous with respect to the Lebesgue measure. Assume also that, in I(α, β), x 0 is Gaussian and x 0 ⊥ x 1 . Then, under the same conditions as in Proposition 2.5, if α t is invertible, for all t ∈ [0, 1] and any ϵ t ⩾ 0, the law of I(α t , β t ) is the same as the law of the solution X ϵ t to
dX ϵ t = αt -ϵ t α -1 t η 0 (α t , β t , X ϵ t )dt + βt η 1 (α t , β t , X ϵ t )dt + √ 2ϵ t dW t , X ϵ 0 d = I(α 0 , β 0 ). (8)
where W t is a Wiener process in R d .
Proof. From the framework of standard stochastic interpolants Albergo and Vanden-Eijnden (2022); Albergo et al. (2023a), we know that the law of the solution X t to the probability flow ODE ( 27) is the same for all t ∈ [0, 1] as the law of X ϵ t solution to to SDE
dX ϵ t = b t (X t )dt + ϵ t s t (X ϵ t )dt + √ 2ϵ t dW t , X ϵ t=0 , d = I t=0 ,(30)
where s t (x) is the score of the probability density function of I t = I(α t , β t ). Since s t (x) = s αt,βt (x), from Lemma 2.4, we have
s t (x) = -α -1 t η 0 (α t , β t , x).(31)
If we insert this expression in (30) and use (29), we see that this SDE reduces to ( 8).
Proposition 3.1. Let
η 0 (α, β, x) = E x 0 |I(α, β) = x , η 1 (α, β, x) = E x 1 |I(α, β) = x ,(16)
be the drifts associated with the interpolant (14) and
η r 0 (α, β, x) = E x 0 |I r (α, β) = x , η r 1 (α, β, x) = E x r 1 |I r (α, β) = x ,(17)
be the drifts associated with the interpolant (15). If α and β are invertible, then
η r 0 (α, β, x) = α -1 ββ -1 r α r η 0 (α r , β r , x r ) + α -1 (x -ββ -1 r x r ) (18) η r 1 (α, β, x) = η 1 (α r , β r , x r )(19)
as long as we can find a pair (α r , β r ) that satisfies
β T r α -T r α -1 r β r = β T α -T α -1 β -A(20)
and x r is given by
x r = α r α T r β -T r β T α -T α -1 x + b .(21)
We will prove this proposition as a corollary of: Proposition A.1 (Posterior distributions). Let µ α,β and µ r α,β be the probability distributions of the stochastic interpolants defined in (14) and (15), respectively. Assume that α and β are invertible, that the equations (20) for α r , β r in Proposition 3.1 have a solution, and that x r is given by (21). Then these distributions are related, up to a constant independent of x and x r , as
µ r α,β (dx) = |α r ||α| -1 e R(α,β,x) µ αr,βr (dx r ),(32)
where
R(α, β, x) = 1 2 |α -1 r x r | 2 -1 2 |α -1 x| 2 .(33)
Proof. By definition of the probability distribution µ r α,β (dx) of I r (α, β), given any integrable and bounded test function ϕ : R d → R we have
R d ϕ(x)µ r α,β (dx) = E[ϕ(I r (α, β))] = R d ×R d ϕ(αx 0 + βx 1 )(2π) -d/2 e -1 2 |x0| 2 dx 0 e r(x1) µ 1 (dx 1 )(34)
If instead of x 0 we use as new integration variable x = αx 0 + βx 1 , this becomes
R d ϕ(x)µ r α,β (dx) = |α| -1 R d ×R d ϕ(x)(2π) -d/2 e -1 2 |α -1 (x-βx1)| 2 dxe r(x1) µ 1 (dx 1 ).(35)
Similarly, for the probability distribution µ α,β (dx) of I(α, β), we have
R d ϕ(x)µ α,β (dx) = |α| -1 R d ×R d ϕ(x)(2π) -d/2 e -1 2 |α -1 (x-βx1)| 2 dxµ 1 (dx 1 )(36)
If in this equation we replace α by α r , β by β r , x by x r , and ϕ(x) by ϕ(x)e R(α,β,x) , and multiply both side by |α r |/|α| it becomes:
|α r ||α| -1 R d ϕ(x)e R(α,β,x) µ αr,βr (dx r ) = |α| -1 R d ×R d ϕ(x)e R(α,β,x) (2π) -d/2 e -1 2 |α -1 r (x-βrx1)| 2 dx r µ 1 (dx 1 ).(37)
We can now require that the right hand side of (37) be the same as at the right hand-side of (35 α,β,x) µ αr,βr (dx r )), we arrive at the requirement that
) (so that, µ r α,β (dx) = |α r ||α| -1 e R(
- 1 2 |α -1 r (x r -β r x 1 )| 2 + R(α, β, x) = - 1 2 |α -1 (x -βx 1 )| 2 + 1 2 ⟨x 1 , Ax 1 ⟩ + ⟨b, x 1 ⟩,(38)
where we used r(x) = 1 2 ⟨x, Ax⟩ + ⟨b, x⟩. Since (38) must hold for all x 1 , we can expand both sides of this equation, and equate the coefficient of order 2, 1 and 0 in x 1 . They are completely equivalent to ( 20), (21), and (33), respectively. So as long as we can find solutions to ( 20), (32) holds.
Proof of Proposition 3.1. By definition of the conditional expectation, we have
η 1 (α, β, x) = R d x 1 e -1 2 |α -1 (x-βx1)| 2 µ 1 (dx 1 ) R d e -1 2 |α -1 (x-βx1)| 2 µ 1 (dx 1 ) ,(39)
η r 1 (α, β, x) = R d x 1 e -1 2 |α -1 (x-βx1)| 2 e r(x1) µ 1 (dx 1 ) R d e -1 2 |α -1 (x-βx1)| 2 e r(x1) µ 1 (dx 1 ) .(40)
If in the first equality we replace α by α r , β by β r , and x by x r , and assume that α r , β r , and x r satisfy (38), by construction we obtain that (19) holds. To get (18), use (19) as well as relation ( 5) twice to deduce
η r 0 (α, β, x) = α -1 x -βη r 1 (x, α, β) = α -1 x -βη 1 (x r , α r , β r ) = α -1 x -ββ -1 r (x r -α r η 0 (x r , α r , β r ) = α -1 ββ -1 r α r η 0 (x r , α r , β r ) + α -1 x -ββ -1 r x r .(41)
this section cite: ['b1', 'b1']

Section: B Experimental details
Details for the experiments in Section 5 are provided here.
this section cite: []

Section: B.1 Multitask inpainting and sequential generation
For all image generation experiments, the U-Net architecture originally proposed in Ho et al. (2020) is used. The specification of architecture hyperparameters as well as training hyperparameters are given in Table 2. Training was done for 200 epochs on batches comprised of 30 draws from the target, and 50 time slices. The objectives given in 3 and 4 were optimized using the Adam optimizer. The learning rate was set to .0001 and was dropped by a factor of 2 every 1500 iterations of training. To integrate the ODE/SDE when drawing samples, we used a simple Euler integrator.
In order to progressively explore the space of the hypercube of α and β, we first learn a model in the diagonal of the hypercube, i.e where all entries of α are all the same value. We then fine-tune the first model for matrices α t uniformly distributed in [0, 1] d . We also fine-tune the first model for matrices α t decomposed by blocks of 4 × 4 where entries of each blocks contains the same values in [0, 1] d .
this section cite: ['b3']

Section: B.2 Details about the ϕ 4 Model
We define the discrete Fourier transform as
φ(k) = L -d/2 a e 2iπk•a/L ϕ(a) ⇔ ϕ(a) = L -d/2 k e -2iπk•a/L φ(k),(42)
where a, k ∈ [0, . . . , L -1] d , we can write the energy ( 24) as E(ϕ) = E 0 (ϕ) + U (ϕ) with
E 0 (ϕ) = Ê0 ( φ) ≡ 1 2 k M (k)| φ(k)| 2 , M (k) = 2α d - ê cos(2πk • ê/L) + β 0 , (43
) where ê denotes the d basis vectors on the lattice and β 0 > 0 is an adjustable parameter; and
U (ϕ) = 1 2 (κ -κ 0 ) a |ϕ(a)| 2 + 1 4 γ a |ϕ(a)| 4 =⇒ Û ( φ) = 1 2 (κ -κ 0 ) k | φ(k)| 2 + 1 4 γ a L -d/2 k e -2iπk•a/L φ(k) 4 . (44
)
where ϕ and φ are Fourier transform pairs as defined in (42): the last term can be implemented via a (ifft( φ)) 4 (a).
this section cite: []

Section: Sampling using the Langevin SDE:
To obtain the ground-truth samples from the ϕ 4 model, one option is to use the SDE
d φt (k) = -M (k) φt (k)dt -(κ -κ 0 ) φt (k)dt -γ ϕ 3 t (k)dt + √ 2d Ŵt (k).(45)
where we denote
ϕ 3 t (k) = L -d/2 a e 2iπk•a/L L -d/2 k e -2iπk•a/L φt (k) 3 (46
)
which can be implemented via fft((ifft( φt )) 3 ). This SDE may be quite stiff, however, a problem that can be alleviated by changing the mobility and using instead
d φt (k) = -φt (k)dt-(κ-κ 0 ) M -1 (k) φt (k)dt-γ M -1 (k) ϕ 3 t (k)dt+ √ 2 M -1/2 (k)d Ŵt (k). (47
)
The discretized version of this equation reads
φtn+1 (k) = φtn (k) -∆t n φtn (k) + (κ -κ 0 ) M -1 (k) φtn (k) + γ M -1 (k) ϕ 3 tn (k) + 2∆t n M -1/2 (k)η n (k),(48)
where ηn is the Fourier transform of η n ∼ N (0, Id).
Computing the generator of the posterior distribution As illustrated with (25), one would like to sample a slightly different ϕ 4 model with energy function, noted E r . E and E r define respectively the prior and posterior distribution, via the Boltzmann's law (23). In this work, only relations of the form E r = E -⟨ϕ, Aφ⟩ -⟨h, φ⟩ are studied, combining a linear and quadratic term. A is assumed to be definite negative. Lemma B.1. Assume that β = 1 -α and let
η(α, x) = E[x 0 -x 1 |αx 0 + (1 -α)x 1 = x] = η 0 (α, 1 -α, x) -η 1 (α, 1 -α, x) (49
)
Then
η 0 (α, 1 -α, x) = x + (1 -α)η(α, x), (50
) η 1 (α, 1 -α, x) = x -αη(α, x).(51)
Proof of lemma B.1. Solving ( 5) and ( 49) in (η 0 , η 1 ) gives ( 50) and ( 51).
The idea is to use the drift of the prior to sample from the posterior. The following proposition makes it possible.
Proposition B.2 (Posterior drift). Assume β = 1 -α with α diagonal and invertible, and let
η r (α, x) = E x 0 -x 1 |αx 0 + (1 -α)x r 1 = x = η r 0 (α, 1 -α, x) -η r 1 (α, 1 -α, x)(52)
where the drifts η r 0 and η r 1 are defined in (18) and (19), respectively. Assume also that A is diagonal, non-positive definite, and invertible. Then, β r = 1 -α r and η r can be expressed as:
η r (α, x) = α -1 α r η(α r , x r ) + α -1 (x -x r ),(53)
where η(α, x) is the drift of the prior defined in (49), and α r and x r are given by
α r = α 1 -2α + α 2 (1 -A) -α 1 -2α -α 2 A , (54
) x r = α 2 r (1 -α) α 2 (1 -α r ) x + α 2 r (1 -α) b,(55)
Proof of the Proposition B.2. Using (50), ( 51) in ( 18) and ( 19), one obtains:
η r 0 (α, β, x) = α -1 ββ -1 r α r ((1 -α r )η(α r , β r , x r ) + x r ) + α -1 (x -ββ -1 r x r ), η r 1 (α, β, x) = x r -α r η(α r , β r , x r )
. Then, take the difference, and regrouping terms together eventually yields:
η r 0 (α, β, x) -η r 1 (α, β, x) = (α -1 βx r + x r ) =α -1 αr η(α r , β r , x r ) + α -1 ββ -1 r (α r -1)x r =-α -1 βxr -x r + α -1 x = α -1 α r η(α r , β r , x r ) + α -1 (x -x r ).
The linear case Assume for now that A = 0 and b = h. Note that one recovers the same case that in (25), that is, one applies a uniform magnetic field of magnitude h over the whole lattice. What follows is simply a corollary of B.2. Proposition B.3. Assume E r = E -(ϕ, h). Then, α r = α, β r = β, and
ϕ r = ϕ + αα T β -T h. (56
)
Also, the posterior drift η r writes:
η r (α, β, ϕ) = η(α, β, ϕ r ) -α T β -T h.(57)
Proof of Proposition B.3. Since A = 0, (20) directly implies α r = α and β r = β. Consequently, (56) follow from ( 21) and ( 57) from ( 53).
In summary, a simple shift proportional to h appears in the posterior field ϕ r . It clearly tends to favor the alignment with the magnetic field, which obeys common sense.
The quadratic case
• Assume that b = 0 and A = -k 2 Id.
In a similar fashion to the linear case, one derives analytical expressions for the quantities of interest.
Proposition B.4. Assume E r = E -(ϕ, Aϕ) = E + k 2 a ϕ(a) 2 , β r = 1 -α r and β = 1 -α. Then α r = α -α + 1 -2α + α 2 (1 + k 2 ) 1 -2α + α 2 k 2 , (58
) ϕ r = (-α + 1 -2α + α 2 (1 + k 2 ))(1 -α) 1 -2α + α 2 (1 + k 2 )(1 -2α + α 2 k 2 ) ϕ,(59)
and
η r (α, β, ϕ) = -α + 1 -2α + α 2 (1 + k 2 ) 1 -2α + α 2 k 2 η(α r , β r , ϕ r )(60)
+ α -1 ϕ 1 - (-α + 1 -2α + α 2 (1 + k 2 ))(1 -α) 1 -2α + α 2 (1 + k 2 )(1 -2α + α 2 k 2 ) .(61)
Proof of the Proposition B.4. Given the assumptions, (20) yields:
(Id -α r ) T α -T r α -1 r (Id -α r ) = (Id -α) T α -T α -1 (Id -α) -k 2 Id.
Observing that α and A are diagonals, α r is also diagonal. Furthermore, assuming that α r is proportional to the identity, the above reduces to the scalar equation (keeping the same notation for conciseness):
(
1 -α r ) 2 α 2 r = (1 -α) 2 α 2 + k 2
After a few elementary manipulations, one arrives at:
(1 -α r ) 2 α 2 = α 2 r (1 -α) 2 + k 2 α 2 r α 2 .
This is a quadratic equation that admits two solutions. Only one is positive, and writes as:
α r = -α 2 + α 1 -2α + α 2 (1 + k 2 ) 1 -2α + α 2 k 2 = α -α + 1 -2α + α 2 (1 + k 2 ) 1 -2α + α 2 k 2 . (62
)
It is quite easy to check that the discriminant is always positive, so it does not pose any problem. Also, if k ⩾ 0 and α ∈ [0, 1], then α r ∈ [0, 1]. This property is necessary, since α → η(•, α) has been trained in the hypercube [0, 1] d .
After elementary simplifications and recalling that β r = 1 -α r and ( 21), one has:
ϕ r = α 2 r α 2 1 -α 1 -α r = -α + 1 -2α + α 2 (1 + k 2 ) 1 -2α + α 2 k 2 2 1 -α 1 -α r . Since 1 -α r = 1-2α+α 2 k 2 +α 2 -α √ 1-2α+α 2 (1+k 2 ) 1-2α+α 2 k 2 = 1-2α+α 2 (1+k 2 )-α √ 1-2α+α 2 (1+k 2 ) 1-2α+α 2 k 2 , it yields: ϕ r = (-α + 1 -2α + α 2 (1 + k 2 )) 2 1 -2α + α 2 k 2 1 -α 1 -2α + α 2 (1 + k 2 ) -α 1 -2α + α 2 (1 + k 2 ) ,(63)
then factorizing by 1 -2α + α 2 (1 + k 2 ) eventually gives (59).
Eventually, after replacing (62) and ( 59) into ( 53), (60) holds.
• Assume that b = 0 and A = k 2 Id.
In this case, the quadratic equation is:
(1 -α r ) 2 α 2 r = (1 -α) 2 α 2 -k 2 , or otherwise stated: (1 -α r ) 2 α 2 -(1 -α) 2 α 2 r + α 2 r α 2 k 2 = 0. The discriminant of this polynomial is ∆ = α 2 (1 -k 2 ) -2α + 1.
Assuming it strictly positive, among the two solutions, only one is positive:
α r = α 1 -2α + α 2 (1 -k 2 ) -α 1 -2α -(αk) 2 .
The polynomial inside the square root is positive if and only if α / ∈ [ 1 1+k , 1 1-k ]. To see that, see there exists always two real roots, since the discriminants is 4k 2 > 0. Those roots are 1 1+k < 1 and 1 1-k > 1. Since α r ∈ [0, 1] must be respected for all α ∈ [0, 1], only k < 1 can be considered with our method. Consequently, sampling using stochastic interpolants from α = 1 to α = 0 appears impossible with this method.
this section cite: []

Section: B.3 Details about the maze experiment
We use the Hadamard interpolant (9) and estimate the drift η(α, x) defined in (10) by approximating it with a U-Net neural network Ho et al. (2020), trained with an Adam optimizer Kingma and Ba (2017). The U-Net comprises 4 stages with 48, 80, 160, and 256 channels respectively for the encoding flow. The decoder has the same architecture as the encoder but in reverse order, with added residual connections Ronneberger et al. (2015). Each stage consists of 2 residual blocks, with the first concatenated with a self-attention block. The input vector has shape d × 2, where row i contains the x and y coordinates of the i-th point in the trajectory.
In contrast to conventional U-Net architectures, we perform interpolation and max pooling operations independently on each coordinate column to increase and reduce dimensions only along the trajectory length axis. The convolution kernel size is 5 × 2, processing each point's coordinates together with those of its two temporal predecessors and successors in the sequence. We add the necessary padding to maintain identical input and output dimensions, which amounts to padding by two rows at the top and bottom of the input vector.
Given a pair of randomly chosen points in the maze, we must determine where to constrain these points along the generated trajectory. If the constraint points are placed too far apart in the sequence (large index difference), the resulting path will likely not be the shortest route; conversely, if placed too close together (small index difference), the generated path has an increased chance of cutting through forbidden regions, making it inadmissible. To address this trade-off, we adopt the following heuristic. We fix the starting point at the beginning of the path (index i = 1) and employ a progressive search for the target point placement using the candidate indices [5,10,20,30,40,45,50]. We first generate a path with the target point constrained at index 5 (creating a short trajectory). If this path intersects forbidden regions, we increase the target index to 10 (allowing a longer path), and continue this process until we generate a valid path that successfully avoids all obstacles.
On Figure 8, we impose paths to go by the bottom-right corner, the constraint is visible as a small white dot. The path length adapts accordingly.
this section cite: ['b3']

Section: C Additional experimental results
Here we provide additional infilling image results, given in Figure 9. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6. Experimental setting/details Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All test details are provided in Appendix (e.g. Table 2).
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The full distribution of the average magnetization has been studied for our ϕ 4 model in Figure 5. Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text. 8.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
this section cite: []

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer [Yes] , [No] , or [NA] .
• [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation.
While " [Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: The abstract claims that the formalism of stochastic interpolants have been adapted for zero-shot conditional generation, in-painting, and posterior sampling. The corresponding numerical experiments are explicitly displayed in their associated sections.
Posterior sampling has been theoretically explored and numerically investigated in the ϕ 4 model with stochastic interpolants. We show that our model can learn to generate images at multiple scales: pixel-wise, blockwise, and in all dimensions simultaneously.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: A discussion on this matter is present in the conclusion.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: All proofs, with their set of assumptions, are provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: The details of all the numerical experiments are discussed either in the main text of the paper or in the Appendix, especially concerning the one-shot conditional generation and inpainting.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?
Answer: [Yes] Justification: All the code and data used for the numerical experiments will be exposed in a Github repository. A set of instructions to fully reproduce the results will be provided in a README file.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
Answer: [Yes] Justification: A Table containing all relevant information will be provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Our paper presents theoretical and mathematical foundations for multitask generative modeling using operator-based interpolants. The research primarily consists of mathematical formulations and theoretical derivations. Our experiments are limited to publicly available benchmark datasets (MNIST digits) and physical simulation data, neither of which contain sensitive information or raise ethical concerns. We do not collect or use personal data, conduct experimentation on humans or animals, or develop technologies with potential for harm or misuse. All sources and prior work are properly cited, and our mathematical derivations, proofs, and experimental procedures are presented transparently. The research was conducted with scientific integrity, honesty, and transparency throughout the process, fully adhering to all aspects of the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [NA] Justification: Our paper presents a theoretical framework for operator-based interpolants in generative modeling, which primarily advances fundamental research in this area. The work is largely theoretical and mathematical in nature, focusing on developing a universal approach to multitask learning rather than specific applications with direct societal implications. While we discuss potential technical benefits in the paper's conclusion, the theoretical nature and early stage of this research makes specific societal impacts, whether positive or negative, difficult to assess meaningfully at this time.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML). 11. Safeguards Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: This question is not applicable to our paper as we do not release any models or datasets that present a high risk for misuse.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [Yes] Justification: In our paper, we properly credit the creators and original owners of all assets used, including the MNIST dataset and any referenced algorithms or methodologies. For the MNIST dataset, which is in the public domain, we acknowledge its source and cite the original publication. All assets are used in accordance with their intended research purposes and we have carefully respected all applicable terms of use and licensing requirements throughout our research process. Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [Yes] Justification: Our paper introduces new theoretical formulations and algorithms for operatorbased interpolants, which are thoroughly documented within the paper itself. The mathematical framework, definitions, lemmas, and propositions are rigorously presented with complete derivations and proofs (in the appendix). For our experimental implementations, we provide detailed descriptions of the model architectures, training procedures, and inference methods in the paper and will release accompanying code with comprehensive documentation that explains the implementation of our operator-based interpolant framework. This documentation includes clear instructions for reproducing our experiments, explanations of key parameters, and examples demonstrating how to apply our methods to various tasks. All assets are carefully documented to ensure transparency and reproducibility of our research.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable
). You can either create an anonymized URL or include an anonymized zip file. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: This question is not applicable to our research as our paper does not involve any human subjects or study participants. Our experiments are conducted exclusively on standard benchmark datasets (MNIST) and physics simulation data, with no human participation involved at any stage. Since no human subjects were part of this research, no IRB approvals or equivalent reviews were necessary, and there were no study-related risks to disclose or manage. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 15. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: No LLM or any kind of transformer architecture is involved in the numerical experiments displayed. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Flow matching for generative modeling Year: (2022)
Ref_id:b1 Title: Building normalizing flows with stochastic interpolants Year: (2022)
Ref_id:b2 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b3 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b4 Title: Generative Modeling by Estimating Gradients of the Data Distribution Year: (2020)
Ref_id:b5 Title: Diffusion schrödinger bridge with applications to score-based generative modeling Year: (2021)
Ref_id:b6 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b7 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b8 Title: Diffusion models beat gans on topology optimization Year: (2023)
Ref_id:b9 Title: Generative adversarial networks and diffusion models in material discovery Year: (2024)
Ref_id:b10 Title: Diffusion posterior sampling for general noisy inverse problems Year: (2023)
Ref_id:b11 Title: Solving inverse problems in medical imaging with score-based generative models Year: (2022)
Ref_id:b12 Title: DMPlug: A plug-in method for solving inverse problems with diffusion models Year: (2024)
Ref_id:b13 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b14 Title: Stochastic interpolants with data-dependent couplings Year: (2023)
Ref_id:b15 Title: A survey of stochastic simulation and optimization methods in signal processing Year: (2015)
Ref_id:b16 Title: Denoising diffusion restoration models Year: (2022)
Ref_id:b17 Title: Pnp-flow: Plug-and-play image restoration with flow matching Year: (2025)
Ref_id:b18 Title: Plug-and-play split gibbs sampler: embedding deep generative priors in bayesian inference Year: (2024)
Ref_id:b19 Title: Provable probabilistic imaging using score-based generative priors Year: (2024)
Ref_id:b20 Title: A variational perspective on solving inverse problems with diffusion models Year: (2023)
Ref_id:b21 Title: Variational diffusion models for blind mri inverse problems Year: (2023)
Ref_id:b22 Title: Score-guided intermediate layer optimization: Fast langevin mixing for inverse problems Year: (2022)
Ref_id:b23 Title: Cold diffusion: Inverting arbitrary image transforms without noise Year: (2022)
Ref_id:b24 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b25 Title: Fractal generative models Year: (2025)
Ref_id:b26 Title: Simplified and generalized masked diffusion for discrete data Year: (2025)
Ref_id:b27 Title: Train for the worst, plan for the best: Understanding token ordering in masked diffusions Year: (2025)
Ref_id:b28 Title: Flow priors for linear inverse problems via iterative corrupted trajectory matching Year: (2024)
Ref_id:b29 Title: D-flow: Differentiating through flows for controlled generation Year: (2024)
Ref_id:b30 Title: Training-free linear image inverses via flows Year: (2024)
Ref_id:b31 Title: Simulation of ϕ 4 theory in the strong coupling expansion beyond the Ising Limit Year: (2010)
Ref_id:b32 Title: Flow-based generative models for markov chain monte carlo in lattice field theory Year: (2019-08)
Ref_id:b33 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
