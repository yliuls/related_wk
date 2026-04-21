Title: ROOT: Rethinking Offline Optimization as Distributional Translation via Probabilistic Bridge
Abstract: This paper studies the black-box optimization task which aims to find the maxima of a black-box function using a static set of its observed input-output pairs. This is often achieved via learning and optimizing a surrogate function with that offline data. Alternatively, it can also be framed as an inverse modeling task that maps a desired performance to potential input candidates that achieve it. Both approaches are constrained by the limited amount of offline data. To mitigate this limitation, we introduce a new perspective that casts offline optimization as a distributional translation task. This is formulated as learning a probabilistic bridge transforming an implicit distribution of low-value inputs (i.e., offline data) into another distribution of high-value inputs (i.e., solution candidates). Such probabilistic bridge can be learned using low-and high-value inputs sampled from synthetic functions that resemble the target function. These synthetic functions are constructed as the mean posterior of multiple Gaussian processes fitted with different parameterizations on the offline data, alleviating the data bottleneck. The proposed approach is evaluated on an extensive benchmark comprising most recent methods, demonstrating significant improvement and establishing a new state-of-the-art performance. Our code is publicly available at https://github.com/cuong-dm/ROOT.

Section: Introduction
Black-box optimization arises in scientific and engineering domains where evaluating each candidate solution is costly, often requiring extensive physical experiments or high-fidelity simulations [61]. For instance, designing energy-efficient hardware accelerators [5,9,37] involves numerous cycle-accurate simulations to assess configuration performance. In materials science, finding nanoporous structures with high adsorption capacity for carbon capture or hydrogen storage demands labor-intensive lab experiments [17,20]. Similar challenges arise in protein design [21], molecular generation [16], and drug discovery [52], where evaluations are likewise expensive.
Prior Literature. Existing approaches to black-box optimization include both online and offline methods with the latter being an emerging alternative of the former. In particular, online methods such as Bayesian optimization [53,34,54,73,74,62,63,64,65,33] have long been explored for black-box design tasks with provable performance guarantee in the asymptotic limit of data. However, their reliance on iterative experimentation makes them less practical in high-cost settings with limited to no experimentation budget. In contrast, black-box methods leverage past data to learn a surrogate model which can be used to find better designs without incurring new experimentation [36,7,18,12,27,40,59,60,14,15]. In this paper, we focus on the offline setting, where the goal is to discover high-performing designs using only past experimentation data.
Challenge. Among offline methods, the main challenge is that surrogate models can become increasingly erroneous when the search moves away from the offline dataset, especially when offline data are biased or sparse, causing these models to overfit. To mitigate this issue, most existing approaches have focused on advancing techniques in (1) forward modeling that penalize high-value surrogate predictions at out-of-distribution (OOD) inputs [60,13,70], (2) inverse modeling that find most promising and reliable regions that contain high-performing inputs [40,45,39,10] to sidestep the OOD issue of forward modeling, and (3) search policies that learn a direct plan to navigate from low-value inputs to high-value inputs [10,38].
Limitations. Despite their promising results, forward and inverse approaches depend on learning a mapping (or inverse mapping) between input designs and their corresponding performance outputs using offline data. As a result, their effectiveness is inherently limited by the availability of data. Likewise, learning direct search policies also suffers from the same data bottleneck since these methods still need to sample heuristic trajectories from the offline dataset to use as learning feedback.
Furthermore, information regarding regions with high-performing inputs is often not observable from the offline data, especially in low-data scenarios, which might further restrict the effectiveness of the learned models/policies. To mitigate such data bottleneck, we propose to approach offline optimization from a new perspective of distributional translation, as highlighted below.
this section cite: ['b61', 'b4', 'b9', 'b37', 'b17', 'b20', 'b21', 'b16', 'b52', 'b53', 'b34', 'b54', 'b73', 'b74', 'b62', 'b63', 'b64', 'b65', 'b33', 'b36', 'b7', 'b18', 'b12', 'b27', 'b40', 'b59', 'b60', 'b14', 'b15', 'b60', 'b13', 'b70', 'b40', 'b45', 'b39', 'b10', 'b10', 'b38']

Section: Distributional Translation.
In essence, we view the offline data as an implicit distribution over lowvalue designs, and recast offline optimization as the task of learning a probabilistic transformation, or bridge, that transports this distribution toward a regime of higher-value inputs. By moving along the learned probabilistic bridge, we can reach regions of the input space associated with better designs. However, learning such a transformation is fundamentally limited by the scarcity of high-value examples. Our key insight to overcome this bottleneck is:
Although the feedback needed to learn such low-to-high probabilistic transformation is absent in the offline dataset, it can be derived from a distribution of synthetic functions that are similar to the (unknown) target function (up to a scale factor).
These synthetic functions can be provably constructed across various output scales, alleviating the data bottleneck and broadening the solution scope of offline optimization.
this section cite: []

Section: Technical Contributions.
Our solution perspective is substantiated via the following:
1. A general-purpose probabilistic bridge model that learns a direct mapping between two implicit data distributions. This perspective rethinks offline optimization through the lens of probabilistic transport. The resulting bridge can incorporate external guiding information from synthetic functions similar to the oracle to mitigate the data bottleneck. Once trained, it enables simulation of paths that move from low-value to high-value regimes (Section 3.1).
this section cite: []

Section: 2.
A pre-training and adaptation framework that (1) learns multiple Gaussian process priors [66] over synthetic functions resembling the target function, and (2) samples representative low-and high-value inputs from their corresponding closed-form mean functions. This generates high-quality training data that better delineates the low-and high-value regimes for learning the probabilistic bridge. The intuition is that if the bridge can consistently map between these regimes across a wide range of functions similar to the oracle, it will be able to do the same for the oracle (Section 3.3).
3. An extensive empirical evaluation on a variety of benchmark datasets [59] and numerous existing baselines, establishing a new state-of-the-art performance, which significantly and consistently improves over previous work. Our empirical evaluation also features rich ablation studies examining in detail the practical impact of different components of our framework on its performance (Section 4).
this section cite: ['b66', 'b59']

Section: Problem Definition and Preliminaries
This section provides a concise formulation of offline black-box optimization (Section 2.1) and important background on Gaussian processes (Section 2.2), which was used later for sampling additional data from synthetic functions similar to the oracle behind the offline data.
this section cite: []

Section: Offline Black-Box Optimization
Offline black-box optimization is formulated as the maximization of a black-box function f (x) using only an offline dataset of observations D o = {(x i , y i )} n i=1 which x i denote a past experiment design and y i = f (x i ) is its corresponding evaluation. A direct approach to this problem is to learn a surrogate g(x; ω * ) of f (x) via fitting its parameter ω * to the offline dataset,
ω ⋆ ≜ arg min ω L(ω) ≜ arg min ω n i=1 ℓ g(x i ; ω), y i ,(1)
where ω denotes a parameter candidate of the surrogate and ℓ(g(x; ω), y) denotes the prediction loss of g(.; ω) on x if its oracle output is y. The (oracle) maxima of f (x) is then approximated via,
x * ≜ arg max x g(x; ω * ) .(2)
The main issue with this approach is that g(x; ω * ) often predicts erratically at out-of-distribution (OOD) inputs. To mitigate this, numerous surrogate or search regularizers have been proposed to either penalize the high-value surrogate prediction at OOD inputs [13,60,70] or find an inverse mapping from the desired output to potential inputs [39,45], as detailed in Section 5. Nonetheless all these approaches are restricted by the limited amount of offline data. Alleviating this bottleneck to reach new SOTA performance is main aim of our proposed approach.
this section cite: ['b13', 'b60', 'b70', 'b39', 'b45']

Section: Gaussian Processes
A Gaussian process (GP) [50] defines a probabilistic prior over a random function h(x). It is parameterized by a mean function m(x) = 0 3 and a kernel function k(x, x ′ ). These functions induce a marginal Gaussian prior over the evaluations h = [h(x 1 ) . . . h(x n )] ⊤ of any finite subset of inputs {x 1 , . . . , x n }. Let x τ be an unseen input whose corresponding output h τ = h(x τ ) we wish to predict. The Gaussian prior over
[h(x 1 ) . . . h(x n ) h(x τ )] ⊤ implies: h(x τ ) | h ∼ N k ⊤ τ K -1 h, k(x τ , x τ ) -k ⊤ τ K -1 k τ ,(3)
where k τ = [k(x τ , x 1 ) . . . k(x τ , x n )] ⊤ and K denotes the Gram matrix induced on {x 1 , . . . ,
x n } for which K ij = k(x i , x j ). Assuming a Gaussian likelihood y ∼ N(h(x), σ 2 ), it follows that h(x τ ) | y ∼ N k ⊤ τ (K + σ 2 I) -1 y, k(x τ , x τ ) -k ⊤ τ (K + σ 2 I) -1 k τ ,(4)
which explicitly forms the predictive distribution of a Gaussian process. The choice of the kernel function k(x, x ′ ) dictates certain properties of the sample functions. In this context, we adopt the commonly used RBF kernel, k(x, x ′ ) = σ 2 exp(-0.5 • ∥x -x ′ ∥ 2 /ℓ 2 ) with the parameter σ 2 represents the signal variance, controlling the function's amplitude, while ℓ denotes the unit length-scale which regulates the function's smoothness. There also exists an extensive literature on improving the complexity of Gaussian process via sparse approximation [47,48,56,41,24,42,57,8] that enables fast inference with linear complexity in large-scale datasets [24,30,31,28,32,29]
𝒚 !"# 𝑿 !"# 𝒟 !""#$%& Mean Functions g ' g % 𝑿 !"#! 𝒚 !"
#! 𝒟 ()%*+&*$, G ra di en t as ce nt 𝑿 $%& 𝒚 $%& 𝑿 !"#! 𝒚 !"#! 𝑿 $%& 𝒚 $%& G ra di en t de sc en t G ra di en t as ce nt G ra die nt de sc en t Classifier-free guidance DP Gaussian Processes … 𝐺𝑃 ' 𝐺𝑃 % Probabilistic Bridge Model 𝑿 #!- 𝒚 #!- 𝑿 +$.+ 𝒚 +$.+ Step 1 : Synthetic data generation Step 3 : Simulation Step 2 : Training 𝑿 * Top-𝐾
this section cite: ['b50', 'b47', 'b48', 'b56', 'b41', 'b24', 'b42', 'b57', 'b8', 'b24', 'b30', 'b31', 'b28', 'b32', 'b29']

Section: Rethinking Offline Optimization as Distributional Translation
This section introduces a new perspective on offline optimization by framing it as a translation task. We intuitively view the offline data as a source language composed of low-value designs and aim to translate it into a target language of high-value inputs. To enable this translation, we introduce the concept of a probabilistic bridge, which identifies local translation examples by explicitly conditioning on both the source and target contexts. This conditioning enables the construction of feasible transformation paths that connect selected low-and high-value designs within their local neighborhoods, grounding the translation process in meaningful design correspondences (Section 3.1).
These local bridges serve as examples of how to move between regimes when both endpoints are known. The learning algorithm then weaves together these examples to form a global translator capable of generalizing beyond the observed pairs and enabling translation in new contexts where only the source is known. This approach generalizes diffusion models [25], which translate between data and noise, and allows external guiding information to be incorporated, enriching the context and addressing the data bottleneck in offline optimization (Section 3.3).
this section cite: ['b25']

Section: Probabilistic Bridge for Distributional Translation
This section formalizes the concept of a probabilistic bridge, which generalizes the widely used Denoising Diffusion Probabilistic Model (DDPM) [25]. While DDPM defines a forward diffusion process that maps data to Gaussian noise and learns to reverse it, the probabilistic bridge constructs a space of localized transformation flows conditioned on both the source and the target, representing low-value and high-value designs drawn from implicit data distributions. Learning and applying a probabilistic bridge model for distributional translation involves two phases: a construction phase (Section 3.1) and a learning phase (Section 3.3).
In the construction phase, the transition kernel of each conditioned bridge is derived, capturing a source-to-target translation plan within its local context. These localized examples serve as concrete demonstrations of how to move between regimes. In the learning phase, they are used to train a global, target-agnostic transformation flow that generalizes beyond the observed pairs and enables translation from arbitrary source inputs to valid target outputs. These two phases are described in detail below. A concrete example of a probabilistic bridge is also given in Section 3.2.
this section cite: ['b25']

Section: Probabilistic Bridge Construction
Given two endpoints x 0 and x T , we define a probabilistic bridge between them as a discrete observation of a vector-value, time-indexed random function x 1 , x 2 , . . . , x T -1 which is distributed by Gaussian process [50] with mean function ψ t (x 0 , x T ) and covariance kernel κ t,k I 4 . This implies a marginal Gaussian on the probabilistic bridge x = vec[x 1 , x 2 , . . . , x T -1 ] with mean ψ = vec[ψ 1 (x 0 , x T ); . . . ; ψ T -1 (x 0 , x T )] which reveals a mass-moving flow between x T and x 0 ,
q x | x 0 , x T ≜ N x; ψ, κ ⊗ I ⇒ q x t | x 0 , x T = N x t ; ψ t (x 0 , x T ), κ t,t I ,(5)
where ⊗ denotes the Kronecker product, κ is a matrix containing the entries κ t,k and the mean function ψ t (x 0 , x T ) must meet boundary conditions ψ 0 (x 0 , x T ) = x 0 and ψ T (x 0 , x T ) = x T . This further reveals a closed-form expression for backward transition conditioned on both endpoints,
q x t-1 | x t , x 0 , x T = N x t-1 ; µ(x t , x 0 , x T ), κt-1 I ,(6)
with the transition mean µ(x t , x 0 , x T ) = ψ t-1 (x 0 , x T ) + κ t-1,t κ -1 t,t x t -ψ t (x 0 , x T ) and covariance κt-1 = κ t-1,t-1 -κ t-1,t κ -1 t,t κ t,t-1 . This reveals a (step-wise) conditional backward transition that induces a valid transformation flow mapping x T back to x 0 . This means for each sampled pair (x T , x 0 ), the corresponding target-conditioned backward transition q(x t-1 | x t , x 0 , x T ) provides an example of a localized transformation flow between them, which can be used to train the desired target-agnostic map p θ (x t-1 | x t , x T ) as detailed next.
this section cite: ['b50']

Section: Learning Probabilistic Bridge Model
To learn the target-agnostic transformation that maps from a source to a plausible target without knowing it beforehand, we parameterize it as
p θ x t-1 | x t , x T = N x t-1 ; µ θ (x t , x T , t), κt-1 I .(7)
which uses the same (known) variance parameter as the localized flow/bridge example above and a (learnable) spatio-temporal network ϵ θ (x t , t) to parameterize the target-agnostic mean transition. This is the centerpiece of the probabilistic bridge model in Eq. 7 which can now be learned via optimizing its parameter θ to match the example flows generated using Eq. 6,
θ PB = arg min θ E (x0,x T ,t) D KL q x t-1 | x t , x 0 , x T ∥p θ x t-1 | x t , x T . (8
)
The KL divergence in Eq. 8 above is between two Gaussians and can be computed in closed form which allows for a direct optimization of θ. Given the optimized θ PB and a source x T (low-value design), we can now simulate the generic transformation flow in Eq. 7 to obtain a plausible target x 0 (high-value design) which is not known apriori,
x t-1 = µ θPB (x t , x T , t) + √ κ ϵ ,(9)
where we sample ϵ ∼ N(0, I) when t > 1 and set ϵ = 0 otherwise. This gives us the desired transformation that translates a candidate in the low-value regime to another in the high-value regime. Furthermore, to improve training efficiency, we can also adopt the practical approach in guided diffusion which further conditions µ θ (x t , x T , t) = µ θ (x t , x T , y 0 , y T , t) on the source and target outputs (i.e., y T and y 0 ) [26].
this section cite: ['b26']

Section: Example and Practical Setup
This section provides an working example to substantiate the above probabilistic bridge framework. Choosing ψ t (x 0 , x T ) = x 0 (1-t/T )+x T (t/T ) and κ t,k = (min(t, k)/T )(1-max(t, k)/T ) results in a Brownian bridge construction,
q x t | x 0 , x T = N x t ; x 0 1 - t T + x T t T , t T 1 - t T I . (10
)
This specifies a transport plan that moves the unit point mass located as x T to x 0 . To see this, note that when t = 0, Eq. 10 reduces to N(x 0 , 0) and when t = T , it reduces to N(x T , 0). This setup emphasizes a point-to-point transformation which is particularly suitable for offline optimization.
As we substantiate Eq. 10 with the appropriate (low-value, high-value) pairs (x T , x 0 ) and derive examples of localized flow mapping from the low-and high-value regimes following the blueprint in Section 3.1.1, we can learn a target-agnostic transformation consistent with using Eq. 8. This setup has been used in our experiments to achieve significant performance improvement over prior work, effectively establishing new SOTA. Further details regarding the specific derivation of the Brownian bridge and its practical traning procedure is detailed in Appendix B.2.
this section cite: []

Section: Remark.
The above framework offers a broad and flexible perspective on offline optimization that has not been fully explored in the existing literature. From this viewpoint, there are many promising directions for future investigation. Each valid specification of the mean and kernel functions in the probabilistic bridge model in Eq. 5 defines a distinct way of transporting probability mass from low-value to high-value regions. While offline optimization ultimately requires only one such transport plan, different choices may lead to significantly different learning behaviors and complexities. Understanding how the choice of bridge design influences learning performance remains an important and open question for future work.
▷ Learning θ PB however requires access to paired samples of low-value and high-value designs, which is nontrivial in the offline optimization setting where the dataset typically reflects only the low-value regime. This raises a fundamental question: where can we obtain representative samples from the high-value regime to support bridge construction? Addressing this challenge is critical for making the probabilistic bridge framework operational, and is the focus of Section 3.3.
this section cite: []

Section: Synthetic Data Generation and Practical Black-Box Optimization Algorithm: ROOT
This section revisits the challenge introduced in Section 1: Learning a probabilistic bridge requires examples of localized flows from low-to high-value regimes. Yet, such examples are not available in standard offline datasets, which by design lack high-performing solutions. To address this, we propose a key hypothesis: if a probabilistic bridge can consistently translate low-value inputs to high-value outputs across a sufficiently large set of functions similar to the oracle, it is likely to generalize well to the oracle.
this section cite: []

Section: Motivation.
This motivates an alternative data generation strategy. Rather than depending solely on the offline dataset, we construct a collection of synthetic functions with closed-form structure, allowing us to sample large quantities of low-and high-value inputs. These synthetic examples enable us to train a meta probabilistic bridge with strong zero-shot adaptation. This strategy reflects the common pre-training paradigm in foundation model development, where knowledge accumulated from broad synthetic tasks can be readily transferred to any target task of interest. This motivation leads to our overall workflow, which is illustrated in Figure 1.
this section cite: []

Section: Function Sampling.
Collecting data from such similar functions is fortunately possible using the Gaussian process [50] which specifies a prior over functions around a given mean function. Assuming that the mean function is set to be the oracle, most sampled functions from the corresponding GP will be similar to it. Since we do not know the oracle function, another approach is to begin with a generic GP prior with kernel parameters ϕ = (σ, ℓ) and compute the corresponding GP posterior using the offline data D o . The resulting GP posterior mean is
g ϕ s (x) = k(ϕ s ) ⊤ K(ϕ s ) + σ 2 I -1 y ,(11)
which has a closed form and is similar the oracle function around the offline data D o . To obtain a wider range of functions similar to the oracle, we compute multiple GP posteriors according to a diverse range of GP priors with different signal and length-scale parameters. These can be obtained from learning a hierarchical GP prior with top-level prior over kernel parameters defining the corresponding GP posteriors at the lower level. We can then sample kernel parameters {ϕ s } ng s=1
and extract the corresponding posterior means -see Eq. 11.
Low-and High-Value Simulation. Given these functions, we can construct X - s and X + s as the set of low-and high-value inputs of g ϕ s (x) via running M -step gradient descent and ascent from a subset of n p offline input points:
X - s ≜ x - M ≜ x - 0 -η M -1 m=0 ∇ x g ϕ s (x - m ) | x - 0 ∈D0 ,(12)
X + s ≜ x + M ≜ x + 0 + η M -1 m=0 ∇ x g ϕ s (x + m ) | x + 0 ∈D0 ,(13)
where
x + m+1 ≜ x + m + η∇ x g ϕ s (x + m ) and x - m+1 ≜ x - m -η∇ x g ϕ s (x - m ).
Here, x + 0 = x - 0 ∈ D o . The corresponding outputs for the above inputs can also be computed using the closed-form of the posterior means. A synthetic dataset D s collecting low-and high-values from those closed-form posterior mean functions can then be created:
D s = (X - s , y - s ), (X + s , y + s ) ng s=1
where
y - s = g ϕ s (X - s ) and y + s = g ϕ s (X + s ) , (14
)
where n g is the number of sampled functions.
this section cite: ['b50']

Section: Learning Probabilistic Bridge.
Following the blueprint in Section 3.1.1, we can sample (x T , y T , x 0 , y 0 ) ∼ D s as training data to train our probabilistic bridge mapping between lowand high-value regions across the aforementioned posterior mean functions, which follows Eq. 8. Furthermore, we also adopt the guided diffusion [26] technique to incorporate the output information y T and y 0 as part of the input to the prediction network µ θ (x t , x T , y 0 , y T , t) as stated after Eq. 9.
Simulation. Once this probabilistic bridge model has been trained, we can utilize its corresponding (step-wise) source-to-target transition in Eq. 9 to map each offine input (presumbaly in the lowvalue regime) to a better solution candidate (in a high-value regime). For example, we can run this simulation for the top 128 offline inputs with highest values. The effectiveness of this approach is thoroughly evaluated in Section 4, where we demonstrate its ability to generate high-quality candidates across various benchmarks. A full description of our algorithm and implementation is presented in Appendix B. Our algorithm's computational complexity is discussed in Appendix. C.4.
this section cite: ['b26']

Section: Experiments
This section evaluates our proposed method, ROOT, through extensive empirical comparisons against various recent baselines on a standard offline optimization benchmark [59]. The experiment setup is summarized in Section 4.1, with detailed results in Section 4.2 and ablation studies in Section 4.3.
this section cite: ['b59']

Section: Experiments Settings
Benchmark Tasks. Our investigation covers four real-world tasks selected from the Design-Bench [59]foot_2 and three RNA-Binding tasks from ViennaRNA [44]. In Design-Bench, the chosen tasks cover both discrete and continuous domains. The discrete tasks, TF-Bind-8 and TF-Bind-10 [3], aim to discover DNA sequences with high binding affinity to a specific transcription factor (SIX6 REF R1). On the continuous side, Ant Morphology [6] and D'Kitty Morphology [2] focus on optimizing the physical structure of a simulated robot ant from OpenAI Gym [6] and the D'Kitty robot from ROBEL [2]. For ViennaRNA, we include three RNA-Binding tasks as RNA-A, RNA-B, and RNA-C [44,35]. For further details on these tasks, please read Appendix A.
Baselines. We selected 21 widely recognized methods, including BO-qEI [59], CMA-ES [23], REINFORCE [67], COMs [60], CbAS [7], MINs [40], RoMA [69], DDOM [39], ICT [70], Tri-mentoring [13], GTG [72], BDI [12], RGD [11], LTR [55], BONET [38], MATCH-OPT [27], PGS [10], GABO [68], DEMO [71], GA on GP (Section 4.3) and standard GA.
Evaluation Protocol. Following the approach in [59], each method generates 128 optimized design candidates, which are then evaluated by the oracle function. The performances are ranked, and results are recorded at the 50 th , 80 th , and 100 th percentiles. To ensure consistency, all results are averaged over 8 independent runs with reported standard deviation.
Hyper-parameter Configuration. For each baseline, we adopt the optimized settings from the original papers. For GP kernel hyper-parameters in our data generation, we sample lengthscales ℓ s and variances σ 2 s uniformly from [ℓ 0 -δ, ℓ 0 + δ] and [σ 2 0 -δ, σ 2 0 + δ], with ℓ 0 = σ 2 0 = 1.0 for continuous tasks and 6.25 for discrete tasks, and δ = 0.25. We use M = 100 gradient steps with step sizes 0.001 (continuous) and 0.05 (discrete). Additional data generation details are in Appendix B.1 and Table 6. For training the Probabilistic Bridge model, we use a Brownian Bridge diffusion process with the Adam optimizer over E = 100 epochs and n g = 800 synthetic functions, running on a single NVIDIA A100-SXM4-80GB GPU. More training details are provided in Appendix B.2.
this section cite: ['b44', 'b6', 'b1', 'b6', 'b1', 'b44', 'b35', 'b59', 'b23', 'b67', 'b60', 'b7', 'b40', 'b69', 'b39', 'b70', 'b13', 'b72', 'b12', 'b11', 'b55', 'b38', 'b27', 'b10', 'b68', 'b71', 'b59']

Section: Experimental Results
Table 1: Experiment results on Design-Bench Tasks. We report the maximum score (100 th percentile) among Q = 128 candidates. Blue denotes the best entry in the column, while Brown indicates the second best. Mean Rank is the average rank across all benchmark tasks.
Benchmarks Method Ant D'Kitty TFBind8 TFBind10 Mean Rank D o (best) 0.565 0.884 0.439 0.467 -BO-qEI 0.812 ± 0.000 0.896 ± 0.000 0.825 ± 0.091 0.627 ± 0.033 16.75 / 22 CMA-ES 1.561 ± 0.896 0.724 ± 0.001 0.939 ± 0.039 0.664 ± 0.034 8.00 / 22 REINFORCE 0.263 ± 0.026 0.573 ± 0.204 0.961 ± 0.034 0.618 ± 0.011 17.00 / 22 GA 0.293 ± 0.029 0.860 ± 0.021 0.985 ± 0.011 0.638 ± 0.032 12.75 / 22 COMs 0.882 ± 0.044 0.932 ± 0.006 0.940 ± 0.027 0.621 ± 0.033 13.25 / 22 CbAS 0.846 ± 0.033 0.895 ± 0.016 0.903 ± 0.028 0.649 ± 0.055 12.50 / 22 MINs 0.894 ± 0.022 0.939 ± 0.004 0.908 ± 0.063 0.630 ± 0.019 12.50 / 22 GA on GP 0.948 ± 0.013 0.946 ± 0.001 0.770 ± 0.087 0.654 ± 0.038 9.25 / 22 RoMA 0.593 ± 0.066 0.829 ± 0.020 0.665 ± 0.000 0.553 ± 0.000 20.00 / 22 ICT 0.911 ± 0.030 0.945 ± 0.011 0.888 ± 0.047 0.624 ± 0.033 13.50 / 22 Tri-mentoring 0.944 ± 0.033 0.950 ± 0.015 0.899 ± 0.045 0.647 ± 0.039 9.00 / 22 MATCH-OPT 0.931 ± 0.011 0.957 ± 0.014 0.977 ± 0.004 0.543 ± 0.002 9.50 / 22 PGS 0.949 ± 0.017 0.966 ± 0.013 0.981 ± 0.015 0.532 ± 0.000 7.75 / 22 LTR 0.907 ± 0.032 0.960 ± 0.014 0.973 ± 0.000 0.652 ± 0.039 6.25 / 22 DDOM 0.930 ± 0.029 0.925 ± 0.008 0.885 ± 0.061 0.634 ± 0.015 13.75 / 22 GTG 0.865 ± 0.040 0.935 ± 0.010 0.901 ± 0.039 0.639 ± 0.016 12.50 / 22 BDI 0.964 ± 0.000 0.941 ± 0.000 0.973 ± 0.000 0.636 ± 0.020 7.50 / 22 RGD 0.922 ± 0.020 0.883 ± 0.014 0.889 ± 0.068 0.644 ± 0.048 13.00 / 22 BONET 0.948 ± 0.025 0.957 ± 0.008 0.894 ± 0.086 0.606 ± 0.024 10.75 / 22 GABO 0.224 ± 0.051 0.719 ± 0.001 0.939 ± 0.038 0.639 ± 0.033 15.25 / 22 DEMO 0.948 ± 0.013 0.956 ± 0.011 0.812 ± 0.054 0.648 ± 0.042 9.25 / 22 ROOT (ours) 0.965 ± 0.014 0.972 ± 0.005 0.986 ± 0.007 0.685 ± 0.053 1.25 / 22 Table 3: Effect of initial-point selection on ROOT's performance.
this section cite: []

Section: Type Ant TFBind8
Random 0.953 ± 0.014 0.976 ± 0.007 Lowest 0.545 ± 0.214 0.969 ± 0.009 Highest 0.965 ± 0.014 0.986 ± 0.007 Table 4: ROOT vs ExPT in fewshot settings.
this section cite: []

Section: Method Ant TFBind8
ExPT 0.940 ± 0.027 0.874 ± 0.071 ROOT 0.942 ± 0.035 0.895 ± 0.086
This section compares our method to 21 baselines, evaluating the 50th, 80th, and 100th percentiles. Due to limited space, we report only the 100th percentile results in the main text; details on the 50th and 80th percentiles and score distributions for our method versus others are in Appendix C.2.
Results on Continuous Tasks: Table 1 (first two columns) shows our continuous-task results. On D'Kitty, we set a new SOTA at 0.972 with a small standard deviation 0.005. On Ant, although CMA-ES surpasses us at the 100th percentile, its standard deviation (0.896) is nearly 64× ours (0.014) and it collapses near zero at the 80th/50th percentiles (see Appendix C.2).
Results on Discrete Tasks: Table 1 (last two columns) present our discrete-task results. Our ROOT achieves both the top rank for the TFBind10 task with a mean score 0.685 and the TFBind8 task with 0.986. Our standard deviation 0.007 for the TFBind8 task is even smaller than that of all other baselines, except for RoMa, which performs poorly.
Results on RNA-Binding Tasks: Table 2 demonstrates ROOT's superior performance, ranking first on all three RNA benchmarks. We outperform Boot-Gen [35] by margins of 0.043 (RNA-A), 0.074 (RNA-B), and 0.136 (RNA-C), while our standard deviations are roughly half those of Boot-Gen, achieving better performance and lower variance (hence, more stable).
Overall, ROOT achieved a mean rank of 1.25 across both discrete and continuous domains, setting a new SOTA on 3 out of 4 Design-Bench tasks, as well as on all three RNA-binding tasks. This demonstrates the robust and consistent effectiveness of ROOT across diverse settings.
this section cite: ['b35']

Section: Ablation Experiments
Selection Strategies for Initial Points. We explore 3 initial point strategies for x + 0 , x - 0 in (Eq. 12 and Eq. 13): random sampling, and selecting points with the lowest or highest objective values from D o . As shown in Table 3, the third strategy achieves the best results.
this section cite: []

Section: Few-Shot Experimental Designs Setting.
In offline optimization, the few-shot experimental designs (ED) setting, introduced in ExPT [45], presents a more challenging task where only a small set of labeled data points,
D label = {(x i , y i )} n l i=1 is available alongside a larger set of unlabeled data, D unlabeled = {x i } nu i=1 .
To evaluate our model in this scenario, we follow ExPT's protocol, using a random 1% of the offline data as labeled points and the remaining 99% as unlabeled. For synthetic function generation, we first fit a prior Gaussian process (GP) to D label , then use the posterior to generate pseudo-labels for D unlabeled . We combine the labeled and pseudo-labeled data and fit another GP to this dataset. The mean function of this refitted GP is used as the synthetic function, following the same procedure as in the main method. As shown in Table 4, ROOT outperforms the ExPT model in this setting by substantial margins.
this section cite: ['b45']

Section: Poor Offline data Coverage Setting.
Beyond the Few-Shot Experimental Design scenario with limited coverage, we further evaluated ROOT's robustness under varying dataset support quality. Specifically, we trained ROOT and the baselines using only the p% lowest-performing designs from the offline dataset. As shown in Table 5, ROOT achieves the best performance across all settings on both TF-Bind-8 and Ant. These results demonstrate that ROOT adapts effectively in challenging scenarios where training data is heavily biased toward low-quality designs.
this section cite: []

Section: Effectiveness of Learning the Probabilistic Bridge Model.
To demonstrate/ablate the effectiveness of our learning probabilistic bridge model (see Section 3.1.1), we further conducted an experiment that runs gradient ascent on GP posterior mean function for comparison. This simple baseline is denoted as GA on GP in Table 1. It is observed that our method significantly outperforms this baseline, confirming the impact of learning probabilistic bridge.
this section cite: []

Section: Number of Gradient Steps (M ).
We experimented with various numbers of gradient steps (M ), from the set {25, 50, 75, 100} to construct X - s and X + s during the data generation phase (see Section 3.3). Our experiments reveal that increasing M consistently improves the overall performance of the algorithm as illustrated in Table 6. Increasing the number of gradient steps allows our model to more precisely distinguish between low-value and high-value regions in the distribution, substantially enhancing our performance. However, increasing the number of gradient steps also increases computational time, so we select M = 100 as the best balance between performance and efficiency.
this section cite: []

Section: Number of Initial (n p ).
For each synthetic function generated by the Gaussian process, we will draw a number of initial data points (n p ) from the offline dataset to initiate the exploration into the low-value and high-value regions via gradient descent and ascent, respectively. We experimented with different numbers of initial points from the set {128, 256, 512, 1024}. As shown in Table 7, increasing the number of initial points n p consistently improves performance. This observation is similar to a previous observation that having more well-curated training data tends to enhance the overall performance. We selected n p = 1024 as the best balance between cost and performance.
this section cite: []

Section: Additional Ablation Studies.
We further performed a series of ablation studies, examining key hyperparameters of our method, alternative strategies for sampling GP kernel parameters, different data sampling techniques beyond GP, the impact of measurement noise, performance on highdimensional continuous tasks, and settings with limited query budgets. For completeness, we also report computational complexity analysis and empirical runtime comparisons of ROOT against baselines. Due to space constraints, all these results are deferred to Appendices C.
this section cite: []

Section: Related Works
Existing approaches in offline optimization can be categorized into three main families: forward modeling, inverse modeling, and learning search policies.
Forward Modeling tackles out-of-distribution (OOD) issues by penalizing high surrogate predictions on OOD inputs [60,13,70,69,14,15,27,46,19,18]. For example, COM [60] identifies OOD regions early during gradient updates and re-trains the surrogate with regularizers to penalize high-value predictions at these inputs. BOSS [14] introduces a sensitivity-aware regularizer for offline optimizers, while ICT, Tri-mentoring [70,13] use co-teaching among surrogates to improve performance.
Inverse Modeling avoids OOD problems by directly learning high-value regions [40,45,39]. For instance, MIN [40] uses model inversion networks to map scores back to inputs, while ExPT [45] combines unsupervised learning and few-shot experimental design for optimizing synthetic functions. DDOM [39] develops a guided diffusion model to generate designs conditioned on function values. The model is trained using weighted sampling from the offline dataset.
Learning Search Policies aims to replicate optimization paths from low-to high-value designs [38,10]. BONET [38] synthesizes trajectories from offline data using a heuristic for monotonic transitions and trains an auto-regressive model. PGS [10] reinterprets offline optimization as a reinforcement learning task, which optimizes for an effective policy using sampled trajectories from offline data.
Overall, these methods remain constrained by the availability of offline data. For instance, DDOM [39] employs guided diffusion to learn an inverse mapping from desired performance outputs to potential input designs. However, the adopted diffusion model must be trained on weighted sampling from the offline data, which may lack critical information regarding potential high-performing input regions that are far from the offline regimes.
To address the challenges posed by limited data, we reframe offline optimization as a distributional translation task. This perspective unveils an intriguing direction: rather than depending solely on scarce high-value observations, one can learn a global translation model by stitching together localized transformation examples between low-and high-value regimes. This allows the optimization process to be guided by a learned probabilistic bridge that generalizes across design landscapes, offering a flexible and data-efficient alternative to traditional surrogate-based methods.
this section cite: ['b60', 'b13', 'b70', 'b69', 'b14', 'b15', 'b27', 'b46', 'b19', 'b18', 'b60', 'b14', 'b70', 'b13', 'b40', 'b45', 'b39', 'b40', 'b45', 'b39', 'b38', 'b10', 'b38', 'b10', 'b39']

Section: Conclusion
We proposed a new perspective on offline black-box optimization by reframing it as a distributional translation task between low-value and high-value input regimes. At the core of this approach is the probabilistic bridge, a model that learns localized transformation flows conditioned on both source and target designs, and synthesizes them into a global translation mechanism. To address the lack of high-value examples in offline datasets, we introduced a synthetic data generation framework that enables pre-training of a meta probabilistic bridge with strong zero-shot generalization. This shifts the focus from modeling the objective function to modeling design transitions, opening new possibilities for data-efficient optimization. Future directions include exploring alternative bridge parameterizations and extending the framework to more complex optimization settings.
this section cite: []

Section: References
Ref_id:b0 Title: Boundary crossing problems and functional transformations for ornstein-uhlenbeck processes Year: (2025)
Ref_id:b1 Title: Robel: Robotics benchmarks for learning with low-cost robots Year: (2020)
Ref_id:b2 Title: Survey of variation in human transcription factors reveals prevalent dna binding changes Year: (2016)
Ref_id:b3 Title: Access: Advancing innovation: Nsf's advanced cyberinfrastructure coordination ecosystem: Services & support Year: (2023)
Ref_id:b4 Title:  Year: ()
Ref_id:b5 Title: Taming extreme heterogeneity via machine learning based design of autonomous manycore systems Year: (2019-10-13)
Ref_id:b6 Title:  Year: (2016)
Ref_id:b7 Title: Conditioning by adaptive sampling for robust design Year: (2019)
Ref_id:b8 Title: Revisiting kernel attention with correlated gaussian process representation Year: (2024)
Ref_id:b9 Title: Arch2030: A vision of computer architecture research over the next 15 years Year: (2016)
Ref_id:b10 Title: Offline model-based optimization via policy-guided gradient search Year: (2024)
Ref_id:b11 Title: Robust guided diffusion for offline black-box optimization Year: (2024)
Ref_id:b12 Title: Bidirectional learning for offline infinite-width model-based optimization Year: (2022)
Ref_id:b13 Title: Xue Steve Liu, and Chris Pal. Parallelmentoring for offline model-based optimization Year: (2024)
Ref_id:b14 Title: Boosting offline optimizers with surrogate sensitivity Year: (2024)
Ref_id:b15 Title: Incorporating surrogate gradient norm to improve offline optimization techniques Year: (2024)
Ref_id:b16 Title: Combining latent space and structured kernels for Bayesian optimization over combinatorial spaces Year: (2021)
Ref_id:b17 Title: Bayesian optimization of nanoporous materials Year: (2021)
Ref_id:b18 Title: Autofocused oracles for model-based design Year: (2020)
Ref_id:b19 Title: Offline model-based optimization via normalized maximum likelihood estimation Year: (2021)
Ref_id:b20 Title: Multi-fidelity Bayesian Optimization of Covalent Organic Frameworks for Xenon/Krypton Separations Year: (2023)
Ref_id:b21 Title: Deep learning in protein structural modeling and design Year: (2020)
Ref_id:b22 Title: Pieter Abbeel, et al. Soft actor-critic algorithms and applications Year: (2018)
Ref_id:b23 Title: The cma evolution strategy: a comparing review. Towards a new evolutionary computation: Advances in the estimation of distribution algorithms Year: ()
Ref_id:b24 Title: Gaussian processes for big data Year: (2013)
Ref_id:b25 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b26 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b27 Title: Learning surrogates for offline black-box optimization via gradient matching Year: (2024)
Ref_id:b28 Title: A generalized stochastic variational Bayesian hyperparameter learning framework for sparse spectrum Gaussian process regression Year: (2017)
Ref_id:b29 Title: Revisiting the sample complexity of sparse spectrum approximation of gaussian processes Year: (2020)
Ref_id:b30 Title: A unifying framework of anytime sparse Gaussian process regression models with stochastic variational inference for big data Year: (2015)
Ref_id:b31 Title: A distributed variational inference framework for unifying parallel sparse Gaussian process regression models Year: (2016)
Ref_id:b32 Title: Collective online learning of Gaussian processes in massive multi-agent systems Year: (2019)
Ref_id:b33 Title: Decentralized high-dimensional bayesian optimization with factor graphs Year: (2018)
Ref_id:b34 Title: Nonmyopic ϵ-Bayes-optimal active learning of Gaussian processes Year: (2014)
Ref_id:b35 Title: Bootstrapped training of score-conditioned generator for offline design of biological sequences Year: (2023)
Ref_id:b36 Title: Offline model-based optimization: Comprehensive review Year: (2025)
Ref_id:b37 Title: Machine learning and manycore systems design: A serendipitous symbiosis Year: (2018)
Ref_id:b38 Title: Generative pretraining for black-box optimization Year: (2022)
Ref_id:b39 Title: Diffusion models for black-box optimization Year: (2023)
Ref_id:b40 Title: Model inversion networks for model-based optimization Year: (2020)
Ref_id:b41 Title: Sparse spectrum Gaussian process regression Year: (2010)
Ref_id:b42 Title: Variational heteroscedastic Gaussian process regression Year: (2011)
Ref_id:b43 Title: Image-to-image translation with brownian bridge diffusion models Year: (2023)
Ref_id:b44 Title: ViennaRNA Package 2.0. Algorithms for Year: (2011-11)
Ref_id:b45 Title: Expt: Synthetic pretraining for few-shot experimental design Year: (2023)
Ref_id:b46 Title: Data-driven offline decision-making via invariant representation learning Year: (2022)
Ref_id:b47 Title: A unifying view of sparse approximate Gaussian process regression Year: (2005)
Ref_id:b48 Title: Approximation methods for gaussian process regression. Large-Scale Kernel Machines Year: (2007)
Ref_id:b49 Title: A unifying view of sparse approximate gaussian process regression Year: (2005-12)
Ref_id:b50 Title: Gaussian Processes for Machine Learning Year: (2006)
Ref_id:b51 Title: FLEXS: Fitness landscape exploration sandbox for biological sequence design Year: (2023)
Ref_id:b52 Title: Rethinking drug design in the artificial intelligence era Year: (2020)
Ref_id:b53 Title: Practical Bayesian optimization of machine learning algorithms Year: (2012)
Ref_id:b54 Title: Scalable bayesian optimization using deep neural networks Year: (2015)
Ref_id:b55 Title: Offline model-based optimization by learning to rank Year: (2025)
Ref_id:b56 Title: Variational learning of inducing variables in sparse Gaussian processes Year: (2009)
Ref_id:b57 Title: Variational inference for Mahalanobis distance metrics in Gaussian process regression Year: (2013)
Ref_id:b58 Title: Mujoco: A physics engine for model-based control Year: (2012)
Ref_id:b59 Title: Design-bench: Benchmarks for data-driven offline model-based optimization Year: (2022)
Ref_id:b60 Title: Conservative objective models for effective offline model-based optimization Year: (2021)
Ref_id:b61 Title: Scientific discovery in the age of artificial intelligence Year: (2023)
Ref_id:b62 Title: Stochastic zeroth-order optimization in high dimensions Year: (2018)
Ref_id:b63 Title: Max-value entropy search for efficient Bayesian optimization Year: (2017)
Ref_id:b64 Title: Bayesian optimization in high dimensions via random embeddings Year: (2013)
Ref_id:b65 Title: Bayesian optimization in a billion dimensions via random embeddings Year: (2016)
Ref_id:b66 Title: Gaussian processes for machine learning Year: (2006)
Ref_id:b67 Title: Simple statistical gradient-following algorithms for connectionist reinforcement learning Year: (1992)
Ref_id:b68 Title: Generative adversarial model-based optimization via source critic regularization Year: (2024)
Ref_id:b69 Title: Robust model adaptation for offline model-based optimization Year: (2021)
Ref_id:b70 Title: Importance-aware coteaching for offline model-based optimization Year: (2023)
Ref_id:b71 Title: Design editing for offline model-based optimization Year: (2025)
Ref_id:b72 Title: Guided trajectory generation with diffusion models for offline model-based optimization Year: (2024)
Ref_id:b73 Title: Near-optimal active learning of multi-output Gaussian processes Year: (2016)
Ref_id:b74 Title: Information-based multi-fidelity bayesian optimization Year: (2017)
