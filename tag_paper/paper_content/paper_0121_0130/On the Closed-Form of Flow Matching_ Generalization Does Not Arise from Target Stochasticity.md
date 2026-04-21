Title: On the Closed-Form of Flow Matching: Generalization Does Not Arise from Target Stochasticity
Abstract: Modern deep generative models can now produce high-quality synthetic samples that are often indistinguishable from real training data. A growing body of research aims to understand why recent methods, such as diffusion and flow matching techniques, generalize so effectively. Among the proposed explanations are the inductive biases of deep learning architectures and the stochastic nature of the conditional flow matching loss. In this work, we rule out the noisy nature of the loss as a key factor driving generalization in flow matching. First, we empirically show that in high-dimensional settings, the stochastic and closed-form versions of the flow matching loss yield nearly equivalent losses. Then, using state-of-the-art flow matching models on standard image datasets, we demonstrate that both variants achieve comparable statistical performance, with the surprising observation that using the closed-form can even improve performance.

Section: Introduction
Recent deep generative models, such as diffusion (Sohl-Dickstein et al., 2015;Ho et al., 2020;Song et al., 2021) and flow matching models (Lipman et al., 2023;Albergo and Vanden-Eijnden, 2023;Liu et al., 2023), have achieved remarkable success in synthesizing realistic data across a wide range of domains. State-of-the-art diffusion and flow matching methods are now capable of producing multi-modal outputs that are virtually indistinguishable from human-generated content, including images (Stability AI, 2023), audio (Borsos et al., 2023), video (Villegas et al., 2022;Brooks et al., 2024), and text (Gong et al., 2023;Xu et al., 2025).
A central question in deep generative modeling concerns the generalization capabilities and underlying mechanisms of these models. Generative models generalization remains a puzzling phenomenon, raising a number of challenging and unresolved questions: whether generative models truly generalize is still the subject of active debate. On one hand, several studies (Carlini et al., 2023;Somepalli et al., 2023b,a;Dar et al., 2023) have shown that large diffusion models are capable of memorizing individual samples from the training set, including licensed photographs, trademarked logos, and sensitive medical data.
On the other hand, Kadkhodaie et al. (2024) have empirically demonstrated that while memorization can occur in low-data regimes, diffusion models trained on a sufficiently large dataset exhibit clear signs of generalization. Taken together, recent work points to a sharp phase transition between memorization and generalization (Yoon et al., 2023;Zhang et al., 2024). Multiple theories have been proposed to explain the puzzling generalization of diffusion and flow matching models. On the one hand, Kadkhodaie et al. (2024); Kamb and Ganguli (2025); Ross et al. (2025) suggested a geometric framework to understand the inductive bias of modern deep convolutional networks on images. On the other hand, Vastola (2025) suggested that generalization is due to the noisy nature of the training loss. In this work, we clearly answer the following question: Does training on noisy/stochastic targets improve flow matching generalization?
If not, what are the main sources of generalization?
Contributions.
• We challenge the prevailing belief that generalization in flow matching stems from an inherently noisy loss (Section 3.1). This assumption, largely supported by studies in low-dimensional settings, fails to hold in realistic high-dimensional data regimes.
• Instead, we observe that generalization in flow matching emerges precisely when the limitedcapacity neural network fails to approximate the optimal closed-form velocity field (Section 3.2).
• We identify two critical time intervals, at early and late times, where neural networks fail to approximate the optimal velocity field (Section 3.3). We show that generalization arises predominantly early along flow matching trajectories, aligning with the transition from the stochastic to the deterministic regime of the flow matching objective.
• Finally, on standard image datasets (CIFAR-10 and CelebA), we show that explicitly regressing against the optimal closed-form velocity field does not impair generalization and can, in some cases, enhance it (Section 4).
The manuscript is organized as follows. Section 2 reviews the fundamentals of conditional flow matching and recalls the closed-form of the "optimal" velocity field. Leveraging the closed-form expression of the flow matching velocity field, Section 3 investigates the key sources of generalization in flow matching. In Section 4, we introduce a learning algorithm based on the closed-form formula. Related work is discussed in detail in Section 5.
this section cite: ['b44', 'b17', 'b47', 'b25', 'b0', 'b27', 'b48', 'b3', 'b53', 'b5', 'b14', 'b54', 'b7', 'b9', 'b21', 'b56', 'b57', 'b21', 'b22', 'b40', 'b52']

Section: Recalls on conditional flow matching
Let p 0 = N (0, Id) be the source distributionfoot_0 and p data the data distribution. We are given n data points x (1) , . . . , x (n) ∼ p data , x (i) ∈ R d . The goal of flow matching is to find a velocity field u : R d × [0, 1] → R d , such that, if one solves on [0, 1] the ordinary differential equation
x(0) = x 0 ∈ R d ẋ(t) = u(x(t), t)(1)
then the law of x(1) when x 0 ∼ p 0 is p data : one says that u transports p 0 to p data . For every value of t between 0 and 1, the law of x(t) defines a probability path, denoted p(•|t) that progressively transforms p 0 to p data . If one knows the velocity field u, new samples can then be generated by sampling x 0 from p 0 , solving the ordinary differential equation, and using x(1) as the generated point.
In conditional flow matching, finding such a velocity field u is achieved in the following way.
(i) First, define a conditioning variable z independent of t, e.g., z = x 1 ∼ p data , (ii) Then, chose a conditional probability path p(•|z, t), e.g., p(•|z = x 1 , t) = N (tx 1 , (1t) 2 Id).
Through the continuity equation (Lipman et al., 2024, Sec. 3.5), the choice (ii) of the conditional probability path p(•|z, t) defines a conditional velocity field u cond (x, z, t). With the choices (i) and (ii), the conditional velocity field writes
u cond (x, z = x 1 , t) = x 1 -x 1 -t .(2)
The choice (ii) of the conditional probability paths p(•|z = x 1 , t) fully defines a probability path p(•|t) (by marginalization against z) and thus defines an optimal velocity field u ⋆ (through the continuity equation), that transports p 0 to p data (Lipman et al., 2023, Thm. 1)
u ⋆ (x, t) = E z|x,t u cond (x, z, t) .(3)
Hence, the optimal velocity u ⋆ could be approximated by a neural network u θ : R d × [0, 1] → R d with parameters θ by minimizing
L FM (θ) = E t∼U ([0,1]) xt∼p(•|t) ∥u θ (x t , t) -u ⋆ (x t , t)∥ 2 .(4)
However, u ⋆ is usually (believed) intractable, as a remedy, Lipman et al. (2023, Thm. 2) showed that L FM (θ) is equal, up to a constant, to the conditional flow matching loss. With the choices (i) and (ii) made above, the conditional flow matching loss reads
L CFM (θ) = E x0∼p0 x1∼p data t∼U ([0,1]) ∥u θ (x t , t) -u cond (x t , z = x 1 , t) = x 1 -x t 1-t =x1-x0 ∥ 2 ,(5)
where x t := (1t)x 0 + tx 1 . The objective L CFM is easy to approximate, since it is easy to sample from p 0 = N (0, Id) and U([0, 1]); sampling from p data is approximated by sampling from pdata := 1 n n i=1 δ x (i) . Although it seems natural, replacing p data by pdata in (5) has a very important consequence: it makes the minimizer û⋆ of L FM available in closed-form, which we recall below.
Proposition 1 (Closed-form Formula of the Optimal Velocity). When p data is replaced by pdata , with the previous choices (i) and (ii), the optimal velocity field û⋆ in (3) has a closed-form formula:
û⋆ (x, t) = n i=1 λ i (x, t) x (i) -x 1 -t ,(6)
with λ(x, t) = softmax((
-∥x-tx (j) ∥ 2 2(1-t) 2 ) j=1,...,n ) ∈ R n .
The notation û⋆ emphasizes the velocity field is optimal for the empirical probability distribution pdata , not the true one p data . Since u cond (x, z = x (i) , t) ∝ x (i)x, the optimal velocity field û⋆ is a weighted average of the n different directions x (i)x. Note that the closed-form formula in Equation ( 6) can be found in various previous works, e.g., Kamb and Ganguli (2025, Eq. 3), Biroli et al. (2024), Gao and Li (2024), Li et al. (2024) or Scarvelis et al. (2025), and can be generalized to other choices of continuous distribution p 0 (e.g., the uniform distribution, see Appendix A.1).
From Equation ( 6), as t → 1, the velocity field û⋆ diverges at any point x that does not coincide with one of the training samples x (i) , and it points in the direction of the nearest x (i) . This creates a paradox: solving the ordinary differential equation ( 1) with the velocity field û⋆ can only produce training samples x (i) (see Gao and Li 2024, Thm. 4.6 for a formal proof). Therefore, in practice, exactly minimizing the conditional flow matching loss would result in u θ = û⋆ , meaning the model memorizes the training data and fails to generalize. This naturally yields the following question:
How can flow matching generalize if the optimal velocity field only generates training samples?
this section cite: ['b25', 'b1', 'b13', 'b24', 'b42']

Section: Investigating the key sources of generalization
In this section, we investigate the key sources of flow matching generalization using the closed-form formula of its velocity field. First in Section 3.1 we challenge the claim that generalization stems from the stochastic approximation u cond of the optimal velocity field û⋆ . Then, in Section 3.2 we show that generalization arises when u θ fails to approximate the perfect velocity û⋆ . Interestingly, the target velocity estimation particularly fails at two critical time intervals. Section 3.3 shows that one of these critical times is particularly important for generalization. 0.0 0.5 1.0 2 moons t = 0.01 t = 0.21 t = 0.40 t = 0.60 t = 0.79 t = 0.99
-1 0 1 0.0 0.5 1.0 CIFAR-10 -1 0 1 -1 0 1 -1 0 1 -1 0 1 -1 0 1 (a) Non stochasticity of û⋆ for high-dimensional real data. û ⋆ (x, t) = n ∑ i=1 p ( z = x (i) | x, t ) u cond ( x, t, z = x (i) ) û ⋆ x û ⋆ x Common belief STOCHASTICITY What really happens NON-STOCHASTICITY (b) Stochasticity vs. non-stochasticity (c) Dimension Dependence
Figure 1: We challenge the hypothesis that target stochasticity plays a major role in flow matching generalization. In Figure 1a, the histograms of the cosine similarities between û⋆ ((1t)x 0 + tx 1 , t) and u cond ((1t)x 0 + tx 1 , z = x 1 , t) = x 1x 0 are displayed for various time values t and two datasets. For real, high-dimensional data, non-stochasticity arises very early (before t = 0.2 for CIFAR-10 with dimension (3, 32, 32)). Figure 1c displays the alignment between û⋆ and u cond over time for varying image dimensions d on Imagenette.
this section cite: []

Section: Target stochasticity is not what you need
One recent hypothesis is that generalization arises from the fact that the regression target u cond of conditional flow matching is only a stochastic estimate of û⋆ . The fact that the target regression objective only equals the true objective on average is referred to by Vastola (2025) as "generalization through variance". To challenge this assumption, we leverage Proposition 1, which states that the optimal velocity field û⋆ (x, t) is a weighted sum of the n values of u cond (x, t, z = x (i) ) = x (i) -x 1-t , for i ∈ [n], and show that, after a small time value t, this average is in practice equal to a single value in the expectation (see Figures 1a and 1b).
Comments on Figure 1a. To produce Figure 1a, we sample 256 pairs (x 0 , x 1 ) from p 0 × pdata . For each value of t, we compute the cosine similarity between the optimal velocity field û⋆ ((1t)x 0 + tx 1 , t) and the conditional target u cond ((1t)x 0 + tx 1 , z = x 1 , t) = x 1x 0 . The resulting similarities are aggregated and shown as histograms. The top row displays the results for the two-moons toy dataset (d = 2), and the bottom row displays the results for the CIFAR-10 dataset (Krizhevsky and Hinton 2009, d = 3072); n = 50k for both. As t increases, the histograms become increasingly concentrated around 1, indicating that û⋆ aligns closely with a single conditional vector u cond . From Equation (6), this corresponds to a collapse towards 0 of all but one of the softmax weights λ i (x t , t). This time corresponds to the collapse time studied by Biroli et al. (2024) for diffusion; we discuss the connection in the related works (Section 5). On the two-moons toy dataset, this transition occurs for intermediate-to-large values of t, echoing the observations made in low-dimensional settings by Vastola (2025, Figure 1). In contrast, for high-dimensional real datasets, û⋆ (x, t) aligns with a single conditional velocity field x (i) -x, even at early time steps, suggesting that the non-stochastic regime dominates most of the generative process. This key difference between lowand high-dimensional data suggests that the transition time between the stochastic and non-stochastic regimes is strongly influenced by the dimensionality of the data.
# samples 10 # samples 100 # samples 1000 # samples 2000 # samples 3000 # samples 4000 # samples 5000 # samples 10000 0 1 t 0.0 0.5 Comments on Figure 1c. To further illustrate the strong impact of dimensionality, Figure 1c reports the proportion of samples x t (from a batch of 256) for which the cosine similarity between û⋆ and u cond ∝ x (i)x exceeds 0.9, as a function of time t. This analysis is performed across multiple spatial resolutions of the Imagenette dataset (Howard, 2019), obtaining dim × dim images by spatial subsampling. Figure 1c reveals a sharp transition: as the dimensionality increases, the proportion of high-cosine matches rapidly converges to 100%. A practical implication of this behavior is that, for sufficiently large t, if x 0 ∼ p 0 and x (i) ∼ pdata , then û⋆ ((1t)x 0 + tx (i) , t) is approximately proportional to x (i)x. Consequently, regressing on x (i) or on the conditional velocity x 1x 0 becomes effectively equivalent. Section 4 investigates how to learn regressing against optimal velocity field û⋆ , and empirically shows similar results between stochastic and non-stochastic targets.
E xt ||u θ (x t , t) -û (x t , t)|| 2 1 0 1 0 0 1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0 5 0 0 0 1 0 0 0 0 # samples 0 1000 2000 FID DINO: Test 1 0 1 0 0 1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0 5 0 0 0 1 0 0 0 0 # samples 0 10 18 Nearest Neighbor Dist.
The regime where flow matching matches stochasticity is mostly concentrated on a very short time interval, for small values of t. We hypothesize that the phenomenon observed here on the optimal velocity field û⋆ has major implications on the learned flow matching model u θ , which we further inspect in the next section.
this section cite: ['b52', 'b1', 'b18']

Section: Failure to learn the optimal velocity field
This subsection investigates how well the learned velocity field u θ approximates the optimal/ideal velocity field û⋆ , and how the quality of this approximation correlates with generalization. To do so, we propose the following experiment.
this section cite: []

Section: Set up of Figure 2.
To build Figure 2, we subsampled the CIFAR-10 dataset from 10 to 10 4 samples.
For each size, we trained a flow matching model using a standard 34 million-parameter U-Net (see Appendix D for details). Following Kadkhodaie et al. (2024), the number of parameters of the network u θ remains fixed across dataset sizes. Importantly, the optimal velocity field û⋆ itself depends on the dataset size: as the number of samples increases, the complexity of û⋆ also grows. Thus, we expect the network u θ to accurately approximate the optimal velocity field û⋆ for smaller dataset sizes.
Comments on Figure 2. The leftmost plot shows the average training error
E x0∼p0 x1∼ pdata ∥u θ (x t , t) -û⋆ (x t , t)∥ 2 , where x t := (1 -t)x 0 + tx 1 ,
between the learned velocity u θ and the optimal empirical velocity field û⋆ , evaluated across multiple time values t and dataset sizes. With only 10 samples (darkest curve), the network u θ closely approximates û⋆ . As the dataset size increases, the complexity of û⋆ grows, and the approximation by u θ becomes less accurate. In particular, the approximation fails at two specific time intervals: around t ≈ 0.15 and near t = 1. The failure near t = 1 is expected, as û⋆ becomes non-Lipschitz at t = 1. Interestingly, the early-time failure at t ≈ 0.15 corresponds to the regime where û⋆ and u cond start to correlate (see Figure 1a in Section 3.1). The middle plot of Figure 2 reports the FID-10k, computed on the test set in the DINOv2 embedding space (Oquab et al., 2024), for various dataset sizes. For a small dataset (e.g., #samples = 10), u θ approximates û⋆ well but does not generalize -the test FID exceeds 10 3 . As the dataset size increases (1000 ≤ #samples ≤ 3000), the approximation u θ becomes less accurate. Despite this, the model achieves lower FID scores on the test set but still memorizes the training data. The rightmost plot of Figure 2 illustrates this memorization by showing the average distance between each generated sample and its nearest neighbor in the training set. For larger datasets (#samples ≥ 3000), this distance increases substantially, indicating that the model generalizes better. Overall, Figure 2 also suggests that the FID metric can be misleading, even when computed on the test set. For example, the model trained with 1000 samples has a low test FID but memorizes training examples.
Figure 2 confirms that generalization arises when the network u θ fails to estimate the optimal velocity field û⋆ , and that this failure occurs at two specific time intervals. In Section 3.3, we investigate which of these two intervals is responsible for driving generalization.
this section cite: ['b21']

Section: When does generalization arise?
To investigate whether the failure to approximate û⋆ matters the most at small or large values of t, we carry out the following experiment.
Set up of Figure 3. We first learn a velocity field u θ using standard conditional flow matching (see Appendix D), then we construct a hybrid model: we define a piecewise trajectory where the flow is governed by the optimal velocity field û⋆ for times t ∈ [0, τ ], and by the learned velocity field u θ for times t ∈ [τ, 1], for a given threshold parameter τ ∈ [0, 1]. For the extreme case τ = 1, the full trajectory follows û⋆ , and samples exactly match training data points. Conversely, when τ = 0, the entire trajectory is governed by u θ , yielding novel samples. Intermediate values of τ produce a mixture of both behaviors, which we interpret as reflecting varying degrees of generalization. To assess generalization, we measure the distance of generated samples to the dataset using the LPIPS metric (Zhang et al., 2018), which computes the feature distance between two images via some pretrained classification network. We define the distance of a generated sample x to a dataset D = {x (1) , . . . , x (n) } as dist(x, D) = min x (i) ∈D LPIPS(x, x (i) ). We fix a random batch of 256 pure noise images from p 0 . Then, for various threshold values τ , we generate 256 images with the hybrid model, always starting from this batch. Finally, we measure the creativity of the hybrid model as the mean of the aforementioned LPIPS distances between the 256 generated samples and the dataset.
Comments on Figure 3. The top row displays the LPIPS distances as τ varies, on the CIFAR-10 (left) and CelebA -64 × 64 (right) datasets. For τ ≤ 0.2, the hybrid model remains as creative as u θ , despite following û⋆ in the first steps. For τ > 0.2, the LPIPS distance starts dropping. On the displayed generated samples (bottom rows), we in fact see that as soon as τ ≥ 0.4, the sample generated by the hybrid model is almost the same as the one obtained with û⋆ (τ = 1). This means the final image is already determined at t = 0.4, and despite the generalization capacity of the learned velocity field u θ , following it only after t ≥ 0.4 is not enough to create a new image: generalization occurs early and cannot fully be explained by the failure to correctly approximate u ⋆ at large t.
Although we have shown that the stochastic phase was limited to small values of t in real-data settings, we have not yet definitively ruled it out as the cause of generalization. In the following Section 4, we introduce a learning procedure designed to address this question directly.
4 Learning with the closed-form formula
In this section, in order to discard the impact of stochastic target on the generalization, we propose to directly regress against the closed-form formula in Equation ( 6).
this section cite: ['b58']

Section: Empirical flow matching
Regressing against the closed-form û⋆ , defined in Equation ( 6), at a point (x t , t) requires computing a weighted sum of the conditional velocity fields over all the n training points x (i) . For a dataset of n samples of size d, and a batch of size |B|, computing the weights of the exact closed-form formula û⋆ (x, t) of flow matching requires O(n × |B| × d). These computations are prohibitive since they must be performed for each batch. One natural idea is to estimate the closed-form formula û⋆ (Equation ( 6)), by a Monte Carlo approximation (Equation ( 8)), using M ≤ n samples b (1) , . . . , b (M ) :
L EFM (θ) = E x0∼p0 x1∼ pdata t∼U ([0,1]) b (2) ,...,b (M ) ∼ pdata ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2 , (7
) with x t = (1 -t)x 0 + tx 1 , b (1) := x 1 , and û⋆ M (x, t) = M j=1 λ(x, t) b (j) -x 1 -t , λ(x, t) = softmax - ∥x -tx (l) ∥ 2 2(1 -t) 2 l=1,...,n . (8)
The formulation in Equation ( 7) may appear naive at first glance. Still, it hinges on a crucial trick: the Monte Carlo estimate is computed using a batch that systematically includes the point x 1 , that generated the current x t . If instead b (1) were sampled independently from pdata , this could introduce a sampling bias (see Ryzhakov et al. 2024, Appendix B, and the corresponding OpenReview commentsfoot_1 for an in-depth discussion). Proposition 2 shows that the estimate û⋆ M is unbiased and has lower variance than the standard conditional flow matching target.
Proposition 2. We denote the conditional probability distribution p(z = x (i) | x, t) over {x (i) } n i=1 by pdata (z | x, t). With no constraints on the learned velocity field u θ , i) The minimizer of Equation (7) writes, for all (x, t)
E b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] .(9)
ii) In addition, for all (x, t), the minimizer of Equation (7) equals the optimal velocity field, i.e.,
E b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] = û⋆ (x, t) .(10)
iii) The conditional variance of the estimator û⋆ M is smaller than the usual conditional variance:
Var b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] ≤ Var b (1) ∼ pdata (•|x,t) u cond (x, b (1) , t) .(11)
The proof of Proposition 2 is provided in Appendix B.3. The estimator û⋆ M of the optimal field û⋆ is closely related to self-normalized importance sampling (see Appendix B.2 and Owen 2013, Chap. 9.2), as well as to Rao-Blackwellized estimators (Casella and Robert, 1996;Cardoso et al., 2022). As discussed in Ryzhakov et al. (2024), self-normalized importance sampling estimators of û⋆ are generally biased, in the sense that:
E b (1) ,...,b (M ) ∼ pdata û⋆ M (x t , t) ̸ = û⋆ (x t , t) .
A key insight is that our estimator includes b (1) ∼ pdata (• | x t , t), which leads to the main result of Proposition 2. In Section 4.2, we demonstrate that Algorithm 2, designed to solve Equation ( 7), yields consistent improvements on high-dimensional datasets such as CIFAR-10 and CelebA. Additional details on the unbiasedness of L EFM can be found in the supplementary material (Appendix B). From a computational perspective, despite requiring M additional samples, Algorithm 2 remains significantly more efficient than increasing the batch size by a factor of M : the M samples are merely averaged (with weights), while the backpropagation remains identical to that of Algorithm 1.
Algorithm 1 Vanilla Flow Matching for k in 1, . . . , n iter do t ∼ U([0, 1]) x 0 ∼ N (0, Id), x 1 ∼ pdata , x t = (1 -t)x 0 + tx 1 u cond (x t , t) = x 1 -x t 1 -t = x 1 -x 0 L(θ) = u θ (x t , t) -u cond (x t , t) 2 Compute ∇L(θ) and update θ return u θ Algorithm 2 Empirical Flow Matching param :M // Number of samples in the empirical mean for k in 1, . . . , n iter do x 0 ∼ N (0, Id), x 1 ∼ pdata , t ∼ U([0, 1]) x t = (1 -t)x 0 + tx 1 b (1) = x 1 ∀j ∈ 2,M , b (j) ∼ pdata // Samples from pdata û⋆ M (x t , t) = M j=1 b (j) -xt 1-t • softmax -∥xt-t•b∥ 2 2(1-t) 2 j L(θ) = ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2
Compute ∇L(θ) and update θ return u θ
this section cite: ['b41', 'b8', 'b6', 'b41']

Section: Experiments
We now learn with empirical flow matching (EFM, Equation ( 7) and Algorithm 2) in practical high-dimensional settings. Our goal with this empirical investigation is first to observe if regressing against a more deterministic target leads to performance improvement/degradation.
this section cite: []

Section: Datasets and Models.
We perform experiments on the image datasets CIFAR-10 ( Krizhevsky and Hinton, 2009) and CelebA 64 × 64 (Liu et al., 2015). For the experiments, we compare vanilla conditional flow matching (Lipman et al., 2023;Liu et al., 2023;Albergo and Vanden-Eijnden, 2023), optimal transport flow matching (Pooladian et al., 2023;Tong et al., 2024), and the empirical flow matching in Algorithm 2, for multiple numbers of samples M to estimate the empirical mean. Training details are in Appendix D.
this section cite: ['b23', 'b28', 'b25', 'b27', 'b0', 'b37', 'b51']

Section: Metrics.
To assess generalization performance, we use the standard Fréchet Inception Distance (Heusel et al., 2017) with Inception-V3 (Szegedy et al., 2016) but we also follow the recommendation of Stein et al. (2023) using the DINOv2 embedding (Oquab et al., 2023), which is known to a more expressive and discriminative embedding, that leads to a less biased evaluation. We also measure the FID between the generated and the train and test sets, rather than only on the training set, as is often done in generative modeling benchmarks. On Figure 2, we also displayed a memorization metric that would detect a pure copy of the training set. Overall, defining and quantifying the generalization ability of generative models is overall a challenging task: train and test FID are known to be imperfect (Stein et al., 2023;Jiralerspong et al., 2023;Parmar et al., 2022), yet no superior competitor has emerged. Regressing against a more deterministic target (EFM -128, 256, 1000) does not yield performance decreases. On the contrary, the more deterministic the target, the better the performance.
Comments on Figure 4. Figure 4 compares vanilla flow matching, OTCFM, and the empirical flow matching (EFM, Algorithm 2) approaches using various numbers of samples to estimate the empirical mean, M ∈ {128, 256, 1000}. First, we observe that learning with a more deterministic target does not degrade either training or testing performance, across both types of embeddings. On the contrary, we consistently observe modest but steady improvements as stochasticity is reduced. For both CIFAR-10 and CelebA, increasing the number of samples M used to compute the empirical mean-i.e., , making the targets less stochastic-leads to more stable improvements. It is worth noting that Algorithm 2 has a computational complexity of O(M × |B| × d), where |B| is the batch size, M is the number of samples used to estimate the empirical mean, and d is the sample dimension.
In our experiments, choosing M = |B| = 128 yielded a modest time overhead. For empirical flow matching, we experimented with several values beyond M = 1000 (e.g., M = 2000, M = 5000).
The results were nearly identical to those obtained with M = 1000, with curves being visually indistinguishable. Therefore, we chose not to report results for M ≥ 1000.
this section cite: ['b16', 'b50', 'b49', 'b33', 'b49', 'b20', 'b36']

Section: Related work
The existing literature related to our study can be roughly divided into three approaches: leveraging the closed-form, studies on the memorization vs generalization, and characterization of the different phases of the generating dynamics.
Leveraging the closed-form. Proposition 1 has been leveraged in several ways. The closest existing work is by Ryzhakov et al. (2024), who propose to regress against û⋆ as we do in Section 4. Nevertheless, their motivation is that reducing the variance of the velocity field estimation makes learning more accurate: as explained in Section 3.1, we argue this claim rests on misleading 2Dbased intuitions (e.g., Figure 1, challenged by Section 3.1). The idea of regressing against a more deterministic target (as Proposition 2 shows) derived from the optimal closed-form velocity field has also been empirically explored for diffusion models (Xu et al., 2023). Scarvelis et al. (2025) bypass training, and suggest using a smoothed version of û⋆ to generate novel samples. In a work specific to images and convolutional neural networks, Kamb and Ganguli (2025) suggested that flow matching indeed ends up learning an optimal velocity, but that instead of memorizing training samples, the velocity memorizes a combination of all possible patches in an image and across the images. They show remarkable agreement between their theory and the trajectories followed by learned vector fields, but their work is limited to convolutional architectures, and was recently extended to a larger class of architectures (Lukoianov et al., 2025).
Memorization and reasons for generalization. Kadkhodaie et al. (2024) directly relates the transition from memorization to generalization to the size of the training dataset, and proposes a geometric interpretation. We provide a complementary experiment in Section 3.2, quantifying how much the network fails to estimate the optimal velocity field. Gu et al. (2025) provide a detailed experimental investigation into the potential causes for generalization, primarily based on the characteristics of the dataset and choices for training and model. Vastola (2025) explores different factors of generalization in the case of diffusion, with a special focus on the stochasticity of the target objective in the learning problem. Through a physic-based modeling of the generative dynamics, they study the covariance matrix of the noisy estimation of the exact score. In our work, we believe that we have shown that this claim was not valid for real high-dimensional data. Niedoba et al. (2025) study the poor approximation of the exact score by the learned models: like Kamb and Ganguli (2025), they suggest that the generalization of the learned models comes from memorization of many patches in the training data.
Temporal regimes. Biroli et al. (2024); Sclocchi et al. (2025) provide an analysis of the exact score, the counterpart of the exact velocity field for diffusion. For a multimodal target distribution, the authors identify three phases (we keep the convention that t = 0 is noise and t = 1 is target): for t < t 1 , all trajectories are indistinguishable; for t 1 < t < t 2 , trajectories converging to different modes separate; for t > t 2 , trajectories all point to the training dataset. In the case of Gaussians mixtures target, they highlight the dependency of t 2 in the dimension and the number of samples, in O ((log n)/d), meaning that the first phases are observable only if the number of training points is exponential in the dimension. The methodology they adopt to validate the existence of such t 2 on real data relies on the stochasticity of the backward generative process, which does not hold in the case of flow matching. Our experiments on learned flow matching models allow us to take this theoretical study on memorization and temporal behaviors of generative processes a step further.
6 Conclusion, limitations and broader impact Conclusion. By challenging the assumption that stochasticity in the loss function is a key driver of generalization, our findings help clarify the role of approximation of the exact velocity field in flow matching models. Beyond the different temporal phases in the generation process that we have identified, we expect further results to be obtained by uncovering new properties of the true velocity field.
this section cite: ['b41', 'b55', 'b42', 'b22', 'b29', 'b21', 'b15', 'b52', 'b32', 'b22', 'b1', 'b43']

Section: Limitation.
Our work is mainly empirical, with a focus on learned models, but did not precisely characterize the learned velocity field, in particular, how it behaves outside the trajectories defined by the optimal velocity. Leveraging existing work on the inductive biases of the architectures at hand seems like a promising venue. Another limitation is that we did not investigate the interaction between the architectural inductive bias, and optimization procedures: this is a very challenging, but active area of research (Boursier and Flammarion, 2025;Bonnaire et al., 2025;Favero et al., 2025).
Broader impact. We hope that identifying the key factors of generalization will lead to improved training efficiency. However, generative models also raise concerns related to misinformation (notably deepfakes), data privacy, and potential misuse in generating synthetic but realistic content.
this section cite: ['b4', 'b2', 'b10']

Section: NeurIPS Paper Checklist
The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.
Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:
• You should answer • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available.
• Please provide a short (1-2 sentence) justification right after your answer (even for NA).
The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.
The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist", • Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: each claim of the abstract refers to a specific subsection of the paper, that provide empirical evidence of the claim.
Guidelines: • The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: We do have a specific section for the limitation of our work
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: all results are encapsulated in clearly defined statements, and proofs are provided in appendix.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes] Justification: We provided as many details as possible in order to reproduce the results, in particular, we refer to the public implementation we used, including the specific (default) parameters used.
Guidelines: • The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Code will be made available along with publication
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We provide a specific appendix with the experimental details Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We do not report error bars, however, we do specify the number of samples used for the FID computation and highlight the strong weaknesses of the FID metric.
Guidelines: • The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: we specified what type of GPU we used Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: [NA] Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes]
Justification: there is a dedicated broader impact section
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [No] Justification: We work on standard image datasets Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We properly refer the torchcfm and PnPflow codebase.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: [NA] Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: [NA] Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16.
Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [No] Justification: LLMs were only used for grammatical purposes. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described. When z ∼ p data , it is difficult to sample from z|x t , t, but the latter equation can be rewritten as u ⋆ (x t , t) = z u cond (x t , z, t) p(z|x t , t) p(z) p(z)dz (32) and one can easily sample from z ∼ pdata using the empirical data distribution x (1) , . . . , x (n) u ⋆ (x t , t) ≈ 1 n n i=1 u cond (x t , x (i) , t) p(z = x (i) |x t , t) p(x (i) )
= n i=1 u cond (x t , x (i) , t)p(z = x (i) |x t , t)(33)
:= û⋆ (x t , t) .
this section cite: []

Section: B.3 Proof of Proposition 2
We first recall Appendix B, which we prove in this section. Proposition 2. We denote the conditional probability distribution p(z = x
(i) | x, t) over {x (i) } n i=1 by pdata (z | x, t). With no constraints on the learned velocity field u θ , i) The minimizer of Equation (7) writes, for all (x, t) E b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] .
ii) In addition, for all (x, t), the minimizer of Equation (7) equals the optimal velocity field, i.e.,
E b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] = û⋆ (x, t) .(10)
iii) The conditional variance of the estimator û⋆ M is smaller than the usual conditional variance:
Var b (1) ∼ pdata (•|x,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x, t)] ≤ Var b (1) ∼ pdata (•|x,t) u cond (x, b (1) , t) .(11)
Proof of Item (i)). With no constraints on u θ , the empirical flow matching loss writes:
E t∼U ([0,1]) x1∼ pdata xt=(1-t)x0+tx1 b (1) :=x1 ; b (2) ,...,b (M ) ∼ pdata ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2 , (36
) = E t∼U ([0,1]) xt∼pt E b (1) ∼ pdata (•|xt,t) b (2) ,...,b (M ) |xt,t ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2 ,(37)
= E t∼U ([0,1]) xt∼pt E b (1) := pdata (•|xt,t) b (2) ,...,b (M ) ∼ pdata ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2 because b (2) , . . . , b (M ) ⊥ ⊥ x t , t , (38
)
which is minimized when for all x t , t u
θ (x t , t) = E b (1) ∼ pdata (•|xt,t) b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x t , t)] .(39)
Proof of Item (ii)). The minimizer for a given (x t , t), removing these elements from the notation for conciseness and abstraction, is a weighted mean:
û⋆ (x t , t) = û⋆ = n l=1 w (l) u (l) , with(40)
w (l) = pdata (z = x (l) |t, x t ) , n l=1 w (l) = 1(41)
u (l) = u cond (x t , x (l) , t)(42)
this section cite: []

Section: 
A Proofs of Section 2 û⋆ (x, t) = n i=1 u cond (x, z = x (i) , t) • p(x|z = x (i) , t)
n i ′ =1 p(x|z = x (i ′ ) , t) .(12)
this section cite: []

Section: A.1 Proof of Proposition 1
Proof.
• In the case where z ∼ pdata , conditional probability writes p(z = x (i) |x, t) = p(x, t, z = x (i) ) p(x, t)
= p(x|t, z = x (i) )p(t, z = x (i) ) p(x, t) ((13)
) = p(x|t, z = x (i) )p(t, z = x (i) ) n i ′ =1 p(x, t, z = x (i ′ ) ) (14
) = p(x|t, z = x (i) )p(t) 1 n p(z = x (i) ) n i ′ =1 p(x|t, z = x (i ′ ) )p(t) p(z = x (i ′ ) ) 1 n (15
) = p(x|t, z = x (i) ) n i ′ =1 p(x|t, z = x (i ′ ) ) .16
Pluging Equation ( 17) in Equation (3) yields the closed-formed formula for the velocity field:
u ⋆ (x, t) = n i=1 u cond (x, t, z = x (i) )p(z = x (i) |x, t)(18)
= n i=1 u cond (x, t, z = x (i) ) p(x|t, z = x (i) ) n i ′ =1 p(x|t, z = x (i ′ ) ) . (19
)
which proves Equation (12); using that x|t, z = x (i) ∼ N (tx (i) , (1t) 2 Id) and u cond (x, t, z = x (i) ) = x (i) -x 1-t yields Equation ( 6).
• For the case z ∼ p 0 × pdata ,
û⋆ (x, t) := z u cond (x, t, z)p(z|x, t) dz (20
) = z u cond (x, t, z) p(x, z, t) p(x, t) dz (21
) = z u cond (x, t, z) p(x|z, t)p(z)p(t) z ′ p(x|t, z ′ )p(t)p(z ′ ) dz ′ dz (22
) = z u cond (x, t, z) p(x|z, t)p(z) z ′ p(x|t, z ′ )p(z ′ ) dz ′ dz (23
)
Since z ∼ p 0 × pdata , the denominator is equal to:
z ′ p(x|t, z ′ )p(z ′ ) dz ′ = 1 n x0 n i=1 δ x ((1 -t)x 0 + tx (i) ) 1 (2π
) d exp(- 1 2 x 2 0 )dx 0 (24
) = 1 n y n i=1 δ x (y) 1 (2π
) d exp(- 1 2(1 -t) 2 ∥y -tx (i) ∥ 2 ) 1 (1 -t) d dy (y = (1 -t)x 0 + tx (i) ) (25
) = 1 n n i=1 1 (2π(1 -t) 2 ) d exp - 1 2(1 -t) 2 ∥x -tx (i) ∥ 2(26)
Likewise, the numerator equals:
z u cond (x, t, z)p(x|z, t)p(z) dz = x0 1 n n i=1 (x (i) -x 0 )δ x ((1 -t)x 0 + tx (i) ) 1 (2π
) d exp(- 1 2 ∥x 0 ∥ 2 )dx 0 (27
) = 1 n n i=1 y x (i) -y 1 -t δ x (y) 1 (2π(1 -t) 2 ) d exp(- 1 2(1 -t) 2 ∥y -tx (i) ∥ 2 )dy (28) = n i=1 x (i) -x 1 -t 1 (2π(1 -t) 2 ) d exp - 1 2(1 -t) 2 ∥x -tx (i) )∥ 2 (29)
Taking the ratio of Equations ( 24) and (29) concludes the proof.
this section cite: []

Section: B Additional details and comments on empirical flow matching
First, recalls on the optimal velocity (Equation ( 6)) and the empirical flow matching loss (Equations (7) and ( 8)) are provided in Appendix B.1. The unbiasedness of the estimator is presented in Appendix B.2, and its proof is in Appendix B.3.
this section cite: []

Section: B.1 Recalls
The closed-form formula of the "optimal" velocity field is:
û⋆ (x, t) = n l=1 x (l) -x 1 -t • softmax - ∥x -tx (k) ∥ 2 2(1 -t) 2 k=1,...,n l .(6)
The proposed loss uses mini-batches of size M (instead of all n training points) to build an estimator û⋆ M of û⋆ :
L EFM (θ) = E t∼U ([0,1]) x0∼p0 x1∼ pdata xt=(1-t)x0+tx1 b (1) :=x1 ; b (2) ,...,b (M ) ∼ pdata ∥u θ (x t , t) -û⋆ M (x t , t)∥ 2 , with û⋆ M (x t , t) = M j=1 b (j) -x t 1 -t • softmax - ∥x t -tb (k) ∥ 2 2(1 -t) 2 k=1,...,M j . (7)(8)
Crucially, in Equation ( 7) the sample b (1) depends on x t and is reused in the estimate û⋆ M . This important detail yields an unbiased estimator of û⋆ .
this section cite: []

Section: B.2 Theoretical properties of the proposed estimator
First, we discuss below the relation between Proposition 2 and the sampling literature.
Links with importance sampling. The estimator û⋆ in Equation ( 6) can be seen as a form of importance sampling (see Robert et al. 1999, Chap. 3 for an in-depth reference). In a nutshell, importance sampling is a way to estimate an expectation when one cannot easily sample from the random variable it depends on. More precisely, in the ideal case z ∼ p data (as opposed to z ∼ pdata ), the velocity field formula is the following
u ⋆ (x t , t) = E z|xt,t u cond (x t , z, t)(30)
= z u cond (x t , z, t)p(z|x t , t)dz .(31)
We express a mini-batch as an M -valued vector of indices, i ∈ 1, n M . The mini-batch estimate from Equation ( 7), considering the definition of the softmax, can be expressed as a mini-batch weighted-mean:
û⋆ M (i) = M j=1 w (i j ) u (i j ) M j=1 w (i j )(43)
The categorical distribution over 1, n with probabilities following the weights w in ( 41) is denoted Cat(w) and the uniform distribution, i.e., Cat(1/n)), is denoted Unif.
The main result of the following is that, in expectation over the biased-mini-batches, where the first point is drawn according to w and the M -1 other points are drawn uniformly, the mini-batch weighted-mean is an unbiased estimate of the w-weighted-mean û⋆ .
E [û ⋆ M (i)] := E i1∼Cat(w) E i2,...,i M ∼Unif [û ⋆ M (i)](44)
= n i1=1 w (i1) E i2,...,i M ∼Unif [û ⋆ M (i)] (45
) = n i1=1 E i2,...,i M ∼Unif w (i1) û⋆ M (i) (46
) = n n i1=1 1 n E i2,...,i M ∼Unif w (i1) û⋆ M (i) (47
) = n E i1∼Unif E i2,...,i M ∼Unif w (i1) û⋆ M (i)(48)
= n E i1,...,i M ∼Unif w (i1) û⋆ M (i)(49)
The expression in Equation ( 49) is invariant with respect to order of the indices i 1 , . . . , i M : the indices in expectation in Equation ( 49) can be exchanged, and one thus has
∀k ∈ 1, M , E [û ⋆ M (i)] = n E i1,...,i M ∼Unif w (i k ) û⋆ M (i) .(50)
Averaging Equation (50) over the indices k ∈ 1, M yields the desired result
1 M M k=1 Eû ⋆ M (i) = 1 M M k=1 n E i1,...,i M ∼Unif w (i k ) û⋆ M (i) (51
)
Eû ⋆ M (i) = 1 M n E i1,...,i M ∼Unif M k=1 w (i k ) û⋆ M (i) (52) = 1 M n E i1,...,i M ∼Unif M k=1 w (i k ) M j=1 w (i j ) u (i j ) M j=1 w (i j ) (53) = 1 M n E i1,...,i M ∼Unif   M k=1 w (i k ) M j=1 w (i j ) u (i j ) M j=1 w (i j )   (54) = 1 M n E i1,...,i M ∼Unif   M j=1 w (i j ) u (i j )   (55
) = 1 M n M j=1 E i1,...,i M ∼Unif w (i j ) u (i j ) (56) = 1 M n M j=1 E ij ∼Unif w (i j ) u (i j ) (57) = 1 M n M E l∼Unif w (l) u (l) (58
) = n E l∼Unif w (l) u (l) (59
) = n n l=1 1 n w (l) u (l) (60
) = n l=1 w (l) u (l) (61) = û⋆ (62
)
Proof of Item (iii)). Using the same ideas as for Item (ii)), one has
E x (1) ∼ pdata (•|xt,t) ; b (2) ,...,b (M ) ∼ pdata û⋆ M (x t , t) 2 (63) = nE i1,...,i M ∼Unif w (i1) û⋆ M (i) 2 (64) = nE i1,...,i M ∼Unif w (i k ) û⋆ M (i) 2 , ∀k ∈ 1, M(65)
= n 1 M E i1,...,i M ∼Unif M k=1 w (i k ) û⋆ M (i) 2 (66) = n 1 M E i1,...,i M ∼Unif   M k=1 w (i k ) M j=1 w (ij ) u (ij ) M j=1 w (ij ) 2   (67) ≤ n 1 M E i1,...,i M ∼Unif M k=1 w (i k ) M j=1 w (ij ) (u (ij ) ) 2 M j=1 w (ij ) by convexity of x → x 2 (68) = n 1 M E i1,...,i M ∼Unif M k=1 w (i k ) M j=1 w (ij ) (u (ij ) ) 2 M j=1 w (ij ) (69) = n 1 M E i1,...,i M ∼Unif   M j=1 w (ij ) (u (ij ) ) 2   (70
) = E i1∼Unif w (i1) (u (i1) ) 2 (71) = E l∼Unif w (l) (u (l) ) 2 . (72
) Hence E x (1) ∼ pdata (•|xt,t) ; b (2) ,...,b (M ) ∼ pdata û⋆ M (x t , t) 2 -(û ⋆ ) 2 ≤ E l∼Unif w (l) (u (l) ) 2 -(û ⋆ ) 2 , (73
) which is exactly Var x (1) ∼ pdata (•|xt,t) ; b (2) ,...,b (M ) ∼ pdata [û ⋆ M (x t , t)] ≤ Var x (1) ∼ pdata (•|xt,t) u cond (x t , x (1) , t) .(74)
this section cite: []

Section: C Additional experiments
We present below the results for the MNIST dataset. The conclusions atre the same as for the CIFAR-10 and CelebA 64 × 64: regressing against a more deterministic velocity field does not hurt generalization. On the contrary, generalization (i.e., lower test FID) appears earlier during training.
For this experiment, we used the Unet with attention and timestep embedding from torchcfm library, with the Adam optimizer and all the default parameters. We used a pretrained classifier with 99% accuracy on MNIST (90% on FMNIST) as a lower-dimensional embedding of size 128 to compute the FID between the test set and the generated set. Table 2: FID FMNIST. FID scores across training epochs for conditional flow matching and empirical flow matching for multiple values of the number of samples M used to estimate the closed-form û⋆ .
this section cite: []

Section: D Experiments details
For all the experiment we used all the same learning hyperparameters, the default ones form Tong et al. (2024). The hyperparameter values are summarized in Table 3. The details specific to each figure are described in Appendices D.2 to D.5 # Channels Batch Size Learning Rate EMA Decay Gradient Clipping 128 128 0.0002 0.9999 1 Table 3: Learning hyperparameters for all the CIFAR-10 and CelebA 64 experiments.
this section cite: ['b51']

Section: D.1 Compute time
Given that regressing against an estimate of the closed-form, EFM, seems to improve on CFM, one may wonder what is the additional cost induced by EFN. To alleviate the non-linearity of GPU computing (parallelism may cause some discontinuities in terms of costs), we ran an exhaustive set of timing experiments, varying the batch size and the EFM sample size. To summarize the measurements (numbers are given for an NVIDIA L4 GPU, on CIFAR-10), denoting b the batch size and e the EFM sample size, the cost follows b × (4.3ms + e × 0.9µs). It can be also be seen as adding ∼ 2% for every 100 EFM samples. Or, for instance with a batch size of 256, 1.1 second will be due to the 256-sample forward/backward, while the additional cost for EFM-1000 will be 230ms (around 17% of the cost) and for EFM-128 under 30ms (under 3%).
this section cite: []

Section: D.2 Figures 1a and 1c
For Figure 1a no deep learning is involved: the datasets 2-moons and CIFAR-10 are loaded. Then, 256 points from p 0 × pdata are drawn, and one computes the mean of the cosine similarities between û⋆ ((1t)x 0 + tx 1 , t) and u cond ((1t)x 0 + tx 1 , z = x 1 , t) = x 1x 0 , for each value of t ∈ {0, 1/100, 2/100, . . . , 99/100}.
No deep learning either is involved in Figure 1c: the Imagenette dataset is loaded and spatially subsampled to resolution dim = 8, dim = 16, . . . , dim = 256, i.e., with d = " • 8 2 , d = 3 • 16 2 , . . . , d = 3 • 256 2 . Then, as for Figure 1a, batches of 256 points from p 0 and p data are drawn, and one computes the percentage of cosine similarities between û⋆ ((1t)x 0 + tx 1 , t) and u cond ((1t)x 0 + tx 1 , z = x 1 , t) = x 1x 0 , that are larger than 0.9, for multiple time values t.
this section cite: []

Section: D.3 Figure 2
In Figure 2, networks are trained with a vanilla conditional flow matching, with the standard 34 million parameters U-Net for diffusion by Nichol and Dhariwal (2021), with default settings from the torchfm codebase 4 (Tong et al., 2024). Training uses the CFM loss. For this specific experiment, we removed the usual random flip transform, for û⋆ to be simpler and easier to estimate by u θ . For each "data" subsampling of the dataset, we trained the model for 5 • 10 4 iterations, with a batch size of 128, i.e., we trained the models for 128 epochs. D.4 Figure 3 In Figure 3, for each dataset (CIFAR-10 and CelebA 64 × 64), one network is trained using a vanilla conditional flow matching with the default parameters of Tong et al. (2024) (the most important ones are recalled in Table 3). Then images are generated first following the closed-form formula of the optimal velocity field û⋆ from 0 to τ . And then following the velocity field learned with a usual conditional flow matching u θ from τ to 1.
this section cite: ['b31', 'b51', 'b51']

Section: D.5 Figure 4
For experiments involving training on CIFAR-10 (Figures 2 and 3), we rely on the standard 34 million parameters U-Net for diffusion by Nichol and Dhariwal (2021), with default settings from the torchfm codebase (Tong et al., 2024). For each algorithm, the networks are trained for 500k iterations with batch size 128, i.e., 1280 epochs.
For CelebA 64 × 64 (Figure 3), we rely on the training script of pnpflow library 5 (Martin et al., 2025), which uses a U-Net from Huang et al. (2021); Ho et al. (2020).
this section cite: ['b31', 'b51', 'b30', 'b19', 'b17']

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b1 Title: Dynamical regimes of diffusion models Year: (2024)
Ref_id:b2 Title: Why diffusion models don't memorize: The role of implicit dynamical regularization in training Year: (2025)
Ref_id:b3 Title: Audiolm: a language modeling approach to audio generation Year: (2023)
Ref_id:b4 Title: Simplicity bias and optimization threshold in two-layer ReLu networks Year: (2025)
Ref_id:b5 Title: Video generation models as world simulators Year: (2024)
Ref_id:b6 Title: Br-snis: bias reduced self-normalized importance sampling Year: (2022)
Ref_id:b7 Title: Extracting training data from diffusion models Year: (2023)
Ref_id:b8 Title: Rao-blackwellisation of sampling schemes Year: (1996)
Ref_id:b9 Title: Investigating data memorization in 3d latent diffusion models for medical image synthesis Year: (2023)
Ref_id:b10 Title: Bigger isn't always memorizing: Early stopping overparameterized diffusion models Year: (2025)
Ref_id:b11 Title: A visual dive into conditional flow matching Year: (2025)
Ref_id:b12 Title: Diffusion meets flow matching: Two sides of the same coin Year: (2025)
Ref_id:b13 Title: How do flow matching models memorize and generalize in sample data subspaces Year: (2024)
Ref_id:b14 Title: Diffuseq: Sequence to sequence text generation with diffusion models Year: (2023)
Ref_id:b15 Title: On memorization in diffusion models Year: (2025)
Ref_id:b16 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b17 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b18 Title: Imagenette: A smaller subset of 10 easily classified classes from imagenet Year: (2019-03)
Ref_id:b19 Title: A variational perspective on diffusion-based generative models and score matching Year: (2021)
Ref_id:b20 Title: Feature likelihood divergence: evaluating the generalization of generative models using samples Year: (2023)
Ref_id:b21 Title: Generalization in diffusion models arises from geometry-adaptive harmonic representations Year: (2024)
Ref_id:b22 Title: An analytic theory of creativity in convolutional diffusion models Year: (2025)
Ref_id:b23 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b24 Title: A good score does not lead to a good generative model Year: (2024)
Ref_id:b25 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b26 Title: Flow matching guide and code Year: (2024)
Ref_id:b27 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2023)
Ref_id:b28 Title: Deep learning face attributes in the wild Year: (2015-12)
Ref_id:b29 Title: Locality in image diffusion models emerges from data statistics Year: (2025)
Ref_id:b30 Title: Pnp-flow: Plug-and-play image restoration with flow matching Year: (2025)
Ref_id:b31 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b32 Title: Towards a mechanistic explanation of diffusion model generalization Year: (2025)
Ref_id:b33 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b34 Title:  Year: ()
Ref_id:b35 Title: Monte Carlo theory, methods and examples Year: (2013)
Ref_id:b36 Title: On aliased resizing and surprising subtleties in gan evaluation Year: (2022)
Ref_id:b37 Title: Multisample flow matching: Straightening flows with minibatch couplings. ICML Year: (2023)
Ref_id:b38 Title: Sidus-the solution for extreme deduplication of an operating system Year: (2013)
Ref_id:b39 Title: Monte Carlo statistical methods Year: (1999)
Ref_id:b40 Title: A geometric framework for understanding memorization in generative models Year: (2025)
Ref_id:b41 Title: Explicit flow matching: On the theory of flow matching algorithms with applications Year: (2024)
Ref_id:b42 Title: Closed-form diffusion models Year: (2025)
Ref_id:b43 Title: A phase transition in diffusion models reveals the hierarchical nature of data Year: (2025)
Ref_id:b44 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b45 Title: Understanding and mitigating copying in diffusion models Year: (2023)
Ref_id:b46 Title: Diffusion art or digital forgery? investigating data replication in diffusion models Year: (2023)
Ref_id:b47 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b48 Title:  Year: (2023)
Ref_id:b49 Title: Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models Year: (2023)
Ref_id:b50 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b51 Title: Improving and generalizing flow-based generative models with minibatch optimal transport Year: (2024)
Ref_id:b52 Title: Generalization through variance: how noise shapes inductive biases in diffusion models Year: (2025)
Ref_id:b53 Title: Phenaki: Variable length video generation from open domain textual descriptions Year: (2022)
Ref_id:b54 Title: Energy-based diffusion language models for text generation Year: (2025)
Ref_id:b55 Title: Stable target field for reduced variance score estimation in diffusion models Year: (2023)
Ref_id:b56 Title: Diffusion probabilistic models generalize when they fail to memorize Year: (2023)
Ref_id:b57 Title: The emergence of reproducibility and consistency in diffusion models Year: (2024)
Ref_id:b58 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
