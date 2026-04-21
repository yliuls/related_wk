Title: Ambient Diffusion mni: Training Good Models with Bad Data
Abstract: We show how to use low-quality, synthetic, and out-of-distribution images to improve the quality of a diffusion model. Typically, diffusion models are trained on curated datasets that emerge from highly filtered data pools from the Web and other sources. We show that there is immense value in the lower-quality images that are often discarded. We present Ambient Diffusion Omni, a simple, principled framework to train diffusion models that can extract signal from all available images during training. Our framework exploits two properties of natural images -spectral power law decay and locality. We first validate our framework by successfully training diffusion models with images synthetically corrupted by Gaussian blur, JPEG compression, and motion blur. We then use our framework to achieve stateof-the-art ImageNet FID and we show significant improvements in both image quality and diversity for text-to-image generative modeling. The core insight is that noise dampens the initial skew between the desired high-quality distribution and the mixed distribution we actually observe. We provide rigorous theoretical justification for our approach by analyzing the trade-off between learning from biased data versus limited unbiased data across diffusion times.

Section: Introduction
Large-scale, high-quality training datasets have been a primary driver of recent progress in generative modeling. These datasets are typically assembled by filtering massive collections of images sourced from the web or proprietary databases [25,44,53,58,59]. The filtering process-which determines which data is retained-is crucial to the quality of the resulting models [13,25,27,33]. However, filtering strategies are often heuristic and inefficient, discarding large amounts of data [13,25,44,51]. We demonstrate that the data typically rejected as low-quality holds significant, underutilized value.
Extracting meaningful information from degraded data requires algorithms that explicitly model the degradation process. In generative modeling, there is growing interest in approaches that learn to generate directly from degraded inputs [1,2,5,7,11,14,15,17,18,31,40,46,47,48,52,55,64,71,72]. A key limitation of existing methods is their reliance on knowing the exact form of the degradation. In real-world scenarios, image degradations-such as motion blur, sensor artifacts, poor lighting, and low resolution-are often complex and lack a well-defined analytical description, making this assumption unrealistic. Even within the same dataset, from ImageNet to internet scale text-to-image datasets, there are samples of heterogeneous qualities [28], as shown in Figures 4, 25, 28, 26. Given access to this mixed-bag of datapoints, we would like to sample from a tilted continuous measure of high-quality images, without sacrificing the diversity present in the training points.
a) Text-to-image results b) ImageNet results
this section cite: ['b24', 'b43', 'b52', 'b57', 'b58', 'b12', 'b24', 'b26', 'b32', 'b12', 'b24', 'b43', 'b50', 'b0', 'b1', 'b4', 'b6', 'b10', 'b13', 'b14', 'b16', 'b17', 'b30', 'b39', 'b45', 'b46', 'b47', 'b51', 'b54', 'b63', 'b70', 'b71', 'b27']

Section: Baseline
Ambient-o Baseline Ambient-o
Figure 1: Effect of using Ambient-o for (a) training a text-to-image model (Micro-Diffusion [54]) and (b) a class-conditional model for ImageNet (EDM-2 [36]). All generations are initialized with the same noise. The baseline models are trained using all the data equally. Ambient-o changes the way the data is used during the diffusion process based on its quality. This leads to significant visual improvements without sacrificing diversity, as would happen with a filtering approach (see Fig. 6).
The training objective of diffusion models naturally decomposes sampling from a target distribution into a sequence of supervised learning tasks [9,10,16,19,30,61,62]. Due to the power-law structure of natural image spectra [65], high diffusion times focus on generating globally coherent, semantically meaningful content [22], while low diffusion times emphasize learning high-frequency details.
Our first key theoretical insight is that low-quality samples can still be valuable for training in the high-noise regime. As noise increases, the diffusion process contracts distributional differences (Theorem 4.2), reducing the mismatch between the high-quality target distribution and the available mixed-quality data. At the same time, incorporating low-quality data increases the sample size, reducing the variance of the learned estimator. Our analysis formalizes this bias-variance trade-off and motivates a principled algorithm for training denoisers at high diffusion times using noisy data.
For low diffusion times, our algorithm leverages a second key property of natural images: locality. We show a direct relationship between diffusion time and the optimal receptive field size for denoising. A consequence of this result is that high-frequency details can be learned from out-of-distribution or synthetic images, as long as the marginal distributions of the crops match those of the target data.
2 Background and Related Work Diffusion Modeling. Diffusion models transform the problem of sampling from p 0 into the problem of learning denoisers for smoothed versions of p 0 defined as p t = p 0 ⊛ N (0, σ 2 (t)I). We typically denote with X 0 ∼ p 0 the R.V. distributed according to the distribution of interest and X t = X 0 + σ(t)Z, the R.V. distributed according to p t . The target is to estimate the set of optimal l 2 denoisers, i.e., the set of the conditional expectations: {E[X 0 |X t = •]} T t=1 . Typically, this can be achieved through supervised learning by minimizing the following loss (or a re-parametrization of it):
J(θ) = E t∈U [0,T ] E x0,xt|t ||h θ (x t , t) -x 0 || 2 , (2.1)
that is optimized over a function family H = {h θ : θ ∈ Θ} parametrized by network parameters θ.
For sufficiently expressive families, the minimizer is indeed:
h θ * (x, t) = E[X 0 |X t = x].
Learning from noisy data. The diffusion modeling framework described above assumes access to samples from the distribution of interest p 0 . An interesting variation of this problem is to learn to sample from p 0 given access to samples from a tilted measure p0 and a known degradation model. In Ambient Diffusion [18], the goal is to sample from p 0 given pairs (Ax 0 , A) for a matrix A : R m×n , m < n, that is distributed according to a known density p(A). The techniques in this work were later generalized to accommodate additive Gaussian Noise [1,15,17] in the measurements.
More recently there have been efforts to further broaden the family of degradation models considered through Expectation-Maximization approaches that involve multiple training runs [5,52].
Recent work from [17] has shown that, at least for the Gaussian corruption model, leveraging the low-quality data can tremendously increase the performance of the trained generative models. In particular, the authors consider the setting where we have access to a few samples from p 0 , let's denote them D 0 {x
(i) 0 } N1
i=1 and many samples from p tn , let's denote them D tn {x
(i) tn } N2 i=1 , where p tn = p 0 ⊛ N (0, σ 2 (t n )I)
is a smoothed version of p 0 at a known noise level t n . The clean samples are used to learn denoisers for all noise levels t ∈ [0, T ] while the noisy samples are used to learn denoisers only for t ≥ t n , using the training objective:
J ambient (θ) = E t∈U (tn,T ] N2 i=1 E xt|x (i) tn α(t)h θ (x t , t) + (1 -α(t))x t -x (i) tn 2 , (2.2)
with α(t) = σ 2 (t)-σ 2 (tn)
σ 2 (t)
. Note that the objective of equation 2.2 only requires samples from p tn (instead of p 0 ) and can be used to train for all times t ≥ t n . This algorithm uses N 1 + N 2 datapoints to learn denoisers for t > t n and only N 1 datapoints to learn denoisers for t ≤ t n . The authors show that even for N 1 << N 2 , the model performs similarly to the setting of training with (N 1 + N 2 ) clean datapoints. The main limitation of this method and its related works is that the degradation process needs to be known. However, in many applications, we have data from heterogeneous sources and various qualities, but there is no analytic form or any prior on the corruption model.
this section cite: ['b53', 'b35', 'b8', 'b9', 'b15', 'b18', 'b29', 'b60', 'b61', 'b64', 'b21', 'b17', 'b0', 'b14', 'b16', 'b4', 'b51', 'b16']

Section: Data filtering.
One of the most crude, but widely used, approaches for dealing with heterogeneous data sources is to remove the low-quality data and train only the high-quality subset [23,25,44]. While this yields better results than naively training on the entire distribution, it leads to a decrease in diversity and relies on heuristics for optimizing the filtering. An alternative strategy is to train on the entire distribution and then fine-tune on high-quality data [13,54]. This approach better trades the quality-diversity trade-off but still incurs a loss of diversity and is hard to calibrate.
Training with synthetic data. Recent works have shown that synthetic data improve the generative capabilities of diffusion models when mixed properly with real data from the target distribution [3,4,24]. In this work, we show that it helps significantly to view synthetic data as corrupted versions of the samples from the real distribution and incorporate this perspective into the training objective.
this section cite: ['b22', 'b24', 'b43', 'b12', 'b53', 'b2', 'b3', 'b23']

Section: Method
We propose a new framework that extends beyond [17] to enable training generative models directly from arbitrarily corrupted and out-of-distribution data, without requiring prior knowledge of the degradation process. We begin by formalizing the setting of interest.
this section cite: ['b16']

Section: Clean image

this section cite: []

Section: Blurry image

this section cite: []

Section: Merging point σ min Blurry image unsafe to use Distributions merge and blurry image is safe to use
Classifier output Problem Setting. We are given a dataset D = {w (i) 0 } N i=1 consisting of N datapoints. Each point in D is drawn from a mixture distribution p0 , which mixes p 0 (target distribution) and an alternative distribution q 0 that may contain various forms of degradation or out-of-distribution content. We assume access to two labeled subsets, S G , S B , where points in S G are known to come from the clean distribution p 0 , and points in S B from the corrupted distribution q 0 . While this assumption simplifies the initial exposition, we relax it in Section G.1. We focus on the practically relevant regime where |S G |≪ |D|, i.e., access to high-quality data is severely limited. The objective is to (approximately) sample from the clean distribution p 0 , leveraging both clean and corrupted samples.
We now describe how degraded and out-of-distribution samples can be effectively leveraged during training in both the high-noise and low-noise regimes of the diffusion process.
this section cite: []

Section: Learning in the high-noise regime (leveraging low-quality data)
Addition of gaussian noise contracts distribution distances. The first key idea of our method is that, at high diffusion times t, the noised target distribution p t and the noised corrupted distribution pt become increasingly similar (Theorem 4.2), effectively attenuating the discrepancy introduced by corruption. This effect is illustrated in Figure 2 (top), where we compare a clean image and its degraded counterpart (in this case, corrupted by Gaussian blur). As the diffusion time t increases, the noised versions of both samples become visually indistinguishable. Consequently, samples from p0 can be leveraged to learn (the score of) p t , for t > t min n . We formalize this intuition in Section 4, and we also quantify that for large t there are statistical efficiency benefits for using a large sample from p0 versus a small sample from p 0 .
this section cite: []

Section: Heuristic selection of the noise level.
From the discussion so far, it follows that to use samples from p0 , we need to assign them to a noise level t min n . One can select this noise level empirically, i.e. we can ablate this parameter by training different models and selecting the one that maximizes the generative performance. However, this approach requires multiple trainings, which can be costly. Instead, we can find the desired noise level in a principled way as detailed below.
Training a classifier under additive Gaussian noise. To identify the appropriate noise level, we train a time-conditional classifier to distinguish between the noised distributions p t and q t across various diffusion times. We use a single neural network c noise θ (x t , t) that is conditioned on the diffusion time t, following the approach of time-aware classifiers used in classifier guidance [21]. The classifier is trained using labeled samples from S G (clean) and S B (corrupted) via the following objective:
J noise (θ) = x0∈S G E xt|x0 -log c noise θ (x t , t) + y0∈S B E yt|y0 -log(1 -c noise θ (y t , t)) (3.1) Annotation.
Once the classifier is trained, we use it to determine the minimal level of noise that must be added to the low-quality distribution q 0 so that it closely approximates a smoothed version of the high-quality distribution p 0 . Formally, we compute:
t min n = inf    t ∈ [0, T ] : 1 |S B | y0∈S B E yt|y0 c noise θ (y t , t) > τ    ,(3.2)
for τ = 0.5 -ϵ and for some ϵ > 0. Subsequently, we form the annotated dataset D annot = {(w
(i) 0 +σ t min n Z (i) , t min n )} N i=1 ∪{(x 0 , 0)|x 0 ∈ S G },
where the random variables Z (i) are i.i.d. standard normals. This procedure means that we use samples from D for diffusion times t ≥ t min n that is safe to use them, i.e. only when the distributions have approximately merged. In fact, the optimal classifier assigns time t n that corresponds to the first time for which d TV (p t , q t ) ≤ ϵ.
0.0 0 0.2 0.3 0.5 0.7 0.9 1.3 2.1 32 σ t HQ Gaussian Blur JPEG MotionBlur Masking HQ images used by all methods HQ data excluded by filtering but used by Ambient-LQ data included by training on everything
this section cite: ['b20']

Section: Sample dependent annotation.
One potential issue with the aforementioned annotation approach is that all the samples in D are treated equally. But, as we noted, the points in D could be drawn from a distribution p0 that mixes p 0 and q 0 . In this case, all the samples in D that came from the p 0 component, will still get a high annotation time, leading to information loss. Instead, we can opt-in for a sample-wise annotation scheme, where each sample w From arbitrary corruption to additive Gaussian noise. The afore-described approach reduces our problem of learning from data with arbitrary corruption to the setting of learning from data corrupted with additive Gaussian noise. The price we pay for this reduction is the information loss due to the extra noise we add to the samples during the annotation stage. We can now extend the objective 2.2 to train our diffusion model. Suppose our annotated dataset is comprised of samples {(x (i) t min i , t min i )}. Then our objective becomes:
J ambient-o (θ) = E t∈U [0,T ] i:t min i <t E xt|x (i) t min i α(t, t min i )h θ (x t , t) + (1 -α(t, t min i ))x t -x (i) t min i 2 ,
where
α(t, t min i ) = σ 2 (t)-σ 2 (t min i ) σ 2 (t) .
Learning something from nothing? The proposed framework comes with limitations worth considering. First, unless the diffusion noise level tends to infinity, the distributions p t and q t never fully converge-there is always a bias when treating samples from q t as if they were from p t .
Moreover, the method is particularly well-suited to certain types of corruptions but is less effective for others. Because the addition of Gaussian noise suppresses high-frequency components-due to the spectral power law of natural images-our approach is most effective for corruptions that primarily degrade high frequencies (e.g., blur). In contrast, degradations that affect low-frequency content-such as color shifts, contrast reduction, or fog-like occlusions-are more challenging. This limitation is illustrated in Figure 3: masked images, for example, require significantly more noise to become usable compared to high-frequency corruptions like blur. In the extreme, the method reduces to a filtering approach, as infinite noise nullifies all information in the corrupted samples.
this section cite: []

Section: Learning in the low-noise regime (synthetic and out-of-distribution data)
So far, our algorithm implicitly results in varying amounts of training data across diffusion noise levels. At high noise, the model can leverage abundant low-quality data, whereas at low noise levels, it must rely solely on the limited set of high-quality samples. We now extend the algorithm to enable the use of synthetic and out-of-distribution data for learning denoisers at low-noise diffusion times.
To achieve this, we leverage another fundamental property of natural images: locality. At low diffusion times, the denoising task can be solved using only a small local region of the image, without requiring full spatial context. We validate this hypothesis experimentally in the Experiments Section (Figures 15,16,17,18), where we show that there is a mapping between diffusion time t and the crop size needed to perform the denoising optimally at this diffusion time. Intuitively, the higher the noise, the more context is required to accurately reconstruct the image. Conversely, for lower noise, the local information within a small neighborhood suffices to achieve effective denoising. We use crop(t) to denote the minimal crop size needed to perform optimal denoising at time t. If there are two distributions p 0 and p0 that agree on their marginals (i.e. crops), they can be used interchangeably for low-diffusion times. Note that the distributions don't have to agree globally, they only have to agree on a local (patch) level. Formally, let A(t) be a random patch selector of size crop(t). Let also p 0 , p0 two distributions that satisfy: A(t)#p 0 = A(t)#p 0 , where A(t)#p 0 denotes the pushforward measurefoot_0 of p 0 under A(t). Then, the cropped portions of the tilted distributions provide equivalent information to the original crops for denoising.
Training a crops classifier. Note that the condition above can be trivially satisfied if A(t) masks all the pixels or even if A(t) just selects a single pixel. We are interested in finding what is the maximum crop size for which this condition is approximately true. Once again, we can use a classifier to solve this task. The input to the classifier, c crops θ , is a crop of an image that either arises from p 0 or p0 , and the classifier needs to classify between these two cases.
Annotation and training using the trained classifier. Once the classifier is trained, we are now interested in finding the biggest crop size for which the distributions p 0 , p0 cannot be confidently distinguished. Formally,
t max n = sup    t ∈ [0, T ] : 1 |S B | y0∈S B [c crops θ (A(t)(y t ))] > τ    ,(3.3)
for τ = 0.5 -ϵ and for some small ϵ > 0foot_1 . For times t ≤ t max n , the out-of-distribution images from p0 can be used with the regular diffusion objective as images from p 0 , as for these times the denoiser only looks at crops and at the crop level the distributions have converged.
The donut paradox. Each sample can be used for t ≥ t min i and for t ≤ t max i , but not for t ∈ (t max i , t min i ). We call this the donut paradox as there is a hole in the middle of the diffusion trajectory for which we have fewer available data. These times do not have enough noise for the distributions to merge globally, but also the required receptive field for denoising is big enough so that there are differences on a crop level. We show an example of this effect in Figure 14.
this section cite: ['b14', 'b15', 'b16', 'b17']

Section: Theory
We study the 1-d case, but all our claims easily extend to any dimension. We compare two algorithms: Algorithm 1. Algorithm 1 trains a diffusion model using access to n 1 samples from a target density p 0 , assumed to be supported in [0, 1] and be λ 1 -Lipschitz.
Algorithm 2. Algorithm 2 trains a diffusion model using access to n 1 + n 2 samples from a density p0 that is a mixture of the a target density p 0 and another density q 0 , assumed to be supported in [0, 1] and be λ 2 -Lipschitz: p0 = n1 n1+n2 p 0 + n2 n1+n2 q 0 .
We want to compare how well these algorithms estimate the distribution p t := p 0 ⊛ N (0, σ 2 t ). We use p(1) t , p(2) t to denote the estimates obtained for p t by Algorithms 1 and 2 respectively.
Diffusion modeling is Gaussian kernel density estimation. We start by making a connection between the optimal solution to the diffusion modeling objective and kernel density estimation. Given a finite dataset {W (i) } n i=1 , the optimal solution to the diffusion modeling objective should match the empirical density at time t, which is:
pt (x) = 1 nσ t n i=1 ϕ W (i) -x σ t ,(4.1)
where
ϕ(u) = 1 √ 2π e -u 2 /2
is the Gaussian kernel. We observe that equation 4.1 is identical to a Gaussian kernel density estimate, given samples {W (i) } n i=1 4 .
We establish the following result for Gaussian kernel density estimation.
Theorem 4.1 (Gaussian Kernel Density Estimation). Let {W (i) } n
i=1 be a set of n independent samples from a λ-Lipschitz density p. Let p be the empirical density, p σ := p ⊛ N (0, σ 2 ) and pσ = p ⊛ N (0, σ 2 ). Then, with probability at least 1 -δ with respect to the sample randomness,
d TV (p σ , pσ ) ≲ 1 n + 1 σ 2 n + log n + log(1 ∨ λ) + log 2/δ σ 2 n . (4.2)
The proof of this result is given in the Appendix.
Comparing the performance of Algorithms 1 and 2. Applying Theorem 4.1 directly to the p 0 density, we immediately get that the estimate p(1) t (x) obtained by Algorithm 1 satisfies:
d TV (p t , p(1) t ) ≲ 1 n 1 + 1 σ 2 t n 1 + log n 1 + log(1 ∨ λ 1 ) + log 2/δ σ 2 t n 1 . (4.3)
Let us now see what we get by applying Theorem 4.1 to Algorithm 2, which uses samples from the tilted distribution p0 . Since this distribution is n1 n1+n2 λ 1 + n2 n1+n2 λ 2 -Lipschitz, we get that:
d TV (p t , p(2) t ) ≲ 1 (n 1 + n 2 ) + 1 σ 2 t (n 1 + n 2 ) + log(n 1 + n 2 ) + log(1 ∨ n1 n1+n2 λ 1 + n2 n1+n2 λ 2 ) + log 2/δ σ 2 t (n 1 + n 2 )
, where pt := p0 ⊛ N (0, σ 2 t ).
Further, we have that: d TV (p t , p(2) t ) ≤ d TV (p t , p t ) + d TV (p t , p(2) t ). We already have a bound for the second term. To bound the first term, we prove the following theorem. Theorem 4.2 (Distance contraction under noise). Consider distributions P and Q supported on a subset of R d with diameter D. Then
d TV (P ⊛ N (0, σ 2 I), Q ⊛ N (0, σ 2 I)) ≤ d TV (P, Q) • D 2σ .
Applying this theorem we get that: d TV (p t , p t ) ≤ 1 2σt d TV (p 0 , p 0 ) ≤ 1 2σt • n2 n1+n2 d TV (p 0 , q 0 ), where for the second inequality we used that d TV (p 0 , p0 ) ≤ n2 n1+n2 d TV (p 0 , q 0 ). Putting everything together, Algorithm (2) achieves an estimation error:
d TV (p t , p(2) t ) ≲ 1 (n 1 + n 2 ) + 1 σ 2 t (n 1 + n 2 ) + log(n 1 + n 2 ) + log(1 ∨ n1 n1+n2 λ 1 + n2 n1+n2 λ 2 ) + log 2/δ σ 2 t (n 1 + n 2 ) + n 2 σ t (n 1 + n 2 ) d TV (p 0 , q 0 ).
Comparing this with the bound obtained in Equation 4.3, we see that if n 2 is sufficiently larger than n 1 or if λ 2 ≤ λ 1 , there is a t min n such that for any t ≥ t min n , the upper-bound obtained by Algorithm 2 is better than the upper-bound obtained by Algorithm 1. That implies that for high-diffusion times, using biased data might be helpful for learning, as the bias term (final term) decays with the amount of noise. Going back to equation 4, note that the switching point t ≥ t min n depends on the distance d TV (p t , p t ) that decays as shown in Theorem 4.2. Once this distance becomes small enough, our computations above suggest that we benefit from biased data. The classifier of Section 3.1, if optimal, exactly tracks the distance d TV (p t , p t ) and, as a result, tracks the switching point.
this section cite: []

Section: Experiments
Controlled experiments to show utility from low-quality data. To verify our method, we first do synthetic experiments on artificially corrupted data. We use EDM [35] as our baseline, and we train networks on CIFAR-10 and FFHQ. For the first experiments, we only use the high-noise part of our Ambient-o method (Section 3.1). We underline that for all of our experiments, we only change the way we use the data, and we keep all the optimization and network hyperparameters as is. We compare against using all the data as equal (despite the corruption) and the filtering strategy of only training on the clean samples. For evaluation, we measure FID [29] with respect to the full uncorrupted dataset (which is not available during training). For the blurring experiments, we use a Table 1: In a controlled experiment with restricted access only to 10% of the clean dataset, our method of Ambient-o uses corrupted and out-of-distribution data to improve performance. Method Parameters Values (σ B ) σmin tn FID Only Clean (10%) --8.79 All data 1.0 0 45.32 0.8 28.26 0.6 11.42 0.4 2.47 Ambient-o 1.0 2.84 6.16 0.8 1.93 6.00 0.6 1.38 5.34 0.4 0.22 2.44 (b) Additional out-of-distribution data. Source Data Additional Data Method σmax tn FID Dogs (10%) None --12.08 Cats Fixed σ 0.2 11.14 Cats Fixed σ 0.1 9.85 Cats Fixed σ 0.05 10.66 Cats Fixed σ 0.025 12.07 Cats Classifier 0.09 8.92 Procedural Classifier 0.042 10.98 Cats (10%) None --5.20 Dogs Classifier 0.13 5.11 Wildlife Classifier 0.08 4.89
Gaussian kernel with standard deviation σ B = 0.4, 0.6, 0.8, 1.0, and we corrupt 90% of the data. We show some corrupted images in Figure 9a. To perform the annotations for our method, we train a blurry image vs clean image classifier under noise, as explained in Section 3.1. For the experiments in the main paper, we use a balanced dataset for the training of the classifier. We ablate the effect of having fewer training samples in Appendix Section F, where we show that reducing the number of clean samples available for classifier training leads to a small drop in performance. Once equipped with the trained classifier, each sample is annotated on its own based on the amount of noise that is needed to confuse the classifier (sample-dependent annotation). We present results in Table 1a.
As shown, for all corruption strengths, Ambient Omni, significantly outperforms the two baseline methods. In the one to the last column of Table 1a, we further show the average annotation of the classifier. As expected, the average assigned noise level increases as the corruption intensifies.
Ablations. We ablate the choice of using fixed vs sample-adaptive annotations in Table 12. We find that the latter performs better, but both methods improve over the baselines. We present results with JPEG compression in Table 3, motion blur in Figure 10 and FFHQ in Table 4. We ablate the impact of the amount of training data and training iterations on the classifier in Section F.
Controlled experiments to show utility from out-of-distribution images. We now want to validate the method developed in Section 3.2 for leveraging crops from out-of-distribution data. To start with, we want to find the mapping between diffusion times and the size of the receptive field required for an optimal denoising prediction. To do so, we take a pre-trained denoising diffusion model and measure the denoising loss at a given location as we increase the size of the context. We provide the corresponding plot in the Supplemental Figures 17,15. The main finding is that while providing more context always leads to a decrease in the average loss, for sufficiently small noise levels, the loss nearly plateaus before the full image context is provided. That implies that the perfect denoiser for a given noise level only needs to look at a localized part of the image. Equipped with the mapping between diffusion times and crop sizes, we now proceed to a fun experiment. We show that it is possible to use images of cats to improve a generative model for dogs (!) and vice-versa. The cats here represent out-of-distribution data that can be used to improve the performance in the distribution of interest (in our toy example, dogs distribution). To perform this experiment, we train a classifier that discriminates between cats and dog images by looking at crops of various sizes (Section 3.2). Figure 5 shows the predictions of an 8 × 8 crops-classifier for an image of a cat, illustrating that there are a number of crops that are misclassified as crops from a dog image. We report results for this experiment in Table 1b and we observe improvements in FID arising from using out-of-distribution data. Beyond natural images, we show that it is even possible to use procedurally generated data from Shaders [6] to (slightly) improve the performance. Figure 21 shows an example of such an image and the corresponding predictions of a crops classifier. Table 1b contains more results and ablations between annotating all the out-ofdistribution at a single noise level vs. sample-dependent annotations.
Takeaway 1: It is possible to use low-quality in-distribution images and high-quality out-ofdistribution images to produce high-quality in-distribution images.
Corruptions of natural datasets -ImageNet results. Up to this point, our corrupted data has been artificially constructed to study our method in a controlled setting. However, it turns out that even in real datasets such as ImageNet, there are images with significant degradations such as heavy blur, low lighting, and low contrast, and also images with fantastic detail, clear lightning, and sharp contrast. Here, the high-quality and the low-quality sets are not given and hence we have to estimate them. We opt to use the CLIP-IQA quality metric [66] to separate ImageNet into high-quality (top 10% CLIP-IQA) and low-quality (bottom 90% CLIP-IQA) sets. Figure 4 shows some of the top and bottom quality images according to our metric. Given the high-quality and low-quality sets, we are now back to the previous setting where we can use the developed Ambient-o methodology. We underline that there is a rich literature regarding quality-assessment methods [49,67,68,69].
We use Ambient-o to refer to our method that uses low-quality data at high diffusion times (Section 5) and Ambient-o+crops to refer to the extended version of our method that uses crops from potentially low-quality images at low-diffusion times. Perhaps surprisingly, there are ImageNet images that have lower global quality but high-quality crops that we can use. We present results in Table 2, where we show the best FID [29] and FD DINOv2 obtained by different methods. We show the highest and lowest quality crops of ImageNet according to CLIP, alongside the full images, in Figure 11. As shown in the Table, our method leads to state-of-the-art FID scores, improving over the baseline EDM-2 [36] at both the low and high parameter count settings. The benefits are more pronounced when we measure test FID as our method memorizes significantly less due to the addition of noise during the annotation stage of our pipeline (Section 3.1). Beyond FID, we provide qualitative results in Figure 1 (bottom) and Appendix Figures 12, 13. We further show that the quality of the generated images measured by CLIP increased compared to the baseline in Appendix Table 5. The observed improvements are proof that the ability to learn from data with heterogeneous qualities can be truly impactful for realistic settings beyond synthetic corruptions typically studied in prior work.
Takeaway 2: Real datasets contain heterogeneous samples. Ambient-o explicitly accounts for quality variability during training, leading to improved generation quality.
Text-to-image results. For our final set of experiments, we show how Ambient-o can be used to improve the performance of text-to-image diffusion models. We use the code-base of MicroDiffusion [54], as it is open-data and trainable with modest compute (≈ 2 days on 8-H100 GPUs). Sehwag et al. [54] use four main datasets to train their model: Conceptual Captions (12M) [56], Segment Anything (11M) [42], JourneyDB (4.2M) [63], and DiffusionDB (10.7M) [70]. Of these four, DiffusionDB is of significantly lower quality than the others as it contains solely synthetic data from an outdated diffusion model. This presents an opportunity for the use of our method. Can we use this lower-quality data and improve the performance of the trained network? We set σ min = 2 for all samples from DiffusionDB and σ min = 0 for all other datasets and we train a diffusion model with Ambient-o. We note that we did not ablate this hyperparameter and it is quite likely that improved results would be obtained by tuning it or by training a high-quality vs lowquality data classifier for the annotation. Despite that, our trained model achieves a remarkable FID of 10.61 in COCO, significantly improving the baseline FID of 12.37 (Table 8). We present qualitative results in Figure 1 and GPT-4o evaluations on DrawBench and PartiPrompt in Figure 7. Ambient-o and baseline generations for different prompts can be found in Figure 1.
33 66 33 66 45 54 31 68 40 59 33 66 42 57 46 53 40 59 43 56 40 59 29 70
Baseline Ambient-o (b) Measuring performance on the GenEval benchmark.
Objects Method Overall Single Two Counting Colors Position Color attribution Baseline 0.44 0.97 0.33 0.35 0.82 0.06 0.14 Ambient-o 0.47 0.97 0.40 0.36 0.82 0.11 0.14
Figure 8: Quantitative benefits of Ambient-o on COCO [45] zero-shot generation and GenEval [26].
As an additional ablation, we compared our method with the recipe of doing a final fine-tuning on the highest-quality subset, as done in [13,54]. Compared to this baseline, our method obtained slightly worse COCO FID (10.61 vs 10.27) but obtained much greater diversity, as seen visually in Figure 6 and quantitatively through > 13% increases in DINO Vendi Diversity on prompts from DiffDB (3.22 vs 3.65.). This corroborates our intuition that data filtration leads to decreased diversity. Ambient-o uses all the data but can strike a fine balance between high-quality and diverse generation.
Takeaway 3: Ambient-o treats synthetic data as corrupted data. This leads to superior visual quality and increased diversity compared to only relying on real samples.
this section cite: ['b34', 'b28', 'b16', 'b14', 'b5', 'b65', 'b48', 'b66', 'b67', 'b68', 'b28', 'b35', 'b53', 'b53', 'b55', 'b41', 'b62', 'b69', 'b44', 'b25', 'b12', 'b53']

Section: Conclusion
Is it possible to get good generators from bad data? Our framework leverages low-quality, synthetic, and out-of-distribution samples. At a time when the ever-growing data demands of GenAI are at odds with the need for quality control, Ambient-o lights a path for both to be achieved simultaneously.
15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: No research with human subjects. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: No important, original, or non-standard usage of LLMs in the paper.
this section cite: []

Section: A Limitations and Future Work
Our work opens several avenues for improvement. On the theoretical side, we aim to establish matching lower bounds to demonstrate that learning from the mixture distribution becomes provably optimal beyond a certain noise threshold. Algorithmically, while our method performs well under high-frequency corruptions, it remains an open question whether more effective training strategies could be used for different types of corruptions (e.g., masking). Moreover, real-world datasets often exhibit patch-wise heterogeneity-for example, facial regions are frequently blurred for privacy, leading to uneven corruption across image crops. We plan to investigate patch-level noise annotations to better capture this structure in future work. Computationally, the full-version of our algorithm requires the training of classifiers for annotations that increases the runtime. This overhead can be avoided by using hand-picked annotation times based on quality proxies as done in our synthetic data experiment. Finally, we believe the true potential of Ambient-o lies in scientific applications, where data often arises from heterogeneous measurement processes.
this section cite: []

Section: B Societal Impact
Given that (1) all the datasets we used are in the public domain and (2) prior works have already made public models trained on this data, we do not believe our work introduces extra risks that do not already exist.
this section cite: []

Section: C Theoretical Results

this section cite: []

Section: C.1 Kernel Estimation
Assumption C.1. The density p is λ lipschitz.
Let {X (i) } n i=1 a set of n independent samples from a density p that satisfies Assumption C.1. Let p be the empirical density on those samples.
We are interested in bounding the total variation distance between p σ := p ⊛ N (0, σ 2 ) and pσ = p ⊛ N (0, σ 2 ). In particular,
pσ (x) = 1 nσ n i=1 ϕ X (i) -x σ , (C.1)
where ϕ(u) = 1 √ 2π e -u 2 /2 is the Gaussian kernel. We want to argue that the TV distance between p σ and pσ is small given sufficiently many samples n. For simplicity, let's fix the support of p to be [0, 1]. We have:
d TV (p σ , pσ ) = 1 2 1 0 |p σ (x) -pσ (x)|dx = L-1 l=0 (l+1)/L l/L |p σ (x) -pσ (x)|dx (C.2)
Now let us look at one of the terms of the summation.
(l+1)/L l/L |p σ (x) -pσ (x)|dx = (l+1)/L l/L |p σ (x) -p σ (l/L) + p σ (l/L) -pσ (x)|dx (C.3) ≤ (l+1)/L l/L |p σ (x) -p σ (l/L)|dx + (l+1)/L l/L |p σ (l/L) -pσ (x)|dx. (C.4)
We first work on the first term. Using Lemma C.6:
(l+1)/L l/L |p σ (x) -p σ (l/L)|dx ≤ λ (l+1)/L l/L |x -l/L|dx (C.5) = λ 2L 2 . (C.6)
Next, we work on the second term. According to Lemma C.5, we have that pσ is λ = 1 σ 2 √ 2πe Lipschitz. Then, the second term becomes:
(l+1)/L l/L |p σ (l/L) -pσ (x)|dx ≤ λ (l+1)/L l/L |l/L -x|dx = λ 2L 2 . (C.9)
It remains to bound the following term
(l+1)/L l/L |p σ (l/L) -pσ (l/L)|dx = |p σ (l/L) -pσ (l/L)| L (C.10)
We will be applying Hoeffding's Inequality, stated below: Theorem C.2 (Hoeffding's Inequality). Let Y 1 , ..., Y n be independent random variables in [a, b] with mean µ. Then,
Pr 1 n n i=1 Y i -µ ≥ t ≤ 2 exp -2nt 2 /(b -a) 2 . (C.11)
Recall that pσ can be written as
pσ (x) = 1 n n i=1 ϕ((X (i) -x)/σ) σ = 1 n n i=1 Y i , (C.12)
in terms of the random variables Y i := ϕ((X (i) -x)/σ) σ . These random variables are supported in 0, 1 √ 2πσ 2 . So, for any x, we have that:
Pr (|p σ (x) -E[p σ (x)]| ≥ t) ≤ 2 exp -4πσ 2 nt 2 . (C.13
)
Taking t = log(2L/δ) 4πσ 2 n and using the above inequality and the union bound, we have that, with probability at least 1 -δ, for all l ∈ {0, 1, . . . , L -1}:
|p σ (l/L) -E[p σ (l/L)]| ≤ log(2L/δ) 4πσ 2 n . (C.14)
Let us now compute the expected value of pσ (x).
E[p σ (x)] = E 1 nσ n i=1 ϕ X (i) -x σ (C.15) = 1 nσ n i=1 E ϕ X (i) -x σ (C.16) = 1 σ p(u)ϕ x -u σ du ≡ (p ⊛ N (0, σ 2 ))(x) = p σ (x).
(C.17)
Combining equation C.14 and equation C.17, we get:
|p σ (l/L) -p σ (x)| ≤ log(2L/δ) 4πσ 2 n . (C.18)
Putting everything together we have:
d TV (p σ , pσ ) ≤ λ 2L + 1 2Lσ 2 √ 2πe + log(2L/δ) 4πσ 2 n .
Choosing L = n • max{λ, 1} we get that:
d TV (p σ , pσ ) ≲ 1 n + 1 σ 2 n + log n + log(1 ∨ λ) + log 2/δ σ 2 n .
this section cite: []

Section: C.2 Evolution of parameters under noise
Proof of theorem 4.2: We will use the following facts: Fact 1 (Direct corollary of the optimal coupling theorem). There exists a coupling γ of P and Q, which samples a pair of random variables (
X, Y ) ∼ γ such that Pr γ [X ̸ = Y ] = d TV (P, Q). Fact 2. For any x, y ∈ R d : d TV (N (x, σ 2 I), N (y, σ 2 I)) ≤ ∥x -y∥/2σ
Proof. The KL divergence between N (µ 1 , Σ 1 ) and
N (µ 2 , Σ 2 ) is KL(N (µ 1 , Σ 1 ), N (µ 2 , Σ 2 )) = 1 2 tr(Σ -1 2 Σ 1 ) + (µ 2 -µ 1 )Σ -1 2 (µ 2 -µ 1 ) -d + log |Σ 2 | |Σ 1 | .
Applying this general result to our case:
KL(N (x, σ 2 I), N (y, σ 2 I)) = 1 2 ∥x -y∥ 2 σ 2 .
this section cite: []

Section: We conclude by applying Pinsker's inequality.
A corollary of Fact 2 and the optimal coupling theorem is the following: Fact 3. Fix arbitrary x, y ∈ R d . There exists a coupling γ x,y of N (0, σ 2 I) and N (0, σ 2 I), which samples a pair of random variables (Z, Z ′ ) ∼ γ x,y such that Pr γx,y [x + Z ̸ = y + Z ′ ] = ∥x -y∥/2σ. Now let us denote by P = P ⊛ N (0, σ 2 I) and Q = Q ⊛ N (0, σ 2 I). To establish our claim in the theorem statement, it suffices to exhibit a coupling γ of P and Q which samples a pair of random variables ( X, Ỹ ) ∼ γ such that:
Pr γ [ X ̸ = Ỹ ] ≤ d TV (P, Q) • D 2σ .
We define coupling γ as follows:
1. Sample (X, Y ) ∼ γ (as specified in Fact 1); then
2. sample (Z, Z ′ ) ∼ γ X,Y (as specified in Fact 3); then 3. output ( X, Ỹ ) := (X + Z, Y + Z ′ ).
Let us argue the following:
Lemma C.3. The afore-described sampling procedure γ is a valid coupling of P and Q.
Proof. We need to establish that the marginals of γ are P and Q. We will only show that for ( X, Ỹ ) ∼ γ according to the afore-described sampling procedure, the marginal distribution of X is P , as the proof for Ỹ is identical. Since γ is a coupling of P and Q, for (X, Y ) ∼ γ, the marginal distribution of X is P . By Fact 3, conditioning on any value of X and Y , the marginal distribution of Z is N (0, σ 2 I). Thus, X = X + Z, where X ∼ P and independently Z ∼ N (0, σ 2 I), and thus the distribution of X is P .
Lemma C.4. Under the afore-described coupling γ:
Pr γ [ X ̸ = Ỹ ] ≤ d TV (P, Q) • D 2σ .
Proof. Notice that, when X = Y , by Fact 3, Z = Z ′ with probability 1, and therefore X = Ỹ . So for event X ̸ = Ỹ to happen, it must be that X ̸ = Y happens and, conditioning on this event, that
X + Z ̸ = Y + Z ′ happens. By Fact 1, Pr γ [X ̸ = Y ] = d TV (P, Q). By Fact 3, for any realization of (X, Y ), Pr γ X,Y [X + Z ̸ = Y + Z ′ ] = ∥X-Y ∥ 2σ ≤ D
2σ , where we used that P and Q are supported on a set with diameter D. Putting the above together, the claim follows.
this section cite: []

Section: C.3 Auxiliary Lemmas
Lemma C.5 (Lipschitzness of the empirical density). For a collection of points X (1)
, . . . , X (n) consider the function pσ (x) = 1 nσ n i=1 ϕ X (i) -x σ , where ϕ(u) = 1 √ 2π e -u 2 /2 is the Gaussian kernel. Then p σ is 1 σ 2 √ 2πe -Lipschitz.
Proof. Let us compute the derivative of pσ :
p′ σ (x) = 1 nσ n i=1 d dx ϕ x -X (i) σ (C.19) = 1 √ 2πnσ n i=1 exp -(X (i) -x) 2 /(2σ 2 ) X (i) -x σ 2 (C.20) ≤ 1 √ 2πσ 2 max u exp(-u 2 /2)u (C.21) ≤ 1 σ 2 √ 2πe . (C.22)
Lemma C.6 (Lipschitzness of a density convolved with a Gaussian). Let p be a density that is λ-Lipschitz. Let p σ = p ⊛ N (0, σ 2 I). Then, p σ is also λ-Lipschitz.
Proof. Let us denote with ϕ σ (•) the Gaussian density with variance σ 2 . We have that:
p σ (x) -p σ (y) = (p(x -τ ) -p(y -τ ))ϕ σ (τ )dτ ⇒ (C.23) |p σ (x) -p σ (y)|≤ |p(x -τ ) -p(y -τ )|ϕ σ (τ )dτ (C.24) ≤ λ|x -y|• ϕ σ (τ )dτ (C.25) = λ|x -y|. (C.26)
this section cite: []

Section: D Additional Results

this section cite: []

Section: D.1 CIFAR-10 controlled corruptions
Figures 9a, 10, 9b show gaussian blur, motion blur, and JPEG corrupted CIFAR-10 images respectively at different levels of severity. Appendix Table 3 shows results for JPEG compressed data at different levels of compression. We also tested our method for motion blurred data with high severity, visualized in the last row of Appendix Figure 10), obtaining a best FID of 5.85 (compared to 8.79 of training on only the clean data).
this section cite: []

Section: D.2 FFHQ-64x64 controlled corruptions
In Appendix 4 we show additional results for learning from blurred data on the FFHQ dataset. Similarly to the main paper, we observe that our Ambient-o algorithm leads to improvements over just using the high-quality data that are inversely proportional to the corruption level.
this section cite: []

Section: D.3 ImageNet results
In the main paper, we used FID as a way to measure the quality of generated images. However, FID is computed with respect to the test dataset that might also have samples of poor quality. Further, during FID computation, quality and diversity are entangled. To disentangle the two, we generate images using the EDM-2 baseline and our Ambient-o model and we use CLIP to evaluate the quality of the generated image (through the CLIP-IQA metric implemented in the PIQ package [38,39]). We present results and win-rates in Table 5. As shown, Ambient-o achieves a better per-image quality compared to the baseline despite using exactly the same model, hyperparameters, and optimization algorithm. The difference comes solely from better use of the available data.  Beyond the scores provided in the main paper, we report additional metrics for our Ambient-o XXL+crops model trained on ImageNet in Table 6. These metrics show more evidence for better distribution learning and lower memorization.
this section cite: ['b37', 'b38']

Section: D.4 Sanity checks on additive Gaussian noise corruption
As a sanity check for the performance of the classifier and the method, we train a classifier on data that has actually been corrupted with additive Gaussian noise and we report the average predicted noise level in Table 7. As shown, the model has small errors but is roughly capable of predicting the noise level used. We trained models using the estimated noise levels and we reproduced the ≈ 2.05 FID result reported in the paper "How much is a noisy image worth" [14]. Our approach generalizes this idea to arbitrary corruptions, without needing to know the corruption type.
this section cite: ['b13']

Section: D.5 Effect of number of clean datapoints
For all our synthetic corruptions in the paper we used 10% clean data. We ablate the effect of this in Table 8. In particular, we provide FID results for training with x=1%, 5%, 10%, 30%, and 50% clean data and (100 -x)% blurry data at blur level σ B = 0.8 on CIFAR-10. As expected, increasing the amount of clean data has a very significant impact on the FID. The interesting challenge is how much can we improve the results were the amount of clean datapoints is small.
this section cite: []

Section: D.6 Patch level FIDs
The usage of out-of-distribution images for small diffusion times has the risk of introducing small artifacts. We do our best to understand if that's the case for the example of using cats images to train a generative model for dogs, as in the paper. To check for artifacts, we report FIDs on the distribution of patches of various sizes for the model trained only on dogs and the model trained on dogs and cats. Results are reported in Table 9. As shown, our method also achieves better FID when looking at patches. This is evidence that we have better fine-grained details compared to the model trained only on clean data.
this section cite: []

Section: E Ambient diffusion implementation details and loss ablations
Similar to the EDM-2 [36] paper, we use a pre-condition weight to balance the importance of different diffusion times. Specifically, we modulate the EDM2 weight λ(σ) by a factor:
λ amb (σ, σ min ) = σ 4 /(σ 2 -σ 2 min ) 2 (E.1)
Table 5: Additional comparison between EDM-2 XXL and our Ambient-o model using the CLIP IQA metric for image quality assesment. Ambient-o leads to improved scores despite using the exact same architecture, data and hyperparameters. For this experiment, we use the models with guidance optimized for DINO FD since they are the ones producing the higher quality images.
this section cite: ['b35']

Section: Metric EDM-2 [36] XXL Ambient-o XXL crops
Average CLIP IQA score 0.69 0.71 Median CLIP IQA score 0.79 0.80
Win-rate 47.98% 52.02%  Table 9: FID across patch sizes.
for our ambient loss based on a similar analysis to [36]. We further use a buffer zone around the annotation time of each sample to ensure that the loss doesn't have singularities due to divisions by 0. We ablate the precondition term and the buffer size in Appendix Table 10.
For our ablations, we focus on the setting of training with 10% clean data and 90% corrupted data with Gaussian blur of σ B = 0.6. Using no ambient pre-conditioning and no buffer, we obtain an FID of 5.56. In the same setting, adding the ambient pre-conditioning weight λ amb (σ, σ min ) improves FID by 0.13 points. Next, we ablate two strategies to mitigate the impact of the singularity of λ amb (σ, σ min ) at σ = σ min . The first strategy clips the ambient pre-conditioning weight at a specified maximum value λ MAX amb , but still trains for σ arbitrarily close to σ min . The second strategy also specifies a maximum value, but imposes a buffer
σ > 1 + 1 λ MAX amb -1 σ min (E.2)
that restricts training to noise levels σ such that λ amb (σ, σ min ) ≤ λ MAX amb . Clipping the ambient weight to λ MAX amb = 2.0 minimally improves FID to 5.35, but clipping to 4.0 significantly worsens it to 5.69. Adding a buffer at λ MAX amb = 2.0 slightly worsens FID to 5.40, but slackening the buffer to 4.0 minimally improves FID to 5.34. We opt for the buffering strategy in favor of the clipping strategy since performance appears convex in the buffer parameter, and because it obtains the best FID.
this section cite: ['b35']

Section: G.2 Details about classifier training details and pitfalls
The sample-dependent annotations lead to stronger experimental performance compared to using a single noise level for all the corrupted samples. However, the developed theory for distribution mixing does not apply here and the use of per sample annotations can lead to an introduction of biases in the distribution. For example, if the target distribution p 0 has dogs and cats and the "corrupted" distribution q 0 has only dogs, using per-sample annotations will increase the probability of generating from the dogs class. Hence, we recommend caution when using sample dependent annotations and we leave it for future work how to account for this problem. In what follows, we provide some details about classifier training.
Crops classifier training details. During training, the classifier takes as input a crop (that can be any size) and tries to detect if the crop came from the high-quality distribution or the low-quality distribution.
At inference time, we split an image into crops of a specific size, let's call it C, and then we see if, on average, they confuse the classifier. The bigger the C, the harder it is to confuse the classifier. If there is no confusion for the initial C, we decrease it and we try again until we find a crop size for which the classifier is (on average) confused. We underline that, in principle, we could do the annotation separately for each crop of the same image, effectively leading to different diffusion times per crop. That said, we used an implementation that averages across crops for simplicity. We finally map from the crop size that confused the classifier to a diffusion time, using Figures 15/17. Higher noise levels require bigger receptive fields (bigger crops) for optimal denoising. So if an image only manages to confuse the classifier at a small crop, then it is only used for a small set of diffusion times.
We also clarify that the diffusion model is never trained on crops of images. We always train using the entire image, even the out-of-distribution images, but for diffusion times small enough (less than t_max) such that for the required receptive field size at that noise level, the target and out-of-distribution data is indistinguishable.
G.3 Datasets CIFAR-10. CIFAR-10 [43] consists of 60,000 32x32 images of ten classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck).
FFHQ. FFHQ [37] consists of 70,000 512x512 images of faces from Flickr. We used the dataset at 64x64 resolution for our experiments.
AFHQ. AFHQ [12] consists of 5,653 images of cats, 5,239 images of dogs and 5,000 images of wildlife, for a total of 15,892 images.
ImageNet. ImageNet [20] consists of 1,281,167 images of variable resolution from 1000 classes. XXL model trained on ImageNet.
this section cite: ['b36', 'b11', 'b19']

Section: Data Availability vs. Noise Level
Figure 14: Amount of samples available at each noise level when training a generative model for dogs in the following setting: (1) we have 10% of the dogs dataset uncorrupted, (2) we have the other 90% of the dogs dataset corrupted with gaussian blur with σ B = 0.6, and (3) we have 100% of the clean dataset of cats. At low noise levels, we can train on both the high quality dogs and a lot of the cats, resulting in > 100% of samples available relative to the original dogs dataset size. As the noise level starts to increase, we stop being able to use to the out-of-distribution cat samples, but start gaining some blurry dog samples. As the noise level approaches the maximum all the blurry dogs become available for training, such that the amount of data available approaches 100%.
10 15 20 25 30 Context Size 0.00 0.01 0.02 0.03 0.04 0.05 MSE Loss on 2x2 Center Patch Loss vs Context Size =0.05 =0.15 =0.25 =0.35 =0.45 =0.55 =0.65 =0.75 =0.85 =0.95
Figure 15: ImageNet-512x512: denoising loss of an optimally trained model, measured at 2 × 2 center patch, as we increase the context size given to the model (horizontal axis) and the noise level (different curves). As expected, for higher noise, more context is needed for optimal denoising. The large dot on each curve marks the point where the loss nearly plateaus. 10 20 30 40 50 60 Context Size 0.00 0.01 0.02 0.03 0.04 0.05 MSE Loss on a 2x2 patch Loss vs Context Size =0.05 =0.15 =0.25 =0.35 =0.45 =0.55 =0.65 =0.75 =0.85 =0.95 As expected, for higher noise, more context is needed for optimal denoising. The large dot on each curve marks the point where the loss nearly plateaus.
0.2 0.4 0.6 0.8 1.0 Noise Level ( ) 10 20 30 40 50 60 Effective Receptive Field Size Receptive Field Size vs. Noise Level (a) Cat image and classification probabilities over patches. (b) Cat image and classification probabilities over patches.
Figure 19: Two examples of cats from the AFHQ dataset. We partition each cat into non overlapping patches and we compute the probabilities of the patch belonging to an image of a dog using a cats vs dogs classifier trained on patches. The cat on the right has a lot more patches that could belong to a dog image according to the classifier, possibly due to the color or the texture of the fur.    (a) Synthetic image and classification probabilities over patches. (b) Synthetic image and classification probabilities over patches.
Figure 22: Two examples of procedurally generated images. We partition each image into non overlapping patches and we compute the probabilities of the patch belonging to an image of a dog using a synthetic image vs dogs classifier trained on patches. The image on the right has a lot more patches that could belong to a dog image according to the classifier, possibly due to the color or the texture.
(a) Cat image and classification probabilities over patches.
(b) Cat image and classification probabilities over patches.
Figure 23: Two examples of cat images. We partition each image into nonoverlapping patches and we compute the probabilities of the patch belonging to an image of wildlife using a cats vs wildlife classifier trained on patches. The image on the right has a lot more patches that could belong to a wildlife image according to the classifier, possibly due to the color or the texture.
(a) Example batch. (b) Noisy batch. (a) Highest quality images from CC12M according to CLIP. (b) Lowest quality images from CC12M according to CLIP. (a) Highest quality images from SA1B according to CLIP. (b) Lowest quality images from SA1B according to CLIP.
this section cite: []

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: Our method does not use any information about the type of corruption, and our experiments show it generalizes to low quality data found in the wild, not just a few artifically controlled corruptions.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We openly discuss the limitations of our approach, such as: (a) The high and low quality distributions never perfectly merge, so our method always introduces a (small) distribution error compared to filtering. (b) Our method does not work well with certain corruption types, such as masking. These "ill-suited" corruptions require a very large amount of noise to merge, such that they are effectively never used during training and our method reduces to filtering in these cases.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [Yes] Justification: Our theorems include all premises and assumptions used to prove the result. Informal proofs are found in the main text, referencing formal proofs in the appendix.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: All the information on the algorithm and the training recipe needed to reproduce our experiments is included in the paper (either in the main text or the appendix). Additionally, we make the training and evaluation code public. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All data used is publicly accessible. We release the full training and evaluation code. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.
) necessary to understand the results? Answer: [Yes] Justification: We provide the core elements in the main text and the full details in the appendix. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: Obtaining error bars would require extremely computationally expensive retraining of diffusion models. 8. Experiments compute resources Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Computational requirements are provided in the Appendix. 9. Code of ethics Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines? Answer: [Yes] Justification: The work does not use human trials, and all data used is publically available. We analyse the potential negative impacts of improving generative model abilities in section B. 10. Broader impacts Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: See section B. 11. Safeguards Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: We are not releasing any datasets. We are releasing the models. That said, there has already been a model trained and open-sourced from the same dataset. Moreover, our work is not close to state-of-the-art text-to-image generation, and thus does not introduce extra risks that do not already exist. 12. Licenses for existing assets Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected? Answer: [NA] Justification: Prior work has already trained and made public models trained on the same data we use to train. Moreover, all datasets are publically available and were introduced by prior research work, which we explicitly state and cite. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: We do not release any new datasets. 14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: No research with human subjects.
this section cite: []

Section: F Classifier annotation ablations
Balanced vs unbalanced data: We ablate the impact of classifier training data on the setting of CIFAR-10 with 10% clean data and 90% corrupted data with gaussian blur with σ B = 0.6. When annotating with a classifier trained on the same unbalanced dataset we train the diffusion model on we obtained a best FID of 6.04, compared to the 5.34 obtained if we train on a balanced dataset.
this section cite: []

Section: Training iterations:
We ablate the impact of classifier training iterations on the setting of CIFAR-10 with 10% clean data and 90% corrupted data with JPEG compression at compression rate of 18%, training the classifier with a balanced dataset. We report minute variations in the best FID, obtaining 6.50, 6.58, and 6.49 when training the classifier for 5e6, 10e6, and 15e6 images worth of training respectively.
Threshold ablations: An important hyperparameter for the classifier annotations is the threshold used for the approximate mixing of the two distributions. We ablate this at Table 11. As shown, the method is relatively robust to miscalibrations. That said, the performance can be improved if this threshold is separately tuned for each corruption type and strength.
Threshold FID ↓ Notes 0.40 6.67 0.43 6.73 0.45 6.43 Baseline (in paper) 0.47 6.21 Best FID 0.48 6.28 Table 11: FID as a function of the classifier threshold. The results obtained here are for CIFAR-10 JPEG corruption at rate 18%. Lower FID values indicate better performance, with the optimal value achieved at threshold 0.47. As shown, the method is relatively robust to miscalibrations.
this section cite: []

Section: G Training Details
G.1 Formation of the high-quality and low-quality sets.
In the theoretical problem setting we assumed the existence of a good set S G from the clean distribution and a bad set S B from the corrupted distribution. In practice, we do not actually possess these sets initially, but we can construct them so long as we have access to a measure of "quality". Given a function on images which tells us wether its good enough to generate or not e.g. CLIP-IQA quality [66] greater than some threshold, we can define our good set S G as the good enough images and S B as the complement. From this point on we can apply the methodology of ambient-o as developed, either employing classifier annotations as in our pixel diffusion experiments, or fixed annotations as in our large scale ImageNet and text-to-image experiments.
this section cite: ['b65']

Section: References
Ref_id:b0 Title: Solving Inverse Problems with Score-Based Generative Priors learned from Noisy Data Year: (2023)
Ref_id:b1 Title: Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models Trained on Corrupted Data Year: (2025)
Ref_id:b2 Title: Self-consuming generative models go mad Year: (2023)
Ref_id:b3 Title: Self-improving diffusion models with synthetic data Year: (2024)
Ref_id:b4 Title: An Expectation-Maximization Algorithm for Training Clean Diffusion Models from Corrupted Observations Year: (2024)
Ref_id:b5 Title: Procedural Image Programs for Representation Learning Year: (2022)
Ref_id:b6 Title: AmbientGAN: Generative models from lossy measurements Year: (2018)
Ref_id:b7 Title: Kernel density estimation via diffusion Year: (2010)
Ref_id:b8 Title: Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions Year: (2022)
Ref_id:b9 Title: Restoration-Degradation Beyond Linear Diffusions: A Non-Asymptotic Analysis For DDIM-type Samplers Year: (2023-07-29)
Ref_id:b10 Title: Denoising Score Distillation: From Noisy Diffusion Pretraining to One-Step High-Quality Generation Year: (2025)
Ref_id:b11 Title: StarGAN v2: Diverse image synthesis for multiple domains Year: (2020)
Ref_id:b12 Title: Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack Year: (2023)
Ref_id:b13 Title: How Much is a Noisy Image Worth? Data Scaling Laws for Ambient Diffusion Year: ()
Ref_id:b14 Title: Consistent diffusion models: Mitigating sampling drift by learning to be consistent Year: (2023)
Ref_id:b15 Title: Soft Diffusion: Score Matching with General Corruptions Year: (2023)
Ref_id:b16 Title: Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data Year: (2024)
Ref_id:b17 Title: Ambient Diffusion: Learning Clean Distributions from Corrupted Data Year: (2023)
Ref_id:b18 Title: Inversion by direct iteration: An alternative to denoising diffusion for image restoration Year: (2023)
Ref_id:b19 Title: ImageNet: A largescale hierarchical image database Year: (2009)
Ref_id:b20 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b21 Title: Diffusion is spectral autoregression Year: (2024)
Ref_id:b22 Title: Optimizing ml training with metagradient descent Year: (2025)
Ref_id:b23 Title: Self-consuming generative models with curated data provably optimize human preferences Year: (2024)
Ref_id:b24 Title: DataComp: In search of the next generation of multimodal datasets Year: (2023)
Ref_id:b25 Title: GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment Year: (2023)
Ref_id:b26 Title: Scaling Laws for Data Filtering-Data Curation cannot be Compute Agnostic Year: (2024)
Ref_id:b27 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019)
Ref_id:b28 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b29 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b30 Title: DiffEM: Learning from Corrupted Data with Diffusion Models via Expectation Maximization Year: (2025)
Ref_id:b31 Title: Adaptive Mixtures of Local Experts Year: (1991-03)
Ref_id:b32 Title: Adaptive data optimization: Dynamic sample selection with scaling laws Year: (2024)
Ref_id:b33 Title: An analytic theory of creativity in convolutional diffusion models Year: (2024)
Ref_id:b34 Title: Elucidating the design space of diffusion-based generative models Year: (2022)
Ref_id:b35 Title: Analyzing and Improving the Training Dynamics of Diffusion Models Year: (2024)
Ref_id:b36 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b37 Title: PyTorch Image Quality: Metrics and Measure for Image Quality Assessment Year: (2019)
Ref_id:b38 Title: PyTorch Image Quality: Metrics for Image Quality Assessment Year: (2022)
Ref_id:b39 Title: Ambient-Flow: Invertible generative models from incomplete, noisy measurements Year: (2023)
Ref_id:b40 Title: Adam: A Method for Stochastic Optimization Year: (2014)
Ref_id:b41 Title: Segment Anything. 2023 Year: ()
Ref_id:b42 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b43 Title: DataComp-LM: In search of the next generation of training sets for language models Year: (2024)
Ref_id:b44 Title: Common Objects in Context. 2015 Year: ()
Ref_id:b45 Title: IMAGING AN EVOLVING BLACK HOLE BY LEVERAGING SHARED STRUCTURE Year: (2024)
Ref_id:b46 Title: ADG: Ambient Diffusion-Guided Dataset Recovery for Corruption-Robust Offline Reinforcement Learning Year: (2025)
Ref_id:b47 Title: SFBD: A Method for Training Diffusion Models with Noisy Data Year: (2025)
Ref_id:b48 Title: Making a "completely blind" image quality analyzer Year: (2012)
Ref_id:b49 Title: Scalable Diffusion Models with Transformers Year: (2023)
Ref_id:b50 Title: The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale Year: (2024)
Ref_id:b51 Title: Learning Diffusion Priors from Observations by Expectation Maximization Year: (2024)
Ref_id:b52 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b53 Title: Stretching Each Dollar: Diffusion Training from Scratch on a Micro-Budget Year: (2024)
Ref_id:b54 Title: Does Generation Require Memorization? Creative Diffusion Models using Ambient Diffusion Year: (2025)
Ref_id:b55 Title: Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning Year: (2018-07)
Ref_id:b56 Title: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer Year: (2017)
Ref_id:b57 Title: Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models Year: (2022)
Ref_id:b58 Title: Understanding and Mitigating Copying in Diffusion Models Year: (2023)
Ref_id:b59 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b60 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b61 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b62 Title: JourneyDB: A Benchmark for Generative Image Understanding Year: (2023)
Ref_id:b63 Title: Diffusion with forward models: Solving stochastic inverse problems without direct supervision Year: (2023)
Ref_id:b64 Title: Foundations of computer vision Year: (2024)
Ref_id:b65 Title: Exploring CLIP for Assessing the Look and Feel of Images Year: (2023)
Ref_id:b66 Title: Exploring clip for assessing the look and feel of images Year: (2023-02)
Ref_id:b67 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b68 Title: Multiscale structural similarity for image quality assessment Year: (2003)
Ref_id:b69 Title: DiffusionDB: A Large-scale Prompt Gallery Dataset for Text-to-Image Generative Models Year: (2022)
Ref_id:b70 Title: Restoration Score Distillation: From Corrupted Diffusion Pretraining to One-Step High-Quality Generation Year: (2025)
Ref_id:b71 Title: Restoration Score Distillation: From Corrupted Diffusion Pretraining to One-Step High-Quality Generation Year: (2025)
Ref_id:b72 Title: Conceptual Captions [56] consists of 12M (image url, caption) pairs Year: ()
Ref_id:b73 Title: Segment Anything [42] consists of 11.1M high-resolution images annotated with segmentation masks. Since the original dataset did not have real captions Year: ()
Ref_id:b74 Title: DiffusionDB consists of 14M synthetic image-caption pairs, mostly generated from Stable Diffusion models [70]. We use the same 10.7M quality-filtered subset created by the MicroD Year: ()
Ref_id:b75 Title: ] codebase as a reference to train class-conditional diffusion models on CIFAR-10. The architecture is a Diffusion U-Net [60] with ~55M paramemeters. We use the Adam optimizer [41] with learning rate 0.001, batch size 512, and no weight decay. While the original EDM paper trained for 200 × 10 6 images worth of training, when training with corrupted data we saw best results around 20 × 10 6 images. On a single 8xV100 node we achieved a throughput of 0.8s per 1k images, for an average of 4.4h per training run. FFHQ. Same as for CIFAR-10, except learning was set to 2e -4, we trained for a maximum of 100 × 10 6 images worth of training Year: ()
Ref_id:b76 Title: We use the EDM2 [36] codebase as a reference to train class-conditional diffusion models on ImageNet. The architecture is a Diffusion U-Net [60] with ~125M paramemeters. We use the Adam optimizer [41] with reference learning rate 0.012, batch size 2048, and no weight decay. Same as the original codebase, we trained for ~2B worth of images Year: ()
Ref_id:b77 Title: We follow their recipe exactly, changing only the standard denoising diffusion loss to the ambient diffusion loss. The architecture is a Diffusion Transformer [50] utilizing Mixture-of-Experiments (MoE) feedforward layers [32, 57], with ~1.1B paramemeters. We use the AdamW optimizer Year: ()
