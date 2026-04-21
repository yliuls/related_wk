Title: Improving the Scaling Laws of Synthetic Data with Deliberate Practice
Abstract: Inspired by the principle of deliberate practice in human learning, we propose Deliberate Practice for Synthetic Data Generation (DP), a novel framework that improves sample efficiency through dynamic synthetic data generation. Prior work has shown that scaling synthetic data is inherently challenging, as naively adding new data leads to diminishing returns. To address this, pruning has been identified as a key mechanism for improving scaling, enabling models to focus on the most informative synthetic samples. Rather than generating a large dataset and pruning it afterward, DP efficiently approximates the direct generation of informative samples. We theoretically show how training on challenging, informative examples improves scaling laws and empirically validate that DP achieves better scaling performance with significantly fewer training samples and iterations. On ImageNet-100, DP generates 3.4× fewer samples and requires six times fewer iterations, while on ImageNet-1k, it generates 8× fewer samples with a 30% reduction in iterations, all while achieving superior performance compared to prior work.

Section: Introduction
A key principle underlying learning in human is deliberate practice (DP)-progress is made not by repeating what is already known but by continuously engaging with tasks that stretch the limits of one's abilities (Ericsson et al., 1993). For example, when learning to play the guitar, simply practicing songs that one has mastered does little to improve skill. Instead, targeted practice on challenging tasks and Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). refining learning through feedback, leads to real progress. This principle highlights that effective learning requires exposure to informative and difficult examples rather than passive repetition.
In contrast, most machine learning models are trained on precollected data that remain static throughout training, limiting their ability to dynamically adapt to their own weaknesses. One promising source of data for visual recognition tasks is large-scale pre-trained text-to-image models (Rombach et al., 2022). They provide an essentially infinite source of synthetic training data, presenting an alternative to realworld datasets, which are often expensive or infeasible to curate (Hemmat et al., 2023;Shin et al., 2023;Zhang et al., 2024). With the great promise of text-to-image models, a natural question arises: what is the potential of learning using only synthetic data? Empirical studies show that increasing the volume of synthetic training data often leads to diminishing returns, with performance gains following a power law stagnation (Fan et al., 2024;Tian et al., 2024a). Instead, pruning to remove uninformative examples has proven effective in improving the effectiveness of training with real or synthetic data (Sorscher et al., 2022;Kolossov et al., 2024;Feng et al., 2024).
Inspired by human learning principles and recent advances in generative image models, we propose the Deliberate Practice (DP) for Synthetic Data Generation framework. Unlike static approaches that generate all synthetic training data upfront (Fan et al., 2024;Shin et al., 2023;Hemmat et al., 2023), our framework incorporates a dynamic loop between a diffusion model and a downstream learner throughout the training. More concretely, rather than generating an entire dataset at once and irrespective of the learner and then pruning it to remove uninformative samples, we propose DP to efficiently generate data directly from the pruned distribution of informative samples. By leveraging the learner's prediction entropy to guide the generation process, our approach generates only the most challenging and informative training examples.
Our framework operates dynamically: we begin with an initial set of synthetic data and train a learner until performance on a real validation set plateaus. At this point, the learner's entropy is used to guide the diffusion model to gen-erate new challenging examples. These examples are added to the training set, and the process repeats, ensuring that the model is continually exposed to increasingly informative data throughout training.
This approach aligns with broader goals in machine learning, such as interactive learning environments, continual learning (Kirkpatrick et al., 2017), and active learning (Settles, 2009). By leveraging a dynamic loop, Deliberate Practice reduces inefficiencies from redundant or already learned data, thereby improving the scaling laws of training with synthetic data.
Our contributions are summarized as:
• We introduce the Deliberate Practice for Synthetic Data Generation framework, which dynamically adds new data points when the learner's validation accuracy plateaus [Section 3]. Our framework leverages the learner's prediction entropy to generate challenging synthetic data, improving the scaling behavior of synthetic data (Figures 1 and 4).
• We provide a theoretical analysis of the scaling behavior of a simple model trained on selected examples (Section 4). Using random matrix theory, we characterize the test error as a function of data size and the example selection function, showing improved scaling when prioritizing hard and informative examples.
• We show that entropy-guided sampling approximates generating from an entropy-pruned distribution (Section 2). We empirically validate that DP can improve the validation accuracy compared to direct pruning while being remarkably cheaper in compute up to 5× (Figure 5).
• We demonstrate that DP outperforms prior work on both ImageNet-100 and ImageNet-1k while requiring significantly less data and fewer training iterations. On ImageNet-100, our approach generated 3.4× less samples and completed training in only one-sixth of the iterations used in prior work, yet still achieved superior performance. Similarly, on ImageNet-1k, we generated 8× less samples and reduced the number of iterations by 30%, while outperforming previous results (Table 1).
• Furthermore, DP exhibits strong performance on outof-distribution (OOD) datasets, even outperforming models trained with real data on ImageNet-R and ImageNet-Sketch, with improvements of up to 15% (Table 1).
this section cite: ['b9', 'b27', 'b15', 'b30', 'b40', 'b11', 'b33', 'b21', 'b12', 'b11', 'b30', 'b15', 'b20', 'b29']

Section: Problem Formulation
Problem Setup. Standard supervised learning relies on a large real labeled training set. Here, however, we assume no real training data is available, and instead, we must rely on a generative model to synthesize training examples.
Formally, let Y denote the set of class labels. Our goal is to train a classifier f ϕ : X → Y, parameterized by ϕ, which maps inputs x ∈ X (e.g., images) to labels y ∈ Y. We are given a predefined label set Y, a fixed (small) validation set D val = {(x i , y i )} n i=1 consisting of real data for evaluation, and a generative model g θ capable of sampling synthetic data conditioned on a label, i.e., x ∼ g θ (y). However, no real training data is available, i.e., D tr = ∅. The objective is to train f ϕ using as few generated examples as possible while maximizing generalization to real data as measured by performance on D val . The key challenge is to generate minimal yet effective training data, requiring a principled mechanism to select/generate informative examples.
The Need for Informative Examples. Not all synthetic samples contribute equally to learning. Prior work shows that simply increasing the synthetic dataset size leads to diminishing returns, as many generated samples are redundant or too easy (Fan et al., 2024). Instead, training should focus on examples that maximize learning efficiency.
Given a measure of informativeness for a synthetic sample x, one approach is to generate a large dataset and prune uninformative examples. Formally, let
D pool = {(x i , y i )} N i=1
be a large set of N generated samples. We define a pruned dataset as D ′ := {(x i , y i ) | i ∈ [N ], q i = 1}, where q i ∈ {0, 1} is a selection variable determining whether a data point (x i , y i ) ∈ D pool is retained. The subset size is constrained by m = N i=1 q i . The quantity N/m is referred to as the over-sampling ratio.
Let P and Q denote the distributions of the original and pruned datasets, respectively. The pruning process operates as an importance sampling scheme:
dQ = π dP,(1)
where π is a normalized weighting function that retains the informative samples. The generate-then-prune approach ensures that only informative examples are kept, it is computationally inefficient, as many generated samples are discarded. This motivates the need to devise mechanisms to directly sample the informative examples.
Approximate Sampling of Informative Examples. Suppose that D pool is generated using a diffusion model with induced probability P . The generative process is governed by a reverse SDE (Song & Ermon, 2019):
dx = v(x, t) -g(t) 2 ∇ log p t (x) dt + g(t) dW (t),(2)
where W (t) is a Wiener process, modeling stochastic noise, v(x, t) is a drift term, g(t) is a coefficient controlling the noise level at time t, and ∇ log p t (x) is the score function. Instead of sampling from P , we aim to sample directly from Q as in Eq. ( 1). By Girsanov's theorem (Oksendal, 2013), modifying the probability measure from P to Q introduces a correction term in the reverse SDE:
dx = v(x, t) -g(t) 2 (∇ log p t (x) + ∇ log π(x, t)) dt + g(t) dW (t).(3)
The term ∇ log π(x, t) effectively modifies the score function and biases the sampling distribution according to the weighting function π(x, t). This modification allows approximating direct sampling from the pruned distribution Q, eliminating the need to first sample uniformly from P and later prune the data.
this section cite: ['b11', 'b32', 'b25']

Section: Efficient Entropy-Guided Sampling with DDIM.
We leverage denoising diffusion implicit models (DDIMs) (Song et al., 2020) for efficient sampling. At each step t, the reverse update for generating a conditional sample is:
xt-1 = ξt-1 x0,t + 1 -ξt-1 -σ 2 t • ϵ (t) θ (xt, y) direction pointing to x t + σtϵt random noise ,
where ϵ t is random noise and σ t and ξ t-1 are timedependent coefficients. The term x0,t approximates the final denoised sample:
x0,t = x t - √ 1 -ξ t ϵ (t) θ (x t , y) √ ξ t , (4
) in which ϵ (t)
θ (x t , y) approximates the conditional score function using a pretrained denoising network (Ho & Salimans, 2022):
ϵ θ (x t , y) ≈ (1 + λ)ε θ (x, y) -λε θ (x) (5
)
where λ is called the classifier-free guidance coefficient which controls the strength of conditional sampling on the label.
An efficient way of sampling from a modified diffusion mode as described in Eq. 3 was proposed by Hemmat et al. (2023), where the weighting function is derived from the entropy of the downstream learner, such that,
log π ∝ H(f ϕ (x 0 )) = - y∈Y f ϕ (y | x 0 ) log f ϕ (y | x 0 ).
(6) To compute the entropy as in Eq. 6, we need the denoised sample x 0 . The term x0,t can be used to cheaply approximate entropy mid-generation. This allows direct sampling of high-entropy examples by modifying the score function:
ε(t) θ (x t , y) = ϵ (t) θ (x t , y) + ω∇ xt H(f ϕ (x 0,t )), (7
)
where ω controls the contribution of the entropy-guidance.
In (Hemmat et al., 2023), real data is used to pre-train the learner, enabling an accurate estimation of ∇ xt H(f ϕ (x 0,t )). However, when real data is unavailable, alternative approaches are needed to assess sample informativeness.
In the next section, we propose to leverage the learner itself Algorithm 1 Deliberate Practice for Synthetic Data Generation 1: Input: Class labels Y, Generative model g θ , Validation set D val , Initial dataset size N , New data size P , Patience T max , Evaluation interval τ . 2: Output: Trained classifier f ϕ 3: Initialize: Generate D tr 0 with N examples from g θ . Start training f ϕ with learning-rate warm-up. 4: Set patience counter T ← 0. 5: while training do 6: Update f ϕ on a mini-batch drawn uniformly from D tr k . 7: if (every τ iterations) then 8: Evaluate validation accuracy A(f ϕ , D val ). 9: Reset T ← 0 if accuracy improves; else increment T ← T + 1. 10: end if 11: if T ≥ T max then 12: Generate P new examples D new with feedback: 13:
∇ zt log p(x t | y) = ∇ zt log p θ (z t ) + ω∇ zt H(f ϕ (x 0,t )) 14: Augment training set: D tr k+1 ← D tr k ∪ D new .
15:
Reset T ← 0.
16:
end if 17: end while 18: Finalize: Apply learning rate decay. during training to evaluate entropy and determine the informativeness of generated samples dynamically.
this section cite: ['b31', 'b18', 'b15', 'b15']

Section: The Deliberate Practice Framework for Synthetic Data Generation
In this section, we describe our Deliberate Practice framework, in which we efficiently train the learner with synthetic data in absence of any real data. In particular, we move to a setup where we dynamically expand the dataset throughout the training. Our framework is summarized in Algorithm 1.
The initial training data. The framework begins by generating an initial set of N synthetic training examples D tr 0 = {(x i , y i )} N i=1 using a pre-trained generative model g θ . For each class y i ∈ Y, the generative model samples images x i ∼ g θ (y i ) in a class-conditional manner. The classifier f ϕ starts training on this dataset, with a learning-rate warm-up phase.
Iterative training and additional data. Training proceeds iteratively with a mechanism to dynamically augment the dataset whenever the classifier's performance stagnates. The process alternates between training the classifier and generating new synthetic examples.
Patience mechanism. At regular iteration intervals, τ , the validation accuracy A(f ϕ , D val ) is evaluated. If no improvement is observed for T max intervals (patience threshold), the framework triggers new data generation.
Entropy guided sampling. When the patience mechanism triggers, P new examples D new = {(x j , y j )} P j=1 are generated. We directly generate samples from the entropy pruned distribution through entropy guided sampling. The entropy is computed based on the current stage of the classifier f ϕ . The ω coefficient controls the effect of entropy-guidance. With ω = 0, we fall back into regular sampling of diffusion models, while ω > 0 results in generations that have a higher entropy under the classifier.
Training resumption. The newly generated examples are added to the dataset, D tr k+1 = D tr k ∪ D new . After augmenting the dataset, training resumes with a constant learning rate until the patience mechanism is triggered again. Minibatches are drawn uniformly from the updated pool, which grows dynamically from size N to N +kP after k iterations of augmentation. This cycle is continued until we reach the cool-down phase where the learning rate is decreased and no more new data is added. See Figure 2 for training dynamics of a classifier training with DP.
In Section 4, we provide an intuitive theoretical framework to study the scaling behavior of a simplified DP. In Section 5, we validate the effectiveness of DP in large-scale experiments.
this section cite: []

Section: Training on Informative Examples Improves the Scaling Laws
Before presenting empirical results, we first analyze how selecting informative examples affects the scaling of synthetic data. We study a high-dimensional linear classifier trained with uniform vs. selective sampling and derive an analytic expression for test error using random matrix theory (RMT). Our results show that selecting hard examples improves scaling laws, providing theoretical justification for our approach.
this section cite: []

Section: Theoretical Analysis under an Idealized Setup.
Consider a simple generative model for training data:
x ∼ N (0, Σ), y = sign(w ⊤ 0 x),(8)
where w 0 ∈ R d is the ground-truth labeling function. This gives a distribution P on R d × R.
We study the impact of uniform sampling versus selective sampling of informative examples on generalization. To formalize this, we assume a pool of n i.i.d. training pairs:
X ∈ R n×d , Y ∈ R n . (9
) 0 10k 20k 30k40k
50k Iterations 1.5 2.0 2.5 3.0 3.5 4.0 4.5 Training Loss Total Data Size: 130k + 130k + 130k Total Data Size: 130k + 130k Total Data Size: 130k Warm-up phase Cool-down phase Iterations at which new data is added 0 10k 20k 30k 40k 50k Iterations 0 25 50 60 70 Validation Accuracy Total Data Size: 130k + 130k + 130k Total Data Size: 130k + 130k Total Data Size: 130k Warm-up phase Cool-down phase Iterations at which new data is added A linear classifier ŵ is trained using the following loss:
ŵ = arg min w 1 n n i=1 q i ℓ(w ⊤ x i , y i ) + λ 2 ∥w∥ 2 . (10
)
where ℓ(z, y) = (z -y) 2 /2 is the squared loss, λ > 0 is a regularization parameter, and q i := q(x ⊤ i w s ) is a selection strategy that determines whether an example is included in training based on its projection in a given direction w s ∈ R d , and an arbitrary measurable binary function q : R → {0, 1} which encodes the selection strategy.
The selection/pruning ratio is given by:
p = E[q(x ⊤ w s )] for x ∼ N (0, Σ).(11)
The resulting classifier has a closed-form solution:
ŵ = 1 n RX ⊤ DY, R := 1 n X ⊤ DX + λI d -1 ,(12)
where D ∈ R n×n is a diagonal matrix with D ii = q i .
Our objective is to analyze the asymptotic test error of ŵ:
E test ( ŵ) = P(sign(x ⊤ ŵ) ̸ = y),(13)
where (x, y) is a test example,
this section cite: []

Section: Asymptotic Behavior of the Test Error.
We leverage random matrix theory (RMT) techniques (Couillet & Liao, 2022;Liao & Mahoney, 2021;Firdoussi et al., 2024) to characterize the test error in Eq. ( 13). Our analysis is based on the spectral density of the resolvent matrix R in Eq. ( 12), allowing us to compute the first two moments of yx ⊤ ŵ for a test sample x and derive an expression for the test error. For simplicity, we assume an isotropic setup where Σ = I d and defer the general case to Appendix A.
We shall work in the following so-called high-dimensional proportionate scaling regime
d, n → ∞, d/n → ϕ,(14)
in which the input-dimension d and the sample size n diverge to infinity at the same rate. The scalar ϕ ∈ (0, ∞) captures the effective dimensionality or over-parametrization rate of the problem.
Key Scalars. WLOG, assume ∥w s ∥ = 1. It turns out that the for fixed, pruning, p, the asymptotic test error is fully captured by the following scalars:
ρ := w ⊤ s w 0 /∥w 0 ∥, τ := ρ 1 -ρ 2 , γ := E[q(G)G 2 ], β := 2E[q(G)φ(τ G)], β := 2E[q(G)Φ(τ G)G],(15)
where G ∼ N (0, 1) with pdf φ and cdf Φ. Note that ρ quantifies the alignment between the pruning direction w s and the ground-truth labeler w 0 , while β and γ capture statistical properties of the pruning strategy q.
this section cite: ['b3', 'b23', 'b13']

Section: Spectral functions.
The Stieltjes transform m of the limiting spectral density of the resolvent matrix R is shown in
2 10 Size of Selected Dataset 55 60 65 70 75 80 85 90 Accuracy Select top 10% Select top 50% Select top 80% Select top 90% Select all Lemma 3 to be given by the exact formula (with z := -λ)
m(z) = p -ϕ -z -(p -ϕ -z) 2 -4ϕz 2ϕz ,(16)
and will play an important role in our theory. The above formula represents a somewhat distorted Marchenko-Pastur law. Indeed, the classical MP (Marčenko & Pastur, 1967) corresponds to p → 1 (i.e. no data pruning).
We further define the following auxiliary functions:
s(z) := γ 1 + ϕm(z) , m(z) := 1 s(z) -z , r(z) := ω 2 • m(z) + ω2 • m(z),
with ω := 1 -ρ 2 β, ω := ρ β.
this section cite: ['b24']

Section: Main Result: Test Error Scaling w.r.t Selection Strategy.
Theorem 1. In the limit Eq. ( 14), the classification test error satisfies:
E test ( ŵ) → arccos |m 0 |/ √ ν 0 /π,where
m 0 := ωm(-λ) + ω m(-λ), ν 0 := pϕm ′ (-λ) + r ′ (-λ) - 2ϕm ′ (-λ) 1 + ϕm(-λ)r(-λ)
.
The scaling behavior of test error is fully determined by the six scalars (λ, ϕ, p, ρ, γ, β, β). Importantly, the choice of the data point selection strategy i → q(x ⊤ i w s ) only influences performance through ρ, γ, β, and β.
this section cite: []

Section: EXAMPLE: SELECTING INFORMATIVE EXAMPLES.
Consider a selection function of the form q i = q(x ⊤ i w s ) for all i, where,
q(t) := 1[|t| ≤ ξ] = 1, if |t| ≤ ξ, 0, else,(18)
for some threshold ξ ≥ 0. Such selection strategy selects only the examples near the decision boundary of w s , analogous to using classifier entropy as a selection criterion but simpler to study. Lemma 1 and 2 derive explicit expressions for (γ, β, β).
Figure 3 presents theoretical predictions for test accuracy across different degrees of example selection, showing that selecting hard examples improves scaling laws, reducing the number of training samples needed for the same performance. However, beyond a certain point, excessive pruning degrades performance, as illustrated in Figure 5.
this section cite: []

Section: ADAPTIVE SELECTION STRATEGY.
Data selection relies on a pruning direction w s to select informative/hard examples: i → q(x ⊤ i w s ) ∈ {0, 1}, but these examples are ultimately used to train ŵ. If w s and ŵ are misaligned, what is considered hard by w s may not be hard for ŵ, reducing the effectiveness of selective sampling. In fact, hard examples change over time: an example that was identified hard, might not remain hard are more training is done. To ensure alignment, w s should periodically update to reflect the evolving decision boundary of ŵ. This adaptive selection mechanism motivates the continuous data generation process of DP, as presented in Section 3.
Data selection relies on a pruning direction w s to identify informative or hard examples: i → q(x ⊤ i w s ) ∈ {0, 1}. However, these selected examples are ultimately used to train ŵ, and if w s and ŵ are misaligned, what is considered hard by w s may not be hard for ŵ, reducing the effectiveness of selective sampling. In fact, w s and ŵ deviate from each other the more ŵ is trained on these examples. Moreover, the definition of "hard" changes over time-an example that was initially difficult may become easier as training progresses. To maintain alignment, w s should be periodically updated to reflect the evolving decision boundary of ŵ. This adaptive selection mechanism underpins the continuous data generation process in DP, as presented in Section 3.
this section cite: []

Section: Experiments
For all the experiments, we use the LDM1.5 (Rombach et al., 2022) as the pre-trained text-to-image (T2I) model. We studied four different T2I models and found this model outperforming the rest. For more details see Appendix D.1.
Datasets. We validate our framework on two datasets. ImageNet-100 (Tian et al., 2020;Sarıyıldız et al., 2023), a subset of ImageNet-1k (Deng et al., 2009), containing 100 classes and 5k validation examples, where the real validation set is used for evaluation and the real training set (126,689 examples) serves as a held-out test set. We also conduct experiment ImageNet-1k, using the 50k validation examples to monitor performance and reserving the real training set (1.3 million examples) as a held-out test set.
this section cite: ['b27', 'b34', 'b28', 'b4']

Section: Scaling Laws of Synthetic Data
We train a Vision Transformer (ViT-B) (Dosovitskiy et al., 2021) classifier with synthetic data. We study two scenarios: 1) Static data generation and 2) Deliberate Practice (DP). In all the experiments in this section we have a fixed and controlled setup. We train the models for 100k and 50k iterations for ImageNet-1k and ImageNet-100 respectively. For additional details, see Appendix D.5.
this section cite: ['b7']

Section: Static data generation.
In this setup, all data is generated before training, and the classifier is trained on a fixed dataset. We experiment with different dataset sizes to see its impact on accuracy.
this section cite: []

Section: Deliberate Practice data generation.
Hyperparameters ω and λ are tuned on ImageNet-100 and found effective for ImageNet-1k as well (see Section D.5 for details). We track validation accuracy throughout training and use it to determine when to generate new data, following a patiencebased criterion. To ensure the model has not over-fitted to the validation set, we also report accuracy on the full real training sets of ImageNet-100 and ImageNet-1k, used as held-out test sets.
Figure 4 compares the scaling laws of the Static and Deliberate Practice (DP) on ImageNet-100 and ImageNet-1k. On both datasets, we note that DP scales well with dataset size and it consistently outperforms the Static setup, achieving higher validation accuracy at any given dataset size. On ImageNet-100 we observe that DP can reach the best accuracy of the static setup (with 3 million examples) using only 400k examples. This means that DP requires 7.5× less data to reach the same performance. On ImageNet-1k, we observe that DP can outperform the best accuracy of the static setup (with 13 million examples), using only 640k examples. This translates to DP requiring 20× less data to outperform the Static setup. For additional details on the hyper-parameters of these experiments, see Appendix D.5.1. Refer to Figure 13 for a visualization of how the dataset evolves from the start to the end of training.
this section cite: []

Section: Comparison with Previous Work
We compare DP with prior works on synthetic data generation for image classification (Sarıyıldız et al., 2023;Fan et al., 2024). Specifically, we evaluate setups that use class-names for prompting and publicly available models for sample generation. Performance is assessed on real Im-ageNet (held-out) training and validation sets, as well as on ImageNet-V2 (Recht et al., 2019), ImageNet-Sketch (Wang et al., 2019), ImageNet-R (Hendrycks et al., 2021a), and ImageNet-A (Hendrycks et al., 2021b) to measure out-ofdistribution (OOD) generalization.
The results in Table 1 show that DP outperforms prior benchmarks on both ImageNet-100 and ImageNet-1k while requiring significantly less data and fewer training iterations. On ImageNet-100, DP generated 4.6 million fewer samples and trained for only one-sixth of the iterations compared to previous works, yet achieved superior performance on the real data. Similarly, on ImageNet-1k, DP reduced sample generation by 56.2 million and cut training iterations by over 30%, while still outperforming previous results.
Furthermore, models trained with DP exhibit strong performance on out-of-distribution datasets, even surpassing models trained on real data on ImageNet-R and ImageNet-Sketch, with improvements of up to 15%.
this section cite: ['b28', 'b11', 'b26', 'b37']

Section: Connection Between Pruning and DP
In Section 2, we discussed how DP approximates direct sampling from a pruned distribution. Here, we validate this experimentally on ImageNet-100 using two setups:
1. Oversampling then Pruning: Generate a large pool and select high-entropy samples.
2. Direct entropy-guided generation: Generate only informative samples (a special case of DP with a single step of data addition).
We start with 130k generated samples (regular vanilla sampling), train for 17k iterations, then add a one-time additional 130k samples, increasing the total data size to 260k and training for an additional 33k iterations.
In setup 1, we vary the pool size, ranging from no pruning (130k pool) up to an oversampling ratio of 18 (2.4M pool), selecting the top 130k high-entropy samples. In setup 2, we generate exactly 130k entropy-guided samples, varying the entropy-gauidance coefficient.
Figure 5 (a, b) shows that both methods improve performance up to a point, after which excessive selection of high-entropy samples leads to degradation-likely due to selecting high-entropy but harmful outliers. This aligns with our theoretical predictions in Figure 5 (c).
Regarding computational costs, generating a single image with entropy-guidance on an Nvidia H100 takes 1.82× longer than standard vanilla sampling. However, achieving similar performance through oversampling requires significantly more data, leading to a linear increase in cost.  As a result, DP is 5× more efficient while also providing higher absolute improvements compared to pruning-based selection. See Figure 5 for details and Figure 11 for some visualizations.
this section cite: []

Section: The Evolution of Hard Examples Over Time
"Does the sample hardness change as training progresses?"
To answer this question, Figure 6 tracks the error on examples that were misclassified at the time they were added. As expected, once introduced, the model gradually learns to classify them correctly. However, an interesting trend emerges: even before these examples were added, their error was lower than at the moment of inclusion. This suggests that the notion of hardness is dynamic-what is considered challenging at one point may become easier over time. Conversely, examples that were once easy might later become difficult due to shifts in the learned decision boundaries. This highlights a key limitation of static pruning approaches and underscores the importance of dynamically adapting the selection of informative examples throughout training, as done in Deliberate Practice (DP). See Figure 12 for some visualization of generations through training.
this section cite: []

Section: Related Work
Synthetic data for training neural networks. Synthetic data has become a powerful tool for training machine learning models across various domains. For instance, text-toimage diffusion models have been successfully used for visual representation learning (Astolfi et al., 2023;Li et al., 2025;Tian et al., 2024a;b;Sarıyıldız et al., 2023). However, limitations of synthetic data are highlighted by Fan et al. (2024), emphasizing the importance of generating more challenging and informative examples. Addressing distribution shifts between synthetic and real data, Hemmat et al. (2023) and Yuan et al. (2023) propose synthesizing training data that matches real data distributions or conditioning on real examples to reduce this gap. Expanding small-scale datasets has also been studied, see e.g. Zhang More training Iterations DP coefficient increases Figure 6: Error trajectories of hard (misclassified) examples added at different training stages. The red curve highlights the first batch of added data for better visibility, but the same trend applies to all batches. Notably, even before being trained on, these examples exhibit a lower error rate than at their point of inclusion, indicating that hardness is not static, it evolves throughout training. et al. (2024). Another related line of work involves using
VLMs and LLMs to generate descriptions for augmenting datasets (Dunlap et al., 2023).
Synthetic data is increasingly used to train (LLMs). For example, LLaMA3 (Grattafiori et al., 2024) employs AIgenerated data for fine-tuning. Similarly, self-play approaches, e.g., Yuan et al. (2024), align with our framework by generating increasingly difficult examples for training.
Continual learning and active learning. Our work is also closely related to principles from active learning (Bang et al., 2024;Evans et al., 2023) and continual learning, which prioritize iterative model updates with tailored data. These methods highlight the importance of selecting informative samples based on the model's current state. (Sorscher et al., 2022) showed that pruning static datasets using metrics like margin scores can improve scaling laws by retaining the most informative examples, albeit in a non-adaptive manner.
Challenges and risks of synthetic data.
The challenges of training models on synthetic data, have gained significant attention. Dohmatob et al. (2024a;b) studied "model collapse", a phenomenon where iterative training on synthetic data degrades performance. They emphasize that data verification mechanisms can mitigate this risk and enable scaling with synthetic data. Similarly, our framework by generating informative examples through a dynamic loop, improves sample efficiency.
this section cite: ['b0', 'b22', 'b28', 'b11', 'b15', 'b39', 'b8', 'b14', 'b38', 'b2', 'b10', 'b33']

Section: Conclusion
We introduced Deliberate Practice for Synthetic Data Generation, a framework that improves scaling laws by dynamically generating challenging and informative training examples. Unlike traditional methods that rely on static datasets, our approach approximates generating data directly from a pruned distribution, reducing inefficiencies and ensuring models continuously training on informative samples. We provided theoretical insights into the benefits of training on pruned distributions and empirically demonstrated that our method significantly improves performance while requiring fewer training iterations. Our results on ImageNet-100 and ImageNet-1K show that Deliberate Practice achieves superior accuracy with far less data and compute, outperforming previous state-of-the-art. Our work highlights the potential of structured synthetic data generation in advancing efficient and adaptive learning.
this section cite: []

Section: References
Ref_id:b0 Title: Instance-conditioned gan data augmentation for representation learning Year: (2023)
Ref_id:b1 Title: Consistency-diversity-realism pareto fronts of conditional image generative models Year: (2024)
Ref_id:b2 Title: Active prompt learning in vision language models Year: (2024)
Ref_id:b3 Title: Random Matrix Methods for Machine Learning Year: (2022)
Ref_id:b4 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b5 Title: Strong model collapse Year: (2024)
Ref_id:b6 Title: A tale of tails: Model collapse as a change of scaling laws Year: (2024)
Ref_id:b7 Title: An image is worth 16×16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b8 Title: Diversify your vision datasets with automatic diffusion-based augmentation Year: (2023)
Ref_id:b9 Title: The role of deliberate practice in the acquisition of expert performance Year: (1993)
Ref_id:b10 Title: Bad students make great teachers: Active learning accelerates large-scale visual understanding Year: (2023)
Ref_id:b11 Title: Scaling laws of synthetic images for model training... for now Year: (2024)
Ref_id:b12 Title: Beyond model collapse: Scaling up with synthesized data requires reinforcement Year: (2024)
Ref_id:b13 Title: Maximizing the potential of synthetic data: Insights from random matrix theory Year: (2024)
Ref_id:b14 Title: The llama 3 herd of models Year: (2024)
Ref_id:b15 Title: Feedback-guided data synthesis for imbalanced classification Year: (2023)
Ref_id:b16 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b17 Title: Natural adversarial examples Year: (2021)
Ref_id:b18 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b19 Title: Minicpm: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b20 Title: Overcoming catastrophic forgetting in neural networks Year: (2017)
Ref_id:b21 Title: Towards a statistical theory of data selection under weak supervision Year: (2024)
Ref_id:b22 Title: Enhancing view quality with pretrained generative model for self-supervised learning Year: (2025)
Ref_id:b23 Title: Hessian eigenspectra of more realistic nonlinear models Year: (2021)
Ref_id:b24 Title: Distribution of eigenvalues for some sets of random matrices Year: (1967-04)
Ref_id:b25 Title: Stochastic differential equations: an introduction with applications Year: (2013)
Ref_id:b26 Title: Do imagenet classifiers generalize to imagenet? Year: (2019)
Ref_id:b27 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b28 Title: Fake it till you make it: Learning transferable representations from synthetic imagenet clones Year: (2023)
Ref_id:b29 Title: Active learning literature survey Year: (2009)
Ref_id:b30 Title: Fill-up: Balancing long-tailed data with generative models Year: (2023)
Ref_id:b31 Title: Denoising diffusion implicit models Year: (2010)
Ref_id:b32 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b33 Title: Beyond neural scaling laws: beating power law scaling via data pruning Year: (2022)
Ref_id:b34 Title: Contrastive multiview coding Year: (2020)
Ref_id:b35 Title: Learning vision from models rivals learning vision from data Year: (2024)
Ref_id:b36 Title: Synthetic images from text-to-image models make strong visual representation learners Year: ()
Ref_id:b37 Title: Learning robust global representations by penalizing local predictive power Year: (2019)
Ref_id:b38 Title: Self-play fine-tuning of diffusion models for text-to-image generation Year: (2024)
Ref_id:b39 Title: Realfake: Effective training data synthesis through distribution matching Year: (2023)
Ref_id:b40 Title: Expanding small-scale datasets with guided imagination Year: (2024)
