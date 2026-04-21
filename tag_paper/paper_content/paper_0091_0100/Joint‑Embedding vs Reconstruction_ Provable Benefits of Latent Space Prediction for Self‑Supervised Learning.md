Title: Joint-Embedding vs Reconstruction: Provable Benefits of Latent Space Prediction for Self-Supervised Learning
Abstract: Reconstruction and joint-embedding have emerged as two leading paradigms in Self-Supervised Learning (SSL). Reconstruction methods focus on recovering the original sample from a different view in input space. On the other hand, joint-embedding methods align the representations of different views in latent space. Both approaches offer compelling advantages, yet practitioners lack clear guidelines for choosing between them. In this work, we unveil the core mechanisms that distinguish each paradigm. By leveraging closed-form solutions for both approaches, we precisely characterize how the view generation process, e.g. data augmentation, impacts the learned representations. We then demonstrate that, unlike supervised learning, both SSL paradigms require a minimal alignment between augmentations and irrelevant features to achieve asymptotic optimality with increasing sample size. Our findings indicate that in scenarios where these irrelevant features have a large magnitude, joint-embedding methods are preferable because they impose a strictly weaker alignment condition compared to reconstruction-based methods. These results not only clarify the trade-offs between the two paradigms but also substantiate the empirical success of joint-embedding approaches on real-world challenging datasets.

Section: Introduction
Training deep neural networks to extract informative data representations is central to AI. Numerous families of methods pursue this goal [52]. In supervised learning, one does so by prescribing labels that encode what is considered informative in the data. While this has been the dominant approach to representation learning over the past decades, it has become clear that labels are often overly specialized. Such specialization prevents learning representations that transfer across an ever-increasing diversity of downstream tasks [24,37]. Self-Supervised Learning (SSL) has emerged as an alternative that moves away from labels [4,50,14]. In SSL, one does not assume a priori what is informative; instead, one specifies which variations are uninformative and should be disregarded. Identifying, a priori, the invariances a representation should satisfy is a broadly applicable principle. For instance, many downstream tasks involving natural images, such as recognition, counting, and segmentation, are inherently robust to minor changes in color or illumination. Consequently, these tasks benefit from representations that exhibit such invariances, typically encoded through a data-augmentation process. Two primary families of methods have emerged to learn representations using this principle: reconstruction-based and joint-embedding approaches. An encoder f E together with a decoder f D is trained to recover x from τ (x). Right: Joint embedding problem of Equation (SSL-JE): two independent augmentations τ 1 , τ 2 ∼ T of the same x are mapped by f W to nearby representations, while embeddings of different inputs are pushed apart.
this section cite: ['b51', 'b23', 'b36', 'b3', 'b49', 'b13']

Section: The Reconstruction-based approach
Reconstruction-based approaches train models by augmenting an input signal, typically by adding noise or masking, and then training the model to restore the original input [41,60,31,67] (left side of Figure 1). This process encourages the model to learn meaningful internal representations of the data's underlying structure and content to enable successful reconstruction. However, because the learning signal arises from minimizing reconstruction error in the input space, the model is naturally steered toward subspaces that explain the majority of the input's variance [65,8]. Whether such variance-explaining features are also the most semantically discriminative or useful for downstream tasks depends strongly on the data modality.
In language, reconstruction-based learning, as used in large language models, is highly effective because textual tokens represent compact, semantically meaningful units that already abstract away most low-level variability. Although data quality may vary, language in itself is a highly compressed and rich modality where reconstruction can prove highly successful. Predicting a missing token provides a learning signal that operates directly in semantic space: to succeed, the model must infer the contextual and syntactic relationships that determine meaning, rather than replicate surface patterns. Consequently, minimizing reconstruction error encourages the emergence of abstract relational representations, capturing compositionality, long-range dependencies, and discourse coherence that align closely with human notions of meaning and reasoning [19,30,62].
In contrast, in visual domains, variance-explaining features often emphasize aspects of the data that are statistically dominant but semantically shallow. Unlike language, visual data are essentially sensorial recordings of the physical world, capturing raw information without inherent semantic compression. As a result, pixel-level reconstruction objectives tend to drive models toward capturing local statistics and textures that account for most of the input's variance, rather than the higher-order structures and object-level relationships that underpin semantic understanding. This local bias can result in representations that are well-suited for low-level perceptual fidelity but suboptimal for recognition, categorization, or other tasks that depend on global context and semantic abstraction [6,28]. Consequently, purely reconstruction-based approaches in computer vision often struggle to produce features that generalize well across tasks without additional supervision or adaptation. Fine-tuning is thus frequently necessary to bridge the gap between variance-focused representations and those that encode meaningful, task-relevant semantics [31].
this section cite: ['b40', 'b59', 'b30', 'b66', 'b64', 'b7', 'b18', 'b29', 'b61', 'b5', 'b27', 'b30']

Section: The Joint-embedding approach
Joint-embedding methods, in contrast, operate entirely in latent space (right side of Figure 1). Their objective is to produce similar representations for different augmented views of the same input while ensuring that representations of distinct samples remain dissimilar. This separation can be enforced explicitly through a contrastive loss [14,33], or implicitly via architectural mechanisms such as self-distillation, stop-gradient operations, momentum encoders, or predictor heads that stabilize training and prevent representational collapse even without negative samples [13,26,66,38].
Unlike reconstruction-based approaches, joint-embedding methods do not predict in the input space and are therefore less biased toward capturing high-variance components of the signal. Empirically, joint-embedding frameworks have shown strong performance across domains where the input signal is high-dimensional and semantically diffuse. Successful applications span histopathology [73], Earth observation [61], and video representation learning [9]. Despite this progress, the mechanisms through which latent consistency objectives outperform reconstruction-based ones remain poorly understood, motivating the analysis presented in this work.
this section cite: ['b13', 'b32', 'b12', 'b25', 'b65', 'b37', 'b72', 'b60', 'b8']

Section: Contributions
The critical role of the prediction target in SSL, specifically whether to predict in the input space (reconstruction) or the latent representation space (joint-embedding), has been demonstrated numerous times [1,6]. However, it remains unclear when to favor one approach over the other. This work clarifies when to prefer each. Our key findings can be summarized as follows.
1. We derive closed-form solutions for both reconstruction-based (Theorem 3.1) and jointembedding (Theorem 3.2) SSL linear models. This enables a precise characterization of data augmentation impacts, analogous to well-known results in supervised learning [11].
2. We then leverage these results to show that optimally aligning the augmentations with the irrelevant components of the input signal can effectively eliminate these components and recover optimal performance for both families of methods (Propositions 4.3 and 4.4). However, in contrast to the supervised learning scenario (Proposition 4.2), simply increasing the sample size cannot overcome any misalignment between the augmentation and the noise (Propositions 4.3 and 4.4). 3. By inspecting the alignment requirements for both reconstruction and joint-embedding methods, we show that in settings with low-magnitude irrelevant noise features, reconstruction methods are preferable, as they require fewer tailored augmentations (Corollary 4.5). Conversely, in scenarios with high-magnitude irrelevant noise features, i.e. , where such features significantly impact the input signal, joint-embedding methods should be preferred, as they impose a strictly weaker alignment condition than reconstruction methods (Corollary 4.5). 4. In Section 5, we experimentally validate these findings on both vectorial and image data. We demonstrate that joint-embedding methods such as DINO [13] and BYOL [27] are considerably more robust to severe data corruption than reconstruction-based methods like MAE [31] (Section 5.2). In Appendix D, we further provide experimental validation for key results from our theoretical analysis. These experiments show that: (i) SSL methods exhibit significantly greater sensitivity to corruptions compared to supervised learning methods (Appendix D.2 and Figure 2); and (ii) SSL performance in noisy data settings is enhanced by aligning augmentations with the underlying noise (Appendix D.3 and Figure 2). Interestingly, when using a linear model f V : x → Vx with V ∈ R ℓ×d , the effect of data augmentation in Equation (SL) can be explicitly characterized as a Tikhonov regularization problem as shown in Lemma B.1 [7,46,11] which proof is provided in Appendix B:
1 n i∈[[n]] E τ ∼T ∥y i -Vτ (x i )∥ 2 2 = ∥V∥ 2 Σ + 1 n i∈[[n]] ∥y i -VE τ ∼T [τ (x i )] ∥ 2 2 (1
)
where ∥V∥ 2 Σ = Tr(VΣV ⊤ ) and
Σ := 1 n i E τ ∼T τ (x i )τ (x i ) ⊤ -E τ ∼T [τ (x i )] E τ ∼T [τ (x i )] ⊤ (Cov)
denotes the covariance of the augmented samples. Therefore, the effect of data augmentation within supervised learning using linear models is well understood from a theoretical standpoint.
Lack of Foundations in Self-Supervised Learning. Similar results are lacking for SSL, where the explicit effect of data augmentation for linear models has not been rigorously studied.
Despite recent efforts to elucidate the underlying principles [64,56,39,36,29,22,71,63], these methods remain only superficially understood [53]. A robust statistical framework is still lacking to fully comprehend SSL methods and to position them relative to their supervised learning counterparts [3]. Key open questions involve precisely characterizing the role of data augmentation in shaping final representations within both reconstruction and joint-embedding frameworks. This work aims to lay the foundation for filling this gap.
this section cite: ['b0', 'b5', 'b10', 'b2', 'b12', 'b26', 'b30', 'b6', 'b45', 'b10', 'b63', 'b55', 'b38', 'b35', 'b28', 'b21', 'b70', 'b62', 'b52', 'b2']

Section: Augmentation-Aware Closed-Form Solutions
In this section, we derive closed-form solutions for the two main families of SSL methods: reconstruction-based and joint-embedding approaches. To the best of our knowledge, the following results are the first instances of closed-form solutions for SSL that are directly parameterized by the data augmentation structure. This stands in contrast to previous solutions highlighted in [5], which focused on the dependency graph between augmented samples and were unaware of augmentations. These results will then allow us to analyze how augmentations affect the learned representations in both families of SSL methods.
In line with previous theoretical works focused on analytical tractability [12,5,47,54,57], we study models that are linear in their parameters. Note that these models can produce arbitrary nonlinear predictions via appropriate feature maps [3] and are known to describe regimes of wide neural networks [44].
this section cite: ['b4', 'b11', 'b4', 'b46', 'b53', 'b56', 'b2', 'b43']

Section: Reconstruction-Based Self-Supervised Learning
We first consider reconstruction-based SSL models. The problem can be framed as follows, where T is the data augmentation distribution:
min E,D 1 n i∈[[n]] E τ ∼T ∥x i -f D (f E (τ (x i )))∥ 2 2 . (SSL-RC)
In this formulation, each data sample is augmented and then passes through an encoder f E , followed by a decoder f D . The objective is to minimize the reconstruction error between the original sample and the reconstructed one. This methodology is analogous to the Denoising Auto-Encoder [59], Masked Auto-Encoder [32] and similar frameworks. Interestingly, the reconstruction problem can be solved in closed form when considering linear models for both encoder and decoder. All proofs can be found in Appendix A. Assume that 1 n X ⊤ X + Σ is positive definite where Σ is defined in Equation (Cov). Consider the singular value decomposition:
1 n X ⊤ X 1 n X ⊤ X + Σ - 1 2 = RΦP ⊤ (2
)
where R ∈ R d×d and P ∈ R d×d are orthogonal and Φ := diag(ϕ 1 , . . . , ϕ d ) with ϕ 1 ≥ • • • ≥ ϕ d ≥ 0. Solutions of Equation (SSL-RC) for f E : x → Ex and f D : x → Dx take the form:
E ⋆ = TP ⊤ k 1 n X ⊤ X + Σ - 1 2 and D ⋆ = R k Φ k T -1 , (3
)
where T is any invertible matrix in R k×k , P k and R k are the first k columns of P and R respectively, and Φ k = diag(ϕ 1 , . . . , ϕ k ).
this section cite: ['b58', 'b31']

Section: Joint-Embedding-Based Self-Supervised Learning
We now consider a joint-embedding SSL problem formulated as follows, where f W is the SSL model and T is the data augmentation distribution:
min W 1 n i∈[[n]] E τ1,τ2∼T ∥f W (τ 1 (x i )) -f W (τ 2 (x i ))∥ 2 2 , subject to 1 n i∈[[n]] E τ ∼T f W (τ (x i ))f W (τ (x i )) ⊤ = I k . (SSL-JE)
In the above Equation (SSL-JE), the objective represents the usual invariance term which ensures consistency between two augmented views of the same sample and is a common component of joint-embedding methods. The constraint enforces orthonormality in the learned representations, promoting diversity in the representation space [64] and thus preventing collapse. Most joint-embedding models incorporate a similar repulsion term, either explicitly within the loss function, such as in Barlow-Twins [70], SimCLR [14], VICReg [10], and MoCo [33], or implicitly through architectural design choices, as demonstrated by BYOL [26] and DINO [13]. In our case, we rely on the sum of the outer products of the representation vectors. This approach closely resembles VICReg [10], specifically its covariance regularization term. Interestingly, under the unifying formalism presented in [25], most popular joint-embedding methods can be framed with this simple repulsive term.
The problem of Equation (SSL-JE) can also be solved in closed form when considering a linear SSL model, as formalized below.
Theorem 3.2. Let S :=
1 n i E τ ∼T τ (x i )τ (x i ) ⊤ , G := 1 n i E τ ∼T [τ (x i )] E τ ∼T [τ (x i )]
⊤ . Assume that S is positive definite. Consider the eigendecomposition:
S -1 2 GS -1 2 = QΩQ ⊤(4)
where Ω = diag(ω 1 , . . . , ω d ) with ω 1 ≥ • • • ≥ ω d . Solutions of Equation (SSL-JE) for a linear model f W : x → Wx take the form:
W ⋆ = UQ ⊤ k S -1 2 , (5
)
where Q k = (q 1 , . . . , q k ) and U is any orthogonal matrix of size k × k.
this section cite: ['b63', 'b69', 'b13', 'b9', 'b32', 'b25', 'b12', 'b9', 'b24']

Section: Augmentation Alignment Requirements in Self-Supervised Learning
In this section, we build on the results of Section 3 to evaluate the ability of both families of SSL models to reach optimal performance. To define such notion of optimality, we model our data as having k important signal components and d -k pure noise components. These noise components represent the variations that SSL methods are typically designed to be invariant to, e.g. background noise for image classification tasks. An ideal SSL encoder would retain the k informative important dimensions while discarding the d -k noise components.
We formalize this scenario in Section 4.1, where a parameter α is introduced to control the alignment between the irrelevant features and the augmentations. In Section 4.2, we demonstrate that supervised learning models can effectively achieve optimal performance either when augmentations are well aligned with the irrelevant noise features at finite sample sizes or when the sample size is large, regardless of the augmentation employed. In Section 4.3, we show that, unlike supervised learning, SSL necessitates a sufficiently good alignment to achieve optimal performance, even in the infinite sample limit. Finally in Section 4.4, we compare the alignment requirements of joint-embedding and reconstruction-based SSL methods, thus providing insights into the characteristics of both families of methods.
this section cite: []

Section: Data, Noise and Augmentation
We consider an input dataset with two parts: important features and irrelevant noise. Optimal performance on downstream tasks is achieved when using only the important features. Let X = LKQ ⊤ be the singular value decomposition of the important features, where K = diag(κ 1 , . . . , κ d ) is the diagonal matrix of singular values. Each sample x i is corrupted by additive Gaussian noise constituting the irrelevant features:
∀ i ∈ [[n]], x i = x i + γ i , γ i ∼ N (0, Γ),(6)
with γ i drawn independently across i ∈ [[n]] and where Γ ∈ R d×d is positive semi-definite. For simplicity, we assume that Γ is diagonalized by the same orthonormal matrix Q from the SVD above i.e. Γ = QΛ Γ Q ⊤ where Λ Γ = diag(λ Γ 1 , . . . , λ Γ d ). The matrix X = (x 1 , . . . , xn ) ⊤ ∈ R n×d then forms the corrupted input data. We assume that the important features are concentrated in exactly k ≥ 1 components, meaning κ i > 0 for i ∈ [[k]] and κ i = 0 for i > k. These components are referred to as the important components. Additionally, we assume that the irrelevant noise are null in these k components, i.e. λ Γ i = 0 for all i ∈ [[k]], and strictly positive otherwise, i.e. λ Γ i > 0 for i ∈ [[k + 1 : d]]. We refer to the [[k + 1 : d]] components as the noise components.
this section cite: []

Section: Data augmentation.
Let Θ ∈ R d×d be positive semi-definite and diagonalized by Q i.e. Θ = QΛ Θ Q ⊤ where Λ Θ = diag(λ Θ 1 , . . . , λ Θ d ). We consider the augmentation distribution,
∀ α ≥ 0, T (α) := τ : R d → R d τ (x) = x + θ + α γ, θ ∼ N (0, Θ), γ ∼ N (0, Γ) , (7
)
where θ and γ are drawn independently for each transformation. Note that the term γ follows the same distribution as the noisy irrelevant features. Increasing the magnitude of α thus aligns the data augmentation with these irrelevant features. Remark 4.1. One can extend our result to augmentations beyond Gaussians by considering any augmentation of covariance (defined in Cov) Θ + α 2 Γ [46].
this section cite: ['b45']

Section: Supervised Learning Consistency Regardless of Augmentations
We first analyze the behavior of supervised learning models by identifying regimes in which the supervised model effectively disregards the noisy irrelevant features in X. This provides the foundation for pinpointing key differences between supervised and SSL models in Section 4.3. We rely on the data augmentation T (α), where α ∈ R + controls the alignment between the data corruption and the augmentation process as presented in Section 4.1.
Proposition 4.2. [Supervised Learning] Let V ⋆ (resp. V ⋆ ) be the linear model solving Equation (SL) with augmentation T (α) for X (resp. the corrupted X). The limit:
V ⋆ --→ a.s. V ⋆ (8
)
holds almost surely in either of the following regimes:
• as α → +∞ (perfect augmentation-noise alignment) for any fixed sample size n ∈ N.
• as n → +∞ (infinite samples) for any fixed alignment α ≥ 0.
The above result shows that, when performing supervised learning with corrupted data, the model can achieve the same performance as if it were trained only on the important features (thus achieving optimal performance) if either of the following conditions holds: i) The data augmentation process is well aligned with the noise corrupting the inputs (α large). ii) A sufficiently large sample size is available to compensate for any misalignment between the augmentation and the input noise (n large).
this section cite: []

Section: Self-Supervised Learning Requires Aligned Augmentation and Noise
Building on the closed-form expressions for SSL provided in Theorems 3.1 and 3.2, we are now interested in studying the ability of SSL models to achieve optimal performance when trained on the corrupted dataset X, as defined in Section 4.1.
this section cite: []

Section: Proposition 4.3. [Reconstruction] Let E ⋆ (resp. E ⋆ ) be the linear (encoder) model solving Equation (SSL-RC)
for X (resp. the corrupted X). The limit:
E ⋆ --→ a.s. E ⋆ (9
)
holds 2 almost surely in either of the following regimes:
• as α → +∞ (perfect augmentation-noise alignment) for any fixed sample size n ∈ N.
• as n → +∞ (infinite samples), if and only if the alignment α ≥ 0 satisfies:
α 2 > α 2 RC := max i∈[[k+1:d]] λ Γ i η 2 - λ Θ i λ Γ i -1 where η = min i∈[[k]] 1 n κ 2 i 1 n κ 2 i + λ Θ i . (10
)
Proposition 4.4. [Joint-Embedding] Let W ⋆ (resp. W ⋆ ) be the linear model solving Equation (SSL-JE) for X (resp. the corrupted X). The limit:
W ⋆ --→ a.s. W ⋆ (11
)
holds 3 almost surely in either of the following regimes:
• as α → +∞ (perfect augmentation-noise alignment) for any fixed sample size n ∈ N.
• as n → +∞ (infinite samples), if and only if the alignment α ≥ 0 satisfies:
α 2 > α 2 JE := max i∈[[k+1:d]] 1 -δ δ - λ Θ i λ Γ i where δ = min i∈[[k]] 1 n κ 2 i 1 n κ 2 i + λ Θ i . (12
)
The above Propositions 4.3 and 4.4 reveal that, when augmentations are well aligned with the noise i.e. when α is large enough, undesired noise components are removed from the obtained SSL representation for both families of models, even when combined with other augmentations. However, the alignment requirement persists even as the sample size n becomes arbitrarily large. Therefore, in SSL, achieving optimal performance requires that the data augmentation process be sufficiently well aligned with the irrelevant noise features in the data. This marks a key difference from supervised models. Unlike SSL, supervised models can overcome misalignment between augmentations and noise with enough samples, as it learns robustness by observing different noise realizations across identically labeled data. This underscores the critical role of augmentations in SSL. Figure 2 illustrates this by visualizing the embedding spaces of a supervised model and the VICReg [10] SSL model, revealing the latter's susceptibility to noise when augmentations lack proper alignment. This finding is consistent with an empirical study by [51], which concluded that improving augmentations is more impactful than altering architectural designs.
Interestingly, the above results reveal that reconstruction (Proposition 4.3) and jointembedding (Proposition 4.4) methods exhibit different alignment requirements to achieve optimal performance. We next analyze these differences.
this section cite: ['b9', 'b50']

Section: Comparison of Joint-Embedding and Reconstruction-Based Methods via Augmentation Alignment Requirement
Leveraging Propositions 4.3 and 4.4, we obtain the following key result.
Corollary 4.5. Let α JE , δ, α RC , and η be defined as in Proposition 4.4 and Proposition 4.3.
• If max i∈[[k+1:d]] λ Γ i < η 2 δ (low noise), then α JE > α RC (reconstruction is preferable).
• If min i∈[[k+1:d]] λ Γ i > η 2 δ (high noise), then α JE < α RC (joint-embedding is preferable).
This result shows that when the spectral norm of the noise covariance Γ is small, reconstruction-based methods impose a less stringent alignment compared to joint-embedding methods. Conversely, when the noise magnitude is large, joint-embedding methods exhibit lower sensitivity to the augmentation-noise alignment compared to their reconstruction-based counterparts. Ultimately, the goal is to minimize the alignment requirement, as the irrelevant components are typically unknown in real-world applications.
Reconstruction-based approaches are therefore well-suited for scenarios with weak, irrelevant noise features. Intuitively, the core important components in such settings possess the greatest magnitude and are consequently prioritized by the model during reconstruction. Due to their reconstruction objective, these methods depend less on data augmentations, which explains their superior performance and robustness over joint-embedding techniques in this particular context.
However, when strong noise features obscure the important features within the raw input signal, joint-embedding methods demonstrate greater robustness. This is because jointembedding techniques prioritize latent space prediction, thereby bypassing the need to reconstruct irrelevant noise components as model outputs. Consequently, in real-world scenarios where the extent of irrelevant features (typically image backgrounds or experimental batch effects in scRNA-seq data) cannot be precisely quantified, joint-embedding approaches appear more reliable. This preference is further underscored by the principle that highamplitude noise naturally exerts a more significant impact on the final representation than low-amplitude noise, making robustness crucial when noise levels are uncertain. This practical advantage also explains the community's preference for joint-embedding approaches in challenging datasets [73,43,20].
this section cite: ['b72', 'b42', 'b19']

Section: Key takeaway.
When irrelevant features have low magnitude and there is limited prior information on effective augmentations, reconstruction is preferable. In contrast, when these irrelevant features are non-negligible (as is common with real-world data) or effective augmentations can be identified, joint-embedding is preferable.
this section cite: []

Section: Experiments
This section validates the theoretical findings of Section 4 through experiments on linear models (Section 5.1) and deep networks (Section 5.2). These experiments confirm that the results from the linear model are consistent with those observed in the nonlinear setting of deep networks.
this section cite: []

Section: Experiments With Linear Models
We first validate the theoretical results of Section 4 through experiments with linear models. Data features are corrupted by adding synthetic Gaussian noise, allowing us to precisely control the noise magnitude and its alignment with data augmentations. The details of our experimental design are provided in Appendix C. Our primary results are illustrated in Figure 3. This figure effectively illustrates contrasting behaviors between the various types of methods as sample size and noise magnitude vary. On the left, one can notice that the supervised model achieves optimal performance with either increasing sample size or increasing alignment, with any augmentation and regardless of the noise magnitude, confirming the result of Proposition 4.2. In contrast, SSL models exhibit different sensitivities, as predicted in Propositions 4.3 and 4.4. The middle panel shows that joint-embedding indeed requires a minimal alignment between augmentation and noise to reach optimal performance (as predicted in Proposition 4.4). Notably, joint-embedding maintains robustness even with increasing noise magnitude, as shown in Corollary 4.5. On the right panel, we can see that under weak noise conditions, reconstruction is robust to the choice of augmentation. However, as noise becomes stronger, reconstruction performance degrades and necessitates a strong alignment between augmentation and noise. These observations confirm the results of Propositions 4.3 and 4.4 and Corollary 4.5. These findings are further supported by experiments on other datasets, including Fashion-MNIST, Kuzushiji-MNIST and the singlecell RNA-seq data from [49], presented in Figures 4 to 7 in Appendix C. All experimental outcomes align with and confirm the theoretical insights detailed in Section 4.
this section cite: ['b48']

Section: Experiments with Deep Networks on Images with Various Corruptions
We conduct experiments using both ViT [21] and ResNet [34] architectures. Our evaluation focuses on top-1 linear probing accuracy on ImageNet-100 [18]. ImageNet images inherently contain non-negligible features irrelevant to the classification task [6] (e.g. background noise elements), which contributes to the superior performance of joint-embedding methods over reconstruction methods, as demonstrated in several benchmarks [17,15]. To further introduce and control the magnitude of irrelevant noise features, we utilize the ImageNet-C corruptions [35], which offers corruptions at various severity levels. The results are presented in Table 1.
The performance of MAE (ViT) [31], which uses a reconstruction objective, is much more affected by the increasing corruptions than DINO (ViT) [13] and BYOL (ResNet) [26], which use a joint-embedding objective. Indeed, there is a 25.1% average drop in accuracy for MAE when the severity of the corruption increases from 1 to 5, while DINO and BYOL only experience a 10.5% drop and 12.4% drop, respectively. All experimental details are provided in Appendix D.1.
We perform further ablation studies in the appendix (Appendix D) highlighting this paper's key results. Specifically, our experiments confirm that: (i) SSL is considerably more sensitive to the alignment between augmentations and noise than supervised learning (see Appendix D.2), and (ii) aligning augmentations with the underlying noise can enhance the performance of SSL models in noisy data settings (see Appendix D.3).
this section cite: ['b20', 'b33', 'b17', 'b5', 'b16', 'b14', 'b34', 'b30', 'b12', 'b25']

Section: Conclusion
A growing body of work demonstrates that joint-embedding methods often outperform reconstruction methods on real-world datasets, particularly where extracting useful features for downstream tasks from raw signals is challenging [40,2]. This is further supported by a consistent empirical finding: reconstruction methods typically necessitate fine-tuning to address the inherent misalignment between the features they learn and those that are perceptually useful for downstream applications [72,6,45,68]. In this work, we have established a theoretical framework to explain this phenomenon. Our analysis provides clear guidelines for practitioners: opt for reconstruction when irrelevant components show low variance and prior information on effective augmentations is scarce. In contrast, prefer joint-embedding when these irrelevant components have high variance magnitude, as is common with real-world data, or when effective augmentations are either readily available or can be found through cross-validation.
This paper offers a basis for future work. One could consider extending these results to finite sample size settings to precisely characterize the interplay between sample complexity and augmentation in SSL.
this section cite: ['b39', 'b1', 'b71', 'b5', 'b44', 'b67']

Section: References
Ref_id:b0 Title: Self-supervised learning from images with a joint-embedding predictive architecture Year: (2023)
Ref_id:b1 Title: Big self-supervised models advance medical image classification Year: (2021)
Ref_id:b2 Title: Learning theory from first principles Year: (2024)
Ref_id:b3 Title: A cookbook of self-supervised learning Year: (2023)
Ref_id:b4 Title: Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods Year: (2022)
Ref_id:b5 Title: How learning by reconstruction produces uninformative features for perception Year: (2024-07)
Ref_id:b6 Title: A data-augmentation is worth a thousand samples: Analytical moments and sampling-free training Year: (2022)
Ref_id:b7 Title: End-to-end optimization of nonlinear transform codes for perceptual quality Year: (2016)
Ref_id:b8 Title: V-jepa: Latent video prediction for visual representation learning Year: (2023)
Ref_id:b9 Title: Vicreg: Variance-invariance-covariance regularization for self-supervised learning Year: (2021)
Ref_id:b10 Title: Training with noise is equivalent to tikhonov regularization Year: (1995)
Ref_id:b11 Title: The ssl interplay: Augmentations, inductive bias, and generalization Year: (2023)
Ref_id:b12 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b13 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b14 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b15 Title: Deep learning for classical japanese literature Year: (2018)
Ref_id:b16 Title: Moin Nabi, Nicu Sebe, and Elisa Ricci. solo-learn: A library of self-supervised methods for visual representation learning Year: (2022)
Ref_id:b17 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b18 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b19 Title: Unbiased single-cell morphology with self-supervised vision transformers Year: (2023)
Ref_id:b20 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b21 Title: Improving self-supervised learning by characterizing idealized representations Year: (2022)
Ref_id:b22 Title: The approximation of one matrix by another of lower rank Year: (1936)
Ref_id:b23 Title: Rethinking supervised pre-training for better downstream transferring Year: (2022)
Ref_id:b24 Title: On the duality between contrastive and non-contrastive self-supervised learning Year: (2022)
Ref_id:b25 Title: Bootstrap your own latent-a new approach to self-supervised learning Year: (2020)
Ref_id:b26 Title: Two convolutional neural networks for bird detection in audio signals Year: (2017)
Ref_id:b27 Title: A survey on self-supervised learning: Algorithms, applications, and future trends Year: (2024)
Ref_id:b28 Title: Provable guarantees for self-supervised deep learning with spectral contrastive loss Year: (2021)
Ref_id:b29 Title: How much semantic information is available in large language model tokens? Year: (2025)
Ref_id:b30 Title: Masked autoencoders are scalable vision learners Year: (2021)
Ref_id:b31 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b32 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b33 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b34 Title: Benchmarking neural network robustness to common corruptions and perturbations Year: (2019)
Ref_id:b35 Title: Towards the generalization of contrastive self-supervised learning Year: (2021)
Ref_id:b36 Title: A broad study on the transferability of visual representations with contrastive learning Year: ()
Ref_id:b37 Title: The common stability mechanism behind most self-supervised learning approaches Year: (2024)
Ref_id:b38 Title: Understanding dimensional collapse in contrastive self-supervised learning Year: (2021)
Ref_id:b39 Title: How does selfsupervised pretraining improve robustness against noisy labels across various medical image classification datasets Year: (2024)
Ref_id:b40 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b41 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b42 Title: Assessing the performance of the dinov2 self-supervised learning vision transformer model for the segmentation of the left atrium from mri images Year: (2024)
Ref_id:b43 Title: Wide neural networks of any depth evolve as linear models under gradient descent Year: (2019)
Ref_id:b44 Title: Supmae: Supervised masked autoencoders are efficient vision learners Year: (2022)
Ref_id:b45 Title: The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective Year: (2024)
Ref_id:b46 Title: How jepa avoids noisy features: The implicit bias of deep linear self distillation networks Year: (2024)
Ref_id:b47 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b48 Title: Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets Year: (2015)
Ref_id:b49 Title: Self-supervised learning of pretext-invariant representations Year: (2020)
Ref_id:b50 Title: Augmentations vs algorithms: What works in self-supervised learning Year: (2024)
Ref_id:b51 Title: Deep representation learning: Fundamentals, technologies, applications, and open challenges Year: (2023)
Ref_id:b52 Title: An empirically grounded identifiability theory will accelerate self-supervised learning research Year: (2025)
Ref_id:b53 Title: On the stepwise nature of self-supervised learning Year: (2023)
Ref_id:b54 Title:  Year: ()
Ref_id:b55 Title: What makes for good views for contrastive learning? Year: (2020)
Ref_id:b56 Title: Understanding self-supervised learning dynamics without contrastive pairs Year: (2021)
Ref_id:b57 Title:  Year: ()
Ref_id:b58 Title: Extracting and composing robust features with denoising autoencoders Year: (2008)
Ref_id:b59 Title: Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion Year: (2010)
Ref_id:b60 Title: Panopticon: Advancing any-sensor foundation models for earth observation Year: (2025)
Ref_id:b61 Title: Word form matters: Llms' semantic reconstruction under typoglycemia Year: (2025)
Ref_id:b62 Title: Understanding the behaviour of contrastive loss Year: (2021)
Ref_id:b63 Title: Understanding contrastive representation learning through alignment and uniformity on the hypersphere Year: (2020)
Ref_id:b64 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b65 Title: The mechanism of prediction head in non-contrastive self-supervised learning Year: (2022)
Ref_id:b66 Title: Ai-powered virtual tissues from spatial proteomics for clinical diagnostics and biomedical discovery Year: (2025)
Ref_id:b67 Title: Composed fine-tuning: Freezing pretrained denoising autoencoders for improved generalization Year: (2021)
Ref_id:b68 Title: Large batch training of convolutional networks Year: (2017)
Ref_id:b69 Title: Barlow twins: Self-supervised learning via redundancy reduction Year: (2021)
Ref_id:b70 Title: Understanding hard negatives in noise contrastive estimation Year: (2021)
Ref_id:b71 Title: Learning from models beyond fine-tuning Year: (2025)
Ref_id:b72 Title: Virchow2: Scaling self-supervised mixed magnification models in pathology Year: (2024)
