Title: DIFFICULT EXAMPLES HURT UNSUPERVISED CON-TRASTIVE LEARNING: A THEORETICAL PERSPECTIVE
Abstract: Unsupervised contrastive learning has shown significant performance improvements in recent years, often approaching or even rivaling supervised learning in various tasks. However, its learning mechanism is fundamentally different from supervised learning. Previous works have shown that difficult examples (wellrecognized in supervised learning as examples around the decision boundary), which are essential in supervised learning, contribute minimally in unsupervised settings. In this paper, perhaps surprisingly, we find that the direct removal of difficult examples, although reduces the sample size, can boost the downstream classification performance of contrastive learning. To uncover the reasons behind this, we develop a theoretical framework modeling the similarity between different pairs of samples. Guided by this framework, we conduct a thorough theoretical analysis revealing that the presence of difficult examples negatively affects the generalization of contrastive learning. Furthermore, we demonstrate that the removal of these examples, and techniques such as margin tuning and temperature scaling can enhance its generalization bounds, thereby improving performance. Empirically, we propose a simple and efficient mechanism for selecting difficult examples and validate the effectiveness of the aforementioned methods, which substantiates the reliability of our proposed theoretical framework.

Section: 
Contrastive learning has demonstrated exceptional empirical performance in the realm of unsupervised representation learning, effectively learning high-quality representations of high-dimensional data using substantial volumes of unlabeled data by aligning an anchor point with its augmented views in the embedding space (Caron et al., 2020;Chen et al., 2020b;c;2021;He et al., 2020). Unsupervised contrastive learning may own quite different working mechanisms from supervised learning, as discussed in Joshi & Mirzasoleiman (2023). For example, difficult examples (also known as difficult-to-learn examples in Joshi & Mirzasoleiman (2023)), which contribute the most to supervised learning, contribute the least or even negatively to contrastive learning performance. They show that on image datasets such as CIFAR-100 and STL-10, excluding 20%-40% of the examples does not negatively impact downstream task performance. More surprisingly, their results showed, but somehow failed to notice, that excluding these samples on certain datasets like STL-10 can lead to performance improvements in downstream tasks.
Taking a step further beyond their study, we find that this surprising result is not just a specialty of a certain dataset, but a universal phenomenon across multiple datasets. Specifically, we run SimCLR on the original CIFAR-10, CIFAR-100, STL-10, and TinyImagenet datasets, the SAS core subsets self-supervised pre-training. It is somewhat related to hard negative samples, a pure unsupervised learning concept defined as highly similar negative samples to the anchor point, but is different in nature. (See Appendix A.1 for more discussions.) In real datasets, as difficult examples rely on the specific classifier trained in the supervised learning manner, we can not precisely know the ground truth difficult examples. Therefore, we in turn add additional difficult examples and observe the effects of these examples. Specifically, we generate a mixing-image dataset containing more difficult samples by mixing a ω fraction of images on CIFAR-10 dataset at the pixel level (these samples lie around the class difficult), termed as ω-Mixed CIFAR-10 datasets. Then, we train the representative contrastive learning algorithm SimCLR (Chen et al., 2020b) on the original, 10%-, and 20%-Mixed CIFAR-10 datasets using ResNet18 model. We report the linear probing accuracy in Figure 2.
Compared with the model trained on the original dataset, we find that with the mixed difficult examples included in the training dataset, the performance of contrastive learning drops. This result indicates that the (mixed) difficult samples significantly negatively impact contrastive learning. As the mixing ratio ω increases, the performance drops, indicating that more difficult examples lead to worse contrastive learning performances.
Moreover, we show that removing the (mixed) difficult samples can boost performance. Specifically, we compare performance on the Mixed CIFAR-10 datasets with that on the datasets removing the mixed examples. As shown in Figure 2, despite being trained with a smaller sample size, models trained on datasets removing the mixed examples perform better than the ones trained with the mixed examples, which further verifies that difficult examples hurt unsupervised contrastive learning, and removal of these difficult examples can boost learning performance.
this section cite: ['b5', 'b34', 'b19', 'b23', 'b23']

Section: THEORETICAL CHARACTERIZATION OF WHY DIFFICULT EXAMPLES HURT CONTRASTIVE LEARNING
In this section, to explain why difficult examples negatively impact the performance of contrastive learning, we provide theoretical evidence on generalization bounds. In Section 3.1 we present the necessary preliminaries that lay the foundation for our theoretical analysis. In Section 3.2, we introduce the similarity graph describing difficult examples. In Section 3.3, we respectively derive error bounds of contrastive learning with and without difficult examples.
this section cite: []

Section: PRELIMINARIES
Notations. Given a natural data x ∈ X := R d , we denote the distribution of its augmentations by A(•|x) and the set of all augmented data by X , which is assumed to be finite but exponentially large. For mathematical simplicity, we assume class-balanced data with n denoting the number of augmented samples per class and r + 1 denoting the number of classes, hence |X | = n(r + 1).
Let n d represent the number of difficult examples per class and D d the set of difficult examples. In addition, we denote k as the feature dimension in contrastive representation learning.
Similarity Graph (Augmentation Graph). As described in HaoChen et al. (2021), an augmentation graph G represents the distribution of augmented samples, where the edge weight w xx ′ signifies the joint probability of generating augmented views x and x ′ from the same natural data, i.e., w xx ′ := E x∼ P [A(x|x)A(x ′ |x)], where P denotes the distribution of natural data. The total probability across all pairs of augmented data sums up to 1, i.e., x,x ′ ∈X w xx ′ = 1. The adjacency matrix of the augmentation graph is denoted as A = (w xx ′ ) x,x ′ ∈X , and the normalized adjacency matrix is Ā = D -1/2 AD -1/2 , where D := diag(w x ) x∈X , and w x := x ′ ∈X w xx ′ . The concept of augmentation graph is further extended to describe similarities beyond image augmentation, such as cross-domain images (Shen et al., 2022), multi-modal data (Zhang et al., 2023), and labeled examples (Cui et al., 2023).
Contrastive losses. For theoretical analysis, we consider the spectral contrastive loss L(f ) proposed by HaoChen et al. (2021) as a good performance proxy for the widely used InfoNCE loss
L Spec (f ) := -2 • E x,x + [f (x) ⊤ f (x + )] + E x,x ′ f (x) ⊤ f (x ′ ) 2 ,(1)
where x, x + , and x ′ represent the anchor, positive sample, and negative sample, respectively. As proved in Balestriero & LeCun (2022); Johnson et al. (2022); Tan et al. (2024), the spectral contrastive loss and the InfoNCE loss share the same population minimum with variant kernel derivations. Further, the spectral contrastive loss is theoretically shown to be equivalent to the matrix factorization loss. For F = (u x ) x∈X , where u x = w 1/2
x f (x), the matrix factorization loss is:
L mf (F ) := ∥ Ā -F F ⊤ ∥ 2 F = L Spec (f ) + const.
(2)
this section cite: ['b18', 'b35', 'b44', 'b12', 'b18', 'b3', 'b22', 'b37']

Section: MODELING OF DIFFICULT EXAMPLES
We start by introducing a similarity graph, to describe the relationships between various samples. In contrastive learning, examples are used in a pairwise manner, so we define difficult sample pairs as sample pairs that include at least one difficult sample. As difficult examples lie around the decision boundary, they should have higher augmentation similarity to examples from different classes. Therefore, it is natural for us to define the difficult pairs as different-class sample pairs with higher similarity. Correspondingly, easy pairs are defined as different-class sample pairs containing no difficult samples, or different-class sample pairs with lower similarity.
Specifically, we define the augmentation similarity between a sample and itself as 1. Then we assume the similarity between same-class samples is α (Figure 3  In addition, the above modeling could be relaxed by adding random terms to the similarity values. Specifically, for some constant ϵ > 0, for a similarity matrix A = (ã ij ), we replace a ij with ãij = a ij + ϵ • ε ij for i ̸ = j, where a ij takes values in {α, β, γ}, ε ij = ε ji are i.i.d. random variables with mean 0 and variance 1. We discuss the relaxation in detail in Section B.3.
In what follows, our theoretical analysis is based on the generalized similarity graph containing |X | = n(r + 1) samples. The formal definition of the generalized adjacency matrix is in Appendix B.
this section cite: []

Section: ERROR BOUNDS WITH AND WITHOUT DIFFICULT EXAMPLES
Based on the similarity graph in Section 3.2, we derive the linear probing error bounds for contrastive learning models trained with and without difficult examples in Theorems 3.3 and 3.4. We mention that we adopt the label recoverability (with labeling error δ) and realizability assumptions from HaoChen et al. (2021). Assumption 3.1 (Labels are recoverable from augmentations). Let x ∼ P X and y(x) be its label.
this section cite: ['b18']

Section: Let the augmentation x ∼ A(•|x).
We assume that there exists a classifier g that can predict y(x) given x with error at most δ, i.e. g(x) = y(x) with probability at least 1 -δ. Assumption 3.2 (Realizability). Let F be a hypothesis class containing functions from X to R k . We assume that at least one of the global minima of L Spec belongs to F.
Assumption 3.1 indicates that labels are recoverable from the augmentations, and Assumption 3.2 indicates that the universal minimizer of the population spectral contrastive loss can be realized by the hypothesis class. The proofs are shown in Appendix B.1. Theorem 3.3 (Error Bound without Difficult Examples). Denote E w.o. as the linear probing error of a contrastive learning model trained on a dataset without difficult examples. Then E w.o. ≤ 4δ 1 -1-α (1-α)+nα+nrβ + 8δ. (3) Theorem 3.4 (Error Bound with Difficult Examples). Denote E w.d. as the linear probing error of a contrastive learning model trained on a dataset with n d difficult examples per class. Then
if r + 1 ≤ k < n d + r + 1, there holds E w.d. ≤ 4δ 1 - (1-α)+r(γ-β) (1-α)+nα+nrβ+n d r(γ-β) + 8δ.(4)
this section cite: []

Section: Discussions.
By comparing Theorems 3.3 and 3.4, also considering that
(1-α)+r(γ-β) (1-α)+nα+nrβ+n d r(γ-β) > 1-α
(1-α)+nα+nrβ , we see the presence of difficult examples leads to a strictly worse linear probing error bound for a contrastive learning model. Moreover, more challenging difficult examples (larger γ -β) result in worse error bounds. When γ = β, i.e. no difficult examples exist, the bound in Theorem 3.4 reduces to that in Theorem 3.3. Intuitively, through the augmentation graph, contrastive learning could be understood as a spectral clustering problem (HaoChen et al., 2021). As the difficult examples lie very close to the classification boundary, they could fall into the wrong clusters during self-supervised pre-training. In the downstream applications, the wrongly clustered examples provide false prior knowledge to the downstream classification, which harms the performance of all test samples.
this section cite: ['b18']

Section: THEORETICAL CHARACTERIZATION ON ELIMINATING EFFECTS OF DIFFICULT EXAMPLES
Building on the above unified theoretical framework, we theoretically analyze that directly removing difficult samples (Section 4.1), margin tuning (Section 4.2), and temperature scaling (Section 4.3) can handle difficult examples by improving the generalization bounds in different ways.
this section cite: []

Section: REMOVING DIFFICULT SAMPLES
In Figures 1 and 2, empirical experiments demonstrated that removing difficult samples can improve learning performance. Corollary 4.1 provides a theoretical explanation for this counter-intuitive phenomenon based on our established framework. Corollary 4.1. Denote E R as the linear probing error of a contrastive learning model trained on a selected subset removing all difficult examples D d . Then there holds
E R ≤ 4δ 1 - 1-α (1-α)+(n-n d )α+(n-n d )rβ + 8δ.(5)
Corollary 4.1 shows that when the difficult examples are removed, the linear probing error bound has the same form as the case where no difficult examples are present (Theorem 3.3), but with n replaced by n -n d . Compared with the case without removing difficult examples (Theorem 3.4), the bound in equation 5 is smaller than that in equation 4 when γ -β >
n d (1-α)(α+rγ) r[(1-α)+(n-n d )(α+rβ)]
. This indicates that removing difficult examples enhances the error bound when these samples are significantly harder than the easy ones (i.e., large γ -β) or when the number of difficult samples is small (i.e., small n d ).
this section cite: []

Section: MARGIN TUNING
Aside from sample removal, we also consider using the margin tuning technique to deal with difficult examples. Specifically, we add additional margin parameters to the similarity of difficult pairs in the loss function (see Eq. 14). Here, we delve into how margin tuning can enhance the generalization in the presence of difficult examples. Theorem 4.2. The margin tuning loss is equivalent to the matrix factorization loss
L mf-M (F ) := ∥( Ā -M ) -F F ⊤ ∥ 2 F , (6
)
where Ā is the normalized adjacency matrix, and M is the normalized margin matrix.
Theorem 4.2 indicates that adjusting margins alters the similarity graph by subtracting a normalized margin matrix M from the normalized similarity matrix Ā. Intuitively, by subtracting the additional similarity values of difficult examples with appropriately chosen margins, the remaining values will match those of easy examples. Specifically, in the following Theorem 4.3, we show that properly chosen margins can eliminate the negative impact of difficult examples. Theorem 4.3. Denote E M as the linear probing error for the margin tuning loss equation 31 trained on a dataset with difficult samples D d . If we let
m x,x ′ = c 0 /(c 2 1 c 2 ) • (γ -β) (7
)
for y(x) ̸ = y(x ′ ), x, x ′ ∈ D d , where c 0 := (1 -α) + nα + (n -n d )rβ, c 1 := (1 -α) + nα + nrβ + n d r(γ -β) and c 2 := (1 -α) + nα + nrβ, and m x,x ′ = 0 for x, x ′ / ∈ D d , then we have
E M = E w.o. . (8
)
Note that when n is large enough, m x,x ′ for x or x ′ / ∈ D d are higher-order infinitesimals relative to equation 7, and primarily affect normalization rather than the core problem. Thus, we focus on cases where x, x ′ ∈ D d and defer specific forms of other m x,x ′ values to the proofs for brevity.
Theorem 4.3 shows that with appropriately chosen margins, the linear probing error bound for the margin tuning loss in the presence of difficult examples becomes equivalent to the standard contrastive loss without such examples, as indicated in Theorem 3.3. Since equation 7 > 0, this suggests applying a positive margin to the difficult example pairs. Additionally, the more challenging the example pairs are (i.e., the larger γ -β), the greater the margin value should be.
this section cite: []

Section: TEMPERATURE SCALING
We also consider the widely used temperature scaling technique in eliminating the negative effects of difficult examples. Specifically, we add an additional temperature scaling parameter to the base temperature of difficult pairs in the loss function and assign the base temperature to all the other pairs (see Eq. 15). Here, we investigate how temperature scaling can enhance generalization. Theorem 4.4. The temperature scaling loss is equivalent to the matrix factorization loss
L mf-T (F ) := ∥T ⊙ Ā -F F ⊤ ∥ 2 wF , (9
)
where Ā is the normalized adjacency matrix of similarity graph, T ⊙ Ā is the element-wise product of matrices T and Ā, and ∥ • ∥ wF is the weighted Frobenius norm (specified in the proof).
Theorem 4.4 shows that adjusting temperatures modifies the similarity graph by multiplying the temperature values with the normalized similarity matrix Ā. Intuitively, by scaling the similarity values between difficult examples, we can match these values to those of easy examples, thereby mitigating the negative effects of difficult examples. Specifically, the following Theorem 4.5 outlines the appropriate temperature values to be chosen.
Theorem 4.5. Denote E T as the linear probing error for the temperature scaling loss equation 40 trained on a dataset with difficult samples D d . If we let
τ x,x ′ = (c 1 /c 2 )(β/γ)(10)
for y(x) ̸ = y(x ′ ), x, x ′ ∈ D d , where c 1 := (1 -α) + nα + nrβ + n d r(γ -β) and c 2 := (1 -α) + nα + nrβ, and τ x,x ′ = 1 for x, x ′ / ∈ D d , then we have
E T ≤ 4[1 -(n d /n) 2 + (γ/β) 2 (n d /n) 2 ]δ 1 - 1-α (1-α)+nα+nrβ + 8δ.(11)
Likewise, here we only focus on the temperature values between difficult examples, and defer the specific forms of other τ x,x ′ values to the proofs for brevity.
Theorem 4.5 shows the linear probing error bound of the temperature scaling loss when trained on data containing difficult examples. Specifically, with large n and
n d /n → 0, we have E T /E w.o. - 1 ≈ O((n d /n) 2 ) and E w.d. /E w.o. -1 ≈ O(1/n). This indicates that, when O(n d ) ≲ O(n 1/2 ), E T /E w.o. ≲ E w.d. /E w.o. , meaning E T converges faster to E w.o. . Detailed calculations show that when n d < r (α+rβ)(γ+β) β • n 1/2 , there holds E T < E w.d.
, which means that temperature scaling improves the error bound. Note that we have approximately τ x,x ′ ∝ β/γ. This inspires us to choose smaller temperature values for the difficult example pairs. The more difficult the example pairs (smaller β/γ), the smaller the temperature values that should be chosen.
this section cite: []

Section: VERIFICATION EXPERIMENTS
This paper primarily focuses on theoretical analysis, explaining how different samples in contrastive learning impact generalization. The experiments in this part are mainly designed to validate the theoretical insights and demonstrate that the proposed directions for improving performance are sound. The experiments are not intended to achieve state-of-the-art results but rather to confirm the correctness of our theoretical findings. We hope that readers will appreciate the theoretical contributions of this work and not focus excessively on the experimental results.
In Section 5.1, we present an efficient mechanism for selecting difficult samples. We then evaluate the removal of difficult samples (Section 5.2), margin tuning (Section 5.3), and temperature scaling (Section 5.4), all of which are theoretically established to mitigate the impact of these difficult examples. In Section 5.5, we propose a combined method, and discuss the scalability under different paradigms and the connection between difficult samples and long-tail distribution. The specific loss forms can be found in Appendix A.2.
this section cite: []

Section: DIFFICULT EXAMPLES SELECTION
In this section, we design a simple yet efficient selection mechanism to validate our theoretical analysis, without relying on additional pretrained models or incurring extra computational overhead (Joshi & Mirzasoleiman, 2023).
To identify difficult sample pairs which from different classes but with high similarity, we compute the cosine similarity of each sample to other samples in the same batch using features before projector mapping. We define posHigh and posLow as percentiles of the similarity sorted in descending order, where Sim posHigh and Sim posLow are the corresponding similarities. Generally, following the characterization in Section 3.2 and Appendix B, we can roughly assume posHigh corresponds to 1/(r + 1), where r + 1 is the class number 1 . Sample pairs with cosine similarities above Sim posHigh are considered from the same class. Sample pairs with the similarity between Sim posHigh and Sim posLow are considered as difficult examples. Sample pairs with cosine similarities below Sim posLow are considered as easy-to-learn samples from different classes. Here for posLow, we note that when optimizing γ of difficult examples, if some easy-to-learn samples are involved, the process will also optimize β, which is a good thing for the representation learning to Published as a conference paper at ICLR 2026 push samples from different classes further apart. Therefore, we can easily find a value close to the bottom of the sorted similarity for posLow, even 100%. Experiments in Figure 4(a) and Figure 4(b) show that our method is not sensitive to the exact values of posHigh and posLow. Using this selection mechanism, for an augmented sample pair (x i , x j ) in the current batch, we define the selecting indicator of difficult pairs as
p i,j := 1 [Sim posLow ≤sij <Sim posHigh ] , (12
)
where s i,j denotes the cosine similarity between the representations of x i and x j , and 1 [condition] denotes the indicator function returning 1 if the condition holds and 0 otherwise. For each sample x i , we get a vector P i = (p i,j ) 2N j=1 representing the indicator of difficult pairs. After calculating these indicators for all samples in the current batch, we stack the vectors P i row-wise to create the selection matrix P . In practice, P i can be computed in parallel, making the computation of P efficient. The elements of P are either 0 or 1, indicating whether pairs are difficult pairs or not.
We can use the class information to verify the proportions of sample pairs from different classes in (Sim posLow , Sim posHigh ) on CIFAR-10, which can demonstrate the effectiveness of our selection mechanism. As shown in Figure 4(c), along with the progress of training, the ratio of sample pairs from different classes approaches close to 100% within the range (Sim posLow , Sim posHigh ).
this section cite: ['b23']

Section: REMOVING DIFFICULT SAMPLES
We here introduce a simple and practical method for removing difficult samples based on our proposed selection mechanism. Eliminating the impact of difficult samples means preventing sample pairs that include difficult samples from interfering with the training process. To achieve this, we use the selection matrix P to identify and remove difficult samples. It can be observed from Table 1 that removing difficult examples yields a 0.8% performance boost on CIFAR-10, a 0.6% performance boost on CIFAR-100, and a 3.7% performance boost on TinyImagenet compared to the baseline method. We reach the same conclusion as in Joshi & Mirzasoleiman (2023): By removing difficult samples, we can achieve comparable results or even slight improvements over the baseline. However, removing difficult samples may not be the most effective method for handling difficult samples, because it shrinks sample size. Next, we investigate two techniques that handle difficult samples better, margin tuning in Section 5.3 and temperature scaling in Section 5.4.
this section cite: ['b23']

Section: MARGIN TUNING ON DIFFICULT SAMPLES
To effectively apply margin tuning in line with our theoretical analysis, we adopt a margin tuning factor σ > 0. For the selected difficult sample pairs identified by the selection matrix P , we add a margin σ to the similarity values, and for the unselected pairs, we use the original InfoNCE. It can be observed from Table 2 that applying margin tuning to all samples directly only achieves comparable results as the baseline SimCLR, highlighting the importance of the selection mechanism for difficult examples. While applying margin tuning to the selected samples brings consistent performance gains on CIFAR10, CIFAR100, and TinyImageNet. These results validate both the effectiveness of our selection mechanism and the reliability of our analysis on margin tuning.
this section cite: []

Section: TEMPERATURE SCALING ON DIFFICULT SAMPLES
We define the temperature scaling factor ρ > 0. Given the base temperature τ > 0, we attach temperature ρτ to the selected difficult sample pairs identified by the selection matrix P , whereas attach base temperature τ to the unselected pairs.
Table 3: Classification accuracy with or without temperature scaling on CIFAR-10, CIFAR-100, STL-10 and TinyImagenet dataset. Results are averaged over three runs. Method CIFAR-10 CIFAR-100 STL-10 TinyImagenet Baseline 88.26 59.95 75.98 69.58 TS (All Samples) 88.38 59.20 75.76 69.36 TS (Selected Samples) 89.24 61.67 76.62 78.52
It can be observed from Table 3 that applying temperature scaling to all samples directly can even hurt the performance of contrastive learning compared to baseline SimCLR, highlighting the importance of selecting difficult examples. In contrast, applying temperature scaling to the selected samples brings consistent performance gains on CIFAR10, CIFAR100, and TinyImageNet. These results validate both the effectiveness of our selection mechanism and the reliability of our analysis on temperature scaling.
this section cite: []

Section: EXTENSIONS
Combined method. From Sections 4.2 and 4.3, we observe that margin tuning and temperature scaling eliminate the effects of difficult examples in different ways. Therefore, it is natural to combine the two methods, and see if the combined method could reach better performances. It can be observed from Table 4 that the combined method yields a 1.6% performance improvement on CIFAR-10, a 4.9% performance improvement on CIFAR-100 and a 15.0% performance improvement on TinyImagenet compared to the baseline SimCLR. The improvement surpasses that achieved by using only margin tuning or temperature scaling. The combined method on the Mixed CIFAR-10 datasets also achieves performance improvements consistently as shown in Section A.5.
The complete algorithm is presented in Algorithm 1.
Alternative contrastive learning paradigm. We delve deeper into the scalability of our methods across various self-supervised learning paradigms. Results in Table 5 demonstrate consistent performance enhancements comparable to those achieved by SimCLR on the MoCo on CIFAR-10.
this section cite: []

Section: Complex classification scenarios.
We explore our method by targeting difficult samples under the long-tail classification scenario, where difficult samples are even more difficult to learn according to the imbalanced distributions. The findings in Table 6 illustrate that our approach outperforms the baseline SimCLR in scenarios involving distributional imbalance, indicating the adaptivity of our approach to complex classification scenarios. Further discussions. We also provide a sensitivity analysis of parameters in Section A.4 and conduct a detailed analysis of results in Table 5 and Table 6 in Section A.5. Furthermore, discussions about which features are advantageous for selecting difficult examples are also presented in Section A.5. In Section A.5, we have also included the experimental results on ImageNet-1K, the trending of the derived bounds with Mixed CIFAR-10 dataset and the significance analysis of γ and β.
this section cite: []

Section: CONCLUSION
In this paper, we construct a theoretical framework to specifically analyze the impact of difficult examples on contrastive learning. We prove that difficult examples hurt the performance of contrastive learning from the perspective of linear probing error bounds. We further demonstrate how techniques such as margin tuning, temperature scaling, and the removal of these examples from the dataset can improve performance from the perspective of enhancing the generalization bounds. The experimental results demonstrate the reliability of our theoretical analysis.
this section cite: []

Section: References
Ref_id:b0 Title: Sequence-to-sequence contrastive learning for text recognition Year: (2021)
Ref_id:b1 Title: A theoretical analysis of contrastive unsupervised representation learning Year: (2019)
Ref_id:b2 Title: Investigating the role of negatives in contrastive representation learning Year: (2022)
Ref_id:b3 Title: Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods Year: (2022)
Ref_id:b4 Title: On the surrogate gap between contrastive and supervised losses Year: (2022)
Ref_id:b5 Title: Unsupervised learning of visual features by contrasting cluster assignments Year: (2020)
Ref_id:b6 Title: Measuring and relieving the oversmoothing problem for graph neural networks from the topological view Year: ()
Ref_id:b7 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b8 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b9 Title: Improved baselines with momentum contrastive learning Year: (2020)
Ref_id:b10 Title: An empirical study of training self-supervised vision transformers Year: (2021)
Ref_id:b11 Title: Toward a geometrical understanding of selfsupervised contrastive learning Year: (2022)
Ref_id:b12 Title: Rethinking weak supervision in helping contrastive learning Year: (2023)
Ref_id:b13 Title: An augmentation-aware theory for self-supervised contrastive learning Year: (2025)
Ref_id:b14 Title: Class-balanced loss based on effective number of samples Year: (2019)
Ref_id:b15 Title: Exploring deep neural networks via layerpeeled model: Minority collapse in imbalanced training Year: (2021)
Ref_id:b16 Title: Eigenvalues, invariant factors, highest weights, and schubert calculus Year: (2000)
Ref_id:b17 Title: Bootstrap your own latent-a new approach to self-supervised learning Year: (2020)
Ref_id:b18 Title: Provable guarantees for self-supervised deep learning with spectral contrastive loss Year: (2021)
Ref_id:b19 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b20 Title: Your contrastive learning is secretly doing stochastic neighbor embedding Year: (2023)
Ref_id:b21 Title: Towards the generalization of contrastive self-supervised learning Year: (2023)
Ref_id:b22 Title: Contrastive learning can find an optimal basis for approximately view-invariant functions Year: (2022)
Ref_id:b23 Title: Data-efficient contrastive self-supervised learning: Most beneficial examples for supervised learning contribute the least Year: (2023)
Ref_id:b24 Title: Hard negative mixing for contrastive learning Year: (2020)
Ref_id:b25 Title: Contrastive self-supervised learning for sensor-based human activity recognition Year: (2021)
Ref_id:b26 Title: Dynamic temperature scaling in contrastive self-supervised learning for sensor-based human activity recognition Year: (2022)
Ref_id:b27 Title: Revisiting contrastive learning through the lens of neighborhood component analysis: an integrated framework Year: (2022)
Ref_id:b28 Title: Temperature schedules for self-supervised contrastive methods on long-tail data Year: (2023)
Ref_id:b29 Title: Learning to discriminate information for online action detection: Analysis and application Year: (2022)
Ref_id:b30 Title: Understanding negative samples in instance discriminative selfsupervised representation learning Year: (2019)
Ref_id:b31 Title: Projection head is secretly an information bottleneck Year: (2025)
Ref_id:b32 Title: Contrastive learning with hard negative samples Year: (2020)
Ref_id:b33 Title: Distributional robustness loss for long-tail learning Year: (2021)
Ref_id:b34 Title: Low: Training deep neural networks by learning optimal sample weights Year: (2021)
Ref_id:b35 Title: Connect, not collapse: Explaining contrastive learning for unsupervised domain adaptation Year: (2022)
Ref_id:b36 Title: Which strategies matter for noisy label classification? insight into loss and uncertainty Year: (2020)
Ref_id:b37 Title: Contrastive learning is spectral clustering on similarity graph Year: (2024)
Ref_id:b38 Title: Residual relaxation for multi-view representation learning Year: ()
Ref_id:b39 Title: Chaos is a ladder: A new theoretical understanding of contrastive learning via augmentation overlap Year: ()
Ref_id:b40 Title: A message passing perspective on learning dynamics of contrastive learning Year: (2023)
Ref_id:b41 Title: Non-negative contrastive learning Year: (2024)
Ref_id:b42 Title: Barlow twins: Self-supervised learning via redundancy reduction Year: (2021)
Ref_id:b43 Title: Temperature as uncertainty in contrastive learning Year: (2021)
Ref_id:b44 Title: On the generalization of multi-modal contrastive learning Year: (2023)
Ref_id:b45 Title: An augmentation overlap theory of contrastive learning Year: (2025)
Ref_id:b46 Title: Zero-mean regularized spectral contrastive learning Year: (2024)
Ref_id:b47 Title: Towards a unified theoretical understanding of non-contrastive learning via rank differential mechanism Year: (2023)
Ref_id:b48 Title: Contrastive learning inverts the data generating process Year: (2021)
