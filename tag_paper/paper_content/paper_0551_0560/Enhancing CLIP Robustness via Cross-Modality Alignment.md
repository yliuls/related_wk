Title: Enhancing CLIP Robustness via Cross-Modality Alignment
Abstract: Vision-language models (VLMs) such as CLIP demonstrate strong generalization in zero-shot classification but remain highly vulnerable to adversarial perturbations. Existing methods primarily focus on adversarial fine-tuning or prompt optimization, they often overlook the gaps in CLIP's encoded features, which is shown as the text and image features lie far apart from each other. This misalignment is significantly amplified under adversarial perturbations, leading to severe degradation in classification performance. To address this problem, we propose CrOss-modaLity Alignment, dubbed COLA, an optimal transport-based framework that explicitly addresses adversarial misalignment by restoring both global image-text alignment and local structural consistency in the feature space. (1) COLA first projects adversarial image embeddings onto a subspace spanned by class text features, effectively filtering out non-semantic distortions while preserving discriminative information. (2) It then models images and texts as discrete distributions over multiple augmented views and refines their alignment via OT, with the subspace projection seamlessly integrated into the cost computation. This design ensures stable cross-modal alignment even under adversarial conditions. COLA is trainingfree and compatible with existing fine-tuned models. Extensive evaluations across 14 zero-shot classification benchmarks demonstrate the effectiveness of COLA, especially with an average improvement of 6.7% on ImageNet and its variants under PGD adversarial attacks, while maintaining high accuracy on clean samples.Recent efforts to improve the adversarial robustness of VLMs can be broadly categorized into three directions: adversarial training [4, 62], which fine-tunes models with perturbed samples; prompt tuning [31,63], which optimizes text input templates to resist attacks, and test-time defenses [21,2,59], which modify inputs or predictions on the fly. While these methods offer promising improvements, they suffer from high computational overhead or introduce substantial inference latency. More critically, they overlook a central issue: the misalignment between image and text modalities [70,16]. This misalignment stems from CLIP's global matching paradigm, where the model is trained to align entire image embeddings with sentence-level textual embeddings. As shown in Figure 1(a), the text † Corresponding author 39th Conference on Neural Information Processing Systems (NeurIPS 2025).2 Related Work Adversarial robustness in VLMs. Adversarial robustness remains a fundamental challenge, as small, imperceptible perturbations can mislead model predictions [51,7]. A common defense is adversarial training (AT) [33,62,47], which improves robustness but incurs high computational cost [50,56]. Recent test-time defenses-such as generative purification [37,61] and optimization-based methods [57, 35]-offer alternatives but often fail under adaptive attacks [12]. Hedge Defense (HD) [57], for example, perturbs inputs by maximizing cross-entropy loss, but requires an adversarially trained model. Meanwhile, several works explore CLIP's [45] robustness, noting its natural tendency to deflect attacks via counteractive perturbations in latent space. To further improve performance, researchers have applied adversarial fine-tuning [36,54] and prompt tuning with frozen weights [31,63].

Section: Introduction
Vision-language models (VLMs) [17,30] like CLIP [45] demonstrate strong generalization ability in zero-shot classification. However, they are highly susceptible to adversarial perturbations, where small but carefully crafted changes to input images can significantly mislead predictions [31,32,36]. Such vulnerabilities pose serious risks in critical applications such as medical diagnosis, autonomous driving, and security systems, where robustness and reliability are paramount.
A golden dog is running on the beach. "A golden dog running on the beach" offers a fine-grained description that includes the object, its attributes, and the surrounding background. However, the image encoder processes the entire scene as a global representation, without explicitly modeling how these textual elements correspond to specific regions of the image. This results in image and text features being distributed independently in separate regions of the embedding space.
Such misalignment becomes particularly problematic under adversarial attacks [15,31]. As illustrated in Figure 1(b), even small perturbations can distort the image embedding and severely disrupt global feature alignment, pushing visual representations away from their semantic prototypes. Beyond global shifts, attacks also damage the local structure within the feature space, causing nearby image embeddings to scatter and lose their internal consistency. As shown in Figure 1(d), this dual breakdown of alignment leads to a near-collapse in classification accuracy.
To address this issue, we propose a training-free framework that explicitly addresses adversarial misalignment at both the feature and semantic levels. First, we project adversarial image embeddings onto a text-induced subspace, eliminating non-semantic distortions and restoring feature space alignment. Then, we model images and texts as discrete distributions over multiple augmented views and refine their correspondence through optimal transport (OT) based on the projected features. Subspace projection is directly embedded into the OT cost, and we theoretically guarantee that it does not increase the transport distance. By jointly aligning feature embeddings and semantic distributions, our approach substantially improves the adversarial robustness of CLIP. Figure 1(d) illustrates the accuracy improvement over the attacked CLIP.
We conduct extensive experiments across 14 zero-shot classification benchmarks to evaluate the effectiveness of our method. Results demonstrate that COLA substantially improves adversarial robustness under multiple attack settings, with notable improvements such as an average gain of 6.7% under PGD and 4.8% under CW attacks on ImageNet and its variants, while maintaining high accuracy on clean images. Moreover, COLA can be directly applied to different CLIP fine-tuned models without any retraining, making it practical for real-world deployment.
These methods enhance robustness but rely on training. In contrast, our work proposes the first testtime defense for CLIP that is training-free, architecture-free, and efficient at inference. In contrast to prior efforts that rely on adversarial training, prompt tuning, or additional inference-time modules, we introduce a simple yet effective test-time defense for CLIP, which improves adversarial robustness by restoring image-text alignment through subspace projection and distribution-level matching, without requiring any model retraining or architectural changes.
this section cite: ['b16', 'b29', 'b44', 'b30', 'b31', 'b35', 'b14', 'b30']

Section: Optimal transport.
Optimal transport offers a principled way to compare probability distributions by capturing their geometric relationships [41]. With the development of efficient solvers such as the Sinkhorn algorithm [13], OT has been widely applied to tasks including generative modeling [3,43,44], domain adaptation [11], and structural alignment [9,60]. In the vision-language domain, OT has enabled fine-grained alignment of image-text distributions in few-shot learning [27,67], distribution calibration [22,68], and prompt learning [8,52]. Of particular relevance are recent OT-based methods for vision-language modeling [71], which improve zero-shot performance by enhancing alignment between visual and textual modalities. However, these approaches typically rely on training-time optimization or prompt tuning. While prior works focus on training-time alignment or require prompt tuning, our approach differs in that it introduces an efficient test-time OT framework for adversarially perturbed images. Specifically, we use OT to align projected image features with augmented textual prototypes, thereby enhancing robustness without any model fine-tuning.
this section cite: ['b40', 'b12', 'b2', 'b42', 'b43', 'b10', 'b8', 'b59', 'b26', 'b66', 'b21', 'b67', 'b7', 'b51', 'b70']

Section: Method
We propose a unified OT framework to enhance robust zero-shot classification by simultaneously addressing the modality misalignment caused by adversarial distortions and the mismatch between images and their text descriptions. We further provide theoretical guarantees that our method better preserves semantic similarity and yields larger margins, suggesting improved generalization.
this section cite: []

Section: Preliminaries
Zero-shot classification. CLIP [45] consists of a vision encoder Φ v (•) and a text encoder Φ t (•).
Given a set of K class names {z y } K y=1 and a hand-crafted template G, e.g., "a photo of a z y ", the textual feature for class y is computed as z y = Φ t (G(z y )). The visual feature for a testing samples x is calculated as x = Φ v (x), where both x and z y lie in the same d-dimensional embedding space (x, z y ∈ R d ). CLIP performs classification by comparing the similarity between the visual feature x and all text prototypes {z y } K y=1 :
y = argmax y∈[K] z ⊤ y x.(1)
Recent practices [42,29,48] replace the hand-crafted prompt G(z y ) with a set of fine-grained class descriptions {z m y } M m=1 = LLM(z y ) generated by large language models (LLMs). The corresponding text features {z m y } M m=1 for class j are obtained via z m y = Φ t (z m j ), and the average feature zy = 1 M M m=1 z m y is used in place of z y in Eq. ( 1) for classification.
this section cite: ['b44', 'b41', 'b28', 'b47']

Section: Adversarial perturbations.
When the attacker has full access of model parameters, it becomes vulnerable to adversarial perturbations δ a , which are typically generated via methods such as Projected Gradient Descent (PGD) [7]:
δ a = arg max δ L(x i + δ, y i ), s.t. ∥δ∥ p ≤ ϵ a (2
)
where y i is the ground-truth and L is a loss function, typically cross-entropy. δ a is constrained by an ℓ p -norm budget ϵ a , making it visually imperceptible yet highly effective at degrading accuracy.
this section cite: ['b6']

Section: Cross Modality Alignment under a Unified OT Framework
Original CLIP aligns clean images and texts into a unified feature space, but adversarial attacks on the visual modality severely disrupt this alignment. Moreover, visual features often capture background or irrelevant objects that are not reflected in LLM-generated descriptions, introducing further semantic misalignment. In this work, we propose a unified OT framework to mitigate both types of misalignment by introducing feature space alignment and local semantics alignment.
Global feature alignment. Despite the contamination in image features, the subspace spanned by clean textual features serves as a reliable proxy for reconstructing the underlying clean image representations, a design inspired by [65]. Specifically, we arrange all class text embeddings {z m y } y,m into a matrix Z ∈ R d×KM and apply singular value decomposition (SVD) to extract the top-C principal components:
Z = UΣV ⊤ , U C = U [:,1:C] .(3)
This defines a subspace U = span(U C ), which captures C dominant directions shared across class embeddings. Since adversarial perturbations distort image features along directions away from U, we project each perturbed image feature x onto U to achieve alignment:
Π(x) = U C U ⊤ C x.(4)
In Sec. 3.3, we show that the projection helps recover the pairwise similarity of clean image features.
Local structural alignment. While feature space alignment restores a unified space, projected image features can still misalign due to visual cues like background or irrelevant objects absent from LLM-generated text. To bridge this gap, we perform local semantics alignment for visual and textual representations. Specifically, for each adversarial image x, we generate N -1 augmented views via random cropping, flipping, or resizing, and include the original to form a set {x n } N n=1 , which are encoded into features {x n } N n=1 . Similarly, for each class name z y , we obtain M textual descriptions by prompting LLMs to generate M -1 fine-grained variants in addition to the hand-crafted prompt, yielding features {z m y } M m=1 . We model each image and class as a discrete distribution, rather than a single embedding. For example, for an image x and class y, we model their distribution as:
P(x) = N n=1 a n δ(x n -x), Q y (z) = M m=1 b m y δ(z m y -z),(5)
where δ(•) denotes the Dirac delta function, and a n , b m y are the associated importance weights. To compute a n for the augmented image feature xn , we assess its entropy with respect to the average class embedding zy = 1 M M m=1 z m y . Specifically, we define:
a n = exp (h(x n )) N n ′ =1 exp (h(x n ′ )) , h(x n ) = - K y=1 p(z y |x n ) log p(z y |x n ). (6
)
The entropy h(x n ) reflects the prediction confidence: views with lower entropy are assigned higher weights. The importance weights b m y for textual features are computed analogously. Unified OT framework. Given the distributions P(x) and Q y (z), the alignment between an adversarial image and each class is measured by the ot distance, which captures the minimal semantic matching cost between image and text features. We seek a transport plan T y ∈ R N ×M that that moves mass from P(x) to Q y (z), subject to the marginal constraints:
d OT (P(x), Q y (z); C j ) = min Ty≥0 ⟨T y , C Π y ⟩, s.t. T y 1 M = a, T ⊤ y 1 N = b j ,(7)
where
a = [a 1 , • • • , a N ] ⊤ and b y = [b 1 y , • • • , b M y ] ⊤
, and 1 N , 1 M are all-ones vectors. C Π y ∈ R N ×M denotes the transportation cost between the N augmented image views and the M textual descriptions of class j, which is usually quantified using the cosine similarity. However, adversarial noise breaks alignment with text features, compromising the reliability of similarity measures. As a result, we design the OT cost matrix based our projected features. For image feature xn , we compute:
C Π y (n, m) = 1 -cos Π(x n ), z m y ,(8)
where cos(•, •) denotes the cosine similarity. We classify by identifying the class y that yields the lowest transport cost:
y = argmin y∈[K] d OT (P(x), Q y (z); C Π y ).(9)
Section 3.3 demonstrates that our OT-based classifier achieves larger decision margins, indicating stronger generalization ability.  .3 65.2 35.9 20.2 6.2 41.3 23.9 44.4 6.9 52.7 22.4 84.0 43.9 59.1 30.8 85.8 67.6  HD 80.9 12.0 58.2 7.3 16.4 1.3 34.9 11.6 39.1 4.6 44.3 2.7 80.3 8.0 53.2 6.4 82.3 31.5  TTC 83.4 57.9 64.2 39.1 18.0 13.8 37.0 27.3 53.2 12.2 48.2 33.0 82.2 57.8 55.1 41.5 86.5 65.8  COLA 87.9 77.2 66.1 50.4 20.9 15.6 41.0 34.0 53.8 19.2 54.2 35.4 84.5 63.8 61.9 45.3 88.1 75.3   CLIP 87.4 1.6 65.5 1.4 20.1 0.0 40.6 2.9 42.6 0.0 52.0 2.4 83.9 1.1 58.5 1.8 85.7 20.9  TeCoA 62.1 37.9 36.8 21.1 5.3 2.3 25.2 16.3 16.6 11.7 20.9 8.7 30.0 12.9 36.7 18.4 71.7 56.2  PMG 65.9 39.3 37.0 21.3 5.6 1.9 21.8 13.7 18.5 11.9 25.4 10.5 36.6 16.6 38.0 20.475.5 61.6 FARE 79.4 33.9 48.0 17.3 10.9 1.4 32.1 14.4 21.9 10.7 38.7 9.1 55.3 12.9 52.4 15.7 81.0 54.9 CW Attacks RN 87.4 3.1 64.6 2.1 19.2 0.0 38.0 3.5 53.2 0.2 52.1 2.4 83.4 1.9 59.7 2.5 86.6 25.9 TTE 88.1 51.1 65.2 35.0 20.2 5.2 41.3 22.6 44.4 6.4 52.7 21.2 84.0 44.6 59.1 29.4 85.8 69.4 HD 80.6 13.8 57.8 8.5 16.2 1.0 34.9 10.1 40.1 3.5 43.6 5.1 81.0 9.8 54.1 7.9 83.0 36.3 TTC 83.4 57.1 64.2 36.8 18.0 12.4 37.0 27.4 53.2 12.7 48.2 30.4 82.2 54.6 55.1 39.4 86.5 66.2 COLA 87.9 63.2 66.1 41.8 20.9 15.3 41.0 31.7 53.8 13.3 54.2 35.2 84.5 54.9 61.9 40.9 88.1 72.9
this section cite: ['b64']

Section: Theoretical Analysis
Global feature alignment preserves pairwise similarity. We show that projection onto the subspace U preserves pairwise similarity among adversarial features. Let x1 and x2 be the adversarial counterparts of two clean features x 1 and x 2 :
x1 = x 1 + δ 1 , x2 = x 2 + δ 2 , (10
)
where each perturbation δ is decomposed as δ = δ ∥ +δ ⊥ , with δ ∥ ∈ U and δ ⊥ ⊥ U. After projection, the adversarial feature satisfies: Π(x) = x + δ ∥ .
(11) Let ∆ and ∆ Π denote the deviation from clean similarity before and after projection, respectively:
∆ = | cos(x 1 , x2 ) -cos(x 1 , x 2 )|, ∆ Π = | cos(Π(x 1 ), Π(x 2 )) -cos(x 1 , x 2 )|.(12)
We show that projection yields lower cosine similarity distortion, i.e., ∆ Π ≤ ∆; the complete proof is given in Appendix A.1.
Our OT-based framework enjoys larger decision margins. Recall that the margin of an OT-based classifier with our projected cost matrix for a discrete distribution P(x) and its label y is defined as:
γ(C Π ) = min y ′ ∈[K] d OT (P(x), Q(z Π y ′ ); C y ′ ) -d OT (P(x), Q(z y ); C Π y ,(13)
which measures the gap between the OT distance to the true class and the closest competing class. Let C y (n, m) = 1 -cos(x n , z m y ) denote the cost matrix using the original perturbed features, and let γ(C) be the corresponding OT classifier margin. We show that the margin of our OT classifier is larger than the original one: γ(C Π ) > γ(C). The detailed proofs are provided in Appendix A.2. Since classifiers with larger margins imply better generalization [5,55], our approach leads to improved robustness against adversarial perturbations.
this section cite: ['b4', 'b54']

Section: Experiments
In this section, we present the experimental results of our method under adversarial perturbations, including performance comparisons, ablation studies, and visualization analyses.
this section cite: []

Section: Setup
Datasets. We evaluate our method on 14 classification datasets spanning a broad range of domains, including generic objects (ImageNet [14], Caltech101 [20]), scenes (SUN397 [58]), textures (DTD [10]), satellite imagery (EuroSAT [23]), and various fine-grained categories such as pets, cars, flowers, food, and aircraft (Pets [39], Cars [26], Flowers [38], Food101 [6], Aircraft [34]). To further assess robustness under distribution shifts, we include five ImageNet variants: ImageNetV2 [46], ImageNet-Sketch [53], ImageNet-A [25], and ImageNet-R [24].
Implementation details. The attack budgets, including PDG attack and CW acctack [36,7], are set of ϵ a = 1/255 in default. The number of steps for attacks is set as 10. All attacks are bounded by a L ∞ radius. For each test image, we generate N = 5 augmented views including the original.
For each class, we use the LLM to generate M = 50 text descriptions. We select the top-C = 256 components from the SVD of class text features to build the projection matrix. All experiments are conducted on a single NVIDIA 3090 GPU if not specified.
this section cite: ['b13', 'b19', 'b57', 'b9', 'b22', 'b38', 'b25', 'b37', 'b5', 'b33', 'b45', 'b52', 'b24', 'b23', 'b35', 'b6']

Section: Comparison methods.
Our experiments are based on the pre-trained CLIP model, using ViT-B/32 as the visual encoder and a Transformer as the text encoder. We compare our method with test-time defences including Anti-Adversary (Anti-Adv) [2], Hedge Defence (HD) [57], Test-Time Transformation Ensembling (TTE) [40], and Test-Time Counterattacks (TTC) [59]. These methods are adapted to CLIP without additional networks. We also include fine-tuning-based baselines: TeCoA [36], PMG [54], and FARE [49], which adversarially fine-tune the vision encoder on TinyImageNet [28].
this section cite: ['b1', 'b56', 'b39', 'b58', 'b35', 'b53', 'b48', 'b27']

Section: Main Results
Results on 14 datasets. We evaluate all methods assuming full access to model weights and gradients by the attacker. Table 1 reports classification accuracy on clean and adversarially perturbed images across 9 diverse datasets. Fine-tuning-based methods such as TeCoA [36], PMG [54], and FARE [49] improve robustness but significantly degrade clean performance. TTC [59], which introduces testtime counterattacks, enhances robustness further but requires stronger counterattack budgets and adds inference complexity.
In contrast, our method consistently improves robustness across all datasets and attack types (PGD and CW), while maintaining competitive clean accuracy. For example, on datasets like Food and Caltech101, our approach achieves over +5% absolute gains in robust accuracy compared to TTC, with only marginal clean performance drops. Table 2 shows results on ImageNet and its challenging variants. Our method consistently outperforms both CLIP and TTC, with especially large robustness gains on ImageNet-A and ImageNet-R-exceeding +7% under PGD attacks.
Results on finetuned CLIP. Our method aligns adversarially perturbed image features with their corresponding textual features and can be flexibly integrated into adversarially fine-tuned models as a plug-and-play module, without requiring architectural changes or additional training. Table 3 and Table 4 present results on ImageNet and its variants under PGD attacks.
TTC [59] improves robustness by generating test-time counterattacks using the fine-tuned model, but it often requires stronger counterattack budgets and introduces additional inference overhead. In contrast, our method consistently improves robust accuracy across all settings while preserving or minimally affecting clean performance. Specifically, on the 9-dataset benchmark, our method improves robust accuracy by +16.5% on TeCoA and +5.0% on PMG over their respective baselines, and by +10.8% and +2.6% over their TTC-augmented variants. On more challenging ImageNet variants, our approach achieves the highest robust accuracy in all cases. particularly excelling on ImageNet-R and ImageNet-Sketch, where robustness improvements of over +10% are observed.
Results on different backbones. To assess the generality of our method, we evaluate its performance across two CLIP backbones: ViT-B/16 and ViT-L/14. As shown in Table 5, our approach consistently achieves superior robustness against PGD attacks compared to both CLIP and TTC across all datasets. On ViT-B/16, our method improves robust accuracy over TTC by up to 12.0%, with consistent gains across 9-datasets, ImageNet, and its variants. Notably, on ViT-L/14, we observe a substantial 35.8% gain in robust accuracy on ImageNet, alongside improvements on the other test domains. These results highlight the adaptability of our method to different model capacities and architectures, and confirm its effectiveness in enhancing adversarial robustness without compromising clean accuracy. Results on large attack budgets. We further evaluate the robustness of all methods under a stronger adversarial budget of ϵ a = 4/255. As shown in Table 6, the performance of all baseline models drops sharply, with most robust accuracies approaching zero. This highlights their vulnerability under high-strength attacks. In contrast, our method maintains significantly higher robust accuracy across all nine datasets, demonstrating strong resistance to adversarial degradation. Notably, our approach achieves over 50% absolute gains in robust accuracy on datasets like Food, SUN, and Caltech101 compared to TTC [59], and outperforms all baselines by a large margin under this challenging setting.
this section cite: ['b35', 'b53', 'b58', 'b58', 'b58']

Section: Ablation Study
Effectiveness of the projected cost. Table 7 presents an ablation study comparing the standard CLIP model, the OT alignment with the original cost matrix C, and our proposed projection-based cost matrix C Π . We observe consistent improvements when applying the subspace projection, indicating its effectiveness in mitigating adversarial perturbations. Specifically, across both PGD and CW attacks, C Π achieves higher robust accuracy than both CLIP and the unprojected OT baseline. On the 9-datasets benchmark, the robust accuracy improves from 2.4% (CLIP) to 46.2%, while clean accuracy is also preserved. Similar trends are observed on ImageNet and its variants, with C Π yielding up to over 3% gain in robust accuracy over C under PGD attakcs.
this section cite: []

Section: Effects of the number of augmentations.
To investigate the sensitivity of our method to augmentation strategies, we evaluate the effect of varying the number of image and class name augmentations on both clean and robust accuracy, as shown in Figure 2. Increasing the number of image augmentations consistently enhances robustness, while the clean accuracy remains stable. However, the improvement becomes marginal when the number exceeds 5. A similar saturation effect is observed in class name augmentation, where performance gains plateau beyond 50 augmentations. These observations indicate that our method is robust to the choice of augmentation hyperparameters.
this section cite: []

Section: Effects of projection matrix construction.
We study how varying the number of singular vectors C used to construct the projection matrix affects classification accuracy. As shown in Figure 3, increasing C steadily improves performance on both clean and adversarial examples across Caltech101 and ImageNet. The gains are more prominent when C is small and gradually saturate beyond C = 200, 100 200 300 400 500 Principal Components 80 85 90 Accuracy (%) Clean Robust (a) Caltech101. 100 200 300 400 500 Principal Components 30 40 50 60 Accuracy (%) Clean Robust (b) ImageNet. 0.1 0.3 0.5 Similarity Score 0 5 10 Density Original Features Attacked Features (a) Attacked Similarity. 0.1 0.3 0.5 Similarity Score 0 5 10 Density Original Features Projected Features (b) Projected Similarity. especially for clean samples. In contrast, robust accuracy improves more slowly, suggesting that while additional components better preserve semantic structure for clean inputs, they provide limited benefit under adversarial perturbations. Based on this observation, we fix C = 256 in subsequent experiments to balance performance and efficiency.
To further understand the effectiveness of projection in Eq. ( 4), we analyze the similarity score distributions between image features and their corresponding text features under different conditions. As shown in Figure 4a, adversarial perturbations significantly reduce the similarity between image and text features, indicating disrupted alignment. In contrast, Figure 4b shows that projecting the attacked features onto the text-induced subspace effectively restores their similarity to the original level. This demonstrates that our projection effectively corrects adversarial misalignment and strengthens semantic consistency across modalities, thereby enhancing robustness in classification. Running time. We compare the inference-time efficiency of our method on ImageNet using CLIP ViT-B/32 with a batch size of 128 on a single NVIDIA 3090 GPU. As shown in Table 8, our method completes evaluation in 28 minutes, significantly faster than TTC (40 minutes) while achieving both higher clean (62.8% vs. 51.7%) and robust (50.0% vs. 40.0%) accuracy. This efficiency stems from the training-free nature of our approach, which avoids the costly iterative optimization required by TTC.
this section cite: []

Section: Limitation and Conclusion
Limitation. While COLA substantially improves adversarial robustness, it still inherits potential biases from the pre-trained vision-language backbone [69,64,66]. In particular, the text-induced subspace may encode dataset-specific priors [19,18], limiting generalization to unseen linguistic or visual domains. Moreover, stronger defenses could provoke more adaptive attacks, suggesting the need for future research on resilience under adaptive adversaries and fairness-aware robustness.
Conclusion. By enhancing robustness against adversarial manipulation, COLA contributes to safer multimodal systems, especially in high-stakes applications such as autonomous driving and medical imaging. We present COLA, a training-free and theoretically grounded framework that improves the adversarial robustness of CLIP by addressing modality misalignment. COLA leverages subspace projection to restore global alignment and employs optimal transport to refine local semantic consistency. By embedding projection into the OT cost computation, it maintains cross-modal alignment without retraining or architectural modification. Theoretical analyses show that COLA reduces cosine distortion and enlarges decision margins, thereby improving generalization. Extensive experiments across 14 benchmarks confirm that COLA consistently enhances zero-shot classification robustness while preserving clean accuracy.
Contents 1 Introduction 1 2 Related Work 2 3 Method 3 3.1 Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3 3.2 Cross Modality Alignment under a Unified OT Framework . . . . . . . . . . . . . 3 3.3 Theoretical Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 4 Experiments 5 4.1 Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 4.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8 5 Limitation and Conclusion 9 A Proofs 15 A.1 Proof of Cosine Similarity Distortion Bound . . . . . . . . . . . . . . . . . . . . . 15 A.2 Proof of OT-Margin Amplification . . . . . . . . . . . . . . . . . . . . . . . . . . 16 B External Results 17 C Algorithm 17 A Proofs A.1 Proof of Cosine Similarity Distortion Bound Let x 1 , x 2 ∈ R d be clean feature vectors with ∥x i ∥ = 1. The adversarial features are denoted by xi = x i + δ i , ∥δ i ∥ ≤ ϵ, where δ i = δ i ∥ + δ i ⊥ , with δ i ∥ ∈ U and δ i ⊥ ⊥ U. Let Π(x i ) = x i + δ i ∥ denote the projection of the adversarial feature onto the subspace U. The cosine similarity between adversarial features and projected features:
cos(x 1 , x2 ) = (x 1 + δ 1 ) ⊤ (x 2 + δ 2 ) ∥x 1 ∥ • ∥x 2 ∥ , cos(Π(x 1 ), Π(x 2 )) = (x 1 + δ 1 ∥ ) ⊤ (x 2 + δ 2 ∥ ) ∥Π(x 1 )∥ • ∥Π(x 2 )∥ .
We define the cosine distortion quantities as
∆ = |cos(x 1 , x2 ) -cos(x 1 , x 2 )| , ∆ Π = |cos(Π(x 1 ), Π(x 2 )) -cos(x 1 , x 2 )| .
We now compare ∆ and ∆ Π . Using a second-order Taylor approximation for small perturbations ∥δ i ∥ ≪ 1, we obtain:
cos(x 1 , x2 ) ≈ (x ⊤ 1 x 2 + x ⊤ 1 δ 2 + δ ⊤ 1 x 2 + δ ⊤ 1 δ 2 )(1 -x ⊤ 1 δ 1 - 1 2 ∥δ 1 ∥ 2 )(1 -x ⊤ 2 δ 2 - 1 2 ∥δ 2 ∥ 2 ). cos(Π(x 1 ), Π(x 2 )) ≈(x ⊤ 1 x 2 + x ⊤ 1 δ 2 ∥ + (δ 1 ∥ ) ⊤ x 2 + (δ 1 ∥ ) ⊤ δ 2 ∥ ) × (1 -x ⊤ 1 δ 1 ∥ - 1 2 ∥δ 1 ∥ ∥ 2 )(1 -x ⊤ 2 δ 2 ∥ - 1 2 ∥δ 2 ∥ ∥ 2 ).
To simplify analysis, assume δ 1 = δ 2 = δ and δ 1 ∥ = δ 2 ∥ = δ ∥ . Since x i ∈ U and δ ⊥ ⊥ U, we have x ⊤ i δ ⊥ = 0 and thus x ⊤ i δ = x ⊤ i δ ∥ . Under this setting, both distortions simplify to:
∆ ≈ 2x ⊤ 1 δ ∥ -2(x ⊤ 1 x 2 )(x ⊤ 1 δ ∥ ) + O(ϵ 2 ), ∆ Π ≈ 2x ⊤ 1 δ ∥ -2(x ⊤ 1 x 2 )(x ⊤ 1 δ ∥ ) + O(ϵ 2 ).
Thus, the first-order terms in ∆ and ∆ Π are identical. However, in the general case where ∥δ ⊥ ∥ > 0, the norm of the full feature xi is larger than that of its projection Π(x i ), reducing the cosine similarity in the unprojected case.
Using Cauchy-Schwarz and bounding terms:
|∆ Π | ≤ 2∥δ ∥ ∥(1 + |x ⊤ 1 x 2 |), |∆| ≥ 2∥δ ∥ ∥(1 + |x ⊤ 1 x 2 |) 1 + ∥δ ⊥ ∥ 2 ∥δ ∥ ∥ 2 .
Therefore, when the perturbation contains a non-zero orthogonal component (∥δ ⊥ ∥ > 0), we have
∆ Π ∆ ≤ ∥δ ∥ ∥ 2 ∥δ∥ 2 < 1,
which shows that the cosine similarity distortion is strictly reduced by projecting the adversarial features onto the subspace U.
this section cite: ['b68', 'b63', 'b65', 'b18', 'b17']

Section: References
Ref_id:b0 Title: Principal component analysis Year: (2010)
Ref_id:b1 Title: Combating adversaries with anti-adversaries Year: (2022)
Ref_id:b2 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b3 Title: Recent advances in adversarial training for adversarial robustness Year: (2021)
Ref_id:b4 Title: Spectrally-normalized margin bounds for neural networks Year: (2017)
Ref_id:b5 Title: Food-101 -mining discriminative components with random forests Year: (2014)
Ref_id:b6 Title: Towards evaluating the robustness of neural networks Year: (2017)
Ref_id:b7 Title: Plot: Prompt learning with optimal transport for vision-language models Year: (2022)
Ref_id:b8 Title: Graph-based global reasoning networks Year: (2019)
Ref_id:b9 Title: Describing textures in the wild Year: (2014)
Ref_id:b10 Title: Optimal transport for domain adaptation Year: (2016)
Ref_id:b11 Title: Evaluating the adversarial robustness of adaptive test-time defenses Year: (2022)
Ref_id:b12 Title: Sinkhorn distances: Lightspeed computation of optimal transport Year: (2013)
Ref_id:b13 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b14 Title: Stabilizing modality gap & lowering gradient norms improve zero-shot adversarial robustness of vlms Year: (2025)
Ref_id:b15 Title: Mitigate the gap: Improving cross-modal alignment in CLIP Year: (2025)
Ref_id:b16 Title: Xiang Wang, and Tat-Seng Chua. Towards neuron attributions in multi-modal large language models Year: (2024)
Ref_id:b17 Title: Xiangnan He, and Tat-Seng Chua. Alphaedit: Null-space constrained knowledge editing for language models Year: (2025)
Ref_id:b18 Title: Safemlrm: Demystifying safety in multi-modal large reasoning models Year: (2025)
Ref_id:b19 Title: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories Year: (2004)
Ref_id:b20 Title: Countering adversarial images using input transformations Year: (2017)
Ref_id:b21 Title: Adaptive distribution calibration for few-shot learning with hierarchical optimal transport Year: (2022)
Ref_id:b22 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b23 Title: The many faces of robustness: A critical analysis of out-of-distribution generalization Year: (2021)
Ref_id:b24 Title: Natural adversarial examples Year: (2021)
Ref_id:b25 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b26 Title: Iterative label cleaning for transductive and semi-supervised few-shot learning Year: (2021)
Ref_id:b27 Title: Tiny imagenet visual recognition challenge Year: (2015)
Ref_id:b28 Title: Visual-text cross alignment: Refining the similarity score in vision-language models Year: (2024)
Ref_id:b29 Title: BLIP: bootstrapping languageimage pre-training for unified vision-language understanding and generation Year: (2022)
Ref_id:b30 Title: One prompt word is enough to boost adversarial robustness for pre-trained vision-language models Year: (2024)
Ref_id:b31 Title: Language-driven anchors for zero-shot adversarial robustness Year: (2024)
Ref_id:b32 Title: Towards deep learning models resistant to adversarial attacks Year: (2018)
Ref_id:b33 Title: Finegrained visual classification of aircraft Year: (2013)
Ref_id:b34 Title: Adversarial attacks are reversible with natural supervision Year: (2021)
Ref_id:b35 Title: Understanding zero-shot adversarial robustness for large-scale models Year: (2023)
Ref_id:b36 Title: Diffusion models for adversarial purification Year: (2022)
Ref_id:b37 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b38 Title: Cats and dogs Year: (2012)
Ref_id:b39 Title: Enhancing adversarial robustness via test-time transformation ensembling Year: (2021)
Ref_id:b40 Title: Computational optimal transport: With applications to data science Year: (2019)
Ref_id:b41 Title: What does a platypus look like? generating customized prompts for zero-shot image classification Year: (2023)
Ref_id:b42 Title: Accelerating diffusion transformer via gradient-optimized cache Year: (2025)
Ref_id:b43 Title: Accelerating diffusion transformer via error-optimized cache Year: (2025)
Ref_id:b44 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b45 Title: Do imagenet classifiers generalize to imagenet Year: (2019)
Ref_id:b46 Title: Overfitting in adversarially robust deep learning Year: (2020)
Ref_id:b47 Title: Waffling around for performance: Visual classification with random words and broad concepts Year: (2023)
Ref_id:b48 Title: Robust CLIP: unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models Year: (2024)
Ref_id:b49 Title:  Year: (2019)
Ref_id:b50 Title: Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks Year: (2014)
Ref_id:b51 Title: Tuning multi-mode token-level prompt alignment across modalities Year: (2023)
Ref_id:b52 Title: Learning robust global representations by penalizing local predictive power Year: (2019)
Ref_id:b53 Title: Pre-trained model guided fine-tuning for zero-shot adversarial robustness Year: (2024)
Ref_id:b54 Title: On the margin theory of feedforward neural networks Year: (2018)
Ref_id:b55 Title: Fast is better than free: Revisiting adversarial training Year: (2020)
Ref_id:b56 Title: Attacking adversarial attacks as a defense Year: (2021)
Ref_id:b57 Title: SUN database: Large-scale scene recognition from abbey to zoo Year: (2010)
Ref_id:b58 Title: Clip is strong enough to fight back: Test-time counterattacks towards zero-shot adversarial robustness of clip Year: (2025)
Ref_id:b59 Title: Scalable gromov-wasserstein learning for graph partitioning and matching Year: (2019)
Ref_id:b60 Title: Adversarial purification with score-based generative models Year: (2021)
Ref_id:b61 Title: Theoretically principled trade-off between robustness and accuracy Year: (2019)
Ref_id:b62 Title: Adversarial prompt tuning for vision-language models Year: (2024)
Ref_id:b63 Title: Robust fine-tuning of zero-shot models via variance reduction Year: (2024)
Ref_id:b64 Title: Project-probe-aggregate: Efficient fine-tuning for group robustness Year: (2025)
Ref_id:b65 Title: Generalized logit adjustment: Calibrating fine-tuned models by removing label bias in foundation models Year: (2023)
Ref_id:b66 Title: Boosting few-shot learning via attentive feature regularization Year: (2024)
Ref_id:b67 Title: Dynamic multimodal prototype learning in visionlanguage models Year: (2025)
Ref_id:b68 Title: Enhancing zero-shot vision models by label-free prompt distribution learning and bias correcting Year: (2024)
Ref_id:b69 Title: Selective vision-language subspace projection for few-shot CLIP Year: (2024)
Ref_id:b70 Title: Awt: Transferring vision-language models via augmentation, weighting, and transportation Year: (2024)
