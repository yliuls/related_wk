Title: Revisiting Continuity of Image Tokens for Cross-Domain Few-shot Learning
Abstract: Vision Transformer (ViT) has achieved remarkable success due to its large-scale pretraining on general domains, but it still faces challenges when applying it to downstream distant domains that have only scarce training data, which gives rise to the Cross-Domain Few-Shot Learning (CDFSL) task. Inspired by Self-Attention's insensitivity to token orders, we find an interesting phenomenon neglected in current works: disrupting the continuity of image tokens (i.e., making pixels not smoothly transited across patches) in ViT leads to a noticeable performance decline in the general (source) domain but only a marginal decrease in downstream target domains. This questions the role of image tokens' continuity in ViT's generalization under large domain gaps. In this paper, we delve into this phenomenon for an interpretation. We find continuity aids ViT in learning larger spatial patterns, which are harder to transfer than smaller ones, enlarging domain distances. Meanwhile, it implies that only smaller patterns within each patch could be transferred under extreme domain gaps. Based on this interpretation, we further propose a simple yet effective method for CDFSL that better disrupts the continuity of image tokens, encouraging the model to rely less on large patterns and more on smaller ones. Extensive experiments show the effectiveness of our method in reducing domain gaps and outperforming state-of-the-art works. Codes and models are available at https://github.com/shuaiyi308/ReCIT.

Section: 
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). We find an interesting phenomenon: although disrupting the continuity of image tokens in the source domain has a substantial impact on the performance of ViT-based models, the model's performance in the target domain, which undergoes an equivalent level of continuity disruption, is only marginally affected. In this paper, we will delve into this phenomenon for an interpretation, explore the role of image tokens' continuity in model generalization, and propose methods based on it for better cross-domain few-shot learning.
this section cite: []

Section: Introduction
Vision Transformer (ViT) has achieved great success across numerous tasks (Yuan et al., 2021;Wu et al., 2022) because of its ability to learn from large-scale datasets (Naseer et al., 2021), which makes it a prevailing option for downstream applications by generalizing an upstream-pretrained ViT to downstream expert tasks. However, in real-world scenarios, downstream tasks can be in domains distant from upstream large-scale datasets, and it may not be easy for downstream tasks to collect sufficient training samples, which makes it challenging for the generalization and finetuning of ViT (Zou et al., 2022;2024b). To mitigate this issue, Cross-Domain Few-Shot Learning (CDFSL) has been proposed, aiming to transfer general knowledge from source domains, such as ImageNet (Deng et al., 2009), to target domains, like medical datasets (Mohanty et al., 2016), that possess only a scarcity of training samples (Oh et al., 2022).
However, the generalization of ViT under large domain gaps is still under-explored. Different from Convolutional Neural Networks (CNN), ViT partitions the image into nonoverlapping patches for input and takes Multi-head Self-Attention (MSA) (Chen et al., 2023) to model the coherence of patches. However, MSA itself is not sensitive to the order of input tokens, i.e., two subsequent tokens do not have to be continuous in pixels. The only assurance of continuity is the positional embeddings (Dosovitskiy et al., 2021), which is relatively weak. Inspired by this characteristic, we find an intriguing phenomenon that is ignored by current works: when the continuity of images is disrupted e.g., by removing positional embeddings, shuffling image patches, or shuffling the amplitude or phase of image patches in the frequency domain (Fig. 1a), the performance of ViT experiences a noticeable decline in the source domain but decreases only marginally on target domains (Fig. 1b). This phenomenon questions the role of image tokens' continuity in ViT's generalization under large domain gaps.
In this paper, we delve into this phenomenon for an interpretation. We discover that disrupting image tokens' continuity can paradoxically be beneficial in reducing domain gaps. Then, we find that the more disrupted image tokens' continuity (by binding fewer patches not disrupted), the more generally increased domain similarity between source and target domains. Based on these experiments, we interpret the continuity as an aid of ViT in learning larger spatial patterns. However, since large spatial patterns, e.g., a whole dog, are harder to transfer to target domains than smaller patterns, e.g., the head of a dog, under extremely large domain gaps, this phenomenon implies that only smaller patterns within each patch could be transferred to target domains, therefore interrupting image tokens' continuity has only marginal effect on target-domain performance.
Drawing upon this interpretation, we further propose a simple but effective method tailored for the CDFSL task that better disrupts the continuity of image tokens, encouraging the model to strengthen the learning of smaller spatial patterns and reduce its reliance on large ones, thereby enhancing the model's generalization to downstream tasks. Specifically, we integrate the continuity disruption of image tokens in both spatial and frequency domains, and construct a balanced disruption among different style distributions, ensuring the diversity of the disruption. Extensive experiments on four CDFSL benchmarks with large domain gaps show that we can outperform state-of-the-art performance and effectively reduce domain gaps.
In summary, our contributions can be listed as follows.
• To the best of our knowledge, we are the first to consider image tokens' continuity in the CDFSL task.
• We find a phenomenon that is neglected by others: pretrained ViT undergoes a much less performance decline when disrupting image tokens' continuity on target domains than on general (source) domains.
• We delve into this phenomenon for an interpretation: continuity aids ViT in learning large spatial patterns; however, under large domain gaps, large patterns across patches are hard to transfer, making the disruption of large patterns less effective on target domains.
• Based on this interpretation, we further propose a novel method to better disrupt the continuity of image tokens for CDFSL, thereby enabling the model to reduce its reliance on large patterns and enhance its learning of smaller ones, thereby enhancing its transferability.
• Extensive experiments on four benchmark datasets validate our rationale and state-of-the-art performance.
this section cite: ['b37', 'b22', 'b43', 'b7', 'b21', 'b23', 'b2', 'b8']

Section: Delve into Continuity of Image Tokens in Cross-Domain Few-Shot Learning

this section cite: []

Section: Preliminaries
Cross-Domain Few-Shot Learning (CDFSL) entails a model to acquire knowledge from a source-domain dataset abundant with training samples (e.g., miniImageNet (Vinyals et al., 2016)), then transfer it to downstream tasks, enabling learning of target-domain datasets using merely a handful of training instances. Finally, the model is evaluated on target datasets.
Specifically, we denote the source dataset as D S = {x S i , y S i } N i=1 with x S i and y S i symbolizing the ith training sample and its corresponding label, respectively. Analogously, D T = {x T i , y T i } N ′ i=1 represents the target dataset. During the learning and evaluation phases on D T , to ensure a fair comparison, current research (Fu et al., 2022;Zou et al., b) employs a k-way n-shot paradigm. This involves sampling from D T to construct limited datasets, known as episodes, each comprising k classes with n training samples per class. Based on these episodes, the model learns from the k * n samples, collectively termed the support set {x T ij , y T ij } k,n i=1,j=1 , and its performance is assessed using testing samples from the same k classes, referred to as the query set {x T q }. The Vision Transformer (ViT) has recently gained significant popularity in vision-related tasks. It operates by dividing an image x ∈ R H×W ×C into fixed-sized patches x p ∈ R M ×(P 2 •C) , Where (H, W ) denotes the resolution of the original image, C represents the number of channels, (P, P ) signifies the resolution of each image patch, and M = HW/P 2 is the resulting count of patches. This count, M , also functions as the number of input tokens for the Transformer. Then each image patch is flattened and projected into a D-dimensional space through a trainable linear projection E ∈ R (P 2 •C)×D termed patch embeddings. Additionally, a learnable embedding called class token (denoted as x class ) is prepended to the sequence of patch embeddings. To maintain positional information, the patch embeddings are added with position embeddings E pos ∈ R (M +1)×D , which can be represented as
z 0 = x class ; x 1 p E; x 2 p E; • • • ; x M p E + E pos , (1
)
The resultant sequence of embedding vectors subsequently serves as the input to an encoder architecture that consists of L stacked blocks, each encompassing a multiheaded selfattention (MSA) network, a Multi-Layer Perceptron (MLP) network, Layernorm (LN) and residual connections. The overall process can be depicted as follows
z ′ l = MSA (LN (z l-1 )) + z l-1 , (2) z l = MLP (LN (z ′ l )) + z ′ l , (3) f (x S i ) = LN (z L ) ,(4)
In this paper, we focus on exploring ViT's downstream generalization on the CDFSL task. We follow (Fu et al., 2023;Zou et al., a) to employ the DINO (Zhang et al., 2022) pretraining on ImageNet (Deng et al., 2009) as the initialization. Then, we train the ViT on D S by minimizing the cross-entropy loss relevant to the source-domain label space |Y S |, with a fully connected (FC) layer as
L = 1 N N i L cls (ϕ(f (x S i )), y S i ),(5)
where ϕ(•) represents the FC layer and f (•) denotes ViT. Finally, we employ prototype-based classification (Zhou et al., 2023) for target-domain recognition with a distance function d(•, •) as
ŷT q = arg min i d( 1 n j f (x T ij ), f (x T q )),(6)
this section cite: ['b30', 'b10', 'b11', 'b38', 'b7', 'b40']

Section: Breaking the continuity of image tokens
As illustrated in Fig. 1a, we first contemplate four of the simplest approaches to disrupt the continuity of images, applying each separately to the training set within the source domain to facilitate the model's training on that domain.
this section cite: []

Section: Remove Position Embedding (RPE):
The first method consists of inputting the image patches directly into the encoder of ViT, without incorporating positional embeddings as
z 0 = x class ; x 1 p E; x 2 p E; • • • ; x M p E ,(7)
Shuffle Patches (SP): The second approach involves shuffling the image patches directly, concatenating them with a class token, and then adding positional embeddings before inputting them into the encoder:
z 0 = x class ; x 1 ′ p E; x 2 ′ p E; • • • ; x M ′ p E + E pos , (8
)
Shuffle Patch Amplitude (SPA): The two subsequent methods involve transitioning image patches from the spatial domain to the frequency domain, starting with obtaining the Fourier transformations of the input patches
x p F (xp)[m, n] = H ′ -1 h=0 W ′ -1 w=0 xp[h, w] exp -2π h H ′ m+ w W ′ n , (9
)
where i 2 = -1 and m, n denote spatial frequencies.
When Re(F (x)[•, •]) and Im(F (x)[•, •]) represent the real and imaginary components of the Fourier spectrum, respectively, the corresponding amplitude spectrum A(x)[•, •] and phase spectrum P (x)[•, •] can be expressed as follows
A(xp)[m, n] = Re(F (xp)[m, n]) 2 + Im(F (x)[m, n]) 2 ,(10)
P (xp)[m, n] = arctan Im(F (xp)[m, n]) Re(F (xp)[m, n]) ,(11)
Our third method disrupts the amplitudes of the patches within an image, merges them with the original phases of these patches, and then applies an inverse Fourier transform to revert the combined data back to the spatial domain as
x k p = iDFT A x k ′ p ⊗ e i•P x k p ,(12)
where k denote kth patch in image, k ′ denote after be shulffed, the kth patch's amplitude.
Shuffle Patch Phase (SPP): The fourth method, in contrast to the third one, disrupts continuity by shuffling the phases while retaining the amplitudes as
x k p = iDFT A x k p ⊗ e i•P x k ′ p ,(13)
common featur e within patches:
e.g. the fin of fish after FFT shuffle patch phase … … after FFT shuffle patch amplitude … … shuffle patches remove position embedding … … … …
Figure 3. Take a fish as an example, although disrupting continuity distorts its overall shape, it is still feasible to recognize the fish's patterns in individual patches, such as fins and eyes. This indicates that the continuity between patches primarily assists the model in learning larger spatial patterns; however, even after disrupting the continuity, the model can only recognize the patterns maintained within each patch, which is smaller but easier to transfer.
this section cite: []

Section: What role does ViT's weak continuity play under large domain gaps?
In Fig. 1, we observe that a pretrained ViT is highly sensitive to disruptions in continuity on the source domain that is similar to its pretraining data, yet exhibits a much smaller impact on target domains that are distant from the source domain. Therefore, we are inspired to question the role that image tokens' continuity plays under large domain gaps.
Firstly, we quantitatively assess the domain distance between the source and target domains using the Centered Kernel Alignment (CKA (Kornblith et al., 2019)) similarity following (Oh et al., 2022;Davari et al., 2022). Specifically, utilizing ViT as the backbone network, we extract features from images belonging to different domains and then compute the CKA similarity between different domains' features by aligning the channel dimension. A higher CKA similarity indicates a smaller domain distance, implying that the model encompasses less domain-specific information.
The results are in Fig. 2, from which we observe the domain similarity between the source and target datasets increased significantly, although in Fig. 1 the model's performance declines on all datasets. Intriguingly, the greater the decline in model performance in Fig. 1, the more the increase in domain similarities. This indicates the surprising benefit of disrupting continuity in reducing domain discrepancies, thereby enhancing the model's transferability.
this section cite: ['b16', 'b23', 'b6']

Section: Why does breaking continuity reduce domain discrepancy?
To account for breaking continuity reduces domain distance, we look back on the disrupted images. Although the connection of each patch is disrupted, patterns within each patch are not disrupted. For example, as shown in Fig. 3, given an image of a fish, although the patches are shuffled, the internal details of each patch still allow us to discern features such as fish scales, eyes, fins, tail, and other distinctive attributes. Based on such a disrupted image, the model extracts features majorly based on the patterns maintained within each patch, while the patterns across patches are lost.
Intuitively, patterns within each patch are spatially smaller, such as fish eyes, while patches are interconnected with their neighbors to build spatially larger patterns, such as a whole fish. Simultaneously, larger patterns are always harder to transfer than smaller ones (Zou et al., 2024b). For example, capturing a pattern similar to a whole fish in a dog is difficult, but capturing a pattern similar to a fish eye is relatively easier, e.g., the dog eyes. Therefore, we hypothesize that breaking continuity essentially breaks the large patterns across patches into small patterns within each patch. Under large domain gaps, large patterns are inherently harder to transfer. Therefore, the major effective patterns on target domains are small patterns within each patch. Consequently, disrupting token continuity has a much smaller effect on target domains, as the recognition of target domains is only marginally based on large patterns across patches. Similarly, since larger patterns are effective for the source domain, the performance of source-domain recognition drops drastically. On the other hand, breaking continuity forces the model to extract features majorly based on smaller patterns, aligning with the patterns used on target domains. Therefore, the domain similarity increases.
To validate our hypothesis, we disrupt images to varying degrees of shuffling, in order to verify how the maintained patterns' spatial size influences the performance and domain similarity. Specifically, each image is divided into pseudopatches of sizes 1 * 1, 2 * 2, 4 * 4, 7 * 7, 8 * 8, and 14 * 14.
For instance, 2 * 2 signifies dividing the whole image into four equal-sized sections along both its length and width 1 . Then, these pseudo-patches are randomly shuffled and then reassembled to form the shuffled images, which are then input into ViT. Since patches within each pseudo-patch are not disrupted, the size of spatial patterns is larger.
As in Fig. 4a, the performance consistently decreases with the growing number of pseudo-patches, i.e., smaller patterns within each pseudo-patch. Meanwhile, domain similarities between source and target domains consistently increase as the pseudo-patch size reduces, verifying it is the maintained pattern's spatial size that influences domain similarities.
this section cite: []

Section: Conclusion and Discussion
Based on it, we interpret as follows. The continuity between image tokens essentially assists models in learning larger spatial patterns, which are beneficial for source-domain classification, therefore breaking the continuity significantly harms source-domain performance. However, excessively large patterns often struggle to transfer to target domains. Conversely, smaller patterns are easier to transfer. Under large domain gaps, most patterns transferred to target domains are those small ones that are within each patch, therefore disrupting the continuity has smaller influences on the target-domain performance. By aligning patterns used in feature extraction for source and target domains to small patterns, the domain similarity also increases.
this section cite: []

Section: References
Ref_id:b0 Title: Emerging properties in self-supervised vision transformers Year: ()
Ref_id:b1 Title: Amplitude-phase recombination: Rethinking robustness of convolutional neural networks in frequency domain Year: (2021)
Ref_id:b2 Title: Accumulated trivial attention matters in vision transformers on small datasets Year: (2023-01)
Ref_id:b3 Title: Conditional positional encodings for vision transformers Year: (2021)
Ref_id:b4 Title: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic) Year: (2019)
Ref_id:b5 Title: Confess: A framework for single source cross-domain few-shot learning Year: (2022)
Ref_id:b6 Title: Reliability of cka as a similarity measure in deep learning Year: (2022)
Ref_id:b7 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b8 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b9 Title: Meta-fdmixup: Crossdomain few-shot learning guided by labeled target data Year: (2021)
Ref_id:b10 Title: Wavesan: Wavelet based style augmentation network for crossdomain few-shot learning Year: (2022)
Ref_id:b11 Title: Meta style adversarial training for cross-domain few-shot learning Year: (2023)
Ref_id:b12 Title: A broader study of cross-domain few-shot learning Year: (2020)
Ref_id:b13 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b14 Title: Adversarial feature augmentation for cross-domain few-shot classification Year: (2022)
Ref_id:b15 Title: A method for stochastic optimization Year: (2017)
Ref_id:b16 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b17 Title: Ranking distance calibration for cross-domain few-shot learning Year: (2022)
Ref_id:b18 Title: Revisiting local descriptor based image-to-class measure for few-shot learning Year: (2019)
Ref_id:b19 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: ()
Ref_id:b20 Title: Reconstruction target matters in masked image modeling for cross-domain fewshot learning Year: (2024)
Ref_id:b21 Title: Using deep learning for image-based plant disease detection Year: (2016-09)
Ref_id:b22 Title: Intriguing properties of vision transformers Year: (2021)
Ref_id:b23 Title: Understanding cross-domain few-shot learning based on domain similarity and few-shot difficulty Year: (2022)
Ref_id:b24 Title: Self-training for few-shot transfer across extreme task differences Year: (2021)
Ref_id:b25 Title: Rapid learning or feature reuse? towards understanding the effectiveness of maml Year: (2019)
Ref_id:b26 Title: Espt: A self-supervised episodic spatial pretext task for improving few-shot learning Year: (2023)
Ref_id:b27 Title: Pushing the limits of simple pipelines for few-shot learning: External data and fine-tuning make a difference Year: (2022)
Ref_id:b28 Title: Explanation-guided training for cross-domain few-shot classification Year: (2021)
Ref_id:b29 Title: Cross-domain few-shot classification via learned featurewise transformation Year: (2020)
Ref_id:b30 Title: Matching networks for one shot learning Year: (2016)
Ref_id:b31 Title: Masked embedding modeling with rapid domain adjustment for few-shot image classification Year: (2023)
Ref_id:b32 Title: Cross-domain few-shot classification via adversarial task augmentation Year: (2021)
Ref_id:b33 Title: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases Year: (2017-07)
Ref_id:b34 Title: Few-shot classification with feature map reconstruction networks Year: (2021)
Ref_id:b35 Title: Tinyvit: Fast pretraining distillation for small vision transformers Year: ()
Ref_id:b36 Title: Enhancing information maximization with distance-aware contrastive learning for source-free crossdomain few-shot learning Year: (2024)
Ref_id:b37 Title: Tokens-to-token vit: Training vision transformers from scratch on imagenet Year: (2021-10)
Ref_id:b38 Title: Detr with improved denoising anchor boxes for end-to-end object detection Year: (2022)
Ref_id:b39 Title: Metagan: An adversarial approach to few-shot learning Year: (2018)
Ref_id:b40 Title: Revisiting prototypical network for cross domain few-shot learning Year: (2023-06)
Ref_id:b41 Title: Attention temperature matters in vit-based cross-domain few-shot learning Year: ()
Ref_id:b42 Title: A closer look at the cls token for cross-domain few-shot learning Year: ()
Ref_id:b43 Title: Margin-based few-shot class-incremental learning with class-level overfitting mitigation Year: (2022)
Ref_id:b44 Title: Flatten longrange loss landscapes for cross-domain few-shot learning Year: (2024)
Ref_id:b45 Title: Compositional few-shot class-incremental learning Year: (2024)
