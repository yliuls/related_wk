Title: Scaling Laws for Task-Optimized Models of the Primate Visual Ventral Stream
Abstract: When trained on large-scale object classification datasets, certain artificial neural network models begin to approximate core object recognition behaviors and neural response patterns in the primate brain. While recent machine learning advances suggest that scaling compute, model size, and dataset size improves task performance, the impact of scaling on brain alignment remains unclear. In this study, we explore scaling laws for modeling the primate visual ventral stream by systematically evaluating over 600 models trained under controlled conditions on benchmarks spanning V1, V2, V4, IT and behavior. We find that while behavioral alignment continues to scale with larger models, neural alignment saturates. This observation remains true across model architectures and training datasets, even though models with stronger inductive biases and datasets with higher-quality images are more computeefficient. Increased scaling is especially beneficial for higher-level visual areas, where small models trained on few samples exhibit only poor alignment. Our results suggest that while scaling current architectures and datasets might suffice for alignment with human core object recognition behavior, it will not yield improved models of the brain's visual ventral stream, highlighting the need for novel strategies in building brain models.

Section: 
The advent of neural networks has revolutionized our understanding and modeling of complex neural processes. A particularly active area of study is the ventral visual stream in primates, a key pathway in the brain responsible for processing visual information (Goodale & Milner, 1992;Grill-Spector et al., 2001;Malach et al., 2002;Kriegeskorte et al., 2008). Neural networks, when trained on extensive datasets, have emerged as the most accurate quantitative Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
a) b)
Figure 1: a) For a given compute budget (C), we determine the scaling laws for maximal neural and behavioral alignment to the primate visual ventral stream. b) We find consistent scaling laws for brain and behavioral alignment across over 600 models. While we predict models to approach perfect behavioral alignment at large scales, the effect of scaling on brain alignment is already saturating. tools for simulating the response patterns of neurons within this stream (Yamins et al., 2014;Schrimpf et al., 2018). These advanced models offer a precise computational account of how neural mechanisms in the brain give rise to visual perception.
Recent developments in machine learning have emphasized the significance of both the volume of training data and the complexity of model architectures (Kaplan et al., 2020;Hoffmann et al., 2022;Zhai et al., 2022;Bahri et al., 2022;Antonello et al., 2023;Muennighoff et al., 2023;Aghajanyan et al., 2023;Isik et al., 2024). These findings raise the question: Can we build better models of the brain by scaling up model architectures and dataset sizes? Recent studies have found that in pre-trained models, the number of parameters and dataset samples respectively seem to improve predictions of fMRI and behavioral measurements (Antonello et al., 2023;Muttenthaler et al., 2023). With the numerous differences between pre-trained models however, the relative contributions of model parameters and dataset size to brain and behavioral alignment are not clear.
Despite recent successes in using neural networks as models of the brain, a comprehensive understanding of how model scale-separately and jointly across parameters, dataset size, and compute-affects functional alignment with different cortical areas remains elusive. Previous studies have often relied on heterogeneously trained models using off-the-shelf checkpoints (Conwell et al., 2024), or focused narrowly on specific brain areas (e.g., IT only (Linsley et al., 2023)), frequently using proxy quantities such as task performance. Our work addresses these limitations through a systematic, from-scratch training protocol spanning over 600 models, enabling controlled comparisons and robust parametric estimation of scaling laws for both behavioral and neural alignment across the entire ventral visual hierarchy. This approach offers a clearer disentanglement of the respective contributions of architecture, data, and optimization objective to brain modeling.
In this paper, we examine how scaling -of model parameters and training dataset size -impacts the alignment of artificial neural networks with the primate ventral visual stream. We systematically train models from a variety of architectural families on image classification datasets which allows us to independently control and observe the effects of model complexity and data volume. To capture the observed trends, we introduce parametric power-law trends that describe the impact of scale on alignment with behavior and brain regions along the visual ventral stream. We summarize the contributions of this work as follows:
• While scale initially improves alignment, brain alignment saturates. Behavioral alignment on the other hand continues to improve.
• Increasing both parameter count and training dataset size improves alignment, with data providing more gains over model scaling.
• Architectures with stronger inductive bias (e.g., convolutions and recurrence) and datasets with higher-quality images are more sample-and compute-efficient.
• Fitting parametric power-law curves, we find that model alignment with higher-level brain regions and especially behavior benefits the most from scaling.
• We publicly release our training code, evaluation pipeline, and over 600 checkpoints for models trained in a controlled manner to enable future research.
this section cite: ['b21', 'b24', 'b47', 'b35', 'b70', 'b57', 'b31', 'b28', 'b72', 'b3', 'b1', 'b50', 'b0', 'b29', 'b1', 'b51', 'b14', 'b41']

Section: Related Work
Primate Visual Ventral Stream. The ventral visual stream, a critical pathway in the primate brain, including humans, plays a key role in visual perception, extending from the occipital to the temporal lobes and serving as the "what pathway" for object recognition and form representation (Goodale & Milner, 1992;Grill-Spector et al., 2001;Malach et al., 2002;Kriegeskorte et al., 2008). Beginning in the primary visual cortex (V1), where basic visual information from retinal ganglion cells is processed, the ventral stream proceeds through areas such as V2, V3, V4, and the inferotemporal cortex (IT), each responsible for increasingly complex features of visual perception (Kandel et al., 2000).
Despite decades of research and a wealth of brain data, the precise neural mechanisms underlying visual perception are not well understood.
this section cite: ['b21', 'b24', 'b47', 'b35', 'b30']

Section: Modeling the Primate Visual Ventral Stream.
Particular artificial neural networks (ANNs) are the most accurate models of brain responses in the visual ventral stream and associated core object recognition behaviors (Schrimpf et al., 2018;2020). Models optimized for ecologically viable tasks (Yamins & DiCarlo, 2016) in particular have demonstrated strong brain and behavioral alignment (Yamins et al., 2014;Khaligh-Razavi & Kriegeskorte, 2014;Cadena et al., 2019;Schrimpf et al., 2018;Nayebi et al., 2018;Kietzmann et al., 2019;Rajalingham et al., 2018;Zhuang et al., 2021;Geiger et al., 2022) -notably these models are trained purely on image classification datasets, without fitting to brain data.
this section cite: ['b57', 'b34', 'b71', 'b70', 'b32', 'b8', 'b57', 'b52', 'b33', 'b55']

Section: Scaling Laws.
Recent advancements in artificial intelligence are driven by scaling the model size and training data. Empirical evidence suggests a power-law relationship between model performance and both model parameters and dataset size, indicating that continued scaling will further improve performance (Kaplan et al., 2020;Cherti et al., 2023;Zhai et al., 2022;Hoffmann et al., 2022;Dehghani et al., 2023;Henighan et al., 2020;Brown et al., 2020;Bahri et al., 2022;Hestness et al., 2017). The power-law exponents enable the optimal allocation of compute between model parameters and dataset samples, such that performance is maximized (Kaplan et al., 2020;Hoffmann et al., 2022).
While scaling laws for machine learning performance has been extensively studied, the scaling laws for brain alignment remain unclear. Recent studies suggest an involvement of both model size and data volume in the functional alignment with brain data (Azabou et al., 2023;Benchetrit et al., 2023;Caro et al., 2024;Antonello et al., 2023). Conversely, Muttenthaler et al. (2023) indicate that sample size is critical for behavioral alignment. We here unify these results, in the realm of the primate visual ventral stream, into quantitative scaling laws for how model and dataset sizes relate to alignment with the brain and behavior.
this section cite: ['b31', 'b12', 'b72', 'b28', 'b26', 'b3', 'b27', 'b31', 'b28', 'b2', 'b4', 'b9', 'b1', 'b51']

Section: Methods
Neural & Behavioral Alignment. To evaluate the alignment of our model with brain function, we utilize a range of benchmarks from Brain-Score (Schrimpf et al., 2018;2020). These benchmarks assess model performance by comparing model activations or behavior with primate neural data using the same images. Specifically, the V1 and V2 benchmarks compare model outputs to primate single-unit recordings from (Freeman et al., 2013), using 315 texture images and data from 102 V1 and 103 V2 neurons. For the V4 and IT benchmarks, 2,560 images are used to match model activations to primate Utah array recordings from (Majaj et al., 2015), based on data from 88 V4 and 168 IT electrodes. A linear regression is trained on 90% of the images to correlate model and neural data, with prediction accuracy for the remaining 10% evaluated using Pearson correlation, repeated ten times for cross-validation. The behavioral benchmark assesses model predictions for 240 images against primate behavioral data from (Rajalingham et al., 2018) using a logistic classifier trained on 2,160 labeled images. Pearson correlation is used to measure the similarity in confusion patterns between model predictions and primate responses.
All benchmark scores are normalized to their respective maximum possible values.
We define the model's alignment score S (and an inverse Misalignment Score L = 1-S) as the average across the V1, V2, V4, IT, and behavioral benchmark scores. Layers are committed to brain regions based on models trained on a full dataset, and applied to all variants trained with subsampled datasets. As we reused the same neural and behavioral data both to select the optimal model layer for readout and to assess the model's alignment, we validated the benchmark results on a private split of each dataset on Brain-Score. We observed an almost perfect correlation between the results on the private and public splits (Appendix C).
Scaling Models and Data. We trained an array of standard models from several architecture families. Specifically, we used ResNet18, 34, 50, 101, 152 from (He et al., 2016); EfficientNet-B0, 1, 2 from (Tan & Le, 2019); Vision Transformer ViTT, S, B, L from (Dosovitskiy et al., 2021); ConvNeXtT, S, B, L from (Liu et al., 2022b); CORnet-S from (Kubilius et al., 2019); and AlexNet from (Krizhevsky et al., 2012). We also trained 33 modified versions of ResNet18: 22 models obtained by scaling the network width from 1/16 to 4 times the original size, and 11 models derived by adjusting the depth. Similarly, we trained four additional ConvNeXt and ViT models by scaling the width of the ConvNeXt-T and ViT-S architectures.
For our experiments, we selected two image classification datasets: ImageNet (Deng et al., 2009) and EcoSet (Mehrer et al., 2021). ImageNet, with millions of labeled images across 1,000 categories, has long been a benchmark in computer vision, designed to challenge and evaluate automated visual object recognition systems. On the other hand, EcoSet is a more recent dataset, designed to provide an ecologically valid representation of human-relevant objects. It contains over 1.5 million images spanning 565 basic-level categories, curated to better reflect the natural distribution of objects in the real world, aligning with human perceptual and cognitive experiences.
To create subsets of ImageNet and EcoSet, we sampled d ∈ 1, 3, 10, 30, 100, 300 images per category. For d ∈ 1, 10, 100, we repeated the runs with three random seeds to ensure robustness. For ConvNeXts (Liu et al., 2022b) and ViTs (Touvron et al., 2022), we used the training recipes developed by the original model authors. The remaining models were trained for 100 epochs using a minibatch size of 512. We employed a stochastic gradient descent (SGD) optimizer with a cosine decaying learning rate schedule, starting with a peak learning rate of 0.1 and incorporating a linear warm-up phase spanning five epochs. We maintained the momentum at 0.9 and applied a weight decay of 10 -4 . Cross-entropy loss was used as the minimization objective. We utilized standard ImageNet data augmentations, specifically random resized cropping and horizontal flipping.
Scaling Power-Law Curves. Following previous work on scaling laws (Zhai et al., 2022;Hoffmann et al., 2022;Besiroglu et al., 2024), we fit power law functions in the form
L = E + AX -α(1)
on the data where L is the misalignment score, and X is an independent variable, such as the number of samples seen (D), number of parameters (N ), and the total training floating point operations (FLOPs) (C). Coefficients E, A, and α are found by minimizing
min a,e,α i∈[#Runs] Huber δ (LSE(a -α log X i , e) -log L i )(2)
where E = exp(e), A = exp(a) and LSE is the logsum-exp operator. We solve Eq. 1 using BFGS minimizer with δ = 1e -3, and use a grid of initialiations as follows: e ∈ {-1, -0.5, . . . , 1}, a ∈ {0, 5, . . . , 25}, α ∈ {0, 0.5, . . . , 2}.
To capture the slow initial increase in benchmark scores of modern architectures like ConvNeXt and ViT models in the low-data regime, we introduce an additional parameter λ to Eq. 2. This parameter allows the fitted curve to saturate at lower scales, better reflecting the observed performance of these models under limited data conditions:
L = E + A X + 10 λ -α(3)
We minimize the modified equation as before, using λ ∈ 0, 0.5, 1.0, 1.5, 2.0. To fit the curve described by Eq. 3, we utilize all data points from the ConvNeXt and ViT models. For fitting the remaining curves, we select ConvNeXt and ViT runs that were trained on datasets with either 300 samples per class or the full dataset. This approach ensures that the fitted curves accurately represent the scaling behavior of these architectures across different data regimes.
Furthermore, we would like to describe the misalignment (L) as a function of both the model and data size (N , D) and predict optimal allocations N * and D * by solving
(N * , D * ) = arg min N,D L(N, D), FLOPs(N D) = C (4)
In that regard, following (Hoffmann et al., 2022;Besiroglu et al., 2024) we fit a parametric function of the form
L(N, D) = E + A N α + B D β(5)
where the loss ( L) is a function of parameter count (N ) and number of samples seen (D). In Eq. 5, the first term represents the loss in an ideal data generation scenario (entropy), the second and the third terms reflect the under-performance of a model due to limitations in parameter and data size (Hoffmann et al., 2022;Muennighoff et al., 2023). Following the example of Hoffmann et al. (2022), we learn variables {E, A, α, B, β} that characterizes misalignment by solving arg min e, a, α, b, β i∈[#Runs]
Huber δ log L i - LSE a -α log N i , b -β log D i , e(6)
with δ = 10 -3 and E = exp(e), A = exp(a) B = exp(b).
Initialiations of b and β follow a and α, respectively.
Both Kaplan et al. (2020); Hoffmann et al. (2022) assume that compute follows the relationship C(N, D) ≈ 6N D to predict the optimal allocation of compute (C) to N and D using a set of equations with the learned variables mentioned above:
N * (C) = G(C/6) a , D * (C) = G -1 (C/6) b
where
a ′ = β α + β , b ′ = α α + β , G = αA βB 1 α+β (7)
However, we observe that C(N, D) ≈ 6N D does not hold with different architectures, and various CNN families have a slightly different relationship of C, N , and D. As such, we assume a power-law relationship of the form
C(N, D) = m(N D) n(8)
where we fit m and n via linear regression of C and N D in log-log scale. Then, the updated equations governing the optimal allocation becomes
N * (C) = G(C/m) a ′ /n , D * (C) = G -1 (C/m) b ′ /n (9)
where a ′ , b ′ , and G are calculated as before.
To evaluate the uncertainty of our model fits, we performed bootstrapping with 1,000 resamples. We compute 95% confidence intervals for each point along the fitted curves based on the variability observed across the bootstrapped estimates.
Finally, to avoid large constants during curve fitting, we rescale the variables C, N , and D by setting C = C/10 13 , Ñ = N/10 5 , and D = D/10 4 .
this section cite: ['b57', 'b34', 'b19', 'b46', 'b55', 'b25', 'b62', 'b18', 'b38', 'b37', 'b16', 'b64', 'b72', 'b28', 'b5', 'b28', 'b5', 'b28', 'b50', 'b28', 'b31', 'b28']

Section: Results

this section cite: []

Section: Scaling drives behavioral alignment, but saturates for neural alignment
Our experiments show a clear and consistent improvement in behavioral alignment as both model size and training dataset size increase. Fig 1.b illustrates this trend across different architectures and scaling axes. The curve S = 1 -1.4 C-0.06 converges to perfect alignment score of 1 in the limit of C.
In contrast to behavioral alignment, neural alignment with specific brain regions demonstrated saturation as training compute scaled up in size. The curve represented by the formula S = 0.48 -0.55 C-0.16 represents a saturation at 0.48. The diminishing returns in neural alignment imply that merely scaling up models and data is insufficient to achieve better alignment with higher-level neural representations.
this section cite: []

Section: Architectural Inductive Bias Influences Alignment and Scaling Dynamics
Experimental results indicate that modern architectures, such as ConvNeXt and Vision Transformers (ViTs), exhibit poorer neural alignment compared to models like ResNets and EfficientNets in low data regime. ResNets and Efficient-Nets, which have stronger inductive biases due to their fully convolutional structures, demonstrate high neural alignment even at initialization. In Fig. 2, alignment score of ResNets and EfficientNets increase steadily with additional compute in the form of training samples, however ConvNeXt and ViT requires more compute in order to start rising.
This difference in initial alignment also affects how the scaling laws evolve for each architecture. Models with weaker inductive biases require more extensive scaling-specifcally in terms of training data-to achieve levels of neural alignment comparable to those with stronger inductive biases. Consequently, the scaling curves for ConvNeXt and ViT models develop differently, highlighting that architectural choices not only impact baseline alignment but also influence the efficiency of scaling strategies.
Fig. 3b highlights that architectural priors critically shape alignment dynamics, particularly in low-data settings. CORnet models, which incorporate recurrence, achieve relatively high alignment early in training-outperforming both convolutional and transformer-based models under limited supervision. Yet as training data increases, this initial advantage wanes, and alignment scores across architectures begin to converge. This suggests that while certain inductive biases offer sample efficiency, their long-term benefits may be outpaced by deeper or more flexible architectures given sufficient data. Overall, these findings emphasize that strong inductive biases-such as convolution and recurrence-facilitate better alignment when data is limited, whereas extensive task-driven optimization on larger datasets eventually mitigates differences across architectures.
this section cite: []

Section: More Data Is Better Than More Parameters
Our analysis reveals that increasing the size of the training dataset has a more significant impact on improving brain alignment than simply enlarging the number of model parameters. While both strategies lead to performance enhancements, the benefits from data scaling exhibit less severe diminishing returns compared to model scaling. Specifically, models trained on larger datasets consistently demonstrate superior neural and behavioral alignment with the primate ventral visual stream, following a predictable power-law relationship.
In contrast, expanding the model size without proportionally increasing the training data results in steeper diminishing returns in alignment performance. Larger models rapidly reach a point where additional parameters do not translate into meaningful improvements. This indicates that scaling training datasets overall improves brain alignment better than models scaling. Furthermore, Fig. 4b demonstrates that larger models of the same architecture family require much more samples to achieve the same level of alignment.
To quantitatively capture the joint interaction between data and model scaling, we fitted a parametric curve based on Eq.5, as shown in Fig. 4a. This curve effectively models how compute (C), dataset size (D), and model size (N ) collectively influence brain alignment. Utilizing the parametric relationships described in Eq. 9, we estimate that additional compute should be allocated following the scaling laws D ≈ C 0.7 and N ≈ C 0.3 . These exponents indicate that, for optimal brain alignment, computational resources should be predominantly invested in increasing the dataset size rather than the model size.
this section cite: []

Section: Ordered effect of scale on alignment
Our study reveals a graded effect of scaling on alignment across the cortical hierarchy of the primate visual system. Specifically, we observe that the benefits of increased training compute-achieved through larger datasets and more complex models-vary systematically among different brain regions, reflecting their position in the visual processing pathway. Fig. 5.a illustrates the alignment as a function of training compute across various brain regions. We categorized the models into two groups based on their architectural inductive biases. Group 1 includes most models with strong inductive biases, such as ResNets and EfficientNets. These models start with higher neural alignment scores even at initialization due to their fully convolutional architectures. Group 2 consists of models with weaker inductive biases, specifically ConvNeXt and Vision Transformers (ViTs). These models exhibit lower neural alignment in the low-data regime and require more compute to achieve similar alignment levels.
To quantify the impact of scaling on each brain region, we define the alignment gain per region as A10 α where A and α are parameters of Eq. 2.
Our findings indicate that higher regions in the cortical hierarchy show greater benefits from increased compute. per region, highlighting how higher cortical areas benefit more from scaling efforts. This ordered effect suggests that regions higher up in the visual hierarchy, such as the Inferior Temporal (IT) cortex and behavioral outputs, gain more substantially from additional data and increased model complexity. In contrast, early visual areas like V1 and V2 exhibit smaller alignment gains with increased compute, indicating a potential saturation effect.
this section cite: []

Section: Discussion
We establish scaling laws governing the effect of model and dataset scale on behavioral and brain alignment with the primate visual ventral stream. While scale is a necessary component for all brain-like models, model architectures with priors such as convolutions, and datasets with highquality images are more sample efficient, leading to alignment with smaller compute requirements. Scale especially improves alignment with higher-level visual regions, but brain alignment saturates across all conditions tested here whereas behavioral alignment continuously improves with increased scale.
We find a saturation of neural alignment under current modeling approaches, consistent with trends reported in prior work (Linsley et al., 2023;Conwell et al., 2024;Muttenthaler et al., 2023). Critically, our results reveal a disconnect between neural and behavioral alignment: while behavioral . alignment continues to improve with increased scale, neural alignment plateaus. By quantifying scaling laws across model families and data regimes, we show that improvements in brain alignment are more efficiently achieved by increasing dataset size rather than model parameters. These findings offer concrete guidance for developing brain-like models more effectively, emphasizing the importance of dataset diversity and biologically inspired architectural priors over brute-force model scaling.
Dissociation of behavioral and neural alignment. Our findings reveal a dissociation between behavioral and neural alignment as models are scaled with more parameters and larger datasets. While behavioral alignment continues to improve consistently with increased model parameters and training data -exhibiting a strong power-law relationshipneural alignment reaches a saturation point beyond which additional scaling yields minimal gains. This divergence suggests that behavioral alignment benefits more substantially from scaling efforts, whereas neural alignment may require alternative approaches beyond merely increasing model size and data volume to achieve further improvements.
This disparity is further highlighted by the correlation between task performance and alignment depicted in Figure 6. Behavioral alignment closely tracks validation accuracy, improving hand-in-hand as models become more accurate. Consistent with prior work (Schrimpf et al., 2018;Linsley et al., 2023), neural alignment eventually saturates, indicating that factors other than task performance influence neural alignment.
Generalization Beyond Supervised Training. We assessed whether alternative training paradigms can overcome the limitations observed in neural alignment under supervised learning. Figure 7a illustrates the scaling of alignment as a function of compute spent during self-supervised training of ResNet models using SimCLR (Chen et al., 2020) on ImageNet. The results confirm the trends observed in supervised training: behavioral alignment continues to improve with increased compute, following a strong power-law relationship, while neural alignment approaches a saturation point. This consistency suggests that the saturation in neural alignment is not exclusive to supervised learning but may be inherent to the models or datasets employed.
The region-specific breakdown (as illustrated in Supp. Fig. S6) further reinforces this observation. Even in a selfsupervised learning context, higher-level visual areas like IT and behavioral outputs demonstrate more pronounced improvements with increased compute, while early visual areas like V1 and V2 show minimal gains. This suggests that the hierarchical nature of neural alignment is a fundamental characteristic that transcends specific training methods.
Additionally, we explored the impact of adversarial finetuning on alignment performance. In Figure 7b, ResNet models trained on subsets of ImageNet were fine-tuned adversarially for 10 epochs using the Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015;Wong et al., 2020). Importantly, the scaling curves were estimated solely from the non-adversarial runs, yet the adversarially finetuned models exhibited improvements along these existing scaling curves. This indicates that adversarial training can enhance alignment without deviating from the established scaling behavior.
this section cite: ['b41', 'b14', 'b51', 'b57', 'b41', 'b11', 'b22', 'b69']

Section: References
Ref_id:b0 Title: Scaling laws for generative mixed-modal language models Year: (2023-07)
Ref_id:b1 Title: Scaling laws for language encoding models in fmri Year: (2023)
Ref_id:b2 Title: A unified, scalable framework for neural population decoding Year: (2023)
Ref_id:b3 Title: Explaining scaling laws of neural network generalization Year: (2022)
Ref_id:b4 Title: Brain decoding: toward real-time reconstruction of visual perception Year: (2023)
Ref_id:b5 Title: Chinchilla scaling: A replication attempt Year: (2024)
Ref_id:b6 Title: Language models are few-shot learners Year: ()
Ref_id:b7 Title: Albumentations: Fast and flexible image augmentations Year: (2020)
Ref_id:b8 Title: Deep convolutional models improve predictions of macaque v1 responses to natural images Year: (2019-04)
Ref_id:b9 Title: A foundation model for brain activity recordings Year: (2024)
Ref_id:b10 Title: Emerging properties in self-supervised vision transformers Year: ()
Ref_id:b11 Title: A simple framework for contrastive learning of visual representations Year: (2020-07)
Ref_id:b12 Title: Reproducible scaling laws for contrastive language-image learning Year: (2023-06)
Ref_id:b13 Title:  Year: ()
Ref_id:b14 Title: A large-scale examination of inductive biases shaping high-level visual representation in brains and machines Year: (2024-10)
Ref_id:b15 Title: Simulating a primary visual cortex at the front of cnns improves robustness to image perturbations Year: (2020-07)
Ref_id:b16 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b17 Title: Dual attention vision transformers Year: (2022)
Ref_id:b18 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b19 Title: A functional and perceptual signature of the second visual area in primates Year: (2013)
Ref_id:b20 Title: Wiring up vision: Minimizing supervised synaptic up-dates needed to produce a primate ventral stream Year: ()
Ref_id:b21 Title: Separate visual pathways for perception and action Year: (1992)
Ref_id:b22 Title: Explaining and harnessing adversarial examples Year: (2015)
Ref_id:b23 Title: A vision transformer in convnet's clothing for faster inference Year: (2021-10)
Ref_id:b24 Title: The lateral occipital complex and its role in object recognition Year: (2001)
Ref_id:b25 Title: Deep Residual Learning for Image Recognition Year: (2016-06)
Ref_id:b26 Title: Scaling laws for autoregressive generative modeling Year: (2020)
Ref_id:b27 Title:  Year: (2017)
Ref_id:b28 Title: An empirical analysis of compute-optimal large language model training Year: (2022)
Ref_id:b29 Title: Scaling laws for downstream task performance of large language models Year: (2024)
Ref_id:b30 Title: Principles of Neural Science Year: (2000)
Ref_id:b31 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b32 Title: Deep supervised, but not unsupervised, models may explain it cortical representation Year: (2014-11)
Ref_id:b33 Title: Recurrence is required to capture the representational dynamics of the human visual system Year: (2019-10)
Ref_id:b34 Title: Torchattacks: A pytorch repository for adversarial attacks Year: (2020)
Ref_id:b35 Title: Matching categorical object representations in inferior temporal cortex of man and monkey Year: (2008)
Ref_id:b36 Title: Cifar-10 (canadian institute for advanced research Year: ()
Ref_id:b37 Title: Imagenet classification with deep convolutional neural networks Year: (2012)
Ref_id:b38 Title: Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs Year: (2019)
Ref_id:b39 Title: The mnist database Year: (1998)
Ref_id:b40 Title: Webvision database: Visual learning and understanding from web data Year: (2017)
Ref_id:b41 Title: Performanceoptimized deep neural networks are evolving into worse models of inferotemporal visual cortex Year: (2023)
Ref_id:b42 Title: A convnet for the 2020s Year: ()
Ref_id:b43 Title: A convnet for the 2020s Year: (2022)
Ref_id:b44 Title:  Year: (2007)
Ref_id:b45 Title: Torchvision: Pytorch's computer vision library Year: (2016)
Ref_id:b46 Title: Simple learned weighted sums of inferior temporal neuronal firing rates accurately predict human core object recognition performance Year: (2015)
Ref_id:b47 Title: The topography of high-order human object areas Year: (2002)
Ref_id:b48 Title: An ecologically motivated image dataset for deep learning yields better models of human vision Year: ()
Ref_id:b49 Title: Mobilevit: Light-weight, general-purpose, and mobile-friendly vision transformer Year: (2022)
Ref_id:b50 Title: Scaling data-constrained language models Year: (2023)
Ref_id:b51 Title: Human alignment of neural network representations Year: (2023)
Ref_id:b52 Title: Task-driven convolutional recurrent models of the visual system Year: (2018)
Ref_id:b53 Title: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b54 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b55 Title: Large-scale, high-resolution comparison of the core visual object recognition behavior of humans, monkeys, and state-of-the-art deep artificial neural networks Year: (2018)
Ref_id:b56 Title: Imagenet-21k pretraining for the masses Year: (2021)
Ref_id:b57 Title: Brain-score: Which artificial neural network for object recognition is most brainlike? bioRxiv preprint Year: (2018)
Ref_id:b58 Title: Integrative benchmarking to advance neurally mechanistic models of human intelligence Year: (2020)
Ref_id:b59 Title: Laion-400m: Open dataset of clipfiltered 400 million image-text pairs Year: (2021)
Ref_id:b60 Title: LAION-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b61 Title:  Year: (2020)
Ref_id:b62 Title: Rethinking model scaling for convolutional neural networks Year: (2019-06)
Ref_id:b63 Title:  Year: ()
Ref_id:b64 Title: Revenge of the vit Year: (2022)
Ref_id:b65 Title: Multi-axis vision transformer. ECCV Year: (2022)
Ref_id:b66 Title: The inaturalist species classification and detection dataset Year: (2018-06)
Ref_id:b67 Title: Fastvit: A fast hybrid vision transformer using structural reparameterization Year: (2023)
Ref_id:b68 Title:  Year: (2019)
Ref_id:b69 Title: Fast is better than free: Revisiting adversarial training Year: (2020)
Ref_id:b70 Title: Performance-optimized hierarchical models predict neural responses in higher visual cortex Year: (2014)
Ref_id:b71 Title: Using goal-driven deep learning models to understand sensory cortex Year: (2016-02)
Ref_id:b72 Title: Scaling vision transformers Year: (2022-06)
Ref_id:b73 Title: Places: A 10 million image database for scene recognition Year: (2017)
Ref_id:b74 Title: Unsupervised neural network models of the ventral visual stream Year: ()
