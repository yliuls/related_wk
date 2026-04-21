Title: Conditional Representation Learning for Customized Tasks
Abstract: Conventional representation learning methods learn a universal representation that primarily captures dominant semantics, which may not always align with customized downstream tasks. For instance, in animal habitat analysis, researchers prioritize scene-related features, whereas universal embeddings emphasize categorical semantics, leading to suboptimal results. As a solution, existing approaches resort to supervised fine-tuning, which however incurs high computational and annotation costs. In this paper, we propose Conditional Representation Learning (CRL), aiming to extract representations tailored to arbitrary user-specified criteria. Specifically, we reveal that the semantics of a space are determined by its basis, thereby enabling a set of descriptive words to approximate the basis for a customized feature space. Building upon this insight, given a user-specified criterion, CRL first employs a large language model (LLM) to generate descriptive texts to construct the semantic basis, then projects the image representation into this conditional feature space leveraging a vision-language model (VLM). The conditional representation better captures semantics for the specific criterion, which could be utilized for multiple customized tasks. Extensive experiments on classification and retrieval tasks demonstrate the superiority and generality of the proposed CRL. The code is available at XLearning-SCU/2025-NeurIPS-CRL.

Section: Introduction
Representation learning aims at extracting meaningful patterns from raw data to create representations that are easier to understand and process. Its impact spans a wide range of downstream tasks, such as classification and retrieval. In classification, representation learning enhances the discrimination and linear separability of features, significantly improving performance across diverse data modalities, including images [29], text [41], and video [59]. Similarly, in retrieval tasks, representation learning underpins efficient and accurate query-to-item matching, as evidenced by developments in image retrieval [18] and cross-modal retrieval [50]. In recent years, driven by self-supervision techniques such as contrastive learning [6,23,19,7,71] and mask prediction [9,22,73,61], representation learning methods have undergone rapid advancements, leading to substantial performance improvements across various fields, including graph [42], point-cloud [64], and skeleton [65].
Though remarkable progress has been made, a crucial yet often overlooked question remains: What underlying criterion governs the learned representation? In fact, most existing representation learning methods inherently impose an implicit criterion. Previous research [56] has demonstrated that representations learned by existing approaches exhibit a strong bias toward a single dominant aspect, typically "shape" or "category"-as these are the most salient features in many datasets. This inherent bias causes models to prioritize specific attributes while disregarding other potentially informative features, such as "texture" and "color". Consequently, the resulting universal embeddings predominantly capture a single prominent criterion, leading to sub-optimal performance in downstream tasks that rely on alternative perspectives. As illustrated in Fig. 1, existing methods primarily identify the elephant "category", which is insufficient for customized tasks like population monitoring or habitat analysis. In comparison, our CRL could adaptively capture "count" and "scene" semantics, demonstrating broader generality. This narrow focus ultimately constrains the generalization capability of representation learning methods, underscoring the need for more adaptable and criterion-aware approaches.
Figure 1: Existing conventional representation learning learns a universal representation that prioritizes the dominant semantics while overlooking other meaningful features, limiting their adaptability to customized tasks. In contrast, our proposed conditional representation learning (CRL) extracts representations conditioned on specific criteria, enhancing its applicability.
To transform the image representation to align with specific criteria, a straightforward approach would be supervised fine-tuning [16,35], where models are retrained using labeled data that adhere to the given criterion. However, such a paradigm is not always practical due to the substantial annotation effort required. In the unsupervised scenario, where only images and a user-specified criterion are provided, a feasible solution is to query visual question answering (VQA) models [52,69,31] to extract relevant attributes from each image. However, this approach is computationally expensive and requires additional representation learning steps for the generated textual responses. With these considerations, an efficient way of learning the criterion-oriented image representation is highly expected.
In recent years, researchers have also been exploring computationally efficient approaches to learning useful representations.
Goalconditioned works [46,43] target learning representations that meet the required outcomes or goal states. An area that is more closely related to our work is task-conditioned works [70,2], which aim to learns representations that reveal the underlying correlations among different tasks. For example, taskonomy [70] computes the optimal transfer learning paths among tasks (point matching, reshading, etc.) to minimize the amount of required annotation. While there are certain commonalities between these works and ours, they haven't investigated the relationship between criteria and representations.
In this paper, we introduce Conditional Representation Learning (CRL), a novel approach that adapts the image representation to any user-specified criterion. Unlike conventional representation learning methods, which primarily focus on general-purpose feature extraction, CRL constructs a customized feature space by leveraging the concept of basis transformation. The key insight behind CRL is that the semantics of a feature space are determined by its basis. For example, in a three-dimensional Cartesian coordinate system, the x, y, and z unit vectors define the space, allowing for the decomposition of any vector. Similarly, in color theory, red, green, and blue serve as the basis for the trichromatic color space, enabling the synthesis of all perceivable hues. Extending this idea to high-dimensional semantic representations, a well-chosen set of descriptive words can form a basis for a customized feature space, which captures specific semantic properties aligned with a user-defined criterion. Building on this perspective, CRL formulates conditional representation learning as a basis transformation process. Given a user-specified criterion, we first employ a large language model (LLM) to generate a set of descriptive texts that serve as a semantic basis, spanning the relevant feature space. We then utilize a vision-language model (VLM) to encode both the generated texts and the images, obtaining their representations respectively. Finally, we project the image representation into the conditional feature space with the textual representation acting as a basis. The transformed conditional representation would be more expressive under the specified criterion, which could be utilized for downstream tasks that require customized semantics.
The major contributions of this paper could be summarized as follows:
• Different from conventional representation learning that primarily captures a single dominant semantics, we propose conditional representation learning (CRL), which enables learning representations tailored to arbitrary user-specified criteria.
• We formulate CRL as a basis transformation process, offering a computationally efficient and highly generalizable solution. It eliminates the reliance on supervised fine-tuning while substantially improving the applicability and interpretability of the learned representation.
• Extensive experiments validate the effectiveness and generality of CRL in customized classification and retrieval, showcasing its superiority in seamlessly adapting to varying criteria and tasks.
2 Related Work
this section cite: ['b28', 'b40', 'b58', 'b17', 'b49', 'b5', 'b22', 'b18', 'b6', 'b70', 'b8', 'b21', 'b72', 'b60', 'b41', 'b63', 'b64', 'b55', 'b15', 'b34', 'b51', 'b68', 'b30', 'b45', 'b42', 'b69', 'b1', 'b69']

Section: Representaion Learning
Representation learning aims to extract informative features from raw data, facilitating downstream tasks like classification and retrieval. As a classic method, autoencoder [24] learns compact representations through unsupervised reconstruction. Building upon it, denoising autoencoders [58] and variational autoencoders [27] have been proposed to enhance the robustness and structure of the learned latent representations. In the past few years, the field has further evolved with self-supervised learning techniques, which encourage models to learn semantical features by addressing pretext tasks such as patch and rotation prediction [10,17], solving jigsaw puzzles [44], and colorization [72]. A notable advancement in this direction is contrastive learning, exemplified by methods like SimCLR [6] and MoCo [23], which leverage instance discrimination to learn discriminative representations. More recently, the emergence of large language models (LLMs) such as GPT [5] and vision-language models (VLMs) like CLIP [48] has introduced a more interpretable approach for representation learning. A series of works [74,15,47,38,21] have then researched using CLIP to improve zero-shot or few-shot image classification performance. By analyzing the Vision Transformer [13] architecture of CLIP, studies such as Text-Span [14] have shed light on the underlying semantics captured by individual attention heads. Leveraging the strengths of LLMs and VLMs, approaches like VCD [40], LaBo [66] and LM4CV [63] have demonstrated that interpretable representation learning can achieve performance on par with black-box methods in downstream image classification.
Despite significant progress, most existing representation learning approaches remain centered on a single criterion, typically "category" or "shape", while overlooking other meaningful semantic dimensions. This narrow focus limits the generalizability of learned representations, often necessitating extensive supervised fine-tuning when adapting to tasks that depend on alternative semantic cues. To address this limitation, we advocate for a paradigm shift from universal to conditional representation learning, an underexplored yet promising direction. Specifically, our approach first constructs a semantic basis composed of descriptive texts aligned with a user-specified criterion. Leveraging this customized basis, we transform the image representation to enable conditional adaptation, enhancing the flexibility and applicability of learned features without additional laborious fine-tuning.
this section cite: ['b23', 'b57', 'b26', 'b9', 'b16', 'b43', 'b71', 'b5', 'b22', 'b4', 'b47', 'b73', 'b14', 'b46', 'b37', 'b20', 'b12', 'b13', 'b39', 'b65', 'b62']

Section: Conditional Similarity
Conditional similarity refers to the similarity between samples based on specific criteria. This concept was first formalized by CSN [57], which learns multiple feature spaces to enable customized fashion item retrieval under different criteria. With the advent of representation learning, a series of tailored fashion retrieval approaches have been developed [39,11,12], significantly improving the retrieval performance. Recently, the idea of conditional similarity has gained traction in the clustering domain [37]. Driven by the powerful language processing capabilities of large-scale pre-trained models, IC|TC [28] pioneers the concept of customized clustering by directly querying VLMs and LLMs to obtain clustering results based on specific criteria. However, this approach incurs high computational costs. To address this limitation, Multi-Map [68] introduces a more cost-efficient alternative, injecting customized semantics from VLM and the LLM to guide the clustering process.
Despite the success of existing methods, they are all delicately designed for specific tasks, limiting their generalization ability to other domains. In contrast, we propose CRL, a simple yet effective method for learning general conditional representation, which could seamlessly adapt to diverse customized tasks.
this section cite: ['b56', 'b38', 'b10', 'b11', 'b36', 'b27', 'b67']

Section: Method
This section details the proposed Conditional Representation Learning (CRL) framework, which consists of basis construction and representation transformation. As depicted in Fig. 2, given a user-specified criterion, CRL first constructs a customized basis by querying an LLM about descriptive words. Subsequently, CRL computes the conditional image representation through a basis transformation operation.
this section cite: []

Section: Basis Construction
Mathematically, a basis refers to a set of linearly independent vectorsfoot_0 that span the entire space. For example, in the three-dimensional Cartesian coordinate system, vectors (1, 0, 0), (0, 1, 0), and (0, 0, 1), which denote the x, y, and z axes, form a basis since any vector in the space can be expressed as a linear combination of these three vectors. Analogously, in the trichromatic color space, "red", "green", and "blue" form a basis as they could compose all possible hues. From a broader view, a set of descriptive words related to the user-specified criterion, that spans the customized feature space, intrinsically acts as the basis as well.
To construct the basis under the specific criterion C, we query an LLM to generate the related descriptive texts W via W = LLM(P 1 , C),
where P 1 denotes the LLM prompt template. As a general solution, we use the following prompt for all customized tasks:
Generate common expressions to describe the C, as many as possible. where C is replaced with the user-specified criterion words such as "color", "shape", "texture", etc. Notably, we incorporate additional instructions to encourage the LLM to produce formatted, comprehensive texts and avoid repetitions, which are detailed in the Appendix.
Given the prompted query, the LLM would generate texts W semantically correlated with the user-specified criterion, transforming the abstract criterion into a concrete textual basis. Once the descriptive texts W are obtained, we feed them into a VLM text encoder VLM text to compute their normalized representation T via
T = VLM text (P 2 , C, W ),(2)
where P 2 denotes the VLM prompt constructed as follows:
Objects with the C of W . It is worth noting that, when prior knowledge about the dataset is available, the word "Objects" could be replaced by more specific descriptions. The complete prompts used for all customized tasks in this paper, as well as the LLM responses, are attached in the Appendix.
As previously discussed, the text representation T could act as the basis spanning the customized feature space. Remarkably, compared with the basis of the classic universal feature space, the constructed basis T enjoys superior interpretability where each dimension has an explicit physical meaning.
this section cite: []

Section: Representation Transformation
After acquiring the text basis, we leverage it to transform the universal representation into the conditional representation, by projecting data into the constructed customized feature space.
Figure 2: The overall framework of the proposed CRL. Given images and a user-specified criterion (e.g., "color"), CRL first queries an LLM to generate descriptive texts semantically related to the criterion (e.g., "red", "green" and "blue"). Then, CRL encodes the generated texts and original images through a VLM. Subsequently, CRL projects the original image representation (e.g., dominated by "shape") into the conditional feature space spanned by the textual representation. The transformed conditional representation would be more expressive under the specified criterion and enjoy superior interpretability, facilitating customized downstream tasks.
To be specific, we first feed the images X into the VLM image encoder VLM image to obtain their normalized representation I via
I = VLM image (X).(3)
Subsequently, we transform the image representation by projecting it to the customized space spanned by text basis T, namely, R = IT ⊤ , (4) where R denotes the transformed conditional representation. The validity of this transformation exploits the alignment between image and text modalities in the VLM's feature space. The conditional representation R emphasizes the attributes related to the user-specified criterion, and thus is more favorable in customized tasks.
The complete process of our CRL is outlined in Algorithm 1. To deliver a more intuitive understanding of CRL's working mechanism and underlying rationale, we provide an example about learning a color-conditioned representation as illustrated in Fig. 2.
Consider the customized clustering task, which aims at grouping images based on their colors. The original image representation is dominated by the most significant shape information, which is suboptimal for color-based grouping. To build a customized feature space focusing on colors, we first query an LLM about the common colors. Supposing the LLM outputs descriptive texts W = {"red", "green", "blue"}, we calculate the text basis as
T = [t ⊤ 1 , t ⊤ 2 , t ⊤ 3 ] ⊤ ,(5)
where {t 1 , t 2 , t 3 } denote the rows of T, corresponding to the representations of "red", "green", and "blue".
Then we project the k-th original image representation i k to conditional representation r k via
r k = i k T ⊤ = [i k • t 1 , i k • t 2 , i k • t 3 ].(6)
As shown in Eq. ( 6), the transformed conditional representation of the k-th image refers to the projection of its original representation onto the text basis T. Consequently, the three elements of r k correspond to its degree of "red", "green", and "blue", respectively. In other words, r k is more expressive than i k under the "color" criterion, leading to superior performance on the customized clustering task.
this section cite: []

Section: Experiments
To assess the conditional representation learning performance of the proposed CRL, we apply it to two classic downstream tasks, including classification and retrieval. Notably, different from standard  4), which could be then utilized for various customized tasks. representation learning, CRL focuses on learning conditional representation, and thus the downstream classification and retrieval are based on various customized criteria. After that, parameter analysis is conducted to investigate the robustness of CRL.
this section cite: []

Section: Customized Classification
As shown in Fig. 3, customized classification aims to classify samples into different semantic categories under the specific criterion, which includes two subtasks, i.e., supervised few-shot learning and unsupervised clustering.
this section cite: []

Section: Customized Few-shot Learning
Dataset. For this task, we utilize Clevr4-10k [56] and Cards [67] as benchmark datasets. Clevr4-10k is a synthetic dataset consisting of 10, 531 samples and 4 distinct data partition criteria, categorized by "shape", "texture", "color", and "count", respectively. Cards is a poker card dataset comprising 8, 029 samples, organized according to 2 criteria, i.e., "number" and "suit".
this section cite: ['b55', 'b66']

Section: Setup.
For fair comparisons, we adopt the logistic regression function from the scikit-learn package [45] to perform few-shot learning, under the number of shots 1, 5, 10 per class, respectively. To alleviate the influence of randomness, we stochastically select the training data 20 times for each shot and report the mean result. As for the backbone, we adopt ViT-B/32 pre-trained on CLIP, keeping the same with Section 4.1.2.
this section cite: ['b44']

Section: Metric.
For the task of customized few-shot learning, we adopt accuracy (ACC) as the evaluation metric.
Baseline. We conduct comparisons between proposed CRL and image representations of CLIP [48], ALIGN [25] and MetaCLIP [62] across six semantic criteria.
Performance. As illustrated in Table . 1, CRL achieves a noticeable improvement over CLIP, ALIGN and MetaCLIP across most experimental settings, with a mean accuracy gain of nearly 10%.
Particularly, CRL gains significant improvements when the target criterion differs substantially from the originally dominant one, such as 'color' (nearly +40% at 1-shot). The consistent performance advantage indicates that CRL's representation exhibits a better generality under multiple criteria.
this section cite: ['b47', 'b24', 'b61']

Section: Setup.
We directly conduct k-means on the representations obtained by CRL to get the clustering. Keeping the same as the customized few-shot learning, we also perform k-means 20 times and report the average clustering result. As for the backbone, we follow the previous method [68], adopting ViT-B/32 pre-trained on CLIP.
this section cite: ['b67']

Section: Metric.
Three widely used clustering metrics, namely Normalized Mutual Information (NMI), Accuracy (ACC), and Adjusted Rand Index (ARI), are used for evaluation. Higher scores indicate better clustering results.  Baseline. We first compare CRL with two traditional clustering methods, CC [32] and SCAN [54]. Furthermore, we incorporate Multi-Map [68], a customized clustering approach that leverages the CLIP model, into the comparison. Additionally, we report the performance of k-means clustering applied to the image representation of CLIP, ALIGN and MetaCLIP, to provide an intuitive baseline analysis.
Performance.As shown in Table . 2, CRL gains consistent performance improvement compared with the original CLIP, ALIGN and MetaCLIP. In particular, CRL obtains an ACC boost of CLIP over 75% on the color criterion. This improvement can be better visualized by T-SNE [53], as shown in the Appendix. Though traditional clustering methods exhibit some superiority on the "shape" criterion, CRL achieves consistently better results on other criteria. This implies that traditional clustering methods have a strong bias towards a single criterion, yet lack the flexibility and capability to cluster data based on other meaningful criteria.
this section cite: ['b31', 'b53', 'b67', 'b52']

Section: Customized Retrieval
For customized retrieval, we also conduct experiments on its two subtasks, namely, customized similarity retrieval and customized fashion retrieval. Given a query image and a condition object, customized similarity retrieval aims to retrieve the most conditionally similar image from candidates, as illustrated in Fig. 4. As shown in Fig. 5, customized fashion retrieval searches all candidate images of fashion items, which share the same value as the query image under the specific criterion.
this section cite: []

Section: Customized Similarity Retrieval
Dataset. We adopt GeneCIS [55] as the benchmark for this task, which comprises two settings. As shown in Fig. 4, (a) "Focus" setting aims to retrieve the candidate that contains both the same scene (e.g., living room) and the condition object (e.g., table) as the query image. (b) In contrast, the "Change" setting requires the target image to maintain the same scene (e.g., railway) as the query image while including the condition object (e.g., tree) that is absent in the query image.
Setup. This benchmark involves two factors, namely, object (text condition) and scene (query image).
To employ CRL, we ask the LLM for the common scenes, obtaining the conditional representation of the query and candidate images. Then we calculate and sum the similarities of these two factors for retrieval. This operation is detailed in the Appendix. Additionally, we use ViT-B/16 pre-trained on CLIP as the backbone, following the previous work [55].
Metric. The recall rates R@1, R@2, and R@3 serve as the evaluation metrics for this task. Higher recall rates imply better retrieval results.  Baseline. Following [55], we first provide three simple CLIP-only baselines, namely CLIP image , CLIP text and CLIP image+text , detailed in the Appendix. In addition, we include five retrieval baselines Pic2Word [49], SEARLE [4], LinCIR [20], CIG [60] and Combiner [55] for benchmarking. Notably, Combiner leverages the external dataset CC3M [51] to fine-tune the CLIP model. Thus we evaluate the performance of CRL under two scenarios: using the original CLIP weights and using the weights fine-tuned by Combiner.
Performance. As can be observed from Table . 3, CRL demonstrates substantial improvements over the original CLIP baselines, achieving a notable gain of 6.9% in the mean recall. When leveraging fine-tuned CLIP weights, CRL further extends its advantage, surpassing Combiner by 3.7% in mean recall, simultaneously maintaining consistent performance gains across all metrics.
this section cite: ['b54', 'b54', 'b54', 'b48', 'b3', 'b19', 'b59', 'b54', 'b50']

Section: Customized Fashion Retrieval
Dataset. Following previous works [39], we use the category and attribute prediction benchmark of DeepFashion [36] as the evaluation dataset for this task, which consists of 221k / 27k / 27k images for training / validating / testing. This benchmark has 5 criteria, namely, "texture", "fabric", "shape", "part" and "style", with 156, 218, 180, 216, and 230 values, detailed in the Appendix.
this section cite: ['b38', 'b35']

Section: Setup.
We first obtain the embeddings by CRL in a training-free manner. After that, we seamlessly append a two-layer MLP to the embeddings, subsequently training this MLP and the backbone. The training process is detailed in the Appendix. In addition, following previous works, ViT-B/16 is adopted as the backbone for this task.
this section cite: []

Section: Metric.
Following existing works, we use the Mean Average Precision (MAP) as the evaluation metric for the customized fashion retrieval task. Higher MAP values indicate better retrieval results.
Baseline. We first add a Random baseline, which randomly sorts all the candidate images. Moreover, we provide a Triplet baseline, which uses the standard triplet ranking loss [57] to train a joint embedding space. Further, we compare CRL with 5 state-of-the-art fashion retrieval methods, including Triplet [57], CSN [57], ASEN [39], ASEN++ [11] and RPF [12]. Besides, we also provide a CLIP baseline that embeds all the images with the image encoder.
Performance. As shown in Table 4, CRL achieves notable improvements over the CLIP baseline in a training-free manner, with a relative mean MAP gain of 30%.
Once the training is completed, CRL establishes new state-of-the-art performance, surpassing the best competitive method RPF by 10% relativelty in mean MAP. These results further validate CRL's effectiveness in customized tasks and its compatibility with model fine-tuning strategies.
this section cite: ['b56', 'b56', 'b56', 'b38', 'b10', 'b11']

Section: Analysis on Textual Basis
To prove the robustness of CRL, we examine CRL's performance based on the CLIP model for the above-mentioned four customized tasks under varying levels of textual basis. To be specific, we explicitly control the number of generated descriptive texts and report the mean value of each task here, while the complete results can be seen in the Appendix. As Fig. 6 shows, CRL achieves stable performance for different numbers of texts except when the number is too small. In other words, CRL is a robust method for various tasks, as long as there is a reasonable number of descriptive texts to establish the semantical basis for the customized space. More ablation studies can be found in the Appendix.
this section cite: []

Section: Limitation
Based on our observations and experiments, we found that our method suffers from two main limitations. Firstly, despite its generalizability across different criteria, it may not outperform clustering methods like CC and SCAN under the universal criterion "shape". This is likely because these methods employ specially targeted designs for clustering under this criterion. Anyway, we acknowledge that CRL is not optimal on the universal criterion. Secondly, our method only roughly approximates the basis. We've tried various strategies to filter the texts generated by the LLM, but none have proven to be effective across all criteria. Nevertheless, we are confident that better strategies could be devised to acquire the text basis.
this section cite: []

Section: Conclusion
In this paper, we identify a fundamental limitation of existing representation learning methods: they predominantly derive universal embeddings that capture the most salient semantic features, making them suboptimal for customized tasks that prioritize non-dominant semantics. To address this, we propose CRL, a simple yet effective conditional representation learning method that adapts the universal representation to specific criteria through a basis transformation process. In brief, CRL utilizes a large language model (LLM) and a vision-language model (VLM) to generate textual descriptors that are semantically aligned with the user-specified criterion. These descriptors form an interpretable text basis, guiding the transformation of the image representation to enhance its expressiveness under the given criterion. Extensive experiments validate the effectiveness and generality of CRL across diverse tasks and criteria. By shifting the focus toward conditional representation learning, an underexplored yet promising paradigm, we hope this work could spark new insights and foster further research in this direction.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Task2vec: Task embedding for meta-learning Year: (2019)
Ref_id:b2 Title: Claude opus 4 & claude sonnet 4 system card Year: (2025-05)
Ref_id:b3 Title: Zero-shot composed image retrieval with textual inversion Year: (2023)
Ref_id:b4 Title: Language models are few-shot learners Year: (2020)
Ref_id:b5 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b6 Title: Exploring simple siamese representation learning Year: (2021)
Ref_id:b7 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b8 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b9 Title: Unsupervised visual representation learning by context prediction Year: (2015)
Ref_id:b10 Title: Fine-grained fashion similarity prediction by attribute-specific embedding learning Year: (2021)
Ref_id:b11 Title: From region to patch: Attribute-aware foreground-background contrastive learning for fine-grained fashion retrieval Year: (2023)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b13 Title: Interpreting clip's image representation via text-based decomposition Year: (2023)
Ref_id:b14 Title: Clip-adapter: Better vision-language models with feature adapters Year: (2024)
Ref_id:b15 Title: Personalized clustering via targeted representation learning Year: (2024)
Ref_id:b16 Title: Unsupervised representation learning by predicting image rotations Year: (2018)
Ref_id:b17 Title: Deep image retrieval: Learning global representations for image search Year: (2016)
Ref_id:b18 Title: Bootstrap your own latent-a new approach to self-supervised learning Year: (2020)
Ref_id:b19 Title: Language-only training of zero-shot composed image retrieval Year: (2024)
Ref_id:b20 Title: Mmrl: Multi-modal representation learning for vision-language models Year: (2025)
Ref_id:b21 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b22 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b23 Title: Reducing the dimensionality of data with neural networks Year: (2006)
Ref_id:b24 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: (2021)
Ref_id:b25 Title: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning Year: (2017)
Ref_id:b26 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b27 Title: Image clustering conditioned on text criteria Year: (2023)
Ref_id:b28 Title: Learning representations for automatic colorization Year: (2016)
Ref_id:b29 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b30 Title: Oscar: Object-semantics aligned pre-training for vision-language tasks Year: (2020)
Ref_id:b31 Title: Contrastive clustering Year: (2021)
Ref_id:b32 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b33 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b34 Title: Interactive deep clustering via value mining Year: (2025)
Ref_id:b35 Title: Deepfashion: Powering robust clothes recognition and retrieval with rich annotations Year: (2016)
Ref_id:b36 Title: A survey on deep clustering: from the prior perspective Year: (2024)
Ref_id:b37 Title: Swapprompt: Test-time prompt adaptation for vision-language models Year: (2023)
Ref_id:b38 Title: Fine-grained fashion similarity learning by attribute-specific embedding network Year: (2020)
Ref_id:b39 Title: Visual classification via description from large language models Year: (2022)
Ref_id:b40 Title: Distributed representations of words and phrases and their compositionality Year: (2013)
Ref_id:b41 Title: Simple unsupervised graph representation learning Year: (2022)
Ref_id:b42 Title: Review of deep reinforcement learning for robot manipulation Year: (2019)
Ref_id:b43 Title: Unsupervised learning of visual representations by solving jigsaw puzzles Year: (2016)
Ref_id:b44 Title: Scikit-learn: Machine learning in python Year: (2011)
Ref_id:b45 Title: Film: Visual reasoning with a general conditioning layer Year: (2018)
Ref_id:b46 Title: What does a platypus look like? generating customized prompts for zero-shot image classification Year: (2023)
Ref_id:b47 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b48 Title: Pic2word: Mapping pictures to words for zero-shot composed image retrieval Year: (2023)
Ref_id:b49 Title: Adversarial representation learning for text-toimage matching Year: (2019)
Ref_id:b50 Title: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning Year: (2018)
Ref_id:b51 Title: Towards vqa models that can read Year: (2019)
Ref_id:b52 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b53 Title: Scan: Learning to classify images without labels Year: (2020)
Ref_id:b54 Title: Genecis: A benchmark for general conditional image similarity Year: (2023)
Ref_id:b55 Title: No representation rules them all in category discovery Year: (2024)
Ref_id:b56 Title: Conditional similarity networks Year: (2017)
Ref_id:b57 Title: Extracting and composing robust features with denoising autoencoders Year: (2008)
Ref_id:b58 Title: Self-supervised video representation learning by pace prediction Year: (2020)
Ref_id:b59 Title: Generative zero-shot composed image retrieval Year: ()
Ref_id:b60 Title: Masked feature prediction for self-supervised visual pre-training Year: (2022)
Ref_id:b61 Title: Demystifying clip data Year: (2024)
Ref_id:b62 Title: Learning concise and descriptive attributes for visual recognition Year: (2023)
Ref_id:b63 Title: Implicit autoencoder for point-cloud self-supervised representation learning Year: (2023)
Ref_id:b64 Title: View-invariant skeleton action representation learning via motion retargeting Year: (2024)
Ref_id:b65 Title: Language in a bottle: Language model guided concept bottlenecks for interpretable image classification Year: (2023)
Ref_id:b66 Title: Augdmc: Data augmentation guided deep multiple clustering Year: (2023)
Ref_id:b67 Title: Multi-modal proxy learning towards personalized visual multiple clustering Year: (2024)
Ref_id:b68 Title: Deep modular co-attention networks for visual question answering Year: (2019)
Ref_id:b69 Title: Taskonomy: Disentangling task transfer learning Year: (2018)
Ref_id:b70 Title: Barlow twins: Self-supervised learning via redundancy reduction Year: (2021)
Ref_id:b71 Title: Colorful image colorization Year: (2016)
Ref_id:b72 Title: Self-supervised visual representations learning by contrastive mask prediction Year: (2021)
Ref_id:b73 Title: Learning to prompt for vision-language models Year: (2022)
