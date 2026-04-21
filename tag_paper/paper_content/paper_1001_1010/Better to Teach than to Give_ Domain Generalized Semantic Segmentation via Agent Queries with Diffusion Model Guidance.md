Title: Better to Teach than to Give: Domain Generalized Semantic Segmentation via Agent Queries with Diffusion Model Guidance
Abstract: Domain Generalized Semantic Segmentation (DGSS) trains a model on a labeled source domain to generalize to unseen target domains with consistent contextual distribution and varying visual appearance. Most existing methods rely on domain randomization or data generation but struggle to capture the underlying scene distribution, resulting in the loss of useful semantic information. Inspired by the diffusion model's capability to generate diverse variations within a given scene context, we consider harnessing its rich prior knowledge of scene distribution to tackle the challenging DGSS task. In this paper, we propose a novel agent Query-driven learning framework based on Diffusion model guidance for DGSS, named QueryDiff. Our recipe comprises three key ingredients: (1) generating agent queries from segmentation features to aggregate semantic information about instances within the scene; (2) learning the inherent semantic distribution of the scene through agent queries guided by diffusion features; (3) refining segmentation features using optimized agent queries for robust mask predictions. Extensive experiments across various settings demonstrate that our method significantly outperforms previous state-of-the-art methods. Notably, it enhances the model's ability to generalize effectively to extreme domains, such as cubist art styles. Code is available at https:  //github.com/FanLiHub/QueryDiff.

Section: Introduction
Semantic segmentation, a fundamental task in computer vision, has seen significant progress in recent years (Long  et al., 2015; Chen et al., 2017; Fu et al., 2019; Zhong et al.,   1 Northwestern Polytechnical University, China. Correspondence to: Yuelei Xu <xuyuelei@nwpu.edu.cn>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). struggles to cover all variations in the target domain, resulting in limited performance. Our method employs agent queries to learn scene distribution knowledge from the diffusion model, capitalizing on the inherent consistency of this distribution across domains to improve segmentation model generalization. 2020; Cheng et al., 2022). However, it experiences considerable performance degradation when there is a gap between the training data and the test data. This has sparked interest in Domain Generalized Semantic Segmentation (DGSS) (Zhao et al., 2022;Jia et al., 2023;Li et al., 2023). DGSS aims to train a model to learn generalizable features from the source domain, enabling it to perform well on a variety of unseen target domains with consistent contextual distribution but significantly varying visual appearances.
The majority of DGSS methods are built upon Domain Randomization (DR) (Wang et al., 2020;Zhao et al., 2022;Zhong et al., 2022;Jiang et al., 2023), which diversifies source domain images through photometric and geometric transformations. Although somewhat effective, such transformations are constrained by the structure and content of the source data, failing to create new contextual dependencies between instances within the scene. This results in limited diversity that fails to encompass broader scene distributions, i.e., the spatial arrangement and contextual dependencies implicit in the inter-instance relationships (e.g., cars on roads, buildings aligned along streets) that enable the model to move beyond isolated instances and grasp unified objective laws across different scenario domains. As a result, the model remains robust only to specific patterns of variation, resulting in poor generalization performance.
Recently, diffusion models have opened up new avenues for DGSS with remarkable capabilities in capturing complex scene distributions to generate high-quality, realistic samples (Dhariwal & Nichol, 2021;Ho et al., 2020;Rombach et al., 2022). Building on this, several studies have utilized the scene distribution priors of diffusion models to expand the semantic distribution of source domain data. For instance, DIDEX (Niemeijer et al., 2024) employs a diffusion model to generate a pseudo-target domain for domain extension. DatasetDM (Wu et al., 2023a) presents a generic dataset generation model capable of producing diverse images and corresponding annotations using diffusion models. DGInStyle (Jia et al., 2025) crafts diverse task-specific images by sampling the rich prior of a pretrained diffusion model while maintaining strict adherence to semantic layout conditions. While these methods have achieved notable improvements in model performance, expanded data remains inadequate to support the almost infinite distributional variations in target scenarios, as shown in Figure 1 (b). Moreover, expanding the training data substantially incurs significant computational costs and time consumption for the model. Adhering to the principle that teaching a man to fish is better than giving him a fish, a natural curiosity has been raised: Can the powerful scene distribution priors embedded in diffusion models be directly leveraged to enhance segmentation generalization?
A straightforward solution is to harness the diffusion model as a general knowledge extractor to enhance DGSS performance by extracting diffusion features and mapping them to segmentation results. However, this comes with two new challenges: 1) The iterative sample process in diffusion models involves multiple steps of denoising, which is computationally expensive and time-consuming, making it inefficient for perception tasks. 2) Diffusion features encompass not only high-level semantic information but also low-level visual appearance details (e.g., color and texture), which, though crucial for generation tasks, are unnecessary and even detrimental for perception tasks.
To address all these challenges, in this work, we propose QueryDiff, an agent queries-driven framework guided by diffusion models for DGSS, which utilizes the powerful scene distribution priors embedded in diffusion models to enhance semantic segmentation generalization. Specifically, for the first challenge, QueryDiff employs learnable queries to interact with multi-scale image features from the segmentation backbone, aggregating instance-level semantic information to form agent queries that serve as the interface throughout the pipeline, instead of directly relying on diffusion features. Next, guided by the diffusion features, the agent queries learn to construct the semantic distribution of the scene. To address the second challenge, we propose diffusion consistency loss (DCL) that aims to eliminate intricate visual appearance details from the diffusion features, allowing the agent queries to focus on comprehending scene distribution for generalized semantic representations. Ultimately, we use these agent queries to refine the features in the segmentation decoder, integrating the prior knowledge of scene distribution from the diffusion features into the segmentation model, and thereby producing more robust predictions. Extensive experiments across various DGSS settings demonstrate that our method surpasses existing approaches, achieving stateof-the-art performance. Moreover, as shown in Figure 1(a), our method generalizes effectively to extreme domains, such as cubist and impressionist paintings. The contributions are summarized as follows:
• We propose a novel agent queries-driven learning framework named QueryDiff, which employs agent queries as the interface to mine the scene distributions embedded in diffusion features, thereby improving the robustness of the model with respect to domain shifts.
• We propose diffusion consistency loss (DCL) to avoid intricate visual details in diffusion features from interfering with agent queries, enabling them to better focus on learning the semantic distribution of the scene.
• QueryDiff is a concise framework with high performance. It outperforms a wide variety of baselines and reaches a new state-of-the-art performance on extensive DGSS benchmarks.
this section cite: ['b6', 'b51', 'b19', 'b27', 'b42', 'b51', 'b53', 'b21', 'b10', 'b15', 'b38', 'b32', 'b20']

Section: Related Work

this section cite: []

Section: Diffusion Model
Diffusion models (Ho et al., 2020;Dhariwal & Nichol, 2021;Rombach et al., 2022) have recently demonstrated stateof-the-art image generation quality. With the success of diffusion models in generative tasks, some pioneer works have explored the application of diffusion models to various visual perception tasks (Xu et al., 2023;Tang et al., 2023;Ji et al., 2023;Lee et al., 2024;Ke et al., 2024). For example, Wu et al. (2023b) exploit cross-attention of the diffusion model to localize class-specific regions and generate a high-resolution segmentation mask. Xu et al. (2023) introduce ODISE, which integrates the internal representations of pre-trained diffusion models and discriminative models to effectively perform panoptic segmentation across any category in the wild. Ji et al. (2023) propose DDP, an extension of denoising diffusion process for semantic segmentation and depth estimation, transforming noise samples into desired predictions iteratively guided by input images. These methods rely on the diffusion model as the backbone, but its iterative sampling process, involving multiple denoising steps, incurs significant computational overhead and time consumption, rendering it inefficient for perception tasks. Thus, our method leverages agent queries as an interface to mine the underlying scene distribution embedded in diffusion features and refine the features of the segmentation model to improve its robustness, avoiding the direct involvement of diffusion features in the segmentation process.
this section cite: ['b15', 'b10', 'b38', 'b47', 'b41', 'b18', 'b25', 'b22', 'b47', 'b18']

Section: Domain Generalized Semantic Segmentation
Domain generalized semantic segmentation (DGSS) aims to enhance model robustness against domain shifts. The current approaches fall into two categories: normalization/whitening and domain randomization. Normalization/whitening methods learn domain invariant features by tailor-made modules to remove domain-specific features (Pan et al., 2018;Choi et al., 2021;Peng et al., 2022;Xu et al., 2022), but they struggle with complex domain shifts. Domain randomization seeks to broaden the distribution of source domain by either diversifying the data style through data augmentation (Lee et al., 2022;Zhao et al., 2022;Kim et al., 2023) or enriching the data content through generating new data (Benigmim et al., 2024), offering improved generalization performance. Recent studies have investigated the use of visual foundation models in DGSS. Benigmim et al. (2024) integrate various foundation models, including employing the diffusion model as a data generator, leveraging the robust features of CLIP, and utilizing SAM to refine pseudo labels for self-training. Jia et al. (2023) propose DGInStyle, a controlled pipeline for generating task-specific images with widely varying appearance data that samples the rich prior of a pre-trained diffusion model. Niemeijer et al. (2024) introduce a novel domain extension method utilizing a diffusion model to generate a pseudo-target domain with diverse text prompts and then training a generalizing model by adapting to this pseudo-target domain. Despite these advancements, current approaches leverage diffusion primarily as an auxiliary tool of data generation within training pipelines. In contrast, this paper introduces an agent queries-driven approach to fully mine domain-invariant semantic information of the diffusion features, thereby enhanc-ing the generalization capability of semantic segmentation.
this section cite: ['b34', 'b7', 'b35', 'b48', 'b26', 'b51', 'b23', 'b2', 'b2', 'b19', 'b32']

Section: Learnable Query Design
Recently, a series of learnable query-based frameworks inspired by DETR (Carion et al., 2020) have been proposed.
For instance, Panoptic SegFormer (Li et al., 2022) introduces a query decoupling strategy to prevent mutual interference between thing queries and stuff queries. MaskFormer (Cheng et al., 2021) and Mask2Former (Cheng et al., 2022) solve both semantic-and instance-level segmentation tasks in a unified framework by employing object queries to aggregate pixels within the same semantic region. MP-Former (Zhang et al., 2023) accelerates training by feeding noisy ground truth masks and learnable queries into the Transformer decoder to reconstruct the originals. Rein (Wei et al., 2024) fine-tunes vision foundation models by incorporating a set of randomly initialized queries as learnable parameters. While these studies have shown that specific learnable queries improve performance and functionality in perception tasks, research on learnable queries for DGSS remains unexplored. Our work aims to address DGSS by designing agent queries, constructed through learnable queries, to develop a concise, general, and efficient framework to improve the generalization of the model.
this section cite: ['b3', 'b28', 'b5', 'b6', 'b50', 'b43']

Section: Preliminary
Problem Definition. The goal of Domain Generalized Semantic Segmentation (DGSS) is to train a segmentation model φ = g • υ, where g and υ denote the backbone and decoder, respectively, on a labeled source domain S, enabling it to generalize effectively to an unseen target domain T . Both domains share the same set of K categories.
this section cite: []

Section: Stable Diffusion.
Stable diffusion involves a forward diffusion process that adds noise to the data and a reverse process that converts the noisy samples into raw data. The forward diffusion process is defined as:
q (z t | z 0 ) := N z t | √ ᾱt z 0 , (1 -ᾱt ) I(1)
where
z 0 = E(x)
and E is a pre-trained encoder that encodes image x ∈ R H×W ×3 into a latent representation space. ᾱt := t s=0 α s = t s=0 (1 -β s ) and β s represents the pre-defined noise schedule (Ho et al., 2020). This process transforms data sample z 0 to a latent noisy sample z t , and t is the diffusion step. Generally, a larger t corresponds to larger noise weights. The reverse process is a denoising procedure that gradually transforms the noisy samples into raw data. A step in the reverse process can be defined as:
p θ (z t-1 | z t ) := N (z t-1 | µ θ (z t , t) , Σ θ (z t , t)) (2)
where the mean µ θ is predicted by noise predictor ϵ θ (•) and the covariance Σ θ is generally fixed as a predefined value.
Figure 2. A brief illustration of our proposed framework. First, we use learnable queries to aggregate hierarchical instance features from the segmentation backbone, progressively merging them to form agent queries. Next, we utilize agent queries to learn the scene distribution information embedded in the diffusion features, optimizing their semantic representations through diffusion consistency loss (DCL) that removes visual appearance information irrelevant to the perceptual task from the diffusion features. Finally, we use the optimized agent queries to refine the instance features of the segmentation decoder and output the prediction mask.
this section cite: ['b15']

Section: Methodology
With their powerful ability to generate high-quality samples, diffusion models have been successfully applied to improve segmentation generalization by expanding source domain data, drawing on their rich priors of scene distribution. Guided by the principle that teaching a man to fish is better than giving him a fish, a natural question arises: Can the scene distribution priors of diffusion models be directly leveraged to enhance segmentation generalization? As illustrated in Figure 2, we propose a novel agent query-driven DGSS framework QueryDiff based on diffusion models, which consists of three key steps: (1) aggregating semantic information within a scene to generate agent queries (Section 4.1), (2) leveraging agent queries to mine scene distribution knowledge encoded in diffusion features (Section 4.2), and (3) reinforcing scene distribution information in the segmentation features via agent queries (Section 4.3).
this section cite: []

Section: Agent Queries Generation
Considering the substantial inefficiency resulting from the computationally intensive iterative sampling process of diffusion models, our approach employs agent queries as the interface throughout the framework, instead of directly incorporating diffusion features into the segmentation process. Specifically, we aggregate hierarchical visual features from the segmentation backbone and progressively merge them into a unified representation, referred to as agent queries.
Aggregating Hierarchical Visual Features. We establish multiple groups of learnable queries, each corresponding to a feature layer in the segmentation backbone. For the features f l seg produced by the l-th layer of the segmentation backbone g, we first construct layer-wise queries as follows:
q l layer = MLP(Softmax(f l seg × q l init ) T × f l seg ) with q layer = q l layer ∈ R r×c | 1 ≤ l ≤ L (3
)
where q l layer are layer-wise queries for l-th feature f l seg , q l init ∈ R c×r are randomly initialized learnable queries. c represents the dimension of each feature layer, r denotes the sequence length of q l init , L represents the total number of feature layers in the segmentation backbone. All layers share the same MLP weights.
Progressive Merging for Unified Representation. Each layer-wise query corresponds to at least one instance of interest in the scene, thus there is information overlap among the layer-wise queries. To ensure that each query ultimately retrieves the features of the same instance in different layers, we merge the layer-wise queries into a whole using a progressive merging strategy. Specifically, for each stage i, we employ learnable queries q i init , map them to queries Q i and the outputs of the previous stage, q i-1 stage , are respectively projected into the keys K i and values V i
Q i = q i init W i Q , K i = q i-1 stage W i K , V i = q i-1 stage W i V q i init ∈ R nir×c , q i-1 stage ∈ R ni-1r×c , 1 ≤ i ≤ M(4)
where W i Q , W i K and W i V are linear projections. q i init are randomly initialized learnable queries, n i denotes the number of the queries produced at the i-th merge stage, M represents the total number of merging stages and q 0 stage = q layer ∈ R Lr×c . Next, we merge the smaller groups into larger ones based on the similarity matrix in the embedding space:
qi stage = FFN exp s i hw j=1 exp (s i ) × V i , s i = Q i (K i ) T √ d i(5)
where qi stage ∈ R nir×c are the output of the current stage, d i is a scaling factor, and FFN consists of a linear mapping followed by an activation layer. h and w denote the height and width of the similarity matrix, respectively. Notably, the number of queries generated at each merge stage decreases progressively, i.e., n i+1 <n i . Next, qi stage are fed to selfattention and FFN to enrich the representation of each query and obtain the output q i stage , which is the next stage of inputs. After the final stage, M , we average their outputs to obtain the final global agent queries:
q agent = AvgPool q M stage × W a + b a , q agent ∈ R r×c(6)
where W a and b a signify the weights and biases, respectively. As a result, we embed semantic information from the segmentation backbone into agent queries and establish an implicit linkage between agent queries and instances within the scene. These agent queries are then guided by the diffusion features to learn contextual relationships between different instances, enabling a more comprehensive understanding of the scene distribution.
this section cite: []

Section: Diffusion-Guided Agent Queries Optimization
The diffusion model encapsulates not only prior knowledge of scene distribution but also various visual appearances (e.g., colors and textures), which are irrelevant or even detrimental to perceptual tasks. To guide agent queries in focusing on the scene distribution, we propose a diffusion consistency loss (DCL) that decouples scene distribution knowledge from intricate visual details in diffusion features.
this section cite: []

Section: Extraction of Diffusion Features.
Before that, we need to extract diffusion features. To do this, we first transform the image x to the latent space using the pre-trained encoder E to obtain the latent code z 0 = E(x), then use Equation 1 to get the noise samples z t , and finally use the noise predictor ϵ θ (•) to perform the denoising process to get diffusion features:
f (ts,j) d = ϵ j θ (z ts , t s , T (e)) , f (tw,j) d = ϵ j θ (z tw , t w , T (e))(7)
where T is the text encoder, e is the empty text, 1 ≤ j ≤ K and K represents the total number of diffusion features. Different timesteps t correspond to different denoising stages, where weak noise added at timestep t w results in diffusion features f (tw,j) d with finer-grained details, while strong noise at timestep t s leads to features f (ts,j) d with coarser-grained semantic information.
The Diffusion Consistency Loss. Our goal is to enable agent queries to comprehensively understand the scene semantic distribution, guided by diffusion features, to capture the complex contextual relationships between instances within the scenario. To achieve this, we firstly perform a dot product operation on the agent queries q agent with the weak noise diffusion features f (tw,j) d to obtain a similarity map:
S tw j = Softmax f (tw,j) d × Linear(q agent ) T (8
)
where Linear is a two-layer MLP with layer normalization. The dot product operation effectively associates the instance semantics of the agent queries with the scene distribution knowledge in the weak-noise diffusion features. As a result, S tw j inherently contains the scene distribution prior embedded in the weak-noise diffusion features. However, the visual detail information inherent in the weak-noise diffusion features f (tw,j) d inevitably propagates into S tw j . To mitigate the interference of these details, it is necessary to strip away the potential visual detail within S tw j . Considering that the primary distinction between strong-noise diffusion features f (ts,j) d and weak-noise diffusion features f (tw,j) d lies in the low-level visual detail, while both remain essentially identical in representing the scene semantic distribution, we leverage S tw j and q agent to reorganize the weak-noise features, with the original strong-noise diffusion features f (ts,j) d supervising the reorganization process:
L dist = Ĥ ĥ=1 Ŵ ŵ=1 f (ts,j) d ( ĥ, ŵ) log f (ts,j) d ( ĥ, ŵ) f j d ( ĥ, ŵ) f j d = S tw j × Linear(q agent )(9)
where Ĥ and Ŵ represent the height and width dimensions of the feature map f (ts,j) d
, respectively. By enforcing a consistency constraint, visual details in S tw j are effectively minimized. This is because if S tw j includes excessive visual details, it would result in a significant discrepancy between the reorganized result f j d and the strong-noise diffusion features f (ts,j) d . The consistency loss minimizes this discrepancy, ensuring that during the construction of S tw j , only the semantic distribution of the scene is preserved while visual details are suppressed.
Ultimately, we utilize S tw j to optimize the instance semantic representation in the agent queries, resulting in updated feature representations:
q j opt = (f (ts,j) d ) T × S tw j (10
)
To guide the agent queries q agent in constructing the scene distribution, we supervise them using q opt :
L sup = K j=1 L δ (q agent , (q j opt ) T ) with L δ (x, y) = 0.5(x -y) 2 if |x -y| < 1 |x -y| -0.5 otherwise (11
)
The diffusion consistency loss is:
L diff = L sup + L dist (12
)
This process enables the agent queries to actively establish contextual relationships between instances within the scene during their generation, fostering a comprehensive understanding of the underlying scene distribution.
this section cite: []

Section: Feature Refinement and Mask Prediction
The final step involves leveraging the optimized agent queries to refine the features in the segmentation decoder and predicting the segmentation mask. The typical decoder can be divided into three components: a pixel decoder that extracts features, a transformer decoder that outputs mask embedding, and a mask decoder that outputs class probabilities (Cheng et al., 2022). In our pipeline, we insert the refinement module after the pixel decoder, which takes the multi-scale features output by the pixel decoder and the optimized agent queries as inputs and outputs the refined features. For the refinement module, we use cross-attention and multiple parallel 3×3 depthwise separable convolutions (Chollet, 2017) with different dilation rates to refine feature representations. In particular, we start by feeding the multi-scale features and optimized agent queries into cross-attention to get new feature representations. Next, we concatenate these with the original multi-scale features, perform fusion using depth-separable convolution, and then align dimensions with 1×1 convolution to obtain refined pixel features. The transformer decoder produces a set of mask embeddings, which take the refined pixel features as input. These mask embeddings are then classified by the mask decoder to produce a set of class probabilities. Finally, we use the segmentation loss L seg as follows:
L seg = λ bce L bce + λ dice L dice + λ cls L cls (13
)
where the predicted mask is optimized using binary cross entropy loss L bce and dice loss L dice , while the predicted classes of the mask embedding are optimized using binary cross entropy loss L cls . The loss weights λ bce , λ dice and λ cls are set to the same values.
Full Objective. The full training objective consists of the segmentation loss and the diffusion consistency loss:
L total = L seg + αL diff (14
)
where α controls the weight of the diffusion consistency loss. Notably, during inference, the step involving diffusion features is omitted, ensuring our method remains concise, efficient, and general.
this section cite: ['b6', 'b8']

Section: Experiments

this section cite: []

Section: Experimental Setups
Datasets. We evaluate the performance of QueryDiff following the domain generalization standard setting (Hümmer et al., 2023;Benigmim et al., 2024;Wei et al., 2024). As synthetic datasets, GTA5 (Richter et al., 2016) provides 24,966 images at a resolution of 1914×1052. As real-world datasets, Cityscapes (Cordts et al., 2016) includes 2,975 images for training and 500 images for validation, with images at 2048×1024 resolution. BDD100K (Yu et al., 2020) consists of 7,000 training images and 1,000 validation images, each at 1280×720 resolution. Mapillary (Neuhold et al., 2017) offers 18,000 training images and 2,000 validation images, with resolutions varying across the dataset. As adverse weather datasets, ACDC (Sakaridis et al., 2021) consists of 406 validated images at 1920×1080 resolution, including night, snow, rain and fog. For simplicity, we abbreviate GTA5, Cityscapes, BDD100K, and Mapillary as G, C, B, and M, respectively; AN, AS, AR and AF denote night, snow, rain and fog subsets of ACDC, respectively.
Implementation Details. Following previous works (Benigmim et al., 2024;Wei et al., 2024), we use the decoder from mask2former (Cheng et al., 2022), a widely-used segmentation head compatible with various backbones, including ResNet50 (He et al., 2016), MiT-B5 (Xie et al., 2021), and DINOv2 (Oquab et al., 2023). For the training phase, the AdamW optimizer (Loshchilov & Hutter, 2017) is employed, setting the learning rate at 1e-5 for the backbone and 1e-4 for both the decoder and the learnable queries. We utilize a configuration of 60,000 iterations with a batch size of 4, and crop images to a resolution of 512 × 512. We employ Stable Diffusion v2-1 (Rombach et al., 2022) as our diffusion model, which is trained on LAION5B (Schuhmann et al., 2022) and remains frozen throughout.
this section cite: ['b17', 'b2', 'b43', 'b37', 'b9', 'b49', 'b31', 'b39', 'b2', 'b43', 'b6', 'b14', 'b46', 'b33', 'b30', 'b38', 'b40']

Section: Comparison with Previous Methods
We comprehensively compare our method with existing DGSS methods. We conduct experiments in three generalization settings: synthetic-to-real (G→C, B, M), real-to-real (C→B, M) and normal-to-adverse (C→AF, AR, AS, AN).
this section cite: []

Section: Synthetic-to-real generalization.
Table 1 presents a comparison between our proposed method and existing DGSS approaches under the G→C, B, M scenario. Compared to the previous SOTA method Rein, our approach achieves a substantial improvement of 2.4 points and consistently surpasses all DGSS methods across various backbone archi- Real-to-real generalization. Under this evaluation setup, models are trained on Cityscapes and tested on BDD100K and Mapillary. The results, starting from the ninth column of Table 1, confirm that our method consistently delivers better domain generalization performance across various datasets and backbone configurations, underscoring its effectiveness in adapting to diverse real-world scenarios.
Normal-to-adverse generalization. Under this evaluation  setup, all the models are trained on Cityscapes and tested on ACDC. Table 2 highlights that our method achieves consistently better performance than competing approaches across all three backbone variants. When using ResNet50, our method improves performance from 51.6 to 53.8 compared with HGFormer. As for the backbone of MiT-B5 and DINOV2-L, our method respectively improves the mIoU by 1.9 and 1.6 compared with the previous SOTA method.
Comparison of the class-wise IoU. We gauge the impact  of our method on class-wise IoU scores, as shown in Figure 3, which demonstrates notable performance gains across multiple classes, especially in rarer ones like rider, truck, train, and motorbike. The heatmap affirms the capability of our method across a wide range of classes.
Qualitative results. We visually compare the segmentation results with the previous SOTA Rein under G→C, B, M setting in Figure 4. The results highlighted by white dash boxes show that our method creates more reasonable class predictions as well as more complete spatial distributions. We think it is because the proposed method learns the correct knowledge of the scenario, which is crucial for improving the generalization of the model.
this section cite: []

Section: Ablation Studies
In this section, we present comprehensive experiments to validate the effectiveness of our method.
this section cite: []

Section: Effect of components.
We evaluate the effectiveness of the primary components under G→C, B, M settings, focusing on agent queries, L sup and L dist in the proposed method.
The baseline model is based on Rein. As shown in Table 3, we can observe: (1) Each module positively contributes to the model's performance, demonstrating the individual effectiveness of these components.
(2) The combination of all three modules yields the highest mIoU of 66.7 (+2.4 over the baseline), highlighting their synergy in enhancing cross-domain segmentation.
Investigating various VFMs. We evaluate QueryDiff across multiple Vision Foundation Models (VFMs), including CLIP (Radford et al., 2021), SAM (Kirillov et al., 2023), and DINOv2 (Oquab et al., 2023), under full finetuning, lightweight fine-tuning (Rein), and frozen backbone schemes (Wei et al., 2024) in the G → C, B, M setting. As shown in Table 4, QueryDiff achieves the highest mIoU, significantly outperforming other methods.
Study on agent queries length r. The core component of our method is a set of agent queries. We explore various lengths for the agent query sequence, ranging from 50 to 150. As demonstrated in Figure 5, models with r = 100 achieve a strong mIoU of 69.1% and 68.6% on Cityscapes and Mapillary datasets, respectively.
The choice of t w and t s . Noise intensity increases with the timestep value, with higher values introducing stronger perturbations. Previous studies (Xu et al., 2023;Baranchuk et al., 2022) reveal that semantically meaningful diffusion features are primarily generated within the range of 0 to 200, and that neighboring timesteps often produce highly similar representations. As shown in Table 5, we empirically adapt t w = 0 and t s = 100 to ensure that agent queries can capture high-dimensional semantics while avoiding being influenced by trivial information.
Comparing different stable diffusion. In
Table 6, we use Rein as the baseline model and conduct the comparative experiment with three currently dominant stable diffusion models. The final results are not sensitive to the choice of different diffusion models, and our proposed method, when combined with different diffusion models, consistently outperforms the baseline model. 6. Conclusion In this work, we propose QueryDiff, a novel agent querydriven learning framework based on diffusion model guidance for DGSS. QueryDiff leverages agent queries as an interface to mine scene distribution priors embedded in diffusion model. Also, to avoid the interference of visual details, we propose diffusion consistency loss to enable agent queries to focus on domain-invariant semantic information. Extensive experiments demonstrate that QueryDiff significantly surpasses previous SOTA methods across various benchmarks, highlighting its effectiveness. This work not only bridges a critical research gap but also establishes a new standard for domain generalized semantic segmentation.
this section cite: ['b36', 'b24', 'b33', 'b43', 'b47', 'b1']

Section: Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.
this section cite: []

Section: References
Ref_id:b0 Title: Style blind domain generalized semantic segmentation via covariance alignment and semantic consistence contrastive learning Year: (2024)
Ref_id:b1 Title: Label-efficient semantic segmentation with diffusion models Year: (2022)
Ref_id:b2 Title: Collaborating foundation models for domain generalized semantic segmentation Year: (2024)
Ref_id:b3 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b4 Title: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs Year: (2017)
Ref_id:b5 Title: Per-pixel classification is not all you need for semantic segmentation Year: (2021)
Ref_id:b6 Title: Masked-attention mask transformer for universal image segmentation Year: (2022)
Ref_id:b7 Title: Improving domain generalization in urbanscene segmentation via instance selective whitening Year: (2021)
Ref_id:b8 Title: Xception: Deep learning with depthwise separable convolutions Year: (2017)
Ref_id:b9 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b10 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b11 Title: Hgformer: Hierarchical grouping transformer for domain generalized semantic segmentation Year: (2023)
Ref_id:b12 Title: A simple recipe for language-guided domain generalized segmentation Year: (2024)
Ref_id:b13 Title: Dual attention network for scene segmentation Year: (2019)
Ref_id:b14 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b15 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b16 Title: Context-aware high-resolution domain-adaptive semantic segmentation Year: (2022)
Ref_id:b17 Title: Simple transfer of clip-based vision-language representations for domain generalized semantic segmentation Year: (2023)
Ref_id:b18 Title: Diffusion model for dense visual prediction Year: (2023)
Ref_id:b19 Title: Dginstyle: Domaingeneralizable semantic segmentation with image diffusion models and stylized semantic control Year: (2023)
Ref_id:b20 Title: Dginstyle: Domaingeneralizable semantic segmentation with image diffusion models and stylized semantic control Year: (2025)
Ref_id:b21 Title: Domain generalization via balancing training difficulty and model capability Year: (2023)
Ref_id:b22 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b23 Title: Texture learning domain randomization for domain generalized segmentation Year: (2023)
Ref_id:b24 Title: Segment anything Year: (2023)
Ref_id:b25 Title: Exploiting diffusion prior for generalizable dense prediction Year: (2024)
Ref_id:b26 Title: Learning domain generalized semantic segmentation from the wild Year: (2022)
Ref_id:b27 Title: Intra-source style augmentation for improved domain generalization Year: (2023)
Ref_id:b28 Title: Panoptic segformer: Delving deeper into panoptic segmentation with transformers Year: (2022)
Ref_id:b29 Title: Fully convolutional networks for semantic segmentation Year: (2015)
Ref_id:b30 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b31 Title: The mapillary vistas dataset for semantic understanding of street scenes Year: (2017)
Ref_id:b32 Title: Generalization by adaptation: Diffusion-based domain extension for domaingeneralized semantic segmentation Year: (2024)
Ref_id:b33 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b34 Title: Two at once: Enhancing learning and generalization capacities via ibn-net Year: (2018)
Ref_id:b35 Title: Semanticaware domain generalized segmentation Year: (2022)
Ref_id:b36 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b37 Title: Playing for data: Ground truth from computer games Year: (2016)
Ref_id:b38 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b39 Title: Acdc: The adverse conditions dataset with correspondences for semantic driving scene understanding Year: (2021)
Ref_id:b40 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b41 Title: Emergent correspondence from image diffusion Year: (2023)
Ref_id:b42 Title: Heterogeneous domain generalization via domain mixup Year: (2020)
Ref_id:b43 Title: Stronger fewer & superior: Harnessing vision foundation models for domain generalized semantic segmentation Year: (2024)
Ref_id:b44 Title: Datasetdm: Synthesizing data with perception annotations using diffusion models Year: (2023)
Ref_id:b45 Title: Diffumask: Synthesizing images with pixel-level annotations for semantic segmentation using diffusion models Year: (2023)
Ref_id:b46 Title: Segformer: Simple and efficient design for semantic segmentation with transformers Year: (2021)
Ref_id:b47 Title: Open-vocabulary panoptic segmentation with text-to-image diffusion models Year: (2023)
Ref_id:b48 Title: Dirl: Domain-invariant representation learning for generalizable semantic segmentation Year: (2022)
Ref_id:b49 Title: Bdd100k: A diverse driving dataset for heterogeneous multitask learning Year: (2020)
Ref_id:b50 Title: Mp-former: Mask-piloted transformer for image segmentation Year: (2023)
Ref_id:b51 Title: Style-hallucinated dual consistency learning for domain generalized semantic segmentation Year: (2022)
Ref_id:b52 Title: Squeeze-and-attention networks for semantic segmentation Year: (2020)
Ref_id:b53 Title: Adversarial style augmentation for domain generalized urban-scene segmentation Year: (2022)
