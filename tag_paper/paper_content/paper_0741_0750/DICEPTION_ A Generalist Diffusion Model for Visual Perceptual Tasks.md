Title: DICEPTION: A Generalist Diffusion Model for Visual Perceptual Tasks
Abstract: This paper's primary objective is to develop a robust generalist perception model capable of addressing multiple tasks under constraints of computational resources and limited training data. We leverage text-to-image diffusion models pre-trained on billions of images and successfully introduce our DICEPTION, a visual generalist model. Exhaustive evaluations demonstrate that DICEPTION effectively tackles diverse perception tasks, even achieving performance comparable to SOTA single-task specialist models. Specifically, we achieve results on par with SAMvit-h using only 0.06% of their data (e.g., 600K vs. 1B pixel-level annotated images). We designed comprehensive experiments on architectures and input paradigms, demonstrating that the key to successfully re-purposing a single diffusion model for multiple perception tasks lies in maximizing the preservation of the pre-trained model's prior knowledge. Consequently, DICEPTION can be trained with substantially lower computational costs than conventional models requiring training from scratch. Furthermore, adapting DICEPTION to novel tasks is highly efficient, necessitating fine-tuning on as few as 50 images and approximately 1% of its parameters. Finally, we demonstrate that a subtle application of classifier-free guidance can improve the model's performance on depth and normal estimation. We also show that pixel-aligned training, as is characteristic of perception tasks, significantly enhances the model's ability to preserve fine details. DICEPTION offers valuable insights and presents a promising direction for the development of advanced diffusion-based visual generalist models.

Section: Introduction
Foundation models [51,90,125,126,123,11,7,86,78,94,6,40], typically requiring extensive training on billions of data samples, play a pivotal role in their respective domains. In natural language processing (NLP), current foundation models [9,105,106,27] have already demonstrated the potential to serve as versatile solutions, solving diverse fundamental tasks and with minimal fine-tuning needed for new tasks. This success can be attributed to the relatively small representational differences among various language tasks. However, in the domain of computer vision, task representations can differ substantially, and up to date, we still lack an effective approach to unify these distinct tasks. Consequently, existing vision foundation models usually excel at one single specific task, such as image segmentation [51,90] or monocular depth estimation [125,126,123], because they are trained on data tailored exclusively to that task. Owing to the pronounced disparity in visual representations across tasks, coupled with the single-task specialization that characterizes current vision foundation models, fine-tuning these models for new tasks remains a formidable DICEPTION can quickly adapt to new tasks by fine-tuning less than 1% of its parameters on as few as 50 images. For additional visualizations, please refer to Figures S8, S11, S10, S15, S16, S17, S18, S19, S20, S21, S22 in the Appendix. We select Person as the instance segmentation example for consistent visualization. Our method is limited to only human instances. challenge. Despite efforts [12,78,40,91] to learn universal visual representations, these models still falls noticeably short compared to specialized models in specific tasks.
Recent studies [115,71,70,75,2,119] on visual generalist models are predominantly trained from scratch, often requiring substantial computational resources and large datasets to achieve good results. Unfortunately, the price of collecting a sufficiently large and high-quality multi-task dataset is substantial. Here, inspired by the success of diffusion models, we propose the hypothesis that leveraging their powerful priors can help mitigate the significant computational and data overhead for training powerful generalist models. While some existing works [49,120,39,129,96] have demonstrated that this is feasible in single-task scenarios, the potential of diffusion model priors in multi-task settings remains largely under-explored.
In this paper, we successfully leverage the priors of diffusion models to achieve results on par with the state-of-the-art models on various tasks with only minimal training data. We name our powerful visual generalist model DICEPTION. For each task, we require substantially less data than specialized foundation models. For instance, compared to SAM segmentation trained on 1 billion pixel-level annotated samples, DICEPTION achieves comparable performance using a significantly smaller dataset of 600K samples, without any training data cherry-picking.
More significantly, DICEPTION highlights that the generative image priors lead to surprisingly more efficient and effective pathways to generalist image understanding models. We analyze a series of design choices for transferring one single modern diffusion model to multiple perception tasks, and identify that the key to successful transfer lies in preserving as much of the pretrained prior as possible, eliminating the need to design any complex module or training recipe. Even more notably, DICEPTION is capable of quickly adapting to new tasks using as few as 50 training images and fine-tuning less than 1% of its parameters. We also demonstrate that pixel-level aligned training for perception tasks significantly enhances the model's ability to preserve fine details and mitigates generated artifacts, which is of high significance for downstream applications. We believe DICEPTION provides valuable insights for the design of strong diffusion-based generalist models.
In summary, our main contributions are as follows.
• We introduce DICEPTION, to the best of our knowledge, the first unified multi-task perception model with fully shared parameters that achieves quantitative performance comparable to specialized models while requiring significantly less data. E.g., we achieve competitive representation space, and distinct decoders are employed to transform tokens into the outputs specific to each task. However, these methods face notable limitations: they need to train a separate encoder and decoder for every individual task and they usually rely on substantial amounts of data to attain optimal performance.
The recent success of high-quality Vision Language Models (VLMs) [66] has also encouraged researchers to leverage them for building multitask models. Yet, these VLM-based methods [4,110,17,69,92,61] typically focus on multimodal understanding tasks, such as image captioning, rather than general visual perception tasks. Meanwhile, some approaches [101,139,79] combine diffusion models with autoregressive models, focusing primarily on instruction-following image generation or editing tasks, rather than addressing image perception tasks. Although certain studies [54,47,18,35] have tried to apply VLMs to more advanced semantic perception tasks, they struggle to establish a unified generalist visual model.
this section cite: ['b50', 'b89', 'b124', 'b125', 'b122', 'b10', 'b6', 'b85', 'b77', 'b93', 'b5', 'b39', 'b8', 'b104', 'b105', 'b26', 'b50', 'b89', 'b124', 'b125', 'b122', 'b11', 'b77', 'b39', 'b90', 'b114', 'b70', 'b69', 'b74', 'b1', 'b118', 'b48', 'b119', 'b38', 'b128', 'b95', 'b65', 'b3', 'b109', 'b16', 'b68', 'b91', 'b60', 'b100', 'b138', 'b78', 'b53', 'b46', 'b17', 'b34']

Section: Compared with One Diffusion
The concurrent work, One Diffusion [55], addresses multi-task image generation, whereas our approach focuses on multi-task image understanding. We excel at performing a broader range of image understanding tasks with higher quality. While One Diffusion's strategy of treating different images as different views benefits generation tasks, their failure to distinguish between conditions and images introduces harmful degrees of freedom for perception tasks, as illustrated in the redhighlighted regions of Figure S14. Specifically, when performing perception tasks, One Diffusion tends to generate an image similar to the original input, rather than the desired perceptual results.
Although One Diffusion suggests that more detailed text prompts can lead to better results, we argue that performance in perception tasks should not overly depend on the quality of text prompts.
In contrast, our method uses only simple task prompts to distinguish between different tasks, rather than allowing the text prompts to dominate the results.
Crucially, while One Diffusion requires a massive amount of data (75 million samples) and computational resources for from-scratch training, we leverage the priors of pretrained models and demonstrate that, with significantly less data (1.8 million samples), we achieve performance on par with state-of-the-art results. In the image understanding tasks shared by both approaches, we consistently produce more stable and higher-quality results than One Diffusion.
this section cite: ['b54']

Section: Method

this section cite: []

Section: Overview
Our methodology builds upon pre-trained text-to-image diffusion models [29], steering perception tasks using text prompts. As shown in Figure 2, we concatenate the input image tokens, the noisy tokens, task prompt embeddings, and point embeddings for interactive segmentation along the token dimension. Training employs a flow matching loss [29], exclusively computed on the noisy tokens. In inference, each denoising step refines only these noisy tokens, leaving all other conditioning tokens unchanged throughout the iterative denoising process.
this section cite: ['b28', 'b28']

Section: Unifying Task Representation into RGB Space
The decision to unify representations of diverse tasks in RGB space was motivated by two key factors: (1) It maximally leverages the priors in text-to-image models, which have been extensively trained within the RGB domain.
(2) RGB serves as a foundational representation in computer vision, providing a common visual framework through which a wide variety of tasks can be coherently and intuitively visualized.
We focus on several of the most fundamental tasks in computer vision: monocular depth estimation, normal estimation, human keypoint estimation and segmentation. Segmentation, in particular, encompasses interactive segmentation, entity segmentation, and instance segmentation. Our instance segmentation segments target instances with category name as input. All these tasks can be unified within an RGB space, with the difference being the number of channels. For single-channel representations, such as depth maps and segmentation masks, we align them with RGB by repeating the
Image2Depth 1. Unified Different Tasks into RGB Image2Depth Image2Normal Image2Entity Image2Seg Image2Pose Image2Instance -Person
this section cite: []

Section: Pipeline

this section cite: []

Section: Image2Seg
Interactive Seg Other Tasks (u,v)
this section cite: []

Section: Input Image Noise Task Prompt Point Embedding

this section cite: []

Section: Valid Embeds

this section cite: []

Section: Invalid Embeds

this section cite: []

Section: Point Embedding Processing
Interactive Seg (5 Points at Most) Point Embedding Processing VAE+Patchify VAE+Patchify Text Encoder Other Tasks = = (u,v) Positional Encoding 2-Layer MLP + Invalid Embeds Valid Embeds Fully Parameter-Sharing DiT Token-wise Concatenation Flow Matching Loss If 2 Points: If 3 Points: We select Person as the instance segmentation example for the purpose of consistent visualization, which does not mean our method is limited to only human instances. At each denoising step, the point embedding, input image latent, and task embedding remain fixed, while only the noise latent is updated.
channel three times. For inherently three-channel representations, such as normal maps, we treat them directly as RGB images.
Entity segmentation is to segment every instance in an image but with no category. We assign each mask within an image a random color and merge them into a three-channel RGB mask. Painter [115] found that assigning color randomly makes the model hard to optimize. However, we find this approach has no adverse impact on the training and enables the model to effectively learn to distinguish different instances by painting them with different colors. Each instance's mask can be extracted from the RGB mask using clustering algorithms during post-processing without significant performance degradation. We also apply the random color assignment in instance segmentation. Our method is capable of segmenting instances of the same semantic category. By default, we use KMeans for mask extraction.
Let x r denote the pre-unified raw representation for each task, and x represents the unified RGB-like output representation. We formalize this process as: x = Ψ(x r ).
this section cite: ['b114']

Section: DICEPTION: A Unified Framework
Architecture. Our model adopts the same architecture as SD3 [29]. We aim to keep the architecture as unchanged as possible, fully leveraging the pre-trained prior knowledge. To do so, we concatenate the input image tokens, noisy tokens, task embeddings, and point embeddings along the token dimension as input to the model. During training, the loss is computed only on the noisy tokens. Similarly, during inference, at each timestep, only the noisy tokens are updated, while the other tokens remain unchanged. We use simple task prompts to direct the model to perform various tasks, such as "image to depth", "image to normal", and "image to segmentation". An additional category name is provided in instance segmentation, such as "image to instance -cat".
this section cite: ['b28']

Section: Introduction of Point Embeddings
For point-prompted interactive segmentation, a naive approach is directly painting points on the image. But this strategy is highly sensitive to the size of the points.
If the painted points are too large, they can obscure small regions, causing segmentation to fail. Conversely, if the painted points are too small, the model may lose relevant point information after VAE downsampling and patchification. To address this, we introduce a minimal straightforward two-layer MLP Φ(•) that enables the model to understand the point prompt.
Inspired by SAM [51], we apply sin-cos positional encoding to the point coordinates p, then pass them into the MLP Φ(•) to produce point embeddings that match the dimension of the input hidden states. We use two learnable embeddings to indicate whether the embedding is valid or not: ξ p for valid point embeddings and ξ np for invalid point embeddings. The processed point embedding is summed with ξ p . For other tasks, we simply use ξ np as the point embedding. During training, we randomly select 1-5 points to guide the segmentation. When the number of selected points is fewer than 5, we pad the point embeddings to a length of 5 with ξ np . When performing tasks that do not require point input, the point embedding is simply a length-5 sequence, where each element is ξ np . By denoting the final point embedding as ξ, this process is formulated as:
ξ = Concat(Φ(PE(p)) + ξ p , ξ np ) if interactive segmentation ξ np else (1
)
Input Formulation and Loss. DICEPTION introduces two additional inputs based on SD3: the input image x ′ and point embedding ξ. For the input image, we first apply VAE to down-sample it by a factor of 8, after which it is 2 × 2 patchified into sequences. We denote this pre-processing as τ . Subsequently, the task prompt token e, point embedding ξ, noisy token z t , and input image token z ′ are concatenated along the token dimension to form the complete input. We follow the flow matching [65,1,68] loss in training SD3 [29], which minimizes the discrepancy between the model's predicted velocity v and the ground-truth velocity u. During training, the loss is applied solely to the noisy tokens:
z 0 = τ (x), z ′ = τ (x ′ )
Loss = E z0,t ∥v θ (z t , z ′ , t, e, ξ) -u(z t )∥ 2 2 .
(2)
this section cite: ['b50', 'b64', 'b0', 'b67', 'b28']

Section: Adapting to New Tasks
Practical applications often require models to adapt quickly to new tasks with limited training data. Traditional foundation models, however, are often domain-specific and require extensive data and architectural modifications for adaptation. Powerful diffusion models also struggle with efficient adaptation to downstream tasks via few-parameter fine-tuning on limited data.
DICEPTION effectively addresses this limitation. We conducted experiments on lung segmentation, tumor segmentation, and image highlighting, which represent tasks with varying degrees of overlap with the model's original domain. We train fewer than 1% of the model's parameters using LoRA [44] without any complex architectural modifications. Notably, despite the limited availability of training samples (50 per task), DICEPTION consistently delivered successful and high-quality performance across all target tasks. These results provide compelling evidence for the potential of DICEPTION as a unified foundation model.
this section cite: ['b43']

Section: Experiments

this section cite: []

Section: Implementation Details
Data. We randomly select 500k images from the OpenImages [53] dataset and use DepthPro [7] and StableNormal [129] to generate depth and normal annotations. For interactive segmentation, we randomly select 400k images from the SA-1B [51] dataset, as well as 200k images with fine-grained hair masks synthesized from the AM2k [58], AIM500 [59], and P3M-10k [57]. Entity segmentation data is from EntityV2 [84], while instance segmentation data comes from the COCO-Rem [97], and human pose data is sourced from COCO [64]. For few-shot fine-tuning, we select 50 samples from the Chest X-Ray dataset [114], LOL-v2 [127], and Kaggle's Brain Tumor dataset as training samples. More details can be found in Appendix A.
Training. Our training lasts for 24 days using 4 NVIDIA H800 GPUs. We employ the AdamW optimizer with a constant learning rate of 2e-5 and a batch size of 28 per GPU. We found that the training process is highly stable. However, the convergence speed for segmentation tasks was slower compared to depth and normal tasks. Therefore, we increased the proportion of segmentation data in each batch. Specifically, in each batch, depth and normal each account for 15%, interactive segmentation, entity segmentation, and instance segmentation each account for 20%, and pose estimation accounts for 20%. We observe that, by the end of training, despite the loss no longer significantly decreasing, the model's performance on segmentation tasks continues to improve.
During few-shot fine-tuning, we apply a rank-128 LoRA to all attention Q, K, and V layers in the network, which accounts for less than 1% of the total network parameters. The task prompts for different tasks are "image-to-segmentation lung," "image-to-segmentation tumor," and "image-tohighlight." LoRA training is conducted on a single NVIDIA H100 GPU, with a constant learning rate of 2e-5 and a batch size of 8. Please refer to Appendix D for more few-shot fine-tuning visualizations.
Inference. We perform 28 steps of denoising during inference which follows the settings of the pre-trained model SD3 [29]. The inference can be run on a GPU of 24GB memory with a batch size of 4. The classifier-free-guidance value is by default set to 2, more analysis in Appendix B.
this section cite: ['b52', 'b6', 'b128', 'b50', 'b57', 'b58', 'b56', 'b83', 'b96', 'b63', 'b113', 'b126', 'b28']

Section: Comparisons with Existing Methods
Table 1: Quantitative comparison of depth estimation with both specialized models and multi-task models on zero-shot datasets. Our visual generalist model can perform on par with SOTA models. We use the same evaluation protocol ( †) as Genpercept [120].
Method Training KITTI [33] NYUv2 [77] ScanNet [24] DIODE [108] ETH3D [95] Samples AbsRel↓
δ 1 ↑ AbsRel↓ δ 1 ↑ AbsRel↓ δ 1 ↑ AbsRel↓ δ 1 ↑ AbsRel↓ δ 1 ↑ MiDaS [
89] 2M 0.236 0.630 0.111 0.885 0.121 0.846 0.332 0.715 0.184 0.752 Omnidata [28] 12.2M 0.149 0.835 0.074 0.945 0.075 0.936 0.339 0.742 0.166 0.778 DPT-large [88] 1.4M 0.100 0.901 0.098 0.903 0.082 0.934 0.182 0.758 0.078 0.946 DepthAnything † [125] 63.5M 0.080 0.946 0.043 0.980 0.043 0.981 0.261 0.759 0.058 0.984 DepthAnything v2 † [126] 62.6M 0.080 0.943 0.043 0.979 0.042 0.979 0.321 0.758 0.066 0.983 Depth Pro † [7] -0.055 0.974 0.042 0.977 0.041 0.978 0.217 0.764 0.043 0.974 Metric3D v2 † [45] 16M 0.052 0.979 0.039 0.979 0.023 0.989 0.147 0.892 0.040 0.983 DiverseDepth [131] 320K 0.190 0.704 0.117 0.875 0.109 0.882 0.376 0.631 0.228 0.694 LeReS [132] 354K 0.149 0.784 0.090 0.916 0.091 0.917 0.271 0.766 0.171 0.777 HDN [134] 300K 0.115 0.867 0.069 0.948 0.080 0.939 0.246 0.780 0.121 0.833 GeoWizard [32] 280K 0.097 0.921 0.052 0.966 0.061 0.953 0.297 0.792 0.064 0.961 DepthFM [34] 63K 0.083 0.934 0.065 0.956 --0.225 0.800 --Marigold † [49] 74K 0.099 0.916 0.055 0.964 0.064 0.951 0.308 0.773 0.065 0.960 DMP Official † [56] -0.240 0.622 0.109 0.891 0.146 0.814 0.361 0.706 0.128 0.857 GeoWizard † [32] 280K 0.129 0.851 0.059 0.959 0.066 0.953 0.328 0.753 0.077 0.940 DepthFM † [34] 63K 0.174 0.718 0.082 0.932 0.095 0.903 0.334 0.729 0.101 0.902 Genpercept † [120] 90K 0.094 0.923 0.091 0.932 0.056 0.965 0.302 0.767 0.066 0.957 Painter † [115] 24K 0.324 0.393 0.046 0.979 0.083 0.927 0.342 0.534 0.203 0.644 Unified-IO † [71] 48K 0.188 0.699 0.059 0.970 0.063 0.965 0.369 0.708 0.103 0.906 4M-XL † [75] 759M 0.105 0.896 0.068 0.951 0.065 0.955 0.331 0.734 0.070 0.953 OneDiffusion † [55] 500K 0.101 0.908 0.087 0.924 0.094 0.906 0.399 0.661 0.072 0.949 Ours-single † 500K 0.064 0.952 0.066 0.953 0.077 0.942 0.283 0.717 0.052 0.971 Ours † 500K 0.069 0.949 0.061 0.960 0.072 0.944 0.289 0.722 0.050 0.975 20 15 10 5 0 5 10 15 IoU delta at 1 point IBD TrasCan Plittersdoff PIDRay WoodScape TimberSeg VISOR EgoHOS ZeroWaste LVIS GTEA STREETS iShape PPDLS ADE20k NDD20 OVIS CityScapes Hypersim DOORS BBBC038v1 NDISPark DRAM -16.7 -15.3 -14.9 -14.0 -12.4 -11.7 -10.2 -10.0 -4.3 -3.9 -1.5 -1.4 +0.1 +2.1 +2.9 +3.1 +4.1 +5.4 +7.1 +7.8 +9.1 +17.1 +18.8 Ours Ours-single SAM-vit-h Ours-single Ours SAM-vit-h 0 10 20 30 40 50 60 46.93 47.10 48.90 Total mIoU Comparison
Figure 3: Comparisons of mIoU with SAM-vit-h. We achieve results on par with SAM using only 0.06% of their data (600K vs. 1B). The performance of SAM is clearly better only on some datasets that are out-of-distribution for us, such as the Woodscape [133] Fisheye dataset. We compare the performance of specialized models, existing multi-task models, and our DI-CEPTION across various tasks. Specifically, we evaluate depth using the same protocol as Genpercept [120], normal estimation using the same method as StableNormal [129], interactive segmentation using the same approach as SAM [90], and human keypoints using the same method as Painter [115]. We also assess instance segmentation and entity segmentation on the MS COCO dataset. For entity segmentation, we assigned all predicted categories to the same label. As in Tables 1 and 2, our DICEPTION outperforms existing multi-task models and achieves performance on par with state-of-the-art specialized models or demonstrates only an acceptable performance decrease. Although some multitask methods achieve marginally better performance on certain datasets, such as Painter [115] and Unified-IO [70], they exhibit considerably poorer results on others such as outdoor settings (KITTI) and NYUv2 normal map benchmark. This further underscores the robust generalization capabilities of our approach. We contend that focusing on a model's performance across diverse datasets is more meaningful, as it better reflects the model's generalization ability and real-world applicability.
For interactive segmentation, as shown in Figure 3, we achieve results on par with SAM-vit-h using only 0.06% of their data. SAM shows a clear advantage only on certain out-of-distribution datasets that are outside the scope of our model's training, such as WoodScape fisheye dataset. Notably, while most specialized models require extensive data or complex data pipelines, our method achieves excellent results with significantly less data and no training data cherry-picking. Evaluation across diverse datasets highlights the strong in-the-wild generalization capability of our model, demonstrating that it does not overfit to the biases inherent in specific datasets.
We observe that, although our model generates high-quality visualizations for human pose and instance segmentation, the corresponding evaluation metrics remain relatively low. This is also observed on the evaluation of small objects in entity segmentation. We found that this is due to the errors introduced by the post-processing rather than our model's performance. In Appendix C, we provide a comprehensive explanation of the post-processing procedure and analyze the underlying causes of metrics degradation.
this section cite: ['b119', 'b32', 'b76', 'b23', 'b107', 'b132', 'b119', 'b128', 'b89', 'b114', 'b114', 'b69']

Section: Ablations and Analysis
Model designs, classifier-free guidance and pixel-aligned training. Our crucial analyses covering the elucidation of critical designs for effectively re-purposing diffusion models for perception tasks, as well as significant findings and insights, are detailed in the Appendix due to space limit. Specifically, the analysis of different architectures and input paradigms is presented in Appendix B.1, B.2 and B.3. The effectiveness of modest classifier-free guidance in improving results is discussed in Appendix B.4. The inherent few-step capability of flow-matching on perception tasks is analyzed in Appendix B.5. The benefits of pixel-aligned training are detailed in Appendix B.6 and B.7. Comparisons with Our Single-task Models. For the training of single-task models, we ensure that the network architecture remains the same and the total amount of training data seen for each specific task is the same as that for the multi-task model. For example, if the multi-task model is trained for 100 iterations with 4 depth data samples per batch, the single-task model will also be trained for 100 iterations with 4 data samples per batch. In our current data setting (approximately 1.8 million samples), we have not observed a significant gap between the multi-task and single-task models, nor have we seen a trend of mutual promotion between different tasks, as shown by "Ours-single" in Tables 1, 2, 5 and Figure 3. We believe that it is more appropriate to explore with larger datasets in order to draw more solid conclusions. We leave this as future work.
this section cite: []

Section: Multi-point Prompted Segmentation.
Ambiguity is a significant issue in interactive segmentation. For example, if a point is placed on a person's clothing, the model may segment the clothing, but the desired result is the person. Therefore, more points are needed to resolve this ambiguity. As illustrated in Table 6, additional points help the model better segment the desired results.
Table 6: Comparisons between 1-point and 5-point as input. 5 points are selected randomly. Method 1-point 5-point mIoU↑ 47.1 57.2 One-step Training and One-step Inference. Genpercept [120] demonstrates that diffusion model trained with one-step denoising significantly enhances both the speed and accuracy of perceptual tasks. However, our experimental results reveal a notable increase of failure cases when applying one-step diffusion in a multi-task setting, as illustrated in Figure 4. We believe that this is due to the potential overlap of denoising trajectories for different tasks. These overlapping trajectories can interfere with each other, resulting in failure cases with one-step inference. In contrast, in a single-task setting, since the denoising trajectories pertain to a single task, one-step is more effective and stable. However, we observe that our model, trained with multi-step denoising, can be applied directly to few-step inference with minimal degradation in performance. We provide results and more detailed analysis in Appendix B.5.
this section cite: ['b119']

Section: Conclusion
Figure 4: The model trained with 1-step denoising tends to produce more failure cases in multi-task scenarios.
We have introduced DICEPTION, a multi-task visual generalist model based on the diffusion model. Our approach unifies different tasks in the RGB space, leveraging the prior knowledge of pre-trained image generation model to achieve results that are on par with specialized foundation models. We achieve good performance without carefully cherry-picking extremely high-quality data or by using an exceptionally large amount of data. In few-shot finetuning, we are able to achieve high-quality results with minimal data and minimal trainable parameters.
Furthermore, we provide in-depth experimental analyses of strategies for transferring diffusion models to perception tasks. We also discuss the contributions of classifier-free guidance in enhancing model performance, demonstrate that there is no performance gap between our single-task and multi-task model, and highlight the improved detail preservation achieved through pixel-aligned perception training. We believe that DICEPTION sheds light on how to effectively use priors of diffusion models to build a strong visual generalist model.
this section cite: []

Section: References
Ref_id:b0 Title: Building normalizing flows with stochastic interpolants Year: (2022)
Ref_id:b1 Title: An any-to-any vision model for tens of tasks and modalities Year: (2024)
Ref_id:b2 Title: Rethinking inductive biases for surface normal estimation Year: (2024)
Ref_id:b3 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b4 Title: Zerowaste dataset: Towards deformable object segmentation in cluttered scenes Year: (2022)
Ref_id:b5 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b6 Title: Depth pro: Sharp monocular metric depth in less than a second Year: (2024)
Ref_id:b7 Title: Instructpix2pix: Learning to follow image editing instructions Year: (2023)
Ref_id:b8 Title: Language models are few-shot learners Year: (2020)
Ref_id:b9 Title: Nucleus segmentation across imaging experiments: the 2018 data science bowl Year: (2019)
Ref_id:b10 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b11 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b12 Title: Pix2video: Video editing using image diffusion Year: (2023)
Ref_id:b13 Title: Stablevideo: Text-driven consistency-aware diffusion video editing Year: (2023)
Ref_id:b14 Title: 3-d instance segmentation of mvs buildings Year: (2022)
Ref_id:b15 Title: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b16 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b17 Title: Grounded spatial reasoning in vision language model Year: (2024)
Ref_id:b18 Title: Maskedattention mask transformer for universal image segmentation Year: (2022)
Ref_id:b19 Title: Night and Day Instance Segmented Park (NDISPark) Dataset: a Collection of Images taken by Day and by Night for Vehicle Detection, Segmentation and Counting in Parking Areas Year: (2022-05)
Ref_id:b20 Title: Domain adaptation for traffic density estimation Year: (2021)
Ref_id:b21 Title: Semantic segmentation in art paintings Year: (2022)
Ref_id:b22 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b23 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b24 Title: Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100 Year: (2022)
Ref_id:b25 Title: Epic-kitchens visor benchmark: Video segmentations and object relations Year: (2022)
Ref_id:b26 Title: The llama 3 herd of models Year: (2024)
Ref_id:b27 Title: Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans Year: (2021)
Ref_id:b28 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b29 Title: Learning to recognize objects in egocentric activities Year: (2011)
Ref_id:b30 Title: Instance segmentation for autonomous log grasping in forestry operations Year: (2022)
Ref_id:b31 Title: Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image Year: (2024)
Ref_id:b32 Title: Vision meets robotics: The kitti dataset Year: (2013)
Ref_id:b33 Title: Fast monocular depth estimation with flow matching Year: (2024)
Ref_id:b34 Title: Regiongpt: Towards region understanding vision language model Year: (2024)
Ref_id:b35 Title: Animatediff: Animate your personalized text-to-image diffusion models without specific tuning Year: (2023)
Ref_id:b36 Title: Lvis: A dataset for large vocabulary instance segmentation Year: (2019)
Ref_id:b37 Title: Socrates: Introducing depth in visual wildlife monitoring using stereo vision Year: (2022)
Ref_id:b38 Title: Diffusion-based visual foundation model for high-quality dense prediction Year: (2024)
Ref_id:b39 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b40 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b41 Title: Video diffusion models Year: (2022)
Ref_id:b42 Title: Trashcan: A semantically-segmented dataset towards visual detection of marine debris Year: (2020)
Ref_id:b43 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b44 Title: Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation Year: (2024)
Ref_id:b45 Title: Oneformer: One transformer to rule universal image segmentation Year: (2023)
Ref_id:b46 Title: Taming multimodal llm for joint perception and understanding Year: (2024)
Ref_id:b47 Title: Imagic: Text-based real image editing with diffusion models Year: (2023)
Ref_id:b48 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b49 Title: Explora: Parameter-efficient extended pre-training to adapt vision transformers under domain shifts Year: (2024)
Ref_id:b50 Title: Segment anything Year: (2023)
Ref_id:b51 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b52 Title: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale Year: (2020)
Ref_id:b53 Title: Reasoning segmentation via large language model Year: (2024)
Ref_id:b54 Title: One diffusion to generate them all Year: (2024)
Ref_id:b55 Title: Exploiting diffusion prior for generalizable dense prediction Year: (2024)
Ref_id:b56 Title: Privacy-preserving portrait matting Year: (2021)
Ref_id:b57 Title: Bridging composite and real: towards end-to-end deep image matting Year: (2022)
Ref_id:b58 Title: Deep automatic natural image matting Year: (2021)
Ref_id:b59 Title: Matching anything by segmenting anything Year: (2024)
Ref_id:b60 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2025)
Ref_id:b61 Title: Delving into egocentric actions Year: (2015)
Ref_id:b62 Title: Photomaker: Customizing realistic human photos via stacked id embedding Year: (2024)
Ref_id:b63 Title:  Year: (2015)
Ref_id:b64 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b65 Title: Visual instruction tuning Year: (2024)
Ref_id:b66 Title: Video-p2p: Video editing with cross-attention control Year: (2024)
Ref_id:b67 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b68 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b69 Title: Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action Year: (2024)
Ref_id:b70 Title: Unified-io: A unified model for vision, language, and multi-modal tasks Year: (2022)
Ref_id:b71 Title: Backlitnet: A dataset and network for backlit image enhancement Year: (2022)
Ref_id:b72 Title: You see it, you got it: Learning 3d creation on pose-free videos at scale Year: (2024)
Ref_id:b73 Title: Finely-grained annotated datasets for image-based plant phenotyping Year: (2016)
Ref_id:b74 Title: 4m: Massively multimodal masked modeling Year: (2023)
Ref_id:b75 Title: T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models Year: (2024)
Ref_id:b76 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b77 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b78 Title: Kosmos-g: Generating images in context with multimodal large language models Year: (2023)
Ref_id:b79 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b80 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b81 Title: Doors: Dataset for boulders segmentation. statistical properties and blender setup Year: (2022)
Ref_id:b82 Title: Occluded video instance segmentation: A benchmark Year: (2022)
Ref_id:b83 Title: High-quality entity segmentation Year: (2022)
Ref_id:b84 Title: Unicontrol: A unified diffusion model for controllable visual generation in the wild Year: (2023)
Ref_id:b85 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b86 Title: Segment anything meets point tracking Year: (2023)
Ref_id:b87 Title: Vision transformers for dense prediction Year: (2021)
Ref_id:b88 Title: Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer Year: (2020)
Ref_id:b89 Title: Segment anything in images and videos Year: (2024)
Ref_id:b90 Title: A unified vision model for open-world object detection and understanding Year: (2024)
Ref_id:b91 Title: Pixellm: Pixel reasoning with large multimodal model Year: (2024)
Ref_id:b92 Title: Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding Year: (2021)
Ref_id:b93 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b94 Title: A multi-view stereo benchmark with high-resolution images and multicamera videos Year: (2017)
Ref_id:b95 Title: Learning temporally consistent video depth from video diffusion priors Year: (2024)
Ref_id:b96 Title: Benchmarking object detectors with coco: A new path forward Year: (2024)
Ref_id:b97 Title: Streets: A novel camera network dataset for traffic flow Year: (2019)
Ref_id:b98 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b99 Title: Deep high-resolution representation learning for human pose estimation Year: (2019)
Ref_id:b100 Title: Generative multimodal models are in-context learners Year: (2024)
Ref_id:b101 Title: Designing bert for convolutional networks Year: (2023)
Ref_id:b102 Title: Latent score-based reweighting for robust classification on imbalanced tabular data Year: (2025)
Ref_id:b103 Title: Decoding correlation-induced misalignment in the stable diffusion workflow for text-to-image generation Year: (2025)
Ref_id:b104 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b105 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b106 Title: Ndd20: A large-scale few-shot dolphin dataset for coarse and fine-grained categorisation Year: (2020)
Ref_id:b107 Title: A dense indoor and outdoor depth dataset Year: (2019)
Ref_id:b108 Title: Towards real-world prohibited item detection: A large-scale x-ray benchmark Year: (2021)
Ref_id:b109 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b110 Title: Dust3r: Geometric 3d vision made easy Year: (2024)
Ref_id:b111 Title: Framer: Interactive frame interpolation Year: (2024)
Ref_id:b112 Title: Autostory: Generating diverse storytelling images with minimal human efforts Year: (2024)
Ref_id:b113 Title: Hospitalscale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases Year: (2017)
Ref_id:b114 Title: Images speak in images: A generalist painter for in-context visual learning Year: (2023)
Ref_id:b115 Title: Segmenting everything in context Year: (2023)
Ref_id:b116 Title: Task-aware low-rank adaptation of segment anything model Year: (2024)
Ref_id:b117 Title: Lavin-dit: Large vision diffusion transformer Year: (2024)
Ref_id:b118 Title: A dynamic feature interaction framework for multi-task visual perception Year: (2023)
Ref_id:b119 Title: Diffusion models trained with large data are transferable visual models Year: (2024)
Ref_id:b120 Title: Vitpose: Simple vision transformer baselines for human pose estimation Year: (2022)
Ref_id:b121 Title: Paint by example: Exemplar-based image editing with diffusion models Year: (2023)
Ref_id:b122 Title: Depth any video with scalable synthetic data Year: (2024)
Ref_id:b123 Title: Haibin Huang, and Haoqiang Fan. ishape: A first step towards irregular shape instance segmentation Year: (2021)
Ref_id:b124 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b125 Title: Depth anything v2 Year: (2024)
Ref_id:b126 Title: From fidelity to perceptual quality: A semi-supervised approach for low-light image enhancement Year: (2020)
Ref_id:b127 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2024)
Ref_id:b128 Title: Stablenormal: Reducing diffusion variance for stable and sharp normal Year: (2024)
Ref_id:b129 Title: Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models Year: (2023)
Ref_id:b130 Title: Diversedepth: Affine-invariant depth prediction using diverse data Year: (2020)
Ref_id:b131 Title: Learning to recover 3d scene shape from a single image Year: (2021)
Ref_id:b132 Title: Woodscape: A multi-task, multi-camera fisheye dataset for autonomous driving Year: (2019)
Ref_id:b133 Title: Multi-view aggregation network for dichotomous image segmentation Year: (2024)
Ref_id:b134 Title: Hrformer: High-resolution transformer for dense prediction Year: (2021)
Ref_id:b135 Title: Fine-grained egocentric hand-object segmentation: Dataset, model, and applications Year: (2022)
Ref_id:b136 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b137 Title: Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport Year: (2025)
Ref_id:b138 Title: Moviedreamer: Hierarchical generation for coherent long visual sequence Year: (2024)
Ref_id:b139 Title: Convolution meets lora: Parameter efficient finetuning for segment anything model Year: (2024)
Ref_id:b140 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b141 Title: Storydiffusion: Consistent self-attention for long-range image and video generation Year: (2024)
Ref_id:b142 Title: Unleashing the potential of the diffusion model in few-shot semantic segmentation Year: (2024)
