Title: VID-LLM: A COMPACT VIDEO-BASED 3D MULTIMODAL LLM WITH RECONSTRUCTION-REASONING SYNERGY
Abstract: The room is a small office with desks arranged along the walls, a sofa in the center, and several chairs. It has a compact layout where most of the workspace is around the edges… Describe the writing desk on the right side of the room in detail. The desk is long and rectangular, placed against the wall with monitors, books, and … Give the coordinates of the chair right of the sofa. [1.13, 1.64, 0.35, 0.92, 0.03, 1.16]What furniture is placed in the center of the room?An orange sofa. How many chairs are in the room?There are six chairs in the room.

Section: ABSTRACT
Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers realscale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities. Project page: https://chenhaijier.github.io/Vid-LLM/.
this section cite: []

Section: INTRODUCTION
Recent advances in Large Language Models (LLMs) (Vaswani et al., 2017;Radford et al., 2019;Naveed et al., 2025) and Multimodal Large Language Models (MLLMs) (Zhang et al., 2024a;Yin et al., 2024;Wu et al., 2023) have reinforced the paradigm of language as a universal interface, substantially improving cross-modal perception and reasoning. Extending this progress to 3D, recent research has focused on 3D-aware Multimodal Large Language Models (3D-MLLMs) (Ren et al., 2025), which unify 3D scene understanding and vision-language reasoning under a linguistic interface. This line of work underscores the importance of grounding language in persistent 3D spatial representations (Cheng et al., 2024a;Roh et al., 2022), offering a unified pathway toward systematic scene-level reasoning.
Recent studies have made substantial progress in 3D vision-language (3D VL) reasoning (Chen et al., 2024c;Huang et al., 2023b), yet most approaches rely on complex 3D inputs, incurring high costs in data collection, preprocessing, and computation. Some models rely on point clouds or reconstructed scenes augmented with rendered views or semantic-geometric features (Hong et al., 2023a;Fu et al., 2024), while others adopt simpler inputs but still depend on explicit 3D scene representations such as reconstructed objects aligned with semantic representations (Chu et al., 2024;Huang et al., 2023a;2024). Despite their effectiveness, these pipelines depend on depth, poses, or external modules, leading to substantial data and engineering overhead as well as high memory and latency costs. This rigid input requirements and system complexity fundamentally limit the scalability and transferability of current 3D-MLLMs.
To overcome these limitations, a more general solution is to enable the model to directly reconstruct scene geometry from video (Leroy et al., 2024;Wang et al., 2024), thereby eliminating the reliance on external depth, pose, or registration modules. More importantly, reconstruction and reasoning are intrinsically interdependent: geometric structures underpin semantic understanding, while semantic reasoning, in turn, provides contextual priors that guide and refine geometric modeling (Cheng et al., 2024a;Ha & Song, 2022).
In this work, we introduce Vid-LLM, a compact model that jointly performs reconstruction and 3D vision-language reasoning from monocular video inputs, as illustrated in Fig. 1. The core component of Vid-LLM is a Cross-Task Adapter (CTA) that tightly couples reconstruction with reasoning, enabling intrinsic geometry-semantics interaction with mutual reinforcement and constraint. CTA disentangles geometry-aware and language-aware features; the geometric stream is then processed by a Global-Frame Attention backbone and specialized heads to estimate camera poses and relative depth, followed by a Metric Depth Model for real-scale calibration. The recovered 3D information is then fused with semantic features to construct 3D patches, which are fed into the LLM for spatial reasoning. Finally, a two-stage training strategy ensures convergence and improves overall performance. Extensive experiments across diverse 3D vision-language benchmarks demonstrate the performance of Vid-LLM and confirm its effectiveness as a practical and scalable framework for video-based 3D multimodal reasoning.
Our main contributions are summarized as follows:
• We propose Vid-LLM for versatile 3D scene understanding. The framework does not rely on dense 3D inputs or prior poses, making it practical for real-world deployment.
• We design a Cross-Task Adapter to align the 3D geometry priors with VL representations, boosting the integration of 3D visual geometry priors into MLLM. A two-stage training strategy is further adopted to improve the stability and performance.
• Extensive experimental evaluations are conducted on real datasets to evaluate the performance of our method. The experimental results demonstrate that our method achieves superior performance in terms of question answering, dense captioning and visual grounding. We will publish our code to facilitate communication.
2 RELATED WORK 3D-MLLMs have achieved significant advances in 3D scene understanding, yet their reliance on explicit 3D data still limits scalability and applicability. Meanwhile, progress in 3D reconstruction shows that geometry can be directly reconstructed from videos. Integrating such geometric priors into 3D-MLLMs represents a promising approach to enhance semantic grounding. We therefore review related work in three directions: (i) 3D-MLLMs, (ii) 3D reconstruction, and (iii) geometry priors in vision-language models.
this section cite: ['b60', 'b50', 'b45', 'b69', 'b65', 'b53', 'b54', 'b16', 'b14', 'b34', 'b63', 'b19']

Section: 3D-aware Multimodal Large Language Models (3D-MLLMs).
3D-MLLMs aim to unify 3D scene understanding and vision-language reasoning within a unified linguistic interface, representing an important extension of multimodal large language models (MLLMs). Existing approaches predominantly rely on explicit geometric inputs: some leverage point clouds or reconstructed scenes, often augmented with rendered views, region-level alignments, or condensed 3D feature grids to support large-scale embodied training (Hong et al., 2023a;Chen et al., 2024b;c;Fu et al., 2024;Huang et al., 2023b); others map 3D features to the language space and model spatial relations to enable interactive dialogue (Huang et al., 2023a;2024;Zheng et al., 2025b;Huang et al., 2025).
Despite their progress, these methods invariably depend on complex inputs such as point clouds, reconstructed scenes, multi-view renderings, or object-level annotations, which impose substantial burdens on data acquisition, preprocessing, and computation, thereby limiting scalability and transferability. In addition, the recent approach VGLLM (Zheng et al., 2025a) explores a video-based 3D-MLLM setting by adopting the geometry encoder of VGGT (Wang et al., 2025a) to extract 3D geometry features from video.
this section cite: ['b16', 'b26']

Section: 3D Reconstruction.
3D reconstruction has evolved from multi-view geometry pipelines to neural implicit representations and, more recently, feed-forward transformer-based architectures. Classical methods yield accurate geometry but require dense views and heavy preprocessing (Schonberger & Frahm, 2016;Furukawa et al., 2015). Neural radiance fields and point-based extensions improve fidelity and efficiency but focus mainly on appearance modeling while lacking semantic reasoning (Mildenhall et al., 2021;Barron et al., 2022;Kerbl et al., 2023). Recent feed-forward approaches enable direct prediction of depth, pose, and point clouds from video inputs (Wang et al., 2024;Leroy et al., 2024;Wang et al., 2025a). Nevertheless, 3D reconstruction is still only weakly integrated into vision-language research, and its role in supporting semantic reasoning remains underexplored.
this section cite: ['b56', 'b17', 'b43', 'b3', 'b29', 'b63', 'b34']

Section: Geometry Priors in Vision-Language Models.
Incorporating geometry priors has become a key approach for Vision-Language Models (VLMs) to enhance spatial understanding. Along a spectrum of reliance on explicit 3D inputs, existing methods can be organized into three categories: first, explicit input injection, which introduces depth, point clouds, or scene graphs as additional modalities to provide metric properties (Cai et al., 2025;Cheng et al., 2024b;Guo et al., 2023); second, internalization at the data and training level, which leverages spatially annotated corpora or geometric distillation to embed geometry implicitly into the alignment space, enabling spatial reasoning without explicit 3D inputs at inference (Chen et al., 2024a;Peng et al., 2023); and third, modular or prompt-based integration, which augments VLMs with lightweight modules or outputs from 3D foundation models, typically without large-scale retraining (Ma et al., 2024;Kerr et al., 2023). In contrast, our approach generates geometry through a video-driven reconstruction branch and achieves alignment with the semantic branch, enabling a structured and reusable integration of geometry priors within a video-based setting.
this section cite: ['b5', 'b18', 'b49', 'b41', 'b30']

Section: METHOD
We present Vid-LLM, a video-based 3D multimodal large language model (3D-MLLM). The main components are presented in the following sections: the Cross-Task Adapter is described in Section 3.1, the reconstruction and reasoning branches are detailed in Sections 3.2 and 3.3, and the training strategy is outlined in Section 3.4. The overall architecture is shown in Fig. 2.
this section cite: []

Section: CROSS-TASK ADAPTER
In Vid-LLM, we employ DINOv2 as the shared visual encoder to extract base tokens T base ∈ R N ×C from the input image sequence, where N denotes the number of tokens and C is the embedding dimension. To enhance feature effectiveness, we introduce a Cross-Task Adapter (CTA) that aligns 3D geometry priors with vision-language (VL) representations, facilitating the integration of geometric cues into multimodal reasoning.
. . . Vision Encoder (DINOv2) Global-Frame Attention × N layers DPT Decoder 1/2 1/4 1/8 1/16 1/32 Metric Bins Module Camera Head Dense Head Z X Y Camera pose Metric depth Point map 3D Position Embedding 3D Patches Pooling Asking Q: What object is placed in the center of the room? A:An orange sofa To adapt the shared visual representations for different branches, we employ two lightweight MLP projection heads, ϕ geom (•) and ϕ lang (•), which map the shared vision tokens to geometry-specific and semantic feature spaces, respectively:
T geom = ϕ geom (T base ), T lang = ϕ lang (T base )(1)
To effectively align 3D geometry priors with vision-language representations thus enhance the integration of spatial cues into multimodal reasoning, we introduce learnable Bridge Tokens, denoted as T bridge ∈ R K×C . Acting as shared memory units, the bridge tokens attend to geometric and semantic features separately, and the updated representation is formulated as:
T ′ bridge = Attn(T bridge , T f used geom , T f used geom ) + Attn(T bridge , T f used lang , T f used lang )(2)
where Attn(•) denotes a standard multi-head attention operation. This operation enables bridge tokens to dynamically capture complementary information from both tasks and update their representations during training. The joint propagation of geometric and semantic signals strengthens the alignment of 3D geometry priors with vision-language features, leading to more robust cross-modal representations.
Finally, the updated bridge tokens T ′ bridge are integrated into the feature streams, yielding enhanced task-specific representations T ′ geom and T ′ lang . These enriched features capture complementary geometric and semantic cues and are subsequently passed to the reconstruction and reasoning branches. In essence, the Cross-Task Adapter establishes intrinsic geometry-semantics interaction at the feature level, allowing the two streams to reinforce and guide each other for more robust representations.
this section cite: []

Section: 3D RECONSTRUCTION MODEL
In the reconstruction branch of Vid-LLM, we build on recent transformer-based architectures for end-to-end 3D reconstruction (Wang et al., 2025a) to recover scene geometry from video inputs. To additionally recover real-scale information, we design a Metric Depth Model that provides robust global scale cues, enabling reconstructions with both fine structural details and metric consistency.
this section cite: []

Section: Geometry Encoding and Prediction Heads.
Based on the cross-task enhanced geometric features T ′ geom , together with camera tokens T cam and register tokens T reg , the Global-Frame Attention backbone produces an integrated geometric representation, which is then fed into two prediction heads: a camera head estimating intrinsic-extrinsic parameters and a DPT head that predicts the relative depth map Drel ∈ R H×W , where H and W denote the image height and width, respectively.
this section cite: []

Section: Metric Depth Model.
To recover real-scale geometry, we equip the DINOv2 features with a DPTstyle decoder that produces multi-scale depth representations. Each pixel's depth is modeled using a bin-based formulation, where the probability p i (k) over the k -th bin and its refined center c i (k) jointly determine the prediction as
d(i) = N k=1 p i (k) c i (k).
we use an ordinal-aware normalization to capture relative depth ordering. To further stabilize scale, the bin centers c i (k) are adaptively refined as c i (k) = c k + ∆c i (k), where ∆c i (k) = r k (F i ) is predicted from the decoder features F i . The resulting metric depth map Dmetric = { d(i)} H×W i=1 provides global scale cues that are aligned with relative predictions for real-scale reconstruction.
Real-Scale Alignment. We estimate a scaling factor between the relative depth Drel predicted by the DPT head and the metric depth Dmetric predicted by the Metric Depth Model via weighted least squares. For each scene, 16 images are randomly sampled to compute per-image scaling factors, and their median is taken median as the final scene-level factor. This factor is then applied to convert both the relative depth and the predicted camera pose into real-world units. Rather than directly using metric depth as the final output, we adopt this alignment strategy since the DPT head provides more accurate texture details compared with the Metric Depth Model, which could also be observed from experimental results.
this section cite: []

Section: 3D VISION-LANGUAGE MODEL
In the reasoning branch of Vid-LLM, cross-task-enhanced semantic features T ′ lang are combined with reconstructed geometry to generate dense 3D patch representations, which are then fed into the LLM for 3D question answering, grounding, and captioning tasks.
this section cite: []

Section: 3D Patch Construction.
Each 2D feature T ′ lang (i, j) is back-projected into 3D using the estimated depth D, camera pose ( R, t) and intrinsics K produced by the 3D Reconstruction Model, yielding its camera-frame coordinates P v (i, j) as:
P v (i, j) = R-1 K -1 [i, j, 1] ⊤ D(i, j) -R-1 t(3)
These coordinates are encoded by an MLP into positional embeddings P ′ v (i, j) that match the dimensionality of the semantic features. Therefore, the final 3D patch tokens are then obtained by fusing geometry and semantic features:
T 3D (i, j) = T ′ lang (i, j) + P ′ v (i, j)(4)
This operation feeds spatial information into the semantic tokens, further enhancing the spatial awareness in the LLM.
this section cite: []

Section: TRAINING STRATEGY
We adopt a two-stage training strategy to utilize the shared encoder for both geometry and semantics.
Stage 1 performs dual-teacher distillation, transferring geometric priors from a reconstruction model and semantic knowledge from a multimodal LLM, enabling the encoder to learn both capabilities in a balanced way. Stage 2 jointly optimizes all downstream modules with 3D vision-language objectives, while incorporating auxiliary reconstruction losses to provide the model with sufficient reconstruction capability and ensure real-scale consistency. The overall pipeline is illustrated in Fig. 3.
this section cite: []

Section: Stage-1 Dual-Teacher Distillation.
In stage 1, we adopt a dual-teacher distillation strategy to jointly train the DINO encoder and the Cross-Task Adapter, enabling the modules to quickly learn both geometric and semantic representations. The pretrained DINO encoder and CLIP encoder serve as the geometry and semantic teachers in the distillation strategy, which are initialized from VGGT (Wang et al., 2025a) and LLaVA-3D (Zhu et al., 2025), respectively. The distillation loss is defined as:
L distill = L f eat geo + L f eat lang + λL sc(5)
where L f eat geo , L f eat lang and L sc are the geometry loss, semantic loss and structural consistency loss, respectively. λ is a balancing hyperparameter. For geometry learning, an L2 loss is applied to patch-level features, following DINO's patch regression strategy. For semantic learning, a cosine similarity loss is applied to pooled features to align with CLIP's global semantic embedding space. The specific forms are defined as follows: geom . Before computing L f eat geo and L f eat lang , we apply lightweight linear projection layers to align the features to the same embedding dimension. To maintain structural consistency, we also introduce the structural consistency loss L sc , which is defined as:
L f eat geo = 1 N N i=1 ∥Norm(T ′ geom,i ) -Norm(T geo tea,i )∥ 2 2 , L f eat lang = 1 -cos(Pool(T ′ lang ), Pool(T lang tea ))(6)
L sc = 1 M 2 ∥S stu -S tea ∥ 2 F , where S stu = Z stu Z ⊤ stu , S tea = Z tea Z ⊤ tea(7)
Z stu = [Norm(T ′ geom ); Norm(T ′ lang )] ∈ R M ×C is obtained by concatenating the geometry and semantic tokens from the student representation. Z tea = [Norm(T geo tea ); Norm(T lang tea )] is defined in the same way as Z stu , but using the teacher representation. ∥ • ∥ F is the Frobenius norm, M is the total number of tokens, and C is the embedding dimension of each token.
Stage-2 Joint Optimization. In Stage 2, we further fine-tune all the modules to optimize overall performance. The joint loss is defined as:
L joint = L recon-task + L V L-task + L M D(8)
where L recon-task is the multi-task loss for 3D reconstruction, consisting of the camera loss, depth loss, and point map loss following Wang et al. (2025a). L V L-task supervises 3D vision-language reasoning, including cross-entropy loss for instruction-following tasks, along with bounding box regression and matching losses for grounding (Zhu et al. (2025)). L M D represents the metric depth loss, combining a global scale penalty and a robust local refinement term, and is defined as:
L MD = b 2 + 1 K K i=1 (e i -b) 2 1 + α|e i -b| , (9
)
The log-depth error is defined as
e i = log(d pred i + ε) -log(d gt i + ε),
where ε is a small constant for numerical stability. b = 1 K K i=1 e i is the mean error across all K valid pixels in the image. The parameter α > 0 controls the robustness by down-weighting large residuals.
During joint training, L recon-task and L M D optimize the 3D reconstruction model, L V L-task optimize the 3D vision-language model, while the shared CTA is jointly optimized by L recon-task and L V L-task . It is worth to noticing that the loss constructed in the 3D-VL reasoning branch is not used to optimize the 3D reconstruction model, which can be seen in Fig. 2. This one-way gradient flow ensures that the CTA acts as a stable bridge for geometry-semantics interaction, enabling effective feature exchange.
Table 1: Evaluation of 3D Question Answering on ScanQA and SQA3D. Methods marked with * are 3D MLLM evaluated in video mode.
† indicates the model consumes VGGT-generated 3D geometry. "C" stands for "CIDEr", "B-4" for "BLEU-4", "M" for "METEOR", "R" for "ROUGE", and "EM@1" for top-1 exact match.
this section cite: []

Section: EXPERIMENTAL SETUP
Training Details. We adopt DINOv2-L as the visual backbone, consisting of 24 Transformer layers with a hidden dimension of 1024. The projection MLPs use two fully connected layers with an expansion factor of 4 and GELU activation. For data processing, we uniformly sample 32 frames per scene, resize the shorter side to 518, crop the resolution so that both height and width are multiples of 14. Optimization is performed using AdamW with β 1 = 0.9, β 2 = 0.999, and weight decay of 0.05, with gradient clipping at 1.0.
Evaluation Details. We follow the standard evaluation protocols and metrics defined for each dataset. For tables that summarize performance across multiple datasets (e.g., Tables 5 and 6), the reported numbers are computed as the mean of the metrics for each benchmark. For the Scan2Cap dataset, we follow prior work (Zhu et al., 2025;Zheng et al., 2025a): instance proposals are generated using Mask3D (Schult et al., 2022), and 3D coordinate tokens are constructed from the predicted instance centers to enable instance-aware caption generation. For the ScanRefer dataset, Mask3D (Schult et al., 2022) serves as the 3D segmentor to remain consistent with supervised baselines (Huang et al., 2024;Chen et al., 2024c;Huang et al., 2023a). For the Nr3D and Sr3D datasets, the provided segmentation annotations are adopted to align with previous methods (Chen et al., 2022;Huang et al., 2023a). For reconstruction evaluation on ScanNet, the metrics are computed in metric scale. All models are trained with the same data and settings for a fair comparison.
this section cite: ['b57', 'b57', 'b23', 'b8']

Section: 3D VISION-LANGUAGE REASONING
Overview. To comprehensively assess the performance of Vid-LLM on 3D vision-language reasoning tasks, we conduct experiments using widely adopted datasets covering three task categories: 3D Question Answering on ScanQA (Azuma et al., 2022) and SQA3D (Ma et al., 2022), 3D Dense Captioning on Scan2Cap (Chen et al., 2021), and 3D Visual Grounding on ScanRefer (Chen et al., 2020), Multi3DRefer (Zhang et al., 2023), and Nr3D/Sr3D (Achlioptas et al., 2020b).
Baseline. We compare Vid-LLM with a broad set of 3D-based and video-based vision-language reasoning models. 3D-based methods rely on explicit 3D scene inputs such as point clouds or reconstructed geometry, whereas video-based methods operate solely on video. The 3D-based baselines include ScanQA (Azuma et al., 2022), Scan2Cap (Chen et al., 2021), 3D-VisTA (Zhu et al., 2023), 3D-LLM (Hong et al., 2023b), LL3DA (Chen et al., 2024b), Grounded3D-LLM (Chen et al., 2024c), Chat-3D v2 (Huang et al., 2023a), LEO (Huang et al., 2023b), Scene-LLM (Fu et al., 2024), ChatScene (Huang et al., 2024), LLaVA-3D (Zhu et al., 2025), Video-3D LLM (Zheng et al., 2025b), and 3DRS (Huang et al., 2025). The video-based baselines include VILA-40B (Lin et al., 2024), IXC (Zhang et al., 2024b), LLaVA-OV (Li et al., 2024), LLaVA-Video (LLaVA Team, 2024), Uni3DR2 (Chu et al., 2024), and VGLLM (Zheng et al., 2025a). To further assess the grounding capability of Vid-LLM, we additionally compare against task-specific 3D visual grounding models, including ScanRefer (Chen et al., 2020), 3D-VisTA (Zhu et al., 2023), ReferIt3D (Achlioptas et al., 2020a), 3DVG-Trans (Zhao et al., 2021), MVT (Huang et al., 2022), ViL3DRel (Chen et al., 2022), and SceneVerse (Jia et al., 2024). To allow comparison under the same video input setting with the most competitive 3D-based baseline in our evaluation, we additionally include 3DRS † , a variant of 3DRS that takes VGGT-generated 3D geometry as input. These comparisons collectively provide a comprehensive evaluation of Vid-LLM across 3D vision-language reasoning tasks.
this section cite: ['b2', 'b42', 'b11', 'b7', 'b73', 'b2', 'b11', 'b80', 'b16', 'b23', 'b26', 'b35', 'b14', 'b7', 'b80', 'b74', 'b25', 'b8', 'b27']

Section: Result & Analysis.
Vid-LLM consistently shows robust performance across all 3D vision-language reasoning benchmarks. As shown in Tab. 1, Vid-LLM achieves the best results among video-based models on both ScanQA and SQA3D, outperforming the second-best baseline (3DRS † ) by an average margin of 11%. Tab. 2 further shows that Vid-LLM attains the highest performance among video-based methods on Scan2Cap; notably, its M@0.5 score (28.7) is close to that of the bestperforming 3D-based model (29.0), despite not using depth or point clouds. These results collectively indicate that the integration of geometric cues into semantic reasoning can significantly improve perception performance. Tab. 3 and Tab. 4 present results on ScanRefer/Multi3DRefer and Nr3D/Sr3D. Vid-LLM achieves the best performance across all metrics on the two grounding benchmarks, surpassing all comparable 3D-based and video-based counterparts. These results demonstrate that the proposed Cross-Task Adapter, which effectively integrates semantic and geometric information, enables effective spatial reasoning and yields substantial gains on geometry-intensive tasks such as 3D visual grounding. For qualitative analysis, we visualize the 3D grounding results on the ScanRefer dataset in Fig. 4, which further demonstrates the 3D visual grounding capability of Vid-LLM. More qualitative visualizations for 3D VL tasks can be found in Appendix A.1.
this section cite: []

Section: COMPARISON WITH JOINT RECONSTRUCTION-REASONING MODELS
Overview. In addition to 3D vision-language reasoning, Vid-LLM also incorporates a reconstruction branch that provides geometric priors and metric-scale cues. This naturally raises the question of how different design choices for coupling reconstruction and reasoning affect performance when both tasks must be performed from a single video input. To provide a comprehensive analysis, we compare Vid-LLM with two representative categories of joint reconstruction-reasoning baselines: (i) simple concatenation pipelines that feed the 3D geometry reconstructed from video into a 3D multimodal LLM, and (ii) end-to-end architectures that integrate reconstruction and 3D VL reasoning within a single model.
Baseline. For the concatenation baselines, we construct pipelines that feed the predicted geometry information of VGGT into LLaVA-3D (Zhu et al., 2025) for 3D vision-language reasoning. Three variants of the concatenation pipeline are evaluated: (i) VGGT+LLaVA-3D, the direct combination of VGGT and LLaVA-3D; (ii) VGGT+LLaVA-3D † , which introduces metric-scale alignment to the predictions of VGGT; (iii) VGGT+LLaVA-3D † ‡ , which further fine-tunes LLaVA-3D using depth and camera poses predicted by VGGT as geometric supervision. For end-to-end joint-task baselines, we include Uni3DR 2 (Chu et al., 2024), an end-to-end framework that integrates a reconstruction  module with a 3D VL reasoning branch. Notably, Uni3DR 2 requires ground-truth camera poses as input, whereas Vid-LLM performs both tasks directly from video, making the comparison conservative in favor of Uni3DR 2 . We also include VGLLM-Rec, an extension of VGLLM (Zheng et al., 2025a). To support joint reconstruction and reasoning, we reconnect the original camera and depth prediction heads of VGGT to the geometry encoder in VGLLM, enabling VGLLM-Rec to generate geometric predictions in addition to performing 3D VL reasoning from video.
this section cite: ['b14']

Section: Result & Analysis.
As shown in Tab. 5, VGGT+LLaVA-3D obtains very low 3D VL accuracy, indicating that relative-scale geometry cannot provide reliable cues for 3D VL reasoning. Building on this, VGGT+LLaVA-3D † shows a clear performance gain once metric-scale alignment is applied. However, even with this setting, the concatenation baseline still underperforms Vid-LLM on all 3D VL tasks despite achieving a higher reconstruction score (0.591 vs. 0.582). This indicates that simply providing accurate reconstructed geometry to a 3D-LLM is insufficient and that effective geometry-semantics interaction is necessary for reliable reasoning performance. Furthermore, VGGT+LLaVA-3D † ‡ yields a slight drop in 3D VL accuracy compared with VGGT+LLaVA-3D † , suggesting that training the 3D-LLM with noisy predicted geometry may introduce biases that degrade the semantic representations of the framework. In contrast to concatenation pipelines, endto-end joint-task models avoid multi-stage processing and achieve lower inference latency, reducing runtime from 2.7 s/scene to 1.6-2.1 s/scene. From Tab. 5, we can observe that Vid-LLM achieves the best overall performance among the joint models on both reconstruction and 3D VL tasks. It marginally outperforms Uni3DR² in reconstruction quality on ScanNet (0.582 vs. 0.580) despite Uni3DR² having access to ground-truth camera poses. On the 3D VL tasks, Vid-LLM demonstrates a clear advantage, with an average relative improvement of 47.6% over Uni3DR² and 12.1% over VGLLM-Rec across ScanQA, Scan2Cap, and ScanRefer. Overall, Vid-LLM delivers the fastest inference speed and the best 3D VL performance, while also achieving the best reconstruction quality among joint-task models. These gains stem from our integrated architecture and the Cross-Task Adapter, which facilitates geometry-semantics interaction and allows the model to effectively leverage geometric cues during joint reasoning. More experimental results about reconstruction performance can be found in Appendix A.2. Figure 5: Test loss vs. data size across training strategies.
this section cite: []

Section: ABLATION STUDIES
Cross-Task Adapter. To assess the contribution of the Cross-Task Adapter, we compare several configurations: removing the CTA module (w/o CTA); in the setting without bridge tokens, applying self-attention to the concatenated T geom and T lang (CTA-Concat-SA) and performing cross-attention between the two feature sets (CTA-w/o Bridge (CA)); and finally using the full CTA with bridge tokens. We also analyze how different numbers of bridge tokens (4, 8, 16, and 32) affect model performance. As shown in Tab. 6, w/o CTA leads to the lowest performance on both tasks, confirming that a shared visual representation alone cannot jointly support reconstruction and 3D VL reasoning. Both CTA-Concat-SA and CTA-w/o Bridge(CA) show clear improvements, yet their effectiveness is limited as neither design provides a stable shared latent space for passing geometric cues to semantic features. The configurations using the full CTA module achieve clear performance improvements, since the bridge tokens provide a dedicated latent space that enables consistent geometry-semantic alignment and more effective cross-task interaction. Among different token counts, using 16 bridge tokens achieves the best balance between accuracy and model complexity.
this section cite: []

Section: Metric Depth Modules.
Tab. 6 also reports ablations on the use of metric depth and our alignment strategy. Without metric depth supervision (w/o MD), scale ambiguity destroys geometric consistency and leads to near failure of 3D VL reasoning. Incorporating metric depth without alignment (MD -w/o Alignment) preserves metric scale but produces less accurate reconstruction, which in turn limits the performance of 3D VL tasks. By contrast, combining metric depth with our scale alignment strategy effectively exploits both global scale cues and relative structural details, yielding the most accurate reconstruction and consistently more reliable results on 3D VL tasks.
this section cite: []

Section: Two-Stage Training.

this section cite: []

Section: CONCLUSION
In this work, we present Vid-LLM, a video-based 3D Multimodal Large Language Model (3D-MLLM). Our compact architecture extracts geometric cues from video and feeds them into the LLM through a 3D patch construction strategy to accomplish spatial reasoning. A central component of our framework is the Cross-Task Adapter that aligns 3D geometry priors with vision-language representations. This design enhances their integration into the MLLM and improves the robustness of reasoning under uncertain geometry. With a two-stage training strategy, our model achieves greater training stability and faster convergence. Extensive experiments on 3D vision-language benchmarks demonstrate that Vid-LLM achieves state-of-the-art results on several benchmarks and remains competitive on the others, while ablation studies validate the effectiveness of each component.
this section cite: []

Section: References
Ref_id:b0 Title: Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes Year: (2020)
Ref_id:b1 Title: Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes Year: (2020)
Ref_id:b2 Title: Scanqa: 3d question answering for spatial scene understanding Year: (2022)
Ref_id:b3 Title: Mip-nerf 360: Unbounded anti-aliased neural radiance fields Year: (2022)
Ref_id:b4 Title: Zoedepth: Zeroshot transfer by combining relative and metric depth Year: (2023)
Ref_id:b5 Title: Spatialbot: Precise spatial understanding with vision language models Year: (2025)
Ref_id:b6 Title: Spatialvlm: Endowing vision-language models with spatial reasoning capabilities Year: (2024)
Ref_id:b7 Title: Scanrefer: 3d object localization in rgb-d scans using natural language Year: (2020)
Ref_id:b8 Title: Language conditioned spatial relation reasoning for 3d object grounding Year: (2022)
Ref_id:b9 Title: Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning Year: (2024)
Ref_id:b10 Title: Grounded 3d-llm with referent tokens Year: (2024)
Ref_id:b11 Title: Scan2cap: Context-aware dense captioning in rgb-d scans Year: (2021)
Ref_id:b12 Title: Spatialrgpt: Grounded spatial reasoning in vision-language models Year: (2024)
Ref_id:b13 Title: Spatialrgpt: Grounded spatial reasoning in vision-language models Year: (2024)
Ref_id:b14 Title: Unified scene representation and reconstruction for 3d large language models Year: (2024)
Ref_id:b15 Title: Scannet: Richly-annotated 3d reconstructions of indoor scenes Year: (2017)
Ref_id:b16 Title: Scene-llm: Extending language model for 3d visual understanding and reasoning Year: (2024)
Ref_id:b17 Title: Multi-view stereo: A tutorial Year: (2015)
Ref_id:b18 Title: Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following Year: (2023)
Ref_id:b19 Title: Semantic abstraction: Open-world 3d scene understanding from 2d vision-language models Year: (2022)
Ref_id:b20 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b21 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b22 Title: Chat-3d v2: Bridging 3d scene and large language models with object identifiers Year: (2023)
Ref_id:b23 Title: Chat-scene: Bridging 3d scene and large language models with object identifiers Year: (2024)
Ref_id:b24 Title: An embodied generalist agent in 3d world Year: (2023)
Ref_id:b25 Title: Multi-view transformer for 3d visual grounding Year: (2022)
Ref_id:b26 Title: Mllms need 3d-aware representation supervision for scene understanding Year: (2025)
Ref_id:b27 Title: Sceneverse: Scaling 3d vision-language learning for grounded scene understanding Year: (2024)
Ref_id:b28 Title: Dg-recon: Depth-guided neural 3d scene reconstruction Year: (2023)
Ref_id:b29 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b30 Title: Lerf: Language embedded radiance fields Year: (2023)
Ref_id:b31 Title: Robust consistent video depth estimation Year: ()
Ref_id:b32 Title: Posediff: Pose-conditioned multimodal diffusion model for unbounded scene synthesis from sparse inputs Year: (2024)
Ref_id:b33 Title: Globally consistent video depth and pose estimation with efficient test-time training Year: (2022)
Ref_id:b34 Title: Grounding image matching in 3d with mast3r Year: (2024)
Ref_id:b35 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b36 Title: On pre-training for visual language models Year: ()
Ref_id:b37 Title: Pixel-perfect structure-from-motion with featuremetric refinement Year: ()
Ref_id:b38 Title: Geometric prior-guided self-supervised learning for multi-view stereo Year: ()
Ref_id:b39 Title: Swin transformer v2: Scaling up capacity and resolution Year: (2022)
Ref_id:b40 Title: Llava-video: Video instruction tuning with synthetic data Year: (2024)
Ref_id:b41 Title: Spatialpin: Enhancing spatial reasoning capabilities of vision-language models through prompting and interacting 3d priors Year: (2024)
Ref_id:b42 Title: Sqa3d: Situated question answering in 3d scenes Year: (2022)
Ref_id:b43 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b44 Title: Atlas: End-to-end 3d scene reconstruction from posed images Year: (2020)
Ref_id:b45 Title: A comprehensive overview of large language models Year: (2025)
Ref_id:b46 Title: All in tokens: Unifying output space of visual tasks via soft token Year: ()
Ref_id:b47 Title: Mvd-net: Semantic segmentation of cataract surgery using multi-view learning Year: (2022)
Ref_id:b48 Title: P3depth: Monocular depth estimation with a piecewise planarity prior Year: ()
Ref_id:b49 Title: Openscene: 3d scene understanding with open vocabularies Year: (2023)
Ref_id:b50 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b51 Title: Vision transformers for dense prediction Year: (2021)
Ref_id:b52 Title: Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction Year: (2021)
Ref_id:b53 Title: A survey of language-grounded multimodal 3d scene understanding. Knowledge-Based Systems Year: (2025)
Ref_id:b54 Title: Languagerefer: Spatial-language model for 3d visual grounding Year: (2022)
Ref_id:b55 Title: Superglue: Learning feature matching with graph neural networks Year: (2020)
Ref_id:b56 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b57 Title: Mask3d: Mask transformer for 3d semantic instance segmentation Year: (2022)
Ref_id:b58 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b59 Title: Neuralrecon: Real-time coherent 3d reconstruction from monocular video Year: (2021)
Ref_id:b60 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b61 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b62 Title: Continuous 3d perception model with persistent state Year: (2025)
Ref_id:b63 Title: Dust3r: Geometric 3d vision made easy Year: (2024)
Ref_id:b64 Title: Panorecon: Real-time panoptic 3d reconstruction from monocular video Year: (2024)
Ref_id:b65 Title: Multimodal large language models: A survey Year: (2023)
Ref_id:b66 Title: Frozenrecon: Posefree 3d scene reconstruction with frozen depth models Year: (2023)
Ref_id:b67 Title: Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass Year: (2025)
Ref_id:b68 Title: Depth anything: Unleashing the power of large-scale unlabeled data Year: (2024)
Ref_id:b69 Title: A survey on multimodal large language models Year: (2024)
Ref_id:b70 Title: Mmllms: Recent advances in multimodal large language models Year: (2024)
Ref_id:b71 Title: Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output Year: (2024)
Ref_id:b72 Title: Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views Year: ()
Ref_id:b73 Title: Multi3drefer: Grounding text description to multiple 3d objects Year: (2023)
Ref_id:b74 Title: 3dvg-transformer: Relation modeling for visual grounding on point clouds Year: (2021)
Ref_id:b75 Title: Unleashing textto-image diffusion models for visual perception Year: ()
Ref_id:b76 Title: Enhancing mllms with 3d vision geometry priors Year: (2025)
Ref_id:b77 Title: Video-3d llm: Learning position-aware video representation for 3d scene understanding Year: (2025)
Ref_id:b78 Title: Stereo magnification: Learning view synthesis using multiplane images Year: (2018)
Ref_id:b79 Title: Llava-3d: A simple yet effective pathway to empowering lmms with 3d capabilities Year: (2025)
Ref_id:b80 Title: 3d-vista: Pretrained transformer for 3d vision and text alignment Year: (2023)
