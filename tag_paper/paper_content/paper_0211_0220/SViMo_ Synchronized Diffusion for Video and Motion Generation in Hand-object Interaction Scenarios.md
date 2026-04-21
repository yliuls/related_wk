Title: SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios
Abstract: Hand-Object Interaction (HOI) generation has significant application potential. However, current 3D HOI motion generation approaches heavily rely on predefined 3D object models and lab-captured motion data, limiting generalization capabilities. Meanwhile, HOI video generation methods prioritize pixel-level visual fidelity, often sacrificing physical plausibility. Recognizing that visual appearance and motion patterns share fundamental physical laws in the real world, we propose a novel framework that combines visual priors and dynamic constraints within a synchronized diffusion process to generate the HOI video and motion simultaneously. To integrate the heterogeneous semantics, appearance, and motion features, our method implements tri-modal adaptive modulation for feature aligning, coupled with 3D full-attention for modeling inter-and intra-modal dependencies. Furthermore, we introduce a vision-aware 3D interaction diffusion model that generates explicit 3D interaction sequences directly from the synchronized diffusion outputs, then feeds them back to establish a closed-loop feedback cycle. This architecture eliminates dependencies on predefined object models or explicit pose guidance while significantly enhancing video-motion consistency. Experimental results demonstrate our method's superiority over state-of-the-art approaches in generating high-fidelity, dynamically plausible HOI sequences, with notable generalization capabilities in unseen real-world scenarios. Project page at https://droliven.github.io/SViMo_project.

Section: Introduction
Human-object or hand-object interaction (HOI) generation serves critical applications across gaming, animation, digital human creation, and robotic action retargeting [16,37,53,58]. Some studies [7,13,72] construct high-precision 3D interaction datasets through laboratory-based multiview camera arrays and motion capture systems, then train diffusion-based motion generators. These object-centric approaches typically predict parametric human/hand motions and the corresponding object pose sequences given object meshes and initial configurations. However, existing datasets [39,71,38] collected in laboratory environments lack diversity in object types and interaction patterns, constraining model generalization and resulting in ambiguous object boundaries, implausible or inconsistent actions (Fig. 1, left). Moreover, the reliance on precise 3D object models fundamentally limits their zero-shot generation capabilities.
Figure 1: Different HOI generation methods. Approaches like MDM [57] rely on limited mocap data without visual guidance, resulting in blurred boundaries and compromised plausibility and consistency. Methods like Animate Anyone [21] leverage large-scale visual priors but exhibit distortions and inconsistencies because of inadequate physical awareness. Our method marries visual priors with 3D motion constraints and eliminates dependency on pre-defined object models or pose guidance.
Recent advances in large video foundation models based on Diffusion Transformers (DiT) [44] (e.g., Sora [5], CogVideo [20,69], HunyuanVideo [26]), have shown impressive capabilities in modeling physical dynamics through large-scale video training. These models can generate interaction videos with high visual fidelity end-to-end from text or reference images. However, their pixel-level generation approaches often struggle to produce accurate and coherent hand-object interactions due to limited explicit modeling of motion dynamics and physical constraints. To address this, some methods extend image-based diffusion models (e.g., SVD [3]) by adding pose-guided pipelines [66,21,75,65]. These approaches combine pose conditions and appearance features to improve human-object interaction generation. While effective, they require pose sequences or externally estimated motion trajectories as inputs, preventing full end-to-end text/image-conditioned generation. Additionally, their single-frame generation leads to poor temporal coherence, causing flickering and identity inconsistencies (Fig. 1, middle).
These challenges reveal a longstanding methodological contradiction: motion generation systems excel at physical constraint modeling but suffer from limited data scales. In contrast, video generation models leverage massive visual priors but lack motion plausibility. We argue that this division stems from neglecting the co-evolution mechanism between visual appearance and motion patterns: they not only share the same foundation of physical dynamics but also could leverage similar diffusion processes. Based on this insight, we propose SViMo, a Synchronized Video-Motion diffusion framework that enables synchronous HOI video generation and motion synthesis within a unified architecture (Fig. 1, right). The core innovation lies in extending a pretrained image-to-video foundation model into the multimodal joint generation framework through the scalable DiT architecture. To better integrate the heterogeneous features of text semantics, visual appearance, and motion dynamics, we introduce the triple modality adaptive modulation to align feature scales and employ a 3D full-attention mechanism to learn their synergistic and complementary dependencies. Additionally, it is difficult for video foundation models to learn explicit 3D interaction motions directly. To bridge the representation gap and reduce optimization complexity, we project 3D motions onto 2D image planes, constructing "rendered motion videos" as SViMo's motion representation.
To further enhance the video-motion consistency, we design a Vision-aware 3D Interaction Diffusion model (VID). This model generates explicit 3D hand poses and object point clouds using denoised latent codes from the synchronized diffusion, which are then reinjected into the SViMo as interaction guidance and gradient constraints. Unlike methods requiring pre-specified action series, our approach integrates video synthesis and 3D interaction generation within an end-to-end denoising pipeline. This creates a closed-loop feedback mechanism where motion guidances refine video generation while video latents update motion results, enabling synergistic co-evolution of both modalities.
In summary, our contributions are threefold:
• A novel synchronized diffusion model for joint HOI video and motion denoising, effectively integrating large-scale visual priors with motion dynamic constraints.
• A vision-aware 3D interaction diffusion that generates explicit 3D interaction sequences, forming a closed-loop optimization pipeline and enhancing video-motion consistency.
• Our method generates HOI video and motion synchronously without requiring pre-defined poses or object models. Experiment results demonstrate superior visual quality, motion plausibility, and generalization capability to unseen real-world data.
2 Related Work 3D interaction synthesis relies on high-precision motion capture datasets, some of which focus on human action conditioned on static objects [18,73], while others simultaneously capture interactions of both human and dynamic objects [8,39,67,71,38,54,14]. Building upon these datasets, existing 3D interaction generation methods employ diffusion models to either introduce intermediate contact maps or milestones for modeling kinematic features [7,13,29,46,45,72,62,28,27,30,36,31,37] or leverage physical simulations to ensure physical dynamics plausibility [64,60,4,63,41]. However, due to the limited availability of 3D interaction data, the generalization capability of these approaches remains constrained. In contrast, our method leverages large-scale visual priors, operates conveniently without requiring 3D object models, and demonstrates promising generalization potential.
this section cite: ['b15', 'b36', 'b52', 'b57', 'b6', 'b12', 'b71', 'b38', 'b70', 'b37', 'b56', 'b20', 'b43', 'b4', 'b19', 'b68', 'b25', 'b2', 'b65', 'b20', 'b74', 'b64', 'b17', 'b72', 'b7', 'b38', 'b66', 'b70', 'b37', 'b53', 'b13', 'b6', 'b12', 'b28', 'b45', 'b44', 'b71', 'b61', 'b27', 'b26', 'b29', 'b35', 'b30', 'b36', 'b63', 'b59', 'b3', 'b62', 'b40']

Section: Interaction Video Generation.
The success of image generation models [19,[50][51][52] has inspired significant advances in video generation. Several approaches [3, 66, 43, 21, 75, 65, 61] extend 2D image denoising U-Nets to video by incorporating temporal attention layers, while improving controllability via pose guidance and reference networks. Alternatively, native large video models [5,20,69,26,25] directly generate videos using spatiotemporal DiT architectures [44]. A separate line of work aims to generate both video and interactive motion dynamics [9,11]. For instance, VideoJAM [9] produces video frames alongside 2D optical flow to improve visual fidelity and motion coherence. In contrast, our method generates videos with explicit 3D motion representation. This enables direct modeling of object geometry, scale, and spatial relationships, leading to superior physical plausibilityparticularly in complex scenarios involving occlusions or intricate interactions. Notably, Vlogger [11] employs a two-stage pipeline: it first generates 3D human poses using one network, then synthesizes video via a separate model conditioned on the predicted motions. In contrast, we unify video and motion generation within a single, synchronized diffusion framework. Since human-object interaction videos and their corresponding motion dynamics inherently obey the same physical laws, joint modeling allows the system to learn intrinsic co-evolutionary patterns, thereby enhancing the quality and consistency of both modalities.
this section cite: ['b18', 'b49', 'b50', 'b51', 'b4', 'b19', 'b68', 'b25', 'b24', 'b43', 'b8', 'b10', 'b8', 'b10']

Section: Multimodal Generative Models.
Driven by advancements in vision-language models [10,34,59,2,68], researchers have explored versatile generative models that align with the multimodal essence of the physical world. Some works [23,70] cascade single-modality generators into asynchronous pipelines, yet suffer from complex workflows and noise accumulation. Others develop end-to-end multimodal joint generation frameworks that rely on massive aligned multimodal data [1,32,17,76], indirect bridging mechanisms [55], or intricate hierarchical attention strategy [35] to ensure crossmodal synchronization. Differently, our method extends a native large video model into synchronized video-motion generation systems, aligning heterogeneous modality features through multimodal adaptive modulation and closed-loop feedback strategies.
this section cite: ['b9', 'b33', 'b58', 'b1', 'b67', 'b22', 'b69', 'b0', 'b31', 'b16', 'b75', 'b54', 'b34']

Section: Methodology
We define the HOI video and motion generation task as follows: Given a reference image frame I ∈ R H×W ×3 and a textual prompt P , generate the future video V ∈ R N ×H×W ×3 and 3D motion sequence M = {(h i , o i )} N i=1 with N time steps. Here h i ∈ R J×3 and o i ∈ R K×3 are the hand joints trajectories and object point clouds, each has J and K nodes respectively. The following sections detail each component of our approach. See the Appendix for training/inference pseudocode and additional details.
this section cite: []

Section: Preliminary: Basic Large Video Generation Model
The core of large video generation models is diffusion and its reverse denoising process [19]. The diffusion process adds Gaussian noise to video latent codes z step-by-step using q(z t |z t-1 ) = N (z t ; √ α t z t-1 , β t I), where β t ∈ [0, 1] increases monotonically with time step t, and α t = 1 -β t .
Repeated application of this formula yields the marginal distribution:
q(z t |z 0 ) = N (z t ; √ ᾱt z 0 , (1 -ᾱt )I), ᾱt = t 1 (1 -β t ). (1
)
The above distribution converges to an isotropic Normal distribution when t grows large.
Correspondingly, the reverse denoising generator G with trainable weights θ predicts the clean latent code ẑ0,θ , and is optimized by:
L x0-prediction = E z,c,t,ϵ∼N (0,I) ∥z 0 -G θ (z t , c, t)∥ 2 2 . (2
)
During the inference phase, the trained model is utilized to iteratively denoise from pure Gaussian noise step by step, following the procedure:
   ẑ0,θ = G θ (z t , c, t), z t-1 ∼ N √ α t (1 -ᾱt-1 ) 1 -ᾱt z t + √ ᾱt-1 (1 -α t ) 1 -ᾱt ẑ0,θ , 1 -ᾱt-1 1 -ᾱt (1 -α t )I . (3
)
Finally, we obtain the clean denoised latent code ẑ⋆ 0 , which is then decoded to the raw video space.
this section cite: ['b18']

Section: Framework Overview
Based on the insight that visual appearance and motion dynamics share inherent physical laws of the real world, we propose an end-to-end framework that unifies HOI video and 3D motion generation by integrating visual priors with dynamic constraints. Our framework consists of two key components. The first is SViMo, which focuses on video generation (Fig. 2, top; Sec. 3.3). It extends a pre-trained image-to-video foundation model into a joint video-motion generation architecture. During joint denoising, SViMo dynamically aligns visual appearance with high-level motion signals, improving both visual quality and motion plausibility. Notably, the motion representations in SViMo refer not to explicit 3D hand trajectories or object point cloud sequences. Because there is a significant gap between 3D motion data and 2D video representations, directly modeling explicit 3D motions would disrupt the pre-trained foundation model's visual priors and degrade performance. Instead, we project explicit 3D interactions onto the 2D image plane to create "rendered motion videos" as intermediate representations for SViMo. The second component is VID, which generates explicit 3D motions (Fig. 2, bottom; Sec. 3.4). It maps the output of SViMo, both the generated video and the 2D rendered motion video, into target 3D interactions, i.e., hand-object trajectories and object point clouds. Additionally, the collaboration between SViMo and VID forms a closed-loop feedback pipeline, ensuring consistency between generated videos and 3D motions (Sec. 3.5).
this section cite: []

Section: Synchronized Video-Motion Diffusion
The SViMo learns to predict the video and motion, given the time step, text prompt, and reference image:
(ẑ V 0 , ẑM 0 ) = G θ (z V
t , z M t ), (P , I), t . The entire video-motion generation process comprises feature embedding, multimodal feature modulation and fusion, and joint denoising.
Feature Embedding. Time step t is sampled from a uniform distribution and then added sinusoidal position encoding, followed by a simple two-layer linear mapping to obtain the time embedding e t ∈ R dtime . For the text prompt, we use a frozen pre-trained language model, Google T5 [48], to extract text embedding e text ∈ R L×dp , and then calculate the semantics features f text with linear projection, where L is the max length of textual tokens. Thirdly, for the original reference image I, to ensure a certain level of robustness, we first add random noise yielding I noised . Then, we compute the compressed latent code z I through a 3D video VAE, z I = E(I noised ) ∈ R H rh × W rw ×dVAE . Additionally, for the target video V , we also map it to the latent space with the same VAE. Then the noised latent code z V t ∈ R ( N -1 rn +1)× H rh × W rw ×dVAE are obtained based on the forward diffusion process in Eq. 1.
To better align it with the image condition signal during the video denoising process, we repeat the image feature z I along the temporal axis, and concatenate them along the channel-axis to get video feature embedding
e V ∈ R ( N -1 rn +1)× H rh × W rw ×(2×dVAE) = z V t ⊕ z I . Then the visual feature is patchified through a 2D stride convolution, esulting in f V ∈ R [( N -1 rn +1)× H 2×rh × W 2×rw ]×d .
The rendered motion video is encoded through the same VAE and diffused to obtain z M t . In contrast, to get motion embedding e M , the diffused latent code here are concatenated with interaction guidance zM 0 provided by VID (Sec. 3.4), rather than being combined with z I . After that, the motion feature f M is obtained in the same way as that of f V .
this section cite: ['b47']

Section: Multimodal Feature Modulation and Fusion.
In our SViMo framework, the DiT token sequence comprises three distinct modalities: text tokens f text , video tokens f V and motion tokens f M , which differ significantly in both feature spaces and numerical scales. To bridge these disparities while preserving modality-specific characteristics, we adopt a triple modality adaptive modulation method that learns modulation parameters from the timestep signal e t . These parameters determine the scaling, shifting, and gating operations of each modality's features separately. Additionally, a 3D full-attention mechanism is employed to capture intra-and inter-modal relationships. Take the processing of text features as an example, the DiT Block B proceeds as follows:
B(•) =                  {α i text , β i text , γ i text } 2 i=1 = MLP(e t ), f ′ text = LN(f text ) ⊙ (1 + α 1 text ) + β 1 text , f ′ text = f ′ text + γ 1 text ⊙ ∩ text 3DFA ∪(f ′ text , f ′ V , f ′ M ) , f ′′ text = LN(f ′ text ) ⊙ (1 + α 2 text ) + β 2 text , f ′′ text = f ′′ text + γ 2 text ⊙ ∩ text FFD ∪(f ′ text , f ′ V , f ′ M ) ,                  same for f V and f M , (4
)
where "3DFA" and "FFD" are unified multi-head 3D full-attention and feedforward layers, ∪ and ∩ denote token concatenation and segmentation along the sequence dimension, respectively.
Video-Motion Joint Denoising. The video and motion features output by the final DiT block then go through an MLP to reconstruct the VAE latent codes, yielding ẑV 0 for video and ẑM 0 for motion. Finally, the SViMo is optimized according to Eq. 2:
L SViMo = E (z V ,z M ),(P ,I),t,ϵ ∥z V 0 -G θ (z V t , (P , I), t)∥ 2 2 + ∥z M 0 -G θ (z M t , (P , I), t)∥ 2 2 . (5
)
this section cite: []

Section: Vision-aware 3D Interaction Diffusion
The VID M ϕ generates the explicit 3D hand pose trajectories ĥ0 and object point cloud sequences ô0 given latent codes of videos z V t and motions (rendered motion video) z M t at any time. The framework operates as follows: First, a dual-stream 3D convolutional module encodes multi-scale spatiotemporal features from both video and motion codes. Then they are fused and subsequently injected into the 3D interaction denoising modules through cross-attention mechanisms to synthesize 3D HOI trajectory sequences. Following the x 0 -prediction formula in Eq. 2, the model is optimized through the following loss function:
   ( ĥ0,ϕ , ô0,ϕ ) = M ϕ (h t , o t ), (z V t , z M t ), t , L VID = E (h,o),(ẑ V θ ,ẑ M θ ),t,ϵ MSE(h 0 , ĥ0,ϕ ) + D chamfer (o 0 , ô0,ϕ ) . (6
)
this section cite: []

Section: Close-loop Feedback and Training Objectives
Close-loop Feedback. To enhance the mutual promotion and co-evolution between SViMo and VID, and improve the consistency between video and 3D motions, we design a closed-loop feedback mechanism. This mechanism includes two pathways: interaction guidance and gradient constraint. Specifically, for the straightforward interaction guidance strategy (Eq. 7, row 1st), we first generate the 3D interaction from the video and motion inputs of SViMo: ( h0 , õ0 ) = M no-grad (h t , o t ), (z V t , z M t ), t , then projected it onto the 2D image plane to obtain rendered motion video M , which are subsequently embedded into the VAE latent space yielding zM 0 . Finally, it is concatenated with the noised motion latent code z M t mentioned in Sec. 3.3, forming an additional interaction guidance for the SViMo. On the other hand, the input of VID could come from the output of the SViMo. Therefore, the gradient of VID will be backpropagated into the SViMo during the training process, forming a gradient constraint path and thereby promoting its optimization (Eq. 7, row 2nd).
     (z V t , z M t )
VIDno-grad
this section cite: []

Section: -----→ ( h0 , õ0 )
Proj.
this section cite: []

Section: --→ M VAE E ---→ zM 0
Inter. Guid. to SViMo
this section cite: []

Section: -----------→ (z
V t , z M t ⊕ zM 0 ), (z V t , z M t ⊕ zM 0 ) SViMo ---⇀ ↽ ----- Gradient (ẑ V 0 , ẑM 0 ) VID -----⇀ ↽ ------ Gradient ( ĥ0 , ô0 ) Loss ---⇀ ↽ ----- Gradient L VID . (7
)
Training Objectives. The training process of our method involves two phases: initially warming up the VID based on Eq. 6, followed by closed-loop training where the SViMo and the VID are jointly optimized according to Eq. 6 and Eq. 5:
L = ω 1 L SViMo + ω 2 L VID .
this section cite: []

Section: Experiments
We conducted extensive experiments to validate the effectiveness of our proposed method. More information, such as additional results and limitation discussions, is provided in the Appendix.
this section cite: []

Section: Experimental Setup
TACO dataset [38] is a large-scale bimanual hand-object interaction dataset capturing diverse tooluse behaviors via multi-view video recordings and high-fidelity 3D motion annotations. Each task is defined as a triplet <tool category, action type, target object category>, describing tool-mediated interactions with objects. The dataset includes 2.5k interaction sequences, covering 20 object categories, 196 3D models, 14 participants, and 15 daily interaction types. It provides allocentric (4096×3000) and egocentric (1920×1080) video streams, totaling 5.2M frames at 30 Hz. To reduce computational load, we crop hand-object interaction regions, adjust their aspect ratio to 3:2, and downsample to 49 frames at 8 FPS. This results in videos with spatiotemporal resolution 416×624×49, aligning with CogVideoX's [69] default settings while lowering spatial resolution. To mitigate overlap between test and training sets that might compromise evaluation of generalization, we implemented a two-stage data partitioning strategy. First, we reserved all instances involving specific actions (e.g., hit), tools (e.g., glue gun), and objects (e.g., toy) as the initial test set, ensuring these elements are absent from the training set. Second, from the remaining instances grouped by <action, tool, object> triplets, we applied weighted sampling according to group sizes to obtain additional test data. Finally, the ratio of the training set to the test set is 9:1.
Evaluation Metric. For video evaluation, we use VBench [22] to assess two key dimensions: Content Quality (including Subject Consistency and Background Consistency) and Dynamic Quality (Temporal Smoothness and Dynamic Degree). More details about these metrics can be found in the Appendix. To address the partiality of individual metrics, we multiply them to derive a Overall score for holistic evaluation. For 3D interaction evaluation, we separately assessed hand poses and object point cloud sequences. For the former, we calculated MPJPE (Mean Per Joint Position Error) and Motion Smoothness metrics. For object evaluation, we measured the Chamfer Distance between generated and ground-truth point clouds. Additionally, we compute a comprehensive FID score via a pretrained interaction autoencoder. Training Details. All models are trained on 4 NVIDIA A800-80G GPUs. With memory optimization techniques including DeepSpeed ZeRO-3 [49], gradient checkpointing, and BF16 mixedprecision trick, we achieve a per-GPU batch size of 4. We first warm up the VID for 5k steps, then conduct joint training with the SViMo. To enhance computational efficiency, we initially train at reduced resolution [H ′ , W ′ ] = [240, 368] for 30k steps before fine-tuning at full resolution for 5k steps. The weights of SViMo and VID terms in the training objectives are ω 1 = 1 and ω 2 = 0.05.
this section cite: ['b37', 'b68', 'b21', 'b48']

Section: Comparison with Previous Approaches
Baselines. For video generation performance, we compare with video models that follow the image-animation paradigm (2.5D Video Models), including Animate Anyone [21] and Easy Animate [61], as well as native 3D large video models, including Hunyuan-13B [26], Wan-14B [25], and CogVideoX-5B [69]. Particularly, due to the high training costs of the first two 3D models, we directly utilized them for zero-shot inference. For 3D motion generation quality, we compare with the classic MDM [57] and its latest improved version EMDM [74]. Our method generates motion from both images and text (image+text-to-motion), whereas MDM and EMDM are inherently textto-motion systems. For fair comparison, we adapted both baselines into text-and-image-to-motion generators by supplying additional reference frames to their CLIP encoders.
Quantitative and Qualitative Evaluation. As shown in Table 1, our method achieves the highest overall score in video generation. Notably, individual metrics often conflict: high scores in one  aspect may compromise others. For instance, Hunyuan-13B [26] attains top subject/background consistency and second-highest temporal smoothness, but its near-static outputs (dynamic degree: 0.49) yield the lowest overall score. Wan-14B [25] also exhibits the same phenomenon, and the lack of TACO dataset fine-tuning results in poor instruction adherence and hallucinations (Fig. 3, Row 1). The 2.5D image animation methods [21,61] achieve high dynamic degree yet exhibit inadequate content consistency and temporal smoothness, visually manifesting as distortions and temporal flickering (Fig. 3, Row 2). CogVideoX-5B [69] achieves the second-highest overall score, but the generated videos still exhibit inconsistencies, as shown on the left of the last row of Fig. 3. In contrast, our method benefits from the synchronized modeling of visual and dynamic, resulting in better comprehensive performance. For motion generation, our method achieves superior performance across all metrics, as shown in Tab. 2. Qualitative results in Fig. 4 reveal that MDM [57] and EMDM [74] produce motions with poor instruction compliance and frame consistency. This stems from two limitations: 1) They compress both reference images and text prompts into 512 dimensions through the CLIP encoder, then simply concatenate them as denoising conditions, which dilutes the instruction signal. 2) Their motion models lack vision awareness, causing large discrepancies between generated point clouds and references. Contrastingly, our approach not only preserves input condition effectiveness through a triple-modality adaptive modulation mechanism, but also enhances object point cloud consistency with low-level visual priors.
this section cite: ['b20', 'b60', 'b25', 'b24', 'b68', 'b56', 'b73', 'b25', 'b24', 'b20', 'b60', 'b68', 'b56', 'b73']

Section: User Study.
To validate our method's effectiveness, we conducted user studies for video and motion generation (Fig. 5). For video generation, 26 image-prompt pairs were used to generate videos with six models each, yielding 1,066 valid responses from 41 participants. Our method achieved a 78.42% preference rate, significantly outperforming all baselines. In motion generation, 10 image-prompt pairs produced 410 valid responses, with our results surpassing the baseline in 97.56% of cases.
These results demonstrate the clear advantages of our video-motion synchronous diffusion model.  Table 3: Ablation studies on the synchronized diffusion and the vision-aware 3D interaction diffusion. Varients Content Dynamic Overall ↑ Hand Object HOI Subj. ↑ Bkg. ↑ TSmoo. ↑ Dyn. ↑ MPJPE ↓ MSmoo. ↓ Cham. ↓ FID ↓ SViMo w/ VID (Ours) 0.9534 0.9546 0.9883 0.9784 0.8800 0.0121 0.0053 0.0019 0.0100 SViMo w/ Inter. Guid. 0.9522 0.9546 0.9877 0.9768 0.8770 0.0157 0.0060 0.0022 0.0100 SViMo w/ Grad. Cons. 0.9499 0.9525 0.9881 0.9757 0.8723 0.0141 0.0058 0.0021 0.0124 SViMo w/o VID 0.9543 0.9545 0.9883 0.9686 0.8719 0.0195 0.0070 0.0037 0.0546 VModel w/ Pred. Mot. 0.9356 0.9392 0.9858 0.9675 0.8381 0.0202 0.0074 0.0040 0.0575
this section cite: []

Section: Generalization and Extensibility
We evaluate our method on manipulation tasks involving common household objects such as rollers, spatulas, spoons, and bowls, for which we collect image-prompt pairs. These inputs are processed by our synchronized diffusion model to generate human-object interaction (HOI) videos and corresponding 3D interactions, as shown in Fig. 6. The successful generation of plausible interactions on real-world object categories demonstrates the generalization capability of our approach. This generalization stems from two key design principles. First, our plug-and-play synchronized diffusion architecture effectively leverages the visual-semantic priors encoded in large video foundation models. Second, the use of 3D point clouds as object representations provides detailed spatial geometry and enhanced physical awareness, which supports robust reasoning across different viewpoints.
Furthermore, the framework is inherently extensible to diverse HOI scenarios. By adjusting point cloud configurations, it readily handles multiple objects, complex geometries, or articulated structures. For example, adding objects only requires including additional point clouds, and articulated objects (e.g., drawers) can be decomposed into rigid sub-components modeled as separate point clouds. Increasing point density captures fine details in deformable or complex shapes. This flexibility highlights the method's practical potential for modeling varied real-world interactions.
this section cite: []

Section: Ablation Study
Effectiveness of Synchronized Diffusion. We argue that integrating visual priors and physical dynamics into a synchronized diffusion process is essential for HOI video and motion generation. To validate our synchronized diffusion mechanism: (1) We first remove VID to avoid confounding factors (SViMo w/o VID).
(2) Then we decompose it into two independent components: a motion generation model with only motion loss and a video generation model conditioned on groundtruth motion (VModel w/ GT Mot. Guid.). After training these models independently, we use the predicted motions from the former as conditions for the latter during inference (VModel w/ Pred. Mot.). The last two rows of Tab. 3 show that modeling video and motion independently not only leads to a 3.88% decrease in video overall score (0.8719 vs. 0.8381), but also results in a 5.31% degradation in motion FID (0.0546 vs. 0.0575). This highlights the importance of our synchronous diffusion model in enabling feature-level synergy between video and motion. This highlights the advantage of integrating visual priors and motion dynamics for our method.
this section cite: []

Section: Impact of Vision-aware 3D Interaction Diffusion.
The vision-aware 3D interaction diffusion model forms a closed-loop feedback and co-evolution mechanism with the synchronized video-motion diffusion, by injecting interaction guidance and gradient constraints into the latter.
To validate its effectiveness, we conduct four variants: removing VID entirely (SViMo w/o VID), applying only gradient constraints (SViMo w/ Grad. Cons.), providing only interaction guidance (SViMo w/ Inter. Guid.), and preserving the full VID (SViMo w/ VID). Evaluation results in Tab. 3 and training loss curves in Fig. 7 show that direct interaction guidance slightly outperforms gradient constraints, while the complete VID achieves the best performance.
Influence of 3D Data on Convergence Speed and Performance. Our method simultaneously generates 2D videos and 3D motions. To evaluate the contribution of 3D motion data to training convergence and output quality, we conduct an ablation study in which 3D motion are replaced with 2D data (Ours-2D-VID). The comparison is performed using three metrics: motion FID, video loss at 1K training steps, and overall video quality. As shown in Tab. 4, incorporating 3D motion diffusion leads to more plausible motions, faster convergence, and a higher overall video score. In contrast, the variant that completely removes the explicit motion generation module (SViMo w/o VID) exhibits further degradation in both convergence speed and final performance. This can be largely attributed to the fact that 3D motion representation enhances the model's understanding of object scale and spatial occlusion relationships.
this section cite: []

Section: Conclusion
In conclusion, we propose a synchronized diffusion model that unifies hand-object interaction video generation and motion synthesis in a single diffusion process. By jointly modeling the co-evolution of appearance and motion, our method produces visually realistic and dynamically plausible results.
A key component is the vision-aware 3D interaction diffusion model, which guides the denoising process via gradient constraints in a closed-loop pipeline, greatly improving video-motion consistency. The framework requires no predefined conditions and shows strong zero-shot generalization in real-world settings. It is also inherently extensible, easily adapting to diverse HOI scenarios, such as those with multiple, complex, or articulated objects, through simple adjustments to the 3D point cloud representation. This paradigm offers a promising direction for multimodal alignment and building world models that understand complex physical interactions.
this section cite: []

Section: References
Ref_id:b0 Title: Body of her: A preliminary study on end-to-end humanoid agent Year: (2024)
Ref_id:b1 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b2 Title: Stable video diffusion: Scaling latent video diffusion models to large datasets Year: (2023)
Ref_id:b3 Title: Physically plausible full-body hand-object interaction synthesis Year: (2024)
Ref_id:b4 Title: Video generation models as world simulators Year: (2024)
Ref_id:b5 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b6 Title: Text2hoi: Text-guided 3d motion generation for hand-object interaction Year: (2024)
Ref_id:b7 Title: Dexycb: A benchmark for capturing hand grasping of objects Year: (2021)
Ref_id:b8 Title: Videojam: Joint appearance-motion representations for enhanced motion generation in video models Year: (2025)
Ref_id:b9 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b10 Title: Vlogger: Multimodal diffusion for embodied avatar synthesis Year: (2025)
Ref_id:b11 Title: Diverse human motion prediction via gumbel-softmax sampling from an auxiliary space Year: (2022)
Ref_id:b12 Title: Cg-hoi: Contact-guided 3d human-object interaction generation Year: (2024)
Ref_id:b13 Title: Arctic: A dataset for dexterous bimanual hand-object manipulation Year: (2023)
Ref_id:b14 Title: Gigahands: A massive annotated dataset of bimanual hand activities Year: (2025)
Ref_id:b15 Title: Learning cooperative human-object interaction with manipulated object dynamics Year: (2024)
Ref_id:b16 Title: Prediction with action: Visual policy learning via joint denoising process Year: (2024)
Ref_id:b17 Title: Stochastic scene-aware motion prediction Year: (2021)
Ref_id:b18 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b19 Title: Cogvideo: Large-scale pretraining for text-to-video generation via transformers Year: (2023)
Ref_id:b20 Title: Animate anyone: Consistent and controllable image-to-video synthesis for character animation Year: (2024)
Ref_id:b21 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b22 Title: The power of sound (tpos): Audio reactive video generation with stable diffusion Year: (2023)
Ref_id:b23 Title: Scaling up dynamic human-scene interaction modeling Year: (2024)
Ref_id:b24 Title: Vace: All-in-one video creation and editing Year: (2025)
Ref_id:b25 Title: A systematic framework for large video generative models Year: (2024)
Ref_id:b26 Title: Nifty: Neural object interaction fields for guided human motion synthesis Year: (2024)
Ref_id:b27 Title: Interhandgen: Two-hand interaction generation via cascaded reverse diffusion Year: (2024)
Ref_id:b28 Title: Controllable human-object interaction synthesis Year: (2024)
Ref_id:b29 Title: Object motion guided human motion synthesis Year: (2023)
Ref_id:b30 Title: Task-oriented human-object interactions generation with implicit neural representations Year: (2024)
Ref_id:b31 Title: Vision-language foundation models as effective robot imitators Year: (2024)
Ref_id:b32 Title: Amt: Allpairs multi-field transforms for efficient frame interpolation Year: (2023)
Ref_id:b33 Title: Visual instruction tuning Year: (2023)
Ref_id:b34 Title: Javisdit: Joint audio-video diffusion transformer with hierarchical spatio-temporal prior synchronization Year: (2025)
Ref_id:b35 Title: Primitive-based 3d humanobject interaction modelling and programming Year: (2024)
Ref_id:b36 Title: Geneoh diffusion: Towards generalizable hand-object interaction denoising via denoising diffusion Year: (2024)
Ref_id:b37 Title: Taco: Benchmarking generalizable bimanual tool-action-object understanding Year: (2024)
Ref_id:b38 Title: Hoi4d: A 4d egocentric dataset for category-level human-object interaction Year: (2022)
Ref_id:b39 Title: Visual-rft: Visual reinforcement fine-tuning Year: (2025)
Ref_id:b40 Title: Omnigrasp: Grasping diverse objects with simulated humanoids Year: (2024)
Ref_id:b41 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b42 Title: Generating hand-object manipulation video with dexterous and generalizable grasping Year: (2024)
Ref_id:b43 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b44 Title: Hoidiff: Text-driven synthesis of 3d human-object interactions using diffusion models Year: (2023)
Ref_id:b45 Title: Hierarchical generation of humanobject interactions with diffusion probabilistic models Year: (2023)
Ref_id:b46 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b47 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b48 Title: Zero: Memory optimizations toward training trillion parameter models Year: (2020)
Ref_id:b49 Title: Zero-shot text-to-image generation Year: (2021)
Ref_id:b50 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b51 Title: Photorealistic textto-image diffusion models with deep language understanding Year: (2022)
Ref_id:b52 Title:  Year: (2024)
Ref_id:b53 Title: Grab: A dataset of whole-body human grasping of objects Year: (2020)
Ref_id:b54 Title: Any-to-any generation via composable diffusion Year: (2023)
Ref_id:b55 Title: Raft: Recurrent all-pairs field transforms for optical flow Year: (2020)
Ref_id:b56 Title: Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model Year: (2023)
Ref_id:b57 Title: Deepsimho: Stable pose estimation for hand-object interaction via physics simulation Year: (2023)
Ref_id:b58 Title: Cogvlm: Visual expert for pretrained language models Year: (2024)
Ref_id:b59 Title: Physics-based imitation of dynamic human-object interaction Year: (2023)
Ref_id:b60 Title: Easyanimate: A high-performance long video generation method based on transformer architecture Year: (2024)
Ref_id:b61 Title: Interdiff: Generating 3d human-object interactions with physics-informed diffusion Year: (2023)
Ref_id:b62 Title: Intermimic: Towards universal wholebody control for physics-based human-object interactions Year: (2025)
Ref_id:b63 Title: Interdreamer: Zero-shot text to 3d dynamic human-object interaction Year: (2024)
Ref_id:b64 Title: Magicanimate: Temporally consistent human image animation using diffusion model Year: (2024)
Ref_id:b65 Title: Anchorcrafter: Animate cyberanchors saling your products via human-object interacting video generation Year: (2024)
Ref_id:b66 Title: Oakink: A large-scale knowledge repository for understanding hand-object interaction Year: (2022)
Ref_id:b67 Title: The dawn of lmms: Preliminary explorations with gpt-4v (ision) Year: (2023)
Ref_id:b68 Title: Cogvideox: Text-to-video diffusion models with an expert transformer Year: (2025)
Ref_id:b69 Title: Diverse and aligned audio-tovideo generation via text-to-video model adaptation Year: (2024)
Ref_id:b70 Title: Oakink2: A dataset of bimanual hands-object manipulation in complex task completion Year: (2024)
Ref_id:b71 Title: Manidext: Hand-object manipulation synthesis via continuous correspondence embeddings and residualguided diffusion Year: (2024)
Ref_id:b72 Title: Towards controllable human-chair interactions Year: (2022)
Ref_id:b73 Title: Emdm: Efficient motion diffusion model for fast and high-quality motion generation Year: (2024)
Ref_id:b74 Title: Champ: Controllable and consistent human image animation with 3d parametric guidance Year: (2024)
Ref_id:b75 Title: Rt-2: Vision-language-action models transfer web knowledge to robotic control Year: (2023)
