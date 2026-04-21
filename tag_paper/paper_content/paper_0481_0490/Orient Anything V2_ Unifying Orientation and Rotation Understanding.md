Title: Orient Anything V2: Unifying Orientation and Rotation Understanding
Abstract: This work presents Orient Anything V2, an enhanced foundation model for unified understanding of object 3D orientation and rotation from single or paired images. Building upon Orient Anything V1, which defines orientation via a single unique front face, V2 extends this capability to handle objects with diverse rotational symmetries and directly estimate relative rotations. These improvements are enabled by four key innovations: 1) Scalable 3D assets synthesized by generative models, ensuring broad category coverage and balanced data distribution; 2) An efficient, model-in-the-loop annotation system that robustly identifies 0 to N valid front faces for each object; 3) A symmetry-aware, periodic distribution fitting objective that captures all plausible front-facing orientations, effectively modeling object rotational symmetry; 4) A multi-frame architecture that directly predicts relative object rotations. Extensive experiments show that Orient Anything V2 achieves state-of-the-art zero-shot performance on orientation estimation, 6DoF pose estimation, and object symmetry recognition across 11 widely used benchmarks. The model demonstrates strong generalization, significantly broadening the applicability of orientation estimation in diverse downstream tasks.

Section: Introduction
Estimating object orientation from images is a fundamental task of computer vision. 3D object orientation information plays crucial roles in robot manipulation [49,37,22], autonomous driving [25], AR/VR [11,30,4,57], and spatial-aware image understanding [21,28,60] and generation [56,50,35].
Orient Anything V1 [48] is a foundation model for estimating the object orientation aligned with an object's unique front face. While it exhibits strong robustness and accuracy in absolute orientation estimation, it lacks an understanding of rotation (despite its intrinsic link to orientation). This deficiency results in difficulties handling numerous rotationally symmetric objects (simply classifying them as having no front face) and understanding object rotation relative to a specified reference frame. These limitations around rotation understanding restrict its utility in many downstream tasks.
In this work, we aim to develop an enhanced orientation estimation model, Orient Anything V2, with stronger generalization and deeper understanding of both object orientation and rotation. Our contributions include a scalable data engine and a more elegant model framework.
From the data perspective, Orient Anything V1 uses advanced VLM [12,31] to annotate real 3D assets from Objaverse [8,7]. Building on this data-driven motivation, we leverage advanced 3D generation models [51,61] to further speed up data scaling-up and improve the data coverage and balance. Additionally, we assemble pseudo labels predicted by the V1 model across multi-view renderings and refine them through model-in-the-loop calibration. The proposed data engine enables highly cost-effective and flexible data scaling up, delivers robust annotation performance, and shows a strong understanding of rotationally symmetric objects. Our final dataset includes 600K assets, 12× larger than the existing orientation dataset, with significantly higher annotation quality, accurately identifying 0 to N valid front faces.
From the model perspective, we first propose symmetry-aware orientation distribution, explicitly teaching the model to capture and predict rotational symmetry. Moreover, our model supports multiframe input to directly predict relative rotations between frames. This design effectively bridges the knowledge transfer between absolute orientation and relative rotation, showing strong potential in reference-known scenarios.
Our experiments demonstrate the enhanced and novel capabilities of our model. It achieves superior performance on zero-shot orientation estimation and sets new records on zero-shot rotation estimation (i.e., 6DoF pose estimation [49,26]), while also accurately handling and predicting different rotational symmetries.
To summarize, we propose Orient Anything V2, which improves Orient Anything V1 as follows:
• We propose a data engine that cost-efficiently scales up 3D asset collection and robustly annotates the 0 to N valid front faces to capture different object rotational symmetries.
• We introduce symmetry-aware distribution fitting as a learning objective, allowing the model to directly predict all plausible object orientations.
• We extend the model architecture to support multi-frame input, enabling it to directly estimate relative object rotations over the reference frame.
• Our model demonstrates strong zero-shot generalization across absolute orientation estimation, relative rotation estimation, and object symmetry recognition.
2 Related Work
this section cite: ['b47', 'b35', 'b20', 'b23', 'b10', 'b28', 'b3', 'b55', 'b19', 'b26', 'b58', 'b54', 'b48', 'b33', 'b46', 'b11', 'b29', 'b7', 'b6', 'b49', 'b59', 'b47', 'b24']

Section: Object Rotational Symmetry
Rotational symmetry [36,34] indicates that an object may retain its original shape after being rotated by certain angles. This property is commonly found across various objects. Understanding an object's rotational symmetry is critical for 3D object recognition and generation [24,59], pose estimation [17,6], and robotic manipulation [39]. While some existing works [40] attempt to detect 3D rotational symmetry from single-view 2D images, they are constrained by limited training data and lack zero-shot generalization to open-world scenarios.
this section cite: ['b34', 'b32', 'b22', 'b57', 'b16', 'b5', 'b37', 'b38']

Section: T-pose T-pose 低质量纹理
(a) Low-quality Texture (b) Limited Pose&Shape 低质量纹理 T-pose Our focus is on object orientation relative to a semantic "front" face. The number of possible valid front-facing orientations an object possesses is determined by its rotational symmetry around its vertical axis. For example, 180-degree symmetry means there are two distinct valid front faces.
Objects with continuous rotational symmetry (symmetric at any angle), like balls, are considered to have no meaningful direction. In this work, we broaden the applicability of orientation estimation models by enabling the prediction of an object's azimuthal symmetry from a single 2D image. Our model demonstrates impressive zero-shot rotational symmetry recognition performance.
this section cite: []

Section: Relative Rotation Estimation
Predicting an object's rotation in the query frame relative to the reference frame is a fundamental capability in 6DoF pose estimation [26,14,55] and is crucial for robotics applications. Early methods [23,18,9] focused on specific instances or object categories. More recent approaches like OnePose [44] and OnePose++ [15] estimate object rotation by solving 2D-3D correspondences across views. POPE [10] follows a similar idea and achieves zero-shot rotation estimation with a single reference frame with the help of SAM [19] and DINOv2 [33]. However, the reliance on pixel matching makes these methods prone to failure under large viewpoint changes.
In contrast, we propose a purely implicit learning approach. Leveraging the inherent coupling between rotation and orientation, we extend the orientation estimation model to support multi-frame inputs, enabling direct zero-shot relative rotation prediction between arbitrary views.
this section cite: ['b24', 'b13', 'b53', 'b21', 'b17', 'b8', 'b42', 'b14', 'b9', 'b18', 'b31']

Section: Single-view Orientation Estimation
Estimating an object's 3D front-facing orientation (interpreted as its rotation relative to the canonical front view) from a single view, requires the model to have an inherent understanding of different objects' standard poses and front-facing appearances. Earlier works [53,46,42] are mainly limited to a small number of categories or specific domains. More recently, ImageNet3D [29] introduce a large-scale dataset with manually annotated 3D orientations. Orient Anything [48] achieves robust orientation estimation for any objects in any scenes by leveraging an advanced automated annotation pipeline, improved learning objectives, and real-world knowledge from the pre-trained vision model.
In this work, we further address several limitations of Orient Anything and upgrade the orientation estimation model from both data-driven (novel and scalable data engine) and model-driven (direct symmetry and rotation prediction) perspectives, resulting in Orient Anything V2.
this section cite: ['b51', 'b44', 'b40', 'b27', 'b46']

Section: Revisiting Orient Anything V1
Orient Anything V1 pioneers zero-shot object orientation estimation from single images. It introduces a VLM-based pipeline to annotate front faces of Objaverse 3D assets [8,7], learns orientation estimation from their renderings via distribution fitting. It also provides a confidence score to indicate whether an object has a unique front face. To further advance the orientation prediction foundation model, we first dig into the potential limitations in its training data and framework.
this section cite: ['b7', 'b6']

Section: Disadvantages of Real 3D Assets 1) Imbalanced Category Distribution:
Stemming from the human biases in asset creation, real 3D datasets, such as Objaverse [8,7], suffer from significant class imbalance.
Common categories like buildings and characters make up a large proportion, while Class Tag Caption Image 3D Mesh LLM Rewrite Image Generation 3D Mesh Generation .. .. Heater Cage Computer Fighter Farmer Fence . . . Fox Worker .. .. . . . .. .. . . . .. .. . . . "matte plastic, ethereal cage" "muted tones, dreamlike computer" "rusted, cinematic heater" "cartoon, dramatic farmer" "granite, symmetrical, fence" "polished wood, pastel fighter" "gold made, ambient glow fox" "dusty, realistic, sidelight, worker"
Figure 3: Overview of 3D Asset Synthesis Pipeline. We begin with class tags and use a series of advanced generative models to progressively generate high-quality 3D assets. others, like uncommon animals, are severely underrepresented. 2) Inconsistent Data Quality: Current large-scale 3D datasets often lack high-quality assets with complete geometry and rich surface details. (Fig. 2 a) Moreover, many human-created meshes exhibit fixed poses, leading to a substantial domain gap from real-world object variations (Fig. 2 b).
Limitations of Object Rotation Understanding 1) Ignored Rotational Symmetries: Orient Anything V1 defines orientation based on the single, unique front face, overlooking the different rotational symmetries (i.e., multiple valid "front" faces). For the many symmetric objects in real world, the model cannot effectively distinguish or identify their potential orientations. 2) Unsupported Relative Rotations: The relative rotation between two views and the front-facing orientation (essentially the rotation relative to the front view) are inherently coupled. However, estimating relative rotation through independent absolute orientation predictions suffers from significant error accumulation, causing Orient Anything V1 to often fail in relative rotation estimation.
this section cite: ['b7', 'b6']

Section: Scalable Data Engine

this section cite: []

Section: 3D Asset Synthesis
Motivated by the recent remarkable progress in generative models and the successful application of synthetic data in downstream tasks [45,56], we explore whether synthetic 3D assets can serve as scalable, high-quality data sources for orientation learning. To fully harness modern generative models, we construct our asset synthesis pipeline as a structured process: Class Tag → Caption → Image → 3D Mesh, as detailed below:
Step 1: Class Tag → Caption. To ensure broad category coverage and diversity, we follow SynCLR's approach [45], starting from ImageNet-21K [38] category tags, and use Qwen-2.5 [54] to generate rich captions that describe detailed object attributes and diverse poses. Finally, we generate 600k 3D assets in total, with approximately 30 items for each class tag in ImageNet-21K. These assets feature complete geometry, detailed textures, and balanced category coverage. In terms of scale, the new synthetic dataset is 12× larger than the filtered real dataset used in Orient Anything V1.
this section cite: ['b43', 'b54', 'b43', 'b36', 'b52']

Section: Robust Annotation
Orient Anything V1 employs VLM to annotate the unique canonical front view of 3D assets. However, this approach is limited by the VLM's underdeveloped spatial perception ability and struggles to handle diverse rotational symmetries. To address these challenges, we introduce a more effective and robust system for annotating 3D asset orientations.
this section cite: []

Section: Multi-view Rendering

this section cite: []

Section: Final Label

this section cite: []

Section: V1 model Annotating

this section cite: []

Section: Pseudo Label

this section cite: []

Section: Voting & Standardize

this section cite: []

Section: Automatic Label

this section cite: []

Section: Manual Calibration
Figure 4: Overview of Robust Annotation Pipeline. "Pseudo Label" visualizes the azimuth direction of pseudo labels and objects in the horizontal plane. By fitting the pseudo labels to standard periodic distribution, we can robustly derive the orientation and symmetry label. Human calibration is only required for categories with symmetry inconsistencies.
this section cite: []

Section: Intra-asset Ensemble Annotation
We first train an improved orientation estimation model as our automatic annotator, based on the Orient Anything V1 paradigm and incorporating the additional real-world orientation dataset, ImageNet3D. Next, for each 3D asset, we employ this model to produce pseudo-labels for various renderings. Finally, we project these pseudo-labels, obtained from different viewpoints, back into a canonical 3D world coordinate system.
As shown in Fig. 4, the overall distribution of pseudo-labels on the horizon plane clearly indicates the object's possible orientations. To capture the main direction and rotational symmetries, we first arrange the discrete predicted azimuth angles over [0°, 360°) into a probability distribution P pseudo ∈ R 360 . This distribution is then fitted to a periodic Gaussian distribution using the least squares method:
( φ, ᾱ, σ) = arg min φ,α,σ 359 i=0   P pesudo (i) - exp cos(α(i-φ)) σ 2 2πI 0 1 σ 2   2 (1
)
where σ is the fitted variance. The phase φ ∈ [0 • , 360 • ) represents the main azimuth direction. The periodicity ᾱ ∈ {1, 2, . . . , N } signifies 360/ᾱ-degree rotational symmetry, possessing ᾱ valid front faces, while ᾱ = 0 indicates no dominant orientation.
Ensembling multiple pseudo labels in the 3D world effectively suppresses outlier errors from singleview predictions, resulting in significantly more reliable annotations.
Inter-assets Consistency Calibration Building on the rotational symmetry and orientation annotations for individual assets, we further perform human-in-the-loop consistency calibration across assets. Specifically, since our 3D assets are generated based on object category tags, they are naturally grouped by category. We assume that objects of the same category should share the same type of rotational symmetry. Based on this assumption, we analyze the annotated rotational symmetries within each category. If all assets within the same category demonstrate the same symmetries, we directly consider the annotations to be correct. If inconsistencies are found, we manually review all assets in that category to re-annotate or filter out incorrect annotations.
As each asset is annotated independently, the cross-asset consistency check and manual calibration offer an orthogonal perspective that efficiently and effectively enhances annotation reliability. Statistically, across 21k source category tags, we observe only minor inconsistencies in around 15% of categories, each involving a small number of assets. The finding further validates the accuracy and robustness of our ensemble annotation strategy.
this section cite: []

Section: Encoder

this section cite: []

Section: Joint Encoder
Orientation Token Rotation Token Encoder
this section cite: []

Section: Framework

this section cite: []

Section: Symmetry-aware Distribution
Orient Anything V1 proposes an orientation distribution fitting task that guides the model to learn circular Gaussian distributions over azimuth, polar, and in-plane rotation angles, that preserve the similarity between neighboring angles. Each angle is modeled with a unimodal target distribution centered on a unique front-facing orientation. For symmetric objects with multiple or no semantic front faces, the model additionally predicts a low orientation confidence to filter them out.
To recognize different types of rotational symmetry and enable general orientation prediction for objects with multiple front faces, we further introduce the symmetry-aware periodic distribution as the training target. As discussed in Sec. 4.2, our ensemble annotation and consistency calibration approach enables accurate and robust labeling of 0 to N valid front-facing directions over the horizontal plane. To incorporate these annotations into prediction, we directly model 0 to N valid front faces within the azimuth angle distribution. This design naturally replaces V1's extra orientation confidence design. Instead, different kinds of rotational symmetries are captured directly from the predicted probability distribution. This more elegant framework enables the model to inherently share knowledge across all object categories.
For training, the target P azi ∈ R 360 for the azimuth angle, originally represented as a circular Gaussian distribution, is adapted to be periodic:
P azi (i| φ, ᾱ, σ) = exp cos( ᾱ(i-φ)) σ 2 2πI 0 1 σ 2 (2)
where φ and ᾱ are the phase (azimuth angle) and periodicity (rotation symmetry) fitted from Sec. 4.2, σ is the variance hyper-parameter, and i = 0 • , . . . , 359 • is the angle index. Target probability distributions for the polar angle P pol ∈ R 180 and in-plane rotation angle P rot ∈ R 360 are constructed using a similar method, but without the periodicity parameter.
During inference, the predicted angle distributions are fitted to a standard distribution model using the least squares method, similar to Eq. 1. The resulting parameters (azimuth periodicity α, azimuth angle φ, polar angle σ and rotation angle δ), directly indicate the object's α valid front faces (i.e., symmetric with 360/α degree rotation) and their corresponding front-facing directions in 3D space.
this section cite: []

Section: Relative Rotation Estimation
To establish a connection between absolute orientation and relative rotation, enabling knowledge sharing and transferring, we modify the network architecture to support dynamic inputs from one or multiple images.
As shown in Fig. 5, we mainly follow VGGT [47], first using a visual encoder, DINOv2 [33], to encode each input image into K tokens, augmented with learnable tokens. The combined set of tokens from all frames is then passed into a unified transformer block. The final learnable token corresponding to each frame is used for prediction. Specifically, the learnable token for the first frame is initialized differently and is used to predict the absolute orientation using the symmetry-aware distribution described in Sec. 5.1. Tokens from subsequent frames predict the object rotation relative to the first frame through a similar probability fitting task, but without considering symmetry.
this section cite: ['b45', 'b31']

Section: Training Setting
Our model is initialized from VGGT, a large feed-forward transformer with 1.2 billion parameters pre-trained on 3D geometry tasks. We repurpose its original "camera" token, designed to predict camera extrinsics, to predict object orientation and rotation. This leverages the inherent correlation between camera pose and object rotation. We train the model to fit target orientation (or rotation) distributions using Binary Cross-Entropy (BCE) loss for 20k iterations. A cosine learning rate scheduler is used with an initial rate of 1e-3. Input frames are resized to 518, and random patch masking is used for data augmentation to simulate real-world occlusion. The effective batch size is set to 48, where 1-2 frames are randomly sampled for each training sample. The training dataset comprises the ImageNet3D training set and newly collected 600k synthetic assets. Furthermore, we observe that most objects exhibit only four types of rotational symmetry: {0, 1, 2, 4}. Therefore, we restrict our training to consider only these four cases. Any fitted periodicity ᾱ > 4 is mapped to 0.
this section cite: []

Section: Experiment

this section cite: []

Section: Zero-shot Orientation Estimation
Benchmark & Baselines Predicting the 3D orientation of objects from a single image is our core focus. We mainly compare with Orient Anything V1 [48] on ImageNet3D [29] test set and unseen test datasets, SUN-RGBD [41], ARKitScenes [3], Pascal3D+ [52], Objectron [1] and the Ori_COCO [48]. Since current testing datasets often provide only one ground truth orientation, even for symmetric objects, when Orient Anything V2 predicts multiple orientations, we simply select the one closest to facing the camera as the prediction. The main evaluation metrics are the median 3D angle error (Med↓) and accuracy within 30 degrees (Acc30°↑). For Ori_COCO, where 20 samples are collected for each class and annotated within 8 horizontal orientations, recognition accuracy (Acc↑) is used.
this section cite: ['b46', 'b27', 'b39', 'b2', 'b50', 'b0', 'b46']

Section: Main Results
In Tab. 1, we present the comparative results on single view-based orientation estimation. Overall, Orient Anything V2 significantly improves upon V1, benefiting from diverse synthetic data and robust ensemble annotation. On the representative Ori_COCO benchmark, our method achieves 86.4% accuracy and performs well on categories where V1 struggled, such as bicycles. Achieving state-of-the-art results on numerous real-world image datasets highlights our method's generalization ability.
this section cite: []

Section: Zero-shot Rotation Estimation
Benchmark & Baselines We benchmark zero-shot 6DoF object pose estimation performance under a single reference view. Evaluation is conducted on four widely used datasets: LINEMOD [16], YCB-Video [5], OnePose++ [15], and OnePose [44]. Objects are prepared using the cropping and matching, following [10]. Comparisons are made against three state-of-the-art zero-shot 6DoF object pose estimation methods: Gen6D [27], LoFTR [43], and POPE [10]. Standard metrics for relative object pose estimation are used: median error (Med) and accuracy within 15 • and 30 • (Acc15 and Acc30), computed for each sample pair.
this section cite: ['b15', 'b4', 'b14', 'b42', 'b9', 'b25', 'b41', 'b9']

Section: Model

this section cite: []

Section: LINEMOD YCB-Video OnePose++ OnePose

this section cite: []

Section: Med↓ Acc30↑ Acc15↑ Med↓ Acc30↑ Acc15↑ Med↓ Acc30↑ Acc15 Med↓ Acc30↑ Acc15↑
POPE's Sampling (Average rotation angle: 14.85°)
Gen6D 44.86 36.4 9.6 54.48 23.2 7.7 35.43 41.1 15.8 17.78 89.3 38.9 LoFTR 33.04 56.2 32.4 19.54 68.6 47.8 9.01 89.1 70.3 4.35 96.3 91.8 POPE 15.73 77.0 48.3 13.94 80.1 54.4 6.27 89.6 72.8 2.16 96.2 91.1 OriAny.V2 7.82 98.07 89.7 6.07 91.6 86.4 6.18 99.7 96.6 6.76 99.7 95.7 Random Sample (Average rotation angle: 78.22°) POPE 98.03 10.3 4.3 41.88 40.9 27.2 88.21 25.6 19.8 45.73 45.1 37.3 OriAny.V2 28.83 51.6 28.3 15.78 61.2 48.7 12.83 85.5 58.8 11.72 86.7 63.4
Table 2: Zero-shot Relative Rotation Estimation (i.e., pose estimation with one reference view). We evaluate two strategies for sampling query-reference view pairs: (1) query-reference pairs provided by POPE [10], and (2) randomly sampled pairs. The average rotation angles between views for each sampling strategy are 14.85°and 78.22°, respectively.
this section cite: ['b9']

Section: Main Results
Tab. 2 includes zero-shot two-view relative rotation estimation results compared with state-of-the-art pose estimation methods. With small relative rotations (using POPE's sampling), our model achieves the overall best performance across the four datasets. More importantly, our method's advantage is significantly larger when the relative rotation between the query and reference frame is larger (using random sampling). The significant performance drop of the previous method stems from the reliance on explicit feature matching, which becomes unreliable with large rotations due to less view overlap and scarcer reliable matching points. In contrast, our approach understands images from different viewpoints by considering overall meaning rather than just detailed matching. This makes it more robust to challenging large rotations.
this section cite: []

Section: Zero-shot Symmetry Recognition
Benchmark & Baselines We assess our method's zero-shot performance in predicting object rotational symmetry in the horizontal plane. This evaluation uses the recent, large-scale 3D object datasets with rotational symmetry annotations: Omni6DPose [58], which contain 149 distinct object classes. To ensure their orientation definition aligns with our front-facing direction, we manually select a subset of 3-5 assets per category and render 2 views per 3D asset for testing. This resulted in 838 testing sample. During inference, models receive a single rendering and predict the four kinds of rotational symmetry predictions. As there are currently no dedicated zero-shot models for predicting object rotational symmetry from a single view, we employ advanced VLMs (Qwen2.5VL-72B [2], GPT-4o [31], GPT-o3 [32], and Gemini-2.5-pro [13]) as baselines. We evaluate their ability to predict horizontal plane rotational symmetry using a multiple-choice format, with recognition accuracy as the metric.
this section cite: ['b56', 'b1', 'b29', 'b30', 'b12']

Section: Omni6DPose
Acc↑ Random 25.0 Qwen2.5VL-72B 55.8 Gemini-2.5-pro 44.4 GPT-4o 62.5 GPT-o3 53.7
OriAny. V2
this section cite: []

Section: 65.2
Table 3: Zero-shot horizontal rotational symmetry recognition.
this section cite: []

Section: Main Results
We present a comparison of our method against various advanced general VLMs for identifying object horizontal rotational symmetry in Tab. 3. Our results indicate that recognizing object rotational symmetry is a challenging problem even for the strongest VLMs, thereby limiting their ability to fully understand the 3D spatial state from 2D images. In contrast, benefiting from high-quality annotations and a unified learning objective, our model achieves 65% accuracy in distinguishing object rotational symmetry. Combining this strong symmetry recognition ability alongside the robust and accurate absolute orientation estimation performance demonstrated in Sec. 6.1, our model can accurately infer multiple potential orientations from a single image in real applications.
this section cite: []

Section: Ablation Study
Quality of Synthetic 3D Assets Fig. 6 visualizes our synthetic dataset and the labelled orientation, qualitatively demonstrating the high quality of both the synthetic data and its annotations. Quantitatively, Rows 1 and 2 of Tab. 4 show the comparison of training with an equal amount of annotated real or synthetic 3D assets. We observe that both data sources yield comparable results for absolute orientation estimation. However, for rotation estimation (on LINEMOD and YCB-Video), training with synthetic assets provides a significant advantage. This may be because synthetic assets possess richer, more realistic textures, which are more crucial for understanding rotation.
Effect of Scaling Data In Rows 2, 3, 4, and 5 of Tab. 4, we explore the impact of data scale on the performance of final orientation and rotation estimation. Overall, with the same training step, encountering more diverse data and 3D assets during training leads to better overall performance. Specifically, we find that rotation estimation is more sensitive to data scale than orientation estimation. This may be because orientation relies on overall semantics and structure, while rotation estimation requires understanding diverse textures and fine-grain details to capture cross-view relationships.
Effect of Geometry Pre-training Tab. 4 (Rows 5-7) presents our experiments of different model initialization strategies. Training without any pre-trained initialization yields the worst results. Initializing the separated visual encoder with DINOv2 introduces valuable high-quality semantic and object structure information, leading to substantial performance gains. We observe further improvements in rotation estimation by using VGGT, pre-trained specifically on 3D geometric tasks, which boosts the model's comprehension of object geometry.
this section cite: []

Section: Conclusion
We present Orient Anything V2, an advanced model for unified object orientation and rotation understanding. Through introducing the scalable data engine, a symmetry-aware distribution learning target, and a multi-frame framework, our model enables: 1) Stronger single-view absolute orientation estimation. 2) Advanced two-frame object relative pose rotation estimation. 3) Powerful object horizontal rotational symmetry recognition. In practice, the model can simultaneously and accurately predict multiple valid front faces of objects, making it well-suited for diverse objects and real-world application scenarios.
Limitation While our models exhibit strong generalization to diverse in-the-wild objects in real images, we find that the inherent ambiguity of monocular images leads to less accurate predictions in views with very low information or severe occlusion. Furthermore, the current framework supports a maximum of two input frames. Extending the model to handle more frames will be an important direction for supporting video understanding applications.
this section cite: []

Section: References
Ref_id:b0 Title: Objectron: A large scale dataset of object-centric videos in the wild with pose annotations Year: (2021)
Ref_id:b1 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b2 Title: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data Year: (2021)
Ref_id:b3 Title: The state of the art of spatial interfaces for 3d visualization Year: (2021)
Ref_id:b4 Title: The ycb object and model set: Towards common benchmarks for manipulation research Year: (2015)
Ref_id:b5 Title: Pose estimation for objects with rotational symmetry Year: (2018)
Ref_id:b6 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2023)
Ref_id:b7 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b8 Title: Dfm: A performance baseline for deep feature matching Year: (2021)
Ref_id:b9 Title: Pope: 6-dof promptable pose estimation of any object in any scene with one reference Year: (2024)
Ref_id:b10 Title: Interaction strategies for effective augmented reality geo-visualization: Insights from spatial cognition Year: (2021)
Ref_id:b11 Title:  Year: (2025)
Ref_id:b12 Title:  Year: (2025)
Ref_id:b13 Title: Hrpose: Real-time high-resolution 6d pose estimation network using knowledge distillation Year: (2023)
Ref_id:b14 Title: Onepose++: Keypointfree one-shot object pose estimation without cad models Year: (2022)
Ref_id:b15 Title: Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes Year: (2012)
Ref_id:b16 Title: Epos: Estimating 6d pose of objects with symmetries Year: (2020)
Ref_id:b17 Title: Cotr: Correspondence transformer for matching across images Year: (2021)
Ref_id:b18 Title: Segment anything Year: (2023)
Ref_id:b19 Title: Perspective-aware reasoning in vision-language models via mental imagery simulation Year: (2025)
Ref_id:b20 Title: What foundation models can bring for robot learning in manipulation: A survey Year: (2024)
Ref_id:b21 Title: Practical stereo matching via cascaded recurrent network with adaptive correlation Year: (2022)
Ref_id:b22 Title: Symmetry strikes back: From single-image symmetry detection to 3d generation Year: (2024)
Ref_id:b23 Title: Drive-1to-3: Enriching diffusion priors for novel view synthesis of real vehicles Year: (2024)
Ref_id:b24 Title: Nicu Sebe, and Ajmal Mian. Deep learning-based object pose estimation: A comprehensive survey Year: (2024)
Ref_id:b25 Title: Generalizable model-free 6-dof object pose estimation from rgb images Year: (2022)
Ref_id:b26 Title: 3dsrbench: A comprehensive 3d spatial reasoning benchmark Year: (2024)
Ref_id:b27 Title: Imagenet3d: Towards general-purpose object-level 3d understanding Year: (2024)
Ref_id:b28 Title: Hands-free interaction in immersive virtual reality: A systematic review Year: (2021)
Ref_id:b29 Title:  Year: (2025)
Ref_id:b30 Title:  Year: (2025)
Ref_id:b31 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b32 Title: Rotational symmetry field design on surfaces Year: (2007)
Ref_id:b33 Title: Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d Year: (2024)
Ref_id:b34 Title: Detecting rotational symmetries Year: (2005)
Ref_id:b35 Title: Language-grounded orientation bridges spatial reasoning and object manipulation Year: (2025)
Ref_id:b36 Title: Imagenet-21k pretraining for the masses Year: (2021)
Ref_id:b37 Title: Symmetrygrasp: Symmetryaware antipodal grasp detection from single-view rgb-d images Year: (2022)
Ref_id:b38 Title: Learning to detect 3d symmetry from single-view rgb-d images with weak supervision Year: (2022)
Ref_id:b39 Title: Sun rgb-d: A rgb-d scene understanding benchmark suite Year: (2015)
Ref_id:b40 Title: Render for cnn: Viewpoint estimation in images using cnns trained with rendered 3d model views Year: (2015)
Ref_id:b41 Title: Loftr: Detector-free local feature matching with transformers Year: (2021)
Ref_id:b42 Title: Onepose: One-shot object pose estimation without cad models Year: (2022)
Ref_id:b43 Title: Learning vision from models rivals learning vision from data Year: (2024)
Ref_id:b44 Title: Normalized object coordinate space for category-level 6d object pose and size estimation Year: (2019)
Ref_id:b45 Title: Vggt: Visual geometry grounded transformer Year: (2025)
Ref_id:b46 Title: Orient anything: Learning robust object orientation estimation from rendering 3d models Year: (2025)
Ref_id:b47 Title: Foundationpose: Unified 6d pose estimation and tracking of novel objects Year: (2024)
Ref_id:b48 Title: Neural assets: 3d-aware multi-object scene synthesis with image diffusion models Year: (2024)
Ref_id:b49 Title: Structured 3d latents for scalable and versatile 3d generation Year: (2024)
Ref_id:b50 Title: Beyond pascal: A benchmark for 3d object detection in the wild Year: (2014)
Ref_id:b51 Title: Few-shot object detection and viewpoint estimation for objects in the wild Year: (2022)
Ref_id:b52 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b53 Title: Fmr-gnet: Forward mix-hop spatial-temporal residual graph network for 3d pose estimation Year: (2024)
Ref_id:b54 Title: Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging Year: (2025)
Ref_id:b55 Title: Gaze-supported 3d object manipulation in virtual reality Year: (2021)
Ref_id:b56 Title: Omni6dpose: A benchmark and model for universal 6d object pose estimation and tracking Year: (2024)
Ref_id:b57 Title: Single depth-image 3d reflection symmetry and shape prediction Year: (2023)
Ref_id:b58 Title: Do vision-language models represent space and how? evaluating spatial frame of reference under ambiguities Year: (2024)
Ref_id:b59 Title: Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation Year: (2025)
