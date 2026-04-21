Title: SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation
Abstract: Controllable image generation has attracted increasing attention in recent years, enabling users to manipulate visual content such as identity and style. However, achieving simultaneous control over the 9D poses (location, size, and orientation) of multiple objects remains an open challenge. Despite recent progress, existing methods often suffer from limited controllability and degraded quality, falling short of comprehensive multi-object 9D pose control. To address these limitations, we propose SceneDesigner, a method for accurate and flexible multi-object 9-DoF pose manipulation. SceneDesigner incorporates a branched network to the pre-trained base model and leverages a new representation, CNOCS map, which encodes 9D pose information from the camera view. This representation exhibits strong geometric interpretation properties, leading to more efficient and stable training. To support training, we construct a new dataset, ObjectPose9D, which aggregates images from diverse sources along with 9D pose annotations. To further address data imbalance issues, particularly performance degradation on low-frequency poses, we introduce a two-stage training strategy with reinforcement learning, where the second stage fine-tunes the model using a reward-based objective on rebalanced data. At inference time, we propose Disentangled Object Sampling, a technique that mitigates insufficient object generation and concept confusion in complex multi-object scenes. Moreover, by integrating user-specific personalization weights, SceneDesigner enables customized pose control for reference subjects. Extensive qualitative and quantitative experiments demonstrate that SceneDesigner significantly outperforms existing approaches in both controllability and quality.

Section: Introduction
Controlling the spatial properties of real-world images has been extensively explored, enabling users to manipulate the structure of interesting subjects or the overall layout of a scene [65,58,23,18,16,33,34,6,54]. However, most existing methods are confined to 2D space and often rely on densely annotated control maps as input, such as depth images. In contrast, 3D spatial control remains a largely underexplored challenge. Consider, for example, a designer aiming to arrange multiple pieces of furniture in a room, each with distinct sizes and orientations, or a user wishing to generate an image where a pet dog is turned away from the camera, gazing at the landscape ahead. These scenarios highlight the need for generation models that support 3D-aware multi-object control, a capability that is both essential for practical applications and insufficiently addressed by current approaches.
There have been several preliminary explorations in 3D-aware controllable generation [38,32,28,17,8,25,48]. For example, LOOSECONTROL [6] employs 3D bounding boxes for controlling the An astronaut walking on Mars A corgi dog on beach A Jeep on a muddy mountain road at sunset A rabbit and a turtle on the grass A bear riding motorbike on the ruins Panda, dog, and eagle in forest … in a lavender field ... on moss in forest … on the old book location and size of the object in 3D space. To enable orientation control, some methods take rotation angles around designated axes as input. Among them, Zero-1-to-3 [28] infers the novel perspective of reference subject, but relies on external inpainting tools [44] to create a complex background. Continuous 3D Words [8] and its following work [38] are constrained by unrealistic visual style and limited control over object quantity and pose diversity. More recently, ORIGEN [32] introduces precise orientation control but is restricted by its dependence on a one-step generative model, which hinders compatibility with widely used multi-step frameworks [44,39,10].
To address the aforementioned challenges, we enhance existing text-to-image models [10,44] by enabling 9D pose control of multiple objects within the same scene, as shown in Fig. 1. To support this goal, we first construct a new dataset, ObjectPose9D, which provides 9D pose annotations across diverse real-world scenarios. To build ObjectPose9D, we begin with the publicly available OmniNOCS dataset [20], which offers accurate pose annotations but is limited in object and background diversity. To overcome this limitation, we further annotate the large-scale MS-COCO dataset [26] with 9D poses to expand the variety of visual concepts and scene types. Specifically, we employ MoGe [57] and Orient Anything [59] to estimate 3D bounding boxes with orientations. All collected images and pose annotations are then carefully checked and manually refined by human annotators to ensure data quality. Together, these efforts yield ObjectPose9D, a diverse and richly annotated dataset for training image generation models with flexible multi-object 9D pose control.
this section cite: ['b64', 'b57', 'b22', 'b17', 'b15', 'b32', 'b33', 'b5', 'b53', 'b37', 'b31', 'b27', 'b16', 'b7', 'b24', 'b47', 'b5', 'b27', 'b43', 'b7', 'b37', 'b31', 'b43', 'b38', 'b9', 'b9', 'b43', 'b19', 'b25', 'b56', 'b58']

Section: Pose Control 2. Control Signals 3. Image Generation

this section cite: []

Section: Prompt

this section cite: []

Section: CNOCS Maps
Figure 2: Overview of SceneDesigner.
Then, how can 9D poses be efficiently encoded for controllable image generation? Some previous studies [8,38] project rotation angle into textural space and combine it with text embeddings. However, this way struggles to capture fine-grained orientation and precise spatial positioning. Inspired by the success of ControlNet-like architectures [65,67,41,34] in structural control, LOOSECONTROL [6] adopts 3D bounding boxes for 3D-aware guidance. However, this representation lacks orientation information. For example, a single bounding box may ambiguously correspond to either a front-or back-facing object, limiting controllability and reliability. To address this, we build upon the Normalized Object Coordinate System (NOCS) [52] to better encode geometric properties. Nevertheless, a traditional NOCS map requires a precise 3D shape of each object category, which is user-unfriendly and often difficult to acquire. To overcome this, we introduce the Cuboid NOCS (CNOCS) map to only consider a cuboid shape for general purpose. This simplification retains essential geometric cues while supporting category-agnostic pose encoding. Moreover, CNOCS map allows for flexible variants and generalizes well across object categories. Besides, to address the imbalanced pose distribution in real-world data, we introduce a two-stage training strategy with reinforcement learning. In the first stage, the model is trained on ObjectPose9D to learn basic pose controllability. In the second stage, it is fine-tuned to improve performance on low-frequency object poses by maximizing our proposed reward function under a balanced distribution. During inference, we apply Disentangled Object Sampling, a technique designed to mitigate concept fusion and insufficient generation in complex multi-object scenes. Furthermore, by integrating user-specific personalized weights, our method enables customized pose control for user-provided reference subjects. Within our framework, users can freely operate the cuboid-shape meshes in 3D space as demonstrated in Fig. 2, identifying the 3D location, size, and orientation of each object.
Our main contributions are:
1) We introduce ObjectPose9D, a dataset with rich real-world scenes and comprehensive 9D pose annotations, facilitating effective training of pose-controllable generation models. 2) We propose SceneDesigner, a framework that supports multi-object 9D pose control. It leverages our proposed pose representation, CNOCS map, which encodes 3D properties of objects from the camera view using a coarse cuboid abstraction. We further introduce a two-stage training strategy with reinforcement learning to mitigate data imbalance and enhance generalization. At inference time, Disentangled Object Sampling is employed to address insufficient or entangled object generation, while user-specific weights enable personalized pose control. 3) Extensive qualitative and quantitative experiments demonstrate that SceneDesigner significantly outperforms previous methods in both single-and multi-object scenarios, achieving high fidelity and controllability.
this section cite: ['b7', 'b37', 'b64', 'b66', 'b40', 'b33', 'b5', 'b51']

Section: Related Works
Controllable generation: Controllable image generation [65,67,41,34,63,24,55,45,13,2,4,21,35,47] enables fine-grained user control over visual content. Among them, ControlNet [65] and its following works [41,67] introduce a branched network to process the geometry guidance, resulting in the image with high structure fidelity. Other methods [2,4] like DreamBooth [45] and Textual Inversion [13] learn new concepts given by users, endowing the text-to-image (T2I) models with customization capability. Despite these efforts, controlling the 9D poses of objects still faces challenges. Most methods [23,58,61] can only handle 2D bounding boxes, which are insufficient to represent 3D properties. Recently, LOOSECONTROL [6] lifts this condition to 3D space, but struggles to represent precise orientation. A group of methods [38,8] leverage precise pose annotations from a synthetic dataset, and receive the rotation angles from users. However, they are constrained by limited controllability in intricate pose and multi-object scenarios, and poor generalization in in-the-wild images. The recent work ORIGEN [32] optimizes initial noise to maximize the constructed reward function. However, it cannot localize objects in designated area. Additionally, it relies on one-step generation models, hampering its compatibility with general models. Collectively, there are currently no approaches that can efficiently control the 9D poses of multiple objects in image generation. Beyond the image domain, recent advances in video generation have explored 3D-aware controllable synthesis. 3DTrajMaster [12] focuses on multi-entity trajectory control through MLP-encoded pose representations, while FMC [48] enables 6D pose control for cameras and objects. Meanwhile, CineMaster [54] extends the LOOSECONTROL [6] paradigm to video synthesis, providing intuitive cinematic control.
this section cite: ['b64', 'b66', 'b40', 'b33', 'b62', 'b23', 'b54', 'b44', 'b12', 'b1', 'b3', 'b20', 'b34', 'b46', 'b64', 'b40', 'b66', 'b1', 'b3', 'b44', 'b12', 'b22', 'b57', 'b60', 'b5', 'b37', 'b7', 'b31', 'b11', 'b47', 'b53', 'b5']

Section: 3D-aware image editing:
There is a growing body of methods [66,28,56,64,37,21] that focus on 3D-aware image editing. Among them, 3DIT [31] was trained on collected synthetic data, manipulating the object by translation or rotation. Similarly, Zero-1-to-3 [28] can generate novel views of the reference subject given by users. NeuralAssets [60] learns object-centric representations that enable disentangled control over object appearance and pose in neural rendering pipelines. Different from these methods, Diffusion Handles [37] leverages the point clouds estimated from the input image, and guides the sampling process through transformed points and depth-conditioned ControlNet [65]. 3DitScene [66] reconstructs the 3D Gaussian Splatting [19] of the scene, and manipulates the 3D Gaussians of the designated object through arbitrary 3D-aware operations.
Aligning the generation model with human preference: Based on the success of reinforcement learning in the field of natural language processing [36,43], some studies [11,7] make efforts to align the generated image with human preference. Some methods [62,40] like ReFL [62] maximizes the proposed reward score through backpropagation. Other RL-based methods like DDPO [7] leverage policy gradient algorithm for finetuning. Different from these methods, Diffusion-DPO [51] performs supervised learning on constructed preference dataset.
this section cite: ['b65', 'b27', 'b55', 'b63', 'b36', 'b20', 'b30', 'b27', 'b59', 'b36', 'b64', 'b65', 'b18', 'b35', 'b42', 'b10', 'b6', 'b61', 'b39', 'b61', 'b6', 'b50']

Section: Method

this section cite: []

Section: Preliminaries
Generative models: The generative models are designed to transform samples from a simple noise distribution into data from the target distribution. Diffusion-based methods [49,50,14] learn a noise prediction model, and remove the estimated noise from noisy image during sampling. From another perspective, Flow matching [3,27,30] achieves the goal by training a neural network to model the velocity field, moving the noise to data along straight trajectories.
this section cite: ['b48', 'b49', 'b13', 'b2', 'b26', 'b29']

Section: Normalized Object Coordinate System (NOCS):
To estimate 6D poses and sizes of multiple objects, NOCS [52] constructs the correspondences between pixels and normalized coordinates. It represents all objects in a normalized space while maintaining consistent category-level orientation. Then, the method constructs NOCS map, an RGB image representing each pixel's normalized coordinates, which is efficient for training the prediction model.
this section cite: ['b51']

Section: Overview
Our goal is to equip T2I models [44,39,10] with the controllability in 9D poses of multiple objects.
For clarity, we only present our flow-based implementation [10]. It's worth noting that our method can also be applied to diffusion models [44,39]. Formally, given the text prompt c p depicting the visual content that contains a set of objects {obj i }  To encode the 9D poses of general objects, i.e.,
this section cite: ['b43', 'b38', 'b9', 'b9', 'b43', 'b38']

Section: CNOCS Map: Effective Representation of 9-DoF poses
{P i } No i=1
, a straightforward method is to project the location, size, and orientation to embeddings, respectively, while integrating the features into network through cross-or self-attention mechanism [8,38,58,23,53,22]. In contrast, control maps used in ControlNet-like methods [65,41] offer an alternative approach for encoding 9D pose information. This spatial representation provides stronger structural constraints compared to direct projection methods. Our subsequent experimental results (Sec. 4) demonstrate that this map-based representation is more effective than direct embedding approaches, therefore, we adopt this encoding strategy for our framework.
We leverage the idea from NOCS [52] and propose CNOCS map to preserve the object's 3D properties, exhibiting strong geometry interpretation. A preliminary idea is to obtain the 3D bounding boxes of objects and render them in depth-sorted order depending on their distances from the camera view, which have already been explored in recent works [6]. Although this kind of representation can perceive the location and size of objects, it is viewpoint-dependent and insufficient to present precise orientation. Inspired by NOCS, it's necessary to establish correspondence between pixels indexed by (u, v) and their associated points on the object's surface, while encoding the point based on its coordinate in object space. However, NOCS has to access the precise 3D shape of each category, leading to cumbersome user input and hindering its application. In contrast, our CNOCS map alleviates the issues by using a cuboid shape inherited from 3D bounding boxes, while formalizing the encoding process to achieve better generality. The process of constructing CNOCS map is illustrated in Fig. 3 and formalized in Eq. 2.
x c u,v , y c u,v , z c u,v , oi = Intersect(u, v, {bbox i } No i=1 ), x o u,v , y o u,v , z o u,v = proj oi c2o (x c u,v , y c u,v , z c u,v ), x no u,v , y no u,v , z no u,v = normalize(x o u,v , y o u,v , z o u,v , bbox oi ), CNOCS[u, v] = f (x no u,v , y no u,v , z no u,v ),(2)
where "Intersect" operation finds the associated camera space point (x c u,v , y c u,v , z c u,v ) of [u, v]-indexed pixel on surface of 3D bounding box belonging to the oi-th object. Specifically, the oi-th object occludes other entities along the view ray from the pixel. Next, proj oi c2o transforms the coordinate from camera space to object space based on the pose of oi-th object. The "normalize" operation then maps the values to [-1, 1] using the side lengths of 3D bounding box. Finally, we assign the feature calculated by the encoding function f to [u, v]-indexed pixel in CNOCS map. There are many choices for f , leading to different variants such as: 1) Constant function. We can simply assign a predefined vector for any point, such as Euler angles. This variant is named C-CNOCS map. 2) Identity function. Similar to NOCS [52], we can directly use the coordinate (x no u,v , y no u,v , z no u,v ) as point embedding. Unlike the original NOCS map, the points are located on surface of 3D bounding box instead of the precise shape determined by CAD model. This variant is named I-CNOCS map. 3) Spherical harmonic function. (x no u,v , y no u,v , z no u,v ) can be further transformed into the form (θ no u,v , ϕ no u,v , r no u,v ) in spherical coordinate system. Then, we can construct a series of Laplace's spherical harmonics Y m l (θ no u,v , ϕ no u,v ) depending on a user-defined degree as point embedding, where l, m represent indices of degree and order, respectively. This variant is named S-CNOCS map. Based on empirical results (Sec. 4), we use I-CNOCS map as the pose representation since it is simple and effective.
this section cite: ['b7', 'b37', 'b57', 'b22', 'b52', 'b21', 'b64', 'b40', 'b51', 'b5', 'b51']

Section: Orientation Estimation

this section cite: []

Section: Point Cloud Estimation

this section cite: []

Section: 3D Box Estimation

this section cite: []

Section: 9D Poses Input Image
Figure 4: Annotation pipeline of 9D poses in MS-COCO [26].
this section cite: ['b25']

Section: ObjectPose9D Dataset
There are currently no existing datasets suitable for learning multi-object 9D pose control. Although some public datasets like Objectron [1] and OmniNOCS [20] contain object pose annotations, they are constrained by limited object categories and scene diversity. Therefore, we select several subsets from OmniNOCS as the base of our dataset, which cover common objects and scenarios in indoor and street scenes. Furthermore, we resort to the COCO dataset [26] to enlarge the data distribution and improve the model generalization ability. The annotation pipeline of 9D poses in COCO dataset [26] is shown in Fig. 4 and consists of the following steps:
1. Selection of objects. First, we obtain the suitable objects from each image using the following criteria: 1) The area of object mask must be limited within predefined lower and upper bounds, excluding the objects that are too small or large; 2) The orientation of the objects should be unambiguous. Categories with inherent orientation ambiguity, such as "bottle", are excluded. 2. Estimation of orientation and 3D bounding boxes. Next, we obtain the orientations and 3D
bounding boxes of suitable objects to construct CNOCS map. First, Orient Anything [59] is used to infer the object orientation and we filter the objects with low prediction confidence. In parallel, we get the initial geometry estimation (point clouds) of the scene via an advanced 3D reconstruction method [57], and identify the points belonging to the object based on the mask, while discarding the points that are too far from centroid by a threshold. Finally, 3D bounding boxes can be derived using the point clouds and predicted orientation, which tightly enclose the object points.
To ensure data quality, human annotators manually filter out low-quality samples and refine inaccurate annotations. Based on the estimated 3D bounding boxes and orientations, we then construct the CNOCS map representation using Eq. 2. Besides, Multimodal Large Language Model (MLLM) [5] is also employed to generate descriptive captions for each image, enriching the dataset with aligned textual information. These steps together yield the final dataset, ObjectPose9D. Further details on dataset statistics and construction procedures are provided in Appendix A.1.
this section cite: ['b0', 'b19', 'b25', 'b25', 'b58', 'b56', 'b4']

Section: SceneDesigner: Multi-object 9-DoF Pose Control
With the constructed dataset ObjectPose9D, we proceed to train our proposed method (SceneDesigner) for 9D pose control of multiple objects. The details about training and inference are introduced below.
Learning for 9-DoF pose control: We simply introduce ControlNet-like [65] branched network into the base model as overall architecture, which receives CNOCS map that encodes {P i }
this section cite: ['b64']

Section: No
i=1 , text prompt c p , and the noisy latent in the current step. v θ is learned in two stages. In the first stage, the parameters from the branched network are optimized using Eq. 1, learning the basic 9D pose controllability. However, the learned model exhibits inferior performance on low-frequency poses from ObjectPose9D, which is caused by imbalanced pose distribution. For example, the model fails to generate the back views of most animals due to biases from datasets [20,26]. Therefore, we resort to the technique from RLHF (Reinforcement Learning from Human Feedback). For aligning the object pose with the condition, we aim to maximize the introduced reward function below:
r ls (x, c p , {P i } No i=1 ) = Σ No i=1 1 -D ls (x, obj i , l i , s i ) , r o (x, c p , {P i } No i=1 ) = Σ No i=1 1 -D o (X (x, obj i ), o i ) , r(x, c p , {P i } No i=1 ) = γr ls (x, c p , {P i } No i=1 ) + λr o (x, c p , {P i } No i=1 ),(3)
where higher r ls represents better accuracy of location and size. For r ls , we use the advanced detection model [29] to estimate the 2D bounding box, and compute its Intersection over Union (IoU) with the projected one from 3D bounding box. On the other hand, r o assesses the orientation precision. For X , we crop the object area and feed it to Orient Anything [59]. Then, D o calculates the KL divergence between the target distribution and the estimated distribution from Orient Anything. The final reward r is calculated by combining r ls , r o through weighting factors γ, λ. Consequently, the objective of the second stage is to minimize:
E (cp,{Pi} No i=1 )∼B,x∼p θ (•|cp,{Pi} No i=1 ) -βr(x, c p , {P i } No i=1 ) + L prior ,(4)
where text prompts and CNOCS maps are sampled from constructed balanced data distribution B, and L prior is calculated as Eq. 1 for stabilizing the training [62]. However, naively training with Eq. ( 4) leads to huge GPU memory consumption for backpropagation throughout multi-step denoising process, which is infeasible. To reduce memory footprint, we leverage randomized truncated backpropagation and gradient checkpointing like in AlignProp [40], while feeding the coarse image estimated from intermediate step to the reward function instead of the clean one. The whole pipeline is presented in Algorithm 2. Through RL finetuning, the model outperforms in pose alignment with the input condition. More details about RL finetuning are provided in Appendix A.2.
Inference: After two-stage training, the model can effectively control the 9D pose of arbitrary object. However, the model fails to associate each object with its pose from CNOCS map in multi-object scenarios, since there is no restriction in constructing the correspondences. Furthermore, insufficient generation and attribute leakage often occur in generating multiple concepts. Therefore, we propose
Algorithm 1: Algorithm pipeline of Disentangled Object Sampling. Input: Initial noise ϵ sampled from N (0, I); text prompt cp consisting of entity names {obj i } No i=1 ; CNOCS map for each object {Pi} No i=1 ; CNOCS map of the whole scene P global ; sampling step T ; x0 = ϵ; for t in {0, 1 . . . , T -1} do vt = v θ (xt, t, cp, P total ); xt+1 = xt + vtdt; for i in {1, . . . , No} do Obtain object mask Mi from CNOCS map Pi; v i t = v θ (xt, t, obj i , Pi); / * it can be computed in parallel with vt * / x i t+1 = xt + v i t dt; xt+1 = (1 -Mi)xt+1 + Mix i t+1 ; x = xT ; Return: The generated image x.
Disentangled Object Sampling to alleviate the challenges, and the process is shown in Algorithm 1. Essentially, we combine multiple noisy latents at each denoising step using region masks, where each latent is sampled based on global or object-specific condition. Through Disentangled Object Sampling, the model is able to match each pose from CNOCS map with corresponding object. Furthermore, we can also load the personalized weights given by users and perform customized pose control of reference subjects.
this section cite: ['b19', 'b25', 'b28', 'b58', 'b61', 'b39']

Section: Experiment
Implementation details: The proposed SceneDesigner is based on Stable Diffusion 3.5 [10], training with 6 NVIDIA A800 80G GPUs. We use AdamW optimizer with an initial learning rate of 5e -6 . The resolution is set to 512 × 512 with 48 batch size. In the first stage, the parameters θ from the introduced ControlNet are updated for 45K iterations in the proposed ObjectPose9D. For the second stage, we further fine-tune the model with RL objective for 5K iterations. More details about training are provided in Appendix A.2. During inference, we only inject the conditions in initial 15 steps during sampling with 20 denoising steps.
Validation details: For validation metrics, we use mean Intersection over Union (mIoU) and spatial accuracy Acc ls for assessing the precision of location and size. Specifically, we use Grounding DINO [29] to detect generated objects. Acc ls is calculated as
Σ N i=1 I(IoUi>0.6) N
, where I is indicator function and N is total number of test cases. Similar to ORIGEN [32], we use the following two metrics for orientation evaluation:1) Abs.Err calculates the absolute error of azimuth angles (in degrees) between the input condition and the estimated one from Orient Anything [59] and 2)Acc.@22.5 • measures the accuracy with 22.5 • tolerance. Furthermore, CLIP [42] is used to estimate the text-image alignment and FID presents visual quality. Specifically, we randomly sample the reference images from LAION [46] to calculate FID. In addition, a user study is also conducted to complement the evaluation based on human preferences. For validation dataset, we introduce two benchmarks to assess the model performance in pose control of single-object and multi-object scenarios, named ObjectPose-Single and ObjectPose-Multi, which are obtained by estimating the 9D poses from validation part of COCO [26] as in Sec. 3.4. Among them, ObjectPose-Single is further divided into ObjectPose-Single-Front and ObjectPose-Single-Back for assessing the orientation accuracy in front-and back-facing scenarios, containing 247 and 156 samples, respectively. Besides, ObjectPose-Multi includes 229 cases.
this section cite: ['b9', 'b28', 'b31', 'b58', 'b41', 'b45', 'b25']

Section: Comparisons with State-of-the-Art Methods
Evaluation in single-object generation: This experiment evaluates the capability of single-object pose control. Although Zero-1-to-3 [28] exhibits considerable orientation controllability, it depends on the reference image from users and exhibits poor generalization in real-world images. Furthermore, since the codes of other relevant methods [32,38] were not open-sourced at the time of our experiment, they are also not discussed in our experiment. Consequently, we choose LOOSECONTROL (LC) [6] and Continuous 3D Words (C3DW) [8] as compared T2I methods due to their abilities for 3D-  aware control, while converting the pose condition into their required formats. As shown in the left of Fig. 5, the compared methods exhibit limited ability to control the location, size, and orientation simultaneously. Although LC [6] can control the location and size of the object effectively, the results present arbitrary orientation with poor quality and fidelity. On the other hand, C3DW is suffered by invariant image layout and unrealistic visual content. Besides, it can only process the front 180 • range of azimuths. As a result, the method lacks the controllability in more complicated orientations (e.g., back-facing case in column 4 of Fig. 5), and is unable to locate the object with a designated size. In contrast, SceneDesigner outperforms in both fidelity and quality. The quantitative results in Tab. 1 also demonstrate the superior performance of our method in pose alignment. As demonstrated by Acc ls and mIoU metrics in Tab. 1, SceneDesigner outperforms LC in controlling spatial location and size of the object, while C3DW exhibits poor precision since the objects are typically centered in the image. For orientation, C3DW achieves higher performance than LC, since the latter does not encode the orientation properties. However it cannot generate back-facing object, and suffers from the low-quality contents as indicated by the FID metric in Tab. 2. On the other hand, our method achieves considerable performances in both front-and back-facing settings with the help of efficient representation and the two-stage training strategy, while outperforming in both quality and text alignment.
this section cite: ['b27', 'b31', 'b37', 'b5', 'b7', 'b5']

Section: Evaluation in multi-object generation:
We also consider the pose control of multiple objects in the same scene. Since C3DW can only handle single object generation, LC is chosen as the compared method in this setting. As shown in the right of Fig. 5, it demonstrates poor performance in both
this section cite: []

Section: User Study:
We additionally conduct a user study to assess the methods in the following aspects: image quality, pose fidelity and text-to-image alignment. Tab. 4 demonstrates the evaluation results from human beings, where the scores are normalized and the higher value indicates better performance. Specifically, we employed 20 volunteers to evaluate outcomes from each method. For each score, we average the results and normalize it by the maximum value.
this section cite: []

Section: Ablation Studies
We conduct comprehensive ablation studies to validate the effectiveness of each component in our proposed SceneDesigner. The quantitative results are summarized in Tab. 3.
Pose Conditioning: To validate our proposed CNOCS map, we compare it with three alternative conditioning strategies: 1) a C-CNOCS map that assigns constant Euler angles to the object region; 2) a variant that directly injects 9D pose embeddings via attention; and 3) a training-free baseline (prompt only) that converts the pose into a textual description based on templates. As shown in Tab. 3, all three baselines underperform, confirming the superiority of our representation.
this section cite: []

Section: Dataset and Training Strategy:
We then analyze the impact of our dataset and two-stage training strategy. First, we train a model using only OmniNOCS data. As shown in Fig. 6 (left) and Tab. 3, the limited categories in OmniNOCS lead to poor generalization on unseen classes (e.g., rabbit) and a significant performance drop. Second, we evaluate the model checkpoint from the first stage, prior to RL finetuning. This model fails to generate back-facing objects, and its orientation accuracy is considerably lower. These comparisons confirm the importance of both our diverse ObjectPose9D and the RL finetuning stage.
Multi-Object Generation with DOS: For multi-object generation scenarios, we assess the effect of the proposed DOS. The results in the right part of Fig. 6 and Tab. 3 highlight its effectiveness in mitigating concept confusion and ensuring each object correctly corresponds to its specified pose in the CNOCS map.
this section cite: []

Section: Limitations and Impacts

this section cite: []

Section: Limitations
Although SceneDesigner achieves high-fidelity pose control, it cannot control the precise shape of the object. Furthermore, the performance in the multi-object scenario is constrained by the inherent capability of the base model. As mentioned in previous literature, increasing the number of semantic concepts in text prompt exacerbates insufficient generation and attribute leakage. While our Disentangled Object Sampling technique mitigates this issue, it introduces additional computation. Therefore, we will explore how to enhance the alignment with conditions in multi-object generation while maintaining computational efficiency in our future work.
this section cite: []

Section: Social Impacts
Positive societal impacts: SceneDesigner supports users with ability in multi-object 9D pose control without extensive resources or professional equipment, while providing customized pose control of user-provided subjects. This capability proves particularly valuable for applications like virtual/ augmented reality and product design, where spatial control is essential.
Potential negative societal impacts: Multi-object 9D pose control raises concerns about potential misuse for generating deceptive content in sensitive areas such as political manipulation and social media, where inaccurate poses could spread misinformation. Without proper safeguards, such as detection methods, ethical guidelines, and public awareness, this technique could be exploited to undermine the trust in digital media.
this section cite: []

Section: Mitigation strategies:
Developing and adhering to strict ethical guidelines for multi-object 9D pose control technologies helps mitigate misuse risks. This includes enforcing usage restrictions for sensitive generation and ensuring transparency in generated outputs through traceability measures.
this section cite: []

Section: Conclusion
We introduce SceneDesigner that achieves multi-object 9D pose control within the same scene.
Our key insight is introducing CNOCS map to encode the 9D pose of general objects, preserving the 3D properties and exhibiting high geometry interpretation. This representation accelerates the convergence and facilitates to provide a user-friendly interface for creating 3D-aware conditions. For learning the newly added control modules, we resort to the public dataset with comprehensive 9D pose annotations, while enhancing the diversity through annotating large-scale datasets. To alleviate the poor performance in low-frequency poses caused by imbalanced data distribution, we introduce a two-stage training process, where the second stage maximizes the proposed reward function to enhance the alignment of object pose with input conditions. During inference, our Disentangled Object Sampling associates each object with the corresponding pose condition, avoiding the confusion in multi-object generation. Furthermore, our method can also achieve customized pose control of reference subjects given by users. Finally, qualitative and quantitative experiments demonstrate that our method outperforms existing methods in both single-and multi-object scenarios.
this section cite: []

Section: References
Ref_id:b0 Title: Objectron: A large scale dataset of object-centric videos in the wild with pose annotations Year: (2021)
Ref_id:b1 Title: A neural space-time representation for text-to-image personalization Year: (2023)
Ref_id:b2 Title: Stochastic interpolants: A unifying framework for flows and diffusions Year: (2023)
Ref_id:b3 Title: Break-a-scene: Extracting multiple concepts from a single image Year: (2023)
Ref_id:b4 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b5 Title: LOOSECONTROL: lifting controlnet for generalized depth conditioning Year: (2024)
Ref_id:b6 Title: Training diffusion models with reinforcement learning Year: (2023)
Ref_id:b7 Title: Learning continuous 3d words for text-to-image generation Year: (2024)
Ref_id:b8 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b9 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b10 Title: Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models Year: (2023)
Ref_id:b11 Title: 3dtrajmaster: Mastering 3d trajectory for multi-entity motion in video generation Year: (2025)
Ref_id:b12 Title: Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion Year: (2023)
Ref_id:b13 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b14 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b15 Title: Cocktail: Mixing multi-modality controls for text-conditional image generation Year: (2023)
Ref_id:b16 Title: Orientdream: Streamlining text-to-3d generation with explicit orientation control Year: (2025)
Ref_id:b17 Title: SCEdit: Efficient and controllable image diffusion generation via skip connection editing Year: (2024)
Ref_id:b18 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b19 Title: OmniNOCS: A unified nocs dataset and model for 3d lifting of 2d objects Year: (2024)
Ref_id:b20 Title: Customizing text-to-image diffusion with object viewpoint control Year: (2024)
Ref_id:b21 Title: TrackDiffusion: Tracklet-conditioned video generation via diffusion models Year: (2023)
Ref_id:b22 Title: GLIGEN: open-set grounded text-to-image generation Year: (2023)
Ref_id:b23 Title: PhotoMaker: Customizing realistic human photos via stacked ID embedding Year: (2023)
Ref_id:b24 Title: Anyi2v: Animating any conditional image with motion control Year: (2025)
Ref_id:b25 Title: Microsoft COCO: common objects in context Year: (2014)
Ref_id:b26 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b27 Title: Zero-1-to-3: Zero-shot one image to 3d object Year: (2023)
Ref_id:b28 Title: Grounding dino: Marrying dino with grounded pre-training for open-set object detection Year: (2024)
Ref_id:b29 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b30 Title: OBJECT 3DIT: Language-guided 3d-aware image editing Year: (2023)
Ref_id:b31 Title: ORIGEN: Zero-shot 3d orientation grounding in text-to-image generation Year: (2025)
Ref_id:b32 Title: FreeControl: Training-free spatial control of any text-to-image diffusion model with any condition Year: (2024)
Ref_id:b33 Title: T2I-Adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models Year: (2024)
Ref_id:b34 Title: Compositional text-to-image generation with dense blob representations Year: (2024)
Ref_id:b35 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b36 Title: Diffusion handles enabling 3d edits for diffusion models by lifting activations to 3d Year: (2024)
Ref_id:b37 Title: Compass Control: Multi object orientation control for text-to-image generation Year: (2025)
Ref_id:b38 Title: SDXL: improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b39 Title: Aligning text-to-image diffusion models with reward backpropagation Year: (2023)
Ref_id:b40 Title: UniControl: A unified diffusion model for controllable visual generation in the wild Year: (2023)
Ref_id:b41 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b42 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b43 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b44 Title: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation Year: (2023)
Ref_id:b45 Title: LAION-400M: open dataset of clip-filtered 400 million image-text pairs Year: ()
Ref_id:b46 Title: A survey of multimodal-guided image editing with text-to-image diffusion models Year: (2024)
Ref_id:b47 Title: Free-Form Motion Control: Controlling the 6d poses of camera and objects in video generation Year: (2025)
Ref_id:b48 Title: Generative modeling by estimating gradients of the data distribution Year: (2019)
Ref_id:b49 Title: Sliced score matching: A scalable approach to density and score estimation Year: (2019)
Ref_id:b50 Title: Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b51 Title: Normalized object coordinate space for category-level 6d object pose and size estimation Year: (2019)
Ref_id:b52 Title: Generating rich and controllable motions for video synthesis Year: (2024)
Ref_id:b53 Title: Cinemaster: A 3d-aware and controllable framework for cinematic text-to-video generation Year: (2025)
Ref_id:b54 Title: InstantID: Zero-shot identitypreserving generation in seconds Year: (2024)
Ref_id:b55 Title: Diffusion models are geometry critics: Single image 3d editing using pre-trained diffusion priors Year: (2024)
Ref_id:b56 Title: MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision Year: (2025)
Ref_id:b57 Title: Instancediffusion: Instance-level control for image generation Year: (2024)
Ref_id:b58 Title: Orient Anything: Learning robust object orientation estimation from rendering 3d models Year: (2024)
Ref_id:b59 Title: Neural Assets: 3d-aware multi-object scene synthesis with image diffusion models Year: (2024)
Ref_id:b60 Title: Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion Year: (2023)
Ref_id:b61 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b62 Title: Text compatible image prompt adapter for text-to-image diffusion models Year: (2023)
Ref_id:b63 Title: Image Sculpting: Precise object editing with 3d geometry control Year: (2024)
Ref_id:b64 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b65 Title: 3DitScene: Editing any scene via language-guided disentangled gaussian splatting Year: (2024)
Ref_id:b66 Title: Uni-ControlNet: All-in-one control to text-to-image diffusion models Year: (2023)
