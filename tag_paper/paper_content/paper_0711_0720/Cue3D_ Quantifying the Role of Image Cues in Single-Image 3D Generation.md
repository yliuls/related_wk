Title: Cue3D: Quantifying the Role of Image Cues in Single-Image 3D Generation
Abstract: Figure 1: We present Cue3D, the first comprehensive, model-agnostic framework for quantifying the influence of individual image cues in single-image 3D generation. Left: Our unified evaluation of single-image 3D generation methods. Right: Performance robustness to the perturbation of each cue, lower values indicate higher importance. We show representative methods on Toys4K dataset for clarity; additional figures are available in the Appendix.

Section: Introduction
Generating a 3D model from a single 2D image is a long-standing goal in computer vision, with broad applications in content creation, AR/VR, and graphics. Humans effortlessly recover 3D shape from a single view by exploiting a variety of monocular cues [3,16,33,42]. Decades of research in classical computer vision studied these explicit monocular cues for shape inference, including shading patterns [21,63], texture cues [40], silhouette contours [27], and many more. Recently, a new generation of single-image-to-3D methods has dramatically advanced the state of the art, fueled by large datasets [11] and advances in deep generative models [19]. These approaches can be grouped into three prominent categories: (i) Regression-based models that directly predict a 3D representation from the input image via a feed-forward network (e.g., LRM [20], SF3D [6]), (ii) Multi-view methods that generates novel views consistent with the input image, then regress to a 3D model (e.g., CRM [55], LGM [49], InstantMesh [60]), and (iii) Native 3D generative models that treat single-image-to-3D as a conditional generation problem in a learned 3D latent space (e.g., Trellis [56] and Hunyuan3D-2 [64]). These approaches have enabled fast generation of textured 3D meshes from a single image, with impressive fidelity and generalization far beyond earlier methods.
Despite this rapid progress, the interpretability of single-image 3D networks remains largely underexplored. Current models are learned end-to-end on 3D supervision, and they operate as complex black boxes: we have little understanding of what information they rely on to infer 3D shape from a single image. Do these networks internally exploit the same set of visual cues as classical methods [21,27,40,63], or do they rely on other information such as high-level semantics? Improving transparency in this process is important both scientifically, to connect with vision science and inform future model design, and practically, to diagnose failure modes and biases of these 3D generators.
To address this gap, we present Cue3D, the first comprehensive, model-agnostic framework for quantifying the influence of individual image cues in single-view 3D generation. We begin by establishing a unified benchmark covering seven state-of-the-art methods spanning regression-based, multiview, and native 3D generative paradigms. We evaluate them on two standard datasets (GSO [12], Toys4K [47]). For each predicted mesh, we assess (1) both 2D appearance and 3D geometric quality for the entire shape, (2) 2D and 3D quality of the visible surface from the input viewpoint, and (3) the agreement between output and ground-truth symmetry. As summarized in Figure 1 left, native 3D generative models consistently outperform other approaches across all metrics.
We then systematically quantify the significance of each image cue. Building on meaningful perturbations [14], we disable or modify specific cues, such as silhouette shape, shading, texture semantics, perspective, and local continuity, and measure the resulting degradation in 3D output quality. Our perturbation analysis uncovers how modern single-image 3D models leverage image cues, revealing the following key insights. (1) Meaningfulness of shape, not texture, dictates generalization. For models to generalize, the input image must indicate a meaningful shape that does not significantly deviate from the training distribution. When we disrupt this cue by providing the models with a stochastic combination of textured primitive shapes [57], every method collapses with distinct failure modes. In contrast, the models perform surprisingly well on meaningless or random textures, with the best-performing models showing near perfect generalization. (2) Semantics alone are not enough; Geometric cues are crucial. Using a state-of-the-art style-transfer method [59], we convert images into artistic styles that preserve high-level semantics but often disrupts geometric cues like realistic shading and texture, as shown in Figure 2. We observe a significant drop on the performance compared to the original images, underscoring the continued importance of geometric cues. (3) Shading is more important than texture. To dissect the contribution of different geometric cues, we dive deeper into the image formation process. Surprisingly, even when all recognizable textures are replaced by procedural noise, natural patterns, or flat gray, for several methods, the quality of the 3D outputs remain almost unchanged, as long as the shading is kept intact. However, removing shading causes a noticeable performance decline. We further discover an interplay between shading and texture cues: intact shading alone suffices to uphold performance regardless of texture content, but when shading is removed, preserving the original texture yields better results than substituting with procedural textures or uniform color. (4) Models are overly sensitive to silhouette and occlusion. Dilating the object mask (without altering interior pixels) inflicts severe errors on regression-based and multi-view methods, whereas one native 3D generator remains relatively robust. In contrast, occlusion of both silhouette and image content dramatically degrades all approaches. (5) Other cues have diverse impact. Perturbing perspective, edge, and local-continuity signals produce measurable performance drops that vary across model categories, which we provide thorough analysis in the experiments section.
Cue3D establishes the first unified, model-agnostic framework for dissecting how modern single-image 3D generators exploit individual visual cues. Our perturbation study reveals that shape, rather than texture, meaningfulness dictates generalization. Geometric cues, especially shading, contribute significantly the 3D generation process. These models may overly rely on the provided silhouette. Meanwhile, edges, perspective, and local continuity each have distinct effects on different model families. By quantifying these dependencies across state-of-the-art approaches, Cue3D deepens our scientific understanding of image-based 3D generation, and provides potential guidance for designing more transparent, robust, and controllable single-image 3D generation methods.
this section cite: ['b2', 'b15', 'b31', 'b40', 'b20', 'b61', 'b38', 'b26', 'b10', 'b18', 'b19', 'b5', 'b53', 'b47', 'b58', 'b54', 'b62', 'b20', 'b26', 'b38', 'b61', 'b11', 'b45', 'b13', 'b55', 'b57']

Section: Related Work
Single-Image to 3D. Recent advances in single-image-to-3D generation have converged on three principal paradigms. (1) Regression-based methods [6,20,22,52,54] employ neural networks to directly predict a 3D representation, such as voxels, deformed meshes, or implicit fields, from encoded image features in a single forward pass. For example, LRM [20] and its successors [6,52] utilize transformer backbones to learn triplane representations, which are then rendered volumetrically, achieving both high fidelity and efficient inference. (2) Multi-view approaches [2,38,45,49,55,60] follow a two-stage pipeline: first synthesizing multi-view RGB images [38], depth or coordinate maps [35,55], normal maps [37,39], or Gaussian splats [49], and then reconstructing 3D structure from these intermediate multi-view representations. Decoupling view synthesis from geometry enables the reuse of powerful 2D generative priors trained on billions of images [43], providing an especially strong texture prior. (3) Native 3D generative models [25,31,32,53,56,62,64] combine a VAE-based latent encoding of 3D data [26,28] with a diffusion or flow model to generate high-quality and diverse 3D samples. Methods differ in their latent structures, input formats, and output representations: for instance, Hunyuan3D-2 [64] encodes point clouds to produce texture-free signed distance fields, while Trellis [56] proposes a sparse structured latent combining geometric and visual features, allowing flexible decoding into radiance fields, Gaussian splats, or meshes. Despite the iterations of approaches, it remains unclear what image cues these models rely on when producing the 3D output. In this paper, we systematically investigate how different single-image to 3D frameworks extract and transform visual signals from images cues into 3D representations.
Image Cues. Humans infer 3D structure from single images by integrating multiple monocular cues. Studies in developmental psychology and psychophysics show that the human visual system encodes properties like surface depth and orientation [10,27,46], and that internal object representations adhere to 3D constraints [44]. In contrast to humans' seamless cue integration, classical computer vision approaches explicitly leverage specific visual cues for shape inference-such as shape-from-shading [21,23], texture gradients [24,41], silhouettes [29,34], contours and junctions [9], perspective effects [17], and symmetry priors [4,51]. Modern deep models instead learn these
this section cite: ['b5', 'b19', 'b21', 'b50', 'b52', 'b19', 'b5', 'b50', 'b1', 'b36', 'b43', 'b47', 'b53', 'b58', 'b36', 'b33', 'b53', 'b35', 'b37', 'b47', 'b41', 'b24', 'b29', 'b30', 'b51', 'b54', 'b60', 'b62', 'b25', 'b27', 'b62', 'b54', 'b9', 'b26', 'b44', 'b42', 'b20', 'b22', 'b23', 'b39', 'b28', 'b32', 'b8', 'b16', 'b3', 'b49']

Section: Multi-View Images

this section cite: []

Section: Feed-Forward Reconstructor

this section cite: []

Section: Multiview Image Diffusion

this section cite: []

Section: Multiview Models

this section cite: []

Section: Input Image Output

this section cite: []

Section: Native 3D Generative Models

this section cite: []

Section: Input Image

this section cite: []

Section: Shape Generator

this section cite: []

Section: Condition

this section cite: []

Section: Shape Latent

this section cite: []

Section: Texture Generator

this section cite: []

Section: Condition

this section cite: []

Section: Output

this section cite: []

Section: Decoder Encoder

this section cite: []

Section: 3D Representations

this section cite: []

Section: Regression-Based Models

this section cite: []

Section: Input Image Output
Figure 3: Illustration of the three single-image 3D generation paradigms evaluated in this paper: regression-based methods (OpenLRM [18], SF3D [6]), multi-view approaches (CRM [55], LGM [49], InstantMesh [60]), and native 3D generative models (Trellis [56], Hunyuan3D-2 [64]).
this section cite: ['b17', 'b5', 'b53', 'b47', 'b58', 'b54', 'b62']

Section: Input Image

this section cite: []

Section: SF3D Instantmesh
Hunyuan3D-2 visual priors implicitly in an end-to-end manner, inspring us to explore the role of these cues in state-of-the-art models.
this section cite: []

Section: Visual Cue Interpretability.
Interpreting the decision-making process of black-box models, especially with respect to the visual cues they exploit, remains an open challenge. A common strategy is input perturbation, where carefully crafted modifications are applied to input data to observe the resulting changes in model output [14,15,50]. For example, Geirhos et al. [15] generate images in which object shape and surface texture are semantically misaligned, revealing the relative importance of each cue in image classification models. Other approaches include latent-space probing, which trains auxiliary networks to investigate whether the internal representations of a pre-trained model fit certain downstream tasks [5,13], and metric-based benchmarking, where models are compared across artificially curated datasets designed to emphasize specific visual attributes or cues [65]. Our work is inspired by these cue-based analysis methods but differs in key ways. Rather than solely focusing on classification or probing general features for diverse downstream tasks, we focus on presenting an in-depth analysis within the scope of single-image 3D generation. We introduce a model-agnostic framework that systematically applies controlled perturbations to distinct image cues and quantifies their effect. By evaluating a range of recent 3D architectures and employing both 2D and 3D performance metrics, our approach provides a faithful and comprehensive view of how state-of-the-art models leverage visual cues during singele-image 3D generation.
this section cite: ['b13', 'b14', 'b48', 'b14', 'b4', 'b12', 'b63']

Section: Cue3D

this section cite: []

Section: Evaluation Settings
Methods. We compare seven recent single-image-to-3D methods that collectively cover all three prevailing paradigms. In particular, we select OpenLRM [18] and SF3D [6] from regression-based networks, CRM [55], LGM [49] and InstantMesh [60] from multi-view reconstruction approaches, and Trellis [56] and Hunyuan3D-2 [64] from native 3D generative methods. We use the official implementation for all methods and evaluate mesh outputs in a unified way. We use 8 NVIDIA L40S GPU for all our experiments.
this section cite: ['b17', 'b5', 'b53', 'b47', 'b58', 'b54', 'b62']

Section: Datasets.
We select two standard evaluation datasets for all methods: GSO [12], a dataset of highquality scanned household items, and Toys4K [47], a collection of user-created 3D toy objects. We manually remove geometrically trivial objects (e.g., boxes) and balancing over-represented categories from these datasets. Our final evaluation sets contains 412 objects from the cleaned GSO dataset and 500 randomly sampled objects from the cleaned Toys4K dataset. Each object is rendered in Blender from a random camera pose (azimuth/elevation sampled within fixed limits) under a random Poly Haven [1] HDRI lighting. More implementation details are in the appendix. To probe performance on shapes without semantic meaning, we additionally use Zeroverse [57], a procedurally generated dataset built from random assemblies of textured primitive. Zeroverse exhibits rich local geometric detail, but the shapes themselves are not meaningful, as it significantly deviate from the training distribution of single-image-to-3D methods.
Overall Quality. We evaluate both 2D appearance fidelity and 3D geometric quality of the 3D mesh results. We align the output mesh to the groundtruth following [6]. For appearance fidelity, we report PSNR, SSIM and LPIPS between rendered output meshes and groundtruth meshes. We render 16 views for each object with 8 uniform azimuth and 2 elevations. For geometry quality, we report the Chamfer Distance (CD) and F-scores at different thresholds to quantify the overall shape quality. The Chamfer distance between two point clouds P 1 = {x i ∈ R 3 } n i=1 and P 2 = {x j ∈ R 3 } m j=1 . is defined as:
chamfer(P 1 , P 2 ) = 1 2n n i=1 |x i -NN(x i , P 2 )| + 1 2m m j=1 |x j -NN(x j , P 1 )| (1
)
where NN(x, P ) = arg min x ′ ∈P ∥x -x ′ ∥ denotes the nearest neighbor of source point x in point cloud P .
Visible Surface Quality. Beyond assessing overall mesh quality, we specifically evaluate how accurately the predicted mesh aligns with the ground truth at the input image's viewpoint. We render RGB images of the output meshes from this viewpoint, obtain the corresponding depth map, and back-project the depth map into point clouds using the ground truth camera parameters. Subsequently, we employ the previously described 2D and 3D metrics on these rendered images and point clouds to quantitatively measure the quality of visible surfaces.
this section cite: ['b11', 'b45', 'b0', 'b55', 'b5']

Section: Symmetry.
We further analyze the predicted object's symmetry agreement with the ground truth. Adopting the symmetry groundtruth generation procedure from [36], we apply a fixed threshold to identify planes of reflection symmetry in both predicted and ground truth meshes. For each method, we compute a binary symmetric-or-not F1 score across all predicted objects relative to their groundtruth counterparts.
this section cite: ['b34']

Section: Perturbations
We assess the importance of individual image cues through targeted perturbations. By selectively removing one cue while preserving others, significant performance degradation indicates the model's reliance on that cue. Conversely, minimal performance changes suggest the model's invariance to that cue. Additionally, preserving a single cue while removing most others can highlight its information contribution in the model's inference process. Below, we introduce the cues and their corresponding perturbations examined in this paper, illustrated in Figure 2. Additional perturbation analyses are detailed in the appendix.
this section cite: []

Section: Style.
We perturb geometric cues while preserving semantic content through reference-based style transfer [59]. We select six distinct styles: ink wash, line art, pointillism, flat design, oil painting, and sculpture. We manually curate five exemplar images per style. Each object image undergoes style transfer using a randomly selected style exemplar for each of the six styles. This approach preserves core 3D structure perceptually while altering geometric cues like shading and texture, as shown in Figure 2.
this section cite: ['b57']

Section: Shading & Texture.
Given their prominence as geometric cues, we jointly analyze shading and texture perturbations within the rendering pipeline. We perturb shading by rendering diffuse maps in Blender. Specifically, since the groundtruth texture in the GSO dataset has baked-in lighting, we employ an image delighting method [64] to remove baked-in lighting for the GSO dataset. Texture perturbations involve swapping original textures with alternatives such as uniform checkerboards, Perlin noise, random textures from Poly Haven [1], and uniform gray. Each texture variant is rendered both with and without lighting (diffuse).
this section cite: ['b62', 'b0']

Section: Silhouette and Occlusion.
Silhouette captures global shape information, and many models explicitly takes object mask as input. We investigate its influence through mask dilation and occlusion. We first dilate the silhouette (alpha mask) of each object by a fixed pixel width, leaving other cues intact. Subsequently, we simulate occlusion by placing scaled masks of randomly selected objects from the dataset onto the original mask boundaries, creating weak, medium, and strong occlusion conditions. Though occlusion partially hides image content, humans typically can still mentally reconstruct the Table 2: Evaluation results on the Zeroverse dataset of shapes without semantic meaning. Performance significantly drops compared to GSO and Toys4K, underscoring the significance of shape meaningfulness.
complete 3D shape by leveraging shape priors. These variants test the model's capability to infer complete 3D structures despite partial visibility. Additional perturbation scenarios to the silhouette are presented in the appendix.
this section cite: []

Section: Edges.
Edges are analyzed due to their role in separating surfaces and indicating curvature. We first extract edge maps using the Canny algorithm from input images. Two perturbation strategies are used: one replaces all internal object cues (except silhouette) with edge maps alone, evaluating if edges can sufficiently provide information for shape inference. The other softens edges by applying Gaussian blurring only in the local neighborhood of detected edges, merging adjacent surface regions visually. Significant performance drops under these perturbations would highlight the model's reliance on precise edge information, while negligible drop would indicate invariance. Additional edge extraction methods and results are included in the appendix.
Perspective. Perspective cue could indicate vanishing points and spatial relationships. This cue is perturbed by switching the rendering camera to an orthographic projection. Eliminating perspective effects enables evaluation of the model's dependence on perspective cues.
this section cite: []

Section: Local Continuity.
To assess sensitivity to local structural details, we perturb local continuity by splitting image foreground into grids of n × n pixels and shuffling pixels within each grid cell independently. This maintains global structure while disrupting local detail continuity. Greater performance degradation under this perturbation reflects higher sensitivity to local information.
this section cite: []

Section: Results

this section cite: []

Section: Unified Evaluation
We begin by conducting a unified evaluation of all seven methods on the GSO and Toys4K datasets. We present the summary of the results in Figure 1 (left), and the full evaluation details in Table 1. Models LGM OpenLRM CRM SF3D InstantMesh Hunyuan3D-2 Trellis Cue Category Geometric cues (Style) Shading & Texture Silhouette Occlusion Edges Perspective Local continuity Cue Category Geometric cues (Style) Shading & Texture Silhouette Occlusion Edges Perspective Local continuity Figure 5: Quantitative analysis of image cue perturbations on single-image 3D generation. We report Chamfer Distance (CD ×1000 for clarity; lower is better) for each model under different perturbations. A larger increase in CD indicates greater performance degradation, revealing the model's reliance on the perturbed cue. A tabulated version of these results is available in Appendix.
Our results show that the native 3D generative methods, see Hunyuan3D-2 and Trellis in Table 1, clearly outperform other methods across both datasets. Generally, these two methods are closely followed by InstantMesh, the best-performing multi-view method, and SF3D, the leading regressionbased method, followed by the remaining methods.
Regarding 2D appearance quality, as shown in Table 1 Overall 2D and Visible Surface 2D columns, native 3D generative methods have only marginal improvements in terms of PSNR and SSIM compared to other methods. However, they substantially outperform the alternatives in terms of LPIPS scores. This suggests that, although pixel-level statistics appear similar across methods, the native 3D generative methods more accurately capture higher-level visual information encoded by deep features.
The most substantial advantage of native 3D generative methods emerges in their 3D geometry quality, see Table 1 Overall 3D and Visible Surface 3D columns. These methods exceed the next-best method by over 10 points in overall geometry evaluation and by more than 4 points on visible surfaces under our CD×1000 metric. The visible surface quality assessments closely align with the overall geometry evaluations. Additionally, native 3D generative methods excel significantly in modeling symmetry, see Table 1 Symmetry column. They closely match the ground-truth symmetry across both datasets.
Comparing the two top methods, Hunyuan3D-2 and Trellis achieve similar 2D quality despite differences in their texture modeling approaches. Trellis demonstrates slightly better overall 3D quality, whereas Hunyuan3D-2 slightly excels in symmetry and visible surface quality. These insights provide valuable guidance for selecting the most appropriate method for practical, real-world applications.
this section cite: []

Section: Image Cues Analysis
In degraded performance. We show qualitative examples in Figure 6. More qualitative results are available on the project webpage and in the Appendix.
this section cite: []

Section: Shape Meaningfulness.
We probe the role of shape meaningfulness in two complementary ways: (i) Zeroverse and (ii) CutMix on GSO. First, we use Zeroverse [57], a dataset comprising procedurally generated combinations of textured primitive shapes. We show qualitative results of representative methods in Figure 4, and the quantitative evaluations in Table 2. Figure 4 shows that the input image does not correspond to a meaningful shape, and has a significant gap to the training distribution. Performance notably declines across both 2D and 3D metrics compared to the more meaningful GSO and Toys4K datasets, confirming that meaningful shape in the input images are critical for the generalization of single-image 3D generation. Native 3D generative methods generally maintain the highest overall quality, while CRM best recovers visible surfaces in 2D, and SF3D and Hunyuan3D-2 perform best in visible surface 3D quality.
We further examine how the absence of shape meaningfulness influences different failure modes qualitatively in Figure 4 and quantitatively in the Appendix. Regression-based methods produce smooth, averaged back surfaces. We quantify this phenomenon by measuring the difference between each normal and the average normal in its local neighborhood, normalized against the ground truth, and we see a substantial drop of this metric on Zeroverse. Multi-view methods fail due to inconsistencies in synthesized views, as evidenced by decreased pairwise DINOv2 similarity scores, contributing to degraded 3D performance. Native 3D generative methods, lacking meaningful shape information, tend to hallucinate symmetrical completions, resulting in higher false-positive symmetry detections. See appendix for the detailed results of these experiments. These diverse failure modes underline the crucial role of meaningful shape cues, particularly for reconstructing occluded surfaces.
Meanwhile, to test shape meaningfulness with minimal domain shift, we introduce shape CutMix variants on GSO. We combine parts of different GSO meshes to perturb shape meaningfulness, while keeping appearance statistics similar. We conducted several shape CutMix experiments of varying difficulty, Given two meshes M and N :
1) Half-and-half : We mix half of mesh M with the other half of mesh N to construct a new mesh from GSO meshes. This limits the distribution shift and preserves many local and global shape cues (e.g., surface smoothness, local symmetry), and also preserves a significant amount of shape meaningfulness to human perception. We show three variants: front-back, left-right, and top-bottom.
2) Default CutMix. We follow the CutMix [61] paper and randomly sample an axis-aligned 3D cube within the bounding cube of the object. We replace the part of mesh M that falls into the cube with the part from mesh N that falls into the same cube. When sampling the cube, we pin one of its corners at the corner of the object bounding cube to avoid significant discontinuity in the output shape. The length ratio (length sampled cube /length bounding cube ) is uniformly sampled from [0.4, 0.6]. Most parts of the object M are outside the chosen cube and remain intact. Meanwhile, the local shape cues are mostly preserved.
3) CutMix by Octant. We center each mesh and split it into 8 octants by the coordinate planes (XY, YZ, and XZ planes). Then we replace the part in each octant by the corresponding part from other random meshes from the same dataset. This variant still preserves the local shape cues, and it has a significantly smaller distribution gap than Zeroverse compared to our original evaluation data (GSO).
Results in Figure 5(a) show that all variants substantially hurt performance. Notably, CutMix by Octant causes a performance drop similar to Zeroverse despite a smaller domain gap. Standard CutMix, which modifies only about 1/8 of the mesh volume, still results in large drops (e.g., 20 points for Hunyuan3D-2). Even minimal half-and-half perturbations, where shape meaningfulness is mostly preserved, typically cause performance drops of over 10 points, which is greater than for most other cues. This confirms that shape meaningfulness is a dominant cue for generalization.
this section cite: ['b55', 'b59']

Section: Geometric Cues (Style).
To explore geometric cues broadly, we apply style transfer to preserve semantics while altering geometric cues (Figure 2). Figure 5 summarizes the results on GSO and Toys4K. Performance significantly deteriorates under style perturbations. Sculpture-style images retain the most geometric information, thus yielding the smallest performance drops in general. Note that lower-performing methods show less degradation not because of their robustness, but due to metric saturation. Overall, semantics alone are insufficient; geometric cues are essential for reliable 3D inference.
this section cite: []

Section: Shading and Texture.
We dissect geometric cues further by separately manipulating shading and texture, which are historically established cues for shape inference. Figure 5 presents evaluations across five texture conditions: original, checkerboard, Perlin noise, random Poly Haven texture [1], and solid gray, each tested with lighting (w/ L) and without lighting (w/o L). Surprisingly, altering texture while preserving shading minimally impacts performance for leading methods (SF3D, Hunyuan3D-2, Trellis). Multi-view approaches show slightly more sensitivity to texture changes but remain relatively robust overall. However, removing shading consistently decreases performance across methods, underscoring shading's significant role. Interestingly, there is an interplay between shading and texture cue: meaningful texture mitigates this drop due to removing lighting to some degree, especially on Toys4K. These results highlight that texture meaningfulness is not a necessary cue for generalizion. Meanwhile, shading is generally more influential than texture, with several top-performing methods exhibiting near texture invariance provided shading cues remain accurate.
Silhouette and Occlusion. Dilating object masks severely reduces performance despite unchanged interior pixels, indicating silhouette cues' importance. Trellis remains comparatively stable, suggesting a level of learned silhouette invariance. Occluding both silhouette and content dramatically reduces performance universally. This shows the combined importance of silhouette and interior visual cues.
Edges. We probe the role of edges cue in two ways, leaving only edges on silhouette and softening edges with localized gaussian filter. Edge-only input significantly degrade performance for most models except OpenLRM, suggesting edges alone may not provide sufficient shape information. Softening edges yields minor performance reductions, confirming edges are supportive but not primary cues.
Perspective. Switching from perspective to orthographic projection notably reduces performance, particularly for regression-based methods (OpenLRM, SF3D), indicating their reliance on perspective cues. CRM remains unaffected, since it uses orthographic images in training. Hunyuan3D-2 is more sensitive than Trellis, potentially due to its latent representation capturing perspective.
this section cite: ['b0']

Section: Local Continuity.
Local cue scrambling significantly impacts regression-based SF3D, while other methods show varied but less severe sensitivity. Hunyuan3D-2 demonstrates the greatest robustness. However, all methods degrade substantially under extensive local scrambling, emphasizing the general importance of local continuity.
this section cite: []

Section: Discussions
Correlation of Different Cues. We choose our cues primarily based on their perceptual importance and interpretability to humans, rather than strict orthogonality. As noted in Section 2, our selected cues originate from psychological studies of human visual perception and have strong foundations in prior vision research, as they represent factors that humans typically find meaningful. While some cues naturally remain disentangled (e.g., shading versus texture), others inherently overlap to some extent (e.g., style with texture).
We assess whether cues impact objects similarly by calculating per-object performance drops in CD for each cue and then computing the Spearman rank correlation between pairs of cues. This produces a correlation matrix showing how similarly each pair of cues affects the same set of objects. We show this result in Table 3. Importantly, this is a similarity-of-effect analysis. It does not test statistical independence nor guarantee disentanglement.
The result suggests that overall the correlation is low. Interestingly, texture and shading cues seem to affect a set of objects in similar ways, though they are inherently disentangled. These results also indicate that, while some appearance-related cues partially overlap, the cue effects are largely isolated at the level of object-wise impact.
this section cite: []

Section: Practical Implications of Our Analysis.
To illustrate how our insights could inspire new research directions, we explore a line-art-to-3D problem: given a line-art image (in our case, extracted from GSO images), we aim to recover the underlying 3D shape. Pure line-art lacks surface appearance, and indeed leads to markedly worse 3D generation than original images. Inspired by our analysis, we enrich line-art with geometric cues by prompting an image diffusion model (Flux ControlNet [30]) conditioned on line-art to synthesize 3D rendering-style shading and texture. We then feed these cue-augmented images into off-the-shelf image-to-3D models (InstantMesh, SF3D, Trellis). As shown in Table 4, injecting geometric cues substantially improves performance, validating that our proposed insights could meaningfully contribute to future research in image-to-3D.
this section cite: []

Section: Limitations.
While Cue3D provides a systematic and comprehensive analysis of cue importance across seven state-of-the-art methods and two widely used datasets, there remain several limitations. First, our study, though broad, is not exhaustive; evaluating a wider range of models and datasets would further strengthen our conclusions. Nevertheless, because our framework is both method and dataset-agnostic, it can be readily extended to additional settings. Second, our experiments focus on clean, object-centric datasets to minimize confounding factors, but extending the analysis to more diverse and nuanced real-world data could reveal additional insights. Third, although we primarily probe individual cues in isolation, understanding the interplay and correlation between multiple cues, beyond the initial shading-texture analysis presented here, remains an important direction for future work.
this section cite: []

Section: Conclusion
We introduce Cue3D, a model-agnostic framework for quantifying the influence of individual image cues in single-image 3D generation. We benchmark seven state-of-the-art methods across three major paradigms and two datasets in a unified approach. Then we apply targeted perturbations to individual cues like shading, texture, silhouette, occlusion, perspective, edges, and local continuity. We reveal that shape meaningfulness is crucial to the generalization of single-image 3D generation, while texture meaningfulness is not a necessary condition. Geometric cues are crucial, especially shading. Our analysis further shows that the models might be overly relying on silhouette cues, while perspective, edge, and local continuity cues affect reconstruction to varying degrees. We hope Cue3D and the insights presented here will deepen our understanding of how deep 3D networks leverage classical vision cues, and inspire future work on cue-aware architectures, robust training, and diagnostic perturbation tests for more transparent and controllable single-image 3D generation.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: Stable Zero123 Year: (2023)
Ref_id:b2 Title: Recognition-by-components: a theory of human image understanding Year: (1987)
Ref_id:b3 Title: A morphable model for the synthesis of 3D faces Year: (2023)
Ref_id:b4 Title: Evaluating multiview object consistency in humans and image models Year: (2024)
Ref_id:b5 Title: Sf3d: Stable fast 3d mesh reconstruction with uv-unwrapping and illumination disentanglement Year: (2025)
Ref_id:b6 Title: A computational approach to edge detection Year: (1986)
Ref_id:b7 Title: Learning to generate line drawings that convey geometry and semantics Year: (2022)
Ref_id:b8 Title: On seeing things Year: (1971)
Ref_id:b9 Title: Perceiving layout and knowing distances: The integration, relative potency, and contextual use of different information about depth Year: (1995)
Ref_id:b10 Title: Objaverse: A universe of annotated 3D objects Year: (2023)
Ref_id:b11 Title: Google scanned objects: A high-quality dataset of 3d scanned household items Year: ()
Ref_id:b12 Title: Probing the 3d awareness of visual foundation models Year: (2024)
Ref_id:b13 Title: Interpretable explanations of black boxes by meaningful perturbation Year: (2017)
Ref_id:b14 Title: Imagenet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness Year: (2018)
Ref_id:b15 Title: The perception of the visual world Year: (1950)
Ref_id:b16 Title: Multiple view geometry in computer vision Year: (2003)
Ref_id:b17 Title: Openlrm: Open-source large reconstruction models Year: (2023)
Ref_id:b18 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b19 Title: Large reconstruction model for single image to 3d Year: (2024)
Ref_id:b20 Title: Shape from shading: A method for obtaining the shape of a smooth opaque object from one view Year: (1970)
Ref_id:b21 Title: Zeroshape: Regressionbased zero-shot shape reconstruction Year: (2024)
Ref_id:b22 Title: Numerical shape from shading and occluding boundaries Year: (1981)
Ref_id:b23 Title: Textons, the elements of texture perception, and their interactions Year: (1981)
Ref_id:b24 Title: Shap-E: Generating conditional 3d implicit functions Year: (2023)
Ref_id:b25 Title: Auto-encoding variational Bayes Year: (2013)
Ref_id:b26 Title: Surface shape and curvature scales Year: (1992)
Ref_id:b27 Title: Nerf-vae: A geometry aware 3d scene generative model Year: (2021)
Ref_id:b28 Title: A theory of shape by space carving Year: (2000)
Ref_id:b29 Title: Ln3diff: Scalable latent neural fields diffusion for speedy 3D generation Year: (2024)
Ref_id:b30 Title: Gaussiananything: Interactive point cloud latent diffusion for 3D generation Year: (2025)
Ref_id:b31 Title: Measurement and modeling of depth cue combination: in defense of weak fusion Year: (1995)
Ref_id:b32 Title: The visual hull concept for silhouette-based image understanding Year: (1994)
Ref_id:b33 Title: Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d Year: (2023)
Ref_id:b34 Title: Symmetry strikes back: From single-image symmetry detection to 3D generation Year: (2025)
Ref_id:b35 Title: Meshformer: High-quality mesh generation with 3d-guided reconstruction model Year: ()
Ref_id:b36 Title: Zero-1-to-3: Zero-shot one image to 3D object Year: (2023)
Ref_id:b37 Title: Wonder3d: Single image to 3d using cross-domain diffusion Year: (2024)
Ref_id:b38 Title: Computing local surface orientation and shape from texture for curved surfaces Year: (1997)
Ref_id:b39 Title: Shading into texture Year: (1986)
Ref_id:b40 Title: Perception of shape from shading Year: (1988)
Ref_id:b41 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b42 Title: Mental rotation of three-dimensional objects Year: (1971)
Ref_id:b43 Title: Mvdream: Multi-view diffusion for 3d generation Year: (2023)
Ref_id:b44 Title: Beyond core knowledge: Natural geometry Year: (2010)
Ref_id:b45 Title: Using shape to categorize: Low-shot learning with an explicit shape bias Year: (2021)
Ref_id:b46 Title: Lightweight pixel difference networks for efficient visual representation learning Year: (2023)
Ref_id:b47 Title: Lgm: Large multi-view gaussian model for high-resolution 3D content creation Year: (2024)
Ref_id:b48 Title: What do single-view 3d reconstruction networks learn Year: (2019)
Ref_id:b49 Title: Shape from symmetry Year: (2005)
Ref_id:b50 Title: Triposr: Fast 3d object reconstruction from a single image Year: (2024)
Ref_id:b51 Title: Lion: Latent point diffusion models for 3d shape generation Year: (2022)
Ref_id:b52 Title: Pixel2mesh: Generating 3d mesh models from single rgb images Year: (2018)
Ref_id:b53 Title: Crm: Single image to 3D textured mesh with convolutional reconstruction model Year: (2024)
Ref_id:b54 Title:  Year: ()
Ref_id:b55 Title: Lrm-zero: Training large reconstruction models with synthesized data Year: (2008)
Ref_id:b56 Title: Holistically-nested edge detection Year: (2015)
Ref_id:b57 Title: Csgo: Content-style composition in text-to-image generation Year: (2024)
Ref_id:b58 Title: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models Year: (2024)
Ref_id:b59 Title: CutMix: Regularization strategy to train strong classifiers with localizable features Year: (2019)
Ref_id:b60 Title: A controllable large-scale generative model for creating high-quality 3D assets Year: (2024)
Ref_id:b61 Title: Shape-from-shading: a survey Year: (1999)
Ref_id:b62 Title: Hunyuan3D 2.0: Scaling diffusion models for high resolution textured 3d assets generation Year: (2005)
Ref_id:b63 Title: A dataset-dispersion perspective on reconstruction versus recognition in single-view 3D reconstruction networks Year: (2021)
