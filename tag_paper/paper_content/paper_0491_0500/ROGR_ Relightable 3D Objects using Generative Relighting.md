Title: ROGR: Relightable 3D Objects using Generative Relighting
Abstract: Input: posed images + target illumination Output: 3D relit object under novel illumination Figure 1: Given a set of posed images under unknown illumination (top), our method reconstructs a relightable neural radiance field (bottom), that can be rendered under any novel environment map without further optimization, on-the-fly relighting and novel view synthesis.

Section: Introduction
Inserting real-world objects into new environments is a long-standing problem in computer graphics [1,2], with numerous applications in the movie and gaming industries. While recent years have seen significant progress in 3D object reconstruction for view synthesis using radiance fields [3,4], these techniques represent an object illuminated by a single fixed environment and they do not enable changing the appearance of the object due to changes in the lighting. This work extends 3D object reconstruction to enable rendering the object under arbitrary illumination.
A typical approach for reconstructing relightable 3D representations from images is inverse rendering: optimizing the material and lighting parameters that together explain the captured images. This is particularly challenging for real-world captures as it is brittle and sensitive to mismatches between the real world's physical light transport and the simplified lighting and material models used during optimization. Furthermore, due to the problem's inherent ambiguities, object properties recovered by inverse rendering often appear implausible and unrealistic when viewed under novel lighting.
At the same time, recent relighting diffusion models [5,6,7] have demonstrated impressive capabilities for generating realistic images of objects under arbitrary illumination. However, they only generate a single relit image at a time, which results in inconsistent relighting results when applied to a sequence of viewpoints. While these inconsistently-relit samples can be reconstructed into a single 3D representation [7], optimizing a new 3D representation for each new target lighting is tedious and precludes interactive use cases.
We propose a strategy for distilling samples from a relighting diffusion model into a relightable 3D Neural Radiance Field (NeRF) that can be rendered from arbitrary viewpoints under arbitrary novel environment illumination. Given images of an object, we first use a multi-view diffusion network to generate view-consistent relit images under a wide array of environment illuminations. We then use these multi-view multi-illumination samples to train a novel NeRF architecture that predicts outgoing view-dependent color conditioned on a target illumination.
We evaluate the efficiacy of our method on the task of 3D relighting using both synthetic and realworld benchmarks. Our approach achieves state-of-the-art results but also demonstrates significantly improved test-time performance compared to prior work. These performance gains are due to our generalizable feed-forward relightable NeRF.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b6']

Section: Related Work
Inverse Rendering for Relighting. Recovering relightable 3D representations of objects from images is a longstanding goal in computer vision and graphics. The prevalent approach is inverse rendering: decomposing the object's appearance into its underlying geometry, illumination, and material parameters, and relighting the object by simulating the physical interaction of a new target illumination with the recovered geometry and materials [8,9,10,11].
Modern methods for reconstructing relightable 3D representations with inverse rendering use differentiable rendering techniques [12,13] to optimize mesh [14,15], distance field [16], or volumetric [17,18,19,20,21,22,23,24] representations of object geometry and material parameters. While these inverse rendering approaches can be effective, they come with significant limitations: errors in estimating an object's geometry and materials can produce unrealistic appearance when the object is relit under a new illumination and physically-accurate relighting involves computationally-expensive Monte Carlo simulation of global illumination, which can be too slow for interactive use cases.
this section cite: ['b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23']

Section: Precomputed Radiance Transfer and Light Stages.
Early approaches for interactive relighting in the field of computer graphics proposed the idea of Precomputed Radiance Transfer (PRT) [25]. As appearance behaves linearly with respect to lighting, an object or scene's appearance can be precomputed under a set of basis lightings and these can be linearly combined to produce appearance under any desired target lighting condition. While PRT-based techniques enable interactive relighting, they struggle with the memory demands of storing the precomputed transfer matrices. To enable rendering under novel illuminations from arbitrary viewpoints, PRT-based methods need to store a full transfer matrix for each 3D point in the scene. These memory requirements are prohibitive even for modestly sized scenes and environment maps, so a large body of work focuses on compressing these PRT matrices [26,27].
One-light-at-a-time (OLAT) captures in light stages [28] can be thought of as directly capturing basis vectors of a real object's radiance transfer matrices. Since each image captured by the light stage is of the object illuminated by a single element of a standard lighting basis, they can be linearly combined to reproduce the object's appearance under any desired environment illumination.
this section cite: ['b24', 'b25', 'b26', 'b27']

Section: Posed images
Multi-view relighting model Environment maps
this section cite: []

Section: Multi-illumination dataset
Figure 2: Multi-view Relight Diffusion. Our multi-view relighting diffusion model takes as input N posed images illuminated with a consistent, but unknown illumination, represented by camera raymaps and the source pixel values, and an environment map per image that has been rotated to the camera pose. The diffusion model generates images of the same object from the same poses, but lit by an input environment map. To generate our multi-illumination dataset, we repeat this relighting process M times with M environment maps.
this section cite: []

Section: Direct Relighting without Inverse Rendering.
In the deep learning era, many methods have trained networks to directly output relit images. Early techniques utilized standard convolutional neural networks [29,30] and more recent approaches are based on powerful diffusion models [5,7,6,31]. These generative relighting models have produced impressive results for image relighting, but they cannot be easily used for 3D relighting as the relit appearance is often not consistent across views. IllumiNeRF [7], MIS [31], and Neural Gaffer [6] perform 3D relighting by reconciling samples from a 2D image relighting diffusion model into a single NeRF. However, the relit 3D representation needs to be optimized separately for each novel target illumination. Recent work on 3D reconstruction from images of an object with varying illumination [32] uses a multi-view diffusion model to relight them to have consistent illumination. However, this approach does not allow relighting using a new unseen illumination condition, so it does not allow for generalization. And it does not utilize the high-frequency details in environment maps. Zeng et al. [33] and RNG [34] utilize shadow and highlight cues as conditioning signals for NeRF. In contrast, we explicitly use the encoded features of incident light from the specular reflection direction. Additionally, in contrast to all of these approaches, our proposed method trains a generalizable relightable NeRF that can be rendered under any target illumination without additional optimization. A concurrent work, RelitLRM [35], directly recovers 3D geometry and appearance but does not guarantee consistent geometry across environment maps. In contrast, our method reconstructs constant geometry under varying illumination. DiffusionRenderer [36] employs a video diffusion model for relighting but lacks 3D consistency, while our approach enforces it by explicitly modeling geometry and rendering. Moreover, our model enables fast inference, avoiding the long sampling times typical of diffusion-based generation.
this section cite: ['b28', 'b29', 'b4', 'b6', 'b5', 'b30', 'b6', 'b30', 'b5', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: Method
Given a set of N posed images D = {(I i , π i )} N i=1 of an object, where I i is the i-th image and π i its pose (including camera extrinsic and intrinsic parameters), we are interested in learning the parameters θ of a relighting function f θ , that allows rendering the object from any viewpoint π illuminated by any lighting E (e.g. an environment map) to produce a new image I = f θ (E, π). In order to learn the parameters θ of this transformation, we propose to generate a dataset of pairs of posed images with their corresponding illumination to train f θ in a supervised manner. In Sec. 3.1 we will describe how to generate such paired data using a generative relighting diffusion model, and in Sec. 3.2 we explain how to optimize f θ , which we implement using a lighting-conditioned, relightable radiance field.
this section cite: []

Section: Multi-View Diffusion Relighting
Our goal is to train a generative multi-view diffusion model g to relight our given posed images D, which are jointly captured under the same unknown lighting. The diffusion model provides samples from the distribution of possible images D E relit by the target illumination E:
p(D E |D, E).(1)
Our work crucially relies on a multi-view diffusion relighting model to provide consistent relit images, which can then be distilled into a relightable 3D model. This is in contrast with prior work on relighting using diffusion models [7,6] which independently relights single images and results in ambiguities that must be resolved at the 3D reconstruction stage.
Our network architecture is inspired by multi-view diffusion architectures such as CAT3D [37], which start with pretrained 2D image diffusion models and inflate them by adding cross-attention layers to process multiple views. We adapt such a scheme for the task of relighting and show our architecture in Fig. 2. Since we use a latent diffusion model (LDM), we first map all images and environment maps into the latent space of the original 2D diffusion model. In particular, the environment map is encoded in a similar manner to Neural Gaffer [6]: We use two separate latents corresponding to high dynamic range (HDR) and low dynamic range (LDR). This representation captures both bright details like direct light sources, as well as relatively dim sources like diffuse objects. Like Neural Gaffer, we also use standard tone mapping for the LDR environment map [38,39], and for the HDR encoding we apply logarithmic tone mapping followed by normalization to [0, 1] by subtracting the minimal value and dividing by the maximal value. Additionally, for each image I i in our dataset, we apply a 3D rotation to the environment map to align it with the corresponding camera pose π i . We combine the HDR and LDR encodings of the environment map with the encoded image and the ray map of the pose of each image by concatenating them, and then apply self-attention. Details of the network architecture for our multi-view relighting diffusion models are provided in Fig. 1 of the supplementary material.
this section cite: ['b6', 'b5', 'b36', 'b5', 'b37', 'b38']

Section: Relightable Neural Radiance Field
Our diffusion model g generates consistent relit images given a target lighting environment. Our end goal is to obtain a 3D representation of the object that can modify the illumination of the scene without additional per-illumination optimization. To do this, we first use the multi-view relighting model to create a new dataset D relit for each object by taking the N images in the original dataset D and relighting them using a collection of M environment maps E = (E 1 , ..., E M ). The new relit dataset of an object can be written as:
D relit = {g(I i , π i , E j ) : i = 1, ..., N, j = 1, ..., M },(2)
where g(I i , π i , E j ) is the diffusion model's estimate for the ith image I i (whose pose is π i ) lit by the jth environment map E j . D relit contains N × M images.
We then use the multi-illumination dataset D relit to train a NeRF. Since we wish to fit the NeRF model to our dataset with varying lighting so that it generalizes to novel lighting environments outside of our training set of illuminants E, care must be taken when designing the model's architecture. We use NeRF-Casting [40] as the base NeRF model since it efficiently captures view-dependent appearance, and we modify it to allow for conditioning on the illumination. Crucially, we encode the environment maps using two types of conditioning signals: general conditioning and specular conditioning. The general conditioning encodes the entire environment map into an embedding that is fed to the appearance MLP and is designed as a general-purpose, low-frequency signal for relighting, while the specular conditioning only encodes incident light coming from the specular direction. Its goal is to improve the model's capacity to capture high-frequency reflections (similar to prior work on reconstructing specular reflections in NeRF [41]). The conditioning signals are illustrated in Fig. 3.
In order to encode the specular appearance of a camera ray, NeRF-Casting [40] casts a small set of reflected rays, traces them through the NeRF's geometry, and volume renders a feature vector f that encodes the scene content observed by these reflected rays. In our setting, we provide the two conditioning signals by concatenating them to the feature vector f of each ray, which is mapped by NeRF-Casting's MLP to the ray's specular color component.
this section cite: ['b39', 'b40', 'b39']

Section: General Conditioning.
Our general conditioning signal encodes the full environment map, and is therefore a complete description of the lighting of the object. To do this, we train an transformerbased encoder which maps the environment map into a vector. While the idea of using a per-lighting encoding is similar to the approach of NeRF-in-the-Wild [42], a crucial difference is that our embeddings are not optimized to fit to individual images, but they are instead parameterized as a learned mapping from the environment maps. This enables us to render the scene under novel illumination at inference time without requiring any additional training.
To encode the illumination features from the environment map, we utilize a transformer encoder with self-attention. Our transformer is trained from scratch, and its architecture is based on ViT-S8 of DINO [43], followed by a single matrix multiplication W to produce a compact 128-dimensional embedding vector:
f general = W • ViT(E) (3
) Prefiltered environment maps Multi-scale Gaussian blur Transformer encoder Environment map Reflection direction Camera ray S ur fa ce
this section cite: ['b41', 'b42']

Section: Specular Conditioning.
Although the general conditioning vector theoretically contains all information necessary for relighting, we found it to be insufficient for reconstructing and rendering high-frequency specular highlights. Our specular conditioning is designed to fix that by explicitly encoding incident light from the specular reflection direction, similar to the encoding used in prior work for reconstructing reflective objects and scenes [41,40,44]. Instead of only sampling the environment map value at the reflection direction, we use a series of blur kernels centered around the reflection direction to simulate the effects of materials with different roughnesses. The i-th component of this conditioning vector can be written as:
f specular i = S 2 E(ω ′ )G(ω ′ ; ω r , σ i )dω ′ (4
)
where G(ω ′ ; ω r , σ i ) is a Gaussian blur kernel around ω r with width σ i , and ω r is the view direction reflected about the surface normal.
For efficiency, we preprocess the environment map by blurring it at all directions using Eq. 4, and then query it at the reflection direction ω r during the NeRF optimization stage.
this section cite: ['b40', 'b39', 'b43']

Section: Network Predictions.
Following NeRF-Casting [40], our relightable NeRF takes as input the 3D coordinates, ray direction, general conditioning, and specular conditioning. The geometry MLP predicts density, roughness, normals, and geometry features, while the color MLP outputs the RGB values.
this section cite: ['b39']

Section: Experimental Setup

this section cite: []

Section: Datasets
Training datasets. To train multi-view relighting diffusion, we use a dataset of 400k synthetic 3D objects, including 100k from Objaverse [46]. Each object is rendered in 64 views × 16 HDR illuminations, with environments sampled from Polyhaven [47] (590 maps) and augmented via random up-axis rotations.
Evaluation datasets. For relighting evaluations, we used two datasets: TensoIR [45] and Stanford-ORB [48]. TensoIR is a synthetic benchmark, which contains renderings of four objects under six lighting conditions. We use the train split of 100 views with "sunset" lighting condition as inputs for relightable NeRF. We then evaluate 200 novel views under other five environment maps, including "bridge", "fireplace", "forest", "city", and "night". In total, we have 4,000 renderings for evaluation metric calculation. Stanford-ORB is real-world benchmarks by data capture in the wild. It has 14 objects composed of various materials. Each object is captured under three distinct lighting conditions, producing a total of 42 (object, lighting) combinations. Following its evaluation protocol, we use images of an object under a single lighting condition and evaluate novel views under the two target lighting settings.
Ground
this section cite: ['b45', 'b46', 'b44', 'b47']

Section: Implementation details
Multi-View Diffusion model. We fine-tune our model starting from a pre-trained latent image generation model, as described in [49]. The multi-view denoiser is derived from the CAT3D network architecture [37], with modifications to the input channel dimensions to align with our relighting task. The inputs to our model are images with resolution 512 × 512 which are encoded to 64 × 64 × 8 latents. We chose the number of views to be 64. The model was trained on 128 TPU v5 chips using a learning rate of 10 -4 , with a total batch size of 128 for 360k iterations. After training, we generate the multi-illumination dataset by running our relighting diffusion inference on 111 environment maps.
Relightable NeRF model. We train our NeRF on 8 H100s for 500k steps. We use a 512 × 512 resolution environment map as the target illumination. We sample each reflection rays 3 times; once using a point sample on the full resolution environment map, and then using Gaussian kernels of sizes 20 × 20 and 40 × 40 pixels in radius with σ i values of 10 and 20 respectively (see Fig. 3).
In order to maximize the number of Illumination conditions we use for training, we make several reductions to the size of model relative to the NeRF-Casting architecture. We lower the batch size to 1,000 and increase the number of training steps to 500,000. We also decrease the size of the "bottleneck" vector b in both the geometry and appearance MLPs relative to NeRF-Casting. Please refer to supplementary material for more details on the base architecture.
this section cite: ['b48', 'b36']

Section: Baselines
We compare our method against existing inverse rendering methods including NeRFactor [50], InvRender [51], PhySG [52], NeRD [19], NVDiffRecMC [14], NVDiffRec [15], Neural-PBIR [53], NeRO [44],TensoIR [45], and recent single-view relighting diffusion methods IllumiNeRF [7] and Neural Gaffer [6]. We also include the most recent Gaussian splatting-based inverse rendering method R3DG [54].
this section cite: ['b49', 'b50', 'b51', 'b18', 'b13', 'b14', 'b52', 'b43', 'b44', 'b6', 'b5', 'b53']

Section: Evaluation Metrics
We evaluate the relighting rendering quality by PSNR, SSIM [55], and LPIPS-VGG [56].
this section cite: ['b54', 'b55']

Section: Results
Our method is the top-performing technique on existing relighting benchmarks for synthetic and realworld objects. Furthermore, it can render images from novel viewpoints under novel illuminations at
this section cite: []

Section: TensoIR benchmark
Method PSNR ↑ SSIM ↑ LPIPS ↓ NeRFactor [50] 23.38 0.908 0.131 InvRender [51] 23.97 0.901 0.101 TensoIR [45] 28.58 0.944 0.081 Neural-PBIR [53] 27.09 0.925 0.085 NeRO [44] 27.00 0.935 0.074 R3DG [54] 29.05 0.937 0.080 NeuralGaffer [6] 27.30 0.918 0.122 IllumiNeRF [7] 29.71 0.947 0.072 Ours 30.74 0.950 0. 069 Table 1: TensoIR benchmark [45]. We evaluate all four objects in the benchmark, each under five novel lightings. Each object is rendered from 200 views for novel view evaluation under each lighting. Thus, we have 4,000 renderings in total for quantitative evaluation. Best and 2nd-best are highlighted.
In Fig. 4, our method achieves stateof-the-art performance on the TensoIR benchmark, improving upon the ability of prior works to capture specularities in the reflective "hot dog" and "ficus" scenes while accurately capturing diffuse appearance from global illumination in the "lego" and "armadillo" scenes. The superiority of our method can also be verified by quantitative results in Tab. 1.
this section cite: ['b50', 'b44', 'b52', 'b43', 'b53', 'b5', 'b6', 'b44']

Section: Stanford-ORB benchmark
In Fig. 5, our method also achieves stateof-the-art performance on the Stanford-ORB dataset, and is most effective at reflective objects like the "baking" and "ball" scenes where consistent reflections clearly show up in the reconstruction. We provide quantitative comparisons in Tab. 2, where our method outperforms others in PSNR-H and SSIM, and acheive second-best results in PSNR-L and LPIPS. As discussed in IllumiNeRF [7], our results are also qualitatively superior to those of Neural-PBIR [53], but they are worse quantitatively due to the mostly-diffuse renderings of Neural-PBIR.
this section cite: ['b6', 'b52']

Section: Real-world Objects
Finally, we demonstrate the ability of our method to relight "in-the-wild" captures of real objects with spatially-varying material properties under natural lighting in Fig. 1. Since we have no ground truth images for real-world relighting evaluation, we only show qualitative results. Our method captures convincing specularities on the wood and metal parts of the model sewing machine, as well as accurate shadows cast by the arm of the sewing machine.
InvRender NVDiffRecMC Neural-PBIR R3DGS Neural Gaffer IllumiNeRF Ours GT Figure 5: Qualitative comparisons on Stanford-ORB [48]. Renderings from all methods are rescaled to the image resolution of the ground truth. Compared to previous work, our method produces high-fidelity renderings with more faithful specular reflections highlighted in the red boxes. Method PSNR ↑ SSIM ↑ LPIPS ↓ (a) No blurring in specular conditioning 31.35 0.90 0.080 (b) No specular conditioning 30.00 0.88 0.079 (c) No general conditioning 21.59 0.70 0.130 (d) Per-image appearance embeddings 19.12 0.62 0.160 (e) 128 × 128 environment map 29.26 0.89 0.082 (f) 64 × 64 environment map 27.69 0.86 0.110 Our full model, 64-view, 111 envmaps 31.88 0.91 0.075
Table 3: Ablation studies on the "hotdog" scene from TensoIR [45]. See the text and Fig. 6 for corresponding qualitative results and additional explanations. Best and 2nd-best are highlighted.
this section cite: ['b44']

Section: Ablation Studies
We next perform an ablation study of the different components of our method, done on the "hotdog" scene of the TensoIR benchmark, chosen since it has the most interesting materials in that dataset. Each ablation is reported in a row of Tab. 3 and its corresponding column in Fig. 6.
Multi-scale specular conditioning. Our multi-scale specular conditioning features, which are computed by blurring the environment map using multiple kernel sizes as in Eq. 4, are provided to the model. Each scale is designed to approximate the incident light averaged over a set of directions corresponding to a particular surface roughness. When we skip this blurring operation and use the Method PSNR ↑ SSIM ↑ LPIPS ↓ Ours full model, 4 view, 111 envmaps 31.04 0.86 0.082 Ours full model, 16 view, 111 envmaps 31.86 0.90 0.077 Our full model, 64-view, 10 envmaps 29.12 0.82 0.086 Our full model, 64-view, 50 envmaps 31.78 0.88 0.079 Our full model, 64-view, 111 envmaps 31.88 0.91 0.075 Our full model, 64-view, 150 envmaps 31.90 0.92 0.075
Table 4: Ablation studies of number of views and illulimantions on the "hotdog" scene from TensoIR [45]. Best and 2nd-best are highlighted.
environment map values directly, our model can still represent highly specular regions (see Fig. 6(a)), but struggles more with rough or diffuse surfaces, leading to a drop in reconstruction metrics.
this section cite: ['b44']

Section: Specular conditioning.
Next we remove the specular conditioning signal altogether, which leads to another small drop in reconstruction metrics as well as a qualitative drop in the accuracy of rendered specular highlights.
this section cite: []

Section: General conditioning.
Removing the general conditioning signal from our network results in significant artifacts and poor reconstruction metrics, since the general conditioning is the main mechanism for providing the target illumination to the relighting model.
this section cite: []

Section: Per-image appearance embeddings.
An alternative to our conditioning signals is a per-image appearance embedding vector, similar to the GLO codes in NeRF-in-the-Wild [42]. While this allows the model to be trained on multiple illuminations, it does not generalize to new unseen lights. Additionally, unlike our model, we found that the embedding vectors do not scale well to a large number of illumination conditions, resulting in significantly worse qualitative and quantitative results.
Environment map resolution. Our full model uses conditioning signals based on an environment map of resolution 512 × 512. Computing the conditioning signals from environment maps of size 128 × 128 results in loss of detail in the specular highlights. Further lowering the resolution to 64 × 64 results in rendering artifacts even for diffuse surfaces.
this section cite: ['b41']

Section: Number of views. Our full model learns the joint distribution of 64-view relighting.
To analyze the impact of the number of views in the diffusion model, we compare relighting novel view synthesis results using 4, 16, and 64-view diffusion on the hotdog scene of TensoIR dataset. As shown in Tab. 4, the 64-view diffusion model consistently outperforms the others across all metrics. This indicates that jointly denoising more views leads to more consistent and higher-quality relit images. Number of environment maps. We also study the effect of environment map count on relightable NeRF training by using 10, 50, 111, and 150 illuminations. As shown in Tab. 4, more illuminations generally improve relighting performance, which saturates around 111 maps.
this section cite: []

Section: Normal map visualization.
As shown in Fig. 7, our relightable neural radiance fields can render high-quality normal maps, which are comparable to prior inverse rendering techniques like TensoIR, as well as novel view synthesis approaches like NeRF-Casting which explicitly encourage geometry to be more surface-like.
this section cite: []

Section: Limitations
While our method improves 3D object relighting by achieving more accurate specularities and fast inference, it can still be further improved in several aspects. First, although our relighting diffusion model is trained on objects with materials varying in diffuse albedo and roughness, which are the main sources of variation in real-life materials, we did not train on objects exhibiting phenomena such as subsurface scattering, refraction, or volumetric effects. Expanding our training data to include these complex materials would improve robustness and generalizability to real data. Second, we use environment maps to define lighting conditions, which assumes that the light sources are infinitely far away from the object. Exploring more general lighting models containing near-field illumination components could enhance realism in diverse illumination scenarios. Finally, our approach focuses on object-centric scenes, and extending it to large-scale scene relighting would be an exciting direction for future research.
this section cite: []

Section: Conclusion
This paper introduces a novel method for 3D object relighting, enabling fast, feed-forward relighting during inference. By modeling a virtual light stage with a generative multi-diffusion model, we create a diverse dataset of multi-illumination images. This dataset serves as a prior to train a lightconditioned Neural Radiance Field (NeRF) model, which subsequently learns to render the object under arbitrary target illumination conditions. Existing methods for 3D relighting often rely on inverse rendering techniques, constrained by shading models limited to specific material types, or they directly generate relit NeRFs, embedding the lighting within the model itself. This requires retraining the NeRF for each new lighting condition. In contrast, our proposed model exhibits generalization across diverse lighting conditions at inference, facilitating efficient feed-forward relighting. Experimental results demonstrate the effectiveness of our method in relighting complex real-world objects with high fidelity.
In Fig. 8, we provide the detailed inputs of our relighting diffusion model. While our diffusion model receives and produces image latents, we depict them as images for better clarity. We encode RGB images under source lighting, HDR, and LDR target environment maps into latents of size 64 × 64 × 8. The raymaps consist of ray origins and ray directions corresponding to image pixels. We downscale them from the original image resolution 512 × 512 × 6 to the latent size 64 × 64 × 6. We concatenate the source image latent, HDR, LDR environment latents, raymaps, and noisy latents along the channel dimension to form a new feature of size 64 × 64 × 38, and then feed it into a multi-view diffusion network to produce clean target latents of size 64 × 64 × 8. Our denoiser network is based on CAT3D [37]. Please refer to Fig. 7 of CAT3D for the network architecture details.
During training, we use the DDPM schedule, with beta values that linearly increase from 8.5 × 10 -4 to 1.2 × 10 -2 over 1024 steps. We use noise prediction as our diffusion objective. The model was trained on 128 TPU v5 chips using a learning rate of 10 -4 , with a total batch size of 128 for 360k iterations and 10K warm-up steps. We adopted a progressive training scheme, where we first trained a 4-view diffusion model for 300k steps, and then fine-tuned it for 16-view diffusion for 15k steps, and finally fine-tuned it for 64-view diffusion for 45 steps. We keep the learning rate as 10 -4 when we fine-tune the model to relight the large number of views. We enable classifier-free guidance (CFG) [60] by randomly dropping the HDR and LDR environment maps with a probability of 0.1. During inference, we use the DDIM schedule [61] with 50 sampling steps and the classifier-free weight is set to 3.0.
this section cite: ['b36', 'b59', 'b60']

Section: References
Ref_id:b0 Title: Rendering synthetic objects into real scenes: Bridging traditional and image-based graphics with global illumination and high dynamic range photography Year: (2008)
Ref_id:b1 Title: Acquiring the reflectance field of a human face Year: (2000)
Ref_id:b2 Title: NeRF: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b3 Title: Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b4 Title: Dilightnet: Fine-grained lighting control for diffusion-based image generation Year: (2024)
Ref_id:b5 Title: Neural gaffer: Relighting any object via diffusion Year: (2024)
Ref_id:b6 Title: Illuminerf: 3d relighting without inverse rendering Year: (2024)
Ref_id:b7 Title: A signal processing framework for inverse rendering Year: (2001)
Ref_id:b8 Title: Inverse global illumination: Recovering reflectance models of real scenes from photographs Year: (1999)
Ref_id:b9 Title: Object shape and reflectance modeling from observation Year: (1997)
Ref_id:b10 Title: Imagebased reconstruction of spatial appearance and geometric detail Year: (2003)
Ref_id:b11 Title: Differentiable monte carlo ray tracing through edge sampling Year: (2018)
Ref_id:b12 Title: Mitsuba 2: A retargetable forward and inverse renderer Year: (2019-12)
Ref_id:b13 Title: Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising Year: (2022)
Ref_id:b14 Title: Extracting triangular 3d models, materials, and lighting from images Year: (2022)
Ref_id:b15 Title: Iron: Inverse rendering by optimizing neural sdfs and materials from photometric images Year: (2022)
Ref_id:b16 Title: Neural reflectance fields for appearance acquisition Year: (2020)
Ref_id:b17 Title: NeRV: Neural reflectance and visibility fields for relighting and view synthesis Year: (2021)
Ref_id:b18 Title: Nerd: Neural reflectance decomposition from image collections Year: (2021)
Ref_id:b19 Title: NeRFactor: Neural factorization of shape and reflectance under an unknown illumination Year: (2021)
Ref_id:b20 Title: Neural microfacet fields for inverse rendering Year: (2023)
Ref_id:b21 Title: Tensoir: Tensorial inverse rendering Year: (2023)
Ref_id:b22 Title: Flash cache: Reducing bias in radiance cache based inverse rendering Year: (2024)
Ref_id:b23 Title: Arm: Appearance reconstruction model for relightable 3d generation Year: (2024)
Ref_id:b24 Title: Precomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments Year: (2002)
Ref_id:b25 Title: All-frequency shadows using non-linear wavelet lighting approximation Year: (2003)
Ref_id:b26 Title: Clustered principal components for precomputed radiance transfer Year: (2003)
Ref_id:b27 Title: Acquiring the reflectance field of a human face Year: (2000)
Ref_id:b28 Title: Deep image-based relighting from optimal sparse samples Year: (2018)
Ref_id:b29 Title: Single image portrait relighting Year: (2019)
Ref_id:b30 Title: A diffusion approach to radiance field relighting using multi-illumination synthesis Year: (2024)
Ref_id:b31 Title: Generative multiview relighting for 3D reconstruction under extreme illumination variation Year: (2024)
Ref_id:b32 Title: Relighting neural radiance fields with shadow and highlight hints Year: (2023)
Ref_id:b33 Title: Rng: Relightable neural gaussians Year: (2024)
Ref_id:b34 Title: Generative relightable radiance for large reconstruction models Year: (2024)
Ref_id:b35 Title: Diffusion renderer: Neural inverse and forward rendering with video diffusion models Year: (2025)
Ref_id:b36 Title: Cat3d: Create anything in 3d with multi-view diffusion models Year: (2024)
Ref_id:b37 Title: Burst photography for high dynamic range and low-light imaging on mobile cameras Year: (2016)
Ref_id:b38 Title: High dynamic range imaging Year: (2004)
Ref_id:b39 Title: Nerf-casting: Improved view-dependent appearance with consistent reflections Year: (2024)
Ref_id:b40 Title: Ref-NeRF: Structured view-dependent appearance for neural radiance fields Year: (2022)
Ref_id:b41 Title: NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections Year: (2021)
Ref_id:b42 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b43 Title: Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images Year: (2023)
Ref_id:b44 Title: Tensoir: Tensorial inverse rendering Year: (2023)
Ref_id:b45 Title: Objaverse: A universe of annotated 3d objects Year: (2023)
Ref_id:b46 Title: Polyhaven: a curated public asset library for visual effects artists and game designers Year: (2021)
Ref_id:b47 Title: Stanford-orb: a real-world 3d object inverse rendering benchmark Year: (2024)
Ref_id:b48 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b49 Title: Nerfactor: Neural factorization of shape and reflectance under an unknown illumination Year: (2021)
Ref_id:b50 Title: Modeling indirect illumination for inverse rendering Year: (2022)
Ref_id:b51 Title: PhySG: Inverse rendering with spherical gaussians for physics-based material editing and relighting Year: (2021)
Ref_id:b52 Title: Neural-pbir: reconstruction of shape, material, and illumination Year: (2023)
Ref_id:b53 Title: Relightable 3d gaussians: Realistic point cloud relighting with brdf decomposition and ray tracing Year: (2025)
Ref_id:b54 Title: Image quality assessment: from error visibility to structural similarity Year: (2004)
Ref_id:b55 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b56 Title: Shape, light, and material decomposition from images using monte carlo rendering and denoising Year: (2022)
Ref_id:b57 Title: Jax: composable transformations of python+ numpy programs Year: (2018)
Ref_id:b58 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b59 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b60 Title: Denoising diffusion implicit models Year: (2020)
Ref_id:b61 Title: Lgm: Large multi-view gaussian model for high-resolution 3d content creation Year: (2024)
Ref_id:b62 Title: Structure-from-motion revisited Year: (2016)
Ref_id:b63 Title: Segment anything Year: (2023)
