Title: ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding
Abstract: Recently, the powerful text-to-image capabilities of GPT-4o have led to growing appreciation for native multimodal large language models. However, its multimodal capabilities remain confined to images and text. Yet beyond images, the ability to understand and generate 3D content is equally crucial. To address this gap, we propose ShapeLLM-Omni-a native 3D large language model capable of understanding and generating 3D assets and text in any sequence. First, we train a 3D vector-quantized variational autoencoder (VQVAE), which maps 3D objects into a discrete latent space to achieve efficient and accurate shape representation and reconstruction. Building upon the 3D-aware discrete tokens, we innovatively construct a large-scale continuous training dataset named 3D-Alpaca, encompassing generation, comprehension, and editing, thus providing rich resources for future research and training. Finally, we perform instruction-based fine-tuning of the Qwen-2.5-vl-7B-Instruct model on the 3D-Alpaca dataset, equipping it with native 3D understanding and generation capabilities. Our work represents an effective step toward extending multimodal large language models with fundamental 3D intelligence, paving the way for future advances in 3D-native AI.

Section: Introduction
Large language models have made significant achievements, including text-only language models (LLMs) Achiam et al. [2023], Liu et al. [2024a], Bai et al. [2023], Touvron et al. [2023], Multimodal Large Language Models (MLLMs) that can understand images Hurst et al. [2024a], GLM et al. [2024], Team [2024], video Guo et al. [2025], Cheng et al. [2024], Maaz et al. [2023], Li et al. [2024b] and 3D Wang et al. [2024b], Siddiqui et al. [2024a], Chen et al. [2023aChen et al. [ , 2025b] ] content. These models employ similar transformer architectures, using dedicated encoders to model each modality independently, thereby integrating images, video, and 3D modalities into existing LLMs.
Recently, ChatGPT-4o Hurst et al. [2024a] has demonstrated remarkable performance. By natively incorporating image generation and understanding into the large language model (LLM) architecture, it enables more fine-grained and precise control through human instructions. However, its multimodal capabilities remain confined to images and text, limiting its potential in more complex spatial domains.
In this work, we propose a unified approach to integrate 3D generation and understanding into a pre-trained multimodal large language model (MLLM). Enhancing LLMs with native 3D capabilities is crucial for downstream applications such as 3D content creation, robotics, digital twins, and immersive virtual environments.
Our method adopts a fully next-token prediction paradigm, which ensures natural compatibility with joint training and large-scale scalability. We leverage a VQVAE to encode 3D meshes into compact discrete tokens, enabling a unified representation. These tokens are utilized for both understanding and generating 3D meshes, following a format analogous to language modeling.
To enable LLMs with 3D ability, we construct a comprehensive training dataset using 3D shapes from a mixture of 3D datasets Deitke et al. [2023a,b], Collins et al. [2022], Chang et al. [2015]. We construct interleaved 710k text/image-3D pairs to enable the model for basic 3D understanding ability and text/image to 3D generation ability.
Furthermore, to enable interactive 3D mesh editing, we introduce a novel dataset of 62k paired 3D meshes and corresponding text-based editing instructions. This facilitates fine-grained manipulation of 3D assets through natural language, making real-time editing more intuitive and controllable.
After that, we train an LLM on the corpus. We resume from Qwen-2.5-VL-Instruct- 7B Bai et al. [2025] to utilize the effective of its large-scale pre-training on text and images. Our model demonstrates a wide range of capabilities, including: (1) generating 3D content from language instructions;
(2) generating 3D objects from image inputs; (3) interactively editing 3D assets using natural language;
(4) understanding and interpreting 3D meshes for semantic and geometric reasoning.
In all, our contributions are:
• We propose a novel framework for unified 3D object generation and understanding based on a fully autoregressive next-token prediction paradigm.
• We present the 3D-Alpaca dataset for training large language models (LLMs) with 3D capabilities. Comprising 3.46 billion tokens, it covers three core tasks: 3D generation, 3D understanding, and 3D editing.
• Our experimental results provide strong empirical evidence supporting the effectiveness of the proposed method.
2 Related Work
this section cite: ['b0', 'b2', 'b82', 'b26', 'b26', 'b27', 'b20', 'b60', 'b10', 'b22']

Section: 3D Mesh Generation
The remarkable achievement of 2D diffusion models Ho et al. [2020], Rombach et al. [2022] has facilitated the exploration of 3D generative models. Early 3D generation methods Poole et al. [2022], Wang et al. [2023e], Chen et al. [2023b], Lin et al. [2023], Raj et al. [2023], Li et al. [2023b], Sun et al. [2023], Chen et al. [2024h], Wang et al. [2022], Tang et al. [2023], Yi et al. [2024] often rely on SDS-based optimization to distill 3D content due to the limited 3D data, but encounter challenges such as long optimization time and Janus problem. Subsequent works such as Wang and Shi [2023], Shi et al. [2023b], Wang et al. [2023c], Liu et al. [2025a], Ye et al. [2024b], Qiu et al. [2024], Chen et al. [2024a] enhance semantic consistency across different views during multi-view image synthesis. To minimize generation time, more recent approaches Long et al. [2024], Zhao et al. [2024], Liu et al. [2023d,c], Shi et al. [2023a], Weng et al. [2023], Liu et al. [2023b], Wu et al. [2024a], Chen et al. [2024i], Voleti et al. [2024], Ye et al. [2024a], Liu et al. [2024b] adopt a two-stage pipeline that integrates multi-view image prediction with 3D reconstruction to produce 3D models. LRM Hong et al. [2023a] and other works Tang et al. [2024a], Wei et al. [2024], Ziwen et al. [2024], Li et al. [2023a], Xu et al. [2023], Wang et al. [2023a], Siddiqui et al. [2024b], Zhang et al. [2024a,b],
this section cite: ['b32', 'b69', 'b63', 'b57', 'b68', 'b76', 'b85', 'b78', 'b113', 'b86', 'b65', 'b66', 'b58', 'b118', 'b96', 'b84', 'b95', 'b122', 'b106']

Section: Autoregressive 3D Generation
Inspired by the success of auto-regressive models in language and image synthesis, some pioneering works explores the integration of natural language instructions with mesh generation and understanding, enabling interactive 3D content creation through a unified framework. However, it treats the 3D OBJ mesh file as text for language model to process, which overlooks the inherent topological structures of 3D data.
this section cite: []

Section: Unified Models for Multimodal Understanding and Generation
Extending large language models (LLMs) to process, generate, and comprehend multiple modalities-such as vision and language-within a unified framework has become a major research frontier. Previous studies Bai et al. [2023], Chen et al. [2024g], Alayrac et al. [2022] have advanced this direction by equipping LLMs with visual understanding capabilities for multimodal tasks. Concurrently, other works Team 3 Method
this section cite: ['b2', 'b1']

Section: Architecture
As shown in Figure 1, we represent both text and 3D data as sequences of discrete tokens, enabling fully autoregressive multimodal generation. This design allows for flexible input and output across modalities in any order. While we adopt token-based representations for both text and 3D modalities, we use continuous features for images. This is because images are only involved in understanding tasks, whereas 3D data supports both understanding and generation. Such a unified modeling approach-based on early fusion-facilitates better modality integration within the language model. Compared to prior work in the 3D domain Table 1, our model is the first unified auto-regressive framework that supports text-to-3D, image-to-3D, 3D understanding, and 3D editing in a single system. It also marks the first attempt at a ChatGPT-4o-style model tailored for 3D tasks.
this section cite: []

Section: 3D VQVAE
In this section, we introduce our 3D representation-voxels-explain why we chose voxels, and how we compress voxels into discrete tokens using a 3D VQVAE. Finally, we describe how to reconstruct high-quality 3D meshes from voxels. our 3D representation. We now explain the rationale behind this choice. First, we do not adopt the Face-Vertex representation because it quantizes mesh geometry into discrete spatial tokens, resulting in excessively long token sequences that hinder the training efficiency of unified models. Second, we do not use VecSets-based representations. On one hand, VecSets encode highly informative and continuous geometric features, making it challenging to train a complete 3D VQ-VAE for their encoding. On the other hand, VecSets are inherently implicit, whereas voxels provide explicit and structured spatial representations that are more suitable for 3D editing tasks requiring direct geometric manipulation. In contrast, voxels strike a favorable balance between compactness and expressiveness: they compress complex 3D information into a much smaller latent space, facilitating efficient training, while effectively preserving an asset's essential shape and skeletal structure, thereby providing sufficient geometric cues for language models. Moreover, open-source reconstruction models can be readily leveraged to convert coarse-resolution voxels into high-quality, detail-rich meshes.
this section cite: []

Section: Model Architecture
We adopt a 64 3 voxel grid resolution, as voxels at this resolution strike the optimal balance for modeling 3D skeletons, preserving essential structural details while avoiding excessive redundancy Xiang et al. [2024]. Although voxel representations are compact, even modeling a single 3D object with a 64 3 voxel grid still requires 64 3 tokens-far beyond what a large language model can handle. Therefore, we further compress voxels using a 3D VQVAE Xiang et al. [2024]: first, we encode the 64 3 grid into a 16 3 latent grid; then we serialize it into 4096 tokens. However, 4096 tokens remain too long. Inspired by Team [2024], which represents images as 1024 tokens, we concatenate every four neighboring tokens along the channel dimension-transforming the original 4096 tokens with 8 channels into 1024 tokens with 32 channels. Finally, we employ an 8192-entry codebook to compress the voxels into 1024 discrete tokens. In all, we represent a single 3D object using 1024 discrete tokens, for both generation and understanding.
this section cite: ['b102', 'b26']

Section: Shape Reconstruction Although we employ voxel-based representations for 3D shape generation, practical deployment often necessitates converting voxels into meshes for downstream applications.
To address this, we adopt the approach proposed by Xiang et al. Xiang et al. [2024], which utilizes a Rectified Flow model to refine and complete voxel information, enabling high-quality mesh reconstruction. By first generating 3D shapes in the voxel domain and then converting them into meshes using this method, our framework achieves a balance between precision and efficiency. This hybrid representation allows large language models to exert fine-grained control over 3D content generation while avoiding the computational burden associated with high-resolution geometry.
this section cite: ['b102']

Section: 3D-Alpaca Dataset Construction
Although a wealth of datasets has been developed for the supervised fine-tuning of multimodal large-language models, dialogue data within the 3D LLM Hong et al. [2023b], Chen et al. [2025b], Xu et al. [2024b] domain remains relatively scarce. To bridge this gap, we introduce 3D-alpaca, a comprehensive dataset encompassing tasks in 3D content generation, comprehension, and editing.
this section cite: []

Section: 3D Generation and Understanding Dataset
We select a high-quality subset of approximately 712k 3D assets from Trellis Xiang et al. [2024] and internal collection. For the image collection, each asset is rendered into a 2D image, and a random offset is applied to the frontal view to create the input. Moreover, these rendered images also underpin the construction of the editing dataset in the Sec. 3.4. To generate the text collection and enable early fusion across all three modalities, we render four orthogonal views-front, back, left, and right-of each asset. These multi-view images are then input into the base model Qwen-2.5-VL-Instruct Bai et al. [2025] to generate descriptive captions. The resulting captions are utilized both as prompts for text-to-3D generation and as ground-truth targets for 3D-to-text captioning tasks.
this section cite: ['b102', 'b3']

Section: 3D Edited Dataset
We aim to build a 3D asset-editing dataset composed of paired 3D assets, where each pair is linked to a specific editing instruction. Despite recent advances in 3D content creation, the field still lacks a model capable of performing consistent edits on 3D assets. In light of the promising performance of current image-editing models, we therefore adopt an image-mediated pipeline: first rendering each 3D asset into images and applying an image-editing model, then reconstructing the edited images back into 3D assets via an image-to-3D generation method. Based on the multimodal alignment demonstrated and with the aim of equipping the model with ChatGPT-4o-level editing capabilities, we follow a six-step pipeline.
(1) Category: We reference the data distribution of Objaverse-XL Deitke et al. [2023a] and manually selected the 100 most representative and frequent object categories, such as cars, tables, cabinets, human figures, etc.
(2) Asset Classification: Using ChatGPT-4o, we classify the 3D assets in our dataset into fine-grained subcategories, with the frontal view renderings of each asset as input. From the 3D asset dataset, we filtered 311k assets belonging to the predefined 100 major categories.
(3) Editing-Prompt Definition: We provide the category names to ChatGPT-4o and instruct it to generate 20 feasible editing-prompts for each category. The instruction given to ChatGPT-4o is: "For each given category name, suggest potential image editing operations that could be applied to objects of that category." Next, we manually review each generated editing prompt and retain only those that meet both our technical feasibility and visual engagement criteria, resulting in 371 unique editing prompts (e.g: "Replace the chair's backrest with a mesh frame").
(4) Asset Sampling & Annotation: Due to time and resource constraints, we build a compact, highquality dataset of editing prompts rather than applying every possible editing prompt to each asset. Specifically, we allocate 200 assets to each editing prompt.
(5) Editing-Image Pair Collection: For each sampled asset, we provide ChatGPT-4o with its frontal render plus the chosen editing-prompt, and ChatGPT-4o produces the corresponding edited image, yielding image-level editing pairs. After filtering out erroneous cases, we end up with 70k valid editing samples.
(6) 3D reconstruction: Finally, we employ Trellis Xiang et al. [2024] to convert the curated images into 3D assets, resulting in 3D pairs before/after editing.
this section cite: ['b102']

Section: Dialogue Data Construction
We define 25 dialogue templates per task (e.g., "Generate a 3D asset of prompt/images") and encode all 3D assets into discrete token sequences with our pre-trained 3D VQVAE (Sec. 3.3). For each 3D-edit instance, we randomly select 6 templates from a pool of 25; for all other instances, we randomly assign one template each. By merging the tokens with these templates, we create a training corpus of 2.5 million 3D dialogues.
this section cite: []

Section: General Conversation
To ensure the model's general conversational capability, we adopt Ultra-Chat Ding et al. [2023] as our text-only dataset, with its data distribution shown in the Table 2. For additional details, please refer to the Appendix.
Putting these together After data processing and construction, we finally arrive at the 3D-Alpaca dataset. As shown in the Table 2, the dataset includes four types of tasks: image-to-3D, text-to-3D, 3D-to-caption, and 3D-editing. Together, these four subsets form a total of 2.56 million samples, comprising 3.46 billion tokens. To ensure the large language model retains its original reasoning and dialogue capabilities, we additionally include the UltraChat Ding et al. [2023] dataset, a high-quality, large-scale multi-turn dialogue corpus.  4.1 Implementation Details For training our 3D VQVAE, we adopt a 3D U-Net VAE architecture introduced in Trellis Xiang et al. [2024]. Our training follows a two-stage strategy: In Stage 1, we freeze the VAE's pre-trained parameters and train only the codebook. In Stage 2, we unfreeze the VAE and jointly fine-tune it with the codebook. Concretely, each stage runs for 1000 steps on 48 NVIDIA H100 GPUs with a batch size of 25, while the learning rate decays from 5 × 10 -3 to 5 × 10 -5 . For the training of ShapeLLM-Omni, we use Qwen-2.5-VL-Instruct-7B Bai et al. [2025], a multimodal large language model (MLLM)
with image-understanding capability, as our backbone. Specifically, we extend its base architecture by adding the 8192 3D VQVAE codebook. To preserve its original image-understanding skills, we freeze the parameters of Qwen2.5-vl's visual encoder. While training, the learning rate decays from 5 × 10 -5 to 5 × 10 -6 , with a per-GPU batch size of 2 and gradient accumulation over 2 steps. The model is trained for 15 epochs on 48 NVIDIA H100 GPUs.
this section cite: ['b25', 'b25']

Section: Quantitative comparisons
Language and Conversational Abilities Table 3   Fine-tuned on 3D-Alpaca for both 3D mesh generation and comprehension, our ShapeLLM-Omni maintains language understanding and reasoning performance on par with baseline models. The result demonstrates that ShapeLLM-Omni effectively extends the MLLM's capabilities to 3D content generation while preserving its native language capabilities. Table 4: Quantitative Evaluation of 3D VQVAE reconstruction performance. We report IoU, Recall, Precision, F1, and Chamfer Distance between original and reconstructed voxel grids. The results demonstrate that our 3D VQVAE effectively preserves geometric structure with high fidelity. Additionally, we report the CLIP score Radford et al. [2021] to measure the semantic alignment between the generated outputs and their input prompts. As shown in the Table 5, our generation results outperform all baseline methods except for Trellis.
this section cite: ['b67']

Section: IOU Average Recall Average F1 Average Precision
3D Understanding Following the evaluation settings provided by PointLLM Xu et al. [2024b], we test the same metrics on the benchmark dataset used by PointLLM. We adopt the same curated test set to assess the 3D-to-caption task. The dialogue prompt is structured as: "<mesh>. Caption this 3D model in detail.". As shown in Table 6, our ShapeLLM-Omni demonstrates strong 3D understanding capabilities, with performance second only to PointLLM, which is specifically tailored for single-task 3D understanding.  Our method achieves better text alignment, with 3D shapes accurately reflecting input descriptions.
this section cite: []

Section: Qualitative comparisons 3D Generation
To evaluate the effectiveness of our image-conditioned generation, we compare against baselines including SAR3D, TRELLIS, CRM, and 3Dtopia-XL. As illustrated in Figure 4, the baselines exhibit limitations in capturing fine-grained visual features, suffering from geometric distortions and texture misalignments. In contrast, our method generates high-quality 3D meshes that preserve both geometry and appearance details. Moreover, our generation quality matches that of TRELLIS, our base model and performance upper bound, due to the integration of a well-trained 3D VQVAE and a carefully constructed image-to-3D dataset for LLM fine-tuning. For text-to-3D tasks, Figure 5 presents qualitative comparisons among baselines. The input prompts are randomly generated by ChatGPT-4o to cover a diverse range of objects. Since 3Dtopia-XL does not support text-to-3D tasks, we use ChatGPT-4o to generate reference images from the prompts. These images are then used as input for image-to-3D generation. It is evident that our method achieves precise alignment with the text prompts and excels at generating intricate, coherent details.
this section cite: []

Section: 3D Editing
As shown in Figure 6, ShapeLLM-Omni can edit 3D assets according to user-provided instructions while maintaining good identity consistency.
+"open the cabinet doors" +"add lid on top" +"add wings" +"grow a tail" Before After   Our results are slightly inferior to Trellis due to two main factors.
1) Trellis employs separate models for text-to-3D and image-to-3D generation, whereas our ShapeLLM-Omni unifies six tasks-text-to-3D and image-to-3D generation, 3D understanding, 3D editing, image understanding, and text reasoning-within a single model that also supports interactive conversation. This all-in-one design introduces optimization trade-offs that can affect generation quality. To verify this, we fine-tuned our model specifically for text-to-3D generation using the pre-trained weights, removing redundant prompts and freezing non-mesh textual embeddings.
With a learning rate of 1e-5, context size of 1536, batch size of 4, gradient accumulation of 2, and 5 epochs of training, the fine-tuned model lost its general text capabilities but ,as shown in the Table 7, achieved text-to-3D results comparable to Trellis-demonstrating the inherent difficulty of balancing multiple tasks within a unified framework.
2) Trellis is built on a Rectified Flow (diffusion) architecture, while our model adopts a discrete autoregressive design. Diffusion and flow-based models currently hold an inherent advantage in visual generation quality, and surpassing them with autoregressive architectures remains an open research challenge. Nonetheless, our focus lies not in pushing the absolute performance of autoregressive models, but in enabling unified 3D generation and understanding under this paradigm-an innovative and promising direction as autoregressive visual generation continues to advance.
this section cite: []

Section: Conclusion
In this work, we introduce ShapeLLM-Omni, a novel framework that advances both 3D generation and understanding through a 3D VQVAE. By constructing a comprehensive 3D-Alpaca dataset, we provide a data foundation to support future research on native 3D-modality large language models.
Limitation Constrained by limited resources, we possess only 70k 3D-editing pairs-far too few to achieve ChatGPT-4o-level results in 3D editing. Due to limited computing resources, our ShapeLLM-Omni only has 7B parameters. As a result, our performance hasn't yet reached the level of a true "3D version of ChatGPT-4o".
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b2 Title: Qwen technical report Year: (2023)
Ref_id:b3 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b4 Title: Demystifying mmd gans Year: (2018)
Ref_id:b5 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b6 Title: An information-rich 3d model repository Year: (2015)
Ref_id:b7 Title: Pointgpt: Auto-regressively generative pre-training from point clouds Year: (2023)
Ref_id:b8 Title: Microdreamer: Zero-shot 3d generation in 20 seconds by score-based iterative reconstruction Year: (2024)
Ref_id:b9 Title: Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation Year: (2023)
Ref_id:b10 Title: Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning Year: (2024-06)
Ref_id:b11 Title: Meshxl: Neural coordinate field for generative 3d foundation models Year: (2025)
Ref_id:b12 Title: Grounded 3d-llm with referent tokens Year: (2024)
Ref_id:b13 Title: Meshanything: Artist-created mesh generation with autoregressive transformers Year: (2024)
Ref_id:b14 Title: Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization Year: (2024)
Ref_id:b15 Title: Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae Year: (2025)
Ref_id:b16 Title: 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion Year: (2024)
Ref_id:b17 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b18 Title: Text-to-3d using gaussian splatting Year: (2024)
Ref_id:b19 Title: Video diffusion models are effective 3d generators Year: (2024)
Ref_id:b20 Title: Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms Year: (2024)
Ref_id:b21 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b22 Title: Abo: Dataset and benchmarks for real-world 3d object understanding Year: (2022)
Ref_id:b23 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2023)
Ref_id:b24 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: (2023)
Ref_id:b25 Title: Enhancing chat language models by scaling high-quality instructional conversations Year: (2023)
Ref_id:b26 Title: A family of large language models from glm-130b to glm-4 all tools Year: (2024)
Ref_id:b27 Title: Fila-video: Spatio-temporal compression for fine-grained long video understanding Year: (2025)
Ref_id:b28 Title: Meshtron: High-fidelity, artist-like 3d mesh generation at scale Year: (2024)
Ref_id:b29 Title: Openlrm: Open-source large reconstruction models Year: (2023)
Ref_id:b30 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b31 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b32 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b33 Title: Lrm: Large reconstruction model for single image to 3d Year: (2023)
Ref_id:b34 Title: 3d-llm: Injecting the 3d world into large language models Year: (2023)
Ref_id:b35 Title: Chat-scene: Bridging 3d scene and large language models with object identifiers Year: (2024)
Ref_id:b36 Title: Roboground: Robotic manipulation with grounded vision-language priors Year: (2025-06)
Ref_id:b37 Title: Spar3d: Stable point-aware reconstruction of 3d objects from single images Year: (2025)
Ref_id:b38 Title: Gpt-4o system card Year: (2024)
Ref_id:b39 Title: Gpt-4o system card Year: (2024)
Ref_id:b40 Title: Robin3d: Improving 3d large language model via robust instruction tuning Year: (2025)
Ref_id:b41 Title: 3d gaussian splatting for real-time radiance field rendering Year: (2023)
Ref_id:b42 Title: Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model Year: (2023)
Ref_id:b43 Title: Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d Year: (2023)
Ref_id:b44 Title: Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner Year: (2024)
Ref_id:b45 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2024)
Ref_id:b46 Title: Magic3d: High-resolution text-to-3d content creation Year: ()
Ref_id:b47 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b48 Title: Reconx: Reconstruct any scene from sparse views with video diffusion model Year: (2024)
Ref_id:b49 Title: Dreamrewardx: Boosting high-quality 3d generation with human preference alignment Year: (2025)
Ref_id:b50 Title: World model on million-length video and language with blockwise ringattention Year: (2024)
Ref_id:b51 Title: Visual instruction tuning Year: (2023)
Ref_id:b52 Title: Quadgpt: Native quadrilateral mesh generation with autoregressive models Year: (2025)
Ref_id:b53 Title: Boosting mesh generation with coordinates merging Year: (2025)
Ref_id:b54 Title: Mesh-rft: Enhancing mesh generation via fine-grained reinforcement fine-tuning Year: (2025)
Ref_id:b55 Title: One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization Year: (2023)
Ref_id:b56 Title: Zero-1-to-3: Zero-shot one image to 3d object Year: (2023)
Ref_id:b57 Title: Syncdreamer: Generating multiview-consistent images from a single-view image Year: (2023)
Ref_id:b58 Title: Wonder3d: Single image to 3d using cross-domain diffusion Year: (2024)
Ref_id:b59 Title: Marching cubes: A high resolution 3d surface construction algorithm Year: (1998)
Ref_id:b60 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b61 Title: Nerf: Representing scenes as neural radiance fields for view synthesis Year: (2021)
Ref_id:b62 Title: Hierarchical transformers are more efficient language models Year: (2021)
Ref_id:b63 Title: Dreamfusion: Text-to-3d using 2d diffusion. arXiv Year: (2022)
Ref_id:b64 Title: Shapellm: Universal 3d object understanding for embodied interaction Year: (2024)
Ref_id:b65 Title: Gpt4point: A unified framework for point-language understanding and generation Year: (2024-06)
Ref_id:b66 Title: Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d Year: (2024)
Ref_id:b67 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b68 Title: Dreambooth3d: Subject-driven text-to-3d generation Year: (2023)
Ref_id:b69 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b70 Title: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b71 Title: Zero123++: a single image to consistent multi-view diffusion base model Year: (2023)
Ref_id:b72 Title: Multi-view diffusion for 3d generation Year: (2023)
Ref_id:b73 Title: Meshgpt: Generating triangle meshes with decoder-only transformers Year: (2024)
Ref_id:b74 Title: Text-to-mesh generation with high-quality geometry, texture, and pbr materials Year: (2024)
Ref_id:b75 Title: Using shape to categorize: Low-shot learning with an explicit shape bias Year: (2021)
Ref_id:b76 Title: Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior Year: (2023)
Ref_id:b77 Title: Rethinking the inception architecture for computer vision Year: (2016)
Ref_id:b78 Title: Generative gaussian splatting for efficient 3d content creation Year: (2023)
Ref_id:b79 Title: Large multi-view gaussian model for high-resolution 3d content creation Year: (2024)
Ref_id:b80 Title: Edgerunner: Auto-regressive auto-encoder for artistic mesh generation Year: (2024)
Ref_id:b81 Title: Chameleon: Mixed-modal early-fusion foundation models Year: ()
Ref_id:b82 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b83 Title: Neural discrete representation learning Year: (2017)
Ref_id:b84 Title: Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion Year: (2024)
Ref_id:b85 Title: Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation Year: (2022)
Ref_id:b86 Title: Imagedream: Image-prompt multi-view diffusion for 3d generation Year: (2023)
Ref_id:b87 Title: Pf-lrm: Pose-free large reconstruction model for joint pose and shape prediction Year: (2023)
Ref_id:b88 Title: Rodin: A generative model for sculpting 3d digital avatars using diffusion Year: (2023)
Ref_id:b89 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b90 Title: Animatabledreamer: Text-guided non-rigid 3d model generation and reconstruction with canonical score distillation Year: (2023)
Ref_id:b91 Title: Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes Year: (2023)
Ref_id:b92 Title: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation Year: ()
Ref_id:b93 Title: Llama-mesh: Unifying 3d mesh generation with language models Year: (2024)
Ref_id:b94 Title: Single image to 3d textured mesh with convolutional reconstruction model Year: (2024)
Ref_id:b95 Title: Meshlrm: Large reconstruction model for high-quality meshes Year: (2024)
Ref_id:b96 Title: Consistent123: Improve consistency for one image to 3d object synthesis Year: (2023)
Ref_id:b97 Title: Generic 3d mesh generation via pivot vertices guidance Year: (2024)
Ref_id:b98 Title: Scaling mesh generation via compressive tokenization Year: (2024)
Ref_id:b99 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning [c] Year: (2023)
Ref_id:b100 Title: Unique3d: High-quality and efficient 3d mesh generation from a single image Year: (2024)
Ref_id:b101 Title: Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer Year: (2024)
Ref_id:b102 Title: Structured 3d latents for scalable and versatile 3d generation Year: (2024)
Ref_id:b103 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b104 Title: Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models Year: (2024)
Ref_id:b105 Title: Pointllm: Empowering large language models to understand point clouds Year: (2024)
Ref_id:b106 Title: Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model Year: (2023)
Ref_id:b107 Title: Ulip: Learning a unified representation of language, images, and point clouds for 3d understanding Year: (2023)
Ref_id:b108 Title: Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation Year: (2024)
Ref_id:b109 Title: Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion Year: (2025)
Ref_id:b110 Title: Stablenormal: Reducing diffusion variance for stable and sharp normal Year: ()
Ref_id:b111 Title: Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging Year: (2025)
Ref_id:b112 Title: Dreamreward: Text-to-3d generation with human preference Year: (2024)
Ref_id:b113 Title: Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models Year: (2024)
Ref_id:b114 Title: 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models Year: (2023)
Ref_id:b115 Title: Geometry-aware large reconstruction model for high-quality 3d gaussian generation Year: (2024)
Ref_id:b116 Title: Gs-lrm: Large reconstruction model for 3d gaussian splatting Year: (2024)
Ref_id:b117 Title: Clay: A controllable large-scale generative model for creating high-quality 3d assets Year: (2024)
Ref_id:b118 Title: Flexidreamer: single image-to-3d generation with flexicubes Year: (2024)
Ref_id:b119 Title: Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning Year: (2025)
Ref_id:b120 Title: Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation Year: (2023)
Ref_id:b121 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
Ref_id:b122 Title: Triplane meets gaussian splatting: Fast and generalizable single-view 3d reconstruction with transformers Year: (2024)
