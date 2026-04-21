Title: NEXTSTEP-1: TOWARD AUTOREGRESSIVE IMAGE GENERATION WITH CONTINUOUS TOKENS AT SCALE
Abstract: Prevailing autoregressive (AR) models for text-to-image generation either rely on heavy, computationally-intensive diffusion models to process continuous image tokens, or employ vector quantization (VQ) to obtain discrete tokens with quantization loss. In this paper, we push the autoregressive paradigm forward with NextStep-1, a 14B autoregressive model paired with a 157M flow matching head, trained on discrete text tokens and continuous image tokens with next-token prediction objectives. NextStep-1 achieves state-of-the-art performance for autoregressive models in text-to-image generation tasks, exhibiting strong capabilities in high-fidelity image synthesis. Furthermore, our method shows strong performance in image editing, highlighting the power and versatility of our unified approach. To facilitate open research, we have released our code and models to the community at https://github.com/stepfun-ai/NextStep-1.

Section: INTRODUCTION
Diffusion models have firmly established themselves as the dominant paradigm for high-fidelity image synthesis (Podell et al., 2024;Esser et al., 2024;Labs, 2024;Wu et al., 2025a). Despite this success, recent state-of-the-art architectures typically operate in a decoupled manner, relying on separate, pre-trained text encoders (e.g., T5 (Raffel et al., 2020) or CLIP (Radford et al., 2021)) to feed semantic features into a Multimodal Diffusion Transformer (MMDiT) (Esser et al., 2024) backbone via cross-attention. This non-end-to-end design imposes rigid constraints: it fixes the input context window, restricts deep multimodal fusion, and hinders the model's ability to handle arbitrary interleaved image-text sequences naturally.
Inspired by the scalability of Large Language Models (LLMs)(OpenAI, 2025a), the "next-token prediction" paradigm offers a compelling alternative for unified multimodal generation. Existing approaches generally fall into two categories, each with distinct limitations. The first category, represented by discrete autoregressive models (Sun et al., 2024a;Chen et al., 2025b;Wang et al., 2024b), relies on Vector Quantization (VQ) (Zheng et al., 2022;Eslami et al., 2021) to discretize images into visual tokens. This process introduces an information bottleneck-manifesting as reconstruction artifacts-and suffers from exposure bias (Han et al., 2025). Furthermore, to mitigate information loss, models like Emu3 (Wang et al., 2024b) must employ low compression rates, resulting in excessively long token sequences that inflate computational costs and training difficulty. The second category involves hybrid architectures, such as Transfusion (Zhou et al., 2025) and Bagel (Deng et al., 2025), which attempt to integrate noisy inputs and diffusion losses directly into the LLM. While promising, these methods often compromise training efficiency. Unlike pure next-token prediction, they typically require processing both noisy latents and clean conditioning signals (or ground truth images) simultaneously within bidirectional attention blocks. This data duplication significantly increases the sequence length and attention overhead, undermining the efficiency benefits traditionally associated with sparse autoregressive modeling. Besides, while recent explorations into continuous latent
Text Tokenizer Image Tokenizer Flow Matching Head LM Head dog on rock [bos] dog on rock …… [eoi] Causal Transformer …… …… …… …… Next Token Prediction …… [boi] [eoi] autoregression (Li et al., 2024c;Fan et al., 2024) attempt to bypass quantization, a substantial performance gap persists. To date, no continuous autoregressive model has matched the visual quality and consistency of state-of-the-art diffusion models while retaining the simplicity and scalability of a standard LLM.
In this work, we demonstrate that the key to closing the performance gap lies in the image representation. We introduce NextStep-1, an autoregressive model built around a novel image tokenizer that is specifically designed to create a well-dispersed and normalized latent space, thereby enabling stable training with high-dimensional continuous latents. By addressing the challenges in image tokenization, we demonstrate that a simple causal transformer with a lightweight flow matching head can achieve state-of-the-art results that rival top diffusion models in both quality and prompt adherence. We showcase the generation quality of our model in Section A. Our contributions are highlighted as follows:
1. We present NextStep-1, an autoregressive text-to-image model that achieves state-of-the-art performance with minimalist architecture, outperforming previous autoregressive methods as well as many strong diffusion-based methods. 2. We reveal key design principles for training a robust image tokenizer that enables stable autoregressive generation with high-dimensional continuous latents. 3. We demonstrate the state-of-the-art performance and versatility of NextStep-1 through comprehensive evaluations on benchmarks for both text-to-image generation (e.g., WISE, GenAI-Bench) and instruction-based image editing (e.g., GEdit-Bench), highlighting its superior capabilities in compositional reasoning and prompt fidelity.
this section cite: ['b28', 'b61', 'b12', 'b34', 'b17']

Section: METHOD

this section cite: []

Section: UNIFIED MULTI-MODEL GENERATION WITH CONTINUOUS VISUAL TOKENS
NextStep-1 extends the well-established autoregressive language modeling paradigm to image generation through a simple and intuitive architecture, as illustrated in Figure 1. To unify multimodal inputs into a single sequence, the images will be tokenized to continuous image tokens by the image tokenizer and combined with discrete text tokens. For a multimodal token sequence x = {x 0 , x 1 , ..., x n }, where x i is either a discrete text token or a continuous visual token, the autoregressive objective under the unified sequence is formalized as:
p(x) = n i=1 p(x i | x <i ).(1)
The unified multi-modal generation task proceeds by sampling the next token x i from the conditional distribution p(x i | x <i ) modeled by a network. Discrete text tokens are sampled via a language modeling head, while continuous image tokens are sampled by a flow-matching head.
Our training objective consists of two distinct losses: a standard cross-entropy loss for discrete text tokens, and a flow matching loss (Lipman et al., 2023b) for continuous image tokens. Specifically, the flow matching loss is the mean squared error between the predicted and target velocity vectors that map a noised patch to its corresponding clean patch. The model is trained end-to-end by optimizing a weighted sum of these two losses:
L total = λ text L text + λ visual L visual(2)
where L text and L visual denote the loss for text and image tokens respectively, which are balanced by the hyperparameters λ text and λ visual .
this section cite: []

Section: MODEL ARCHITECTURE
Image Tokenizer. Our image tokenizer is fine-tuned from flux VAE (Labs, 2024) with only reconstruction and perceptual losses. The tokenizer first encodes an image into 16-channel latents z, applying an 8× spatial downsampling factor. To stabilize and normalize the latent space, we apply token-wise normalization (detailed in Appendix C), standardizing each channel to zero mean and unit variance. Furthermore, to enhance the robustness of the image tokenizer and encourage a more uniform latent distribution, we introduce a stochastic perturbation to the normalized latents. This technique is adapted from σ-VAE (Sun et al., 2024c), where it was employed to prevent variance collapse. z = Normalization(z) + α • ε, where α ∼ U[0, γ] and ε ∼ N (0, I)
where ε is standard Gaussian noise, and its magnitude is scaled by a random factor α sampled uniformly from [0, γ]. The γ is a hyperparameter controlling the maximum noise intensity.
The latents from the image tokenizer are pixel-shuffled into a more compact sequence. This is achieved by applying a space-to-depth transformation with a 2×2 kernel, which flattens 2×2 spatial latents into the channel dimension. For example, this converts the latents of a 256×256 image into 16×16 grid of 64-channel tokens. This grid is then flattened into a 1D sequence of 256 tokens to serve as input for the following Causal Transformer.
this section cite: ['b28']

Section: Causal Transformer.
We initialize our model from the decoder-only Qwen2.5-14B (Yang et al., 2024), leveraging its strong language understanding and reasoning capabilities for text-to-image generation. We organize the multimodal input sequence in the following format:
{text} <image area>h*w <boi>{image} <eoi>...
where {text} denotes discrete text tokens, and {image} represents continuous image tokens. <boi> and <eoi> are special tokens marking the beginning-of-image and end-of-image. <image area>h*w represents the metadata about the spatial dimensions of the 2D image tokens.
Then the output hidden states from LLM are passed to two lightweight heads for modality-specific loss:
• Language Modeling Head. We compute Cross-Entropy loss for hidden states of texts.
• Patch-wise Flow Matching Head. Following (Li et al., 2024c), we use each patch-wise image hidden states as condition, denoise target patch at timesteps t, and compute the patchwise flow-matching loss (Lipman et al., 2023a) with a 157M, 12-layer, and 1536 hiddendimensions MLP.
For positional information, we use the standard 1D RoPE (Su et al., 2024). Despite the availability of more complex 2D or multimodal RoPE alternatives (Bai et al., 2025;Wang et al., 2024a), we found that the simple 1D formulation remains highly effective for mixed text-image sequences, and thus retain it for simplicity and efficiency.
this section cite: ['b89', 'b34', 'b66', 'b0']

Section: TRAINING RECIPES

this section cite: []

Section: TRAINING IMAGE TOKENIZER
Our image tokenizer is initialized from the Flux.1-dev VAE (Labs, 2024), selected for its strong reconstruction performance. We fine-tune this model on the image-text dataset detailed in Section B.2 to adapt it to our specific data distribution. For optimization, we employ the AdamW optimizer (Loshchilov and Hutter, 2019) with (β 1 = 0.9, β 2 = 0.95, ε = 1 × 10 -8 ) for its convergence stability. The model is trained for 50K steps with a total batch size of 512, using a constant learning rate of 1 × 10 -5 preceded by linear warm-up of 1000 steps.
this section cite: ['b28', 'b43']

Section: PRE-TRAINING
The specific hyperparameters and data ratios for our pre-training are detailed in Table 1. Specifically, the pre-training follows a three-stage curriculum designed to progressively refine the model's capabilities. Throughout these stages, all model parameters are trained end-to-end except for the pre-trained image tokenizer.
this section cite: []

Section: Stage1.
In this initial stage, the model learns a foundational understanding of image structure and composition. For computational efficiency, all images are resized and randomly cropped to a fixed 256×256 resolution. The training curriculum is composed of a diverse data mixture: 20% text-only corpora, 60% image-text pairs, and 20% interleaved data. This stage consumed approximately 1.23T tokens.
Stage2. We implement a dynamic resolution strategy to train the model on a range of higher resolutions, targeting 256×256 and 512×512 base areas. This strategy utilizes different aspect ratio buckets for computational efficiency. In this stage, we enrich the data mixture with more text-rich and video-interleaved data, leveraging the model's enhanced capacity to process fine details at these resolutions.
Annealing. In the final stage of pre-training, we perform an annealing phase to sharpen the model's capabilities on a highly curated dataset. This is achieved by training the model for one epoch on a high-quality subset of 20M samples, which were selected from Section B.2 by applying stricter filtering thresholds for aesthetic score, image clarity, semantic similarity, watermark, and so on. This annealing step significantly improves the model's final output, enhancing overall image structure, composition, texture, and aesthetic appeal.
this section cite: []

Section: POST-TRAINING
Following pre-training on a broad corpus to establish a generalist model, post-training serves to align the model's output with human preferences and downstream tasks. We achieve this alignment via a two-stage process: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO) (Rafailov et al., 2023). The hyperparameters for each stage are in Table 1.
this section cite: []

Section: Supervised Fine-Tuning (SFT).
The SFT stage enhances the model's instruction-following capabilities and aligns its outputs with human preferences. The SFT dataset, comprising a total of 5M samples, is organized into three components: 1) a corpus of human-selected image-text pairs with high semantic consistency and visual appeal, augmented by images from other generative models to improve the model's handling of complex and imaginative prompts through distillation; 2) Chainof-Thought (CoT) data (Wei et al., 2022;Deng et al., 2025), improving text-to-image generation by incorporating a language-based reasoning step before the final image is created; 3) high-quality instruction-guided image-to-image data from Section B.3 to strengthen the model's image editing capabilities.
this section cite: ['b12']

Section: Direct Preference Optimization (DPO).
To align our model with human preferences, we employ Direct Preference Optimization (DPO) (Rafailov et al., 2024), a method inspired by Diffusion-DPO (Wallace et al., 2024). To this end, we construct two distinct types of preference datasets from a curated set of approximately 20,000 diverse prompts.
1. Standard DPO Dataset: For each prompt c, we directly use the SFT model to generate 16 candidate images. These images is then scored by ImageReward (Xu et al., 2023) to form a preference pair (y w , y l ), where the winning image y w is randomly sampled from the top 4 candidates, while the losing image y l is randomly sampled from the remaining 12.
2. Self-CoT DPO Dataset: To enhance the model's reasoning capabilities, we introduce an explicit reasoning step. For each prompt c, we first prompt our model to generate a detailed textual CoT, which is then extended to the original prompt. Using this CoT-enhanced prompt, we follow the identical pipeline as above to form a preference pair (y w , y l ).
this section cite: ['b60']

Section: DATA
We construct a diverse training corpus designed to foster both robust text-to-image generation and versatile image editing capabilities. Detailed pipelines are described in Section B.
this section cite: []

Section: MODEL PERFORMANCE

this section cite: []

Section: PERFORMANCE OF TEXT-TO-IMAGE GENERATION
We comprehensively evaluate the text-to-image (T2I) generation performance of NextStep-1 on several representative benchmarks, each targeting different aspects of image generation, including visual-textual alignment and world knowledge (full results in section A). As shown in Table 2, we assess NextStep-1's prompt-following ability across three key benchmarks, GenEval (Ghosh et al., 2023) and GenAI-Bench (Li et al., 2024a), and OneIG-Bench (Chang et al., 2025). Results demonstrate that NextStep-1 is competitive with leading diffusion models. Furthermore, its outstanding performance on long-context and multi-object scene, confirms its reliable compositional fidelity under complex prompts.
this section cite: ['b5']

Section: PERFORMANCE OF IMAGE EDITING
Quantitative Results on Editing Benchmarks. We developed NextStep-1-Edit by finetuning NextStep-1 on 1M high-quality edit-only data in Section B.3, demonstrates competitive performance against advanced diffusion-based models. As shown in Table 3, NextStep-1-Edit achieves scores of 6.58 on GEdit-Bench-EN (Liu et al., 2025c) and 3.71 on ImgEdit-Bench (Ye et al., 2025), indicating its strong practical editing capabilities.
this section cite: ['b92']

Section: DISCUSSIONS

this section cite: []

Section: WHAT GOVERNS IMAGE GENERATION: THE AR TRANSFORMER OR THE FM HEAD?
A key architectural distinction of our framework is its direct, autoregressive modeling of continuous image tokens using a flow matching objective. Prevailing autoregressive models for image generation (Sun et al., 2023;2024b;Dong et al., 2024;Zhou et al., 2025;Chen et al., 2025a) typically rely on heavy, diffusion models for a entire image: an autoregressive model first produces a semantic Table 3: Comparison of image editing performance on GEdit-Bench (Full Set) (Liu et al., 2025c) and ImgEdit-Bench (Ye et al., 2025). G SC, G PQ, and G O refer to the metrics evaluated by GPT-4. 1 (OpenAI, 2025a).
Performance is evaluated based on the NextStep-1-Edit with 1:1 aspect ratio.
Model GEdit-Bench-EN (Full Set)↑ GEdit-Bench-CN (Full Set)↑ ImgEdit-Bench↑ G SC G PQ G O G SC G PQ G O Proprietary Gemini
2.0 (Gemini2, 2025) 6.87 7.44 6.51 5.26 7.60 5.14 -Doubao (Shi et al., 2024) 7.22 7.89 6.98 7.17 7.79 6.84 -GPT-4o (OpenAI, 2025b) 7.74 8.13 7.49 7.52 8.02 7.30 4.20 Flux.1-Kontext-pro (Labs et al., 2025) 7.02 7.60 6.56 1.11 7.36 1.23 -Open-source Instruct-Pix2Pix (Brooks et al., 2023) 3.30 6.19 3.22 ---1.88 MagicBrush (Zhang et al., 2023a) 4.52 6.37 4.19 ---1.83 AnyEdit (Yu et al., 2024a) 3.05 5.88 2.85 ---2.45 OmniGen (Xiao et al., 2024) 5.88 5.87 5.01 ---2.96 OmniGen2 (Wu et al., 2025b) 7.16 6.77 6.41 ---3.44 Step1X-Edit v1.0 (Liu et al., 2025c) 7.13 7.00 6.44 7.30 7.14 6.66 3.06 Step1X-Edit v1.1 (Liu et al., 2025c) 7.66 7.35 6.97 7.65 7.40 6.98 -BAGEL (Deng et al., 2025) 7.36 6.83 6.52 7.34 6.85 6.50 3.42 Flux.1-Kontext-dev (Labs et al., 2025) --6.26 ---3.71 GPT-Image-Edit (Wang et al., 2025c) --7.24 ---3.80 NextStep-1 7.15 7.01 6.58 6.88 7.02 6.40 3.71
embedding, which is then used to condition a diffusion model that generates an entire image in a single denoising process. In contrast, our model autoregressively generates the image patch-by-patch, modeling the distribution of each patch with a significantly more lightweight flow matching model. We argue that this establishes our framework under the pure autoregressive paradigm with next-token prediction (NTP) modeling, rather than a diffusion model merely orchestrated by a Transformer.   A key finding from our experiments is the model's surprising insensitivity to the size of its flowmatching head. We ablated this on three heads with different sizes (small, base, and large). For each experiment, we re-initialized and trained only the head for 10k steps. Despite the significant variation in model size, all three heads produced remarkably similar results (Table 5, Figure 2). This insensitivity to the head's size strongly suggests that the transformer backbone performs the core generative modeling of the conditional distribution p(x i | x <i ). The flow-matching head, akin to the LM head in language models, primarily acts as a lightweight sampler that translates the transformer's contextual prediction into a continuous token. Consequently, the essential generative logic resides within the transformer's autoregressive NTP process.
this section cite: ['b92', 'b84', 'b12']

Section: TOKENIZER IS THE KEY TO IMAGE GENERATION

this section cite: []

Section: Mitigating Instability under Strong Classifier-Free Guidance.
A known failure mode in VAEbased autoregressive models is the emergence of visual artifacts, such as gray patches, particularly under strong classifier-free guidance scales (Fan et al., 2024). While prior work hypothesized this instability stemmed from discontinuities in 1D positional embeddings, our analysis reveals that the root cause lies in the amplification of token-level distributional shifts under high guidance scales.
At inference time, CFG is introduced to enhance conditional fidelity. The guided prediction ṽ is computed via an interpolation:
ṽ(x|y) = (1 -w) • v θ (x|∅) + w • v θ (x|y)(4)
where v θ (x|∅) and v θ (x|y) are the unconditional and conditional predictions, and w is guidance scale. In diffusion models, inference with high guidance scale is stable because latent variables are typically normalized, ensuring that conditional and unconditional predictions maintain a consistent scale. However, in token-level autoregressive models, global normalization of the entire latent tensor does not enforce per-token statistical consistency. Consequently, small discrepancies between conditional and unconditional predictions are magnified by a large guidance scale, leading to a significant drift in the statistics of generated tokens over the sequence.
We empirically demonstrate this phenomenon in Figure 3. At a moderate guidance scale of 1.5, the per-token mean and variance remain stable throughout the generation process. In contrast, at a high guidance scale of 3.0, both statistics diverge significantly for later tokens, a distributional shift that corresponds directly to the appearance of visual artifacts. Our tokenizer design, which incorporates token-wise normalization (see Equation ( 3)), directly addresses this issue by enforcing per-token statistical stability. This simple but critical design choice mitigates the instability, enabling the use of strong guidance without degrading image quality.
A Regularized Latent Space is Critical for Generation A key finding of our work is a counterintuitive inverse correlation between the generation loss and the final synthesis quality of the autoregressive model. Specifically, applying higher noise intensity (γ in Equation ( 3)) during tokenizer training increases generation loss but paradoxically improves the quality of the generated images.
Table 6: Comparison of reconstruction performance on ImageNet-1K 256×256 (Deng et al., 2009). (Zheng et al., 2022) 32x32 27.04 0.74 LlamaGen (Sun et al., 2024a) 32x32 24.44 0.77 VAR (Tian et al., 2024) 680 22.12 0.62 TiTok-S-128 (Yu et al., 2024b) 128 17.52 0.44 Sefltok (Wang et al., 2025b) 1024 26.30 0.81
Tokenizer Latent Shape PSNR ↑ SSIM ↑ Discrete Tokenizer SBER-MoVQGAN (270M)
this section cite: ['b17', 'b13', 'b17']

Section: Continuous Tokenizer
Stable Diffusion 1.5 (Rombach et al., 2022) 32x32x4 25.18 0.73 Stable Diffusion XL (Podell et al., 2024) 32x32x4 26.22 0.77 Stable Diffusion 3 Medium (Esser et al., 2024) 32x32x16 30.00 0.88 Flux.1-dev (Labs, 2024) 32x32x16 31.64 0.91 NextStep-1 32x32x16 30.60 0.89
For instance, NextStep-1 uses a tokenizer trained at γ = 0.5, which incurred the highest generation loss yet produced the highest-fidelity images. Conversely, tokenizers trained for low generation loss caused the autoregressive model to yield outputs resembling pure noise.
We attribute this phenomenon to noise regularization cultivating a well-conditioned latent space. This process enhances two key properties: the tokenizer decoder's robustness to latent perturbations (Figure 4) and a more dispersed latent distribution (Figure A3), a property prior work has also found beneficial for generation (Yang et al., 2025;Yao et al., 2025;Sun et al., 2024c). While it remains unclear whether robustness or dispersion plays a critical role, these results underscore the practical benefits of noise-based regularization and highlight promising directions for future analysis.
Reconstruction Quality is the Upper Bound of Generation Quality. The reconstruction fidelity of the image tokenizer fundamentally determines the upper bound for the quality of the final generated image, particularly for fine details and textures. This principle has been validated in numerous recent studies (Esser et al., 2024;Labs, 2024;Dai et al., 2023), leading to a trend in the diffusion paradigm of building generative models on top of VAEs with exceptional reconstruction performance (e.g., PSNR >30). In contrast, VQ-based autoregressive models have historically struggled to surpass this threshold, as shown in Table 6. While a trade-off between reconstruction and generation quality is often debated (Yao et al., 2025), our work successfully applies autoregressive models to high-fidelity continuous VAEs, bridging this gap.
this section cite: ['b90', 'b28', 'b9']

Section: RELATED WORK
Diffusion and Flow Matching Models. Diffusion models have established themselves as the dominant paradigm for high-fidelity image synthesis. While early foundations relied on U-Net (Ronneberger et al., 2015) architectures, recent advancements have shifted towards Diffusion Transformers (DiTs) (Peebles and Xie, 2023) to leverage the scalability of attention mechanisms. More recently, Flow Matching (Lipman et al., 2023a;Liu et al., 2022) has been adopted to rectify generation trajectories, with models like Flux (Labs, 2024) demonstrating SOTA visual quality. However, these systems typically rely on separate, pre-trained text encoders (e.g., T5-XXL (Raffel et al., 2020), CLIP (Radford et al., 2021)) to process prompts. This decoupled design imposes a fixed context window, limiting deep multimodal fusion and the model's ability to handle interleaved inputs or long-context reasoning naturally. Furthermore, the generation process in these models-often involving bidirectional attention over the entire image latent space-is computationally intensive and distinct from the token-by-token reasoning inherent to Large Language Models (LLMs).
this section cite: ['b63', 'b42', 'b28', 'b61']

Section: Discrete Autoregressive Models.
Another line of research treats image generation as a sequence modeling problem, aiming to unify vision and language under a single transformer backbone. Pioneering works such as LlamaGen (Sun et al., 2024a), Janus-Pro (Chen et al., 2025b), Emu3 (Wang et al., 2024b), and InfinityStar (Liu et al., 2025a) have successfully adapted the "next-token prediction" paradigm to vision. However, these methods predominantly rely on Vector Quantization (VQ) (Zheng et al., 2022) to discretize continuous images into a finite codebook. This introduces a fun-damental bottleneck: the discretization process incurs information loss, limiting the reconstruction upper bound.
Unified and Hybrid Architectures. To bridge the gap between the generative quality of diffusion and the scalability of LLMs, recent studies have explored hybrid architectures. Models like Dream-LLM (Dong et al., 2024), Emu2 (Sun et al., 2024b), and Qwen-Image (Wu et al., 2025a) achieve SOTA generation by appending a heavy diffusion head (often a MMDiT (Esser et al., 2024)) to an LLM. While effective, relying on the separate text encoder and diffusion model limits the potential for unified multi-modality fusion and making performance sensitive to the interleaved generation. Another direction, represented by Transfusion (Zhou et al., 2025) and Bagel (Deng et al., 2025), integrates continuous noisy inputs and diffusion loss directly into the LLM training. However, these methods often compromise the "pure" autoregressive nature of LLMs for image data. They typically employ bidirectional attention and require feeding both the noisy latent and the clean image (or condition) into the model simultaneously. These special designs increase the attention cost and sequence length during training, thereby hindering the efficient application of scaling laws compared to pure next-token prediction.
In contrast to the aforementioned approaches, NextStep-1 adheres to a strict causal autoregressive objective on continuous image tokens (Li et al., 2024c;Fan et al., 2025). NextStep-1 demonstrates that a standard LLM backbone, equipped with a lightweight flow-matching head, can serve as the primary generative engine. This approach avoids the quantization artifacts of discrete AR models while maintaining the causal reasoning benefits and infrastructure compatibility of LLMs. By eliminating the need for bidirectional attention or noisy input duplication, NextStep-1 achieves highfidelity generation with a unified, scalable, and simplified architecture.
this section cite: ['b12', 'b34', 'b18']

Section: CONCLUSION
We propose NextStep-1, a fully autoregressive model with versatile image generation and editing capabilities that sequentially predicts the next continuous image patch token via a lightweight flow matching head. To address the stability and quality issues associated with high-dimensional continuous image tokens, we develop a robust autoencoder using noise perturbation and token-wise input latent normalization. We show that this design is crucial for the performance of autoregressive model, and demonstrate its competitive results across a wide variety of image generation and editing tasks, not only achieving state-of-the-art results among existing autoregressive image generation models, but also showing competitive performance when compared to leading diffusion-based methods. We plan to release the model and code to inspire further research, foster collaboration, and accelerate progress in this exciting frontier.
Table A1: Comparison on OneIG-Bench (Chang et al., 2025) in English prompts. Method Alignment Text Reasoning Style Diversity Overall↑ Proprietary Imagen3 (Baldridge et al., 2024) 0.843 0.343 0.313 0.359 0.188 0.409 Recraft V3 (team, 2024) 0.810 0.795 0.323 0.378 0.205 0.502 Kolors 2.0 (team, 2025) 0.820 0.427 0.262 0.360 0.300 0.434 Seedream 3.0 (Gao et al., 2025) 0.818 0.865 0.275 0.413 0.277 0.530 Imagen4 (deepmind Imagen4 team, 2025) 0.857 0.805 0.338 0.377 0.199 0.515 GPT-4o (OpenAI, 2025b) 0.851 0.857 0.345 0.462 0.151 0.533 Diffusion Stable Diffusion 1.5 (Rombach et al., 2022) 0.565 0.010 0.207 0.383 0.429 0.319 Stable Diffusion XL (Podell et al., 2024) 0.688 0.029 0.237 0.332 0.296 0.316 Stable Diffusion 3.5 Large (Stability-AI, 2024) 0.809 0.629 0.294 0.353 0.225 0.462 Flux.1-dev (Labs, 2024) 0.786 0.523 0.253 0.368 0.238 0.434 CogView4 (Z.ai, 2025) 0.786 0.641 0.246 0.353 0.205 0.446 SANA-1.5 1.6B (PAG) (Xie et al., 2025a) 0.762 0.054 0.209 0.387 0.222 0.327 SANA-1.5 4.8B (PAG) (Xie et al., 2025a) 0.765 0.069 0.217 0.401 0.216 0.334 Lumina-Image 2.0 (Qin et al., 2025) 0.819 0.106 0.270 0.354 0.216 0.353 HiDream-I1-Full (Cai et al., 2025) 0.829 0.707 0.317 0.347 0.186 0.477 BLIP3-o (Chen et al., 2025a) 0.711 0.013 0.223 0.361 0.229 0.307 BAGEL (Deng et al., 2025) 0.769 0.244 0.173 0.367 0.251 0.361 Show-o2-1.5B (Xie et al., 2025b) 0.798 0.002 0.219 0.317 0.186 0.304 Show-o2-7B (Xie et al., 2025b) 0.817 0.002 0.226 0.317 0.177 0.308 OmniGen2 (Wu et al., 2025b) 0.804 0.680 0.271 0.377 0.242 0.475 Qwen-Image (Wu et al., 2025a) 0.882 0.891 0.306 0.418 0.197 0.539 AutoRegressive Emu3 (Wang et al., 2024b) 0.737 0.010 0.193 0.361 0.251 0.311 Janus-Pro (Chen et al., 2025b) 0.553 0.001 0.139 0.276 0.365 0.267 NextStep-1 0.826 0.507 0.224 0.332 0.199 0.417 Table A2: Comparison of world knowledge reasoning on WISE (Niu et al., 2025). † result is with Self-CoT. Model Cultural Time Space Biology Physics Chemistry Overall↑ Overall (Rewrite)↑ Proprietary GPT-4o (OpenAI, 2025b) 0.81 0.71 0.89 0.83 0.79 0.74 0.80 -Diffusion Stable Diffusion 1.5 (Rombach et al., 2022) 0.34 0.35 0.32 0.28 0.29 0.21 0.32 0.50 Stable Diffusion XL (Podell et al., 2024) 0.43 0.48 0.47 0.44 0.45 0.27 0.43 0.65 Stable Diffusion 3.5 Large (Stability-AI, 2024) 0.44 0.50 0.58 0.44 0.52 0.31 0.46 0.72 PixArt-Alpha (Chen et al., 2024) 0.45 0.50 0.48 0.49 0.56 0.34 0.47 0.63 Playground v2.5 (Li et al., 2024b) 0.49 0.58 0.55 0.43 0.48 0.33 0.49 0.71 Flux.1-dev (Labs, 2024) 0.48 0.58 0.62 0.42 0.51 0.35 0.50 0.73 MetaQuery-XL (Pan et al., 2025) 0.56 0.55 0.62 0.49 0.63 0.41 0.55 -BAGEL (Deng et al., 2025) 0.44/0.76 † 0.55/0.69 † 0.68/0.75 † 0.44/0.65 † 0.60/0.75 † 0.39/0.58 † 0.52/0.70 † 0.71/0.77 † Qwen-Image (Wu et al., 2025a) 0.62 0.63 0.77 0.57 0.75 0.40 0.62 -AutoRegressive Show-o-512 (Xie et al., 2024) 0.28 0.40 0.48 0.30 0.46 0.30 0.35 0.64 VILA-U (Wu et al., 2024) 0.26 0.33 0.37 0.35 0.39 0.23 0.31 -Emu3 (Wang et al., 2024b) 0.34 0.45 0.48 0.41 0.45 0.27 0.39 0.63 Janus-Pro-7B (Chen et al., 2025b) 0.30 0.37 0.49 0.36 0.42 0.26 0.35 0.71 NextStep-1 0.51/0.70 † 0.54/0.65 † 0.61/0.69 † 0.52/0.63 † 0.63/0.73 † 0.48/0.52 † 0.54/0.67 † 0.79/0.83 † Quantitative Results. We also conduct a fine-grained analysis on OneIG-Bench (Chang et al., 2025) with English prompts. Table A1 reflects NextStep-1's significant advantage across areas such as alignment, text rendering, reasoning and stylistic control over existing autoregressive models. To evaluate NextStep-1's ability to integrate world knowledge into image generation, we use the WISE benchmark (Niu et al., 2025), which emphasizes factual grounding and semantic understanding. As shown in Table A2, NextStep-1 achieves the best performance among autoregressive models, also exceeding most diffusion models, with an even more significant boost under the prompt rewrite protocol. Collectively, these results demonstrate NextStep-1's robust knowledge-aware semantic alignment and cross-domain reasoning capabilities. Web data collection Coarse filtering Face detection & character binding Frame extraction & MLLM captioning Frame extraction & MLLM captioning Step I: Step II: Step III: Step I: Step II: Step III: uniform sampling ArcFace Get face embeddings within each scene Face detection & character binding Matching faces based on cosine similarity Char id: Frame id Bbox array Char id: Frame id Bbox array Char id: Frame id Bbox array Char id: Frame id Bbox array Saving meta infos for each character Char id: Frame id Bbox array Frame extraction using meta infos Step 1o "The camera shifts to the outdoors, where the man is talking to a woman" Checklist: • Object Consistency • Attribute Consistency • Relation Consistency • …… Step 1o
Figure A2: Data processing of character-centric data.
context. A key contribution, detailed in Figure A2, is our character-centric dataset, NextStep-Video-Interleave-5M. For this dataset, we extracted video frames centered around specific characters and generated rich, storytelling-style captions akin to (Oliveira and de Matos, 2025), thereby significantly improving the model's capacity for multi-turn interaction. Finally, to bolster geometric reasoning, we curated multiview data from two open-source datasets, MV-ImageNet-v2 (Han et al., 2024) and Objaverse-XL (Deitke et al., 2023), which enhances the model's ability to maintain multiview consistency.
this section cite: ['b25']

Section: References
Ref_id:b0 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b1 Title:  Year: (2024)
Ref_id:b2 Title: Improving image generation with better captions Year: (2023)
Ref_id:b3 Title: Instructpix2pix: Learning to follow image editing instructions Year: ()
Ref_id:b4 Title: Hidreami1: A high-efficient image generative foundation model with sparse diffusion transformer Year: (2025)
Ref_id:b5 Title: Oneig-bench: Omni-dimensional nuanced evaluation for image generation Year: (2025)
Ref_id:b6 Title: Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation Year: ()
Ref_id:b7 Title: Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset Year: (2025)
Ref_id:b8 Title: Janus-pro: Unified multimodal understanding and generation with data and model scaling Year: (2025)
Ref_id:b9 Title: Emu: Enhancing image generation models using photogenic needles in a haystack Year: (2023)
Ref_id:b10 Title: Imagen4 team. Imagen4 Year: (2025)
Ref_id:b11 Title: Objaverse-xl: A universe of 10m+ 3d objects Year: ()
Ref_id:b12 Title: Emerging properties in unified multimodal pretraining Year: (2025)
Ref_id:b13 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b14 Title: Dreamllm: Synergistic multimodal comprehension and creation Year: ()
Ref_id:b15 Title: Taming transformers for high-resolution image synthesis Year: ()
Ref_id:b16 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: ()
Ref_id:b17 Title: Fluid: Scaling autoregressive text-to-image generative models with continuous tokens Year: (2024)
Ref_id:b18 Title: Unified autoregressive visual generation and understanding with continuous tokens Year: (2025)
Ref_id:b19 Title: Seedream 3.0 technical report Year: (2025)
Ref_id:b20 Title: Seed-x: Multimodal models with unified multi-granularity comprehension and generation Year: (2024)
Ref_id:b21 Title: Experiment with gemini 2.0 flash native image generation Year: (2025-02)
Ref_id:b22 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: ()
Ref_id:b23 Title: Better & faster large language models via multi-token prediction Year: (2024)
Ref_id:b24 Title: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis Year: ()
Ref_id:b25 Title: Mvimgnet2. 0: A larger-scale dataset of multi-view images Year: (2024)
Ref_id:b26 Title: Ella: Equip diffusion models with llm for enhanced semantic alignment Year: (2024)
Ref_id:b27 Title: IEEE International Conference on Computer Vision (ICCV) Year: ()
Ref_id:b28 Title:  Year: (2024)
Ref_id:b29 Title: Flux.1-fill-dev Year: (2025)
Ref_id:b30 Title: Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space Year: (2025)
Ref_id:b31 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b32 Title: Evaluating and improving compositional text-to-visual generation Year: ()
Ref_id:b33 Title: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation Year: (2024)
Ref_id:b34 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b35 Title: Mogao: An omni foundation model for interleaved multi-modal generation Year: (2025)
Ref_id:b36 Title: Evaluating text-tovisual generation with image-to-text generation Year: (2024)
Ref_id:b37 Title: Flow matching for generative modeling Year: ()
Ref_id:b38 Title: Flow matching for generative modeling Year: (2023)
Ref_id:b39 Title: Infinitystar: Unified spacetime autoregressive modeling for visual generation Year: (2025)
Ref_id:b40 Title: Improving video generation with human feedback Year: (2025)
Ref_id:b41 Title: Step1xedit: A practical framework for general image editing Year: (2025)
Ref_id:b42 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b43 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b44 Title: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps Year: (2022)
Ref_id:b45 Title: Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models Year: (2025)
Ref_id:b46 Title:  Year: (2025)
Ref_id:b47 Title: Tokenshuffle: Towards high-resolution image generation with autoregressive models Year: (2025)
Ref_id:b48 Title: On distillation of guided diffusion models Year: (2023)
Ref_id:b49 Title: Wise: A world knowledge-informed semantic evaluation for text-to-image generation Year: (2025)
Ref_id:b50 Title: Storyreasoning dataset: Using chain-of-thought for scene understanding and grounded story generation Year: (2025)
Ref_id:b51 Title: Introducing gpt-4.1 in the api Year: ()
Ref_id:b52 Title: Introducing 4o image generation Year: (2025)
Ref_id:b53 Title: Transfer between modalities with metaqueries Year: (2025)
Ref_id:b54 Title: Scalable diffusion models with transformers Year: ()
Ref_id:b55 Title: Dreambench++: A human-aligned benchmark for personalized image generation Year: (2024)
Ref_id:b56 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: ()
Ref_id:b57 Title: Lumina-image 2.0: A unified and efficient image generative framework Year: (2025)
Ref_id:b58 Title: Learning transferable visual models from natural language supervision Year: ()
Ref_id:b59 Title: Direct preference optimization: Your language model is secretly a reward model Year: ()
Ref_id:b60 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b61 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b62 Title: High-resolution image synthesis with latent diffusion models Year: ()
Ref_id:b63 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b64 Title: Seededit: Align image re-generation to image editing Year: (2024)
Ref_id:b65 Title:  Year: (2024)
Ref_id:b66 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b67 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b68 Title: Emu: Generative pretraining in multimodality Year: ()
Ref_id:b69 Title: Generative multimodal models are in-context learners Year: (2024)
Ref_id:b70 Title: Multimodal latent language modeling with next-token diffusion Year: (2024)
Ref_id:b71 Title:  Year: (2025)
Ref_id:b72 Title: Recraft v3 Year: (2024)
Ref_id:b73 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: ()
Ref_id:b74 Title: Diffusion model alignment using direct preference optimization Year: ()
Ref_id:b75 Title: Step-3 is large yet affordable: Model-system co-design for cost-effective decoding Year: (2025)
Ref_id:b76 Title: Discrete visual tokens of autoregression, by diffusion, and for reasoning Year: (2025)
Ref_id:b77 Title: Qwen2-vl: Enhancing visionlanguage model's perception of the world at any resolution Year: (2024)
Ref_id:b78 Title: Emu3: Next-token prediction is all you need Year: (2024)
Ref_id:b79 Title: Gpt-image-edit-1.5 m: A million-scale, gpt-generated image dataset Year: (2025)
Ref_id:b80 Title: Chain-ofthought prompting elicits reasoning in large language models Year: ()
Ref_id:b81 Title: Qwen-image technical report Year: (2025)
Ref_id:b82 Title: Omnigen2: Exploration to advanced multimodal generation Year: (2025)
Ref_id:b83 Title: Vilau: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b84 Title: Omnigen: Unified image generation Year: (2024)
Ref_id:b85 Title: Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer Year: (2025)
Ref_id:b86 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b87 Title: Show-o2: Improved native unified multimodal models Year: (2025)
Ref_id:b88 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: ()
Ref_id:b89 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b90 Title: Latent denoising makes good visual tokenizers Year: (2025)
Ref_id:b91 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: ()
Ref_id:b92 Title: Imgedit: A unified image editing dataset and benchmark Year: (2025)
Ref_id:b93 Title: Anyedit: Mastering unified high-quality image editing for any idea Year: (2024)
Ref_id:b94 Title: An image is worth 32 tokens for reconstruction and generation Year: (2024)
Ref_id:b95 Title:  Year: (2025)
Ref_id:b96 Title: Magicbrush: A manually annotated dataset for instruction-guided image editing Year: ()
Ref_id:b97 Title: Adding conditional control to text-to-image diffusion models Year: (2023)
Ref_id:b98 Title: 2.5 years in class: A multimodal textbook for vision-language pretraining Year: (2025)
Ref_id:b99 Title: Movq: Modulating quantized vectors for high-fidelity image generation Year: ()
Ref_id:b100 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: ()
Ref_id:b101 Title: Each category is curated to serve a distinct role in fostering different aspects of the model's generative abilities. B.1 TEXT-ONLY CORPUS To preserve the extensive language capabilities inherent in the large language model (LLM), we incorporate 400B text-only tokens sampled from Step-3 (Wang et al., 2025a) during training. B.2 IMAGE-TEXT PAIR DATA Data consisting of image-text pairs forms the foundation of the model's text-to-image generation capabilities Year: ()
Ref_id:b102 Title: We collected a large-scale dataset from diverse sources, including web data, multi-task VQA data and text-rich documents Year: ()
Ref_id:b103 Title: Quality-Based Filtering: We then applied a rigorous filtering process, evaluating each image on aesthetic quality, watermark presence, clarity, OCR detection, and text-image semantic alignment Year: ()
Ref_id:b104 Title: Re-captioning: After deduplicating the filtered images, we used the Step-1o-turbo to generate rich and detailed captions for each image in both English and Chinese Year: ()
Ref_id:b105 Title: This multi-stage pipeline yields a final dataset of 550M high-quality image-text pairs, providing a foundation for training a model with both strong aesthetic sense and broad world knowledge. B.3 INSTRUCTION-GUIDED IMAGE-TO-IMAGE DATA To enable a wide range of practical applications, we curated a high-quality dataset for instructionguided image-to-image tasks Year: (2023)
Ref_id:b106 Title: 2025c), all editing data were subjected to a rigorous VLM-based filtering pipeline that assessed both image-pair quality, rationality, consistency, and instruction alignment Year: ()
Ref_id:b107 Title: INTERLEAVED DATA Interleaved data seamlessly integrates text and images, offering rich and nuanced sequential associations between modalities. Specifically, our knowledge-rich interleaved dataset is primarily composed of four distinct categories: general video-interleaved data, tutorials, character-centric scenes, and multi-view data. To endow our model with extensive world knowledge, we first constructed a large-scale, 80Msample video-interleaved dataset. This was achieved through a meticulous curation pipeline, inspired by Step-Video (Ma et al., 2025a), which encompasses frame extraction, deduplication, and captioning. Furthermore, following the methodology of mmtextbook Year: (2025)
