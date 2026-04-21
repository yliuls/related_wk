Title: UniTok: A Unified Tokenizer for Visual Generation and Understanding
Abstract: Visual generative and understanding models typically rely on distinct tokenizers to process images, presenting a key challenge for unifying them within a single framework. Recent studies attempt to address this by connecting the training of VQVAE (for autoregressive generation) and CLIP (for understanding) to build a unified tokenizer. However, directly combining these training objectives has been observed to cause severe loss conflicts. In this paper, we show that reconstruction and semantic supervision do not inherently conflict. Instead, the underlying bottleneck stems from limited representational capacity of discrete token space. Building on these insights, we introduce UniTok, a unified tokenizer featuring a novel multi-codebook quantization mechanism that effectively scales up the vocabulary size and bottleneck dimension. In terms of final performance, UniTok sets a new record of 0.38 rFID and 78.6% zero-shot accuracy on ImageNet. Besides, UniTok can be seamlessly integrated into MLLMs to unlock native visual generation capability, without

Section: Introduction
The advent of GPT-4o [33] highlights the immense potential of Multimodal Large Language Models (MLLMs) with native visual generation capabilities [12,46,62,73,58]. These unified models offer precise control in multimodal interactions, enabling exceptional fluency in tasks such as multi-turn image editing and visual in-context learning. However, a fundamental dilemma remains in the choice of visual tokenizers for unified MLLMs -e.g., the CLIP [38,71] tokenizer excels in multimodal understanding but complicates generative modeling due to its high-dimensional, continuous feature space; Conversely, the discrete VQVAE [8] tokenizer fits autoregressive generation but struggles to capture essential semantics for understanding [62].
In this work, we aim to design a unified visual tokenizer to bridge the gap in multimodal generation and understanding. Intuitively, this can be achieved by integrating CLIP supervision into VQVAE training, resulting in a discrete tokenizer capturing both fine-grained details and high-level semantics. However, we empirically find this training recipe confronts severe convergence issues [61] and largely falls behind the CLIP baseline in multimodal understanding (Figure 1). While prior studies commonly attribute these challenges to conflicts between semantic and pixel-level feature learning [61,37,58], recent progress in visual generation suggests the opposite, showing semantic regularization could benefit tokenizers in reconstruction-oriented training [63,4,19]. Such disparity motivates us question: Do reconstruction and semantic losses truly conflict in tokenizer training?
To study the problem, we conduct a comprehensive ablation on the unified tokenizer training paradigm (Figure 3), which yields several intriguing findings: First, we show that removing reconstruction supervision, which leads to a vector-quantized CLIP model, does not improve understanding performance compared to the unified tokenizer. This observation indicates that the performance gap between unified and CLIP tokenizers mainly arises from vector quantization, rather than conflicts between learning objectives; Further analysis reveals that this gap is driven by two key factors: token factorization, which projects tokens into a lower-dimensional space for code index lookup [65], and discretization. These operations are essential for vector quantization but inevitably compromise the expressiveness of visual tokens. We thus argue that the primary bottleneck of unified tokenizers lies in the limited representational capacity of discrete token space.
In light of the issue, we consider expanding the vocabulary size and latent code dimension, which allows for a closer approximation of the continuous feature space. However, extensive studies have shown that doing so could result in low codebook utilization [74,65] and diminishing performance gains [67]. To address this, we introduce multi-codebook quantization to partition the visual token into several chunks, each discretized using a small, separate sub-codebook, akin to the multi-head attention mechanism [54]. This design exponentially scales the vocabulary size with the number of sub-codebooks, while avoiding the optimization problems of large monolithic codebooks. Besides, we replace traditional linear projection layers with adapted attention modules for token factorization, which is observed to consistently improve training stability and understanding performance.
Building upon these techniques, we train a unified tokenizer called UniTok to bridge visual generation and understanding. Through extensive experiments, we demonstrate that UniTok achieves comparable or even better performance to domain-specific tokenizers: On ImageNet evaluation, UniTok records an impressive 0.38 reconstruction FID and 78.6% zero-shot accuracy at 256×256 resolution; In building unified MLLMs, UniTok enables the MLLM with native visual generation capabilities while maintaining decent understanding performance. It outperforms the Liquid [59] baseline with a VQGAN tokenizer by 5.5% on VQAv2 [13], 9.2% on TextVQA [42], and 339 points on MME [64]; In addition, we demonstrate that semantic supervision leads to improved latent space structure for autoregressive generation, i.e., for class-conditional image generation on ImageNet 256×256, UniTok significantly reduces generation FID without classifier-free guidance from 14.6 to 2.5 under the LlamaGen [43] framework, which aligns with recent findings in diffusion modeling [19,4,63].
this section cite: ['b32', 'b11', 'b45', 'b61', 'b72', 'b57', 'b37', 'b70', 'b7', 'b61', 'b60', 'b60', 'b36', 'b57', 'b62', 'b3', 'b18', 'b64', 'b73', 'b64', 'b66', 'b53', 'b58', 'b12', 'b41', 'b63', 'b42', 'b18', 'b3', 'b62']

Section: Related Work
Image Tokenization for Generation. In the domain of visual generation, image tokenization plays an important role in encoding raw pixels into compact latent features for generative modeling [53,39]. Among a variety of tokenizers, the vector-quantized tokenizer [53] is favored for its discrete latent space and compatibility with autoregressive or masked generative models [48,43,3,66]. The pioneering work VQVAE [53] initially introduced the concept of discretizing continuous tokens by mapping them to the nearest neighbors in a learnable codebook. Built on this, VQGAN [8] added perceptual loss [72] and discriminator loss [16] to improve the reconstruction quality. ViT-VQGAN [65] subsequently advanced the framework with the transformer architecture. In recent literature, considerable efforts have been devoted to developing better quantization methods such as residual quantization [18] and lookup-free quantization [67], which also constitute a focal point of this paper.
Image Tokenization for Understanding. The unprecedented success of large language models (LLMs) [57,1,51,47] has catalyzed the development of multimodal large language models (MLLMs) [28,25,31]. As a critical component of MLLMs, the selection of an effective vision tokenizer has been the subject of extensive study [55,49]. A common choice of the vision tokenizer is the pretrained CLIP model [38], which undergoes alignment with language during its pretraining phase. While self-supervised learning models, such as DINOv2 [34], are shown to be advantageous at region-level tasks [30]. However, these tokenizers predominantly encode images into a continuous feature space, presenting challenges for uniformly modeling both vision and text tokens. To address this, some works have explored discretizing CLIP tokens [10] or employing VQVAE encoders [27,62]. Yet, these methods have been observed to substantially impair understanding performance of MLLMs.
this section cite: ['b52', 'b38', 'b52', 'b47', 'b42', 'b2', 'b65', 'b52', 'b7', 'b71', 'b15', 'b64', 'b17', 'b66', 'b56', 'b0', 'b50', 'b46', 'b27', 'b24', 'b30', 'b54', 'b48', 'b37', 'b33', 'b29', 'b9', 'b26', 'b61']

Section: Unified Vision-Language Models.
The rise of MLLMs is not limited to the realm of visual understanding. Recent advancements have witnessed an increasing focus on unifying visual generation and understanding within one MLLM [7,60,46,73,62,50,21]. Specifically, a line of works employs continuous visual tokenizers for image encoding, and leverages pretrained diffusion models for image synthesis [7,11,44]. This approach inevitably increases model complexity and disconnects the visual sampling process from the MLLM. In contrast, another stream of research adopts VQVAE models to encode images into discrete tokens [46,56,62,61,59]. These tokens are subsequently modeled using the same cross-entropy loss that is applied to text tokens, facilitating a unified approach to multimodal learning. However, as reconstruction-oriented VQVAE does not naturally align with the LLM token space, these models typically suffer from degraded visual comprehension capabilities. Our research aligns with the second approach, with a particular focus on the tokenizer design that is suitable for both generation and understanding tasks.
this section cite: ['b6', 'b59', 'b45', 'b72', 'b61', 'b49', 'b20', 'b6', 'b10', 'b43', 'b45', 'b55', 'b61', 'b60', 'b58']

Section: Method
In this section, we introduce UniTok, a unified tokenizer well-suited for both visual generation and understanding tasks. We start with a unified training recipe that integrates reconstruction (VQVAE) and semantic (CLIP) supervisions (Section 3.1). However, we find that simply combining both training objectives leads to severe performance degradation, which can be mainly attributed to limited representational capacity of discrete tokens (Section 3.2). To this end, we propose multi-codebook quantization and attention projection to enhance the latent feature space and derive unified visual representations (Section 3. 3). An overview of the framework is presented in Figure 2.
Contrastive Loss Reconstruction Loss Vision Encoder Vision Decoder Text Encoder Multi-codebook Quantization VQ VQ VQ VQ An oil painting depicting the countryside scenery.
this section cite: []

Section: Continuous token

this section cite: []

Section: Discrete token
Attn. Proj.
Attn. Proj.
this section cite: []

Section: Figure 2:
An overview of UniTok. The tokenizer is trained to reconstruct the input image while aligning its discrete latent features with the text caption. For vector quantization, each visual token is split into multiple chunks, which then undergo code index lookup on corresponding sub-codebooks.
this section cite: []

Section: Unified Supervision
Visual generative and understanding models typically impose distinct demands on the visual tokenizers. For instance, generation emphasizes precise encoding of the visual signals, whereas understanding prioritizes capturing high-level semantics. To accommodate both requirements, we jointly train the tokenizer with (i) a VQVAE-based reconstruction loss to preserve low-level information, and (ii) an image-text contrastive loss that enhances high-level semantics of the features.
To be specific, the VQVAE-based loss term L recon consists of a pixel-level reconstruction loss L R , a perceptual loss L P based on the LPIPS metric [72], a discriminator loss L G to enhance reconstruction fidelity [16], and a vector quantization loss L VQ to minimize distance between the encoder output and its nearest code entry. It is denoted as:
L recon = L R + λ VQ L VQ + λ P L P + λ G L G ,(1)
where λ is the weight factor for the corresponding loss term. The image-text contrastive loss term L contra is basically the same as in CLIP [38]. Therefore, the final loss term can be written as:
L = L recon + λ contra L contra .(2)
We simply choose λ contra = 1 in this paper.
this section cite: ['b71', 'b15', 'b37']

Section: Quantization Bottleneck
Despite being augmented with CLIP supervision, we find that the unified tokenizer exhibits unsatisfactory performance in visual understanding tasks, significantly lagging behind the commonly used CLIP tokenizer. To figure out the underlying cause of this underperformance, we break down the key components involved in training a unified tokenizer, as illustrated in Figure 3. Starting with the CLIP baseline, we provide a step-by-step walk-through of all changes in following paragraphs.
54 56 58 60 62 64 66 68 CLIP Baseline + Factorization + Discretization + Reconstruction +Multi-codebook + Attn. Projection VQA Score VE TE ℒ!"#$ VQ ℒ%&!'( VE TE ℒ!"#$ 768d→8d 8d→768d VE TE ℒ!"#$ 768d→8d 8d→768d VQ VE TE ℒ!"#$ 768d→8d 8d→768d VD ℒ%&!'( MCQ VE TE ℒ!"#$ VD ℒ%&!'( MCQ VE TE ℒ!"#$ VD 768d→64d 64d→768d 768d→64d 64d→768d VE Vision Encoder VD Vision Decoder TE Text Encoder Factorization. Modern VQ-tokenizers typically project continuous tokens to a lower-dimensional latent space for code index lookup (e.g. from 768-d to 8-d), known as token factorization [65]. This increases the relative density of codes by compressing the latent code space, thereby reducing quantization error. To evaluate the impact of factorization in CLIP training, we add two linear projection layers on top of the CLIP vision encoder (right before average pooling), which transforms tokens from 768-d to 16-d and then back to 768-d. Notably, vector quantization and reconstruction supervision are not included at this stage. Surprisingly, it turns out that this channel compression operation significantly compromises the expressiveness of tokens, leading to severe performance degradation in downstream VQA tasks.
this section cite: ['b64']

Section: Discretization.
Based on the implementation described above, we further introduce vector quantization to CLIP training, which maps factorized tokens to their nearest code entries. Compared to language tokenizers with vocabularies exceeding 200k entries, the vocabulary size of modern VQ-tokenizers is markedly smaller (i.e., typically ranging from 4k to 16k). Mapping continuous tokens to such a small codebook results in considerable information loss. This is validated in our experiment, which demonstrates that discretizing the factorized tokens with a 16k codebook causes an average accuracy drop of 2.1 in VQA tasks.
Reconstruction Supervision. Finally, we integrate reconstruction losses into the training process to build a unified tokenizer, as outlined in Section 3.1. Previous literature suggests that loss conflict between VQVAE and CLIP is a major cause of performance degradation in joint training [61]. We observe a similar phenomenon where joint training results in sub-optimal ImageNet zero-shot classification accuracy and reconstruction FID compared to specialized training. However, surprisingly, we find that this degradation has negligible impacts on downstream understanding performance.
Moreover, the degradation in classification accuracy and reconstruction FID diminishes after we improve the quantization methods (detailed in the next section). Based on these observations, we speculate that the perceived loss conflict is only a superficial issue, and the primary cause of the underperformance lies in the limited representational capacity of discrete tokens.
this section cite: ['b60']

Section: UniTok
A straightforward solution to breaking the quantization bottleneck could be increasing the codebook size and the latent code dimension. However, current studies on VQVAE tokenizers suggest that there is diminishing gain in scaling and the performance saturates after the codebook size reaches 16k [67,43]. Continuing expansion results in a substantial portion of codes being rarely used or becoming 'dead' during training, which negatively impacts downstream task performance [65]. To address this, we propose multi-codebook quantization and attention projection in the following paragraphs.
Multi-codebook quantization (MCQ) discretizes the latent tokens with a set of independent codebooks. Specifically, the latent vector f ∈ R d is first evenly split into n chunks {f 1 , f 2 , ..., f n }, where
f i ∈ R d n .
The subsequent quantization process is denoted as:
f = Concat (Q (Z 1 , f 1 ) , Q (Z 2 , f 2 ) , ..., Q (Z n , f n )) (3
)
where f is the discretized latent vector, Q is the code index lookup operation, and Z i is i-th subcodebook. Compared to conventional quantization methods, the proposed MCQ effectively scales up the vocabulary size. For instance, by increasing the number of sub-codebooks from 1 to 4, and suppose each sub-codebook contains 16k code entries, the theoretical vocabulary size exponentially increases from 2 14 to 2 56 (i.e., there are up to 2 14×4 possible combinations of codes for each token). As the size of each individual codebook remains constant, it circumvents the optimization problem associated with large codebooks. Besides, the dimensionality of the latent codes also scales proportionally with the number of codebooks (i.e., increasing from 16-d to 64-d in this case), which further enhances the representational capacity of discrete representations.
Discussions. MCQ shares a similar concept with residual quantization (RQ) [18] in using multiple codes to quantize a token, but differs fundamentally in design philosophy: RQ follows a coarse-tofine quantization order, whereas MCQ adopts a divide-and-conquer strategy. This distinction gives MCQ unique advantages when operating in high-dimensional latent spaces, where codes tend to become increasingly sparse. For instance, with a latent dimension of 64-d, we observe that MCQ's quantization loss is 15 to 45 times lower than that of RQ. This is because MCQ partitions the original latent space into multiple low-dimensional subspaces for quantization. Our ablation study in Table 7 further confirms the superiority of MCQ in unified tokenizer training.
this section cite: ['b66', 'b42', 'b64', 'b17']

Section: Attention projection.
Existing VQ methods usually employ linear or convolutional projection layers for token factorization. But as shown in Figure 3, this over-simplified design fails to preserve rich semantics when compressing the feature dimensions, leading to degraded understanding performance.
To alleviate this problem, we suggest adapting the multi-head attention modules for factorization. Specifically, instead of concatenating features from multiple heads after the attention calculation, we replace the concatenation operation with average pooling to realize channel compression. Figure 6 provides a detailed illustration of the adaptation. Despite its simplicity, we find this design effectively strengthens the representational power of factorized tokens and stabilizes training.
this section cite: []

Section: Unified MLLM
We proceed to develop a unified multimodal model with UniTok. Particularly, we leverage the unified framework introduced in Liquid [59], which models (discrete-valued) vision and language sequences with a universal next-token prediction loss. But instead of learning the visual codebook from scratch, we reuse code embeddings of UniTok by projecting them to the MLLM token space with an MLP projector. Notably, despite UniTok encodes an image into H × W × K codes (where K represents the number of sub-codebooks), we simplify this for MLLM input by merging every K consecutive codes into a single visual token. Similarly, when it comes to visual token prediction, we make each token autoregressively predict the next K codes, using a depth transformer head as implemented in RQ-Transformer [18] and VILA-U [61]. This design maintains efficiency for visual generation in the context of multi-codebooks.
this section cite: ['b58', 'b17', 'b60']

Section: Experiments

this section cite: []

Section: Implementation Details
Tokenizer Setup. Leading VQVAE tokenizers predominantly adopt the CNN architecture, while ViT is preferred in CLIP training for its scalability. To take advantage of both, we choose a hybrid architecture, ViTamin-L/16 [5], to instantiate UniTok. We configure UniTok with eight sub-codebooks, each containing 4,096 code entries and a latent dimension set to 8-d (the global latent dimension is thus 64-d). The discriminator is initialized with pretrained DINOv2-S [34]. We train the tokenizer for one epoch on the public dataset DataComp-1B [9] consisting of 1.28B image-text pairs, with all images resized to 256 × 256 resolution and a global batch size of 16k. The learning rate is set to 1e-3 for the tokenizer and 2e-4 for the discriminator. Besides, we prepare two settings for evaluation: one with pretrained CLIP weight initialization and one with random initialization (the default setting).
this section cite: ['b4', 'b33', 'b8']

Section: MLLM Setup.
We instantiate a unified MLLM described in Section 3.4 with the Llama-2-7B base model [52]. Following Liquid, we first pretrain the model on a mix of multimodal data, which is composed of 10M language data from DCLM [22], 30M internal MidJourney-style synthetic data, and 30M re-captioned image-text pairs from COYO [32] and Laion [41]. Subsequently, we finetune the model on 1.5M text-to-image data and 1.5M multimodal instruction tuning data introduced in Mini-Gemini [23]. Specifically, the learning rate is set to 5e-5 in the pretraining stage and 2e-5 in the finetuning stage. For visual understanding evaluation, we report results on standard VQA benchmarks including VQAv2 [13], GQA [14], TextVQA [42], POPE [24], MME [64], and MM-Vet [70]. For visual generation evaluation, we report results on GenAI-Bench [26] and MJHQ-30K [20].
this section cite: ['b51', 'b21', 'b31', 'b40', 'b22', 'b12', 'b13', 'b41', 'b23', 'b63', 'b69', 'b25', 'b19']

Section: Tokenizer Comparison
Table 1: Comparison on ImageNet reconstruction FID and zero-shot classification accuracy. rFID is measured at 256×256 resolution with 16× downsample ratio. † indicates model using pretrained CLIP weights for initialization. * indicates model trained on OpenImages. Method #Tokens rFID ↓ Accuracy VQVAE Model VQ-GAN * [8] 256 4.98 -RQ-VAE [18] 256 1.30 -VAR * [48] 680 0.90 -UniTok * 256 0.33 -CLIP Model CLIP [38] 256 -76.2 SigLIP [71] 256 -80.5 ViTamin [5] 256 -81.2 Unified Model TokenFlow † [37] 680 1.37 -VILA-U † [61] 256 1.80 73.3 UniTok 256 0.41 70.8 UniTok † 256 0.38 78.6
We benchmark UniTok on ImageNet using two primary metrics: Fréchet Inception Distance (FID) to evaluate reconstruction quality, and top-1 zero-shot accuracy to assess image-text alignment. The results are presented in Table 1. To provide a fair comparison with tokenizers trained on small datasets, we also train a version of UniTok on OpenImages [17] solely with reconstruction supervision. It can be seen that UniTok excels in reconstruction quality compared to both unified and domain-specific tokenizers, recording an impressive 0.38 rFID on ImageNet with 16× downsampling ratio. As a discrete tokenizer, UniTok even surpasses the continuous VAE tokenizer from Stable Diffusion v2.1 [40], showcasing the superiority of the proposed multi-codebook quantization. For the perception performance, we observe that randomly initialized UniTok demonstrates suboptimal zero-shot classification accuracy. This is expected as current training schedule (i.e., one epoch on 1.28B samples) is insufficient for CLIP training to fully converge. It can be seen that initializing the model with pretrained CLIP weights largely alleviates the problem, boosting the zero-shot accuracy from 70.8% to 78.6%. In complement to quantitative results, we provide examples of reconstructed images in Figure 4.
this section cite: ['b16', 'b39']

Section: Class-Conditional Image Generation
Recent studies on diffusion models indicate that injecting semantics into VAE training leads to a betterstructured latent space, considerably enhancing guidance-free generation performance [63,4,19]. To evaluate whether UniTok possesses similar properties, we test it within the LlamaGen framework for class-conditional image generation. As shown in Table 2, UniTok reduces the FID by 12.11 compared to the VQGAN baseline in CFG-free generation, under the same generator setup. This implies that UniTok learns a more structured code distribution, benefiting autoregressive modeling.
this section cite: ['b62', 'b3', 'b18']

Section: Unified Understanding and Generation
Understanding Performance. We evaluate the understanding performance of UniTok on diverse VQA benchmarks in Table 3. Our unified MLLM showcases clear advantages when compared to other unified models that also utilize a discrete visual tokenizer. Specifically, UniTok significantly outperforms the Chameleon model, which relies on a traditional VQVAE tokenizer, by 7.2% higher accuracy on VQAv2. Additionally, it surpasses VILA-U, another model with a unified tokenizer, by 3.3% in accuracy on the TextVQA benchmark and by a notable margin of 112 points on the MME-Perception scores. Furthermore, we can see that UniTok largely narrows the performance gap with MLLMs that incorporate continuous visual tokenizers. These strong results confirm the candidacy of UniTok as a unified visual tokenizer for multimodal models.
Generation Performance. Table 4 presents the text-to-image generation performance of our unified MLLM on the GenEval benchmark. We show that UniTok not only outperforms most of the unified MLLMs, but also demonstrates competitive performance against domain experts (diffusion models) trained on billions of images. Besides, UniTok achieves non-trivial improvements over Liquid while using exactly the same set of text-to-image training data, highlighting the importance of a unified tokenizer. We also provide results on GenAI-Bench in Table 10 and Table 11 in Appendix.
We further evaluate the quality of images generated by our model on the MJHQ-30K benchmark, details of which are presented in Table 5. Notably, as this benchmark primarily relies on the FID score Autoregressive 512 5.47 Janus [58] Autoregressive 384 10.10 LWM [27] Autoregressive 256 17.77 Show-o [62] Discrete Diff. 256 15.18 VILA-U [61] Autoregressive 256 12.81 UniTok Autoregressive 256 7. 46 for evaluation, high-resolution images are preferred because they potentially capture more fine-grained details. Despite this makes FID across different resolutions less comparable, we show that our model achieves impressive performance even at the the smallest resolution, showcasing its ability to generate high-quality, detail-rich images.
We present some examples of the images generated by our model in Figure 5, using text prompts sampled from MJHQ-30K. The visualization results demonstrate our model is capable of synthesizing photorealistic and visually appealing images. Moreover, the model is able to comprehend a wide spectrum of concepts, such as 'Vincent van Gogh painting style' and 'bitcoin', and flexibly combine these concepts to synthesize creative images.
this section cite: ['b57', 'b26', 'b61', 'b60', 'b45']

Section: Ablation Studies
Impact of Supervision Types. To ablate the impact of contrastive and reconstruction losses in UniTok training, we conduct experiments on tokenizers trained with different supervision types, as shown in Table 6. It is worth noting that all the tokenizers are vector-quantized even though some do not have reconstruction supervision. First, we show that reconstruction-oriented tokenizer significantly lags behind tokenizers with contrastive supervision in visual understanding performance. This observation evidences the limitations of traditional VQVAE. Second, we demonstrate that reconstruction and contrastive training objectives do not inherently conflict, or can be addressed by enhancing discrete feature space. With multi-codebook quantization, the jointly trained tokenizer not only exhibits understanding performance on par with the tokenizer trained solely with contrastive loss, but also slightly improves generation performance over the reconstruction-oriented tokenizer.
MCQ v.s. RQ. Following discussions in Section 3.3, we provide an apple-to-apple comparison between multi-codebook quantization and residual quantization in Table 7. For fair comparisons, we
this section cite: []

Section: Number of Sub-Codebooks.
To gain deeper insights into multi-codebook quantization, we evaluate how tokenizer performance changes with the number of sub-codebooks in Table 8. Specifically, the size of a codebook is denoted as A × B, where A is the number of sub-codebook and B is the size of sub-codebook. For rFID evaluation, we train the tokenizer solely with reconstruction loss on OpenImages [17], and evaluated it on ImageNet (256 × 256) validation set. While for ImageNet zero-shot accuracy evaluation, the tokenizer is trained on DataComp-1B 128m subset using only contrastive loss. Given a constant global codebook size, we see that increasing the number of subcodebooks consistently improves reconstruction FID and classification accuracy. This indicates that MCQ generally benefits vector-quantized models, independent of the training objectives.
this section cite: ['b16']

Section: CLIP Weight Initialization.
We notice that higher ImageNet accuracy does not guarantee superior downstream performance. In Table 9, we ablate the impact of CLIP weight initialization on visual understanding performance. Specifically, we adopt the classic LLaVA framework for evaluation, replacing the original CLIP tokenizer with UniTok while keeping all other the training settings unchanged. One tokenizer is initialized with the pretrained ViTamin-L-256 [5] weights, while the other is randomly initialized. To our surprise, UniTok that is trained from scratch surpasses the one initialized with pretrained CLIP weights, despite the latter actually achieves better zeroshot classification accuracy. This suggests downstream VQA performance may not be highly correlated with ImageNet classification accuracy. More importantly, it also implies that CLIP weight initialization may serve as a negative prior for unified tokenizers, as the unified visual feature space could drastically differ from CLIP feature space.
this section cite: ['b4']

Section: Limitations and Conclusion
This paper studies unified visual tokenization for generation and understanding, which serves as the cornerstone of unified multimodal large language models. We investigate the training paradigm of unified tokenizers and identify that the current challenge in unification mainly arises from the limited representational power of discrete tokens. To address this limitation, we introduce multi-codebook quantization and attention projection to build a unified tokenizer called UniTok. We show that UniTok excels in downstream visual generation and understanding tasks. The ablation study further reveals that discriminative and generative representation learning does not inherently conflict. We hope our findings could inspire future research in this domain.
However, due to limited computational resources, UniTok is only trained for one epoch, which is not sufficient for CLIP-based semantic representation learning. We believe extending the training schedule could further benefit the tokenizer, especially in understanding performance.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Improving image generation with better captions Year: (2023)
Ref_id:b2 Title: Masked generative image transformer Year: (2022)
Ref_id:b3 Title: Masked autoencoders are effective tokenizers for diffusion models Year: (2025)
Ref_id:b4 Title: Vitamin: Designing scalable vision models in the vision-language era Year: (2024)
Ref_id:b5 Title: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b6 Title: Synergistic multimodal comprehension and creation Year: (2023)
Ref_id:b7 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b8 Title: Datacomp: In search of the next generation of multimodal datasets Year: (2024)
Ref_id:b9 Title: Planting a seed of vision in large language model Year: (2023)
Ref_id:b10 Title: Seed-x: Multimodal models with unified multi-granularity comprehension and generation Year: (2024)
Ref_id:b11 Title: Experiment with gemini 2.0 flash native image generation Year: (2025)
Ref_id:b12 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b13 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b14 Title: Unified language-vision pretraining with dynamic discrete visual tokenization Year: (2023)
Ref_id:b15 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b16 Title: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale Year: (2020)
Ref_id:b17 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b18 Title: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers Year: (2025)
Ref_id:b19 Title: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation Year: (2024)
Ref_id:b20 Title: Synergen-vl: Towards synergistic image understanding and generation with vision experts and token folding Year: (2024)
Ref_id:b21 Title: Datacomp-lm: In search of the next generation of training sets for language models Year: (2024)
Ref_id:b22 Title: Mini-gemini: Mining the potential of multi-modality vision language models Year: (2024)
Ref_id:b23 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b24 Title: On pre-training for visual language models Year: (2024)
Ref_id:b25 Title: Evaluating text-to-visual generation with image-to-text generation Year: (2025)
Ref_id:b26 Title: World model on million-length video and language with ringattention Year: (2024)
Ref_id:b27 Title: Visual instruction tuning Year: (2024)
Ref_id:b28 Title: Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action Year: (2024)
Ref_id:b29 Title: Groma: Localized visual tokenization for grounding multimodal large language models Year: (2025)
Ref_id:b30 Title: Mm1: Methods, analysis & insights from multimodal llm pre-training Year: (2024)
Ref_id:b31 Title: Saehoon. Coyo-700m: Image-text pair dataset Year: (2022)
Ref_id:b32 Title: Introducing 4o image generation Year: (2025)
Ref_id:b33 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b34 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b35 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b36 Title: Tokenflow: Unified image tokenizer for multimodal understanding and generation Year: (2024)
Ref_id:b37 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b38 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b39 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b40 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b41 Title: Towards vqa models that can read Year: (2019)
Ref_id:b42 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b43 Title: Generative multimodal models are in-context learners Year: (2024)
Ref_id:b44 Title: Generative pretraining in multimodality Year: (2023)
Ref_id:b45 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b46 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b47 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b48 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b49 Title: Metamorph: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b50 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b51 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b52 Title: Neural discrete representation learning Year: (2017)
Ref_id:b53 Title: Attention is all you need Year: (2017)
Ref_id:b54 Title: What makes for good visual tokenizers for large language models? arXiv preprint Year: (2023)
Ref_id:b55 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b56 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b57 Title: Janus: Decoupling visual encoding for unified multimodal understanding and generation Year: (2024)
Ref_id:b58 Title: Liquid: Language models are scalable and unified multi-modal generators Year: (2024)
Ref_id:b59 Title: Next-gpt: Any-to-any multimodal llm Year: (2023)
Ref_id:b60 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b61 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b62 Title: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models Year: (2025)
Ref_id:b63 Title: A survey on multimodal large language models Year: (2023)
Ref_id:b64 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b65 Title: Magvit: Masked generative video transformer Year: (2023)
Ref_id:b66 Title: Language model beats diffusion-tokenizer is key to visual generation Year: (2023)
Ref_id:b67 Title: Scaling autoregressive multi-modal models: Pretraining and instruction tuning Year: ()
Ref_id:b68 Title: An image is worth 32 tokens for reconstruction and generation Year: (2024)
Ref_id:b69 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b70 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b71 Title: The unreasonable effectiveness of deep features as a perceptual metric Year: (2018)
Ref_id:b72 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
Ref_id:b73 Title: Scaling the codebook size of vqgan to 100,000 with a utilization rate of 99% Year: (2024)
