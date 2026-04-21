Title: LaViDa: A Large Diffusion Language Model for Multimodal Understanding
Abstract: Modern Vision-Language Models (VLMs) can solve a wide range of tasks requiring visual reasoning. In real-world scenarios, desirable properties for VLMs include fast inference and controllable generation (e.g., constraining outputs to adhere to a desired format). However, existing autoregressive (AR) VLMs like LLaVA struggle in these aspects. Discrete diffusion models (DMs) offer a promising alternative, enabling parallel decoding for faster inference and bidirectional context for controllable generation through text-infilling. While effective in language-only settings, DMs' potential for multimodal tasks is underexplored. We introduce LaViDa, a family of VLMs built on DMs. We build LaViDa by equipping DMs with a vision encoder and jointly fine-tune the combined parts for multimodal instruction following. To address challenges encountered, LaViDa incorporates novel techniques such as complementary masking for effective training, prefix KV cache for efficient inference, and timestep shifting for high-quality sampling. Experiments show that LaViDa achieves competitive or superior performance to AR VLMs on multi-modal benchmarks such as MMMU, while offering unique advantages of DMs, including flexible speed-quality tradeoff, controllability, and bidirectional reasoning. On COCO captioning, LaViDa surpasses Open-LLaVa-Next-Llama3-8B by +4.1 CIDEr with 1.92× speedup. On bidirectional tasks, it achieves +59% improvement on Constrained Poem Completion. These results demonstrate LaViDa as a strong alternative to AR VLMs. Code and models is

Section: Introduction
Vision-Language Models (VLMs) have shown remarkable utility across diverse domains, from enduser applications like virtual assistants [37], to research tasks such as scientific image captioning and document understanding [46,16,5,1,48]. In industrial settings, VLMs support automated product tagging, content moderation, and quality control in manufacturing [69,2]. Their ability to jointly process visual and textual information makes them indispensable for practical applications and cutting-edge research. Currently, nearly all popular VLMs-such as Qwen-VL [5,75], Intern-VL [17,85], and GPT-4 [58] are built on top of large language models (LLMs) that generate text in an autoregressive (AR) manner; that is, they produce tokens one by one in a left-to-right sequence.
While these models have demonstrated strong performance on many tasks, they suffer from several key limitations. First, their sequential generation process is inherently hard to parallelize, resulting in slow inference speed [8]. More critically, their left-to-right generation makes it difficult to handle tasks that benefit from bidirectional context or structural constraints-such as text infilling [22]. For example, generating a poem where each line starts with a specific syllable, or extracting structured information from an image in a predefined JSON format, often requires the model to fill in or coordinate content across the sequence. Autoregressive models still struggle to consistently satisfy such constraints, even with carefully crafted prompts and examples.
Recently, discrete diffusion models (DMs) have emerged as a promising alternative to AR LLMs. Most notably, LLaDA [57] and Dream [79] achieved comparable results to AR LLMs across diverse language tasks. Unlike AR LLMs, DMs treat text generation as a diffusion process over discrete tokens. A forward process gradually corrupts a sequence of discrete text tokens into a sequence of mask tokens. At inference, we start with a sequence of mask tokens and gradually transform them into a sequence of meaningful text tokens through a learned reverse process.
Compared to AR LLMs, diffusion models offer several theoretical advantages that directly address the limitations of autoregressive generation. While AR LLMs have a fixed throughput-generating one token at a time-DMs allow flexible control over the speed-quality trade-off by adjusting the number of diffusion steps [79,64,49,57]. Moreover, their ability to model bidirectional context makes them well-suited for tasks like text infilling, enabling more effective constrained generation and structured output formatting-capabilities especially valuable in vision-language settings where outputs may need to follow specific schemas.
In this work, we propose LaViDa (Large Vision-Language Diffusion Model with Masking), the first family of VLMs based on diffusion. LaViDa enables pretrained DMs to perceive visual inputs by integrating vision features into the diffusion backbone via a vision encoder-analogous to how LLaVA [46,44] augments large language models (LLMs) with visual inputs. Specifically, we adopt a two-stage training pipeline with a diffusion objective: pretraining followed by supervised fine-tuning.
Adapting DMs for vision-language tasks presents several practical challenges. First, standard DM training is data-inefficient: the model learns only from the subset of corrupted tokens at each timestep. For example, in a question-answering task where the target answer is "The answer is dog", the corruption process may mask "The [mask] is dog", leaving the key token "dog" unmasked and thus excluded from the loss. This is especially problematic for multimodal tasks, where answer tokens often carry crucial semantic content grounded in the image [61]. To address this, we introduce a complementary masking scheme that ensures every token in the output sequence contributes to learning, improving data efficiency.
Second, inference algorithms used by existing DMs are slow in practice due to the lack of KV cache support-an inherent limitation of bidirectional context modeling [3]. This leads to repeated recomputation over the full prompt at every decoding step. While tolerable for short-text settings, it becomes a significant bottleneck in vision-language tasks, where multimodal prompts may include hundreds of visual tokens. To address this, we propose Prefix-DLM decoding, a simple yet effective technique that enables caching of multimodal prompts (i.e., image and question tokens), significantly accelerating inference.
Lastly, DMs offer a unique advantage over autoregressive models: the ability to trade off speed and quality by varying the number of diffusion steps. However, the widely used linear masking schedule-which unmasks a fixed number of tokens per step-performs poorly at low step counts.
Motivated by text-to-image diffusion models like SD3 [21], we adopt a timestep shifting strategy that adaptively adjusts how many tokens are unmasked per step. This leads to better sample quality under aggressive step reduction, allowing faster generation without large degradation in quality.
We conducted extensive evaluations of LaViDa across a wide range of vision-language tasks. Results show that LaViDa achieves competitive performance on most benchmarks, including MMMU [80], MathVista [51], ChartQA [53] and ScienceQA [52], when compared with AR VLMs like LLaVa-1.6-7B [45,43] and Open-LLaVa-Next-Llama3-8B [15]. We highlight these results in Figure 1. On constrained generation tasks, LaViDa greatly outperforms AR baselines (+59% on Poem Completion). It also supports flexible speed-quality tradeoff, achieving higher quality on COCO image captioning (+ 4.1 CIDEr) and a 1.92× speedup [42]. In summary, our contributions are:
• We introduce LaViDa, the first family of VLMs based on diffusion models. Our models achieve competitive performance across a wide range of tasks compared to AR VLMs, while offering the unique benefits of DMs.
• We introduce several novel training and inference techniques for DMs, including complementary masking, Prefix-DLM, and timestep shifting that improve the training efficiency, inference speed, and sample quality of LaViDa.
• We conduct a systematic study of various design choices for adapting DMs to visionlanguage tasks (e.g. image resolution), offering insights for future work in this direction.
2 Background and Related Works
this section cite: ['b36', 'b45', 'b15', 'b4', 'b0', 'b47', 'b68', 'b1', 'b4', 'b74', 'b16', 'b84', 'b57', 'b7', 'b21', 'b56', 'b78', 'b78', 'b63', 'b48', 'b56', 'b45', 'b43', 'b60', 'b2', 'b20', 'b79', 'b50', 'b52', 'b51', 'b44', 'b42', 'b14', 'b41']

Section: Vision-Language Models
Vision-Language Models (VLM) extend the capability of Large Language Models to visual understanding tasks [36,46,75,5,9,85,58]. The common recipe to build a VLM is to start with a strong large language model and combine it with a vision encoder [46,75]. These VLMs typically undergo multiple stages of training, which can be generally categorized into pretraining and finetuning stages. The pretraining data usually consists of text-image pairs for vision-language alignment, while the finetuning data consists of a wide range of instruction-following tasks. There are several dedicated lines of work focusing on different components of this overarching framework, such as the design of vision encoders [26,77] and training techniques [41,72]. To this date, most vision-language models such as LLaVa [46,36] and Qwen-VL [5,75] series employ an autoregressive training objective.
this section cite: ['b35', 'b45', 'b74', 'b4', 'b8', 'b84', 'b57', 'b45', 'b74', 'b25', 'b76', 'b40', 'b71', 'b45', 'b35', 'b4', 'b74']

Section: Diffusion Language Models
Diffusion models first emerged as a powerful alternative to GANs for generating continuous data such as images [62,59,21]. Early explorations in diffusion language models directly built continuous diffusion models for latent text embeddings [39,50], with limited success. More recently, discrete diffusion models [4,64,49] have proven to be better candidates for language modeling, achieving performance comparable to AR models while offering unique advantages, such as speed-quality tradeoffs and controllability. Most notably, LLaDa-8B and Dream-8B [57,79] demonstrated that DLMs can achieve competitive performance against AR LLMs at scale.
Formally, given a text sequence of L tokens
X 0 = [X 1 0 , X 2 0 ...X L 0 ], the forward process q(X t |X s ) gradually converts it to a sequence full of mask tokens "[M]", denoted by X 1 = [X 1 1 , X 2 1 ...X L 1 ]
, through the continuous time-interval [0, 1], with 1 ≥ t ≥ s ≥ 0. A neural network p θ is used to model the reverse process p(X s |X t ). The diffusion language modeling objective can be defined as:
L DLM = -E t,X0,Xt 1 t log p θ (X 0 |X t )(1)
where log p θ (X 0 |X t ) is assumed to be factorized into
L i=1 p θ (X i 0 |X t ).
At each training step, we sample t uniformly from the interval [0, 1] and sample X 0 from some data distribution D. X t is then sampled from the forward process q(X t |X 0 ). The loss is only computed over the masked tokens in X t , since p θ (X i 0 |X t ) has a closed form representation that does not depend on θ when X i t ̸ = [M ]. We offer additional background on the details of these formulations in Appendix A.1. We also incorporate addition backgrounds on other relevant literature, such as masked generative models [10,11] and multimodal diffusion models [68,38], in Appendix C.
this section cite: ['b61', 'b58', 'b20', 'b38', 'b49', 'b3', 'b63', 'b48', 'b56', 'b78', 'b9', 'b10', 'b67', 'b37']

Section: Crop & Resize

this section cite: []

Section: Diffusion Language Model
What is in the image [?] There is a [M] dog on the [M] [M] [PAD] [M] Prompt P Noised Answer X Prediction X lovely grass [.] [PAD] 1 2 3 4 5 Input Image I Multiple Views I1 ,I2 …I5 Vision Encoder Concat & Flatten MLP Vision Embeddings V1 ,V2 …V5 [Projected Vision Embeddings]
this section cite: []

Section: Method

this section cite: []

Section: Model Architecture
LaViDa's model architecture follows a similar design to common AR VLMs like LLaVa [36]. It consists of a vision encoder and a diffusion language model. These two parts are connected by a MLP projection network. The overall design is illustrated in Figure 2.
this section cite: ['b35']

Section: Vision Encoder.
Given an input image I and text prompt P , we first resize the image to 768 2 and divide it into four non-overlapping views of 384 2 , denoted I 1:4 . We also resize the original image to 384 2 to obtain a fifth view, I 5 , following the design of prior works [36,45]. These five views are independently encoded by the vision encoder (SigLIP-400M [82]), each producing 27 2 embeddings, denoted V 1:5 . In total, this yields 3645 embeddings per image. To reduce sequence length for efficient training, we apply 2 × 2 average pooling on each view, reducing embeddings to 14 2 per view, or 980 total. The embeddings of five views are then flattened and concatenated into a 1D sequence before being processed by the projection network to obtain the final visual context of the diffusion language model. This process mirrors the vision encoding process in AR LLMs [26] and is illustrated in the bottom part of Figure 2.
this section cite: ['b35', 'b44', 'b81', 'b25']

Section: Diffusion Language Model.
The diffusion language model is a multi-layer Transformer [71] whose architecture resembles that of LLMs. The only major difference is that its attention mask is noncausal, and it uses the diffusion language modeling objective described in Section 2.2 instead of the next-token prediction used in AR models. The input to the diffusion language model consists of the projected vision embeddings, the prompt P , and partially masked response X t . The outputs of the last transformer block are passed through a final linear layer to obtain token-wise logits p θ (X i 0 |I, P, X t ) for the unmasked response X 0 . In our experiments, we explored LLaDA-8B (default) and Dream-7B as our diffusion language model. This process is illustrated in the upper half of Figure 2.
this section cite: ['b70']

Section: Training Algorithms
Our training objective is based on the diffusion language modeling objective described in Section 2.2. Each training sample consists of an image I, text prompt P and clean text answer X 0 from the training data. For multi-round conversation, we sample one round as the "answer" and treat the history as "prompt". We first sample a timestep t ∈ [0, 1] and a partially masked answer X t using the forward process described in Section 2.2. LaViDa then implements the conditioned reverse process p θ (X 0 |I, P, X t ). The canonical diffusion vision-language modeling objective is formulated as:
L D-VLM = -E t,I,P,X0,Xt 1 t log p θ (X 0 |I, P, X t )(2)
There is a dog on the [PAD] DLM There is a [M] dog on the [M] [M] [PAD] [M] lovely grass [.] [PAD] [M] [M] [M] lovely [M] [M] [M] grass [M] [M] [PAD] DLM Loss Loss [ Vision and Prompt Embeddings] [ Vision and Prompt Embeddings] Copy (a) Complementary Masking where p θ (X 0 |I, P, X t ) factorizes into L i=1 p θ (X i 0 |I, P, X t ) following the formulation in Section 2.2. Notably, the loss is only computed over mask tokens where
X i t = [M ], because p θ (X i 0 |I, P, X t ) is not dependent on θ when X i t ̸ = [M ]
. Complementary Masking. Prior diffusion language models (e.g., LLaDa, Dream) apply a stochastic estimator to Equation 2, masking tokens independently across samples in a batch. However, for visionlanguage tasks, this leads to inefficiencies: (1) only ∼50% of tokens contribute to the loss on average, and (2) critical answer tokens-often short and sparse in vision tasks like VQA-may not be masked, resulting in misaligned gradients for the vision encoder. For example, in "The answer is dog." the key token "dog" might be unmasked in X t and thus ignored in loss computation. To address this, we introduce complementary masking: for each sample, we generate two masked versions X t and X C t with disjoint corrupted spans (e.g., one masks "The [M] [M] dog .", the other "[M] answer is [M] [M]", ensuring all tokens are eventually used in training and improving sample efficiency and gradient flow. When computing the loss over X t and X C t , we copy the encoded vision embeddings to further boost training efficiency. This process is illustrated in Figure 3a.
this section cite: []

Section: Inference Algorithms
At inference time, we first create a sequence of L mask tokens as X 1 , where L is the response generation length. Then we gradually unmask them through K discrete timestamps t 1 ..t K , where t 1 = 1 and t K = 0, until we reach a clean, mask-free sequence X 0 . At each timestamp t i , we sample a fully unmasked sequence through p θ (X 0 |X ti ) and re-mask L × t i+1 tokens to obtain X ti+1 . Both L and K are hyperparameters for inference. Additionally, we define K L as "fraction of the number of functional evaluations (NFE)" to measure sample efficiency. For example, when NFE = 100%, the diffusion model generates one token per forward pass; at NFE = 50%, it generates an average of two tokens per forward pass. Overall, the inference process of LaViDa is similar to prior DMs such as LLaDa, with two key exceptions: Prefix-DLM. While DLMs theoretically offer superior speed-quality tradeoffs at inference, they are often slower than AR models in practice because they cannot leverage KV caching [57]. This issue is particularly evident for multimodal prompts containing many visual tokens. To avoid recomputing keys and values for the visual embeddings and text prompts, we propose a novel Prefix-DLM scheme inspired by the autoregressive prefix-LM. Prefix-DLM adopts a specialized attention mask in which visual and prompt tokens can only attend to other visual and prompt tokens, while answer tokens can attend to all tokens. Figure 3b illustrates this setup. With this design, we can cache the keys and values of the visual and prompt tokens. Empirically, this leads to a speedup of up to 3.9× on COCO captioning tasks. Further details are provided in Section 4.5. Schedule Shift. Diffusion language models (DLMs) allow trading speed for quality via the number of discretization steps K. Prior models like LLaDa and Dream use a linear schedule, unmasking L K tokens uniformly over t ∈ [0, 1]. However, we find this leads to performance degradation at low sampling steps. Inspired by SD3 [21], we adopt a shifting schedule:
t ′ i = s α (t i ) = αt i 1 + (α -1)t i(3)
Here, s α (t) is a monotonic map with t 0 = t ′ 0 = 0, t K = t ′ K = 1. When α < 1 (we use α = 1 3 ), the schedule is convex-leading to more tokens being unmasked earlier. We found that this setup outperforms alternatives. Notably, this conclusion differs from that of continuous diffusion models like SD3 and previous masked diffusion models for image generation [11], which showed concave schedules (α > 1) are more preferable. We ensure at least one token is unmasked per step. Further details are provided in Section 4 and Appendix A.2.
this section cite: ['b56', 'b20', 'b10']

Section: Experiments

this section cite: []

Section: Setup
At a high level, LaViDa employs a two-stage training process. In the pretraining phase (stage-1), only the projector is updated to align the visual embeddings with the latent space of the DLM. In the finetuning phase (stage-2), we jointly train all components end-to-end for instruction-following. Additionally, we further finetune the stage-2 model for additional steps and obtain two specialized models for reasoning and text-infilling tasks (LaViDa-Reason and LaViDa-FIM). We provide more details of these specialized models in Section 4.3 and 4.4. We use 558K image-text pairs as our stage-1 data, and 1M visual instruction-following examples as our stage-2 data. Further details on the dataset and training setup are provided in Appendix B.
We evaluate LaViDa on a wide range of vision-language tasks. Unless otherwise stated, we report results obtained using the stage-2 model with LLaDa-8B as the language backbone. We use lmms-eval package [83] for evaluation and set the sequence length L to be the maximum generation length used for evaluating AR models. We set K = L, or NFE=100% by default. Results under differet NFE are explored in Section 4.5 and Appendix B.3 and B.4.
this section cite: ['b82']

Section: Main Results
Table 1 reports the results of LaViDa using LLaDA-8B (LaViDa-L) and Dream-7B (LaViDa-D) as the language backbones on vision-understanding tasks. We compare with several open-source, open-data models with similar data sizes and parameter counts: LLaVa-1.6-7B [45,43] and Open-LLaVa-Next-Llama3-8B [15]. We also include comparisons with frontier open-sourced models of similar size that are trained on larger datasets, namely LLaVa-OneVision-7B [36], Qwen2.5-VL-7B [5], and InternVL-38B [85].
LaViDa demonstrates competitive performance across a wide range of tasks spanning General, Reasoning, OCR, and Science categories. In general vision-language understanding, LaViDa-L achieves the highest score on MMMU [80] (43.3), outperforming all comparable models. LaViDa-D also ranks second on several benchmarks in this category. For reasoning tasks, both models surpass similarly scaled baselines on math-heavy and spatially grounded benchmarks. In Science, LaViDa achieves the best and second-best scores on ScienceQA [52] (81.4 and 80.2, respectively) while performing on par with Open-Llava-Next on AI2D [32], a complex diagram-based benchmark. Finally, in OCR LaViDa shows competitive performance but lags behind some of the latest AR models. This gap is primarily due to our use of average pooling for vision token compression, which leads to the loss of fine-grained spatial information. While this was a necessary trade-off given our limited compute budget, it poses challenges for tasks requiring precise text recognition and layout understanding. These results highlight the strength of LaViDa, demonstrating that diffusion-based approaches can scale competitively with AR models while achieving robust performance across a wide range of vision-language tasks.
this section cite: ['b44', 'b42', 'b14', 'b35', 'b4', 'b84', 'b79', 'b51', 'b31']

Section: Reasoning Distillation
Prior work has distilled LLMs [70] and VLMs [18,73] using long chain-of-thought (CoT) data to elicit strong reasoning capabilities [25]. In the same spirit, we study the reasoning abilities of LaViDa by conducting additional stage-3 training using 19.2K CoT examples distilled from VL-Rethinker-7B, a strong reasoning model. We refer to the finetuned model as LaViDa-Reason. We evaluate it on MathVista [51], MathVerse [84], and MathVision [74] with CoT generation, comparing against the stage-2 results without CoT. We set the maximum generation length L = 1024 for these tasks. We report these results in Table 2a. We find that LaViDa-Reason outperforms LaViDa across all benchmarks, with the most significant gains observed on the most challenging MathVision reasoning dataset (+18% relative improvement). Further details are provided in Appendix B.3.
Write a poem with 4 lines.
this section cite: ['b69', 'b17', 'b72', 'b24', 'b50', 'b83', 'b73']

Section: A [M] [M] … [FIM] Be [M] [M] … [FIM] Ca [M] [M] … [FIM] De [M] [M] … [FIM]
A torii gate stands tall, symbol of good fortune, Behold the sacred shrine, a place of good fortune. Cafton hangs above, a symbol of good fortune, Deities watch over us, a place of good fortune.
this section cite: []

Section: Draft Answer
A red torii gate stands tall, Beige pillars reach for the sky, Caesars flutter in the breeze, Deities watch over the world below LaViDa LaViDa -FIM Write a poem with 4 lines, starting with A Be Ca De respectively.
A torii stands tall and bright, Beneath the sky's vast expanse.
Cresting the path, it beckons, Dawn whispers in the breeze.
this section cite: []

Section: Text Infilling
LaViDa offers strong controllability for text generation, particularly in text infilling. Given a draft of L tokens containing L M masks, we jump to timestep t = L M L and run standard inference to reach t = 0. This directly replaces L M masks with L M tokens. However, in practice, the intended completion may require fewer tokens-e.g., "There is a [M][M][M][M] in the image" might become either "dog" or "traffic light". To allow variable-length completions, we conduct an additional stage-3 training using a 20% subset of stage-2 data and refer to this model as LaViDa-FIM. During training, we insert random-length [S]...[S][FIM] sequences mid-text. At inference, we append [FIM] to masked segments (e.g., [M][M][M][M][FIM]) to signal flexible termination.
this section cite: []

Section: The model can then generate completions like [dog][S][S][S][FIM] or [traffic][light][S][S][FIM].
While FIM objectives are often discussed in the context of language tasks (e.g., code completion) [63,7], they are equally relevant to multimodal applications. Figure 4a shows qualitative results on constrained poem generation, where the models generate a poem describing an image, with each line starting with specific syllables. Both LaViDa and LaViDa-FIM complete the task successfully, unlike AR models. Notably, LaViDa-FIM adapts token counts per line. Table 2b shows quantitative results over 100 samples: both LaViDa variants achieve 100% constraint satisfaction, while AR baselines remain below 50%. Additional results on other text-infilling use cases are provided in Appendix B.2.
this section cite: ['b62', 'b6']

Section: Speed vs. Quality Trade Off
LaViDa offers a convenient way to achieve speed-quality tradeoffs by controlling the number of discretization steps K. We compare the performance on image captioning with 500 images from the COCO 2017 val dataset [42] with varying K. We set the maximum generation length to 32, and experimented with K ∈ {32, 24, 16, 8}, or equivalently, NFE∈ {100%, 75%, 50%, 25%}. We report the average latency per image measured on a single A5000 GPU and the CIDEr score in Figure 4b. At NFE=100%, LaViDa achieves a higher CIDEr score than AR baselines but is slightly slower. At NFE=75% and NFE=50%, LaViDa is faster than the AR baselines and achieves better quality. At NFE=25%, it is significantly faster but trails in performance. This indicates that LaViDa can flexibly adjust its inference speed based on application needs-allowing users to trade off generation latency and output quality depending on their specific requirements.
Effect of KV Cache. The speed of LaViDa relies on our proposed Prefix-DLM setup, which allows us to cache the keys and values of visual and prompt tokens [3]. In Table 3a, we compare the speed and sample quality between the proposed Prefix-DLM step and the uncached full-attention-mask step used in prior works like LLaDa. We find that Prefix-DLM significantly reduces latency and achieves a maximum speedup of 3.9×, with marginal performance cost. These experiments are performed using our stage-2 model, which is trained on a full-attention mask. We discuss training with customized masks and customized kernels in Appendix B.4. In short, we found that these alternatives lead to considerable training overhead while offering little benefit. Noise Schedule. To study the effect of the proposed time-step shifting schedule, we compare the performance of different schedules with NFE∈ {100%, 75%, 50%, 25%}. We compare the proposed time-step shifting with α = 3, and α = 3 -1 , as well as linear and cosine schedule baselines. We report results on COCO image captioning [42] in Table 3b. The convex schedule with α = 3 -1 works the best. We also observe similar behaviors when conducting CoT inference using LaViDa-Reason on MathVision [74] dataset. At NFE=50%, α = 3 -1 achieves an accuracy of 21.05, which is 30% higher than 16.12 achieved by a linear schedule. We provide results on MathVision in Appendix B.3.
this section cite: ['b41', 'b2', 'b41', 'b73']

Section: Ablation Studies
We conducted a thorough ablation of various design choices. In the main paper, we discuss the effect of complementary masking and input image resolution. We provide further discussion of other design choices in the Appendix B.5. We conducted these experiments using a 200k subset of our (stage-2) training data. We report results in Tables 4a and 4b respectively. Table 4a shows that our proposed complementary masking scheme leads to considerable improvements across all benchmarks, most notably, complementary masking leads to a relative improvement of 67% on ScienceQA [52] with affordable compute overhead during the training (8% slowdown). Table 4b shows that high-resolution input improves overall performance, with the gain on OCR tasks being more pronounced than generic vision tasks (e.g., VQAv2). We did not use average pooling for the low-resolution setup. We provide additional details in the Appendix B.
this section cite: ['b51']

Section: Table 4: Ablation Studies.
We study the effect of (a) complementary masking and (b) image resolution. For (a), we also report the wall clock time of 1000 training steps with a batch size of 128.
(a) Effect of complementary masking. w/o Comp.M. w/ Comp.M. MME↑ 260.00 297.00 MathVista↑ 28.40 33.40 ScienceQA↑ 48.74 81.49 MMMU↑ 38.56 41.78 Runtime↓ 8.2 hr 8.9 hr (b) Effect of image resolution. 384 2 768 2 TextVQA↑ 48.40 55.65 DocVQA↑ 43.22 58.72 ChartQA↑ 42.20 57.70 InfoVQA↑ 26.48 36.23 VQAv2↑ 65.92 66.78
this section cite: []

Section: Conclusion
In conclusion, we propose LaViDa, the first family of vision-language models based on DMs. To address various challenges, we introduce several novel training and inference techniques, including complementary masking, Prefix-DLM cache, and timestep shifting. Through extensive experiments, we show that these techniques significantly outperform a naive adaptation of DMs for visual tasks.
Using a comprehensive evaluation suite, we demonstrate that LaViDa achieves competitive performance compared to AR models trained under similar settings, while offering unique advantages such as speed-quality tradeoffs and controllability via text infilling. Our work proves that LaViDa can be a powerful alternative to exitsing AR VLMs, extending the prior success of DMs in the language domain to the vision space.
this section cite: []

Section: References
Ref_id:b0 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b1 Title:  Year: ()
Ref_id:b2 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b3 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b4 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b5 Title: Smaller, weaker, yet better: Training llm reasoners via compute-optimal sampling Year: (2024)
Ref_id:b6 Title: Efficient training of language models to fill in the middle Year: (2022)
Ref_id:b7 Title: Scheduled sampling for sequence prediction with recurrent neural networks Year: (2015)
Ref_id:b8 Title: Level up your tutorials: Vlms for game tutorials quality assessment Year: (2024)
Ref_id:b9 Title: Text-to-image generation via masked generative transformers Year: (2023)
Ref_id:b10 Title: Masked generative image transformer Year: (2022)
Ref_id:b11 Title: Allava: Harnessing gpt4vsynthesized data for a lite vision-language model Year: (2024)
Ref_id:b12 Title: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning Year: (2021)
Ref_id:b13 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2023)
Ref_id:b14 Title: Open-llava-next: An open-source implementation of llava-next series for facilitating the large multi-modal model community Year: (2024)
Ref_id:b15 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b16 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b17 Title: Openvlthinker: An early exploration to complex vision-language reasoning via iterative selfimprovement Year: (2025)
Ref_id:b18 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b19 Title: Flex attention: A programming model for generating optimized attention kernels Year: (2024)
Ref_id:b20 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b21 Title: Incoder: A generative model for code infilling and synthesis Year: (2022)
Ref_id:b22 Title: A comprehensive evaluation benchmark for multimodal large language models Year: (2023)
Ref_id:b23 Title: Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering Year: (2017)
Ref_id:b24 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b25 Title: Llava-uhd: an lmm perceiving any aspect ratio and highresolution images Year: (2024)
Ref_id:b26 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b27 Title: Ruler: What's the real context size of your long-context language models Year: (2024)
Ref_id:b28 Title: Unified discrete diffusion for simultaneous vision-language generation Year: (2022)
Ref_id:b29 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b30 Title: Dvqa: Understanding data visualizations via question answering Year: (2018)
Ref_id:b31 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b32 Title: Ocr-free document understanding transformer Year: ()
Ref_id:b33 Title: Segment anything Year: (2023)
Ref_id:b34 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2017)
Ref_id:b35 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b36 Title: Multimodal foundation models: From specialists to general-purpose assistants Year: (2024)
Ref_id:b37 Title: Omniflow: Any-to-any generation with multi-modal rectified flows Year: (2024)
Ref_id:b38 Title: Diffusion-lm improves controllable text generation Year: (2022)
Ref_id:b39 Title: Open-ended long text generation via masked language modeling Year: (2023)
Ref_id:b40 Title: On pre-training for visual language models Year: (2024)
Ref_id:b41 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b42 Title:  Year: (2023)
Ref_id:b43 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b44 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024-01)
Ref_id:b45 Title: Visual instruction tuning Year: (2023)
Ref_id:b46 Title: Mmbench: Is your multi-modal model an all-around player? Year: (2024)
Ref_id:b47 Title: Efficient frontier visual language models Year: (2024)
Ref_id:b48 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b49 Title: Latent diffusion for language generation Year: (2023)
Ref_id:b50 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2023)
Ref_id:b51 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: ()
Ref_id:b52 Title: ChartQA: A benchmark for question answering about charts with visual and logical reasoning Year: (2022-05)
Ref_id:b53 Title: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Year: (2022)
Ref_id:b54 Title: Docvqa: A dataset for vqa on document images Year: (2021)
Ref_id:b55 Title: Ocr-vqa: Visual question answering by reading text in images Year: (2019)
Ref_id:b56 Title: Large language diffusion models Year: (2025)
Ref_id:b57 Title: Gpt-4o system card Year: (2024)
Ref_id:b58 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b59 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b60 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b61 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b62 Title: Code llama: Open foundation models for code Year: (2023)
Ref_id:b63 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b64 Title: Towards vqa models that can read Year: (2019)
Ref_id:b65 Title: Unified multimodal discrete diffusion Year: (2025)
Ref_id:b66 Title: Improved artgan for conditional synthesis of natural image and artwork Year: (2019)
Ref_id:b67 Title: Any-to-any generation via composable diffusion Year: (2023)
Ref_id:b68 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b69 Title: OpenThoughts Team. Open Thoughts Year: (2025-01)
Ref_id:b70 Title: Attention is all you need Year: (2017)
Ref_id:b71 Title: Q-vlm: Post-training quantization for large vision-language models Year: (2024)
Ref_id:b72 Title: Vlrethinker: Incentivizing self-reflection of vision-language models with reinforcement learning Year: (2025)
Ref_id:b73 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b74 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b75 Title:  Year: (2023)
Ref_id:b76 Title: Pvc: Progressive visual token compression for unified image and video processing in large vision-language models Year: (2024)
Ref_id:b77 Title: Multimodal large diffusion language models Year: (2025)
Ref_id:b78 Title:  Year: (2025)
Ref_id:b79 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b80 Title: Star: Bootstrapping reasoning with reasoning Year: (2022)
Ref_id:b81 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b82 Title: Lmms-eval: Reality check on the evaluation of large multimodal models Year: (2024)
Ref_id:b83 Title: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint Year: (2024)
Ref_id:b84 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
Ref_id:b85 Title: Masked audio generation using a single non-autoregressive transformer Year: (2024)
