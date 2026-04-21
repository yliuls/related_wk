Title: FUDOKI: Discrete Flow-based Unified Understanding and Generation via Kinetic-Optimal Velocities
Abstract: The rapid progress of large language models (LLMs) has catalyzed the emergence of multimodal large language models (MLLMs) that unify visual understanding and image generation within a single framework. However, most existing MLLMs rely on autoregressive (AR) architectures, which impose inherent limitations on future development, such as the raster-scan order in image generation and restricted reasoning abilities in causal context modeling. In this work, we challenge the dominance of AR-based approaches by introducing FUDOKI, a unified multimodal model purely based on discrete flow matching, as an alternative to conventional AR paradigms. By leveraging metric-induced probability paths with kinetic optimal velocities, our framework goes beyond the previous masking-based corruption process, enabling iterative refinement with self-correction capability and richer bidirectional context integration during generation. To mitigate the high cost of training from scratch, we initialize FUDOKI from pre-trained AR-based MLLMs and adaptively transition to the discrete flow matching paradigm. Experimental results show that FUDOKI achieves performance comparable to state-of-the-art AR-based MLLMs across both visual understanding and image generation tasks, highlighting its potential as a foundation for next-generation unified multimodal models. Furthermore, we show that applying test-time scaling techniques to FUDOKI yields significant performance gains, further underscoring its promise for future enhancement through reinforcement learning.

Section: Introduction
Driven by the rapid progress of large language models (LLMs) [1][2][3][4][5], a new wave of large-scale multimodal models has emerged, delivering remarkable advances in the two fundamental pillars of artificial general intelligence (AGI): understanding [6][7][8][9][10] and generation [11][12][13][14][15]. Building on this momentum, a growing body of work [16][17][18][19][20][21][22][23][24][25] seeks to unify perception and synthesis within a single framework, introducing versatile multimodal large language models (MLLMs) that seamlessly integrate visual understanding with image generation. 3   In prior research, most MLLMs adopt the autoregressive (AR) architecture of standard LLMs, processing multimodal tokens sequentially from left to right for both understanding and generation tasks [26,27]. While these MLLMs deliver strong performance across many multimodal tasks, their inherent AR design's limitations have become increasingly apparent as shown in recent studies, such as weaker performance in complex reasoning [28][29][30], challenges in future planning [31], and difficulties with self-correction [32]. These shortcomings are particularly critical for emerging domains such as embodied AI and autonomous agents, where complex reasoning and deep contextual understanding are essential. This prompts a fundamental question for the future of AGI development: what architectural paradigm could define the next generation of MLLMs?
To this end, discrete-space generative flow and diffusion models have gained attention as a promising alternative for generative modeling. These models have seen success in the domain of text generation [33][34][35][36][37][38], protein design [39], image synthesis [37,38], and code generation [37,40]. Unlike sequential autoregressive models, these models usually begin with a fully corrupted sequence and iteratively denoise the entire sequence in parallel, which allows richer integration of information from both directions to enhance prolonged reasoning. Moreover, these models enable flexible and controllable generation through their inherent iterative refinement process, while offering the potential for accelerated sampling via novel training designs [41][42][43]. Recent studies like LLaDA [44] and Dream [45] have also scaled discrete diffusion models to 7B parameters, further highlighting their growing potential to overcome the fundamental limitations of autoregressive approaches.
To advance the application of discrete generative flow modeling and challenge the dominance of the AR-based paradigm in MLLMs, we present FUDOKI, a unified multimodal model purely based on discrete flow matching. Different from previous diffusion-based unified multimodal models [46][47][48] focusing solely on the case of masking as a corruption process, we adopt the novel framework of discrete flow matching [37,38], which substantially expanded the design space of discrete-space generative models by enabling metric-induced probability paths with kinetic optimal velocities. This design enables better performance than masked construction [38] and allows models to continuously self-correct their responses during the iterative refinement process. Moreover, to mitigate the high training cost of training large discrete flow matching models for multimodal tasks, we leverage the pre-trained AR-based MLLM [20] as the initialization and adaptively transfer it to the discrete flow matching paradigm [49].
The contributions of this paper can be summarized as follows: 1) We introduce FUDOKI 4 , the first general-purpose unified multimodal model built entirely on discrete flow matching. Unlike traditional approaches that rely on masking-based corruption, FUDOKI leverages a metric-induced probability path with kinetically optimal velocities, expanding the design space of discrete multimodal modeling and offering advantages during inference; 2) Through extensive experiments, we show that FUDOKI achieves competitive performance on both visual understanding and text-to-image generation tasks, rivaling autoregressive-based MLLMs; 3) We apply test-time inference scaling techniques to FUDOKI inspired by [50], which yield substantial improvements across visual generation and understanding benchmarks. This suggests strong potential for future enhancement of FUDOKI via reinforcement learning [1,51]. We believe that FUDOKI provides a compelling foundation for the development of next-generation unified multimodal models.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b35', 'b36', 'b35', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b35', 'b36', 'b36', 'b18', 'b47', 'b48', 'b0', 'b49']

Section: Preliminary: Discrete Flow Matching
In this section, we present key concepts and notations in discrete flow matching [37] to facilitate understanding in the following sections. Generally speaking, the objective of discrete flow matching is to approximate the target underlying data distribution q(x) from the source known distribution p(x), where x = (x 1 , x 2 , ..., x D ) belongs to the discrete space S = T D , where D is the number of discrete variables and T = [K] = {1, 2, . . . , K} represents a finite set of possible discrete values.
Probability Paths. Given a source distribution p(x) and a target distribution q(x) defined over a finite state space S, discrete flow matching defines a family of time-indexed probability distributions {p t (x)} t∈[0,1] to describe a smooth transformation from p to q, referred to as probability paths. Each p t (x) is constructed as: p t (x) := x1∈S p t (x | x 1 )q(x 1 ), where the conditional distribution is factorized across dimensions, namely p t (x | x 1 ) := D i=1 p t (x i | x i 1 ). Here, each p t (x i | x i 1 ) defines a univariate interpolation between a base distribution p(x i ) and a point mass δ x i 1 (x i ), i.e., δ x i 1 (x i ) = 1 if x i = x i 1 else 0. A common design for such interpolations is the mixture path, defined via a time-dependent scheduler κ t (x i 1 ) ∈ [0, 1]:
p t (x i | x i 1 ) = (1 -κ t (x i 1 ))p(x i ) + κ t (x i 1 )δ x i 1 (x i ),(1)
where κ 0 (•) = 0 and κ 1 (•) = 1. This class of paths recovers the masked data construction when p(x i ) = δ m (x i ) with m denoting the mask token, which are widely used in previous studies [35,36].
this section cite: ['b35', 'b33', 'b34']

Section: Probability Velocities.
To simulate the generative process that evolves along the prescribed path {p t (x)} t∈[0,1] , we consider a continuous-time Markov chain (CTMC) {x t } t∈[0,1] over the discrete space S, such that: x t ∼ p t . Specifically, we describe this CTMC via a probability velocity u i t (•, x t )
(also known as the rate matrix), describing the rate of probability change of x t in its i-th token. Reminiscent of the velocity field in the continuous Flow Matching [42,41], discrete flow matching features the following definition: Definition 1. A probability velocity u t is said to generate the probability path p t if, for all t ∈ [0, 1) and for any sample x t ∼ p t , the updated sample x i t+h ∼ δ x i t (•) + hu i t (•, x t ) for each coordinate i satisfies the condition that x t+h ∼ p t+h + o(h) 5 as h → 0. Besides, the probability velocity u t should satisfy the following rate condition:
x i ∈[K] u i t (x i , z) = 0, and u i t (x i , z) ≥ 0 ∀i ∈ [D], x i ̸ = z i ,(2)
such that the updated x i t+h can be sampled from a valid probability distribution. Further, previous studies [37,39] also demonstrate the Continuity Equation (also known as the Kolmogorov forward equation) in discrete flow matching, which describes the state probability rate ṗt (x), x ∈ S by:
ṗt (x) + div x (p t u t ) = 0.(3)
where
div x (p t u t ) = z∈S D i=1 δ x (z i ) p t (x)u i t (z i , x) -p t (z)u i t (x i , z)
, measuring the total outgoing flux x → z minus the total incoming flux z → x for state x ∈ S. Here δ x (z i ) = j̸ =i δ x j (z j ), which indicates that we only consider x and z when they only differ in the i-th coordinate for calculating the flux [37,34]. Intuitively, Eq. 3 expresses that the rate of probability at x is equal to the final remaining probability flux p t u t at x. Previous studies [37,39] have shown that if the Continuity Equation is satisfied, then u t is said to generate the probability path p t as in Definition 1.
this section cite: ['b40', 'b39', 'b35', 'b37', 'b35', 'b32', 'b35', 'b37']

Section: FUDOKI: A Multimodal Model Purely Based on Discrete Flow Matching
This section introduces FUDOKI, a new multimodal architecture that unifies vision and language through the novel lens of discrete flow matching. By adopting this framework, FUDOKI enables an integrated approach to both perception and generation across visual and textual modalities.
this section cite: []

Section: Metric-induced Probability Paths with Kinetic Optimal Velocities
Based on the recent theoretical advancement of discrete flow matching [38], we adopt a more general probability path for FUDOKI, instead of the commonly used mask-based mixture paths [37,36,35,46,45]. Specifically, we consider the probability paths induced by discrete metrics. Given a distance function d : T × T → R ≥0 satisfying d(x i , x i 1 ) = 0 if and only if x i = x i 1 , we define a path of conditional distributions via:
p t (x i | x i 1 ) = softmax -β t • d(x i , x i 1 ) ,(4)
where β t : [0, 1] → R ≥0 is a monotonic schedule with boundary values β 0 = 0, β 1 = ∞. At t = 0, this yields a uniform distribution, and as t → 1, the distribution converges to a delta function at x i 1 . Compared to the previous mask-based probability path (i.e., Eq. 1), this metric-induced probability path defines a more semantically meaningful transformation, allowing the probabilities of tokens similar to x i 1 to also increase as t → 1, when setting d(•, •) to measure token embedding distances. After defining the prescribed metric-induced probability path, we then obtain the probability velocities via minimizing the kinetic energy [38]. In other words, it is expected to minimize the magnitude of flux p t u t for probability velocities to obtain a smooth transformation along the probability path. Meanwhile, the obtained velocities should also satisfy several conditions, including the Continuity Equation (i.e., Eq. 3), the non-negativity of the flux between different states (i.e., Eq. 2), and the boundary conditions for p and q. We leave the detailed mathematical formulations in the appendix. In this way, the kinetic optimal velocity for Eq. 4 can be formulated as follows [38],
u i t (x i , z | x 1 ) = p t (x i | x i 1 ) βt [d(z i , x i 1 ) -d(x i , x i 1 )] +(5)
where [•] + = max{•, 0} is the ReLU operator and βt is the derivative of β t w.r.t t. Intuitively, for the i-th coordinate z i ∈ T , this velocity ensures that probability mass flows from state z i to state x i only when x i lies closer to x i 1 than z i does, i.e., d(x i , x i 1 ) < d(z i , x i 1 ). As a result, the flow monotonically progresses toward x i 1 . After introducing the mathematical foundation of discrete flow matching, we now dive into FUDOKI's model structure details.
this section cite: ['b36', 'b35', 'b34', 'b33', 'b44', 'b43', 'b36', 'b36']

Section: Bi-directional Transformer

this section cite: []

Section: Causal Transformer

this section cite: []

Section: Bi-directional Transformer
(a) Autoregressive (AR): Janus, Janus-Pro, Illume, Illume+, ViLA-U (2025); Chameleon, EMU3, LaViT (2024) (d) Discrete Diffusion: UniDisc (2024) (e) Discrete Flow Matching: FUDOKI Partially Causal Transformer (b) AR (Text) + Diffusion (Image) JanusFlow (2025); Transfusion, Show-o (2024) Bi-directional Transformer (c) Continuous + Discrete Diffusion: D-DiT (2025) … Text Tokenizer
this section cite: []

Section: What is it?

this section cite: []

Section: Text De-Tokenizer

this section cite: []

Section: A red panda.

this section cite: []

Section: Text Tokenizer
A red panda.
this section cite: []

Section: Image Decoder
Image Encoder
this section cite: []

Section: Text Tokenizer
What is it?
this section cite: []

Section: Text De-Tokenizer

this section cite: []

Section: A red panda.

this section cite: []

Section: Text Tokenizer
A red panda.
this section cite: []

Section: Image Decoder Image Encoder

this section cite: []

Section: Continuous Discrete
Image Encoder
Text Tokenizer
this section cite: []

Section: Text Tokenizer
What is it? <mask> <mask> … <mask> <mask>
this section cite: []

Section: Text De-Tokenizer

this section cite: []

Section: A red panda.
A red panda.
this section cite: []

Section: Image Decoder
Text Tokenizer
this section cite: []

Section: Text Tokenizer
A red panda.
Text Tokenizer Image Decoder pré tion 学 … ment 的 ist Text Tokenizer What is it? Image Encoder (Semantic) Text Tokenizer Text De-Tokenizer A red panda.
Text Tokenizer A red panda. Image Encoder (Pixel) Pixel Decoder (Pixel) Image Encoder <mask> <mask> … <mask> <mask> Text De-Tokenizer A red panda. Img2Text Text2Img Img2Text Text2Img Img2Text Text2Img Text2Img Img2Text Img2Text Text2Img Continuous Discrete Discrete Figure 2: Comparison of Model Architectures in Unified Multimodal Models. (a) AR-based models [20, 26, 21, 52-54, 18, 55] perform multimodal tasks via sequential token generation under strictly causal context modeling. (b) Hybrid AR+Diffusion models, such as Transfusion [19] and Show-o [56], integrate AR for text and diffusion models for images, enabling improved visual generation quality. (c-d) Diffusion-based models: D-DiT [46] applies mask-based discrete diffusion to text and continuous diffusion to images, while UniDisc [48] employs mask-based discrete diffusion for both modalities. (e) FUDOKI adopts a unified discrete flow matching framework for both modalities, leveraging a metric-induced probability path to enhance performance in understanding and generation tasks. The inference advantages of FUDOKI over mask-based discrete diffusion modeling used in (c-d) are shown in Fig. 3.
this section cite: []

Section: Architecture Overview
As shown in Fig. 2(e), FUDOKI is based on the Janus-1.5B [20] architecture, with minor adaptations to support unified vision-language discrete flow modeling. Specifically, to facilitate effective learning and accelerate convergence, 1) we adopt a full attention mask instead of the standard causal mask to allow all tokens to attend to each other, which helps the model better capture global context; 2) we apply a shifting operation [49] to the output logits by one position, so that our model can inherit the next-token prediction capabilities of AR-based MLLMs as much as possible; 3) unlike continuous diffusion models [57,12], we do not incorporate additional time embedding layers in the model to explicitly indicate the noise level in the corrupted input. Following the intuition of mask-based discrete diffusion models [49,58], we observe that our discrete generative model can also implicitly infer the timesteps from the corrupted input along our defined metric-induced probability path (i.e., Eq. 4), resulting in faster adaptation in experiments. The rest of the architecture remains identical to Janus-1.5B. For the text modality, we use the tokenizer with a vocabulary size of 102, 400. For images, we decouple the processing paths for understanding and generation. The semantic encoder SigLIP [59] extracts high-dimensional features for image understanding, which are reshaped and mapped into the LLM input space via an adaptor. For image generation, we follow LlamaGen [60], employing a pixel encoder and decoder to convert images into discrete tokens, with the image token vocabulary size set to 16, 384. Each image token embedding is further transformed into an input
Current State (𝑥𝑡) Predicted final state (𝑥 1 𝑝𝑟𝑒𝑑 ) Next State (𝑥𝑡+ℎ) Bi-directional Transformer Image Encoder Text Encoder This of panda Mah . image … … This image panda … 在 red 在 . Jump Token Selection This panda : Jump : No Jump : Always Updatable (b) Discrete Flow Matching (DFM): Fudoki (a) Mask-based Discrete Diffusion (MDD): D-DiT, UniDisc Current State (𝑥𝑡) Predicted final state (𝑥 1 𝑝𝑟𝑒𝑑 ) Next State (𝑥𝑡+ℎ) Bi-directional Transformer Image Encoder Text Encoder This <mask> panda <mask> . image … … This image panda … <mask> red <mask> .
this section cite: ['b18', 'b47', 'b55', 'b11', 'b47', 'b56', 'b57', 'b58']

Section: Jump Token Selection
This panda : Jump : No Jump : Not Considered : Fixed after Unmasking Update 𝑥𝑡 with 𝑥𝑡+ℎ Denoise 𝑥𝑡 to 𝑥𝑡+ℎ Denoise 𝑥𝑡 to 𝑥𝑡+ℎ Update 𝑥𝑡 with 𝑥𝑡+ℎ Flow Matching-Based FUDOKI. In mask-based discrete diffusion models, once a token is unmasked, it typically cannot be modified again, which hinders self-correction. In contrast, our proposed FUDOKI allows its responses to be continuously updated during inference, enabling potential corrections.
feature via a generation adaptor before being fed into the LLM. At the output stage, we use two output heads, a text head and an image head, which convert the transformer outputs into discrete categorical distributions. The appropriate head is selected depending on the target modality during inference. Comparisons with previous AR-based and diffusion-based MLLMs are shown in Fig. 2.
this section cite: []

Section: Training
We follow the discrete flow matching framework [34] for model training. Our model is initialized from the pretrained weights of Janus-1.5B [20] and further adapted to our collected dataset, which contains both text-to-image (generation) and image-to-text (understanding) data. Specifically, we divide the training of FUDOKI into two stages: 1) The main goal of the first stage is to quickly relearn the AR-based LLM such that it can effortlessly support the discrete flow matching paradigm. To this end, we only fine-tune the parameters of the transformer while keeping other parts of the model frozen, including the semantic encoders and embedding adaptors. This can help accelerate convergence and stabilize our training; 2) After the first stage, we further fine-tune the whole model to enhance its overall performance on understanding and generation based on discrete flow matching.
Specifically, in each training stage, the ground-truth target x 1 is drawn from the data distribution q(•), where the condition is either a text prompt (for T2I) or an image-question pair (for I2T). The target x 1 is the image token sequence in the T2I setting and the textual token sequence in the I2T setting. At each training step, a time t ∈ [0, 1] is uniformly sampled, and a noised sequence x t is sampled according to the defined probability path p t (• | x 1 ) in Eq. 4. We set the distance function d(•, •) to measure the L2-distances between normalized token embeddings, which helps increase the probability of sampling tokens whose embeddings are close to the corresponding ground-truth token x i 1 in the embedding space, thereby making the corruption process more semantically meaningful and facilitating learning. The model then receives x t as input and predicts x 1 , outputting per-token logits for each position. The training loss is defined as the expected cross-entropy between the ground-truth sequence x 1 and the model's predicted distribution:
L CE (θ) = E t∼U [0,1], x1∼q(•), xt∼pt(•|x1) - D i=1 log p θ 1|t x i 1 | x t (6
)
where p θ 1|t (• | x t ) denotes the model's predicted categorical distribution for the i-th position, parameterized by θ, given input x t .
this section cite: ['b32', 'b18']

Section: Inference
During inference, we apply an Euler solver for more robust sampling as suggested in [38]. This solver simulates the continuous-time Markov chain (CTMC) process (x t ) 0≤t≤1 . Given that x t ∼ p t , the solver updates the i-th coordinate from time t to t + h using the following procedure:
• Sample x i 1 ∼ p i 1|t (•|x t ) from our model; • Compute the total conditional transition rate λ i = x i ̸ =x i t u i t (x i , x i t |x i 1 ) (see Eq. 5); • Draw a uniform random variable Z i change ∼ U [0, 1]; • Sample x i t+h as follows: if Z i change ≤ 1 -e -hλ i , sample x i t+h from u i t (•,x i t |x i 1 ) λ i (1 -δ x i t (•)); otherwise set x i t+h = x i t . Here δ x i t (•) is a delta function.
We provide a detailed understanding of this inference process as follows. In the second step, λ i can be interpreted as the intensity with which the probability mass at x i t flows to other states x i ̸ = x i t . The probability that x i t will change at the current timestep is determined by comparing the threshold 1 -e -hλ i with a uniform random variable Z i change : the larger λ i is, the more likely a jump will occur. If a change happens, x i t+h is sampled from all other possible states according to the distribution proportional to u i t (•, x i t |x i 1 ), as defined in Eq. 5. This means the update tends to move x i t+h towards states that are closer to the model's prediction x i 1 . In this way, our sampling process enables the model to: (1) continuously refine its predictions along the probability path, and (2) flexibly adjust tokens towards semantically similar alternatives at each timestep. As shown in Fig. 3, this is in contrast to previous mask-based discrete diffusion models [36,35,45], where once a token is unmasked, it generally cannot be modified again, even if it contains an error.
this section cite: ['b36', 'b34', 'b33', 'b43']

Section: Experiments

this section cite: []

Section: Implementation Details
In both training stages, we use approximately 13M supervised finetuning data to learn our FUDOKI, including 9M in-house generation data for text-to-image generation and 4M public understanding data, which covers various aspects including OCR [61,62], doc [63], chart [64], screen [65], math [66,67], language [68], etc. This is less than Chameleon's 1.4B data [54] and LWM's 1B data [69]. We leave the detailed dataset collections in the appendix. For text generation, the sequence length for the response is set to 500, while for image generation, it is set to 576 to match the input size of the image encoder. The text embeddings for calculating the metric distance function d(•, •) are taken from the original embedding layer of Janus-Pro-7B [26] and the image embeddings are obtained from the codebook of LlamaGen [60]. We set β t = c t 1-t a with c = 3 and a = 0.9, as suggested in [38]. Besides, following previous studies [45,44], for the text modality, we pad each sequence with <eos> (end-of-sequence) and <pad> tokens to the maximum length during training, and compute the loss over model's answer tokens, including these special tokens. After the sampling process, we only keep the model responses ahead of the first <eos> token. The sampling iterations are set as 32 by default, and the resolution of generated images by FUDOKI is 384 × 384. The entire training process spanned approximately 43,000 GPU hours.
this section cite: ['b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b52', 'b67', 'b24', 'b58', 'b36', 'b43', 'b42']

Section: Comparison with State-of-the-arts
Visual Generation Performance. We evaluate the generation capabilities of FUDOKI on the widely used GenEval benchmark [75]. Table 1 presents the summarized comparisons, where FUDOKI achieved competitive overall performance (0.77), matching the top score of prior models in the category of both the generation-only and the understanding-and-generation categories. These results underscore our model's advantages in accurate multi-object understanding and attribute binding, making it promising for complex visual generation tasks that go beyond simple object depiction. This can be attributed to the discrete flow matching framework of FUDOKI, which allows visual information to integrate in both directions for better layout design of generated images.
Besides, we evaluate the visual generation performance of FUDOKI on DPG-Bench [76] (Dense Prompt Graph Benchmark), a comprehensive dataset comprising 1,065 lengthy and densely composed prompts specifically designed to assess the fine-grained semantic alignment capabilities of text-toimage models. As shown in Table 2, FUDOKI demonstrates competitive performance compared to both generation-specialized and unified multimodal models. These results highlight FUDOKI's strong ability to handle complex, information-rich prompts, establishing it as a robust and versatile solution for multi-aspect visual generation tasks.
this section cite: ['b73', 'b74']

Section: Multimodal Understanding.
We evaluate the understanding capabilities of FUDOKI on several benchmarks, including POPE [91], MME-P [92], SEED [93], MMB [94], GQA [95], MMMU [96],  and MM-Vet [97]. Table 3 presents the summarized results 6 . Notably, our FUDOKI model (1.5B parameters) achieved highly competitive results, which are on par with or surpass several AR-based MLLMs of similar or even larger scale. This demonstrates that FUDOKI delivered robust multimodal understanding capabilities, which can be attributed to the bidirectional reasoning property of discrete flow matching. Moreover, we provide generation process comparisons for understanding in Fig. 4, which further highlight the advantages of sampling through discrete flow matching for reasoning, e.g., self-correcting the reasoning process for coherency. Our findings highlight the effectiveness and efficiency of FUDOKI, making it a strong alternative to the established AR-based MLLMs.
Inference Scaling. We applied test-time inference scaling techniques [50] to FUDOKI, leveraging a judge model to score multiple candidate outputs and select the highest-scoring responses. The last rows of Table 1 and Table 3 illustrate the impact of inference scaling on visual generation and understanding. For generation, we used the VILA-Judge model [98] to select the top 4 images from 32 candidates per prompt in the GenEval benchmark, resulting in significant performance gains. For understanding, we employed an LLM as the judge to choose the best response from 8 candidates in the challenging MMVet benchmark, where improvements were observed. These results highlight FUDOKI's potential for further enhancement through reinforcement learning approaches [1,99].  The last equation in the table is 10 + a = 12. To find the value of y, we substitute a = 12 into the equation:
The last equation in the table is 10 + a = 12. To find the value of y, we substitute a = 12 into the equation: 10 + 12 = 12 + a 12 + 12 = 24
The last equation in the table is 10 + a = 12. To find the value of y, we substitute a = 12 into the equation: 10 + 12 = 12 + a 12 + 12 = 24 So, the value of y is 24. <mask><mask><mask><mask><mas k><mask><mask><mask><mask><m ask><mask><mask><mask><mask> <mask><mask><mask><mask><mas k><mask><mask><mask><mask> <mask> the the <mask><mask> to numbers in<mask>last<mask>, <mask><mask><mask> the overall<mask><mask><mask><mask>, a six<mask><mask>
In the the business<mask>to numbers in the last answer, the famous<mask> the overall problem<mask><mask>da six-<mask><mask> Question: What is y in the last equation?
Ground Truth: 5.
this section cite: ['b89', 'b90', 'b91', 'b92', 'b93', 'b94', 'b95', 'b48', 'b96', 'b0', 'b97']

Section: t=0 t=1
In the the business-to numbers in the last answer, the famousthe overall problem is d,a six-Strategy.
this section cite: []

Section: Ablation Studies
Training Strategies. 1) AR Initialization vs Training from Scratch: As shown in Fig. 5 (left), we compare models initialized with autoregressive (AR) weights [20] against models trained from scratch.
The results indicate that AR initialization provided a substantial advantage for accelerating model training, leading to consistently lower training loss throughout the optimization process. 2) Effects of Time-embedding Layers: We also evaluate the impact of incorporating time embedding layers into the model architecture. The results in Fig. 5 (middle) show that the model without time embedding layers consistently achieves slightly lower training loss than the version with time embeddings. This  suggests that our discrete generative model can implicitly infer timesteps from corrupted input, and removing time embeddings reduces model complexity.
Quality-Speed Trade-off. Results on the Self-Correction Capability. We quantitatively evaluated the self-correcting capabilities of FUDOKI and performed comparisons with the AR-based models. In experiments, both FUDOKI and AR-based models were tasked with correcting baseline responses where necessary. The baseline responses were obtained from Janus-Pro-1B on the MMVet benchmark, using the Open-Compass VLMEvalKit codebase [100]. To assess their correction abilities: 1) For AR-based models, we appended the following prompt to the original prompt: "Your original response is: <placeholder>. Please correct it if needed. Otherwise, you may keep it the same." The models were then evaluated on their ability to revise or retain the response as appropriate; 2) For FUDOKI, we initialized the responses with the baseline responses (rather than uniformly-sampled noise tokens) and performed iterative refinements over 32 steps, as described in the paper. As shown in Table 4, FUDOKI achieved the highest performance improvement, while Janus-Pro-1B's performance declined and Janus-Pro-7B showed less increase, despite its larger model size than ours. We attribute such results to the increased context length introduced by the baseline responses, which may distract the AR-based model's focus. This further highlights the limitations of the AR paradigm for effective self-correction.
this section cite: ['b18', 'b98']

Section: Conclusion
In this work, we introduced FUDOKI, a multimodal model that uses discrete flow matching to unify visual understanding and generation. Unlike conventional autoregressive and masking-based approaches, FUDOKI leverages discrete flow matching for iterative self-correction, bidirectional reasoning, and flexible generation. Experiments show that FUDOKI performs competitively with leading AR-based MLLMs on both visual understanding and text-to-image generation tasks. These results highlight discrete generative flow models-exemplified by FUDOKI-as a promising direction for advancing multimodal language models and meeting future AGI challenges.
this section cite: []

Section: References
Ref_id:b0 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b1 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b2 Title: The llama 3 herd of models Year: (2024)
Ref_id:b3 Title:  Year: (2024)
Ref_id:b4 Title:  Year: (2023)
Ref_id:b5 Title: Visual instruction tuning Year: (2024)
Ref_id:b6 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b7 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b8 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b9 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b10 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b11 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b12 Title: Pixart-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis Year: (2023)
Ref_id:b13 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b14 Title: Planting a seed of vision in large language model Year: (2023)
Ref_id:b15 Title: Making llama see and draw with seed tokenizer Year: (2023)
Ref_id:b16 Title:  Year: (2024)
Ref_id:b17 Title: Transfusion: Predict the next token and diffuse images with one multi-modal model Year: (2024)
Ref_id:b18 Title: Janus: Decoupling visual encoding for unified multimodal understanding and generation Year: (2024)
Ref_id:b19 Title: Illume: Illuminating your llms to see, draw, and self-enhance Year: (2024)
Ref_id:b20 Title: Muse-vl: Modeling unified vlm through semantic discrete encoding Year: (2024)
Ref_id:b21 Title: Lmfusion: Adapting pretrained language models for multimodal generation Year: (2024)
Ref_id:b22 Title: Omnimamba: Efficient and unified multimodal understanding and generation via state space models Year: (2025)
Ref_id:b23 Title: Emerging properties in unified multimodal pretraining Year: (2025)
Ref_id:b24 Title: Janus-pro: Unified multimodal understanding and generation with data and model scaling Year: (2025)
Ref_id:b25 Title: Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement Year: (2025)
Ref_id:b26 Title: Sparks of artificial general intelligence: Early experiments with gpt-4 Year: (2023)
Ref_id:b27 Title:  Year: (2023)
Ref_id:b28 Title: The pitfalls of next-token prediction Year: (2024)
Ref_id:b29 Title: Beyond autoregression: Discrete diffusion for complex reasoning and planning Year: (2024)
Ref_id:b30 Title: Large language models cannot self-correct reasoning yet Year: (2023)
Ref_id:b31 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b32 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2024)
Ref_id:b33 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b34 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b35 Title: Discrete flow matching Year: (2024)
Ref_id:b36 Title: Flow matching with general discrete paths: A kinetic-optimal perspective Year: (2025)
Ref_id:b37 Title: Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design Year: (2024)
Ref_id:b38 Title: Mercury coder Year: (2025)
Ref_id:b39 Title: Flow straight and fast: Learning to generate and transfer data with rectified flow Year: (2022)
Ref_id:b40 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b41 Title: Consistency models Year: (2023)
Ref_id:b42 Title: Large language diffusion models Year: (2025)
Ref_id:b43 Title:  Year: (2025)
Ref_id:b44 Title: Dual diffusion for unified image generation and understanding Year: (2024)
Ref_id:b45 Title: Unified discrete diffusion for simultaneous vision-language generation Year: (2022)
Ref_id:b46 Title: Unified multimodal discrete diffusion Year: (2025)
Ref_id:b47 Title: Scaling diffusion language models via adaptation from autoregressive models Year: (2024)
Ref_id:b48 Title: Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer Year: (2025)
Ref_id:b49 Title: Weilin Huang, and Ping Luo. Dancegrpo: Unleashing grpo on visual generation Year: (2025)
Ref_id:b50 Title: Illume+: Illuminating unified mllm with dual visual tokenization and diffusion refinement Year: (2025)
Ref_id:b51 Title: VILA-u: a unified foundation model integrating visual understanding and generation Year: (2025)
Ref_id:b52 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b53 Title: Unified language-vision pretraining in LLM with dynamic discrete visual tokenization Year: (2024)
Ref_id:b54 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b55 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b56 Title: Diffusionbert: Improving generative masked language models with diffusion models Year: (2023)
Ref_id:b57 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b58 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b59 Title: Enhanced visual instruction tuning for text-rich image understanding Year: (2023)
Ref_id:b60 Title: wendlerc/renderedtext Year: (2023)
Ref_id:b61 Title: Docvqa: A dataset for vqa on document images Year: (2021)
Ref_id:b62 Title: Chart-to-text: Generating natural language descriptions for charts by adapting the transformer model Year: (2020)
Ref_id:b63 Title: Visualmrc: Machine reading comprehension on document images Year: (2021)
Ref_id:b64 Title: G-llava: Solving geometric problem with multi-modal large language model Year: (2023)
Ref_id:b65 Title: Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning Year: (2022)
Ref_id:b66 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b67 Title: World model on million-length video and language with blockwise ringattention Year: (2025)
Ref_id:b68 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b69 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b70 Title: Improving image generation with better captions Year: (2023)
Ref_id:b71 Title: Seed-x: Multimodal models with unified multi-granularity comprehension and generation Year: (2024)
Ref_id:b72 Title: Tokenflow: Unified image tokenizer for multimodal understanding and generation Year: (2024)
Ref_id:b73 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2024)
Ref_id:b74 Title: Ella: Equip diffusion models with llm for enhanced semantic alignment Year: (2024)
Ref_id:b75 Title: Lumina-Next: Making Lumina-T2X stronger and faster with Next-DiT Year: (2024)
Ref_id:b76 Title: Playground v2. 5: Three insights towards enhancing aesthetic quality in text-to-image generation Year: (2024)
Ref_id:b77 Title: Hunyuan-DiT: A powerful multiresolution diffusion transformer with fine-grained chinese understanding Year: (2024)
Ref_id:b78 Title: Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation Year: (2024)
Ref_id:b79 Title: A fast, reproducible and strong vision language assistant for mobile devices Year: (2023)
Ref_id:b80 Title: Mobilevlm v2: Faster and stronger baseline for vision language model Year: (2024)
Ref_id:b81 Title: Llava-phi: Efficient multi-modal assistant with small language model Year: (2024)
Ref_id:b82 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b83 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b84 Title: Introducing idefics: An open reproduction of state-of-the-art visual language model Year: (2023)
Ref_id:b85 Title: Unified language-vision pretraining with dynamic discrete visual tokenization Year: (2023)
Ref_id:b86 Title: Metamorph: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b87 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b88 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b89 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b90 Title: A comprehensive evaluation benchmark for multimodal large language models Year: (2023)
Ref_id:b91 Title: Seedbench: Benchmarking multimodal llms with generative comprehension Year: (2023)
Ref_id:b92 Title: Is your multi-modal model an all-around player? arXiv preprint Year: (2023)
Ref_id:b93 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b94 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b95 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b96 Title: Efficient frontier visual language models Year: (2024)
Ref_id:b97 Title: Flow-grpo: Training flow matching models via online rl Year: (2025)
Ref_id:b98 Title: Vlmevalkit: An open-source toolkit for evaluating large multi-modality models Year: (2024)
Ref_id:b99 Title: Unified-io: A unified model for vision, language, and multi-modal tasks Year: (2022)
Ref_id:b100 Title: Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action Year: (2024)
Ref_id:b101 Title: Unified multimodal llm with discrete sequence modeling Year: (2024)
Ref_id:b102 Title: Generative multimodal pretraining with discrete diffusion timestep tokens Year: (2025)
Ref_id:b103 Title: Dreamllm: Synergistic multimodal comprehension and creation Year: (2024)
Ref_id:b104 Title: Minigpt-5: Interleaved vision-and-language generation via generative vokens Year: (2023)
Ref_id:b105 Title: Next-gpt: Any-to-any multimodal llm Year: (2024)
Ref_id:b106 Title: Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset Year: (2025)
Ref_id:b107 Title: Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation Year: (2024)
Ref_id:b108 Title: Building normalizing flows with stochastic interpolants Year: (2023)
Ref_id:b109 Title: Efficient video prediction via sparsely conditioned flow matching Year: (2023)
Ref_id:b110 Title:  Year: (2025)
Ref_id:b111 Title: Video generation models as world simulators Year: (2024)
Ref_id:b112 Title:  Year: (2025)
Ref_id:b113 Title: Generative pre-training for speech with flow matching Year: (2023)
Ref_id:b114 Title: Voicebox: Text-guided multilingual universal speech generation at scale. Advances in neural information processing systems Year: (2023)
Ref_id:b115 Title: Audiobox: Unified audio generation with natural language prompts Year: (2023)
Ref_id:b116 Title: Fast protein backbone generation with se (3) flow matching Year: (2023)
Ref_id:b117 Title: Alphafold meets flow matching for generating protein ensembles Year: (2024)
Ref_id:b118 Title: )-stochastic flow matching for protein backbone generation Year: (2023)
Ref_id:b119 Title: A vision-language-action flow model for general robot control Year: (2024)
Ref_id:b120 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b121 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b122 Title: Diffusion-lm improves controllable text generation Year: (2022)
Ref_id:b123 Title: Diffuseq: Sequence to sequence text generation with diffusion models Year: (2023)
Ref_id:b124 Title: Likelihood-based diffusion language models Year: (2023)
Ref_id:b125 Title: Argmax flows and multinomial diffusion: Learning categorical distributions Year: (2021)
Ref_id:b126 Title: Masked generative image transformer Year: (2022)
Ref_id:b127 Title: Film: Fill-in language models for any-order generation Year: (2023)
Ref_id:b128 Title: A reparameterized discrete diffusion model for text generation Year: (2024)
Ref_id:b129 Title: Score-based continuoustime discrete diffusion models Year: (2023)
Ref_id:b130 Title: A continuous time framework for discrete denoising models Year: (2022)
Ref_id:b131 Title: Scaling up masked diffusion models on text Year: (2024)
Ref_id:b132 Title: Diffusion of thought: Chain-of-thought reasoning in diffusion language models Year: (2024)
Ref_id:b133 Title: Implicit search via discrete diffusion: A study on chess Year: (2025)
Ref_id:b134 Title: Beyond autoregression: Discrete diffusion for complex reasoning and planning Year: (2025)
Ref_id:b135 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: ()
Ref_id:b136 Title: Sharegpt-4o: Comprehensive multimodal annotations with gpt-4o Year: (2023)
Ref_id:b137 Title: Visual spatial reasoning Year: (2023)
Ref_id:b138 Title: Allava: Harnessing gpt4v-synthesized data for lite vision-language models Year: (2024)
Ref_id:b139 Title: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning Year: (2021)
Ref_id:b140 Title: To see is to believe: Prompting gpt-4v for better visual instruction tuning Year: (2023)
Ref_id:b141 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2023)
Ref_id:b142 Title: ViQuAE, a dataset for knowledge-based visual question answering about named entities Year: (2022)
Ref_id:b143 Title: Raven: A dataset for relational and analogical visual reasoning Year: (2019)
Ref_id:b144 Title: Visual7W: Grounded Question Answering in Images Year: (2016)
Ref_id:b145 Title: Icdar2019 competition on scanned receipt ocr and information extraction Year: (2019)
Ref_id:b146 Title: Funsd: A dataset for form understanding in noisy scanned documents Year: (2019)
Ref_id:b147 Title: Ocr-vqa: Visual question answering by reading text in images Year: (2019)
Ref_id:b148 Title: Mlhme-38k Year: (2025)
Ref_id:b149 Title: Scene text recognition using higher order language priors Year: (2012)
Ref_id:b150 Title: Syntax-aware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b151 Title: Ocr-free document understanding transformer Year: ()
Ref_id:b152 Title: Visual information extraction in the wild: practical dataset and end-to-end solution Year: (2023)
Ref_id:b153 Title: The iam-database: an english sentence database for offline handwriting recognition Year: (2002)
Ref_id:b154 Title: Textcaps: a dataset for image captioning with reading comprehension Year: (2020)
Ref_id:b155 Title: Coco-text: Dataset and benchmark for text detection and recognition in natural images Year: (2016)
Ref_id:b156 Title: Proceedings of ieee international conference on frontiers in handwriting recognition Year: (2014)
Ref_id:b157 Title:  Year: (2025)
Ref_id:b158 Title: Kleister: key information extraction datasets involving long documents with complex layouts Year: (2021)
Ref_id:b159 Title: Towards complex document understanding by discrete reasoning Year: (2022)
Ref_id:b160 Title: Compositional semantic parsing on semi-structured tables Year: (2015)
Ref_id:b161 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: ()
Ref_id:b162 Title: Robut: A systematic study of table qa robustness against humanannotated adversarial perturbations Year: (2023)
Ref_id:b163 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b164 Title: Plotqa: Reasoning over scientific plots Year: (2020)
Ref_id:b165 Title: Dvqa: Understanding data visualizations via question answering Year: (2018)
Ref_id:b166 Title: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Year: (2022)
Ref_id:b167 Title: Vistext: A benchmark for semantically rich chart captioning Year: (2023)
Ref_id:b168 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b169 Title: Aligning large multi-modal model with robust instruction tuning Year: (2023)
Ref_id:b170 Title: Websrc: a dataset for web-based structural reading comprehension Year: (2021)
Ref_id:b171 Title: Mathematical visual instruction tuning Year: (2024)
Ref_id:b172 Title: Geomverse: A systematic evaluation of large models for geometric reasoning Year: (2023)
Ref_id:b173 Title: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: (2021)
Ref_id:b174 Title: Measuring multimodal mathematical reasoning with math-vision dataset Year: (2024)
Ref_id:b175 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b176 Title: Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension Year: (2017)
Ref_id:b177 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: ()
Ref_id:b178 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b179 Title: Mammoth: Building math generalist models through hybrid instruction tuning Year: (2023)
Ref_id:b180 Title: Openbezoar: Small, cost-effective and open models trained on mixes of instruction data Year: (2024)
Ref_id:b181 Title: Mammoth2: Scaling instructions from the web Year: (2024)
Ref_id:b182 Title: Emova: Empowering language models to see, hear and speak with vivid emotions Year: (2024)
