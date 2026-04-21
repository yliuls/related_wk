Title: PARTITION GENERATIVE MODELING: MASKED MODELING WITHOUT MASKS
Abstract: Masked generative models (MGMs) can generate tokens in parallel and in any order, unlike autoregressive models (ARMs), which decode one token at a time, left-to-right. However, MGMs process the full-length sequence at every sampling step, including [MASK]  tokens that carry no information. In contrast, ARMs process only the previously generated tokens. We introduce "Partition Generative Models" (PGMs), which replace masking with partitioning. Tokens are split into two groups that cannot attend to each other, and the model learns to predict each group conditioned on the other, eliminating [MASK] tokens entirely. Because the groups do not interact, PGMs can process only the clean tokens during sampling, like ARMs, while retaining parallel, any-order generation, like MGMs. On Open-WebText, PGMs achieve 5-5.5× higher throughput than MDLM while producing samples with lower Generative Perplexity. On ImageNet, PGMs reach comparable FID to MaskGIT with a 7.5× throughput improvement. With twice as many steps, the FID improves to 4.56 while remaining 3.9× faster than MGMs. Finally, PGMs remain compatible with existing MGM samplers and distillation methods.

Section: INTRODUCTION
Masked generative models (MGMs) offer two key advantages over autoregressive models (ARMs): they can generate tokens in parallel and in any order, rather than one-by-one, left-to-right. These properties have led to strong results across images (Chang et al., 2022), video (Yu et al., 2023;Villegas et al., 2022), audio (Comunità et al., 2024), and language (Austin et al., 2023;Lou et al., 2024;Sahoo et al., 2024;Shi et al., 2025;Campbell et al., 2024;Gat et al., 2024). However, MGMs are slow at inference. Indeed, at every sampling step, they process the full-length sequence, including [MASK] tokens that carry no information, whereas ARMs process only the previously generated tokens. This limits the practicality of MGMs in large-scale and real-time settings, and is a crucial disadvantage for test-time compute scaling (Snell et al., 2024;Wu et al., 2024) compared to ARMs.
Addressing the inference inefficiency of MGMs is not trivial because training and sampling must be consistent. MGMs are trained with bidirectional architectures over the full sequence, so every hidden representation depends on all L positions, including masked ones. Furthermore, naively decoding tokens block-by-block means feeding the model shorter sequences at inference, which differs from training and leads to poor sample quality (Deschenaux & Gulcehre, 2024).
Prior work addresses the slow inference of MGMs from different angles. Decoding more tokens per step increases throughput, but degrades sample quality. Distillation (Deschenaux & Gulcehre, 2025;Zhu et al., 2025;Sahoo et al., 2025a) reduces the number of sampling steps, but each step remains equally expensive, and distillation can affect the sample diversity (Gandikota & Bau, 2025). Block Diffusion (Arriola et al., 2025) enables partial KV-caching by generating tokens block-by-block, but sacrifices the any-order generation capability. None of these approaches make individual sampling steps cheaper while preserving the full flexibility of MGMs.
We introduce Partition Generative Models (PGMs), which replace masking with partitioning. Tokens are split into two disjoint groups, and a group-wise attention mechanism ensures that no information flows between them. The model learns to predict each group conditioned on the other, eliminating [MASK] tokens entirely. Because the two groups do not interact, PGMs process only the clean tokens during sampling, just like ARMs, while retaining the ability to generate tokens in parallel and in any order, like MGMs. We propose the Partition Transformer, a dedicated architecture that prevents information flow between groups.
Contributions (1) We introduce PGMs and the Partition Transformer, a new architecture that enables MGM-style parallel, any-order generation without [MASK] tokens. PGMs are compatible with existing MGM samplers (Besnier et al., 2025) and distillation methods (Deschenaux & Gulcehre, 2025), making them a drop-in replacement. (2) On OpenWebText (Gokaslan & Cohen, 2019), PGMs generate samples with lower Generative Perplexity than MDLM (Sahoo et al., 2024) and reach similar downstream task performance, before and after distillation, while achieving 5-5.5× higher throughput. On ImageNet, PGMs reach comparable FID to MaskGIT (Chang et al., 2022) with a 7.5× throughput improvement.
(3) We show that PGM are trained with denser supervision than MGM. Since each group predicts the other, a single sequence yields two complementary training signals. This reduces gradient variance and yields a 1.95 reduction in validation perplexity on LM1B (Chelba et al., 2014) compared to MDLM with the same number of layers.
this section cite: ['b6', 'b61', 'b55', 'b9', 'b1', 'b32', 'b42', 'b48', 'b5', 'b19', 'b50', 'b58', 'b12', 'b13', 'b63', 'b17', 'b0', 'b3', 'b13', 'b42', 'b6', 'b7']

Section: BACKGROUND

this section cite: []

Section: SEQUENCE MODELING
We consider the task of generating sequences x = (x 1 , . . . , x L ) of length L over a vocabulary V = {0, . . . , N -1}. The training dataset D contains finitely many sequences drawn from an unknown data distribution p data over V L . Autoregressive models (ARMs) factorize the distribution as p θ (x) = L i=1 p θ (x i | x <i ), where x <i denotes the prefix before position i. Tokens are sampled sequentially, and because each conditional only depends on the prefix x <i , ARMs process only the previously generated tokens instead of the whole sequence.
this section cite: []

Section: MASKED GENERATIVE MODELS
MGMs augment the vocabulary with a special [MASK] token, absent from the training data. Given x ∈ D, let z t denote a corrupted sequence where the token z ℓ t at position ℓ is [MASK] with probability p t , or the clean value x ℓ otherwise. The masking probability p t is an increasing function of t ∈ [0, 1] with p 0 = 0 and p 1 = 1. MGMs train a denoiser x θ : V L → R L×N whose sampling distribution is modeled as factorized marginals:
ℓ:z ℓ t =[MASK] p ℓ θ (. | z t ). Because MGMs sample independently from each p ℓ θ (. | z t ), they cannot model arbitrary joint distributions, unlike ARMs. However, they can decode tokens in parallel. The training objective for MGMs is:
L MGM := E x∼D,t∼U [0,1] [w(t)CE(x θ (z t ; t), x)] , (1
)
where w : [0, 1] → R ≥0 is a weighting function and CE(x, x) denotes the cross-entropy loss over masked positions. To generate samples, MGMs start from a fully masked sequence and iteratively unmask subsets of positions over multiple evaluations of x θ , selecting positions at random or based on confidence scores. We now describe two instantiations used in this work.
this section cite: []

Section: MDLM Masked Diffusion Language
Models (MDLM; Sahoo et al. (2024); Ou et al. (2025); Shi et al. (2025)) are MGMs for language modeling. Analogously to continuous diffusion (Sohl-Dickstein et al., 2015; Song & Ermon, 2020; Ho et al., 2020; Kingma et al., 2023), MDLM defines a forward process that corrupts clean data and a generative process that recovers samples from noise. The forward process is: q t (.|x) := Cat(.; α t x + (1 -α t )π),
(2) where x is the one-hot representation, π = m is the one-hot encoding of [MASK] , and α t is a strictly decreasing noise schedule with α 0 = 1, α 1 = 0. (2) is applied independently at every position. The posterior distribution is:
p s|t (.|z t , x) = Cat(.; z t ), z t ̸ = m, Cat .; (1-αs)m+(αs-αt)x (1-αt) , z t = m.(3)
To generate samples, we fix a decreasing sequence of times 1 = τ T > • • • > τ 0 = 0, set z τ T to the all [MASK] tokens sequence, and iteratively sample from
z τi-1 ∼ p θ,τi-1|τi (. | z τi ) = p τi-1|τi (.|z t , x θ (z τi , τ i )).(4)
MDLM optimizes a variational bound on log-likelihood that reduces to (1) with w(t) = α ′ t 1-αt . Only masked positions contribute to the loss.
MaskGIT MaskGIT (Chang et al., 2022) is an MGM that operates in the latent space of a pretrained VQGAN (Esser et al., 2021) tokenizer. MaskGIT proposes tokens xℓ at masked position and uses the predicted likelihood of the sampled token xℓ as a confidence score: c ℓ = x ℓ θ (z t ; t) xℓ . A predefined schedule determines the number of positions to unmask, and the most confident positions are kept. Tokens generated in earlier steps are kept unchanged. This differs from MDLM, which denoises at random positions. Besnier et al. (2025) observed that confidence-based sampling tends to decode spatially clustered tokens, since the denoiser is most confident near previously generated positions. Because MGMs sample independently from a product of marginals ℓ∈S p θ (x ℓ | z τ ) rather than the joint token distribution, decoding nearby tokens increases the risk of generating inconsistent samples. By sampling according to a low-discrepancy sequence (Halton, 1964)
log pθ (x | c) = (1 + ω) log p θ (x | c) -ω log p θ (x | ø),(5)
Self-Distillation Through Time Self-Distillation Through Time (SDTT; Deschenaux & Gulcehre (2025)) accelerates sampling from MGMs by distilling a teacher trained for denoising with many steps into a few-steps student. Let p (m) θ denote the distribution of samples generated with m steps using a denoiser x θ , and let p (k) ν denote the distribution when using k < m steps with a student denoiser x ν . SDTT trains x ν with the following objective:
min ν E z0∼D,zt∼qt(zt|z0) δ(x ν (z t , t) || xteacher θ (z t , t, m /k)) ,(6)
where δ is a divergence measure (e.g., KLD) and xteacher θ (z t , t, m /k) are the distillation targets. These targets are constructed using m /k sampling steps with the teacher, starting from z t and collecting the predicted log-probabilities for each token at the step where a token was denoised. After training, one step of the student should match m /k teacher steps. SDTT can be applied iteratively by reusing the student as teacher in each round (Salimans & Ho, 2022), progressively halving the number of required steps. Empirically, distilling 2 steps per round is most effective.
this section cite: ['b6', 'b16', 'b3', 'b22']

Section: PARTITION GENERATIVE MODELING
At each sampling step, MGMs process the entire sequence, including many [MASK] tokens that will not be decoded yet. In contrast, ARMs process clean tokens only, but generate one token at a time. Partition Generative Models (PGMs) combine the strengths of both approaches, by generating multiple tokens in parallel, like MGMs, while processing only the clean tokens, like ARMs. PGMs are a direct extension of the MGM paradigm. As a result, sampling algorithms, guidance mechanisms, and distillation methods developed for MGMs apply directly to PGMs. Only the neural network architecture must be adapted (Sec. 4).
this section cite: []

Section: TRAINING
From Masking to Partitioning Instead of replacing tokens with [MASK] , PGMs partition the sequence into two complementary groups. Given x ∈ D and t ∼ U[0, 1], each token is assigned to group 1 with probability p t = 1 -α t , and to group 0 otherwise. Let g ∈ {0, 1} L denote the group membership vector. We propose a Transformer variant in Sec. 4 that ensures that information cannot flow between groups. Predictions at positions in group 0 depend only on tokens in group 1, and vice-versa (Figure 2). This is consistent with MGMs, where masked tokens are predicted from clean ones, except that PGMs learn from both groups.
Connection to the MDLM Variational Bound In MDLM, the forward process (2) masks each position independently, so at time t an expected fraction α t of tokens remain clean. PGMs assign an expected fraction α t of tokens to group 0, which plays the same role as the clean tokens in MDLM. By treating group 0 as clean and group 1 as masked, the MDLM loss weight w(t) = α ′ t 1-αt is applied to tokens in group 1. By symmetry, tokens in group 0 are weighted by w(1 -t). Therefore, in a single forward pass, PGMs evaluate the MDLM training objective at two complementary masking rates. Hence, the training objective is
L PGM := E x∼D,t∼U [0,1] w PGM (g, t)CE(x θ (x; g; t), x) ,(7)
where Variance Reduction Unlike MGMs, which compute the loss over masked positions only (Figure 2), PGMs compute the loss at every position, yielding two gradient contributions per training sample. By training on two complementary copies, PGMs reduce the variance. Empirically, training diffusion models with lower variance improves the validation likelihood (Kingma et al., 2023;Sahoo et al., 2024). We study the variance reduction in Sec. 5.3.
w PGM (g, t) i = w(t) if g i = 0 w(1 -t) if g i = 1.(8)
this section cite: ['b31', 'b42']

Section: SAMPLING
During inference, PGMs process clean tokens only, like ARMs, yet decode tokens in parallel at arbitrary positions, like MGMs (Figure 2). Let C τ ⊆ {1, . . . , L} denote the clean token indices at step τ ∈ {1, . . . , T }, with n τ = |C τ | and m τ = L -n τ . At each step, we select k τ masked positions, sample from p θ (• | x Cτ ), and add the decoded tokens to C τ +1 . For text, we find that using a fixed schedule with k τ = k tokens per step (Algo. 2) improves sample quality and throughput compared to the MDLM posterior that decodes each position with probability αs-αt 1-αt and requires padding for batched generation (Algo. 3; Suppl. E.2). For images, we experiment with both the confidence and Halton samplers (Suppl. B, Besnier et al. (2025)). The Halton sampler performs better empirically, so we report confidence-based sampler results in Suppl. D.3.
this section cite: ['b3']

Section: THE PARTITION TRANSFORMER
PGMs require a careful architectural design. In particular, since our goal is to process a single group only during inference, tokens across groups should not attend to each other. As shown in Figure 2 and Figure 3, we build the Partition Transformer such that the predictions for tokens in group 0 are based on tokens in group 1 only. The Partition Transformer implements a mechanism to swap the physical location of information across groups. During training, this allows using the input sequence x as target. During sampling, it moves information about the input tokens from the clean positions to the positions to predict. Our architecture consists of an encoder, a GroupSwap layer, and a decoder, which we describe below.
Encoder The encoder is made of partition-wise self-attention blocks, which are similar to standard bidirectional transformer blocks except that tokens in separate groups do not attend to each other.
Table 1: Validation perplexity, sampling latency, and throughput (TP) on LM1B and OpenWebText. PGM k / m uses k encoder and m decoder layers. The best PGM per dataset is highlighted. Latency and TP are measured at batch size 32. † Trained with a 2× larger batch size (Sec. 5.3). See
Table 5 for architecture ablations. Model #Params Val. PPL ↓ Latency (sec) ↓ TP (tok/sec) ↑ LM1B (ctx len. 128) MDLM 170M 27.67 3.78 1'081.57 MDLM † (Compl. masking) 170M 25.72 3.78 1'081.57 PGM 6 / 6 171M 26.80 2.12 1'930.93 OpenWebText (ctx len. 1024) MDLM 170M 23.07 31.41 1'043.22 MDLM † (Compl. masking) 170M 22.98 31.41 1'043.22 PGM 8 / 8 203M 22.61 5.86 5'585.57 PGM 6 / 6 (dim. 1024) 268M 21.43 5.93 5'518.09
Decoder The decoder uses cross-attention layers, whose keys and values are computed based on the output of the encoder. In contrast, the queries are computed using either the output of the GroupSwap layer (for the first block of the decoder) or the output of the previous decoder block (see Sec. 4.1).
Importantly, there is no self-attention layer in the decoder, which allows efficient generation, as we can compute predictions solely at the positions that we will decode.
this section cite: []

Section: THE GROUPSWAP LAYER
In the encoder, information remains localized. If a token belongs to group 0, its hidden representation only depends on tokens in group 0. For prediction, however, we require the opposite: representations at positions in group 0 must depend exclusively on group 1, and vice versa. To enforce this, we introduce the GroupSwap layer (Figure 3B), which exchanges information between groups. The GroupSwap layer is implemented using cross-attention, and to prevent information leakage, the queries used in cross-attention cannot depend on tokens in the other group. We describe two ways of initializing queries.
Data-Independent Queries Let u ∈ R H be a learnable vector. To initialize the queries, we replicate u across the sequence length, add fixed positional encodings, and apply layer normalization followed by a linear projection. The query matrix V ∈ R L×H (where
V i;• is the i-th row) satisfies V i;• = W LN u + pos i;• + b ,(9)
where W ∈ R H×H , b ∈ R H are learnable parameters and LN denotes layer normalization (Ba et al., 2016). We use sinusoidal positional encoding (Vaswani et al., 2023):
pos i,j = cos i 10000 2j/H if j < H /2 sin i 10000 2j/H-1 otherwise (10
)
Data-Dependent Queries Let X ∈ R L×H be the encoder output. We first perform a group-wise aggregation over the sequence length (e.g., logsumexp or mean) to obtain vectors Y 0 , Y 1 ∈ R H , the aggregate representations of groups 0 and 1. The queries V ′ are then We evaluate them using the validation perplexity and downstream task accuracy before and after distillation with SDTT (Deschenaux & Gulcehre, 2025). We compare PGM with MaskGIT (Chang et al., 2022) on VQGAN-quantized (Esser et al., 2021) ImageNet256 (Deng et al., 2009) (Sec. 5.2).
V ′ i;• = V i;• + Y 1 , if g i = 0 Y 0 otherwise. (11
As described in Sec. 3, by predicting each group from the other, PGMs implement a mechanism akin to training on two complementary masked sequences per batch, while also introducing a new architecture (Sec. 4). The effect of complementary masking is studied in isolation in Sec. 5.3. Our experiments show that, for both language and image modeling, and after distillation, PGMs are competitive with MDLM and MaskGIT, while providing a 5-5.5× throughput improvement for text and a 7.5× improvement for images. Find more experimental details in Suppl. C.
this section cite: ['b2', 'b54', 'b13', 'b6', 'b16', 'b11']

Section: LANGUAGE MODELING

this section cite: []

Section: Experimental settings
We closely follow the settings of Sahoo et al. (2024). MDLM uses a modified Diffusion Transformer (Peebles & Xie, 2023;Lou et al., 2024) with RoPE (Su et al., 2023), with 12 layers and an embedding dimension of 768, without time conditioning. We train with a global batch size of 512 for 1M steps, dropout of 0.1, and the Adam optimizer with learning rate 3 × 10 -4 and no weight decay. We maintain an Exponential Moving Average (EMA) of the weights with decay 0.9999. For PGM, we use the Partition Transformer architecture (Sec. 4) with 12 or 16 layers, embedding dimensions of 768 or 1024, and varying numbers of encoder and decoder layers. On LM1B, all models use a context length of 128, with shorter documents padded and tokenized using the bert-base-uncased (Devlin et al., 2019) tokenizer. On OWT, we use a context length of 1024 with sentence packing (Raffel et al., 2023) with the GPT-2 tokenizer and insert an [EOS] token between documents. Since the dataset lacks an official validation split, the last 100k documents are reserved for validation. To evaluate the sample quality, we use the Generative Perplexity (Gen. PPL), computed using GPT-2 Large (Radford et al., 2019), following Sahoo et al. (2024). We cast the logits in float64 prior to sampling, following Zheng et al. (2025).
Likelihood Evaluation After 1M steps, PGMs with as many layers as MDLM achieve a validation perplexity of 1.95 lower than MDLM on LM1B (Table 1). Table 5 (left) shows that balanced models with equal numbers of encoder and decoder layers outperform imbalanced variants. Interestingly, data-independent queries perform comparably to data-dependent queries, so we use the simpler, dataindependent version in all subsequent experiments. On OpenWebText, PGMs with the same number of layers and embedding dimension as MDLM slightly underperform (Table 5, right). Increasing the number of encoder and decoder layers by two, or increasing the embedding dimension to 1024, allows PGMs to surpass MDLM in validation perplexity, while achieving at least 5× higher sampling throughput. This improved efficiency makes PGMs particularly attractive for scaling test-time computation (Madaan et al., 2023;Yao et al., 2023;Snell et al., 2024;Wu et al., 2024;Chen et al., 2024;Brown et al., 2024;Goyal et al., 2024).
this section cite: ['b42', 'b38', 'b32', 'b53', 'b14', 'b41', 'b40', 'b42', 'b62', 'b34', 'b60', 'b50', 'b58', 'b8', 'b4', 'b20']

Section: Downstream Evaluation
Following Deschenaux & Gulcehre (2024); Nie et al. (2025), we evaluate MDLM and PGMs trained on OpenWebText using the lm-eval-harness suite (Gao et al., 2024). As shown in Table 2, PGMs slightly outperform MDLM on six out of eight tasks, although the overall accuracy across models is similar. This suggests that PGM achieves faster inference without sacrificing downstream performance. Since lm-eval-harness is originally designed for ARMs, we must adapt it for MGMs. Fortunately, both MDLM and PGM can compute a variational bound on the likelihood, which is used in place of the true likelihood to select the most probable answer in multiple-choice tasks. Additional details and tasks are provided in Suppl. D.5.
this section cite: ['b35']

Section: Distillation of PGMs
After likelihood training, PGMs achieve 5 -5.5× higher throughput than MDLM. To further accelerate sampling, we apply Self-Distillation Through Time (SDTT; Deschenaux & Gulcehre (2025)). To remain as faithful as possible to the implementation of Deschenaux & Gulcehre (2025), we apply the distillation loss to a single group while treating the other as [MASK] tokens. This shows that PGMs are compatible with distillation methods designed for MGMs. We leave the development of new distillation strategies for PGMs to future work. Hence, the setup naturally favors MDLM. Figure 4 (right) and Table 6 compare the Gen. PPL, unigram entropy, and sampling speed of PGM and MDLM. After five rounds of distillation, and with standard ancestral sampling, PGMs achieve higher Generative Perplexity and entropy than MDLM. With nucleus sampling (p = 0.9) (Holtzman et al., 2020), PGMs produce samples with comparable perplexity and entropy. Due to the overhead of nucleus sampling, the speed advantage of PGMs decreases from at least 5× to approximately 4.6× faster than MDLM for the same number of steps (Fig. 4). Generative perplexity alone does not fully capture language model performance, hence we also evaluate distilled models on downstream tasks. As shown in Table 2, distillation slightly shifts accuracy across tasks, but overall performance remains similar. PGMs still achieve slightly higher accuracy than MDLM on most tasks after distillation.
this section cite: ['b13', 'b28']

Section: IMAGE MODELING
0 5000 10000 15000 20000 Throughput (tokens/sec.) 35 40 50 60 90 Gen. PPL (5.5) (5.4) (5.4) (5.4) (5.4) (5.3) (5.5) (5.5) (5.5) (5.4) (5.4) (5.4) (5.5) (5.4) (5.4) (5.4) (5.4) (5.4) MDLM PGM PGM+nucleus (p=0.9) Experimental Settings We train MaskGIT (Chang et al., 2022) and PGM on ImageNet256. Images are cropped to a centered square along the longer side and then rescaled to 256 × 256. We use the MaskGIT implementation of Besnier et al. (2025), including their pre-trained VQ-GAN tokenizer. We train for 500k steps with a batch size of 256 using AdamW (weight decay 0.03, learning rate 1e-4, cosine schedule with 2500 warmup steps). We use a dropout of 0.1 in the Transformer. All models are classconditional, with a class-label dropout of 0.1 to enable classifier-free guidance (CFG) at sampling time. As Besnier et al. (2025), we train with one register (Darcet et al., 2024) for the MaskGIT baseline, and two (one per group) for PGM, so that we can use one register during sampling. We sample with the confidence and Halton samplers.
this section cite: ['b6', 'b3', 'b3', 'b10']

Section: Results
In Figure 1 (left), we compare the Fréchet Inception Distance (FID; Heusel et al. ( 2018)) of samples from MaskGIT (Chang et al., 2022) and PGM, using the Halton sampler and classifierfree guidance with the guidance weight w ∈ {0, 1, . . . , 6} that yields the lowest FID. PGM 12/12 achieves a 7.5× higher throughput with only a slight FID degradation (5.54 vs. 5.35). Increasing the sampling steps to 64 further improves the FID to 4.56, while remaining 3.9× faster than MaskGIT. See Suppl. D.3 for full results across guidance strengths.
this section cite: ['b6']

Section: ISOLATING THE EFFECT OF COMPLEMENTARY MASKING
Experimental Setup To disentangle the contributions of PGM, we isolate the effect of complementary masking (Sec. 3) by training a standard bidirectional Transformer with double the batch size. Each input sequence is turned into two complementary masked copies: if the token at position ℓ is masked in one copy, it remains unmasked in the other. This setup provides an upper bound on the potential gains, as it directly measures the benefit of complementary masks during training.
Results Table 1 shows that complementary masking improves the validation perplexity on LM1B and OWT, though with smaller gains on OWT. On both datasets, a gap remains between PGM and MDLM with complementary masking. This suggests that the current neural network architecture can be improved further. Because of the smaller improvement on OWT, we must increase the parameter count to surpass MDLM. Nonetheless, recall that despite having more parameters, PGMs remain at least 5× faster than MDLM during sampling. In Suppl. D.1, we present preliminary experiments exploring why complementary masking improves performance on LM1B but not on OpenWebText. (Austin et al., 2023;Lou et al., 2024;Shi et al., 2025;Sahoo et al., 2024;von Rütte et al., 2025;Schiff et al., 2025;Haxholli et al., 2025;Sahoo et al., 2025a) and discrete flow matching (Campbell et al., 2024;Gat et al., 2024) have demonstrated that MGMs can Table 2: Accuracy on downstream tasks (Gao et al., 2024). HS: HellaSwag, OQA: OpenBook QA. Arc: Arc-easy. We select the tasks following Nie et al. (2025). We see that distillation slightly changes the downstream tasks performance, but that PGMs continue to outperform MDLM on most tasks. The best performance is bolded, while the second best is underlined.
this section cite: ['b1', 'b32', 'b48', 'b42', 'b56', 'b47', 'b24', 'b5', 'b19', 'b35']

Section: RELATED WORK

this section cite: []

Section: Discrete Diffusion Although autoregressive models currently dominate text generation, recent advances in discrete diffusion
LAMBADA Arc BoolQ HS OQA PIQA RACE SIQA Before Distillation MDLM 38.52 37.88 49.42 31.36 28.60 58.27 28.04 38.84 PGM 8 / 8 46.98 40.40 53.49 33.20 26.60 58.92 26.89 39.97 PGM 6 / 6 (1024) 41.39 39.98 49.82 34.27 25.40 59.19 27.37 40.28 After Distillation (SDTT) MDLM 41.34 33.80 48.59 30.75 28.80 57.73 27.94 38.79 PGM 8 / 8 47.22 37.42 51.50 31.62 25.80 59.03 30.62 39.61 PGM 6 / 6 (1024) 44.48 36.70 49.36 32.55 25.00 59.85 27.37 39.25 approach AR models in generation quality. We propose a simple framework that allows sampling without processing any [MASK] tokens, but remains compatible with methods developed for MGMs (such as distillation and alternative samplers). Variable Length Masked Diffusion Block Diffusion (BD; Arriola et al. (2025)) enables partial KV-caching (Pope et al., 2022) by generating tokens block-by-block using discrete diffusion. BD improves throughput but sacrifices the any-order generation capabilities of MGMs. We do not experiment with integrating causal attention to enable KV-caching. However, Ma et al. (2025); Wu et al. (2025) show that KV caching can be integrated post-hoc into MGMs despite being trained without causal attention. FlexMDM (Kim et al., 2025) and Edit Flows (Havasi et al., 2025) enable variable-length generation via insertion, deletion, and replacement. While promising, these depart from the simplicity of MGM and PGM. Finally, Eso-LMs (Sahoo et al., 2025b) train with a hybrid AR-MGM objective. Sahoo et al. (2025b) first sample a draft in MGM mode, then fill in the remaining tokens autoregressively. During training, Eso-LMs must choose the fraction of examples to process in AR versus MGM mode, which adds a hyperparameter to tune. Eso-LMs use [MASK] tokens during training in MGM mode, whereas PGMs do not because of the Partition Transformer.
Non-Autoregressive Language Models Any-order and any-subset autoregressive models (Yang et al., 2020;Pannatier et al., 2024;Shih et al., 2022;Guo & Ermon, 2025) factorize the sequence distribution autoregressively over permutations of tokens. Hence, these models use causal attention and generate tokens one by one. In contrast, MGMs use bidirectional attention and generate multiple tokens in parallel, which is the setting PGM builds on.
this section cite: ['b52', 'b37', 'b49', 'b21']

Section: CONCLUSION
We introduce Partition Generative Modeling (PGM), a novel approach to masked generative modeling that eliminates [MASK] tokens entirely. PGM achieves significant improvements in inference speed on both text and images, with minimal effect on quality. The significant improvements suggest that PGM might be suited for domains that benefit from test-time scaling, such as coding and reasoning. We show that PGMs can be distilled for further acceleration. Future work should explore optimizations to the PGM architecture, investigate distillation techniques specifically designed for PGMs, and extend the approach to multimodal settings. In summary, PGM offers an alternative to masked generative models, with particular advantages for applications where inference speed is critical.
Algorithm 1 Building the Halton Unmasking Schedule
1: Input: Grid size H 2: Output: Ordered list of L = H 2 grid positions 3: schedule ← [] 4: seen ← ∅ 5: i ← 1 6: while |schedule| < L do 7: cell ← (⌊Φ 2 (i) • H⌋, ⌊Φ 3 (i) • H⌋) 8: if cell / ∈ seen then 9: seen ← seen ∪ {cell} 10: schedule.append(cell) 11: end if 12: i ← i + 1 13: end while 14: return schedule
this section cite: []

Section: A LIMITATIONS
To match the validation perplexity of the MDLM baseline at a context length of 1024, our models require a slight increase in parameters. We attribute this to the GroupSwap layer, and future work will explore more efficient mechanisms for information exchange between groups in PGMs. While PGMs offer faster inference, their training is slightly more computationally expensive (Appendix E), as we use torch's default attention implementation ("sdpa") for simplicity. By reordering tokens according to their group assignment, the self-attention matrices becomes block-diagonal. Future work will explore efficient kernel implementations that exploit this block-diagonal sparsity. Partition Generative Modeling is a general framework, and its application to multimodal settings remains an open direction for future research.
this section cite: []

Section: B SAMPLING WITH HALTON SEQUENCES (MASKGIT)
Let S = {ℓ | z ℓ τ = [MASK] } denote the set of masked positions at step τ . MGMs samples independently at all the masked positions from a product of marginal predictions ℓ∈S p θ (x ℓ | z τ ), not from the joint p(x | z τ ). The KL divergence between the joint and the product of marginals is the mutual information (MI) (Besnier et al., 2025):
D KL p({x ℓ } ℓ∈S | z τ ) ℓ∈S p θ (x ℓ | z τ ) = MI({x ℓ } ℓ∈S | z τ ).(12)
Empirically, the denoiser is most confident at positions close to previously generated tokens. Therefore, the decoded tokens tend to cluster together. This leads to a larger MI, with a higher risk of generating inconsistent tokens. Besnier et al. (2025) propose to replace the confidence-based ordering with a Halton sequence (Halton, 1964), a low-discrepancy sequence whose consecutive points are far apart in space. Formally, let a j (i) denote the j-th digit of the representation of i ∈ N + in base b, so that i = j a j (i) b j . Let the radical-inverse function Φ b denote the inverse of i in base b:
Φ b (i) = m j=0 a j (i) b -(j+1) ∈ [0, 1).(13)
Recall that MaskGIT operates over a square grid of VQGAN tokens (Esser et al., 2021), of size H × H (L = H 2 ). To build the unmasking order (Algo. 1), Besnier et al. (2025) iterate over positive integers i and compute the cell coordinates (⌊Φ 2 (i) • H⌋, ⌊Φ 3 (i) • H⌋). If the cell was already visited for some i ′ < i, it is skipped. This continues until all L cells are visited. The Halton scheduler consistently improves the FID and, unlike confidence-based sampling, continues to improve with more inference steps (Besnier et al., 2025). In our image experiments, we use both confidence and Halton schedulers.
this section cite: ['b3', 'b3', 'b16', 'b3', 'b3']

Section: C EXPERIMENTAL DETAILS
We trained all models from scratch. Our baselines achieve similar performance as reported by Sahoo et al. (2024). On LM1B, we obtain a validation perplexity of 27.67 after 1M steps (compared to MDLM's reported 27.04), while on OWT, we reach 23.07 (versus MDLM's 23.21).
Minor differences can be expected since estimating the perplexity of diffusion language models involves a Monte-Carlo approximation of the NELBO (1) with finitely many samples. Although we used libraries (e.g., PyTorch) with the same version as MDLM, differences in compute environments and underlying software stacks may also contribute to these variations. Since the performance gap is small, we are confident that we used the code of MDLM correctly.
this section cite: ['b42']

Section: C.1 LM1B
For the LM1B dataset, we employed the bert-base-uncased tokenizer with a context length of 128 tokens, padding shorter sequences. Our architecture consisted of a Diffusion Transformer (DiT) with 12 transformer blocks, 12 attention heads, a hidden dimension of 768, and a dropout rate of 0.1. We optimized the model using Adam (Kingma & Ba, 2017) (learning rate 3e-4, betas of (0.9, 0.999), epsilon 1e-8) without weight decay. We based our implementation on the official MDLM codebase. We trained with a global batch size of 512 across 8 GPUs (2 nodes with 4 GPUs), gradient clipping at 1.0, and a constant learning rate with 2,500 steps of linear warmup. We trained for 1 million steps with an EMA rate of 0.9999. Besides the neural network hyperparameters, the other parameters were unchanged when training the PGM.
C.2 OWT For the OpenWebText (OWT) dataset, we used the GPT-2 tokenizer with a context length of 1024 tokens. Our architecture consisted of a Diffusion Transformer (DiT) with 12 transformer blocks, 12 attention heads, a hidden dimension of 768, and a dropout rate of 0.1. We optimized the model using Adam (Kingma & Ba, 2017) with a learning rate of 3e-4, betas of (0.9, 0.999), and epsilon of 1e-8, without weight decay. We trained with a global batch size of 512 across 16 GPUs (4 nodes with 4 GPUs). We applied gradient clipping at 1.0 and used a constant learning rate schedule with 2,500 steps of linear warmup. The model was trained for 1 million steps with an EMA rate of 0.9999. Besides the neural network hyperparameters, the other parameters were unchanged when training the PGM.
C.3 IMAGENET For the ImageNet experiments, we used a pre-trained VQGAN tokenizer (Esser et al., 2021;Besnier et al., 2025), following the setup of Besnier et al. (2025). The images are tokenized into sequences of 1024 tokens. This allowed for a direct comparison between PGM and MaskGIT, both trained in the codebase of Besnier et al. (2025) and the FID is evaluated using the Halton sampler and the confidence sampler. We compute the FID between 50k generated images and the validation set, following Besnier et al. (2025) All models use 24 transformer blocks. For PGM, we add a GroupSwap layer to enable information exchange between partition groups. We use the same hyperparameters as HaltonMaskGIT for all models, except we reduce the training duration to 500k steps (from 2M) due to computational constraints. All models are trained to be class-conditional, which enables the use of classifier-free guidance to significantly improve performance. Zheng et al. (2025) identified that Masked Diffusion Models often achieve lower Generative Perplexity results because of underflow in the logits when sampling using low precision. The resulting decrease in token diversity can make evaluations based solely on Generative Perplexity misleading. Hence, we always cast the logits to FP64 before sampling.
this section cite: ['b30', 'b30', 'b16', 'b3', 'b3', 'b3', 'b3', 'b62']

Section: C.4 IMPACT OF NUMERICAL PRECISION ON SAMPLING

this section cite: []

Section: C.5 SAMPLE-BASED EVALUATION

this section cite: []

Section: Generative Perplexity
We use the Generative Perplexity to evaluate the quality of samples, following prior work (Lou et al., 2024;Sahoo et al., 2024;Deschenaux & Gulcehre, 2025). The Generative Perplexity measures how well a reference model (in our case, GPT-2 Large) can predict the next token in generated sequences. Specifically, we generate 1 ′ 024 samples from each model being evaluated.
For each generated sample, we compute the Generative Perplexity using GPT-2 Large as follows:
Perplexity = exp - 1 L L i=1 log p GPT-2 Large (x i |x <i ) ,(14)
Using higher precision also affects distillation, which compresses two sampling steps into one. As shown in Table 7, models distilled with float32 achieve lower Generative Perplexity than those trained with mixed precision (bfloat16). We therefore report float32 results in the main body.
this section cite: ['b32', 'b42', 'b13']

Section: D.3 ADDITIONAL RESULTS ON IMAGENET
Table 8 and Table 9 show the FID, IS, latency, and throughput for the Confidence and Halton samplers. Overall, the Halton sampler works best for both MaskGIT and PGM. With 32 steps and the confidence-based sampler, PGM 12/12 gets a better FID than MaskGIT and is 3.58× faster. With 32 steps and the Halton sampler, PGM has a slightly higher FID than MaskGIT (5.54 vs 5.35), but is 7.5× faster. If we increase the number of sampling steps to 64, PGM achieves an FID of 4.56, which is better than MaskGIT, and is 3.92× faster. Generally, the 12/12 variant outperforms the 14/10 variant, which suggests that balanced number of layers in the encoder and decoder is beneficial, just as for language modeling (Table 5).
this section cite: []

Section: D.4 TRAINING STABILITY
Complementary masking introduces occasional spikes in the training loss in both MDLMs and PGMs, as shown in Figure 6. This phenomenon should be kept in mind when scaling PGMs to larger sizes. Despite these spikes, all runs converged on the first attempt. We observed different precision requirements between models. For loss computations, MDLMs performed best with BF16 precision, while PGMs achieved better results with FP32 precision. Both models use mixed precision within the neural network; the precision difference only affects computations performed outside the model, such as the loss calculation. D.5 ADDITIONAL DOWNSTREAM TASKS Table 4 reports additional downstream results as in Deschenaux & Gulcehre (2025), where PGM outperforms MDLM on all but one benchmark, with only a small gap on the latter. We evaluate models with the lm-eval-harness library (Gao et al., 2024), originally designed for autoregressive LMs and adapted here for MDLM. For multiple-choice tasks, lm-eval-harness computes the loglikelihood of each candidate answer y i given a prefix x, i.e., p(y i |x), and selects the answer with the highest score.
While lm-eval-harness uses the log-likelihood of the continuation, the NELBO objective (1) bounds the log-likelihood of the complete sequence (x, y i ). However, we only need to know which continuation achieves the highest log-likelihood, not to compute the exact log-likelihood. Using Bayes' theorem, we note that log p(
y i |x) = log p(x, y i ) -log p(x) ∝ log p(x, y i ),(16)
since log p(x) is constant with respect to y i . Therefore, we can simply evaluate the variational bound on log p(x, y i ) to select the most likely continuation y i .
this section cite: ['b13']

Section: D.6 PERFORMANCE ON LONGER CONTEXT LENGTH
Due to the high computational cost, we were unable to train models with context lengths greater than 1024. Nevertheless, we report the latency and throughput of both MDLM and PGM at a context length of 4096. As shown in Table 10, PGM remains substantially faster than MDLM in this setting.
this section cite: []

Section: E COMPUTATIONAL COSTS
This section presents the computational costs associated with the models reported in this paper. We exclude costs associated with exploratory experiments that yielded inferior results and were not included in this manuscript. latency and throughput using a single NVIDIA A100-SXM4-80GB GPU, with results reported in Table 3. We compute the mean and standard deviation over 100 batches after 2 warmup batches.
The total training duration approximately equals the per-step latency multiplied by the number of steps. Experiments with complementary masking required twice the computational resources due to larger batch sizes and gradient accumulation. Training times for 1M steps varied by dataset: approximately 22 hours for LM1B, 4.5 days for OWT, and 3.8 days for ImageNet.
Despite the current training overhead, we are confident that future work can improve the training efficiency of PGMs, thanks to their block-diagonal attention patterns, once the tokens are grouped together along the sequence-length axis.
this section cite: []

Section: E.2 INFERENCE COSTS
We evaluate the inference efficiency of PGMs compared to MDLMs and GPT-2 with KV caching. As shown in Figure 1, PGMs achieve around 5 -5.5× improvements in throughput over MDLM while reaching superior Generative Perplexity. For inference measurements, we use a single NVIDIA A100-SXM4-80GB GPU. The efficiency gain stems from the ability of PGMs to process only unmasked tokens during inference, as illustrated in Figure 2.
Table 6 compares MDLM and PGMs on the Generative Perplexity, unigram entropy, latency, and throughput. We compute the mean and standard deviation of the latency and throughput over 20 batches after two warmup batches. E.3 LICENSING Our code and model artifacts will be released under the MIT license. The OWT dataset (Gokaslan & Cohen, 2019) is available under the Apache License 2.0. We were unable to identify a specific license for the LM1B dataset (Chelba et al., 2014). The images in ImageNet remain the property of their respective copyright holders.
Algorithm 2 Simplified Sampling for PGMs denotes an MDLM trained with the complementary masking strategy discussed in Section 5.3. The row PGM k / m denotes a PGM with k encoder and m decoder layers, and we highlighted the best PGM results in gray. lsm and mean denote the logsumexp and mean queries initializations (Section 4).
Takeaway: using the same number of layers in the encoder and decoder, and data-independent queries performed best. On LM1B, our PGM reaches 1.95 lower perplexity than MDLM after 1M steps. On OWT, we grow the embedding dimension or the number of layers to outperform MDLM on OWT.
this section cite: ['b7']

Section: 
Complementary masking seems to introduce spikes in the loss, even though it did not cause the models to diverge.
where L is the length of the sequence, x i is the i-th token, and p GPT-2 Large (x i |x <i ) is the probability assigned by GPT-2 Large to token x i given the preceding tokens x <i .
Unigram Entropy Unfortunately, a low Generative Perplexity can be achieved by generating repetitive text. To catch such cases, we compute the average unigram entropy of the generated samples:
Unigram Entropy = - 1 N N i=1 v∈V c(v, x (i) ) L log c(v, x (i) ) L ,(15)
where V is the vocabulary, v is a token of the vocabulary, and c(v, x) is the empirical appearance count of the token v in the sequence x. Low unigram entropy helps us to catch degenerate generation, as shown by Dieleman et al. (2022).
this section cite: ['b15']

Section: Fréchet Inception Distance and Inception Score
On image generation tasks, we evaluate the quality of samples using the Fréchet Inception Distance (FID) (Heusel et al., 2018) and Inception Score (IS) (Salimans et al., 2016). Both metrics are computed using 50 ′ 000 images, following the standard practice.
this section cite: ['b25', 'b46']

Section: D ADDITIONAL RESULTS

this section cite: []

Section: D.1 IMPACT OF CONTEXT LENGTH ON THE EFFECTIVENESS OF COMPLEMENTARY MASKING
There are three key differences between our experiments on LM1B and OWT. First, we used different tokenizers: bert-base-uncased for LM1B and GPT2's tokenizer for OWT, following the setup of MDLM (Sahoo et al., 2024). Second, the context lengths differ significantly: 128 tokens for LM1B versus 1024 for OWT. Third, we train on different datasets that might have different characteristics.
We observed that complementary masking helps when training on OWT using a shorter context length of 128 tokens with the GPT-2 tokenizer. Indeed, after 200k training step, the MDLM with complementary masking achieved a validation PPL of 37.92, outperforming the standard MDLM, which reached 39.90. This suggests that PGMs may not need extra parameters when the sequence length is short. Exploring the use of PGMs in domains where the sequence length is short, such as modeling chemical sequences, is a promising direction for future work.
this section cite: ['b42']

Section: D.2 MDLM+SDTT VS PGM+SDTT
The precision of logits during sampling can have a significant effect on sample quality, as noted in Appendix C.4. Hence, we cast all logits to FP64 prior to sampling, unlike the original MDLM and SDTT implementations.
this section cite: []

Section: References
Ref_id:b0 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b1 Title: Structured denoising diffusion models in discrete state-spaces Year: (2023)
Ref_id:b2 Title: Layer normalization Year: (2016)
Ref_id:b3 Title: Halton scheduler for masked generative image transformer Year: (2025)
Ref_id:b4 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b5 Title: Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design Year: (2024)
Ref_id:b6 Title: Masked generative image transformer Year: (2022)
Ref_id:b7 Title: One billion word benchmark for measuring progress in statistical language modeling Year: (2014)
Ref_id:b8 Title: Are more llm calls all you need? towards scaling laws of compound inference systems Year: (2024)
Ref_id:b9 Title: Specmaskgit: Masked generative modeling of audio spectrograms for efficient audio synthesis and beyond Year: (2024)
Ref_id:b10 Title: Vision transformers need registers Year: (2024)
Ref_id:b11 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b12 Title: Promises, outlooks and challenges of diffusion language modeling Year: (2024)
Ref_id:b13 Title: Beyond autoregression: Fast llms via self-distillation through time Year: (2025)
Ref_id:b14 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b15 Title: Continuous diffusion for categorical data Year: (2022)
Ref_id:b16 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b17 Title: Distilling diversity and control in diffusion models Year: (2025)
Ref_id:b18 Title: The language model evaluation harness Year: ()
Ref_id:b19 Title: Discrete flow matching Year: (2019)
Ref_id:b20 Title: Think before you speak: Training language models with pause tokens Year: (2024)
Ref_id:b21 Title: Reviving any-subset autoregressive models with principled parallel sampling and speculative decoding Year: (2025)
Ref_id:b22 Title: Algorithm 247: Radical-inverse quasi-random point sequence Year: (1964)
Ref_id:b23 Title: Edit flows: Flow matching with edit operations Year: (2025)
Ref_id:b24 Title: Efficient perplexity bound and ratio matching in discrete diffusion language models Year: (2025)
Ref_id:b25 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2018)
Ref_id:b26 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b27 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b28 Title: The curious case of neural text degeneration Year: (2020)
Ref_id:b29 Title: Any-order flexible length masked diffusion Year: (2025)
Ref_id:b30 Title: Adam: A method for stochastic optimization Year: (2017)
Ref_id:b31 Title: Variational diffusion models Year: (2023)
Ref_id:b32 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2024)
Ref_id:b33 Title: Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models Year: (2025)
Ref_id:b34 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b35 Title: Scaling up masked diffusion models on text Year: (2025)
Ref_id:b36 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2025)
Ref_id:b37 Title: Sigma-gpts: A new approach to autoregressive models Year: (2024)
Ref_id:b38 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b39 Title: Efficiently scaling transformer inference Year: (2022)
Ref_id:b40 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b41 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2023)
Ref_id:b42 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b43 Title: The diffusion duality, 2025a Year: ()
Ref_id:b44 Title: Esoteric language models Year: (2025)
Ref_id:b45 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b46 Title: Improved techniques for training gans Year: (2016)
Ref_id:b47 Title: Simple guidance mechanisms for discrete diffusion models Year: (2025)
Ref_id:b48 Title: Simplified and generalized masked diffusion for discrete data Year: (2025)
Ref_id:b49 Title: Training and inference on any-order autoregressive models the right way Year: (2022)
Ref_id:b50 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b51 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015-07)
Ref_id:b52 Title: Generative modeling by estimating gradients of the data distribution Year: (2020)
Ref_id:b53 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b54 Title: Attention is all you need Year: (2023)
Ref_id:b55 Title: Phenaki: Variable length video generation from open domain textual description Year: (2022)
Ref_id:b56 Title: Generalized interpolating discrete diffusion Year: (2025)
Ref_id:b57 Title: Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding Year: (2025)
Ref_id:b58 Title: An empirical analysis of compute-optimal inference for problem-solving with language models Year: (2024)
Ref_id:b59 Title: Xlnet: Generalized autoregressive pretraining for language understanding Year: (2020)
Ref_id:b60 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b61 Title: Magvit: Masked generative video transformer Year: (2023)
Ref_id:b62 Title: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling Year: (2025)
Ref_id:b63 Title: Distilling masked diffusion models into one-step generator Year: (2025)
