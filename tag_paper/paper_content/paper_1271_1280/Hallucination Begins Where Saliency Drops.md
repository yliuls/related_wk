Title: HALLUCINATION BEGINS WHERE SALIENCY DROPS
Abstract: Recent studies have investigated attention dynamics in large vision language models (LVLMs), yet existing methods remain limited in reliably distinguishing hallucinated from correct outputs -primarily because they rely solely on forward-pass attention, ignoring gradient-based signals that reveal how token influence propagates through the model. To bridge this gap, we introduce LVLMs-Saliency, an gradient-aware diagnostic tool that quantifies the grounding strength of each output token by fusing attention weights with their gradients. Through analysis, we identify a decisive pattern: Hallucinations occur when prior output tokens shows low saliency to the next token prediction, indicating a failure of contextual memory. Building on this insight, we propose a dual-mechanism inference-time framework: (1) Saliency-Guided Rejection Sampling (SGRS), which dynamically filters candidate tokens during decoding by rejecting those with saliency below a context-adaptive threshold, thereby preventing coherence-breaking tokens from entering the sequence; and (2) Local Coherence Reinforcement (LocoRE), a lightweight plug-and-play module that strengthens attention from the current token to its most recent outputs, actively counteracting the "forgetting" behavior identified by LVLMs-Saliency. Experimental results demonstrate that our method significantly reduces hallucinations across multiple LVLMs, offering a robust and interpretable solution to improve model reliability. The code can be accessed in https://https://github.com/zhangbaijin/LVLMs-Saliency.

Section: INTRODUCTION
Large Vision Language Models (LVLMs) have made significant strides in cross-modal tasks. However, hallucinations remain a key challenge, particularly in visual question answering and image captioning. Current mitigation strategies such as incorporating external knowledge, retraining with additional data Li et al. (2023a); Liu et al. (2023); Park et al. (2024); Ma et al. (2025e;a;d;b;2024;2025c) or training-free methods Neo & Chen (2024); Li et al. (2025a;b); Zhang et al. (2025a); Wu et al. (2025a); Liu et al. (2024c); Gong et al. (2024); Zhou et al. (2024); Shang et al. (2024); Min et al. (2024); Liu et al. (2024b);Fang et al. (2025); Wu et al. (2025b). Although the above methods have made great progress, their interpretability is insufficient, especially without a clear explanation of the causes of hallucinations in the autoregressive generative model.
Recent studies on attention sinks have provided new perspectives for understanding hallucinations. For example, OPERA Huang et al. (2024), DOPRA Wei & Zhang (2024), PAI Liu et al. (2024d), FastV Chen et al. (2024b), EAH Zhang et al. (2024a), TAME Tang et al. (2025a) and Farsight Tang et al. (2025b) have revealed the relationship between attention sinks and hallucinations. They prove that when a token continues to attract high attention weights in subsequent tokens, this over-reliance may cause hallucinations in the model output. However, the relationship between attention maps and hallucinated tokens remains inadequately explained. This is because attention maps only reflect the
this section cite: ['b22', 'b37', 'b36', 'b9', 'b67', 'b39', 'b35', 'b8']

Section: Please describe this image in detail
The image depicts a cozy, vintage-style room with a mix of traditional and rustic elements. The room features a sloped ceiling with a striped wallpaper pattern in shades of blue, white, and red. The walls are also adorned with similar striped wallpaper. model's decision-making in the forward pass, without capturing how changes in input tokens influence the final output. Moreover, existing methods often overlook gradient information, which is essential for understanding the interdependencies among different tokens during the generation process. As illustrated in Figure 1, it is nearly impossible to discern meaningful patterns in attention maps that distinguish correct outputs from hallucinated ones. Therefore, a token-level, interpretable observation tool is essential to uncover the mechanistic origins of hallucinations in large vision-language models, revealing not just when they occur, but why and where in the generation process they emerge.
To address this limitation mentioned above, we draw inspiration from the concept of information flow introduced in "Label Words" Wang et al. (2023), which highlights how information within LLMs tends to converge on specific user-specified tokens. Adapting this insight to the autoregressive generation setting of LVLMs, we propose an unsupervised metric called LVLMs-Saliency, defined as the element-wise product of attention weights and their corresponding gradients. This measure quantifies how strongly each previously generated output token influences the prediction of the next token, offering a fine-grained, token-level view of contextual grounding -or its absence -during generation. As shown in Figure 1 and Figure 2, we observe saliency patterns in Qwen2-VL and LLaVA-1.5 that are distinct from conventional attention maps:
Pattern: Hallucinations occur when prior output tokens shows low saliency to the next token.
which reveals a breakdown in contextual grounding that attention-only methods fail to capture. When generating the correct token, the model maintains high saliency on previous related tokens, thereby ensuring the coherence of context tokens. However, hallucinations occur when the model "forgets" the past context, resulting in weak dependencies between tokens and low saliency of previous output. By the way, although there is a noticeable difference in the saliency of user prompts for correct versus hallucinated tokens, our analysis of 500 samples indicates that these saliency scores do not significantly affect the model's predictive accuracy. This finding suggests that although prompt saliency plays a role in the model's behavior, it is not the primary cause of hallucinations. Unlike previous methods of intervening in image attention (Zhang et al., 2024a;Liu et al., 2024d;Jiang et al., 2024;Tang et al., 2025a;b) to alleviate hallucinations, we focus exclusively on the dynamics of output token saliency during autoregressive generation. To mitigate hallucinations caused by context loss when the model outputs tokens, we propose a dual-intervention approach in the inference phase that incorporates saliency: Saliency-Guided Rejection Sampling (SGRS): A proactive filtering mechanism that evaluates the grounding quality of each candidate output token before it is committed to the sequence. By computing the token's saliency, SGRS rejects candidates that exhibit weak contextual dependencies (i.e., low saliency), forcing the model to resample until a contextually grounded token is selected. This directly prevents the injection of "coherence-breaking" tokens that trigger cascading hallucinations.
this section cite: ['b45', 'b13']

Section: Local Coherence Reinforcement (LocoRE):
A reactive stabilization mechanism that activates after a token is accepted. LocoRE strengthens the attention weights from the current query token to the most recent w s output tokens, using a distance-aware gain factor γ (P ) j = 1 + β • I ((P -j) ≤ w s ). This ensures that even as the sequence grows, the model maintains strong attentional links to its immediate past, counteracting the "forgetting" behavior observed in Pattern 1.
Together, SGRS and LocoRE form a closed-loop coherence preservation system: SGRS acts as a gatekeeper, blocking low-saliency tokens at the point of entry; LocoRE acts as a stabilizer, reinforcing contextual dependencies after commitment. With extensive experiments, our method demonstrates significant hallucination-mitigating performance across different LVLMs on image hallucination and generation benchmarks, proving its effectiveness. Our contributions are as follows:
• We propose LVLMs-Saliency, an unsupervised, gradient-based metric for quantifying token-level hallucination in autoregressive LVLMs. Through systematic analysis, we establish a direct causal link between low output token saliency and hallucination: when the model fails to maintain attention on recently generated tokens (Pattern 1), contextual memory collapses, leading to semantically inconsistent outputs.
• We introduce Saliency-Guided Rejection Sampling (SGRS), the first inference-time mechanism that dynamically filters candidate tokens based on their saliency with respect to prior output context. By rejecting low-saliency tokens before commitment, SGRS proactively prevents the injection of coherence-breaking elements into the generation stream -directly mitigating the root cause of context-drift hallucinations.
• We introduce Local Coherence Reinforcement (LocoRE), a lightweight, plug-and-play module that strengthens attention from the current token to its most recent w s predecessors. Unlike prior methods that rebalance cross-modal attention, LocoRE operates purely within the output stream. SGRS ensures only coherent tokens enter, LocoRE ensures they are not forgotten.
this section cite: []

Section: ANALYSIS AND MOTIVATION

this section cite: []

Section: HALLUCINATION TOKEN SALIENCY ANALYSIS
We propose a gradient-based attention analysis framework for quantifying token-level hallucination saliency in autoregressive language models. Given an input sequence x ∈ V n , where V denotes the vocabulary space and n represents the sequence length, we process x through the model M to obtain:
(y, {A (l,h) } L,H l=1,h=1 , s) = M(x),(1)
where A (l,h) ∈ [0, 1] n×n denotes the attention weight matrix at layer l ∈ {1, . . . , L} and head h ∈ {1, . . . , H}, s ∈ R |V| represents the logits corresponding to the target hallucination token, y ∈ R |V| is the model's output probability distribution. The cross-entropy loss function L : R |V| × R |V| → R + is defined as:
L(y, s) = - T t=1 y t log σ(s t ),(2)
where σ(•) denotes the softmax function and t indexes the token position in the sequence. The gradient of the loss with respect to attention matrices is computed as:
∇A (l,h) = ∂L ∂A (l,h) ∈ R n×n .(3)
The saliency matrix S (l,h) ∈ R n×n for each attention head is obtained through the Hadamard product followed by triangular masking:
S (l,h) = tril A (l,h) ⊙ ∇A (l,h) ,(4)
where tril(•) : R n×n → R n×n preserves the lower triangular portion to maintain causal structure, and ⊙ denotes element-wise multiplication. The layer-wise normalized saliency S(l) ∈ R n×n is computed by averaging across attention heads and applying ℓ 2 -normalization:
S(l) = H h=1 S (l,h) H h=1 S (l,h) 2 .(5)
As demonstrated in Figures 1, 2, and 5, our quantitative analysis reveals statistically significant differences in saliency patterns between veridical and hallucinated tokens across both Qwen2-VL-7B Yang et al. (2024) and LLaVA1.5-7B Liu et al. (2024a) architectures.
this section cite: []

Section: METHODOLOGY

this section cite: []

Section: SALIENCY-GUIDED REJECTION SAMPLING (SGRS)
SGRS dynamically evaluates the grounding quality of each candidate token before commitment; the complete algorithm is formalized in Algorithm 1. At the decoding step corresponding to absolute position P , given context x <P and image I, the model produces logits s (P ) ∈ R |V| . We sample K candidates C (P ) via top-K sampling. For each c i ∈ C (P ) , we compute its hallucination saliency S(c i ) as:
S(c i ) = 1 |L target | • |J | l∈Ltarget j∈J S(l) P,j ,(6)
where S(l) is the layer-wise normalized saliency matrix defined in Section 2.1, L target denotes the set of target layers (e.g., middle-to-deep layers), and J = {j | Sys L + Img L ≤ j < P } is the set of positions corresponding to previously generated output tokens, with Sys L = 35 and Img L = 576 for LLaVA-1.5.
A candidate is accepted only if S(c i ) ≥ τ (P ) , where the adaptive threshold is computed over the most recent W output tokens:
τ (P ) = α • 1 |H| j∈H S(x j ), H = {j ∈ J | (P -1) -j ≤ W },(7)
with α ∈ (0, 1) controlling sensitivity and W the history window size. It scales the historical average saliency to control: "How many times the saliency of the current candidate token needs to reach the historical average before it is accepted". If all candidates are rejected, we fall back to selecting the token with the highest saliency score. This mechanism directly operationalizes our finding in Pattern 1: low output-token saliency precedes hallucination. By rejecting such tokens, SGRS enforces a generation path grounded in textual context -specifically, the model's own prior outputs.
this section cite: []

Section: LOCAL COHERENCE REINFORCEMENT (LOCORE)
While SGRS ensures token-level grounding, LocoRe addresses sequence-level context drift by explicitly reinforcing attention dependencies among output tokens, the complete algorithm is formalized in Algorithm 2. Formally, at absolute position P (where
P > Sys L + Img L ), let J P = {j ∈ N | Sys L + Img L ≤ j < P }
denote the set of positions corresponding to previously generated output tokens. For the prediction of token at position P + 1, we enhance the attention weights from query P + 1 to keys in J P within a local window of size w s .
Define the distance-weighted gain for each j ∈ J P as:
γ (P ) j = 1 + β • I ((P -j) ≤ w s ) ,(8)
where β ≥ 0 is the reinforcement strength, and I(•) is the indicator function. Let A (P +1) ∈ R B×n h ×(P +1)×(P +1) denote the attention weight matrix computed during the forward pass for position P + 1. We modify the submatrix corresponding to attention from query P + 1 to keys in J P :
A (P +1) [b, h, P + 1, j] ← A (P +1) [b, h, P + 1, j] • γ (P ) j , ∀b ∈ [B], h ∈ [n h ], j ∈ J P . (9)
Equivalently, in vectorized form, let γ (P ) ∈ R |J P | be the gain vector with entries γ (P ) j , and let A (P +1) P +1,J P ∈ R B×n h ×|J P | denote the slice of attention weights from query P + 1 to keys in J P . The update is:
A (P +1) P +1,J P ← A (P +1) P +1,J P ⊙ γ (P ) ,(10)
where ⊙ denotes element-wise multiplication broadcasted over batch and head dimensions. The modified attention weights are then used in the softmax and weighted sum operations of the selfattention mechanism, ensuring that the model's prediction for token P+1 is more strongly grounded in its recent output history. This operation amplifies the influence of recent context on the prediction of token P + 1, directly countering the saliency decay observed in Pattern 1. Crucially, LocoRE operates purely on the attention structure -no gradient computation or model parameter modification is required.
Synergistic Workflow. SGRS and LocoRE operate sequentially at each decoding step: SGRS filters and selects the current token x P based on its saliency to prior outputs; LocoRE then modifies the attention weights used in the next forward pass (for position P + 1) to reinforce dependencies on recent tokens. This closed-loop design ensures that each accepted token is both well-grounded (SGRS) and unlikely to be forgotten (LocoRE).
this section cite: []

Section: Algorithm 1 SGRS
Require: M, x, K, R, α, W, L, S=35, I=576, H Ensure: xP : accepted token at position P 1: logits ← M(xinput, KV)[:, -1, :] 2: C ← TopK(softmax(logits), K), accepted ← False 3: for r = 1 to R do 4: c ∼ Sample(C) 5: S(c) ← SALIENCY(M, c, Ltarget, P, S, I) ▷ Eq. ( 1) 6:
JP ← {j | S + I ≤ j < P } ▷ Output token positions 7: HP ← {j ∈ JP | (P -1) -j ≤ W } ▷ Recent W outputs 8: τ ← α • 1 |H P | j∈H P H[j] ▷
Eq. (2) 9: if S(c) ≥ τ then 10: xP ← c, H.append(S(c)), accepted ← True, break 11: else 12: C ← C \ {c} 13: end if 14: end for 15: if not accepted then 16: xP ← arg maxc∈original C S(c) ▷ Fallback: best saliency 17: end if 18: return xP Algorithm 2 LOCORE Require: 1: A (P +1) ∈ R B×n h ×(P +1)×(P +1) : attention weights for step P + 1 2: S = 35, I = 576: system and image token lengths 3: ws: local window size, β ≥ 0: gain strength Ensure: A (P +1) : modified attention weights for step P + 1 4: P ← current position ▷ Last generated token position 5: t ← P -(S + I) 6: if t ≤ 0 then return A (P +1) 7: end if ▷ No output yet 8: JP ← {j | S + I ≤ j < P } ▷ Historical output positions 9: if JP = ∅ then return A (P +1) 10: end if 11: for all j ∈ JP do 12: dj ← P -j ▷ Distance to current position 13: γj ← 1 + β • I(dj ≤ ws) ▷ Eq. (3) 14:
for all b ∈ [B], h ∈ [n h ] do 15: 4) 16:
A (P +1) [b, h, P + 1, j] ← A (P +1) [b, h, P + 1, j] • γj ▷ Eq. (
end for 17: end for 18: return A (P +1)
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUPS
Baselines. To demonstrate the broad applicability of our method in LVLM architecture, we applied and evaluated the latest models, including LLaVA-v1.5-7/13B Liu et al. (2024a) (2025b), which aim to enhance the truthfulness of the model's output during inference by adjusting attention heads. Among these methods,reaching SOTA on the POPE dataset, and achieved significant results second only to EAH on descriptive datasets such as CHAIR. Compared with EAH's approach
this section cite: []

Section: Saliency map Visualization with LocoRE.
As shown in Figure 3, which visualizes the LVLMs-Saliency maps from prior output tokens to the current token, applying LocoRE significantly increases the saliency scores assigned to recently generated context tokens -particularly those within the local coherence window. This demonstrates that LocoRE effectively strengthens the model's dependency on its immediate output history, counteracting the "forgetting" behavior observed in the baseline. The saliency boost under LocoRE confirms our design principle: by explicitly reinforcing attention to recent outputs, the model maintains stronger contextual links during autoregressive generation. This prevents the decay of intra-output saliency that leads to hallucinations, ensuring that each new token remains grounded in its textual predecessors.
this section cite: []

Section: ABLATION STUDY ON KEY HYPERPARAMETERS
We evaluate α (SGRS) and β (LocoRE) on both CHAIR and POPE benchmarks. As shown in Table 3 and Figure 4, our full method (α = 0.6, β = 0.15) reduces CHAIR hallucination rate by 28.3% (LLaVA-1.5) and 22.8% (Qwen2-VL) compared to baseline. SGRS alone (α = 0.6, β = 0.0) contributes most of the improvement, but LocoRE adds further gains (e.g., POPE F 1 -score from 85.4% to 86.9% in LLaVA-1.5). Increasing α to 0.9 yields marginal improvement at high latency cost (+33%). We recommend α = 0.6, β = 1.2 as the optimal balance. While increasing α to 0.9 further reduces hallucination rates (CHAIR S : 35.6% → 30.0%; POPE: 87.0% → 87.1%), it incurs a 33% α β LLaVA-1.5 Qwen2-VL-7B CHAIR POPE CHAIR POPE S↓ I↓ F1↑ Acc↑ S↓ I↓ F1↑ Acc↑ 0.0 0.0 48.0 13.9 85.4 84.0 25.0 7.3 86.6 87.6 0.0 0.15 38.4 10.2 86.9 87.3 ----0.0 0.20 - ---23.5 6.8 87.5 88.2 0.6 0.0 36.5 9.0 86.9 87.4 20.5 5.6 87.9 88.9 0.6 0.15 35.6 8.2 87.0 87.5 ----0.6 0.20 ----19.3 5.1 88.0 89.0 0.6 1.0 50.2 20.9 60.3 57.8 37.5 18.5 55.3 54. 6 Table 3: Ablation study on α (SGRS) and β (LocoRE). Best in bold. β: 0.15 (LLaVA-1.5), 0.20 (Qwen2-VL).
higher latency cost (30.8 ms/token → 41.2 ms/token) and risks degrading generation fluency due to over-rejection. In extreme cases, correct but moderately salient tokens may be rejected, leading to fallback-generated outputs that are less diverse or natural. We thus recommend α = 0.6 as the optimal trade-off -it suppresses 28.3%+ of hallucinations while maintaining practical inference speed and output quality.  2026) uses Grad-CAM and attention maps to visualize the interaction between images and text in complex reasoning tasks. Attention scores highlight relevant areas through forward propagation. The EAH Zhang et al. (2024a) identifies that most hallucinations stem from the attention sink pattern marked by images in the attention matrix. Based on this insight, EAH proposes a method that enhances attention heads without additional training. TAME Tang et al. (2025a) and Farsight Tang et al. (2025b) investigate the causes of hallucinations by analyzing local self-attention patterns of "anchor tokens" and defines the degree of attentional localization as the probability of token propagation.
this section cite: []

Section: CONCLUSION
In this work, we revisit the conventional explanations linking attention sinks to hallucinations and propose a saliency-based framework to complement existing analyses. Our findings reveal that hallucinations frequently correlate with weak saliency in prior output tokens. To this end, we introduce SGRS and LocoRE, a plug-and-play intervention that dynamically boosts visual attention and reinforces local coherence during text generation. Experiments confirm that LocoRE consistently improves output accuracy across various benchmarks without requiring model retraining.  While the full SGRS+LocoRE framework achieves the strongest hallucination suppression, its reliance on gradient computation introduces non-negligible latency overhead -making it less suitable for real-time applications. In practice, however, LocoRE alone serves as a highly effective compromise: as a forward-only module that manipulates attention weights in-place, it incurs <2% latency increase while still significantly mitigating context-drift hallucinations.
As shown in Figure 6, compared to prior plug-and-play methods -such as VCD Leng
Sys P Ouput Sys P Ouput Sys P Ouput Sys P Ouput The image features...The bus is... There are several other "benches"/"person" "benches" "benches" "person" "person" Attention map Saliency socre Sys P Ouput Sys P Ouput Sys P Ouput Sys P Ouput
The image features.. The bus is... The street appears to be a residential area with a few "people"/"houses" "people" "people" "houses"
this section cite: []

Section: References
Ref_id:b0 Title: Agla: Mitigating object hallucinations in large vision-language models with assembly of global and local attention Year: (2024)
Ref_id:b1 Title: Next token prediction towards multimodal intelligence: A comprehensive survey Year: (2024)
Ref_id:b2 Title: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models Year: (2024)
Ref_id:b3 Title: Object hallucination reduction via adaptive focal-contrast decoding Year: (2024)
Ref_id:b4 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b5 Title: Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms Year: (2024)
Ref_id:b6 Title: Dola: Decoding by contrasting layers improves factuality in large language models Year: (2023)
Ref_id:b7 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b8 Title: Grounding language with vision: A conditional mutual information calibrated decoding strategy for reducing hallucinations in lvlms Year: (2025)
Ref_id:b9 Title: Damro: Dive into the attention mechanism of lvlm to reduce object hallucination Year: (2024)
Ref_id:b10 Title: Vizwiz grand challenge: Answering visual questions from blind people Year: (2018)
Ref_id:b11 Title: Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation Year: (2024)
Ref_id:b12 Title: Self-introspective decoding: Alleviating hallucinations for large vision-language models Year: (2025)
Ref_id:b13 Title: Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens Year: (2024)
Ref_id:b14 Title: Why language models hallucinate Year: (2025)
Ref_id:b15 Title: See what you are told: Visual attention sink in large multimodal models Year: (2025)
Ref_id:b16 Title: Mitigating object hallucinations in large vision-language models through visual contrastive decoding Year: (2024)
Ref_id:b17 Title: Mitigating hallucination for large vision language model by inter-modality correlation calibration decoding Year: (2025)
Ref_id:b18 Title: Fine-tuning multimodal llms to follow zero-shot demonstrative instructions Year: (2023)
Ref_id:b19 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b20 Title: The hidden life of tokens: Reducing hallucination of large vision-language models via visual information steering Year: (2025)
Ref_id:b21 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b22 Title: Aligning large multi-modal model with robust instruction tuning Year: (2023)
Ref_id:b23 Title: Visual instruction tuning Year: (2024)
Ref_id:b24 Title: Reducing hallucinations in vision-language models via latent space steering Year: (2024)
Ref_id:b25 Title: Paying more attention to image: A training-free method for alleviating hallucination in lvlms Year: (2024)
Ref_id:b26 Title: Paying more attention to image: A training-free method for alleviating hallucination in lvlms Year: (2024)
Ref_id:b27 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b28 Title: Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation Year: (2024)
Ref_id:b29 Title: Controllable video generation: A survey Year: (2025)
Ref_id:b30 Title: Follow-your-creation: Empowering 4d creation through video inpainting Year: (2025)
Ref_id:b31 Title: Follow-your-click: Open-domain regional image animation via motion prompts Year: (2025)
Ref_id:b32 Title: Follow-your-motion: Video motion transfer via efficient spatial-temporal decoupled finetuning Year: (2025)
Ref_id:b33 Title: Follow-your-emoji-faster: Towards efficient, fine-controllable, and expressive freestyle portrait animation Year: (2025)
Ref_id:b34 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b35 Title: Mitigating hallucinations in large vision-language models via summary-guided decoding Year: (2024)
Ref_id:b36 Title: Vord: Visual ordinal calibration for mitigating object hallucinations in large vision-language models Year: (2024)
Ref_id:b37 Title: Mitigating dialogue hallucination for large multi-modal models via adversarial instruction tuning Year: (2024)
Ref_id:b38 Title: Object hallucination in image captioning Year: (2018)
Ref_id:b39 Title: From pixels to tokens: Revisiting object hallucinations in large visionlanguage models Year: (2024)
Ref_id:b40 Title: Massive activations in large language models Year: (2024)
Ref_id:b41 Title: Intervening anchor token: Decoding strategy in alleviating hallucinations for MLLMs Year: ()
Ref_id:b42 Title: Seeing far and clearly: Mitigating hallucinations in mllms with attention causal decoding Year: (2025)
Ref_id:b43 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b44 Title: Only: One-layer intervention sufficiently mitigates hallucinations in large vision-language models Year: (2025)
Ref_id:b45 Title: Label words are anchors: An information flow perspective for understanding in-context learning Year: (2023)
Ref_id:b46 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b47 Title: Dopra: Decoding over-accumulation penalization and re-allocation in specific weighting layer Year: (2024)
Ref_id:b48 Title: Ritual: Random image transformations as a universal anti-hallucination lever in lvlms Year: (2024)
Ref_id:b49 Title: Mitigating hallucinations in multimodal spatial relations through constraint-aware prompting Year: (2025)
Ref_id:b50 Title: Generate, but verify: Reducing hallucination in vision-language models with retrospective resampling Year: (2025)
Ref_id:b51 Title: Efficient streaming language models with attention sinks Year: (2023)
Ref_id:b52 Title: Mitigating object hallucination via concentric causal attention Year: (2024)
Ref_id:b53 Title: Qwen2 technical report Year: (2024)
Ref_id:b54 Title: A survey on multimodal large language models Year: (2023)
Ref_id:b55 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b56 Title: Unveiling and harnessing hidden attention sinks: Enhancing large language models without training through attention calibration Year: (2024)
Ref_id:b57 Title: Less is more: Mitigating multimodal hallucination from an eos decision perspective Year: (2024)
Ref_id:b58 Title: Self-correcting decoding with generative feedback for mitigating hallucinations in large vision-language models Year: (2025)
Ref_id:b59 Title: Seeing clearly by layer two: Enhancing attention heads to alleviate hallucination in lvlms Year: (2024)
Ref_id:b60 Title: From redundancy to relevance: Enhancing explainability in multimodal large language models Year: (2024)
Ref_id:b61 Title: Simignore: Exploring and enhancing multimodal large model complex reasoning via similarity computation Year: (2025)
Ref_id:b62 Title: Enhancing multimodal large language models complex reason via similarity computation Year: (2025)
Ref_id:b63 Title: What drives attention sinks? a study of massive activations and rotational positional encoding in large vision-language models Year: (2026)
Ref_id:b64 Title: Alleviating hallucinations of large language models through induced hallucinations Year: (2023)
Ref_id:b65 Title: Mca-llava: Manhattan causal attention for reducing hallucination in large vision-language models Year: (2025)
Ref_id:b66 Title: Context tokens are anchors: Understanding the repetition curse in diffusion MLLMs from an information flow perspective Year: (2026)
Ref_id:b67 Title: Mitigating modality priorinduced hallucinations in multimodal large language models via deciphering attention causality Year: (2024)
Ref_id:b68 Title: Mitigating modality priorinduced hallucinations in multimodal large language models via deciphering attention causality Year: (2025)
Ref_id:b69 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b70 Title: Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models Year: (2024)
