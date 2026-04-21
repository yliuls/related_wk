Title: MokA: Multimodal Low-Rank Adaptation for MLLMs
Abstract: In this paper, we reveal that most current efficient multimodal fine-tuning methods are hindered by a key limitation: they are directly borrowed from LLMs, often neglecting the intrinsic differences of multimodal scenarios and even affecting the full utilization of all modalities. Inspired by our empirical observation, we argue that unimodal adaptation and cross-modal adaptation are two essential parts for the effective fine-tuning of MLLMs. From this perspective, we propose Multimodal low-rank Adaptation (MokA), a multimodal-aware efficient fine-tuning strategy that takes multimodal characteristics into consideration. It compresses unimodal information by modality-specific parameters while explicitly enhancing crossmodal interaction, ensuring both unimodal and cross-modal adaptation. Extensive experiments cover three representative multimodal scenarios (audio-visual-text, visual-text, and speech-text), and multiple LLM backbones (LLaMA2/3, Qwen2, Qwen2.5-VL, etc). Consistent improvements indicate the efficacy and versatility of the proposed method. Ablation studies and efficiency evaluation are also conducted to fully asses our method. Overall, we think MokA provides a more targeted solution for efficient adaptation of MLLMs, paving the way for further exploration. The project page is at https://gewu-lab.github.io/MokA.

Section: Introduction
Large language models (LLMs) have gained remarkable popularity due to their impressive ability to understand and generate content. To extend their capabilities to more general multimodal scenarios, recent advancements of Multimodal Large Language Models (MLLMs) [38,40,20] have focused on aligning other modalities, such as images, with text tokens, thereby equipping LLMs with the ability to interpret and process content of other modalities. However, due to the massive parameter scale of LLMs, fully fine-tuning such models on downstream tasks is computationally prohibitive and inefficient in most cases.
A promising direction has emerged in the field of LLM fine-tuning before, which involves selectively updating a subset of parameters rather than the full model. These Parameter-Efficient Fine-Tuning (PEFT) strategies have seen widespread adoption and have been successfully extended to the finetuning of MLLMs. In particular, LoRA [12] and its variants, which assume that over-parameterized models in fact reside on a low intrinsic dimension, have been broadly applied [6,7,39], demonstrating strong adaptability and efficiency. However, the development of efficient multimodal LLM finetuning is at present obscured by a "dark cloud": most current methods are directly borrowed from LLMs, often overlooking the fundamental differences of multimodal scenarios. Indeed, prior studies
this section cite: ['b37', 'b39', 'b19', 'b11', 'b5', 'b6', 'b38']

Section: LLM (LoRA🔥)
Audio token Text token Visual token
this section cite: []

Section: LLM Tokenizer
Question: what is the baby doing?  in multimodal learning have demonstrated that the inherent heterogeneity of different modalities necessitates modality-specific utilization strategies, rather than a fully unified way [24,34].
To this end, we are motivated to observe the fine-tuning efficacy of the widely used LoRA strategy. A common MLLM fine-tuning framework is shown in Figure 1a: encoded representation of non-text modality (e.g., audio or visual) is first aligned with the text embedding space via a projector (usually Q-former or MLP), after which the resulting multimodal tokens are integrated and processed jointly by the LLM. In the efficient fine-tuning case, the LLM backbone is frozen, and parameters of additional LoRA modules are optimized. Figure 1b provides the sketch of classic LoRA module. A and B matrices are shared across different modalities.
To further observe how well tokens of different modalities are utilized, we conduct partial modality inference experiments. The training stage retains the original setting, wherein all multimodal tokens are processed by the LoRA module. Specifically, we evaluate the model's performance when only tokens from a selected modality are passed through the LoRA adaptation pathway at the prefilling stage during inference. As illustrated in Figure 1c, visual token inference is used as an example. And it should be noted that the pre-trained weights still receive full tokens of all modalities. Results shown in Figure 1d-1f demonstrate a surprising phenomenon across three representative multimodal scenarios, audio-visual-text case, visual-text case, and speech-text case. Text token inference can achieve quite comparable performance to the regular full modalities case. However, Non-text token inference (e.g., audio or visual) leads to a noticeable drop in performance.
The above results suggest that the optimization of all-modality-shared LoRA parameters is overly influenced by text tokens, resulting in non-text tokens being less effectively utilized during finetuning. Although these all-modality-shared parameters implicitly improve cross-modal interaction, this phenomenon reveals the need to consider individual modality during fine-tuning. This fact inspires us that unimodal and cross-modal adaptation are equally critical in the fine-tuning of MLLMs, which is mostly ignored as mentioned above.
To this end, we propose the Multimodal low-rank Adaptation (MokA), a fine-tuning strategy designed to achieve unimodal adaptation while explicitly enhancing cross-modal interaction. While MokA retains the widely adopted low-rank decomposition matrices, it redefines the roles of matrices A and B to better accommodate multimodal characteristics. Specifically, matrix A is designed to be modality-specific, allowing each modality to compress information independently and thus avoid interference from others. After that, a cross-attention mechanism is introduced to strengthen the interaction between text tokens and non-text tokens, emphasizing task-relevant features. Finally, a shared multimodal matrix B projects the unimodal low-rank representations into a unified space, facilitating effective alignment across modalities. These three parts jointly ensure both unimodal and cross-modal adaptation. In experiments, noticeable improvement in multiple multimodal scenarios demonstrates the effectiveness of our method. We think MokA represents a first-step attempt at multimodal-aware adaptation, and further possibilities exist under our basis that simultaneously accounts for both unimodal and cross-modal adaptation.
this section cite: ['b23', 'b33']

Section: Method

this section cite: []

Section: Rethinking of low-rank adaptation in the multimodal scenario
LoRA [12] is based on the assumption that the weight updates during fine-tuning lie in a subspace of low "intrinsic rank." Rather than updating the entire pre-trained weight matrix directly, LoRA introduces a low-rank decomposition approach, where the update ∆W ∈ R d×k to a pre-trained matrix W 0 ∈ R d×k is parameterized as the product of two much smaller matrices: B ∈ R d×r and A ∈ R r×k , with r ≪ min(d, k). The resulting fine-tuned weight matrix W ′ is given by W 0 + ∆W = W 0 + BA. Therefore, for h = W 0 x, the modified update forward pass yields:
h = W 0 x + ∆W x = W 0 x + BAx.(1)
Here, W 0 remains fixed during training, while only the matrices A and B are learned. To ensure stable training, A is initialized using a uniform Kaiming distribution [11], and B is initialized to zero, leading to an initial update ∆W = BA = 0 at the beginning of fine-tuning. LoRA [12] and its variants have been extensively employed in the parameter-efficient fine-tuning of MLLMs [6,7,39]. These methods typically employ shared parameters to uniformly process tokens from all modalities, implicitly facilitating cross-modal interactions during adaptation. However, our empirical results reveal that such shared tuning leads to limited utilization of all modalities. This highlights the need to consider individual modality during fine-tuning.
To better support multimodal adaptation, we argue that both unimodal and cross-modal updates should be considered during fine-tuning. In other words, the model should be able to learn from each modality independently while also ensuring the cross-modal interaction. Therefore, the design of the update mechanism should ensure that both types of information are properly captured during the forward pass:
h = W 0 x + ∆W x = W 0 x + ∆W [x m1 ; x m2 ; • • • ; x mn ], (2
) = W 0 x + [∆W 1 x m1 ; ∆W 2 x m2 ; • • • ; ∆W n x mn ] unimodal adaptation + ∆W cross [x m1 ; x m2 ; • • • ; x mn ] cross-modal adaptation , (3
)
where n is the number of modalities. x mi is the token sequence of modality i. ∆W i is the unimodal update parameters of modality i, and ∆W cross is cross-modal update parameters.
this section cite: ['b11', 'b10', 'b11', 'b5', 'b6', 'b38']

Section: Multimodal low-rank Adaptation (MokA)
Pretrained Weights Based on the above perspective, we propose Multimodal low-rank Adaptation (MokA) strategy, a parameterefficient fine-tuning method tailored for the multimodal nature of MLLMs. Considering the efficiency advantage of LoRA, MokA retains the core idea of low-rank adaptation, but redefines the roles of the projection matrices A and B
this section cite: []

Section: Multimodal B

this section cite: []

Section: Cross Attention
to better reflect the characteristics of multimodal scenarios. By unimodal compression and explicitly reinforcing crossmodal interaction, MokA enables both unimodal and crossmodal adaptation, leading to more effective fine-tuning of MLLMs. The overall structure of MokA is depicted in Figure 2.
Concretely, MokA has three core parts: unimodal matrix A, task-centric cross-attention, and shared multimodal matrix B. Here we take the audio-visual-text case as an example, and other cases can be well extended.
this section cite: []

Section: Unimodal matrix A
For an arbitrary pretrained weight W 0 in the LLMs, we suppose its input sequence is
x = [x a 1 ; x a 2 ; • • • ; x a Na ; x v 1 ; x v 2 ; • • • ; x v Nv ; x t 1 ; x t 2 ; • • • ; x t Nt ].
Here {N i } i∈{a,v,t} is the token length of modality i. x a 1 is the first token of modality a, and so on. For simplicity, we use x i to denote the token sequence of modality i. Then the whole input sequence can be rewritten as x = [x a ; x v ; x t ].
To ensure the well compression of unique unimodal information and avoid the interruption from others, matrix A is designed to be individual for each modality, allowing tokens from different modalities to be processed independently through their respective parameter. The compressed sequence after matrix A is:
Ax = [A a x a ; A v x v ; A t x t ],(4)
where {A i } i∈{a,v,t} is the parameter of modality i. After processing by unimodal matrix A, embeddings of each modality are individually mapped into a low-rank space, without the potential influence of other modalities.
this section cite: []

Section: Task-centric cross-attention
In the fine-tuning process of MLLMs, text and non-text tokens typically serve distinct roles. Specifically, under supervised instruction tuning, text tokens often function as task descriptions or prompts, whereas non-text tokens (e.g., audio or visual inputs) primarily convey contextual information upon which the task is based. The following example illustrates a typical instruction format:
<audio> <visual> Please answer the question: which clarinet makes the sound first? In this case, <audio> and <visual> provide the event information. "Please answer the question: which clarinet makes the sound first?" describes the concrete task for LLMs. Successfully answering such questions relies on effectively capturing the semantic association between the task description conveyed by text tokens and the event cues provided by non-text tokens. Therefore, it becomes intuitive and necessary to explicitly emphasize the most relevant cross-modal information to support accurate reasoning. Since unimodal information has been extracted individually after the processing of unimodal matrices A, this stage is well-suited for introducing cross-modal interaction. Additionally, as the token embeddings are projected into a low-rank space, the computational burden of performing cross-modal interaction is significantly reduced. Hence, we place the cross-attention part after the low-rank compression to ensure both effectiveness and efficiency. The concrete attention mechanism is illustrated in Figure 3, and is conducted as follows:
Att A a x a , A t x t , A t x t = softmax (A a x a )(A t x t ) ⊤ √ r A t x t ,(5)
Att A v x v , A t x t , A t x t = softmax (A v x v )(A t x t ) ⊤ √ r A t x t ,(6)
where r is the rank. Then, the enhanced audio and visual tokens are:
A a x a + λ a Att A a x a , A t x t , A t x t ,(7)
A v x v + λ v Att A v x v , A t x t , A t x t ,(8)
where λ a and λ v are the hyperparameters that control the strength of explicit cross-modal interaction. Finally, the sequence after cross-attention is:
Ax = [A a x a + λ a Att a,t,t ; A v x v + λ v Att v,t,t ; A t x t ].(9)
Here we use Att i,t,t to simply denote the cross-attention between modality i and text. It should be noted that while we adopt a cross-attention module to explicitly enhance the interaction between text and non-text tokens, alternative designs that serve a similar purpose can also be considered. Further discussion is provided in Section 4.4. In addition, in MokA, linear projections (W q , W k , and W v ) are not included in the cross-attention module, since low-rank matrices A of each modality actually can be considered as the linear projection in attention in this case. We also provide more discussion and comparison in Appendix B.
this section cite: []

Section: Shared multimodal matrix B
After unimodal compression and explicit cross-modal interaction enhancement, it becomes crucial to project the resulting unimodal representations into a shared space to facilitate cross-modal alignment. To this end, a shared multimodal matrix B is employed to perform this projection. The final output of the MokA pathway is thus given by:
BAx = [B(A a x a + λ a Att a,t,t ); B(A v x v + λ v Att v,t,t ); BA t x t ].(10)
this section cite: []

Section: Overview
In conclusion, in MokA, for a pretrained weight matrix W 0 ∈ R d×k , its update ∆W ∈ R d×k is parameterized as the product of much smaller matrices: B ∈ R d×r and {A i ∈ R r×k } i∈{a,v,t} , with r ≪ min(d, k). For input sequence x, the forward pass yields:
h = W 0 x + ∆W x = W 0 x + ∆W [x a ; x v ; x t ],(11)
= W 0 x + [B(A a x a + λ a Att a,t,t ); B(A v x v + λ v Att v,t,t ); BA t x t ],(12)
= W 0 x + [BA a x a ; BA v x v ; BA t x t ] unimodal adaptation + [λ a BAtt a,t,t ; λ v BAtt v,t,t ; 0 Nt ] cross-modal adaptation ,(13)
where 0 Nt denotes the zero vector of dimension N t , since text-token remains unchanged after crossattention. During fine-tuning, W 0 remains unchanged, with A i and B being subject to optimization. Also, A i is initialized using the uniform Kaiming distribution [11], while B is initialized to zero. It leads to an initial update ∆W = 0 at the beginning of fine-tuning, to provide a smooth starting point.
Based on Equation 13, MokA ensures both unimodal and cross-modal adaptation, offering a more tailored solution for fine-tuning MLLMs.
3 Training and evaluation details
this section cite: ['b10']

Section: Implement details
Our framework follows the common MLLM framework as illustrated in Figure 1a, but with MokA strategy. Text input is processed by the corresponding LLM tokenizer, and non-text input is first encoded by its encoder, and then aligned with the text embedding space via a projector. Here we use Q-former followed by a two-layer MLP as the projector. Finally, all tokens are fed into LLM.
For the visual branch of audio-visual-text and visual-text scenarios, we use CLIP-ViT/L-14 [25] as the visual encoder to extract the last layer patch level embedding of each frame or image. For the audio branch of the audio-visual-text scenario, we use the BEATs [5] encoder to extract features. For the speech branch of the speech-text scenario, OpenAI's Whisper model [26] is used. The number of query tokens in Q-Former of all branches is 32.
this section cite: ['b24', 'b4', 'b25']

Section: Training procedure and benchmarks
Our experiment of MLLM follows the widely used two-stage training paradigm: pre-training stage that aims to cross-modal alignment and supervised instruction-tuning for downstream tasks.
Pre-training: LLM backbone is frozen. Projectors are trainable for cross-modal alignment. For the visual branch of audio-visual-text and visual-text scenarios, trainable modules are trained on video-LLaVA [19] dataset, including the video captioning and the image captioning tasks. For the audio branch of the audio-visual-text scenario, trainable modules are trained on AudioCaps [14] dataset on the audio captioning task. For the speech branch of the speech-text scenario, trainable modules are trained on GigaSpeech-M [4] dataset on the speech recognition task. During pre-training, using the AdamW optimizer with a cosine learning rate schedule. The initial learning rate is 1e -4 with a warmup ratio of 0.03.
Instruction-tuning: At this stage, we train the model on downstream tasks in different scenarios.
Trainable parameters include all projectors and our MokA module. For the audio-visual-text case, the model is fine-tuned on the train set of MUSIC-AVQA [17], and AVE [30], respectively. For the visual-text case, the model is fine-tuned on the LLaVA-Instruct-150K [20] and a 12k subset of  A-OKVQA. For the speech-text case, the model is fine-tuned on the LibriSpeech [23], using the annotations provided by [28]. Rank of low-rank matrices is 4. The remaining settings are the same as the first stage.
Inference: To well assess the effectiveness of our fine-tuning strategy, we evaluate our trained models on in-domain test sets or public benchmarks. Details are provided in the supplementary materials.
• Audio-visual-text: in-domain test set of MUSIC-AVQA and AVE dataset.
• Visual-text: public benchmarks: MME percep [9], MMBench [22], POPE [18], SEED-Bench [16].
• Speech-text: public benchmarks: MMAU mini-speech [27] as well as the foundation subset of AIR-Bench speech-en [37].
Large Language Model. For all three cases, LLaMA-2-7b-Chat [31], LLaMA-3-8B-Instruct [10], and Qwen2-7B-Instruct [36] are used as the LLM base model, respectively. For the audio-visual-text case, Qwen2.5-VL-7B-Instruct [2] is also used as the LLM base model. Throughout the training process, weights of LLM are kept frozen. More experiments of Qwen3 are provided in Appendix A.
this section cite: ['b18', 'b13', 'b16', 'b29', 'b19', 'b22', 'b27', 'b8', 'b21', 'b17', 'b15', 'b26', 'b36', 'b30', 'b9', 'b35', 'b1']

Section: Experiments

this section cite: []

Section: Audio-visual-text scenario
To validate the effectiveness of our MokA fine-tuning strategy, we compare it with LoRA [12] and its variants, including multiple LoRA, LoRAMoE [8], DoRA [21], HydraLoRA [29], Uni-modal LoRA [1]. In addition, we also compare with two additional baselines, whose frameworks are provided in Figure 4. Concretely, the Uni LoRA + MM LoRA strategy employs unimodal low-rank matrices A to extract unimodal information independently, while incorporating an additional fully shared multimodal LoRA module to implicitly promote cross-modal interaction. The Uni LoRA + MM LoRA + Gate variant further introduces a gating mechanism to dynamically integrate the outputs of the Uni LoRA and MM LoRA branches for improved fusion. These two baselines incorporate our multimodal-aware basis that ensures both unimodal and cross-modal adaptation, but involve more parameters and offer limited cross-modal interaction. Based on the results in Table 1, we can have the following observations:
Our proposed MokA method achieves the superior overall performance across multiple audio-visualtext datasets, consistently outperforming other baselines and compared methods. While MokA introduces a slight increase in parameter scale compared to standard LoRA, this does not account for the observed performance improvements. Based on the Table 1, multiple LoRA, a baseline that uses 3 A matrices and B A matrices, underperforms both standard LoRA and MokA. Simply increasing the number of low-rank matrices does not necessarily lead to better fine-tuning performance. This suggests that MokA's advantage stems not from parameter quantity, but from its insurance for both unimodal and multimodal adaptation.
The mentioned two baselines, Uni LoRA + MM LoRA and Uni LoRA + MM LoRA + Gate, achieve competitive results. These results further support the validity of our multimodal-aware basis that unimodal and cross-modal adaptation are both essential for the fine-tuning of MLLMs. However, despite their effectiveness, MokA achieves superior results with fewer parameters and further enhanced cross-modal interactions.
In addition, Qwen2.5VL with LoRA outperforms both LLaMA2 and Qwen2 under the LoRA finetuning setting. But when using MokA, the performance of Qwen2.5VL is slightly lower than that of LLaMA2 and Qwen2. A possible reason is that the official visual connector in Qwen2.5VL,
this section cite: ['b11', 'b7', 'b20', 'b28', 'b0']

Section: Partial modality inference of MokA
To further examine how effectively MokA leverages tokens from different modalities, we also conduct partial modality inference experiments where only tokens from a selected modality are passed through the LoRA adaptation pathway at the first generation during inference. It should be noted that the evaluated model is MokA w/o cross-attention, as cross-attention computation requires the presence of both text and non-text tokens. The results, presented in Figure 5, show that MokA w/o crossattention significantly enhances the utilization of individual modalities compared to LoRA (as shown in Figure 1d-1f). These findings highlight that the multimodal-aware design of MokA facilitates more effective use of all available modalities.
this section cite: []

Section: Cross-modal interaction variants
In the original MokA framework, cross-attention is employed to explicitly strengthen the interaction between text and non-text tokens, thereby facilitating improved cross-modal adaptation. As previously discussed, alternative modules that similarly enhance this interaction can also be considered. In this Table 4: Evaluation results of MokA and variants on audio-visual-text and visual-text cases. Results are based on LLaMA2. Method Music-AVQA AVE MME percep MMBench POPE SEED-Bench LoRA 73.41 69.84 908.52 50.64 70.28 39.71 Multiple LoRA 72.66 71.77 882.87 49.83 68.20 38.44 Cross-attention* 74.94 72.59 955.18 51.25 72.94 39.91 Naive interaction 75.04 73.18 996.73 51.49 73.52 40.17 MokA 75.71 74.68 1025.86 52.74 74.23 40.45
section, we explore several variants of the cross-modal interaction module, as summarized in Table 4.
The cross-attention* variant also adopts a cross-attention mechanism; however, it uses text tokens as queries. Consequently, the updated text tokens integrate information from the relevant non-text tokens-reversing the direction of interaction compared to the original MokA. The naive interaction variant performs a simple, uniform mapping from text tokens to non-text tokens without employing any attention mechanism.
Experimental results show that all proposed variants outperform the LoRA baseline, demonstrating the general effectiveness of explicitly enhancing cross-modal interactions. However, the cross-attention* variant performs slightly worse than the others. One possible explanation is that, unlike in other variants where text tokens remain unchanged, this variant alters text tokens by integrating non-text features. Although cross-modal interaction is enhanced, the modification of text representations may adversely affect language modeling capabilities. In addition, while naive interaction yields competitive results, MokA achieves further improvement through its dynamic attention mechanism. These findings suggest that the core idea of explicitly reinforcing cross-modal interactions is beneficial, and the effectiveness is not restricted to one specific module design. To thoroughly validate the efficacy of our method, we conduct ablation studies across all three multimodal scenarios. Results are shown in Table 5.
this section cite: []

Section: Ablation study
Based on the results, even without the cross-attention module, MokA w/o CA outperforms the LoRA baseline, demonstrating the effectiveness of enhancing unimodal adaptation. Furthermore, the introduction of the cross-attention module leads to additional performance improvements, indicating the benefit of explicitly enhancing cross-modal adaptation. These results indicate the necessity of each part in MokA. To enable a more comprehensive comparison, we further evaluate the proposed MokA and LoRA baselines on the proportion of trainable parameters in the full model, and inference latency. As reported in Table 6, although MokA introduces additional parameters due to the inclusion of more low-rank matrices, the increase is quite modest compared to the full LLM. Also, despite MokA incurs a slight increase in inference latency compared to standard LoRA, it achieves a notable performance gain of 3.95% on the POPE benchmark. These results suggest that the additional computational cost introduced by MokA is acceptable, and the performance improvement is considerable. More detailed efficiency evaluations are provided in Appendix D.
this section cite: []

Section: Efficiency evaluation

this section cite: []

Section: Related works
MLLMs built upon powerful LLM backbones are increasingly demonstrating impressive capabilities across diverse downstream tasks [33,13]. However, fine-tuning these models remains computationally expensive, prompting growing interest in parameter-efficient fine-tuning (PEFT) techniques that reduce memory and storage overhead during adaptation. Among them, LoRA has emerged as a widely adopted, and researchers have proposed several variants to further improve its efficiency and flexibility [8,21,29,1]. For instance, LoRAMoE [8] introduces multiple LoRA heads combined via a gating mechanism, while DoRA [21] focuses solely on optimizing the gradient direction, enabling more efficient updates. Despite these advancements, most PEFT strategies for MLLMs are direct extensions of LLM techniques and fail to account for the inherent characteristics of multimodal learning. To address this gap, we propose MokA, a fine-tuning strategy specifically designed for MLLMs. It explicitly ensures both unimodal and cross-modal adaptation to better preserve unimodal representations and enhance cross-modal interaction, offering a targeted solution for efficient and effective multimodal adaptation.
this section cite: ['b32', 'b12', 'b7', 'b20', 'b28', 'b0', 'b7', 'b20']

Section: Discussion
In this paper, we argue that both unimodal adaptation and cross-modal adaptation are essential parts for the effective fine-tuning of MLLMs, yet have largely been neglected before. To this end, we propose Multimodal low-rank Adaptation (MokA) for efficient multimodal fine-tuning. MokA redefines the roles of low-rank matrices A and B, ensuring unimodal information is preserved while enhancing cross-modal interaction by cross-attention. We think MokA is a preliminary step toward multimodal-aware adaptation, highlighting the potential for future extensions that jointly consider both unimodal and cross-modal adaptation.
Justification: In Section 4.6, we discuss the efficiency and efficacy of MokA and baselines.
MokA is with a slight increase in parameters and inference latency, but also brings noticeable improvement.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras Year: (2025)
Ref_id:b1 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b2 Title: Model composition for multimodal large language models Year: (2024)
Ref_id:b3 Title: Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio Year: (2021)
Ref_id:b4 Title: Beats: Audio pre-training with acoustic tokenizers Year: (2022)
Ref_id:b5 Title: Llava-mole: Sparse mixture of lora experts for mitigating data conflicts in instruction finetuning mllms Year: (2024)
Ref_id:b6 Title: Salm: Speech-augmented language model with in-context learning for speech recognition and translation Year: (2024)
Ref_id:b7 Title: Alleviate world knowledge forgetting in large language models via moe-style plugin Year: (2023)
Ref_id:b8 Title: Mme: A comprehensive evaluation benchmark for multimodal large language models Year: (2024)
Ref_id:b9 Title: The llama 3 herd of models Year: (2024)
Ref_id:b10 Title: Delving deep into rectifiers: Surpassing human-level performance on imagenet classification Year: (2015)
Ref_id:b11 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b12 Title: Cmsl: Cross-modal style learning for few-shot image generation Year: (2025)
Ref_id:b13 Title: Audiocaps: Generating captions for audios in the wild Year: (2019)
Ref_id:b14 Title: Reformer: The efficient transformer Year: (2020)
Ref_id:b15 Title: Seed-bench: Benchmarking multimodal llms with generative comprehension Year: (2023)
Ref_id:b16 Title: Learning to answer questions in dynamic audio-visual scenarios Year: (2022)
Ref_id:b17 Title: Evaluating object hallucination in large vision-language models Year: (2023)
Ref_id:b18 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b19 Title: Visual instruction tuning Year: (2023)
Ref_id:b20 Title: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b21 Title: Mmbench: Is your multi-modal model an all-around player? Year: (2024)
Ref_id:b22 Title: Librispeech: an asr corpus based on public domain audio books Year: (2015)
Ref_id:b23 Title: Balanced multimodal learning via on-the-fly gradient modulation Year: (2022)
Ref_id:b24 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b25 Title: Robust speech recognition via large-scale weak supervision Year: (2023)
Ref_id:b26 Title: A massive multi-task audio understanding and reasoning benchmark Year: (2024)
Ref_id:b27 Title: Towards generic hearing abilities for large language models Year: (2023)
Ref_id:b28 Title: Hydralora: An asymmetric lora architecture for efficient fine-tuning Year: (2024)
Ref_id:b29 Title: Audio-visual event localization in unconstrained videos Year: (2018)
Ref_id:b30 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b31 Title: Linformer: Self-attention with linear complexity Year: (2020)
Ref_id:b32 Title: Large-scale multi-modal pre-trained models: A comprehensive survey Year: (2023)
Ref_id:b33 Title: On-the-fly modulation for balanced multimodal learning Year: (2024)
Ref_id:b34 Title:  Year: (2025)
Ref_id:b35 Title:  Year: (2024)
Ref_id:b36 Title: Air-bench: Benchmarking large audio-language models via generative comprehension Year: (2024)
Ref_id:b37 Title: mplug-owl: Modularization empowers large language models with multimodality Year: (2023)
Ref_id:b38 Title: Where visual speech meets language: Vspllm framework for efficient and context-aware visual speech processing Year: (2024)
Ref_id:b39 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
