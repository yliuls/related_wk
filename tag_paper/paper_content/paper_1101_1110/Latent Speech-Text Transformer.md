Title: LATENT SPEECH-TEXT TRANSFORMER
Abstract: Auto-regressive speech-text models pre-trained on interleaved text tokens and discretized speech tokens demonstrate strong speech understanding and generation, yet remain substantially less compute-efficient than text LLMs, partly due to the much longer sequences of speech tokens relative to text. This modality imbalance disproportionately allocates pre-training and inference compute to speech, potentially hindering effective cross-modal alignment and slowing performance scaling by orders of magnitude. We introduce the Latent Speech-Text Transformer (LST), which aggregates speech tokens into latent speech patches that serve as higher-level autoregressive units. This design aligns the sequence-modeling granularity between speech and text while improving computational efficiency. The resulting patches can align with textual units to facilitate cross-modal knowledge transfer and compactly capture recurring acoustic patterns such as silence. Across story-completion benchmarks under both computecontrolled and data-controlled settings, LST consistently improves speech accuracy while also improving text performance, achieving up to +6.5% absolute gain on speech HellaSwag in compute-controlled training (+5.3% in data-controlled training). Under compute-controlled scaling from 420M to 1.8B parameters in a near compute-optimal regime, gains grow with scale, and improvements persist up to 7B parameters under fixed-token budgets. These benefits extend to downstream tasks: LST stabilizes ASR adaptation and reduces the effective autoregressive sequence length during ASR and TTS inference, lowering computational cost without degrading reconstruction quality. The code is available at https://github.com/facebookresearch/lst.

Section: INTRODUCTION
Inspired by the strong zero-and few-shot understanding and generation capabilities of large autoregressive textual language models with billions of parameters that are pre-trained on trillions of tokens, Lakhotia et al. (2021) introduce the task of Generative Spoken Language Modeling (GSLM) a.k.a Textless NLP, where raw speech is encoded as a sequence of discrete tokens based on a dictionary of quantized speech features, and an auto-regressive language model (LM) is trained on these tokens with Next Token Prediction (NTP). While initially successful, Cuervo & Marxer (2024) estimate that this approach would require up to three orders of magnitude more data to obtain equivalent capabilities as textual LLMs, largely owing to the same information requiring a significantly larger number of speech tokens to represent compared to text. This increased sequence length also means that these models utilize considerably more compute during inference to process the same amount of semantic content compared to text.
To improve scaling properties of large speech models by taking advantage of the comparatively larger corpus of web text compared to speech, recent efforts have leveraged transfer learning from textual modalities in the form of warm initialization from large pre-trained text models (Hassid et al., 2023), pre-training with interleaved speech-text data (Nguyen et al., 2025), and modeling ⋄Work done while at Meta. speech and text in multiple streams to leverage the textual chain of thought or "inner monologue" (Défossez et al., 2024). All these works attempted to some extent to achieve representational alignment between text and speech, where a perfect alignment means the model can treat the two modalities interchangeably without any performance difference. Despite this, there remains a large gap between text-to-text and speech-to-speech performance on the same benchmarks, highlighting the incompleteness of the alignment. We hypothesize that the severe information-density mismatch between speech and text tokens is one of the primary factors hindering speech-text alignment.
To overcome the aforementioned challenges, we introduce the Latent Speech-Text Transformer (LST) based on the byte-latent transformer (BLT) architecture (Pagnoni et al., 2024), comprising an encoder that dynamically groups sequences of speech tokens into higher-level speech patches, a global speech-transformer that auto-regressively models interleaved sequences of textual tokens and speech patches, and a light-weight transformer decoder (Vaswani et al., 2017) that maps patches back into speech tokens of dynamic sizes. Working in terms of speech patches allows the model to encode more content given the same training cost, makes inference more efficient. These speech patches can represent higher-level speech concepts or prolonged silences, and serve to level the information density between speech and text, thus making them easier to align (see Figure 1).
We first demonstrate that LST models with fixed-size speech patching schemes similar to what Yu et al. (2023) did with text, are able to significantly outperform their non-patching counterparts. Such models are aware of the internals of patches without expending much compute in the process, in contrast with methods that expand the speech token vocabulary by applying subword tokenization, which yield poor downstream performance (Cuervo & Marxer, 2024). We further improve the performance by introducing speech-patching based on textual alignment at the word/subword levels, which crucially also includes patching large sequences of silences. Since this approach requires text-speech alignment timestamps during training and inference, we also introduce a curriculumbased method to eliminate the need for such alignments during inference.
To summarize, this paper makes the following contributions:
(1) We show that LST improves performance in both data-and compute-controlled settings compared to prior interleaved speech-text models on speech versions of popular text understanding benchmarks such as HellaSwag (Zellers et al., 2019) (see Figure 1), while substantially reducing training and inference compute and enabling efficient downstream ASR and TTS transfer.
(2) We introduce latent speech patching as a unified mechanism for compressing autoregressive speech sequences and analyze static, alignment-based, and curriculum patching strategies.
(3) We demonstrate that the benefits of LST persist and grow with scale from 1B to 7B parameters, indicating improved sample efficiency and more favorable compute-optimal scaling behavior for spoken language modeling.
Collectively, these contributions demonstrate that aligning the autoregressive modeling granularity of speech and text helps mitigate a key barrier to efficient and scalable spoken language modeling. 2 BACKGROUND Generalized spoken language models (Lakhotia et al., 2021) typically comprise three components:
(1) a speech tokenizer model that maps a raw speech waverform s to a sequence of speech tokens {s 0 , . . . , s n }, (2) a decoder-only transformer model (Vaswani et al., 2017) with parameters θ that models the distribution of the next speech token given the previous context i.e. p θ (s i |s <i ), and (3) a vocoder model that maps speech token sequences back to a speech waveform, such as HiFi-GAN (Kong et al., 2020).
this section cite: ['b17', 'b7', 'b11', 'b8', 'b23', 'b34', 'b36', 'b7', 'b38', 'b17', 'b34', 'b15']

Section: Speech tokenization.
Approaches for speech tokenization include semantic tokens represented by cluster-ids obtained by k-means clustering of frame representations as in Hubert (Hsu et al., 2021), acoustic tokens obtained as discretized embeddings from residual-vector quantization bottlenecks from self-supervised neural codec models (Zeghidour et al., 2021;Défossez et al., 2024), as well as additional tokens for expressivity and also, combinations of different token categories. In this paper, we follow Hassid et al. (2023);Nguyen et al. (2025) and use Hubert tokens using a codebook of 501 speech tokens at 25Hz. Unlike Nguyen et al. (2025) we do not need to deduplicate Hubert tokens as this is organically handled by the LST architecture.
Sequence Modeling. Similar to LLMs for text, speech token modeling is typically done using a large transformer decoder model using causal self-attention, to maximize the likelihood of sequences from a large speech pre-training corpus (D) in an auto-regressive fashion:
L(D; θ) = s∈D i log p θ (s i |s <i )(1)
Interleaved Data. Since speech sequences are longer and less compact that their corresponding text sequences, such models can require several orders of magnitude more data in order to achieve performance comparable to text models (Cuervo & Marxer, 2024). In order to bridge the gap, Nguyen et al. (2025) find that training on interleaved sequences of text and speech data directly correlates with improved performance. For a subset of the pre-training dataset that contains the textual sequence {t 0 , . . . , t m }, where text tokens are obtained using a tokenizer (we use the Llama 2 tokenizer (Touvron et al., 2023) in this paper) and each text token can correspond to a span of speech tokens, the model is trained on an interleaved sequence obtained by replacing arbitrary spans of speech tokens in the sequence sequence with text tokens separated by special modality tokens. This allows the same model to be used for S→S, S→T, T→S and T→T tasks. We discuss the process of producing interleaved data from parallel text-speech data in Appendix A.1.1.
this section cite: ['b13', 'b37', 'b8', 'b11', 'b7', 'b30']

Section: LATENT SPEECH-TEXT TRANSFORMERS
The core idea of the LST architecture is to auto-regressively model latent patches of tokens (using a global transformer), rather than individual tokens, similar in spirit to BLT (Pagnoni et al., 2024) which models dynamic-sized patches of bytes. The transformation of speech/text spans to patches and vice-versa takes place with the help of a light-weight patch encoder and patch decoder, and the entire model is trained end-to-end using the same token-level likelihood as before. Figure 2 illustrates this architecture specialized to the task of speech-text modeling. The majority of the compute expended in terms of FLOPs is in the global transformer, which yields savings by operating on information-dense speech patches instead of granular speech tokens. Latent patching performs bounded-context temporal aggregation, preserving phonetic discriminability while reducing sequence-level redundancy.
Patch Encoder. Similar to BLT, the patch encoder uses a series of sliding window self-attention and cross-attention layers to aggregate token representations into patch representations. In LST, we only patch spans of speech tokens using strategies described in Section 3.1. Note that a simple alternative to patching is to use subword tokenization methods like Byte Pair Encoding (BPE) on the speech tokens. This was also explored by Cuervo & Marxer (2024) and similar to them, failed to improve performance in our experiments (ablations in Section 5). Unlike BLT, we do not use hash embeddings, as they did not provide improvements in our experiments.
this section cite: ['b23', 'b7']

Section: Patch Decoder.
A light-weight transformer is used as a decoder and trained with NTP loss, with cross-attention layers inserted between every transformer layer. Each token attends to both the previously generated speech patches and text tokens to incorporate patch-level information (using cross-attention) as well as a sliding window of the past 512 tokens (using self-attention). Further architectural details of the local patch encoder and decoder, including attention roles and initialization, are provided in Appendix A.2.1.
3.1 PATCHING Let X = [x 0 , . . . , x T ] ∈ R T ×d be speech token embeddings obtained using a learned embedding matrix applied to speech tokens {s 0 , . . . , s T } . The process of patching maps X to a shorter sequence of patch embeddings Z = [z 0 , . . . , z T ′ ] ∈ R T ′ ×d by aggregating local frame segments. For a frame-index set P i ⊆ {0, . . . , T }, a patch embedding is formed via the patch encoder:
z i = PatchEnc(X Pi ) ,
integrating the frames indexed by P i into a single patch embedding. Different patching strategies correspond to different segmentation {P i }.
Static Patching. Speech sequence is split into non-overlapping segments of a fixed length p (patch size). Each patch token is obtained by the patch encoder from the embeddings in the patch:
P i = {ip, . . . , min((i + 1)p -1, T )}.
420m 630m 810m 1.1b 1.4b 1.8b Model Sizes 27.5 30.0 32.5 35.0 37.5 40.0 42.5 45.0 HellaSwag Accuracy (%) 39.0% 46.3% 35.3% 45.7% LST (Speech) LST (Text) Baseline (Speech) Baseline (Text)
(a) Compute-optimal scaling (420M-1.8B). LST outperforms the baseline, with gains increasing at scale.  For p = 3 and input embeddings X = [x 0 , x 1 , x 2 , x 3 , x 4 , x 5 , x 6 , . . . ], the first patch is {x 0 , x 1 , x 2 }, the second {x 3 , x 4 , x 5 }, and so on. Each segment is encoded into a single patch embedding z i by the patch encoder. This provides a uniform compression ratio independent of alignment information.
Alignment Patching. To better synchronize speech and text at the semantic level, alignment patching leverages forced alignment timestamps between speech frames and textual units (e.g. words or BPE tokens). Let A = {(b k , e k )} K k=1 denotes the aligned frame ranges, where [b k , e k ] spans the k-th textual unit. The corresponding patch is
P k = {b k , . . . , e k }.
Frames outside text spans (e.g., silence) are grouped into separate patches (Fig. 3a). If consecutive words align to [2, 4] and [6, 7], patches are {x 2 , x 3 , x 4 } and {x 6 , x 7 }, with silence forming {x 0 , x 1 } and {x 5 }. We obtain alignments with Wav2Vec2+CTC (Baevski et al., 2020), yielding one patch per text unit and silence segment (Fig. 3b). While this enforces cross-modal correspondence, it requires an auxiliary model at inference, introducing possible errors. Curriculum patching (Sec. 3.1) mitigates this by gradually shifting from aligned to static patching during training.
Curriculum Patching. Curriculum patching gradually transitions from alignment-based to static patching. Let P (u) ∈ [0, 1] denote the probability of using alignment at training step u:
P (u) =      1, u < τ 1 , 1 -u-τ1 τ2-τ1 , τ 1 ≤ u < τ 2 , 0, u ≥ τ 2 .
At step u, we choose alignment patches with probability P (u) and static patches otherwise. This retains alignment benefits during early training while enabling simple static-only inference.
this section cite: ['b1']

Section: EXPERIMENTAL SETUP

this section cite: []

Section: TRAINING AND EVALUATION DATASETS
Our pre-training data comprises a mixture of text and interleaved speech datasets.
Text. Our text training data consists of web and academic corpora, sourced from a subset of the Llama 2 pre-training collection (Touvron et al., 2023), totaling 1.8T tokens. We follow the LLaMA 2 setup and apply its SentencePiece (Kudo & Richardson, 2018) BPE tokenizer with a 32K vocabulary.
Speech. Our speech training data includes speech which is discretized into HuBERT tokens (501entry codebook at 25Hz) together with paired text transcriptions. We use LibriLight (60k hours), People's Speech (30k hours), Multilingual LibriSpeech (50k hours), and Spotify (60k hours), detailed in Table 1. All corpora are aligned using the Wav2Vec2 + CTC framework to provide tokenlevel correspondence between speech and text (Figure 3b).
Table 1: Speech training datasets with total speech hours and the amount of Hubert tokens.
this section cite: ['b30', 'b16']

Section: Dataset Hours Hubert Tokens (B)
LibriLight (Kahn et al., 2020) 44,174 3.7 People Speech (Galvez et al., 2021) 14,699 1.2 Multilingual LibriSpeech (Pratap et al., 2020) 50,601 4.2 Spotify (Clifton et al., 2020a) 55,309 4.6
We evaluate the model on three benchmarks, where each dataset provides a narrative context and candidate endings, and the model selects the most plausible continuation. Together, they test narrative understanding, commonsense reasoning, and topic coherence. We evaluate the model in both speech-to-speech (S→S) and text-to-text (T→T) modes. For the speech mode, we apply Kokoro TTS model (hexgrad, 2025) to generate the speech for evaluation.
this section cite: ['b14', 'b10']

Section: sHellaSWAG (HS).
We create a speech version of HellaSwag (Zellers et al., 2019) with Kokoro TTS. This benchmark evaluates everyday commonsense reasoning with spoken inputs and outputs.
To ensure fairness, we generate the speech for prompts and responses independently and concatenate them afterwards, so that all responses are evaluated against the same speech prompt.
StoryCloze and Topic StoryCloze (SC/TSC). SC (Mostafazadeh et al., 2016) and its topic-based extension TSC (Hassid et al., 2023) are widely used in prior multimodal work (e.g., Nguyen et al. 2025) to test coherence and topic-sensitive reasoning. We resynthesize both datasets with Kokoro TTS for higher-quality speech inputs.
Table 2: Evaluation datasets for story completion (MC = Multiple Choice).
this section cite: ['b38', 'b19', 'b11']

Section: Dataset Format Focus
HellaSwag (Zellers et al., 2019) 1-in-4 MC Commonsense reasoning StoryCloze (Mostafazadeh et al., 2016) 1-in-2 MC Narrative coherence TopicStoryCloze (Hassid et al., 2023) 1-in-2 MC Topic consistency
this section cite: ['b38', 'b19', 'b11']

Section: LST MODELS AND BASELINES
LST Models. We explore four patching strategies for speech tokens:
• Static Patching. Fixed-length patches (4 HuBERT tokens) as in Yu et al. (2023), independent of alignment and consistent across training/inference.
• Aligned Patching. Uses Wav2Vec2+CTC boundaries (Fig. 3b). For each text span [b k , e k ], we form patch set P k = {b k , . . . , e k }, synchronizing speech and text tokens (Fig. 3a).
• Mixed Patching. Randomly applies static or aligned patching per sequence, combining the robustness of static patching with the fine-grained sync of aligned.
• Curriculum Patching. Training shifts from aligned (first third) to mixed (middle) to static (final), leveraging early alignment while ensuring robustness to static-only inference.
Baselines. We include two speechLLM systems as baselines:
• Base SpeechLLM. Processes speech tokens directly with text tokens, without patching, similar to SpiritLM (Nguyen et al., 2025).
• BPE SpeechLLM. Maps speech tokens into 1k BPE units using a SentencePiece tokenizer (Kudo & Richardson, 2018) trained on 100k random speech sequences, replacing speech tokens with BPE-derived unitsfoot_0 .
this section cite: ['b36', 'b16']

Section: TRAINING SETTINGS
To balance modalities, we set speech tokens to account for about one third (33%) of the total training data, while the rest (67%) is text-only. This ensures that the model benefits from large-scale text pre-  For comparison, SpiritLM (Nguyen et al., 2025) adopts a different composition: 33% pure speech, 33% interleaved, and 33% text tokens. Since SpiritLM starts from a text-pretrained model, the relatively smaller text fraction is sufficient. In contrast, when training from scratch, we find that using 33% interleaved and 66% text tokens yields better performance (see Appendix A.6).
this section cite: []

Section: RESULTS

this section cite: []

Section: PERFORMANCE UNDER CONTROLLED BUDGETS
Compute-controlled. We fix the number of training iterations and per-step sequence budget so that all methods process the same number of units (baseline tokens = LST patches). Table 3 shows three trends on HellaSwag. First, patching increases the effective token budget, benefiting both modalities: Curriculum Patching improves T→T by +5.2 (47.0→52.2) and S→S by +6.5 (39.0→45.5). Second, Aligned Patching is less effective at evaluation, since variable word spans often yield longer patches, reducing the test-time compute. Finally, Mixed and Curriculum patching combine the advantages of shorter evaluation patches with alignment information, consistently outperforming Static and Aligned across datasets.
this section cite: []

Section: Data-controlled.
Here we fix the data budget with the same amounts of speech and text tokens. Since LST compresses sequences into patches, it processes fewer patch tokens than the baselines, leading to higher efficiency. Table 4 shows that the BPE baseline fails to surpass vanilla SpeechLLM, whereas LST continues to achieve consistent gains. On HellaSwag, Curriculum Patching improves T→T accuracy from 49.6 to 52.2 despite reduced computation, while boosting S→S from 40.2 to 45.5. Similar improvements are observed on StoryCloze and TopicStoryCloze. Overall, LST with Curriculum Patching reduces the speech-text performance gap from 9.4 to 6.7, demonstrating that alignment through patching benefits both modalities while offering meaningful compute savings.
this section cite: []

Section: SCALING BEHAVIOR
Compute-Optimal Scaling. Figure 4a evaluates HellaSwag accuracy under compute-optimal training from 420M to 1.8B parameters. Following Hoffmann et al. ( 2022), text is trained with 20× model-size tokens, while speech uses half as many tokens to preserve a 2:1 text-speech ratio.
this section cite: []

Section: Sub-Optimal Token Scaling at 7B.
Figure 4b shows training dynamics for a 7B model under a fixed processed-token budget (70B tokens), which remains below the scaling-law optimal regime (≈140B). Under this sub-optimal compute setting, LST exhibits consistently faster improvement and maintains higher accuracy throughout training. Additional comparisons at 1B and 7B are summarized in Appendix A.4, where LST achieves larger gains at 1B and persistent improvements at 7B. Notably, the speech data used in our training has already been reused for multiple epochs (≈6×), indicating that further token scaling becomes data-limited rather than compute-limited in this regime. The continued upward trend at 7B nevertheless suggests that training toward the scaling-law optimum with additional data would likely further amplify LST's advantage.
this section cite: []

Section: DOWNSTREAM TRANSFER AND ANALYSIS

this section cite: []

Section: ASR Adaptation and Efficient TTS Generation.
To assess ASR, we fine-tune both models on LibriSpeech clean for 1k-4k iterations (batch size 4, sequence length 4096; Table 5). The baseline at 1k steps (140% / 202% WER) frequently hallucinates transcripts and produces unreliable stopping behavior, consistent with observations in (Nguyen et al., 2025). Although performance improves with iterations, it remains far worse even at 4k (>20% / 40% WER). In contrast, LST achieves 6.8% / 10.4% WER at 1k while reducing the context units during ASR inference. Under the same setup, we evaluate TTS reconstruction after 20k fine-tuning steps (Table 5). LST matches the baseline in CER and reduces the generation length by ∼ 4× during TTS inference. CER is computed from Whisper-based transcriptions (Radford et al., 2023). Together, LST enables faster ASR adaptation and preserves TTS quality while requiring fewer context units and decoding steps at inference.
this section cite: ['b26']

Section: Visualization of Word-Level Speech Patch Embeddings
We visualize word-level speech patch embeddings using t-SNE (van der Maaten & Hinton, 2008) (Fig. 5) from the aligned-patching LST model. Across different categories, embeddings of the same word consistently form tight clusters, while different words remain well separated. Each word forms its own cluster (e.g., he, she, they in pronouns; knife, scissors, sharpener in tools; boat, canoe, surfing in water-related terms). Related variants such as sail-sailing show stability under inflection, while semantically similar pairs like scissors-shears also appear nearby despite being distinct words. These qualitative patterns match quantitative results: within-word similarity is high (∼0.87), between-word similarity is much lower (∼0.43), and silhouette scores (0.65-0.68) (Rousseeuw, 1987) confirm well-separated clusters.
Ablation on Patching Strategies.
Table 6 compares static and aligned patching. Aligned patching uses word boundaries from alignment, producing semantically coherent patches. We consider two variants: Align (sil sep.), keeping silence spans as separate patches, and Align (sil merged), merging them with adjacent words. Both outperform static patching at similar patch sizes-for instance, Align (sil sep.) reaches 60.3 on StoryCloze S→S vs. 58.7 for static size 6, and Align (sil merged) scores 38.5 on HellaSwag S→S vs. 37.2 for static size 9. Curriculum starts with Align (sil sep.) and gradually shifts to Static during training, retaining alignment benefits while matching the shorterpatch evaluation regime; it yields the strongest and most consistent results (e.g., 41.3 on HellaSwag S→S). Overall, aligned patching better preserves semantics than static, and curriculum combines alignment supervision with static-style evaluation for the best performance. For completeness, we also report BPE-aligned patching experiments in Appendix A.7.
this section cite: ['b33', 'b28']

Section: RELATED WORK
LLMs using speech tokens. Early neural audio generation methods included direct auto-regressive generation of the speech waveform (van den Oord et al., 2016), or using adversarial approaches (Kong et al., 2020). Following this, textless NLP work (Lakhotia et al., 2021) showed that by using discrete speech tokens obtained from self-supervised speech encoders (CPC, wav2vec 2.0, Hu-BERT) as targets for language modeling, can enable fully spoken LLMs. AudioLM (Borsos et al., 2023) further uses hierarchical generation, first predicts semantic tokens, and subsequent stages predict fine-grained acoustic tokens from SoundStream (Zeghidour et al., 2021), to achieve both high audio quality as well as long-term consistency. In addition to augmenting semantic speech tokens with pitch and style tokens to explicitly model expressivity, SpiritLM (Nguyen et al., 2025) also introduced interleaving speech modeling with text-tokens. More recently, Moshi (Défossez et al., 2024) propose a hierarchical inner monologue method, that jointly predicts time-aligned text and acoustic tokens (with distilled semantic information), together with modeling multiple-stream audio for handling full-duplex audio dialogues. Finally, similar to scaling laws for text LLMs (Hoffmann et al., 2022), Cuervo & Marxer (2024) fit scaling law curves to predict the performance of spoken LLMs, and find that they scale upto three order of magnitude more slowly than text LLMs.
Transferring textual knowledge into speech LMs. Slower scaling trends, together with a disproportionately lower amount of data, lead to a knowledge and reasoning gap between speech and text LLMs. To bridge this, AudioPaLM and TWIST (Rubenstein et al., 2023;Hassid et al., 2023) initialize a spoken LLM from a strong text model (PaLM-2, LLaMA), improving both speech understanding/generation and cross-lingual transfer. SpiritLM demonstrates that interleaved speech-text training significantly improves inter-modality knowledge transfer. Spectron (Nachmani et al.) uses a "Chain-of-Modality" pipeline to first produce text and then speech conditioned on the text, trading latency for stronger textual control, while Moshi (Défossez et al., 2024) uses a similar approach but generates interleaved text and speech as an inner monologue. To improve latency, LLaMA-Omni (Fang et al., 2024) style systems decode text and speech simultaneously, by upsampling textual LLM hidden states to decode speech units, before proceeding to decode the next text token.
this section cite: ['b32', 'b15', 'b17', 'b3', 'b37', 'b8', 'b12', 'b7', 'b29', 'b11', 'b8', 'b9']

Section: Speech model efficiency.
Compared to text, speech yields much longer token sequences, owing to higher frequency audio codecs, that consume many times additional compute to pre-train and generate. Efforts to mitigate this include methods to produce coarser speech units (Baade et al.;Tseng et al., 2025), hierarchical generation (Borsos et al., 2023), and producing residual tokens using parallel streams (Copet et al., 2023). Attempts at text-inspired approaches to compress token sequences such as BPE (Ren et al., 2022;Li et al., 2024) achieved limited success. In this paper, we take inspiration from recent dynamic patching approaches that have yielded improvements in other modalities such as text (Pagnoni et al., 2024;Yu et al., 2023;Videau et al., 2025) and vision (Pang et al., 2024;Beyer et al., 2023), and extend these methods to speech-text LLMs.
this section cite: ['b0', 'b31', 'b3', 'b6', 'b27', 'b18', 'b23', 'b36', 'b35', 'b24', 'b2']

Section: Speech Understanding Benchmarks.
Going beyond measuring only acoustic and phonetic capabilities of speech models using scores such as ABX (Kahn et al., 2020), Nguyen et al. (2020) established the Zero Resource Speech Benchmark 2021, comprising datasets/metrics to evaluate lexical (sWUGGY), syntactic (sBLIMP) and lexical-semantic (sSIMI) capabilities of spoken LLMs. Since these benchmarks contrast between very short speech segments, we found that dynamic compute approaches such as ours, do not yield significant improvements (see Appendix A.9). However, subsequently, Hassid et al. (2023) introduced the sStoryCloze and TopicStoryCloze datasets, which are story completion benchmarks in the speech modality measuring commonsense/understanding abilities of Spoken LLMs. We use these benchmarks in this paper, together with a speech version of the popular HellaSWAG textual benchmark, also measuring commonsense reasoning capabilities.
this section cite: ['b14']

Section: LIMITATIONS
Our study has several limitations. First, we focus on half-duplex speech-text modeling, where speech and text alternate in turns, and do not yet address full-duplex interaction required for realtime dialogue such as Moshi (Défossez et al., 2024). Second, our analysis is restricted to the pretraining stage, without exploring instruction fine-tuning or downstream adaptation, which we leave for future work. Third, some of our patching strategies, such as alignment and curriculum, rely on forced alignments during pre-training; although curriculum patching reduces this dependency at inference, fully alignment-free approaches remain an open challenge. Finally, our experiments are limited to the speech-text modality, and we have not yet extended LST to additional modalities such as image or video, which represent a promising next direction.
this section cite: ['b8']

Section: CONCLUSION
We presented the Latent Speech-Text Transformer (LST), a patch-based framework that aggregates speech tokens into latent units to improve computational efficiency and balance across modalities in multimodal language modeling. Across controlled-budget evaluations, scaling analyses, and downstream transfer experiments, LST consistently outperforms SpeechLLM baselines while reducing autoregressive sequence length and inference cost. Importantly, the gains arise from shortening the effective autoregressive sequence through latent speech patching, enabling more compute-efficient model scaling without sacrificing speech coverage or reconstruction quality. These findings highlight token-density imbalance as a key bottleneck in scaling spoken language models and suggest LST as a practical step toward compute-efficient unified speech-text foundation models.
this section cite: []

Section: References
Ref_id:b0 Title: Syllablelm: Learning coarse semantic units for speech language models Year: ()
Ref_id:b1 Title: wav2vec 2.0: A framework for self-supervised learning of speech representations Year: (2020)
Ref_id:b2 Title: Flexivit: One model for all patch sizes Year: (2023)
Ref_id:b3 Title: Audiolm: a language modeling approach to audio generation Year: (2023)
Ref_id:b4 Title: The spotify podcast dataset Year: (2020)
Ref_id:b5 Title: 100,000 podcasts: A spoken english document corpus Year: (2020)
Ref_id:b6 Title: Simple and controllable music generation Year: (2023)
Ref_id:b7 Title: Scaling properties of speech language models Year: (2024)
Ref_id:b8 Title: Moshi: a speech-text foundation model for real Year: (2024)
Ref_id:b9 Title: Llama-omni: Seamless speech interaction with large language models Year: (2024)
Ref_id:b10 Title: The people's speech: A large-scale diverse english speech recognition dataset for commercial usage Year: (2021)
Ref_id:b11 Title: Textually pretrained speech language models Year: (2023)
Ref_id:b12 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b13 Title: Self-supervised speech representation learning by masked prediction of hidden units Year: (2021)
Ref_id:b14 Title: Librilight: A benchmark for asr with limited or no supervision Year: (2020)
Ref_id:b15 Title: Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis Year: (2020)
Ref_id:b16 Title: Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing Year: (2018)
Ref_id:b17 Title: On generative spoken language modeling from raw audio Year: (2021)
Ref_id:b18 Title: On the effectiveness of acoustic bpe in decoder-only tts Year: (2024)
Ref_id:b19 Title: A corpus and evaluation framework for deeper understanding of commonsense stories Year: (2016)
Ref_id:b20 Title: Spoken question answering and speech continuation using spectrogram-powered llm Year: ()
Ref_id:b21 Title: The zero resource speech benchmark 2021: Metrics and baselines for unsupervised spoken language modeling Year: (2020)
Ref_id:b22 Title: Spiritlm: Interleaved spoken and written language model Year: (2025)
Ref_id:b23 Title: Byte latent transformer: Patches scale better than tokens Year: (2024)
Ref_id:b24 Title: Next patch prediction for autoregressive visual generation Year: (2024)
Ref_id:b25 Title: Mls: A large-scale multilingual dataset for speech research Year: (2020)
Ref_id:b26 Title: Robust speech recognition via large-scale weak supervision Year: (2023)
Ref_id:b27 Title: Speech pre-training with acoustic piece Year: (2022)
Ref_id:b28 Title: Silhouettes: a graphical aid to the interpretation and validation of cluster analysis Year: (1987)
Ref_id:b29 Title: A large language model that can speak and listen Year: (2023)
Ref_id:b30 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b31 Title: Taste: Textaligned speech tokenization and embedding for spoken language modeling Year: (2025)
Ref_id:b32 Title: Wavenet: A generative model for raw audio Year: (2016)
Ref_id:b33 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b34 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b35 Title: From bytes to ideas: Language modeling with autoregressive u-nets Year: (2025)
Ref_id:b36 Title: Megabyte: Predicting million-byte sequences with multiscale transformers Year: (2023)
Ref_id:b37 Title: Soundstream: An end-to-end neural audio codec Year: (2021)
Ref_id:b38 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
