Title: Word-Level Emotional Expression Control in Zero-Shot Text-to-Speech Synthesis
Abstract: While emotional text-to-speech (TTS) has made significant progress, most existing research remains limited to utterance-level emotional expression and fails to support word-level control. Achieving word-level expressive control poses fundamental challenges, primarily due to the complexity of modeling multi-emotion transitions and the scarcity of annotated datasets that capture intra-sentence emotional and prosodic variation. In this paper, we propose WeSCon, the first self-training framework that enables word-level control of both emotion and speaking rate in a pretrained zero-shot TTS model, without relying on datasets containing intra-sentence emotion or speed transitions. Our method introduces a transition-smoothing strategy and a dynamic speed control mechanism to guide the pretrained TTS model in performing word-level expressive synthesis through a multi-round inference process. To further simplify the inference, we incorporate a dynamic emotional attention bias mechanism and fine-tune the model via self-training, thereby activating its ability for word-level expressive control in an end-to-end manner. Experimental results show that WeSCon effectively overcomes data scarcity, achieving state-ofthe-art performance in word-level emotional expression control while preserving the strong zero-shot synthesis capabilities of the original TTS model.

Section: Introduction
Humans possess the ability to regulate emotional expression during speech flexibly [1]. To simulate this expressive capability, recent advances in text-to-speech synthesis (TTS) have increasingly focused on controllable generation of various aspects of speech, such as timbre, emotion, and speaking rate [2]. Such control is a key objective in the development of human-like and expressive TTS.
Most current TTS models exhibit zero-shot capabilities, enabling them to synthesize speech from text while cloning attributes such as timbre, emotion, and speaking rate from a reference speech sample [3,4,5]. Despite these advances, as shown in Figure 1, emotional and speaking rate control in current models is typically limited to the utterance level. This differs significantly from how humans naturally express emotion in speech. Unlike global speaker identity, emotional expression and speaking rate are dynamic and often vary within a single sentence [6,7]. Therefore, word-level control of these factors is essential for achieving more natural and expressive speech synthesis [8]. To address this limitation, some approaches have proposed phoneme-level emotion prediction from target 2 Ours 3 Target Text Emotion Prompt 4 4 2 3 ...... Speed Control or (1~5: slow ~ fast)
Figure 1: Word-level control of emotion and speaking rate aims to modulate both attributes within an utterance, guided by multiple emotional prompts and emotion-speed-tagged text. Our approach, WeSCon, achieves this using only a small-scale public dataset without emotion transitions. text to guide expressive synthesis [9,10,11]. While these methods show potential for word-level emotion control, relying solely on text makes it difficult to capture essential acoustic cues such as prosody and intensity, which are vital to emotional expression control [12,13,14]. To address this limitation, recent studies such as ELaTE [15] and EmoCtrl-TTS [16] have demonstrated that reference speech with emotional content can support intra-utterance control of time-varying expressive patterns, such as transitions from laughter to crying. These works reflect a growing interest in TTS with word-level control over both emotion and speaking rate, but they also underscore several fundamental challenges. First, word-level expression control requires multiple emotional speech prompts, which introduces the challenge of guiding the model to attend to the appropriate emotion at each word. In addition, current methods for fine-grained expression control rely on large-scale emotional speech datasets with time-aligned emotion transitions. However, such datasets are limited in both scale and accessibility [17], making fine-grained control even more difficult to realize in practice. These challenges lead us to ask: Is it possible to achieve effective word-level control of both emotion and speaking rate without relying on speech datasets containing emotion or speed transitions?
In this work, motivated by the zero-shot potential of pretrained TTS models, we propose WeSCon, a two-stage self-training framework that achieves Word-level Emotion and Speed Control for TTS using only a small amount of public speech data without emotion or speed transitions. In the first stage, we design a multi-round inference framework that incorporates a transition-smoothing module and a dynamic speed control mechanism. Without relying on any emotional training data, this approach enables a pretrained zero-shot TTS model to perform high-quality word-level emotional expression control in TTS. In the second stage, the original TTS model is repurposed as a student and trained under the supervision of the 1st-stage teacher. A dynamic emotional attention bias is introduced, enabling the student to acquire word-level control of emotion and speed through a simplified end-toend inference process, without the need for complex iterative generation or smoothing. Experimental results show that WeSCon achieves state-of-the-art performance on the task of word-level emotional expression control in TTS, while preserving the zero-shot generalization and generation capabilities of the pretrained TTS model. Our contributions are summarized as follows:
• We propose a multi-round inference mechanism equipped with transition smoothing and dynamic speaking rate control, which is the first to achieve word-level control of both emotion and speaking rate in TTS without relying on any emotional training data.
• We further introduce a novel self-training framework with a dynamic emotional attention bias mechanism that empowers a pretrained TTS model with end-to-end word-level emotion and speaking rate control, using limited data without intra-sentence emotion or speed transitions.
• We conduct comprehensive experiments to validate the effectiveness of our proposed framework. Results show that our method enables a pretrained zero-shot TTS model to achieve SOTA performance in word-level emotional expression control, while preserving its original zero-shot capabilities. Ablation studies further confirm the contribution of each key design component. Our samples are available at https://wangtianrui.github.io/wescon/.
2 Related Work
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16']

Section: Scarcity of Emotional Dataset
The development of controllable TTS, particularly for emotional expression control, depends heavily on high-quality emotional speech datasets [18,19,20]. While
You came late and I was surprised but it's okay.. but it's okay.. You came late and I was surprised 3 Zero-Shot TTS Transition Smooth DynSpeed Control Zero-Shot TTS Transition Smooth DynSpeed Control Zero-Shot TTS Transition Smooth DynSpeed Control 2 4 Zero-Shot TTS 3 2 4 2 4 3 + + + + DEAB 1st-Stage: Teacher Model 2nd-Stage: Self-Training 2 5 3 4 2 3 public corpora such as ESD [21], IEMOCAP [22], and CREMA-D [23] are available, they primarily provide utterance-level annotations and lack word-level or time-aligned emotional labels. These datasets are also limited in size and diversity, often consisting of scripted speech and covering a narrow range of emotions and speakers [24]. More importantly, emotional datasets with intrasentence variation, which are essential for learning word-level control, remain extremely scarce and are typically restricted to private use [15]. Creating such datasets is expensive, requiring detailed word-or frame-level annotation and subjective emotional labeling [25]. This lack of fine-grained emotional data poses a major challenge for training models capable of word-level expressive TTS.
this section cite: ['b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b14', 'b24']

Section: From Utterance-Level to Word-Level Controllability of Emotion and Speaking Rate
Most controllable TTS systems support only utterance-level control, where a single label or reference speech governs the entire sentence [26,27]. To achieve word-level control, some methods attempt to predict frame-or phoneme-level emotional indicators from text alone [28,29,30], but they often fail to capture expressive variability due to the lack of acoustic cues such as intensity and prosody [9,10,11].
Other approaches, such as ELaTE [15] and EmoCtrl-TTS [16], introduce emotional reference speech to enable intra-utterance control of specific expressive patterns like laughter or crying. While these represent progress, they are typically limited in expressiveness or rely on large-scale emotional datasets that are rarely publicly available. Consequently, achieving general and flexible word-level control over both emotion and speaking rate remains a major challenge.
Self-Training under Data Scarcity Self-training has become a promising approach for lowresource speech signal processing, enabling knowledge transfer without fine-grained datasets [31,32].
While it has been applied to tasks like speaker adaptation [33], paralinguistic modeling [34], and speech translation [35,36], its use for fine-grained emotional control in TTS remains unexplored, especially without detailed expressive labels. To address the scarcity of fine-grained datasets for word-level expressive control, we propose a self-training framework where a teacher model with multi-round inference, transition smoothing, and dynamic speed control generates expressive pseudolabels. A student model, sharing the teacher's backbone, is then fine-tuned under its supervision to perform word-level emotion and speaking rate control through a simplified end-to-end inference process, using only a small public dataset without intra-sentence emotion or speed transitions.
this section cite: ['b25', 'b26', 'b27', 'b28', 'b29', 'b8', 'b9', 'b10', 'b14', 'b15', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35']

Section: WeSCon

this section cite: []

Section: Overview
WeSCon is a two-stage self-training framework that enables word-level control of emotion and speaking rate in a pretrained zero-shot TTS model, using only a small amount of emotional speech data without intra-sentence emotion transitions as prompts. As shown in Figure 2, in the first stage, we introduce a multi-round inference process with transition smoothing and dynamic speaking rate control to generate speech with word-level expression variations. In the second stage, the 1st-stage model acts as a teacher to guide the original TTS model, equipped with a dynamic emotional attention bias (DEAB), toward word-level control through a simplified end-to-end inference. Sections 3.2 and 3.3 describe the two stages, and Section 3.4 provides the training details.
Text-Speech Language Model S 1 2 1 2 B •••••• Aligner 1 2 •••••• Interpolation (Speed Down) Downsampling (Speed Up) × × Text-Speech Language Model S 1 2 1 2 B •••••• Aligner 1 2 •••••• 2' Interpolation (Speed Down) Downsampling (Speed Up) × × •••••• •••••• ••••••
Flow Matching
Text token of prompt First Word Segment to Speech with Emotion Ⅰ Second Word Segment to Speech with Emotion Ⅱ S B •••••• Vocoder Token to Wavform Speech token of prompt Text token tail of last round Speech token tail of last round Speaker embedding Target text token Generated speech token Special token (BOS) Special token (SOS) Speech with different emotional expression (color) Generated tokens of multi-round inferences Speaker embedding Frozen pre-trained module Module need to be trained
this section cite: []

Section: Teacher Model

this section cite: []

Section: Word-Level Emotion Control
As discussed in Section 2, current TTS models can perform utterance-level emotion and speaker cloning. Building on this, we adopt the high-performance CosyVoice2 [37] as our backbone (details of the backbone architecture are provided in Appendix A) and propose a multi-round inference strategy, where the model synthesizes multiple segments using different emotional prompts to achieve word-level emotion control. While this approach enables flexible emotional modulation, it often causes unnatural acoustic discontinuities at segment boundaries. To address this, we introduce a transition-smoothing mechanism that improves coherence across inference rounds, as illustrated in Figure 3. Without modifying CosyVoice2, we append a lightweight content aligner, composed of non-causal Transformer [38] and convolutional layers. Trained on ASR data, this module predicts the corresponding text token for each speech token and requires no emotional supervision. During inference, the input text is segmented based on a user-defined emotion plan. At each inference round, the final text and speech tokens from the previous round are appended to the current prompt, forming an explicit tail-to-head linkage. This aligns naturally with CosyVoice2's continuation-style generation [39,40], enabling smooth and coherent emotional transitions.
this section cite: ['b36', 'b37', 'b38', 'b39']

Section: Word-Level Speaking Rate Control
In CosyVoice2, utterance-level temporal prosody, including speaking rate and duration, is entirely determined by the reference speech prompt. To support more flexible and word-level control of speaking rate within a single utterance, we introduce a dynamic speed control mechanism as part of our multi-round inference framework, as illustrated in Figure 3. The core idea is to adjust the prompt speech tokens using either nearest-neighbor interpolation or downsampling. Interpolation extends the prompt length, which slows down the generated speech, while downsampling shortens the prompt, resulting in a faster speaking rate. As demonstrated in Appendix B, this resampling method provides effective global prosody control. By integrating it into the multi-round inference process, the speaking rate can be dynamically controlled at the word level as needed.
this section cite: []

Section: Speaker Consistency
Although the speech tokens in CosyVoice2's language model (LM) are primarily designed to encode semantic information (as introduced in Appendix A), these speech tokens may still inadvertently leak a small amount of speaker-related information. In contrast, the flow matching serves as a voice conversion-based reconstructor that transforms the generated speech tokens into the voice of a specified target speaker. This design implies that as long as speaker inconsistency is avoided during the multi-round inference process in the LM part, the flow matching can effectively enforce speaker consistency in the final output. To ensure this consistency, we adopt a speaker-aware prompt selection strategy. Specifically, during multi-round inference, we prioritize selecting emotional prompts from different emotions of the same speaker. Then, a reference sample from the target speaker is randomly selected to provide the speaker identity to flow matching for generating the target speaker's speech.
this section cite: []

Section: Self-Training
In the previous section, we enabled word-level control of emotion and speaking rate by introducing a multi-round inference framework for CosyVoice2 [37]. However, components such as the noncausal content aligner, multi-round inference, and tail-to-head linkage introduce significant inference complexity. To reduce this overhead while preserving controllability, we adopt a self-training strategy. As shown in Figure 4, the enhanced first-stage model serves as a teacher to supervise the original TTS model. The student model, equipped with a dynamic emotional attention bias, learns to achieve word-level emotion and speaking rate control through a simplified end-to-end inference.
this section cite: ['b36']

Section: Self-Training with Teacher-Generated Emotion-Transition Speech
Our teacher model achieves word-level control of emotion and speaking rate without modifying the original TTS parameters, relying instead on a complex inference pipeline with dynamic speed control and multi-round generation. To transfer this fine-grained control ability to a simplified end-to-end model, we propose a self-training strategy. Specifically, the 1st-stage teacher model guides the student model to learn word-level controllability. We first use GPT-4o [41] to generate emotion-transition text sequences (details are shown in Appendix D), which are paired with public emotional speech samples (without emotion transitions) as prompts. The teacher then synthesizes speech with word-level variation in emotion and speaking rate. These outputs are filtered based on character accuracy and expressive similarity (details are introduced in Appendix E), and the student model is fine-tuned on the filtered supervisions with a small learning rate. This enables word-level emotional expression control during inference without requiring multi-round generation or dynamic concatenation.
this section cite: ['b40']

Section: Dynamic Emotional Attention Bias
We aim to preserve the strong zero-shot capability of the original TTS model while enabling wordlevel control of emotional expression under the self-training framework. To achieve this, we formulate the input structure as { S , C prompt I , C prompt II , . . . , C tgt , B , S prompt I , S prompt II , . . . , S tgt }, where C prompt i and S prompt i denote the text and speech tokens of the i-th emotional prompt, respectively. C tgt is the target text token sequence, and S tgt is the corresponding speech token sequence used as supervision. The symbols S and B indicate the beginning of text and speech. This design remains fully compatible with the original input format { S , C, B , S} of CosyVoice2, preserving the autoregressive pattern of the pretrained model. To further encode word-level emotional variation within this unified format, we extend the text-side input by inserting explicit emotion indicator tokens that mark the boundaries between emotional segments. As illustrated in Figure 4, the final input sequence preceding B becomes { S , E I , C prompt I , E II , C prompt II , . . .}, where each E i acts as a soft anchor guiding the model to modulate emotion transitions during generation.
While the above data formatting preserves CosyVoice2's generalization by avoiding interference with learned knowledge, it introduces a new challenge: during synthesis, the model may incorrectly attend to emotion-inconsistent prompts. For instance, when generating speech aligned with Emotion I, attention may drift toward prompts labeled with Emotion II, leading to emotional inconsistency and degraded synthesis quality. To address this, we propose a dynamic attention bias mechanism that constrains the model's focus to emotion-relevant prompt regions based on the predicted emotional trajectory. Concretely, we introduce a causal lightweight Transformer to predict token-level emotion labels E tgt t for each speech token S tgt t from historical context. Using the predicted emotion sequence, we introduce a dynamic attention bias mechanism at each Transformer layer. We first concatenate the current text-speech representation with the predicted emotion features and project it through a linear layer. The output is processed in two ways: one path adds a residual and feeds into the next layer, while the other is passed to an MLP [42] and softmax to produce a weight vector ω ∈ R 1×7 . The ω is then used to compute a dynamic attention bias by linearly combining seven predefined attention bias templates B temp ∈ R 7×T ×T (see Appendix F for details). The resulting bias is computed as:
B bias = 6 i=0 ω i • B temp i .
(
Then we multiply the bias with the softmax-normalized attention to selectively emphasize regions aligned with the current emotional context. The final self-attention output is computed as:
O =      Softmax QK ⊤ √ d ⊙ B bias T j=1 Softmax QK ⊤ √ d ⊙ B bias :,j      V ,(2)
where Q, K, V ∈ R H×T ×d denote the multi-head (H) query, key, and value, respectively, and d is the attention head dimension. The operator ⊙ denotes element-wise multiplication. This formulation enables the model to dynamically focus on emotionally relevant prompt segments at each generation step, thereby improving alignment between the generation and the intended emotional trajectory.
this section cite: ['b41']

Section: Detail Training Setup
WeSCon is trained in two stages. The first stage trains a content aligner to ensure smooth transitions during multi-round inference. In the second stage, a self-training strategy is adopted to transfer the teacher model's ability to control word-level emotional expression to the original TTS model.
this section cite: []

Section: The First Stage (Teacher Model)
We use forced alignment [43] to generate token-level alignments between transcripts and speech, which serve as supervision for the content aligner. The TTS model remains frozen throughout this stage. Training of the content aligner is conducted without multiround forwards. Let C and S denote the input text and speech token sequences, Y token ∈ N T denote the aligned target token sequence, where each label corresponds to one of V 1 token classes. Let Y bd ∈ R T ×1 be the binary label sequence for content boundary detection. The content aligner is jointly trained with a token-level content classification loss and a binary boundary detection loss:
L aligner = - T S -1 t=T C log p Y token t | S , C, B , S; θ tts , θ ca -log p Y bd t | S , C, B , S; θ tts , θ ca , (3
)
where T C and T S denote the last frame indices for text and speech, and T = T S -T C is the total number of speech tokens. The learnable parameters θ ca correspond to the content aligner, while θ tts is the frozen TTS model parameter used during forward propagation. We also apply class weighting during loss computation to reduce the impact of overrepresented silence tokens and address the imbalance in boundary label distribution [44].
this section cite: ['b42', 'b43']

Section: The Second Stage (Self-Training)
The teacher model generates supervision via multi-round inference using GPT-4o-generated texts with emotion labels. Token-level emotion labels are aligned based on emotion-text correspondence. The student model is optimized by two objectives. The first is a negative log-likelihood for speech token prediction:
L tts = - T tgt -1 t=T prompt log p S tgt t | S , C prompt , C tgt , E text , B , E speech <t , S prompt , S tgt <t ; θ tts , θ ea , (4
)
where C and S are text and speech tokens for prompt and target, E are text-level and token-level emotion labels, and trainable θ tts , θ ea denote TTS model and emotion aligner parameters. The second is a token-level cross-entropy loss for emotion prediction:
L e = - T tgt -1 t=T prompt log p E tgt t | S , C prompt , C tgt , E text , B , E speech <t , S prompt , S tgt <t ; θ tts , θ ea . (5
)
4 Experiments
this section cite: []

Section: Experimental Setup
Data and Model Configuration In the first stage, the content aligner is trained on 200 hours of nonemotional English-Chinese speech from LibriSpeech-100-Clean [45] and AISHELL-1 [46]. In the second stage, the teacher model uses non-transition emotional train-set from ESD [21] as prompts to synthesize training samples based on emotion-transition texts generated by GPT-4o (see Appendix D for generation details and examples). We adopt CosyVoice2 [37] as the backbone TTS model. The content aligner is composed of five non-causal Transformer layers and two 5×5 convolutional layers with stride 1 and batch normalization [47], following CosyVoice2's configuration for architectural consistency. In the second stage, the emotion aligner is a lightweight two-layer causal Transformer. The emotional attention bias module includes a linear layer with a hidden dimension of 14 and an MLP output dimension of 7.
this section cite: ['b44', 'b45', 'b20', 'b36', 'b46']

Section: Setup of Training and Inference
In the first stage, the content aligner is trained for 400k steps on 2 NVIDIA 3090 GPUs using Adam [48] with a learning rate linearly warmed up to 2.5e-4 over the first 10% of steps, then linearly decayed to 0. Each batch contains 90 seconds of speech. In the second stage, the student model is trained for 600k steps on 4 NVIDIA 3090 GPUs. The TTS model is frozen for the first 20k steps to focus on training the emotion aligner. Each batch contains 40 seconds of speech, and Adam is used with a fixed learning rate of 5e-7. Repetition-aware top-k sampling [49] is applied during inference, with k = 50 and temperature = 0.9.
Evaluation To evaluate word-level control over emotion and speaking rate, we construct test sets based on test set of ESD and use outstanding zero-shot TTS models [50,51,52,37] with multi-round concatenative inference as baselines (see Appendix G for details). We use objective and subjective metrics to assess system performance (see Appendix G.3 for details). For intelligibility, we report WER using Whisper-Large [53] for English and CER using Paraformer [54] for Chinese. Speaker similarity (S-SIM) is computed via cosine similarity of WavLM-Large embeddings [55]. To evaluate prosody alignment, we use AutoPCP [56]. Emotion similarity metrics (Emo2v. and Aro.) are computed using emotion2vec-Large [57] and a wav2vec-based model [58], respectively. We use the variance of DNSMOS-Pro [59] (DNSV) to assess the naturalness of emotion transition. Subjective evaluation includes four kinds of Mean Opinion Score (MOS): SMOS (speaker similarity), NMOS (naturalness of emotion transition), EMOS (emotion match), and SPMOS (speed match), each rated on a 5-point scale. Both the mean and 95% confidence intervals of MOS are reported.
this section cite: ['b47', 'b48', 'b49', 'b50', 'b51', 'b36', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58']

Section: Experimental Results

this section cite: []

Section: Comparison with Reference Models
Objective Evaluation We evaluate our method on word-level emotion and speaking rate control in both English and Chinese TTS. As shown in Table 1, WeSCon (1st-stage) and WeSCon (2nd-stage) consistently outperform baselines on expressive metrics. Notably, the 2nd-stage model achieves   Capability on Zero-shot TTS In addition to introducing word-level controllability, we evaluate the performance of our method on the standard zero-shot TTS task using the SEED test set (test-zh) [60]. As shown in Table 3, the WeSCon (1st) model yields results identical to CosyVoice2, as the backbone TTS is frozen during this stage. The 2nd-stage model also achieves comparable results. Together with the findings in Table 1, these results demonstrate that our method enables word-level emotion and speaking rate control without significantly degrading the original zero-shot TTS performance of the pretrained model.
this section cite: ['b59']

Section: Ablation Study
Transition-Smoothing Mechanism We evaluate the impact of the transition-smoothing mechanism by removing the tail-to-head alignment during multi-round inference in the 1st-stage model. As shown in Table 4, removing this mechanism ("w/o smoothing") leads to a substantial increase in DNSV (from 4.980 to 7.568), indicating degraded smoothness between expressive transitions. Additionally, speaker (S-SIM) and emotion similarity (Emo2V. and Aro.) drop notably, suggesting that the discontinuity negatively affects both emotional expression and speaker consistency. These results confirm that our smoothing strategy plays a crucial role in ensuring coherent segment transitions during generation. Speaking Rate Control To examine the effectiveness of our dynamic speaking rate control, we remove this component from the 1st-stage model ("w/o speed control"). As shown in Table 4, DNSV slightly increases from 4.980 to 5.067, and performance drops are observed across most expressive metrics, such as AutoPCP (2.650 to 2.499) and Emo2v. (0.866 to 0.844). This suggests that speaking rate variation provides important prosodic cues for emotional expression in TTS. In addition, we further investigate the interaction between speaking rate control and emotional expression in Appendix C.
this section cite: []

Section: Dynamic Emotional Attention Bias
In the 2nd-stage model, we evaluate the effect of removing the dynamic emotional attention bias ("w/o attention bias"). As shown in Table 4, this results in a clear performance drop across all metrics, especially emotion similarity. DNSV also increases, indicating reduced smoothness. The results confirm the importance of the attention bias module in enabling the 2nd-stage model to focus on the correct emotional prompt during inference.
Data format of Self-training We further investigate the importance of data formatting in selftraining. As shown in Table 4, removing the emotion flags ("w/o emotion flag") results in performance drops across all metrics, indicating that these flags play a crucial role in signaling the locations of emotional shifts to the model. Furthermore, replacing our input data format with a naive one that simply concatenates prompts and targets ("w/o data format"), as {C prompt I , B , S prompt II , . . . , C tgt , B , S tgt } leads to the most significant degradation in expressive metrics, including a sharp increase in CER from 2.166 to 4.141. These results suggest that aligning the data organization with the structure used during pretraining allows the model to better leverage its pre-trained knowledge.
this section cite: []

Section: Self-Training Data Size
We evaluate the impact of training data size in the self-training process by varying the amount of synthetic speech used to fine-tune the 2nd-stage model. Metrics are normalized between 0 and 1. As shown in Figure 5, performance improves with more data and peaks at 500 hours. Beyond this point, metrics begin to decline. This trend is attributed to the limited variety of emotional categories and speaker identities in the ESD, which restricts expressive diversity and leads to overfitting when the data scale becomes overly redundant.
this section cite: []

Section: Out-of-Domain Generalization and Alignment
To further assess the model's robustness, we evaluate its generalization ability on out-of-domain data (Appendix H). In addition, we report the alignment accuracy achieved in both training stages (Appendix I).
this section cite: []

Section: Conclusion, Limitations, and Broader Impact
Conclusion In this paper, we propose WeSCon, the first method to overcome expressive data scarcity and enable word-level emotional expression control through end-to-end inference, under a self-training framework with a dynamic emotional attention bias mechanism. Experimental results show that WeSCon achieves state-of-the-art performance using only limited data without emotion or speed transitions, while maintaining strong zero-shot TTS capabilities.
Limitations and Future Work 1) Gradual emotion transitions. While WeSCon achieves smooth signal-level transitions, it lacks semantic modeling of emotional evolution. In human speech, emotional changes often involve intermediate states. 2) Emotion diversity and composition. The model is limited to a fixed set of discrete emotions and does not support compositional or blended expressions, such as combining anger and sadness to convey despair. 3) Conditioned control. Emotional transitions are currently predefined by GPT-4o-based plans, which restricts flexibility. Future work will explore more dynamic, context-aware control strategies to enable natural, interactive emotional expression.
this section cite: []

Section: References
Ref_id:b0 Title: Expression of emotion in voice and music Year: (1995)
Ref_id:b1 Title: Towards controllable speech synthesis in the era of large language models: A survey Year: (2024)
Ref_id:b2 Title: Speak, read and prompt: High-fidelity text-to-speech with minimal supervision Year: (2023)
Ref_id:b3 Title: Neural codec language models are zero-shot text to speech synthesizers Year: (2023)
Ref_id:b4 Title: Naturalspeech 3: Zero-shot speech synthesis with factorized codec and diffusion models Year: (2024)
Ref_id:b5 Title: Prosody and emotions Year: (2002)
Ref_id:b6 Title: Speech emotion recognition considering local dynamic features Year: (2017-10-16)
Ref_id:b7 Title: Unsupervised word-level prosody tagging for controllable speech synthesis Year: (2022)
Ref_id:b8 Title: ED-TTS: Multi-scale emotion modeling using cross-domain emotion diarization for emotional speech synthesis Year: (2024)
Ref_id:b9 Title: Fine-grained emotion strength transfer, control and prediction for emotional speech synthesis Year: (2021)
Ref_id:b10 Title: Rich prosody diversity modelling with phone-level mixture density network Year: (2021)
Ref_id:b11 Title: Fine-grained style control in transformer-based text-tospeech synthesis Year: (2022)
Ref_id:b12 Title: Exploring speech style spaces with language models: Emotional tts without emotion labels Year: (2024)
Ref_id:b13 Title: An emotion speech synthesis method based on vits Year: (2023)
Ref_id:b14 Title: Making flow-matching-based zero-shot text-to-speech laugh as you like Year: (2024)
Ref_id:b15 Title: Laugh now cry later: Controlling time-varying emotional states of flow-matching-based zero-shot text-to-speech Year: (2024)
Ref_id:b16 Title: Analyzing the influence of different speech data corpora and speech features on speech emotion recognition: A review Year: (2024)
Ref_id:b17 Title: Metts: Multilingual emotional text-to-speech by cross-speaker and cross-lingual emotion transfer Year: (2024)
Ref_id:b18 Title: Emodiff: Intensity controllable emotional text-to-speech with soft-label guidance Year: (2023)
Ref_id:b19 Title: Emovoice: Llm-based emotional text-to-speech model with freestyle text prompting Year: (2025)
Ref_id:b20 Title: Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset Year: (2021)
Ref_id:b21 Title: Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation Year: (2008)
Ref_id:b22 Title: Crema-d: Crowd-sourced emotional multimodal actors dataset Year: (2014)
Ref_id:b23 Title: Emobox: Multilingual multi-corpus speech emotion recognition toolkit and benchmark Year: (2024)
Ref_id:b24 Title: Emphasis rendering for conversational text-to-speech with multi-modal multi-scale context modeling Year: (2024)
Ref_id:b25 Title: Seed-tts: A family of high-quality versatile speech generation models Year: (2024)
Ref_id:b26 Title: Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens Year: (2024)
Ref_id:b27 Title: Naturalspeech: End-to-end text-to-speech synthesis with human-level quality Year: (2024)
Ref_id:b28 Title: Emoq-tts: Emotion intensity quantization for fine-grained controllable emotional text-to-speech Year: (2022)
Ref_id:b29 Title: Emotion-controllable speech synthesis using emotion soft label, utterance-level prosodic factors, and word-level prominence Year: ()
Ref_id:b30 Title: Ekin Dogus Cubuk, and Quoc Le. Rethinking pre-training and self-training Year: (2020)
Ref_id:b31 Title: Self-training: A survey Year: (2025)
Ref_id:b32 Title: Unsupervised domain adaptation for speech recognition via uncertainty driven self-training Year: (2021)
Ref_id:b33 Title: Frame-wise breath detection with selftraining: An exploration of enhancing breath naturalness in text-to-speech Year: (2024)
Ref_id:b34 Title: Self-training for end-to-end speech translation Year: (2020)
Ref_id:b35 Title: STEMM: Self-learning with speech-text manifold mixup for speech translation Year: (2022-05)
Ref_id:b36 Title: Cosyvoice 2: Scalable streaming speech synthesis with large language models Year: (2024)
Ref_id:b37 Title: Attention is all you need Year: (2017)
Ref_id:b38 Title: Audiolm: a language modeling approach to audio generation Year: (2023)
Ref_id:b39 Title: Neural codec language models are zero-shot text to speech synthesizers Year: (2025)
Ref_id:b40 Title: Gpt-4o system card Year: (2024)
Ref_id:b41 Title: Multilayer perceptron and neural networks Year: (2009)
Ref_id:b42 Title: Montreal forced aligner: Trainable text-speech alignment using kaldi Year: (2017)
Ref_id:b43 Title: Learning from imbalanced data Year: (2009)
Ref_id:b44 Title: Librispeech: an asr corpus based on public domain audio books Year: (2015)
Ref_id:b45 Title: Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline Year: (2017)
Ref_id:b46 Title: Batch normalization: Accelerating deep network training by reducing internal covariate shift Year: (2015)
Ref_id:b47 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b48 Title: Vall-e 2: Neural codec language models are human parity zero-shot text to speech synthesizers Year: (2024)
Ref_id:b49 Title: Indextts: An industrial-level controllable and efficient zero-shot text-to-speech system Year: (2025)
Ref_id:b50 Title: F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching Year: (2024)
Ref_id:b51 Title: Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled speech tokens Year: (2025)
Ref_id:b52 Title: Robust speech recognition via large-scale weak supervision Year: (2023)
Ref_id:b53 Title: Paraformer: Fast and accurate parallel transformer for non-autoregressive end-to-end speech recognition Year: (2022)
Ref_id:b54 Title: Wavlm: Large-scale self-supervised pretraining for full stack speech processing Year: (2022)
Ref_id:b55 Title: Seamless: Multilingual expressive and streaming speech translation Year: (2023)
Ref_id:b56 Title: emotion2vec: Self-supervised pre-training for speech emotion representation Year: (2023)
Ref_id:b57 Title: wav2vec 2.0: A framework for self-supervised learning of speech representations Year: (2020)
Ref_id:b58 Title: Dnsmos pro: A reduced-size dnn for probabilistic mos of speech Year: (2024-09-01)
Ref_id:b59 Title:  Year: (2024)
Ref_id:b60 Title: Funaudiollm: Voice understanding and generation foundation models for natural interaction between humans and llms Year: (2024)
Ref_id:b61 Title: A universal neural vocoder with large-scale training Year: (2022)
Ref_id:b62 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b63 Title: Scalable diffusion models with transformers Year: (2023-10)
Ref_id:b64 Title: Maximilian Nickel, and Matt Le. Flow matching for generative modeling Year: (2022)
Ref_id:b65 Title: Convnext v2: Co-designing and scaling convnets with masked autoencoders Year: (2023)
Ref_id:b66 Title: Eric Battenberg, and Oriol Nieto. librosa: Audio and music signal analysis in python Year: (2015)
Ref_id:b67 Title: Cheavd: a chinese natural emotional audio-visual database Year: (2017)
