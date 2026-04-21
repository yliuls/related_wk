Title: Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
Abstract: We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multiaudio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closedsource models trained on much larger datasets.

Section: 
Audio-including speech, sounds, and music-is central to human perception and interaction. It enables us to understand our surroundings, engage in conversations, express emotions, interpret videos, and enjoy music. For AI systems to approach artificial general intelligence (AGI) [88], they must similarly develop the ability to comprehend and reason over diverse audio signals. While Large Language Models (LLMs) excel at language-based reasoning, their audio comprehension remains limited -both in accessibility and capability [54,106]. Extending LLMs to process and reason over audio is essential for building truly context-aware, intelligent agents.
Audio-Language Models (ALMs) extend the capabilities of LMs to the auditory domain. Early works such as CLAP [33] align audio and text in a shared embedding space, enabling them with tasks like retrieval [89].
39th Conference on Neural Information Processing Systems (NeurIPS 2025).
this section cite: ['b88', 'b53', 'b106', 'b32', 'b89']

Section: Models
Audio Understanding Voice Multi-turn Chat Long Audio (>30 secs) Open-Source
Sound Music Speech In Out* Single A Multiple A Speech Sound Music Model Data Code LTU ✓ ✓ × × × × × × × × ✓ ✓ ✓ LTU-AS ✓ ✓ ✓ × × × × × × × ✓ ✓ ✓ GAMA ✓ ✓ × × × × × × × × ✓ ✓ ✓ SALMONN ✓ ✓ ✓ × × × × × × × ✓ ✓ ✓ MuLLaMa × ✓ × × × × × × × × ✓ ✓ ✓ Phi-4-mm ✓ ✓ ✓ × × × × ✓ ✓ ✓ ✓ × × Qwen-Audio ✓ ✓ ✓ ✓ ✓ ✓ × × × × ✓ × × Qwen2-Audio ✓ ✓ ✓ ✓ ✓ ✓ × × × × ✓ × × Qwen2.5-Omni ✓ ✓ ✓ ✓ ✓ ✓ × ✓ ✓ ✓ ✓ × × GPT-4o Audio ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ × × × Gemini 2.0 / 2.5 ✓ ✓ ✓ ✓ ✓ ✓ × ✓ ✓ ✓ × × × Audio Flamingo ✓ ✓ × × × ✓ × × × × ✓ ✓ ✓ Audio Flamingo 2 ✓ ✓ × × × × × × ✓ ✓ ✓ ✓ ✓ Audio Flamingo 3 ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Table 1: Comparison of various LALMs in terms of capabilities and openness. AF3 stands out as the most capable and open model to date, achieving SOTA results across benchmarks (A in Chat stands for Audio). *Voice-out is powered by our novel streaming TTS implementation, which is also applicable to other LALMs.
More recently, the emergence of Large ALMs (LALMs)-decoder-only language models augmented with audio understanding [20,19,105]-has unlocked powerful capabilities, including open-ended audio question-answering (AQA) that demands both reasoning and world knowledge [101]. These capabilities have further enabled tasks like audio analysis [32,60], conversational assistants [24], etc.
However, existing models still fall short in key areas critical to AGI, such as expert-level reasoning [88,101], multi-turn and multi-audio dialogue [44], and long audio understanding [40]. We identify two core limitations: (i) most LALMs are trained primarily on short audio for recognition tasks rather than ones that require deliberate reasoning; and (ii) in turn, they lack exposure to the skill sets required for complex tasks. Additionally, most LALMs that support all three modalities of speech, sound, and music are closed-source: while some have publicly released models weights [20,19,1], they offer limited to no information about their data, code, or recipes (more details in Table 1).
this section cite: ['b19', 'b18', 'b105', 'b101', 'b31', 'b60', 'b23', 'b88', 'b101', 'b43', 'b39', 'b19', 'b18', 'b0']

Section: Main Contributions.
To address these issues, we introduce Audio Flamingo 3 (AF3), a fully open-sourcefoot_0 LALM with state-of-the-art performance in audio understanding and reasoning across 20+ benchmarks. In addition, AF3 brings several novel capabilities, including multi-turn, multi-audio chat, on-demand thinking, voice-to-voice interaction, and long-context audio reasoning (up to 10 minutes). We propose three core innovations to enable these capabilities: (i) Data: We focus on curating high-quality data at scale and propose (a) AudioSkills-XL: a large-scale dataset of 8M diverse AQA pairs, (b) LongAudio-XL: large-scale dataset of 1.25M diverse audio QA pairs for long audio reasoning; (c) AF-Chat: a multi-turn multi-audio chat dataset curated using a novel algorithm with 75k instances and (d) AF-Think: a dataset with 250k+ AQA pairs with short length prefixes to encourage CoT-type reasoning before arriving at the answer (ii) AF-Whisper: We train AF-Whisper, a unified audio encoder pretrained using a novel strategy on large-scale audio-caption pairs, capable of learning general-purpose representations across speech, sounds, and music; and (iii) Learning Curriculum: We train AF3 with a five-stage curriculum-based training strategy that progressively increases context length and task complexity. In summary, our main contributions are:
• We introduce Audio Flamingo 3 (AF3), the most open and capable foundational LALM to date. AF3 introduces key capabilities including: (i) long-context audio QA (extending beyond sounds as in [40] and including speech), and (ii) flexible, on-demand thinking, enabling the model to generate concise, CoT-style reasoning steps when prompted. AF3 achieves state-of-the-art performance on 20+ audio understanding and reasoning benchmarks.
• We also present AF3-Chat, a fine-tuned variant of AF3 designed for multi-turn, multi-audio chat and voice-to-voice interaction.
• We propose novelties in data curation, audio encoder representation learning, and training strategies. Being fully open, we release our code, training recipes, and 4 new datasets to promote research in this space.
this section cite: ['b39']

Section: Related Work
Audio Language Models. The rapid progress of LLMs has catalyzed the development of multimodal LLMs (MLLMs) capable of understanding and reasoning across diverse data modalities, including audio. Within this space, ALMs specifically target reasoning over auditory inputs such as speech, sounds, and music. ALMs typically follow two main architectural paradigms: (i) Encoder-only ALMs, which learn a joint embedding space for audio and text, enabling tasks like cross-modal retrieval. Representative models include CLAP [33], Wav2CLIP [112], and AudioCLIP [48]. (ii) Encoderdecoder ALMs, also referred to as LALMs, which use decoder-only LLMs augmented with an audio encoder. Notable examples include LTU [46], LTU-AS [45], SALMONN [104], Pengi [27], Audio Flamingo [65], Audio Flamingo 2 [40], AudioGPT [53], GAMA [41], Qwen-Audio [20], and Qwen2-Audio [19]. These LALMs have significantly improved performance on core audio understanding tasks such as automatic speech recognition (ASR) [96], audio captioning [60], and acoustic scene classification [16]. More importantly, they have enabled new capabilities such as open-ended audio question answering, which requires complex reasoning and external world knowledge.
Despite these advancements, current LALMs fall short in supporting various capabilities, including multi-turn, multi-audio chat, long-context audio comprehension, etc. Moreover, most LALMs are limited to specific audio types, lacking the ability to unify understanding across speech, sounds, and music. Finally, the most advanced LALMs remain only partially open, releasing model checkpoints without accompanying training code or data. This lack of transparency limits reproducibility and impedes scientific progress by obscuring the development process.
Reasoning and Long-Context Understanding. Recent progress in LLMs has increasingly emphasized long-context understanding. In the vision-language space, substantial strides have been made in modeling long videos [17]. In the audio domain, AF2 marked the first step toward long-context audio comprehension, though it is limited to sounds and music.
Parallel efforts have aimed to enhance reasoning in LLMs and MLLMs through improved reasoning datasets [101,110], advancements in multimodal perception [116,105], and emerging paradigms like chain-of-thought (CoT) prompting [80], which encourages models to "think before answering."
In developing AF3, we combine these advances-integrating controlled reasoning supervision, longcontext training, and modality diversity-to equip the model with strong reasoning capabilities and long-context comprehension, including speech.
this section cite: ['b32', 'b112', 'b47', 'b45', 'b44', 'b104', 'b26', 'b65', 'b39', 'b52', 'b40', 'b19', 'b18', 'b96', 'b60', 'b15', 'b16', 'b101', 'b110', 'b116', 'b105', 'b80']

Section: Methodology

this section cite: []

Section: Audio Flamingo 3 Architecture
In this section, we discuss our proposed architecture for Audio Flamingo 3 as shown in Figure 2. AF3 consists of i) AF-Whisper: an audio encoder with sliding window feature extraction, ii) audio projector, iii) an LLM, and iv) a streaming TTS. We provide details of each component below.
this section cite: []

Section: AF-Whisper Audio Encoder.
Prior work in audio representation learning typically treats speech, sounds, and music as separate modalities, and LALMs often rely on distinct encoders for each [104,41]. Using separate encoders for LALMs increases model complexity, introduces framerate mismatches, and can lead to training instability. To address this, we propose AF-Whisper, a unified audio encoder trained with a simple yet effective representation learning strategy to model all three audio types.
As illustrated in Figure 2, we start with the pre-trained Whisper large-v3 encoder [96], attach it to a standard Transformer decoder, and train using the audio captioning task with the next-token-prediction objective. To achieve this, we generate a natural language caption for each audio, describing its speech, sound, and music content. First, we pool several datasets and then prompt GPT-4.1 to generate the audio caption. For prompting, we use available metadata for each sample, which includes transcripts, ambient sound descriptions, and music attributes. For samples lacking any of the 3 metadata, we synthesize it using AF2 [40] or Whisper-Large-v3 ASR [96]. All datasets used for training are detailed in Section A.2. We choose Whisper as the backbone due to its existing speech understanding capabilities and its dense, high-resolution audio features, which are more informative than those from models like CLAP [33]. We connect it with a Transformer decoder using cross-attention (similar to RECAP [77] and AF2 [40]) with 24 layers, 8 attention heads, and 1024 hidden size. This was the India's top wheelchair …
The speech is in English by a male ....
A man gives a speech on a stage, as the crowd cheers and claps alon.
this section cite: ['b104', 'b40', 'b96', 'b39', 'b96', 'b32', 'b77', 'b39']

Section: No audible music.
A man, addresses the audience on stage amidst cheering and clapping, and says "This was the India's top wheelchair …. Feature Extraction. Given an audio input A, we first resample it to 16kHz mono. The raw waveform is then transformed into a 128-channel mel-spectrogram using a window size of 25ms and a hop size of 10ms. This mel-spectrogram is processed by AF-Whisper, producing hidden representations, denoted as h a = f a (A), where h a ∈ R N ×d . As shown in Figure 2, each audio is processed in 30-second chunks of non-overlapping sliding windows, and N or the temporal resolution depends on the length of the audio and the maximum number of sliding windows (which varies according to the stage of training). AF-Whisper produces audio features at a frame rate of 50Hz, and we further apply a pooling layer with a stride of two similar to [19]. d denotes the hidden dimension, which is 1280.
this section cite: ['b18']

Section: Audio Adaptor.
To align the audio modality with the text embedding space of the LLM, we introduce audio adaptor layers, denoted by A(.). Specifically, the encoded hidden representations h a from AF-Whisper are passed through these adaptor layers (2-layer MLP with a GeLU layer) to produce embeddings: a = A(h a ). These resulting embeddings serve as prompts to the LLM, alongside the textual instruction.
Large Language Model (LLM). We employ Qwen-2.5-7B [118] as our backbone, a decoder-only causal LLM with 7B parameters, 36 hidden layers, and 16 attention heads.
this section cite: ['b118']

Section: Streaming TTS.
To enable voice-to-voice interaction, we employ a TTS module for streaming speech generation, supporting streaming inputs and outputs. Our TTS module employs a decoderonly transformer architecture: it predicts the subsequent audio token conditioned on incoming subword text tokens from the LLM and the history of previously generated audio tokens. Similar streaming TTS techniques have been explored with LLMs [115] (for voice-out on LLM outputs), but not in the context of LALMs (which we define as models designed to perceive and reason over diverse audio inputs). Since not a core novelty of our work, we provide more details, including training and architecture, in Appendix I.
this section cite: ['b115']

Section: Audio Flamingo 3 Training Data
We present detailed statistics for all datasets used to train AF3 in Table 11. AF3 has a total of 5 stages of training, where each stage employs a unique combination of datasets with unique weights (number of passes over that dataset for that particular stage). For Stages 1 and 2, we use open-source, recognition-focused foundational datasets converted to QA format. In the following sub-sections, we introduce our four novel skill-focused and unique datasets, each accompanied by custom data curation strategies, used in Stages 3, 3.5, and 4, which form a core contribution of this work.
this section cite: []

Section: AudioSkills-XL: Expanding AudioSkills with Reasoning-Focused QAs
Audio QA pairs derived from foundational benchmarks focused on recognition tasks (e.g., ASR, acoustic event classification) are insufficient for training models in expert-level reasoning [101]. Therefore, in Stage 3 fine-tuning, we prioritize the development of reasoning and problem-solving abilities by curating large-scale, high-quality Audio QA data. Inspired by AF2, we limit this stage to short audio clips (≤30s) and defer long audio reasoning to later stages. We expand the AudioSkills dataset [40] by 4.5M new Audio QA pairs (majorly multiple-choice questions (MCQ)-based) to create AudioSkills-XL, a high-quality corpus containing 8M Audio QA pairs, using two strategies:
(1) We expand coverage of existing reasoning skills and introduce new ones using additional audio sources, increasing the dataset by 3.5M QA pairs: (a) For sounds, we incorporate data from YouTube8M and synthetic sources. (b) For music, we include Music4All [102] and the Million Song Dataset [8]. For YouTube8M, we adapt captions from AudioSetCaps [6] and generate QA using GPT-4.1 with general reasoning prompts from AF2. Additionally, we introduce new reasoning skills and design corresponding prompts to support them. For music, we generate data for novel skills (as AudioSkills was focused more on sounds; details in Table 6) and go beyond captionswe leverage metadata such as song titles, artist names, album names, etc (see Fig. 4 for full list) to generate more complex, reasoning-focused QAs. We also use this metadata to generate rich music captions for Stage 1 and 2 pre-training (see Fig. 4), demonstrating how text-based knowledge can enhance audio understanding, particularly in knowledge-driven domains like music. This method can be seen as synthetic knowledge generation, where we leverage text-based knowledge to enrich audio understanding and enable models to acquire domain-specific knowledge from unlabeled audios in the wild. Our analysis shows that LLMs like GPT-4.1 hold substantial world knowledge about music, and that metadata improves QA quality significantly.
(2) We augment AudioSkills with 1M speech QA samples using YouTube8M [2], LibriSpeech [92] (read speech), GigaSpeech [14] (conversational), and VoxCeleb2 [21] (interviews). From YouTube8M, we introduce a new task: Speech-in-Sound QA, where the model must reason over both speech content and ambient sounds to understand complex auditory scenes. To create these QAs, we create Speech-in-Sound-Caps, a new dataset with ≈2M speech-aware auditory scene captions from YouTube8M. To curate this, we first filter the dataset for English speech (using AF2) and transcribe the spoken content with Whisper-Large-v3. We then generate two types of descriptions: one capturing sound events and another summarizing speech characteristics such as tone, emotion, and pitch (both using AF2 and custom prompts; see Appendix 26). Finally, we prompt GPT-4.1 to synthesize a speech-aware scene caption. These captions significantly improve the quality of final audio captions (compared to only using sound information) by providing a more holistic representation of the audio. For LibriSpeech and GigaSpeech, we concatenate shorter segments into clips of 15-30 seconds, selecting information-dense segments filtered by prompting an LLM. To move beyond basic spoken content understanding common in most current datasets [121], we design five distinct types of speech QA that require diverse reasoning skills (explained in the next subsection).
this section cite: ['b101', 'b39', 'b102', 'b7', 'b5', 'b1', 'b92', 'b13', 'b20', 'b121']

Section: LongAudio-XL: Expanding LongAudio with Long Speech QA
To our knowledge, Long Speech QA (i.e., audio ≥ 30 seconds) has not been explored in prior work, despite its relevance to real-world applications such as long-form conversation understanding, meeting summarization, and narrative comprehension. To bridge this gap, we extend the existing LongAudio dataset [40] (focused on sounds and music) by incorporating over 1M reasoning-focused QA examples from long-form speech (30s-10min). We curate audios from diverse sources including: Single-speaker speech: LibriSpeech (audiobooks) [92], EuroParl [62], VoxPopuli (parliamentary debates) [107] and Multi-speaker conversations: Spotify Podcasts [23], Switchboard [43], Fisher (dyadic calls) [22], MELD [94], DailyTalk [71], MMDialog (natural dialogues) [35]. We merge consecutive short segments in chronological order to construct longer, coherent audios. We construct QAs across a wide range of skills, as illustrated in Figure 3: 1. Sarcasm Identification: Inferring sarcasm by analyzing content, tone, and emotional cues.
this section cite: ['b39', 'b92', 'b62', 'b107', 'b22', 'b42', 'b21', 'b94', 'b71', 'b34']

Section: Emotional State Reasoning: i) Identification:
Determine the speaker's emotion at a specific utterance. ii) Causal Reasoning: Identify the reason behind a speaker's emotional state using conversational context. iii) Emotion Flip: Explain shifts in a speaker's emotional state during the conversation. 3. Topic Relationship Reasoning: Understand how two ideas or topics are related within the overall discourse. 4. Information Extraction (IE): i) Needle QA: Targeted QA on specific utterances or parts of the speech (e.g., entity or fact extraction, general knowledge linkage). ii) Causal QA: Identify causes for a particular utterance in context. iii) Response QA: Extract how one speaker responds to another's statement. iv) Topic QA: Identify the main topic of the speech or conversation. 5. Summarization: Generate a concise summary of the speech content.
this section cite: []

Section: AF-Think
Question: Which genre best describes the song? Choose one among the following options: (A) Electronic (B) Jazz (C) Pop rock (D) Country. Please think and reason about the input music before you respond. Answer: Based on the presence of electric guitars, drums with a driving beat, and a vocal style that is melodic and accessible, blending elements of both popular music and rock instrumentation, the most fitting genre among the options is: *(C) Pop rock* Caption: A male voice, sounding disgusted, slowly says 'Thank you' amid the rustling of book pages and occasional laughter.
Question: What can be inferred about the speaker's cultural context from his statement and the background audio?
this section cite: []

Section: Emotion State
Identification: In the input conversation, when discussing a forgotten item, what was the speaker's emotional state while acknowledging their oversight? Causal Reasoning: In the input conversation, why does the speaker feel excited while describing the fish phenomenon? Emotion Flip: In the conversation, how does the speaker's emotional state shift from anger to frustration when discussing the actions of another person related to a past relationship?
this section cite: []

Section: Sarcasm Identification
Question: In the conversation, why might the suggestion 'You're smart. You could go back to school--finish your Master's that you were started never finished' be considered sarcastic?
this section cite: []

Section: Topic Relationship Reasoning
Question: How does the speaker's personal motivation for visiting relate to their professional engagement with the person they are seeing?
this section cite: []

Section: Information Extraction
Needle QA: What was the specific example given to highlight the difference between investing and speculating with cryptocurrencies? Causal QA: What caused one speaker to become frustrated about the bag situation? Response QA: How does the speaker respond when asked about their study program?
this section cite: []

Section: Order
Temporal Order: What is the order in which the speaker discusses the topics in the speech? (A) The speaker introduces the last crime and its seasonal timing. .... Temporal Attribute: How does the focus of the speech evolve over time? (A) It moves from general grievances to more specific legislative conflicts., (B) It starts optimistic and becomes critical. ... Temporal Referring: When does the speaker discuss the revival of mills and factories in relation to other topics? (A) At the very beginning, (B) After describing the general social mood, ...
this section cite: []

Section: Temporal Grounding:
At what point in the speech does the speaker describe the specific setting of the last crime?\nChoose the correct option from the following options: (A) At the beginning, (B) In the middle, (C) At the end User (turn 1): <sound-0>\This upbeat, energetic piece feels like it could power a retro game level. If I wanted to make a chill, lo-fi remix of it, what elements would you suggest I change? AI: To turn this high-energy chiptune into a chill, lo-fi track, start by slowing down the tempo and simplifying the rhythms ...... recontextualized in a mellow vibe. User (turn 2): <sound-1>\nSwitching to this French ballad, could I blend melodic ideas from the retro one into this kind of easy listening style? AI: Absolutely. You could reinterpret the melodic hooks ... User (turn 3): Would the harmonic structure of the retro track need to change much to fit this? Or could I keep most of it as is? AI: You could keep the core melodic structure, but adapting the harmony to richer, more complex chords would help it feel at home in an ... Resolve references to specific time points (e.g., "at the end") iv) Temporal Grounding: Identify when in the audio a specific topic was discussed.
this section cite: []

Section: AF-Chat Speech-in-Sound

this section cite: []

Section: AF-Think: Towards flexible, on-demand reasoning
Recent studies show that making an LLM "think", similar to chain-of-thought (CoT) prompting [111], can improve reasoning performance in LLMs [47], especially for complex tasks like coding and math (e.g., DeepSeek-R1, OpenAI-o1). Visual MLLMs have also benefited from this paradigm [116,109].
In the audio domain, early attempts such as Audio-CoT [80], Audio-Reasoner [114], and R1-AQA [73] have explored CoT-style reasoning, but often yield limited gains and involve complex or inefficient training procedures. Moreover, consistent with findings in [73], we observe that deep, explicit thinking does not always improve performance in audio understanding tasks.
In AF3, we adopt a lightweight thinking mechanism with two key modifications: (i) We create AF-Think, a dataset of 250k MCQ-based QAs with short, controlled thought preceding the answer. This additional thinking serve as a prefix to the answer and are limited to an average of approximately 40 words, providing concise yet effective context for audio QA (example in Figure 3). (ii) Instead of explicitly post-training for CoT, we add a special suffix to QA prompts (highlighted in Figure 3). We include AF-Think in the Stage 3.5 training mixture, upweighted relative to standard QA data. This allows AF3 to think only when prompted, offering flexible, on-demand additional reasoning.
To generate AF-Think, we first sample a subset of multiple-choice reasoning QAs from AudioSkills-XL and LongAudio-XL (originally with just the correct option as the answer). Next, we prompt Gemini 2.0 Flash with the input audio, the question, and the answer to generate short thinking prefixes. We found Gemini to hallucinate less and generate more accurate reasoning when guided by the ground-truth answer, rather than producing CoT from scratch. We restrict this process to only high-quality datasets and filter out noisy instances.
this section cite: ['b111', 'b46', 'b116', 'b109', 'b80', 'b114', 'b73', 'b73']

Section: AF-Chat: Multi-turn Multi-audio Chat Data
While single-turn single-audio QA training equips LALMs to reason over individual audio inputs, enabling free-form, multi-turn, multi-audio conversations requires a dedicated chat alignment tuning stage, akin to the instruction-tuning phases used for LLMs [122]. Chat becomes significantly more complex when multiple audio inputs must be integrated across turns, requiring the model to track context, reason over relationships between past and current inputs, and generate coherent follow-ups. Despite its importance and chat being the most used application of LLMs, this capability remains underexplored in LALMs primarily due to the absence of open, high-quality training data.
To address this gap, we introduce AF-Chat, a high-quality fine-tuning dataset consisting of 75k multi-turn, multi-audio chat instances. On average, each dialogue includes 4.6 audio clips and 6.2 dialogue turns, with a range of 2-8 audio clips and 2-10 turns. To construct this dataset, we draw from Speech-in-Sound Caps (for speech and sounds), and Music4All and MSD (for music). We follow a two-step curation process: First, for each seed audio, we identify its top 8 most semantically similar and dissimilar clips using a combination of captions, NV-Embed-v2 [68] embeddings, and FAISS-based clustering [31] (details in Appendix E.2). For every dialogue, we restrict the audios to this pool. This targeted clustering yields significantly higher-quality dialogues than random audio selection by ensuring each instance is grounded in a diverse yet semantically coherent audio pool.
Next, we prompt GPT-4.1 using carefully designed expert exemplars (Fig. 36 and 35) to generate natural, multi-turn chat sessions under the following constraints: (i) the model may choose any subset of the similar/dissimilar audios (up to 10 turns), prioritizing conversation quality; (ii) not all turns require a new audio-follow-up and clarification questions are encouraged; and (iii) later turns may refer back to earlier audios or responses to simulate real conversational grounding. The design of AF-Chat is informed by extensive internal human studies to reflect how users naturally interact with audio-language models. As a result, it provides rich, diverse supervision for aligning LALMs to handle complex, contextual, and naturalistic audio conversations. Finally, we select 200 high-quality samples for the test set, known as AF-Chat-test, and ensure that the audios in these instances have audio clips that were not seen during training.
this section cite: ['b122', 'b68', 'b30']

Section: Audio Flamingo 3 Training Strategy
AF3 is trained using a five-stage strategy designed to progressively enhance its capabilities by increasing audio context length, improving data quality, and diversifying tasks. A full list of datasets used at each stage is provided in Appendix 11.
this section cite: []

Section: Stage 1: Alignment pre-training.
For this stage, we train only the audio adaptor layers while keeping the audio encoder and LLM frozen. This step aligns encoder representations with the language model. Stage 2: Encoder Tuning. The main purpose of this stage is to adapt AF-Whisper to diverse datasets and broaden and improve its audio understanding capabilities. We fine-tune both the audio encoder and adaptor while keeping the LLM frozen. In both Stages 1 and 2, the audio context length is limited to 30 seconds, and training uses recognition-focused datasets (e.g., classification, captioning, and ASR). Stage 3: Full Fine-Tuning. The primary purpose of this stage is to emphasize reasoning and skill acquisition by the LALM. As mentioned earlier, since skill-specific data is easy to scale on short audios, we still stick to short audios in this stage and use high-quality foundational and QA datasets and our proposed AudioSkills-XL. However, we increase the audio context length up to 2.5 minutes now to accommodate the moderately long audios in AudioSkills. The resulting model at the end of Stage 3.5 is referred to as AF3. Stage 3.5: Context Extension and Thinking. This stage focuses on extending context length and encouraging CoT-style reasoning. In addition to the Stage 3 data mixture, we incorporate LongAudio-XL and AF-Think. We adopt LoRA-based training [51]-similar to LTU and GAMA-by freezing the model's original weights and training LoRA adapters for the LLM. This approach allows end-users to flexibly enhance the model's reasoning and long-context understanding capabilities on demand.
this section cite: ['b50']

Section: Experiments
Experimental Setup. We train AF3 on 128 NVIDIA A100 GPUs, each with 80GB of memory. Details about batch size, learning rates, and optimizers for each stage of training are in Appendix H.
Baselines. We evaluate our model against recent SOTA LALMs, including GAMA [41], Audio Flamingo [65], Audio Flamingo 2 [40], Qwen-A(udio) [20], Qwen2-A(udio) [19], Qwen2-A(udio)-(Inst)ruct, Qwen2.5-O(mni) [117], R1-AQA [73], Pengi [27], Phi-4-mm [1], Baichun Audio [75], Step-Audio-Chat [52], LTU [46], LTU-AS [45], SALMONN [104], AudioGPT [53], and Gemini (2.0 Flash, 1.5 Pro, 2.5 Flash and 2.5 Pro) [105] (note we do not evaluate Gemini on ASR benchmarks due to low rate limits), as well as GPT-4o-audio [54]. For LongAudioBench, for models that do not support longer audio, we follow the cascaded approach for evaluation proposed by [40]. For Table 3, we only compare against open LALMs. All results reported in the tables correspond to the best-performing model. Evaluation for voice-to-voice capabilities is beyond our scope. Evaluation Datasets. We evaluate AF3 on a variety of tasks and benchmarks, including audio classification (CochlScene [57], NSynth (Source and Instrument) [34], NonSpeech7k [99], IEMOCAP [11]), audio QA (ClothoAQA [76], MusicAVQA [74], Music Instruct [26], LibriSQA [121]), reasoningfocused audio QA (MMAU [101] (v05.15.25), MuchoMusic (perceptual version) [120,110], MMAR [81], MMSU [108], CompA-R-test [42], Audio Entailment [29]), multimodal hallucination
Table 3: Comparison of AF3 with open LALMs on AF-Chat, voice-text and TTS benchmarks. WER ↓ (Word Error Rate), SIM ↑ (Similarity), Human ↑ (Human evaluation) and GPT4o ↑ (GPT evaluation) indicate metrics and whether lower or higher is better.
detection (CMM [72]), audio captioning (Clotho-v2 [32], AudioCaps [60]), ASR (Librispeech (clean and other) [92], SPGISpeech [90], TEDLIUM [100,49], GigaSpeech (Large) [14], Common Voice 15 [5] and Voxpopuli [107]) and long audio captioning and QA (LongAudioBench -which we augment with 2.5k human-annotated long-speech QA instances). For evaluating chat capabilities, we conduct a human study of model outputs on AF-Chat-test (more details in Appendix E) and compare only with Qwen2-Audio. Each annotator is asked to rate the response of the model for every turn on a scale of 1-5 for factuality, usefulness, and depth. We report results averaged across all instances across all turns. Furthermore, we evaluate the voice-text capabilities of our AF3-Chat model on two datasets, OpenAudioBench [75] and VoiceBench [18]. These benchmarks consist of voice queries (synthetically generated speech from text queries) and assess aspects such as instruction following, question answering, trivia knowledge, and reasoning. Finally, we evaluate our speech generation module using zero-shot TTS evaluation on the English subset of the SEED benchmark [4].
To calculate accuracy, we use either exact string matching with the ground truth or CLAP-based retrieval following [27], implemented with open-source AF-CLAP [40]. For MCQ, AF3 typically outputs only the selected option. In cases where the model provides more verbose or open-ended responses (e.g., with thinking mode), we apply multiple regex patterns to extract the chosen option.
this section cite: ['b40', 'b65', 'b39', 'b19', 'b18', 'b117', 'b73', 'b26', 'b0', 'b75', 'b51', 'b45', 'b44', 'b104', 'b52', 'b105', 'b53', 'b39', 'b56', 'b33', 'b99', 'b10', 'b76', 'b74', 'b25', 'b121', 'b101', 'b120', 'b110', 'b81', 'b108', 'b41', 'b28', 'b72', 'b31', 'b60', 'b92', 'b90', 'b100', 'b48', 'b13', 'b4', 'b107', 'b75', 'b17', 'b3', 'b26', 'b39']

Section: Audio Understanding and Reasoning Evaluation
AF3 is the strongest and fully open-source LALM. Table 2 shows AF3 outperforming previous SOTA open-weight and closed-source models across a wide range of audio understanding and reasoning benchmarks. AF3 sets new highs on MMAU (72.42) (note for Qwen2.5-Omni on MMAU we report the "parsed score" for fair evaluation), ClothoAQA (91.1), Clotho Entailment (92.9), and CMM Hallucination (86.7). On tasks like NSynth and MusicInstruct, it shows significant gains, highlighting strong sound and music understanding. For LongAudioBench (sound and speech), AF3 outperforms Gemini 2.5 Pro by a wide margin, demonstrating its strength in long-context reasoning. We also evaluate AF3 with thinking prompts (+Think) on reasoning-heavy benchmarks like MMAU and MuchoMusic, observing a performance boost. Although the thinking mode is activated after Stage 3.5 only when using our specific thinking prompt, the checkpoint remains usable without it. We report average scores of 73.16 and 74.26 on MMAU-test and MMAU-test-mini, respectively. Additionally, AF3 achieves state-of-the-art ASR results on LibriSpeech, SPGISpeech, and VoxPopuli-even compared to dedicated ASR models-despite not being trained on large-scale ASR datasets like many open-weight models. We illustrate a demo of AF3's capabilities in Fig. 14.
this section cite: []

Section: Chat and TTS Evaluation
Multi-turn multi-audio chat evaluation. On AF-Chat-test AF3-Chat shows a relative improvement of 30% over Qwen2.5-Omni, thereby showing the capability of effectively handling extended dialog turns, allowing for deeper contextual reasoning and more accurate references to multiple audio inputs.
Voice-Text and Speech Generation Evaluation. Table 3 evaluates AF3-Chat on two key tasks: voice-to-text and text-to-speech generation. In the voice-to-text setting (spoken QA), AF3-Chat achieves strong gains across all of OpenAudioBench, surpassing Qwen2.5-Omni. On VoiceBench, which tests spoken QA robustness across AdvBench, CommonEval, and OpenBookQA, AF3-Chat performs comparably to Qwen2.5-Omni and Qwen2-Audio Chat. For TTS (evaluated on SEED testen), AF3-Chat shows improved performance with a lower WER of 2.02 (vs. 2.72 for Qwen2.5-Omni) and a speaker similarity score of 0.61, closely matching Qwen2.5's 0.63.
Furthermore, AF3-Chat exhibits significant advantages in generation speed. For a 10-second audio generation on an A100 GPU, AF3-Chat's text-to-audio token generation is 5.94 seconds with an additional 0.02 seconds for waveform synthesis. In comparison, the Talker model of Qwen2.5-Omni requires 14.62 seconds for token generation and an additional 1.26 seconds for waveform synthesis. This efficiency allows our streaming text-to-speech to achieve a time-to-first-token of 0.15 seconds and an inter-token latency of 0.06 seconds (both including waveform synthesis), producing a 10-second audio clip in 6.68 seconds.
this section cite: []

Section: Ablation Studies
In this section, we ablate our key components (using just 10% of the training data) to support the paper's main claims.
Evaluating AF-Whisper as a Unified Encoder. Table 4 compares AF3 trained with our unified AF-Whisper encoder against a dual-encoder setup using CLAP for sounds/music and Whisper-v3 for speech [33,96]. AF-Whisper outperforms the dual-encoder model under the same data budget, demonstrating its effectiveness as a single encoder for sound, music, and speech.
AudioSkills-XL: A Key Dataset for Performance Gains.: To measure the impact of AudioSkills-XL, we ablate it from Stage 3 of training and compare results to the full setup. As shown in Table 4, removing AudioSkills-XL causes a significant performance drop-particularly on MMAU-underscoring its role in improving generalization and robustness. These findings highlight the value of large-scale, skill-targeted audio QA data for fine-tuning multi-modal models. We detail our practices, including architecture, training, inference, and the evaluation pipeline, and open-source two large datasets. For future work, we aim to address current limitations, including:
(1) mitigating the need for a cascaded system for voice chat, (2) making AF3 multi-lingual, and (3) reducing dependency on closed-source models for synthetic data.
this section cite: ['b32', 'b96']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: Detailed experiments and evaluation in Section 6 support our claims.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Limitations of our approach are discussed in Section 7.
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

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
this section cite: []

Section: Answer: [NA]
Justification: There is no theoretical result in the paper. Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We give details about datasets, experimental setup and training hyperparameters in Appendix H. Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All the code, data and models are open sourced at https://github.com/  NVIDIA/audio-flamingo/tree/audio_flamingo_3.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: We provide all the settings in Appendix H.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material.
this section cite: []

Section: Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?
Answer: [No] Justification: It is expensive to run multiple experiments and report such metrics.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: We provide all the details in Appendix H.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: Yes, the paper conforms to the NeurIPS Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: We discuss the impact of our work in Appendix K.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: Our paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: We provide all the details in Appendix G.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes]
Justification: Details about all datasets, including the method for curation, are provided in the main paper. More detailed statistics and examples are provided in the Appendix.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [Yes] Justification: We provide details about this in Appendix E.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
Answer: [Yes] Justification: Most of the human manual analysis was done by the authors of the paper, except chat evaluation, for which we obtained IRB approval from our institution. More details will be provided in the camera-ready.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [Yes] Justification: LLMs have been used only to help in writing and writing code for parts of experimentation. Guidelines: • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM)
for what should or should not be described.
this section cite: ['b15']

Section: References
Ref_id:b0 Title: Compact yet powerful multimodal language models via mixture-of-loras Year: (2025)
Ref_id:b1 Title: Youtube-8m: A large-scale video classification benchmark Year: (2016)
Ref_id:b2 Title: Generating music from text Year: (2023)
Ref_id:b3 Title: Seed-tts: A family of high-quality versatile speech generation models Year: (2024)
Ref_id:b4 Title: Common voice: A massively-multilingual speech corpus Year: (2020)
Ref_id:b5 Title: Audiosetcaps: Enriched audio captioning dataset generation using large audio language models Year: (2024)
Ref_id:b6 Title: The omg-emotion behavior dataset Year: (2018)
Ref_id:b7 Title: The million song dataset Year: (2011)
Ref_id:b8 Title: The million song dataset Year: (2011)
Ref_id:b9 Title: Medleydb: A multitrack dataset for annotation-intensive mir research Year: (2014)
Ref_id:b10 Title: Iemocap: Interactive emotional dyadic motion capture database. Language resources and evaluation Year: (2008)
Ref_id:b11 Title: Sonyc-ust-v2: An urban sound tagging dataset with spatiotemporal context Year: (2020)
Ref_id:b12 Title: Action2sound: Ambient-aware generation of action sounds from egocentric videos Year: (2024)
Ref_id:b13 Title: Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio Year: (2021)
Ref_id:b14 Title: Vggsound: A large-scale audio-visual dataset Year: (2020)
Ref_id:b15 Title: Beats: Audio pre-training with acoustic tokenizers Year: (2022)
Ref_id:b16 Title: Scaling long-context visual language models for long videos Year: (2024)
Ref_id:b17 Title: Voicebench: Benchmarking llm-based voice assistants Year: (2024)
Ref_id:b18 Title: Qwen2-audio technical report Year: (2024)
Ref_id:b19 Title: Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models Year: (2023)
Ref_id:b20 Title: Voxceleb2: Deep speaker recognition Year: (2018)
Ref_id:b21 Title: The fisher corpus: A resource for the next generations of speech-to-text Year: (2004)
Ref_id:b22 Title: 000 podcasts: A spoken English document corpus Year: ()
Ref_id:b23 Title: Toward truly personal chatbots: on the development of custom conversational assistants Year: (2018)
Ref_id:b24 Title: Fma: A dataset for music analysis Year: (2016)
Ref_id:b25 Title: Musilingo: Bridging music and text with pre-trained language models for music captioning and query response Year: (2023)
Ref_id:b26 Title: Pengi: An audio language model for audio tasks Year: (2023)
Ref_id:b27 Title: Audio retrieval with wavtext5k and clap training Year: (2022)
Ref_id:b28 Title: Audio entailment: Assessing deductive reasoning for audio understanding Year: (2025)
Ref_id:b29 Title: Lp-musiccaps: Llm-based pseudo music captioning Year: (2023)
Ref_id:b30 Title: The faiss library Year: (2024)
Ref_id:b31 Title: Clotho: An audio captioning dataset Year: (2020)
Ref_id:b32 Title: Clap: Learning audio concepts from natural language supervision Year: (2022)
Ref_id:b33 Title: Neural audio synthesis of musical notes with wavenet autoencoders Year: (2017)
Ref_id:b34 Title: Mmdialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation Year: (2022)
Ref_id:b35 Title: Fsd50k: an open dataset of humanlabeled sound events Year: (2021)
Ref_id:b36 Title: Freesound datasets: A platform for the creation of open audio datasets Year: (2017)
Ref_id:b37 Title: Chime-home: A dataset for sound source recognition in a domestic environment Year: (2015)
Ref_id:b38 Title: Audio set: An ontology and human-labeled dataset for audio events Year: (2017)
Ref_id:b39 Title: Audio flamingo 2: An audio-language model with long-audio understanding and expert reasoning abilities Year: (2025)
Ref_id:b40 Title: Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities Year: (2024)
Ref_id:b41 Title: Compa: Addressing the gap in compositional reasoning in audio-language models Year: ()
Ref_id:b42 Title: Switchboard: Telephone speech corpus for research and development Year: (1992)
Ref_id:b43 Title: Audio dialogues: Dialogues dataset for audio and music understanding Year: (2024)
Ref_id:b44 Title: Joint audio and speech understanding Year: (2023)
Ref_id:b45 Title: Listen, think, and understand Year: (2023)
Ref_id:b46 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b47 Title: Audioclip: Extending clip to image, text and audio Year: (2021)
Ref_id:b48 Title: Ted-lium 3: Twice as much data and corpus repartition for experiments on speaker adaptation Year: (2018)
Ref_id:b49 Title: The benefit of temporally-strong labels in audio event classification Year: (2021)
Ref_id:b50 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b51 Title: Step-audio: Unified understanding and generation in intelligent speech interaction Year: (2025)
Ref_id:b52 Title: Audiogpt: Understanding and generating speech, music, sound, and talking head Year: (2023)
Ref_id:b53 Title: Gpt-4o system card Year: (2024)
Ref_id:b54 Title: Video recap: Recursive captioning of hour-long videos Year: (2024)
Ref_id:b55 Title: An open source emotional speech corpus for human robot interaction applications Year: (2018)
Ref_id:b56 Title: Cochlscene: Acquisition of acoustic scene data using crowdsourcing Year: (2022)
Ref_id:b57 Title: Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) Year: (2022)
Ref_id:b58 Title: Miradata: A large-scale video dataset with long durations and structured captions Year: (2024)
Ref_id:b59 Title: Libriheavy: A 50,000 hours asr corpus with punctuation casing and context Year: (2024)
Ref_id:b60 Title: Audiocaps: Generating captions for audios in the wild Year: (2019)
Ref_id:b61 Title: Efficient generative modeling with residual vector quantization-based tokens Year: (2024)
Ref_id:b62 Title: Europarl: A parallel corpus for statistical machine translation Year: (2005)
Ref_id:b63 Title: Audio retrieval with natural language queries: A benchmark study Year: (2022)
Ref_id:b64 Title: Libritts-r: A restored multi-speaker text-to-speech corpus. INTER-SPEECH 2023 Year: (2023)
Ref_id:b65 Title: Audio flamingo: A novel audio language model with few-shot learning and dialogue abilities Year: (2024)
Ref_id:b66 Title: High-fidelity audio compression with improved rvqgan Year: ()
Ref_id:b67 Title: Evaluation of algorithms using games: the case of music annotation Year: (2010)
Ref_id:b68 Title: Nv-embed: Improved techniques for training llms as generalist embedding models Year: (2024)
Ref_id:b69 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b70 Title: DiTTo-TTS: Diffusion transformers for scalable text-to-speech without domain-specific factors Year: (2025)
Ref_id:b71 Title: Dailytalk: Spoken dialogue dataset for conversational textto-speech Year: (2023)
Ref_id:b72 Title: The curse of multi-modalities: Evaluating hallucinations of large multimodal models across language, visual, and audio Year: (2024)
Ref_id:b73 Title: Reinforcement learning outperforms supervised fine-tuning: A case study on audio question answering Year: (2025)
Ref_id:b74 Title: Learning to answer questions in dynamic audio-visual scenarios Year: (2022)
Ref_id:b75 Title: Baichuan-audio: A unified framework for end-to-end speech interaction Year: (2025)
Ref_id:b76 Title: Clotho-aqa: A crowdsourced dataset for audio question answering Year: (2022)
Ref_id:b77 Title: Recap: retrieval-enhanced context-aware prefix encoder for personalized dialogue response generation Year: (2023)
Ref_id:b78 Title: Music understanding llama: Advancing textto-music generation with question answering and captioning Year: (2024)
Ref_id:b79 Title: A convnet for the 2020s Year: (2022)
Ref_id:b80 Title: Audio-cot: Exploring chain-of-thought reasoning in large audio language model Year: (2025)
Ref_id:b81 Title: Mmar: A challenging benchmark for deep reasoning in speech, audio, music, and their mix Year: (2025)
Ref_id:b82 Title: The msp-conversation corpus Year: (2020)
Ref_id:b83 Title: Wavcaps: A chatgpt-assisted weakly-labelled audio captioning dataset for audio-language multimodal research Year: (2024)
Ref_id:b84 Title: Toward controllable text-to-music generation Year: (2023)
Ref_id:b85 Title: A multi-device dataset for urban acoustic scene classification Year: (2018)
Ref_id:b86 Title: Language-independent sleepy speech detection Year: (2022)
Ref_id:b87 Title: Diversity and bias in audio captioning datasets Year: (2021)
Ref_id:b88 Title: Position: Levels of agi for operationalizing progress on the path to agi Year: (2024)
Ref_id:b89 Title: Audio retrieval with natural language queries Year: (2021)
Ref_id:b90 Title: Spgispeech: 5,000 hours of transcribed financial audio for fully formatted end-to-end speech recognition Year: (2021)
Ref_id:b91 Title: Mqad: A large-scale question answering dataset for training music large language models Year: (2025)
Ref_id:b92 Title: Librispeech: an asr corpus based on public domain audio books Year: (2015)
Ref_id:b93 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b94 Title: Meld: A multimodal multi-party dataset for emotion recognition in conversations Year: (2018)
Ref_id:b95 Title: Mls: A large-scale multilingual dataset for speech research Year: (2020)
Ref_id:b96 Title: Robust speech recognition via large-scale weak supervision Year: (2022)
Ref_id:b97 Title: The musdb18 corpus for music separation Year: (2017)
Ref_id:b98 Title: Synthetic or not-identifying counterfeit songs Year: (2024)
Ref_id:b99 Title: Nonspeech7k dataset: Classification and analysis of human non-speech sound Year: (2023)
Ref_id:b100 Title: Ted-lium: an automatic speech recognition dedicated corpus Year: (2012)
Ref_id:b101 Title: Mmau: A massive multi-task audio understanding and reasoning benchmark Year: (2024)
Ref_id:b102 Title: Music4all: A new music database and its applications Year: (2020)
Ref_id:b103 Title: Vocos: Closing the gap between time-domain and fourier-based neural vocoders for high-quality audio synthesis Year: ()
Ref_id:b104 Title: Salmonn: Towards generic hearing abilities for large language models Year: (2023)
Ref_id:b105 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b106 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b107 Title: Voxpopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation Year: (2021)
Ref_id:b108 Title: Mmsu: A massive multi-task spoken language understanding and reasoning benchmark Year: (2025)
Ref_id:b109 Title: Multimodal chain-of-thought reasoning: A comprehensive survey Year: (2025)
Ref_id:b110 Title: Muchomusic: Evaluating music understanding in multimodal audio-language models Year: (2024)
Ref_id:b111 Title: Chain-ofthought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b112 Title: Wav2clip: Learning robust audio representations from clip Year: (2021)
Ref_id:b113 Title: Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation Year: (2023)
Ref_id:b114 Title: Audio-reasoner: Improving reasoning capability in large audio language models Year: (2025)
Ref_id:b115 Title: Mini-omni: Language models can hear Year: (2024)
Ref_id:b116 Title: Llava-o1: Let vision language models reason step-by-step Year: (2024)
Ref_id:b117 Title: Qwen2. 5-omni technical report Year: (2025)
Ref_id:b118 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b119 Title: Sound-vecaps: Improving audio generation with visually enhanced captions Year: (2025)
Ref_id:b120 Title: Are you really listening? boosting perceptual awareness in music-qa benchmarks Year: (2025)
Ref_id:b121 Title: Librisqa: Advancing free-form and open-ended spoken question answering with a novel dataset and framework Year: (2023)
Ref_id:b122 Title: Less is more for alignment Year: (2023)
