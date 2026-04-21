Title: VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction
Abstract: Recent Multimodal Large Language Models (MLLMs) have typically focused on integrating visual and textual modalities, with less emphasis placed on the role of speech in enhancing interaction. However, speech plays a crucial role in multimodal dialogue systems, and implementing high-performance in both vision and speech tasks remains a challenge due to the fundamental modality differences. In this paper, we propose a carefully designed multi-stage training methodology that progressively trains LLM to understand both visual and speech information, ultimately enabling fluent vision and speech interaction. Our approach not only preserves strong vision-language capacity, but also enables efficient speech-to-speech dialogue capabilities without separate ASR and TTS modules, significantly accelerating multimodal end-to-end response speed. By comparing against state-of-the-art counterparts across benchmarks for image, video, and speech, we demonstrate that our omni model is equipped with both strong visual and speech capabilities, making omni understanding and interaction.

Section: Introduction
Recent advancements in MLLMs [1,2,3,4,5,6,7,8,9] have led to significant progress, particularly in integration of visual and textual modalities. The introduction of visual information into LLMs has notably enhanced model capabilities across various multimodal tasks. However, with the growing appeal of human-computer interaction, the role of the speech modality has become increasingly prominent, especially in multimodal dialogue systems. In such a system, speech not only serves as a key medium for information transmission but also greatly improves the naturalness and convenience of interactions. Consequently, integrating visual and speech modalities to achieve multimodal interactions has emerged as a critical research focus.
The integration of vision and speech in MLLMs is not straightforward due to their inherently differences [10]. For example, visual data, such as images, convey spatial information, while speech data convey dynamic changes in time series. These fundamental differences pose challenges for simultaneous optimization of both modalities, often leading to conflicts during training. For instance, the inclusion of speech data may degrade performance on vision tasks, and vice versa. In addition, traditional speech-to-speech systems rely on separate modules for Automatic Speech Recognition (ASR) and Text-to-Speech, which can increase latency and reduce coherence, limiting their practicality in real-time applications [11,12,13,14,15].
In this paper, we introduce VITA-1.5, a multimodal LLM that integrates vision, language, and speech through a carefully designed three-stage training methodology. The training strategy progressively incorporates vision and speech data, relieving modality conflicts while maintaining strong multimodal performance. In the first stage, we focus on vision-language by training visual adapters and finetuning the model with descriptive caption and visual QA data. This step establishes the model's foundational visual capabilities, enabling robust image and video understanding. The second stage introduces audio input processing by training an audio encoder using speech-transcription paired data, followed by fine-tuning with speech QA data. This stage equips the model with the ability to understand and respond to audio inputs effectively. Finally, in the third stage, we train an audio decoder to enable end-to-end speech output, eliminating the need for external TTS modules. This allows VITA-1.5 to generate fluent speech replies, enhancing the naturalness and interactivity of multimodal dialogue systems.
We have conducted extensive evaluations on various benchmarks related to image, video, and speech understanding, comparing the results with both open-source and proprietary models. VITA-1.5 demonstrates comparable perception and reasoning capabilities comparable to leading image/video based MLLMs, and shows significant improvements in the speech capability.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14']

Section: Related Work
Recently, thanks to the rapid development of language models such as GPTs [16,17], LLaMA [18,19], Alpaca [20], Vicuna [21], and Mistral [22], researchers have successfully extended text comprehension to multimodal understanding/reasoning through techniques like multimodal alignment and instruction tuning. For example, models such as LLaVA [1], Qwen-VL [23], Cambrian-1 [24], Mini-Gemini [25], MiniCPM-V 2.5 [26], DeepSeek-VL [27], and SliME [28] have made significant advances in image perception and reasoning, while models like LongVA [29] and Video-LLaVA [30] have showcased the latest progress in video understanding. These models are increasingly capable of handling diverse data types, driving the continuous improvement of multimodal perception and understanding capabilities.
Beyond visual modalities, recent years have also witnessed significant progress in incorporating speech capabilities into LLMs, driven by the increasing demand for natural human-computer interaction. The dominant approach has been to cascade ASR, LLM, and TTS modules. This text-centric approach faces fundamental limitations due to the loss of paralinguistic features like tones and emotions. While works like [31] and [32] have attempted to address these issues by incorporating speech encoders and emotion vectors, they still rely on speech transcription, resulting in substantial latency issues that impact the user experience. The emergence of proprietary models like GPT-4o [33] has demonstrated the possibility of end-to-end speech interaction, inspiring a new wave of research in speech-enabled MLLMs. Following this trend, several notable works have emerged in the opensource community. Models such as Mini-Omni2 [34], LLaMA-Omni [35], and Moshi [36] have explored various strategies for aligning speech modality with LLMs and achieving duplex dialogue capabilities. While these open-source efforts have successfully enabled duplex speech interaction with LLMs, they still lack the capability to handle visual modalities as demonstrated by GPT-4o, limiting their applications in scenarios requiring both visual and speech understanding.
Despite these advances in both visual and speech modalities, a significant gap remains between proprietary and open-source models. Compared to proprietary models that support multiple modalities, including audio, image, and text, e.g., GPT-4o [37] and Gemini-Pro 1.5 [38], most open-source models have primarily focused on image and text modalities [2]. Moreover, few open-source models have involved multimodal interaction capabilities, which is a relatively unexplored area. While works like VITA-1.0 [12] have made initial attempts to introduce speech for human-computer interaction, introducing additional speech data poses challenges to the model's original multimodal abilities. Furthermore, speech generation typically relies on existing TTS systems, which often results in high latency, thus impacting user experience. In this paper, we present VITA-1.5 that leverages refined training strategies, excelling in perceiving data across four modalities (video, image, text, and audio), while also realizing near real-time vision and speech interaction.
3 VITA-1.5 Vision Encoder Image Speech Vision Adapter Speech Encoder Speech Adapter NAR Speech Decoder AR Speech Decoder … Discrete Token Text Speech Figure 2: Overall Architecture of VITA-1.5. The input side consists of vision and audio encoders, along with their adapters. The output side has an end-to-end speech generation module, rather than directly using an TTS model.
this section cite: ['b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b0', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b1', 'b11']

Section: Model Architecture

this section cite: []

Section: Codec
The overall architecture of VITA-1.5 is depicted in Fig. 2. The input side is the same as that of the VITA-1.0 version [12], that is, adopting the configuration of "Multimodal Encoder-Adaptor-LLM". It combines the Vision/Audio Transformer and the Multi-Layer Connector with an LLM for joint training, aiming to enhance the unified understanding of vision, language, and audio. With respect to the output side, VITA-1.5 has its own end-to-end speech module, instead of using the external TTS model like the original VITA-1.0 version.
this section cite: ['b11']

Section: Visional Modality
Visual Encoder. VITA-1.5 adopts InternViT-300Mfoot_0 as the visual encoder, with an input image size of 448×448 pixels, generating 256 visual tokens per image. For high-resolution images, VITA-1.5 employs a dynamic patching [39] strategy to capture local details, improving the accuracy of image understanding.
this section cite: ['b38']

Section: Video Processing.
Videos are treated as a special type of multipleimage input. If the video length is shorter than 4 seconds, 4 frames are uniformly sampled; for videos between 4 and 16 seconds, one frame per second is sampled; for videos longer than 16 seconds, 16 frames are uniformly sampled. No dynamic patching is applied to video frames to avoid excessive visual tokens that could hinder processing efficiency.
this section cite: []

Section: Vision Adapter.
A two-layer MLP is used to map the visual features to visual tokens suitable for the subsequent understanding of LLM.
this section cite: []

Section: Audio Modality
Speech Encoder. Similar to [40], our audio encoding module consists of multiple downsampling convolutional layers (4x downsampling) and 24 Transformer blocks (with a hidden size of 1024). The downsampling layers help reduce the frame rate of the audio features, improving the processing speed of LLM. The audio encoder has about 350M parameters and an output frame rate of 12.5Hz. Mel-filter bank features are used as the input of the audio encoder, with a window size of 25ms and a shift of 10ms [40].
this section cite: ['b39', 'b39']

Section: Speech Adapter. It consists of multiple convolutional layers with 2x downsampling.
Speech Decoder. TiCodec [41] is used as our codec model, customizing a single codebook with a size of 1024. This single-codebook design simplifies the decoding process during the inference phase.
The codec model is responsible for encoding continuous speech signals into discrete speech tokens with the frequency of 40Hz, and at the same time has the ability to decode them back into speech signals with the sample rate of 24,000Hz.
The current LLM can only output text tokens, and the speech generation capability requires the LLM to be able to output speech tokens. To this end, we add two speech decoders after the text tokens following [40]: 1) Non-Autoregressive (NAR) Speech Decoder, which processes text tokens globally and models semantic features, with the aim of generating an initial distribution of speech tokens; 2) Autoregressive (AR) Speech Decoder generates higher quality speech tokens step by step, based on the speech information produced by the NAR decoder. The final sequence of speech tokens is then decoded into a continuous speech signal flow (waveform) using the speech decoder of the Codec model. We adopt 4 LLaMA decoder layers for both NAR and AR speech decoders, where the hidden size is 896 and the parameter size is about 120M.
this section cite: ['b40', 'b39']

Section: Training Data
As shown in Table 1, the training data of multimodal instruction tuning encompass a wide range of categories, such as caption data and QA data, both Chinese and English. During different training phases, subsets of the overall dataset are selectively sampled to serve different objectives. Specifically, the datasets are categorized as follows:
• Image Captioning Data. Datasets such as ShareGPT4V [42], ALLaVA-Caption [43], SharedGPT4o-Imagefoot_1 , and synthetic data are used to train the model to generate descriptive languages for images.
• Image QA Data. Datasets like LLaVA-150Kfoot_2 , LLaVA-Mixture-sample [1], LVIS-Instruct [44], ScienceQA [45], ChatQA [46], and subsets sampled from LLaVA-OV [47], such as general image QA and mathematical reasoning datasets, are utilized to train the model in answering image-based questions and performing visual reasoning tasks.
• OCR & Diagram Data. This category supports the model in understanding OCR and diagram content, using datasets such as Anyword-3M [48], ICDAR2019-LSVT 4 , UReader [49], SynDOGfoot_4 , ICDAR2019-LSVT-QA 6 , and corresponding data sampled from LLaVA-OV.
• Video Data. Datasets like ShareGemini [50] and synthetic data are used to train the model to handle video inputs and perform tasks such as captioning and video-based QA.
• Pure Text Data. This category enhances the model's capability to understand and generate languages, facilitating text-based QA tasks.
In addition to the image and video data listed in Table 1, 110,000 hours of internal speech-transcription paired ASR data, covering both Chinese and English, are incorporated to train the audio encoder and align the audio encoder with the LLM. Furthermore, 3,000 hours of text-speech paired data generated by a TTS system are used to train the speech decoder.
Table 1: Training data of multimodal instruction tuning. The images of the synthetic data come from open-source datasets like Wukong [51], LAION [52], and CC12M [53].
Data Scenario QA Type Dataset Name Questions (K) Language General Image Description ShareGPT4V 99.50 Eng ALLaVA-Caption 697.40 Eng ShareGTP4o-Image 55.50 Eng Synthetic Data 593.70 CN QA LLaVA-150K 218.36 CN LLaVA-Mixture-sample 1872.10 Eng LVIS-Instruct 939.36 Eng ScienceQA 12.72 Eng ChatQA 7.39 Eng LLaVA-OV General 1754.65 Eng LLaVA-OV Math Reasoning 1140.92 Eng Synthetic Data 212.68 CN OCR & Diagram Description Anyword-3M 1709.30 CN ICDAR2019-LSVT 366.30 CN UReader 100.00 Eng SynDOG-EN 100.00 Eng SynDOG-CN 101.90 CN QA ICDAR2019-LSVT-QA 630.08 CN LLaVA-OV Doc Chart Screen 4431.50 Eng LLaVA-OV General OCR 404.20 Eng General Video Description ShareGemini 205.70 CN Synthetic Data 569.40 CN & Eng QA Synthetic Data 4336.30 CN & Eng Pure Text QA Synthetic Data 1574.20 CN & Eng Total 22133.16 CN & Eng
this section cite: ['b41', 'b42', 'b0', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52']

Section: Three Stage Training Strategies
In order to ensure that VITA-1.5 performs well in tasks involving vision, language, and audio, we have to face a key challenge, i.e., training conflicts between different modalities. For example, adding the speech data could negatively impact the understanding of the vision data, as the features of speech differ significantly from those of vision, causing interference during the learning process. To address this challenge, we devise a three-stage training strategy as shown in Fig. 3. The core idea is to gradually introduce different modalities into the model, allowing it to increase the power of a new modality while maintaining the power of the existing modalities.
this section cite: []

Section: Stage 1: Vision-Language Training Stage 1.1 Vision Alignment.
In this stage, our goal is to bridge the gap between vision and language. The features of the former are extracted from the pre-trained vision encoder InternViT-300M, and the latter is introduced through the LLM. We use 20% of the descriptive caption data from Table 1 for training, where only the visual adapter is trainable, while the other modules are frozen. This approach allows the LLM to initially align the visual modality.
this section cite: []

Section: Stage 1.2 Vision Understanding.
In this stage, our goal is to teach the LLM to transcribe image content. Toward this end, we use all the descriptive caption data from Table 1. During this process, the encoder and adapter of the visual module, as well as the LLM, are trainable. The focus is to enable the model to establish a strong connection between vision and language by learning from descriptive texts about images, allowing it to understand image content via generating natural language descriptions.
this section cite: []

Section: Stage 1.3 Vision SFT.
Following Stage 1.2, the model has acquired a basic understanding of images and videos. However, the instruction following ability is still limited, and it is difficult to cope with the visual QA task. To achieve this, we use all the QA data from Table 1 while retaining 20% of the descriptive caption data to increase the diversity of the dataset and the complexity of the tasks.
During training, the encoder and adapter of the visual module, as well as the LLM, are trainable. The key objective of this stage is to enable the model not only to understand visual content but also to answer questions following instructions. 2). The percentages shown in the image correspond to the data sampling ratios specified in Table 1.
this section cite: []

Section: Stage 2: Audio Input Tuning
Stage 2.1 Audio Alignment. After completing the training of Stage 1, the model has developed a strong foundation in image and video understanding. In this stage, our goal is to reduce the discrepancy between audio and language based on Stage 1, enabling the LLM to understand audio inputs. The training data consists of 11,000 hours of speech-transcription pairs. We follow a two-step approach: (a) Speech Encoder Training: We adopt a training framework used in common speech recognition systems, using a Connectionist Temporal Classification (CTC) loss function [54] to train the speech encoder. The aim is for the encoder to predict the transcription text from the speech input. This step ensures that the audio encoder can extract speech features and map them to the text representation space. (b) Speech Adapter Training: After training the speech encoder, we integrate it with the LLM, using an audio adapter to introduce audio features into the input layer of the LLM. The training objective at this stage is to enable the LLM to output the transcription text of the speech data.
Besides, in step (b), we introduce special trainable input tokens to guide the speech understanding process. These tokens provide additional contextual information that guides the LLM used for the QA task to perform the ASR task.
this section cite: ['b53']

Section: Stage 2.2 Audio SFT.
The focus of this stage is to introduce the QA functionality with speech questions and text answers. To achieve this, we sample 4% of the caption data and 20% of the QA data from Table 1. In terms of data processing, approximately half of the text-based questions are randomly replaced with their corresponding speech versions, generated using a TTS system.
In this stage, both the visual encoder and adapter, the audio encoder and adapter, as well as the LLM are trainable, aiming to improve the model's adaptability with multimodal inputs. In addition, we add a classification head to the LLM's output. This head is used to distinguish whether the input comes from speech or text. As a result, the model can more accurately interpret speech inputs and process different modalities efficiently and flexibly.
this section cite: []

Section: Stage 3: Audio Output Tuning
In the first two stages of training, the VITA-1.5 model has effectively developed its multimodal understanding capabilities. However, a crucial capacity, i.e., speech output, remains absent, which is essential for its role as an interactive assistant. To introduce speech output functionality without compromising the model's fundamental abilities, we draw on the strategy [40], using 3,000 hours of text-speech data and employing a two-step training approach (see Fig. 3).
this section cite: ['b39']

Section: Stage 3.1 Codec Training.
The goal of this step is to train a codec model with a single codebook using speech data. The encoder of the codec model has the ability to map speech to discrete tokens, while the decoder can map the discrete tokens back to speech stream. During the inference phase of VITA-1.5, only the decoder is used.
this section cite: []

Section: Stage 3.2 NAR + AR Decoder Training.
The training of this stage uses text-speech paired data, where the text is fed into the tokenizer and the embedding later of the LLM to obtain its embedding vectors, and the speech is fed into the encoder of the codec model to obtain its speech tokens. The text embedding vectors are sent to the NAR speech decoder to get global semantic features, and then the features are sent to the AR speech decoder, which predicts the corresponding speech tokens. Note that the LLM is frozen during this stage, thus the multimodal performance is not affected.
this section cite: []

Section: Evaluation

this section cite: []

Section: Vision-Language Evaluation
Baselines. We compare a series of open-source MLLMs, including VILA-1.5 [55], LLaVA-Next [56], CogVLM2 [57], InternLM-XComposer2.5 [58], Cambrian-1 [24], MiniCPM-V-2.6 [26], Ovis1.5 [59], InternVL-Chat-1.5, InternVL-2 [60], LLaVA-OV [47], and Video-LLaVA [30], SliME [28], and LongVA [29], as well as 5 closed-source MLLMs, including GPT-4Vfoot_6 , GPT-4ofoot_7 , GPT-4o-mini, Gemini 1.5 Pro [38], and Claude 3.5 Sonnetfoot_8 . Evaluation Benchmarks. To assess the image perception and understanding capabilities of VITA-1.5, we utilize several evaluation benchmarks, including MME [61], MMBench [62], MMStar [63], MMMU [64], MathVista [65], HallusionBench [66], AI2D [67], OCRBench [68], and MMVet [69]. These benchmarks cover a wide range of aspects, including general multimodal capabilities (e.g., MME, MMBench, and MMMU), mathematical reasoning (MathVista), hallucination detection (HallusionBench), chart (AI2D) and OCR (OCRBench) understanding, providing a comprehensive evaluation results. For video understanding, we use representative evaluation benchmarks including Video-MME [70], MVBench [71], and TempCompass [72].
Vision-Language Capabilities. Table 2 presents a comparison of VITA-1.5's image understanding performance. After the training of the three stages, VITA-1.5 performs comparably to the most advanced open-source models and even surpasses some closed-source models like GPT-4V and GPT-4o-mini. This result highlights the robust capabilities of VITA-1.5 in image-language tasks. As shown in Table 3, VITA-1.5 shows comparable performance to the top open-source models in the evaluation of video understanding. The notable gap compared to proprietary models suggests that VITA-1.5 still has significant room for improvement and potential for further enhancement in video understanding. Please note that after the training of Stages 2 (Audio Input Tuning) and 3 (Audio Output Tuning), VITA-1.5 retains almost its original visual-language capabilities in Stage 1 (Vision-Language Training).
this section cite: ['b54', 'b55', 'b56', 'b57', 'b23', 'b25', 'b58', 'b59', 'b46', 'b29', 'b27', 'b28', 'b37', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b71']

Section: Speech Evaluation
Baselines. The following three baseline models are used for comparison: Wav2vec2-base [73], Mini-Omni2 [74], Freeze-Omni [40], and VITA-1.0 [12].
Evaluation Benchmarks. The Mandarin Evaluation Sets consists of three datasets: aishell-1 [75], test net [76], and test meeting [77]. These datasets are used to evaluate the model's performance on Mandarin speech. The evaluation metric is the Character Error Rate (CER). The English Evaluation Sets include four datasets: dev-clean, dev-other, test-clean, and test-other [78], which are used to evaluate the model's performance on English speech. The evaluation metric is Word Error Rate (WER). The evaluation results in Table 4 indicate that VITA-1.5 achieves leading accuracy in both Mandarin and English ASR tasks. This demonstrates that VITA-1.5 has successfully integrated advanced speech capability to support multimodal interaction.
this section cite: ['b72', 'b73', 'b39', 'b11', 'b74', 'b75', 'b76', 'b77']

Section: Conclusion and Future Work
In this paper, we has presented VITA-1.5, a multimodal LLM designed to integrate vision and speech through a carefully crafted three stage training strategy. By relieving the inherent conflicts between modalities, VITA-1.5 achieves robust capabilities in both vision and speech understanding, enabling efficient speech-to-speech interactions without relying on separate ASR or TTS modules. Extensive evaluations demonstrate that VITA-1.5 performs competitively across multimodal benchmarks. We hope that VITA-1.5 can promote the progress of open-source models in the field of real-time multimodal interaction. Although VITA-1.5 has made some contributions, such as multi-modality joint training, end-to-end architecture, response latency, and basic performance, there are two major areas that can be improved in our future work:
1. Personalized MLLM. Currently, VITA-1.5 is generic and do not incorporate individual preferences during interaction. For example, after learning about personal preferences in the interaction, the content and manner of answers can be adjusted accordingly.
2. Long-term memory. The process of human-computer interaction can last 10 minutes or even several hours, in which case it is important for the human-computer interaction in real scenarios.
this section cite: []

Section: References
Ref_id:b0 Title: Visual instruction tuning Year: (2023)
Ref_id:b1 Title: Unified multimodal llm with discrete sequence modeling Year: (2024)
Ref_id:b2 Title: Video-llama: An instruction-tuned audio-visual language model for video understanding Year: (2023)
Ref_id:b3 Title: Towards generating whole-body motion from speech Year: (2025)
Ref_id:b4 Title: video-salmonn: Speech-enhanced audio-visual large language models Year: (2024)
Ref_id:b5 Title: Audio-visual llm for video understanding Year: (2023)
Ref_id:b6 Title: Dycrowd: Towards dynamic crowd reconstruction from a large-scene video Year: (2025)
Ref_id:b7 Title: Empowering llms with pseudo-untrimmed videos for audio-visual temporal understanding Year: (2025)
Ref_id:b8 Title: An overview of large ai models and their applications Year: (2024)
Ref_id:b9 Title: Improving multimodal speech recognition by data augmentation and speech representations Year: (2022)
Ref_id:b10 Title: Speech-to-text and text-to-speech recognition using deep learning Year: (2023)
Ref_id:b11 Title: Towards open-source interactive omni multimodal llm Year: (2024)
Ref_id:b12 Title: Speechgpt: Empowering large language models with intrinsic cross-modal conversational abilities Year: (2023)
Ref_id:b13 Title: Geometry-guided dense perspective network for speech-driven facial animation Year: (2021)
Ref_id:b14 Title: Logavatar: Local gaussian splatting for human avatar modeling from monocular video Year: (2025)
Ref_id:b15 Title:  Year: (2023)
Ref_id:b16 Title: Language models are few-shot learners Year: (2020)
Ref_id:b17 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b18 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b19 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b20 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-04)
Ref_id:b21 Title: Mistral 7b Year: (2023)
Ref_id:b22 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b23 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b24 Title: Mini-gemini: Mining the potential of multi-modality vision language models Year: (2024)
Ref_id:b25 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b26 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b27 Title: Beyond llava-hd: Diving into high-resolution large multimodal models Year: (2024)
Ref_id:b28 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b29 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b30 Title: Emotion-sensitive spoken dialogue system with large language models Year: (2023)
Ref_id:b31 Title: Paralinguistics-enhanced large language modeling of spoken dialogue Year: (2023)
Ref_id:b32 Title:  Year: (2024)
Ref_id:b33 Title: Towards open-source gpt-4o model with vision, speech and duplex Year: (2024)
Ref_id:b34 Title: Llama-omni: Seamless speech interaction with large language models Year: (2024)
Ref_id:b35 Title: Moshi: a speech-text foundation model for real Year: (2024)
Ref_id:b36 Title:  Year: (2023)
Ref_id:b37 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b38 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b39 Title: Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm Year: (2024)
Ref_id:b40 Title: Fewer-token neural speech codec with time-invariant codes Year: (2024)
Ref_id:b41 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2023)
Ref_id:b42 Title: Allava: Harnessing gpt4vsynthesized data for a lite vision-language model Year: (2024)
Ref_id:b43 Title: To see is to believe: Prompting gpt-4v for better visual instruction tuning Year: (2023)
Ref_id:b44 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b45 Title: Chatqa: Surpassing gpt-4 on conversational qa and rag Year: (2024)
Ref_id:b46 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b47 Title: Anytext: Multilingual visual text generation and editing Year: (2023)
Ref_id:b48 Title: Ureader: Universal ocr-free visually-situated language understanding with multimodal large language model Year: (2023)
Ref_id:b49 Title: Sharegemini: Scaling up video caption data for multimodal large language models Year: (2024-06)
Ref_id:b50 Title: Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark Year: (2022)
Ref_id:b51 Title: Laion-5b: An open large-scale dataset for training next generation image-text models Year: (2022)
Ref_id:b52 Title: Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts Year: (2021)
Ref_id:b53 Title: Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks Year: (2006)
Ref_id:b54 Title: On pre-training for visual language models Year: (2023)
Ref_id:b55 Title: Llava-next: Stronger llms supercharge multimodal capabilities in the wild Year: (2024-05)
Ref_id:b56 Title: Cogvlm2: Visual language models for image and video understanding Year: (2024)
Ref_id:b57 Title: Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition Year: (2023)
Ref_id:b58 Title: Ovis: Structural embedding alignment for multimodal large language model Year: (2024)
Ref_id:b59 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2023)
Ref_id:b60 Title: A comprehensive evaluation benchmark for multimodal large language models Year: (2023)
Ref_id:b61 Title: Is your multi-modal model an all-around player? arXiv preprint Year: (2023)
Ref_id:b62 Title: Are we on the right way for evaluating large vision-language models? arXiv preprint Year: (2024)
Ref_id:b63 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b64 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2023)
Ref_id:b65 Title: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models Year: (2024)
Ref_id:b66 Title: Ai2d-rst: A multimodal corpus of 1000 primary school science diagrams Year: (2021)
Ref_id:b67 Title: On the hidden mystery of ocr in large multimodal models Year: (2023)
Ref_id:b68 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b69 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b70 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b71 Title: Tempcompass: Do video llms really understand videos? arXiv preprint Year: (2024)
Ref_id:b72 Title: wav2vec 2.0: A framework for self-supervised learning of speech representations Year: (2020)
Ref_id:b73 Title: Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex capabilities Year: (2024)
Ref_id:b74 Title: Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline Year: (2017)
Ref_id:b75 Title: Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio Year: (2021)
Ref_id:b76 Title: Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition Year: (2022)
Ref_id:b77 Title: Librispeech: an asr corpus based on public domain audio books Year: (2015)
