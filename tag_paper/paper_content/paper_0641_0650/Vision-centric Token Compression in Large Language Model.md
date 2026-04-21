Title: Vision-centric Token Compression in Large Language Model
Abstract: Real-world applications are stretching context windows to hundreds of thousand of tokens while Large Language Models (LLMs) swell from billions to trillions of parameters. This dual expansion send compute and memory costs skyrocketing, making token compression indispensable. We introduce VISION CENTRIC TOKEN COMPRESSION (VIST), a slow-fast compression framework that mirrors human reading: the fast path renders distant tokens into images, letting a frozen, lightweight vision encoder skim the low-salience context; the slow path feeds the proximal window into the LLM for fine-grained reasoning. A Probability-informed Visual Enhancement (PVE) objective masks high-frequency tokens during training, steering the Resampler to concentrate on semantically rich regions-just as skilled reader gloss over function words. On eleven in-context learning benchmarks, VIST achieves the same accuracy with 2.3× fewer tokens, cutting FLOPs by 16% and memory by 50%. This method delivers remarkable results, outperforming the strongest text encoder-based compression method CEPE by 7.6% on average over benchmarks like TriviaQA, NQ, PopQA, NLUI, and CLIN, setting a new standard for token efficiency in LLMs. The project is at https://github.com/CSU-JPG/VIST.

Section: Introduction
Large language models (LLMs) excel at short snippets, yet many real-world tasks, e.g., long-document understanding [1,2] and question answering [3,4]-already require inputs far beyond the thousandtoken regimes of early GPT-3 [1]. At the same time, parameter counts have leapt from billions to trillions [5,6,7]. In this dual squeeze of longer context & larger models, compression shifts from a convenience to a necessity: without shrinking the input, even the most powerful LLM cannot afford to reason over the information we want it to see.
Psycholinguistics shows that our eyes dance across text: we fixate on rare, content-rich words and skip almost one-third of high-frequency function words [8,9,10]. This selective-reading strategy forms a natural slow-fast circuit. A fast visual pass skims distant, low-salience context to maintain global context, while a slow cognitive pass focuses on nearby sentences that matter. (Figure 1 (a)).
Motivated by this circuit, we present VISION CENTRIC TOKEN COMPRESSION (VIST), a slow-fast token compression framework that mirrors human skimming. As illustrated in Figure 1 (b), VIST first converts loosely relevant long context into images, which are processed by a frozen vision encoder and a trainable Resampler to produce semantically compact visual tokens. These compressed tokens and the main input tokens are then consumed by the LLM. In this slow-fast setup, the vision encoder   acts like the human eye-selectively attending to salient information-while the LLM functions as the brain, concentrating on the most informative content for deeper reasoning.
Specifically, the frozen visual encoders (e.g., CLIP [11]) trained on paired image-text data naturally acquire OCR capabilities [11,12], making them a powerful tool for image-based text understanding. However, the inherent redundancy in long text leads to redundant visual tokens. To address the problem, we design Probability-informed Visual Enhancement (PVE), a contrastive scheme that enforces Resampler to prioritize informative content over redundancy. Concretely, PVE applies frequency-based masking strategy to text token embeddings from the LLM tokenizer, suppressing high-frequency (less informative) text tokens. This semantically rich text supervision guides the Resampler to focus on informative content, bridging the semantic gap between visual and text tokens, and enabling more effective token compression. Unlike previous work [13,14,15,16,17] that rely on LLMs to compute token-level information entropy for assessing importance, VIST adopts token frequency as a simple yet effective proxy, and further reveals rare tokens are key contributors to overall semantic meaning (cf. Figure 3 and §5).
VIST leverages a lightweight vision encoder to compress loosely relevant long contexts, offering a cost-efficient alternative to full-scale LLM computation. Furthermore, the vision encoder serves as a visual text tokenizer, offering several compelling advantages over traditional text tokenizers. ❶ Simplified Tokenization. Text tokenizers rely on complex tokenization rules and vocabulary constraints, typically involving nearly ten human-defined preprocessing steps (e.g., lowercasing, punctuation and stop word removal, and tokenization) [18]. However, the vision encoder processes text more directly by treating rendered text images as visual inputs. ❷ Vocabulary Bottleneck Mitigation. Text tokenization, constrained by a finite vocabulary, becomes a bottleneck when scaling to many languages. A larger vocabulary increases memory and computational costs in the embedding matrix and output layer. However, vision encoder eliminates the need for text tokenizers and unifies various languages into a single image format that removes the need for a vocabulary [19,20]. ❸ Robustness to Character-Level Noise. Vision encoders are more resilient to typos and low-level orthographic attacks, as they capture holistic visual patterns rather than relying on discrete token matching [20]. ❹ Multilingual Efficiency. While our work focuses on English, visual text tokenizer can reduce the number of tokens compared to traditional text tokenizer for languages (e.g., 62% for Japanese, 78% for Korean, and 27% for Chinese). This reduction is particularly impactful in long-text scenarios. Taken together, leveraging vision encoders for long-context compression is a promising and worthwhile direction to explore.
To validate the effectiveness of VIST, we primarily compare with the text-encoder-based token compression counterpart CEPE [21]. VIST requires 2.3× fewer visual tokens than text tokens for the same input, reducing FLOPs by 16% and memory usage by 50%. VIST also delivers consistent gains over CEPE on both In-Context Learning and Open-domain Question Answering tasks, with average gains of 3.6% across 11 datasets and 5.7% across 3 datasets, respectively-highlighting the effectiveness of visual representations for long-context modeling in LLMs.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b0', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b19', 'b20']

Section: Related Work
Token Compression. There has been a growing interest in expanding the context window for LLMs. A line of methods leverages LLM itself to compress raw long input. One may classify these works into two principal groups. i) soft prompt-based methods that adapt LLMs to compress context into fewer tokens [22,23,24,25,26]. ii) selection-based methods that remove redundant tokens based on information entropy computed by LLMs [13,14,15,16,17,27]. All the long inputs typically need to be handled by the heavy LLMs, which incur high costs. Another line of work [28,29] augments LLMs with the capacity to memorize previous long context information by external memory bank and retrieve relevant knowledge [30,31,32,33]. Our method is orthogonal to these existing strategies and can be combined with them to achieve longer context length. The most related work is CEPE [21], which employs a lightweight text encoder to handle long contexts and integrates the information into LLM via cross-attention. While CEPE reduces workload on the LLM, it overlooks the redundancy in long text, making it harder for LLMs to effectively allocate attention to key content. In contrast, VIST compresses long text into compact visual tokens guided by high-density semantic text tokens.
Vision-centric Method. Text tokenization [34,35,36] breaks down text into tokens, serving as a fundamental step in natural language processing. However, tokenization-based methods lack robustness against spelling errors and face vocabulary bottlenecks. A new line of work tackles these issues in a tokenizer-free paradigm [37,20,38]. The representative method Pixel [20] renders text as images and learns to reconstruct masked image patches at the pixel level. It demonstrates strong cross-language translation capabilities and tolerance for text perturbation. Along this direction, recent work explores different pre-training objectives [39,40], e.g., contrastive learning [41], patch-and-text prediction [38]. Despite advancements, these methods overlook long-text scenarios and rely on complicated training pipelines, e.g., OCR-based text understanding [19]. In contrast, VIST directly processes text images by leveraging a vision encoder pretrained on image-text pairs with strong OCR capabilities, and enhancing visual features using enriched text embeddings from LLM tokenizer. An emerging family of multimodal methods [42,43,44,45,46] leverage visual representations to process text and images together, enabling a wide range of applications involving visually-situated text, e.g., webpage parsing [43], tables images analysis [47], and document understanding [42,48].
In this work, we explore incorporating long-context information into LLMs from a visual perspective.
this section cite: ['b21', 'b22', 'b23', 'b24', 'b25', 'b12', 'b13', 'b14', 'b15', 'b16', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b32', 'b20', 'b33', 'b34', 'b35', 'b36', 'b19', 'b37', 'b19', 'b38', 'b39', 'b40', 'b37', 'b18', 'b41', 'b42', 'b43', 'b44', 'b45', 'b42', 'b46', 'b41', 'b47']

Section: Methodology
In this section, we present our method VIST, which processes long in-context text by a lightweight visual encoder, effectively and efficiently extending the context length of LLMs.
this section cite: []

Section: Overall Pipeline
Our VIST, a slow-fast compression framework, is designed to efficiently process long texts by mimicking human reading. The fast visual path skims distant, low-salience long context via a lightweight vision encoder, while the slow cognitive path performs fine-grained reasoning on important content by LLM. As illustrated in Figure 2, the input long text (i.e., T text tokens) is split into two parts: the first T e text tokens processed in a visual view and the remaining T d raw text tokens given to LLM, where T = T e + T d . Specifically, the T e text tokens are evenly rendered into M images and fed into a frozen vision encoder. Then VIST employs a learnable Perceiver Resampler to compress text-rendered image features into a fixed count of tokens. Such compressed visual tokens are integrated into the LLM via cross-attention for the next-token prediction. The Perceiver Resampler is jointly trained with the LLM during tuning the cross-attentions in an end-to-end manner.
To empower the model with the ability to comprehend dense text in images, we devise Probabilityinformed Visual Enhancement (PVE, §3.4). PVE maximizes agreement between visual features obtained from the Perceiver Resampler and text token embeddings extracted from LLM tokenizer. This alignment bridges the global semantic gap between visual tokens and raw text tokens. Furthermore, to address token redundancy, VIST incorporates a frequency-based masking mechanism Tokenization && Embedding
this section cite: []

Section: Perceiver Resampler

this section cite: []

Section: Large Language Model
Vision Encoder Feedforward Cross-attention
this section cite: []

Section: Self-attention

this section cite: []

Section: Frequency-based Masking Strategy
The "primary identification" with the father in individual pre
this section cite: []

Section: Render Text as Image 3.2

this section cite: []

Section: Pooling
The …
this section cite: []

Section: Pooling

this section cite: []

Section: CLS Token
Frozen Parameter Trained Parameter
this section cite: []

Section: Masked Token

this section cite: []

Section: Probability-Informed Visual Enhancement

this section cite: []

Section: M Images
The " primary identification" with the father in individual pre
this section cite: []

Section: 3.4
Next token prediction (Eq. 1)
Rare Frequent
this section cite: []

Section: … …
The " primary identification" with the father in individual pre history would be the means, the within PVE that selectively masks high-frequency, low-information text tokens, thereby improving the information density of the text embeddings. These refined embeddings serve as enriched supervision signals, encouraging visual features to be more compact and semantically meaningful.
this section cite: []

Section: Vision-centric Implementation
VIST transforms raw textual data into M uniformly distributed RGB images X = {x m ∈ R H×W ×C } M m=1 , where M can be dynamically adjusted based on the length of the input text. Concretely, each image is configured with height H = 14, width W = 3, 584, and C = 3 RGB channels, which corresponds to a square color image with a resolution of 224 × 224. Text is rendered using a 10px font size and Google Noto Sans typeface. If text incompletely fill the image, white empty patches are masked to exclude them from attention score computation and loss calculation. Compared to text tokenizer-based methods, this rendering method does not lead to slower training speeds [44].
this section cite: ['b43']

Section: Token Reduction
The M text-rendered images are first processed by frozen vision encoder, specifically the ViT-L/14 [11] from OpenCLIP. The extracted features F ∈ R M ×L×D are then fed into a trainable Perceiver Resampler [49], producing a fixed set of N +1 visual tokens per image (including a CLS token), denoted as F ′ ∈ R M ×(N +1)×D , where N = 64 and D is the feature dimension. During training, raw text data (T e = 4096 text tokens) is rendered onto M = 28 images, resulting in 64 × 28 = 1792 visual tokens, passed to the cross-attention layer in LLM. This compression reduces the computational complexity of the cross-attention layer within the LLM. Moreover, the number of images M and tokens N can be dynamically adjusted during both training and inference, allowing VIST to flexibly control the compression ratio. VIST using a lightweight vision encoder, offers a more efficient approach than processing all text tokens directly within the LLM.
this section cite: ['b10', 'b48']

Section: Probability-informed Visual Enhancement
In VIST, the frozen vision encoder is pre-trained primarily on general visual data (such as natural images) without exposure to rendered text images. Hence its ability to interpret dense textual information within images is constrained. To alleviate this problem, we develop a novel training objective, named Probability-informed Visual Enhancement (PVE). PVE enhances the understanding capabilities of Perceiver Resampler for rendered text images, enabling them to serve as robust substitutes for traditional text tokenizers. Text-anchored Semantic Consistency. PVE encourages the Perceiver Resampler to learn a shared embedding space, aligning visual text features F ′ with text token embeddings from text tokenizer. Concretely, PVE is formulated as a contrastive loss:
L ij PVE = -log exp(⟨ F ′ i , F t j ⟩/τ ) B k=1 exp(⟨ F ′ i , F t k ⟩/τ ) , (1
)
where B is batch size and F ′ i is obtained by applying average pooling to the CLS tokens from
F ′ i . F t j
is the averaged text token embedding after frequency-based masking and pooling. τ is the temperate parameter. Importantly, F ′ i and F t i are different representations derived from the same text. Frequency-based Masking. PVE employs text token embeddings as supervision signals to guide the Resampler in extracting textual information from text images. However, long-form text is inherently redundant, where structural components and function words may dominate the token distribution. Such redundancy introduces noise that impedes Resampler from capturing key semantic content.
Our solution draws inspiration from Shannon information theory [50], which provides a formal way to quantify the information content of an event or a message. The formula is given by:
I(y) = -log 2 P (y),(2)
where I(y) is the information content of event or messages y and P (y) is the probability of y. It highlights the inverse relationship between the probability of an event and the information it carries. When applied to tokens in a corpus: Rare tokens (low-frequency) are treated as high-information tokens because they often carry domain-specific or contextually important information. Frequent tokens (high-frequency) have lower information content because they may serve more structural or grammatical purposes, contributing less to the unique meaning of the text. Figure 3 shows that masking 50% of the most frequent tokens based on corpus-level (i.e., training set) frequency distribution still preserves most high-information-gain (IG) tokens, ensuring minimal loss of critical information while reducing redundancy. This aligns with the selective reading strategies [10,51] observed in skilled readers. Based on this principle, we devise frequency-based masking strategy that uses token frequency as a proxy for semantic importance. This strategy masks frequent tokens but low-information tokens to improve the information density of text token embeddings. The importance score for each token is calculated as follows:
s w = log |S| 1 + count(w) ,(3)
where |S| denotes the total number of samples, count(w) is the count of the token w (subword), and s w is the importance score of token w. Token frequency statistics can be easily computed online with negligible overhead or precomputed. Based on the importance score for each token, we apply a 50% masking rate, where tokens are randomly masked with tokens of lower importance score being more likely to be masked. This ensures the Resampler prioritizes key content-bearing tokens, learning richer semantic representations and improving its ability to interpret dense text in rendered images.
this section cite: ['b49', 'b9', 'b50']

Section: Experiment

this section cite: []

Section: Experimental Setup
Pretraining. We validate VIST with TinyLlama [52]. The frozen vision encoder in our model is ViT-L/14 [11]. To reduce computational overhead, our model employs float16 precision and DeepSpeed Zero-2 with CPU off-loading [53]. Refer to Appendix A for details. Competitors. i) Long-context models: Replug [54] and Stream [55] with TinyLlama [52]. ii) Text-encoder-based compression method: To compare the effectiveness of leveraging text tokens v.s.visual tokens for processing long contexts in LLM, we implement CEPE * by applying CEPE [21] to TinyLlama [52], replacing the vision encoder in VIST with a lightweight text encoder. All other architectural and training settings are kept identical. iii) Vision-centric compression methods: ToMe [56] merges, and FastV [57] prunes visual features from frozen vision encoders. For fairness, we match their compression rates to ours. Directly using these visual features in LLMs leads to high perplexity (>1k) due to the mismatch between visual and text tokens. Thus, we retain the Resampler and PVE in our VIST, denoting ToMe † and FastV † . See Appendix A.5 for more details.
Pretraining Dataset. Our pertaining dataset is an official sample of the RedPajama dataset [58], including 1B tokens from seven domains: ArXiv, Book, C4, Commoncrawl, GitHub, StackExchange, and Wikipedia. The training set of the corpus is preprocessed into 4608 text token sequences, where the first 4096 text token sequences are fed into the vision encoder (or text encoder for CEPE * ) and the remaining 512 text tokens are provided to LLM.
this section cite: ['b51', 'b10', 'b52', 'b53', 'b54', 'b51', 'b20', 'b51', 'b55', 'b56', 'b57']

Section: Downstream Evaluation.
We primarily evaluate tasks requiring long context processing, revealing vision tokens effectively handle extended context, outperforming previous text-encoder-based model.
Comparisons across more methods and datasets are provided in the Appendix A.6.
this section cite: []

Section: Long-context Language Modeling
To assess the long-context language modeling (LCM) ability, we evaluate on ArXiv and Book from RedPajama [58] test split, alongside long-context datasets: PG19 [59], Proof [60], and Code [61]. The evaluation metric is perplexity (PPL) over the last 256 tokens of each input. Early tokens (typically farther from the current prediction point) are handled by the vision encoder, while the more recent tokens are passed to the LLM, under the assumption that proximity correlates with relevance.
Impact of Increased Text Length. Table 1 summarizes the results across different input lengths.
Long-context language modeling can benefit from previous long contextual information. However, TinyLlama [52] supports only fixed-size inputs of 2048 tokens. Beyond this length, its performance drops sharply, with perplexity exceeding 10 3 . In contrast, VIST demonstrates a consistent decrease in perplexity as the input text length increases. Vision-centric token compression models ToMe † and FastV † focus on natural images, where redundancy arises from local visual similarity. However, they are ill-suited for text redundancy and struggle to preserve key semantic content in text image, yielding higher perplexity than our VIST. Moreover, VIST obtains the lowest PPL (14.973) on Book datasets, when T e and T d are 2048. These results prove that VIST effectively enhances the capability of modeling long-form language.
Comparison on Inference Cost. In Table 1, VIST renders text into multiple images of size 224 × 224. 1024 text tokens need 7 images and VIST requires 56% fewer visual tokens than text tokens for the same input (i.e., compression ratio ∆ is 2.3, from 1024 text tokens to 448 = 7 × 64 visual tokens). We report the throughput of each model relative to TinyLlama. VIST achieves comparable performance with text-encoder-based compression model CEPE * , with 16% fewer FLOPs, 50% less memory usage, and higher throughput when processing 16k tokens.
Table 2: In-context learning accuracy averaged across 3 seeds (42, 43 and 44). Green highlights the gain from the additional demos. ne is the number of demos for encoder and nd for decoder (LLM).
Method n e n d SST2 MR AGN SST5 NLUS NLUI TREC TREF DBP BANK CLIN Avg. TinyLlama [52] -2 76.0 67.7 63.4 27.6 5.2 4.4 28.8 9.6 38.0 23.0 22.4 33.3 TinyLlama [52] -20 87.6 71.7 75.0 30.1 46.1 32.6 72.0 38.5 80.4 42.9 53.7 57.3(24.0↑) CEPE * [21] 18 2 76.9 82.3 66.9 29.1 9.6 30 39.2 12.7 71.1 27.2 39.8 44.1(10.8↑) VIST 18 2 77.7 79.2 61.5 42.7 15.6 40.6 36.5 14.6 71.9 25.0 43.8 46.3(13.0↑) TinyLlama [52] -50 88.6 64.8 21.4 42.5 34.2 30.4 81.1 44.7 3.4 49.7 39.7 45.5(12.2↑) CEPE * [21] 48 2 82.9 79.4 63.9 42.3 28.1 31.1 32.6 14.7 71.5 29.0 39.1 46.8(13.5↑) VIST 48 2 78.9 85.2 71.9 44.4 27.2 43.1 38.3 18.4 73.1 25.4 48.1 50.4(17.1↑)
this section cite: ['b57', 'b58', 'b59', 'b60', 'b51']

Section: In-context Learning
We evaluate VIST on In-Context Learning (ICL) tasks across 11 widely-used text-classification datasets: SST2 [62], MR [63], AGN [64], SST5 [62], TREC, TREF [65], DBP [64], NLUS, NLUI [66], BANK [67], and CLIN [68]. Following [21], we randomly sample 250 text examples per dataset. The ICL results in Table 2 are reported as the average accuracy over three random seeds. For VIST and CEPE * , we provide two demonstrations directly to the decoder, while the rest are processed by the encoder. More details in Appendix B.2.
Results. Table 2 examines the influence of increasing the number of demonstrations, where n e is the number of demos for encoder and n d for LLM. VIST shows a 13% accuracy improvement (from 33.3% to 46.3%) as more demonstrations (n e is 18) are provided to the visual encoder, showcasing the capacity of LLM to comprehend text within visual signals when integrated with VIST. Furthermore, VIST outperforms CEPE * in average accuracy across all 11 datasets, which indicates visual-based text understanding can effectively match or even surpass text encoder performance. Though VIST and CEPE * (n e = 18, n d = 2) underperform TinyLlama (n d = 20), they achieve lower computational cost by processing most demonstrations (18) with lightweight encoder. The performance gap on NLUS, TREC and TREF may stem from high category diversity, where the weak relevance between queries and demonstrations makes the lightweight encoding less effective than using the full LLM for all demos. Notably, the performance of TinyLlama declines with 50 demonstrations due to context window limit, while VIST remains efficient and stable.
this section cite: ['b61', 'b62', 'b63', 'b61', 'b64', 'b63', 'b65', 'b66', 'b67', 'b20']

Section: Open-domain Question Answering
Open-domain Question Answering (QA) is a challenging task that requires model to generate accurate answers based on retrieved relevant information. Experiments are conducted on three open-domain QA datasets, including TriviaQA [69], NQ [70], and PopQA [71]. We use Contriever [72] to retrieve relevant k passages from Wikipedia, as in CEPE [21]. We prioritize passing the most relevant passages to the decoder to enhance performance. In Table 3, we report the exact match (EM) scores. Results. TinyLlama is limited by a maximum context window of 2048 tokens, restricting it to processing no more than 10 passages at a time. Beyond this limit, performance drops sharply, with EM score falling below 1. For passages k d = 10, the input already approaches or even exceeds the 2048-token limit of TinyLlama, so we truncate the input to avoid performance degradation. When processing 5 extra passages (i.e., k e = 5, k d = 10), VIST results in an EM score improvement of 3.75 compared to TinyLlama on TriviaQA [69] dataset. Moreover, it even surpasses text encoder-based approach CEPE * under the same input conditions, e.g., delivering an EM score 9.11 points higher on the TriviaQA dataset when k e = 20, k d = 10. This enhancement may be attributed to PVE in VIST which leverages enriched text embeddings to guide the Resampler in capturing key semantics. By emphasizing critical details and filtering out noise from lengthy inputs, our VIST prioritizes relevant information-a crucial factor for success in opendomain QA tasks. In contrast, CEPE * degrades when more passages are provided to the encoder,     as additional passages introduce more noise and redundancy, making it harder to extract relevant answers. We also investigate the effectiveness of using only the fast path of VIST (i.e., vision encoder) on the Open-domain QA task, which requires the model to generate accurate answers based on given relevant passages. Specifically, we feed only the top-10 relevant passages to the visual encoder (i.e., k e = 10), without providing any passages to the LLM directly (i.e., setting k d = 0). Surprisingly, this configuration yields performance on par with TinyLLaMA, despite the latter processing all 10 passages with a much heavier LLM. This demonstrates that the fast path of our method can distill and preserve the critical information from long contexts, providing a compact yet effective representation.
this section cite: ['b68', 'b69', 'b70', 'b71', 'b20', 'b68']

Section: Ablation Study
We explore the effects of ❶ the masking strategy employed in PVE, ❷ the length of text provided to the encoder during training, ❸ the number of compressed tokens in each image (i.e., N in §3.3), and ❹ extension to other LLM. In open-domain QA tasks, the model is fed 10 passages for LLM and 5 for encoder. ICL tasks use a fixed setup of 18 demonstrations for encoder and 2 for LLM. (More details in Appendix F.)
this section cite: []

Section: Number of Tokens in Each Image.
The Resampler transforms image features into a fixed number of visual tokens. Table 4 analyzes the impact of visual token count for each image. Increasing the number of visual tokens reduces compression ratio, but as shown, a lower compression ratio does not always yield better results. With 64 visual tokens, the model performs best on 4 out of 5 datasets, whereas 128 tokens only perform best on MR dataset. This discrepancy could be attributed to the trade-off between the amount of information preserved and the noise introduced during compression. Fewer tokens risk losing critical details, while more tokens may retain irrelevant information, which can hinder generalization. These findings emphasize that achieving a balance between compression and information retention is crucial for optimal performance across different datasets.
this section cite: []

Section: Masking Strategy in PVE ( §3.4).
Long texts often contain significant redundancy. To address this, we integrate a Frequency-based Masking (FM) strategy into PVE, improving the information density of text token embeddings. Table 5 compares the performance of VIST with FM, w/o FM, and with random masking. Excluding FM causes a notable decline in ICL and open-domain QA performance. This highlights the critical role of information-dense text token embeddings in guiding the visual encoder to capture more semantically meaningful and discriminative features. The "primary identification" with the "father in individual prehistory" would be the means, the link that might enable one to become reconciled with the loss of the Thing. Primary identification initiates a compensation for the Thing and at the same time secures the subject to another dimension, that of imaginary adherence, reminding one of the bond of faith, which is just what disintegrates in the depressed person. In-context Text Length. Table 6 investigates the impact of in-context text length (measured in text tokens) provided to the visual encoder during training. Our model, trained with a longer encoder input length, generally yields higher EM scores on open-domain QA tasks. This may be because exposure to more lengthy texts during training helps our model better extract key information from extensive contexts, which is crucial for open-domain QA. Table 6 shows that longer training text inputs often boost ICL task accuracy. For instance, the best result on SST5 (42.7) was achieved with an encoder input length of 4096 text tokens, while the highest accuracy on MR (90.5) was obtained with the longest input length of 6144 tokens. Interestingly, we observed that training with an input length of 1024 tokens performed comparably to 2048 tokens, possibly because the total demo length for the encoder was close to 1024 tokens.
Extension to Other LLM. Mixture-of-expert models effectively scale model capacity while saving resources. To prove the generality of VIST, we also apply VIST to Mistral 7B [73]. In LCM task, VIST ‡ and CEPE ‡ process 4096 tokens, compared to the 2048 tokens processed by Mistral. For ICL, VIST ‡ and CEPE ‡ use 20 demonstrations, while Mistral uses only 2. As shown in Table 7, VIST ‡ demonstrates superior performance over CEPE ‡ , by effectively leveraging additional context.
To further validate the scalability of our method, results on larger-scale LLMs are included in Appendix F.
this section cite: ['b72']

Section: Discussion
Exploring Token Frequency as a Proxy for Semantic Importance. To assess the impact of rare versus frequent text tokens on global semantics across in-domain (Book [58]) and out-of-domain datasets (PG19 [59], Proof [60], and Code [61]), we first calculate importance score for each token using Eq. 3 based on training-set token frequency statistics. Two masking operations are then applied to the text token embeddings: ❶ Masking tokens with high importance scores (red line in Figure 4). ❷ Masking tokens with low importance scores (blue line in Figure 4). The distance between masked text embeddings and visual features is computed.
At low masking ratios (0.0 to 0.4), masking rare tokens causes a sharp increase in distance, while masking frequent tokens has minimal impact and even reduces distance. This suggests that rare tokens carry more critical semantic information, and their removal disrupts the alignment between text and visual tokens. In contrast, frequent tokens may contain more redundant or less informative content, so masking them has little impact or even improves the alignment by reducing noise. These findings support the hypothesis that token frequency is a reasonable indicator of semantic importance, with rare tokens playing a more pivotal role in preserving semantic integrity.
The Effect of Frequency-based Masking on Text-Visual Semantic Gap. VIST employs Frequencybased Masking (FM) within the Probability-informed Visual Enhancement to improve the semantic richness of text token embeddings. Figure 5 presents the impact of FM on the semantic alignment between text token embeddings and visual features extracted by the Perceiver Resampler. We experimented with random masking ratios (0.0 to 0.9) on text token embeddings and calculated the sum of cosine distances across all test samples in the Arxiv and Book datasets [58]. Across all ratios, VIST with FM consistently exhibits smaller semantic distances than VIST without FM, highlighting FM effectively enhances semantic coherence.
this section cite: ['b57', 'b58', 'b59', 'b60', 'b57']

Section: References
Ref_id:b0 Title: Language models are few-shot learners Year: (2020)
Ref_id:b1 Title:  Year: (2023)
Ref_id:b2 Title: Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps Year: (2020)
Ref_id:b3 Title: musique: Multihop questions via single-hop question composition Year: (2022)
Ref_id:b4 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b5 Title: Pangu ultra: Pushing the limits of dense large language models on ascend npus Year: (2025)
Ref_id:b6 Title: 52b to 1t: Lessons learned via tele-flm series Year: (2024)
Ref_id:b7 Title: The effect of word frequency, word predictability, and font difficulty on the eye movements of young and older readers Year: (2006)
Ref_id:b8 Title: One page of text: Eye movements during regular and thorough reading, skimming, and spell checking Year: (2018)
Ref_id:b9 Title: Why do skimmers perform better with grammar-preserving text saliency modulation (gp-tsm)? evidence from an eye tracking study Year: (2024)
Ref_id:b10 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b11 Title: Parrot captions teach clip to spot text Year: (2025)
Ref_id:b12 Title: Compressing context to enhance inference efficiency of large language models Year: (2023)
Ref_id:b13 Title: Llmlingua: Compressing prompts for accelerated inference of large language models Year: (2023)
Ref_id:b14 Title: Selective perception: Learning concise state descriptions for language model actors Year: (2024)
Ref_id:b15 Title: Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression Year: (2023)
Ref_id:b16 Title: Llmlingua-2: Data distillation for efficient and faithful taskagnostic prompt compression Year: (2024)
Ref_id:b17 Title: A new algorithm for data compression Year: (1994)
Ref_id:b18 Title: Auto-regressive language modeling in pixel space Year: (2024)
Ref_id:b19 Title: Language modelling with pixels Year: (2022)
Ref_id:b20 Title: Long-context language modeling with parallel context encoding Year: (2024)
Ref_id:b21 Title: Prompt compression and contrastive conditioning for controllability and toxicity reduction in language models Year: (2022)
Ref_id:b22 Title: Soaring from 4k to 400k: Extending llm's context with activation beacon Year: (2024)
Ref_id:b23 Title: Learning to compress prompts with gist tokens Year: (2024)
Ref_id:b24 Title: Adapting language models to compress contexts Year: (2023)
Ref_id:b25 Title: In-context autoencoder for context compression in a large language model Year: (2024)
Ref_id:b26 Title: Snapkv: Llm knows what you are looking for before generation Year: (2024)
Ref_id:b27 Title: Random-access infinite context length for transformers Year: (2023)
Ref_id:b28 Title: Focused transformer: Contrastive training for context scaling Year: (2024)
Ref_id:b29 Title: Retrieval meets long context large language models Year: ()
Ref_id:b30 Title: Retrieve anything to augment large language models Year: (2023)
Ref_id:b31 Title: Memorizing transformers Year: ()
Ref_id:b32 Title: Augmenting language models with long-term memory Year: (2024)
Ref_id:b33 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b34 Title: SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing Year: (2018)
Ref_id:b35 Title: Neural machine translation of rare words with subword units Year: (2016)
Ref_id:b36 Title: Robust open-vocabulary translation from visual text representations Year: (2021)
Ref_id:b37 Title: Improving language understanding from screenshots Year: (2024)
Ref_id:b38 Title: Dual modalities of text: Visual and textual generative pre-training Year: (2024)
Ref_id:b39 Title: Text rendering strategies for pixel language models Year: (2023)
Ref_id:b40 Title: Pixel sentence representation learning Year: (2024)
Ref_id:b41 Title: Ocr-free document understanding transformer Year: (2022)
Ref_id:b42 Title: Pix2struct: Screenshot parsing as pretraining for visual language understanding Year: (2023)
Ref_id:b43 Title: Clippo: Image-and-language understanding from pixels only Year: (2023)
Ref_id:b44 Title: Leveraging visual tokens for extended text contexts in multi-modal learning Year: (2024)
Ref_id:b45 Title: Text as images: Can multimodal large language models follow printed instructions in pixels? arXiv preprint Year: (2023)
Ref_id:b46 Title: Mpmqa: multimodal question answering on product manuals Year: (2023)
Ref_id:b47 Title: mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding Year: (2024)
Ref_id:b48 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b49 Title: A mathematical theory of communication Year: (1948)
Ref_id:b50 Title: Word skipping: Implications for theories of eye movement control in reading Year: (1998)
Ref_id:b51 Title: Tinyllama: An open-source small language model Year: (2024)
Ref_id:b52 Title: Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters Year: (2020)
Ref_id:b53 Title: Replug: Retrieval-augmented black-box language models Year: (2024)
Ref_id:b54 Title: Efficient streaming language models with attention sinks Year: ()
Ref_id:b55 Title: Token merging: Your vit but faster Year: ()
Ref_id:b56 Title: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models Year: (2024)
Ref_id:b57 Title: Redpajama: an open dataset for training large language models Year: (2024)
Ref_id:b58 Title: Compressive transformers for long-range sequence modelling Year: (2019)
Ref_id:b59 Title: Proofpile: A pre-training dataset of mathematical texts Year: (2023)
Ref_id:b60 Title: A dataset of python files from github Year: (2023)
Ref_id:b61 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013)
Ref_id:b62 Title: Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales Year: (2005)
Ref_id:b63 Title: Character-level convolutional networks for text classification Year: (2015)
Ref_id:b64 Title: Building a question answering test collection Year: (2000)
Ref_id:b65 Title: Benchmarking natural language understanding services for building conversational agents Year: (2021)
Ref_id:b66 Title: Efficient intent detection with dual sentence encoders Year: (2020)
Ref_id:b67 Title: An evaluation dataset for intent classification and out-of-scope prediction Year: (2019)
Ref_id:b68 Title: Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017)
Ref_id:b69 Title: Natural questions: a benchmark for question answering research Year: (2019)
Ref_id:b70 Title: When not to trust language models: Investigating effectiveness of parametric and non-parametric memories Year: (2023)
Ref_id:b71 Title: Unsupervised dense information retrieval with contrastive learning Year: (2022)
Ref_id:b72 Title: Mistral 7b Year: (2023)
Ref_id:b73 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b74 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b75 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b76 Title: Kevin Leyton-Brown, and Yoav Shoham. Parallel context windows for large language models Year: (2022)
Ref_id:b77 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b78 Title: Recurrent memory transformer Year: (2022)
Ref_id:b79 Title: Surgical video workflow analysis via visual-language learning Year: (2025)
Ref_id:b80 Title: Locality-aware cross-modal correspondence learning for dense audio-visual events localization Year: (2024)
Ref_id:b81 Title: Mvp-shot: Multivelocity progressive-alignment framework for few-shot action recognition Year: (2025)
Ref_id:b82 Title: Learning clustering-based prototypes for compositional zero-shot learning Year: (2025)
Ref_id:b83 Title: Omnigaze: Reward-inspired generalizable gaze estimation in the wild Year: (2025)
