Title: PRUNEVID: VISUAL TOKEN PRUNING FOR EFFICIENT VIDEO LARGE LANGUAGE MODELS
Abstract: In this paper, we introduce PruneVid, a visual token pruning method designed to enhance the efficiency of multi-modal video understanding. Large Language Models (LLMs) have shown promising performance in video tasks due to their extended capabilities in comprehending visual modalities. However, the substantial redundancy in video data presents significant computational challenges for LLMs. To address this issue, we introduce a training-free method that 1) minimizes video redundancy by merging spatial-temporal tokens, and 2) leverages LLMs' reasoning capabilities to selectively prune visual features relevant to question tokens, enhancing model efficiency. We validate our method across multiple video benchmarks, which demonstrate that PruneVid can prune over 80% tokens while maintaining competitive performance combined with different model networks. This highlights its superior effectiveness and efficiency compared to existing pruning methods.

Section: INTRODUCTION
Large Language Models (LLMs) (Achiam et al., 2023;Yang et al., 2024;Touvron et al., 2023) have significantly advanced multi-modal understanding owing to their exceptional reasoning capabilities and proficiency in following instructions. Within the realm of video understanding, recent studies (Li et al., 2023c;Lin et al., 2023;Zhang et al., 2023;Li et al., 2024b;2023a;Xu et al., 2024a;Wang et al., 2024a) have capitalized on the use of pre-trained LLMs as foundational models to address video question-answering tasks. However, the redundancy inherent in video content can lead to significant computational expenses for LLMs due to the quadratic complexity of attention mechanisms. Consequently, effectively reducing the number of video tokens while preserving model performance emerges as an intriguing area of research.
Previous approaches attempt to address this challenge in various ways. LLaMA-VID (Li et al., 2023c) proposes compressing each frame into two distinct tokens: context and content tokens. However, this method necessitates extensive pretraining and fine-tuning phases, which limits its broader applicability with readily available video LLMs. Alternatively, LLaVA-PruMerge (Shang et al., 2024) leverages the correlation between the [CLS] token and patch tokens within CLIP (Radford et al., 2021) to identify important visual tokens and merges other less important ones. Yet, this approach does not consider the relevance of the selected tokens to the questions being asked, potentially selecting tokens that are unrelated to the task at hand. In a related vein, methods like Look-M (Wan et al., 2024) and Elastic Cache (Liu et al., 2024d) employ Key-Value (KV) cache eviction strategies (Zhang et al., 2024;Liu et al., 2024c) to merge the KV cache for multi-modal inputs. These strategies prioritize retaining text tokens or treating visual and textual tokens equally without explicitly identifying the informative visual tokens. Moreover, eviction-based methods require encoding all visual tokens during the prefilling stage, which becomes inefficient when handling long visual sequences. Recently, FastV (Chen et al., 2025) has leveraged attention patterns in LLMs to prune visual tokens. However, it is not specifically tailored for video understanding and does not adequately address the reduction of video inputs. What happened after the person took the phone? Vision Encoder Word Embedding LLM … Attention Layer … Question … Temporal Merging of Static Tokens Spatial Merging Merged Static Tokens Dynamic Tokens … … Attention-based Token Selection … Input Regions Static Regions Dynamic Regions … … … … … Figure 1: (a) PruneVid first identifies the static regions in the video that exhibit minimal variation, thereby compressing the redundancy of static tokens along the temporal dimension. It then further reduces spatial redundancy through compression in the spatial dimension. Subsequently, within the LLM, PruneVid utilizes question-to-visual attention scores to guide the selection of relevant visual tokens. (b) Static regions refer to areas with minimal change, while dynamic regions exhibit motion. Therefore, static regions can be compressed together along the temporal dimension. (c) Visualization of how attention evolves from shallow to deep layers (32 layers in total). The question tokens attend to semantically related visual regions (e.g., the hands and window) throughout different layers.
Building on the analysis presented above, we identify three essential criteria that an optimal pruning method for multi-modal video understanding ought to meet: (1) It should ideally be training-free, facilitating smooth integration with readily available models while reducing the need for extensive retraining or fine-tuning. (2) Inherent video redundancy needs to be reduced to save computations on tokens with similar representations along both spatial and temporal dimensions. (3) It is crucial to retain visual tokens specifically relevant to the given questions. This ensures the model maintains high performance while efficient and mitigates the risk of hallucinations when LLMs lack pertinent information (Huang et al., 2024).
To achieve our objectives, we present PruneVid, a training-free approach for pruning video tokens to achieve efficient video understanding. As shown in Fig. 1 (a), our method initially identifies static regions with minimal variation due to motion or camera movements, which could be interpreted as background (illustrated in Fig. 1 (b)). We merge these static tokens along the temporal dimension to reduce the computational burden from redundant temporal data. Next, we employ a clustering technique (Du et al., 2016) to merge similar spatial tokens for both static and dynamic regions. In the subsequent step within the LLM, we use attention scores between the question and video tokens in an intermediate layer to discern and preserve the discriminative visual tokens essential for answering the question, while pruning irrelevant ones. As depicted in Fig. 1 (c), attention visualizations consistently highlight crucial features, such as hand movements and related objects (e.g., a window), which are directly relevant to the question. This indicates that important visual regions can be effectively pinpointed using attention layers, benefiting from the LLM's reasoning and instruction-following prowess. Additionally, for the KV caches from previous layers, we retain the essential visual tokens and eliminate others, thus reducing computational demands during the decoding phase.
We integrate PruneVid with three video LLMs: PLLaVA (Xu et al., 2024a), ST-LLM (Liu et al., 2024b), and LLaVA-OneVision (Li et al., 2024a), and evaluate their performance on several video benchmarks, including MVBench (Li et al., 2024b), Video-MME (Fu et al., 2024), Egoschema (Mangalam et al., 2023), and VideoChatGPT-Bench (Maaz et al., 2023). Our extensive experiments demonstrate that PruneVid can prune over 80% of visual tokens with only minimal performance degradation in certain cases. Notably, our method can occasionally enhance model performance. Furthermore, it achieves competitive results compared to the baseline model while boosting inference speed-up to 1.55 times faster-reducing FLOPs by 74% to 80%, and minimizing GPU memory usage. The main contributions of this paper are as follows: (1) We introduce PruneVid, a framework that efficiently prunes video tokens for video understanding without the need for retraining or fine-tuning, which can be seamlessly integrated with off-the-shelf video LLMs. (2) We introduce a token pruning method that minimizes video redundancy by merging static tokens over time and clustering spatially similar ones. Furthermore, our approach leverages attention scores between the question and video tokens within the LLM to retain only the visual tokens pertinent to answering the questions. (3) Extensive experiments are conducted across multiple benchmarks to demonstrate that PruneVid can consistently achieve superior efficiency and effectiveness with different video LLMs compared to existing approaches. 2 RELATED WORK 2.1 VIDEO LARGE LANGUAGE MODEL Recent advancements in Video LLMs focus on enabling LLMs to comprehend video content. These approaches are broadly categorized into training-free methods and training-required methods.
For training-free approaches (Wu, 2024;Kim et al., 2024;Xu et al., 2024b), they directly adapt the image LLMs for video tasks. FreeVA (Wu, 2024) compacts frame features for LLM processing, and IG-VLM (Kim et al., 2024) merges frames into a single grid, simplifying video-to-image conversion. SF-LLaVA (Xu et al., 2024b) uses a SlowFast (Feichtenhofer et al., 2019) network design, balancing detailed spatial analysis with broad temporal scope efficiently within existing LLM token limits. These methods are ingeniously simple but are limited to handling only brief video clips due to their reliance on the inherent abilities of LLMs to understand temporal sequences.
Conversely, training-required Video LLMs improve comprehension by using extensive video datasets. Models like Video-ChatGPT (Maaz et al., 2023), Video-LLaVA (Lin et al., 2023), and PLLaVA (Xu et al., 2024a) extend Image LLMs with video-specific tuning, greatly enhancing complex video understanding. Other approaches, such as VideoChat2 (Li et al., 2024b), VILA (Lin et al., 2024), Tarsier Wang et al. (2024a Unlike the methods mentioned above, PruneVid aims to enhance the efficiency of existing video LLMs without additional training, which can be applied to both training-free and training-required methods.
this section cite: ['b0', 'b35', 'b26', 'b15', 'b36', 'b25', 'b23', 'b28', 'b37', 'b2', 'b6', 'b3', 'b5', 'b22', 'b21', 'b8', 'b8', 'b4', 'b21', 'b15', 'b16']

Section: VISUAL TOKEN PRUNING
Due to the quadratic computational complexity inherent in attention mechanisms, optimizing efficiency through token pruning becomes essential. This optimization highlights a crucial distinction between methods designed for vision-centric and multi-modal tasks.
DynamicViT (Rao et al., 2021) employs a prediction module to selectively prune less important tokens, thereby streamlining the model's efficiency in processing visual data. Similarly, FastViT (Vasu et al., 2023) reduces architectural complexity and memory demands through a novel token mixing operation, catering specifically to vision-only models. Further contributing to this domain, Token Merging (ToMe) (Bolya et al., 2023) merges tokens via token matching, while SPViT (Kong et al., 2022) introduces a method for softly aggregating redundant tokens into a single 'package token', efficiently preserving essential information while minimizing computational load.
Shifting away from vision-centric methods, LLaVA-Prumerge focuses on pruning visual tokens in multi-modal tasks. By employing an adaptive token reduction strategy, which utilizes CLIP's inherent attention characteristics (Radford et al., 2021), LLaVA-Prumerge improves multi-modal understanding efficiency. FastV Chen et al. (2025) addresses the inefficiency of visual attention patterns by pruning visual tokens with lower attention weights relative to the [EOS] token.
However, previous methods lack a specific focus on multi-modal video understanding. In contrast, this paper presents PruneVid, designed to eliminate redundancy in videos while utilizing LLMs to identify relevant video tokens for question answering. This approach enhances model efficiency without compromising performance.
this section cite: ['b24', 'b27', 'b1', 'b9', 'b23', 'b2']

Section: METHOD
Our method is designed to efficiently process video data by minimizing redundancy in visual tokens before inputting them into the LLM and identifying question-relevant visual tokens within the LLM.
In this section, we introduce the necessary preliminaries and provide a detailed explanation of our method.
this section cite: []

Section: PRELIMINARIES

this section cite: []

Section: PRE-FILLING STAGE
In the pre-filling stage, the model processes the input question tokens and visual tokens to construct the initial representations and prepare the key-value (KV) caches for attention computations. Let X q ∈ R Nq×C denote the question tokens, where N q is the length of the question and C is the channel dimension.
After the token merging, we obtain a compressed set of visual tokens Xv ∈ R N ′ v ×C , where N ′ v is the reduced number of visual tokens. The combined input sequence X ∈ R (Nq+N ′ v )×C is formed by concatenating the question tokens and the merged visual tokens. The model employs a Transformer architecture with L layers. In each layer l, the self-attention mechanism computes queries Q (l) , keys K (l) , and values V (l) through linear projections of the input.
Based on this, the attention scores are computed using scaled dot-product attention with causal masking to prevent attending to future positions:
A (l) = Softmax Q (l) (K (l) ) ⊤ √ C + m ,(1)
where
m ∈ R (Nq+N ′ v )×(Nq+N ′ v )
is a causal mask with entries m ij = -∞ if position i < j (future positions) and 0 otherwise.
The KV caches KV (l) = (K (l) , V (l) ) are stored for each layer l to facilitate efficient computation during decoding.
this section cite: []

Section: DECODING STAGE
In the decoding stage, the model generates the answer tokens autoregressively, utilizing the stored KV caches from the pre-filling stage. At each decoding step, given the previously generated tokens, the model computes the necessary representations to predict the next token. By using the KV caches, the model efficiently attends to the input sequence without recomputing the attention for the entire sequence. This process reduces computational overhead and speeds up the generation of the response.
Video LLM … Question Tokens Transformer Block 𝑴 × Store … Cross Attention Transformer Block (𝑳 -𝑴) × Store Predict Retained Token Pruned Token KV Cache 𝑲 (𝑴-𝟏) 𝑽 (𝑴-𝟏) 𝑲 (… ) 𝑽 (… ) 𝑲 (𝟎) 𝑽 (𝟎) First 𝑴 Layers 𝑲 (𝑳-𝟏) 𝑽 (𝑳-𝟏) 𝑲 (… ) 𝑽 (… ) 𝑲 (𝑴) 𝑽 (𝑴) Last 𝑳 -𝑴 Layers Select … … … … Vision Encoder Temporal Clustering & Feature Similarity Calculation Static Token Merge … Segment 1 Segment 2 … Spatial Merge Static Token Dynamic Token
Figure 2: Illustration of the PruneVid framework. We begin by segmenting the video into different scenes and then decouple the video tokens into static and dynamic ones. Next, we compress the static tokens along the temporal dimension and merge similar tokens in the spatial dimension to further reduce redundancy. Afterward, by using the question-to-video attention weights learned from an intermediate layer, we determine which tokens should be pruned to improve efficiency.
this section cite: []

Section: SPATIAL-TEMPORAL TOKEN MERGING
As depicted in Fig. 2, given an input video consisting of T frames, we first extract visual tokens from each frame using a visual encoder. Let X
v ∈ R Nv×C denote the visual tokens for frame t, where N v is the number of tokens per frame. The complete set of visual tokens for the video is then
X v = {X (1) v , X (2) v , . . . , X (T ) v }.
To identify temporal segments with different scenes in the video, we perform temporal clustering based on the visual content. We compute the average pooled feature f (t) ∈ R C for each frame t by averaging over its tokens. Using the features {f (1) , . . . , f (T ) }, we employ the Density Peaks Clustering with k-Nearest Neighbors (DPC-KNN) (Du et al., 2016) algorithm to group the frames into B temporal segments {T 1 , T 2 , . . . , T B }. Here, B is linearly related to T with the coefficient γ, and each segment T b comprises a subset of consecutive frames with similar content.
Within each temporal segment T b , we analyze the spatial tokens across the frames to identify static tokens-tokens that remain largely unchanged throughout the segment. For each spatial location i (where 1 ≤ i ≤ N v ), we extract the sequence of tokens {X (t) v (i) | t ∈ T b } and compute the feature similarities between every pair of tokens in this sequence. Specifically, for tokens at times t and t ′ within T b , the similarity is measured using cosine similarity s
(t,t ′ ) i : s (t,t ′ ) i = X (t) v (i) ⊤ X (t ′ ) v (i) ∥X (t) v (i)∥∥X (t ′ ) v (i)∥ .
(2)
We then compute the average similarity for each spatial location i within the segment:
si = 2 |T b |(|T b | -1) t,t ′ ∈T b ,t<t ′ s (t,t ′ ) i .
(
Tokens with average similarity above a threshold τ are considered static:
I static = {i | si ≥ τ } .(4)
For these static tokens, we perform temporal averaging within the segment to compress temporal redundancy:
X(b) v (i) = 1 |T b | t∈T b X (t) v (i), ∀i ∈ I static .(5)
The dynamic tokens, corresponding to I dynamic = {1, . . . , N v } \ I static , are retained without temporal averaging.
To further reduce spatial redundancy, we perform spatial clustering within each frame also using the DPC-KNN algorithm. We apply this clustering separately to the static and dynamic tokens. For frame t, we cluster the tokens
{X (t) v (i) | i ∈ I static } and {X (t) v (i) | i ∈ I dynamic } to obtain clusters C (t) 1 , . . . , C (t)
Cs for the static tokens and
D (t) 1 , . . . , D (t)
C d for the dynamic tokens. For both static and dynamic tokens, the cluster number is linearly related to |I static | and |I dynamic | with the coefficient β. We average the tokens within each cluster to represent them with a single token X(t) v (c) for c = 1, . . . , C s in the static clusters, and similarly for the dynamic clusters X(t) v (d) for d = 1, . . . , C d .
After these merging operations, we obtain a reduced set of visual tokens Xv with significantly less redundancy. The merged visual tokens for the entire video are then collected and concatenated to form the final token sequence to be input to the LLM:
Xv = B b=1 X(b) v ,(6)
where X(b) v contains the merged tokens from segment T b .
this section cite: ['b3']

Section: LLM-GUIDED TOKEN SELECTION
We further reduce the visual tokens by leveraging the LLM's internal attentions to select the most relevant tokens with respect to the given question.
Consider the LLM with L layers. During the pre-filling stage, we target the M -th layer, where 1 ≤ M ≤ L, to compute cross-attention weights between the question tokens and the merged visual tokens to obtain a measure of relevance.
At the M -th layer, we calculate the attention scores
A (M ) ∈ R (Nq+N ′ v )×(Nq+N ′ v )
according to Eq. ( 1). To obtain the cross-attention scores between question and visual tokens, we extract a submatrix A (M ) qv ∈ R Nq×N ′ v as follows:
A (M ) qv = A (M ) [N ′ v :, : N ′ v ],(7)
where A (M ) [N ′ v :, : N ′ v ] selects the attention scores from the question tokens to the visual tokens. Next, we compute the maximum attention values a v ∈ R N ′ v for each visual token by applying max pooling over all question tokens. This approach captures the most informative tokens, as not all question tokens are equally important:
a v = Nq max i=1 A (M )
qv (i, :).
We then sort the attention scores in descending order and select the top α% of visual tokens. The set of indices for the selected tokens is represented by S and is defined as:
S = {j ∈ {1, . . . , N ′ v } | Rank (a v (j)) ≤ ⌈αN ′ v ⌉} ,(8)
where Rank (a v (j)) indicates the rank of a v (j) within the sorted attention scores, and ⌈•⌉ denotes the ceiling function.
By focusing on the top α% tokens, we align the model's attention with the most question-relevant visual information. To finalize the pre-filling stage, we combine the selected visual tokens with the question tokens, enabling processing in the remaining (L -M ) layers of the LLM. The KV vectors derived from the retained visual tokens and question tokens, calculated in the last (L -M ) layers, are stored in the KV cache for the decoding process.
this section cite: []

Section: COMPRESSED KEY-VALUE CACHES
To reduce memory and computational costs during the decoding stage, we compress the KV caches stored from the previous M layers by retaining only the selected visual tokens. For each layer l (1 ≤ l ≤ M ), the original key and value matrices for the visual tokens are
K (l) v ∈ R N ′ v ×C and V (l) v ∈ R N ′ v ×C .
We create the compressed key and value matrices K(l) v and Ṽ (l) v by selecting the rows corresponding to the indices in S:
K(l) v = K (l) v [S, :] , Ṽ (l) v = V (l) v [S, :] ,(9)
where
K (l) v [S, :] and V (l)
v [S, :] denote the selection of rows corresponding to the indices in S. Similarly, we adjust the key and value matrices for the entire sequence by combining the question tokens and the selected visual tokens:
K(l) = K(l) v ; K (l) q , Ṽ (l) = Ṽ (l) v ; V (l) q , (10
)
where
K (l) q and V(l)
q are the key and value matrices for the question tokens. By compressing the KV caches, we effectively reduce the sequence length from N q + N ′ v to N q + |S|, where |S| represents the total number of selected visual tokens. This compression significantly reduces the memory requirements and computational complexity during decoding, enabling efficient processing of long video sequences within the LLM framework.
this section cite: []

Section: EXPERIMENT

this section cite: []

Section: DATASETS AND EVALUATION METRICS
Generic Multi-Choice VideoQA. MVbench (Li et al., 2024b) encompasses 20 temporally challenging tasks that cannot be addressed using a single frame. Each task includes 200 test samples, formatted as multiple-choice VideoQA. These samples require the model to choose the correct answer from several provided options.
Long-form Multi-Choice VideoQA. We conduct evaluations of our models using two wellregarded benchmarks for long-form video benchmarks: Video-MME (Fu et al., 2024) and Egoschema (Mangalam et al., 2023). In these evaluations, the models are tasked with selecting the correct answer from multiple-choice options.
Text Generation. VideoChatGPT-Bench, introduced by (Maaz et al., 2023), focuses on five aspects: Correctness of Information (CI), Detail Orientation (DO), Contextual Understanding (CU), Temporal Understanding (TU), and Consistency (CO). For evaluation, we use GPT-3.5-Turbo-0125 for scoring.
this section cite: ['b5', 'b22', 'b21']

Section: BASELINES
To evaluate the effectiveness of our approach, we compare it with three visual token pruning methods: LLaVA-PruMerge (Shang et al., 2024), Look-M (Wan et al., 2024), and FastV (Chen et al., 2025). LLaVA-PruMerge utilizes attention score sparsification within CLIP to identify crucial tokens and employs an outlier detection method to adaptively determine the optimal pruning ratio. Conversely, Look-M extends the concept of text-only KV cache compression to a multi-modal context by implementing strategies for evicting text-prior KV pairs and merging them through a pivotal merging strategy. Additionally, FastV (Chen et al., 2025) uses attention weights to prune visual tokens with low attention scores. To ensure a fair comparison, we use the official implementations of these methods and apply them to video benchmarks.
this section cite: ['b25', 'b28', 'b2', 'b2']

Section: IMPLEMENTATION DETAILS
All experiments are conducted on NVIDIA A100 GPUs with 80GB of memory. We implement PruneVid, LLaVA-PruMerge, Look-M, and FastV on three video LLMs: PLLaVA (Xu et al., 2024a), ST-LLM (Wang et al., 2024a), and LLaVA-OneVision (Li et al., 2024a). LLaVA-PruMerge is incompatible with ST-LLM, so it is excluded from related comparisons. As per the official settings, the input frames are set to 16 for both PLLaVA and ST-LLM, and 32 for LLaVA-OneVision. For the VideoChatGPT-Bench, ST-LLM uses 64 input frames. Besides, The threshold τ is set to 0.8, the temporal segment ratio γ is 0.25, and the cluster ratio β is 0.5. Across all benchmarks, the token selection ratio α is 0.4, and attention calculations use the 10th layer (M ). For FastV, we prune the tokens at the 2nd layer and set the retained ratio to 0.3 to achieve roughly comparable FLOPs to our method. Additionally, the FLOPs in the experiments are measured in relation to the visual tokens in the LLM.
this section cite: []

Section: MAIN RESULT
As illustrated Tab. 1, our method consistently achieves the best performance in almost all cases compared to existing pruning methods (FastV, Prumerge, and Look-M) while retaining fewer tokens and achieving lower FLOPs. For instance, on PLLaVA, our approach retains only 16.2% of tokens yet surpasses the performance of other pruning methods and even the baseline model under MVBench, VideoMME, and Egoschema. A similar pattern is observed for ST-LLM and LLaVA-OneVision, where our method maintains robust performance with retained ratios as low as 15.1% and 17.0%, respectively, across all benchmarks. This underscores the versatility of our approach in balancing accuracy with a substantial reduction in computational overhead.
Moreover, while Prumerge also maintains competitive accuracy on some models, it fails to do so with substantially reduced token budgets. Similarly, although Look-M can achieve decent performance, it requires using the vanilla attention implementation for all layers, resulting in relatively low efficiency. Additionally, we find that FastV struggles to maintain consistent performance across different models. For instance, while it performs well on PLLaVA and LLaVA-OneVision, its accuracy on ST-LLM is unsatisfactory, indicating a lack of robustness across diverse architectures. In contrast, our method effectively adapts by identifying and preserving only the most informative tokens for video understanding, thereby delivering strong overall performance with significantly reduced computational costs.
this section cite: []

Section: DIAGNOSTIC STUDY
In this section, we conduct a diagnostic study based on the PLLaVA model for efficiency comparison, hyper-parameter investigations, and qualitative visualizations.
this section cite: []

Section: Efficiency Analysis.
As shown on Tab. 2, We compare multiple visual token pruning methods with respect to FLOPs, TTFT speed-up, GPU memory usage, and accuracy. Overall, FastV and Prumerge both demonstrate notable FLOPs reduction and moderate speed gains, though their influence on accuracy remains slightly adverse. By contrast, Look-M manages to preserve accuracy comparable to the baseline but incurs a very low TTFT speed up and escalates GPU memory to a high level, suggesting that its underlying layer-wise strategy increases overhead. In contrast, our proposed method achieves the most efficient balance by yielding the fastest TTFT speed up, the largest FLOPs reduction, the smallest memory footprint, and the highest accuracy, clearly underscoring the advantages of our token pruning approach over existing techniques.
Ablation Study on Token Selection Ratio α and the Position of Pruning Layer M . As shown in Fig. 3 (a), we observe that when pruning attention from the 10th layer onward, model accuracy, despite minor fluctuations, gradually saturates. Therefore, selecting M as 10 for token pruning results in lower computational costs compared to pruning at later layers. Additionally, we find that using a larger α does not necessarily yield better results. For instance, when M is 10, an α of 0.4 achieves better accuracy than 0.5, as retaining more tokens may introduce irrelevant information, adversely affecting the outcome.
Ablation Study on threshold τ and temporal segment ratio γ. As depicted in Fig. 3 (b), we observe that performance consistently improves as τ increases from 0.6 to 0.8, while the performance between τ = 0.8 and τ = 0.9 remains similar. Given that setting τ = 0.8 allows the model to merge more tokens along the temporal dimension, resulting in a higher compression ratio, we select τ = 0.8. Concerning the temporal segment ratio γ, the variation in its values does not significantly affect performance. We found that both γ = 0.25 and γ = 1.0 deliver good results across the two datasets. However, since γ = 1.0 treats each input frame as an individual segment, which hinders effective temporal merging of static tokens, we choose to set γ to 0.25.
Ablation Study on threshold τ and spatial merging ratio β. As shown in Fig. 3 (c), the performance is comparable for τ values of 0.8 and 0.9, with τ = 0.8 being slightly superior, which is a similar phenomenon as in Fig. 3 (b). Regarding the parameter β, setting it to 0.5 provides the optimal accuracy. This is because a smaller β leads to overly aggressive merging of spatial tokens, which degrades performance. On the other hand, setting β to 1.0 is unable to merge redundant tokens, which also adversely affects performance.
this section cite: []

Section: Side-by-side Visualizations of Attention Maps and Token Selection.
In Fig. 4, we present a side-by-side comparison demonstrating how our model selects tokens guided by attention scores, highlighting the LLM's strength in focusing on informative regions related to the questions. MVBench VideoMME (b) Ablation study on threshold 𝝉 and temporal segment ratio γ of token merging. MVBench VideoMME (c) Ablation study on threshold 𝝉 and spatial merging ratio β of token merging. Below, we provide detailed analyses on how our model leverages the reasoning capabilities of LLMs to locate relevant visual regions that are not explicitly mentioned in the questions: Fig. 4 (a): The model identifies the key object mentioned in the question, the blanket, and uses the temporal cue (after) from the question to locate additional relevant visual elements, such as objects on the table and the person's hand movements in frames 2, 6, and 7. These details are not directly provided in the question and are inferred through the model's reasoning abilities.
Fig. 4 (b): The model accurately detects the action of the person holding food in frames 7 and 8, and infers that the presence of a bag the person puts down is relevant for answering the question, even though the bag is not mentioned. This demonstrates the model's ability to reason about relevant objects based on contextual cues.
Fig. 4 (c): Despite the absence of any mention of a book in the question, the model correctly identifies critical visual regions related to the book by reasoning over the visual content and context provided. These examples showcase how our model utilizes LLM reasoning to identify and focus on pertinent visual information that is not explicitly described in the questions.
this section cite: []

Section: CONCLUSION
We present PruneVid, a training-free visual token pruning method that enhances efficiency in multimodal video understanding. By reducing video redundancy through merging static tokens over time and clustering similar spatial tokens, PruneVid minimizes the number of tokens processed. It leverages attention mechanisms within LLMs to retain only the visual tokens relevant to questions, ensuring high performance while reducing computational overhead. Experiments across multiple benchmarks demonstrate that PruneVid can prune over 80% of visual tokens while maintaining, or even improving, model performance. By eliminating the need for retraining or fine-tuning, PruneVid offers a practical and efficient solution that integrates seamlessly with existing video LLMs.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Token merging: Your vit but faster Year: (2023)
Ref_id:b2 Title: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models Year: (2025)
Ref_id:b3 Title: Study on density peaks clustering based on k-nearest neighbors and principal component analysis Year: (2016)
Ref_id:b4 Title: Slowfast networks for video recognition Year: (2019)
Ref_id:b5 Title: Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis Year: (2024)
Ref_id:b6 Title: Visual hallucinations of multimodal large language models Year: (2024)
Ref_id:b7 Title: Chat-univi: Unified visual representation empowers large language models with image and video understanding Year: (2024)
Ref_id:b8 Title: An image grid can be worth a video: Zero-shot video question answering using a vlm Year: (2024)
Ref_id:b9 Title: Spvit: Enabling faster vision transformers via latency-aware soft token pruning Year: (2022)
Ref_id:b10 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b11 Title: Videochat: Chat-centric video understanding Year: ()
Ref_id:b12 Title: Unmasked teacher: Towards training-efficient video foundation models Year: (2023)
Ref_id:b13 Title: Mvbench: A comprehensive multi-modal video understanding benchmark Year: (2024)
Ref_id:b14 Title: Llama-vid: An image is worth 2 tokens in large language models Year: (2023)
Ref_id:b15 Title: Video-llava: Learning united visual representation by alignment before projection Year: (2023)
Ref_id:b16 Title: On pre-training for visual language models Year: (2024)
Ref_id:b17 Title:  Year: ()
Ref_id:b18 Title: St-llm: Large language models are effective temporal learners Year: ()
Ref_id:b19 Title: Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. NeurIPS Year: (2024)
Ref_id:b20 Title: Efficient inference of vision instruction-following models with elastic cache Year: (2024)
Ref_id:b21 Title: Video-chatgpt: Towards detailed video understanding via large vision and language models Year: (2023)
Ref_id:b22 Title: Egoschema: A diagnostic benchmark for very long-form video language understanding Year: (2023)
Ref_id:b23 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b24 Title: Dynamicvit: Efficient vision transformers with dynamic token sparsification Year: (2021)
Ref_id:b25 Title: Llava-prumerge: Adaptive token reduction for efficient large multimodal models Year: (2024)
Ref_id:b26 Title: Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b27 Title: Fastvit: A fast hybrid vision transformer using structural reparameterization Year: (2023)
Ref_id:b28 Title: Look-m: Look-once optimization in kv cache for efficient multimodal long-context inference Year: (2024)
Ref_id:b29 Title: Tarsier: Recipes for training and evaluating large video description models Year: ()
Ref_id:b30 Title: Actionclip: A new paradigm for video action recognition Year: (2021)
Ref_id:b31 Title: Internvideo2: Scaling video foundation models for multimodal video understanding Year: ()
Ref_id:b32 Title: Offline mllm as training-free video assistant Year: (2024)
Ref_id:b33 Title: Pllava: Parameterfree llava extension from images to videos for video dense captioning Year: ()
Ref_id:b34 Title: Slowfast-llava: A strong training-free baseline for video large language models Year: ()
Ref_id:b35 Title: Qwen2 technical report Year: (2024)
Ref_id:b36 Title: Video-llama: An instruction-tuned audio-visual language model for video understanding Year: (2023)
Ref_id:b37 Title: H2o: Heavy-hitter oracle for efficient generative inference of large language models Year: (2024)
