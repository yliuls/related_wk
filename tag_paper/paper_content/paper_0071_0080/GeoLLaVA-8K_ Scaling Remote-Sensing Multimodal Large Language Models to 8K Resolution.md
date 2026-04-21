Title: GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution
Abstract: Ultra-high-resolution (UHR) remote sensing (RS) imagery offers valuable data for Earth observation but pose challenges for existing multimodal foundation models due to two key bottlenecks: (1) limited availability of UHR training data, and (2) token explosion caused by the large image size. To address data scarcity, we introduce SuperRS-VQA (avg. 8,376×8,376) and HighRS-VQA (avg. 2,000×1,912), the highest-resolution vision-language datasets in RS to date, covering 22 real-world dialogue tasks. To mitigate token explosion, our pilot studies reveal significant redundancy in RS images: crucial information is concentrated in a small subset of object-centric tokens, while pruning background tokens (e.g., ocean or forest) can even improve performance. Motivated by these findings, we propose two strategies: Background Token Pruning and Anchored Token Selection, to reduce the memory footprint while preserving key semantics. Integrating these techniques, we introduce GeoLLaVA-8K, the first RS-focused multimodal large language model capable of handling inputs up to 8K×8K resolution, built on the LLaVA framework. Trained on SuperRS-VQA and HighRS-VQA, GeoLLaVA-8K sets a new state-of-the-art on the XLRS-Bench. Datasets and code were released at GeoLLaVA-8K.

Section: Introduction
With the rapid development of Earth science, the collection, process, and representation of remote sensing (RS) data have become increasingly important. [1]. Among these data sources, satellite imagery is able to capturing extensive spatiotemporal information about Earth's surface [2,3], significantly enhancing our geographic understanding of this planet.
Recent advances in multimodal large language models (MLLMs) [4,5,6,7,8] have significantly improved visual understanding and reasoning, simultaneously facilitating remarkable scientific progress in geoscience for handling remote sensing data [9,10,11]. However, despite significant progress of MLLMs across both general and RS domains, current models still fall short in addressing real-world RS tasks, especially in the case of ultra-high-resolution (UHR) scenarios. For instance, even leading models like GPT-4o [12] and Qwen-VL series [13] are limited up to 4K resolution, resulting in limited performance (Accuracy < 0.45) on the XLRS-Bench [14], a recent RS evaluation benchmark featuring large image size (e.g., 8K-10K). This gap raises an urgent problem we aim to investigate in this paper:
this section cite: ['b0', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13']

Section: Route Planning
In the image , What is the best path from the roundabout left of the center of the picture to the trapezoidal land in the lower right corner? Follow the exit of the roundabout and walk towards the road in the lower -right direction. At the second intersection, turn from the main road to the path and walk straight along this path, then you will reach the trapezoidal land.
this section cite: []

Section: More Reasoning …
(c) Example of our dataset Figure 1: We introduce SuperRS-VQA and HighRS-VQA, the highest-resolution VQA datasets for MLLM training. GeoLLaVA-8K, trained on these datasets, significantly outperforms existing MLLMs on ultra-high-resolution remote sensing tasks. In response to this question, we identify two critical challenges that remain underexplored: (1) The lack of image-text training data for UHR RS images; (2) The massive number of visual tokens in UHR RS imagery increases training difficulty for MLLMs. To address the first issue, we introduce two novel multimodal RS datasets featuring large image sizes: SuperRS-VQA (about 8K×8K) and HighRS-VQA (about 2K×2K). To our knowledge, they are the largest image-size RS vision-language datasets to date, covering 22 real-world subtasks and significantly surpassing previous RS image-text datasets in both scale and diversity.
For the second problem, from our intuition, the excessive sequence length of visual tokens may lead to two major issues in MLLM's training: (1) Expensive Computation Overhead: Current models are not designed to operate at such scales. For example, directly adapting LLaVA to 8K×8K inputs results in memory overflow. (2) Low Semantic Density: UHR RS imagery contains an abundance of homogeneous background tokens, which contribute little useful information. In contrast, semantically rich foreground tokens are sparse and risk being overlooked without effective token management.
To tackle these challenges, we propose two token compression strategies: Background Token Pruning and Anchored Token Selection, which focus on token aggregation and refinement to efficiently manage the large number of visual tokens introduced by UHR inputs. Building on these techniques, we develop GeoLLaVA-8K, a UHR-oriented, RS-specific MLLM capable of processing inputs up to 8K resolution. Experiments on large image-size benchmarks (e.g., XLRS-Bench) demonstrate that our model outperforms all existing open-and closed-source MLLMs, setting a new state-of-the-art.
In summary, our main contributions are as follows:
(1) We curate SuperRS-VQA and HighRS-VQA, two RS image-text datasets covering 22 realworld subtasks for UHR scenes, featuring so far the largest image sizes in our knowledge. (2) We investigate the challenges in training MLLMs caused by the excessive visual tokens in UHR RS imagery. To address the problems of redundant backgrounds and the sparsity of semantically rich objects, we propose two strategies for token pruning and selection. (3) We develop GeoLLaVA-8K, the first RS MLLM tailored for UHR scenes, capable of processing inputs up to 8K. Experiments on representative UHR RS benchmarks demonstrate its superior performance compared to existing open-and closed-source MLLMs.
2 Related Work Remote Sensing Multimodal Datasets. With the rapid progress of large multimodal models in general domains, the RS field has witnessed significant advancements in MLLM development [9,10,11,15,16], leading to the emergence of several RS-specific vision-language datasets. RSVQA [17] comprises image/question/answer triplets, with questions and answers derived from OpenStreetMap (OSM). RSIVQA [18] generates samples automatically using existing scene classification and object detection datasets. RSSA [19] targets hallucination, while FIT-RSRC [16] focuses on object relationship understanding. VRSBench [20] includes 29,614 images, 52,472 object references, and 123,221 QA pairs. Recent datasets for the SFT stage of MLLM training include GeoChat_Instruct [9], ChatEarthNet [21], VHM_sft [22], FIT-RS [16], and MMRS-1M [15]. However, most of these datasets are aggregated from existing sources and contain limited original annotations. Critically, their average image resolution remains below 1K×1K, which is insufficient for real-world UHR RS tasks.Recently, XLRS-Bench [14] introduces the largest image-size RS benchmark to date (average 8,500×8,500) for evaluation purposes only, underscoring the persistent lack of corresponding UHR training data in the RS MLLM field.
this section cite: ['b8', 'b9', 'b10', 'b14', 'b15', 'b16', 'b17', 'b18', 'b15', 'b19', 'b8', 'b20', 'b21', 'b15', 'b14', 'b13']

Section: Multimodal Large Language Model.
Leveraging advanced large language models (LLMs) like GPTs [5] and LLaMA [6], MLLMs have demonstrated strong visual understanding and reasoning capabilities [23,24]. Proprietary models such as Gemini [4] and GPT-4o [5], along with opensource alternatives like Qwen-VL [8], InternLM-XComposer [25], MiniCPM [26], LLaVA [27], and MiniGPT-4 [7], show competitive performance but are typically limited to input resolutions of 2K-4K. To overcome this limitation, LLaVA-Next [28] processes images in patches and connects them via global tokens, Monkey [29] and LLaVA-UHD [30] compress patches to reduce redundancy, Cambrian [31] uses learnable queries for multi-scale interaction, and SliME [32] applies dual compression to preserve global context. In the RS domain, several MLLMs have been developed. For example, GeoChat [9] enables multi-task RS dialogue based on the LLaVA-1.5 framework [33]. EarthGPT [15] unifies multisensor interpretation tasks by converting RS annotations into question-answer pairs. Meanwhile, LHRS-Bot [10] improves vision-language understanding through multi-level alignment and curriculum learning. Nevertheless, both general-domain and RS MLLMs struggle to effectively handle UHR RS imagery. Universal MLLMs lack domain adaptation and are unable to meet the higher resolution requirements of real-world RS applications. Although RS-specific MLLMs incorporate domain knowledge, they still operate at significantly lower resolutions, i.e., less than 1K, highlighting the urgent need for a UHR (8K×8K) and domain-aligned MLLMs in the RS field.
this section cite: ['b22', 'b23', 'b7', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b30', 'b31', 'b8', 'b32', 'b14', 'b9']

Section: Dataset
The RS domain currently lacks sufficient UHR image-text training data for MLLM development.
To validate this judgment, we firstly unified existing RS image-text datasets with general-domain data in a 2:1 ratio for supervised fine-tuning (SFT). However, we observed a notable performance drop in LLaVA-Next-2K [34] on UHR benchmarks (XLRS-Bench [14]) in the Tab. 1, which we attribute to the limited resolution of current RS training datasets, highlighting the urgent need for UHR image-text pairs to support real-world RS applications.
To address this, following XLRS-Bench [14], we first manually annotated 12K UHR samples. Notably, existing MLLMs such as GPT-4o [12] either encounter memory overflow or generate lowquality outputs when processing UHR RS imagery. Therefore, we adopt a fully manual annotation approach in this work. Nevertheless, due to the high time and labor costs, scaling manual annotation of UHR data is impractical. To address this, we develop a semi-automated annotation pipeline that leverages GPT-4o [12] alongside existing RS detection and segmentation datasets, generating 100K medium-to-high-resolution (MHR, 2K×2K) image-text pairs. To bridge the distribution gap between MHR and UHR data, we further apply an influence-based data selection method built on the LESS framework [35]. Next, we detailed the procedure of constructing datasets. GeoChat_Instruct [9] 283k 632×619 ChatEarthNet [21] 6k 256×256 VHM_sft [22] 150k 643×638 FIT-RS [16] 1200k 512×512 MMRS-1M [15] 308K 499×491 VRSBench-train [20] 85k 512×512 Source Screening. Our datasets have diverse and extensive image sources. We follow a key principle: to avoid using the same data sources as existing vision-language datasets as far as possible, ensuring diversity and effectiveness in training data. Specifically, we utilize datasets such as Deep-Globe [36], STAR [37], FAIR1M 2.0 [38], LoveDA [39], Inria [40], OpenSatMap [41], HRSCD [42], MiniFrance [3], and DOTA [2].
To minimize redundancy of image-text pairs, we deduplicate images within these datasets and remove overlaps with existing benchmark datasets like XLRS-Bench [14]. Nonetheless, it should be noted that, since generating MHR image-text pairs requires leveraging existing annotations with tools like GPT-4o, the overlap with prior VQA datasets is unavoidable.
this section cite: ['b33', 'b13', 'b13', 'b11', 'b11', 'b34', 'b8', 'b20', 'b21', 'b15', 'b14', 'b19', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b13']

Section: Annotation.
For UHR data, we formed a team of 5 MLLM and RS experts (Ph.D. holders or candidates) and 30 crowd-sourcing annotators (undergraduate and masters students). All samples were manually labeled and rigorously cross-validated over 40 days of annotation and 10 days of verification, resulting in a 12K UHR image-text dataset, named SuperRS-VQA. For MHR data, we developed a semi-automated annotation. Using task-specific prompts and existing annotations (e.g., bounding boxes in RS detection datasets), we generated text via GPT-4o [12]. Despite high token costs (>$1,000), outputs still needed to be quality-checked by annotators.
this section cite: ['b11']

Section: Data Selection Pipeline for MHR Data.
We further adopted an influence-based data selection pipeline (see Fig. 3) to improve the relevance of our dataset to UHR downstream tasks and ensure its cultivation of reasoning capabilities for models fine-tuned on it. This pipeline quantifies the exact contribution of each training data sample by measuring the change in validation performance when that sample is removed (i.e., a Leave-One-Out influence score). Concretely, we leverage gradient information from a warm-up fine-tuned model θ trained on the crude dataset: for each candidate z ∈ Z among training examples and validation examples x ∈ X, we compute
Inf SGD (z i , X) = max x∈X cos⟨∇ℓ θ (z i ), ∇ℓ θ (x i )⟩(1)
where ℓ θ is the negative log-likelihood loss on a data sample. The above estimation is a simplified but empirically valid form of Influence Function [43], a widely used data valuation method which is a first-order approximation of performance change. Intuitively, a higher Inf SGD indicates removing z i from the training set would cause a larger, more directionally coherent shift in validation loss, and hence that z i is more influential. Unlike heuristic selection methods based on the apparent semantics of VQA pairs, which often require laborious task-specific rule engineering, our approach directly measures data influence utilizing the preferences of a fine-tuned model.
We follow LESS [35] for a practical and efficient implementation of our method. The main challenge of using influence-based methods is gradient storage consumption, despite the Hessian calculation is already omitted in our formulation. Therefore, only gradients of LoRA adapters are calculated. Moreover, the gradients are projected to 8192-dimensional subspace via a fixed random matrix. We
Original Datasets Add Random LoRA Adapter LoRA Gradient Use Pretrained warm up Checkpoint Random mapping Gradient Datastore Examples in the Validation Set Extract gradient features G val D val Compute Inf SGD Select data with the highest similarity D train . . . . . . D train Extract gradient features G train Selected Data
Figure 3: Data Selection Pipeline for MHR Data. In step 1, we train a warmup model on the crude dataset to acquire gradient features for both the training and validation sets. In step 2, we match training data candidates with the validation set and select the most influential samples. note that this leverages the Johnson-Lindenstrauss Lemma [44] that random projection preserves the inner product of gradients, and the recent empirical validation [45] for data selection. The cosine similarity is used instead of the original inner product in common influence definitions because the gradient norm of a language model may have numerical biases [35]. After profiling gradients of the crude training set and the validation set, we rank and retain the top 70% most influential samples for downstream fine-tuning. 1 and 4.2 separately analyze the low information density of RS imagery from two views: Section 4.1 reveals the redundancy of background tokens and their negative impact on MLLM modeling, while Section 4.2 explores the correlation between scarce objects visual tokens. In Section 4.3, we propose a targeted solution to overcome these challenges.
this section cite: ['b42', 'b34', 'b43', 'b44', 'b34']

Section: Overwhelming Background Tokens Hinder MLLM Fine-Tuning on RS Data

this section cite: []

Section: Do background tokens dominate UHR RS imagery?
Figure 4: Examples of tokens and their positions that the logit lens yields in the late layers.
In order to learn about a holistic recognition for the background ratio in UHR RS scenes, we conduct preliminary experiments, where the LLaVA-1.5 [33] is employed. Specifically, we randomly selected 100 images from the XLRS-Bench [14], covering natural backgrounds such as sea surfaces, forests, and fields, to provide a diverse set of test samples for the evaluation of background redundancy. Then, we extract 64×64 small images through non-overlapped sliding windows, and sequentially input each into the model. We evaluated the background rate of the UHR RS scene using two methods, following the previous work [46]: Generative Description: We prompt the model with "Describe the image" and analyze the generated description to extract key information. Binary Polling: We ask the model "Is this image mainly background?" where the background is defined as low-information natural areas (e.g., sea, deserts, vegetation), excluding artificial structures (e.g. buildings, roads, urban areas).
We developed a multi-level semantic parsing framework to precisely quantify MLLM's responses. The framework separates the description from the evaluation content and processes it in five steps: Urban Semantic Recognition, Human-made Structure Analysis, Natural Element Classification, Semantic Categorization, and Probability Gradient Quantification. Details are shown in the appendix. Results show background coverage in RS images reaches up to 73.14%.
To better understand the semantic information of visual tokens, we use the logit lens technique [47].
For each layer, we decode the activation at each token position via unembedding. Details are shown in the appendix. With the logit lens technique, we found that in large background areas, only a small portion of the information in the visual tokens is effectively mapped to specific semantic scenes, while most of the background tokens do not align with clear semantic representations (Fig. 4(b)).
this section cite: ['b32', 'b13', 'b45', 'b46']

Section: Do background tokens in RS imagery hinder MLLMs from effectively modeling UHR satellite images?
Then, we feed the whole UHR RS imagery into MLLM, where the LLaVA-Next-2K [34], a variant of LLaVA-Next [28], is adopted to support higher resolution as far as possible. Initially, we directly perform the inference on XLRS-Bench [14], whose images have an average resolution of 8K×8K. Naturally, the resulting long visual token sequences lead to out-of-memory (OOM) errors, aligning with the first challenge presented in Section 1.
44.3 43.2 42.5 43 43.5 44 44.5 Visual Encoder LLM indicates the pruned background token. ① Average Pooling 6 x 6 ② Pruning Background Tokens after 6x6 Pooling OOM Error LLaVA-Next-2K LLaVA-Next-9K LLaVA-Next-9K Based on semantic similarity to background. Accuracy ( % ) + 1.1% 2 x 2 16 tokens / grid desert e.g. ocean grass 8 tokens / grid Pruned 50% background 9,408×9,408 9,408×9,408 9,408×9,408 LLaVA-Next-9K LLaVA-Next-9K LLaVA-Next-9K-Pruning Pruning 50% tokens As Fig. 5 shows, LLaVA-Next-9K underperforms due to its lack of exposure to 9K-resolution training data, emphasizing the need for UHR RS datasets. Surprisingly, despite LLaVA-Next-9K-Pruning uses only half the tokens of LLaVA-Next-9K, it achieves better accuracy. This suggests that excessive background tokens not only introduce additional computational overhead but also impair performance. These findings underscore a critical insight: reducing background redundancy is critical to perform effective modeling under UHR RS images.
From the above two experiments, we confirm that at the token level, RS images exhibit significant background redundancy, where key target information constitutes only a small fraction of the image, while most regions lack explicit semantic significance. This presents a major challenge to the efficiency of multimodal models.
this section cite: ['b33', 'b27', 'b13']

Section: Scarce Object Tokens Drive MLLM Fine-Tuning on RS Data
After recognizing the background redundancy of UHR RS imagery in visual tokens, we turn to the second key aspect of understanding RS's low semantic density: the localization of critical information.
Whether essential information is concentrated in small targets and captured by corresponding visual tokens? x We ablate some visual tokens that potentially contain objectspecific information, y prompt the model to describe the image, or answer objectspecific questions, then z measure the impact of token ablation by calculating the percentage of initially correct object identifications that become incorrect after ablation.
We conduct ablation experiments to test whether the foreground information is concentrated in specific visual tokens. By removing selected tokens and observing the drop in recognition performance, we assess their importance.
The images are sourced from XLRS-Bench, with two pre-processing steps applied to ensure reliability: High signal-to-noise subimages and Hallucination control. Details are shown in the appendix. Following this procedure, the final dataset includes 1,189 VQA pairs. Fig. 6 provides an overview of our ablation. Details are shown in the appendix.
We define the token subset S for ablation using four settings: (1) Object Tokens: Tokens aligned with image patches that originally contain the target object; (2) Object Tokens with Buffer: Object tokens along with their surrounding tokens.(3) Register Tokens: Tokens whose norms deviate by more than two standard deviations from the mean, corresponding to the register tokens identified as encoding global image features [48]; (4) Random Tokens: A baseline in which n tokens are randomly ablated. Tab. 3 shows that ablating object tokens significantly hinders the models ability to recognize targets. Generative decrease and VQA decrease experiments are following the previous work [46]. A larger percentage drop in performance indicates a greater impact of the ablation, meaning the model is more likely to answer incorrectly, thus suggesting that the ablated tokens contain more localized object information. Notably, with a comparable number of ablated tokens, removing object tokens consistently results in a larger performance degradation than random ablation, highlighting the precise localization of object-specific information. logit lens analysis [47] further reveals that visual tokens for scarce objects efficiently converge to accurate semantic representations (Fig. 4 (a)). Collectively, these findings underscore a key insight in RS: object tokens not only encode critical information but also align closely with actual visual features.
this section cite: ['b47', 'b45', 'b46']

Section: Background Token Pruning and Anchored Token Selection
Through the above analysis, we have identified the Low Semantic Density challenge in UHR RS imagery, characterized by both background and foreground tokens, which poses a significant obstacle in training MLLMs. Based on these findings, we argue that an RS-specific MLLM capable of efficiently handling UHR imagery is both necessary and feasible. As shown in Fig. 7, we propose a two-step token selection strategy for background and object tokens in RS imagery. Specifically, building on our two high-resolution RS datasets: SuperRS-VQA and HighRS-VQA, we SFT existing MLLMs, where we select LLaVA-Next-2K [34], to create RS MLLMs. In practice, we initialize from LLaVA-Next-2K's general-domain pretrained weights.
this section cite: ['b33']

Section: Step 1: Pruning Background Token via Semantic Affinity

this section cite: []

Section: Background Token Pruning via Semantic Affinity.
To address the redundancy of background tokens in visual token sequences, we propose an adaptive token clustering strategy for compressing background tokens. Specifically, we construct token-to-token associations by assigning each token p = (u, v) to a neighboring token s with probability q_s(p). Rather than applying this globally, we restrict associations to the local neighborhood N _p, ensuring:
∑ s∈Np q s (p) = 1(2)
After performing this computation once, we iteratively reapply the same process to the selected tokens for N steps, gradually forming an initial cluster. We assume that background tokens, such as those containing ocean features, exhibit high similarity and thus are suited for clustering. We perform this process for all tokens. Note that the clusters formed during later iterations may overlap with those from earlier steps. In such cases, we merge the overlapping clusters into a single one. This adaptive clustering strategy allows us to effectively group similar background tokens, e.g., oceanrelated tokens, into one irregular cluster, and forest-related tokens into another, thus compressing redundant visual information more effectively.
this section cite: []

Section: Anchored Token Selection for Scarce Object Retaining.
To prevent small-object or otherwise informative tokens from being lost in a background-oriented pruning stage, we introduce Anchored Token Selection (ATS). ATS leverages the attention map formed between the pretrained ViT's [49] [CLS] token and the remaining image tokens after background pruning. Tokens receiving higher [CLS] to patch attention are deemed more semantically important and are kept, as they likely correspond to informative objects. Specially, the attention map a [CLS] ∈ R 1×n from the [CLS] token z [CLS] ∈ R 1×d to other patch tokens Z v ∈ R n×d is computed by
a [CLS] = Softmax ( z [CLS] W Q (Z v W V ) T √ d ) = Softmax ( q [CLS] K T v √ d ) , (3
)
where n is the number of remaining image tokens, d is the dimension of hidden states, and W Q , W V are the query and key projection matrices of this encoder layer. Note that we utilize the attention map at the second-to-last layer of the visual encoder (i.e., the output layer of CLIP-ViT in LLaVA-Next [28]). Given a compression ratio r, we calculate the final number of tokens to retain as R = n × r. Ultimately, we select the top R tokens with the highest attention scores from the attention map a [CLS] , obtaining the final retained tokens Z ′ v ∈ R R×d . After passing through a multi-modal projector g, the remaining image tokens are concatenated with language instructions H q and fed into the language model with trainable parameters f ϕ to generate the response X a with L text tokens in the auto-regression manner, where the probability of X a is computed by p(X a |g(
Z ′ v ), H q ) = ∏ L i=1 f ϕ (x i |g(Z ′ v ), H q , X a<i ).
this section cite: ['b48', 'b27']

Section: Experiments
We perform SFT training on the SuperRS-VQA and HighRS-VQA datasets, with a brief overview of the training details and results in this section. Exploratory and ablation studies are presented in Section 3 to clarify the research motivation. Additional details, ablation studies of dataset and method, as well as case analyses, are provided in the appendix.
this section cite: []

Section: Main Results
Experimental Setup. We use XLRS-Bench [14] for evaluation. The MLLMs evaluated on XLRS-Bench are grouped into three categories: (a) open-source MLLMs; (b) closed-source MLLMs and (c) the specialized RS model. For fair comparison, we used a zero-shot setting with uniform prompts for all MLLMs, including our work. The appendix details the architecture and parameter sizes of each open-source MLLMs, and includes additional results across various settings. Except for GeoChat which was evaluated using its native framework, all other models were evaluated using LMMs-Eval [50,51]. Following XLRS-Bench [14], we evaluated the accuracy and reported of L-1 dimension for the VQA task, with L-3 and L-4 results available in the appendix. Main Results. After fine-tuning on SuperRS-VQA and HighRS-VQA, our GeoLLaVA-8K delivers outstanding performance across various evaluation tasks. It not only outperforms domain-specific models but also surpasses all existing open-and closed-source models, including the latest Qwen2.5 and InternVL3. Remarkably, with just 7B parameters, GeoLLaVA-8K even outperforms Qwen2.5-VL-72B, the largest and best-performing open-source MLLMs. This impressive gain stems from the high-quality dataset and the targeted compression strategy designed for the low semantic density of RS imagery.
this section cite: ['b13', 'b49', 'b50', 'b13']

Section: Further Analyses

this section cite: []

Section: Effect of High-Resolution Data vs. Token Optimization Strategies
We conducted an ablation study to analyze the effects of high-resolution data and token optimization strategies.
As shown in Table 6, using high-resolution datasets (SuperRS-VQA and HighRS-VQA) already improves performance over the baseline. When combined with Background Token Pruning (BTP) and Anchored Token Selection (ATS), the model achieves the best average accuracy of 51.5%. This shows that while high-resolution data enhances visual understanding, most performance gains come from token optimization, which focuses attention on semantically important regions.
this section cite: []

Section: Small Object Preservation and CLS Attention under Token Compression.
We analyzed the impact of token compression and CLS-based attention on small-object scenes using a subset of XLRS-Bench defined by COCOs area criteria. Small objects were defined as occupying less than 5% of the image area, and categorized into three levels: Extremely Small (<0.1%), Very Small (0.1%-2.0%), and Normal Small (2.0%-5.0%). A dedicated subset of 1,570 samples covering six tasks (classification, color, motion state, counting, spatial relations) was curated. As shown in Table 7, GeoLLaVA-8K outperformed LLaVA-Next across all tasks, with notable improvements in spatial reasoning, counting, and motion state. These results show that token optimization effectively preserves smallregion semantics and reduces information loss. The CLS attention mechanism generally focuses on task-relevant small objects but occasionally favors a single dominant region, suggesting room for further improvement in multi-object focus and attention balance. Table 8: Generalization performance on LRS-VQA.
this section cite: []

Section: Method Accuracy (%) Evaluation Format
LLaVA-Next [28] 55.07 MCQ GeoLLaVA-8K (ours)
this section cite: ['b27']

Section: MCQ

this section cite: []

Section: Generalization on External Datasets
To evaluate the generalization of GeoLLaVA-8K, we tested it on the new ultra-high-resolution benchmark LRS-VQA [59]. Since LRS-VQA only provides openended QA annotations, we converted it to a multiplechoice format using an LLM [12] to generate three plausible distractors per question, followed by structural validation. GeoLLaVA-8K achieved 56.28% accuracy, outperforming LLaVA-Next [28] (55.07%) by 1.21%, confirming strong generalization to new datasets and formats.
this section cite: ['b58', 'b11', 'b27']

Section: Conclusion
In this paper, we addressed two fundamental challenges in scaling vision-language models to UHR RS imagery: the lack of suitable training data and the computational burden of token explosion. To tackle these issues, we introduced two new UHR image-text datasets, SuperRS-VQA and HighRS-VQA, which greatly expand the data available for vision-language tasks on UHR RS images. We also proposed two novel token-efficient strategies: Background Token Pruning and Anchored Token Selection, which effectively reduce the number of visual tokens processed from 8K-resolution images while preserving essential information. Building on these contributions, we developed GeoLLaVA-8K, the first RS-specific MLLM that can directly handle inputs up to 8K×8K resolution. GeoLLaVA-8K achieved state-of-the-art performance on the XLRS-Bench benchmark, outperforming both open-and closed-source MLLMs and demonstrating the effectiveness of our token-efficient approach. These results underscore the value of domain-adapted, token-efficient modeling and provide a new foundation for high-fidelity image understanding in RS.
Limitation. We have primarily focused on optical satellite imagery; evaluating GeoLLaVA-8K on other sensor modalities (e.g., synthetic aperture radar or multispectral images) will be important to ensure broader applicability.
this section cite: []

Section: References
Ref_id:b0 Title: Crop classification based on feature band set construction and object-oriented approach using hyperspectral images Year: (2016-09)
Ref_id:b1 Title: Dota: A large-scale dataset for object detection in aerial images Year: (2018)
Ref_id:b2 Title: Semi-supervised semantic segmentation in earth observation: The minifrance suite, dataset analysis and multi-task network study Year: (2022)
Ref_id:b3 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b4 Title: Gpt-4 technical report Year: (2024)
Ref_id:b5 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b6 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2024)
Ref_id:b7 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b8 Title: Geochat: Grounded large vision-language model for remote sensing Year: (2024)
Ref_id:b9 Title: Lhrs-bot: Empowering remote sensing with vgi-enhanced large multimodal language model Year: (2024)
Ref_id:b10 Title: Rsgpt: A remote sensing vision language model and benchmark Year: (2023)
Ref_id:b11 Title: Hello gpt Year: (2024)
Ref_id:b12 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b13 Title: Xlrs-bench: Could your multimodal llms understand extremely large ultra-high-resolution remote sensing imagery Year: (2025)
Ref_id:b14 Title: Earthgpt: A universal multi-modal large language model for multi-sensor image comprehension in remote sensing domain Year: (2024)
Ref_id:b15 Title: Skysensegpt: A fine-grained instruction tuning dataset and model for remote sensing vision-language understanding Year: (2024)
Ref_id:b16 Title: Rsvqa: Visual question answering for remote sensing data Year: (2020)
Ref_id:b17 Title: Mutual attention inception network for remote sensing visual question answering Year: (2021)
Ref_id:b18 Title: H2rsvlm: Towards helpful and honest remote sensing large vision language model Year: (2024)
Ref_id:b19 Title: Vrsbench: A versatile vision-language benchmark dataset for remote sensing image understanding Year: (2024)
Ref_id:b20 Title: Chatearthnet: A globalscale image-text dataset empowering vision-language geo-foundation models Year: (2024)
Ref_id:b21 Title: Vhm: Versatile and honest vision language model for remote sensing image analysis Year: (2025)
Ref_id:b22 Title: Learning to learn better visual prompts Year: (2024)
Ref_id:b23 Title: Decodingtrust: A comprehensive assessment of trustworthiness in gpt models Year: (2023)
Ref_id:b24 Title: Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition Year: (2023)
Ref_id:b25 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b26 Title: Visual instruction tuning. Advances in neural information processing systems Year: (2023)
Ref_id:b27 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024)
Ref_id:b28 Title: Monkey: Image resolution and text label are important things for large multi-modal models Year: (2024)
Ref_id:b29 Title: Llava-uhd: an lmm perceiving any aspect ratio and highresolution images Year: (2024)
Ref_id:b30 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b31 Title: Beyond llava-hd: Diving into high-resolution large multimodal models Year: (2024)
Ref_id:b32 Title: Improved baselines with visual instruction tuning Year: (2023)
Ref_id:b33 Title: Long context transfer from language to vision Year: (2024)
Ref_id:b34 Title: Less: Selecting influential data for targeted instruction tuning Year: (2024)
Ref_id:b35 Title: Deepglobe 2018: A challenge to parse the earth through satellite images Year: (2018)
Ref_id:b36 Title: Star: A first-ever dataset and a large-scale benchmark for scene graph generation in large-size satellite imagery Year: (2024)
Ref_id:b37 Title: Fair1m: A benchmark dataset for fine-grained object recognition in high-resolution remote sensing imagery Year: (2022)
Ref_id:b38 Title: Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation Year: (2021)
Ref_id:b39 Title: Large-scale semantic classification: outcome of the first year of inria aerial image labeling benchmark Year: (2018)
Ref_id:b40 Title: Opensatmap: A fine-grained high-resolution satellite dataset for large-scale map construction Year: (2024)
Ref_id:b41 Title: Multitask learning for large-scale semantic change detection Year: (2019)
Ref_id:b42 Title: Understanding black-box predictions via influence functions Year: (2017)
Ref_id:b43 Title: Extensions of lipschitz mappings into a hilbert space Year: (1984)
Ref_id:b44 Title: What is your data worth to gpt? llm-scale data valuation with influence functions Year: (2024)
Ref_id:b45 Title: Towards interpreting visual information processing in vision-language models Year: (2024)
Ref_id:b46 Title: 8wDpdaN6v6ru/interpreting-gpt-the-logit-lens Year: (2020-08)
Ref_id:b47 Title: Vision transformers need registers Year: (2023)
Ref_id:b48 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b49 Title: Lmms-eval: Reality check on the evaluation of large multimodal models Year: (2024)
Ref_id:b50 Title: Lmms-eval: Accelerating the development of large multimodal models Year: (2024-03)
Ref_id:b51 Title: Gpt-4o mini: advancing cost-efficient intelligence Year: (2024)
Ref_id:b52 Title: Anthropic ai Year: (2023)
Ref_id:b53 Title: Mastering free-form text-image composition and comprehension in vision-language large model Year: (2024)
Ref_id:b54 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b55 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
Ref_id:b56 Title: Enhancing the reasoning ability of multimodal large language models via mixed preference optimization Year: (2024)
Ref_id:b57 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b58 Title: When large vision-language model meets large remote sensing imagery: Coarseto-fine text-guided token pruning Year: (2025)
Ref_id:b59 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b60 Title: Datasheets for datasets Year: (2021)
Ref_id:b61 Title: How many instances are there in total (of each type, if appropriate)? Year: ()
Ref_id:b62 Title: Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? Year: ()
Ref_id:b63 Title: Is there a label or target associated with each instance? Year: ()
Ref_id:b64 Title: Is any information missing from individual instances Year: ()
Ref_id:b65 Title: Are relationships between individual instances made explicit (e.g., users movie ratings, social network links)? Year: ()
Ref_id:b66 Title: Are there recommended data splits (e.g., training, development/validation, testing)? Year: ()
