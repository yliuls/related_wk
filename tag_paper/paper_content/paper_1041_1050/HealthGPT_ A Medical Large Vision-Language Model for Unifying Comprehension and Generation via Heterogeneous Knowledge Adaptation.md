Title: 
Abstract: We present HealthGPT, a powerful Medical Large Vision-Language Model (Med-LVLM) that

Section: 
The X-ray shows no pleural effusion or pneumothorax.
Here is the chest X-ray image for you.
this section cite: []

Section: Gen. Perf.
Figure 1: HealthGPT enables medical multimodal comprehension and generation, outperforming both state-of-the-art unified visual models and medical-specific models across various tasks. This highlights its superior capability in tackling complex tasks in healthcare applications. Comp.Perf. and Gen.Perf. denote the results of comprehension and generation.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
integrates medical visual comprehension and generation capabilities within a unified autoregressive paradigm. Our bootstrapping philosophy is to progressively adapt heterogeneous comprehension and generation knowledge to pretrained Large Language Models (LLMs). This is achieved through a novel heterogeneous lowrank adaptation (H-LoRA) technique, which is complemented by a tailored hierarchical visual perception (HVP) approach and a three-stage learning strategy (TLS). To effectively learn the HealthGPT, we devise a comprehensive medical
this section cite: []

Section: Introduction
Large Vision-Language Models (LVLMs) (Liu et al., 2023;OpenAI, 2023;Liu et al., 2024c;Chen et al., 2024b) have demonstrated outstanding open-world visual comprehension and reasoning abilities through language-based interactive dialogue over the past years, simultaneously opening up new opportunities for applications in specialized domains. Specifically, recent studies (Li et al., 2024b;Tu et al., 2024;Li et al., 2025) have utilized pre-trained Large Language Models (LLMs) and visual instruction data to build interactive diagnostic tools and treatment planning systems, revealing the immense potential of LVLMs in medical scenarios. However, these studies primarily concentrate on visual comprehension tasks that produce text-based outputs, such as medical visual question answering (Li et al., 2024b) or report generation (Nath et al., 2024), and deficient the "drawing" capability needed for medical visual generation. In practice, integrating visual comprehension and generation can significantly enhance the multifunctionality of medical LVLMs (Med-LVLMs).
Recent studies have increasingly focused on developing unified LVLMs capable of comprehending and generating content across diverse visual modalities. Earlier approaches predominantly utilized continuous visual tokens fed into LLMs, using the LLMs themselves as conditional generators for external generative models (Ge et al., 2024;Wu et al., 2024b;Dong et al., 2024). More recent research has explored the use of discrete visual tokens for image representation and generation within a fully autoregressive framework (Team, 2024a;Wang et al., 2024a;Xie et al., 2024). These methods not only enhance controllability but also demonstrate early success in open-world, any-to-any tasks, highlighting the preliminary potential of a unified autoregressive learning paradigm in multimodal tasks.
While unified LVLMs have achieved initial success in general scenarios, such a unified framework remains underexplored in the medical domain. Adapting the aforementioned general unified model paradigm to the medical domain presents two major challenges: (i) High-scale and -quality Data Limitations. Open-world models necessitate extensive pre-training on billions or even more diverse, multimodal data samples for comprehension and generation tasks (Lu et al., 2024;Team, 2024a). However, the accessible medical data significantly lacks in scale and quality compared to natural multimodal datasets. Its specialized and domain-specific characteristics make it challenging to develop a unified medical model from scratch. (ii) Conflicts between Comprehension and Generation. Comprehension tasks often strip away visual details to focus on abstraction, while generation tasks require detailed preservation, making tokens sensitive to all visual alterations. As shown in Figure 2, which features experiments conducted on medical images, the performance in comprehension (or generation) tasks steadily decreases as the proportion of generation (or comprehension) data increases, and vice versa.
This highlights a dilemma in autoregressive multimodal training, stemming from the need to maintain consistency of features between pre-LVLMs and post-LVLMs. Although some methods have explored mutual enhancement between comprehension and generation tasks (Pan et al., 2024;Tong et al., 2024), joint training in medical scenarios still faces challenges such as task conflict, data bias, and optimization saturation, leading to limited performance gains and persistent degradation issues.
To tackle the aforementioned challenges, we propose HealthGPT (see Figure 1) , which progressively adapts a pre-trained LLM as an unified medical multimodal model with a small amount of visual instruction data. We devise innovative Parameter-Efficient Fine-Tuning (PEFT) approach (Ding et al., 2023), called Heterogeneous Low-Rank Adaptation (H-LoRA), which decouples the learning process of LVLMs for comprehension and generation tasks. Inspired by the plug-and-play nature of LoRA (Hu et al., 2022), H-LoRA enables the model to store heterogeneous comprehension and generation knowledge in independent "plugins", thus avoiding joint optimization issues caused by conflicts between comprehension and generation tasks. In addition, we also consider the variety of sub-tasks among comprehension or generation tasks. Qualitative research highlights the limitations of a single LoRA in handling multidimensional task scenarios, mainly due to catastrophic forgetting and interference (Liu et al., 2024d;Lin et al., 2024). To address this, we draw on the concept of Mixture of Experts (MoE) (Masoudnia & Ebrahimpour, 2014) and introduce LoRA experts. The aim is to dynamically transfer task-shared knowledge to adapt to downstream tasks. Unlike MoELoRA (Luo et al., 2024a), H-LoRA employs reversible matrix block multiplication to combine LoRA experts, significantly reducing the overhead of multiple matrix multiplications. Notably, when using four experts, it requires only 67% of the MoELoRA training time.
To effectively leverage H-LoRA in HealthGPT, we further introduce a Hierarchical Visual Perception (HVP) and devise a corresponding Three-stage Learning Strategy (TLS). HVP: we separate visual details learning from Vision transformer (ViT) for comprehension and generation.
As is widely recognized, the ViT encodes visual concepts with increasing abstraction, generally, becoming finer as we progress over levels (Vig, 2019). Thus, we maintain the visual features of the anterior and posterior layers to accommodate the differing requirements for visual granularity in comprehension and generation tasks while preventing potential task interference. TLS: In the first and second stages, given the heterogeneity between comprehension and generation tasks, we first train H-LoRA plugins for HealthGPT to incorporate both medical comprehension and generation knowledge, thus endowing the LLMs with capabilities for vision-language alignment and vision-to-vision reconstruction. Additionally, through minimal mixed-task training, we built fusion embedding layers and output heads that merge text and visual tokens, establishing a unified Med-LVLM foundation for visual instruction fine-tuning. In the third stage, by only training the H-LoRA plugins, HealthGPT is able to rapidly adapt to a wide range of downstream medical tasks, covering various types of medical comprehension and generation tasks.
To effectively implement our approach, we have curated a dataset for training unified Med-LVLMs, called VL-Health, including seven comprehension tasks and five generation tasks (Figure 1). Through quantitative analysis and validation on multimodal tasks, the results demonstrate that HealthGPT is capable of unifying medical multimodal abilities in data-constrained scenarios, achieving performance comparable to or better than existing state-of-the-art (SOTA) models across multiple metrics. Overall, the main contributions of this paper are summarized as follows:
• Unified Med-LVLM. We introduce HealthGPT, which, to the best of our knowledge, is the first unified framework for multimodal comprehension and generation in complex medical scenarios.
• Effective Learning Paradigm. We present H-LoRA, an optimized multi-LoRA PEFT architecture based on task-gated decoupling, is designed to effectively mitigate data conflict issues.
• Holistic Training Dataset. We curated VL-Health, a comprehensive dataset designed for both comprehension and generation tasks.
• Superior Downstream Improvements: Extensive experiments are conducted and the results confirm effectiveness of our approach in medical vision-language comprehension and generation.
this section cite: ['b32', 'b54', 'b28', 'b43', 'b14', 'b61', 'b10', 'b62', 'b38', 'b45', 'b53', 'b9', 'b16', 'b29', 'b41', 'b55']

Section: Related Work
Medical Vision Large Language Models. In recent years, the development of deep learning brings prosperity to the field in medical applications (Zhang et al., 2022;Ong et al., 2022). Most recently, medical vision large language models (Med-VLLMs) demonstrate excellent performance in understanding medical images and responding to human queries based on these images (Zhou et al., 2023;Tian et al., 2023).
XrayGPT (Thawakar et al., 2023) combines a medical visual encoder (MedClip) (Wang et al., 2022) with a fine-tuned LLM , using a simple linear transformation layer to achieve alignment between visual and textual information, significantly enhancing the understanding of medical images. On this basis, some methods (Wu et al., 2023;Li et al., 2024a;Bansal et al., 2025) further enhance visual-text alignment in medical contexts by selecting high-quality image-text pairs from PubMed papers and synthesized VQA datasets. BiomedGPT (Luo et al., 2024b) employs a BERT-style encoder and GPT-style decoder architecture, pre-trained on interdisciplinary datasets. Compared to commercial models like Med-PaLM (Singhal et al., 2023), BiomedGPT significantly reduces model size while maintaining superior performance. However, issues of language adaptability and dataset specificity still remain. To address these, HuatuoGPT-Vision (Chen et al., 2024a) introduces the Pub-MedVision dataset, which contains 1.3 million high-quality medical samples, significantly improving the model's adaptability across diverse medical applications. However, current Med-VLLMs mainly focus on medical comprehension and lack the capability for the medical vision-language generation.
this section cite: ['b66', 'b44', 'b69', 'b52', 'b50', 'b57', 'b59', 'b2', 'b47']

Section: Unified Visual Comprehension and Generation Models.
Recent research has increasingly concentrated on creating unified LVLMs that are adept at understanding and producing content across various visual modalities. NExT-GPT (Wu et al., 2024b) achieves perception and generation for arbitrary combinations of multimodal inputs and outputs by aligning LLMs. Similarly, SEED (Ge et al., 2023), SEED-X (Ge et al., 2024), and DreamLLM (Dong et al., 2024) employ learnable queries and leverage nexttoken prediction to generate visual tokens, providing conditional inputs to external generation modules. Unlike these methods, which function as external conditioners, Unified-IO (Lu et al., 2022), Unified-IO 2 (Lu et al., 2024), and Chameleon (Team, 2024a) internalize multimodal generation tasks within a unified Transformer architecture by ex- tending multimodal vocabularies, enabling direct generation based on next-token prediction. Building on this concept, Lumina-mGPT (Liu et al., 2024a) and ANOLE (Chern et al., 2024) further enhance the generation capabilities of unified models using high-quality data, particularly improving the quality and flexibility of image generation.
this section cite: ['b61', 'b13', 'b14', 'b10', 'b37', 'b38']

Section: Preliminaries
Large Vision-Language Models. The input to a LVLM typically consists of an image x img and a discrete text sequence x txt . The visual encoder E img converts the input image x img into a sequence of visual tokens V = [v i ] Nv i=1 , while the text sequence x txt is mapped into a sequence of text tokens T = [t i ] Nt i=1 using an embedding function E txt . The LVLM M LVLM (•|θ) models the joint probability of the token sequence U = {V, T }, which is expressed as:
P θ (R|U) = Nr i=1 P θ (r i |{U, r <i }),(1)
where R = [r i ] Nr i=1 is the text response sequence. The LVLM iteratively generates the next token r i based on r <i . The optimization objective is to minimize the cross-entropy loss of the response R. It is worth noting that most LVLMs adopt a design paradigm based on ViT, alignment adapters, and pre-trained LLMs (Liu et al., 2023;2024b), enabling quick adaptation to downstream tasks.
VQGAN. VQGAN (Esser et al., 2021) employs latent space compression and indexing mechanisms to effectively learn a complete discrete representation of images. VQGAN first maps the input image x img to a latent representation z = E(x) through a encoder E. Then, the latent representation is quantized using a codebook Z = {z k } K k=1 , generating a discrete index sequence I = [i m ] N m=1 , where i m ∈ Z represents the quantized code index:
I = Quantize(z|Z) = arg min z k ∈Z ∥z -z k ∥ 2 .
(
In our approach, the discrete index sequence I serves as a supervisory signal for the generation task, enabling the model to predict the index sequence Î from input conditions such as text or other modality signals. Finally, the predicted index sequence Î is upsampled by the VQGAN decoder G, generating the high-quality image ximg = G( Î).
this section cite: ['b32', 'b61', 'b12']

Section: Low Rank Adaptation.
LoRA (Hu et al., 2022) effectively captures the characteristics of downstream tasks by introducing low-rank adapters. The core idea is to decompose the bypass weight matrix ∆W ∈ R d in ×d out into two low-rank matrices {A ∈ R d in ×r , B ∈ R r×d out }, where r ≪ min{d in , d out }, significantly reducing learnable parameters. The output with the LoRA adapter for the input x is then given by:
h = xW 0 + αx∆W /r = xW 0 + αxAB/r, (3)
where matrix A is initialized with a Gaussian distribution, while the matrix B is initialized as a zero matrix. The scaling factor α/r controls the impact of ∆W on the model.
this section cite: ['b16']

Section: HealthGPT

this section cite: []

Section: Unified Autoregressive Generation.
Our approach (Figure 3) utilizes a discrete token representation that covers both text and visual outputs, unifying visual comprehension and generation as an autoregressive task. For comprehension, pre-trained LLM M LLM (•|θ) receives the input joint sequence U and outputs a series of text token R = [r 1 , r 2 , . . . , r Nr ], where r i ∈ V txt , and V txt represents the vocabulary of LLM:
P θ (R | U) = Nr i=1 P θ (r i | U, r <i ).
(4)
For generation, M LLM first receives a special start token ⟨|START IMG|⟩, then generates a series of tokens corresponding to the VQGAN indices I = [i 1 , i 2 , . . . , i Ni ], where i j ∈ V vq , and V vq represents the index range of VQGAN. Upon completion of generation, the LLM outputs an end token ⟨|END IMG|⟩:
P θ (I | U) = Ni j=1 P θ (i j | U, i <j ). (5
)
Finally, the generated index sequence I is fed into the decoder G, which reconstructs the target image ximg = G(I).
this section cite: []

Section: Hierarchical Visual Perception
Given the differences in visual perception between comprehension and generation tasks -where the former focuses on abstract semantics and the latter emphasizes complete semantics-we employ ViT to compress the image into discrete visual tokens at multiple hierarchical levels. Specifically, the image is converted into a series of features {f 1 , f 2 , . . . , f L } as it passes through L ViT blocks.
To address the needs of various tasks, the hidden states are divided into two types: (i) Concrete-grained features F Con = {f 1 , f 2 , . . . ,f k } for k < L, derived from the shallower layers of ViT, containing sufficient global features, suitable for generation tasks; (ii) Abstract-grained features F Abs = {f k+1 , f k+2 , . . . , f L }, derived from the deeper layers of ViT, which contain abstract semantic information closer to the text space, suitable for comprehension tasks.
The task type T (comprehension or generation) determines which set of features is selected as the input for the downstream large language model:
F img T = F Con , if T = Generation Task F Abs , if T = Comprehension Task (6
)
We integrate the image features F img T and text features T into a joint sequence through simple concatenation, which is then fed into the LLM M LLM for autoregressive generation.
this section cite: []

Section: Heterogeneous Knowledge Adaptation
We devise H-LoRA, which stores heterogeneous knowledge from comprehension and generation tasks in separate modules and dynamically routes to extract task-relevant knowledge from these modules. At the task level, for each task type T , we dynamically assign a dedicated H-LoRA submodule θ T , which is expressed as:
R = M LLM (U|θ, θ T ), θ T = {A T , B T , R T outer }. (7)
At the feature level for a single task, H-LoRA integrates the idea of Mixture of Experts (MoE) (Masoudnia & Ebrahimpour, 2014) and designs an efficient matrix merging and routing weight allocation mechanism, thus avoiding the significant computational delay introduced by matrix splitting in existing MoELoRA (Luo et al., 2024a). Specifically, we first merge the low-rank matrices (rank = r) of k LoRA experts into a unified matrix:
A merged , B merged = Concat({A i } k 1 ), Concat({B i } k 1 ), (8
)
where A merged ∈ R d in ×rk and B merged ∈ R rk×d out . The k-dimension routing layer generates expert weights W ∈ R token num×k based on the input hidden state x, and these are expanded to R token num×rk space as follows:
W expanded = αkW/r ⊗ 1 r ,(9)
where ⊗ denotes the replication operation. The overall output of H-LoRA is computed as:
O H-LoRA = (xA merged ⊙ W expanded )B merged ,(10)
where ⊙ represents element-wise multiplication. Finally, the output of H-LoRA is added to the output of frozen pretrained weights to produce the final output: task, we train abstract-grained visual adapters using highquality image-text pairs to align visual embeddings with textual embeddings, thereby enabling the model to accurately describe medical visual content. During this process, the pre-trained LLM and its corresponding H-LoRA submodules remain frozen. In contrast, the medical generation task requires training concrete-grained adapters and H-LoRA submodules while keeping the LLM frozen. Meanwhile, we extend the textual vocabulary to include multimodal tokens, enabling the support of additional VQGAN vector quantization indices. The model trains on image-VQ pairs, endowing the pre-trained LLM with the capability for image reconstruction. This design ensures pixel-level consistency of pre-and post-LVLM. The processes establish the initial alignment between the LLM's outputs and the visual inputs.
O = xW 0 + O H-LoRA . (11
2nd Stage: Heterogeneous H-LoRA Plugin Adaptation.
The submodules of H-LoRA share the word embedding layer and output head but may encounter issues such as bias and scale inconsistencies during training across different tasks. To ensure that the multiple H-LoRA plugins seamlessly interface with the LLMs and form a unified base, we fine-tune the word embedding layer and output head using a small amount of mixed data to maintain consistency in the model weights. Specifically, during this stage, all H-LoRA submodules for different tasks are kept frozen, with only the word embedding layer and output head being optimized. Through this stage, the model accumulates foundational knowledge for unified tasks by adapting H-LoRA plugins.
3rd Stage: Visual Instruction Fine-Tuning. In the third stage, we introduce additional task-specific data to fur-ther optimize the model and enhance its adaptability to downstream tasks such as medical visual comprehension (e.g., medical QA, medical dialogues, and report generation) or generation tasks (e.g., super-resolution, denoising, and modality conversion). Notably, by this stage, the word embedding layer and output head have been fine-tuned, only the H-LoRA modules and adapter modules need to be trained. This strategy significantly improves the model's adaptability and flexibility across different tasks.
this section cite: ['b41']

Section: Experiments

this section cite: []

Section: Data and Experimental Setup
Data Details. We curate VL-Health dataset (see Figure 4). For medical visual comprehension, we leverage multiple medical-specific datasets, including PubMedVision (Chen et al., 2024a), LLaVA-Med (Li et al., 2024a), PathVQA (He et al., 2020), MIMIC-CXR-VQA (Bae et al., 2024), SLAKE (Liu et al., 2021), and VQA-RAD (Lau et al., 2018). Additionally, we incorporate high-quality open-world data from LLaVA-1.5 (Liu et al., 2024b) to preserve the model's general knowledge and instructionfollowing capabilities. For generation tasks, we construct a reconstruction dataset based on LLaVA-558k (Liu et al., 2024b), and also explore two key tasks in personalized medical image enhancement-super-resolution and modality conversion-using the IXI (Davies et al., 2014) and SynthRAD2023 (Thummerer et al., 2023) datasets. In addition to the above datasets, we further evaluate the comprehension task on the OmniMedVQA (Hu et al., 2024) and MMMU (Yue et al., 2024) benchmarks. Detailed data
this section cite: ['b15', 'b1', 'b30', 'b22', 'b8', 'b51', 'b17', 'b65']

Section: Model Details.
We select CLIP-L/14 (Radford et al., 2021) as the visual encoder and used the hidden states of its second and penultimate layers as concrete-grained and abstractgrained features for dynamic hierarchical visual perception of models. Drawing on the successful experiences of LLaVA architectures, we employ a MLP to align the multimodal feature embeddings. We choose the parameter-efficient phi-3-mini (Abdin et al., 2024) and phi-4 (Abdin et al., 2024) as the base models. Additionally, we employ the larger-scale Qwen2.5 (Team, 2024b) to explore the scalability of the proposed method. For visual comprehension and generation tasks, we set the rank of H-LoRA to 16/8 and 64/32 for different base models, with four experts. Additionally, we use the f8-8192 version of VQGAN as the image indexing and upsampling module. The model parameters and training configurations are provided in Appendix A.1 and A.2.
this section cite: ['b46', 'b0', 'b0']

Section: Main Experiments
Comprehension. We compare HealthGPT with several existing models, including medical-specific LVLMs (e.g., Med-Flamingo (Moor et al., 2023), LLaVA-Med (Li et al., 2024a), HuatuoGPT-Vision (Chen et al., 2024a)) as well as recent open-world LVLMs (e.g., BLIP-2 (Li et al., 2023b), LLaVA-v1.5 (Liu et al., 2024b), InstructBLIP (Dai et al., 2023), Yi-VL (Young et al., 2024), InternVL2 (Chen et al., 2024b), Llama-3.2 (Dubey et al., 2024)). Additionally, we test several SOTA unified visual comprehension and generation models, including Show-o (Xie et al., 2024), Unified-IO 2 (Lu et al., 2024), Janus (Wu et al., 2024a), Janus-Pro (Chen et al., 2025), and Emu3 (Wang et al., 2024a). The experimental results are shown in
Table 1, with the following key observations: (i) SOTA Results Compared with LVLMs: In medical visual comprehension tasks, HealthGPT demonstrates superior performance, significantly outperforming both medical-specific models (e.g., HuatuoGPT-Vision) and general-purpose models (e.g., Llama-3.2). (ii) Surpassing Current Unified LVLMs: Despite being trained on billions of data points, unified models still exhibit poor generalization performance in medical visual comprehension. For instance, Unified-IO 2 scored only 33.8. In contrast, HealthGPT-M3, with only 3.8B parameters, scored 61.3 on the medical multimodal unified task, significantly outperforming existing unified models in medical downstream scenarios. (iii) Stable Improvement with Large Base Model: Our method demonstrates excellent scalability, with HealthGPT-L14 and HealthGPT-XL32 achieving a score of 66.4 and 71.1 in the larger model configuration. This result significantly outperforms all other models, highlighting the effectiveness of scaling up the base model for enhanced performance in medical tasks.
It is noteworthy that some Med-LVLMs (Xie et al., 2025) perform continual fine-tuning and evaluation on the training sets of specific benchmarks, which may lead to structural degradation in model outputs and negatively impact generalization. To ensure fair comparison and reliable evaluation of generalization ability, we adopt a unified experimental setting and only compare with models trained on diverse and comprehensive datasets.
Generation. We study three key tasks in medical imaging. (i) Modality Conversion: In this task, we focus on the conversion between CT and MRI modalities for the brain and pelvic regions, designing four specific subtasks. All comparative models (Pix2Pix (Isola et al., 2017), CycleGAN (Zhu et al., 2017), BBDM (Li et al., 2023a), Vmamba (Liu et al., 2024e), and DiffMa (Wang et al., 2024b)) trained a separate model for each sub-task, while HealthGPT unify all tasks into a single training process.
The experimental results, shown in Table 2, demonstrate that our approach outperforms other methods across multiple evaluation metrics. For instance, in the CT2MRI-Brain   task, HealthGPT-M3 achieves an SSIM of 79.38, significantly surpassing traditional methods like Pix2Pix (71.09) and the recent DiffMa (71.47). (ii) Super-Resolution: We conduct 4× super-resolution experiments on the IXI dataset, with the results presented in
Table 3. Notably, most existing methods fail to fully leverage the prior knowledge of key structures in medical images, resulting in significant shortcomings in detail recovery. In contrast, our method significantly mitigates this issue. Specifically, HealthGPT-M3 excels in key metrics such as SSIM, PSNR, and MSE, achieving scores of 78.19, 32.76, and 34.47, respectively. Additionally, HealthGPT-M3 achieves the lowest perception distance of 12.34, further validating its exceptional performance in human visual perception. (iii) Reconstruction: We compare HealthGPT-M3 with unified models with reconstruction capabilities, such as Unified-IO 2 and SEED-X.
The results show that our approach performs better controllability for visual reconstruction. Details are in the Appendix C.5.
To explore the scalability of the proposed method in generation tasks, we also train HealthGPT-L14 with a similar number of trainable parameters to the M3 version. Hence, the similar performance between the two models in Figure 2 and 3 meets our expectations. For efficiency considerations, most open-world unified models are limited to a size of no more than 13B. In alignment with this practical constraint, we did not train the HealthGPT-XL32 on generation tasks.
this section cite: ['b42', 'b7', 'b64', 'b11', 'b62', 'b38', 'b4', 'b63', 'b20', 'b70']

Section: In-Depth Study
Effect of Heterogeneous Low-Rank Adaptation. H-LoRA provides an optimized multi-LoRA architecture for multitask learning. We conduct extensive validation of this structure, with results presented in Table 4, comparing the performance of LoRA, MoELoRA, and H-LoRA in medical unified comprehension and generation tasks. In the majority of comprehension tasks and all generation tasks, H-LoRA demonstrates superior performance, particularly in the Om-niMedVQA benchmark, where it improved from 64.90 to 68.50. Notably, despite some applications of MoELoRA in certain scenarios, it do not show advantages in this task and had a training time approximately 50% longer than LoRA. Figure 5 illustrates the performance of the three PEFT methods in medical visual comprehension and generation tasks across different ranks, with H-LoRA consistently outperforming the other methods in all scenarios, demonstrating significant advantages in handling diverse tasks.
this section cite: []

Section: Different Learning Strategy.
We propose a three-stage learning strategy for H-LoRA that decouples comprehension and generation tasks. Unlike methods that train both tasks simultaneously, our approach reduces performance degradation from task conflicts (see Table 5). In the medical visual comprehension task, mixed training causes catastrophic forgetting and degrades visual reconstruction, whereas our strategy effectively uses the medical embedding knowledge in pre-trained LLMs to mitigate these conflicts. Meanwhile, we examine how fusing heterogeneous H-LoRA plugins in the second training stage results in minimal performance degradation. Detailed results are in the Appendix C.3.
this section cite: []

Section: Hierarchical Visual Perception Analysis.
We conduct an ablation analysis on visual perceptual inputs suitable for comprehension and generation tasks. Figure 6 illustrates that the convergence efficiency for comprehension tasks is significantly higher with abstract-grained visual inputs compared to concrete-grained inputs, whereas generation tasks perform better with concrete-grained inputs. This result further underscores the necessity of the hierarchical visual perception we propose, which suggests that customizing visual inputs at different hierarchies for specific tasks can substantially enhance efficiency.
Report-to-CXR Task. Text-to-image generation is a common component in unified models (Lee et al., 2023;Huang et al., 2023;2024); therefore, we further extend our investigation to this task within the medical domain. We explore the Report-to-CXR task without reference images, using a small amount of CXR data (Johnson et al., 2019) for instruction fine-tuning. Figure 7 annotates images with varying injury degrees and locations, comparing them to healthy CXR images. We observe that HealthGPT effectively generates CXR images based on the instructions, showcasing its potential in healthcare education and auxiliary diagnosis.
this section cite: ['b23', 'b18', 'b48', 'b21']

Section: Conclusion
In this paper, we introduce HealthGPT, a Med-LVLM that unifies medical vision-language understanding and generation through a novel heterogeneous knowledge adaptation approach. Experimental results demonstrate that HealthGPT achieves significant performance improvements across multiple medical understanding and generation tasks, showcasing its potential for healthcare applications.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: Mimic-ext-mimic-cxrvqa: A complex, diverse, and large-scale visual question answering dataset for chest x-ray images Year: (2024)
Ref_id:b2 Title: Medmax: Mixed-modal instruction tuning for training biomedical assistants Year: (2025)
Ref_id:b3 Title: Towards injecting medical visual knowledge into multimodal llms at scale Year: (2024)
Ref_id:b4 Title: Janus-pro: Unified multimodal understanding and generation with data and model scaling Year: (2025)
Ref_id:b5 Title: How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites Year: (2024)
Ref_id:b6 Title: Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation Year: (2024)
Ref_id:b7 Title: Towards general-purpose vision-language models with instruction tuning Year: (2023-12-10)
Ref_id:b8 Title: The ixi dataset Year: (2014)
Ref_id:b9 Title: Parameterefficient fine-tuning of large-scale pre-trained language models Year: (2023)
Ref_id:b10 Title: Dreamllm: Synergistic multimodal comprehension and creation Year: (2024)
Ref_id:b11 Title: The llama 3 herd of models Year: (2024)
Ref_id:b12 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b13 Title: Planting a seed of vision in large language model Year: (2023)
Ref_id:b14 Title: Seed-x: Multimodal models with unified multi-granularity comprehension and generation Year: (2024)
Ref_id:b15 Title: Pathvqa: 30000+ questions for medical visual question answering Year: (2020)
Ref_id:b16 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b17 Title: A new large-scale comprehensive evaluation benchmark for medical lvlm Year: (2024)
Ref_id:b18 Title: T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation Year: (2023)
Ref_id:b19 Title: T2ifactualbench: Benchmarking the factuality of text-toimage models with knowledge-intensive concepts Year: (2024)
Ref_id:b20 Title: Image-toimage translation with conditional adversarial networks Year: (2017)
Ref_id:b21 Title: Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs Year: (2019)
Ref_id:b22 Title: A dataset of clinically generated visual questions and answers about radiology images Year: (2018)
Ref_id:b23 Title: Holistic evaluation of text-to-image models Year: (2023)
Ref_id:b24 Title: Image-toimage translation with brownian bridge diffusion models Year: (2023)
Ref_id:b25 Title: Llava-med: Training a large language-and-vision assistant for biomedicine in one day Year: (2024)
Ref_id:b26 Title: Llava-med: Training a large language-and-vision assistant for biomedicine in one day Year: (2024)
Ref_id:b27 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b28 Title: Boosting comprehensive ophthalmology understanding with tailored dataset, benchmark and model Year: (2025)
Ref_id:b29 Title: Teamlora: Boosting lowrank adaptation with expert collaboration and competition Year: (2024)
Ref_id:b30 Title: A semantically-labeled knowledge-enhanced dataset for medical visual question answering Year: (2021)
Ref_id:b31 Title: Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining Year: (2024)
Ref_id:b32 Title: Visual instruction tuning Year: (2023)
Ref_id:b33 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b34 Title: Llava-next: Improved reasoning, ocr, and world knowledge Year: (2024)
Ref_id:b35 Title: When moe meets llms: Parameter efficient finetuning for multi-task medical applications Year: (2024)
Ref_id:b36 Title: Vmamba: Visual state space model Year: (2024)
Ref_id:b37 Title: Unified-io: A unified model for vision, language, and multi-modal tasks Year: (2022)
Ref_id:b38 Title: Unified-io 2: Scaling autoregressive multimodal models with vision language audio and action Year: (2024)
Ref_id:b39 Title: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models Year: (2024)
Ref_id:b40 Title: Biomedgpt: An open multimodal large language model for biomedicine Year: (2024)
Ref_id:b41 Title: Mixture of experts: a literature survey Year: (2014)
Ref_id:b42 Title: Med-flamingo: a multimodal medical few-shot learner Year: (2023)
Ref_id:b43 Title: Vila-m3: Enhancing vision-language models with medical expert knowledge Year: (2024)
Ref_id:b44 Title: Application of artificial intelligence methods for imaging of spinal metastasis Year: (2022)
Ref_id:b45 Title: Auto-encoding morphtokens for multimodal llm Year: (2024)
Ref_id:b46 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b47 Title: Large language models encode clinical knowledge Year: (2023)
Ref_id:b48 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b49 Title: Qwen2.5: A party of foundation models Year: ()
Ref_id:b50 Title: Chest radiographs summarization using large medical vision-language models Year: (2023)
Ref_id:b51 Title: Synthrad2023 grand challenge dataset: Generating synthetic ct for radiotherapy Year: (2023)
Ref_id:b52 Title: The role of large language models in medical image processing: a narrative review Year: (2023)
Ref_id:b53 Title: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b54 Title: Towards generalist biomedical ai Year: (2024)
Ref_id:b55 Title: A multiscale visualization of attention in the transformer model Year: (2019)
Ref_id:b56 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b57 Title: Contrastive learning from unpaired medical images and text Year: (2022)
Ref_id:b58 Title: Soft masked mamba diffusion model for ct to mri conversion Year: (2024)
Ref_id:b59 Title: Towards generalist foundation model for radiology by leveraging web-scale 2d3d medical data Year: (2023)
Ref_id:b60 Title: Decoupling visual encoding for unified multimodal understanding and generation Year: (2024)
Ref_id:b61 Title: Next-gpt: Any-to-any multimodal llm Year: (2024)
Ref_id:b62 Title: Show-o: One single transformer to unify multimodal understanding and generation Year: (2024)
Ref_id:b63 Title: Medtrinity-25m: A large-scale multimodal dataset with multigranular annotations for medicine Year: (2025)
Ref_id:b64 Title: Open foundation models by 01 Year: (2024)
Ref_id:b65 Title: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b66 Title: Boosting medical image semi-supervised learning with adaptive pseudo labeling and informative active annotation Year: (2022)
Ref_id:b67 Title: Hyperllava: Dynamic visual and language expert tuning for multimodal large language models Year: (2024)
Ref_id:b68 Title: Revisiting the domain shift and sample uncertainty in multi-source active domain transfer Year: (2024)
Ref_id:b69 Title: A survey of large language models in medicine: Progress, application, and challenge Year: (2023)
Ref_id:b70 Title: Unpaired image-to-image translation using cycle-consistent adversarial networks Year: (2017)
