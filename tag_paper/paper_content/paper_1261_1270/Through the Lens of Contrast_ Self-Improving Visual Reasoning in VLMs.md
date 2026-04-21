Title: THROUGH THE LENS OF CONTRAST: SELF-IMPROVING VISUAL REASONING IN VLMS
Abstract: Reasoning has emerged as a key capability of large language models. In linguistic tasks, this capability can be enhanced by self-improving techniques that refine reasoning paths for subsequent finetuning. However, extending these language-based self-improving approaches to vision language models (VLMs) presents a unique challenge: visual hallucinations in reasoning paths cannot be effectively verified or rectified. Our solution starts with a key observation about visual contrast: when presented with a contrastive VQA pair, i.e., two visually similar images with synonymous questions, VLMs identify relevant visual cues more precisely. Motivated by this observation, we propose Visual Contrastive Self-Taught Reasoner (VC-STaR), a novel self-improving framework that leverages visual contrast to mitigate hallucinations in model-generated rationales. We collect a diverse suite of VQA datasets, curate contrastive pairs according to multi-modal similarity, and generate rationales using VC-STaR. Consequently, we obtain a new visual reasoning dataset, VisCoR-55K, which is then used to boost the reasoning capability of various VLMs through supervised finetuning. Extensive experiments show that VC-STaR not only outperforms existing self-improving approaches but also surpasses models finetuned on the SoTA visual reasoning datasets, demonstrating that the inherent contrastive ability of VLMs can bootstrap their own visual reasoning.

Section: INTRODUCTION
The scaling of large language models (LLM) has led to the emergence of reasoning capabilities (Wei et al., 2022a), making a transition from System 1 to System 2 (Kahneman, 2011) and enabling language models to tackle complex, multi-step problems (Wei et al., 2022b;Kojima et al., 2022). This emergent ability can be further enhanced by various techniques (Wang et al., 2023b;Li et al., 2023b;Hao et al., 2023;Gao et al., 2023;OpenAI, 2024b;Guo et al., 2025). Among them, self-improving approaches (Zelikman et al., 2022;Gulcehre et al., 2023;Madaan et al., 2023;Qu et al., 2024;Ma et al., 2025) form a prominent branch, mainly because they can be easily applied and extended without external reward models (Lu et al., 2024a), predefined step decomposition (Liu et al., 2025), or specially designed reasoning structures (Li et al., 2025).
However, it is infeasible to directly adapt such language-based self-improving methods to vision language models (VLMs) (Liu et al., 2023;Bai et al., 2025). Previous self-improving approaches focus on textual coherence and the quality of the final answer (Zelikman et al., 2022;Zhang et al., 2024a), while they are unable to verify or rectify the visual hallucinations that persist in current VLMs (Tong et al., 2024;Li et al., 2024). Even worse, they may get stuck in speculative reasoning that privileges textual priors over real visual evidence (Favero et al., 2024;Wu et al., 2025). We claim that the key problem for self-improving in VLMs is: how to rectify visual hallucinations in VLMs' reasoning paths for high-quality visual rationale generation.
Our solution is built upon an interesting observation: VLMs can see better during contrasting. As shown in Fig. 1a, the VLM generates a wrong rationale with visual hallucinations given a single visual question answering (VQA) sample. Instead, when presented with a contrastive VQA pair,
this section cite: ['b29', 'b34', 'b92', 'b16', 'b21', 'b88', 'b20', 'b54', 'b63', 'b53', 'b46', 'b39', 'b44', 'b3', 'b88', 'b71', 'b35', 'b14', 'b80']

Section: Naïve Answering
Think step by step to answer this question: what kind of trick is this skateboarder performing?
The skateboarder appears to be executing an ollie, one of the most fundamental tricks in skateboarding where they jump into mid-air with their board attached underneath them without using any handrails or other objects… ❌ 🤔 User 😵💫
this section cite: []

Section: VLM

this section cite: []

Section: Contrasting and Answering
Discriminate between two images and think step by step to answer two similar questions for two images respectively. For the first image: What kind of trick is this skateboarder performing? For the second image: What's the name of the skateboarding trick the man is doing? …The first image shows a skateboarder balancing with one foot while sliding along the edge of an elevated ramp using their back wheels (tail), indicating they might be executing or attempting what could resemble a "tail slide." … For both questions regarding identifying specific tricks based purely visually from these descriptions alone: 1) **First Image:** The trick appears more aligned towards 'Tail Slide', where balance and control over rear portion dominate. … ✅   (Zelikman et al., 2022) that enhances the quality of reasoning with hints (ground-truth answers), contrasting with hints can rectify more cases. The blocks along the x-axis mark initial VLM failures. The color of each block indicates the outcome of rectifying: green for success and gray for failure. Tested VLM is Qwen2.5VL-7B (Bai et al., 2025).
i.e., two similar images with synonymous questions (Setting C in sub-figure 1b), the model captures fine-grained visual evidence more accurately and rectifies the erroneous rationale. Statistics of this phenomenon on a group of failure cases are shown in Fig. 1b. Compared with the hints-only (Setting H in sub-figure 1b) self-improving (provide the model with the ground-truth answers), the hints and contrasting (Setting C&H in sub-figure 1b) setting not only prevents the model from making new errors but also leads to the rectification of its original hallucinations.
Motivated by this, we propose a new self-improving framework, Visual Contrastive Self-Taught Reasoner (VC-STaR). VC-STaR contains three steps: (1) think step by step and generate a coarse rationale; (2) compare visual queries in a contrastive VQA pair and provide a contrastive analysis;
(3) rethink and refine the coarse rationale via an LLM based on the contrastive analysis. In order to guarantee the scalability of VC-STaR, we also propose a task-agnostic contrastive VQA pair curation framework, which can be readily adapted to various VQA tasks, e.g., reasoning (Lu et al., 2021b), math (Gao et al., 2025), chart Liu et al. (2024a), and OCR (Yuan et al., 2022). Specifically, we curate the contrastive VQA pairs within individual datasets, based on the similarity of both images and questions. We utilize these contrastive VQA pairs to generate faithful rationales, resulting in a novel Visual Contrastive Reasoning dataset (VisCoR-55K) as illustrated in Fig. 2. Finetuning with VisCoR-55K enhances VLMs' visual reasoning capability.
VC-STaR achieves prominent results on a wide range of challenging benchmarks, including MMVP (Tong et al., 2024), HallusionBench (Guan et al., 2024), MathVista (Lu et al., 2024b), MathVision Wang et al. (2024), and MMStar (Chen et al., 2024c).
On the one hand, VC-STaR outperforms existing self-improving baselines. On the other hand, it exhibits a clear advantage over models trained on recently proposed reasoning datasets. The experimental results validate that visual reasoning capability of VLMs can be bootstrapped through the lens of contrast. 2 RELATED WORKS Reasoning in Language. Dual-system theory (Kahneman, 2011) illustrates two systems in human cognition: a fast, intuitive System 1 and a slow, deliberate System 2 which is akin to emergent General (42.7%) OCR (7.8%) Where are they located relative to each other? The dog is on the left side, and the cat is on the right side. The cat is positioned to the left of the dog. Please write out the expression of the formula in the image using LaTeX format. \cos \angle 5 = \frac { P E } { B P } = \frac { \frac { 1 } { 2 } x } { 4 -x } \cos \angle B = \frac { B E } { B D } = \frac { 4 } { 5 } 21 VQA Datasets 5 Categories Reasoning (25.0%) If you select a marble without looking, which color are you less likely to pick? The answer is blue. The answer is black. Math (15.7%) Find x. Assume that segments that appear to be tangent are tangent. The answer is 30. The answer is 9. Graph/Chart (8.8%) What is the smallest number of orange candies? The smallest number of orange candies is 11 orange candies. The smallest number of orange candies is 31 orange candies. reasoning capability of LLMs (Wei et al., 2022a). Consequently, reasoning enhancement (Wei et al., 2022b;Kojima et al., 2022) is considered a pathway to elevate LLMs' cognitive performance. One solution involves a reward model (Li et al., 2023b;Lu et al., 2024a), often coupled with Monte Carlo tree search (Hao et al., 2023;Zhang et al., 2024a), to discover optimal reasoning paths. However, this solution is constrained by the need of an auxiliary model and the requirement for reasoning step dividing (Liu et al., 2025). Another way employs macro reasoning actions (Gao et al., 2023;Khot et al., 2023;Yang et al., 2025a) to inject human prior knowledge, however, hand-crafted macro actions struggle to adapt to diverse reasoning scenarios. While reinforcement learning (Rafailov et al., 2023;Trung et al., 2024;Guo et al., 2025) has also attracted the attention, its success relies on the data format and the design of reward functions. Self-improving methods (Zhang et al., 2024a) offer a more scalable alternative, enabling LLMs to refine its own reasoning by constructing highquality reasoning data (Wang et al., 2023b), utilizing ground-truth answers as hints (Zelikman et al., 2022), or leveraging internal feedback (Qu et al., 2024). With fewer external constraints, selfimproving methods pave the way for more flexible and general language reasoners.
this section cite: ['b88', 'b3', 'b15', 'b86', 'b71', 'b19', 'b75', 'b34', 'b92', 'b46', 'b16', 'b31', 'b65', 'b72', 'b21', 'b88', 'b63']

Section: Reasoning in Vision.
Human reasoning is stimulated not only by textual input but also by visuallyrelated queries. Fostering the visual reasoning ability (Zhang et al., 2024c) for VLMs (Liu et al., 2023;Li et al., 2023a) is therefore a critical frontier topic. Early attempts often rely on external scaffolding like scene graphs (Mitra et al., 2024), macro actions (Xu et al., 2025;Dong et al., 2025), or bounding boxes highlighting key region in images (Shao et al., 2024). However, such approaches suffer from fundamental limitations: they are constrained by the data structure or tend to generate stereotyped reasoning paths. Despite these advances, the self-improving paradigm which has shown its effectiveness in text-only domain is underexplored for visual reasoning. The primary obstacle is the visual hallucinations embedded in reasoning paths cannot be easily rectified by existing textcentric self-improving frameworks (Zhang et al., 2024a;Zelikman et al., 2022;Qu et al., 2024). The proposed VC-STaR attempts to bridge this gap through the lens of contrast.
this section cite: ['b44', 'b57', 'b81', 'b11', 'b68', 'b88', 'b63']

Section: Power of Contrasting.
Contrasting has shown effectiveness in a wide range of machine learning topics. By comparing different views (Tian et al., 2020), e.g., data augmentations, of the same sample while distinguishing them from others (Wang & Isola, 2020), contrastive self-supervised learning methods (He et al., 2020;Grill et al., 2020;Radford et al., 2021;Liang et al., 2022;Pan et al., 2023) excel at learning potent feature representations. Explicitly cross-image contrasting is also studied under uni-modal setting (Pan et al., 2023;Ding et al., 2024;Chen et al., 2024a) and multi-modal setting (Park et al., 2019;Kim et al., 2021;Yao et al., 2022;Dunlap et al., 2024). Based on these advancements, VLMs are endowed with robust capabilities for multi-image comprehension and comparison (Alayrac et al., 2022;Bai et al., 2025;Chameleon, 2025;Lin et al., 2025). Some prior works have leveraged contrasting to create better instruction-tuning data (Jiao et al., 2025;Ma et al., 2024). However, how contrasting can help visual reasoning remains an open question. We observe that VLMs' inherent comparative ability can be repurposed to actively suppress its own visual hallucinations, bootstrapping their visual reasoning capability. This discovery offers a new perspective about the power of contrasting in reasoning.
this section cite: ['b70', 'b76', 'b23', 'b18', 'b64', 'b40', 'b50', 'b50', 'b10', 'b62', 'b33', 'b84', 'b13', 'b1', 'b3', 'b5', 'b42', 'b27', 'b52']

Section: VISUAL CONTRASTIVE SELF-TAUGHT REASONER (VC-STAR)
Let θ be a VLM and D = {(v i , q i , a i )} N i=1 be a set of visual question answering (VQA). The VQA set consists N triplets, where v i , q i , and a i represent the i-th image, question, and corresponding ground-truth answer, respectively. Following previous self-taught reasoners (Zelikman et al., 2022;Madaan et al., 2023), the original VQA dataset D can be enriched by generating a rationale r with θ for each triplet, which transforms D into a visual reasoning dataset R = {(v i , q i , a i , r i )} M i=1 . However, as mentioned in Sec. 1, rationale r i may be contaminated by visual hallucinations. Motivated by the observation illustrated in Fig. 1, VC-STaR aims to refine rationale r i into a more faithful one ri by contrasting the (v i , q i , a i ) with a contrastive VQA counterpart sample ( vi , qi , âi ) where q i is synonymous with qi and v i shares similar context with vi . The contrastive VQA pairs
P = { (v i , q i , a i ), ( vi , qi , âi ) } K
i=1 support the contrasting and rationale refining process. The contrastive VQA pairs are curated by searching ( vi , qi , âi ) for (v i , q i , a i ) within diverse data groups in D for different VQA tasks, ensuring the generalization of VC-STaR. The VC-STaR is designed to address two key challenges: (1) how to curate meaningful contrastive VQA pairs; (2) how to transfer the fine-grained discriminative ability from dual-image contrasting to refine the single-image reasoning. Sec. 3.1 elaborates on the pipeline for curating contrastive VQA pairs. Building upon this foundation, Sec. 3.2 introduces our contrasting and rethinking procedure which embeds the dual-image comparison into a new reasoning path, guided by an LLM, to produce a more faithful rationale. The refined rationales are then used to construct a new reasoning dataset R = {(v i , q i , a i , ri )} L i=1 , which we name the Visual Contrastive Reasoning dataset (VisCoR-55K). The VLM θ is updated to a new version θ with improved reasoning capability by finetuning on VisCoR-55K.
this section cite: ['b88', 'b54']

Section: CONTRASTIVE VQA PAIR CURATION
To ensure the generalization of VC-STaR, the contrastive VQA pair curation pipeline should be flexible enough across a wide spectrum of VQA tasks. For better contrasting, each contrastive VQA pair (v i , q i , a i ), ( vi , qi , âi ) should possess three key properties: (1) q i and qi are synonymous. This shared question acts as a semantic anchor, grounding the two images v i and vi at the same point in the semantic space. The images thus represent different manifestations of this anchor, providing a solid basis for contrasting; (2) v i and vi are visually similar. v i and vi should not be trivially distinct but exhibit visual similarity, creating a confusing contrasting. This visual proximity compels VLMs to engage in fine-grained contrasting to discriminate subtle differences; (3) q i is reasoning dependent. q i should be reasoning-provoking rather than one that can be solved by a straightforward answer. To achieve these requirements, as illustrated in Fig. 3, we propose a three-stage curation pipeline: Data Collection. We collect 21 VQA datasets spanning five categories: reasoning (Zhang et al., 2019;Kiela et al., 2020;Lu et al., 2021b), graph/chart (Kembhavi et al., 2016;Mathew et al., 2022;Masry et al., 2022;Tang et al., 2023;Lu et al., 2023;Liu et al., 2024a), math (Lu et al., 2021a;Cao & Xiao, 2022;Gao et al., 2025), general (Zhu et al., 2016;Johnson et al., 2017;Acharya et al., 2019;Schwenk et al., 2022;Wang et al., 2023a;Chen et al., 2024b), andOCR (ICDAR, 2019;Yuan et al., 2022;Zhang et al., 2024b). This broad collection enriches the diversity of our curated pairs, which ensures the generalization ability of the finetuned model.
Contrastive VQA Pair Hunting. In order to compute the similarity of VQA pairs, we first represent the question q i and the image v i by high-dimensional embeddings, denoted as e q
i and e v i respectively. We use GTE (Li et al., 2023c) text embeddings to represent the questions. In terms of image embedding, existing models fall into two types, i.e., vision-language contrastive learning approaches (Radford et al., 2021;Tschannen et al., 2025) while vision-only self-supervised learning methods (Zhang et al., 2023;Oquab et al., 2024). The former ones mainly capture at global semantic information, while the later ones are good at instance discrimination. Neither of them are
this section cite: ['b89', 'b32', 'b30', 'b56', 'b55', 'b69', 'b50', 'b4', 'b15', 'b98', 'b28', 'b0', 'b67', 'b86', 'b64', 'b73', 'b92', 'b60']

Section: Contrastive VQAs Think
Q: What is the person doing？ A: She is reading a book.
Q: What is she doing？ A: She is enjoying the scenery.
To answer this question, I am focusing on the woman's behavior and her interaction with the surroundings. I notice that she isn't actively engaging with anything; she's simply sitting on the bench. Therefore, the most direct and factual conclusion is that the woman is sitting on the bench.
Please think step by step to answer this question.
The first image shows a woman sitting on a bench in a park or garden. She's holding what looks like a book or some papers, so it seems she's reading. In contrast, the second image captures a woman on a bench by the water on a cloudy day. She's holding a red umbrella and looking out at the horizon. So, for questions regarding their actions: 1) **First Image** she is most likely reading. 2) **Second Image** she is simply enjoying the scenery.
Please discriminate clearly between provided images, then answer the quesLon for both images respecLvely.
this section cite: []

Section: Contrast Rethink
Please refine the rationale based on the contrastive analysis.
To answer what the woman is doing now, I am focusing on her behavior and her interaction with the surroundings. She is sitting on a bench in a park, and the key detail is what she's holding: it looks like a book or a booklet. If she were looking off into the distance or holding something else, I'd probably guess she was just enjoying the scenery. But all the visual content point to her interacting with the book in her hands. So, the most likely answer is that she's reading.  1, the contrastive analysis is then used to trigger a rethinking procedure, which refines the naive rationale into a more faithful one. This pipeline is designed to generate rationales for supervised finetuning. generic enough to adapt to the diverse domains. To tackle the dilemma, we build a versatile visual embedding model based on ID-based visual metric learning (Ypsilantis et al., 2024;An et al., 2023). Hunting for a counterpart ( vi , qi , âi ) is then performed dataset-by-dataset. A sample (v j , q j , a j ) is recalled as a valid counterpart if it satisfies: γ(e v i , e v j ) < ϕ v and γ(e q i , e q j ) < ϕ q , where γ(•, •) is the cosine distance, and ϕ v and ϕ q are pre-defined thresholds for visual and question similarity, respectively. Any sample that fails to meet both conditions is dropped.
this section cite: ['b85', 'b2']

Section: Difficulty-Based Data Sampling.
For the goal of developing visual reasoning capability, q i should be a difficult question requires reasoning rather than a straightforward one. We define the levels of difficulty based on the performance of VLM θ: (1) easy samples are with the simple q i which can be correctly answered by θ without any auxiliary help; (2) median samples are with q i which makes θ initially fails but succeeds when contrasting with ( vi , qi , âi ) based on provided hint a i (the C&H setting introduced in Fig. 1); (3) hard samples are the ones with q i that cannot be correctly addressed by θ even with the help of contrasting. We only keep median-difficult contrastive VQA pairs for the rationale generating.
this section cite: []

Section: CONTRASTING AND RETHINKING
Rationales in the reasoning dataset R = {(v i , q i , a i , r i )} M i=1 generated by the VLM θ itself include visual hallucinations. To achieve the goal:
R = {(v i , q i , a i , r i )} M i=1 → R = {(v i , q i , a i , ri )} M i=1 ,(1)
where ri is the rectified rationale, we use the contrastive VQA counterpart ( vi , qi , âi ) to provoke a rethinking action to refine r i into ri . As illustrated in Fig. 4, this pipeline includes three steps:
Thinking step. Following the design of Zelikman et al. (2022) to provide the VLM θ with groundtruth answer a i as hints, we prompt the VLM θ to generate the coarse rationale r i for the target VQA sample (v i , q i , a i ) as follows:
r
i = f (v i , q i , a i |θ, δ t ) ,(2)
where f is a inference process with a "thinking prompt" δ t . Details of δ t are in the Sec. A.3.
this section cite: ['b88']

Section: Contrasting step.
Asking the VLM θ to compare the target VQA sample (v i , q i , a i ) with its contrastive counterpart ( vi , qi , âi ) results in a contrastive analysis c i which may provide more faithful visual information:
c i = f ( (v i , q i , a i ), ( vi , qi , âi ) |θ, δ c ) ,(3)
where the δ c is the "contrasting prompt". When a i has the same meaning as âi , δ c requires summarizing the common patterns of v i and vi ; When a i is different from âi , δ c expects the analysis about the fine-grained differences between v i and vi . Details of δ c are in the Sec. A.3.
this section cite: []

Section: Rethinking step.
As demonstrated in Fig. 1, c i is more trustworthy than r i . Hence, we adopt a LLM ψ to transfer the information from c i to a new reasoning path according to r i :
ri = f (r i , c i |ψ, δ r ) ,(4)
where δ r is the "rethinking prompt" which asks the LLM ψ to rectify the visual hallucinations in r i according to the visual information from c i . δ r requires LLM ψ to respond like directly answering the question q i , details are in the Sec. A.3.
To ensure the quality of the R, we finalize the visual reasoning dataset by employing a text-matching post-processing to filter out samples that contain incorrect reasoning patterns. The final visual reasoning dataset contains 55K VQA samples with corresponding rationales, a.k.a., the VisCoR-55K.
4 EXPERIMENTS Section 4.1 details our experimental setup, including the supervised finetuning process and the benchmarks used to evaluate the effectiveness of the VC-STaR. In Section 4.2, we present a comprehensive performance comparison. As a self-improving method for visual reasoning, we benchmark VC-STaR against two primary groups: (1) other self-improving baselines adaptable to visual reasoning, and (2) models trained on off-the-shelf visual reasoning datasets. Finally, Section 4.3 provides in-depth ablation studies on designs of our method, including the contrastive VQA pair construction, the generalization on other base models, the difficulty sampling strategy, and the effect of the types of contrastive VQA counterpart.
this section cite: []

Section: SETUP
Implementation Details. Using the LLaMA-factory framework (Zheng et al., 2024), we finetune the model for 3 epochs via full-parameter supervised finetuning (SFT), with the vision tower's parameters frozen. The SFT utilizes a learning rate of 1e-5, a batch size of 256. The inference process of the finetuned model does not require such a contrastive pipeline illustrated in Fig. 4, and it follows the standard inference paradigm of VLMs. As for the curation of contrastive VQA pair, the question similarity threshold ϕ q is set to 0.15 and the visual similarity threshold ϕ v is set as 0.5 for datasets of general images. For the datasets including icon, geometry, chart or graph images, the visual similarity threshold ϕ v is set as 0.3. The LLM ψ used in the rethinking step of our rationale generation pipeline is the open-sourced Qwen2.5-72B.
Evaluation Benchmarks. We employ 6 benchmarks designed to assess its robustness against hallucination, mathematical reasoning, and general abilities. The MMVP (Tong et al., 2024) and Hallusion (Guan et al., 2024) benchmarks focus on visual hallucination, and the MathVista (Lu et al., 2024b) and MathVision (Wang et al., 2024) benchmarks are about the mathematical reasoning. The MMStar (Chen et al., 2024c) is a highly curated benchmark, composed of purified samples from multiple benchmarks, e.g., MMMU (Yue et al., 2024)
this section cite: ['b97', 'b71', 'b75', 'b87']

Section: MAIN RESULTS

this section cite: []

Section: Comparison with the base model.
To evaluate the effectiveness of our approach, we employ Qwen2.5VL-7B as the base model and adopt the "think step by step" prompt to enable chain-ofthought reasoning. We compare our method against this baseline, with results summarized in Table 1. VC-STaR demonstrates consistent performance gains across diverse challenging benchmarks, achieving an average improvement of 2.4%. Notably, it yields substantial improvements of 5.7% and 3.2% on MMVP and the Hallusion Benchmark, respectively, validating its efficacy in mitigating hallucinations within the reasoning process. Our approach also shows its enhanced reasoning capabilities on mathematical benchmark, i.e., MathVista and MathVision. Furthermore, the improvement on the MMStar and MME-RealWorld underscore the generalizability of the VC-STaR under varied challenging general-purpose scenarios.
For qualitative validation, Figure 5 provides visual comparisons that offer deeper insights. The visualizations reveal that our model excels at grounding its textual rationales in the corresponding visual evidence. This capability remains robust even when confronted with visually complex patterns, thereby effectively mitigating hallucinations.
Comparison with self-improving baselines. We reproduce three self-improving baselines and compare VC-STaR against them. Each baseline is applied to the Qwen2.5VL-7B base model and generates rationales on VisCoR-55K for finetuning, differing in their core improvement mechanism:
(1) STaR (Zelikman et al., 2022): Leverages ground-truth answers to regenerate rationales for incorrect predictions. (2) Verifier (Lu et al., 2024a): Filters out visually hallucinated rationales via a self-verification step (Zhang et al., 2025a) to ensure visual grounding. (3) Feedback (Qu et al., 2024): Refines rationales based on self-generated feedback in a recursive manner. Table 1 reveals a critical trade-off: existing self-improving methods boost performance on hallucination benchmarks at the expense of math and general capabilities. Our approach mitigates this pattern and achieves robust, consistent performance gains.
this section cite: ['b88', 'b63']

Section: Comparison with off-the-shelf visual reasoning datasets.
We also evaluate VC-STaR against base model finetuned on four off-the-shelf visual reasoning datasets. These datasets represent diverse strategies for rationale generation. For instance, Virgo (Du et al., 2025) makes the VLM think slowly
Question Qwen2.5VL-7B
The white sign in the picture has a red circle with a line through it, which is a universal symbol indicating "no parking." This sign is typically used to prohibit vehicles from parking in a specific area for a certain period of time.
The exact duration or conditions under which parking is prohibited would be specified by additional text on the sign, but the symbol itself clearly indicates that parking is not allowed.
There are two forks visible in the image, one on each plate with the cake slices.
Qwen2.5VL-7B "think step by step"
The white sign in the image has a red circle with a line through it, which is a universal symbol indicating "no" or "prohibited." Specifically, this sign is a "No Parking" sign. It means that parking is not allowed in the area where the sign is posted. This type of sign is typically used to prevent vehicles from blocking traffic or causing congestion in certain areas.
To determine how many forks are visible in the image, let's follow these steps: 1. Identify the objects: Look closely at the image and identify all the objects that could be forks. 2. Count the forks: -There is one fork placed on the napkin next to the slice of cake on the left plate. -There is another fork partially visible on the right side of the image, resting on the plate with the chocolate cake. 3. Verify the count: Double-check to ensure no additional forks are hidden or partially obscured. After following these steps, we can conclude that there are two forks visible in the image. How many forks are visible in the image? We highlight the key visual evidences with red boxes for clarity of visualization. More results are in Sec. A.4. w HQ-Edit w DOCCI w VisCoR 68 70 72 74 76 78 Performance 69.7 73.7 75.7 MMVP w HQ-Edit w DOCCI w VisCoR 52 54 56 58 Performance 56.9 55.9 56.3 Hallusion w HQ-Edit w DOCCI w VisCoR 59 60 61 62 Performance 59.8 61.2 62.4 MMStar w HQ-Edit w DOCCI w VisCoR 66 67 68 69 70 Performance 66.5 67.9 69.7 MathVista w HQ-Edit w DOCCI w VisCoR 22 23 24 25 26 Performance 24.3 25.1 25.3 MathVision w HQ-Edit w DOCCI w VisCoR 50 55 60 Performance 49.6 50.4 59.3 MME-RealWorld with purely textual rationales. In contrast, LLaVA-CoT (Xu et al., 2025) leverages hand-crafted templates filled by the powerful GPT-4o (OpenAI, 2024a). Other approaches first convert visual information into text; R1-Onevision (Yang et al., 2025b) generates rationales from image captions using the DeepSeek-R1 model (Guo et al., 2025), while Long Perceptual Thought (LPT) (Liao et al., 2025) extends this by using dense captions (Onoe et al., 2024) and keywords like "wait" to elicit more detailed outputs from a similar LLM. In our experiments, we directly finetune the base model on each of these datasets. Based on the results shown in Table 1, we can draw the following conclusion: (a) Enhancing visual reasoning with purely textual rationales from Virgo is ineffective. This strongly indicates that visual modality matters. (b) The model trained on LLaVA-CoT suffers limited improvement, which demonstrates that the hand-crafted template struggle to generalize across diverse VQA tasks. (c) The models trained on datasets generated by DeepSeek-R1 based on captions achieves notable improvements. However, the performance gap between them and ours highlights the clear advantage of our visually-native approach over relying on textual captions.
this section cite: ['b12', 'b81', 'b21', 'b41', 'b58']

Section: ANALYSIS
Can contrastive VQA pairs constructed in other ways? To answer this, we explore alternative strategies for curating contrastive VQA pairs. The first strategy is editing-based, utilizing the HQ-   Edit dataset (Hui et al., 2025). By prompting an LLM to create questions from editing instructions, we generate pairs where an original and an edited image yield different answers. The second strategy is caption-based, leveraging a dense caption dataset, i.e.DOCCI (Onoe et al., 2024). For this, we instruct an LLM to parse dense captions of visually similar images and generate a question that hinges on their subtle differences. For both strategies, we generate rationales for these newly created contrastive pairs using our proposed VC-STaR and finetune the Qwen2.5VL-7B. The results, presented in Fig. 6, lead to several observations: (a) VC-STaR is broadly effective, but performance is data-dependent. This is attributable to the biased data distribution of HQ-Edit and DOCCI, highlighting a key limitation of their curation scope. (b) VisCoR-55K includes contrastive pairs from a broader range of reasoning tasks, resulting in a more balanced performance.
this section cite: ['b25', 'b58']

Section: Does VC-STaR generalize to other base models?
We conduct experiments on Qwen2.5VL-3B and InternVL2.5-8B (Chen et al., 2025). Following the same self-improving procedure, we use VC-STaR to generate visual reasoning datasets from our VisCoR contrastive pairs, specifically for the two base models. We then finetune the Qwen2.5VL-3B and InternVL2.5-8B via the LLaMAfactory and SWIFT (Zhao et al., 2025), respectively. The results, presented in Table 2, demonstrate the model-agnostic effectiveness of our approach. These consistent and significant gains confirm that VC-STaR is a versatile and broadly applicable strategy for enhancing the visual reasoning ability.
What is the effect of easy samples on visual reasoning? Starting with our VisCoR-55K datasets, we incrementally add easy samples of two batches with 20K each. As illustrated in Table 3, we observe that the inclusion of easy samples is harmful. Specifically, when the number of easy samples increases, performance decreases. Therefore, we do not use the easy samples to avoid the potential "overthink" for straightforward problems.
How the contrastive VQA pairs of different types contribute? A contrastive VQA pair can be categorized as "positive" if both samples yield the same answer, and "negative" if their answers differ.
To investigate the respective contributions of these two types of counterparts to our method's performance, we conducted a controlled experiment on the GQA dataset (Hudson & Manning, 2019). The structured nature of GQA allows for the reliable curation of both positive and negative pairs via simple text matching. We applied VC-STaR to three distinct training sets: one generated from only positive contrastive pairs, one from only negative pairs, and a combined set including both. The results, detailed in Table 4, reveal a clear and significant trend. While both types of pairs are beneficial, negative counterparts are substantially more effective than positive ones, and their combination yields the optimal total gain, highlighting their complementary roles. We attribute the superior efficacy of negative counterparts to their ability to induce stronger semantic contrast. Accordingly, our approach incorporates both positive and negative pairs without restriction to achieve optimal gain.
this section cite: ['b9', 'b96', 'b24']

Section: CONCLUSION
We demonstrate that visual hallucination can be effectively mitigated through the lens of contrast, thereby enhancing visual reasoning. Based on the insight that VLMs can see better by contrast, we propose the VC-STaR. The VC-STaR refines hallucinatory reasoning paths through analysis over curated contrastive VQA pairs, which yields our high-quality VisCoR-55K. Finetuning on VisCoR-55K delivers a consistent performance gain across six benchmarks, significantly surpassing other self-improving baselines and models trained on state-of-the-art visual reasoning datasets. Looking forward, we hope our work will offer a new perspective on visual reasoning and inspire the exploration of novel contrast-driven training and inference paradigms. procedure, we initially curated a large pool of 240k raw contrastive VQA pairs. The distribution is visualized by the bar chart in Fig. 7, with the left y-axis indicating the number of samples.
L V I S T a ll y Q A V is u a l7 W I c o n Q A ( P a r t 1 ) C L E V R R A V E N I c o n Q A ( P a r t 2 ) T a b M W P H M E 1 0 0 K S R O I E G e o Q A + S T -V Q A G e o 1 7 0 K Q A V is T e x t G e o m e t r y 3 K L L a V A R H a t e f u lM e m e s A O K V Q A L R V U R
This initial pool of 240k pairs then undergoes a rigorous filtering and refinement pipeline. First, we apply the difficulty-based sampling strategy (as detailed in Sec. 3.1) to select only the median samples, which are most effective for enhancing the model's reasoning capabilities. The proportion of median samples varies significantly across datasets, and is illustrated by the line graph in Fig. 7 (plotted against the right y-axis). This critical filtering step narrows our collection down to 86k challenging contrastive pairs. Subsequently, we leverage the contrasting and rethinking pipeline to generate a high-quality rationale for each of these 86k samples. As a final quality control measure, we employ a text-matching-based post-processing step to automatically filter out any rationales containing unexpected or erroneous reasoning patterns. This process culminates in our final VisCoR-55K dataset, a collection of high-fidelity visual reasoning samples ready for finetuning. The pie charts in Fig. 7 provide a categorical overview of the data composition throughout this pipeline.
this section cite: []

Section: References
Ref_id:b0 Title: Tallyqa: Answering complex counting questions Year: (2019)
Ref_id:b1 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b2 Title: Unicom: Universal and compact representation learning for image retrieval Year: (2023)
Ref_id:b3 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b4 Title: An augmented benchmark dataset for geometric question answering through dual parallel text encoding Year: (2022)
Ref_id:b5 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2025)
Ref_id:b6 Title: Changemamba: Remote sensing change detection with spatiotemporal state space model Year: (2024)
Ref_id:b7 Title: Sharegpt4v: Improving large multi-modal models with better captions Year: (2024)
Ref_id:b8 Title: Are we on the right way for evaluating large visionlanguage models Year: (2024)
Ref_id:b9 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2025)
Ref_id:b10 Title: Joint spatiotemporal modeling for semantic change detection in remote sensing images Year: (2024)
Ref_id:b11 Title: Insight-v: Exploring long-chain visual reasoning with multimodal large language models Year: (2025)
Ref_id:b12 Title: A preliminary exploration on reproducing o1like mllm Year: (2025)
Ref_id:b13 Title: Describing differences in image sets with natural language Year: (2024)
Ref_id:b14 Title: Multi-modal hallucination control by visual information grounding Year: (2024)
Ref_id:b15 Title: G-llava: Solving geometric problem with multi-modal large language model Year: (2025)
Ref_id:b16 Title: Pal: Program-aided language models Year: (2023)
Ref_id:b17 Title: Structure-mapping: A theoretical framework for analogy Year: (1983)
Ref_id:b18 Title: Bootstrap your own latent -a new approach to self-supervised learning Year: (2020)
Ref_id:b19 Title: Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large visionlanguage models Year: (2024)
Ref_id:b20 Title: Reinforced self-training (rest) for language modeling Year: (2023)
Ref_id:b21 Title:  Year: (2025)
Ref_id:b22 Title: Reasoning with language model is planning with world model Year: (2023)
Ref_id:b23 Title: Momentum contrast for unsupervised visual representation learning Year: (2020)
Ref_id:b24 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b25 Title: HQ-edit: A high-quality dataset for instruction-based image editing Year: (2025)
Ref_id:b26 Title: Overview -icdar 2019 robust reading challenge on scanned receipts ocr and information extraction Year: (2019)
Ref_id:b27 Title: Img-diff: Contrastive data synthesis for multimodal large language models Year: (2025)
Ref_id:b28 Title: Clevr: A diagnostic dataset for compositional language and elementary visual reasoning Year: (2017)
Ref_id:b29 Title: Thinking, fast and slow Year: (2011)
Ref_id:b30 Title: A diagram is worth a dozen images Year: (2016)
Ref_id:b31 Title: Decomposed prompting: A modular approach for solving complex tasks Year: (2023)
Ref_id:b32 Title: Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes Year: (2020)
Ref_id:b33 Title: Viewpoint-agnostic change captioning with cycle consistency. 2021 ieee Year: (2021)
Ref_id:b34 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b35 Title: Naturalbench: Evaluating visionlanguage models on natural adversarial samples Year: (2024)
Ref_id:b36 Title: BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b37 Title: Making language models better reasoners with step-aware verifier Year: (2023)
Ref_id:b38 Title: Towards general text embeddings with multi-stage contrastive learning Year: (2023)
Ref_id:b39 Title: DeepSolution: Boosting complex engineering solution design via tree-based exploration and bi-point thinking Year: (2025)
Ref_id:b40 Title: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning Year: (2022)
Ref_id:b41 Title: Longperceptualthoughts: Distilling system-2 reasoning for system-1 perception Year: (2025)
Ref_id:b42 Title: Comparison visual instruction tuning Year: (2025)
Ref_id:b43 Title: Aligning large multi-modal model with robust instruction tuning Year: (2024)
Ref_id:b44 Title: Visual instruction tuning Year: (2023)
Ref_id:b45 Title: Mmbench: Is your multi-modal model an all-around player Year: (2024)
Ref_id:b46 Title: Adaptivestep: Automatically dividing reasoning step through model confidence. ICML Year: (2025)
Ref_id:b47 Title: Autopsv: Automated process-supervised verifier Year: (2024)
Ref_id:b48 Title: Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning Year: ()
Ref_id:b49 Title: Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning Year: (2021)
Ref_id:b50 Title: Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning Year: (2023)
Ref_id:b51 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: ()
Ref_id:b52 Title: C3 l: content correlated vision-language instruction tuning data generation via contrastive learning Year: (2024)
Ref_id:b53 Title: S 2 R: Teaching LLMs to self-verify and self-correct via reinforcement learning Year: (2025)
Ref_id:b54 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b55 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b56 Title:  Year: (2022)
Ref_id:b57 Title: Compositional chain-ofthought prompting for large multimodal models Year: (2024)
Ref_id:b58 Title: DOCCI: Descriptions of Connected and Contrasting Images Year: (2024)
Ref_id:b59 Title: Gpt-4o system card Year: (2024)
Ref_id:b60 Title: Learning robust visual features without supervision Year: (2024)
Ref_id:b61 Title: Find beauty in the rare: Contrastive composition feature clustering for nontrivial cropping box regression Year: (2023)
Ref_id:b62 Title: Robust change captioning Year: (2019)
Ref_id:b63 Title: Recursive introspection: Teaching language model agents how to self-improve Year: (2024)
Ref_id:b64 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b65 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b66 Title: Cognitive representations of semantic categories Year: (1975)
Ref_id:b67 Title: A-okvqa: A benchmark for visual question answering using world knowledge Year: (2022)
Ref_id:b68 Title: Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning Year: (2024)
Ref_id:b69 Title: Vistext: A benchmark for semantically rich chart captioning Year: (2023)
Ref_id:b70 Title: Contrastive multiview coding Year: (2020)
Ref_id:b71 Title: Eyes wide shut? exploring the visual shortcomings of multimodal llms Year: (2024)
Ref_id:b72 Title: ReFT: Reasoning with reinforced fine-tuning Year: (2024)
Ref_id:b73 Title: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features Year: (2025)
Ref_id:b74 Title: To see is to believe: Prompting gpt-4v for better visual instruction tuning, 2023a Year: ()
Ref_id:b75 Title: Measuring multimodal mathematical reasoning with MATH-vision dataset Year: (2024)
Ref_id:b76 Title: Understanding contrastive representation learning through alignment and uniformity on the hypersphere Year: (2020)
Ref_id:b77 Title: Self-consistency improves chain of thought reasoning in language models Year: ()
Ref_id:b78 Title:  Year: ()
Ref_id:b79 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b80 Title: Combating multimodal llm hallucination via bottom-up holistic reasoning. AAAI Year: (2025)
Ref_id:b81 Title: Llava-cot: Let vision language models reason step-by-step Year: (2025)
Ref_id:b82 Title: Supercorrect: Advancing small llm reasoning with thought template distillation and selfcorrection Year: ()
Ref_id:b83 Title: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization Year: (2025)
Ref_id:b84 Title: Image difference captioning with pre-training and contrastive learning Year: (2022)
Ref_id:b85 Title: Udon: Universal dynamic online distillation for generic image representations Year: (2024)
Ref_id:b86 Title: Syntaxaware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b87 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b88 Title: Star: Bootstrapping reasoning with reasoning Year: (2022)
Ref_id:b89 Title: Raven: A dataset for relational and analogical visual reasoning Year: (2019)
Ref_id:b90 Title: Rest-mcts*: Llm self-training via process reward guided tree search Year: (2024)
Ref_id:b91 Title: Incentivizing llms to self-verify their answers Year: (2025)
Ref_id:b92 Title: Detr with improved denoising anchor boxes for end-to-end object detection Year: (2023)
Ref_id:b93 Title: Llavar: Enhanced visual instruction tuning for text-rich image understanding Year: (2024)
Ref_id:b94 Title: Mmerealworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans Year: (2025)
Ref_id:b95 Title: Multimodal chain-of-thought reasoning in language models Year: (2024)
Ref_id:b96 Title: Swift: a scalable lightweight infrastructure for fine-tuning Year: (2025)
Ref_id:b97 Title: Unified efficient fine-tuning of 100+ language models Year: (2024)
Ref_id:b98 Title: Visual7w: Grounded question answering in images Year: (2016)
