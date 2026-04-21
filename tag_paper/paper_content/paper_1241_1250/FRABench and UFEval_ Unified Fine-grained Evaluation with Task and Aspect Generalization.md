Title: FRABENCH AND UFEVAL: UNIFIED FINE-GRAINED EVALUATION WITH TASK AND ASPECT GENERALIZA-TION
Abstract: Evaluating open-ended outputs of Multimodal Large Language Models has become a bottleneck as model capabilities, task diversity, and modality rapidly expand. Existing "MLLM-as-a-Judge" evaluators, though promising, remain constrained to specific tasks and aspects (i.e., specific evaluation criteria such as fluency for text and image quality for images). In this paper, we argue that, on one hand, based on the interconnected nature of criteria, learning specific aspects can generalize to unseen aspects; on the other hand, jointly learning to assess multiple visual criteria and tasks may foster a synergistic effect. To this end, we propose UFEval, the first unified fine-grained evaluator with task and aspect generalization for four evaluation tasks -Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. However, training such a unified evaluator is hindered by the lack of a large-scale, multi-modal, and aspectlevel resource. To address this gap, we introduce FRABench, a comprehensive fine-grained evaluation dataset. Specifically, (1) We first construct a hierarchical aspect taxonomy encompassing 112 distinct aspects across the aforementioned four tasks. (2) Based on this taxonomy, we create FRABench, comprising 60.4k pairwise samples with 325k evaluation labels obtained from a combination of human and GPT-4o annotations. (3) Finally, leveraging FRABench, we develop UFEval, a unified fine-grained evaluator. Experiments show that learning on specific aspects enables UFEval to generalize to unseen aspects, and joint learning to assess diverse visual tasks and aspects can lead to substantial mutual benefits.

Section: INTRODUCTION
As Multimodal Large Language Models (MLLMs) have shown amazing abilities in human-like question answering (Grattafiori et al., 2024), assessing the quality of their free-form outputs has become increasingly challenging. Automated evaluators, a.k.a. "MLLM-as-a-Judge" paradigm (Saha et al., 2025) have therefore received much research attention (Li et al., 2024c). Despite their progress, we posit two concerns: (1) Current evaluators (Liu et al., 2023b;Ke et al., 2023) are typically tailored to specific aspects, which limits their adaptability to unseen aspects. (2) They are also limited to specific tasks and modalities, which sharply constrains their scope of application. Table 1 shows the detailed comparison with existing evaluators. We intuitively argue that, on one hand, evaluation aspects are inherently interconnected (Fu et al., 2023). Specifically, aspects such as engagement, naturalness, and creativity are closely linked. Thus, similar semantics and evaluation standards can be transferred across diverse tasks. On the other hand, jointly learning to assess multiple visual aspects and tasks may foster synergistic effects (Wang et al., 2025b). For instance, learning object alignment in image captioning improves character consistency evaluation in multi-image scenarios, while progress in image understanding enhances image generation evaluation through better judgments of content quality and contextual appropriateness. This cross-aspect and cross-task synergy motivates the development of a unified, fine-grained evaluator for improved generalization and performance.
Table 1: Comparison of our UFEval with recent evaluators. NLG, IU, IG, and ITIG represent Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation, respectively. The number of aspects shown for each evaluator is comparable in granularity, with the scope covering their respective target domains. "-" indicates that the number of supported aspects is not explicitly specified.
this section cite: ['b16', 'b56', 'b28', 'b11']

Section: Method
Task Modalty Aspect Generalizable NLG IU IG ITIG Text Image AUTO-J (Li et al., 2023b) ✓ ✗ ✗ ✗ ✓ ✗ 332 ✗ X-Eval (Liu et al., 2023b) ✓ ✗ ✗ ✗ ✓ ✗ 27 ✓ Prometheus 2 (Kim et al., 2024a) ✓ ✗ ✗ ✗ ✓ ✗ -✗ Themis (Hu et al., 2024b) ✓ ✗ ✗ ✗ ✓ ✗ 50 ✓ ImageReward (Xu et al., 2023a) ✗ ✗ ✓ ✗ ✗ ✓ 3 ✗ VisionReward (Xu et al., 2024) ✗ ✗ ✓ ✗ ✗ ✓ 37 ✗ LLaVA-Critic (Xiong et al., 2024
) ✗ ✓ ✗ ✗ ✓ ✗ - ✗ UFEval (ours) ✓ ✓ ✓ ✓ ✓ ✓ 112 ✓
To this end, we propose UFEval, the first unified fine-grained evaluator with task and aspect generalization for four evaluation tasks (i.e., NLG, IU, IG, and ITIG) across 28 sub-tasks. However, training such a unified evaluator requires large-scale, multi-modal, aspect-level evaluation datasets, which are currently unavailable. Therefore, we develop the Fine-grained Aspect Benchmark (FRABench) to address this gap. Specifically, we first conduct a survey on the four tasks to identify key evaluation aspects. We then manually organize, extend, and redefine 112 distinct aspects with hierarchical relations as a universal evaluation taxonomy-aspect tree. Using this aspect tree, we construct FRABench with comprehensive aspect coverage. The aspect tree guides us in selecting multiple relevant aspects for each of the 60.4k pairwise responses, for which we obtain evaluation labels through a hybrid approach combining available human annotations with GPT-4o-assisted completions, ensuring cost efficiency. In total, this process yields 325k evaluation labels. Using FRABench, we develop UFEval, a unified and fine-grained evaluator.
Our experiments show that UFEval exhibits excellent evaluation quality and aspect generalization capabilities. This is attributed to learning multiple aspects and tasks jointly, which yields significant mutual enhancement, as well as to fully leveraging the interconnections between learning and unlearning aspects. By conducting ablation experiments across diverse aspects and task baselines, we observe a gradual improvement in evaluation quality, thereby validating our hypothesis. Additionally, we demonstrate the value of using UFEval for preference data generation to align the outputs of models with human preferences via direct preference optimization (DPO). These results validate the effectiveness of the FRABench as a valuable resource for training unified evaluators.
In summary, our contributions are as follows: (1) We construct FRABench, a large-scale multimodal aspect-level evaluation dataset to train and test evaluators. (2) Upon FRABench, we develop UFEval, the first unified fine-grained evaluator for multiple tasks assessment with task and aspect generalization.
(3) Our experiments show that inter-aspect correlations enable generalizable capability, and learning to assess multiple visual tasks and aspects jointly leads to a synergistic improvement in evaluation performance, while also demonstrating UFEval's value for preference alignment.
2 RELATED WORK
this section cite: ['b75', 'b73']

Section: COARSE-GRAINED SPECIFIC-TASK EVALUATION
Previously, coarse-grained evaluators, which produce overall judgments based on one or several aspects, have been widely explored across various evaluation tasks (Zhu et al., 2023;Xu et al., 2023b;Ye et al., 2023;Liu et al., 2024b;Wang et al., 2023a;Jiang et al., 2023). For instance, in NLG, Wang et al. (2023b) first introduces PandaLM, a fine-tuned LLM designed to evaluate pairwise texts based on several aspects. Similarly, Li et al. (2023b) proposes Auto-J, an evaluator capable of handling a broader range of tasks and aspects, supporting both pointwise and pairwise evaluation settings.
In image generation evaluation, Xu et al. (2023a) develops ImageReward, an evaluator trained on a large-scale human annotation dataset. Using alignment and fidelity as reference dimensions, ImageReward provides an overall assessment. Despite showing good overall performance, coarse-grained evaluations lack the granularity needed to diagnose specific model deficiencies and often introduce aspect bias into the evaluation process.
this section cite: ['b72', 'b79', 'b27']

Section: FINE-GRAINED SPECIFIC-TASK EVALUATION
To address the limitations of coarse-grained evaluation, recent studies have shifted toward fine-grained evaluation (Ye et al., 2023;Kim et al., 2024b;Ke et al., 2023;Kim et al., 2023;Li et al., 2023d;Ying et al., 2024;Bai et al., 2023), where MLLMs are fine-tuned on multi-aspect datasets to produce aspect-specific judgments. For example, in NLG, Hu et al. (2024b) proposes Themis, an LLM trained on the GPT-4 annotated NLG-Eval corpus. In IU, LLaVA-Critic (Xiong et al., 2024) is the first fine-grained evaluator integrating diverse criteria, showing strong correlation with GPT-4o. For IG, Xu et al. (2024) introduces VisionReward, a VQA-based evaluator that assesses image quality across fine-grained aspects. While these methods improve over coarse-grained evaluation by targeting specific aspects, they struggle with scalability across different tasks and aspects. Liu et al. (2023b) further investigates aspect generalization through X-Eval, a two-stage learning framework that incorporates auxiliary aspects and demonstrates generalization in NLG. However, its reliance on predefined reference aspects and lack of open-source implementation restricts its reproducibility.
this section cite: ['b79', 'b28', 'b29', 'b80', 'b1', 'b73', 'b75']

Section: METHODOLOGY
To develop UFEval, we need to construct a large-scale, multi-modal, and aspect-level evaluation dataset. However, existing datasets predominantly focus on overall quality assessment rather than fine-grained aspect evaluation, limiting the development of such evaluators. To address this gap, we construct FRABench through two main steps: (1) Evaluation Aspect Construction, and (2) Fine-grained Evaluation Dataset Construction. The following sections detail these two steps.
this section cite: []

Section: EVALUATION ASPECT CONSTRUCTION

this section cite: []

Section: ASPECT COLLECTION AND EXTENSION.
To fully leverage existing aspects, we first collect 28 sub-tasks under the four types of tasks (i.e., NLG, IU, IG, and ITIG). These sub-tasks span all six combinations of input types (text and textwith-image) and output types (text, image, and text-with-image), ensuring comprehensive coverage across multimodal tasks. Subsequently, we collect literature related to each sub-task to gather relevant aspects and, where available, their definitions. Moreover, we define aspects without available definitions according to their practical meaning.
Beyond collecting existing aspects, we extend aspects for the ITIG, which lacks sufficient aspects. To this end, we apply a cross-task transfer by identifying analogous sub-tasks from other categories and adapting their aspects. For example, both story generation (NLG) and visual story completion (ITIG) involve narrative creation, enabling aspects like engagingness to be adaptedfoot_0 . Ultimately, we obtain 112 different aspects, with detailed information and sources listed in Appendix A.
this section cite: []

Section: ASPECT TAXONOMY CONSTRUCTION.
To facilitate aspect selection within the dataset construction and evaluation pipeline, we organize the collected aspects into an aspect tree serving as a standardized taxonomy. We first use the "overall" aspect as the root node of the aspect tree, and then divide the remaining aspects into two subtrees based on their task independence and task specificity:
• Universal Aspects (UAs): Aspects in this subtree exhibit task independence, as they primarily focus on assessing the quality of the model's output. They are often modality-specific (for example, fluency for text or fidelity for image) and serve as fundamental quality aspects across tasks.
• Task-specific Aspects (TAs): Aspects in this subtree exhibit task dependence, as they primarily focus on task completion rather than output quality. They are typically closely tied to the task type. For example, engagingness for story generation or accuracy for mathematical reasoning.
Figure 1: An illustration of the evaluation pipeline. The pipeline consists of two steps: (1) Aspect Selection: First, appropriate aspects are selected from the TAs and UAs Trees based on task property and output modality. As illustrated, the question focuses on "cat image next to the orange dog picture", while the image shows no cat adjacent to the orange dog. This discrepancy enables evaluation of the model's ability to avoid hallucinations, specifically its performance on Context Inconsistency. Second, given the text output modality, aspects are selected from the Text Branch in the UAs Tree. (2) Evaluating: UFEval generates feedback and scores based on the input content and selected aspects.
For hierarchical construction within the two subtrees, we prioritized aspect trees from peer-reviewed literature to ensure consistency and robustness. Specifically, we adopted the following structures: Readability (Hu et al., 2024a), Bias (DeAlcala et al., 2023), Instruction Following (Zeng et al., 2023), Hallucination (Huang et al., 2025), Alignment (Huang et al., 2023), Complement (Vempala & Preoţiuc-Pietro, 2019), Image Coherence (Liu et al., 2024a), Image Quality (Zhang et al., 2025), Semantic Consistency (Chen et al., 2019) and Toxicity (Gehman et al., 2020).
For the remaining aspects without an established hierarchical structure, we handle them as follows:
• Bidirectional Matching Strategy: First, we check whether the remaining aspect's name appears within the definition of the existing tree's root node. If a match is found, we recursively traverse the child nodes to identify the most specific insertion point and append the aspect as a child of the last matched node. Conversely, if no downward match is found, we verify whether the root node's name appears within the definition of the remaining aspect. In this case, the aspect implies a broader concept and is inserted as the parent of the current root. This applies to the following aspects: Clarity in Readability, Semantic Consistency in Image Coherence, and Integration, Complement, and Alignment in the Text-Image Relationship.
• Creation of new root nodes: For aspects that do not match any existing tree, we established them as independent root nodes to minimize subjective bias and avoid misleading categorizations. These nodes correspond to all green peer-level nodes within the TAs tree in Figure 1.
Tasks Dataset Sub-Tasks Split Size Public NLG ■ Auto-J Summarization, Creative Writing, Rewriting, etc. Train / FRA-ID 2.5k / 0.6k Question Generation, Title Generation, etc. FRA-OOD 0.6k IU ■ VLFeedback Image Captioning, Text-rich Understanding, etc. Train / FRA-ID 14.5k / 2.4k Medical VQA, Academic VQA, etc. FRA-OOD 1.2k IG ■ GenAI-Bench Text-to-Image Generation FRA-OOD 1.4k ■ ImageRewardDB Text-to-Image Generation Train / FRA-ID 6k / 1.5k Generated IU ■ TextVQA Text Reasoning Train / FRA-ID 2.1k / 0.4 k ■ ChartVQA Chart Reasoning Train / FRA-ID 2.1k / 0.4 k ■ InfographicsVQA Graph Reasoning Train / FRA-ID 2.1k / 0.4 k IG ■ MagicBrush Image Editing Train / FRA-ID 3.5 k / 0.5 k ■ COCO Text-to-Image Generation Train / FRA-ID 4.8k / 1k ITIG ■ VIST Visual Story Completion Train / FRA-ID 5.5k / 0.7k ■ wikiHow Multimodal Script Generation Train / FRA-ID 6.4k / 0.6k ■ InterleavedBench Storytelling Generation, Activitynet Continuation FRA-OOD 90 Based on aspect trees, the fine-grained evaluation process can be streamlined, as illustrated in Figure 1.
this section cite: ['b8', 'b82', 'b26', 'b25', 'b64', 'b86', 'b7']

Section: FINE-GRAINED EVALUATION DATASET CONSTRUCTION
Based on the aspect tree, we can construct fine-grained evaluation datasets for training and testing UFEval. Specifically, we construct FRABench-a large-scale, multi-modal, and aspect-level evaluation dataset. Following prior studies (Kim et al., 2024a;Ye et al., 2024) that demonstrate pointwise scoring is more susceptible to contextual bias and available for reward model training, we adopt a pairwise comparison evaluation method in FRABench.
this section cite: ['b78']

Section: FRABENCH CONSTRUCTION.
We first collect questions from the datasets corresponding to the 28 sub-tasks we gathered, and generate pairwise samples. Specifically, we first obtain 29.3K response pairs from public datasets and generate the remaining 30.1K pairs using different MLLMs (Details regarding the MLLMs used can be found in Appendix B.1). After obtaining the queries and response pairs, we assign aspects from our aspects tree. As a result, each pairwise sample is annotated with an average of 8 UAs and 3 TAs (Detailed information on aspect assignment is provided in Appendix B.4). We then generate evaluation labels via two approaches: (1) Human annotations: directly using three-aspect human-annotated scores from ImageRewardDB (Xu et al., 2023a), supplemented with feedback generated by GPT-4o; (2) GPT-4o annotations: due to the lack of human annotations for most aspects, we use GPT-4o to generate evaluation labels for all other pairwise samples. During the GPT-4o annotation process, we observed that GPT frequently incorporates response correctness into its assessment of UAs. To address this, we provide only the response, excluding the query, when evaluating UAs. All templates are provided in Appendix G. Moreover, to mitigate position bias (Ye et al., 2024), we balance the number of samples where response 1 is preferred over response 2 and vice versa by reversing the response positions in half of the surplus samples from the majority class and re-annotating them accordingly (Appendix B.2 presents detailed analysis about position bias of FRABench). Finally, we generate 325k fine-grained evaluation labels.
this section cite: ['b78']

Section: DATASET PARTITIONING AND EVALUATOR TRAINING.
Due to the lack of a large-scale aspect-level benchmark to verify UFEval's aspect generalizable capability and support our arguments, we can only divide FRABench into training, In-Domain test set (FRA-ID), and Out-of-Domain test set (FRA-OOD). The training and FRA-ID consist of 18 randomly selected sub-tasks from four tasks, covering 22 UAs and 35 TAs, while FRA-OOD contains 10 unseen sub-tasks with 28 seen UAs and 27 unseen TAs. Finally, the training set contains 255.4K samples, FRA-ID contains 45.2K samples, and FRA-OOD contains 24.4K samples. The statistics of FRABench are presented in Figure 2, with more detailed analysis provided in Appendix B.
An evaluator's main job is to judge model responses like humans would, so their effectiveness depends on how closely their judgments match human opinions. To measure this, we create humanannotated evaluation datasets. Specifically, we extract partial samples from FRA-ID and FRA- Table 2: The results correspond to task and aspect generalization evaluation. Average accuracy serves as the evaluation metric. Bold and underline indicate the first and second best results, respectively. Method Task Generalization Evaluation Aspect Generalization Evaluation FRA-OOD (GPT4o) FRA-OOD-H (Human) FRA-OOD (GPT4o) FRA-OOD-H (Human) NLG IU IG ITIG NLG IU IG ITIG NLG IU IG ITIG NLG IU IG ITIG GPT-4o ----84.0 82.1 72.3 93.1 ----83.2 82.1 74.2 93.1 Claude-3.5 74.6 85.8 72.5 75.6 83.0 76.5 63.1 91.0 84.1 84.3 65.6 85.0 82.6 76.5 65.1 91.0 Qwen2VL-72B 70.2 82.4 65.8 60.0 78.3 75.3 48.6 83.7 76.3 81.5 65.6 85.0 77.3 75.3 53.8 83.7 Qwen2VL-7B 50.4 65.9 61.4 43.4 50.9 65.9 40.9 44.3 54.5 69.1 37.7 69.1 49.1 65.9 46.0 44.3 Themis 56.7 ---58.9 ---55.0 ---58.8 ---LLaVA-Critic -52.2 ---76.2 ---80.8 ---76.2 --Ours 81.7 90.4 69.0 83.1 79.0 80.9 62.1 90.6 83.0 86.3 62.9 89.3 78.3 80.9 66.1 90.6
OOD, and recruited three humans with master's degrees for annotation. Finally, we retain 6.9K in-domain evaluation samples (FRA-ID-H) and 6.0K out-of-domain evaluation samples (FRA-OOD-H), respectively. Details of the human annotation process and annotation consistency are provided in Appendix E. When training UFEval, we use SFT to fine-tune Qwen2-VL-7B-Instruct on the training set. The detailed training configurations and methods are presented in Appendix C.1.
this section cite: []

Section: EXPERIMENTS
To assess UFEval's generalizability from inter-aspect correlations, we first conduct evaluations on previously unseen tasks and aspects, termed Out-of-Domain Evaluation. Next, we evaluate UFEval as MLLM-as-a-Judge using public benchmarks. We further explore the effectiveness of Multi-aspect and Multi-task Assessment Learning for evaluators. Lastly, we validate UFEval's application in generating preference data for DPO-based alignment. In the following sections, we present our experimental setup and results (In-Domain Evaluation and other experimental results are in Appendix D).
this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: BENCHMARKS.
(1) Out-of-Domain Evaluation: we use FRA-OOD and FRA-OOD-H to validate. (2) Evaluation as MLLM-as-a-Judge: we select public benchmarks across three tasks: For NLG, we use MT-  Bench (Zheng et al., 2023), SummEval (Fabbri et al., 2021), and MANS (Guan et al., 2021). Since SummEval and MANS only provide scores, we generate pairwise samples for evaluation through sample pairing. For IU, we use WildVision (Lu et al., 2024), MLLM-as-a-Judge (Chen et al., 2024), and VLRewardBench (Li et al., 2024e). For IG, GenAI-Bench (Li et al., 2024a), Winoground (Thrush et al., 2022), and Pick-a-Pic (Kirstain et al., 2023) are selected. To ensure fair comparison, we carefully check that the training set contains no overlapping samples with the benchmarks. (3) Multi-Aspect and Multi-Task Assessment Learning: We use FRA-ID to investigate the multi-aspect learning and benchmarks above to verify multi-task synergy. (4) Preference Alignment Comparison:
We leverage UFEval for image generation and understanding model alignment. For image generation, we generate images using captions from the HPDv2 (Wu et al., 2023). For image understanding, we use MMHal (Sun et al., 2023), LLaVABen.Wild (Li et al., 2024b), and LLaVABen (Liu et al., 2023a). Detailed descriptions and usage of each benchmark are in Appendix C.2.
this section cite: ['b88', 'b10', 'b17', 'b50', 'b5', 'b63', 'b32', 'b72', 'b61']

Section: BASELINES.
We experiment with representative models from two categories: (1) Prompting Models: GPT-4o, Claude-3.5, Qwen3-VL-8B (Qwen3VL-8B), Qwen2-VL-72B-Instruct (Qwen2VL-72B), and Qwen2-VL-7B-Instruct (Qwen2VL-7B), implemented using the same prompts as UFEval.
(2) Fine-tuned Evaluators: Themis-8B (Themis) (Hu et al., 2024b) , Auto-J (Li et al., 2023b), Prometheus 2 (Kim et al., 2024a), LLaVA-Critic-7B (LLaVA-Critic) (Xiong et al., 2024), ImageReward (Xu et al., 2023a), VisionReward (Xu et al., 2024), Q-Eval (Zhang et al., 2025) and CIGEval (Wang et al., 2025a). Notably, ImageReward and VisionReward use different evaluation paradigms from ours. Auto-J cannot assess specific aspects, and Prometheus 2 requires reference answers. Therefore, we use these methods only as baselines in the MLLM-as-a-Judge evaluation.
this section cite: ['b73', 'b75', 'b86']

Section: OUT-OF-DOMAIN EVALUATION

this section cite: []

Section: Tasks Generalization Evaluation.
For task generalization, we employ the seen UAs and unseen tasks from UFEval during testing to control for test variables. The results are shown in the regions outside the black sectors in Figure 3 foot_1 . Detailed scores for each aspect are reported in Appendix D.4. UFEval demonstrates strong alignment with both GPT-4o and human annotators on all tasks. In the "Tasks Generalization Evaluation" column of Table 2, UFEval achieves overall average accuracies of 85% and 83% on FRA-OOD and FRA-OOD-H, respectively, and outperforms both Themis and LLaVA-Critic on all tasks. Supporting the effectiveness of aspect-level evaluation for cross-task generalization. We also provide several good and bad cases in Appendix H.
this section cite: []

Section: Aspects Generalization Evaluation.
We use the unseen aspects of unseen tasks in FRA-OOD and FRA-OOD-H to evaluate UFEval's generalization to aspects. Specifically, this evaluation encompasses two distinct dimensions: (1) Contextual generalization, which validates whether similar semantics and standards can be transferred across diverse tasks (evaluated using 12 aspects); and (2) Novel aspect generalization, which assesses the capability to handle concepts semantically distinct from the training set (evaluated using 15 aspects). A detailed breakdown of these aspects is provided in Table 11. Moreover, to support the validity of aspect generalization evaluation, we provide ROUGE_L (Wang et al., 2022) tests on unseen aspects in Appendix B.3.
The results are shown in the black sectors of Figure 3. In terms of overall coverage, UFEval outperforms both Themis and LLaVA-Critic, the state-of-the-art evaluators in their respective domains.
In the right column of Table 2, UFEval continues to demonstrate strong performance on the NLG, IU, IG and ITIG, with high consistency with both GPT-4o and human annotators, achieving overall accuracies of 86.2% and 83.2%, respectively. These results show that even without exposure to unseen aspects during training, UFEval remains effective in evaluation. This is largely due to the naturally related nature of aspects, which enables the transfer of similar meanings and standards.
Additionally, we sample three types of aspects for aspect-level analysis (Figure 4) from four test sets based on UFEval's performance: (1) Weak aspects, where the best model outperforms UFEval by over 20%;
(2) Universally poor aspects, where all evaluators score very low; and (3) Strong aspects, where UFEval excels. Only two weak aspects occur in all test sets, and both are included; representative subsets are selected for the other categories. Regarding the weak aspects, as shown in Figure 4c, UFEval performs poorly on the harmfulness of IG in FRA-ID, mainly because it tends to classify shadowy or gloomy images as harmful due to heightened sensitivity to elements that may evoke psychological discomfort. In Figure 4b, all evaluators perform poorly on fidelity, which requires recognizing fine-grained object features (e.g., well-formed human facial features) that current models, including GPT-4o, often fail to detect (see Appendix H for more representative samples).
this section cite: ['b71']

Section: EVALUATION AS MLLM-AS-A-JUDGE
For NLG Evaluations. The experimental results, shown in Table 3, demonstrate the effectiveness of UFEval in NLG tasks. Specifically, in SummEval and MT-Bench, UFEval outperforms most baselines, with Claude being the sole exception. In the MANA benchmark, UFEval achieves the best performance, with an accuracy of 69.3%. These results highlight UFEval's strong capability for text understanding evaluation compared to domain-specific models in this field.
this section cite: []

Section: For IU Evaluations.
The results are presented in Table 4. Additionally, we assess model-level consistency on WildVision using Elo ratings and Kendall's Tau (τ ) to compare model rankings (Xiong et al., 2024). The findings demonstrate that UFEval consistently outperforms LLaVA-Critic across all benchmarks. LLaVA-Critic focuses on respective datasets training, whereas UFEval uses joint learning across multiple tasks, which leads to relatively superior performance. For more comprehensive MLLM-as-a-Judge results, please refer to Table 49.
For IG Evaluations. The experimental results are shown in Table 5. Our evaluator outperforms ImageReward in all benchmarks, achieving accuracies of 65.5%, 65.7%, and 57.3%, respectively.
Even compared to the state-of-the-art image evaluator VisionReward, the performance gap is minimal-only 0.9% lower on GenAI-Bench and 0.3% lower on Winogrounded. These results demonstrate the promising capability of UFEval in the image generation evaluation task.
this section cite: ['b73']

Section: MULTI-ASPECT AND MULTI-TASK ASSESSMENT LEARNING
The construction of UFEval is based on our intuitive argument: jointly learning to assess multiple visual aspects and tasks may foster a synergistic effect. Therefore, we explore the effectiveness of multi-aspect and multi-task learning on the evaluators, respectively. Specifically, we experiment with various training data configurations to train the model, analyzing the influence of multi-aspect and multi-task learning. For instance, for IU tasks, we design three training configurations to investigate the impact of multi-aspect and multi-task training: (1) learning solely on IU aspect-level assessment (w /IU), (2) jointly learning IU and ITIG aspect-level assessment (w /IU+ITIG), and (3) jointly learning IU and IG aspect-level assessment (w /IU+IG).
this section cite: []

Section: Multi-task Assessment Learning.
The results are presented in Table 4 and 5. Our results demonstrate that multi-task learning outperforms single-task training in enhancing evaluator performance. For instance, as shown in Table 4, when evaluating IU tasks, models trained jointly on both IU and IG tasks achieve superior overall accuracy compared to those trained exclusively on IU. These findings underscore the significant advantages of exploiting shared representations and complementary knowledge across diverse visual tasks, ultimately yielding a more high-performing evaluator.
this section cite: []

Section: Multi-aspect Assessment Learning.
The results presented in Figure 5 show that learning with multi-aspect from different tasks improves the UFEval's performance across most aspects. For example, in the IG task, incorporating relevant aspects from the IU, such as alignment in caption generation and recognition accuracy, enhances the evaluation of aspects like object alignment in IG. Similar improvements are observed in IU and ITIG. These results highlight the benefits of leveraging multi-aspect data to enrich the evaluator's understanding and shared aspect knowledge.
this section cite: []

Section: PREFERENCE ALIGNMENT COMPARISON
To validate the effectiveness of UFEval to generate preference data across both IU and IG tasks, we employ it to construct training data for DPO-based model alignment. The application principle of DPO for model alignment in both IU and IG tasks is comprehensively described in Appendix F.
For IU Tasks. Building upon UFEval, we use DPO to improve the image understanding capabilities of LLaVA-Next-7B (Li et al., 2024d). For fair comparison, both UFEval and the LLaVA-Critic ues identical image-question pairs sourced from RLHF-V (Yu et al., 2024) and LLaVA-RLHF (Sun et al., 2023) to construct preference data. Ultimately, we construct 15k preference samples. We then train LLaVA-Next-7B on 8 A100 GPUs using a batch size of 2, gradient accumulation steps of 2, a learning rate of 5 × 10 -7 , and set β u to 0.1. The results are shown in Table 6, UFEval consistently surpasses LLaVA-Critic across all evaluated benchmarks. Notably, it achieves a 2.4% improvement on LLaVABench.Wild, underscoring its enhanced effectiveness in visual understanding tasks.
For IG Tasks. Based on UFEval, we apply DPO to SDXL-Turbo (Podell et al., 2023), a conditional diffusion model, to better align its outputs with human preferences without explicit reward modeling. We extract prompt and two corresponding images from Pick-a-Pic and use UFEval to construct training preference data. In total, we construct 14k preference samples for DPO image generation.
We then train SDXL-Turbo on 8 A100 GPUs using β g = 5000 with a batch size of 32 for three epochs. HPSv2 (Wu et al., 2023), ImageReward, and VisionReward are used for quality assessment.
The results are shown in Table 7, training on the constructed data using UFEval achieves better performance compared to directly training on the original dataset. The qualitative comparison results are shown in Appendix I. This demonstrates the effectiveness of our approach in refining preference data for improved model alignment in image generation tasks.
this section cite: ['b81', 'b61', 'b55', 'b72']

Section: CONCLUSION, LIMITATION AND FUTURE WORK
This paper proposes UFEval, the first unified fine-grained evaluator with task and aspect generalization. Specifically, we start by building a comprehensive aspect tree that leads to the creation of FRABench, a large-scale, multi-modal, and aspect-level evaluation dataset. We then fine-tune an MLLM on FRABench to develop UFEval. Our experimental results demonstrate that joint learning across diverse visual tasks and aspects yields significant mutual benefits and generalization capabilities. We also leverage UFEval to automatically construct high-quality preference pair datasets for DPO training to align models' outputs. These results validate the effectiveness of the FRABench as a valuable resource for training unified evaluators. However, compared to the other three tasks, UFEval has relatively limited performance in IG tasks. This may primarily be due to existing LMMs' insufficient active visual semantic understanding. In future work, we plan to incorporate video understanding and generation tasks into our evaluation system and add their corresponding aspects to the aspect tree.
Table 8: The detailed statistical information of the sub-tasks, which is not shown in Figure 2.
Task Sub-Task Split Size Task Sub-Task Split Size NLG (Public) Summarization Train 218 IU (Public) Detailed Image Captioning Train 2.1k Test 54 Test 0.4k Creative Writing Train 575 Robustness-oriented Instructions Train 4k Test 120 Test 0.4k Rewriting Train 194 Medical Image Understanding Train 2.1k Test 26 Test 0.4k General Communication Train 1080 Text-rich Understanding Train 2.1k Test 263 Test 0.4k Functional Writing Train 433 General Visual Conversation Train 2.1k Test 147 Test 0.4k Question Generation Test 132 Simple Image Captioning Train 2.1k Test 0.4k Title Generation Test 38 Embodied Decision-making Test 0.4k Keywords Extraction Test 80 Medical VQA Test 0.4k Data Analysis Test 170 Academic VQA Test 0.4k Translation Test 130 ITIG (Generated) Storytelling Generation Test 50 Activity Continuation Test 40
Table 9: Different MLLMs used for generating pairwise responses for sub-tasks under the "Generated" column in Figure 2 of the main text. We provide the information about sub-tasks for NLG and IU in Table 8, as this information is not shown in Figure 2 of the main text. For the tasks in the "Generated" column in Figure 2 of the main text, due to the absence of pairwise data, we use four MLLMs with varying performance levels to generate pairwise responses for each query, as illustrated in Table 9.
this section cite: []

Section: References
Ref_id:b0 Title: A benchmark for measuring harmfulness of llm agents Year: (2024)
Ref_id:b1 Title: Benchmarking foundation models with language-model-as-an-examiner Year: (2023)
Ref_id:b2 Title: Artificial intelligence for drug toxicity and safety Year: (2019)
Ref_id:b3 Title: Generating coherent sequences of visual illustrations for real-world manual tasks Year: (2024)
Ref_id:b4 Title: A survey on evaluation of large language models Year: (2024)
Ref_id:b5 Title: Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark Year: (2024)
Ref_id:b6 Title: Towards end-to-end embodied decision making via multi-modal large language model: Explorations with gpt4-vision and beyond Year: (2023)
Ref_id:b7 Title: Neural storyboard artist: Visualizing stories with coherent image sequences Year: (2019)
Ref_id:b8 Title: Measuring bias in ai models: an statistical approach introducing n-sigma Year: (2023)
Ref_id:b9 Title: Bold: Dataset and metrics for measuring biases in open-ended language generation Year: (2021)
Ref_id:b10 Title: Summeval: Re-evaluating summarization evaluation Year: (2021)
Ref_id:b11 Title: Evaluate as you desire Year: (2023)
Ref_id:b12 Title: Bias and fairness in large language models: A survey Year: (2024)
Ref_id:b13 Title: RealToxici-tyPrompts: Evaluating neural toxic degeneration in language models Year: (2020-11)
Ref_id:b14 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b15 Title: Benchmarking spatial relationships in text-to-image generation Year: (2022)
Ref_id:b16 Title: The llama 3 herd of models Year: (2024)
Ref_id:b17 Title: Openmeva: A benchmark for evaluating open-ended story generation metrics Year: (2021)
Ref_id:b18 Title: Bias in large language models: Origin, evaluation, and mitigation Year: (2024)
Ref_id:b19 Title: Evaluating large language models: A comprehensive survey Year: (2023)
Ref_id:b20 Title: Sentence and word complexity Year: (2011)
Ref_id:b21 Title: Unnatural instructions: Tuning language models with (almost) no human labor Year: (2022)
Ref_id:b22 Title: A comprehensive survey of deep learning for image captioning Year: (2019)
Ref_id:b23 Title: Are llm-based evaluators confusing nlg quality criteria Year: (2024)
Ref_id:b24 Title: A reference-free nlg evaluation language model with flexibility and interpretability Year: (2024)
Ref_id:b25 Title: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation Year: (2023)
Ref_id:b26 Title: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions Year: (2025)
Ref_id:b27 Title: Tigerscore: Towards building explainable metric for all text generation tasks Year: (2023)
Ref_id:b28 Title: Towards an informative critique generation model for evaluation of large language model generation Year: (2023)
Ref_id:b29 Title: Prometheus: Inducing fine-grained evaluation capability in language models Year: (2023)
Ref_id:b30 Title: Prometheus 2: An open source language model specialized in evaluating other language models Year: (2024)
Ref_id:b31 Title: Evallm: Interactive evaluation of large language model prompts on user-defined criteria Year: (2024)
Ref_id:b32 Title: Picka-pic: An open dataset of user preferences for text-to-image generation Year: (2023)
Ref_id:b33 Title: Wikihow: A large scale text summarization dataset Year: (2018)
Ref_id:b34 Title: Do llms have political correctness? analyzing ethical biases and jailbreak vulnerabilities in ai systems Year: (2024)
Ref_id:b35 Title: Genai-bench: Evaluating and improving compositional text-to-visual generation Year: (2024)
Ref_id:b36 Title: Llava-next: Stronger llms supercharge multimodal capabilities in the wild Year: (2024-05)
Ref_id:b37 Title: Llava-med: Training a large language-and-vision assistant for biomedicine in one day Year: (2023)
Ref_id:b38 Title: Multimodal foundation models: From specialists to general-purpose assistants Year: (2024)
Ref_id:b39 Title: Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models Year: (2024)
Ref_id:b40 Title: Generative judge for evaluating alignment Year: (2023)
Ref_id:b41 Title: M 3 it: A large-scale dataset towards multi-modal multilingual instruction tuning Year: (2023)
Ref_id:b42 Title: Vlrewardbench: A challenging benchmark for vision-language generative reward models Year: (2024)
Ref_id:b43 Title: Vlfeedback: A large-scale ai feedback dataset for large vision-language models alignment Year: (2024)
Ref_id:b44 Title: Exploring the reliability of large language models as customized evaluators for diverse nlp tasks Year: (2023)
Ref_id:b45 Title: Advancing a multi-modality foundation model for human image aesthetic assessment Year: (2025)
Ref_id:b46 Title: Visual instruction tuning Year: (2023)
Ref_id:b47 Title: X-eval: Generalizable multi-aspect text evaluation via augmented instruction tuning with auxiliary evaluation aspects Year: (2023)
Ref_id:b48 Title: Holistic evaluation for interleaved text-and-image generation Year: (2024)
Ref_id:b49 Title: Hd-eval: Aligning large language model evaluators through hierarchical criteria decomposition Year: (2024)
Ref_id:b50 Title: Wildvision: Evaluating vision-language models in the wild with human preferences Year: (2024)
Ref_id:b51 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b52 Title: Infographicvqa Year: (2022)
Ref_id:b53 Title: Language complexity measurement as a noisy zero-shot proxy for evaluating llm performance Year: (2025)
Ref_id:b54 Title: Clipcap: Clip prefix for image captioning Year: (2021)
Ref_id:b55 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2023)
Ref_id:b56 Title: Learning to plan & reason for evaluation with thinking-llm-as-a-judge Year: (2025)
Ref_id:b57 Title: Revealing persona biases in dialogue systems Year: (2021)
Ref_id:b58 Title: Towards vqa models that can read Year: (2019)
Ref_id:b59 Title: Stories for images-in-sequence by using visual and narrative components Year: (2018-09-17)
Ref_id:b60 Title: Evaluation metrics for machine reading comprehension: Prerequisite skills and readability Year: (2017)
Ref_id:b61 Title: Aligning large multimodal models with factually augmented rlhf Year: (2023)
Ref_id:b62 Title: Readability as applied to an abe assessment instrument Year: (1999)
Ref_id:b63 Title: Winoground: Probing vision and language models for visio-linguistic compositionality Year: (2022)
Ref_id:b64 Title: Categorizing and inferring the relationship between the text and image of twitter posts Year: (2019)
Ref_id:b65 Title: Diffusion model alignment using direct preference optimization Year: (2024)
Ref_id:b66 Title: A unified agentic framework for evaluating conditional image generation Year: (2025-07)
Ref_id:b67 Title: A critic for language model generation Year: (2023)
Ref_id:b68 Title: Mitigating the language mismatch and repetition issues in llm-based machine translation via model editing Year: (2024)
Ref_id:b69 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b70 Title: Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization Year: (2023)
Ref_id:b71 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2022)
Ref_id:b72 Title: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis Year: (2023)
Ref_id:b73 Title: Llava-critic: Learning to evaluate multimodal models Year: (2024)
Ref_id:b74 Title: Imagereward: Learning and evaluating human preferences for text-to-image generation Year: (2023)
Ref_id:b75 Title: Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation Year: (2024)
Ref_id:b76 Title: Instructscore: Explainable text generation evaluation with finegrained feedback Year: (2023)
Ref_id:b77 Title: A survey on large language model (llm) security and privacy: The good, the bad, and the ugly. High-Confidence Computing Year: (2024)
Ref_id:b78 Title: Justice or prejudice? quantifying biases in llm-as-a-judge Year: (2024)
Ref_id:b79 Title: Flask: Fine-grained language model evaluation based on alignment skill sets Year: (2023)
Ref_id:b80 Title: Automating dataset updates towards reliable and timely evaluation of large language models Year: (2024)
Ref_id:b81 Title: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback Year: (2024)
Ref_id:b82 Title: Evaluating large language models at evaluating instruction following Year: (2023)
Ref_id:b83 Title: Magicbrush: A manually annotated dataset for instruction-guided image editing Year: (2023)
Ref_id:b84 Title: Pmc-vqa: Visual instruction tuning for medical visual question answering Year: (2023)
Ref_id:b85 Title: Enhanced visual instruction tuning for text-rich image understanding Year: (2023)
Ref_id:b86 Title: Q-eval-100k: Evaluating visual quality and alignment level for text-to-vision content Year: (2025)
Ref_id:b87 Title: Svit: Scaling up visual instruction tuning Year: (2023)
Ref_id:b88 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
