Title: Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition
Abstract: Handwritten Mathematical Expression Recognition (HMER) remains a persistent challenge in Optical Character Recognition (OCR) due to the inherent freedom of symbol layouts and variability in handwriting styles. Prior methods have faced performance bottlenecks by proposing isolated architectural modifications, making them difficult to integrate coherently into a unified framework. Meanwhile, recent advances in pretrained vision-language models (VLMs) have demonstrated strong cross-task generalization, offering a promising foundation for developing unified solutions. In this paper, we introduce Uni-MuMER, which fully fine-tunes a VLM for the HMER task without modifying its architecture, effectively injecting domainspecific knowledge into a generalist framework. Our method integrates three datadriven tasks: Tree-Aware Chain-of-Thought (Tree-CoT) for structured spatial reasoning, Error-Driven Learning (EDL) for reducing confusion among visually similar characters, and Symbol Counting (SC) for improving recognition consistency in long expressions. Experiments on the CROHME and HME100K datasets show that Uni-MuMER achieves super state-of-the-art performance, outperforming the best lightweight specialized model SSAN by 16.31% and the top-performing VLM Gemini2.5-flash by 24.42% under zero-shot setting. Our datasets, models, and code are open-sourced at: https://github.com/BFlameSwift/Uni-MuMER

Section: 
1 Introduction 38.64 37.75 37.45 62.58 62.51 65.30 72.62 64.26 69.14 82.05 77.94 79.23 30 40 50 60 70 80 90 100 CROHME 2014 CROHME 2016 CROHME 2019 ExpRate(%) SSAN(2025) Qwen2.5VL Zero-Shot Qwen2.5VL SFT(Vanilla) Uni-MuMER † Handwritten Mathematical Expression Recognition (HMER) seeks to translate handwritten expressions into machine-readable markup, supporting document understanding and digital preservation of scientific material. Unlike standard Optical Character Recognition (OCR), HMER involves parsing complex 2D structures [15], ambiguous symbols, and the inherent freedom of symbol layout and variability in handwriting styles [68], requiring not just recognizing individual symbols but also layout reasoning and parsing their complex spatial relationships. Recent approaches have primarily leveraged RNNs [23] or Transformers [48] due to their powerful sequential modeling capabilities. Additional modules, such as tree-structured decoders [64,52,70] and relative-position-aware mechanisms [18,62,53,54], have been introduced to further enhance performance. These dominant research directions in HMER have focused on incorporating prior human knowledge into model architectures, particularly through manually designed structural modules or attention mechanisms [63,65,66,67,30,18,4].
Despite the importance of HMER and numerous proposed improvements, the field has seen only marginal progress in recent years. As is shown in Tab. 1, performance on the CROHME datasets [38,39,37] has improved by merely 3% (from CoMER [66] to SSAN [62]), underscoring an urgent need for a novel paradigm. The limited progress of these models arose from three constraints: (1) Improvements are isolated and model-specific, making them hard to integrate or scale. (2) Optimizing across multiple auxiliary tasks remains challenging, with many approaches focusing on singular priors rather than adopting a unified, multi-faceted enhancement [7]; and (3) Models trained on single-domain datasets, struggling to generalize to other datasets, lack essential scalability and transferability [68,46]. Additionally, widely standard metrics like ExpRate fail to capture visual equivalence in L A T E X outputs, and the scarcity of diverse data exacerbates these constraints. Recently introduced visual metrics such as CDM and large-scale datasets like Mathwriting offer opportunities for large-scale cross-dataset training and multiple L A T E X syntax style evaluation.
Rapid advancements in pretrained vision-language models (VLMs) have significantly enhanced foundational recognition capabilities and generalization across related tasks [8,25,28,31,32]. While early benchmarks like OCRBench [34] reported poor HMER performance for generalist VLMs, Tab. 1 presents our evaluation of recent open-source and closed-source VLMs, such as Qwen2.5-VL [3], Doubao-1.5-pro [45], Gemini2.5-flash [11], and GPT4o [41]. Our results indicate that these large-scale models have exhibited unexpectedly strong capabilities in handling structured recognition tasks. However, these high-performance closed-source models trained on massive amounts of undisclosed data [1] make it difficult to pinpoint how to systematically improve performance on HMER. Consequently, empowering open-source VLMs to achieve comparable or superior HMER performance remains an open and urgent challenge.
To bridge this gap, we propose Uni-MuMER, a unified multi-task fine-tuning framework for enhancing open-source VLMs in HMER. Unlike previous methods constrained by single-domain datasets or isolated architectural improvements, Uni-MuMER fully exploits existing data resources through data-driven fine-tuning. Another motivation of our work is to unify previously fragmented HMER tasks into a unified framework, shifting the focus from incremental architectural modifications toward generalizable recognition capabilities. Concretely, we introduce specialized training data and learning objectives, employing novel tasks such as Tree-Aware Chain-of-Thought for explicit structural reasoning [51,57], Error-driven learning to reduce symbol confusion [36], and Symbol Counting [30] to enhance parsing expressions capabilities. Fig. 1 nicely illustrates the jump with Uni-MuMER; it exceeds Qwen-2.5-VL in the zero-shot setting and the vanilla SFT setting across multiple CROHME datasets. Besides, owing to advancements in open-source inference frameworks (e.g., vllm [29]), our method achieves superior inference speeds compared to traditional methods, enhancing its practical applicability. Our contributions are listed below:
• We propose a new unified paradigm for HMER. Unlike prior methods that heavily rely on specialized networks and single-task training, our data-driven multi-task method injects domain knowledge into a generalist VLM, yielding cumulative performance gains.
• We introduce three data-driven tasks: Tree-Chain-of-Thought, Error-Driven Learning, and Symbol Counting. These systematically address the challenges of HMER, which are two-dimensional structure, ambiguous handwriting, and output consistency.
• Our Uni-MuMER method achieves new SOTA results on the CROHME and HME100K datasets.
Notably, it attains 79.74% averaged across CROHME datasets (+41.79% over Qwen2.5-VL, +24.42% over Gemini2.5-flash in zero-shot setting, and +16.31% over specialized models SSAN).
2 Related Work
this section cite: ['b14', 'b68', 'b23', 'b48', 'b64', 'b52', 'b70', 'b17', 'b62', 'b53', 'b54', 'b63', 'b65', 'b66', 'b67', 'b30', 'b17', 'b3', 'b38', 'b39', 'b37', 'b66', 'b62', 'b0', 'b6', 'b68', 'b46', 'b7', 'b25', 'b28', 'b31', 'b32', 'b34', 'b2', 'b45', 'b10', 'b41', 'b0', 'b51', 'b57', 'b36', 'b30', 'b29']

Section: HMER
Early rule-based approaches attempted to address the challenges of HMER through symbol segmentation, recognition, and syntactic parsing based on handcrafted grammars [6,60,10,2,22,47]. These methods struggled with generalization to diverse handwriting styles and complex layout structures.
this section cite: ['b5', 'b60', 'b9', 'b1', 'b22', 'b47']

Section: Sequence-based decoding methods
The emergence of deep learning shifted HMER towards endto-end sequence modeling tasks. Early encoder-decoder architectures based on RNNs, WAP [65] first applied sequence-to-sequence learning to HMER. Subsequent RNN-based models improved visual encoding [63] and robustness to handwriting style [53,54]. Inspired by the Transformer's success [48], BTTR [67] introduced the first Transformer-based model for HMER. To enhance alignment during decoding, CoMER [66] introduced a coverage attention mechanism.
Multi-task Learning Beyond sequence modeling, researchers have explored structural decoders and multi-task models to better capture 2D structure. One central line of work focuses on modeling the hierarchical tree structure of expressions. TreeDecoder (TD) [64] and its improved version TDv2 directly predict a tree-structured representation of the expression. SAN [58] introduced syntactic constraints into the decoding process, and TAMER [70] jointly optimizes both sequence and tree decoding within a unified Transformer framework. Various auxiliary tasks have also been proposed to inject domain knowledge: ABM [4] adds a dual-direction decoder loss to enhance context modeling. RLFN [9] fuses a language-model-based module with the recognizer to leverage L A T E X syntax and context. ICAL [69] proposed an implicit character construction to capture latent symbol-level semantics. To address symbol omission and repetition, CAN [30] incorporated an auxiliary symbolcounting task. UniMERNet [49] adds a Length-Aware Module targeting real-world expressions' extreme length variance. PosFormer [18] and SSAN [62] explicitly model spatial relationships among symbols to guide the network's attention.
this section cite: ['b65', 'b63', 'b53', 'b54', 'b48', 'b67', 'b66', 'b64', 'b58', 'b70', 'b3', 'b8', 'b69', 'b30', 'b49', 'b17', 'b62']

Section: Vision-Language Models
Vision-Language Models (VLMs), initially popularized by contrastive learning frameworks like CLIP [43], laid the groundwork for powerful zero-shot multimodal recognition. Donut [27] and LayoutLMv3 [24] extended this capability specifically to OCR tasks, effectively recognizing diverse textual content. High-resolution, document-centric VLMs, notably Monkey [33] and TextMonkey [35], enhanced OCR performance by combining patch-based image encoding with explicit textual supervision. For mathematical OCR, models like Im2LaTeX [12] and Nougat [5] focused explicitly on end-to-end LaTeX reconstruction from printed equations and scientific documents. The recent FERMAT benchmark [40] explicitly evaluated multiple SOTA VLMs on handwritten mathematics recognition, revealing critical gaps between general text OCR and handwritten mathematical recognition. Domain-specific adaptations such as VLPG [20] and HiE-VL [21] propose graph-based or hierarchical adapters to enhance mathematical recognition, although their extensive architectural modifications and modest performance limit broader adoption. In contrast, recent general-purpose frameworks like Qwen2.5-VL [3], equipped with dynamic-resolution ViTs and structured outputs, demonstrate promising potential for adaptation to mathematical OCR tasks.
this section cite: ['b43', 'b27', 'b24', 'b33', 'b35', 'b11', 'b4', 'b40', 'b20', 'b21', 'b2']

Section: Method
We introduce Uni-MuMER, a unified multi-task fine-tuning framework for enhancing open-source VLMs in HMER. The overall pipeline is shown in Fig. 2. Given an input image of a handwritten expression and a task-specific instruction, the model is trained to produce the corresponding output. Uni-MuMER integrates four tasks: Vanilla HMER, Tree-aware Chain-of-Thought, Error-Driven Learning, and Symbol Counting, into a unified training paradigm. The primary goal is to adapt a general-purpose VLM to the highly structured and domain-specific knowledge of HMER without modifying any original architecture. Subsequent subsections provide additional details for each task.
this section cite: []

Section: Vanilla HMER
Base Model We adopt Qwen2.5-VL-3B [3] as the VLM Backbone. It comprises a ViT-based visual encoder and a transformer-based language decoder, pre-trained to perform various image-to-text tasks, which provide robust visual grounding and structured sequence-generation capabilities, rendering it an effective foundation for end-to-end fine-tuning in HMER tasks.
Vanilla HMER As is shown at the top of Fig. 2, Vanilla HMER involves providing an image of a mathematical expression alongside a textual instruction, prompting the model to directly output the corresponding L A T E X formatted expression. Traditional lightweight specialized models took images alone as input and generated the expressions, whereas VLMs now exhibit strong generalization capabilities across related tasks. We also utilize models fine-tuned specifically on the Vanilla HMER I have an image... and its OCR recognition result. Please correct the errors marked <error_start> <error_end> and <deleted> and give the correct mathematical expression.
this section cite: ['b2']

Section: Marked Expression:
I have an image ... and its OCR recognition result. Please help me to detect possible errors in the recognition result and mark the places where errors occur with <error_start> <error_end> and <deleted>. Expression: Please recognise this image of a mathematical expression and give the result in latex format.
|
I have an image of a handwritten mathematical expression. Please write out the expression of the formula in the image using LaTeX format.
this section cite: []

Section: Error Detection and Correction

this section cite: []

Section: Vanilla HMER
Result: task as baselines for subsequent comparison. Through Vanilla HMER, the model acquires essential recognition capabilities and ensures accurate and structured outputs. Tree-CoT The construction procedure of Tree-CoT is illustrated in Fig. 3. We first parse the input expression to derive its Abstract Syntax Tree (AST), explicitly capturing hierarchical and spatial relationships among symbols. We subsequently perform a depth-first traversal (DFS) of the AST to sequentially linearize the tree structure. To encode this structured representation, we introduce a specific serialization scheme, using tab-based indentation to reflect tree depth and newline separation to distinguish individual nodes clearly, and the raw text is in Fig. 2. Each serialized line explicitly contains the symbol label and its spatial relation to its immediate parent node, resulting in a coherent textual representation of the AST.
this section cite: []

Section: Qwen-VL

this section cite: []

Section: 🔥 🔥 🔥

this section cite: []

Section: Error Detection

this section cite: []

Section: Uni-MuMER

this section cite: []

Section: Symbol Counting

this section cite: []

Section: Error-Driven Learning
Error-Driven Learning (EDL) leverages a learning-from-mistakes paradigm to enhance model accuracy. Its input consists of an error corpus generated by the model itself, structured into two distinct subtasks: error detection and error correction. Inspired by self-improvement strategies and previous work indicating the suitability of language models for correcting HMER errors [9], we integrate these subtasks explicitly. The subsequent sections detail the generation of the error corpus and the definitions of error detection and correction subtasks.
... ...
(n-1) Fold as Train Data ...
this section cite: ['b8']

Section: multi-sample on

this section cite: []

Section: Sample 1 Sample 2
...
this section cite: []

Section: Sample k Sample m
Error Corpus
this section cite: []

Section: 🔥 🔥

this section cite: []

Section: Error Detection & Correction Data
Build Partitioning Out-of-Fold Training Duel Task of EDL Mulit-Sample Corpus Error Corpus Generation As is shown in the Fig. 4. Initially, the complete dataset (e.g., CROHME) is randomly partitioned into multiple distinct subsets denoted as F 1 , F 2 , . . . , F n . Subsequently, we perform cross-validation training, where each VLM is trained on n -1 folds and evaluated on the remaining held-out fold F i . During this, multiple candidate predictions are generated through multiple sampling for each input image to collect potential model outputs. By comparing these candidate predictions to their corresponding ground-truth labels, we explicitly identify erroneous outputs and compile these erroneous predictions into the final error corpus.
this section cite: []

Section: Error Detection and Correction
Based on this error corpus, we formulate two related subtasks: error detection and error correction. As shown in the Fig. 2, the error-detection task takes as input the original expression image alongside a potentially erroneous predicted expression, outputting a marked expression wherein errors are explicitly enclosed by <error_start> and <error_end> tags, and omissions are explicitly indicated by <deleted> tags. The subsequent error-correction subtask receives this masked expression as input and produces the correction log and corresponding L A T E X expression. The scale of the error corpus obtained for each training set used in Sec. 4.1 is approximately equal to that of the original training data.
this section cite: []

Section: Symbol Counting Auxiliary Task
We introduce a Symbol Counting (SC) auxiliary task designed to encourage the model to explicitly account for all symbols appearing in expressions. The input to this task comprises an expression image along with a counting instruction, and the expected output consists of the total count of visible symbols in the expression alongside the corresponding L A T E X expression. Inspired by the CAN [30], we observe that models frequently produce locally coherent yet globally inconsistent outputs, especially involving repeated symbols. To mitigate this issue, we explicitly integrate symbolcounting information into the textual output representation. This task compels the model to accurately predict symbol counts, thus reducing the likelihood of symbol omissions or hallucinations during final L A T E X expression generation.
this section cite: ['b30']

Section: Symbol Counting.
We construct training targets that prepend the count string to the actual L A T E X. Specifically, for a given handwritten expression image, the target output becomes <Count String>\n<Expression string>. For example: For the expression a 2 +1 2 , the textual count string is: \frac:1,a:1,2:2,+:1.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
This section details the experimental setup used to evaluate the Uni-MuMER, including datasets, evaluation metrics, training settings, and baselines.
this section cite: []

Section: Datasets
The CROHME series [37,38,39] is the most widely used dataset for HMER. It comprises 8836 training samples, and three test sets contain 986, 1147, and 1199 images for the CROHME 2014, 2016, 2019. Recently, the CROHME 2023 [56] iteration provided an expanded training set of 10979 images and a test set of 2300 images. HME100K [59] is a large-scale, real-world dataset. It consists of 74502 training and 24607 testing images, encompassing various writing styles and conditions. MathWriting [17], released by Google Research in 2024, is the largest HME corpus to date with 230K human-written and 400K synthetically-generated expressions. Finally, Im2Latexv2 [44] builds upon the printed expression dataset Im2Latex-100k [12] and enhances it with improved L A T E X normalization and 59 diverse rendering styles for more realistic expression images. Training We fine-tune the Qwen2.5-VL-3B model on all datasets with three data-driven tasks simultaneously for a single epoch. Further details can be found in the Appendix B.4.
this section cite: ['b37', 'b38', 'b39', 'b56', 'b59', 'b16', 'b44', 'b11']

Section: Baselines

this section cite: []

Section: Main Results
Tab. 1 displays the strong performance of our Uni-MuMER and previous methods on the CROHME dataset. As CDM is a recently proposed metric, we reproduced all reported results involving CDM. For a fair comparison, the Vanilla (baseline) reported in Tab. 1 and Tab. 2 was trained exclusively on either CROHME or HME100K dataset. The Uni-MuMER variant incorporated training samples constructed from the corresponding images for three tasks. Further utilizing external data, UniMuMER † was trained on approximately 1.6M samples, comprising data from 386K images across the three tasks and all datasets. Due to space limitations, additional results are presented in Appendix D.
Comparison with Previous SOTA Methods Data augmentation has traditionally improved performance in lightweight models, yet inherent limitations still restrict overall effectiveness. Recently, VLPG [20] and HiE-VL [21], have integrated VLM into HMER. HiE-VL enhances performance through non-end-to-end training, architectural refinements, and extra data. However, both Uni-MuMER and Uni-MuMER † (use external data) maintain a notable performance advantage over these VLM-based methods. Notably, we also trained CoMER a † [66] using the same external data (386K images) and observed a significant performance degradation. This finding indicates that the inherent limitations of lightweight models hinder their ability to effectively utilize large and diverse datasets.
Comparison with Large VLMs (Zero-Shot) Next, we investigate the zero-shot performance of both open-source (Qwen-2.5VL family) and closed-source large VLMs on CROHME. Although open-source models exhibit nontrivial results, they tend to lag behind previous SOTA methods when no fine-tuning is involved. Contrastingly, certain closed-source VLMs like Gemini2.5-flash exhibit remarkable accuracy, surpassing lightweight specialized models in a purely zero-shot setting. They still underperform relative to our proposed Uni-MuMER and Uni-MuMER † method.
this section cite: ['b20', 'b21', 'b66']

Section: Performance on HME100K
To focus the evaluation on mathematical expression recognition, following [16], we exclude all test instances containing CJK characters. Detailed impacts are discussed in Appendix C.5. Tab. 2 shows that open-source and closed-source VLMs exhibit modest performance, which is due to the challenging real-world conditions in HME100K dataset. In contrast, several prior lightweight specialized methods exclusively trained on HME100K, such as TAMER, achieve strong performance. Although HiE-VL utilizes external training data, it achieves limited results on HME100K, even underperforming compared to prior models. Our Uni-MuMER achieves SOTA performance. Additionally, by using external data, Uni-MuMER † further enhances its performance, clearly surpassing HiE-VL. In this section, we present an ablation study to quantify the contribution of the three proposed tasks, Tree-CoT, EDL, and SC. As is shown in Tab. 3, each individual task enhances performance compared to the Vanilla(baseline).
this section cite: ['b15']

Section: Analysis

this section cite: []

Section: Ablation Study
The Uni-MuMER combined integration of all tasks results in the best overall performance, while removing any module leads to noticeable performance degradation, highlighting their complementary roles and collective importance.
Tree-CoT enhances comprehension of structurally complex expressions Fig. 5a presents a comparative analysis of accuracy across models with varying structural complexities of expressions. The Vanilla model experiences a significant degradation in accuracy as structural complexity increases. Introducing the Tree-CoT strategy notably alleviates this performance decline. Specifically, for structurally complex expressions, Vanilla+Tree-CoT demonstrates an accuracy improvement of approximately 5-6% over the Vanilla baseline, closely matching the performance of the Full model. Conversely, Tree-CoT offers marginal improvements for simpler expressions, suggesting its primary utility is enhancing model robustness and accuracy when handling structurally intricate expressions.
this section cite: []

Section: Effects of Data Diversity
In this section, we examine the influence of incrementally scaling training datasets on model performance. As illustrated in Fig. 6, start with the CROHME dataset, which serves as the standard benchmark in HMER. We then sequentially incorporate increasingly diverse datasets, including the updated CROHME-2023 set, the real-scene HME100K dataset, the extensive handwritten corpus MathWriting, and finally the printed-expression Im2LaTeXv2 dataset. At each step, the model is retrained from scratch on the cumulative training data and evaluated across multiple distinct test sets.
To effectively account for variations in L A T E X notation styles, we employ ExpRate@CDM as the standardized metric for consistent performance comparison.
77.19 79.61 78.01 83.20 82.86 67.56 74.10 73.83 73.33 74.42 47.90 50.47 74.24 74.51 74.30 23.93 27.46 32.89 68.93 69.11 30.17 22.16 35.20 75.17 93.29 20 30 40 50 60 70 80 90 100 CROHME Train 8K + CROHME-23 (10K) + HME100K (72K) + MathWriting (220K) + Im2Latexv2 (69K printed) Test Set ExpRate@CDM(%) CROHME Avg. Test Sets CROHME-23 Test Set HME100K Test Set Mathwriting Test Set Im2LaTeXv2 Test Set Benefits of Enhanced Data Diversity Each incremental expansion of the training sets with more varied handwritten expressions yields consistent gains on all evaluation sets, confirming that performance scales with data diversity.
this section cite: []

Section: Out-of-Domain Generalization
Accuracy on printed expression images gradually improves with increased handwritten training data, reflecting partial transfer of structural knowledge for expression recognition. Nevertheless, its accuracy improves significantly after adding Im2LaTeXv2, reflecting the value of in-domain printed data for that target.
this section cite: []

Section: Leave-One-Out Ablation
To further investigate the out-of-domain generalization performance, we conducted an additional "Leave-One-Out" experiment, training Uni-MuMER on N-1 datasets and evaluating it on the held-out N-th dataset.
The results below present the expression recognition accuracy in each scenario. For comparison, we include the baseline CoMER and Doubao-1.5-pro under zero-shot settings.
Table 5: Leave-one-out ablation of training datasets. (ExpRate@CDM%) We train Uni-MuMER in six configurations: using all data (all five sources), and using all-minus-one for each major dataset. Results are reported on CROHME-Average, CROHME 2023, HME100K, MathWriting, and Im2LaTeXv2. For reference, we also show a specialized model (CoMER) trained with and without CROHME, and a closed-source VLM (Doubao-1.5-pro) for reference.
this section cite: []

Section: Model CROHME-Avg CROHME 2023 HME100K MathWriting Im2LaTeXv2
CoMER † (w/o CROHME) 45.99 58.28 39.49 23.23 40.39 CoMER † 52.29 59.91 44.63 28.45 53.37 Doubao-1.5-pro 65.77 58.18 55.08 26.34 27.66 Uni-MuMER † (w/o CROHME) 78.22 72.19 74.47 68.08 89.18 Uni-MuMER † (w/o CROHME-2023) 82.31 71.66 74.68 68.85 89.77 Uni-MuMER † (w/o HME100K) 82.30 72.01 52.38 68.94 91.19 Uni-MuMER † (w/o MathWriting) 79.58 73.51 74.29 32.55 93.31 Uni-MuMER † (w/o Im2LaTeXv2) 83.20 73.33 74.51 68.93 75.17 Uni-MuMER † 82.89 74.42 74.30 69.11 93.29
The leave-one-out evaluation confirms that our unified model can generalize to an unseen dataset reasonably well. When the unseen domain is very different, performance drops more markedly, which highlights an area for improvement. HME100K for real-life, low-quality images, MathWriting for densely structured expression, and Im2LaTeXv2 for printed expression images, each of which represents a markedly different and challenging domain.
Crucially, after introducing even modest amounts of domain-relevant data, our unified approach easily achieves top-tier performance. Thus, our results indicate that, compared to existing specialized models and general VLM baselines, our approach provides superior generalization with minimal domainspecific data requirements, highlighting a clear advantage and direction for future improvement.
this section cite: []

Section: Mixing General-Purpose VLM Data
From the perspective of general-purpose LLMs and VLMs, mixing data from different domains can yield substantial performance gains. Although our Uni-MuMER addresses previous limitations by introducing a unified solution, achieving super SOTA performance, it's necessary to validate whether Uni-MuMER benefits from general domain data beyond HMER-only training and how Uni-MuMER-Data affects the model's general capabilities. To this end, we introduce Uni-MuMER-LLAVA, fine-tuned using an equal 1:1 mixture of HMER and LLaVA-OneVision data, and compare its performance with Uni-MuMER across benchmarks assessing HMER-specific and general capabilities. As shown in Tab. 6 and Tab. 7, Uni-MuMER exhibits strong generalization capabilities despite being fine-tuned exclusively on HMER-specific data, achieving performance comparable to Qwen2.5VL-3B. Notably, Uni-MuMER-LLAVA further enhances overall performance, surpassing Uni-MuMER on general tasks like MMMU and Math, while incurring marginal performance decreases on specific HMER test sets. It's promising to integrate our method and Uni-MuMER-Data to empower VLMs.
this section cite: []

Section: Discussion
For Small Models: Larger Model, Better Performance, Faster Speed Previous HMER models were lightweight and relied on task-specific architectural designs (e.g., tree-structured modules). Uni-MuMER builds upon both these task-specific insights and current VLM, taking a significant step forward with a unified and more powerful model. Benefiting from recent advances in efficient inference frameworks, Uni-MuMER not only achieves substantially improved performance but also delivers faster inference compared to smaller models. Further detail is provided in Appendix C.2.
For VLM: A Chain-of-Thought Perspective on Expression Recognition Unlike direct expression recognition, Uni-MuMER adopts a chain-of-thought perspective by formulating three tasks: expression tree construction, error detection and correction, and symbol counting. These tasks guide the model step by step toward the final prediction, enhancing both interpretability and generalization. This approach offers a novel direction for advancing expression recognition with current VLM.
this section cite: []

Section: References
Ref_id:b0 Title: Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Recognition of on-line handwritten mathematical expressions using 2d stochastic context-free grammars and hidden markov models Year: (2014)
Ref_id:b2 Title:  Year: ()
Ref_id:b3 Title: Handwritten mathematical expression recognition via attention aggregation based bi-directional mutual learning Year: (2022)
Ref_id:b4 Title: Nougat: Neural optical understanding for academic documents Year: (2023)
Ref_id:b5 Title: Mathematical expression recognition: a survey Year: (2000)
Ref_id:b6 Title: Mmhmer: Multi-viewer and multi-task for handwritten mathematical expression recognition Year: (2025)
Ref_id:b7 Title: A jointly-scaled multilingual language-image model Year: (2002)
Ref_id:b8 Title: Language model is suitable for correction of handwritten mathematical expressions recognition Year: (2023-12)
Ref_id:b9 Title: Recognition of equations using a two-dimensional stochastic context-free grammar Year: (1989)
Ref_id:b10 Title: Gemini 2.5 Flash: Enhanced multimodal model Year: (2025-04-17)
Ref_id:b11 Title: Image-to-markup generation with coarse-to-fine attention Year: (2017)
Ref_id:b12 Title: Palm-e: An embodied multimodal language model Year: (1932)
Ref_id:b13 Title: Representing online handwriting for recognition in large vision-language models Year: ()
Ref_id:b14 Title: Handwritten mathematical symbol recognition using machine learning techniques: Review Year: (2020)
Ref_id:b15 Title: Uniblock: Scoring and filtering corpus with unicode block information Year: (2019-11)
Ref_id:b16 Title: Mathwriting: A dataset for handwritten mathematical expression recognition Year: (2024)
Ref_id:b17 Title: Posformer: recognizing complex handwritten mathematical expression with position forest transformer Year: ()
Ref_id:b18 Title:  Year: (2024)
Ref_id:b19 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b20 Title: Vision-language pre-training for graph-based handwritten mathematical expression recognition Year: (2025)
Ref_id:b21 Title: HiE-VL: A large vision-language model with hierarchical adapter for handwritten mathematical expression recognition Year: (2025-04)
Ref_id:b22 Title: Understanding mathematical expressions from document images Year: (1995)
Ref_id:b23 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b24 Title: Layoutlmv3: Pre-training for document ai with unified text and image masking Year: (2022)
Ref_id:b25 Title: Scaling up visual and vision-language representation learning with noisy text supervision Year: ()
Ref_id:b26 Title: Latte: Improving latex recognition for tables and formulae with iterative refinement Year: (2025)
Ref_id:b27 Title: Ocr-free document understanding transformer Year: (2022)
Ref_id:b28 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b29 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b30 Title: When counting meets hmer: countingaware network for handwritten mathematical expression recognition Year: (2006)
Ref_id:b31 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b32 Title: BLIP: Bootstrapping language-image pre-training for unified visionlanguage understanding and generation Year: (2002)
Ref_id:b33 Title: Monkey: Image resolution and text label are important things for large multi-modal models Year: (2024)
Ref_id:b34 Title: Ocrbench: on the hidden mystery of ocr in large multimodal models Year: (2024)
Ref_id:b35 Title: Textmonkey: An ocr-free large multimodal model for understanding document Year: (2024)
Ref_id:b36 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b37 Title: ICDAR 2019 CROHME+ TFD: Competition on recognition of handwritten mathematical expressions and typeset formula detection Year: (2019)
Ref_id:b38 Title: Icfhr 2014 competition on recognition of on-line handwritten mathematical expressions (crohme 2014) Year: (2014)
Ref_id:b39 Title: Icfhr2016 crohme: Competition on recognition of online handwritten mathematical expressions Year: (2016)
Ref_id:b40 Title: Can vision-language models evaluate handwritten math? arXiv preprint Year: (2025)
Ref_id:b41 Title: GPT-4o: Multimodal generative transformer Year: (2002)
Ref_id:b42 Title: Image to latex with graph neural network for mathematical formula recognition Year: (2021)
Ref_id:b43 Title: Learning transferable visual models from natural language supervision Year: (2003)
Ref_id:b44 Title: MathNet: A data-centric approach for printed mathematical expression recognition Year: (2024)
Ref_id:b45 Title: Official release reported 22 Year: (2025-01)
Ref_id:b46 Title: A survey on handwritten mathematical expression recognition: The rise of encoder-decoder and GNN models Year: (2002)
Ref_id:b47 Title: Structure analysis and recognition of mathematical expressions Year: (1995)
Ref_id:b48 Title: Attention is all you need Year: (2017)
Ref_id:b49 Title: Unimernet: A universal network for real-world mathematical expression recognition Year: (2024)
Ref_id:b50 Title: Image over text: Transforming formula recognition evaluation with character detection matching Year: (2025)
Ref_id:b51 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b52 Title: TDv2: A Novel Tree-Structured Decoder for Offline Mathematical Expression Recognition Year: (2004-02)
Ref_id:b53 Title: Image-to-markup generation via paired adversarial learning Year: (2018)
Ref_id:b54 Title: Handwritten mathematical expression recognition via paired adversarial learning Year: (2020)
Ref_id:b55 Title: Tst: Tree structured transformer for handwritten mathematical expression recognition Year: (2025)
Ref_id:b56 Title: Icdar 2023 crohme: Competition on recognition of handwritten mathematical expressions Year: (2023)
Ref_id:b57 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b58 Title: Syntax-aware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b59 Title: Syntax-aware network for handwritten mathematical expression recognition Year: (2022)
Ref_id:b60 Title: Recognizing mathematical expressions using tree transformation Year: (2002)
Ref_id:b61 Title: Evaluating structural pattern recognition for handwritten math via primitive label graphs Year: (2013)
Ref_id:b62 Title: SSAN: A symbol spatial-aware network for handwritten mathematical expression recognition Year: (2002)
Ref_id:b63 Title: Multi-scale attention with dense encoder for handwritten mathematical expression recognition Year: (2007)
Ref_id:b64 Title: A tree-structured decoder for image-to-markup generation Year: (2020)
Ref_id:b65 Title: Watch, attend and parse: An end-to-end neural network based approach to handwritten mathematical expression recognition Year: (2017)
Ref_id:b66 Title: Comer: Modeling coverage for transformer-based handwritten mathematical expression recognition Year: (2007)
Ref_id:b67 Title: Handwritten mathematical expression recognition with bidirectionally trained transformer Year: (2021)
Ref_id:b68 Title: Online handwritten mathematical expression recognition and applications: A survey Year: (2021)
Ref_id:b69 Title: Ical: Implicit character-aided learning for enhanced handwritten mathematical expression recognition Year: (2024)
Ref_id:b70 Title: Tamer: Tree-aware transformer for handwritten mathematical expression recognition Year: (2007)
