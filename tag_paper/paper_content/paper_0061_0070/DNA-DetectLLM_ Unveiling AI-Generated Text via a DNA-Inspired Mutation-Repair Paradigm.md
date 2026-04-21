Title: DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm
Abstract: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.

Section: Introduction
The rapid advancement of large language models (LLMs) has created increasingly human-like textual content, substantially narrowing the distinguishable gap between AI-generated and humanwritten text. While these improvements have catalyzed significant technological breakthroughs, they simultaneously pose critical societal challenges, including misinformation dissemination, authorship ambiguity, and threats to intellectual property rights [2,1,13]. Consequently, there is an urgent and growing need for effective and reliable methods to accurately detect AI-generated text.
While significant research efforts have been dedicated to AI-generated text detection, existing methodologies typically adopt either training-based or training-free methods. Training-based methods [28,16,12,39,11] depend upon large volumes of annotated data, limiting their scalability and generalization to new domains. In contrast, training-free approaches [22,4,14,36] leverage intrinsic statistical differences to distinguish human-written and AI-generated texts. Both paradigms fundamentally operate by attempting to identify distinct, separable boundaries within the feature space. However, recent advancements in generative language modeling have produced outputs increasingly indistinguishable from human-authored content, causing these classification boundaries to become progressively blurred. Empirical studies [7,27] have highlighted substantial overlap regions in the feature distributions of human-written and AI-generated texts, significantly undermining detection accuracy in practical scenarios. Therefore, one capable of more precisely and intrinsically capturing differences between the generative processes of AI and human writing is urgently needed.
In molecular biology, DNA's double-helix structure ensures stable transmission of genetic information, yet mutations during replication introduce variations that can lead to individual differences or even diseases such as cancer. In a similar vein, an ideal AI-generated text sequence can be seen as a "template strand", representing the most probable token choices at each position. Human-written texts, by contrast, resemble mutated strands, where token selections deviate from the optimal probabilities, creating measurable differences. Inspired by this biological mechanism, we propose a new perspective for AI-generated text detection: by analogizing to DNA base-repair processes, we iteratively "correct" non-optimal tokens in a text and measure the difficulty of restoring it to the ideal AI-generated form. This repair-based approach captures the intrinsic divergence between AI-generated and human-written texts in a direct and interpretable manner.
Building on this intuition, we propose DNA-DetectLLM, a novel method for zero-shot detection of AIgenerated texts. For each input sequence, we first construct its corresponding ideal AI sequence-that is, the sequence formed by greedily selecting the most probable token at each position under a reference language model. We then perform a token-by-token repair process on the input sequence, progressively modifying tokens toward their optimal choices until the sequence fully aligns with the ideal AI sequence. To quantify the difficulty of this repair process, we introduce a repair score that captures the cumulative effort required to complete the transformation. Finally, by comparing the repair score against a calibrated threshold, DNA-DetectLLM robustly distinguishes AI-generated texts from human-written ones, leveraging the fundamental differences in their deviation patterns from ideal generation.
DNA-DetectLLM consistently achieves state-of-the-art performance across multiple datasets and LLMs. In particular, it obtains relative improvements of 5.55% in AUROC and 2.08% in F1 score on three public benchmark datasets. Additionally, the method exhibits notable robustness against various adversarial attacks and across different input lengths. Efficiency experiments further indicate rapid detection capability, processing each sample in under 0.8s.
Our contributions are summarized as follows:
• Inspired by the mutation and repair mechanisms of nucleotide bases in DNA replication, we introduce the mutation-repair paradigm into AI-generated text detection.
• We propose DNA-DetectLLM, a novel zero-shot method for detecting AI-generated text that incrementally repairs mutated tokens within the input sequence until it perfectly aligns with the ideal AI-generated sequence, subsequently quantifying the repair difficulty as a metric for text detection.
• Extensive evaluations validate that DNA-DetectLLM offers a reliable, efficient, and broadly generalizable solution for AI-generated text detection, with consistent gains across various detection settings.
this section cite: ['b1', 'b0', 'b12', 'b27', 'b15', 'b11', 'b38', 'b10', 'b21', 'b3', 'b13', 'b35', 'b6', 'b26']

Section: Related Works
Detecting AI-generated text is essential for enhancing public trust and preventing misuse, driving growing interest from both academia and industry. Beyond watermarking techniques [20], which embed identifiable markers during generation, current post hoc detection methods are broadly categorized into training-based and training-free methods.
Training-based Methods. Such approaches typically involve training classification models to distinguish between AI-generated and human-written texts. Specifically, early efforts by OpenAI [28] employed RoBERTa-based models for training text classifiers. Subsequently, RADAR [16] introduced adversarial learning to enhance the robustness against paraphrased texts. DeTeCtive [12] utilized multi-level contrastive learning to map texts generated by different LLMs into corresponding feature spaces, classifying them based on similarity metrics. DPIC [39] extracted deep textual features by reconstructing prompts and regenerating texts. Biscope [11] proposed employing a bidirectional cross-entropy loss to extract statistical features for binary classifier training. R-Detect [29] employs a nonparametric kernel relative test to detect AI-generated text, thereby reducing the false positive rate compared to two-sample tests. However, existing research [5,32] indicates that training-based methods consistently overfit to in-distribution features, resulting in poor generalization to out-ofdistribution (OOD) texts. Consequently, researchers have increasingly focused on developing more universally applicable training-free methods.
Training-free Methods. These training-free methods emphasize exploiting probabilistic characteristics of texts, constructing statistical scores based on specific hypotheses, and making decisions according to the comparison of scores against thresholds. For example, LogRank [9], Likelihood [15], and Entropy [17] calculate the average probability ranking, likelihood probabilities, and entropy values to measure the uncertainty of AI-generated texts. DetectGPT [22] pioneered a paradigm that uses perturbations to generate numerous contrast samples to evaluate the overall distribution.
Although methods such as DetectLLM-NPR [30] and DNA-GPT [37] have further developed this paradigm, their efficiency limitations prevent real-time or large-scale detection implementations. Fast-DetectGPT [4] has since updated sampling techniques to compute conditional probability curvature, significantly improving detection efficiency and broadening potential applications. Binoculars [14] achieved state-of-the-art classification performance by calculating cross-perplexity from dual-model perspectives. Lastde++ [36] proposed focusing on local textual features by calculating Diversity Entropy to optimize classification performance.
this section cite: ['b19', 'b27', 'b15', 'b11', 'b38', 'b10', 'b28', 'b4', 'b31', 'b8', 'b14', 'b16', 'b21', 'b29', 'b36', 'b3', 'b13', 'b35']

Section: DNA-DetectLLM

this section cite: []

Section: Preliminary
This study primarily involves two statistical metrics: log-perplexity, which quantifies the average token-level negative log-likelihood under a single model, and cross-perplexity, which captures the average per-token cross-entropy between the probability distributions of two models:
log PPL M1 (s) = - 1 L L i=1 log P M1 (x i |x <i ), log X-PPL M1,M2 (s) = - 1 L L i=1 P M1 (x i |x <i ) log P M2 (x i |x <i ),(1)
〇 AI-generated or human-written text Input text President Trump met with North Korean leader Kim Jong-un in June 2018... Input sequence Let's go ahead and detect it. 𝑥 𝑥 𝑥 𝑥 ① Obtainning Ideal AI sequence Reference Model Input sequence 𝑠 Idea AI sequence 𝑠x 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 … … 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 … … 𝑥 I'll generate it! 𝑥 = argmax ∈ 𝑃 (𝑥 |𝑥 ) ② Mutation Repair Mechanism 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 … … 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 … … 𝑥 𝑥 Repair mutated token Iteratively repair 𝑥 𝑥 …… 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 … … 𝑥 𝑥 The repair is complete. ③ Repair Score- Based Detection Repair score AI-generated Text Human-written Text Let's get result. Repair score > Threshold Repair score ≤ Threshold Threshold Comparison 𝑥 𝑥 𝑥 𝑥 Mutated token Ideal token where s is the input sequence of length L, x i denotes the i-th token, and P M (x i |x <i ) is the conditional probability of x i given its preceding tokens under reference model M 1 or observer model M 2 . Furthermore, their ratio σ(s) has been empirically demonstrated to serve as an effective score for distinguishing AI-generated text [14]. To quantify the effect of local token-level modifications on these metrics, this work introduces the conditional log-perplexity and conditional score:
log PPL M1 (s|s) = - 1 L L i=1 log P M1 (x i |x <i ), σ(s|s) = log PPL M1 (s|s) log X-PPL M1,M2 (s) ,(2)
where s denotes the sequence obtained by modifying tokens in the input sequence s.
this section cite: ['b13']

Section: Overview of DNA-DetectLLM
The entire workflow of DNA-DetectLLM can be summarized in 3 key steps, shown in Figure 2.
this section cite: []

Section: Step 1: Obtaining the ideal AI-generated Sequence.
We construct the ideal AI-generated sequence for a given input by greedily selecting the most probable token at each position.
this section cite: []

Section: Step 2: Mutation Repair Mechanism.
We perform iterative token-level modifications on the input sequence until it fully aligns with the ideal AI-generated sequence.
this section cite: []

Section: Step 3: Repair Score-Based Detection.
We introduce a repair score to quantify the difficulty of the repair, which is compared against a calibrated threshold to determine the detection result.
this section cite: []

Section: Obtaining the ideal AI-generated Sequence
We propose the concept of an ideal AI-generated sequence ŝ, analogous to the error-free template strand in DNA replication, where each token is selected by maximizing the conditional probability at its position:
ŝ = {x 1 , x2 , . . . , xL }, where xi = arg max x∈V P M1 (x|x <i ),(3)
with V denoting the vocabulary and x <i = {x 1 , . . . , x i-1 } representing the preceding i-1 tokens of the input sequence s.
this section cite: []

Section: Mutation Repair Mechanism
We treat non-max probability tokens as mutated tokens and max-probability tokens as ideal tokens.
Analogous to the mutation-repair paradigm in DNA, we propose a mutation repair mechanism aiming to uncover fundamental differences in deviation patterns between AI-generated and human-written texts. Under this mechanism, mutated tokens in the input sequence are iteratively repaired with their ideal tokens step by step, until the input fully aligns with the ideal form:
x i ∈ s = {x 1 , x 2 , . . . , x L } → xi = arg max x∈V P M1 (x|x <i ), if x i ̸ = xi ,(4)
this section cite: []

Section: Repair Score-Based Detection
We introduce the repair score R(s) to quantify the difficulty of the repair process, defined as the average conditional score accumulated throughout the repair trajectory:
R(s) = 1 T + 1 T t=0 σ(s t |s) = T t=0 log PPL M1 (s t |s) (T + 1) log X-PPL M1,M2 (s) ,(5)
where s t is the sequence after t repair steps, and T is the total number of mutated tokens to be corrected.
Human-written texts typically exhibit more substantial mutations, resulting in greater repair difficulty. In contrast, AI-generated texts are generally easier to repair. Accordingly, the detection result for the input sequence is determined as:
D(s) = Human-written Text, R(s) > τ AI-generated Text, R(s) ≤ τ.(6)
this section cite: []

Section: Sensitivity to Repair Order and Score Simplification
Figure 3 shows that different repair orders yield varying repair scores for the same input sequence, due to the unequal impact of each mutated token on the conditional score. Mutated tokens can be broadly categorized into high-and low-probability types. Repairing low-probability tokens typically causes larger shifts in the conditional score, while high-probability tokens lead to smaller changes.
To systematically analyze the influence of repair order, we identify four types of principal repair strategies as follows:
• High-to-low: Repairing high-probability tokens first, followed by low-probability ones, results in a convex "repair curve" with a higher repair score.
• Low-to-high: Repairing low-probability tokens before high-probability ones yields a concave "repair curve" with a lower repair score.
• Sequential Repair: Tokens are repaired in their original order of appearance in the input sequence, regardless of their probability values.
• Random Repair: Tokens are repaired in a randomly chosen order. Averaging the repair scores across multiple random repairs leads to a more stable estimate.
These findings highlight the sensitivity of the repair score to the chosen repair order. Performing multiple random repairs effectively mitigates biases, resulting in a repair score close to the midpoint between the initial and final scores. Consequently, we further derive that the average repair score converges as the number of random repairs N approaches infinity. The derivation is as follows:
Let{δ 1 , δ 2 , . . . , δ T } be a set of non-negative real numbers satisfying T i=1 δ i = σ(s) -σ(ŝ|s). Moreover, δ t = σ(s t-1 |s) -σ(s t |s) quantifies the impact of repairing the current token on the score. Since the influence of each token repair on the conditional log-perplexity is independent and fixed, the set {δ 1 , δ 2 , . . . , δ T } consists of fixed values for a given input sequence, whose order varies depending on the repair strategy. It further follows:
σ(s t |s) = σ(s) - t i=1 δ i for t = 1, 2, . . . , T, where σ(s) = σ(s 0 |s).(7)
For a specific permutation ϕ ∈ [1, N ], the corresponding repair score R ϕ (s) is:
R ϕ (s) = 1 T + 1 T t=0 σ(s) - t i=1 δ ϕ i = σ(s) - 1 T + 1 T i=1 δ ϕ i • (T -i + 1).(8)
Since each token is equally likely to be repaired at random, the expected value of δ ϕ i appearing in the i-th position across all permutations is 1 T (σ(s) -σ(ŝ|s). We further derive the expected value of the repair score as follows:
E[R(s)] = σ(s) - 1 T + 1 T i=1 E δ ϕ i • (T -i + 1) = σ(s) - 1 T + 1 • σ(s) -σ(ŝ|s) T • T (T + 1) 2 . (9
)
Therefore, as the number of random permutations N approaches infinity, the average repair score converges as follows:
lim N →∞ 1 N N n=1 R (n) (s) = lim N →∞ 1 N N n=1 1 T + 1 T t=0 σ (n) (s t |s) = 1 2 (σ(s) + σ(ŝ|s)).(10)
We thus simplify the repair score to R(s) = 1 2 (σ(s) + σ(ŝ|s)), improving detection performance while avoiding intermediate score computations during repair.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. To evaluate performance across diverse domains, we collect 4,800 human-written texts from three representative tasks: news article writing (XSum [23]), story generation (WritingPrompts [8]), and academic writing (Arxiv [24]). For each text, we construct task-specific prompts (see Appendix C) and generate corresponding AI outputs using three advanced LLMs: GPT-4 Turbo, Gemini-2.0 Flash, and Claude-3.7 Sonnet. We further sample 2,000 balanced examples from each of three high-quality detection benchmarks-M4 [33], DetectRL [35], and RealDet [41]-to ensure fair and comprehensive evaluation across real-world scenarios.
Metrics. We adopt the area under the receiver operating characteristic curve (AUROC [18]) and F1 score to evaluate detection performance, where higher values indicate better separability between human-written and AI-generated texts.
Baselines. We compare DNA-DetectLLM with existing training-based and training-free methods. For training-based methods, we include OpenAI-D [28], Biscope [11] and R-Detect [29]. For trainingfree methods, we consider classic zero-shot detectors including Likelihood [15], LogRank [9], and Entropy [17], along with several recent SOTA approaches such as DetectGPT [22], Fast-DetectGPT [4], Binoculars [14], and Lastde++ [36]. More baseline comparisons are provided in Appendix E.
this section cite: ['b22', 'b7', 'b23', 'b32', 'b34', 'b40', 'b17', 'b27', 'b10', 'b28', 'b14', 'b8', 'b16', 'b21', 'b3', 'b13', 'b35']

Section: Implementation details.
In real-world detection scenarios, the source and distribution of textual data are often unknown, constituting an out-of-distribution (OOD) detection problem. To ensure fairness for training-based methods, we exclusively train on the HC3 dataset [10], which is entirely disjoint from the test sets. For training-free methods, the choice of LLM used for scoring can introduce significant performance variation [3]. To eliminate this factor, we standardize the reference (or scoring) model across all methods by employing Falcon-7B-Instruct [25] to compute token generation probabilities. Moreover, Fast-DetectGPT, Binoculars, Lastde++, and DNA-DetectLLM utilize Falcon-7B [25] as the observer (or sampling) model, while DetectGPT uses T5-3B [26]. During testing, the maximum input token length is capped at 1024. More details are in Appendix D.
this section cite: ['b9', 'b2', 'b24', 'b24', 'b25']

Section: Main Results
Table 1 compares the detection performance of DNA-DetectLLM against other baselines across different writing tasks and various generation models. DNA-DetectLLM consistently achieves state-of-the-art performance under all settings, with an average AUROC of 98.30%, representing a relative improvement of 0.93%. Specifically, it yields relative gains of 1.36%, 0.87%, and 0.58% on the XSum, WritingPrompts, and Arxiv datasets, respectively, demonstrating strong crossdomain generalization. This strong generalization can be attributed to DNA-DetectLLM's ability to dynamically capture generation discrepancies between domain-specific text and its ideal AI-generated counterpart through the mutation-repair mechanism, enabling robust identification of human-written versus AI-generated text across diverse domains.
Table 2 evaluates the real-world detection performance of all methods on three high-quality public benchmarks. DNA-DetectLLM demonstrates superior reliability, with average AUROC and F1 score improvements of 5.55% and 2.08%, respectively. Notably, it achieves significant AUROC gains on the challenging DetectRL settings-6.92% on Multi-LLM and 13.92% on Multi-Domain. This improvement can be attributed to the inherent difficulty of DetectRL, where both positive and negative samples may include mixtures of texts with the same label due to the dataset's construction. Such complexity hinders traditional training-free methods that rely on fixed statistical scores. In contrast, DNA-DetectLLM accurately computes repair scores for intricately constructed input texts by aligning them with their respective ideal AI-generated sequences. This flexible repair-based scoring allows for more accurate detection under distributional overlap and ambiguous cases, underscoring the practical utility of DNA-DetectLLM in complex detection scenarios. )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH *377XUER ,QVHUWLRQ )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH 'HOHWLRQ )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH 6XEVWLWXWLRQ )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH 3DUDSKUDVH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH *HPLQL)ODVK )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH &ODXGH6RQQHW )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH )DOVH3RVLWLYH5DWH 7UXH3RVLWLYH5DWH 2SHQ$,' %LVFRSH 5'HWHFW (QWURS\ /LNHOLKRRG /RJ5DQN 'HWHFW*37 )DVW'HWHFW*37 %LQRFXODUV /DVWGH '1$'HWHFW//0 Figure 4: AUROC curves of DNA-DetectLLM and baselines under paraphrasing and editing attacks. 1%. The paraphrasing attacks employed DIPPER [21] to rephrase AI-generated texts. To maintain clean labels, attacks were exclusively applied to AI-generated texts.
The results demonstrate that DNA-DetectLLM exhibits strong robustness against a variety of adversarial attacks. For instance, on GPT-4 Turbo-generated text, our method achieves relative AUROC improvements of 6.65%, 3.17%, 6.62%, and 0.81% under insertion, deletion, substitution, and paraphrasing attacks, respectively. Notably, the improvement is particularly pronounced under low false positive rate (FPR) conditions. We attribute this robustness to the observation that although token-level edits are limited in scope, they can substantially alter the generation probability distribution of the input sequence while having minimal impact on its ideal AI-generated sequence. As a result, DNA-DetectLLM is still able to compute accurate repair scores for reliable detection. Moreover, even under paraphrasing attacks using Dipper, the method effectively captures intrinsic deviations from the ideal sequence and maintains a high AUROC of 97.23%. These findings highlight DNA-DetectLLM's capacity to detect adversarially manipulated AI-generated text, even when such attacks are designed to evade detection.
this section cite: ['b20']

Section: Robustness on Different Lengths
Prior research [4,31] indicates that token length significantly affects detection performance, with shorter texts proving more challenging to detect. We investigate the impact by truncating the input  texts to various target tokens. Figure 5 presents the detection performance across varying lengths for five methods: DNA-DetectLLM, Binoculars, Fast-DetectGPT, Lastde++, and Biscope. Results show that DNA-DetectLLM consistently outperforms all baselines across varying lengths. On GPT-4 Turbo-generated text, it achieves an average AUROC improvement of 2.46%. While all methods benefit from longer inputs, DNA-DetectLLM exhibits a greater advantage on shorter texts. At a token length of 40, it surpasses the second-best method by 3.38%, 1.80%, and 3.27%, respectively. These findings suggest that our method enables detection at shorter lengths by extracting more discriminative features from limited textual input.
this section cite: ['b3', 'b30']

Section: Ablation Studies
We further evaluated the importance of repair order and different base LLMs through two types of ablation experiments. More detailed ablation studies are available in Appendix G.
Repair Score-based Detection under Various Repair Orders. Table 1 compares the detection performance of DNA-DetectLLM under various repair orders. When the repair order is changed to High-to-low, Low-to-high, or Sequential Repair, a slight performance drop is observed. Although these strategies still outperform other baselines, they require recalculating the conditional score after each mutated token repair, resulting in significantly increased computational cost. In contrast, the simplified repair score (R(s) = 1 2 (σ(s) + σ(ŝ|s))) maintains strong performance while improving efficiency by an order of magnitude, highlighting its necessity in practical deployment.
DNA-DetectLLM's Performance with Different M 1 and M 2 . Figure 6 evaluates four different LLM combinations: "Falcon-7B-Instruct + Falcon-7B", "Llama-3-8B-Instruct + Llama-3-8B", "Mistral-7B-Instruct + Mistral-7B", and "Llama-2-7B + Llama-7B". Results demonstrate that any of these combinations significantly outperform existing baselines, with an average performance improvement of 15.28%. Interestingly, the combination "Llama-2-7B + Llama-7B" slightly exceeds the default combination "Falcon-7B-Instruct + Falcon-7B" used in our main experiments, achieving AUROC of 92.4% and 90.7%. These findings highlight the inherent effectiveness of DNA-DetectLLM, suggesting its robust detection performance is not reliant on any specific LLM combination, with potential for further enhancement through better LLM pairings.
this section cite: []

Section: Efficiency Analysis
Efficiency is critical for AI-generated text detection, as slow detection speeds hinder large-scale or real-time monitoring in practical scenarios. Figure 7 illustrates the average processing time per sample for each method. To eliminate the confounding factor of text length, we randomly sampled 1,000 long texts from the RealDet dataset, truncated them to 300 tokens, and measured average detection cost with a batch size of 1. We observe that training-based methods such as Biscope and OpenAI-D were the fastest, requiring less than 0.1s per text, but these methods entail significant training overhead. Among training-free methods, classical methods like Likelihood, Logrank, and Entropy are faster, with inference times around 0.3s, but their detection accuracy did not meet our requirements. DNA-DetectLLM, Binoculars, and Fast-DetectGPT processed each sample in 0.8s, with DNA-DetectLLM achieving the better detection performance.
this section cite: []

Section: Conclusion
In this paper, we introduce DNA-DetectLLM, a novel zero-shot AI-generated text detection method via a DNA-inspired mutation-repair paradigm. Extensive experiments demonstrate that DNA-DetectLLM consistently achieves SOTA detection performance while exhibiting strong robustness across diverse scenarios. We hope our work offers new insights and perspectives for AI-generated text detection and plan to further explore the mutation-repair paradigm to enhance detection performance.
this section cite: []

Section: References
Ref_id:b0 Title: Generating sentiment-preserving fake online reviews using neural language models and their human-and machine-based detection Year: (2019)
Ref_id:b1 Title: Detecting fake news using machine learning : A systematic literature review Year: (2021)
Ref_id:b2 Title: Glimpse: Enabling white-box methods to use proprietary models for zero-shot LLM-generated text detection Year: (2025)
Ref_id:b3 Title: Fast-detectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature Year: (2024)
Ref_id:b4 Title: On the possibilities of ai-generated text detection Year: (2023)
Ref_id:b5 Title: Imitate before detect: Aligning machine stylistic preference for machine-revised text detection Year: (2025-04)
Ref_id:b6 Title: RAID: A shared benchmark for robust evaluation of machine-generated text detectors Year: (2024-08)
Ref_id:b7 Title: Hierarchical neural story generation Year: (2018-07)
Ref_id:b8 Title: GLTR: Statistical detection and visualization of generated text Year: (2019-07)
Ref_id:b9 Title:  Year: (2023)
Ref_id:b10 Title: Biscope: Ai-generated text detection by checking memorization of preceding tokens Year: (2024)
Ref_id:b11 Title: Detective: Detecting ai-generated text via multi-level contrastive learning Year: (2024)
Ref_id:b12 Title: Robust spammer detection using collaborative neural network in internet-of-things applications Year: (2021)
Ref_id:b13 Title: Spotting LLMs with binoculars: Zero-shot detection of machine-generated text Year: (2024)
Ref_id:b14 Title: Unifying human and statistical evaluation for natural language generation Year: (2019-06)
Ref_id:b15 Title: Radar: Robust ai-text detection via adversarial learning Year: (2023)
Ref_id:b16 Title: Automatic detection of generated text is easiest when humans are fooled Year: (2020-07)
Ref_id:b17 Title: Insights into the area under the receiver operating characteristic curve (auc) as a discrimination measure in species distribution modelling Year: (2012)
Ref_id:b18 Title: PubMedQA: A dataset for biomedical research question answering Year: (2019-11)
Ref_id:b19 Title: A watermark for large language models Year: (2023)
Ref_id:b20 Title: Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense Year: (2023)
Ref_id:b21 Title: DetectGPT: Zero-shot machine-generated text detection using probability curvature Year: (2023-07)
Ref_id:b22 Title: Don't give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization Year: (2018-11)
Ref_id:b23 Title:  Year: (2021)
Ref_id:b24 Title: The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only Year: (2023)
Ref_id:b25 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b26 Title: Can ai-generated text be reliably detected? Year: (2025)
Ref_id:b27 Title: Release strategies and the social impacts of language models Year: (2019)
Ref_id:b28 Title: Deep kernel relative test for machine-generated text detection Year: (2025)
Ref_id:b29 Title: Detectllm: Leveraging log rank information for zero-shot detection of machine-generated text Year: (2023)
Ref_id:b30 Title: Multiscale positive-unlabeled detection of AI-generated texts Year: (2024)
Ref_id:b31 Title: Authorship attribution for neural text generation Year: (2020-11)
Ref_id:b32 Title: Multi-generator, multi-domain, and multi-lingual black-box machine-generated text detection Year: (2024-03)
Ref_id:b33 Title: Who wrote this? the key to zero-shot LLM-generated text detection is GECScore Year: (2025-01)
Ref_id:b34 Title: DetectRL: Benchmarking LLM-generated text detection in real-world scenarios Year: (2024)
Ref_id:b35 Title: Training-free llm-generated text detection by mining token probability sequences Year: (2024)
Ref_id:b36 Title: Dna-gpt: Divergent n-gram analysis for training-free detection of gpt-generated text Year: (2024)
Ref_id:b37 Title: DNA-GPT: Divergent n-gram analysis for training-free detection of GPT-generated text Year: (2024)
Ref_id:b38 Title: Dpic: Decoupling prompt and intrinsic characteristics for llm generated text detection Year: (2024)
Ref_id:b39 Title: Beat LLMs at their own game: Zero-shot LLMgenerated text detection via querying ChatGPT Year: (2023-12)
Ref_id:b40 Title: Reliably bounding false positives: A zero-shot machine-generated text detection framework via multiscaled conformal prediction Year: (2025)
