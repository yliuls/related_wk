Title: LEARNING TO SEE BEFORE SEEING: DEMYSTIFY-ING LLM VISUAL PRIORS FROM LANGUAGE PRE-TRAINING
Abstract: Large Language Models (LLMs), despite being trained on text alone, surprisingly develop rich visual priors. These priors allow latent visual capabilities to be unlocked for vision tasks with a relatively small amount of multimodal data, andto perform symbolic visual generation tasks without ever having seen an image. Through systematic analysis, we reveal that visual priors-the implicit, emergent knowledge about the visual world acquired during language pre-training-are composed of separable perception and reasoning priors with unique scaling trends and origins. We show that an LLM's latent visual reasoning ability is predominantly developed by pre-training on reasoning-centric data (e.g., code, math, academia) and scales progressively. This reasoning prior acquired from language pre-training is transferable and universally applicable to visual reasoning. In contrast, the perception prior emerges more diffusely from broad corpora, and perception ability is more sensitive to the vision encoder and visual instruction tuning data. In parallel, text describing the visual world proves crucial, though its performance impact saturates rapidly. Leveraging these insights, we propose a data-centric recipe for pre-training vision-aware LLMs and verify it in 1T token scale pretraining. Our findings are grounded in over 100 controlled experiments consuming 500,000 GPU-hours, spanning the full MLLM construction pipeline-from LLM pre-training to visual alignment and supervised multimodal fine-tuning-across five model scales, a wide range of data categories and mixtures, and multiple adaptation setups. Along with our main findings, we also propose and investigate several hypotheses, and introduce a Multi-Level Existence Bench (MLE-Bench) to facilitate future research. Together, this work provides a new way of deliberately cultivating visual priors from language pre-training, paving the way for the next generation of multimodal LLMs. We recommend a visit to our project page for an interactive reading.

Section: INTRODUCTION
A compelling phenomenon has emerged at the forefront of AI research: Large Language Models (LLMs), despite being trained exclusively on vast corpora of text, appear to develop profound priors about the visual world. This latent capability is paradoxical, suggesting that the statistical patterns within language might be rich enough to encode fundamental principles of vision, from object properties to spatial relationships, without ever observing a single image. This emergent visual prior presents in several surprising and powerful ways:
(1) Programmatic visual knowledge: LLMs possess a rich visual knowledge, enabling them to generate executable code that renders complex 2D and 3D scenes, from objects to spatial layouts (Sharma et al., 2024;Ge et al., 2025;Ashutosh et al., 2025). This demonstrates a grasp of visual concepts without ever seeing a single image. The resulting synthetic data is of sufficient quality to pre-train standard vision models for successful generalization to real-world images (Sharma et al., 2024).
(2) Data-efficient visual adaptation: LLMs are highly efficient for visual adaptation. With a vision encoder, high-level reasoning emerges from instruction tuning on a small scale of image-text pairs, bypassing the need for massive multimodal pretraining (Alayrac et al., 2022;Liu et al., 2023a;Li et al., 2023;Grattafiori et al., 2024;Tong et al., 2024a;Bai et al., 2025b). This data-efficient instruction tuning extends to unified models, where visual generation is unlocked with minimal data (Tong et al., 2024b). Furthermore, this efficiency enables adaptation to low-level visual tasks using vision-only data (Zheng et al., 2024;Du et al., 2025), proving that an LLM's reasoning framework can function independently of cross-modal alignment.
(3) LLMs as strong vision encoders: The learned representations of LLMs can directly benefit pure vision tasks without language (Kumar et al., 2024;Pang et al., 2024;Lai et al., 2024;Bai et al., 2025a). When repurposed as visual encoders, the transformer layers of LLMs offer competitive performance on image classification, segmentation, and video understanding, even surpassing visionspecific backbones (Pang et al., 2024). These findings suggest that the hierarchical abstraction and long-range dependency modeling intrinsic to LLMs are not modality-specific, but rather capture general-purpose computational motifs that are well-suited to processing visual signals. This is also shown in neuron-level studies, which have identified multimodal neurons within LLMs that respond to the same abstract concept regardless of whether it is presented through text or vision (Schwettmann et al., 2023;Pan et al., 2023;Verma et al., 2024).
Collectively, these phenomena are not isolated curiosities; they point toward a deeper principle of representation learning. They lend strong empirical support to the Platonic Representation Hypothesis (Huh et al., 2024b;Jha et al., 2025), which posits that as models scale across diverse data and tasks, their latent representations-whether trained on text or images-converge toward a shared, underlying statistical model of reality. In this view, text and images are different "projections" or "shadows" of the world, and a powerful enough model can learn the structure of the world itself from any single projection. The visual priors in LLMs, therefore, may be a direct consequence of them recovering this unified internal world model from text alone.
These observations motivate a systematic investigation into the visual priors that LLMs acquire from language pre-training. Rather than a hand-crafted bias or a Bayesian prior distribution, we frame visual priors as implicit knowledge or prior vision capabilities encoded in LLMs, whose primary effect is to grant both enhanced capability for vision tasks and greater ease of transfer to vision. We seek to determine their origins, dissect whether they form a uniform block of knowledge or are composed of distinct, separable abilities, and explore how they can be leveraged to build more capable MLLMs.
Our work presents the first systematic investigation into the nature and origins of visual priors in the pre-training of LLMs and shows three key contributions: (1): Structure of visual priors. We establish that visual priors can be decomposed into perceptual and reasoning components. (2): Source of visual priors. We identify that the model's latent visual reasoning is predominantly cultivated by and scales progressively with reasoning-centric data, whereas its perception ability emerges more diffusely from broad, diverse data. (3): Vision-aware language pre-training. We propose a pre-training data-mixing strategy that strategically balances reasoning-centric and visually descriptive text to deliberately cultivate powerful visual priors for training LLMs that can result in stronger multimodal performance.
this section cite: ['b79', 'b28', 'b5', 'b79', 'b1', 'b45', 'b31', 'b106', 'b23', 'b68', 'b43', 'b68', 'b78', 'b58', 'b91', 'b39', 'b40']

Section: General Knowledge OCR & Chart
Is the dirty ground made of plastic?
When a lightning flash of mean current I and time t occurs, the electric field strength between the cloud and Earth is reduced by how much?
What brand name is visible in the white text inside the green square?
Which one picture follows the same pattern established by the previous pictures?
Vision-Centric 0 20 40 60 80 100 Data (B) 27 28 29 30 31 32 33 34 35 Accuracy AVG VQA 0 20 40 60 80 100 Data (B) 34 35 36 37 38 39 40 General VQA 0 20 40 60 80 100 Data (B) 23 24 25 26 27 28 Knowledge VQA 0 20 40 60 80 100 Data (B) 4 6 8 10 12 14 16 18 20 22 OCR & Chart VQA 0 20 40 60 80 100 Data (B) 37 38 39 40 41 42 43 44 45 Vision-Centric VQA Model sizes 340M 1B 3B 7B 13B Figure 2: Impact of model and data sizes. The plots illustrate the performance of MLLMs, built upon LLMs of five different sizes (340M to 13B parameters), as a function of the amount of web-crawl pre-training data (0B to 100B tokens). The general trend shows that performance improves with both increasing model size and data volume, but the scaling behavior differs across task categories.
on 16 benchmarks grouped into four categories: General, Knowledge, OCR & Chart VQA, and Vision-Centric. Figure 1 presents example questions representing each category.
Additionally, we measure the representational alignment between language and vision modalities using a kernel-based similarity metric (Huh et al., 2024a).
this section cite: []

Section: DEMYSTIFYING LLM VISUAL PRIORS: STUDIES AND FINDINGS
This section presents our main results and findings. We first conduct a series of controlled experiments to systematically deconstruct the origins of LLM visual priors. Each subsection details its specific experimental setup or analytical approach, followed by the results and key findings.
this section cite: []

Section: IMPACT OF MODEL AND DATA SIZES.
Finding 1: VQA performance scales positively with model and data size. However, this scaling is not uniform across all visual abilities. 32.1 32.0 31.5 31.3 31.1 31.1 30.7 30.7 30.7 30.5 30.5 30.3 30.2 30.1 29.9 29.8 AVG VQA co de ar ts fo od we b-cr aw l m at h ac ad em ia ph ilo so ph y la w ec on om ics po lit ics lit er at ur e co m pu te r sc i en cy clo pe di a q-a fo ru m m ed ici ne bi ol og y 35 36 37 38 39 General VQA co de ar ts fo od we b-cr aw l m at h ac ad em ia ph ilo so ph y la w ec on om ics po lit ics lit er at ur e co m pu te r sc i en cy clo pe di a q-a fo ru m m ed ici ne bi ol og y 22 23 24 25 26 27 28 29 Knowledge VQA co de ar ts fo od we b-cr aw l m at h ac ad em ia ph ilo so ph y la w ec on om ics po lit ics lit er at ur e co m pu te r sc i en cy clo pe di a q-a fo ru m m ed ici ne bi ol og y 8 10 12 14 16 18 20 OCR & Chart VQA co de ar ts fo od we b-cr aw l m at h ac ad em ia ph ilo so ph y la w ec on om ics po lit ics lit er at ur e co m pu te r sc i en cy clo pe di a q-a fo ru m m ed ici ne bi ol og y 36 39 42 45 Vision-Centric VQA We begin our analysis by investigating the fundamental impact of scale. To study how model size and pre-training data volume influence downstream multimodal capabilities, we perform a set of experiments to pre-train five LLMs of varying sizes (340M, 1B, 3B, 7B, and 13B parameters). Each model size was trained on eight different scales of data, ranging from 0B to 100B tokens. The training dataset is web-crawl for all experiments.
As illustrated in Figure 2, both model sizes and pre-training data sizes generally lead to stronger downstream multimodal performance. This holds true for the overall average VQA. However, a closer look at the different VQA categories reveals significant nuances. Performance on General VQA and Knowledge VQA demonstrates a similar scaling trend, consistently improving with both model and data size. In sharp contrast, OCR & Chart VQA is far more sensitive to model size than data volume; the performance gap between models is significantly wider. Meanwhile, Vision-Centric VQA also presents a unique pattern where the largest models benefit disproportionately from more data, while smaller models plateau much earlier. These divergent scaling patterns across different abilities demonstrate different visual abilities do not scale uniformly, but instead possess different properties that govern how they benefit from increased model and data size.
this section cite: []

Section: IMPACT OF PRE-TRAINING DATA SOURCES.
Finding 2: Specific categories of language pre-training data can enhance certain visual capabilities in the resulting MLLM.
Having characterized the effects of data and model scales, we now transition our analysis to the composition of the data itself. To investigate the role of different pre-training sources, we fix the model size to 3B parameters and the total training data volume to 30B tokens. We then pre-train 16 distinct models, each trained exclusively on data from one of the 16 sources outlined in our pre-training sources (e.g., academia, biology, code, etc.). This setup allows us to attribute performance variations directly to the specific data source used for pre-training.
As illustrated in Figure 3, the results reveal a significant variance in downstream multimodal performance depending on the pre-training data source. This divergence suggests that different categories of text data contribute to distinct and non-uniform visual priors. Notably, strong performance on Vision-Centric VQA tasks is highly correlated with two types of data: reasoning-centric (e.g., code,
0% 25% 50% 75% 100% Proportions (%) 30.0 30.5 31.0 31.5 32.0 32.5 33.0 Accuracy AVG VQA (Reasoning) 0% 25% 50% 75% 100% Proportions (%) 39 40 41 42 43 44 45 Vision-Centric VQA (Reasoning) 0% 25% 50% 75% 100% Proportions (%) 30.0 30.5 31.0 31.5 32.0 32.5 33.0 AVG VQA (Visual) 0% 25% 50% 75% 100% Proportions (%) 38 39 40 41 42 43 44 45 Vision-Centric VQA (Visual) code reasoning math reasoning science reasoning reasoning combination optimal point visual concept visual attribute visual relationship visual combination mathematics, academia) and corpora rich in visual world descriptions (e.g., arts, food).
The top-performing models in Vision-Centric VQA, all scoring above 42%, are trained on these specific sources. This finding motivates a more granular analysis in the next section.
this section cite: []

Section: IMPACT OF REASONING AND VISUAL DATA CATEGORIES AND PROPORTIONS.
Finding 3: A small amount of data about the visual world is crucial, but its contribution saturates quickly; in contrast, increasing the proportion of reasoning data in the pre-training mix progressively enhances visual abilities, with gains observed up to a 75% ratio.
Findings in the previous section show that reasoning-centric categories and categories related to the visual world were the most potent drivers of downstream visual capabilities. To dissect this phenomenon further, we focus our next set of experiments specifically on these two domains.
The reasoning-centric data was partitioned into code reasoning, math reasoning, science reasoning, and a reasoning combination category, which aggregates the three aforementioned categories. Concurrently, we define four categories for data related to the visual world: visual concept: Text naming visual entities like objects, people, places, and scenes. visual attribute: Descriptions of visual properties such as color, shape, texture, and style. visual relationship: Language detailing spatial arrangements or part-whole connections. visual combination: A combination of all three visual categories.
We begin by creating a data pool of approximately 300B tokens, comprising all sources used in Section 3.2. To partition this corpus into these categories, we employ a 32-B LLM (Yang et al., 2025a) to classify the text into finer-grained visual world and reasoning categories. Detailed classification settings and results are presented in the Appendix F.
With this fine-level categorization, we conduct a series of controlled mixing experiments to study how varying the proportion (mixing ratio) of these data types affects the final MLLM's performance.
For each category, we train five separate models, systematically varying its proportion in the data mixture to 0%, 25%, 50%, 75%, and 100%. The remainder of the data for each run is drawn from a proportional mix of all other data within the 300B token pool, ensuring the total training volume is held constant at 30B tokens.
As shown in Figure 4, the results reveal a critical divergence in how visual world and reasoning data categories contribute to visual priors. The impact of reasoning-centric data is profound and progressive, with performance scaling steadily up to a 75% proportion. The contribution from data explicitly describing the visual world saturates quickly; a small initial amount seems to be crucial, but further increases yield diminishing returns.
this section cite: []

Section: DERIVING A DATA MIXTURE FOR MORE VISION-AWARE LLMS.
Finding 4: Maximizing MLLM VQA performance is best achieved by pre-training on a data mixture heavily skewed towards reasoning-centric content but with necessary vision world knowledge. The balance point between language and vision proficiency is reached via a calibrated data mixture between language-favorable and vision-favorable.
Data Ratio Avg VQA Data Ratio Avg VQA rea vis rea vis 50 5 30.7 55 5 30.9 10 31.3 10 31.7 15 31.8 15 32.2 60 5 31.9 65 5 32.0 10 32.4 10 32.2 15 32.7 15 32.5 20 32.5 20 32.1 25 32.4 25 31.9 30 31.6 30 31.4 70 5 31.9 75 5 31.6 10 32.3 10 31.5 15 32.6 15 32.4 80 5 31.5 85 5 31.2 10 32.4 10 31.6 15 32.2 15 31.8 Table 1: Grid Search for a vision-favorable data mixture. Results from pre-training a 3B parameter LLM on 30 distinct data blends, each totaling 30B tokens. The table explores how varying the proportions of reasoning-centric (rea) and visual-world (vis) data affects various capabilities.
Building on these findings, our objective is to derive a single, practical data mixture that not only excels on language tasks but also serves as a powerful foundation for MLLMs. Our approach proceeds in three stages. First, we determine a vision-favorable blend using our 300B token pool to establish a target. For the subsequent, more practical stages of our analysis, we then narrow our focus to six primary data categories: web-crawl, encyclopedia, academia, literature, math, and code. Within this practical set of sources, we then identify a language-favorable mixture (second), and finally, derive a balanced mixture by interpolating between these two optima (third). Since our goal also involves downstream VQA performance, we adapt the most direct approach of using a grid search over mixture ratios.
Vision-favorable mixture. We first aim to identify a data mixture that excels at visual tasks. To do so, we conduct a grid search over the proportions of reasoning-centric and visual-world data drawn from our 300B token pool. Specifically, we perform a grid search across 24 data blends constructed by sampling from a space where the reasoning combination ranges from 50% to 85% and the visual combination ranges from 5% to 30%, following the conclusions drawn from Section 3.3. The comprehensive results of this search are presented in Table 1.
From this search, we find that the best-performing models for downstream MLLM tasks emerge from a mixture containing approximately 60% reasoning and 15% visual content. Results show a powerful visual foundation is not built by simply maximizing exposure to visual descriptions, but by establishing a strong reasoning faculty, which is then grounded by a smaller amount of visual world knowledge. This experiment is performed on our 300B token pool, provides a target ratio for maximizing VQA performance. In the next part, we proceed to a more practical experiment at the data source level, guided by this result.
Language-favorable mixture. We begin by establishing a language-favorable mixture that achieves the best performance on our language task suite. Guided by recent literature (Shukor et al., 2025a;Ge et al., 2024;Ye et al., 2024) and empirical testing over 10 experiments, we identify this as a mix of 50% web-crawl, 2.5% encyclopedia, 2.5% academia, 20% literature, 5% math, and 20% code. This blend, designated as mix0 in Table 2, serves as our baseline for strong language proficiency, achieving the highest text accuracy (53.0%) and the best perplexity (13.46).
this section cite: ['b27', 'b100']

Section: Balanced mixture.
To reconcile these two objectives, we seek a single, balanced mixture that offers strong performance across both modalities. We achieve this by performing a series of interpolation experiments, detailed as mix0 through mix10. We shift the data composition from our languagefavorable baseline (mix0) towards an endpoint representing the vision-favorable blend (approximated by mix9 and mix10). To ensure stabilized results, each model in this series is trained for 50B tokens.
Recipe Data Source Mixture (%) Performance Metrics Overall Rank web-crawl encyclopedia academic literature math code reasoning visual t-acc (%) ppl (↓) v-acc (%) mix0 language ↑ 50.0 2.5 2.5 20.0 5.0 20.0 33.1 21.7 53.0 13.46 32.4 5 mix1 48.3 3.4 2.9 17.0 5.8 22.5 36.2 20.6 52.8 13.48 32.4 4 mix2 46.7 4.3 3.3 14.0 6.7 25.0 39.4 19.4 52.6 13.51 32.6 8 mix3 45.0 5.2 3.8 11.0 7.5 27.5 42.6 18.2 52.5 13.56 32.9 9 mix4 43.3 6.1 4.2 8.0 8.3 30.0 45.7 17.1 52.4 13.62 32.7 10 mix5 ↓ vision 41.7 7.1 4.6 5.0 9.2 32.5 48.9 16.0 52.6 13.57 33.0 6 mix6 40.0 8.0 5.0 2.0 10.0 35.0 52.0 14.8 52.7 13.52 33.3 1 mix7 36.5 7.0 7.5 2.0 11.5 35.5 55.5 14.4 52.5 13.56 33.1 3 mix8 33.0 6.5 9.5 2.0 12.0 37.0 57.2 14.0 52.7 13.52 33.2 2 mix9 29.5 6.0 11.5 2.0 12.5 38.5 59.0 13.6 52.3 13.71 33.2 7 mix10 26.0 5.5 12.5 2.0 13.0 41.0 61.3 13.3 52.1 13.88 33.4 11
Table 2: Deriving a data mixture for more vision-aware LLMs. This table details a series of 11 data mixtures, from mix0 (language-favorable blend) to mix10 (approximating the vision-favorable blend), all trained on a 3B-parameter LLM with 50B tokens. The results highlight a trade-off, with mix6 emerging as the most balanced mixture, achieving top-ranked overall performance by improving visual capabilities without a significant drop in language proficiency.
The performance metrics in Table 2 reveal the expected trade-off: as the mixture becomes more reasoning-centric, vision accuracy (v-acc) generally improve, while language proficiency (t-acc and ppl) shows a slight decline. Our analysis identifies mix6 as the balanced mixture, achieving the highest overall rank. Mixtures in its vicinity (e.g., mix5, mix7, mix8) also achieve high rankings. This demonstrates that a carefully calibrated data mixture can cultivate powerful visual priors without substantially compromising core language abilities.
this section cite: []

Section: THE STRUCTURE AND ORIGIN OF LEARNED VISUAL PRIORS.
Finding 5: The learned visual prior is not a single entity but decomposes into at least a perception prior and a reasoning prior with different origins.
We now synthesize our previous results to investigate the internal structure of the visual prior. We conceptualize the visual prior as a collection of distinct abilities, each measured by one of our four VQA categories.
Internal structure of visual priors. We aggregate the performance data across all 105 3B models from our previous experiments, encompassing variations in data sources, mixing ratios, and training scales. We then compute the Spearman correlation matrix across the four VQA performance categories to identify which abilities scale together and which diverge. The results in Figure 5 suggest a potential internal structure within the visual prior, hinting at a separation into at least two distinct types of abilities. We observe a moderate correlation (0.37) between General and OCR performance. This connection seems to point towards a perception prior, as success in both categories relies heavily on the model's perceptual acuity-the ability to accurately process raw visual input-rather than complex, multi-step reasoning.
In contrast, we find another moderate correlation (0.33) between the Knowledge and Vision-Centric tasks. This link appears to emerge because both categories often require abstract inference that goes beyond simple perception. For instance, the Knowledge category demands multistep reasoning to solve complex scientific or mathematical problems, while Vision-Centric tasks include challenges like visual IQ puzzles, object counting, and correspondence matching, necessitate a blend of perception and reasoning, often with a heavier reliance on the latter. The correlation matrix also reveals very weak, or even slightly negative, correlations between these two 0% 25% 50% 75%  The plots show the VQA performance of MLLMs built using three distinct vision encoders based on the proportion of reasoning-centric data used in the LLM's pre-training mix. Despite differences in their absolute performance, all three configurations show a consistent improvement on reasoning-heavy tasks as the LLM's reasoning pre-training proportion increases, similar to trends observed before.
groups (perception-heavy vs. reasoning-heavy)foot_0 . This lack of a strong positive correlation raises the possibility that these are largely independent abilities, potentially stemming from loosely-coupled priors within the LLM's representation. Our observations on the separability of these visual priors in MLLMs align with and extend the findings of recent research Chen et al. (2025), which identified a similar dissociation through parameter merging.
Different origins of priors. The statistical independence of these two priors implies they are cultivated through different mechanisms. As our analysis in Section 3.2 and Section 3.3 demonstrated, the reasoning prior is from reasoning-centric data and can be predictably enhanced by increasing the proportion of reasoning-centric data.
In contrast, the origins of the perception prior appear more diffuse. A signal comes from our singlesource experiments (Section 3.2), where web-crawl data yields the best performance on General and OCR tasks. However, web-crawl is an extremely general category, and no other, more specific data category consistently boosts perceptual abilities. This mixed effect suggests the perception prior may be a general byproduct of large-scale language modeling, emerging from the sheer diversity of language rather than a specific category.
To further investigate this perception prior and characterize its properties more directly, we introduce a multi-level existence benchmark (MLE-Bench) designed to assess pure perception abilities across multiple levels. In Section A.1, we use this benchmark to test our hypothesis that the perception prior is scale-dependent and is most pronounced for small and medium-sized objects. Further details on the benchmark's construction are presented in Appendix I. We also present an evaluation of some MLLMs on MLE-Bench, revealing that they exhibit varying levels of ability in perceiving objects of different sizes. Detailed results and analysis are available in Appendix J.
this section cite: ['b85']

Section: DECONSTRUCTING MULTIMODAL ABILITIES: VISION OR LANGUAGE.
Finding 6: Visual reasoning is primarily shaped by reasoning prior acquired from language pre-training; perception is more dependent on post-training (visual instruction tuning).
Here, we conduct further analysis to first verify the universality of learned visual priors and then deconstruct the source of different multimodal abilities, distinguishing between those inherited more from the LLM and those acquired more from the visual instruction tuning.
this section cite: []

Section: Universality of the learned visual priors.
To test the general influence of the visual prior, we apply two more vision encoders (DINOv2-G (Oquab et al., 2023) and MAE-H (He et al., 2022)) other than our default MetaCLIP-B/16. We pair these with LLMs pre-trained on varying proportions of our reasoning combination data category, from 0% to 100%.
AV G VQ A Ge ne ral VQ A Kn ow led ge VQ A OC R & Ch art VQ A Vis ion -Ce ntr ic VQ A 90 92 94 96 98 100 Remaining performance (%) 98.05 98.66 99.46 94.53 98.11 95.85 95.34 98.60 90.92 96.69 98.78 98.79 99.26 98.59 98.55 97.35 97.55 98.02 96.05 97.31 Relative performance after data removal 100% Baseline Perception (50% remaining data) Perception (0% remaining data) Reasoning (50% remaining data) Reasoning (0% remaining data) The chart shows the remaining performance (%) on Avg VQA and per-category VQA (x-axis) relative to a 100% baseline. The bars show performance after ablating perception or reasoning instruction data in stages (removing 50% and then 100% of the data).
As illustrated in Figure 6, the results reveal a dual-faceted pattern. Firstly, they confirm the universality of the reasoning prior. For reasoning-heavy tasks, all three vision encoder configurations exhibit a nearly identical, strong upward trend in performance as the proportion of reasoning data in the LLM's pre-training increases. This demonstrates that the visual reasoning prior cultivated in the LLM is a foundational, modality-agnostic prior that benefits the multimodal system regardless of the specific vision encoder used. In contrast, the perception prior lacks this universality. The performance trends for perception-oriented tasks are more inconsistent across the different vision encoders.
Source of abilities, from visual priors or visual instruction tuning. Second, we conduct targeted studies to determine whether perception and reasoning skills originate primarily from the LLM's visual priors or the subsequent visual instruction tuning stage. We use an MLLM to classify our Cambrian-7M dataset that contains 5M text-image pairs into these two categories, resulting in 1.8M perception and 0.6M reasoning data, and the remaining 2.6M data as others. Further classification details are provided in the Appendix G.
We partition our instruction-tuning data into perception, reasoning, and other categories and trained five tuning configurations that ablate perception and reasoning data in stages (100% → 50% → 0%) while leaving other data unchanged. The results, presented in Figure 7, show two observations: (1) reducing perception-targeted tuning produces the largest performance drops on perception-heavy benchmarks and modest drops on reasoning tasks; (2) removing reasoning-targeted tuning causes only small incremental drops on perception tasks and modest drops on reasoning tasks.
Together, results in this section show two mechanisms. First, the LLM encodes a robust, transferable visual reasoning prior primarily via language pre-training; this prior benefits reasoning-centered VQA across different vision encoders. Second, perception performance depends more on vision-encoder characteristics and on subsequent supervised visual instruction tuning.
In addition to our main findings, we investigate the universality of reasoning priors acquired during pre-training. Our experiments show that this reasoning ability is highly transferable and modalityagnostic, as detailed in Appendix A.2. Furthermore, in Appendix A.3, we study whether language data structures drive representational alignment with vision, finding that they can indeed partially drive this alignment.
this section cite: ['b66', 'b34']

Section: SCALING UP AND TRAINING A VISION-AWARE LLM

this section cite: []

Section: SETTINGS AND MODELS
Building upon our findings, we scale up our approach to validate our findings and develop a visionaware LLM on a larger-scale. The goal is to test whether the principles identified in our controlled, smaller-scale studies hold true when applied to larger training runs. To this end, we pre-train two 7B parameter LLMs, each on 1T tokens, based on the two data mixtures identified previously: Language-favorable model: Following the mix0 mixture, which is the best-performing blend for pure language tasks. Balanced model: Based on the mix6 recipe, our proposed balanced mixture is designed to deliberately cultivate strong visual priors without compromising language proficiency.
Model Language Vision ppl avg acc General Knowledge OCR&Chart QA Vision-Centric Overall Language-Favorable 8.72 0.647 46.92 28.35 21.49 46.31 37.32 Balanced 7.49 0.655 49.59 29.02 23.63 46.59 38.64
Table 3: Performance comparisons of the Language-Favorable and Balanced models across both language and vision-language benchmarks. The table summarizes key language metrics (perplexity and accuracy) and provides average scores for a suite of vision tasks.
this section cite: []

Section: RESULTS
As shown in Table 3, the Balanced model, pre-trained with balanced recipe, exhibits competitive language proficiency. Notably, it achieves a lower (better) average perplexity of 7.49 compared to the Language-favorable model's 8.72, while also maintaining a slightly higher average accuracy (0.655 vs. 0.647). An interesting dynamic observed during pre-training was that the Balanced model's language performance initially lagged behind the Language-favorable model, beginning to surpass it after approximately 600B tokens. This may suggest that when the pre-training token volume is sufficiently large, the benefits from reasoning-related tokens can be more effectively unleashed when grounded in a substantial amount of world knowledge.
On VQA benchmarks detailed, the Balanced model consistently outperforms the Language-Favorable model in most of the benchmarks, achieving a higher overall VQA average (38.64 vs. 37.32). This confirms that the deliberate pre-training on a data mixture rich in reasoning and visual world text successfully imbues the LLM with stronger visual priors in a larger scale.
We also introduce Blind Visual Instruction Tuning, a trick that serves the MLLM community as both a practical tool for visual adaptation and a probe for revealing how models "hack" visual tasks with language (see Appendix K). Together with this trick, we find that many MLLMs, even state-of-the-art ones, fail to recognize the absence of an image input. This leads to hallucinations in visual question answering scenarios without a visual context. A study of this behavior is presented in Appendix L.
this section cite: []

Section: CONCLUSION
This work has undertaken a systematic deconstruction of the visual priors that LLMs acquire from text-only pre-training. Through a series of controlled experiments manipulating data composition, we moved beyond observing the phenomenon of vision priors to interrogating its fundamental drivers.
Our investigation provides a data-centric roadmap for developing multimodal systems, shifting the paradigm from serendipitous emergence to the deliberate cultivation of visual capabilities.
Looking forward, we hope this research encourages a paradigm where LLM development is more considerate of vision and multimodality, prompting the cultivation of visual priors from the earliest stages of pre-training. We also hope it inspires a deeper investigation into the fundamental correlations between cross-modal representations, contributing to a more unified understanding of how knowledge is structured across modalities.
this section cite: []

Section: References
Ref_id:b0 Title: Llama 3 model card Year: (2024)
Ref_id:b1 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b2 Title: Efficient online data mixing for language model pre-training Year: (2023)
Ref_id:b3 Title: Icml 2024 tutorial: Physics of language models Year: (2024)
Ref_id:b4 Title: To code, or not to code? exploring impact of code in pre-training Year: (2024)
Ref_id:b5 Title: Llms can see and hear without any training Year: (2025)
Ref_id:b6 Title: The sciqa scientific question answering benchmark for scholarly knowledge Year: (2023-05)
Ref_id:b7 Title: Frozen language models are gradient coherence rectifiers in vision transformers Year: (2025)
Ref_id:b8 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b9 Title: Multi-agent collaborative data selection for efficient llm pretraining Year: (2024)
Ref_id:b10 Title: Managing extreme ai risks amid rapid progress Year: (2024)
Ref_id:b11 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b12 Title: Omni3d: A large benchmark and model for 3d object detection in the wild Year: (2023)
Ref_id:b13 Title: A unified optimization framework for language model data mixing Year: (2024)
Ref_id:b14 Title: Bring reason to vision: Understanding perception and reasoning through model merging Year: (2025)
Ref_id:b15 Title: Chatbot arena: An open platform for evaluating llms by human preference Year: (2024)
Ref_id:b16 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b17 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b18 Title: Emerging properties in unified multimodal pretraining Year: (2025)
Ref_id:b19 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b20 Title: Unveiling encoder-free vision-language models Year: (2024)
Ref_id:b21 Title: Evev2: Improved baselines for encoder-free vision-language models Year: (2025)
Ref_id:b22 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b23 Title: Large language model for lossless image compression with visual prompts Year: (2025)
Ref_id:b24 Title: Scaling language-free visual representation learning Year: (2025)
Ref_id:b25 Title: Mme: a comprehensive evaluation benchmark for multimodal large language models Year: (2023)
Ref_id:b26 Title: Blink: Multimodal large language models can see but not perceive Year: (2024)
Ref_id:b27 Title: Data mixing made efficient: A bivariate scaling law for language model pretraining Year: (2024)
Ref_id:b28 Title: Designing structured visuals from scratch Year: (2025)
Ref_id:b29 Title: Planting a seed of vision in large language model Year: (2023)
Ref_id:b30 Title:  Year: (2023)
Ref_id:b31 Title: The llama 3 herd of models Year: (2024)
Ref_id:b32 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b33 Title: What images are more memorable to machines? arXiv preprint Year: (2022)
Ref_id:b34 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b35 Title: Optimizing pretraining data mixtures with llm-estimated utility Year: (2025)
Ref_id:b36 Title: Ai2d-rst: A multimodal corpus of 1000 primary school science diagrams Year: (2021)
Ref_id:b37 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b38 Title: The platonic representation hypothesis Year: (2024)
Ref_id:b39 Title: Position: The platonic representation hypothesis Year: (2024)
Ref_id:b40 Title: Harnessing the universal geometry of embeddings Year: (2025)
Ref_id:b41 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b42 Title: Medvisionllama: Leveraging pre-trained large language model layers to enhance medical image segmentation Year: (2024)
Ref_id:b43 Title: Residual-based language models are free boosters for biomedical imaging tasks Year: (2024)
Ref_id:b44 Title: What matters when building vision-language models? arXiv preprint Year: (2024)
Ref_id:b45 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b46 Title: Temporal reasoning transfer from text to video Year: (2024)
Ref_id:b47 Title: On pre-training for visual language models Year: (2024)
Ref_id:b48 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b49 Title: Visual instruction tuning Year: (2023)
Ref_id:b50 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b51 Title: Llava-next: Improved reasoning, ocr, and world knowledge, 2024b Year: ()
Ref_id:b52 Title: Towards generalizable reasoning across modalities and domains Year: (2025)
Ref_id:b53 Title: Mmbench: Is your multi-modal model an all-around player Year: (2024)
Ref_id:b54 Title: On the hidden mystery of ocr in large multimodal models Year: (2023)
Ref_id:b55 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b56 Title: Frozen pretrained transformers as universal computation engines Year: (2022)
Ref_id:b57 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: ()
Ref_id:b58 Title: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2023)
Ref_id:b59 Title: At which training stage does code data help llms reasoning? arXiv preprint Year: (2023)
Ref_id:b60 Title: Chartqa: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b61 Title: Llms on the line: Data determines loss-to-loss scaling laws Year: (2025)
Ref_id:b62 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b63 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b64 Title: Scaling data-constrained language models Year: (2023)
Ref_id:b65 Title:  Year: (2024)
Ref_id:b66 Title: Dinov2: Learning robust visual features without supervision Year: (2023)
Ref_id:b67 Title: Finding and editing multi-modal neurons in pre-trained transformers Year: (2023)
Ref_id:b68 Title: Frozen transformers in language models are effective visual encoder layers Year: (2024)
Ref_id:b69 Title: Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context Year: (2016)
Ref_id:b70 Title: The refinedweb dataset for falcon llm: outperforming curated corpora with web data, and web data only Year: (2023)
Ref_id:b71 Title: Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models Year: (2023)
Ref_id:b72 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b73 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b74 Title:  Year: (2025)
Ref_id:b75 Title: Coqa: A conversational question answering challenge Year: (2019)
Ref_id:b76 Title: Procedural knowledge in pretraining drives reasoning in large language models Year: (2024)
Ref_id:b77 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b78 Title: Multimodal neurons in pretrained text-only transformers Year: (2023)
Ref_id:b79 Title: A vision check-up for language models Year: (2024)
Ref_id:b80 Title: Anton van den Hengel, and Damien Teney. Transformers pretrained on procedural data contain modular structures for algorithmic reasoning Year: (2025)
Ref_id:b81 Title: Scaling laws for optimal data mixtures Year: (2025)
Ref_id:b82 Title: Scaling laws for native multimodal models Year: (2025)
Ref_id:b83 Title: Textcaps: a dataset for image captioning with reading comprehension Year: (2020)
Ref_id:b84 Title: Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning Year: (2021)
Ref_id:b85 Title: Hovle: Unleashing the power of monolithic vision-language models with holistic vision-language embedding Year: (2025)
Ref_id:b86 Title: Chameleon: Mixed-modal early-fusion foundation models Year: (2024)
Ref_id:b87 Title: Morgane Rivière, et al. Gemma 3 technical report Year: (2025)
Ref_id:b88 Title: Cambrian-1: A fully open, vision-centric exploration of multimodal llms Year: (2024)
Ref_id:b89 Title: Metamorph: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b90 Title: LLaMA 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b91 Title: Cross-modal projection in multimodal llms doesn't really project visual attributes to textual space Year: (2024)
Ref_id:b92 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b93 Title: Open vision reasoner: Transferring linguistic cognitive behavior for visual reasoning Year: (2025)
Ref_id:b94 Title: Janus: Decoupling visual encoding for unified multimodal understanding and generation Year: (2024)
Ref_id:b95 Title: Data selection for language models via importance resampling Year: (2023)
Ref_id:b96 Title:  Year: (2023)
Ref_id:b97 Title: Qwen3 technical report Year: (2025)
Ref_id:b98 Title: Gated linear attention transformers with hardware-efficient training Year: (2023)
Ref_id:b99 Title: R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization Year: (2025)
Ref_id:b100 Title: Data mixing laws: Optimizing data mixtures by predicting language modeling performance Year: (2024)
Ref_id:b101 Title: Crosslingual reasoning through test-time scaling Year: (2025)
Ref_id:b102 Title: Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b103 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b104 Title: Harnessing diversity for important data selection in pretraining large language models Year: (2024)
Ref_id:b105 Title: Unveiling the impact of coding data instruction fine-tuning on large language models reasoning Year: (2025)
Ref_id:b106 Title: Lm4lv: A frozen large language model for low-level vision tasks Year: (2024)
Ref_id:b107 Title: Semantic understanding of scenes through the ade20k dataset Year: (2019)
Ref_id:b108 Title: Less is more for alignment Year: (2024)
Ref_id:b109 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
