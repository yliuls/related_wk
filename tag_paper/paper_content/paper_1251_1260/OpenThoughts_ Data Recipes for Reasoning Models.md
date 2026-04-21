Title: DATA RECIPES FOR REASONING MODELS
Abstract: UCLA, 8 JSC, 9 LAION, 10 NYU, 11 UNC Chapel Hill, 12 ASU, 13 Lila Sciences, 14 Cornell Tech 15 TUM 16 Open-Ψ (Open-Sci) Collective *, † denote equal contribution. ζ denotes additional core contributors. The order is determined randomly.

Section: ABSTRACT
Reasoning models have made rapid progress on many benchmarks involving math, code, and science. Yet, there are still many open questions about the best training recipes for reasoning since state-of-the-art models often rely on proprietary datasets with little to no public information available. To address this, the goal of the OpenThoughts project is to create open-source datasets for training reasoning models. Our OpenThoughts2-1M dataset led to OpenThinker2-32B, the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks such as AIME and LiveCodeBench. We then improve our dataset further by systematically investigating each step of our data generation pipeline with 1,000+ controlled experiments, which led to OpenThoughts3. Scaling the pipeline to 1.2M examples and using QwQ-32B as teacher yields our OpenThinker3-7B model, which achieves state-of-the-art results: 53% on AIME 2025, 51% on LiveCodeBench 06/24-01/25, and 54% on GPQA Diamond -improvements of 15.3, 17.2, and 20.5 percentage points compared to the DeepSeek-R1-Distill-Qwen-7B. All of our datasets and models are available on openthoughts.ai. All models are finetuned from Qwen-2.5-7B-Instruct. We compare to large SFT datasets (AM, Nemotron Nano) and small curated datasets (s1.1, LIMO) on AIME 2025 (left), LiveCodeBench 06/24-01/25 (middle), and GPQA Diamond (right). Scaling curves for all evaluation benchmarks are in Figure 6.
Sci GPQA-D 53.7 33.2 52.9 48.3 58.9 52.9 52.9 50.2 24.6 JEEBench 72.4 50.4 61.0 61.1 68.7 70.7 64.3 55. 3 33.9 Held Out HMMT 02/25 42.7 25.0 24.7 19.0 25.7 26.7 33.3 32.7 2.0 HLE MCQ 10.2 12.4 2.1 9.5 12.4 12.0 10.9 10.7 12.7 AIME25 53.3 38.0 41.3 28.7 39.7 48.0 50.7 47.3 8.0 LCB 06/24-01/25 51.7 34.5 42. 2 40.3 30.7 50.9 44.3 43.8 16. 3 Table 1: OpenThinker3-7B outperforms all open-data 7B and 8B reasoning models across domains. Our model also performs well on held out benchmarks which are not measured during our main experimentation, such as HMMT and AIME25. In our table, denotes a model trained from Qwen-2.5-7B-Instruct, M for Qwen-2.5-Math-Base, for Llama-3.1-8B-Instruct, and for DeepSeek-R1-Distill-Qwen-7B. "Base Model" denotes the starting checkpoint of the training strategy. "Method" denotes the model's optimization algorithm. In each row, we bold values within two standard errors of the highest-scoring model.
this section cite: []

Section: INTRODUCTION
Recent models, such as DeepSeek-R1 (Guo et al., 2025) and o3 (OpenAI, 2024), have demonstrated strong performance in reasoning-based domains, including math, coding, and science. These models often start from a strong base model, then introduce reasoning capabilities through a series of post-training techniques like supervised finetuning (SFT) or reinforcement learning (RL). This posttraining process equips these models with the ability to output long chains of thought, or "thinking tokens," during inference time, which can guide the model toward the correct answer. Yet, the complete recipes for frontier reasoning models are not public, making research for building reasoning models difficult.
Innovating on SFT data curation is a powerful method for building reasoning models (Abdin et al., 2024;Lin et al., 2024). For instance, the R1-Distill models show that it is possible to get state-of-theart small-to mid-scale reasoning models, with performance of 51% on AIME and 33% on GPQA, without any RL steps, based only on supervised fine-tuning on a large, carefully curated dataset of question-thinking tokens-answer triplets, where the thinking tokens and answers are generated using a reasoning teacher model or are taken from the real data that contains reasoning traces. Existing works, such as SkyT1 (NovaSky-Team, 2025b) and S1 (Muennighoff et al., 2025), adopt nearly identical model architectures and training setups as typical instruction tuning, yet still achieve performance improvements by focusing on improving the training datasets. These examples highlight the importance of curating high-quality SFT data as a key lever for reasoning performance.
this section cite: ['b68', 'b0', 'b48', 'b63']

Section: RELATED WORK
The release of models such as Gemini (Gemini-Team et al., 2023), QwQ (Qwen-Team, 2025), and DeepSeek-R1 (Guo et al., 2025), which made long reasoning traces visible to users, opened the possibility of training small models via the distillation of traces from larger ones. DeepSeek released strong distilled models together with DeepSeek-R1 (e.g., DeepSeek-R1-Distill-Qwen-7B), showing how promising this strategy can be. Following this, many open-data efforts have attempted to replicate these models by building SFT reasoning datasets through distillation from teacher models such as QwQ-32B (NovaSky-Team, 2025b) or DeepSeek-R1 (Bespoke-Labs, 2025).
Many datasets target math, code, and science to develop reasoning capabilities. Datasets such as OpenR1 (Face, 2025), OpenMathReasoning (Moshkov et al., 2025), and OpenCodeReasoning (Ahmad et al., 2025b) collect questions from public forums and competition sites like CodeForces, AoPS, and StackOverflow, while others like Natural Reasoning (Yuan et al., 2025) use large pre-training corpora as seed data for generating reasoning traces. Efforts like S1 (Muennighoff et al., 2025) and LIMO (Ye et al., 2025) emphasize manual curation of small datasets of challenging, high-quality prompts. In practice, many reasoning projects (e.g., DeepMath-103K (He et al., 2025c), OpenR1 (Face, 2025), and Nvidia Nemotron (Adler et al., 2024)) introduce innovations across multiple stages, such as data sourcing, filtering, and scaling. Beyond SFT, works such as AceReason (Chen et al., 2025) and Skywork-OR1 (He et al., 2025a) build reasoning datasets for reinforcement learning.
this section cite: ['b26', 'b28', 'b20', 'b61', 'b106', 'b63', 'b104', 'b20', 'b11']

Section: OPENTHOUGHTSDATA PIPELINE

this section cite: []

Section: EXPERIMENTAL SETUP
Training Our goal is to create the best dataset of question-response pairs for SFT reasoning. The best dataset is the one that produces the highest-performing model. To approach this systematically, we ablate each step of our pipeline individually, isolating the effect of a given strategy while keeping the rest of the pipeline constant. For each experiment, we utilize the full pipeline to generate 31,600 data points for each data strategy, and we finetune Qwen2.5-7B-Instruct (Yang et al., 2024b) on each dataset. Our experiments are conducted at a dataset scale that is small enough to be cost-effective yet
🙋 🧹 👑 Filter Answers Final Recipe Filter Questions 🧹 Generate Answers 💡 Mix Questions Select Teacher Model Source Questions
this section cite: []

Section: 👨🏫
Figure 2: The OpenThoughts experiment pipeline aims to build the strongest reasoning dataset recipe. We investigate (1) sourcing questions from existing and newly generated datasets, (2) mixing questions from the top-performing sources, (3) filtering for high-quality questions using fastText or LLMs, (4) deduplicating questions and sampling multiple answers per question, (5) filtering out low-quality answers using LLM verification or majority consensus, and ( 6) selecting the best teacher model.
large enough to provide a meaningful signal. We choose 31,600 as a log-scale midpoint between 10K and 100K, as √ 10 ≈ 3.16. These experiments inform the design choices for the final OpenThoughts3 pipeline. Appendix A contains details on hyperparameters and training setup.
Evaluation Setup We evaluate our models on a set of reasoning benchmarks containing math, code, and science questions. Per domain, these benchmarks are: AIME24 (MAA, 2024), AMC23 (MAA, 2023) and MATH500 (Hendrycks et al., 2021b) for math; CodeElo (Quan et al., 2025), CodeForces (Penedo et al., 2025a), and LiveCodeBench 05/23-05/24 (Jain et al., 2024) for code; GPQA Diamond (Rein et al., 2024) and JEEBench (Arora et al., 2023) for science. We score each model based on average performance on these eight tasks. Evalchemy (Raoof et al., 2025) is our primary evaluation tool, and we use the default setup provided for each benchmark. Further details on evaluation setup are in Appendix B. We also decontaminate our datasets against our benchmarks by removing samples with high similarity. Details for this process are in Appendix C. To measure generalization, our pipeline experiments exclude a held out set of benchmarks, which are only measured once pipeline experiments are over. This held out set consists of AIME 2025 (MAA, 2025), HMMT 02/25 (Balunović et al., 2025), Humanity's Last Exam (multiple choice questions subset) (Phan et al., 2025), and LiveCodeBench 06/24-01/25 (Jain et al., 2024).
Pipeline At each pipeline step, we select the top-performing approach based on the average benchmark score across all domains, and then proceed to the next step in the pipeline experimentation with this selection. DeepSeek-R1 is the default teacher model unless specified otherwise.
this section cite: ['b80', 'b38', 'b83', 'b129', 'b82', 'b126', 'b75', 'b38']

Section: QUESTION SOURCING
The first step in our data generation pipeline is finding questions for each data domain. We can broadly categorize our question sourcing techniques into three types: (1) Fully synthetic -an existing LLM generates questions with little-to-no seed material. Examples include CodeAlpaca (Chaudhary, 2023) and CamelChemistry (Li et al., 2023a). These often involve prompting an LLM with a template to generate multiple questions. (2) Semi-synthetic -an LLM uses existing data sources such as CommonCrawl or FineWeb (Penedo et al., 2024a) as seeds to form questions. Examples include TigerLabMath (Yue et al., 2024) and AutoMathText (Zhang et al., 2024b). (3) Non-synthetichumans write the questions. Examples include StackExchange and ShareGPTCode. These questions often arise from online forums, contests, chatbot interactions, and other sources.
Our experiments cover 27 different question sources for code questions, 21 sources for math, and 14 sources for science. The details of these sources are in Appendix O.1. The first step of our ablation is to generate 31,600 questions using each source. For sources that produce fewer datapoints, we repeat the questions until we reach the desired amount. We use GPT-4o-mini for all sources that we generate which require an LLM. Finally, we use DeepSeek-R1 to generate responses for each question, even if a pre-existing answer exists.
The experimental results are in Table 2. For code, CodeGolf questions from StackExchange and competitive coding questions from OpenCodeReasoning (Ahmad et al., 2025b) perform well, achieving scores of 25.3 and 27.5 on average on code benchmarks. For math, both LLM-generated questions in openmath-2-math (Toshniwal et al., 2024a) and human-written questions in NuminaMath (LI et al., SFT Datasets Benchmarks Code Question Source Average Code Avg Math Avg Science Avg StackExchange-CodeGolf * 38.8 0.4 25.3 0.6 50.9 1.1 40.7 0.5 OpenCodeReasoning 38.4 0.3 27.5 0.4 47.9 0.7 40.7 0.6 KodCode-V1 37.7 0.3 23.9 0.4 49.8 0.7 40.4 0.3 . . . . . . . . . . . . . . . bugdaryan/sql-create-. . . 21.6 0.6 7.0 0.7 34.1 1.4 24.7 0.9 Math Question Source Average Code Avg Math Avg Science Avg OpenMath-2-Math 38.1 0.3 12.4 0.2 58.8 1.0 45.6 0.2 NuminaMath-1.5 37.4 0.5 11.4 0.5 58.5 1.0 45.0 1.2 MathPile * 36.2 0.5 11.5 0.7 55.1 0.9 44.6 1.1 . . . . . . . . . . . . . . . Lap1official/Math * 24.4 0.3 7.3 0.3 38.6 1.0 28.5 0.3 Science Question Source Average Code Avg Math Avg Science Avg StackExchange-Physics * 34.3 0.4 11.9 0.5 50.9 0.8 43.2 0.7 OrganicChemistry-PDF * 34.0 0.3 8.4 0.3 52.1 0.7 45.3 0.8 CQADupStack-Physics 33.3 0.4 7.4 0.3 51.9 1.1 44.1 0.9 . . . . . . . . . . . . . . . AdapterOcean/biology_dataset. . . 21.9 0.4 3.1 0.3 41.3 1.1 21.1 0.8
Table 2: Evaluating question sources and generation strategies. We show only the top 3 scoring sources for each domain; descriptions of each source are in Appendix O.1 and full results are in Tables 31 to 33. Each row represents a unique source of questions. Question quality greatly affects performance, yielding a 17.2 gap between the strongest and the weakest code datasets. The * symbol denotes a new dataset we created with a programmatic generation strategy. Gray subscripts represent standard errors, and we bold values within two standard errors of the highest-scoring data strategy.
this section cite: ['b113', 'b9', 'b108']

Section: SFT Datasets Benchmarks

this section cite: []

Section: Code Question Mixing Strategy Average Code Avg Math Avg Science Avg
Top 1 Code Sources 39.9 0.6 23.
1 1.0 54.5 0.8 43.1 1.2 Top 2 Code Sources 41.3 0.4 27.3 0.3 54.7 0.9 42.1 1.0 Top 4 Code Sources 38.6 0.4 24.2 0.6 52.2 0.8 39.8 0.9 Top 8 Code Sources 37.0 0.4 21.8 0.3 51.9 1.2 37.7 0.6 Top 16 Code Sources 36.4 0.4 20.8 0.4 50.1 0.9 39.1 1.0
Table 3: Mixing different code question sources. Our experiments show that choosing only the two best question sources outperforms mixing more question sources. Similar results hold for science and math data domains. Full results including the math and science datasets are in Tables 34 to 36.
2024) score the highest, achieving 58.8 and 58.5 on average across math benchmarks. Lastly, for science, the highest-scoring question generation strategies are physics questions from StackExchange and LLM-extracted questions from organic chemistry textbooks, which achieve an average score of 43.2 and 45.3, respectively, on science benchmarks. No clear pattern emerges across question generation strategies -simple synthetic methods perform comparably to, and occasionally better than, more complex or manually curated pipelines. These top-performing question sources provide the foundation for subsequent stages of the pipeline.
this section cite: []

Section: SFT Datasets Benchmarks

this section cite: []

Section: Math Question Filtering Strategy Average Code Avg Math Avg Science Avg
Response Length Selection (GPT-4.
1-mini) 41.9 0.3 13.4 0.3 66.0 0.8 48.6 0.4 Response Length Selection (GPT-4.1-nano) 39.4 0.3 11.0 0.4 64.5 0.7 44.3 0.7 AskLLM Selection 36.3 0.4 9.5 0.5 58.1 1.1 43.8 0.6 FastText (P: Numina; N: Lap1official) 35.6 0.4 11.0 0.2 54.9 1.1 43.5 0.8 . . . . . . . . . . . . . . . Code Question Filtering Strategy Average Code Avg Math Avg Science Avg Difficulty-based Selection 43.0 0.5 27.7 0.4 56.0 1.3 46.4 0.7 Response Length Selection (GPT-4.1-nano) 42.2 0.4 26.6 0.5 55.4 1.3 46.0 0.2 AskLLM Selection 41.6 0.5 28.8 0.5 52.1 1.2 45.2 0.8 Response Length Selection (GPT-4o-mini) 40.8 0.5 25.6 0.5 53.1 0.9 45.2 1.1 . . . . . . . . . . . . . . .
Table 4: Filtering questions provides an effective tool for extracting high-quality questions.
Using LLM-based methods to find the best questions from the question sources outperformed classical filtering methods such as fastText and embedding-based filters. This table shows the top-performing strategies. Full results including science datasets are reported in Tables 37 to 39.
this section cite: []

Section: MIXING QUESTIONS
After obtaining high-quality questions from various question sources, the challenge becomes how to combine them effectively -should we rely on a single best-performing strategy, or blend multiple strong ones to maximize downstream performance? This mixing strategy is a key design choice in many generation pipelines (Yue et al., 2024;Moshkov et al., 2025;Lambert et al., 2024). Intuitively, adding more question sources into the mix introduces the risk of incorporating lower-quality strategies in exchange for greater diversity. Our experiments aim to assess whether the additional question diversity justifies this tradeoff in terms of question quality.
For simplicity, we use the rankings of the previous step in our pipeline (Section 3.2) as a heuristic for candidate dataset selection. Our mixing strategy selects the top-ranked N datasets, randomly samples 31,600   N questions from each source, and concatenates them to form a dataset of size 31,600. We sweep values of N ∈ {1, 2, 4, 8, 16}. The results for the code domain are in Table 3 while the other results are in Appendix P.2. Our experiments show that mixing many question sources degrades performance: mixing at most two sources yields the best results across all data domains. Using two high-quality code question sources instead of 16 strategies results in a 5% accuracy increase on average across all benchmarks. This result indicates that downstream performance benefits from increased quality of source data rather than diversity induced by mixing multiple question sources.
Takeaway: We use OpenMath-2-Math as our sole math question source, CodeGolf and Open-CodeReasoning as our code question sources, and StackExchangePhysics and OrganicChemistry-PDFs as our science question sources.
this section cite: ['b108', 'b61', 'b40']

Section: QUESTION FILTERING
Since each data source can contain millions of potential questions, answering and training on every possible question is infeasibly expensive. Therefore, the next step is to select a high-quality subset of questions from each source. Across the literature, a wide range of filtering strategies have consistently improved overall dataset quality (Soldaini et al., 2024;Su et al., 2024;Li et al., 2024;Wettig et al., 2025;Penedo et al., 2024a;Gao et al., 2020;Shum et al., 2025). Using the best question sources from Section 3.3 as a starting point, we extensively explore various filtering methods, including fastText classifiers, difficulty scores, and embedding distance, to select higher-quality questions. A detailed description of these filtering methods is in Appendix O.2. The results of these experiments for math and code are reported in Table 4 while the science results are in Appendix P.3 (Table 39).
this section cite: ['b89', 'b91', 'b8', 'b100', 'b24', 'b87']

Section: SFT Datasets Benchmarks

this section cite: []

Section: Science Answer Generation Strategy Average Code Avg Math Avg Science Avg
Exact Dedup w/ 16× sampling 36.2 0.5 9.0 0.4 54.5 1.0 49.7
1.2 Fuzzy Dedup w/ 16× sampling 36.1 0.4 10.9 0.2 52.9 1.3 48.8 0.5 Exact Dedup w/ 4× sampling 35.8 0.5 10.6 0.7 51.8 1.0 49.6 1.2 No Dedup w/ 4× sampling 35.8 0.4 10.0 0.4 55.2 0.8 45.4 0.9 No Dedup w/ 16× sampling 35.7 0.4 7.6 0.5 53.8 1.0 50.9 0.5 No Dedup w/ 1× sampling 35.5 0.3 9.3 0.3 54.2 1.1 46.9 0.2 Exact Dedup w/ 1× sampling 35.0 0.4 7.6 0.4 54.0 1.2 47.5 0.5 Fuzzy Dedup w/ 4× sampling 34.9 0.4 7.4 0.5 55.0 1.0 46.0 0.7 Fuzzy Dedup w/ 1× sampling 34.2 0.3 5.8 0.4 52.5 0.7 49.5 0.4
Table 5: Deduplication and repeated teacher sampling provide an axis of scale. Using fewer questions and annotating more times performs similarly or even outperforms annotating more questions fewer times. There does not seem to be a clear trend in types of deduplication that improve performance. Full results including math and code datasets are in Tables 40 to 42.
The two highest performing question filtering methods are difficulty-based filtering and response length filtering. Difficulty-based filtering asks an LLM (GPT-4o-mini) to assess the difficulty of each question, then retains the most difficult questions. Difficulty-based filtering is the winning strategy for code. Meanwhile, response length filtering asks an LLM to respond to each question directly, then selects the questions with the longest LLM-generated responses. Response length filtering performs the best for math and science. For math and code domains, using the best question filtering strategy resulted in an average improvement of 4% and 6% over the random filtering baseline, respectively. We test different LLMs such as GPT-4o-mini and GPT-4.1-nano for response length filtering and find that using stronger models for response length filtering typically outperforms using weaker models. For all domains, using LLM-based filtering methods outperformed classical filtering methods such as embedding-based and fastText filters.
Takeaway: We use difficulty-based filtering with GPT-4o-mini for code questions, and response length filtering with GPT-4.1-mini for math and science questions.
this section cite: []

Section: DEDUPLICATION AND SAMPLING MULTIPLE ANSWERS PER QUESTION
Deduplication is a powerful strategy for improving dataset quality (Li et al., 2024;Lee et al., 2022;Penedo et al., 2024b;Fang et al., 2025;Liu et al., 2024). Our ablations investigate the effects of question deduplication on downstream reasoning performance. We explore three degrees of deduplication strictness: no deduplication, exact match deduplication, and fuzzy deduplication using a threshold-based string similarity. Further details are in Appendix O.3. While deduplication enhances question diversity by reducing repetition, a natural counterpart for enhancing answer diversity is to query the teacher multiple times to elicit distinct responses. This strategy trades off higher answer diversity for lower question diversity and provides another axis of data scale. We explore three levels of sampling multiple answers per question, at 1×, 4×, and 16×.
To address the interplay between naturally occurring duplicate questions in the source datasets and the need to query the teacher multiple times per question, we sweep all combinations of deduplication levels (none, fuzzy, exact) and sampling multiple answers (1×, 4×, 16×). The results for the science domain are presented in Table 5, while the other domains' results are in Appendix P.4 (Table 40 and Table 41). For code and science data, various combinations of deduplication and multiple answer generation yield similar results. For example, the baseline of no deduplication with 1× answer per question performs 0.7 points worse on average than exact deduplication with 16× answers per question for the code domain. Meanwhile, for math, exact deduplication with 4× answers per question performs the best, and 16× answers per question is the second-best option. We adopt the second-best option moving forward, as it provides better scalability. Similar to Section 3.3, the results here indicate that the benefits of question diversity may be limited for the reasoning datasets we measure performance on, at least when answer diversity increases. Thus, for math and science, we
Math Answer Filtering Strategy Average Code Avg Math Avg Science Avg No Filtering (not compute-controlled) 41.9 0.4 15.2 0.5 65.6 0.9 46.4 0.7 Random Filtering 41.6 0.4 14.9 0.4 64.8 0.9 46.7 0.5 Shortest Answers Selection 41.1 0.4 14.8 0.4 63.7 1.1 46.7 0.7 Removing Non-English Answers 41.1 0.5 14.2 0.5 63.1 1.0 48.6 1.0 . . . . . . . . . . . . . . . GPT Verification 40.0 0.5 13.1 0.3 61.4 1.1 48.3 1.1 Removing Long Paragraphs 38.0 0.4 5.7 0.2 64.5 0.9 46.8 1.0 Table 7: Using a weaker teacher outperformed using a stronger teacher. Across all domains, QwQ-32B was the strongest teacher model, despite being a weaker model than DeepSeek-R1. Further results can be seen in Table 48.
select the optimal strategy, which is exact deduplication with 16× answers per question. For code, we employ the second-best strategy, which involves no deduplication with 16× answers per question.
Takeaway: Our final pipeline uses 16× answers per question for all domains. It uses exact deduplication for math and science and no deduplication for code.
this section cite: ['b8', 'b42', 'b22', 'b49']

Section: ANSWER FILTERING
Verification or removing low-quality annotations is a common step in many reasoning data pipelines.
Intuitively, removing data that may be incorrect should improve downstream performance. Our experiments explore various answer filtering techniques. To ensure that we can still obtain datasets of size 31,600 after filtering, we first generate 63,200 answers, apply each answer-filtering strategy, and then sample 31,600 question-answer pairs from the filtered dataset. Our ablations also include a baseline with no filtering, which is not compute-controlled, as it contains 63,200 questions.
Table 6 shows the result of each filtering method for math datasets, and the results for code and science datasets are shown in Appendix P.5. For math datasets, the random filtering baseline outperformed all other filtering methods. A fastText (Joulin et al., 2017) classifier was the best answer-filtering method for code question-answer pairs. The positives for the fastText classifier came from CodeForces (Penedo et al., 2025a) answered with DeepSeek-R1, and the negatives came from CodeForces answered with GPT-4o-mini. For science, keeping the top 8 longest answers was the strongest filtering strategy. However, across all domains, the no-filtering strategy (training on all samples without controlling compute) led to performance similar to that of all other methods of filtering. This result suggests that the benefits of answer filtering are not significant enough to justify reducing the number of samples in the dataset. As such, we opt to skip this part in the pipeline.
Takeaway: We do not perform answer filtering because no filtering strategy outperformed the baseline, which uses all the answers. 3.7 TEACHER MODEL The previous experiments have relied on using DeepSeek-R1 as a teacher model. However, there are many possible candidates for teacher reasoning models, including DeepSeek R1, Phi-4-Reasoning-Plus-14B (Abdin et al., 2025), and QwQ-32B. Our experiments measure the downstream effects of selecting different teacher models for each strategy, as described in Section 3.6. The sampling hyperparameters are kept constant across all teacher models we studied. The results of this experiment are in Table 7. Across all domains, using QwQ-32B as a teacher model outperforms all other teacher models, yielding an average accuracy improvement of 1.9% and 2.6% over using DeepSeek-R1 as a teacher for code and math, respectively. This is despite the fact that QwQ-32B scores lower on average when compared to DeepSeek-R1. For example, DeepSeek-R1 outperforms QwQ-32B by 9%, 8%, and 23% on CodeElo, GPQA Diamond, and JEEBench, respectively. A comparison of the empirical strengths of each teacher is in Table 28.
this section cite: ['b39']

Section: Math

this section cite: []

Section: Generate Multiple Answers
All
Takeaway: We use QwQ-32B as the teacher model.
this section cite: []

Section: SCALING OUR PIPELINE TO OPENTHOUGHTS3-1.2M
Dataset scaling plays a key role in achieving strong performance. We investigate how well our pipeline scales by identifying the winning strategy in each successive pipeline step and plotting its performance from 316 to 31.6k examples. Figure 5 demonstrates that the scaling behavior improves as we successively stack the best choices from each stage in the pipeline. Additionally, Figure 5 also shows a strong positive correlation between scale and performance. This suggests that further scaling the dataset size could yield even greater gains. We thus mix and scale up the data pipelines from Section 3 to build OpenThoughts3-1.2M, our 1.2 million-sized dataset. OpenThoughts3-1.2M contains 850,000 math, 250,000 code, and 100,000 science datapoints. We chose this ratio following the OpenThoughts2-1M mixture used to train OpenThinker2, which exhibited strong and balanced performance on par with the DeepSeek-R1-Distill models. To arrive at the target number of samples in each domain, we work backwards to estimate how many questions we need at the beginning of the pipeline. Then, we apply the highest performing strategy at each stage in the pipeline, opting for the more scalable choices if performance is equal. This construction process of OpenThoughts3-1.2M is illustrated in Figure 3.
As seen in Table 1, OpenThinker3-7B is the best open-data reasoning model at the 7B scale, regardless of optimization algorithm choice (SFT, RL, or both). OpenThinker3-7B also generalizes well to evaluations held-out throughout the pipeline process, exhibiting the best scores on HMMT, AIME25, and LCB 06/24-01/25. Further results on scaling can be seen in Appendix D.
this section cite: []

Section: CONCLUSION
Through iterative experimentation, our pipeline surfaced several key insights into effective SFT reasoning data curation. These findings collectively shape our final pipeline, allowing us to build OpenThoughts3-1.2M, a state-of-the-art open-data SFT reasoning dataset, composed of science, math, and code data. Our final model, OpenThinker3-7B, trained on this data, is the SOTA open-data reasoning model at its model scale.
This work has several limitations. We did not explore datasets for reinforcement learning, a standard training regime for building reasoning models. Within the SFT realm, we did not explore the use of staged SFT or curriculum learning to further improve performance. We nonetheless believe this work serves as a valuable foundation for the community's continued progress on open reasoning models.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly Anshul Bansal. Basics of Organic Chemistry: A Textbook for Undergraduate Students Year: (2024)
Ref_id:b1 Title: Llama-nemotron: Efficient reasoning models Year: (2025)
Ref_id:b2 Title: Bespoke-stratos: The unreasonable effectiveness of reasoning distillation Year: (2025)
Ref_id:b3 Title: self-oss-instruct-sc2-exec-filter-50k. Hugging Face Year: (2024)
Ref_id:b4 Title: Advanced Organic Chemistry: Part A: Structure and Mechanisms. Advanced Organic Chemistry Year: (2007)
Ref_id:b5 Title: Organic Chemistry Year: (1996)
Ref_id:b6 Title: Organic Chemistry. TMH Year: (2010)
Ref_id:b7 Title: Modern Methods of Organic Synthesis Year: (1978)
Ref_id:b8 Title: Mceval: Massively multilingual code evaluation Year: (2024)
Ref_id:b9 Title: Code alpaca: An instruction-following llama model for code generation Year: (2023)
Ref_id:b10 Title: Do not think that much for 2+ 3=? on the overthinking of o1-like llms Year: (2024)
Ref_id:b11 Title: Acereason-nemotron: Advancing math and code reasoning through reinforcement learning Year: (2025)
Ref_id:b12 Title: Organic Chemistry Year: (2012)
Ref_id:b13 Title: Organic Chemistry Year: (2012)
Ref_id:b14 Title: On the use of arxiv as a dataset Year: (2019)
Ref_id:b15 Title: Rosetta code -rosetta code Year: (2022-12)
Ref_id:b16 Title: Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement Year: (2025)
Ref_id:b17 Title: Essentials of Organic Chemistry Year: (2011)
Ref_id:b18 Title: Lawma: The power of specialization for legal tasks Year: ()
Ref_id:b19 Title: MAMUT: A novel framework for modifying mathematical formulas for the generation of specialized datasets for language model training Year: (2025)
Ref_id:b20 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025-01)
Ref_id:b21 Title: react-code-instructions Year: (2025)
Ref_id:b22 Title: Datasets, documents, and repetitions: The practicalities of unequal data quality Year: (2025)
Ref_id:b23 Title: Par-four-fineweb-edu-fortified-chemistry-physics-astronomy-math-reason Year: (2024)
Ref_id:b24 Title: The Pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b25 Title: The Practical Methods of Organic Chemistry Year: (1917)
Ref_id:b26 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b27 Title: glaiveai/glaive-code-assistant-v3: A dataset for code assistance Year: (2023)
Ref_id:b28 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b29 Title: Organic Chemistry: A Short Course. Available Titles OWL Series. Cengage Learning, 2011. ISBN 9781111425562 Year: ()
Ref_id:b30 Title: Fernando Fernandes, and Cognitive Computations. Dolphin-coder Year: (2024)
Ref_id:b31 Title: Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. 2025b Year: (2025)
Ref_id:b32 Title: Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning Year: (2025)
Ref_id:b33 Title: Measuring coding challenge competence with apps Year: ()
Ref_id:b34 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b35 Title: Organic Chemistry: An Intermediate Text Year: (2004)
Ref_id:b36 Title: Cqadupstack: A benchmark data set for community question-answering research Year: (2015)
Ref_id:b37 Title: Opencoder: The open cookbook for top-tier code large language models Year: (2023)
Ref_id:b38 Title: Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code Year: (2024)
Ref_id:b39 Title: Bag of tricks for efficient text classification Year: (2017-04)
Ref_id:b40 Title: Tulu 3: Pushing frontiers in open language model post-training Year: (2024)
Ref_id:b41 Title: Lap1official Year: (2024)
Ref_id:b42 Title: Deduplicating training data makes language models better Year: (2022-05)
Ref_id:b43 Title: Camel: Communicative agents for "mind" exploration of large scale language model society Year: (2023)
Ref_id:b44 Title: Camel: Communicative agents for "mind" exploration of large scale language model society Year: (2023)
Ref_id:b45 Title: Datacomp-lm: In search of the next generation of training sets for language models Year: (2024)
Ref_id:b46 Title:  Year: ()
Ref_id:b47 Title: Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode Year: (2022)
Ref_id:b48 Title: Rho-1: Not all tokens are what you need Year: (2024)
Ref_id:b49 Title: Llm360 k2-65b: Scaling up fully transparent open-source llms Year: (2024)
Ref_id:b50 Title: Scp-116k: A high-quality problem-solution dataset and a generalized pipeline for automated extraction in the higher education science domain Year: (2025)
Ref_id:b51 Title: A Concise Text-Book of Organic Chemistry: The Commonwealth and International Library: Chemistry Division Year: (2016)
Ref_id:b52 Title: Amc 2023 problems Year: (2023)
Ref_id:b53 Title:  Year: (2024)
Ref_id:b54 Title:  Year: (2025)
Ref_id:b55 Title:  Year: ()
Ref_id:b56 Title: A standardized evaluation framework for automated red teaming and robust refusal Year: (2024)
Ref_id:b57 Title: Fundamentals of Organic Chemistry. Brooks/Cole, Cengage Learning Year: (2010)
Ref_id:b58 Title: Organic Chemistry. Cengage Learning, 10 edition Year: (2023)
Ref_id:b59 Title: Techniques in Organic Chemistry. W. H. Freeman, 2010. ISBN 9781429219563 Year: ()
Ref_id:b60 Title: Advanced Organic Synthesis: Methods and Techniques Year: (2012)
Ref_id:b61 Title: Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset Year: (2025)
Ref_id:b62 Title: Instruction tuning code large language models Year: (2023)
Ref_id:b63 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b64 Title: Alice in wonderland: Simple tasks showing complete reasoning breakdown in state-of-the-art large language models Year: (2024)
Ref_id:b65 Title: The Principles of Organic Chemistry. International Chemical Series Year: (1922)
Ref_id:b66 Title: Think less, achieve more: Cut reasoning costs by 50 Year: (2025)
Ref_id:b67 Title: Sky-t1: Train your own o1 preview model within 450 Year: (2025)
Ref_id:b68 Title: Openai o3 and o4-mini system card Year: (2024)
Ref_id:b69 Title:  Year: (2019)
Ref_id:b70 Title: Mayank Mishra, and Matt Stallone. tiny-codes: A dataset for learning reasoning with language and code Year: (2024)
Ref_id:b71 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: ()
Ref_id:b72 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: ()
Ref_id:b73 Title:  Year: (2025)
Ref_id:b74 Title:  Year: (2025)
Ref_id:b75 Title: Humanity's last exam Year: (2025)
Ref_id:b76 Title: Primeintellect/stackexchange-question-answering Year: (2025)
Ref_id:b77 Title:  Year: (2025)
Ref_id:b78 Title: GitHub repository. prithivMLmods. Coder-Stat dataset Year: (2024-09-19)
Ref_id:b79 Title: Fine-tuning aligned language models compromises safety, even when users do not intend to! Year: (2023)
Ref_id:b80 Title: Benchmarking competition-level code generation of llms with human-comparable elo ratings Year: (2025)
Ref_id:b81 Title: Qwq-32b: Embracing the power of reinforcement learning Year: (2025-03)
Ref_id:b82 Title:  Year: (2025)
Ref_id:b83 Title: GPQA: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b84 Title: Reflectioncoder: Learning from reflection sequence for enhanced one-off code generation Year: (2024)
Ref_id:b85 Title: Xstest: A test suite for identifying exaggerated safety behaviours in large language models Year: (2023)
Ref_id:b86 Title: Analysing mathematical reasoning abilities of neural models Year: ()
Ref_id:b87 Title: Predictive data selection: The data that predicts is the data that teaches Year: (2025)
Ref_id:b88 Title: March's Advanced Organic Chemistry: Reactions, Mechanisms, and Structure Year: (2007)
Ref_id:b89 Title: Dolma: an open corpus of three trillion tokens for language model pretraining research Year: (2024-08)
Ref_id:b90 Title: Dataset of all Stack Exchange sites, released under CC BY-SA Year: (2024)
Ref_id:b91 Title: Nemotron-cc: Transforming common crawl into a refined long-horizon pretraining dataset Year: (2024)
Ref_id:b92 Title: Stop overthinking: A survey on efficient reasoning for large language models Year: (2025)
Ref_id:b93 Title: Reflectioncoder: Learning from reflection sequence for enhanced one-off code generation Year: (2024)
Ref_id:b94 Title: TIGER-Lab/MATH-plus: A benchmark for evaluating large language models on math reasoning problems Year: (2025)
Ref_id:b95 Title: Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data Year: (2024)
Ref_id:b96 Title: Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data Year: (2024)
Ref_id:b97 Title: Mathpile: A billion-token-scale pretraining corpus for math Year: (2024)
Ref_id:b98 Title: Star-1: Safer alignment of reasoning llms with 1k data Year: (2025)
Ref_id:b99 Title: Source code is all you need Year: (2023)
Ref_id:b100 Title: Organize the web: Constructing domains enhances pre-training data curation Year: (2025)
Ref_id:b101 Title: Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding Year: (2025)
Ref_id:b102 Title: Qwen2 technical report Year: (2024)
Ref_id:b103 Title:  Year: (2024)
Ref_id:b104 Title: Less is more for reasoning Year: (2025)
Ref_id:b105 Title: Bootstrap your own mathematical questions for large language models Year: (2023)
Ref_id:b106 Title: Naturalreasoning: Reasoning in the wild with 2.8m challenging questions Year: (2025)
Ref_id:b107 Title: Mammoth: Building math generalist models through hybrid instruction tuning Year: (2023)
Ref_id:b108 Title: Mammoth2: Scaling instructions from the web Year: (2024)
Ref_id:b109 Title: Infinitymath: A scalable instruction tuning dataset in programmatic mathematical reasoning Year: (2024)
Ref_id:b110 Title: Autonomous data selection with language models for mathematical texts Year: (2024)
Ref_id:b111 Title: Autonomous data selection with zero-shot generative classifiers for mathematical texts Year: ()
Ref_id:b112 Title: Opencodeinterpreter: Integrating code generation with execution and refinement Year: (2024)
Ref_id:b113 Title: 24: a mathematics competition for high-school students held in 2024. It involves 30 questions of different levels of difficulty Year: ()
Ref_id:b114 Title: 25: a mathematics competition for high-school students held in 2025. It involves 30 questions of different levels of difficulty Year: ()
Ref_id:b115 Title: AMC23: a mathematics competition for high-school students held in 2023. It consists of 40 questions with different difficulty levels. The answers are numerical Year: ()
Ref_id:b116 Title: MATH500: consists of 500 diverse problems in probability, algebra, trigonometry, and geometry Year: ()
Ref_id:b117 Title: CodeForces: consists of 453 real-world programming problems sourced from the Code-Forces platform. The benchmark measures unit test-based execution accuracy with a human-comparable Elo rating Year: ()
Ref_id:b118 Title: CodeElo: consists of 391 real-world programming problems curated from a variety of contests. The benchmark measures unit test-based execution accuracy with a difficultycalibrated Elo rating Year: ()
Ref_id:b119 Title: LiveCodeBench: a benchmark of real-world programming tasks that evaluate a model's ability to generate, execute, verify, and iteratively repair solutions using unit-test feedback Year: (2023-05)
Ref_id:b120 Title: GPQA Diamond: a set of 198 challenging questions from the Graduate-Level Google-Proof Q&A Benchmark (GPQA). Questions are in multiple-choice format Year: ()
Ref_id:b121 Title: JEEBench: contains 515 questions spanning Physics, Chemistry and Mathematics subjects collected from the Joint Entrance Examination (JEE): Advanced held from 2016 to 2023. Questions are in multiple-choice and numerical formats Year: ()
Ref_id:b122 Title: HMMT: 30 questions from the HMMT high school mathematics competition held in February 2025 Year: ()
Ref_id:b123 Title: HLE: a subset of 512 multiple-choice, text-only questions from the Humanity's Last Exam (HLE) benchmark. Benchmark Domain / Description Number of Questions Code Generation CodeElo Year: (2025)
Ref_id:b124 Title: 2025a) Benchmarking competition-level code generation Year: ()
Ref_id:b125 Title: Holistic code benchmark with iterative repair. 369 Mathematical Problem Solving AIME 24 (MAA, 2024) 2024 AIME math-reasoning dataset. 30 AIME 25 (MAA, 2025) 2025 AIME math-reasoning dataset Year: (2024)
Ref_id:b126 Title: High school mathematics competition Year: (2025)
Ref_id:b127 Title: 2021b) 500-problem split from Year: ()
Ref_id:b128 Title: Graduate-level, Google-proof Q&A benchmark Year: ()
Ref_id:b129 Title: Subject-matter expert questions. 512 SFT Datasets Benchmarks Filtering Strategy Average Code Year: (2023)
Ref_id:b130 Title:  Year: ()
Ref_id:b131 Title: Length-based Selection Year: ()
Ref_id:b132 Title:  Year: ()
Ref_id:b133 Title:  Year: ()
Ref_id:b134 Title:  Year: ()
Ref_id:b135 Title:  Year: ()
Ref_id:b136 Title: Table 39: Full Ablation for Science Question Filtering Year: ()
