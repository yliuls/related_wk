Title: The Best Instruction-Tuning Data are Those That Fit
Abstract: High-quality supervised finetuning (SFT) data are essential for unlocking pretrained LLMs' capabilities. Typically, instructions are paired with responses from various sources by humans annotators or other LMs, which are often out of the distribution of the target model to be finetuned. This, at scale, can lead to diminishing returns and even hurt the models' performance and robustness. We hypothesize that SFT is most effective with data aligned to the model's pretrained distribution and propose GRAPE-a novel SFT framework that tailors supervision to the target model. For each instruction, it gathers responses from various sources, and selects the one that aligns most closely to the target model's pretrained distribution, as measured by the normalized probability. We then proceed with standard SFT with these selected responses. We first evaluate GRAPE with a controlled experiment, where we sample various solutions for each question in UltraInteract from multiple models and finetune on GRAPE-selected data using LMs from different families including LLaMA.1-8B, Mistral-7B and Qwen2.5-7B. GRAPE significantly outperforms strong baselines, including distilling from the strongest model with absolute gain up to 13.8% averaging across benchmarks, and a baseline trained on 3× more data with maximum 17.3% performance improvements. GRAPE's strong performance generalizes to off-the-shelf SFT data. We use GRAPE to subsample responses from the post-training data used for Tulu3 and Olmo-2. GRAPE can outperform strong baselines with 4.5 times the data by 6.1% and state-of-the-art data selection approaches by 3.9% on average performance. Remarkably, using 1/3 data and half number of epochs, GRAPE allows LLaMA.1-8B to surpass the performance of Tulu3-SFT by 3.5%. Our findings highlight that aligning supervision with the pretrained distribution offers a simple yet powerful way to improve SFT efficiency and performance.

Section: Introduction
High-quality, large-scale supervised data is crucial for supervised fine-tuing(SFT; Databricks, 2023;Köpf et al., 2023;Zhao et al., 2024a;Zheng et al., 2024). A common practice of collecting SFT data involves sampling responses from strong language models, predominantly focusing on expanding the size of the dataset and improving the overall quality of the responses (Sun et al., 2023;Taori et al., 2023;Wang et al., 2023;Xu et al., 2024c;Chen et al., 2024). However, recent research suggests that there is more complex dynamics involved (Xu et al., 2024d). A plateau effect in synthetic data scaling, where performance either stagnates or even declines as the size of the synthetic data increases beyond a certain point, has been widely observed. This phenomenon arises due to issues such as diminishing diversity (Padmakumar & He, 2024;Guo et al., 2023) and distortion in the data distribution (LeBrun et al., 2021), which ultimately undermine the base model's performance and robustness (Alemohammad et al., 2024;Gerstgrasser et al., 2024;Shumailov et al., 2023;Dohmatob et al., 2024;Hataya et al., 2023;Martínez et al., 2023a,b;Bohacek & Farid, 2023;Briesch et al., 2023).
Thus, effective SFT requires more than scaling up the data; it often needs "tailoring" the data to the unique characteristics of the target model. Existing works focus on enhancing the model's existing knowledge and capabilities (Du et al., 2023) and optimizing the curriculum progression for instruction tuning (Zhao et al., 2024b;Lee et al., 2024;Feng et al., 2023;Setlur et al., 2024). They typically find questions the model should best learn, instead of what answers the model should imitate.
Meanwhile, tailoring the responses to the target model has been a crucial ingredient for the success of later phases of LLM development, particularly through on-policy preference learning (Tajwar et al., 2024;Zhang et al., 2024c,a;Miao et al., 2024;Gulcehre et al., 2023;Azar et al., 2023;Tang et al., 2024a;Zhuang et al., 2023), and on-policy/online reinforcement learning (Guo et al., 2024b;Liu et al., 2024d;Zhou et al., 2024c).
Inspired by these insights, we hypothesize that SFT can similarly benefit from aligning data with the model, the core idea behind GRAPE. For each instruction, GRAPE gathers and selects response(s) from various sources that are closest to the target model's pretrained distribution. This is achieved by calculating the probability of each response using the target model and selects the one with the highest length-normalized probability( §3).After obtaining these more "in-distribution" responses, GRAPE proceeds with standard SFT without any modification to the training.
Unlike existing datasets that usually contains one-size-fits-all responses for each instruction without customization (Yu et al., 2024;Yuan et al., 2024b;Lian et al., 2023b;Teknium, 2023), GRAPE curates model-dependent SFT datasets that better matches the base model's distribution, better mitigating the risks associated with distribution shift like spurious correlations (Zhou et al., 2024d) and catastrophic forgetting (Luo et al., 2025;Kotha et al., 2024), while posing minimum overhead of single forward pass over the candidate set. In return, GRAPE allows better downstream performance with reduced training compute. To GRAPE's advantage, many existing datasets share overlapping instructions but contain different high-quality responses, e.g., the instructions in Flan (Longpre et al., 2023), GSM-8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), and the post-training recipes that re-use SFT instructions for preference learning (Lambert et al., 2024;OLMo et al., 2025). Therefore, GRAPE can directly select, for each model, the best fit(s) among the off-the-shelf responses without having to produce new responses, which proves effective in our experiments ( § 5).
We first validate our hypothesis through extensive controlled experiments on a reasoning dataset with chain-of-thoughts, UltraInteract-SFT (Yuan et al., 2024b), and demonstrate the importance of supervising base models with in-distribution responses( §4). We experimented on 4 popular pretrained LMs from Mistral (MistralAI, 2024c), Llama3.1 (Dubey et al., 2024) and Qwen2.5 (Hui et al., 2024) families. Notably, models fine-tuned with GRAPE-selected responses outperform those trained on a 3× larger datasets up to 17.3% absolute gain on average performances even with less compute, even surpass models trained on responses from the strongest teacher under consideration-LLAMA3.1-405B-INSTRUCT-by significant margins.
We then experiment with a more realistic setting for general-domain instruction-tuning, collecting responses from post-training data for Tulu3 and Olmo-v2. Again, GRAPE demonstrates its effectiveness by outperforming state-of-the-art data selection approaches by avg. 4.6% and a strong baseline trained on all these available data that is 4.5 times larger by up to 6.1%. Remarkably, GRAPE allows finetuning a Llama3.1-8B base model to exceed the performance of Tulu-8B-SFT using 1/3 data and half of the epochs.
Our results reveal that distributional alignment is a crucial and previously underappreciated dimension of effective instruction tuning. GRAPE introduces this perspective with a simple, scalable algorithm that consistently outperforms strong baselines with far less data and compute for the actual training.
this section cite: ['b22', 'b54', 'b150', 'b106', 'b112', 'b118', 'b12', 'b38', 'b60', 'b1', 'b104', 'b25', 'b40', 'b8', 'b9', 'b27', 'b61', 'b33', 'b102', 'b108', 'b85', 'b35', 'b6', 'b158', 'b137', 'b115', 'b79', 'b55', 'b77', 'b16', 'b59', 'b91', 'b28', 'b47']

Section: Background and Motivation

this section cite: []

Section: Data Engineering for Instruction Tuning
Data is central to the success of effective instruction tuning, (Xu et al., 2023;Xia et al., 2024;Chan et al., 2024), featuring both automated data synthesis (Xu et al., 2024a;Zeng et al., 2024;Yu et al., 2024;Wei et al., 2023) and selection (Xia et al., 2024;Chen et al., 2023a;Parkar et al., 2024;Li et al., 2024e). Some selection approaches focus on high-quality data by leveraging LLMs (Chen et al., 2023a;Parkar et al., 2024;Li et al., 2024b) or employing principled metrics (Kang et al., 2024;Mekala et al., 2024;Xia et al., 2024), while others, such as Yang et al. (2024b); Das & Khetan (2023), aim to identify diversity-optimized subsets for greater efficiency. An emerging trend is the customization of training data based on the characteristics of the base models. (Li et al., 2024c;Du et al., 2023;Li et al., 2024a). These methods typically reweight or filter instructions. However, a key overlooked aspect is response selection-specifically, choosing responses that align with the model's pretrained distribution, which may be critical for preserving useful behaviors and ensuring effective fine-tuning.
this section cite: ['b128', 'b28', 'b11', 'b142', 'b137', 'b121', 'b28', 'b95', 'b95', 'b53', 'b84', 'b28', 'b21', 'b27']

Section: Toward Distribution-Aligned Supervised Fine-Tuning

this section cite: []

Section: An Analogy from Reinforcement Learning and Preference Learning
The investigation of this work into the distribution match between the pre-trained LM and supervised fine-tuning (SFT) data is inspired by recent findings on policy optimization for LM alignment with RL (Ouyang et al., 2022) and preference learning (Rafailov et al., 2023;Ethayarajh et al., 2024). While the importance of matching training data distribution with the policy has been well noted in both traditional RL (Shi et al., 2023;Fujimoto et al., 2018;Kumar et al., 2019;Peng et al., 2019;Wang et al., 2021;Arora & Goyal, 2023;Jiang & Li, 2016;Tang & Abbeel, 2010) and LM settings (Xiong et al., 2024), preference learning algorithms like DPO (Rafailov et al., 2023), IPO (Azar et al., 2023) and KTO (Ethayarajh et al., 2024) first emerged as off-policy algorithms. However, subsequent research has highlighted the performance gap between on-policy and off-policy training due to distribution shifts (Xu et al., 2024b;Tang et al., 2024b) and proposed various mitigation strategies (Zhuang et al., 2023;Zhou et al., 2024c;Zhang et al., 2024a;Xiong et al., 2024;Guo et al., 2024b), showing that training models on data more closely aligned with their policy distribution can significantly improve performance, while failing to do so can yield sub-optimal policies or those that are harder to generalize. Works like SPIN (Yuan et al., 2024a) echo the intuition by gradually improving policy through self-play to mitigate distribution shift.
this section cite: ['b92', 'b99', 'b30', 'b103', 'b56', 'b97', 'b119', 'b4', 'b52', 'b109', 'b125', 'b99', 'b6', 'b30', 'b158', 'b125']

Section: Hypothesis: Supervised Fine-tuning Benefits From Data That Better Matches Base Distribution
The base distribution of pre-trained language models-shaped by extensive training on vast and diverse datasets-is inherently robust and generalizable (Brown et al., 2020;Saunshi et al., 2021). Therefore, during supervised fine-tuning phase, the pre-trained distribution should be carefully preserved (Kumar et al., 2022;Cohen-Wang et al., 2024;He et al., 2023;Yang et al., 2024d;Ding et al., 2023), to best retain the knowledge and capabilities that emerge during pre-training (Zhou et al., 2023). If the proximity between the pre-trained distribution and the fine-tuning data is not maintained, the limited number of training examples available during SFT, compared to the vast scale of pre-training data, can increase the risk of distribution distortion. This misalignment can lead to issues such as catastrophic forgetting (Aghajanyan et al., 2020;Yang et al., 2024d) and the emergence of spurious correlations (Feldman, 2021).
The central premise of our work is that by using responses closely aligned to the pre-trained distribution, we can minimize distribution shift during SFT and therefore achieve better data efficiency and stronger performance. stage of post-training than RL, sampling responses from the base model itself alone can lead to model collapse, as noted in (Shumailov et al., 2024). Prior studies have documented similar risks like instability, bias reinforcement, knowledge stagnation, and overfitting (Herel & Mikolov, 2024;Mobahi et al., 2020;Allen-Zhu & Li, 2023;Ghosh et al., 2024;Dong et al., 2025;Zhang et al., 2024d). To address this, we advocate for an approach that stays more in-distribution while delivering effective supervision to the base model. To this end, we propose to gather and select responses from various sources, and select one that is closest to the target model's pretrained distribution, which we name GRAPE.
this section cite: ['b10', 'b101', 'b57', 'b17', 'b41', 'b24', 'b152', 'b0', 'b32', 'b105', 'b44', 'b90', 'b2', 'b26']

Section: From On-Policy Alignment To Distribution-Aligned SFT We build on principles of on-policy alignment techniques with key distinctions tailored for SFT. Yet, given that SFT represents an earlier
3 Methodology
0 200 400 600 800 1000 Step 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 Loss Best Random Worst We introduce GRAPE, a surprisingly simple yet effective methodology to enhance supervised fine-tuning (SFT) by customizing the training data for the base model. The key idea is to find a response, among a candidate pool, for each instruction x i that aligns closely with the base model's pretrained distribution π θ0 .
As diagrammed in Figure 3, GRAPE consists of two main steps, followed by standard SFT: Response Collection ( §3.1) Collect a pool of high-quality candidate responses from various sources. Customization ( §3.2): For the target model to be finetuned π θ0 , find the response(s), for each instruction, that are closest to the pretrained distribution of π θ0 .
this section cite: []

Section: Collecting Responses from Existing Resources
For instruction-tuning of language models, high-quality instructions are more difficult to collect than responses (Xu et al., 2024c;Liu et al., 2024a). Therefore, it is a common practice to reuse existing instruction-tuning prompts while generating diverse responses using various methods tailored to specific requirements. For instance, instructions from Flan (Longpre et al., 2023), OpenOrca (Lian et al., 2023a), ShareGPT (Team, 2023), and the training splits of GSM-8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b), and CodeContests (Li et al., 2022) are frequently reused in datasets like Olmo (OLMo et al., 2025), Tulu (Lambert et al., 2024), OpenHermes (Teknium, 2023), OpenOrca (Lian et al., 2023a), MetaMath (Yu et al., 2024), MathInstruct (Yue et al., 2023), UltraFeedback (Cui et al., 2024), and UltraInteract (Yuan et al., 2024b), whether for SFT or preference learning. The solutions are generated using different models or follow varying styles depending on the specific needs. This naturally leads to a situation where a single instruction with multiple responses becomes a readily available resource. GRAPE therefore leverages these pre-existing response candidates to tailor training dataset that better aligns with the base model's distribution. When such resources are unavailable or insufficient, practitioners can generate new responses and apply GRAPE.
For each instruction, we collect multiple responses from various datasets. These responses form a candidate set associated with the instruction.
this section cite: ['b77', 'b114', 'b16', 'b67', 'b91', 'b59', 'b115', 'b137', 'b141', 'b19']

Section: Customize Dataset For Models
We then compute the conditional probability of each response π θ0 (y j i | x i ). Practically, we format each example using a simple prompt template: Question: {instruction} \n Answer: {response}. For each instruction, we rank its candidate responses based on the conditional log-probability assigned by the base model, normalized by response length. This is equivalent to ranking them from lowest to highest perplexity where Perplexity = exp -1 N N t=1 log P (x t | x <t ) . We then select the responses with the highest normalized probability (i.e., lowest perplexity) for supervision. Figure 4 shows a clear difference in models' choices among the same candidate pool, indicating that GRAPEselected datasets are highly customized towards different models.
Since GRAPE only involves forward-pass log-probability computation (no gradients or optimization), it is highly efficient and simple to integrate into any SFT pipeline with minimal overhead compared with model-based data selection approaches (Xia et al., 2024;Yang et al., 2024b;Liu et al., 2024b;Zhao et al., 2021;Zhang et al., 2024b;Pan et al., 2024). Additionally, it is important to distinguish GRAPE from the other perplexity-based data selection and curriculum planning methods (Wu et al., 2024;Li et al., 2024b;Liu et al., 2024c). Existing approaches focus on selecting instructions by using perplexity as a difficulty measure, which differs from GRAPE that uses probability to select for each instruction in a fixed instruction set, responses that better matches with the base model's distribution. Our experiments in §5 demonstrate that low-probability responses with fixed set of instructions are detrimental to performance, further emphasizing the fundamental difference in the two processes.
this section cite: ['b28', 'b147', 'b94', 'b122']

Section: Controlled Experiments Show the Benefits of Distributional Alignment
We conduct controlled experiments using UltraInteract-SFT to test whether selecting responses aligned with a base model's distribution improves fine-tuning outcomes more than relying on stronger generators or scaling data size. This setup-focused on verifiable tasks in coding, logic, and math-isolates the effect of distribution matching. Results show that GRAPE-selected responses consistently outperform alternatives, validating distributional alignment as a key supervision signal. These findings motivate our broader evaluations in §5.
this section cite: []

Section: Experimental Setup
Training Data Curation In this controlled experiment, we focus on chain-of-thought reasoning (Wei et al., 2022;Wang et al., 2024a;Luo et al., 2024;Cobbe et al., 2021;Li et al., 2023;Lightman et al., 2023). Different models may follow different reasoning paths to solve a problem, while their final solutions can be easily verified.
We use UltraInteract-SFT (Yuan et al., 2024b), which contains approximately 80, 800 unique instructions covering coding, math (chain-of-thought and program-aided) and logic reasoning domains , where each instruction is paired with varying numbers (avg. 3.5/instruction) of different responses to contain a total > 280, 000 training examples. The responses in the dataset are strictly in step-wise format. For each instruction, we construct a response pool consisting of both (i) original UltraInteract-SFT responses and (ii) additional responses generated from a diverse set of LLMs. GRAPE is then applied to select the most in-distribution response per instruction to this enlarged candidate pool and ensure the number of responses matches the original UltraInteract-SFT dataset for fair comparisons.
We collect responses from a diverse set of models of various sizes across model families, including MIXTRAL-7X7B-INSTRUCT (Jiang et al., 2024), CODESTRAL-22B (MistralAI, 2024a), MISTRAL-SMALL (MistralAI, 2024b), LLAMA-3.1-70B-INSTRUCT and LLAMA-3.1-405B-INSTRUCT (Dubey et al., 2024), and QWEN2.5-72B-CHAT (Yang et al., 2024a), resulting in approximately 10x additional responses per instruction. The responses are then filtered based on the answers to ensure their validity following Yuan et al. (2024b).
Base Models To demonstrate the generalizability of GRAPE, we evaluate its performance across multiple LLMs, including LLAMA-3.1-8B and LLAMA-3.2-3B from LLAMA-3 (Grattafiori et al., 2024) family, MISTRAL-7B (Jiang et al., 2023)  and QWEN2.5-7B (Hui et al., 2024). We ensure that all training configurations (GRAPE, baselines) use the same number of instructions and responses per instruction as original UltraInteract-SFT, unless otherwise stated.
Evaluation We evaluate the model on coding and math reasoning benchmarks. For coding tasks, we consider HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), LeetCode (Guo et al., 2024a); for math datasets, we consider MATH dataset (Hendrycks et al., 2021b), GSM-Plus (Li et al., 2024d) and TheoremQA (Chen et al., 2023b) dataset. HumanEval and MBPP are natural-language-to-code benchmarks testing language models' ability to produce functionally correct programs. LeetCode contains interview-level programming problems that are more challenging. MATH contains highschool level math competition problems, whereas GSM-Plus is a more challenging variant of GSM-8k (Cobbe et al., 2021) and Theorem-QA contains complex math reasoning problems.
Baselines We compare GRAPE against several baselines to isolate the impact of response selection: Original Dataset performs standard SFT on the unmodified UltraInteract-SFT dataset, which includes verified correct responses. Strongest-Model Responses uses only responses from the most powerful generator available-LLAMA3.1-405B-INSTRUCT. This represents a strong upper bound on data quality and helps determine whether GRAPE offers additional gains beyond simply choosing a strong model. 3×Data: use 3 times the number of distinct, validated responses per instruction relative to UltraInteract, while keeping all training hyperparameters (learning rate, number of epochs, etc.) fixed. It directly tests whether GRAPE's gains stem from strategic data selection rather than merely scaling up data volume. 1 We train all models for 1 epoch with a learning rate of 10 -5 .
this section cite: ['b120', 'b78', 'b16', 'b68', 'b72', 'b51', 'b28', 'b50', 'b47', 'b14', 'b5', 'b16']

Section: MATH

this section cite: []

Section: Results and Analysis
Table 1 summarizes the performance of GRAPE across benchmarks. Our approach consistently outperforms the various baselines across the board, including the original UltraInteract-SFT dataset. GRAPE-selected solutions can outperform those directly sampled from the strongest model under consideration (LLAMA3.1-405B-INSTRUCT) up to 13.8% absolute improvement. This implies that customization for base models should be prioritized over identifying the presumably highest-quality responses. This verifies our central premise that being in-distribution with each base model is an important ingredient for the responses we supervise the base models on, to boosting downstream performance. Furthermore, we demonstrate that merely adding more responses does not always lead to continuous improvement in model performance, which aligns with findings in prior studies (Li et al., 2024c;Du et al., 2023). By properly aligning with models' base distributions, GRAPE outperforms those trained with 3x responses with at least 3.6% and up to 17.3% absolute improvement. These results reinforce the notion that scaling data without considering its alignment with the base model's initial distribution risks diminishing returns and, in some cases, even performance degradation.
this section cite: ['b27']

Section: GRAPE-Picking From Real-World SFT Datasets
In this section, we leverage the findings from the earlier experiments and demonstrate the effectiveness of GRAPE to customize training data for each base model by selecting from available datasets with overlapping instructions. Here, we do not generate any new responses for the instructions; it only selects from existing ones. We evaluate GRAPE on the fully open dataset used in post-training phases of TULU-3 (Lambert et al., 2024) and OLMO-2 (OLMo et al., 2025). The details are presented below.
We discuss additional results in Appendix B  12.79 9.08 8.70 11.00 10.10 6.65 34.40 7.29 9.80 10.56 9.41 9.92 10.05 7.89 8.78 33.59 10.67 14.82 12.25 9.88 12.65 14.23 12.45 13.04 6.78 10.99 8.55 7.19 9.50 39.76 7.33 9.91 15.62 15.62 18.75 14.06 12.50 6.25 10.94 6.25 9.43 27.36 15.09 12.26 11.32 11.32 4.72 8.49 22.00 12.53 13.03 14.56 13.34 6.42 9.98 8.15 11.69 14.43 14.52 18.81 17.17 7.76 7.21 8.40 7.63 13.36 31.68 10.31 12.98 8.40 8.02 7.63 5 10 15 20 25 30 35 Percentage Figure 4: Breakdown of GRAPE-selected responses for 1K Tulu instructions vary significantly across base models, reflecting its highly model-oriented nature over responses. Details in Appendix H From the preference data, we retained only the winning responses. We formed our candidate pool with those instructions with at least two distinct responses, resulting in a dataset of 350.4K unique instructions and about 1.03 million total instructionresponse pairs for evaluation with GRAPE.
We do not apply further processing of these data or any filtering on top of GRAPE.
this section cite: ['b59']

Section: Evaluation
We evaluate on a set of commonly used benchmarks spanning over coding, math, knowledge and instruction-following. We evaluated on LeetCode (Guo et al., 2024a), MATH (Hendrycks et al., 2021b), Big-BenchHard(BBH) (Suzgun et al., 2022), MMLU (Hendrycks et al., 2021a), and AlpacaEval-V2 (Dubois et al., 2024). Leet-Code, MATH, BBH and MMLU are evaluated the in the same way as in (Yuan et al., 2024b), where we use zero-shot for MATH and MMLU, 3-shot example for BBH. We use the same AlpacaEval-v2 as in OpenInstruct.
this section cite: ['b107', 'b29']

Section: Baselines
We extensively compare against three baseline types: controlled baselines with fixed instructions, scaling baselines with increased data, and state-of-the-art selection methods. Additional details in Appendix C
Controlled Baselines We include three baselines to isolate the effect of GRAPE's response selection. SFT-only replaces GRAPE-selected responses with those from the original SFT dataset, pairing each instruction with a standard reference response to measure improvement over presumably good SFT responses. Random selects candidate responses uniformly at random from the pool for the same set of instructions, establishing a noise-tolerant baseline. Reverse-GRAPE instead selects responses with the highest perplexity. We test if responses diverging the most from the base model's distribution degrade performance, providing a contrast that sharpens the effectiveness of GRAPE.
Scaling Baselines To assess how GRAPE compares under larger-scale training, we consider three scaling-oriented baselines. Tulu3-SFT uses all 939K SFT instances from the Tulu3 training mixture to test whether GRAPE can still outperform despite using only a subset of the instruction pool. All Responses trains over the entire 1.04M-instance candidate pool to demonstrate the effect of selection versus brute-force inclusion. All Available Data uses all 1.58M instruction-response pairs under consideration, roughly 4.5× the data used by GRAPE, to test whether data volume alone suffices.
this section cite: []

Section: State-Of-The-Art SFT Data Selection Approaches
We compare GRAPE against recent state-ofthe-art data selection methods. LESS (Xia et al., 2024) selects data based on influence scores on validation tasks. We follow the implementation setup from Dai et al. (Dai et al., 2025). Emb-NV selects training data close to validation in embedding space using NV-Embed-V2 (Lee et al., 2025), following (Ivison et al., 2025). S2L (Yang et al., 2024b) clusters data via loss trajectories from small reference models (LLAMA-3.2-1B, QWEN2.5-0.5B, MISTRAL-V0.3-7B) and samples uniformly across clusters to match GRAPE's data budget.
AlpacaEval2 Model Data Num. Instances LC WR BBH MMLU MATH LeetCode Avg. Abs. ∆ Highest 350.4k 10.1 6.2 68.9 63.2 22.9 13.3 30.8 5.2 Random 350.4k 12.8 10.8 68.6 63.1 27.9 13.3 32.8 3.2 SFT-Only 350.4k 7.1 5.5 68.9 64.1 20.2 17.2 30.5 5.4 Tulu3-SFT 939k 12.4 8.0 67.9 65.9 31.5* 7.8 32.4 3.5 All Responses 1.03M 12.9 11.4 68.7 62.8 32.1 17.2 34.2 1.8 S2L 350.4k 8.5 7.6 68.6 63.1 26.5 16.1 31.7 4.2 Emb-NV 350.4k 8.4 7.0 67.9 63.7 32.1 17.2 32.7 3.2 LESS 350.4k 7.3 6.0 68.2 63.3 25.1 16.1 31.0 4.0 All Available Data 1.58M 8.8 10.1 69.8 62.1 32.5 16.1 33.2 2.7 LLAMA 3.1-8B GRAPE 350.4k 14.8 15.2 69.6 64.5 32.1 19.4 35.9 -Highest 350.4k 7.0 5.5 58.8 56.1 15.1 10.6 25.5 6.4 Random 350.4k 10.6 9.2 60.0 57.8 19.6 11.1 28.1 3.9 SFT-Only 350.4k 7.1 5.3 53.9 57.0 14.4 12.0 24.9 7.0 Full-SFT-Data 939k 11.5 10.5 59.0 55.2 25.8 15.6 29.9 2.0 All Responses 1.03M 10.5 11.5 61.0 57.9 24.2 14.4 29.9 2.0 S2L 350.4k 10.5 11.9 61.9 57.1 22.4 13.9 29.6 2.3 Emb-NV 350.4k 6.0 5.2 60.7 56.5 23.9 11.7 27.3 4.6 LESS 350.4k 6.4 4.8 59.2 55.4 16.0 8.3 25.0 6.9 All Available Data 1.58M 8.0 7.0 55.3 53.7 25.4 12.3 26.9 5.0 MISTRAL -7B GRAPE 350.4k 13.6 13.9 62.3 59.2 24.2 18.3 31.9 -Highest 350.4k 8.0 10.7 72.2 73.2 49.4 42.2 42.6 5.9 Random 350.4k 16.1 14.9 73.3 73.1 56.0 43.3 46.1 2.4 SFT-Only 350.4k 9.9 7.8 71.2 74.1 51.1 46.6 43.4 5.1 Full-SFT-Data 939k 9.5 7.1 71.4 73.1 47.0 48.3 42.7 5.8 All Responses 1.03M 16.0 14.5 71.4 72.1 51.7 43.3 44.8 3.7 S2L 350.4k 13.4 14.9 72.7 73.1 53.4 40.6 44.7 3.9 Emb-NV 350.4k 11.9 10.6 72.1 72.5 53.3 43.3 44.0 4.6 LESS 350.4k 7.8 5.9 71.3 72.9 48.3 41.1 41.2 7.4 All Available Data 1.58M 13.3 12.3 70.3 71.8 44.0 42.8 42.4 6.1 QWEN2.5 -7B GRAPE 350.4k 20.0 20.4 73.2 73.3 60.0 44.4 48.6 -Table 2: GRAPE on the Tulu-Olmo collection. For Llama3.1-8B base model, we included Tulu3-SFT model's results. The "*"-marked number for MATH is obtained using 4-shot prompting. We train all the models (except that we took Tulu3-SFT numbers directly) for 1 epoch with a learning rate of 10 -5 . Abs.∆ is GRAPE's absolute improvement on average performance over that row. 5.4 Results GRAPE QWEN2.5 -72B LLAMA3.1 -405B GEMMA -IT-9B LC 28.1 25.8 16.3 26.9 WR 33.1 24.3 17.0 20.2 Table 3: Alpaca-Eval2 On Magpie-Zoo Xu et al. (2024d).
As shown in Table 2, models fine-tuned on responses selected by GRAPE outperforms the strong baselines we constructed, especially the one that trains over all available data by significant margins across the 3 models.
Remarkably, using roughly 1/6 training computation (Tulu3-8B-SFT was trained for 2 epochs on 3 times of data), our performance exceeds that of TULU3-8B-SFT.
Also, GRAPE outperforms state-of-the-art data-selection approaches like S2L, despite its simplicity and efficiency, further highlighting its effectiveness in diverse real-world scenarios and making it a practical option for real-world SFT setups with minimal data engineering effort. Without the need to synthesize any new data, one can easily leverage established datasets sourced from the web to customize a dataset for each base model that yields better fine-tuning outcome.
These results highlight GRAPE as an effective and efficient selection strategy for real-world SFT.
this section cite: ['b28', 'b20', 'b62', 'b49']

Section: Ablations Remains Effective Even with a Single Response Source
In Section 4, we demonstrated how GRAPE enables practitioners to refine model-generated responses for improved training outcomes.
Beyond that, GRAPE can optimize responses from a single generator. The Magpie-Zoo (Xu et al., 2024d) dataset contains a fixed set of instructions and multiple versions of response sets each generated by different language models. Using the Magpie-Zoo instruction set, we sample 10 responses per instruction from Qwen2.5-72B-Instruct, select in-distribution responses with GRAPE, to train a Mistral-v0.3-7B model. We compare with replicas that achieve top-3 performance from Magpie-Zoo.
this section cite: []

Section: Model Data Avg.
Self-Distilled 28.4(-) MISTRAL-7B Original-UI 32.7 Self-Distilled 29.4(-) LLAMA3.1-8B Original-UI 36.8 Self-Distilled 15.1(-) LLAMA3.2-3B
Original-UI 20. 3 Table 4: Performance Degradation From Self-Distillation On UI.
As shown in Table 3, GRAPE-selected responses further boost the performance. Given that batch sampling from a strong model introduces minimal latency (Zhong et al., 2024;Zhou et al., 2024e), this result positions GRAPE as a practical and efficient data curation strategy-achieving strong results with just a single generator and no additional engineering overhead.
this section cite: ['b151']

Section: Why GRAPE Outperforms Self-Generated Responses
To probe the effectiveness of GRAPE, we ablate it against a degenerate alternative: self-generation, where the model is fine-tuned on its own outputs.
Dataset Model Response Generator N Acc. MATH MISTRAL Llama3.1-70B 10 18.2 MATH MISTRAL FT-Mistral 10 15.9 (-) MATH LLEMMA Llama3.1-70B 10 26.2 MATH LLEMMA FT-Llemma 10 23.6 (-) MATH MISTRAL MM-AnsAug -22.3 MATH MISTRAL FT-Mistral 10 20.6 (-) MATH LLEMMA MM-AnsAug -28.1 MATH LLEMMA FT-Llemma 10 21.4 (-)
Table 5: Self-generation on MATH dataset. FT-MISTRAL refers to the model right above that row finetuned from either MM-AnsAug or Llama3.1-70B-Instruct produced solutions. N stands for the number of responses sampled.
We fine-tune a base model using responses generated by its previously fine-tuned variant. This setup consistently degrades performance (Table 4). We further confirm this on the MATH dataset (Hendrycks et al., 2021b): responses from strong models like LLAMA3.1-70B-INSTRUCT or MetaMathQA (Yu et al., 2024) are used to fine-tune a model that then generates new solutions for another round of fine-tuning. Again, performance drops (Table 5).
This failure arises from distributional collapse (see § 2.2): self-generated responses become increasingly narrow and repetitive, reinforcing biases and reducing exposure to diverse reasoning. Correctness alone is insufficient-external diversity is essential for generalization.
GRAPE avoids this collapse by selecting external responses that are both diverse and distributionaligned, preserving semantic breadth and stylistic variability while staying true to the model's pretraining. This enables more stable and generalizable fine-tuning.
this section cite: ['b137']

Section: Conclusion
We present GRAPE, a simple yet effective method for improving supervised fine-tuning by selecting responses aligned with the base model's pretrained distribution. GRAPE requires only a forward pass over candidate responses, making it highly efficient and easy to integrate. Despite its simplicity, GRAPE consistently outperforms stronger baselines using significantly larger datasets and surpasses more complex, costly data selection methods. Our study affirms that carefully aligning SFT data with a model's pretrained distribution yields substantial performance and efficiency gains. In contrast, GRAPE focuses on the quality of supervision-that is, selecting responses to provide the most effective learning signal. Importantly, this response-centric perspective is complementary, not contradictory, to instance-level or instruction-based selection methods; both might be combined to enhance overall data quality and training efficiency.
Limitations Like many other data selection algorithms (Du et al., 2023;Li et al., 2024c;Xia et al., 2024;Das & Khetan, 2023;Kang et al., 2024;Mekala et al., 2024;Yang et al., 2024c;Zhang et al., 2024b;Pan et al., 2024;Dai et al., 2025;Ivison et al., 2025;Bhatt et al., 2024;Yin & Rush, 2024;Liu et al., 2024b), GRAPE assumes a quality-controlled candidate pool from which to select samples. Furthermore, because GRAPE relies on the base model itself, its selection effectiveness may be influenced by the model's inherent capabilities. Fujimoto, S., Meger, D., and Precup, D. Off-policy deep reinforcement learning without exploration. In International Conference on Machine Learning, 2018. URL https://api.  semanticscholar.org/CorpusID:54457299.
Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Korbak, T., Sleight, H., Agrawal, R., Hughes, J., Pai, D. B., Gromov, A., Roberts, D., Yang, D., Donoho, D. L., and Koyejo, S. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=  5B2K4LRgmz.
Ghosh, S., Evuru, C. K. R., Kumar, S., S, R., Aneja, D., Jin, Z., Duraiswami, R., and Manocha, D. A closer look at the limitations of instruction tuning, 2024. URL https://arxiv.org/abs/2402.  05119.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra, C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D., Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes, D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan, E., Smith, E. M., Radenovic, F., Guzmán, F., Zhang, F., Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail, G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A., Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J., Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V., Prasad, K., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary, L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L., Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat, L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh, M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham, M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M., Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N., Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N., Duchenne, O., C ¸elebi, O., Alrassy, P., Zhang, P., Li, P., Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan, P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan, R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic, R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R., Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva, R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S., Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang, S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang, S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S., Collot, S., Gururangan, S., Borodinsky, S., Herman, T., Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speckbacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V., Gupta, V., Ramanathan, V., Kerkez, V., Gonguet, V., Do, V., Vogeti, V., Albiero, V., Petrovic, V., Chu, W., Xiong, W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang, X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Goldschlag, Y., Gaur, Y., Babaei, Y., Wen, Y., Song, Y., Zhang, Y., Li, Y., Mao, Y., Coudert, Z. D., Yan, Z., Chen, Z., Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey, A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand, A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A., Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A., Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco, A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A., Bharambe, A., Eisenman, A., Yazdan, A., James, B., Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola, B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock, B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B., Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C., Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C., Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty, D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine, D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang, D., Le, D., Holland, D., Dowling, E., Jamil, E., Montgomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T., Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun, F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Caggioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz, G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov, G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H., Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H., Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan, H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I., Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli, J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J., Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J., Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard,
this section cite: ['b27', 'b28', 'b21', 'b53', 'b84', 'b94', 'b20', 'b49', 'b7', 'b136']

Section: References
Ref_id:b0 Title: Better fine-tuning by reducing representational collapse Year: (2020)
Ref_id:b1 Title: Self-consuming generative models go MAD Year: (2024)
Ref_id:b2 Title: Towards understanding ensemble, knowledge distillation and self-distillation in deep learning Year: (2023)
Ref_id:b3 Title: Perplexed by perplexity: Perplexity-based data pruning with small reference models Year: (2024)
Ref_id:b4 Title: A theory for emergence of complex skills in language models Year: (2023)
Ref_id:b5 Title: Program synthesis with large language models Year: (2021)
Ref_id:b6 Title: A general theoretical paradigm to understand learning from human preferences Year: (2023)
Ref_id:b7 Title: An experimental design framework for label-efficient supervised finetuning of large language models Year: (2024)
Ref_id:b8 Title: Nepotistically trained generative-ai models collapse Year: (2023)
Ref_id:b9 Title: Large language models suffer from their own output: An analysis of the self-consuming training loop Year: (2023)
Ref_id:b10 Title: Language models are few-shot learners Year: (2020)
Ref_id:b11 Title: Balancing cost and effectiveness of synthetic data generation strategies for llms Year: (2024)
Ref_id:b12 Title: Generating millions of instructions from a handful of prompts Year: (2024)
Ref_id:b13 Title: Training a better alpaca with fewer data Year: (2023)
Ref_id:b14 Title:  Year: (2021)
Ref_id:b15 Title: Theoremqa: A theorem-driven question answering dataset Year: (2023)
Ref_id:b16 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b17 Title: Ask your distribution shift if pre-training is right for you Year: (2024)
Ref_id:b18 Title: Elements of Information Theory Year: (2006)
Ref_id:b19 Title: Boosting language models with scaled ai feedback Year: (2024)
Ref_id:b20 Title: Improving influence-based instruction tuning data selection for balanced learning of diverse capabilities Year: (2025)
Ref_id:b21 Title: Deft: Data efficient fine-tuning for large language models via unsupervised core-set selection Year: (2023)
Ref_id:b22 Title: Databricks dolly-15k Year: (2023)
Ref_id:b23 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b24 Title: Parameter-efficient fine-tuning of large-scale pre-trained language models Year: (2023)
Ref_id:b25 Title: Strong model collapse Year: (2024)
Ref_id:b26 Title: Self-boosting large language models with synthetic preference data Year: (2025)
Ref_id:b27 Title: Mods: Model-oriented data selection for instruction tuning Year: (2023)
Ref_id:b28 Title:  Year: (2024)
Ref_id:b29 Title: Length-controlled alpacaeval: A simple way to debias automatic evaluators Year: (2024)
Ref_id:b30 Title:  Year: (2024)
Ref_id:b31 Title: Open r1: A fully open reproduction of deepseek-r1 Year: (2025-01)
Ref_id:b32 Title: Does learning require memorization? a short tale about a long tail Year: (2021)
Ref_id:b33 Title: Citing: Large language models create curriculum for instruction tuning Year: (2023)
Ref_id:b34 Title:  Year: (2024)
Ref_id:b35 Title: Reinforced self-training (rest) for language modeling Year: (2023)
Ref_id:b36 Title: Deepseek-coder: When the large language model meets programmingthe rise of code intelligence Year: (2024)
Ref_id:b37 Title: Direct language model alignment from online ai feedback Year: (2024)
Ref_id:b38 Title: The curious decline of linguistic diversity: Training language models on synthetic text Year: (2023)
Ref_id:b39 Title: Evaluation of similarity-based explanations Year: (2021)
Ref_id:b40 Title: Will large-scale generative models corrupt future datasets? Year: (2023-10)
Ref_id:b41 Title: Preserving pre-trained features helps calibrate fine-tuned language models Year: (2023)
Ref_id:b42 Title: Measuring massive multitask language understanding Year: (2021)
Ref_id:b43 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b44 Title: Collapse of self-trained language models Year: (2024)
Ref_id:b45 Title:  Year: (2024)
Ref_id:b46 Title:  Year: (2024)
Ref_id:b47 Title: Qwen2.5-coder technical report Year: (2024)
Ref_id:b48 Title: Information Geometry and Its Applications Year: (2016)
Ref_id:b49 Title: Large-scale data selection for instruction tuning Year: (2025)
Ref_id:b50 Title:  Year: (2023)
Ref_id:b51 Title:  Year: (2024)
Ref_id:b52 Title: Doubly robust off-policy value evaluation for reinforcement learning Year: (2016)
Ref_id:b53 Title: Get more for less: Principled data selection for warming up fine-tuning in llms Year: (2024)
Ref_id:b54 Title: Openassistant conversations -democratizing large language model alignment Year: (2023)
Ref_id:b55 Title: Understanding catastrophic forgetting in language models via implicit inference Year: (2024)
Ref_id:b56 Title: Stabilizing off-policy q-learning via bootstrapping error reduction Year: (2019)
Ref_id:b57 Title: Fine-tuning can distort pretrained features and underperform out-of-distribution Year: (2022)
Ref_id:b58 Title: Active instruction tuning: Improving cross-task generalization by training on prompt sensitive tasks Year: (2023)
Ref_id:b59 Title: Tulu 3: Pushing frontiers in open language model post-training Year: (2024)
Ref_id:b60 Title: Evaluating distributional distortion in neural language modeling Year: (2021)
Ref_id:b61 Title: Instruction tuning with human curriculum Year: (2024)
Ref_id:b62 Title: Nv-embed: Improved techniques for training llms as generalist embedding models Year: (2025)
Ref_id:b63 Title: Selective reflection-tuning: Student-selected data recycling for LLM instruction-tuning Year: (2024-08)
Ref_id:b64 Title: Superfiltering: Weak-to-strong data filtering for fast instruction-tuning Year: (2024)
Ref_id:b65 Title: From quantity to quality: Boosting LLM performance with self-guided data selection for instruction tuning Year: (2024-06)
Ref_id:b66 Title: Gsm-plus: A comprehensive benchmark for evaluating the robustness of llms as mathematical problem solvers Year: (2024)
Ref_id:b67 Title: Competitionlevel code generation with alphacode Year: (2022-12)
Ref_id:b68 Title: Making language models better reasoners with step-aware verifier Year: (2023-07)
Ref_id:b69 Title: Scar: Efficient instruction-tuning for large language models via style consistency-aware response ranking Year: (2024)
Ref_id:b70 Title: Openorca: An open dataset of gpt augmented flan reasoning traces Year: (2023)
Ref_id:b71 Title: Slimorca: An open dataset of gpt-4 augmented flan reasoning traces Year: (2023)
Ref_id:b72 Title: Let's verify step by step Year: (2023)
Ref_id:b73 Title: Best practices and lessons learned on synthetic data for language models Year: (2024)
Ref_id:b74 Title: What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning Year: (2024)
Ref_id:b75 Title: Let's learn step by step: Enhancing in-context learning ability with curriculum learning Year: (2024)
Ref_id:b76 Title: Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer Year: (2024)
Ref_id:b77 Title: The flan collection: Designing data and methods for effective instruction tuning Year: (2023)
Ref_id:b78 Title: Improve mathematical reasoning in language models by automated process supervision Year: (2024)
Ref_id:b79 Title: An empirical study of catastrophic forgetting in large language models during continual fine-tuning Year: (2025)
Ref_id:b80 Title: When less is more: Investigating data pruning for pretraining llms at scale, 2023a Year: ()
Ref_id:b81 Title: When less is more: Investigating data pruning for pretraining llms at scale, 2023b Year: ()
Ref_id:b82 Title: Combining generative artificial intelligence (ai) and the internet: Heading towards evolution or degradation? Year: (2023)
Ref_id:b83 Title: Towards understanding the interplay of generative artificial intelligence and the internet Year: (2023)
Ref_id:b84 Title: Smaller language models are capable of selecting instructiontuning training data for larger language models Year: (2024)
Ref_id:b85 Title: Aligning codellms with direct preference optimization Year: (2024)
Ref_id:b86 Title: Prioritized training on points that are learnable, worth learning, and not yet learnt Year: (2022)
Ref_id:b87 Title:  Year: (2024)
Ref_id:b88 Title: Mistral-small-instruct Year: (2024)
Ref_id:b89 Title: Codestral-22b-v0.1, 2024c Year: ()
Ref_id:b90 Title: Self-distillation amplifies regularization in hilbert space Year: (2020)
Ref_id:b91 Title:  Year: (2025)
Ref_id:b92 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b93 Title: Does writing with language models reduce content diversity Year: ()
Ref_id:b94 Title: Scalable bilevel optimization for llm data reweighting Year: (2024)
Ref_id:b95 Title: Can llms select important instructions to annotate? arXiv preprint Year: (2024)
Ref_id:b96 Title: Deep learning on a data diet: Finding important examples early in training Year: (2023)
Ref_id:b97 Title: Advantage-weighted regression: Simple and scalable off-policy reinforcement learning Year: (2019)
Ref_id:b98 Title: Unleashing the power of data tsunami: A comprehensive survey on data assessment and selection for instruction tuning of language models Year: (2024)
Ref_id:b99 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b100 Title: Learning to retrieve prompts for in-context learning Year: (2022)
Ref_id:b101 Title: A mathematical exploration of why language models help solve downstream tasks Year: (2021)
Ref_id:b102 Title: Rl on incorrect synthetic data scales the efficiency of llm math reasoning by eight-fold Year: (2024)
Ref_id:b103 Title: Offline reinforcement learning with on-policy q-function regularization Year: (2023)
Ref_id:b104 Title: The curse of recursion: Training on generated data makes models forget Year: (2023)
Ref_id:b105 Title: Ai models collapse when trained on recursively generated data Year: (2024-07)
Ref_id:b106 Title: Principle-driven self-alignment of language models from scratch with minimal human supervision Year: (2023)
Ref_id:b107 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: (2022)
Ref_id:b108 Title: Preference fine-tuning of llms should leverage suboptimal Year: (2024)
Ref_id:b109 Title: On a connection between importance sampling and the likelihood ratio policy gradient Year: (2010-01)
Ref_id:b110 Title: Understanding the performance gap between online and offline alignment algorithms Year: (2024)
Ref_id:b111 Title: Understanding the performance gap between online and offline alignment algorithms Year: (2024)
Ref_id:b112 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b113 Title:  Year: (2025-01)
Ref_id:b114 Title: Vicuna llm: An open-source chatbot developed by fine-tuning the llama model on user-shared conversations, achieving performance comparable to other advanced chatbots Year: (2023)
Ref_id:b115 Title: Openhermes 2.5: An open dataset of synthetic data for generalist llm assistants Year: (2023)
Ref_id:b116 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2024)
Ref_id:b117 Title: Diversity measurement and subset selection for instruction tuning datasets Year: (2024)
Ref_id:b118 Title: Selfinstruct: Aligning language models with self-generated instructions Year: (2023)
Ref_id:b119 Title:  Year: (2021)
Ref_id:b120 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b121 Title: Source code is all you need Year: (2023)
Ref_id:b122 Title: Curriculum learning with quality-driven data selection Year: (2024)
Ref_id:b123 Title: Training trajectories of language models across scales Year: (2023)
Ref_id:b124 Title: LESS: Selecting influential data for targeted instruction tuning Year: ()
Ref_id:b125 Title: Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KLconstraint Year: (2024)
Ref_id:b126 Title: Empowering large pre-trained language models to follow complex instructions Year: ()
Ref_id:b127 Title: Is dpo superior to ppo for llm alignment? a comprehensive study Year: ()
Ref_id:b128 Title: Rethinking the instruction quality: Lift is what you need Year: (2023)
Ref_id:b129 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b130 Title: Stronger models are not stronger teachers for instruction tuning Year: (2024)
Ref_id:b131 Title: Learning to reason under off-policy guidance Year: (2025)
Ref_id:b132 Title:  Year: (2024)
Ref_id:b133 Title: Smalltolarge (s2l): Scalable data selection for fine-tuning large language models by summarizing training trajectories of small models Year: (2024)
Ref_id:b134 Title: Smalltolarge (s2l): Scalable data selection for fine-tuning large language models by summarizing training trajectories of small models Year: (2024)
Ref_id:b135 Title: Self-distillation bridges distribution gap in language model fine-tuning Year: (2024)
Ref_id:b136 Title: Compute-constrained data selection Year: (2024)
Ref_id:b137 Title: Bootstrap your own mathematical questions for large language models Year: (2024)
Ref_id:b138 Title: Self-play fine-tuning of diffusion models for text-to-image generation Year: ()
Ref_id:b139 Title: c1c657deafe09f64c013c2888bd7b-Paper-Conference Year: (2024)
Ref_id:b140 Title: Advancing llm reasoning generalists with preference trees Year: (2024)
Ref_id:b141 Title: Building math generalist models through hybrid instruction tuning Year: (2023)
Ref_id:b142 Title: Automatic instruction evolving for large language models Year: (2024)
Ref_id:b143 Title: PLUM: Improving code lms with execution-guided on-policy preference learning driven by synthetic test cases Year: (2024)
Ref_id:b144 Title: Tagcos: Task-agnostic gradient clustered coreset selection for instruction tuning data Year: (2024)
Ref_id:b145 Title: Self-exploring language models: Active preference elicitation for online alignment Year: (2024)
Ref_id:b146 Title: Forcing diffuse distributions out of language models Year: ()
Ref_id:b147 Title: Dataset condensation with gradient matching Year: (2021)
Ref_id:b148 Title: Wildchat: 1m chatGPT interaction logs in the wild Year: ()
Ref_id:b149 Title: A preliminary study of the intrinsic relationship between complexity and alignment Year: (2024)
Ref_id:b150 Title: LMSYS-chat-1m: A large-scale real-world LLM conversation dataset Year: (2024)
Ref_id:b151 Title: Disaggregating prefill and decoding for goodput-optimized large language model serving Year: (2024)
Ref_id:b152 Title: Less is more for alignment Year: (2023)
Ref_id:b153 Title: Gauging learnability in supervised fine-tuning data Year: (2024)
Ref_id:b154 Title: Davir: Data selection via implicit reward for large language models Year: (2024)
Ref_id:b155 Title: Enhancing rlhf with weighted preference optimization Year: (2024)
Ref_id:b156 Title: Explore spurious correlations at the concept level in language models for text classification Year: (2024)
Ref_id:b157 Title: A survey on efficient inference for large language models Year: (2024)
Ref_id:b158 Title: Behavior proximal policy optimization Year: (2023)
Ref_id:b159 Title: LogProb-based Methods Year: ()
Ref_id:b160 Title: LogProb-based methods also directly utilize the target LLM to evaluate the utility of each training data point Year: ()
Ref_id:b161 Title: 1 Simple Uncertainty-based Indicators Year: ()
