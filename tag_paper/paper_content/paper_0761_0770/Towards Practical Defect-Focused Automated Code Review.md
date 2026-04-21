Title: Towards Practical Defect-Focused Automated Code Review
Abstract: The complexity of code reviews has driven efforts to automate review comments, but prior approaches oversimplify this task by treating it as snippet-level code-to-text generation and relying on text similarity metrics like BLEU for evaluation. These methods overlook repository context, real-world merge request evaluation, and defect detection, limiting their practicality. To address these issues, we explore the full automation pipeline within the online recommendation service of a company with nearly 400 million daily active users, analyzing industry-grade C++ codebases comprising hundreds of thousands of lines of code. We identify four key challenges: ❶ capturing relevant context, ❷ improving key bug inclusion (KBI), ❸ reducing false alarm rates (FAR), and ❹ integrating human workflows. To tackle these, we propose ❶ code slicing algorithms for context extraction, ❷ a multi-role LLM framework for KBI, ❸ a filtering mechanism for FAR reduction, and ❹ a novel prompt design for better human interaction. Our approach, validated on real-world merge requests from historical fault reports, achieves a 2× improvement over standard LLMs and a 10× gain over previous baselines. While the presented results focus on C++, the underlying framework design leverages languageagnostic principles (e.g., AST-based analysis), suggesting potential for broader applicability.

Section: Introduction
Code review is essential for improving code quality and detecting defects (Fagan, 2002). Modern Code Review † Work done during the internship or tenure at Kuaishou Technology. 1 Laboratory of Precise Computing, Institute of Software, Chinese Academy of Sciences, Beijing, China 2 University of Chinese Academy of Sciences, Beijing, China 3 Kuaishou Technology, Beijing, China 4 Independent Researcher 5 Sinosoft Company Limited, Beijing, China. Correspondence to: Li Yang <yan-gli2017@iscas.ac.cn>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
(MCR) is widely used in open-source (Rigby et al., 2008;2014;Rigby & Bird, 2013) and industrial settings (Sadowski et al., 2018;Shan et al., 2022), typically involving: (A) code submission, (B) reviewer examination, (C) feedback, and (D) developer revisions.
Despite its benefits, MCR is labor-intensive and timeconsuming (Yang et al., 2016), driving research toward automated review comment generation. Existing methods-whether retrieval-based (Gupta & Sundaresan, 2018;Siow et al., 2020;Hong et al., 2022) or deep-learning-driven (Tufano et al., 2021;2022;Li et al., 2022b;a;Lin et al., 2023;Lu et al., 2023)-often frame it as a snippet-level code-to-text task. However, this oversimplification diverges from the core goal of reviewers: detecting defects (Bacchelli & Bird, 2013) (see Section A). Furthermore, current evaluations rely excessively on textual similarity metrics (e.g., BLEU (Papineni et al., 2002), ROUGE (Lin, 2004)), which fail to measure real-world effectiveness (Lu et al., 2025).
Challenges. To address these limitations, we investigate a full code review pipeline within a real-world online service (Figure 1). Our system integrates with an internal DevOps platform, generating review reports, filtering comments, and aligning them with code lines. A detailed description of this real-world workflow integration, designed for seamless adoption by developers, is provided in Appendix B. This deployment reveals four key challenges (Appendix C):
Capturing Proper Code Context: Effective review requires analyzing dependencies beyond the immediate diff hunk (e.g., variable declarations or method calls). However, excessively long inputs degrade LLM performance, necessitating efficient context extraction.
this section cite: ['b37', 'b38', 'b36', 'b39', 'b41', 'b51', 'b11', 'b42', 'b12', 'b45', 'b22', 'b26', 'b3', 'b34', 'b23', 'b28']

Section: Improving Key Bug Inclusion (KBI):
The goal of automated review is to detect critical defects, yet existing methods rely on textual similarity metrics, which fail to measure defect detection capability. More robust evaluation methods, such as Key-Bug Inclusion (KBI), are needed.
Reducing False Alarm Rates (FAR): Generative models often produce irrelevant or overly strict comments (e.g., nitpicks, hallucinations), burdening developers. A robust filtering mechanism is required to reduce false positives and enhance signal-to-noise ratio.
Human-Centric Workflow Integration: Practical review tools must seamlessly integrate into developers' workflows, ensuring comment alignment with code lines while minimizing cognitive overhead. Existing solutions often overlook this critical usability aspect.
Our Approach. To address these challenges, we propose: ❶ A static analysis system using code slicing to extract relevant context. ❷ A multi-role LLM framework with chain-of-thought reasoning to enhance defect detection. ❸ A filtering mechanism to eliminate false positive nitpicks and hallucinations. ❹ A line-aware prompt design for precise comment placement.
Evaluation. We validate our framework on real-world system failures, including historical core dumps and fault reports that caused significant financial losses. We evaluate it using multiple open-source LLM engines, demonstrating a 2× performance improvement over standard LLM methods and a 10× improvement over prior baselines. An ablation study further confirms the contribution of each component, highlighting the impact of code slicing, multi-role reasoning, and filtering mechanisms.
this section cite: []

Section: Contributions.
Our key contributions include being the first to: ❶ Repository-Level and Merge-Request Granularity: Elevating automated code review from snippet-level tasks to repository-wide and merge-request (pull-request) granularity. ❷ Integration with Real-World DevOps Workflows: Deploying automation into a practical online review system with more practical and objective evaluation metrics beyond text similarity. ❸ Validation on Industry-Scale Defects: Demonstrating effectiveness on real-world, high-impact failures in industry-level codebases instead of synthetic test data. ❹ Code-Review-Oriented LLM Framework: Designing a specialized framework leveraging code slicing, multi-role collaboration, and filtering mechanisms, achieving substantial improvement in code review performances.
this section cite: []

Section: Background: Code Review Automation
Automating code review is crucial for maintaining software quality by identifying critical bugs early. The goal is to detect severe issues in new merge requests and provide necessary comments. In 2022, company reports showed that 30% of severe P1+ incidents (asset losses exceeding $350,000) and 24.04% of P4+ incidents stemmed from lowlevel faults due to inadequate reviews. Even in 2024, changerelated core failures accounted for 67% of incidents, with code change-related graded incidents comprising 19.54%, highlighting the urgent need for effective automated review tools. These tools help ensure thorough, compliant reviews, reducing defect risks.
To understand reviewer needs, we surveyed a super reviewer group, summarizing findings in Section D. Background on code slicing and multi-role systems, key techniques in our work, is introduced in Sections E and F.
this section cite: []

Section: Proposed Approach

this section cite: []

Section: Overview
Figure 2 illustrates our decoupling process of code review automation architecture: 1) Code Slicing: Extracting code from the diff hunk within repository context (Section 3.2); 2) Multi-role Code Review System: Employing a multi-role system to conduct reviews and compile the results (Section 3.3); 3) Redundancy Comment Filter Mechanism: Filtering out redundant and irrelevant comments to avoid nitpicks and hallucinations (Section 3.4); 4) Line Number Localization: Ensuring precise identification of code lines where issues occur (Section 3.5). To evaluate the automation, we construct a dataset from historical fault reports, simulating real-world merge requests that introduced defects (Section 3.6).
this section cite: []

Section: Code Slicing
Previous work used method-level or diff-level code snippets as independent inputs. However, new code is integrated into a larger codebase during reviews, and understanding the structural context is crucial. We developed a code slicing process that integrates multiple slicing strategies, selectable based on the analysis needs. To avoid redundant slices, we use a caching mechanism to enhance efficiency.
The pseudo code of our slicing algorithms is presented in Section G. Initially, the repository is cloned, and the merge request commit is checked out. A static analysis tool is then applied to generate abstract syntax trees (ASTs), which serve as the foundation for our slicing process. Based on data dependencies and control flow analysis, one or more of the following four optional slicing algorithms may be applied: 1) Original Diff: The basic code diff without transformations, capturing essential changes in the commit. 2) Parent Function: Locates the smallest parent function containing the changes, providing functional context. 3) Left Flow: Tracks the flow of all left-hand values (L-values) in the function and control structures, focusing on the lifecycle of variables. 4) Full Flow: Extends Left Flow by tracing right-hand values (R-values) and collecting the signatures of callee functions, offering coverage of variable usage and modifications.
this section cite: []

Section: Multi-role Code Review System
Our multi-role code review system involves four key roles: Reviewer, Meta-Reviewer, Validator, and Translator. These roles collaborate to enhance the accuracy and efficiency of the review process. The system design is illustrated in Figure 3, and we detail the roles and their processes below.
❶ Reviewer: Reviews each code snippet generated by the code slicing algorithm (Section 3.2) and provides detailed comments on potential issues in a predefined format. ❷ Meta-Reviewer: Aggregates comments from multiple Reviewers, filtering and sorting them based on predefined thresholds. It merges common issues across reviews. ❸ Validator: Validates and refines the merged comments, rescores them, and ensures that only comments exceeding a certain threshold are retained. ❹ Translator: Translates the final comments into the required language for multinational teams, ensuring proper formatting for direct integration into the development environment. Each role is integrated with Chain-of-Thought technique, as detailed in Section H.
this section cite: []

Section: Redundancy Comment Filter Mechanism
LLMs often produce an overwhelming number of comments, many of which are either nitpicks or hallucinations. To mitigate this issue, we implemented a Redundancy Comment Filter Mechanism to reduce the number of irrelevant comments.
Our filtering mechanism, integrated within the multi-role system (Section 3.3), operates by answering three key questions for each comment: Q1: Is this comment a nitpick? Typical nitpicks include excessive code comments, handling unnecessary edge cases, or overly complex error handling. Q2: Does the comment identify a fake problem (i.e., a non-existent bug)? For example, if the comment flags a function call to a known reliable internal library, null pointer checks are considered irrelevant. Q3: How critical is the issue identified by this comment? Minor issues, like missing comments, are less severe than potential core dumps or infinite loops.
Each question is rated on a scale from 1 to 7, with 1 indicating a nitpick, fake problem, or minimal issue, and 7 indicating a severe and real issue. The scoring scale (1 to 7) is inspired by other related work (McAleese et al., 2024). We chose this scale to enable a fine-grained and manageable distinction. These scores form the basis of the filtering process throughout the review workflow.
Coarse Filtering and Sorting by Reviewer. During the review process, the Reviewer LLMs score each comment based on Q1-Q3. Comments with Q1 or Q2 scores of 4 or below are discarded. This specific threshold was established heuristically to enhance interpretability and has been validated by developer feedback during internal piloting. The remaining comments are then sorted based on their Q3 score and truncated to the Top-N comments.
Fine Filtering and Sorting by Meta-Reviewer. The Meta-Reviewer further refines the filtered comments by merging those flagged by multiple Reviewers and removing comments mentioned by only one Reviewer.
this section cite: ['b32']

Section: Validation and Re-scoring by Validators.
Validators then re-score the comments by revisiting the original code snippets and applying the same Q1-Q3 criteria. A secondary filter is applied, ensuring that only the most relevant and critical comments proceed to translation and integration into the development platform.
Integration with the Multi-role System. The filtered comments are processed by the remaining multi-role components, including translation (if necessary) and final submission to the development platform. This multi-stage process ensures that the delivered comments are both relevant and concise, minimizing redundancy and false alarms. The heuristic approach to threshold definition described herein was chosen to prioritize generalizability, interpretability, and mitigate overfitting in this study. While providing a robust baseline, exploring adaptive or machine-learned thresholds remains a valuable direction for future enhancement to achieve more nuanced filtering.
this section cite: []

Section: Line Number Localization
A key challenge overlooked in prior work is the precise localization of comments within the code. Unlike code summarization tasks, code reviews require pinpointing specific lines of code where issues are identified. Without this information, developers face inefficiencies in verifying and addressing comments. For example, the change-involved function has 94.54 lines of code in average based on our statistics, missing line localization can result in significant delays for developers.
We propose a code formatting approach inspired by Aider (Gauthier, 2024), tailored for code review tasks. As shown in Table 1, the format includes an operation label (indicating whether a line is kept, added, or deleted), the line number, and the code content. For non-contiguous code lines, ellipses are used to indicate omissions.
Table 1. Code formatting with line position information.
this section cite: ['b9']

Section: linenumber|{kept code line}
Represents lines that remain unchanged.
-linenumber|{deleted code line} Indicates lines that have been removed.
this section cite: []

Section: +linenumber|{added code line}
Marks newly added lines.
...|... Indicates the omission of non-essential lines.
this section cite: []

Section: Offline Validation
To systematically assess the performance of our system, we developed a dataset curated from the company's fault report platform. Each case in this dataset corresponds to an issue that resulted in actual company losses. For each reported fault, we trace back to the merge request that introduced the fault and its subsequent fixing merge request. Using these, we generate ideal reference comments containing details such as affected files, specific lines of code, fault location, root cause, suggested fix, example code, and issue category. The motivation for conducting such validation is illustrated in Section I.
this section cite: []

Section: Evaluation Design

this section cite: []

Section: Research Questions
We define the following research questions (RQs) to guide our evaluation, whose detailed illustrations are in Section J:
RQ1: How does the overall performance of our framework compare with previous works?
RQ2: How do code slicing algorithms impact the performance of the framework?
RQ3: How do the different components of the multi-role system impact the performance of our framework?
RQ4: How does the redundancy comment filter mechanism address nitpicks and hallucinations?
RQ5: How does the representation of line number position information impact overall performance and line number localization success rate?
this section cite: []

Section: Dataset and Studied Models
The primary goal of code review is to prevent problematic code from being merged into the target branch. To simulate real-world code review scenarios, we collected data from a company's core framework team, which is responsible for the production code of the short video recommendation core service. This data was gathered using fault reports recorded on an online platform. These cases come from four repositories and involve total 4,090 developers. By analyzing these reports, we traced the merge requests (MRs) that introduced the issues and examined the specific commits to reproduce the code snapshots. The detailed statistics are presented in Section K.
Our framework supports multiple LLM engines. To mitigate security risks, we only studied open-source models that can be deployed locally. We exclusively selected large instructed models due to the complex human-instruction-based tasks in our workflow. The final list of models includes: LLaMA-3.1 (70B), Qwen2 (72B), Command R+ (104B), Mistral-large-2407 (123B), and LLaMA3.1 (405B). The reasons for not selecting other models are outlined in Section L.
this section cite: []

Section: Metrics
In accordance with the real-world developer expectations discussed in Section D, we evaluate performance at the merge request (MR) level using four metrics, with their formal definitions provided in Section M:
❶
Key Bug Inclusion (KBI): Assesses the model's ability to recall critical issues that could lead to tangible losses. ❷ False Alarm Rate (FAR): Captures the proportion of irrelevant or erroneous comments, with two variants (F AR 1 for all MRs and F AR 2 for MRs where key bugs are recalled). ❸ Comprehensive Performance Index (CPI): Balances the completeness of key issue detection (KBI) and precision (100 -FAR), analogous to the F1-score. It is also computed in two variants (CP I 1 and CP I 2 ). ❹ Line Localization Success Rate (LSR): Measures the accuracy of line-level references by checking whether comments point to the correct code lines.
this section cite: []

Section: Baselines and Experimental Setups
Since our framework focuses on C++, we selected stateof-the-art baselines that support this language: CodeReviewer (Li et al., 2022b): A T5 model pre-trained for code review tasks and then fine-tuned. CCT5 (Lin et al., 2023): A T5 model pre-trained on CodeChangeNet, then fine-tuned. LLaMA-Reviewer (Lu et al., 2023): A large LLM finetuned for code review tasks based on the LLaMA. DIS-COREV (Ben Sghaier & Sahraoui, 2024): A T5 model enhanced via cross-task knowledge distillation for code review. The detailed experimental setups of our framework and baselines are presented in Section N.
this section cite: ['b22']

Section: Evaluation Results

this section cite: []

Section: RQ1. Comparison with Baselines
We evaluated the performance of our framework on the fault merge request dataset, comparing it with several baseline approaches. Our framework was tested with different large language model (LLM) engines. Our main experiments primarily utilized a homogeneous setup, employing the same LLM across all roles. This approach was chosen to isolate and clearly assess whether a single, powerful model could effectively address key challenges in code review. Recognizing the practical importance and potential benefits of diverse model deployments, we also conducted extended comparison experiments with heterogeneous LLM assignments for reviewer and validator roles. These experiments, detailed in Appendix O, show that strategic combinations, such as pairing a strong validator with a smaller reviewer, can achieve comparable or even superior performance while potentially optimizing resource usage.
For baselines, since they do not prioritize comments, we evaluated their comments based on whether they passed their respective "quality estimation" filters, which assess whether a code snippet requires a comment. The results are in Table 2.
The results indicate that our framework significantly outperforms the baselines by a factor of 10x across most key metrics, such as key bug inclusion (KBI) and comprehensive performance index (CPI). This marked improvement is likely due to our framework's end-to-end approach to code review automation, which addresses the key challenges of the task and introduces strategies specifically designed to tackle each challenge.
Among the LLM engines tested in our primary setup, LLaMA3.1-405B demonstrated the best overall performance, which aligns with the general scaling laws of language models where capability often increases with parameter count on complex tasks such as code review. However, our evaluations (detailed in Table 2) also included more compact LLMs. These results show that certain smaller models, particularly those with strong inherent reasoning capabilities, can still achieve competitive performance within our framework. This finding is particularly relevant given the industry trend towards increasing 'capacity density' in newer architectures, where smaller models are progressively narrowing the performance gap. While the largest models may provide peak effectiveness, these observations suggest that a range of LLMs can be effectively utilized, allowing for a balance between performance and computational resource demands, a point further explored in our heterogeneous model assignments (Appendix O).
this section cite: []

Section: Summary of RQ1.
Our framework surpasses baseline approaches significantly (up to 10x on KBI/CPI), thanks to its end-to-end design. LLaMA3.1-405B stands out among tested engines, highlighting the role of model capability.
Investigations into heterogeneous LLM combinations also suggest the potential for optimized deployments. (See Appendix T.1 for the extended conclusion.)
this section cite: []

Section: RQ2. Effectiveness of Code Slicing
We tested the four code slicing algorithms described in Section 3.2: Original Diff, Parent Function, Left Flow, and Full Flow. It is important to clarify that while our framework does not employ an explicit Retrieval-Augmented Generation (RAG) pipeline, our code slicing mechanism is designed with a RAG-aligned objective. Specifically, it serves a similar purpose to RAG by strategically retrieving and providing the LLM with only the most relevant contextual code 'slices' from the broader codebase. This process aims to focus the model on pertinent information, thereby enhancing its reasoning and effectiveness in the code review task. Our focus in this section is on KBI and CP I 1 , as these metrics indicate how input content affects the maximum recall capability of LLMs for code review.
The experiments were structured to evaluate the comments generated by the large language models under different conditions, including all comments, comments after applying a coarse filter, and top-k ranked comments (based on scores from Q3). We also tested multi-reviewer settings, where the meta-reviewer merges the comments, and validator settings, where validators further refine the comments. The average results are shown in Table 3, based on the LLaMA3.1-405B-AWQ-Int4 LLM engine. To provide further insight into the variability of these results, the minimum and maximum values for each reported metric across the three runs are detailed in Appendix R.
The results reveal that using only the diff or parent function is less effective, while more detailed slicing (Left Flow and Full Flow) improves performance, especially in key bug inclusion. Surprisingly, Left Flow performs better than Full Flow, likely due to the large language model's reduced capability when provided with longer contexts, which can cause distraction. This finding supports our assumption that providing targeted and relevant code context is critical for maximizing LLM performance in code review tasks, an observation consistent with the principles underpinning RAG systems where curated information significantly enhances model outputs.
During our analysis of the recalled merge requests (MRs), we found another interesting pattern. Although some slicing algorithms perform worse overall, each algorithm uniquely succeeds in specific cases. This means that each slicing strategy provides valuable context in certain situations. Figure 4 presents a Venn diagram showing the union and differences among the key bugs recalled by each slicing algorithm under the "All" and "+Meta Reviewer" settings. Notably, Left Flow and Full Flow recall most, with significant overlap, but almost each method also uniquely recalls some.
This phenomenon mirrors how human reviewers operate-expanding their focus to different levels of granularity, such as inspecting parent functions or understanding variable usage in different contexts. Some defects are easier to spot in one context, while others require a different view. Therefore, a combination of various slicing strategies might be a promising direction.
this section cite: []

Section: Summary of RQ2.
Left Flow and Full Flow significantly improve key bug inclusion and overall performance compared to simpler slicing. Left Flow often outperforms Full Flow, possibly because shorter context helps maintain focus. Notably, each slicing approach has exclusive successes, suggesting that combining them could further improve detection. (See Appendix T.2 for the extended conclusion.)
this section cite: []

Section: RQ3. Effectiveness of Multi-role System
To better understand the capabilities of our multi-role system, we conduct experiments on: ❶ Leveraging the non-determinism of large language models; ❷ The selfcorrection capability (validator); ❸ The chain-of-thought (CoT) prompting strategy.
this section cite: []

Section: NUMBER OF REVIEWERS
Previous research has shown that the non-determinism of large language models (LLMs) can impact results. Specifically, with a best-of-N sampling approach, smaller LLMs can sometimes match or surpass larger models. Since our framework includes a multi-reviewer scenario, where a meta-reviewer merges comments from multiple reviewers, we conduct experiments to assess whether increasing the number of reviewers improves performance.
The results in Table 4 show that increasing the number of reviewers from one to three improves KBI but also  Table 4. Impact of increasing the number of reviewers from one to three. The "+Meta Reviewer" setting represents the meta-reviewer merging the reviewers' comments, while the "+Validator" setting denotes the validator refining the comments after the meta-reviewer. All settings use Top-5 truncation of reviewer comments.
Processing Stage Reviewer Num KBI↑ FAR 1 ↓ CPI 1 ↑ FAR 2 ↓ CPI 2 ↑ Original Diff + Meta
Reviewer 1 8.89 86.15 10.83 69.17 13.80 3 13.33 96.74 5.24 75.56 17.25 + Validator 1 4.44 76.59 7.47 73.33 7.62 3 11.11 90.11 10.46 71.00 16.07 Parent Function + Meta Reviewer 1 15.56 80.37 17.36 73.81 19.52 3 20.00 92.41 11.01 73.15 22.92 + Validator 1 6.67 63.70 11.26 55.56 11.59 3 11.11 89.48 10.81 65.33 16.83 Left Flow + Meta Reviewer 1 26.67 83.26 20.57 70.56 27.99 3 31.11 87.81 17.51 67.98 31.56 + Validator 1 11.11 69.26 16.32 23.33 19.41 3 20.00 75.37 22.07 43.52 29.54 Full Flow + Meta Reviewer 1 22.22 78.04 22.09 71.17 25.10 3 31.11 89.41 15.80 73.10 28.86 + Validator 1 15.56 71.78 20.06 61.43 22.17 3 20.00 77.96 20.97 67.59 24.73
leads to higher F AR 1 and F AR 2 , which negatively affect CP I 1 and CP I 2 in the "+Meta Reviewer" setting. However, after introducing the validator, the performance for three reviewers significantly improves in terms of CP I 1 and CP I 2 . While more reviewers boost KBI, they also increase false alarms, making the validator essential to overall performance.
this section cite: []

Section: Summary of RQ3.1.
Increasing the number of reviewers lifts key bug inclusion but raises false alarms. A validator mitigates these alarms, implying a trade-off between coverage and precision. (See Appendix T.3 for extended conclusions.)
this section cite: []

Section: SELF-CORRECTION ABILITY OF LLMS
In our framework, the validator refines and validates generated comments to correct hallucinations. Table 5 shows that the validator lowers F AR 1 and F AR 2 but also reduces KBI, indicating a trade-off between precision and recall.
Our analysis suggests such erroneous rejections of valid comments by validators primarily stem from factors including context propagation from earlier pipeline stages, minor inaccuracies in comment positioning, occasional model input token limits, and inherent scoring variances.
this section cite: []

Section: Summary of RQ3.2. Self-correction (validator) reduces false alarms but can inadvertently discard critical bug-

this section cite: []

Section: EFFECTIVENESS OF CHAIN-OF-THOUGHT
We compared our specified CoT approach with free-form reasoning. Table 6 shows that CoT prompts often excel in complex slicing tasks (Left Flow, Full Flow), but in simpler tasks (Original Diff, Parent Function), free-form can be just as good or better.
this section cite: []

Section: Summary of RQ3.3.
CoT prompting is especially beneficial in complex contexts. For simpler code slices, the model may perform well without explicit CoT guidance. As more powerful reasoning models, such as GPT-O1 and DeepSeek-R1, emerge, the advantage of specified CoT over free-form reasoning may further diminish. (Appendix T.3)
this section cite: []

Section: RQ4. Effectiveness of Comment Filter Mechanism
The comment filter mechanism includes ❶ Coarse reviewer filter, ❷ Top-k truncation, ❸ Meta-reviewer filter, and ❹ Validator validation. Table 7 shows that in flow-based slic- ing (Left Flow, Full Flow), adding these filters sequentially decreases F AR 1 and improves CP I 1 . In simpler slicing (Original Diff, Parent Function), only the coarse filter proves particularly effective, likely due to limited context causing more hallucinations. A comprehensive sensitivity analysis of the Top-k truncation hyperparameter k-detailing its impact on single-reviewer paths with various k values (as presented in Table 7) and an extended analysis within our multi-reviewer framework-is provided in Appendix S.
this section cite: []

Section: Summary of RQ4.
Our comment filter significantly reduces false alarms and improves performance in more detailed slicing methods. In simpler slicing, the coarse filter stage is the most impactful step. (See Appendix T.4 for the extended conclusion.)
this section cite: []

Section: RQ5. Line Number Position
Line number localization is crucial for real-world applications. We tested three formats: No: No line position information is provided; Relative: Code is provided with a separate list containing relative line positions; and Inline: Position information is integrated directly into the code using the format in Table 1.
Table 8 shows that providing line number information (especially inline) significantly improves performance and localization success rate (LSR).
this section cite: []

Section: Summary of RQ5.
Embedding line numbers inline yields the highest performance and LSR, likely because it helps the model anchor comments to specific lines accurately. (See Appendix T.5 for the extended conclusion.)  (Wang et al., 2024) and automating commit message generation (Tao et al., 2024), underscoring the expanding utility of large models in diverse software engineering contexts.
Despite these advances, previous works have oversimplified the code review process by treating it as a set of snippetlevel code-comment pairs. These approaches typically split merge requests into independent snippets and framed the task as a one-to-one neural machine translation (NMT) problem, converting code into natural language. While innova-tive, this approach provides a limited and idealized view of code review, often evaluated with text similarity metrics, such as BLEU or ROUGE, which do not fully capture the expectations of real-world developers for finding defects.
In practice, code review is more complex, evaluated at the level of entire merge requests of repository codebases rather than individual code-comment pairs. The focus on text similarity fails to consider the broader context, including how comments address the full scope of changes in a MR. Although contributing valuable insights, these studies fall short of replicating the holistic, real-world workflow.
this section cite: ['b50', 'b44']

Section: Conclusion
Motivated by the limitations of prior research that oversimplified code review automation and fell short of practical applications, we explored the complete automation pipeline within a real-world company. We identified and addressed key challenges such as capturing relevant code context, improving key bug inclusion (KBI), reducing false alarm rates (FAR), and integrating human-centric workflows. Our approach introduces four code slicing algorithms, a multi-role LLM framework, a comment filtering mechanism, and a prompt format with inline line number localization. Evaluations on real-world data demonstrated that we significantly outperforms existing methods, achieving up to a 10x improvement in the comprehensive performance index (CPI) over previous baselines.
Key insights include: ❶ Flow-based slicing (Left Flow and Full Flow) provided better context and outperformed simpler methods. ❷ Increasing the number of reviewers improved KBI but required validation to manage false alarms effectively. ❸ The validator role reduced hallucinations but slightly lowered KBI, highlighting a trade-off between precision and recall. ❹ Chain-of-thought guidance proved more valuable in complex slicing scenarios. ❺ Inline line number localization enhanced both comment accuracy and localization success rates.
Looking ahead, four key areas for future research are: ❶ Enhancing code slicing algorithms to capture more relevant context, potentially combining different slicing levels. ❷ Refining LLM interactions and enhancing engine LLM capability to improve key bug recall. ❸ Further optimizing the filtering mechanism, including the investigation of adaptive or learned thresholds, to reduce nitpicks and hallucinations more effectively. ❹ Streamlining pipeline to make automation more accessible.
Limitation. We discuss limitations in Section V.
this section cite: []

Section: References
Ref_id:b0 Title: Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Software inspections: an effective verification process Year: (1989)
Ref_id:b2 Title: How to refactor this code? an exploratory study on developer-chatgpt refactoring conversations Year: (2024)
Ref_id:b3 Title: Expectations, outcomes, and challenges of modern code review Year: (2013)
Ref_id:b4 Title: Improving the learning of code review successive tasks with cross-task knowledge distillation Year: (2024)
Ref_id:b5 Title: Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors Year: ()
Ref_id:b6 Title: Chatunitest: A framework for llm-based test generation Year: (2024)
Ref_id:b7 Title: Deepwukong: Statically detecting software vulnerabilities using deep graph neural network Year: (2021)
Ref_id:b8 Title: Design and code inspections to reduce errors in program development Year: ()
Ref_id:b9 Title: Aider -ai pair programming in your terminal Year: (2024)
Ref_id:b10 Title: Large language model based multi-agents: A survey of progress and challenges Year: (2024)
Ref_id:b11 Title: Intelligent code reviews using deep learning Year: (2018)
Ref_id:b12 Title: Commentfinder: a simpler, faster, more accurate code review comments recommendation Year: (2022)
Ref_id:b13 Title: Ruler: What's the real context size of your long-context language models? Year: (2024)
Ref_id:b14 Title: Robustkv: Defending large language models against jailbreak attacks via kv eviction Year: (2025-04)
Ref_id:b15 Title: Code review quality: How developers see it Year: (2016)
Ref_id:b16 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b17 Title: Auger: automatically generating review comments with pre-training models Year: (2022)
Ref_id:b18 Title: Vuldeepecker: A deep learning-based system for vulnerability detection Year: (2018)
Ref_id:b19 Title: Sysevr: A framework for using deep learning to detect software vulnerabilities Year: (2021)
Ref_id:b20 Title: Automating code review activities by large-scale pretraining Year: (2022)
Ref_id:b21 Title: Graphrag under fire Year: (2025)
Ref_id:b22 Title: Cct5: A code-change-oriented pre-trained model Year: (2023)
Ref_id:b23 Title: Rouge: A package for automatic evaluation of summaries Year: (2004)
Ref_id:b24 Title: Activation-aware weight quantization for on-device llm compression and acceleration Year: (2024)
Ref_id:b25 Title: Lost in the middle: How language models use long contexts Year: (2024)
Ref_id:b26 Title: Llama-reviewer: Advancing code review automation with large language models through parameter-efficient fine-tuning Year: (2023)
Ref_id:b27 Title: Exploring the impact of code review factors on the code review comment generation Year: (2024)
Ref_id:b28 Title: Revisiting the evaluation of code review comment generation Year: (2025)
Ref_id:b29 Title: Knowledge enhanced pre-trained language model for log understanding Year: (2024)
Ref_id:b30 Title: An exploratory study on using large language models for log parsing Year: (2024)
Ref_id:b31 Title: Cppcheck -static analysis of c/c++ code Year: (2024)
Ref_id:b32 Title: Llm critics help catch llm bugs Year: (2024)
Ref_id:b33 Title: Flask -the python micro framework for building web applications Year: (2024)
Ref_id:b34 Title: Bleu: a method for automatic evaluation of machine translation Year: (2002)
Ref_id:b35 Title: A comprehensive review on background, applications, key challenges, bias, ethics, limitations and future scope Year: (2023)
Ref_id:b36 Title: Convergent contemporary software peer review practices Year: (2013)
Ref_id:b37 Title: Open source software peer review practices: a case study of the apache server Year: (2008)
Ref_id:b38 Title: Peer review on open-source software projects: Parameters, statistical models, and theory Year: (2014)
Ref_id:b39 Title: Modern code review: a case study at google Year: (2018)
Ref_id:b40 Title: An empirical evaluation of using large language models for automated unit test generation Year: (2023)
Ref_id:b41 Title: Using nudges to accelerate code reviews at scale Year: (2022)
Ref_id:b42 Title: Automating review recommendation for code changes Year: (2020)
Ref_id:b43 Title: Software engineering 9th edition Year: (2011)
Ref_id:b44 Title: Knowledge-aware denoising learning for commit message generation Year: (2023)
Ref_id:b45 Title: Towards automating code review activities Year: (2021)
Ref_id:b46 Title: Using pre-trained models to boost code review automation Year: (2022)
Ref_id:b47 Title: Deep learning-based code reviews: A paradigm shift or a double Year: (2024)
Ref_id:b48 Title: Attention is all you need Year: (2017)
Ref_id:b49 Title: Does every inspection need a meeting? Year: (1993)
Ref_id:b50 Title: Unity is strength: Collaborative llm-based agents for code reviewer recommendation Year: (2024)
Ref_id:b51 Title: Mining the modern code review repositories: A dataset of people, process and product Year: (2016)
Ref_id:b52 Title: Pscvfinder: a prompt-tuning based framework for smart contract vulnerability detection Year: (2023)
Ref_id:b53 Title: Smartllama: Two-stage post-training of large language models for smart contract vulnerability detection and explanation Year: (2024)
Ref_id:b54 Title: Licensed under Creative Commons Attribution 4.0 International License. RQ1: How does the overall performance of our framework compare with previous works? This question evaluates our framework's overall performance against existing baselines, focusing on metrics such as KBI, FAR, and CPI (defined in Section 4.3). Since our framework decouples from base LLMs, we utilize various open-source large language model Original Diff Year: (2025-07)
