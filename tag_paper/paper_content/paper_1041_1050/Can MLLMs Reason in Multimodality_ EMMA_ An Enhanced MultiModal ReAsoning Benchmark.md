Title: Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark
Abstract: The ability to organically reason over and with both text and images is a pillar of human intelligence, yet the ability of Multimodal Large Language Models (MLLMs) to perform such multimodal reasoning remains under-explored. Existing benchmarks often emphasize text-dominant reasoning or rely on shallow visual cues, failing to adequately assess integrated visual and textual reasoning. We introduce EMMA (Enhanced MultiModal reAsoning), a benchmark targeting organic multimodal reasoning across mathematics, physics, chemistry, and coding. EMMA tasks demand advanced cross-modal reasoning that cannot be addressed by reasoning independently in each modality, offering an enhanced test suite for MLLMs' reasoning capabilities. Our evaluation of state-of-the-art MLLMs on EMMA reveals significant limitations in handling complex multimodal and multi-step reasoning tasks, even with advanced techniques like Chain-of-Thought prompting and test-time compute scaling underperforming. These findings underscore the need for improved multimodal architectures and training paradigms to close the gap between human and model reasoning in multimodality. The project homepage can be accessed at https:  //emma-benchmark.github.io/.

Section: Introduction
Multimodal reasoning is fundamental to human intelligence. For example, interior designers combine textual descriptions with mental imagery to optimize room layouts. Text-based reasoning allows us to analyze abstract concepts, while Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
Multimodal reasoning question in EMMA Human reasoning Model reasoning (GPT-4o)
The direction of the electric force due to -2Q and +3Q can be visualized with a quick free-body sketch of the electric forces: Question: Three point charges, of charge +Q, -2Q , and +3Q , are placed equidistant as shown. Which vector best describes the net direction of the electric force acting on the +Q charge?
this section cite: []

Section: attract repel
Combining the forces: ∵ |+3Q|>|-2Q| ∴ repel > attract
Error: While the model understands that like charges repel, it incorrectly identifies the force from +3Q to +Q as downward-right⬊ , when it should be upward-left ⬉.
this section cite: []

Section: ……

this section cite: []

Section: Figure 1: A sample multimodal reasoning question in EMMA.
Humans engage in graphical reasoning: guided by the principles of electric force, they draw force vectors with appropriate directions and visually compute their sum. While GPT-4o understands that like charges repel, it mistakes the direction of the repulsive force, highlighting its limitations in multimodal reasoning.
visual reasoning enables us to draw insights from complex visual information. Combining these skills provides a robust framework for solving technical and creative challenges.
Recent advancements in Large Language Models (LLMs) have significantly enhanced their reasoning abilities (Ope-nAI, b;Qwen, 2024;Zhao et al., 2024;DeepSeek, 2024), enabling strong performance on tasks such as formal logic reasoning (Hendrycks et al., 2020b), graduate-level academic question answering (Rein et al., 2023), and competitive programming (codeforces; Bai et al., 2023;Hendrycks et al., 2020a). Despite these successes, these models primarily focus on text-only reasoning, leaving an open question: can Multimodal LLMs (MLLMs) effectively reason across both language and visual inputs?
A major bottleneck in addressing this question is the lack of appropriate benchmarks. Existing multimodal benchmarks
this section cite: ['b51', 'b17', 'b54', 'b7']

Section: Graph Reasoning
Q: The \%yield of ammonia as a function of time in the reaction at is given below. If this reaction is conducted at with , the \%yield of ammonia as a function of time is represented by: and triple bonds but excluding those involving hydrogen. Note: Disregard arrows. Consider all components present in the transition-state structure shown in the image.  largely test surface-level visual understanding (Antol et al., 2015;Goyal et al., 2017;Yu et al., 2023;Akter et al., 2024) or textual knowledge (Yue et al., 2024b;a;Marino et al., 2019) recall with multimodal inputs. While some benchmarks (Lu et al., 2024b;Wang et al., 2024a) include math questions with images, Zhang et al. (2024a) have shown that many of these tasks reduce to language-only reasoning, as the visual content is often fully described in text.
To address this gap, we introduce EMMA: an Enhanced MultiModal reAsoning benchmark, specifically designed to evaluate the ability to solve problems that require both visual-and language-based problem-solving. EMMA features questions that are difficult to solve by relying solely on text-based reasoning or a single visual pass. Instead, solving these problems necessitates a back-and-forth process between interpreting visual inputs and applying multimodal reasoning steps, where visual aids are often integral or more efficient for arriving at the solution. For instance, Figure 1 illustrates a sample physics problem that asks for the direction of the net electric force. While GPT-4o understands that like charges repel, it mistakes the direction of the repulsive force, highlighting its limitations in multimodal reasoning.
Unlike recent benchmarks (Ramakrishnan et al., 2024;Chollet, 2019), which focus on spatial cognition, or visual puzzles that can be perfectly represented in text, EMMA introduces domain-specific challenges where reasoning can often be strengthened by visual aids. These include tasks like 3D spatial transformations, chemical structure recognition, multi-step physical simulations, and program output visualization (Figure 2). EMMA consists of 992 multimodal reasoning questions gathered from existing benchmarks through a rigorous filtering pipeline, and 1,796 newly constructed questions created manually in collaboration with domain experts. Our evaluation of ten state-of-the-art (SoTA) MLLMs on EMMA reveals three key findings:
MLLMs struggle with multimodal reasoning: All models perform suboptimally on EMMA, regardless of the usage of Chain-of-Thought (CoT) prompting (Wei et al., 2022). On the balanced subset of EMMA, the best-performing model, Gemini 2.0 Flash Thinking, scores only 48.00%, which is 10.75% higher than the best non-reasoning MLLM, Qwen2-VL (Wang et al., 2024b), but still trails human experts by 29.75%. These results suggest a limitation of current MLLMs in performing in-depth multimodal reasoning.
Test-time compute scaling methods with textual CoTs are insufficient: We explore test-time compute scaling of SoTA MLLMs with different methods (e.g., majority voting, best-of-N, and tournament) up to 16 times, yet they still fail to address the multimodal reasoning challenges in EMMA. Simply increasing the number of candidate responses with textual CoTs does little to compensate for the models' inability to produce valid visual reasoning steps, particularly for tasks requiring fine-grained spatial understanding or multistep reasoning. In addition, current MLLMs and specialized reward models struggle with complex multimodal reasoning themselves, which can make their reward signals unreliable and limit the utility of test-time compute scaling.
Visual reasoning is the bottleneck: Through error analysis, we find that SoTA MLLMs frequently struggle with tasks requiring precise spatial simulations, multi-hop visual reasoning, and integration of visual and textual information. These shortcomings are particularly pronounced in problems where visual aids offer a simpler path to the solution. Further, textual CoT negatively impacts model performance on visual-reasoning-heavy tasks, highlighting the need for new paradigms to improve visual reasoning.
These insights suggest that the performance gap between text-based and multimodal reasoning arises from MLLMs' limited ability to perform fine-grained visual reasoning. EMMA highlights the need for new architectures and training paradigms that can better integrate and reason over diverse modalities, enabling models to leverage both visual and linguistic information more effectively.
this section cite: ['b6', 'b19', 'b76', 'b3', 'b44', 'b53', 'b12', 'b68']

Section: Related Work
Multimodal Large Language Models Recent years have witnessed rapid progress in MLLM development. Building upon early techniques in vision-language modeling (Tan & Bansal, 2019;Lu et al., 2019;Chen et al., 2020;Radford et al., 2021;Li et al., 2020;Zhang et al., 2021;Yu et al., 2022), modern MLLMs (Li et al., 2024a;Lu et al., 2024a;Team, c;Liu et al., 2024a;Yang et al., 2024a;Achiam et al., 2023;Team, a;Li et al., 2023) leverage the success of LLMs and various visual instruction tuning techniques (Liu et al., 2024b;a;Zhu et al., 2023), achieving impressive performance in many multimodal tasks.
LLM and MLLM Reasoning State-of-the-art models (Ope-nAI, b;Qwen, 2024;Zhao et al., 2024;DeepSeek, 2024) now achieve strong performance on tasks such as formal logic reasoning (Hendrycks et al., 2020b), graduate-level academic question answering (Rein et al., 2023), and competitive programming (codeforces). These advancements in text-based reasoning have spurred growing interest in multimodal reasoning, exemplified by visual CoT models (Shao et al., 2024;Zhang et al., 2023) and visual CoT prompting techniques (Zhou et al., 2024). Although visual CoT prompting techniques have shown promise, their focus is primarily on enhancing perception through methods like cropping images to simulate attention. Hence, these approaches offer limited support for tasks that demand more advanced visual reasoning skills, such as visual manipulation or imagination.
Multimodal Reasoning Benchmarks Most existing reasoning benchmarks are text-based (e.g., (Cobbe et al., 2021;Hendrycks et al., 2021;Srivastava et al., 2022;Jin et al., 2023;Suzgun et al., 2022)), but the rising demand for multimodal evaluation has led to the development of benchmarks across diverse domains (e.g., (Lu et al., 2024b;Wang et al., 2024a;Li et al., 2024b;Yang et al., 2024c;Ying et al., 2024;Chen et al., 2024a;Li et al., 2024c;Cheng et al., 2024)). Recent efforts focus on spatial and relational reasoning (Akter et al., 2024;Ramakrishnan et al., 2024) and college-level reasoning requiring domain knowledge (Yue et al., 2024a). However, many multimodal benchmarks contain redundant text-image information, allowing models to bypass visual reasoning (Zhang et al., 2024a). To address this, MMMU-Pro (Yue et al., 2024b) incorporates a filtering pipeline to enhance multimodal evaluation. In this work, we further refine such approaches by curating a benchmark that focuses explicitly on tasks requiring strong visual reasoning. Unlike existing benchmarks, our test suite emphasizes multimodal reasoning challenges that are difficult to solve with text-based reasoning and a single visual pass.  1, and its composition is presented in Figure 3.
To provide fine-grained insights into how MLLMs might fail in multimodal reasoning, we assign labels to each problem in our benchmark. These labels are either created by domain experts or assigned by GPT-4o and subsequently verified by experts. As shown in Figure 2, questions in EMMA assess a wide array of multimodal reasoning skills. For example, the pattern inference problem in math challenges models to identify and generalize visual patterns; the visual decomposition simulation problem in physics requires graphically decomposing forces to determine resultant effects; the reaction simulation problem in chemistry demands precise interpretation and simulation of electron movement; the 3D visualization problem in coding 1 evaluates spatial imagination by requiring models to associate function calls with their corresponding 3D representations.
this section cite: ['b61', 'b41', 'b9', 'b52', 'b34', 'b80', 'b75', 'b2', 'b32', 'b84', 'b51', 'b17', 'b54', 'b55', 'b81', 'b83', 'b13', 'b25', 'b59', 'b27', 'b60', 'b74', 'b11', 'b3', 'b53']

Section: Data Curation
As discussed in Section 2, most existing multimodal reasoning benchmarks likely contain many problems that primarily measure text-based reasoning. To address this, we employ a two-step approach to constructing EMMA (Figure 4). First, we source problems from existing multimodal reasoning benchmarks and apply rigorous filtering to exclude those solvable through text-based reasoning and a single visual pass. Next, we categorize the remaining problems for each subject into fine-grained multimodal reasoning skill taxonomies and manually collect more samples aligned with these taxonomies to expand our dataset. 1 Different from the other subjects in EMMA, coding questions can be assigned more than one category since our visualizations tend to employ multiple advanced techniques.
Graph Reasoning 0.32% Visual Decomposition 2% Field Simulation 1% Multi-Hop Rea 1% Graph Rea 1% Path Tracing 0.47% Filtering Mechanisms To filter for questions that require multimodal reasoning, Yue et al. (2024b) provide only the text from multimodal reasoning questions to LLMs and discard questions that can be correctly answered this way. Nonetheless, some of the remaining questions may still not truly measure visual reasoning, as a single pass of visual perception and language understanding may suffice to answer them. We extend this one step further (illustrated in Figure 4): we first caption the images in multimodal reasoning questions using GPT-4o and then pass both the original text and our generated captions to MLLMs, filtering out questions that can be answered under this condition. For each candidate question, we query Llama-3-70B-Instruct (Dubey et al., 2024), GPT-4o, and Qwen2-72B-Instruct (Yang et al., 2024a) ten times; if any model answers a question correctly at least five times, we discard it, following the 5/10 threshold in MMMU (Yue et al., 2024a). This more stringent filtering ensures that the remaining questions require models to engage deeply with visual information. We introduce the data collection process for each project in detail below.
this section cite: ['b18']

Section: Math
We first apply the filtering pipeline to Math-Vision (Wang et al., 2024a) and MathVista (Lu et al., 2024b), and then manually inspect the remaining set and craft a taxonomy consisting of five categories with a strong focus on multimodal reasoning, including 3D Simulation, 2D Transformation, Path Tracing, Multi-hop Object Counting, and Pattern Inference. Next, we use GPT-4o to categorize all questions based on this taxonomy, followed by a manual verification. In addition, we supplement our benchmark with additional pattern inference questions from RAVEN (Zhang et al., 2019), which inherently require multi-hop visual reasoning. This process results in a total of 892 math questions.
Physics We apply the filtering pipeline to multimodal physics problems in OlympiadBench (He et al., 2024), EXAMS-V (Das et al., 2024), and MMMU (Yue et al., 2024a), which yields only 80 problems. In addition, we manually collect more problems online from Learn AP Physics (Physics) and Khan Academy (Academy) and filter them, resulting in 76 more new problems. Through manual labeling, we verify that these problems span a wide range of topics, including 3D Field Simulation, Graph Reasoning, and Path Tracing. We note that despite our best efforts, multimodal physics problems meeting our criteria are difficult to source and construct.
this section cite: ['b79', 'b22', 'b14']

Section: Chemistry
After filtering the chemistry portion of MMMU (Yue et al., 2024a) and EXAMS-V (Das et al., 2024), we are only left with 42 questions. Since these questions mostly involve reasoning about molecular formulas, we construct more problems on organic chemistry. We analyze the chemical properties of molecules in SMiCRM (Leung et al., 2024) with RDKit (lan, 2013), a computational chemistry toolkit, and develop 904 novel questions over chemical structure recognition and bond counting. In addition, we draw from the collection of chemical reactions in Li (2009) and collaborate with PhD students in chemistry to annotate reaction outcomes, contributing another 210 questions on reaction simulation.
Coding We design four coding tasks to assess MLLMs in real-world visualization creation scenarios. For instance, to evaluate the ability to reproduce a visualization, we construct "Vis Choose Code" questions where models select the code that generates a target chart. Since previous visualization design benchmarks all use MLLM judges, we do not source from them, but manually construct all coding questions from scratch. We first identify "seed visualizations" using advanced visualization techniques from CharXiv (Wang et al., 2024c), the matplotlib example gallery (Team, b) following (Wu et al., 2024), and our prior experience. Next, we generate four variations for each seed visualization to form a "set" by introducing design variations (e.g., changes in spine configuration, line style, and axis scaling) either manually or through prompting MLLMs, with post-hoc manual verification. We provide these design variations as labels for each problem. Finally, we construct different types of questions using these visualization sets. Our curation results in 564 multiple-choice coding questions.
this section cite: ['b14', 'b29', 'b31', 'b70']

Section: Comparison with Existing Benchmarks
Our enhanced data filtering pipeline ensures that EMMA focuses on questions requiring in-depth multimodal reasoning, i.e., those that cannot be solved solely using text-based reasoning or a single visual pass. While MMMU-Pro (Yue et al., 2024b) removes questions solvable through their text portion alone, it may still retain problems for which visual reasoning is inessential. In contrast, EMMA applies a stricter filtering criterion, discarding questions solvable with text and image captions. For instance, the left example in Figure 4 (adapted from MathVista) asks whether a depicted function is even or odd. Although unsolvable without the image, the problem can be shortcut by extracting the function's text expression embedded in the image. In this case, the role of vision is more aligned with visual perception than with visual reasoning. By eliminating such problems, which MMMU-Pro's approach would retain, EMMA better evaluates the multimodal reasoning capabilities of models.
We also contribute 1,796 novel multimodal reasoning problems across physics, chemistry, and coding. After filtering physics and chemistry problems from all relevant benchmarks to our knowledge (e.g., (He et al., 2024;Das et al., 2024;Yue et al., 2024a)), only 100 remain. We expand this to 1,332 in EMMA by manually sourcing additional data and hiring domain experts. For coding, EMMA is the first benchmark to systematically evaluate data visualization skills using a multiple-choice format, enabling a standardized assessment and obviating the need for MLLMs as judges. Moreover, through meticulous manual labeling or verification, we provide fine-grained labels for each question (Figure 2), categorizing them based on the specific skills they assess. These labels enable a detailed analysis of MLLM performance, as we demonstrate in Section 5.
this section cite: ['b22', 'b14']

Section: Experiments

this section cite: []

Section: Evaluation Settings
Data Split To create a more balanced subset of EMMA, we randomly sample 400 questions (100 per subject) from the benchmark, hereafter referred to as EMMA-mini. Within each subject, we aim for equal representation across categories to the extent possible.
Human Performance To estimate expert-level performance on EMMA-mini, we hire two human experts per subject and report their average score. This score serves as a baseline  (Wang et al., 2024b), QVQ-72B-Preview (Alibaba), LLaVA-Onevision (72B) (Li et al., 2024a), InternVL2 (76B) (Team, c), and InternVL2.5 (78B) (Chen et al., 2024b)) and five proprietary ones (GPT-4o (OpenAI, a), Claude 3.5 Sonnet (Anthropic), Gemini 2.0 Flash (Deepmind, a), Gemini 2.0 Flash Thinking (Deepmind, b), and o1 (OpenAI, b)). Due to rate limits, we report o1 and QVQ performance on EMMA-mini only. All other models are evaluated on the entire benchmark.
Prompting Strategies For all models except o1, QVQ, and Gemini 2.0 Flash Thinking, we test two prompting strategies: (1) Direct prompting, which instructs models to output the answers without reasoning steps; and (2) Chainof-Thought (CoT) prompting (Wei et al., 2022), where we prompt models to "think step-by-step".
this section cite: ['b68']

Section: Main Results
Are MLLMs Multimodal Reasoners? Table 2 demonstrates that all models perform suboptimally across the subjects in EMMA. On EMMA-mini, the best-performing model, Gemini 2.0 Flash Thinking-0121, achieves an accuracy of 48.00%, trailing human experts by 29.75%. At the lower end, LLaVA-OneVision-72B scores only 25.25%, barely surpassing random choice by 2.5%. Drilling down into subjects, the best models show the smallest gap with human performance on physics, with Gemini 2.0 Flash Thinking scoring 1.5% lower than human experts. This smaller gap may reflect the inherent difficulty of physics problems, leading human experts to achieve a score of 64.5%. For other subjects, however, the best-performing models lag significantly behind human experts, with gaps of 34%, 39%, and 32.5% in math, chemistry, and coding, respectively. These results underscore the limitations of current MLLMs in addressing complex multimodal reasoning tasks.
On the full EMMA benchmark, closed-source models generally outperform open-source ones, particularly with CoT prompting. Across all subjects, Qwen2-VL-72B-Instruct is the only open-source model to place in the top two for any subject. Gemini 2.0 Flash Thinking-0121 scores best on every subject (note that o1 is not evaluated on the full set due to rate limits). In particular, it performs exceptionally well in physics, leading by 16.67% over the second-best model (excluding Gemini 2.0 Flash Thinking-1219), GPT-4o. Coding is another area where it excels, leading the second-best model (excluding Gemini 2.0 Flash Thinking-1219), Gemini 2.0 Flash, by 6.03%. Notably, on EMMA-mini, o1 outperforms Gemini 2.0 Flash Thinking-0121 by 5% in coding, suggesting even stronger coding capabilities. In sum, these results highlight the advantages of optimizing models for reasoning by training them to generate thought processes over traditional MLLM paradigms.
this section cite: []

Section: Does CoT help?
We observe divergent tendencies in the effectiveness of CoT prompting across closed-and opensource models. We exclude o1, QVQ, and Gemini 2.0 Flash Thinking from this analysis, as they inherently generate CoT as part of their responses. Under direct prompting, accuracies achieved by the best open-source models are well within 2% of Claude 3.5 Sonnet (the best closed-source model). However, the gap widens significantly under CoT prompting, with the best open-source model underperforming by al- to fully leverage the potential of language to assist in multimodal reasoning tasks where language could be helpful. We elaborate on this hypothesis in detail in Section 5.
this section cite: []

Section: Results on Test-Time Compute Scaling
We test three test-time compute scaling methods (Snell et al., 2024;Yang et al., 2024b; OpenAI, b) on EMMA-mini: majority voting, Best-of-N selection, and Tournament selection. Both Best-of-N and Tournament selection require a reward model to select the best response among multiple candidates. We use CoT prompting to generate the candidate responses, so that the reward model has enough context to score the responses. For each test-time scaling method, we experiment with N = 1, 2, 4, 8, and 16, as long as the context length of the reward model allows.
Majority Voting: Majority voting selects the most frequent response among batches of N candidate responses, breaking ties by randomly choosing from the most frequent answers.
Best-of-N: Best-of-N selection (Cobbe et al., 2021;Lightman et al., 2023) chooses the highest-scoring response (Gu et al., 2024) according to a reward model. We explore two configurations: using the base model itself or a stronger reasoning model (e.g., Gemini 2.0 Flash Thinking) as the reward model.
Tournament Selection: In Tournament Selection (Son et al., 2024;OpenAI, c), responses compete in matches, with winners advancing through rounds until the final selection. We use the best-performing reward model identified in the Bestof-N experiments, which is Gemini 2.0 Flash Thinking.
this section cite: ['b57', 'b13', 'b36', 'b20', 'b58']

Section: Does test-time compute scaling help?
Table 3 presents results of test-time compute scaling methods on top of GPT-4o, Gemini 2.0 Flash, and Gemini 2.0 Flash Thinking-1219.
Overall, test-time compute scaling improves model performance, but it fails to close the gap to human expert performance. The highest accuracy improvements are 5.25% for GPT-4o, 4.5% for Gemini 2.0 Flash, and 7.5% for Gemini 2.0 Flash Thinking. Notably, without test-time scaling (N=1), Gemini 2.0 Flash Thinking-1219's accuracy is 2.25% lower than that of o1, but it overtakes o1 by 5% with majority voting at N=16. Nonetheless, its best performance still lags human performance by 27%.
We also observe distinct patterns in test-time compute scaling performance across different models. While scaling beyond N=8 for GPT-4o and Gemini 2.0 Flash leads to performance degradation, Gemini 2.0 Flash Thinking continues to benefit incrementally from additional test-time compute, at least up to N=16. In fact, stronger base models also achieve higher Pass@N accuracy: Gemini 2.0 Flash Thinking's Pass@N consistently surpasses those of the other two models by around 7%, suggesting that a stronger base reasoner is more likely to cover the correct response when given multiple attempts. In sum, these results suggest that using a stronger model as the base model raises the upper bound for test-time scaling.
Comparing scaling strategies for each model, we find that GPT-4o and Gemini 2.0 Flash achieve their greatest improvements when Gemini 2.0 Flash Thinking is used as the reward model. Additionally, tournament selection consistently outperforms Best-of-N (BoN) selection. These results suggest that using a stronger model as the reward model enables weaker models to achieve better results, particularly when the reward model can make fine-grained decisions involving a couple candidate responses each time. This is intuitive, as evaluating responses also requires reasoning.
30.19% 52.83% 9.43% 7.55% Perceptual Error Visual Reasoning Error Text Reasoning Error Lack of Knowledge Error Type On the other hand, we find that self-reward modeling often underperforms. Even using Gemini 2.0 Flash Thinking for self-reward modeling yields performance consistently below that of majority voting. We conjecture that self-reward modeling may be less effective because the model's evaluation criteria may be disrupted by its own generation patterns, making it less sensitive to differences in the reasoning of the generated responses than an independent reward model.
this section cite: []

Section: Error Analysis
Error Distribution We present an analysis of the errors made by o1 on the math and coding portions of EMMA-mini.
In total, o1 incorrectly answers 59 math questions and 47 coding questions. Figure 5 categorizes these errors into four types. Perceptual errors, such as misinterpreting visual information, account for 30.19% of all errors. Lack of knowledge errors, including mistakes related to API usage, contribute 7.55%. Visual reasoning errors, such as failures to simulate 3D processes, constitute the largest category at 52.83%. Finally, textual reasoning errors, including calculation mistakes or logical missteps, represent 9.43%. The predominance of visual reasoning errors underscores the limitations of current models in addressing complex visual reasoning tasks. Figure 6 illustrates a representative case: while o1 correctly identifies that the problem calls for the application of the Right-Hand Rule, it fails to simulate where the thumb would point to when the right-hand fingers are curled in the direction of the current. We provide more error cases in Appendix D.
this section cite: []

Section: The Effects of Textual CoT
Table 2 reveals a notable discrepancy: while CoT prompting improves reasoning in closed-source MLLMs, it tends to hurt performance in opensource models. Although the cause remains unclear without access to training data or methods, we analyze error rate per skill category to propose a conjecture. Figure 7 compares accuracy differences between CoT and Direct prompting for six models on two math tasks: 2D Transformation and Multi-Hop Object Counting. On 2D Transformation, all but one model shows reduced performance with CoT; on Multi-Hop Object Counting, while CoT generally helps closed-source Question: A long bar magnet is placed above a current loop oriented as shown. In which direction will the North pole of the bar magnet feel a force due to the current loop? a. +x b. -x c. +y d. -y e. The bar magnet will feel no force due to the current loop.
this section cite: []

Section: Human Reasoning (Ground Truth):
The current loop will produce a magnetic field B according to the Right-Hand Rule as shown below.
Then, as shown on the left, B points out from the middle of the current loop I, directed in the +x direction at the location where the bar magnet is o1: A current loop behaves like a small bar magnet whose magnetic-dipole moment points according to the Right-Hand Rule.
• From the figure, the current goes around the loop in such a way that if you curl your right-hand fingers in the direction of the current, your thumb points up
this section cite: []

Section: Physics: 3D Field Simulation
Error Analysis:
located. This magnetic field, pointing to the right, will have the effect of repelling the North pole of the bar magnet, forcing the bar magnet to the right, in the +x direction. The correct answer is a.
(the + y direction). Consequently, the loop acts like a magnet whose "north" side is on top (at + y). …… Hence the force on the north pole is c. +y .
The error occurred because of limited visual spatial imagination, leading to the wrong conclusion about the direction of the thumb when applying the Right-Hand Rule. Additionally, the 3D spatial nature of the magnetic field is not properly simulated, leading to a faulty reasoning of the "north" pole's position. models, it makes all but one open-source model perform worse. In fact, we notice that CoT prompting introduces more hallucinations for open-source models.
2D Transformation questions primarily test for visual simulation and spatial imagination, which are difficult to verbalize. The question in Figure 8, for example, calls for spatial imagination beyond the power of language. In contrast, Multi-Hop Object Counting can leverage language to describe the relative positions of objects. Hence, we conjecture that visual-centric tasks, such as 2D Transformation, are poorly suited for textual CoT, whereas tasks that benefit from text-based reasoning, such as Multi-Hop Object Counting, allow models to achieve greater performance gains with textual CoT, as evidenced by the closed-source models.
this section cite: []

Section: Conclusion
We contribute EMMA, an Enhanced MultiModal reAsoning benchmark. EMMA features multimodal questions that cannot be solved by independently reasoning within each modality. Evaluation of ten MLLMs reveals a substantial performance gap compared to human experts on EMMA, with techniques such as Chain-of-Thought prompting and test-time compute scaling offering only marginal gains. EMMA highlights the need for new architectures and training paradigms that can better integrate and rea- Question: Rebecca folds a square piece of paper twice. Then she cuts off one corner as you can see in the diagram. Then she unfolds the paper. What could the paper look like now? Math: 2D Transformation GPT-4o with CoT : To solve this problem, we need to carefully analyze the folding and cutting process and determine the resulting pattern when the paper is unfolded. ✓ Step 1: Understand the folding …… ✓ Step 2: Understand the cutting process 1. Rebecca cuts off one corner of the folded paper. ✓ • Since the paper is folded into a smaller square, this \"corner\" is actually a corner of the folded square, which corresponds to multiple corners of the original square.✘ …… Final Answer:\boxed{E} Human Reasoning (Ground Truth): Direct GPT-4o : B. Error Analysis: After folding into 1/4, the "corners" of the folded paper are not the original corners. The solution fails to properly simulate the unfolding process. son over diverse modalities. Like any benchmark, EMMA has its limitations, which can be improved in future works. Future iterations could also enrich the currently underrepresented physics section or expand the chemistry section to incorporate a broader range of chemistry topics. Nonetheless, EMMA sets a new standard for assessing MLLMs on multimodal reasoning.
this section cite: []

Section: References
Ref_id:b0 Title: A software suite for cheminformatics, computational chemistry, and predictive modeling. Greg Landruwu2024plot2codem Year: (2013)
Ref_id:b1 Title:  Year: ()
Ref_id:b2 Title: Gpt-4 technical report Year: (2023)
Ref_id:b3 Title: Complex visual reasoning with unanswerable questions Year: (2024)
Ref_id:b4 Title: Qvq: To see the world with wisdom Year: ()
Ref_id:b5 Title:  Year: ()
Ref_id:b6 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b7 Title: Qwen technical report Year: (2023)
Ref_id:b8 Title: Viseval: A benchmark for data visualization in the era of large language models Year: (2024)
Ref_id:b9 Title: Uniter: Universal image-text representation learning Year: (2020)
Ref_id:b10 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b11 Title: Comt: A novel benchmark for chain of multi-modal thought on large vision-language models Year: (2024)
Ref_id:b12 Title: On the measure of intelligence Year: (2019)
Ref_id:b13 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b14 Title: Exams-v: A multi-discipline multilingual multimodal exam benchmark for evaluating vision language models Year: (2024)
Ref_id:b15 Title: Introducing gemini 2.0: our new ai model for the agentic era Year: ()
Ref_id:b16 Title: Gemini 2.0 flash thinking mode Year: ()
Ref_id:b17 Title: Deepseek-r1-lite-preview is now live: unleashing supercharged reasoning power Year: (2024)
Ref_id:b18 Title: The llama 3 herd of models Year: (2024)
Ref_id:b19 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b20 Title: A survey on llm-as-ajudge Year: (2024)
Ref_id:b21 Title: A multimodal llm for chart understanding and generation Year: (2023)
Ref_id:b22 Title: Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems Year: (2024)
Ref_id:b23 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b24 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b25 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b26 Title: Novachart: A large-scale dataset towards chart understanding and generation of multimodal large language models Year: (2024)
Ref_id:b27 Title: Assessing causal reasoning in language models Year: (2023)
Ref_id:b28 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b29 Title: A benchmark dataset of mechanistic molecular images Year: (2024)
Ref_id:b30 Title: Llavaonevision: Easy visual task transfer Year: (2024)
Ref_id:b31 Title: Name reactions. a collection of detailed mechanisms and synthetic applications 4th edition Year: (2009)
Ref_id:b32 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b33 Title: Evaluating multi-modal code large language models with visually rich programming problems Year: (2024)
Ref_id:b34 Title: Oscar: Objectsemantics aligned pre-training for vision-language tasks Year: (2020)
Ref_id:b35 Title: Mmsci: A multimodal multi-discipline dataset for phd-level scientific comprehension Year: (2024)
Ref_id:b36 Title: Let's verify step by step Year: (2023)
Ref_id:b37 Title: One-shot visual language reasoning by plotto-table translation Year: (2022)
Ref_id:b38 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b39 Title: Visual instruction tuning Year: (2024)
Ref_id:b40 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b41 Title: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems Year: (2019)
Ref_id:b42 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b43 Title: Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts Year: (2024)
Ref_id:b44 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b45 Title: A benchmark for question answering about charts with visual and logical reasoning Year: (2022)
Ref_id:b46 Title: Reasoning over scientific plots Year: (2020)
Ref_id:b47 Title: Hello gpt Year: ()
Ref_id:b48 Title: Learning to reason with llms Year: ()
Ref_id:b49 Title: Introducing chatgpt pro Year: ()
Ref_id:b50 Title: Learn ap physics Year: ()
Ref_id:b51 Title:  Year: (2024)
Ref_id:b52 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b53 Title: Does spatial cognition emerge in frontier models? arXiv preprint Year: (2024)
Ref_id:b54 Title: A graduate-level google-proof q&a benchmark Year: (2023)
Ref_id:b55 Title: Visual cot: Unleashing chainof-thought reasoning in multi-modal language models Year: (2024)
Ref_id:b56 Title: Chartmimic: Evaluating lmm's cross-modal reasoning capability via chartto-code generation Year: (2024)
Ref_id:b57 Title: Scaling llm testtime compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b58 Title: Varco arena: A tournament approach to reference-free benchmarking large language models Year: (2024)
Ref_id:b59 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2022)
Ref_id:b60 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: (2022)
Ref_id:b61 Title: Learning cross-modality encoder representations from transformers Year: (2019)
Ref_id:b62 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b63 Title:  Year: ()
Ref_id:b64 Title: Internvl2: Better than the best-expanding performance boundaries of open-source multimodal models with the progressive scaling strategy Year: ()
Ref_id:b65 Title: Measuring multimodal mathematical reasoning with mathvision dataset Year: (2024)
Ref_id:b66 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b67 Title: Charting gaps in realistic chart understanding in multimodal llms Year: (2024)
Ref_id:b68 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b69 Title: Transformers: Stateof-the-art natural language processing Year: (2020-10)
Ref_id:b70 Title: Plot2code: A comprehensive benchmark for evaluating multi-modal large language models in code generation from scientific plots Year: (2024)
Ref_id:b71 Title: Chartx & chartvlm: A versatile benchmark and foundation model for complicated chart reasoning Year: (2024)
Ref_id:b72 Title: Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement Year: (2024)
Ref_id:b73 Title: Swe-bench multimodal: Do ai systems generalize to visual software domains? Year: (2024)
Ref_id:b74 Title: Mmtbench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi Year: (2024)
Ref_id:b75 Title: Contrastive captioners are imagetext foundation models Year: (2022)
Ref_id:b76 Title: Mm-vet: Evaluating large multimodal models for integrated capabilities Year: (2023)
Ref_id:b77 Title: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi Year: (2024)
Ref_id:b78 Title: Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark Year: (2024)
Ref_id:b79 Title: Raven: A dataset for relational and analogical visual reasoning Year: (2019)
Ref_id:b80 Title: Revisiting visual representations in vision-language models Year: (2021)
Ref_id:b81 Title: Multimodal chain-of-thought reasoning in language models Year: (2023)
Ref_id:b82 Title: Is gpt-4v (ision) all you need for automating academic data visualization? exploring vision-language models' capability in reproducing academic charts Year: (2024)
Ref_id:b83 Title: Image-of-thought prompting for visual reasoning refinement in multimodal large language models Year: (2024)
Ref_id:b84 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b85 Title: Main Structure: The image depicts the net of a cube, consisting of six numbered square faces. The net layout includes: A center square labeled 2. Three squares adjacent to the center square Year: ()
Ref_id:b86 Title: Task: Sascha is tasked with folding the cube, determining the opposite faces, and adding the numbers on those faces. o1-preview: To solve this problem, we need to determine which faces of the cube are opposite each other after folding the given net Year: ()
Ref_id:b87 Title: Let's first reconstruct the net as described: Square 2 is in the center Year: ()
Ref_id:b88 Title: Step 2: Assigning Squares to Cube Faces. We can assign the squares to the faces of the cube as follows: Front Face: Square 3, Back Face: Square 4 Left Face: Square 1 Year: ()
Ref_id:b89 Title: Step 3: Identifying Opposite Faces. The pairs of opposite faces are: (1) Year: ()
Ref_id:b90 Title: Front and Back Faces Year: ()
