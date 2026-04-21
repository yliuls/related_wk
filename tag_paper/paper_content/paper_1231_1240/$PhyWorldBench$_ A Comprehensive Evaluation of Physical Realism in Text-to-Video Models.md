Title: PH YWO R L DBE N C H : A COMPREHENSIVE EVALUATION OF PHYSICAL REALISM IN TEXT-TO-VIDEO MODELS
Abstract: Figure 2: Success rates of video generation models on PhyWorldBench. Among open-source models, Wanx demonstrated the highest performance, while Pika achieved the best results among proprietary models with a success rate of 0.262. Despite these advancements, substantial progress remains necessary to refine the capability of these models to accurately simulate the intricate dynamics of the real world. ical phenomena with varying prompt types, deriving targeted recommendations for crafting prompts that enhance fidelity to physical principles.

Section: ABSTRACT
Video generation models have achieved remarkable progress in creating highquality, photorealistic content. However, their ability to accurately simulate physical phenomena remains a critical and unresolved challenge. This paper presents PhyWorldBench, a comprehensive benchmark designed to evaluate video generation models based on their adherence to the laws of physics. The benchmark covers multiple levels of physical phenomena, ranging from fundamental principles like object motion and energy conservation to more complex scenarios involving rigid body interactions and human or animal motion. Additionally, we introduce a novel "Anti-Physics" category, where prompts intentionally violate real-world physics, enabling the assessment of whether models can follow such instructions while maintaining logical consistency. Besides large-scale human evaluation, we also design a simple yet effective method that could utilize current MLLM to evaluate the physics realism in a zero-shot fashion. We evaluate 12 state-of-the-art text-to-video generation models, including five open-source and five proprietary models, with a detailed comparison and analysis. we identify pivotal challenges models face in adhering to real-world physics. Through systematic testing of their outputs across 1,050 curated prompts-spanning fundamental, composite, and antiphysics scenarios-we identify pivotal challenges these models face in adhering to real-world physics. We then rigorously examine their performance on diverse phys-
this section cite: []

Section: INTRODUCTION
The field of video generation has made remarkable progress, with models producing visually compelling and often photorealistic outputs. These advances have enabled transformative applications across industries such as entertainment, education, and scientific visualization. However, despite their visual fidelity, do video generation models truly understand the laws of physics in the real world? To answer this question, we introduce PhyWorldBench, a rigorous benchmark designed to evaluate how well video generation models can simulate real-world physics. As illustrated in Figure 1, PhyWorldBench systematically tests models across multiple levels of physical phenomena, from fundamental concepts like object motion to complex dynamics, including rigid body interactions and human/animal motion. Additionally, we propose a novel Anti-Physics category, where prompts deliberately violate real-world physics. On one hand, this design verifies whether models genuinely understand physical laws-rather than merely reproducing patterns from real-world training data. On the other hand, anti-physics content itself holds practical value in creative applications, where imaginative or otherwise impossible scenarios are beneficial.
We meticulously designed and annotated 1,050 prompts and the standard set for each prompt individually to cover a broad range of physical scenarios. This substantial annotation work ensures that our benchmark is both comprehensive and precise, allowing for a more thorough assessment of video generation models' capabilities. Furthermore, we present a context-aware-prompt metric using MLLM (OpenAI Team, 2024;Gemini Team, 2024), which directly assesses if the video satisfies the physics standards or not. Such evaluation not only provided an unbiased metric but also significantly reduced the evaluation cost.
To examine the current status of video generation models and provide a detailed analysis, we selected five proprietary models-Sora-Turbo (OpenAI, 2024), Gen-3 (Runway Team, 2024), Kling 1.6 (KlingAI, 2024), Pika 2.0 (Pika Labs Team, 2024), and Luma (Luma AI Team, 2024)-along with seven open-source models: Hunyuan 720p (Kong et al., 2024), Open-Sora 2.0 (Peng et al., 2025), Open-Sora-Plan 1.3 (Lin et al., 2024), CogVideoX-1.5 (Hong et al., 2022), Step-video-T2V (Ma et al., 2025), Wanx-2.1 (WanTeam et al., 2025), and LTX-Video (HaCohen et al., 2024). We generated 12,600 videos to evaluate SOTA models and analyze their ability to simulate real-world physics. We identify challenges, difficult scenarios, and key physics categories while proposing a structured approach to improve physical realism through prompt design.
The insights obtained from this benchmark will contribute to the development of more robust and physically accurate video generation models, addressing both fundamental scientific questions and practical challenges in the field. Figure 2 presents the overall benchmark results on PhyWorldBench. Despite recent advancements, models continue to struggle with temporal consistency, realistic motion, and physical plausibility, emphasizing the need for further research to enhance their physics correctness for real-world simulation.
Our work provides a benchmark for video generation models with a focus on physical realism. The key contributions are:
• We propose PhyWorldBench, a large-scale, multi-dimensional physics benchmark for evaluating the physics ability of the video generation model.
• We conduct an extensive evaluation of twelve state-of-the-art video generation models (five proprietary and seven open-source) with 12,600 generated videos and identified key challenges in simulating real-world physics.
• We study the effect of prompt variation on the performance of the video generation model and provided prompt guidelines for generating physics-following videos.
this section cite: ['b3', 'b17', 'b13', 'b23', 'b15', 'b10', 'b31', 'b6']

Section: RELATED WORK
Benchmarks for Text-to-Video Generation. Proprietary video generators (OpenAI, 2024; KlingAI, 2024; Pika Labs Team, 2024; Runway Team, 2024; HailuoAI, 2024; Luma AI Team, 2024) achieve striking quality and temporal coherence but remain opaque. Open-source counterparts (Peng et al., 2025;Lin et al., 2024;Hong et al., 2022;Wang et al., 2023;HaCohen et al., 2024;Kong et al., 2024;Agarwal et al., 2025) enable reproducible research and flexible benchmarking. The development of Text-to-Video (T2V) Generation models has been accelerated by benchmarks that evaluate performance across diverse aspects like quality, realism, and compositionality. VBench (Huang et al., 2024) and EvalCrafter (Liu et al., 2024) provide comprehensive frameworks for assessing video generation models, focusing on metrics such as diversity, temporal coherence, and scalability. Besides, T2V-CompBench (Sun et al., 2024) emphasizes compositionality, ensuring generated videos align with intricate textual prompts, a key challenge in real-world applications. Recent works emphasize the importance of physical plausibility in video generation. Morpheus Zhang et al. (2025a) focuses on experimental physics. VideoPhy (Bansal et al., 2024) and PhyGenBench (Meng et al., 2024) assess models' adherence to physical commonsense, while DEVIL (Liao et al., 2024) and VideoPhy-2 (Bansal et al., 2025) evaluates dynamic realism, including motion and temporal consistency. Physics-IQ (Motameda et al., 2025) further tests understanding of fundamental principles like fluid dynamics and thermodynamics, highlighting the demand for physics-aware video generation.
Many physics-focused video generation models are also built (Zhang et al., 2025b;Yuan et al., 2025;Wang et al., 2025).
We introduce a novel and substantially expanded dataset for evaluating video generation models across a broader range of physical laws. Unlike prior benchmarks, it includes diverse, carefully curated scenarios with both real and deliberately unphysical dynamics. Models perform significantly worse on our benchmark under identical settings, revealing its greater difficulty and realism.
Evaluation Metrics for Text-to-Video Generation. Traditional text-to-video evaluation metrics like Fréchet Video Distance (FVD) (Unterthiner et al., 2018) and Inception Score (IS) (Salimans et al., 2016) focus on visual quality but fail to assess motion plausibility and adherence to physical laws, especially without reference videos. Recent methods address this gap: VideoScore (He et al., 2024) integrates human feedback, T2V-CompBench (Sun et al., 2024) uses VLMs for spatial and temporal evaluation, and PhyGenBench (Meng et al., 2024) explicitly tests physical commonsense. However, these methods can be computationally intensive or subjective. We introduce a Yes/No answering metric for efficient, scalable, and objective evaluation of physical commonsense in video generation. Additionally, we propose a simple yet effective strategy leveraging MLLMs like GPT-o1 to assess video quality.
this section cite: ['b23', 'b15', 'b10', 'b30', 'b6', 'b13', 'b0', 'b11', 'b16', 'b27', 'b1', 'b19', 'b14', 'b2', 'b20', 'b33', 'b29', 'b28', 'b26', 'b9', 'b27', 'b19']

Section: Create Physics Categories Prompt Crafting Evaluation Standards
Physics Textbooks Expert Consultations
this section cite: []

Section: GPT-4o Human
Human-in-the-loop Review  (2) Physics-enhanced prompt: "A rocket launches vertically into the sky, traveling straight into the air."
✓ Clear ✓ Diverse ✓ Correct ✓ Same standard ✓ Correct ✕ Ambiguous ✓ Clear ✓ Exclusive ✓ Comprehensive Human
(3) Detailed narrative depiction: "In the predawn twilight, a sleek, towering rocket emblazoned with a nation's flag thunders to life, its engines roaring as it ascends into the darkening indigo sky. The ground trembles, and a plume of fiery exhaust illuminates the surrounding landscape with a flickering glow. As the rocket streaks upwards, its silhouette momentarily casts elongated shadows across the control platform below, while excited onlookers shield their eyes from the intense brightness, and camera lenses follow its path, capturing every moment of its majestic climb into the stratosphere." Time Time Time
this section cite: []

Section: PH YWO R L DBE N C H
To evaluate the ability of text-to-video generation models to simulate physical reality, we introduce PhyWorldBench, a comprehensive benchmark spanning a wide range of physical principles and scenarios. Its design is rooted in fundamental physics concepts such as motion, energy conservation, and interaction dynamics, drawing from established literature, including Fundamentals of Physics (Halliday et al., 2013) and Classical Mechanics (Goldstein et al., 2002). Developed with input from experts in physics and video generation.
this section cite: ['b8', 'b4']

Section: DATASET CREATION PIPELINE
As shown in Figure 3, the creation pipeline consists of three steps, with human-in-the-loop reviews integrated at each step to ensure the benchmark's accuracy and diversity.
this section cite: []

Section: Step 1: Initial Physics Categories Definition.
A collaborative process between physics experts and the authors to define the key (main) physics categories to be included in the benchmark. Our benchmark categorizes three levels:
• Fundamental Physics: Covers basic laws such as object motion, energy transfer, and optics.
• Composite Physics: Involves real-world phenomena that emerge from multiple interacting principles, such as human motion. • Anti-Physics: Scenarios intentionally designed to violate real-world physics.
The inclusion of anti-physics cases is crucial, as discrepancies in model performance between physically accurate and unphysical scenarios reveal whether a model truly understands physics or merely replicates patterns from its training data, which predominantly follows real-world laws.
We identify broad categories-such as kinematics, rigid body dynamics, fluid behavior, optics, and thermal dynamics-as depicted in the main physics categories in Figure 1. Each category is systematically divided into five subcategories to ensure comprehensive coverage of physics principles from multiple perspectives.
Step 2: Prompt Creation Each subcategory is further expanded into 7 distinct scenarios, with each scenario incorporating three prompt variations: Event Prompt, Physics-enhanced Prompt, and Detailed Narrative Prompt, as illustrated in Figure 4. Specifically, we define these three types of prompts as follows:
• Event Prompt: A concise and straightforward description of an event. These prompts are designed to assess the model's ability to generate based on minimal input, focusing on high-level scene comprehension.
• Physics-Enhanced Prompt: This type incorporates physics-related phenomena for Event Prompt to enrich the description naturally.
• Detailed Prompt: Inspired by how current T2V models generate videos, these prompts are enriched with vivid details and contextual elements, aiming to create more immersive and visually rich outputs. This level of prompt allows us to assess whether providing richer context leads to improved physical accuracy in video generation.
We first use state-of-the-art large language models (LLMs), specifically GPT-4o (OpenAI Team, 2024) and Gemini-1.5-Pro (Gemini Team, 2024) to generate 20 Event Prompts under the predefined category and subcategory. Then we ask humans to select 7 out of them based on diversity and physics representation. Then human experts meticulously review and validate the generated prompts, providing feedback to iteratively enhance their quality and relevance to the corresponding physics subcategory, ensuring a balanced and representative assessment across various aspects of physical understanding. Furthermore, an ethical review is conducted to ensure that all prompts are fair, unbiased, and free from potentially harmful content. Based on the Event Prompt, human annotators create a Physics-Enhanced Prompt by enriching it with physical consequences. Detailed Narrative Prompt is generated by LLM based on Event Prompt using query in Tab. 12. This rigorous verification process results in a refined set of 1,050 high-quality prompts, systematically organized into main and subcategories.
Step 3: Standard Creation. We adopt a Yes/No evaluation metric to assess whether a model can generate videos that accurately align with a given prompt. This metric is chosen for its simplicity and effectiveness and for allowing for an objective assessment while minimizing subjectivity in evaluation. As shown in Figure 5, we define two types of evaluation standards: Basic Standards and Key Standards. The Basic Standards specify the essential object(s) that should appear in the video and ensure that the main action or event is depicted. Meanwhile, the Key Standards describe the key physical phenomena that would occur if the event took place in the real world. The annotators are instructed to focus on the natural physical consequences of the action and to be explicit and concise in their annotations. By following these guidelines, we ensure that the evaluation standards remain clear and objective. Following (Bansal et al., 2024;Meng et al., 2024), we use Semantic Adherence (SA) and Physical Commonsense (PC) to evaluate video performance. SA checks if both "objects" and "Action/Event" align with video, while PC assesses adherence to real-world physics and it checks Prompt: Two football players tackle each other.
Prompt: An apple falls from a tree branch.
Time
this section cite: ['b3', 'b1', 'b19']

Section: Basic Standards
Objects: two football players Action/Event: tackle each other
this section cite: []

Section: Key standards:
(1) The contact point between the players should occur naturally, with the bodies realistically reacting and deforming upon impact. (2) The momentum and direction changes of both players after the collision should be smooth and believable, respecting the laws of physics.
Yes/No ✔ ✔ ✔ ✗ Yes/No
Basic Standards Key standards:
(1) The apple initially falls slowly from the branch, with its speed visibly increasing as it approaches the ground. (2) The trajectory of the apple is a straight vertical path, showing no horizontal deviation or external disturbances.
this section cite: []

Section: Yes/No
Objects: an apple, a tree branch  In the two examples, prompt (a) explicitly tells the MLLM that the video might contain potential issues, while prompt b does not. We found that the quality assessment is usually more accurate when the context is used. The text is irrelevant to quality is ignored for readability. This phenomenon applied to all our tested models including GPT-4o, GPT-o1, Gemini-2.0-flash, and Qwen-VL-2.0.
when all "Key Standards" are satisfied. Annotated as SA, PC ∈ 0, 1 (1 indicating proper grounding), we report the fraction of videos satisfying a success when SA = 1, PC = 1, and both jointly.
this section cite: []

Section: STATISTICS OF THE DATASET
We compare our benchmark with existing physics-focused T2V benchmarks (Bansal et al., 2024;Meng et al., 2024;Guo et al., 2025;Motameda et al., 2025), with a detailed statistical comparison presented in Tab. 1. PhyWorldBench offers the most extensive coverage in terms of the number of physics categories, curated prompts, human annotations, and evaluated T2V videos.  Team, 2024) and Gemini (Gemini Team, 2024) show promise for video evaluation but fall short in reliably assessing physical realism (Bansal et al., 2024;He et al., 2024;Meng et al., 2024). We find that these models tend to rationalize unrealistic content, but their accuracy improves when informed that the video is AI-generated, as illustrated in Figure 6. We hypothesize that this improvement stems from MLLMs being primarily trained on real-world videos, leading them to inherently justify observed phenomena unless prompted otherwise. Inspired by this, we propose a simple yet effective method, Context-Aware Prompt (CAP), which explicitly informs the MLLM that the video is generated rather than real. This approach enables the MLLM to provide a more accurate assessment of the video's physical realism. Critically, CAP does not assume videos are incorrect by default but instead uses a structured chain-of-thought prompt-first asking the model to describe the video and then to reason through any potential physics issues-and only labels a video as incorrect if it clearly violates the category-specific standards. We validate CAP neutrality by measuring false positive rates in physics correct videos and observe only a 0.001 absolute change in semantic adhesion (0.188 to 0.187) and a 0.014 change in physical commonsense (0.172 to 0.186), showing that it does not significantly increase misclassification of correct videos.
Besides, inspired by Chain-of-Thought (Wei et al., 2022), we design a two-step prompting strategy, where we first ask the MLLM to give a thorough description of the video including Object, Event, and Observations, then continue to prompt the MLLM for the final "Yes/No" answer for video quality. Please refer to Appx. K for prompts used in the two steps. We found this leads to consistent improvement in various MLLMs. In Tab. 5, we present the ROC AUC calculations comparing CAP with human results on PhyWorldBench with videos generated by models in Figure 2, where a random guess would achieve a score of 50. We evenly sample 8 frames. In this evaluation, CAP is powered by GPT-o1. CAP significantly outperforms other methods, particularly in Physics Commonsense, where it achieves an absolute performance boost of 13.5 compared to the standard GPT-o1. We find CAP also achieves great performance when equipping other proprietary or opensourced MLLM. The bottom of Tab. 5 shows the ablation study of various components in CAP. "without CoT" means we directly ask the MLLM to give the answer of the question, and "without Context" means we do not inform MLLM that the video could be of low quality. Please refer Appx. D for performance of our method on other MLLM and analysis.
this section cite: ['b1', 'b19', 'b18', 'b20', 'b1', 'b9', 'b19', 'b32']

Section: VIDEO GENERATION RESULTS AND ANALYSIS
We evaluate both closed-source models including Sora-Turbo (OpenAI, 2024), Gen-3 (Runway Team, 2024), Kling 1.6 (KlingAI, 2024), Pika 2.0 (Pika Labs Team, 2024), Luma (Luma AI Team, 2024), and open-source models including Hunyuan 720p (Kong et al., 2024), Open-Sora 2.0 (Peng et al., 2025), Open-Sora-Plan 1.3 (Lin et al., 2024), CogVideoX-1.5 (Hong et al., 2022) Appx. E for the implementation details. We generate 1050 videos for each model. We conduct a human evaluation of generated videos using Amazon Mechanical Turk. We first introduce model performance and model-wise analysis and then break down the challenge for the video generation model to achieve real-world physics.
this section cite: ['b17', 'b13', 'b23', 'b15', 'b10']

Section: MODEL PERFORMANCE
Tab. 3 presents a comprehensive evaluation of twelve video generation models. We report the percentage of videos that satisfy both semantic adherence and physical commonsense criteria (Both).  Luma (Luma AI Team, 2024 The Overall Performance column provides an aggregate score for each model, averaged across 10 physics categories. Pika 2.0 achieves the best overall performance across all models. In open-sourced models, Hunyuan 720p achieves the best performance on PC, Wanx-2.1 achieves the best performance on Both, and CogVideoX-1.5 achieves the best SA performance. It is worth noting that they even achieve a better performance compared with some of the proprietary. Please check Appx. P for Leaderboard.
this section cite: []

Section: MODEL SPECIFIC ANALYSIS
Our evaluation reveals a trade-off between realism, stylization, and consistency in text-to-video models.
this section cite: []

Section: EFFECT OF PHYSICS AND PROMPT ENHANCEMENT
Understanding real-world physics is essential for text-to-video models. Event-based prompts describe actions, while physics-enhanced prompts include natural consequences. A model's ability to generate realistic videos based on these prompts reflects its real-world understanding. As proprietary models conceal their generation processes, we cannot verify prompt refinement. Instead, we analyze prompt types using open-source models, where prompt integrity can be manually ensured. In Tab. 4, we compare model performance across different prompt types to gauge the effects of physics-aware prompting and refinement. While refinement improves clarity, object detail, and aesthetics, it does not notably enhance physics accuracy. The persistent inconsistencies suggest that these refinements primarily address surface-level quality, leaving deeper physics limitations unresolved. On the other hand, physics-enhanced prompts often yield better results, reflecting the models' strong promptfollowing ability-even though they still struggle with true physical comprehension. Consequently, we propose a straightforward receipt: explicitly integrate physical phenomena into the prompt to nudge the model toward more realistic outcomes.
this section cite: []

Section: ANTI-PHYSICS ANALYSIS
As shown in Tab. 3, all models show a performance decline from Fundamental to Composite to Anti-Physics categories, reflecting the challenge of generating complex physical phenomena. While they capture basic physics, simulating intricate interactions and "unphysical" scenes remains difficult, highlighting a gap in physics adherence. Our analysis found that failures often stem from models interpreting prompts in a more "reasonable" way rather than attempting and failing. For example, when prompted with "A glass sits on the table, and the wine in the glass has reversed gravity," the model generates a still image, adhering to real-world physics instead of the prompt. This suggests that models prioritize mimicking training data over true physics understanding.
this section cite: []

Section: FAILURE TO HANDLE ERRATIC VISUAL CHANGES
Certain physical events inherently result in erratic visual changes, such as collisions, breaking, or sudden state transitions. However, current video generation models struggle to accurately depict these scenarios. In the third example of Figure 10, a glass cup falls from a table, an event that should cause it to shatter upon impact. Instead, the model rationalizes the action, generating a video where the glass remains intact. Rather than simulating realistic disruptions in structure, motion, or object state, the models tend to default to continuous, smoothed animations, avoiding abrupt transformations that are critical for representing real-world physics. We created 20 paired prompts for each test, applied them to 12 video generation models (240 videos total), and only evaluate the visual change correctness through human evaluation. We also highlight Pika, the strongest model overall. Tab. 16 shows models struggle with abrupt visual changes -only 22.1% success overall vs. 42.1% for softened prompts; Pika 30% vs. 75%.When prompted with erratic visual changes, the model tends to simplify the task and generate the easier version.
this section cite: []

Section: BREAKDOWN IN PHYSICS AT HIGHER COMPLEXITY
Models' performance deteriorates under complex interactions involving multiple forces or constraints.
As demonstrated in Appx. R, prompts involving multiple objects introduce significantly higher requirements for semantic adherence. For instance, when with the prompt "A feather and a stone are dropped in the air, where both fall, and the stone drops much faster", all models fail to present the phenomenon correctly. The current models lack a robust understanding of how multiple physical forces interact in complex environments. We similarly built 20 prompt pairs focused on interactions of multiple forces or collisions, applied them to the 12 models, and highlighting Pika as the best model as in Tab. 17. Physics further breaks down at higher complexity -12% success overall vs. 18% for simpler cases; Pika 15% vs. 40%. This suggest that the current video models do not really understand the physics thoroughly and are more likely to mimic the pattern in the training set.
Table 4: Physics-following percentage on different types of prompt. Explicitly adding physics usually increases the physics-following ability of video generation model, while a prompt refinement process does not necessary leads to improvement.
this section cite: []

Section: Model Event Prompt Physics-enhanced Prompt Detailed Prompt
CogVideoX-1.5 (Hong et al., 2022) 0.123 0.177 0.168 Hunyuan 720p (Kong et al., 2024) 0.159 0.198 0.155 LTX-Video (HaCohen et al., 2024) 0.056 0.065 0.066 Open-Sora-Plan 1.3 (Lin et al., 2024) 0.062 0.067 0.063 Open-Sora 2.0 (Peng et al., 2025) 0.167 0.177 0.173 Wanx-2.1 (WanTeam et al., 2025) 0.175 0.202 0.190 Step-Video-T2V (Ma et al., 2025) 0.158 0.182 0.179 4.7 CINEMATIC EFFECTS VS. PHYSICAL PLAUSIBILITY A major challenge in video generation is the tendency for models to prioritize cinematic aesthetics over strict physical realism. Well-performed models like Sora, Gen-3, and Pika often introduce exaggerated motion dynamics, such as objects moving with heightened fluidity or unnatural acceleration, making scenes appear choreographed rather than physically grounded. For example, an object that should fall naturally under gravity might instead descend too smoothly or float unrealistically. This issue is particularly evident when momentum conservation and force dynamics are distorted for dramatic effect. The first example in Figure 11 presented an apple floating in the air, then it suddenly dropped as the camera moved; such stylized rendering, while beneficial for artistic storytelling, poses significant challenges for applications requiring strict physical fidelity.
this section cite: ['b10', 'b13', 'b6', 'b15', 'b23', 'b31', 'b18']

Section: CONTRIBUTIONS AND FUTURE WORK
We propose PhyWorldBench, a large-scale, thorough, and multi-dimensional benchmark for evaluating text-to-video generation models. It provides insights into which physical phenomena are hardest to simulate and what capabilities current models lack. Additionally, we propose an automated evaluator to measure physical accuracy and a straightforward receipt for designing prompt. Future work will involve continuously evaluating emerging video generation models, updating our benchmark to reflect advancements in the field, and refining our leaderboard and analysis to track progress in physical accuracy and model capabilities. LLM usage is described in Appx. A.
second, refining prompts through iterative reviews with large language models and human experts to ensure clarity and alignment with physics principles; and third, conducting detailed curation to create a final set of 1,050 high-quality prompts for a diverse set of physics phenomena. The evaluation utilizes a yes/no framework to assess model adherence to physical laws, categorized into basic standards and key standards, enabling comprehensive performance evaluation. This benchmark establishes a foundation for evaluating models' ability to generate videos that align with both physical principles and real-world scenarios.
Following Bansal et al. (2024) and Meng et al. (2024), we use Semantic Adherence (SA) and Physical Commonsense (PC) as metric to evaluate the video performance. SA measures if a caption aligns with video frames, while PC assesses if actions obey real-world physics. Annotated as SA, PC ∈ {0, 1}, where 1 indicates proper grounding. We report the fraction of videos satisfying SA = 1, PC = 1, and both jointly. We create SA and PC based on a thorough review of all prompts and extensive feedback from human annotators, we establish a set of evaluation criteria to systematically assess the generated videos. The evaluation is structured around two key standards, each encompassing specific rules:
• Semantic Adherence: These criteria evaluate fundamental aspects of video generation to ensure prompt fidelity. The assessment includes:
1. Whether the video contains the correct number of objects specified in the prompt. (Yes/No) 2. Whether the depicted event or action is accurately represented. (Yes/No)
• Physical Commonsense: This aspect evaluates the model's capability to capture and represent underlying physical phenomena within the video. The assessment measures the number of distinct physical phenomena accurately showcased in the generated video.
By integrating these two standards, which consist of three core evaluation rules, we define a comprehensive evaluation metric that ensures a balanced and thorough assessment of T2V model performance across multiple dimensions.
this section cite: ['b1', 'b19']

Section: C ANNOTATION DETAILS
Prompt Annotation After designing the physics categories, we further prepared prompts within each category. To achieve this, we gathered undergraduate and graduate students majoring in Computer Science for an annotation session. They were instructed to provide high-quality annotations by creating both an Event Prompt and a corresponding Physics-enhanced Prompt that aligned with the primary physics category and sub-category. The Detailed Prompt was generated using GPT-4o with the MLLM prompt (as described in Tab. 12) and subsequently verified by human annotators. Finally, each prompt underwent a final verification by the authors.
this section cite: []

Section: Standard Annotation
Undergraduate and graduate students in Computer Science were also tasked with annotating the objects and events described in the prompts. These annotations were used for SA (Semantic Adherence) evaluation. Additionally, they identified key physics phenomena that should be visually observable in the generated videos, which were used for PC (Physics Commonsense) evaluation. All annotations were reviewed and verified by the authors.
this section cite: []

Section: Video Evaluation
For large-scale video evaluation, we utilized Amazon Mechanical Turk. The interface is presented in Figure 7 Annotators were asked to assess videos based on three criteria: the existence of objects, the presence of the described event, and the visibility of key physics phenomena, selecting either "Yes" or "No" for each. Each video was evaluated by three independent annotators, with the final decision determined by a majority vote. All annotators were compensated at a rate of $18 USD per hour.
this section cite: []

Section: D DETAILED ANALYSIS ON EVALUATOR
While some existing physics metrics have been proposed, none fit PhyWorldBench's needs due to domain or structural mismatches (Guo et al., 2025;Zhang et al., 2025a). For example, PhyGenEval (Meng et al., 2024) assumes a fixed event order, which many of PhyWorldBench's scenarios do not have, and VideoCon-Physics (Bansal et al., 2024) is trained specifically on its own  VideoPhy benchmark, yielding sub-optimal performance on PhyWorldBench-so we exclude both to avoid unfair comparisons. Besides, VideoPhy-2 (Bansal et al., 2025) focused on action and dynamic instead of general physics, and its evaluator is also designed and finetuned on the associated benchmark annotation, therefore for the same reason, we also did not include it in this paper.
In Tab. 6, we therefore evaluate our CAP across multiple MLLMs, including proprietary GPT-4o, GPT-o1, Gemini-Flash-2.0, and open-source Qwen-VL-2.0. Crucially, CAP uses a two-step chain-of-thought prompt-first asking the model to describe the video, then to reason about any physics issues-so it does not assume videos are incorrect by default and yields only minimal false-positive increases (0.001 in Semantic Adherence, 0.014 in Physical Commonsense), confirming its neutrality. Moreover, because it operates solely on the final video and its physics-based standards, CAP is agnostic to the underlying generative pipeline-whether text-to-video, image-to-video, video-to-video, or hybrid.
We do observe two common failure modes: 1) Aesthetic bias, where visually compelling videos receive high scores despite only moderate physical accuracy (visual appeal and physical plausibility should ideally be assessed independently); and 2) Overconfidence, with the MLLM giving firm judgments in ambiguous or edge-case scenarios where human annotators often express uncertainty or disagreement.
To further validate our design, we also measure Precision and Recall for both Semantic Adherence (SA) and Physical Commonsense (PC), which is shown in Table 5.
this section cite: ['b18', 'b19', 'b1', 'b2']

Section: E MODEL APPLICATION DETAILS
Open-source generators (code + weights).
• Hunyuan (Tencent Hunyuan Community License Agreement) : based on the official 720p repo https://github.com/Tencent/HunyuanVideo. • CogVideo v1.5 (Apache-2.0 License) : https://github.com/THUDM/CogVideo.
• Open-Sora v2.0 (Apache-2.0 License) : https://github.com/hpcaitech/  Open-Sora.
• Wanx v2.1 (Apache-2.0 License) : https://github.com/Wan-Video/Wan2.1.
• Step-Video-T2V (MIT License) : https://github.com/stepfun-ai/  Step-Video-T2V.
• Open-Sora-Plan v1.3 (MIT License) : https://github.com/PKU-YuanGroup/  Open-Sora-Plan.
Commercial / proprietary generators.
• Sora-Turbo, Pika 2.0, Gen-3, Kling 1.6: accessed via the vendors' public web interfaces under their standard Terms of Service (no OSS license).
• Luma: accessed through the official API.
All third-party code or data are used under the above licenses/ToS but are not redistributed in our release. Our own benchmark assets are released under the MIT License.
this section cite: []

Section: F CAP RESULTS
Here we further include the automatic evaluation results of CAP on the video generation models on Tab. 7.
this section cite: []

Section: G INSIGHTS INTO PERFORMANCE BETWEEN MODELS
Our analysis reveals that both model size and sampling budget significantly impact physics performance. Larger models consistently outperform smaller ones on foundational physics tasks, as shown in Table 8. For instance, the fact that Cosmos 14B outperforms Cosmos 7B across all SA/PC metrics confirming that scaling yields clear gains in semantic adherence and physical commonsense.
We also observe that generation time correlates with physics accuracy: models with longer sampling schedules (e.g., Hunyuan at 40 min/video or Wanx-2.1 at 1 hr/video) achieve stronger performance, and simply increasing Open-Sora's sampling steps from 50 to 200 yields a significant boost. These
this section cite: []

Section: H DETAILED PERFORMANCE OVER ALL PHYSICS CATEGORIES
In the main paper, we presented results across different levels of physics. Here, we provide a detailed breakdown of results for each specific physics category in Tab. 9.
this section cite: []

Section: I LIMITATIONS
While PhyWorldBench spans a broad set of physical scenarios with its 1,050 prompts across ten categories, it may not fully capture very specialized or highly intricate interactions that occur in some domains or advanced applications. Similarly, our CAP evaluator generally provides balanced feedback, yet it can display a slight preference for videos with particularly polished visual presentation, such as smooth lighting or dynamic camera movement. Future work will focus on expanding the prompt collection to include additional niche phenomena and on enhancing the evaluation workflow to more clearly distinguish visual style from physical correctness.
this section cite: []

Section: J CODE AND DATA AVAILABILITY
To ensure full reproducibility, we provide both the generated video assets and all experiment code:
• Generated videos and prompts: All videos used in our experiments, along with the corresponding prompt JSON files, are hosted on HuggingFace at https://huggingface.  co/datasets/phyworldbench/phyworldbench. Note that these generated videos are not part of PhyWorldBench, they were instead utilized for our paper experiments. Also, the HuggingFace does not contain the evaluation standards, those are part of the GitHub listed below.
• Experimentation code and instructions: The complete implementation of the experiments code as well as detailed setup instructions are available at https://github.com/  ashwin-333/phy-world-bench. • LTX-Video: Produces grainy footage with distorted objects that are difficult to identify. Moving objects often shift shape and struggle to maintain form.
• Luma: Tends to generate realistic footage but with overly vibrant or bright colorization. Moving objects sometimes experience distortion or loss of form.
• Open-Sora-Plan: Generates relatively realistic videos but with low fidelity, often lacking detail and featuring still objects. Movement can appear overly smoothed or surreal, sometimes resembling slow-motion. The output often has a VHS-like quality.
• Open-Sora: Generates videos where objects frequently distort and warp, failing to hold their shape. The outputs often appear less realistic, resembling animation or 3D graphics.
• Pika: Produces realistic yet highly stylized videos. Lighting and objects have an unrealistic softness, contributing to a distinctive stylized aesthetic.
• Sora: Generates highly realistic videos with cinematic or stylized lighting. Movement is often exaggerated, enhancing the dramatic quality of the footage.
• Wanx: Produces visually crisp footage, yet physical interactions are frequently unreliable; rigid bodies may pass through one another, gravity-driven motions appear "floaty," and impacts are often muted or entirely missing.
• Step-Video-T2V: Creates sharp, aesthetically pleasing clips, yet physical reasoning is routinely oversimplified; light fails to refract, flexible objects stay rigid, mass differences do not affect motion, and phenomena such as duplication or buoyancy are either absent or behave contrary to real-world expectations.
this section cite: []

Section: M STATIC-SCENE RATIONALIZATION AS AN EMERGING FAILURE MODE
Beyond explicit physical violations, we observe that several text-to-video models exhibit a distinct failure mode in which the model generates a partially or fully static scene to avoid producing incorrect physical motion. Instead of attempting the intended physical interaction, the model suppresses motion entirely while still producing a visually coherent video. This behavior effectively "rationalizes" the violation by minimizing dynamics.
To systematically investigate this phenomenon, we analyzed 600 failure cases from our evaluation set, sampling 50 failed videos from each of the 12 models. Each video was annotated for whether it exhibits static-scene rationalization, defined as replacing the expected motion with a static or minimally animated scene.
A clear trend emerges: models with higher overall generation quality tend to exhibit substantially higher rationalization rates. Lower-performing models generally fail due to more fundamental issues-such as incorrect object identities, distorted geometry, or inconsistent motion-leaving little opportunity to "rationalize." In contrast, stronger models, once capable of producing semantically aligned and visually coherent scenes, often avoid explicit physics violations by freezing or minimizing motion, thus prioritizing aesthetic quality over physical correctness.
This pattern highlights a key challenge for future text-to-video systems. As models advance in visual fidelity, they may increasingly adopt conservative strategies that circumvent difficult physics rather than attempting them. Quantifying this emerging failure mode is essential for developing models that do not merely avoid errors but actively engage in physically grounded reasoning.            Book exhibits abnormal phasing during its motion. (a) A glass cup is table-phasing, and it passes through a solid table instead of resting on it.              (a) A beam of light hits the surface of a calm pond, reflecting partially while refracting into the water.  An object strikes a calm pond, but there is no visible beam of light or refraction as it enters the water. b. A tree branch remains rigid in the wind, never bending or springing back. c. A child and an adult jump on a trampoline yet rebound to identical heights, ignoring mass-dependent dynamics. d. A single coin was expected to duplicate continuously, but no new coins appear. e. A balloon with a hole should deflate downward; instead, it inexplicably rises, contradicting physics.
A child experiences circular motion on a merry go round, traveling in a perfect circle.
As a pendulum is released from a raised position, it swings downward, accelerating as it reaches its lowest point, and decelerating as it ascends.
Two billiard balls collide, bouncing off of each other continuing to move in opposite directions.
A boat moves across a lake, leaving a V-shaped wake as it displaces water along its path.
The gymnast tries to adjust their body to keep their center of mass aligned over the narrow beam, maintaining balance.
A ceiling fan spins under a bright light, intermittently blocking the light and creating rhythmic moving shadows around the room.
A steel rod bends when pushed by a hydraulic press, deforming slightly and springing back to its original form when the force is removed.
A candle flame and a bonfire react to a sudden gust of wind.
A child balances while walking on a curb.
A ball rolls uphill on its own, gaining energy as it moves. Object Motion and Kinematics Interaction Dynamics Energy Conservation Fluid and Particle Dynamics Rigid Body Dynamics Lighting and Shadows Deformations and Elasticity Scale and Proportions Human and Animal Motion (Biomechanics) Anti-Physics Fundamental Physics Complex Physics Anti-Physics (c) Human and Animal Motion: Anatomical Feasibility, Balance and Posture, Locomotion Across Terrains, Reflexive and Involuntary Motion, Coordination and Fine Motor Skills ANTI-PHYSICS (a) Anti-Physics: Defying Gravity, Energy Creation or Perpetual Motion, Object Phasing and Surreal Interactions, Time Reversal and Alteration, Infinite Duplication and Division
This hierarchical structure ensures coverage of realistic and non-realistic physical scenarios for diverse analysis. Figure 21 visualises how these ten categories map onto the three physics levels and highlights the numbers of prompts in each branch.
this section cite: []

Section: IMPACT STATEMENT
PhyWorldBench establishes a new standard for physics-focused text-to-video generation, significantly elevating the field's expectations of realism and accuracy. By rigorously assessing models' capabilities across a comprehensive spectrum of physical phenomena-including motion, energy transfer, and complex interactions-this benchmark not only highlights critical gaps in current systems but also guides researchers toward developing more robust and physics-aware solutions. Moreover, PhyWorldBench acknowledges the ethical implications of physics-driven video synthesis, emphasizing the need for transparency, fairness, and responsible deployment. By mitigating risks such as misinformation, biased simulations, and unintended real-world consequences, it ensures that advancements in this field are aligned with scientific integrity and societal well-being. Ultimately, PhyWorldBench paves the way for advanced applications in education, scientific visualization, and industry, demanding both photorealism and fidelity to real-world physics while upholding ethical standards.
this section cite: []

Section: 
Ethics statement The authors confirm that we have carefully reviewed and adhered to the ICLR Code of Ethics in this work.
Reproducibility statement We open-sourced our codebase to ensure reproducibility. We first released all prompts, evaluation standards, and we released the full evaluation code of our benchmark. We also released all the generated videos and their corresponding prompts.
this section cite: []

Section: 
; * Co-advisors Object Motion and Kinematics a. Linear Motion b. Parabolic Motion c. Circular Motion d. Rotational Motion e. Combined Translational & Rotational Motion Interaction Dynamics a. Collisions b. Friction and Sliding c. Gravity and Free Fall d. Electromagnetic Interactions e. Tension and Compression Energy Conservation a. Potential to Kinetic Energy b. Elastic Energy c. Energy Loss (Damping) d. Conservation of Mechanical Energy e. Heat Transfer and Phase Changes Fluid and Particle Dynamics a. Water Motion b. Smoke and Gas Dynamics c. Particle Behavior d. Buoyancy and Floating e. Viscosity and Flow Rigid Body Dynamics a. Rotation and Torque b. Balance and Stability c. Center of Mass d. Impact and Deformation e. Shear and Cutting Lighting and Shadows a. Consistency of Illumination b. Object Occlusion c. Chromatic Aberration & Dispersion d. Reflection and Refraction e. Shadow Softness & Penumbra Deformations and Elasticity a. Material Flexibility b. Elastic Rebound c. Plastic Deformation d. Brittle Fracture e. Thermal Expansion & Contraction Scale and Proportions a. Consistency of Forces Across Scales b. Environmental Effects on Different Scales c. Proportional Energy Transfer d. Structural Integrity Across Scales e. Scaling of Mechanical Properties
this section cite: []

Section: A LLM USAGE STATEMENT
The large language model is used during the manual script proofreading to detect grammatical errors and rephrase some overly long sentences. Meanwhile, as mentioned in the paper, our designed evaluator is based on LLM, the prompt draft creation process also involved LLM.
this section cite: []

Section: B BENCHMARK DETAILS
The benchmark revolves around 10 main physics categories, each divided into 5 subcategories that cover specific physics phenomena, such as object occlusion, center of mass, and circular motion. Each subcategory includes 7 distinct scenarios, with 3 variations of prompts: a general prompt, a physics-enhanced prompt, and a detailed narrative prompt. These variations are designed to provide differing levels of complexity and context for the text-to-video models. The dataset was curated through a three-stage process: first, defining key categories based on fundamental physics literature; K QUERY USED Here we list the three MLLM queries used in this work, corresponding to the first-stage description prompt (Tab. 10), the second-stage structured-answer prompt (Tab. 11), and the detailed-prompt refinement query (Tab. 12). Note that for proprietary MLLMs such as GPT-o1, GPT-4o and Gemini-2.0-Flash, the model may refuse to evaluate a small fraction of generated videos; the refusal rate is below 3 Table 10: First prompt for MLLM to generate a detailed description.
Suppose you are an expert in judging and evaluating the quality of AI-generated videos. These are frames evenly sampled from a generated video from the beginning to the end. This is a generated video from a video model rather than captured from real world, so the video could be low quality, such as fuzzy, inconsistency, especially not following real world physics. Please tell me what is in this video, including what happened and any physics phenomena you observe. Please be sure to include objects in the video, the main event, and any physics phenomena you observe.
Table 11: Second prompt for MLLM to generate the final answer Suppose you are an expert in summarization and finding answers. Here is the text description from another large language model about an AI generated video: "{previous response}". Based on this description, compare the objects and quantities present in the video with the specified object(s): "{object}". Answer "Yes" if the object(s) could be found in the video, otherwise answer "No". Also, please check if "{event}" is visually depicted in the video, and answer "Yes" or "No". Lastly, please check if video satisfies the standards in list: "{physics phenomenon list}", and answer "Yes" or "No" for each standard in the list. Return your evaluation in the following JSON format: "Objects": "Yes/No", "Event": "Yes/No", "Standard 1": "Yes/No", "Standard 2": "Yes/No", "...": "Yes/No" Table 12: MLLM prompt used to get detailed video generation prompts.
Refine the sentence: "{prompt}" to contain subject description, action, scene description. (Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. Make sure it is a fluent sentence, not nonsense.
this section cite: []

Section: L QUALITATIVE ANALYSIS ON INDIVIDUAL MODEL
We present here qualitative examples from various models, as shown Figure 8 • CogVideoX-1.5: Tends to produce distorted objects, such as non-spherical balls that appear dented. Movements are often excessively fast, blurred, erratic, or jittery, sometimes appearing unnaturally sped up.
• Gen-3: Generates videos with a cinematic quality, featuring rich lighting and a high level of realism, akin to footage captured with a high-quality camera. Lighting is particularly smooth and well-balanced.
• Hunyuan: Produces highly realistic videos, with a quality resembling footage taken by a high-end camera.
• Kling: Generates realistic videos, often incorporating dynamic, moving shots that simulate footage captured with a moving camera.
this section cite: []

Section: P LEADERBOARD
Tab. 15 presents the leaderboard performance on both human evaluation and our designed CAP.
The performance shows a highly consistency and only difference is that for proprietary models, Sora ranked number 2 while Kling ranked number 2 in CAP. Through further analysis, we found that Kling's more cinematic style can bias CAP toward higher scores despite only moderate physical correctness, whereas our human annotators-who were instructed to focus strictly on physics plausibility-penalize these artifacts, causing Sora to rank higher under human evaluation.    We found that generation models tend to fail when the prompt is relatively complex, such as when containing multiple objects.
this section cite: []

Section: R INFLUENCE OF MULTIPLE OBJECTS
We noticed that generation models often fail when the prompt is more complex, such as when containing multiple objects. We show examples of Sora failing when given prompts with multiple objects in Figure 20.
this section cite: []

Section: S DATASET STRUCTURE
The dataset is organized into three main categories of physics phenomena: Fundamental Physics, Composite Physics, and Anti-Physics. Each category is divided into subcategories, which are further broken down into five sub-subcategories, as summarized below: We publicly released a comprehensive dataset that includes the core assets of PhyWorldBench, the 1,050 example json file, the evaluation standards, and the physics categories and subcategories, used to assess physical realism in text-to-video models.
this section cite: []

Section: References
Ref_id:b0 Title: Cosmos world foundation model platform for physical ai Year: (2025)
Ref_id:b1 Title: Evaluating physical commonsense for video generation Year: (2024)
Ref_id:b2 Title: Videophy-2: A challenging action-centric physical commonsense evaluation in video generation Year: (2025)
Ref_id:b3 Title: A family of highly capable multimodal models Year: (2024)
Ref_id:b4 Title:  Year: (2002)
Ref_id:b5 Title: T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation Year: (2025)
Ref_id:b6 Title: Ltx-video: Realtime video latent diffusion Year: (2024)
Ref_id:b7 Title:  Year: (2024)
Ref_id:b8 Title: Fundamentals of physics Year: (2013)
Ref_id:b9 Title: Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation Year: (2024)
Ref_id:b10 Title: Cogvideo: Large-scale pretraining for text-to-video generation via transformers Year: (2022)
Ref_id:b11 Title: Vbench: Comprehensive benchmark suite for video generative models Year: (2024)
Ref_id:b12 Title:  Year: (2024)
Ref_id:b13 Title: Hunyuanvideo: A systematic framework for large video generative models Year: (2024)
Ref_id:b14 Title: Evaluation of text-to-video generation models: A dynamics perspective Year: (2024)
Ref_id:b15 Title: Open-sora plan: Open-source large video generation model Year: (2024)
Ref_id:b16 Title: Evalcrafter: Benchmarking and evaluating large video generation models Year: (2024)
Ref_id:b17 Title: Generative AI platform specializing in 3D content and photorealistic modeling Year: (2024)
Ref_id:b18 Title:  Year: (2025)
Ref_id:b19 Title: Towards world simulator: Crafting physical commonsense-based benchmark for video generation Year: (2024)
Ref_id:b20 Title: Physics-iq: Evaluating physical understanding in generative video models Year: (2025)
Ref_id:b21 Title: OpenAI's Multimodal Agent Year: ()
Ref_id:b22 Title:  Year: (2024)
Ref_id:b23 Title: Open-sora 2.0: Training a commercial-level video generation model in $200k Year: (2025)
Ref_id:b24 Title: Generative AI platform for creating video and visual content Year: (2024)
Ref_id:b25 Title: Platform for AI-powered video editing and generative media creation Year: (2024)
Ref_id:b26 Title: Improved techniques for training gans Year: (2016)
Ref_id:b27 Title: T2vcompbench: A comprehensive benchmark for compositional text-to-video generation Year: (2024)
Ref_id:b28 Title: Towards accurate generative models of video: A new metric & challenges Year: (2018)
Ref_id:b29 Title: World simulator assistant for physics-aware text-to-video generation Year: (2025)
Ref_id:b30 Title: Lavie: High-quality video generation with cascaded latent diffusion models Year: (2023)
Ref_id:b31 Title: Open and advanced large-scale video generative models Year: (2025)
Ref_id:b32 Title: Chain of thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b33 Title: Newtongen: Physics-consistent and controllable text-to-video generation via neural newtonian dynamics Year: (2025)
Ref_id:b34 Title: Morpheus: Benchmarking physical reasoning of video generative models with real physical experiments Year: (2025)
Ref_id:b35 Title: Videorepa: Learning physics for video generation through relational alignment with foundation models Year: (2025)
