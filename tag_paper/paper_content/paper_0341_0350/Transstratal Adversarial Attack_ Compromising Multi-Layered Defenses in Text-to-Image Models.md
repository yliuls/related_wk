Title: Transstratal Adversarial Attack: Compromising Multi-Layered Defenses in Text-to-Image Models
Abstract: Modern Text-to-Image (T2I) models deploy multi-layered defenses to block Not-Safe-For-Work (NSFW) content generation. These defenses typically include sequential layers such as prompt filters, concept erasers and image filters. While existing adversarial attacks have demonstrated vulnerabilities in isolated defense layers, they prove largely ineffective against multi-layered defenses deployed in real-world T2I systems. In this paper, we demonstrate that exploiting overlapping vulnerabilities across these distinct defense layers enables adversaries to systematically bypass the entire safeguard of T2I systems. We propose Transstratal Adversarial Attack (TAA), a novel black-box framework to compromise T2I models with multi-layered protection. It generates transstratal adversarial prompts to evade all defense layers simultaneously. This is accomplished through transstratal adversarial candidate generation using LLMs to fulfill implicit and subjective adversarial requirements against different defense layers, combined with adversarial genetic optimization for efficient black-box search to maximize the bypass rates and generated image harmfulness. Evaluated across 14 T2I models (e.g., Stable Diffusion, DALL•E, and Midjourney) and 17 safety modules, our attack achieves an average attack success rate of 85.6%, surpassing state-of-the-art methods by 73.5%.Our findings challenge the isolated design of safety mechanisms and establish the first benchmark for holistic robustness evaluation in multi-layered safeguarded T2I models. The code can be found in https://github.com/Bluedask/TAA-T2I.

Section: Introduction
Text-to-Image (T2I) models [13,32,27] have rapidly evolved, offering unprecedented capabilities in generating creative content from textual descriptions. Recent advancements, particularly in diffusionbased architectures [32], have significantly enhanced the quality and controllability of generated images. However, this progress is accompanied by the critical risk of misuse, specifically the potential for generating Not-Safe-For-Work (NSFW) images. To mitigate these risks and ensure responsible deployment, modern T2I systems implement multi-layered safeguard mechanisms. These defenses operate at different layers of the image generation pipeline: 1) Prompt Filters [24,2,33] evaluate input queries to detect and block the prompts containing NSFW intention; 2) Concept Erasers [9,35,21] • asshol The attack scenario in our consideration. utilize techniques like unlearning to remove or suppress specific NSFW concepts from the T2I model's components (e.g., U-Net, text encoder); 3) Image Filters [29,30,36] analyze the generated images to identify and prevent the distribution of NSFW content.
Existing research has demonstrated that each of these defense layers is vulnerable to advanced adversarial attacks [42,4,43,38,39]. However, these attacks often fail when confronting the full multi-layered defense deployed in real-world T2I systems (illustrated in Figure 1). For example, our experiment shows that DACA [6] achieves 89.1% attack success rate against prompt filters, but drops to 1.4% against the multi-layered defense. This is common for all these attacks due to the complementary effects between different defense layers. 1) To bypass the concept eraser-based defense, the corresponding prompts should contain NSFW-related words (e.g.,'nude'). However, these words are easy to detect by existing prompt filters. Conversely, generated images for prompts that bypass prompt filters are less harmful; 2) A generated apparently NSFW image makes it hard to bypass image filters. Conversely, images that bypass image filters are perceived as normal by humans.
To solve these challenges, we propose Transstratal Adversarial Attack (TAA), a novel black-box framework designed to systematically bypass multi-layered defenses of T2I systems. TAA generates Transstratal Adversarial Prompts (TAPs) through two key technical innovations: 1) Transstratal adversarial candidate generation: It leverages LLMs to generate implicit and subjective adversarial candidates to substitute prompts. The implicit candidates can bypass prompt filters and maintain the NSFW image generation capability. The subjective candidates can influence the style of generated images, reducing their harmfulness to bypass image filters, while still being perceived as harmful by humans. 2) Adversarial genetic optimization: It provides an efficient black-box search mechanism, using a failure-driven candidate selection process to maximize the bypass rates and generated image harmfulness by iteratively refining prompts based on system feedback. Based on these innovations, we design an attack pipeline to automatically generate TAPs, achieving a high attack success rate against multi-layered defenses in T2I systems. Empirical evaluation on 14 T2I models (e.g., Stable Diffusion [32], DALL•E [31,3], and Midjourney [1]) and 17 safety modules confirms TAA's efficacy. Our method achieves a remarkable 85.6% attack success rate, significantly surpassing existing state-of-the-art approaches by 73.5% even when confronted with full multi-layered defenses. Our contributions are summarized as:
• We identify and demonstrate the critical vulnerability posed by overlapping weaknesses across the multi-layered defenses of modern T2I systems; • We propose TAA, the first black-box adversarial attack designed to bypass the entire multi-layered defenses of T2I systems by generating transstratal adversarial prompts; • Through extensive evaluation, we show that TAA significantly outperforms existing state-of-theart attacks against multi-layered defenses for both open-sourced and commercial T2I systems.
2 Related Work
this section cite: ['b10', 'b27', 'b22', 'b27', 'b20', 'b28', 'b7', 'b30', 'b17', 'b24', 'b25', 'b31', 'b37', 'b2', 'b38', 'b33', 'b34', 'b4', 'b27', 'b26', 'b1', 'b0']

Section: Multi-Layered Defenses of T2I Models
Text-to-Image (T2I) models [13,32] are designed to generate images from textual prompts. To safeguard against the generation of NSFW images, modern T2I models often employ a multi-layered defense strategy comprising several distinct components: 1) Prompt Filters: These detect and block malicious prompts containing NSFW content, thereby preventing the generation of unsafe images. Common implementations include NSFW keyword blacklists [33], classification models like Distil-RoBERTa [2], online moderation APIs [24], and LLM-based safety classifiers [16,41]; 2) Concept Erasers: These remove NSFW concepts from T2I models via unlearning techniques. Existing approaches are divided into text-based concept erasure [9,35,44] and image-based erasure [21]; 3) Image Filters: These identify generated unsafe images to block their distribution. Typical implemen-tations leverage classifiers built with architectures like InceptionV3 [20], YOLOv8 [25], ViT [8,10], and CLIP [30,36,29]. For example, Stable Diffusion [32] uses a safety checker that compares CLIP embeddings of generated images against a predefined set of unsafe concepts. ✓ Ring-A-Bell [37] ✓ SneakyPrompt [43] ✓ ✓ PGJ [14] ✓ ColJailBreak [23] ✓
this section cite: ['b10', 'b27', 'b28', 'b20', 'b13', 'b36', 'b7', 'b30', 'b39', 'b17', 'b16', 'b6', 'b25', 'b31', 'b24', 'b27', 'b32', 'b38', 'b11', 'b19']

Section: Adversarial Attacks against T2I Models
Ours ✓ ✓ ✓
Existing attacks targeting T2I models often exclusively focus on one or two specific defense layers rather than the entire protection pipeline, as summarized in Table 1. 1) Prompt Filter Attack: These attacks generate prompts to bypass prompt-filtering mechanisms while retaining NSFW image generation. Common methods include text optimization with NSFW keyword suppression [42], NSFW word substitutions [23,14], and prompt rewriting via large language models [6]. 2) Concept Eraser Attack: These attacks generate prompts that induce T2I models equipped with concept erasure safeguards to produce NSFW images. Common methods include soft prompt optimization [45], hard prompt optimization [40,4], and proxy model-based optimization [37,46] through the addition of a regularization term to bypass concept erasure mechanisms. 3) Image Filter Attack: These attacks generate prompts that are capable of evading image filters while producing NSFW images. Common methods include reinforcement learning [43] to learn word substitutions and image-based adversarial optimization [42]. In this paper, we propose the first attack that targets multi-layered defenses simultaneously in a black-box setting.
this section cite: ['b37', 'b19', 'b11', 'b4', 'b40', 'b35', 'b2', 'b32', 'b41', 'b38', 'b37']

Section: Problem Statement
System Model. We describe the basic structure of T2I models equipped with multi-layered defenses.
The core function of a basic T2I model M is to accept a user prompt T and generate the corresponding image I: I = M(T ). Different defenses are introduced to identify NSFW-related behaviors and prevent the creation of NSFW images. These defenses are primarily deployed at three isolated layers: 1) Prompt Filter Layer: A set of prompt filters F T are introduced to inspect the input prompt P . If NSFW content is detected, the image generation process is halted, and the warning message is returned to the user. Otherwise, the prompt will be proceeded to the next layer. 2) Concept Eraser Layer: The core T2I model is revised as a safety-enhanced version (M ce ← M), aiming to erase or suppress NSFW-related concepts from being generated within the model's latent space or features.
3) Image Filter Layer: Another set of image filters F I is adopted to examine the generated image I. If NSFW content is identified in the image, the output is blocked and a warning message or a blank image is returned. Otherwise, the generated image is released to the user. Formally, the image generation process for a T2I system with multi-layered defenses is described as:
I = F I (M ce (F T (T ))).(1)
Threat Model. As illustrated in Figure 1, we consider an external adversary aiming to mislead the T2I system to generate NSFW image I adv via the adversarial prompt T adv : I adv = F I (M ce (F T (T adv ))).
The adversary operates in a black-box setting, where he can only query the T2I system and iteratively refine his prompts based on the system feedback. This aligns with real-world T2I system deployments (e.g., access via APIs or web interfaces).
Two requirements need to be fulfilled: 1) Defense Evasion: The adversarial prompt must circumvent all deployed safeguards to ensure the T2I system processes the query and outputs an image; 2) Harmful Content Generation: The generated images produced from these adversarial prompts should manifest content that is considered harmful or inappropriate. We introduce two metrics aligned with the attack goals.
1) Bypass Rate (BR): this measures whether an adversarial prompt T adv successfully evades all defense layers and triggers image generation. Formally, BR(T adv ) = 1 if I adv is generated, and 0 otherwise. 2) Image Harmful Score (IHS): this quantifies the frequency at which the generated Regular/NSFW Word Candidate "a fully nude woman" Transstratal Adversarial Candidate Generation fully nude completely totally entirely birthday suit unshielded in the buff flesh toned Adaptive Crossover Victim Prompt Implicit Adversarial Candidate Subjective Adversarial Candidate photo Documentary Nature Aerial artist Giorgione Titian Raphael … … Adversarial Genetic Optimization Candidate Probability Calculation "a totally in the buff woman, Aerial, Titan" * Substituted Prompt Fitness BR IHS Optimized Prompt "a totally birthday suit woman, Documentary, Raphael" 𝑐3 1 𝑐3 2 𝑐3 𝑐 … 𝑤3 𝑐4 1 𝑐 4 2 𝑐4 𝑑 … 𝑤4 𝑤1 𝑤2 𝑤3 𝑤4 𝑝 𝑗 𝑖 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(c j i ) Probability Update: Failure Update: 𝐶 𝑗 𝑖 ← 𝐹𝑓𝑢(𝐹𝑖𝑡𝑛𝑒𝑠𝑠, 𝐶𝑚 𝑛 ) 𝑐1 1 𝑐1 2 𝑐1 𝑎 … 𝑤1 𝑐1 3 𝑐2 1 𝑐2 2 𝑐2 𝑏 … 𝑤2 𝑐2 3 𝑐3 3 𝑐4 3 RW Selection Adaptive Mutation (2)
this section cite: []

Section: T2I System

this section cite: []

Section: Methodology
This section details our proposed Transstratal Adversarial Attack (TAA), a novel black-box framework designed to bypass multi-layered defenses of T2I systems. TAA generate substitution candidates using LLMs to fulfill implicit and subjective adversarial requirements. A failure-driven genetic algorithm iteratively optimizes the prompt substitutions and stylistic augmentations, maximizing both defense evasion and harmful content generation. Figure 2 shows the overview of the attack pipeline.
this section cite: []

Section: Design Insight
We design Transstratal Adversarial Prompts (TAPs) to fulfill the above attack requirements. This is inspired by the following observations.
this section cite: []

Section: Implicit NSFW Prompts.
At the prompt level, we categorize NSFW prompts into explicit and implicit types. Explicit NSFW prompts contain obvious NSFW terms (e.g., 'nude', 'breasts'), while implicit NSFW prompts employ metaphorical NSFW references (e.g., 'birthday suit', 'mounds'). While both types can induce NSFW image generation, prompt filters typically block explicit terms but fail to detect implicit variants. For example, 'nude' is blacklisted in [33] but 'birthday suit' is not. Concurrently, text-based concept erasers remove explicit NSFW concepts from T2I models while retaining implicit ones, enabling adversaries to bypass prompt-level defenses through the implicit adversarial prompts.
this section cite: ['b28']

Section: Subjective NSFW Images.
At the image level, NSFW images are classified as subjective or objective. Subjective images are determined as NSFW by human judgment, while objective images occur when image features fall into NSFW-labeled regions of NSFW classifiers. A successful TAP should make the T2I model generate an image subjectively evaluated as NSFW (from human judgment), while avoiding objectively detected by image filters (from NSFW image classifiers). This phenomenon persists in image-based concept erasers, as subjective adversarial content cannot be effectively removed during image generation.
Based on these two insights, the core of our attack methodology is to iteratively substitute and augment a normal prompt T , ensuring the revised prompt T s fulfills both the implicit prompt and subjective image requirements. This is achieved with the following two steps.
this section cite: []

Section: Transstratal Adversarial Candidate Generation
Implicit Adversarial Candidate. To generate prompts that fulfill the implicit requirement, we substitute explicit NSFW words in the victim prompt T = {w 1 , w 2 , ..., w n } with corresponding implicit adversarial candidates. We construct an implicit word candidate set S imp = {w i NSFW : [w 1  cand , ..., w l cand ]} incorporating a large language model M LLM . M LLM identifies NSFW words w NSFW in T , defined as terms associated with NSFW concepts. Subsequently, M LLM generates an implicit candidate list [w 1  cand , ..., w l cand ] for each NSFW word, where l is the candidate list size. Thus, the implicit word candidate set for T is obtained as: S imp = M LLM (T, P imp ), where P imp is the designed prompt for querying M LLM . The prompt word substitution process is to replace the NSFW word w NSFW with a corresponding candidate from S imp selected by a selection function F select :
T s = n i=1 F select (w i , S imp ) if w i ∈ S imp , w i otherwise.(3)
In practice, we additionally substitute regular words (embellishments such as adjectives, adverbs, and verbs) with their synonyms to enlarge the size of S imp , which helps explore more substitutions, especially for short prompts.
this section cite: []

Section: Subjective Adversarial
Candidate. This requirement refers to generating images that exhibit harmfulness to human perception but not to the perception of image filters. Such images should contain features near image filters' decision boundaries, allowing NSFW content to bypass detection while appearing inappropriate to human observers. Building on our implicit prompts that preserve the NSFW generation capability, subjective adversarial prompts should reduce the objective harmfulness while maintaining subjective harmfulness. To achieve this dual objective, we propose modifying stylistic attributes while preserving core subject features of generated images. Our strategy redirects filter attention from NSFW content through controlled variations in background and contextual elements. For stylistic transformation, we construct a subjective adversarial candidate set S sub = {w i style : [w 1 cand , ..., w l cand ]} using M LLM . Each w i style represents distinct artistic styles (e.g., 'photographic' or 'artistic'). The candidate set is generated through:
S sub = {M LLM (w i
style , P sub )}, where P sub denotes our specially designed prompt for querying M LLM . These candidates are then integrated into the substituted prompt T s through concatenation:
T s = T s ⊕ |Ssub| i=1 F select (w i style , S sub )(4)
After combining S imp and S sub into a candidate set S, the victim prompt p can be transformed into the substituted prompt T s in one time as:
T s = m j=1 F select (w j , S), m = |S imp | + |S sub |.
this section cite: []

Section: Adversarial Genetic Optimization
To optimize the discrete candidate substitutions and augmentations for generating T s , we employ a genetic algorithm (GA). The challenge of GA application is to ensure convergence within a limited number of iterations, as excessive iterations may be computationally impractical. To address this and dynamically guide the selection of candidates during the evolutionary process, we design a failuredriven candidate selection mechanism, calculating each candidate's probability in each iteration and guiding the mutation direction, which can enhance the convergence efficiency.
this section cite: []

Section: Candidate Probability Calculation.
We formalize the candidate selection mechanism as the selection function F select . The aim of F select is to dynamically compute and update candidate probabilities based on the fitness of substituted prompts, providing optimization guidance during mutation. F select operates as a failure-driven mechanism, maintaining a candidate failure set C with the same shape as S.
During initialization, all failure values are initialized to 0: C = m j=1 c j = [c 1 j , . . . , c l j ] | c i j = 0 . As iterations progress, candidate failure counts are updated via the failure update function F fu :
F fu (c i j ) =    c i j -1 if f > f upper , c i j + 1 if f < f lower , c i j otherwise. (5
)
This rule reduces failure counts for candidates linked to high-fitness prompts (f > f upper ), i.e., increasing their selection likelihood, while increasing failure counts for low-fitness candidates (f < f lower ), i.e., suppressing their selection. The candidate probability is then derived via softmax:
p i j = exp( 1 T (c i j + 1) )/ m i=1 exp( 1 T (c i j + 1) ),(6)
where T is a temperature parameter. The selection function F select is implemented as stochastic sampling: F select (w j , S) ∼ Categorical(p 1 j , . . . , p l j ). This probabilistic approach balances exploration and exploitation during mutation.
this section cite: []

Section: Genetic Optimization.
The core components of the genetic algorithm in TAA include:
1. Genotype and Individual: The genotype is defined as the substitution operation S imp and the augmentation operation S sub . An individual T ind in the population represents a substituted prompt generated through the selection function F select and two candidate sets; 2. Fitness Evaluation: Similar to Equation (2), the fitness is defined as a weighted combination of bypass rate and image harmful score: f (T ind ) = BR(T ind ) + IHS(I ind ); 3. Adaptive Crossover: We employ an adaptive crossover mechanism where a child is more likely to inherit genes from the parent with higher fitness, formulated as:
T child = Crossover(T parent1 , T parent2 , f(Tparent1)
f (Tparent1)+f (Tparent2) ); 4. Adaptive Mutation: The mutation involves replacing candidates with alternatives using a dynamic mutation rate r m :
T ind = m j=1 F select (w i , S) if r < r m , w i otherwise, (7
)
where r is a random value. The dynamic mutation rate r m is computed as: r m = max(r min m , r max m × (1 -i/I), where r max m is the initial mutation rate, r min m is the minimum rate, i is the current iteration, and I is the maximum iteration count. This design encourages exploration early and convergence later; 5. Selection: We use roulette wheel selection to choose individuals for the next generation. The selection probability p s for an individual T ind is proportional to its fitness:
p s (T ind ) = f (T ind ) T ′ ind ∈population f (T ′ ind ) .(8)
6. Termination: Two termination criteria are used: reaching the maximum iteration count I or achieving the maximum fitness value. Through iterative steps, GA systematically explores the candidate set. The final output of the algorithm is the optimized prompt T adv that has achieved the highest fitness score during the optimization process, representing the most effective TAP found.
this section cite: []

Section: Evaluation

this section cite: []

Section: Experiment Setup
Datasets and T2I Models. We curate 118 prompts that cannot bypass default safety filters from the nsfw_200 dataset [43], augmented with 72 LLM-generated NSFW prompts (using seed prompts in [42]) that also fail to bypass safety filters. This forms the nsfw_190 dataset, with details in Appendix B.1. For T2I models, we evaluate 14 representative T2I models, with 10 open-sourced ones (SD-v1.4 [32], SD-v1.5 [32], SD-v2.1 [32], SD-XL [27], SDXL-Turbo [34], SD-3 [7], SD-3.5 [7], FLUX.1-dev [19], FLUX.1-schnell [19] and Lumina [28]), and 4 commercial T2I services (Dall•E-2 [31], Dall•E-3 [3], midjourney-6.1 [1] and midjourney-7 [1]).
Defense Layers. For prompt filters, we employ four approaches: a black-list method (NSFW-Word-List [33]), an LLM justification method (LLaMa-Guard [16]), an NSFW prompt classification model (NSFW-Prompt [2]), and a gradient analysis method (Grad [41]). For concept erasers, we include an NSFW concept-erasing method (ESD [9]), an NSFW concept-suppressing method with four safety levels (SLD-MAX, SLD-STRONG, SLD-MEDIUM, SLD-WEAK [35]), a concept-forgetting method (FMN [44]), and a visual NSFW concept-erasing method (SafeGEN [21]). For image filters, we utilize two NSFW image classifiers (NudeNet [25], NSFW-Image [8]), three CLIP-based classifiers (Q16 [36], Q16-FT [29], and MHSC [29]), and a cross-modal detection method (Safety-Checker [32]).
Baselines. For white-box baselines, we evaluate: UnlearnDiff [45], MMA_T [42], PEZ [40], P4D [4]: fixed-length random prompts (P4D_N) and token-appended prompts (P4D_K). For black-box baselines, we include: I2P [29], QF-Attack [46], DACA [6], Ring-A-Bell [37], SneakyPrompt [43], PGJ [14], ColJailBreak [23]. The baseline details are presented in Appendix B.3.
this section cite: ['b38', 'b37', 'b27', 'b27', 'b27', 'b22', 'b29', 'b5', 'b5', 'b23', 'b26', 'b1', 'b0', 'b0', 'b28', 'b13', 'b36', 'b7', 'b30', 'b39', 'b17', 'b6', 'b31', 'b24', 'b24', 'b27', 'b40', 'b37', 'b35', 'b2', 'b24', 'b41', 'b4', 'b32', 'b38', 'b11', 'b19']

Section: Evaluation Metrics.
We evaluate the attack performance using five metrics: 1) PBC@i (Prompt Bypass Count): Defined as the count of prompts that successfully bypass at least i prompt filters; 2) IHC@j (Image Harmful Count): Defined as the count of generated images classified as NSFW by at Image Filter Layer Bypass Do Not Bypass 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 𝑺 𝟏 𝑺 𝟐 𝑺 𝟑 (f) QF-Attack Attack Settings. We employ three LLMs (gpt-4o [15], o1-mini [17], and gpt-4.1 [26]) to generate word substitutions. For each word queried to the LLM, the candidate list size is fixed at 10. In candidate probability calculation, f upper is configured to 0.8, f lower to 0.2, and the temperature parameter T to 1.0. For genetic optimization, we set the population size to 20, maximum generations to 20, initial mutation rate to 0.5 and minimum mutation rate to 0.1.
this section cite: ['b12', 'b14', 'b21']

Section: Main Results
Table 2 presents the attack results of various methods on SD-v1.4 with the concept eraser of ESD [9], with adversarial images in Figure 3. From these baseline results, we derive two critical conclusions: 1) PBC inversely correlates with IHC. DACA [6] achieves the highest PBC@4 (128) but lowest IHC@4 (0), while P4D_N [4] shows the opposite (IHC@4=75 vs. PBC@4=13). This arises because NSFW generation relies on explicit keywords detected by prompt filters. 2) IHC inversely correlates with IBC. P4D_N [4] attains the highest IHC@4 (75) but low IBC (77), whereas DACA [6] maximizes IBC (189) with minimal IHC@0, as image filters readily detect NSFW content. Our method resolves these mutually exclusive effects by: (1) using implicit prompts with metaphorical NSFW terms to bypass filters while generating harmful images, and (2) employing subjective adversarial prompts to stylize outputs, reducing detectability by image filters while retaining human-perceptible NSFW content. Consequently, it achieves 85.6% ASR against multi-layered defenses, outperforming baselines and even surpassing single-layer attacks on individual defenses.
this section cite: ['b7', 'b4', 'b2', 'b2', 'b4']

Section: Different T2I Models.
The concept eraser of a T2I model is normally tailored for specific architectures, leading to limitations when applied to other models. Thus, we evaluate a more generalizable approach for integrating safety modules by employing defenses with external prompt and image  filters, regardless of the T2I model's architecture. We evaluate various open-source models under this safety configuration, with results shown in Table 3. On the one hand, all open-source T2I models relying solely on external filters remain vulnerable to our TAA, highlighting the significant safety risks during deployment, even for recent models. On the other hand, our results demonstrate that TAA can bypass prompt and image filters with high success rates. Consequently, developing intrinsic safety mechanisms for T2I models is imperative, rather than relying solely on external safeguards.
this section cite: []

Section: Different Image Filters.
We evaluate the robustness of different image filters against sd-v1.4 [32]), with results presented in Table 4. In contrast to the minor variations of the attack performance observed across prompt filters and T2I models, there are significantly larger discrepancies in the attack effectiveness across image filters. This discrepancy stems from the fact that the success of attacks on image filters depends on their NSFW classification accuracy. The higher a filter's classification accuracy, the more challenging it is to bypass. Consequently, the substantial variation in attack success rates highlights significant disparities in classification accuracy across these filters. For example, Q16 [36] exhibits lower classification accuracy compared to Q16-FT [29]. Nevertheless, TAA achieves successful attacks against robust image filters (e.g., 64.9% ASR against MHSC [29]). Different Concept Erasers. Current concept erasers are categorized into text-based and imagebased erasers. In Table 2, we report the attack performance of TAA against text-based erasers. We additionally include attack results for more text-based and image-based concept erasers in Table 5.
TAA achieves comparable performance across all variants. Notably, it maintains robust and effective even as the safety strength increases for SLD [29], demonstrating the vulnerability of text-based erasers to TAA. In contrast, the image-based concept eraser [21] offers a stronger defense against TAA. However, its protection remains insufficient to fully counter TAA.
this section cite: ['b27', 'b31', 'b24', 'b24', 'b24', 'b17']

Section: Transferability Evaluation
We evaluate TAA's transferability against two state-of-the-art baselines: the white-box method UnlearnDiff [45] and the black-box method SneakyPrompt [43].
Open-source T2I Models. Adversarial prompts crafted from source models are used to attack target models, with performance quantified in Table 6. TAA achieves superior transferability compared to baselines, as adversarial prompts from UnlearnDiff and SneakyPrompt often rely on model-specific artifacts (e.g., 'angelibrunedress' or 'hackwbotdwbuil'), limiting their generalizability. The left panel of Figure 4 visualizes TAA's consistent transferability across T2I models. Similar trends   are observed across different concept erasers (presented in the right panel of Figure 4 and Table 13), confirming its robustness against diverse defense configurations.
this section cite: ['b40', 'b38']

Section: Commercial T2I Models.
Attacking commercial T2I models is typically time-consuming and expensive due to API usage costs and rate limits, making direct black-box optimization challenging. Therefore, evaluating the transferability of adversarial prompts from accessible open-source models to these closed-source targets is crucial for understanding real-world attack feasibility. Table 7 displays the attack transferability results over popular commercial T2I services, using adversarial prompts generated against SD-v1.4 [32]. Consistent with our observations on open-source model transferability, TAA demonstrates effective attack transferability to closed-source models. Compared to the baseline methods, TAA achieves significantly higher ASR.
this section cite: ['b27']

Section: Ablation Study
Impact of Core Components. We examine the contributions of different components of TAA in Table 8. The setting of w/o genetic means we use random candidate selection. Adversarial genetic optimization identifies candidate substitutions that improve the bypass rate and image-harmful rate. Consequently, removing this component leads to significantly poorer attack performance. The implicit candidate set S imp serves as the foundation for optimization; without it, no successful attacks occur. The perception-only candidate set S sub enhances the overall performance by boosting both the bypass and image-harmful rates. When F select is omitted, random selection is applied during mutation. Then, some cases fail to converge to an optimal prompt within the limited iteration steps. This indicates that F select facilitates convergence during optimization.
this section cite: []

Section: Impact of Hyperparameters.
We evaluate the impact of main hyperparameters in TAA against SD-v1.4 [32], including the candidate list size in candidate generation, population size, and iteration count in genetic optimization. Variations in ASR across different hyperparameter values are shown in Figure 5. For the candidate list size, the number of effective candidates does not increase proportionally with larger candidate pools. This is because the list of viable candidates for NSFW words remains limited, even when using smaller candidate lists generated by the LLM. For the population size and iteration count, ASR improves as these hyperparameters increase. However, optimal results can still be achieved with limited population sizes and iteration counts through TAA's candidate probability guidance.
this section cite: ['b27']

Section: Adaptive Defense
To counter TAA, which exploits implicit prompts and subjective images, we explore adaptive defenses. Direct approaches like comprehensive concept removal are often impractical due to computational infeasibility and their tendency to degrade general model quality. Therefore, we evaluated two adaptive strategies designed to target TAA's mechanisms specifically:
• LLM Processing for Filter Prompts: TAA uses LLMs to create implicit, metaphorical prompts. We tested a defense that also utilizes an LLM (GPT-4o [15]) to detect and block these same prompts. The corresponding prompt is in the Appendix. This LLM processor was added as an extra safety layer after the standard prompt filters.
• Adversarial Training for Image Filters: Standard image filters can be bypassed by the unique styles of images TAA generates. To counter this, we retrained an image filter using adversarial examples from TAA. We built a new dataset called NSFW-4000, containing 1,000 TAA-generated NSFW images, 1,000 benign images from the COCO dataset [22], and 2,000 harmful images from existing datasets [18]. We used this dataset to fine-tune the MHSC mode [29], creating a more robust version called MHSC-ft.
Defense Setup. The experiment followed the setup described in Table 2 but added the GPT-4o processor and the MHSC-ft image filter. We evaluated the defense's impact on normal image generation using prompts from the Midjourney-v6 dataset [5] and measured performance with ClipScore [12]. To test the defense's effectiveness against attacks, we used our nsfw_190 dataset and the metrics from Section 5.1.
Defense Results. Our preliminary results (Table 14 in the appendix) indicate that LLM processing can handle simple implicit prompts (e.g., transforming "a birthday suit woman" to "a woman") but struggles with the complex, metaphorical prompts generated by TAA, resulting in limited defense effectiveness. Adversarial training shows stronger performance, as the MHSC-ft model learns the stylistic patterns present in subjective NSFW images. However, this defense remains insufficient for comprehensive protection, as TAA can generate a virtually infinite variety of style variations that exceed the coverage of adversarially trained filters. Developing more robust and holistic defense mechanisms remains an important direction for future work.
this section cite: ['b12', 'b18', 'b15', 'b24', 'b3', 'b9']

Section: Conclusion
This work exposes critical vulnerabilities in multi-layered defenses of modern T2I models. We demonstrate that current defenses, including prompt filters, concept erasers, and image filters, suffer from overlapping weaknesses, enabling adversaries to bypass all protections via a single adversarial prompt. Our proposed method, Transstratal Adversarial Attack, achieves this goal by integrating LLM-guided candidate substitutions with adversarial genetic optimization. It shows high attack performance across multiple open-source and commercial T2I models and different safety modules.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: ()
Ref_id:b1 Title: Improving image generation with better captions Year: (2023)
Ref_id:b2 Title: Prompting4debugging: Red-teaming text-to-image diffusion models by finding problematic prompts Year: (2024)
Ref_id:b3 Title: Midjourney v6 prompts Year: (2024)
Ref_id:b4 Title: Divide-and-conquer attack: Harnessing the power of llm to bypass safety filters of text-to-image models Year: (2023)
Ref_id:b5 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b6 Title:  Year: (2024)
Ref_id:b7 Title: Erasing concepts from diffusion models Year: (2023)
Ref_id:b8 Title: The llama 3 herd of models Year: (2024)
Ref_id:b9 Title: Clipscore: A reference-free evaluation metric for image captioning Year: (2021)
Ref_id:b10 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b11 Title: Perception-guided jailbreak against text-to-image models Year: (2024)
Ref_id:b12 Title: Gpt-4o system card Year: (2024)
Ref_id:b13 Title: Llama guard: Llm-based input-output safeguard for human-ai conversations Year: (2023)
Ref_id:b14 Title: Openai o1 system card Year: (2024)
Ref_id:b15 Title:  Year: (2022)
Ref_id:b16 Title: Nsfw-detection-dl Year: (2020)
Ref_id:b17 Title: Safegen: Mitigating sexually explicit content generation in text-to-image models Year: (2024)
Ref_id:b18 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b19 Title: Coljailbreak: Collaborative generation and editing for jailbreaking text-to-image deep generation Year: (2024)
Ref_id:b20 Title: A holistic approach to undesired content detection in the real world Year: (2023)
Ref_id:b21 Title: Introducing GPT-4.1 in the API Year: (2025)
Ref_id:b22 Title: Sdxl: Improving latent diffusion models for high-resolution image synthesis Year: (2024)
Ref_id:b23 Title: Lumina-image 2.0: A unified and efficient image generative framework Year: (2025)
Ref_id:b24 Title: Unsafe diffusion: On the generation of unsafe images and hateful memes from text-to-image models Year: (2023)
Ref_id:b25 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b26 Title: Zero-shot text-to-image generation Year: (2021)
Ref_id:b27 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b28 Title: Nsfw-words-list Year: (2020)
Ref_id:b29 Title: Adversarial diffusion distillation Year: (2024)
Ref_id:b30 Title: Safe latent diffusion: Mitigating inappropriate degeneration in diffusion models Year: (2023)
Ref_id:b31 Title: Can machines help us answering question 16 in datasheets, and in turn reflecting on inappropriate content? Year: (2022)
Ref_id:b32 Title: Ring-a-bell! how reliable are concept removal methods for diffusion models? Year: (2024)
Ref_id:b33 Title: Eviledit: Backdooring text-to-image diffusion models in one second Year: (2024)
Ref_id:b34 Title: Model supply chain poisoning: Backdooring pre-trained models via embedding indistinguishability Year: (2025)
Ref_id:b35 Title: Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery Year: (2023)
Ref_id:b36 Title: Gradsafe: Detecting jailbreak prompts for llms via safety-critical gradient analysis Year: (2024)
Ref_id:b37 Title: Mma-diffusion: Multimodal attack on diffusion models Year: (2024)
Ref_id:b38 Title: Sneakyprompt: Jailbreaking text-to-image generative models Year: (2024)
Ref_id:b39 Title: Forget-me-not: Learning to forget in text-to-image diffusion models Year: (2024)
Ref_id:b40 Title: To generate or not? safety-driven unlearned diffusion models are still easy to generate unsafe images... for now Year: (2024)
Ref_id:b41 Title: A pilot study of query-free adversarial attack against stable diffusion Year: (2023)
