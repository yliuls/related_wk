Title: VERITAS: GENERALIZABLE DEEPFAKE DETECTION VIA PATTERN-AWARE REASONING
Abstract: Deepfake detection remains a formidable challenge due to the evolving nature of fake content in real-world scenarios. However, existing benchmarks suffer from severe discrepancies from industrial practice, typically featuring homogeneous training sources and low-quality testing images, which hinder the practical usage of current detectors. To mitigate this gap, we introduce HydraFake, a dataset that contains diversified deepfake techniques and in-the-wild forgeries, along with rigorous training and evaluation protocol, covering unseen model architectures, emerging forgery techniques and novel data domains. Building on this resource, we propose VERITAS, a multi-modal large language model (MLLM) based deepfake detector. Different from vanilla chain-of-thought (CoT), we introduce pattern-aware reasoning that involves critical patterns such as "planning" and "self-reflection" to emulate human forensic process. We further propose a two-stage training pipeline to seamlessly internalize such deepfake reasoning capacities into current MLLMs. Experiments on HydraFake dataset reveal that although previous detectors show great generalization on cross-model scenarios, they fall short on unseen forgeries and data domains. Our VERITAS achieves significant gains across different out-of-domain (OOD) scenarios, and is capable of delivering transparent and faithful detection outputs.* This work was done during the first author's internship at Ant Group.

Section: INTRODUCTION
Recent advances in Generative AI (Esser et al., 2024;Tian et al., 2024) have revolutionized our digital life, unprecedentedly enriching the diversity of content on social media and short-video platforms. Though bringing immense creativity, such techniques also enable highly convincing deepfakes with minimal cost, posing significant security risks to society. Consequently, Deepfake Detection (DFD), which aims at discerning between real and generated facial images, has become a heated research frontier, galvanizing extensive efforts.
However, current detectors mostly follow a standard evaluation, which involves training on one dataset (Rossler et al., 2019) and testing on others (Dolhansky et al., 2019;Li et al., 2020b;Dolhansky et al., 2020;Zi et al., 2020;Zhou et al., 2021). Despite its popularity, this protocol fails to align with practical industrial scenarios, where abundant training samples are available yet significant out-of-distribution (OOD) generalization challenges (e.g., brand-new forgery types and meticulously synthesized facial images) emerge during testing. Such discrepancy severely hinders the practical deployment of current detectors. To mitigate the gap, we construct HydraFake dataset. As shown in Figure 2, we systematically collect and reproduce advanced deepfake methods, covering diversified deepfake techniques and in-the-wild forgeries from social media. To simulate potential challenges in real-world scenarios, we establish a rigorous and holistic evaluation protocol, where the training set consists of abundant samples but is restricted to three basic forgery types, and the Fake Image from unseen domain, generated using Dreamina apps GPT-4o: InternVL3-78B:
<fast> At first glance, the image appears convincingly real, … </fast> <planning> Given the initial impression of realism, a thorough and systematic examination is necessary … </planning> <reasoning> ..., the background is a solid blue, which is common in professional settings but can sometimes be a sign of a controlled or generated environment. … </reasoning> <reflection> Wait a moment-there's another aspect to consider. The badge on the left side of the suit displays text, … difficult to read clearly. In real images, such text would usually be legible or at least discernible, especially when the subject is facing the camera. This ambiguity could be a deliberate attempt to obscure or mislead, a common tactic in AI-generated images to avoid scrutiny of facial features. … </reflection> <conclusion> … potential text obfuscation-suggest it is AI-generated. </conclusion> <answer> fake </answer> Veritas (ours):
This image appears to be artificially generated. … The most prominent is the unusual and overly smooth fabric texture of the suit … The
text on the name badge also lacks proper alignment and realistic detail, … <think> … 1. **Facial Features**: The eyes and mouth appear naturally proportioned and coherent. 3. **Skin Texture and Reflections**: The skin texture appear natural… </think> <answer> real </answer> Veritas (w/o MiPO):
<fast> … </fast> <planning> … </planning> <reasoning> … the skin texture appears unnaturally smooth, lacking high-frequency details such as pores </reasoning> <reflection> But wait--there's another layer of subtlety. The holistic stylization and uniformity, while not violating any explicit physical laws, reflect a lack of organic randomness seen in real human faces … </reflection> … <answer> fake </answer>
Please determine the authenticity of the image Figure 1: Comparison of the detection outputs. InternVL3-78B (Zhu et al., 2025) gets incorrect answer. GPT-4o (Hurst et al., 2024) and our model trained without the proposed MiPO both fail to provide precise explanation. In contrast, our model gives transparent and faithful decision process. evaluation involves hierarchical OOD testing, spanning in-domain, cross-model, cross-forgery and cross-domain scenarios, enabling fine-grained understanding of the model's capacities. As presented in Figure 2 (d), under such rigorous evaluation, current SOTA detectors show great generalization on cross-model deepfakes, but limited abilities in cross-forgery and cross-domain scenarios.
To improve the robustness on unseen forgeries and data domains, we seek to ground the generalization abilities of multi-modal large language models (MLLMs) into deepfake detection. Recent efforts (Huang et al., 2024;Guo et al., 2025b;Peng et al., 2025) have made initial attempts, while they focus on the explainability and the classification is still based on expert vision models. In constrast, we explore to seamlessly internalize MLLMs into deepfake detection through their intrinsic reasoning abilities. However, directly applying deep reasoning faces a critical challenge: current MLLMs are extremely short for deepfake detection (Ren et al., 2025;Tariq et al., 2025). Effective reasoning data is necessary to ground the abilities of base model. To achieve this goal, we must answer two key questions: (1) what kind of reasoning process is helpful to DFD task? and (2) with sufficient data, how can we ensure the model is learning to reason for DFD rather than memorizing?
For the first question, we introduce a pattern-aware reasoning framework. Drawing inspiration from recent studies (Zhao et al., 2025;Muennighoff et al., 2025) that demonstrate critical reasoning patterns greatly elevate the OOD performance of LLMs, we consider the human mindset for deepfake detection: when determining the authenticity of an image, we tend to make a quick judgment based on our first impression (fast judgement), then identify one or two prominent features (reasoning) to draw a conclusion (conclusion). For more challenging samples, we may conduct a layered analysis (planning), and may also engage in more in-depth thinking to support or overturn our initial judgement (self-reflection). Based on this analogy, we extract these five thinking patterns to facilitate logical and holistic reasoning. Table 2 empirically shows the benefits of such pattern-aware reasoning over vanilla Chain-of-Thought (CoT). For the second question, we introduce a two-stage training pipeline consisting of pattern-guided cold-start and pattern-aware exploration, yielding our VERITASfoot_0 model. During cold-start, we employ SFT to internalize thinking patterns. Besides, we introduce a Mixed Preference Optimization (MiPO) strategy that leverages mixed non-preference data and human-annotated preference data to steer the model toward faithful and fine-grained reasoning. As shown in Figure 1, MiPO greatly improves the reasoning quality, mitigating the memorizing behavior. To further facilitate adaptive planning and self-reflection, we propose Pattern-aware Group Relative Policy Optimization (P-GRPO), which shapes reasoning behavior through online sampling and pattern-aware reward mechanism. As a result, VERITAS shows great generalization on unseen forgeries and data domains, providing transparent and precise decision process (Figure 1).
To sum up, our main contributions are:
• Dataset: We introduce HydraFake, a dataset that simulates real-world challenges with hierarchical generalization testing, advancing the evaluation protocol in deepfake detection and helping developers better locate the deficiencies of their detectors.
• Method: We propose a two-stage training pipeline that grounds the capabilities of MLLMs into deepfake detection through pattern-aware reasoning. Our model supports adaptive planning and self-reflection, delivering transparent and human-aligned decision end-to-end.
• Performance: Our VERITAS model achieves significant improvements over state-of-theart detectors on cross-forgery and cross-domain scenarios, and our cold-start model serves as a strong reasoning foundation for further customization.
2 RELATED WORK
this section cite: ['b14', 'b66', 'b54', 'b12', 'b13', 'b110', 'b107', 'b109', 'b34', 'b31', 'b51', 'b53', 'b61', 'b102', 'b48']

Section: DEEPFAKE DETECTION AND DATASETS
Deepfake detection aims to distinguish generated facial images from authentic human faces. Previous efforts have explored spatial-level (Ojha et al., 2023;Yan et al., 2024b;Tan et al., 2024b;Nguyen et al., 2024;Fu et al., 2025;Yan et al., 2024a;c;Yang et al., 2025c), frequency-level (Qian et al., 2020;Tan et al., 2024a;Zhou et al., 2024;Kashiani et al., 2025) and sequence-level (Gu et al., 2021;2022b;a;Yan et al., 2025) approaches, achieving remarkable progress on traditional benchmarks. To train a generalizable detector, some methods attempt to find "bias-free" fake images either through spatial-domain blending (Li et al., 2020a;Shiohara & Yamasaki, 2022;Zhao et al., 2021), frequencydomain blending (Zhou et al., 2024;Kashiani et al., 2025) or feature-level augmentation (Yan et al., 2024b). However, the commonly adopted protocol, i.e., training on FF++ (Rossler et al., 2019) and testing on others (Dolhansky et al., 2019;Li et al., 2020b;Dolhansky et al., 2020;Zi et al., 2020;Zhou et al., 2021), suffers from two problems: (1) the training sources are overly narrow, and (2) the testing data exhibit limited forgery types and low-resolution. Although many timely datasets (Yan et al., 2024a;Zhang et al., 2024b;Li et al., 2025;Huang et al., 2025b;Wang et al., 2025a;Wen et al., 2025a;Xia et al., 2025) have been proposed for AIGC detection, the pace of deepfake detection has lagged behind. As a result, previous methods are biased towards such settings, exhibiting degraded generalization when learning from varying sources or mixed artifacts. To mitigate this problem, we introduce a hierarchical protocol in our HydraFake dataset, aiming to comprehensively reflect the generalization capability of the detectors.
this section cite: ['b50', 'b49', 'b16', 'b52', 'b105', 'b39', 'b18', 'b87', 'b56', 'b103', 'b105', 'b39', 'b54', 'b12', 'b13', 'b110', 'b107', 'b43', 'b76']

Section: MLLMS FOR DEEPFAKE DETECTION
With the proliferation of MLLMs (Liu et al., 2023;Bai et al., 2025;Zhu et al., 2025), recent focus has shifted to explainable deepfake and AIGC detection. However, most methods still rely on small vision models for the final decision. For instance, M2F2-Det (Guo et al., 2025b) determines the authenticity purely based on CLIP models, where LLM is leveraged as a plug-in interpreter.
Similarly, DD-VQA (Zhang et al., 2024a), FFAA (Huang et al., 2024) and VLF-FFD (Peng et al., 2025) develop post-processing system to aggregate embeddings from small vision models. Some methods (He et al., 2025;Sun et al., 2025;Chen et al., 2025b) attempt to directly adopt the outputs from LLMs, e.g., Sun et al. (Sun et al., 2025) construct precise forgery explanations to release the power of MLLMs. Recent methods (Huang et al., 2025a;Xu et al., 2024b;Zhou et al., 2025) also adopt MLLMs and curated datasets for AIGC detection. However, these methods generate post-hoc explanations by first determining the answer. The potential of reasoning abilities for deepfake detection is still underexplored. The most recent methods (Gao et al., 2025;Xia et al., 2025) explore the reasoning for AIGC detection, while neglecting adaptive reasoning patterns and is not tailored for facial forgery. Different from previous methods, we introduce human-like reasoning into deepfake detection, achieving promising improvements and delivering transparent decisions end-to-end.
this section cite: ['b45', 'b2', 'b109', 'b31', 'b51', 'b26', 'b57', 'b57', 'b108', 'b17', 'b76']

Section: HYDRAFAKE DATASET
In this part, we introduce our HydraFake dataset, including the construction process and evaluation protocol. Detailed statistics and information are provided in Appendix A.1.
this section cite: []

Section: DATA COLLECTION
Real Images. As shown in Figure 2 (a), the real images are collected from 8 public datasets, containing both low-resolution (i.e., LFW (Huang et al., 2008), CelebA (Liu et al., 2015), FaceForen-sics++ (FF++) (Rossler et al., 2019), FFIW (Dolhansky et al., 2019)) and high-resolution images (i.e., FFHQ (Karras et al., 2019), VFHQ (Xie et al., 2022), UADFV (Yang et al., 2019) and Cele-bAHQ (Karras et al., 2017)). The collected images are rigorously partitioned for training and testing.
this section cite: ['b30', 'b46', 'b54', 'b12', 'b38', 'b78', 'b89', 'b37']

Section: Fake Images.
The fake images come from three sources:
• Classic deepfake data sampled from FF++ (Rossler et al., 2019) DF40 (Yan et al., 2024d) and FFIW (Dolhansky et al., 2019), which mainly contain face swapping (FS) and face reenactment (FR) forgeries from 10 generative models. The artifacts are mostly localized.
• Public deepfake data sampled from WILD (Bongini et al., 2025), seeprettyface website and Talk-ingHeadBench (Xiong et al., 2025). This contains carefully synthesized faces from 16 popular generators. However, there still exist corner cases such as fresh forgery types.
• Advanced deepfake data where we further reimplemented and crawled 10K deepfake data from 10 advanced generators. Besides traditional deepfake techniques, HydraFake dataset contains Face Restoration (Zhou et al., 2022), Face Relighting (Zhang et al., 2025b), Face Personalization (Jiang et al., 2025;Guo et al., 2024), Generative Face Swapping (Han et al., 2024) and deepfakes from Visual AutoRegressive models (VAR) (Han et al., 2025;Tang et al., 2024). To simulate real-world challenges, we also crawled 1K deepfake images from social media, which include practical deepfakes generated from commercial apps, including GPT-4o (Hurst et al., 2024), Dreamina (team, 2025a) and Hailuo AI (team, 2025b).
Quality Control. For classic deepfake datasets, we only select FF++ (Rossler et al., 2019) and FFIW (Zhou et al., 2021), while not involving DFDC (Dolhansky et al., 2020), DFDCP (Dolhansky et al., 2019) and WDF (Zi et al., 2020) due to their low quality (e.g., unexpected blurring in real images). For our self-constructed deepfake data, we conduct strict quality control, e.g., for face personalization, we use Qwen2.5-VL-72B to tailor sample-specific prompts rather than using template-like prompts as in (Bongini et al., 2025). For face relighting, we generate multiple lighting sources for each identity and manually select high-quality samples. After filtering and balancing, our HydraFake dataset contains 50K real images and 50K fake images.
this section cite: ['b54', 'b12', 'b3', 'b79', 'b106', 'b36', 'b23', 'b25', 'b24', 'b60', 'b34', 'b54', 'b107', 'b13', 'b12', 'b110', 'b3']

Section: EVALUATION PROTOCOL
Training. As shown in Figure 2 (b), the training set contains 48K images. Real images are from 5 subsets, with other 3 subsets left out for testing. Fake images involves 21 subsets while only contains 3 forgery types (i.e., FS, FR and EFG). This is to simulate practical setting, where abundant training images are available but various forgery types and generative models remain unseen.
Evaluation. The evaluation is divided into four distinct levels:
• In-Domain (14K): testing images share the training data source but with different identities.
• Cross-Model (11K): fake images are generated by unseen models under controlled conditions like the template-based textual prompts. This includes SOTA models from recent years (e.g., FLUX1.1-Pro (Black Forest Labs, 2024), Adobe FireFly (Adobe, 2023), Starry AI (AI, 2023)), distinct model architectures (e.g., VAR (Han et al., 2025;Tang et al., 2024) and Video AR model (Sand-AI, 2025)). The real images are from in-domain set but with different identities.
• Cross-Forgery (12K): fake images are generated by unseen manipulation techniques, involving attribute editing, generative face swapping, IP-preserved personalization, face relighting and face restoration. The real images are from in-domain set but with different identities. This split is to evaluate the model's capacity to detect fake images generated by unseen manipulation.
• Cross-Domain (15K): fake images are either generated under controlled conditions or collected from the web, including both unseen forgeries and unseen models. The real images are from unseen datasets (i.e., VFHQ (Xie et al., 2022), UADFV (Yang et al., 2019) and FFIW (Dolhansky et al., 2019)). The images are of different qualities, posing strong challenges.
this section cite: ['b0', 'b24', 'b60', 'b78', 'b89', 'b12']

Section: METHOD
In this section, we detail the two-stage training pipeline of VERITAS, including pattern-guided coldstart and pattern-aware reinforcement learning, as shown in Figure 3.
this section cite: []

Section: PATTERN-GUIDED COLD-START
To internalize thinking patterns for deepfake detection, we first employ a pattern-guided cold-start. Different from common practice, we involve two steps: Supervised Fine-Tuning (SFT) for format injection, and a Mixed Preference Optimization (MiPO) strategy to align the reasoning process.
this section cite: []

Section: SFT Pattern Injection.
Suppose the SFT dataset is denoted as D 1 = {(q, s) i } N1 i=1 , where s is the target output sequence including pattern-aware reasoning and final answer. q denotes input image and user query. The training objective maximizes the likelihood of generating s given input q:
L 1 = -E (q,s)∼D1 T t=1 log π θ (s t | q, s <t ),(1)
where π θ denotes the token distribution from the current model. In the following we introduce the construction process of our training data D 1 .
To minimize human costs, we use MLLMs for automated annotation, similar to recent practices (Huang et al., 2024;Xu et al., 2024b). However, this encounters two challenges in our case: (1)
The MLLMs tend to overlook some subtle artifacts like abnormal optical focusing.
(2) The model prioritizes producing logical paths than to accurately locating artifacts. To mitigate the two issues, we construct a multi-step annotation pipeline. We first manually inspect a subset and summarize a comprehensive artifacts taxonomy (Figure 9
this section cite: ['b31']

Section: MiPO Reasoning Alignment.
To further facilitate human-aligned reasoning, we meticulously curate a mixed preference dataset
D 2 = {(q, s w , s ϕ l ) i } N2 i=1 ∪{(q, s w , s ψ l ) i } N ′ 2 i=1
. Specifically, we collect two types of non-preference data for the fake images: (1) the trajectories where the answer is correct but the reasoning content is not precise or detailed enough (i.e., s ϕ l ). (2) the trajectories where the answer is incorrect (i.e., s ψ l ). s w denotes preferred reasoning traces, which are precisely annotated
Base Model <fast> … </fast> <reasoning> … </reasoning> <conclusion> … <conclusion> <answer> … </answer> <fast> … </fast> <planning> … </planning> <reasoning> … </reasoning> <conclusion> … <conclusion> <reflection> … </reflection> <answer> … </answer> … 𝒟 ! : SFT Set (36K) Pattern Injection … <reasoning> … skin texture lacks minor imperfections, … </reasoning> … <answer> fake </answer> … <reasoning> … </reasoning> … <answer> real </answer> Human Evaluation Reasoning Alignment Cold-started Model l Mixed non-preference: l Human annotated preference: 𝒟 " : MiPO Set (3K) inexact/vague/ recitation-like reasoning process Initial Policy Model Sample incorrect answers … <reasoning> … The left iris is distorted and presents an oddly chaotic white patch., … </reasoning> … <answer> fake </answer> precise & fine-grained reasoning process Human Annotation Semi-Automated Annotation Stage 1: Pattern-Guided Cold-Start Cold-started Model Stage 2: Pattern-Aware Group Relative Policy Optimization (P-GRPO) <fast> … </fast> <reasoning> … </reasoning> <reflection> Observing the image more judiciously, the inconsistencies in focus and the illogical fabric interaction do not align with … </reflection> ... <answer> fake </answer> 𝒟 # : P-GRPO Set to add custom data Flexible for community Online Sampling <fast> … </fast> <reasoning> … The facial structure is symmetric. … </reasoning> … <answer> real </answer> … 𝒐 ! 𝑅 ! 𝑅 " … Pattern Incentivization Veritas Group Computation ℒ ! <fast> … </fast> <reasoning> … </reasoning> <conclusion> … </conclusion> <answer> … </answer> A set of valid formats, e.g.: … <reflection> Observing the image more judiciously, … </reflection> ... Novelty Measurement l Pattern-Aware Reward l Reflection Quality Reward l Format Reward … <answer> real </answer> <planning> … </ planning > … <answer> fake </answer> Ø Reward: helpful planning/reflection Ø Penalize: unnecessary planning/reflection Score 𝒐 " <fast> … </fast> <reasoning> …The lighting on the face is coherent. … </reasoning> … <answer> real </answer> by our human experts. Both s ϕ l and s ψ l are sampled from the outputs of the SFT model, yielding 3K high-quality paired samples for dataset D 2 . Note that the images in D 2 strictly come from the in-domain training set, without introducing any OOD samples. Suppose the SFT model is denoted as π θSFT , the training objective for MiPO is formulated as:
L 2 = -E (q,sw,s l )∼D2 log σ β log π θ (s w |q) π θSFT (s w |q) -β log π θ (s l |q) π θSFT (s l |q) ,(2)
where σ(•) denotes the sigmoid function and β controls the strength that the model deviates from the reference model. As shown in Figure 1, by learning from such mixed rejected traces, our model can perform more precise and fine-grained reasoning compared to pure SFT cold-start.
this section cite: []

Section: PATTERN-AWARE EXPLORATION
After cold-start, the trained model possesses the fundamental reasoning capacities for deepfake detection. However, it still fails on more challenging samples. To mitigate this, we introduce Pattern-Aware GRPO (P-GRPO) to encourage the model to perform comprehensive reasoning and potential self-reflection. Unlike recent approaches (Tu et al., 2025;Xiao et al., 2025) that encourage adaptive reasoning through length reward, we suppose that absolute reasoning length is not critical. Instead, we incentivize appropriate thinking patterns through the pattern-aware reward mechanism.
Suppose the training data for P-GRPO is D 3 = {(q, a) i } N3 i=1 , where a denotes the binary answer. We randomly sampled 9K images from in-domain training set. For a given query q, P-GRPO samples G responses {o 1 , o 2 , ..., o G } using the current policy model π θold . The quality of each response {R 1 , R 2 , ..., R G } is evaluated through reward functions. Suppose the cold-started model π θcold is adopted as reference policy, the training objective is formulated as:
L 3 =-E (q,a)∼D3,{oi} G i=1 ∼π θ old (•|q) 1 G i=1 |o i | G i=1 |oi| t=1 min (r i,t (θ)A i,t , clip(r i,t (θ), 1-ϵ, 1+ϵ)A i,t ) -β ′ D KL [π θ ∥π θ cold ] ,(3)
where
r i,t (θ) = π θ (o i,t | I, o i,<t ) π θold (o i,t | I, o i,<t ) , A i,t = R i -mean({R 1 , . . . , R G }) std({R 1 , . . . , R G }) .(4)
The reward R i for each response is evaluated from three perspectives:
Pattern-aware Reward. Suppose C ∈ {0, 1} represents the correctness of the final answer, with C = 1 denoting the answer is right. P ∈ {0, 1} and R ∈ {0, 1} represents whether the reasoning involves "planning" and "self-reflection", respectively. The pattern-aware reward is defined as:
R pattern =              2.0, if C = 1 ∧ (P = 1 ∨ R = 1), 1.0, if C = 1 ∧ P = 0 ∧ R = 0, 0.0, if C = 0 ∧ P = 0 ∧ R = 0, -0.5, if C = 0 ∧ P = 1 ∧ R = 0, -1.0, if C = 0 ∧ R = 1.
(5) Specifically, we encourage the model to reach correct answers through planning and self-reflection by assigning a larger reward (i.e., 2.0) if they are involved in the reasoning process. However, if these patterns lead to incorrect answers, we impose a penalty for its overthinking. Since self-reflection is a more decisive pattern, we assign a larger penalty (i.e., -1.0) for errors resulting from it.
this section cite: ['b67', 'b22']

Section: Reflection Quality and Format Reward.
To facilitate meaningful self-reflection, we assess the quality of reflection by an external model M: R ref = M(S). The criterion is the originality of the reflection, i.e., whether it introduces new perspectives rather than restating prior discoveries. The model only obtains R ref when the answer is correct. For format reward R fmt , we predefine some combinations of reasoning patterns and set R fmt = 1 when the response conforms to valid formats. Suppose I(•) is the indicator function. The final reward R for each response is defined as:
R = R pattern + λ 1 R ref • I(C = 1) + λ 2 R fmt .(6)
In practice, given that only verifiable answers are required, the training data D 3 can be freely expanded. Our cold-start model serves as a solid reasoning foundation, upon which the community can utilize custom data with P-GRPO to achieve more powerful reasoning model for deepfake detection.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETUP

this section cite: []

Section: State-of-the-Art Methods.
We trained 10 state-of-the-art (SOTA) detectors on our dataset, including F3Net (Qian et al., 2020), UniFD (Ojha et al., 2023), IID (Huang et al., 2023), FreqNet (Tan et al., 2024a), ProDet (Cheng et al., 2024), NPR (Tan et al., 2024b), AIDE (Yan et al., 2024a), Co-SPY (Cheng et al., 2025), D 3 (Yang et al., 2025b), Effort (Yan et al., 2024c). We also assess 4 open-source MLLMs of similar size to our model, including Qwen2.5-VL-7B (Bai et al., 2025), InternVL3-8B (Zhu et al., 2025), MiMo-VL-7B (Team, 2025) and GLM-4.1V-9B-Thinking (Hong et al., 2025), along with 2 powerful closed-source models GPT-4o (Hurst et al., 2024) and Gemini-2.5-Pro (Comanici et al., 2025). Besides, we evaluate recent MLLM-based forgery detectors, including FakeShield (Xu et al., 2024b), M2F2-Det (Guo et al., 2025b), SIDA (Huang et al., 2025a), FakeVLM (Wen et al., 2025c), FFAA (Huang et al., 2024). More details are in Appendix A.2.
Metrics. Following previous works (Zhang et al., 2024a;Guo et al., 2025b), we take Accuracy (Acc) to measure the model performance. Precision and Recall are reported in Appendix A.5.
this section cite: ['b52', 'b50', 'b29', 'b8', 'b76', 'b2', 'b109', 'b27', 'b34', 'b11', 'b31']

Section: Implementation Details.
We implement VERITAS with InternVL3-8B (Zhu et al., 2025). For the cold-start SFT, we train the model for 3 epochs using LoRA (Hu et al., 2022) (rank=128, α=256). The learning rate is set to 5 × 10 -5 , with a batch size of 64. For cold-start MiPO, the model is trained for 2 epochs with the same setting of SFT. For P-GRPO, we further train the model for 2 epochs with the same LoRA setting. The learning rate is set to 1 × 10 -6 with a batch size of 16. G is set to 4, with a temperature of 1.0. β and β ′ are set to 0. We take UnifiedReward-Qwen-3B (Wang et al., 2025d) as the reward model M. For each stage, we directly adopt model from the last step. 51.2 50.0 50.0 49.7 50.0 52.0 52.9 50.5 56.7 50.7 53.6 54.5 51.6 50.7 53.6 80.2 67.5 52.5 50.5 54.1 InternVL3-8B 54.0 54.0 49.8 49.0 56.6 55.8 57.2 62.9 54.2 62.9 63.6 54.8 67.7 54.4 67.1 77.1 66.5 47.4 51.8 58.3 MiMo-VL-7B 63.8 74.5 77.1 82.5 60.3 82.4 81.4 48.7 82.6 76.4 79.7 78.4 82.8 57.7 75.6 79.4 70.7 67.7 54.9 72.5 GLM-4.1V-9BThink 56.4 55.2 52.3 50.5 51.6 68.4 60.7 54.3 68.4 63.3 65.7 55.1 81.0 58.7 72.7 83.7 69.2 52.0 53.9 61.7 GPT-4o 53.5 57.7 52.0 51.4 59.9 81.2 54.8 66.4 58.9 52.5 64.4 60.9 55.5 49.4 62.0 90.7 73.7 58.0 52.8 60.8 Gemini-2.5-Pro 72.2 64.9 92.4 82.8 62.5 93.4 93.2 73.7 83.3 87.4 85.5 84.7 85.6 67.2 75.6 87.5 82.4 70.9 53.0 78.9 MLLM-based Forgery Detectors M2F2-Det (CVPR'25) -56.0 57.7 59.8 61.8 61.3 55.4 78.9 65.5 80.0 57.4 57.5 76.
3 73.0 56.3 67.2 50.6 53.0 70.6 63.2 FakeShield (ICLR'25) -64.3 64.0 61.5 63.1 61.8 63.3 64.0 57.3 60.9 58.1 63.6 63.7 50.2 83.8 53.8 51.3 53.9 55.6 60.8 SIDA-7B (CVPR'25) -97.3 97.7 79.5 59.3 98.5 95.0 59.8 60.6 62.3 89.7 94.4 63.3 50.4 81.9 80.0 78.0 68.9 57.3 76.3 SIDA-13B (CVPR'25) -80.7 78.5 54.8 52.5 91.3 82.4 63.7 61.2 68.2 56.7 67.1 84.3 60.8 58.2 88.3 74.0 74.1 59.9 69.8 FFAA (Arxiv'24) -55.1 50.9 72.9 63.5 60.8 57.6 82.7 70.9 71.8 58.4 62.4 86.0 67.7 58.4 55.3 59.2 49.6 68.3 64.0 FakeVLM (NIPS'25) -78.2 78.5 77.0 74.5 76.5 76.8 70.8 76.2 76.2 76.9 76.5 77.7 75.7 83.6 81.5 80.8 78.7 74.5 77.3 VERITAS-MINI -95.5 99.1 97.3 72.8 97.0 96.1 82.5 76.3 90.0 83.7 82.9 79.3 72.5 78.7 92.0 93.0 85.5 70.6 85.8 VERITAS (cold-start)
96.8 79.5 99.6 96.0 99.9 99.7 99.9 84.0 65.3 94.8 86.2 93.4 86.7 55.9 73.5 93.7 89.3 88.1 76.4 87.3 VERITAS (ours) 97.3 94.8 99.8 97.0 99.9 99.9 99.9 90.3 75.7 97.0 91.8 95.1 91.7 58.6 84.1 92.3 90.2 89.2 78.5 90.7
this section cite: ['b109']

Section: MAIN RESULTS
Comparison to SOTA detectors. As shown in Table 1, our VERITAS model achieves SOTA performance on four evaluation scenarios, achieving 6.0% averaged gains over the previous best. Existing detectors show great performance on cross-model split (over 90% for D 3 ) but fall short on cross-forgery and cross-domain scenarios (mostly less than 85%). VERITAS mitigates the gap, achieving over 90.0% accuracy on unseen forgery such as face restoration and personalization, and over 90.0% on in-the-wild data from Dreamina and 89.2% on GPT-4o. The cold-start model also achieves promising results, but without incentivizing planning and self-reflection, the cross-forgery results are degraded. More results and analyses can be found in Appendix A.5.
Comparison to SOTA MLLMs. Compared to our base model, VERITAS achieves 32.4% averaged gain, suggesting the effectiveness of our training strategy. The models with similar sizes show limited abilities for deepfake detection, with less than 60% accuracy. Gemini-2.5-Pro shows the best capacities among these MLLMs, even outperforming some of the fine-tuned detectors. VERITAS surpasses Gemini-2.5-Pro by 11.8%, demonstrating great generalization.
Comparisons to MLLM-based detectors. For fair comparisons, we restrict our training scope (Table 1). Even with limited data scope, VERITAS-MINI still outperforms existing MLLM-based detectors, indicating the effectiveness of the proposed framework. M2F2-Det and FFAA, though targeted at deepfake detection, suffer from poor generalization on HydraFake. SIDA-7B and FakeVLM achieve promising results by contrast. Moreover, VERITAS exhibits certain advantages in both detection accuracy and reasoning depth (Figure 6). More cases can be found in Appendix A.8. Figure 4: Ablations on the training stages. "Avg" is directly averaged across four splits.
In-Domain
this section cite: []

Section: 90.3
Cross-Model Cross-Forgery Cross-Domain
87.4 89.6 97.3 98.6 90.0 87.9 86.4 68.7 69.4 72.1 90.3 61.0 69.3 73.1 82.2
this section cite: []

Section: ABLATION STUDIES
We provide primary ablations in main text. More analyses on the training protocol (A.5.2), results on recent benchmark (Li et al., 2025)(A.5.2), selection of P-GRPO training data (A.5.4) and reward model (A.5.6), hyperparameters (A.5.5) and efficiency analysis (A.5.3) can be found in Appendix.
Effect of pattern-aware reasoning. As shown in Table 2, we compare different reasoning paradigms using SFT and P-GRPO training. Although the improvements on in-domain datasets are marginal, our pattern-aware reasoning demonstrates clear advantages to flexible reasoning on OOD scenarios, achieving 6.2% and 3.3% gains on CF and CD testing respectively. The post-hoc explanation adopted in recent methods exhibits degraded performance in OOD testing, further verifying the superiority of pattern-aware reasoning.
Ablations on different training stages. As shown in Figure 4, we investigate the effect of each training stage. Applying MiPO or P-GRPO upon SFT model both achieve significant gains, with P-GRPO performing better, which is due to the online sampling and pattern-aware incentivization. Applying MiPO before P-GRPO yields the best performance, achieving 2.9% and 2.1% gains on CF and CD testing respectively. This is because MiPO ensures high-quality rollouts in subsequent stage, facilitating more accurate policy updates for online RL.
this section cite: ['b43']

Section: Effect of Pattern-guided Cold-Start.
As shown in Figure 5, we investigate different RL settings without cold-start. The training data keeps consistent with our two-stage pipeline. Answer-only model achieves better ID results while incorporating thinking improves CM and CD performance. However, all settings underperform the model with cold-start. The low-quality explorations lead to unstable training. Results in Figure 4 further verify the effectiveness of MiPO during cold-start.
Effect of Pattern-aware GRPO. As shown in Table 3, our P-GRPO achieves noticeable improvements compared to original GRPO. Specifically, pattern-aware reward outperforms the vanilla accuracy reward especially on CF and CD scenarios. The reflection quality reward benefits both original GRPO and our P-GRPO, which demonstrates the importance of high-quality self reflection. In Appendix A.5.4, we observe that by adding several "unseen" data in P-GRPO, the ODD performance can be further improved, demonstrating promising scalability with only binary labels required.
Effect of specific reasoning patterns. As shown in Table 5, "fast judgement" is helpful for CF and CD, but is not critical overall. "planning" is more effective on CM, since the fully synthesized images require a more holistic and structured analysis. "self-reflection" is critical especially on CF and CD, as it incentivizes the model to discover those unseen artifacts. "conclusion" provides certain gains, suggesting that synthesizing separate evidence into a coherent verdict is also important.
this section cite: []

Section: Ablations on the non-preference in MiPO.
As shown in Table 6, s ϕ l helps improve the performance on CF (+1.3%) and CD (+0.8%) scenarios. To understand the effects, we provide a qualitative case in Figure 20. Without sϕ, the model still gets correct answers, but the analysis is superficial and less detailed, which causes certain failures on unseen forgeries that might require in-depth reasoning. <fast> At first glance, the image appears convincingly realistic, with no immediate signs of manipulation or artificial generation. </fast> <planning> Given the initial impression of authenticity, a thorough and systematic examination is essential. This involves assessing structural integrity, identifying subtle visual artifacts, and verifying physical plausibility, particularly in the expression and texture. This multi-layered approach is necessary to uncover any hidden indicators of AI generation. </planning> <reasoning> Upon closer inspection, the facial structure and proportions appear natural, with no significant asymmetry or distortion. The lighting and shadows are consistent, and the background does not introduce any anomalies. However, the expression raises some suspicion; the severe wrinkling of the forehead is unusually intense and lacks the natural coordination typically seen between the eyes and mouth. In genuine expressions, these features usually move in harmony, but here, the mouth appears slightly open as if in mid-speech, while the eyes remain neutral, suggesting a mismatch. Furthermore, the texture of the skin, especially around the forehead, is excessively detailed, almost to the point of being unnaturally exaggerated. This level of detail is often a result of AI overemphasis on texture realism, potentially masking other subtle inconsistencies. The hairline and beard are well-defined, adding to the image's realistic appearance, but the overall facial texture seems slightly too perfect, lacking the minor imperfections found in real human skin. </reasoning> <reflection> The combination of an exaggerated expression and overly detailed skin texture is a known challenge for AI models. While individually these elements might seem plausible, together they create an uncanny effect. The AI might have prioritized detail in texture and expression to simulate realism, inadvertently creating an expression that feels staged or artificially intense. </reflection> <conclusion> Although the image initially projects authenticity, the mismatch in expression and the overly detailed skin texture suggest artificial generation. These nuanced elements, not immediately apparent, betray the image's AI origin, classifying it as a medium-hard detection case. The image is convincingly realistic but ultimately reveals its artificial nature through these subtle yet telling discrepancies. </conclusion> <answer> fake </answer> <s> The image appears to be a digital artwork or manipulated photo due to the smooth skin and the pattern of the skin texture looks too perfect, smooth and controlled to be natural. The person has a photoshop fake.</s> This is a fake image. The image exhibits underlying characteristic inconsistencies in its features that suggest it is artificially created.
this section cite: []

Section: FakeShield (22B):
1. Whether the picture has been tampered with / Description of the tampered area: The picture has not been tampered with. It appears to be a genuine photograph taken directly by a camera. 2. Judgment basis:
The image shows a person's hair and ears against a plain background. The lighting on the hair and ears is consistent, with no unusual shadows or highlights that would suggest manipulation.
The edges of the hair and ears are natural, with no signs of unnatural pixel distribution or jagged edges that would indicate compositing.
The resolution across the image is uniform, without any areas of blurring or compression that would be expected. The perspective is appropriate for a close-up shot, with the ears and hair appearing in proportion to each other. There are no reflections or shadows that contradict the light source, and there are no texts or logos present to assess for incongruity. Lastly, nothing in the image violates physical laws. All these details support the conclusion that this is a real, unaltered photograph.
this section cite: []

Section: SIDA-13B-description:
Type: part
tampered Areas:<tampered position> The tampering is focused on the upper right quadrant of the image, specifically affecting the man's hair and the background. Tampered Content:<Types of objects or parts> Hair and background Visual Inconsistencies:<lighting> The lighting on the hair and background does not match the overall lighting of the rest of the image<edges> The edges around the hair and background are unnaturally sharp or blurred<resolution> The resolution of the tampered areas is either overly sharp or unnaturally blurred<shadows> Shadows and reflections in the tampered areas do not align with the lighting in the rest of the image<boundary> The boundaries of the tampered areas are unnaturally sharp or blurred, indicating a lack of seamless integration Summary: The given image is {part tampered} based on the above analysis.
this section cite: []

Section: FFAA (7B):
Image description: The image shows a close-up of a man's face with a serious expression. The lighting is even, and the resolution is high, allowing for clear visibility of facial details. The man appears to be in his 30s or 40s, with light skin, short dark hair, and stubble. His face is oriented directly towards the camera, and he has prominent forehead wrinkles, deep-set eyes, and slightly parted lips. Forgery reasoning: Upon close examination, the facial features appear natural and well-integrated. The skin texture is consistent, with no signs of blurring, pixelation, or unnatural smoothness.
The wrinkles on the forehead and the stubble on the face are detailed and blend seamlessly with the surrounding skin. There are no visible fusion artifacts or inconsistencies in skin color that would suggest manipulation. The expression and facial attributes appear natural and consistent with the rest of the image. The lighting and shadows are uniformly distributed, further supporting the authenticity of the image. Analysis result: real, Forgery type: None Match score: 0.8858; Difficulty: hard Veritas (ours, 8B): (partially correct)
this section cite: []

Section: FURTHER ANALYSES
Evaluation of reasoning quality. To evaluate the reasoning quality, we take two types of assessments: (1) score evaluation which is based on predefined criteria (Figure 33). (2) Pairwise comparison which directly compares outputs from two models. We adopt MLLM-as-a-Judge (Chen et al., 2024a), using GPT-4o and Gemini-2.5-Pro for evaluation. Similar to (Zhou et al., 2025), we randomly select 1K samples for evaluation. As shown in Table 7, our model achieves the best score and ELO rating, where MiPO greatly improves the reasoning quality. Moreover, our MiPO outperforms DPO in raising reasoning quality, which verifies the effectiveness of mixed non-preference strategy.
Different fine-tuned base models and model sizes. As shown in Table 4, we adopt different MLLMs as our base model. InternVL3-8B outperforms Qwen2.5-VL-7B and MiMo-VL-7B, due to the dynamic high resolution strategy. InternVL3-2B achieves promising performance with fewer parameters, while scaling up to 14B yields considerable gains on CM and CF scenarios.
Robustness evaluation. We investigate the performance under JPEG compression and Gaussian blur. Results in Table 8 highlight the robustness of our model. Our model achieves consistently high performance under JPEG compression and maintains state-of-the-art results across different perturbations. Notably, this robustness is achieved without training on corresponding data augmentations such as random Gaussian blur, which instead are commonly adopted in previous methods.
this section cite: ['b108']

Section: CONCLUSION
In this paper, we introduce HydraFake dataset and VERITAS model. HydraFake introduces a holistic evaluation protocol to comprehensively measure the generalization capacities. We then train a multi-modal large language model (MLLM) based deepfake detector trained with our two-stage pipeline. Results on HydraFake show that current detectors struggle on cross-forgery and crossdomain scenarios, while our model greatly mitigates the gap and is capable of delivering transparent decision process. We hope this work can inspire more generalizable and reliable deepfake detection.
this section cite: []

Section: References
Ref_id:b0 Title: Adobe Firefly Year: (2023)
Ref_id:b1 Title:  Year: (2023)
Ref_id:b2 Title: Black Forest Labs. Flux v1.1 pro Year: (2024)
Ref_id:b3 Title: Wild: a new in-the-wild image linkage dataset for synthetic image attribution Year: (2025)
Ref_id:b4 Title: Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark Year: (2024)
Ref_id:b5 Title: Advancing multimodal reasoning: From optimized cold start to staged reinforcement learning Year: (2025)
Ref_id:b6 Title: Mgffd-vlm: Multi-granularity prompt learning for face forgery detection with vlm Year: (2025)
Ref_id:b7 Title: X2-dfd: A framework for explainable and extendable deepfake detection Year: (2024)
Ref_id:b8 Title: Can we leave deepfake data behind in training deepfake detector? Year: (2024)
Ref_id:b9 Title: Co-spy: Combining semantic and pixel features to detect synthetic images by ai Year: (2025)
Ref_id:b10 Title: Stargan v2: Diverse image synthesis for multiple domains Year: (2020)
Ref_id:b11 Title: Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities Year: (2025)
Ref_id:b12 Title: The deepfake detection challenge (dfdc) preview dataset Year: (2019)
Ref_id:b13 Title:  Year: (2020)
Ref_id:b14 Title: Scaling rectified flow transformers for high-resolution image synthesis Year: (2024)
Ref_id:b15 Title: Sophiavl-r1: Reinforcing mllms reasoning with thinking reward Year: (2025)
Ref_id:b16 Title: Exploring unbiased deepfake detection via token-level shuffling and mixing Year: (2025)
Ref_id:b17 Title: Towards generalizable forgery detection and reasoning Year: (2025)
Ref_id:b18 Title: Spatiotemporal inconsistency learning for deepfake video detection Year: (2021)
Ref_id:b19 Title: Delving into the local: Dynamic inconsistency learning for deepfake video detection Year: (2022)
Ref_id:b20 Title: Hierarchical contrastive inconsistency learning for deepfake video detection Year: (2022)
Ref_id:b21 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b22 Title: Rethinking vision-language model in face forensics: Multi-modal interpretable forged face detector Year: (2025)
Ref_id:b23 Title: Pulid: Pure and lightning id customization via contrastive alignment Year: (2024)
Ref_id:b24 Title: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis Year: (2025)
Ref_id:b25 Title: Face-adapter for pre-trained diffusion models with fine-grained id and attribute control Year: (2024)
Ref_id:b26 Title: Vlforgery face triad: Detection, localization and attribution via multimodal large language models Year: (2025)
Ref_id:b27 Title: Lihang Pan, et al. Glm-4.1 v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning Year: (2025)
Ref_id:b28 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b29 Title: Implicit identity driven deepfake face swapping detection Year: (2023)
Ref_id:b30 Title: Labeled faces in the wild: A database forstudying face recognition in unconstrained environments Year: (2008)
Ref_id:b31 Title: Ffaa: Multimodal large language model based explainable open-world face forgery analysis assistant Year: (2024)
Ref_id:b32 Title: Sida: Social media image deepfake detection, localization and explanation with large multimodal model Year: (2025)
Ref_id:b33 Title: So-fake: Benchmarking and explaining social media image forgery detection Year: (2025)
Ref_id:b34 Title: Gpt-4o system card Year: (2024)
Ref_id:b35 Title: Openai o1 system card Year: (2024)
Ref_id:b36 Title: Infiniteyou: Flexible photo recrafting while preserving your identity Year: (2025)
Ref_id:b37 Title: Progressive growing of gans for improved quality, stability, and variation Year: (2017)
Ref_id:b38 Title: A style-based generator architecture for generative adversarial networks Year: (2019)
Ref_id:b39 Title: Freqdebias: Towards generalizable deepfake detection via consistency-driven frequency debiasing Year: (2025)
Ref_id:b40 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b41 Title: Face x-ray for more general face forgery detection Year: (2020)
Ref_id:b42 Title: Celeb-df: A large-scale challenging dataset for deepfake forensics Year: (2020)
Ref_id:b43 Title: Is artificial intelligence generated image detection a solved problem Year: (2025)
Ref_id:b44 Title: Longperceptualthoughts: Distilling system-2 reasoning for system-1 perception Year: (2025)
Ref_id:b45 Title: Visual instruction tuning Year: (2023)
Ref_id:b46 Title: Deep learning face attributes in the wild Year: (2015)
Ref_id:b47 Title: Visual-rft: Visual reinforcement fine-tuning Year: (2025)
Ref_id:b48 Title: Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling Year: (2025)
Ref_id:b49 Title: Laa-net: Localized artifact attention network for quality-agnostic and generalizable deepfake detection Year: (2024)
Ref_id:b50 Title: Towards universal fake image detectors that generalize across generative models Year: (2023)
Ref_id:b51 Title: Mllm-enhanced face forgery detection: A vision-language fusion solution Year: (2025)
Ref_id:b52 Title: Thinking in frequency: Face forgery detection by mining frequency-aware clues Year: (2020)
Ref_id:b53 Title: Can multi-modal (reasoning) llms work as deepfake detectors Year: (2025)
Ref_id:b54 Title: Faceforensics++: Learning to detect manipulated facial images Year: (2019)
Ref_id:b55 Title: Magi-1: Autoregressive video generation at scale Year: (2025)
Ref_id:b56 Title: Detecting deepfakes with self-blended images Year: (2022)
Ref_id:b57 Title: Towards general visual-linguistic face forgery detection Year: (2025)
Ref_id:b58 Title: Frequencyaware deepfake detection: Improving generalizability through frequency space domain learning Year: (2024)
Ref_id:b59 Title: Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection Year: (2024)
Ref_id:b60 Title: Efficient visual generation with hybrid autoregressive transformer Year: (2024)
Ref_id:b61 Title: Llms are not yet ready for deepfake image detection Year: (2025)
Ref_id:b62 Title:  Year: (2025)
Ref_id:b63 Title:  Year: (2025)
Ref_id:b64 Title: -vl technical report Year: (2025)
Ref_id:b65 Title: Mimo-vl technical report Year: (2025)
Ref_id:b66 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b67 Title: Learning when to think Year: (2025)
Ref_id:b68 Title: Dfbench: Benchmarking deepfake image detection capability of large multimodal models Year: (2025)
Ref_id:b69 Title: Forensics-bench: A comprehensive forgery detection benchmark suite for large vision language models Year: (2025)
Ref_id:b70 Title: Dynamicface: High-quality and consistent video face swapping using composable 3d facial priors Year: (2025)
Ref_id:b71 Title: Unified reward model for multimodal understanding and generation Year: (2025)
Ref_id:b72 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b73 Title: Busterx: Mllm-powered ai-generated video forgery detection and explanation Year: (2025)
Ref_id:b74 Title: Busterx++: Towards unified cross-modal ai-generated content detection and explanation with mllm Year: (2025)
Ref_id:b75 Title: Spot the fake: Large multimodal model-based synthetic image detection with artifact explanation Year: (2025)
Ref_id:b76 Title: Towards ai-generated image detection in the wild Year: (2025)
Ref_id:b77 Title: Fast-slow thinking for large vision-language model reasoning Year: (2025)
Ref_id:b78 Title: Vfhq: A high-quality dataset and benchmark for video face super-resolution Year: (2022)
Ref_id:b79 Title: Talkingheadbench: A multi-modal benchmark & analysis of talking-head deepfake detection Year: (2025)
Ref_id:b80 Title: Llava-cot: Let vision language models reason step-by-step Year: (2024)
Ref_id:b81 Title: Explainable image forgery detection and localization via multi-modal large language models Year: (2024)
Ref_id:b82 Title: A sanity check for ai-generated image detection Year: (2024)
Ref_id:b83 Title: Deepfakebench: A comprehensive benchmark of deepfake detection Year: (2023)
Ref_id:b84 Title: Transcending forgery specificity with latent space augmentation for generalizable deepfake detection Year: (2024)
Ref_id:b85 Title: Orthogonal subspace decomposition for generalizable ai-generated image detection Year: (2024)
Ref_id:b86 Title: Df40: Toward next-generation deepfake detection Year: (2024)
Ref_id:b87 Title: Generalizing deepfake video detection with plug-and-play: Videolevel blending and spatiotemporal adapter tuning Year: (2025)
Ref_id:b88 Title: Qwen3 technical report Year: (2025)
Ref_id:b89 Title: Exposing deep fakes using inconsistent head poses Year: (2019)
Ref_id:b90 Title: Dˆ3: Scaling up deepfake detection by learning from discrepancy Year: (2025)
Ref_id:b91 Title: All patches matter, more patches better: Enhance aigenerated image detection via panoptic patch learning Year: (2025)
Ref_id:b92 Title: Dreamid: High-fidelity and fast diffusion-based face swapping via triplet id group learning Year: (2025)
Ref_id:b93 Title: A comprehensive synthetic data detection benchmark using large multimodal models Year: (2024)
Ref_id:b94 Title: Echo-4o: Harnessing the power of gpt-4o synthetic images for improved image generation Year: (2025)
Ref_id:b95 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b96 Title: Towards general multimodal reasoning via cue-guided rethinking Year: (2025)
Ref_id:b97 Title: R1-vl: Learning to reason with multimodal large language models via step-wise group relative policy optimization Year: (2025)
Ref_id:b98 Title: Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport Year: (2025)
Ref_id:b99 Title: Ivy-fake: A unified explainable framework and benchmark for image and video aigc detection Year: (2025)
Ref_id:b100 Title: Common sense reasoning for deepfake detection Year: (2024)
Ref_id:b101 Title: A-bench: Are lmms masters at evaluating aigenerated images Year: (2024)
Ref_id:b102 Title: Echo chamber: Rl post-training amplifies behaviors learned in pretraining Year: (2025)
Ref_id:b103 Title: Learning selfconsistency for deepfake detection Year: (2021)
Ref_id:b104 Title: Swift:a scalable lightweight infrastructure for fine-tuning Year: (2024)
Ref_id:b105 Title: Enhancing deepfake detection by blending frequency knowledge Year: (2024)
Ref_id:b106 Title: Towards robust blind face restoration with codebook lookup transformer Year: (2022)
Ref_id:b107 Title: Face forensics in the wild Year: (2021)
Ref_id:b108 Title: Aigi-holmes: Towards explainable and generalizable ai-generated image detection via multimodal large language models Year: (2025)
Ref_id:b109 Title: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models Year: (2025)
Ref_id:b110 Title: A challenging real-world dataset for deepfake detection Year: (2020)
Ref_id:b111 Title: Performance comparison on the In-Domain (ID) subset of HydraFake dataset. The best results are bolded and the second best are underlined Year: ()
Ref_id:b112 Title:  Year: ()
Ref_id:b113 Title: Table 11: Performance comparison on the Cross-Model (CM) subset of HydraFake dataset. The best results are bolded and the second best are underlined Year: ()
Ref_id:b114 Title:  Year: ()
Ref_id:b115 Title:  Year: ()
Ref_id:b116 Title:  Year: ()
Ref_id:b117 Title: Precision (P.) and Recall (R.) and the averaged results (Avg.) are reported in Accuracy. Method StarGANv2 IC-Light CodeFormer InfiniteYou PuLID FaceAdapter Avg Year: ()
Ref_id:b118 Title:  Year: ()
Ref_id:b119 Title:  Year: ()
Ref_id:b120 Title:  Year: ()
Ref_id:b121 Title: Precision (P.) and Recall (R.) and the averaged results (Avg.) are reported in Accuracy. Method DeepFaceLab InfiniteYou-CD Dreamina Hailuo AI GPT-4o FFIW Avg Year: ()
Ref_id:b122 Title:  Year: ()
Ref_id:b123 Title:  Year: ()
Ref_id:b124 Title:  Year: ()
