Title: VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank
Abstract: Clear, and the composition is centered. The main subject, a bird in the middle, is clearly defined. Therefore, the quality is good.Please describe the visual quality of this image.

Section: 
The image is of high quality, featuring a clear and sharp depiction of a bird standing in water with its reflection visible below. The details of the bird's feathers, beak, and legs are well-defined, indicating good focus and lighting. The water's surface is smooth, creating a clean and symmetrical reflection that enhances the overall composition. The background appears to be a natural setting, possibly a wetland or marsh, which adds context to the scene without distracting from the main subject. The colors are vibrant, with the bird's brown and white plumage contrasting nicely against the blue tones of the water. Overall, the image is wellcomposed and captures the essence of the bird in its natural habitat.
this section cite: []

Section: VisualQuality-R1 Q-Instruct
Are rich, and the lighting is sufficient. The texture details are not clear. The composition is centered, and the main subject is clearly the man in the middle. Therefore, the quality is poor.
this section cite: []

Section: Please describe the visual quality of this image.
The image appears to be of low quality, with several noticeable issues: 1. Blurriness: The subject and background are not in sharp focus, making it difficult to discern fine details. 2. Lighting: The lighting seems uneven, with some areas overexposed and others underexposed, particularly around the subject's face and the microphone. 3. Color Balance: The colors appear somewhat muted or distorted, possibly due to the lighting conditions or the camera settings. 4. Noise: There is visible noise or graininess in the image, which can be seen as small specks or dots, especially in darker areas. Overall, the image lacks clarity and detail, which affects its quality. Improvements could include better focus, more consistent lighting, and potentially adjusting the color balance and reducing noise.
this section cite: []

Section: Abstract
DeepSeek-R1 has demonstrated remarkable effectiveness in incentivizing reasoning and generalization capabilities of large language models (LLMs) through reinforcement learning. Nevertheless, the potential of reasoning-induced computation has not been thoroughly explored in the context of image quality assessment (IQA), a task depending critically on visual reasoning. In this paper, we introduce VisualQuality-R1, a reasoning-induced no-reference IQA (NR-IQA) model, and we train it with reinforcement learning to rank, a learning algorithm tailored to the intrinsically relative nature of visual quality. Specifically, for a pair of images, we employ group relative policy optimization to generate multiple quality scores for each image. These estimates are used to compute comparative probabilities
this section cite: []

Section: Introduction
Image quality assessment (IQA) aims to quantify the visual quality of digital images consistent with human perceptual judgments. Commonly, IQA models are classified into full-reference (FR) and noreference (NR) approaches [47], depending on the availability of pristine-quality reference images. In this paper, we focus on NR-IQA due to its practical relevance in real-world scenarios where reference images are unavailable. Over the decades, NR-IQA has evolved from knowledge-driven [33,12] to data-driven approaches [30,19,54], and shifted from regression-based to ranking-based [58,59] techniques. Nevertheless, achieving strong model generalization (e.g., generalization to unseen image distortions) remains a significant, unresolved challenge, driving recent research toward multi-dataset training [6], active fine-tuning [44], and continual model adaptation [57].
The rapid advancement of vision-language models (VLMs) offers promising avenues for enhancing NR-IQA generalization by contextualizing it into broader vision tasks [51]. VLMs can effectively integrate multi-modal information, enabling understanding of both low-level image distortions (e.g., noise and blur) and high-level perceptual attributes (e.g., aesthetics and content semantics). This multi-modal semantic contextualization allows VLMs to articulate nuanced quality descriptions with stronger generalization.
However, current NR-IQA methods mainly leverage VLMs through supervised fine-tuning (SFT), which face several critical limitations [49,56]. First, constructing informative quality descriptions demands extensive human effort, rendering the annotation process labor-intensive and expensive 2 . Second, models trained via SFT often overfit to the biases and idiosyncrasies present in training data, and may unintentionally encounter catastrophic forgetting of acquired knowledge during pre-training. Third, SFT typically yields overly rigid and templated outputs (see Fig. 1) that may be less useful.
Reinforcement learning (RL) has recently emerged as a powerful alternative, enhancing the reasoning capabilities of LLMs, while aligning their responses with human preferences [35,13]. In particular, DeepSeek-R1 [13] demonstrates the effectiveness of RL in promoting generalization by encouraging automated exploration of plausible reasoning paths and employing rule-based rewards to prevent reward hacking [37]. However, a direct adaptation of RL techniques to NR-IQA, as exemplified by the recent Q-Insight model [21], has been limited by its reliance on dataset-specific reward design and additional distortion-type classification. These constraints stem from its treatment of visual quality as an absolute perceptual quantity, thereby framing NR-IQA naïvely as a regression task.
In this paper, we introduce VisualQuality-R1, a reasoning-induced NR-IQA model, and we train it via reinforcement learning to rank (RL2R), a learning algorithm explicitly designed to capture the inherently relative nature of visual quality. Specifically, we employ group relative policy optimization (GRPO) [36] to derive multiple quality scores for each image in a pair. We then compute comparative probabilities between images using the Thurstone model [41] by assessing the difference between the mean quality score of one image and individual quality scores of another, normalized by their sample variances. Unlike previous methods, we define the reward function using the continuous fidelity measure [42], which provides precise guidance to facilitate quality ranking. Extensive experiments confirm that VisualQuality-R1 effectively assesses visual quality across a diverse range of distortion scenarios, outperforming discriminative deep learning-based NR-IQA models as well as a recent reasoning-induced quality regression method [21]. Moreover, VisualQuality-R1 generates contextually rich, human-aligned quality descriptions (see Fig. 1), which can be leveraged to provide targeted feedback for downstream image processing algorithms, and support fine-grained quality control in digital photography pipelines. Additionally, we demonstrate that VisualQuality-R1 remains effective across multi-dataset training scenarios without requiring perceptual scale realignment.
this section cite: ['b46', 'b32', 'b11', 'b29', 'b18', 'b53', 'b57', 'b58', 'b5', 'b43', 'b56', 'b50', 'b48', 'b55', 'b34', 'b12', 'b12', 'b36', 'b20', 'b35', 'b40', 'b41', 'b20']

Section: Related Work
This section provides a structured review of related NR-IQA models, emphasizing recent advancements, particularly those leveraging VLMs.
Regression-based Models NR-IQA models primarily employed regression-based approaches, wherein image quality was treated as an absolute perceptual quantity directly estimated from extracted "quality-aware" features. Initially, features were handcrafted based on natural scene statistics [33,34], degradation-specific characteristics [46,48,24], and perceptual models inspired by the human visual system [45]. Nonetheless, these methods were limited by the representational capacity of handcrafted features. Later, deep learning-based regression models emerged as the dominant paradigm, using end-to-end trainable neural networks to directly predict quality scores (or, in some cases, quality distributions) [18,30,3,39,54,52]. These models typically utilize standard regression losses such as the mean squared error and mean absolute error, or statistical distances such as the earth mover's distance [39] and Kullback-Leibler (KL) divergence [55]. Regression-based models often struggle with generalization issues, and require labor-intensive perceptual scale realignment [31] when training on multiple IQA datasets.
this section cite: ['b32', 'b33', 'b45', 'b47', 'b23', 'b44', 'b17', 'b29', 'b2', 'b38', 'b53', 'b51', 'b38', 'b54', 'b30']

Section: Ranking-based Models
To address these shortcomings, ranking-based NR-IQA models were introduced, modeling visual quality as an intrinsically relative perceptual quantity. Gao et al. [10] pioneered the concept of quality ranking in NR-IQA, although their initial implementation relied on predefined anchor images and was not end-to-end optimized. Ma et al. [29] adapted RankNet [4] to NR-IQA by training (though not fully end-to-end) on quality-discriminable image pairs. Their subsequent work established the first end-to-end ranking-based NR-IQA method grounded in the Thurstone model [41]. Nevertheless, their approach suffers from scaling ambiguity during variance estimation. Zhang et al. [58] incorporated a hinge loss to regularize variance estimation, yet the scaling ambiguity persisted. Their study also demonstrated the superiority of the fidelity loss [42] over the conventional cross-entropy loss in ranking-based NR-IQA. Subsequent research has adopted a simpler approach by fixing the variance parameter to one (corresponding to the Thurstone Case V model), facilitating active fine-tuning of NR-IQA models on challenging examples [44,43] and allowing for continual adaptation to novel distortion scenarios [57]. Other losses that enable quality ranking include the margin ranking loss [25], differentiable approximations of Spearman's rank correlation coefficient (SRCC) [2], Pearson linear correlation coefficient (PLCC) [53], and statistical distances between permutation probabilities [5,38,17].
this section cite: ['b9', 'b28', 'b3', 'b40', 'b57', 'b41', 'b43', 'b42', 'b56', 'b24', 'b1', 'b52', 'b4', 'b37', 'b16']

Section: VLM-based Models
The integration of VLMs into NR-IQA has recently gained traction, particularly due to their proficiency in capturing contextual semantics through multi-modal representation learning. Early attempts include multitask adaptation of CLIP [59], as well as SFT-based methods like Q-Align [50], Compare2Score [60], DepictQA [56], and DeQA-Score [55], which trained VLMs to generate either quality scores, distributions, or descriptions. Closest to ours, Q-Insight [21] explored reasoning-induced quality regression through RL. However, Q-Insight struggles with the dataset-specific reward calibration, added complexity of auxiliary distortion-type classification, and generalization to novel distortion scenarios. In contrast, our VisualQuality-R1 redefines the use of VLMs in NR-IQA by shifting from absolute regression to relative ranking, leading to enhanced generalization across distortion scenarios with better quality justifications.
this section cite: ['b58', 'b49', 'b59', 'b55', 'b54', 'b20']

Section: Reasoning-Induced NR-IQA
To harness both the powerful reasoning-inducing capabilities of RL and the intrinsically relative nature of visual quality, we propose an NR-IQA model-VisualQuality-R1-and an RL2R method of training it that seamlessly integrates the Thurstone model within GRPO. Fig. 2 shows the system diagram of VisualQuality-R1. VisualQuality-R1 Image Text Prompt <think>The image appears to be blurry, which significantly reduces its clarity and detail. The colors are somewhat muted […] </think><answer>1.20</answer> Response 1 … <think>The image appears to be blurry and lacks sharpness, which affects the clarity and detail that can be discerned […] </think><answer>1.22</answer> Response K Predicted Scores … Image <think>The image appears to be a closeup of an animal's eye, likely a fox or similar creature, […] </think><answer>3.50</answer> Response 1 … <think>The image appears to be a closeup of an animal's face, likely a fox, given the fur pattern and eye color […] </think><answer>4.20</answer> Response K Predicted Scores … Thurstone Model Fidelity Reward Human Preference Reinforcement Learning to Rank (RL2R) Text Prompt VisualQuality-R1 𝒌-th Estimate and Variance Mean Estimate and Variance Given an image pair (x i , x j ) with a shared text prompt c, VisualQuality-R1 generates K responses. Following GRPO [36], each response includes a detailed reasoning process and a predicted quality score. To assess relative visual quality, we calculate the asymmetric comparative probability that image x i is perceived better than x j under the Thurstone model [41]. This involves subtracting the mean predicted score of x j from the k-th score of x i , standardized by their sample variances. A fidelity reward is derived from human preference, providing continuous supervisory signals for policy optimization.
this section cite: ['b35', 'b40']

Section: VisualQuality-R1 via RL2R
Given a text prompt c and an image x, our goal is to fine-tune a pre-trained VLM, with policy π θ (•|c, x), to produce a scalar quality score in the range of [1,5], following a step-by-step reasoning process, encapsulated within specially designated tags for explicit instruction and enhanced interpretability. The complete structured text prompt is provided in Table 1.
More specifically, for a training batch of images {x 1 , x 2 , . . . , x B }, where B is the minibatch size, we apply GRPO to generate K quality predictions for x i , q(x i ) = [q 1 (x i ), q 2 (x i ), . . . , q K (x i )] ⊺ . This output naturally encodes predictive uncertainty, which is crucial for making reliable relative quality ranking. Under the Thurstone model [41], the visual quality of an image is assumed to follow a Gaussian distribution. Thus, we compute the asymmetric comparative probability for each of the B × (B -1) ordered image pairs by subtracting the mean quality score of x j from the k-th quality score of x i , standardized by their sample variances:
p k (x i , x j ) = Φ q k (x i ) -µ(q(x j )) σ 2 (q(x i )) + σ 2 (q(x j )) + γ , for i ̸ = j,(1)
where Φ(•) is the standard Gaussian cumulative distribution function. µ(q(x j )) and σ 2 (q(x j )) represent the mean and variance of the quality predictions for x j , respectively. γ is a small positive constant to avoid any potential division by zero. Compared to previous ranking-based NR-IQA models that fix the variance parameter in Eq. ( 1) to one, we explicitly leverage sample variances derived from GRPO. This gives us an opportunity to dynamically accommodate predictive uncertainty for different images. Meanwhile, using the sample mean for quality comparison stabilizes the asymmetric probability estimate and the subsequent reward calculation by appropriately penalizing outlier predictions.
The true preference p(x i , x j ) is derived from human mean opinion scores (MOSs):
p(x, y) =    1 if MOS(x) > MOS(y) 0.5 if MOS(x) = MOS(y) 0 otherwise .(2)
An important aspect of our RL2R algorithm is that we define the reward function r k (x i ) for each quality estimate q k (x i ) as the fidelity measure [42]-a continuous analogue of the discretized binary Table 1: Structured text prompt used in VisualQuality-R1. You are doing the image quality assessment task. Here is the question: What is your overall rating on the quality of this picture? The rating should be a float between 1 and 5, rounded to two decimal places, with 1 representing very poor quality and 5 representing excellent quality. First output the thinking process in <think> </think> tags and then output the final answer with only one score in <answer> </answer> tags. reward [13,21], averaged across all B -1 image pairs:
r k (x i ) = 1 B -1 j̸ =i p(x i , x j )p k (x i , x j ) + (1 -p(x i , x j ))(1 -p k (x i , x j )) .(3)
This continuous reward feedback provides precise guidance during RL2R by capturing subtle variations in quality ranking, thus improving generalization across diverse distortion scenarios. We collect K fidelity rewards for
x i into the vector r(x i ) = [r 1 (x i ), r 2 (x i ), . . . , r K (x i )] ⊺
, and compute the relative advantage a k (x i ) by standardizing rewards within group:
a k (x i ) = r k (x i ) -µ(r(x i )) σ(r(x i )) .(4)
The final policy update of π θ (•|c, x i ) is guided by the regularized objective in GRPO:
ℓ(θ) = 1 BK B i=1 K k=1 min π θ (o k |c, x i ) π θold (o k |c, x i ) a k (x i ), clip π θ (o k |c, x i ) π θold (o k |c, x i ) , 1 -ϵ, 1 + ϵ a k (x i ) -β D KL (π θ (o k |c, x i )∥π ref (o k |c, x i )) .
(5) Here, π θref (•|c, x i ) denotes the stable reference policy obtained after VLM pre-training, and π θold (•|c, x i ) is the policy from the previous RL2R training epoch, from which we sample K reasoning trajectories o = {o k } K k=1 . The second KL divergence term is approximated by
D KL (π θ (o k |c, x i )∥π ref (o k |c, x i )) = π ref (o k |c, x i ) π θ (o k |c, x i ) -log π ref (o k |c, x i ) π θ (o k |c, x i ) -1,(6)
incorporated to ensure that the updated policy π θ (•|c, x i ) does not deviate excessively from π ref (•|c, x i ). ϵ is the clipping threshold to prevent large and potentially destabilizing updates to the policy. The coefficient β serves as a balancing factor between the reward-weighted likelihood term and the KL regularization term.
We conclude this section by highlighting the key strengths of our VisualQuality-R1. First, VisualQuality-R1 inherits all the advantages of ranking-based NR-IQA models, enabling effective multi-dataset training, active fine-tuning, and continual model adaptation without requiring perceptual scale realignment [31], a feature notably absent in regression-based NR-IQA approaches. Second, trained via RL2R, VisualQuality-R1 mitigates the scalability and overfitting issues inherent in SFT-based models. Third, VisualQuality-R1 promises to both improve model generalizability and furnish contextually rich textual justifications alongside numerical quality scores, thereby boosting its practical relevance in real-world IQA applications.
this section cite: ['b0', 'b4', 'b40', 'b41', 'b12', 'b20', 'b30']

Section: Experiments
To validate VisualQuality-R1, we conduct comprehensive experiments across diverse distortion scenarios, ablation studies on key design components, and in-depth analysis of model behaviors.
this section cite: []

Section: Experimental Setups
Competing Models and Training Details Competing methods encompass three categories: 1) handcrafted models: NIQE [34] and BRISQUE [33]; 2) discriminative deep-learning-based models: UNIQUE [58], MUSIQ [19], and MANIQA [54]; 3) VLM-based models: LIQE [59], Q-Align [50], DeQA-Score [55], Q-Insight [21], as well as the pre-trained Qwen2.5-VL-7B [1] baseline.
We fine-tune Qwen2.5-VL-7B [1] as the backbone for VisualQuality-R1 using GRPO [36]. The AdamW optimizer [27] is employed with an initial learning rate of 1 × 10 -6 and a linear decay schedule. For GRPO, we generate six candidate responses per prompt (i.e., K = 6) and set the balance coefficient β to 0.04. Training runs on 16 NVIDIA A100 GPUs with a minibatch size of eight per GPU, taking approximately five hours for a total of 10 epochs.
this section cite: ['b33', 'b32', 'b57', 'b18', 'b53', 'b58', 'b49', 'b54', 'b20', 'b0', 'b0', 'b35', 'b26']

Section: Main Results
Single-Dataset Training We first train NR-IQA models on the synthetic KADID-10K [23] training set (6 : 2 : 2 split while ensuring content independence) and test in a zero-shot setting across eight datasets with distortions arising from digital imaging and (post-)processing stages: BID [7], CLIVE [11], KonIQ-10k [15], SPAQ [8], Liu13 (deblurring) [26], SRIQA-Bench (superresolution) [6], Min19 (dehazing) [32], and AGIQA-3K (image generation) [20].
The SRCC and PLCC results presented in Table 2 reveal several key observations. First, all VLMbased models outperform traditional and discriminative deep-learning-based ones, with the base Qwen2.5-VL-7B achieving an SRCC of 0.708 despite no IQA-specific training. This underscores Multi-Dataset Training Our RL2R approach enables multi-dataset training without the need for perceptual scale realignment. To exploit this, we train VisualQuality-R1 † on a combination of KADID-10K [23] and SPAQ [8] (again 6 : 2 : 2 split while ensuring content independence). As shown in Table 2, VisualQuality-R1 † yields consistent performance gains. Despite a minor dip in the image generation scenario, the average SRCC/PLCC rises from 0.777/0.814 to 0.791/0.831. In stark contrast, Q-Insight † [21] fails to benefit from multi-dataset training due to its inability to address perceptual scale variationsfoot_1 : KADID-10K uses ratings from 1 to 5, while SPAQ spans 0 to 100.
this section cite: ['b22', 'b6', 'b10', 'b14', 'b7', 'b25', 'b5', 'b31', 'b19', 'b22', 'b7', 'b20']

Section: Ablation Studies
Effect of K in GRPO We vary the number of generated responses, K, while keeping all other settings fixed during GRPO. Table 3 shows that reducing K from six (default) to four or five has only a marginal effect, offering a favorable trade-off between computational cost and accuracy.
Binary Reward vs. Continuous Fidelity Reward Table 4 shows that, within the same RL2R framework, our continuous fidelity reward generalizes better than the binary reward adopted in GRPO [36]. Moreover, both reward variants consistently outperform the regression-based Q-Insight [21], underscoring the effectiveness of our RL2R optimization.
this section cite: ['b35', 'b20']

Section: Thurstone Model Variants
To evaluate the effectiveness of mean quality computation in Eq. ( 1), we compare it with an alternative that averages probabilities across individual quality comparisons: We randomly select 20 images from each of CLIVE [11], KonIQ-10k [15], SRIQA-Bench [6], and AGIQA-3K [20]. At successive training steps, we generate multiple responses per image, compute the std of the predicted quality scores, and plot the average std across images. The uniformly downward trend confirms that VisualQuality-R1 becomes steadily more stable in assessing image quality as training progresses.
p k (x i , x j ) = 1 K K k ′ =1 p k,k ′ (x i , x j ) = 1 K K k ′ =1 Φ q k (x i ) -q k ′ (x j ) σ 2 (q(x i )) + σ 2 (q(x j )) + γ .(7)
As reported in Table 4, averaging quality scores rather than probabilities yields higher performance across distortion scenarios, indicating more reliable comparative probability estimates and reward assignments. Taking a step further, we fix the variances in Eq. ( 7) to one-reducing the model to Thurstone Case V [41]. The constant-variance simplification degrades performance on nearly all datasets. This provides a strong indication that sample variances are capable of capturing the perceptual difficulty of image pairs, thus improving comparison reliability and stabilizing fidelity reward computation. Together, these findings verify that RL2R effectively embeds the Thurstone model within GRPO.
this section cite: ['b10', 'b14', 'b5', 'b19', 'b40']

Section: Further Analysis
Predicted Score Variability over Iterations We randomly sample 20 images from each of CLIVE [11], KonIQ-10k [15], SRIQA-Bench (super-resolution) [6], and AGIQA-3K (image generation) [20], respectively. At successive training checkpoints, we generate multiple responses per image and compute the standard deviation (std) of the resulting K quality scores. As illustrated in Fig. 3, the std falls steadily across all datasets, indicating that predictions of VisualQuality-R1 become progressively more stable and confident.
this section cite: ['b10', 'b14', 'b5', 'b19']

Section: Visual Reasoning Evolution over Iterations
Fig. 4 tracks how the visual reasoning capabilities of VisualQuality-R1 mature over the course of training. The test image is super-resolved by SwinIR [22], which contains subtle, processing-related artifacts, making it an informative probe. Q-Insight notices that the image is "blurry" and "overexposed," but assigns an extremely low score (i.e., 2.00), indicating limited sensitivity to super-resolution artifacts. The base model Qwen2.5-VL-7B [1] swings to the opposite extreme: it praises the "clear details" and "vibrant colors," declares the absence of blur or noise, and outputs an inflated score (i.e., 4.80). The model clearly over-trusts superficial sharpness cues and misses hidden processing traces. In contrast, the proposed VisualQuality-R1 progressively refines its visual reasoning over iterations. At the 50-th step, it starts to suspect artificial stylization and questions the image's realism, yet it still values the apparent clarity. By the 200-th step, the description becomes more balanced. It acknowledges the level of detail and clarity, yielding a slightly higher but still cautious rating. At the last step, the explanation is now decidedly nuanced. VisualQuality-R1 attributes the remaining softness to possible filtering or to the object's inherent structure, labels the appearance "surreal," and reduces the score to 3.00, reflecting a judicious penalty for unnatural post-processing. In summary, RL2R guides VisualQuality-R1 from naïve, superficial remarks to sophisticated, human-aligned reasoning that correctly identifies subtle super-resolution artifacts and calibrates quality scores accordingly.
this section cite: ['b21', 'b0']

Section: Conclusion and Discussion
We have introduced VisualQuality-R1, a reasoning-induced NR-IQA model optimized via RL2R. Our approach is grounded in the intrinsic relativity of visual quality, seamlessly integrating the Thurstone model within GRPO to capture predictive uncertainty. By introducing the continuous fidelity reward, VisualQuality-R1 delivers more precise policy-gradient signals.
this section cite: []

Section: References
Ref_id:b0 Title: Qwen2.5-VL technical report Year: (2025)
Ref_id:b1 Title: Fast differentiable sorting and ranking Year: (2020)
Ref_id:b2 Title: Deep neural networks for no-reference and full-reference image quality assessment Year: (2017)
Ref_id:b3 Title: Learning to rank using gradient descent Year: (2005)
Ref_id:b4 Title: Learning to rank: From pairwise approach to listwise approach Year: (2007)
Ref_id:b5 Title: Toward generalized image quality assessment: Relaxing the perfect reference quality assumption Year: (2025)
Ref_id:b6 Title: No-reference blur assessment of digital pictures based on multifeature classifiers Year: (2010)
Ref_id:b7 Title: Perceptual quality assessment of smartphone photography Year: (2020)
Ref_id:b8 Title: DreamSim: Learning new dimensions of human visual similarity using synthetic data Year: (2023)
Ref_id:b9 Title: Learning to rank for blind image quality assessment Year: (2015)
Ref_id:b10 Title: Massive online crowdsourced study of subjective and objective picture quality Year: (2015)
Ref_id:b11 Title: Perceptual quality prediction on authentically distorted images using a bag of features approach Year: (2017)
Ref_id:b12 Title: DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning Year: (2025)
Ref_id:b13 Title: Human age estimation using bio-inspired features Year: (2009)
Ref_id:b14 Title: KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment Year: (2020)
Ref_id:b15 Title: Alec Radford, et al. GPT-4o system card Year: (2024)
Ref_id:b16 Title: Mallows and generalized mallows model for matchings Year: (2019)
Ref_id:b17 Title: Convolutional neural networks for no-reference image quality assessment Year: (2014)
Ref_id:b18 Title: MUSIQ: Multi-scale image quality Transformer Year: (2021)
Ref_id:b19 Title: AGIQA-3K: An open database for AI-generated image quality assessment Year: (2023)
Ref_id:b20 Title: Q-Insight: Understanding image quality via visual reinforcement learning Year: (2025)
Ref_id:b21 Title: Image restoration using Swin Transformer Year: (2021)
Ref_id:b22 Title: KADID-10K: A large-scale artificially distorted IQA database Year: (2019)
Ref_id:b23 Title: A no-reference metric for perceived ringing artifacts in images Year: (2009)
Ref_id:b24 Title: RankIQA: Learning from rankings for noreference image quality assessment Year: (2017)
Ref_id:b25 Title: A no-reference metric for evaluating the quality of motion deblurring Year: (2013)
Ref_id:b26 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b27 Title: Group maximum differentiation competition: Model comparison with few samples Year: (2018)
Ref_id:b28 Title: dipIQ: Blind image quality assessment by learning-to-rank discriminable image pairs Year: (2017)
Ref_id:b29 Title: End-to-end blind image quality assessment using deep neural networks Year: (2017)
Ref_id:b30 Title: Consolidated dataset and metrics for high-dynamic-range image quality Year: (2021)
Ref_id:b31 Title: Quality evaluation of image dehazing methods using synthetic hazy images Year: (2019)
Ref_id:b32 Title: No-reference image quality assessment in the spatial domain Year: (2012)
Ref_id:b33 Title: Making a "completely blind" image quality analyzer Year: (2012)
Ref_id:b34 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b35 Title: DeepSeekMath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b36 Title: Defining and characterizing reward gaming Year: (2022)
Ref_id:b37 Title: Models for distributions on permutations Year: (1990)
Ref_id:b38 Title: NIMA: Neural image assessment Year: (2018)
Ref_id:b39 Title: Content-based photo quality assessment Year: (2013)
Ref_id:b40 Title: A law of comparative judgment Year: (1927)
Ref_id:b41 Title: FRank: A ranking method with fidelity loss Year: (2007)
Ref_id:b42 Title: Active fine-tuning from gMAD examples improves blind image quality assessment Year: (2021)
Ref_id:b43 Title: Troubleshooting blind image quality models in the wild Year: (2021)
Ref_id:b44 Title: Reduced-and no-reference image quality assessment Year: (2011)
Ref_id:b45 Title: Blind measurement of blocking artifacts in images Year: (2000)
Ref_id:b46 Title: Image quality assessment: From error visibility to structural similarity Year: (2004)
Ref_id:b47 Title: Local phase coherence and the perception of blur Year: (2003)
Ref_id:b48 Title: Q-Instruct: Improving low-level visual abilities for multi-modality foundation models Year: (2024)
Ref_id:b49 Title: Q-ALIGN: Teaching LMMs for visual scoring via discrete text-defined levels Year: (2024)
Ref_id:b50 Title: A comprehensive study of multimodal large language models for image quality assessment Year: (2024)
Ref_id:b51 Title: Multi-sequence network for blind omnidirectional image quality assessment Year: (2023)
Ref_id:b52 Title: Boosting image quality assessment through efficient Transformer adaptation with local feature enhancement Year: (2024)
Ref_id:b53 Title: MANIQA: Multi-dimension attention network for no-reference image quality assessment Year: (2022)
Ref_id:b54 Title: Teaching large language models to regress accurate image quality scores using score distribution Year: (2025)
Ref_id:b55 Title: Depicting beyond scores: Advancing image quality assessment through multi-modal language models Year: (2024)
Ref_id:b56 Title: Continual learning for blind image quality assessment Year: (2022)
Ref_id:b57 Title: Uncertainty-aware blind image quality assessment in the laboratory and wild Year: (2021)
Ref_id:b58 Title: Blind image quality assessment via vision-language correspondence: A multitask learning perspective Year: (2023)
Ref_id:b59 Title: Adaptive image quality assessment via teaching large multimodal model to compare Year: (2024)
