Title: Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2
Abstract: Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can be directly transferred to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: directional guidance from the prompt and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For cross-prompt transferability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms state-of-the-art (SOTA) attacks by a large margin. Our codes are available at: https://github.com/CGCL-codes/UAP-SAM2.

Section: Introduction
Recent advances in deep learning have led to the emergence of large segmentation foundation models [4,16,33,40,43] with impressive generalization capabilities, enabling object segmentation from unseen images. Among them, Segment Anything Model (SAM) [4] can output a class-free mask by leveraging prompts (e.g., points or boxes) to accurately localize target objects. Despite its strong segmentation ability, SAM is limited to images. Therefore, SAM2 [28] is recently proposed to integrate a memory mechanism to store features of previous frames, extending SAM to generalpurpose video segmentation. Given a prompt (typically on the first frame), SAM2 can continuously track and segment the target object across subsequent frames [11,32,49].  Deep neural networks (DNNs) are known to be vulnerable to adversarial examples [34,35,46], where imperceptible perturbations lead to incorrect predictions. Recent studies show that such perturbations can significantly impair the ability of the SAM to segment target objects. Attack-SAM [42] applies PGD [23] to generate sample-wise perturbations for each image. DarkSAM [48] introduces a spatialfrequency universal attack framework, crafting universal adversarial perturbations (UAPs) [30,44,45] that generalize across diverse images. Given the prompt sensitivity of SAM-based models, recent works [14,21,48] also design adversarial examples with cross-prompt transferability. However, despite their effectiveness on SAM, the robustness of SAM2 against these attacks remains unexplored.
Motivated by existing video attacks [36,37], we investigate whether existing attacks designed for SAM can be directly transferred to SAM2. We evaluate several representative methods, including PGD [23], Attack-SAM [42] (A-SAM), S-RA [29], UAD [22], and DarkSAM [48] on the YouTube [41] and MOSE [6] datasets with ϵ = 10/255. As shown in Fig. 1, existing attacks effectively fool SAM but fail to deceive SAM2. For example, DarkSAM reduces the average segmentation performance of SAM by 98.25% relative to its original performance on two datasets, while causing only a 22.26% drop in SAM2. These results highlight the difficulty of directly transferring attacks from SAM to SAM2.
Given the shared design philosophy between SAM and SAM2, we conduct a comprehensive analysis to uncover the underlying causes of their performance gap in Sec. 3.1. To achieve consistent and accurate segmentation across frames, SAM2 stores the user-provided prompt as a persistent, videospecific representation. It also maintains a memory bank that caches semantic features from some of the previous frames. During inference, SAM2 jointly leverages the prompt and memory bank to guide the segmentation of each frame, repeating this process throughout the sequence. Our findings in Sec. 3.1 highlight two key factors behind the failure of existing attacks: (1) directional guidance from the prompt, and (2) semantic entanglement across consecutive frames. To effectively attack SAM2, we establish two key objectives. First, the perturbation must generalize across diverse prompts to maintain attack effectiveness. Second, since videos contain numerous frames, we aim to craft a UAP rather than sample-wise perturbations that applies to any frame across different videos.
In this paper, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. It generates a UAP that generalizes across videos, frames, and prompts, effectively preventing SAM2 from segmenting target objects, making them vanish into thin air. For the first objective, we begin by designing a target-scanning strategy that divides each frame into m regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. Moreover, instead of directly attacking the prompt-dependent masks, we disrupt the semantic features produced by the image encoder to improve the cross-prompt transferability. For the second objective, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Specifically, we design a semantic confusion attack to hinder SAM2's understanding of target objects by injecting noise into the semantic space, a feature shift attack to maximize the semantic distance between adversarial and benign frames, and a memory misalignment attack to amplify inter-frame semantic inconsistency by breaking temporal alignment.
We conduct a comprehensive evaluation of UAP-SAM2 and its sample-wise variant UAP-SAM2 * across six datasets spanning both video and image segmentation tasks. The comparative experiments demonstrate that our approach significantly outperforms SOTA attacks against SAM2. Additionally, defense experiments further validate the robustness of our proposed method. Our main contributions are summarized as follows:
• We propose the first cross-prompt universal adversarial attack against SAM2, revealing the vulnerability of video segmentation foundation models. By designing a UAP, our method consistently misleads SAM2's segmentation across videos, frames, and prompts.
• We design a brand-new dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames.
• We conduct extensive experiments on six datasets across two segmentation tasks to demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms SOTA attacks by a large margin.
this section cite: ['b3', 'b15', 'b32', 'b39', 'b42', 'b3', 'b27', 'b10', 'b31', 'b48', 'b33', 'b34', 'b45', 'b41', 'b22', 'b47', 'b29', 'b43', 'b44', 'b13', 'b20', 'b47', 'b35', 'b36', 'b22', 'b41', 'b28', 'b21', 'b47', 'b40', 'b5']

Section: Preliminaries
Given an input sequence of frames X = {x i } N i=1 and prompts
P = {p i } L i=1 , the SAM2 f θ (•) predicts segmentation masks Y = {y i } N
i=1 for each frame x i . For a frame x i , a pixel at coordinates (m, n), denoted as x mn i , is considered part of the masked region if its corresponding mask value y mn i exceeds a predefined threshold of zero. SAM2 consists of an image encoder E img that encodes each frame x i into a feature embedding F i = E img (x i ); a prompt encoder E prompt processes the input prompt p i and produces the corresponding embedding Q i = E prompt (p i ); a memory bank M i stores the past K embeddings E i preceding frame x i . A memory attention module A integrates F i , M i , and Q i to generate an enhanced representation. Finally, a mask decoder D takes this representation and predicts the segmentation mask y i . We can simplify the above process as follows:
Y = f θ (X , P)(1)
Following [42,48], we assume that attackers are able to obtain the open-source SAM2 and can collect publicly available datasets from the Internet to make adversarial examples. The attackers' objective is to craft an adversarial perturbation δ for each frame that prevents SAM2 from accurately segmenting target objects across different prompts, i.e., a cross-prompt universal adversarial attack. Furthermore, δ should be sufficiently small and constrained by the l p norm of the predefined perturbation magnitude ϵ. Next, we formally define this type of attack.
Definition 2.1 (Cross-prompt universal adversarial attacks for SAM2). For an input sequence of frames X , we generate a UAP δ for each frame x i ∈ X to shift its predicted mask away from its ground truth y i under different prompt P. This problem can be formulated as:
min δ E xi∼X [∀p i ∈ P, IoU (f θ (x i + δ, p i ), y i )] , s.t. ∥δ∥ p ≤ ϵ(2)
In this paper, we evaluate UAP-SAM2 on both video and image segmentation tasks, with a primary focus on UAPs. For fair comparison with existing work, we also adapt our proposed approach into a sample-wise form UAP-SAM2 * without modifying the loss function.
this section cite: ['b41', 'b47']

Section: Methodology

this section cite: []

Section: Observation and Design Philosophy
Motivated by the significant performance gap between SAM and SAM2 under existing attacks shown in Fig. 1, we investigate the underlying causes by examining their architectural differences. We then hope to design an effective UAP for the emerging SAM2 based on our findings.
Observation I: First-frame attacks fail to transfer to later frames. The first design difference lies in the prompting strategy. Unlike SAM, which provides a sample-level prompt for each frame, SAM2 offers only an initial prompt on the first frame, which is then stored and reused for segmenting subsequent frames. Existing video attacks [37] show that image-level perturbations can transfer to video, a natural idea is to attack only the first frame and examine the effect on later frames. We apply DarkSAM as a one-shot attack to the first frame in the YouTube dataset to evaluate its effectiveness against SAM2. After generating adversarial perturbations on the first frame, we add them to all
Result ퟏ ퟐ�� ퟏ ퟐ�� ퟑퟐ ퟐ�� Benign Frame1 Frame k Frame k-1 (a) (b) Frame 1 Frame k-2 Frame k-1 Frame k (Adv)  Attack only the k-th frame ...
퐌퐞퐦 Feature ... subsequent frames and assess the attack's impact. We gradually increase the perturbation budget from 10/255 to 32/255 to investigate the impact of attack strength. As depicted in Fig. 2 (a), even at the highest budget of 32/255, DarkSAM still fails to significantly degrade segmentation performance. This may be attributed to directional guidance from the prompt and the enhanced robustness of SAM2, likely due to its advanced architecture and diverse training data. Moreover, perturbations that fail to disrupt the first frame typically cannot be effectively transferred to subsequent frames.
this section cite: ['b36']

Section: Insight I.
Although attacking the first frame can somewhat mislead SAM2 across the video sequence, its effectiveness is limited. This motivates us to investigate other design differences to improve attack efficacy.
Observation II: Joint modeling of past and current frames hinders frame-specific attacks.
According to [28], besides using a fixed prompt from the first frame, SAM2 maintains a memory bank that stores semantic features from the past k frames. A memory attention module integrates these features to guide the segmentation of the current frame. We refer to this use of both historical context and current-frame features as a dual-guidance mechanism. 0 1 2 3 4 5 6 7 8 9 0.4 0.5 0.6 0.7 0.8 0.9 1.0
this section cite: ['b27']

Section: Cosine Similarity

this section cite: []

Section: First Sim Adjacent Sim

this section cite: []

Section: Figure 3: Avalanche effect
To evaluate its impact, we inject adversarial noise into a randomly selected middle frame from the YouTube dataset. We then visualize the features extracted by the image encoder from preceding frames and stored in the memory bank. As illustrated in the 1st and 2nd rows of Fig. 2 (b), attacking a single frame alone does not significantly degrade SAM2's segmentation accuracy due to semantic entanglement across consecutive frames. However, the 3rd and 4th rows show that perturbing features from past frames to disrupt the memory bank noticeably impairs segmentation performance on the current frame. To probe the vulnerability of this dual-guidance mechanism, we examine two complementary perspectives: semantic misalignment in the current frame and semantic discontinuity across adjacent frames. We segment each frame by thresholding the output masks into foreground and background. We focus on forcing SAM2 to misinterpret foreground objects as background while simultaneously enhancing background saliency to confuse the current-frame features. Fig. 2 already suggests that single-frame attacks are insufficient due to the memory module's influence. We therefore extend the attack by injecting a UAP across consecutive frames to amplify inter-frame semantic gaps. From Fig. 3, this results in a progressive decline in similarity between the current frame and both adjacent frames and the first frame. We refer to this finding as the avalanche effect phenomenon.
this section cite: []

Section: Insight II.
Attacking only a single frame is ineffective in the presence of memory-based guidance. Instead, disrupting both the current semantics and temporal consistency across frames creates a stronger mismatch between guidance and segmentation, undermining SAM2's understanding of video content.
this section cite: []

Section: Vision Encoder Memory Attention
Mask Decoder Prompt Encoder
this section cite: []

Section: UAP-SAM2: A Complete Illustration
Based on the design philosophy outlined in Sec. 3.1, we construct our attack from two perspectives: (i) semantic distortion within the current frame and (ii) semantic discontinuity across consecutive frames.
For the current frame, we enhance attack effectiveness by jointly introducing semantic confusion and feature shift. To reduce our method's reliance on a specific prompt, we design a target-scanning strategy that selects random prompts during optimization. Specifically, we divide each video frame evenly into m regions and randomly generate one prompt per region. Moreover, our optimization primarily targets the output features of the image encoder, whose input is only the image.
In this section, we present UAP-SAM2, a novel cross-prompt universal adversarial attack driven by dual semantic deviation against SAM2. As depicted the in Fig. 4, the UAP-SAM2 pipeline implements a memory-misalignment attack J ma to disrupt temporal guidance, a feature shift attack J f a to distort local representations, and a semantic confusion attack J sa to confuse object-level semantics. The overall optimization objective of UAP-SAM2 is as follow:
J total = J sa + J f a + J ma(3)
Semantic confusion attack. We apply a binary mask m + to separate the object from the background in each frame. Similar to prior attacks [48] on SAM, we aim to mislead the model by optimizing the foreground region to resemble the background. Meanwhile, we further shift foreground pixels near the decision boundary toward the background class, while reinforcing pixels originally identified as background to preserve their classification. By adding the UAP to the target frame x i , we obtain the adversarial frame xi . This objective can be formalized as:
J sa = 1 N N i=1 [(f θ (x i , P) • m + -y -)] 2 + [((1 -f θ (x i , P)) • m --y -)] 2(4)
where y -is a mask that matches the frame shape, containing threshold values (e.g.-1) in regions corresponding to the target objects, and 0 elsewhere. m -represents a binary mask for the background regions in each frame, which is the opposite of the mask that highlights the foreground.
To further effectively confuse the foreground and background, we use the binary cross-entropy (BCE) loss function, treating logits close to zero as having low confidence in the model's classification.
In contrast, the larger the absolute value of the logits (whether positive or negative), the higher the model's confidence in its prediction. To enhance the attack's effectiveness, we increase the overall confidence of pixel positions, naturally strengthening the updates on pixels near the decision boundary (i.e., logits close to 0), thereby pushing their logits toward the background and achieving a stronger confusion effect. Accordingly, we define the semantic confusion attack J sa as follow:
J sa = 1 N N i=1 [BCE (f θ (x i , P) • m + , y -) + BCE ((1 -f θ (x i , P)) • m -, y -)](5)
Feature shift attack. We optimize the UAP to minimize the similarity between the features of the perturbed and benign frames extracted by SAM2's image encoder. We formalize this as:
J f a = - 1 N N i=1 cos (E img (x i )E img (x i )))(6)
To further increase the feature discrepancy between adversarial and benign frames, we adopt a contrastive learning [2] approach. We first apply ρ times random augmentations T (•) to the target frame and aggregate their features into a prototype through
e i = 1 ρ ρ j=1 E img (T (x i )).
We then treat the adversarial frame xi and the prototype e i of the original frame as a negative pair, while randomly sampling frames from other videos as positive pairs. By maximizing the distance between the xi and e i and minimizing the distance among positive samples, we effectively drive the adversarial features away from their original semantics. Therefore, we can obtain J f a :
J f a = - 1 N N i=1 log exp (cos (E img (x i ), e i )) /τ ) N k=1 1 k̸ =i exp (cos (E img (x i ), E img (x k ))) /τ )(7)
where 1 k̸ =i is an indicator function and τ denotes a temperature parameter.
Memory misalignment attack. Starting from the second frame, we disrupt the memory bank in SAM2 by maximizing the feature discrepancy between consecutive adversarial frames. By progressively increasing the semantic difference between the current adversarial frame and the previous one, we induce the avalanche effect illustrated in Fig. 3. This process is formulated as:
J ma = - 1 N N i=1 cos (E img (x i+1 ), E img (x i )))(8)
4 Experiments
this section cite: ['b47', 'b1']

Section: Experimental Setup
Datasets and models. We evaluate our attack on there public video segmentation datasets: YouTube-VOS2018 (YouTube) [8], DAVIS 2017 (DAVIS) [26], and MOSE [6] for video segmentation tasks.
To further investigate the performance of UAP-SAM2 on image segmentation tasks, we construct corresponding image-based datasets by randomly sampling frames from the original video datasets, denoted as YouTube * , DAVIS * , and MOSE * . We resize all frames in the videos to a uniform size of 3×1024×1024. We use pre-trained SAM2-T, SAM2-S, and SAM2.1-T from the official repository as the target models. To further validate the transferability, we will conduct evaluations on Sam2long [7], which enhances the capabilities of SAM2 in long-video tasks. More details are in Appendix-B.
this section cite: ['b7', 'b25', 'b5', 'b6']

Section: Attack settings.
We set the perturbation bound ϵ of the universal adversarial attack UAP-SAM2 to 10/255, and that of the sample-wise variant attack UAP-SAM2 * to 8/255, using a batch size of 1 and training for 10 epochs. We use a fixed random seed of 30 for all experiments to ensure reproducibility. We use point prompts for the default evaluation.
Evaluation metrics. Following [42,48], we use the mean Intersection over Union (mIoU) metric to evaluate the effectiveness of UAP-SAM2. mIoU is a widely used metric in segmentation tasks [4,24,28] that measures the average overlap between the predicted and ground truth segmentation masks. A lower mIoU value indicates better attack performance.
M 1 M 2 M 3 M 1 M 2 M 3 M 1 M 2 M 3 M 1 M 2 M 3 UAP D 137
.03 36.26 42.47 45.17 28.39 41.06 27.54 28.36 21.89 52.07 27.19 42.36 D 2 42.47 40.85 52.08 49.03 38.28 48.47 48.45 47.20 46.47 44.51 37.32 45.83 D 3 33.67 37.03 52.46 49.13 34.38 49.66 50.13 52.22 49.16 61.63 40.28 52.03 AVG 37.72 38.05 49.00 47.78 33.68 46.40 42.04 42.59 39.17 52.74 34.93 46.74 Sample-wise D 1 45.39 27.72 39.02 57.72 55.18 61.57 33.42 28.49 34.19 32.42 29.28 37.93 D 2 41.64 23.57 43.24 50.26 43.37 51.73 36.92 31.44 39.21 35.62 30.81 37.65 D 3 46.60 35.91 49.17 60.69 52.88 60.89 45.81 38.42 42.51 48.13 43.44 50.71 AVG 44.54 29.07 43.81 56.22 50.48 58.06 38.72 32.78 38.64 38.72 34.51 42.10 Platform. Experimental hardware details. We conduct experiments on a machine with two NVIDIA A100-SXM4 GPUs, two Intel(R) Xeon(R) Gold 6132 CPUs and 314GB RAM.
this section cite: ['b41', 'b47', 'b3', 'b23', 'b27']

Section: Attack Performance
To comprehensively evaluate the effectiveness of UAP-SAM2, we conduct experiments on six datasets (YouTube, DAVIS, MOSE, YouTube * , DAVIS * , and MOSE * ) covering both video and image segmentation tasks. We evaluate three model variants: SAM2-T, SAM2-S, and SAM2.1-T. For clarity, we denote the datasets (including their corresponding variant datasets) as D 1 -D 3 and the models as M 1 -M 3 throughout the paper. Across six datasets for both image and video segmentation tasks, SAM2 achieves an average mIoU above 76%, demonstrating strong segmentation capability and generalization. Tab. 1 shows that adversarial examples generated by UAP-SAM2 consistently and significantly degrade SAM2's performance with cross-prompt transferability. Notably, on the DAVIS dataset with point prompts, UAP-SAM2 and its variant UAP-SAM2 * reduce SAM2's mIoU by over 45.79% and 54.77%, respectively. As presented in Tab. 1, regardless of whether point or box prompts are used, our method consistently achieves stronger attack performance on video segmentation than on image segmentation. This further validates its effectiveness in disrupting semantic consistency across consecutive frames.
We further evaluate the transferability of our approach across different datasets and models.
this section cite: []

Section: Comparison Study
Given the absence of adversarial attacks tailored for SAM2, we evaluate our method comprehensively by comparing UAP-SAM2 against the latest adversarial attacks targeting SAM, such as Attack-SAM [42], S-RA [29], UAD [22], and DarkSAM [48]. We further compare UAP-SAM2 against representative adversarial attack methods originally designed for classification, image segmentation, and video segmentation tasks, including UAPGD [5], SegPGD [10], and VOSPGD [13]. To ensure a fair comparison, we adapt all baseline methods into a universal adversarial attack framework and apply the same optimization settings as used in UAP-SAM2. To evaluate the cross-prompt generalization ability of these methods, we uniformly adopt random prompts, i.e., the prompts used during training and testing are different, for method optimization to generate adversarial examples. We choose SAM2-T as the target model and evaluate the performance of all methods on both image and video segmentation tasks across six datasets. As depicted in Tab. 2, UAP-SAM2 outperforms all existing attacks on video segmentation across three datasets. For image segmentation, our method also surpasses most baselines. The above results can be attributed to the tailored design of video features in our method.
this section cite: ['b41', 'b28', 'b21', 'b47', 'b4', 'b9', 'b12']

Section: Ablation Study
In this section, we investigate the different factors on the attack performance of UAP-SAM2. We use SAM2-T and SAM2-S as the target models, with DAVIS as the dataset.
The effect of the module. We perform ablation studies to assess the contribution of individual components to the attack effectiveness of UAP-SAM2. For clarity, we denote L sa , L f a , and L ma as A, B, and C, respectively. As observed in Fig. 7 (a), none of the ablated variants surpass the full model, highlighting the importance of each module in achieving optimal attack performance.
The effect of prompt numbers. We investigate how the prompt number m of segmented regions in the proposed target-scanning strategy influences the attack performance of UAP-SAM2. We vary the region count from 8 to 512 and report the results in Fig. 7 (b). The attack achieves optimal performance under both settings when m = 256, which we adopt as the default configuration.
The effect of evaluation modes. We study how different evaluation prompt settings affect the attack performance of UAP-SAM2. Specifically, we evaluate the perturbation generated on SAM2-T using five randomly sampled box prompts (B1 -B5) and five point prompts (P1 -P5). As depicted in Fig. 7 (c) -(d), UAP-SAM2 consistently maintains strong performance across different prompt configurations, demonstrating the robustness of our method.
The effect of iteration numbers. We investigate the effect of the iteration numbers on attack performance of UAP-SAM2. We conduct experiments with varying numbers of iterations, ranging from 1 to 20. The results shown in Fig. 7 (e) indicate that the attack performance stabilizes after the number of iterations reaches 10. Therefore, we set it as the default configuration for our experiments.
The effect of ϵ. We evaluate UAP-SAM2's performance with ϵ from 2/255 to 32/255 in Fig. 7 (f). With the increase in ϵ, there is a corresponding enhancement in attack performance. Notably, even at the 4/255 setting, our method still maintains high attack efficacy, with an average mIoU decrease of over 33.08%.
The effect of negative sample numbers. We explore the effect of varying the number of negative samples from 10 to 100 on the performance of UAP-SAM2 in Fig. 7 (g). Considering both computational efficiency and attack effectiveness, we set 30 as our default testing.
The effect of testing frame numbers. We study the effect of the number of selected frames per video on the performance of UAP-SAM2. As shown in Fig. 7 (h), the results with 15 frames are comparable to those using all frames. Therefore, for efficiency considerations, we set 15 frames as the default configuration for our experiments.
this section cite: []

Section: Defense Study
Since there is no dedicated adversarial defense tailored for SAM2, we explore the robustness of UAP-SAM2 through two common defense strategies: model pruning [50] and data pre-processing [27].
Model pruning is a widely used compression technique that removes redundant parameters to simplify network complexity, thereby potentially reducing sensitivity to perturbations. We evaluate the attack performance under various pruning ratios ranging from 0 to 0.9 on DAVIS. As shown in Fig. 8 (a) -(b), the mIoU of benign samples consistently decreases as the pruning ratio increases, whereas that of adversarial examples remains relatively stable. Notably, even when the pruning ratio reaches 0.4, the performance of benign samples degrades significantly, while the mIoU of adversarial examples remains largely unaffected. These results suggest that model pruning offers limited robustness against UAP-SAM2.
Data pre-processing suppresses the impact of adversarial noise by introducing distortions such as occlusion or blur into the image. We apply spatter (sp_) and saturate (sa_) corruption at severity levels from 0 to 5 to assess the effectiveness of this strategy on DAVIS. As observed in Fig. 8 (c) -(d), increasing the corruption strength leads to a consistent drop in benign sample mIoU, while adversarial mIoU remains largely unaffected. These findings suggest that our method remains effective even when facing pre-processing defenses based on input corruption.
6 Related Works
this section cite: ['b49', 'b26']

Section: Segment Anything Models
Segment Anything Model (SAM) [17] has achieved remarkable success in image segmentation due to its strong generalization ability. SAM2 [28], the latest improved version, is applied to both image and video segmentation tasks through the memory mechanism, extending SAM to general-purpose video segmentation. The user only needs to provide a prompt on the first frame, and SAM2 can then perform real-time object localization and segmentation in subsequent frames. Building on its strong generalization, recent studies [1,3,7,15,19,32,39,49] develop task-specific variants of SAM to better address various downstream applications. SAM2 has been rapidly applied to various downstream tasks, such as medical video segmentation [49], 3D segmentation [11], and camouflaged object detection [32].
this section cite: ['b16', 'b27', 'b0', 'b2', 'b6', 'b14', 'b18', 'b31', 'b38', 'b48', 'b48', 'b10', 'b31']

Section: Adversarial Attacks on SAM
Recent studies [12,20,21,29,38,42,48] reveal that the SAM is vulnerable to adversarial examples [9,23,25,47,18,31,44], which are crafted by adding imperceptible perturbations to induce incorrect predictions. Existing adversarial attacks on SAM fall into two categories: sample-wise, which tailor perturbations to each input, and universal, which create a single perturbation that works across many images. Attack-SAM [42] is the first to adopt PGD [23] to manipulate the predicted masks of image-prompt pairs. UAD [22] extends this direction by simulating spatial deformations to optimize adversarial noise that disrupts the feature representations of the image encoder, enabling prompt-free attacks. In parallel, DarkSAM [48] introduces the first universal adversarial attack against SAM. It designs a hybrid spatial-frequency framework that prevents objects in the image from being segmented and proposes a shadow target strategy to improve cross-prompt transferability. Other studies [20,29] focus on localized attacks that deceive SAM into failing to segment specific objects within an image. Despite their effectiveness on SAM, these methods cannot be directly applied to SAM2 due to the modality gap between images and videos, and the architectural novelties of SAM2.
this section cite: ['b11', 'b19', 'b20', 'b28', 'b37', 'b41', 'b47', 'b8', 'b22', 'b24', 'b46', 'b17', 'b30', 'b43', 'b41', 'b22', 'b21', 'b47', 'b19', 'b28']

Section: Conclusions, Limitations, and Broader Impact
In this paper, we investigate the performance gap of existing attacks against SAM and SAM2, and attribute it to two key challenges: directional guidance from the prompt and semantic entanglement across consecutive frames. To this end, we propose UAP-SAM2, the first cross-prompt universal adversarial attack driven by dual semantic deviation against SAM2. We design a target-scanning strategy and directly perturb the output features of the image encoder to enhance the cross-prompt transferability. To further boost the attack effectiveness, we jointly exploit semantic confusion and feature deviation. We conduct extensive experiments on six datasets across two segmentation tasks to demonstrate the effectiveness of the proposed method for SAM2.
While our work focuses on prompt-based video segmentation models, a potential limitation is that UAP-SAM2 may not directly generalize to traditional segmentation models, as their outputs are not label-free masks. Although SAM-based models are gaining popularity, it remains unclear how well adversarial examples crafted for SAM2 transfer to other segmentation frameworks. As these models are increasingly adopted in safety-critical applications such as autonomous driving and medical imaging, understanding their vulnerabilities is a crucial direction for future research. Justification: See Sec. 7.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: References
Ref_id:b0 Title: Fs-medsam2: Exploring the potential of sam2 for few-shot medical image segmentation without fine-tuning Year: (2024)
Ref_id:b1 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b2 Title: Sam2-adapter: Evaluating & adapting segment anything 2 in downstream tasks: Camouflage, shadow, medical image segmentation, and more Year: (2024)
Ref_id:b3 Title: Sam-adapter: Adapting segment anything in underperformed scenes Year: (2023)
Ref_id:b4 Title: Universal adversarial attack via enhanced projected gradient descent Year: (2020)
Ref_id:b5 Title: Mose: A new dataset for video object segmentation in complex scenes Year: (2023)
Ref_id:b6 Title: Sam2long: Enhancing sam 2 for long video segmentation with a training-free memory tree Year: (2024)
Ref_id:b7 Title: Capsulevos: Semi-supervised video object segmentation using capsule routing Year: (2019)
Ref_id:b8 Title: Explaining and harnessing adversarial examples Year: (2015)
Ref_id:b9 Title: Segpgd: An effective and efficient adversarial attack for evaluating and boosting segmentation robustness Year: (2022)
Ref_id:b10 Title: Segment any 3d as videos in zero-shot and promptable manners Year: (2024)
Ref_id:b11 Title: On the robustness of segment anything Year: (2023)
Ref_id:b12 Title: Exploring the adversarial robustness of video object segmentation via one-shot adversarial attacks Year: (2023)
Ref_id:b13 Title: Cross-point adversarial attack based on feature neighborhood disruption against segment anything model Year: (2024)
Ref_id:b14 Title: Sam2 for image and video segmentation: A comprehensive survey Year: (2025)
Ref_id:b15 Title: Segment anything in high quality Year: ()
Ref_id:b16 Title: Segment anything Year: (2023)
Ref_id:b17 Title: Transferable adversarial facial images for privacy protection Year: (2024)
Ref_id:b18 Title: Surgical sam 2: Real-time segment anything in surgical video by efficient frame pruning Year: (2024)
Ref_id:b19 Title: Region-guided attack on the segment anything model Year: ()
Ref_id:b20 Title: Cross-prompt adversarial attack on segment anything model Year: ()
Ref_id:b21 Title: Unsegment anything by simulating deformation Year: (2024)
Ref_id:b22 Title: Towards deep learning models resistant to adversarial attacks Year: (2017)
Ref_id:b23 Title: Image segmentation using deep learning: A survey Year: (2021)
Ref_id:b24 Title: Universal adversarial perturbations Year: (2017)
Ref_id:b25 Title: The 2017 davis challenge on video object segmentation Year: (2017)
Ref_id:b26 Title: Deflecting adversarial attacks with pixel deflection Year: (2018)
Ref_id:b27 Title: Sam 2: Segment anything in images and videos Year: ()
Ref_id:b28 Title: Practical region-level attack against segment anything models Year: (2024)
Ref_id:b29 Title: Pb-uap: Hybrid universal adversarial attack for image segmentation Year: ()
Ref_id:b30 Title: Segtrans: Transferable adversarial examples for segmentation models Year: (2025)
Ref_id:b31 Title: Evaluating sam2's role in camouflaged object detection: From sam to sam2 Year: (2024)
Ref_id:b32 Title: Can sam segment anything? when sam meets camouflaged object detection Year: (2023)
Ref_id:b33 Title: Breaking barriers in physical-world adversarial examples: Improving robustness and transferability via robust feature Year: ()
Ref_id:b34 Title: Advedm: Fine-grained adversarial attack against vlm-based embodied agents Year: ()
Ref_id:b35 Title: Transferable adversarial attacks for image and video object detection Year: (2018)
Ref_id:b36 Title: Cross-modal transferable adversarial attacks from images to videos Year: (2022)
Ref_id:b37 Title: Transferable adversarial attacks on sam and its downstream models Year: ()
Ref_id:b38 Title: Sam2-unet: Segment anything 2 makes strong encoder for natural and medical image segmentation Year: (2024)
Ref_id:b39 Title: Efficientsam: Leveraged masked image pretraining for efficient segment anything Year: (2023)
Ref_id:b40 Title: Youtube-vos: A large-scale video object segmentation benchmark Year: (2018)
Ref_id:b41 Title: Sung-Ho Bae, and In So Kweon. Attack-sam: Towards evaluating adversarial robustness of segment anything model Year: (2023)
Ref_id:b42 Title: Personalize segment anything model with one shot Year: (2023)
Ref_id:b43 Title: Advclip: Downstream-agnostic adversarial examples in multimodal contrastive learning Year: (2023)
Ref_id:b44 Title: Downstream-agnostic adversarial examples Year: (2023)
Ref_id:b45 Title: Numbod: A spatial-frequency fusion attack against object detectors Year: ()
Ref_id:b46 Title: Securely fine-tuning pre-trained encoders against adversarial examples Year: ()
Ref_id:b47 Title: Darksam: Fooling segment anything model to segment nothing Year: ()
Ref_id:b48 Title: Medical sam 2: Segment medical images as video via segment anything model 2 Year: (2024)
Ref_id:b49 Title: To prune, or not to prune: exploring the efficacy of pruning for model compression Year: (2017)
