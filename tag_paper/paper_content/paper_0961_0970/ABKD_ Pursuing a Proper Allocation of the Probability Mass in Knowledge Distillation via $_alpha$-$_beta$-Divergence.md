Title: ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence
Abstract: Knowledge Distillation (KD) transfers knowledge from a large teacher model to a smaller student model by minimizing the divergence between their output distributions, typically using forward Kullback-Leibler divergence (FKLD) or reverse KLD (RKLD). It has become an effective training paradigm due to the broader supervision information provided by the teacher distribution compared to one-hot labels. We identify that the core challenge in KD lies in balancing two mode-concentration effects: the Hardness-Concentration effect, which refers to focusing on modes with large errors, and the Confidence-Concentration effect, which refers to focusing on modes with high student confidence. Through an analysis of how probabilities are reassigned during gradient updates, we observe that these two effects are entangled in FKLD and RKLD, but in extreme forms. Specifically, both are too weak in FKLD, causing the student to fail to concentrate on the target class. In contrast, both are too strong in RKLD, causing the student to overly emphasize the target class while ignoring the broader distributional information from the teacher. To address this imbalance, we propose ABKD, a generic framework with α-β-divergence. Our theoretical results show that ABKD offers a smooth interpolation between FKLD and RKLD, achieving an effective trade-off between these effects. Extensive experiments on 17 language/vision datasets with 12 teacher-student settings confirm

Section: Introduction
Knowledge Distillation (KD) (Hinton, 2015) is a widelyadopted technique for transferring knowledge from large models (teachers) to smaller models (students). In this setup, the student model, with a predictive distribution q θ , learns to mimic the predictive distribution p of the teacher model. This imitation is typically achieved by minimizing a predefined divergence D between the teacher distribution p and the student distribution q θ : ℓ KD ≜ D(p∥q θ ). This way, KD allows the student to leverage richer soft label information from p compared to one-hot labels, often leading to better performance than traditional supervised fine-tuning. This has been shown in tasks like image classification (Dosovitskiy, 2020; Radford et al., 2021;Yang et al., 2023b;Wang et al., 2022b) and text generation (Vaswani, 2017;Touvron et al., 2023a).
A key step in KD is to choose a proper divergence D for distribution matching. One popular choice in previous works (Cho & Hariharan, 2019;Mirzadeh et al., 2020;Zhou et al., 2021;Zhao et al., 2022;Jin et al., 2023;Sun et al., 2024;Zheng & Yang, 2024) is the forward Kullback-Leibler divergence (FKLD). However, FKLD's asymmetry often results in a student distribution q θ that is overly smooth, spreading across the entire support of p. To address this, recent studies (Lee et al., 2023;Gu et al., 2024a;Kim et al., 2024;Gu et al., 2024b) have explored the reverse KLD (RKLD), which allows q θ to focus on a few prominent modes of p. Despite the effectiveness, empirical results (Wen et al., 2023;Wu et al., 2024;Ko et al., 2024) suggest that RKLD often yields suboptimal performance across a range of tasks. What is worse, there is no systematic approach to identify the essential issues hidden behind, which hinders the development of a more generic and effective KD framework. To get out of this dilemma, we first pose the following question:
What underlying factors contribute to the suboptimal performance of FKLD and RKLD?
To answer this, we analyze how different divergence func- tions affect the allocation of probability mass in the student distribution during training by tracking the log mass ratio LogR. Notably, LogR is proportional to the gradient of the loss function w.r.t the logits. This insight allows us to frame the problem as understanding how divergence algorithms influence the reduction of LogR. Through this lens, we identify two key mode-concentration effects: Hardness-Concentration and Confidence-Concentration. Hardness-Concentration refers to focusing on modes in the loss where there is a large error between p and q θ , while Confidence-Concentration refers to focusing on modes in the loss where q θ has high confidence.
On top of this, we find that the limitations of FKLD and RKLD stem from the extreme ways they utilize these concentration effects: a) FKLD exhibits weak concentration effects, treating mismatches equally from all classes, which fails in guiding the student to concentrate on the target class and causes incorrect predictions (Fig. 1d). b) RKLD exhibits strong concentration effects, focusing on both hard classes with large errors and classes where the student has high confidence. This often leads to a trivial solution, where the well-trained student focuses exclusively on the target class and ignores broader knowledge from p (Fig. 1e). With the limitations revealed, we continue to seek an answer to the following question:
Can we find a generic, theoretically grounded method to balance hardness-concentration and confidenceconcentration?
In pursuit of this, we introduce the α-β-divergence, a general extension of divergences that unifies FKLD and RKLD, while also extending to previously unexplored divergences like the Hellinger distance and β-divergence. Our theoretical results demonstrate that the α-β-divergence provides a flexible mechanism to smoothly interpolate between the extremes of FKLD and RKLD by controlling the trade-off between hardness-concentration (Fig. 1b) and confidenceconcentration (Fig. 1c) via the hyperparameters α and β. This mechanism ensures a more proper allocation of probability mass (Fig. 1g). Motivated by these insights, we propose ABKD, a generic distillation framework based on α-β-divergence. Empirical results across a variety of tasks, including instruction-following and image classification, demonstrate ABKD's generality and effectiveness. For instance, by modifying only the loss function, ABKD achieves performance improvements of 0.81 to 3.31 over FKLD and RKLD on five instruction-response datasets when distilling GPT-2 XL (1.5B) into GPT-2 (0.1B).
In summary, the contributions of this work are three-fold:
• Theoretically: We analyze the limitations of FKLD and RKLD from novel perspectives of hardnessconcentration and confidence-concentration, and show that the α-β-divergence offers a flexible approach to balance these effects.
• Methodologically: We propose ABKD, a flexible distillation framework that unifies FKLD and RKLD and generalizes to several other divergences, offering greater versatility and applicability.
• Empirically: Extensive experiments on 17 language and vision datasets with 12 teacher-student configurations (0.85M-0.46M to 7B-3B) validate the theoretical insights. ABKD outperforms or matches state-of-theart methods without extra trainable parameters and allows further gains by rectifying their loss functions.
Prior Arts. We discuss related work and defer a concentrated account to App. A.
this section cite: ['b22', 'b48', 'b61', 'b8', 'b44', 'b81', 'b79', 'b27', 'b55', 'b80', 'b36', 'b30', 'b69', 'b70', 'b33']

Section: Preliminaries
KD involves using a fixed teacher model f T to improve the performance of a parameterized student model f S . Given an input x, the teacher f T and student f S produce probability distributions p and q θ , respectively.
The goal of KD can be achieved by letting q θ mimic p for all samples in dataset D. A direct way to do this is minimizing:
ℓ KD ≜ D(p∥q θ ),(1)
where D is a distribution measure. Optionally, practitioners can substitute p with the one-hot vector y, where y ≜ [0, . . . , 1, . . . , 0] with 1 at ground-truth label y and 0 elsewhere. In this case, the loss is ℓ CE ≜ D(y∥q θ ), where D is typically the FKLD. The final training loss is:
ℓ = ℓ CE + λℓ KD ,(2)
where λ is a hyperparameter. Since p provides richer information (i.e., soft label) than the one-hot vector y, KD outperforms traditional supervised fine-tuning on many downstream tasks, such as instruction-following and image classification. The settings for these tasks in KD are as follows.
Instruction-following. Let x and y represent the input and output sequences, respectively. A token-level autoregressive model produces an C-dimensional probability distribution for the n-th token over the vocabulary V, conditioned on x and y <n , where y <n ≜ (y 1 , y 2 , . . . , y n-1 ) denote the generated output sequence up to the (n -1)-th token, The discrepancy between token-level distributions of p and q θ is defined as D(p∥q θ ) ≜ 1 Ly Ly n=1 D(p(• | y <n , x)∥q θ (• | y <n , x)), where L y denotes the sequence length.
Image classification. Let x ∈ R H×W be an image and y ∈ R C its one-hot label, with H, W , and C representing the image dimensions and number of classes. A vision model produces a C-dimensional probability distribution conditioned on x. The discrepancy between p and q θ is defined as D(p∥q θ ).
this section cite: []

Section: The Limitations of FKLD and RKLD
Prior arts primarily use FKLD D KL (p∥q θ ) or RKLD D KL (q θ ∥p) to measure distribution discrepancy:
D KL (p∥q θ ) = k p(k) log p(k) q θ (k) , (3
) D KL (q θ ∥p) = k q θ (k) log q θ (k) p(k) .(4)
Despite promising success, recent empirical studies show that these two divergences cause suboptimal performance (Wen et al., 2023;Ko et al., 2024;Wu et al., 2024). Next, we uncover the underlying factors that contribute to the limitations of FKLD and RKLD by tracking how they allocate probability mass in the student distribution during gradient updates. These insights will guide us in Sec. 4 to identify a more suitable divergence for KD.
this section cite: ['b69', 'b33', 'b70']

Section: Tracking Probability Allocation with Log Mass Ratio
To find a proper probability matching scheme, KD algorithms must keep allocating the probability mass of the student distribution during training. The key in our theory is to keep track of the probability mass change in each gradient update step. To do this, we define a monitoring quantity called log mass ratio inspired by Tajwar et al. (2024):
LogR A t (y) ≜ log q A t+1(
this section cite: ['b57']

Section: y) q t (y) ,
where q A t+1 (y) is the probability mass for class y obtained from algorithm A at step t + 1; q t (y) is original probability mass for class y at step t.
To define probability mass q t , we follow the most popular convention that the class probability is approximated by a softmax function such that: q t (y) ∝ exp(f t y ), where f t y is the logit for the y-class channel at step t. Interestingly, one can show that (see App. C.1) LogR A t (y) is proportional to the gradient of the logit f t y as follows:
LogR A t (y) = -η • ∇ f t y ℓ + N A t (y),(5)
where η denotes the learning rate and N A t (y) is a normalizing factor independent of y, which vanishes to zero when all the class channel gradients ∇ fy ℓ vanish. Note that under a mild assumption, we can show that (see App. C.2) when ∇ W ℓ, the overall gradient w.r.t. the model weights W , goes to zero, ∇ fy ℓ also goes to zero. In this sense, to reach a local minimum, the algorithm automatically reduces the magnitude of ∇ fy ℓ, and also the |LogR A t (y)|. Based on the discussion above, we next show how reducing |LogR A t (y)| in different divergences affects hardnessconcentration and confidence-concentration.
this section cite: []

Section: FKLD and RKLD as Two Extreme Cases
First, we have the following upper bounds for the log mass ratio for FKLD (Eq. 3) and RKLD (Eq. 4).
Proposition 3.1. The updates induced by FKLD and RKLD for q t within one gradient descent step are given by:
FKLD: LogR F t (y) ≤ η• 1 (a)
• p(y) -q t (y)
+ N F t (y) , RKLD:
LogR R t (y) ≤ η • q t (y)(a1)
log p(y) -log q t (y)
+ k q t (k) ((b1)
) log p(k) -log q t (k)a2
+ N R t (y) ,(b2)
where N F t (y) and N R t (y) denote constant normalization factors independent of y and vanish to zero when p = q t .
This proposition is inspired by (Tajwar et al., 2024), but our work distinguishes itself by shifting the focus towards explicitly identifying the factors that drive the decrease in LogR A t (y) . The proof is in App. G.1. By minimizing the log mass ratio, there are two types of effects hidden in the results. We first analyze their independent roles. 1. The first type, represented by terms (b), (b 1 ), and (b 2 ), has the general form of |s(p(k)) -s(q t (k))|, which measures the matching loss between student and teacher distribution. If considered independently, such terms control the effect of hardness-concentration.
More precisely, a sharper term with a larger rate-ofchange corresponds to an aggressive student who aims to focus on the hardest classes to reach a good matching performance w.r.t. the teacher (Fig. 1b).
2. The second type, denoted by terms (a), (a 1 ), (a 2 ), can be expressed as the student's confidence weighting function: q t (y) β (β ≥ 0). If considered independently, such terms control the effect of confidenceconcentration. In other words, a sharper weighting function corresponds to a confident student who only cares about the matching performance in classes that the student believes to be the ground truth (Fig. 1c).
What is the joint effect of the two? Prop. 3.1 provides two extreme answers, FKLD and RKLD. FKLD in the results picks a very weak hardness-concentration effect with s(x) = x and a very weak confidence-concentration effect with β = 0. As a result, FKLD forces the student to treat all matching penalties equally for all classes since there is no weighting function ((a) = 1). This fails to concentrate on the target classes. By contrast, RKLD picks a very strong hardness-concentration effect with s(x) = log(x) (recall that 0 < x < 1, log is much sharper than linear function) and a very strong confidence-concentration effect with β = 1. Recall that a well-trained student distribution q t primarily concentrates probability on the target class and assigns smaller probabilities to others. In this case, an overly strong confidence-concentration effect suppresses the hardness-concentration effect on non-target classes while emphasizing this effect on the target class. This results in a trivial solution: the student concentrates solely on the target class and neglects the overall matching effect.
The following theorem makes the above intuition more rigorous. The results suggest an asymmetric mass allocation for RKLD and an equally important allocation for FKLD. Due to the space limit, the readers are referred to App. G.2 for a formal expression and the proof.
this section cite: ['b57']

Section: Theorem 3.2 (Informal).
Given the student distribution q θ and teacher distribution p, FKLD and RKLD differ as follows within one gradient update:
1. FKLD allocates the mass across all classes equally.
2. RKLD preferentially increases the mass of underestimated (p(y) > q θ (y)) classes with higher q θ (y).
3. RKLD preferentially reduces the mass of overestimated (p(y) < q θ (y)) classes with smaller q θ (y).
As shown in Fig. 1(d), the equally weighted matching scheme in FKLD drives students to sub-optimal modes, which induces wrong predictions. For RKLD, the theorem states that it only favors small mass classes when the teacher score is over-estimated while only favors large mass classes under the opposite scenario. As a total effect, the small mass tends to get smaller; the large mass tends to get larger. As an extreme result shown in Fig. 1(e), RKLD eventually forces the student to focus on one class. This makes the teacher's supervision degenerate to a ont-hot label, which loses the distributional information hidden inside the teacher's prediction. This leads to the following conclusion:
A proper divergence should achieve a moderate tradeoff between hardness-concentration and confidenceconcentration.
this section cite: []

Section: Weighted Sum of FKLD and RKLD
In pursuit of this, a naive solution is to take a weighted sum of FKLD and RKLD, which we call the weighted sum divergence (WSD):
D WSD (p∥q) ≜ αD KL (p||q) + βD KL (q||p),(6)
where α and β are hyperparameters. A more principled approach is to adapt the weighting coefficients dynamically during training based on the discrepancy (e.g., entropy) between p and q, as done in previous works (Amara et al., 2022;Wu et al., 2024).
Unfortunately, such a composite metric overemphasizes modes with small probabilities in p and q. To see this, when either q(k) ≈ 0, p(k) > 0 or p(k) ≈ 0, q(k) > 0, we have D WSD (p∥q) → ∞. Hence, the algorithm must focus on extreme cases to minimize the objective function, leading to improper probability allocation. Moreover, similar to the analysis in Ko et al. (2024), one can easily show that the gradient norm in this case also grows excessively, leading to significant and potentially noisy parameter updates. Such behaviors can destabilize the optimization process and hinder convergence.
Another attractive approach is to use the Jensen-Shannon divergence (Binici et al., 2022;Agarwal et al., 2024
) D JSD (p∥q) ≜ 1 2 D KL p m + 1 2 D KL q m , where m = 1 2 (p + q).
However, a major drawback of JSD is that it suffers from gradient vanishing (Arjovsky et al., 2017) when the distributions p and q θ are far apart (a common scenario in early training stages), which hinders model convergence.
Above all, balancing hardness and confidence concentration is non-trivial if one only resorts to FKLD and RKLD.
In the next section, we will introduce a generic notion of divergence to address this issue.
this section cite: ['b1', 'b70', 'b33', 'b4', 'b0', 'b2']

Section: ABKD: The Proposed Method

this section cite: []

Section: ABKD
One way to pursue a harmonic utilization of hardness-and confidence-concentration is to find a subtle point between FKLD and RKLD. The following α-β-divergence exactly serves this purpose (Cichocki et al., 2011).
Definition 4.1 (α-β-divergvence). Consider α and β ∈ R, satisfying α, β, α + β ̸ = 0. the α-β-divergence of two distributions is given by:
D (α,β) AB (p ∥ q) ≜ - 1 αβ k p(k) α q(k) β - α α + β p(k) α+β - β α + β q(k) α+β ,
where p = [p(k)] C k=1 and q = [q(k)] C k=1 are two discrete distributions over C classes.
As will soon be seen in Sec.4.2, both hardness-concentration and confidence-concentration effects in α-β-divergence could be regarded as an interpolation between the corresponding effect of FKLD and RKLD. Such ability allows the α-β-divergence to ensure a more proper allocation of probability mass.
Inspired by this, we propose ABKD, which is formally defined as minimizing the following objective:
ℓ = ℓ CE + λD (α,β) AB (p ∥ q θ ),(7)
Beyond this issue, α-β-divergence is also a generic notion Table 1. Some divergence functions and their corresponding choices of α and β. The α-β-divergence can be extended by continuity (by applying l'Hôpital formula) to cover all the values of α, β ∈ R, as shown in App. B.
this section cite: ['b9']

Section: Distribution Measure Reference Range
Kullback-Leibler (KL) divergence Kullback & Leibler (1951) α = 1, β = 0 Reverse KL divergence Kullback & Leibler (1951) α = 0, β = 1 α-divergence Chernoff (1952) α + β = 1 β-divergence Basu et al. (1998) α = 1 Hellinger distance Hellinger (1909) α = β = 0.5 Squared euclidean distance Heath (1956)
α = β = 1
of a family distribution divergences, which includes FKLD, RKLD, and other typical divergences as special cases. For example, when (α = 1, β = 0), one obtains FKLD; when (α = 0, β = 1), one obtains RKLD. Please see Tab. 1 for other special cases. In this way, ABKD nature provides a generic framework for divergence-based distillation algorithms.
this section cite: ['b35', 'b35', 'b7', 'b3', 'b19']

Section: Trading off Hardness-Concentration and Confidence-Concentration via α-β-divergence
ABKD offers a unified space to trade off the hardnessconcentration and confidence-concentration effects.
To explain this, we go back to the log mass ratio, the following proposition explains how the hyperparameters α and β influence the reduction of |LogR
(α,β) t (y)|.
Proposition 4.2. The updates induced by α-β-divergence for q t within one gradient descent step are given by:
LogR (α,β) t (y) ≤ η q t (y) β (a) p(y) α -q t (y) α α (b) +ηq t (y) k q t (k) β (a1) p(k) α -q t (k) α α (b1) + N (α,β) t (y) ,
where N α,β t (y) denotes constant normalization factor independent of y and vanishes to zero when p = q t .
The proof is in App. G.5. In (a) and (a 1 ), α-βdivergence employs a power form q t (k) β for confidenceconcentration effect. It is easy to see when β → 1, it degenerates to the effect of RKLD, and when β → 0 to the effect of FKLD. A larger β provides a stronger effect of confidence-concentration, focusing the matching performance on its most confident classes (Fig. 1c). Meanwhile, terms (b) and (b 1 ) uses | p(y) α -qt(y) α α | for hardnessconcentration effect. It is easy to see when α → 1, it degenerates to the effect of FKLD, and when α → 0 to the effect of RKLD. A smaller α amplifies the hardness-concentration effect and tends to be more aggressive in achieving better matching by penalizing errors on hard classes (Fig. 1b).
In this sense, by tuning α and β, we can flexibly balance the influence of the two effects and avoid extreme cases (Fig. 1g). For a finer-grained theoretical analysis and hyperparameter tuning guidelines, please see App. D, Thm. D.1.
Comparing with the Weighted Sum. As discussed earlier in Sec. 3.3, WSD often focuses excessively on the extreme values of p/q, leading to unstable optimization. Fortunately, one can show that the α-β-divergence can finely adjust the focus on different likelihood ratios p/q, thus enjoying a more stable gradient. For further analysis, see App. E.
Comparing with the α-divergence. One might also recall the α-divergence to achieve the trade-off, which is defined as
D α (p∥q) ≜ 1 α(α-1) k p(k) α q(k) 1-α -1 . It includes D KL (p∥q θ )
as α → 1, and D KL (q θ ∥p) as α → 0. Note that when β = 1-α, it becomes a special case of our framework. According to Prop. 4.2, to decrease α, one has to increase β to ensure that they add up to 1. Such unnecessary restriction hinders its ability to achieve better performance, as shown in Fig. 1(a) and (f). For further analysis, please see App. F.
this section cite: []

Section: Experiments
In the following, we investigate to what extent our theoretical results translate into practice on natural language and vision tasks. Due to space limitations, please see App. I for more details on datasets, competitors, and implementation.
this section cite: []

Section: Natural Language Processing Tasks
Datasets. We evaluate our methods on five task-agnostic instruction-following benchmarks. Evaluation metric is based on ROUGE-L (Lin, 2004). Details about the datasets and evaluation metric can be found in App. I.1.1.
Competitors. We consider the following state-of-the-art (SOTA) baselines: 1) supervised fine-tuning (SFT) with only student model on fixed datasets; 2) KD with FKLD on fixed datasets; 3) SeqKD with SFT to teacher-generated output; 4) MiniLLM with RKLD using a policy gradient approach on student-generated outputs (SGOs); 5) GKD with JSD on  (a) Training Speed (b) Effects of SGOs a mixture of SGOs and fixed datasets; 6) DISTILLM with S(R)KL on a mixture of SGOs and fixed datasets. Please refer to App. I.1.2 for details about competitors and SGOs.
Results. From Tab. 2, we have the following observations: 1) Distillation methods often outperform SFT, showcasing their potential. However, they can sometimes yield worse results (e.g., KD on Unnatural when distilling GPT-2 XL into GPT-2), highlighting the importance of selecting a proper distillation objective. 2) By simply modifying the distillation objective, our framework outperforms vanilla KD and SFT across various datasets when distilling GPT-2 XL (1.5B) to smaller-scale families of GPT-2 (0.1B∼0.8B); 3) Prior arts (Ko et al., 2024;Agarwal et al., 2024) show that training with SGOs can lead to significant improvements. However, even when compared to SGOs-based methods (e.g., GKD, DISTILLM) under this inherently unfair setting, our approach consistently achieves superior or comparable results, especially on Super-Natural and Unnatural datasets.
Efficiency Comparison. Fig. 3(a) shows that our framework matches the training speed of vanilla KD, as it only modifies the distillation objective without introducing additional cost. This addresses concerns regarding the scalability of our method. In contrast, other distillation methods require 1.6 to 7 times longer training time due to the continuous need to sample student's outputs during training.
this section cite: ['b41', 'b33', 'b0']

Section: Effects of SGOs.
We examine the robustness of our framework by evaluating its performance with various SGOs ap- proaches. As shown in Fig. 3(b), our framework consistently delivers high performance across different settings, highlighting its adaptability and effectiveness. Further experimental results and analyses are provided in App. J.1.1.
Effects of Loss Functions. Tab.3 compares the performance between various loss functions. The results show that α-β-divergence consistently outperforms the others, while using only αor β-divergence degrades performance due to limited expressivity. In particular, α-β-divergence achieves improvements of 0.81 to 3.31 over FKLD and RKLD across five datasets, whereas WSD, which combines weighted FKLD and RKLD, fails to deliver comparable results. Furthermore, Fig. 2 demonstrates the superior performance of α-β-divergence during the entire training phase.
In summary, these empirical results align with the theoretical insights in Sec. 3 and show that even modest adjustments to the loss function can yield significant improvements.
this section cite: []

Section: Vision
Tasks Datasets. We conduct experiments on 12 popular image recognition datasets. Dataset details are referred to App. I.2.1. The evaluation metric used is accuracy. Apart
(a) WRN-40-2 → WRN-16-2 (b) WRN-40-2 → WRN-40-1 (c) resnet56 → resnet20 (d) resnet110 → resnet 20 (e) resnet110 → resnet32 (f) resnet32x4 → resnet8x4 (g) vgg13 → vgg8 (h) resnet110 → resnet44 (a) alpha on CIFAR-100 (b) alpha on Dolly (c) beta on CIFAR-100 (d) beta on Dolly from the standard training-evaluation paradigm, we further consider a novel base-to-new setting (Zhou et al., 2022;Hua et al., 2025) to more thoroughly analyze the student model's generalization across classes. In this setup, training is performed on base classes, and accuracy is evaluated on both base and new classes. Please see App. I.2.3 for more details.
Competitors. We consider the following SOTA distillation methods: 1) KD, 2) DKD, 3) LSD, and 4) TTM. For the base-to-new setting, we also compare with SOTA SFT methods: 5) CoCoOp, 6) MaPLe, and 7) PromptSRC. Please refer to App. I.2.2 for method details.
Results. Fig. 4 and Fig. 5 show results from 9 teacherstudent architectures on 12 datasets. Based on these, we conclude: 1) Without modifying the distillation objective, methods that more effectively utilize teacher distribution knowledge (e.g., DKD, TTM, and LSD) can outperform vanilla KD; 2) However, their scores fall short in some cases, such as LSD in base-to-new setting; 3) Orthogonal to them, our framework selects more suitable distillation objectives for specific teacher-student pairs, showing competitive or superior results, particularly in base-to-new setting.
Apply to Other Distillation Techniques. Fig. 5 also shows that our framework can act as a simple plug-and-play tool to rectify the loss functions used by existing methods, yielding further improvements (e.g., ABDKD vs DKD).
this section cite: ['b82', 'b24']

Section: Sensitivity Analysis
We next analyze the effects of hardness-concentration and confidence-concentration, which helps validate the theoretical insights shown in Prop. 4.2 and Thm. D.1.
Effect of α on hardness-concentration. Figs. 6(a) and (b) show performance during training for different α. In CIFAR-100, with its relatively low-dimensional output distribution, a smaller α (stronger hardness-concentration) aggressively penalizes errors but offers limited gains. However, in Dolly, with a higher-dimensional output (e.g., GPT-2's vocabulary size of 50,257), a well-tuned smaller α is crucial to avoid local optima, especially in early training stages.
this section cite: []

Section: Effect of β on confidence-concentration.
Figs. 6(c) and (d) show how β affects Shannon entropy of the output distribution and Self-BLEU score (Zhu et al., 2018) of output sequences (100 indicates deterministic outputs and 0 denotes maximum diversity). The smaller β (weaker confidenceconcentration) places more emphasis on classes with low student confidence, encouraging the student to focus more on learning the soft label information from the teacher distribution. This leads to a smoother output distribution (higher entropy) and more diverse generated sequences (lower Self-BLEU). Thus, selecting an appropriate β ensures a balance between focusing on the target class and learning more soft label information.
this section cite: ['b84']

Section: Conclusion
In this paper, we argue that the key to KD lies in trading off two mode-concentration effects: hardness-concentration and confidence-concentration. The widely used FKLD and RKLD fail to achieve this balance, instead representing two extreme cases that lead to improper probability allocation. To address this issue, we introduce ABKD, a generic distillation framework based on α-β-divergence. ABKD generalizes FKLD and RKLD to a broader family of divergences, offering greater flexibility. Our theoretical results show that ABKD can flexibly interpolate between the above two extremes, enabling an effective trade-off. Extensive experiments further demonstrate its effectiveness.
this section cite: []

Section: References
Ref_id:b0 Title: On-policy distillation of language models: Learning from self-generated mistakes Year: (2024)
Ref_id:b1 Title: Bd-kd: balancing the divergences for online knowledge distillation Year: (2022)
Ref_id:b2 Title: Wasserstein generative adversarial networks Year: (2017)
Ref_id:b3 Title: Robust and efficient estimation by minimising a density power divergence Year: (1998)
Ref_id:b4 Title: Preventing catastrophic forgetting and distribution mismatch in knowledge distillation via synthetic data Year: (2022)
Ref_id:b5 Title: Food-101mining discriminative components with random forests Year: (2014)
Ref_id:b6 Title: Knowledge distillation with the reused teacher classifier Year: (2022)
Ref_id:b7 Title: A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations Year: (1952)
Ref_id:b8 Title: On the efficacy of knowledge distillation Year: (2019)
Ref_id:b9 Title: Generalized alpha-beta divergences and their application to robust nonnegative matrix factorization Year: (2011)
Ref_id:b10 Title: Free dolly: Introducing the world's first truly open instructiontuned llm Year: (2023)
Ref_id:b11 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b12 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b13 Title: Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories Year: (2004)
Ref_id:b14 Title: Openwebtext corpus Year: (2019)
Ref_id:b15 Title: Knowledge distillation of large language models Year: (2024)
Ref_id:b16 Title: Knowledge distillation for pre-training language models Year: (2024)
Ref_id:b17 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: The thirteen books of Euclid's Elements Year: (1956)
Ref_id:b20 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b21 Title: Neue begründung der theorie quadratischer formen von unendlichvielen veränderlichen Year: (1909)
Ref_id:b22 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b23 Title: Unnatural instructions: Tuning language models with (almost) no human labor Year: (2023-07)
Ref_id:b24 Title: Openworldauc: Towards unified evaluation and optimization for open-world prompt tuning Year: (2025)
Ref_id:b25 Title: Knowledge distillation from a stronger teacher Year: (2017)
Ref_id:b26 Title: Distilling bert for natural language understanding Year: (2019)
Ref_id:b27 Title: Multi-level logit distillation Year: (2023)
Ref_id:b28 Title: Multi-modal prompt learning Year: (2023)
Ref_id:b29 Title: Self-regulating prompts: Foundational model adaptation without forgetting Year: (2023)
Ref_id:b30 Title: PromptKD: Distilling student-friendly knowledge for generative language models via prompt tuning Year: (2024-11)
Ref_id:b31 Title: URL Year: ()
Ref_id:b32 Title: Sequence-level knowledge distillation Year: (2016)
Ref_id:b33 Title: Towards streamlined distillation for large language models Year: (2024)
Ref_id:b34 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b35 Title: On information and sufficiency. The annals of mathematical statistics Year: (1951)
Ref_id:b36 Title: Self-knowledge distillation via dropout Year: (2023)
Ref_id:b37 Title: Detkds: Knowledge distillation search for object detectors Year: (2024)
Ref_id:b38 Title: Asymmetric temperature scaling makes larger networks teach well again Year: (2022)
Ref_id:b39 Title: Homotopic task-agnostic distillation of pretrained transformers Year: (2023)
Ref_id:b40 Title: Less is more: Task-aware layer-wise distillation for language model compression Year: (2023)
Ref_id:b41 Title: Rouge: A package for automatic evaluation of summaries Year: (2004)
Ref_id:b42 Title: Wasserstein distance rivals kullback-leibler divergence for knowledge distillation Year: (2024)
Ref_id:b43 Title: Fine-grained visual classification of aircraft Year: (2013)
Ref_id:b44 Title: Improved knowledge distillation via teacher assistant Year: (2020)
Ref_id:b45 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b46 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b47 Title: Cats and dogs Year: (2012)
Ref_id:b48 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b49 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b50 Title: Unintentional unalignment: Likelihood displacement in direct preference optimization Year: (2024)
Ref_id:b51 Title: Learning dynamics of llm finetuning Year: (2024)
Ref_id:b52 Title: Hints for thin deep nets Year: (2014)
Ref_id:b53 Title: Very deep convolutional networks for largescale image recognition Year: (2012)
Ref_id:b54 Title: Patient knowledge distillation for bert model compression Year: (2019)
Ref_id:b55 Title: Logit standardization in knowledge distillation Year: (2024)
Ref_id:b56 Title: Mobilebert: a compact task-agnostic bert for resourcelimited devices Year: (2020)
Ref_id:b57 Title: Preference fine-tuning of llms should leverage suboptimal Year: (2024)
Ref_id:b58 Title: Contrastive representation distillation Year: (2019)
Ref_id:b59 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b60 Title: Llama: Open and efficient foundation language models Year: (2023)
Ref_id:b61 Title: Attention is all you need Year: (2017)
Ref_id:b62 Title: Alphanet: Improved training of supernets with alphadivergence Year: (2021)
Ref_id:b63 Title: Minilmv2: Multi-head self-attention relation distillation for compressing pretrained transformers Year: (2020)
Ref_id:b64 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b65 Title: Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks Year: (2022-12)
Ref_id:b66 Title: URL Year: ()
Ref_id:b67 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2023-07)
Ref_id:b68 Title: Openauc: Towards auc-oriented open-set recognition Year: (2022)
Ref_id:b69 Title: L. f-divergence minimization for sequence-level knowledge distillation Year: (2023)
Ref_id:b70 Title: Rethinking kullback-leibler divergence in knowledge distillation for large language models Year: (2024)
Ref_id:b71 Title: Sun database: Large-scale scene recognition from abbey to zoo Year: ()
Ref_id:b72 Title: Qwen3 technical report Year: (2025)
Ref_id:b73 Title: Learning with multiclass auc: Theory and algorithms Year: (2021)
Ref_id:b74 Title: Auc-oriented domain adaptation: From theory to algorithm Year: (2023)
Ref_id:b75 Title: Revisiting auc-oriented adversarial training with loss-agnostic perturbations Year: (2023)
Ref_id:b76 Title: Harnessing hierarchical label distribution variations in test agnostic long-tail recognition Year: (2024)
Ref_id:b77 Title: Wide residual networks Year: (2016)
Ref_id:b78 Title: Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer Year: (2016)
Ref_id:b79 Title: Decoupled knowledge distillation Year: (2022)
Ref_id:b80 Title: Knowledge distillation based on transformed teacher matching Year: (2024)
Ref_id:b81 Title: Rethinking soft labels for knowledge distillation: A bias-variance tradeoff perspective Year: (2021)
Ref_id:b82 Title: Conditional prompt learning for vision-language models Year: (2022)
Ref_id:b83 Title: Valuing training data via causal inference for in-context learning Year: (2025)
Ref_id:b84 Title: Texygen: A benchmarking platform for text generation models Year: (2018)
Ref_id:b85 Title:  Year: ()
Ref_id:b86 Title:  Year: ()
Ref_id:b87 Title:  Year: ()
Ref_id:b88 Title:  Year: ()
Ref_id:b89 Title:  Year: ()
Ref_id:b90 Title:  Year: ()
Ref_id:b91 Title:  Year: ()
Ref_id:b92 Title: 3 How does ABKD perform with alpha/beta outside [0,1]? Year: ()
Ref_id:b93 Title: Input Rami went to new york city on business. While he was there, he met his good friend ali Year: ()
Ref_id:b94 Title: Ground-truth Rami went to New York City on business. While he was there, he met his good friend Ali Year: ()
Ref_id:b95 Title: While she was there, he met His good friend ali Year: ()
Ref_id:b96 Title: KD Rami went to new yORK city on business. While she was there, he met His good friend ali, who taught him how to go city Year: ()
Ref_id:b97 Title: SeqKD Rami went to new yorks city on business. While being there, he met his bad friend ali Year: ()
Ref_id:b98 Title: MiniLLM Rami went to new yorks city on business. While being there, he met his friend ali Year: ()
Ref_id:b99 Title: GKD Rami went to new yORK city on business. While there, he met his bad friend Ali Year: ()
Ref_id:b100 Title: Holds: ali, who showed him that you are a good friend Year: ()
Ref_id:b101 Title: ABKD) Rami went to New York Citys on business. While he was there, he met her good friend Ali Year: ()
Ref_id:b102 Title:  Year: ()
Ref_id:b103 Title: Instruction You will be given a text consisting of multiple sentences. The task is to find the number of questions present in the text, and then print them out separately. A question is defined as a sentence that ends with a question mark Year: ()
Ref_id:b104 Title: Do you know when you were born? What month are we in currently? Year: (1984)
Ref_id:b105 Title: Do you know when you were born? \n What month are we in currently? Year: ()
Ref_id:b106 Title: What year are we in currently? KD I was born on October 3nd, 1984. Do you actually know? I'm not sure because I don't know when I was born Year: (1984)
Ref_id:b107 Title: SeqKD You are in currently Year: ()
