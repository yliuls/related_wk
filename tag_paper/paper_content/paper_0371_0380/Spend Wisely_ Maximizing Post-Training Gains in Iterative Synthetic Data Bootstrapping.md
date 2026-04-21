Title: Spend Wisely: Maximizing Post-Training Gains in Iterative Synthetic Data Bootstrapping
Abstract: Modern foundation models often undergo iterative "bootstrapping" in their posttraining phase: a model generates synthetic data, an external verifier filters out low-quality samples, and the high-quality subset is used for further fine-tuning. Over multiple iterations, the model performance improves, raising a crucial question: How should the total budget for generation and training be allocated across iterations to maximize final performance? In this work, we develop a theoretical framework for analyzing budget allocation strategies. Specifically, we show that constant policies fail to converge with high probability, while increasing policiesparticularly exponential growth policies-exhibit significant theoretical advantages. Experiments on image denoising with diffusion probabilistic models and math reasoning with large language models show that both exponential and polynomial growth policies consistently outperform constant policies, with exponential policies often providing more stable performance.

Section: Introduction
Supervised fine-tuning is a critical stage in training large language models (LLM). Pre-training supplies them with broad linguistic and factual knowledge, but task-specific skills-such as tool use (Schick et al., 2023), reasoning pattern (Gandhi et al., 2025), and agentic behavior (Shao et al., 2023)-emerge when the model is further refined on carefully curated, supervised examples. Securing these high-quality human-annotated data, however, remains a major bottleneck, as it requires domain expertise and substantial resources.
To address this limitation, synthetic data have emerged as a promising alternative that offers scalability and cost-effectiveness. Despite concerns about potential risks of model collapse (Shumailov et al., 2024;Dohmatob et al., 2024b), synthetic data is typically selected and verified before use in posttraining (Feng et al., 2024;Setlur et al., 2024), ensuring its quality. A common paradigm for leveraging synthetic data employs an iterative bootstrapping process (Zelikman et al., 2022;Trung et al., 2024): the model generates synthetic data, rewards or verifiers are used to filter and select highquality data, and the model is then fine-tuned on selected data. This process is repeated iteratively to fully improve performance. An illustration of this approach is provided in Figure 1.
However, for practitioners who implement this approach, an important question arises: How should a fixed computational budget be allocated to decide the amount of synthetic data to generate and select at each iteration to maximize performance? In this framework, synthetic data is generated, filtered using a reward model, and the selected data is used to further train the generator. The budget policy is defined as the quantity of data retained after selection, n t . Our goal is to identify the optimal policy across iterations to achieve the best final performance, given a fixed budget.
In this paper, we establish some foundational principles for crafting optimal strategies for synthetic data generation across iterations. To the best of our knowledge, this is the first attempt to address this problem. We begin with a theoretical analysis of policies that control the amount of selected synthetic data used in each iteration, as illustrated in Figure 1. In a simplified setting with Gaussian data and exponential reward functions, we could identify the optimal policy. Contrary to common strategies that maintain a fixed number of synthetic data across iterations (constant policies), our analysis shows that the optimal strategy requires exponentially increasing the amount of data at each iteration. Furthermore, in more general settings with only mild assumptions about regularity, we demonstrate that constant policies fail to converge with high probability. In contrast, policies that increase the amount of synthetic data across iterations can converge to optimal performance. Among such increasing policies, we prove that exponential growth policies guarantee exponential convergence, and in the worst-case scenarios, they achieve no larger computational costs compared to polynomial growth policies when attaining similar near-optimal performance.
Building on these theoretical insights, we validate our findings with two experiments: an imagedenoising task using diffusion probabilistic models (DPMs), and a math-reasoning task with large language models (LLMs). Across these experiments, exponential and linear (polynomial) growth policies outperform constant policies. Moreover, the best exponential policies match or exceed the performance of linear policies in every case, confirming their theoretical robustness.
We summarize our contribution as follows.
• We formulate the problem of optimizing post-training gains under a limited budget for iterative learning in Section 3.1.
• In a solvable Gaussian setting, we prove the optimality of an exponential growth policy in Section 3.2 and validate it through numerical simulations.
• In a more general setting, we demonstrate in Section 4 that constant policies fail to converge to the optimal reward with high probability, while increasing policies ensure convergence. Among these, the exponential growth policy achieves an exponential convergence rate and outperforms polynomial growth policies in the worst case.
• Experiments on image denoising (diffusion probabilistic model) and math reasoning (large language model) in Section 5 confirm that exponential and polynomial growth policies outperform constant policies, with exponential policies exhibiting greater stability.
2 Related Work Synthetic Data and Iterative Bootstrapping. The iterative bootstrapping framework, illustrated in Figure 1, is widely used for training foundational models in both text and vision domains. In this approach, synthetic data are generated and then iteratively refined or filtered to produce new training sets that more closely align with the target objectives. For large language models (LLMs) and vision language models (VLMs), this approach has been applied to instruction following (Touvron et al., 2023) and alignment tasks (Dong et al., 2023), and vision question answering (Yang & Dong, 2025), wherein reward models filter synthetic data to retain only high-quality samples. Recently, it has been particularly effective in improving reasoning abilities, as first demonstrated in Zelikman et al. (2022). Subsequent works (Lehnert et al., 2024;Su et al., 2024;Trung et al., 2024;Guan et al., 2025;Singh et al., 2025) have extended this approach to generate diverse reasoning paths with search algorithms. Beyond text, iterative bootstrapping has also been utilized to generate training data for image segmentation (Kirillov et al., 2023), create image-text pairs (Fan et al., 2023(Fan et al., , 2024)), and synthesize images to address distribution shifts (Hemmat et al., 2024;Azizi et al., 2023). However, in these empirical works, a principled method for controlling generation in each iteration is still missing.
To our knowledge, Ferbach et al. (2024) provides the only theoretical analysis of this paradigm, proving convergence to the optimal policy under the infinite-sample assumption. Crucially, their framework fails to characterize finite sample requirements, offering little actionable guidance to practitioners implementing this framework. We address this gap by establishing optimal computational budget allocation strategies across iterations to achieve maximum performance gains.
this section cite: ['b28', 'b15', 'b30', 'b32', 'b6', 'b13', 'b29', 'b41', 'b37', 'b36', 'b8', 'b39', 'b41', 'b25', 'b35', 'b37', 'b17', 'b33', 'b24', 'b11', 'b12', 'b20', 'b1', 'b14']

Section: Model Collapse.
A series of papers has shown that the incorporation of synthetic data into the training corpus can lead to performance degradation, either in a single training instance or iteratively over time (Shumailov et al., 2023;Alemohammad et al., 2024;Dohmatob et al., 2024bDohmatob et al., ,a, 2025)). Feng et al. (2024) has explored the use of data selection to overcome model collapse by employing synthetic data in a single generation, combined with verification. In our work, we also leverage verification, but focus on addressing the question of how to optimally generate synthetic data during learning.
this section cite: ['b31', 'b0', 'b6', 'b7', 'b13']

Section: Problem Formulation and Preliminary Analysis
In this section, we formalize the iterative learning setting in Section 3.1 and then analyze optimal budget allocation strategies with a simple Gaussian case in Section 3.2.
this section cite: []

Section: Problem Formulation
The setup is illustrated in Figure 1. Our goal is to train a parameterized generative model f (•; θ), for example, given by a DPM or an LLM, that generates data x ∼ P θ . A reward model R evaluates the quality of the generated data, serving as a measure of the model performance. We aim to train the generative model to maximize the expected reward
r(θ) = E x∼P θ [R(x)],
where we assume R(x) ∈ [0, 1].
The iterative training process leverages the generative nature of the model to perform self-generation and refinement based on reward signals. Specifically, we start with an initial model θ (0) . At the (t + 1)-th iteration, the model undergoes improvement through the following three steps: t) ).
• Generation step: Generate N t data by f (•; θ (
• Selection step: For each synthetic data point x, include it into dataset D t with probability R(x). Suppose n t of the N t data are selected to form the dataset D t , formally, |D t | = n t .
• Updating step: Update θ (t) to θ (t+1) with the dataset D t .
This iterative approach generates additional training data using synthetic samples, with the rewardguided selection process ensuring its quality. The selected samples are then used to further enhance the model's generative performance (Feng et al., 2024). This process, carried out iteratively to maximize performance gains, is particularly effective in scenarios where high-quality specialized data is scarce, for example, in mathematical reasoning (Zelikman et al., 2022;Guan et al., 2025;Singh et al., 2025). The algorithm framework is formalized in Algorithm 1. In the selection step, the data may be selected with noise. We present a simple form here for ease of understanding, while our results extend to the noisy case in Appendix B.5.
When practitioners use this algorithm, a key challenge is determining how to set N t and n t across iterations. This decision is particularly important, as the generation process of foundation models unually involves multiple forward propagations, and consequently, the generation cost often exceeds the training cost. Given a limited budget, it is crucial to design the process in a way that maximizes post-training gains.
Specifically, we focus on identifying the optimal policy of N t and n t correlated with the generation cost and the training cost, respectively. Given the reward model and generator, N t can be viewed as a random variable dependent on n t and the selection rate. Therefore, we define the policy over n t as π : {n t } t=0,1,••• , and our objective is to identify the best scheme for π.
Algorithm 1 Iterative learning with synthetic data
1: Input: An initial generative model f (•; θ (0) ), a reward model R, and a policy π : {n t } t with a termination time T . 2: Output: A fine-tuned generative model f (•; θ (T ) ). 3: for t ← 0 to T -1 do 4: Initialize selected synthetic dataset D t = ∅. 5: while |D t | < n t do 6: Generate a sample x ∼ P θ (t) . 7: Add x to D t with probability R(x). 8: end while 9:
Update θ (t) to θ (t+1) using the dataset D t .
10: end for
this section cite: ['b13', 'b41', 'b17', 'b33']

Section: Gaussian Case Study
To begin, we present a warm-up example with Gaussian data, where the optimal policy can be derived analytically. This example provides theoretical insight into the general iterative learning framework. The setup includes Gaussian data, an exponential reward function, and maximum likelihood estimation (MLE) as the updating algorithm.
this section cite: []

Section: Gaussian Generator and Exponential Reward Model.
To simplify the setting, we consider the generative model as a one-dimensional Gaussian distribution P θ = N (θ, σ 2 ), where the mean θ is the parameter to learn, and the variance σ 2 is fixed. The reward model is an exponential function R(x) = exp -x 2 /(2κ 2 ) , where κ is also fixed.
this section cite: []

Section: Updating via MLE.
In Algorithm 1, we employ MLE as the updating algorithm at line 9. Specifically, the parameter θ (t+1) updated in the t-th iteration is obtained by
θ (t+1) = arg max θ x∈Dt p θ (x),
where p θ (x) is the density function of the distribution P θ .
this section cite: []

Section: Optimization Problem.
We aim to find the optimal iterative learning policy within this setup. We assume that the cost of sampling the Gaussian distribution is negligible, while the training cost for MLE is proportional to the size of the dataset, n t . The objective is to maximize the expected performance of the generator in iteration T , as measured by the reward. Given a fixed budget C, the optimal policy will be the solution to
max {nt} T -1 t=0 E θ (T ) r θ (T ) s.t. T -1 t=0 n t ≤ C.(1)
For the above setup, we can theoretically identify the optimal policy in the following theorem. We compare the exponential, constant, and linear policy, and show the gap to the optimal expected reward as a function of the computational cost ( t n t ). All the results are averaged over 1,000 runs.
Theorem 3.1. If the initial parameter satisfies θ (0) ≤ (1 + σ 2 /κ 2 ) T (σ 2 + κ 2 ) 1/2
, then the optimal iterative policy, i.e., the solution of the optimization problem in Equation (1), is given by
n t ∝ 1 + σ 2 κ 2 t .
The proof is deferred to Appendix A and extends naturally to the high-dimensional case, as detailed in Appendix E.1. Theorem 3.1 demonstrates that the optimal policy in this setting follows an exponential scheme. The optimal strategy follows this rationale: In an iterative optimization algorithm, the contribution of an early update to the overall error decays exponentially with the total number of iterations T . As a result, achieving a low final error increasingly depends on the accuracy of later updates. To keep those updates precise enough, each iteration must therefore draw on an exponentially larger number of samples. We will further analyze this exponential scheme in the general analysis in Section 4.
this section cite: []

Section: Experiments.
We include an empirical experiment to confirm Theorem 3.1 with data in twodimensional space. The generator is set to N (θ, I 2 ). The reward function is R(x) = exp -∥x∥ 2 /(2κ 2 ) , which favors data concentrated near the origin. κ controls the flatness of the reward. Full details are deferred to Appendix E.2. Figure 2 plots the gap to the optimal reward as a function of the cumulative cost, defined as the total computational cost incurred up to that point.
The results demonstrate that the exponential policy consistently approaches the optimal reward faster than the other schemes. While the linear policy converges slightly more slowly, the constant policy, as shown in the left figure, clearly fails to converge. These findings validate our theoretical proof. Additional results with various parameters are left in Appendix F.1.
this section cite: []

Section: Main Theoretical Results
We now proceed to analyze the general setting. We will show that constant policies fail to converge with high probability, whereas there exists an exponential growth policy that achieves an exponential convergence rate and outperforms polynomial alternatives in the worst case. The assumptions and setup are outlined below, while detailed proofs for the subsequent theorems are provided in Appendix B.4.
Assumptions. We consider a general function class for the generative model, a general loss with regularity assumptions (Assumptions B.1 and B.2), and a reward model with mild assumptions about its relationship with the loss (Assumption B.3). The formal descriptions are provided in Appendix B. The warm-up example in the previous section serves as a special case that satisfies all these assumptions. Our results also naturally extend to noisy rewards, as discussed in Appendix B.5.
this section cite: []

Section: Updating via Gradient Descent.
In Algorithm 1, we use gradient descent with a learning rate η > 0 as the update algorithm at line 9. Specifically, the update for the t-th iteration writes
θ (t+1) = θ (t) - η n t x∈Dt ∇ θ l x; θ (t) , t = 0, 1, • • • ,
where l(x; θ) is the loss function. For any policy π, let θ (t) π t denote the iterative trajectory of the parameter.
this section cite: []

Section: Reward and Cost.
Similarly to the warm-up example, we evaluate the generative model using the expected reward r(θ). The optimal performance is the supremum of the expected reward over the whole parameter space r * = sup θ r(θ).
We consider a general computational cost that accounts for both generation and training, with constant coefficients c g and c t for each data point, respectively. Then, the total cost is given by
C(π, T ) = T -1 t=0 c g N t + c t n t .
The first policy class that we consider is
• constant policy: π const : n t = n 0 , t = 0, 1, • • • .
This is the most standard policy, where n 0 can be set either as the total number of prompts in the dataset (Zelikman et al., 2022), or as a multiple of the batch size in an online setting (Guo et al., 2024). However, in the following theorem we show that constant policies fail to converge to the optimal reward with high probability.
Theorem 4.1 (Bounded Reward for Constant policy). Under Assumption B.3, there exists a constant c > 0 such that, for any T and any constant policy π const , with probability at least 1/4, r * -r θ (T ) πconst ≥ cn -1/2 0 .
The theory establishes the non-convergence of constant policies, which stems from persistent random noise in the gradient. Specifically, the noise term is proportional to n -1 t and remains non-decaying under any constant policy, resulting in suboptimal performance.
To address this limitation, we introduce increasing policies, where n t grows monotonically over iterations. The following theorem shows that increasing policies ensure convergence to the optimal reward with high probability. Theorem 4.2 (Optimal Reward for Increasing Policies). Under Assumption B.1 to Assumption B.3, for any increasing policy π and ε > 0, there exists a sufficiently large T such that, with probability greater than 1 -
T -1 t=0 n -4 t , r * -r θ (T ) π ≤ ε.
This theorem suggests that increasing policies should be used in practice to maximize performance. Hence, we consider the following two classes, namely,
• polynomial growth policy:
π poly : n t = n 0 (1 + t) α , α > 0, t = 0, 1, • • • ; • exponential growth policy: π exp : n t = n 0 (1 + u) t , u > 0, t = 0, 1, • • • .
For the exponential policy, we can identify a specific policy that achieves an exponential convergence rate as follows. This theorem also applies to noisy rewards, as discussed in Appendix B.5.
Notice that the above three theorems analyze the convergence of the respective policies (constant, polynomial, and exponential) as the number of iterations T grows without imposing any limit on the total computational budget. We now introduce the budget constraint and compare the exponential policy with the polynomial policy. Specifically, for any policy π, we introduce
T * (π, ε) = min T T | r * -r θ (T ) π ≤ ε ,(2)
denoting the minimum number of iterations needed for the policy π to achieve a reward that is within ε of the optimal r * . The cost C(π, T * (π, ε)) is then the minimum cost to attain the target performance of the policy π. The following theorem compares the total cost incurred by the exponential growth policy and polynomial policies in a worst-case scenario. Theorem 4.4 (Worst-Case Optimality of the Exponential Policy, informal). For any constant or polynomial growth policy π and a sufficiently small ε, we have
sup C(π * exp , T * (π * exp , ε)) < sup C(π, T * (π, ε))
with probability at least 1 -
T * (π * exp ,ε)-1 t=0 n -4 t .
Here, the supremum is taken over all feasible problem settings, including parameter spaces, loss functions, and reward functions under mild assumptions.
For the formal version, please refer to Theorem B.7. This theorem shows that, in a worst-case analysis, the exponential policy π * exp from Theorem B.6 incurs a total cost no greater than that of any polynomial growth or constant policy to reach a specified performance threshold, establishing its efficiency and robustness. This analysis theoretically highlights the exponential policy as the preferred choice for our iterative learning tasks.
In summary, we draw the following conclusion:
• Constant policies lead to rewards that are consistently below optimal by a constant margin.
• Increasing policies such as polynomial and exponential ones can achieve optimal performance.
• In the worst case, there exists an exponential growth policy ensuring the same performance at equal or lower computational cost compared to any constant or polynomial growth policy.
this section cite: ['b41', 'b19']

Section: Experiments
Our theoretical results show that, in iterative bootstrapping, synthetic data curation with an exponential scheme outperforms both linear and polynomial schemes, while linear and polynomial schemes outperform the constant scheme. In our experiments, we evaluate constant, linear (as a simplified polynomial scheme), and exponential schemes.
We conduct two experiments to evaluate our schemes: image denoising with DPMs, and mathematical reasoning with LLMs, covering both image and text domains. The setups and implementation details are summarized in Table 1. In these settings, we emphasize that our goal is not to propose a novel task-solving method or generation technique. Rather, we focus on optimizing the allocation of generation and training costs in iterative learning with synthetic data-an objective that is method-agnostic and can be applied to any generation approach. Here, we validate our theoretical results from Section 4 and conduct a performance comparison among exponential, linear, and constant schemes on image data (using DPMs) and text data (with LLMs).
this section cite: []

Section: Image Denoising with Diffusion Models
We consider applying iterative learning to the image denoising task, a representative problem in image processing, where the goal is to recover x from a noisy observation y = x + η with an unknown Gaussian noise η.
Setup. We choose diffusion probabilistic models (DPMs) (Ho et al., 2020;Song et al., 2021) as denoisers due to their intrinsic design, which aligns closely with the denoising process (Xie et al., 2023) since the reverse process of a DPM gradually estimates and removes this noise using a trained network, effectively performing denoising. Specifically, we fine-tune a pre-trained diffusion modelfoot_0 for denoising on the MNIST (Deng, 2012) dataset, and readers may refer to Appendix C for more details.
We follow the iterative learning framework to annotate synthetic "clean" images to improve the denoising process. In each iteration, the diffusion model first generates synthetic denoised images x from the noisy observations. These are then filtered with a reward model linear to the Peak Signal-to-Noise Ratio (PSNR) values compared with the real images. The filtered synthetic data are then used to fine-tune the model through a standard noise-prediction training procedure, iteratively improving its denoising capability. Although an oracle reward model may not be practical in many real-world scenarios, its use here helps to validate the generality of our theoretical results. Furthermore, existing models or correction functions with prior knowledge could be utilized as the reward, as demonstrated in Gillman et al. (2024).
Implementations. We perform denoising for two noise levels corresponding to the diffusion step s = 10 and 20, where the PSNR of noisy images are 33.25 and 28.40, respectively. A summary of the algorithm is provided in Algorithm 2. We consider three constant and three linear policies, each with small, medium, and large n 0 . For the exponential policy, we have performed a small hyperparameter search over n 0 and u, selecting good configurations for two settings. The complete results of the hyperparameter search are presented as an ablation study on different exponential schemes in Appendix F.2, demonstrating its robustness. The configurations for all schemes are presented in the caption and legend of Figure 3.
Table 2: Comparison of each policy on the image denoising and math reasoning tasks. The final accuracy are determined using a held-out validation set. We use boldface for the best result while the underline for the second best. For the exponential policies, we only exhibit a single set of configurations for the comparison. We plot for all the policies in Figure 3 and Figure 4, respectively.
(a) Image denoising POLICY nt s = 10 s = 20 PRE-TRAINED N/A 40.92 36.96 CONSTANT 1 • B 42.75 38.91 10 • B 43.15 39.45 100 • B 44.18 39.91 LINEAR t • B 43.90 39.90 3t • B 44.23 40.08 10t • B 44.64 40.26 EXPONENTIAL -44.84 40.14 (b) Math reasoning nt symbolic p1 p2 N/A 55.60 34.98 15.72 10 • B 60.28 38.50 16.93 30 • B 60.16 39.68 17.73 100 • B 64.04 41.66 18.81 3t • B 62.54 38.88 17.01 10t • B 61.96 39.94 17.29 30t • B 64.24 40.84 19.17 -65.66 47.26 20.65
Results. We summarize the final performance of different schemes and baselines in Table 2 and show the trade-off between performance and computational cost in Figure 3. We observe that the exponential policy yields the best performance for s = 10, and performs comparable to the best linear policy for s = 20. This observation demonstrates both the benefits of an increasing policy (Theorem 4.2) and the robustness of the exponential growth policy (Theorem 4.4). In contrast, the constant policy only matches the exponential policy in the early stages of training, but soon reaches a performance plateau and eventually decays due to overfitting, corroborating Theorem 4.1.
this section cite: ['b21', 'b34', 'b38', 'b4', 'b16']

Section: Math Reasoning with LLMs
Now we move to natural language processing. Iterative learning with synthetic data is also an emerging method for LLMs. Previous research (Zelikman et al., 2022) leverages the model to generate synthetic data, which is then filtered using a reward model or accuracy metrics and subsequently utilized to fine-tune the model. This framework is employed in various settings, including instruction tuning (Touvron et al., 2023), alignment (Dong et al., 2023), and recent efforts to enhance reasoning capabilities through chain-of-thought (CoT) approaches (Hosseini et al., 2024;Lehnert et al., 2024;Singh et al., 2025).
Setup. We select mathematical reasoning to evaluate the effectiveness of our strategy in iterative learning. Let q and a represent the question and its corresponding answer, respectively. We use the LLM to generate synthetic solutions with CoT and in-context examples. These synthetic data are filtered according to the correctness of the final generated answer â, defined as the reward R(â) = 1(â = a). The filtered data are used to further train the model in an iterative process, without filtering based on the correctness of CoT itself. In summary, synthetic data with the correct final solutions are generated and used for training in each iteration, closely mimicking the STaR method (Zelikman et al., 2022).
Implementations. Many contemporary LLMs are known to overfit to GSM8k (Cobbe et al., 2021) or exhibit a certain degree of data leakage (Zhang et al., 2024;Mirzadeh et al., 2025), where even minor changes to numbers or names can result in significant performance degradation. This issue renders the evaluation results on GSM8k unreliable. To address this, we regenerate three math datasets-GSM_symbolic, GSM_p1, and GSM_p2-using symbolic math templates from Mirzadeh et al. (2025), one of the most outstanding papers on evaluating math reasoning. Each dataset consists of questions paired with their correct answers. GSM_symbolic retains a similar difficulty level to GSM8k, while GSM_p1 and GSM_p2 introduce one and two additional clauses per question, respectively, making them progressively more challenging. Together, these datasets complete a robust evaluation set.
In our iterative learning framework, we generate synthetic data using a mixing ratio of 7:2:1 across these three datasets, filter the data based on the correctness of the final answers, and fine-tune the model on the selected data. We use the Llama-3-8B-Base model (Dubey et al., 2024) with full- weight fine-tuning. For evaluation, the model is tested on held-out datasets from all three difficulty levels. During data generation and evaluation, in-context examples and CoT reasoning are used, but in-context examples are excluded during fine-tuning.
A summary of the algorithm is provided in Algorithm 3. Similarly to image denoising, we consider three constant and three linear policies, each with small, medium, and large n 0 . For the exponential scheme, we have selected one configuration and left the full ablation results in Appendix F.3. The configurations for all schemes are detailed in Figure 4 and its caption. We also include the performance of the pre-trained model as a baseline.
Results. Figure 4 shows the performance versus computational cost across the three datasets under iterative training, while Table 2 summarizes the final performance for all policies. Across all three difficulty levels, the exponential growth policy demonstrates the steadiest improvement compared to other schemes. Although the linear policy ranks second overall, it fails to match the exponential growth policy when the training budget is large. In particular, with the exponential growth policy, iterative learning increases accuracy on GSM_p1 from 35% to 47% and on GSM_p2 from 16% to 21%.
Based on these results, we recommend the exponential scheme for iterative learning.
this section cite: ['b41', 'b36', 'b8', 'b22', 'b25', 'b33', 'b41', 'b2', 'b42', 'b26', 'b26', 'b9']

Section: Conclusion
In this work, we take the first step toward understanding how to design optimal budget allocation policies in iterative bootstrapping for supervised fine-tuning. Through a combination of theoretical analysis and empirical validations, we demonstrate that constant policies are insufficient for achieving convergence, whereas increasing policies offer a more effective alternative. Among increasing policies, the exponential growth policy emerges as a robust and efficient method.
Although our focus here is on SFT in an iterative learning framework, recent research has highlighted the potential of Reinforcement Learning with Human Feedback (RLHF) (Yuan et al., 2024;Pang et al., 2024;Setlur et al., 2024) for similar iterative setups. Extending our framework to incorporate RLHF is a promising direction for future work. By answering these open questions, we aim to lay a foundation for more efficient iterative training approaches that can improve the post-training of foundation models.
this section cite: ['b40', 'b27', 'b29']

Section: References
Ref_id:b0 Title: Self-consuming generative models go MAD Year: (2024)
Ref_id:b1 Title: Synthetic data from diffusion models improves imagenet classification Year: (2023)
Ref_id:b2 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b3 Title: OpenCompass: A universal evaluation platform for foundation models Year: (2023)
Ref_id:b4 Title: The MNIST database of handwritten digit images for machine learning research Year: (2012)
Ref_id:b5 Title: Model collapse demystified: The case of regression Year: (2024)
Ref_id:b6 Title: A tale of tails: Model collapse as a change of scaling laws Year: (2024)
Ref_id:b7 Title: Strong model collapse Year: (2025)
Ref_id:b8 Title: RAFT: Reward rAnked FineTuning for generative foundation model alignment Year: (2023)
Ref_id:b9 Title: The Llama 3 herd of models Year: (2024)
Ref_id:b10 Title: Maximizing the potential of synthetic data: Insights from random matrix theory Year: (2025)
Ref_id:b11 Title: Improving CLIP training with language rewrites Year: (2023)
Ref_id:b12 Title: Scaling laws of synthetic images for model training... for now Year: (2024)
Ref_id:b13 Title: Beyond model collapse: Scaling up with synthesized data requires reinforcement Year: (2024)
Ref_id:b14 Title: Self-consuming generative models with curated data provably optimize human preferences Year: (2024)
Ref_id:b15 Title: Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective STars Year: (2025)
Ref_id:b16 Title: Selfcorrecting self-consuming loops for generative model training Year: (2024)
Ref_id:b17 Title: Small LLMs can master math reasoning with self-evolved deep thinking Year: (2025)
Ref_id:b18 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b19 Title: Direct language model alignment from online AI feedback Year: (2024)
Ref_id:b20 Title: Feedback-guided data synthesis for imbalanced classification Year: (2024)
Ref_id:b21 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b22 Title: Training Verifiers for Self-Taught Reasoners Year: (2024)
Ref_id:b23 Title: OpenRLHF: An easy-to-use, scalable and high-performance RLHF framework Year: (2024)
Ref_id:b24 Title: Segment anything Year: (2023)
Ref_id:b25 Title: Beyond A*: Better planning with transformers via search dynamics bootstrapping Year: (2024)
Ref_id:b26 Title: GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models Year: (2025)
Ref_id:b27 Title: Iterative reasoning preference optimization Year: (2024)
Ref_id:b28 Title: Toolformer: Language models can teach themselves to use tools Year: (2023)
Ref_id:b29 Title: RL on incorrect synthetic data scales the efficiency of LLM math reasoning by eight-fold Year: (2024)
Ref_id:b30 Title: Character-LLM: A trainable agent for role-playing Year: (2023)
Ref_id:b31 Title: The curse of recursion: Training on generated data makes models forget Year: (2023)
Ref_id:b32 Title: AI models collapse when trained on recursively generated data Year: (2024)
Ref_id:b33 Title: Beyond human data: Scaling self-training for problem-solving with language models Year: (2025)
Ref_id:b34 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b35 Title: Controllable fast and slow thinking by learning with randomized reasoning traces Year: (2024)
Ref_id:b36 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b37 Title: Reasoning with Reinforced Fine-Tuning Year: (2024)
Ref_id:b38 Title: Diffusion model for generative image denoising Year: (2023)
Ref_id:b39 Title: Mocoll: Agent-based specific and general model collaboration for image captioning Year: (2025)
Ref_id:b40 Title: Self-rewarding language models Year: (2024)
Ref_id:b41 Title: STaR: Bootstrapping reasoning with reasoning Year: (2022)
Ref_id:b42 Title: A careful examination of large language model performance on grade school arithmetic Year: (2024)
