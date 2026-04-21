Title: ENHANCING GENERATIVE AUTO-BIDDING WITH OFFLINE REWARD EVALUATION AND POLICY SEARCH
Abstract: Auto-bidding is a critical tool for advertisers to improve advertising performance. Recent progress has demonstrated that AI-Generated Bidding (AIGB), which learns a conditional generative planner from offline data, achieves superior performance compared to typical offline reinforcement learning (RL)-based autobidding methods. However, existing AIGB methods still face a performance bottleneck due to their inherent inability to explore beyond the static dataset with feedback. To address this, we propose AIGB-Pearl (Planning with EvaluAtor via RL), a novel method that integrates generative planning and policy optimization. The core of AIGB-Pearl lies in constructing a trajectory evaluator to assess the quality of generated scores and designing a provably sound KL-Lipschitzconstrained score-maximization scheme to ensure safe and efficient exploration beyond the offline dataset. A practical algorithm that incorporates the synchronous coupling technique is further developed to ensure the model regularity required by the proposed scheme. Extensive experiments on both simulated and real-world advertising systems demonstrate the state-of-the-art performance of our approach.

Section: INTRODUCTION
The increasing demand for commercial digitalization has facilitated the development of the autobidding technique in online advertising. Distinguished from traditional manual bidding products, auto-bidding provides advertisers with an efficient and flexible scheme to automatically optimize bids in dynamic and competitive environments (Balseiro et al., 2021a;Deng et al., 2021;Balseiro et al., 2021b). Technically, auto-bidding constitutes an offline sequential decision-making problem that aims to maximize advertising performance over a bidding episode, constrained to a static offline dataset due to operational safety concerns (Mou et al., 2022).
As a standard approach to offline decision-making problems, offline reinforcement learning (RL) (Kumar et al., 2020;Mao et al., 2024b) is widely adopted to solve the auto-bidding problem. By employing conservative policy search schemes, offline RL mitigates the infamous out-of-distribution (OOD) problem (Fujimoto et al., 2019), enabling reliable generalization beyond the offline dataset. However, their reliance on bootstrapped value estimates renders offline RL methods prone to training instability (Peng et al., 2024), potentially compromising policy performance.
Recent advances in generative models shed new light on offline decision-making problems (Zhu et al., 2023;Kang et al., 2023). Specifically, AI-generated bidding (AIGB) models auto-bidding as a trajectory-generation task and employs a generative model to approximate the conditional trajectory distribution of the offline dataset (Guo et al., 2024). AIGB avoids bootstrapping and exhibits more stable training and superior performance. However, the modeling approach in AIGB does not explicitly align with the performance-maximization of the auto-bidding problem. Inherently, AIGB relies on imitating trajectories from the offline dataset (Ajay et al., 2023) and lacks the ability to improve its generation quality beyond the offline dataset based on the performance feedback. Consequently, its conditional generation in the extrapolation regime can become unreliable, potentially leading to performance degradation or even to risky trajectory generations.
Hence, there arises a question: built on AIGB, the latest state-of-the-art auto-bidding method, can we devise a plausible scheme to involve policy optimization in its generative model? To this end, a natural idea is to integrate offline RL methods into AIGB. However, it is nontrivial to implement in the auto-bidding problem since (i) there is a lack of reward signals in AIGB to guide the generative model. Specifically, the generation quality of the generative model remains unknown during training, making it infeasible to explore new trajectories beyond the offline dataset; (ii) no dedicated offline RL algorithm exists for AIGB. In particular, theoretical analysis that guarantees safe generalization and mitigates OOD issues for generative models in auto-bidding remains largely unexplored.
To address these critical challenges, we propose AIGB-Pearl (Planning with EvaluAtor via RL), an RL-enhanced version of AIGB that learns a trajectory evaluator to score generation quality and drive exploration of the generative model through continuous interaction. The evaluator is trained through supervised learning on the offline dataset. Crucially, to mitigate the OOD problem and ensure reliable use of the evaluator, we examine the theoretical upper bound on the evaluator's bias. Then, guided by this analysis, we formulate a KL-Lipschitz-constrained score-maximization objective with a provable suboptimality bound, enabling safe and effective exploration beyond the offline data. Moreover, to perform constrained score maximization, we design a practical algorithm that incorporates the synchronous coupling technique to satisfy the generative model's Lipschitz condition. In addition, we note that AIGB-Pearl does not require bootstrapping and exhibits greater training stability than offline RL methods.
To summarize, our contributions in this paper are fourfold: (i) we propose a novel generative autobidding method, AIGB-Pearl, that enables continuous improvement in generation quality through exploration beyond the offline dataset; (ii) we propose a provable KL-Lipschitz constrained score maximization objective with a sub-optimality bound, ensuring a safe and effective generalization beyond the offline dataset; (iii) we devise a practical algorithm with synchronous coupling that effectively ensures the Lipschitz requirement for the generative model; (iv) extensive simulated and real-world experiments demonstrate that AIGB-Pearl achieves SOTA performance and verify the effectiveness of the developed techniques in enhancing safe and effective generalization.
this section cite: ['b9', 'b34', 'b26', 'b12', 'b38', 'b54', 'b22', 'b17', 'b1']

Section: PRELIMINARIES

this section cite: []

Section: PROBLEM STATEMENT
This work studies the auto-bidding problem for a single advertiser subject to a budget constraint. The auction mechanism follows a sealed-bid, second-price rule. The objective is to devise a bidding policy that maximizes the cumulative value of the impressions won over a finite bidding episode (e.g., a day) within a budget B > 0. As established in (He et al., 2021), the optimal bid for each impression i is proportional to its intrinsic value v i > 0, scaled by a non-negative factor a ≥ 0 that remains consistent across all impressions. Under this strategy, the advertiser wins an impression i if av i ≥ p i and pays p i upon winning, where p i > 0 is the market price. The Return on Investment (ROI) of impression i is defined as v i /p i , and we denote its upper bound as R m ≜ max i v i /p i . However, the scaling factor is unknown a priori, and impression volatility drives its continual change throughout the bidding process. Hence, a standard practice involves recalibrating the scaling factor a at fixed intervals of T ∈ N + time steps (Guo et al., 2024;He et al., 2021;Mou et al., 2022). This casts the auto-bidding to a sequential decision-making problem.
Specifically, the auto-bidding problem can be modeled as a Markov Decision Process (MDP) < S, A, R, P >. The state s t ≜ [t, ct-1 , x] ∈ S is composed of the current time step t ∈ [T ], the cost ratio ct-1 = c t-1 /B > 0 where c t-1 is the advertiser's cost for impressions won between time step t -1 and t, and a static advertiser-specific feature x that includes the budget and many other individual information. The action a t ∈ A denotes the calibrated scaling factor at time step t. The reward r t ≥ 0 describes the value of the impressions won between time steps t and t + 1, and P denotes the state transition rule. The auto-bidding problem can be formulated as:
max a1,a2,••• ,a T E st+1∼P(•|st,at) T t=1 r t , s.t. T t=1 c t ≤ B.(1)
Offline Setting. Due to safety concerns-common in real-world advertising systems-we are restricted to learning the optimal bidding policy from a static offline dataset D comprising historical states and actions along with associated rewards. This makes the considered auto-bidding problem an offline sequential decision-making task.
this section cite: ['b18', 'b17', 'b18', 'b34']

Section: OFFLINE RL METHODS
RL constitutes a standard approach for auto-bidding problems, seeking an optimal bidding policy π : S → A that maximizes cumulative reward. Specifically, this is typically achieved by learning a Q-value function, Q(s t , a t ) ≜ E π [ T t ′ =t r t ′ ], through temporal difference (TD) error minimization:
min Q E (st,at,rt,st+1)∼D [Q(s t , a t ) -r t -max at+1 Q(s t+1 , a t+1 )] 2 , (2
)
where Q is a target Q-value function with parameters updated via Polyak averaging (Mnih et al., 2015). Upon convergence, the optimal bidding policy is derived as π(s t ) = arg max at Q(s t , a t ).
Due to the offline setting of the considered auto-bidding problem, directly employing Eq. 2 results
in the infamous out-of-distribution (OOD) problem (Fujimoto et al., 2019), making the policy erroneously deviate from the offline dataset D (Mao et al., 2024a). As a standard solution, offline RL (Yu et al., 2020;Kumar et al., 2020;Mao et al., 2023;Kidambi et al., 2020) constrains the policy's behavior near D during TD learning, enabling reliable generalization beyond the offline dataset.
However, offline RL methods are notoriously unstable due to training instability caused by TDlearning (Peng et al., 2024), in which the bootstrapped value of the Q-function serves as its training label, resulting in an erroneous ground truth. Training stability is critical in auto-bidding due to the absence of an accurate offline policy evaluation method and the high cost of online policy evaluation in real-world advertising systems (Mou et al., 2022).
this section cite: ['b33', 'b12', 'b53', 'b26', 'b30', 'b23', 'b38', 'b34']

Section: GENERATIVE AUTO-BIDDING METHODS
Definition 1 (Trajectory and Trajectory Quality).
The trajectory is formalized as the state sequence throughout the bidding episode, i.e., τ ≜ [s 1 , s 2 , • • • , s T ]. The trajectory quality is defined as the normalized cumulative reward of the trajectory, i.e., y(τ ) ≜ T t=1 rtfoot_0 , where rt = r t /B.
Unlike RL methods, the AI-generated auto-bidding (AIGB) (Guo et al., 2024) treats the auto-bidding problem as a sequence generation task. Specifically, a conditional generative model is employed to fit the conditional trajectory distribution p θ (τ |y(τ )) within the offline dataset D, i.e.,
max θ E (τ,y(τ ))∼D [log p θ (τ |y(τ ))],(3)
where θ denotes the parameter. Let y m > 0 be the maximum trajectory quality in D, we have ∀y ∈ D, y ∈ [0, y m ]. During inference, AIGB follows a planning-and-control architecture. Specifically, at each time step, a trajectory is sampled from the trained generative model that acts as the planner, with a manually set condition y * ≜ (1 + ϵ)y m , where ϵ > 0 is a hyper-parameter. Then, an extra off-the-shelf inverse dynamic model (Agrawal et al., 2016), acting as the controller, is employed to compute the action. See Appendix B for detailed descriptions. AIGB avoids TD learning and generally outperforms offline RL methods (Guo et al., 2024).
However, AIGB relies on imitating trajectories from the offline dataset (Ajay et al., 2023) and lacks the ability to improve its generation quality beyond the offline dataset based on the performance feedback. Consequently, AIGB's conditional generation in the extrapolation regime (y * > y m ) can be unreliable without explicit reward guidance, rendering exploration undirected and potentially leading to performance degradation or even risky trajectory generation. Furthermore, no theoretical guarantee exists for the quality of the generated trajectory in this extrapolation regime.
this section cite: ['b17', 'b0', 'b17', 'b1']

Section: METHOD
Enabling AIGB to explore higher-quality trajectories beyond the offline dataset with explicit reward guidance can enhance its performance and generalization. To this end, we propose AIGB-Pearl (Planning with EvaluAtor via RL) that constructs a trajectory evaluator (hereinafter referred to as the evaluator for simplicity) to integrate RL methods into AIGB's planner training. Specifically, the evaluator learns a score ŷϕ (τ ) to estimate the trajectory quality y(τ ) via supervised learning based on the offline dataset D, i.e., min ϕ E τ ∼D [(ŷ ϕ (τ ) -y(τ )) 2 ], where ϕ denotes the evaluator parameter. Then, with ϕ fixed, the planner tries to maximize the score of its generation through iterative interactions with the evaluator, as shown in Fig. 2. Formally, this can be formulated as:
max θ L(θ) ≜ E τ ∼p θ (τ |y * ) [ŷ ϕ (τ )],(4)
where the condition is fixed to y * in both training and inference stages to ensure consistency.
As shown in Eq. 4, the effectiveness of AIGB-Pearl hinges critically on the evaluator's reliability. However, given the offline nature of the considered auto-bidding problem, the evaluator training is confined to the fixed dataset D. Although we incorporate several techniques into the evaluator's supervised training to enhance its prediction accuracy (as detailed in Section 3.2.1 and Appendix E.1), directly applying Eq. 4 to the planner can still trigger the infamous OOD problem due to the evaluator's generalization limits outside the data support, potentially degrading the planner's true performance. This issue is particularly acute in auto-bidding, a risk-sensitive domain in which suboptimal or anomalous trajectory generation can result in substantial monetary losses or campaign failures. Notably, there remains a lack of theoretically principled approaches to this challenge.
To address this challenge, we first analyze the theoretical bounds on the evaluator's bias in Section 3.1. Then, guided by this analysis, we propose a KL-Lipschitz-constrained score-maximization objective for the planner to ensure reliable use of the evaluator. Notably, this objective is theoretically justified by a sub-optimality bound established in Section 3.1.1. Finally, a practical algorithm is presented in Section 3.2, which employs a synchronous coupling method to satisfy the planner's Lipschitz constraint in Section 3.2.2.
this section cite: []

Section: KL-LIPSCHITZ-CONSTRAINED SCORE MAXIMIZATION
This section focuses on the reliable exploitation of the evaluator-guided score maximization.
Our basic idea is to optimize θ within a domain where the gap between the planner's score L(θ) and its true performance J(θ) ≜ E τ ∼p θ (τ |y * ) [y(τ )] is bounded by a small certifiable upper bound. This ensures the score maximization occurs only in regions where the evaluator is reliable.
|J(θ) -L(θ)| = |E τ ∼p θ (τ |y * ) [y(τ )] -E τ ∼p θ (τ |y * ) [ŷ ϕ (τ )]|.(5)
In the following, we investigate this gap. Specifically, we find that the trajectory quality y(τ ) is Lipschitz continuous as stated in Theorem 1. The proof is given in Appendix C.1.
Theorem 1 (Lipschitz Continuous of y(τ ).). The trajectory quality y(τ ) is √ T R m -Lipschitz continuous with respect to the Frobenius norm.
Motivated by Theorem 1, we enforce a √ T R m -Lipschitz regularity on the evaluator's training to inherit the Lipschitz continuity of the true trajectory quality y(τ ) (as described in Section 3.2.1). As the trained evaluator's Lipschitz constant may not equal √ T R m exactly, we denote it as k √ T R m , where k ≥ 0 quantifies the degree of violation. Note that a tighter adherence of the evaluator to the Lipschitz constraint yields a value of k closer to 1.
Equipped with Theorem 1 and the Lipschitz property of the evaluator, we derive the following upper bound on the performance gap between J(θ) and L(θ), and the proof is given in Appendix C.2. Theorem 2 (Evaluator Bias in Planning Performance Bound). Let the upper bound of the evaluator's bias on its training dataset D be δ D > 0, i.e., E τ ∼D |y(τ ) -ŷϕ (τ )| ≤ δ D , and let the Lipschitz constant of ŷϕ (τ ) be k √ T R m . The gap between the planner's score L(θ) and its true performance J(θ) can be bounded by:
|J(θ) -L(θ)| ≤ δ D + (1 + k) √ T R m E y∼p D (y) W 1 (p θ (τ |y * ), p θ (τ |y)) Generation sensitivity to y + W 1 (p θ (τ |y), p D (τ |y)) Imitation error on D ,
where W 1 denotes the 1-Wasserstein distance.
Note that δ D could be regulated to a small value via supervised training of the evaluator, and k depends on the Lipschitz property of the resulting evaluator as stated beforefoot_1 . Consequently, bounding the evaluator bias in the planner's performance requires constraining the following two factors:
• the planner's generation sensitivity to condition y (the first Wasserstein term)
• the planner's imitation error on the offline dataset (the second Wasserstein term).
Specifically, we establish that the first Wasserstein term can be bounded by the Lipschitz constant Lip W1 (p θ (τ |y)) of the planner with respect to the condition y measured by W 1 :
E y∼p D (y) [W 1 (p θ (τ |y * ), p θ (τ |y))] ≤ (1 + ϵ)y m Lip W1 (p θ (τ |y)). (6
)
The proof is given in Appendix C.3. Therefore, to ensure the boundedness of the first Wasserstein term, we constrain the planner's Lipschitz constant Lip W1 (p θ (τ |y)) to a positive hyperparameter L p . A lower bound analysis of L p is provided later in Eq. 10.
Moreover, we establish that a constrained KL divergence E y∼p D (y) [D KL (p D (τ |y)∥p θ (τ |y))] ≤ δ K could bound the expectation of the second Wasserstein distance term as follows, where δ K > 0 is a hyperparameter and can be set to a small value, close to zero. See Appendix C.4 for the proof. Note that the KL divergence constraint here inherently makes the planner perform conditional behavior cloning on the offline dataset D (Guo et al., 2025).
E y∼p D (y) [W 1 (p θ (τ |y), p D (τ |y))] ≤ δ K .(7)
Collectively, to effectively perform score maximization with a small, certifiable evaluator bias, we enforce Lipschitz continuity of the planner with respect to the condition y, while preserving its behavior cloning fidelity to the offline dataset D. Formally, Eq. 4 is transformed to:
max θ L(θ) (Score Maximization) (8) s.t. E y∼p D (y) [D KL (p D (τ |y)∥p θ (τ |y))] ≤ δ K (KL Constraint) (8a
) Lip W1 (p θ (τ |y)) ≤ L p (Lipschitz Constraint) (8b
)
Eq. 8 forms the score maximization objective in AIGB-Pearl. Remark 1. Intuitively, the KL and Lipschitz constraints jointly ensure the planner's generation under condition y * remains within a certified neighborhood of the high-quality trajectories in the offline dataset D. As illustrated in Fig. 1, the green region represents the feasible set of trajectories p θ (τ |y) satisfying the KL constraint for y ∈ Dfoot_2 . The Lipschitz constraint makes the generated trajectories p θ (τ |y * ) remain within a neighborhood of the best-quality trajectory in the offline dataset p θ (τ |y m ), as illustrated by the blue circle with radius ϵL p y m . The union of all such circles constitutes the total trajectory exploration range. Meanwhile, the evaluator trained on D maintains high accuracy within this Dproximal region, and the Lipschitz regularization on the evaluator bounds its sensitivity to input perturbations, preventing drastic value fluctuations in OOD regions and promoting more reliable extrapolation. Overall, KL-Lipschitz constrained optimization, guided by a Lipschitz continuity evaluator, enables safe trajectory improvement within a theoretically certified neighborhood of high-quality offline trajectories, effectively mitigating risky OOD exploration.
Published as a conference paper at ICLR 2026
this section cite: []

Section: SUB-OPTIMALITY GAP BOUND
This section focuses on presenting and analyzing the sub-optimality bound of the solution to the proposed Eq. 8. Specifically, denote the solution to the true performance J(θ) as θ * ≜ arg max θ J(θ), and denote the solution to the proposed Eq. 8 as θ. The following theorem gives the sub-optimality bound of the planner's performance, and the proof is given in Appendix C.5.
Theorem 3 (Sub-optimality Gap Bound). Let δ M ≜ E y∼p D (y) [D KL (p D (τ |y)∥p θ * (τ |y * ))] be the expected distance between the optimal trajectory distribution and the trajectory distribution of the offline dataset D. The true performance gap between the optimal parameter θ * and the solution θ to Eq. 8 is bounded by:
J(θ * ) -J( θ) ≤ 2δ D + (1 + 2k) √ T R m δ M + δ K + (1 + ϵ)y m L p .(9)
Theoretical Result Analysis. In Theorem 3, the constants δ M , R m , T, ϵ, y m characterize domainspecific properties of the auto-bidding task and the offline dataset D. Nonetheless, a lower training error δ D of the evaluator and a closer k to 1 correspond to a smaller sub-optimality gap. Note that k cannot be smaller than 1 without compromising δ D , as excessively small k prevents the evaluator from fitting the offline dataset D.
Moreover, in Theorem 3, a lower behavior cloning error δ K and a lower Lipschitz constant L p of the planner lead to a smaller sub-optimality gap. However, an excessively small L p prevents the planner from behavior cloning the offline dataset D (as required by the KL constraint), resulting in a large δ K . Actually, a theoretical lower bound for L p is given by the Lipschitz constant of the conditional trajectory distribution of the offline dataset p D (τ |y):
L p ≥ sup y1̸ =y2 W 1 (p D (τ |y 1 ), p D (τ |y 2 )) |y 1 -y 2 | .(10)
where y 1 , y 2 ∈ D. Consequently, we leverage this lower bound of L p in AIGB-Pearl. A further discussion on the theoretical performance range of AIGB-Pearl is given in Appendix D.
this section cite: []

Section: PRACTICAL ALGORITHM DESIGN
This section focuses on the practical algorithm implementation of Eq. 8. Section 3.2.1 first presents our reliability-enhanced evaluator architecture, followed by the synchronous-coupling-based Lipschitz planner design in Section 3.2.2.
this section cite: []

Section: LIPSCHITZ TRAJECTORY EVALUATOR
As shown in Fig. 2, the evaluator processes the trajectory τ to predict a score ŷϕ (τ ) for quality estimation. The evaluator is trained via supervised learning using the offline dataset D. Besides, to satisfy √ T R m -Lipschitz constraint requirement according to Theorem 2, we add Lipschitz regularization term to the training loss of the evaluator, which can be expressed as:
l e (ϕ) = E τ ∼D (ŷ ϕ (τ ) -y(τ )) 2 fitting the ground truth +β 1 E τ1,τ2 |ŷ ϕ (τ 1 ) -ŷϕ (τ 2 )| - √ T R m ∥τ 1 -τ 2 ∥ F + Lipschitz penalty ,(11)
where
β 1 > 0 is a hyper-parameter, [•] + ≜ max{0, •}.
Moreover, to further enhance the prediction accuracy of the evaluator, we devise two specific techniques, including the LLM Embedding enhancement and pair-wise learning, whose details are given in Appendix E.1.
this section cite: []

Section: LIPSCHITZ PLANNER WITH SYNCHRONOUS COUPLING
As shown in Fig. 2, the planner is implemented by a Causal Transformer (Chen et al., 2021) that generates trajectories in an auto-regressive manner. Specifically, the model takes the condition y and history states s 1:t as input tokens, and predicts the next state as a Gaussian distribution, p θ (s t+1 |s 1:t , y) = N (µ θ (s 1:t , y, t), σ 2 θ (s 1:t , y, t)), where µ θ denotes the mean and σ θ > 0 the standard deviation. During the auto-regressive generation process, each output state is sampled from the Gaussian distribution using the reparameterization trick, i.e., s t+1 = µ θ (s 1:t , y, t)+σ θ (s 1:t , y, t)•η t , where η t ∼ N (0, I)foot_3 .
this section cite: ['b7']

Section: Regularized Planner Training Loss.
To perform the score maximization in Eq. 8, we involve two regularization terms in the planner's training loss l p (θ), including a conditional behavior cloning loss, corresponding to the KL constraint Eq. 8a, and a Lipschitz penalty loss, corresponding to the Lipschitz constraint Eq. 8b, i.e.,
l p (θ) = -E τ ∼p θ (τ |y * ) [ŷ ϕ (τ )] planner score L(θ) -β 2 E (τ,y)∼p D [log p θ (τ |y)] conditional behavior clone + β 3 E y1,y2∈D∪{y * } W 1 (p θ (τ |y 1 ), p θ (τ |y 2 )) -L p |y 1 -y 2 | + Lipschitz penalty, where W1(p θ (τ |y1), p θ (τ |y2)) is replaced by Ŵ1(y1, y2; θ) , (12
)
where β 2 , β 3 > 0 are two hyper-parameters. With prior RL works (Sutton et al., 1999), we can derive the closed-form expression of planner's score gradient ∇ θ L(θ) as shown in Appendix C.6. The core of the planner loss lies in the computation of W 1 (p θ (τ |y 1 ), p θ (τ |y 2 )).
Wasserstein Upper Bound as Surrogate. Accurate computation of this Wasserstein distance term is challenging, as it requires finding the optimal coupling between p θ (τ |y 1 ) and p θ (τ |y 2 ) that minimizes the expected transportation cost. Nonetheless, we can choose a certain coupling γ ∈ Γ(p θ (τ |y 1 ), p θ (τ |y 2 )) to obtain an upper bound of this Wasserstein term, i.e.,
W 1 (p θ (τ |y 1 ), p θ (τ |y 2 )) ≜ inf γ∈Γ(p θ (τ |y1),p θ (τ |y2)) E (τ1,τ2)∼γ t ∥s 1 t -s 2 t ∥ ≤ E η 1:T ∼N (0,I) t ∥s 1 t -s 2 t ]∥ ≜ Ŵ1 (y 1 , y 2 ; θ).(13)
where Ŵ1 (y 1 , y 2 ; θ) denotes the upper bound, and s i t is the t-th state in trajectory τ i . It can be seen that Ŵ1 (y 1 , y 2 ; θ) ≤ L p |y 1 -y 2 | acts as a sufficient condition to make the planner L p -Lipschitz continuous. Thus, we replace W 1 (p θ (τ |y 1 ), p θ (τ |y 2 )) by this upper bound in the planner loss.
Synchronous Coupling Wasserstein. Instead of using random couplings, we employ a synchronous coupling γ sync to make the upper bound tighter. Specifically, two trajectories τ 1 and τ 2 -conditioned on y 1 and y 2 , respectively-are generated using the same sequence of Gaussian noise {η 1 , η 2 , ...η T }.  The definition of Ŵ1 (y 1 , y 2 ; θ) is given in Eq. 13. Compared to random couplings, the synchronous coupling reduces spurious variance in the trajectory comparison by aligning stochasticity through shared noise, resulting in a tighter upper bound on the Wasserstein distance (Lindvall, 2002).
Moreover, if we make the predicted variance σ θ of the planner a fixed constant, then the expression of Ŵ1 (y 1 , y 2 ; θ) can be further simplified to Ŵ1 (y 1 , y 2 ; θ) = t ∥µ θ (s 1 1:t , y 1 , t) -µ θ (s 2 1:t , y 2 , t)∥. The overall AIGB-Pearl algorithm is summarized in Algorithm 1 in Appendix E due to page limits.
this section cite: ['b43', 'b29']

Section: EXPERIMENTS
We conduct both simulated and real-world experiments to validate the effectiveness of our approach.
In the experiments, we mainly investigate the following Research Questions (RQs): (1) Does enhancing AIGB with policy optimization improve overall performance, and can it generalize better to unseen data compared to existing AIGB methods? (Section 4.2) (2) How does the KL-Lipschitz constraint affect the performance of the planner? (Section 4.3) (3) Can the proposed method guarantee the Lipschitz property of the evaluator and the planner? (Section 4.4). ( 4) What is the evaluator's accuracy on the training data, and how well does it generalize to unseen data? (Section 4.5). The training stability of AIGB-Pearl is studied in Appendix F.5.
this section cite: []

Section: EXPERIMENT SETUP
Experiment Environment. We conduct simulated experiments in an open-source offline advertising system with 30 advertisers of four budget levels (1.5k, 2.0k, 2.5k, and 3.0k), as in (Mou et al., 2022;Guo et al., 2024). The offline dataset comprises 5k trajectories generated by 20 advertisers. Extra detailed settings of simulated experiments are given in Appendix F.1. For real-world experiments, we conduct online A/B tests on one of the world's largest E-commerce platforms, TaoBao. The offline dataset comprises 200k trajectories of 10k advertisers. See Appendix F.2 for extra detailed settings of real-world experiments. In both simulated and real-world experiments, we employ the same inverse dynamics model from Agrawal et al. (2016) as the controller in AIGB. Moreover, the evaluator is trained on the entire offline dataset, and its generalization ability is evaluated using K-fold cross-validation with K = 5.
Baselines. We compare our method with state-of-the-art AIGB methods, including DiffBid (Guo et al., 2024) and DT (Chen et al., 2021), which learn from conditional behavior cloning of offline datasets using a diffusion model and a Causal Transformer, respectively. We also compare our method with RL auto-bidding methods, including USCB (He et al., 2021) that learns the autobidding policy in a manually constructed advertising system with DDPG (Silver et al., 2014); and offline RL auto-bidding methods, including model-free offline RL methods BCQ (Fujimoto et al., 2019), CQL (Kumar et al., 2020), and IQL (Kostrikov et al., 2022), and model-based offline RL methods MOPO (Yu et al., 2020) and MORL (Mou et al., 2025).  Performance Index. The objective in the auto-bidding problem Eq. 1, i.e., the cumulative rewards over the bidding episode, acts as the main performance index in our experiments and is referred to as the gross merchandise volume, GMV. In addition, we utilize three other metrics commonly used in the auto-bidding problem to evaluate the performance of our approach. The first metric is the total number of impressions won over the bidding episode, referred to as the BuyCnt. The second metric is the Cost over the bidding episode, and the third is the return on investment ROI, defined as the ratio of GMV to Cost. Note that larger values of GMV, BuyCnt, and ROI with a Cost oscillating within an acceptable tolerance (±2%) indicate a better performance.
this section cite: ['b34', 'b17', 'b0', 'b17', 'b7', 'b18', 'b40', 'b12', 'b26', 'b25', 'b53', 'b35']

Section: OVERALL PERFORMANCE
To answer RQ(1): Table 1 shows that our method consistently outperforms all baselines in GMV across all four budget levels in simulated experiments. In real-world experiments, Table 2 shows that our method also achieves superior performance on GMV, BuyCnt, and ROI, with Cost fluctuations of less than 2%. Notably, both simulated and real-world experiments consistently demonstrate that AIGB-Pearl achieves a +3% improvement in GMV over the AIGB, the state-of-the-art auto-bidding method. Since our method and DiffBid share the same controller, the performance gain stems solely from the planner. This provides strong empirical evidence that the proposed conservative RL learning for score maximization effectively enhances overall performance.
Notably, we also apply AIGB-Pearl to another important auto-bidding problem, named TargetROAS (See Appendix F.3). Real-world experiments show that AIGB-Pearl achieves a +5% improvement in GMV compared to AIGB. It is worth noting that a GMV uplift exceeding 2% is highly significant, translating to millions of RMB in additional daily GMV on Taobao-scale advertising platforms.
this section cite: []

Section: Generalization Ability.
We evaluate AIGB-Pearl on advertisers not used to generate trajectories in the offline dataset and compare it with existing AIGB methods. For simplicity, we refer to these advertisers as advertisers outside the offline dataset. Table 3 reports the performance on 4k advertisers outside the offline dataset in real-world experiments. AIGB-Pearl consistently delivers better results in GMV (+3%), BuyCnt, and ROI, while maintaining Cost fluctuations within 2% compared to the baselines. This indicates that the proposed method has better generalization ability than AIGB.
this section cite: []

Section: ABLATION STUDY
To answer RQ(2): We remove the KL and Lipschitz constraints from AIGB-Pearl individually and evaluate the model's performance in each ablated variant using real-world A/B tests. The results are presented in Table 4. It can be seen that the KL constraint contributes +1.1% improvement in GMV, and the Lipschitz constraint provides +1.8% improvement in GMV, demonstrating their respective roles in enhancing AIGB-Pearl's performance.
Visualization. Three AIGB-Pearl-generated trajectory examples are presented in Fig. 3. As can be observed, the trajectories generated by AIGB-Pearl are plausible. In contrast, the ablated variant without the KL and Lipschitz constraints produces trajectories that deviate significantly from the optimal trajectory in the offline dataset and exhibit clear pathological behaviors-such as excessive budget consumption, backward-trending pacing, and under-utilization of available budgets (see Appendix F.4 for explanation)-which further validate the KL-Lipschitz constraint necessity.    To answer RQ(3): We report that the Lipschitz value of the trajectory quality y(τ ) and the conditional trajectory distribution p D of the offline dataset are 1.62 and 0.38, respectively. We set L p = 0.50, which is close to its lower bound estimate of 0.38foot_4 . To calculate the Lipschitz constants of the evaluator and the planner, we sample 8, 000 pairs of trajectories and compute their Lipschitz constants. The results are shown in Fig. 4 and Fig. 5. It can be observed that most sample values satisfy the Lipschitz constraint, and the Lipschitz constants of the evaluator ŷϕ (τ ) and planner p θ (τ |y) are 2.2 and 0.56, respectively, near 1.62 and 0.50. This indicates that the models' Lipschitz constraints are successfully satisfied.
this section cite: []

Section: EVALUATOR ACCURACY EXAMINATION
Accuracy Metrics. The evaluator's accuracy is assessed along two dimensions, including the absolute accuracy measured by mean absolute error (MAE) metrics, reflecting how close the predicted scores are to ground truth scores, and the ranking accuracy by AUC metric, reflecting the correctness of relative rankings between trajectory pairs. Note that the MAE of each advertiser's data sample is normalized by its budget to ensure comparability across advertisers. A lower MAE, together with a higher AUC, indicates better evaluator accuracy.
To answer RQ(4): We report the accuracy of the trained evaluator in both simulated and realworld experiments in Table 5. We evaluate the evaluator's accuracy on the training data and its generalization ability using K-fold cross-validation, where K = 5. To the best of our knowledge, we are the first to introduce the trajectory evaluator into the generative auto-bidding framework. The reasonableness of our evaluator is evidenced by its pairwise ranking accuracies of 86% AUC and 75% AUC on OOD trajectories in simulated and real-world experiments, respectively, substantially above the 50% random-chance level, despite the high complexity and dynamic nature of the bidding environment. Importantly, with the guidance of the trained evaluator, the planner outperforms stateof-the-art AIGB methods even on OOD data, as demonstrated in the
Table. 3. 5 CONCLUSIONS This paper proposes AIGB-Pearl, which enhances AIGB by incorporating reward evaluation and policy optimization. By introducing a trajectory evaluator and a provably KL-Lipschitz-constrained score-maximization objective, our approach ensures safe and efficient generalization beyond the offline dataset, supported by theoretical guarantees. Extensive simulated and real-world experiments validate the state-of-the-art performance of our approach. CONTENTS 1 Introduction 2 Preliminaries 2.1 Problem Statement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.2 Offline RL Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2.3 Generative Auto-bidding Methods . . . . . . . . . . . . . . . . . . . . . . . . . . 3 Method 3.1 KL-Lipschitz-constrained score maximization . . . . . . . . . . . . . . . . . . . . 3.1.1 sub-optimality gap bound . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2 Practical Algorithm Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.2.1 Lipschitz Trajectory Evaluator . . . . . . . . . . . . . . . . . . . . . . . . 3.2.2 Lipschitz Planner With Synchronous Coupling . . . . . . . . . . . . . . . 4 Experiments 4.1 Experiment Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.2 Overall Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.3 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.4 Lipschitz Value Examination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4.5 Evaluator Accuracy Examination . . . . . . . . . . . . . . . . . . . . . . . . . . . 5 Conclusions A Related Works A.1 RL-based Auto-bidding Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . A.2 Generative Auto-bidding Methods . . . . . . . . . . . . . . . . . . . . . . . . . . B AIGB Method Details B.1 Training Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Inference Stage . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C Theoretical Proofs C.1 Proof of Theorem 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.2 Proof of Theorem 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.3 Proof of Eq. 6 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.4 Proof of Eq. 7 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.5 Proof of Theorem 3 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . C.6 Proof of Score Gradient . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
this section cite: []

Section: D Theoretical Performance Range Tradeoff Discussion

this section cite: []

Section: A RELATED WORKS
A.1 RL-BASED AUTO-BIDDING METHODS Auto-bidding plays a critical role in online advertising by automatically placing bids, allowing advertisers to participate efficiently in real-time auctions (Balseiro et al., 2021a;Deng et al., 2021;Balseiro et al., 2021b). The auto-bidding problem can be modeled as a Markov Decision Process and addressed using reinforcement learning techniques. USCB (He et al., 2021) proposes a unified solution to the constrained bidding problem, employing the RL method DDPG (Silver et al., 2014) to dynamically adjust parameters toward an optimal bidding strategy. Mou et al. (2022) propose a sustainable online reinforcement learning framework that alternates between online exploration and offline training, thereby alleviating the sim2rel problem. A few studies explore multi-agent RL for auto-bidding (Jin et al., 2018;Guan et al., 2021;Wen et al., 2022), while several focus on budget allocation and bidding strategies in multi-channel scenarios using RL-based approaches (Wang et al., 2023;Deng et al., 2023;Duan et al., 2025). Importantly, offline RL methods such as BCQ (Fujimoto et al., 2019), CQL (Kumar et al., 2020), IQL (Kostrikov et al., 2022), and MOPO (Yu et al., 2020) have demonstrated significant potential in this domain. These methods allow policy learning from pre-collected datasets without requiring online interaction. Moreover, offline RL, such as Diffusion-QL (Wang et al., 2022b), employs a generative policy architecture to improve expressive capacity.
However, RL-based methods often suffer from training instability arising from bootstrapping and alternating training between critics and actors. Training instability typically deteriorates policy performance (Sutton et al., 1998). Moreover, training stability is even more critical in auto-bidding, given two domain-specific challenges: the absence of an accurate offline policy evaluation method and the high cost of online policy evaluation in a real-world advertising system (Mou et al., 2022). Therefore, stable convergence to a well-performed policy is essential to ensure deployment reliability and system safety.
this section cite: ['b9', 'b18', 'b40', 'b34', 'b21', 'b15', 'b51', 'b47', 'b10', 'b11', 'b26', 'b25', 'b53', 'b42', 'b34']

Section: A.2 GENERATIVE AUTO-BIDDING METHODS
Generative models exhibit strong capabilities for capturing and reproducing underlying data distributions across a wide range of fields (Kingma & Welling, 2022;Goodfellow et al., 2020;Pan et al., 2023;Sohl-Dickstein et al., 2015;Ho et al., 2020;Vaswani et al., 2017). They can be effectively incorporated into decision-making systems by generating complete trajectories that guide agents toward high-reward behaviors (Zhu et al., 2023;Kang et al., 2023;Li et al., 2025). In particular, Decision Transformer (DT) (Chen et al., 2021) reframes RL as a conditional sequence modeling problem and leverages transformer architectures to generate actions conditioned on desired returns, historical states, and actions. AIGB (Guo et al., 2024) extends the generative perspective to the autobidding domain by formulating auto-bidding as a conditional generative modeling problem. DiffBid generates a state trajectory based on the desired return utilizing a conditional diffusion model, and then generates actions aligned with the optimized trajectory. These methods achieve superior performance in auto-bidding and offer distinct advantages over traditional RL methods. They do not rely on the bootstrapping mechanism commonly used in RL, thereby avoiding the instability caused by the deadly triad. Even so, these generative auto-bidding methods still face a performance bottleneck due to their neglect of fine-grained generation-quality evaluation and their inability to explore beyond static datasets. In contrast, our method facilitates both reward evaluation and policy search through a learned trajectory evaluator.
this section cite: ['b24', 'b14', 'b37', 'b41', 'b20', 'b45', 'b54', 'b22', 'b28', 'b7', 'b17']

Section: B AIGB METHOD DETAILS
AIGB models the sequential decision-making problem via conditional diffusion, enabling effective trajectory generation for auto-bidding scenarios. Specifically, AIGB utilizes the denoising diffusion probabilistic model (DDPM) (Ho et al., 2020) for generation. The forward and reverse processes are modeled as:
q(τ k+1 |τ k ), p θ (τ k |τ k+1 , y(τ )),(14)
respectively, where q represents the forward noising process while p θ the reverse denoising process.
Forward Process. In the forward process, the noise is gradually added to the latent variable by a Markov chain with pre-defined variance schedule β k :
q(τ k |τ k-1 ) = N (τ k ; 1 -β k τ k-1 , β k I)(15)
where k ∈ [K] refers to the diffusion step, τ k ≜ [s 1 , s 2 , • • • , s T ] k represents the latent variable in the k-th diffusion step, and τ 0 is the original trajectory. A notable property of the forward process is that τ k at an arbitrary time-step k can be sampled in closed form as:
q(τ k |τ 0 ) = N (τ k ; √ ᾱk τ 0 , (1 -ᾱk )I),(16)
where α k = 1 -β k and ᾱk = k i=1 α k . When k → ∞, τ k approaches a standard Gaussian distribution. In particular, AIGB employs a cosine noise schedule (Nichol & Dhariwal, 2021) to control the schedule β k .
Reverse Process. In the reverse process, diffusion models aim to remove the added noise on τ K and recursively recover τ 0 . This process is governed by the conditional model p θ (τ k-1 |τ k , y(τ )), which is parameterized through a noise prediction model ϵ θ (τ k , y(τ ), k). AIGB adopts a temporal U-Net (Ronneberger et al., 2015) for the noise prediction model, a common choice in diffusion-based decision-making methods (Ajay et al., 2023).
this section cite: ['b20', 'b36', 'b39', 'b1']

Section: B.1 TRAINING STAGE
The training of the diffusion model is typically formulated as minimizing the mean squared error between the predicted noise ϵ θ and the true noise applied during the forward diffusion process. Specifically, during each iteration, we randomly sample a trajectory from the offline dataset D and pick a time step t ∈ [T ]. We recursively add the Gaussian noise ϵ to the states in τ with time steps bigger than t and predict the added noises with ϵ θ (τ k , y(τ ), k), where the states between 0 and t in τ k are set to real history states s 1 , s 2 , • • • , s t . In addition to this standard objective, AIGB also incorporates a supervised loss that measures the discrepancy between the true actions and the actions predicted by an inverse dynamics model fϕ (s t , ŝt+1 ). Overall, the complete training objective of AIGB can be expressed as:
L(θ, ϕ) = E k,τ ∈D [||ϵ -ϵ θ (τ k , y(τ ), k)|| 2 ] + E (st,at,ŝt+1)∈D [||a t -fϕ (s t , ŝt+1 )|| 2 ]. (17
)
During training, the condition y(τ ) is randomly dropped to enhance model robustness. This technique ensures that both the unconditional model ϵ θ (τ k , k) and the conditional model ϵ θ (τ k , y(τ ), k) are effectively trained together.
this section cite: []

Section: B.2 INFERENCE STAGE
Starting with Gaussian noise, trajectories are iteratively generated through a series of denoising steps. Specifically, AIGB uses a classifier-free guidance strategy (Ho & Salimans, 2021) to guide the generation of bidding and extract high-likelihood trajectories in the dataset. During generation, AIGB combines conditional and unconditional score estimates linearly:
εk := ϵ θ (τ k , k) + ω (ϵ θ (τ k , y(τ ), k) -ϵ θ (τ k , k)) , (18
)
where ω is the guidance scale that controls the influence of the condition y(τ ). This formulation effectively steers the trajectory generation toward regions of the data distribution that are most consistent with the given condition. The predicted state at each step is sampled from p θ (τ k-1 |τ k , y(τ )):
τ k-1 ∼ N (τ k-1 |µ θ (τ k , y(τ ), k) , Σ θ (τ k , k)) ,(19)
with mean and variance defined as µ θ (τ k , y(τ
), k) = 1 √ α k (τ k -β k √ 1-α k εk )
and Σ θ (•) = β k . Note that the initial noisy trajectory τ ′ K ∼ N (0, I) is assigned with history states s 1:t for the first t states to ensure history consistency. This is consistent with the training process. By recursively applying the reverse diffusion process using:
τ ′ k-1 = µ θ (τ ′ k , y(τ ), k) + β k z,(20)
where z ∼ N (0, I), we obtain the final denoised trajectory τ ′ 0 , from which the next state ŝt+1 is derived. Then the action is generated through an inverse dynamics ât = fϕ (s t , ŝt+1 ). C THEORETICAL PROOFS C.1 PROOF OF THEOREM 1. Theorem 1 (Lipschitz Continuous of y(τ ).). The trajectory quality y(τ ) is √ T R m -Lipschitz continuous with respect to the Frobenius norm.
Proof. Recall from Section 2.1 that the cost c t and reward r t under action a t between time step t and t + 1 can be written as:
c t = i 1 a t ≥ p i t v i t p i t and r t = i 1 a t ≥ p i t v i t v i t ,(21)
where p i t and v i t denote the market price and the value of the i-th impression between time step t and t + 1. Accordingly, the cost ratio ct and the normalized reward rt can be written as:
ct = 1 B i 1 a t ≥ p i t v i t p i t and rt = 1 B i 1 a t ≥ p i t v i t v i t ,(22)
Consider two different trajectories τ 1 and τ 2 with actions, cost ratios and normalized rewards sequences {a 1,t , c1,t , r1,t } T t=1 and {a 2,t , c2,t , r2,t } T t=1 , respectively. The trajectory quality gap between τ 1 and τ 2 holds that:
|y(τ 1 ) -y(τ 2 )| = t r1,t - t r2,t ≤ t |r 1,t -r2,t |.(23)
Consider the reward gap between time step t and t + 1, as shown in Fig. 6. Without loss of generality, let a 2,t ≥ a 1,t . We have:
|r 1,t -r2,t | = 1 B i 1 a 2,t ≥ p i t v i t -1 a 1,t ≥ p i t v i t v i t = 1 B i 1 a 2,t ≥ p i t v i t ≥ a 1,t v i t = 1 B i 1 a 2,t ≥ p i t v i t ≥ a 1,t v i t p i t p i t ≤ R m B i 1 a 2,t ≥ p i t v i t ≥ a 1,t p i t .(24)
Note that the cost ratio gap between time step t and t + 1 can be written as:
|c 1,t -c2,t | = 1 B i 1 a 2,t ≥ p i t v i t -1 a 1,t ≥ p i t v i t p i t = 1 B i 1 a 2,t ≥ p i t v i t ≥ a 1,t p i t .(25)
Therefore, combining Eq. 24 and Eq. 25, we have:
|r 1,t -r2,t | ≤ R m |c 1,t -c2,t |.(26)
We examine the Frobenius norm of the gap between τ 1 and τ 2 :
∥τ 1 -τ 2 ∥ F =     1 c1,0 x 2 c1,1 x . . . . . . . . . T c 1,T -1 x     -     1 c2,0 x 2 c2,1 x . . . . . . . . . T c2,T -1 x     F = t (c 1,t -c2,t ) 2 ≥ 1 √ T t |c 1,t -c2,t | (Cauchy-Schwarz Inequality) (27
)
Combining Eq. 23, Eq. 26 and Eq. 27, we can obtain that:
|y(τ 1 ) -y(τ 2 )| ≤ t |r 1,t -r2,t | ≤ R m t |c 1,t -c2,t | ≤ √ T R m 1 √ T t |c 1,t -c2,t | ≤ √ T R m ∥τ 1 -τ 2 ∥ F . (28
)
This concludes the proof. Proof. Recall the Reverse Triangle Inequality states that ∀a, b, we have ||a| -|b|| ≤ |a -b|. Then, ∀x, y, we have:
||f 1 (x) + f 2 (x)| -|f 1 (y) + f 2 (y)|| ≤ |f 1 (x) + f 2 (x) -f 1 (y) -f 2 (y)| ≤ |f 1 (x) -f 1 (y)| + |f 2 (x) -f 2 (y)| ≤ (L 1 + L 2 )|x -y|. (29
)
This concludes the proof.
Lemma 2 (Kantorovich-Rubinstein Duality Theorem (Villani, 2021)). Let (X, d) be a metric space, and let p and q be two probability distributions on X. Let f : X → R be an L-Lipschitz function, and W 1 (p, q) denotes the 1-Wasserstein distance between p and q. Then we have:
|E x∼p f (x) -E x∼q f (x)| ≤ L • W 1 (p, q).(30)
We next give the proof of Theorem 2.
Theorem 2 (Evaluator Bias in Planning Performance Bound). Let the upper bound of the evaluator's bias on its training dataset D be δ D > 0. The gap between the planner's score L(θ) and its true performance J(θ) can be bounded by:
|J(θ) -L(θ)| ≤ δ D + (1 + k) √ T R m E y∼p D (y) W 1 (p θ (τ |y * ), p θ (τ |y)) Lipschitz sensitivity to y + W 1 (p θ (τ |y), p D (τ |y)) imitation error on D ,
where W 1 denotes the 1-Wasserstein distance.
Proof. The evaluator bias in the planner's performance can be written as:
|J(θ) -L(θ)| = |E τ ∼p θ (τ |y * ) [y(τ ) -ŷϕ (τ )]| ≤ E τ ∼p θ (τ |y * ) |y(τ ) -ŷϕ (τ )| ≜f (τ )(31)
Let f (τ ) ≜ |y(τ ) -ŷϕ (τ )| be the evaluator bias in trajectory τ . From Theorem 1 and Lemma 1, we know that f (τ ) is a (1 + k) √ T R m -Lipschitz continuous function. Then, we have:
|J(θ) -L(θ)| ≤ E τ ∼p θ (τ |y * ) f (τ ) = E y∼p D (y) E τ ∼p θ (τ |y * ) f (τ ) -E τ ∼p D (τ |y) f (τ ) + E τ ∼p D (τ |y) f (τ ) = E y∼p D (y) E τ ∼p D (τ |y) f (τ ) ≤δ D +E y∼p D (y) E τ ∼p θ (τ |y * ) f (τ ) -E τ ∼p D (τ |y) f (τ ) = δ D + E y∼p D (y) E τ ∼p θ (τ |y * ) f (τ ) -E τ ∼p θ (τ |y) f (τ ) ≤(1+k) √ T RmW1(p θ (τ |y * ),p θ (τ |y)), (Lemma 2) + E y∼p D (τ ) E τ ∼p θ (τ |y) f (τ ) -E τ ∼p D (τ |y) f (τ ) ≤(1+k) √ T RmW1(p θ (τ |y),p D (τ |y)), (Lemma 2) ≤ δ D + (1 + k) √ T R m E y∼p D (y) [W 1 (p θ (τ |y * ), p θ (τ |y))] + (1 + k) √ T R m E y∼p D (y) [W 1 (p θ (τ |y), p D (τ |y))].(32)
Therefore, we have:
|J(θ) -L(θ)| ≤ δ D + (1 + k) √ T R m E y∼p D (y) W 1 (p θ (τ |y * ), p θ (τ |y)) + W 1 (p θ (τ |y), p D (τ |y)) .(33)
This concludes the proof.
this section cite: ['b19', 'b46']

Section: C.3 PROOF OF EQ. 6
We give the proof of Eq. 6 as follows. Denote Lip W1 (p θ (τ |y)) as the planner's Lipschitz constant with respect to y regarding the Wasserstein distance W 1 , we have:
E y∼p D (y) [W 1 (p θ (τ |y * ), p θ (τ |y))] ≤ Lip W1 (p θ (τ |y))E y∼p D (y) [((1 + ϵ)y m -y)] = Lip W1 (p θ (τ |y)) ym 0 p D (y)[(1 + ϵ)y m -y]dy ≤ Lip W1 (p θ (τ |y)) ym 0 p D (y)[(1 + ϵ)y m ]dy = (1 + ϵ)y m Lip W1 (p θ (τ |y)),(34)
where we leverage the non-negativity property of the condition y ≥ 0, ∀y ∈ D. This completes the proof.
this section cite: []

Section: C.4 PROOF OF EQ. 7
Lemma 3 (Pinsker's Inequality (Tsybakov, 2008)). Let P and Q be two probability measures defined on the same measurable space, and assume that P is absolutely continuous with respect to Q, i.e., P ≪ Q. Then the total variation distance between P and Q is bounded above by the KL divergence from P to Q as follows:
∥P -Q∥ TV ≤ 1 2 D KL (P ∥Q).(35)
Lemma 4 (Wasserstein-Total Variation Inequality on Bounded Metric Spaces (Villani et al., 2008)). Let (Z, d) be a metric space with diameter diam(Z) ≜ sup z1,z2∈Z d(z 1 , z 2 ). Let P and Q be two probability measures on Z. Then the 1-Wasserstein distance between P and Q satisfies:
W 1 (P, Q) ≤ diam(Z)∥P -Q∥ TV .(36)
We give the proof of Eq. 7 as follows. Equipped with the above two lemmas, we have:
W 1 (p θ (τ |y), p D (τ |y)) ≤ diam(T )∥p θ (τ |y) -p D (τ |y)∥ TV ≤ diam(T ) 1 2 D KL (p D (τ |y)∥p θ (τ |y)),(37)
where T is the trajectory space. Note that due to the budget constraint t c t ≤ Bfoot_5 , we have the sum of the cost ratio satisfies t ct ≤ 1. The trajectory space can be expressed as:
T = [1, c0 , x], [2, c1 , x], • • • , [T, cT -1 , x] ct ≥ 0, ∀t, and t ct ≤ 1(38)
We next prove that the diameter of the trajectory space, diam(T ), can be bounded by a constant. Specifically, the diameter only depends on the largest possible distance between the cost ratio sequences in two trajectories since:
diam(T ) = sup τ1,τ2∈T ∥τ 1 -τ 2 ∥ F = sup τ1,τ2∈T     1 c1,0 x 2 c1,1 x . . . . . . . . . T c 1,T -1 x     -     1 c2,0 x 2 c2,1 x . . . . . . . . . T c2,T -1 x     F = sup τ1,τ2∈T t (c 1,t -c2,t ) 2 .(39)
For convenience, we let
c i ≜ [c i,0 , ci,2 , • • • , ci,T -1 ], i ∈ {1, 2}.
Then, the key part in the above result, t (c 1,t -c2,t ) 2 , can be written as:
t (c 1,t -c2,t ) 2 = t (c 2 1,t -2c 1,t c2,t + c2 2,t ) = ∥c 1 ∥ 2 2 + ∥c 2 ∥ 2 2 -2⟨c 1 , c 2 ⟩ ≤ ∥c 1 ∥ 2 2 + ∥c 2 ∥ 2 2 ,(40)
where ⟨c 1 , c 2 ⟩ ≥ 0. As ci,t ≥ 0 and t ci,t ≤ 1, we have 0 ≤ ci,t ≤ 1. Therefore, it holds that:
∥c i ∥ 2 2 = t c2 i,t ≤ t ci,t ≤ 1.(41)
Combining Eq. 40 and Eq. 41, we have:
t (c 1,t -c2,t ) 2 ≤ ∥c 1 ∥ 2 2 + ∥c 2 ∥ 2 2 ≤ √ 2.(42)
Therefore, we have diam(T ) = √ 2. According to Eq. 37, we have:
W 1 (p θ (τ |y), p D (τ |y)) ≤ D KL (p D (τ |y)∥p θ (τ |y)).(43)
Recall that we impose the KL-constraint as:
E y∼p D (y) [D KL (p D (τ |y)∥p θ (τ |y))] ≤ δ K ,(44)
Taking the expectation over y ∼ p D (y) on both sides of Eq. 43, we have:
E y∼p D (y) [W 1 (p θ (τ |y), p D (τ |y))] ≤ E y∼p D (y) D KL (p D (τ |y)∥p θ (τ |y)) ≤ E y∼p D (y) [D KL (p D (τ |y)∥p θ (τ |y))] (Jensen Inequality) ≤ δ K (KL constraint Eq.44).
(45) This completes the proof. C.5 PROOF OF THEOREM 3
Theorem 3 (Sub-optimality Gap Bound). Let δ M ≜ E y∼p D (y) [D KL (p D (τ |y)∥p θ * (τ |y))]
be the expected distance between the optimal trajectory distribution and the trajectory distribution of the offline dataset D. The true performance gap between the optimal parameter θ * and the solution θ to Eq. 8 is bounded by:
J(θ * ) -J( θ) ≤ 2δ D + (1 + 2k) √ T R m δ M + δ K + (1 + ϵ)y m L p .(46)
Proof. The sub-optimality gap can be expressed as follows:
J(θ * ) -J( θ) = J(θ * ) -L(θ * ) + L(θ * ) -L( θ) + L( θ) -J( θ) ≤ |J(θ * ) -L(θ * )| evaluator bias in p θ * + |L(θ * ) -L( θ)| score gap + |L( θ) -J( θ)| evaluator bias in p θ .(47)
We examine the above three terms accordingly.
(1) Evaluator Bias in p θ * . Denote the evaluator bias on trajectory τ as f (τ
) ≜ |y(τ ) -ŷϕ (τ )|.
Following the derivation process in Eq. 32, we have:
|J(θ * ) -L(θ * )| ≤ E y∼p D (y) E τ ∼p D (τ |y) f (τ ) + E y∼p D (y) E τ ∼p θ * (τ |y * ) f (τ ) -E τ ∼p D (τ |y) f (τ ) ≤ δ D + (1 + k) √ T R m E y∼p D (y) [W 1 (p θ * (τ |y * ), p D (τ |y))],(48)
where W 1 (p θ * (τ |y * ), p D (τ |y)) denotes the probability distribution distance between the optimal planner and the offline dataset. Based on the derivation in Appendix C.4, we have:
E y∼p D (y) [W 1 (p θ * (τ |y * ), p D (τ |y))] ≤ E y∼p D (y) [D KL (p D (τ |y)∥p θ * (τ |y * ))] (49
) Let δ M ≜ E y∼p D (y) [D KL (p D (τ |y)∥p θ * (τ |y *
))] be the distance between the optimal trajectory distribution and the offline dataset trajectory distribution. We have:
|J(θ * ) -L(θ * )| ≤ δ D + (1 + k) √ T R m δ M .(50)
(2) Score Gap. Recall that the trained evaluator ŷϕ (τ ) is a k √ T R m -Lipschitz continuous function with the Lipschitz constraint design. With Lemma 2, we have:
|L(θ * ) -L( θ)| = |E τ ∼p θ * (τ |y * ) ŷϕ (τ ) -E τ ∼p θ (τ |y * ) ŷϕ (τ )| ≤ k √ T R m W 1 (p θ * (τ |y * ), p θ (τ |y * )) ≤ k √ T R m E y∼p D (y) W 1 (p θ * (τ |y * ), p D (τ |y)) + W 1 (p D (τ |y), p θ (τ |y * )) ≤ k √ T R m δ M + (1 + ϵ)y m L p + δ K ,(51)
Figure 7: Performance range of the planner with respect to hyper-parameters δ k , L p and ϵ. There is a trade-off in the selection of hyper-parameters: larger δ k , L p , ϵ results in a smaller lower bound but a higher upper bound.
where we leverage the fact that θ is a solution to Eq. 8 which satisfies the KL and Lipschitz constraint, and we leverage the results in Eq. 6, Eq. 7 and Eq. 49.(3) Evaluator Bias in p θ . Since θ satisfies the KL and Lipschitz constraint in Eq. 8, we can use the results in Theorem 2, Eq. 6 and Eq. 7 to obtain:
|J( θ) -L( θ)| ≤ δ D + (1 + k) √ T R m [(1 + ϵ)y m L p + δ K ],(52)
Overall, combining the results in Eq. 50, Eq. 51, and Eq. 52, we have:
J(θ * ) -J( θ) ≤ 2δ D + (1 + 2k) √ T R m δ M + δ K + (1 + ϵ)y m L p .(53)
This concludes the proof.
this section cite: ['b44', 'b46']

Section: C.6 PROOF OF SCORE GRADIENT
The probability of the trajectory generated by the Causal Transformer can be decomposed into: p θ (s 1:T |y) = Π t p θ (s t |s 1:t-1 , y).
(54) Then, we have:
∇ θ L(θ) = ∇ θ τ p θ (τ |y * )ŷ ϕ (τ )dτ = ∇ θ s1,••• ,s T p θ (s 1:T , τ |y * )ŷ ϕ (τ )ds 1 • • • ds T = s 1:T p θ (s 1:T |y * ) ∇ θ p θ (s 1:T |y * ) p θ (s 1:T |y * ) ŷϕ (τ )ds 1 • • • ds T = E s 1:T ∼p θ (s 1:T |y * ) ∇ θ log p θ (s 1:T |y * )ŷ ϕ (τ ) = E s 1:T ∼p θ (s 1:T |y * ) ∇ θ log Π t p θ (s t |s 1:t-1 , y * )ŷ ϕ (τ ) = E s 1:T ∼p θ (s 1:T |y * ) t ∇ θ log p θ (s t |s 1:t-1 , y * )ŷ ϕ (τ ) . (55
)
this section cite: []

Section: D THEORETICAL PERFORMANCE RANGE TRADEOFF DISCUSSION
We note that Theorem 3 actually gives the lower bound of the planner's true performance J( θ), i.e.,
J(θ * ) -2δ D + (1 + 2k) √ T R m δ M + δ K + (1 + ϵ)y m L p ≜f (δ K ,Lp,ϵ) ≤ J( θ),(56)
where we denote that gap term f as a function of hyper-parameters δ K , L p and ϵ that we can adjust. As f (δ K , L p , ϵ) is monotonically increasing with respect to δ K , L p and ϵ, smaller values of these terms result in a higher lower bound of J( θ). However, as illustrated in Fig. 1, these three terms also determine the planner's exploration range, denoted as D e (δ k , L p , ϵ). Specifically, higher values of δ k , L p , ϵ indicate a larger exploration range, which thereby leads to a higher performance upper bound of the planner, i.e.,
J( θ) ≤ max p(τ ) E τ ∼p(τ ) [y(τ )] s.t., supp(p(τ )) ⊆ D e (δ k , L p , ϵ).(57)
This introduces a trade-off in the selection of hyper-parameters. As illustrated in Fig. 7, higher δ k , L p , ϵ results in a smaller lower bound but a higher upper bound. To this end, we conduct hyperparameter tuning and give hyper-parameter determination in Appendix F.8.
this section cite: []

Section: E AIGB-PEARL ALGORITHM SUMMARY
Algorithm 1 summarizes the training process of AIGB-Pearl. Specifically, we compute the Lipschitz constants of y(τ ) and p D (τ |y) using the offline dataset D and, accordingly, determine the Lipschitz constraints for the evaluator and the planner, respectively. Then, we sequentially perform evaluator learning, planner pretraining, and KL-Lipschitz-constrained score maximization for the planner. The development of AIGB-Pearl is supported by ROLL (Wang et al., 2025).
this section cite: ['b49']

Section: E.1 ADDITIONAL DESIGNS FOR EVALUATOR ACCURACY ENHANCEMENT
To further enhance the reliability of the trajectory evaluator, we design two additional techniques for the evaluator learning. Specifically, as described in the following, we (i) integrate an LLM embedding into its input feature for better representational capacity; and (ii) employ the pair-wise loss for better score estimation accuracy. Fig. 8 illustrates the complete evaluator learning losses. The effectiveness of these two methods is studied in Appendix F.6.
this section cite: []

Section: E.1.1 LLM EMBEDDING ENHANCEMENT
Motivated by the success of integrating user-specific features into recommendation systems (Chen et al., 2015), we incorporate advertiser-specific features into the trajectory evaluator to enhance its representational capacity and improve scoring accuracy. Note that some advertiser-specific features are textual in nature-such as product titles, categories, and reviews-and are therefore difficult to incorporate directly into the vectorized trajectory τ . To address this, we construct a prompt containing all such textual attributes and employ a pre-trained large language model (LLM), QWen2.5-1.5B-Instruct, with general world knowledge to extract T representation, which we refer to as the LLM embedding. Specifically, we use the output of the last hidden layer as the LLM embedding. This embedding is then used as an additional positional encoding in the evaluator. The prompt template is given below:
LLM Prompt Template. I am an [advertising platform] advertiser, operating the [brand name] brand in the category of [category name], classified as a [advertiser tier] tier advertiser. I have a product titled [product name] currently running in advertising campaigns. This product belongs to the leaf category of [leaf category], priced at [product price], and is positioned in the [price range]. Its price ranks within the top [price ranking in the leaf category] % in the leaf category. Historical Average Performance: The product generates an average of [average daily transactions] daily transactions, with a GMV of [average daily GMV], driven by advertising. It receives an average [average daily impressions] daily impressions from search and recommendation traffic, [click numbers] clicks, [average daily BuyCnt] BuyCnt, and a GMV of [GMV]. It ranks within the top [sales volume ranking in leaf category] % in sales volume in the leaf category, with an average transaction value of [average transaction value]. Historical Time-based Average Performance: This product has undergone continuous exposure to advertising for number of advertising days days. The average hourly advertising spend distribution per day (from 0:00 to 24:00) is historical spend distribution sequence. The average GMV distribution across its category during this period is historical GMV distribution sequence. The average daily advertising budget is daily advertising budget. E.1.2 PAIR-WISE LOSS Unlike human feedback scores used in LLM post-training (Christiano et al., 2017)-which can suffer from subjective biases in their absolute valuations-the trajectory quality y(τ ) has real physical meaning and is comparable among different trajectories. Therefore, we can adopt a hybrid pointwise and pair-wise loss for the score to capture the absolute value of y(τ ) and their relative preference, respectively.
This approach has demonstrated superior performance in recommender systems (Cao et al., 2007;Lei et al., 2017;Wang et al., 2022a). Specifically, the pair-wise loss function can be expressed as:
E (τw,τ l )∼Dp log σ(ŷ ϕ (τ w ) -ŷϕ (τ l )) ,(58)
where the pair-wise loss is implemented by the typical Bradley-Terry (BT) function (Bradley & Terry, 1952). Here, D p = {(τ w , τ l , y(τ w ), y(τ l ))} denotes a pair-wise dataset extracted from the offline dataset D, where τ w denotes the trajectory with higher trajectory quality, i.e., y(τ w ) ≥ y(τ l ).
this section cite: ['b6', 'b5', 'b27', 'b4']

Section: E.1.3 OVERALL EVALUATOR LOSS
Combining the previous techniques, the overall evaluator loss is:
l e (ϕ) = E τ ∼D (ŷ ϕ (τ ) -y(τ )) 2 point-wise loss +β 4 E (τw,τ l )∼Dp log σ(ŷ ϕ (τ w ) -ŷϕ (τ l )) pair-wise loss + β 1 E τ1,τ2 |ŷ ϕ (τ 1 ) -ŷϕ (τ 2 )| - √ T R m ∥τ 1 -τ 2 ∥ F + Lipschitz penalty ,(59)
where β 4 > 0 is a hyper-parameter.
this section cite: []

Section: F ADDITIONAL EXPERIMENTS

this section cite: []

Section: F.1 SIMULATED EXPERIMENT SETTINGS
We include the detailed simulated experiment settings in
Table 6. Specifically, we consider the bidding process in a day, where the bidding episode is divided into 96 time steps. Thus, the duration Algorithm 1: AIGB-Pearl (Planning with EvaluAtor via RL) Input : Offline dataset D, desired condition y * , hyper-parameters β 1 , β 2 , β 3 . Output : Optimized θ and ϕ Initialization: randomly initialized planner parameter θ, trajectory evaluator parameters ϕ // Determining the Lipschitz Value Calculate the Lipschitz value of y(τ ) and p D (τ |y) using the offline dataset D. Set the Lipschitz constraint value L e for the evaluator and L p for the planner to be bigger than the Lipschitz value of y(τ ) and p D (τ |y), respectively. // Training the trajectory evaluator while not converged do Update ϕ by minimizing Eq. 11; end // Training the generative planner Warm start with pretrained planner p θ ; while not converged do Generate bidding trajectories τ ∼ p θ (τ |y * ); Score generated trajectories with frozen ϕ: ŷϕ (τ ); Update θ by maximizing Eq. 12. end We include the detailed real-world experiment settings in Table 7. Specifically, we consider the bidding process in a day, where the bidding episode is divided into 96 time steps. Thus, the duration between two adjacent time steps t and t + 1 is 15 minutes. The number of impression opportunities between time steps t and t + 1 fluctuates from 100 to 2,500. The minimum and maximum budgets of advertisers are 50 Yuan and 10,000 Yuan, respectively. The upper bound of the bid price is 25 Yuan, and the values of impressions are positive.
Hardware Resource. The training process in the real-world experiments is conducted using 10 NVIDIA T4 Tensor Core GPUs in a distributed manner. For each distributional worker, we use 10 CPUs and 200 GB of memory.  F.3 REAL-WORLD EXPERIMENTS ON TARGETROAS BIDDING PROBLEM
In addition to the budget-constrained auto-bidding problem, we also apply the proposed AIGB-Pearl algorithm to a more challenging type of auto-bidding problem, named TargetROAS, with an extra ROI (Return on Investment) constraint. We evaluate our method in a real-world experiment on TaoBao involving 300k advertisers over 22 days. The offline dataset comprises 16 million trajectories of 800k advertisers. The results are given in Table 8. AIGB-Pearl achieves a +5.1% improvement in GMV compared to the SOTA auto-bidding method, DiffBid, demonstrating its effectiveness in managing more complex and realistic constraints.
this section cite: []

Section: F.4 PATHOLOGICAL TRAJECTORY BEHAVIOR EXPLANATION
In industrial practice, stable and effective metrics have been developed to evaluate pathological behaviors. For the case of the budget-constrained auto-bidding problem with bidding cycles structured as 24-hour episodes (T = 96 time steps), the following three key metrics are commonly used to identify pathological behaviors:
• Excessive budget consumption: there exists a time step t such that the cost between time step t and t + 1 exceeds 10% of the budget B;
• Forward-(or Backward-) trending pacing: the cost between time step 1 and 24 (or between time step T -24 and T ) exceeds 40% (or 40%) of the budget B;
• under-utilization of available budgets: the total cost over the bidding episode is lower than 90% of the budget B.  F.5 TRAINING STABILITY We present additional comparisons between the training curves of the offline RL with bootstrapping and those of AIGB-Pearl in Fig. 10, Fig. 11, Fig. 12, and Fig. 13 concerning:
• Cumulative Rewards: the main performance index of the considered auto-bidding problem;
• Online Rate: the ratio between the bidding period before the budget runs out and the total bidding period. A larger Online Rate indicates a better performance.
• Bad Case Rate: the ratio between the number of "bad" trajectories and the total number of generated trajectories. A lower Bad Case Rate indicates a better performance.
• Cost Rate: the ratio between the cost and the budget. A larger cost rate indicates a better performance.
It can be observed that the offline RL method tends to exhibit significant instability throughout training, with high variance across different seeds. In contrast, AIGB-Pearl achieves much smoother and more consistent learning progress, demonstrating the improved training stability.
this section cite: []

Section: F.6 EVALUATOR ACCURACY
Accuracy Metric. We evaluate the accuracy of the evaluator along two dimensions, including the absolute accuracy, reflecting how close the predicted scores are to ground truth qualities, and the order accuracy, reflecting the correctness of relative rankings between trajectory pairs. Specifically, we use the symmetric mean absolute percentage error, SMAPE, as the metric for the absolute accuracy and the AUC, defined as the ratio of correctly predicted ordinal pairs to the total number of pairs, as the metric for the order accuracy. The SMAPE ranges from 0% to 200%, and the AUC ranges from 0% to 100%. Lower SMAPE and larger AUC indicate better evaluator accuracy.
In the following, we investigate the effectiveness of using LLM embeddings and a pairwise loss for evaluator learning.
this section cite: []

Section: LLM Embedding Effectiveness.
We examine the accuracy of the trajectory evaluator without using LLM embeddings, and the results are reported in the lines of "w/o LLM" in Table 9 and Table 10. It can be observed that LLM embeddings can improve both absolute and order accuracy. Fig. 9 compares the training progress with and without LLM embeddings in terms of the SMAPE
this section cite: []

Section: G EXTENDING AIGB-PEARL TO FIRST-PRICE AUCTIONS
We note that the proposed method remains effective in first-price auctions with a proper adaptation. Specifically, unlike second-price auctions where the optimal bid for impression i takes the form bid i = av i , in first-price auctions, the optimal bid for impression i is given by bid i = min(av i , p i ), which typically involves an extra bid shading method to predict the winning price (Gligorijevic et al., 2020;Wu et al., 2015). Equipped with an off-the-shelf bid shading method (whose design is beyond the scope of this work), the auto-bidding problem in a first-price auction remains an offline sequential decision problem, i.e., making decisions over an a-sequence, to which the proposed method applies directly.
To validate the effectiveness of the proposed method under first-price auctions, we additionally conduct a simulated first-price auction experiment against the state-of-the-art AIGB method, where all methods are equipped with the same bid shading method. The results are presented in Table 13, demonstrating the effectiveness of our proposed method.
this section cite: ['b13', 'b52']

Section: H EXTENDING AIGB-PEARL TO ONLINE SETTING DISCUSSION
We note that AIGB-Pearl can be extended to online settings when equipped with a safe online exploration policy. Specifically, due to safety constraints, the auto-bidding policy during training cannot interact directly with the live advertising system; only safe exploration policies are permitted to collect data online (Mou et al., 2022). Consequently, an online auto-bidding framework typically involves two parts:
• a safe online exploration policy, which is a well-established component in existing work (Mou et al., 2022) and beyond the scope of this paper;
• an offline policy training method that leverages the data collected online.
AIGB-Pearl can be directly applied as the offline policy training method within the online framework without modification. In practice, due to the safety and stability concerns, many industrial autobidding systems adopt an offline optimization paradigm. For this practical reason, we focus on the offline setting in this work.
this section cite: ['b34', 'b34']

Section: I LLM USAGE
The authors have used Large Language Models (LLMs) exclusively for grammar checking and lexical refinement during the writing process. No LLM-generated content, data analysis, or substantive contributions to the research methodology, results, or conclusions are involved in this work.
this section cite: []

Section: 
G Extending AIGB-Pearl to first-price auctions H Extending AIGB-Pearl to Online Setting Discussion I LLM Usage 0 500 1000 1500 2000 Steps 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Online Rate Offline RL with Bootstrapping seed:1 seed:2 seed:3 seed:4 seed:5 seed:6 seed:7 seed:8 seed:9 seed:10 0 500 1000 1500 2000 Steps 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 Online Rate AIGB-Pearl seed:1 seed:2 seed:3 seed:4 seed:5 seed:6 seed:7 seed:8 seed:9 seed:10 Figure 11: Learning curves of online rate between offline RL with bootstrapping method and AIGB-Peral under 10 seeds. 0 500 1000 1500 2000 Steps 0.0 0.1 0.2 0.3 0.4 Bad Case Rate Offline RL with Bootstrapping seed:1 seed:2 seed:3 seed:4 seed:5 seed:6 seed:7 seed:8 seed:9 seed:10 0 500 1000 1500 2000 Steps 0.0 0.1 0.2 0.3 0.4 Bad Case Rate AIGB-Pearl seed:1 seed:2 seed:3 seed:4 seed:5 seed:6 seed:7 seed:8 seed:9 seed:10 metric. As illustrated, the evaluator incorporating LLM embeddings converges faster and achieves a lower SMAPE than the one without LLM embeddings. The performance gain stems from LLM embeddings' ability to encode high-level semantic information, thereby facilitating a more nuanced understanding of dependencies among sequential bidding states.
Hybrid Point-wise and Pair-wise Losses Effectiveness. Table 9 and Table 10 present the SMAPE and AUC of the trajectory evaluator when trained with point-wise loss only, pair-wise loss only, and a combination of both. It can also be seen that using only pairwise loss yields significantly worse SMAPE performance, despite some improvement in AUC. This suggests that while pairwise loss can enhance ranking consistency, it falls short of providing accurate absolute-value predictions.
When both point-wise and pair-wise losses are used together, the evaluator achieves lower SMAPE and higher AUC. This indicates that combining these two loss types not only improves absolute accuracy but also enhances order accuracy in trajectory evaluation.
this section cite: []

Section: F.7 EMPIRICAL PERFORMANCE WITH GENERAL OFFLINE DATA DISTRIBUTIONS
Note that in many real-world auto-bidding systems, including the one considered in the paper, due to operational safety constraints, the online-deployed bidding policy is typically a single fixed model, and the offline dataset is collected exclusively from this single policy over multiple days, where an advertiser contributes a single trajectory per day. For example, in the considered auto-bidding system, the online-deployed baseline policy is a conditional generative model that, given a given advertiser and identical conditions, generates identical trajectory plans each day. The variation across  different trajectories of the same advertiser in the offline dataset is solely due to stochastic environmental factors (e.g., traffic fluctuations). Since these exogenous perturbations are typically the sum of many independent impression-level sources of noise, the resulting trajectory deviations can be reasonably approximated as a Gaussian distribution.
To demonstrate the broad applicability of the proposed algorithm, we evaluate its performance in settings where multiple policies are used for data collection. Specifically, we collect trajectories in the simulated environment using nine distinct bidding policies, thereby constructing an offline dataset with a multi-modal distribution that violates the Gaussian distribution. The empirical results are presented in Table 11. It can be observed that AIGB-Pearl still outperforms AIGB by 4.9%, indicating that its performance is robust to the offline dataset's specific distribution.
this section cite: []

Section: F.8 HYPER-PARAMETER TUNING
We conduct sensitivity experiments with respect to the hyper-parameter δ k . For hyper-parameter L p , we leverage the lower bound given in Eq. 10 in the main experiments. For the hyper-parameter ϵ, we use the same empirical value of 5% as in AIGB in the main experiments, which is typically set based on operational requirements.
Specifically, in the planner loss given in Eq. 12, β 2 is the penalty factor corresponding to the KL constraint. Actually, we control the KL divergence δ k by tuning β 2 . Table . 12 gives the hyperparameter tuning results. It can be observed that, as long as δ k is neither too large (in which case the algorithm degenerates into AIGB with a pure demonstration likelihood maximization term) nor too small (e.g., β 2 = 0, which completely removes the demonstration likelihood maximization), the proposed method consistently outperforms the original AIGB. This also validates the discussion on performance bounds in Appendix D, which demonstrates that a moderate δ k balances the lower and upper bounds, thereby yielding optimal performance.
this section cite: []

Section: References
Ref_id:b0 Title: Learning to poke by poking: Experiential learning of intuitive physics Year: (2016)
Ref_id:b1 Title: Is conditional generative modeling all you need for decision making? Year: (2023)
Ref_id:b2 Title: Robust auction design in the auto-bidding world Year: (2021)
Ref_id:b3 Title: The landscape of auto-bidding auctions: Value versus utility maximization Year: (2021)
Ref_id:b4 Title: Rank analysis of incomplete block designs: I. the method of paired comparisons Year: (1952)
Ref_id:b5 Title: Learning to rank: from pairwise approach to listwise approach Year: (2007)
Ref_id:b6 Title: Recommender systems based on user reviews: the state of the art Year: (2015)
Ref_id:b7 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b8 Title: Deep reinforcement learning from human preferences. Advances in neural information processing systems Year: (2017)
Ref_id:b9 Title: Towards efficient auctions in an autobidding world Year: (2021)
Ref_id:b10 Title: Multichannel autobidding with budget and roi constraints Year: (2023)
Ref_id:b11 Title: An adaptable budget planner for enhancing budget-constrained auto-bidding in online advertising Year: (2025)
Ref_id:b12 Title: Off-policy deep reinforcement learning without exploration Year: (2019)
Ref_id:b13 Title: Bid shading in the brave new world of first-price auctions Year: (2020)
Ref_id:b14 Title: Generative adversarial networks Year: (2020)
Ref_id:b15 Title: Multi-agent cooperative bidding games for multi-objective optimization in e-commercial sponsored search Year: (2021)
Ref_id:b16 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b17 Title: Generative auto-bidding via conditional diffusion modeling Year: (2024)
Ref_id:b18 Title: A unified solution to constrained bidding in online display advertising Year: (2021)
Ref_id:b19 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b20 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b21 Title: Real-time bidding with multi-agent reinforcement learning in display advertising Year: (2018)
Ref_id:b22 Title: Efficient diffusion policies for offline reinforcement learning Year: (2023)
Ref_id:b23 Title: Morel: Modelbased offline reinforcement learning Year: (2020)
Ref_id:b24 Title: Auto-encoding variational bayes Year: (2022)
Ref_id:b25 Title: Offline reinforcement learning with implicit qlearning Year: (2022)
Ref_id:b26 Title: Conservative q-learning for offline reinforcement learning Year: (2020)
Ref_id:b27 Title: Alternating pointwise-pairwise learning for personalized item ranking Year: (2017)
Ref_id:b28 Title: Generative models in decision making: A survey Year: (2025)
Ref_id:b29 Title: Lectures on the coupling method Year: (2002)
Ref_id:b30 Title: Supported value regularization for offline reinforcement learning Year: (2023)
Ref_id:b31 Title: Offline reinforcement learning with ood state correction and ood action suppression Year: (2024)
Ref_id:b32 Title: Doubly mild generalization for offline reinforcement learning Year: (2024)
Ref_id:b33 Title: Human-level control through deep reinforcement learning Year: (2015)
Ref_id:b34 Title: Sustainable online reinforcement learning for auto-bidding Year: (2022)
Ref_id:b35 Title: Permutation equivariant model-based offline reinforcement learning for auto-bidding Year: (2025)
Ref_id:b36 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b37 Title: Better training of gflownets with local credit and incomplete trajectories Year: (2023)
Ref_id:b38 Title: Deadly triad matters for offline reinforcement learning Year: (2024)
Ref_id:b39 Title: U-net: Convolutional networks for biomedical image segmentation Year: (2015)
Ref_id:b40 Title: Deterministic policy gradient algorithms Year: (2014)
Ref_id:b41 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b42 Title: Reinforcement learning: An introduction Year: (1998)
Ref_id:b43 Title: Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems Year: (1999)
Ref_id:b44 Title: Introduction to Nonparametric Estimation Year: (2008)
Ref_id:b45 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b46 Title: Cédric Villani et al. Optimal transport: old and new Year: (2008)
Ref_id:b47 Title: Hibid: A cross-channel constrained bidding system with budget allocation by hierarchical offline deep reinforcement learning Year: (2023)
Ref_id:b48 Title: Mp2: A momentum contrast approach for recommendation with pointwise and pairwise learning Year: (2022)
Ref_id:b49 Title: Reinforcement learning optimization for largescale learning: An efficient and user-friendly scaling library Year: (2025)
Ref_id:b50 Title: Diffusion policies as an expressive policy class for offline reinforcement learning Year: (2022)
Ref_id:b51 Title: A cooperative-competitive multi-agent framework for autobidding in online advertising Year: (2022)
Ref_id:b52 Title: Predicting winning price in real time bidding with censored data Year: (2015)
Ref_id:b53 Title: Mopo: Model-based offline policy optimization Year: (2020)
Ref_id:b54 Title: Diffusion models for reinforcement learning: A survey Year: (2023)
