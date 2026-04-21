Title: PiKE: Adaptive Data Mixing for Large-Scale Multi-Task Learning Under Low Gradient Conflicts
Abstract: Modern foundation models are trained on diverse datasets to enhance generalization across tasks and domains. A central challenge in this process is determining how to effectively mix and sample data from multiple sources. This naturally leads to a multi-task learning (MTL) perspective. While prior work in MTL has emphasized mitigating gradient conflicts, we observe that large-scale pretraining scenarios-such as multilingual or multi-domain training-often exhibit little to no gradient conflict. Motivated by this observation, we propose PiKE (Positive gradient interaction-based K-task weights Estimator), an adaptive data mixing algorithm that dynamically adjusts sampling weights during training. PiKE exploits non-conflicting gradient interactions to minimize a near-tight upper bound on the average loss decrease at each step, while incurring negligible computational overhead. We provide theoretical convergence guarantees and show that PiKE outperforms static and non-adaptive mixing baselines. Furthermore, we extend PiKE to promote balanced learning across tasks. Extensive experiments on largescale language model pretraining confirm that PiKE achieves faster convergence and improved downstream performance compared to existing approaches.

Section: Introduction
Foundation models, such as large language models (LLMs), owe their strong generalization and multitask abilities to pretraining on diverse datasets spanning multiple domains [62,36,10]. The effectiveness of these models depend heavily on the composition of their training data [29,18]. However, standardized practices for curating optimal pretraining data are lacking. Common approaches involve heuristic filtering, deduplication, and categorization into heterogeneous domains (e.g., The Pile [22] has 22 domains; GLaM [18] has 6). Even after such preprocessings, determining the optimal data mixing remains a key challenge-amplified by the scale of modern models and datasets.
A common strategy in pre-training is to use fixed data mixtures, typically chosen heuristically or via smaller proxy models. For example, mT5 [71] weights datasets by relative size, while GLaM [18] uses downstream performance from proxy models. DoReMi [69] also relies on proxy models, using group distributionally robust optimization (group DRO) to set dataset weights. However, these approaches have notable limitations. First, their optimality is unclear: heuristic methods lack theoretical backing, and policies from small models may not necessarily transfer to larger ones [74]. Second, proxy models introduce substantial computational overhead, often scaling linearly or worse with the number of domains. Third, static weights fixed at initialization may become suboptimal as training progresses [77,34], a limitation discussed further in Section 2.
In this work, we frame adaptive data mixture selection problem as a multitask optimization problem, enabling a principled approach to dynamically adjusting data mixture. This view is natural: each data domain typically is related to a (set of) tasks and yields a distinct gradient. Prior multitask optimization methods [75,66,9,16] focus on resolving gradient conflicts that impede convergence. However, faster. Right (750M models, GLaM six domains): PiKE achieves a 3.4% higher accuracy over DoReMi [69].
PiKE scales efficiently with model size and number of tasks. Detailed results are in Tables 15 and 16.
most are impractical for current LLMs due to their O(Kd) memory cost for storing K gradients or O(K 2 ) computation from repeated backpropagation. A notable exception is FAMO [38], which scales more efficiently. Additionally, most MTL methods are tailored for vision tasks where gradient conflicts are prevalent, a condition less common in LLM pretraining: For example, GradVaccine [66] observed that in multilingual BERT (178M parameters), task gradients are mostly positively aligned or nearly orthogonal. We extend this finding to much larger autoregressive, decoder-only models (e.g., 1B parameters), which better reflect modern LLMs. Our results show minimal gradient conflict in multilingual and multi-domain pretraining (Section 2), suggesting that modern LLMs naturally exhibit cooperative (or sometimes nearly-orthogonal) gradient structures-potentially reducing the need for explicit conflict-mitigation. This motivates the central question of our work:
Can we design an effective adaptive data mixing strategy based on multitask optimization, that exploits non-conflicting nature of gradients and scales to large settings with minimal memory and computational overhead?
To address this, we propose PiKE, an adaptive data mixing that is empirically effective and scalable for pretraining LLMs. PiKE enjoys theoretical guarantees while remaining practical for large-scale settings-with many tasks, large models, and diverse data-at negligible memory and compute overhead. We summarize PiKE's key features below; related work is discussed in Appendix 5.
this section cite: ['b61', 'b9', 'b28', 'b17', 'b21', 'b17', 'b70', 'b17', 'b68', 'b73', 'b76', 'b33', 'b74', 'b65', 'b15', 'b68', 'b37', 'b65']

Section: Key Features of PiKE
1. Adaptively adjusts mixture weights using per-task gradient magnitude and variance. 2. Theoretically, achieves near-optimal per-iteration objective decrease and enjoys convergence guarantees (Section 3). 3. Incorporates tilted empirical risk minimization [35,46], promoting balanced learning across tasks and preventing task under-representation (Section 3.3). 4. Scales efficiently (linearly) with model size and the number of tasks (Section 4). 5. Consistently outperforms existing methods across different scales (110M to 1B parameters) and scenarios (multilingual to domain mixing) (Figure 1 and Section 4).
this section cite: ['b34', 'b45']

Section: Problem Definition and Notations
We aim to train a single model with parameters θ ∈ R d to perform K ≥ 2 tasks simultaneously. Each task k is associated with a smooth (possibly non-convex) loss ℓ k (θ, x) : R d × R dx → R where x is the data point. It is common to minimize the total loss:
min θ∈R d L(θ) := K k=1 E x∼D k [ℓ k (θ; x)],(1)
where D k represents the data distribution for task k. We define L k (θ) := E x∼D k [ℓ k (θ; Task gradient cosine similarity for a 750M GPT-2 style model pre-trained on GLaM datasets. "data1-data2" indicates gradient similarity between tasks data1 and data2. Further results in Appendix F.2.
L-Lipschitz if ∥h(θ)h(θ ′ )∥ ≤ L∥θθ ′ ∥ for any θ, θ ′ in the domain of h(•). A function f (•) is L-smooth if its gradient is L-Lipschitz continuous.
this section cite: []

Section: Main Building Blocks for PiKE
This section presents our main observations which form the main building blocks of PiKE.
this section cite: []

Section: Bulding Block #1: Mixing Domains per Batch Improves LLM Generalization
Optimizing equation (1) with stochastic methods, such as Adam or SGD, requires forming batches from one or more tasks at each step. The batch selection strategy strongly influences model performance [4,24,74,69,40]. Even with fixed data proportions, a key question remains: how should one form a batch from K data domains at each step?
Batch construction is a critical design choice in multitask training, as it directly impacts learning dynamics and final model performance [4,24,74,69,40]. We focus on three standard strategies: Random, Round-Robin, and Mix. Let us first define these methods assuming static, uniform sampling weights (i.e., 1/K per task). Let b t = (b t,1 , . . . , b t,K ) denote the number of samples from each task D k at iteration t, with total batch size b = K k=1 b t,k . The strategies are defined as:
b t =    b • e k * Random, where k * ∼ Uniform({1, . . . , K}) b • e (t mod K)+1 Round-Robin b • 1 K , . . . , 1 K Mix
where e k ∈ R K is the k-th standard basis vector. That is, Random selects one domain per batch, Round-Robin cycles through domains, and Mix includes all tasks in each batch.
Historically, Mix has been widely used in computer vision multitask settings [14,45,9,54,75,38], while earlier language modeling efforts favored Random or Round-Robin [41,44,42]. Recent studies on large-scale language models [17,51,7,61] have revisited this question and found that Mix generally performs best-especially in diverse-data settings [18,11,69,22,65]. Our experiments reaffirm this trend: across a wide range of tasks and scales, Mix consistently outperforms the alternatives (Figure 2, Figure 4). This observation underpins the design of our proposed method, PiKE.
this section cite: ['b0', 'b23', 'b73', 'b68', 'b39', 'b23', 'b73', 'b68', 'b39', 'b13', 'b44', 'b53', 'b74', 'b37', 'b40', 'b43', 'b41', 'b16', 'b50', 'b60', 'b17', 'b10', 'b68', 'b21', 'b64']

Section: Bulding Block #2: Large-Scale Training: Low Gradient Conflict & MTL Scalability Challenges
The Mix batching strategy offers a natural lens for analyzing data mixing in multitask learning (MTL). When a batch of total size b is constructed with b k samples from each task k, the overall gradient at iteration t, denoted g t , is the average of the individual sample gradients:
g t = 1 b K k=1 b k i=1 ∇ℓ k (θ t ; x k,i ) = K k=1 b k b ḡt,k ,(2)
where ḡt,k = 1 b k b k i=1 ∇ℓ k (θ t ; x k,i ) is the average gradient from task k, and x k,i ∼ D k . This formulation highlights how batch gradients blend contributions from different tasks and serves as the foundation for analyzing task interactions.
A central challenge in prior MTL literature is gradient conflict [75,66,47], where task gradients oppose the overall update direction. Formally, a conflict exists if: ⟨g t , ḡt,k ⟩ < 0, indicating that the shared update could increase the loss for task k. While such conflicts are well-documented in vision tasks, we observe they are rare in large-scale language model pretraining. In particular, as shown in Figures 2, 5, and 6, task gradients are generally non conflicting in pretraining LLMs. This observation shifts the our goal: instead of mitigating conflict, we can leverage naturally non-conflicting gradients to improve training efficiency.
This insight renders many conflict-aware MTL methods-such as PCGrad [75], AdaTask [73], MGDA [56], and NashMTL [47]-less effective for pre-training LLMs: PCGrad acts only when conflicts occur (rare in LLMs) and thus performs similarly to simple Mix. AdaTask treats different tasks as completely independent and thus does not utilizes potential cooperative nature of tasks, leading to slower convergence. More fundamentally, these methods incur substantial memory and compute overhead, making them unsuitable for large-scale models (see Appendix A for discussions).
A notable exception is the recent work FAMO [38], which is designed for scalability.
In summary, the lack of gradient conflict in LLM pretraining, combined with the limitations of existing MTL approaches, motivates the need for scalable methods that exploit non-conflicting gradient structures-rather than fixating on rare conflicts.
this section cite: ['b74', 'b65', 'b46', 'b74', 'b72', 'b55', 'b46', 'b37']

Section: Bulding Block #3: Adaptively Changing Mix Sampling Weights Is Necessary
Prior work using the Mix sampling strategy typically relies on fixed (static) sampling weights, keeping (b 1 , . . . , b K ) constant throughout training. However, dynamically adjusting batch composition can significantly enhance efficiency. We illustrate this fact with a simple example: Example 2.1. Consider training on K = 2 tasks with losses ℓ 1 (θ;
x 1 ) = 1 2 (θ ⊤ e 1 ) 2 + x ⊤ 1 θ and ℓ 2 (θ; x 2 ) = 1 2 (θ ⊤ e 2 ) 2 + x ⊤ 2 θ
, where e 1 = [1 0] ⊤ , e 2 = [0 1] ⊤ , and θ ∈ R 2 . Data for task 1 follows x 1 ∼ N (0, σ 2  1 I), while task 2 follows x 2 ∼ N (0, σ 2 2 I). The loss for task k simplifies to L k (θ) = 1 2 (θ ⊤ e k ) 2 . Using b k samples from task k, the gradient at iteration t is given by
g t = 1 b 1 + b 2 b 1 e 1 e ⊤ 1 + b 2 e 2 e ⊤ 2 θ t + z,
where z ∼ N (0,
b1σ 2 1 +b2σ 2 2 b 2 I) with b = b 1 + b 2 . Updating θ t via SGD, θ t+1 = θ t -ηg t , we have E[L(θ t+1 )] = 1 2 (1 -η b 1 b ) 2 θ 2 1,t + 1 2 (1 -η b 2 b ) 2 θ 2 2,t + η 2 b 1 σ 2 1 + b 2 σ 2 2 b 2 , (3
)
where θ 1,t and θ 2,t denote the first and second component of the vector θ t . The derivation details of equation ( 3) can be found in Appendix G.1. Letting w 1 := b1 b , w 2 := b2 b , and relaxing them to take real values, we can optimize the mixing weights w 1 and w 2 as
w * 1 = Π b -1 (σ 2 2 -σ 2 1 ) + η -1 (θ 2 1,t -θ 2 2,t ) + θ 2 2,t θ 2 1,t + θ 2 2,t(4)
and w * 2 = 1w * 1 where Π(ξ) = min{max{ξ, 0}, 1} is the projection onto [0, 1]. This result shows that optimal batch composition should change over iterates to maximize training efficiency.
Figure 3 illustrates the superiority of the adaptive mixing approach based on equation (4) over various static mixing strategies. Moreover, the adaptive mixing strategy in this example does not require any hyperparameter tuning, while finding the best static mixing requires tuning.
Despite its simplicity, this example mirrors key aspects of MTL in large models: 1) The optimal solution θ * = 0 minimizes all task losses simultaneously, reflecting the high expressive power of large models. 2) Task gradients are non-conflicting, resembling real-world gradient interactions observed in Building Block #2. Moreover, equation (4) further reveals that optimal data mixing depends on (1) the gradient norm squared per task ∥∇L 1 (θ)∥ 2 = θ 2  1 , ∥∇L 2 (θ)∥ 2 = θ 2 2 and (2) gradient variance (σ 2  1 , σ 2 2 ). As we will see in the next section, these factors play a crucial role in defining optimal mixing strategies for more general settings.
this section cite: []

Section: Method

this section cite: []

Section: PiKE: Conceptual Version
To develop our method, we first start by quantifying gradient conflicts: Definition 3.1. For a given point θ, we say gradients are c-conflicted (with c ≥ 0) if, for all task pairs j, k, j ̸ = k,
-c ∥∇L j (θ)∥ 2 + ∥∇L k (θ)∥ 2 ≤ ⟨∇L j (θ), ∇L k (θ)⟩ .
The above definition is implied by a lower bound on the gradients cosine similarity. In particular, if ⟨∇Lj (θ),∇L k (θ)⟩ ∥Lj (θ)∥∥L k (θ)∥ ≥ -c, then the gradients are c-conflicted for c = c/2. Therefore, experiments in Section 2 and Figures 5 and 6 in Appendix F.2 show that c is typically small for LLM training.
While task gradient conflict are rare in LLM training, we observed in Section 2 that task gradients are not fully aligned either. To quantify the level of alignment, we define the following concept: Definition 3.2. For a given point θ, we say that the gradients are c-aligned (with c ≥ 0) if, for all task pairs j, k, j ̸ = k,
⟨∇L j (θ), ∇L k (θ)⟩ ≤ c∥∇L j (θ)∥ 2 ∥∇L k (θ)∥ 2 .
Notice that Definition 3.1 and 3.2 always hold for c = 1 and c = 1/2. However, smaller values allow for more refined analysis and more desirable properties. Notably, when both c and c are small (as we observed in our experiments), the value of ∥∇L(θ)∥ is small if and only if ∥∇L k (θ)∥ is small for all k (see Lemma G.1 in Appendix G). To proceed further, we make the following standard assumption: Assumption 3.3. Per-task gradients are L-Lipschitz, unbiased, and have bounded variance, i.e., ∀k :
∥∇L k (θ 1 ) -∇L k (θ 2 )∥ ≤ L∥θ 1 -θ 2 ∥, ∀θ 1 , θ 2(5)
E x∼D k [∇ℓ k (θ; x)] = ∇L k (θ), ∀θ(6)
E x∼D k [∥∇ℓ k (θ; x) -∇L k (θ)∥ 2 ] ≤ σ 2 k , ∀θ(7)
Using a mix batch with b k samples per task k, the estimated gradient follows equation (2). The next theorem characterizes the descent amount in one step of SGD under low conflict conditions:
E[L(θ t+1 )] ≤ L(θ t ) + K k=1 b k - η b β∥∇L k (θ t )∥ 2 + Lη 2 2b 2 σ 2 k + K k=1 b 2 k Lη 2 2b 2 γ∥∇L k (θ t )∥ 2 (8)
for SGD update θ t+1 = θ tηg t . Here, β ≜ min k (1 + c(-K + 2 -b b k )), γ ≜ 1 + c(K -1), and the expectation is over batch sampling randomness under the mix strategy (b 1 , . . . , b K ).
A formal proof is provided in Theorem G.3 (Appendix G). Theorem 3.4 provides an upper-bound on the decrease in the objective depending on the mix batch composition (b 1 , . . . , b K ). As we show in Theorem G.4 in the appendix, this upperbound is tight in the non-conflicting, non-aligned regime of c = c = 0 (which is a good approximation for pre-training LLMs according to our experiments).
According to Theorem 3.4, to maximize descent in Mix sampling, we need to minimize the RHS of equation (8). Thus, relaxing w k = b k /b to continuous values, we need to solve:
min w1,...,w K ≥0 K k=1 w k λ k + 1 2 w 2 k κ k s.t. K k=1 w k = 1 (9
)
where
λ k ≜ -ηβ∥∇L k (θ)∥ 2 + Lη 2 2b σ 2 k and κ k ≜ Lη 2 γ∥∇L k (θ)∥ 2 .
Using KKT conditions, the optimal solution is given by w * k = max 0, -
µ+λ k κ k
, where µ is chosen such that
E∥∇L k ( θ)∥ 2 ≤ δ, ∀k = 1, . . . , K.(10)
Particularly, if we choose η = βδ Lσ 2 max /b+Lηδ , then the Conceptual PiKE algorithm requires at most
T = 2L∆ L (σ 2 max /b+γδ) δ 2 β 2
iterations to find a point satisfying equation (10).
This theorem (restated as Theorem G.5) is proved in Appendix G. Remark 1. Theorem 3.5 guarantees that, given enough iterations, all tasks are learned jointly, i.e., the gradients of all losses become small. Moreover, the convergence rate is T = O(1/δ 2 ), which matches the optimal iteration complexity for smooth, nonconvex stochastic optimization.
Remark 2. The Conceptual PiKE algorithm maximizes the expected decrease at each iteration by finding the optimal mix batching strategy. This property leads to improved iteration complexity bound than (static) uniform mix batching, as discussed in Appendix G.3, when c and c are small.
this section cite: ['b1', 'b7', 'b9']

Section: PiKE: Simplified Computationally Efficient Version
Solving equation (9) requires estimating {∥∇L k (θ t )∥ 2 } K k=1 and {σ k } K k=1 ; however, per-iteration estimation of these terms with sufficient accuracy often necessitates large batch computations, thereby impeding convergence. To speed up the algorithm, we can update these estimates every T 0 iterations. However, this can cause abrupt changes in (w 1 , . . . , w K ), leading to instability, especially with optimizers like Adam, where sudden shifts may disrupt momentum estimates. To mitigate this, we update (w 1 , . . . , w K ) using a single mirror descent step on equation ( 9), ensuring gradual adjustments:
w k ← w k exp αη(β -Lηγw k ) gradient norm ∥∇L k (θ)∥ 2 - αLη 22b
gradient variance σ 2 k followed by normalization: w ← w/∥w∥ 1 , where α is the mirror descent step size.
Even after the above simplifications, fine-tuning L, γ, α, and β can be challenging in practice. We simplify this finetuning by noting two observations: 1) The coefficient in front of σ 2 k is constant, independent of w k . 2) For small η and w k < 1, the coefficient in front of ∥∇L k (θ)∥ remains nearly constant: αη(β -Lηγw k ) ≈ αηβ. Therefore, in our practical implementation, PiKE employs tunable constant coefficients for terms related to task-specific gradient variance and gradient norms, which simplifies its application. The resulting algorithm is detailed in Algorithm 1. if t mod T 0 = 0 then 5: Estimate ∥∇L k (θ t )∥ 2 and σ 2 k for every k 6:
w k ← w k exp ζ 1 ∥∇L k (θ t )∥ 2 -ζ2 2b σ 2 k 7:
w ← w/∥w∥ 1 8:
(b 1 , . . . , b K ) ← round(b(w 1 , . . . , w K )) 9:
end if 10:
Sample b k data points from each task k 11:
Compute the gradient g using the estimates samples 12:
Update: θ t+1 ← Optimizer(η, θ t , g) 13: end for Remark 3. The inclusion of the variance-related term, σ 2 k , is crucial in PiKE; its omission would cause PiKE to disproportionately prioritize tasks with currently large losses even if their gradients are very noisy (high variance). Our ablation study, presented in Appendix F.6, underscores the importance of this term for achieving robust and balanced multi-task performance.
this section cite: []

Section: Balanced-PiKE: Balanced Learning Across Different Tasks
While Algorithm 1 optimizes the average loss across tasks equation (1), this objective can lead to unbalanced learning. In practice, some tasks may dominate due to larger datasets or easier loss reduction, causing others to be under-optimized. This imbalance is problematic when certain domains (e.g., code, math) are more important to user, or when balanced performance across all tasks is desired. To address this shortcoming, we consider a balance-promoting objective based on tilted empirical risk minimization [35], also known as the α-fairness utility [46]:
min θ L(τ ; θ) := 1 τ log K k=1 e τ L k (θ) . (11
)
This objective interpolates between average loss (τ → 0) and worst-case loss (τ → ∞), allowing users to control the trade-off between efficiency and fairness. Moderate values of τ (τ ≈ 1 ∼ 3 in our experiments) encourage balanced learning without sacrificing overall performance.
To leverage our PiKE developments in solving equation (11), we need to connect it to our intital objective in equation (1). The following lemma achieves this through the use of Fenchel duality [53]: Lemma 3.6. Assume L k (θ) > 0, ∀k and let 0 < τ ∈ R. Then equation ( 11) is equivalent to solving
min θ max y∈R K + K k=1 y k =τ K k=1 y k L k (θ) - K k=1 y k τ log y k τ .(12)
Moreover, for any fixed θ, the inner maximization problem is maximized at
y ⋆ k = τ e τ L k (θ) K j=1 e τ L j (θ) , ∀k.
This lemma, which is proved in Appendix G.4, provides a natural alternative minimization algorithm for solving equation (11): Fixing y, equation (12) reduces to a weighted minimization over tasks, where regular PiKE sampling with proper weights y k in front of each loss can be applied to determine the optimal mixing strategy. On the other hand, fixing θ, the optimal solution y ⋆ k can be computed according to Lemma 3.6. This leads to Balanced-PiKE algorithm, described in Appendix D, which balances overall loss minimization and balanced/fair learning of all tasks. Although prior MTL literature explored fair learning [3,47], such methods often grapple with scalability limitations, a critical aspect where PiKE is designed to excel through efficient large-scale performance.
this section cite: ['b34', 'b45', 'b10', 'b0', 'b52', 'b10', 'b11', 'b2', 'b46']

Section: Experiments
We evaluate PiKE in two multitask pretraining scenarios: 1) Pretraining language models on multilingual mC4 dataset [71], a dataset covering diverse languages from Common Crawl corpus. 2) Pretraining language models on the GLaM dataset [18], an English dataset spanning six domains. As we will show, across multiple model sizes (110M, 270M, 750M, and 1B parameters), PiKE consistently outperforms all existing methods, including heuristic or static data mixing, multitask optimization, and adaptive data mixture approaches. We first start by explaining our setup.
this section cite: ['b70', 'b17']

Section: Experiment Setup
Baselines: We evaluate a range of sampling strategies: (1) (Uniform) Mix, (2) Round-Robin, (3) Random, (4) FAMO [38], (5) ADO [30], (6) GLaM [18], (7) DoReMi [69], (8) PiKE, and (9) Balanced-PiKE. Among these, DoReMi, GLaM, and ADO are specifically designed for LLM pretraining. We also include FAMO, a recent multitask learning (MTL) method, because it is scalable to large model sizes and has demonstrated competitive performance-serving as a useful comparison point for our work. DoReMi estimates task weights by training a small proxy model, while GLaM assigns static weights based on downstream performance from smaller models. However, since both methods report weights only for the GLaM dataset and do not provide configurations for multilingual C4, we exclude them from our multilingual experiments. In contrast to these static methods, PiKE dynamically updates task sampling weights during training using gradient information, enabling adaptive optimization throughout training. PiKE is scalable to both large models and many tasks, with minimal overhead, as detailed in Appendix F.4.
this section cite: ['b37', 'b29', 'b17', 'b68']

Section: Datasets:
For multilingual experiments, we use mC4 [71], focusing on English (en), Hindi (hi), and German (de). An overview of these datasets is provided in Table 3. For GLaM-based experiments, we use the six-domain GLaM dataset [18]. Additional details are presented in Table 4.
Evaluation: Perplexity is measured on held-out validation data. Downstream evaluation follows the OLMES suite [26]. For multilingual downstream tasks, we use multilingual HellaSwag [13], covering 26 languages. For models trained on GLaM, we evaluate on downstream tasks ARC-Easy [12], CommonsenseQA [59], PIQA [5], and HellaSwag [76]. HellaSwag and ArcE tasks have 4 choices, CSQA has 5 choices, and PIQA has 2 choices.
Further details on our experimental setup and evaluation are in Appendix E.
this section cite: ['b70', 'b17', 'b25', 'b12', 'b11', 'b58', 'b4', 'b75']

Section: Main Observations on Multilingual Pretraining Experiments
Table 1 presents results for pretraining a 1B multilingual GPT-2 model [49] on English, Hindi, and German. Mix batching outperforms Round-Robin and Random strategies, supporting our analysis in Section 2 and justifying our focus on Mix as the foundation for PiKE. Additional results with different model sizes (270M and 1B) and language settings are reported in Table 15. Our main observation is that PiKE and its Balanced variant achieve the highest average downstream accuracy across all language settings and model sizes, demonstrating their effectiveness for multilingual pretraining.
We also observe that Balanced-PiKE promotes more fair learning across tasks. In particular, we pre-trained 1B models using Balanced-PiKE with different values of parameter τ ∈ {1, 3, 5}. As τ increases, task losses become more uniform, reflecting improved balanced learning. At τ = 5, perplexity becomes more balanced across languages, while τ = 3 offers the best trade-off-achieving both the lowest perplexity and highest downstream accuracy. These results highlight the importance of incorporating fairness/balanced learning into data mixing strategies during pretraining.
this section cite: ['b48']

Section: Main Observations on Pretraining Experiments with GLaM Datasets
Table 2 shows results for pretraining a 750M GPT-2 model on the GLaM dataset. Additional results with both 110M and 750M models across six domains are provided in Table 16. Across both 110M and 750M model sizes, PiKE consistently outperforms DoReMi, GLaM, and Mix in downstream accuracy. For the 750M model, PiKE improves average accuracy by 3.4% over DoReMi, 6.2% over GLaM, 7.1% over FAMO, and 4.8% over ADO. In the 110M setting, PiKE achieves 37.8% accuracy, exceeding DoReMi (36.0%), GLaM (35.3%), and FAMO (35.9%). Unlike DoReMi, which requires a separate proxy model, or GLaM, which depends on tuning with smaller models, PiKE delivers these gains with negligible additional overhead.
PiKE benefits from apriori downstream-tuned weights. We evaluate PiKE with two initializations:
(1) uniform weights b k = b/K and (2) GLaM-tuned weights. In both small and large GPT-2 configurations, PiKE benefits from utilizing already fine tuned weights as initialization, achieving 48.1% accuracy with GLaM-tuned weights vs. 47.6% with uniform initialization. This shows that PiKE can effectively leverage pre-existing fine-tuned weights while still outperforming other methods with uniform initialization.
Mixing datasets improves language model generalization. We compare models trained on individual domains to those trained on mixed-domain datasets. Table 16 shows that single-domain training underperforms compared to mixed-domain training, even with simple Mix sampling. This reinforces the importance of diverse data for pretraining and aligns with prior work [40,29].
Perplexity versus downstream performance. Table 16 reveals that validation perplexity does not always align with downstream performance. For instance, while Mix sampling yields lower perplexity in 750M models, PiKE achieves better downstream accuracy. This aligns with prior findings [23,60,39,67], suggesting that perplexity alone is not a reliable performance metric.
this section cite: ['b39', 'b28', 'b22', 'b59', 'b38', 'b66']

Section: Related Work
Data Curation and Selection. The effectiveness of language models heavily depends on the quality of the pre-training corpus. Consequently, significant efforts have been made to enhance pre-training data. These efforts include heuristic-based filtering [51,50,32,48,57] and deduplication [1,33,10,19].
Recently, [64] proposed an automated method for constructing large, diverse, and balanced datasets for self-supervised learning by applying hierarchical k-means clustering. [55] introduced techniques that leverage instruction-tuned models to assess and select high-quality training examples, along with density sampling to ensure diverse data coverage by modeling the data distribution. Additionally, [27] simulated training runs to model the non-additive effects of individual training examples, enabling the analysis of their influence on a model's predictions.
Multitask Learning Optimization Multitask learning (MTL) optimization is closely related to our framework. Among different MTL approaches, one idea is to modify gradient updates to mitigate gradient conflicts-situations where task gradients point in opposing directions, slowing down optimization [63,75]. The Multiple Gradient Descent Algorithm (MGDA) [16,56] updates the model by optimizing the worst improvement across all tasks, aiming for equal descent in task losses. Projected Conflicting Gradient Descent (PCGrad) [75] modifies task gradients by iteratively removing conflicting components in a randomized order, ensuring that updates do not interfere destructively across tasks. Conflict-Averse Gradient Descent (CAGRAD) [37] optimizes for the worst task improvement while ensuring a decrease in the average loss. NASHMTL [47] determines gradient directions by solving a bargaining game that maximizes the sum of log utility functions. While these methods improve performance, they introduce computational and memory overhead, making them impractical for large-scale models with numerous tasks [70]. Similar challenges exist in AdaTask [73], which improves multitask learning by balancing parameter updates using task-wise adaptive learning rates, mitigating task dominance, and enhancing overall performance. Unlike previous approaches that requires O(K) storage for task gradients (e.g. PCGrad) or optimizer states (e.g. AdaTask), FAMO [38] balances task loss reductions efficiently using O(1) space and time. However, these methods fail to exploit the non-conflicting interactions among tasks, focusing instead on resolving conflicts that seldom arise. This motivates our work that actively leverages lack of gradient conflicts to enhance training efficiency.
Data Mixture Problem Another closely related line of work to our paper is data mixture problem. Early works relied on manual heuristics for forming data compositions to construct datasets [18,71,7,22] for training LLMs. To create more principled and less labor-intensive methods, subsequent research has focused on optimizing these mixtures. One line of work uses offline optimization with proxy models, where a smaller, less expensive model is trained to find optimal data weights that are then used for the final large model training. Methods in this area include DoReMi [69], which optimizes domain weights to accelerate training convergence, and DOGE [20], which re-weights data by relying on the inner product between task gradients. A third category involves adaptive online data mixing, where the data distribution is adjusted dynamically during the training process. This includes methods that efficiently adjust sampling rates based on training signals ADO [30] and skill-based curriculum learning like Skill-it! [8], which dynamically samples data corresponding to specific skills to improve model proficiency. Our work PiKE is also in this category. In particular, PiKE dynamically leverages the near-orthogonality of task gradients, making it far more computational and memory-efficient than offline approaches like DoGE by only needing two scalars per task: per-task gradient norm and variance. PiKE also uniquely incorporates gradient variance or noise, a fundamental factor impacting the performance of the training, which was largely overlooked in recent data mixing literature such as DoGE. Overall, PiKE offers a reliable way to enhance final model performance.
this section cite: ['b50', 'b49', 'b31', 'b47', 'b56', 'b0', 'b32', 'b9', 'b18', 'b63', 'b54', 'b26', 'b62', 'b74', 'b15', 'b55', 'b74', 'b36', 'b46', 'b69', 'b72', 'b37', 'b17', 'b70', 'b21', 'b68', 'b19', 'b29', 'b7']

Section: Conclusion
In this work, we introduced PiKE, an adaptive data mixing algorithm for multitask learning that dynamically adjusts task sampling based on observed gradient interactions. Unlike prior methods that aim to resolve gradient conflicts, PiKE exploits the predominantly non-conflicting gradients seen in large-scale language model pretraining. We provided theoretical analysis and showed, through extensive experiments, that PiKE improves both convergence speed and downstream performance across multilingual and multi-domain settings. To promote balanced task learning, we extended PiKE with a fairness-aware objective, resulting in Balanced-PiKE, which reduces task-level performance gaps without sacrificing overall accuracy.
One limitation of PiKE is its lack of sensitivity to dataset size when assigning sampling weights. Future work could incorporate data abundance or downstream performance feedback into the sampling strategy. Additionally, extending PiKE to other domains beyond language modeling remains a promising direction for further research.
this section cite: []

Section: References
Ref_id:b0 Title: Semdedup: Data-efficient learning at web-scale through semantic deduplication Year: (2023)
Ref_id:b1 Title: Layer normalization Year: (2016)
Ref_id:b2 Title: Fair resource allocation in multi-task learning Year: (2024)
Ref_id:b3 Title: Curriculum learning Year: (2009)
Ref_id:b4 Title: Reasoning about physical commonsense in natural language Year: (2019)
Ref_id:b5 Title: JAX: composable transformations of Python+NumPy programs Year: (2018)
Ref_id:b6 Title: Language models are few-shot learners Year: (2020)
Ref_id:b7 Title: Skill-it! a datadriven skills framework for understanding and training language models Year: (2023)
Ref_id:b8 Title: Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks Year: (2018)
Ref_id:b9 Title: Scaling language modeling with pathways Year: (2022)
Ref_id:b10 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b11 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b12 Title: Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback Year: (2023)
Ref_id:b13 Title: Instance-aware semantic segmentation via multi-task network cascades Year: (2016)
Ref_id:b14 Title: Scaling vision transformers to 22 billion parameters Year: (2023)
Ref_id:b15 Title: Multiple-gradient descent algorithm (mgda) for multiobjective optimization Year: (2012)
Ref_id:b16 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b17 Title: Glam: Efficient scaling of language models with mixture-of-experts Year: (2022)
Ref_id:b18 Title: The llama 3 herd of models Year: (2024)
Ref_id:b19 Title: Domain reweighting with generalization estimation Year: (2023)
Ref_id:b20 Title:  Year: (2023)
Ref_id:b21 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b22 Title: Metadata conditioning accelerates language model pre-training Year: (2025)
Ref_id:b23 Title: Data mixing made efficient: A bivariate scaling law for language model pretraining Year: (2024)
Ref_id:b24 Title: Grain -feeding jax models Year: (2023)
Ref_id:b25 Title: Olmes: A standard for language model evaluations Year: (2024)
Ref_id:b26 Title: Simfluence: Modeling the influence of individual training examples by simulating training runs Year: (2023)
Ref_id:b27 Title: Flax: A neural network library and ecosystem for JAX Year: (2023)
Ref_id:b28 Title: An empirical analysis of compute-optimal large language model training Year: (2022)
Ref_id:b29 Title: Adaptive data optimization: Dynamic sample selection with scaling laws Year: (2024)
Ref_id:b30 Title: In-datacenter performance analysis of a tensor processing unit Year: (2017)
Ref_id:b31 Title: The bigscience roots corpus: A 1.6 tb composite multilingual dataset Year: (2022)
Ref_id:b32 Title: Deduplicating training data makes language models better Year: (2021)
Ref_id:b33 Title: Measuring the intrinsic dimension of objective landscapes Year: (2018)
Ref_id:b34 Title: Tilted empirical risk minimization Year: (2020)
Ref_id:b35 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b36 Title: Conflict-averse gradient descent for multi-task learning Year: (2021)
Ref_id:b37 Title: Famo: Fast adaptive multitask optimization Year: (2024)
Ref_id:b38 Title: Same pre-training loss, better downstream: Implicit bias matters for language models Year: (2023)
Ref_id:b39 Title: Regmix: Data mixture as regression for language model pre-training Year: (2024)
Ref_id:b40 Title: Representation learning using multitask deep neural networks for semantic classification and information retrieval Year: (2015)
Ref_id:b41 Title: Multi-task deep neural networks for natural language understanding Year: (2019)
Ref_id:b42 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b43 Title: Multi-task sequence to sequence learning Year: (2015)
Ref_id:b44 Title: Cross-stitch networks for multi-task learning Year: (2016)
Ref_id:b45 Title: Fair end-to-end window-based congestion control Year: (2000)
Ref_id:b46 Title: Multi-task learning as a bargaining game Year: (2022)
Ref_id:b47 Title: The refinedweb dataset for falcon llm: Outperforming curated corpora with web data only Year: (2023)
Ref_id:b48 Title: Language Models are Unsupervised Multitask Learners Year: (2019)
Ref_id:b49 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b50 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b51 Title: {ZeRO-Offload}: Democratizing {Billion-Scale} model training Year: (2021)
Ref_id:b52 Title: Convex analysis:(pms-28) Year: (2015)
Ref_id:b53 Title: Latent multi-task architecture learning Year: (2019)
Ref_id:b54 Title: How to train data-efficient llms Year: (2024)
Ref_id:b55 Title: Multi-task learning as multi-objective optimization Year: (2018)
Ref_id:b56 Title: An open corpus of three trillion tokens for language model pretraining research Year: (2024)
Ref_id:b57 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2021)
Ref_id:b58 Title: Commonsenseqa: A question answering challenge targeting commonsense knowledge Year: (2018)
Ref_id:b59 Title: Scale efficiently: Insights from pre-training and fine-tuning transformers Year: (2021)
Ref_id:b60 Title: a family of highly capable multimodal models Year: (2023)
Ref_id:b61 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b62 Title: Multi-task learning for dense prediction tasks: A survey Year: (2021)
Ref_id:b63 Title: Automatic data curation for self-supervised learning: A clustering-based approach Year: (2024)
Ref_id:b64 Title: Superglue: A stickier benchmark for general-purpose language understanding systems Year: (2019)
Ref_id:b65 Title: Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models Year: (2020)
Ref_id:b66 Title: Qurating: Selecting high-quality data for training language models Year: (2024)
Ref_id:b67 Title: Small-scale proxies for large-scale transformer training instabilities Year: (2023)
Ref_id:b68 Title: Doremi: Optimizing data mixtures speeds up language model pretraining Year: (2024)
Ref_id:b69 Title: Do current multi-task optimization methods in deep learning even help? Advances in neural information processing systems Year: (2022)
Ref_id:b70 Title: mt5: A massively multilingual pre-trained text-to-text transformer Year: (2020)
Ref_id:b71 Title: Towards a token-free future with pre-trained byte-to-byte models. corr, abs/2105.13626 Year: (2021)
Ref_id:b72 Title: Adatask: A task-aware adaptive learning rate approach to multi-task learning Year: (2023)
Ref_id:b73 Title: Data mixing laws: Optimizing data mixtures by predicting language modeling performance Year: (2024)
Ref_id:b74 Title: Gradient surgery for multi-task learning Year: (2020)
Ref_id:b75 Title: Hellaswag: Can a machine really finish your sentence? arXiv preprint Year: (2019)
Ref_id:b76 Title: Why transformers need adam: A hessian perspective Year: (2024)
Ref_id:b77 Title: C4 (hi), and C4 (de) datasets, GPT-2 large style, 1B params, 36 Layers default, 120K training steps Year: ()
Ref_id:b78 Title: of different models on four different Q/A tasks using 0-and 7-Shot settings. Bolding indicates the best model in the task, Metrics means the average across different tasks. ArcE CSQA HellaSwag PIQA ArcE CSQA HellaSwag PIQA Accuracy ↑ 0-shot ↑ 0-shot ↑ 0-shot ↑ 0-shot ↑ Accuracy(%) ↑ 7-shot ↑ 7-shot ↑ 7-shot ↑ 7-shot ↑ Six domains of GLaM dataset Year: ()
Ref_id:b79 Title: C4 (hi), and C4 (de) datasets, GPT-2 large style, 1B params, 36 Layers default, 120K training steps Mix Year: ()
Ref_id:b80 Title: 7-shot ↑ 7-shot ↑ 7-shot ↑ 7-shot ↑ Single domain of GLaM dataset, GPT-2 small style, 110M params, 12 layers default Wikipedia Year: ()
Ref_id:b81 Title: GPT-2 small style, 110M params, 12 layers default Mix 18 Year: ()
Ref_id:b82 Title: GPT-2 large style, 750M params, 36 layers default Wikipedia Year: ()
Ref_id:b83 Title: GPT-2 large style, 750M params, 36 layers default Mix 12 Year: ()
