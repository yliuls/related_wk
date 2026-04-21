Title: MGUP: A Momentum-Gradient Alignment Update Policy for Stochastic Optimization
Abstract: Efficient optimization is essential for training large language models. Although intra-layer selective updates have been explored, a general mechanism that enables fine-grained control while ensuring convergence guarantees is still lacking. To bridge this gap, we propose MGUP, a novel mechanism for selective updates. MGUP augments standard momentum-based optimizers by applying larger stepsizes to a selected fixed proportion of parameters in each iteration, while applying smaller, non-zero step-sizes to the rest. As a nearly plug-and-play module, MGUP seamlessly integrates with optimizers such as AdamW, Lion, and Muon. This yields powerful variants such as MGUP-AdamW, MGUP-Lion, and MGUP-Muon. Under standard assumptions, we provide theoretical convergence guarantees for MGUP-AdamW (without weight decay) in stochastic optimization. Extensive experiments across diverse tasks, including MAE pretraining, LLM pretraining, and downstream fine-tuning, demonstrate that our MGUP-enhanced optimizers achieve superior or more stable performance compared to their original base optimizers. We offer a principled, versatile, and theoretically grounded strategy for efficient intra-layer selective updates, accelerating and stabilizing the training of large-scale models. The code is publicly available at https://github.com/MaeChd/MGUP.

Section: Introduction
Recent studies reveal that the learning matrix during Large Language Model (LLM) training exhibits low-rank properties, suggesting that learning predominantly occurs in a low-dimensional space [1,2]. This observation has catalyzed the development of methods such as Galore [3] and LDAdam [4], which use gradient low-rank decomposition to achieve performance comparable to full-rank updates while reducing memory consumption. Although low-rank properties do not directly imply sparsity, the insight that optimization occurs in a low-dimensional space provides a crucial foundation for selective parameter updates. This principle is exemplified by SIFT [5], which achieves efficient adaptation through gradient-based sparse parameter updates, leveraging the low intrinsic dimensionality and sparse gradient characteristics inherent in LLMs. Building on this foundation, several innovative layer-wise selective update methods have emerged, including AutoFreeze [6], LOMO [7], LISA [8], and BAdam [9]. By strategically freezing certain layers while updating others, these methods achieve performance comparable to, or even surpassing, that of full-parameter updates.
While layer-wise selective updates show promise, finer-grained parameter selection remains underexplored. Although SIFT [5] investigates sparse intra-layer updates, a systematic methodology for identifying the most critical parameters within each layer is still lacking. This research gap motivates the development of novel intra-layer sparse update strategies.
Here, f : R d → R is a differentiable and possibly nonconvex function, ξ represents a random vector, such as a training data point, sampled from an unknown data distribution D.
In the context of solving problem (1), momentum-based methods are foundational in large-scale machine learning optimization, accumulating past gradient information to accelerate convergence and navigate complex loss landscapes. The standard momentum update, an exponentially weighted moving average (EWMA) of gradients, is given by m t = β 1 m t-1 + (1 -β 1 )g t , where β 1 is the decay factor, m t denotes the momentum vector, and g t denotes the gradient at the t-th iteration. This technique smooths gradient estimates, empirically and theoretically accelerating convergence and enhancing training stability [14,15,16,17].
While standard momentum is a robust baseline, research has sought to improve it, primarily through: (i) reducing stochastic gradient estimate variance and (ii) adapting learning based on momentum and gradient characteristics.
Variance reduction techniques, such as SPIDER [18], STORM [19], SUPER-ADAM [20], and MARS [21], operate by substituting the original stochastic gradient g t with a gradient estimator g ′ t that exhibits lower variance. This refined estimator is then used in the momentum update: m t = βm t-1 + (1 -β)g ′ t . While these methods theoretically accelerate convergence, they often necessitate additional computation or storage (e.g., storing past gradients). In contrast, MGUP adopts a distinct strategy, focusing on adaptively adjusting the update magnitude based on the characteristics of momentum and the current stochastic gradient, rather than directly altering the variance of the gradient estimation.
Another significant method involves adapting the optimization step based on the perceived reliability or characteristics of the momentum estimate. The intuition guiding this class of methods can be summarized as:
Increase step size for trustworthy momentum; Decrease step size for untrustworthy momentum.
This adaptation is often implemented by modulating the momentum vector, which can be represented generally as:
x t+1 = x t -η t m t ⊙ ϕ t , (2) where ϕ t is a scaling factor, often applied element-wise, determined by gradient statistics.
Early adaptive methods, like Adagrad [22], introduced per-parameter learning rates by accumulating squared gradients. The widely adopted Adam optimizer [23] builds on this by using EWMAs for both the first moment m t and the second moment v t of the gradients:
v t = β 2 v t-1 + (1 -β 2 )g 2 t .
The update step is then element-wise scaled by 1/ √ vt + ϵ, with vt being a bias-corrected v t and ϵ > 0 is a small constant. This enables Adam to adapt the learning rate per parameter based on historical gradient magnitudes. Subsequent research delved into various scaling factors, frequently investigating the interplay between the current gradient g t and the accumulated momentum m t . The AdaBelief optimizer [11] modifies Adam's second moment by using the squared difference between momentum and the current gradient, (m t -g t ) 2 , instead of the raw squared gradient g 2 t . The update rule for the second moment v t is as follows, with the initial condition v 0 = 0: 2 . The term (m t -g t ) 2 measures "belief" in the current gradient by its consistency with momentum. Significant deviation increases the corresponding element in v t , reducing that parameter's effective step size. This mechanism aims to merge Adam's rapid convergence with SGD's generalization. Denote m t,i and g t,i as the i-th elements of the momentum vector m t and gradient vector g t , respectively. If m t,i and g t,i have different signs, (m t,i -g t,i ) 2 is typically larger than g 2 t,i (for similar magnitudes), increasing v t,i and adaptively decreasing the step size. Meanwhile, a more direct approach to leveraging the sign consistency between momentum and gradient is taken by the Cautious Optimizers [10]. It employs an element-wise mask φ t to selectively apply momentum updates: φ t = α • I(m t ⊙ g t > 0), x t+1 = x t -η t m t ⊙ φ t .
v t = β 2 v t-1 + (1 -β 2 )(m t -g t ) 2 = (1 -β 2 ) t i=1 β t-i 2 (m i -g i )
Here, I(•) is the indicator function, which equals 1 when its argument is positive and 0 otherwise. If the signs m t,i and g t,i are align, the momentum component m t,i is scaled by α > 1; otherwise, the update for that component is nullified. This "Cautious Updating" strategy aims to prevent updates from potentially conflicting gradient information.
However, these advanced adaptive methods have notable limitations. AdaBelief's reliance on second-moment estimation restricts its applicability primarily to Adam-style optimizers, thereby rendering it incompatible with newer methods like Lion [12] and Muon [13] that perform well without this component. The Cautious Optimizer, while more broadly applicable, lacks formal stochastic convergence guarantees. As analyzed in Section 4, its binary masking mechanism can aggressively discard gradient information. This behavior may slow convergence, especially in scenarios where the signs of the momentum and gradient align infrequently. Furthermore, Cautious Adam exhibits nonconvergent behavior, highlighting a critical flaw in its design (see the counterexample in Appendix A).
f(x) x opt x t t m   I 1 1 t t m     I 1 t x  t x 1 t x  f(x) x opt x 1 t x  t x 1 t x  2 t x 
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b4', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b10', 'b9', 'b11', 'b12']

Section: Increase step size for trustworthy momentum
Decrease step size for untrustworthy momentum
2 2 t t m     D 1 1 t t m     D 1 { | ( ; ) 0} t t t t i i i f x m           I I I 1 1 1 2 { | ( ; ) 0} t t t t i i i f x m              D D D Figure 1:
The key idea of MGUP involves adaptively adjusting the learning rate by leveraging the element-wise product of the stochastic gradient and momentum.
this section cite: []

Section: The Proposed Method
This section introduces the MGUP (Momentum-Gradient alignment Update Policy) mechanism for solving Problem (1). Our motivation is to address limitations observed in methods such as AdaBelief and Cautious Optimizers. Figure 1  ∈ I topK , its effective step size is set to γ • η t,i . Here, α > 1 represents the amplification factor, while γ denotes the decay factor. In practice, α and γ can be set to 1/τ and τ , respectively, where τ ∈ (0, 1).
An adjustment based on sign judgment is a concept from prior work (e.g., Cautious Optimizers [10]). Similarly, we can define the Cautious-MGUP mechanism as:
ϕ t,i = 1/τ if m t,i • g t,i > 0 τ if m t,i • g t,i ≤ 0.(3)
In contrast, MGUP offers a more flexible and robust adjustment strategy by introducing a top-K selection and sorting mechanism based on the element-wise product between the momentum and gradient.
We clarify the selection criterion associated with the momentum. While intuitively related to momentum-gradient consistency, MGUP-AdamW's implementation in Algorithm 1 can use the product of the final update vector u t (typically m t /( √ v t + ϵ)) and the gradient g t , not just momentum m t and gradient g t . This is because, in specific contexts, especially when training large language models, the difference between the selections based on u t,i • g t,i and m t,i • g t,i may be negligible. Research [24,25,26,27] suggests that within certain model layers, the second moment v t 's adaptive scaling might be relatively uniform. This implies an approximation where
(m 1 / √ v 1 , . . . , m d / √ v d ) ≈ (
m 1 /c, . . . , m d /c) for some constant c. Consequently, the sign and relative magnitude ordering from u t,i • g t,i would closely mirror that from m t,i • g t,i . Thus, MGUP-AdamW can be intuitively seen as a selection strategy guided by momentum-gradient alignment. Remark 3.1. For optimizers with simpler update structures, such as Lion, Muon, or standard SGD+Momentum, m t,i • g t,i can be directly used as the alignment score. Remark 3.2. We explain why MGUP is expected to accelerate convergence. (i) MGUP's mechanism uses a greedy strategy (due to sorting). Greedy strategies play a key role in accelerating heuristic algorithms. When the stochastic gradient and the update share the same sign, their positive product favors larger step sizes; when their signs differ, the negative product favors smaller ones. Large steps drive acceleration, while small nonzero steps ensure convergence. The necessity of small yet nonzero step sizes is analyzed in Section 4, with a counterexample in Appendix A illustrating the failure of zero step sizes. (ii) MGUP increases the average update magnitude. In MGUP, a fraction τ of parameters update with an increased learning rate of (1/τ )lr, while the remaining 1 -τ use a reduced rate of τ lr. The resulting average learning rate is τ
• lr • (1/τ ) + (1 -τ ) • lr • τ = lr • (1 + τ -τ 2 ) > lr.
When individual step sizes are roughly uniform-for example, when Adam's early updates approximate sign(g t )-the overall update magnitude of MGUP increases by a factor of (1 + τ -τ 2 ) compared to Adam. This provides an intuitive explanation for its acceleration effect. We recommend τ = 1 2 as the default, since arg max τ ∈(0,1) (1 + τ -τ 2 ) = 1  2 .
this section cite: ['b9', 'b23', 'b24', 'b25', 'b26']

Section: Algorithm 1 MGUP-AdamW
Input: Learning rate η > 0, initial solution x 0 ∈ R d , momentum factors β 1 , β 2 ∈ [0, 1), weight decay coefficient λ, stability term ϵ > 0, ratio τ ∈ (0, 1).
Set m 0 = 0, v 0 = 0. for t = 1 to T do Compute the stochastic gradient g t = ∇f (x t ; ξ t ) m t = β 1 m t-1 + (1 -β 1 )g t v t = β 2 v t-1 + (1 -β 2 )(g t ⊙ g t ) u t = mt √ v t +ϵ , η t = η √ 1-β t 2 1-β t 1 ϕ t = MGUP(u t ⊙ g t ) x t = (1 -η t λ)x t x t+1 = x t -η t ϕ t ⊙ u t end for Algorithm 2 MGUP Input: Alignment score vector s t = u t ⊙ g t ∈ R d , ratio τ ∈ (0, 1).
(S1) Let I topK be the index set of the largest K elements of s t ∈ R d with K = ⌊τ • d⌋.
(S2) Set ϕ t,i = 1/τ, i ∈ I topK ; τ, else. return ϕ t Remark 3.3. The MGUP method can be easily plugged into existing momentum-based optimization algorithms in a plug-and-play manner. Examples include Lion [12] and Muon [13] (see Appendix H for the pseudocode of MGUP-Lion and MGUP-Muon).
this section cite: ['b11', 'b12']

Section: Convergence Analysis
In this section, we rigorously establish both the expected convergence and high-probability convergence guarantees for Algorithm 1 in the stochastic setting.
For the convergence analysis of Algorithm 1, we make the following assumptions: Assumptions 4.1 and 4.2 are standard in the analysis of nonconvex optimization [28,29,20,30,31].
We first show that, under the following standard assumption, the MGUP-AdamW(without weight decay) can achieve the expected convergence rate of O(log(T )/ √ T ).
Assumption 4.3. The stochastic gradient is unbiased with bounded variance. That is, there exists σ > 0 such that for all x ∈ R d , E[∇f (x; ξ)] = ∇f (x), and E[∥∇f (x; ξ) -∇f (x)∥ 2 2 ] ≤ σ 2 . Additionally, we assume that f (x; ξ) is M -Lipschitz, i.e., ∥∇f (x; ξ)∥ ≤ M for all x and ξ. Assumption 4.3 is very common in the literature [31,32,33]. Theorem 4.1 states our general non-convex convergence result.
Theorem 4.1. Let β 1,t = 1 -t -1/2 , 0 < β 2 ≤ 1, and η t = ηt -1/2 /ρ. We define the following:
ε 1 = σ 2 L , ε 2 = 1 ρ u 2 min 2u 3 max -5L ρu 2 min
, and ε 3 = 1 2L . Here, u min = ϵ η and u max = M ηγ for some constant learning rate η and any ϵ > 0. Let ρ > 10Lu 3 max u 4 min so that ε 2 > 0, and define ε min = min(ε 1 , ε 2 , ε 3 ). Under Assumptions 4.1, 4.3, and 4.2, for Algorithm 1 (without weight decay), it holds that:
1 T T t=1 E∥∇f (x t+1 )∥ 2 2 ≤ Ĝ.
where ill-defined, as ρ would need to be infinitely large, which is unattainable and adversely affects convergence. Remark 4.2. While our analysis assumes global Lipschitz continuity, the algorithm can be implemented using M T = max j∈[T ] ∥∇f (x j ; ξ)∥ instead of a global bound M. This approach only requires bounded gradients along the optimization trajectory, typically yields tighter bounds, and remains fully compatible with our theoretical guarantees. Furthermore, setting
Ĝ = 3L 2 η 2 +3ρ 2 ϵ 2 ρ 2 ϵ 2 T f (x1)-f (x * )+2σ 2 L -1 log(T +1) εmin √ T -2( √ T -1) .
η t = η • t -1/2 /ρ instead of η √ 1-β t 2 1-β t 1 • t -1/2 /ρ is justified since √ 1-β t 2 1-β t 1
is bounded and can be absorbed into the constant η without loss of generality. See Appendix C for details.
Next, under the assumption of coordinate-wise random noise, we show that the MGUP-AdamW(without weight decay) also achieves a rate of O(poly(log(T ))/ √ T ) with high probability. Assumption 4.4. The stochastic gradient is unbiased with coordinate-wise bounded variance. That is, there exists σ i > 0 such that for all x ∈ R d , E[∇f (x; ξ)] = ∇f (x), and E[(∇f (x; ξ) i -∇f (x) i ) 2 ] ≤ σ 2 i for all i.
Assumption 4.4 is commonly used in the literature [34,35,36,37,38,32]. Note that the coordinatewise noise bound in Assumption 4.4 is stronger than the standard bound E∥∇f (x; ξ) -∇f (x)∥ 2 2 ≤ σ 2 , as the latter can be readily derived from the former. This relaxed choice is made to facilitate the application of probabilistic inequalities, thereby achieving improved convergence properties.
Theorem 4.2. Let 0 ≤ β 1 < β 2 < 1, β 2 = 1 -1/T , η = C 0 √ 1 -β 2 , ω = ( 1 + 1/β 2 +
1) max{1, γ, 1/γ}, γ ∈ ( 2 β , 1), and
β 3 = max 1-β2 √ 1-β2 , 2-γ 2 (1+β2) γ √ 1-β2 , |β2-γ 2 |+1-γ 2 γ √ 1-β2
for some constants C 0 > 0, β > 2. Under Assumptions 4.1,4.2, and 4.4, for Algorithm 1(without weight decay), then for any given δ ∈ (0, 1/2), it holds that with probability at least 1 -2δ,
1 T T t=1 ∥∇f (x t )∥ 2 2 ≤ Õ(T -1/2
). Remark 4.3. Setting γ > 0 is crucial for ensuring the stable convergence of the algorithm. The convergence proof relies on surrogate stepsizes (defined in equations ( 10) and ( 12)) to manage the complex interplay between stochastic gradients and adaptive stepsizes. The theoretical framework for employing these surrogate stepsizes within the proof is informed by the methodologies presented in [35,39,40], as follows:
y t+1 = y t -η t ϕ t ⊙ gt bt + β1 1-β1 ηtbt-1⊙ϕt ηt-1bt⊙ϕt-1 -1 d ⊙ (x t -x t-1 ).
where the precise definitions of all terms are provided in Appendix E. Notably, if γ were set to 0, the ratio ϕt,i ϕt-1,i could approach infinity for some component i when ϕ t,i = α and ϕ t-1,i = γ = 0. Such occurrences might prevent parameter updates in certain iterations, thereby hindering convergence. Consequently, γ is set to a positive value instead of 0. For a more detailed discussion, please refer to Appendix E.
this section cite: ['b27', 'b28', 'b19', 'b29', 'b30', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b31', 'b34', 'b38', 'b39']

Section: Experiments
In this section, we evaluate the performance of the proposed MGUP optimizers on both pretraining and supervised fine-tuning (SFT) tasks. All experiments are conducted using two NVIDIA V100 (32GB) GPUs and four NVIDIA RTX 4090 (24GB) GPUs. Detailed experimental settings are provided in Appendix G.
▶ Datasets. We use the image dataset CIFAR-10, the text dataset Wikitext-103, and the language model fine-tuning benchmarks GLUE and GSM-8K.
▶ Compared Methods. We compare MGUP-AdamW, MGUP-Lion, MGUP-Muon with (i) AdamW [41], (ii) Cautious Optimizers(C-AdamW, C-Lion, C-Muon) [10], (iii) Lion [12], (iv) Muon [13,42], as well as other state-of-the-art memory-efficient optimization methods such as (v) GaLore [3], (vi) LDAdam [4], (vii) Adam-mini [27] and (viii) Adam-8Bit [43].
Unless stated otherwise, the default setting for MGUP-enhanced optimizers is τ = γ = 1/α = 1 2 .
this section cite: ['b40', 'b9', 'b11', 'b12', 'b41', 'b2', 'b3', 'b26', 'b42']

Section: Pretraining

this section cite: []

Section: ▶ Image MAE Pretraing
We pre-train a straightforward ViT model [44] with the Masked Autoencoder (MAE) framework [45] on the CIFAR-10 dataset. For this experiment, we set the learning rate to 1.5 × 10 -4 , the MAE mask rate to 75%, and train for 200 epochs. We compare MGUP-AdamW with the standard AdamW and C-AdamW optimizers by evaluating their training and validation losses. The results, presented in Figure 2, show that MGUP-AdamW consistently achieves lower training and validation loss throughout the training process. In contrast, the performance of C-AdamW gradually falls behind that of AdamW.
this section cite: ['b43', 'b44']

Section: ▶ Language Modeling
We employ a straightforward LLaMA2-71M model [46] and a Qwen2.5-150M model [47], both of which are pretrained on the WikiText-103 dataset.
this section cite: ['b45', 'b46']

Section: LLaMA2-71M on WikiText-103.
To assess optimizer performance on a smaller language model, we train LLaMA2-71M on WikiText-103, evaluating validation loss. We compare AdamW, Lion, and Muon variants using a learning rate of 3e-4, a batch size of 480, and 2000 training steps. As shown in Figure 3a, the results highlight several key differences. Among the Adam-type optimizers, MGUP-AdamW achieves a 1.6x speedup over standard AdamW and exhibits superior generalization compared to C-AdamW. For the Lion-type optimizers, MGUP-Lion demonstrates a 2.5x speedup over standard Lion; unlike the unstable C-Lion which shows early loss spikes, MGUP-Lion maintains training stability. With the Muon-type optimizers, MGUP-Muon yields a ∼1.2x speedup relative to Muon and delivers better generalization than C-Muon.
While τ serves as the primary hyperparameter in our approach, it is essential to examine how variations in γ influence the performance of MGUP-AdamW. We conduct experiments with τ ∈ {0.3, 0.5, 0.7} and γ ∈ {0, 0.1, 0.5, 0.9} to evaluate this relationship across different hyperparameter configurations.
The comparative results are presented in Figure 3b. The analysis indicates the following: (i) with γ fixed, increasing τ beyond a certain threshold degraded performance; (ii) with τ fixed, a larger γ generally improved performance. The findings in (ii) precisely corroborate the discussion on the setting of γ in Section 4.
this section cite: []

Section: Qwen2.5-150M on WikiText-103.
We also evaluate optimizers on a larger Qwen2.5-150M model using WikiText-103 (Figure 4). For these experiments, we use a learning rate of 1e-3, a batch size of 160, and 1500 training steps. For Adam-type optimizers, MGUP-AdamW demonstrates a higher speedup than standard AdamW and better generalization than C-AdamW. For Muon-type optimizers, MGUP-Muon achieves a 1.1x speedup over standard Muon and superior generalization compared to C-Muon. (a) Qwen2.5 training curve (b) Qwen2.5 validation curve Figure 4: Qwen2.5-150M training and validation curves on WikiText-103 5.2 Fine-Tuning (a) AdamW-type (b) Lion-type (c) Muon-type Figure 5: Adamw-type, Lion-type,Muon-type optimizers average performance across GLUE tasks
We conduct comprehensive experiments on downstream tasks, with particular emphasis on supervised fine-tuning (SFT) scenarios. Our evaluation encompassed two representative tasks: fine-tuning the RoBERTa-base model [48] on the GLUE benchmark and the LLaMA2-7B model [46] on the GSM-8K.
this section cite: ['b47', 'b45']

Section: ▶ GLUE Benchmark Evaluation.
To evaluate performance and generalization on diverse Natural Language Understanding (NLU) tasks, we experiment on the GLUE benchmark, which comprises tasks varying in dataset size and complexity. We perform a learning rate search within the range of 1e-5 to 5e-5 for most optimizers, and within the range of 1e-6 to 5e-6 for Lion-type optimizers.
The best performance for each task is reported in Table 1. On most tasks, MGUP-AdamW and MGUP-Muon achieve state-of-the-art results. Notably, MGUP-AdamW reaches an average optimal performance of 85.15 across all GLUE tasks. ▶ GSM-8K Fine-tuning. We further evaluate MGUP-AdamW by fine-tuning LLaMA2-7B on the challenging GSM-8K dataset, a critical indicator of fine-tuning effectiveness due to typically low zero-shot accuracy [49]. We conduct a learning rate grid search (from 1e-5 to 5e-5), consistent with [4]. As shown in Table 2, MGUP-AdamW achieves lower training loss per epoch and the highest validation accuracy 34.96%, outperforming baseline optimizers.
this section cite: ['b48', 'b3']

Section: Conclusion
We introduce MGUP, a novel intra-layer parameter selection mechanism based on momentumgradient alignment, and integrated it into AdamW, Lion, and Muon yields MGUP-AdamW, MGUP-Lion, and MGUP-Muon. Empirically, MGUP Optimizers demonstrate competitive convergence speeds and superior generalization over their base versions across diverse tasks, including large language model training. Theoretically, we establish stochastic convergence guarantees for MGUP-AdamW(without weight decay) under standard non-convex assumptions, achieving a rate near the known optimum. Limitations include the pre-selection of τ , inviting future work on adaptive methods. Our theoretical analysis also primarily covers MGUP-AdamW (without weight decay). Thus, while empirically effective with optimizers like Lion and Muon, MGUP's theoretical properties (e.g., the necessity of γ > 0) in these diverse frameworks require further study.
f i (x) = nx, x ≥ -1 n 2 (x + 2) 2 -3n 2 , x < -1 for i = 0. f i (x) = -x, x ≥ -1 -1 2 (x + 2) 2 + 3 2 , x < -1 for i > 0.
The full objective is f (x) = n-1 i=0 f i (x), which simplifies to:
f (x) = x, x ≥ -1 1 2 (x + 2) 2 -3 2 , x < -1.
We analyze the behavior of C-Adam and MGUP-Adam on a counterexample with its global minimum at x * = -2, starting from an initial point x 0 = -0.5. In this environment, the optimizer encounters frequent, small negative gradients g t = -1 and rare, large positive gradients g t = n.
this section cite: []

Section: References
Ref_id:b0 Title: Gradient descent happens in a tiny subspace Year: (2018)
Ref_id:b1 Title: How many degrees of freedom do we need to train deep networks: a loss landscape perspective Year: ()
Ref_id:b2 Title: Galore: Memory-efficient LLM training by gradient low-rank projection Year: ()
Ref_id:b3 Title: Ldadam: Adaptive optimization from low-dimensional gradient statistics Year: ()
Ref_id:b4 Title: Sparse is enough in fine-tuning pre-trained large language models Year: ()
Ref_id:b5 Title: Autofreeze: Automatically freezing model blocks to accelerate fine-tuning Year: (2021)
Ref_id:b6 Title: Full parameter fine-tuning for large language models with limited resources Year: (2023)
Ref_id:b7 Title: LISA: layerwise importance sampling for memory-efficient large language model fine-tuning Year: ()
Ref_id:b8 Title: Badam: A memory efficient full parameter optimization method for large language models Year: ()
Ref_id:b9 Title: Cautious optimizers: Improving training with one line of code Year: (2024)
Ref_id:b10 Title: Adabelief optimizer: Adapting stepsizes by the belief in observed gradients Year: (2020)
Ref_id:b11 Title: Symbolic discovery of optimization algorithms Year: ()
Ref_id:b12 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b13 Title: On the importance of initialization and momentum in deep learning Year: (2013)
Ref_id:b14 Title: Demon: Improved neural network training with momentum decay Year: (2019)
Ref_id:b15 Title: Towards understanding how momentum improves generalization in deep learning Year: ()
Ref_id:b16 Title: When and why momentum accelerates sgd: An empirical study Year: (2023)
Ref_id:b17 Title: SPIDER: near-optimal nonconvex optimization via stochastic path-integrated differential estimator Year: (2018)
Ref_id:b18 Title: Momentum-based variance reduction in non-convex SGD Year: (2019)
Ref_id:b19 Title: SUPER-ADAM: faster and universal framework of adaptive gradients Year: (2021)
Ref_id:b20 Title: Mars: Unleashing the power of variance reduction for training large models Year: (2024)
Ref_id:b21 Title: Adaptive subgradient methods for online learning and stochastic optimization Year: (2011)
Ref_id:b22 Title: Adam: A method for stochastic optimization Year: (2015)
Ref_id:b23 Title: Deconstructing what makes a good optimizer for autoregressive language models Year: ()
Ref_id:b24 Title: Adashift: Decorrelation and convergence of adaptive learning rate methods Year: (2019)
Ref_id:b25 Title: Adasgd: Bridging the gap between sgd and adam Year: (2020)
Ref_id:b26 Title: Adam-mini: Use fewer learning rates to gain more Year: ()
Ref_id:b27 Title: On the convergence of adaptive gradient methods for nonconvex optimization Year: (2018)
Ref_id:b28 Title: On the convergence of A class of adam-type algorithms for non-convex optimization Year: (2019)
Ref_id:b29 Title: A novel convergence analysis for algorithms of the adam family Year: (2021)
Ref_id:b30 Title: Convergence of adam under relaxed assumptions Year: ()
Ref_id:b31 Title: Closing the gap between the upper bound and lower bound of adam's iteration complexity Year: ()
Ref_id:b32 Title: Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models Year: (2024)
Ref_id:b33 Title: Convergence guarantees for rmsprop and adam in generalized-smooth non-convex optimization with affine noise variance Year: (2025)
Ref_id:b34 Title: On convergence of adam for stochastic optimization under relaxed assumptions Year: ()
Ref_id:b35 Title: A high probability analysis of adaptive sgd with momentum Year: (2020)
Ref_id:b36 Title: A simple convergence proof of adam and adagrad Year: (2022)
Ref_id:b37 Title: High probability convergence of adam under unbounded gradients and affine variance noise Year: (2023)
Ref_id:b38 Title: Global convergence of the heavy-ball method for convex optimization Year: (2014)
Ref_id:b39 Title: A unified analysis of stochastic momentum methods for deep learning Year: (2018-07-13)
Ref_id:b40 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b41 Title: Muon is scalable for llm training Year: (2025)
Ref_id:b42 Title: 8-bit optimizers via block-wise quantization Year: ()
Ref_id:b43 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: ()
Ref_id:b44 Title: Masked autoencoders are scalable vision learners Year: (2021)
Ref_id:b45 Title:  Year: (2023)
Ref_id:b46 Title:  Year: (2024)
Ref_id:b47 Title: Roberta: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b48 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b49 Title: Adam can converge without any modification on update rules Year: ()
Ref_id:b50 Title: Adafactor: Adaptive learning rates with sublinear memory cost Year: (2018)
Ref_id:b51 Title: Q-galore: Quantized galore with INT4 projection and layer-adaptive lowrank gradients Year: (2025-03-27)
Ref_id:b52 Title: Greedy layer-wise training of deep networks Year: (2006)
Ref_id:b53 Title: A fast learning algorithm for deep belief nets Year: (2006)
Ref_id:b54 Title: Adalomo: Low-memory optimization with adaptive learning rate Year: (2024)
Ref_id:b55 Title: A method for solving the convex programming problem with convergence rate Year: (1983)
Ref_id:b56 Title: Lecture 6a overview of minibatch gradient descent Year: (2012)
Ref_id:b57 Title: On the convergence of adam and beyond Year: (2018)
Ref_id:b58 Title: Incorporating nesterov momentum into adam Year: (2016)
Ref_id:b59 Title: Adaptive gradient methods with dynamic bound of learning rate Year: (2019)
Ref_id:b60 Title: On the variance of the adaptive learning rate and beyond Year: (2020)
Ref_id:b61 Title: Adaptive inertia: Disentangling the effects of adaptive learning rate and momentum Year: (2020)
Ref_id:b62 Title: Sophia: A scalable stochastic second-order optimizer for language model pre-training Year: ()
Ref_id:b63 Title: Shampoo: Preconditioned stochastic tensor optimization Year: (2018)
Ref_id:b64 Title: A sufficient condition for convergences of adam and rmsprop Year: (2019)
Ref_id:b65 Title: Convergence of adam for non-convex objectives: relaxed hyperparameters and non-ergodic case Year: (2025)
