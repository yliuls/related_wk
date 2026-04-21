Title: COUPLING EXPERTS AND ROUTERS IN MIXTURE-OF-EXPERTS VIA AN AUXILIARY LOSS
Abstract: Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router's decisions align well with the experts' capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router's decisions with expert capabilities. Our approach treats each expert's router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain intermediate activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert's capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on n 2 activations, where n is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs.

Section: INTRODUCTION
Mixture-of-Experts (MoE, Shazeer et al., 2017;Fedus et al., 2022;Lepikhin et al., 2021;Zoph et al., 2022) is a core architecture in modern large language models (LLMs). In MoE models, the feedforward layer is split into multiple small, specialized "experts." A linear classifier, known as the "router," selects which experts process each input token. By activating a few experts per token, MoE balances efficiency with scaled parameter counts, enabling the training of trillion-parameter models.
Ideally, a router should possess an accurate representation of each expert's capabilities to enable effective token routing. However, traditional MoEs offer no explicit constraints to guarantee this. Without direct access to expert parameters (and therefore their true capabilities), routers resort to trial-and-error learning of routing strategies, often resulting in misrouted tokens whose gradients interfere with expert specialization. While some methods (Lv et al., 2025;Pham et al., 2024) incorporated all experts' activations for routing guidance, they incur substantial computational and memory costs due to denser activation. A lightweight and effective solution to better couple routing decisions with true expert capabilities remains an open challenge.
We propose expert-router coupling loss (ERC loss), a novel auxiliary loss for MoE models that tightly couples routers and experts with negligible overhead. The loss is based on interpreting the router parameter matrix R ∈ R n×d as cluster centers, where each row R[i] serves as the center for the token set X i routed to expert i. The ERC loss comprises three key steps:
(1) Each R[i] is augmented with bounded random noise δ i to obtain R[i], serving as a proxy for tokens in X i . Here, δ i is bounded by half the minimum distance between adjacent cluster centers, Figure 1: Three steps for computing the expert-router coupling loss.
ensuring that the noise simulates input variations within X i while preventing the crossing of cluster boundaries.
(2) Inspired by prior works (Geva et al., 2021;Liu et al., 2023;Lv et al., 2025), the intermediate activation norm serves as an indicator of how well its capabilities align with the token. We measure the intermediate activation norms of all experts that take R[i] as input. This step produces a matrix M ∈ R n×n , with M [i, j] being the activation norm from expert j given input R[i].
(3) For all i ̸ = j, the ERC loss imposes a penalty wherever the off-diagonal elements M [i, j] or M [j, i] exceed αM [i, i], where α is a scalar hyperparameter:
L ERC = 1 n 2 n i=1 n j̸ =i
(max (M [i, j] -αM [i, i], 0) + max (M [j, i] -αM [i, i], 0)) .
Minimizing it tightly couples experts and routers through two effects:
• Expert specialization: The proxy token R[i] elicits the strongest activation from expert i versus all other experts. This indicates that expert i is optimized to best match the features of its assigned token cluster X i .
• Precise token routing: Expert i is most activated by its designated vector R[i] than to any other R[j] for j ̸ = i. This demonstrates that R[i] aligns well with the capabilities of expert i, ensuring that the router assigns to this expert the tokens that need it most.
We conducted large-scale pre-training experiments on models from 3B to 15B parameters, using a total of several trillion tokens. The ERC loss not only significantly enhances model performance and narrows the performance gap with a competitive yet more computationally expensive MoE variant (Lv et al., 2025) but also retains the efficiency of vanilla MoEs.
Furthermore, building on the first effect, we establish that the ERC loss serves as a powerful tool for studying expert specialization. This property arises from two key features of the ERC loss: (1) the specialization level is explicitly controlled by α, and (2) the bound of noise δ i provides a quantitative measure for this level. Through this lens, we reveal a trade-off between specialization and model performance. Our findings challenge some beliefs about expert specialization that were derived from small-scale experiments. These quantitative and qualitative analysis methods offer new pathways to advance the understanding of MoE models.
this section cite: ['b7', 'b19', 'b7', 'b25', 'b29', 'b8', 'b22', 'b25', 'b25']

Section: BACKGROUND
Mixture-of-Experts Our description follows the prevailing SwiGLU structure used by advanced LLMs (Qwen, 2024;DeepSeek-AI, 2025;OpenAI, 2025). An MoE layer consists of n experts, where each expert i is parameterized by three matrices:
W i g ∈ R d×D , W i p ∈ R d×D , and W i o ∈
R D×d . The layer also includes a router with the weight matrix R ∈ R n×d , which takes a token x ∈ R d as input and outputs an expert weightfoot_0 vector:
w = softmax(xR ⊤ ) ∈ R n .
Typically, the top-K experts with the highest expert weights are selected to process the token. The processing of x by expert i is given by:
E i (x) = SiLU(xW i g ) ⊙ (xW i p ) W i o ,
where ⊙ denotes element-wise multiplication. The final output of the entire MoE layer is the weighted sum of the outputs of the selected experts:
K k w[k]E k (x
), where k ∈ Top-K(w).
Expert-router coupling via denser activation Autonomy-of-Experts (AoE; Lv et al., 2025) encodes the routing function into expert parameters. AoE factorizes W g into two r-rank matrices W i down ∈ R d×r and W i up ∈ R r×D . Each expert processes a token up to the point after the W i down projection. The expert weight vector is computed using the activation norm at this stage:
w = softmax {∥xW i down ∥ for i = 1, . . . , n} . (a) Mixture-of-Experts (b) Autonomy-of-Experts Router x w 𝑬 𝟏 𝑬 𝟑 𝑬 𝟐 x 𝑬 𝟑 𝑬 𝟏 𝑬 𝟐 w Compute 𝑳 𝟐 Norm & Softmax Used Param. Unused Param.
this section cite: ['b31', 'b6', 'b25']

Section: Figure 2: The overview of MoE and AoE models.
The top-K experts exhibiting the highest activation norms are selected to continue their forward computation, and the others are terminated early. This norm-based selection is justified by the fact that the activation norm of MLPs represents how well their capabilities match their inputs (Geva et al., 2021;Liu et al., 2023). The computational overhead of AoE scales with the number of tokens during both training and inference. Moreover, this inefficiency worsens as the number of experts n increases or the selection count K decreases. Wu et al. (2025) found that using only a small, representative subset of neurons per expert is sufficient for "autonomous expert selection," reducing but not eliminating AoE's token-dependent cost.
Pham et al. (2024) use experts' final output norms to supervise router logits. There is no inference overhead but the model is fully dense-activated during training, contradicting the core sparsity principle of MoE. Therefore, we include it only for background discussion, not as a baseline.
this section cite: ['b8', 'b22', 'b45']

Section: METHOD
After analyzing the strengths and limitations of prior work, we distill three design principles to ensure a lightweight, effective, and practically applicable enhancement for expert-router coupling in MoE-LLMs:
(1) Routers must be retained in MoE architectures to preserve routing efficiency.
(2) An auxiliary loss that enables interaction between experts and routers can strengthen their coupling.
(3) The loss must have complexity independent of the number of input tokens and must not introduce activation density beyond that of a vanilla MoE.
Below, we introduce expert-router coupling loss, which fulfills all these principles.
this section cite: []

Section: EXPERT-ROUTER COUPLING LOSS
The expert-router coupling (ERC) loss is motivated by a clustering-based interpretation of MoE routing: The routing mechanism in traditional MoE models can be interpreted as a clustering process, where router parameters R ∈ R n×d are viewed as n cluster centers. For any input token x ∈ R d , the router computes an n-dimensional logit vector representing the weight assigned to each expert. Specifically, the weight for expert i is derived from the inner product between x and the cluster center R[i]. When x belongs to the cluster centered at R[i], this inner product is maximizedfoot_1 , making expert i the top choice.
A key advantage of this clustering view is that it enables probing an expert's responsiveness to a set of tokens without feeding every token to all experts, unlike prior methods (See §2). Instead, we leverage each cluster center R[i] as a proxy for tokens routed to expert i (denoted as X i ), enabling us to derive intermediate activations and evaluate how well the expert aligns with a proxy token.
Our ERC loss is computed in three key steps:
(1) For each cluster center R[i], we create a perturbed proxy token R
[i] = R[i] ⊙ δ i . δ i ∈ R d
is bounded multiplicative random noise, which we elaborate in §3.2. This noise ensures the proxy generalizes to tokens in X i . Notably, the perturbed R is used only for loss computation; routing still uses the clean R to compute router logits, as in standard MoEs.
(2) Each proxy token is processed by the W g parameter of all n experts, yielding a total of n 2 intermediate activations. The L 2 norm of each activation is computed to form a matrix M ∈ R n×n , where M [i, j] corresponds to the norm from expert j given input R[i]:
M [i, j] = R[i] • W j g .
(3) To enforce expert-router coupling, for all i and j ̸ = i, the ERC loss imposes two constraints, where a scalar α ∈ [0, 1] determines their strength:
M [i, j] < αM [i, i],(1)
M [j, i] < αM [i, i].(2)
Constraint 1 ensures the proxy token R[i] activates its corresponding expert i more than any other expert j. Since tokens similar to R[i] are routed to expert i, and given their similarity to R[i], they also elicit a stronger activation in expert i than in other experts. This strongest activation indicates that expert i is optimized to develop capabilities best suited to X i (Lv et al., 2025).
Constraint 2 requires that expert i responds more strongly to its own proxy token R[i] than by any other R[j]. This ensures each R[i] accurately represents expert i, guaranteeing that tokens most needing expert i are correctly routed to it.
As α decreases, the two constraints become stricter, thereby enforcing stronger expert-router coupling. Additionally, α enables flexible regulation of specialization: a smaller α increases the gap between M [i, i] and M [i, j], reflecting greater expert specialization as experts exhibit more differentiated responses to the same inputs. This feature makes the ERC loss a useful tool for investigating expert specialization and provides deeper insight into MoE behavior, as demonstrated in §4.2.
We translate these two constraints into expert-router coupling loss, formally defined as:
L ERC = 1 n 2 n i=1 n j̸ =i (max (M [i, j] -αM [i, i], 0) + max (M [j, i] -αM [i, i], 0)) .(3)
The three steps for computing expert-router coupling loss are illustrated in Figure 1. For implementation details, we provide PyTorch-style pseudocode in Figure 8.
this section cite: ['b25']

Section: BOUNDED RANDOM NOISE FOR GENERATING PROXY TOKENS
The perturbed proxy token R[i] = R[i] ⊙ δ i makes expert i's coupling generalize effectively from R[i] alone to X i . To ensure the perturbed point R[i] remains within its original cluster, we require a bounded perturbation. We therefore model the noise δ i as a multivariate uniform distribution,
δ i ∼ U(1 -ϵ i , 1 + ϵ i ) d . Let j = arg min j * ̸ =i ∥R[i] -R[j *
]∥ be the nearest cluster center. For the noise level ϵ to be sufficient to avoid perturbing the cluster, it must satisfy:
ϵ i ≤ ∥R[i] -R[j]∥ 2∥R[i]∥ .(4)
The derivation of this bound is provided in Appendix B. We set ϵ i to its maximum value, i.e., the right-hand side of this inequality. Notably, the value of ϵ i is dynamically computed at each layer and every training step.
this section cite: []

Section: EFFICIENCY ANALYSIS

this section cite: []

Section: Training efficiency
In a standard MoE layer, T tokens are processed by K experts, resulting in a total computational cost of 6T KDd FLOPs. expert-router coupling loss introduces only 2n 2 Dd additional FLOPs, a cost that is negligible in practical pre-training setups where K is often in the millions. In contrast, AoE introduces an additional overhead of 2T (n-K)dr FLOPs (recall that r is AoE's factorization rank; see §2). Given that typical MoE-LLMs operate at sparsity levels far below 25% (i.e., n > 4K), this overhead ratio exceeds r/D, making it prohibitive. A detailed breakdown of the FLOP calculations supporting the above theoretical analysis is provided in Appendix C.1.
The efficiency of our method is confirmed in practice. The ERC loss maintains low overhead during LLM pre-training under multiple parallelism strategies, adding only 0.2-0.8% overhead in our experiments. We provide a complete analysis of these real-world distributed conditions and measured throughputs in Appendix C.2.
Overhead-free inference Expert-router coupling loss introduces no additional inference overhead, since it is not applied at that stage. In contrast, AoE maintains the same forward computation, along with its associated overhead.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: EXPERIMENTAL SETTINGS
We compare the ERC-loss-augmented MoE against both the vanilla MoE and AoE baselines. All models are trained from scratch with 3B parameters. This parameter size is chosen because it represents the largest scale at which we could successfully train the AoE model under our available resources. Our implementation is based on OLMoE (Muennighoff et al., 2025). The models comprise 12 layers with d = 1536 and D = 768. Each Transformer (Vaswani et al., 2017) layer has 16 attention heads and n = 64 experts, where K = 8 experts are selected per token. For the AoE model, we set r = 512 to ensure consistent total parameter count. The number of activated parameters is 500M. Each model is trained on 500B tokens from the open-source dataset dolmap-v1.5-sample (Soldaini et al., 2024), using a batch size of 3 million tokens. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with (β 1 , β 2 ) = (0.9, 0.95), a weight decay of 0.1, and a learning rate of 4e-4 with a cosine schedule decaying to 4e-5. A load balancing loss (Fedus et al., 2022) with a weight of 0.01 is applied consistently in all experiments.
For simplicity, the loss weight of the ERC loss is fixed at 1, and we use α = 1 by default if not specified.
We evaluate LLMs on the following tasks: ARC-Challenge (Clark et al., 2018), Common-senseQA (Talmor et al., 2019), COPA (Roemmele et al., 2011), BoolQ (Clark et al., 2019), Hel-laSwag (Zellers et al., 2019), OpenbookQA (Mihaylov et al., 2018), SciQ (Welbl et al., 2017), Social IQa (Sap et al., 2019), WinoGrande (Sakaguchi et al., 2021), and MMLU (Hendrycks et al., 2021a). In terms of efficiency, MoE models with and without ERC loss have nearly identical throughput and memory costs. By contrast, AoE requires 1.6× more training hours and 1.3× higher memory usage, limiting further scaling due to impractical training times and out-of-memory issues.
Expert-router coupling loss is compatible with the load balancing loss. As shown in Figure 3(b), the difference in load balancing loss between MoE combined with L ERC and the vanilla MoE is on the order of 10 -5 . This difference is negligible given that the overall load balancing loss magnitude remains around 10 -2 . By comparison, the loss difference between AoE and vanilla MoE is approximately 4 × 10 -4 . Although this difference is still small, it is notably larger than the difference exhibited by ours.
56.5 55.0 54.5 500 400 300 0.0110 0.0108 0.0106 0.0104 0.0102 Tokens (B) 200 500 400 300 100 (b) Load Balance Loss (a) Average Downstream Task Accuracy Tokens (B) 55.5 56.0 54.0 MoE +ℒ !"#
this section cite: ['b27', 'b41', 'b36', 'b23', 'b7', 'b3', 'b38', 'b32', 'b2', 'b47', 'b26', 'b44', 'b34', 'b33']

Section: MoE AoE
Figure 3: The 3B-scale MoE with ERC loss achieves substantial, stable gains while maintaining effective load balancing. Figure 9 shows task-specific details.
this section cite: []

Section: VALIDATING ERC LOSS IN 15B-PARAMETER MOES
We scale models to 15 billion parameters by increasing n to 256 (keeping K=8) and doubling the model depth. This configuration results in a total of 15B parameters with approximately 700M activated. Other training hyper-parameters largely follow the setup in Section 4.1. As a large-scale, high-sparsity model, the AoE method failed to train due to being overly costly and is thus omitted from comparison.
Table 1 shows that the benefits of the ERC loss persist across various public benchmarks more challenging than those used for 3B models, including MMLU (Hendrycks et al., 2021a), C-Eval (Huang et al., 2023), MMLU-Pro (Wang et al., 2024b), AGI-Eval (Zhong et al., 2024), BBH (Suzgun et al., 2023), MATH (Hendrycks et al., 2021b), GSM8K (Cobbe et al., 2021), and TriviaQA (Joshi et al., 2017). The consistent performance improvements demonstrate that our method effectively addresses the expert-router decoupling problem even at scale. Throughout this large-scale training, we observed no loss spikes or abnormal gradients.
this section cite: ['b15', 'b48', 'b37', 'b4', 'b16']

Section: THE ERC LOSS IS AN EFFECTIVE TOOL FOR EXPLORING EXPERT SPECIALIZATION
With the ERC loss, experts are more specialized, as they exhibit greater discrimination between tokens they process and those they do not, compared to vanilla MoE (without the ERC loss). An intuitive demonstration of this specialization comes from visualizing expert parameters. Following Beyond merely promoting specialization, the ERC loss can also serve as a powerful tool for exploring it. We show this capability through two features below.
Feature 1: α enables a controllable investigation into optimal specialization In the ERC loss, α governs the coupling strength between experts and the router. When α = 0, the ERC loss encourages R[i] to be orthogonal to the parameters of other experts, thereby maximizing specialization. Conversely, when α → 1, the loss permits smaller differences in how all experts' responsiveness to R[i], thus reducing specialization. Notably, α = 1 only weakens the ERC loss's constraints to their maximum extent; it still retains a degree of specialization stronger than the spontaneously emerged specialization in a vanilla MoE model.
Feature 2: ϵ provides a quantitative measure for specialization The noise level ϵ exhibits a strong correlation with α, and it can reflect changes in expert specialization throughout the training process. This correlation exists because as α increases, experts are allowed to be more homogeneous. This growing homogeneity among experts, in turn, reduces the separation between the cluster centers in the router as they are tightly coupled. A smaller separation between cluster centers ultimately derives a smaller ϵ. Thus, ϵ is a quantitative metric tracking expert specialization.
this section cite: []

Section: Experiments
The following experiments support these two features. In Figure 5(a), we plot ϵ at each training step across a parameter search over α ∈ {0.4, 0.6, 0.8, 1.0}. Consistent with our analysis, increasing α, which reduces expert specialization, indeed leads to a corresponding decrease in ϵ. Note that measuring router cluster distance is uninformative in vanilla MoE training without the ERC loss, because the router and experts are uncoupled and cluster distances do not reflect expert capability dynamics. We further compared downstream task performance across different values of α. Figure 5(b) shows that all tested α values outperform the vanilla MoE model. This not only confirms the robust effectiveness of the ERC loss but also demonstrates that the specialization spontaneously formed by vanilla MoE models is inadequate.
The optimal specialization degree Figure 5(b) shows that pursuing extreme specialization is not advisable, as model performance degrades with overly strict α. This highlights a trade-off between promoting expert specialization and maintaining effective collaboration, which is under-discussed in previous work.
The optimal specialization degree is influenced by several factors. The core consideration is whether, among all n K possible expert combinations, an effective K-expert set can be assembled for any given input. In general, smaller values of n favor more generalist experts, while larger n can support a higher degree of specialization. However, we currently lack quantitative metrics to characterize "large" or "small" n and K across different models; as a result, determining the optimal trade-off remains largely empirical. For example, in our experiments with a fixed K = 8: When n = 64, the optimal α = 1, suggesting n = 64 is not "large" for our 3B-parameter models. In contrast, with n = 256, we searched for an optimal α = 0.5, indicating n = 256 is "large" for our 15Bparameter models. This trade-off is also shaped by other architectural choices, such as the use of shared expertsfoot_2 . A deeper investigation into these interacting factors, reliable quantitative metrics for specialization, and an automated evaluation of the optimal specialization degree for a given model are left as important problems for future works. For practitioners implementing the ERC loss, we recommend starting with α = 1, which eliminates expert decoupling and should provide some gains. Further improvement may be achieved by experimenting with lower α values, depending on the specific configuration of your model.
Several studies (Guo et al., 2025;Liu et al., 2024;Hendawy et al., 2024) have promoted specialization via expert output orthogonality. We argue, however, that orthogonalizing expert outputs does not equate to achieving extreme specialization, as the magnitude (norm) of an expert's response to a token remains unconstrained. Moreover, finding a set of orthogonalized high-dimensional vectors is not difficult, making it unclear whether such orthogonality yields sufficiently discriminative representations. Consequently, one should not interpret these fine-tuning experiments as supporting a broad claim that "more specialization is always better." On a separate note, orthogonality among router embeddings (Baidu-ERNIE-Team, 2025) is only weakly correlated with specialization, since the router and experts are typically decoupled. As demonstrated in ablation studies, enforcing router orthogonality might not be a critical factor for pre-training MoE models.
this section cite: ['b10', 'b21', 'b12']

Section: ABLATION STUDIES
Choice of activations for computing M We considered five candidates for calculating M : using the norms of (a) RW g , (b) RW p , (c) SiLU( RW g ), (d) the post-SwiGLU activations (i.e., SiLU( RW g ) ⊙ RW p ), and (e) experts' final outputs (i.e., (SiLU( RW g ) ⊙ RW p )W o ). As shown in Figure 6(a), RW g is the most effective among all alternatives. While using the final output achieves comparable performance, it incurs a higher cost. We therefore adopt RW g as our default choice.
Random noise δ enables the generalization of coupling The random noise δ allows R[i] to better capture the samples within X i . To validate its importance, we conducted an ablation study where we trained an MoE with the ERC loss but removed δ. Specifically, we computed M directly using the original R instead of the noise-augmented R. As shown in Figure 6(b), removing δ greatly degrades performance. This is because the coupling between routers and experts becomes overfitted to R, failing to generalize to the real inputs that R[i]s represent.
this section cite: []

Section: Comparison with contrastive regularization solely on routers
The router orthogonalization loss (Baidu-ERNIE-Team, 2025) requires R (the row-wise normalization of R) to satisfy: R R⊤ = I.
As shown in Figure 6(c), the orthogonalization loss yields only limited gains. We attribute this to our finding that the router embeddings in our baseline MoE model are already nearly orthogonal, with an average absolute cosine similarity of 0.15. This value corresponds to angles between router embeddings mostly ranging from arccos(0.15) = 81°to arccos(-0.15) = 99°. Notably, we do not imply that all MoEs always have nearly orthogonal router embeddings, as this may depend on the data or specific architecture; we report this only as a characteristic of our models, which explains the limited gains from the orthogonalization loss.
This result further demonstrates that weak coupling between routers and experts is a more critical issue than imperfect orthogonality in router embeddings. The significant gains from ERC, even when applied to a baseline with already near-orthogonal routers, provide clear evidence.
Furthermore, it is important to note that even if both routers and experts are orthogonalized, there is no guarantee that each R[i] will be aligned with W i g . Therefore, the ERC loss cannot be reduced to contrastive techniques applied individually to routers or experts, such as orthogonalization loss.
Additional analyses Appendix A provides analyses addressing several frequently asked questions, including the effect of α > 1, and verifying that the model decreases the ERC loss by learning meaningful coupling rather than by manipulating parameter norms.
this section cite: []

Section: RELATED WORKS
Auxiliary loss for MoEs Auxiliary losses are crucial for training large-scale MoE models. Most existing work in this area focuses primarily on enhancing training stability. For instance, many studies have proposed auxiliary losses to address load balancing challenges (Fedus et al., 2022;Qiu et al., 2025;Wang et al., 2024a);Zoph et al. (2022) introduced the z-loss, which penalizes excessively large logits in the gating network to enable stable training. MoE concepts have also inspired mixtures of attention heads or entire layers (Gong et al., 2024;Lin et al., 2024), where auxiliary losses play a critical role in effective optimization. Our ERC loss is the first tailored to strengthen the expert-router coupling. Other related auxiliary losses enhancing expert specialization or orthogonality are discussed below. Dai et al. (2024) introduced a shared expert to handle general capabilities, encouraging the others to be more specialized. Guo et al. (2025) proposes an auxiliary loss to minimize the pairwise projections of the selected top-K experts' outputs for each token, reducing expert overlap but incurring high cost due to K 2 cosine similarity calculations per token. Other methods scale the number of tiny experts to millions, making each expert more atomic and thus more specialized (Yang et al., 2025;Park et al., 2025;He, 2024), but are memory-bounded. Beyond efficiency, these methods face three major limitations: (1) no quantitative control over specialization degree; (2) no exploration of the specialized-generalized ability trade-off; and (3) failure to strengthen expertrouter coupling. Our method addresses all three, both efficiently and effectively.
this section cite: ['b7', 'b30', 'b7', 'b9', 'b20', 'b5', 'b10', 'b46', 'b28']

Section: Expert specialization
Some works (Guo et al., 2025;Liu et al., 2024;Hendawy et al., 2024) maximize specialization by training orthogonal experts, but their evaluations are based on fine-tuning (or reinforcement learning) experiments. We contend that orthogonalizing expert outputs is not equivalent to achieving extreme specialization, and further, that the optimal degree of specialization is a complex problem affected by various factors and requires further exploration.
Contrastive learning Constraints 1 and 2 bear similarity to contrastive learning (Chen et al., 2020;van den Oord et al., 2019;Khosla et al., 2020). Some MoE research (Luo et al., 2024;Guo et al., 2025) applied contrastive learning to expert outputs, encouraging specialization. However, naively applying contrastive learning to either routers or experts leaves the weak expert-router coupling unaddressed.
this section cite: ['b10', 'b21', 'b12', 'b1', 'b39', 'b24', 'b10']

Section: CONCLUSIONS
The weak coupling between router decisions and expert capabilities limits MoE models in multiple important aspects. We propose expert-router coupling loss that tightly couples router parameters with their corresponding experts. The proposed ERC loss improves MoE-based LLMs on downstream tasks while incurring negligible training overhead. In addition, it exhibits several desirable properties that not only provide deeper insight into the behavior of MoE models but also offer a promising tool for future research on expert specialization.
this section cite: []

Section: References
Ref_id:b0 Title:  Year: (2025)
Ref_id:b1 Title: A simple framework for contrastive learning of visual representations Year: (2020)
Ref_id:b2 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b3 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b4 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b5 Title: Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b6 Title: Deepseek-v3 technical report Year: (2025)
Ref_id:b7 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b8 Title: Transformer feed-forward layers are key-value memories Year: (2021-11)
Ref_id:b9 Title: Mixture-of-modules: Reinventing transformers as dynamic assemblies of modules Year: (2024-11)
Ref_id:b10 Title: Advancing expert specialization for better moe Year: (2025)
Ref_id:b11 Title: Mixture of a million experts Year: (2024)
Ref_id:b12 Title: Multi-task reinforcement learning with mixture of orthogonal experts Year: (2024)
Ref_id:b13 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b14 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b15 Title: Maosong Sun, and Junxian He. C-eval: A multilevel multi-discipline chinese evaluation suite for foundation models Year: (2023)
Ref_id:b16 Title: TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension Year: (2017-07)
Ref_id:b17 Title:  Year: ()
Ref_id:b18 Title: Advances in Neural Information Processing Systems Year: (2020)
Ref_id:b19 Title: {GS}hard: Scaling giant models with conditional computation and automatic sharding Year: (2021)
Ref_id:b20 Title: Mixture of in-context experts enhance llms'long context awareness Year: (2024)
Ref_id:b21 Title: Diversifying the mixture-of-experts representation for language models with orthogonal optimizer Year: (2024)
Ref_id:b22 Title: Deja vu: contextual sparsity for efficient llms at inference time Year: (2023)
Ref_id:b23 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b24 Title: Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models Year: (2024)
Ref_id:b25 Title: Autonomy-of-experts models Year: (2025)
Ref_id:b26 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b27 Title: Olmoe: Open mixture-of-experts language models Year: (2025-08)
Ref_id:b28 Title: Monet: Mixture of monosemantic experts for transformers Year: (2025)
Ref_id:b29 Title: Competesmoe -effective training of sparse mixture of experts via competition Year: (2024)
Ref_id:b30 Title: Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models Year: (2025-07)
Ref_id:b31 Title: Qwen2.5 technical report Year: (2024)
Ref_id:b32 Title: Choice of plausible alternatives: An evaluation of commonsense causal reasoning Year: (2011)
Ref_id:b33 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b34 Title: Social IQa: Commonsense reasoning about social interactions Year: (2019-11)
Ref_id:b35 Title: Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer Year: (2017)
Ref_id:b36 Title: Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research Year: (2024)
Ref_id:b37 Title: Challenging BIG-bench tasks and whether chain-of-thought can solve them Year: (2023-07)
Ref_id:b38 Title: CommonsenseQA: A question answering challenge targeting commonsense knowledge Year: (2019-06)
Ref_id:b39 Title: Representation learning with contrastive predictive coding Year: (2019)
Ref_id:b40 Title: Visualizing data using t-sne Year: (2008)
Ref_id:b41 Title: Attention is all you need Year: (2017)
Ref_id:b42 Title: Auxiliary-loss-free load balancing strategy for mixture-of-experts Year: (2024)
Ref_id:b43 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b44 Title: Crowdsourcing multiple choice science questions Year: (2017)
Ref_id:b45 Title: Union-of-experts: Experts in mixture-of-experts are secretly routers Year: (2025)
Ref_id:b46 Title: Mixture of experts made intrinsically interpretable Year: (2025)
Ref_id:b47 Title: HellaSwag: Can a machine really finish your sentence? Year: (2019-07)
Ref_id:b48 Title: AGIEval: A human-centric benchmark for evaluating foundation models Year: (2024-06)
