Title: Advancing Expert Specialization for Better MoE
Abstract: Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training. To address this, we propose a simple yet effective solution that introduces two complementary objectives: (1) an orthogonality loss to encourage experts to process distinct types of tokens, and (2) a variance loss to encourage more discriminative routing decisions. Gradient-level analysis demonstrates that these objectives are compatible with the existing auxiliary loss and contribute to optimizing the training process. Experimental results over various model architectures and across multiple benchmarks show that our method significantly enhances expert specialization. Notably, our method improves classic MoE baselines with auxiliary loss by up to 23.79%, while also maintaining load balancing in downstream tasks, without any architectural modifications or additional components. Our code is available at this link.

Section: Introduction
Large language models (LLMs) [67,65,62,6] have demonstrated remarkable generalization capabilities [52,69,74,73] across a wide range of tasks [53,24], but their inference cost [15,57] grows rapidly with scale, hindering practical deployment and efficiency. Mixture-of-Experts (MoE) [9,3,37] architectures alleviate this problem by activating only a subset of experts per input [19], thus enabling greater model capacity without a commensurate increase in computational overhead [22,49,33]. To maximize parameter utilization, MoE systems typically introduce load balancing [56,20] objectives that encourage a more uniform routing of tokens across experts during pre-training.
While load balancing is effective in avoiding idle experts during large-scale pre-training, it often hinders model adaptation in the post-training stage for downstream tasks, where data distributions are narrower and more domain-specific. In such settings, token occurrences are typically concentrated within particular subspaces (e.g., numeric or symbolic tokens in math tasks), intensifying the tension between balanced routing and expert specialization. A widely observed phenomenon is that load balancing encourages uniform expert routing across inputs, resulting in highly overlapping token distributions [14,79]. This overlap leads to convergence in expert representations [46], ultimately compromising the development of specialized functionalities. The lack of specialization [14] becomes particularly problematic during fine-tuning [17,60,2,80] on downstream tasks with strong domain preferences, where the model struggles to adapt and exhibits degraded performance [34]. This highlights a core challenge in MoE post-training: the inherent conflict between encouraging expert specialization [50,38,36]and enforcing routing uniformity [83] via auxiliary losses. From the expert perspective, load-balanced routing causes overlapping training intentions across experts [14,45,46,7], suppressing the development of distinct expert behaviors. From the router perspective, as experts become less specialized, the router receives less variation across experts, leading to increasingly uniform and less informed token-to-expert assignments [82]. These dynamics form a self-reinforcing loop: diminished specialization and uniform routing exacerbate each other over time, progressively degrading both expert expressiveness and routing quality [20]. This compounding effect reveals a deeper limitation of existing training objectives, which lack mechanisms to decouple expert specialization from the uniformity constraints imposed by auxiliary losses.
this section cite: ['b66', 'b64', 'b61', 'b5', 'b51', 'b68', 'b73', 'b72', 'b52', 'b23', 'b14', 'b56', 'b8', 'b2', 'b36', 'b18', 'b21', 'b48', 'b32', 'b55', 'b19', 'b13', 'b78', 'b45', 'b13', 'b16', 'b59', 'b1', 'b79', 'b33', 'b49', 'b37', 'b35', 'b82', 'b13', 'b44', 'b45', 'b6', 'b81', 'b19']

Section: Experts Token Balance

this section cite: []

Section: Experts Load Variance Decrease
To address this challenge, we propose a gradient-based multi-objective optimization framework that promotes expert specialization and routing diversification, while preserving load balance from auxiliary loss. We introduce two complementary objectives, as shown in Figure 1: 1) Expert Specialization, which fosters distinct expert representations by ensuring that each expert specializes in processing different tokens. 2) Routing Diversification, which drives differentiated routing decisions, enabling more precise token-to-expert assignments by enhancing the variance in routing. By jointly optimizing these objectives, our method mitigates the trade-off between model performance and routing efficiency in MoE training. We demonstrate that our approach successfully achieves:
• Enhanced expert-routing synergy. Our joint objectives reduce expert overlap by up to 45% and increase routing score variance by over 150%, leading to clearer specialization and more discriminative expert assignment.
• Stable load balancing. Despite introducing new objectives, our method matches the baseline's MaxVioglobal across all models, with RMSE under 8.63 in each case.
• Improved downstream performance. We achieve 23.79% relative gains across 11 benchmarks and outperform all baselines on 92.42% of tasks ,all without modifying the MoE architecture.
this section cite: []

Section: Motivation

this section cite: []

Section: Preliminaries of MoE
In a typical MoE layer, let there be n experts, and a sequence of input tokens represented by X = {x 1 , x 2 , • • • , x N }, where N is the total number of tokens in the sequence. The routing score matrix after applying the top-k mechanism is denoted as:
S =     s 11 s 12 • • • s 1n s 21 s 22 • • • s 2n . . . . . . . . . . . . s N 1 s N 2 • • • s N n     , n j=1 s ij = 1, i = 1, 2, • • • , N(1)
where s ij represents the routing weight assigned to the i-th token for the j-th expert.
Let F = {f 1 , f 2 , • • • , f n } represent the proportion of tokens assigned to each expert, where f j is the number of tokens assigned to the j-th expert. For any given MoE layer, the total loss function L consists of two parts, the main loss L h and the auxiliary loss L aux :
L = L h + α • L aux = L h + α n j=1 f j • p j , p j = N i=1 s ij ,(2)
where L h is the loss computed from the output of the MoE layer, and L aux is the auxiliary loss term, α denotes the weighting coefficient for the auxiliary loss. Here, p j represents the total routing score for the j-th expert, which is the sum of the routing weights for all tokens assigned to that expert.
this section cite: []

Section: Observations
Obs I (Expert Overlap): Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert.
It has been observed that the auxiliary loss function is independent of the expert parameter matrices θ Ej . Therefore, for the j-th expert, its gradient can be written as:
∂L ∂θ Ej = ∂L h ∂θ Ej + α • ∂L aux ∂θ Ej = ∂L ∂y h • ∂y h ∂θ Ej = N i=1 x i • s ij , j = 1, 2, • • • , n.(3)
where θ Ej is the parameter matrix of the j-th expert, and y h is the output of the MoE layer. During gradient descent, the addition of the auxiliary loss L aux forces the routing mechanism to evenly distribute the tokens across experts as much as possible.
This results in input token x i being assigned to an expert that may not be semantically aligned with it, causing an unintended gradient flow to expert j. Mathematically, after applying the top-k mechanism, the routing score s ij transitions from 0 to a non-zero value, introducing gradients from tokens that originally had no affinity with expert j.
Obs II (Routing Uniformity): As training progresses, the routing output tends to become more uniform, with the expert weight distribution gradually converging towards an equal allocation.
To understand this phenomenon, we first examine the source of gradients with respect to the routing parameters θ R . Since the routing mechanism produces only the score matrix S = s ij , the gradient ∂L/∂θ R can be written as:
∂L ∂θ R = ∂L h ∂θ R + α • ∂L aux ∂θ R = N i=1 x i n j=1 θ Ej • ∂s ij ∂θ R + α • n j=1 f j N i=1 ∂s ij ∂θ R ,(4)
where x i • θ Ej represents the output of expert j for token x i , and f j denotes the frequency with which expert j is selected. This formulation reveals that the routing gradient is primarily influenced by the expert outputs and the token distribution across experts.
The auxiliary loss L aux is introduced to encourage balanced token assignment by optimizing the uniformity of f j . However, since f j is non-differentiable, direct optimization is not feasible. Instead, a surrogate variable p j , which is differentiable and positively correlated with f j , is employed to approximate the objective and enable gradient flow back to the routing network.
As training proceeds, the optimization objective increasingly favors the uniformity of p j , which drives f j toward an even distribution. Moreover, as discussed in Observation I, incorrect token assignments caused by auxiliary regularization introduce overlapping gradients among experts, increasing the similarity of x i • θ Ej across different j.
Obs III (Expert-Routing Interaction): While Obs I concerns expert specialization, while Obs II reflects the uniformity of routing. These two effects interact during training, jointly driving the model toward degraded performance.
• Expert-side interference caused by Obs I leads to blurred specialization. Tokens are assigned to mismatched experts, and the resulting gradient interference reduces expert distinctiveness. As the routing weights become more uniform, different experts receive similar gradients from the same tokens, increasing their functional overlap.
• This expert similarity feeds back into the routing mechanism. As expert outputs become less distinguishable, the routing network finds fewer cues to differentiate among experts, leading to even more uniform weight distributions. This promotes random top-k selection and further misalignment between tokens and their optimal experts.
Together, this loop gradually steers the model toward more uniform token allocation and reduced expert specialization, highlighting potential opportunities for improving the routing strategy and expert assignment.
this section cite: []

Section: Method
Based on the observations above, we propose the following design to mitigate expert overlap and routing uniformity, the overall loss function L is defined as follows:
L = L h + L balance , L balance = α • L aux + β • L o + γ • L v ,(5)
where L aux represents the existing auxiliary loss, with coefficient α, and the newly introduced orthogonality loss L o and variance loss L v (see Subsec 3.1), with coefficients β and γ respectively. It is worth noting that the theoretical complementarity of these optimization objectives, rather than any inherent conflict, is formally analyzed and demonstrated in Subsection 3.2.
this section cite: []

Section: Implementations of Losses L o and L v
In this section, we introduce two critical loss functions L o and L v that act on the expert and router components, respectively.
this section cite: []

Section: Expert Specialization.
We introduce an orthogonalization objective that encourages independent expert representations. Specifically, we design the following orthogonality loss:
L o = N i=1 n j=1 n k=1 k̸ =j ⟨x ij , xik ⟩ ⟨x ik , xik ⟩ + ϵ xik 2 , xij = x i • θ Ej • I {sij >0} ,(6)
where ⟨•⟩ denotes the inner product between two vectors, and Isij > 0 is an indicator function that evaluates to 1 when s ij > 0 and 0 otherwise. Here, xij represents the output of expert j for token x i after the top-k routing selection.
The orthogonality loss L o reduces the overlap between different expert outputs within the same top-k group by minimizing their projections onto each other. This encourages experts to develop more distinct representations, promoting specialization in processing different token types.
this section cite: []

Section: Routing Diversification.
We introduce a variance-based loss to encourage more diverse routing decisions and promote expert specialization. Specifically, we define the variance loss as:
L v = - N i=1 n j=1 1 n • (s ij -sj ) 2 , sj = 1 N • N i=1 s ij ,(7)
where sj denotes the average routing score for expert j across the batch. By maximizing the variance of routing scores, L v discourages uniform token-to-expert assignments and encourages more deterministic and distinct routing patterns, thereby facilitating expert specialization.
this section cite: []

Section: Compatibility of Multi-Objective Optimization
In this section, we analyze how each component influences the optimization dynamics of expert parameters θ Ej and routing parameters θ R during training. Meanwhile, we will focus on the optimization and compatibility of the two losses L o and L v with respect to load balancing and expert specificity. The following two key questions guide our analysis.
this section cite: []

Section: Balancing Expert and Routing.
How can expert (L o ) and routing (L v ) optimizations be designed to complement each other without compromising their respective objectives?
We first demonstrate that L o and L v are compatible in their optimization directions within MoE, then show that they mutually reinforce each other.
this section cite: []

Section: Mutually Compatible.
We elaborate on the compatibility of L o and L v from the perspectives of expert and Routing.
From the expert perspective, we observe that the auxiliary loss L aux and the variance loss L v do not directly contribute gradients to the expert parameter matrix θ Ej . Consequently, the gradient of the total loss with respect to θ Ej is derived solely from the primary task loss L h and the orthogonality loss L o :
∂L ∂θ Ej = N i=1   sij • g yi + β • n k=1 k̸ =j xik x⊤ ik ⟨x ik , xik ⟩ + ϵ • xij    • x ⊤ i (8
)
Here, g yi = ∇ yi L h denotes the gradient of the primary task loss with respect to the model output. This gradient is influenced by both the routing score s ij and the expert representation xij . As training progress, the variance of expert weights increases, and the gradient encourages stronger preferences in different directions for each token.
From the routing perspective, we notice that L o does not affect the gradient with respect to routing parameters θ R . The gradient of the total loss with respect to θ R is:
∂L ∂θ R = ∂L ∂s ij • ∂s ij ∂θ R = N i=1 n j=1 xij + α • f j -γ • 2(N -1) nN • (s ij -sj ) • ∂s ij ∂θ R . (9
)
This gradient is influenced by expert representations xij , expert load f j , and routing weights s ij . As the model converges, the expert load f j becomes more balanced, and the variance of routing weights s ij increases. Orthogonalizing expert representations causes the routing gradients to flow in more orthogonal directions, making the weight allocation more biased towards the representations and increasing the weight variance.
this section cite: []

Section: Summary.
Expert parameters θE j are solely influenced by the gradients of Lo without conflict. While routing parameters θR are affected by both Lo and Lv, the objectives of these two losses (orthogonalityfriendliness vs. score diversification) remain non-conflicting.
Mutually Reinforcing. L o aims to encourage the effective output vectors of different selected experts j and k to tend to be orthogonal for the same input token x i , i.e., ⟨x ij , xik ⟩ ≈ 0. The learning signal for the routing mechanism partially originates from the gradient of the primary task loss L h with respect to the routing score s ij :
∂L ∂s ij = g T yi xij from L h + α ∂L aux ∂s ij from Laux -γ 2(N -1) nN (s ij -sj ) from Lv , y i = j s ij xij , g yi = ∂L h ∂y i (10
)
Assuming p ij = g T yi xij , when the expert outputs tend to be orthogonal, for any given task gradient g yi , the projections p ij onto these approximately orthogonal expert outputs are more likely to exhibit significant differences. The increased variance of the primary task-related signals p ij implies that the routing mechanism receives more discriminative and stronger learning signals, which creates more favorable conditions for L v to achieve diversification of routing scores.
L v enhances the diversity of routing scores s ij by optimizing routing parameters θ R . Meanwhile, due to the influence of L o 's gradient β ∂Lo ∂sij on θ R , routing tends to assign more specialized token subsets T j to each expert j. Expert parameters θ Ej learn the unique features of tokens within T j , leading to gradual functional divergence among experts, thereby promoting expert orthogonality.
this section cite: []

Section: Summary.
Lo induces orthogonal expert outputs xij, enhances the discriminative power of routing signals g T y i xij, and generates diverse routing scores sij to support Lv. Meanwhile, Lv drives experts to specialize in distinct token subsets via sij and promotes parameter divergence of θE j to support Lo. Together, they form a mutually reinforcing cycle.
Multi-Objective Optimization. How do expert and routing maintain their balance while enhancing L aux and L h independently, ensuring mutually beneficial performance improvements? Lemma 1 Let S ∈ R N ×n be a matrix that satisfies following conditions: each row sums to 1, each row contains k non-zero elements and n -k zero elements. Then, there always exists a state in which the following two objectives are simultaneously optimized: 1. The sum of the elements in each column tends to the average value N n ; 2. The variance of the non-zero elements in each row increases.
this section cite: []

Section: Lemma 2
For two sets of points A and B of equal size, it is always possible to partition A ∪ B such that A ∩ B = ∅ and |A| = |B|.
The overall objective function L optimizes four key dimensions: accurate data fitting(L h ), expert orthogonalization(L o ), balanced expert routing weights(L aux ), and increased variance in routing outputs(L v ). Our core objective is to achieve an optimal balance by jointly optimizing these multiple objectives, ensuring they complement each other for enhanced model performance.
As shown by Lemma 1, expert load f j and routing weights s ij can be optimized together. As demonstrated in Lemma 2, the objectives of orthogonalization and load balancing are not in conflict and can be jointly optimized. Thus, both expert and routing modifications can be optimized alongside load balancing (balanced expert routing weights).
Moreover, orthogonalization enhances routing weight variance, in turn, improves expert specialization (as discussed in Section 2.2). This leads to more distinctive expert representations, aligning with performance (accurate data fitting) improvements when optimized together.
this section cite: []

Section: Experiments
In this section, we conduct experiments to address the following research questions:
• RQ1: Does introducing the orthogonality loss (L o ) and variance loss (L v ) lead to better overall performance in downstream tasks compared to baseline approaches?
• RQ2: To what extent does our method maintain expert load balancing during training?
• RQ3: How do the orthogonality loss (L o ) and variance loss (L v ) interact with each other, and what are their respective and joint impacts on expert specialization and routing behavior?
• RQ4: What are the individual and combined contributions of L o , L v , and the auxiliary loss L aux to the final model performance?
this section cite: []

Section: Experimental Setup

this section cite: []

Section: Environment.
All experiments are performed on a CentOS Linux 7 server with PyTorch 2.3. The hardware specifications consist of 240GB of RAM, a 16-core Intel Xeon CPU, and two NVIDIA A800 GPUs, each having 80GB of memory. Implementation details are provided in the Appendix F.
this section cite: []

Section: Datasets.
We evaluate our method on a total of 11 benchmarks. Specifically, we use the training sets from Numina [41], GLUE [66], and the FLAN collection [72] to train our models. Our benchmarks include: ❶ Mathematics: GSM8K [12], MATH500 [44], and Numina [41]; ❷ Multi-Domain Tasks: MMLU [31,30], MMLU-pro [70], BBH [63], GLUE [66]; LiveBench [76] and GPQA [59]. ❸ Code generation: HumanEval [10] and MBPP [4]. We group training and test sets by language, reasoning, science, math, and code to match downstream evaluation needs. Detail in Appendix D.
this section cite: ['b40', 'b65', 'b71', 'b11', 'b43', 'b40', 'b30', 'b29', 'b69', 'b62', 'b65', 'b75', 'b58', 'b9', 'b3']

Section: Baselines.
We compare our method with 4 existing MoE training strategies. With Aux Loss [46] applies auxiliary load-balancing losses during routing to encourage expert utilization diversity. GShard [39] introduces a foundational sparse expert framework with automatic sharding and routing; ST-MoE [85] enhances training stability via router dropout and auxiliary losses; Loss-Free Balancing [68] achieves balanced expert routing without auxiliary objectives. Detail in Appendix G.
this section cite: ['b45', 'b38', 'b84', 'b67']

Section: Metrics.
We employ 6 evaluation metrics to test our method in terms of accuracy, expert load balancing (MaxVio global [68]), clustering quality (Silhouette Coefficient), expert specialization (Expert Overlap), routing stability (Routing Variance), and prediction error (RMSE). Detail in Appendix E.
this section cite: ['b67']

Section: Performance in Downstream Tasks (RQ1)
To verify that our L balance enhances model performance in downstream task scenarios through expert orthogonality and routing output diversification, as shown in Table 1, we design downstream task scenarios on 11 well-known benchmarks and validate our method against four baseline methods with distinct loss designs on three widely used MoE models. We make the following observations:
Obs.❶ Baseline methods without guidance for expert specialization exhibit varied performance and fail to effectively improve downstream task performance. As shown in Table 1, the four baseline methods show no clear overall performance ranking across the 11 tasks, with performance variations within 2% in many tasks. Their overall performance is significantly lower than our method, demonstrating no potential to improve downstream task performance.
Obs.❷ Our method guiding expert specialization effectively enhances model performance in downstream tasks. As shown in Table 1, we achieve state-of-the-art (SOTA) results in over 85% of the 33 tasks across the three models. In some tasks, the average across multiple measurements even outperforms the next-best method by nearly 7%. Extensive experiments indicate that our method significantly improves model performance in downstream task scenarios by enhancing expert specialization. More results on additional baselines and MoE architectures are provided in Appendix I.
this section cite: []

Section: Load Balancing (RQ2)
To verify that our newly added losses L v and L o do not affect the load balancing effect, we conduct statistical measurements on the load balancing of all combinations of L aux , L v , and L o across various models during training.
Figure 2 shows the variation of M axV io global ↓ across training steps for different loss combinations, as well as the RMSE of differences between our method and other combinations. We make the following observations:
Obs.❸ Loss combinations without L aux exhibit significantly worse load balancing performance than those with L aux . As shown in Figure 2, across three distinct models, the M axV io global of the w/o all method (with no losses added) is significantly higher than that of other methods, indicating
6WHS RXUV RQO\DX[ ZROY ZROR ZRDOO 0 H WK R G MaxVio global 506(YVRXUV RQO\DX[ ZROY ZROR ZRDOO 'HHS6HHN0RH% ZRDOO ZROR ZROY RQO\DX[ RXUV 6WHS RXUV RQO\DX[ ZROY ZROR ZRDOO 0 H WK R G MaxVio global 506(YVRXUV RQO\DX[ ZROY ZROR ZRDOO 0RRQOLJKW%$% ZRDOO ZROR ZROY RQO\DX[ RXUV 6WHS RXUV RQO\DX[ ZROY ZROR ZRDOO 0 H WK R G MaxVio global 506(YVRXUV RQO\DX[ ZROY ZROR ZRDOO 'HHS6HHN9/LWH ZRDOO ZROR ZROY RQO\DX[ RXUV notably poorer load balancing. In particular, for the DeepSeek-V2-Lite model, the method without L aux converges to 6.14, whereas methods with L aux converge to 2.48, demonstrating that loss combinations containing L aux achieve significantly better load balancing.
Obs.❹ Incorporating any combination of L v and L o into L aux does not affect load balancing.
As shown in Figure 2, for methods with L aux , the trends of "only aux" (no additional losses), "w/o lv" (only L o ), "w/o lo" (only L v ), and "ours" (both L v and L o ) are nearly identical. Additionally, the RMSE (root mean squared error) of our method relative to other baselines does not exceed 0.03, further corroborating the conclusion that the combination of L v and L o does not impact load balancing.
this section cite: []

Section: Behaviors of Experts and Routing (RQ3)
To verify that L v and L o can jointly promote expert orthogonality and routing score diversification, following the method setup in Section 4.3, we will conduct evaluations of expert orthogonality and measurements of routing score diversification for different loss combinations.
VLOKRXHWWHFRHIILFLHQW 'HHS6HHN 9/LWH 0RRQOLJKW %$% 'HHS6HHN 0RH% 0RGHO 6LOKRXHWWH&RHIILFLHQW &RPSDULVRQ RXUV ZROR ZROY ZRDOO RQO\DX[ H[SHUWRYHUODS 'HHS6HHN 9/LWH 0RRQOLJKW %$% 'HHS6HHN 0RH% 0RGHO ([SHUW2YHUODS &RPSDULVRQ RXUV ZROR ZROY ZRDOO RQO\DX[ YDULDQFH 'HHS6HHN 9/LWH 0RRQOLJKW %$% 'HHS6HHN 0RH% 0RGHO 5RXWLQJ9DULDQFH &RPSDULVRQ RXUV ZROR ZROY ZRDOO RQO\DX[ As shown in Figure 3, the first two subplots demonstrate the orthogonality of experts, while the last subplot illustrates the diversification of routing outputs. We make the following observations:
Obs.❺ L o directly promotes expert orthogonality, and L v also aids in expert orthogonality. As shown in the first two panels of Figure 3, our method with both L o and L v achieves state-of-the-art (SOTA) results across three models, with Expert Overlap even dropping below 0.3. The method with only L o and L aux (w/o lv) consistently ranks second-best, indicating that L o has a more significant impact on expert orthogonality. Notably, the method with only L v and L aux (w/o lo) significantly outperforms the method with only L aux across all three models, confirming that L v also contributes to expert orthogonality.
Obs.❻ L v directly enhances routing output diversification, and L o also supports this diversification. Similarly, our method exhibits the highest routing score variance (exceeding 0.010), followed by the method with only L v and L aux , while the method with only L aux performs worst. This strongly supports the conclusion.
Obs.❼ L aux leads to higher expert overlap and more homogeneous routing outputs. Compared to the w/o all method (no losses), the aux only method (with only L aux ) shows a Silhouette Coefficient that is over 0.05 higher and a routing output variance that is 0.0045 higher. This indicates that w/o all exhibits significantly greater expert orthogonality and routing output diversification than aux only.
this section cite: []

Section: Ablation among Losses (RQ4)
To demonstrate that both L o and L v have positive effects on the model's performance in downstream task scenarios, and their combination synergistically enhances each other's efficacy, we design ablation experiments for these two losses on three models.
*60. 0$7+ 1XPLQD7HVW 00/8 00/8SUR %%+ */8( *34$ +XPDQ(YDO 0%33 $EODWLRQ([SHULPHQWRQ 'HHS6HHN0RH% ZRDOO ZROR ZROY RQO\DX[ 2XUV *60. 0$7+ 1XPLQD7HVW 00/8 00/8SUR %%+ */8( *34$ +XPDQ(YDO 0%33 $EODWLRQ([SHULPHQWRQ 'HHS6HHN9/LWH ZRDOO ZROR ZROY RQO\DX[ 2XUV *60. 0$7+ 1XPLQD7HVW 00/8 00/8SUR %%+ */8( *34$ +XPDQ(YDO 0%33 $EODWLRQ([SHULPHQWRQ 0RRQOLJKW%$% ZRDOO ZROR ZROY RQO\DX[ 2XUV
this section cite: []

Section: Figure 4: Ablation Experiments.
The figure illustrates the performance differences of different ablation method combinations across three models on various benchmarks. The vertices on the circles represent the corresponding benchmark names, with the same type connected by the same color. The numbers inside the circles denote the accuracy represented by each circle.
Figure 4 illustrates the performance of different ablation method combinations across various downstream tasks. We make the following observations:
Obs.❽ The combination of L o and L v significantly enhances model performance in downstream tasks, and each loss individually also improves performance. Our method (combining L o and L v ) exhibits the largest coverage area across all three models, nearly encompassing other methods. When either L o or L v is ablated (i.e., w/o lv or w/o lo), the coverage areas of these methods are larger than that of the only aux method (with only L aux ), indicating performance improvements over the baseline.
Obs.❾ L aux impacts model performance on downstream tasks. Figure 4 clearly shows that the only aux method (with only L aux ) is nearly entirely enclosed by other methods across all three models, consistently exhibiting the smallest coverage area. Notably, the w/o all method (with no losses) achieves performance improvements and a larger coverage area than the only aux method when L aux is removed, supporting this conclusion.
Beyond the ablation results in Fig. 4, we further conduct a sensitivity analysis on the loss-weight coefficients α, β, and γ. The detailed results and discussions are provided in Appendix H.1.
this section cite: []

Section: Related Work
Auxiliary Losses in MoE Training. Auxiliary losses [39,85] are commonly used to prevent expert collapse by encouraging balanced expert utilization [14]. Early approaches focus on suppressing routing imbalance, while later works [81] introduce capacity constraints or multi-level objectives to separate routing stability from load balancing [65,39,20]. Recent methods [75] further reduce manual tuning by dynamically adjusting auxiliary weights or replacing them with entropy-based routing [42]. However, fixed-rule strategies may underutilize expert capacity, and dynamic schemes can introduce instability or overhead, making robust balancing still a challenge [32,68].
this section cite: ['b38', 'b84', 'b13', 'b80', 'b64', 'b38', 'b19', 'b74', 'b41', 'b31', 'b67']

Section: Orthogonality in MoE.
Orthogonalization [47,28] improves expert diversity by encouraging independent representations [29]. Some methods [54,84,51] regularize expert weights directly, while others [14,29] assign experts to disentangled subspaces based on task semantics. Recent routingbased approaches [47,58] also impose orthogonality on token-to-expert assignments to reduce redundancy. Nonetheless, static constraints [11] often fail to adapt to dynamic inputs, and dynamic ones [78,35,25,64] may conflict with balancing, complicating expert allocation [32,82,27,68]. Our work addresses these tensions by integrating orthogonalization and balance into a unified, gradient-consistent optimization framework.
this section cite: ['b46', 'b27', 'b28', 'b53', 'b83', 'b50', 'b13', 'b28', 'b46', 'b57', 'b10', 'b77', 'b34', 'b24', 'b63', 'b31', 'b81', 'b26', 'b67']

Section: Limitation & Future Discussion
While L balance balances load and enhances performance in downstream tasks, its potential in other domains remains unexplored. Specifically, it could be extended to visual models, as suggested in recent work [26], and multimodal or full-modal settings [8], offering opportunities for crossdomain applications. Additionally, investigating L balance within lightweight MoE fine-tuning, such as LoRA-MoE [21], could make our approach viable for resource-constrained environments [43].
Furthermore, there is considerable potential in exploring expert-distributed deployment, where L balance can optimize both parameter inference efficiency and model performance. This avenue could significantly enhance the scalability and practicality of MoE models in real-world applications, providing new opportunities for distributed expert architectures.
this section cite: ['b25', 'b7', 'b20', 'b42']

Section: Conclusion
In this work, we present a theoretically grounded framework that resolves the inherent conflict between expert specialization and routing uniformity in MoE training. By introducing orthogonality and variance-based objectives, our method significantly improves downstream performance without any architectural changes. This demonstrates that MoE efficiency and specialization can be simultaneously optimized through loss-level innovations alone. Experiments show the effectiveness of our method.
this section cite: []

Section: References
Ref_id:b0 Title: Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007) Year: (2007-06)
Ref_id:b1 Title: Falcon-40b: an open large language model with state-of-the-art performance Year: (2023)
Ref_id:b2 Title: Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts Year: (2021)
Ref_id:b3 Title: Program synthesis with large language models Year: (2021)
Ref_id:b4 Title:  Year: (2025)
Ref_id:b5 Title: Deepseek llm: Scaling open-source language models with longtermism Year: (2024)
Ref_id:b6 Title: Shortcut-connected expert parallelism for accelerating mixture-of-experts Year: (2024)
Ref_id:b7 Title: A survey on mixture of experts Year: (2024)
Ref_id:b8 Title: A survey on mixture of experts in large language models Year: (2025)
Ref_id:b9 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b10 Title: Sparse moe as the new dropout: Scaling dense and self-slimmable transformers Year: (2023)
Ref_id:b11 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b12 Title: The PASCAL recognising textual entailment challenge Year: (2006)
Ref_id:b13 Title: Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models Year: (2024)
Ref_id:b14 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b15 Title: Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model Year: (2024)
Ref_id:b16 Title: Qlora: Efficient finetuning of quantized llms Year: (2023)
Ref_id:b17 Title: Automatically constructing a corpus of sentential paraphrases Year: (2005)
Ref_id:b18 Title: A review of sparse expert models in deep learning Year: (2022)
Ref_id:b19 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b20 Title: Mixture-of-loras: An efficient multitask tuning for large language models Year: (2024)
Ref_id:b21 Title: Mola: Moe lora with layer-wise expert allocation Year: (2025)
Ref_id:b22 Title: The third PASCAL recognizing textual entailment challenge Year: (2007)
Ref_id:b23 Title: The llama 3 herd of models Year: (2024)
Ref_id:b24 Title: Dynamic mixture of experts: An auto-tuning approach for efficient transformer models Year: (2024)
Ref_id:b25 Title: Vimoe: An empirical study of designing vision mixture-of-experts Year: (2024)
Ref_id:b26 Title: Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference Year: (2024)
Ref_id:b27 Title: Multi-task reinforcement learning with mixture of orthogonal experts Year: (2023)
Ref_id:b28 Title: Multi-task reinforcement learning with mixture of orthogonal experts Year: (2024)
Ref_id:b29 Title: Aligning ai with shared human values Year: ()
Ref_id:b30 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b31 Title: Harder task needs more experts: Dynamic routing in moe models Year: (2024)
Ref_id:b32 Title: Ders: Towards extremely efficient upcycled mixture-of-experts models Year: (2025)
Ref_id:b33 Title: Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference Year: (2024)
Ref_id:b34 Title: Mixture of nested experts: Adaptive processing of visual tokens Year: (2024)
Ref_id:b35 Title: Automoe: Heterogeneous mixture-of-experts with adaptive computation for efficient neural machine translation Year: ()
Ref_id:b36 Title:  Year: (2024)
Ref_id:b37 Title: Self-moe: Towards compositional large language models with self-specialized experts Year: (2024)
Ref_id:b38 Title: Gshard: Scaling giant models with conditional computation and automatic sharding Year: (2020)
Ref_id:b39 Title: The Winograd schema challenge Year: (2011)
Ref_id:b40 Title:  Year: ()
Ref_id:b41 Title: Locmoe: A low-overhead moe for large language model training Year: (2024)
Ref_id:b42 Title: Expert-token resonance: Redefining moe routing through affinity-driven active selection Year: (2024)
Ref_id:b43 Title: Ilya Sutskever, and Karl Cobbe. Let's verify step by step Year: (2023)
Ref_id:b44 Title: Moe-llava: Mixture of experts for large vision-language models Year: (2024)
Ref_id:b45 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b46 Title: Diversifying the mixture-of-experts representation for language models with orthogonal optimizer Year: (2023)
Ref_id:b47 Title: Muon is scalable for llm training Year: (2025)
Ref_id:b48 Title: Netmoe: Accelerating moe training through dynamic sample placement Year: (2025)
Ref_id:b49 Title: Not all experts are equal: Efficient expert pruning and skipping for mixture-ofexperts large language models Year: (2024)
Ref_id:b50 Title: Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models Year: (2024)
Ref_id:b51 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b52 Title: Open models based on gemini research and technology Year: (2024)
Ref_id:b53 Title: Multimodal contrastive learning with limoe: the language-image mixture of experts Year: (2022)
Ref_id:b54 Title: Load balancing mixture of experts with similarity preserving routers Year: (2025)
Ref_id:b55 Title: Dense training, sparse inference: Rethinking training of mixtureof-experts language models Year: (2024)
Ref_id:b56 Title: Efficiently scaling transformer inference Year: (2023)
Ref_id:b57 Title: Alphalora: Assigning lora experts based on layer training quality Year: (2024)
Ref_id:b58 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2024)
Ref_id:b59 Title: Mixture-of-experts meets instruction tuning: A winning combination for large language models Year: (2023)
Ref_id:b60 Title: Recursive deep models for semantic compositionality over a sentiment treebank Year: (2013)
Ref_id:b61 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2022)
Ref_id:b62 Title: Challenging big-bench tasks and whether chain-of-thought can solve them Year: (2022)
Ref_id:b63 Title: Hobbit: A mixed precision expert offloading system for fast moe inference Year: (2024)
Ref_id:b64 Title: Attention is all you need Year: (2017)
Ref_id:b65 Title: Glue: A multi-task benchmark and analysis platform for natural language understanding Year: (2018)
Ref_id:b66 Title: A comprehensive survey in llm (-agent) full stack safety: Data, training and deployment Year: (2025)
Ref_id:b67 Title: Auxiliary-loss-free load balancing strategy for mixture-of-experts Year: (2024)
Ref_id:b68 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2023)
Ref_id:b69 Title: Mmlu-pro: A more robust and challenging multi-task language understanding benchmark Year: (2024)
Ref_id:b70 Title: Neural network acceptability judgments Year: (2018)
Ref_id:b71 Title: Finetuned language models are zero-shot learners Year: (2022)
Ref_id:b72 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b73 Title: Symbol tuning improves in-context learning in language models Year: (2023)
Ref_id:b74 Title: Skywork-moe: A deep dive into training techniques for mixture-of-experts language models Year: (2024)
Ref_id:b75 Title: Livebench: A challenging, contamination-free LLM benchmark Year: (2025)
Ref_id:b76 Title: A broad-coverage challenge corpus for sentence understanding through inference Year: (2018)
Ref_id:b77 Title: Routing experts: Learning to route dynamic experts in existing multi-modal large language models Year: (2025)
Ref_id:b78 Title: Openmoe: An early effort on open mixture-of-experts language models Year: (2024)
Ref_id:b79 Title: Moral: Moe augmented lora for llms' lifelong learning Year: (2024)
Ref_id:b80 Title: Adamoe: Tokenadaptive routing with null experts for mixture-of-experts language models Year: (2024)
Ref_id:b81 Title: Mixture-of-experts with expert choice routing Year: (2022)
Ref_id:b82 Title: Llama-moe: Building mixture-of-experts from llama with continual pre-training Year: (2024)
Ref_id:b83 Title: Sparse mixture of low rank adaptation Year: (2023)
Ref_id:b84 Title: St-moe: Designing stable and transferable sparse expert models Year: (2022)
