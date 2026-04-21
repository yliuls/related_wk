Title: DISTILLM-2: A Contrastive Approach Boosts the Distillation of LLMs
Abstract: Despite the success of distillation in large language models (LLMs), most prior work applies identical loss functions to both teacher-and student-generated data. These strategies overlook the synergy between loss formulations and data types, leading to a suboptimal performance boost in student models. To address this, we propose DISTILLM-2, a contrastive approach that simultaneously increases the likelihood of teacher responses and decreases that of student responses by harnessing this synergy. Our extensive experiments show that DISTILLM-2 not only builds high-performing student models across a wide range of tasks, including instruction-following and code generation, but also supports diverse applications, such as preference alignment and vision-language extensions. These findings highlight the potential of a contrastive approach to enhance the efficacy of LLM distillation by effectively aligning teacher and student models across varied data types.

Section: Introduction
Large language models (LLMs) have continuously improved their text generation abilities by increasing the number of parameters and the amount of high-quality training data. However, LLMs typically require extensive computational resources during inference, which makes them difficult to be deployed practically. Therefore, compressing them by reducing the number of parameters while maintaining their performance becomes important for using these powerful models effectively.
As the demand for reducing computational overhead grows, knowledge distillation (KD; Hinton et al. 2015) has emerged as a promising technique for compressing LLMs into more lightweight student models. By transferring knowledge from a high-capacity teacher model to a smaller student model, KD can significantly improve the performance of small language models (sLMs) as demonstrated by Llama 3.2 (Meta, 2024) and Gemma-2 (DeepMind et al., 2024).
Over the years, research on LLM distillation has largely focused on either by designing new loss or by curating training data. From a loss perspective, several studies suggest that Kullback-Leibler (KL) divergence, a common loss for KD, may fail to capture the teacher model's complex generative behavior (Wen et al., 2023;Gu et al., 2024). Consequently, alternative loss functions, such as skew KL (SKL; Ko et al. 2024), have been proposed to better guide the student. On the other hand, from data perspective, previous works emphasize how the training data is curated to enlarge the effectiveness of KD. For instance, relying solely on offline data (e.g., teacher-generated outputs; TGOs) can be problematic where student's outputs at inference time deviate significantly from fixed training samples (Agarwal et al., 2024). To address this mismatch, some works incorporate student-generated outputs (SGOs) directly into training (Lin et al., 2020;Xu et al., 2024b). However, these works often overlook the synergy between loss formulations and data types, which might have limited the extent of performance improvement of student models.
Recently, contrastive approaches such as direct preference optimization (DPO; Rafailov et al. 2023), have gained popularity for their efficacy and efficiency in preference alignment (Tajwar et al., 2024) or reasoning (Pang et al., 2024), by explicitly employing different learning strategies to handle two distinct responses. Despite their success, few works have focused on extending their schema to KD for LLMs. While Li et al. (2024b) attempted to simply apply DPO by replacing the reference model to teacher model (see Equation 4), we observed that their method is prone to reward hacking, which may limit its broader applicability (see Figure 1). This motivates us to design a scalable contrastive approach to boost LLM distillation.
Contributions. In this paper, we introduce DISTILLM-2, which features a novel contrastive approach for KD of LLMs. Our DISTILLM-2 builds a contrastive framework upon DistiLLM (Ko et al., 2024), which has shown significant improvements by using SKL-based loss and balanced SGOs. Our detailed contributions include:
• Contrastive approach with asymmetric loss dynamics:
We analyze the behavior of forward and reverse KL (and SKL) during the training process on responses from the student and teacher models, respectively. This analysis motivated the development of a contrastive approach for LLM distillation (CALD; §3.1), which applies distinct loss functions to different types of training samples. By doing so, CALD effectively incorporates the synergy between loss formulations and data perspectives.
• Development of the contrastive approach: Additionally, we introduce optimized dataset curation strategies ( §3.2) and curriculum-based adaptive loss mechanisms ( §3.3). These enhancements to CALD, which are collectively coined to as DISTILLM-2, provide solid guidelines for our contrastive approach for practitioners.
• Advanced performance and versatility: DISTILLM-2 achieves state-of-the-art performance for sLMs across various text-generation tasks, including instructionfollowing, mathematical reasoning, and code generation ( §4). Furthermore, we demonstrate the diverse applications of our proposed KD approach ( §6), such as preference alignment with better reference models and its expansion to vision-language models.
this section cite: ['b22', 'b14', 'b63', 'b19', 'b30', 'b1', 'b38', 'b52', 'b58', 'b50', 'b30']

Section: Backgrounds

this section cite: []

Section: Related Work
KD (Hinton et al., 2015) effectively compresses neural networks, enabling smaller student models to match the performance of larger teachers. This technique recently has been adapted to address the scalability challenges of LLMs, enhancing their viability in compute-intensive environments. ImitKD (Lin et al., 2020) demonstrated the use of SGO as training data for distillation. Building on this, Agarwal et al. (2024) introduced an on-policy approach with objectives like reverse KL or Jensen-Shannon divergence (JSD). Wen et al. (2023) explored various f-divergences, including total variation distance and JSD, in auto-regressive LMs, while Gu et al. (2024) proposed a policy gradient method to mitigate high variance in RL-based techniques. Recently, Xu et al. (2024b) combined static datasets with on-policy methods using speculative decoding for training data generation. Among these, DistiLLM (Ko et al., 2024) achieved state-of-the-art performance and greater efficiency by introducing SKL and an adaptive off-policy approach. A more discussion of related works is available in the Appendix A.
this section cite: ['b22', 'b38']

Section: Preliminary: KD in LLMs and DistiLLM
Loss function of KD in LLMs. Given a prompt and response pair, denoted as (x, y), KD minimizes divergence between the distributions of a teacher p(y|x) and a student q θ (y|x) parameterized by θ. Conventionally, KL, denoted as D KL , is the most widely used loss in KD due to its simplicity and tractability. The sequence-level distillation using KL is accurately decomposed into a sum of token-wise distillation (Ko et al., 2024):
D KL (x, y; p∥q θ ) = T t=1 p(y t |y <t , x) log p(y t |y <t , x) q θ (y t |y <t , x) .
(1) We can also define reverse KL as D RKL (x, y; p∥q θ ) = D KL (x, y; q θ ∥p). Despite its tractability, such KL has limitations of either mode-averaging or mode-collapsing for forward and reverse version, respectively. To address this issue, Ko et al. (2024) proposed skew KL (SKL) and skew RKL (SRKL), defined as follows:
D (α) SKL (x, y; p∥q θ ) = D KL (x, y; p∥αp + (1 -α)q θ ), D (α) SRKL (x, y; p∥q θ ) = D KL (x, y; q θ ∥(1 -α)p + αq θ ).
Despite the simple modification, SKL demonstrated higher convergence speed and achieved better performance compared to recent baselines, such as MiniLLM (Gu et al., 2024) and GKD (Agarwal et al., 2024). This effectiveness has been proven from both empirical and theoretical perspectives. For brevity, we will denote D
this section cite: ['b30', 'b30', 'b19', 'b1']

Section: Data curation of KD in LLMs.
To address the training inefficiency and low quality of SGO, which can lead to inaccurate teacher feedback in on-policy approaches (Lin et al., 2020;Agarwal et al., 2024), Ko et al. (2024) introduced an adaptive off-policy approach, which bridges offline and purely on-policy setups, striking a balance between the efficiency and efficacy of KD. This balanced strategy reuses SGO by introducing replay buffer, significantly improving computational efficiency while preserving the effectiveness of on-policy distillation. This approach has proven effective in subsequent works on preference alignment of LLMs (Rosset et al., 2024) as in more generalized version.
this section cite: ['b38', 'b1', 'b30', 'b53']

Section: Summary & Connection to our work.
Building on the insights from DistiLLM (Ko et al., 2024) -where SKL (or SRKL) and adaptive off-policy have shown efficacy -we introduce a contrastive approach that further refines these objectives. On the data curation side, we adopt a batch approach (Rosset et al., 2024) that collects SGO ahead of every training epoch in our setup, rather than on-policy approach, which samples at every training iteration. This also ensures compatibility with advanced LLM inference techniques, such as vLLM (Kwon et al., 2023), thereby increasing generation efficiency and preserving the core philosophy of the adaptive off-policy approach. As shown in our preliminary results in Appendix D.1, this greatly reduces the computational cost of gathering training samples with minimal impact on student performance.
this section cite: ['b30', 'b53', 'b32']

Section: Method: DISTILLM-2
We introduce DISTILLM-2, a novel approach to LLM dis-
tillation, which lies in its new loss function as presented in Equation 2. This equips with a contrastive schema simultaneously accounting for different types of training responses ( §3.1), along with dedicated data curation ( §3.2) and curriculum-based adaptive loss design ( §3.3). L DISTILLM-2 := (2) 1 2|D| (x,yt,ys)∼D (1 -β)D (αt) SKL (x, y t ) + βD (αs) SRKL (x, y s ) , where D (αt) SKL (x, y t ) and D (αs)
SRKL (x, y s ) are SKL and SRKL tailored for teacher-and student-generated responses, respectively; D is the training dataset; β is a coefficient in [0, 1] to balance SKL and SRKL terms. In the following subsections, we provide detailed motivations, derivations, and use of the loss function in Equation 2 to formulate our DISTILLM-2 training process as stated in Algorithm 1.
this section cite: []

Section: Contrastive Approach

this section cite: []

Section: MOTIVATION

this section cite: []

Section: Concept.
Recently, contrastive approach in preference alignment, including DPO (Rafailov et al., 2023), which increases the likelihood of the preferred response (y w ) while decreasing the likelihood of the dis-preferred response (y l ), has demonstrated effective in enhancing LM performance.
-log σ λ log q θ (y w |x) q ref (y w |x) increase q θ (yw|x) -λ log q θ (y l |x) q ref (y l |x) decrease q θ (y l |x) ,(3)
where σ is sigmoid function, q ref is a reference model, and λ is hyperparameter for DPO. This improvement stems from its dual mechanism: not only does it reduce the likelihood of undesired responses (Tajwar et al., 2024) but it also increases the likelihood of preferred responses, effectively reinforcing alignment with the desired behavior.
Similarly, we can apply this concept into KD to increase the likelihood of q θ (y t |x) as match that of p(y t |x) and decrease the likelihood of q θ (y s |x) as match that of p(y s |x) by bringing different types of loss function for each type of response. This approach allows better alignment of TGOs and SGOs in the contrastive manner than simply using a single type of loss function.
Challenges of contrastive approach into KD. While the concept itself is appealing, there are critical issues in di-
Algorithm 1 Training pipeline of DISTILLM-2 1: Input: training iterations T , initial skew coefficient α 0 , teacher p, student q θ0 with parameter θ 0 , prompt set 2: Output: Student model q θ E with trained parameters θ E 3: for epoch e = 1, 2, . . . , E do 4: /* Sample batched on-policy responses */ 5: Sample responses y t , y s from teacher p(•|x) and student q θe-1 (•|x) for given prompt x 6:
Construct D t = {(x, y t , y s )} for training dataset for training epoch e.
this section cite: ['b52', 'b58']

Section: 7:
Initialize θ e ← θ e-1 8:
for iteration τ = 1, 2, . . . , T do 9: Sample mini-batch: B = {(x (i) , y (i) t , y (i) s )} |B| i=1 from D t 10: /* Curriculum-based adaptive update for α */ 11: Update α t ← 1 -(1 -α 0 ) • m p(ys|x)-q θ (ys|x) and α s ← 1 -(1 -α 0 ) • m p(yt|x)-q θ (yt|x)
12: /* Gradual increasing coefficient for SRKL */ 13: Update β ← clip( e E + τ T , β 0 , 1) 14: /* Improved contrastive loss function ( §3.3)*/ 15: Update θ e by minimizing L DISTILLM-2 = 1 2B (1 -β)D (αt) SKL (x, y t ) + βD (αs) SRKL (x, y s ) 16:
end for 17: end for rectly applying DPO into KD. We observed that DPKD (Li et al., 2024b), which simply applies DPO by substituting the reference model with the teacher model, frequently suffers from reward hacking, leading to degenerate sentences:
-log σ λ log q θ (y t |x) p(y t |x) -λ log q θ (y s |x) p(y s |x) inherently small p(ys|x) → overly decrease q θ (ys|x) ,(4)
where y t and y s are TGO and SGO, respectively. This is because DPKD only focuses on maximizing the gap between q θ (yt|x) p(yt|x) and q θ (ys|x) p(ys|x) . As illustrated in Figure 1(b), we observe that this loss dynamics excessively decreases the likelihood of q θ (y s |x) (e.g., 91.25 in terms of negative log-likelihood; NLL), causing the student model to lose pre-trained information instead of fitting to teacher responses (e.g., 20.29 in terms of NLL), as it replaces q ref with p where p(y s |x) is inherently small. Addressing this limitation requires rethinking and redesigning algorithm to integrate contrastive strategies into LLM distillation.
this section cite: []

Section: CONTRASTIVE APPROACH FOR LLM DISTILLATION
To bring contrastive strategy into KD, we propose a new loss function L CALD , using a combination of SKL and SRKL (Ko et al., 2024). Our design stems from the follows.
Observation on behavior of KL and RKL. Here, we pro-
0 20 40 60 80 x 0.00 0.02 0.04 0.06 p(x) or q (x) (a) Toy Dataset Results
p(x) q (x) (KL) q (x) (RKL) KL RKL SKL SRKL CALD (SKL+SRKL) DPKD 0 1 2 3 4 5 logq (x) (NLL) 2.50 2.90 2.69 2.79 2.18 20.29 1.61 1.71 1.54 1.75 2.06 91.25 (b) LLM Experimental Results teacher student 5 10 15 20 Val Iteration vide an observation on the behavior of KL and RKL: they can increase and decrease the likelihood of q θ for TGOs (KL) and SGOs (RKL), respectively. As shown in Figure (•|x) q θ (•|x) for the region where p(•|x) are large to minimize weighted average. Conversely, RKL attempts to reduce the ratio q θ (•|x) p(•|x) . Consequently, q θ (•|x) decreases in region where p(•|x) are small (i.e., pushingdown effect), such as the tail of teacher distribution in Figure 1(a) or student responses in Figure 1(b). Detailed mathematical explanation can be found in Appendix B.1.
Our solution. For implementing CALD, an optimal choice among various KL-based loss functions would be one that demonstrates state-of-the-art results while exhibiting similar behavior to KL and RKL, as observed in Figure 1. To this end, we utilize skew KL (SKL) and RKL (SRKL), introduced in DistiLLM (Ko et al., 2024), as the backbone loss functions. Specifically, we design the loss function for CALD, using SKL for teacher responses (i.e., y t ) where most of p(y t |x) ≫ 0 and using SRKL for student responses, y s , where the most of p(y s |x) ≃ 0. Formally, our proposed loss function can be written as follows:
L CALD = 1 2|D| (x,yt,ys)∼D D (α) SKL (x, y t ) + D (α) SRKL (x, y s ).
(5) Despite its simplicity, this loss function implies that the importance of simultaneous consideration of responses type during objective function design. Note that Ko et al. (2024) demonstrated that a vanilla interpolation between γD
(α) SKL (x, •) + (1 -γ)D (α) SRKL (x,
•) for all γ ∈ [0, 1] over the same type of responses, (e.g., either y t or y s ), does not improve performance compared to using either SKL or SRKL alone. However, we find that the new approach of using different types of responses for different terms significantly enhances performance. L CALD achieves faster convergence and greater effectiveness compared to the ex-clusive use of SKL or SRKL in DistiLLM (see Figure 1(c)). Note that while simple KL and RKL also prove effectiveness for CALD, using SKL and SRKL as backbone achieves higher efficacy, consistent with Ko et al. (2024).
Mathematical connection to DPKD and DPO. We now reveal that our proposed loss function L CALD can be mathematically interpreted as exhibiting similar yet different behavior to DPKD (or DPO).
Remark 1. Equation 5 can be re-written as follows:
-E yt∼p(•|x), ys∼q θ (•|x) 1 λ • λ log qθ (y t |x) p(y t |x) -λ log q θ (y s |x) p(y s |x) ,(6)
where qθ (•|x) = αp(•|x) + (1 -α)q θ (•|x) and p(•|x) = αq θ (•|x) + (1 -α)p(•|x).
This indicates CALD enable to increase qθ (y t |x) (and implicitly q θ (y t |x)) and decrease q θ (y s |x), simultaneously. The detailed derivation can be found in Appendix B.2. Despite this similarity, there are two critical and non-trivial differences between CALD and DPKD (or DPO). First, rather than employing the log-sigmoid function used in DPKD, Equation 6 adopts a linear formulation that allows token-level decomposition and explicit weighting by p(y t |x) or q θ (y s |x) (as in Equation 1). Second, by inherently linear dependency between qθ (•|x) and p(•|x) (or between p(•|x) and q θ (•|x)), this regularizes the overly decreasing q θ (y s |x), which resolves the challenges in DPKD. From this, CALD (i.e., DISTILLM-2) outperforms DPO and DPKD by a large margin, as shown in Appendix D.1.
this section cite: ['b30', 'b30', 'b30', 'b30']

Section: Optimal Data Curation for Contrastive Approach
In the context of datasets for LLM distillation, one common question might be:
"How can we effectively utilize well given SGO and high-quality fixed datasets in distillation of LLMs?" While previous works (Xu et al., 2024b;Li et al., 2024a) have proposed effective strategies for leveraging these two complementary dataset types in an SFT manner, we observed that their techniques -such as speculative generation or ys (green) with yspec, responses from speculative decoding (Cai et al., 2024), varying the hyperparameter ε. (b) replacing yt with responses generated using stronger LLMs (e.g., Llama-3, Gemma-2, Phi-3) than the teacher models (i.e., Mistral) for the SKL term. We also show the negative log-likelihood (NLL) of the student (cyan) and teacher (blue) models on the replaced responses, along with the corresponding WR (red). or the use of high-quality responses (which may outperform teacher generations) -are less effective in CALD. From our further discussion, we conclude that utilizing teacher and student generations for SKL and SRKL, respectively, may be the optimal strategy for CALD, as it consistently aligns with the core philosophy of CALD.
Exploring the trade-offs between teacher and student generations. Previous works (Agarwal et al., 2024) have discussed that while teacher responses provide useful information, they can cause training-inference mismatches. In contrast, SGOs, though lower in quality, effectively reduce such mismatches, leading to higher efficacy. To explore these complementary perspectives, we use speculative decoding 12 to find the key factors for dataset curation.
In speculative generation, student drafts K tokens, and teacher verify them in parallel for 1 ≤ k ≤ K based onfoot_2 :
q θ (y n+k |y <n+k ) > min(ε 2 , ε • exp(-H(p(•|y <n+k )))),
where H(•) and ε are entropy function and hyperparameter.
When we replace y t with y spec , as ε decreases (i.e., more acceptance of drafts), the distilled model better aligns with the student distribution q θ (•|x). However, as shown with orange bars, its performance is highest with y spec at ε = 1.0 (i.e., identical to y t ). This implies that on the SKL side, mitigating the training-inference mismatch via SGOs does not always lead to performance improvement. Rather, strong guidance from the teacher response is highly related to the distillation performance. Conversely, on the responses for SRKL, the distilled model achieves the highest performance with y spec at ε = 0.0 (i.e., identical to y s ), as shown with green bars, although these responses are of the lowest quality. This implies that using low-quality SGO samples on the SRKL side may be beneficial for our contrastive approach. The effectiveness of reduced training-inference mismatch via SGOs can be attributed to this edge of alignments.
High-quality does not always guarantee success. One additional question that arises is whether the success of teacher responses on the SKL term is due to their higher quality. It is natural to consider if using higher-quality responses from powerful LLMs like ChatGPT would improve performance, similar to black-box KD (Li et al., 2024a). To investigate, we replaced the responses for SKL term with those generated from stronger LLMs (e.g., Llama3-8B) instead of the Mistral-7B teacher's responses. As shown in Figure 2(b), although these stronger LLMs generate high-quality answers, the student trained on the teacher's responses still performs
Table 2. Comparison winning rates (WR) using pairwise comparison (Zheng et al., 2023) on three instruction-following benchmarks. The baseline is text-davinci-003 in AlpacaEval and gpt-3.5-turbo in Evol-Instruct and UltraFeedback. The judges are GPT-4o for AlpacaEval and Evol-Instruct, GPT-4o-mini for UltraFeedback. The best and the second best win rates are in bold and underline.
Qwen2-7B-Inst (M T ) → Qwen2-1.5B (M S ) Mistral-7B-Inst (M T ) → Danube2-1.8B (M S ) Gemma-2-9B-Inst (M T ) → Gemma-2-2B (M S ) Method AlpacaEval
Evol-Inst UltraFeed AVG. AlpacaEval Evol-Inst UltraFeed AVG. AlpacaEval Evol-Inst UltraFeed AVG. WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) WR(%) M T 88.41 70.70 69.25 76.12 91.92 73.51 83.59 83.01 95.78 88.76 85.90 90.15 M S 51.06 18.00 21.93 30.33 48.17 12.84 20.06 27.02 42.51 16.74 26.60 28.62 KD 57.49 28.23 37.86 41.19 60.21 18.23 41.56 40.00 61.78 32.45 54.37 49.53 SeqKD 58.02 29.11 38.35 41.83 59.76 18.45 42.11 40.11 62.43 33.21 55.18 50.27 ImitKD 59.37 30.58 39.92 43.29 58.34 17.89 40.87 39.03 63.12 31.89 53.92 49.64 GKD 66.07 44.61 57.74 56.14 69.75 24.54 57.74 50.68 81.43 50.57 77.20 69.73 DistiLLM 66.30 44.61 58.18 56.35 70.16 28.78 58.18 52.37 82.95 51.26 76.68 70.30 Speculative KD 61.52 44.95 56.82 54.43 64.58 38.87 60.04 54.50 78.45 57.11 72.21 69.26 DISTILLM-2 69.88 47.13 59.05 58.69 74.04 32.84 62.46 56.45 85.97 59.53 78.99 74.83
better. This suggests that the high log-probability of responses from the teacher model may be a more important factor in data curation than their higher quality.
Discussion on the observations. These findings align with the motivation of CALD in §3.1: the "pulling-up" effect of SKL is maximized at the head of p(•|x) (i.e., y t ), while the "pushing-down" effect of SRKL is maximized at the tail of p(•|x) (i.e., y s ). First, while speculative generations are effective with vanilla KL in Speculative KD (Xu et al., 2024b), they are less effective with our contrastive loss because (1) speculative generations are an interpolation of y t and y s , which may weaken both the "pulling-up" and "pushing-down" effect -core mechanisms underlying CALD; and (2) the contrastive loss already exploits both complementary response types simultaneously, reducing the need for interpolation compared to single-loss settings.
The second observation also supports our claim that pure teacher generation may be optimal for SKL where they completely align with p(•|y), rather than relying on higherquality responses, from the perspective of maximizing the "pulling-up" effect at the head of p(•|x).
this section cite: ['b6', 'b1', 'b72']

Section: Curriculum-based Adaptive Learning
We introduce two modifications, inspired by our empirical observations, to implement difficulty-based adaptive learning and facilitate the conversion from Equation 6to Equation 2: a curriculum approach for α and a gradual increasing of coefficient for SRKL.
Curriculum Approach for α. One limitation of SKL (Ko et al., 2024) is that we need to manually determine α, which interpolates between the teacher and student distributions.
A larger α improves optimization stability and accelerates convergence, but it limits the acquisition of informative knowledge by inherently small gap between p(•) and αp(•)+ (1 -α)q θ (•). Conversely, a smaller α allows for greater knowledge acquisition but reduces optimization stability and slows convergence (Ko et al., 2024). While previous work suggests that α values in a moderate range (e.g., 0.1-0.3) are generally robust, we observed that the optimal values can still vary across different setups due to the variation of teacher-student pairs and the dynamic requirements of different training epochs (see Table 1).
Regarding the dynamic of different training epoch, we observe that the optimal values for α for the second or third epoch are either equal to or smaller than than those in the first epoch (Table 1). Building on this observation, we propose a curriculum-based approach for updating α. For "easy" samples, where p(•) and q θ (•) are sufficiently similar, we select a small α. On the other hand, for "hard" samples, where the difference between p(•) and q θ (•) is large, we choose a larger α.
To implement this, we introduce an updating rule for α ∈ [0, 1] based on the following approximation:
log p(y|x) q(α) θ (y|x) ≃ (1 -α) • (p(y|x) -q θ (y|x)) , (7
)
where q(α) θ (y|x) = αp(y|x) + (1 -α)q θ (y|x). Note that this approximation originates from the Mercator series expansion (Zwillinger, 2002)
: log(1 + x) = ∞ n=1 (-1) n+1 • x n
n . This series allows the first-order approximation log p(x) ≃ p(x) -1. The detailed derivation can be found in Appendix. Using this formula, we can compute a suitable α in closed-form for each sample, allocating proper α by making (1 -α) • (p(•) -q θ (•)) consistent across entire training. The detailed implementation for this updating rule can be found in Algorithm 1.
(Linearly) Gradual increasing of coefficient for SRKL. Based on the behavior of TGOs with SKL and SGOs with SRKL in Equation 5, the first term enables the acquisition of advanced information by matching high-probability on TGOs, while the second term suppresses undesirable behavior by preventing the matching of similarly low probabilities in SGOs. However, achieving q θ (•|x) = p(•|x) for all y t and y s is challenging with limited dataset sizes due to inher- ent capacity gap between the teacher and student models, which arises from factors such as the number of parameters. Nevertheless, we observe that gradually increasing the SRKL coefficient β in Equation 2, following the linear schedule shown in Algorithm 1, significantly improves student performance (see Table 5). This improvement is achieved by compromising the imitation of the teacher's behavior, which is relatively hard to achieve, while focusing on directly obtaining feedback from SGOs, thereby effectively reducing the training-inference mismatch.
this section cite: ['b30', 'b30', 'b75']

Section: Experiments

this section cite: []

Section: General Instruction-Following
Setup. We first construct the training datasets, by randomly sampling 50k prompts from UltraChat200k (Ding et al., 2023) and use the corresponding teacher and student to generate the responses. After training, we evaluate DISTILLM-2 for general purpose instruction-following task on AlpacaEval (Li et al., 2023), Evol-Instruct (Xu et al., 2024a), and UltraFeedback (Cui et al., 2024). For evaluation, we adopt LLM-as-a-Judge (Zheng et al., 2023) with GPT-4o or GPT-4o-mini as judge models. For experiments, we use Qwen2-7B (Hui et al., 2024), Mistral-7B (Jiang et al., 2023), Gemma-2-9B (DeepMind et al., 2024) instruction models as teachers and Qwen2-1.5B, Danube2-1.8B (Singer et al., 2024), and Gemma-2-2B as students, respectively.
Results. We report experimental results in Table 2. This comparison between DISTILLM-2 and other baselines shows that our proposed method performs best in the most of evaluation setups, except for Danube2-1.8B in Evol- Instruct. It outperforms the second best methods by +2.34%, +1.95%, and +4.53% on average for Qwen2-1.5B, Danube2-1.8B, and Gemma2-2B, respectively. As these evaluation benchmarks cover a wide range of domains relevant to real-world applications, these results demonstrate that DISTILLM-2 can be widely used to build strong sLMs.
this section cite: ['b15', 'b12', 'b72', 'b24', 'b25', 'b55']

Section: Mathematical Reasoning
Setup. We conduct experiments on two standard mathematical reasoning benchmarks: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021). For teacher and student pairs, we select the Qwen2-Math-7B-Inst and Qwen2.5-Math-7B-Inst as teacher models and Qwen2-Math-1.5B and Qwen2.5-Math-1.5B as student models, respectively. The student models are trained using 50k randomly selected samples from the MetaMathQA (Yu et al., 2024a) dataset. Specifically, the student models are fine-tuned in a supervised manner on the entire MetaMathQA for a single epoch.
this section cite: ['b10', 'b21']

Section: Results.
Table 3 summarizes the effectiveness of DISTILLM-2 compared to recent competitive baselines, including GKD and DistiLLM. In both the Qwen2 and Qwen2.5 experimental setups, DISTILLM-2 achieves higher performance than other baselines on the GSM8K and MATH evaluations. Interestingly, the Qwen2.5 student with DISTILLM-2 demonstrates competitive average performance and even outperforms the Qwen2 teacher on the MATH evaluation, which is a challenging milestone.
this section cite: []

Section: Code Generation
Setup. We utilize prompts from WizardCoder (Luo et al., 2024) dataset which is developed using the Evol-Instruct method (Xu et al., 2024a) code instruction datasets. We apply Qwen2.5-Coder-7B-Inst (Hui et al., 2024) and Table 7. Evaluation on OK-VQA and TextVQA, two popular benchmark for visual question answering. We utilized VQA accuracy (Antol et al., 2015). The best results are highlighted in bold.
VQA Acc. (%) M T M S GKD DistiLLM DISTILLM-2 OK-VQA 54.70 36.87 41.83 39.38 44.72 TextVQA 42.91 28.34 33.84 31.10 34.98 AVG. 48.81 32.61 37.84 35.24 39.85
DeepSeek-Coder-6.7B-Inst as teacher models and Qwen2.5-Coder-1.5B and DeepSeek-Coder-1.3B as student models, respectively. Similarly, we train the student models for 2 epochs. We evaluate performance on two standard coding benchmarks: HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021).
this section cite: ['b43', 'b24', 'b2', 'b8', 'b3']

Section: Results.
The results are presented in Table 4. Across both HumanEval and MBPP, DISTILLM-2 consistently outperforms the baseline methods, GKD and DistiLLM. Notably, GKD achieves higher scores than DistiLLM, its effectiveness remains lower than that of DISTILLM-2. This outcome highlights DISTILLM-2 's ability to integrate a specialized alignment strategy -effectively incorporating SKL (or SRKL) with harmonized response type.
this section cite: []

Section: Additional Ablation Study
Here, we provide additional ablation experiments on DISTILLM-2. Our ablation studies are conducted with Qwen2 (or Qwen2.5) from its diversified model sizes. We use GPT-4o-mini as a judge model for all ablation studies due to its cost-efficiency.
this section cite: []

Section: Component Analysis.
Here, we conducted a component analysis of DISTILLM-2 's technical components, which included (1) applying a contrastive approach ( §3.1 & §3.2), (2) increasing the β parameter, and (3) introducing curriculumbased updates to α ( §3.3). Table 5 shows a componentwise analysis demonstrating how progressively incorporating these improvements into DistiLLM brings its performance in line with that of DISTILLM-2. As each component is added, we observe incremental performance gains, indicating that all of the examined components enhance DISTILLM-2 's overall effectiveness.
Training Size. We investigated how varying the training data size affects the performance of DISTILLM-2. In Figure 3, we show its performance across various tasks-such as instruction-following, math reasoning, and code generation-and compare it against baselines including GKD and DistiLLM. We observe that our proposed method consistently outperforms these baselines, demonstrating the highest effectiveness among all considered LLM distillation methods.
Capacity Gap. It is well known that a substantial capac- ity gap between large teacher models and compact student models makes KD more challenging, a phenomenon referred to as the capacity gap (Mirzadeh et al., 2020). We experimented with diverse size of Qwen-1.5-Chat with 1.8B, 7B, and 14B parameters as teacher and SFT of Qwen1.5-0.5B student to seize the behavior of DISTILLM-2 across the different size of teacher models. As shown in Table 8, DISTILLM-2 demonstrates monotonic improvement and consistently outperforms other baselines as the teacher size increases. This result highlights DISTILLM-2's effectiveness in addressing capacity gap issues, whereas the previous version (Ko et al., 2024) struggled with capacity gaps, particularly with the 7B and 14B teacher models.
this section cite: ['b48', 'b30']

Section: Broader Impacts
Furthermore, we present a range of diverse applications for DISTILLM-2, demonstrating its broad versatility and highlighting its potential for future use. We also provide additional applications of DISTILLM-2 in Appendix 6.3 (i.e., recovering quantized model) and 6.4 (i.e., achieving higher inference speed in speculative decoding).
this section cite: []

Section: Additional Results for DISTILLM-2 + DPO
In preference alignment (Ouyang et al., 2022;Rafailov et al., 2023), training usually involves two steps: (1) SFT and (2) preference fine-tuning using either PPO or DPO. While most previous works have concentrated on the second step, we highlight that the first step is also important. In our study, we replace the standard SFT method with KD of LLMs and evaluate its effectiveness by comparing how the policy LLMs perform after the second step, which uses DPO. Specifically, we use the reference model for each setup as trained student in Table 2. The results in Table 6 show that replacing SFT with distillation in the first phase leads to  higher overall alignment performance in preference finetuning. Notably, our DISTILLM-2 achieved a substantially higher WR, more than doubling the WR in Qwen2-1.5B, and showed a similar improvement in Gemma2-2B. These indicate that DISTILLM-2 can build effective reference models for the subsequent preference alignment phase.
this section cite: ['b49', 'b52']

Section: Expansion to Vision-Language Models
We also applied DISTILLM-2 on the distillation setup of vision-language models (VLMs) to boast the versatility of proposed method that can be applied in a wide range of modalities. We select LLaVA-1.5-7B (Liu et al., 2024) and TinyLLaVA-1.4B (Zhou et al., 2024a) as a teacher and a student model, respectively. For training dataset, we utilize the prompt from RLAIF-V-Dataset (Yu et al., 2024b) which contains 83K prompts, and evaluate the trained models on two popular benchmark, OK-VQA (Marino et al., 2019) and TextVQA (Singh et al., 2019). Table 7 shows that the superiority of DISTILLM-2 over other distillation methods holds true not only in LLM setups but also with VLMs. Although other baselines also demonstrated effectiveness compared to original student models (i.e., M S ), DISTILLM-2 outperformed GKD and DistiLLM by +2.01% and +4.61% on average, respectively.
this section cite: ['b41', 'b44', 'b56']

Section: Restoring the Performance of Quantized LLMs
Using parameter-efficient fine-tuning methods, such as LoRA, can help recover the performance of quantized LLMs after post-training quantization (Frantar et al., 2023), introducing only a negligible number of additional parameters.
Here, we demonstrate the effectiveness of DISTILLM-2 in restoring the performance of 4-bit quantized LLMs using LoRA by replacing regular SFT with KD baselines. Figure 9 shows that all KD methods can significantly improve the performance of quantized models while adding only a few trainable parameters. Additionally, DISTILLM-2 achieves the best average performance among KD baselines. Distillation can also be straightforwardly applied to pairs of original and compressed models, such as pruned or quantized versions, enabling efficient deployment on mobile devices.
this section cite: ['b18']

Section: Inference Speedup of Speculative Decoding
DistillSpec (Zhou et al., 2024b) demonstrate that KD can improve speculative decoding by better aligning the drafter and verifier models. Building on their work, we evaluate the inference speedup of speculative decoding using Phi3.5mini and Phi3-medium (Abdin et al., 2024) as verifiers and Llama-68m (Miao et al., 2024) as the drafter trained with various KD. Table 10 summarizes that the inference speedup of drafter with DISTILLM-2 surpasses other drafter models, including trained with SFT and DistiLLM for both Phi3.5-mini and Phi3-medium verifiers. These results indicate that the DISTILLM-2 enables higher token-level alignment of distribution compared to other LLM distillation baselines, including Zhou et al. (2024b).
this section cite: ['b0', 'b47']

Section: Conclusion
In this work, we introduce DISTILLM-2, a novel distillation framework for large language models that combines contrastive loss, curated data, and curriculum-based learning. By differentiating teacher and student outputs, our method overcomes key limitations of traditional distillation and achieves stronger alignment and generalization. Extensive experiments across instruction following, mathematical reasoning, and code generation show that DISTILLM-2 delivers state-of-the-art performance with improved sample efficiency, reducing reliance on expensive preference-labeled data. Leveraging high-quality explanations and contrastive objectives further enhances reasoning ability and robustness in both language and vision-language models. We believe DISTILLM-2 lays the groundwork for future advances in efficient model alignment and multi-modal learning, enabling more accessible and capable AI systems.
this section cite: []

Section: References
Ref_id:b0 Title: Phi-3 technical report: A highly capable language model locally on your phone Year: (2024)
Ref_id:b1 Title: On-policy distillation of language models: Learning from self-generated mistakes Year: (2024)
Ref_id:b2 Title: Vqa: Visual question answering Year: (2015)
Ref_id:b3 Title: Program synthesis with large language models Year: (2021)
Ref_id:b4 Title: A general theoretical paradigm to understand learning from human preferences Year: (2024)
Ref_id:b5 Title: Training a helpful and harmless assistant with reinforcement learning from human feedback Year: (2022)
Ref_id:b6 Title: Simple LLM inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b7 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b8 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b9 Title: Deep reinforcement learning from human preferences. Advances in neural information processing systems Year: (2017)
Ref_id:b10 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b11 Title: Free dolly: Introducing the world's first truly open instruction-tuned llm Year: (2023)
Ref_id:b12 Title: Ultrafeedback: Boosting language models with scaled ai feedback Year: (2024)
Ref_id:b13 Title: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b14 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b15 Title: Enhancing chat language models by scaling high-quality instructional conversations Year: (2023-12)
Ref_id:b16 Title: URL Year: ()
Ref_id:b17 Title: Alpacafarm: A simulation framework for methods that learn from human feedback Year: (2024)
Ref_id:b18 Title: OPTQ: Accurate quantization for generative pre-trained transformers Year: (2023)
Ref_id:b19 Title: Knowledge distillation of large language models Year: (2024)
Ref_id:b20 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b21 Title: Measuring mathematical problem solving with the MATH dataset Year: (2021)
Ref_id:b22 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b23 Title: Low-rank adaptation of large language models Year: (2022)
Ref_id:b24 Title: Qwen2. 5-coder technical report Year: (2024)
Ref_id:b25 Title: Mistral 7b Year: (2023)
Ref_id:b26 Title: Revisiting intermediate layer distillation for compressing language models: An overfitting perspective Year: (2023-05)
Ref_id:b27 Title: URL Year: ()
Ref_id:b28 Title: A simple unified framework of structured pruning for accelerating encoderdecoder language models Year: (2023-12)
Ref_id:b29 Title: URL Year: ()
Ref_id:b30 Title: DISTILLM-2: A Contrastive Approach Boosts the Distillation of LLMs Year: (2024)
Ref_id:b31 Title: SeRA: Self-reviewing and alignment of LLMs using implicit reward margins Year: (2025)
Ref_id:b32 Title: Efficient memory management for large language model serving with pagedattention Year: (2023)
Ref_id:b33 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b34 Title: Selective reflection-tuning: Student-selected data recycling for LLM instruction-tuning Year: (2024-08)
Ref_id:b35 Title: BiLD: Bi-directional logits difference loss for large language model distillation Year: (2025-01)
Ref_id:b36 Title: Alpacaeval: An automatic evaluator of instruction-following models Year: ()
Ref_id:b37 Title: Direct preference knowledge distillation for large language models Year: (2024)
Ref_id:b38 Title: Autoregressive knowledge distillation through imitation learning Year: (2020-11)
Ref_id:b39 Title: URL Year: ()
Ref_id:b40 Title: ROUGE: A package for automatic evaluation of summaries Year: (2004-07)
Ref_id:b41 Title: Visual instruction tuning Year: (2024)
Ref_id:b42 Title: Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation Year: (2023)
Ref_id:b43 Title: Empowering code large language models with evol-instruct Year: (2024)
Ref_id:b44 Title: Ok-vqa: A visual question answering benchmark requiring external knowledge Year: (2019)
Ref_id:b45 Title: Simple preference optimization with a reference-free reward Year: (2024)
Ref_id:b46 Title: Llama 3.2: Revolutionizing edge AI and vision with open Year: (2024)
Ref_id:b47 Title: Accelerating large language model serving with tree-based speculative inference and verification Year: (2024)
Ref_id:b48 Title: Improved knowledge distillation via teacher assistant Year: (2020)
Ref_id:b49 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b50 Title: Iterative reasoning preference optimization Year: (2024)
Ref_id:b51 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b52 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b53 Title: Direct nash optimization: Teaching language models to self-improve with general preferences Year: (2024)
Ref_id:b54 Title: Omnidirectionally calibrated quantization for large language models Year: (2024)
Ref_id:b55 Title: b technical report Year: (2024-08-01)
Ref_id:b56 Title: Towards vqa models that can read Year: (2019)
Ref_id:b57 Title: Learning to summarize with human feedback Year: (2020)
Ref_id:b58 Title: Preference fine-tuning of LLMs should leverage suboptimal, on-policy data Year: (2024)
Ref_id:b59 Title: SALMONN: Towards generic hearing abilities for large language models Year: (2024)
Ref_id:b60 Title: Direct distillation of lm alignment Year: (2023)
Ref_id:b61 Title: Deep self-attention distillation for task-agnostic compression of pre-trained transformers Year: (2020)
Ref_id:b62 Title: Self-instruct: Aligning language models with self-generated instructions Year: (2023-07)
Ref_id:b63 Title: f-divergence minimization for sequence-level knowledge distillation Year: (2023-07)
Ref_id:b64 Title: Rethinking kullback-leibler divergence in knowledge distillation for large language models Year: (2024)
Ref_id:b65 Title: Rethinking Kullback-Leibler divergence in knowledge distillation for large language models Year: (2025-01)
Ref_id:b66 Title: Empowering large pre-trained language models to follow complex instructions Year: ()
Ref_id:b67 Title: Speculative knowledge distillation: Bridging the teacherstudent gap through interleaved sampling Year: (2024)
Ref_id:b68 Title: Bootstrap your own mathematical questions for large language models Year: ()
Ref_id:b69 Title: Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness Year: (2024)
Ref_id:b70 Title: Dualspace knowledge distillation for large language models Year: (2024-11)
Ref_id:b71 Title: Slic-hf: Sequence likelihood calibration with human feedback Year: (2023)
Ref_id:b72 Title: Judging llm-as-a-judge with mt-bench and chatbot arena Year: (2023)
Ref_id:b73 Title: A framework of small-scale large multimodal models Year: (2024)
Ref_id:b74 Title: Distillspec: Improving speculative decoding via knowledge distillation Year: ()
Ref_id:b75 Title: CRC standard mathematical tables and formulae Year: (2002)
