Title: Cost-Aware Contrastive Routing for LLMs
Abstract: We study cost-aware routing for large language models across diverse and dynamic pools of models. Existing approaches often overlook prompt-specific context, rely on expensive model profiling, assume a fixed set of experts, or use inefficient trialand-error strategies. We introduce Cost-Spectrum Contrastive Routing (CSCR), a lightweight framework that maps both prompts and models into a shared embedding space to enable fast, cost-sensitive selection. CSCR uses compact, fast-to-compute logit footprints for open-source models and perplexity fingerprints for black-box APIs. A contrastive encoder is trained to favor the cheapest accurate expert within adaptive cost bands. At inference time, routing reduces to a single k-NN lookup via a FAISS index, requiring no retraining when the expert pool changes and enabling microsecond latency. Across multiple benchmarks, CSCR consistently outperforms baselines, improving the accuracy-cost tradeoff by up to 25%, while generalizing robustly to unseen LLMs and out-of-distribution prompts.

Section: Introduction
After a burst of reinforcement-learning and specialized finetuning, the Large Language Model (LLM) [65,66,6,86,17] ecosystem has fractured: code models excel at generating code but hallucinate outside programming contexts, math-tuned variants solve AIME [46] yet mishandle open-ended dialogue, and instruction chatbots trade being precise for fluency. Production systems therefore host a pool of models with different sizes, licenses, and domain strengths, and decide at run time which one to call for every user prompt, or worse: burden the users with picking the model they need.
A router (a.k.a. model selector, mixture-gate) adjudicates that choice online. It dynamically selects the most appropriate LLM from a pool for each input. Without it, users either over-pay by defaulting to the largest model or risk quality regressions by selecting cheaper ones. Current predictive routers [33,32,51,76] for a pool of LLMs fall into two broad camps: parametric routers and non-parametric ones.
Parametric routing methods, such as softmax-based classifiers [24], optimize exclusively for top-1 accuracy without explicit consideration of inference costs. Consequently, they tend to default to selecting expensive models and require full retraining whenever new models are introduced. Recent non-parametric approaches, such as UMR [40], enhance generalization by routing across a joint space of prompt clusters and model footprints. On the other hand, bandit-based methods [61,34,48], including Thompson sampling [85,2] and UCB [3], track simplified quality-cost metrics that disregard detailed prompt characteristics, resulting in slower convergence and reduced effectiveness, especially when facing heterogeneous and diverse input distributions. All these methods remain cost-agnostic during training, solely relying on post-hoc hyperparameter tuning to achieve an effective balance between cost and accuracy.
In our view, routing boils down to similarity search. If we can embed both prompts and experts in one metric space where cosine distance trades off {quality, cost}, routing reduces to a microsecond Nearest Neighbor query -no brittle softmax gate, no retraining when the pool changes.
In this paper we introduce:
• Universal ultra-compact descriptors: Two lightweight and fast to compute fingerprints that work across the full spectrum of LLMs. (1) Logit footprints that require only <10 forward passes through an open-weights model. (2) Perplexity fingerprints that score any black-box API with a small public LM, enabling vendor-agnostic routing.
• Cost-Spectrum InfoNCE: A novel contrastive objective that (1) selects correct positives within adaptive cost bands, (2) temperature-scales each band separately, and (3) downweights negatives in proportion to their cost. This aligns the learned metric with the accuracy-cost Pareto frontier.
• Routing Efficiency: A shared space turns routing into a single k-NN lookup, eliminating brittle softmax gates and retraining whenever the pool changes. Then a lightweight FAISS index makes routing a microsecond lookup.
• Comprehensive Evaluation: We evaluate our method on three routing benchmarks spanning both open-source checkpoints and proprietary APIs. It achieves up to 25% higher accuracy-cost efficiency on a fixed pool of LLMs and demonstrates strong robustness to unseen models and out-of-distribution prompts at inference time.
2 Related Work
this section cite: ['b64', 'b65', 'b5', 'b85', 'b16', 'b45', 'b32', 'b31', 'b50', 'b75', 'b23', 'b39', 'b60', 'b33', 'b47', 'b84', 'b1', 'b2', 'b0']

Section: LLM Routing
Non-Predictive Routing. Non-predictive methods generate outputs from one or more models before making a selection. FrugalGPT [10] uses a sequential strategy and a response quality threshold to minimize cost. Other works adopt layered inference architectures to escalate hard queries to more powerful models [91], or leverage cascades with self-verification [54,98,43,51].
Predictive Routing. In contrast, predictive routing aims to select the best model before any inference is performed. Strategies include supervised learning [76], reward-model-based routing [32], and meta-models trained to predict LLM performance given an input [71]. Router models vary widely in implementation, including neural networks [19,88,11,1], k-nearest neighbors [34,76,80,43], matrix factorization [62,105,48], and graph neural networks [25]. Others incorporate model-specific tokens or train across multiple domains [18,7].
Academic routers usually assumed the expert set is static. Recently routing with a dynamic pool of experts has been explored [40,48]. UMR [40] clusters probes and stores coarse capability footprints but still rebuilds them offline whenever the model pool changes. LLM-Bandit [48] optimizes cost but ignores prompt semantics and offers no cold-start prior for unseen experts.
this section cite: ['b9', 'b90', 'b53', 'b97', 'b42', 'b50', 'b75', 'b31', 'b70', 'b18', 'b87', 'b10', 'b0', 'b33', 'b75', 'b79', 'b42', 'b61', 'b104', 'b47', 'b24', 'b17', 'b6', 'b39', 'b47', 'b39', 'b47']

Section: Routing within MoE and Hybrid Architectures
Routing LLMs can be viewed as a coarse-grained MoE, where each expert is a full LLM. Routing is a central mechanism in MoE models [36,41,75], where expert modules are dynamically activated based on input. While classical MoEs involved equally-sized sub-models, modern approaches like Switch Transformer [24] and Mixtral [38] employ sparse activation to minimize cost. Approaches like UltraFuser [20] highlight recent advances in combining model specialization and flexibility.
this section cite: ['b35', 'b40', 'b74', 'b23', 'b37', 'b19']

Section: Model Fusion, Merging and Cascading
Fusion strategies synthesize outputs from multiple LLMs to improve output quality [68,39,29,89,52]. Fusion approaches often rely on unsupervised metrics [99,73,97] or ensemble voting to determine the final output [44]. A related but distinct technique is model merging [50], where weights from multiple pre-trained or fine-tuned models are combined, either directly via methods like weight averaging [92], Task Arithmetic [35], or Fisher merging [56]. In contrast, cascading invokes models sequentially (often ordered by computational cost) and halts once a satisfactory output is generated [10,98,30].
To the best of our knowledge, no prior work simultaneously (i) embeds both prompts and arbitrary experts into a unified metric space, (ii) incorporates inference cost explicitly into its learning objective, and (iii) generalizes effectively to new LLMs and out-of-distribution prompts using simple, efficiently computable descriptors.
this section cite: ['b67', 'b38', 'b28', 'b88', 'b51', 'b98', 'b72', 'b96', 'b43', 'b49', 'b91', 'b34', 'b55', 'b9', 'b97', 'b29']

Section: Method
This section formalizes our Cost-Spectrum Contrastive Router (CSCR) and its two drop-in, modelagnostic descriptors: logit fingerprints and perplexity fingerprints. CSCR is trained once on a fixed pool of LLMs and deployed without modification on any subset of that pool. At inference time it performs a k-NN lookup in a FAISS [21] indexfoot_0 to return the k most cost-effective experts for a prompt. Throughout this section, let H = {h (1) , . . . , h (M ) } denote the available LLMs, c(h) their normalized cost, and Φ(x) ∈ R D our frozen query encoder with a trainable MLP head g θ (.).
this section cite: []

Section: Model Fingerprints
We map every LLM to a compact, task-independent vector d h ∈ R D ′ . Both descriptors are gradientfree. They can be computed off-line, cached, and shipped without IP-sensitive weights.
this section cite: []

Section: Logit-Footprint Descriptors for Transparent LLMs
Let S probe = {x (i) } N i=1 be a fixed set of short, diverse prompts shared across all experts. For an autoregressive LLM h, denote by
p h (v | x, t) = softmax logits h (x) t v , v ∈ V, t ≥ 1 (1
)
the probability that h emits vocabulary token v at generation step t conditioned on the prompt prefix x. We compress these probabilities into a fixed-length logit footprint:
d logit (h) = 1 N T N i=1 T t=1 p h v k | x (i) , t K k=1 ∈ R K , (2
)
where T is a small horizon, and {v k } K k=1 are the K most frequent tokens across all probes. We ℓ 2 -normalize d logit (h) to live on the unit hypersphere, after which cosine similarity is a proxy for KL divergence between the first token distributions.
Why logits? Equation (2) directly samples the model's internal predictive distribution p h (•)-the very quantity trained by maximum-likelihood objective -log p θ [6]. It therefore encodes both topical preference and generation style while remaining inexpensive (only N ×T forward passes with greedy decoding). Thus, we use (2) as the primary descriptor whenever logits are available. See Appendix C.1 for a more detailed discussion.
this section cite: ['b5']

Section: Perplexity Fingerprints for Black-Box or API-Only LLMs
Closed-source APIs (e.g., GPT-o3 [37], Gemini-2.5 [69]) expose responses but hide almost all logits. For such models we adopt a per-prompt cross-entropy fingerprint:
ℓ h (x) = - 1 L x Lx j=1 log p h w j | w <j , (3
)
d PPL (h) = normalize ℓ h x (i) N i=1 ∈ R N ,(4)
where w 1:Lx are the gold target tokens (ground-truth answers if available, or probe continuations); normalize(•) denotes mean-centering and unit-variance scaling. Practically, we approximate (3) with a lightweight open LM that scores the API output ŷh (x) instead of inaccessible p h :
lh (x) = -1 |ŷ h (x)| j log p gpt2 ŷh,j | ŷh,<j . (5
)
See Appendix C.2 for a more detailed discussion. Why perplexity? Cross-entropy is proportional to KL(p true ∥ p h ) plus entropy of the data distribution. It therefore quantifies text fit and has long been a proxy for LM quality [58,66]. Prior work has shown that perplexity can serve as an effective metric for distinguishing between humangenerated text and LLM-generated output [31]. Moreover, although lh is an approximation, the descriptor vector in (4) still captures how hard each prompt is for a given expert. So routing on these vectors recovers much of the benefit of logit footprints while remaining viable for black-box LLMs. Figure 2 shows that, despite using the same model to compute the perplexity of text generated by different LLMs, we still observe a clear separation between the expert descriptors.
Unified metric space. We can make both descriptors reside on the unit sphere S K-1 by setting N = K (i.e., the number of top tokens in logit fingerprints equals the number of prompts in perplexity fingerprints) with cosine similarity σ(d 1 , d 2 ) = d ⊤ 1 d 2 . A single k-NN router can thus mix open-weight experts (logit fingerprints) and API experts (perplexity fingerprints) without altering downstream training loss and only the descriptor extraction pipeline has to change (See Table 4) .
this section cite: ['b36', 'b68', 'b57', 'b65', 'b30']

Section: Cost-Spectrum Contrastive Router
A contrastive router learns a shared embedding space where each query vector is pulled toward the descriptor of the right-sized expert and pushed away from less suitable ones. This yields three key pay-offs. First, because routing reduces to a nearest-neighbor lookup in that space, inference is a microsecond operation that adds virtually no latency to serving large expert pools. Second, contrastive objectives let the router exploit implicit supervision ("expert X solved this prompt while expert Y failed" or "X is cheaper than an equally accurate Y") so it can be trained with only correctness or cost signals with no dense human annotations required. Third, the geometry learned by contrastive learning naturally generalizes: queries that look semantically or structurally similar land near the same expert regions, which improves robustness to distribution shift and unseen prompts, a behavior long noted in contrastive representation learning. Together, these properties make a contrastive router an efficient, supervision-light and highly adaptable choice for directing traffic in modern multi-LLM systems.
this section cite: []

Section: Background: The Classic INFONCE Loss.
Let a minibatch contain B queries {x i } B i=1 and a memory bank of M keys {e m } M m=1 . A query encoder f θ : X → R d produces representations q i = f θ (x i )/∥f θ (x i )∥ 2 , and the keys are ℓ 2 -normalized in advance, e m = E m /∥E m ∥ 2 (here, E m is the expert descriptor from Equation (2) or (4), i.e. E m = d(h m )). For each query i let P(i) ⊂ [M ] denote the positives (e.g. correct experts) and N (i) = [M ] \ P(i) the negatives. The vanilla InfoNCE objective [87] maximizes a log-softmax over cosine similarities
L InfoNCE = - 1 B B i=1 log m∈P(i) exp q ⊤ i em τ M m ′ =1 exp q ⊤ i e m ′ τ ,(6)
where τ > 0 is a temperature. It encourages queries to be close to any positive but far from all negatives, thereby learning a metric embedding.
this section cite: ['b86']

Section: Cost-Spectrum InfoNCE.
Why Incorporate Cost: Routing must balance two competing axes: quality (the expert is correct) and inference cost c m (e.g. dollars or latency). The classical InfoNCE loss ignores c m , so the encoder can satisfy the objective by clustering any correct experts-most often the cheapest ones, since there are usually more of them in the pool-around the query embedding. Compounding this, easier prompts tend to occur more frequently, so training examples are skewed toward cases where cheap experts suffice. Once those low-cost positives are nearby, there is no training signal to learn where the slightly more expensive but markedly more accurate models live. Empirically, this drives the router to over-use the bargain-bin checkpoints, hurting accuracy and leaving significant potential untapped, even though paying a little more would buy a large quality jump (see Table 5).
We therefore introduce a cost-aware spectrum version that:
1. Selects all positive per cost band, preventing domination by extremely cheap or extremely costly experts 2. Assigns band-specific temperatures so that harder (costlier) positives yield smoother gradients 3. Penalizes negatives proportionally to their cost, pushing the encoder to prefer cheaper mistakes if it must err.
Formally we first normalize costs c m ∈ [0, 1] and partition them into K disjoint percentile bands B k = {m : c m ∈ [β k , β k+1 )} with quantiles
β 0 = 0 < • • • < β K = 1.
For each query i and band k, let P ik = P(i) ∩ B k denote the set of correct experts in that band. All experts in P ik are treated as positives, and weighted by a softmax over similarities scaled by a band-specific temperature
τ k = τ min + α • ck ,(7)
where ck is the mean cost of experts in B k .
With Φ(x) ∈ R D being our frozen query encoder with a lightweight trainable MLP head g θ (.), the loss for a query q i = g θ (Φ(x i ))/∥g θ (Φ(x i ))∥ 2 is then an average over all non-empty bands:
ℓ CS i = - 1 |Ki| k∈Ki log m∈P ik exp q ⊤ i em τ k M m ′ =1 exp q ⊤ i e m ′ -γc m ′ τ k ,(8)
where K i is the set of cost bands that contain at least one positive and γ ≥ 0 controls the negative cost penalty. Averaging over the minibatch yields
L CS = 1 B B i=1 ℓ CS i .
Banded positives. By retaining all positives within each cost band, we ensure that high-cost, correct experts still receive gradient signal, even when low-cost models also answer correctly. This prevents cost-collapse, the failure mode discussed in 3.4, where training signal concentrates on cheap experts due to prompt and model imbalances.
Cost-dependent temperature. Higher bands (larger ck ) get larger τ k , flattening their softmax and avoiding vanishing gradients when few difficult positives exist. In contrast, cheap bands keep a low temperature, sharpening the push towards inexpensive correct experts.
Negative cost penalty. Subtracting γc m in the denominator (not the numerator) means that expensive wrong experts contribute more to the partition function, hence increase the loss; the encoder is thus encouraged to separate from them first.
Thus we align three signals in the same metric space: (i) semantic proximity via the query encoder, (ii) expert capability via fingerprints, and (iii) user preference via cost scaling. Previous cost-aware objectives for retrieval weight the final scoring function at inference time (e.g. [48,40]). Our formulation also integrates cost during representation learning, inducing a feature geometry that naturally interpolates accuracy and cost. Equation ( 8) collapses to standard InfoNCE when K = 1 and γ = 0. See Appendices C.3 and C.4 for a more detailed discussion.
this section cite: ['b47', 'b39']

Section: Inference Router
Given a test prompt x we retrieve r(x) = arg max
h∈Top k (x) cos⟨g θ (Φ(x)), d h ⟩ -λ c(h) ,(9)
λ is the cost weight and Top k (x) retrieves the k most similar experts to the prompt from the FAISS index. We use k = 4 by default. During training, a similar composite score appears in the costspectrum InfoNCE objective (Equation ( 8)), allowing the encoder to rank candidate models by similarity to expert descriptors plus the cost term γc(h), just as in the inference rule of Equation 9.
We show in the next section that this alignment between training and inference is highly effective.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Settings
Baselines We compare our proposed method against a comprehensive set of baselines. Specifically, we include UMR [40], a recent technique that clusters prompt embeddings to route queries to LLM pools efficiently; Thompson Sampling [48,2], which frames routing as a bandit exploration-exploitation problem to balance cost and accuracy dynamically; Pareto-optimal routing [34], a strategy that selects models by explicitly considering the cost-accuracy Pareto frontier; and two extreme baselines: Random, which selects models uniformly at random to represent naive routing without intelligent selection, and Oracle [40], which always selects the most accurate model at the lowest possible cost and thus represents a theoretical performance ceiling. Additionally, we evaluate against parametric-softmax gating methods inspired by mixture-of-experts architectures(e.g. [62]) and SoftMoE [64], which models router decisions via differentiable soft gating functions.
this section cite: ['b39', 'b47', 'b1', 'b33', 'b39', 'b61', 'b63']

Section: Datasets & Benchmarks
We train our router and evaluate it on three datasets: EmbedLLM [105], MixInstruct [39], and RouterBench [34]. For EmbedLLM and MixInstruct, we sample 192 probes from their respective validation sets. Each probe is processed to extract logit-based descriptors by capturing the top K = 256 tokens over a horizon of T = 10 tokens (Equation ( 2)). For RouterBench, we sample 192 probes from its training set. We compute perplexity-based descriptors on RouterBench and use GPT-2 [66]. On both EmbedLLM and RouterBench, we use binary accuracy as the per-sample evaluation metric. For MixInstruct, we employ exponentiated BARTScore [97] as the evaluation metric, following the approach in prior work [40,39].
Training We use a frozen sentence-transformers/all-MiniLM-L6-v2 [70] model as the embedding backbone across all experiments. Our trainable router component is a two-layer MLP which projects prompt embeddings into the expert descriptor space. We train our contrastive router on the training splits of each dataset. For the cost spectrum loss (Equation ( 8)), we set the number of cost bands K = 5 and the negative cost penalty γ = 0.2. The hyperparameters for band-specific temperatures (Equation ( 7)) are set as α = 0.25 and τ min = 0.05. See D.1 for full details.
this section cite: ['b104', 'b38', 'b33', 'b65', 'b96', 'b39', 'b38', 'b69']

Section: Evaluation
We evaluate each routing strategy using a deferral curve [40] which plots the average response quality against the total inference cost. Sweeping the routing penalty parameter λ over the interval λ ∈ [0, λ max ] (Equation ( 9)) traces the deferral curve. For the EmbedLLM and MixInstruct datasets, we define the cost of processing a prompt as the number of parameters in the LLM, a proxy for computational resources and latency. In the case of RouterBench, we utilize the actual API call costs in USD, as provided in the dataset. Following [40] we employ evaluation metrics including Area Under the Deferral Curve (AUDC), peak accuracy and Query-Normalized Cost (QNC), the minimum relative cost required to match the performance of the most accurate tested LLM.
this section cite: ['b39', 'b39']

Section: Results
Table 1 presents Deferral curve metrics across the benchmarks. Our CSCR consistently outperforms all relevant baselines, achieving the highest AUDC and demonstrating competitive or superior peak accuracy. Notably, it attains lower QNC, indicating more cost-effective routing decisions. These results show the effectiveness of our cost-aware router learning approach in balancing performance and inference cost. The Oracle router, which selects the optimal LLM for each query, establishes an upper bound for performance. The benchmarks are dominated by lower cost experts, hence the lower QNC for random baselines. See Appendix D.6 for results on statistical significance.
this section cite: []

Section: Generalization to New LLMs
We also evaluate the robustness of our method and baselines in scenarios where new LLMs are introduced during testing. Specifically, we select two-thirds of the EmbedLLM training models to train our router and the baselines following [40], using only responses from these selected models. Table 2 summarizes the performance metrics, while Figure 3 illustrates the corresponding deferral curves. Testing is conducted exclusively on the unseen LLM pool. Our results indicate that our approach exhibits superior robustness under these conditions.   Table 3: Deferral curve metrics on OOD prompts. CSCR shows superior robustness.
this section cite: ['b39']

Section: Generalization to Out-of-Distribution Prompts
We evaluate our approach on an out-of-distribution (OOD) prompt at test time scenario. We split the EmbedLLM dataset into two subsets: one focusing on STEM-related prompts (e.g., science and technology) and the other comprising all remaining categories (see Appendix D.2.1 for detailed experimental settings). As illustrated in Figure 4 and quantified in Table 3, CSCR significantly outperforms all baselines across all key metrics. Specifically, our router achieves an AUDC of 0.5146 compared to the next-best baseline, Thompson, at 0.4863, highlighting its superior robustness and accuracy when handling diverse OOD prompts. This performance advantage demonstrates that our method generalizes exceptionally well, maintaining reliable decision-making capability across varied and harsh distributional shifts.
this section cite: []

Section: Ablative Studies

this section cite: []

Section: Descriptor Choice
Table 4 compares two descriptors on Mix-Instruct, the only benchmark where we can compute both. Perplexity descriptors (obtained by running every candidate answer through an auxiliary language model) lift the AUDC from 0.461 to 0.467. The absolute peak accuracy, however, changes less than 0.1%. Because the perplexity pipeline requires (i) generating text with the model in the pool and (ii) a second forward pass through a public LM, it is often at least 2× slower. Logit descriptors, in contrast, need only a single pass on open-weights models and still deliver competitive AUDC. We therefore adopt logit-based descriptors for all open LLMs and fall back to perplexity descriptors only when logits are inaccessible. The "mixed" row in 4 represents the results obtained by using logit descriptors for 6 randomly selected LLMs and perplexity descriptors for the remaining 5. This shows that both descriptors can be combined within the same pool without negatively affecting the results. In fact, performance slightly improves compared to using only a single descriptor type. This observation further supports our discussion of the "unified metric" in Section 3.1.2. Table 5: Effect of cost-aware training: injecting cost awareness into the contrastive loss prevents the router from concentrating on cheap experts, and boosts the AUDC and peak accuracy of the router. The trade-off is a higher QNC. Replacing the vanilla InfoNCE loss with our cost-spectrum variant significantly increases the AUDC (0.342 → 0.495) and raises the peak attainable accuracy from 36% to 54% as shown in Table 5. The trade-off is a higher Quality-Neutral Cost QNC, meaning the router now leans more on expensive but accurate models; however, the large AUDC gain shows that, for any realistic cost budget, users receive better accuracy-per-dollar overall. This confirms our discussion in Section 3.4, that explicitly teaching the encoder to respect the cost hierarchy of experts is crucial.
this section cite: []

Section: Cost-
Aware
Furthermore, Figure 1 and Table 1 show that while cost-aware contrastive training incurs more cost than the vanilla variant, it remains far more efficient than the baselines, achieving lower QNC by learning to distinguish good cheap experts from both bad cheap and bad expensive ones. See Figure 5 for an example. Figure 6 shows that performance varies with the number of cost bands. Using 5 bands yields the best results, with the highest AUDC and Peak accuracy, suggesting a good balance between flexibility and generalization. Fewer bands (e.g., 2) limit routing precision, while too many bands (e.g., 15 or 20) degrade performance, likely due to over-fragmentation and increased decision noise. This highlights the importance of tuning the number of bands per dataset to avoid both under-and overfitting. See Appendix D.2 for more ablations and detailed discussion. where λ ∈ R ≥0 trades accuracy for cost [23].
this section cite: ['b22']

Section: Cost-Spectrum Granularity

this section cite: []

Section: Theoretical Analysis
Our router embeds queries via Φ q : X → R d and experts via E = e 1 ; . . . ; e M ∈ R M ×d , using either (i) logits fingerprints (EmbedLLM, Mix-Instruct) or (ii) perplexity fingerprints (RouterBench). Given a query x, the k-NN rule selects
rk (x; λ) = arg min m∈[M ] 1 k j∈N k (x) γ x j , h m local error +λ c(h m ) .
where N k (x) are the k nearest training prompts to x in ∥Φ q (•)∥ 2 .
this section cite: []

Section: Excess-risk of cost-spectrum k-NN Assumption (Lipschitz Bayes error).
There exists L > 0 s.t. for all x, x ′ ∈ X and h ∈ H, |γ(
x, h) -γ(x ′ , h)| ≤ L ∥Φ q (x) -Φ q (x ′ )∥ 2 .
this section cite: []

Section: Theorem 5.2 (Excess risk).
Let rk be trained on n i.i.d. prompt embeddings. Under Assumption 5.1, for any λ ≥ 0 and any k ≤ n,
E R λ (r k ) -R λ (r ⋆ ) ≤ C k n + k -1/d
, where C depends only on L and the diameter of Φ q (X ), and r ⋆ is the Bayes-optimal rule r ⋆ (x) = arg min m γ(x, h m ) + λc(h m ) .
The proof follows the classical k-NN bound of [53,5] with an extra λc(h) term that is constant w.r.t.
x and therefore preserves the rate.
this section cite: ['b52', 'b4']

Section: Consistency of Cost-Spectrum InfoNCE
Fingerprints live on S d-1 , so dot products are scaled similarities S im = q ⊤ i em τ k with band-dependent temperature τ k (Equation ( 7)). Let β 0 = 0 < β 1 < • • • < β K = 1 partition costs into bands B k = {m : c(h m ) ∈ [β k , β k+1 )}, and denote P ik = P(i) ∩ B k the correct experts for query i that fall in band k. The Cost-Spectrum InfoNCE loss for a single query i is
ℓ CS i = - 1 |K i | k∈Ki log m∈P ik exp S im M m ′ =1 exp S im ′ -γ c m ′ ,
where K i = {k : P ik ̸ = ∅} and γ ≥ 0 is the negative cost penalty.
this section cite: []

Section: Lemma 5.3 (Directional alignment with cost bands). At any stationary point of L
CS = 1 B i ℓ CS
i , for every query i and any m + ∈ P ik , m -∈ N (i) with c m + ≤ c m -, q ⊤ i e m + > q ⊤ i e m -. Lemma 5.3 shows the optimum ranks cheaper correct experts ahead of expensive or wrong ones, explaining the empirical benefit of the cost term.
this section cite: []

Section: Discussion
Theorem 5.2 guarantees that if query-expert descriptors are Lipschitz, CSCR converges to the Bayes-optimal router at the usual k-NN rate. Lemma 5.3 justifies the specific form of our InfoNCE objective. Sec. 4 verifies these claims on three benchmarks. See proofs in Appendix B.
this section cite: []

Section: Conclusion
We presented CSCR, a simple and efficient framework for cost-aware routing across a pool of LLMs.
Our method uses two lightweight expert descriptors and trains a contrastive encoder to select the cheapest accurate expert within adaptive cost bands. Despite its simplicity, CSCR outperforms more complex routing baselines and demonstrates strong generalization to unseen models and out-ofdistribution prompts. Our findings highlight the importance of embedding cost-awareness directly into the training objective of routers, rather than deferring it to test time. As model pools grow in size and diversity, activating the right-sized expert per query is critical for minimizing latency and cost. We see CSCR as a step toward more sustainable and adaptive LLM deployments, and believe this line of research is essential to avoid defaulting to unnecessarily large models for simple tasks.
this section cite: []

Section: Limitations and Broader Impact
This paper proposes a routing framework that improves inference efficiency in large language models (LLMs). By directing simple queries to smaller models, it cuts computation and memory overhead, lowering costs and environmental impact. Although the framework is built for large-scale deployments, we could not test very large LLM pools due to resource limits. Still, our experiments validate the concept. Future work will examine routing across specialized subnetworks and conditional computation within a single LLM. Moreover, future work can answer: How big must a model be to recognize a problem's difficulty even if it can't solve the problem itself? Future work can study whether size and fine-tuning helps, would RL on LLMs makes them able to purely as routers, and whether there is a model size when difficulty awareness kicks in.
this section cite: []

Section: A Related Work

this section cite: []

Section: A.1 Enhancing and Optimizing LLMs
Large language models (LLMs) have demonstrated remarkable capabilities across diverse NLP tasks [66,6]. To further improve their performance and efficiency, numerous strategies have been proposed.
Single-LLM Techniques. Enhancement approaches targeting individual LLMs include finetuning [67], prompting strategies like Chain-of-Thought (CoT) [102,90], Tree-of-Thoughts [96], and inference-acceleration techniques such as early exiting [84,103,72] and speculative decoding [77,83,9,45,8]. Additionally, Mixture-of-Experts (MoE) architectures [36,41,75,24,104,38] route inputs through sparse sub-models or "experts", reducing cost while retaining performance. However, these methods typically operate within a single LLM's structure and may not generalize to multi-model scenarios.
this section cite: ['b65', 'b5', 'b66', 'b101', 'b89', 'b95', 'b83', 'b102', 'b71', 'b76', 'b82', 'b8', 'b44', 'b7', 'b35', 'b40', 'b74', 'b23', 'b103', 'b37']

Section: Model Fusion and Merging.
Fusion strategies synthesize outputs from multiple LLMs to improve output quality [68,39,29,89,52]. Fusion approaches often rely on unsupervised metrics [99,73,97] or ensemble voting to determine the final output [44]. A related but distinct technique is model merging [50], where weights from multiple pre-trained or fine-tuned models are combined-either directly via methods like weight averaging [92], Task Arithmetic [35], or Fisher merging [56]; or using more sophisticated techniques like TIES [94], AdaMerging [95], and ZipIt [78].
Cascading. In contrast, cascading invokes models sequentially-often ordered by computational cost-and halts once a satisfactory output is generated [10,98,10,30]. Such approaches strike a balance between quality and efficiency, making them particularly attractive in production settings.
this section cite: ['b67', 'b38', 'b28', 'b88', 'b51', 'b98', 'b72', 'b96', 'b43', 'b49', 'b91', 'b34', 'b55', 'b93', 'b94', 'b77', 'b9', 'b97', 'b9', 'b29']

Section: A.2 LLM Routing
Routing methods dynamically select the most appropriate LLM from a pool for each input, aiming to optimize performance and cost without querying all models. Two primary strategies dominate: non-predictive and predictive routing.
Non-Predictive Routing. Non-predictive methods generate outputs from one or more models before making a selection. FrugalGPT [10] exemplifies this category, using a sequential strategy and a response quality threshold to minimize cost. Other works adopt layered inference architectures to escalate hard queries to more powerful models [91], or leverage cascades with self-verification [54,98,43,51].
Predictive Routing. In contrast, predictive routing aims to select the best model before any inference is performed. Strategies include supervised learning [76], reward-model-based routing [32,51], and meta-models trained to predict LLM performance given an input [71]. Router models vary widely in implementation, including neural networks [19,88,11,1], k-nearest neighbors [34,76,80,43], matrix factorization [62,105,48], and graph neural networks [25]. Others incorporate model-specific tokens or train across multiple domains [18,7].
Theoretical Foundations and Robustness. Routing and cascading are grounded in broader literature, including selective classification [27,60], learning to defer [55], and learning to reject [12,4,14]. Several works explore supervision levels [51,101], robustness [16,59,74], and evaluation frameworks for routers [33,32].
this section cite: ['b9', 'b90', 'b53', 'b97', 'b42', 'b50', 'b75', 'b31', 'b50', 'b70', 'b18', 'b87', 'b10', 'b0', 'b33', 'b75', 'b79', 'b42', 'b61', 'b104', 'b47', 'b24', 'b17', 'b6', 'b26', 'b59', 'b54', 'b11', 'b3', 'b13', 'b50', 'b100', 'b15', 'b58', 'b73', 'b32', 'b31']

Section: A.3 Routing as Recommendation
Routing can also be framed as a recommendation problem, wherein the input query plays the role of a "user", the pool of LLMs corresponds to "items", and past performance metrics form the implicit interaction history [100,93]. However, unlike conventional recommender systems, routing has limited "user" features (i.e., input metadata), making label collection and generalization especially challenging [61,49].
Matrix factorization, attention-based models, and graph neural networks are used in both recommenders and routers [62,105,25], reinforcing the close link between the two domains.
this section cite: ['b99', 'b92', 'b60', 'b48', 'b61', 'b104', 'b24']

Section: A.4 Scaling Laws and Architecture Trends
Scaling laws [42,47] describe predictable trends between model size, data, and performance, guiding the development of efficient LLM architectures. These insights have been extended to MoEs [24,13], sparse models [26], and hybrid systems [28,63], offering context for when routing or merging approaches might be most beneficial.
this section cite: ['b41', 'b46', 'b23', 'b12', 'b25', 'b27', 'b62']

Section: A.5 Routing within MoE and Hybrid Architectures
Routing is a central mechanism in MoE models [36,41,75], where expert modules are dynamically activated based on input. While classical MoEs involved equally-sized sub-models, modern approaches like Switch Transformer [24] and Mixtral [38] employ sparse activation to minimize cost.
Routing LLMs can be viewed as a coarse-grained MoE, where each expert is a full LLM. Approaches like UltraFuser [20], Branch-Train-MiX [81], and token-level fusion highlight recent advances in combining model specialization and flexibility.
this section cite: ['b35', 'b40', 'b74', 'b23', 'b37', 'b19', 'b80']

Section: B Proofs of Theoretical Results

this section cite: []

Section: B.1 Notation and Preliminaries
Let the Bayes-optimal router be r ⋆ (x) = arg min m∈[M ] γ(x, h m ) + λc m and define the excess cost-error gap
∆ m (x) = γ(x, h m ) + λc m -γ(x, h r ⋆ (x) ) + λc r ⋆ (x)
. Hence ∆ r ⋆ (x) (x) = 0 and ∆ m (x) ≥ 0. For a query x let r k (x) be the radius of the ball B(x, r) ⊂ S d-1 (in the cosine metric) that contains the k-th nearest training neighbor. If the marginal on q(X ) has a density, E r k (x) d ≤ C 1 k/n [15,79].
this section cite: ['b14', 'b78']

Section: B.2 Proof of Theorem 5.2
Writing q i = q(x i ) to lighten notation, decompose E R λ (r k ) -R λ (r ⋆ ) = E x   ∆r k (x) (x) -∆ rk (x) (x j∈N k (x) ) (A) +∆ rk (x) (x j∈N k (x) ) -∆ r ⋆ (x) (x j∈N k (x) ) (B)    . Term (A): Lipschitz bias. Assumption 5.1 gives |γ(x, h) -γ(x ′ , h)| ≤ L∥q(x) -q(x ′ )∥ 2 , so |(A)| ≤ L r k (x).
this section cite: []

Section: Taking expectations and using
E[r k (x)] ≤ (C 1 k/n) 1/d yields E[(A)] ≤ C 2 k -1/d .
this section cite: []

Section: Term (B): Estimation variance.
Let the empirical cost-adjusted risk be
∆ m (x) = 1 k j∈N k (x) γ(x j , h m ) + λc m .
Hoeffding's inequality bounds
P | ∆ m (x) -E[ ∆ m (x) | x]| ≥ t ≤ 2e -2kt 2 ,
and a union bound over m ≤ M plus integration gives
E[max m | ∆ m (x) -E[ ∆ m (x) | x]|] ≤ C 3 log M k . Because rk (x) minimizes ∆ m (x), (B) ≤ 2 max m | ∆ m (x) -E[ ∆ m (x) | x]|. Combining with (A) implies E[R λ (r k )] -R λ (r ⋆ ) ≤ C k/n + k -1/d .
this section cite: []

Section: B.3 Proof of Lemma 5.3
For convenience write S im = q ⊤ i e m /τ k when m ∈ B k . The single-query loss (Equation ( 8)) is
ℓ CS i = - 1 |K i | k∈Ki log m∈P ik exp(S im ) M m ′ =1 exp S im ′ -γc m ′ . Let p im = exp(S im )/ j∈P ik exp(S ij )
when m ∈ P ik and
q im = exp(S im -γc m )/ M j=1 exp(S ij -γc j )
Then
ℓ CS i = -1 |Ki| k log m∈P ik p im /q im
Taking the gradient w.r.t. q i and setting it to zero gives m (p im -q im )e m = 0. Project onto q i :
m (p im -q im )S im = 0. Fix m + ∈ P ik , m -∈ N (i) with c m + ≤ c m -. Because p im -= 0 while p im + > 0, the equality forces q im + > q im -, hence S im + -γc m + > S im --γc m -. Rearranging yields q ⊤ i e m + > q ⊤ i e m -, establishing directional alignment. 2
this section cite: []

Section: C Method

this section cite: []

Section: C.1 Logit-Footprint Descriptors
Why take the most frequent tokens? We use the most frequent tokens so the basis is shared and stable: they appear in all models, give low-noise estimates with few probes, and make calibration comparable across experts. They give less noisy estimates because they get non-negligible probability across many contexts, so their averaged log-probs vary less than rare or Out-of-Vocabulary tokens.
In all experiments, we set K = 256 and T = 10 (D.1), which is large enough that the basis isn't dominated by a few function words.
Are frequent tokens trivial? These tokens aren't used for their meaning. They're probes of each model's output behavior. Even common words get scored differently across models (temperature, punctuation/number handling, style). By averaging over many prompts and steps, the descriptor captures overall model behavior, not any single word's semantics. Also, a recent work [82] shows that LLMs exhibit stable, word-level idiosyncrasies (as the authors call them) that enable nearperfect model attribution using only the first few generated tokens (even after paraphrasing or translation), implying that common tokens still provide discriminative signals about a model's predictive calibration.
this section cite: ['b81']

Section: Shared token set vs. per-model selection.
A shared basis gives all descriptors a common coordinate system. If each model used a different token set, cosine distances would mix basis changes with true behavior, hurting comparability. It would also require computing many more probes to align per-model bases that are different across models.
this section cite: []

Section: Beyond raw frequency.
We deliberately kept the descriptors simple to isolate and quantify the contrastive router's contribution. Nonetheless, frequency is a pragmatic, not necessarily optimal, choice. Two variants that we considered and could be explored are:
• TF-IDF-weighted selection over the probe corpus.
• Picking tokens with the largest across-model log-prob variance.
These can be dropped into Equation 2 without changing downstream training or inference.
this section cite: []

Section: C.2 The Step from Equation 3 to Equation 5
Equation ( 3) defines a per-prompt token NLL that requires access to an expert's next-token distribution p h (• | •). For API-only (black-box) experts, logits/probabilities are not exposed, so Equation ( 3) is not computable. Our remedy is to (a) let the API expert h produce a deterministic continuation ŷh (x) for prompt x (greedy decoding), and (b) evaluate that sequence under a single shared, public scorer p S (kept fixed across all experts). This yields Equation ( 5), a pseudo-perplexity:
lh (x) = - 1 |ŷ h (x)| |ŷ h (x)| t=1 log p S (ŷ h,t | ŷh,<t ) ,
which we then normalize (similar to Equation ( 4)) and use as the fingerprint coordinate(s) for blackbox experts.
If ŷh is a typical (high-probability) output of h (i.e., ŷh ∼ p h ) then averaging the pseudo-perplexity lh (
x) = -1 |ŷ h (x)| t log p S (ŷ h,t | ŷh,<t ) over many prompts/tokens is equivalent to taking an expectation over y ∼ p h : E y∼p h -log p S (y) = H(p h , p S ) = H(p h ) + KL p h ∥ p S .
Here H(p h , p S ) is the cross-entropy of p h with respect to p S , which decomposes into the entropy of h's own distribution H(p h ) and its divergence from the scorer KL(p h ∥p S ).
Because p S is fixed for all experts, H(p h , p S ) is a stable, model-specific quantity that makes descriptors comparable across experts ("same yardstick"). It is not the true NLL under p h , but it preserves differences between experts via H(p h ) and their mismatch to p S via KL(p h ∥p S ). In practice we use deterministic (greedy) decoding to reduce variance, averaging over many prompts/tokens makes the empirical lh (x) closely track the expectation above. Figure 2 empirically validates this argument.
this section cite: []

Section: C.3 Band-Specific Temperatures and Smoother Gradients
Most prompts in everyday interactions (and in our datasets) can be handled by cheaper models; plus there are usually fewer very expensive experts overall. These expensive experts are only needed for a small fraction of hard prompts, so within those high-cost bands there are fewer suitable positives per query. With few positives, the similarity distribution becomes very peaky.
A larger τ k in Equation 7 flattens the per-band softmax, reducing gradient variance and preventing the update from collapsing onto a single rare positive. Formally, for band k the per-query gradient w.r.t. the query embedding is
∇ q L k = - m∈P ik p (+) m e m τ k + M m ′ =1 p (-) m ′ e m ′ τ k ,(10)
where p (+) and p (-) are the band-restricted softmaxes over positives and all experts (with the negative cost penalty in the denominator). As τ k increases, both softmaxes become less concentrated, so (i) Table 6: Full breakdown of training and testing categories used in OOD experiments.
this section cite: []

Section: Set Categories
Train (STEM) asdiv, gsm8k, medmcqa, mathqa, piqa, logiqa, gpqa_main_cot_n_shot, gpqa_main_cot_zeroshot, gpqa_main_n_shot, gpqa_main_zeroshot, gpqa_diamond_cot_n_shot, gpqa_diamond_cot_zeroshot, gpqa_diamond_n_shot, gpqa_diamond_zeroshot, gpqa_extended_cot_n_shot, gpqa_extended_cot_zeroshot, gpqa_extended_n_shot, gpqa_extended_zeroshot, mmlu_college_medicine, mmlu_astronomy, mmlu_conceptual_physics, mmlu_college_computer_science, mmlu_college_biology, mmlu_electrical_engineering, mmlu_medical_genetics, mmlu_college_physics, mmlu_high_school_chemistry, mmlu_computer_security, mmlu_clinical_knowledge, mmlu_virology, mmlu_machine_learning, mmlu_college_mathematics, mmlu_elementary_mathematics, mmlu_professional_medicine, mmlu_college_chemistry, mmlu_high_school_biology, mmlu_anatomy, mmlu_high_school_statistics, mmlu_high_school_physics, mmlu_high_school_computer_science, mmlu_high_school_mathematics
Test (Non-STEM)
social_iqa, truthfulqa_mc1, mmlu_high_school_european_history, mmlu_us_foreign_policy, mmlu_high_school_microeconomics, mmlu_business_ethics, mmlu_public_relations, mmlu_jurisprudence, mmlu_nutrition, mmlu_high_school_world_history, mmlu_miscellaneous, mmlu_formal_logic, mmlu_management, mmlu_high_school_psychology, mmlu_high_school_government_and_politics, mmlu_high_school_geography, mmlu_world_religions, mmlu_international_law, mmlu_human_aging, mmlu_sociology, mmlu_professional_accounting, mmlu_prehistory, mmlu_logical_fallacies, mmlu_moral_disputes, mmlu_human_sexuality, mmlu_professional_psychology, mmlu_high_school_us_history, mmlu_high_school_macroeconomics, mmlu_abstract_algebra, mmlu_global_facts, mmlu_security_studies, mmlu_philosophy, mmlu_professional_law, mmlu_moral_scenarios, mmlu_marketing
We evaluate each routing strategy using a deferral curve [40] which plots the average response quality against the total inference cost. Sweeping the routing penalty parameter λ over the interval λ ∈ [0, λ max ] (refer to Equation ( 9)) traces the deferral curve. For the EmbedLLM and MixInstruct datasets, we define the cost of processing a prompt as the number of parameters in the LLM, serving as a proxy for computational resources and latency. In the case of RouterBench, we utilize the actual API call costs in USD, as provided in the dataset. Following [40] we employ evaluation metrics including Area Under the Deferral Curve (AUDC), Query-Normalized Cost (QNC), and peak accuracy. QNC is the minimum relative cost required to match the performance of the most accurate tested LLM.
Training We use a frozen sentence-transformers/all-MiniLM-L6-v2 [70] model as the embedding backbone (Φ(x) in Section3) across all experiments. Our trainable router component is a two-layer MLP, denoted as g θ (.), which projects prompt embeddings into the expert descriptor space. We train our contrastive router on the training splits of each dataset, excluding the probe examples from RouterBench. Training is performed for 10 epochs using the AdamW optimizer with a batch size of 512 and a learning rate of 5 × 10 -4 . For the cost spectrum loss (Equation ( 8)), we set the number of cost bands to K = 5 and the negative cost penalty to λ = 0.1. The hyperparameters for the linear schedule of band-specific temperatures (Equation ( 7)) are set as α = 0.25 and τ min = 0.05. All training and descriptor extraction are done on RTX6000Ada GPUs with ∼48GB GPU memory.
this section cite: ['b39', 'b39', 'b69']

Section: D.2 Results

this section cite: []

Section: D.2.1 Out-of-Distribution Prompt Experiments
In the out-of-distribution (OOD) experiments, we divided the prompts in the EmbedLLM dataset into two challenging sets based on their categories: STEM-related (Science, Technology, Engineering, and Mathematics) and Non-STEM-related (covering Social sciences, Humanities, Arts, etc.). A detailed summary of the train and test categories for the OOD experiments is provided in Table 6.
Our splitting yielded 18,193 out of 36,054 total training questions and 1,060 distinct test prompts.
Training and testing is done on all available LLMs across both splits.
this section cite: []

Section: D.3 Ablation of Cost Bands
We measured AUCD and peak accuracy with and without bands. Table 7 presents these results, showing having cost bands is indeed effective.
this section cite: []

Section: Setting AUCD Peak Accuracy
Without bands 0.4574 0.482 With bands 0.4951 0.540 Table 7: Effect of banded cost temperatures on model performance  Table 9: Effect of Number of Nearest Neighbors (K) on AUDC and Average Cost
this section cite: []

Section: D.3.1 Ablation of Number of Cost bands
Table 8 and Figure 7 show that performance varies with the number of cost bands. Using 5 bands yields the best results, with the highest AUDC (0.5480) and Peak accuracy (0.590), suggesting a good balance between flexibility and generalization. Fewer bands (e.g., 2) limit routing precision, while too many bands (e.g., 15 or 20) degrade performance, likely due to over-fragmentation and increased decision noise. This highlights the importance of tuning the number of bands to avoid both underand overfitting.
this section cite: []

Section: D.3.2 Ablation of the Number of Neighbors
Increasing k (the number of ANN neighbors selected before cost-aware scoring) generally provides modest improvements before reaching a plateau. (The baseline UMR [40] also included an ablation on k in a K-NN router.) Beyond a small K, the gains in accuracy or AUDC become minimal, while both latency and the likelihood of choosing unnecessarily expensive experts increase. We selected a default of K = 4 and it worked reasonably well so we did not do further hyperparameter tuning. We performed an ablation on a subset of embedllm prompts. The results are shown in Table 9 which are consistent with the trend observed in UMR.
this section cite: ['b39']

Section: D.3.3 Ablation of Negative Cost Penalty
γ is a soft deterrent against assigning probability mass to costly, wrong experts during training (it appears in the denominator of the band-softmax in the loss in Equation 8. We ablate this hyperparameter in Table 10. If too small, the router learns to over-consider expensive hub experts (cost creep). If too large, it over-penalizes cost and undertrains on valid high-cost positives, hurting hard prompts. We picked 0.2 as it's a light regularizer: enough to push apart costly negatives first, but not so strong that it drowns the similarity signal for truly necessary expensive experts.
this section cite: []

Section: D.3.4 Ablation of Band-Specific Temperature Slope
In the band-specific temperature schedule τ k = τ min + α ck (Equation 7), the slope α sets how much flatter the softmax is in higher-cost bands. Increasing α raises τ k for expensive bands, which flattens their per-band softmax over experts. This reduces gradient variance in those bands (where each query Table 11: Ablation over the band slope α in τ k = τ min + α ck . Moderate α (0.1-0.25) maximizes AUDC and lowers the cost required to reach peak accuracy. Large α oversmooths high-cost bands and reduces peak accuracy.
has fewer suitable positive) mitigating collapse onto a single rare positive and preventing cost-creep during training.
See Table 11 for an ablation of this hyperparameter. As α increases from 0 to a moderate value (0.1-0.25), AUDC improves and the cost@max_acc drops, indicating we reach peak quality at lower cost, while max_acc remains comparable. For larger α (≥ 0.4), the high-cost bands become oversmoothed, weakening discrimination among expensive experts so max_acc and AUDC decline despite further cost reductions. This validates our choice of adopting a moderate setting (default α=0.25, with 0.1 performing slightly better).
this section cite: []

Section: D.3.5 Ablation of Cheapest Cost Band
τ min is the softmax temperature for the cheapest cost band in Equation 7. All other bands inherit τ k = τ min + α ck . A very small τ min makes the cheap-band softmax sharp (highly discriminative but prone to noisy, peaky gradients) while a larger τ min smooths the distribution, lowering variance but also blurring differences among cheap experts.
Table 12 ablates τ min . Raising τ min from 0 to 0.02 increases AUDC and peak accuracy, showing that a touch of smoothing stabilizes learning without hurting discrimination. At τ min = 0.05 we keep nearly the same AUDC while cutting the cost needed to achieve peak accuracy by ≈ 14% (from 51.7 to 44.6). Pushing to τ min = 0.08 oversmoothes the cheap band: accuracy at low cost rises slightly, but max_acc and AUDC both fall. Thus a moderate setting (τ min ≈ 0.02-0.05) offers the best efficiency-stability trade-off.
this section cite: []

Section: D.4 Larger Encoder Size
We kept the router is deliberately small: a frozen sentence-transformer plus a 2-layer MLP. We performed an ablation where we increased the dimension of the middle layer in Table 13. As the router size increases, AUDC improves, but latency also increases. Table 14: Prompt Counts by Band Index
Broadly, future work can answer: How big must a model be to recognize a problem's difficulty even if it can't solve the problem itself? Future work can also study whether size and fine-tuning helps and whether RL on LLMs makes them able to purely as routers? Future work can also study different model sizes to pinpoint when difficulty awareness kicks in.
this section cite: []

Section: D.5 Qualitative Insights and Interpretability of Routing
We performed an analysis on embedllm, which features a large pool of models with diverse costs. Note that these observed trends are specific to the dataset and may not generalize to other datasets or prompt pools with more challenging examples.
this section cite: []

Section: D.5.1 Selection Profiles
Qwen/Qwen1.5-0.5B-Chat (selected 91 times), google/gemma-2b-it (233), and microsoft/phi-2 (292) are smaller experts that are selected frequently. Some expensive experts (e.g., Qwen/Qwen-72B, ibivibiv/alpaca-dragon-72b-v1) are rarely chosen, since a less costly correct expert typically exists in the dataset.
We also report per-expert selection rates by cost band in Table 14.
this section cite: []

Section: D.5.2 Routing Error Breakdown
We present a confusion-style breakdown of routing errors in  16: Paired significance versus the strongest baseline. ∆ AUDC = AUDC CSCR -AUDC UMR (area under the deferral curve. Higher is better). "95% CI" and the one-sided p come from a paired bootstrap over prompts (N = 3000, B = 5000, H 1 : ∆ > 0). c ⋆ is the matched budget used for McNemar. n 10 /n 01 are discordant counts (CSCR correct / baseline correct), and "p (McNemar)" is the one-sided exact binomial p for CSCR > baseline at c ⋆ . CIs that exclude 0 and small p-values indicate a statistically significant improvement of CSCR.
this section cite: []

Section: D.6 Statistical Significance
We perform full evaluation on Embedllm [105] using paired, prompt-level significance tests to concretely assess statistical significance. Specifically, we computed a paired bootstrap [22] (sampling 3,000 prompts with replacement, 5,000 times) to obtain a 95% confidence interval for ∆ AUDC = AUDC CSCR -AUDC UMR (UMR is the best baseline overall). We also report a one-sided p-value for the hypothesis ∆ > 0. Additionally, we ran McNemar's [57] test at a matched budget (using the median of the combined cost grids) to compare per-prompt wins and losses at equivalent operating cost. These tests quantify uncertainty over the test prompts.
∆AUDC is positive with CIs that exclude zero, and McNemar shows win rates above 0.5 with very small p-values at the matched budget. In other words: CSCR's deferral curve encloses more area (higher accuracy at the same or lower cost on average), and at a fixed budget it wins on more prompts than it loses. This complements the Pareto-frontier plots: the gains are not an artifact of a single operating point or random variation, but hold paired, prompt by prompt. The paired test establishes statistical significance.
this section cite: ['b104', 'b21', 'b56']

Section: 
the gradient magnitude scales like 1/τ k and (ii) its direction is averaged over more positives, lowering variance across minibatches. This is the sense in which band-specific temperatures yield smoother gradients. It is especially helpful in high-cost bands that otherwise have few positives and highly variable similarities. Without band-specific temperatures, the router can exhibit oscillatory updates on hard prompts (rare positives dominate, then vanish), slowing convergence and encouraging over-use of cheap experts. Empirically, we observed that bands and band-specific temperatures are important (Table 7)
this section cite: []

Section: C.4 Dense Human Annotations
We considered using human annotations but intentionally avoided them: model pools change quickly, so adding/replacing experts would require fresh labels that are costly and often unavailable. Instead, we train with sparse correctness and cost signals, which remain portable across experts. If dense feedback is available, it could help in several ways:
• Positive sets P (i) with preference structure. Replace binary "correct expert" labels with pairwise preferences (cheap-and-good ≻ expensive-and-similar ≻ clearly wrong), yielding band-aware positives and margin constraints. This can be implemented by expanding P (i) and adding a lightweight pairwise ranking (DPO-style) regularizer within each cost band.
• Difficulty-aware reweighting. Use human "hardness" scores to upweight rare/hard prompts when computing the contrastive loss, especially in higher cost bands. this could balance the effective sample sizes across easy vs. hard (and cheap vs. expensive band) cases so the gradient isn't dominated by the abundant, easy examples.
• Band calibration. We can ask users how much quality they're willing to trade for a lower cost, then use that to set the cost bands and the penalty for picking more expensive models, so the router's choices match what users actually prefer.
this section cite: []

Section: D Experiments

this section cite: []

Section: D.1 Experimental Settings
Baselines We compare our proposed method against a comprehensive set of baselines designed to capture key routing strategies and their trade-offs. Specifically, we include UMR [40], a recent state-of-the-art technique that clusters prompt embeddings to route queries to LLM pools efficiently; Thompson Sampling [48,2], which frames routing as a bandit exploration-exploitation problem to balance cost and accuracy dynamically; Pareto-optimal routing [34], a strategy that selects models by explicitly considering the cost-accuracy Pareto frontier; and two extreme baselines-Random, which selects models uniformly at random to represent naive routing without intelligent selection, and Oracle (Clairvoyant Upper-Bound [40]), which always selects the most accurate model at the lowest possible cost and thus represents a theoretical performance ceiling. Additionally, we evaluate against parametric gating methods (Parametric Softmax Router) inspired by classical mixtureof-experts architectures [62] and SoftMoE, which models router decisions via differentiable soft gating functions [64]. Collectively, these baselines enable us to rigorously assess whether our contrastive routing approach delivers meaningful improvements in performance, cost-efficiency, and generalization capabilities relative to existing strategies.
this section cite: ['b39', 'b47', 'b1', 'b33', 'b39', 'b61', 'b63']

Section: Datasets, Benchmarks, and Evaluation
We train our router and evaluate it on three datasets: EmbedLLM [105], MixInstruct [39], and RouterBench [34]. For EmbedLLM and MixInstruct, we sample 192 probes from their respective validation sets. Each probe is processed to extract logit-based descriptors by capturing the top K = 256 tokens over a horizon of T = 10 tokens (Equation ( 2)), resulting in a 256-dimensional vector per model. For RouterBench, we sample 192 probes from its training set, ensuring these probes are excluded from the training data used for the contrastive router. We compute perplexity-based descriptors on RouterBench and use GPT-2 [66]. On both EmbedLLM and RouterBench, we use binary accuracy as the per-sample evaluation metric, meaning an LLM response is classified strictly as correct or incorrect. For MixInstruct, we employ exponentiated BARTScore [97] as the evaluation metric, following the approach in prior work [40,39].
this section cite: ['b104', 'b38', 'b33', 'b65', 'b96', 'b39', 'b38']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
Answer: [Yes] Justification: We provide theoretical analysis in Section 5 and empirical results in Section 4.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes] Justification: Provided in Section 7
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

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: We provide a theoretical analysis with proofs and assumptions in Section 5.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: All experimental details are provided in Section 4 and D.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results. 5. Open access to data and code Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: All datasets used are open source. We will provide the code for our experiments after paper decision is available.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: All experimental details are provided in Section 4 and D.1.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [No] Justification: We could not provide error bars deviation due to computational costs. No other prior work and baselines in our work do this either.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).
• It should be clear whether the error bar is the standard deviation or the standard error of the mean.
• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
• If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.
this section cite: []

Section: Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?
Answer: [Yes] Justification: All experimental details are provided in Section 4 and D.1. They include this information.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
this section cite: []

Section: Answer: [Yes]
Justification: There are not ethical concerns that we know of.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: A discussion of broader impact is provided in Section 7.
Guidelines:
• The answer NA means that there is no societal impact of the work performed.
• If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).
this section cite: []

Section: Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?
Answer: [NA] Justification: The paper poses no such risks.
Guidelines:
• The answer NA means that the paper poses no such risks.
• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All original owners of assets have been properly cited.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators. 13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets? Answer: [NA] Justification: We do not release new assets.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
14. Crowdsourcing and research with human subjects Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector. 15. Institutional review board (IRB) approvals or equivalent for research with human subjects Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained? Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines: • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review. 16. Declaration of LLM usage Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required. Answer: [NA] Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components. Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Automix: Automatically mixing language models Year: (2024)
Ref_id:b1 Title: Analysis of thompson sampling for the multi-armed bandit problem Year: (2012)
Ref_id:b2 Title: Finite-time analysis of the multiarmed bandit problem Year: (2002)
Ref_id:b3 Title: Classification with a reject option using a hinge loss Year: (2008)
Ref_id:b4 Title: Lectures on the nearest neighbor method Year: (2015)
Ref_id:b5 Title: Language models are few-shot learners Year: (2020)
Ref_id:b6 Title: Flextron: Many-in-one flexible large language model Year: (2024-07)
Ref_id:b7 Title: MEDUSA: Simple LLM inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b8 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b9 Title: How to use large language models while reducing cost and improving performance Year: (2023)
Ref_id:b10 Title: RouterDC: Query-based router by dual contrastive learning for assembling large language models Year: (2024)
Ref_id:b11 Title: On optimum recognition error and reject tradeoff Year: (1970)
Ref_id:b12 Title: Unified scaling laws for routed language models Year: (2022)
Ref_id:b13 Title: Learning with rejection Year: (2016)
Ref_id:b14 Title: Nearest neighbor pattern classification Year: (1967)
Ref_id:b15 Title: Domain adaptation for robust model routing Year: (2024)
Ref_id:b16 Title: Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model Year: (2024)
Ref_id:b17 Title: MatFormer: Nested transformer for elastic inference Year: (2024)
Ref_id:b18 Title: Hybrid LLM: Cost-efficient and quality-aware query routing Year: (2024)
Ref_id:b19 Title: Mastering text, code and math simultaneously via fusing highly specialized language models Year: (2024)
Ref_id:b20 Title:  Year: (2024)
Ref_id:b21 Title: An introduction to the bootstrap Year: (1994)
Ref_id:b22 Title: The foundations of cost-sensitive learning Year: (2001)
Ref_id:b23 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b24 Title: GraphRouter: A graph-based router for llm selections Year: (2024)
Ref_id:b25 Title: Scaling laws for sparselyconnected foundation models Year: (2023)
Ref_id:b26 Title: SelectiveNet: A deep neural network with an integrated reject option Year: (2019-06)
Ref_id:b27 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b28 Title: Smoothie: Label free language model routing Year: (2024)
Ref_id:b29 Title: Language model cascades: Token-level uncertainty and beyond Year: (2024)
Ref_id:b30 Title: Spotting llms with binoculars: Zero-shot detection of machine-generated text Year: (2024)
Ref_id:b31 Title: Tryage: Real-time, intelligent routing of user prompts to large language models Year: (2023)
Ref_id:b32 Title: How good are GPT models at machine translation? a comprehensive evaluation Year: (2023)
Ref_id:b33 Title: RouterBench: A benchmark for multi-LLM routing system Year: (2024)
Ref_id:b34 Title: Editing models with task arithmetic Year: (2023)
Ref_id:b35 Title: Adaptive mixtures of local experts Year: (1991)
Ref_id:b36 Title: Openai o1 system card Year: (2024)
Ref_id:b37 Title:  Year: (2024)
Ref_id:b38 Title: Llm-blender: Ensembling large language models with pairwise ranking and generative fusion Year: (2023-07)
Ref_id:b39 Title: Universal model routing for efficient llm inference Year: (2025)
Ref_id:b40 Title: Hierarchical mixtures of experts and the EM algorithm Year: (1993)
Ref_id:b41 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b42 Title: Orchestrallm: Efficient orchestration of language models for dialogue state tracking Year: (2023)
Ref_id:b43 Title: Tahira Naseem Asim Munawar, Radu Florian, Salim Roukos, and Ramón Fernandez Astudillo. Ensemble-instruct: Generating instruction-tuning data with a heterogeneous mixture of lms Year: (2023)
Ref_id:b44 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b45 Title: Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions Year: (2024)
Ref_id:b46 Title: (mis) fitting: A survey of scaling laws Year: (2025)
Ref_id:b47 Title: LLM Bandit: Cost-efficient llm generation via preference-conditioned dynamic routing Year: (2025)
Ref_id:b48 Title: Optllm: Optimal assignment of queries to large language models Year: (2024)
Ref_id:b49 Title: Merge, ensemble, and cooperate! a survey on collaborative strategies in the era of large language models Year: (2024)
Ref_id:b50 Title: Routing to the expert: Efficient reward-guided ensemble of large language models Year: (2023)
Ref_id:b51 Title: Blending is all you need: Cheaper, better alternative to trillion-parameters llm Year: (2024)
Ref_id:b52 Title: Distance-based classification with lipschitz functions Year: (2004-06)
Ref_id:b53 Title: Automix: Automatically mixing language models Year: (2023)
Ref_id:b54 Title: Predict responsibly: Improving fairness and accuracy by learning to defer Year: (2018)
Ref_id:b55 Title: Merging models with fisher-weighted averaging Year: (2022)
Ref_id:b56 Title: Note on the sampling error of the difference between correlated proportions or percentages Year: (1947)
Ref_id:b57 Title: Context dependent recurrent neural network language model Year: (2012)
Ref_id:b58 Title: Adversarial robustness in two-stage learning-to-defer: Algorithms and guarantees Year: (2025)
Ref_id:b59 Title: Learning to reject for balanced error and beyond Year: (2024)
Ref_id:b60 Title: Metallm: A high-performant and cost-efficient dynamic framework for wrapping llms Year: (2024)
Ref_id:b61 Title: RouteLLM: Learning to route LLMs with preference data Year: (2024)
Ref_id:b62 Title: Mechanistic design and scaling of hybrid architectures Year: (2024)
Ref_id:b63 Title: From sparse to soft mixtures of experts Year: (2024)
Ref_id:b64 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b65 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b66 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2023)
Ref_id:b67 Title: Summareranker: A multi-task mixture-of-experts re-ranking framework for abstractive summarization Year: (2022)
Ref_id:b68 Title: Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context Year: (2024)
Ref_id:b69 Title: Sentence-bert: Sentence embeddings using siamese bert-networks Year: ()
Ref_id:b70 Title: Fly-swat or cannon? cost-effective language model choice via meta-modeling Year: (2023)
Ref_id:b71 Title: Confident adaptive language modeling Year: (2022)
Ref_id:b72 Title: BLEURT: Learning robust metrics for text generation Year: (2020)
Ref_id:b73 Title: Rerouting LLM routers Year: (2025)
Ref_id:b74 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017-04-24)
Ref_id:b75 Title: Large language model routing with benchmark datasets Year: (2023)
Ref_id:b76 Title: Blockwise parallel decoding for deep autoregressive models Year: (2018)
Ref_id:b77 Title: Zipit! merging models from different tasks without training Year: (2024)
Ref_id:b78 Title: Consistent nonparametric regression. The annals of statistics Year: (1977)
Ref_id:b79 Title: TensorOpera router: A multi-model router for efficient LLM inference Year: (2024)
Ref_id:b80 Title: Branch-train-mix: Mixing expert llms into a mixture-of-experts llm Year: (2024)
Ref_id:b81 Title: Idiosyncrasies in large language models Year: (2025)
Ref_id:b82 Title: Spectr: Fast speculative decoding via optimal transport Year: (2024)
Ref_id:b83 Title: BranchyNet: Fast inference via early exiting from deep neural networks Year: (2016)
Ref_id:b84 Title: On the likelihood that one unknown probability exceeds another in view of the evidence of two samples Year: (1933)
Ref_id:b85 Title: LLaMA: Open and efficient foundation language models Year: (2023)
Ref_id:b86 Title: Representation learning with contrastive predictive coding Year: (2019)
Ref_id:b87 Title: Fly-swat or cannon? Cost-effective language model choice via meta-modeling Year: (2024)
Ref_id:b88 Title: Improving text embeddings with large language models Year: (2024)
Ref_id:b89 Title: Self-consistency improves chain of thought reasoning in language models Year: (2022)
Ref_id:b90 Title: Tabi: An efficient multi-level inference system for large language models Year: (2023)
Ref_id:b91 Title: Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time Year: (2022)
Ref_id:b92 Title: Disentangled contrastive learning for social recommendation Year: (2022)
Ref_id:b93 Title: Ties-merging: Resolving interference when merging models Year: (2023)
Ref_id:b94 Title: Adamerging: Adaptive model merging for multi-task learning Year: ()
Ref_id:b95 Title: Tree of thoughts: Deliberate problem solving with large language models. Neural Information Processing Systems Year: (2023)
Ref_id:b96 Title: Bartscore: Evaluating generated text as text generation Year: (2021)
Ref_id:b97 Title: Large language model cascades with mixture of thoughts representations for cost-efficient reasoning Year: (2023)
Ref_id:b98 Title: Bertscore: Evaluating text generation with bert Year: (2020)
Ref_id:b99 Title: Recommender systems in the era of large language models (llms) Year: (2024)
Ref_id:b100 Title: Lo-raRetriever: Input-aware LoRA retrieval and composition for mixed tasks in the wild Year: (2024-08)
Ref_id:b101 Title: Least-to-most prompting enables complex reasoning in large language models Year: (2023)
Ref_id:b102 Title: BERT loses patience: Fast and robust inference with early exit Year: ()
Ref_id:b103 Title: Mixture-of-experts with expert choice routing Year: (2022)
Ref_id:b104 Title: Learning compact representations of large language models Year: (2024)
