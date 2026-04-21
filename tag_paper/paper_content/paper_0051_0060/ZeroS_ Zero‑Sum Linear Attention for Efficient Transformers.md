Title: ZeroS: Zero-Sum Linear Attention for Efficient Transformers
Abstract: Linear attention methods offer Transformers O(N ) complexity but typically underperform standard softmax attention. We identify two fundamental limitations affecting these approaches: the restriction to convex combinations that only permits additive information blending, and uniform accumulated weight bias that dilutes attention in long contexts. We propose Zero-Sum Linear Attention (ZeroS), which addresses these limitations by removing the constant zero-order term 1/t and reweighting the remaining zero-sum softmax residuals. This modification creates mathematically stable weights, enabling both positive and negative values and allowing a single attention layer to perform contrastive operations. While maintaining O(N ) complexity, ZeroS theoretically expands the set of representable functions compared to convex combinations. Empirically, it matches or exceeds standard softmax attention across various sequence modeling benchmarks. The code implementation is available at this link.

Section: Introduction
The Transformer architecture [1] has revolutionized sequence modeling across NLP, vision, speech, and reinforcement learning [2][3][4][5][6][7]. While its self-attention mechanism offers exceptional modeling flexibility, the quadratic O(N 2 ) complexity in both time and memory with sequence length N limits its efficient implementation to long-context scenarios [8,9]. Researchers have developed numerous linear-time attention mechanisms [8,[10][11][12][13][14] that preserve Transformer's strengths while scaling to longer sequences. Approaches include sparse attention patterns [15][16][17], kernel methods [8,13,14], low-rank approximation [18,19], and efficient factorizations [20,21]. Despite reducing from O(N 2 ) to O(N ), these variants often underperform standard softmax attention, raising the question: Why do linear approximations save computation but sacrifice accuracy? Recent efforts to bridge this gap typically: 1) hybridize linear attention with local quadratic windows [22,23], 2) learn softmax matrix low-rank projections [19,24], or 3) sharpen linear kernels through normalization and gating [11,12,25]. While offer incremental gains, these approaches often compromise O(N ) efficiency, rely on task-specific hyperparameters, or introduce instabilities, limiting their practical use.
In this paper, we identify two fundamental limitations affecting linear and even softmax attention: 1) Bottleneck of convex combination [26][27][28]: softmax attention produces convex combinations of value vectors, with linear attention also aiming to achieve this primarily for numerical stability. However, these combinations can only blend information additively, unable to express subtractive or contrastive operations directly, forcing models to use multiple layers even for simple differencing tasks. 2) Uniform weight bias and attention dilution [11,12,29]: In long contexts, attention mechanisms incorporate a roughly uniform 1  N component in their weight expansion, introducing a persistent averaging effect that weakens focused attention and limits modeling of complex patterns. These limitations stem from the Taylor expansion exp(q • k) = 1 + ⟨q, k⟩ + 1 2 ⟨q, k⟩ 2 + . . . , where the constant zero-order term enforces non-negativity for stability but creates an average-pooling bias that diminishes high-order token interactions. Rather than designing complex kernels to approximate softmax while preserving the constant term, we propose a simpler solution: remove it. Subtracting the uniform component creates naturally zero-sum weights that permit both positive and negative values, enabling contrastive updates and sharper attention distributions while maintaining stability.
From this insight, we introduce ZeroS (Zero-Sum linear attention), achieving linear complexity while matching or exceeding quadratic softmax attention performance through three key elements: 1) Zero-order subtraction: removing the uniform 1/t term from each softmax row to create stable zero-sum weights; 2) Radial-angular decoupling: separating magnitude from direction by applying learned gates to first-order (linear) and higher-order (non-linear) softmax residuals, then reintroducing signed cos θ terms to restore directional effects; 3) Linear-time implementation: using separable logits and gating for the reweighted zero-sum softmax, combined with linearizable angular computations via prefix sums, maintaining O(N d 2 ) runtime and O(d 2 ) memory.
Our contributions include: 1) Identifying why the uniform zero-order softmax term limits attention mechanisms and demonstrating that its removal is safe and beneficial. 2) Developing Zero-Sum Linear Attention (ZeroS), a linear-time attention supporting negative weights with theoretical stability independent of sequence length. 3) Proving ZeroS offers greater expressivity than convex combinations while maintaining numerical stability. 4) Demonstrating that ZeroS matches or exceeds standard softmax attention on various benchmarks while maintaining linear time complexity.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b7', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b7', 'b12', 'b13', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b18', 'b23', 'b10', 'b11', 'b24', 'b25', 'b26', 'b27', 'b10', 'b11', 'b28']

Section: Background

this section cite: []

Section: Preliminaries: Attention Mechanisms
We consider an input token sequence of length N , represented by the feature matrix X ∈ R N ×d , where each row x t ∈ R 1×d is the embedding at time step t. With Q = XW q , K = XW k , V = XW v , an autoregressive (causal) single-head attention layer can be written in its matrix form as
Attn(X) = σ M ⊙ (QK ⊤ ) V W o , X ← X + Attn LN(X) ,
where W q , W k , W v , W o ∈ R d×d are learned projections, LN(•) denotes layer normalization, and M ∈ R N ×N is the causal mask with M ij = 1{i ≥ j} -∞ • 1{i < j}, ensuring each position attends only to itself and the past. When σ is the row-wise softmax with a 1/ √ d factor, this represents standard self-attention with O(N 2 ) complexity; replacing σ by the linearized kernels yields the linear attention variants that can be computed in O(N ) [8,14]. Omitting the causal mask M reverts this to encoder-only attention, attending to all pairs of positions.
this section cite: ['b7', 'b13']

Section: Recurrent Form Attention admits an equivalent step-by-step formulation. At time t, let q
t = x t W q , k t = x t W k , v t = x t W v . Then the output o t ∈ R 1×d is o t = t i=1 σ(qt,ki) vi t i=1 σ(qt,ki) where σ(q, k) = exp(q k ⊤ / √ d) for vanilla attention.
By choosing a kernel feature map ϕ(•) such that σ(q t , k i ) = ϕ(q t ) ϕ(k i ) ⊤ , the summations can be rearranged to maintain only the d × d hidden state t i=1 ϕ(k i ) ⊤ v i , avoiding the full N × N matrix QK ⊤ . This yields the linear attention formulation:
o t = ϕ(qt) t i=1 ϕ(ki) ⊤ vi ϕ(qt) t i=1 ϕ(ki) ⊤ .
Replacing the summation limit t with N converts this from the decoder-only autoregression into a encoder-only global recurrence, summing over all positions.
this section cite: []

Section: The intuition from existing linear attention research
We begin with insights from previous research on linear attention to introduce two key elements of our ZeroS structure: 1) radial-angular decoupling, and 2) zero-sum reweighted softmax. In softmax attention, each value vector v i is assigned a weight exp(qtki) t i=1 exp(qtki) , forming a convex combination that ensures numerical stability by keeping outputs within the convex hull of {v i } [26,27,30]. Linear attention variants attempt to approximate this using weights of linearized kernel form ϕ(qt)ϕ(ki) t i=1 ϕ(qt)ϕ(ki) [8,11,14]. However, without constraining the sign of ϕ(q t )ϕ(k i ), this reduces to an affine combination that lacks the stability-ensuring bounds of convexity. While researchers have addressed this using non-negative feature maps like 1+ELU and ReLU [8,12,31,32], these stability-ensuring modifications still underperform compared to standard softmax attention [8,11,14].
this section cite: ['b25', 'b26', 'b29', 'b7', 'b10', 'b13', 'b7', 'b11', 'b30', 'b31', 'b7', 'b10', 'b13']

Section: Coupling Interaction Between Radial and Angular Components
In softmax attention, the core weight term exp(∥q t ∥∥k i ∥ cos θ) is controlled by both vector magnitudes and their angle θ. Crucially, when cosine flips from positive to negative, large positive values transform into very small ones, with step t and i highly coupled within the exponential. In contrast, linear attention applies nonlinear mappings ϕ(•) to query and key [8,12,33], calculating ∥ϕ(q t )∥∥ϕ(k i )∥ cos θ ′ . Since these mappings yield only positive values, angles between vectors become restricted to less than 90 degrees, and the angular representation loses its flipping effect-cosine values merely serve as smooth gating signals between (0, 1). Previous research shows minimal performance changes when replacing softmax with sigmoid, ReLU, or similar functions [34][35][36][37][38][39], indicating that softmax attention's performance derives from modeling coupled angular and magnitude of (t, i) pairs rather than from the exponential property itself. Therefore, when constructing linear attention, we should reimplement these complex interactions rather than attempting to approximate softmax or merely mimicking an inner product.
this section cite: ['b7', 'b11', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38']

Section: Convexity of Sum-to-One Weights
Under this perspective, we revisit the convex combination in softmax attention, which primarily serves numerical stability by preserving norm regardless of sequence length. As weights become more uniform, output norm expectation decreases at approximately 1/ √ t with sequence length t (assuming zero-mean vectors). However, these strictly positive weights mean input signals v i can only contribute additively to outputs. In linear attention, without methods to suppress historical weights, this accumulation leads to attention dilution [11,12,29], where uniform signals increasingly dominate as sequence length grows. While some approaches address this using local windows or convolutional methods [9,31,40], these represent engineering solutions rather than resolving fundamental limitations of positive weights. Studies [27,[41][42][43] show that with softmax weights, a single attention layer cannot express differential or contrastive operations (even with just two tokens). The strictly positive convex combination inherently constrains ability to compress complex operations, limiting parameter efficiency. To enable more flexible parameterization with negative values, we must maintain numerical stability without relying on convex combinations' norm-preserving property while satisfying linear-time requirements.
Flexible Weighting in Related Works Implementing both the angular flipping effect and expressiveness requires numerically stable modeling of negative weights. Previous research [28,44] demonstrated that negative weights improve model performance, while Differential transformer [45] showed benefits from differencing two attention matrices to obtain flexible weights. In linear attention, operations that reduce or delete historical state matrix elements outperform simple accumulation approaches [9,14,31,46,47]. In the following sections, we will show that our ZeroS method constructs zero-sum weights based on softmax, improving performance while maintaining numerical stability compared to both standard and linear attention variants. Compared to previous linear attention, ZeroS enables more effective control of radial weights and decoupled angular components in (t, i) pairs from step t information.
this section cite: ['b10', 'b11', 'b28', 'b8', 'b30', 'b39', 'b26', 'b40', 'b41', 'b42', 'b27', 'b43', 'b44', 'b8', 'b13', 'b30', 'b45', 'b46']

Section: Methodology
In this section,we demonstrate that using softmax residual terms with zero-sum weights (eliminating zero-order terms) and decoupling radial-angular components in linear attention achieves three key objectives: 1) enabling numerically stable negative weights in a single attention layer for expressing differential and contrastive operations, 2) capturing the essential length-angle interactions in attention weights that allow positive-negative flipping effects, and 3) permitting the current step t to effectively influence shareable accumulated weights while maintaining linear time complexity. The overall architecture of the final ZeroS block introduced in this section is shown in Fig. 1.
this section cite: []

Section: The Expansion of Softmax Function
Recent research has attempted to approximate softmax using Taylor expansions [48][49][50][51]. For input scalars {s i } t i=1 , with s = 1 t t j=1 s j and δ i = s i -s, the second-order Taylor expansion is:
softmax(s i ) ≈ 1 t + 1 t δ i + 1 2t δ 2 i - 1 t t j=1 δ 2 j + O(∥s∥ 3 ).
The zero-order term 1 t ensures i softmax(s i ) = 1, while first-order terms reflect linear response, and higher-order terms capture nonlinear interactions and competitive relationships between weights. Computing second-order terms based on s t,i = q t k ⊤ i would require O(d 3 ) complexity [49], making them impractical. Our approach differs: we use logits that depend only on step i, calculate full
L2 Normalize Deviation Logit Full Softmax 0th-order Softmax 1st-order Softmax Higher-order Softmax Reweighted Zero-sum Softmax Radial Angular Value Subtract Sigmoid + 1 2 1 2 3 3 dot Layer Norm + --× × (RoPE) t i t i i Smoothed Mean Dot Logit 2 (c) The overall architecture of ZeroS (Zero-Sum Linear Attention) (b) Reweighted zero-sum softmax block (a) Calculation of the deviation logits Linear Attn ✓ ZeroS ⊕ Figure 1: Illustration of the zero-sum linear attention block, including the computation of deviation logits and the reweighted zero-sum softmax operation softmax, zero and first-order terms, derive higher-order terms through their differentiation, and employ t-step-dependent gating factors to achieve interaction between (t, i) pairs at different orders.
The zero-order baseline primarily provides accumulated magnitude measurement, contributing 1 √ tlevel norm reduction and convexity properties. However, it enables no interaction between scores. Eliminating this term creates zero-sum residual weights with both positive and negative values reflecting interaction strength. While full softmax encodes higher-order competitive effects only through positive weight magnitudes, zero-sum residual weights directly express these relationships between vectors based on the positive and negative weights, emphasizing contrastive components. Proposition 3.1 (Convex vs. Zero-Sum Span). Let {v i } t i=1 ⊂ R d , and write C = { i α i v i :
α i ≥ 0, i α i = 1}, Z = { i w i v i : i w i = 0}
, where we denote the (t -1)-simplex by ∆ t-1 = {α ∈ R t :
α i ≥ 0, i α i = 1}. Then, letting v avg = 1 t i v i , { i α i v i -v avg : α ∈ ∆ t-1 } ⊊ { i w i v i :
i w i = 0}, i.e. the zero-sum span of {v i -v avg } strictly contains the deviations achievable by convex weights, with strictness whenever the v i are not all identical.
this section cite: ['b47', 'b48', 'b49', 'b50', 'b48']

Section: Corollary 3.2 (Expressive Gain of Zero-Sum Attention). In a residual block x
t → x t + i w i v i , softmax weights w i = α i yield head deviations in { i α i v i -v avg : α ∈ ∆ t-1 }; zero-order subtraction w i = α i -1 t yields head deviations in { i w i v i : i w i = 0}. Since { i α i v i - v avg } ⊊ { i w i v i : i w i = 0}
, zero-sum attention enlarges the set of deviation vectors the head can produce (and hence its expressivity), only without the uniform average direction.
Zero-sum weights can express more complex interactions after removing the zero-order term, with expressivity reduction only in the orthogonal direction of v avg . This direction typically represents the lowest-cost basis since it requires only average pooling. We can recover this capability through multiple attention heads and layer stacking. To strictly ensure this direction is not lost, our implementation retains the zero-order term in the first layer, removing it in subsequent layers as described below.
this section cite: []

Section: Reweighted Zero-sum Softmax
We define the reweighted zero-sum softmax operation. For logit input s t,i at step t, we compute: st = 1 t t j=1 s t,j , δ t,i = s t,i -st . Subtracting zero-order (1/t) and first-order (δ t,i /t) terms from softmax yields the residual:
ε t,i = exp(s t,i ) t i=1 exp(s t,i ) - 1 t - δ t,i t = O(∥δ∥ 2 ),
where i ε t,i = 0 and i δ t,i /t = 0. We gate these components using learned scalars σ 1 t = sigmoid(g 1 t ) and σ h t = sigmoid(g h t ), defining zero-sum weights:
w t,i = σ 1 t δ t,i t + σ h t ε t,i , t i=1 w t,i = 0.
This form assigns two gating weights: one for the first-order orthogonal direction and another for all directions of second-order and above. For the first attention layer, we can optionally preserve the zero-order term using σ 0 t = tanh(g 0 t ), giving w ′ t,i = σ 0 t 1 t + σ 1 t δt,i t + σ h t ε t,i , though experiments show this has minimal impact across most tasks.
this section cite: []

Section: Remark.
A key advantage of this formulation for linear-time attention is that even with logits s t,i = s i that are independent of t, we can still control interactions of different orders in the final weights w t,i through the t-step gating mechanism σ (•) t across orthogonal directions from softmax expansion. This gate reweighting approach w t,i = σ 1 t δi t + σ h t ε i effectively replaces the traditional linearization that decomposes exp(q t k i ) into ϕ(q t )ϕ(k i ).
this section cite: []

Section: Proposition 3.3 (Preservation of Affine Hull and Expressivity).
Let {v i } t i=1 ⊂ R d and write v avg = 1 t t i=1 v i , ∆ i = v i -v avg .
A single head with full softmax (or full reweighted softmax with the zero-order term kept) can produce any point in the affine hull
Aff{v 1 , . . . , v t } = v avg + t i=1 α i ∆ i : t i=1 α i = 1 .
A single head without the zero-order term (i.e. zero-sum weights) can produce any point in the linear span
Span{∆ 1 , . . . , ∆ t } = t i=1 w i ∆ i : t i=1 w i = 0 .
Therefore, if you use one head (or one layer) that retains the zero-order term and then stack one or more heads (layers) that subtract it, the Minkowski sum of their reachable sets is exactly
Aff{v i } + Span{∆ i } = Aff{v i }.
In other words, after the first full attention layer, it already cover the entire affine hull, and the subsequent zero-sum attentions do not shrink that. The overall network can still express any affine combination of the v i .
this section cite: []

Section: Residual Stream Alignment
When value vectors {v i } are centered and i.i.d., zero-sum attention produces o t = t i=1 w t,i v i with i w t,i = 0 and E[o t ] = 0. This aligns with decoder-only Transformer's residual stream ideology where x t ← x t + Attn(x t ) should provide pure updates without constant bias. Subtracting the zero-order term naturally centers these residuals, improving training stability.
We now show that the reweighted zero-sum softmax achieves the same level of numerical stability as the original softmax. Lemma 3.4 (Numerical Stability of Zero-Sum Softmax). Let w t,i be the reweighted zero-sum softmax weights with i w t,i = 0, and assume each value vector satisfies ∥v i ∥ ≤ B. Then for any step t,
t i=1 w t,i v i ≤ max i |w t,i | t i=1 ∥v i ∥ ≤ B t max i |w t,i |. Moreover, since |w t,i | ≤ max 1 t , |δt,i| t , |ρt,i|2t
and δ, ρ = O(1) under bounded logits, we have max i |w t,i | = O(1/t) and hence i w t,i v i = O(B), independent of t. With controllable logits generation methods independent of t, such as the scaled dot-product in original softmax attention, the numerical stability of the above method remains well-controlled. This allows us to employ zero-sum weights that permit negative values while still achieving numerically stable outputs, even without the norm-preserving property of convex combinations. Proposition 3.5 (Uniform Lipschitz Bound of Zero-Sum Softmax with decay factor 1/ √ t). Assume each value vector satisfies ∥v i ∥ ≤ B, each pre-softmax logit s t,i (x) ∈ [-S, S] is L s -Lipschitz in the input x, each residual weight w t,i (x) obeys the scaling| w t,i (x) -w t,i (x ′ )| ≤ Lw t ∥x -x ′ ∥, for some constant L w depending only on S and the sigmoid gates. Let the head output be o t (
x) = 1 √ t t i=1 w t,i (x) v i , t i=1 w t,i (x) = 0. Then for any two inputs x, x ′ , o t (x) -o t (x ′ ) ≤ B L w √ t ∥x -x ′ ∥.
The zero-sum update is ℓ 2 -Lipschitz in its inputs with constant O(1/ √ t), ensuring stable gradients and activations independent of sequence length.
this section cite: []

Section: This proposition introduces a 1/
√ t decay factor that ensures reweighted zero-sum softmax maintains variance reduction similar to convex combinations, promoting training stability. However, since linear attention methods typically apply Layer Normalization to control output variance, LayerNorm effectively supersedes this factor and is sufficient to ensure gradient stability during training.
this section cite: []

Section: Reweighted Zero-sum Softmax in Linear Time
To achieve linear-time computation, we simplify logits from s t,i to s i by removing t-dependency. While we could use basic forms like
s i = x i W d×1 s or quadratic forms s i = x i W s W ⊤ s x ⊤ i /d
to emulate dot-products, we instead propose a design with a more meaningful representation.
We want these logits to express the deviation of step i relative to previous steps. We calculate the negative inner product between each step's vector u i = x i W u and its cumulative average. For better assessment of initial steps, we introduce trainable parameters µ ∈ R 1×d and τ ∈ R as a smoothing prior, calculating deviation logits s i as:
s i = - 1 √ d u i ū⊤ i , where ūi = e τ µ + i j=1 u j e τ + i
Let r t,i represent the final computed reweighted softmax result:
δ t,i = s i -st , ε t,i = exp(s i ) t i=1 exp(s i ) - 1 t - δ t,i t , r t,i = σ 1 t δ t,i t + σ h t ε t,i = σ h t t i=1 exp(s i ) exp(s i ) + (σ 1 t -σ h t ) t (s i -st ) - σ h t t
where
σ 1 t = sigmoid(x t W 1 g ), σ h t = sigmoid(x t W h g ), W 1 g , W h g ∈ R d×1
. The terms dependent on t and i are effectively separated, enabling linear-time computation through prefix sums.
this section cite: []

Section: ZeroS Linear Attention: Interaction Between Radial and Angular Components
The reweighted zero-sum softmax provides strong foundations for linear attention by yielding numerically stable weights (including negative values) with computational simplicity while enabling high-order (t, i) interactions in token mixing. We leverage linear-time logit inputs that depend only on step i and implement effects on different softmax orders through step t gating.
However, our earlier discussion showed that the angle-flipping effect in softmax attention's exp(∥q t ∥∥k i ∥ cos θ) significantly impacts final weights. While reweighted zero-sum softmax effectively models length interactions through i-step logits and t-step gating, it lacks control over directional influence when measuring vector differences in (t, i) pairs. Since zero sum weights provide inherent stability, no longer need to place cos θ in the denominator normalizer, we can directly multiply the angular component (cos θ) with the reweighted softmax radial component without positivity constraints. This approach enables seamless integration with rotary positional embedding (RoPE) [52], making the angle term's role in measuring relative distance more explicit.
this section cite: ['b51']

Section: Zero-Sum Linear Attention (ZeroS)
We use normalized vectors ki = k i /∥k i ∥ and qt = q t /∥q t ∥, with r t,i as the radial component and cos θ as the angular component. ZeroS produces the output:
o t = t i=1 r t,i cos θ v i , cos θ = qt k⊤ i
With RoPE's block-diagonal rotary matrix applied, the angular term becomes cos θ ′ = qt R t-i k⊤ i . Both r t,i and cos θ are centered values, preserving zero-sum properties in the weights r t,i cos θ. Though not strictly positive (unlike traditional radial components), r t,i captures magnitude effects from step i, reflecting length-related interactions between (t, i) pairs. Linear-Time Scan With logits that depend only on step i (e.g.
s i = -1 √ d u i ū⊤ i with u i = x i W u and ūi = e τ µ+ i j=1 uj e τ +i
), the radial weight at time t can be decomposed into the full softmax term, the 0th-order baseline, and the 1st-order term:
e si t j=1 e sj Full , 1 t 0th , 1 t s i - 1 t 2 t j=1 s j 1st
. Hence the higherorder zero-sum residual is Full -0th -1st. We realize ZeroS by gating the first-order zero-sum and the higher-order residual with σ 1 t = sigmoid(x t W 1 g ) and σ h t = sigmoid(x t W h g ), and optionally in the first layer retaining the 0th-order baseline σ 0 t = sigmoid(x t W 0 g ) (with fixed σ 0 t = 0 by default). Using normalized directions qt = q t /∥q t ∥ and ki = k i /∥k i ∥, we maintain the following prefix scans at step t:
E t = t i=1 e si , P t = t i=1 s i , F t = t i=1 e si k⊤ i v i , G t = t i=1 s i k⊤ i v i , H t = t i=1 k⊤ i v i .
The output is then a gated activation of these scans by the current step's angular vector qt :
o t = qt σ h t 1 E t F t - 1 t G t + P t t 2 - 1 t H t σ h t (Full -0th -1st) + σ 1 t 1 t G t - P t t 2 H t σ 1 t (1st restore) + σ 0 t 1 t H t σ 0 t (optional 0th restore) = qt α t F t + β t G t + γ t H t ,
where
α t = σ h t E t , β t = σ 1 t -σ h t t , γ t = P t t 2 - 1 t σ h t - P t t 2 σ 1 t + σ 0 t t .
This scan keeps only O(d 2 ) state (F t , G t , H t ) and updates in O(d 2 ) per step, yielding overall O(N d 2 ) time and O(d 2 ) memory while implementing the zero-sum weighting. Moreover, our reweighted zero-sum approach can also be directly applied to standard softmax attention. See section A.1.6 for more details. ZeroS's zero-sum formulation enhances the attention layer's expressivity for complex operations, particularly evident in in-context learning tasks [37,53]. We evaluate both linear-time ZeroS and quadratic-time ZeroS-SM on recent in-context learning benchmarks, along with experiments on NLP, image, and time series tasks. In all experiments, we directly replaced the multi-head attention module with ZeroS under original benchmark settings, preserving all other components (MLP/GLU, embeddings, hyperparameters) to ensure strict alignment with previous standards.
We previously described the prefix-sum computation of autoregressive ZeroS. For the encoder-only ZeroS, the summation simply spans all timesteps. We use the causal version of ZeroS for all datasets except image modeling. We provide a more detailed description of the experimental datasets in Appendix A.5. For simplicity, we do not apply first-layer 0-th order term addition in our experiments.
this section cite: ['b36', 'b52']

Section: MAD
We evaluate ZeroS on the MAD benchmark [54], which tests sequence models on in-context tasks. As shown in Table 1, ZeroS outperforms other lineartime models (Hyena, Mamba, GLA, DeltaNet, LinAttn [9,14,31,55]), achieving performance closest to Transformer, while ZeroS-SM further improves upon Transformer's average score. Task-level analysis shows Ze-roS significantly outperforms LinAttn on In-Context and Noisy Recall tasks, supporting our hypothesis that zerosum weights enhance algorithmic abilities. However, on tasks like Compress and Memorize that rely less on complex representations, ZeroS provides minimal gains. Unlike DeltaNet, which actively deletes memory states, Ze-roS maintains strong memorization despite using negative weights, indicating that our zero-order modifications preserve sequence memory capacity.
this section cite: ['b53', 'b8', 'b13', 'b30', 'b54']

Section: MQAR
We follow the setup of [56] for the MQAR task, which evaluates models' ability to learn induction heads for in-context associative recall. Using the same hyperparameter sweep, Fig. 3 shows ZeroS performs comparably to vanilla attention across most configurations.
64 128 256 512 Model Dimension 0.0 0.2 0.4 0.6 0.8 1.0 Accuracy Sequence Length: 64 64 128 256 512 Model Dimension Sequence Length: 128 64 128 256 512 Model Dimension Sequence Length: 256 64 128 256 512 Model Dimension Sequence Length: 512 Attention Hyena RWKV H3 BaseConv HGRN1 HGRN2 ZeroS 6WHS /RVV 7UDLQLQJ/RVV /LQHDU7UDQVIRUPHU =HUR660 7UDQVIRUPHU =HUR6 *DWHG/LQHDU$WWHQWLRQ $)7 6WHS 9DOLGDWLRQ/RVV /LQHDU7UDQVIRUPHU =HUR660 7UDQVIRUPHU =HUR6 *DWHG/LQHDU$WWHQWLRQ $)7 Figure 4: Performance Evaluation of ZeroS on OWT2
RegBench We evaluate ZeroS on Reg-Bench [57] following the original experimental setup (Figure 2). RegBench tests models' ability to infer regular language structures from examples. ZeroS outperforms linear-time baselines including GLA, RetNet [58], and RWKV [59].
this section cite: ['b55', 'b56', 'b57', 'b58']

Section: Language Modeling
WikiText We conduct language modeling on WikiText-103 following [60]'s setup, with results in Table 2. ZeroS outperforms vanilla Transformer at this smaller scale, demonstrating its efficiency. ZeroS-SM yields further improvements, showing enhanced reasoning capability from the zero-order term removal.
this section cite: ['b59']

Section: OWT2
We evaluate ZeroS on OpenWebText2 (OWT2) [61] using a 12-layer, 768-dimensional GPT-2 architecture with various token layers (see Appendix A.5). Figure 4 shows ZeroS tracks much closer to vanilla Transformer than other linear methods like AFT [62] and GLA, while ZeroS-SM further improves upon vanilla Transformer performance.
Image Modeling Following HGRN2 [60], wee valuate ZeroS on ImageNet by replacing the DeiT-Tiny architecture's softmax atten- tion with our encoder-only implementation. As shown in Table 3, ZeroS outperforms previous 294 methods including TNN [63] and HGRN1 [64] under comparable parameter budgets.
Time Series Following the setup in [65], we evaluate ZeroS on time series forecasting tasks. ZeroS outperforms both efficient sequence models (GLA, AFT) and domain-specific approaches (iTransformer [66], PatchTST [67]) on most datasets.
this section cite: ['b60', 'b61', 'b59', 'b62', 'b63', 'b64', 'b65', 'b66']

Section: Ablation Studies

this section cite: []

Section: Conclusion and Limitation
We introduced Zero-Sum Linear Attention (ZeroS), addressing fundamental limitations of linear attention by removing the constant zero-order term from softmax and reweighting the resulting zero-sum residuals. Our approach enables higher-order token interactions while maintaining O(N) complexity, bridging the performance gap between linear and quadratic attention methods. Evaluations across diverse tasks show ZeroS matches or exceeds standard softmax attention while offering significant efficiency advantages, challenging the belief that expressivity-efficiency tradeoffs in attention mechanisms are inevitable.
As for the limitation, our research prioritizes improving attention's algorithmic expressivity rather than providing engineering optimizations like GPU acceleration implementations found in Mamba or GLA [9,14]. Also, our resource constraints prevented large-scale model training and evaluation on LLM benchmarks, which would involve numerous factors. This focused approach allowed us to precisely identify ZeroS's algorithmic improvements without requiring extensive engineering or computational resources that are typically needed for optimizing large benchmark metrics. Additionally, our evaluation of ZeroS primarily focuses on autoregressive tasks. Future work may explore its capabilities on non-causal tasks to further extend its applicability.
this section cite: ['b8', 'b13']

Section: References
Ref_id:b0 Title: Attention is all you need Year: (2017)
Ref_id:b1 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b2 Title: Language models are few-shot learners Year: (2020)
Ref_id:b3 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b4 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b5 Title: Speech-transformer: a no-recurrence sequence-tosequence model for speech recognition Year: (2018)
Ref_id:b6 Title: Decision transformer: Reinforcement learning via sequence modeling Year: (2021)
Ref_id:b7 Title: Transformers are rnns: Fast autoregressive transformers with linear attention Year: (2020)
Ref_id:b8 Title: Linear-time sequence modeling with selective state spaces Year: (2023)
Ref_id:b9 Title: Transformer quality in linear time Year: (2022)
Ref_id:b10 Title: The devil in linear transformer Year: (2022)
Ref_id:b11 Title: cosformer: Rethinking softmax in attention Year: (2022)
Ref_id:b12 Title: Rethinking attention with performers Year: (2020)
Ref_id:b13 Title: Gated linear attention transformers with hardware-efficient training Year: (2024)
Ref_id:b14 Title: The long-document transformer Year: (2020)
Ref_id:b15 Title: Generating long sequences with sparse transformers Year: (2019)
Ref_id:b16 Title: Big bird: Transformers for longer sequences Year: (2020)
Ref_id:b17 Title: Nyströmformer: A nyström-based algorithm for approximating self-attention Year: (2021)
Ref_id:b18 Title: Linformer: Self-attention with linear complexity Year: (2020)
Ref_id:b19 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b20 Title: Flashattention-2: Faster attention with better parallelism and work partitioning Year: (2023)
Ref_id:b21 Title: Griffin: Mixing gated linear recurrences with local attention for efficient language models Year: (2024)
Ref_id:b22 Title: Lite transformer with long-short range attention Year: (2020)
Ref_id:b23 Title: Synthesizer: Rethinking self-attention for transformer models Year: (2021)
Ref_id:b24 Title: On the properties of the softmax function with application in game theory and reinforcement learning Year: (2017)
Ref_id:b25 Title: Stand-alone self-attention in vision models Year: (2019)
Ref_id:b26 Title: Normalized attention without probability cage Year: (2020)
Ref_id:b27 Title: The quarks of attention: Structure and capacity of neural attention building blocks Year: (2023)
Ref_id:b28 Title: Vicinity vision transformer Year: (2023)
Ref_id:b29 Title: Hybrid random features Year: (2021)
Ref_id:b30 Title: Parallelizing linear transformers with the delta rule over sequence length Year: (2024)
Ref_id:b31 Title: A faster and better large language model with improved transnormer Year: (2023)
Ref_id:b32 Title: Rethinking attention with performers Year: (2021)
Ref_id:b33 Title: Replacing softmax with relu in vision transformers Year: (2023)
Ref_id:b34 Title: Theory, analysis, and best practices for sigmoid self-attention Year: (2024)
Ref_id:b35 Title: A study on relu and softmax in transformer Year: (2023)
Ref_id:b36 Title: Transformers as statisticians: Provable in-context learning with in-context algorithm selection Year: (2023)
Ref_id:b37 Title: What can a single attention layer learn? a study through the random features lens Year: (2023)
Ref_id:b38 Title: Sigmoid selfattention is better than softmax self-attention: A mixture-of-experts perspective Year: (2025)
Ref_id:b39 Title: Transformers are ssms: Generalized models and efficient algorithms through structured state space duality Year: (2024)
Ref_id:b40 Title: Theoretical limitations of self-attention in neural sequence models Year: ()
Ref_id:b41 Title: On the expressivity role of layernorm in transformers' attention Year: (2023)
Ref_id:b42 Title: More expressive attention with negative weights Year: (2025)
Ref_id:b43 Title: Bridging the divide: Reconsidering softmax and linear attention Year: (2024)
Ref_id:b44 Title: Differential transformer Year: (2025)
Ref_id:b45 Title: Gating is weighting: Understanding gated linear attention through in-context learning Year: (2025)
Ref_id:b46 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b47 Title: Qt-vit: Improving linear attention in vit with quadratic taylor expansion Year: (2024)
Ref_id:b48 Title: Taylorshift: Shifting the complexity of self-attention from squared to linear (and back) using taylor-softmax Year: (2025)
Ref_id:b49 Title: Simple linear attention language models balance the recall-throughput tradeoff Year: (2025)
Ref_id:b50 Title: Mbtaylorformer: Multi-branch efficient transformer expanded by taylor formula for image dehazing Year: (2023)
Ref_id:b51 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b52 Title:  Year: (2022)
Ref_id:b53 Title: Mechanistic design and scaling of hybrid architectures Year: (2024)
Ref_id:b54 Title: Hyena hierarchy: Towards larger convolutional language models Year: (2023)
Ref_id:b55 Title: Zoology: Measuring and improving recall in efficient language models Year: (2023)
Ref_id:b56 Title: -context language learning: Architectures and algorithms Year: (2024)
Ref_id:b57 Title: Retentive network: A successor to transformer for large language models Year: (2023)
Ref_id:b58 Title:  Year: (2023)
Ref_id:b59 Title: Hgrn2: Gated linear rnns with state expansion Year: (2024)
Ref_id:b60 Title: The pile: An 800gb dataset of diverse text for language modeling Year: (2020)
Ref_id:b61 Title:  Year: (2021)
Ref_id:b62 Title: Toeplitz neural network for sequence modeling Year: (2023)
Ref_id:b63 Title: Hgrn2: Gated linear rnns with state expansion Year: (2024)
Ref_id:b64 Title: Linear transformers as var models: Aligning autoregressive attention mechanisms with autoregressive forecasting Year: (2025)
Ref_id:b65 Title: itransformer: Inverted transformers are effective for time series forecasting Year: (2024)
Ref_id:b66 Title: A time series is worth 64 words: Long-term forecasting with transformers Year: (2022)
Ref_id:b67 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b68 Title: Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting Year: (2021)
Ref_id:b69 Title: Modeling long-and shortterm temporal patterns with deep neural networks Year: (2018)
Ref_id:b70 Title: Informer: Beyond efficient transformer for long sequence time-series forecasting Year: (2021)
