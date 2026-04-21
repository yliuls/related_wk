Title: LOCALITY-AWARE PARALLEL DECODING FOR EFFICIENT AUTOREGRESSIVE IMAGE GENERATION
Abstract: We present Locality-aware Parallel Decoding (LPD) to accelerate autoregressive image generation. Traditional autoregressive image generation relies on nextpatch prediction, a memory-bound process that leads to high latency. Existing works have tried to parallelize next-patch prediction by shifting to multi-patch prediction to accelerate the process, but only achieved limited parallelization. To achieve high parallelization while maintaining generation quality, we introduce two key techniques: (1) Flexible Parallelized Autoregressive Modeling, a novel architecture that enables arbitrary generation ordering and degrees of parallelization. It uses learnable position query tokens to guide generation at target positions while ensuring mutual visibility among concurrently generated tokens for consistent parallel decoding. (2) Locality-aware Generation Ordering, a novel schedule that forms groups to minimize intra-group dependencies and maximize contextual support, enhancing generation quality. With these designs, we reduce the generation steps from 256 to 20 (256×256 res.) and 1024 to 48 (512×512 res.) without compromising quality on the ImageNet class-conditional generation, and achieving at least 3.4× lower latency than previous parallelized autoregressive models.

Section: INTRODUCTION
Autoregressive modeling has achieved state-of-the-art results in large language models in terms of scalability and generalizability (Brown et al., 2020;OpenAI, 2023;Touvron et al., 2023a;b;Grattafiori et al., 2024;Jiang et al., 2024;Yang et al., 2024;2025;Liu et al., 2024a).
LPD (32 steps) LPD (20 steps) ARPG RandAR PAR 350M 700M 1.4B NAR Frechet Inception Distance (FID) 1.5 2 2.5 3 3.5 4 Latency(s, batch size=1) 0.1 1 10 NAR 1.5 2 2.5 3 3.5 4 0.1 1 10 4.2x faster 3.4x faster Naturally, many works have applied this powerful paradigm to visual generation (Esser et al., 2021;Lee et al., 2022;Ramesh et al., 2021;Yu et al., 2022;Sun et al., 2024;Tian et al., 2024). Moreover, this autoregressive formulation of visual generation has become increasingly crucial for unified multimodal generation (OpenAI, 2025;Wang et al., 2024a;Wu et al., 2024c;a;Chen et al., 2025a;Ma et al., 2025;Jiao et al., 2025;Song et al., 2025;Chen et al., 2025b;Zhao et al., 2025;Lin et al., 2025;Deng et al., 2025;Liao et al., 2025;Xie et al., 2025) since it is highly compatible with language modeling.
Prevailing autoregressive visual generation methods typically follow two paradigms: (1) next-patch prediction by flattening the image into a sequence of patches (Esser et al., 2021) and (2) next-scale prediction via coarse-to-fine multi-scale representations (Tian et al., 2024).
this section cite: ['b4', 'b44', 'b81', 'b20', 'b76', 'b45', 'b17', 'b33', 'b52', 'b79', 'b60', 'b36', 'b45', 'b42', 'b31', 'b59', 'b84', 'b39', 'b14', 'b38', 'b75', 'b17', 'b36']

Section: Locality Locality
Figure 2: Visualization of attention maps in the LLAMAGEN-1.4B model. There is strong spatial locality, as the attention of a decoding token is concentrated on nearby spatial tokens. LLAMAGEN encodes images into 24 × 24 tokens, where a token that is 24 positions earlier in the attention map corresponds to the token directly above it in the 2D grid.
In the first formulation, generating one token per step creates a memory-bound workloadfoot_0 , causing latency to scale with the number of steps. The second formulation substantially reduces generation steps and thus latency. However, its multi-scale token representation fundamentally differs from the universal flat token representation, making it incompatible with widely used flat vision perception foundation models (e.g., CLIP (Radford et al., 2021;Zhai et al., 2023), DINO (Caron et al., 2021;Oquab et al., 2023)) and thereby limiting interoperability with perception backbones that have been proven critical for unified multimodal systems (Wu et al., 2024c;Ma et al., 2025;Jiao et al., 2025;Song et al., 2025;Chen et al., 2025b;Zhao et al., 2025;Lin et al., 2025;Tong et al., 2024;Wu et al., 2025;2024b).
Thus, autoregressive visual generation should be (1) highly efficient: minimizing latency and maximizing throughput;
(2) remain flat token representations for universality and compatibility with vision backbones and, by extension, unified multimodal models. Recent works (Wang et al., 2024b;Pang et al., 2024;Li et al., 2025a) have tried to parallelize next-patch prediction by shifting to multi-patch prediction to accelerate the process, but only achieved limited parallelization. Non-autoregressive mask-prediction models like MASKGIT (Chang et al., 2022) enable multi-patch prediction but require full attention for bidirectional context, making them less efficient than autoregressive methods.
To address the challenges, we introduce Locality-aware Parallel Decoding (LPD), a framework that consists of a novel flexible parallelized autoregressive modeling architecture and a novel localityaware generation order schedule. We design a new modeling architecture as conventional decoder-only autoregressive models struggle with flexible generation order and parallelization, limiting efficiency. In contrast, ours enables arbitrary generation order and degrees of parallelization. This is achieved by using learnable position query tokens to guide the model in generating tokens at target positions. Moreover, the generation is parallel-aware, as we leverage specialized attention mechanism to ensure mutual visibility among tokens generated concurrently. Notably, our design also inherits the KV caching mechanism, avoiding redundant computation.
Furthermore, we observe strong spatial locality in image generation attention where tokens predominantly attend to nearby regions as shown in Figure 2. This indicates a high dependency among nearby tokens, meaning that spatially closer tokens provide stronger conditioning. Recent works (Wang et al., 2024b;Besnier et al., 2025) also identify that minimizing mutual dependency among simultaneously generated tokens is essential to maintain sample consistency. With these insights, we introduce a locality-aware generation order schedule that selects parallel decoding groups to maximize contextual support while minimizing intra-group dependencies, enabling higher degrees of parallelization.
We examine the effectiveness of our proposed method on ImageNet class-conditional image generation. Our results reveal that we reduce the generation steps of traditional raster-order autoregressive generation from 256 to 20 (256×256 res.) and 1024 to 48 (512×512 res.) without compromising quality, and achieving at least 3.4× lower latency (Figure 1) than previous parallelized autoregressive models. Thanks to the design of flexible autoregressive modeling, our models are also capable of zero-shot image editing including class-conditional editing, inpainting and outpainting.
this section cite: ['b51', 'b83', 'b6', 'b46', 'b42', 'b31', 'b59', 'b84', 'b39', 'b64', 'b73', 'b47', 'b7', 'b3']

Section: METHOD

this section cite: []

Section: RETHINKING AUTOREGRESSIVE MODELING
In next-patch autoregressive modeling, images are split into patches and usually discretized via a tokenizer into image tokens. While the joint distribution of the N tokens x 1 , • • • , x N and condition c is extremely high dimensional and therefore hard to model directly, the autoregressive framework makes this amenable by factorizing the total joint distribution as
p(x 1 , x 2 , . . . , x N ; c) = N n=1 p(x n |x <n ; c)(1)
The training objective of the autoregressive model is therefore to optimize parametric approximations p θ (x n |x <n ; c) for those one-step conditionals. This factorization needs a predefined order, typically raster order, as shown in Figure 3 (a). However, during sampling, this leads to N sequential steps, creating a major efficiency bottleneck.
To reduce the number of sequential generation steps, we can partition tokens into G disjoint groups
{X 1 , • • • , X G }, where each group X g = {x g1 , • • • , x gm } is predicted jointly, resulting in the following: p(x 1 , x 2 , . . . , x N ; c) = G g=1 p(X g | X <g ; c)(2)
The training objective becomes optimizing p θ (X g | X <g ; c). Previous work has shown that directly grouping tokens in raster order causes significant performance degradation (Wang et al., 2024b;Pang et al., 2024). This is because spatially adjacent tokens exhibit strong mutual dependencies, and independent sampling usually leads to generation inconsistencies inside a group. It is essential to break the raster order when grouping. In addition, the size of the prediction group |X g | should gradually increase. As the context size |X <g | grows, it offers stronger conditioning, allowing more tokens to be predicted in parallel. Previous work using masked transformers (Chang et al., 2022) also mirrors this intuition by predicting fewer tokens early when context is sparse and predicting more tokens over time. Therefore, an effective parallelized autoregressive model should support: (1) Flexible generation order to alleviate the issue caused by mutual interdependency of concurrently predicted tokens and (2) Dynamic group sizes increasing the number of tokens predicted per step with available context.
However, it is difficult to achieve these within the standard decoder-only autoregressive models, which are inherently designed with a fixed input-output structure, e.g. next-token prediction. In this modeling, each token simultaneously serves two roles: it provides context via its hidden state and enables generation via its output logits. This coupling limits flexibility in the the generation order and output size. To address these challenges, we propose a novel flexible parallelized autoregressive modeling which is able to support arbitrary generation order and degrees of parallelization.
this section cite: ['b47', 'b7']

Section: FLEXIBLE PARALLELIZED AUTOREGRESSIVE MODELING
Our core idea is to decouple the context representation and token generation by leveraging separate tokens. We illustrate this in Figure P1 <C> 4 3 5 3 5 P1 P2 P6 P2 P6 Query Key Cache Figure 5: Illustration of the inference attention mask. Encoding with image tokens and Decoding with position query tokens can be fused into a single step. Taking step 2 in Figure 3 (b) as the example, it simultaneously encodes the previously generated image tokens 3, 5 to update the KV-cache and decodes the desired image tokens 1, 2 and 6 in parallel.
to provide context and the generation is driven by learnable position query tokens corresponding to the desired target positions. These position query tokens are constructed by adding the positional embedding of the target location to a shared learnable embedding. By directly inputting these positionspecific queries, the model can generate tokens at arbitrary target positions in parallel. This design allows the model to leverage positional information in both the context and generation pathways, enabling arbitrary generation order.
Training formulation. We train the model to transform each position query token into the corresponding ground-truth image token, conditioned on all ground-truth tokens that precede it. To preserve teacher-forcing while allowing parallel prediction, we interleave position query tokens with ground-truth tokens and apply a specialized training attention mask as shown in Figure 4 that contains two attention patterns:
1. Context Attention allows subsequent tokens to attend to context tokens causally.
this section cite: []

Section: Query Attention ensures mutual visibility among the position query tokens within the same step, and prevents any subsequent tokens from attending to the query tokens.
Inference formulation. At test time we alternate between encoding the generated image tokens and decoding with position query tokens.
1. Encoding. Sampled image tokens go through a forward pass to store the KV cache, providing context for future decoding steps. 2. Decoding. Learnable position query tokens attend to all previously generated tokens in the KV cache, and the forward pass outputs logits for each target position in parallel. KV cache for query tokens is not stored.
However, sequentially execute these two operations double the generation steps. As shown in Figure 3 (b), these two operations can be fused into a single step via a specialized inference attention mask as shown in Figure 5.
Comparison with other methods. Recent efforts have also pursued parallel generation in autoregressive modeling, yet each carries inherent limitations. One line of work, exemplified by SAR (Liu et al., 2024b) and ARPG (Li et al., 2025a), adopts an encoder-decoder architecture where target-aware query tokens attend to the encoder's key-value cache via cross-attention. However, as illustrated in Figure 6 (a), the target positions themselves do not contribute any key-value pairs, resulting in the tokens generated within the same parallel step being produced independently of one another. Another approach, represented by RANDAR (Pang et al., 2024), adheres to the prevailing decoderonly architecture. It achieves arbitrary order by inserting positional instruction tokens to designate target positions. However, it still leverages a standard causal mask during training. This strategy, as depicted in Figure 6 (b), leads to two notable issues: (1) the parallel generation degenerates into a batched next-token prediction instead of joint prediction and (2) the positional instruction tokens must be stored in the KV cache during inference, doubling the memory consumption. Compared with these two methods, our method as shown in Figure 6 (c) guarantees the visibility among all concurrently predicted target positions and only stores the generated tokens in the KV cache.
PAR (Wang et al., 2024b), NAR (He et al., 2025), and ZipAR (He et al., 2024) preserve the standard decoder-only architecture and increase the number of tokens generated per step. Although they guarantee mutual visibility among concurrently generated tokens, they rely on a fixed parallel generation order, which prevents them from supporting arbitrary generation orders. This limits the generation flexibility thus achieved limited parallelization and generation quality. ACDIT (Hu et al., 2024) shares similar attention scheme with us, yet it was used for evenly interpolating between autoregressive and diffusion modeling.
this section cite: ['b47', 'b26', 'b25', 'b29']

Section: LOCALITY-AWARE GENERATION ORDER SCHEDULE
To fully leverage our flexible parallelized autoregressive modeling architecture, we introduce a locality-aware generation order schedule. This schedule is guided by two key principles (1) High proximity to previously generated tokens: target positions should be spatially close to existing context to ensure strong conditioning and (2) Low proximity among concurrently generated tokens: tokens predicted in the same parallel step should be spatially distant to reduce mutual dependency.
These principles are derived from a systematic analysis of the attention patterns in autoregressive image generation by the widely adopted LLAMAGEN (Sun et al., 2024) model. Using LLAMAGEN, we generate 50,000 images and collect attention scores at each decoding step. Qualitative attention patterns are shown in Figure 2, and quantitative results are presented in Figure 7. To quantify locality, we define the Per-Token Attention (PTA) to a neighborhood of radius sfoot_1 as:
P T A s = 1 N N i=1 j Attention(T i , T j ) • I[d(T i , T j ) = s] j I[d(T i , T j ) = s](3)
where Attention(T i , T j ) denotes the attention weight from token T i to token T j , and d(T i , T j ) is their Euclidean distance on the 2D image grid.
As shown in Figure 7 (a), PTA decreases sharply with increasing distance, indicating a strong spatial locality in the attention mechanism. This suggests that nearby tokens carry significantly more useful information during decoding, and that spatially adjacent tokens are highly dependent on one another for accurate prediction. This locality pattern is consistently observed across all attention heads. In Figure 7 (b), we visualize the Attention Sum, defined as the total attention score a decoding token assigns to tokens within a relative distance s. The plot uses s = 3 and confirms that most attention is concentrated within local neighborhoods, reinforcing the importance of spatial locality. This analysis supports our two principles: decoding tokens should remain close to previously generated tokens to maximize contextual support, and distant from concurrently generated tokens to minimize intra-group dependency.
Based on these principles, we implement a locality-aware generation order schedule described in Algorithm 1. Suppose we use K decoding steps to generate N 2 tokens, with group sizes O = [o 1 , o 2 , . . . , o K ], where o k is the number of tokens generated in step k, typically increasing via a cosine schedule. At each step k, we compute the euclidean distance between unselected and already selected tokens to measure spatial proximity, where closer distance leads to higher proximity. We sort unselected tokens by proximity and split them into two sets: c 1 are tokens with sufficient proximity larger than the threshold τ which are eligible for the following high-proximity selection, and c 2 are the rest. We sequentially select tokens from c 1 , adding each to the selected set while filtering out nearby tokens that the relative distance is smaller than the repulsion threshold ρ, which are added to c 2 . If all the grids in c 1 are considered and the number of selected grids is less than o k , we use farthest point sampling (Qi et al., 2017) to select the remaining grids from c 2 to ensure spatial low dependency. It is worth noting that the generation order can be precomputed and stored for direct use during inference, incurring no additional latency. We provide the PyTorch implementation in Appendix B.1. We further clarify the key distinction between our method and prior work in Appendix B.2.
Algorithm 1: Locality-aware Generation Order Schedule For intuitive understanding, we illustrate an example of our generation order schedule in Figure 8. We also plot the schedule for raster order, random order and Halton order (Besnier et al., 2025) for comparison. The raster order generates tokens in a raster-scan manner and the random order generates tokens in a random manner. The Halton order is a low-discrepancy sequence to arrange the generation positions which spreads out the tokens to achieve uniform image coverage step by step.
Input: decoding steps K, group sizes O = [o 1 , o 2 , . . . , o K ], grids G = {(i, j)} N i,j=1 , proximity threshold τ , repulsion threshold ρ; schedule S = [ ]; for k = 1, . . . , K do s = [ ]; p = 1/ euclidean(G \ S, S) ; ▶ proximity measurement c = sorted(G \ S,
this section cite: ['b60', 'b50', 'b3']

Section: EXPERIMENT
3.1 SETUP Models. For fair comparisons with existing autoregressive image generation methods, we use the LLAMAGEN tokenizer (Sun et al., 2024) with codebook size 16384 and downsample factor 16. We train three models of different sizes: 337M, 752M, and 1.4B parameters. We use a standard decoder-only transformer architecture, and refer to them as LPD-L, LPD-XL, and LPD-XXL, respectively. Please refer to the Appendix A.1 for more details.
Step1 Step2 Step4 Step6 Step9 Step12 Step15 Step18(
Training and Evaluation. We train and evaluate our models on the class-conditional ImageNet (Russakovsky et al., 2015) 256×256 and ImageNet 512×512 datasets. We first train all models on ImageNet 256×256 for 450 epochs, with 50 epochs of learning rate warmup followed by constant learning rate and finally 50 epochs of cosine decay. For 512-resolution models, we load the pre-trained 256-resolution models and interpolate the positional embeddings and continue training on ImageNet 512×512 for another 50 epochs. During training, the image tokens are randomly shuffled while the class token is kept at the beginning. We train on a range of predefined decoding steps where the tokens per step follows a cosine schedule. We reportuse Fréchet Inception Distance (FID) (Heusel et al., 2017) as the primary metric computed on 50k,000 generated samples as the primary metric as well asnd also report Inception Score (IS) (Salimans et al., 2016), Precision, and Recall (Kynkäänniemi et al., 2019). Please refer to the Appendix A.2 for more details.
this section cite: ['b60', 'b27', 'b57', 'b32']

Section: Efficiency Profiling.
We profile all the efficiency results on a single NVIDIA A100 GPU with BFloat16 precision. We measure the latency with a batch size of 1 and throughput with a batch size of 64. We report the average latency over 500 inference steps, with a 100-step warm-up period.
this section cite: []

Section: MAIN RESULTS
We compare our models against a broad set of generative baselines on ImageNet 256×256 (Table 1).
For a fair comparison, we also create a raster order counterpart following the same setup. As shown in the table, we reduce the generation steps from 256 to 20, achieving 12.8× generation steps reduction, without sacrificing the generation quality. Compared with other parallelized autoregressive models, we achieve significantly better image generation quality and efficiency. Taking LPD-XL model as an example, it achieves a FID of 2.10 with only 20 steps, reducing the number of generation steps by 3.2× compared to ARPG and achieving 4.2× lower latency. Increasing the steps slightly to 32 yields a FID of 1.92, even matching ARPG-XXL, while reducing latency by 3.4×. We further report our results on ImageNet 512×512 (Table 2). As shown in the table, we reduce the generation steps from 1024 to 48, achieving 21.3× generation steps reduction, without sacrificing the generation quality. These results validate the effectiveness of our flexible parallelized autoregressive modeling and the locality-aware generation order schedule. We also provide qualitative generation and zero-shot editing results in Figure 12.
Our method is general and can be readily extended to higher-resolution text-to-image generation (e.g., 1024×1024). We provide a detailed description of this extension in the Appendix A.3. We evaluate on the widely used GenEval (Ghosh et al., 2023) benchmark and report results in Table 3. As demonstrated in the table, LPD reduces the sampling steps for 1024×1024 image generation from 4096 to just 64 and simultaneously improves the GenEval score. This provides compelling evidence that LPD is a generalizable method capable of supporting high-resolution text-to-image generation. We also include qualitative generation results in Figure 13 in the appendix.
this section cite: ['b19']

Section: EFFICIENCY ANALYSIS
Our method introduces position query tokens to enable flexible generation. These tokens add extra queries and thereby increase FLOPs. However, the resulting computational overhead has a negligible impact on wall-clock latency in memory-bound settings such as small-batch inference. In these scenarios, the reduction in generation steps translates almost linearly into latency reduction. As the batch size increases, the system progressively shifts toward a compute-bound regime, where the additional overhead begins to matter and diminish the speedup. We provide a quantitative analysis in Figure 14 to illustrate this trend. By gradually increasing the batch size until reaching the memory limit, we observe that the model transitions from memory-bound to compute-bound when the batch size exceeds 16. Nevertheless, even at the maximum feasible batch size, our method retains a throughput advantage of approximately 3× over the raster-order baseline.
this section cite: []

Section: ABLATION

this section cite: []

Section: Effectiveness of Flexible Parallelized Autoregressive Modeling.
One key design of our flexible parallelized autoregressive modeling is the guarantee of the mutual visibility among all concurrently generated tokens. This is critical to maintain the consistency in the same group when the degree of the parallelization is high. We show the effectiveness of this design in Figure 9 (a). We compare our model with RANDAR and ARPG which lack this design. To only ablate the effectiveness of our flexible parallelized autoregressive modeling, we use random generation order for all models without our locality-aware parallel generation order schedule. As shown in the figure, with the generation steps decrease and the parallelization increases, our model exhibits a smaller FID increase compared with the other two models. For example, with 32 steps, our model almost maintain the performance with 256 steps but ARPG and RANDAR have a significant FID increase. This design is crucial for us to achieve fewer generation steps while maintaining the generation performance.
this section cite: []

Section: Effectiveness of Locality-aware Generation Order Schedule.
We compare our schedule with another two generation order schedules as shown in Figure 9 (b). Random order just arrange the generation positions randomly. Halton order leverages the Halton low-discrepancy sequence to arrange the generation positions which spreads out the tokens to achieve uniform image coverage step by step. Intuitively it mainly focus on reducing the dependency inside a parallel group which shares the same insight with our second principle that low proximity is needed among concurrently generated tokens. However, the low-discrepancy sequence omits the importance of the already generated context which is our first principle that we need to maintain high proximity to previously generated tokens. As shown in the figure, our locality-aware parallel decoding order consistently outperforms the other two orders, showing the effectiveness of our method.
this section cite: []

Section: Effectiveness of the Locality Principles.
As introduced in Section 2.3, our locality-aware generation order schedule is guided by two principles. We ablate the effectiveness of these two principles in Figure 9 (c). As shown, the random order baseline yields an FID of 2.11. We first apply Principle 1 only, selecting points close to previously generated tokens without considering their mutual dependency. This improves the performance to 2.00. We then apply Principle 2 alone, using farthest point sampling at each step to ensure concurrently generated tokens are well separated, without considering context from previously generated tokens. This improves the FID to 2.06. Combining both in our locality-aware generation order achieves 1.92, highlighting the synergy of both principles. We further provide a sensitivity analysis of the hyperparameters τ and ρ in Appendix B.3.
this section cite: []

Section: RELATED WORKS

this section cite: []

Section: AUTOREGRESSIVE IMAGE GENERATION
Autoregressive models generate the current output conditioned only on previous outputs. Usually this dependency is captured by causal attention mechanisms, enabling efficient inference via KV caching. Autoregressive modeling with GPT-style "next-token-prediction" (Brown et al., 2020;OpenAI, 2023;Touvron et al., 2023a;b;Chiang et al., 2023;Jiang et al., 2024) has dominated the field of language generation. Inspired by this success, autoregressive visual generation has shifted from operating on sequences of pixels (Van Den Oord et al., 2016;Van den Oord et al., 2016;Parmar et al., 2018;Chen et al., 2018;Salimans et al., 2017;Yu et al., 2021;Li et al., 2025b) to sequences of latent discrete tokens (Esser et al., 2021;Lee et al., 2022;Ramesh et al., 2021;Razavi et al., 2019;Yu et al., 2021;2022;Sun et al., 2024;Yu et al., 2024;Wang et al., 2024a;Teng et al., 2024;Ren et al., 2025;He et al., 2025;2024). However, the token-by-token decoding strategy is often bottlenecked by memory bandwidth. This limitation prevents full utilization of computation and results in high latency. Recently, "next-scale-prediction" (Tian et al., 2024;Han et al., 2024) has emerged to predict the next scale of the image instead of the next token thus accelerates the generation process. However, its multi-scale token representation fundamentally differs from the universal flat token representation, making it incompatible with widely used flat vision perception foundation models.
this section cite: ['b4', 'b44', 'b81', 'b13', 'b67', 'b67', 'b48', 'b10', 'b58', 'b78', 'b17', 'b33', 'b52', 'b53', 'b78', 'b60', 'b82', 'b62', 'b54', 'b26', 'b36', 'b23']

Section: PARALLEL GENERATION IN SEQUENCE MODELING
Parallel generation has been widely studied in the field of language modeling. Prior to the era of large language models, masked-prediction architectures (Gu et al., 2017;Ghazvininejad et al., 2019;Gu et al., 2019) were commonly used to do parallel generation and iterative refinement. Recently, with the rapid success of large language models, speculative decoding (Chen et al., 2023;Leviathan et al., 2023) and its derivatives (Cai et al., 2024;Ankner et al., 2024) employ a draft model to generate the next few tokens and then the main model conducts the verification. In visual generation, masked-prediction models (Chang et al., 2022;Yu et al., 2023a;b;Chang et al., 2023) are widely used to generate masked tokens step by step leveraging a masked prediction transformer similar to BERT (Devlin et al., 2019;Bao et al., 2021;He et al., 2022), which are able to generate multiple tokens in parallel. However, they are non-autoregressive models and need bidirectional attention which is computationally expensive and KV cache is not applicable to accelerate the inference. Recent works (Wang et al., 2024b;Pang et al., 2024;Li et al., 2025a;He et al., 2025) have explored parallel generation in autoregressive models, but with limited parallelization and generation quality.
Our proposed method enables greater parallelization without sacrificing performance.
this section cite: ['b21', 'b18', 'b22', 'b9', 'b34', 'b5', 'b0', 'b7', 'b81', 'b8', 'b15', 'b2', 'b24', 'b47', 'b26']

Section: CONCLUSION
Our contributions lie in two key aspects: (1) flexible parallelized autoregressive modeling and (2) locality-aware generation order schedule. We significantly reduce the generation steps required by the traditional autoregressive models without compromising the generation quality and achieve at least 3.4× lower latency than previous parallelized autoregressive models.
continue 20 21 # Calculate the proximity score for all remaining grid coords 22 candidates = [] 23 for coord in grid_coords: if coord in selected_coords: 25 continue 26 # Calculate the proximity score based on euclidean distance to already selected grid coords proximity_score = 0 29 for selected_coord in selected_coords: if abs(coord[0] -selected_coord[0]) <= 1 and abs(coord[1] -selected_coord[1]) <= 1: distance = euclidean(coord, selected_coord) 32 if distance > 0: 33 proximity_score += 1.0 / distance 34 candidates.append([proximity_score, coord]) # Shuffle candidates so that grid coords with the same proximity score are randomly ordered # Set already selected candidates to 0 distance 94 min_distances[selected_indices] = 0 95 96 # Select the candidate with maximum minimum distance 97 idx = np.argmax(min_distances) 98 selected_np = np.vstack([selected_np, candidates_np[idx]]) 99 100 selected_indices.append(idx) 101 102 return [candidate_points[i] for i in selected_indices]
this section cite: []

Section: References
Ref_id:b0 Title: Sequentially-dependent draft heads for medusa decoding Year: (2024)
Ref_id:b1 Title: Qwen2.5-vl technical report Year: (2025)
Ref_id:b2 Title: Bert pre-training of image transformers Year: (2021)
Ref_id:b3 Title: Halton scheduler for masked generative image transformer Year: (2025)
Ref_id:b4 Title: Language models are few-shot learners Year: (2020)
Ref_id:b5 Title: Simple llm inference acceleration framework with multiple decoding heads Year: (2024)
Ref_id:b6 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b7 Title: Masked generative image transformer Year: (2022)
Ref_id:b8 Title: Text-to-image generation via masked generative transformers Year: (2023)
Ref_id:b9 Title: Accelerating large language model decoding with speculative sampling Year: (2023)
Ref_id:b10 Title: Pixelsnail: An improved autoregressive generative model Year: (2018)
Ref_id:b11 Title: Janus-pro: Unified multimodal understanding and generation with data and model scaling Year: (2025)
Ref_id:b12 Title: Semhitok: A unified image tokenizer via semantic-guided hierarchical codebook for multimodal understanding and generation Year: (2025)
Ref_id:b13 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-04)
Ref_id:b14 Title: Emerging properties in unified multimodal pretraining Year: (2025)
Ref_id:b15 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b16 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b17 Title: Taming transformers for high-resolution image synthesis Year: (2021)
Ref_id:b18 Title: Mask-predict: Parallel decoding of conditional masked language models Year: (2019)
Ref_id:b19 Title: Geneval: An object-focused framework for evaluating text-to-image alignment Year: (2023)
Ref_id:b20 Title: The llama 3 herd of models Year: (2024)
Ref_id:b21 Title: Non-autoregressive neural machine translation Year: (2017)
Ref_id:b22 Title: Levenshtein transformer. Advances in neural information processing systems Year: (2019)
Ref_id:b23 Title: Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis Year: (2024)
Ref_id:b24 Title: Masked autoencoders are scalable vision learners Year: (2022)
Ref_id:b25 Title: Accelerating autoregressive image generation through spatial locality Year: (2024)
Ref_id:b26 Title: Neighboring autoregressive modeling for efficient visual generation Year: (2025)
Ref_id:b27 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b28 Title: Cascaded diffusion models for high fidelity image generation Year: (2022)
Ref_id:b29 Title: Interpolating autoregressive conditional modeling and diffusion transformer Year: (2024)
Ref_id:b30 Title:  Year: (2024)
Ref_id:b31 Title: Unitoken: Harmonizing multimodal understanding and generation through unified visual encoding Year: (2025)
Ref_id:b32 Title: Improved precision and recall metric for assessing generative models Year: (2019)
Ref_id:b33 Title: Autoregressive image generation using residual quantization Year: (2022)
Ref_id:b34 Title: Fast inference from transformers via speculative decoding Year: (2023)
Ref_id:b35 Title: Autoregressive image generation with randomized parallel decoding Year: (2025)
Ref_id:b36 Title: Autoregressive image generation without vector quantization Year: (2024)
Ref_id:b37 Title: Lijie Fan, and Kaiming He. Fractal generative models Year: (2025)
Ref_id:b38 Title: Mogao: An omni foundation model for interleaved multi-modal generation Year: (2025)
Ref_id:b39 Title: Toklip: Marry visual tokens to clip for multimodal comprehension and generation Year: (2025)
Ref_id:b40 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b41 Title: Customize your visual autoregressive recipe with set autoregressive modeling Year: (2024)
Ref_id:b42 Title: Unitok: A unified tokenizer for visual generation and understanding Year: (2025)
Ref_id:b43 Title: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers Year: (2024)
Ref_id:b44 Title:  Year: (2023)
Ref_id:b45 Title: Introducing 4o image generation Year: (2025-03)
Ref_id:b46 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b47 Title: Decoder-only autoregressive visual generation in random orders Year: (2024)
Ref_id:b48 Title: Image transformer Year: (2018)
Ref_id:b49 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b50 Title: Pointnet++: Deep hierarchical feature learning on point sets in a metric space Year: (2017)
Ref_id:b51 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b52 Title: Zero-shot text-to-image generation Year: (2021)
Ref_id:b53 Title: Generating diverse high-fidelity images with vq-vae-2. Advances in neural information processing systems Year: (2019)
Ref_id:b54 Title: Beyond nexttoken: Next-x prediction for autoregressive visual generation Year: (2025)
Ref_id:b55 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b56 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b57 Title: Improved techniques for training gans Year: (2016)
Ref_id:b58 Title: Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications Year: (2017)
Ref_id:b59 Title: Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies Year: (2025)
Ref_id:b60 Title: Autoregressive model beats diffusion: Llama for scalable image generation Year: (2024)
Ref_id:b61 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b62 Title: Accelerating auto-regressive text-to-image generation with training-free speculative jacobi decoding Year: (2024)
Ref_id:b63 Title: Visual autoregressive modeling: Scalable image generation via next-scale prediction Year: (2024)
Ref_id:b64 Title: Metamorph: Multimodal understanding and generation via instruction tuning Year: (2024)
Ref_id:b65 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b66 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b67 Title: Aäron Van Den Oord, Nal Kalchbrenner, and Koray Kavukcuoglu. Pixel recurrent neural networks Year: (2016)
Ref_id:b68 Title: Next-token prediction is all you need Year: (2024)
Ref_id:b69 Title: Parallelized autoregressive visual generation Year: (2024)
Ref_id:b70 Title: Maskbit: Embedding-free image generation via bit tokens Year: (2024)
Ref_id:b71 Title: Janus: Decoupling visual encoding for unified multimodal understanding and generation Year: (2024)
Ref_id:b72 Title: Liquid: Language models are scalable multi-modal generators Year: (2024)
Ref_id:b73 Title: Harmonizing visual representations for unified multimodal understanding and generation Year: (2025)
Ref_id:b74 Title: Vila-u: a unified foundation model integrating visual understanding and generation Year: (2024)
Ref_id:b75 Title: Show-o2: Improved native unified multimodal models Year: (2025)
Ref_id:b76 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b77 Title: Qwen3 technical report Year: (2025)
Ref_id:b78 Title: Vector-quantized image modeling with improved vqgan Year: (2021)
Ref_id:b79 Title: Scaling autoregressive models for contentrich text-to-image generation Year: (2022)
Ref_id:b80 Title: Magvit: Masked generative video transformer Year: (2023)
Ref_id:b81 Title: Language model beats diffusiontokenizer is key to visual generation Year: (2023)
Ref_id:b82 Title: Randomized autoregressive visual generation Year: (2024)
Ref_id:b83 Title: Sigmoid loss for language image pre-training Year: (2023)
Ref_id:b84 Title: Qlip: Text-aligned visual tokenization unifies auto-regressive multimodal understanding and generation Year: (2025)
