Title: Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions
Abstract: In recent years, masked diffusion models (MDMs) have emerged as a promising alternative approach for generative modeling over discrete domains. Compared to autoregressive models (ARMs), MDMs trade off complexity at training time with flexibility at inference time. At training time, they must learn to solve an exponentially large number of infilling problems, but at inference time, they can decode tokens in essentially arbitrary order. In this work, we closely examine these two competing effects. On the training front, we theoretically and empirically demonstrate that MDMs indeed train on computationally intractable subproblems compared to their autoregressive counterparts. On the inference front, we show that a suitable strategy for adaptively choosing the token decoding order significantly enhances the capabilities of MDMs, allowing them to sidestep hard subproblems. On logic puzzles like Sudoku, we show that adaptive inference can boost solving accuracy in pretrained MDMs from < 7% to ≈ 90%, even outperforming ARMs with 7× as many parameters and that were explicitly trained via teacher forcing to learn the right order of decoding.

Section: Introduction
While diffusion models (Ho et al., 2020;Song et al., 2021) are now the dominant approach for generative modeling in continuous domains like image, video, and audio, efforts to extend this methodology to discrete domains like text and proteins (Austin et al., 2021;Lou et al., 2024;Hoogeboom et al., 2021b) remain nascent. Among numerous proposals, masked diffusion models (MDMs) (Lou et al., 2024;Sahoo et al., 2025;Shi et al., 2024) have emerged as a leading variant, distinguished by a simple and principled objective: to generate samples, learn to reverse a noise process which independently and randomly masks tokens.
In many applications, such as language modeling, masked diffusion models (MDMs) still underperform compared to autoregressive models (ARMs) (Nie et al., 2024;Zheng et al., 2024), which instead learn to reverse a noise process that unmasks tokens sequentially from left to right. However, recent studies suggest that MDMs may offer advantages in areas where ARMs fall short, including reasoning (Nie et al., 2024;Kitouni et al., 2025), planning (Ye et al., 2024), and infilling (Gong et al., 2024). This raises a key question: what are the strengths and limitations of MDMs compared to ARMs, and under what conditions can MDMs be scaled to challenge the dominance of ARMs in discrete generative modeling?
To understand these questions, we turn a microscope to two key competing factors when weighing the merits of MDMs over ARMs:
• Complexity at training time: By design, the prediction task that MDMs are trained on is more challenging. Whereas ARMs seek to predict the next token given an unmasked prefix, MDMs seek to predict a token conditioned on a set of unmasked tokens in arbitrary positions.
• Flexibility at inference time: On the other hand, the sampling paths taken by an MDM are less rigid. The order in which tokens are decoded at inference time is random instead of fixed to left-to-right. In fact, even more is possible: MDMs can actually be used to decode in any order.
Therefore, we ask:
Are the benefits of inference flexibility for MDMs enough to outweigh the drawbacks of training complexity?
In this work, we provide dual perspectives on this question.
(1) Training for the worst. First, we provide theoretical and empirical evidence that the overhead imposed by training complexity quantifiably impacts MDMs' performance.
We prove that even for simple, benign models of data, there are noise levels at which a large fraction, but not all, of the corresponding subproblems solved by MDMs are computationally intractable. We then show this imbalance in computational complexity across subproblems persists even in real-world text data (Fig. 2, left).
(2) Planning for the best. While the above might appear to be bad news for MDMs, in the second part of this paper we answer our guiding question in the affirmative by building upon the observation (Zheng et al., 2023) that MDMs which can perfectly solve all masking subproblems can be used to decode in any order.
In place of vanilla MDM inference whereby tokens are unmasked in random order, we consider adaptive strategies that carefully select which token to unmask next. Our key insight is that this adaptivity makes it possible to sidestep the hard subproblems from training (Fig. 1). In fact, we find that even without modifying how MDMs are trained, the resulting models' logits contain enough information to determine the right order in which to unmask.
Our main empirical result is to show that the performance of MDMs pretrained on logic puzzle data dramatically improves when one goes from vanilla to adaptive inference. For example, on Sudoku puzzles, a simple adaptive strategy (Section 4.1) improves the accuracy of MDMs from < 7% to almost 90%. Remarkably, this not only outperforms vanilla ARMs, but even bespoke ARMs trained to learn the right decoding order via supervised teacher forcing (Shah et al., 2024;Lehnert et al., 2024) (Table 2).
Organization. In Section 2, we provide preliminaries on MDMs and set notation. In Section 3, we examine MDM training and demonstrate the imbalance in computational intractability across subproblems. In Section 4, we consider adaptive inference in MDMs and investigate its impact on likelihood modeling across various tasks.
this section cite: ['b13', 'b46', 'b3', 'b28', 'b28', 'b39', 'b42', 'b31', 'b54', 'b31', 'b20', 'b52', 'b12', 'b55', 'b41', 'b22']

Section: Masked Diffusion Models (MDM)
In this section, we explain the framework of Masked Diffusion Models (Shi et al., 2024;Sahoo et al., 2025) and its interpretation as an order-agnostic learner. MDMs gradually add noise to the true discrete data and learn the marginal distribution of the induced reverse process. Below, we formulate the forward and reverse processes for MDMs.
Let the distribution p data on {1, . . . , m} L be the data distribution over sequences of length L and with vocabulary {1, . . . , m}. We use 0 to denote the "mask" token.
Forward process. For a given x 0 ∼ p data and a noise level t ∈ [0, 1], the forward process x t ∼ q t|0 (• | x 0 ) is a coordinate-independent masking process via q t|0 (x t |x 0 ) = L-1 i=0 q t|0 (x i t |x i 0 ), where q t|0 (
x i t | x i 0 ) = Cat α t e x i 0 +(1- Figure 1.
(Top) MDM training can be seen as learning multiple masked prediction problems, where some are harder to learn, leading to performance imbalance (Section 3). (Bottom) During inference, adaptive MDM can avoid difficult problem instances, improving performance (Section 4).
α t )e 0 and α t is the predefined noise schedule satisfying α 0 ≈ 1, α 1 ≈ 0 and e x i 0 ∈ R m+1 denotes a one-hot vector corresponding to the value of token x i 0 . Cat(π) denotes the categorical distribution given by π ∈ ∆ m . In other words, for each i-th coordinate, x i t is masked to the mask token 0 with probability 1 -α t and unchanged otherwise.
Reverse process. The reverse process of the above forward process is denoted using q s|t (x s |x t , x 0 ) and is given by
q s|t (x s |x t , x 0 ) = L-1 i=0 q s|t (x i s |x t , x 0 )
for any s < t, where
q s|t (x i s | xt, x0) = Cat(e x i t ) x i t ̸ = 0 Cat 1-αs 1-α t em + αs-α t 1-α t ex 0 x i t = 0 .
The reverse transition probability q s|t (
x i s |x t , x 0 ) is approx- imated using g θ (x i s |x t ) ≜ q s|t (x i s | x t , x 0 ← p θ (x t , t))
where p θ (x t , t) is a denoising network trained to predict the marginal on x 0 via an ELBO-based loss. To be precise, q s|t x i s | x t , x 0 ← p θ (x t , t) indicates the conditional probability where
p θ (x t , t) is placed in the position of x 0 within q s|t (x i s | x t , x 0 ). L θ = 1 0 α ′ t 1 -α t E x0∼p data xt∼q t|0 (•|x0) i:x i t =0 -log p θ (x i 0 |x t , t)dt.
Here, α ′ t = dαt dt and δ xt,0 is the indicator function; the summation is computed over coordinates i s.t. x i t = 0. In practice, a time-embedding-free architecture for the denoising network, i.e., p θ (x t , t) = p θ (x t ), is usually employed as x t implicitly contains information about t via the number of masked tokens.
The reverse sampling process starts from the fully masked sentence x 1 = (0, . . . , 0). At a given noise level t ∈ (0, 1], suppose we have a partially masked sequence x t . For predetermined noise level s < t, we sample x s ∼ g θ (•|x t ). This process is repeated recursively from t = 1 to t = 0.
this section cite: ['b42', 'b39']

Section: Reformulating the training and inference of MDMs
In this section, we first discuss vanilla order-agnostic training of MDMs and compare it with "left-to-right" order training of autoregressive models in Section 2.1.1. Then, we reformulate vanilla MDM inference in Section 2.1.2 to set the stage for the upcoming discussion.
this section cite: []

Section: ORDER-AGNOSTIC TRAINING OF MDMS
Recent works (Zheng et al., 2024;Ou et al., 2024) have observed that the learning problem of MDM is equivalent to a masked language model. Building upon their analysis, we reformulate the loss L θ to show that L θ is a linear combination of the loss for all possible infilling masks. We first define x 0 [M ] as a masked sequence, obtained from original sequence x 0 where indices in the mask set M (regarded as a subset of [L] ≜ {1, 2, . . . , L}) are replaced with mask token 0.
Proposition 2.1. Assume α 0 = 1, α 1 = 0 and denoising network p θ is time-embedding free. Then L θ ≤ -E x0∼p data [log p θ (x 0 )] and The proof of the above proposition is given in Appendix E. As the MDM loss is a linear combination of the loss for all possible infilling mask M , the minimizer of the loss L θ learns to solve every masking problem. In other words, the optimal predictor p θ is the posterior marginal of the i-th token, conditioned on x 0 [M ] for all masks M . The training objective of MDM aims to predict x 0 from x 0 [M ] across all possible masks. Hence, we will refer to the MDM training as order-agnostic training.
L θ = - M ⊆[L],i∈M 1 |M | 1 L |M | E x0∼p data [log p θ (x i 0 |x 0 [M ])],(1)
On the other hand, Autoregressive Models (ARMs) learn to solve a smaller set of infilling problems (L infilling problems in ARMs as opposed to exp(L) infilling problems in MDM) by predicting i th token x i given all previous tokens x 0 , . . . , x i-1 . This prediction problem is equivalent to predicting x i by masking at positions {i, . . . , L-1}. Therefore, we can write it as
log p θ (x 0 ) = L-1 i=0 log p θ (x i 0 |x 0 [{i, . . . , L -1}]). (2)
ARMs are trained to predict tokens sequentially from left to right in all sequences. We refer to this as left-to-right training. In general, one can also consider predicting tokens sequentially under some fixed, known permutation of the sequence; we refer to this as order-aware training.
this section cite: ['b54', 'b33']

Section: ORDER-AGNOSTIC INFERENCE OF MDMS
The MDM inference can be decomposed into two steps: (a) randomly selecting a set of positions to unmask and (b) assigning token values to each position via the denoising network p θ . More precisely, we can reformulate the reverse process x s ∼ g θ (•|x t ) as follows.
Vanilla MDM inference
(a) Sample a set of masked tokens S ⊆ {i | x i t = 0}, P(i ∈ S) = αs-αt 1-αt . (b) For each i ∈ S, sample x i s ∼ p θ (x i |x t ).
Therefore, the inference in MDM is implemented by randomly selecting S and then filling each token value according to the posterior probability p θ (x i s |x t ).
this section cite: []

Section: MDMs train on hard problems
In this section, we theoretically and empirically demonstrate that a large portion of masking subproblems p θ (
x i 0 | x 0 [M ]) can be difficult to learn. For intuition, consider solving a masked prediction problem p θ (x i | x 0 [M ]
) on text data like masking an arbitrary sentence in the middle of a document and predicting the correct word for a specific position in that sentence. It is reasonable that this task should be more complex, even for humans, than left-to-right prediction, and in this section, we place this intuition on a rigorous footing.
In Section 3.1, we show several examples of simple, non-pathological distributions for which: (1) the masking problems encountered during order-aware training are computationally tractable, yet (2) many of the ones en- countered during order-agnostic training are computationally intractable. In Section 3.2, we empirically show that text data also exhibits this gap between the computational complexity of order-aware and order-agnostic training. In Section 3.3, we reveal that this discrepancy in computational complexity manifests empirically in performance imbalance across tasks: as predicted by the theory, MDMs trained on data from such distributions exhibits small errors on easy subproblems but suffers from large errors on harder ones.
this section cite: []

Section: Benign distributions with hard masking problems
We now describe a simple model of data under which we explore the computational complexity of masking problems.
Definition 3.1. A latents-and-observations (L&O) distribution is a data distribution p data over sequence of length L with alphabet size m (precisely, p data is over {0, . . . , m} L ) is specified by a permutation π over indices {1, 2, . . . , L}, number of latent tokens N , number of observation tokens P such that N + P = L, prior distribution p prior of latent variables over {1, . . . , m} and efficiently learnable observation functions O 1 , . . . , O P : {1, . . . , m} N → ∆({0, . . . , m}),foot_0
• (Latent tokens) For i = 1, . . . , N , sample x π(i) independently from the prior distribution p prior of the latents.
• (Observation tokens) For j = 1, . . . , P , sample x π(N +j) independently from O j (x π(1) , . . . , x π(N ) ).
L&O distributions contain two types of tokens: (1) latent tokens and (2) bservation tokens. Intuitively, latent tokens are tokens in the sequence, indexed by π(1), π(2), . . . , π(N ) that serve as "seeds" that provide randomness in the sequence; the remaining tokens, called observation tokens (indexed by π(N + 1), π(N + 2), . . . , π(N + P )), are determined as (possibly randomized) functions of the latent tokens via O 1 , . . . , O P .
Note that by design, order-aware training, e.g. by permuting the sequence so that π becomes the identity permutation and then performing autoregressive training, is computationally tractable: predicting x π(i) given x π(1) , . . . , x π(i-1) is trivial when i ≤ N as the tokens are independent, and computationally tractable when i > N because x π(i) only depends on x π(1) , . . . , x π(N ) and is efficiently learnable by assumption. In contrast, below we will show examples where if one performs order-agnostic training à la MDMs, one will run into hard masking problems with high probability.
First note that if the observations (O 1 , . . . , O P ) are given by a cryptographic hash function, then the masking problem of predicting (x π(1) , . . . , x π(L) ) given (x π(N +1) , . . . , x π(N +P ) ) is computationally intractable by design because it requires inverting the hash function. While this is a well-known folklore observation regarding the role of token ordering in language modeling, it is not entirely satisfying because this construction is worst-case in nature -in real-world data, one rarely trains on sequences given by cryptographic hash functions. Furthermore, it only establishes hardness for a specific masking pattern which need not be encountered in the course of running the reverse process.
We provide several simple instances of L&O distributions that address these issues: instead of leveraging delicate cryptographic constructions, they are average-case in nature and furthermore we can establish hardness for typical masking problems encountered along the reverse process.
In all these examples, the hardness results we establish hold even if the algorithm knows all of the parameters of p data as well as the observation functions O 1 , . . . , O P . Due to space constraints, here we focus on the following example, deferring two others to Apps. B.1 and B.2.
this section cite: []

Section: Example 3.2 (Sparse predicate observations).
Consider the following class of L&O distributions. Given arity k ≥ 2, fix a predicate function g : {1, . . . , m} k → {0, 1}. Consider the set of all ordered subsets of {1, 2, . . . , N } of size k and set the total number of observation latents P equal to the size of this set (hence
P = N !/(N -k)! = N (N -1) • • • (N -k + 1)).
To sample a new sequence, we first sample latent tokens x π(1) , . . . , x π(N ) from the prior distribution p prior and an observation latent corresponding to a k-sized subset S is given by g({x π(i) } i∈S ). In other words, each observation latent corresponds to a k-sized subset S of {1, 2, . . . , N } and the corresponding observation function O S (x π(1) , . . . , x π(N ) ) is given by g({x π(i) } i∈S ).
Proposition 3.3. Let x be a sample from an L&O distribution p data with sparse predicate observations as defined in Example 3.2, with arity k and predicate g satisfying Assumption B.11, and let γ be the probability that g is satisfied by a random assignment from {1, . . . , m} k . Let D KS and D cond be some constants associated with the predicate function g (see Definition B.12). Suppose each token in x is independently masked with probability α, and M is the set of indices for the masked tokens. If
1 -γ -1 D KS /kN k-1 ≤ α ≤ 1 -γ -1 D cond /kN k-1
, then under the 1RSB cavity prediction (see Conjecture B.13), with probability Ω k (1) over the randomness of the masking, no polynomial-time algorithm can solve the resulting subproblem of predicting any of the masked tokens among x π(1) , . . . , x π(N ) given x[M ].
The complete proof of the proposition is given in Appendix B.4. We also provide a proof outline in Appendix B.3 for a comprehensive understanding.
this section cite: []

Section: Empirical evidence of hardness via likelihoods
Recent studies (Nie et al., 2024;Zheng et al., 2024) have shown that masked diffusion models (MDMs) underperform compared to autoregressive models (ARMs) on natural text data. In this section, we provide evidence that this performance gap is primarily due to the order-agnostic training of MDMs. Since natural text follows a left-to-right token order, we demonstrate that as training deviates from this order, model performance gradually deteriorates.
To understand the importance of the order during the training, we use the following setting: Given a permutation π of indices {0, 1, . . . , L -1}, define a π-learner to be a likelihood model log p θ (x 0 ) given as follows:
log p θ (x 0 ) = L-1 i=0 log p θ x π(i) 0 x 0 [π{i, . . . , L -1}] (3)
In other words, the π-learner predicts the token at position π(i) given the clean tokens x π(0) 0 , . . . , x π(i-1) 0 and masked tokens x
π(i) 0 , . . . , x π(L-1) 0
. If π is the identity permutation, this reduces to the standard (left-to-right) autoregressive model. Note that the MDM loss encodes a π-learner for every permutation π because the MDM loss (1) is equivalent to the average loss of those π-learners over π sampled from Unif(S L ):
L θ = -Eπ,x 0 ∼p data L-1 i=0 log p θ x π(i) 0 x0[π{i, . . . , L -1}] ,
where S L denotes the set of all permutations over {0, 1, . . . , L -1}. The proof of the above equivalence is given in Appendix E. Therefore, by measuring the 'hardness' of each π-learner, we can probe differences in hardness between arbitrary masking problems and left-to-right masking problems.
Experimental setup. We use the Slimpajama dataset (Soboleva et al., 2023) to evaluate the performance of training in different orders. To train a π-learner, we employ a transformer with causal attention and use permuted data π(x 0 ) as input. By varying π while maintaining all other training configurations (e.g., model, optimization), we can use the resulting likelihood (computed using Equation ( 3)) as a metric to capture the hardness of subproblems solved by the π-learner.
In our experiments, the sequence length L is approximately 10 3 , so repeating the above for each π is infeasible. Instead, we sample π ∼ Unif(S L ) and examine the scaling law of the π-learner's likelihood. We leverage the codebase from (Nie et al., 2024), where the baseline scaling laws of MDM and ARM were introduced. Moreover, given that RoPE has an inductive bias towards left-to-right ordering, we employ a learnable positional embedding layer for all experiments to correct this. Consequently, we also re-run the baseline results, where RoPE was employed. To investigate how the distance between π and the identity permutation affects the scaling law, we sample π from other distributions interpolating between Unif(S L ) and the point mass at the identical permutation. Further experimental details are provided in Appendix C.1.
Results. As shown in Fig. 2, the scaling law for a π-learner with uniformly random π is worse than that of an ARM. This elucidates the inherent hardness of masking problems p θ (x i | x 0 [M ]) beyond left-to-right prediction and also explains why MDM, which is trained simultaneously on all π ∈ S L , is worse than ARM in likelihood modeling. Additionally, as π gets closer to the identity permutation, the scaling laws also get closer to ARM (π-learner-closer and π-learner-much-closer in Fig. 2). This also supports the common belief that ARM is a good fit for text data as it inherently follows a left-to-right ordering.
That said, it should also be noted that even though MDMs are trained on exponentially more masking problems than ARM (Θ(L2 L ) versus L), its performance is not significantly worse than π-learners. We attribute this to the blessing of task diversity; multi-task training can benefit both the optimization dynamics (Kim et al., 2024) and validation performance (Tripuraneni et al., 2021;Maurer et al., 2016;Ruder, 2017) due to positive transfers across tasks.
this section cite: ['b31', 'b54', 'b44', 'b31', 'b19', 'b48', 'b29', 'b38']

Section: Error is imbalanced across masking problems
In previous sections, we have demonstrated that the hardness of different masking problems p θ (x i | x 0 [M ]) can vary significantly, potentially hindering the MDM's learning. In this section, we provide empirical evidence that the MDM's final performance exhibits a similar imbalance across subproblems. Details are provided in App. C.2.
this section cite: []

Section: L&O-NAE-SAT.
Consider an L&O distribution with π given by the identity permutation and where each observation O j is deterministically given by NAE(
x i1 , x i2 , x i3 ) ≜ 1 -1[x i1 = x i2 = x i3 ]
for some randomly chosen (prefixed) triples (i 1 , i 2 , i 3 ) ∈ [N ]. For an MDM trained on this distribution, we measure the error it achieves on each task log p
θ (x 0 |x 0 [M ]) via E x0 log p θ (x 0 |x 0 [M ]) - log p data (x 0 |x 0 [M ]) 2
, where p data (x 0 |x 0 [M ]) denotes the Bayes-optimal predictor. Technically, we do not have access to this, so instead we train another MDM for a much larger number of iterations and use this as a proxy. Fig. 2 reveals that prediction tasks for latent positions (light region) exhibit larger errors compared to those for observation positions (dark region).
Text. Here we revisit the text experiment from Section 3.2.
Since we do not have access to the Bayes-optimal predictor, we use the metric
Ex 0 ∼p data L-1 i=0 log p θ x π(i) 0 x0[π{i, . . . , L -1}] .
This captures the accumulation of error across subproblems p θ x
π(i) 0 x 0 [π{i, . . . , L -1}] , since p θ (x 0 |x 0 [M ]) = p data (x 0 |x 0 [M ]
) minimizes this metric. Fig. 2 shows a clear gap between different subproblems.
The theoretical and empirical evidence demonstrates that MDMs perform better in estimating p θ (x 0 |x 0 [M ]) for some subproblems M than for others. We therefore want to avoid encountering hard subproblems M at inference time. In the next section, we show that while vanilla MDM inference can run into such subproblems, simple modifications at the inference stage can effectively circumvent these issues, resulting in dramatic, training-free performance improvements.
this section cite: []

Section: MDMs can plan around hard problems
We previously argued that due to the complex nature of masking subproblems, MDM must perform poorly on certain ones p θ (x i |x t ). Therefore, during vanilla MDM inference, MDM inevitably encounters such difficult subproblems at Step (b). While this might suggest that we need to fundamentally revisit how MDMs are trained, in this section we show that, surprisingly, simple modifications at the inference stage-without any further training-can sidestep these issues and lead to significant performance improvements.
MDM offers multiple sampling paths. The vanilla MDM inference (Algorithm 1) aim to align the intermediate distributions with the forward process, as used in continuous diffusion. However, unlike continuous diffusion, the reverse process of MDM allows multiple valid sampling paths (different orders of unmasking the tokens) that match the starting distribution of the forward process of MDM.
We first show that when we have an ideal MDM that perfectly solves all masking problems, i.e., p θ (
x i 0 |x 0 [M ]) = p data (x i 0 |x 0 [M ]
), then using any sampling path (unmasking the tokens in any order) results in the same distribution. Consider the following sampler: For every step, S is a set with one index selected agnostically (without following any distribution). For any clean sample x 0 generated by this sampler, note that p θ (
x 0 ) = L-1 i=0 p θ x π(i) 0 x 0 [π{i, . . . , L -1}] by chain rule, and this is equal to L-1 i=0 p data x π(i) 0 x 0 [π{i, . . . , L -1}] =
p data (x 0 ). Therefore, other choices of S, not necessarily following Algorithm 1, still capture the true likelihood.
In practice, unlike this ideal case, MDM does not perform equally well on all subproblems, as shown in Section 3.3. Consequently, different sampling paths result in varying likelihood modeling abilities. Motivated by this observation, we consider adaptive inference for MDMs:
Adaptive MDM inference (a) Sample a set of masked tokens S = F (θ, x t ) ⊆ {i | x i t = 0}. (b) For each i ∈ S, sample x i s ∼ p θ (x i |x t ).
Instead of selecting S randomly, adaptive MDM inference leverages an oracle F(θ, x t ) to select S strategically to avoid hard masking problems. This naturally raises the question of how to design an effective oracle F.
In the following sections, we demonstrate that adaptive MDM inference with careful choices of F enhance MDM's likelihood matching ability. In other words, a pretrained MDM, even if it performs poorly on certain hard subproblems, still contains sufficient information to avoid them when paired with an effective oracle F.
this section cite: []

Section: Effective design of ordering oracle
We introduce two different oracles, Top-K and Top-K probability margin. Intuitively, both strategies are based on the idea that S should be selected based on how "certain" the model is about each position. We caution that these strategies should not be confused with notions like nucleus sampling in ARMs (Holtzman et al., 2019); the oracles we describe are for selecting the position of the next token to decode, rather than the value, and thus are only meaningful in the context of MDMs.  Top-K probability (Zheng et al., 2023). Suppose we want to unmask K positions at time step t, i.e., select |S| = K. In the Top-K strategy, the uncertainty of a position is estimated by the maximum probability assigned to any value in the vocabulary. More precisely, the certainty at position i is max j∈{0,...,m-1} p θ (x i = j|x t ) and F(θ, x t ) = Top K max p θ (x i |x t ) .
Top-K strategy is a good proxy for many tasks and works well in practice (Zheng et al., 2023;Ye et al., 2024;Wang et al., 2024). However, this approach can often provide misleading estimates of uncertainty. Consider when an MDM is confused between two token values, thus assigning them almost equal but high probabilities. In this case, Top-K strategy may still choose to unmask this position, despite its uncertainty. To mitigate this issue, we propose the following alternative strategy.
Top-K probability margin. In this strategy, the uncertainty of a position is instead estimated using the absolute difference between the two most probable values at position i. More precisely, if j 1 and j 2 are the two most probable values in vocabulary according to p θ (x i |x t ) in position i, the certainty in the position is given by |p θ (
x i = j 1 |x t ) -p θ (x i = j 2 |x t )| and F(θ, x t ) = Top K |p θ (x i = j 1 |x t ) -p θ (x i = j 2 |x t )| .
When multiple values have similar probabilities at a position, Top-K probability margin will provide a better estimate of the uncertainty of a position, and when there is a single best choice of value then Top-K and Top-K probability margin work similarly.
this section cite: ['b15', 'b55', 'b55', 'b52', 'b50']

Section: Adaptive MDM inference
In this section, we experimentally validate that adaptive MDM inference helps MDMs avoid hard subproblems, leading to better likelihood matching. We first show our results on L&O-NAE-SAT and text data, before turning to our primary application to logic puzzles.
L&O-NAE-SAT and text data. For the L&O-NAE-SAT distribution defined in Section 3.3, we evaluate the effectiveness of adaptive inference by measuring the accuracy in predicting the observation tokens. Table 1 in the appendix reveals a clear improvement over vanilla inference. For the text dataset, we evaluate using the standard metric of generative perplexity, by which likelihood is measured by a large language model. We also compute the entropy of the generated samples to ensure both inference strategies exhibit similar levels of diversity. As shown in Fig. 3, we observe a substantial decrease in generative perplexity using adaptive inference. We defer further experimental details to Appendix D.1.
Logic puzzles. We consider two different types of logic puzzles: Sudoku and Zebra (Einstein) puzzles. Intuitively, for Sudoku, some empty (masked) cells are significantly easier to predict than others and we want to choose the cells that are easier to predict during the inference. We evaluate the effectiveness of adaptive MDM inference over vanilla MDM inference in selecting such cells. 2To measure the performance of an inference method, we use the percentage of correctly solved puzzles. For both puzzles, we use train and test datasets from (Shah et al., 2024). For the Sudoku puzzle (Table 2) we observe that adaptive MDM inference, in particular Top-K probability margin, obtains substantially higher accuracy (89.49%) compared to vanilla MDM inference (6.88%). Additionally, Top-K probability margin obtains higher accuracy (89.49%) than Top-K (18.51%). As mentioned in Section 4.1, this is because Top-K probability margin more reliably estimates uncertainty when multiple competing values are close in probability at a given position, as is often the case in Sudoku. For the Zebra puzzle, as shown in Table 3, we observe a consistent result: Top-K (98.5%) and Top-K probability margin (98.3%) outperform vanilla MDM inference (76.9%). In this section, we study the effectiveness of adaptive MDM inference in finding the right reasoning/generation order for tasks where every sequence has a different "natural" order.
To do so, we will compare the performance of adaptive MDM inference to that of ARM on Sudoku and Zebra puzzles. For these puzzles, the natural order of generation is not only different from left-to-right, but it is also sequencedependent. For such tasks, prior works have shown that ARMs struggle if the information about the order is not provided during the training (Shah et al., 2024;Lehnert et al., 2024). Therefore, to obtain a strong baseline, we not only consider an ARM trained without the order information but also consider an ARM trained with the order information for each sequence in the training data. Note that the latter is a much stronger baseline than the former as one can hope to teach the model to figure out the correct order by some form of supervised teacher forcing (as performed in Shah et al. (2024); Lehnert et al. (2024)), eliminating the issue of finding the right order in an unsupervised manner.
We compare ARMs and MDMs for Sudoku in Table 2 and Zebra puzzles in Table 3. We observe that for both, Top-K probability margin-based adaptive MDM inference not only outperforms the ARM trained without ordering information, but it even outperforms the ARM trained with ordering information! This shows that the unsupervised way of finding the correct order and solving such logic puzzles using adaptive MDM inference outperforms the supervised way of finding the correct order and solving such puzzles using an ARM, and is significantly less computationally intensive.
this section cite: ['b41', 'b41', 'b22']

Section: Adaptive MDM inference on text benchmarks
To examine the effect of different inference strategies on text benchmarks, we adapted LLaDA, the 8B MDM model from (Nie et al., 2025). We compare three inference strategies: Vanilla, Top-K probability, and Top-K probability margin .
The results are presented in Table 4.
We see that both adaptive MDM inference strategies, Top-K probability and Top-K probability margin , consistently outperform vanilla MDM inference. Notably, Top-K probability margin demonstrates a clear advantage over Top-K in challenging tasks like HumanEval-Multiline, HumanEval-Split Line, and Math. This is because Top-k Prob. Margin provides a more reliable estimate of uncertainty when multiple tokens have similar probabilities, a frequent occurrence in these difficult tasks. These results further underscore the potential for developing new, sophisticated adaptive inference strategies for various tasks.
this section cite: ['b32']

Section: Easy to hard generalization
In the previous section we showed that when the training and inference sequences come from the same distribution, order-agnostic training of MDMs combined with adaptive inference can perform very well on logic puzzles. To evaluate if the model has learned the correct way of solving the puzzles and test the robustness of adaptive inference, we  We measure the accuracy of MDMs and ARMs on the hard test set and present the results in Table 5. We see that the Top-K probability margin-based adaptive MDM inference strategy (49.88%) again significantly outperforms ARMs trained with order information (32.57%). In particular, although the accuracy drops for both methods due to the more challenging test set, MDMs with adaptive inference appear to be more robust to this distribution shift than ARMs. We believe this is due to the fact that MDMs try to solve a significantly higher number of infilling problems than ARMs (exp(L) compared to L) and therefore are able to extract knowledge about the problem more efficiently than ARMs.
this section cite: []

Section: Conclusion
In this work, we examined the impact of token ordering on training and inference in MDMs. We provided theoretical and experimental evidence that MDMs train on hard masking problems. We also demonstrated that adaptive inference strategies can be used to sidestep these hard problems. For logic puzzles, we find that this leads to dramatic improvements in performance not just over vanilla MDMs, but even over ARMs trained with teacher forcing to learn the right order of decoding.
An important direction for future work is to explore settings beyond logic puzzles where adaptive inference can help MDMs match or surpass ARMs. For these, it may be crucial to go beyond the relatively simple adaptive strategies like Top-K and Top-K probability margin considered here.
Acknowledgements. JK thanks Kiwhan Song for discussions about MDM training. KS and VK are supported by the NSF AI Institute for Foundations of Machine Learning (IFML). KS thanks Nishanth Dikkala for the initial discussions about the project. SC is supported by the Harvard Dean's Competitive Fund for Promising Scholarship and thanks Brice Huang and Sidhanth Mohanty for enlightening discussions about computational-statistical tradeoffs for planted CSPs.
this section cite: []

Section: References
Ref_id:b0 Title: Hardness of sampling solutions from the symmetric binary perceptron Year: (2024)
Ref_id:b1 Title: More on average case vs approximation complexity Year: (2003)
Ref_id:b2 Title: Storage capacity in symmetric binary perceptrons Year: (2019)
Ref_id:b3 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b4 Title: A coupling argument for the random transposition walk Year: (2011)
Ref_id:b5 Title: Masked generative image transformer Year: (2022)
Ref_id:b6 Title: Convergence analysis of discrete diffusion model: Exact implementation through uniformization Year: (2024)
Ref_id:b7 Title: Premise order matters in reasoning with large language models Year: (2024)
Ref_id:b8 Title: Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications Year: (2011-12)
Ref_id:b9 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b10 Title: The overlap gap property: A topological barrier to optimizing over random structures Year: (2021)
Ref_id:b11 Title: Reverse training to nurse the reversal curse Year: (2024)
Ref_id:b12 Title: Scaling diffusion language models via adaptation from autoregressive models Year: (2024)
Ref_id:b13 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b14 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b15 Title: The curious case of neural text degeneration Year: (2019)
Ref_id:b16 Title: Autoregressive diffusion models Year: ()
Ref_id:b17 Title: Argmax flows and multinomial diffusion: Learning categorical distributions Year: (2021)
Ref_id:b18 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b19 Title: Task diversity shortens the icl plateau Year: (2024)
Ref_id:b20 Title: The factorization curse: Which tokens you predict underlie the reversal curse and more Year: (2025)
Ref_id:b21 Title: Hiding quiet solutions in random constraint satisfaction problems Year: (2009)
Ref_id:b22 Title: Beyond a*: Better planning with transformers via search dynamics bootstrapping Year: (2024)
Ref_id:b23 Title: Probabilistically masked language model capable of autoregressive generation in arbitrary word order Year: (2020)
Ref_id:b24 Title: Discrete copula diffusion Year: (2024)
Ref_id:b25 Title: On statistical inference when fixed points of belief propagation are unstable Year: (2022)
Ref_id:b26 Title: Think while you generate: Discrete diffusion with planned denoising Year: (2024)
Ref_id:b27 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b28 Title: Discrete diffusion modeling by estimating the ratios of the data distribution Year: (2024)
Ref_id:b29 Title: The benefit of multitask representation learning Year: (2016)
Ref_id:b30 Title: Estimating random variables from random sparse observations Year: (2008)
Ref_id:b31 Title: Scaling up masked diffusion models on text Year: (2024)
Ref_id:b32 Title: Large language diffusion models Year: (2025)
Ref_id:b33 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2024)
Ref_id:b34 Title: Arrows of time for large language models Year: (2024)
Ref_id:b35 Title: Path planning for masked diffusion model sampling Year: (2025)
Ref_id:b36 Title: 3 million sudoku puzzles with ratings Year: (2020)
Ref_id:b37 Title: Steering masked discrete diffusion models via discrete denoising posterior prediction Year: (2024)
Ref_id:b38 Title: An overview of multi-task learning in deep neural networks Year: (2017)
Ref_id:b39 Title: Simple and effective masked diffusion language models Year: (2025)
Ref_id:b40 Title: Simple guidance mechanisms for discrete diffusion models Year: (2024)
Ref_id:b41 Title: Causal language modeling can elicit search and reasoning capabilities on logic puzzles Year: (2024)
Ref_id:b42 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b43 Title: Training and inference on any-order autoregressive models the right way Year: (2022)
Ref_id:b44 Title: Slimpajama: A 627b token cleaned and deduplicated version of redpajama Year: (2023-06)
Ref_id:b45 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b46 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b47 Title: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b48 Title: Provable metalearning of linear representations Year: (2021)
Ref_id:b49 Title: Glauber generative model: Discrete diffusion models via binary classification Year: (2024)
Ref_id:b50 Title: Diffusion language models are versatile protein learners Year: (2024)
Ref_id:b51 Title: Energy-based diffusion language models for text generation Year: (2024)
Ref_id:b52 Title: Discrete diffusion for complex reasoning and planning Year: (2024)
Ref_id:b53 Title: An open-source small language model Year: (2024)
Ref_id:b54 Title: Masked diffusion models are secretly timeagnostic masked models and exploit inaccurate categorical sampling Year: (2024)
Ref_id:b55 Title: A reparameterized discrete diffusion model for text generation Year: (2023)
