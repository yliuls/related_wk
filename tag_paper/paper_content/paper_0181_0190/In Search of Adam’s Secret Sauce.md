Title: In Search of Adam's Secret Sauce
Abstract: Understanding the remarkable efficacy of Adam when training transformer-based language models has become a central research topic within the optimization community. To gain deeper insights, several simplifications of Adam have been proposed, such as the signed gradient and signed momentum methods. In this work, we conduct an extensive empirical study -training over 1,500 language models across different data configurations and scales -comparing Adam to several known simplified variants. We find that signed momentum methods are faster than SGD, but consistently underperform relative to Adam, even after careful tuning of momentum, clipping setting and learning rates. However, our analysis reveals a compelling option that preserves near-optimal performance while allowing for new insightful reformulations: constraining the Adam momentum parameters to be equal, β 1 = β 2 . Beyond robust performance, this choice affords new theoretical insights, highlights the "secret sauce" on top of signed momentum, and grants a precise statistical interpretation: we show that Adam in this setting implements a natural online algorithm for estimating the mean and variance of gradients-one that arises from a mean-field Gaussian variational inference perspective.

Section: 
1 Introduction + # ) )( . + ( ) + +, + + , ) + + # ) + + # ) + + # 25% slower jama with Chinchilla-optimal [Hoffmann et al., 2022] scaling. Both momentum and learning rates for Signum are extensively tuned ( §3). While Signum closes 96% of the perplexity gap between Adam and SGD with momentum (Table 1), still results in a 25% slowdown : Adam achieves the same performance with 3/4 of the budget.
Despite a decade of research into efficient and performant adaptive optimizers for deep learning, the de facto choice for largescale training today remains Adam [Kingma and Ba, 2014], especially for training language models (LMs) [Grattafiori et al., 2024, Liu et al., 2024]. At the root of this choice is the peculiar geometry of optimization landscapes induced by the transformer architecture [Noci et al., 2022, Zhang et al., 2024a], as well as the noisy/unbalanced nature of tokenized text data [Zhang et al., 2020a, Kunstner et al., 2024].
In recent years, the surge of extremely large-scale and expensive-to-pretrain language models has further pushed the community to better understand Adam's performance and to propose faster, efficient, and robust alternatives. Towards achieving this goal, contemporary studies [Kunstner et al., 2023, Bernstein andNewhouse, 2024] have brought up a close similarity between the performance of Adam and SignSGD [Bernstein et al., 2018] with momentum. While such results are extremely valuable to forward our understanding, they are not precise enough : already at a scale of 160M parameters we found that extensive tuning of Signum (SignSGD with momentum), while closing 96% of the perplexity gap between SGD and Adam, results in a 25% effective slowdown (Figure 1).
Table 1: (Signum closes 96% of the perplexity gap between Adam and SGD) Validation perplexity comparison
of widely used optimizers that interpolate between SGD and Adam, evaluated on a language modeling task (160M parameters, 3.2B SlimPajama tokens, sequence length 2048, batch size 256 -Chinchilla optimal). We report the mean and 2-sigma interval of validation perplexity (on 100M held-out tokens) across 3 initialization seeds.
Weight decay is always decoupled [Loshchilov and Hutter, 2019] and set to 0.1 [Biderman et al., 2023, Liu et al., 2024] except for SGD where we further tune ( §B). RMSprop does not use momentum, and Gclip is global norm clipping to 1 (before applying momentum), Cclip is coordinate-wise clipping (after applying momentum).
Other hyperparameters, for all other methods, are carefully tuned, see e.g. Figure 2 and §3.
To optimally tune hyperparameters (e.g. Figure 2), we performed a total of 582 full training runs.
this section cite: ['b13', 'b11', 'b28', 'b19', 'b3', 'b25', 'b4']

Section: Adam Signum
RMSprop SGD+Cclip SignSGD SGD+Gclip SGD Val ppl. 21.86± 0.21 23.23± 0.16 27.04± 0.34 33.40± 0.39 36.78± 0.57 37.76± 0.61 53.62± 5.14 While for large-scale training, the slowdown in Figure 1 is not acceptable, it may seem unnecessary or anachronistic to further explain it, in light of recent algorithms claiming to have further improved the performance of Adam, e.g. Muon [Jordan et al., 2024, Liu et al., 2025, Shah et al., 2025], Scion [Pethick et al., 2025], and Shampoo-based [Gupta et al., 2018] methods such as SOAP [Vyas et al., 2025]. However, a close inspection of such optimizers reveals that, while gains over vanilla Adam are solid, most of these methods still use Adam on a specific subset of parameters: For instance, in recent scaled-up versions of Muon [Liu et al., 2025, Shah et al., 2025], Adam is used to update embedding, LM heads and normalization parametersfoot_0 , and on the other parameters the Muon update is normalized to have a similar RMS value similar to the Adam update. Further, SOAP's improvements stem from the application of Adam in the preconditioner's eigenbasis.
The discussion above and the results in Figure 1 inspires us to further dissect -once again [Balles and Hennig, 2018] -the mechanisms of Adam compared to those of simpler methods in language modeling with transformers. Towards improving our understanding of Adam, we make the following contributions:
• We perform a large-scale evaluation (∼ 10 thousand NVIDIA A100-SXM4-80GB GPU hours) of the performance of established algorithms which claim a theoretical or empirical similarity/dissimilarity with Adam on 160M parameters LMs with usual configurations [Biderman et al., 2023, Black et al., 2022], at a compute-optimal budget on different datasets, at different batch-sizes and sequence lengths (up to 2048 tokens). Crucially, we sweep over all momentum parameters for each method, for each learning rate in our grid -for each of our settings. We find that, while clipping and sign descent methods can close most of the gap with SGD, their performance is not satisfactory in comparison to Adam (Figure 2). We make all of our data, e.g. loss dynamics for all our settings, publicly available at https://github.com/aorvieto/SecretSauce.
• Through our extensive tuning of Adam (e.g., Figure 2, comprising 200 distinct hyperparameter settings), we identify one simplification that does perform well: that of setting β 1 = β 2 (emerging practical choice in contemporary literature [Zhao et al., 2025, Shah et al., 2025, Cattaneo and Shigida, 2025, Zhang et al., 2025]). We validate this finding ( §3.2) at different batchsizes, data source, token budget, sequence length and larger scale (410M): β 1 = β 2 performs at near-optimality across the majority of our experiments, see Figure 3. Given the breadth of our evaluation and the robustness of this finding, we recommend adopting β 1 = β 2 as the default setting for Adam for training language models at similar data and parameter scales. More broadly, this perspective suggests that Adam can be effectively treated as a one-parameter optimizer (as Signum).
• We show in §4, that reducing β 1 = β 2 = β to a single parameter, leads to a surprising new interpretation of Adam: it is built on top of a nontrivial yet principled online method for estimating mean and variance of the gradients. Indeed, we can view the two momentum buffers as the result of an online Gaussian Variational inference method for tracking the mean and variance of the gradients as they change across iterations. This viewpoint directly adds to the discussion by Balles and Hennig [2018], yet affords more precision induced by our empirically-informed simplification.
• We offer a toy quadratic example illustrative of our findings, building on top of recent works on the peculiar landscape of transformer-based language modeling problems [Noci et al., 2022, Zhang et al., 2024a]. This example replicates the gaps between tuned SGD, Signum, and Adam with equal betas in a 9-dimensional setting, helpful for future research and to gain intuition. Signum with weight decay 0.1 -top row -is around 23.23 (see Table 1 for multiple seeds at optimal tuning). We ablate on the momentum parameter, learning rate, and presence of global clipping before averaging. The best performance of Signum is reported as a green horizontal line on the second row (200 Adam runs, with weight decay of 0.1). Most Adam runs perform better than optimally tuned Signum. Takeaway 2: For each β1, the optimal corresponding β2 (after learning rate tuning) is similar. The higher β1, the higher β2 for optimal performance (optimal βs are correlated).
this section cite: ['b35', 'b32', 'b12', 'b44', 'b23', 'b35', 'b1', 'b4', 'b5', 'b57', 'b35', 'b6', 'b51', 'b1', 'b28']

Section: Preliminaries and Related Works
For a signal (s k ) k∈N and β ∈ [0, 1), we define the β-normalized exponential moving average:
EMA β [s k ] = βEMA β [s k-1 ] + (1 -β)s k , EMA β [s 0 ] := s 0 (or zero).(1)
The Adam optimizer [Kingma and Ba, 2014] without bias correctionfoot_1 takes the following form:
w k+1 = w k -η k EMA β2 [g 2 k ] + ϵ -1 EMA β1 [g k ](Adam)
where all division and multiplications are element-wise, w k , g k ∈ R d are model parameters and gradients at iteration k, η k is the scheduled learning rate, and ϵ > 0 is a small constant. RMSprop [Tieleman and Hinton, 2012] is an earlier method that sets β 1 = 0.
One special case, and simplification, of Adam is to consider β 1 = β 2 = ϵ = 0 which gives SignSGD:
w k+1 = w k -η k sign[g k ].(SignSGD)
A practical variant of SignSGD, which has shown strong performance in language modeling [Kunstner et al., 2023], first computes an exponential moving average (EMA) -or momentum -of the gradients before applying the sign operator [Bernstein et al., 2018]:
w k+1 = w k -η k sign[EMA β [g k ]].(Signum)
In practice, every language modeling pipeline (see e.g. [Karpathy, 2022]) incorporates some gradient clipping strategy [Pascanu et al., 2013], a component known to stabilize training in the autoregressive setting and to make gradients more robust to the stochasticity of language [Zhang et al., 2020b].
Global norm clipping (that we abbreviate Gclip), processes gradients fresh out of the backward pass:
Gclip[g k ] = min 1, 1 ∥g k ∥ 2 g k .
In our experiments, we start from vanilla SGD with momentum: w k+1 = w k -η k EMA β [g k ] and ablate on the positive effect of Gclip before applying momentum. Regarding coordinate clipping (Cclip), a softer version of sign, we consider applying it to EMA β [g k ] -in connection with Signum.
Research on Adam, a short summary. Despite initial concerns on generalization [Wilson et al., 2017] and convergence [Reddi et al., 2018], after the introduction of decoupled weight decay (i.e., AdamW [Loshchilov and Hutter, 2019]) Adam rapidly became the de-facto standard optimizer in deep learning, with works highlighting its landscape adaptation properties [Orvieto et al., 2022] and its debated connections to empirical Fisher preconditioning [Kunstner et al., 2019].
With the advent of Transformers [Vaswani et al., 2017], early works noticed an intriguing gap with SGD performance in language modeling [Xiong et al., 2020] (much larger than what can be observed, e.g., in CNNs on image data), that was initially attributed to heavy-tail noise in text data [Simsekli et al., 2019, Zhang et al., 2020a] -suggesting Adam performance to be correlated with its adaptive coordinate clipping mechanism [Zhang et al., 2020a].
As models became larger and more hardware-demanding, interest spiked in the community to reduce the memory footprint of Adam [Li et al., 2023, Zhang et al., 2024b] and to search for more efficient options [Chen et al., 2023, Liu et al., 2023]. Current trends, draw an intriguing connection between Adam and SignSGD [Bernstein and Newhouse, 2024], and in particular with its momentum variant: Signum [Bernstein et al., 2018]. This connection was first suggested in early attempts to understand Adam's empirical performance [Balles and Hennig, 2018], and has recently gained renewed attention in light of transformer architectures and their heterogeneous optimization landscapes [Noci et al., 2022, Zhang et al., 2024a, Tomihari and Sato, 2025, Kunstner et al., 2024, Zhao et al., 2025]. These landscape-based arguments are now more compelling, as recent evidence shows that Adam and signed momentum methods outperform SGD even in deterministic settings [Kunstner et al., 2023].
this section cite: ['b16', 'b40', 'b18', 'b3', 'b15', 'b30', 'b46', 'b34', 'b25', 'b29', 'b17', 'b43', 'b48', 'b37', 'b20', 'b7', 'b22', 'b2', 'b3', 'b1', 'b28', 'b41', 'b19', 'b18']

Section: Our approach.
Although recent literature highlights many connections between Adam and simpler methods such as Signum-which involve fewer hyperparameters, the computational demands of thoroughly studying Adam on small-to medium-scale language models remain prohibitive for most academic optimization researchers. This challenge is amplified by the combinatorial explosion of hyperparameter configurations required for rigorous comparisons. In §3, we aim to provide a comprehensive empirical reference for optimizer performance across a range of language modeling settings. Our key findings are distilled into two main takeaways (Figure 2), which are further supported by theoretical insights in §4.
this section cite: []

Section: Experiments
In our experiments, we systematically explore Transformer-based language models using a nanoGPT [Karpathy, 2022] implementationfoot_2 enhanced by recent advancements such as Rotational Positional Embeddings [Su et al., 2024], RMSNorm normalization [Zhang and Sennrich, 2019], and SwiGLU activation functions [Shazeer, 2020]. We adopt a robust training protocol inspired by successful practices established in large language models like LLaMa [Touvron et al., 2023], GPT-neox [Black et al., 2022], GPT-J [Wang and Komatsuzaki, 2022] and Pythia [Biderman et al., 2023], leveraging techniques including bfloat16 precision, linear warm-up followed by a cosine annealing schedule [Loshchilov and Hutter, 2016], and global gradient norm clipping (unless specified).
Our model configurations follow [Biderman et al., 2023] and are presented, alongside a detailed description of all tuning settings and resources, in §A.
this section cite: ['b15', 'b39', 'b50', 'b36', 'b42', 'b5', 'b45', 'b4', 'b24', 'b4']

Section: Extensive benchmarking at 160M parameters
We conduct 475 compute-optimal pretraining runs on the SlimPajama-627B dataset [Soboleva et al., 2023], using a sequence length of 2048, a batch size of 256, and a decoupled weight decay of 0.1 [Loshchilov and Hutter, 2019] (except for SGD). We always report validation perplexity on a held-out subset of 100M tokens. Results from these tuning sweeps are summarized in Table 1, Figure 2, and Appendix B.1. The runs span the following configurations:
• SGD (131 runs): Tuned parameters include weight decay (too large causes instability), global norm clipping (Gclip). We also consider clipping coordinates after applying momentum (Cclip). For all these options, momentum and learning rates are independently tuned.
• RMSprop (48 runs): Tuned parameters include momentum on the preconditioner and learning rate.
• Signum (70 runs): Tuned parameters include global norm clipping, momentum, and learning rate.
• Momentum on SignSGD (35 runs): This variant inverts the order of the sign and EMA operations (and performs worse than Signum). Clipping has no effect here due to the sign operation.
• AdamW (200 runs): Tuned parameters include both momentum terms and the learning rate.
Two additional seeds are provided for the best performing hyperparameter settings, see Table . 1.
Choice for betas grid. While we vary the learning rate by powers of two, our choice of moving average parameters is guided by recent insights into Adam scaling behavior [Malladi et al., 2022, Compagnoni et al., 2025]: we choose β = 1-κ(1-β base ) where β base = 0.9 and κ ∈ {2 -5 , 2 -4 , . . . , 2 2 }. This makes it such that the accumulation factor 1/(1 -β) = 1/(κ(1 -β base )).
Takeaway 1. As shown in Figure 2 and Table 1, optimally tuning Signum with weight decay leads to significant improvements over standard SGD, in line with recent findings [Kunstner et al., 2023, Zhao et al., 2025]. Nonetheless, Adam consistently outperforms the alternatives across most settings, suggesting that it retains a key advantage-a "secret sauce"-that continues to set it apart from better-understood methods in large-scale optimization tasks.
This gap is not limited to this specific setup. In §3.2 we discuss results on another dataset (Fineweb), with disabled weight decay, and shorter sequence lengths. Further, we ablate on other potential confounders (initialization of moving averages, bias corrections, Adam ϵ value) in §3.3.
batch size 512 batch size 256 batch size 128 Val ppl. gap to best 0 0.17 0.34 0.08 0 0.17 0.34 0.08 0 0.17 0.34 0.08 Val ppl. gap to best Val ppl. gap to best ) and the gap between its performance and that of other options in the grid. We notice high correlation between beta values (e.g., β2 = 0.9875 is a terrible option at β1 = 0.9, but a good one at β1 = 0.975). While results are noisy, notice that β1 = β2 never degrades performance more than 0.3 points. In contrast (Table 1, the gap with Signum can be as high as 1.37 points.
Takeaway 2 (a). In Figure 2, we clearly see that β 1 = β 2 yields near-optimal performance in Adam, for the five β 1 values we considered. In § 3.2 we show similar results at different batch sizes, different sequence lengths, and with disabled weight decay on a different dataset. We also extend this observation to 410M parameters models (Figure 5). This empirical finding serves as a basis for our theory in §4.
Takeaway 2 (b). As a corollary to Takeaway 2, Figure 3 shows that the best performance is not only achieved when β 1 = β 2 , but also improves as the two values become closer. Among 500 runs on 160M-parameter models, we observe a clear correlation: lower loss is associated with smaller differences between β 1 and β 2 . This suggests that gradient smoothing (β 1 ) and preconditioner smoothing (β 2 ) should not be treated as independent operations-optimal performance often arises when they act in concert.
To put to the test our second takeaway in different training settings, we consider shorter sequence lengths (512, Figure 14), higher/lower batch sizes (Figure 16 & Figure 17), different data (Fineweb) and absence of weight decay (Fig, 18). See discussion in §3.2.
Standard choice for betas. While in standard deep learning (also Pytorch default) β 2 > β 1 (0.999, 0.9), in language modeling the choice β 1 = 0.9, β 2 = 0.95 is much more common. A lower value for β 2 was shown to help mitigate loss spikes [Cattaneo andShigida, 2025, Compagnoni et al., 2025], and several recent studies have started to adopt β 1 = β 2 = 0.95 as a default [Zhao et al., 2025, Shah et al., 2025, Zhang et al., 2025]. All our findings confirm this choice for tuning (see e.g. Figure 2), of which we evaluate validity extensively for several values of β 1 .
10 -3 10 -2 learning rate 15.7 15.8 15.9 16.0 16.1 16.2 16.3 16.4 16.5 final test ppl best with equal betas AdamW, β1 = 0.9 10 -3 10 -2 learning rate best with equal betas AdamW, β1 = 0.95 10 -3 10 -2 learning rate best with equal betas AdamW, β1 = 0.975 β2 = 0.8 β2 = 0.9 β2 = 0.95 β2 = 0.975 β2 = 0.9875 β2 = 0.99375
Figure 5: The final validation performance (100M held-out tokens) for 44 trained LMs with 410M parameters trained on 8.2 B SlimPajama tokens (Chinchilla-optimal). Equal betas yields near-optimal performance. We use gradient clipping and a batch size of 512 (scaled by 2 compared to Figure 2, as suggested by Zhang et al. [2025]). Sequence length is 2048, weight decay is 0.1. Note that the standard setting (0.9, 0.95) is quite suboptimal here.
Theoretical relations between betas. We note that a correlation between β parameters was also noted first by Reddi et al. [2018], Alacaoglu et al. [2020] for AMSgrad, and later by Zhang et al. [2022] for Adam, where it is shown that if β 2 is large enough and β 1 < √ β 2 , it converges to the neighborhood of critical points. Further, Xie and Li [2024] showed that weight decay in AdamW leads to convergence to a constrained minimizer only if β 2 > β 1 .
this section cite: ['b38', 'b25', 'b26', 'b9', 'b18', 'b6', 'b57', 'b35', 'b51', 'b51', 'b34', 'b0', 'b54', 'b47']

Section: Ablations
More Tokens. We find our Takeaway 2 to also hold at a higher token budget. In §B.2, we show a trend very similar to Fig. 2 for models trained for 2× the Chinchilla-optimal budget. Different batch size. We find our Takeaway 2 to be robust to batch size. In the same setting as Figure 2 yet at a slightly lower compute budget due to hardware limitations (2.5B parameters), we find that, even at batch size 128 and 512 the choice β 1 = β 2 yields near-optimal performance. This step involves training 500 models, see §B.4 for visualizations similar to Figure 2 and a discussion.
Different sequence length. In §B.3, we find our Takeaway 2 to also hold at shorter sequence length of 512 (Figure 14). We note that here performance of Signum is closer to that of Adam compared to Figure 2 -yet, Adam is still superior by a substantial margin ( 0.7 validation perplexity), Takeaway 1. This pattern agrees well with the results in [Zhao et al., 2025], who found other methods to be competitive with Adam at short context lengths. Our experiments in Figure 14 and Figure 2 suggest that Adam performance particularly shines at higher sequence lengths.
Different data and weight decay. In Figure 18 we test both Takeaway 1 and Takeaway 2 on Fineweb [Penedo et al., 2024]. We take this opportunity to also deactivate weight decay (λ = 0), as the optimal Signum learning rates in Figure 2 suggest decoupled weight decay w = w -ληw acts differently for the two methods, likely needing different tuning. When deactivated, we still see a substantial gap between Signum and Adam, as well as strong performance with equal betas.
) ( ) ( ) ( ) ( )
1e-03
Figure 4: Adding an ϵ mollifier to Signum, i.e., using m k /( m 2 k + ϵ) offered little to no improvement. We also test both zero initialization (ZI) and gradient initialization (GI) for m, and find similar results with no significant improvement. ϵ = 1e -3 is significantly worse, hence is not shown. Similar finding: Figure 7.
Larger Models. We restrict our attention to the SlimPajama dataset and to validation of Takeaway 2. Results are presented in Figure 5, comprising 44 full compute-optimal training runs of 410M parameter models, which confirm yet again strong and near-optimal performance at β 1 = β 2 .
this section cite: ['b57', 'b31']

Section: Checking for confounders
When comparing Signum with Adam, here are a few confounders that might affect results: The value of ϵ in Adam was shown to be important for numerical stability, and might affect performance [Yuan and Gao, 2020]. We show in Table 2 that one can choose an extremely small ϵ value in our setting. We cross-check the impact of including an ϵ factor in Signum: we found that little can be gained from this strategy (Figure 4). In short, we found that ϵ is not a crucial parameter in our setup. This is also liked to our findings on adaptive mollifiers, cf. §4.
this section cite: ['b49']

Section: Initialization of moving average parameters.
In Figure 4 we also ablate on initialization of the moving average in Signum and found no substantial differences. We perform this same ablation for Adam and report comprehensive results with seeds in §B.6.
this section cite: []

Section: Bias correction.
While bias correction in Adam is helpful in early training, final validation performance is almost unchanged, see the full training curve and results with seeds in §B.6.
this section cite: []

Section: New Viewpoints of Adam
We now show that restricting to the case β 1 = β 2 = β yields a useful interpretation of Adam. Since the Adam update is coordinate-wise, it suffices to analyze a single scalar gradient g k ∈ R. Moreover, ablations (Table 2, Table 3) indicate that neither the ϵ-term nor the bias correction significantly affect performance. Thus, for clarity, we set ϵ = 0 and study the simplified Adam update:
d k = EMA β [g k ] EMA β [g 2 k ]
.
(
We next rewrite (proof in the Appendix) the update to explicitly highlight the role of variance.
Proposition 1. Let m k = EMA β [g k ].
Then the update (2) admits the equivalent representation:
d k = m k m 2 k + β EMA β [(m k-1 -g k ) 2 ] .(3)
This shows that the denominator depends on the exponential moving average of the squared deviation between the momentum m k-1 and the incoming gradients g k , with an interesting multiplier β. As we demonstrate in the next section, this quantity is in fact an online estimator of the gradient variance.
this section cite: []

Section: Adam Estimates Mean and Variance using Variational Inference
We show that Adam admits a natural interpretation as an online variational inference method, where
m k := EMA β [g k ] and σ 2 k := β EMA β [(m k-1 -g k ) 2 ]
correspond to online estimates of the mean and variance of the stochastic gradients. We reintroduce Adam through this lens.
Suppose we are given a sequence of stochastic gradients {g 1 , . . . , g k }, where each g k is sampled from an unknown Gaussian distribution whose mean and variance may vary with k. Rather than taking steps directly along these noisy gradients, we aim to estimate their mean and variance online and use these estimates to define a more informed search direction.
At iteration k, let (m k , σ 2 k ) denote our current estimates of the gradient mean and variance, respectively. Upon receiving a new gradient sample g k+1 ∼ N (m, σ 2 ) with unknown (m, σ 2 ), we wish to update our estimates to (m k+1 , σ 2 k+1 ) so that it becomes more likely that g k+1 was drawn from N (m k+1 , σ 2 k+1 ). Since we also expect the underlying distribution to vary slowly over time, we prefer that N (m k+1 , σ 2 k+1 ) remain close to the previous estimate N (m k , σ 2 k ). These two goals-fitting the new observation and ensuring smooth updates-can be traded off via a regularized maximum likelihood problem:
min m,σ 2 ≥0 -log p(g k+1 | m, σ 2 ) + 1 λ KL N (m k , σ 2 k ) ∥ N (m, σ 2 ) ,(4)
where p(g k+1 | m, σ 2 ) is the Gaussian likelihood, λ ≥ 0 is a regularization parameter, and KL denotes the Kullback-Leibler divergence:
-log p(g k+1 | m, σ 2 ) = 1 2 log σ 2 + 1 2σ 2 (g k+1 -m) 2 , (5
) KL N (m k , σ 2 k ) ∥ N (m, σ 2 ) = 1 2 σ 2 k σ 2 + (m k -m) 2 σ 2 -1 -log σ 2 k σ 2 . (6
)
The following result, proved in the appendix, characterizes the solution of (4), showing that the moving averages used in Adam correspond exactly to an instance of online variational inference:
Theorem 4.1. Let β = 1 1+λ . Then the solution to the optimization problem (4) is given by
m k+1 = βm k + (1 -β)g k+1 = EMA β [g k+1 ],(7)
σ 2 k+1 = βσ 2 k + β(1 -β)(m k -g k+1 ) 2 = β EMA β (m k -g k+1 ) 2 . (8
)
As a consequence, the Adam update direction in (3) can be rewritten as
d k = m k m 2 k + βEMA β [(m k-1 -g k ) 2 ] = m k m 2 k + σ 2 k = sign(m k ) 1 + σ 2 k /m 2 k . (9
)
This shows that Adam may be interpreted as an adaptive mollified variant of Signum, where the mollification depends on the local noise-to-signal ratio. This mollified viewpoint aligns well with one of the first papers on understanding Adam [Balles and Hennig, 2018], as discussed after Proposition 1.
Using these insights, we can better formalize the noise-to-signal interpretation of Adam [Balles and Hennig, 2018] (see also §4.2). Let m k /σ k denote the signal-to-noise ratio (SNR). We show that Adam can be viewed as a steepest descent method whose trust region is modulated by the SNR.
To build this connection, consider first the Signum update. It corresponds to the steepest descent direction under an ℓ ∞ -norm constraint [Balles and Hennig, 2018], solving
-sign(m k ) = argmin θ∈R -m k • θ subject to |θ| ≤ 1. (10
)
That is, Signum selects the direction most aligned with -m k within a unit trust region.
In contrast, Adam can be interpreted as a steepest descent method with a variable trust region, defined by the (inverse) signal-to-noise ratio:
- sign(m k ) 1 + σ 2 k /m 2 k = argmin θ∈R -m k • θ subject to |θ| ≤ 1 1 + σ 2 k /m 2 k .(11)
Here, the effective step size shrinks when the noise dominates the signal, and expands toward 1 as uncertainty decreases. In this sense, Adam adapts its update magnitude according to a confidenceweighted trust region.
this section cite: ['b1', 'b1', 'b1']

Section: Comparison with Balles and Hennig [2018]
Balles and Hennig [2018] first drew a connection between Adam, Signum and Signal-to-noise Ratio regularization. Their observation was as follows. Let
m k = EMA β1 [g k ], and v k = EMA β2 [g 2 k ].
We can trivially re-write the Adam direction as
d k = m k √ v k = m k m 2 k + v k -m 2 k .
If we now assume that σ 2 k := v k -m 2 k is a measure of variance, then dividing the Adam direction through by |m k |, as done in (9), we arrive at a Signal-to-noise Ratio regularized variant of the Signum method. In particular, as the noise goes to zero (σ 2 k → 0), we arrive at the Signum method. The missing piece in their insight was to show when and if the term v k -m 2 k is a measure of variance. We show that β 1 = β 2 , a choice that was not commonfoot_3 at the time of Balles and Hennig [2018], allows for more precise claims: Proposition 1 shows that when
β 1 = β 2 = β the term v k -m 2 k is precisely equal to βEMA β [(m k-1 -g k ) 2
], which in turn is a online estimate of variance (Theorem 4.1). We further show that v k -m 2 k only has a precise variance interpretation for the case β 1 = β 2 : indeed, we prove in §C.2 that Adam can be represented as
d k = m k m 2 k + γ EMA τ [(am k-1 -bg k ) 2 ](12)
for some a, b, γ ∈ R and τ ∈ (0, 1) if and only if β 1 = β 2 . In other words, connecting v k -m 2 k to variance estimation, and in turn Adam to an SNR-controlled trust region method (11), can only be done precisely for the case of equal betas.
Ablating hyperparameters in our reformulation. While (12) reduces to Adam with equal betas if and only if a, b = 1 and β = γ = τ , we found it interesting to consider (12), with a = b = 1, as a new method with no precise connection to simultaneous variance and mean estimation, with hyperparameters β, γ, τ . In §C.4, we train 150 additional language models ablating on such parameters, and found no advantage in setting β ̸ = τ or τ ̸ = γ. We believe such evidence further strengthens our claims: best performance is aligned to the theoretical choice τ = γ = β.
5 Why an adaptive trust region? Insights from heterogeneous quadratics While our theoretical analysis in §4 offers a new perspective on Adam, it is not tied to any specific architecture. To enhance intuition and provide a controlled setting for future work, we validate our findings on a simplified model of transformer loss landscapes introduced by Zhang et al. [2024a], building on signal propagation theory [Noci et al., 2022]. As noted in Zhang et al. [2024a], Kunstner et al. [2024], Zhao et al. [2025], the landscape of autoregressive language models is highly heterogeneous: Hessian blocks associated with semantically distinct parameter groups (e.g., normalization layers, embeddings, or softmax-related parameters) exhibit markedly different eigenspectra and thus demand different learning rates. In contrast to homogeneous models (e.g., CNNs), this heterogeneity is where Adam significantly outperforms SGD [cf. Zucchet and Orvieto, 2024]. [2024a]. We also observe that Signum closes much of the gap but still falls short of Adam. This is consistent with our findings in Table 1 for language models. In Proposition 1, we showed that the key difference between Signum and Adam lies in the variance correction term βEMA β [(m k-1 -g k ) 2 ] in the denominator. Understanding how this term evolves is essential: it cannot be approximated by a constant. In the second row of Figure 6, we observe that the variance estimate not only varies over time, but also differs in scale across the three blocks-mimicking the parameter groupings in transformer models. This block-wise variation reinforces the idea that the variance term dynamically adapts to the local curvature and cannot be substituted by a fixed value. In Figure 7 and 4, we show a similar effect in heterogeneous quadratic and language models, respectively: replacing
βEMA β [(m k-1 -g k ) 2
] with a fixed constant ϵ cannot provide the same adaptive effect.
this section cite: ['b1', 'b28', 'b19', 'b57', 'b58']

Section: Conclusion
We have presented an extensive numerical study of Adam, comparing it against several proposed simplifications. Our main finding is that, on generative language modeling tasks, Adam significantly outperforms these simplified variants. Notably, we observe that setting β 1 = β 2 is often optimal or near-optimal. Based on this observation, we recommend Adam with β 1 = β 2 as a simplified model, and we provide a new variational inference interpretation for this setting.
Our findings come with some limitations. First, our numerical experiments fix a grid over the hyperparameters; the results are therefore sensitive to the choice of grid, and different grids may lead to different conclusions. However, for all our hyperparameters, we show explicitly all tuning curves demonstrating that we are always at optimality inside the grid (and not at the edge). Second, while β 1 = β 2 often performs well, we note that at small batch sizes, Figure 3 suggests a slight shift. Finally, although Theorem 4.1 shows that Adam's two momentum buffers can be interpreted as online estimates of the gradient's mean and variance, it does not explain why these estimates should be arranged into the quotient used in Adam (9). Lemma 1 in [Balles and Hennig, 2018] can provide a starting point to further dissect this interesting choice and explore alternatives.
Contents 1 Introduction 2 Preliminaries and Related Works 3 Experiments 3.1 Extensive benchmarking at 160M parameters . . . . . . . . . . . . . . . . . . . . 3.2 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3.3 Checking for confounders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4 New Viewpoints of Adam 4.1 Adam Estimates Mean and Variance using Variational Inference . . . . . . . . . . 4.2 Comparison with Balles and Hennig [2018] . . . . . . . . . . . . . . . . . . . . . 5 Why an adaptive trust region? Insights from heterogeneous quadratics 6 Conclusion A Experimental details A.1 Experiments on SlimPajama -160M parameters model . . . . . . . . . . . . . . . A.1.1 Sequence Length 2048, Batch size 256, 3.2 B Tokens (6200 gradient steps) A.1.2 Sequence Length 2048, Batch size 256, 6.4 B Tokens (12400 gradient steps) A.1.3 Sequence Length 512, Batch size 256, 3.2 B Tokens (24800 gradient steps) A.1.4 Sequence Length 2048, Variable batch size, 2.5 B Tokens . . . . . . . . . . A.2 Experiments on SlimPajama -410M parameters model, 8.2 B tokens . . . . . . . . A.3 Experiments on Fineweb -160M parameters model, 3.2B tokens -no weight decay B Complementary Experimental Results B.1 Tuning for Table 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . B.2 Effect of More Training Tokens in Figure
this section cite: ['b1']

Section: References
Ref_id:b0 Title: A new regret analysis for adam-type algorithms Year: (2020)
Ref_id:b1 Title: Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients Year: (2018)
Ref_id:b2 Title: Old optimizer, new norm: An anthology Year: (2024)
Ref_id:b3 Title: signsgd: Compressed optimisation for non-convex problems Year: (2018)
Ref_id:b4 Title: Pythia: A suite for analyzing large language models across training and scaling Year: (2023)
Ref_id:b5 Title: Gpt-neox-20b: An open-source autoregressive language model Year: (2022)
Ref_id:b6 Title: Tuning adam(w): Default β 2 may be too large Year: (2025)
Ref_id:b7 Title: Symbolic discovery of optimization algorithms Year: (2023)
Ref_id:b8 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b9 Title: Adaptive methods through the lens of SDEs: Theoretical insights on the role of noise Year: (2025)
Ref_id:b10 Title: Flashattention: Fast and memoryefficient exact attention with io-awareness Year: (2022)
Ref_id:b11 Title: The llama 3 herd of models Year: (2024)
Ref_id:b12 Title: Shampoo: Preconditioned stochastic tensor optimization Year: (2018)
Ref_id:b13 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b14 Title: Muon: An optimizer for hidden layers in neural networks Year: (2024)
Ref_id:b15 Title:  Year: (2022)
Ref_id:b16 Title: Adam: A method for stochastic optimization Year: (2014)
Ref_id:b17 Title: Limitations of the empirical fisher approximation for natural gradient descent Year: (2019)
Ref_id:b18 Title: Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be Year: (2023)
Ref_id:b19 Title: Heavy-tailed class imbalance and why adam outperforms gradient descent on language models Year: (2024)
Ref_id:b20 Title: Memory efficient optimizers with 4-bit states Year: (2023)
Ref_id:b21 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b22 Title: A scalable stochastic second-order optimizer for language model pre-training Year: (2023)
Ref_id:b23 Title: Muon is scalable for LLM training Year: (2025)
Ref_id:b24 Title: Stochastic gradient descent with warm restarts Year: (2016)
Ref_id:b25 Title: Decoupled weight decay regularization Year: (2019)
Ref_id:b26 Title: On the sdes and scaling rules for adaptive gradient algorithms Year: (2022)
Ref_id:b27 Title: Transformers without tears: Improving the normalization of self-attention Year: (2019)
Ref_id:b28 Title: Signal propagation in transformers: Theoretical perspectives and the role of rank collapse Year: (2022)
Ref_id:b29 Title: Vanishing curvature in randomly initialized deep relu networks Year: (2022)
Ref_id:b30 Title: On the difficulty of training recurrent neural networks Year: (2013)
Ref_id:b31 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b32 Title: Training deep learning models with norm-constrained lmos Year: (2025)
Ref_id:b33 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b34 Title: On the convergence of adam and beyond Year: (2018)
Ref_id:b35 Title: Practical efficiency of muon for pretraining Year: (2025)
Ref_id:b36 Title: Glu variants improve transformer Year: (2020)
Ref_id:b37 Title: A tail-index analysis of stochastic gradient noise in deep neural networks Year: (2019)
Ref_id:b38 Title: SlimPajama: A 627B token cleaned and deduplicated version of RedPajama Year: (2023)
Ref_id:b39 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b40 Title: Lecture 6.5-rmsprop, coursera: Neural networks for machine learning Year: (2012)
Ref_id:b41 Title: Understanding why adam outperforms sgd: Gradient heterogeneity in transformers Year: (2025)
Ref_id:b42 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b43 Title: Attention is all you need Year: (2017)
Ref_id:b44 Title: SOAP: Improving and stabilizing shampoo using adam for language modeling Year: (2025)
Ref_id:b45 Title: Gpt-j-6b: A 6 billion parameter autoregressive language model Year: (2021)
Ref_id:b46 Title: The marginal value of adaptive gradient methods in machine learning Year: (2017)
Ref_id:b47 Title: Implicit bias of adamw: ℓ ∞ -norm constrained optimization Year: (2024)
Ref_id:b48 Title: On layer normalization in the transformer architecture Year: (2020)
Ref_id:b49 Title: Eadam optimizer: How ϵ impact adam Year: (2020)
Ref_id:b50 Title: Root mean square layer normalization Year: (2019)
Ref_id:b51 Title: How does critical batch size scale in pre-training? Year: (2025)
Ref_id:b52 Title: Why gradient clipping accelerates training: A theoretical justification for adaptivity Year: ()
Ref_id:b53 Title: Why are adaptive methods good for attention models? Year: (2020)
Ref_id:b54 Title: Adam can converge without any modification on update rules Year: (2022)
Ref_id:b55 Title: Why transformers need adam: A hessian perspective Year: (2024)
Ref_id:b56 Title: Adam-mini: Use fewer learning rates to gain more Year: (2024)
Ref_id:b57 Title: Deconstructing what makes a good optimizer for autoregressive language models Year: (2025)
Ref_id:b58 Title: Recurrent neural networks: vanishing and exploding gradients are not the end of the story Year: (2024)
