Title: Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
Abstract: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling to arbitrary depth at test-time. This stands in contrast to mainstream reasoning models that scale up compute by producing more tokens. Unlike approaches based on chain-of-thought, our approach does not require any specialized training data, can work with small context windows, and can capture types of reasoning that are not easily represented in words. We train a proof-of-concept model from scratch with 3.5 billion parameters and 800 billion tokens. We show that this model can effortlessly use varying levels of compute, significantly improving with additional compute especially on reasoning tasks, such as math and coding. Further, this architecture naturally reduces compute costs via zero-shot per-token adaptive compute, KV-cache sharing and speculative decoding. P "Hello" "World" Input Injection Residual Stream Prelude Recurrent Block Coda 𝒩(0,σ 2 I n⋅h )

Section: Scaling by Thinking in Continuous Space
Humans naturally expend more mental effort solving some problems than others. While humans are capable of thinking over long time spans by verbalizing intermediate results and writing them down, a substantial amount of thought happens through complex, recurrent firing patterns in the brain, before the first word of an answer is uttered.
Early attempts at increasing the power of language models focused on scaling model size, a practice that requires extreme amounts of data and computation. More recently, researchers have explored ways to enhance the reasoning capability of models by scaling test time computation. The mainstream approach involves post-training on long chain-of-thought examples to develop the model's ability to verbalize intermediate calculations in its context window and thereby externalize thoughts.
However, the constraint that expensive internal reasoning must always be projected down to a single verbalized next token appears wasteful; it is plausible that models could be more competent if they were able to natively "think" in their continuous latent space. One way to unlock this untapped dimension of additional compute involves adding a recurrent unit to a model. This unit runs in a loop, iteratively processing and updating its hidden state and enabling computations to be carried on indefinitely. While this is not currently the dominant paradigm, this idea is foundational to machine learning and has been (re-)discovered in every decade, for example as recurrent neural networks, diffusion models, feature recycling, and as universal or looped transformers.
In this work, we show that depth-recurrent language models can learn effectively, be trained in an efficient manner, and demonstrate significant performance improvements under the scaling of test-time compute. Our proposed transformer architecture is built upon a latent depth-recurrent 39th Conference on Neural Information Processing Systems (NeurIPS 2025). block that is run for a randomly sampled number of iterations during training. We show that this paradigm can scale to several billion parameters and over half a trillion tokens of pretraining data. At test-time, the model can improve its performance through recurrent reasoning in latent space, enabling it to compete with other open-source models that benefit from more parameters and training data. Additionally, we show that recurrent depth models naturally support a number of features at inference time that require substantial tuning and research effort in non-recurrent models, such as per-token adaptive compute, (self)-speculative decoding, and KV-cache sharing. We finish our study by tracking token trajectories in latent space, showing that a number of interesting computation behaviors simply emerge with scale, such as the model rotating shapes in latent space for numerical computations.
2 Why Train Models with Recurrent Depth?
Recurrent layers enable a transformer model to perform arbitrarily many computations before emitting a token. In principle, recurrent mechanisms provide a simple solution for test-time compute scaling. Compared to a more standard approach of long context reasoning OpenAI (2024); DeepSeek-AI et al. (2025), latent recurrent thinking has several advantages.
• Latent reasoning does not require construction of bespoke training data. Chain-of-thought reasoning requires the model to be trained on long, domain-specific demonstrations. Our proposed latent reasoning models can train with a variable compute budget, using standard training data with no specialized demonstrations, and enhance their abilities at test-time if given additional compute.
• Latent reasoning models require less memory for training and inference than chain-of-thought reasoning models. Because the latter require extremely long context windows, specialized training methods such as token-parallelization Liu et al. (2023a) may be needed.
• Recurrent-depth networks perform more FLOPs per parameter than standard transformers, significantly reducing communication costs between accelerators at scale. This especially enables higher device utilization when training with slower interconnects.
• By constructing an architecture that is compute-heavy and small in parameter count, we hope to set a strong prior towards models that solve problems by "thinking", i.e. by learning meta-strategies, logic and abstraction, instead of memorizing. The strength of recurrent priors for learning complex algorithms has already been demonstrated in the "deep thinking" literature Schwarzschild et al. (2021b); Bansal et al. (2022); Schwarzschild et al. (2023).
On a more philosophical note, we hope that latent reasoning captures facets of human reasoning that defy verbalization, such as spatial thinking, physical intuition or (motor) planning. Over many iterations of the recurrent process, reasoning in a high-dimensional vector space would enable the deep exploration of multiple directions simultaneously, instead of linear thinking, leading to a system capable of exhibiting novel and complex reasoning behavior.
Scaling compute in this manner is not at odds with scaling through extended (verbalized) inference scaling (Shao et al., 2024), or scaling parameter counts in pretraining (Kaplan et al., 2020), we argue it may build a third axis on which to scale model performance.
this section cite: ['b111', 'b38', 'b13', 'b128', 'b130', 'b73']

Section: A Scalable Recurrent Architecture
In this section we will describe our proposed architecture for a transformer with latent recurrent depth, discussing design choices and small-scale ablations. A diagram of the architecture can be found in Figure 2. We always refer to the sequence dimension as n, the hidden dimension of the model as h, and its vocabulary as the set V .
this section cite: []

Section: Macroscopic Design
The model is primarily structured around decoder-only transformer blocks (Vaswani et al., 2017;Radford et al., 2019). However these blocks are structured into three functional groups, the prelude P , which embeds the input data into a latent space using multiple transformer layers, then the core recurrent block R, which is the central unit of recurrent computation modifying states s ∈ R n×h , and finally the coda C, which un-embeds from latent space using several layers and also contains the prediction head of the model. The core block is set between the prelude and coda blocks, and by looping the core we can put an indefinite amount of verses in our song.
Given a number of recurrent iterations r, and a sequence of input tokens x ∈ V n these groups are used in the following way to produce output probabilities p ∈ R n×|V | e = P (x)
s 0 ∼ N (0, σ 2 I n•h ) s i = R(e, s i-1 ) for i ∈ {1, . . . , r} p = C(s r ),
where σ is some standard deviation for initializing the random state. This process is shown in Figure 2. Given an init random state s 0 , the model repeatedly applies the core block R, which accepts the latent state s i-1 and the embedded input e and outputs a new latent state s i . After finishing all iterations, the coda block processes the last state and produces the probabilities of the next token. This architecture is based on deep thinking literature, where it is shown that injecting the inputs e in every step (Bansal et al., 2022) and initializing the latent vector with a random state stabilizes the recurrence and promotes convergence to a steady state independent of initialization, i.e. path independence (Anil et al., 2022).
this section cite: ['b152', 'b114', 'b13', 'b6']

Section: Microscopic Design
Within each group, we broadly follow standard transformer layer design. Each block contains multiple layers, and each layer contains a standard, causal self-attention block using RoPE (Su et al., 2021) with a base of 50000, and a gated SiLU MLP (Shazeer, 2020). We use RMSNorm (Zhang and Sennrich, 2019) as our normalization function. The model has learnable biases on queries and keys, and nowhere else. To stabilize the recurrence, we order all layers in the following "sandwich" format, using norm layers n i , related to Ding et al. (2021); Team Gemma et al. (2024):
xl =n 2 (x l-1 + Attn(n 1 (x l-1 ))) x l =n 4 ( xl + MLP(n 3 ( xl )))
While at small scales, most normalization strategies, e.g. pre-norm, post-norm and others, work almost equally well, we ablate these options and find that this normalization is required to train the recurrence at scale.
Given an embedding matrix E and embedding scale γ, the prelude block first embeds input tokens x as γE(x), and then to applies l P many prelude layers with the layout described above. Our core recurrent block R starts with an adapter matrix A : R 2h → R h mapping the concatenation of s i and e into the hidden dimension h (Bansal et al., 2022). While re-incorporation of initial embedding features via addition rather than concatenation works equally well for smaller models, we find that concatenation works best at scale. This is then fed into l R transformer layers. At the end of the core block the output is again rescaled with an RMSNorm n c . The coda contains l C layers, normalization by n c , and projection into the vocabulary using tied embeddings E T .
In summary, we can summarize the architecture by the triplet (l P , l R , l C ), describing the number of layers in each stage, and by the number of recurrences r, which may vary in each forward pass. We train a number of small-scale models with shape (1, 4, 1) and hidden size h = 1024, in addition to a large model with shape (2, 4, 2) and h = 5280. This model has only 8 "real" layers, but when the recurrent block is iterated, e.g. 32 times, it unfolds to an effective depth of 2 + 4r + 2 = 132 layers, constructing computation chains that can be deeper than even the largest fixed-depth transformers (Levine et al., 2021;Merrill et al., 2022;Saunshi et al., 2024).
this section cite: ['b138', 'b131', 'b173', 'b43', 'b147', 'b13', 'b101', 'b121']

Section: Training Objective

this section cite: []

Section: Training Recurrent Models through Unrolling.
To ensure that the model can function when we scale up recurrent iterations at test-time, we randomly sample iteration counts during training, assigning a random number of iterations r to every input sequence (Schwarzschild et al., 2021b). We optimize the expectation of the loss function L over random samples x from distribution X and random iteration counts r from distribution Λ.
L(θ) = E x∈X E r∼Λ L (m θ (x, r), x ′ ) .
Here, m represents the model output, and x ′ is the sequence x shifted left, i.e., the next tokens in the sequence x. We choose Λ to be a log-normal Poisson distribution. Given a targeted mean recurrence r + 1 and a variance that we set to σ = 1 2 , we can sample from this distribution via
τ ∼ N (log(r) - 1 2 σ 2 , σ) (1) r ∼ P(e τ ) + 1,(2)
given the normal distribution N and Poisson distribution P, see Figure 2. The distribution most often samples values less than r, but it contains a heavy tail of occasional events in which significantly more iterations are taken.
Truncated Backpropagation. To keep computation and memory low at train time, we backpropagate through only the last k iterations of the recurrent unit. This enables us to train with the heavy-tailed Poisson distribution Λ, as maximum activation memory and backward compute is now independent of r. We fix k = 8 in our main experiments. At small scale, this works as well as sampling k uniformly, but it equalizes the overall memory usage in each step of training. Note that the prelude block still receives gradient updates in every step, as its output e is injected in every step. This setup resembles truncated backpropagation through time, as commonly done with RNNs, although our setup is recurrent in depth rather than time (Williams and Peng, 1990;Mikolov et al., 2011). Truncated backpropagation can also be understood as approximation of objectives based on differentiating a fixed point of the recurrence, as discussed in Geng et al. (2021).
this section cite: ['b157', 'b103', 'b54']

Section: How to Train a Large-Scale Recurrent-Depth Model In the Wild
After verifying that we can reliably train small test models up to 10B tokens, we move on to largerscale runs. Given our limited compute budget, we could either train multiple tiny models too small to show emergent effects or scaling, or train a single medium-scale model. Based on this, we prepared a single large-scale run. We train a 3.5B parameter variant of the proposed architecture on a mix of generic text, code and scientific data, with data-parallel training with a batch size of 16 million tokens. We provide comprehensive information on all training details in Appendix C.
this section cite: []

Section: Benchmark Results
We ultimately train the final model for 800B tokens, and a non-recurrent baseline comparison for 180B tokens. We evaluate these checkpoints against other open-source models trained on fully public datasets (like ours) of a similar size. We compare against Amber (Liu et al., 2023c), Pythia (Biderman et al., 2023) and a number of OLMo 1&2 variants (Groeneveld et al., 2024;AI2, 2024;Team OLMo et al., 2025). We execute all standard benchmarks through the lm-eval harness (Biderman et al., 2024) and code benchmarks via bigcode-bench (Zhuo et al., 2024).
this section cite: ['b8', 'b61']

Section: Standard Benchmarks
Overall, it is not straightforward to place our model in direct comparison to other large language models, all of which are small variations of the standard fixed-depth transformer architecture.
While our model has only 3.5B parameters and hence requires only modest interconnect bandwidth during pretraining, it consumes FLOPs (but not memory bandwidth) close to what a 32B parameter transformer would consume during pretraining, and can continuously improve in performance with test-time scaling up to FLOP budgets equivalent to what would be a standard 50B parameter fixed-depth transformer. Finally, the model is trained on only 800B tokens, while large in comparison to older fully open-source models such as the Pythia series, is small in comparison to modern open-source efforts such as OLMo, and tiny in comparison to the datasets used to train industrial open-weight models.
We collect results for established benchmark tasks (Team OLMo et al., 2025) in Table 1 and show all open-source models side-by-side. In direct comparison we see that our model outperforms the older Pythia series and is roughly comparable to the first OLMo generation, OLMo-7B in most metrics, but lags behind the later OLMo models trained larger, more carefully curated datasets. For the first recurrent-depth model for language to be trained at this scale, and considering the limitations of the training run, we find these results promising and certainly suggestive that further research into latent recurrence as an approach to test-time scaling is warranted.
this section cite: []

Section: Math and Coding Benchmarks
We also evaluate the model on math and coding. For math, we evaluate GSM8k (Cobbe et al., 2021) (as 5-shot and in the 8-way CoT setup), MATH ((Hendrycks et al., 2021b) with the Minerva evaluation Table 1: Results on zero-shot evaluations across open-source models. We show ARC (Clark et al., 2018), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021a), OpenBookQA (Mihaylov et al., 2018), PiQA (Bisk et al., 2020), SciQ (Johannes Welbl, 2017), and WinoGrande (Sakaguchi et al., 2021). We report normalized accuracy on PiQA, OBQA, ARC-C and HellaSwag.
Model Param Tokens ARC-E ARC-C HellaSwag MMLU OBQA PiQA SciQ WinoGr random 25.0 25.0 25.0 25.0 25.0 50.0 25.0 50.0 Amber 7B 1.2T 65.70 37.20 72.54 26.77 41.00 78.73 88.50 63.22 Pythia-2.8b 2.8B 0.3T 58.00 32.51 59.17 25.05 35.40 73.29 83.60 57.85 Pythia-6.9b 6.9B 0.3T 60.48 34.64 63.32 25.74 37.20 75.79 82.90 61.40 Pythia-12b 12B 0.3T 63.22 34.64 66.72 24.01 35.40 75.84 84.40 63.06 OLMo-1B 1B 3T 57.28 30.72 63.00 24.33 36.40 75.24 78.70 59.19 OLMo-7B 7B 2.5T 68.81 40.27 75.52 28.39 42.20 80.03 88.50 67.09 OLMo-7B-0424 7B 2.05T 75.13 45.05 77.24 47.46 41.60 80.09 96.00 68.19 OLMo-7B-0724 7B 2.75T 74.28 43.43 77.76 50.18 41.60 80.69 95.70 67.17 OLMo-2-1124 7B 4T 82.79 57.42 80.50 60.56 46.20 81.18 96.40 74.74 OLMo-2-32B-0.8T 32B 0.8T 79.46 54.69 79.85 59.49 48.00 81.07 92.70 75.69 Ours, (r = 4) 3.5B 0.8T 49.07 27.99 43.46 23.39 28.20 64.96 80.00 55.24 Ours, (r = 8) 3.5B 0.8T 65.11 35.15 58.54 25.29 35.40 73.45 92.10 55.64 Ours, (r = 16) 3.5B 0.8T 69.49 37.71 64.67 31.25 37.60 75.79 93.90 57.77 Ours, (r = 32) 3.5B 0.8T 69.91 38.23 65.21 31.38 38.80 76.22 93.50 59.43  Table 4: Baseline comparison, comparing the recurrent model with a non-recurrent (fixed-depth) model with the same parameter count, trained in the same training setup and data.
Comparing the recurrent model with its non-recurrent baseline, we see that even at 180B tokens, the recurrent substantially outperforms on harder tasks. Model Tokens ARC-E ARC-C HellaSwag MMLU OBQA PiQA SciQ WinoGr GSM8K CoT Non-Recurrent Baseline 0.18T 46.42 26.96 37.34 24.16 29.60 64.47 73.20 51.78 1.82/2.20 Ours, early ckpt, (r = 32) 0.18T 53.62 29.18 48.80 25.59 31.40 68.88 80.60 52.88 9.02/10.24 Ours, early ckpt, (r = 1) 0.18T 34.01 23.72 29.19 23.47 25.60 53.26 54.10 53.75 0.00/0.15 Ours, (r = 32) 0.8T 69.91 38.23 65.21 31.38 38.80 76.22 93.50 59.43 34.80/42.08 Ours, (r = 1) 0.8T 34.89 24.06 29.34 23.60 26.80 55.33 47.10 49.41 0.00/0.00
rules (Lewkowycz et al., 2022)) and MathQA (Amini et al., 2019). For coding, we check MBPP (Austin et al., 2021) and HumanEval (Chen et al., 2021). Here we find that our model significantly surpasses all models except the latest OLMo-2 model in mathematical reasoning, as measured on GSM8k and MATH. On coding benchmarks the model beats all other general-purpose open-source models, although it does not outperform dedicated code models, such as StarCoder2 (Lozhkov et al., 2024), trained for several trillion tokens. We also note that while further improvements in language modeling are slowing down, as expected at this training scale, both code and mathematical reasoning continue to improve steadily throughout training, see Figure 3.
this section cite: ['b31', 'b30', 'b171', 'b102', 'b19', 'b118', 'b86', 'b4', 'b25', 'b16']

Section: Where does recurrence help most?
How much of this performance can we attribute to recurrence, and how much to other factors, such as dataset, tokenization and architectural choices? In Table 4, we compare our recurrent model against its non-recurrent twin, which we trained to 180B tokens in the exact same setting. In direct comparison of both models at 180B tokens, we see
that the recurrent model outperforms its baseline with an especially pronounced advantage on harder tasks, such as the ARC challenge set. On other tasks, such as SciQ, which requires straightforward recall of scientific facts, performance of the models is more similar. We observe that gains through reasoning are especially prominent on GSM8k, where the 180B recurrent model is already 5 times better than the baseline at this early snapshot in the pretraining process. We also note that the recurrent model, when evaluated with only a single 100 200 300 400 500 600 700 800 Tokens Trained (Billion) 0 5 10 15 20 25 30 35 GSM8K CoT 1 Rec 4 Rec 8 Rec 16 Rec 32 Rec 64 Rec 100 200 300 400 500 600 700 800 Tokens Trained (Billion) 25 30 35 40 45 50 55 60 65 HellaSwag 1 Rec 4 Rec 8 Rec 16 Rec 32 Rec 64 Rec 100 200 300 400 500 600 700 800 Tokens Trained (Billion) 0 5 10 15 20 HumanEval 1 Rec 4 Rec 8 Rec 16 Rec 32 Rec 64 Rec Figure 3: GSM8K CoT, HellaSwag, and HumanEval performance over the training tokens with different recurrences at test-time. We evaluate GSM8K CoT with chat template and 8-way few shot as multiturn. HellaSwag and HumanEval are zero-shot with no chat template. Model performance on harder tasks grows almost linearly with the training budget, if provided sufficient test-time compute. 0 5 10 15 20 25 30 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 high school mathematics Default ( =12.7) Cont. CoT ( =11.9) 0 5 10 15 20 25 30 philosophy Default ( =14.6) Cont. CoT ( =13.5) 0 5 10 15 20 25 30 logical fallacies Default ( =15.6) Cont. CoT ( =14.4) 0 5 10 15 20 25 30 moral scenarios Default ( =16.2) Cont. CoT ( =16.0) Steps to KL-based Threshold Density
Figure 4: Histograms of zero-shot, per-token adaptive exits based on KL difference between steps for questions from MMLU categories. The mean of each distribution is given in the legends. The exit threshold is fixed to 5 × 10 -4 . We see that the model converges quicker on high school mathematics than tasks such as logical fallacies or moral scenarios. Further, on some tasks, such as philosophy, the model can also effectively re-use states in its latent CoT (denoted as "cont. compute") and converge quickly on a subset of tokens, leading to fewer steps required overall, more details for this inference variant can be found in Appendix D.6.
recurrence, effectively stops improving between the early 180B checkpoint and the 800B checkpoint on hard tasks, showing that further improvements are not built into the fixed, non-recurrent parts but encoded entirely into the iterations of the recurrent block.
this section cite: []

Section: Improvements through Weight Averaging
Due to our constant learning rate, we can materialize further improvements through weight averaging (Izmailov et al., 2018) to simulate the result of a cooldown (Hägele et al., 2024;DeepSeek-AI et al., 2024). We use an exponential moving average starting from our last checkpoint with β = 0.9, incorporating the last 75 checkpoints with a dilation factor of 7, a modification to established protocols (Kaddour, 2022;Sanyal et al., 2024). We evaluate this EMA model as well, which further improves GMS8k performance to 47.23% flexible (38.59% strict), when tested at r = 64.
this section cite: ['b69', 'b62', 'b39', 'b72', 'b120']

Section: Recurrent Depth simplifies LLMs
Aside from encouraging performance in mathematical and code reasoning, recurrent-depth models turn out to be surprisingly natural tools to support a number of methods that require substantial effort with standard transformers. In the next section, we provide a non-exhaustive overview.
Zero-Shot Adaptive Compute at Test-Time. We have shown that the model is capable of varying compute on a per-query level, running the model in different recurrence modes. This is after all also how the model is trained, as in Equation ( 1). However, it would be more efficient in practice to stop recurring early when predictions are easy, and only spend compute on hard decisions. Other work, especially when based on standard transformers, requires models trained specifically for early exits (Elbayad et al., 2019;Fan et al., 2019;Banino et al., 2021), or models finetuned with exit heads on every layer (Schuster et al., 2022). To test our model's zero-shot exit abilities, we choose a simple exit criterion to evaluate convergence, the KL-divergence between two the next token probabilities of two successive recurrence steps. If this divergence falls below 5 × 10 -4 , we stop iterating, sample the output token, and move to generate the next token. This is just one early-exit option, and we experiment with several other schemes in Appendix Table 9.
We show this zero-shot per-token adaptive compute behavior in Figure 4, where we plot the distribution of steps taken before the exit condition is hit. We do this for the first 50 questions from different MMLU categories, asked in free-form chat. Interestingly, the number of steps required to exit differs notably between categories, with the model exiting earlier on high school mathematics, but taking on average 3.5 steps more on moral scenarios. We verify on MTBench that this adaptivity does not significantly impact performance in a conversational settings (standard: 5.63, early exits: 5.56), and even on hard tasks such as GSM8k, the merged model still reaches 44.8% (at r = 32, instead of 46% when exiting early, see Table 9).
this section cite: ['b44', 'b47', 'b12', 'b124']

Section: Zero-Shot KV-cache Sharing.
A different avenue to increase efficiency is to reduce the memory footprint of the KV-cache by sharing the cache between layers (character.ai, 2024; Brandon et al., 2024). Typically, transformers must be trained from scratch with this capability. However, as discussed in the previous section, we find that we can simply share KV-caches in our model with minimal impact to performance. We set a fixed KV-cache budget for the recurrence at every token k, and at iteration i, read and write the cache entry i mod k. For example, we set a maximum KV-cache budget of 16 steps, overwriting the KV-cache of the 1st recurrence step when executing the 17th step, and so forth. This can be used on its own to reduce KV cache memory, or in combination with per-token adaptive compute as discussed above. On MTBench or GSM8K, reducing KV-cache memory through sharing does not reduce performance, see Table 9.
Zero-Shot Self-Speculative Decoding. Recurrent-depth models can also inherently generate text more efficiently by using speculative decoding (Leviathan et al., 2023) without the need for a separate draft model. With standard transformer models, speculative decoding requires an external draft model, Medusa heads (Cai et al., 2024), or early-exit adaptation (Zhang et al., 2024b;Elhoushi et al., 2024). Zhang et al. (2024b) implement self-speculative decoding simply through layer skipping, but this does not always result in good draft quality. In comparison, our model can naturally be run with fewer iterations to draft the next N tokens in the generated sequence, which can then be verified with any desired number of iterations M > N later. Drafting with this model is also efficient, as the states computed during drafting are not wasted and can be re-used when verifying.
this section cite: ['b22', 'b84', 'b24', 'b45']

Section: What Mechanisms Emerge at Scale in Recurrent-Depth Models
Finally, what is the model doing while recurring in latent space? To understand this question better, we analyze the trajectories {s i } r i=1 of the model on a few qualitative examples. We are especially interested in understanding what patterns emerge, simply by training this model at scale. In comparison to previous work, such as Bai et al. (2019), where the training objective directly encodes a prior that pushes trajectories to a fixed point, we only train with our truncated unrolling objective.
Figure 5 shows the norm distance ||s i -s * || between each s i in a trajectory and an approximate limit point s * at r = 128. We show the sentence top to bottom and iterations from left to right. We clearly see that convergence behavior depends on context. We see that key parts of the question, and the start of the model response, are "deliberated" much more in latent space. The context dependence can also be seen in the different behavior among the three identical tokens representing each of the three dots. Also note that the distance to s * does not always decrease monotonically (e.g. for school); the model may also trace out complicated orbits in its latent trajectory while processing information, even though this is not represented explicitly in our training objective.
We look at trajectories for select tokens in more detail in Figure 6. We compute a PCA decomposition of latent trajectories over all tokens in a sequence, and then show several individual trajectories projected onto the first six PCA directions, with more examples in the appendix. Many tokens simply converge to a fixed point. Yet, for harder questions, such as in the 1st rowfoot_0 , the state of the token quickly falls into an orbit pattern in all three pairs of PCA directions. The use of multi-dimensional orbits like these could serve a similar purpose to periodic patterns sometimes observed in fixed-depth transformers trained for arithmetic tasks (Nanda et al., 2022), but we find these patterns extend far beyond arithmetic for our model. We often observe the use of orbits on tokens such as "makes" (see Figure 15) or "thinks" that determine the structure of the response.
Aside from orbits, we also observe the model encoding particular key tokens as "sliders", as seen in the middle of the 2nd row in Figure 6 (which is the token "wrong", from the same message as already shown in Figure 5). In these motions the trajectory noticeably drifts in a single direction, which the model could use to implement a mechanism to count how many iterations have occurred.
The emergence of structured trajectories in latent space gives us a glimpse into how the model performs its computations. Unlike the discrete sequential chain of reasoning seen in verbalized chain-of-thought approaches, we observe rich geometric patterns including orbits, convergent paths, and drifts -means to organize its computational process spatially. This suggests the model is independently learning to leverage the high-dimensional nature of its latent space to implement reasoning in new ways.
this section cite: ['b9', 'b109']

Section: Related Work Overview
The extent to which recurrence is a foundational concept of machine learning is hard to overstate (Amari, 1972;Hopfield, 1982;Braitenberg, 1986;Gers and Schmidhuber, 2000;Sutskever et al., 2008). Aside from using recurrence to move along sequences, as in recurrent neural networks, it was understood early to also be the key to adaptive computation (Schmidhuber, 2012;Graves, 2017). For transformers, recurrence was applied in Dehghani et al. (2019), who highlight the aim of recurrent depth to model universal, i.e. Turing-complete, machines (Graves et al., 2014). It was used at scale (but with fixed recurrence) in Lan et al. (2019) and an interesting recent improvement in this line of work are described in Tan et al. (2023 2024show that depth recurrence is advantageous when learning generalizable algorithms when training with randomized unrolling and input injections. Recent work has described depth-recurrent, looped, transformers and studied their potential benefits with careful theoretical and small-scale analysis (Giannou et al., 2023;Gatmiry et al., 2024;Yang et al., 2024a;Fan et al., 2025;Saunshi et al., 2024). Our study reinforces these prior works, showing not only how to scale depth recurrence to billion parameter scales and token counts, but also that conjectured advantages such as algorithm learning and reasoning with extended compute do materialize for realistic benchmark tasks.
From another angle, these models can be described as neural networks learning a fixed-point iteration, as studied in deep equilibrium models (Bai et al., 2019(Bai et al., , 2022;;Schöne et al., 2025). The variant of latent recurrent depth we discuss in this work is also related to diffusion models (Song and Ermon, 2019), especially latent diffusion models (Rombach et al., 2022), but we note that language diffusion models are usually run with a per-sequence, instead of a per-token, iteration count (Lee et al., 2018).
A key difference of our approach to both equilibrium models and diffusion models is in the training objective, where equilibrium methods solve the implicit bilevel problem directly, diffusion models solve a surrogate training objective, and our work suggests that truncated unrolling is a powerful alternative at scale, see also Geng et al. (2021).
More generally, all architectures that recur in depth can also be understood as directly learning the analog to the gradient of a latent energy-based model (LeCun and Huang, 2005;LeCun, 2022), to an implicitly defined intermediate optimization layer (Amos and Kolter, 2017), or to a Kuramoto layer (Miyato et al., 2024). Analogies to gradient descent at inference time also show the connection to test time adaptation (Sun et al., 2020), especially test-time adaptation of output states (Boudiaf et al., 2022).
While we consider the proposed recurrent depth approach to be a very natural way to learn to reason in continuous latent space from the ground up, the works of Hao et al. (2024); Cheng and Durme (2024) and Liu et al. (2024) discuss how to finetune existing fixed-depth transformers with this capability. These works have a similar aim to ours, enabling reasoning in latent space, but approach this goal from separate directions. For additional discussions related to the idea of constructing a prior that incentivizes reasoning and algorithm learning at the expense of memorization of simple patterns, we also refer to Chollet (2019), Schwarzschild (2023), Li et al. (2020) and Moulton (2023).
this section cite: ['b2', 'b67', 'b21', 'b55', 'b141', 'b122', 'b59', 'b40', 'b60', 'b80', 'b56', 'b53', 'b49', 'b121', 'b9', 'b10', 'b123', 'b137', 'b117', 'b83', 'b54', 'b82', 'b81', 'b5', 'b104', 'b140', 'b20', 'b63', 'b91', 'b28', 'b125', 'b105']

Section: Limitations and Conclusions
While the experiments in this paper demonstrate the viability of (latent) recurrent-depth architectures for language modeling at scale, the models described are ultimately still a proof-of-concept. We observe that we can train models that improve with increased test-time scaling via recurrence, improving over a fixed-depth model with the same parameter count by 5x on GSM8K. We observe that in our training recipe, performance saturation depends on task complexity, but is always sigmoidal in the number of iterations. Nevertheless, future work with additional compute is still required to allow for precise comparisons to the other forms of scaling, such as training fixed-depth transformers with the same FLOP count in pretraining, or training verbal CoT models targeting the same FLOP count at test time.
Yet, the interesting behaviors already observable in this work, such as the context-dependent problemsolving speed, with the model learning to solve easy problems with fewer recurrences than harder problems, various zero-shot abilities and emergence of structured thinking in latent space, lead us to believe that latent reasoning is a promising research direction to complement existing approaches for test-time compute scaling. Our work validates the motivations and observations of prior work developed at smaller scales for universal, looped and deep thinking transformers, and we are excited about the potential impact of imbuing generative models with the ability to reason in continuous latent space without the need for specialized data at train time or verbalization at inference time.
this section cite: []

Section: References
Ref_id:b0 Title: Adaptivity and Modularity for Efficient Generalization Over Task Complexity Year: (2023)
Ref_id:b1 Title: Physics of language models: Part 3.1, knowledge storage and extraction Year: (2024)
Ref_id:b2 Title: Learning Patterns and Pattern Sequences by Self-Organizing Nets of Threshold Elements Year: (1972)
Ref_id:b3 Title: AMD Instinct™ MI250X Accelerators Year: (2021)
Ref_id:b4 Title: Mathqa: Towards interpretable math word problem solving with operation-based formalisms Year: (2019)
Ref_id:b5 Title: OptNet: Differentiable Optimization as a Layer in Neural Networks Year: (2017)
Ref_id:b6 Title: Path Independent Equilibrium Models Can Better Exploit Test-Time Computation Year: (2022)
Ref_id:b7 Title: Quoc Le, and 1 others. 2021. Program synthesis with large language models Year: ()
Ref_id:b8 Title: The Twelfth International Conference on Learning Representations Year: (2023)
Ref_id:b9 Title: Deep Equilibrium Models Year: (2019)
Ref_id:b10 Title: Neural Deep Equilibrium Solvers Year: (2022)
Ref_id:b11 Title: LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs Year: (2024)
Ref_id:b12 Title: PonderNet: Learning to Ponder Year: (2021)
Ref_id:b13 Title: End-to-end Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking Year: (2022)
Ref_id:b14 Title: Rethinking Deep Thinking: Stable Learning of Algorithms using Lipschitz Constraints Year: (2024)
Ref_id:b15 Title: Machine Learning Engineering Open Book Year: (2023)
Ref_id:b16 Title:  Year: (2024)
Ref_id:b17 Title: Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. 2023. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling Year: ()
Ref_id:b18 Title:  Year: ()
Ref_id:b19 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b20 Title: Parameter-Free Online Test-Time Adaptation Year: (2022)
Ref_id:b21 Title: Vehicles: Experiments in Synthetic Psychology Year: (1986)
Ref_id:b22 Title: Reducing Transformer Key-Value Cache Size with Cross-Layer Attention Year: (2024)
Ref_id:b23 Title: Digitised Books. c. 1510 -c. 1900. JSONL (OCR Derived Text + Metadata) Year: (2021)
Ref_id:b24 Title: Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads Year: (2024)
Ref_id:b25 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b26 Title: Compressed Chain of Thought: Efficient Reasoning Through Dense Representations Year: (2024)
Ref_id:b27 Title:  Year: (2023)
Ref_id:b28 Title: On the Measure of Intelligence Year: (2019)
Ref_id:b29 Title: Vinodkumar Prabhakaran, and 48 others Year: (2022)
Ref_id:b30 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b31 Title: Training Verifiers to Solve Math Word Problems Year: (2021)
Ref_id:b32 Title: Open-phi/textbooks • Datasets at Hugging Face Year: (2024)
Ref_id:b33 Title: MoEUT: Mixture-of-Experts Universal Transformers Year: (2024)
Ref_id:b34 Title:  Year: (2024)
Ref_id:b35 Title: Getting the most out of your tokenizer for pre-training and domain adaptation Year: (2024)
Ref_id:b36 Title: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning Year: (2023)
Ref_id:b37 Title: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness Year: (2022)
Ref_id:b38 Title: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning Year: (2025)
Ref_id:b39 Title:  Year: (2024)
Ref_id:b40 Title:  Year: (2019)
Ref_id:b41 Title: From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step Year: (2024)
Ref_id:b42 Title: Fewer Truncations Improve Language Modeling Year: (2024)
Ref_id:b43 Title: CogView: Mastering Text-to-Image Generation via Transformers Year: (2021)
Ref_id:b44 Title: Depth-Adaptive Transformer Year: (2019)
Ref_id:b45 Title: LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding Year: (2024)
Ref_id:b46 Title: Scaling Exponents Across Parameterizations and Optimizers Year: (2024)
Ref_id:b47 Title: Reducing Transformer Depth on Demand with Structured Dropout Year: (2019)
Ref_id:b48 Title: Addressing Some Limitations of Transformers with Feedback Memory Year: (2021)
Ref_id:b49 Title: Looped Transformers for Length Generalization Year: (2025)
Ref_id:b50 Title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity Year: (2022)
Ref_id:b51 Title: ChessGPT: Bridging Policy Learning and Language Modeling Year: (2023)
Ref_id:b52 Title: Locutusque/hercules-v5.0 • Datasets at Hugging Face Year: (2024)
Ref_id:b53 Title: Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning? Jonas Geiping and Tom Goldstein Year: (2023)
Ref_id:b54 Title: On Training Implicit Models Year: (2021)
Ref_id:b55 Title: Recurrent nets that time and count Year: (2000)
Ref_id:b56 Title: Looped Transformers as Programmable Computers Year: (2023)
Ref_id:b57 Title: MastermindEval: A Simple But Scalable Reasoning Benchmark Year: (2025)
Ref_id:b58 Title: Yangqing Jia, and Kaiming He Year: (2018)
Ref_id:b59 Title: Adaptive Computation Time for Recurrent Neural Networks Year: (2017)
Ref_id:b60 Title: Neural Turing Machines Year: (2014)
Ref_id:b61 Title: Yuling Gu, Jack Hessel, and 24 others Year: (2024)
Ref_id:b62 Title: Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations Year: (2024)
Ref_id:b63 Title: Training Large Language Models to Reason in a Continuous Latent Space Year: (2024)
Ref_id:b64 Title: Towards Large Language Models with Training-Free Consolidated Associative Memory Year: (2024)
Ref_id:b65 Title: 2021a. Measuring massive multitask language understanding Year: ()
Ref_id:b66 Title: Dawn Song, and Jacob Steinhardt. 2021b. Measuring Massive Multitask Language Understanding Year: ()
Ref_id:b67 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b68 Title: miniCTX: Neural Theorem Proving with Year: (2024)
Ref_id:b69 Title: Averaging weights leads to wider optima and better generalization: 34th Conference on Uncertainty in Artificial Intelligence 2018, UAI 2018. 34th Conference on Uncertainty in Artificial Intelligence Year: (2018)
Ref_id:b70 Title: Multilingual Mathematical Autoformalization Year: (2023)
Ref_id:b71 Title: Crowdsourcing multiple choice science questions Year: (2017)
Ref_id:b72 Title: Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging Year: (2022)
Ref_id:b73 Title: Scaling Laws for Neural Language Models Year: (2020)
Ref_id:b74 Title: Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention Year: (2020)
Ref_id:b75 Title:  Year: (2024)
Ref_id:b76 Title: Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models Year: (2024)
Ref_id:b77 Title: Adam: A Method for Stochastic Optimization Year: (2015)
Ref_id:b78 Title: Caiming Xiong, and Dragomir Radev. 2022. Book-Sum: A Collection of Datasets for Long-form Narrative Summarization Year: ()
Ref_id:b79 Title: Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs Year: (2024)
Ref_id:b80 Title: AL-BERT: A Lite BERT for Self-supervised Learning of Language Representations Year: (2019)
Ref_id:b81 Title: A Path Towards Autonomous Machine Intelligence. Preprint, Version 0.9 Year: (2022)
Ref_id:b82 Title: Loss functions for discriminative training of energy-based models Year: (2005)
Ref_id:b83 Title: Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement Year: (2018)
Ref_id:b84 Title: Fast Inference from Transformers via Speculative Decoding Year: (2023)
Ref_id:b85 Title: Or Sharir, Hofit Bata, and Amnon Shashua. 2021. The Depth-to-Width Interplay in Self-Attention Year: ()
Ref_id:b86 Title: Solving quantitative reasoning problems with language models Year: (2022)
Ref_id:b87 Title: StarCoder: May the source be with you! Transactions on Machine Learning Research Year: (2023)
Ref_id:b88 Title: Pushmeet Kohli, and Oriol Vinyals. 2020. Strong Generalization and Efficiency in Neural Programs Year: ()
Ref_id:b89 Title: A top-quality LLM pre-training dataset requires the perfect blend Year: (2024)
Ref_id:b90 Title: 2023a. Ring attention with blockwise transformers for near-infinite context Year: ()
Ref_id:b91 Title: Deliberation in Latent Space via Differentiable Cache Augmentation Year: (2024)
Ref_id:b92 Title: WebGLM: Towards An Efficient Web-Enhanced Question Answering System with Human Preferences Year: (2023)
Ref_id:b93 Title: Liping Tang, Nikhil Ranjan, and 9 others. 2023c. LLM360: Towards fully transparent open-source LLMs Year: ()
Ref_id:b94 Title:  Year: (2017)
Ref_id:b95 Title:  Year: ()
Ref_id:b96 Title: MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code Year: (2024)
Ref_id:b97 Title:  Year: (2024)
Ref_id:b98 Title: Avi Schwarzschild, and Petar Veličković. 2024. The CLRS-Text Algorithmic Reasoning Language Benchmark Year: ()
Ref_id:b99 Title: MIND over Body: Adaptive Thinking using Dynamic Computation Year: (2024)
Ref_id:b100 Title: Transformers Can Do Arithmetic with the Right Embeddings Year: (2024)
Ref_id:b101 Title: Saturated Transformers are Constant-Depth Threshold Circuits Year: (2022)
Ref_id:b102 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b103 Title: Extensions of recurrent neural network language model Year: (2011)
Ref_id:b104 Title: Artificial Kuramoto Oscillatory Neurons Year: (2024)
Ref_id:b105 Title: The Many Ways that Digital Minds Can Know Year: (2023)
Ref_id:b106 Title: Xiangru Tang, Leandro von Werra, and Shayne Longpre. 2024. OctoPack: Instruction Tuning Code Large Language Models Year: ()
Ref_id:b107 Title: Tiny-textbooks Year: (2023)
Ref_id:b108 Title: Tiny-strange-textbooks Year: (2024)
Ref_id:b109 Title: Progress measures for grokking via mechanistic interpretability Year: (2022)
Ref_id:b110 Title: Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse Year: (2022)
Ref_id:b111 Title: New reasoning models: Openai o1-preview and o1-mini Year: (2024)
Ref_id:b112 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b113 Title: OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text Year: (2023)
Ref_id:b114 Title: Language Models are Unsupervised Multitask Learners Year: (2019)
Ref_id:b115 Title: Compressive Transformers for Long-Range Sequence Modelling Year: (2019)
Ref_id:b116 Title: ZeRO: Memory optimizations Toward Training Trillion Parameter Models Year: (2020)
Ref_id:b117 Title: High-Resolution Image Synthesis With Latent Diffusion Models Year: (2022)
Ref_id:b118 Title: WinoGrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b119 Title: Nihal Nayak, Debajyoti Datta, and 21 others. 2021. Multitask Prompted Training Enables Zero-Shot Task Generalization Year: ()
Ref_id:b120 Title: Early weight averaging meets high learning rates for LLM pre-training Year: (2024)
Ref_id:b121 Title: Understanding Reasoning With Looped Model Year: (2024)
Ref_id:b122 Title: Self-Delimiting Neural Networks Year: (2012)
Ref_id:b123 Title: Implicit Language Models are RNNs: Balancing Parallelization and Expressivity Year: (2025)
Ref_id:b124 Title: Confident Adaptive Language Modeling Year: (2022)
Ref_id:b125 Title: Deep Thinking Systems: Logical Extrapolation with Recurrent Neural Networks Year: (2023)
Ref_id:b126 Title: 2021a. Datasets for Studying Generalization from Easy to Hard Examples Year: ()
Ref_id:b127 Title: 2021b. Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks Year: ()
Ref_id:b128 Title: Algorithm Design for Learned Algorithms Year: (2023)
Ref_id:b129 Title: Neural Machine Translation of Rare Words with Subword Units Year: (2016)
Ref_id:b130 Title: Deepseekmath: Pushing the limits of mathematical reasoning in open language models Year: (2024)
Ref_id:b131 Title: GLU Variants Improve Transformer Year: (2020)
Ref_id:b132 Title: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer Year: (2017)
Ref_id:b133 Title: AxoNN: An asynchronous, message-driven parallel framework for extreme-scale deep learning Year: (2022)
Ref_id:b134 Title: Democratizing AI: Open-source Scalable LLM Training on GPU-based Supercomputers Year: (2024)
Ref_id:b135 Title: SlimPajama: A 627B token cleaned and deduplicated version of RedPajama Year: (2023)
Ref_id:b136 Title: Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research Year: (2024)
Ref_id:b137 Title: Generative Modeling by Estimating Gradients of the Data Distribution Year: (2019)
Ref_id:b138 Title: RoFormer: Enhanced Transformer with Rotary Position Embedding Year: (2021)
Ref_id:b139 Title: Augmenting Self-attention with Persistent Memory Year: (2019)
Ref_id:b140 Title: Test-Time Training with Self-Supervision for Generalization under Distribution Shifts Year: (2020)
Ref_id:b141 Title: The Recurrent Temporal Restricted Boltzmann Machine Year: (2008)
Ref_id:b142 Title: Memory-Augmented Recurrent Neural Networks Can Learn Generalized Dyck Languages Year: (2019)
Ref_id:b143 Title: Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them Year: (2022)
Ref_id:b144 Title: Lessons on Parameter Sharing across Layers in Transformers Year: (2023)
Ref_id:b145 Title: Spike No More: Stabilizing the Pre-training of Large Language Models Year: (2024)
Ref_id:b146 Title: Sparse Universal Transformer Year: (2023)
Ref_id:b147 Title: Improving Open Language Models at a Practical Size Year: (2024)
Ref_id:b148 Title:  Year: ()
Ref_id:b149 Title: Llama-2-7B-32K-Instruct -and fine-tuning for Llama-2 models with Together API Year: (2023)
Ref_id:b150 Title: Accelerating AI for Math with Massive Open-Source Instruction Data Year: ()
Ref_id:b151 Title: 2024b. OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset Year: ()
Ref_id:b152 Title: Attention Is All You Need Year: (2017)
Ref_id:b153 Title: MathPile: A Billion-Token-Scale Pretraining Corpus for Math Year: (2024)
Ref_id:b154 Title: Makesh Narsimhan Sreedhar, and Oleksii Kuchaiev. 2024b. HelpSteer2: Open-source dataset for training top-performing reward models Year: ()
Ref_id:b155 Title: BLiMP: The Benchmark of Linguistic Minimal Pairs for English Year: (2020)
Ref_id:b156 Title: RedPajama: An Open Dataset for Training Large Language Models Year: (2024)
Ref_id:b157 Title: An Efficient Gradient-Based Algorithm for On-Line Training of Recurrent Network Trajectories Year: (1990)
Ref_id:b158 Title: 2023a. Stable and low-precision training for large-scale vision-language models Year: ()
Ref_id:b159 Title: 2023b. Stable and low-precision training for large-scale vision-language models Year: ()
Ref_id:b160 Title: Enhancing PyTorch Performance on Frontier with the RCCL OFI-Plugin Year: (2024)
Ref_id:b161 Title: Memorizing Transformers Year: (2022)
Ref_id:b162 Title: LEAN-GitHub: Compiling GitHub LEAN repositories for a versatile LEAN prover Year: (2024)
Ref_id:b163 Title: Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing Year: (2024)
Ref_id:b164 Title: LeanDojo: Theorem Proving with Retrieval-Augmented Language Models Year: (2023)
Ref_id:b165 Title: 2024a. Looped Transformers are Better at Learning Learning Algorithms Year: ()
Ref_id:b166 Title: 2024b. Parallelizing Linear Transformers with the Delta Rule over Sequence Length Year: ()
Ref_id:b167 Title: Lean Workbook: A large-scale Lean problem set formalized from natural language math problems Year: (2024)
Ref_id:b168 Title: MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models Year: (2023)
Ref_id:b169 Title: Revisiting BFloat16 Training Year: (2021)
Ref_id:b170 Title: Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking Year: (2024)
Ref_id:b171 Title: Hellaswag: Can a machine really finish your sentence? Year: (2019)
Ref_id:b172 Title: Scaling Vision Transformers Year: (2022)
Ref_id:b173 Title: Root Mean Square Layer Normalization Year: (2019)
Ref_id:b174 Title: Bill Lin, and 26 others. 2024a. MAP-Neo: Highly Capable and Transparent Bilingual Large Language Model Series Year: ()
Ref_id:b175 Title: Gang Chen, and Sharad Mehrotra. 2024b. Draft& Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding Year: ()
Ref_id:b176 Title: Autonomous Data Selection with Language Models for Mathematical Texts Year: (2024)
Ref_id:b177 Title: OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement Year: (2024)
Ref_id:b178 Title: Programming Every Example: Lifting Pre-training Data Quality like Experts at Scale Year: (2024)
Ref_id:b179 Title:  Year: (2024)
