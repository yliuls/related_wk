Title: ENERGY-BASED TRANSFORMERS ARE SCALABLE LEARNERS AND THINKERS
Abstract: Inference-time computation, analogous to human System 2 Thinking, has recently become popular for improving model performance. However, most existing approaches suffer from several limitations: they are modality-specific (e.g., working only in text), problem-specific (e.g., verifiable domains like math and coding), or require additional supervision/training on top of unsupervised pretraining (e.g., verifiers or verifiable rewards). In this paper, we ask the question "Is it possible to generalize these System 2 Thinking approaches, and develop models that learn to think solely from unsupervised learning?" We find the answer is yes, by learning to explicitly verify the compatibility between inputs and candidatepredictions, and then re-framing prediction problems as optimization with respect to this verifier. Specifically, we train Energy-Based Transformers (EBTs)-a new class of Energy-Based Models (EBMs)-to assign an energy value to every input and candidate-prediction, enabling predictions through energy minimization until convergence. To support this approach, we introduce several key techniques for stable and parallelizable training, which enable the emergence of strong System 2 Thinking capabilities and scalable EBMs. Across discrete and continuous modalities, we find EBTs outperform the Transformer++ approach, scaling up to 35% faster during pretraining, and improving inference-time performance by up to 29%. EBTs also surpass Diffusion Transformers on image denoising while requiring 99% fewer forward passes. Moreover, System 2 Thinking with EBTs yields larger performance gains on data that is farther out-of-distribution, and EBTs achieve better results than existing models on most downstream tasks despite achieving the same or worse pretraining performance, enabling EBTs to generalize better than existing approaches. Consequently, EBTs are a flexible and promising new approach for scaling both the learning and thinking capabilities of models.

Section: INTRODUCTION
In psychology, human thinking is often classified into two different types: System 1 (thinking fast) and System 2 (thinking slow) Evans (2011); Frankish (2010); Kahneman (2011); Kahneman et al. (2002). System 1 thinking is characterized by quick, intuitive and automatic responses, relying on previous experience to solve simple or familiar problems. Alternatively, System 2 Thinking is slow and deliberate, requiring effort to solve complex problems that go beyond automatic pattern recognition, such as in mathematics or out-of-distribution situations Goel et al. (2000); Neys (2006). Current models perform well on tasks suitable for System 1 thinking Li et al. (2025c), but continue to struggle with tasks that demand System 2 capabilities Mirzadeh et al. (2024); Yan et al. (2025).
As a result, System 2 Thinking has become a growing research focus, driving the development of foundation models such as O1 Jaech et al. (2024), R1 Guo et al. (2025), and Claude Anthropic (2025). These "reasoning models" excel on math and coding benchmarks by increasing the time models spend thinking. However, publicly available information from the open-source R1 model Guo et al. (2025), suggests that the Reinforcement Learning (RL) approach for training these models only works in domains where rule-based rewards can easily verify answers, such as math and coding. This limits applicability, and often harms performance on tasks such as writing OpenAI (2024); Su et al. (2025). Moreover, recent evidence indicates this approach may not foster new reasoning patterns, restricting performance on tasks requiring exploration Yue et al. (2025). For a discussion on Related Works, see Section E.
As one of the primary goals of AI is to figure out how we can create systems that learn to think on their own on any problem type, these approaches ultimately bring about the following core research question: "Can we rely entirely on unsupervised learning to develop System 2 Thinking?" Such a capability could enable generalization of current System 2 Thinking approaches to many problems, many modalities, and avoid the reliance on human, reward, or model supervision.
We argue and our empirical results suggest that the answer to this question is yes, but that there are several limitations in existing models that prevent this general Thinking from emerging. Particularly, when comparing the qualities of human System 2 Thinking with current modeling approaches (Figure 1, Table 1), we observe several key differences, outlined below as two key Facets of System 2 Thinking:
Table 1: Architectures and Cognitive Facets. For each prediction, Feed-Forward (FF) Transformers and RNNs generallyfoot_0 have a finite amount of computation. DiTs (Diffusion Transformers) can increase inference computation by denoising longer, but lack explicit prediction verification. In contrast, EBMs support dynamic computation through flexible iteration, and give an energy scalar for prediction verification.
Arch. Dynamic Compute Allocation (Facet 1) Prediction Verification (Facet 2)
FF Trans. ✗ ✗ RNNs ✗ ✗ DiTs ✓ ✗ EBTs ✓ ✓
Facet 1: Dynamic Allocation of Computation. Humans naturally allocate varying amounts of effort to different tasks depending on difficulty, which is widely supported by psychology and neuroscience Ditterich (2006); Kahneman (2011); Rougier et al. (2005). 2 For example, a decision regarding whether to change careers generally takes more time than deciding what to eat.
Facet 2: Verification of Predictions. In addition to allocating computation, human thinking also benefits from the ability to verify predictions Alkouri (2016); Loesche et al. (2018), which can guide decisions about when to stop thinking or to select the most accurate predictions. This also supports more dynamic inference time behavior, such as early stopping when a prediction is known to be correct, or allocating more compute when a problem is difficult.
For more information on additional Facets, please refer to Section F. At each step, a new prediction is fed into the model, which gives an energy scalar for the prediction's current compatibility (unnormalized likelihood) with the context (Facet 2). Then, the gradient of this energy with respect to the prediction is calculated and used to update the prediction. This gradient descent update is done iteratively to refine the prediction until convergence of the predicted energy, which allows for dynamic use of computation (Facet 1).
To achieve the two facets described, we propose viewing thinking as optimization with respect to a learned verifier, which evaluates the compatibility (unnormalized probability) between an input and candidate prediction (Figure 2). Specifically, we train Energy-Based Models (EBMs) to learn an energy (unnormalized probability) landscape over all possible input-prediction pairs, where lower energy indicates higher compatibility (Facet 2). Thinking then corresponds to starting from an initial random prediction and refining it through optimization along the energy landscape until convergence (visualized in Figure 3). This naturally enables dynamic compute allocation (Facet 1) in the form of more challenging problems utilizing additional optimization steps.
While this thinking perspective is promising, Energy-Based Models (EBMs) have struggled with scalability Du & Mordatch (2019), with no known foundation EBMs. This stems from issues with training instability and long training times Arbel et al. (2020); Du & Mordatch (2019). To address these challenges, we introduce Energy-Based Transformers (EBTs), or Transformers specifically for EBMs. We further propose practical training improvements, theoretical insights into EBM training scalability, and novel energy landscape regularization techniques that improve System 2 Thinking.
To assess learning and thinking scalability, we compare EBTs to the Transformer++ (autoregressive) and DiT (bidirectional) across discrete and continuous modalities. EBTs show up to 35% higher scaling rates than the Transformer++ across data, batch size, parameters, FLOPs, and depth. At inference, EBTs outperform existing models on System 2 Thinking-for example, by improving language model performance by 29% more than the Transformer++, and by outperforming DiTs in image denoising with 99% fewer forward passes. We observe two key effects: (1) EBTs often outperform baselines at inference even with worse pretraining performance, demonstrating the importance of System 2 Thinking; and (2) System 2 Thinking yields greater gains on more out-ofdistribution data, paralleling human thinking. We believe the EBT implementations, along with novel techniques for EBMs to maximize the learning and thinking scalability, will advance the EBM approach by addressing key challenges in stable, parallelizable, and efficient training.
this section cite: ['b36', 'b64', 'b65', 'b40', 'b89', 'b78', 'b87', 'b146', 'b61', 'b47', 'b3', 'b124', 'b150', 'b27', 'b64', 'b2', 'b84', 'b29', 'b4', 'b29']

Section: ENERGY-BASED TRANSFORMERS (EBT) INTUITION

this section cite: []

Section: LEARNING TO VERIFY
Verifying solutions is often substantially more tractable than generating them, a distinction wellknown in complexity theory Cook (2023); Goldwasser et al. (2019); Gödel (1956). For example, in solving a maze, verifying the correctness of a given path is significantly easier than discovering such a path. This asymmetry has been recognized and utilized for several decades, notably in the field of cryptography Goldwasser et al. (2019); Lavin et al. (2024); Rivest et al. (1978). EBMs are built on this principle that verification is easier than generation: rather than learning to generate directly, as in most existing approaches, EBMs learn to generate by optimizing predictions with respect to the learned energy function (shown in Figure 3). An example energy landscape and its optimization through gradient descent, interpreted as thinking. Here, the model predicts a distribution over text tokens, progressively shifting from an initial random distribution to the target distribution. At each step, the EBM assigns an energy scalar indicating how compatible the current prediction is with the context, visualized as the landscape's height (Facet 2). This scalar's convergence allows the model to determine whether the prediction is adequate or if further thinking is necessary (Facet 1). We include a more detailed toy example in Section C.3. Adapted from Li et al. (2018).
Recent works have attempted to leverage verifiers Ma et al. (2025); Team (2023); Yao et al. (2023), but these approaches decouple the verifier and generator, resulting in adversarial dynamics Ma et al. (2025) and challenges in scalability Yao et al. (2023). For example, researchers combining tree search and LLMs required thousands or even millions of samples to achieve optimal performance Team (2023). In contrast, EBMs combine the verifier and generator into a single model, where the generator is defined implicitly by the gradient of the verifier Du & Mordatch (2019). We show that this coupling addresses scalability and adversarial issues (Figures 7b and B.4a).
An additional advantage of verifiers is generalization. Because verification is usually easier than generation Swamy et al. (2025), prediction verification on Out-Of-Distribution (OOD) data is often easier than explicit prediction generation for OOD data Du et al. (2022). This characteristic often results in better generalization of verifiers than explicit generators Du et al. (2022). This may explain why EBMs often generalize better than existing models Du et al. (2022;2024), which we further support in our experiments (Figure 7a and Table 4).
this section cite: ['b41', 'b48', 'b41', 'b71', 'b106', 'b74', 'b85', 'b128', 'b147', 'b85', 'b147', 'b128', 'b29', 'b126', 'b31', 'b31', 'b31', 'b90']

Section: LEARNING TO UNDERSTAND
This verifier-centric perspective also relates to a deeper limitation in generative models, referred to as "The Generative AI Paradox West et al. (2023)." Although current generative models achieve strong generative capabilities, they frequently lack basic discrimination skills, such as the ability to assess the plausibility or coherence of their own predictions Stojnić et al. (2023); West et al. (2023), impeding their ability to engage in reasoning, planning, and decision-making Kambhampati et al. (2024); Yan et al. (2025). In contrast, EBMs offer a potential solution to this challenge: as EBMs generate by learning a verifier (which is similar to a discriminator), they develop strong discrimination skills Wang et al. (2023). Experimental results further support this observation (Table 4).
this section cite: ['b144', 'b122', 'b144', 'b66', 'b146', 'b140']

Section: ENERGY-BASED TRANSFORMERS (EBT) APPROACH

this section cite: []

Section: ENERGY-BASED MODELS (EBM) BACKGROUND
Energy-Based Models (EBMs) assign a scalar energy value to each configuration of input variables, enabling them to model the compatibility and interactions between variables, such as between a context and candidate-prediction. For probabilistic EBMs, this defines a probability distribution using a Boltzmann distribution
p θ (x) = e -E θ (x) Z(θ)
where Z(θ) = e -E θ (x) dx is the intractable partition function involving an integral over all possible values of x. To avoid the intractability of the partition function, it is common to work with unnormalized EBMs, which dispense of the partition function in favor of representing relative unnormalized probabilities. This formulation shifts the Algorithm 1: Training
Inputs: Context x, Target y, EBM E θ (x, ŷ) Hparams: Steps M , Step Size α, Loss J(•) 1 Sample ŷ0 ∼ N (0, I); 2 for i = 0, . . . , M -1 do 3 ŷi+1 ← ŷi -α∇ ŷi E θ (x, ŷi); 4 L ← J(ŷM , y); 5 return L, update E θ ;
Algorithm 2: Inference with Verification
Inputs: Context x, EBM E θ (x, ŷ) Hparams: Steps M , Step Size α, Samples N 1 for j = 1, . . . , N do 2 Sample ŷ0,j ∼ N (0, I); 3 for i = 0, . . . , M -1 do 4 ŷi+1,j ← ŷi,j -α∇ ŷi,j E θ (x, ŷi,j); 5 return ŷ * = argmin j E θ x, ŷM,j ;
focus from addressing the partition function, to simply assigning low energy to the true data manifold and high energy elsewhere Dawid & LeCun (2024); Du & Mordatch (2019), offering benefits such as scalability to spaces where the true data manifold is thin and therefore a probabilistic EBMs would have an infinite score Dawid & LeCun (2024). In supervised or predictive self-supervised learning (e.g., classification, autoregressive modeling, masked modeling), unnormalized EBMs can be formulated as: ,ŷ) , where the goal of the EBM is to learn to predict ŷ given x. Contrastive methods increase the energy of negative samples while decreasing the energy of positive samples. Due to the curse of dimensionality Dawid & LeCun (2024), where the volume of spaces grows exponentially with their dimension, contrastive methods struggle to scale because they must increase the energy of an exponentially higher number of negative samples.
p θ (x, ŷ) ∝ e -E θ (x
An alternative is to frame EBM learning as an optimization problem Du et al. (2022); Wang et al. (2023), which avoids the curse of dimensionality by implicitly regularizing the energy landscape, enabling scalable learning. In this approach, EBMs are trained to optimize an initial prediction to the ground truth solution through gradient descent, as shown in Figure 3. This pushes the energy landscape to have a local minima surrounding the ground truth solution, thereby regularizing the energy landscape to only have low energy on the true data manifold. Intuitively, this optimizationbased training approach is similar to GANs Goodfellow et al. (2014). During the forward pass, EBMs can be seen as a GAN discriminator by giving an energy "verification"; on the backward pass they can be seen a GAN generator by optimizing predictions through energy minimization to try and fool the discriminator.
Training EBMs to perform optimization can be formalized as follows. We begin with an EBM E θ , an initial prediction ŷ0 , an input (context) for the model x, and seek to predict y. We aim to find the minimum energy (most compatible) ŷ given an x, which we search for using gradient descent:
ŷi+1 = ŷi -α∇ ŷi E θ (x, ŷi ),(1)
where α is the step size (formalized in Algorithm 1). Then, the loss can be computed using any standard objective function. Importantly, this loss is backpropagated through the entire optimization process, requiring second-order derivatives (i.e., gradients of gradients). These are computed efficiently via Hessian-vector products, which scale linearly with model size Dagréou et al. (2024), similar to standard gradient descent in feed-forward models. More details and pseudocode can be found in Section I.2.
this section cite: ['b22', 'b29', 'b22', 'b22', 'b31', 'b140', 'b42', 'b21']

Section: SCALABLE EBM THINKING
While this training approach is scalable, achieving smooth energy landscapes with a single local minimum remains challenging on real-world problems. Because y is high-dimensional, the energy landscape spans a high-dimensional space, and must remain well-shaped throughout. To address this, we found three key energy landscape regularization techniques to be helpful for learning smoother, more well-behaved energy landscapes, enabling strong thinking capabilities to emerge during training.
First, we found a replay buffer helps simulate longer optimization trajectories, enabling energy landscapes to be well defined near their minimum. Second, a variant of Langevin Dynamics Du &
(a) Scaling for data. (b) Scaling for batch size. (c) Scaling for depth.
Figure 4: Language Learning Scalability-Data, Batch Size, and Depth. A comparison between the scaling of the Transformer++ recipe Touvron et al. (2023) and EBTs across data, batch size, and depth during pretraining. On all axes, EBTs out-scale the Transformer++ recipe significantly, indicating improved data efficiency. The improved depth scaling offers promise for reasoning, where depth is crucial Ye et al. (2024). These results suggest that EBTs offer promise at large data scale.
Mordatch (2019) was found to be helpful for encouraging exploration of the energy landscape:
ŷi+1 = ŷi -α∇ ŷi E θ x, ŷi + η i , η i ∼ N (0, σ),(2)
where σ is the magnitude of the noise η. Without this noise term, exploration is often limited to paths leading directly to the energy minimum, leaving other regions poorly defined. Third, varying the paths taken towards predicting solutions, by randomizing the gradient descent step size α and number of optimization steps, significantly improved generalization. Together, these techniques improved the System 2 Thinking capabilities of models, as confirmed by ablation experiments in Table 2.
With these techniques established, we explored two main Thinking approaches. First, corresponding to dynamic computation allocation (Facet 1), we conduct experiments that involve changing the number of steps taken for optimization of a single prediction. Second, corresponding to the ability to verify predictions (Facet 2), we generate N predictions from an EBM and choose the minimum energy prediction (BoN or Self-Verification, which is formalized in Algorithm 2). This is conceptually similar to Best of N (BoN) sampling using language models Stiennon et al. (2020). However, EBMs generalize this approach to both discrete and continuous modalities and perform it on every single prediction, not just to entire sequences. We demonstrate performance improvements gained from both of these techniques in several Thinking experiments (Figures 7, 6, and B.6), which further confirm the importance of the described cognitive facets. This thinking process is formalized in Algorithm 2 and we more formally define and justify usage of the term thinking in Section C.1.
this section cite: ['b148', 'b121']

Section: ENERGY-BASED TRANSFORMERS (EBTS) ARCHITECTURE
Transformers excel across domains due to their parallelizability, stability, and scalability
Borsos et al. (2023); Oquab et al. (2023); Radford et al. (2019); Vaswani et al. (2017). In contrast, Energy-Based Models (EBMs) struggle with these aspects Du & Mordatch (2019); Du et al. (2020); Li et al. (2023), making Transformers a natural fit for scaling EBMs. Consequently, we introduce Energy-Based Transformers (EBTs), Transformer implementations designed for EBMs. We developed two variants: a GPT-style Radford et al. (2018) causal decoder-only EBT, for autoregressive modeling, and a bidirectional EBT with full sequence attention Devlin et al. (2019); more details are in Section C.6. While the bidirectional EBT implementation is straightforward, the autoregressive EBT requires care to prevent information leakage (see Section C.6).
this section cite: []

Section: EXPERIMENTATION AND RESULTS
We experiment with EBTs across both Autoregressive (AR) Radford et al. (2019) as well as bidirectional models Devlin et al. (2019) in discrete and continuous spaces. 4 In discrete spaces, we focus on the language modeling objective. In continuous spaces, we focus on vision tasks of next frame prediction (Section B.1) and image denoising. All models are pretrained from scratch, as EBT's architecture is incompatible with existing foundation models, and therefore cannot be fine-tuned. We focus   Touvron et al. (2023) and EBTs across model size (parameters), compute (FLOPs), and width (embedding dimension). EBTs have an 8.97% higher scaling rate than the Transformer++ in FLOP and parameter scaling (a and b), suggesting that EBTs offer promise as a pretraining approach. on two primary types of results. First, we examine learning scalability, investigating how quickly models can fit the pretraining data, which is standard in pretraining Gu & Dao (2023); Hoffmann et al. (2022); Kaplan et al. (2020); Touvron et al. (2023). Second, we study thinking scalability, or how model performance changes as we scale the System 2 Thinking of models (Definition C.1), measured with the Number of Function Evaluations (NFEs) Chen et al. (2018); Ma et al. (2025) (forward passes). 4.1 AUTOREGRESSIVE LANGUAGE MODELING EXPERIMENTS In this section, we detail and discuss the results for all NLP experiments using Autoregressive (AR) Language Models trained to predict the next discrete token in a text sequence Radford et al. (2019). All language models are pretrained on the RedPajamaV2 text corpus Computer (2023); Weber et al. (2024) 100B sample from HuggingFace using the GPT-NeoX tokenizer Black et al. (2022b) (as in Gu & Dao (2023)). Following existing pretraining work, we compare AR EBT with the standard Transformer++ recipe Gu & Dao (2023); Sun et al. (2024); Touvron et al. (2023).
For downstream evaluation, we used four key datasets in addition to the pretraining dataset, spanning reasoning, question answering, and syntax understanding. Ordered roughly by increasing perplexity difficulty, these include GSM8K Cobbe et al. (2021), SQuAD Rajpurkar et al. (2016), BigBench Elementary Math QA Srivastava et al. (2022), and BigBench Dyck Languages Srivastava et al. (2022). We focus on reasoning benchmarks due to their close alignment with System 2 Thinking.
We conduct scaling experiments for six different axes-including data, batch size, depth, parameters, FLOPs,foot_4 and embedding dimension. The results for the data, batch size, and depth scaling are shown in Figure 4; and the results for parameters, FLOPs, and embedding dimension are visualized in Figure 5. Across all axes, EBTs consistently have a higher scaling rate than the Transformer++ recipe, suggesting that EBTs offer promise at large scalefoot_5 . Building on the learning results, we investigate EBTs for thinking at inference time. We found that the thinking capabilities of EBT emerge with a sufficiently large data scale, and therefore, due to limited resources, we focus on conducting thinking experiments with smaller models trained on substantial amounts of data. In Table 2 we conduct ablation studies to confirm the benefits of our energy landscape regularization techniques for System 2 Thinking on Out-of-Distribution Data from the BigBench Dyck Languages benchmark Srivastava et al. (2022). We find that using all techniques yields the best System 2 Thinking performance when combining extended thinking and self-verification. Additionally, the results show that randomizing the step size is critical-removing it nearly eliminates thinking gains. In contrast, disabling Langevin Dynamics degrades combined performance but improves results without verification, offering a performance-compute tradeoff.
Having established the importance of these landscape regularization techniques, in Figure 7, we analyze the scalability of thinking with EBTs, where the results yield two main insights. First, as shown in Figure 7a, EBTs are able to improve performance by as much as 29% by increasing the amount of forward passes (thinking time), whereas the Transformer++ cannot improve performance. 7 This aligns with our claims that because traditional feed-forward Transformers cannot dynamically allocate additional computation for each prediction being made, they are unable to improve performance for each token by thinking for longer.
Figure 6: OOD Thinking Performance. As data becomes more OOD, thinking with EBTs leads to greater performance improvements-highlighting how thinking is critical for generalization to OOD data. Performance is measured on 5 datasets varying in Out-of-Distribution (OOD) magnitude shift, which is measured as the ratio of downstream perplexity to pretraining perplexity. Max Thinking combines thinking longer and self-verification.
Second, as demonstrated in Figure 7b, the thinking capabilities of EBTs scale, showing that as EBTs are trained for longer, their ability to achieve improvements from verification improves, increasing up to 12% -14% from 4% -8%. This suggests that EBTs trained at the same scale as modern foundation models, such as the 15T tokens Llama3 Grattafiori et al. (2024) was trained on (≈ 1000× the current scale), could have more substantial self-verification results.
As System 2 Thinking in humans is associated with generalization to novel scenarios, we conduct experiments directly aimed at measuring the effects of System 2 Thinking on generalization. In Figure 6, we visualize the performance of EBTs on the datasets described, which have varying levels of Outof-Distribution (OOD) shift (measured as the ratio of downstream task perplexity to pretraining perplexity). We observe a strong linear trend: as the data becomes more OOD, thinking leads to greater performance improvements. Therefore, these findings suggest that the benefits of EBTs' thinking are not uniform across all data, but scale positively with the magnitude of distributional shifts, highlighting thinking as a critical mechanism for robust generalization. These findings align with observations in psychology, where humans rely on deliberate System 2 Thinking to tackle challenging OOD tasks.
Next, we investigate the relation between OOD generalization and pretraining performance. To investigate this, we compare models with identical training setups with respect to data and parameters,
(a) OOD Thinking Performance Comparison. (b) Verification Capabilities as Scale Increases.  where EBTs have slightly worse pretraining perplexity than Transformer++ models. As shown in Table 3, despite achieving a higher pretraining perplexity, EBTs achieve lower (better) perplexity on most downstream tasks, suggesting stronger generalization, particularly to OOD data. Together, with the better learning scalability results, and knowing that improved pretraining performance usually leads to improved downstream task performance Chen et al. (2024); Isik et al. (2024), these results suggest that EBTs offer promise at scale during both pretraining and inference.
this section cite: ['b102', 'b26', 'b18', 'b103', 'b119', 'b119', 'b44', 'b16', 'b59']

Section: BIDIRECTIONAL IMAGE EXPERIMENTS
In addition to investigating autoregressive EBTs, we explore the performance of EBTs trained bidirectionally. Following Chen et al. (2020); Du et al. (2022), models are trained to denoise images with a fixed noise level. At inference, we test both the training noise level and a higher OOD level. The results are in Table 4, where we observe that EBTs perform better than DiTs at both in and out of distribution image denoising across various metrics. Following Chen et al. (2018), we plot the performance based on the number of forward passes (NFEs) in Figure B.6. These results demonstrate that EBTs perform better than DiTs while using 99% less denoising steps. Lastly, qualitative results for denoised out-of-distribution images for EBT compared to the DiT baseline are shown in Figure 8, demonstrating the improved visual quality of denoised images from EBTs.
In an effort to understand whether the representations learned from denoising captured useful visual features, we perform a linear probe evaluation on ImageNet-1k Russakovsky et al. (2015) of the models learned from denoising, following common practice in visual representation learning Oquab et al. (2023). The results are shown in Table 4, where the accuracy of EBTs is around 10× higher than that of DiTs, demonstrating that EBTs learn better image representations than DiTs.  For more experiments on video, uncertainty estimation, and other topics, please refer to Section B.
this section cite: ['b31', 'b14', 'b109', 'b91']

Section: LIMITATIONS AND CONCLUSION
Limitations. Despite demonstrating strong preliminary results, EBTs have several limitations. First, because EBTs generate predictions through an optimization process, they introduce additional hyperparameters. Second, while EBTs scale well up to 800M parameters, larger models were unexplored due to resource constraints. Third, for distributions that are highly multimodal, such as images, EBTs with the current formulation struggle to capture the many modes, hence why we often combine EBTs with autoregression. Finally, current EBTs lag behind feed-forward Transformers by a large margin in FLOP-efficiency, posing a high barrier to short-term adoption. Future researchers interested in leveraging EBTs will need to weigh the tradeoff between using more computation and improved generalization and reasoning.
Conclusion. We introduced Energy-Based Transformers (EBTs), a new approach that frames System 2 Thinking as an optimization procedure with respect to a learned verifier (an Energy-Based Model), enabling System 2 Thinking to emerge across many problems and modalities from unsupervised learning. Across discrete and continuous modalities, our results demonstrate that EBTs scale at a faster rate than the Transformer++ during pretraining across all measured axes, including data, batch size, depth, parameters, FLOPs, and width-with an up to 35% higher scaling rate. This suggests that EBTs offer promise at larger scale, even without System 2 Thinking. With System 2 Thinking, EBTs improve even further-increasing performance by up to 29% on text tasks, which we observe increases with data that is more Out-of-Distribution (OOD). Comparisons to DiTs on image denoising also reveal significantly better thinking scalability: EBTs match or exceed DiT's performance with only 1% of the forward passes. EBTs also learn substantially better representations, achieving approximately 10× higher accuracy than DiTs. Ultimately, the improved scaling of EBTs during both training and inference positions them as a promising new approach.
this section cite: []

Section: References
Ref_id:b0 Title: URL Year: ()
Ref_id:b1 Title: Faithfulness vs. plausibility: On the (un) reliability of explanations from large language models Year: (2024)
Ref_id:b2 Title: Using contents and containers to investigate problem solving strategies among toddlers Year: (2016)
Ref_id:b3 Title: Claude 3.7 sonnet and claude code Year: (2025)
Ref_id:b4 Title: Generalized energy based models Year: (2020)
Ref_id:b5 Title: Residual energy-based models for text Year: (2021)
Ref_id:b6 Title: The reversal curse: Llms trained on" a is b" fail to learn" b is a Year: (2023)
Ref_id:b7 Title: A conceptual introduction to hamiltonian monte carlo Year: (1920)
Ref_id:b8 Title: Energy-based reranking: Improving neural machine translation using energy-based models Year: (2020)
Ref_id:b9 Title: Mixture density networks Year: (1994)
Ref_id:b10 Title: Gpt-neox-20b: An open-source autoregressive language model Year: (2022)
Ref_id:b11 Title: Gpt-neox-20b: An open-source autoregressive language model Year: (2022-07)
Ref_id:b12 Title: Audiolm: a language modeling approach to audio generation Year: (2023)
Ref_id:b13 Title: Transformer flops Year: (2023)
Ref_id:b14 Title: Neural ordinary differential equations Year: (2018)
Ref_id:b15 Title: Learning to stop while learning to predict Year: (2020)
Ref_id:b16 Title: Scaling laws for predicting downstream performance in llms Year: (2024)
Ref_id:b17 Title: Diffusion policy: Visuomotor policy learning via action diffusion Year: (2023)
Ref_id:b18 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b19 Title: Redpajama: an open dataset for training large language models Year: (2023)
Ref_id:b20 Title: The complexity of theorem-proving procedures Year: (2023)
Ref_id:b21 Title: How to compute hessian-vector products? Year: (2024)
Ref_id:b22 Title: Introduction to latent variable energy-based models: a path toward autonomous machine intelligence Year: (2024)
Ref_id:b23 Title:  Year: (2018)
Ref_id:b24 Title: Causal diffusion transformers for generative modeling Year: (2024)
Ref_id:b25 Title: Autoregressive video generation without vector quantization Year: (2024)
Ref_id:b26 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b27 Title: Evidence for time-variant decision making Year: (2006)
Ref_id:b28 Title: Recurrent neuronal circuits in the neocortex Year: (2007)
Ref_id:b29 Title: Implicit generation and modeling with energy based models Year: (1920)
Ref_id:b30 Title: Improved contrastive divergence training of energy based models Year: (2020)
Ref_id:b31 Title: Learning iterative reasoning through energy minimization Year: (2022)
Ref_id:b32 Title: Jascha Sohl-Dickstein, Arnaud Doucet, and Will Sussman Grathwohl. Reduce, reuse, recycle: Compositional generation with energy-based diffusion models and mcmc Year: (2023)
Ref_id:b33 Title: Learning iterative reasoning through energy diffusion Year: (2024)
Ref_id:b34 Title: Dual-process theories of reasoning: Contemporary issues and developmental applications Year: (2011)
Ref_id:b35 Title:  Year: (2019)
Ref_id:b36 Title: Dual-process and dual-system theories of reasoning Year: (2010)
Ref_id:b37 Title: Mapping sentence form onto meaning: The syntax-semantic interface Year: (2007)
Ref_id:b38 Title: Scaling up test-time compute with latent reasoning: A recurrent depth approach Year: (2025-02-23)
Ref_id:b39 Title: Understanding the difficulty of training deep feedforward neural networks Year: (2010)
Ref_id:b40 Title: Dissociation of mechanisms underlying syllogistic reasoning Year: (2000)
Ref_id:b41 Title: The knowledge complexity of interactive proof-systems Year: (2019)
Ref_id:b42 Title: Generative adversarial nets Year: (2014)
Ref_id:b43 Title: The" something something" video database for learning and evaluating visual common sense Year: (2017)
Ref_id:b44 Title: The llama 3 herd of models Year: (2024)
Ref_id:b45 Title: Mamba: Linear-time sequence modeling with selective state spaces Year: (1920)
Ref_id:b46 Title: Long-context autoregressive video modeling with next-frame prediction Year: ()
Ref_id:b47 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b48 Title: Letter to john von neumann Year: (1956)
Ref_id:b49 Title: Training large language models to reason in a continuous latent space Year: (2024)
Ref_id:b50 Title: Out-of-distribution detection with a single unconditional diffusion model Year: (2024)
Ref_id:b51 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b52 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b53 Title: Energy transformer Year: (2024)
Ref_id:b54 Title: Neural networks and physical systems with emergent collective computational abilities Year: (1982)
Ref_id:b55 Title: Predicting emergent abilities with infinite resolution evaluation Year: (2023)
Ref_id:b56 Title: T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation Year: (2023)
Ref_id:b57 Title: Diffusion models for video prediction and infilling Year: (2022)
Ref_id:b58 Title: Time matters: Scaling laws for any budget Year: ()
Ref_id:b59 Title: Scaling laws for downstream task performance of large language models Year: (2024)
Ref_id:b60 Title: Eqa-mx: Embodied question answering using multimodal expression Year: (2023)
Ref_id:b61 Title: Openai o1 system card Year: (2024)
Ref_id:b62 Title: Planning with diffusion for flexible behavior synthesis Year: (2022)
Ref_id:b63 Title: Less is more: Recursive reasoning with tiny networks Year: (2025)
Ref_id:b64 Title: Thinking, fast and slow. macmillan Year: (2011)
Ref_id:b65 Title: Representativeness revisited: Attribute substitution in intuitive judgment. Heuristics and biases: The psychology of intuitive judgment Year: (2002)
Ref_id:b66 Title: Position: Llms can't plan, but can help planning in llm-modulo frameworks Year: (2024)
Ref_id:b67 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b68 Title: Auto-encoding variational bayes Year: (2013)
Ref_id:b69 Title: When can transformers compositionally generalize in-context? arXiv preprint Year: (2024)
Ref_id:b70 Title: Transformers in speech processing: A survey Year: (2023)
Ref_id:b71 Title: A survey on the applications of zero-knowledge proofs Year: (2024)
Ref_id:b72 Title: A path towards autonomous machine intelligence version 0.9 Year: (2022)
Ref_id:b73 Title: A tutorial on energy-based learning Year: (2006)
Ref_id:b74 Title: Visualizing the loss landscape of neural nets Year: (2018)
Ref_id:b75 Title: (mis) fitting: A survey of scaling laws Year: (2025)
Ref_id:b76 Title: Autoregressive image generation without vector quantization Year: (2006)
Ref_id:b77 Title: Learning energy-based models in high-dimensional spaces with multiscale denoising-score matching Year: (2023)
Ref_id:b78 Title: From system 1 to system 2: A survey of reasoning large language models Year: (2025-01)
Ref_id:b79 Title: Let's verify step by step Year: (2023)
Ref_id:b80 Title: Implicit reasoning in transformers is reasoning through shortcuts Year: (2025)
Ref_id:b81 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b82 Title: Video-t1: Test-time scaling for video generation Year: (2025)
Ref_id:b83 Title: Compositional visual generation with composable diffusion models Year: (2022)
Ref_id:b84 Title: Paving the way to eureka-introducing "dira" as an experimental paradigm to observe the process of creative problem solving Year: (2018)
Ref_id:b85 Title: Inference-time scaling for diffusion models beyond scaling denoising steps Year: (2025)
Ref_id:b86 Title: Adaptive inference-time compute: Llms can predict if they can do better, even mid-generation Year: (2024)
Ref_id:b87 Title: Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models Year: (2024)
Ref_id:b88 Title: Do deep generative models know what they don Year: (2018)
Ref_id:b89 Title: Dual processing in reasoning: Two systems but one reasoner Year: (2006)
Ref_id:b90 Title: Learning to reason with llms Year: (2024)
Ref_id:b91 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b92 Title: Recurrent relational networks Year: (2018)
Ref_id:b93 Title: Active inference: the free energy principle in mind, brain, and behavior Year: (2022)
Ref_id:b94 Title: Pytorch: An imperative style, high-performance deep learning library Year: (2019)
Ref_id:b95 Title: Fast exact multiplication by the hessian Year: (1994)
Ref_id:b96 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b97 Title: Transformer uncertainty estimation with hierarchical stochastic attention Year: (2022)
Ref_id:b98 Title: The fineweb datasets: Decanting the web for the finest text data at scale Year: (2024)
Ref_id:b99 Title: Reinventing rnns for the transformer era Year: (2023)
Ref_id:b100 Title: Uncertainty and stress: Why it causes diseases and how it is mastered by the brain Year: (2017)
Ref_id:b101 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b102 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b103 Title: Squad: 100,000+ questions for machine comprehension of text Year: (2016)
Ref_id:b104 Title: Latent video transformer Year: (2020)
Ref_id:b105 Title: Message-passing for graph-structured linear programs: Proximal methods and rounding schemes Year: (2010)
Ref_id:b106 Title: A method for obtaining digital signatures and public-key cryptosystems Year: (1978)
Ref_id:b107 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b108 Title: Prefrontal cortex and flexible cognitive control: Rules without symbols Year: (2005)
Ref_id:b109 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b110 Title: Transformer with uncertainty estimation Year: (2022)
Ref_id:b111 Title: Uncertainty during anticipation modulates neural responses to aversion in human insula and amygdala Year: (2010)
Ref_id:b112 Title: Reasoning with latent thoughts: On the power of looped transformers Year: (2025)
Ref_id:b113 Title: Are emergent abilities of large language models a mirage? Year: (2023)
Ref_id:b114 Title: Input complexity and out-of-distribution detection with likelihood-based generative models Year: (2019)
Ref_id:b115 Title: Glu variants improve transformer Year: (2020)
Ref_id:b116 Title: A general framework for inference-time scaling and steering of diffusion models Year: (2025)
Ref_id:b117 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b118 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b119 Title: Beyond the imitation game: Quantifying and extrapolating the capabilities of language models Year: (2022)
Ref_id:b120 Title:  Year: ()
Ref_id:b121 Title: Learning to summarize with human feedback Year: (2020)
Ref_id:b122 Title: Commonsense psychology in human infants and machines Year: (2023)
Ref_id:b123 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b124 Title: Expanding rl with verifiable rewards across diverse domains Year: (2025)
Ref_id:b125 Title: Learning to (learn at test time): Rnns with expressive hidden states Year: (2024)
Ref_id:b126 Title: All roads lead to likelihood: The value of reinforcement learning in fine-tuning Year: (2025)
Ref_id:b127 Title:  Year: ()
Ref_id:b128 Title: Alphacode 2 technical report Year: (2004)
Ref_id:b129 Title: Daniel Cremers, and Mark Ibrahim. Uncertainty-based abstention in llms improves safety and reduces hallucinations Year: (2024)
Ref_id:b130 Title:  Year: ()
Ref_id:b131 Title: Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting Year: (2023)
Ref_id:b132 Title: Neural discrete representation learning Year: (2017)
Ref_id:b133 Title: Attention is all you need Year: (2017)
Ref_id:b134 Title: Differential representations of prior and likelihood uncertainty in the human brain Year: (2012)
Ref_id:b135 Title: Will we run out of data? an analysis of the limits of scaling datasets in machine learning Year: ()
Ref_id:b136 Title: A connection between score matching and denoising autoencoders Year: (2011)
Ref_id:b137 Title: Hierarchical reasoning model Year: (2025)
Ref_id:b138 Title: Satnet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver Year: (2019)
Ref_id:b139 Title: Your autoregressive generative model can be better if you treat it as an energy-based one Year: (2022)
Ref_id:b140 Title: Energy-inspired self-supervised pretraining for vision models Year: (2023)
Ref_id:b141 Title: Redpajama: an open dataset for training large language models Year: (2024)
Ref_id:b142 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b143 Title: Scaling autoregressive video models Year: (2019)
Ref_id:b144 Title: The generative ai paradox Year: (2023)
Ref_id:b145 Title: Grok 3 Beta -The Age of Reasoning Agents Year: (2025)
Ref_id:b146 Title: Do phd-level llms truly grasp elementary addition? probing rule learning vs. memorization in large language models Year: (2025)
Ref_id:b147 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b148 Title: Physics of language models: Part 2.1, gradeschool math and the hidden reasoning process Year: (2024)
Ref_id:b149 Title: Video prediction by efficient transformers Year: (2023)
Ref_id:b150 Title: Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint Year: (2025)
Ref_id:b151 Title: Dino-wm: World models on pre-trained visual features enable zero-shot planning Year: (2024)
