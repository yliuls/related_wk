Title: FP4 All the Way: Fully Quantized Training of LLMs
Abstract: We demonstrate, for the first time, fully quantized training (FQT) of large language models (LLMs) using predominantly 4-bit floating-point (FP4) precision for weights, activations, and gradients on datasets up to 1T tokens. We extensively investigate key design choices for FP4, including block sizes, scaling formats, and rounding methods. Our analysis shows that the NVFP4 format, where each block of 16 FP4 values (E2M1) shares a scale represented in E4M3, provides optimal results. We use stochastic rounding for backward and update passes and round-to-nearest for the forward pass to enhance stability. Additionally, we identify a theoretical and empirical threshold for effective quantized training: when the gradient norm falls below approximately √ 3 times the quantization noise, quantized training becomes less effective. Leveraging these insights, we successfully train a 7-billion-parameter model on 256 Intel Gaudi2 accelerators. The resulting FP4-trained model achieves downstream task performance comparable to a standard BF16 baseline, confirming that FP4 training is a practical and highly efficient approach for large-scale LLM training. A reference implementation is supplied in https://github.com/Anonymous1252022/fp4-all-the-way.

Section: Introduction
The rapid advancement of Large Language Models (LLMs) has led to unprecedented breakthroughs in natural language understanding and generation. State-of-the-art models now scale to hundreds of billions of parameters, enabling remarkable capabilities across diverse applications. However, this progress comes at a significant cost, with training and inference demanding immense computational power and memory bandwidth. As model sizes grow, hardware constraints become a major bottleneck, necessitating innovations in numerical precision and memory-efficient architectures.
Until recently, the dominant numerical format for pretraining Large Language Models (LLMs) was BF16, which provided a balance between precision and efficiency. However, as model sizes and dataset scales have grown, researchers have explored lower-precision alternatives to improve computational efficiency and reduce memory requirements. A few pioneering studies have demonstrated that full training in FP8 is not only feasible but also effective. [13] showcased the potential of FP8 training on small-scale datasets (100B), [8] extended it to trillions of tokens dataset with a 7B parameters model, and [6] demonstrated FP8's viability at an even larger scale, successfully training a massive 671B parameter Mixture-of-Experts (MoE) model on a vast dataset, achieving state-of-the-art results. FP8 is quickly emerging as the new standard for large-scale LLM training. As research continues to push precision boundaries further, the next logical step is exploring FP4, which promises even greater efficiency while maintaining training stability and model accuracy.
advancements, with recent work demonstrating competitive performance at 4-bit precision [9] and below [20].
Fully Quantized Training (FQT) is a more challenging task than PTQ or QAT, as it requires training from scratch with low-precision weights, activations, and gradients to accelerate all matrix multiplications. Until recently, applying FQT beyond 16-bit precision was considered difficult due to instability and convergence issues. However, recent works have demonstrated its feasibility. [13] presented the first FQT of a large language model in FP8 on a dataset of up to 100 billion tokens. [8] extended this to 2 trillion tokens, revealing stability issues in later training stages and proposing a modified activation function to address them. [6] further advanced the field by training a large Mixture-of-Experts (MoE) model with FP8 FQT, mitigating instability through finer-grained quantization.
Finer granularity quantization is emerging as a key direction for enabling Fully Quantized Training (FQT) beyond FP8, particularly in the context of FP4 precision. By reducing the quantization block size, these methods aim to better capture local variations in data distributions, improving stability and accuracy. Notable examples include MXFP4 [15] and NVFP4 [1].
The two works most closely related to ours are [21,19]. [21] proposes training large language models using a vector-wise FP4 format combined with two key techniques: a Differentiable Gradient Estimator (DGE) to replace the standard Straight-Through Estimator (STE), and Outlier Clamp Compensation (OCC) to handle activation outliers wih an additional sparse residual matrix. Their models are trained on up to 100 billion tokens, but they quantize only weights and activations, keeping gradients in higher precision-thus accelerating only one of the three matrix multiplications involved in training. [19], in contrast, focuses on gradient quantization using the MXFP4 format and applies stochastic rounding alongside the Hadamard transform to stabilize training. They train models up to 40 billion tokens. However, like [21], they only accelerate part of the three matrix multiplications. In contrast, our work is the first to demonstrate full FP4 Fully Quantized Training (FQT) of large-scale LLMs, enabling the acceleration of all matrix multiplications during training.
this section cite: ['b12', 'b7', 'b5', 'b8', 'b19', 'b12', 'b7', 'b5', 'b14', 'b0', 'b20', 'b18', 'b20', 'b18', 'b20']

Section: FP4 training
Going beyond FP8 to FP4 training presents significant challenges due to the limited dynamic range of FP4, making it difficult to capture the full variability of activations and gradients without excessive quantization error. However, the recently introduced microscaling floating-point family (MXFP) [15] offers a promising alternative by dynamically adjusting the scale at finer granularity, mitigating precision loss. MXFP4, includes 1 sign bit, 2 exponent bits, and 1 mantissa bit (E2M1) is a floating-point format that enhances low-precision training by dividing data into blocks of size 32, with each block sharing a common scale. The scale for each block is stored using the E8M0 format, an 8-bit exponent-only representation that provides a wide dynamic range without a mantissa and sign. Another potential format for FP4 training is NVFP4, which uses the same E2M1 data representation as MXFP4 but differs in block size and scaling format. NVFP4 divides data into smaller blocks of size 16, compared to MXFP4's 32, allowing for finer-grained scaling adjustments. Additionally, NVFP4 employs an E4M3 format for storing scales, providing a balance between dynamic range and precision. Both MXFP4 and NVFP4 are supported in NVIDIA's Blackwell architecture [1]. In Table 1 we compare these 2 formats.
this section cite: ['b14', 'b0']

Section: E8M0 E4M3 Per-Tensor Scale

this section cite: []

Section: No Yes

this section cite: []

Section: Exploring block size and scale format
Seeing the differences between the two FP4 formats supported in Blackwell-MXFP4 and NVFP4-in terms of block size and scale format, we decided to investigate their impact further. Specifically, we aim to compare the full range of possible scale formats, while maintaining the FP8 data format. Additionally, we explore different block sizes to understand their effect on numerical stability, training efficiency, and model accuracy. This analysis will provide deeper insights into the trade-offs between dynamic range, precision, and computational efficiency in low-precision training.
In Fig. 1 we train a 350M Llama-style model with FP4 format (E2M1) with block size 16 and different scaling formats. Note that, similar to NVFP4, most configurations (except for E8M0, which corresponds to MXFP4) do not utilize the sign bit in the scale. This may represent a potential inefficiency that future work could aim to exploit. We notice that the best results are achieved with E3M4 and E4M3, where the latter is used in the NVFP4 format. In Fig. 2 we compare different block sizes both with scales formats E8M0 and E4M3, which are the scales used in MXFP4 and NVFP4, respectively. Note that while selecting the appropriate scale has a significant impact on final accuracy-for example, E1M6 leads to complete divergence-the block size has a more modest effect, with smaller block sizes generally yielding better results. This leads us to proceed with the NVFP4 format, which uses a block size of 16 and a scale format of E4M3. 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Proceed tokens (Billions) 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Loss Baseline E1M6 E2M5 E3M4 E4M3 E5M2 E6M1 E8M0 Figure 1: Formats E4M3 (used in NVFP4) and E3M4 achieved the best results. Comparison of different scaling formats (E1M6, E2M5, E3M4, E4M3, E5M2, E6M1, E8M0) when training a 350M Llama model using FP4 format (E2M1) with block size 16. The formats E3M4 and E4M3 achieve the best results (recall E4M3 is used in NVFP4), whereas E1M6 results in complete divergence.
this section cite: []

Section: Exploring the rounding modes
After completing our exploration of the various block sizes and scales within the FP4 format, we now turn our attention to another critical aspect: the choice of rounding modes. In the next section, we will delve into the different rounding strategies available for FP4 and analyze their impact on numerical stability and model performance.
Fully quantized training encompasses the quantization of three key general-matrix-multiplications (GEMMs): forward, backward, and update. Each GEMM involves two quantized operands-resulting in six distinct quantization points across the training pipeline:
[Forward] z l = Q(W l )Q(a l-1 ); a l = f l (z l )(1)
[
Backward] g l-1 = Q(W T l )Q(δ l ); δ l = f ′ l (z l ) ⊙ g l (2
) [Update] ∂C ∂W l = Q(δ l )Q(a T l-1 ) ,(3)
0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Processed tokens (Billions) 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Loss Baseline Block 128 Block 64 Block 32 Block 16 Block 8 (a) 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 Processed tokens (Billions) 2.8 3.0 3.2 3.4 3.6 3.8 4.0 Loss Baseline Block 128 Block 64 Block 32 Block 16 Block 8 (b) Figure 2: Block size 16 is the best option. We examine the impact of different block sizes (8, 16, 32, 64, 128) on training accuracy using scaling formats: (a) E8M0 (used in MXFP4) and (b) E4M3 (used in NVFP4). Smaller block sizes yield modest improvements in accuracy, with diminishing returns below 16 elements per block. Thus, a block size of 16 provides an optimal compromise between performance and computational overhead. where C is the loss function, Q is a quantization operation, ⊙ is a component-wise product and, in each layer l, f l is the activation function, W l is weight matrix, z l are the pre-activations, and g l ≜ ∂C ∂a l . 6 7 8 9 10 11 12 13 14 15 Proceed tokens (Billions) 2.90 2.95 3.00 3.05 3.10 3.15 3.20 Loss Full RtN Weights Forward SR Activations Forward SR Weights Backward SR Activations Update SR Grad Update SR Grad Backward SR In each graph, we apply SR in one of the six elements in one of the GEMMs while the rest use round-to-nearest (RtN). Notice that applying SR to neural gradients during both 'Update' and 'Backward' GEMMs and activations during the 'Update' GEMM leads to lower training loss, while applying SR to other components has the opposite effect, increasing the loss.
Notably, we have the flexibility to select the rounding mode independently for each of these six elements. In Fig. 3, we present results of training a 350M Llama model using NVFP4 where stochastic rounding (SR) is applied to each of these elements separately while the rest of the elements use round-to-nearest (RtN), allowing us to evaluate its individual contribution. Applying SR to neural gradients during update and backward GEMMs and activations during the update GEMM helps reduce training loss, whereas using SR in other components leads to an increase in loss. In Fig. 7 in the Appendix, we present additional experiments that support the same conclusion. As a result, we adopt the following selective rounding scheme:
[Forward] z l = Q RtN (W l )Q RtN (a l-1 ); a l = f l (z l )(4)
[
Backward] g l-1 = Q RtN (W T l )Q SR (δ l ); δ l = f ′ l (z l ) ⊙ g l (5
) [Update] ∂C ∂W l = Q SR (δ l )Q SR (a T l-1 ) ,(6)
this section cite: []

Section: Analysis of Quantized SGD with Stochastic Rounding
This section analyzes when training with low-precision gradients stops being effective when using stochastic rounding (SR). While SR removes bias and enables stable descent during much of training, its benefits diminish once gradients become too small relative to quantization noise. We derive a threshold on the gradient-to-noise ratio that signals when further progress stalls, motivating a precision switch for backward and update passes. This switch improves convergence without altering the forward pass or the deployed model.
this section cite: []

Section: Key takeaways from the analysis:
• SR enables unbiased updates, allowing stable descent even under aggressive FP4 quantization. In contrast, deterministic rounding introduces a persistent bias, leading to an irreducible error floor and preventing convergence (see Appendix B.2).
• There exists a critical threshold: With SR, we show in Section 4.1 (and in more detail in Appendix B.1) that the average per-coordinate gradient magnitude falls approximately below √ 3 times the quantization noise standard deviation, training no longer yields effective loss reduction. This threshold is derived under simplifying assumptions-e.g., gradient descent with optimal step size, Taylor approximation of the loss, and a concentrated Hessian spectrum.
• Empirical evidence supports the theory: in Section 4.2, for both synthetic and realmodel settings, we show empirically performance degrades sharply below this threshold. A precision switch guided by the theory restores convergence.
this section cite: []

Section: Theoretical Derivation
We begin with a second-order Taylor expansion of the loss function around the current parameter vector θ t , which reveals a descent term proportional to -∇L T ∆θ and a curvature term involving the Hessian H. We then replace the full-precision gradient ∇L with its quantized version g q = ∇L + ε, where ε is zero-mean noise introduced by stochastic rounding.
Using the update rule ∆θ = -ηg q and taking expectations under SR (E[ε] = 0, E[εε T ] = σ 2 q I), we obtain the expected loss change:
E[∆L] = -η ∥∇L∥ 2 + 1 2 η 2 ∇L T H ∇L + σ 2 q tr(H) .
Balancing the descent and noise terms yields the optimal step size:
η * = ∥∇L∥ 2 ∇L T H ∇L + σ 2 q tr(H) .
Substituting η * back into the expected loss gives:
E[∆L] = - ∥∇L∥ 4 2 ∇L T H ∇L + σ 2 q tr(H) .
Differentiating with respect to σ q , and using some simplying assumptions reveals that sensitivity to quantization noise peaks when:
σ critical = ∥∇L∥ √ 3d .
Once the per-coordinate gradient magnitude drops below √ 3 σ q , the descent becomes negligible and higher-precision gradients are needed to continue improving the loss.
this section cite: []

Section: Empirical Validation
To validate this theoretical threshold, we present two types of experiments.
First, we simulate training on a simple quadratic loss with an adaptive noise schedule. We scale the quantization noise to σ q = k • σ critical for k = 2, 1, 0.5. As shown in Fig. 4, convergence completely stalls at high noise levels (e.g., k = 2), slows near the critical threshold (k = 1), and closely tracks full-precision training when noise is reduced below the threshold (k = 0.5). Second, we test in Fig. 5, the threshold on a real 60M-parameter Llama model. During training, we monitor the ratio ∥∇L∥/(σ q √ d), and switch to higher-precision gradients at the 1000th iteration. When this ratio crosses √ 3, the loss gap to the full-precision baseline closes immediately, validating the predictive power of the threshold.
this section cite: []

Section: Experiments
Setup. We used the Llama2 model [18] as our baseline. This model is a decoder-only Transformer [2] with pre-normalization RMSNorm [23], Smooth-SwiGLU activation function [8], and rotary positional embeddings [17]. We trained the models on the open-source Red Pajama dataset [5] for 1T tokens, maintaining hyperparameters consistent with [18], including train-test split and initialization. Specifically, we used AdamW optimizer with β 1 = 0.9, β 2 = 0.95. We used cosine learning rate schedule, with 2000 steps of warmup, peak learning rate of 3 × 10 -4 and decay to 0.1 of the peak learning rate. We used a global batch-size of 4M tokens. All training was conducted on 256 Intel Gaudi2 devices, during ∼ 30 days.
this section cite: ['b17', 'b1', 'b22', 'b7', 'b16', 'b4', 'b17']

Section: FP4 training.
In Fig. 6a we present our main experiment and present the training loss of Llama2 7B with the proposed FP4 scheme, which includes the use of the NVFP4 format (block size 16, scale format E4M3) and applying SR in the neural gradients (update + backward GEMMs) and activations (update GEMM), while applying RtN for the weights (forward + backward GEMMs) and activations (forward GEMM). In Table 2 we compare the quantization settings of our work with two previous FP4 training works [19,21], showing we are the first work that allows the acceleration of all matrix multiplication during training. In the Appendix Table 4 we show the similarity of the FP4 training losses at different seeds, showing the noise robustness of the proposed method.  Quantization Aware Fine-tuning (QAF). FP4 training results (Fig. 6a) in a small gap in training loss compared to the BF16 baseline. As shown in Fig. 5, increasing the precision raises the gradientto-noise ratio higher above the critical threshold. To close the remaining gap, we introduce a brief quantization-aware finetuning (QAF) phase, where pretraining continues on the same dataset with the forward pass kept in FP4, while the backward pass is executed in BF16. Keeping the forward path in FP4 ensures that the model remains fully compatible with low-precision inference, without requiring any additional processes such as post-training-quantization. During this stage, we reset the learning rate and apply a short warmup (40 iterations) followed by a cosine decay schedule with an initial peak learning rate. In Fig. 6b we present that this short QAF can completely close the gap with BF16 baseline, achieving an average bits of 4.3 bits for GEMMs across training + QAF. In the Appendix Table 5 we show an ablation study of the QAF length ratio required to get similar final loss as BF16. We find it decreases for larger datasets.
Zero-shot Performance. Table 3 compares the zero-shot performance (accuracy and perplexity) on downstream tasks between the BF16 baseline and our FP4 model -both after the FP4 training phase (1T tokens) and after QAF phase (+40B tokens). Notice that while a small gap is observed in part of the tasks after the FP4 training phase, it is completely closed after the QAF.
this section cite: ['b18', 'b20']

Section: Discussion
This work presents the first demonstration of fully quantized FP4 training-covering weights, activations, and gradients-at large scale. Our experiments on Llama2 7B model show that, while a small gap in training loss initially appears compared to BF16, this gap can be fully closed with a short QAF phase, where the forward pass remains in FP4 and only the backward pass switches to BF16. Importantly, downstream task performance remains on par with BF16, confirming FP4's practical viability.
A key contribution is our investigation of FP4 format design. We find that NVFP4 (E4M3 with block size 16) offers the best trade-off between dynamic range and precision. Other blocks size or alternative exponent/mantissa configurations lead to instability or diminishing returns, aligning with NVIDIA Blackwell's hardware decisions.
We also introduce a split rounding strategy, using stochastic rounding only in the backward pass, which substantially improves training stability. Furthermore, our theoretical analysis identifies a critical transition point: when the full-precision gradient standard deviation falls approximately below √ 3 times the quantization noise, training stagnates. This insight guides the design of the final fine-tuning phase to boost the signal-to-noise ratio and match BF16 convergence.
this section cite: []

Section: Limitations.
A key limitation of this work is the lack of dedicated FP4 support in current Gaudi hardware, which prevents us from directly measuring the potential speedup and energy efficiency benefits of native FP4 execution. As a result, all experiments are conducted using FP4 simulations in Gaudi2, which incur additional overhead from precision casting and lead to longer runtimes. Based on previous FP8 works [13,8] we expect in a rough estimation to ∼ 35 -40% time-to-train acceleration in comparison to FP8, which corresponds to ∼ 85% time-to-train acceleration in comparison to the BF16 baseline. Our work centers on the LLaMA architecture, one of the most widely adopted frameworks in modern LLMs. Preliminary experiments indicate that the approach can be directly applied to Mixture-of-Experts (MoE) architectures. Further analysis of MoE models and extensions to vision tasks are reserved for future work.
this section cite: ['b12', 'b7']

Section: NeurIPS Paper Checklist

this section cite: []

Section: Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The main claim of the paper is presenting the first full fp4 training scheme on a 7B model. This is presented in the experiment section.
Guidelines:
• The answer NA means that the abstract and introduction do not include the claims made in the paper.
• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.
this section cite: []

Section: Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: In Section 6 we present the limitations of the paper.
Guidelines:
• The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate "Limitations" section in their paper.
• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.
this section cite: []

Section: Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?
Answer: [Yes] Justification: The theoretical results presented in Appendix B , includes all assumptions taken and is a complete proof.
Guidelines:
• The answer NA means that the paper does not include theoretical results.
• All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced.
• All assumptions should be clearly stated or referenced in the statement of any theorems.
• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.
this section cite: []

Section: Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: The paper Guidelines:
• The answer NA means that the paper does not include experiments.
• If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.
this section cite: []

Section: Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The paper use open source dataset and publish the full code to reproduce all experiments.
Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.cc/public/  guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.
this section cite: []

Section: Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?
Answer: [Yes] Justification: Section 5 includes all details about the experiments in the paper.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
• The full details can be provided either with the code, in appendix, or as supplemental material. 7. Experiment statistical significance Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We explain in the experiments section about the use of standard opensource train-test split, initialization.
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
Answer: [Yes]
Justification: The paper explain in Section 5 all details about the computer resources of the experiments.
Guidelines:
• The answer NA means that the paper does not include experiments.
• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.
this section cite: []

Section: Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?
Answer: [Yes] Justification: All data used in the paper is open source. The paper include full citation of all related works.
Guidelines:
• The answer NA means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
• The authors should state which version of the asset is used and, if possible, include a URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
• If this information is not available online, the authors are encouraged to reach out to the asset's creators.
13. New assets Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
Answer: [Yes]
Justification: The paper include anonymous github with all the code to reproduce the experiments. The code include full documentation required.
Guidelines:
• The answer NA means that the paper does not release new assets.
• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
• The paper should discuss whether and how consent was obtained from people whose asset is used.
• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.
this section cite: []

Section: Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.
this section cite: []

Section: Institutional review board (IRB) approvals or equivalent for research with human subjects
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
The loss at step n is
Ln = L( E[θn] ) = 1 2 λ e 2 n = λ 2 a n e0 -µε λ (1 -a n ) 2 .
As n → ∞, a n → 0 (for a < 0, which is required for successful optimization), yielding the stationary error and residual loss
e∞ = - µε λ , L∞ = µ 2 ε 2λ .
Thus, instead of converging to θ * with zero loss, biased SGD settles at
E[θ∞] = θ * - µε λ ,
and leaves a residual loss
L( E[θ∞] ) = µ 2 ε 2λ .
this section cite: []

Section: C Additional Experimental Results

this section cite: []

Section: 7 8 9 10 11 12 13 14 15
Proceed tokens (Billions)
2.90 2.95 3.00 3.05 3.10 3.15 3.20 Loss Full SR Weights Forward RtN Activations Forward RtN Weights Backward RtN Activations Update RtN Grad Update RtN Grad Backward RtN
this section cite: []

Section: 
• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).
this section cite: []

Section: Code of ethics
Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?
Answer: [Yes] Justification: The paper conform the Code of Ethics.
Guidelines:
• The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
• If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).
this section cite: []

Section: Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?
Answer: [Yes] Justification: In Appendix A we discuss the broader impacts of the paper.
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
Answer: [NA] Justification: The paper does not involve crowdsourcing nor research with human subjects.
Guidelines:
• The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.
this section cite: []

Section: Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.
Answer: [NA] Justification: LLM was used only for editing or formatting purposes.
Guidelines:
• The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
• Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.
this section cite: []

Section: References
Ref_id:b0 Title: Nvidia blackwell architecture Year: ()
Ref_id:b1 Title: Language models are few-shot learners Year: (2020)
Ref_id:b2 Title: Optimize weight rounding via signed gradient descent for the quantization of llms Year: (2023)
Ref_id:b3 Title: Accurate neural training with 4-bit matrix multiplications at standard formats Year: (2021)
Ref_id:b4 Title: Redpajama: an open dataset for training large language models Year: (2023)
Ref_id:b5 Title:  Year: ()
Ref_id:b6 Title: Qlora: Efficient finetuning of quantized llms Year: (2023)
Ref_id:b7 Title: Scaling fp8 training to trillion-token llms Year: (2024)
Ref_id:b8 Title: Gptq: Accurate post-training quantization for generative pre-trained transformers Year: (2022)
Ref_id:b9 Title: Automating interpretability: Discovering and testing visual concepts learned by neural networks Year: (2019)
Ref_id:b10 Title: Awq: Activation-aware weight quantization for on-device llm compression and acceleration Year: (2023)
Ref_id:b11 Title: Quest: Stable training of llms with 1-bit weights and activations Year: (2025)
Ref_id:b12 Title: Fp8-lm: Training fp8 large language models Year: (2023)
Ref_id:b13 Title: Nonlinear random matrix theory for deep learning Year: (2017)
Ref_id:b14 Title: Microscaling data formats for deep learning Year: (2023)
Ref_id:b15 Title: Empirical analysis of the hessian of over-parametrized neural networks Year: (2017)
Ref_id:b16 Title: Roformer: Enhanced transformer with rotary position embedding. Neurocomput., 568(C) Year: (2024-03)
Ref_id:b17 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b18 Title: Training llms with mxfp4 Year: (2025)
Ref_id:b19 Title: Bitnet: Scaling 1-bit transformers for large language models Year: (2023)
Ref_id:b20 Title: Optimizing large language model training using fp4 quantization Year: (2025)
Ref_id:b21 Title: Smoothquant: Accurate and efficient post-training quantization for large language models Year: (2022)
Ref_id:b22 Title: Root mean square layer normalization Year: (2019)
