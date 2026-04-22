Title: BEYOND AUTOREGRESSION: FAST LLMS VIA SELF-DISTILLATION THROUGH TIME
Abstract: Autoregressive (AR) Large Language Models (LLMs) have demonstrated significant success across numerous tasks. However, the AR modeling paradigm presents certain limitations; for instance, contemporary autoregressive LLMs are trained to generate one token at a time, which can result in noticeable latency. Recent advances have indicated that search and repeated sampling can enhance performance in various applications, such as theorem proving, code generation, and alignment, by utilizing greater computational resources during inference. In this study, we demonstrate that diffusion language models are capable of generating at least 32 tokens simultaneously, while exceeding the performance of AR models in text quality and on the LAMBADA natural language understanding benchmark. This outcome is achieved through a novel distillation method for discrete diffusion models, which reduces the number of inference steps by a factor of 32-64. Practically, at the 1.3B parameters scale, diffusion models, even without caching, can generate tokens at a rate that is up to 8 times faster than AR models employing KV-caching, and we anticipate further improvements with the inclusion of caching. Moreover, we demonstrate the efficacy of our approach for diffusion language models with up to 860M parameters.

Section: 
In recent years, autoregressive (AR) large language models (LLM) have exceeded expectations (Vaswani et al., 2017;Devlin et al., 2018;Radford et al., 2019;Brown et al., 2020b;Kaplan et al., 2020;Raffel et al., 2020;Fedus et al., 2022;Hoffmann et al., 2022;Chowdhery et al., 2023;Google, 2023;Touvron et al., 2023). Importantly, many breakthroughs in coding (Chen et al., 2021), mathematics, and reasoning (Trinh et al., 2024b;a;Romera-Paredes et al., 2024;Hosseini et al., 2024;Wang et al., 2024) were achieved based on decoding large amounts of completions from a base LLM.
Importantly, the benefits of repeated sampling can be so significant that it is often more efficient to use a smaller, faster model rather than a larger, slower one. More generally, one can improve the performance of a fixed model by scaling up computational resources at inference time (Madaan et al., 2023;Yao et al., 2023;Snell et al., 2024;Wu et al., 2024;Chen et al., 2024;Brown et al., 2024;Goyal et al., 2024), a phenomenon that was previously observed for games (Campbell et al., 2002;Silver et al., 2016;Lerer et al., 2019;Brown et al., 2020a;Jones, 2021). Hence, when tackling reasoning tasks, a major bottleneck is the latency of the model. In this work, we improve the decoding speed of LLMs by moving away from AR modeling. We build on recent breakthroughs in discrete diffusion (Lou et al., 2023;Sahoo et al., 2024;Shi et al., 2024;Ou et al., 2024). Our approach can generate text up to 8 times faster than AR models that Figure 2: Performance on LAMBADA after multiple rounds of SDTT with different distillation losses. We pre-train with the masked diffusion language modeling objective (MDLM) (Sahoo et al., 2024) and distill with 7 rounds of SDTT. Note that a single word in the LAMBADA data set often consists of multiple tokens. We greedily decode all tokens a single forward pass for the diffusion models and decode autoregressively for the AR models. use KV caching (Pope et al., 2022). Diffusion models are typically trained to maximize the evidence lower bound (ELBO), which does not consider the desired number of inference steps. Hence, vanilla diffusion models typically require thousands of decoding steps. Fortunately, it is possible to drastically reduce the inference costs of continuous diffusion models via distillation (Luhman & Luhman, 2021;Salimans & Ho, 2022). Continuous distillation methods rely on deterministic mappings from noise to data, such as DDIM (Song et al., 2022). The deterministic mappings can be efficiently learned by a student diffusion model to sample in fewer steps. We hypothesize that such deterministic map cannot exist for the diffusion language models studied in this work. Indeed, those models always initialize the denoising process with a sequence of masked token, hence a deterministic algorithm can only generate a single sample. As such, we devise a distillation method that does not does depend on deterministic maps. This is a significant finding because faster decoding mechanisms allow exploring a larger search space in applications that require search, planning, and reranking. In summary, our core contributions are as follows:
• We introduce Self-Distillation Through Time (SDTT), which allows generating at least 32 tokens at a time, while achieving better perplexity than GPT-2 with nucleus sampling for conditional and unconditional generation. Unlike many distillation methods for continuous diffusion models, SDTT does not rely on deterministic mappings such as DDIM (Song et al., 2022). SDTT is very simple and easy to implement.
• We show that SDTT can generate tokens up to 8 times faster than AR models that use KV caching, for models with 1.3B parameters, in 16 decoding steps. Importantly, the discrete diffusion model does not rely on activation caching, suggesting that there is potential for even greater efficiency gains. The latency gains for smaller models are even greater.
• We demonstrate the effectiveness of SDTT for models with up to 860M parameters. To the best of our knowledge, this represents the largest publicly available discrete diffusion language model.
• We evaluate the distilled students on LAMBADA (Paperno et al., 2016) and 6 multiplechoice questions benchmarks from Gao et al. (2021). We find that SDTT preserves the natural language understanding performance of the teacher.
this section cite: ['b74', 'b15', 'b52', 'b32', 'b53', 'b17', 'b28', 'b12', 'b20', 'b71', 'b10', 'b56', 'b30', 'b75', 'b42', 'b80', 'b63', 'b78', 'b9', 'b3', 'b21', 'b8', 'b62', 'b35', 'b31', 'b38', 'b58', 'b61', 'b46', 'b58', 'b51', 'b39', 'b66', 'b66', 'b48', 'b18']

Section: BACKGROUND

this section cite: []

Section: MASKED DIFFUSION LANGUAGE MODELING
We follow the notation of Sahoo et al. (2024) to introduce masked diffusion language modeling (MDLM). Language modeling can be framed as the sequential prediction task of discrete tokens (x i ) coming from a vocabulary X = Z <N = {0, ..., N -1} that can take N possible discrete values. A language model would predict sequences of length L, which can be defined as the sequences of x i 's originating from X L = x (i) = (x (i) 0 , . . . , x (i) L-1 ) i∈Z <K . Let D := x (0) , . . . , x (K-1) : x (i) ∈ X L denote the training set. The goal of language modeling is to sample from the unknown distribution p 0 : X L → [0, 1] that generated the samples in D.
Similarly to continuous diffusion, we sample from an approximation of p 0 by learning to denoise corrupted examples. One can sample from the model through ancestral sampling, starting from a stationary distribution. The stationary distribution of Sahoo et al. (2024) is such that all tokens of the sentence are replaced with a special MASK token like the MASK token used for pre-training BERT models. However, a key difference between BERT and MDLM is that MDLM is trained on sequences with varying levels of corruption, while BERT uses a fixed ratio.
Discrete absorbing diffusion process MDLM defines a forward process to corrupt data and a backward process to learn to recover data. MDLM uses a continuous-time formulation, with the data distribution denoted as p 0 and the stationary noise distribution as p 1 = π. The forward process linearly interpolates between the one-hot distribution defined by the original document x and the stationary distribution π, which places all mass on the MASK token. Mathematically,
q(z t |x) := Cat(z t ; α t x + (1 -α t )π),(1)
where the noise injection schedule is defined by α t , for t ∈ [0, 1]. The constraints on α t are that α t ∈ [0, 1], α t should be a strictly decreasing function of t, and α 0 ≈ 1, α 1 ≈ 0. The forward process is called absorbing because once a token is assigned to a MASK token, it cannot be reverted to a real token.
We can derive the analytical form of the reverse process q(z s |z t , x), with t > s and α t|s = αt αs as
q(z s |z t , x) = Cat z s ; [α t|s z t + (1 -α t|s )1π ⊤ z t ] ⊙ [α s x + (1 -α s )π] α t z ⊤ t x + (1 -α t )z ⊤ t π .(2)
Algorithm 1 Computing the Self-Distillation Through Time targets xteacher θ (z t , t, m /k)
1: Inputs: Noisy tensor x t ∈ R N ×L , Starting sampling time t start ∈ [0, 1] N , Number of sampling steps m /k ≥ 2, such that m /k ∈ N + , Sampling step size ∆ ∈ (0, 1), Mask token index M ∈ N, Minimal sampling time ϵ. 2: Output: Distillation targets xteacher θ (z t , t, m /k)
3: 4: target ← zeros(N , L, K) ▷ Allocate empty tensor for xteacher θ (z t , t, m /k) 5: z ← x t 6: for i = 0, ..., m /k -1 do 7: t curr ← max(t start -i • ∆, ϵ) ▷ Sampling step for the current time 8:
z new , ℓ teacher ← reverse sample(z, t curr , ∆) ▷ Updated z & log-probabilities x θ (z, t curr ) 9: U = z new ̸ = z
▷ Create mask U of tokens that were denoised
10: target[U ] ← ℓ teacher [U ] ▷ Extract log-probs for the denoised tokens 11: z ← z new ▷ Update z for the next iteration 12: end for 13: target[z == M ] = ℓ teacher [z == M ] ▷ Use log-probs of the last denoising step for masked tokens 14: return target ▷ Target log-probs for all masked tokens in x t Objective and parameterization To generate new samples, we can simulate the reverse process from eq. (2). Since the ground-truth sample x is unknown, Sahoo et al. (2024) learn an approximation x θ using a neural network with parameters θ. Sahoo et al. (2024) then use x θ instead of x to simulate the reverse process. The sampling distribution is denoted as p θ (z s |z t ) := q(z s |z t , x θ (z t , t)). Sahoo et al. (2024) optimize θ using a continuous version of the negative evidence lower bound (NELBO) of Sohl-Dickstein et al. (2015a). Previous research has shown that continuous-time objectives optimize the data likelihood better (Kingma et al., 2023). Due to the definition of the absorbing diffusion process, the NELBO simplifies to a weighted cross-entropy loss between the ground-truth x and the model predictions x θ :
L ∞ NELBO = E q t=1 t=0 α ′ t 1 -α t log⟨x θ (z t , t), x⟩dt.(3)
To derive eq. ( 3), Sahoo et al. (2024) impose two properties on p θ (z s |z t ). First, denoised tokens are never re-masked during sampling. Practically, this is achieved by manipulating the output of the neural network x θ (z t , t) to ensure that no probability mass is assigned to the MASK token. Secondly, already-denoised tokens are carried-over to the next sampling step. Sahoo et al. (2024) showed that both constraints lead to improved likelihood.
this section cite: ['b58', 'b58', 'b33', 'b58']

Section: KNOWLEDGE DISTILLATION
Knowledge distillation (Bucila et al., 2006;Hinton et al., 2015) is a technique where a student neural network is trained to imitate the predictions of a more complex teacher model. One of the main advantages of distillation is the ability to reduce the inference cost associated with sampling from large LLMs while surpassing the performance of smaller models trained without distillation (Gu et al., 2024;Agarwal et al., 2024). The most relevant to our work are the distillation methods that match the predictions of the teacher and the student using a divergence measure δ:
E x∼D [δ(µ s (x t |x <t ); µ t (x t |x <t ))] ,(4)
Where µ s , µ t are the AR distributions of the student and teacher, respectively, and D represent the training dataset. Common divergence measures include f -divergences (Wen et al., 2023) such as the Kullback-Leibler divergence (KLD) or the total variation distance (TVD).
Algorithm 2 One training round of Self-Distillation Through Time t start ∼ U[0, 1] ▷ Sample t uniformly at random 8:
x t ∼ q t (x t |x 0 ) ▷ Forward diffusion process. See eq. ( 1) 9:
x student ← x ν (x t , t)
10:
x teacher ← teacher SDTT(x t , t start , m /k, ∆, M , 1e-5) ▷ See algorithm 1 11:
L ← δ(x student ||x teacher ) ▷ Compute divergence between student and SDTT targets.
12:
ν ← backprop optim(L, ν) ▷ Update the parameters of the student with AdamW 13: end for 14: return x ν
this section cite: ['b6', 'b26', 'b22', 'b0', 'b77']

Section: METHOD

this section cite: []

Section: SELF-DISTILLATION THROUGH TIME
As explained in section 2.1, discrete diffusion language models optimize the NELBO over the training examples. Fewer decoding steps typically lead to lower sample quality because the approximation of the reverse process is less accurate, as visible in the teacher curve in fig. 4.
To address the issue of low sample quality with fewer decoding steps, we propose Self-Distillation Through Time (SDTT). SDTT fine-tunes a pre-trained MDLM to allow decoding with significantly fewer steps. Interestingly, our final model decodes samples with lower generative perplexity in 32 steps than the teacher would with 1024 forward passes. In short, SDTT improves the sampling speed by distilling the inference time computation to sample multiple steps into the student.
Let p (m) θ be the distribution of samples generated with m steps, using a denoiser with parameters θ. SDTT trains a denoiser with parameters ν to minimize a divergence d between p (m) θ and p (k) ν . Here k < m, and k divides m (e.g., m = 1024 and k = 512):
min ν d p (k) ν ||p (m) θ .(5)
Since x θ and x ν are the only learnable elements of the sampling process, they completely determine the sampling distributions p (m) θ and p
ν . As such, training x ν to match the predictions of x θ with fewer steps minimizes eq. ( 5). We now present a method for generating targets xteacher θ (z t , t, m /k) to train x ν . Mathematically, we optimize the following objective:
min ν E z0∼D,zt∼qt(zt|z0) δ(x ν (z t , t)||x teacher θ (z t , t, m /k)) ,(6)
where δ a divergence measure between the student and the teacher targets xteacher θ (z t , t, m /k)). We consider the Kullback-Leibler divergence (KLD), Total Variation Distance (TVD), and Mean-Squared Error (MSE). See appendix B for details on those divergence measures.
Generating the Teacher Targets Following the terminology of knowledge distillation, we call the denoiser x θ used for many steps decoding as the teacher and the denoiser x ν used for a few steps decoding as the student. To train x ν to match the predictions of x θ , we sample from the teacher for m /k steps. Whenever a MASK token is denoised, we collect the log probabilities predicted by the teacher for this MASK token. These log-probabilities become the distillation targets xteacher θ (z t , t, m /k). Algorithm 1 outlines this process and fig. 3a presents it visually. While fig. 3a shows how to distill two decoding steps in one, the procedure can be extended to larger values of m /k. The complete SDTT training loop is presented in algorithm 2.
Iterated SDTT SDTT reduces the number of decoding steps by a factor m /k. If we want to reduce the number of decoding steps further, we can apply SDTT with k ′ < k, or alternatively apply SDTT n times, using the newly distilled student as teacher for the next round, which we refer to as iterated SDTT. Instead of directly optimizing the divergence in eq. ( 5), we introduce n intermediate distributions p ki  νi such that m /ki is an increasing sequence as a function of i. In practice, we choose m = 2 10 and k i = 2 10-i with 0 ≤ i ≤ 7 and sequentially minimize the objective
min ν d p (kj+1) νj +1 ||p (kj ) νj ,(7)
for 0 ≤ j < 7, where ν j denotes the parameters of the j-th denoiser, with ν 0 = θ (teacher). If the minimization procedure was perfect, minimizing eq. ( 5) or eq. ( 7) should result in the same solution.
However in practice, we observe that it is easier to minimize eq. ( 7) sequentially for increasing values of i, in a progressive fashion, similar to Salimans & Ho (2022).
As an alternative to iterated SDTT, we tried using a single model and slowly growing the step size used to generate xteacher θ (z t , t, m /k). Unfortunately, this approach was unstable and the loss diverged after 30-50 steps, irrespective of how small the sampling step size was. Similar behavior was observed by Norouzi et al. (2023).
this section cite: ['b44']

Section: EXPERIMENTS
We distill MDLMs on the OpenWebText dataset (Gokaslan & Cohen, 2019) as it was used to train recent discrete diffusion language models (Lou et al., 2023;Sahoo et al., 2024). We use the Adam optimizer with a learning rate of 6e -5, a batch size of 128 and no weight decay. We linearly increase the learning rate for 500 training steps and keep it constant afterwards. As a base model, we reuse the checkpoint released by Sahoo et al. (2024). See appendix C for more details.
In section 4.1, we evaluate 3 distillation divergences and show that iterated SDTT can reduce the number of sampling steps by a factor 16-32. In section 4.2, we ablate on the importance of hyperparameters, including the duration of each round of iterated SDTT and the number of sampling steps to generate the targets xteacher θ (z t , t, m /k). In section 4.3, we scale SDTT to models with of up to 860M parameters. Finally, in section 4.4, we compare the latency of SDTT against autoregressive models that use KV caching.
Generative perplexity Following prior work (Dieleman et al., 2022;Lou et al., 2023;Sahoo et al., 2024), we use a larger model to compute the generative perplexity of unconditional and conditional samples. We evaluate the smallest students using GPT-2 (large) (Radford et al., 2019). In the scaling experiments, we use Llama3 8B (Touvron et al., 2023), since we compare models with up to 860M parameters. As noted by Zheng et al. (2024a), the generative perplexity is sensitive to the floating-point precision. In this section, we sample using bfloat16, and report results using float64 in appendix A. The conclusion are similar.
this section cite: ['b38', 'b58', 'b58', 'b16', 'b38', 'b58', 'b52', 'b71']

Section: MAUVE
We evaluate conditional generation using the MAUVE score (Pillutla et al., 2021). MAUVE measures how well a model follows a prompt by comparing multiple generations with a reference continuation. We use the first 1024 samples with at least 1024 tokens from the WebText dataset (OpenAI, 2019), take the first 50 tokens as a prompt, and generate 50 tokens of continuation. For each prompt, we generate 5 continuations, as done in Lou et al. (2023).
this section cite: ['b50', 'b38']

Section: Sample diversity
Post-training can drastically reduce the diversity of language models (Kirk et al., 2024;Agarwal et al., 2024;Li et al., 2024). Hence, we measure the diversity of samples using the self-BLEU score (Zhu et al., 2018) with the same completions used to compute MAUVE.
this section cite: ['b34', 'b0', 'b37', 'b85']

Section: Downstream performance
We measure the downstream performance using the LAMBADA dataset (Paperno et al., 2016), as well as 6 multiple-choice question (MCQ) tasks from Gao et al. (2021). On LAMBADA, we report an upper bound on the perplexity, computed using the ELBO (3). We also report the suffix accuracy by masking all tokens of the last word and predicting all of them in a single forward pass, using the argmax of the predictions. The diffusion model is correct only if all the masked tokens are decoded correctly in a single decoding step. The 6 other benchmarks from Gao et al. (2021) evaluate the MCQ accuracy. We measure the trade-off between quality and diversity using self-BLEU (Zhu et al., 2018). Deterministic sampling yields a score of 1. The diversity minimally decreases after distillation.
this section cite: ['b48', 'b18', 'b18', 'b85']

Section: ABLATION ON THE TRAINING DIVERGENCE
SDTT requires choosing a divergence δ and we study the Mean-Squared Error (MSE), Total Variation Distance (TVD) and (reverse) Kullback-Leibler Divergence (KLD). We apply iterated SDTT for 7 rounds of 10k training iterations and generate xteacher θ (z t , t, m /k) with 2 sampling steps from the teacher (algorithm 1). We use an exponential moving average (EMA) of the weights with a decay of 0.9999 that we do not reset between rounds.
Figure 2 shows that students distilled with the KLD clearly outperform students trained using the MSE and TVD on LAMBADA. The LAMBADA accuracy of students tuned with the KLD slightly improves over the teacher, while the perplexity remains better or matches the AR baselines for all but the last round of SDTT. The improved accuracy on LAMBADA suggests that the model is better at predicting multiple tokens in parallel after distillation with SDTT, since we evaluates the accuracy by decoding all tokens of the last word simultaneously.
Figure 5 shows that the KLD seem to outperform the MSE and TVD objectives on MAUVE. Since we generate sequences of 100 tokens only for MAUVE, following (Lou et al., 2023), we sample with at most 128 steps, and use samples generated with 128 sampling steps from the teacher as a baseline. Note that as observed by Deschenaux & Gulcehre (2024), discrete diffusion models typically achieve slightly lower MAUVE scores than AR models. Nonetheless, distillation with the KLD objective improves the MAUVE score of the students. Similarly fig. 18 shows that continuations from the student distilled with the KLD reaches the lowest perplexity and match GPT-2 with nucleus sampling in 32 forward passes.
In table 1, we compare the downstream performance on the tasks of Gao et al. (2021) before and after distillation. We observe that SDTT minimally affects the results, and that student distilled with the KLD objective reaches higher accuracies than other students in all but one task Figure 4a measures the diversity of samples using the self-BLEU score (Zhu et al., 2018), for the students distilled with the KLD objective. See appendix A for results with the MSE and TVD. We find that SDTT minimally decreases the diversity. Compared to distilling autoregressive models (Agarwal et al., 2024), SDTT minimally reduces the diversity. For reference, Agarwal et al. (2024) routinely observes an increase of 15 in self-BLEU while we observe a change of at most 2 for the KLD student. See appendix A for more results and details on the self-BLEU score.
Figure 6 shows that students distilled with KLD have higher unconditional generative perplexity than those distilled with the MSE. However, KLD is the only objective that preserves performance
8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 1 round 8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 2 rounds 8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 3 rounds 8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 4 rounds 8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 5 rounds 8 16 32 64 128 Num. sampling steps 0.80 0.85 0.90 MAUVE 6 rounds Teacher KLD MSE TVD in the LAMBADA data set while still significantly reducing the generative perplexity compared to the teacher. Therefore, in the remainder of this work, we focus on the KLD.
this section cite: ['b38', 'b14', 'b18', 'b85', 'b0', 'b0']

Section: ADDITIONAL ABLATIONS
Number of steps in each SDTT round In section 4.1, each round of SDTT consists of 10k training iterations. Since the magnitude of the distillation loss does not reliably indicate convergence, we experiment with shorter rounds. We find that reducing the number of training iterations to 5k or 2.5k negatively impacted conditional generation performance, as shown in fig. 7. However, shorter rounds slightly improved the final generative perplexity (fig. 8) and resulted in marginally better LAMBADA perplexity (fig. 10). Since SDTT does not directly optimize the ELBO, an increase in perplexity is expected. Interestingly, the LAMBADA accuracy remains unchanged with shorter rounds.
Number of sampling steps to generate the targets In section 4.1, the targets xteacher θ (z t , t, m /k) are generated using 2 sampling steps from the teacher. We explore distilling a larger number of sampling steps at once (4 or 8), since using more rounds of SDTT may induce more error accumulation in approximating the original teacher. Figure 13 shows that distilling more than two steps at a time is difficult and results in weaker results on LAMBADA. This suggests that the higher stochasticity of the targets generated with four or eight steps makes the task too difficult for the student.
Generating targets with the analytical sampler Lou et al. (2023) observe that using an analytical sampler (Campbell et al., 2022) results in higher quality samples compared to ancestral sampling. However, when generating targets xteacher θ (z t , t, m /k) with analytical sampling, we observed minimal difference with ancestral sampling, as shown in fig. 11 and 12.
Resetting the optimizer and Exponential Moving Average between rounds Using an Exponential Moving Average (EMA) of the weights is known to improve the quality of samples from diffusion models (Nichol & Dhariwal, 2021). However, when applying SDTT for multiple rounds, it is unclear whether the EMA or current weights should be used as the teacher for successive rounds. Additionally, it could be favorable to reset the optimizer state between rounds as we grow the decoding step size. We experiment with two approaches: either resetting the optimizer state only, or resetting both the EMA and optimizer state. Figure 14 shows the generative perplexity when resetting the optimizer state and using the EMA as the teacher instead of the current weights, while fig. 15 presents the corresponding results for MAUVE. When using the EMA as teacher, since we accumulate updates in the EMA over 10k training iterations only, we use a slightly lower decay rate of 0.999. We find that using the EMA of the weights as the teacher may slightly improve performance. While the KLD leads to a higher perplexity than the MSE; we focus on the KLD because it is the only divergence that retains the performance on the LAMBADA dataset.
this section cite: ['b38', 'b7', 'b43']

Section: SCALING SDTT TO 860M PARAMETERS
We apply SDTT to larger discrete diffusion models with up to 860M parameters. In this experiment, we train the models from scratch for 400k steps with a batch size of 512, a context length of 1024 and the Adam optimizer. We reuse the training configuration of Sahoo et al. (2024) and scale the models to larger sizes. We train 3 model sizes, small (169M), medium (424M) and large (863M). Details of the model architecture for each scale are shown in table 2. As for the other experiments, the models are diffusion transformers (Peebles & Xie, 2023) and we use an EMA with a decay of 0.9999. Although the results in section 4.2 suggest that short distillation rounds might be sufficient, it is unclear whether this result also holds on larger scales. Therefore, we use 10k steps per round of SDTT. For simplicity, we generate targets using 2 teacher ancestral decoding steps and do not reset the optimizer state or EMA between rounds.
Since we train larger models, we evaluate the generative perplexity using Llama3 8B (Touvron et al., 2023). The generative perplexity over the 3 model sizes is shown in fig. 4b. Interestingly, the smaller diffusion model (169M) sampled from with 64 steps or more after distillation achieves better generative perplexity than the largest model (863M) when sampling with 1024 steps. In fig. 16, we show that the MAUVE performance also improves after distillation for the medium and larger model. Finally, in fig. 17, we see that the LAMBADA accuracy improves after distillation, similar as in the smaller scale, when using the KLD objective.
this section cite: ['b58', 'b49', 'b71']

Section: LATENCY WITH SDTT
While SDTT allows sampling from discrete diffusion models with 32-64 times less decoding steps, a quantity of interest to practitioners is the actual latency of text generation. Indeed, while the reduction in the number of sampling steps is large, since discrete diffusion uses a non-causal architecture, we cannot use KV caching (Pope et al., 2022). KV caching improves the inference performance drastically for AR models, hence we compare the latency of SDTT with GPT-2 with KV caching. We successfully reproduce the results of Deschenaux & Gulcehre (2024), which showed a 4x improvement when sampling with 32 steps, and measure an 8x improvement with 16 decoding steps. We compute the latency using untrained models with around 1.3B parameters, using the same hyperparameters as Deschenaux & Gulcehre (2024). We use a batch size of 8 and time the sampling 10 times after one warm-up step on a single A100 GPU with 80 GiB of RAM. All models use FlashAttention (Dao et al., 2022). See Appendix A for additional experiments on the latency. (Sohl-Dickstein et al., 2015b;Ho et al., 2020;Song & Ermon, 2020) are the basis of many state-of-the-art text-to-image models (Ramesh et al., 2022;Rombach et al., 2022;Saharia et al., 2022). After their introduction by Sohl-Dickstein et al. (2015b), Ho et al.  (2020)showed that diffusion models can achieve FID scores (Heusel et al., 2017) comparable to GANs (Goodfellow et al., 2014;Arjovsky et al., 2017).  2024b) develop a family of re-parameterized discrete diffusion models to enhance the training and decoding efficiency. In parallel, several studies have explored continuous diffusion for language modeling (Li et al., 2022;Dieleman et al., 2022;Han et al., 2023;Chen et al., 2023;Gulrajani & Hashimoto, 2024). Despite recent breakthroughs, diffusion language models still have some drawbacks (Deschenaux & Gulcehre, 2024). Ye et al. (2024) adapt Chain-of-Thought reasoning (Wei et al., 2023) to diffusion models.
this section cite: ['b51', 'b14', 'b14', 'b13', 'b27', 'b67', 'b54', 'b55', 'b57', 'b25', 'b19', 'b1', 'b16', 'b24', 'b11', 'b23', 'b14', 'b81', 'b76']

Section: Diffusion Models Diffusion models

this section cite: []

Section: Discrete Diffusion & Diffusion Language Models

this section cite: []

Section: Distillation of Continuous Diffusion models
Distilling continuous diffusion models is a wellstudied area. For a comprehensive survey, see Luo (2023). Many distillation methods rely on Denoising Diffusion Implicit Models (DDIM) (Song et al., 2022), which showed that diffusion models can be sampled deterministically. Luhman & Luhman (2021) unroll trajectories sampled with DDIM and train a student to map noise directly to images. Luhman & Luhman (2021) pre-compute a dataset of noise-image pairs. Close to our work, Salimans & Ho (2022) teaches the student to match multiple sampling steps of the teacher, given corrupted training examples. However, unlike Salimans & Ho (2022), we cannot rely on the existence of a deterministic map via DDIM. Consistency distillation (Song et al., 2023) fine-tunes a pre-trained diffusion model to predict the final sample from intermediate points of the sampling trajectory, which enable faster sampling. Luo et al. (2024) distills a pre-trained diffusion model into single-step generator through a novel loss, Integral Kullback-Leibler divergence. SD-XL Turbo (Sauer et al., 2023) uses an adversarial formulation to sample with 1-4 steps from a latent diffusion model (Rombach et al., 2022).
Masked & Non Auto-Regressive Language Modeling BERT (Devlin et al., 2018) introduced the masked language modeling objective. While BERT focuses on representation learning, discrete diffusion language models are generative. XLNet (Yang et al., 2020) uses a generalized AR pretrtaining method to model the text distribution over all permutations of the training sequences, outperforming BERT on downstream tasks. Pannatier et al. (2024) adopt a similar objective to XLNet for generative modeling instead of natural language understanding.
this section cite: ['b40', 'b66', 'b39', 'b39', 'b69', 'b41', 'b60', 'b55', 'b15', 'b67', 'b47']

Section: DISCUSSION
In this work, we introduce Self-Distillation Through Time (SDTT), a distillation method for discrete diffusion models. Recent works (Lou et al., 2023;Sahoo et al., 2024;Shi et al., 2024;Ou et al., 2024) suggest that discrete diffusion models can match or outperform autoregressive models in text quality. However, those models require more inference resources than AR models to achieve good performance, because of the non-causal architecture of the neural network that prevents the use of KV caching. We show that SDTT can reduce the number of decoding steps while retaining performance. Our final student is up to 8x faster than AR models that use KV caching and we demonstrate that SDTT is applicable to larger models as well. In future work, we plan to evaluate SDTT on tasks that involve generating a large number of completions from a base language model.
We provide details on model architectures, hyperparameters, and provide pseudocode for our algorithm. We built on top of the open source model of Sahoo et al. (2024), which makes it relatively easy for researchers to reproduce our results. Additionally, upon de-anonymization, we will release our code and artifacts.
this section cite: ['b38', 'b58', 'b61', 'b46', 'b58']

Section: References
Ref_id:b0 Title: On-policy distillation of language models: Learning from self-generated mistakes Year: (2024)
Ref_id:b1 Title: Wasserstein gan Year: (2017)
Ref_id:b2 Title: Structured denoising diffusion models in discrete state-spaces Year: (2023)
Ref_id:b3 Title: Large language monkeys: Scaling inference compute with repeated sampling Year: (2024)
Ref_id:b4 Title: Combining deep reinforcement learning and search for imperfect-information games Year: (2020)
Ref_id:b5 Title: Language models are few-shot learners Year: (2020)
Ref_id:b6 Title: Model compression Year: (2006)
Ref_id:b7 Title: A continuous time framework for discrete denoising models Year: (2022)
Ref_id:b8 Title: Deep blue. Artificial Intelligence Year: (2002)
Ref_id:b9 Title: Are more llm calls all you need? towards scaling laws of compound inference systems Year: (2024)
Ref_id:b10 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b11 Title: Analog bits: Generating discrete data using diffusion models with self-conditioning Year: (2023)
Ref_id:b12 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b13 Title: Flashattention: Fast and memory-efficient exact attention with io-awareness Year: (2022)
Ref_id:b14 Title: Promises, outlooks and challenges of diffusion language modeling Year: (2024)
Ref_id:b15 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b16 Title: Continuous diffusion for categorical data Year: (2022)
Ref_id:b17 Title: Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity Year: (2022)
Ref_id:b18 Title: A framework for few-shot language model evaluation Year: (2019)
Ref_id:b19 Title: Generative adversarial networks Year: (2014)
Ref_id:b20 Title: Gemini: a family of highly capable multimodal models Year: (2023)
Ref_id:b21 Title: Think before you speak: Training language models with pause tokens Year: (2024)
Ref_id:b22 Title: Knowledge distillation of large language models Year: (2024)
Ref_id:b23 Title: Likelihood-based diffusion language models. Advances in Neural Information Processing Systems Year: (2024)
Ref_id:b24 Title: Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control Year: (2023)
Ref_id:b25 Title: Gans trained by a two time-scale update rule converge to a local nash equilibrium Year: (2017)
Ref_id:b26 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b27 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b28 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b29 Title: Argmax flows and multinomial diffusion: Learning categorical distributions Year: (2021)
Ref_id:b30 Title: V-star: Training verifiers for self-taught reasoners Year: (2024)
Ref_id:b31 Title: Scaling scaling laws with board games Year: (2021)
Ref_id:b32 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b33 Title: Variational diffusion models Year: (2023)
Ref_id:b34 Title: Understanding the effects of rlhf on llm generalisation and diversity Year: (2024)
Ref_id:b35 Title: Improving policies via search in cooperative partially observable games Year: (2019)
Ref_id:b36 Title: Diffusion-lm improves controllable text generation Year: (2022)
Ref_id:b37 Title: Entropic distribution matching in supervised fine-tuning of llms Year: (2024)
Ref_id:b38 Title: Discrete diffusion language modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b39 Title: Knowledge distillation in iterative generative models for improved sampling speed Year: (2021)
Ref_id:b40 Title: A comprehensive survey on knowledge distillation of diffusion models Year: (2023)
Ref_id:b41 Title: Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models Year: (2024)
Ref_id:b42 Title: Selfrefine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b43 Title: Improved denoising diffusion probabilistic models Year: (2021)
Ref_id:b44 Title: DiMS: Distilling multiple steps of iterative non-autoregressive transformers for machine translation Year: (2023-07)
Ref_id:b45 Title: Gpt-2 output dataset Year: (2019)
Ref_id:b46 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2024)
Ref_id:b47 Title: Sigma-gpts: A new approach to autoregressive models Year: (2024)
Ref_id:b48 Title: The lambada dataset: Word prediction requiring a broad discourse context Year: (2016)
Ref_id:b49 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b50 Title: Mauve: Measuring the gap between neural text and human text using divergence frontiers Year: (2021)
Ref_id:b51 Title: Efficiently scaling transformer inference Year: (2022)
Ref_id:b52 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b53 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b54 Title: Hierarchical textconditional image generation with clip latents Year: (2022)
Ref_id:b55 Title: Highresolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b56 Title: Mathematical discoveries from program search with large language models Year: (2024)
Ref_id:b57 Title: Photorealistic text-to-image diffusion models with deep language understanding Year: (2022)
Ref_id:b58 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b59 Title: Progressive distillation for fast sampling of diffusion models Year: (2022)
Ref_id:b60 Title: Adversarial diffusion distillation Year: (2023)
Ref_id:b61 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b62 Title: Mastering the game of go with deep neural networks and tree search Year: (2016-01)
Ref_id:b63 Title: Scaling llm test-time compute optimally can be more effective than scaling model parameters Year: (2024)
Ref_id:b64 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015-07)
Ref_id:b65 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b66 Title: Denoising diffusion implicit models Year: (2022)
Ref_id:b67 Title: Generative modeling by estimating gradients of the data distribution Year: (2020)
Ref_id:b68 Title: Score-based generative modeling through stochastic differential equations Year: (2021)
Ref_id:b69 Title: Consistency models Year: (2023)
Ref_id:b70 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2023)
Ref_id:b71 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b72 Title: Solving olympiad geometry without human demonstrations Year: (2024-01)
Ref_id:b73 Title: Solving olympiad geometry without human demonstrations Year: (2024)
Ref_id:b74 Title: Attention is all you need. Advances in neural information processing systems Year: (2017)
Ref_id:b75 Title: Math-shepherd: Verify and reinforce llms step-by-step without human annotations Year: (2024)
Ref_id:b76 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2023)
Ref_id:b77 Title: f-divergence minimization for sequence-level knowledge distillation Year: (2023)
Ref_id:b78 Title: An empirical analysis of compute-optimal inference for problem-solving with language models Year: (2024)
Ref_id:b79 Title: Xlnet: Generalized autoregressive pretraining for language understanding Year: (2020)
Ref_id:b80 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2023)
Ref_id:b81 Title: Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models Year: (2024)
Ref_id:b82 Title: Unified discrete diffusion for categorical data Year: (2024)
Ref_id:b83 Title: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling Year: (2024)
Ref_id:b84 Title: A reparameterized discrete diffusion model for text generation Year: (2024)
Ref_id:b85 Title: Texygen: A benchmarking platform for text generation models Year: (2018)
