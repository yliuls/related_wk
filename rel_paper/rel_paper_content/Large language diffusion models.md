Title: Large Language Diffusion Models
Abstract: The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised finetuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multiturn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: https://ml-gsai.github.io/LLaDA-demo/.

Section: Introduction
Large language models (LLMs) [1] fall entirely within the framework of generative modeling. Specifically, LLMs aim to capture the true but unknown language distribution p data (•) by optimizing a model distribution p θ (•) through maximum likelihood estimation, or equivalently KL divergence minimization between the two distributions:
max θ E pdata(x) log p θ (x) ⇔ min θ KL(p data (x)||p θ (x)) Generative modeling principles . (1
)
The predominant approach relies on the autoregressive modeling (ARM)-commonly referred to as the "next-token prediction" paradigm-to define the model distribution:
p θ (x) = p θ (x 1 ) L i=2 p θ (x i | x 1 , . . . , x i-1 ) Autoregressive formulation ,(2)
Figure 1: Zero/Few-Shot Benchmarks. We scale LLaDA to 8B parameters from scratch and observe competitive zero/few-shot performance compared with strong autoregressive LLMs [6].
where x is a sequence of length L, and x i is the i-th token. This paradigm has proven remarkably effective [2][3][4][5] and has become the foundation of current LLMs. Despite its widespread adoption, a fundamental question remains unanswered: Is the autoregressive paradigm the only path to achieving the core capabilities of LLMs, such as scalability, in-context learning, and instruction-following?
We argue that the answer is not a simple "yes". The key insight overlooked previously is: It is the generative modeling principles (i.e., Eq. ( 1)), rather than the autoregressive formulation (i.e., Eq. ( 2)) itself, that fundamentally underpin the essential properties of LLMs.
In particular, we argue that scalability is primarily a consequence of the interplay between Transformers [7], model size, data size, and Fisher consistencyfoot_0 [8] induced by the generative principles in Eq. ( 1), rather than a unique result of the ARMs in Eq. ( 2). The success of diffusion transformers [9,10] on visual data [11] supports this claim. Furthermore, the instruction-following and in-context learning [4] capabilities appear to be intrinsic properties of all conditional generative models on structurally consistent linguistic tasks, rather than exclusive advantages of ARMs. In addition, while ARMs can be interpreted as a lossless data compressor [12,13], any sufficiently expressive probabilistic model can achieve similar capabilities [14].
However, certain inherent limitations of LLMs can be directly attributed to their autoregressive nature. For instance, the left-to-right generation process restricts their ability to handle reversal reasoning tasks [15], highlighting a representative failure in the generalization capabilities of current models.
Motivated by these insights, we introduce LLaDA (Large Language Diffusion with mAsking) to investigate whether the capabilities exhibited by LLMs can emerge from generative modeling principles beyond ARMs, thereby addressing the fundamental question posed earlier. In contrast to traditional ARMs, LLaDA leverages a masked diffusion model (MDM) [16][17][18][19][20], which incorporates a forward data masking process and trains a mask predictor to approximate its reverse process. This design enables LLaDA to construct a model distribution with bidirectional dependencies and optimize a variational lower bound of its log-likelihood, offering a principled and previously unexplored perspective on the core capabilities of LLMs discussed above.
We adopt the standard pipeline of data preparation, pre-training, supervised fine-tuning (SFT), and evaluation, scaling LLaDA to an unprecedented language diffusion of size 8B. In particular, LLaDA 8B was pre-trained from scratch on 2.3 trillion tokens using 0.13 million H800 GPU hours, followed by SFT on 4.5 million pairs. Across diverse tasks, including language understanding, math, code, and Chinese, LLaDA demonstrates the following contributions:
• LLaDA scales effectively to a compute budget of 10 23 FLOPs, achieving comparable results to ARM baselines trained on the same data across six tasks, e.g., MMLU and GSM8K. • The pre-trained LLaDA 8B Base surpasses LLaMA2 7B Base [21] on nearly all 15 standard zero/few-shot learning tasks while performing on par with LLaMA3 8B Base [6], showcasing effective in-context learning capability.
• LLaDA significantly enhances the ability to follow instructions after SFT, as demonstrated in case studies such as multi-turn dialogue.
• LLaDA effectively breaks the reversal curse [15] with consistent performance across forward and reversal tasks. Notably, it outperforms GPT-4o in a reversal poem completion task.
this section cite: ['b0', 'b5', 'b1', 'b2', 'b3', 'b4', 'b6', 'b7', 'b8', 'b9', 'b10', 'b3', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b5', 'b14']

Section: Approach
In this section, we introduce the probabilistic formulation 6 , along with the pre-training, supervised fine-tuning, and inference procedures for LLaDA, as illustrated in Fig. 2.
this section cite: []

Section: Probabilistic Formulation
Unlike ARMs in Eq. ( 2), LLaDA defines a model distribution p θ (x 0 ) through a forward process and a reverse process [16][17][18][19][20]. The forward process gradually masks tokens independently in x 0 until the sequence is fully masked at t = 1. For t ∈ (0, 1), the sequence x t is partially masked, with each being masked with probability t or remaining unmasked with probability 1 -t. The reverse process recovers the data distribution by iteratively predicting masked tokens as t moves from 1 to 0.
The core of LLaDA is a mask predictor, a parametric model p θ (•|x t ) that takes x t as input and predicts all masked tokens (denoted as M) simultaneously. It is trained using a cross-entropy loss computed only on the masked tokens [18][19][20]:
L(θ) ≜ -E t,x0,xt 1 t L i=1 1[x i t = M] log p θ (x i 0 |x t ) ,(3)
where x 0 is a training sample, t is a continuous random variable drawn uniformly from [0, 1], x t is sampled from the forward process and L is the sequence length. The indicator function 1[•] ensures that the loss is computed only for masked tokens.
Once trained, we can simulate a reverse process (see Sec. 2.4 for details) parameterized by the mask predictor and define the model distribution p θ (x 0 ) as the marginal distribution induced at t = 0. The loss function in Eq. ( 3) has been proven to be an upper bound on the negative log-likelihood of the model distribution, making it a principled objective for generative modeling:
-E pdata(x0) [log p θ (x 0 )] ≤ L(θ).(4)
Notably, LLaDA employs a masking ratio that varies randomly between 0 and 1 while BERT [22] uses a fixed ratio. The subtle differences have significant implications, especially at scale: as shown in Eq. ( 4), LLaDA is a principled generative model with the potential to perform in-context learning and instruction-following naturally, akin to LLMs. Moreover, its generative perspective implies strong scalability with large data and models as discussed in Sec. 1. In addition, MaskGIT [23] adopts a heuristic training objective, which misses the 1 t term compared to Eq. ( 3), and lacks a theoretical link to maximum likelihood. We emphasize that it is precisely the theoretical foundation of maximum likelihood estimation that motivated us to scale discrete diffusion models for language modeling.
this section cite: ['b15', 'b16', 'b17', 'b18', 'b19', 'b17', 'b18', 'b19', 'b21', 'b22']

Section: Pre-training
LLaDA employs a Transformer [7] as the mask predictor, similar to existing LLMs. However, LLaDA does not use a causal mask, as its formulation allows it to see the entire input for predictions.
We trained two variants of LLaDA with different sizes: 1B and 8B. We summarize the model architecture of LLaDA 8B and LLaMA3 8B [6] here, and details are provided in Appendix B.2. We have ensured consistency in most hyperparameters while making several necessary modifications. We use vanilla multi-head attention instead of grouped query attention [24] for simplicity, as LLaDA is incompatible with KV caching, resulting in a different number of key and value heads. Consequently, the attention layer has more parameters, and we reduce the FFN dimension to maintain a comparable model size. Additionally, the vocabulary size differs due to a tokenizer [4] adapted on our data.
The LLaDA model is pre-trained on a dataset comprising 2.3 trillion (T) tokens, adhering to a data protocol that aligns closely with existing LLMs [25,26], without the incorporation of any special techniques. The data are derived from online corpora, with low-quality content filtered through manually designed rules and LLM-based approaches. Beyond general text, the dataset encompasses high-quality code, math, and multilingual data. Please refer to Appendix B.1 for more details about datasets. The mixing of data sources and domains is guided by scaled-down ARMs. The pre-training process utilizes a fixed sequence length of 4096 tokens, incurring a total computational cost of 0.13 million H800 GPU hours, similar to ARMs of the same scale and dataset size.
For a training sequence x 0 , we randomly sample t ∈ [0, 1], mask each token independently with the same probability t to obtain x t (see Fig. 2 (a)) and estimate Eq. ( 3) via the Monte Carlo method for stochastic gradient descent training. In addition, following Nie et al. [27], to enhance the ability of LLaDA to handle variable-length data, we set 1% of the pre-training data to a random length that is uniformly sampled from the range [1,4096].
We adopted the Warmup-Stable-Decay [28] learning rate scheduler to monitor the training progress without interrupting continuous training. Specifically, we linearly increased the learning rate from 0 to 4 × 10 -4 over the first 2000 iterations and maintained it at 4 × 10 -4 . After processing 1.2T tokens, we decayed the learning rate to 1 × 10 -4 and held it constant for the next 0.8T tokens to ensure stable training. Finally, we linearly reduced the learning rate from 1 × 10 -4 to 1 × 10 -5 for the last 0.3T tokens. Furthermore, we utilized the AdamW optimizer [29] with a weight decay of 0.1, a batch size of 1280, and a local batch size of 4 per GPU. The 8B experiment was executed once, without any hyperparameter tuning.
this section cite: ['b6', 'b5', 'b23', 'b3', 'b24', 'b25', 'b26', 'b0', 'b27', 'b28']

Section: Supervised Fine-Tuning
We enhance the capability of LLaDA to follow instructions by supervised fine-tuning (SFT) with paired data (p 0 , r 0 ), where p 0 is the prompt and r 0 denotes the response. This is the simplest and most basic post-training method for LLMs. Technically, this requires to model the conditional distribution
p θ (r 0 |p 0 ) instead of p θ (x 0 ) in pre-training.
The implementation is similar to pre-training. As shown in Fig. 2 (b), we leave the prompt unchanged and mask the tokens in the response independently, as done for x 0 . Then, we feed both the prompt and the masked response r t to the pre-trained mask predictor to compute the loss for SFT:
-E t,p0,r0,rt   1 t L ′ i=1 1[r i t = M] log p θ (r i 0 |p 0 , r t )   ,(5)
where L ′ denotes a dynamic length specified later, and all other notations remain the same as before.
Note that this approach is fully compatible with pre-training. Essentially, the concatenation of p 0 and r 0 can be treated as clean pre-training data x 0 , while the concatenation of p 0 and r t serves as the masked version x t . The process is identical to pre-training, with the only difference being that all masked tokens happen to appear in the r 0 portion.
The LLaDA 8B model undergoes SFT on a dataset comprising 4.5 million pairs. Consistent with the pre-training process, both data preparation and training follow the SFT protocols utilized in existing LLMs [25,26], without introducing any additional techniques to optimize LLaDA's performance. The dataset spans multiple domains, including code, mathematics, and instruction-following. We append |EOS| tokens to the end of short pairs in each mini-batch to ensure equal lengths across all data. We treat |EOS| as a normal token during training and remove it during sampling, enabling LLaDA to control the response length automatically. Please refer to Appendix B.1 for more details.
We train for 3 epochs on the SFT data using a similar schedule to the pre-training phase. The learning rate is linearly increased from 0 to 2.5 × 10 -5 over the first 50 iterations and then kept constant. During the final 10% of iterations, it is linearly reduced to 2.5 × 10 -6 . Additionally, we set the weight decay to 0.1, the global batch size to 256, and the local batch size to 2 per GPU. The SFT experiment was executed once, without any hyperparameter tuning.
this section cite: ['b24', 'b25']

Section: Inference
As a generative model, LLaDA can sample new text and evaluate the likelihood of candidate text in a diffusion manner instead of the left-to-right autoregressive fashion.
We begin with the reverse generation process. As illustrated in Fig. 2 (c), given a prompt p 0 , we discretize the reverse process to sample from the model distribution p θ (r 0 |p 0 ), starting from a fully masked response. The total number of sampling steps is a hyperparameter, which naturally provides LLaDA with a trade-off between efficiency and sample quality, as analyzed in Sec. 3.3. We employ uniformly distributed timesteps by default. In addition, the generation length is also treated as a hyperparameter, specifying the length of the fully masked sentence at the beginning of the sampling process. After generation, tokens appearing after the |EOS| token are discarded. As detailed in Appendix B.5, since both pre-training and SFT are conducted using datasets with variable lengths, the final results are insensitive to this length hyperparameter.
At an intermediate step from time t ∈ (0, 1] to s ∈ [0, t), we feed both p 0 and r t into the mask predictor and predict all masked tokens simultaneously. Subsequently, we remask s t of the predicted tokens in expectation to obtain r s , ensuring that the transition of the reverse process aligns with the forward process for accurate sampling [18][19][20]. In principle, the remasking strategy should be purely random. However, inspired by the annealing tricks of sampling in LLMs [4,30], we adopt a low-confidence remasking strategy, where s t of predicted tokens with the lowest confidence are remarked based on the predictions, same as the approach of Chang et al. [23].
We mention that LLaDA enables flexible sampling. In particular, it supports autoregressive and block diffusion [31] sampling directly after the pre-training or SFT processes described above, without requiring any further modifications or training. We provide a detailed analysis in Appendix B.4. Nevertheless, the diffusion sampling (i.e., the reverse generation process) yields the best performance and is adopted as the default throughout this paper, especially for all experiments presented in Sec. 3.
For conditional likelihood evaluation, we can naturally utilize the upper bound in Eq. ( 5). However, we find that the following equivalent form [20] exhibits lower variance and is more stable:
-E l,r0,r l L l L i=1 1[r i l = M] log p θ (r i 0 |p 0 , r l ) ,(6)
where L is the sequence length of r 0 , l is uniformly sampled from {1, 2, . . . , L}, and r l is obtained by uniformly sampling l tokens from r 0 without replacement for masking.
We present the training and inference algorithms, along with theoretical details, in Appendix A.
this section cite: ['b17', 'b18', 'b19', 'b3', 'b29', 'b22', 'b30', 'b19']

Section: Experiments
We evaluate the scalability, instruction-following, and in-context learning capabilities of LLaDA on standard benchmarks, followed by analyses and case studies to provide a comprehensive assessment.
this section cite: []

Section: Scalability of LLaDA on Language Tasks
We first investigate the scalability of LLaDA on downstream tasks in comparison with the ARM baselines we constructed. Specifically, at the 1B scale, we ensured that LLaDA and ARM shared the same architecture, data, and all other configurations. At larger scales, we also report results for LLaDA and ARM models of slightly different sizes trained on the same data due to resource limitations. Please refer to Appendix B.2 for more details. We use the pre-training computational cost as a unified scaling metric. For evaluation, we focused on six standard and diverse tasks.
Fig. 3 shows that LLaDA demonstrates impressive scalability, with its overall trend highly competitive with ARMs. Notably, on tasks such as MMLU and GSM8K, LLaDA exhibits even stronger scalability. Even on relatively weaker tasks like PIQA, the performance gap with ARMs narrows as scale increases. To account for the influence of outliers, we opted not to fit quantitative curves, avoiding potential misinterpretation. Nevertheless, the results clearly demonstrate the scalability of LLaDA. Considering LLaDA's advantages on certain benchmarks, we hypothesize that this performance gain stems from a key architectural difference: while autoregressive models optimize only left-to-right conditional probabilities, LLaDA is trained to consider multiple conditioning directions, as detailed in Appendix A.2, which may offer greater flexibility and lead to better generalization. This hypothesis is motivated by LLaDA's strong performance on reversal reasoning in Sec. 3.3 and the ablation studies on sampling strategies in Appendix B.4.
Nie et al. [27] suggests that MDM requires 16 times more computation than ARM to achieve the same likelihood. However, key differences make our findings more broadly applicable. In particular, likelihood is a relatively indirect metric for downstream task performance, and diffusion optimizes a bound of the likelihood, making it not directly comparable to ARM. Additionally, we extended the scaling range from 10 18 ∼ 10 20 FLOPs in Nie et al. [27] to 10 20 ∼ 10 23 FLOPs in this work.
this section cite: ['b26', 'b26']

Section: Benchmark Results
To comprehensively evaluate the in-context learning and instruction-following capabilities of LLaDA 8B, we conducted detailed comparisons with existing LLMs [6,21,25,26,32,33] of similar scale. Task selection and evaluation protocols followed existing studies, covering popular benchmarks in general tasks, mathematics, code, and Chinese. Further details are provided in Appendix B.6. For a more direct comparison, we re-evaluated representative LLMs [6,21] in our implementation.
As shown in Tab. 1, after pretraining on 2.3T tokens, LLaDA 8B Base demonstrates remarkable performance, surpassing LLaMA2 7B Base on nearly all tasks, and is overall competitive with LLaMA3 8B Base. LLaDA shows advantages in math and Chinese tasks. We conjecture that the Table 1: Benchmark Results of Pre-trained LLMs. * indicates that models are evaluated under the same protocol, detailed in Appendix B.6. Results indicated by † and ¶ are sourced from Yang et al. [25,26] and Bi et al. [32] respectively. The numbers in parentheses represent the number of shots used for in-context learning. "-" indicates unknown data. strengths stem from the same factors as its relatively weaker performance in some tasks-differences in data quality and distribution, largely due to the closed-source situation of LLM datasets.
LLaDA 8B * LLaMA3 8B * LLaMA2 7B * Qwen2 7B † Qwen2.5 7B † Mistral 7B † Deepseek 7B ¶ Model
Notably, we have carefully ruled out the possibility of data leakage by taking GSM8K as an example. First, as shown in Fig. 3, LLaDA outperformed ARM baselines regarding GSM8K. Moreover, the conclusion remains on a fully unseen GSM8K-like task [34] in Appendix B.8. Further, Tab. 2 compares the performance of LLaDA 8B Instruct with existing LLMs. SFT improved LLaDA's performance on most downstream tasks. A few metrics, such as MMLU, showed declines, possibly due to the suboptimal quality of the SFT data. Overall, since we did not perform alignment
Table 3: Visualization of the Sampling Process and a Generated Multi-round Dialogue. In the response of LLaDA, darker colors indicate tokens predicted in the later stages of sampling, while lighter colors correspond to earlier predictions. with reinforcement learning (RL), our results are slightly behind LLaMA3 8B Instruct, though the gaps in many metrics remain small. Notably, even with only SFT, LLaDA demonstrates impressive instruction-following abilities, as detailed in Sec. 3.4. We leave RL-based alignment for future work.
All results in Sec. 3 are based on pure diffusion methods, as they achieve better overall performance than approaches incorporating autoregressive components. Specifically, we use Eq. ( 6) for conditional likelihood estimation and apply low-confidence remasking for sampling. For LLaDA 8B Instruct, block diffusion style sampling performs better on GSM8K and Math, with scores of 78.6 and 42.2, compared to 69.4 and 31.9 in Tab. 2. This gain is due to extensive |EOS| token padding in the SFT data, causing early termination in low-confidence remasking. Please refer to Appendix B.4 for details.
Overall, despite the lack of data transparency, we have made every effort to adopt standardized procedures and introduce diverse tasks, we believe they sufficiently demonstrate the extraordinary capabilities of LLaDA, which is the only competitive non-autoregressive model to our knowledge.
this section cite: ['b5', 'b20', 'b24', 'b25', 'b31', 'b32', 'b5', 'b20', 'b24', 'b25', 'b31', 'b33']

Section: Reversal Reasoning and Analyses
To quantify the reversal reasoning [15] ability of models, we follow the protocol established in Allen-Zhu and Li [35]. Specifically, we construct a dataset of 496 famous Chinese poem sentence pairs. Given a sentence from a poem, models are tasked with generating the subsequent line (forward) or the preceding line (reversal) without additional fine-tuning. Examples can be found in Section B. 9. This setting provides a straightforward and more realistic evaluation compared to previous studies [27,36].
As shown in Tab. 4, LLaDA effectively addresses the reversal curse [15], demonstrating consistent zero-shot performance across both forward and reversal tasks. In contrast, both Qwen 2.5 and GPT-4o exhibit a significant gap between the two. The results on forward generation confirm that both ARMs are strong, benefiting from significantly larger datasets and greater computational resources than LLaDA. However, LLaDA outperforms both by a large margin in the reversal task. We did not design anything special for reversal tasks. Intuitively, LLaDA treats tokens uniformly without inductive bias, leading to balanced performance. See Appendix A.2 for details.
We also analyze the effect of different sampling strategies for LLaDA, including autoregressive sampling, block diffusion [31] sampling, and pure diffusion sampling, showing that pure diffusion sampling achieves the best overall performance, as detailed in Appendix B.4.
In addition, we examine LLaDA's sampling speed and memory consumption, showing that it enables a flexible trade-off between generation quality and speed. See Appendix B.7 for more details.
Classifier-free guidance (CFG) [37,27] is a widely used technique in diffusion models to improve generation quality. To ensure a fair comparison with ARMs, we do not apply CFG to LLaDA in the main text. However, we show that LLaDA is compatible with CFG and consistently benefits from its application. See Appendix B.3 for more details.
this section cite: ['b14', 'b34', 'b8', 'b26', 'b35', 'b14', 'b30', 'b36', 'b26']

Section: Case Studies
We present samples generated by LLaDA 8B Instruct in Tab. 3, showcasing its instruction-following capabilities. First, the table illustrates LLaDA's ability to generate coherent, fluent, and extended text in a non-autoregressive manner. Second, it highlights the model's multi-turn dialogue capability, effectively retaining conversation history and producing contextually appropriate responses across multiple languages. Such chat capabilities of LLaDA are impressive, as it departs from conventional ARMs for the first time, to the best of our knowledge. See more case studies in Appendix B.10.
this section cite: []

Section: Related Work
Diffusion models [38][39][40] have achieved remarkable success in visual domains but remain unverified for large-scale (e.g., models trained with over 10 23 FLOPs) language modeling, despite growing interest and extensive research efforts.
A simple approach is to continuousize text data and apply continuous diffusion models directly [41][42][43][44][45][46][47][48][49][50][51]. Alternatively, some methods model continuous parameters of discrete distributions instead [52][53][54][55][56]. However, scalability remains a significant challenge for these approaches. For instance, a 1B model may require 64 times the compute of an ARM to achieve comparable performance [57].
Another approach replaces continuous diffusion with discrete processes featuring new forward and reverse dynamics, leading to numerous variants [58][59][60][61][62][63][64][65][66][67][68][69][70][71]. The original diffusion model paper [38] introduced both continuous-state and discrete-state transition kernels under a unified diffusion framework. Austin et al. [16] was among the pioneering works that introduced discrete diffusion models into language modeling, demonstrating the feasibility of this approach. Lou et al. [17] showed that masked diffusion, as a special case of discrete diffusion, achieves perplexity comparable to or surpassing ARMs at GPT-2 scale. Shi et al. [18], Sahoo et al. [19], Ou et al. [20] established fundamental theoretical results, which motivated our model design, training, and inference (see Appendix A for details). Nie et al. [27] introduced the scaling laws for MDMs in language modeling and explored how MDMs can be leveraged for language tasks such as question answering at the GPT-2 scale. Gong et al. [72] demonstrated the potential of fine-tuning an ARM within the MDM framework. However, the improvements observed by Gong et al. [72] are limited to specific metrics, and their approach does not address the performance achievable through pure diffusion-based training. Concurrent work [73] demonstrates the potential of diffusion language models in code generation and highlights their advantages in inference efficiency. Nonetheless, as it is a closed-source product, specific details such as training procedures and sampling methods remain unknown.
In comparison, this study scales MDM to an unprecedented size of 8B parameters from scratch, achieving performance comparable to leading LLMs such as LLaMA 3.
Additionally, a parallel line of work on image generation [23,74,75] aligns well with the application of MDMs to text data. Moreover, MDMs have also shown promise in other domains such as protein generation [76,77], where they have achieved promising results. Notably, a series of studies [31,[78][79][80][81][82][83][84][85][86][87] have explored techniques such as architectural optimization, distillation, and sampling algorithm design to accelerate MDMs sampling.
this section cite: ['b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b65', 'b66', 'b67', 'b68', 'b69', 'b70', 'b37', 'b15', 'b16', 'b17', 'b18', 'b19', 'b26', 'b71', 'b71', 'b72', 'b22', 'b73', 'b74', 'b75', 'b76', 'b30', 'b77', 'b78', 'b79', 'b80', 'b81', 'b82', 'b83', 'b84', 'b85', 'b86']

Section: Conclusion and Discussion
We introduce LLaDA, a diffusion language model trained from scratch with an unprecedented scale of 8B parameters. LLaDA demonstrates strong capabilities in scalability, in-context learning, and instruction-following, achieving performance comparable to strong LLMs such as LLaMA3. In addition, LLaDA offers unique advantages, such as bidirectional modeling and enhanced robustness, effectively addressing the relevant limitations of existing LLMs. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that these essential capabilities are inherently tied to ARMs. These results represent a new paradigm for language modeling and uncover novel insights, demonstrating a high degree of scientific innovation.
Limitations. While promising, the full potential of diffusion models remains to be fully explored. Several limitations of this work present significant opportunities for future research. The generation length is a user-specified hyperparameter. Although LLaDA is insensitive to this hyperparameter as detailed in Appendix B.5, we believe that adopting an adaptive generation length would offer a more efficient solution. Due to computational constraints, direct comparisons between LLaDA and ARMs-such as training on identical datasets-were restricted to a computational budget of less than 10 23 FLOPs. To allocate resources for training the largest possible LLaDA model and showcasing its potential, we were unable to scale the ARM baseline to the same extent. Moreover, no specialized attention mechanisms or position embeddings were designed for LLaDA, nor were any system-level architectural optimizations such as KV cache applied. On the inference side, more efficient and controllable [37,88,89] sampling algorithms remain preliminary. Furthermore, LLaDA has yet to undergo alignment with reinforcement learning [90,91], which is crucial for improving its performance and alignment with human intent.
Looking ahead, both the model scale and the amount of training data for LLaDA remain smaller than those of leading ARM counterparts [6,26,[92][93][94][95], highlighting the need for further scaling to fully evaluate its capabilities. In addition, LLaDA's ability to process multi-modal data remains unexplored. Its impact on prompt tuning techniques [96] and integration into agent-based systems [97,98] is still not fully understood. Finally, a systematic investigation into post-training for LLaDA (e.g., O1-like systems [99,100]) is needed to further unlock the potential of diffusion language models.
this section cite: ['b36', 'b87', 'b88', 'b89', 'b90', 'b5', 'b25', 'b91', 'b92', 'b93', 'b94', 'b95', 'b96', 'b97', 'b98', 'b99']

Section: References
Ref_id:b0 Title: A survey of large language models Year: (2023)
Ref_id:b1 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b2 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b3 Title: Language models are few-shot learners Year: (2020)
Ref_id:b4 Title: Optimizing Language Models for Dialogue. OpenAI blog Year: (2022-11)
Ref_id:b5 Title: The llama 3 herd of models Year: (2024)
Ref_id:b6 Title: Attention is all you need Year: (2017)
Ref_id:b7 Title: On the mathematical foundations of theoretical statistics Year: (1922)
Ref_id:b8 Title: All are worth words: A vit backbone for diffusion models Year: (2023)
Ref_id:b9 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b10 Title: Video generation models as world simulators Year: (2024)
Ref_id:b11 Title: Language modeling is compression Year: ()
Ref_id:b12 Title: Compression represents intelligence linearly Year: (2024)
Ref_id:b13 Title: A mathematical theory of communication Year: (1948)
Ref_id:b14 Title: The reversal curse: Llms trained on" a is b" fail to learn" b is a Year: (2023)
Ref_id:b15 Title: Structured denoising diffusion models in discrete state-spaces Year: (2021)
Ref_id:b16 Title: Discrete diffusion language modeling by estimating the ratios of the data distribution Year: (2023)
Ref_id:b17 Title: Simplified and generalized masked diffusion for discrete data Year: (2024)
Ref_id:b18 Title: Simple and effective masked diffusion language models Year: (2024)
Ref_id:b19 Title: Your absorbing discrete diffusion secretly models the conditional distributions of clean data Year: (2024)
Ref_id:b20 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b21 Title: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b22 Title: Masked generative image transformer Year: (2022)
Ref_id:b23 Title: Gqa: Training generalized multi-query transformer models from multihead checkpoints Year: (2023)
Ref_id:b24 Title: Zhifang Guo, and Zhihao Fan. Qwen2 technical report Year: (2024)
Ref_id:b25 Title:  Year: (2024)
Ref_id:b26 Title: Scaling up masked diffusion models on text Year: (2024)
Ref_id:b27 Title: Unveiling the potential of small language models with scalable training strategies Year: (2024)
Ref_id:b28 Title: Decoupled weight decay regularization Year: (2017)
Ref_id:b29 Title: The curious case of neural text degeneration Year: (2019)
Ref_id:b30 Title: Block diffusion: Interpolating between autoregressive and diffusion language models Year: (2025)
Ref_id:b31 Title: Deepseek llm: Scaling open-source language models with longtermism Year: (2024)
Ref_id:b32 Title: Mistral 7b Year: (2023)
Ref_id:b33 Title: Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process Year: (2024)
Ref_id:b34 Title: Physics of Language Models: Part 3.2, Knowledge Manipulation Year: (2023-09)
Ref_id:b35 Title: The factorization curse: Which tokens you predict underlie the reversal curse and more Year: (2024)
Ref_id:b36 Title: Classifier-free diffusion guidance Year: (2022)
Ref_id:b37 Title: Deep unsupervised learning using nonequilibrium thermodynamics Year: (2015)
Ref_id:b38 Title: Denoising diffusion probabilistic models Year: (2020)
Ref_id:b39 Title: Score-based generative modeling through stochastic differential equations Year: (2020)
Ref_id:b40 Title: Diffusion-lm improves controllable text generation Year: (2022)
Ref_id:b41 Title: Diffuseq: Sequence to sequence text generation with diffusion models Year: (2022)
Ref_id:b42 Title: Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control Year: (2022)
Ref_id:b43 Title: Self-conditioned embedding diffusion for text generation Year: (2022)
Ref_id:b44 Title: Analog bits: Generating discrete data using diffusion models with self-conditioning Year: (2022)
Ref_id:b45 Title: Conor Durkan, et al. Continuous diffusion for categorical data Year: (2022)
Ref_id:b46 Title: Categorical sdes with simplex diffusion Year: (2022)
Ref_id:b47 Title: Ar-diffusion: Auto-regressive diffusion model for text generation Year: (2023)
Ref_id:b48 Title: Tess: Text-to-text self-conditioned simplex diffusion Year: (2024)
Ref_id:b49 Title: Dinoiser: Diffused conditional sequence learning by manipulating noises Year: (2023)
Ref_id:b50 Title: Planner: Generating diversified paragraph via latent language diffusion model Year: (2023)
Ref_id:b51 Title:  Year: (2023)
Ref_id:b52 Title: Bayesian flow networks Year: (2023)
Ref_id:b53 Title: Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise Year: (2023)
Ref_id:b54 Title: Unifying bayesian flow networks and diffusion models through stochastic differential equations Year: (2024)
Ref_id:b55 Title: Target concrete score matching: A holistic framework for discrete diffusion Year: (2025)
Ref_id:b56 Title: Likelihood-based diffusion language models Year: (2024)
Ref_id:b57 Title: Argmax flows and multinomial diffusion: Learning categorical distributions Year: (2021)
Ref_id:b58 Title: Rianne van den Berg, and Tim Salimans. Autoregressive diffusion models Year: (2021)
Ref_id:b59 Title: Diffusionbert: Improving generative masked language models with diffusion models Year: (2022)
Ref_id:b60 Title: A continuous time framework for discrete denoising models Year: (2022)
Ref_id:b61 Title: Concrete score matching: Generalized score matching for discrete data Year: (2022)
Ref_id:b62 Title: Diffuser: Discrete diffusion via edit-based reconstruction Year: (2022)
Ref_id:b63 Title: Score-based continuoustime discrete diffusion models Year: (2022)
Ref_id:b64 Title: Disk: A diffusion model for structured knowledge Year: (2023)
Ref_id:b65 Title: A reparameterized discrete diffusion model for text generation Year: (2023)
Ref_id:b66 Title: Fast sampling via de-randomization for discrete diffusion models Year: (2023)
Ref_id:b67 Title: Diffusion language models can perform many tasks with scaling and instruction-finetuning Year: (2023)
Ref_id:b68 Title: Discrete flow matching Year: (2024)
Ref_id:b69 Title: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling Year: (2024)
Ref_id:b70 Title: Diffusion on syntax trees for program synthesis Year: (2024)
Ref_id:b71 Title: Scaling diffusion language models via adaptation from autoregressive models Year: (2024)
Ref_id:b72 Title: Ultra-fast language models based on diffusion Year: (2025)
Ref_id:b73 Title: Text-to-image generation via masked generative transformers Year: (2023)
Ref_id:b74 Title: Effective and efficient masked image generation models Year: (2025)
Ref_id:b75 Title: Diffusion language models are versatile protein learners Year: (2024)
Ref_id:b76 Title: Dplm-2: A multimodal diffusion protein language model Year: (2024)
Ref_id:b77 Title: Cllms: Consistency large language models Year: (2024)
Ref_id:b78 Title: Show-o turbo: Towards accelerated unified multimodal understanding and generation Year: (2025)
Ref_id:b79 Title: Think while you generate: Discrete diffusion with planned denoising Year: (2024)
Ref_id:b80 Title: Dimo: Distilling masked diffusion models into one-step generator Year: (2025)
Ref_id:b81 Title: Fast solvers for discrete diffusion models: Theory and applications of high-order algorithms Year: (2025)
Ref_id:b82 Title: Distillation of discrete diffusion through dimensional correlations Year: (2024)
Ref_id:b83 Title: Informed correctors for discrete diffusion models Year: (2024)
Ref_id:b84 Title: Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling Year: (2024)
Ref_id:b85 Title: Jump your steps: Optimizing sampling schedule of discrete diffusion models Year: (2024)
Ref_id:b86 Title: Beyond autoregression: Fast llms via self-distillation through time Year: (2024)
Ref_id:b87 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b88 Title: Simple guidance mechanisms for discrete diffusion models Year: (2024)
Ref_id:b89 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b90 Title: Direct preference optimization: Your language model is secretly a reward model Year: (2024)
Ref_id:b91 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b92 Title: Our next-generation model: Gemini 1.5, 2024 Year: ()
Ref_id:b93 Title: Claude 3.5 sonnet Year: (2024)
Ref_id:b94 Title: Deepseek-v3 technical report Year: (2024)
Ref_id:b95 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b96 Title: Generative agents: Interactive simulacra of human behavior Year: (2023)
Ref_id:b97 Title: A survey on large language model based autonomous agents Year: (2024)
Ref_id:b98 Title: Learning to reason with llms Year: (2024)
Ref_id:b99 Title: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning Year: (2025)
Ref_id:b100 Title: A deep and tractable density estimator Year: (2014)
Ref_id:b101 Title: Training and inference on any-order autoregressive models the right way Year: (2022)
Ref_id:b102 Title: Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing Year: (2024)
Ref_id:b103 Title: Empowering code generation with oss-instruct Year: (2023)
Ref_id:b104 Title: Root mean square layer normalization Year: (2019)
Ref_id:b105 Title: Glu variants improve transformer Year: (2020)
Ref_id:b106 Title: Roformer: Enhanced transformer with rotary position embedding Year: (2024)
Ref_id:b107 Title: Scaling laws for neural language models Year: (2020)
Ref_id:b108 Title: Training compute-optimal large language models Year: (2022)
Ref_id:b109 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b110 Title: Challenging bigbench tasks and whether chain-of-thought can solve them Year: (2022)
Ref_id:b111 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b112 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b113 Title: Measuring how models mimic human falsehoods Year: (2021)
Ref_id:b114 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b115 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b116 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b117 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b118 Title: Gpqa: A graduate-level google-proof q&a benchmark Year: (2023)
Ref_id:b119 Title: Evaluating large language models trained on code Year: (2021)
Ref_id:b120 Title: Efficient training of language models to fill in the middle Year: (2022)
Ref_id:b121 Title: Program synthesis with large language models Year: (2021)
Ref_id:b122 Title: Measuring massive multitask language understanding in chinese Year: (2023)
Ref_id:b123 Title: C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models Year: (2024)
Ref_id:b124 Title: A framework for few-shot language model evaluation Year: ()
