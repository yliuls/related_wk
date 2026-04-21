Title: Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning
Abstract: Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability.

Section: Introduction
Transformers, the foundation of modern large language models (LLMs), leverage attention and feedforward layers to compute logits for sequence generation [77]. Pre-trained on general-domain corpora, these models capture rich statistical patterns and exhibit strong generation capabilities [13,80,57]. On top of that, supervised fine-tuning (SFT) serves as a critical technique for adapting pre-trained LLMs to specific domains [40,65,80]. While SFT enables significant flexibility and task-specific optimization, the performance of fine-tuned LLMs during inference often remains suboptimal, exhibiting misalignment between training and testing stages [47,79]. This gap arises from the model's inability to fully capture task-specific nuances or from overfitting to patterns within the training data, ultimately degrading its final performance [65,56,95,53]. Without data-side interventions [52,54,27] or receiving external feedback [58,72,91], this paper aims to address a fundamental question: Can we enhance the inference performance by retaining and leveraging the model's own learning signals in standard fine-tuning?
To address this question, our core idea draws inspiration from a common strategy by human learners: maintaining a log to record mistakes during practice, reflecting, and using insights to improve performance in formal tests. Rather than merely memorizing these mistakes, proficient learners engage in reflective thinking-analyzing their internal cognitive states at the moment the errors 2 Definition of Mistake Log
this section cite: ['b76', 'b12', 'b79', 'b56', 'b39', 'b64', 'b79', 'b46', 'b78', 'b64', 'b55', 'b94', 'b52', 'b51', 'b53', 'b26', 'b57', 'b71', 'b90']

Section: Preliminary and Notations
Let f P (•; θ P ) denote the function computed by a standard Transformer model [77], parameterized by θ P . In our context, we refer to f P as the Pilot model. Suppose there are T fine-tuning rounds. For each round t ∈ [T ], given an input sequence X t = (x t,1 , . . . , x t,n ) where n is the maximum sequence length, the input is sampled from a data distribution D = D X ,Y over input-output pairs. The Pilot model then generates an output sequence Ŷt = (ŷ t,1 , . . . , ŷt,n ) in an auto-regressive manner to approximate the target sequence Y t = (y t,1 , . . . , y t,n ), where (X t , Y t ) ∼ D.
During t-th fine-tuning round, let X t denote the input representation of X t , defined as either the encoder output in an encoder-decoder Transformer or the output of the token and positional embedding layer in a decoder-only Transformer. In the forward pass through the residual stream of the model, let L P be the total number of decoder layers in the Pilot model. For each layer l ∈ [L P ], we define h t,i,l ( X t ; θ P t-1 ) as the (decoder) hidden representations of the i-th token. After the final decoder layer, the Pilot model outputs logits over the vocabulary V , conditioned on the input X t and shifted target sequence y t,<i . The resulting output probabilities for the i-th token are given by: pt,i = softmax f P (X t , y t,<i ; θ P t-1 ) .
(1) We denote p t,i the ground-truth distribution over V for the i-th token, which places full probability mass on the correct token y t,i . The objective of training f P is to minimize the cross-entropy loss between the predicted and ground-truth tokens, formulated as:
L P t = - n i=1
log pt,i (y t,i | X t , y t,<i ).
2. 2 The Mistake Log Next, we define the Mistake Log in fine-tuning scenarios. As shown in Figure 1, the Mistake Log concludes three key components: the input representations (Questions), internal hidden states representations (Rationales), and the token-level error made by the model (Mistakes).
In each round t ∈ [T ], draw the sequence pair (X t , Y t ) ∼ D. As defined in Section 2.1, we set X t as the input representation component, as it provides contextual grounding for the Pilot model's specific input sequence. Inspired by prior works [25,20,14,45], the intermediate states' hidden representations produced by Transformer blocks also encapsulate rich contextual and semantic information, reflecting the model's internal rationales. Therefore, we define h t (X t ; θ P t-1 ) as the collection of these internal hidden representations for each token in round t: h t ( X t ; θ P t-1 ) = h t,i ( X t ; θ P t-1 ) n i=1 , with h t,i ( X t ; θ P t-1 ) = h t,i,l ( X t ; θ P t-1 )
L P l=1 , (3
)
where h t,i ( X t ; θ P t-1 ) captures the i-th token level internal states representation at the point when the i-th token error occurs. Then, to quantify the token-level error of the Pilot model, we compute the discrepancy between the predicted distribution pt,i and the ground-truth distribution p t,i for each token, with the error defined as: ℓ t (p t , pt ) = {ℓ t (p t,i , pt,i )} n i=1 , with ℓ t (p t,i , pt,i ) = p t,i -pt,i .
(4) We use the encoder-decoder architecture as an example here.
…
Consistent with standard LLM fine-tuning procedures, where the loss L P t is used to compute gradients and update the Pilot model's parameters across T rounds, we simultaneously collect key intermediate signals described above into the Mistake Log throughout this process. Formally, we define the Mistake Log as: M T = X t , h t ( X t ; θ P t-1 ), ℓ t (p t , pt )
T t=1 . (5
)
The Mistake Log systematically records contextual inputs, internal representations, and token-level prediction errors of the Pilot model throughout its entire fine-tuning trajectory. We next investigate how to leverage the Mistake Log during fine-tuning to enhance the Pilot model's final inference performance.
Motivation for Transformer Copilot. Recall that the goal of SFT is to optimize θ P by minimizing the expected loss E (Xt,Yt)∼D L P t . While this process adjusts model parameters using gradient descent, it treats each error as a transient signal, consumed and discarded immediately after the parameter update. As a result, the final model parameters θ P T might not retain an explicit memory of where, how, or why errors occurred during the training trajectory. This oversight leaves valuable training-time information, which we captured in the Mistake Log, untapped at inference time. To address this, we propose a new Copilot model to learn from the Mistake Log. Rather than altering the Pilot's optimization path, the Copilot operates as an auxiliary module that internalizes the distribution of past mistakes and corrects the Pilot's output at inference time. This design enables the Copilot to assist the Pilot model by reflecting on prior missteps and adaptively revising the predictions.
this section cite: ['b76', 'b1', 'b24', 'b19', 'b13', 'b44']

Section: Transformer Copilot
We introduce our proposed framework, Transformer Copilot, which is designed for both encoderdecoder and decoder-only Transformer architectures. In the following sections, we will elaborate on the Copilot model design, the training paradigm, and the inference paradigm, respectively.
this section cite: []

Section: The Copilot Model Design
The Copilot model is initialized from the decoder module of the corresponding Pilot model, but with several new architectural modifications. Consistent with the Pilot model f P , we denote the Copilot model as f C , parameterized by θ C . The Copilot model is also auto-regressive, generating outputs over the vocabulary V . However, the objective of the Copilot model is to learn from the Mistake Log M T and output rectified logits that correct the predictions made by the Pilot model. Below, we specify the Copilot model design for the encoder-decoder and decoder-only Pilot model separately.
Encoder-Decoder Copilot. As shown in Figure 2.1, the Copilot model receives its inputs from the Mistake Log, M T = {( X t , h t ( X t ; θ P t-1 ), ℓ t (p t , pt ))} T t=1 . Specifically, the Copilot is conditioned on the sequence of token-level errors made by the Pilot model, as recorded in M T , i.e. ℓ t,<i = (p t,1 -pt,1 , . . . , p t,i-1 -pt,i-1 ). These discrepancy sequences are provided as labels during training from M T and are auto-regressively generated during inference. As positional information is inherently preserved through the Pilot's output, we apply a single linear layer to project the token-level errors from vocabulary space into the Copilot's hidden dimension. Next, to incorporate additional information from the Pilot's input and internal hidden representations ( X t and h t from M T ), we propose a modified cross-attention mechanism in each layer of the Copilot, defined as:
New Q = H C l-1 • W Q , for l = 1, ..., L C , New K = Concat X t , Pool L P h t ( X t ; θ P t-1 ) • W K , New V = Concat X t , Pool L P h t ( X t ; θ P t-1 ) • W V ,(6)
where Pool L P (•) denotes the mean pooling across L P layers of the Pilot and Concat(•) indicates concatenation along the sequence dimension to ensure input dimensional compatibility and computational efficiency; H C l-1 is the Copilot model's hidden state from the previous layer (or input projection layer at l = 1); and W Q , W K , W V are learnable attention weights. We then apply the standard scaled dot-product attention using the new Q, K, and V . This modified attention allows the Copilot to jointly attend to both the external input context and the internal processing dynamics of the Pilot. Note that all components retrieved from the Mistake Log can be directly accessed during the forward pass of the Pilot model, without incurring additional computational overhead. After the final layer L C , we add a linear projection layer in the Copilot model to map the residual hidden representation into the vocabulary space, producing rectified logits as the output.
Decoder-only Copilot. We slightly adapt the Copilot model to accommodate the corresponding decoder-only Transformer [76,1], while keeping the majority of the model input and design above unchanged. Specifically, we modify the self-attention mechanism to incorporate the information from the Mistake Log: In the odd-numbered layers of L C , we retain the standard self-attention to allow the Copilot model to capture intra-sequence dependencies; In the even-numbered layers, we replace self-attention with the modified cross-attention mechanism defined in Eq. 6, enabling the Copilot to attend to the Pilot's input and internal state representations stored in M T . This alternating structure is consistent with the encoder-decoder Copilot to capture its own error-correction dynamics and attend to informative signals from the Pilot's behavior. We also explore several alternative designs and empirically validate the effectiveness of our proposed design against these variants in Appendix F.4.
this section cite: ['b75', 'b0']

Section: Learning Objective.
Give the sequence pair (X t , Y t ), at t-th round, the objective of training the Copilot model f C at i-th token is defined as: 9 Mt ← Mt-1 ∪ ( Xt, ht( Xt; θ P t-1 ), ℓt(pt, pt)) 10 Compute L P t via Eq.2; 11 Update θ P t ← θ P t-1 -ηP ∇ θ P t-1 L P t 12 /* For brevity, we reuse notation t */ 13 Draw ( Xt, ht( Xt; θ P t-1 ), ℓt(pt, pt)) ∼ Mt 14 ▽ Copilot -learn from the Mistake Log ( §3.1) 15 for i = 1, . . . , n do 16 f C t,i ← f C ( Xt, ht,<i, ℓt,<i; θ C t-1 ) 17 end 18 Compute L C t via Eq.7; 19 Update θ C t ← θ C t-1 -ηC ∇ θ C t-1 L C t 20 end 21 return θ P T , θ C T where f C t,i is the Copilot model's prediction, ℓ t (p t,i , pt,i ) = p t,i -pt,i is the corresponding label for the Copilot model, and h t,<i is the collection of Pilot's hidden states for the preceding tokens. We adopt the RMSE loss to prevent the distribution error from being further diminished by the square operation, avoiding the over-smoothing effect that squaring may introduce in the gradient signal during backpropagation. Next, we show how to jointly train the Pilot model f P and the Copilot model f C during fine-tuning, and collaborate on the generation during inference. Algorithm 2: Inference Paradigm Input: θ P T , θ C T ; Tuning parameter λ 1 Draw new Xt ∼ DX , t > T 2 for i = 1, . . . , n do 3 pt,i ← softmax(f P (Xt, ŷt,<i; θ P T )) 4 Observe Xt, ht,<i from f P 5 f C t,i ← f C ( Xt, ht,<i, f C t,<i ; θ C T ) 6 pt,i ← pt,i + λf C t,i (via Eq.8) 7 ŷt,i ← Decoding(pt,i) 8 end 9 return (ŷt,1, . . . , ŷt,n)
L C t = n i=1 ∥f C t,i -ℓ t (p t,i , pt,i )∥ 2 , with f C t,i = f C ( X t , h t,<i , ℓ t,<i ; θ C t-1 ),(7)
this section cite: []

Section: Training Paradigm.
Algorithm 1 outlines the process for jointly training the Pilot and Copilot model. In training round t ∈ [T ], one sequence pair (X t , Y t ) is drawn from the data distribution D. For each token i ∈ [n], we first compute the Pilot model's output distribution pt,i (Line 5-7). We then retrieve information directly from the forward pass of the Pilot model and update the Mistake log M t by recording X t , h t , and ℓ t for each token (Line 9). Meanwhile, we compute the Pilot model's cross-entropy loss L P t and update its parameters (Lines 10-11). Next, we prepare the input for training the Copilot model. Given all collected previous training rounds' information, we draw a sample ( X t , h t , ℓ t ) from the updated mistake log M t (Line 13). We obtain the Copilot model's output f C t,i for each token i ∈ [n] (Line 15-17). Finally, we compute the Copilot model's RMSE loss L C t and update its parameters (Line 18-19). After T rounds of iterative training, we obtain the final θ P T and θ C T for the Pilot and Copilot model, respectively. Note that this fine-tuning process can be readily extended to mini-batch stochastic gradient descent for scalability.
this section cite: []

Section: Inference Paradigm
After learning from the Mistake Log, the Copilot model is deployed alongside the Pilot model to enhance inference-time generation. To avoid abuse of notation, we reuse the same symbols as in training. Given a new input sequence X t ∼ D X , t > T , where X t is not part of the training data, t indexes the inference-time inputs and does not correspond to training rounds. As the objective of the Copilot model is to predict the token-level probability discrepancy p t,i -pt,i , we directly use the Copilot model's output to rectify the Pilot model's prediction pt,i towards the ground-truth p t,i . Formally, the rectified predicted distribution is given by:
pt,i = pt,i + λf C t,i ,(8)
where λ (typically set to 1) is a tunable hyperparameter controlling correction strength. Introducing λ at inference allows for more flexible modulation, and as we later show in Section 4, with a proper λ, the rectified pt,i theoretically provides a closer approximation to the target distribution p t,i . Algorithm 2 outlines the overall inference paradigm. Given X t , the Pilot model outputs a predicted distribution pt,i at each token generation step i ∈ [n] (Line 3). Subsequently, the Copilot model auto-regressively computes its output f C t,i (Line 5). Finally, the rectified pt,i is obtained via Eq.8 and used to generate the next token via a decoding function (Lines 6-7). The inference process is adaptive and can optionally terminate upon generation of the [EOS] (end-of-sequence) token.
this section cite: []

Section: Analyses -Why Learn from the Mistake Log?
To elucidate the roles of the Mistake Log and Copilot model in enhancing the Pilot model's inferencetime performance, we present both theoretical and empirical analyses in this section.
this section cite: []

Section: Theoretical Guarantee.
Recall that the Copilot model f C is designed to analyze the Pilot model's internal cognitive states X t , h t via the collected Mistake Log M T , and learns to predict errors measured by the token-level discrepancies ℓ t (p t,i , pt,i ). During inference, we use the rectified prediction as pt,i = pt,i + λf C t,i . In the following analysis, we show that, under mild assumptions, the adjusted prediction pt,i yields improved inference performance over the original estimate pt,i . Let A P , A C denote the distributions over the function classes of θ P , θ C , induced by the randomness in the fine-tuning process. Let [k] denote the k-th dimension of a vector in R |V | . Then, we define the expected error and variance of the Pilot and Copilot model at the k-th output dimension as:
ϵ 2 P = E (X t ,Y t )∼D (pt,i[k] -E θ P ∼A P [pt,i[k] | ŷt,<i]) 2 , σ 2 P = E (X t ,Y t )∼D Var θ P ∼A P [pt,i[k] | ŷt,<i] , ϵ 2 C = E θ P ∼A P (X t ,Y t )∼D pt,i[k] -pt,i[k] -E θ C ∼A C [f C t,i [k] | f C t,<i ] 2 | ŷt,<i , σ 2 C = E θ P ∼A P (X t ,Y t )∼D Var θ C ∼A C [f C t,i [k] | f C t,<i ] | ŷt,<i .
Theorem 4.1. For any k ∈ [|V |], suppose that ϵ 2 P > 0 and ϵ C < ϵ 2 P + σ 2 P . Then there exists λ 0 > 0 such that for any 0 < λ < λ 0 , the rectified prediction pt,i = pt,i + λf C t,i yields a strictly closer approximation to the ground-truth distribution p t,i at dimension k. Specifically, at the i-th token prediction step for X t ∼ D X , we have:  Pilot + Copilot Answer: Question: Choose the correct answer to the question: Carson was at a friend's house but suddenly announced they needed to go home. Why did Carson do this? Answer1: caught a bus; Answer2: called a cab; Answer3: forgot to feed the dog. Answer format: answer1/answer2/answer3.
E θ P ∼A P θ C ∼A C (X t ,Y t )∼D (pt,i[k] -pt,i[k]) 2 f C t,<i , ŷt,<i < E θ P ∼A P (X t ,Y t )∼D (pt,i[k] -pt,i[k]) 2 ŷt,<i .
this section cite: []

Section: Pilot Model Only Answer:
The choice is forgot to feed the dog.
The choice is answer 3
this section cite: []

Section: Empirical Analysis.
Complementing our theoretical analysis, we empirically examine the rectification effectiveness of the Copilot model during inference. We leave the setups in Appendix C. We further verify that this Copilot's adjustment indeed steers the token prediction toward the correct direction: We analyze representative error patterns frequently observed in the Pilot model's output, particularly factual and formatting mistakes. Figure 4 shows a detailed example of token-level logits rectification on Pilot model LLaMA-3.2-3B by the 1B Copilot, visualized using the layer-wise Logits Lens [8]. At mid-inference, the Pilot does not follow the correct answer format and makes mistakes (the correct token 'answer' has a high but suboptimal logit). The Copilot rectifies the prediction by decreasing the logit of the incorrect token 'forgot' and amplifying that of the correct token, thereby correcting the token prediction error. We leave analyses on other error patterns in Appendix C.
this section cite: ['b7']

Section: Empirical Evaluations
Tasks and Datasets. To comprehensively evaluate T-Copilot, we utilize a broad suite of reasoning and generation tasks: (i) Commonsense reasoning: PIQA [10], HellaSwag [93], WinoGrande [68], BoolQ [18], SIQA [70], and OpenbookQA (OBQA) [55]. (ii) Arithmetic reasoning: AQuA [48], GSM8K [19], MAWPS [42], and SVAMP [60]. and (iii) Downstream Recommendation: Beauty [30] and LastFM [67]. Detailed dataset descriptions are provided in Appendix D.
this section cite: ['b9', 'b92', 'b67', 'b17', 'b69', 'b54', 'b47', 'b18', 'b41', 'b59', 'b29', 'b66']

Section: Implementation Details.
For T-Copilot, we construct the Copilot model using the same type of decoder architecture as the Pilot model to ensure consistency. We use the AdamW optimizer and Cosine learning rate scheduler for both Pilot and Copilot models. We modify the generate in HuggingFace Transformers [22] to perform token-level logits fusion and rectified next-token generation during inference. All experiments are conducted on NVIDIA A100 GPUs. We leave all hyperparameter setups and training/inference details in Appendix E.1.
this section cite: ['b21']

Section: Models and Baselines.
We incorporate T-Copilot with varying backbone Pilot models. For encoderdecoder Pilots, we utilize T5 [65] and FLAN-T5 [17] across small/base/large variants. For decoderonly Pilots, we employ multiple models from LLaMA-3 [21] and Qwen2.5 [90] families. We denote T-Copilot-small/base/0.5B/1B/3B as the Copilot model on different scales. Detailed model configuration and implementation details are provided in Appendix E.2. We compare against three baseline types: (i) Pilot-only models as described above. (ii) Frontier LLMs with comparable and larger parameters, including LLaMA-3.1-8B [21], Gemma-2-9B [75], and Qwen2.5-14B. (iii) Layer/Adapter expansion methods, including MoE models [71] (Mistral-7B, Ministral-8B), LLaMA/Mistral-Pro-8B [84], Mergekit-9B [26], and TIES [89]. Detailed baseline descriptions are provided in Appendix E.3.
this section cite: ['b64', 'b16', 'b20', 'b89', 'b20', 'b74', 'b70', 'b83', 'b25', 'b88']

Section: Incorporating T-Copilot into Pilot Models Yields Better Performance
Effectiveness of Copilot in Enhancing Pilot. Table 2: Performance comparison (%) with baselines under matched parameter scales. Results are averaged over 3 runs. Adding T-Copilot consistently surpasses baselines of equal or even larger size. Model Params Commonsense Reasoning (Acc. ↑) Arithmetic Reasoning (Acc. ↑) PIQA WinoG. HellaS. BoolQ SIQA OBQA Avg. AQuA GSM8K MAWPS SVAMP Avg. LLaMA-3.1-8B 8B 85.4 84.3 90.9 69.6 79.9 82.6 82.1 37.3 63.5 89.1 73.6 65.9 LLaMA-3.2-3B + T-Copilot-3B 6B (-2B) 85.6 83.7 91.3 72.8 79.2 81.3 82.3 40.1 63.1 91.2 71.4 66.5 Qwen2.5-7B 7B 87.2 82.1 91.4 71.2 79.3 89.1 83.4 61.0 75.3 91.2 84.8 78.1 Qwen2.5-3B + T-Copilot-3B 6B (-1B) 87.8 81.7 94.0 68.7 79.9 89.4 83.6 59.4 76.8 92.6 83.5 78.1 Qwen2.5-14B 14B 91.8 85.6 94.3 75.2 84.5 93.1 87.4 63.5 79.5 92.4 87.9 80.8 Qwen2.5-7B + T-Copilot-3B 10B (-4B) 92.5 87.2 95.3 74.8 84.3 94.9 88.2 64.2 79.7 94.8 88.1 81.7 Comparison with Layer/Adapter Expansion Baselines Mistral-Pro-8B 8B 83.1 81.9 86.1 70.8 76.1 80.6 79.8 35.5 54.4 88.2 68.5 61.7 LLaMA-Pro-8B 8B 88.4 81.4 86.9 73.9 76.1 77.8 80.8 38.2 57.2 92.5 63.5 62.9 Ministral-8B 8B 85.7 84.1 91.3 70.3 77.5 81.3 81.7 37.4 62.9 90.2 73.2 65.9 LLaMA-3.2-3B + T-Copilot-3B 6B (-2B) 85.6 83.7 91.3 72.8 79.2 81.3 82.3 40.1 63.1 91.2 71.4 66.5 MergeKit-9B 9B 86.1 84.7 91.1 71.1 79.3 80.2 82.1 37.0 65.2 90.3 75.2 66.9 LLaMA-3.1-8B + T-Copilot-1B 9B 86.2 86.8 93.5 71.8 82.7 83.2 84.0 38.9 66.1 90.8 75.4 67.8
improves performance across all T5, LLaMA, and Qwen models on 10 commonsense and arithmetic reasoning tasks. In particular, a lightweight Copilot (e.g., T-Copilot-small) can deliver meaningful improvements (6.5% on arithmetic) when paired with a much larger Pilot model (e.g., FLAN-T5large). Moreover, scaling up the Copilot model leads to additional improvement, underscoring its effectiveness in rectifying the Pilot model's predictions during inference.
Comparison with Size-Matched Baselines. As shown in Table 2, we first compare our method against stronger models with larger parameters under the same model backbones. While LLaMA-3.2-3B initially lags significantly behind LLaMA-3.1-8B, incorporating T-Copilot-3B enables the model to outperform LLaMA-3.1-8B, despite using 2B fewer total parameters. Similarly, for the Qwen2.5 series, incorporating T-Copilot-3B enables the smaller Qwen2.5-7B to surpass Qwen2.5-14B with 4B fewer parameters. To provide a broader perspective, we also compare with strong baselines from different methods and model types. For instance, although LLaMA-3.2-3B originally trails behind models like Ministral-8B and LLaMA-Pro-8B, incorporating T-Copilot-3B enables it to outperform the strongest baseline under the 8B scale, Ministral-8B, while maintaining a 2B parameter advantage. Due to page limits, full comparison results are provided in Appendix F.1.
Downstream Tasks. Additional evaluation of T-Copilot and baseline comparisons on downstream recommendation tasks is provided in Appendix F.2.
this section cite: []

Section: Efficiency, Transferability, and Scalability
Efficiency. To thoroughly evaluate T-Copilot's running efficiency, we compare against Pilot and baseline models with the same LLaMA-3 backbone architecture under similar parameter scales. As shown in Figure 5, T-Copilot maintains comparable inference throughput (Figure 5 (a)) and training speed (Figure 5 (b)) to its corresponding Pilot models, while incurring only a 4% marginal average increase in time overhead (Figure 5 (c)). In contrast, other baselines such as LLaMA-Pro-8B and MergeKit-9B suffer from significantly higher latency and computational costs relative to their base model LLaMA-3.1-8B. We provide a more detailed inference latency report in Appendix F.3 (Table 15) and discuss the architectural advantage of our model design in Appendix A.1.
Transferability. In the T-Copilot learning framework, the Copilot model is fine-tuned alongside but separately from the Pilot model. Since the same type of models generally have similar learning trajectories under identical training settings, we further investigate if the Copilot model can leverage the mistake log of one Pilot model and still be effective on another Pilot model of the similar type.
We conduct controlled experiments on LLaMA-3 series models in which we directly apply a finetuned 1B Copilot model to new Pilot models during inference. The new Pilot model shares the same architecture as the original one but is trained independently. Note that the Copilot model does not "see" or "learn" any information from the new Pilot model, as they are not jointly trained during finetuning. In Table 3, transferring the Copilot model leads to a slight ±0.2% performance difference compared to applying the Copilot to the initial Pilot models (jointly training together). We hypothesize that the minor discrepancy is due to the hardware inference differences between the original and new Pilot models. Nonetheless, the transferred Copilot model still delivers substantial performance gains for the new Pilot and consistently outperforms competing baselines. These results demonstrate that T-Copilot's error-correction capabilities are not tightly coupled to a specific Pilot model and can be effectively transferred without additional rounds of fine-tuning.
Scalability.  Ablation Studies. Detailed ablation studies on T-Copilot, including model design choices, input insertion patterns, and the effect of the hyperparameter λ, are presented in Appendix F.4.
this section cite: []

Section: Related Works
LLMs Supervised Fine-tuning. Supervised fine-tuning (SFT) serves as the standard post-training method for specializing pre-trained LLMs to downstream tasks [80,69,94]. It enables models to incorporate task-specific knowledge and improves their performance in domain-relevant settings [94,59,96]. While effective, SFT often suffers from misalignment between training-time objectives and inference-time behavior [56,79], leading to suboptimal generalization. Recent work has explored parameter-efficient tuning methods [32,46], alongside advanced adaptation strategies [74,49,85] that improve learning effectiveness and efficiency. These methods primarily focus on model capacity and optimization rather than leveraging learning dynamics. Building upon prior SFT methods, our approach is compatible with existing fine-tuning frameworks and further improves by incorporating model-internal signals into the fine-tuning process. By adaptively learning from mistake patterns observed during fine-tuning, T-Copilot enables error-aware prediction and helps reduce the gap between training and inference performance.
this section cite: ['b79', 'b68', 'b93', 'b93', 'b58', 'b95', 'b55', 'b78', 'b31', 'b45', 'b73', 'b48', 'b84']

Section: Self-refinement in Language Models.
Recent research has explored various self-refinement techniques in LLMs to generate high-quality outputs. Models either iteratively prompt themselves with updated responses [52,27,72] or optimize their behavior using external human or synthetic feedback [58,91,54,99]. Orthogonal to external supervision such as additional prompting, multistage feedback, or explicit reward optimization, our work focuses on capturing model-internal signals during fine-tuning to achieve token-level rectification, without modifying the training objective or data distribution. We leave the additional related work and discussions in the Appendix G.
this section cite: ['b51', 'b26', 'b71', 'b57', 'b90', 'b53', 'b98']

Section: Conclusion
In this paper, we introduce Transformer Copilot, a novel learning framework that enhances Transformer-based Pilot models by integrating an auxiliary Copilot model during fine-tuning. By capturing the Pilot model's learning signals in a Mistake Log during fine-tuning, the Copilot model learns to rectify the Pilot's logits at inference time, enabling error-aware predictions. We provide both theoretical and empirical evidence that our method improves the Pilot model's inference predictions. Experiments on 12 benchmarks demonstrate the effectiveness, efficiency, scalability, and transferability of Transformer Copilot. Discussions on limitations are provided in Appendix H.
this section cite: []

Section: References
Ref_id:b0 Title: Shyamal Anadkat, et al. Gpt-4 technical report Year: (2023)
Ref_id:b1 Title: Character-level language modeling with deeper self-attention Year: (2019)
Ref_id:b2 Title: The falcon series of open language models Year: (2023)
Ref_id:b3 Title:  Year: (2016)
Ref_id:b4 Title: Neural active learning beyond bandits Year: (2024)
Ref_id:b5 Title: Ee-net: Exploitation-exploration neural networks in contextual bandits Year: (2021)
Ref_id:b6 Title: Pagerank bandits for link prediction Year: (2024)
Ref_id:b7 Title: Eliciting latent predictions from transformers with the tuned lens Year: (2023)
Ref_id:b8 Title: A comparative analysis of gradient boosting algorithms Year: (1937)
Ref_id:b9 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b10 Title: Reflective learning: Key to learning from experience Year: (1983)
Ref_id:b11 Title: Reflective learning in practice Year: (2017)
Ref_id:b12 Title: Language models are few-shot learners Year: (2020)
Ref_id:b13 Title: Discovering latent knowledge in language models without supervision Year: (2022)
Ref_id:b14 Title: Long short-term memory-networks for machine reading Year: (2016)
Ref_id:b15 Title: Palm: Scaling language modeling with pathways Year: (2023)
Ref_id:b16 Title: Scaling instruction-finetuned language models Year: (2024)
Ref_id:b17 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b18 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b19 Title: Knowledge neurons in pretrained transformers Year: (2021)
Ref_id:b20 Title: The llama 3 herd of models Year: (2024)
Ref_id:b21 Title:  Year: (2024)
Ref_id:b22 Title: Beam search strategies for neural machine translation Year: (2017)
Ref_id:b23 Title: A short introduction to boosting Year: (1999)
Ref_id:b24 Title: Transformer feed-forward layers are key-value memories Year: (2020)
Ref_id:b25 Title: Arcee's mergekit: A toolkit for merging large language models Year: (2024)
Ref_id:b26 Title: Reinforced self-training (rest) for language modeling Year: (2023)
Ref_id:b27 Title: Parameter-efficient fine-tuning for large models: A comprehensive survey Year: (2024)
Ref_id:b28 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b29 Title: Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering Year: (2016)
Ref_id:b30 Title: Reflection at work-a conceptual model and the meaning of its components in the domain of vet teachers Year: (2023)
Ref_id:b31 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b32 Title: Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models Year: (2023)
Ref_id:b33 Title: Large language models can self-improve Year: (2022)
Ref_id:b34 Title: Gpt-4o system card Year: (2024)
Ref_id:b35 Title: Neural tangent kernel: Convergence and generalization in neural networks Year: (2018)
Ref_id:b36 Title: Mistral 7b Year: (2023)
Ref_id:b37 Title: Self-attentive sequential recommendation Year: (2018)
Ref_id:b38 Title: Compacter: Efficient low-rank hypercomplex adapter layers Year: (2021)
Ref_id:b39 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2019)
Ref_id:b40 Title: Large language models are zero-shot reasoners Year: (2022)
Ref_id:b41 Title: Mawps: A math word problem repository Year: (2016)
Ref_id:b42 Title: The winograd schema challenge Year: (2012)
Ref_id:b43 Title: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension Year: (2019)
Ref_id:b44 Title: Inferencetime intervention: Eliciting truthful answers from a language model Year: (2024)
Ref_id:b45 Title: Prefix-tuning: Optimizing continuous prompts for generation Year: (2021)
Ref_id:b46 Title: When hindsight is not 20/20: Testing limits on reflective thinking in large language models Year: (2024)
Ref_id:b47 Title: Program induction by rationale generation: Learning to solve and explain algebraic word problems Year: (2017)
Ref_id:b48 Title: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b49 Title: Summary of chatgpt-related research and perspective towards the future of large language models Year: (2023)
Ref_id:b50 Title: Full parameter fine-tuning for large language models with limited resources Year: (2023)
Ref_id:b51 Title: Self-refine: Iterative refinement with self-feedback Year: (2023)
Ref_id:b52 Title: Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference Year: (2019)
Ref_id:b53 Title: Selfcheck: Using llms to zero-shot check their own step-by-step reasoning Year: (2023)
Ref_id:b54 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b55 Title: Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint Year: (2022)
Ref_id:b56 Title: Large language model (llm) ai text generation detection based on transformer deep learning algorithm Year: (2024)
Ref_id:b57 Title: Training language models to follow instructions with human feedback Year: (2022)
Ref_id:b58 Title: The ultimate guide to fine-tuning llms from basics to breakthroughs Year: (2024)
Ref_id:b59 Title: Are nlp models really able to solve simple math word problems? arXiv preprint Year: (2021)
Ref_id:b60 Title: Graph neural bandits Year: (2023)
Ref_id:b61 Title: Improving language understanding by generative pre-training Year: (2018)
Ref_id:b62 Title: Language models are unsupervised multitask learners Year: (2019)
Ref_id:b63 Title: Scaling language models: Methods, analysis & insights from training gopher Year: (2021)
Ref_id:b64 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b65 Title: Edu-larp@ chi Year: (2023)
Ref_id:b66 Title: Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models Year: (2020)
Ref_id:b67 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b68 Title: Multitask prompted training enables zero-shot task generalization Year: (2021)
Ref_id:b69 Title: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b70 Title: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer Year: (2017)
Ref_id:b71 Title: Reflexion: Language agents with verbal reinforcement learning Year: (2023)
Ref_id:b72 Title: Lora vs full fine-tuning: An illusion of equivalence Year: (2024)
Ref_id:b73 Title: Lst: Ladder side-tuning for parameter and memory efficient transfer learning Year: (2022)
Ref_id:b74 Title: Gemma 2: Improving open language models at a practical size Year: (2024)
Ref_id:b75 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b76 Title: Attention is all you need Year: (2017)
Ref_id:b77 Title: What language model architecture and pretraining objective works best for zero-shot generalization Year: (2022)
Ref_id:b78 Title: On the loss of context-awareness in general instruction fine-tuning Year: (2024)
Ref_id:b79 Title: Finetuned language models are zero-shot learners Year: (2021)
Ref_id:b80 Title: Emergent abilities of large language models Year: (2022)
Ref_id:b81 Title: Chain-of-thought prompting elicits reasoning in large language models Year: (2022)
Ref_id:b82 Title: Defending against indirect prompt injection by instruction detection Year: (2025)
Ref_id:b83 Title: Llama pro: Progressive llama with block expansion Year: (2024)
Ref_id:b84 Title: Representation finetuning for language models Year: (2024)
Ref_id:b85 Title: Parameter-efficient fine-tuning methods for pretrained language models: A critical review and assessment Year: (2023)
Ref_id:b86 Title: Neural contextual bandits with deep representation and shallow exploration Year: (2020)
Ref_id:b87 Title: Openp5: An open-source platform for developing, training, and evaluating llm-based recommender systems Year: (2023)
Ref_id:b88 Title: Ties-merging: Resolving interference when merging models Year: (2024)
Ref_id:b89 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b90 Title: React: Synergizing reasoning and acting in language models Year: ()
Ref_id:b91 Title: Demystifying reinforcement learning in agentic reasoning Year: (2025)
Ref_id:b92 Title: Hellaswag: Can a machine really finish your sentence Year: (2019)
Ref_id:b93 Title: Instruction tuning for large language models: A survey Year: (2023)
Ref_id:b94 Title: Calibrate before use: Improving few-shot performance of language models Year: (2021)
Ref_id:b95 Title: Fine-tuning large language models for domain-specific machine translation Year: (2024)
Ref_id:b96 Title: Less is more for alignment Year: (2023)
Ref_id:b97 Title: Tattoo: Tool-grounded thinking prm for test-time scaling in tabular reasoning Year: (2025)
Ref_id:b98 Title: Reasonflux-prm: Trajectory-aware prms for long chain-of-thought reasoning in llms Year: (2025)
Ref_id:b99 Title: Promptintern: Saving inference costs by internalizing recurrent prompt during large language model fine-tuning Year: (2024)
