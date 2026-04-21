Title: Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective
Abstract: In this paper, we address the challenge of determining the layer-wise sparsity rates of large language models (LLMs) through a theoretical perspective. Specifically, we identify a critical issue of "reconstruction error explosion" in existing LLMs sparsification methods. This refers to the cumulative effect of reconstruction errors throughout the sparsification process, where errors from earlier layers propagate and amplify in subsequent layers. As a result, the overall reconstruction error increases significantly, leading to a substantial degradation in model performance. Through theoretical analysis, we derive a simple yet effective approach to layer-wise sparsity allocation that mitigates this issue. Our method uses a monotonically increasing arithmetic progression, reducing the process of determining sparsity rates for multiple layers to the determination of a single common difference hyperparameter. Remarkably, this allows for the optimal layer-wise sparsity rates to be identified with just a few trials. Both our theoretical analysis and experimental results demonstrate that this sparsity allocation scheme is near optimal. Extensive experiments show that our method significantly improves the performance of sparse LLMs across various architectures, outperforming existing layer-wise sparsity methods. Furthermore, it enhances the performance of various compression techniques and is applicable to vision and multimodal models. Notably, our method achieves a reduction of 52.10 in perplexity for the 70% sparse LLaMA2-7B model obtained via Wanda, improves average zero-shot accuracy by 10.50%, and deliv-

Section: Introduction
Large Language Models (LLMs) have demonstrated outstanding capabilities in various natural language processing tasks (Meta, 2024;Yang et al., 2024;Liu et al., 2024a). However, their vast number of parameters and high computational demands present significant challenges to model deployment, impeding further applications (Zhu et al., 2024;Wang et al., 2024). Network sparsity (Rao et al., 2021;Paul et al., 2022;Huang et al., 2025b) methods remove less important parameters from LLMs, enabling model compression without sacrificing performance. This can reduce the model's memory footprint and computational complexity (Li et al., 2024a;An et al., 2024). Existing sparsity methods for LLMs, such as SparseGPT (Frantar & Alistarh, 2023) and Wanda (Sun et al., 2023), adopt a post-training approach which prune all weights in one-shot and can obtain sparse LLMs without the need for additional fine-tuning.
However, these sparsity methods set a uniform layer-wise sparsity rate for different layers, without considering the varying importance of each layer, which harms the accuracy of sparse LLMs. To address the above issue, many studies have proposed various methods to determine the layer-wise sparsity rate of LLMs. Based on their methodological designs, we categorize these methods into two main groups: Metric based methods. These methods determine the importance of each layer of LLMs through hand-crafted metrics, thereby obtaining the sparsity rate of each layer. For example, OWL (Yin et al., 2023) proposes an outlier weighted layer importance metric. By setting the sparsity rate to be inversely proportional to the outlier ratio, it effectively protects the layers with a higher ratio of outliers. AlphaPruning (Lu et al., 2024) utilizes the heavy-tailed self-regularization theory (Martin & Mahoney, 2019), especially the shape of the empirical spectral density (Martin et al., 2021) of the weight matrix, to determine the importance of each layer of LLMs. ALS (Li et al., 2024c) proposes an importance metric based on mutual information (Tschannen et al., 2019), presents a comparison between our method and other layer-wise sparsity methods. The metric-based method calculates the importance of each layer to obtain the sparsity rate. However, this method is heuristically designed by human experts and is not optimal. And the search-based method requires a large number of iterative searches, which is time-consuming. In contrast, we analyze the causes of "reconstruction error explosion" from a theoretical perspective, and deduce theoretically that using a monotonically increasing arithmetic progression to determine the layer-wise sparsity rate can alleviate the problem of "reconstruction error explosion".
and sets a higher sparsity rate for layers with higher mutual information. Although these metrics have proven their effectiveness experimentally, manually designing metric requires extensive validation and complex calculations are needed to obtain the sparsity rate of each layer. Most importantly, most of these methods lack theoretical analysis, making it impossible to ensure that the solutions obtained are optimal.
Search based methods. In addition to these heuristics designed by humans, recently, there have also been some methods that adopt a search-based approach to determine the layer-wise sparsity rate of LLMs. For example, DSA (Li et al., 2024b) develops an expression discovery framework to explore potential sparsity rate allocation strategies and obtains layer-wise sparsity allocation function by searching. However, the evolutionary search method employed by DSA requires 2000 iterations. For large-scale LLMs with a vast number of parameters, this demands a search process lasting several days, which incurs a significant cost.
In addition, in order to obtain the final allocation function, DSA designs a complex search space and process, which means the effectiveness of the method heavily depends on the experience of human experts.
In this paper, we rethink the approach of determining the layer-wise sparsity rate of LLMs, and derive the layer-wise sparsity rate of LLMs from reconstruction error perspective. Specifically, we first prove that increasing the sparsity rate leads to an increase in the reconstruction error of the corresponding layer. Additionally, we show that an increase in the reconstruction error of one layer causes an increase in the reconstruction error of subsequent layers. This implies that increasing the sparsity rate of earlier layers not only increases the reconstruction error of the corresponding layer, but also leads to an increase in the reconstruction errors of all subsequent layers. As the network propagates forward, the reconstruction errors accumulate, causing the total reconstruction error to grow significantly, thus causing "reconstruction error explosion" (See the left in Figure 1).
Through the above theoretical analysis, we provide a simple yet effective rule for determining the layer-wise sparsity rates of LLMs: the sparsity rate should be lower in earlier layers, and the layer-wise sparsity rates should follow a increasing pattern. This approach effectively alleviates the issue of "reconstruction error explosion", resulting in a well-performing sparse LLM. To achieve this, we use a monotonically increasing arithmetic progression to determine the sparsity rates for all layers of LLMs and employ grid search to find the common difference of the arithmetic progression. Since the range of valid values for the common difference is narrow, our search is highly efficient, and after only a few attempts, we can determine the common difference that yields the best accuracy.
Furthermore, we prove that the total reconstruction error obtained from the monotonically increasing sparsity scheme is less than that of any non-monotonically increasing sparsity scheme. This indicates that our method is theoretically close to the optimal solution. Additionally, we compare our sparsity rate scheme with the optimal solution obtained through Bayesian search, we find that our scheme is close to the optimal solution found by search. This indicates that our method is empirically close to the optimal solution.
To evaluate the effectiveness of our ATP 1 method, we con-1 Determining layer-wise sparsity for LLMs through A Theoretical Perspective (ATP).
duct extensive experiments on LLMs of various architectures, with parameter counts ranging from 6.7 billion to 70 billion. Our evaluation metrics include perplexity, average accuracy across seven zero-shot datasets, and performance on arithmetic and knowledge reasoning tasks. Our ATP method demonstrate substantial improvements over existing post-training sparsity techniques, significantly surpassing other layer-wise sparsity methods. Notably, ATP reduce the perplexity of the 70% sparse LLaMA2-7B pruned using Wanda (Sun et al., 2023) by 52.10 and increase average zeroshot accuracy by 10.50%, outperforming the state-of-the-art AlphaPruning method (Lu et al., 2024) by 6.71 and 2.46%, respectively. Additionally, ATP achieved 2.63× and 2.23× speedups on CPU and GPU, respectively, and require only 18 minutes to compute layer-wise sparsity rates. Furthermore, we evaluate ATP's enhancements on various compression techniques, including N:M sparsity, structured pruning, and network quantization, as well as its benefits for sparse multimodal and sparse vision models. These experimental results clearly demonstrate that ATP provides substantial performance improvements for compressed models.
this section cite: ['b66', 'b76', 'b64', 'b54', 'b51', 'b1', 'b14', 'b58', 'b67', 'b38', 'b42', 'b43', 'b62', 'b58', 'b38']

Section: Related Work
LLMs Sparsity. Before the advent of LLM, a variety of sparsity techniques had been developed to compress models such as ResNet (Yu et al., 2022b;Zhang et al., 2024) , BERT (Xia et al., 2022;Li et al., 2023), and ViT (Yu et al., 2022a;He et al., 2024). Meanwhile, researchers have developed several post-training sparsity methods specifically for LLMs. For example, SparseGPT (Frantar & Alistarh, 2023) uses the inverse of the Hessian matrix for pruning and pruned weight updates. Wanda (Sun et al., 2023) uses a metric that combines weight magnitude and input activation to prune LLMs, while Pruner-zero (Dong et al., 2024) searches for symbolic pruning metric using genetic programming. Additionally, ALPS (Meng et al., 2024) uses an Alternating Direction Method of Multiplier (ADMM) (Boyd et al., 2011)-based approach to prune LLMs in one-shot. The above methods focus on determining the mask within the layer of LLMs and setting a uniform layer-wise sparsity rate. Our work study the layer-wise sparsity allocation problem in sparse LLMs from the perspective of reconstruction error, thereby effectively improving the accuracy of above methods.
Layer-wise Sparsity. Layer-wise sparsity rate determines the number of weights to be retained in each layer of the network (Lee et al., 2020;Frankle et al., 2020;Liu et al., 2022a;Huang et al., 2025a). To determine the layer-wise sparsity rate in LLMs, OWL (Yin et al., 2023) proposes an outlier-weighted metric, Alphapruning (Lu et al., 2024) uses heavy-tailed self-regularization theory (Martin & Mahoney, 2019), ALS (Li et al., 2024c) proposes a layer redundancy metric based on mutual information (Kraskov et al., 2004), DSA (Li et al., 2024b) develops an expression discovery algorithm to explore potential sparsity allocation. However, the above metric and search based methods all lack theoretical proof of effectiveness and require complex calculations or search to obtain layer-wise sparsity rate. In comparison, our method directly derives the layer-wise sparsity rates based on a monotonically increasing arithmetic progression from the perspective of reconstruction error, requiring only the determination of the common difference to quickly obtain the sparsity rate for each layer. More importantly, we have validated the effectiveness of the above method through rigorous proof.
Reconstruction Error. Reconstruction error is a metric for measuring the difference between the output of a compressed network and that of the original network. A smaller reconstruction error generally implies that the compressed network can better preserve the performance of the original network (Yun & Wong, 2021;Hubara et al., 2021;Ma et al., 2023b). In order to minimize the reconstruction error of sparse LLMs, SparseGPT (Frantar & Alistarh, 2023) proposes a mask selection and weight update algorithm based on Hessian inverse and DSnoT (Zhang et al., 2023) proposes a training-free dynamic weight pruning and growing algorithm. In this paper, we explore the "reconstruction error explosion" problem in sparse LLMs. Specifically, the reconstruction error accumulates and magnifies across layers, leading to an extremely large overall reconstruction error, which undermines the model's accuracy. Therefore, we propose our layer-wise sparsity allocation method to alleviate the above "reconstruction error explosion" problem.
this section cite: ['b72', 'b65', 'b33', 'b17', 'b14', 'b58', 'b10', 'b44', 'b3', 'b28', 'b13', 'b67', 'b38', 'b42', 'b27', 'b70', 'b23', 'b14', 'b74']

Section: Methodology
Notation. In this paper, we use bold typeface indicates matrices (e.g.,W, X) and calligraphic font represents loss functions or models (e.g.,L, M).
this section cite: []

Section: Preliminaries
Without loss of generality, we take each layer in the LLMs as the basic unit of analysis. These layers contain modules such as Attention (Vaswani et al., 2017), MLP (Popescu et al., 2009), LayerNorm (Lei Ba et al., 2016), and residual connections (He et al., 2016), etc. We represent a layer's computation as W X, where W is the layer's weight and X is the input. Consider an LLM composed of L layers, we define the reconstruction error of the i-th layer (i = 1, 2, • • • , L) as follows:
L(W i , X i ) = W i X i -W i X i 2 F (1)
where W i ∈ R cout×cin and X i ∈ R cin×d are the weights and input of the i-th layer respectively. W i and X i are the corresponding sparse versions, where c in and c out represent the number of input and output feature dimensions, d is the hidden dimension, and ∥•∥ F is Frobenius norm.
Existing post-training sparsity methods all attempt to minimize the reconstruction error of sparse LLMs (e.g., SparseGPT and Wanda) and a large amount of experimental evidence in the these papers indicates that a sparse LLM with good accuracy often has a low reconstruction error.
In the next section, we reveal that the existing post-training sparsity methods all have the problem of "reconstruction error explosion".
this section cite: ['b63', 'b53', 'b29', 'b18']

Section: "Reconstruction Error Explosion" in Sparse LLMs
First, we analyze the relationship between the sparsity rate and the reconstruction error. Briefly, a higher sparsity rate leads to a higher reconstruction error. Formally, we propose Theorem 3.1. Theorem 3.1 (Effect of increased sparsity on reconstruction error). When the input is the same, increasing the sparsity of the weights in the i-th layer will lead to an increase in the reconstruction error of this layer.
Proof of Theorem 3.1. We only consider the effect of sparse weights on the reconstruction error, and we ignore the error caused by the input at this time. Therefore, we consider the impact of the sparse weights of the i-th layer on the reconstruction error when the input is the same. That is, X i = X i . Then, the reconstruction error of the i-th layer is expressed as:
L = W i X i -W i X i 2 F (2)
Consider two weights W (1) i and W
(2) i of different sparsity, such that W (1) i has lower sparsity (i.e., has fewer zero elements). The difference in reconstruction error corresponding to these two sparse weights is:
L (1) -L (2) = W i X i -W (1) i X i 2 F -W i X i -W (2) i X i 2 F = W i X i -W (1) i X i 2 F -(W i X i -W (1) i X i ) + ( W (1) i X i -W (2) i X i ) 2 F = -2⟨(W i -W (1) i )X i , ( W (1) i -W (2) i )X i ⟩ F -W (1) i X i -W (2) i X i 2 F (3)
The first term of the inner product, (W i -W (2) i )X i are generated through the same sparsification method, they point in similar directions within the vector space. This alignment ensures that their Frobenius inner product satisfies
⟨(W i -W (1) i )X i , ( W (1) i -W (2) i )X i ⟩ F > 0. And since W (1) i X i -W (2) i X i 2 F > 0, therefore: L (1) -L (2) < 0(4)
Therefore, the reconstruction error L (1) for lower sparsity is smaller than L (2) for higher sparsity. In other words, increasing the sparsity will lead to an increase in the reconstruction error of this layer. So, we have completed the proof of the above theorem.
On the other hand, we find that there is an cumulative effect of reconstruction error during the sparsification process. It is manifested as the reconstruction error in LLMs showing an increasing trend due to the influence of previous sparse layers. In other words, if the reconstruction error of the previous layer increases, the reconstruction error of the subsequent layers will also increase accordingly. Formally, we propose Theorem 3.2.
Theorem 3.2 (The cumulative effect of reconstruction error). When the reconstruction error of the i-th layer increases, it leads to an increase in the lower bound of the reconstruction error for the (i + 1)-th layer.
To prove Theorem 3.2, we need to define the following lemma:
Lemma 3.3. Let A ∈ R m×n and B ∈ R n×p be arbitrary matrices. Then, it holds that ∥AB∥ The proof of the Lemma 3.3 can be found in Appendix B. Now we formally prove Theorem 3.2.
Proof of Theorem 3.2. The reconstruction error for the (i + 1)-th layer can be expressed as:
L(W i+1 , X i+1 ) = W i+1 X i+1 -W i+1 X i+1 2 F = (W i+1 -W i+1 )X i+1 + W i+1 (X i+1 -X i+1 ) 2 F = (W i+1 -W i+1 )X i+1 2 F + W i+1 (X i+1 -X i+1 ) 2 F + 2tr((W i+1 -W i+1 )X i+1 ) ⊤ ( W i+1 (X i+1 -X i+1 ))
(5) Since the third term is the trace of the product of two matrices, the first and second term is the square of the Frobenius norm, and considering that the weights and inputs in LLMs are matrices with large dimensions, the magnitude of the first and second term is much greater than the third term, so we ignore the third term and get:
L(W i+1 , X i+1 ) ≈ (W i+1 -W i+1 )X i+1 2 F + W i+1 (X i+1 -X i+1 ) 2 F (6) Since (W i+1 -W i+1 )X i+1 2 F > 0. Therefore: L(W i+1 , X i+1 ) > W i+1 (X i+1 -X i+1 ) 2 F (7)
According to Lemma 3.3, we get:
L(W i+1 , X i+1 ) > σ 2 min ( W i+1 ) (X i+1 -X i+1 ) 2 F = σ 2 min ( W i+1 ) (W i X i -W i X i ) 2 F = σ 2 min ( W i+1 )L(W i , X i ) (8) Since σ 2 min ( W i+1
) > 0, we have proven that the increase of the reconstruction error of the i-th layer will lead to the increase of the lower bound of the reconstruction error of the (i + 1)-th layer.
Theorem 3.2 shows that an increase in the reconstruction error of the previous layer in a sparse LLM usually leads to a further increase in the lower bound of the reconstruction error of the subsequent layer. In practice, this often means that an increase in the reconstruction error of the previous layer will lead to an increase in the reconstruction error of the subsequent layer. We have also observed this phenomenon in the left of Figure 1. We can see that when the reconstruction error of the earlier layers is smaller, the reconstruction error of the subsequent layers is also smaller. Conversely, when the reconstruction error of the earlier layers is larger, the reconstruction error of the subsequent layers is also larger.
According to Theorems 3.1 and 3.2, we can easily get the following Theorem 3.4: Theorem 3.4 (Impact of the sparsity of the previous layer on the reconstruction error of the next layer.). Increasing the sparsity of the i-th layer will lead to an increase in the lower bound of the reconstruction error of the (i + 1)-th layer.
Proof of Theorem 3.4. According to Theorems 3.1 and 3.2, we have the following relationship:
sparsity of layer i ↑=⇒ L(W i , X i ) ↑ =⇒ L(W i+1 , X i+1 ) > σ 2 min ( W i+1 )L(W i , X i ) ↑ .(9)
Therefore, we complete the proof of Theorem 3.4.
To summarize all the above, we can get the following logical chain of our paper: sparsity rate of 1-st layer ∝ reconstruction error of 1-st layer ∝ reconstruction error of L-th layer ∝ total error ∝ accuracy loss. That is, when the sparsity rate of the 1-st layer increases, it will lead to an increase in the reconstruction error of this layer. As the reconstruction error accumulates continuously from the 1-st layer to the L-th layer, the reconstruction error of the L-th layer also increases accordingly. Then, the reconstruction errors of all layers of the model will exhibit an explosion phenomenon, resulting in a serious decline in the accuracy of the sparse model. We refer to the above phenomenon as the "reconstruction error explosion". This phenomenon can be observed on the left side of Figure 1.
From the above theoretical analysis, we understand that the earlier layers are more important than the later layers. Setting a lower sparsity rate for the previous layers helps alleviate the problem of "reconstruction error explosion". In the next section, we will introduce our method of determining the layer-wise sparsity rate in detail.
this section cite: []

Section: Discussion about the Rationality of Theoretical Modeling
In Sec. 3.1, we represent a Transformer layer's computation as W X. The Transformer layer includes components such as Attention, MLP, nonlinearities, and layer normalization. Due to the more complex nonlinear calculations in a Transformer layer, there are differences between theoretical analysis based on W X and the actual architecture. However, our analysis remains reasonable for the following reasons:
1. The theoretical modeling of W X is sufficient to analyze the layer's reconstruction error. Our method sparsifies the linear layers in Attention and MLP modules, while other components remain unaffected. These linear layers account for the majority of the parameter count and significantly influence the computation results of the layer. The sparsified linear layers dominate the computation of the reconstruction error for each layer. Although various nonlinear operations exist in the actual architecture, they typically do not fundamentally alter the reconstruction error of each layer and have minimal impact on theoretical analysis. Therefore, modeling the primary computation of a layer as W X is sufficient for analyzing the reconstruction error of that layer. This is also sufficient for us to analyze how reconstruction errors accumulate and propagate across the network.
2. Transformer's linear modeling is supported by existing research. Razzhigaev et al. (Razzhigaev et al., 2024) employs Procrustes similarity analysis to discover that the embedding transformations between sequential layers in LLMs such as GPT, LLaMA, OPT, and BLOOM exhibit a near-perfect linear relationship, with linear score is 0.99. This indicates that despite the non-linear operations within Transformer layers, the mapping between adjacent layers can still be approximated as a linear transformation. Therefore, it reasonable and natural to model Transformer layer computations using W X.
this section cite: ['b55']

Section: 3.
Modeling the computation of modules as W X is a common practice in many works. AdaRound (Nagel et al., 2020) and MRECG (Ma et al., 2023b) simplify the computation of the CONV+BatchNorm+RELU modules in quantized convolutional neural networks as W X when analyzing reconstruction error. This approach of ignoring unnecessary computations and focusing on the core computations is a common practice, which facilitates the derivation of theoretical results.
this section cite: ['b49']

Section: Determining Layer-wise Sparsity Rates for LLMs
According to the theorem in Sec. 3.2, the reconstruction error in sparse LLMs have "reconstruction error explosion" problem. Specifically, the error from the earlier layers will cause an increase in the error of the later layers. When the error from all earlier layers are accumulated, it will lead to a sharp increase in the error of the later layers. Meanwhile, this can lead to an increase in the total reconstruction error of sparse LLMs, thereby damaging the final accuracy of the sparse LLMs. Therefore, in order to mitigate the negative impact of the "reconstruction error explosion" of reconstruction errors on sparse LLMs, we can set the earlier layers to have lower sparsity and the later layers to have higher sparsity. Therefore, we propose to determine the sparsity rate of each layer in LLMs according to the following monotonically increasing arithmetic progression:
s i = S - β(L -1) 2 + β × (i -1), i = 1, 2, . . . , L (10)
where s i is the sparsity rate (fraction of zero entries) of the i-th layer, L is the total layer number of LLMs, and S is the average sparsity rate of all layers. β is a hyperparameter that controls the degree of difference in the sparsity rate of each layer of LLMs. The above formula means that we only need to determine the hyperparameter β to get the sparsity rate of each layer of LLMs.
We use grid search (Jiménez et al., 2008) to determine β for sparse LLMs. Specifically, since 0 ≤ s 0 , s L ≤ 1 and considering the relatively low sparsity rate set for the earlier layers, the arithmetic progression should be increasing. Therefore, we can deduce that the possible range of values for β is 0 < β ≤ min( 2S L-1 , 2(1-S) L-1 ). This is a small range for a sparse LLMs. For example, for a LLaMA3-8B (Meta, 2024) model with 32 layers and the average sparsity rate is S = 0.7, then the range is 0 < β ≤ 0.019. In order to find the optimal value of β within the above range, we adopt an grid search method with a step size of 0.002. The goal is to find the value of β that minimizes perplexity of the sparse LLMs on the WikiText-2 (Merity et al., 2016) dataset. Since the reasonable range of β is very small, this ensures that we can find the optimal β very quickly. For example, for 0 < β ≤ 0.019, we only need to make 9 attempts. Even for the largest 70B model, the reasonable range of β is 0 < β ≤ 0.0075, and only 3 attempts are required in this case. We present our ATP approach in Algorithm 1.
Algorithm 1 Using ATP method to obtain sparse LLMs Input: Dense LLMs M dense , average sparsity rate S Output: Sparse LLMs M sparse Use grid search to get β in Eq. 10; Determine layer-wise sparsity rate according to Eq. 10; Combine with post-training sparsity method (e.g., SpaeseGPT or Wanda) to obtain sparsity mask W; Apply W to M dense yields M sparse .
Although the above method of determining the layer-wise sparsity rate of LLMs by a monotonically increasing arithmetic progression is very simple, our theoretical analysis in Sec. 3.5 fully proves its rationality, and we have fully demonstrated through a large number of experiments in Sec. 4 that our method can effectively improve the accuracy of existing post-training sparsity methods and significantly outperforms current layer-wise sparsity methods.
this section cite: ['b45']

Section: Analysis of the Proposed Determining Sparsity Method
We propose the following theorem to prove that the method for determining the layer-wise sparsity rate proposed in Eq. 10 is theoretically close to the optimal solution:
Theorem 3.5. The total reconstruction error obtained from the monotonically increasing sparsity scheme proposed in Eq. 10 is strictly less than that obtained from any nonmonotonically increasing sparsity scheme.
The proof of the above theorem is detailed in Appendix C.
In addition, we compare the layer-wise sparsity rates determined by Eq. 10 with those obtained by Bayesian search in Sec. 4.6. The experimental results show that our method is All in all, the above theoretical analysis and experimental validation demonstrate the superiority of our method, indicating that the sparsity allocation scheme proposed in Eq. 10 is near-optimal both theoretically and empirically.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Models. We evaluate ATP across a diverse range of widely-used LLMs, including LLaMA-1 (7B, 13B, 30B and 65B) (Touvron et al., 2023a), LLaMA-2 (7B, 13B and 70B) (Touvron et al., 2023b), LLaMA-3-8B (Meta, 2024), LLaMA-3.1-8B (Meta, 2024), LLaMA-3.2-1B and 3B (Meta, 2024), OPT-13B (Zhang et al., 2022), Vicuna-13B (Chiang et al., 2023), Qwen2.5-7B (Yang et al., 2024), and Mistral-7B (Jiang et al., 2023) and Mixtral-8x7B (Jiang et al., 2024).
Evaluation. Our evaluation protocol aligns with established sparsification methods for LLMs (e.g.,, SparseGPT (Frantar & Alistarh, 2023) and Wanda (Sun et al., 2023)), encompassing both zero-shot learning and language modeling capabilities. Specifically, we assess the perplexity of the models on the validation set of WikiText-2 (Merity et al., 2016) and evaluate zero-shot performance on seven downstream tasks: BoolQ (Clark et al., 2019), ARC Easy and Challenge (Clark et al., 2018), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), OpenbookQA (Mihaylov et al., 2018), and PIQA (Bisk et al., 2020). Additionally, we measure performance on arithmetic and knowledge reasoning benchmarks, including 8-shot accuracy on the GSM8K dataset (Cobbe et al., 2021) and 5-shot accuracy on the MMLU dataset (Hendrycks et al., 2020).
Baselines. We apply the layer-wise sparsity rates determined by ATP to several state-of-the-art post-training sparsification methods, including SparseGPT (Frantar & Alistarh, 2023), Wanda (Sun et al., 2023), DSnoT (Zhang et al., 2023), Pruner-zero (Dong et al., 2024), and ALPS (Meng et al., 2024). Furthermore, we compare ATP with recent methods for determining layer-wise sparsity rates in LLMs, such as OWL (Yin et al., 2023), AlphaPruning (Lu et al., 2024), DSA (Li et al., 2024b), and ALS (Li et al., 2024c).
More Models, Evaluations and Baselines. In Sec. F, we present additional experimental results, including the application of ATP to multimodal and vision models, integration with other compression techniques, LoRA fine-tuning and sparsity-preserving PEFT methods, and comparisons with an expanded set of layer-wise sparsity baselines.
Implementation Details. Our pruning implementation builds upon the methods used by SparseGPT and Wanda, with the primary modification being the integration of layerwise sparsity rates generated by ATP.
this section cite: ['b73', 'b4', 'b66', 'b24', 'b25', 'b14', 'b58', 'b45', 'b5', 'b6', 'b71', 'b56', 'b47', 'b2', 'b7', 'b19', 'b14', 'b58', 'b74', 'b10', 'b44', 'b67', 'b38']

Section: Zero-shot Tasks
Quantitative Evaluation. We report the average performance of 70% sparse LLMs across seven zero-shot tasks in Table 1. The results demonstrate that our ATP method consistently improves accuracy compared to the uniform sparsity baseline and significantly outperforms other state-ofthe-art layer-wise sparsity methods. For instance, with the LLaMA3-8B model pruned using the Wanda method, ATP achieves a 3.43% higher accuracy than the best-performing DSA method, highlighting the effectiveness and superiority of ATP in enhancing sparse LLM performance.
this section cite: []

Section: Varying Sparsity Rates.
We also evaluate the performance of sparse LLMs under reduced sparsity constraints.
Specifically, Table 2 presents the zero-shot accuracy of LLMs pruned using the Wanda method at a 50% sparsity rate. Even under this lower sparsity setting, ATP demonstrates substantial improvements in accuracy across all models, maintaining its performance advantage over existing layer-wise sparsity methods. This indicates that ATP is robust and effective in optimizing LLM accuracy under varying sparsity rates.
this section cite: []

Section: Language Modeling
We present the perplexity of 50% to 80% sparse LLaMA-7B and LLaMA2-7B models pruned using the Wanda method on the WikiText-2 dataset in Table 3. The results show that our ATP method achieves lower perplexity compared to other layer-wise sparsity methods. Furthermore, this advantage becomes increasingly significant at higher sparsity rates.
this section cite: []

Section: More LLM Architectures
We validate the effectiveness of our ATP method on LLMs with a broader range of architectures. Specifically, we use ATP method to obtain 70% sparse models, including LLaMA3.1-8B, LLaMA3.2-1B/3B, OPT-13B, Vicuna-13B, Qwen2.5-7B, Mistral-7B, and Mixtral-8x7B. These models cover a wide range of parameter sizes, from 1B to 46.7B, including both LLaMA-like architectures and sparsely ac-tivated Mixture of Experts (MoE) models. We report the experimental results in Table 4. The results demonstrate that our ATP method consistently enhances the performance of LLMs across various architectures, further demonstrating its generalizability across different model architectures.
this section cite: []

Section: More Results
We provide more experimental results in the appendix. Specifically, in Sec. F.1, we demonstrate the enhanced performance of our ATP method on arithmetic and knowledge reasoning tasks for sparse LLMs. Additionally, in Secs. F.2 and F.3, we present the performance gains of ATP on sparse multimodal and vision models, respectively. In Sec. F.4, we highlight the performance improvements when integrating ATP with other compression techniques, including N:M sparsity, structured pruning, and quantization. Furthermore, we compare ATP with additional layer-wise sparsity baselines (Sec. F.5), demonstrate its performance enhancements across various post-training sparsity methods (Sec. F.6), and showcase its effectiveness when combined with LoRA fine-tuning for sparse LLMs (Sec. F.7). Furthermore, we show the zero-shot accuracy of LLMs at 60% sparsity in Sec. F.8.
this section cite: []

Section: Ablation Study Searching Step.
In Sec. 3.4, we perform a grid search with a step size of 0.002 to determine the optimal value of β. Here, we analyze the impact of different step sizes on the search results. As shown in Table 5, searches conducted with larger step sizes yield inferior results compared to those with a step size of 0.002. This is because larger step sizes fail to sufficiently explore the possible optimal values of β. Conversely, further reducing the step size for a more finegrained search shows limited improvement in perplexity. Therefore, to balance both accuracy and efficiency, we adopt a grid search with a step size of 0.002.  6. The results indicate that the performance of our ATP method is comparable to the optimal solution obtained through Bayesian search, demonstrating that the layer-wise sparsity rates determined by our method are experimentally close to the optimal values identified by the search approach. However, Bayesian search requires approximately 33 hours to complete on a single NVIDIA A100 80GB GPU, whereas ATP only takes 18 minutes, demonstrating significantly higher efficiency. Inference Speedup. We evaluate the acceleration performance of the sparse LLaMA2-7B model, with results summarized in Table 7. The end-to-end token generation time was measured using the DeepSparse (NeuralMagic, 2021) inference engine on an Intel Xeon Silver 4314 CPU and the nm-vllm (NeuralMagic, 2024) inference engine on an NVIDIA RTX 4090 GPU. Our method achieves significant speedups, ranging from 1.79× to 2.63× on the CPU and 1.71× to 2.23× on the GPU, compared to the dense model, at sparsity rates between 50% and 70%. In addition, there is a discussion about which devices can be used with the DeepSparse and nm-vllm inference engines to deploy sparse LLMs for inference acceleration, please refer to Sec. E.
this section cite: []

Section: Conclusion
In this paper, we propose a theoretically grounded method for determining layer-wise sparsity rates in LLMs, effectively addressing the challenge of "reconstruction error explosion". Our approach utilizes an arithmetic progression to streamline sparsity allocation, reducing the complexity to a single hyperparameter while achieving near-optimal performance with minimal tuning. We have demonstrated through theoretical analysis and experimental validation that our sparsity allocation scheme is close to the optimal solution. Extensive experiments demonstrate the effectiveness of our method, yielding significant improvements in model perplexity, accuracy, and inference speed. Furthermore, our approach exhibits strong generalization across diverse architectures and modalities, establishing it as a versatile and robust solution for optimizing compressed models.
this section cite: []

Section: A. Limitation and Future Work
In this paper, we derive that layer-wise sparsity rates should gradually increase based on the reconstruction error analysis and determine these rates using a monotonically increasing arithmetic progression. Extensive experiments validate the effectiveness of our method. However, the arithmetic progression configuration may not be optimal, and we plan to explore more diverse sparsity rate schemes in the future. Additionally, while our method significantly enhances the accuracy of existing post-training sparsity techniques for LLMs, there remains a performance gap compared to lossless sparsification, particularly under high sparsity conditions. In future work, we aim to develop more advanced post-training sparsity methods to further improve the accuracy of sparse LLMs.
this section cite: []

Section: B. Proof of Lemma 3.3
Proof of Lemma 3.3. Let A ∈ R m×n and B ∈ R n×p be any matrices. Consider the singular value decomposition (SVD) of A:
A = U ΣV T (11
)
where:
• U ∈ R m×m is an orthogonal matrix (U T U = I),
• V ∈ R n×n is an orthogonal matrix (V T V = I), and
• Σ ∈ R m×n is a diagonal matrix with non-negative singular values σ 1 ≥ σ 2 ≥ • • • ≥ σ min(m,n) ≥ 0 on the diagonal.
Substituting the SVD of A into the expression for AB, we have:
AB = U ΣV T B (12
)
Therefore,
∥AB∥ 2 F = U ΣV T B 2 F (13
)
Since U is an orthogonal matrix, the Frobenius norm is invariant under orthogonal transformations. Specifically:
∥U X∥ F = ∥X∥ F ∀ X (14
)
Applying this property:
∥AB∥ 2 F = ΣV T B 2 F (15
)
The matrix Σ is diagonal with singular values σ i on the diagonal. Let σ min = σ min (A) denote the smallest singular value of A. Then, for any matrix X, we have:
ΣX ≥ σ min X (16
)
in the sense that each singular value scales the corresponding component of X.
Therefore, applying this to our case:
ΣV T B 2 F ≥ σ 2 min V T B 2 F (17
)
This inequality holds because scaling each component by at least σ min results in the squared norm being scaled by at least σ 2 min . Similarly to U , the orthogonal matrix V preserves the Frobenius norm:
V T X F = ∥X∥ F ∀ X (18
)
Applying this property:
V T B F = ∥B∥ F (19
)
Substituting back, we obtain:
∥AB∥ 2 F = ΣV T B 2 F ≥ σ 2 min V T B 2 F = σ 2 min ∥B∥ 2 F (20
)
Thus, we have shown that:
∥AB∥ 2 F ≥ σ 2 min (A) ∥B∥ 2 F(21)
C. Proof of Theorem 3.5
Proof of Theorem 3.5. Recall that each layer i ∈ {1, 2, . . . , L} in an LLM has a sparsity rate s i ∈ [0, 1], where s i indicates the fraction of zero entries in the weight matrix of layer i. Let L i denote the reconstruction error of layer i, and let L = L i=1 L i be the total reconstruction error of the entire LLM. To prove Theorem 3.5, we will make the following assumptions based on the preceding sections:
1. According to Theorem 3.1, The reconstruction error for each layer i, denoted as L i , is an increasing function of the sparsity rate s i , denote as f (s i ).
2. According to Theorem 3.2, the reconstruction error propagates through layers such that the increase of reconstruction error of i-th layer will lead to the increase of reconstruction error of (i + 1)-th layer. Therefore we assume L i+1 = cL i + f (s i ), where c > 1. The above formula indicates that the reconstruction error of a layer not only accumulates the reconstruction errors from previous layers but also that the current sparse layer contributes to the reconstruction error.
The monotonically increasing sparsity scheme given by Eq. 10 (in Sec. 3.4) implies that s 1 < s 2 < • • • < s L , let us denote such a scheme as
s ↑ = (s ↑ 1 , s ↑ 2 , . . . , s ↑ L ),(22)
where
s ↑ 1 < s ↑ 2 < • • • < s ↑ L .
Consider a different non-monotonically increasing sparsity scheme:
s ⋄ = (s ⋄ 1 , s ⋄ 2 , . . . , s ⋄ L ),(23)
which is not monotonically increasing. That is, there exists at least one index k such that
s ⋄ k > s ⋄ k+1
Moreover, suppose both s ↑ and s ⋄ satisfy the constraint that their average sparsities match the same overall budget. Formally,
1 L L i=1 s ↑ i = 1 L L i=1 s ⋄ i = S,(24)
and each s ↑ and s ⋄ results in a total reconstruction error
L ↑ = L i=1 L ↑ i , L ⋄ = L i=1 L ⋄ i .(25)
We will show that for any non-monotonic sparsity vector s ⋄ , we can redundantly reorder it layer by layer to get a monotonically increasing vector s ↑ of the same average sparsity, such that the overall reconstruction error L ↑ is strictly smaller than L ⋄ . Ultimately, we will deduce
L ↑ < L ⋄ .(26)
Let us focus on any adjacent pair
(s ⋄ k , s ⋄ k+1 ) where s ⋄ k > s ⋄ k+1 . Define s * k = s ⋄
k+1 and s * k+1 = s ⋄ k , effectively swapping these two sparsities. We then compare the total contribution of layers k and k + 1 to the overall reconstruction error, first in the non-monotonic case and then in the swapped case.
We restrict our attention to layers k and k + 1:
Layer k ⇒ L ⋄ k = f (s ⋄ k ), Layer k + 1 ⇒ L ⋄ k+1 = c L ⋄ k + f (s ⋄ k+1 ).(27)
Strictly speaking, L ⋄ k+1 depends on partial errors from layer k and its own sparsity s ⋄ k+1 . Intuitively, a higher error L ⋄ k "propagates" or "magnifies" into layer k + 1. We keep the rest of the layer-wise sparsities fixed outside of these two positions to isolate their effect.
Now, consider swapping the two sparsities:
(s ⋄ k , s ⋄ k+1 ) → (s * k , s * k+1 ) = (s ⋄ k+1 , s ⋄ k ). Let L * k = f (s * k ) = f (s ⋄ k+1 ), L * k+1 = c L * k + f (s * k+1 ) = c f (s ⋄ k+1 ) + f (s ⋄ k ).(28)
Therefore, the total for these two layers is
L swapping = L * k + L * k+1 = f (s ⋄ k+1 ) + c f (s ⋄ k+1 ) + f (s ⋄ k ) = (1 + c) f (s ⋄ k+1 ) + f (s ⋄ k ).(29)
Meanwhile, the original total is
L original = L ⋄ k + L ⋄ k+1 = f (s ⋄ k ) + c f (s ⋄ k ) + f (s ⋄ k+1 ) = (1 + c) f (s ⋄ k ) + f (s ⋄ k+1 ).(30)
The difference between the two is
L original -L swapping = (1 + c) f (s ⋄ k ) + f (s ⋄ k+1 ) -(1 + c) f (s ⋄ k+1 ) + f (s ⋄ k ) = (1 + c) f (s ⋄ k ) -f (s ⋄ k+1 ) -f (s ⋄ k ) -f (s ⋄ k+1 ) = c f (s ⋄ k ) -f (s ⋄ k+1 ) .(31)
Since c > 1 and f (s ⋄ k ) > f (s ⋄ k+1 ), it follows that c f (s ⋄ k ) -f (s ⋄ k+1 ) > 0. Hence we obtain L original > L swapping .(32)
This shows that locally swapping a pair (s ⋄ k , s ⋄ k+1 ) with s ⋄ k > s ⋄ k+1 reduces the total reconstruction error contributed by layers k and k + 1. Globally across all L layers, repeating such a swap for each adjacent pair that breaks the monotonicity moves s ⋄ to a strictly monotonically increasing sequence of sparsities.
By iterating this argument over each adjacent pair that fails monotonicity, we reorder s ⋄ into s ↑ . Since every swap strictly reduces the local error, the resulting monotonically increasing scheme s ↑ has an overall reconstruction error:
L ↑ = L i=1 L ↑ i < L i=1 L ⋄ i = L ⋄ .(33)
Therefore, the total reconstruction error obtained from the monotonically increasing sparsity scheme proposed in Eq. 10 is strictly less than that obtained from any non-monotonically increasing sparsity scheme, under the same average sparsity constraint.
this section cite: []

Section: D. Zero-shot Evaluation Setting
We use the lm-eval-harness framework (Gao et al., 2021) to evaluate the zero-shot performance of LLMs. By default, we follow the settings used in Wanda (Sun et al., 2023) and AlphaPruning (Lu et al., 2024), reporting the "acc" metric across all datasets. However, lm-eval-harness provides multiple metrics depending on the dataset, including both "acc" and "acc norm". In contrast, ALS (Li et al., 2024c) employs different metrics for different datasets. The evaluation metrics are summarized in Table 8.
this section cite: ['b15', 'b58', 'b38']

Section: E. Sparse Inference Engine Supported Devices
Due to the sparsity of weights in unstructured pruning, we must use a specific sparse inference engine to accelerate inference. We use DeepSparse and nm-vllm to accelerate inference on general deployment environments, including CPUs and GPUs. For CPUs, DeepSparse supports architectures including: x86 AVX2, AVX-512, AVX-512 VNNI, and ARM v8.2+, which covers most Intel, AMD, and Apple M-series CPUs. For GPUs, as long as the device supports CUDA, inference acceleration can be achieved using the nm-vllm. Similarly, if CUDA is supported for other deployment environments, nm-vllm can also be used to achieve acceleration.
this section cite: []

Section: F. More Results

this section cite: []

Section: F.1. Arithmetic and Knowledge Reasoning Tasks
We further evaluate the performance improvements of our ATP method on arithmetic and knowledge reasoning tasks for sparse models. Specifically, We evaluate the 8-shot accuracy of 40% sparse models on the GSM8K dataset and the 5-shot accuracy of 60% sparse models on the MMLU dataset. The results are presented in Table 9. Our ATP method significantly improves the accuracy of sparse models on both tasks, further demonstrating the generalization and effectiveness of our approach. We demonstrate the applicability of our method to multimodal models by integrating it with Wanda to sparsify the Vicuna-7B model (Chiang et al., 2023) in LLaVA-1.5 (Liu et al., 2024b). We only sparsify the Vicuna model within it. Similar to LLaMA, we determine the sparsity rate for each layer and sparsify linear layers. The performance of the sparse model was evaluated on various visual question-answering and reasoning benchmarks, including VQA (Singh et al., 2019), VQAv2 (Goyal et al., 2017), and SQA (Lu et al., 2022). We compared our method with magnitude-based pruning, SparseGPT, Wanda, and the DSA method combined with Wanda, under a 50% sparsity rate. The results in Table 10 show that our method outperforms the magnitude-based, SparseGPT, and Wanda approaches and demonstrates superiority over the DSA method.
this section cite: ['b4', 'b57', 'b16', 'b39']

Section: F.3. Vision Models
We compare our ATP method with other approaches for determining layer-wise sparsity rates on vision models. Experiments are conducted on both CNN and Transformer architectures, including ConvNeXt (Liu et al., 2022b), ViT (Dosovitskiy et al., 2021), and DeiT (Touvron et al., 2021), and we report the Top-1 accuracy on the ImageNet-1K dataset (Deng et al., 2009). Sparse models are obtained using the Wanda (Sun et al., 2023) method, and ATP is compared against the following baselines: uniform sparsity rate, OWL (Yin et al., 2023), and AlphaPruning (Lu et al., 2024). Following Wanda, we only sparsify the linear layers within each block of the ConvNeXt, and we use our ATP method to determine the sparsity rate for each block.
For ViT, we use ATP to determine the sparsity rate for each layer. Each layer contains modules such as attention and MLP. We use Wanda to sparsify linear layers. The results for ConvNeXt are presented in Table 11. These results demonstrate that our method is effective for determining layer-wise sparsity rates in vision models and outperforms existing methods. This further confirms the broad applicability of our monotonically increasing sparsity scheme in improving the accuracy of diverse sparse models. We also present the accuracy results for sparse ViT and DeiT models in Table 12. Additionally, we compare our method with several widely-used non-uniform layer-wise sparsity strategies in computer vision, including ERK (Mocanu et al., 2018), Global (Frankle & Carbin, 2018), and LAMP (Lee et al., 2020), as well as AlphaPruning (Lu et al., 2024). The results indicate that our method outperforms all the aforementioned baselines.  (Mocanu et al., 2018) 70.89 60.49 33.15 80.05 76.22 63.49 Global (Frankle & Carbin, 2018) 66.81 45.75 8.09 79.94 75.09 57.01 LAMP (Lee et al., 2020) 69.45 57.51 26.99 80.19 76.35 63.32 AlphaPruning (Lu et al., 2024) 71.58 64.29 44.21 80.21 77.11 64.56 ATP 72.03 65.46 47.74 80.50 78.02 69.73
this section cite: ['b37', 'b60', 'b9', 'b58', 'b67', 'b38', 'b48', 'b12', 'b28', 'b38', 'b48', 'b12', 'b28', 'b38']

Section: F.4. Integrate with other Compression Technologies
To demonstrate the generalization ability of our method for determining layer-wise sparsity rates, we combine ATP with other compression techniques, including N:M sparsity, structured pruning, and mixed-precision quantization. For N:M sparsity, we follow the mixed N:8 settings (Sun et al., 2021), using ATP to determine the value of N for each layer while maintaining average sparsity at 2:8, 3:8 and 4:8. In terms of structured pruning, we integrated ATP with LLM-Pruner (Ma et al., 2023a), where ATP determines the layer-wise sparsity rates, and LLM-Pruner applies pruning accordingly. For mixed-precision quantization, we combine ATP with the LIMPQ method (Tang et al., 2022) to determine the quantization bits for each layer. We apply these compression techniques to the LLaMA-7B model and report the perplexity of the compressed models on the WikiText-2 validation set. The experimental results are presented in Table 13. It shows that ATP significantly enhances the performance of various compression methods and outperforms both the OWL and AlphaPruning approaches.
this section cite: ['b59', 'b60']

Section: F.5. More Layer-wise Sparsity Baselines
We compare our method with additional approaches for determining layer-wise sparsity rates, including Uniform (Zhu & Gupta, 2017), Global (Frankle & Carbin, 2018), ER (Mocanu et al., 2018), ER-Plus (Liu et al., 2022a), and LAMP (Lee et al., 2020). These methods are combined with the Wanda pruning approach to obtain sparse LLaMA-7B models. The experimental results are presented in Table 14. It shows that across sparsity rates ranging from 50% to 80%, the perplexity of models pruned using the ATP method is consistently lower than that of other baselines, further demonstrating the superiority of our approach. We have demonstrated that our method improves performance over Wanda and SparseGPT methods. Notably, ATP can be integrated with any post-training sparsity method to further enhance their effectiveness. In this section, we showcase the performance improvements achieved by combining ATP with other post-training sparsity methods. Specifically, we apply ATP to DSnoT (Zhang et al., 2023), Pruner-Zero (Dong et al., 2024), and ALPS (Meng et al., 2024) to obtain a 70% sparse LLaMA2-7B model. The perplexity and zero-shot accuracy results of the sparse models are presented in Table 15. The results indicate that ATP significantly enhances the performance of DSnoT, Pruner-Zero, and ALPS methods.
this section cite: ['b75', 'b12', 'b48', 'b28', 'b74', 'b10', 'b44']

Section: F.7. Integrate with LoRA Fine-tuning
We further demonstrate the effectiveness of LoRA fine-tuning (Hu et al., 2021) in narrowing the performance gap between highly sparse LLMs and dense models. Specifically, we obtain a 70% sparse LLaMA2-7B model using the Wanda method and fine-tune it on 10,000 samples from the Alpaca-GPT4 (Peng et al., 2023) dataset. We compare the results against models sparsified using uniform sparsity, OWL, and AlphaPruning methods. The results in Table 16 show that LoRA fine-tuning significantly improves the accuracy of the sparse model, further reducing the gap with the dense model. Additionally, the sparse model obtained through the ATP method achieves higher accuracy, and this advantage is retained even after LoRA fine-tuning. needed to obtain the sparsity rates due to the narrow range of reasonable values for the hyperparameter β. Furthermore, as the number of model parameters increases, this range narrows even further, enabling the sparsity rates to be determined with fewer searches. For example, for the largest 70B model, only three searches are necessary. This demonstrates that our method is highly computationally efficient.
this section cite: ['b20', 'b52']

Section: G.2. Analyze the Layer-wise Sparsity Distribution.
We analyze the sparsity rate distribution across different average sparsity levels in Figure 2. Our findings indicate that at lower average sparsity rates, the differences in sparsity rates across layers are minimal. In contrast, these differences become more pronounced at higher average sparsity rates.
this section cite: []

Section: G.3. Comparison of Sparsity Rates Distribution with other Methods
Figure 3 illustrates the sparsity rate distributions obtained from various layer-wise sparsity methods. We observe that the sparsity rates generally follow an increasing pattern from low to high, further validating the rationale behind our method.
this section cite: []

Section: G.4. Different β Settings.
Figure 4 shows the impact of different β settings on the perplexity of the 70% sparse LLaMA2-7B model obtained using the Wanda method. We observe that as β increases, the perplexity initially decreases and then rises. Furthermore, the model's perplexity under various β settings remains lower than that of the uniform sparsity rate scheme.
this section cite: []

Section: G.5. Robustness of ATP under Different Random Seeds
Following OWL (Yin et al., 2023), DSA (Li et al., 2024b) and AlphaPruning (Lu et al., 2024), all experimental results are conducted under a single fixed random seed. We also report WikiText2 perplexity of 70% sparse LLaMA2-7B obtained by Wanda across five random seeds and different calibration sets in Table 19. The variance across random seeds is very low, suggesting the robustness of ATP.
this section cite: ['b67', 'b38']

Section: H. Comparison with NEURONAL
The previous work NEURONAL (Cunegatti et al., 2024) also proposed adopting a monotonically increasing sparsity schedule for LLMs, but there are significant differences between our ATP and NEURONAL:
1. Significant differences in algorithms for determining monotonically increasing sparsity schedules for LLMs.
Although both ATP and NEURONAL propose using monotonically increasing arithmetic progression to allocate sparsity rates for each block of LLMs, there are significant differences in the algorithms by which ATP and NEURONAL derive the final arithmetic sequences.
ATP calculates the reasonable range of values for the common difference β of the arithmetic progression, and then searches for the optimal value of β within this reasonable range using a grid search with a step size of 0.002. The goal of search is to find the value of β that minimizes perplexity of the sparse LLMs on the WikiText-2 dataset.
In contrast, NEURONAL defines a sparsity window, [s -λ, s + λ], in advance. Given an average sparsity rate of s, NEU-RONAL searches for the optimal λ value within a predefined range of [0.01, 0.02, 0.03, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.25]. The optimal λ value is determined by maximizing the alignment between the activation values of the sparse and dense models. Due to the differences in these algorithms, the final sparsity allocation schemes obtained by ATP and NEURONAL are different.
2. Significant differences between the two sparsity methods beyond the monotonically increasing sparsity schedule.
ATP focuses on deriving that existing LLM sparsity methods have a reconstruction error explosion problem, thus proposing the use of a monotonically increasing arithmetic sequence to determine the sparsity rates for LLMs to alleviate the above problem. Our method provides theoretical insights into the effectiveness and rationality of monotonically increasing arithmetic sequence sparsity schemes. We only determined non-uniform sparsity rates for each block of LLMs, with linear layers within blocks using uniform sparsity rates.
In contrast, NEURONAL proposes utilizing functional information from dense pre-trained models, i.e., their activations, to obtain sparse models that maximize alignment of activations with the corresponding dense model. NEURONAL adaptively selects the best hyperparameters for block sparsity rates and row-wise sparsity rates based on the model and desired sparsity level, to maximize neuron alignment between activations.
Therefore, ATP and NEURONAL also have significant differences at the more macro algorithmic level of sparsity algorithms.
this section cite: ['b8']

Section: I. Detailed Results for Zero-shot Tasks
In this section, we provide a detailed presentation of the performance of each zero-shot task introduced in Sec. 4.
this section cite: []

Section: 
Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective
this section cite: []

Section: References
Ref_id:b0 Title: A next-generation hyperparameter optimization framework Year: (2019)
Ref_id:b1 Title: Fluctuationbased adaptive structured pruning for large language models Year: (2024)
Ref_id:b2 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b3 Title: Distributed optimization and statistical learning via the alternating direction method of multipliers Year: (2011)
Ref_id:b4 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023-03)
Ref_id:b5 Title: Exploring the surprising difficulty of natural yes/no questions Year: (2019)
Ref_id:b6 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b7 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b8 Title: Zeroth-order adaptive neuron alignment based pruning without retraining Year: (2024)
Ref_id:b9 Title: Imagenet: A large-scale hierarchical image database Year: (2009)
Ref_id:b10 Title: Pruner-zero: Evolving symbolic pruning metric from scratch for large language models Year: (2024)
Ref_id:b11 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b12 Title: The lottery ticket hypothesis: Finding sparse, trainable neural networks Year: (2018)
Ref_id:b13 Title: Pruning neural networks at initialization: Why are we missing the mark? arXiv preprint Year: (2020)
Ref_id:b14 Title: Massive language models can be accurately pruned in one-shot Year: (2023)
Ref_id:b15 Title: A framework for few-shot language model evaluation Year: (2021)
Ref_id:b16 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b17 Title: Pruning self-attentions into convolutional layers in single path Year: (2024)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Measuring massive multitask language understanding Year: (2020)
Ref_id:b20 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b21 Title: Towards efficient automatic self-pruning of large language models Year: (2025)
Ref_id:b22 Title: Dynamic low-rank sparse adaptation for large language models Year: ()
Ref_id:b23 Title: Accelerated sparse neural training: A provable and efficient method to find n: m transposable masks Year: (2021)
Ref_id:b24 Title: Mistral 7b Year: (2023)
Ref_id:b25 Title: Mixtral of experts Year: (2024)
Ref_id:b26 Title: Finding optimal model parameters by discrete grid search Year: ()
Ref_id:b27 Title: Estimating mutual information Year: (2004)
Ref_id:b28 Title: Layer-adaptive sparsity for the magnitude-based pruning Year: (2020)
Ref_id:b29 Title: Layer normalization Year: (2016)
Ref_id:b30 Title: Transformer sublayers deserve differentiated structured compression for large language models Year: (2024)
Ref_id:b31 Title: Discovering sparsity allocation for layer-wise pruning of large language models Year: (2024)
Ref_id:b32 Title: Adaptive layer sparsity for large language models via activation correlation assessment Year: (2024)
Ref_id:b33 Title: Losparse: Structured compression of large language models based on low-rank and sparse approximation Year: (2023)
Ref_id:b34 Title: Deepseekv3 technical report Year: (2024)
Ref_id:b35 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b36 Title: The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training Year: (2022)
Ref_id:b37 Title: A convnet for the 2020s Year: (2022-06)
Ref_id:b38 Title: Alphapruning: Using heavy-tailed self regularization theory for improved layer-wise pruning of large language models Year: (2024)
Ref_id:b39 Title: Learn to explain: Multimodal reasoning via thought chains for science question answering Year: (2022)
Ref_id:b40 Title: On the structural pruning of large language models Year: (2023)
Ref_id:b41 Title: Solving oscillation problem in post-training quantization through a theoretical perspective Year: (2023)
Ref_id:b42 Title: Traditional and heavytailed self regularization in neural network models Year: (2019)
Ref_id:b43 Title: Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data Year: (2021)
Ref_id:b44 Title: ALPS: Improved optimization for highly sparse one-shot pruning for large language models Year: (2024)
Ref_id:b45 Title: Pointer sentinel mixture models Year: (2016)
Ref_id:b46 Title:  Year: (2024)
Ref_id:b47 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b48 Title: Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science Year: (2018)
Ref_id:b49 Title: Up or down? adaptive rounding for post-training quantization Year: (2020)
Ref_id:b50 Title: NeuralMagic. Neuralmagic nm-vllm inference engine Year: (2021)
Ref_id:b51 Title: Unmasking the lottery ticket hypothesis: What's encoded in a winning ticket's mask? Year: (2022)
Ref_id:b52 Title: Instruction tuning with gpt-4 Year: (2023)
Ref_id:b53 Title: Multilayer perceptron and neural networks Year: (2009)
Ref_id:b54 Title: Efficient vision transformers with dynamic token sparsification Year: (2021)
Ref_id:b55 Title: Your transformer is secretly linear Year: (2024)
Ref_id:b56 Title: An adversarial winograd schema challenge at scale Year: (2021)
Ref_id:b57 Title: Towards vqa models that can read Year: (2019)
Ref_id:b58 Title: A simple and effective pruning approach for large language models Year: (2023)
Ref_id:b59 Title: Dominosearch: Find layer-wise finegrained n: M sparse schemes from dense neural networks Year: (2021)
Ref_id:b60 Title: Mixed-precision neural network quantization via learned layer-wise importance Year: (2021)
Ref_id:b61 Title: Llama 2: Open foundation and finetuned chat models Year: (2023)
Ref_id:b62 Title: On mutual information maximization for representation learning Year: (2019)
Ref_id:b63 Title: Attention is all you need Year: (2017)
Ref_id:b64 Title: Model compression and efficient inference for large language models: A survey Year: (2024)
Ref_id:b65 Title: Structured pruning learns compact and accurate models Year: (2022)
Ref_id:b66 Title: Qwen2. 5 technical report Year: (2024)
Ref_id:b67 Title: Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity Year: (2023)
Ref_id:b68 Title: Width & depth pruning for vision transformers Year: (2022)
Ref_id:b69 Title: The combinatorial brain surgeon: pruning weights that cancel one another in neural networks Year: (2022)
Ref_id:b70 Title: Do all mobilenets quantize poorly? gaining insights into the effect of quantization on depthwise separable convolutional networks through the eyes of multi-scale distributional dynamics Year: (2021)
Ref_id:b71 Title: Can a machine really finish your sentence? arXiv preprint Year: (2019)
Ref_id:b72 Title: How sparse can we prune a deep network: A fundamental limit perspective Year: (2024)
Ref_id:b73 Title: Open pre-trained transformer language models Year: (2022)
Ref_id:b74 Title: Dynamic sparse no training: Training-free fine-tuning for sparse llms Year: (2023)
Ref_id:b75 Title: To prune, or not to prune: exploring the efficacy of pruning for model compression Year: (2017)
Ref_id:b76 Title: A survey on model compression for large language models Year: (2024)
Ref_id:b77 Title: Zero-shot Accuracy at 60% Sparsity We present the zero-shot accuracy results of LLaMA3-8B at 60% sparsity in Table 17. The sparse model is obtained using the Wanda method, and we compare ATP with other layer-wise sparsity methods. We can observe that ATP significantly improves the accuracy of Wanda, increasing the average zero-shot accuracy by 4.13% and narrowing the performance gap between sparse and dense model. Furthermore, ATP outperforms OWL, DSA, and AlphaPruning, with 1.05% higher accuracy than the best-performing AlphaPruning. Table 17. Zero-shot accuracy results of sparse LLaMA3-8B at 60% sparsity obtained using the Wanda method Year: ()
Ref_id:b78 Title: Table 18, we report the time required to determine the layer-wise sparsity rates for 70% sparse LLMs using our ATP method. The measurements were conducted on NVIDIA A100 80GB GPUs. The results show that only a few searches are Year: ()
