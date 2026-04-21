Title: AuroRA: Breaking Low-Rank Bottleneck of LoRA with Nonlinear Mapping
Abstract: Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method validated across NLP and CV domains. However, LoRA faces an inherent low-rank bottleneck: narrowing its performance gap with full finetuning requires increasing the rank of its parameter matrix, resulting in significant parameter overhead. Recent linear LoRA variants have attempted to enhance expressiveness by introducing additional linear mappings; however, their composition remains inherently linear and fails to fundamentally improve LoRA's representational capacity. To address this limitation, we propose AuroRA, which incorporates an Adaptive Nonlinear Layer (ANL) between two linear projectors to capture fixed and learnable nonlinearities. This combination forms an MLP-like structure with a compressed rank, enabling flexible and precise approximation of diverse target functions while theoretically guaranteeing lower approximation errors and bounded gradients. Extensive experiments on 22 datasets and 6 pretrained models demonstrate that AuroRA: (I) not only matches or surpasses full fine-tuning performance with only 6.18% ∼ 25% of LoRA's parameters but also (II) outperforms competitive PEFT methods by up to 10.88% in both NLP and CV tasks, and (III) exhibits robust performance across various rank configurations.

Section: 
1 Introduction In recent years, pretrained models have demonstrated excellent generalization performance across numerous tasks in various domains [1,2,3,4,5,6]. In practical applications, to further unleash their powerful capabilities on specific downstream tasks, these models often require relevant fine-tuning [7,8,9,10,11,12]. However, the increasing size of their parameters poses a significant challenge to finetuning all parameters [13,14]. To address this issue, the field of Parameter-Efficient Fine-Tuning (PEFT) has made substantial progress [13,15,16,17,18,19,20]. The core idea is to fine-tune only a small subset of the model's parameters while freezing the majority of the pretrained parameters, achieving performance comparable to full fine-tuning [21]. LoRA is a commonly used state-of-theart PEFT method [16]. Specifically, it assumes that the weight updates conform to a low-rank hypothesis and represents these updates using two low-rank matrices, i.e., W 0 + ∆W = W 0 + BA. Its performance has been validated in fields such as natural language processing (NLP) [16,22] and computer vision (CV) [23,24]. Despite its significant success, LoRA still faces an inherent limitation, namely the low-rank bottleneck, as illustrated in Figure 1. As the rank of LoRA increases, the model's performance improves, thereby narrowing the gap with full fine-tuning [25]; however, the parameter cost grows proportionally with the rank, which weakens its parameter efficiency. This dilemma leads us to the first research question: ❶ Can we achieve a further balance between parameters and performance?
Recently, several linear LoRA variants have emerged [26,27,28,29]. They introduce an additional matrix between the B and A matrices of LoRA to weaken the correlation constraints between them, thereby enhancing LoRA's learning and expressive capabilities. Specifically, one approach involves the introduction of a diagonal matrix to facilitate singular value decomposition [26,27], while another approach incorporates an arbitrary matrix to fuse subspaces [28,29]. Nevertheless, LoRA's inherent linearity persists even when an additional matrix is introduced, preserving its fundamental structure as a linear mapping. As illustrated in Figure 2, when the rank is extremely low, MoSLoRA [28] (a linear variant that incorporates an arbitrary matrix) has only a marginal effect on expanding the exploration of ∆W , leading to a failure to further boost performance (2.2% ↑ on DTD and 0.56% ↓ on RESISC45). The structural characteristics and resultant performance limitations of linear variants naturally prompt our second research question: ❷ Can we achieve more than marginal performance improvements by introducing a nonlinear transformation between LoRA's two linear layers? Motivated by the above two research questions, this paper focuses on introducing nonlinear mappings into LoRA and further compressing the rank to achieve a better balance between parameters and performance. To this end, we propose a method called Activate Your Low-Rank Adaptation (AuroRA). We revisit LoRA through the lens of linear mappings and identify two critical limitations: (I) insufficient expressiveness and (II) limited training flexibility. To fully harness LoRA's potential, AuroRA introduces an Adaptive Nonlinear Layer (ANL) between the low-rank matrices, forming an MLP-like structure. ANL employs a hybrid design of fixed and learnable nonlinearities to enhance model expressivity within a more compressed rank while enabling flexible training strategies to expand the explorable parameter space (Figures 1 and 2). Theoretical analysis demonstrates that AuroRA not only achieves a strictly lower approximation error than LoRA but also preserves bounded gradient norms. Experiments across NLP and CV tasks confirm the efficiency, generalizability, and robustness of AuroRA. We further conduct ablation studies to dissect the contributions of fixed and learnable components, and evaluate its robustness against linear LoRA variants across multiple rank configurations. Our contributions can be summarized as follows:
❶ Perspective Shift. We systematically revisit two research lines of LoRA: the low-rank bottleneck and linear LoRA variants. By interpreting LoRA through the lens of linear mappings, we address both research questions within a unified framework, providing theoretical analyses. ❷ Nonlinear Proposal. We propose AuroRA, which introduces nonlinear mappings into LoRA and further compresses the rank, resulting in a superior balance between parameters and performance, paving the way for further unlocking the significant potential of LoRA. ❸ Experimental Validation. Extensive experiments on 22 datasets and 6 pretrained models showcase that AuroRA: (I) not only matches or surpasses full fine-tuning performance with only 6.18% ∼ 25% of LoRA's parameters but also (II) outperforms competitive PEFT methods by up to 10.88% in NLP and CV tasks, and (III) exhibits robust performance across various rank configurations.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b12', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b15', 'b15', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b25', 'b26', 'b27', 'b28', 'b27']

Section: Methodology
As illustrated in Figure 3, we introduce AuroRA, an extension of LoRA that incorporates nonlinear mappings to overcome the inherent low-rank bottleneck. We reinterpret LoRA as a two-layer linear mapping, whereas our proposed AuroRA transforms it into an MLP-like structure by introducing an adaptive nonlinear layer.
this section cite: []

Section: LoRA: A Two-Layer Linear Mapping
In standard LoRA [16], the weight update ∆W for a pre-trained weight matrix W 0 is approximated as the product of two low-rank matrices:
∆W = BA,(1)
where A ∈ R r×din and B ∈ R dout×r , with the rank r satisfying r ≪ min(d in , d out ). The forward propagation for an input vector x ∈ R din is thus expressed as:
h = W 0 x + ∆Wx = W 0 x + BAx.(2)
The above process can be interpreted as a two-layer linear mapping, where A serves as a downward projector P down that maps the input x from a high-dimensional space R din to a lower-dimensional hidden space R r , and B serves as an upward projector P up that maps back to R dout . However, we note that LoRA is constrained by its sequential linear mapping structure, leading to two significant shortcomings: ❶ insufficient expressiveness: being a purely linear structure, it requires increasing the hidden dimension to handle more complex incremental weights and improve performance; ❷ limited training flexibility: the direct low-rank decomposition induces strong interdependencies between the linear layers, imposing rigid structural constraints that reduce training flexibility [30].
this section cite: ['b15', 'b29']

Section: AuroRA: An MLP-like Structure
To address these limitations, AuroRA introduces an Adaptive Nonlinear Layer (ANL) between A and B, modifying the weight update as follows:
∆W = B • σ(A),(3)
where σ is the element-wise ANL that maps from R r to R r . Here, r denotes the compressed hidden dimension ( r ≪ r), i.e., the low dimension to which the input is projected by P down . Formally, the forward propagation equation in the training phase is given by:
h = W 0 x + B • σ(Ax).(4)
The introduction of ANL enables AuroRA to form an MLP (Multilayer Perceptron)-like structure.
this section cite: []

Section: Adaptive Nonlinear Layer
Consider an arbitrary input vector z. After projecting z into an r-dimensional hidden layer, our objective is to introduce sufficient nonlinearity to capture as many complex relationships as possible within this limited hidden space. To achieve this, we propose the following components: ❶ fixed nonlinearity (F), which utilizes parameter-free nonlinear activation functions to activate neurons in the hidden space, thereby achieving coarse fitting; and ❷ learnable nonlinearity (L), which employs parameterized nonlinear functions during the training process of the weight update increments, facilitating fine fitting. By combining ❶ and ❷, the Adaptive Nonlinear Layer (ANL) can be formally expressed as:
σ(Z) = F (Z) + L(Z),(5)
where F represents the fixed nonlinear activation, L denotes the learnable nonlinear function, and Z denotes the input to ANL. We provide a detailed comparison of fixed and learnable nonlinearity in Section 3.4.
For ❶, we adopt widely used activation functions in deep learning, such as ReLU [31], sigmoid, and tanh. A detailed comparison of different activation functions and their impact on AuroRA's performance is provided in Section 3.4. Through our comparative evaluations, tanh emerges as the top-performing activation function, and theoretical analysis concurrently ensures its training stability. This preference is consistent with empirical findings in prior studies [32,33] that demonstrate the robust performance of the tanh activation function for large-scale models, leading us to employ tanh in our implementation. The depth of the network influences the number of activation functions that can be introduced. Specifically, we introduce a self-projection P self ∈ R r× r between P down and P up , which extends the depth of the standard LoRA structure. Subsequently, we introduce tanh activation functions between P down and P self , and between P self and P up . Formally, the fixed nonlinear component is defined as:
F(Z) = tanh (H (tanh(Z))) ,(6)
where H ∈ R r× r denotes P self .
To achieve ❷, we propose using spline functions to model complex relationships [34]. Numerous prior studies [35,36,37,38] have demonstrated that splines are flexible, piecewise polynomial functions capable of approximating a wide range of nonlinear behaviors. Specifically, we employ B-spline basis functions to construct the learnable component. Formally, the learnable nonlinear component is defined as:
L(Z) = w s • s(Z),(7)
where w s ∈ R r is the spline weight vector, and s(Z) = r i=1 B(z i ) represents the spline basis functions applied to each dimension z i of Z. The learnable parameters in this component are the spline weights w s , which determine the contribution of each basis function B(z i ) to the overall output of L(Z). During training, these weights are iteratively updated to minimize the task-specific loss function.
By introducing ❶ and ❷ in the hidden layer with dimension r, ANL effectively captures complex relationships without significantly increasing the number of additional parameters. The combination of coarse fitting and fine fitting enhances the standard LoRA structure, improving its expressive capacity and training flexibility, achieving what we refer to as Activate Your Low-Rank Adaptation. The complete Adaptive Nonlinear Layer (ANL) developed in our work can then be formally represented as:
σ(Z) = tanh (H (tanh(Z))) + w s • s(Z).(8)
Further details and the complete algorithmic workflow of AuroRA are provided in Appendix B.
this section cite: ['b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37']

Section: Theoretical Analysis
In this subsection, we propose two theoretical propositions concerning AuroRA and analyze its parameter and computational cost. Additionally, we present an intuitive case in Appendix C to help better understand the role of nonlinearities. Proposition 2.1 (Lower Approximation Error). Let M ∈ R dout×din with rank(M ) > r. Define
ε r (M ) = inf U ∈R d out ×r , V ∈R r×d in ∥ M -U V ∥.
Then ε r (M ) > 0, and for our proposed update of the form
M nonlinear (x) = B σ A x , A ∈ R r×din , B ∈ R dout×r ,
where σ is our adaptive nonlinear layer, there exists a parameter set (A * , B * , σ * ) such that M -M nonlinear ≤ c ε r (M ), 0 < c < 1.
Hence, the approximation error is strictly below the linear rank-r limit ε r (M ), using the same rank r.
▶ Proposition 2.1 indicates that, thanks to the introduction of nonlinear mappings, AuroRA achieves a strictly lower approximation error compared to LoRA at the same rank, meaning that the resulting weight updates are closer to the optimal solution. Furthermore, our empirical results demonstrate that this improvement persists even when further compressing the hidden dimensions of AuroRA. A rigorous proof, along with technical details and error bounds, is provided in Appendix D. Proposition 2.2 (Gradient Boundedness). In the AuroRA, the use of the tanh activation function and B-spline basis functions results in bounded gradients with respect to both the inputs and the model parameters.
▶ Proposition 2.2 posits that, despite the introduction of fixed and learnable nonlinearities, AuroRA maintains bounded gradients during training, thereby ensuring training stability. The corresponding proof is provided in Appendix E.
this section cite: []

Section: Parameter Cost
In Section 1, we discussed the relationship between trainable parameters and rank in LoRA, where the number of introduced trainable parameters is O(r(d in + d out )). Here, d in and d out represent the input and output dimensions, respectively, i.e., PARAMS ∝ r. In AuroRA, we aim to further compress the parameter count by setting the hidden layer dimension to r = r/k, where k is a constant and r is the optimal rank setting of LoRA. This means that the number of trainable parameters in AuroRA is 1/k of that in LoRA. In this work, we set r to 2, corresponding to values of k such as 4 and 8. The additional parameters introduced in ANL are of the order O(2 r 2 ), which, compared to the significant reduction in parameter count, can be considered negligible.
Computational Cost The computational complexity of AuroRA's forward pass in the training phase, ∆h = Bσ • (Ax), is analyzed as follows. Let b denote the batch size, d in and d out the input/output feature dimensions, r the rank, and G the collective B-spline parameters (a small constant, G = O(r)).
The linear projections by A ∈ R r×din and B ∈ R dout×r incur complexities of O(bd in r) and O(brd out ), respectively. The intermediate fixed and learnable non-linearities, σ(•), each contribute an additional O(br 2 ) term (with the learnable component's O(brG) complexity simplifying due to G = O(r)). Consequently, the total complexity for AuroRA is O(b(d in r + 2r 2 + rd out )). Given the standard low-rank setting where r ≪ min(d in , d out ), the quadratic overhead O(br 2 ) introduced by the non-linearities is negligible compared to the dominant linear terms, thus maintaining a computational footprint comparable to that of LoRA.
this section cite: []

Section: Experiments
In this section, we conduct extensive experiments to answer the following research questions: (RQ1) Can AuroRA effectively achieve efficiency in NLP tasks? (RQ2) Can AuroRA effectively achieve efficiency in CV tasks? (RQ3) What are the respective roles of fixed and learnable nonlinearity? (RQ4) How do different activation functions in fixed nonlinearity affect performance? (RQ5) How does AuroRA's sensitivity to rank compare to that of linear LoRA variants? 1
this section cite: []

Section: Experimental Setup

this section cite: []

Section: Datasets and Pre-Trained Models Datasets
For our experiments, we evaluate the ability of AuroRA to achieve parameter-efficient fine-tuning using four categories of datasets spanning both NLP and CV domains: ■ Natural Language Understanding: We employ GLUE (General Language Understanding Evaluation) [39], a widely used multi-task benchmark in NLU, which includes datasets such as SST-2, MRPC, CoLA, QNLI, RTE, and STS-B. The evaluation metrics are as follows: CoLA is assessed using Matthew's correlation coefficient, STS-B with Pearson's correlation coefficient, and accuracy is used for the other tasks. ■ Commonsense Reasoning: We use a collection of commonly used datasets, including BoolQ [40], PIQA [41], SocialIQA [42], HellaSwag [43], WinoGrande [44], ARC-e, ARC-c [45], and OpenBookQA [46]. For fair comparison, we follow the setup proposed by [28], fine-tuning the pretrained models on the Commonsense170K dataset, which serves as a mixture of the aforementioned benchmark datasets. We then evaluate using accuracy as the performance metric. ■ Image Classification: We use five datasets with small label spaces-OxfordPets [47], CIFAR-10 [48], DTD [49], EuroSAT [50], and RESISC45 [51], and three datasets with large label Table 2: Commonsense reasoning evaluation results for LLaMA3-8B on eight tasks. * indicates numbers taken from [57]. The best results are highlighted in bold, and the runners-up are underlined. For all eight tasks, higher values are considered better. Method Params. BoolQ PIQA SIQA HellaSwag WinoGrande ARC-e ARC-c OBQA Avg. LoRA* 56.6M 70.8 85.2 79.9 91.7 84.3 84.2 71.2 79.0 80.8 PiSSA* 83.8M 67.1 ↓3.7 81.1 ↓4.1 77.2 ↓2.7 83.6 ↓8.1 78.9 ↓5.4 77.7 ↓6.5 63.2 ↓8.0 74.6 ↓5.4 75.4 ↓5.4 MiLoRA* 56.6M 68.8 ↓2.0 86.7 ↑1.5 77.2 ↓2.7 92.9 ↑1.2 85.6 ↑1.3 86.8 ↑2.6 75.5 ↑4.3 81.8 ↑2.8 81.9 ↑1.1 AuroRA 3.5M 72.5 ↑1.7 87.4 ↑2.2 79.0 ↓0.9 94.2 ↑2.5 83.0 ↓1.3 89.3 ↑5.1 78.8 ↑7.6 84.8 ↑5.8 83.6 ↑2.8
spaces, namely StanfordCars [52], FGVC [53], and CIFAR-100 [48]. ■ Subject-Driven Generation: Following [54], we use the DreamBooth dataset. More detailed descriptions of the datasets can be found in Appendix F.1, F.2, F.3.
this section cite: ['b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b27', 'b46', 'b48', 'b49', 'b50', 'b51', 'b52', 'b47', 'b53']

Section: Pre-Trained Models
We focus on a selection of representative pretrained models, including RoBERTa (Base & Large) [1], LLAMA3-8B [3], ViT (Base & Large) [55] and SDXL [56].
this section cite: ['b0', 'b2', 'b54', 'b55']

Section: Baselines
In the baseline evaluation, we adopt a range of representative and competitive fine-tuning methods, categorized into three groups: Full Fine-Tuning, PEFT methods, and LoRA variants. The PEFT methods we use include BitFit [20], Adapter H [32], Adapter D [58], Adapter P [59], and LoRA [16].
For the LoRA variants, we consider AdaLoRA [26], DyLoRA [60], FourierFT [61], LoRA-drop [62], DoRA [63], MoSLoRA [28], PiSSA [64] and MiLoRA [57].
this section cite: ['b19', 'b31', 'b57', 'b58', 'b15', 'b25', 'b59', 'b60', 'b61', 'b62', 'b27', 'b63', 'b56']

Section: AuroRA Achieves Efficiency in NLP Tasks (RQ1)
To answer RQ1, we design two tasks: Natural Language Understanding (NLU) and Commonsense Reasoning. In the NLU task, we select RoBERTa-Base and RoBERTa-Large [1] as pretrained models and compare AuroRA with ten other widely-used fine-tuning methods across all six datasets of the GLUE benchmark [39]. The results of this extensive comparison are shown in Table 1, with additional hyperparameter configuration details provided in Appendix G.1. Following [16], we fine-tune only the query and value weights of each transformer block, while fully fine-tuning the classification head. For the commonsense reasoning task, we select LLaMA3-8B [3] as the base model and compare AuroRA with LoRA and two other LoRA variants (PiSSA [64] and MiLoRA [57]). The results are shown in Table 2. The relevant hyperparameters are listed in Appendix G.2. Our observations can be summarized as follows:
Table 3: Fine-tuning results with ViT Base and Large models on different image classification datasets. We report the accuracy (%) after 10 epochs. Avg. represents the average accuracy across all datasets for each method. * indicates numbers taken from [61]. The best results are highlighted in bold, and the runners-up are underlined (excluding full fine-tuning).
Model Method Params. OxfordPets StanfordCars CIFAR10 DTD EuroSAT FGVC RESISC45 CIFAR100 Avg. ViT-Base Full Fine-Tuning* 85.8M 93.1 79.8 98.9 77.7 99.1 54.8 96.1 92.4 86.5 Linear Probing* -90.3↓2.8 25.8↓54.0 96.4↓2.5 69.8↓7.9 88.7↓10.4 17.4↓37.4 74.2↓21.9 84.3↓8.1 68.4↓18.1 LoRA* 581K 93.2↑0.1 45.4↓34.4 98.8↓0.1 75.0↓2.7 98.4↓0.7 25.2↓29.6 92.7↓3.4 92.0↓0.4 77.6↓8.9 FourierFT* 239K 93.1↑0.0 56.4↓23.4 98.7↓0.2 77.3↓0.4 98.8↓0.3 32.4↓22.4 94.3↓1.8 91.5↓0.9 80.3↓6.2 AuroRA 74K 93.9↑0.8 75.7↓4.1 98.8↓0.1 79.6↑1.9 98.8↓0.3 48.2↓6.6 93.6↓2.5 92.0↓0.4 85.1↓1.4 ViT-Large Full Fine-Tuning* 303.3M 94.4 88.9 99.2 81.8 99.0 68.3 96.4 93.6 90.2 Linear Probing* -91.1↓3.3 37.9↓51.0 97.8↓1.4 73.3↓8.5 92.6↓6.4 24.6↓43.7 82.0↓14.4 84.3↓9.3 73.0↓17.2 LoRA* 1.57M 94.8↑0.4 73.3↓15.6 99.1↓0.1 81.8↑0.0 98.6↓0.4 42.3↓26.0 94.7↓1.7 94.9↑1.3 84.9↓5.3 FourierFT* 480K 94.8↑0.4 79.1↓9.8 99.1↓0.1 81.9↑0.1 98.7↓0.3 51.3↓17.0 95.2↓1.2 93.4↓0.2 86.7↓3.5 AuroRA 197K 94.9↑0.5 82.5↓6.4 99.1↓0.1 82.1↑0.3 98.9↓0.1 59.8↓8.5 94.9↓1.5 93.3↓0.3 88.2↓2.0
Obs. ❶ AuroRA demonstrates strong efficiency in NLP tasks. It is evident that AuroRA outperforms the baseline across all datasets and pretrained models in both tasks. Compared to Full Fine-Tuning, AuroRA achieves a performance improvement ranging from 0.1% ∼ 8.3% while using only 0.04% ∼ 0.06% of the total parameters. Compared to PEFT baselines, including LoRA, AuroRA achieves a performance improvement of up to 24.7% and an average improvement of 1.25% ∼ 10.88%, using only 6.25% ∼ 25% of the parameters. Specifically, in the commonsense reasoning task using LLaMA3-8B as the pretrained model, AuroRA achieves a significant 10.7% performance boost on ARC-C with just 6.25% of LoRA's parameter budget. In the NLU task, although AuroRA uses more parameters than FourierFT, it demonstrates significant performance gains across all pretrained models and datasets. For instance, using RoBERTa-Base, AuroRA improves performance by 7.7% on RTE.
Obs. ❷ AuroRA can be scaled up to fine-tune large pretrained models. AuroRA scales effectively to fine-tuning larger pretrained models. In the NLU task, when the pretrained model changes to RoBERTa-Large from Base, nearly all PEFT methods show a performance drop compared to Full Fine-Tuning, with the largest decrease reaching 3.7%. In contrast, AuroRA still achieves performance improvements of 0.1% ∼ 3.8% across all datasets. In the commonsense reasoning task, when the model size increases to 8B, AuroRA continues to outperform LoRA by 2.4% ∼ 10.7%.
this section cite: ['b0', 'b38', 'b15', 'b2', 'b63', 'b56', 'b60']

Section: AuroRA Achieves Efficiency in CV Tasks (RQ2)
To answer RQ2, we design two tasks: Image Classification and Subject-Driven Image Generation.
In the image classification task, following [61], we select ViT-Base and ViT-Large [55], two popular CV foundation models, which are pretrained on the ImageNet-21K [65] dataset. We then compare AuroRA with Full Fine-Tuning, Linear Probing (fine-tuning only the classification head), LoRA, and FourierFT. The results are presented in Table 3, with more implementation details available in Appendix G.3. In the subject-driven image generation task [54], following [61] and [28], we use the SDXL model [56] as our backbone, and then fine-tune it using both LoRA and AuroRA. The objective is to generate images based on specified prompts for a particular subject, which is defined using a set of reference images. Initially, we fine-tune a text-to-image model by pairing the input images with text prompts that include a unique identifier (e.g., "A photo of a [V] dog"). Subsequently, the model can generate images corresponding to other prompts that incorporate the same unique identifier, thereby producing images of the defined subject. The results are presented in Figure 4, and more generated cases are in Appendix H. Our observations can be summarized as follows:
Obs. ❸ AuroRA achieves the best performance, excluding Full Fine-Tuning, with the least number of parameters. It is evident that AuroRA outperforms all other PEFT baseline methods across all eight datasets with the lowest parameter count (12.7% of LoRA and 31.0% of FourierFT) when using both the Base and Large models. Compared to Full Fine-Tuning, AuroRA uses only 0.086% of the parameters and achieves a performance improvement of 0.4% ∼ 2.4% on some datasets, with only a 1.6% ∼ 2.2% gap in average performance. When using ViT-Base on STANFORDCARS, other baselines show a significant performance drop of 29.3% ∼ 67.7% compared to Full Fine-Tuning. In contrast, AuroRA only experiences a moderate drop of 5.1%. Compared to PEFT methods, AuroRA achieves an average performance improvement of 1.73% ∼ 9.66%.
Obs. ❹ AuroRA demonstrates stronger adaptability in the text-to-image domain. We observe that in the subject-driven image generation task, AuroRA aligns better with the environment specified in the prompt. Specifically, when given the prompt "A [V] bear plushie on top of green grass with
Input Images LoRA AuroRA A [V] bear plushie on top of a white rug A [V] bear plushie on top of green grass with sunflowers around it Input Images LoRA AuroRA A [V] dog in the snow A purple [V] dog Input Images LoRA AuroRA A [V] robot toy on the beach A [V] robot toy floating in an ocean of milk Input Images A cube shaped [V] duck toy LoRA AuroRA A [V] duck toy with the Eiffel Tower in the background sunflowers around it", LoRA generates an environment with only green grass but no sunflowers. In contrast, AuroRA successfully generates green grass with sunflowers.
this section cite: ['b60', 'b54', 'b64', 'b53', 'b60', 'b27', 'b55']

Section: Study Ablation Study (RQ3)
To evaluate the contribution of different modules in AuroRA, we introduce two variants: (1) AuroRA w/o F, and (2) AuroRA w/o L, which correspond to the removal of the fixed and learnable nonlinearity in AuroRA, respectively. We compare these two variants with AuroRA by fine-tuning ViT-Base on OXFORDPETS, CIFAR10, DTD, and EUROSAT in the image classification task. From Table 4, we observe that: ❶ removing any component results in a performance drop for AuroRA; ❷ AuroRA w/o L consistently underperforms across all datasets, indicating that the learnable nonlinearity plays a more crucial role in the success of our method. Specifically, the learnable nonlinearity enables fine fitting, while the fixed nonlinearity contributes to coarse fitting.
this section cite: []

Section: Effect of Activation Function (RQ4)
We investigate the impact of the choice of activation function in the fixed nonlinearity on AuroRA's performance. Specifically, we introduce two variants:
(1) AuroRA-lr, and (2) AuroRA-sm, which correspond to replacing the activation function in the fixed nonlinearity (tanh) with LeakyReLU and Sigmoid, respectively. We compare these variants with AuroRA by fine-tuning the ViT-Base model on STANFORDCARS, FGVC, RESISC45, and CIFAR100 in the image classification task. From Table 5, we observe that Sigmoid results in the lowest performance, while tanh achieves the highest performance. Therefore, we choose tanh as the activation function for fixed nonlinearity in all our experiments.
this section cite: []

Section: Sensitivity to Rank & Comparison with Linear LoRA Variants (RQ5)
To further investigate the impact of introducing nonlinearity, we examine its sensitivity to rank and compare it with the linear LoRA variant under identical experimental settings. Specifically, we select LLaMA3-8B as the pretrained model and fine-tune it using AuroRA, MoSLoRA, and LoRA, varying the rank among {2, 4, 8, 16}. We evaluate their performance across four datasets. From Figure 5, we observe the following: ❶ the introduction of nonlinearity results in smaller performance fluctuations as the rank varies, i.e., more robustness to rank; ❷ AuroRA consistently outperforms across almost all rank settings and datasets, indicating that incorporating nonlinearity further enhances the model's expressiveness compared to linear approaches.    59] and Hyperformer [69]; prompt-based methods, like Prefix-tuning [17] and p-tuning v2 [70]. ❷ Selective PEFT methods optimize a chosen subset of a pretrained model's parameters while keeping the majority frozen [71,72,73,74,75,76]. This selection is often achieved through unstructured masking based on criteria like parameter significance, as seen in FishMask [77] and Child-tuning [78], or via structured techniques that group parameters, such as Bitfit [20] and SPT [79]. ❸ Reparameterized PEFT techniques transform model weights into more efficient, often low-rank, representations during fine-tuning, without altering the core architecture for inference [80,60,81,82,83]. A prominent example is LoRA [16], which introduces low-rank matrices for updates. ❹ Memory-Efficient PEFT methods focus on reducing the memory footprint of fine-tuning by optimizing the training dynamics rather than the model architecture [84,85,86,87]. A representative example is GaLore [87], which projects gradients into low-rank subspaces to lower optimizer-state memory while preserving full-parameter adaptability. ❺ Hybrid PEFT methods integrate multiple strategies from different PEFT categories to capitalize on their respective advantages [19,88,89]. For instance, NOAH [90] and AUTOPEFT [91], leverage neural architecture search to identify effective PEFT combinations for specific tasks. In this paper, we primarily focus on LoRA, a reparameterized PEFT method.
this section cite: ['b58', 'b68', 'b16', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b75', 'b76', 'b77', 'b19', 'b79', 'b80', 'b59', 'b81', 'b82', 'b83', 'b15', 'b84', 'b85', 'b86', 'b87', 'b87', 'b18', 'b88', 'b89', 'b90', 'b91']

Section: LoRA and its Variants
The core idea of LoRA [16] is to approximate weight updates using mergeable, low-rank matrix pathways. Its variants can be broadly categorized into several types: ❶ Novel Branch Designs primarily focus on remodeling or reformulating the original low-rank matrix approximation pathway, with notable examples including VeRA [80], FourierFT [61], PiSSA [64], and DoRA [92]. ❷ Multi-Task Variants, exemplified by MoELoRA [81], MoA [82], CA-LoRA [93], and HydraLoRA [94], are engineered to enhance cross-task generalization-particularly in scenarios such as multi-task learning, domain adaptation, and continual learning-often through the strategic employment of LoRA module mixtures or ensembles. ❸ Linear Variants, including AdaLoRA [26], SaLoRA [27], MoSLoRA [28], and FLoRA [29], typically augment the LoRA framework by incorporating an additional linear matrix between the two original low-rank factors, thereby bolstering information capture during the training phase. Beyond these, several nonlinear LoRA variants have recently emerged, including LoRAN [95], SineLoRA [96], LoDA [97], NEAT [98], and CoLA [99]. However, these recent nonlinear variants do not resolve inherent low-rank bottleneck in LoRA. In contrast, our method pairs nonlinearities with a focus on LoRA's fundamental structural limitations, achieving a superior balance between performance and parameter efficiency.
this section cite: ['b15', 'b80', 'b60', 'b63', 'b92', 'b81', 'b82', 'b93', 'b94', 'b25', 'b26', 'b27', 'b28', 'b95', 'b96', 'b97', 'b98', 'b99']

Section: Conclusion
In this paper, we revisit LoRA from the perspective of linear mappings and introduce nonlinearity into LoRA by proposing AuroRA, an MLP-like structure. AuroRA incorporates an adaptive nonlinear layer that includes both fixed and learnable nonlinearities between the two low-rank matrices. AuroRA achieves a superior balance between performance and parameters across tasks in both the NLP and CV domains. We hope that AuroRA will inspire further exploration of nonlinear extensions to LoRA.
Limitation A potential limitation is that, due to limited computational resources, we do not evaluate performance on larger pretrained models in this study, leaving this exploration for future work.
Broader Impact As a novel nonlinear method, AuroRA is envisioned for broad future applications in key sectors such as healthcare and finance. It is anticipated to deliver more accurate and reliable services while significantly reducing resource consumption, thereby better serving human society.
supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.
IMPORTANT, please:
• Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist",
• Keep the checklist subsection headings, questions/answers and guidelines below.
• Do not modify the questions and only use the provided macros for your answers.
this section cite: []

Section: References
Ref_id:b0 Title: A robustly optimized bert pretraining approach Year: (2019)
Ref_id:b1 Title: Deberta: Decoding-enhanced bert with disentangled attention Year: (2021)
Ref_id:b2 Title: Llama 3 model card Year: (2024)
Ref_id:b3 Title: Debertav3: Improving deberta using electra-style pre-training with gradient-disentangled embedding sharing Year: (2021)
Ref_id:b4 Title: Internlm2 technical report Year: (2024)
Ref_id:b5 Title: Tree of thoughts: Deliberate problem solving with large language models Year: (2024)
Ref_id:b6 Title: Stanford alpaca: An instruction-following llama model Year: (2023)
Ref_id:b7 Title: Universal language model fine-tuning for text classification Year: (2018-07)
Ref_id:b8 Title: Language models are few-shot learners Year: (2020)
Ref_id:b9 Title: Language models for dialog applications Year: (2022)
Ref_id:b10 Title: Meta-r1: Empowering large reasoning models with metacognition Year: (2025)
Ref_id:b11 Title: Measuring human and ai values based on generative psychometrics with large language models Year: (2025)
Ref_id:b12 Title: Towards a unified view of parameter-efficient transfer learning Year: (2021)
Ref_id:b13 Title: Exploring the limits of transfer learning with a unified text-to-text transformer Year: (2020)
Ref_id:b14 Title:  Year: (2024)
Ref_id:b15 Title: LoRA: Low-rank adaptation of large language models Year: (2022)
Ref_id:b16 Title: Prefix-tuning: Optimizing continuous prompts for generation Year: (2021)
Ref_id:b17 Title: Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning Year: (2022)
Ref_id:b18 Title: LLM-adapters: An adapter family for parameter-efficient fine-tuning of large language models Year: (2023-12)
Ref_id:b19 Title: Bitfit: Simple parameter-efficient finetuning for transformer-based masked language-models Year: (2022)
Ref_id:b20 Title: Parameter-efficient fine-tuning for large models: A comprehensive survey Year: (2024)
Ref_id:b21 Title: Exploring the impact of low-rank adaptation on the performance, efficiency, and regularization of rlhf Year: (2023)
Ref_id:b22 Title: Convolution meets loRA: Parameter efficient finetuning for segment anything model Year: (2024)
Ref_id:b23 Title: Low-rank rescaled vision transformer fine-tuning: A residual design approach Year: (2024)
Ref_id:b24 Title: LoRA learns less and forgets less Year: (2024)
Ref_id:b25 Title: Adaptive budget allocation for parameter-efficient fine-tuning Year: (2023)
Ref_id:b26 Title: Structure-aware low-rank adaptation for parameter-efficient fine-tuning Year: (2023)
Ref_id:b27 Title: Mixture-of-subspaces in low-rank adaptation Year: (2024-11)
Ref_id:b28 Title: Flora: Low-rank core space for n-dimension Year: (2024)
Ref_id:b29 Title: See further for parameter efficient fine-tuning by standing on the shoulders of decomposition Year: (2024)
Ref_id:b30 Title: Deep learning using rectified linear units (relu) Year: (2018)
Ref_id:b31 Title: Parameter-efficient transfer learning for nlp Year: (2019)
Ref_id:b32 Title: Transformers without normalization Year: ()
Ref_id:b33 Title: A practical guide to splines Year: (1978)
Ref_id:b34 Title: KAN: Kolmogorov-arnold networks Year: (2025)
Ref_id:b35 Title: Deep spline networks with control of lipschitz regularity Year: (2019)
Ref_id:b36 Title: Exsplinet: An interpretable and expressive spline-based neural network Year: (2022)
Ref_id:b37 Title: Learning activation functions in deep (spline) neural networks Year: (2020)
Ref_id:b38 Title: GLUE: A multi-task benchmark and analysis platform for natural language understanding Year: (2019-05-06)
Ref_id:b39 Title: BoolQ: Exploring the surprising difficulty of natural yes/no questions Year: (2019-06)
Ref_id:b40 Title: Piqa: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b41 Title: Social IQa: Commonsense reasoning about social interactions Year: (2019-11)
Ref_id:b42 Title: HellaSwag: Can a machine really finish your sentence Year: (2019-07)
Ref_id:b43 Title: Winogrande: An adversarial winograd schema challenge at scale Year: (2020)
Ref_id:b44 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b45 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018-11)
Ref_id:b46 Title: Cats and dogs Year: (2012)
Ref_id:b47 Title: Learning multiple layers of features from tiny images Year: (2009)
Ref_id:b48 Title: Describing textures in the wild Year: (2014)
Ref_id:b49 Title: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification Year: (2019)
Ref_id:b50 Title: Remote sensing image scene classification: Benchmark and state of the art Year: (2017)
Ref_id:b51 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b52 Title: Finegrained visual classification of aircraft Year: (2013)
Ref_id:b53 Title: Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation Year: (2023)
Ref_id:b54 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b55 Title: SDXL: improving latent diffusion models for high-resolution image synthesis Year: (2024)
Ref_id:b56 Title: MiLoRA: Harnessing minor singular components for parameter-efficient LLM finetuning Year: (2025-04)
Ref_id:b57 Title: On the efficiency of adapters in transformers Year: (2021-11)
Ref_id:b58 Title: AdapterFusion: Non-destructive task composition for transfer learning Year: (2021-04)
Ref_id:b59 Title: DyLoRA: Parameterefficient tuning of pre-trained models using dynamic search-free low-rank adaptation Year: (2023-05)
Ref_id:b60 Title: Parameter-efficient fine-tuning with discrete fourier transform Year: (2024)
Ref_id:b61 Title: LoRAdrop: Efficient LoRA parameter pruning based on output evaluation Year: (2025-01)
Ref_id:b62 Title: DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution Year: (2024-08)
Ref_id:b63 Title: PiSSA: Principal singular values and singular vectors adaptation of large language models Year: (2024)
Ref_id:b64 Title: Imagenet-21k pretraining for the masses Year: ()
Ref_id:b65 Title: Scaling & shifting your features: A new baseline for efficient model tuning Year: ()
Ref_id:b66 Title: Counterinterference adapter for multilingual machine translation Year: (2021-11)
Ref_id:b67 Title: Conditional adapters: Parameter-efficient transfer learning with fast inference Year: (2023)
Ref_id:b68 Title: Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks Year: (2021-08)
Ref_id:b69 Title: P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks Year: (2021)
Ref_id:b70 Title: Parameter-efficient transfer learning with diff pruning Year: (2021-08)
Ref_id:b71 Title: Neural architecture search for parameter-efficient fine-tuning of large pre-trained language models Year: (2023-07)
Ref_id:b72 Title: Parameter-efficient fine-tuning without introducing new latency Year: (2023-07)
Ref_id:b73 Title: Unified low-resource sequence labeling by sample-aware dynamic sparse finetuning Year: (2023-12)
Ref_id:b74 Title: Composable sparse fine-tuning for cross-lingual transfer Year: (2022-05)
Ref_id:b75 Title: HFT: half fine-tuning for large language models Year: (2025-08-01)
Ref_id:b76 Title: Training neural networks with fixed sparse masks Year: (2021)
Ref_id:b77 Title: Raise a child in large language model: Towards effective and generalizable fine-tuning Year: ()
Ref_id:b78 Title: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing Year: (2021-11)
Ref_id:b79 Title: Sensitivity-aware visual parameter-efficient fine-tuning Year: (2023)
Ref_id:b80 Title: Vector-based random matrix adaptation Year: (2024)
Ref_id:b81 Title: When moe meets llms: Parameter efficient fine-tuning for multi-task medical applications Year: (2024)
Ref_id:b82 Title: Mixture-of-LoRAs: An efficient multitask tuning method for large language models Year: (2024-05)
Ref_id:b83 Title: Mixture of loRA experts Year: (2024)
Ref_id:b84 Title: Badam: A memory efficient full parameter optimization method for large language models Year: (2024-12-10)
Ref_id:b85 Title: LISA: layerwise importance sampling for memory-efficient large language model fine-tuning Year: (2024)
Ref_id:b86 Title: Outlier-weighed layerwise sampling for LLM fine-tuning Year: (2025-08-01)
Ref_id:b87 Title: Galore: Memory-efficient LLM training by gradient low-rank projection Year: (2024)
Ref_id:b88 Title: UniPELT: A unified framework for parameter-efficient language model tuning Year: (2022-05)
Ref_id:b89 Title: Parameterefficient fine-tuning design spaces Year: (2023)
Ref_id:b90 Title: Neural prompt search Year: (2024)
Ref_id:b91 Title: AutoPEFT: Automatic configuration search for parameter-efficient fine-tuning Year: (2024)
Ref_id:b92 Title: DoRA: Weight-decomposed low-rank adaptation Year: (2024)
Ref_id:b93 Title: CA-loRA: Adapting existing loRA for compressed LLMs to enable efficient multi-tasking on personal devices Year: (2024)
Ref_id:b94 Title: Hydralora: An asymmetric lora architecture for efficient fine-tuning Year: ()
Ref_id:b95 Title: Improved low-rank adaptation by a nonlinear transformation Year: (2024-11)
Ref_id:b96 Title: Efficient learning with sine-activated low-rank matrices Year: (2025)
Ref_id:b97 Title: Loda: Low-dimensional adaptation of large language models Year: (2023)
Ref_id:b98 Title: Neat: Nonlinear parameter-efficient adaptation of pre-trained models Year: (2025)
Ref_id:b99 Title: Cola: Compute-efficient pre-training of llms via low-rank activation Year: (2025)
Ref_id:b100 Title: Topics in matrix analysis Year: (1994)
