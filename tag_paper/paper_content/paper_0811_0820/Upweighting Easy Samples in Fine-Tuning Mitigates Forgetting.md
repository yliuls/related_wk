Title: Upweighting Easy Samples in Fine-Tuning Mitigates Forgetting
Abstract: Fine-tuning a pre-trained model on a downstream task often degrades its original capabilities, a phenomenon known as "catastrophic forgetting". This is especially an issue when one does not have access to the data and recipe used to develop the pre-trained model. Under this constraint, most existing methods for mitigating forgetting are inapplicable. To address this challenge, we propose a sample weighting scheme for the fine-tuning data solely based on the pre-trained model's losses. Specifically, we upweight the easy samples on which the pre-trained model's loss is low and vice versa to limit the drift from the pre-trained model. Our approach is orthogonal and yet complementary to existing methods; while such methods mostly operate on parameter or gradient space, we concentrate on the sample space. We theoretically analyze the impact of fine-tuning with our method in a linear setting, showing that it stalls learning in a certain subspace, which inhibits overfitting to the target task. We empirically demonstrate the efficacy of our method on both language and vision tasks. As an example, when fine-tuning Gemma 2 2B on MetaMathQA, our method results in only a 0.8% drop in accuracy on GSM8K (another math dataset) compared to standard finetuning, while preserving 5.4% more accuracy on the pre-training datasets.

Section: Introduction
In the modern era of large-scale machine learning, one of the central goals is to design models capable of performing * Equal contribution. Rudrajit was a PhD student at the University of Texas at Austin when this work was done. 1 University of Texas at Austin 2 Google Research. Correspondence to: Sunny Sanyal <sanyal.sunny@utexas.edu>, Hayden Prairie <haydenprairie@utexas.edu>, Rudrajit Das <dasrudrajit@google.com>, Ali Kavis <kavis@austin.utexas.edu>, Sujay Sanghavi <sanghavi@mail.utexas.edu>.
Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s). multiple tasks. Traditionally, this is achieved by training an appropriately large model over datasets of multiple tasks, ensuring that the model jointly learns multiple tasks at once. Unfortunately, it is not viable to repeat this process with every new additional task due to the scale of contemporary models, necessitating effective strategies that can essentially learn without full retraining. A resource-efficient convention in machine learning is to take a pre-trained model which is trained on some vast and diverse dataset, and fine-tune it on a new dataset/task. Such pre-trained models are typically large and expensive to train from scratch but perform well on a variety of tasks while offering a versatile basis for learning a new task.
Fine-tuning is a delicate process that should ideally serve multiple objectives simultaneously; we would like to use the base model and its capabilities to facilitate learning a strong model on the downstream task, and in the meantime, preserve the existing abilities of the pre-trained model. On this particular front, the major challenge in standard, unregulated fine-tuning is the catastrophic forgetting phenomenon. In broad terms, it describes the performance decline of the pre-trained model on previously observed data/tasks after fine-tuning on a new one. When the learning process for the downstream task interferes with the previously-learned representations beyond tolerable margins, the pre-trained model loses its prior capabilities and significantly underperforms on previously-learned tasks.
Mitigating catastrophic forgetting is an active area of research with many fundamental questions awaiting solutions.
The key idea is to constrain the fine-tuning process to prevent the degeneration of the learned representations while guiding the learning of the new task to augment existing capabilities. The literature on the topic offers various approaches based on the available knowledge pertaining to the pre-training process. In fact, pre-training-specific data availability and how it is treated predominantly dictates the success of mitigating forgetting. In many real-life scenarios, however, the data and the training recipe used for generating the pre-trained model are not available (Radford et al., 2021;Touvron et al., 2023a;b;Grattafiori et al., 2024;Jiang et al., 2023). Naturally, one needs to approach the forgetting phenomenon accordingly to design realistic methods.
Therefore, we focus on the case in which we have no ac-cess to the pre-training-specific information during the finetuning process; we call it the data-oblivious setting. The only piece of information available during fine-tuning is indeed the pre-trained model. Therefore, one needs to devise a strategy to regulate and guide the fine-tuning process to preserve the pre-trained model capabilities while learning the new task in the absence of prior knowledge. Under this challenging setting, we present an answer to the question:
Can we design a principled method that mitigates forgetting during fine-tuning in the data-oblivious setting?
In this paper, we propose Fine-tuning with Pre-trained Loss-Oriented Weighting (FLOW) to mitigate catastrophic forgetting in the data-oblivious setting. Our key insight is upweighting the "easy" samples on which the pre-trained model's loss is low and vice versa. We believe that boosting the samples on which the pre-trained model performs well (i.e., has low loss) will introduce supervised bias to the gradient updates in favor of the pre-trained model. Intuitively, this will prevent the parameters from deviating too much from the initial pre-trained state, thus mitigating forgetting.
Some prior papers assign more importance to samples with larger losses to accelerate the training process (Loshchilov & Hutter, 2015;Shrivastava et al., 2016;Katharopoulos & Fleuret, 2017;Kawaguchi & Lu, 2020;Das et al., 2024). We follow the reciprocal reasoning; we tweak the fine-tuning process in favor of the pre-trained model by assigning larger weights to samples with smaller pre-trained loss values. We elaborate on this while stating our contributions next. 1. To mitigate forgetting, we propose FLOW, which fine-tunes the pre-trained model using a sample-wise weighted loss. Inspired by robust optimization ideas, we derive the i th sample's weight to be exp(-ℓ i /τ ), where ℓ i is the i th sample's pre-trained loss and τ is a parameter which we set as median(ℓ i ) in practice. Thus, our method is essentially parameter-free.
2. We demonstrate the superiority of FLOW over relevant baselines (model averaging, ℓ 2 regularization, LoRA, etc.) in both vision and language model experiments.
For instance, ResNet-50 fine-tuned with FLOW on six image classification datasets achieves ∼ 17% higher average accuracy (over pre-training and fine-tuning data) than standard fine-tuning, while also surpassing other relevant baselines (see Table 1). When finetuning Gemma 2 2B on math datasets, the corresponding improvement of FLOW over standard fine-tuning is ∼ 4% (see Table 2).
3. We also empirically show that combining FLOW with existing methods for mitigating forgetting improves the performance of the base methods (see Tables 3 and 4).
this section cite: ['b73', 'b37', 'b59', 'b85', 'b41', 'b43', 'b18']

Section: 4.
We theoretically analyze the effect of fine-tuning with
Standard FT 2-Reg. WiSE-FT LP FLOW (Ours) 10 20 30 40 50 60 70 80 90 Accuracy (%) 64.34 68.18 71.52 77.73 81.17 Fine-tuning Acc. (Avg. over 6 tasks) Pre-training Acc. (Top-1 Imagenet-1K) Average Figure 1. FLOW versus standard fine-tuning (FT) and relevant baselines for a ResNet-50 model pre-trained on ImageNet-1K (from Table 1). FLOW achieves the best average accuracy (between pre-training and target fine-tuning accuracies).
FLOW for linear models. In particular, the covariance matrix of the fine-tuning data weighted by FLOW has a small eigenvalue and training is stalled along the corresponding eigenvector, impeding overfitting to the fine-tuning task (see Remark 7.4).
We end this section with a preview of the comparison of our method FLOW with some relevant baselines (in the data-oblivious setting) in Figure 1.
this section cite: []

Section: Related Work

this section cite: []

Section: Mitigating Catastrophic Forgetting
We begin by summarizing the vast literature on catastrophic forgetting with a focus on prior works most relevant to our proposed setting. For a streamlined presentation, we survey prior work in two settings -data-aware and data-oblivious. Due to space limitations, we refer the readers to Appendix A for a more detailed and explanatory review of the literature.
this section cite: []

Section: DATA-AWARE APPROACHES
The majority of the approaches for mitigating forgetting assume task-specific knowledge access to different extents; either (a subset of) the pre-training dataset itself or some information/statistic computed from pre-training data. Below, we describe the data-aware approaches based on how they make use of task-specific knowledge.
Regularization-based methods. This line of work aims to preserve existing capabilities by keeping the parameters close to the pre-trained model. The key idea is to introduce task-specific regularization to penalize modifications along the "important" directions for the old tasks (Ahn et al., 2019). Kirkpatrick et al. (2016) introduces the elastic weight consolidation (EWC) algorithm, which estimates the important directions by approximating the Fisher information matrix. Several variants of EWC have been proposed (Schwarz et al., 2018;Ritter et al., 2018;Lee et al., 2020;Liu et al., 2018). Zenke et al. (2017); Aljundi et al. (2018) infer the importance of each parameter by their variational effect on the outputs. In a similar spirit, Lee et al. (2017) aims to match the posteriors of the pre-trained and fine-tuned models.
Optimization-driven methods. Another perspective to mitigating forgetting is guiding the optimization process by constraining the algorithms directly as opposed to manipulating the loss function. The core idea is to keep track of "important directions" for the old tasks, and train on the new task "orthogonally." This could be done by storing prior data samples or gradients in a buffer (Lopez-Paz & Ranzato, 2017;Farajtabar et al., 2020;Chaudhry et al., 2019a) or by incrementally expanding the subspace of important directions without storing task-specific information (Zeng et al., 2019;Wang et al., 2021;2023b).
this section cite: ['b0', 'b45', 'b83', 'b78', 'b49', 'b57', 'b109', 'b2', 'b51', 'b58', 'b23', 'b95']

Section: Replay-based methods.
A more direct approach is to store old task samples in buffers and introduce them into the training process for the new task to refresh task-specific representations periodically. There are several components to such methods. Some prior work focus on data selection based on the nature of old data access (Rebuffi et al., 2017;Aljundi et al., 2019;Bang et al., 2021;Chaudhry et al., 2019b;Isele & Cosgun, 2018;De Lange & Tuytelaars, 2021;Borsos et al., 2020;Tiwari et al., 2021) (e.g., streaming versus on-demand). Another important perspective is the re-introduction strategy of the stored information into the fine-tuning process (Silver & Mercer, 2002;Li & Hoiem, 2016;Triki et al., 2017;Lee et al., 2019b;Dhar et al., 2019;Rebuffi et al., 2017;Riemer et al., 2019;Chaudhry et al., 2019b;De Lange & Tuytelaars, 2021;Tiwari et al., 2021).
this section cite: ['b76', 'b3', 'b5', 'b12', 'b36', 'b19', 'b9', 'b86', 'b55', 'b91', 'b20', 'b76', 'b77', 'b12', 'b19']

Section: Architecture-driven methods.
Another technique to limit interference between tasks is to allocate a separate trainable set of parameters per task. This could be done by initializing a sub-networks per new task (Rusu et al., 2016;Aljundi et al., 2017;Collier et al., 2020;Rajasegaran et al., 2019;Ramesh & Chaudhari, 2021;Wang et al., 2023a;2022a), gradually expanding the parameters of a base network (Yoon et al., 2018;Ostapenko et al., 2019;Hung et al., 2019), or segregating a fixed model into task-specific subsets (Mallya et al., 2018;Kang et al., 2022;Serra et al., 2018;Wortsman et al., 2020;Mallya & Lazebnik, 2017;Mustafa B Gurbuz, 2022;Jung et al., 2020). The main downside with this line of work is that task identities must be known for inference to (de)activate relevant sub-networks (Aljundi et al., 2017).
this section cite: ['b80', 'b1', 'b17', 'b74', 'b75', 'b105', 'b69', 'b34', 'b40', 'b84', 'b101', 'b61', 'b66', 'b39', 'b1']

Section: DATA-OBLIVIOUS APPROACHES
In the less-explored data-oblivious setting, it is particularly challenging to devise a principled approach, as there is no access to any data-specific information, except for the pre-trained model. One line of work explores the simple idea of "model averaging" (MA) which essentially does a convex combination of the parameters of the pre-trained model and that of the fully fine-tuned model for the new task.
MA and more sophisticated model merging variants have been studied in relevant context to forgetting (Lubana et al., 2021;Wortsman et al., 2021;Ilharco et al., 2023;Lin et al., 2023;Kleiman et al., 2025). Some recent works Chen et al. (2024b); Panda et al. (2024) introduce different strategies to selectively update a subset of parameters in a pre-training data-agnostic manner. Finally, Biderman et al. (2024) has shown that LoRA (Hu et al., 2022) could be effective for mitigating catastrophic forgetting in transformers. Unlike the methods discussed above which focus on the parameter or gradient space, ours focuses on the sample space.
this section cite: ['b60', 'b102', 'b35', 'b56', 'b46', 'b70', 'b7', 'b33']

Section: Sample Selection and Weighting
Sample-wise importance selection/weighting has been studied in optimization papers (Needell et al., 2014;Zhao & Zhang, 2015;Alain et al., 2015;Stich et al., 2017) and ML papers (Loshchilov & Hutter, 2015;Shrivastava et al., 2016;Katharopoulos & Fleuret, 2017;2018;Kawaguchi & Lu, 2020;Das et al., 2024) to speed up the optimization/training process by reducing the variance of the gradient updates. Such papers advocate focusing on "hard" samples with highgradient norms or losses. In contrast, we focus on "easy" samples to mitigate forgetting. Another line of work focuses on robust learning under uncertain data distributions. Distributionally robust optimization (DRO) proposes to minimize the worst-case weighted loss, where the sample weights are constrained or regularized (Ben-Tal et al., 2013;Levy et al., 2020;Duchi & Namkoong, 2021;Qi et al., 2021). Some recent works (Xie et al., 2024;Chen et al., 2024a;Anonymous, 2025) propose dynamic sample-weighting strategies for LLM training based on the previously discussed ideas.
this section cite: ['b67', 'b110', 'b87', 'b59', 'b85', 'b41', 'b43', 'b18', 'b6', 'b52', 'b22', 'b72', 'b104']

Section: Notation and Definitions
1(.) denotes the indicator variable. For any n ∈ N, the set {1, . . . , n} is denoted by [n]. Vectors and matrices are in lowercase and uppercase bold font, respectively. The ℓ p norm of a vector v is denoted by ∥v∥ p . The inner product between two vectors v and v ′ is denoted as ⟨v,
v ′ ⟩. A set of n linearly independent n-dimensional vectors {u 1 , . . . , u n } is said to be an orthonormal basis for R n if ⟨u i , u j ⟩ = 1(i = j). A vector v = [v 1 , . . . , v n ] ⊤ is said to belong to the n-dimensional probability simplex ∆ n if n i=1 v i = 1 and v i ≥ 0 ∀ i ∈ [n].
For any n ∈ N, I n denotes the identity matrix of dimension n.
this section cite: []

Section: Proposed Algorithm
Our proposed algorithm consists of two main steps: (i) computing weights for the samples based on their respective pre-trained loss values; and (ii) fine-tuning with a weighted loss wherein the per-sample losses are scaled by their respec-tive weights. The sample-wise weights are computed once and used throughout the entire fine-tuning process. We formally state our proposed fine-tuning protocol in Algorithm 1 and delve into its design details in the sequel.
this section cite: []

Section: Algorithm 1 Fine-tuning with Pre-trained Loss-Oriented Weighting (FLOW)
Input: Pre-trained model θ * , dataset {(x i , y i )} n i=1 for the new task, and temperature parameter τ . f i (θ) → i th sample's loss at θ, with a non-negative loss function (e.g., cross-entropy loss).
this section cite: []

Section: Compute sample weights: w
i = exp -fi(θ * ) τ . 2. Weighted loss: L(θ) = n i=1 w i f i (θ).
3. Fine-tune with weighted loss: θ * := arg min θ L(θ).
Output: Fine-tuned model θ * . Remark 4.1. Depending on the setting, our model might have task-specific components, such as per-task prediction heads (e.g., in vision). Algorithm 1 can be slightly modified in the presence of task-specific components to enhance performance. Refer to Appendix B for these modifications.
this section cite: []

Section: Remark 4.2.
As a heuristic prescription, we set τ = median (f i (θ * )) in all our experiments (unless otherwise stated), which leads to consistently good performance. Thus, our algorithm is essentially parameter-free in practice.
Algorithm design. Our main intuition is that we can control forgetting by not drifting away too much from the pretrained model (i.e., θ * ) during fine-tuning. In the presence of pre-training data, this is done by introducing datadependent constraints on the parameter space or gradient space. Since we have no access to pre-training data, we redirect our focus towards strategies on the sample space depending only on the pre-trained model.
To that end, we propose to infer the easiness of each sample of the fine-tuning dataset with respect to the pre-trained model, based on the per-sample losses f i (θ * )'s (see Alg. 1). We say that the i th sample is "easy" if f i (θ * ) is "small". 1 Intuitively, prioritizing the "easy" samples during fine-tuning would limit the drift from θ * . On the other hand, overfocusing on the "easy" samples would probably lead to poor performance on the fine-tuning task. Thus, it is important to strike a balance.
Let us formalize these ideas mathematically. For fine-tuning on the new task, let us consider the objective function L π (θ) = n i=1 π i f i (θ), where π = [π 1 , . . . , π n ] ⊤ is a static design-choice ∈ ∆ n (i.e., n i=1 π i = 1 and π i ≥ 0 ∀ i ∈ [n]) which we allow to only depend on the pre-trained 1. for all i ̸ = j such that f i (θ * ) ≤ f j (θ * ), π i ≥ π j , 2. π does not concentrate around one or a few samples but rather spreads uniformly over the samples.
These two requirements can be enforced by minimizing the following function (w.r.t. π) involving negative entropic regularization:
g(π) = n i=1 π i f i (θ * ) + τ n i=1 π i log π i .(1)
Here τ > 0 is a parameter controlling the extent of the second requirement which is facilitated by the entropy term. We now state the minimizer of g(π) (proof is in Appendix C).
Proposition 4.3. Let π * = [π * 1 , . . . , π * n ] ⊤ = arg min π∈∆n g(π).
Then we have
π * i = 1 Z exp -fi(θ * ) τ
, where Z is the normalizing factor.
Modulo the normalizing factor Z (it does not matter when optimizing w.r.t. θ), note that w i and L(θ) in Algorithm 1 are equivalent to π * i and L π * (θ), respectively. Distributionally robust optimization (DRO) perspective. Our formulation above is motivated by prior work on DRO (Qi et al., 2021), but it is exactly the opposite of DRO in spirit. Specifically, in our setting, Qi et al. (2021) consider the following min-max problem:
min θ max π∈∆n n i=1 π i f i (θ) -τ n i=1 π i log π i .(2)
The first term in Eq. ( 2) is the worst-case weighted loss at θ, while the second term (i.e., entropic regularization) promotes uniform weights. The optimal solution to the inner max function w.r.t. π turns out to be π * i ∝ exp fi(θ) τ . Note that this is essentially the inverse of our weighting function (modulo the normalizing factor) because it assigns a higher weight to samples with larger losses (i.e., the "hard" samples). The weighting function of DRO would be very conducive to forgetting because it focuses more on the "hard" samples. Further, our weighting function is static (or one-shot) as it depends only on the losses at θ * . On the other hand, the weighting function of DRO is dynamic (i.e., it depends on the current point θ). In fact, after plugging in the optimal value of π into Eq. ( 2) and simplifying, the DRO objective reduces to min θ n i=1 exp fi(θ)   τ ; this is noticeably different from our objective L(θ) in Algorithm 1.
this section cite: ['b72', 'b72']

Section: Experimental Setup
We empirically evaluate the performance of FLOW (Algorithm 1) on vision and language tasks, showcasing its effectiveness across different model architectures and modalities. 2 Here, we explain details of our experiments: baselines, model architectures, datasets, and evaluation metrics.
Baselines. In our language and vision experiments, we compare FLOW against relevant baselines in the dataoblivious setting, namely, standard fine-tuning (fine-tuning with vanilla unweighted loss), ℓ 2 -regularization [following Kirkpatrick et al. (2016)], and WiSE-FT (Wortsman et al., 2021) (model averaging of pre-trained and standard finetuned models). Additionally, we compare against linear probing (fine-tuning only the classification head, keeping the body frozen) and low-rank adaptation (LoRA) (Hu et al., 2022) in language experiments. More details on the baselines can be found in Appendix G.1.
this section cite: ['b45', 'b102', 'b33']

Section: Vision Experiments
We compare the performance of FLOW and associated baselines in a transfer learning setup.
this section cite: []

Section: Models. We experimented with ResNet-18 and ResNet-50 (Wightman et al., workshop) pre-trained on Imagenet-1K (IN-1K).
Datasets. We used seven widely-used image classification datasets: CIFAR-10 (Krizhevsky, 2009), CIFAR-100 (Krizhevsky, 2009), Flowers102 (Nilsback & Zisserman, 2008), Caltech101 (Li et al., 2022), Cars (Krause et al., 2013), and Dogs (Parkhi et al., 2012).
Evaluation metrics. Vision models are trained with taskspecific parts, such as classification head (head) and batchnorm (BN); see Appendix B for how FLOW works with with task-specific parts. Forgetting is measured by how much the model's top-1 validation accuracy on ImageNet-1K (subsequently referred to as IN-1K accuracy) reduces after fine-tuning. We report the fine-tuning performance in terms of average fine-tuning accuracy over all the related datasets following (Goyal et al., 2023;Ilharco et al., 2023). For IN-1K evaluation after fine-tuning, we replace the task-specific components of the fine-tuned model with their pre-trained counterparts. An extended discussion on experimental details, evaluation, and hyper-parameters are in Appendix G.4. We also report the average of IN-1K accuracy and averaged fine-tuning accuracy for each method; this is a reasonable unified metric to evaluate the performance of a method jointly on the pre-training and fine-tuning data. 2 Our code is publicly available here.
this section cite: ['b48', 'b68', 'b54', 'b47', 'b71', 'b27', 'b35']

Section: Language Model Experiments
We follow a similar setup to Biderman et al. (2024); Chen et al. (2024b), where a language model's general capabilities are evaluated before and after fine-tuning on a mathematical reasoning dataset. All training for language experiments is done with HuggingFace peft (Mangrulkar et al., 2022), transformers (Wolf et al., 2020), datasets (Lhoest et al., 2021), and accelerate (Gugger et al., 2022).
Models. We use Gemma 2 2B (Team et al., 2024)  and Llama 3.2 3B ( Grattafiori et al., 2024) as our base language models. Further details on training hyper-parameters can be found in Appendix G.2.
Datasets. Following previous work (Biderman et al., 2024;Chen et al., 2024b), we fine-tune on MetaMathQA (Yu et al., 2023), a mathematical reasoning dataset that is bootstrapped from the training set of GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b) using a LLM. We train with all 395K samples in MetaMathQA.
Evaluation metrics. To evaluate the validity of FLOW, we break down our metrics into general capability and target fine-tuning evaluations. To evaluate general capabilities, we again follow a similar setup to Chen et al. (2024b), where we use commonsense reasoning, 5-shot MMLU (Hendrycks et al., 2021a), and 3-shot MBPP (Austin et al., 2021) metrics. To evaluate the target domain, we use 5-shot GSM8K (Cobbe et al., 2021). All evaluations are performed with lm-evaluation-harness (Gao et al., 2024). More details on evaluation and the commonsense metric can be found in Appendix G.3. Similar to vision, we also report the average of general capabilities and the target fine-tuning accuracies as a unified metric.
this section cite: ['b7', 'b63', 'b53', 'b29', 'b88', 'b7', 'b106', 'b16', 'b4', 'b16', 'b122']

Section: Experimental Results

this section cite: []

Section: Comparing FLOW and Related Baselines
For our vision experiments on ResNets, Table 1 lists the accuracies of all the baselines and FLOW. We observe similar trends for both ResNet models, so we discuss the results for the larger ResNet-50 model here. The pre-trained ResNet-50 model achieves a top-1 accuracy of 79.02% on ImageNet-1K's validation set. Standard fine-tuning experiences a significant 42.11% drop in IN-1K accuracy, while achieving an average fine-tuning accuracy of 91.78% across the target datasets. In contrast, FLOW suffers only a 2.93% drop in IN-1K accuracy and exhibits a reasonable 86.25% average accuracy on target fine-tuning datasets, demonstrating a significant improvement over standard fine-tuning. Overall, FLOW's average on IN-1K and target domain accuracy is 16.83% higher than standard fine-tuning.
Going beyond standard fine-tuning, our results in Table 1 show that FLOW comprehensively outperforms other base- Our language model results are in Table 2. Results for Gemma 2 2B show that FLOW helps preserve (and even somewhat enhance) the general capabilities of the pretrained model. Specifically, compared to standard finetuning, FLOW improves general capability accuracy by 2.52% in commonsense reasoning, 3.73% in MMLU, and 10.00% in MBPP, with a minor degradation of 0.83% in GSM8K. We see a similar trend in our Llama 3.2 3B experiments. Furthermore, while alternative baselines show specific strengths (such as WiSE-FT's general capability performance and ℓ 2 -regularization's target fine-tuning performance), FLOW outperforms all baselines, on average, for both models, striking the best balance between preserving general capabilities and achieving good target fine-tuning performance. Additional details on commonsense reasoning results are in Appendix I.1 and an ablation for our choice of sample weighting in LLMs is in Appendix I.2.
In summary, FLOW strikes a good balance between learning a new task and retaining knowledge from pre-training.
this section cite: []

Section: Combining FLOW with Baselines
To complement our results in Tables 1 and 2, we investigate the performance of baselines when combined with FLOW.
In the vision setting, we consider uniform model averaging with WiSE-FT (with α = 0.5) and report its performance with and without FLOW in Table 3. Interestingly, averaging the pre-trained IN-1K model and the fine-tuned model obtained with FLOW improves over standard WiSE-FT (i.e., averaging the pre-trained IN-1K model and the standard fine-tuned model) by 4.18% and 4.52% for ResNet-18 and ResNet-50, respectively, in average performance.
Further, as seen in Table 4, FLOW boosts the performance of other baselines in language modeling. When combined with ℓ 2 -regularization, we observe improvements in general capability between 0.5% and 1.80%, with only a 0.83% reduction in GSM8K performance. Furthermore, the integration of FLOW with LoRA yields even stronger results, enhancing general capability performance by 1.07% to 3.40%, while simultaneously improving GSM8K performance by 1.06%. Further details and discussion combining FLOW with ℓ 2 -regularization and LoRA are in Appendix I.1.
this section cite: []

Section: Theoretical Analysis
Here we consider linear pre-training and fine-tuning tasks 3and theoretically analyze the effect of fine-tuning with our proposed method FLOW (Alg. 1). Specifically, we compare the non-asymptotic trajectories of FLOW and vanilla finetuning. A key insight of our analysis is that FLOW stalls training in a certain direction, impeding overfitting to the fine-tuning task (see Remark 7.4). We also demonstrate that FLOW goes beyond the simple idea of model averaging (see Remark 7.5).
We begin by describing the problem setting.
Pre-training task: The label y ∈ R for a d-dimensional data point x ∼ P is given by y = ⟨θ * , x⟩, where θ * ∈ R d is the ground-truth model. Let D denote the joint distribution of (x, y), where x ∼ P. Let Σ = E x∼P xx ⊤ be the data covariance matrix. Without loss of generality, let Σ ⪰ I d .
Fine-tuning task: The label y ∈ R for a d-dimensional data point x ∼ P is given by y = θ * , x , where θ * ∈ R d is the ground-truth model. Let D denote the joint distribution of ( x, y), where x ∼ P. Also, let e := θ * -θ * , e := e/∥e∥ 2 , and e ⊥ be a unit vector orthogonal to e. We consider the case of P = N ( ⃗ 0 d , Σ), where
Σ = I d + ρ ee ⊤ ⊥ + e ⊥ e ⊤ ,(3)
where ρ ∈ [0, 1) is a constant. Note that Σ is the data covariance matrix here. Remark 7.1 (Regarding Σ). We study the case of Σ as given in Eq. ( 3) because it is the minimal analytically tractable case where we can show that FLOW goes beyond model averaging (MA) (see Remark 7.5). Specifically, if ρ = 0 and Σ = I d , then FLOW reduces to MA. Moreover, for an arbitrary Σ, characterizing the eigen-spectrum of the matrix dictating the trajectory of FLOW becomes intractable. For the analysis to be tractable, we need some relationship between Σ and e (i.e., the difference between the optima of the pre-training and fine-tuning tasks). 4For a model parameterized by θ ∈ R d , let
err 1 (θ) := E D y -⟨θ, x⟩ 2 = θ -θ * ⊤ Σ θ -θ * , err 2 (θ) := E D y-⟨θ, x⟩ 2 = θ-θ * ⊤ Σ θ-θ * (4
)
be the population errors on the pre-training and fine-tuning tasks, respectively. Also, the total error with θ on the two tasks is denoted by err tot (θ) = err 1 (θ) + err 2 (θ).
We assume that initially, we learn θ * with the pre-training data; so θ * is our pre-trained model. Note that
err tot (θ * ) = err 2 (θ * ) = e ⊤ Σe = ∥e∥ 2 2 ,(5)
where the last step follows by using Σ from Equation ( 3).
We start fine-tuning starting from θ * . Specifically, we assume access to the population( x, y) ∼ D of the fine-tuning task, but we lose access to the pre-training data.
Vanilla fine-tuning (FT): We minimize err 2 (θ) (Eq. ( 4)) with gradient descent (GD) starting from θ * using a constant learning rate η. Our iterate θ K at the K th iteration is given by (using the value of Σ from Eq. ( 3) and θ * -θ * = e):
θ K = θ * + I d -2η I d + ρ ee ⊤ ⊥ + e ⊥ e ⊤ K e. (6
)
FLOW: For some temperature τ , the weight of (
x, y) ∼ D is w( x, y) = exp -( y-⟨θ * , x⟩) 2 τ . We minimize err 2 ( θ) := E D w( x, y) y -⟨ θ, x⟩ 2 ,(7)
with GD starting from θ * using a constant learning rate η. Suppose our iterate at the K th iteration is θ K .
Theorem 7.2 (FLOW). Let µ = τ τ +2∥e∥ 2 2 1/2
. Then:
θ K = θ * + I d -2 η Σ ′ K e,(8)
where
Σ ′ = µ I d -Q with Q = (1 -µ 2 )ee ⊤ + ρ 2 (1 -µ 2 )e ⊥ e ⊤ ⊥ -ρµ 2 ee ⊤ ⊥ + e ⊥ e ⊤ . (9
)
We prove Thm. 7.2 in Appendix D. The main technical challenge is the evaluation of Σ ′ , viz., the covariance matrix of the weighted fine-tuning data; see Lemma F.1 for this.
Now, we are going to compare vanilla FT (6) with η = 1 2 and FLOW (8) with η = 1 2µ . We believe these are comparable learning rates for vanilla FT and FLOW because the resultant matrices (Eqs. (10) and ( 11)) dictating the convergence of both methods have exactly two non-zero eigenvalues and the corresponding eigenvectors lie in the span of e and e ⊥ . Plugging in η = 1 2 into Eq. ( 6), we get:
θ K = θ * + P K e, with P = -ρ ee ⊤ ⊥ + e ⊥ e ⊤(10)
for vanilla FT. Plugging in η = 1 2µ into Eq. ( 8), we get:
θ K = θ * + Q K e,
with Q given by Eq. ( 9) (11)
for FLOW. The non-zero eigenvalues of P are ∓ρ and the corresponding eigenvectors are 1 √ 2 e ± e ⊥ . Using this in (10) and simplifying, we get for vanilla FT:
θK = θ * + ρ K 1 K is even e -1 K is odd ∥e∥2e ⊥ . (12)
Remark 7.3 (Vanilla FT). Since ρ < 1, θ K converges to θ * rapidly, and we cannot impede this convergence.
Note that (we use Σ ⪰ I d below):
err tot ( θ * ) = err 1 ( θ * ) = e ⊤ Σe ≥ ∥e∥ 2 2 .(13)
On the other hand, the non-zero eigenvalues and corresponding eigenvectors of Q are not as straightforward to compute. We do this computation in Lemma F.3 with the re-parameterization of µ = β(1-ρ 2 ) (1+β)(1-βρ 2 ) for some β ∈ (0, 1]. 5 Using this in Eq. ( 11) and simplifying, we get for FLOW:
θ K = θ * + λ K 1 + λ K 2 β 2 ρ 2 1 + β 2 ρ 2 e-βρ λ K 1 -λ K 2 1 + β 2 ρ 2 ∥e∥e ⊥ ,(14)
where λ 1 = 1+βρ 2 1+β and λ 2 = ρ 2 1-β 1-βρ 2 . Remark 7.4 (FLOW's trajectory). Note that we can control λ 1 by varying β. Specifically, we can make λ 1 arbitrarily close to 1 by choosing a small enough β. On the other hand, 1-β  1-βρ 2 < 1+βρ 2 1+β = λ 1 and so, λ 2 < ρ 2 λ 1 . Hence, beyond a certain number of iterations K, Eq. ( 14) becomes:
θ K ≈ θ K := θ * + γ(K, β) e -βρ∥e∥ 2 e ⊥ ,(15)
with γ(K, β) :=
λ K 1 1+β 2 ρ 2 .
Because we can control λ 1 by varying β, we can control γ(K, β). Thus, we can stall convergence along e -βρ∥e∥ 2 e ⊥ , 6 impeding the convergence of θ K to θ * .
this section cite: []

Section: Remark 7.5 (FLOW goes beyond model averaging).
If we perform model averaging between θ * and θ * with parameter ω ∈ [0, 1], then our averaged model is:
θ avg (ω) = ωθ * + (1 -ω) θ * = θ * + ωe.(16)
Comparing the above with Eq. ( 15), we see that FLOW goes beyond model averaging because of the component along e ⊥ . But we can make θ K (Eq. ( 15)) → θ avg (ω) by choosing β → 0 and K such that γ(K, β) → ω. So, we expect FLOW to be at least as powerful as model averaging.
As per Lemma F.4, the minimum total error on both tasks with optimally tuned model averaging is given by:
min ω∈[0,1] err tot θ avg (ω) = e ⊤ Σe e ⊤ Σe + 1 ∥e∥ 2 2 < ∥e∥ 2 2 ,(17)
where recall that Σ is the covariance matrix of the pretraining data. On the other hand, using Eqs. ( 5) and (13) min err tot (θ * ), err tot ( θ * ) = ∥e∥ 2  2 .
(18) 5 The corresponding temperature is τ = 2β(1-ρ 2 )∥e∥ 2 2 (1-β 2 ρ 2 ) . 6 This direction is the top eigenvector of Q. Since Σ ′ = µ I d -Q , this is also the eigenvector of Σ ′ with the smallest eigenvalue. 17) and ( 18), we see that optimally tuned model averaging attains a smaller total error than both θ * (i.e., the pre-trained model) and θ * to which vanilla FT converges rapidly (Remark 7.3). More importantly, following our discussion in Remark 7.5, we conclude that optimally tuned FLOW's total error is at least as good as the one in Eq. ( 17).
this section cite: []

Section: Remark 7.6 (Error comparison). By comparing Eqs. (

this section cite: []

Section: Conclusion
In this paper, we studied the problem of catastrophic forgetting in pre-trained models during fine-tuning when we do not have access to the pre-training data. To mitigate this issue, we proposed FLOW, a method which upweights easy samples based on the pre-trained loss values. Empirically, we showed that FLOW, on average, outperforms relevant baselines and is also complementary to these baselines in both vision and language settings. We also theoretically analyzed FLOW for linear models.
Discussion and limitations. We would like to conclude with an overview of our work's limitations and potential future directions. In lay terms, mitigating forgetting of pretraining capabilities comes at the cost of relatively lower fine-tuning performance. FLOW maintains this balance by sacrificing performance on hard samples from the finetuning data. Table 5 indicates that FLOW has lower accuracy on samples with high pre-training loss ("hard samples") compared to standard FT. Our method selectively downweighs samples with high pre-training losses for preserving pre-training performance. An interesting future direction is improving performance on such samples while maintaining or improving overall performance. On the theoretical side, we hope to extend our analysis to generalized linear models (GLMs) and even non-linear models.
Table 5. Comparison of FLOW and Standard-FT on hard samples across three vision datasets. We evaluate performance on the top 10% hardest samples (those with the highest pre-trained losses). Indeed, the samples with high pre-training losses have lower accuracy when using FLOW compared to standard finetuning (FT). This is an unsurprising outcome of our approach; we sacrifice performance on hard examples of the fine-tuning data to maintain performance on the pre-training data.
Dataset # of Hard Samples Standard FT FLOW CIFAR-10 1000 86.60 30.70 CIFAR-100 1000 56.40 21.30 Stanford Cars 805 71.30 13.18
this section cite: []

Section: References
Ref_id:b0 Title: Uncertaintybased continual learning with adaptive regularization Year: (2015)
Ref_id:b1 Title: Expert gate: Lifelong learning with a network of experts Year: (2017)
Ref_id:b2 Title: Memory aware synapses: Learning what (not) to forget Year: (2018)
Ref_id:b3 Title: Anonymous. Dynamic loss-based sample reweighting for improved large language model pretraining Year: (2019)
Ref_id:b4 Title: Program synthesis with large language models Year: (2021)
Ref_id:b5 Title: Rainbow memory: Continual learning with a memory of diverse samples Year: (2021)
Ref_id:b6 Title: Robust solutions of optimization problems affected by uncertain probabilities. Management Year: (2013)
Ref_id:b7 Title: LoRA learns less and forgets less Year: (2024)
Ref_id:b8 Title: Reasoning about physical commonsense in natural language Year: (2020)
Ref_id:b9 Title: Coresets via bilevel optimization for continual learning and streaming Year: (2014)
Ref_id:b10 Title: Online learned continual compression with adaptive quantization modules Year: (2020)
Ref_id:b11 Title: Efficient lifelong learning with a-GEM Year: (2019)
Ref_id:b12 Title: Continual learning with tiny episodic memories Year: (2019)
Ref_id:b13 Title: Take the bull by the horns: Hard sample-reweighted continual training improves llm generalization Year: (2024)
Ref_id:b14 Title: Momentum-filtered optimizer for mitigating forgetting in llm fine-tuning Year: (2024)
Ref_id:b15 Title: Think you have solved question answering? try arc, the ai2 reasoning challenge Year: (2018)
Ref_id:b16 Title: Training verifiers to solve math word problems Year: (2021)
Ref_id:b17 Title: Routing networks with co-training for continual learning Year: (2020)
Ref_id:b18 Title: Understanding the training speedup from sampling with approximate losses Year: (2024)
Ref_id:b19 Title: Continual prototype evolution: Learning online from non-stationary data streams Year: (2021)
Ref_id:b20 Title: Learning without memorizing Year: (2019-06-16)
Ref_id:b21 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b22 Title: Learning models with uniform performance via distributionally robust optimization Year: (2021)
Ref_id:b23 Title: Orthogonal gradient descent for continual learning Year: (2020-08)
Ref_id:b24 Title: A framework for few-shot language model evaluation Year: ()
Ref_id:b25 Title: Does fine-tuning llms on new knowledge encourage hallucinations Year: (2024)
Ref_id:b26 Title: Understanding finetuning for factual knowledge extraction Year: (2024)
Ref_id:b27 Title: Finetune like you pretrain: Improved finetuning of zero-shot vision models Year: (2023)
Ref_id:b28 Title:  Year: (2024)
Ref_id:b29 Title: Accelerate: Training and inference at scale made simple Year: (2022)
Ref_id:b30 Title: Measuring massive multitask language understanding Year: ()
Ref_id:b31 Title: Measuring mathematical problem solving with the math dataset Year: (2021)
Ref_id:b32 Title: Distilling the knowledge in a neural network Year: (2015)
Ref_id:b33 Title: Low-rank adaptation of large language models Year: (2022)
Ref_id:b34 Title: picking and growing for unforgetting continual learning Year: (2019)
Ref_id:b35 Title: Editing models with task arithmetic Year: (2023)
Ref_id:b36 Title: Selective experience replay for lifelong learning Year: (2018)
Ref_id:b37 Title:  Year: (2023)
Ref_id:b38 Title: Less-forgetful learning for domain expansion in deep neural networks Year: (2017)
Ref_id:b39 Title: Continual learning with node-importance based adaptive group sparse regularization Year: (2020)
Ref_id:b40 Title: Forget-free continual learning with winning subnetworks Year: (2022-07)
Ref_id:b41 Title: Biased importance sampling for deep neural network training Year: (2017)
Ref_id:b42 Title: Not all samples are created equal: Deep learning with importance sampling Year: (2018-07)
Ref_id:b43 Title: Ordered sgd: A new stochastic optimization framework for empirical risk minimization Year: (2020)
Ref_id:b44 Title: Fearnet: Brain-inspired model for incremental learning Year: (2018)
Ref_id:b45 Title: Overcoming catastrophic forgetting in neural networks Year: (2016)
Ref_id:b46 Title: Soup to go: mitigating forgetting during continual learning with model averaging Year: (2025)
Ref_id:b47 Title: 3d object representations for fine-grained categorization Year: (2013)
Ref_id:b48 Title: Wide neural networks of any depth evolve as linear models under gradient descent Year: (2009)
Ref_id:b49 Title: Continual learning with extended kronecker-factored approximate curvature Year: (2020)
Ref_id:b50 Title: Overcoming catastrophic forgetting with unlabeled data in the wild Year: (2019)
Ref_id:b51 Title: Overcoming catastrophic forgetting by incremental moment matching Year: (2017)
Ref_id:b52 Title: Largescale methods for distributionally robust optimization Year: (2020)
Ref_id:b53 Title: Datasets: A community library for natural language processing Year: (2021-11)
Ref_id:b54 Title:  Year: (2022-04)
Ref_id:b55 Title: Learning without forgetting Year: (2016)
Ref_id:b56 Title: Mitigating the alignment tax of rlhf Year: (2023)
Ref_id:b57 Title: Rotate your networks: Better weight consolidation and less catastrophic forgetting Year: (2018)
Ref_id:b58 Title: Gradient episodic memory for continual learning Year: (2017)
Ref_id:b59 Title: Online batch selection for faster training of neural networks Year: (2015)
Ref_id:b60 Title: How do quadratic regularizers prevent catastrophic forgetting: The role of interpolation Year: (2021)
Ref_id:b61 Title: Adding multiple tasks to a single network by iterative pruning Year: (2017)
Ref_id:b62 Title: Adapting a single network to multiple tasks by learning to mask weights Year: (2018)
Ref_id:b63 Title: Peft: State-of-the-art parameterefficient fine-tuning methods Year: (2022)
Ref_id:b64 Title: Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory Year: ()
Ref_id:b65 Title: Can a suit of armor conduct electricity? a new dataset for open book question answering Year: (2018)
Ref_id:b66 Title: Neuro-inspired stabilityplasticity adaptation for continual learning in sparse networks Year: (2022)
Ref_id:b67 Title: Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm Year: (2014)
Ref_id:b68 Title: Automated flower classification over a large number of classes Year: (2008)
Ref_id:b69 Title: Learning to remember: A synaptic plasticity driven framework for continual learning Year: (2019)
Ref_id:b70 Title: Lottery ticket adaptation: Mitigating destructive interference in llms Year: (2024)
Ref_id:b71 Title: Dogs: A dataset for recognising dog breeds from images Year: (2012)
Ref_id:b72 Title: An online method for a class of distributionally robust optimization with non-convex objectives Year: (2021)
Ref_id:b73 Title: Learning transferable visual models from natural language supervision Year: (2021-07)
Ref_id:b74 Title: Random path selection for continual learning Year: (2019)
Ref_id:b75 Title: Model zoo: A growing brain that learns continually Year: (2021)
Ref_id:b76 Title: Incremental classifier and representation learning Year: (2017)
Ref_id:b77 Title: Learning to learn without forgetting by maximizing transfer and minimizing interference Year: (2019-05-06)
Ref_id:b78 Title: Online structured laplace approximations for overcoming catastrophic forgetting Year: (2018)
Ref_id:b79 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b80 Title: Progressive neural networks Year: (2016)
Ref_id:b81 Title: An adversarial winograd schema challenge at scale Year: (2019)
Ref_id:b82 Title: Social iqa: Commonsense reasoning about social interactions Year: (2019)
Ref_id:b83 Title: Progress & compress: A scalable framework for continual learning Year: (2018-07)
Ref_id:b84 Title: Overcoming catastrophic forgetting with hard attention to the task Year: (2018-07)
Ref_id:b85 Title: Training regionbased object detectors with online hard example mining Year: (2016)
Ref_id:b86 Title: The task rehearsal method of life-long learning: Overcoming impoverished data Year: (2002)
Ref_id:b87 Title: Safe adaptive importance sampling Year: (2017)
Ref_id:b88 Title:  Year: (2024)
Ref_id:b89 Title: Gradient coreset based replay buffer selection for continual learning Year: (2023)
Ref_id:b90 Title: Llama 2: Open foundation and fine-tuned chat models Year: (2023)
Ref_id:b91 Title: Encoder based lifelong learning Year: (2017)
Ref_id:b92 Title: Coscl: Cooperation of small continual learners is stronger than a big one Year: (2022)
Ref_id:b93 Title: Memory replay with data compression for continual learning Year: ()
Ref_id:b94 Title: Incorporating neuro-inspired adaptability for continual learning in artificial intelligence Year: ()
Ref_id:b95 Title: Training networks in null space of feature covariance for continual learning Year: (2021)
Ref_id:b96 Title: Orthogonal subspace learning for language model continual learning Year: (2023-12)
Ref_id:b97 Title: URL Year: ()
Ref_id:b98 Title: Resnet strikes back: An improved training procedure in timm Year: ()
Ref_id:b99 Title: Transformers: Stateof-the-art natural language processing Year: ()
Ref_id:b100 Title: Online Year: (2020-10)
Ref_id:b101 Title: Supermasks in superposition Year: (2020)
Ref_id:b102 Title: Robust fine-tuning of zero-shot models Year: (2021)
Ref_id:b103 Title: Incremental classifier learning with generative adversarial networks Year: (2018)
Ref_id:b104 Title: Optimizing data mixtures speeds up language model pretraining Year: (2024)
Ref_id:b105 Title: Lifelong learning with dynamically expandable networks Year: (2018)
Ref_id:b106 Title: Metamath: Bootstrap your own mathematical questions for large language models Year: (2023)
Ref_id:b107 Title: Can a machine really finish your sentence? Year: (2019)
Ref_id:b108 Title: Continual learning of context-dependent processing in neural networks Year: ()
Ref_id:b109 Title: Continual learning through synaptic intelligence Year: (2017-08)
Ref_id:b110 Title: Stochastic optimization with importance sampling for regularized loss minimization Year: (2015)
Ref_id:b111 Title: 2021) model averaging experiments, we use α = 0.5. For our LoRA (Hu et al., 2022) experiments, we use α = r = 64. For ℓ 2 regularization we use λ = 1e -3 which is taken from (Chen et al., 2024b). Most training hyper-parameters for our language experiments are taken from Chen et al. (2024b) Year: ()
Ref_id:b112 Title: On top of the commonsense metric, we evaluate MMLU (Hendrycks et al., 2021a) and MBPP (Austin et al., 2021) to estimate the general capabilities of a language model and to measure the effects of catastrophic forgetting when fine-tuning a model on MetaMathQA Year: (2018)
Ref_id:b113 Title: HellaSwag presents a context followed by several plausible endings, and the model must choose the most appropriate continuation Year: (2019)
Ref_id:b114 Title: A benchmark part of the AI2 reasoning challenge designed to test basic scientific reasoning and knowledge. ARC Easy presents 5,197 multiple-choice science questions drawn from grade 3-9 standardized tests, where each question typically includes a brief scientific scenario or statement followed by four possible answer choices Year: (2018)
Ref_id:b115 Title: ARC Challenge presents 2,590 multiple-choice science questions drawn from grade 3-9 standardized tests, where each question typically includes a scientific scenario or phenomenon followed by four possible answer choices Year: (2018)
Ref_id:b116 Title: A benchmark designed to evaluate physical commonsense understanding in natural language. PIQA presents a goal and two possible solutions, requiring models to choose the most appropriate solution that demonstrates an understanding of everyday physical interactions Year: (2020)
Ref_id:b117 Title: SIQA presents a social situation context followed by a question and three possible answers, requiring models to demonstrate an understanding of social interactions, emotional responses, and behavioral implications Year: (2019)
Ref_id:b118 Title: OBQA presents 5,957 multiple-choice questions paired with a small "book" of 1,326 core science facts, requiring models to combine these facts Year: (2018)
Ref_id:b119 Title: 2021a): A benchmark designed to evaluate massive multitask language understanding. MMLU presents approximately 16,000 multiple-choice questions spanning 57 subjects including mathematics, philosophy, law, and medicine, requiring models to demonstrate broad knowledge and reasoning capabilities Year: ()
Ref_id:b120 Title: A benchmark designed to evaluate basic Python programming capabilities. The entire MBPP dataset presents 974 Python programming problems, where each problem includes a natural language task description and three test cases written as assert statements, requiring models to generate functionally correct Python code solutions Year: (2021)
Ref_id:b121 Title: A benchmark designed to evaluate multi-step mathematical reasoning capabilities. The GSM8K test set contains 1,000 grade school math word problems, where each problem requires 2-8 steps to solve using basic arithmetic operations Year: (2021)
Ref_id:b122 Title: We follow the standard evaluation process for each of these datasets and specifically use lm-evaluation-harness Year: (2024)
Ref_id:b123 Title: 2013) refers to the Stanford Cars dataset, which includes 16,185 images of 196 classes of cars. It provides a rich resource for fine-grained car classification task Year: ()
Ref_id:b124 Title: 2012) pertains to the Stanford Dogs dataset, containing 20,580 images of 120 breeds of dogs. This dataset is widely used for fine-grained dog breed classification and recognition tasks Year: ()
Ref_id:b125 Title: is a large-scale dataset for food classification containing 101 categories with 1,000 images per class, commonly used to evaluate models on fine-grained object recognition tasks Year: (2014)
