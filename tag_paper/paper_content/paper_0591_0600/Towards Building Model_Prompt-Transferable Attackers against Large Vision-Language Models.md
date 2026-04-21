Title: Towards Building Model/Prompt-Transferable Attackers against Large Vision-Language Models
Abstract: Although Large Vision-Language Models (LVLMs) exhibit impressive multimodal capabilities, their vulnerability to adversarial examples has raised serious security concerns. Existing LVLM attackers simply optimize adversarial images that easily overfit a certain model/prompt, making them ineffective once they are transferred to attack a different model/prompt. Motivated by this research gap, this paper aims to develop a more powerful attack that is transferable to black-box LVLM models of different structures and task-aware prompts of different semantics. Specifically, we introduce a new perspective of information theory to investigate LVLMs' transferable characteristics by exploring the relative dependence between outputs of the LVLM model and input adversarial samples. Our empirical observations suggest that enlarging/decreasing the mutual information between outputs and the disentangled adversarial/benign patterns of input images helps to generate more agnostic perturbations for misleading LVLMs' perception with better transferability. In particular, we formulate the complicated calculation of information gain as an estimation problem and incorporate such informative constraints into the adversarial learning process. Extensive experiments on various LVLM models/prompts demonstrate our significant transfer-attack performance.

Section: Introduction
Large Vision-Language Models (LVLMs) have garnered significant attention for their impressive capabilities in both visual perception and language interaction. Unlike pure-text Large Language Models, LVLMs incorporate visual encoders, enabling them to excel in a variety of multimodal tasks such as text-to-image generation [1,2,3], visual question-answering [4,5,6], and etc.. However, since model complexity increases and their applications expand into real-world scenarios, security concerns [7] of LVLM models have become increasingly prominent.
Recent studies [7,8,9,10,11,12,13] have revealed that LVLMs are highly vulnerable to adversarial attacks, which can significantly degrade performance and pose serious security risks. Specifically, these works [14,15,16,17,18,19,20,9] manipulate LVLMs by injecting imperceptible perturbations into benign images, misleading the model to produce incorrect or jailbreak results. Despite the progress in this area, existing attack methods face primary challenges. In particular, they are typically optimized for specific LVLM architectures or fixed prompts, making the generated adversarial examples difficult to transfer effectively across different models or downstream tasks. That is, they often fail to maintain effectiveness across diverse models and prompts simultaneously in practice, requiring attackers to craft separate perturbations for each model and each prompt, resulting in significant time and resource overhead. Although a few recent studies [8,10] have explored promptagnostic attack strategies, they not only rely on complex multi-prompt joint training schemes, but also fail to address the more challenging problem of cross-model transferability.
Therefore, in this paper, we make the first attempt to design a superior LVLM attacker that can achieve both model-and prompt-transfer attacks within a single adversarial learning process. Unlike previous 2D/3D transfer works [21,22,23,24] that improve the generalization of adversarial perturbations by resisting various distortions, we propose to investigate the agnostic/generalizable harmfulness of perturbations from a new information theory perspective [25,26,27,28]. Our core idea is: the informative dependence between the output of the LVLM model and the input images explicitly reflects the LVLMs' decision trajectory to make the final predictions and, therefore, a generalizable adversarial perturbation should have as more harmful effect as possible to control the flip of the LVLMs' prediction than the benign pattern in the image input. As in Figure 1 (a)(b), the existing LVLM attackers adversarially train the adversarial samples by implicitly restricting the mixed outputinput dependency via misleading loss functions, which may confuse the LVLM model to focus on the joint distribution of benign and adversarial patterns of inputs, resulting in an interference overfitting. Instead, once we explicitly adjust the LVLM's focus solely on the adversarial noise to enhance the corresponding adversarial harmfulness, the learned adversarial perturbation is able to jump out of the mixed overfitting and contributes more attacker-chosen guidance effects than the benign one to mislead the reasoning process even the sample is transferred to unknown LVLM models or prompts.
Based on the above observations, we propose a novel LVLM attack method to adversarially constrain the informative dependence between the benign/adversarial pattern of the input and the LVLM's output for improving the model/prompt-aware transferability. In particular, we exploit mutual information (MI) to explicitly measure such dependence via coefficient degrees, where a larger MI degree indicates a stronger dependence between the two variables. Since the mixed MI of the entire adversarial input cannot consider the dependence of the output on the different patterns, we theoretically demonstrate that this mixed MI is closely related to the linear sum of benign MI (between the output and the benign pattern) and adversarial MI (between the output and the adversarial pattern), therefore, we can disentangle the adversarial input into benign and adversarial parts for separate MI learning. We utilize lightweight neural networks to train with these two MI information as effective MI estimators via maximization strategy [29,27,30]. During adversarial learning, we dynamically enlarge the adversarial MI and decrease the benign MI of adversarial samples to force reasoning process to focus more on perturbations to enhance harmfulness. Results show that our adversarial samples containing larger adversarial MI achieve significant transfer-attack performance across various LVLMs/prompts.
The key contributions of our work are outlined as follows:
• We propose to address a practical but challenging LVLM attack setting, i.e., model/prompttransfer attack. This new setting can efficiently generate effective adversarial examples against different models/prompts compared to existing time/resource-consuming attacks.
• To obtain generalizable adversarial examples, we introduce to enhance the harmfulness of perturbations from a novel information theory perspective to improve transferability. An effective MI constraint for individual benign/adversarial patterns is devised to adjust the focus of LVLM solely to the additive perturbations.
• Extensive experiments are conducted to verify the strong adversarial transferability of our proposed attack on four prevalent LVLM models and three multimodal datasets with a spectrum of task-aware prompts.
2 Related Work LVLM Attackers. LVLMs generally combine the capabilities of processing visual information with natural language understanding by using pre-trained vision encoders with language models [31,32]. Due to this multimodal nature [33,34,35,36,37,38,39,40,41,42,43,44,45,46], LVLMs are particularly vulnerable as the multi-modal integration not only amplifies their vulnerable utility but also introduces new attack vectors that are absent in unimodal systems [47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65]. Most of the existing LVLM attackers [14,15,66,67,68,17,19,69,7] are inspired by the adversarial vulnerability observed in vision tasks. To evaluate the adversarial robustness of LVLMs and generate adversarial examples, they generally add and optimize imperceptible perturbations on the whole image to benign image inputs via back-propagation. Although they can achieve significant attack performance in both targeted and untargeted settings, they are easily limited by their perturbation-specific design that can solely produce adversarial examples to deceive a particular LVLM model and prompt within a singular process. That is, to compromise different LVLMs and prompts, they must generate distinct adversarial perturbations, which incur significant time and resource expenditure. Some recent works [8,10] try to develop cross-prompt attack approaches, however, they require complicated multi-prompt joint training and the challenging cross-model attack issue is still unexplored. Therefore, this paper aims to develop a model/prompt-transferable attack method that can efficiently and effectively fool the practical LVLM applications.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b6', 'b7', 'b8', 'b9', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b8', 'b7', 'b9', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b26', 'b29', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b59', 'b60', 'b61', 'b62', 'b63', 'b64', 'b13', 'b14', 'b65', 'b66', 'b67', 'b16', 'b18', 'b68', 'b6', 'b7', 'b9']

Section: Adversarial Transferability.
In the general 2D image and 3D point cloud fields, numerous works [21,22,23,24] have been proposed to improve adversarial transferability. These methods claim that the adversarial examples easily overfit the targeted model, therefore, they should generate more generalizable and harmful perturbations. Most methods [24,70,23,71,72] exploit diverse transformations to force the adversarial examples to resist them for improving the generalization, leading to transferable perturbations with much more harmfulness. Advanced momentum-based optimization strategies [73,74,70,75,76] are further introduced to stabilize the optimization procedure and escape the local optima. There are also some ensemble attacks [77,78,79] that generate more transferable adversarial examples by attacking multiple models simultaneously. Several works [80,81] also disrupt the feature space with model-agnostic designs to generate adversarial examples. Since there is no related work that systematically analyzes the adversarial transferability of LVLM attack methods across models/prompts, we follow previous works to generate as harmful as possible adversarial perturbations to improve the transferability of LVLM attacks.
this section cite: ['b20', 'b21', 'b22', 'b23', 'b23', 'b69', 'b22', 'b70', 'b71', 'b72', 'b73', 'b69', 'b74', 'b75', 'b76', 'b77', 'b78', 'b79', 'b80']

Section: Methodology

this section cite: []

Section: Problem Definition and Notations
We generally define an LVLM model as F , which receives an image x v and a task-specific prompt x p as the input pair to return a corresponding ground-truth answer y.
Threat Model. In this paper, we explore the setting of transferable LVLM attacks, where we assume that the attacker solely has knowledge of a certain victim model, including its parameters, training procedure, etc. The attackers are required to generate adversarial examples on this white-box victim model, and feed them to attack other unknown black-box target LVLM models. This setting is more challenging and practical as the attackers cannot always access the details of real-world LVLM applications.
Attacker's Goal. The objective of the attacker is to devise and add a harmful but imperceptible perturbation ∆ on x v , to generate an adversarial image as
x adv v = x v + ∆.
This adversarial example, upon application to any textual prompt across different LVLM models, is designed to compel the model to output a target label predetermined by the attacker. Therefore, such perturbation needs to exhibit persistence and robustness when deployed on unseen LVLM models, and to induce adversarial semantic alterations across different task-aware prompts for the same image, rendering the attack cross-model and cross-prompt applicable. In this paper, we mainly focus on targeted adversarial attacks that aim to craft the adversarial image x adv v to misguide the predicted answer of LVLM from the ground-truth label y to the specific targeted label y tar . The optimization goal is formulated as:
min J(F (x adv v , x p ), y tar ), s.t.∥x adv v -x v ∥ ∞ ≤ ϵ,(1)
where J(•) is the loss function, and we utilize l ∞ -norm to regularize the adversarial perturbation to the range ϵ.
this section cite: []

Section: Overview of Our Attack
Our Motivation. We consider improving attack transferability from the information theory perspective by separately increasing the harmful effect of adversarial perturbations while making the LVLM less sensitive to the benign pattern of the perturbed image. Specifically, to generate more harmful perturbations, we explicitly study the informative dependency between adversarial/benign patterns of the input and the output of LVLMs. Our main goal is to enhance the dependence of the output on the adversarial pattern, so that the learned perturbation can be more generalizable and contribute more attacker-chosen guidance effects than the benign pattern, ensuring that the attack remains effective when transferred to unknown models/prompts (more analysis is in Appendix H). As shown in Figure 2, empirical results prove that adversarial dependency plays an important role in enhancing the harmful effect of adversarial examples for improving transferability. In particular, mutual information (MI) is an entropy-based information measurement tool that quantifies the dependency between two random variables. A higher MI indicates a stronger dependency between the variables. However, directly using the general MI calculation between adversarial examples and the corresponding outputs of LVLMs to measure the dependency presents a limitation. This is because adversarial images consist of both benign and adversarial patterns, and both of them have significant impacts on the output results. Directly maximizing the mixed MI between the adversarial image and the output may inadvertently increase the dependency of the output on the benign image, thereby hindering the increase in perturbation harmfulness. To address this issue, this paper makes an in-depth investigation on more fine-grained constraints of disentangled adversarial/benign MI information to achieve attacks.
Overall Pipeline. We present the overview of our proposed attack method in Figure 1 (c). Specifically, we first theoretically decompose the mixed MI information into benign MI (between the benign image and the LVLM output) and adversarial MI (between the adversarial perturbation and the LVLM output) components. Since the direct computation of MI is infeasible, we then train two MI estimators for separate MI calculations. Finally, we incorporate the benign and adversarial MI constraints into the optimization strategy to generate transferable adversarial examples.
this section cite: []

Section: How to Represent Adversarial/Benign MI?
We cannot simply utilize the separate benign image x v and adversarial perturbation ∆ to calculate the two MI values with the LVLM output, as the adversarial impact is produced by the joint effects of their combination x adv v . Therefore, to specifically represent the separate adversarial MI and benign MI, we need to disentangle them from the joint/mixed MI I(x adv v ; L ′ ), where L ′ = F (x adv v , x p ) is the logits of targeted output by the LVLM model.
Specifically, we define I A as the adversarial MI between solely the additive adversarial perturbation and the LVLM's logit output, i.e., I A (∆, L ′ ) or I A (∆, L), where L = F (x v , x p ) is the logits of benign output. Benign MI I B is also defined between solely the benign image pattern and the logit output, i.e., I B (x v , L ′ ) or I B (x v , L). We first provide Theorem 1 to illustrate the components and their relationship [27] within the mixed MI I(x adv v ; L ′ ). Theorem 1. Let x adv v , x v , ∆, L ′ represent four random variables, then the mixed MI I(x adv v ; L ′ ) has the following expression (proofed in the Appendix B):
I(x adv v ; L ′ ) = I(x v ; L ′ ) + I(∆; L ′ ) + H(L ′ |x v , ∆) -H(L ′ |x adv v ) -I(x v ; ∆; L ′ ),(2)
where H(•|•) represents conditional entropy. In particular, H(L ′ |x v , ∆) and H(L ′ |x adv v ) can be formulated as:
H(L ′ |x v , ∆) = - L ′ ,xv,∆ p(L ′ , x v , ∆) log p(L ′ |x v , ∆), H(L ′ |x adv v ) = - L ′ ,x adv v p(L ′ , x adv v ) log p(L ′ |x adv v ),(3)
where p(L ′ , x v , ∆), p(L ′ , x adv v ) are the joint probability, p(L ′ |x v , ∆), p(L ′ |x adv v ) are the conditional probability. Assumption 1. ∆, x adv v are bijections of x v , i.e., ∆, x adv v are dependently and uniquely determined by x v and the decompositon of x adv v is also unique (the theoretical basis of this assumption is detailed in the Appendix C).
Based on this assumption, there exists p(x adv v ) = p(x v , ∆). Substituting this relation into Equation (3), we can obtain H(L ′ |x v , ∆) ≈ H(L ′ |x adv v ) (proofed in the Appendix D,E). Besides, since the effects of benign and adversarial patterns on the output are mutually exclusive, thus I(x v ; ∆; L ′ ) is presented to be very small that can be ignored. Therefore, according to the above derivations, now the mixed MI I(x adv v ; L ′ ) can be linearly expressed as:
I(x adv v ; L ′ ) ≈ I(x v ; L ′ ) + I(∆; L ′ ).(4)
In this manner, we can approximately disentangle the mixed MI into the benign MI I(x v ; L ′ ) and the adversarial MI I(∆; L ′ ) and calculate them separately. These two MIs can not only reflect the dependency between the whole adversarial input and output like the mixed MI, but also provide independent measurements for different patterns.
this section cite: ['b26']

Section: How to Calculate Adversarial/Benign MI?
Although we can approximately represent the adversarial/benign MI following the aforementioned disentanglement, directly calculating MI is typically very challenging in high-dimensional spaces as it is a relative value. Luckily, many methods have been proposed to estimate MI [82,29]. Among them, the Deep InfoMax (DIM) estimation method has been shown to be more effective [29]. Therefore, we adopt local DIM (details in Appendix F) and the Donsker-Varadhan representation [83] based on the KL divergence to estimate the adversarial/benign MI in our scenarios as:
I(X; Y ) := D KL (J∥M) ≥ I (DV )
ω,ψ (C ψ (X); Y ) := E J [T ω (C ψ (x), y)] -log E M [e Tω(C ψ (x),y) ], (5) where X is a random variable, which can be the representation of any visual input {x v , ∆, x adv v } of LVLM. Y is also the random variable, which is the representation of the LVLM's output logits L or L ′ . J is the joint probability distribution of X and Y . M is the product of the marginal probability distributions of X and Y . We denote I (DV ) ω,ψ as this MI estimation network based on the Donsker-Varadhan mechanism, which consists of two sub-networks C ψ and T ω . Specifically, C ψ is an encoder composed of a neural network with parameters ψ, which maps the image input to a local feature map in the same latent space of Y . T ω is a discriminator function modeled by a neural network with parameters ω to determine the relations between X and Y . Therefore, we define two estimation networks I (DV ) ω A ,ψ A and I (DV ) ω B ,ψ B to calculate adversarial and benign MI, respectively. Due to the close relevance between the adversarial/benign patterns and the outputs of the separate perturbation/benign pattern, we utilize adversarial perturbations and benign images to train the I (DV ) ω A ,ψ A and I (DV ) ω B ,ψ B . For network I (DV ) ω A ,ψ A , we maximize the adversarial MI between the adversarial pattern of the perturbed image and the targeted output while minimizing the adversarial MI between the adversarial pattern of the perturbed image and the benign output. For network I (DV ) ω B ,ψ B , we maximize the benign MI between the benign pattern of the perturbed image and the benign output while minimizing the benign MI between the benign pattern of the perturbed image and the targeted output. The optimization objectives are formulated as follows:
(ω A , ψA ) = arg max ω A ,ψ A [ I (DV ) ω A ,ψ A (C ψ A (∆); L ′ ) -I (DV ) ω A ,ψ A (C ψ A (∆); L)],(6)
(ω B , ψB ) = arg max ω B ,ψ B [ I (DV ) ω B ,ψ B (C ψ B (x v ); L) -I (DV ) ω B ,ψ B (C ψ B (x v ); L ′ )],(7)
where I (DV ) ωA , ψA (•), I (DV ) ωB , ψB (•) are the estimated adversarial MI values and benign MI values.
this section cite: ['b81', 'b28', 'b28', 'b82']

Section: Improving Transferability with Informative Constraints of Adversarial/Benign MI
To guide the image contents focusing more on the adversarial impacts of perturbations for improving the transferability, we develop an informative optimization strategy based on both adversarial and benign MI constraints to generate more harmful and generalizable adversarial samples. Specifically, by increasing the informative dependence between the adversarial perturbation of the input image and the adversarial output of the LVLM, while decreasing the dependence between the benign image pattern and the adversarial output of the LVLM, we can enhance the strength of the adversarial perturbation and ensure that this perturbation contributes more guidance for the attacker's choice compared to the benign image pattern. In this manner, the perturbation can always have more effect than the benign pattern, thus the adversarial example can still mislead the LVLM's reasoning when it is transferred to attack unknown models or prompts.
To achieve this goal, we utilized two MI evaluation networks trained by Section 3.4 to construct the optimization objective for generating adversarial examples as follows:
arg max
||∆||p≤ϵ [ I (DV ) ωA , ψA (C ψA (∆); F (x v + ∆, x p )) -I (DV ) ωB , ψB (C ψB (x v ); F (x v + ∆, x p ))],(8)
where Formula 8 is recorded as l mi . Besides, to better adjust the LVLM's output towards the target text y tar , we also utilize a cross-entropy CE(•)loss to minimize the difference between the adversarial output and the target text. The cross-entropy loss l ce is as follows:
l ce = CE(F (x v + ∆, x p ), y tar ).(9)
The overall loss for generating transferable adversarial examples is formulated as follows:
J = w 1 * l ce -w 2 * l mi ,(10)
where w 1 , w 2 are weights to balance the loss. The algorithm of our attack is detailed in Appendix G. InstructBLIP PGD [67] 0.046 0.0 0.0 0.037 0.0 0.0 0.133 4.0 6.5 0.529 47.7 60.1 CroPA [8] 0.049 0.0 0.0 0.053 0.0 1.4 0.266 16.8 26.2 0.859 83.7 85.0 UniAtt [10] 0.158 6.7 12.1 0.262 20.8 23.5 0.384 29.9 36.7 0.836 77.9 84.5 Ours 0.482 44.6 47.4 0.521 47.1 53.0 0.599 49.0 55.5 0.767 75.6 82.5 4 Experiments
this section cite: ['b66', 'b7']

Section: Implementation Details
LVLM Models. In this paper, following existing LVLM attack methods [8,10], we conduct experiments on the same open-source LVLM models, including LLaVA-1.5 (integrated with Vicuna-7B) [84], MiniGPT-4 (integrated with Llama-2-7B-Chat) [85], BLIP-2 (integrated with OPT-2.7b) [5], and InstructBLIP (integrated with Vicuna-7B) [86], for comparison.
LVLM Datasets and Tasks. We evaluate the adversarial robustness of three multi-modal datasets for the image captioning, image classification, and VQA tasks. The datasets consist of both images and prompts. The images are collected from DALL-E [87], SVIT [88] and VQAv2 [89]. The prompts for three tasks derive from the CroPA [8].
Basic Setups. We consider two evaluation metrics: the semantic similarity (SS) utilizes the Sentence-Transformer [90] to generate embeddings of both adversarial output and target text for calculating their cosine similarity, and the success rates "ExactMatch" (EM) and "ConditionalContain" (CC) to assess the word-level overlap between adversarial output and target text.
We utilize the same architectures to initialize the adversarial MI and benign MI estimation networks, but they are trained separately. Specifically, C ψ is implemented as a light two-layer convolutional neural network, while T ω simply incorporates an attention mechanism, 1 × 1 convolutional blocks, and residual connections. For training, we first use the selected adversarial examples generated by PGD [67] attack with ϵ = 16/255. We then feed the same prompt with benign image x v and
v -x v , L, L ′
) is used to train the adversarial MI estimation network following Equation 6, while the tuple (x v , L, L ′ ) is used as input to train the benign MI estimation network following Equation 7. We train both networks using the Adam optimizer for 100 epochs, with an initial learning rate of 0.01 that decays by a factor of 0. 5 every 20
this section cite: ['b7', 'b9', 'b83', 'b84', 'b4', 'b85', 'b86', 'b87', 'b88', 'b7', 'b89', 'b66']

Section: Main Results

this section cite: []

Section: Transfer-Attack Performance across LVLMs.
To investigate the transferability of our proposed attack, we first provide the performance across different LVLM models in Table 1. Here, we select the target text "I am sorry", and all the performances are averagely evaluated on three tasks. We can find that: (1) Our generated adversarial examples have competitive harmfulness compared to existing attacks in the diagonal values. This demonstrates that our attack also contributes to improve the harmful impact of the samples. (2) Our attacks achieve significant transfer-attack performance compared to previous works, demonstrating the effectiveness of our designed informative constraints. Furthermore, we transfer the adversarial examples generated on MiniGPT-4 model to realistic LVLM applications GPT-4o (GPT-4o-0513) [91] and Claude-3.5-Sonnet [92]. As shown in Table 3, our attack still achieves better performance. Transfer-Attack Performance across Prompts. We then investigate the transferattack performance across different numbers of prompts in Table 2. Here, we directly transfer the adversarial examples generated by a certain LVLM model and prompt to the same LVLM model with different prompts. We can find that previous attacks achieve worse performance with the increase of the prompt numbers. Instead, our method achieves better attack performance across different prompts, demonstrating the effectiveness of our developed informative constraints.
Joint transferability across models and prompts. We further evaluate the joint transferability of our proposed attack across both different models and prompts at the same time. As shown in Figure 5, the results still demonstrate that our method retains strong transferability even under this more challenging setting.
We also provide experiments on more datasets and architecturally distinct LVLMs in Appendix I.1, I.2. Justification of our transfer attack are in Appendix I.3. To verify generality, we also evaluate under a universal setting and on jailbreak/rewiring attacks, see Appendix I.6 and I.7.
this section cite: ['b90', 'b91']

Section: MiniGPT-4
Raw Answer: Image of a boy sitting on a skateboard with a helmet and knee pads, surrounded by people skateboarding in the background. Adv Answer: I am sorry.
this section cite: []

Section: InstructBLIP
Raw Answer: Skateboarding.
this section cite: []

Section: Adv

this section cite: []

Section: Attack Efficiency and Robustness
Complexity Analysis. To investigate the scalability and practicality of our transfer-attack method, we provide the complexity analysis in Table 4, which evaluates the usage of GPU time and memory of a single adversarial sample generation on LLaVA-1.5 model. It indicates that our attack costs relatively fewer GPU resources, as our informative constraints are easily achieved with solely loss designs, while our samples can achieve better transfer-attack performance within a single generation process.
this section cite: []

Section: Table 4: Complexity comparison on adversarial sample generation.
LVLM Attack GPU Time (↓) GPU Memory (↓) PGD [67] 4 min 16.7 GB CroPA [8] 12 min 20.4 GB UniAtt [10] 294 min 57.5 GB Ours 9 min 18.2 GB
this section cite: ['b66', 'b7', 'b9']

Section: Robustness to Defenses.
To evaluate the robustness of our attack against potential defense strategies, we conduct experiments on three pre-processing defense methods, i.e, Randomization [93,94], JPEG Compression [95], and Diffusion Restoration [96] in Figure 3 (a). Compared to previous attacks, our attack is relatively more robust to potential defenses because we explicitly constrain the adversarial perturbation to be as harmful as possible. This allows it to provide more guidance to the LVLM's reasoning than the benign pattern, having more opportunities to lead to wrong results. More defense experiments can be found in Appendix I.4.
this section cite: ['b92', 'b93', 'b94', 'b95']

Section: Effectiveness of MI Estimation Networks
During the training process of each MI estimation network, directly maximizing positive MI without minimizing negative MI may not clearly learn the accurate effect for adversarial perturbation pattern or benign pattern (i.e., solely maximizing (ω A , ψA ) = arg max ω A ,ψ A I (DV )
ω A ,ψ A (C ψ A (∆); L ′ ) or (ω B , ψB ) = arg max ω B ,ψ B I (DV ) ω B ,ψ B (C ψ B (x v ); L)
). Therefore, we design the joint maximizationminimization optimization mechanism to train each MI estimation network via Equation (6) (7). To demonstrate its effectiveness, we compare the MI estimation performance of these two training strategies and compute the average MI value for all samples as shown in Figure 3 (b). The results indicate that our optimization mechanism helps to better capture the inherent differences between adversarial perturbation patterns and benign patterns in terms of adversarial MI and benign MI.
this section cite: ['b6']

Section: Ablation Study
Ablation on Different Target Texts. To demonstrate that the effectiveness of our attack is not constrained to the specific case of the target text "I am sorry", we extend our evaluation to more complex and sophisticated target texts. As shown in Table 5, despite the increased difficulty and complexity of these target texts, our attack strategy still demonstrates significant effectiveness and consistently maintains high performance in transfer attacks.
this section cite: []

Section: Ablation on Different MI Components.
To elucidate the role of each component of our method in improving transferability, we conduct ablation studies: (1) removing the adversarial MI constraint, and (2) removing the benign MI constraint. As shown in Figure 3 (c), the results demonstrate that each component of our method contributes positively to improving transfer-attack performance.
this section cite: []

Section: Conclusion
This paper proposes a powerful LVLM attack method that is transferable across different LVLM models and prompts. We introduce a new perspective of information theory to investigate LVLMs' transferable characteristics by exploring the relative dependence between outputs of the LVLM and input adversarial samples. With appropriate informative constraints between the disentangled adversarial/benign patterns of the image input and output text, our generated adversarial examples are proven to be more generalizable and harmful to unseen LVLMs and prompts. Extensive experiments indicate the effectiveness of our proposed attack.
this section cite: []

Section: References
Ref_id:b0 Title: Glide: Towards photorealistic image generation and editing with text-guided diffusion models Year: (2021)
Ref_id:b1 Title: Hierarchical text-conditional image generation with clip latents Year: (2022)
Ref_id:b2 Title: High-resolution image synthesis with latent diffusion models Year: (2022)
Ref_id:b3 Title: Multimodal few-shot learning with frozen language models Year: (2021)
Ref_id:b4 Title: Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b5 Title: Flamingo: a visual language model for few-shot learning Year: (2022)
Ref_id:b6 Title: A survey of attacks on large vision-language models: Resources, advances, and future trends Year: (2024)
Ref_id:b7 Title: An image is worth 1000 lies: Adversarial transferability across prompts on vision-language models Year: (2024)
Ref_id:b8 Title: On evaluating adversarial robustness of large vision-language models Year: (2024)
Ref_id:b9 Title: Pandora's box: Towards building universal attackers against real-world large vision-language models Year: (2024)
Ref_id:b10 Title: Imperceptible transfer attack on large vision-language models Year: (2025)
Ref_id:b11 Title: Are large vision-language models robust to adversarial visual transformations? Year: (2024)
Ref_id:b12 Title: Can't see the wood for the trees: Can visual adversarial patches fool hard-label large vision-language models? Year: (2024)
Ref_id:b13 Title: Image hijacks: Adversarial images can control generative models at runtime Year: (2023)
Ref_id:b14 Title: How robust is google's bard to adversarial image attacks Year: (2023)
Ref_id:b15 Title: Instructta: Instruction-tuned targeted attack for large vision-language models Year: (2023)
Ref_id:b16 Title: Stop reasoning! when multimodal llms with chain-of-thought reasoning meets adversarial images Year: (2024)
Ref_id:b17 Title: Towards evaluating the robustness of large vision-language model on adversarial visual-instructions Year: (2024)
Ref_id:b18 Title: Test-time backdoor attacks on multimodal large language models Year: (2024)
Ref_id:b19 Title: Imgtrojan: Jailbreaking visionlanguage models with one image Year: (2024)
Ref_id:b20 Title: Imperceptible transfer attack and defense on 3d point cloud classification Year: (2022)
Ref_id:b21 Title: Improving the transferability of adversarial samples with adversarial transformations Year: (2021)
Ref_id:b22 Title: Improving transferability of adversarial examples with input diversity Year: (2019)
Ref_id:b23 Title: Evading defenses to transferable adversarial examples by translation-invariant attacks Year: (2019)
Ref_id:b24 Title: Estimation of mutual information using kernel density estimators Year: (1995)
Ref_id:b25 Title: Estimation of the information by an adaptive partitioning of the observation space Year: (1999)
Ref_id:b26 Title: Improving adversarial robustness via mutual information estimation Year: (2022)
Ref_id:b27 Title: Mutual information neural estimation Year: (2018)
Ref_id:b28 Title: Learning deep representations by mutual information estimation and maximization Year: (2018)
Ref_id:b29 Title: Learning adversarially robust representations via worst-case mutual information maximization Year: (2020)
Ref_id:b30 Title: Benchmarking micro-action recognition: Dataset, method, and application Year: (2024)
Ref_id:b31 Title: Prvr: Partially relevant video retrieval Year: (2025)
Ref_id:b32 Title: Intriguing properties of neural networks Year: (2013)
Ref_id:b33 Title: Context-aware biaffine localizing network for temporal sentence grounding Year: (2021)
Ref_id:b34 Title: Jointly cross-and self-modal graph attention network for query-based moment localization Year: (2020)
Ref_id:b35 Title: Saanet: Siamese action-units attention network for improving dynamic facial expression recognition Year: (2020)
Ref_id:b36 Title: Memory-guided semantic learning network for temporal sentence grounding Year: (2022)
Ref_id:b37 Title: Adaptive proposal generation network for temporal sentence localization in videos Year: (2021)
Ref_id:b38 Title: Unsupervised temporal video grounding with deep semantic clustering Year: (2022)
Ref_id:b39 Title: You can ground earlier than see: An effective and efficient pipeline for temporal sentence grounding in compressed videos Year: (2023)
Ref_id:b40 Title: Reducing the vision and language bias for temporal sentence grounding Year: (2022)
Ref_id:b41 Title: Hypotheses tree building for one-shot temporal sentence localization Year: (2023)
Ref_id:b42 Title: Rethinking the video sampling and reasoning strategies for temporal sentence grounding Year: (2023)
Ref_id:b43 Title: Spatiotemporal graph neural network based mask reconstruction for video object segmentation Year: (2021)
Ref_id:b44 Title: Few-shot temporal sentence grounding via memory-guided semantic learning Year: (2022)
Ref_id:b45 Title: A survey on text-guided 3d visual grounding: Elements, recent advances, and future directions Year: (2024)
Ref_id:b46 Title: Stabilizing modality gap & lowering gradient norms improve zero-shot adversarial robustness of vlms Year: (2025)
Ref_id:b47 Title: Improving zero-shot adversarial robustness in vision-language models by closed-form alignment of adversarial path simplices Year: (2025)
Ref_id:b48 Title: Robust distillation via untargeted and targeted intermediate adversarial samples Year: (2024)
Ref_id:b49 Title: Adversarially robust few-shot learning via parameter co-distillation of similarity and class concept learners Year: (2024)
Ref_id:b50 Title: Adversarially robust distillation by reducing the student-teacher variance gap Year: (2024)
Ref_id:b51 Title: The enemy of my enemy is my friend: Exploring inverse adversaries for improving adversarial training Year: (2023)
Ref_id:b52 Title: Improving adversarially robust few-shot image classification with generalizable representations Year: (2022)
Ref_id:b53 Title: Restricted black-box adversarial attack against deepfake face swapping Year: (2023)
Ref_id:b54 Title: Survey on adversarial attack and defense for medical image analysis: Methods and challenges Year: (2024)
Ref_id:b55 Title: Imperceptible beam-sensitive adversarial attacks for lidar-based object detection in autonomous driving Year: ()
Ref_id:b56 Title: Imperceptible backdoor attacks on text-guided 3d scene grounding Year: (2025)
Ref_id:b57 Title: Frequency-aware gan for imperceptible transfer attack on 3d point clouds Year: (2024)
Ref_id:b58 Title: Hiding imperceptible noise in curvature-aware patches for 3d point cloud attack Year: (2024)
Ref_id:b59 Title: Explicitly perceiving and preserving the local geometric structures for 3d point cloud attack Year: (2024)
Ref_id:b60 Title: Robust geometry-dependent attack for 3d point clouds Year: (2023)
Ref_id:b61 Title: 3dhacker: Spectrumbased decision boundary generation for hard-label 3d point cloud attack Year: (2023)
Ref_id:b62 Title: Exploring the devil in graph spectral domain for 3d point cloud attacks Year: (2022)
Ref_id:b63 Title: Point cloud attacks in graph spectral domain: When 3d geometry meets graph signal processing Year: (2023)
Ref_id:b64 Title: Seeing is not believing: Adversarial natural object optimization for hard-label 3d scene attacks Year: (2025)
Ref_id:b65 Title: Misusing tools in large language models with visual adversarial examples Year: (2023)
Ref_id:b66 Title: On the robustness of large multimodal models against image adversarial attacks Year: (2024)
Ref_id:b67 Title: Adversarial robustness for visual grounding of multimodal large language models Year: (2024)
Ref_id:b68 Title: Inducing high energy-latency of large vision-language models with verbose images Year: (2024)
Ref_id:b69 Title: Nesterov accelerated gradient and scale invariance for adversarial attacks Year: (2019)
Ref_id:b70 Title: Improving adversarial transferability via neuron attribution-based attacks Year: (2022)
Ref_id:b71 Title: Improving the transferability of adversarial samples by pathaugmented method Year: (2023)
Ref_id:b72 Title: Boosting adversarial attacks with momentum Year: (2018)
Ref_id:b73 Title: Boosting adversarial transferability by achieving flat local maxima Year: (2023)
Ref_id:b74 Title: Enhancing the transferability of adversarial attacks through variance tuning Year: (2021)
Ref_id:b75 Title: Boosting adversarial transferability through enhanced momentum Year: (2021)
Ref_id:b76 Title: Learning transferable adversarial examples via ghost networks Year: (2020)
Ref_id:b77 Title: Delving into transferable adversarial examples and black-box attacks Year: (2016)
Ref_id:b78 Title: Stochastic variance reduced ensemble adversarial attack for boosting the adversarial transferability Year: (2022)
Ref_id:b79 Title: Boosting the transferability of adversarial samples via attention Year: (2020)
Ref_id:b80 Title: Transferable adversarial perturbations Year: (2018)
Ref_id:b81 Title: Mine: mutual information neural estimation Year: (2018)
Ref_id:b82 Title: Asymptotic evaluation of certain markov process expectations for large time Year: (1983)
Ref_id:b83 Title: Visual instruction tuning Year: (2024)
Ref_id:b84 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b85 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2024)
Ref_id:b86 Title: Zero-shot text-to-image generation Year: (2021)
Ref_id:b87 Title: Svit: Scaling up visual instruction tuning Year: (2023)
Ref_id:b88 Title: Making the v in vqa matter: Elevating the role of image understanding in visual question answering Year: (2017)
Ref_id:b89 Title: Sentence-bert: Sentence embeddings using siamese bertnetworks Year: ()
Ref_id:b90 Title: Hello gpt Year: (2024)
Ref_id:b91 Title: The claude 3 model family: Opus, sonnet Year: (2024)
Ref_id:b92 Title: The best defense is a good offense: adversarial augmentation against adversarial attacks Year: (2023)
Ref_id:b93 Title: Mitigating adversarial effects through randomization Year: (2017)
Ref_id:b94 Title: Countering adversarial images using input transformations Year: (2017)
Ref_id:b95 Title: Diffusion models for adversarial purification Year: (2022)
Ref_id:b96 Title: Information Theory: From Coding to Learning Year: (2025)
Ref_id:b97 Title: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution Year: (2024)
Ref_id:b98 Title: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks Year: (2024)
Ref_id:b99 Title: Gemma 3 technical report Year: (2025)
Ref_id:b100 Title: Failures to find transferable image jailbreaks between vision-language models Year: (2024)
Ref_id:b101 Title: Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models Year: (2024)
Ref_id:b102 Title: Defending lvlms against vision attacks through partial-perception supervision Year: (2025)
Ref_id:b103 Title: Visual adversarial examples jailbreak aligned large language models Year: (2024)
Ref_id:b104 Title: White-box multimodal jailbreaks against large vision-language models Year: (2024)
