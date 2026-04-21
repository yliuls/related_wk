Title: MGD 3 : Mode-Guided Dataset Distillation using Diffusion Models
Abstract: Dataset distillation has emerged as an effective strategy, significantly reducing training costs and facilitating more efficient model deployment. Recent advances have leveraged generative models to distill datasets by capturing the underlying data distribution. Unfortunately, existing methods require model fine-tuning with distillation losses to encourage diversity and representativeness. However, these methods do not guarantee sample diversity, limiting their performance. We propose a mode-guided diffusion model leveraging a pre-trained diffusion model without the need to fine-tune with distillation losses. Our approach addresses dataset diversity in three stages: Mode Discovery to identify distinct data modes, Mode Guidance to enhance intra-class diversity, and Stop Guidance to mitigate artifacts in synthetic samples that affect performance. Our approach outperforms stateof-the-art methods, achieving accuracy gains of 4.4%, 2.9%, 1.6%, and 1.6% on ImageNette, ImageIDC, ImageNet-100, and ImageNet-1K, respectively. Our method eliminates the need for fine-tuning diffusion models with distillation losses, significantly reducing computational costs. Our code is available on the project webpage: https://jachansantiago.github.io/mode- guided-distillation/

Section: Introduction
The rapid advancements in machine learning are marked by a trend towards increasingly large datasets and models to achieve state-of-the-art performance. However, this trend Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).
Optimization-based Dataset Distillation
S Dataset Distribution Learning T S Dataset Matching Loss Original Dataset Distilled Dataset Distilled Dataset
this section cite: []

Section: Generative Dataset Distillation

this section cite: []

Section: T

this section cite: []

Section: Original Dataset Generative Model

this section cite: []

Section: Dataset synthesis

this section cite: []

Section: Generative Model
Gradient Flow
Figure 1. Optimization-based Dataset Distillation: Optimizes the distilled dataset to match the statistics of gradient/features of the Original Dataset. Generative Dataset Distillation: First, it learns the dataset distribution of the original dataset and then sample a dataset that approximates the original dataset distribution.
presents significant challenges for researchers constrained by limited computation and storage resources. In response, the research community started to focus on developing techniques to address these limitations. While model pruning (Liu et al., 2017;He et al., 2019;Ding et al., 2019;Sharma & Foroosh, 2022) and quantization (Wu et al., 2016;Chen et al., 2021;Chauhan et al., 2023;Xu et al., 2023) have been introduced to improve model efficiency, core set selection and dataset distillation (Wang et al., 2018;Liu et al., 2022) have emerged as prominent techniques for reducing the size of training datasets to accelerate model training.
The process of reducing the training dataset involves removing redundant information while retaining essential data. Core set selection (Welling, 2009;Chen et al., 2010;Rebuffi et al., 2017;Castro et al., 2018) based approaches were initially introduced for building condensed datasets, which involves selecting a few prototypical examples from the original dataset to build the smaller dataset. However, these approaches are limited to choosing the samples from the original dataset, which considerably restricts the expressiveness of the condensed dataset. The task of dataset distillation is to distill information from a large training dataset into a smaller dataset with few synthetic samples such that a model trained on the smaller dataset achieves performance comparable to the model trained on the original dataset.
Optimization-based dataset distillation methods follow the data matching framework (Cazenavette et al., 2022;2023;Zhao & Bilen, 2023), where the distilled dataset is updated to mimic the influence of the original dataset when training (see the top of Fig. 1). These methods minimize the distribution gap between the original and distilled datasets by considering different aspects, such as model parameters, long-range training trajectories, or feature distribution. However, these methods are far from optimal, as they need to repeat the execution of their method to synthesize distilled datasets of different sizes. In addition, they tend to generate out-of-distribution samples.
To address these challenges, generative dataset distillation methods (Wang et al., 2023;Zhang et al., 2023;Su et al., 2024) propose storing the knowledge of the dataset into the parameters of a generative model instead of directly condensing it into a smaller synthetic set (see the bottom of Fig. 1). Once trained, the same generative model can generate synthetic datasets of varied sizes. This typically, is achieved by training the generative model with representative and diversity losses.
Among the generative models, diffusion models (Ho et al., 2020) are known for their impressive capabilities in image synthesis. These models achieve perceptual quality comparable to GANs while offering higher distribution coverage, as evidenced by (Dhariwal & Nichol, 2021b). However, they tend to concentrate on denser regions (modes) of the data distribution, resulting in a synthetic dataset that, while representative, often lacks the full diversity of the original data (Gu et al., 2024) (refer to Fig. 2a). Previous works (Gu et al., 2024) address this by explicitly fine-tuning the model with representative and diversity losses to generate representative and diverse samples. With this fine-tuning, the samples are more likely to be generated from different modes of a class (See Fig. 2b). However, this approach requires additional training, which can be computationally expensive and limit its practicality in resource-constrained settings.
We propose a novel approach that extracts diverse and representative samples from a pre-trained diffusion model trained on the target dataset, without additional training or finetuning. Our method first estimates prevalent data modes in the Mode Discovery stage. Then, diversity is ensured by guiding each sample to a different mode with Mode Guidance. However, guiding samples to modes may reduce quality, so we introduce Stop Guidance to preserve synthetic data quality (see Fig. 2c). In summary, the key contributions are:
• A novel dataset distillation approach leveraging a pretrained diffusion model without retraining or fine-tuning.
• Improved diversity and representativeness compared to previous diffusion-based methods.
• Matching or surpassing state-of-the-art results on multiple benchmarks while reducing computational cost.
this section cite: ['b15', 'b9', 'b8', 'b24', 'b31', 'b4', 'b3', 'b32', 'b29', 'b14', 'b30', 'b5', 'b20', 'b0', 'b1', 'b28', 'b35', 'b25', 'b11', 'b9', 'b9']

Section: Related Work
Dataset distillation has received increased interest in recent years due to its applications in continual learning (Zhao et al., 2021;Zhao & Bilen, 2021;2023), privacy-preserving datasets (Li et al., 2020;Sucholutsky & Schonlau, 2021), neural architecture search (Zhao et al., 2021;Zhao & Bilen, 2021), and model explainability (Loo et al., 2022). Prior works have explored the problem of dataset distillation and shown how challenging it is to encapsulate datasets in a limited set of examples. Initially, this task was approached using non-generative models, then with generative priors, and more recently with generative models and with decoupled dataset distillation. Below, we discuss works belonging to these categories in detail.
this section cite: ['b13', 'b26', 'b16']

Section: Non-generative Dataset Distillation Methods.
Dataset distillation condenses information from a large dataset into a smaller one with synthetic images, enabling model training on the smaller dataset with performance comparable to the full dataset. Initially, Zhao et al. (2021) proposed gradient matching to align the model's gradient trained on synthetic data with that on the original dataset. However, this bi-level optimization approach was time-consuming and unscalable. Further advances included feature matching (Zhao & Bilen, 2023), which improved efficiency by removing dependence on bi-level optimization. Later, Cazenavette et al. (2022) proposed long-range matching by matching training trajectories (MTT), optimizing network parameters over multiple training iterations to better synthesize relevant features for updates.
this section cite: ['b1']

Section: Dataset Distillation with Generative Priors.
Recent advancements have introduced generative priors into the optimization process. GAN-IT (Zhao & Bilen, 2022) shifted the focus from the pixel space to latent codes of pre-trained GANs, optimizing these codes rather than working directly in image space. GLaD (Cazenavette et al., 2023) built on this by incorporating generative priors with StyleGAN for high-resolution datasets, yielding images that more closely match the dataset distribution and improve performance. H-GLaD (Zhong et al., 2024) further enhanced this by focusing on deeper feature layers for hierarchical optimization. Additionally, LD3M (Moser et al., 2024) utilized a latent diffusion model to optimize synthetic datasets directly in the model's latent space, improving performance by refining latent codes through the denoising and diffusion processes. Despite their success on small-resolution datasets, these methods struggle with high-resolution datasets (e.g., 256 × 256, 20 images per class), often being computationally expensive and less efficient, leading to the emergence of more effective distillation methods from generative models.
Generative Dataset Distillation Methods. Recent works (Zhang et al., 2023;Gu et al., 2024;Su et al., 2024) have explored dataset distillation via generative models, moving beyond using generative priors merely to optimize latent codes. Instead, generative dataset distillation trains models to synthesize entire distilled datasets.
Zhang et al. (2023) introduced a class-conditional GAN with a learnable codebook per image, optimized using multiple losses for realism, representativeness, and diversity. Gu et al. (2024) extended this to diffusion models by fine-tuning a pretrained model with representative and diversity losses. Su et al. (2024) proposed D 4 M , which uses Stable Diffusion and replaces random noise with noisy modes during sampling; however, early denoising noise often leads to limited diversity and representativeness. While prior methods rely on complex loss designs and additional training, we propose a training-free approach that achieves both representativeness and diversity without such overhead. Decoupled Dataset Distillation. Recent advances in dataset distillation have introduced decoupled formulations that scale to ImageNet. Yin et al. (2023) proposed SRe 2 L, a Squeeze-Recover-Relabel framework that: (1) squeezes dataset statistics into a model through training, (2) recovers information by optimizing synthetic data to match batchnorm statistics, and (3) boosts performance via soft labels from a pretrained model. Extending this, Shao et al. (2024a) introduced G-VBSM, applying statistical matching to convolutional layers with multi-backbone support, achieving state-of-the-art results from CIFAR-100 to ImageNet-1K. To improve sample fidelity, Sun et al. (2024) proposed a fast, diversity-driven method, distilling ImageNet-1K into 10 images per class within minutes. Shao et al. (2024b) further explored the design space, introducing soft category-aware matching and optimization strategies such as small batches and adaptive learning rates. In contrast to methods that use discriminative models and image optimization-often producing artifacts and poorly aligned samples-we train a generative model to encode the data distribution and recover samples via guided sampling, yielding results more consistent with the original dataset.
this section cite: ['b2', 'b40', 'b17', 'b35', 'b9', 'b25']

Section: Preliminaries
Dataset Distillation: Given a large-scale dataset with the training set
T = {(X i , y i )} N T i=1 , the goal of dataset distillation is to build a smaller synthetic dataset S = {( Xi , ỹi )} N S
i=1 , where N S << N T and X i , Xi are the original and synthetic images with the corresponding class labels y i , ỹi . In addition, the model ϕ T trained on the original training set should achieve similar test performance as the model ϕ S trained on the smaller synthetic dataset; i.e. if A is the accuracy of a model on the test set (T e ), then A(ϕ T ) ∼ A(ϕ S ). During the evaluation, the size of the distilled dataset N S , is set based on the distillation budget, denoted by IPC, the number of images allocated per class.
Our approach builds on the foundations of prior generative models, such as Gu et al. (2024); Su et al. (2024);Zhang et al. (2023), which address dataset distillation by approximating the dataset distribution through sampling diverse and representative instances. This line of work can be characterized as dataset distillation through dataset matching.
Where the objective of the data distillation is defined as
E x∼P (D) ℓ(ϕ T (x), y) -E x∼P (D) ℓ(ϕ S (x), y) < ϵ
where P (D) denotes the real data distribution, and ℓ is a loss function. Note that this formulation is similar to the coreset methods. However, the use of generative models is more flexible because it's not limited to only choosing original samples.
this section cite: ['b9', 'b25', 'b35']

Section: Diffusion Model:
The denoising probabilistic diffusion model (DDPM) is a generative model, G, that learns a mapping between Gaussian noise and the data distribution through a series of T denoising steps. G assumes a Markov chain that gradually adds noise to a sample x 0 in the data distribution, which is called the forward process. The forward process of G is defined as q(x t |x t-1 ) = N ( √ 1 -β t x t-1 , β t I), where β t is the variance schedule for the time step t. In practice, this is done using the reparametrization trick
x t = √ αx 0 + √ 1 -αϵ t , where ϵ t ∼ N (0, I).
Image generation is done by the reverse process of G, where ϵ θ is the noise prediction network, trained to reverse the Markov chain p
θ (x t-1 |x t ) = N (µ θ (x t ), Σ θ (x t )),
where θ corresponds to the parameters of the model and µ θ (x t ), Σ θ (x t ) are the µ and Σ predictions of the denoising models. µ θ (x t ) is computed as follows:
µ θ (x t ) = 1 √ 1 -β t x t - 1 √ 1 -α ϵ θ (x t , t) + σ t z (1)
where z ∼ N (0, 1) and σ t is the variance schedule. ϵ θ (x t , t) is the output of the noise prediction network that is trained to predict the added noise with the simple loss defined as
L θ = ||ϵ θ (x t , t) -ϵ t || 2 .(2)
After training, G can generate samples by sampling from the noise distribution and running the reverse process. In this work, we use a class-conditioned diffusion model G c , where the output of the noise prediction network conditioned with the class c, is denoted as ϵ θ (x t , t, c).
this section cite: []

Section: Diffusion Guidance:
The sampling process of DDPM is equivalent to score-based generative models by interpreting ϵ θ (x t , t) = -√ α ∇ x log p(x t ), where ∇ x log p(x t ) is an estimation of the score function. For the case of classconditioned generation, by using Bayes' rule the score function can be derived as:
∇ x log p(x t |c) = ∇ x log p(x t ) + ∇ x log p(c|x t ), (3)
where ∇ x log p(c|x t ) is the gradient of the class-conditional log-likelihood. It's important to note that ∇ x log p(c|x t ) represents the drift of the diffusion process towards the distribution of the class c. Dhariwal & Nichol (2021a) employ a classifier to estimate the class-conditional log-likelihood and used it as a guidance signal to direct the diffusion process towards the desired class. Later, Ho & Salimans (2021) suggested using a combination of unconditional generation and conditional diffusion (eq. 4) to remove the dependency on the classifier and demonstrated improved results and called this classifier-free guidance. Classifier-free guidance is defined as
εθ (x t , t, c) = (1 -w) • ϵ θ (x t , t, c) -w • ϵ θ (x t , t), (4)
where the w is the guidance scale that controls how strong the guidance is applied.
this section cite: ['b10']

Section: Method
We propose a method for generating diverse and representative class samples by harnessing a diffusion model trained on the target dataset. The core idea is to sample from the denser regions of the data distribution, known as modes, during the reverse process. These modes correspond to clusters of images with similar features and are representative of the class. However, diffusion models often oversample the most prominent modes, which creates redundancies in the distilled dataset, especially when the number of dominant modes for a class is smaller than the desired number of images per class (IPC).
Our three-stage approach, shown in Fig. 3, eliminates the need for fine-tuning while preserving mode diversity. In the first stage, mode discovery, we estimate a diverse set of modes for each class in the dataset. The second stage leverages our proposed mode guidance to control the reverse process and enable sampling from the estimated mode distribution. During sampling, the guidance is applied until the stop guidance-the third stage-is triggered, ensuring control over the quality of the generated samples.
this section cite: []

Section: Mode Discovery
In the mode discovery stage, the main objective is to identify the N modes of a specific class in the original dataset distribution. This discovery is performed using the original dataset in the latent space of the VAE encoder (V enc ). The motivation for this approach is that the generative space captures the overall content of the image rather than discriminative features, which can be limited to specific textures in the image. Any clustering algorithm can be used to estimate the modes for a particular class. In our experiments, we use K-Means centroids, as they are shown to be effective in our ablations with various mode discovery algorithms (see Appendix Section D). Once the modes are identified, our goal is to sample images from these estimated modes.
this section cite: []

Section: Mode Guidance
At the image synthesis stage, our goal is to generate highquality images belonging to a specific class mode. Given a class c and a set of discovered modes for that class denoted as M c = {m 1 , ..., m N }, the mode guidance score is computed for a particular mode m i using the following equation:
g t = (m i -x0 t ),(5)
where x0 t is the predicted denoised latent vector at timestep t during the reverse process. We apply this guidance signal at the x t timestep as follows:
εθ (x t , t, c) = εθ (x t , t, c) + λ • g t • σ t ,(6)
where λ is a scalar that controls the strength of the guidance signal.
To synthesize an image from a particular mode m i , in the diffusion model G the mode guidance score is computed at each iteration of the reverse process using Eq.6. This score represents the direction from the predicted value to the mode m i . The guidance signal is then added to the noise function at the appropriate time step in the diffusion process. By adjusting the strength of the guidance signal, we can regulate the impact of the mode on the generated image.
this section cite: []

Section: Stop Guidance
The reverse diffusion process can be divided into three distinct stages: the chaotic stage (first 20%), the semantic stage (20% to 50%), and the refinement stage (final 50%) (Yu et al., 2023). During the refinement stage, mode guidance becomes unnecessary since its primary purpose is to guide the synthetic image towards the mode in the high semantic space. Our initial experiments revealed that maintaining strong guidance towards a particular mode m i throughout the full reverse process often compromises class fidelity and introduces image artifacts (See Fig. 7b t stop = 0). To address these issues, we introduce the stop guidance mechanism, which involves setting the guidance parameter λ to zero in Equation 6 when the timestep t falls below a timestep t stop during the reverse process. In the Appendices A and J we examine the effects of different stop guidance timesteps (t stop ) on image generation quality.
this section cite: []

Section: Experiments
Datasets and evaluation. To assess our approach's effectiveness, we thoroughly examine the available benchmarks for distilling high-resolution datasets (256 × 256). The  datasets we evaluate include ImageNet-1K, ImageNet-100, ImageNetIDC, ImageNette, and ImageNet-A to ImageNet-E. Additionally, we include results from ImageWoof in the Appendix E. We use two protocols for evaluation: a hardlabel protocol and a soft-label protocol.
The hard-label protocol generates a dataset with its corresponding class labels, trains a network from scratch, and evaluates the network on the original test set. This process is repeated three times for target architectures, and the accuracy mean and standard deviation are reported. Random resize-crop and CutMix are applied as augmentation techniques during the target network's training. For more detailed technical information about the protocol, please refer to Gu et al. (2024). Similar to the existing literature, we evaluate our model in various IPCs ranging from 10 to 100. This protocol was used to evaluate ImageNet-100, ImageNette, and ImageNetIDC datasets.
In soft-label protocol, region-based soft-labels are generated with a pre-trained network as proposed by Sun et al. (2024).
The region-based soft-labels y i,m are generated as follows: y i,m = ϕ T (x i,m ), where ϕ T is the pretrained model and x i,m is the m-th crop of the i-th image. When training a model ϕ S on the distilled dataset the objective loss is L = j m y j,m log ϕ S (x j,m ). For ImageNet-1k evaluation, we follow this protocol. Similarly
to Sun et al. (2024); Gu et al. (2024), we use ResNet-18 as a teacher and student network architecture for this setup. Baselines. We compare several baselines to contextualize the performance of our method. First, we include the pre-trained DiT XL/2, which represents diffusion models without mode guidance. Second, we evaluate MinMax diffusion with DiT XL/2, where the model is fine-tuned to encourage diversity and representativeness. Additionally, for the ImageNette and IDC datasets, we incorporate a classconditioned Latent Diffusion Model (LDM) (Rombach et al., 2022) trained on ImageNet-1k. This allows us to compare the U-Net architecture (used in LDM) with the Transformerbased DiT architecture within the diffusion framework. In our experiments, both DiT and LDM by default use the DDPM sampler. Lastly, to enable a fair comparison with D 4 M (Su et al., 2024) under our hard label protocol, we apply its disentangled diffusion stage without incorporating the soft labels used in their Training Time Matching procedure on ImageNette and IDC datasets. Text-to-Image Diffusion Model. Our method is adaptable to various diffusion models, with optimal performance observed when the model is pre-trained on the target dataset.
To assess the generalizability of our approach, we test it on a general-purpose diffusion model, specifically a text-toimage diffusion model. This evaluation poses challenges due to the potential mismatch between the model's training data and the target dataset. For this setup, the baseline is Text-to-Image Stable Diffusion model without mode guidance, allowing us to demonstrate the impact of integrating mode guidance in the generated dataset. For sampling, we use the class names as a text prompt.
Implementation details. Our pre-trained model G is DiT-XL/2 trained on ImageNet, and the image size is 256 x 256. We use the sampling strategy described in Peebles & Xie (2023), which uses 50 sampling steps using classifier-free guidance with a guidance scale of 4.0. For Mode Guidance, we set λ to 0.1, and in our experiments, we use stop guidance t stop = 25. We use K-means to perform mode discovery; we set k = IP C. We use a single NVIDIA RTX A5000 GPU with 24GB VRAM to run our experiments.
this section cite: ['b9', 'b27']

Section: Comparison with state-of-the-art methods
We compare our method with current SOTA methods on various image datasets and architectures. Our method significantly outperforms previous approaches across various benchmark datasets and target architectures.
ImageNette and ImageIDC. On the ImageNette dataset, our method using DiT achieves notable performance gains of 4.4%, 4.4%, and 2.9% for IPC values of 10, 20, and 50, respectively, surpassing previous state-of-the-art (SOTA) methods (see Tab. 1). Similarly, on the ImageIDC dataset, our method demonstrates improvements of 2.8%, 2.9%, and 2.5% for IPC 10, 20, and 50, respectively, outperforming prior SOTA results. Tab. 1 highlights that our approach consistently enhances the performance of DiT and LDM. Furthermore, in the Text-to-Image evaluation mode, our method with guidance surpasses Stable Diffusion on both datasets, as illustrated in Fig. 4a and 4b.
ImageNet-100 and ImageNet-1K. Tab. 2 shows comparison to SOTA in ImageNet-100 in IPC 10 and 20 in various target architectures. Our method surpasses the previous SOTA by 1.3%, 1.6%, and 1.4% in IPC 20 for various target architectures. It also outperformes the MinMax diffusion approach in IPC 10 and achieves the best performance with the ResNetAP-10 target architecture while delivering the second-best results for ConvNet-6 and ResNet-18 architectures. It is important to note that our method is substantially more computationally efficient compared to IDC and Min-Max (see Computational Cost below). We also compare our method with SOTA in ImageNet-1K on the soft-label protocol on IPC 10 and 50 in Fig. 4d. Our method achieves SOTA outperforming previous SOTA by 1.3% and 1.6%.
While using a Text-to-Image diffusion in ImageNet-1k, our method shows an improvement of 3.4% and 2.3% in IPC 10 and IPC 50 over Stable Diffusion as shown in Fig. 4c.
Performance on Larger Models. To evaluate the scalability of our approach, we assess its performance on larger backbone architectures-ResNet-50 and ResNet-101-under the IPC50 setting on ImageNet-1k. Table 3 compares our method against several existing approaches across ResNet-18, ResNet-50, and ResNet-101. Our method consistently outperforms prior work on both larger backbones, demonstrating strong generalization to high-capacity models. Notably, while ResNet-18 achieves 69.8% accuracy when trained on the full dataset, our method achieves 86% accuracy using only 3.9% of the data, highlighting both its data efficiency and strong relative performance.
this section cite: []

Section: Computational Cost.
Our method achieves state-of-the-art performance on all datasets, except ImageNet-100, where the best-performing method, IDC-1 (Kim et al., 2022), has slightly better results than ours but with much higher computational cost. For example, MinMax (Gu et al., 2024) took 10 hours to produce a distilled dataset for ImageNet-100 with IPC-10, while IDC-1 (Kim et al., 2022) took over 100 hours for the same. The optimization strategy proposed in IDC-1 (Kim et al., 2022) can not scale up to the ImageNet-1K, and MinMax diffusion requires expensive fine-tuning of the diffusion model, especially for larger datasets like
Table 3. Comparison of top-1 accuracy across different methods and backbone architectures (ResNet-18, ResNet-50, ResNet-101) under the IPC50 setting on ImageNet. A dash (-) indicates that the result was not reported.
this section cite: ['b12', 'b9', 'b12', 'b12']

Section: Method ResNet-18 ResNet-50 ResNet-101
Full Dataset 69.8 80.9 81.9 SR 2 L (Yin et al., 2023) 46.8 ± 0.2 55.6 ± 0.3 60.8 ± 0.5 G-VBSM (Shao et al., 2024a) 51.8 ± 0.4 58.7 ± 0.3 61.0 ± 0.4 RDED (Sun et al., 2024) 56.5 ± 0.1 -61.2 ± 0.4 EDC (Shao et al., 2024b) 58.0 ± 0.2 64.3 ± 0.2 64.9 ± 0.2 D 4 M (Su et al., 2024) 55.2 ± 0.1 62.4 ± 0.1 63.4 ± 0.1 Ours 60.2 ± 0.1 64.6 ± 0.4 67.7 ± 0.4
ImageNet-1k. In contrast, we use pre-trained diffusion models to create a distilled dataset with no additional computational cost for fine-tuning and minimal overhead for mode discovery. For comparison, our method takes 0.42 hours to generate a synthetic dataset for ImageNet-100 with IPC-10. This highlights the computational efficiency of our model compared to previous approaches.
this section cite: ['b33', 'b27', 'b25']

Section: Comparison with Generative Prior Methods.
We compare our method against GLaD, H-GLaD, and LM3D in their cross-architecture setup, using AlexNet, VGG11, ResNet18, and ViT for performance evaluation. The evaluation was done by running the evaluation five times per architecture and reporting the mean performance across all the architectures. We evaluate our model in 5 subsets: A, B, C, D, and E of ImageNet. Our method was trained using the hardlabel protocol. Tab. 4 shows that our method outperforms previous approaches in this setup. Additionally, these methods face scalability challenges for large datasets such as ImageNet-1K or higher IPC values (>50) due to their high time and space complexity.
this section cite: []

Section: Ablation Experiments
Effect of each component. To assess the impact of each proposed component, we incrementally evaluated the following: 1) Mode Discovery, 2) Mode Guidance, and 3) Stop Guidance. Mode Discovery involves performing K-means per class on the original dataset and selecting the closest sample to the k-means centroid. We conduct the evaluation on the ImageNette dataset with IPC 10, and report the accuracy of ConvNet-6, ResNet10 with average pooling, and ResNet18. Tab. 5 demonstrates that using diffusion with mode guidance enhances mode discovery and that stop guidance is crucial for achieving improved performance.
this section cite: []

Section: Visuzalizing t-SNE.
To analyze the distilled dataset's coverage, we visualize a t-SNE plot of the distilled dataset from the DiT, MinMax Diffusion, and our method. Fig. 5 illustrates that the DiT distilled dataset is mostly contained in one region of the original dataset distribution, while MinMax Diffusion extends to a broader area of the data distribution. However, the distilled dataset from our method covers a broader area of the data distribution than both methods.
this section cite: []

Section: Representativeness and Diversity.
While t-SNE provides a qualitative visualization of diversity, it does not present the complete picture. We are also interested in representativeness. With this in mind, our goal is to empirically measure diversity and representativeness in the t-SNE space described above. To measure diversity, we calculate the pairwise distance of all samples within a class for the distilled dataset and report the minimum distance per sample. To measure representativeness, we calculate the mean distance to the 50 closest samples in the original dataset, where a greater distance indicates lower representativeness and a smaller distance indicates higher representativeness.
We compare the diversity and representativeness of each class for DiT, MinMax diffusion, and our method as shown in Fig. 6. For clarity in visualization, we plot 1representativeness, so that higher values indicate higher representativeness. Our experiment indicates that DiT examples show partial representative and partial diversity. On the other hand, MinMax produces more diverse examples than DiT, although some classes lack diversity. Our method demonstrates that our samples are both diverse and representative. Furthermore, we provide additional results about representativeness and diversity in the Appendix B.  Mode Guidance with DDIM. Our approach, similar to classifier guidance (Nichol & Dhariwal, 2021), can be incorporated into DDIM using Algorithm 1. In Table 6, we compare the effect of our approach in DDPM and DDIM across LDM and DiT diffusion architectures. Our results demonstrate the effectiveness of our method with denoising samplers in both architectures, showcasing its flexibility with respect to diffusion architecture and sampler choice. This highlights the significant impact of our approach in enhancing the performance while being adaptable with different denoising diffusion models.
Algorithm 1 Mode Guidance with DDIM sampling, given a diffusion model ϵ θ (x t ), an estimated mode m k and mode guidance scale λ.
Input: estimated mode m k and mode guidance scale λ x T ← sample from N (0, I) for all t from T to 1
do g t = (m i -xt 0 ) ε ← ϵ θ (x t ) - √ 1 -ᾱt • λ • g t x t-1 ← √ ᾱt-1 xt- √ 1-ᾱtε √ ᾱt + √ 1 -ᾱt-1 ε end for return: x 0
this section cite: []

Section: Conclusion
Dataset distillation is an important task of condensing information from large training sets. Despite several efforts, the distilled datasets have limited representativeness and diversity in their synthetic samples. Our proposed method, leveraging latent diffusion with mode guidance, addresses this limitation and achieves state-of-the-art performance in dataset distillation across multiple benchmarks and experimental setups. Notably, our approach outperforms previous methods without requiring fine-tuning, as demonstrated by our results on ImageNette, ImageIDC, ImageNet-100, and ImageNet-1K. We conducted a detailed analysis of our method's key components and demonstrated their utility through rigorous ablation studies. Furthermore, we showed that our approach is compatible with general diffusion models, such as Text-to-Image Stable Diffusion, even when the training data does not overlap with the target dataset.
this section cite: []

Section: References
Ref_id:b0 Title: End-to-end incremental learning Year: (2018)
Ref_id:b1 Title: Dataset distillation by matching training trajectories Year: (2022)
Ref_id:b2 Title: Generalizing dataset distillation via deep generative prior Year: (2023)
Ref_id:b3 Title: Post training mixed precision quantization of neural networks using first-order information Year: (2023)
Ref_id:b4 Title: Towards mixed-precision quantization of neural networks via constrained optimization Year: (2021)
Ref_id:b5 Title: Super-samples from kernel herding Year: (2010)
Ref_id:b6 Title: Diffusion models beat gans on image synthesis Year: (2021)
Ref_id:b7 Title: Diffusion Models Beat GANs on Image Synthesis Year: (2021)
Ref_id:b8 Title: Centripetal sgd for pruning very deep convolutional networks with complicated structure Year: (2019)
Ref_id:b9 Title: Filter pruning via geometric median for deep convolutional neural networks acceleration Year: (2019)
Ref_id:b10 Title: Classifier-free diffusion guidance Year: (2021)
Ref_id:b11 Title: Denoising Diffusion Probabilistic Models Year: (2020)
Ref_id:b12 Title: Dataset condensation via efficient synthetic-data parameterization Year: (2022)
Ref_id:b13 Title: Soft-label anonymous gastric x-ray image distillation Year: (2020)
Ref_id:b14 Title: Dataset distillation via factorization Year: (2022)
Ref_id:b15 Title: Learning efficient convolutional networks through network slimming Year: (2017)
Ref_id:b16 Title: Efficient dataset distillation using random feature approximation Year: (2022)
Ref_id:b17 Title: Latent dataset distillation with diffusion models Year: (2024)
Ref_id:b18 Title: Improved Denoising Diffusion Probabilistic Models Year: (2021)
Ref_id:b19 Title: Scalable diffusion models with transformers Year: (2023)
Ref_id:b20 Title: Incremental classifier and representation learning Year: (2017)
Ref_id:b21 Title: High-Resolution Image Synthesis With Latent Diffusion Models Year: (2022)
Ref_id:b22 Title: Generalized large-scale data condensation via various backbone and statistical matching Year: (2024)
Ref_id:b23 Title: Elucidating the design space of dataset condensation Year: (2024)
Ref_id:b24 Title: Rapid: A single stage pruning framework Year: (2022)
Ref_id:b25 Title: Dˆ4: Dataset distillation via disentangled diffusion model Year: (2024)
Ref_id:b26 Title: Secdd: Efficient and secure method for remotely training neural networks (student abstract) Year: (2021)
Ref_id:b27 Title: On the diversity and realism of distilled dataset: An efficient dataset distillation paradigm Year: (2024)
Ref_id:b28 Title: Distilling dataset into generative model Year: (2023)
Ref_id:b29 Title:  Year: (2018)
Ref_id:b30 Title: Herding dynamical weights to learn Year: (2009)
Ref_id:b31 Title: Quantized convolutional neural networks for mobile devices Year: (2016)
Ref_id:b32 Title: Eq-net: Elastic quantization neural networks Year: (2023)
Ref_id:b33 Title: Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective Year: (2023)
Ref_id:b34 Title: Training-free energy-guided conditional diffusion model Year: ()
Ref_id:b35 Title: Dataset condensation via generative model Year: (2023)
Ref_id:b36 Title: Dataset condensation with differentiable siamese augmentation Year: (2021)
Ref_id:b37 Title: Synthesizing informative training samples with gan Year: (2022)
Ref_id:b38 Title: Dataset condensation with distribution matching Year: (2023)
Ref_id:b39 Title: Dataset condensation with gradient matching Year: (2021)
Ref_id:b40 Title: Hierarchical features matter: A deep exploration of gan priors for improved dataset distillation Year: (2024)
Ref_id:b41 Title: Performance comparison with pre-trained diffusion models and other state-of-the-art methods on ImageWoof. All the results are reproduced by us for the 256×256 resolution. The missing results are due to out-of-memory. The best results are marked as bold Year: ()
Ref_id:b42 Title: Our method outperforms the previous SOTA across various IPC values for different target architectures. Notably, our method demonstrates superior performance in all IPC values for the ResNet-18 architecture, achieves SOTA in IPC 10 Year: (2009)
