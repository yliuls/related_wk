Title: Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation
Abstract: Cross-domain few-shot segmentation (CD-FSS) is proposed to pre-train the model on a sourcedomain dataset with sufficient samples, and then transfer the model to target-domain datasets where only a few samples are available for efficient fine-tuning. There are majorly two challenges in this task: (1) the domain gap and (2) fine-tuning with scarce data. To solve these challenges, we revisit the adapter-based methods, and discover an intriguing insight not explored in previous works: the adapter not only helps the fine-tuning of downstream tasks but also naturally serves as a domain information decoupler. Then, we delve into this finding for an interpretation, and find the model's inherent structure could lead to a natural decoupling of domain information. Building upon this insight, we propose the Domain Feature Navigator (DFN), which is a structure-based decoupler instead of loss-based ones like current works, to capture domain-specific information, thereby directing the model's attention towards domain-agnostic knowledge. Moreover, to prevent the potential excessive overfitting of DFN during the source-domain training, we further design the SAM-SVN method to constrain DFN from learning sample-specific knowledge. On target domains, we freeze the model and fine-tune the DFN to learn target-specific knowledge specific. Extensive experiments demonstrate that our method surpasses the state-of-the-art method in CD-FSS significantly by 2.69% and 4.68% MIoU in 1-shot and 5-shot scenarios, respectively.

Section: Introduction
In recent years, advancements in large-scale annotated datasets and deep neural networks (Chen et al., 2014;Long Figure 1: Cross-domain few-shot segmentation (CD-FSS) aims to transfer the source-domain-trained model to target domains for efficient learning with scarce data. By inserting adapters into the common network structure (e.g., HSNet) for CD-FSS, we find an insight not explored in previous works: adapter naturally serves as a domain information decoupler based on its structure instead of training losses, which "grabs" domain information from the en-coder+decoder structure and encourages the model to learn domain-agnostic information on the source domain. et al., 2015;Zhao et al., 2017;Yuan et al., 2020) have driven the rapid progress of large vision models (Dosovitskiy et al., 2020;Kirillov et al., 2023;Zhang et al., 2024), resulting in impressive segmentation task outcomes. However, when applied to downstream tasks, these models face significant challenges when there is a substantial distributional difference between upstream pretraining data and downstream data, where collecting downstream data may be difficult. To address this issue, the Cross Domain Few-Shot Segmentation (CD-FSS) task (Lei et al., 2022) has been introduced (see Figure 1 top). CD-FSS involves pre-train a model on a source-domain dataset and then adapting it to generate pixel-level predictions for unseen categories in targetdomain datasets with only a few annotated samples, which still remains challenging.
There are majorly two challenges in the CD-FSS task: (1) a huge domain gap between source and target domains, making it difficult for the generalization from the source dataset to the target dataset; and (2) extremely limited target domain data, making it challenging for the model to adapt to the distribution of the novel domain. To address the second challenge, currently, a group of methods based on adapters have been proposed (Houlsby et al., 2019;Mahabadi et al., 2021;Hu et al., 2021), which fixes the backbone network and only finetune the extra appended structures. In this paper, we revisit the adapter-based methods, and discover an intriguing insight not explored in prior works: adapters not only help the fine-tuning on downstream tasks but also naturally serves as a domain information decoupler. This finding indicates that the adapter can address two challenges simultaneously: (1) by decoupling the source domain information into a domain-agnostic part, which aids in generalizing from the source to the target domain, and (2) by parameter-efficient fine-tuning to adapt to downstream data.
In this paper, we first delve into this phenomenon for an interpretation. We first conduct experiments to verify which factors determine the adapter serving as a decoupler, including the adapter's insertion position and structure. We find this phenomenon holds only when adapters are inserted into deeper layers of the backbone network with scratch training and residual connections, without applying any domaindecoupling loss. This indicates the structural design could lead to the inherent capability of domain-information decoupling, which inspires us to design a structure-based domain decoupler, instead of the loss-based decoupler adopted by current works (Tzeng et al., 2015;Motiian et al., 2017;Kang et al., 2019;Lu et al., 2022).
Based on these findings and interpretations, we propose the Domain Feature Navigator (DFN, Fig. 1), a structurebased domain decoupler built on adapters with specific structures and positions. In the source-domain phase, DFN absorbs domain-specific knowledge without any domaindecoupling losses, directing the model's attention toward acquiring domain-agnostic information. Then, during the target-domain phase, we fine-tune the DFN to capture targetspecific features. The fusion of domain-specific features with the model's domain-agnostic features serves to align the feature spaces for each domain.
However, the implicit absorption of domain information could potentially lead to excessive overfitting of sourcedomain samples, as the learning of domain information can also be understood as a kind of overfitting (to the source domain), but there are no labels to guide the magnitude of overfitting like other loss-based decouplers. To further address this problem, we introduce SAM-SVN to constrain the DFN from excessive overfitting in source-domain training. Specifically, we tailor the sharpness-aware minimization (Foret et al., 2020) to apply it to the singular value matrix of the DFN. By doing so, excessive overfitting is avoided but the absorption of domain information is maintained.
To sum up, our primary contributions are as follows:
• To the best of our knowledge, we are the first to discover the phenomenon that the adapter naturally serves as a decoupler, which we then delve into for an interpretation.
• Building upon this finding and interpretation, we propose the DFN to decouple source domain information into domain-agnostic knowledge and domain-specific one, solely based on the adapter's structure and position.
• We propose the SAM-SVN to avoid the potential excessive overfitting introduced by source-domain training of the DFN, while maintaining the DFN's absorption of domain information.
• Extensive experiments show the effectiveness of our work on four different CD-FSS scenarios. Our model significantly outperforms the state-of-the-art method.
this section cite: ['b4', 'b54', 'b49', 'b9', 'b27', 'b52', 'b29', 'b21', 'b33', 'b22', 'b45', 'b35', 'b26', 'b32', 'b13']

Section: Adapter Naturally Serves as Decoupler: Phenomenon and Interpretation
In this section, we conduct experiments to demonstrate that an adapter can naturally act as a decoupler. Furthermore, we examine which type of adapter is suitable for this role and investigate the underlying reasons. Following the standard CD-FSS setting, we use Pascal (Shaban et al., 2017) as the source dataset and the other four datasets as target datasets.
this section cite: ['b39']

Section: Adapter Decouples Domain Information
The network structure studied in this paper is shown in Fig. 2 (top), which consists of a backbone network, an encoder, and a decoder. We choose this structure because it is widely recognized as versatile architecture (Min et al., 2021). We first attempt to attach a simple adapter (implemented as a 1 × 1 convolution) to the backbone with the residual connection, then train them jointly in the source domain.
Then, given ResNet50 (He et al., 2016) as the backbone, we measure the domain similarity by the CKA 1 (Kornblith et al., 2019;Zou et al., 2022;Tong et al., 2024b;Zou et al., 2024c) similarity (lower values indicate more domain-specific information) based on the backbone's stage-4 output before and after attaching the adapter (Fig. 2). As shown in Table 1 (row1,2), after training with an adapter attached, the CKA is lower, which means the adapter captures domain-specific information, therefore decreasing the domain similarity.
Subsequently, we study the change that the adapter brings to the output of the encoder (which is different from the backbone network). We also measure the domain similarity by the CKA similarity through the encoder's output with and without the adapter attached to the backbone network (BKB). As shown in Table 1 (row 3,4), the CKA is higher after the adapter is attached, which implies that the encoder focuses more on the domain-agnostic information.
Without the adapter (Fig. 2 top), the encoder needs to si-
1 See the Appendix B for detailed formulations.   2). The adapter captures domain-specific information, guiding the encoder and decoder to learn domaininvariant knowledge. BKB: backbone network. multaneously capture both domain-specific and domainagnostic information extracted by the feature extractor. After attaching the adapter (Fig. 2 bottom), the adapter captures domain-specific information, therefore decreasing the BKB's domain similarity. Consequently, the encoder parameters could focus less on the domain-specific information and pay more attention to the domain-agnostic one, increasing the domain similarity. In other words, the adapter decouples the domain information from the original model.
this section cite: ['b34', 'b18', 'b28', 'b55']

Section: Why adapter can serve as a decoupler?
The above phenomenon inspires us to ask: Can all types of adapters decouple features, and why are adapters able to decouple features? In this section, we delve into this phenomenon by exploring two factors that determine whether an adapter can be a decoupler: position and structure.
this section cite: []

Section: POSITION
We categorize the insertion positions of adapters into three types as shown in Fig 3 : (1) shallow layers of the fixed backbone;
(2) deep layers of the fixed backbone;
(3) between learnable encoder and decoder. We use CKA to assess if position affects an adapter's decoupling ability.
Following Table 1, if the CKA of the backbone output decreases and that of the encoder output increases, we can recognize the attached adapter as a decoupler. As shown in Table 2, the adapter can serve as a decoupler only when the adapter is inserted into the deeper layers of the backbone.
Comparing the positions (2) and (3), the distinction is that Position baseline(w/o adapter) BKB shallower BKB deeper between enc-dec BKB encoder BKB encoder BKB encoder BKB encoder FSS-1000 0.5371 0.0709 0.4679↓ 0.0737↑ 0.4788↓ 0.0733↑ 0.5371-0.0695↓ Deepglobe 0.4147 0.0498 0.3736↓ 0.0452↓ 0.3687↓ 0.0679↑ 0.4147-0.0469↓ ISIC 0.5266 0.0678 0.5028↓ 0.0633↓ 0.4653↓ 0.0724↑ 0.5266-0.0658↓ Chest X-ray 0.4865 0.0554 0.4639↓ 0.0532↓ 0.4390↓ 0.0626↑ 0.4865-0.0529↓
Table 2: Verify the impact of different insertion points of the adapter on its decoupling ability by domain similarity.
(2) is positioned between the fixed, pretrained backbone and subsequent learnable modules. The pretrained backbone is fixed on the source domain, while the adapter is trained from scratch. Such a difference guides the backbone network to extract general features and leads the adapter to learn more from the source domain to capture domain information.
Comparing insertion methods (1) and ( 2), the distinction is that (2) is located in the deeper layers of the fixed backbone, whereas (1) is positioned in the shallower layers. As is well known, the deeper the neural network is, the more semantic information its features can encompass. For cross-domain tasks, the knowledge learned by deeper network layers tends to be more domain-specific. To verify it, we first visualized feature maps with different examples using insertion method (2), and the results are shown in Fig. 4. For each example, the adapter outputs more complex features to focus on the objects' profile, such as the eagle's wings, fish tail and fins, and the clock face of the clock tower, which means that inserting the adapter into deeper layers enables it to capture features that are more semantic and complex.
Figure 4: The visualization results of the feature maps with and without the adapter attached to the backbone network.
Furthermore, as shown in Table 3, we evaluate the CKA similarity between features extracted at deep layers and that of the first convolution layer (conv1) in the source domain following (Zou et al., 2022). The higher the CKA similarity, the simpler the features are. The row results indicate that the deeper the layer is, the lower the CKA would be. This shows high-level features are more complex, which means it could be more overfitting to data, e.g., domain information in the first row. Comparing these two rows, we can see that the CKA is lower after the feature passes through the adapter, which means the adapters' features are more complex and could capture more domain-specific knowledge.
this section cite: ['b55']

Section: Conclusion for position:
(1) The disparity between the backbone network (pretrained, fixed) and the adapter (learnable, from scratch), enables the adapter to acquire specific features during training.
(2) The semantics become richer in deeper layers, for cross-domain tasks, the knowledge learned by deeper network layers tends to be more domainspecific. These two factors enable the adapter to act as a decoupler when it is inserted into the deeper layers of the fixed, pretrained backbone network.
this section cite: []

Section: References
Ref_id:b0 Title: Multimodality helps few-shot 3d point cloud semantic segmentation Year: (2024)
Ref_id:b1 Title: Rethinking few-shot 3d point cloud semantic segmentation Year: (2024)
Ref_id:b2 Title: Few-shot segmentation without meta-learning: A good transductive inference is all you need? Year: (2021)
Ref_id:b3 Title: Lung segmentation in chest radiographs using anatomical atlases with nonrigid reg-istration Year: (2013)
Ref_id:b4 Title: Semantic image segmentation with deep convolutional nets and fully connected crfs Year: (2014)
Ref_id:b5 Title: Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation Year: (2019)
Ref_id:b6 Title: Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic) Year: (2019)
Ref_id:b7 Title: Deepglobe 2018: A challenge to parse the earth through satellite images Year: (2018)
Ref_id:b8 Title: Few-shot semantic segmentation with prototype learning Year: (2018)
Ref_id:b9 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b10 Title: The pascal visual object classes (voc) challenge Year: (2010)
Ref_id:b11 Title: Self-support few-shot semantic segmentation Year: (2022)
Ref_id:b12 Title: Model-agnostic metalearning for fast adaptation of deep networks Year: (2017)
Ref_id:b13 Title: Sharpness-aware minimization for efficiently improving generalization Year: (2020)
Ref_id:b14 Title: Few-shot learning with graph neural networks Year: (2017)
Ref_id:b15 Title: A kernel two-sample test Year: (2012)
Ref_id:b16 Title: A broader study of cross-domain few-shot learning Year: (2020)
Ref_id:b17 Title: Semantic contours from inverse detectors Year: (2011)
Ref_id:b18 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b19 Title: Apseg: Auto-prompt network for crossdomain few-shot semantic segmentation Year: (2024)
Ref_id:b20 Title: Adapt before comparison: A new perspective on cross-domain few-shot segmentation Year: (2024)
Ref_id:b21 Title: Parameter-efficient transfer learning for nlp Year: (2019)
Ref_id:b22 Title: Low-rank adaptation of large language models Year: (2021)
Ref_id:b23 Title: Restnet: Boosting crossdomain few-shot segmentation with residual transformation network Year: (2023)
Ref_id:b24 Title: Automatic tuberculosis screening using chest radiographs Year: (2013)
Ref_id:b25 Title: When do flat minima optimizers work? Year: (2022)
Ref_id:b26 Title: Decoupling representation and classifier for long-tailed recognition Year: (2019)
Ref_id:b27 Title: Segment anything Year: (2023)
Ref_id:b28 Title: Similarity of neural network representations revisited Year: (2019)
Ref_id:b29 Title: Cross-domain few-shot semantic segmentation Year: (2022)
Ref_id:b30 Title: Fss-1000: A 1000-class dataset for few-shot segmentation Year: (2020)
Ref_id:b31 Title: Fully convolutional networks for semantic segmentation Year: (2015)
Ref_id:b32 Title: Domaininvariant feature exploration for domain generalization Year: (2022)
Ref_id:b33 Title: Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks Year: (2021)
Ref_id:b34 Title: Hypercorrelation squeeze for few-shot segmentation Year: (2021)
Ref_id:b35 Title: Unified deep supervised domain adaptation and generalization Year: (2017)
Ref_id:b36 Title: Normalization layers are all that sharpness-aware minimization needs Year: (2023)
Ref_id:b37 Title: Cross-domain few-shot segmentation via iterative support-query correspondence mining Year: (2024)
Ref_id:b38 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b39 Title: Oneshot learning for semantic segmentation Year: (2017)
Ref_id:b40 Title: Prototypical networks for few-shot learning Year: (2017)
Ref_id:b41 Title: Prior guided feature enrichment network for few-shot segmentation Year: (2020)
Ref_id:b42 Title: Dynamic knowledge adapter with probabilistic calibration for generalized few-shot semantic segmentation Year: (2024)
Ref_id:b43 Title: Lightweight frequency masker for cross-domain few-shot semantic segmentation Year: (2024)
Ref_id:b44 Title: The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions Year: (2018)
Ref_id:b45 Title: Simultaneous deep transfer across domains and tasks Year: (2015)
Ref_id:b46 Title: Matching networks for one shot learning Year: (2016)
Ref_id:b47 Title: Panet: Few-shot image semantic segmentation with prototype alignment Year: (2019)
Ref_id:b48 Title: Prototype mixture models for few-shot semantic segmentation Year: (2020)
Ref_id:b49 Title: Object-contextual representations for semantic segmentation Year: (2020)
Ref_id:b50 Title: Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning Year: (2019)
Ref_id:b51 Title: Featureproxy transformer for few-shot segmentation Year: (2022)
Ref_id:b52 Title: Personalize segment anything model with one shot Year: (2024)
Ref_id:b53 Title: Sg-one: Similarity guidance network for one-shot semantic segmentation Year: (2020)
Ref_id:b54 Title: Pyramid scene parsing network Year: (2017)
Ref_id:b55 Title: Margin-based few-shot class-incremental learning with class-level overfitting mitigation Year: (2022)
Ref_id:b56 Title: Flatten longrange loss landscapes for cross-domain few-shot learning Year: (2024)
Ref_id:b57 Title: Attention temperature matters in vit-based cross-domain few-shot learning Year: (2024)
Ref_id:b58 Title: Compositional few-shot class-incremental learning Year: (2024)
