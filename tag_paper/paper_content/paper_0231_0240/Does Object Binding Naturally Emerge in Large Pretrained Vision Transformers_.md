Title: Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
Abstract: Object binding, the brain's ability to bind the many features that collectively represent an object into a coherent whole, is central to human cognition. It groups low-level perceptual features into high-level object representations, stores those objects efficiently and compositionally in memory, and supports human reasoning about individual object instances. While prior work often imposes object-centric attention (e.g., Slot Attention) explicitly to probe these benefits, it remains unclear whether this ability naturally emerges in pre-trained Vision Transformers (ViTs). Intuitively, they could: recognizing which patches belong to the same object should be useful for downstream prediction and thus guide attention. Motivated by the quadratic nature of self-attention, we hypothesize that ViTs represent whether two patches belong to the same object, a property we term IsSameObject. We decode IsSameObject from patch embeddings across ViT layers using a quadratic similarity probe, which reaches over 90% accuracy. Crucially, this object-binding capability emerges reliably in DINO, CLIP, and ImageNet-supervised ViTs, but is markedly weaker in MAE, suggesting that binding is not a trivial architectural artifact, but an ability acquired through specific pretraining objectives. We further discover that IsSameObject is encoded in a low-dimensional subspace on top of object features, and that this signal actively guides attention. Ablating IsSameObject from model activations degrades downstream performance and works against the learning objective, implying that emergent object binding naturally serves the pretraining objective. Our findings challenge the view that ViTs lack object binding and highlight how symbolic knowledge of "which parts belong together" emerges naturally in a connectionist system. 1

Section: Introduction
Humans naturally parse scenes into coherent objects [1] (e.g., grouping features such as rounded shape, smooth surface, and muted color into the mug) and further ground their identities in context (e.g., recognizing my coffee mug on the desk rather than just a mug). This is assumed to be made possible by what cognitive scientists call object binding [2], the brain's ability to group an object's low-level features (color, shape, motion, etc.) into a unified representation. This in turn enables objects to be stored efficiently and compositionally in memory and used as high-level symbols for reasoning. The binding problem is a genuine computational challenge, as evidenced by humans' limited competence in conjunction-search tasks [3] and clinical dissociations such as Balint's syndrome, where feature perception remains intact but binding breaks down [4]. If AI systems could replicate the human ability for object binding, that may help them ground symbols for perception and exploiting compositionality [5]. The key question is: do current AI systems solve the binding problem? IsSameObject, with scores near 1 for same-object pairs and near 0 for different-object pairs. (b) Downstream tasks that benefit from strong object binding include instance segmentation and visual reasoning (e.g., locating and counting objects with specific features), where patches triggered by certain features are bound to the rest of their object to allow extraction of the entire object.
Object binding has received little attention in mainstream AI research. Cognition-inspired models [6,7] build in human-like object-based attention. By contrast, mainstream vision models are assumed to implicitly learn to handle multiple objects from training data, yet empirical studies show they often "attend" only to the most salient regions and overlook the rest [8]. While ViT attention scores can capture global image structure and salient regions (often corresponding to target objects) [9], empirical evidence shows that self-attention tends to group patches by low-level feature similarity rather than reliably producing object-level binding [10]. Object-centric methods like Slot Attention [11] fix this by allocating a small set of learnable slots that compete for token features, enforcing binding by design. However, whether AI vision models, especially leading ViTs, can achieve robust object binding without explicit mechanisms remains an open question.
Cognitive scientists have questioned whether ViTs can bind objects at all: arguing that they lack mechanisms for dynamically and flexibly grouping features [5]; they lack recurrence necessary for iterative refinement of object representations [12,7]; and as purely connectionist models, they appear incapable of true symbolic processing [8]. However, these architectural limitations do not preclude binding from emerging through learning. If a model encodes whether two patches belong to the same object (IsSameObject), this signal can guide attention and improve prediction [9,13]. Human-labeled data also reflects object-level structure, so ViTs can acquire binding by imitation. This suggests that ViTs may learn to bind objects directly from large-scale training data, without requiring explicit architectural inductive biases.
Here, we ask whether object binding naturally emerges in large, pretrained Vision Transformers, which is a question that matters for both cognitive science and AI. We propose IsSameObject (whether two patches belong to the same object) and show that it is reliably decodable (with 90.20% accuracy) using a quadratic similarity probe starting from mid-layers of the transformer layers. This effect is robust across DINO, CLIP and ImageNet-supervised ViTs, but largely absent in MAE, suggesting that binding is an acquired ability rather than a trivial architectural artifact. Across the ViT's layer hierarchy, it progressively encodes IsSameObject in a low-dimensional projection-space on top of the features of the object, and it guides self-attention. Ablating IsSameObject from model activations hurts downstream performance and works against the pretraining objective.
Our main contributions are as follows: (i) We demonstrate that object binding naturally emerges in large, pretrained Vision Transformers, challenging the cognitive-science assumption that such binding isn't possible given their architecture. (ii) We show that ViTs encode a low-dimensional signature of IsSameObject (whether two patches belong to the same object) on top of their feature representations. (iii) We suggest that learning-objective-based inductive biases can enable object binding, pointing future work toward implicitly learned object-based representations.
this section cite: ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10', 'b4', 'b11', 'b6', 'b7', 'b8', 'b12']

Section: Related Work
Object Binding in Cognitive Science and Neuroscience. The object binding problem asks how the brain integrates features that are processed across many distinct cortical areas into coherent object representations [14]. The concept of bindingfoot_2 rests on three key hypotheses: First, visual processing is widely understood to be hierarchical, parallel, and distributed across the cortex [16][17][18][19][20]. Second, we perceive the world primarily in terms of objects, rather than as a collection of scattered features [12,1]. This abstraction is fundamental to both perception and interaction with the world, allowing us to recognize, reason about, and manipulate our environment effectively [21][22][23]. Third, feature binding requires a mechanism that correctly assigns features, represented in spatially distinct cortical areas, to their corresponding object [15,24,2]. This third hypothesis is where the core of the binding problem lies, and it has been a longstanding point of debate among neuroscientists and cognitive scientists [25][26][27].
Despite their substantial difference, vision transformers (ViTs) share several key computational parallels with the mammalian visual system: they both rely on parallel, distributed and hierarchical processes. More importantly, ViTs do have two of the three architectural and computational elements hypothesized to enable binding in the brain. The explicit position embeddings in ViTs resemble spatial tagging and the spatiotopic organization observed in the ventral stream [28,26]; and the self-attention mechanism is akin to dynamic tuning and attentional modulation, which are thought to be primary mechanisms for object binding [26,29,30] (although attention is believed to be of recurrent nature in the brain [31,32]). These parallels position ViTs as potential computational models for exploring object binding in both artificial and biological systems.
Object-Centric Learning. Motivated by how humans naturally reason about individual objects, Object-Centric Learning (OCL [11]) aims to represent a scene as a composition of disentangled object representations. While segmentation only partitions an image into object masks, OCL goes further by encoding each object into its own representation [33]. Unsupervised approaches such as MONet [34], IODINE [33], and especially Slot Attention [11] encode scenes into a small, permutation-invariant set of "slots" that are iteratively refined, producing robust object representations on both synthetic [11,35] and real-world data [36,37] and enabling compositional generation and manipulation [38][39][40]. However, since Slot Attention is added as an external module rather than integrated into the transformer architecture, it introduces additional challenges for scaling and training [41]. Other explicit objectcentric approaches include Tensor Product Representations [42] and Capsule Networks [43].
Instead of object-centric approaches that explicitly enforce object-level attention, we propose an alternative view that ViTs may already encode implicit object-level structure. Prior work has assumed this and attempted to group patches into objects directly from activations or attention maps ViTs, using methods like clustering [44] or GraphCut [45]. [46] conduct a behavioral experiment where participants judge whether two dots belong to the same object at varying distances, and show that patch-level feature similarity in self-supervised ViTs supports object-based grouping. Building on this line of work, we show that ViT patch embeddings intrinsically encode whether any two patches belong to the same object, and analyze how this information is structured through probing.
this section cite: ['b13', 'b15', 'b16', 'b17', 'b18', 'b19', 'b11', 'b0', 'b20', 'b21', 'b22', 'b14', 'b23', 'b1', 'b24', 'b25', 'b26', 'b27', 'b25', 'b25', 'b28', 'b29', 'b30', 'b31', 'b10', 'b32', 'b33', 'b32', 'b10', 'b10', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45']

Section: Binding in Transformers.
Binding has received growing recognition in transformer-based machine learning research and binding failures are seen as examples of performance breakdowns in modern applications [47][48][49][50]. Diffusion models rely on binding attributes to entities, and failures cause attribute leakage (e.g., both a dog and a cat end up wearing sunglasses and a sun-hat) [48,47]. Visionlanguage models face similar binding challenges, struggling with differentiating multiple objects with feature conjunctions [50]. Despite these binding failures, transformers still demonstrate some binding capability, yet the underlying mechanism is not well understood. Feng and Steinhardt [51], Dai et al. [52] study binding in language models, showing that attributes (e.g., "lives in Shanghai") are linked to their subjects (e.g., "Alice") via a low-dimensional binding-ID code that is added to the activation and can be edited to swap or redirect relations. Binding mechanisms in vision transformers remain unexplored, and our study aims to fill this gap.
this section cite: ['b46', 'b47', 'b48', 'b49', 'b47', 'b46', 'b49', 'b50', 'b51']

Section: Assessing Object Binding in ViTs through
IsSameObject (a) (b) (c)
Figure 2: IsSameObject predictions distinguish objects in highly complex scenarios [53].
Quadratic probe results for DINOv2-Large at layer 18 show that overlapping deer can be distinguished and that disconnected regions of the same deer are correctly retrieved in (c).
this section cite: ['b52']

Section: Probing IsSameObject representations
Vision Transformers (ViTs) tokenize images by dividing them into a grid of fixed-size patches [54].
Because the token is the minimal representational unit, any grouping of features into objects must arise through relations between tokens, not within them. The only mechanism ViTs have for such cross-token interaction is scaled dot-product attention, where attention scores can be viewed as dynamic edge weights in a graph that route information between tokens [5]. Therefore, if ViTs perform any form of object binding, we expect to observe a pairwise token-level representation that indicates whether two patches belong to the same object, which we term IsSameObject.
Since object binding is the ability to group an object's features together, decoding IsSameObject reliably from ViT patch embeddings would provide direct evidence of object binding (and its representation) in the model. We adopt probing, which takes measurements of ViT activations with lightweight classifiers [55], to determine whether IsSameObject is encoded or unrecoverable by simple operations.
Formally, we define the IsSameObject predicate on a pair of token embeddings (x
(ℓ) i , x (ℓ) j ) at layer ℓ by IsSameObject x (ℓ) i , x (ℓ) j = ϕ x (ℓ) i , x (ℓ) j , ϕ : R d × R d → [0, 1],
where ϕ scores the probability that tokens i and j belong to the same object.
Here, we ask whether models reliably encode IsSameObject and, if so, what mechanisms they use to do so. We consider the following hypotheses about how IsSameObject may be encoded in the model's activations:
• It may be linear (recoverable by a weighted sum of features) or fundamentally quadratic (recoverable only through pairwise feature interactions).
• It is a pairwise relationship versus a pointwise mapping (i.e. the model first maps each patch to a discrete object identity or class, then compares).
• The model tells objects apart using only broad class labels or object identities-i.e., it may rely on class-level recognition ("dog vs. chair") instead of explicitly binding pixels to objects, as class labels already encode a coarse notion of object identity.
• The signal is stored in a few specialized dimensions versus distributed across many dimensions. In the former case, binding information would be isolated to a small subset of channels, while in the latter it would be encoded diffusely (e.g., as rotated combinations of features) such that no single dimension carries the signal on its own.
To test these hypotheses, we decode IsSameObject using several probe architectures, each parameterized by a learnable matrix W and a scalar bias b. All probes are constructed to be symmetric in their inputs, reflecting the constraint IsSameObject(x, y) = IsSameObject(y, x). Throughout, σ(•) denotes the sigmoid function.
1. Linear probe.
IsSameObject lin (x, y) = σ(W x + W y + b) , W ∈ R 1×d , b ∈ R.
this section cite: ['b53', 'b4', 'b54']

Section: Diagonal quadratic probe (specialized dimensions)
.
IsSameObject diag (x, y) = σ x ⊤ W y + b ,
where W ∈ R d×d is constrained to be diagonal, so the probe uses only d parameters, each corresponding to a single feature dimension.
this section cite: []

Section: Quadratic probe (distributed)
.
IsSameObject quad (x, y) = σ x ⊤ W ⊤ 1 W 2 y + b ,
where W 1 , W 2 ∈ R k×d with k ≪ d. To enforce symmetry, we set W 2 = SW 1 , where S is a diagonal matrix with entries in {±1}, yielding a low-rank quadratic form with O(kd) parameters.
this section cite: []

Section: Object-class / object-identity probes (pointwise).
We first map each embedding to a probability distribution:
p = softmax(W c x + b), q = softmax(W c y + b),
where W c is trained using multiclass cross-entropy on object-class labels (and similarly W N for object-identity labels). The pointwise IsSameObject score is then defined as the inner product of the two distributions:
IsSameObject class/identity (x, y) = p ⊤ q = Nc c=1
p(c) q(c).
this section cite: []

Section: 3.2
IsSameObject is best decodable in quadratic form We extract DINOv2-Large [13] activations at each layer and train the probes on the ADE20K dataset [56] using cross-entropy loss for all pairwise probes to classify same-object vs. different-object patch pairs (see Figure 2 for IsSameObject visualizations). Figure 3 shows probe accuracy across layers. To test our hypotheses about how IsSameObject is represented, we compare:
• Linear < quadratic probes: (Diagonal) quadratic probes significantly outperform linear ones, suggesting that IsSameObject is a quadratic representation, consistent with the quadratic form used by the self-attention mechanism.
• Quadratic (pairwise) > object-identity probes (pointwise): Mapping each patch to a discrete object identity and then comparing them pointwise underperforms direct pairwise comparison of embeddings, as the pointwise approach discards information by collapsing continuous representations into discrete classes.
• Quadratic > object class probes: The model encodes not only shared object class but also finer-grained identity cues (e.g., distinguishing two identical cars of the same make and model).
• Full > diagonal quadratic probes: TheIsSameObject information is more distributed across dimensions rather than restricted to specific channels.
this section cite: ['b12', 'b55']

Section: Object binding emerges broadly across self-supervised ViTs
We extend our analysis beyond DINOv2 to a broader set of pretrained Vision Transformers, including CLIP, MAE, and fully supervised ViTs. To enable direct comparison, we standardize input patch coverage by resizing all inputs so that each model processes the same spatial patch divisions as the DINOv2 family. Under this setup, every probe starts from the same trivial baseline of 72.6% accuracy, which corresponds to always predicting "different", reflecting the class imbalance that most patch pairs do not belong to the same object in the dataset.
Table 1 reports IsSameObject decoding accuracy across models. DINO models show the strongest binding signal, with large and giant variants exceeding +16 percentage points over baseline. ImageNetsupervised ViT and CLIP also exhibit clear object-binding ability, though to a lesser degree. In contrast, MAE yields poor object-binding performance, suggesting that binding is an acquired ability under specific pretraining objectives rather than being a universal property of all vision models. Our findings thus produce a much wider coverage of ViTs and we provide an understanding of potential reasons why binding emerges:
• DINO. The contrastive teacher-student loss enforces consistency across augmented views containing the same objects. This objective encourages the model to learn object-level features that persist under augmented views [9].
• Supervised ImageNet training. Although ImageNet labels correspond to the dominant object in each image [57], class-level supervision still provides useful signals for object identity, consistent with the strong performance of our object-class probes.
• CLIP. By aligning images with text captions, CLIP effectively assigns each object a symbolic label (e.g., "the red car"), which can act like a pointer that pulls together all patches of that object. This supervision likely encourages patches from the same object to cluster in feature space.
this section cite: ['b8', 'b56']

Section: Extracting the Binding Subspace of ViT Representations

this section cite: []

Section: Decomposing IsSameObject from features
Following the linear feature hypothesis [58], and similar to [51], we assume that at layer ℓ each token embedding decomposes into a "feature" part and a "binding" part:
h (ℓ) (x t ) = f (ℓ) (x t , c) + b (ℓ) (x t ),
where f (ℓ) (x t , c) ∈ R d encodes all attributes of token x t (texture, shape, etc.) given context c = {x 1 , . . . , x T }, excluding any information about which other tokens it binds with, and b (ℓ) (x t ) ∈ R d encodes the binding information that determines which other tokens belong to the same object (i.e., the IsSameObject relation).
Consider two identical patches x Ai and x Bi at corresponding positions of identical objects A and B in the same image, and let their residual be ∆ ABi . It may be tempting to cancel the feature term directly. Indeed, without positional encoding (see proof in Appendix A.4.1), we have f (ℓ) (x Ai ) = f (ℓ) (x Bi ), since for identical tokens the positional encoding is the only signal that can differentiate their cross-token interactions.
We can approximate f (ℓ) (x Ai ) ≈ f (ℓ) (x Bi ), since the two patches are visually identical, appear in nearly the same context, and any positional difference can be offloaded into the binding component. This yields:
∆ ABi = h(x Ai ) -h(x Bi ) = f (x Ai ) -f (x Bi ) + b(x Ai ) -b(x Bi ) ≈ b(x Ai ) -b(x Bi ).
If ∆ ABi remains roughly consistent across patch pairs with the same index i, then b(x Ai ) and b(x Bi ) can form linearly separable clusters, which can thus serve as object identity representations. However, this becomes problematic in natural images, where identical patches are rare.
Instead, we take a supervised approach to decoding the binding component. Our quadratic probe serves as a tool for separating binding from feature information within each token (Fig. 4). Conceptually, the quadratic probe can be viewed as projecting an activation h into the IsSameObject subspace, yielding b
(ℓ) query (x) = h (ℓ) (x) ⊤ W 1 and b (ℓ) key (x) = h (ℓ) (x) ⊤ W 2 ,
and then measures the dot-product similarity between two projected vectors. Given that natural image datasets contain numerous objects where b is the primary distinguishing factor, the probe should be optimized to discover a direction that isolates b. With this strategy we can separate the binding signal from the rest of the representation.
The observation in [51] that binding vectors remain meaningful under linear combination, and become hard to discriminate when they are close together, is consistent with this interpretation. In later ablation studies, we use our trained quadratic probe via b (ℓ) (x) = h (ℓ) (x) ⊤ W .
this section cite: ['b57', 'b50', 'b50']

Section: A Toy Experiment: distinguishing identical objects and similar looking objects
To probe the limits of object binding in ViTs, we construct a test image with two identical red cars, a third red car of a different brand, and a red boat. This setup lets us track IsSameObject representations across layers by evaluating three distinctions: different object-class but similar appearance, same class with subtle differences, and exact duplicates. As expected, these distinctions become progressively harder. We chose natural objects rather than abstract shapes because both the ViT and our probe are trained on real-world images, which allows us to analyze binding in a nontrivial setting.
Layer 0 Layer 6 Layer 12 Layer 18 Layer 23 Car A Car B Car C Boat Boat Car C Car A (a) (b) IsSameObject 1.0 0.0 To analyze where binding emerges, we plot the IsSameObject scores predicted by our trained quadratic probe (Figure 5). We observe that, from early to mid-layers, the model increasingly discerns the local object (the one to which each patch belongs). Surprisingly, from mid-layers to later layers, the model shifts toward class-based grouping, increasingly treating all red cars as the same. Binding emerges in the middle of the network and is then progressively lost towards the top.
The IsSameObject representation is low-dimensional. We use four identical red-car images and split each one into patches using exactly the same grid alignment. We perform principal component analysis (PCA) on the residuals sets {∆ BA , ∆ CA , ∆ DA }, where
∆ BA = h Bi -h Ai ≈ b Bi -b Ai
and visualize the first three components (see Figure 6). ∆ BA , ∆ CA , ∆ DA fall into three linearly separable clusters in the first three principal component space. The separation of these clusters in a very small number of principal directions demonstrates that IsSameObject lies in a low-dimensional subspace: patches from the same object instance map to closely aligned binding vectors, and different instances are linearly separable with large margins.
Mid-layers capture local objects, and higher layers shift towards grouping patches by object class. A surprising observation is the sudden increase in the cross-object IsSameObject score (Fig. 5) in the mid-layers of the DINOV2 model for instances of the same class (Fig. 5). This is consistent with prior work showing that ViTs represent different types of information at different layers [59]. At the same time, token-position decodability drops in deeper layers (see Appendix A.4.3), suggesting that the model is deliberately discarding positional information. Our interpretation is that the network initially relies on positional cues to support binding, since location is necessary to disambiguate tokens that share similar feature content. In later layers, the network removes positional signals once they are no longer useful and repurposes capacity for semantically relevant object structure. Our findings are consistent with experimental evidence from the ventral stream in the brain, showing that while the retinotopic organization of early ventral areas is necessary for perception and binding, global spatial information is instead processed and maintained by the dorsal stream [60][61][62]. The percentage in parentheses indicates the variance explained by that principal component.
this section cite: ['b58', 'b59', 'b60', 'b61']

Section: Attention weights (query-key similarity) correlate with IsSameObject
In Section 3.2 we showed that IsSameObject is best decoded quadratically. Since self-attention is also a quadratic interaction, binding information in the residual stream at layer ℓ can in principle guide how attention is allocated at layer ℓ + 1, allowing the model to selectively route attention within the same object to build a coherent object-level representation.
To test this, we compute the Pearson correlation between attention weights and the IsSameObject scores (see Fig. 7 and Appendix A.5). In mid-level layers, we observe a positive but modest correlation, indicating that the model does make use of the IsSameObject signal when allocating attention. The modest strength of the effect is expected, because attention serves many roles beyond binding.
this section cite: []

Section: Ablation of
IsSameObject hurts downstream performance and works against the minimization of the pretraining loss  • Uninformed Ablation: Randomly shuffle b(x i ) across patches in the image at a specified ratio.
• Informed Ablation (Injection): Using ground-truth instance masks, we inject the true IsSameObject signal by linearly combining the mean object direction with each patch's binding vector b
i : bi = (1 -α) 1 |I| j∈I b object,j + α b object,i .
We evaluate the semantic and instance segmentation performance with retrained segmentation heads on a subset of ADE20K under these variations. We also evaluate the teacher-student self-distillation loss as employed in DINO (see Appendix A.6 for details).
Results show that uninformed ablation, which randomly shuffles the binding vector, reduces segmentation performance, whereas injecting the mean object direction improves accuracy. Ablating IsSameObject with random shuffling leads to a noticeable gradual increase in the DINO loss, suggesting that ablation of IsSameObject works against this pretraining loss.
this section cite: []

Section: Limitations
We assume the trained probe cleanly splits each patch embedding into "feature" and "binding" components, a simplification that would benefit from further empirical exploration. We do not establish a causal relationship between object binding and downstream task performance, and further analysis is needed to understand how different pretraining objectives induce object binding. Finally, our downstream evaluations focus only on segmentation, leaving open whether these emergent binding signals also benefit other vision tasks such as visual reasoning. More broadly, this paper studies object binding at the patch level; more general forms of binding are not explored and are left for future work.
this section cite: []

Section: Conclusion
In this paper, we show that object binding naturally emerges in large, pretrained vision transformers, especially in DINOv2, and this effect is consistent across multiple models. We also show that it is an acquired rather than innate ability through comparisons across vision models. IsSameObject, whether two patches belong to the same object, is reliably decodable and lies in a low-dimensional latent space. Our results emergent object binding arises as a natural solution to self-supervised learning objectives. More broadly, our study bridges what psychologists identify as object binding with emergent behavior in ViTs, challenges the belief that ViTs lack such ability.
Looking ahead, we suggest that addressing binding failures in vision models may not require explicit object-centric modules (e.g., Slot Attention [11]), but could instead be achieved by strengthening the intrinsic object-binding mechanisms of ViTs through tailored training objectives or minimal architectural modifications. Another important direction for future work is to study how bound object representations interact with one another, potentially through low-dimensional "object files" [63]. Together, these efforts will deepen our understanding of how symbolic processing of objects can emerge in connectionist models.
this section cite: ['b10', 'b62']

Section: References
Ref_id:b0 Title: Objects and attention: The state of the art Year: (2001)
Ref_id:b1 Title: The binding problem Year: (1996)
Ref_id:b2 Title: Illusory conjunctions in the perception of objects Year: (1982)
Ref_id:b3 Title: The interaction of spatial and object pathways: Evidence from balint's syndrome Year: (1997)
Ref_id:b4 Title: On the binding problem in artificial neural networks Year: (2020)
Ref_id:b5 Title: Learning what and where to attend Year: (2018)
Ref_id:b6 Title: Modeling attention and binding in the brain through bidirectional recurrent gating Year: (2024)
Ref_id:b7 Title: How structured are the representations in transformer-based vision encoders? an analysis of multi-object representations in visionlanguage models Year: (2024)
Ref_id:b8 Title: Emerging properties in self-supervised vision transformers Year: (2021)
Ref_id:b9 Title: Self-attention in vision transformers performs perceptual grouping, not attention Year: (2023)
Ref_id:b10 Title: Object-centric learning with slot attention Year: (2020)
Ref_id:b11 Title: Capturing the objects of vision with neural networks Year: (2021)
Ref_id:b12 Title: Learning robust visual features without supervision Year: (2023)
Ref_id:b13 Title: The what and why of binding: the modeler's perspective Year: (1999)
Ref_id:b14 Title: The neural binding problem (s) Year: (2013)
Ref_id:b15 Title: Functional specialisation in the visual cortex of the rhesus monkey Year: (1978)
Ref_id:b16 Title: Segregation of form, color, movement, and depth: anatomy, physiology, and perception Year: (1988)
Ref_id:b17 Title: Object vision and spatial vision: two cortical pathways Year: (1983)
Ref_id:b18 Title: Distributed hierarchical processing in the primate cerebral cortex Year: (1991)
Ref_id:b19 Title: The functional architecture of the ventral temporal cortex and its role in categorization Year: (2014)
Ref_id:b20 Title: Organization in vision: Essays on gestalt perception Year: ()
Ref_id:b21 Title: Hierarchical structure in perceptual representation Year: (1977)
Ref_id:b22 Title: Recognition-by-components: a theory of human image understanding Year: (1987)
Ref_id:b23 Title: The correlation theory of brain function Year: (1994)
Ref_id:b24 Title: Beyond binding: from modular to natural vision Year: (2025)
Ref_id:b25 Title: Binding, spatial attention and perceptual awareness Year: (2003)
Ref_id:b26 Title: The binding problem Year: (1999)
Ref_id:b27 Title: Focused attention in the perception and retrieval of multidimensional stimuli Year: (1977)
Ref_id:b28 Title: The role of neural mechanisms of attention in solving the binding problem Year: (1999)
Ref_id:b29 Title: Solving the binding problem: Assemblies form when neurons enhance their firing rate-they don't need to oscillate or synchronize Year: (2023)
Ref_id:b30 Title: Going in circles is the way forward: the role of recurrence in visual inference Year: (2020)
Ref_id:b31 Title: Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior Year: (2019)
Ref_id:b32 Title: Multi-object representation learning with iterative variational inference Year: (2019)
Ref_id:b33 Title: Unsupervised scene decomposition and representation Year: (2019)
Ref_id:b34 Title: Conditional object-centric learning from video Year: (2021)
Ref_id:b35 Title: Bridging the gap to real-world object-centric learning Year: (2022)
Ref_id:b36 Title: Simple unsupervised object-centric learning for complex and naturalistic videos Year: (2022)
Ref_id:b37 Title:  Year: (2023)
Ref_id:b38 Title: Learning to compose: Improving object centric learning by injecting compositionality Year: (2024)
Ref_id:b39 Title: Leveraging image augmentation for object manipulation: Towards interpretable controllability in object-centric learning Year: (2023)
Ref_id:b40 Title: Are we done with object-centric learning? arXiv preprint Year: (2025)
Ref_id:b41 Title: Towards discrete object representations in vision transformers with tensor products Year: (2023)
Ref_id:b42 Title: Dynamic routing between capsules Year: (2017)
Ref_id:b43 Title: Recasting generic pretrained vision transformers as object-centric scene encoders for manipulation policies Year: (2024)
Ref_id:b44 Title: Tokencut: Segmenting objects in images and videos with selfsupervised transformer and normalized cut Year: (2023)
Ref_id:b45 Title: Affinity-based attention in self-supervised transformers predicts dynamics of object grouping in humans Year: (2023)
Ref_id:b46 Title: Object-attribute binding in text-to-image generation: Evaluation and control Year: (2024)
Ref_id:b47 Title: Token merging for training-free semantic binding in text-to-image synthesis Year: (2024)
Ref_id:b48 Title: Is the reversal curse a binding problem? uncovering limitations of transformers from a basic generalization failure Year: (2025)
Ref_id:b49 Title: Understanding the limits of vision language models through the lens of the binding problem Year: (2024)
Ref_id:b50 Title: How do language models bind entities in context? arXiv preprint Year: (2023)
Ref_id:b51 Title: Representational analysis of binding in large language models. arXiv e-prints Year: (2024)
Ref_id:b52 Title:  Year: (2025)
Ref_id:b53 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2020)
Ref_id:b54 Title: Understanding intermediate layers using linear classifier probes Year: (2016)
Ref_id:b55 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b56 Title: Imagenet large scale visual recognition challenge Year: (2015)
Ref_id:b57 Title: The linear representation hypothesis and the geometry of large language models Year: (2023)
Ref_id:b58 Title: Deep vit features as dense visual descriptors Year: (2021)
Ref_id:b59 Title: Retinotopic organization of human ventral visual cortex Year: (2009)
Ref_id:b60 Title: What and where pathways Year: (2008)
Ref_id:b61 Title: The dorsal visual pathway represents objectcentered spatial relations for object recognition Year: (2022)
Ref_id:b62 Title: The reviewing of object files: Objectspecific integration of information Year: (1992)
Ref_id:b63 Title: Attention is all you need Year: (2017)
