Title: MULTIMODAL ALIGNED SEMANTIC KNOWLEDGE FOR UNPAIRED IMAGE-TEXT MATCHING
Abstract: While existing approaches address unpaired image-text matching by constructing cross-modal aligned knowledge, they often fail to identify semantically corresponding visual representations for Out-of-Distribution (OOD) words. Moreover, the distributional variance of visual representations associated with different words varies significantly, which negatively impacts matching accuracy. To address these issues, we propose a novel method namely Multimodal Aligned Semantic Knowledge (MASK), which leverages word embeddings as bridges to associate words with their corresponding prototypes, thereby enabling semantic knowledge alignment between the image and text modalities. For OOD words, the representative prototypes are constructed by leveraging the semantic relationships encoded in word embeddings. Beyond that, we introduce a prototype consistency contrastive learning loss to structurally regularize the feature space, effectively mitigating the adverse effects of variance. Experimental results on the Flickr30K and MSCOCO datasets demonstrate that MASK achieves superior performance in unpaired matching.

Section: INTRODUCTION
Image-text matching has become an essential technique for various applications, such as visual question answering Özdemir & Akagündüz (2024); Lerner et al. (2024), image captioning Fu et al. (2024); Wang et al. (2024a), cross-modal retrieval Wang et al. (2024b); Li et al. (2024b) and so forth. Due to the heterogeneous representations and asymmetry of information between images and texts, accurately learning cross-modal semantic correspondences remains a challenging problem. Although training on large-scale paired image-text data has substantially improved matching accuracy, collecting and annotating such data at scale is often impractical in real-world scenarios. To reduce the dependence of image-text matching on paired data, the unpaired image-text matching paradigm Huang et al. (2022) was first proposed, in which domain-specific paired images and texts are assumed to be unavailable during model training. Inspired by the fact that the human brain can correlate arbitrary images and texts well while does not need to learn from large-scale paired images and texts, the unpaired image-text matching is implemented by modeling human brain-like knowledge, which is multimodal aligned and used to associate visual and linguistic information.
Note that unimodal visual or linguistic knowledge has been widely used for vision and language understanding task Chen & Zhao (2023). There are also some works Li et al. (2025); Gao et al. (2025) directly combining these two types of knowledge together, but the resulting knowledge is not multimodal aligned. Another alternative is Multimodal Aligned Conceptual Knowledge Huang et al. (2024a), which establishes correspondences between prototypical region representations and words, as shown in Figure 1 (a). However, these knowledge-based methods still face the following issues: 1) OOD words have not been thoroughly investigated. Existing knowledge-based methods fail to leverage the underlying semantic structure to transfer the visual prototypes of known words to OOD words; 2) The influence of distributional variance has been largely overlooked.
The region representations corresponding to different words exhibit substantial appearance variations. Consequently, certain instances that deviate substantially from the distributional mean may be prone to misclassification into other words; 3) The raw region representation is insufficient in effectively capturing the semantic relationships between words. The raw region representation is predominantly influenced by the co-occurrence relationships among regions. However, there is no inherent relationship between semantic relevance and co-occurrence patterns. For instance, while 'human' and 'hat' often co-occur in visual contexts, 'human' and 'gentleman' may exhibit a higher semantic similarity.
To address these issues, we propose a new method namely Multimodal Aligned Semantic Knowledge for unpaired image-text matching, which establishes semantic alignment between prototypical region representations and word embeddings, as shown in Figure 1 (b). We summarize our key contributions as follows:
• We propose a novel cross-modal semantic alignment method, MASK, which constructs representative prototypes for OOD words by exploiting the intrinsic relationships among word embeddings, thereby enhancing the model's generalization ability in unpaired imagetext matching.
• We introduce a prototype consistency contrastive learning loss to structurally regularize the feature space, which explicitly encourages region representations associated with the same word to align closely with their prototype, thereby mitigating the adverse impact of distributional variance.
• We incorporate external knowledge from pre-trained word vectors as auxiliary supervision signals, which establishes a relation-preserving equivariant mapping between region representations and word embeddings, enabling the region representations to effectively capture semantic relationships among words.
2 RELATED WORK
this section cite: ['b24', 'b15', 'b3', 'b9', 'b1', 'b17', 'b4']

Section: MODEL-BASED MATCHING
Extensive model-based matching works have been made on measuring the semantic correlation between vision and language. To our knowledge, Socher et al. Socher et al. (2013) might propose the first framework of Visual-Semantic Embedding (VSE) to correlate images and their class labels in a two-stream manner. Lee et al. Lee et al. (2018) propose a Stacked Cross Attention Network(SCAN) to discover all latent alignments by using regions of the image and words in a sentence as context.
The SCAN has been extensively studied from various aspects such as memory modeling Huang et al. (2021), context modeling Zhang et al. (2020) and graph structure Liu et al. (2020). Later, by using millions or billions of paired images and texts for supervised model learning, many models Li et al. (2020); Pan et al. (2023); Wu et al. (2024); Li et al. (2024a); Pham et al. (2024); Ge et al. (2024) based on multimodal versions of Transformer have been proposed and have achieved remarkable results. However, while these existing methods achieve relatively strong performance, they rely heavily on extensive paired image-text datasets for supervised training, which significantly restricts their applicability.
this section cite: ['b30', 'b14', 'b8', 'b38', 'b21', 'b16', 'b25', 'b34', 'b26', 'b5']

Section: MULTIMODAL ALIGNED SEMANTIC KNOWLEDGE
In addition to the semantic concepts, the studied knowledge also has another important property of cross-modal one-to-one alignment. For each word, its semantically related objects in different regions often exhibit diverse visual appearances, which could easily lead to confusion in practice. Therefore, rather than align each word to multiple related regions in an one-to-many manner, the MASK aligns each word to a single prototypical region, with the goal to alleviate the issue of appearance variation. In particular, we formulate the knowledge as a set of semantic concepts having paired multimodal representations {(w k , v k )} k=1,...,K , where w k and v k are the word embedding and prototypical region representation of the k-th semantic concept, respectively, and K is the total number of semantic concepts.
As shown in Figure 2, for each word, we compute the word embeddings w k by using pre-trained word vectors. For each region, we first extract the raw region representations r j (j = 1, . . . , J k ) by feeding a bounding box and an image into the pre-trained object detection model Faster-RCNN. Then we extract region representations µ j by utilizing the Prototype-Aware Encoder (P AE) h, with r j as the input:
µ j , σ j = h(r j ; Θ h ),(1)
where Θ h is the parameters of h and σ j represents the variance of distribution. Finally, we compute the prototypical region representations v k by averaging all related region representations {µ j } j=1,...,J k :
v k = 1 J k J k j=1 µ j ,(2)
where the J k indicates the number of regions for the k-th semantic concept.
this section cite: []

Section: IMAGE EMBEDDING BRANCH
Given a batch B of paired regions and words, we first obtain raw region representations R = {r n } n=1,...,B (R ∈ R B×M ), where M is the dimension of r n . We then extract region representations µ using the P AE model h, which takes R as input and consists of a Fully Connected (FC) layer followed by three self-attention layers:
µ, σ = h(R; Θ h ),(3)
where the mean µ ∈ R B×Z and variance σ ∈ R B×Z are used to preserve the information of the raw region representations R by using the Feature Restoration Module (F RM ) g comprising a self-attention layer and two FC layers:
R = g(µ, σ, z; Θ g ),(4)
where the z is a random vector sampled from a standard normal distribution and Θ g is the parameters of g. The Z represents the feature dimension of latent space. As shown in Figure 2, the P AE model h and F RM model g are trained jointly using the information retention loss function L ir :
L ir = D KL ( N (µ, σ 2 ) || N (0, 1) ) + E (rn,r n )∼(R,R ) [ r n -r n 2 2 ],(5)
where the D KL ( N (µ, σ 2 ) || N (0, 1) ) implies that the data distribution in the latent space gradually approaches the standard normal distribution. The
E (rn,r n )∼(R,R ) [ r n -r n 2 2
] measures the difference between the reconstructed raw region representations R and the raw region representations R. The loss L ir ensures that the mean µ retains a significant amount of information from the raw region representations R.
Inspired by clustering theory and contrastive learning, we design a prototype consistency contrastive learning loss L cl to reduce the influence of distributional variance between prototypes and their related region representations. The loss L cl employs prototypes as class centers, maximizing the similarity between region representations and their corresponding prototypes while minimizing similarity with other prototypes, thereby achieving intra-class aggregation and inter-class separation. Compared to traditional instance-to-instance contrastive learning, L cl introduces prototypes as global semantic representatives, explicitly aggregating instances of the same class around their corresponding prototypes. This process constructs a more structured and discriminative representation space, enabling the model to capture clearer semantic boundaries. The loss L cl is defined as follows (Appendix G):
L cl = - 1 B B k=1 log exp(v k • µ + /τ ) B n=1 exp(v k • µ n /τ ) ,(6)
where the µ + refers to the region representations associated with the prototypical region representations v k , i.e., positive examples. The hyperparameter τ regulates the capacity of the model to distinguish between negative examples. The loss L cl encourages all region representations corresponding to the same word to be closer to each other, while driving the region representations corresponding to different words farther apart, which effectively mitigates the impact of variance among different words on the similarity computation.
this section cite: []

Section: TEXT EMBEDDING BRANCH
Given a batch B of paired regions and words, we obtain word embeddings V = {w k } k=1,...,B (V ∈ R B×N ) by utilizing the pretrained word vectors, where N is the dimension of w k . Pre-trained word embeddings typically exhibit well-structured semantic properties, where semantically related words are mapped to vectors that are close to each other in the embedding space. To enable region representations to effectively capture semantic correlations between words, we utilize a Modality Transfer Model (M T M ) f with three self-attention layers and three FC layers that can map the mean µ output by the P AE model h into the word embedding space:
V = f (µ; Θ f ),(7)
where the Θ f is the parameters of the M T M model f and V ∈ R B×N represents the predicted word embeddings. The model f is a relation-preserving equivariant mapping that lays the foundation for constructing prototypical region representations corresponding to OOD words. Formally, for any two region representations µ i and µ j , the function f should satisfy (Appendix B):
d s (f (µ i ; Θ f ), f (µ j ; Θ f )) ∝ d s (µ i , µ j ),(8)
where the distance metric d s captures the pairwise relations between representations within each modality. The P AE model h and M T M model f are trained jointly using the cross-modal alignment loss function L cm (Appendix C, E and F):
L cm = E[(1 -cos( w i w i 2 , w i w i 2 ))] + E[((cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 ))) 2 ], (9
)
where w i , w j ∈ V (i = j) and w i ∈ V . The loss function L cm enforces the predicted word embeddings V to gradually converge toward the word embeddings V , while simultaneously ensuring that the region representations effectively capture the semantic relationships between words.
this section cite: []

Section: KNOWLEDGE-BASED IMAGE-TEXT MATCHING
To decide whether a given image and a text are matched or not, we first obtain a set of raw region representations R = {r i } i=1,...,I (R ∈ R I×M ) using the Faster-RCNN above and a set of parsed words through tokenization operation implemented via NLTKfoot_0 , as shown in Figure 2. Then, we use the knowledge as a cross-modal bridge to represent all the words into the corresponding prototypical region representations U = {v j } j=1,...,J (U ∈ R J×Z ). For the set of regions R, we extract region representations µ ∈ R I×Z by utilizing the P AE model h. Finally, we obtain the desired global similarity score s for the given image and text as:
s = ρ( µ • U T ),(10)
where ρ(•) denotes the max-mean pooling operation, which first performs max pooling along the column dimension and then mean pooling along the row dimension of the input matrix.
However, the scope of knowledge is inherently limited and heavily reliant on the volume of paired data available in public datasets. The vocabulary size supported by the pre-trained word vectors significantly surpasses the scale of the existing knowledge. Therefore, for OOD words relative to the knowledge, their corresponding word embeddings can typically be obtained by leveraging pre-trained word vectors. To fully utilize these OOD words, we first sample m paired multimodal representations {(w q , v q )} m q=1 from the knowledge. Then, we calculate the similarity scores {s q } m q=1 between the m word embeddings {w q } m q=1 and the word embedding w out :
{s q } m q=1 = sof tmax(w out • {w q } m q=1 ).(11)
By utilizing the sampled prototypical region representations {v q } m q=1 as base vectors and the similarity scores {s q } m q=1 , we can obtain the prototypical region representation v out corresponding to word embedding w out :
v out = m q=1 s q • v q . (12
)
In unpaired image-text matching, constructing prototypical region representations based on semantic similarities between words enables the effective utilization of information from OOD words.
To ensure the semantic quality of the visual prototypes constructed for OOD words, we select the top-m paired multimodal representations from the knowledge whose word embeddings are most relevant to OOD words. This selection strategy is motivated by the local linearity property of word embeddings on the semantic manifold. Semantically related words lie close to each other in the embedding space and approximately reside in a locally linear subspace. Consequently, the topm neighbors provide the most informative directions for reconstructing the corresponding visual representations. Moreover, the L cm constrains local alignment between the word embedding space and the visual prototype space, making nearest neighbors in the embedding space more likely to preserve geometric relationships in the prototype space, thereby reducing reconstruction bias. In this way, the top-m neighbors effectively capture the most salient semantic and structural information needed for accurate and robust prototype estimation. To obtain these top-m semantic neighbors for OOD words, we first normalize all word embeddings {w q } K q=1 in the knowledge, and denote the normalized embeddings as { wq wq } K q=1 . Similarly, the normalized embedding of OOD words as wout wout . The similarity between wout wout and { wq wq } K q=1 are computed as:
{s q } K q=1 = w out w out • { w q w q } K q=1 . (13
)
The top-m nearest neighbors {s q } m q=1 are then selected based on {s q } K q=1 to support subsequent visual prototype construction.
this section cite: []

Section: MODEL TRAINING
The studied knowledge mainly contains dataset-independent semantic concepts, with the goal to be generally applicable to different scenarios. The semantic concepts are multimodal, which includes objects and attributes in images, and nouns and adjectives in texts. To obtain them, we resort to publicly available dataset Visual Genome (VG) Krishna et al. (2017) foot_1 and collect corresponding words and regions. For the textual knowledge, we obtain various words from synsets in the dataset. For the visual knowledge, we detect regions from images and then associate them with the words.
After collecting a set of words and their semantically related image regions from publicly available datasets, we train our model on these paired data to construct MASK. The loss of the entire training process is expressed as L:
L = L ir + λ 1 L cm + λ 2 L cl ,(14)
where λ 1 and λ 2 are trade-off factors for balancing different losses. By optimizing the loss L, the region representations exhibit properties of high cohesion and low coupling, indicating that representations corresponding to the same word become more compact and semantically consistent.
this section cite: ['b13']

Section: RE-RANKING EXTENSION
The proposed MASK is a knowledge-based approach, which differs significantly from existing datadriven models. Due to this distinction, it is expected to exhibit complementary properties when combined with existing models. We extend MASK into a re-ranking method to re-rank the initial results produced by existing multimodal models.
Taking the sub-task of image retrieval as an example, given a text query and a gallery of L images, an existing model can compute similarity scores and produce a similarity vector s ∈ R L×1 . By sorting the values of s in descending order, the model ranks the images and identifies the top-k candidates, denoted as sk ∈ R k×1 . For the text query and the top-k retrieved images, we then compute an additional similarity vector s k ∈ R k×1 using MASK in an unpaired image-text matching setting.
Finally, the two similarity vectors are combined using a balancing factor α:
ŝk = ZS(s k ) + α • ZS(s k ),(15)
where ZS represents the Z-Score normalization and ŝk is the new similarity vector that can be used to re-rank the top-k images to improve the rank of matched images.
this section cite: []

Section: EXPERIMENTS

this section cite: []

Section: DATASETS AND METRICS
We test the performance of MASK on two standard datasets: Flickr30k and MSCOCO. The commonly used evaluation criterions are "R@1", "R@5" and "R@10", i.e., recall rates at the top-1, 5 and 10 results. Following existing works Ge et al. (2024), we use an additional criterion of "Rs" by summing all the recall rates to evaluate the overall performance. Experimental details are provided in the Appendix I.2. Compared to Flickr30k, MSCOCO exhibits greater sample diversity, with images typically containing multiple target objects and semantic regions, resulting in more complex visual structures. The knowledge-based matching constructs explicit multimodal-aligned knowledge as a bridge between regions and words, facilitating more accurate modeling of local visual-semantic relationships in complex visual scenes. The region prototypes and max-mean pooling have a significant impact on knowledge-based unpaired matching methods. However, MASK consistently outperforms existing methods. This is primarily because the MASK exhibits a strong intra-class cohesion among region representations, i.e., the variance between any region representation and the prototypical region representation is relatively small. Consequently, replacing the prototypical region representation with a randomly selected region representation has a minimal impact on overall performance. Furthermore, incorporating semantic relationships between word embeddings reduces the coupling among region representations across words. Therefore, substituting max-mean pooling with global mean has a minor effect on overall performance.
this section cite: ['b5']

Section: ZERO-SHOT IMAGE-TEXT MATCHING
To evaluate the effectiveness of the MASK in complementing with pre-trained models for zero-shot image-text matching, we compare several state-of-the-art re-ranking strategies, including M ACK  Wei et al. (2025). The original and re-ranked performance are compared in Table 2. We can see that although the original accuracies are already high, further improvements can be achieved by applying re-ranking strategies to these pre-trained models. Among them, MASK yields more substantial performance gains. This can be attributed to the high cohesion and low coupling of the region representations, which ensure that each region representation remains closest to its corresponding prototypical region representation while maintaining substantial spatial separation from those of non-corresponding terms. Consequently, MASK reduces the risk of region representations being misclassified in zero-shot image-text matching. These evidences demonstrate that the MASK can be well combined with existing models to further improve their performance.
this section cite: ['b33']

Section: KNOWLEDGE VISUALIZATION
To qualitatively illustrate major differences between MACK and MASK, we visualize two lowdimensional word distributions in Figure 3. The words in the four numbered groups (marked by dashed lines in different colors) are about animals, transports, faces and humans, respectively. In the left distribution corresponding to MACK, there are still some related words that have remote distances. In contrast, the right distribution generated by MASK is semantically more compact. The underlying mechanism is that semantic relationships between word embeddings are incorporated during model training, ensuring that the corresponding prototypical region representations also exhibit semantic associations. These evidences indicate that the MASK can make their prototypical region representations more discriminative.
this section cite: []

Section: OOD WORDS ANALYSIS AND LOSS ABLATION ANALYSIS
To evaluate the impact of OOD words on image-text matching accuracy, we conduct a comparative experiment in Table 3. We observe that image-text matching accuracy significantly improves in both image retrieval and image annotation tasks when OOD words are incorporated, and this improvement is consistently validated across different datasets. Therefore, leveraging the semantic relationships between OOD words and known words to construct corresponding prototypical region representations for OOD words is an effective approach. This phenomenon can be attributed to the relation-preserving equivariant mapping, as shown in Eq (8). As a result, region representations inherit the semantic structure encoded in the word embeddings, allowing the relationships between regions to reflect semantic distances and similarities, thereby enhancing the generalization ability of the matching process.  To testify the contribution of each component of loss L to overall performance, we compare the performance of various losses in Table 3. It can be observed that the performance of the overall loss function L deteriorates when certain components are omitted. Among them, the prototype consistency contrastive learning loss L cl makes the largest contribution to the performance. It is reasonable since the loss L cl constrains all region representations related the same word to be close to each other during model training. Additionally, the semantic relationships between word embeddings in loss L cm serve only as a reference for determining the degree of separation between region representations. Therefore, the loss L cl can make the prototypical region representations more discriminative, which obviously affects the accuracy of matching.
this section cite: []

Section: HYPERPARAMETER ANALYSIS
The proposed MASK involves two trade-off parameters λ 1 and λ 2 in Eq.( 14). To evaluate the impact of different hyperparameters on matching accuracy, we design three controlled experiments in Table 4. The results indicate that MASK achieves the best performance in unpaired image-text matching at λ 1 = λ 2 . Compared to λ1 λ2 = 3.0 (i.e., λ 1 > λ 2 ), the λ1 λ2 = 0.3 achieves better performance, with improvements of approximately 3.5% and 10.3% in Rs on Flickr30k and MSCOCO, respectively. These findings suggest that the cross-modal alignment loss and prototype consistency contrastive loss are complementary, each contributing distinct yet essential benefits to the overall performance. By jointly optimizing these two losses in a balanced manner, we mitigate the risk of overfitting to a single loss term, thereby improving the model's generalization capability. The sampling size m for selecting paired multimodal representations critically affects the semantic quality of the constructed prototypes. To evaluate the impact of different sampling sizes on matching accuracy, we perform the experiment of unpaired image-text matching by MASK using different sampling sizes m on the Flickr30k and MSCOCO datasets in Table 5. Experimental results show that the matching accuracy follows a rise-then-fall trend as the sampling size m increases, achieving its optimum around m = 50. This phenomenon can be explained as follows. When the sampling size is too small, the constructed visual prototype relies excessively on only a few nearest neighbors.
Although this preserves strong local semantic characteristics, it also makes the prototype highly sensitive to noise and outliers in the word embedding space. As the sampling size increases to a moderate level, more semantically relevant neighbors contribute their visual information, thereby enhancing robustness and discriminability. However, when the sampling size becomes too large, semantically weak or marginal neighbors begin to dominate. Their less relevant visual cues dilute the contributions of the core semantic neighbors, ultimately reducing matching accuracy.  15). To investigate the impact of α on final performance, we conduct zero-shot image-text matching experiments using MASK (CLIP) in Table 6. The results show that the best performance on both datasets is obtained at α = 0.15. When α is relatively small, the re-ranking method yields limited improvements, as matching performance is largely dominated by the pre-trained CLIP model. In contrast, as α increases, the influence of the MASK on matching performance becomes more pronounced. Since the knowledge is not learned from domain-specific paired image-text data, potential distributional discrepancies may arise, which can adversely affect the overall performance. Therefore, an appropriate value of α must be carefully chosen to achieve an optimal balance between semantic alignment capability and generalization performance.
this section cite: []

Section: A TABLE OF NOTATION
We list the notation used in this paper in Table 7, for the convenience of reference.
Table 7: Notation used in the paper.
this section cite: []

Section: Symbol Description w k
Word embedding directly from pre-trained word vectors v k
Learnable prototypical region representation (prototype) of k-th semantic concept r j
Raw region representation from pre-trained object detection model Faster-RCNN µ j
The mean of the latent space corresponds to the region representation σ j
The variance of the latent space z Random vector sampled from Gaussian distribution R {r n } n=1,...,B (R ∈ R B×M ) µ +
this section cite: []

Section: Region representation associated with the prototypical region representations
v k V {w k } k=1,...,B (V ∈ R B×N ) d s Distance metrics, e.g., cosine similarity U {v j } j=1,...,J (U ∈ R J×Z ) ρ(•) Max-mean pooling w out
Word embedding w out corresponding to OOD word s q
this section cite: []

Section: Similarity score v out
Prototypical region representation v out corresponding to word embedding w out sk Top-k similarity vector from existing multimodal model s k
Top-k similarity vector from MASK
this section cite: []

Section: B PROOF: EQ. 8 FACILITATES THE CONSTRUCTION OF CORRESPONDING PROTOTYPES FOR OOD WORDS.
Given a paired multimodal knowledge {(w k , v k )} k=1,...,K , where w k ∈ R N and v k ∈ R M are the word embedding and prototypical region representation of the k-th semantic concept, respectively, and K is the total number of semantic concepts. We sample from the multimodal knowledge {(w k , v k )} k=1,...,K to obtain a subset {(w k , v k )} k=1,..., K . Our objective is to utilize the known word embedding w out and {(w k , v k )} k=1,..., K to construct v out such that the following equation is satisfied:
d s (f (v k ), f (v out )) ∝ d s (v k , v out ), k ∈ {1, . . . , K},(16)
and f (v out ) ≈ w out .
We first rewrite Eq. 16 into an equidistant form. Then, there exists a β > 0 such that:
d s (f (v k ), f (v out )) = β d s (v k , v out ).(18)
Define the scaled mapping F := 1 β f . For any k, out:
d s (F (v k ), F (v out )) = d s (v k , v out ). (19
)
That is, F precisely preserves the euclidean distance among these prototypes. An isometric mapping in euclidean space exhibits a well-defined structural property, which is a fundamental result in classical geometry and functional analysis. Specifically, any distance-preserving mapping defined on the euclidean space must be a rigid transformation Iovino (2021). In other words, there exists an orthogonal matrix A that represents a rotation or reflection, and a translation vector t, such that the mapping can be expressed as:
F (v k ) = Av k + t.(20)
Therefore, the f can be expressed as:
f (v k ) = βAv k + βt. (21
)
Since A is an orthogonal matrix, its inverse A -1 must exist and is equal to its transpose A . Therefore, we can derive:
v k = A (F (v k ) -t) = A ( 1 β f (v k ) -t).(22)
For any given w k in the knowledge, according to Eq. 17, we have:
v k = A ( 1 β w k -t).(23)
Therefore, we only need to determine the values of A, t, and β to obtain the visual prototype vector v out corresponding to the OOD words w out . Extending Eq. 20 to the general case where A is no longer a strictly orthogonal matrix, we obtain the following equation using the column orthogonality condition:
||Av k -Av out || 2 = ||v k -v out || 2 ⇔ A A = I min{M,N } ,(24)
Here, there exists an A such that Eq. 23 holds. We do not need to know the values of A, t, and β a priori, as they can be uniquely determined from the paired knowledge {(w k , v k )} k=1,...,K using standard similarity Procrustes decomposition. First, we construct matrix W 1 = [v 1 , . . . , v K ] and matrix W 2 = [w 1 , . . . , w K ]. Next, we compute the centroids of each set:
v = 1 K K k=1 v k ∈ R M , w = 1 K K k=1 w k ∈ R N .(25)
We centralize the knowledge using the computed centroids v and w:
W1 = [v 1 -v, . . . , v K -v] ∈ R M ×K , W2 = [w 1 -w, . . . , w K -w] ∈ R N ×K .(26)
From here, we can separate out t, and subsequently only need to consider the linear transformation and scaling. Next, we construct the covariance matrix C:
C = W1 W2 ∈ R M ×N .(27)
Let M > N . And then, we utilize SVD decomposition to obtain the values of A, t, and β:
C = U ΣV , U ∈ R M ×M , Σ ∈ R M ×N , V ∈ R N ×N ,(28)
A = U I N 0 (M -N )×N V , A ∈ R M ×N ,(29)
β = trace(A W2 W1 ) trace( W2 W1 ) ,(30)
t = w -βAv, t ∈ R M . (31) Finally, we substitute the values of A, t, and β into Eq. 23 to obtain the prototype vector v out corresponding to the word embedding w out .
this section cite: ['b12']

Section: C PROOF: THE COSINE SIMILARITY IN EQ. 9 IS A REASONABLE DISTANCE METRIC d s .
Euclidean space possesses strict linear structure preservation properties. It is the only metric satisfying translation and rotation invariance, and is naturally compatible with linear mappings. This allows the structural alignment problem between visual and linguistic spaces to be transformed into a standard orthogonal Procrustes problem. Therefore, we can equate Eq. 19 with Eq. 20 in euclidean space.
Cosine similarity and euclidean distance are not inherently equivalent. However, after vector normalization, there exists a strict monotonic mapping between cosine similarity and euclidean distance. This implies that, cosine similarity can be regarded as a form of "Euclidean-like distance," thereby satisfying the prerequisites for similarity transformations. Let two vectors w k and w k , their cosine similarity is defined as:
cos(w k , w k ) = w k • w k ||w k || ||w k || .(32)
where || • || denotes the L1-norm. The euclidean distance is defined as:
d E (w k , w k ) = ||w k -w k ||.(33)
We normalize each vector such that ||w k ||=||w k ||=1, yielding:
d E (w k , w k ) 2 = ||w k -w k || 2 = ||w k || 2 + ||w k || 2 -2 • w k w k = 2 -2cos(w k , w k ).(34)
cos(w k , w k ) = 1 - 1 2 d E (w k , w k ) 2 .(35)
Combining Eq. 34 and Eq. 35, it can be concluded that after normalization, a one-to-one monotonic functional relationship exists between cosine similarity and euclidean distance.
L cm = E [(1-cos( w i w i 2 , w i w i 2 ))] word-alignment +E [((cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 ))) 2 ] structure-preserving . (36
)
The loss L cm comprises two components: the first item enforces alignment between predicted and pre-trained word embeddings, while the second ensures that region representations capture the structural relationships among words.
∂(1 -cos( wi wi 2 , w i w i 2 )) ∂ w i w i 2 = (w i w i )w i w i 2 2 w i 2 - w i w i 2 .(37)
∂(1 -cos( wi wi 2 , w i w i 2 )) ∂ w i w i 2 = 0 =⇒ w i = w i .(38)
∂([(cos(
w i w i 2 , w j w j 2 ) -cos( µi µi 2 , µj µj 2 )) 2 ]) ∂ w i w i 2 = 2(cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 )) ∂(cos( w i w i 2 , w j w j 2 )) ∂ w i w i 2 .(39)
∂(cos(
w i w i 2 , w j w j 2 )) ∂ w i w i 2 = w j w j 2 - (w i • w j )w i w i 2 2 w j 2 .(40)
∂([(cos(
w i w i 2 , w j w j 2 ) -cos( µi µi 2 , µj µj 2 )) 2 ]) ∂ w i w i 2 = 0 =⇒ cos( w i w i 2 , w j w j 2 ) = cos( µ i µ i 2 , µ j µ j 2 ).(41)
this section cite: []

Section: D PROOF: THE OBJECTIVE EXISTENCE OF THE ISOMETRIC HYPOTHESIS.
Proof: For any finite set of region representations {µ i } n i=1 , there necessarily exists a mapping f in an appropriate Euclidean space such that:
d s (f (µ i ; Θ f ), f (µ j ; Θ f )) ∝ d s (µ i , µ j ).(42)
We first convert the point-wise cosine error into the Euclidean difference between vectors. According to Appendix C, we have:
ŵ i -ŵi 2 2 = 2(1 -cos( ŵ i , ŵi )).(57)
For each case where i ∈ B, let δ i := 1 -cos( ŵ i , ŵi ) ≥ 0. Then:
ŵ i -ŵi 2 = 2δ i , (58
) i∈B δ i = L word ≤ w ,(59)
δ i ≤ w ⇒ ŵ i -ŵi 2 ≤ √ 2 w .(60)
To quantify how point-wise alignment errors influence the pairwise cosine discrepancies, we characterize the variation of individual entries in the cosine similarity matrix using the norm difference. Then:
cos( ŵ i , ŵ j ) -cos( ŵi , ŵj ) = ŵ i • ŵ j -ŵi • ŵj ,(61)
ŵ i • ŵ j -ŵi • ŵj = ( ŵ i -ŵi ) • ŵ j + ŵi • ( ŵ j -ŵj ) ≤ ŵ i -ŵi 2 • ŵ j 2 + ŵi 2 • ŵ j -ŵj 2 ≤ ŵ i -ŵi 2 + ŵ j -ŵj 2(62)
ŵ i • ŵ j -ŵi • ŵj ≤ √ 2 w + √ 2 w = 2 √ 2 w .(63)
To quantify the influence of the structural loss L struct on the reconstruction error, we convert the structural loss into an upper bound on the per-pair error. Let the per-pair error be defined as e ij := cos( ŵ i , ŵ j ) -cos(µ i , µ j ). Then we obtain:
i =j e 2 ij = L struct ≤ s ,(64)
max i =j |e ij | ≤ i =j e 2 ij ≤ √ s .(65)
If we distribute s uniformly across all pairs, then the absolute error of each pair is bounded above by s /M . Then:
cos( ŵ i , ŵ j ) -cos(µ i , µ j ) = cos( ŵ i , ŵ j ) -cos( ŵi , ŵj ) + cos( ŵi , ŵj ) -cos(µ i , µ j ) ≤ cos( ŵ i , ŵ j ) -cos( ŵi , ŵj ) + |cos( ŵi , ŵj ) -cos(µ i , µ j )| ≤ 2 √ 2 w + cos( ŵi , ŵj ) -cos( ŵ i , ŵ j ) + cos( ŵ i , ŵ j ) -cos(µ i , µ j ) ≤ 4 √ 2 w + s M (66
)
Therefore, by enforcing w → 0 and s → 0 during training, we obtain the following expression: The w in Eq. ( 9) cannot be replaced by w. Doing so would remove the "structure-preservation" supervision imposed on the M T M model f , as the second term in Eq. ( 9) would no longer contribute a meaningful loss for training f . The loss function recommended in the article for preserving structure (the second term of Eq.( 9)) is defined as L A struct :
cos( ŵ i , ŵ j ) -cos(µ i , µ j ) → 0. (67
)
d s (f (µ i ; Θ f ), f (µ j ; Θ f )) ∝ d s (µ i , µ j ).(68
L A struct = E[(cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 )) 2 ].(69)
The loss function obtained after replacing w with w in L A struct is defined as L B struct :
L B struct = E[(cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 )) 2 ].(70)
Next, we compare the constraint capabilities of L A struct and L B struct on the M T M model f (with parameters Θ f ).
this section cite: []

Section: F.1 FROM THE PERSPECTIVE OF THE GRADIENT OF THE LOSS FUNCTION
For the loss function L A struct , we apply the chain rule to Θ f as follows:
∂L A struct ∂Θ f = 2E(cos( w i w i 2 , w j w j 2 ) -cos( µ i µ i 2 , µ j µ j 2 )) • ∂ cos( w i w i 2 , w j w j 2 ) ∂Θ f .(71)
∂ cos(
w i w i 2 , w j w j 2 ) ∂Θ f = ∂ cos( w i w i 2 , w j w j 2 ) ∂w i • ∂w i ∂Θ + ∂ cos( w i w i 2 , w j w j 2 ) ∂w j • ∂w j ∂Θ . (72
)
As long as the parameters of f influence w i (i.e., ∂w i ∂Θ = 0), the above equation generally does not vanish. Consequently, L A struct produces a nonzero gradient that drives the update of Θ f , forcing the mapping f to bring cos(
w i w i 2 , w j w j 2 ) closer to cos( µi µi 2 , µj µj 2
). Therefore, L A struct directly imposes constraints on the M T M model f . Moreover, the cos( wi
wi 2 , wj wj 2 ) in L B
struct is a constant. We apply the chain rule to Θ f as follows:
L B struct = E[(cos( w i w i 2 , w j w j 2 ) constant -cos( µ i µ i 2 , µ j µ j 2 )) 2 ].(73)
∂ L B struct ∂Θ f = 0.(74)
L B struct provides no constraints or gradient information on f , and therefore cannot compel the mapping f to preserve input similarity in any manner. Although µ is affected, this influence does not propagate to Θ f , making it impossible to fulfill the objective of structural preservation.
this section cite: []

Section: F.2 FROM THE PERSPECTIVE OF THE JOINT OPTIMIZATION OBJECTIVE
The loss function recommended in the article for word alignment (the first term of Eq.( 9)) is defined as L word :
L word = E[(1 -cos( w i w i 2 , w i w i 2 ))].(75)
Next, we compared the differences between the two loss functions when jointly optimized with L word . The joint loss function for L word and L A struct is expressed as follows:
L A cm = L word + L A struct .(76)
Both loss terms involve Θ f (through w i ) and therefore jointly constrain the mapping. Specifically, L word pulls each single-point mapping w i toward its corresponding w i , while L A struct enforces that f preserve the pairwise similarity structure among samples, consistent with the semantic relationships in the µ-space (prototype). The joint effect of these two losses ensures that the learned mapping f achieves both accurate pointwise alignment and global structural preservation.
The joint loss function for L word and L B struct is expressed as follows:
L B cm = L word + L B struct . (77
)
It is worth noting that L B struct does not impose any constraints on the model f . Consequently, L word is the only term that directly supervises f , and minimizing it merely pulls w i toward its corresponding w i . In contrast, L B struct adjusts the µ-space (governed by the parameters of the P AE model) to make cos( µi
µi 2 , µj µj 2 ) approach cos( wi wi 2 , wj wj 2
), but it provides no mechanism to enforce structural preservation in the mapping produced by f .
this section cite: []

Section: G PROOF: THE PROTOTYPE CONSISTENCY CONTRASTIVE LEARNING LOSS L cl ENHANCES THE DISCRIMINABILITY OF REGION REPRESENTATIONS.
The prototype consistency contrastive learning loss enhances the discriminability of region representations by reducing intra-word variance and increasing inter-word separation. Intra-word variance σ 2 k and inter-word separation D k,k can be defined as follows:
σ 2 k = 1 J k J k j=1 ||µ j -v k || 2 , (78
)
D k,k := ||v k -v k || 2 . (79
)
For any µ j :
||µ j -v k || 2 = ||µ j -μ + μ -v k || 2 = ||µ j -μ|| 2 + ||μ -v k || 2 + 2 µ j -μ, μ -v k ,(80)
σ 2 k ≈ 1 J k J k j=1 ||µ j -μ|| 2 + ||μ -v k || 2 , (81
)
where μ serves as the temporary mean for a given batch. The intra-word variance can be decomposed into the within-batch intra-word variance and the deviation between the batch-specific word centers and the prototypes. The loss L cl is updated along the gradient direction:
µ j ←-µ j -η(µ j -v k ).(82)
µ j (t + 1) = (1 -η) µ j (t) + ηv k .(83)
σ 2 k (t + 1) = (1 -η) 2 σ 2 k (t).(84)
As iterations progress, σ 2 k gradually approaches zero, causing samples within the same word to cluster tightly around their prototypes and thereby enhancing the discriminability of the region representations. The loss L cl is updated by moving along the gradient direction while simultaneously being repelled away from other prototypes:
µ j ←-µ j + η k =k (µ j -v k ). (85
) v k = 1 J k J k j=1 µ j (t + 1) = v k + η k =k (v k -v k ). (86
)
D k,k (t + 1) = ||v k -v k || 2 = ||v k -v k + η( k = k(v k -v k) - k = k(v k -v k)|| 2(87)
At convergence, each prototype is positioned as far as possible from all others, resulting in distinct separation between different prototypes and further enhancing the discriminability of the region representations.
this section cite: []

Section: H FINE-TUNED DOMAIN KNOWLEDGE
To better adapt the multimodal aligned semantic knowledge to specific datasets, we can further finetune the prototypical region representations to obtain domain-specific knowledge. This fine-tuning step is optional and depends on the availability of unpaired data in the target dataset. Notably, the multimodal aligned semantic knowledge alone can also be applied directly and achieves strong performance. Given that annotating paired image-text data is costly, while unpaired images and texts are typically more accessible, we adopt a bidirectional region-word cycle-consistent learning approach in an unpaired learning setting.
In particular, given a batch of unpaired images and texts, we first obtain a set of raw region representations R = {r n } n=1,...,I (R ∈ R I×M ) using the pre-trained Faster-RCNN above and a set of parsed words through tokenization operation implemented via NLTK. Then, we use the knowledge as a cross-modal bridge to represent all the words into the corresponding prototypical region representations u = {v j } j=1,...,J (u ∈ R J×Z ). For the set of regions R, we extract region representations µ ∈ R I×Z by utilizing the P AE model h.
We utilize a bidirectional region-word cycle-consistent loss to learn a parametric transformation matrix W ∈ R Z×Zfoot_3 . This loss incorporates two cross-modal similarity measurement processes: region-to-word (R2W) and word-to-region (W2R). In the R2W process, similarities between each word and all regions are first computed, and these similarities are then used as weights to aggregate all regions into a reconstructed word representation. Conversely, in the W2R process, similarities between each region and all words are computed to reconstruct regions from the word representations.
The corresponding formulations are
S = uW (µW ) ,(88)
û = sof tmax(S) µW, μ = sof tmax(S ) uW,(89)
where S ∈ R J×I is the similarity matrix between transformed word and region representations, û ∈ R J×Z contains the reconstructed word representations from region representations, and μ ∈ R I×Z contains the reconstructed region representations from word representations.
Each original word (or region) representation is then compared with its reconstructed counterpart to determine whether they correspond to the same entity. By minimizing the cross-entropy between the predicted and ground-truth labels, we obtain a self-supervised loss, L ss , which is used to optimize W :
Ŷ R2W = sof tmax(uW û ) , Ŷ W 2R = sof tmax(µW μ ) ,(90)
L R2W = - J j=1 y j log(ŷ j ),(91)
L W 2R = - I n=1 y n log(ŷ n ),(92)
L ss = L R2W + L W 2R ,(93)
where Ŷ R2W ∈ R J×J and Ŷ W 2R ∈ R I×I are two matrices including the predicted labels in R2W and W2R directions, respectively. In Ŷ R2W and Ŷ W 2R , the j-th and n-th columns are denoted as ŷj and ŷn , respectively. y j and y n are two groundtruth label vectors, whose the j-th and n-th values are ones and the rest are zeros. After the cycle consistent learning, we can use the learnable W to transform all prototypical region representations into the fine-tuned domain knowledge, denoted as {(w k , vk )} k=1,...,K , where vk = v k W ∈ R Z .
this section cite: []

Section: I EXPERIMENTAL SETTING I.1 DATASET
The details of experimental datasets and metrics are described as follows.
Flickr30k Young et al. (2014) consists of 31,783 images collected from the Flickr website. Each image has 5 human annotated texts. We use the public training, validation and testing splits, which contain 29,783, 1,000 and 1,000 images, respectively. Lin et al. (2014) consists of 123,287 images, each of which is associated with 5 texts. We use the public training, validation and testing splits, with 113,287, 5,000 and 5,000 images, respectively.
this section cite: ['b36', 'b20']

Section: MSCOCO

this section cite: []

Section: I.2 IMPLEMENTATION DETAILS
In the multimodal aligned semantic knowledge, we collect all words from the VG dataset and filter out some special characters and rare words, resulting in a total of K=12,385 semantic concepts. For each image, we initially employ the pre-trained object detection model Faster-RCNNfoot_4 to extract raw region representations, setting the number of detected regions to I=36 and the dimensionality of each region representation to M =2048. For each word, we obtain its word embedding using the pre-trained word vectors glove-840B-300dfoot_5 . The batch size is 4096 for the first 200 epochs and 2048 for the next 200 epochs. The trade-off factors λ 1 and λ 2 are set to 3. The sampling size m is set to 10. We use the Adam to optimize the loss with a learning rate of 1e-4.
this section cite: []

Section: I.3 PRETRAINED MODELS AND WORD VECTORS
Faster-RCNN is a widely used deep learning model for object detection, tasked with both identifying and localizing objects within an image. Building on earlier approaches such as R-CNN and Fast R-CNN, it introduces a Region Proposal Network (RPN) that generates object proposals directly within the model. This integration significantly improves both speed and accuracy, enabling efficient, real-time detection of multiple objects with high precision. We use Detectron2 as the backend to support comprehensive functions, including training, testing, and feature extraction. Additionally, we migrate the pre-trained Caffe-based model from the original repository, ensuring that it extracts visual features consistent with the original model, with deviations of less than 0.01.
GloVe is an unsupervised learning algorithm designed to generate vector representations of words. It is trained on aggregated global word-word co-occurrence statistics from a corpus, producing embeddings that capture meaningful semantic relationships and exhibit interpretable linear substructures within the word vector space. We obtain word embedding using the pre-trained word vectors glove-840B-300d. It is a set of pre-trained word embeddings derived from the GloVe model developed by Stanford University, trained specifically on Common Crawl data. The model is built using approximately 840B tokens, resulting in a vocabulary of 2.2 million words and producing 300-dimensional word vectors.
CLIPfoot_6 is a cross-modal pre-trained model proposed by OpenAI, designed to learn a shared semantic embedding space for images and text. It is trained on large-scale natural image-text pairs, mapping images and text into the same-dimensional vector space using an image encoder and a text encoder. Contrastive learning is employed to pull corresponding image-text pairs closer in the embedding space while pushing non-corresponding pairs farther apart.
ALBEFfoot_7 is a vision-language pre-trained model proposed by Salesforce, designed to enhance cross-modal semantic representations through an "align before fuse" strategy. Its core idea is to first align image and text features in a shared space via image-text contrastive learning, and then fuse them using a multimodal encoder to capture richer cross-modal interactions. Additionally, AL-BEF incorporates a momentum distillation mechanism, where a continuously updated momentum model generates pseudo-labels to improve training robustness. The model demonstrates strong performance on tasks such as image-text retrieval, visual question answering, and natural language visual reasoning, making it a key approach in the vision-language pretraining field.
this section cite: []

Section: I.4 MODEL DETAILS
In this section, we present the architectures of the models involved in the proposed MASK framework, as shown in Table 8. The [B, M, 2048] indicates that a batch contains B images, each image is divided into M regions, and each region has a feature dimension of 2048. In our experiments, we set B = 128 and M = 36. It is worth noting that the model architectures are not fixed. In later sections, we will discuss how model size impacts the accuracy of image-text matching. Our proposed MASK can also enhance the generalization of conventional image-text matching models when applied to unseen datasets. Specifically, we try to re-rank conventional image-text matching models for the task of cross-dataset image-text matching. The experimental setup is as follows: (1) two representative image-text matching models (i.e., VSRN Radford et al. (2021) and SAEM Wu et al. (2019)) are trained on a source dataset (e.g., Flickr30k or MSCOCO), (2) these models are then evaluated on a different target dataset (e.g., MSCOCO or Flickr30k), and (3) using the proposed MASK to re-rank these models on the target dataset. It is important to note that neither the re-ranking methods nor the base models are trained on the target dataset. Published as a conference paper at ICLR 2026
In Table 9, the results of two kinds of cross-dataset image-text matching are both presented, which are explained as follows. MSCOCO → Flickr30k: training existing models on the MSCOCO dataset and testing them on the Flickr30k dataset. Flickr30k → MSCOCO: training existing models on the Flickr30k dataset and testing them on the MSCOCO dataset. We observe that applying MASK to re-rank the outputs of VSRN and SAEM consistently enhances their generalization performance on unseen datasets, with substantial relative improvements observed across both sub-tasks. For example, V SRN + M ASK performs much better than V SRN by 5.0% and 4.8% in R@1 on the MSCOCO → Flickr30k task, and by 3.6% and 6.3% in R@1 on the Flickr30k → MSCOCO task. Compared with V SRN +F R, the relative improvements are also large, i.e., 7.2% ∼ 8.7% and 7.8% ∼ 8.1% in Rs when re-ranking VSRN and SAEM, respectively. Comparing the results in Table 9 with those in Table 2, we observe that the relative performance gains are more substantial for VSRN and SAEM, primarily because CLIP and ALBEF exhibit much higher baseline performance. In other words, re-ranking yields larger improvements when applied to less accurate models.
this section cite: []

Section: J.2 VISUALIZATION OF POSITIVE AND NEGATIVE EXAMPLES FOR IMAGE RETRIEVAL AND IMAGE ANNOTATION
To better understand the OOD words, we show some representative examples of retrieved images or texts based on text or image queries by CLIP in Table 10. These examples are selected according to the following criteria: 1) Positive examples are those in which the ground-truth matched image or text is not ranked at top-1 by CLIP but is successfully promoted to a higher rank by MASK through re-ranking, and 2) Negative examples refer to situations where the ground-truth image or text is initially ranked at top-1 by CLIP but is pushed to a lower position after re-ranking by MASK. In the positive examples, it seems that the CLIP cannot well understand the semantic concepts such as "glasses", "pierced", and "broken". For instance, in the top-1 retrieved image for the first text query, there are no clear clues indicating "pierced" or "glasses", yet its rank is still higher than that of the ground-truth one. Similarly, in the top-1 retrieved text for the second image query, the annotation contains the word "glasses", even though the image itself does not include such information. While our MASK especially focuses on understanding these semantic concepts and can thus increase the corresponding similarity scores between the matched regions and words. However, MASK may also make incorrect decisions. For example, in the images retrieved for the fourth text query, MASK reduces the rank of the ground-truth image. This behavior can be attributed to the presence of adverbs (e.g., "very", "quite"), adjectives (e.g., "large", "excited"), and pronouns (e.g., "they", "this") in the text. Our further analysis indicates that the OOD words negatively affecting MASK are typically those that cannot correspond to specific visual regions. Attempting to construct region prototypes for such words introduces substantial semantic noise. Finally, it is important to note that negative examples contain not only non-visual adjectives, adverbs, and pronouns, but also some informative OOD words with tense or plural variations. Consequently, the final image-text matching accuracy is influenced by the combined effects of all these words.
J.3 DIFFERENT DETECTOR COMPARISON This work derives region representations using the Bottom-Up and Top-Down (BUTD) model, i.e., Faster-RCNN, which consists of a Region Proposal Network (RPN) Ren et al. (2015) and a 101layer Residual Network (ResNet101) He et al. (2016) pretrained on the VG dataset. Since our constructed multimodal knowledge is also based on the VG dataset, this pretraining step is crucial for learning discriminative region representations and achieving strong performance. To validate this, we experiment with three alternative detectors within the proposed MASK framework. The first is DETR Carion et al. (2020), a recently popular Transformer-based detector. The second is DINO Zhang et al. (2023), evaluated in two versions: the original model and a variant pretrained on the VG dataset in the same manner as BUTD. The third is an enhanced version of BUTD, referred to as BUTD+, which employs ResNet152, ConvNeXt Liu et al. (2022), and Swin Transformer Liu et al. (2021) as the backbone networks to replace ResNet101.
We evaluate these detectors on the task of unpaired image-text matching and compare their performance on the Flickr30k dataset in Table 11. The results show that directly using either DETR or DINO leads to poor performance. This is primarily because they fail to extract region representations as accurately as BUTD when constructing multimodal knowledge. Specifically, BUTD uses Faster R-CNN as its backbone, a two-stage object detector that allows ground-truth bounding boxes The girl with the red belt is kicking a pad that the person in black is holding.
A martial artist wearing a white Gi and a black belt is pinning another man in a blue Gi to the ground.
A girl wearing glasses is in a blue harness while rock climbing.
The person has a striped shirt on and is holding on to a rope on a mountain. (GT)
A mountaineer about to descend down a mountain with a blue helmet on.
this section cite: []

Section: Pos
The person has a striped shirt on and is holding on to a rope on a mountain. (GT)
A girl wearing glasses is in a blue harness while rock climbing.
A mountaineer about to descend down a mountain with a blue helmet on.
A young boy is quite excited in the throes of a ballgame. (GT)
A child wearing a blue shirt is jumping in the air.
A boy wearing jeans leaps in the air and shows shadow below.
this section cite: []

Section: Neg
A boy wearing jeans leaps in the air and shows shadow below.
A child wearing a blue shirt is jumping in the air.
A young boy is quite excited in the throes of a ballgame. (GT)
This man bravely cuts down trees on the job.
this section cite: []

Section: (GT)
A young adult is doing a back flip on a trampoline near a lake.
A man in blue overalls and red shirt holding a chainsaw.
this section cite: []

Section: Neg
A young adult is doing a back flip on a trampoline near a lake.
This man bravely cuts down trees on the job.
this section cite: []

Section: (GT)
A man in blue overalls and red shirt holding a chainsaw. to be directly provided, yielding highly accurate feature representations. In contrast, DETR and DINO require generating hundreds of candidate bounding boxes and then selecting the one with the highest Intersection over Union (IoU) for each ground-truth box. This additional step inevitably introduces noise, thereby degrading the quality of the constructed knowledge and resulting in lower performance. In addition to inaccuracies in bounding box generation, the superior performance of BUTD may also be attributed to its more powerful feature extraction network. To investigate this, we replace the ResNet101 backbone with ResNet152, ConvNeXt, and Swin Transformer, resulting in an enhanced version called BUTD+. This modification leads to a substantial performance improvement, demonstrating the benefit of stronger feature extraction.
this section cite: []

Section: J.4 MODEL ARCHITECTURE ANALYSIS
This work acquires multimodal aligned knowledge based on three models: P AE, F RM , and M T M . Since the architecture of these models can significantly influence image-text matching accuracy, we use different architectures to perform the experiment of unpaired image-text matching and compare their performance on the Flickr30k dataset in Table 12.
We can see that increasing the number of fully connected (FC) layers leads to a decline in image-text matching performance, whereas adding more self-attention layers results in performance gains. For example, in the PAE model, increasing the number of fully connected layers from 1 to 3 results in a performance drop of approximately 19.4% in Rs on the Flickr30k dataset. When the number of layers is further increased from 3 to 5, accuracy decreases by an additional 11.3%. These findings demonstrate that FC layers have a substantial impact on model performance, and excessive depth can severely degrade accuracy. This can be explained as that FC layers apply fixed nonlinear transformations, where even minor training errors accumulate as the network deepens, gradually distorting the spatial distribution of features. In contrast, self-attention layers explicitly model similarity relationships among entities, inherently preserving or even reinforcing the semantic structure of the representations. The semantic geometry of word embeddings directly influences the construction of OOD prototypes. To evaluate the impact of different pretrained word vectors on matching accuracy, we perform the experiment of unpaired image-text matching by MASK using different word vectors on the Flickr30k and MSCOCO datasets in Table 13. Experimental results show that GloVe consistently outperforms Word2Vec and FastText across both datasets, demonstrating its superior suitability for constructing OOD prototypes. Specifically, GloVe achieves 122.8% R@s on Flickr30k and 209.5% R@s on MSCOCO, surpassing Word2Vec by 2.7 ∼ 5.2% and FastText by 8.4 ∼ 14.4%. These performance differences can be attributed to the distinct characteristics of the word vectors. GloVe encodes global word-word co-occurrence statistics, enabling it to capture broader contextual relatedness. Such global semantic structure is crucial in cross-modal matching, where visual regions and textual words need to align through high-level associative semantics rather than strict synonymy. In contrast, Word2Vec, which learns from local context windows, excels at modeling fine-grained synonymy but is less capable of capturing the broader semantic relations required for cross-modal
this section cite: []

Section: L THE USE OF LARGE LANGUAGE MODELS
We use LLMs solely to assist in checking grammatical correctness. After the initial check by the model, we further refine and correct any remaining grammatical issues manually. Therefore, the role of LLMs in this work is limited.
this section cite: []

Section: 
Published as a conference paper at ICLR 2026 Given n non-zero vectors µ 1 , . . . , µ n ∈ R Z , normalize them to obtain: μi := µ i µ i 2 , i = 1, . . . , n.
We aim to construct a set of vectors {y i } n i=1 in a Euclidean space as f (µ i ), and demonstrate that cos(y i , y j ) = cos(µ i , µ j ) can be achieved. Next, we organize the pairwise similarities into a matrix, facilitating the use of spectral decomposition to construct vectors that satisfy the given inner product relationships. We define the n × n Gram matrix G as follows:
G ij := cos(μ i , μj ) = μ i μj . (44
)
Here G is a real symmetric positive semi-definite (PSD) matrix, so for any vector z ∈ R n , it satisfies:
z Gz = i z i μi 2 2 ≥ 0,(45)
where the G ii = 1. According to the properties of real symmetric PSD matrix, there exist an orthogonal matrix U ∈ R n×n and a diagonal matrix
Λ = diag(λ 1 , . . . λ n )(λ k ≥ 0) such that: G = UΛU ,(46)
where r = rank(G) ≤ n. Take the non-zero eigenvalue part of Λ as Λ r ∈ R r×r , and take the first r columns of the corresponding eigenvectors to obtain U r ∈ R n×r :
G = U r Λ r U r ,(47)
Y := U r Λ 1/2 r ∈ R n×r ,(48)
where we write Y in row-vector form, and denote its i-th row as y i (y i ∈ R r ):
(Y Y ) ij = y i y j = (U r Λ 1/2 r )(U r Λ 1/2 r ) ij = U r Λ r U r ij = G ij .(49)
Therefore, we have constructed n vectors y 1 , . . . , y n ∈ R r that satisfy the inner-product relation
y i y j = G ij : y i 2 2 = y i y i = G ii = 1,(50)
cos(y i , y j ) = y i y j y i y j = G ij 1 • 1 = cos(μ i , μj ),(51)
cos(y i , y j ) = cos(μ i , μj ) = cos(µ i , µ j ).
(52) By setting f (µ i ) := y i (i = 1, . . . , n), we obtain:
cos(f (µ i ), f (µ j )) = cos(µ i , µ j ) ∀ i, j,(53)
d s (f (µ i ; Θ f ), f (µ j ; Θ f )) ∝ d s (µ i , µ j ).(54)
E PROOF: L cm (EQ. ( 9
this section cite: []

Section: )) ENCOURAGES f TO APPROACH THE ISOMETRY ASSUMPTION (EQ. (8)).
Given the pre-trained word embeddings w i and the predicted word embeddings w i = f (µ i ; Θ f ), we normalize them to obtain:
ŵ i := w i w i , ŵi := w i w i .(55)
Similarly, we normalize the region representations to obtain μi := µi µi . L cm consists of L word and L struct (Appendix F). For a given batch size B, there exist constants w > 0 and s > 0 such that, when L word ≤ w and L struct ≤ s , the following inequality holds:
cos( ŵ i , ŵ j ) -cos(µ i , µ j ) ≤ 4 √ 2 w + s M ,(56)
where M denotes the number of ordered pairs (i = j), and i, j ∈ B. When w → 0 and s → 0, cos( ŵ i , ŵ j ) -cos(µ i , µ j ) admits an upper bound arbitrarily close to 0, meaning that the cosine similarity between the mapped vector pairs approaches that in the original µ-space, thereby achieving
d s (f (µ i ; Θ f ), f (µ j ; Θ f )) ∝ d s (µ i , µ j ).
alignment. FastText places greater emphasis on morphological similarity. However, morphological similarity does not necessarily imply semantic similarity, which introduces significant noise and ultimately reduces matching accuracy.  14. To eliminate the influence of other factors, such as the matching framework and feature extraction, we fix the batch size to 1 for all models during testing. In the table, the testing time is measured by seconds (s), and the model size is measured by the millions of model parameters (M).
From the table, we observe that MACK and MASK have significantly smaller model sizes compared to CLIP and ALBEF. The parameters of MACK come from the BUTD module used for extracting region representations. While the knowledge-based image-text matching is lightweight and requires no additional parameters. Similarly, for MASK, the majority of the testing time is attributed to the inference processes of the BUTD and P AE models, which is considerably faster than the testing time required by CLIP and ALBEF.
this section cite: []

Section: K LIMITION AND FUTURE WORK
It is important to acknowledge certain limitations of the proposed MASK, which will be addressed in future work. First, the raw region representations are extracted using the pre-trained object detection model BUTD. It would be better to pretrain more advanced detectors on the VG dataset to provide more discriminative region presentations. Second, relying solely on nouns for unpaired image-text matching is suboptimal. It would be better to take all the other words into consideration for more accurate image-text matching.
this section cite: []

Section: References
Ref_id:b0 Title: End-to-end object detection with transformers Year: (2020)
Ref_id:b1 Title: Divide and conquer: Answering questions with object factorization and compositional reasoning Year: (2023)
Ref_id:b2 Title: Unsupervised image captioning Year: (2019)
Ref_id:b3 Title: Noise-aware image captioning with progressively exploring mismatched words Year: (2024)
Ref_id:b4 Title: Mixed-curvature multi-modal knowledge graph completion Year: (2025)
Ref_id:b5 Title: 3shnet: Boosting image-sentence retrieval via visual semantic-spatial self-highlighting Year: (2024)
Ref_id:b6 Title: Unpaired image captioning via scene graph alignments Year: (2019)
Ref_id:b7 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b8 Title: Few-shot image and sentence matching via aligned cross-modal memory Year: (2021)
Ref_id:b9 Title: Multimodal aligned conceptual knowledge for unpaired image-text matching Year: (2022)
Ref_id:b10 Title: Unpaired image-text matching via multimodal aligned conceptual knowledge Year: (2024)
Ref_id:b11 Title: Unpaired image-text matching via multimodal aligned conceptual knowledge Year: (2024)
Ref_id:b12 Title: Stable banach spaces and banach space structures, i: Fundamentals Year: (2021)
Ref_id:b13 Title: Visual genome: Connecting language and vision using crowdsourced dense image annotations Year: (2017)
Ref_id:b14 Title: Gang Hua, Houdong Hu, and Xiaodong He. Stacked cross attention for image-text matching Year: (2018)
Ref_id:b15 Title: Cross-modal retrieval for knowledge-based visual question answering Year: (2024)
Ref_id:b16 Title: Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training Year: (2020)
Ref_id:b17 Title: Towards structure-aware model for multi-modal knowledge graph completion Year: (2025)
Ref_id:b18 Title: Improving image-text matching with bidirectional consistency of cross-modal alignment Year: (2024)
Ref_id:b19 Title: Integrating listwise ranking into pairwise-based image-text retrieval Year: (2024)
Ref_id:b20 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b21 Title: Graph structured network for image-text matching Year: (2020)
Ref_id:b22 Title: Swin transformer: Hierarchical vision transformer using shifted windows Year: (2021)
Ref_id:b23 Title: A convnet for the 2020s Year: (2022)
Ref_id:b24 Title: Enhancing visual question answering through questiondriven image captions as prompts Year: (2024)
Ref_id:b25 Title: Fine-grained image-text matching by cross-modal hard aligning network Year: (2023)
Ref_id:b26 Title: Composing object relations and attributes for image-text matching Year: (2024)
Ref_id:b27 Title: Learnable pillar-based re-ranking for image-text retrieval Year: (2023)
Ref_id:b28 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b29 Title: Faster r-cnn: Towards real-time object detection with region proposal networks Year: (2015)
Ref_id:b30 Title: Zero-shot learning through cross-modal transfer Year: (2013)
Ref_id:b31 Title: Regular constrained multimodal fusion for image captioning Year: (2024)
Ref_id:b32 Title: Multimodal llm enhanced cross-lingual cross-modal retrieval Year: (2024)
Ref_id:b33 Title: Dynamic visual semantic sub-embeddings and fast re-ranking for image-text retrieval Year: (2025)
Ref_id:b34 Title: Dual stream relation learning network for image-text retrieval Year: (2024)
Ref_id:b35 Title: Learning fragment self-attention embeddings for image-text matching Year: (2019)
Ref_id:b36 Title: From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions Year: (2014)
Ref_id:b37 Title: DINO: DETR with improved denoising anchor boxes for end-to-end object detection Year: (2023)
Ref_id:b38 Title: Context-aware attention network for imagetext retrieval Year: (2020)
