Title: UFO: A Unified Approach to Fine-grained Visual Perception via Open-ended Language Interface
Abstract: Generalist models have achieved remarkable success in both language and visionlanguage tasks, showcasing the potential of unified modeling. However, effectively integrating fine-grained perception tasks like detection and segmentation into these models remains a significant challenge. This is primarily because these tasks often rely heavily on task-specific designs and architectures that can complicate the modeling process. To address this challenge, we present UFO, a framework that Unifies Fine-grained visual perception tasks through an Open-ended language interface. By transforming all perception targets into the language space, UFO unifies object-level detection, pixel-level segmentation, and image-level vision-language tasks into a single model. Additionally, we introduce a novel embedding retrieval approach that relies solely on the language interface to support segmentation tasks. Our framework bridges the gap between fine-grained perception and visionlanguage tasks, significantly simplifying architectural design and training strategies while achieving comparable or superior performance to methods with intricate task-specific designs. After multi-task training on five standard visual perception datasets, UFO outperforms the previous state-of-the-art generalist models by 12.3 mAP on COCO instance segmentation and 3.3 mIoU on ADE20K semantic segmentation. Furthermore, our method seamlessly integrates with existing MLLMs, effectively combining fine-grained perception capabilities with their advanced language abilities, thereby enabling more challenging tasks such as reasoning segmentation. Code and models are available at https://github.com/nnnth/UFO.

Section: Introduction
Multimodal large language models (MLLMs) [51,92,40,2,12,46] have made significant progress, exhibiting outstanding performance on various visual tasks. Despite these achievements, their scopes are largely confined to image-level vision-language tasks, leaving fine-grained perception (e.g., detection and segmentation) as a critical weakness. Recent studies have shown that enabling MLLMs to collaborate with off-the-shelf detectors and segmenters can enhance precise visual understanding [80,18] and facilitate advanced applications such as mobile agents [71,70,82,41], indicating that endowing MLLMs with fine-grained perception capabilities is beneficial. However, seamlessly integrating these tasks into MLLMs poses challenges because traditional specialized methods heavily rely on complex and task-specific designs, such as RPN [59] and mask decoders [34]. <bg>… <sheep>… Multi-Modal Transformer Please locate/segment the bottom goat. It is <DET>. Multi-Modal Transformer <56><4><96><12> Multi-Modal Transformer Please locate/segment the bottom goat. It is <MASK>. Please locate/segment the bottom goat.
(a) Task-Decoder-based (b) Text-based (c) Ours … Image Features … <sp> <sp> <sp> <sp> <sp> <sp> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> <bg> Mask Decoder It is <SEG>. It is <box>56,4,96,12</box>.
this section cite: ['b50', 'b91', 'b39', 'b1', 'b11', 'b45', 'b79', 'b17', 'b70', 'b69', 'b81', 'b40', 'b58', 'b33']

Section: Or

this section cite: []

Section: Polygon Textual Class retrieval
Box Decoder Figure 1: Methods to augment MLLMs with fine-grained perception tasks. (a) Relying on task decoders [37,77], (b) Previous text-based methods represent boxes with location tokens [52] and represent masks with suboptimal polygons [74,69] or textual classes [69,38], (c) Ours: predicting open-ended text sequences while using a simple yet effective embedding retrieval approach for masks.
Many existing works [37,77,57,60,53,89,11] augment MLLMs with task-specific decoders, such as LISA [37] using SAM for segmentation or VisionLLM v2 [77] adding box and mask decoders. However, this combination introduces several limitations. First, task decoders add architectural complexity, necessitating compatibility among multiple components whenever the LLM is scaled up or new task decoders are introduced. Second, it complicates the end-to-end training. For example, the last stage in VisionLLM v2 [77] is dedicated to finetuning the task decoders because they fail to converge in earlier stages. These issues create a significant discrepancy with traditional visionlanguage modeling, limiting their broader application in general-purpose MLLMs. To remove task decoders, another line of research [52,7,81,74,54] converts boxes into location tokens or textual numbers and transform masks into polygon vertices. However, using a limited number of vertices for masks introduces quantization errors, especially for masks with complex shapes and multiple regions.
To address the above limitations, GiT [69] uses two mask representations: for instance segmentation, it uses polygons, and for semantic segmentation, it predicts textual classes for each pixel, as shown in Figure 1 (b). However, it still falls short of unifying fine-grained perception tasks into MLLMs. Firstly, although textual class can represent masks with any shape, it results in overly long sequences and slow inference. Secondly, to achieve better performance, GiT sets specific vocabularies for each task (e.g., detection can only output location tokens), which is incompatible with open-ended text generation. Finally, GiT does not scale to MLLMs and uses a Vision Transformer (ViT [21]) for multimodal tasks, resulting in poor language abilities. Hence, it is essential to develop a more effective approach to unify fine-grained perception into MLLMs. This method should effortlessly align with open-ended language interfaces, involve minimal structural complexity and deliver excellent performance.
In this paper, we present UFO, which unifies fine-grained perception tasks through the same openended language interface as vision-language tasks, without any task decoders. By carefully organizing and translating all task outputs into open-ended text sequences, we demonstrate that competitive performance can be achieved without complex task-specific designs. As illustrated in Figure 1 (c), we reformulate segmentation as an embedding retrieval problem, where the mask token embedding computes similarity with image features by dot product, retrieving high-similarity positions to generate the mask. This design effectively leverages the output image features processed by MLLMs, which are often overlooked in previous methods. Our intuition is that since MLLMs achieve strong visual understanding, the mask information is already in the image features and we just need to retrieve it. Furthermore, we introduce a novel method that upsamples output masks by predicting multiple mask tokens, resulting in more refined masks and improved performance. Thanks to this strategy, we can efficiently and accurately represent masks of any shape using only 16 tokens.
We first validate our method following GiT [69], which uses a lightweight ViT but can share the same formulation as MLLMs (see Table 1), allowing efficient validation. GiT constructs a comprehensive multi-task benchmark, which covers various granularity fine-grained perception tasks. Under the same evaluation protocols, UFO outperforms GiT by 12.3 mAP and 3.3 mIoU in COCO instance segmentation and ADE20K semantic segmentation (see Table 2). We then scale our method to MLLMs to integrate language abilities with fine-grained perception. As shown in Table 3, UFO achieves competitive results in visual grounding without decoders or polygon approximations. Benefiting from the shared representations of the open language interface, UFO can deeply unify textual reasoning and image segmentation, surpassing the previous state-of-the-art method on the challenging ReasonSeg [37] benchmarks by 6.2 gIoU (see Table 4).
In summary, our contributions are listed as follows:
(1) We introduce UFO, a unified framework for diverse fine-grained perception tasks through the same open-ended language interface as vision-language tasks, without task-specific decoders.
(2) We reformulate segmentation as an embedding retrieval problem, exploring both text generation and image representation abilities of the language interface, significantly outperforming previous text-based methods on instance and semantic segmentation tasks.
(3) Our framework seamlessly integrates with MLLMs, delivering better performance than previous state-of-the-art methods on the ReasonSeg benchmarks.
2 Related Work
this section cite: ['b36', 'b76', 'b51', 'b73', 'b68', 'b68', 'b37', 'b36', 'b76', 'b56', 'b59', 'b52', 'b88', 'b10', 'b36', 'b76', 'b76', 'b51', 'b6', 'b80', 'b73', 'b53', 'b68', 'b20', 'b68', 'b36']

Section: Multimodal Large Language Models
Inspired by the success of large language models (LLMs), multimodal large language models (MLLMs) have rapidly advanced in recent years. Early efforts [43,92,17] finetune LLMs with instruction datasets, demonstrating strong multimodal understanding. More advanced MLLMs like Qwen2.5-VL [2], and InternVL2.5 [12] have emerged recently, offering superior multimodal comprehension through larger model sizes and extensive training data. However, these models mainly focus on image-level vision-language tasks, with less exploration of fine-grained visual perception.
this section cite: ['b42', 'b91', 'b16', 'b1', 'b11']

Section: Extend MLLMs with Fine-grained Perception
Extend MLLMs with Task Decoders. Recent works [37,77,57,60,3,87,53,78,88,84,89,76] introduce task decoders to extend MLLMs with tasks like detection and segmentation. These models treat the MLLM as a coarse proposal extractor, passing the task-relevant embeddings to specialized decoders. The decoders then manage task-specific details, such as regressing boxes or generating masks. Although this approach yields strong performance, extra task decoders complicate architectures and training, undermining the unified design of MLLMs and limiting their potential. Recently, HiMTok [73] reformulates segmentation as mask image generation. However, this approach still requires training a specialized VQ decoder, increasing both training and structural complexity.
this section cite: ['b36', 'b76', 'b56', 'b59', 'b2', 'b86', 'b52', 'b77', 'b87', 'b83', 'b88', 'b75', 'b72']

Section: Extend
MLLMs with Text Outputs. For object-level tasks, previous methods [52,7,81,68,74,6,54] have employed location tokens or textual numbers to represent boxes. For pixel-level tasks such as segmentation, a common format is polygonal approximation [74,54]. Although VistaLLM [54] reduces the errors of polygons through adaptive sampling, it is inadequate for general segmentation tasks. First, polygons struggle to accurately represent "stuff" categories with amorphous regions (e.g., roads with parked cars), which are common in real world [90,4,16]. Second, polygons inherently cause information loss, especially for detailed structures like retinal vessels [64]. Text4Seg [38] directly predicts textual labels for image patches but still requires an additional refiner (e.g., SAM [34]) to achieve better performance. In contrast, UFO leverages the multimodal outputs of MLLMs to generate precise masks for any shape, offering greater expressiveness and improved performance.
this section cite: ['b51', 'b6', 'b80', 'b67', 'b73', 'b5', 'b53', 'b73', 'b53', 'b53', 'b89', 'b3', 'b15', 'b63', 'b37', 'b33']

Section: Vision Generalist Models
Vision generalist models aim to establish a unified framework supporting various vision-centric tasks. Inspired by the seq2seq framework in NLP, previous generalist models [72,69,9,47] transform visual tasks into sequence generation problems. Notably, GiT [69] unifies five core visual tasks by language interface, supporting box, mask, and text outputs. However, these models typically focus solely on visual tasks and lack the advanced language capabilities required for complex reasoning [37].
this section cite: ['b71', 'b68', 'b8', 'b46', 'b68', 'b36']

Section: Methods
As our method is applicable to various multimodal architectures, we first present a unified architectural abstraction in Section 3.1. Then, in Sections 3.2 and 3.3, we explain how to integrate box and mask representations into the open-ended language interface. Finally, in Section 3.4, we describe our multi-task data template for joint training.
Table 1: We abstract current multimodal architectures into three components: (1) Image tokenizer, converting images into visual tokens; (2) Text tokenizer, outputting text tokens; (3) Multimodal transformer, jointly processing visual and text tokens. We construct three variants by this formulation.
this section cite: []

Section: Model
Image Tokenizer Text Tokenizer Multimodal Transformer LLaVA [43] CLIP [56],MLP Llama Tokenizer [67] Vicuna [14] EVE [20] Patch embedding Llama Tokenizer [67] Vicuna [14] GiT [69] Patch embedding Bert Tokenizer [19] ViT [21] UFO-ViT Patch Embedding Bert Tokenizer [19] ViT [21] UFO-LLaVA-1.5-7B CLIP [56],MLP Llama Tokenizer [67] Vicuna 1.5 [14] UFO-InternVL2.5-8B InternViT [12],MLP InternLM2.5 Tokenizer [86] InternLM2.5-7B [86]
this section cite: ['b42', 'b55', 'b66', 'b13', 'b19', 'b66', 'b13', 'b68', 'b18', 'b20', 'b18', 'b20', 'b55', 'b66', 'b13', 'b11', 'b85']

Section: Preliminary
Our goal is to unify fine-grained perception tasks into the open-ended language interface, thereby ensuring compatibility with any multimodal architecture that supports the same interface. We abstract existing multimodal architecture into three components based on the modalities they process: image tokenizer, text tokenizer and multimodal transformer, as shown in Table 1. For example, in LLaVA [43], the image tokenizer includes a vision encoder and MLP connector that extract visual features and map them into the LLM's input space, while the multimodal transformer corresponds to the LLM. This abstraction applies not only to MLLMs with various image tokenizers [43,40,20] but also to vision generalist models with similar architectures [69], significantly broadening the scope of our method. To avoid confusion, we will refer to MLLMs by default in the following sections.
this section cite: ['b42', 'b42', 'b39', 'b19', 'b68']

Section: Bounding Box Representation
To align with the open-ended language interface while avoiding the addition of extra location tokens, we directly translate boxes into textual numbers. Each box is represented by the coordinates of its top-left (x 1 , y 1 ) and bottom-right (x 2 , y 2 ) corners. The continuous values of these coordinates are discretized into integers within [0, range], enclosed by <box> and </box> tokens. If a class label is required, we simply prepend the textual class before the <box> token. For example, a box of a person can be represented as: person,<box>465,268,589,344</box>. This method converts boxes to open-ended sequences, effectively aligning with vision-language tasks.
this section cite: []

Section: Multi-Modal Transformer
It is <MASK>.
this section cite: []

Section: Input Texts 𝒙 𝒕

this section cite: []

Section: Interpolate
It is <MASK><MASK><MASK><MASK>.
N 2 =4 Input Image 𝒙 𝒗 Output Mask ! 𝑀(
16x16) Final Mask (448x448) dot product Image Feature 𝒉 𝒗 Interpolate Output Mask ! 𝑀 "# (32x32) Final Mask (448x448) Image Tokenizer Text Tokenizer dot product It is located at <Box>56,4,96,122</Box>. (a) Segmentation (b) Seg with multiple mask tokens (c) Detection Outputs Can you segment/ locate the goat nearest to the bottom stone? Image Feature 𝒉 𝒗 𝒆 𝒎 > 0 > 0 𝒆 𝒎𝟏 𝒆 𝒎𝟐 𝒆 𝒎𝟑 𝒆 𝒎𝟒 𝒉 𝒕 𝒉 𝒗 𝒚 𝒕 𝒚 𝒕 𝒚 𝒕 𝒉 𝒗 𝒉 𝒕 𝒉 𝒕
this section cite: []

Section: Mask Representation
Representing masks via the language interface is more challenging because masks contain more detailed information than boxes. Previous methods either use polygon formats, which sacrifice details, or assign textual classes to each pixel, resulting in overly long sequences. Therefore, a more efficient method to represent detailed masks is needed.
We observe that in MLLMs, the language interface is actually multimodal, where projected image features and text features are combined and jointly processed by the LLM. However, most existing methods ignore the output image features processed by the LLM. We argue that since MLLMs can express where and what objects are in text form, the mask information is already encoded in the image features. We just need to teach the model to decode this information. Therefore, we design a representation method based on image features and text embeddings. Instead of storing mask information in text embeddings, we use the text embeddings as query embeddings to extract mask information from the image features. The detailed approach is described below. Segmentation by Embedding Retrieval. To incorporate the segmentation task using only the language interface, we reformulate it as an embedding retrieval problem. We first augment the basic vocabulary of the model with a <MASK> token, which serves as the indicator for mask generation.
When performing segmentation, the model is trained to output the <MASK> token, as shown in Figure 2 (a). Formally, given an input image x v and a segmentation prompt x t , the model F generates the text response y t and corresponding output embeddings h t , image features h v as:
h v , y t , h t = F(x v , x t ).(1)
We extract the mask token embedding e m corresponding to the <MASK> token from h t . To generate the segmentation mask, we compute the similarity between the mask token embedding e m and the image features h v via a scaled dot product. Positive scores are retrieved to form the binary mask M. This process is expressed as:
s = e m h v ⊤ √ d , M = I(s > 0),(2)
where d is the embedding dimension, s represents the similarity scores, and I is the indicator function that converts the similarity scores into a binary mask. By computing the dot product similarity between the mask token embedding and image features, we retrieve the most relevant image features corresponding to the mask token, thereby producing a mask aligned with the original image.
Our approach leverages MLLMs' inherent capabilities for segmentation without task decoders. We hypothesize that, in well-encoded image features, features with the same semantics will group into clusters. Therefore, generating a mask token embedding equates to identifying the center of the relevant image feature cluster, while computing the similarity reflects this relationship. This approach can easily apply to other pixel-level tasks, such as depth estimation (see Table 20 in the appendix).
this section cite: []

Section: Upsampling by Multiple Mask Tokens.
Due to the redundancy in visual information, it is common to process visual features at reduced resolutions. For example, the CLIP-L/14 [56] downsamples image features by a factor of 14. In above method, similarities are computed using downsampled image features, resulting in low-resolution masks. However, directly upsampling by interpolation leads to coarse results and suboptimal performance due to the high interpolation factor.
To address this issue, we propose an upsampling method by predicting multiple mask tokens. For an image x v ∈ R H×W ×3 , we obtain image features h v ∈ R Hp×Wp×d downsampled by the patch size p, where d represents the feature dimension. Our target is to upsample the generated mask by N times, producing Mup ∈ R (HpN )×(WpN ) from image features h v ∈ R Hp×Wp×d . This requires decoding an N × N mask for each position in the image features. To achieve this, we train the model to autoregressively predict N 2 <MASK> tokens with embeddings {e m i } N 2 i=1 . Each token corresponds to a single position in the N × N upsampling grid, as illustrated in Figure 2 (b). For each mask token embedding e m i , we compute the similarity with the visual features h v :
s i = e m i h v ⊤ √ d ,(3)
where e m i ∈ R 1×d , h v ⊤ ∈ R d×Hp×Wp , and s i ∈ R 1×Hp×Wp . These similarity scores {s i } N 2 i=1 are then concatenated and reshaped into an upsampled similarity map:
s concat = concat({s i } N 2 i=1 ), s concat ∈ R N 2 ×Hp×Wp ,(4)
s up = reshape(s concat ), s up ∈ R (HpN )×(WpN ) .(5)
Sure, It is <MASK>…<MASK>. Can you segment the goat nearest to the stone? Multi-Modal Transformer <Text Prompt> <Image> Mask Output Detect cow, person,duck. Multi-Modal Transformer <Text Prompt> <Image> Multi Prediction <Local Feature> cow, <box>…</box> … … duck, <box>…</box> Can you locate the goat nearest to the stone? Multi-Modal Transformer <Text Prompt> <Image> Box Output Sure, it is <box>56,4,96,122</box> Finally, we retrieve positive scores in s up to generate the upsampled binary mask Mup . By default, we set N = 4, predicting 16 <MASK> tokens, which upsamples the output mask by a factor of 4. The mask is then aligned with the original image resolution through interpolation.
Our method effectively leverages mask token embeddings as upsampling parameters, offering greater flexibility than traditional methods like bilinear interpolation and transposed convolution. Bilinear interpolation uses non-learnable parameters, while transposed convolution allows for learnable parameters, the same parameters are applied to all images after training. In contrast, we use embeddings generated by the network as the parameters, which can be customized for each image. This approach enables the model to generate optimal upsampling parameters dynamically, enhancing flexibility while achieving better performance (see Table 6).
Note that our method is fully compatible with open-ended language interfaces. We refer "open-ended" as the capability to generate variable-length text sequences terminated by an end-of-sequence token. In our approach, while we require fixed-length <MASK> tokens for segmentation, the text generation itself remains open-ended. Only after the generation is complete, we check if the output contains <MASK> segments of the required length. Furthermore, the interaction with image features is also performed after text generation is finished.
this section cite: ['b55']

Section: Multi-Task Data Template
Based on the above designs, we construct multi-task data templates for joint training. We classify tasks into two categories based on prediction number: single-prediction tasks like grounding produce one box or mask, and multi-prediction tasks like object detection generate several boxes. Merging multiple outputs into a long sequence is inefficient and the order among them is hard to define, making autoregressive learning of the sequence difficult [8]. Therefore, we adopt a parallel decoding approach that splits multi-prediction tasks into independent subtasks, each handling one prediction in parallel. This strategy effectively accelerates inference and enhances task scalability. Single prediction. For tasks only require a single prediction, our task template is: <Text Prompt><Image><Text Response>. As shown in Figure 3, we follow previous methods to construct text prompts and use our unified box and mask representation for text responses.
this section cite: ['b7']

Section: Multiple predictions.
To efficiently support multi-prediction tasks, we split complex tasks into independent subtasks with single prediction, enabling parallel decoding within a batch. The key to achieving parallelism is to ensure all subtasks are independent. Typically, multiple boxes and masks correspond to different locations. Therefore, we introduce local image features in the input to differentiate these sub-tasks, serving as visual prompts. The template is structured as follows: <Text Prompt><Image><Local><Text Response>, where <Local> refers to local image features interpolated by grids sampled on the image. The core idea is that each grid point is responsible for detecting its spatially nearest objects. During training, each ground-truth object is assigned to its nearest grid point, while the remaining grid points are assigned to predict end-of-sequence tokens. During inference, as illustrated in Figure 3, we sample points over the image, typically of size K × K, resulting in a total of M = K 2 points. We then interpolate the image features at each of the M grid locations to extract distinct grid features. These grid features, along with the global image features and the text prompt, are fed into the LLM. The example input sequence is structured as follows:
Detect cow, person, duck. < Image >< Local 1 >< Local 2 > . . . < Local M >(6)
Table 2: Results on GiT [69]'s multi-task benchmark. "⋆" denotes the model is capable of the task but no number is reported. "-" means incapability. We highlight joint training improvements with bold font and follow [69] to list modules for specific functions.
Specific Modules Object Detection Instance Seg Semantic Seg Captioning REC Methods Examples Num #Params AP AP50 AP75 AP AP50 AP75 mIoU(SS) BLEU-4 CIDEr Acc@0.5 Specialist Models Deformable-DETR [94] RegressionHead 5 40M 45.4 64.7 49.0 -------Mask R-CNN [26] FPN,RPN 6 46M 41.0 61.7 44.9 37.1 58.4 40.1 ----Polar Mask [79] CenternessHead 5 55M ---30.5 52.0 31.1 ----Mask2Former [13] PixelDecoder 5 44M ---43.7 --47.2 ---VL-T5 [15] Faster R-CNN 3 440M -------34.5 116.5 -MDETR [30] RoBERTa,DETR 6 188M ---------86.8 Generalist Models (MultiTask-Training) Uni-Perceiver [95] None 1 124M -------32.0 ⋆ ⋆ Uni-Perceiver-MoE [93] None 1 167M -------33.2 ⋆ ⋆ VisionLLM-R50 [74] Deform-DETR 6 7B 44.6 64.0 48.1 25.1 50.0 22.4 -31.0 112.5 80.6 GiT-Bsingle-task [69] None 1 131M 45.1 62.7 49.1 31.4 54.8 31.2 47.7 33.7 107.9 83.3 GiT-Bmulti-task [69] None 1 131M 46.7 64.2 50.7 31.9 56.4 31.4 47.8 35.4 112.6 85.8 GiT-Lmulti-task [69] None 1 387M 51.3 69.2 55.9 35.1 61.4 34.7 50.6 35.7 116.0 88.4 GiT-Hmulti-task [69] None 1 756M 52.9 71.0 57.8 35.8 62.6 35.6 52.4 36.2 118.2 89.2 UFO-ViT-Bsingle-task None 1 131M 47.8 65.7 52.0 42.6 65.8 46.1 49.5 34.2 111.1 83.6 UFO-ViT-Bmulti-task None 1 131M 48.3 66.6 52.6 43.5 66.2 47.0 50.2 35.3 114.2 85.8 Improvement (single→multi) +0.5 +0.9 +0.6 +0.9 +0.4 +0.9 +0.7 +1.1 +3.1 +2.2 UFO-ViT-Lmulti-task None 1 387M 52.9 71.3 57.9 47.3 70.9 51.6 54.0 35.9 118.6 88.5 UFO-ViT-Hmulti-task None 1 756M 54.1 72.4 58.9 48.1 71.6 53.0 55.7 37.6 123.6 89.2 UFO-InternVL2.5-8Bmulti-task None 1 8B 52.3 71.7 56.5 45.8 69.5 49.7 54.6 39.6 131.6 90.4
To enforce the independence of predictions for each grid point, we modify the self-attention mask to isolate each grid feature from the others. This ensures that the generation for one point does not influence another. Then we start generating in an autoregressive manner, with the difference that we predict M tokens at each forward step instead of just one. The generation process looks like this:
< Local 1 > . . . < Local M > | < T 1 1 > . . . < T 1 M > | < T 2 1 > . . . < T 2 M > | . . .(7)
where T j i denotes the j-th generated token for the i-th grid sequence, and | distinguishes the tokens produced in each forward pass. This decoding strategy shares the same philosophy as blockwise prediction [65] in LLMs, accelerating inference by generating multiple tokens simultaneously.
After decoding, we obtain M output sequences. For detection, some might identify objects (Duck, <box>... or Cow, <box>...), while others corresponding to empty regions will predict end-ofsequence tokens. For instance segmentation, the process is identical, with the textual box representations (<box>...) being replaced by mask tokens (<MASK>...). This approach not only enhances efficiency but also simplifies the problem by breaking it down into simple, localized prediction tasks.
this section cite: ['b68', 'b68', 'b64']

Section: Training
To ensure efficient validation and fair comparison, we first follow GiT [69], using a smaller ViT as the multimodal transformer for multi-task training across five standard visual perception tasks. We then scale to MLLMs, validating on the same multi-task benchmark. Finally, we enrich the data by incorporating more diverse datasets, enabling fine-grained instruction tuning of MLLMs. After instruction tuning, the fine-grained perception capabilities are seamlessly integrated with the robust language abilities of MLLMs, thereby applying to perception tasks that require advanced language capabilities, such as reasoning segmentation.
this section cite: ['b68']

Section: Multi-Task Training
Architecture. To ensure fair comparison and validate our effectiveness across various architectures, we conduct multi-task training using two variants: UFO-ViT and UFO-InternVL2.5-8B. UFO-ViT strictly follows GiT [69], employing a SAM [34]-pretrained ViT [21] and a text tokenizer from BERT [19]. It is available in three sizes: ViT-B, ViT-L, and ViT-H. UFO-InternVL2.5-8B utilizes the pretraining weight of InternVL2.5-8B [12], with detailed model specifications provided in Table 1. Datasets. We use the same multi-task dataset as GiT: COCO 2017 [42] for object detection and instance segmentation, COCO Caption [10] for image captioning, the RefCOCO series [48,83] for referring expression comprehension (REC), and ADE20K [90] for semantic segmentation.  .3 84.4 90.3  78.7 86.4 86.8 87.2 77.2 80.1  76.4 71.8 77.9  70.2 74.1 73.5
this section cite: ['b68', 'b33', 'b20', 'b18', 'b11', 'b41', 'b9', 'b47', 'b82', 'b89']

Section: Fine-grained Instruction Tuning
Architecture. To demonstrate that our method is applicable to various MLLMs, we use not only InternVL2.5-8B [12] but also the LLaVA-1.5-7B [44] for pretraining, specifically UFO-LLaVA-1.5-7B. Architecture details are in Table 1.
this section cite: ['b11', 'b43']

Section: Datasets.
To enhance the model's versatility, we enrich the training data to 2.5M across 6 tasks, including VQA data from [39], COCO-Stuff [4], LVIS [25], etc. We additionally add RES task on the basis of five tasks in multi-task training. More details of data composition are in Table 8.
this section cite: ['b38', 'b3', 'b24']

Section: Experiments

this section cite: []

Section: Experimental Settings
Multi-Task Training Details. To facilitate comparison with specialist models, we also conduct single-task training independently on five selected tasks. For both single-task and multi-task training, we use a batch size of 24 and employ the AdamW [33] optimizer with a cosine annealing schedule, setting the initial learning rate to 0.0002. More details are in the appendix.
Fine-grained Instruction Tuning Details. In training, we use a batch size of 32 with gradient accumulation set to 16, running on 8 NVIDIA A100 GPUs for 120K iterations. The AdamW [33] optimizer and a cosine annealing schedule are employed, with a learning rate of 0.0002 and weight decay of 0.01. For efficient training, we employ LoRA [27] with a rank of 8, freezing the image tokenizer while keeping only the LLM trainable. More details are in Appendix Table 12.
Training Objectives. All tasks utilize a CrossEntropy Loss as they are unified under the open-ended language interface. For segmentation tasks, we additionally apply focal loss [61] and dice loss to supervise the mask output. The final loss for segmentation tasks is expressed as:
L seg = λ CE L CE + λ focal L focal + λ dice L dice
We find that setting all weights to 1 offers better overall performance. See appendix for more details.
this section cite: ['b32', 'b32', 'b26', 'b60']

Section: Multi-Task Evaluation
We evaluate performance in both single-task and multi-task settings across five vision-centric tasks, benchmarking it against specialized and generalist models. Without task decoders, our model adapts to various tasks by the open-ended language interface and achieves outstanding performance.
Comparison with Specialist Models. As shown in Table 2, our single-task model effectively bridges the performance gap with specialized models, achieving superior performance. For example, we achieve 47.8 mAP in detection compared to 45.4 mAP with Deformable-DETR [94] and 49.5 mIoU in semantic segmentation against 47.2 mIoU with Mask2Former [13]. In instance segmentation, we also outperform specialized methods like Mask R-CNN [26] while matching Mask2Former [13].
Comparison with Generalist Models. To facilitate comparison with GiT [69], we adopt its one-stage training without task-specific tuning. This involves jointly training on a mixed dataset of the five tasks and directly testing on their respective validation or test sets. Table 2 shows that our model outperforms the previous leading generalist model, GiT, across all tasks, with the same pretraining and data. Notably, in the largest ViT size, we outperform GiT by 12.3 mAP on COCO instance segmentation and 3.3 mIoU on ADE20K semantic segmentation, demonstrating the superiority of our segmentation modeling. We also surpass GiT 5.3 CIDEr in captioning, primarily due to our shared vocabulary across all tasks, while GiT uses task-specific vocabularies, hindering the task synergy.
We also observe a multi-task synergy effect like GiT, with performance on instance segmentation improved by 0.9 mAP and captioning increased by 3.1 CIDEr. Our multi-task improvements on segmentation also outperform GiT (0.7 vs. 0.1 mIoU). We attribute this to unified modeling across segmentation tasks, whereas GiT employs separate methods for instance and semantic segmentation.
After scaling to MLLMs, we observe improved performance on captioning and REC, while other tasks remain comparable to the UFO-ViT-L. We speculate that this performance difference primarily arises from different pretraining. For UFO-ViT, we use SAM [34] pretraining, making it more aligned with detection and segmentation. In contrast, InternVL2.5-8B is mainly pre-trained on image-level vision-language tasks, which better suit captioning and REC.
this section cite: ['b93', 'b12', 'b25', 'b12', 'b68', 'b33']

Section: Fine-grained Instruction Tuning Results
Visual Grounding can be categorized into referring expression comprehension (REC) and segmentation (RES). We comprehensively list the results for the two tasks in Table 3. We report results in two settings: direct evaluation after joint training and specifically finetuning. Without using box decoders, our best model can surpass the VisionLLM v2 [77] by an average of 3.0%. After specific finetuning, our model achieves comparable performance with the state-of-the-art method Ferret-v2-7B [85]. While all previous approaches rely on mask decoders or polygon approximations for segmentation, our method delivers superior or comparable performance without them. For instance, our InternVL2.5 variant outperforms the SAM4MLLM [11] by an average of 1.9 cIoU and matches with HiMTok [73]. These outcomes validate the effectiveness of our method, demonstrating that with proper task modeling, MLLMs can handle fine-grained perception tasks without task decoders.
Reasoning Segmentation (ReasonSeg) is a challenging benchmark introduced by LISA [37], which presents more sophisticated and nuanced instructions, requiring models to leverage world knowledge and engage in deeper logical reasoning. We report both zero-shot and finetuned results. As shown in Table 4, with the same pretraining, our InternVL2.5 variant outperforms HiMTok [73] by 6.2 gIoU in finetuned settings. Notably, both Cores [3] and HiMTok [73] design a CoT strategy to generate segmentation masks progressively: answer the question with text first and then perform segmentation on the answered objects. In contrast, our method achieves better performance without this strategy, which implies that we can effectively perform reasoning while generating mask embeddings.
Since ReasonSeg requires both reasoning and precise segmentation, we attribute our improvement to better task integration through unified modeling. In decoder-based methods, the MLLM handles only language reasoning and generates coarse segmentation prompts, relying on an additional mask decoder for finer segmentation. This leads to information loss and insufficient synergy. In our unified modeling, the MLLM manages both language reasoning and precise segmentation, allowing different task capabilities to fully integrate within a shared parameter space, thereby enhancing synergy.
this section cite: ['b76', 'b84', 'b10', 'b72', 'b36', 'b2', 'b72']

Section: Visual Question Answering.
Table 5 presents the performance on four VQA benchmarks (GQA [28], MMBench [45], MMVP [66], HallusionBench [24]). Thanks to our unification with the language interface, the model's original performance is essentially maintained. Notably, we achieved a 0.6 improvement on [24], which may indicate that fine-grained fine-tuning helps reduce hallucinations.
this section cite: ['b27', 'b44', 'b65', 'b23', 'b23']

Section: Ablation Study
Number of Mask tokens. We ablate the number of mask tokens on the COCO instance segmentation task. As shown in Table 6, using multiple tokens significantly improves performance compared to a single token, but the gains plateau after 16. Considering the increased training and inference costs, we set N 2 = 16 by default for balance. The visualization of mask tokens is in Appendix Figure 8.
Open-ended decoding. We explore the impact of our open-ended decoding on single-task detection.
As noted in Section 3.4, we split object detection into sub-tasks for each grid point (e.g., 625 grid points for a 1120×1120 image), which leads to an imbalance between positive and negative samples due to more grid points than objects. Our method utilizes a standard vocabulary (e.g., BERT's 30,524 tokens) for open-ended decoding. This output space is much bigger than the range of positive classes (e.g., 80 in COCO), worsening class imbalance and reducing positive predictions in inference. As shown in Table 7, while using decoding rules like removing negative classes from the vocabulary [69] could force all outputs to be positive, this compromises our generality. Therefore, we use beam search [58], which allows the model to explore multiple potential sequences. This approach effectively increases positive predictions and improves performance. By default, we only apply beam search for COCO detection, instance segmentation, and image captioning.
this section cite: ['b68', 'b57']

Section: Conclusion
In this paper, we present UFO, a unified approach for various fine-grained visual perception tasks with an open-ended language interface. We translate all perception targets into open-ended text sequences and introduce a novel embedding retrieval method for segmentation. Experiments show that our method can achieve excellent performance on MLLMs without requiring architecture modifications.
Our unification fully aligns with vision-language tasks, providing a flexible, effective, and scalable solution to enhance the fine-grained perception capabilities of MLLMs, paving the way to build stronger and more general multimodal models.
Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Our training data is comprised of public datasets, which are detailed in Table 8.
Training and testing code is provided in supplemental materials. Guidelines:
• The answer NA means that paper does not include experiments requiring code.
• Please see the NeurIPS code and data submission guidelines (https://nips.cc/  public/guides/CodeSubmissionPolicy) for more details.
• While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:  //nips.cc/public/guides/CodeSubmissionPolicy) for more details.
• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted. 6.
this section cite: []

Section: References
Ref_id:b0 Title: Qwen-vl: A frontier large vision-language model with versatile abilities Year: (2023)
Ref_id:b1 Title: Qwen2. 5-vl technical report Year: (2025)
Ref_id:b2 Title: Cores: Orchestrating the dance of reasoning and segmentation Year: (2024)
Ref_id:b3 Title: Coco-stuff: Thing and stuff classes in context Year: (2018)
Ref_id:b4 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b5 Title: Minigpt-v2: large language model as a unified interface for vision-language multi-task learning Year: (2023)
Ref_id:b6 Title: Shikra: Unleashing multimodal llm's referential dialogue magic Year: (2023)
Ref_id:b7 Title: Pix2seq: A language modeling framework for object detection Year: (2022)
Ref_id:b8 Title: A unified sequence interface for vision tasks Year: (2022)
Ref_id:b9 Title: Microsoft coco captions: Data collection and evaluation server Year: (2015)
Ref_id:b10 Title: Sam4mllm: Enhance multi-modal large language model for referring expression segmentation Year: (2024)
Ref_id:b11 Title: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling Year: (2024)
Ref_id:b12 Title: Maskedattention mask transformer for universal image segmentation Year: (2022)
Ref_id:b13 Title: Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality Year: (2023)
Ref_id:b14 Title: Unifying vision-and-language tasks via text generation Year: (2021)
Ref_id:b15 Title: The cityscapes dataset for semantic urban scene understanding Year: (2016)
Ref_id:b16 Title: Instructblip: Towards general-purpose vision-language models with instruction tuning Year: (2023)
Ref_id:b17 Title: Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models Year: (2024)
Ref_id:b18 Title: Bert: Pre-training of deep bidirectional transformers for language understanding Year: (2018)
Ref_id:b19 Title: Unveiling encoder-free vision-language models Year: (2024)
Ref_id:b20 Title: An image is worth 16x16 words: Transformers for image recognition at scale Year: (2021)
Ref_id:b21 Title: Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image Year: (2024)
Ref_id:b22 Title: Simple copy-paste is a strong data augmentation method for instance segmentation Year: (2021)
Ref_id:b23 Title: Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models Year: (2024)
Ref_id:b24 Title: Lvis: A dataset for large vocabulary instance segmentation Year: (2019)
Ref_id:b25 Title: Mask r-cnn Year: (2017)
Ref_id:b26 Title: Lora: Low-rank adaptation of large language models Year: (2022)
Ref_id:b27 Title: Gqa: A new dataset for real-world visual reasoning and compositional question answering Year: (2019)
Ref_id:b28 Title: Detrs with hybrid matching Year: (2023)
Ref_id:b29 Title: Mdetr-modulated detection for end-to-end multi-modal understanding Year: (2021)
Ref_id:b30 Title: Referitgame: Referring to objects in photographs of natural scenes Year: (2014)
Ref_id:b31 Title: Repurposing diffusion-based image generators for monocular depth estimation Year: (2024)
Ref_id:b32 Title: A method for stochastic optimization Year: (2014)
Ref_id:b33 Title: Segment anything Year: (2023)
Ref_id:b34 Title: The hungarian method for the assignment problem Year: (1955)
Ref_id:b35 Title: The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale Year: (2020)
Ref_id:b36 Title: Reasoning segmentation via large language model Year: (2024)
Ref_id:b37 Title: Text4seg: Reimagining image segmentation as text generation Year: (2025)
Ref_id:b38 Title: Llava-onevision: Easy visual task transfer Year: (2024)
Ref_id:b39 Title: Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models Year: (2023)
Ref_id:b40 Title: Mastering universal user interface understanding across platforms Year: (2024)
Ref_id:b41 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b42 Title: Visual instruction tuning Year: (2023)
Ref_id:b43 Title: Improved baselines with visual instruction tuning Year: (2024)
Ref_id:b44 Title: Mmbench: Is your multi-modal model an all-around player Year: (2024)
Ref_id:b45 Title: Deepseek-vl: towards real-world vision-language understanding Year: (2024)
Ref_id:b46 Title: UNIFIED-IO: A unified model for vision, language, and multi-modal tasks Year: (2023)
Ref_id:b47 Title: Generation and comprehension of unambiguous object descriptions Year: (2016)
Ref_id:b48 Title: The mapillary vistas dataset for semantic understanding of street scenes Year: (2017)
Ref_id:b49 Title: Convmixer: Feature interactive convolution with curriculum learning for small footprint and noisy far-field keyword spotting Year: (2022)
Ref_id:b50 Title:  Year: (2023)
Ref_id:b51 Title: Kosmos-2: Grounding multimodal large language models to the world Year: (2023)
Ref_id:b52 Title: Perceptiongpt: Effectively fusing visual perception into llm Year: (2024)
Ref_id:b53 Title: Rama Chellappa, and Amjad Almahairi. Jack of all tasks master of many: Designing general-purpose coarse-to-fine vision-language model Year: (2024)
Ref_id:b54 Title: Geonet++: Iterative geometric neural network with edge-aware refinement for joint depth and surface normal estimation Year: (2020)
Ref_id:b55 Title: Learning transferable visual models from natural language supervision Year: (2021)
Ref_id:b56 Title: Glamm: Pixel grounding large multimodal model Year: (2024)
Ref_id:b57 Title: Speech understanding systems: A summary of results of the five-year research effort at carnegie mellon university Year: (1977)
Ref_id:b58 Title: Faster r-cnn: Towards real-time object detection with region proposal networks Year: (2015)
Ref_id:b59 Title: Pixellm: Pixel reasoning with large multimodal model Year: (2024)
Ref_id:b60 Title: Focal loss for dense object detection Year: (2017)
Ref_id:b61 Title: Objects365: A large-scale, high-quality dataset for object detection Year: (2019)
Ref_id:b62 Title: Indoor segmentation and support inference from rgbd images Year: (2012)
Ref_id:b63 Title: Ridgebased vessel segmentation in color images of the retina Year: (2004)
Ref_id:b64 Title: Blockwise parallel decoding for deep autoregressive models Year: (2018)
Ref_id:b65 Title: Eyes wide shut? exploring the visual shortcomings of multimodal llms Year: (2024)
Ref_id:b66 Title: Open and efficient foundation language models Year: (2023)
Ref_id:b67 Title: Tokenformer: Rethinking transformer scaling with tokenized model parameters Year: (2024)
Ref_id:b68 Title: Git: Towards generalist vision transformer through universal language interface Year: (2024)
Ref_id:b69 Title: Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration Year: (2024)
Ref_id:b70 Title: Mobile-agent: Autonomous multi-modal mobile device agent with visual perception Year: (2024)
Ref_id:b71 Title: Ofa: Unifying architectures, tasks, and modalities through a simple sequence-tosequence learning framework Year: (2022)
Ref_id:b72 Title: Himtok: Learning hierarchical mask tokens for image segmentation with large multimodal model Year: (2025)
Ref_id:b73 Title: Visionllm: Large language model is also an open-ended decoder for vision-centric tasks Year: (2023)
Ref_id:b74 Title: Images speak in images: A generalist painter for in-context visual learning Year: (2023)
Ref_id:b75 Title: Towards universal visual segmentation with large language model Year: (2024)
Ref_id:b76 Title: Visionllm v2: An end-to-end generalist multimodal large language model for hundreds of vision-language tasks Year: (2024)
Ref_id:b77 Title: Gsva: Generalized segmentation via multimodal large language models Year: (2024)
Ref_id:b78 Title: Polarmask: Single shot instance segmentation with polar representation Year: (2020)
Ref_id:b79 Title: Set-of-mark prompting unleashes extraordinary visual grounding in gpt Year: (2023)
Ref_id:b80 Title: Ferret: Refer and ground anything anywhere at any granularity Year: (2023)
Ref_id:b81 Title: Ferret-ui: Grounded mobile ui understanding with multimodal llms Year: (2024)
Ref_id:b82 Title: Modeling context in referring expressions Year: (2016)
Ref_id:b83 Title: Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos Year: (2025)
Ref_id:b84 Title: Ferret-v2: An improved baseline for referring and grounding with large language models Year: (2024)
Ref_id:b85 Title: Internlm-xcomposer-2.5: A versatile large vision language model supporting long-contextual input and output Year: (2024)
Ref_id:b86 Title: Omg-llava: Bridging image-level, object-level, pixel-level reasoning and understanding Year: (2024)
Ref_id:b87 Title: Groundhog: Grounding large language models to holistic segmentation Year: (2024)
Ref_id:b88 Title: Psalm: Pixelwise segmentation with large multi-modal model Year: (2024)
Ref_id:b89 Title: Scene parsing through ade20k dataset Year: (2017)
Ref_id:b90 Title: Unet++: A nested u-net architecture for medical image segmentation Year: (2018)
Ref_id:b91 Title: Minigpt-4: Enhancing vision-language understanding with advanced large language models Year: (2023)
Ref_id:b92 Title: Uni-perceiver-moe: Learning sparse generalist models with conditional moes Year: (2022)
Ref_id:b93 Title: Deformable detr: Deformable transformers for end-to-end object detection Year: (2020)
Ref_id:b94 Title: Uniperceiver: Pre-training unified architecture for generic perception for zero-shot and few-shot tasks Year: (2022)
Ref_id:b95 Title: Generalized decoding for pixel, image, and language Year: (2023)
Ref_id:b96 Title: Segment everything everywhere all at once Year: (2024)
