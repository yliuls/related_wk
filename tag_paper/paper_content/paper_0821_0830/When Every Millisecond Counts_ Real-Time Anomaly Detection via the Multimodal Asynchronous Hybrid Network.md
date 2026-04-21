Title: When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network
Abstract: Anomaly detection is essential for the safety and reliability of autonomous driving systems. Current methods often focus on detection accuracy but neglect response time, which is critical in time-sensitive driving scenarios. In this paper, we introduce real-time anomaly detection for autonomous driving, prioritizing both minimal response time and high accuracy. We propose a novel multimodal asynchronous hybrid network that combines event streams from event cameras with image data from RGB cameras. Our network utilizes the high temporal resolution of event cameras through an asynchronous Graph Neural Network and integrates it with spatial features extracted by a CNN from RGB images. This combination effectively captures both the temporal dynamics and spatial details of the driving environment, enabling swift and precise anomaly detection. Extensive experiments on benchmark datasets show that our approach outperforms existing methods in both accuracy and response time, achieving millisecond-level real-time performance. The code is available at https:  //github.com/PKU-XD/EventAD.

Section: Introduction
revolutionize the transportation industry by enhancing road safety, reducing traffic congestion, and improving fuel efficiency (Huang et al., 2018;Caesar et al., 2020). The core of autonomous vehicles lies in their ability to perceive and interpret complex and dynamic driving environments accurately and efficiently. Among the various perception tasks, anomaly detection plays a pivotal role in ensuring the safety and reliability of autonomous systems. Anomalies, such as unexpected obstacles, erratic behaviors of other road users, or sudden changes in the environment, can pose significant risks if not promptly identified and addressed (Fang et al., 2024). For example, a pedestrian darting onto the road from behind a parked vehicle or a sudden road obstruction requires the autonomous vehicle to react within milliseconds to prevent potential accidents (Gehrig & Scaramuzza, 2024).
Anomaly detection is a vital component in ensuring the safety of autonomous driving systems. Despite substantial progress, current methods often prioritize detection accuracy over an equally crucial factor: response time (Cui et al., 2023;Zeng et al., 2023). Many state-of-the-art solutions rely on increasingly sophisticated deep neural networks, which can incur large inference latencies (Yao et al., 2022;Karim et al., 2023). In a domain where a delay of even a few hundred milliseconds can dictate the difference between safe braking and a collision (Tian et al., 2024), these lengthy detection times significantly compromise the very safety guarantees that autonomous vehicles are designed to deliver (Wang et al., 2023).
To overcome this limitation, we focus on the task of realtime anomaly detection in autonomous driving, specifically targeting the detection of sudden hazardous anomalies from the ego-vehicle perspective, while explicitly incorporating response time into the performance evaluation metrics. In addition to detection accuracy, this perspective emphasizes minimizing the total response time-encompassing both inference latency and the delay between anomaly occurrence and detection (Tian et al., 2024). By centering on this timecritical requirement, our approach seeks to bridge the gap between high-accuracy detection and the urgent demands of real-world driving scenarios, where prompt decisionmaking is paramount for safety.
To address the challenge of real-time anomaly detection, we introduce a multimodal asynchronous hybrid network that reduces inference latency and detection delays while preserving high accuracy. Our approach strategically combines event cameras and conventional RGB cameras to exploit their complementary advantages. Event cameras capture brightness changes at microsecond resolution, providing sparse yet highly informative data for dynamic scenes (Gallego et al., 2020). At the same time, RGB cameras deliver rich spatial context, albeit with higher latency and susceptibility to motion blur (Zhou & Jiang, 2024). By fusing the asynchronous, sparse event streams with continuous image data, we enable the model to capture both fine-grained spatial details and rapid temporal cues.
In particular, we employ an asynchronous Graph Neural Network (GNN) to process the event stream data, harnessing the inherent sparsity and asynchronous nature of event cameras (Li et al., 2021). Meanwhile, a CNN-based ResNet extracts high-level spatial features from the RGB images (He et al., 2016). Subsequently, by leveraging a GRU module to jointly learn the spatio-temporal relationships of objectlevel and frame-level event streams and RGB features, our model can anticipate anomaly trends in advance, achieving both accurate and rapid anomaly detection. Notably, we are the first to introduce the unique characteristics of event streams as critical features for road traffic anomaly detection, and we fully exploit their high temporal resolution and asynchronous nature through an asynchronous network architecture. This approach is specifically designed for realtime operation, enabling crucial millisecond-level response times in autonomous driving environments. Extensive experiments on multiple benchmark datasets demonstrate that our approach not only outperforms existing methods in de-tection accuracy but also substantially lowers response time, fulfilling the stringent safety requirements of real-world autonomous driving environments (Karim et al., 2023;Yao et al., 2022).
Our main contributions are summarized as follows:
• We formalize the task of real-time anomaly detection, emphasizing the pivotal role of rapid response in safetycritical autonomous driving scenarios. In doing so, we underscore how existing methods overlook this timesensitive aspect, compromising overall system safety.
• We propose a novel network architecture that synergistically fuses event stream data with RGB images to strike an optimal balance between minimal inference latency and high detection accuracy. By capitalizing on the asynchronous, fine-grained temporal information from event cameras and the rich spatial features of conventional images, our model delivers reliable performance even under challenging conditions.
• We conduct extensive experiments on multiple benchmark datasets, demonstrating that our approach not only surpasses state-of-the-art baselines in detection accuracy but also significantly reduces response time. These findings validate the practical effectiveness of our method, reinforcing its suitability for real-world autonomous driving applications.
this section cite: ['b24', 'b3', 'b12', 'b14', 'b8', 'b53', 'b50', 'b26', 'b43', 'b46', 'b43', 'b13', 'b55', 'b28', 'b21', 'b26', 'b50']

Section: Related Works
Ego-View Traffic Accident Detection (TAD). TAD aims to identify accidents within specific time frames and regions using two primary approaches: frame-level and object-level methods. Frame-level methods extract features from video frames and classify them to detect accidents (Vijay et al., 2022;Zhou et al., 2022). For example, You and Han (You & Han, 2020) developed a traffic dataset and utilized a 3D-CNN for accident localization. Reconstruction-based techniques (Chong & Tay, 2017;Zhao et al., 2017;Gong et al., 2019) identify anomalies by comparing reconstructed frames with actual ones. Due to the limited availability of real accident data, synthetic datasets and domain adaptation methods are often employed to enhance performance (Batanina et al., 2019;Tamagusko et al., 2022). Object-level methods focus on the consistency of object movements over time by using detectors and trackers to generate trajectories, thereby reducing the impact of dynamic backgrounds.
These methods analyze trajectories (Santhosh et al., 2021;Chakraborty et al., 2018) or evaluate the consistency of object positions (Le et al., 2020;Taccari et al., 2018;Hu et al., 2021a) to detect accidents. Yao et al. (Yao et al., 2022;2019) proposed an unsupervised approach that predicts future object positions, flagging significant deviations as potential accidents. Additionally, object-level strategies model interactions among objects to detect abrupt contextual changes that may indicate accidents (Fang et al., 2022;Yamamoto et al., 2022;Roy et al., 2020;Vijay et al., 2022). MOVAD (Rossi et al., 2024) achieves efficient online detection of traffic anomalies based solely on dashcam videos by combining a Video Swin Transformer and an LSTM module, and it is the first to introduce the concept of online traffic anomaly detection.
this section cite: ['b45', 'b56', 'b51', 'b7', 'b54', 'b16', 'b1', 'b41', 'b38', 'b4', 'b27', 'b40', 'b50', 'b10', 'b48', 'b37', 'b45', 'b36']

Section: Ego-View Traffic Accident Anticipation (TAA).
TAA focuses on predicting potential collisions by identifying unusual behaviors in advance, providing critical time for safe decision-making. It primarily involves predicting object trajectories to assess accident likelihood through detection, tracking, and prediction workflows (Haris et al., 2021;Thakur et al., 2024). In complex environments like highways, methods such as SVMs and HMMs analyze vehicle trajectories to forecast accidents (Xiong et al., 2017;Gutierrez-Osorio & Pedraza, 2020). To handle frequent obstructions in dashcam footage, recent approaches employ Dynamic Spatial Attention (DSA) (Chan et al., 2017) and RNNs to capture complex spatial and temporal relationships (Karim et al., 2022). Beyond trajectory prediction, TAA also involves identifying high-risk areas (Karim et al., 2023;Shimomura et al., 2024) and modeling driver attention (Chen et al., 2023). Risk localization techniques integrate agent representations with regional interactions to highlight areas with high accident probabilities (Zeng et al., 2017). The DRAMA dataset (Malla et al., 2023), which combines visual and textual data, enhances prediction accuracy by providing detailed descriptions of potential accident scenarios. Driver attention models utilize gaze direction to emphasize possible dangers, focusing on risky areas (Bao et al., 2021). Additionally, attention maps improve TAA interpretability by highlighting high-risk zones, supporting integrated approaches that combine trajectory prediction, risk localization, and driver attention modeling (Monjurul Karim et al., 2021).
this section cite: ['b19', 'b42', 'b47', 'b17', 'b5', 'b25', 'b26', 'b39', 'b6', 'b52', 'b33', 'b0', 'b34']

Section: Real-Time Anomaly Detection
Real-Time Anomaly Detection in autonomous driving is essential for ensuring safety by swiftly identifying and responding to unexpected objects and behaviors in the driving environment. This task demands a system that operates with minimal latency, capable of detecting dynamic changes such as pedestrians suddenly crossing the road or vehicles appearing abruptly. The primary challenge is to achieve high precision in anomaly recognition while maintaining response times at the millisecond level.
Problem Formulation. Let {X t } T t=1 denote a sequence of sensor observations, where X t ∈ R n represents the sensor data at time t. The objective is to detect anomalies in real-time, identifying unexpected objects or behaviors that may pose risks.
We define an anomaly indicator function A t as:
A t = I(X t is anomalous),(1)
where I(•) is the indicator function, returning 1 if the condition is true and 0 otherwise.
A detection model f (•) assigns an anomaly score s t = f (X t ). An anomaly is detected when the score exceeds a threshold θ:
Ât = I(s t > θ).(2)
Response Time. Response time R is a critical metric, comprising the detection delay and the model's inference time:
R = ∆T detection + T inference ,(3)
where ∆T detection = T detection -T occurrence is the delay between the anomaly occurrence and its detection. T inference is the time taken by the model to process the input and produce a result. The goal of Real-Time Anomaly Detection is to minimize R while ensuring high detection accuracy by:
• Minimizing Inference Time (T inference ): Developing efficient algorithms that process data rapidly to reduce computational delays.
• Reducing Detection Delay (∆T detection ): Enhancing the model's ability to promptly identify anomalies immediately after they occur.
this section cite: []

Section: Method
To achieve real-time anomaly detection with minimal inference time and detection delays, we propose a multimodal asynchronous hybrid network that integrates sparse event streams with RGB image data. Our framework processes RGB images using a ResNet to extract appearance features and captures event data through an asynchronous GNN with spline convolution. The image features are shared unidirectionally with the GNN, enabling the GNN to enhance event feature representation without reciprocal communication. This design significantly improves performance, especially in scenarios with sparse events, such as static or slow-moving conditions.
The features from both modalities are fused and passed to a detection head, which generates object-bounding boxes. These object-level features are further refined using a global graph that incorporates bounding box priors, enhancing spatial anomaly detection. To capture temporal dependencies, we employ a Gated Recurrent Unit (GRU) that processes the video sequences. An attention mechanism assigns higher weights to potentially anomalous objects, ensuring focused and efficient analysis. The combination of spatial, temporal, and attention-enhanced features enables accurate and swift anomaly detection.
In Sec. 4.1, we describe the network backbone, highlighting the unidirectional integration of image and event features and the role of asynchronous GNNs in feature representation. Sec. 4.2 elaborates on how spatial and temporal features are fused for anomaly detection. Figure 2 presents the overall architecture of our proposed framework.
this section cite: []

Section: Multimodal Asynchronous Hybrid Network
To achieve real-time anomaly detection, we propose a Multimodal Asynchronous Hybrid Network that efficiently integrates RGB images and event streams. Our network consists of two parallel branches: a CNN for processing image data and a GNN for handling event streams. This dual-branch architecture enables rapid and accurate feature extraction from both modalities. Image Feature Extraction. The CNN branch, denoted as F I , processes input images I ∈ R H×W ×3 to extract rich spatial features. Utilizing a ResNet architecture, F I generates detection outputs D I and intermediate feature maps G I = {g l I } L l=1 at various layers. These intermediate features are reused in the GNN branch to enhance computational efficiency.
this section cite: []

Section: Asynchronous Event Graph Construction.
Event streams E = {e i = (x i , t i , p i )}, where x i = (u i , y i ) denotes pixel coordinates, t i is the timestamp, and p i ∈ {-1, 1} represents polarity, are captured by event cameras when luminance changes exceed a threshold C:
|∆L| > C.(4)
These events are modeled as nodes in a graph G = (V, E) with normalized spatial coordinates xi = ui W , yi H and scaled timestamps ti = βt i . Edges are formed based on spatial and temporal proximity within a radius R, and edge features are defined as:
e ij = 1 2 (n j,xy -n i,xy ) + 1 2 ,(5)
where n i,xy and n j,xy are the normalized spatial coordinates of nodes i and j. Each node connects to up to 16 neighbors to maintain computational efficiency.
this section cite: []

Section: Event Feature Extraction.
We employ a Deep Asynchronous Graph Neural Network (DAGr) (Gehrig & Scaramuzza, 2024) to process the event graph using residual graph convolutional layers with spline convolutions:
f ′ i = W c f i + j∈N (i) W (e ij )f j ,(6)
where W c and W (e ij ) are learnable weights, and N (i) denotes the neighbors of node i. Spline convolutions enable efficient aggregation of neighbor information, accelerating computation through lookup tables during deployment. Temporal consistency is maintained by aggregating nodes into a voxel grid and applying directional voxel pooling, which preserves the temporal order of events.
this section cite: ['b14']

Section: Feature Fusion.
To integrate the extracted features from both modalities, we fuse the CNN and GNN outputs by augmenting each GNN node feature f i with the corresponding CNN feature g I (x i ) sampled at the node's location:
f ′ i = [f i , g I (x i )].(7)
This fusion enhances the model's ability to leverage spatial information from images alongside the temporal dynamics captured by event streams, improving the detection of anomalies in diverse driving scenarios.
The Multimodal Asynchronous Hybrid Network is optimized for real-time performance by utilizing asynchronous processing and efficient feature fusion. This design ensures minimal latency and high accuracy in detecting anomalies, making it well-suited for the stringent requirements of autonomous driving systems.
this section cite: []

Section: Anomaly Detection Network
To enable real-time anomaly detection, our Anomaly Detection Network efficiently extracts and processes object-level features from both event streams and RGB images, capturing spatial and temporal dynamics with minimal latency.
this section cite: []

Section: Object Feature Extraction.
We utilize an asynchronous GNN to extract features from event data overlapping with detected bounding boxes. For each object i at time t, the GNN generates an event-based feature o t,i :
o t,i = AsyncGNN (E t,i ; θ GNN ) ,(8)
where E t,i represents the event points within the bounding box of object i, and θ GNN are the GNN parameters.
Concurrently, features from the RGB image are extracted using a CNN, denoted as g t,i . We concatenate the GNN and CNN features to form a comprehensive feature vector:
p t,i = [o t,i ; g t,i ],(9)
which is then reduced in dimensionality via a fully connected layer:
f t,i = ϕ(p t,i ; θ 0 ).(10)
Spatio-Temporal Relational Learning. To model temporal dependencies and interactions between objects, we employ Gated Recurrent Units (GRUs). For each object i, the bounding box features b t,i and the fused features f t,i are processed as follows: Here, θ 1 and θ 2 are the parameters for the GRUs handling bounding box and fused features, respectively.
h b,t,i = GRU(b t,i , h b,t-1,i ; θ 1 ),(11)
h f,t,i = GRU(f t,i , h f,t-1,i ; θ 2 ). (12
)
Attention Mechanism. To prioritize significant objects, we apply an attention mechanism to the GRU outputs. The attention weights for bounding box and fused features are computed as:
α b,t = softmax tanh H ⊤ b,t w b ,(13)
Ĥb,t = H b,t α b,t ,(14)
α f,t = softmax tanh H ⊤ f,t w f , (15
) Ĥf,t = H f,t α f,t ,(16)
where H b,t and H f,t are the hidden states for bounding box and fused features, and w b , w f are learnable parameters.
this section cite: []

Section: Risk Score Prediction.
The attention-weighted features are concatenated to form a unified representation for each object:
ĥt,i = [ ĥb,t,i ; ĥf,t,i ],(17)
which is then passed through a fully connected layer and softmax activation to compute the riskiness score s t,i :
s t,i = softmax ϕ ĥt,i ; θ 3 .(18)
Here, θ 3 are the parameters of the final classification layer.
This network architecture ensures that anomalies are detected accurately and promptly by integrating spatial features from RGB images with temporal dynamics from event streams, all processed through efficient asynchronous and recurrent mechanisms tailored for real-time performance.
this section cite: []

Section: Experiments

this section cite: []

Section: Experimental Setup
Datasets. We employ two datasets, ROL (Karim et al., 2023) and DoTA (Yao et al., 2022), both annotated with detailed temporal, spatial, and categorical information, making them highly suitable for traffic anomaly detection research.
The ROL dataset provides comprehensive annotations for each video clip, which encompass object descriptions, accident details, and scene contexts. Temporal annotations pinpoint the initial appearance of a risky traffic agent and the onset of an accident, offering insights into the dynamics of risk development and collision occurrence. Spatially, the dataset includes bounding boxes for each traffic agent, which are initially detected using YOLOv5, subsequently tracked across frames with DeepSort (Veeramani et al., 2018), and finally refined by human annotators to ensure high accuracy. Categorical annotations in the dataset include traffic agent types and scene contexts, enhancing the dataset's utility for indepth traffic behavior analysis and research. DoTA stands as the first openly available dataset tailored for traffic video anomaly detection, featuring robust temporal, spatial, and categorical annotations. It comprises over 4,600 video clips, collected under a variety of regional, weather, and lighting conditions, with each video documenting one specific anomaly.
Temporal annotations in DoTA detail the start, duration, and conclusion of anomalies, while spatial annotations provide bounding boxes coupled with unique tracking IDs for each object involved in the anomalies. Currently, our DoTA dataset, ROL dataset, and all existing real-world (non-synthetic) first-person autonomous driving anomaly detection datasets lack the event modality. To address this, we utilized the v2e (Hu et al., 2021b) conversion technique to generate and supplement event modality data. This allows us to simulate the continuous event streams that would be captured by event cameras in real-world scenarios. It is important to note that v2e is solely used for generating supplementary event data and serves no other purpose in our work. Implementation Details. Our proposed multimodal anomaly detection model is implemented in PyTorch. For object detection, RGB frames are resized to 224×224 pixels and processed using a ResNet50 backbone for feature extraction. We adopt the YOLOX framework for bounding box detection, optimizing with IoU loss, class loss, and regression loss. For anomaly detection, features from the tracked objects are fed into a GRU network to capture temporal dependencies, complemented by an attention mechanism that focuses on potential anomalies. Asynchronous event data are handled using an asynchronous GNN layer, which models interactions at the frame level with bounding box priors aiding spatial-temporal analysis. The integration of spatial and temporal features from the ResNet backbone facilitates comprehensive anomaly detection.
To address cross-modality training schedule ambiguity, both modalities are trained with clearly defined parameters. The RGB component is trained for 30 epochs using a batch size of 64, and the dataset contains 1,920 images per epoch, resulting in a total of 57,600 data passes during training. For the event-based component, derived from the v2e conversion of the image-based dataset, we employ a batch size of 32 and a dataset size of 1,920 samples per epoch, with training spanning 150,000 iterations. This is equivalent to approximately 2,500 passes over the event dataset, ensuring comprehensive learning of the event-derived features.
Training employs the Adam optimizer for the GRU-attention module with a learning rate of 0.001, and the AdamW optimizer for the GNN-ResNet combination with a learning rate of 2 × 10 -4 . A ReduceLROnPlateau scheduler adjusts learning rates to optimize training stability. Class weights in the ROL dataset are set to 0.27 for the negative class and 1 for the positive class to balance the model's response. These detailed training schedules and parameters ensure effective integration of detection and attention mechanisms for both spatial and temporal anomaly identification.
this section cite: ['b26', 'b50', 'b44']

Section: Evaluation Metrics
To comprehensively evaluate our real-time anomaly detection model, we employ a set of metrics designed to measure both predictive accuracy and timeliness:
Area Under the Curve (AUC). AUC (Hanley & McNeil, 1982) quantifies how well the model distinguishes between risky and non-risky agents by computing the area under the ROC curve, thereby providing a single metric that balances true positive rate (TPR) and false positive rate (FPR).
this section cite: ['b18']

Section: Average Precision (AP).
AP (Everingham et al., 2010) calculates the area under the Precision-Recall curve, offering an intuitive measure for imbalanced datasets by emphasizing both precision and recall.
this section cite: ['b9']

Section: Mean Average Precision (mAP).
mAP (Lin et al., 2014) evaluates detection performance across varying IoU thresholds (from 0.5 to 0.95 in increments of 0.05), thus capturing a more nuanced perspective on overall detection robustness.
Mean Time-to-Accident (mTTA). mTTA (Fang et al., 2023) measures the average earliest time at which a risky agent's score s t,i surpasses a threshold s, reflecting the model's capability to foresee accidents before they occur.
Frame-Level AUC (AUC-Frame). AUC-Frame evaluates the model's ability to detect risky frames within a video. It is defined as the area under the ROC curve at the frame level:
AUC-Frame = 1 0 TPR(t) d(FPR(t)),(19)
where TPR(t) and FPR(t) denote the true positive rate and false positive rate at threshold t, respectively.
this section cite: ['b31', 'b11']

Section: Mean Response (mResponse).
To capture not just if but also how quickly anomalies are detected at various sensitivity levels, we introduce mResponse. Unlike a singlethreshold evaluation, mResponse measures the average detection delay across multiple thresholds, offering a more holistic view of real-time performance. Formally, it is defined as:
mResponse = 1 n n j=1 Response j , (20
)
where n is the number of thresholds and Response j is the detection delay at the j-th threshold. By aggregating response times across varying operational sensitivities, mResponse provides a more robust measure of how promptly the model flags anomalies, making it particularly suitable for real-world, safety-critical scenarios.
this section cite: []

Section: Result Analysis
We evaluated the effectiveness of our proposed model (OURS) against several established methods for anomaly detection on the ROL dataset. Performance was assessed using key metrics including Area Under the Curve (AUC), Average Precision (AP), Frame-Level AUC (AUC-Frame), and mean Time-to-Accident (mTTA). The compared models include ConvAE (Hasan et al., 2016), ConvLSTMAE (Chong & Tay, 2017), AnoPred (Liu et al., 2018), FOL (Yao et al., 2019) (with variants FOL-IoU, FOL-Mask, FOL-STD, and FOL-Ensemble (Yao et al., 2022)), MAMTCF (Liang et al., 2023), AM-Net (Karim et al., 2023), STFE (Zhou et al., 2022), and TTHF (Liang et al., 2024). As summarized in Table 1, OURS achieves leading performance in both AUC and AP metrics, demonstrating superior capability in distinguishing risky agents and balancing precision with recall. While the TTHF method outperforms OURS in the AUC-Frame metric, OURS significantly surpasses all other models, including TTHF, in terms of response time. This superior response time is attributed to OURS's reliance on asynchronous Graph Neural Networks (GNNs) and event stream integration, which enable an inference speed approaching 600 FPS. Additionally, OURS achieves an exceptionally low mean response time (mResponse), highlighting its promptness in detecting anomalies. This low latency is a result of the model's high inference speed and efficient anomaly detection mechanisms, as further evidenced by the favorable mTTA values. In contrast, the TTHF method, which integrates text information fusion, exhibits slower response times despite higher detection performance. Overall, the comparative analysis underscores that OURS not only sets a new benchmark in AUC and frame-level anomaly detection but also significantly enhances early risk localization capabilities. These results establish OURS as the leading model in real-time anomaly detection, combining high accuracy with exceptional timeliness, as detailed in Table 1.
this section cite: ['b20', 'b7', 'b32', 'b49', 'b50', 'b29', 'b26', 'b56', 'b30']

Section: Ablation Studies
We conducted ablation experiments to investigate the impact of various modules on model performance, focusing on metrics including AUC, AP, AUC-Frame, mTTA, and mAP. Table 2 provides a summary of the results on the ROL dataset.
this section cite: []

Section: RGB + Event.
Incorporating both RGB and event data significantly enhances the model's overall performance. RGB features offer rich visual information that aids in distinguishing between object categories such as vehicles and pedestrians, while also complementing event stream data to improve object detection accuracy, as reflected by increased mAP in diverse driving scenarios. Meanwhile, event features capture asynchronous and dynamic motion cues, effectively representing the relative movement between objects and the autonomous vehicle. This enables the model to rapidly and reliably detect anomalies, particularly in challenging conditions such as extreme lighting or at night, further strengthening the robustness of the system.
this section cite: []

Section: GRU Module for Temporal Dynamics.
The GRU module is critical for capturing and leveraging temporal information.
Integrating GRUs increases AUC from 0.805 to 0.817 and AP from 0.479 to 0.508, indicating a more accurate classification of anomalies. Moreover, mTTA improves from 1.44 to 1.98 seconds, demonstrating the GRU's effectiveness in accumulating temporal features and enabling early detection-an essential feature for real-time anomaly detection.
this section cite: []

Section: Attention Module.
Although secondary to GRUs in modeling temporal dependencies, the Attention module considerably boosts the model's sensitivity to anomalies by concentrating on salient regions. This targeted approach proves especially beneficial in complex or cluttered environments, where focusing on relevant features is critical for accurate anomaly detection.
this section cite: []

Section: BBox and Object Modules for Precise Localization.
The BBox module refines the model's localization capabilities, delivering more precise bounding box information and thereby improving AUC and AP metrics. Building on this, the Object module leverages these bounding boxes to further enhance detection accuracy by extracting detailed objectlevel features. This enhancement is particularly valuable in crowded or occluded scenes, where precise object recognition is challenging.
Together, the multimodal integration (RGB + Event), GRU, Attention, BBox, and Object modules comprehensively elevate the model's performance across all metrics. While RGB images increase mAP through richer spatial details, the GRU and Attention modules substantially enhance temporal detection accuracy and sensitivity. Simultaneously, BBox and Object modules refine localization and object recognition, culminating in a robust, high-performing anomaly detection framework.
Our multimodal asynchronous hybrid network is designed in a modular fashion, allowing the network depth (i.e., number of ResBlocks and look-up-table Spline convolution layers) to be increased for more complex data. Experiments show that increasing the number of layers slightly improves detection accuracy, with a modest increase in inference latency. See Table 3. These results indicate that our model is highly scalable, with only minor latency trade-offs for improved detection accuracy in increasingly complex environments.
To further enhance global feature modeling, we replaced the original CNN backbone with ViT and Swin Transformer.
Transformer architectures can capture long-range dependencies and improve detection accuracy. However, the selfattention mechanism has O(N 2 ) complexity, introducing additional inference latency. Experimental results are shown in Table 4. These results show that Transformer backbones can improve detection accuracy when latency is carefully managed, but a trade-off exists between accuracy and real-time requirements.
this section cite: []

Section: Qualitative Evaluation
Our model leverages multiple feature types: RGB features encode appearance, bounding box features represent object position and movement, event stream features capture rapid and abnormal motion, and object-level features describe local details. The object-level attention mechanism assigns scores to detected objects, highlighting those most relevant to anomaly detection.
As illustrated in Figure 4, when a vehicle suddenly cuts in, its attention score increases as it approaches, indicating its growing importance as a potential anomaly. Visualizations of these attention scores across frames demonstrate how the model dynamically focuses on critical objects, providing insight into which features are most vital for identifying anomalies. This enhanced interpretability aids in understanding the model's decision-making process and improves its credibility and deployment safety.
Figure 3 presents three challenging scenarios from the ROL dataset that highlight our model's effectiveness in detecting traffic anomalies critical for autonomous vehicle safety. These examples involve small objects and complex abnormal movements. Each example includes three rows. The top row shows Ground Truth, where normal objects appear in white and anomalies in red. The middle row displays our model's predictions, highlighting anomalies with scores above 0.5 in red. The bottom row compares frame-level anomaly scores to the Ground Truth, offering a straightforward visual measure of our model's accuracy.
Lane Merger Scenario. In Figure 3(a), a vehicle abruptly merges into the autonomous vehicle's lane. At frame 20, the anomaly score is 0.33, signifying the early onset of abnormal behavior. By frame 30, the score rises to 0.58, accurately highlighting the vehicle's unusual trajectory and intensified event flow. Here, the GRU module proves essential by aggregating temporal cues, boosting the detection sensitivity as the threat unfolds.   Oncoming Collision Risk. Figure 3(b) depicts an oncoming vehicle veering toward the autonomous vehicle, posing a high collision risk. The model's anomaly score quickly escalates, driven by erratic movement and close proximity, which are detected by leveraging bounding box data to assess relative positions. This underscores the importance of precise spatial cues for robust anomaly detection.
this section cite: []

Section: Early Detection Advantage.
Comparisons in Figure 3 indicate that our method achieves earlier anomaly detection than AM-Net (Karim et al., 2023), thereby reducing latency in critical situations. These examples confirm that our model not only covers a wide spectrum of anomalies but also reacts with temporal precision essential for real-world autonomous driving applications.
Inter-Frame Anomaly Detection. Figure 5 illustrates a sudden pedestrian appearance across two consecutive frames. Harnessing the asynchronous event stream properties allows the model to detect anomalies between frames, effectively anticipating the presence of fast-moving objects before they fully emerge in the scene. This capability significantly lowers detection latency, which can be crucial for avoiding potential collisions.
Figure 5. In scenarios where high-speed objects suddenly emerge, a continuous stream of events helps the model perform inter-frame anomaly detection, allowing for earlier and more timely anomaly detection.
this section cite: ['b26']

Section: Conclusion
We focus on the task of real-time anomaly detection in autonomous driving, underscoring the need to minimize response time without compromising detection accuracy.
To address this challenge, we proposed a multimodal asynchronous hybrid network that combines event streams from event cameras with RGB image data. By employing an asynchronous GNN for high-temporal-resolution event data and a CNN for rich spatial features, our framework captures both the temporal dynamics and spatial details of driving environments. Extensive experiments on benchmark datasets demonstrate that our approach significantly surpasses existing methods in detection accuracy and response time, achieving millisecond-level responsiveness. This work lays a foundation for future research in time-critical perception tasks and advances safe, reliable deployment of autonomous vehicles.
this section cite: []

Section: References
Ref_id:b0 Title: Drive: Deep reinforced accident anticipation with visual explanation Year: (2021)
Ref_id:b1 Title: Domain adaptation for car accident detection in videos Year: (2019)
Ref_id:b2 Title: Efficientad: Accurate visual anomaly detection at millisecond-level latencies Year: (2024)
Ref_id:b3 Title: nuscenes: A multimodal dataset for autonomous driving Year: (2020)
Ref_id:b4 Title: Freeway traf-fic incident detection from cameras: A semi-supervised learning approach Year: (2018)
Ref_id:b5 Title: Anticipating accidents in dashcam videos Year: (2016)
Ref_id:b6 Title: Fblnet: Feedback loop network for driver attention prediction Year: (2023)
Ref_id:b7 Title: Abnormal event detection in videos using spatiotemporal autoencoder Year: (2017-06-21)
Ref_id:b8 Title: Gorela: Go relative for viewpoint-invariant motion forecasting Year: (2023)
Ref_id:b9 Title: The pascal visual object classes (voc) challenge Year: (2010)
Ref_id:b10 Title: Traffic accident detection via self-supervised consistency learning in driving scenarios Year: (2022)
Ref_id:b11 Title: Vision-based traffic accident detection and anticipation: A survey Year: (2023)
Ref_id:b12 Title: Abductive ego-view accident video understanding for safe driving perception Year: (2024)
Ref_id:b13 Title: Event-based vision: A survey Year: (2020)
Ref_id:b14 Title: Low-latency automotive vision with event cameras Year: (2024)
Ref_id:b15 Title: Dsec: A stereo event camera dataset for driving scenarios Year: (2021)
Ref_id:b16 Title: Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection Year: (2019)
Ref_id:b17 Title: Modern data sources and techniques for analysis and forecast of road accidents: A review Year: (2020)
Ref_id:b18 Title: The meaning and use of the area under a receiver operating characteristic (roc) curve Year: (1982)
Ref_id:b19 Title: Vehicle crash prediction using vision Year: (2021)
Ref_id:b20 Title: Learning temporal regularity in video sequences Year: (2016)
Ref_id:b21 Title: Deep residual learning for image recognition Year: (2016)
Ref_id:b22 Title: Cost-sensitive semi-supervised deep learning to assess driving risk by application of naturalistic vehicle trajectories Year: ()
Ref_id:b23 Title: v2e: From video frames to realistic dvs events Year: (2021)
Ref_id:b24 Title: The apolloscape dataset for autonomous driving Year: (2018)
Ref_id:b25 Title: A dynamic spatial-temporal attention network for early anticipation of traffic accidents Year: (2022)
Ref_id:b26 Title: An attention-guided multistream feature fusion network for early localization of risky traffic agents in driving videos Year: (2023)
Ref_id:b27 Title: Attention r-cnn for accident detection Year: (2020)
Ref_id:b28 Title: Graph-based asynchronous event processing for rapid object recognition Year: (2021)
Ref_id:b29 Title: A memoryaugmented multi-task collaborative framework for unsupervised traffic accident detection in driving videos Year: (2023)
Ref_id:b30 Title: Text-driven traffic anomaly detection with temporal high-frequency modeling in driving videos Year: (2024)
Ref_id:b31 Title: Microsoft coco: Common objects in context Year: (2014)
Ref_id:b32 Title: Future frame prediction for anomaly detection-a new baseline Year: (2018)
Ref_id:b33 Title: Joint risk localization and captioning in driving Year: (2023)
Ref_id:b34 Title: Towards explainable artificial intelligence (xai) for early anticipation of traffic accidents Year: (2021)
Ref_id:b35 Title: Self-distilled masked autoencoders are efficient video anomaly detectors Year: (2024)
Ref_id:b36 Title: Memory-augmented online video anomaly detection Year: (2024)
Ref_id:b37 Title: Detection of collision-prone vehicle behavior at intersections using siamese interaction lstm Year: (2020)
Ref_id:b38 Title: Vehicular trajectory classification and traffic anomaly detection in videos using a hybrid cnn-vae architecture Year: (2021)
Ref_id:b39 Title: Potential risk localization via weak labeling out of blind spot Year: (2024)
Ref_id:b40 Title: Classification of crash and near-crash events from dashcam videos and telematics Year: (2018)
Ref_id:b41 Title: Deep learning applied to road accident detection with transfer learning and synthetic images Year: (2022)
Ref_id:b42 Title: Graph (graph): A nested graph-based framework for early accident anticipation Year: (2024)
Ref_id:b43 Title: Latency-aware road anomaly segmentation in videos: A photorealistic dataset and new metrics Year: (2024)
Ref_id:b44 Title: Deepsort: deep convolutional networks for sorting haploid maize seeds Year: (2018)
Ref_id:b45 Title: Detection of road accidents using synthetically generated multi-perspective accident videos Year: (2022)
Ref_id:b46 Title: Prophnet: Efficient agent-centric motion forecasting with anchor-informed proposals Year: (2023)
Ref_id:b47 Title: A new framework of vehicle collision prediction by combining svm and hmm Year: (2017)
Ref_id:b48 Title: Classifying near-miss traffic incidents through video, sensor, and object features Year: (2022)
Ref_id:b49 Title: Unsupervised traffic accident detection in firstperson videos Year: (2019)
Ref_id:b50 Title: Unsupervised detection of traffic anomaly in driving videos Year: (2022)
Ref_id:b51 Title: Traffic accident benchmark for causality recognition Year: (2020)
Ref_id:b52 Title: Agent-centric risk assessment: Accident anticipation and risky region localization Year: (2017)
Ref_id:b53 Title: Systems and methods for actor motion forecasting within a surrounding environment of an autonomous vehicle Year: (2023-11-02)
Ref_id:b54 Title: Spatio-temporal autoencoder for video anomaly detection Year: (2017)
Ref_id:b55 Title: Deep event-based object detection in autonomous driving: A survey Year: (2024)
Ref_id:b56 Title: Spatio-temporal feature encoding for traffic accident detection in vanet environment Year: (2022)
