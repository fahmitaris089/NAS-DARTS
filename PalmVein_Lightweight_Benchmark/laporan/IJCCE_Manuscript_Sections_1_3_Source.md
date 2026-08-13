# Hardware-Aware Progressive Differentiable Architecture Search for Compact Palm-Vein Identification on Raspberry Pi

> Working manuscript source: Sections 1-3 only. The author block, abstract, keywords, results, discussion, conclusion, CRediT statement, declarations, and data-availability statement will be added only after authorship and the final experiment ledger are frozen. Numeric citations are an internal drafting convention and must be converted to IJCCE's APA author-year style before submission.

# 1. Introduction

Palm-vein recognition represents an identity through vascular patterns captured beneath the skin, usually with near-infrared illumination. The modality is attractive for contactless recognition because its discriminative structure is internal, difficult to observe under visible light, and obtainable without placing the palm on a shared surface. These properties have supported its use in access control and other identity-management settings. They do not, however, make the recognition problem trivial. Contactless acquisition introduces changes in hand position, distance, rotation, illumination, and visible palm area, while the vessels themselves often appear as low-contrast, fine-grained structures. A useful model must retain this local information while tolerating acquisition variation [1], [2].

Recent palm-vein systems increasingly rely on convolutional neural networks rather than fixed texture, line, or transform-domain descriptors. Learned representations can adapt to the appearance of the available sensor and preprocessing pipeline, and multi-scale designs can combine narrow vessel details with broader palm context [2], [3], [4]. The computational cost of the resulting model matters when recognition is expected to run locally. Sending biometric images to a remote service adds network dependence and expands the system boundary within which sensitive data must be handled. Local inference avoids some of that dependence, but an edge device imposes limits on memory, CPU throughput, and response time; surveys of lightweight learning and edge-assisted deep learning describe the same resource constraint [5], [6]. Edge-oriented biometric work in IJCCE similarly treats the recognizer as one constrained component of a local intelligent system [7]. A model that is accurate on a workstation is therefore not automatically suitable for a Raspberry Pi.

Compact palm-vein networks have been developed through efficient convolution, knowledge distillation (KD), pruning, and low-precision inference. Chen et al. designed a depthwise-separable student and trained it with a larger teacher [8], while another lightweight convolutional design addressed contactless multispectral palm-vein recognition [9]. Ding et al. combined neural architecture search (NAS) with pruning and INT8 compression [10]. These studies establish that accuracy and compactness can be considered jointly within the palm-vein domain. Their reported values nevertheless cannot be pooled as if they came from one experiment. The studies differ in database, class definition, image modality, preprocessing, train-test construction, and recognition task. A cross-paper table is useful for positioning; it is not a controlled ranking of architectures.

The distinction between identification and verification is especially important. Standard biometric terminology separates one-to-many identification from one-to-one verification, while biometric performance standards require the task, trial construction, and reported measures to be defined consistently [11], [12]. This work studies closed-set identification: each query is assigned to one of the palm identities represented during training, and performance is measured by top-1 identification accuracy, also described as the correct recognition rate (CRR). Verification instead evaluates whether a claimed or paired identity matches, using genuine and impostor similarity scores and a decision threshold. Equal error rate, false acceptance rate, and false rejection rate are meaningful only after that score protocol has been defined. Their use in a multi-class softmax classifier with one test image per class would not create a valid verification experiment. Recent biometric work in IJCCE illustrates the required similarity-based framing when reporting verification errors [13]. Accordingly, this study does not report EER, FAR, FRR, ROC, or AUC.

Model compactness is also multidimensional. Parameter count measures stored coefficients; multiply-accumulate operations or floating-point operations approximate arithmetic demand; an exported file measures storage; and latency measures the behavior of a particular graph on a particular runtime and processor. These quantities are related but not interchangeable. An operator with few arithmetic operations may still incur memory movement, graph-boundary, or kernel-launch overhead. Conversely, an apparently larger operator may map efficiently to an optimized kernel. Hardware-aware NAS addresses this mismatch by using cost information from the deployment target while selecting an architecture [14]. ProxylessNAS, MnasNet, and FBNet demonstrated complementary ways to include platform feedback in mobile network design [15], [16], [17]. Their target platforms and visual tasks differ from the present setting, so their latency preferences cannot simply be assumed to hold for near-infrared palm images on a Raspberry Pi.

This study integrates target-device measurements with progressive differentiable architecture search. The search begins from a cell-based supernetwork in which candidate operators compete through continuous architecture weights. P-DARTS progressively increases supernetwork depth and reduces the operator set, narrowing the depth gap between search and evaluation [18], [19]. Here, operator costs are obtained from INT8 ONNX probes executed on the target Raspberry Pi configuration and stored in a latency lookup table (LUT). The normalized LUT cost is included in the architecture objective alongside classification loss. The LUT guides the search but does not replace end-to-end benchmarking, because a sum of isolated operator costs cannot reproduce graph fusion, memory reuse, or scheduling in the exported network.

The experimental design separates four sources of performance change. First, architecture quality is evaluated by training the searched genotype and comparison models from random initialization with the same optimizer, augmentation, split, epoch budget, checkpoint rule, and seeds. Second, official ImageNet weights are evaluated only for architectures with auditable pretrained releases, and the transfer results are kept separate from the scratch comparison. Third, KD is treated as a training intervention: the same frozen teacher and loss settings must be applied to the proposed student and selected lightweight baselines before a KD-specific claim is made. Fourth, static post-training quantization (PTQ) is evaluated after ONNX export, with calibration restricted to training images and accuracy and latency measured again for the INT8 graph.

The study addresses four research questions. RQ1 asks how increasing the target-device latency penalty changes the operators and cell topology selected by P-DARTS. RQ2 asks whether the fixed searched architecture provides a favorable accuracy-efficiency trade-off against general hardware-aware NAS models and palm-vein-specific comparators under controlled scratch training. RQ3 asks how official pretrained initialization and protocol-matched KD alter recognition performance when their effects are reported separately from architecture quality. RQ4 asks how static INT8 PTQ changes accuracy, file size, and Raspberry Pi latency relative to the corresponding FP32 ONNX model.

The contributions are intentionally bounded. First, the work supplies a palm-vein P-DARTS formulation whose resource signal comes from operator measurements on the intended Raspberry Pi runtime rather than from parameter count or FLOPs alone. Second, it defines a layered benchmark that prevents scratch training, ImageNet transfer, and KD from being combined into one ambiguous ranking. Third, it specifies an end-to-end PyTorch-to-ONNX-to-INT8 evaluation with fixed calibration and runtime controls. Fourth, it maintains implementation provenance, distinguishing official models, audited adaptations, and paper-constrained independent reconstructions. The study does not claim universal biometric superiority: its primary evidence concerns closed-set identification on the available SCUT_PV_v1 subset and deployment under the reported Raspberry Pi and ONNX Runtime configuration.

# 2. Related Work

## 2.1. Deep Learning for Palm-Vein Identification

A palm-vein pipeline normally combines acquisition, localization of the palm region, photometric normalization, representation learning, and identity inference. Earlier systems often encoded vessel lines, local texture, orientation, or frequency information before matching. Deep models shift much of the representation design into the learning process, but they do not eliminate the influence of acquisition and preprocessing. A network trained on tightly centered, contrast-enhanced regions solves a different statistical problem from a network trained on unregistered raw captures. Reviews of palm-vein recognition consequently treat region-of-interest (ROI) extraction and normalization as integral parts of the recognition method rather than neutral preparation [1].

Deep learning has been used both as an end-to-end classifier and as one component within a hybrid representation. Qin et al. combined generative modeling with a convolutional identification system to address multi-scale and multi-direction variation [3]. Wulandari et al. coupled wavelet and histogram-of-oriented-gradient features with convolutional recognition [4]. Other domain studies have examined deep hashing for palmprint-palmvein fusion [20], conventional CNNs for palm-vein recognition [21], and classic CNN behavior across palmprint and palm-vein data [22]. These approaches show that vessel evidence may be represented at several spatial scales and through different inductive biases. They are informative domain references, but their published recognition rates are not direct baselines for the present experiment unless they are retrained on the same classes and split.

SCUT_PV_v1 was introduced for unconstrained and weak-cooperative palm-vein recognition, and Luo et al. proposed AMPVNet to aggregate multi-scale features under those conditions [2]. This source is central for two reasons. It documents the dataset family used here, and it provides a domain-specific architecture whose design objective is closer to palm-vein variation than that of a generic mobile classifier. The current research subset contains 834 palm identities and is smaller than the complete class set described in the source publication. The subset was supplied by the data owner; the investigators did not choose which identities to exclude. Because the reason the remaining identities were unavailable has not been formally documented, the subset is described without attributing a cause.

Terminology and metrics must follow the actual decision problem. In closed-set identification, the output space is the enrolled class set and the top-scoring logit determines the predicted identity. Verification compares a query with an enrollment template or claimed identity and requires genuine and impostor trials. A system can support both modes, but one evaluation cannot be relabeled as the other. Tanna et al. reported verification-oriented error measures after defining similarity-based iris comparison [13]; that protocol is structurally different from the 834-way classifier used here. The present study therefore reports top-1/CRR, correct and incorrect predictions, and variation across training seeds. Macro precision, recall, and F1 are omitted because the test split contains exactly one example from every class; under this balanced single-example construction they add little independent information beyond accuracy.

Dataset construction also limits interpretation. Ten images are available per identity in the research subset. Eight are assigned to training, one to validation, and one to testing. The manifest prevents the same filename from appearing in two partitions, but it contains neither acquisition-session identifiers nor capture timestamps. It therefore supports a stratified image-level split, not a demonstrated session-disjoint test. High identification accuracy under this split should not be interpreted as evidence of cross-session, cross-sensor, or open-set generalization. Those claims require metadata-aware partitions or a separate dataset.

## 2.2. Domain-Specific Lightweight Palm-Vein Models

Lightweight palm-vein research has taken two broad routes: design a compact network directly or compress a stronger model after training. Chen et al. used depthwise-separable convolution in a MobileNet-inspired StudentNet and transferred information from Inception-v3 through KD [8]. The paper reports a compact model and strong correct identification rate on CASIA, but the architecture description contains ambiguities, including a parameter total that cannot be recovered from the published diagram under standard depthwise-pointwise definitions. The local Chen implementation is therefore retained only as an unregistered audit artifact and is not included in the controlled benchmark.

Chen, Hsia, and Chen proposed a lightweight convolutional network for contactless multispectral palm-vein recognition [9]. Its design is domain-specific and compact, but the multispectral input and evaluation setting differ from the single-stream preprocessed images used in this study. The work is consequently included in the literature-positioning table, not transplanted as a numerical result. This distinction prevents modality differences from being misread as architecture effects.

Luo et al.'s AMPVNet is another important domain comparator because it was created with SCUT_PV_v1 and targets weakly cooperative acquisition [2]. Its multi-scale aggregation addresses the variation for which the dataset was designed. For a controlled table, however, AMPVNet must be instantiated with the present 834-class output and trained using the common scratch protocol. A value copied from the paper would remain contextual because the available class subset and evaluation construction are not identical.

Ding et al. combined greedy NAS with pruning and integer compression, producing baseline, pointwise-converted, and pruned network variants [10]. Their comparison table includes palm-vein methods based on manually designed CNNs, a GAN-assisted CNN, and PalmNet; it is not restricted to NAS-generated models. A separate PalmNet study combines ShuffleNetV2, MobileNetV3, and MBConv blocks in a configurable lightweight palm-vein architecture [23]. The paper does not publish complete layer dimensions or official code in the audited materials. The local PalmNet and Ding variants are therefore paper-constrained independent reconstructions whose reported and reconstructed details are kept separate. This status supports controlled comparison under one protocol but does not establish equivalence to unavailable author implementations.

Table 1 separates scientific role from implementation status. Palm-vein studies show relevance to the sensing domain; general hardware-aware NAS models test whether a palm-specific search offers an advantage over established mobile designs. ProxylessNAS-Mobile, FBNet-C, and MnasNet-A1 were not originally created for palm veins, but this does not make them irrelevant. Under a common 834-class scratch protocol, they provide strong controls for the claim that the proposed search procedure discovers a useful edge architecture rather than merely a small network. Conversely, a comparison using only general mobile models would omit the strongest domain-specific question. Both groups are needed, but they answer different questions and are reported with their provenance.

**Table 1. Positioning of representative palm-vein and hardware-aware lightweight recognition studies.**

| Study | Domain/data context | Efficiency strategy | Hardware-aware search | Role in this study |
|---|---|---|---|---|
| Qin et al. [3] | Single palm-vein identification | Multi-scale/multi-direction GAN and CNN | No | Contextual domain evidence only |
| Chen et al. [8] | CASIA palm vein | Depthwise-separable StudentNet and KD | No | Contextual evidence; local artifact excluded because the architecture cannot be audited completely |
| Chen et al. [9] | Contactless multispectral palm vein | Manually designed lightweight CNN | No | Contextual evidence because modality differs |
| Luo et al. [2] | SCUT_PV_v1, weakly cooperative capture | AMPVNet multi-scale aggregation | No | Priority domain comparator when retrained on the fixed split |
| Ding et al. [10] | PolyU palm vein | Greedy NAS, pruning, and INT8 compression | Hardware-friendly objective; target-device evidence must be interpreted from the paper | Three paper-constrained reconstruction variants |
| Luo and Huang [23] | Palm-vein identification | PalmNet with ShuffleNetV2, MobileNetV3, and MBConv stages | No | Two paper-constrained reconstruction variants under the common scratch protocol |
| ProxylessNAS [15] | General image classification | Direct target-task/hardware NAS | Yes | General NAS-mobile scratch and pretrained control |
| MnasNet [16] | General image classification | Platform-aware multi-objective NAS | Yes | General NAS-mobile scratch and pretrained control |
| FBNet [17] | General image classification | Differentiable NAS with latency LUT | Yes | General NAS-mobile scratch and pretrained control |
| This study | Closed-set SCUT_PV_v1 subset | Raspberry Pi LUT, P-DARTS, KD, and PTQ | Yes, Raspberry Pi-specific | Controlled and deployment evidence |

*Table note.* Published results from different rows remain contextual unless the architecture is retrained on the same 834-class manifest and protocol. Reconstruction status is reported with every local result.

## 2.3. Hardware-Aware Neural Architecture Search

NAS treats architecture design as an optimization problem over candidate operations and connections [24]. Early approaches often required training many sampled networks. DARTS reduced this burden by replacing a discrete operation choice with a continuous mixture: each candidate operator on an edge receives a softmax weight, allowing network and architecture parameters to be optimized by gradient methods [18]. The relaxation makes search tractable, but the shared supernetwork remains an approximation; analyses of differentiable NAS have documented instability and poor discretization behavior under some search conditions [25]. The operator that performs well within a mixture may not retain the same advantage after discretization, and a shallow search network can favor cells that transfer poorly to a deeper evaluation network.

P-DARTS responds to the depth mismatch by increasing supernetwork depth across search stages while progressively shrinking the active operator set [19]. The procedure also regularizes skip connections, which can otherwise dominate because they optimize more easily than convolutional paths. This study adopts the progressive principle but changes the candidate set and objective for palm-vein deployment. The search space includes separable, dilated, mobile inverted bottleneck, re-parameterizable convolution, pooling, identity, and null operations. Mobile inverted bottlenecks follow the depthwise and linear-bottleneck rationale established by MobileNetV2 [26], whereas the re-parameterizable operator family follows the training-time multi-branch and inference-time single-path principle exemplified by RepVGG [27]. The resource term comes from Raspberry Pi operator profiling rather than from the original P-DARTS image-benchmark setting.

Hardware-aware NAS methods demonstrate why measured cost is preferable to a single theoretical proxy. MnasNet uses a platform-aware objective that balances accuracy and measured latency [16]. ProxylessNAS removes the need for a proxy architecture and can search directly with target-hardware feedback [15]. FBNet trains a stochastic supernetwork and uses a device latency LUT within differentiable optimization [17]. These works support the methodological choice to profile operators, but they do not establish which operator is fastest on every platform. Their original mobile devices, software stacks, input resolutions, and tasks differ from a Raspberry Pi executing ONNX graphs.

An operator LUT offers a differentiable and inexpensive resource signal, but it has limitations. Its entries depend on tensor shape, numerical representation, execution provider, threading, graph optimization, and the probe boundary. Isolated QDQ probes can include conversion overhead that would be fused or amortized inside a complete model. A correction applied to that overhead must be documented rather than treated as raw latency. Moreover, expected LUT cost during search is not end-to-end latency: it cannot fully represent concatenation, memory allocation, scheduling, or graph fusion. The present design therefore uses the LUT to bias selection and reserves the deployment claim for measurements of the final exported models.

## 2.4. Knowledge Distillation and Integer Quantization

KD trains a compact student from both the ground-truth label and information supplied by a teacher. In Hinton-style logit distillation, temperature softens the teacher and student probability distributions so that the loss reflects relative confidence across non-target classes [28]. Surveys show that the effect depends on teacher quality, student capacity, temperature, weighting, and the form of transferred knowledge [29]. KD is consequently an optimization treatment, not an intrinsic property of the student's topology. A distilled proposed model compared only with non-distilled baselines cannot isolate the contribution of NAS.

The controlled KD layer in this study uses one frozen teacher checkpoint and the same temperature, hard-label weight, data manifest, augmentation, training budget, and checkpoint rule for every participating student. P-DARTS is compared with at least two lightweight models selected by validation performance in the non-distilled experiment. The legacy thesis configuration used EfficientNetV2M [30] with temperature 20 and a hard-label weight of 0.5, but archived metadata also lists test accuracy among the teacher-selection criteria. Those legacy results are exploratory. A journal-level controlled KD claim requires the teacher to be fixed using validation evidence alone, or fixed independently of the current test split, before the compared students are trained.

Quantization addresses a different stage. Integer-arithmetic inference represents weights and activations at lower precision, reducing storage and enabling optimized integer kernels [31]. Static PTQ estimates activation ranges from representative calibration samples after model training; it does not update the network weights. The calibration distribution and quantizer configuration therefore affect the result [32]. Quantization may reduce file size without reducing latency when conversion nodes, unsupported operators, or inefficient kernels dominate. Accuracy may also improve slightly or decline because a one-image-per-class test has coarse resolution. Both directions must be reported as measured observations rather than assumed behavior.

The deployment stack uses ONNX Runtime because it provides a common graph format and CPU execution path for all compared models. Reproducibility requires more than naming the runtime: opset, graph validity, output parity, quantization format, signedness, per-channel settings, calibration membership, thread counts, warm-up, timed iterations, and software versions must be recorded. The LUT and complete-model quantization recipes must also be aligned. If the LUT uses signed activations while deployed models use unsigned activations, the architecture search is still informed by a device measurement, but it cannot be described as optimizing the identical deployment kernel mix until the mismatch is resolved.

## 2.5. Research Gap and Study Positioning

Prior research covers the components required for compact palm-vein inference, but often in separate settings. Domain-specific networks address vessel representation; NAS automates architecture choice; KD improves a chosen student; and PTQ reduces the numerical footprint. The methodological gap is a controlled chain that connects device-measured search to a fixed palm-vein identification protocol while preserving attribution. Without that separation, higher accuracy may be caused by pretrained weights or KD, and lower latency may be asserted from FLOPs rather than measured on the target device.

This study is positioned around attribution and deployment validity rather than a claim that no related component has previously been used. The scratch experiment asks about topology under equal training. The pretrained experiment asks about practical transfer for models with official weights. The KD experiment asks whether the same teacher benefits the proposed student more than strong alternatives. The PTQ and Raspberry Pi experiments ask whether the exported graph retains recognition quality and delivers a favorable size-latency trade-off. Literature values remain in a contextual table; only models trained and exported through the fixed local protocol enter the controlled result tables.

The scope also defines what the manuscript does not establish. One image-level split from one available subset cannot prove cross-session or cross-dataset generalization. Three seeds characterize optimization variability but do not justify strong distributional significance claims. Paper-constrained reconstructions cannot validate the original authors' topology. A Raspberry Pi result does not generalize to every edge processor. These boundaries allow the central claim to remain testable: within the recorded dataset, runtime, and comparison protocols, does the searched architecture occupy a useful Pareto position in identification accuracy, model size, and latency?

# 3. Materials and Methods

## 3.1. Study Design and Claim Boundaries

The study has six linked stages: data preparation, target-device operator profiling, hardware-aware architecture search, controlled retraining, optional KD, and ONNX deployment with PTQ. Figure 1 shows the intended separation of data roles. Training images update ordinary network weights and provide the calibration subset. During NAS, the original training partition is divided again into equal search-weight and search-architecture subsets. The fixed validation partition supports model monitoring, checkpoint selection, and protocol decisions. Test images are excluded from weight updates and quantizer calibration.

The decision task is 834-way closed-set identification. A model receives a preprocessed palm image and returns one logit for each identity. The predicted identity is the maximum-logit class. No enrollment template, pairwise similarity score, acceptance threshold, or unknown-identity rejection rule is defined; the method is therefore not evaluated as authentication or verification. This boundary applies to the title, methods, result tables, and conclusions.

Two evidence layers are maintained. The controlled layer contains local runs that share the fixed split, input representation, and predefined training protocol. The contextual layer summarizes published palm-vein studies with their original datasets and protocols. Contextual values can motivate design choices, but they are never merged into a local rank or statistical comparison. Likewise, official implementations and paper-constrained reconstructions are labeled separately.

The historical exposure of the test split is recorded. The same manifest was used during thesis development, and archived configurations show that test accuracy contributed to teacher selection and that λ and capacity candidates were compared after test evaluation. The split is therefore not an untouched confirmatory test in the strict sense. In each new benchmark run, the software selects one checkpoint by minimum validation loss and evaluates the test loader once after training; this operational rule prevents further per-run tuning but does not erase prior human exposure. Results on this manifest are described as a fixed-split retrospective controlled benchmark. A newly sealed test partition, session-disjoint split, or external dataset is required for a stronger confirmatory generalization claim.

NAS is performed only on the SCUT_PV_v1 research subset. If PolyU is later added, the frozen P-DARTS genotype, a domain-specific comparator, and one strong lightweight baseline can be retrained there without repeating search. Such an experiment would test architecture transfer, not cross-dataset search consistency. It is outside the present Sections 1-3 protocol until its manifest is frozen.

**Figure 1.** Study workflow and data roles for hardware-aware architecture search, controlled training, knowledge distillation, quantization, and Raspberry Pi evaluation.

## 3.2. Dataset Provenance and Data Partitioning

The experiments use 8,340 preprocessed near-infrared images from 834 palm identities in SCUT_PV_v1. Ten images are available for every identity. The subset was supplied directly by the dataset owner for this research. It contains fewer identities than the full dataset described by Luo et al. [2], but no documented reason for the unavailable identities was found in the supplied material. The manuscript therefore reports the provenance and size without speculating about access restrictions or sample quality.

The manifest assigns eight images per identity to training, one to validation, and one to testing, producing 6,672, 834, and 834 images, respectively. All partitions contain all 834 class labels. The SHA-256 hash of the manifest used by the benchmark is `8e393a52fbc93c19d420c942adf104b1910c708e796fcdb13e17ac90482966de`. Every architecture and seed uses this same file. Seeds 42, 123, and 2026 affect parameter initialization, data order, worker random state, and augmentation; they do not resample the partition.

The dataset validator checks the expected counts, 8,340 unique `(identity, filename)` pairs, absence of exact file-level overlap, correspondence between each filename prefix and its identity directory, and existence of every image. These checks establish manifest integrity but not independence of acquisition sessions. The supplied filenames encode identity and image order only. No session identifier, timestamp, or capture-distance label is present, so near-duplicate and session-overlap risk cannot be ruled out from the manifest. The resulting protocol is explicitly identified as a stratified image-level split.

For architecture search, the 6,672 training images are shuffled with seed 42 and divided equally. The first 3,336 samples update ordinary network weights, and the remaining 3,336 update architecture parameters. The original 834-image validation split monitors the supernetwork but does not provide the alternating architecture batch. The 834-image test split is constructed but not consumed by the search loop. This nested separation avoids using the fixed test data for gradient updates, while the historical-exposure limitation stated in Section 3.1 remains.

Static INT8 calibration uses exactly one training image per identity, yielding 834 samples. The manifest selects the lexicographically first training filename within each numeric identity and stores a SHA-256 hash for every file. Its own SHA-256 is `4588628c122b833f6fa049148564571581875a4f06c43d5edb7dae2a53f443d0`. Validation and test images are rejected by the manifest validator. Using the same calibration membership for every model removes calibration-set variation from the deployment comparison.

**Table 2. Dataset composition, provenance, and partition roles.**

| Partition | Images/identity | Images | Classes | Permitted role |
|---|---:|---:|---:|---|
| Search-weight subset | 4 on average after seeded 50/50 division | 3,336 | 834 | Supernetwork weight updates |
| Search-architecture subset | 4 on average after seeded 50/50 division | 3,336 | 834 | Architecture-parameter updates |
| Full training | 8 | 6,672 | 834 | Controlled training, augmentation, and calibration source |
| Validation | 1 | 834 | 834 | Monitoring, checkpoint selection, and protocol decisions |
| Legacy test | 1 | 834 | 834 | Post-checkpoint descriptive evaluation; previously observed during thesis work |
| Calibration manifest | 1 training image | 834 | 834 | Static PTQ range estimation only |

*Table note.* The internal 50/50 search division is sample-level and seeded; exact images per identity in each half may vary. The outer 8/1/1 manifest is fixed and has no exact filename overlap.

## 3.3. ROI Extraction and Image Preprocessing

Raw images were transformed offline before the controlled benchmark. The archived preprocessing implementation reads each image as grayscale, blurs it with a 7 x 7 Gaussian kernel, and uses Otsu thresholding [33] to separate the bright palm from the darker background. Morphological closing with a 15 x 15 elliptical kernel for three iterations fills small gaps, followed by two opening iterations to suppress isolated regions. The largest contour provides a rough palm centroid. If no contour is found or its spatial moment is zero, the image center is used as a documented fallback.

An intensity-weighted centroid refines the rough center within a 360 x 360 neighborhood. Pixel intensities contribute only where the palm mask is active, shifting the crop center toward the bright palm region. A square 384 x 384 ROI is placed around the refined center and clamped to image boundaries. If an input is smaller than the requested crop, the implementation pads the unavailable region with zeros. Unreadable source images raise an error rather than being silently omitted.

The cropped ROI is enhanced with contrast-limited adaptive histogram equalization (CLAHE) [34] using a clip limit of 2.0 and an 8 x 8 tile grid. Intensities are then min-max normalized to the 8-bit range and resized to 224 x 224 using Lanczos interpolation. The benchmark consumes these stored outputs; it does not run ROI extraction during model training. Reapplying a different ROI function to the stored images would create a new experimental condition and is prohibited within the controlled table.

The data loader opens each stored image as one grayscale channel, converts it to a tensor, and replicates the channel three times. Replication preserves compatibility with audited ImageNet backbones without changing their stems. Every protocol uses ImageNet normalization with means `(0.485, 0.456, 0.406)` and standard deviations `(0.229, 0.224, 0.225)`. Scratch models receive the same normalization as pretrained models so that input scaling is not confounded with initialization.

Training augmentation is deliberately mild: random rotation within 5 degrees, translation up to 3% in each direction, isotropic scale sampled from 0.97 to 1.08, brightness jitter of 0.08, and contrast jitter of 0.05. Horizontal flipping is disabled because left and right palms are distinct identities rather than interchangeable views. Validation, test, and calibration transforms consist only of resize, tensor conversion, channel replication, and normalization. The same transform builder is used for every controlled model. Figure 2 summarizes the offline ROI and preprocessing stages that precede these loader-level transforms.

**Figure 2.** Palm-vein image preparation: (a) region-of-interest extraction and (b) preprocessing to a 224 x 224 model input.

## 3.4. Target-Device Latency Characterization

Target-device profiling is performed before architecture search to obtain relative operator costs on the Raspberry Pi 5. The resulting latency lookup table (LUT) supplies a device-specific resource prior for the search objective; it is not an estimate of the end-to-end latency of a complete network.

Each of the 12 candidate operators is instantiated at five representative tensor configurations: `(C=8, H=56, stride=1)`, `(C=16, H=28, stride=1)`, `(C=32, H=14, stride=1)`, `(C=16, H=28, stride=2)`, and `(C=32, H=14, stride=2)`. These configurations represent resolution-preserving and downsampling edges in normal and reduction cells. Re-parameterizable convolution branches are fused before export. The full design contains at most 60 operator-shape probes before unsupported or non-computational cases are handled.

Probes are exported with ONNX opset 13 and converted to static QDQ INT8 graphs. The archived LUT quantizer uses per-channel signed INT8 weights and signed INT8 activations; probes without quantizable nodes, such as identity or pooling-only graphs, retain their FP32 representation. Random calibration tensors are acceptable for these isolated probes because the measurement concerns kernel execution rather than recognition accuracy. This probe calibration must not be confused with full-model PTQ, which uses real training images.

Measurement uses ONNX Runtime `CPUExecutionProvider`, sequential execution, four intra-operation threads, and one inter-operation thread on a Raspberry Pi 5. Each probe is warmed up before 200 timed iterations. The measurements are summarized by the median for each tensor configuration and the arithmetic mean across configurations for each operator. A measured 0.03299 ms QDQ-boundary floor is subtracted from affected isolated probes to reduce a boundary artifact that may not recur inside a fused full graph. The correction does not represent complete-model latency. Raw measurements, corrected values, and aggregation records are retained together for audit. Figure 3 summarizes the probe-to-LUT measurement path.

**Figure 3.** Construction of the Raspberry Pi latency lookup table from operator-shape benchmarks.

The LUT is device- and runtime-specific. Raspberry Pi model, RAM, operating-system build, CPU governor, cooling condition, ONNX Runtime version, power state, and thread affinity must be captured in the final experiment ledger. The current archive establishes Raspberry Pi 5, CPU execution, four threads, and the iteration count, but not every environmental field. Missing fields remain submission gates rather than values to be inferred retrospectively.

The corrected operator costs in Figure 3 serve as relative resource priors rather than end-to-end latency estimates. Their normalization and integration into the architecture objective are described in Section 3.5.2, whereas deployment latency is measured from complete ONNX graphs.

## 3.5. Hardware-Aware P-DARTS

### 3.5.1. Search Space and Candidate Operators

The search network follows the DARTS cell representation [18]. A cell receives the outputs of the preceding two cells and contains four intermediate nodes. Every intermediate node may receive edges from both cell inputs and all earlier intermediate nodes. A normal cell preserves spatial resolution, whereas a reduction cell applies stride two to edges arriving from the two cell inputs. The outputs of the four intermediate nodes are concatenated.

The initial operator set contains `none`, `skip_connect`, `sep_conv_3x3`, `sep_conv_5x5`, `dil_conv_3x3`, `dil_conv_5x5`, `mbconv3_3x3`, `mbconv6_3x3`, `rep_conv_3x3`, `rep_conv_5x5`, `avg_pool_3x3`, and `max_pool_3x3`. The set combines conventional DARTS operations with mobile inverted bottlenecks and re-parameterizable convolutions. `none` permits an edge to disappear, while `skip_connect` provides an identity or factorized reduction according to stride. The search input is 112 x 112, the initial channel width is 16, and the classification head predicts 834 identities.

For an edge [[MATH:edge_ij]] and candidate operator [[MATH:o]], the continuous mixture weight is

[[EQ:darts_mixture]]

The edge output is the sum of candidate outputs weighted by [[MATH:pi_edge]]. Ordinary network parameters and architecture parameters are optimized on separate data subsets. The implementation uses the first-order DARTS approximation: it does not compute the unrolled second-order update.

### 3.5.2. Latency-Regularized Search Objective

Let [[MATH:c_o]] denote the corrected LUT latency of operator [[MATH:o]], and let [[MATH:c_max]] be the maximum active operator cost at the current stage. The normalized cost is

[[EQ:lut_normalization]]

For one cell type, the expected cost is the mean over searchable edges:

[[EQ:cell_latency]]

The implementation averages the normal-cell and reduction-cell penalties:

[[EQ:combined_latency]]

Architecture parameters are updated on the search-architecture batch using

[[EQ:architecture_objective]]

where [[MATH:l_ce]] is cross-entropy with label smoothing 0.1 and [[MATH:lambda]] controls device pressure. Separate searches use [[MATH:lambda_set]]. The normalization makes the coefficient less sensitive to the absolute unit of the cost source, but it does not turn [[MATH:lambda]] into a universal hyperparameter. Operator-cost distributions and search dynamics remain specific to this search space and LUT.

Architecture optimization combines two information paths: classification evidence from the search-architecture batch and normalized device cost from the corrected latency lookup table. Figure 4 summarizes how these signals enter the architecture objective.

**Figure 4.** Integration of the device-specific latency lookup table into the hardware-aware P-DARTS search objective.

Gradients from the combined objective update the architecture parameters [[MATH:alpha]] while all active operators remain in the relaxed supernetwork. Discrete operators and the final genotype are selected only after the search has finished, as described in Section 3.5.4. As illustrated in Figure 4, the LUT guides operator selection during search but does not replace full-model benchmarking. Final FP32 and INT8 latency measurements are therefore obtained independently after ONNX export.

### 3.5.3. Progressive Search Schedule

Search proceeds through three 25-epoch stages with 5, 8, and 11 cells. The active operator count is reduced from 12 to 7 and then 4. Each stage begins with ten epochs during which only ordinary network weights are updated; architecture updates start in epoch 11. Low-scoring operations are removed between stages using architecture weights aggregated across normal and reduction cells. Identity is retained during pruning, and convolutional-family safeguards prevent the final active set from collapsing into only non-parametric operations.

Network weights are optimized using stochastic gradient descent with learning rate 0.025, momentum 0.9, weight decay [[MATH:three_e_minus_four]], and cosine annealing to 0.001. Architecture parameters use Adam with learning rate [[MATH:six_e_minus_four]], beta values 0.5 and 0.999, and weight decay [[MATH:one_e_minus_three]]. Batch size is 16, gradient norm is clipped at 5.0, and skip-connection dropout rises linearly from 0 to 0.5 within every stage. Searches use seed 42. Running multiple search seeds would be needed to characterize search instability; the three-seed requirement in the present benchmark applies to training the frozen genotype, not to repeating NAS.

**Table 3. Hardware-aware P-DARTS search configuration.**

| Component | Configuration |
|---|---|
| Task/input | 834-class identification; 3 x 112 x 112 search input |
| Cell topology | Two inputs, four intermediate nodes, normal and reduction cells |
| Candidate operators | 12: null, identity, 2 separable, 2 dilated, 2 MBConv, 2 RepConv, average pool, max pool |
| Progressive stages | 5/8/11 cells; 12/7/4 active operators; 25 epochs each |
| Architecture warm-up | 10 epochs per stage |
| Weight optimizer | SGD; lr 0.025; momentum 0.9; weight decay [[MATH:three_e_minus_four]]; cosine minimum 0.001 |
| Architecture optimizer | Adam; lr [[MATH:six_e_minus_four]]; betas (0.5, 0.999); weight decay [[MATH:one_e_minus_three]] |
| Regularization | Label smoothing 0.1; gradient clip 5.0; skip dropout 0 to 0.5; maximum two skips/cell |
| Data for bilevel updates | 3,336 training samples for weights; 3,336 for architecture parameters |
| Latency coefficients | 0.00, 0.05, 0.10, and 0.20 |
| Search seed | 42 |

### 3.5.4. Genotype Derivation and Architecture Freezing

After the third stage, `none` is excluded. For each intermediate node, candidate incoming edges are scored by their highest non-null softmax weight. The two strongest incoming edges are retained, and the highest-weight non-null operation is chosen for each retained edge. At most two skip connections are allowed in each cell; excess skips are replaced by an eligible non-skip operation according to the archived post-processing rule. The final normal and reduction cells are serialized as a genotype.

The controlled benchmark uses the frozen [[MATH:lambda_005]], initial width 12, ten-cell genotype with an eightfold stem downsampling configuration. Only topology and explicit architecture fields are imported from the thesis artifact. Supernetwork weights, old student checkpoints, and old training hyperparameters are not reused. The local benchmark verifies the source configuration hash and constructs new model weights for every seed. Because the genotype and capacity were exposed to the legacy test results during thesis development, its evaluation is retrospective under the fixed split, as stated in Section 3.1.

## 3.6. Comparative Training Protocols

Three protocols answer different questions and must not be pooled into one ranking. The primary controlled scratch protocol trains every eligible architecture from random initialization for at most 600 epochs with seeds 42, 123, and 2026. The optimizer is AdamW with learning rate [[MATH:one_e_minus_three]] and weight decay 0.05. A ten-epoch linear warm-up begins at 1% of the base learning rate, followed by cosine decay to [[MATH:one_e_minus_six]]. Batch size is 64, label smoothing is 0.2, and gradient norm is clipped at 1.0. All architectures use the augmentation in Section 3.3 and no KD or pretrained weights.

The current audited comparator roles are: the frozen P-DARTS genotype; ProxylessNAS-Mobile, FBNet-C, and MnasNet-A1 as general hardware-aware NAS models; DingBaseline, DingPW, and DingPruned as paper-constrained reconstructions; and PalmNet-0.5x2413 and PalmNet-0.5x2411 as domain-specific paper-constrained reconstructions. Chen StudentNet remains an unregistered audit artifact. AMPVNet remains a possible domain-specific addition, but it cannot enter a controlled result table until an auditable implementation is adapted and trained through the same engine. A model absent from the completed experiment ledger is not silently replaced with a literature value.

The secondary transfer protocol uses verified public ImageNet weights only for ProxylessNAS-Mobile, FBNet-C, and MnasNet-B1 (`torchvision.mnasnet1_0`) in the current benchmark registry. The original classifier is replaced by an 834-class head. The backbone is frozen for five epochs, with classifier learning rate [[MATH:one_e_minus_three]]; after unfreezing, backbone and classifier learning rates are [[MATH:one_e_minus_four]] and [[MATH:one_e_minus_three]], respectively. Training lasts at most 200 epochs with five warm-up epochs, the same minimum learning rate, weight decay, batch size, augmentation, seeds, and checkpoint rule as the scratch protocol. MnasNet-A1 remains a scratch comparator because no official PyTorch checkpoint for the exact A1 architecture was used. A model without audited weights is marked not applicable rather than initialized from an unofficial checkpoint.

The KD protocol is a separate controlled intervention. Its final epoch budget and participating baseline pair must be frozen before execution. They must be identical across students, as must the teacher checkpoint, optimizer, scheduler, augmentation, temperature, hard-label weight, and checkpoint rule. The legacy [[MATH:t_20]], [[MATH:alpha_05]] setting is a candidate rather than automatically confirmatory because it was explored on the previously observed test split. KD results enter the journal comparison only after this selection issue is resolved and documented.

For every training protocol, the engine saves the checkpoint with minimum validation loss. Test evaluation occurs once after that checkpoint is loaded. Training is not stopped by test accuracy, and the final epoch model is not substituted when its test result is better. Completed-seed count is reported so that an interrupted batch cannot be mistaken for a three-seed summary.

**Table 4. Controlled scratch, pretrained, and knowledge-distillation protocols.**

| Protocol | Initialization and eligible models | Training budget | Scientific purpose |
|---|---|---|---|
| Controlled scratch | Random; all audited architectures | 600 epochs; seeds 42/123/2026; AdamW [[MATH:one_e_minus_three]]; warm-up 10 | Isolate architecture quality |
| Pretrained transfer | Verified ImageNet weights; ProxylessNAS-Mobile, FBNet-C, MnasNet-B1 (`torchvision.mnasnet1_0`) | 200 epochs; three seeds; freeze 5; backbone/head lr [[MATH:one_e_minus_four]] / [[MATH:one_e_minus_three]] | Measure practical transfer separately |
| Controlled KD | Frozen validation-selected teacher; P-DARTS plus at least two validation-selected lightweight baselines | Same frozen budget and hyperparameters for every student | Attribute gains to KD rather than topology |
| Deployment | Minimum-validation-loss checkpoint from each reported seed/protocol | ONNX FP32 then static INT8 PTQ | Measure accuracy-size-latency trade-off |

*Table note.* The scratch and pretrained rows produce separate result tables. The KD row is a submission gate until teacher and hyperparameter selection are independent of the legacy test observations.

## 3.7. Knowledge Distillation

The candidate teacher is EfficientNetV2M [30]. In the journal protocol, the teacher must be selected from the predefined candidate set using minimum validation loss, then frozen before any student comparison. Capacity may be used as an inclusion criterion established in advance, but test accuracy, one-vs-rest AUC, and the legacy softmax-derived EER cannot participate in selection. Teacher test accuracy is descriptive and is computed only after the teacher decision has been fixed.

For input [[MATH:x]] and label [[MATH:y]], let [[MATH:z_s]] and [[MATH:z_t]] be student and teacher logits. With temperature [[MATH:temperature]], the softened student and teacher distributions are, respectively,

[[EQ:student_distribution]]

[[EQ:teacher_distribution]]

The Hinton-style loss is

[[EQ:kd_loss]]

The teacher remains in evaluation mode and receives no gradients. The factor [[MATH:t_squared]] preserves the scale of the soft-target gradient. The same formula, temperature, and [[MATH:alpha]] are used for every student in the controlled KD table. If embedding, relation, top-k, or multi-teacher terms are explored, they constitute separate ablations and cannot be folded into the Hinton-KD result without explicit labels.

The baseline students for KD are selected using the non-distilled validation results and efficiency constraints, not test ranking. At least two strong lightweight alternatives should participate, ideally one general NAS-mobile model and one compact domain-specific model whose implementation status is auditable. This design tests whether the teacher's benefit is specific to the searched architecture or broadly available to compact networks.

The archived thesis teacher configuration lists test accuracy, best validation loss, and representation capacity as selection criteria. It also contains a grid of KD temperatures and alpha values evaluated on the same legacy test split. Those files remain valuable exploratory evidence but do not satisfy the frozen validation-only rule. Before submission, the experiment ledger must either document a new validation-only teacher/KD selection and rerun, or restrict the paper's KD claims to exploratory analysis. No wording change can convert test-informed model selection into untouched evaluation.

## 3.8. ONNX Export and INT8 Quantization

The minimum-validation-loss PyTorch checkpoint is exported to ONNX [35] with opset 13, input name `input`, output name `logits`, and a dynamic batch dimension. The export path records model name, training protocol, seed, parameter count, checkpoint path, and file hash. The ONNX checker validates graph structure. Numerical parity is evaluated on a representative tensor using absolute tolerance [[MATH:one_e_minus_four]] and relative tolerance [[MATH:one_e_minus_three]]; the maximum absolute output difference is stored. A graph that fails parity does not proceed to quantization or deployment timing.

Static PTQ uses ONNX Runtime's QDQ representation, per-channel QInt8 weights, QUInt8 activations, and MinMax calibration. The calibration reader iterates over the fixed 834-image training manifest described in Section 3.2, applies the evaluation transform, and supplies batches to the quantizer. No validation or test image is used for range estimation. The quantized graph is validated, and a one-sample smoke test must produce output shape `[1,834]`.

Accuracy is evaluated independently for FP32 and INT8 ONNX graphs on all 834 test images. The output record includes correct, total, and accuracy, together with FP32 and INT8 file hashes and byte sizes. Operators left in FP32 by the quantizer must be enumerated from the final graph before the manuscript is submitted; the QDQ label alone does not imply that every operation executes as integer arithmetic.

The full-model deployment recipe differs from the archived probe recipe in activation signedness: the corrected search LUT was created with signed INT8 probe activations, whereas the benchmark deployment config specifies QUInt8 activations. Until the LUT is regenerated with the same quantization recipe or an equivalence experiment is supplied, the manuscript describes the search as INT8-informed but does not claim that its LUT exactly matches the deployed quantizer. This alignment check is a required reproducibility item. Figure 5 summarizes the separation between checkpoint export, training-only calibration, accuracy evaluation, and target-device timing.

**Figure 5.** ONNX model export, static INT8 post-training quantization, accuracy validation, and Raspberry Pi latency benchmarking.


## 3.9. Evaluation and Statistical Reporting

Recognition effectiveness is reported as top-1 closed-set identification accuracy:

[[EQ:top1_accuracy]]

The report includes [[MATH:n_correct]], [[MATH:n_test_834]], and [[MATH:n_error]]. Because the test split contains one image per identity, one changed prediction corresponds to [[MATH:test_resolution]] percentage point. This resolution is reported when small differences are interpreted. EER, FAR, FRR, ROC, and AUC are excluded because no verification score protocol is defined. Macro precision, recall, and F1 are also excluded because the balanced single-example-per-class test makes them nearly redundant with accuracy.

For every model-protocol combination, the accuracies from seeds 42, 123, and 2026 are summarized by their arithmetic mean and sample standard deviation:

[[EQ:sample_sd]]

The report lists all seed-level values and the number of completed seeds. Three observations are insufficient for strong claims about normality or statistical significance, so the analysis remains descriptive. A difference smaller than one or two test errors is not emphasized without consistent seed behavior and a meaningful efficiency advantage.

Efficiency reporting includes trainable parameter count, multiply-accumulate operations or FLOPs at input `[1,3,224,224]`, ONNX FP32 and INT8 byte sizes, and target-device latency. The tool and counting convention used for arithmetic complexity must be named because one MAC may be reported as one or two FLOPs. Latency is measured with batch size one, ONNX Runtime `CPUExecutionProvider`, sequential execution, four intra-operation threads, and one inter-operation thread. Each graph receives 50 warm-up executions and 500 timed executions. The timing script reports mean, median, p95, minimum, and maximum latency. A Raspberry Pi claim is accepted only when the recorded machine is ARM64 Linux; host-machine timing is retained as engineering evidence but not labeled Raspberry Pi performance.

The final analysis compares FP32 with INT8 using accuracy change in percentage points, size ratio, and latency speedup. No model is called best without the metric and precision state. Pareto analysis identifies models for which no alternative is simultaneously at least as accurate, no larger, and no slower under the same protocol. Scratch, pretrained, and KD models form separate comparison sets unless the figure explicitly labels initialization and training treatment.

**Table 5. ONNX quantization and Raspberry Pi benchmarking configuration.**

| Component | Fixed configuration | Required record |
|---|---|---|
| ONNX export | Opset 13; input `input`; output `logits`; dynamic batch | Checkpoint/file hashes; model, seed, protocol; checker result |
| Parity | [[MATH:atol_1e4]], [[MATH:rtol_1e3]] | All-close result and maximum absolute difference |
| Full-model PTQ | Static QDQ; per-channel QInt8 weights; QUInt8 activations; MinMax | Quantizer/runtime version and remaining FP32 operators |
| Calibration | 834 training images; one per identity | Manifest hash and membership validation |
| Runtime | CPUExecutionProvider; sequential; intra-op 4; inter-op 1; batch 1 | Raspberry Pi model/RAM, OS, ORT version, governor, cooling, power state |
| Timing | 50 warm-up; 500 timed iterations | Mean, median, p95, minimum, maximum, raw timing file |
| Accuracy | FP32 and INT8 evaluated on 834-image legacy test | Correct, errors, total, and accuracy per seed |

The reproducibility package retains configuration JSON files, split and calibration manifests, genotype files, model-provenance notes, checkpoint and ONNX hashes, seed-level metrics, and Raspberry Pi runtime records. The manuscript will be updated from these machine-readable artifacts after all prespecified runs finish; no result is entered from a console screenshot or reconstructed from a rounded thesis table.

# Internal Figure Production Checklist - Remove Before Submission

Temporary artwork status: the six source images from thesis Figures 3.1, 3.2, 3.3, 3.5, 3.8, and 3.10 are embedded for layout development only. They must be redrawn or updated in English before journal submission.

## Required for Sections 1-3

Figure 1 - redraw thesis Figure 3.1 with the 6,672/834/834 split, the 3,336/3,336 internal NAS split, separate scratch/pretrained/KD branches, training-only calibration, and distinct FP32/INT8 deployment paths.

Figure 2 - merge and update thesis Figures 3.2 and 3.3. Match the archived preprocessing implementation: Gaussian 7 x 7, Otsu mask, 15 x 15 morphology, 384 x 384 crop, CLAHE 2.0/8 x 8, min-max normalization, and Lanczos resize to 224 x 224.

Figure 3 - redraw thesis Figure 3.5 as the operator-shape probe and latency-LUT construction pipeline. Add the five tensor configurations, 12 operator families, QDQ conversion, raw and corrected latency, median per configuration, and mean per operator. Do not include architecture-parameter optimization in this figure.

Figure 4 - redraw thesis Figure 3.8 as the LUT-to-search integration diagram. Add max-cost normalization, classification and resource-information paths, the architecture objective, and the gradient path to the architecture parameters. Keep full-model deployment benchmarking outside this figure.

Figure 5 - update thesis Figure 3.10 with minimum-validation-loss checkpoint selection, ONNX checker and parity gates, the 834-image training-only calibration manifest, FP32 and INT8 accuracy paths, file hashes, and Raspberry Pi mean, median, and p95 latency reporting.

## Artwork Requirements

Export every final figure as a separate file, retain editable captions in the manuscript, use consistent English lettering, and preserve raw images and transformation history. Target at least 1000 dpi for line drawings or 500 dpi for mixed image-line artwork. Do not transfer generic background diagrams or unreconciled Chapter 4 result charts into the condensed article.

# References (temporary numbered drafting style)

[1] W. Wu, S. J. Elliott, S. Lin, S. Sun, and Y. Tang, “Review of palm vein recognition,” *IET Biometrics*, vol. 9, no. 1, pp. 1-10, 2020, doi: 10.1049/iet-bmt.2019.0034.

[2] D. Luo, Y. Qiao, D. Xie, S. Zhang, and W. Kang, “Palm vein recognition under unconstrained and weak-cooperative conditions,” *IEEE Transactions on Information Forensics and Security*, vol. 19, pp. 4601-4614, 2024, doi: 10.1109/TIFS.2024.3378427.

[3] H. Qin, M. A. El-Yacoubi, Y. Li, and C. Liu, “Multi-scale and multi-direction GAN for CNN-based single palm-vein identification,” *IEEE Transactions on Information Forensics and Security*, vol. 16, pp. 2652-2666, 2021, doi: 10.1109/TIFS.2021.3059340.

[4] M. Wulandari, R. Chai, B. Basari, and D. Gunawan, “Hybrid feature extractor using discrete wavelet transform and histogram of oriented gradient on convolutional-neural-network-based palm vein recognition,” *Sensors*, vol. 24, no. 2, article 341, 2024, doi: 10.3390/s24020341.

[5] H.-I. Liu, M. Galindo, H. Xie, L.-K. Wong, H.-H. Shuai, Y.-H. Li, and W.-H. Cheng, “Lightweight deep learning for resource-constrained environments: A survey,” *ACM Computing Surveys*, vol. 56, no. 10, article 267, pp. 1-42, 2024, doi: 10.1145/3657282.

[6] J. Chen and X. Ran, “Deep learning with edge computing: A review,” *Proceedings of the IEEE*, vol. 107, no. 8, pp. 1655-1674, 2019, doi: 10.1109/JPROC.2019.2921977.

[7] N. Saxena and D. Varshney, “Smart home security solutions using facial authentication and speaker recognition through artificial neural networks,” *International Journal of Cognitive Computing in Engineering*, vol. 2, pp. 154-164, 2021, doi: 10.1016/j.ijcce.2021.10.001.

[8] Z.-C. Chen, S.-Y. Jhong, and C.-H. Hsia, “Design of a lightweight palmf-vein authentication system based on model compression,” *Journal of Information Science and Engineering*, vol. 37, no. 4, pp. 809-825, 2021, doi: 10.6688/JISE.202107_37(4).0005.

[9] Y.-Y. Chen, C.-H. Hsia, and P.-H. Chen, “Contactless multispectral palm-vein recognition with lightweight convolutional neural network,” *IEEE Access*, vol. 9, pp. 149796-149806, 2021, doi: 10.1109/ACCESS.2021.3124631.

[10] Z. Ding, N. Pu, Q. Miao, Z. Chen, Y. Xu, and H. Liu, “Efficient palm vein recognition optimized by neural architecture search and hybrid compression,” in *2025 International Conference on Multi-Agent Systems for Collaborative Intelligence (ICMSCI)*, pp. 826-832, 2025, doi: 10.1109/ICMSCI62561.2025.10894245.

[11] A. K. Jain, A. Ross, and S. Prabhakar, “An introduction to biometric recognition,” *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 14, no. 1, pp. 4-20, 2004, doi: 10.1109/TCSVT.2003.818349.

[12] ISO/IEC, *ISO/IEC 19795-1:2021 Information technology—Biometric performance testing and reporting—Part 1: Principles and framework*, 2021. Available: https://www.iso.org/standard/73515.html (accessed Aug. 10, 2026).

[13] R. Tanna, T. Patel, F. M. Alotaibi, R. H. Jhaveri, and T. R. Gadekallu, “OcclusionNetPlusPlus: A multi-scale similarity network with adaptive occlusion detection for robust iris recognition,” *International Journal of Cognitive Computing in Engineering*, vol. 7, pp. 74-85, 2026, doi: 10.1016/j.ijcce.2025.09.002.

[14] H. Benmeziane, K. E. Maghraoui, H. Ouarnoughi, S. Niar, M. Wistuba, and N. Wang, “Hardware-aware neural architecture search: Survey and taxonomy,” in *Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence*, pp. 4322-4329, 2021, doi: 10.24963/ijcai.2021/592.

[15] H. Cai, L. Zhu, and S. Han, “ProxylessNAS: Direct neural architecture search on target task and hardware,” in *International Conference on Learning Representations*, 2019. Available: https://openreview.net/forum?id=HylVB3AqYm

[16] M. Tan, B. Chen, R. Pang, V. Vasudevan, M. Sandler, A. Howard, and Q. V. Le, “MnasNet: Platform-aware neural architecture search for mobile,” in *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2815-2823, 2019, doi: 10.1109/CVPR.2019.00293.

[17] B. Wu, X. Dai, P. Zhang, Y. Wang, F. Sun, Y. Wu, Y. Tian, P. Vajda, Y. Jia, and K. Keutzer, “FBNet: Hardware-aware efficient ConvNet design via differentiable neural architecture search,” in *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 10726-10734, 2019, doi: 10.1109/CVPR.2019.01099.

[18] H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable architecture search,” in *International Conference on Learning Representations*, 2019. Available: https://openreview.net/forum?id=S1eYHoC5FX

[19] X. Chen, L. Xie, J. Wu, and Q. Tian, “Progressive differentiable architecture search: Bridging the depth gap between search and evaluation,” in *2019 IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 1294-1303, 2019, doi: 10.1109/ICCV.2019.00138.

[20] T. Wu, L. Leng, M. K. Khan, and F. A. Khan, “Palmprint-palmvein fusion recognition based on deep hashing network,” *IEEE Access*, vol. 9, pp. 135816-135827, 2021, doi: 10.1109/ACCESS.2021.3112513.

[21] M. A.-E. Eladlani, S. Si Kaddour, L. Boubchir, and B. Daachi, “On the use of convolutional neural networks for palm vein recognition,” in *2022 IEEE International Conference on Big Data (Big Data)*, pp. 3586-3592, 2022, doi: 10.1109/BigData55660.2022.10020251.

[22] W. Jia, R.-X. Hu, Y.-K. Lei, Y. Zhao, and J. Gui, “A performance evaluation of classic convolutional neural networks for 2D and 3D palmprint and palm vein recognition,” *International Journal of Automation and Computing*, vol. 18, no. 1, pp. 18-44, 2021, doi: 10.1007/s11633-020-1257-9.

[23] S. Luo and X. Huang, “A lightweight neural network for palm vein recognition,” *Frontiers in Computing and Intelligent Systems*, vol. 2, no. 3, pp. 101-105, 2023, doi: 10.54097/fcis.v2i3.5412.

[24] T. Elsken, J. H. Metzen, and F. Hutter, “Neural architecture search,” in *Automated Machine Learning: Methods, Systems, Challenges*, F. Hutter, L. Kotthoff, and J. Vanschoren, Eds. Cham, Switzerland: Springer, 2019, pp. 63-77, doi: 10.1007/978-3-030-05318-5_3.

[25] A. Zela, T. Elsken, T. Saikia, Y. Marrakchi, T. Brox, and F. Hutter, “Understanding and robustifying differentiable architecture search,” in *International Conference on Learning Representations*, 2020. Available: https://openreview.net/forum?id=Div-GO6PO1

[26] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 4510-4520, 2018, doi: 10.1109/CVPR.2018.00474.

[27] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding, and J. Sun, “RepVGG: Making VGG-style ConvNets great again,” in *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 13733-13742, 2021. Available: https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html (accessed Aug. 10, 2026).

[28] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015, doi: 10.48550/arXiv.1503.02531.

[29] J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,” *International Journal of Computer Vision*, vol. 129, no. 6, pp. 1789-1819, 2021, doi: 10.1007/s11263-021-01453-z.

[30] M. Tan and Q. V. Le, “EfficientNetV2: Smaller models and faster training,” in *Proceedings of the 38th International Conference on Machine Learning*, vol. 139, pp. 10096-10106, 2021. Available: https://proceedings.mlr.press/v139/tan21a.html

[31] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 2704-2713, 2018, doi: 10.1109/CVPR.2018.00286.

[32] R. Krishnamoorthi, “Quantizing deep convolutional networks for efficient inference: A whitepaper,” arXiv:1806.08342, 2018, doi: 10.48550/arXiv.1806.08342.

[33] N. Otsu, “A threshold selection method from gray-level histograms,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62-66, 1979, doi: 10.1109/TSMC.1979.4310076.

[34] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in *Graphics Gems IV*, P. S. Heckbert, Ed. San Diego, CA, USA: Academic Press Professional, 1994, pp. 474-485, doi: 10.1016/B978-0-12-336156-1.50061-6.

[35] Microsoft, “ONNX Runtime: Export PyTorch models and quantize ONNX models.” [Online]. Available: https://onnxruntime.ai/docs/tutorials/accelerate-pytorch/pytorch.html and https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html (accessed Aug. 10, 2026).
