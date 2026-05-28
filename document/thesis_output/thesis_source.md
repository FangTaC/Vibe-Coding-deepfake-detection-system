TITLE_CN: 智能体协作的深度伪造人脸检测系统设计与实现
TITLE_EN: Design and Implementation of an Agent-Collaborative Deepfake Face Detection System
SCHOOL: 湖南大学
COLLEGE: 待填写学院
MAJOR: 待填写专业
STUDENT: 待填写
STUDENT_ID: 待填写
SUPERVISOR: 待填写
DATE_CN: 二〇二六年四月
---

# 中文摘要
深度伪造技术在图像与视频生成领域发展迅速，其低门槛、高逼真度和强传播性使得伪造内容在舆情传播、身份冒用、隐私侵犯和社会信任破坏等方面带来了明显风险。针对传统单模型深度伪造检测方案在可解释性不足、部署链路脆弱、异常场景处理能力有限等问题，本文结合当前代码仓库的实际实现，设计并实现了一套面向图像级 Deepfake 人脸检测的智能体协作系统。该系统以“任务驱动、双智能体协作、结构化证据链输出、可回退部署”为总体设计思路，面向本科毕业设计场景构建了完整的前后端闭环。

系统总体上由前端交互层、后端任务调度层、特征提取智能体、决策智能体、多后端推理模块以及结果可视化与证据链模块构成。用户在前端上传图像后，后端首先创建异步检测任务并完成状态持久化；随后，特征提取智能体执行人脸检测、人脸裁剪、视觉风险分析、图像质量评估、频域与压缩痕迹评估、公众人物候选匹配以及热力图与标注图生成；决策智能体在此基础上构建任务画像，根据风险水平在视觉直判、视觉加语义联合分析和人工复核三类预定义策略中进行受限选择，并结合阈值策略、冲突证据和复核条件输出最终结论。系统最终给出“疑似真实”“疑似伪造”“需人工复核”三态结果，同时返回置信度、融合分数、证据链、模型版本、阈值来源、原图标注、裁剪人脸和热力图等信息，以增强结果的可解释性与可审计性。

在模型构建方面，本文基于 FaceForensics++ 数据集构建图像级训练流程，完成了从子集下载、视频抽帧、训练集整理到模型训练、元数据固化和后端接入的完整工程链路。视觉分支采用 EfficientNet-B0 作为轻量级卷积神经网络主干，对检测到的人脸裁剪图进行真假判别；融合分支采用 XGBoost 对最大人脸风险、平均人脸风险、频域特征、图像质量、一致性、压缩痕迹、人脸数量和语义分数等结构化特征进行二阶段融合。考虑到毕业设计场景中的资源限制和部署稳定性要求，系统在关键环节均实现了多级回退机制：人脸检测模块可在 InsightFace、OpenCV Haar 和启发式检测器之间降级切换；视觉分析模块可在训练好的 EfficientNet-B0 与启发式视觉后端之间切换；融合模块可在 XGBoost 与启发式加权融合之间切换；语义分析模块可在远程视觉大模型、本地多模态模型与规则语义后端之间切换，从而保证在模型权重缺失、依赖未就绪或外部服务不可用时，系统仍能完成检测闭环。

结合当前仓库实现可知，本文工作的重点不仅在于训练单一检测模型，更在于完成“数据准备、模型训练、服务接入、任务管理、结果展示、证据组织和异常回退”一体化工程实现。本文在系统设计中强调结构化结果输出与人工复核入口，避免在灰区样本、低质量输入、多人脸场景和语义冲突场景下进行过度自信的硬判定，体现了深度伪造检测系统在实际应用中的可靠性要求。论文最后从系统功能联调、回退链路验证、前端交互呈现和当前阶段结果分析等方面对系统进行了总结，并对后续跨数据集评估、真实语义特征离线注入和模型能力持续扩展进行了展望。

关键词：深度伪造检测；人脸图像取证；智能体协作；证据链；多级回退；多特征融合

# Abstract
With the rapid development of deepfake generation techniques, forged facial content has become increasingly realistic, accessible, and influential in online communication. Such content poses substantial risks to media credibility, identity security, privacy protection, and public trust. To address the limitations of conventional single-model detection approaches, especially in terms of interpretability, deployment robustness, and exception handling, this thesis designs and implements an image-level deepfake face detection system based on agent collaboration and structured evidence output. The system is built around a complete engineering workflow rather than a standalone classifier, and emphasizes task-driven processing, dual-agent collaboration, interpretable decision making, and graceful degradation under constrained runtime conditions.

The proposed system consists of a front-end interaction layer, a back-end task orchestration layer, a feature extraction agent, a decision agent, multiple inference backends, and result visualization modules. After an image is uploaded by the user, the back-end creates an asynchronous task and persists task states to the database. The feature extraction agent then performs face detection, face cropping, visual risk analysis, image quality estimation, frequency-domain analysis, compression artifact estimation, public figure candidate matching, and artifact generation including annotated images, cropped faces, and heatmaps. Based on the extracted evidence, the decision agent builds a task profile and selects one of three constrained strategies, namely visual-only judgment, visual-plus-semantic analysis, or review-only routing. The final decision is generated through threshold-aware fusion together with conflict handling and review conditions. The system outputs three-way labels, namely likely real, likely fake, and manual review required, together with confidence scores, fusion scores, evidence chains, model version records, threshold sources, and visualization artifacts, which improves interpretability and auditability.

For model construction, this thesis implements a complete training pipeline based on FaceForensics++, including subset download, video frame extraction, dataset organization, model training, metadata generation, and back-end integration. The visual branch uses EfficientNet-B0 as a lightweight convolutional backbone for cropped face classification, while the fusion branch employs XGBoost to combine structured features such as maximum face risk, mean face risk, frequency-domain statistics, image quality, augmentation consistency, compression traces, face count, and semantic score. To improve deployment stability, hierarchical fallback mechanisms are introduced in all critical modules. Face detection can degrade from InsightFace to OpenCV Haar and then to a heuristic detector; visual analysis can degrade from the trained EfficientNet-B0 model to a heuristic visual backend; score fusion can degrade from XGBoost to heuristic weighted fusion; semantic analysis can degrade from remote or local multimodal models to a rule-based semantic backend. As a result, the system can preserve its processing chain even when model assets or external services are not fully available.

Grounded in the current repository implementation, the work of this thesis lies not only in model training, but also in the integrated realization of data preparation, training workflow, runtime services, task management, result presentation, structured evidence organization, and exception fallback. The system explicitly preserves a human-in-the-loop review path for gray-zone samples, low-quality inputs, multi-face scenes, and semantically conflicting cases, which better reflects the reliability requirements of practical deepfake forensic systems. Finally, this thesis summarizes the implemented functionality, system workflow, key modules, and current-stage evaluation, and discusses future improvements including cross-dataset benchmark completion, offline semantic feature export, and further multimodal enhancement.

Keywords: deepfake detection; face image forensics; agent collaboration; evidence chain; fallback mechanism; feature fusion

# 目录
说明：生成的 Word 文档将自动插入目录域，首次打开后如目录未刷新，可在 Word 中全选后更新域。

# 第1章 绪论
## 1.1 研究背景与意义
随着生成式人工智能、换脸算法和多媒体编辑工具的快速发展，深度伪造内容的生成成本不断下降，而视觉逼真度却持续提高。与早期依赖复杂图形学流程的伪造方式相比，当前基于深度学习的人脸合成、身份替换和表情迁移方法能够在较短时间内生成高拟真度样本，并依托社交媒体、短视频平台和即时通讯工具迅速传播。这种变化使深度伪造检测从单纯的图像分析问题，逐步演化为涉及媒体可信度、平台审核、社会安全和个人权益保护的综合性问题。[1][6]

在实际应用中，深度伪造检测不仅需要输出一个真假判断结果，更需要对“为何做出该判断”给出结构化解释。一方面，伪造样本常常存在压缩、裁剪、重采样、上传转码、画质退化和局部遮挡等复杂情况，单一模型的置信分数难以直接支撑可信决策；另一方面，许多工程系统需要面对模型权重尚未就绪、外部服务短时不可用、运行环境依赖缺失等现实约束。如果检测系统过度依赖单一模型或单一接口，那么一旦出现异常，就会导致整条链路中断，不利于真实场景中的稳定使用。因此，面向毕业设计的系统实现不能只停留在“训练一个分类模型”，而应进一步考虑任务调度、异常处理、可视化证据、人工复核入口和系统级可扩展性。

本项目最初的开发理念正是围绕这一点展开：不将 Deepfake 检测简单视为一个黑盒二分类问题，而是将其设计为一条可运行、可解释、可回退、可审计的检测链路。当前仓库实现了一套基于双智能体协作的图像级 Deepfake 人脸检测系统。其中，特征提取智能体负责感知、取证和中间产物生成；决策智能体负责策略选择、融合决策和结论输出。与传统“输入图像、输出标签”的线性流程相比，该系统更加注重结构化证据组织、人工复核保护和前后端联动展示，从而更符合实际部署与答辩展示的双重需求。

从研究意义上看，本文的工作具有三个层面的价值。第一，在工程实现层面，系统完整打通了从数据准备、模型训练、服务接入到结果展示的闭环，实现了面向真实用户交互的任务化检测流程。第二，在方法设计层面，本文将轻量级视觉模型、结构化特征融合、规则约束决策以及可选语义分析组合起来，形成了兼顾效果与可解释性的检测框架。第三，在应用可靠性层面，本文通过多级回退机制和三态输出策略降低了误判风险，使系统在灰区样本和异常环境下仍能维持稳定运行，并保留人工复核入口，这对于深度伪造检测这一高风险任务尤其重要。

## 1.2 国内外研究现状
从国外研究现状来看，深度伪造检测大致经历了三个阶段。第一阶段主要关注低层图像取证特征，例如压缩痕迹、颜色统计差异、重采样痕迹和边界不连续性等，这类方法解释性较强，但在复杂压缩和多样化伪造条件下鲁棒性有限。第二阶段逐渐转向数据驱动的卷积神经网络方法，例如 MesoNet 通过关注中尺度视觉特征来检测人脸伪造，FaceForensics++ 则通过大规模构造数据集与基准推动了检测模型的标准化训练与评估。[1][2] 第三阶段则进一步走向多分支融合、多模态分析、时空建模和大模型辅助解释等方向，强调对复杂场景、跨数据集泛化和真实部署问题的适应能力。[6][7]

在典型的数据资源方面，FaceForensics++ 是深度伪造检测领域最常见的标准数据集之一，其包含 DeepFakes、Face2Face、FaceSwap 和 NeuralTextures 等多种典型篡改方式，并提供不同压缩等级下的大规模伪造图像与视频样本。[1] 这一数据集为后续大量检测方法提供了统一训练与比较基础。与此同时，研究者也提出了针对特定伪造痕迹的检测方法，例如利用人脸 warping 痕迹检测伪造视频的方法，说明深度伪造检测不仅可以依赖端到端模型，也可以结合具体成因进行特征建模。[8]

国内研究更多强调工程化落地、场景适配以及结合平台审核需求构建稳健系统。近年来，随着深度伪造检测从论文实验走向平台实践，研究重点逐步从“单模型离线精度”扩展到“系统级可靠性、解释性和人工协同能力”。这意味着一个可用的检测系统不仅需要模型本身，还需要任务持久化、前端交互、模型版本管理、阈值来源追踪、可视化证据输出以及异常回退能力。对于毕业设计而言，这类系统性工作往往比单独报告一个高精度数值更能体现工程训练价值。

综合现有研究可以发现，当前仍存在几个值得关注的问题。其一，多数工作仍偏向离线实验，对真实部署中的依赖缺失、服务超时与异常输入考虑不够。其二，很多检测结果只给出单一概率值，缺少面向人类使用者的证据链解释。其三，实际场景中存在大量灰区样本，例如多脸图像、压缩严重图像、公众人物图像和语义冲突图像，这些样本不适合被强制二分类。其四，许多系统在模型未就绪时无法工作，不利于原型系统的持续演示和稳定迭代。本文正是在这些问题背景下，结合现有仓库实现，构建一套更强调系统闭环与可靠性的 Deepfake 检测方案。

## 1.3 研究目标与内容
本文的总体目标是围绕当前代码仓库实现一套面向图像级 Deepfake 人脸检测的毕业设计系统，并在论文中系统梳理其设计动机、总体架构、关键模块、训练流程、前后端实现、测试结果和后续展望。具体而言，本文的研究内容主要包括以下几个方面。

第一，完成系统总体方案设计，明确前端上传、后端任务调度、特征提取、决策推理、结果可视化与历史追踪之间的关系。第二，围绕特征提取智能体与决策智能体构建双智能体协作流程，使前者负责感知与取证，后者负责受限策略选择和最终结论输出。第三，完成视觉模型与融合模型的训练链路，包括 FaceForensics++ 子集下载、视频抽帧、训练集整理、EfficientNet-B0 训练、XGBoost 融合训练以及 metadata 固化。第四，设计多级回退机制，使系统在模型权重、依赖库或语义服务缺失时仍然能够生成可解释结果。第五，完成前后端联调和结果展示，使证据链、热力图、裁剪人脸、模型版本与阈值来源能够在页面中呈现。第六，对当前阶段系统实现进行测试和分析，并指出其在正式定量实验方面仍需补充的内容。

## 1.4 论文组织结构
为便于系统展开本文内容，论文后续章节安排如下：第一章介绍研究背景、研究意义、国内外研究现状以及本文研究目标；第二章介绍与系统实现密切相关的理论基础和关键技术；第三章对系统需求与总体方案进行分析；第四章重点说明数据准备、抽帧策略、训练流程与模型构建；第五章分析后端任务调度、双智能体协作和多级回退机制；第六章介绍前端页面、结果可视化与证据链展示；第七章对系统测试、当前阶段结果与不足进行分析；第八章对全文进行总结，并给出后续研究方向。最后，论文附录中给出图片补充建议和制作指南，以便后续完成论文定稿与排版优化。

# 第2章 相关技术与理论基础
## 2.1 深度伪造技术概述
深度伪造内容通常是指利用深度学习模型对图像、视频或音频进行合成、替换、重建或属性编辑后生成的伪造媒体。面向人脸图像场景，常见操作包括身份替换、表情迁移、整脸生成与局部属性操控等。这类技术大多以生成对抗网络、编码器-解码器结构或扩散类生成框架为基础，其共同特征是能够从大规模数据中学习人脸纹理、姿态、光照和局部结构分布，从而生成肉眼难以快速区分的伪造结果。[5][7]

在检测角度看，深度伪造样本通常会在多个层面暴露异常痕迹。例如，换脸过程中可能出现边界过渡不自然、局部纹理与周围区域不一致、频域分布异常、压缩重编码后产生的不均匀伪影、光照与阴影不匹配、不同脸部区域清晰度不一致以及增强前后结果不稳定等。部分伪造还会在多脸场景中造成身份冲突，或在公众人物图像中带来更高的误用风险。正因如此，单一尺度、单一模态的判断往往难以覆盖所有情况，融合多类证据成为提升系统稳定性的关键思路。

## 2.2 人脸检测与对齐技术
人脸检测是本系统的前置步骤。只有在图像中准确定位人脸区域，后续的视觉分析、局部热力图生成和融合决策才具有明确对象。在当前仓库中，人脸检测模块采用了分层回退设计。优先级最高的是 InsightFace 检测器，该模块在具备依赖和缓存资产时能够提供更完整的人脸框、关键点和 embedding 信息，为后续公众人物候选匹配与更精确的人脸裁剪提供支持。若其不可用，则系统回退到 OpenCV Haar 级联检测器；若后者也不可用，则最终使用启发式候选框方案完成保底检测。

这种分层方案体现了本文的工程思想：对检测准确率要求高的部分优先使用高质量后端，但不把整个系统绑定在单一依赖上。从理论上说，基于深度学习的人脸检测器对复杂姿态和尺度变化有更强适应性，传统级联检测器在资源受限场景中更轻量，而启发式候选框虽然精度较低，但在演示和应急场景中仍能为系统提供最基础的目标定位能力。通过这种设计，本文将“检测是否成功”从单点依赖问题转化为“多级保障问题”。

## 2.3 卷积神经网络与 EfficientNet
卷积神经网络在图像分类和视觉识别领域长期占据核心地位，其优势在于能够通过局部感受野、参数共享和层级特征抽取自动学习纹理、边缘、形状和语义结构。对于深度伪造检测而言，卷积神经网络尤其适合捕捉局部纹理异常、融合边界伪影和细粒度结构差异。[2][8]

EfficientNet 系列模型在模型缩放方面提出了 compound scaling 思路，通过联合平衡网络深度、宽度和输入分辨率，实现了性能与复杂度之间的较好折中。[3] 本文选择 EfficientNet-B0 作为视觉分支主干，主要出于三个考虑：其一，模型体量相对适中，便于在毕业设计阶段完成训练与部署；其二，配合 ImageNet 预训练权重可以更快收敛，更适合中等规模的人脸图像训练任务；其三，该模型较适合作为服务化后端嵌入整个系统，而不是仅停留在离线实验中。

在当前系统中，EfficientNet-B0 负责对单张人脸裁剪图进行真假分类，并输出视觉风险得分。同时，系统还保留了 DCT、压缩痕迹、清晰度和增强一致性等统计特征，用于进一步解释和融合。这说明本文并没有将卷积网络视作唯一结论来源，而是将其视为证据链中的高质量视觉证据节点。

## 2.4 多特征融合与 XGBoost
深度伪造检测任务中，不同类型特征往往各有优势。视觉网络能够学习复杂纹理边界，频域特征能够反映重采样与压缩异常，图像质量指标可以揭示模糊与退化，一致性分数能够描述增强前后预测稳定性，语义分数则可以反映场景层面的冲突风险。如果仅依赖单一 CNN 输出，很难充分利用这些异构证据。因此，本文采用二阶段融合策略，在视觉分析基础上使用 XGBoost 对结构化特征做进一步判别。

XGBoost 是一种高效的梯度提升树模型，适合处理中小规模结构化特征，并能够学习非线性特征组合关系。[4] 与手工加权方案相比，XGBoost 不必预先写死每个特征的权重，而是通过训练数据学习更适合当前任务的决策边界。在本系统中，融合模型输入包括最大人脸分数、平均人脸分数、频域特征、质量分数、一致性分数、压缩痕迹、人脸数量和语义分数等八维特征。系统在训练后还将阈值与特征键顺序写入 metadata 文件，确保运行时与训练时的一致性。

## 2.5 大模型辅助分析与结构化证据链
近年来，大模型在复杂场景理解和自然语言解释方面展现出优势，但在高风险检测任务中，直接让大模型自由输出最终结论并不可取。本文在系统设计中没有把大模型当作唯一判断器，而是将其放在“受限策略选择”和“可选语义辅助分析”的位置上。也就是说，大模型只能在预定义策略空间中进行选择，并返回结构化说明，最终标签仍要经过阈值规则、冲突判断和复核条件约束。

与此相配套的，是系统对证据链的结构化组织。特征提取阶段和决策阶段都会生成带有 step、actor、timestamp、summary 和 details 字段的证据项，最终汇聚为 evidence chain 返回前端。这种设计兼顾了可解释性和可审计性：用户既能看到整体结论，也能查看每个关键步骤的中间信息、采用的后端以及是否发生回退。对于毕业设计答辩而言，这种结构化证据链不仅提升系统展示效果，也能够更清晰地说明代码实现与设计思路之间的对应关系。

# 第3章 系统需求分析与总体方案设计
## 3.1 系统需求分析
结合当前项目背景与毕业设计目标，系统需求可分为功能需求和非功能需求两个方面。功能需求方面，系统应支持用户上传单张人脸图像，自动创建检测任务并返回可追踪任务编号；应支持显示任务执行进度、执行阶段和当前智能体；应在任务完成后返回图像级结论、置信度、融合分数、复核原因、可视化产物和证据链；应支持查看历史任务列表，便于多样本比较和答辩演示；还应在无脸输入、低质量输入或异常环境下提供合理保护而非直接崩溃。

非功能需求方面，系统首先应具备稳定性，即在模型权重不完整、依赖未安装或外部服务不可用时仍能完成基础检测闭环；其次应具备可解释性，即不仅输出标签，还应输出支持该标签的结构化证据；再次应具备可扩展性，使后续能够替换检测器、视觉模型、融合模型或语义后端；最后应具备较好的交互性，保证前端页面能够清晰呈现检测历史、执行过程和结果细节。对于毕业设计系统而言，这些非功能指标的重要性并不低于单次分类精度，因为它们直接决定了系统能否作为完整原型稳定展示和持续迭代。

## 3.2 总体架构设计
从总体上看，当前系统可概括为“前端上传入口 + 后端异步任务调度 + 双智能体协作分析 + 多后端融合判断 + 结构化结果展示”的五层结构。前端层使用 Vue3 + Vite 实现上传交互、历史任务展示、结果页面渲染和证据链时间线展示；后端层使用 FastAPI 构建统一接口，并通过任务调度器管理检测任务生命周期；智能体层由特征提取智能体与决策智能体组成，分别承担取证与决策职责；模型与规则层由人脸检测、视觉分析、融合分析和语义分析等多个可回退后端组成；存储与展示层负责数据库持久化、产物文件管理、模型版本追踪和前端结果输出。

IMAGE: 图3.1 系统总体架构图。建议绘制内容包括“前端页面、API 路由、任务调度器、特征提取智能体、决策智能体、人脸检测后端、视觉模型后端、融合模型后端、语义分析后端、SQLite 数据库、产物存储目录”之间的关系。可使用 draw.io、ProcessOn 或 Visio 绘制，采用分层矩形框图形式，颜色区分前端层、调度层、智能体层和模型层。

## 3.3 双智能体协作流程设计
双智能体协作是本系统最核心的结构设计之一。与单体式流程相比，双智能体架构更容易实现职责分离与结果解释。特征提取智能体面对原始输入图像，其主要工作是完成感知与证据收集，包括读取图像、检测人脸、裁剪人脸、执行视觉分析、生成热力图与标注图以及组装输入摘要和初始证据链。此阶段输出的是一个结构化特征包，而不是最终结论。

决策智能体并不重新处理原图，而是消费特征提取智能体给出的结构化结果。它先构建任务画像，再基于预设规则或大模型策略后端选择适当策略；当任务画像显示风险或复杂度较高时，再触发语义辅助分析；最后，通过融合模块得到图像级分数，并结合双阈值规则输出最终标签。通过这种方式，系统实现了“先取证、后决策”的处理链，使前一阶段聚焦于证据质量，后一阶段聚焦于结论可靠性。这样的设计思路与实际代码中的 `FeatureExtractionAgent` 和 `DecisionAgent` 完全对应，具有较强的一致性。

## 3.4 三态输出与人工复核机制
传统二分类系统通常只输出“真实”或“伪造”，但在深度伪造检测场景中，这种设计容易在灰区样本上造成过度自信。当前系统采用双阈值策略：当融合分数低于真实阈值时，输出“疑似真实”；当高于伪造阈值时，输出“疑似伪造”；当位于中间灰区时，输出“需人工复核”。此外，若图像质量过低、无有效人脸、语义后端超时、视觉与语义证据明显冲突或当前策略本身为 review only，系统也会直接引导到人工复核。

这一设计在毕业设计场景中非常重要。首先，它表明系统不是追求表面上的“全自动”，而是承认实际输入中存在不确定性。其次，它能够降低错误判断的潜在风险，特别是在公众人物图像和多脸复杂场景中。再次，人工复核机制与证据链输出相结合，为后续人工判断提供了依据。最后，这种设计在答辩中也容易阐明系统的安全性与可靠性思路，即系统不会在不确定时强行给出结论，而是主动降级。

## 3.5 多级回退机制设计
多级回退机制是本系统在总体方案上区别于普通实验代码的重要特征。当前仓库中的关键能力几乎都实现了主方案与备选方案。人脸检测优先使用 InsightFace，其次回退至 OpenCV Haar，最后回退至启发式框选；视觉模型优先使用训练好的 EfficientNet-B0，失败时回退至启发式视觉评分；融合层优先使用 XGBoost，失败时回退到经验权重融合；语义层优先走远程视觉大模型，再尝试本地多模态模型，最后回退到规则语义后端。

这类设计本质上不是为了“掩盖模型未就绪”，而是为了保证系统级鲁棒性。毕业设计的系统实现往往面临资源、时间和环境差异的限制，如果一切能力都建立在单个模型和单个依赖之上，那么系统演示与后续迭代都会非常脆弱。通过多级回退，本文让系统在不同阶段都具备“能跑通、能展示、能解释”的能力，这也是本文在工程实现上的重要贡献之一。

# 第4章 数据集构建与模型训练流程设计
## 4.1 数据来源与选择理由
当前项目的视觉模型训练以 FaceForensics++ 为主要数据来源。该数据集在深度伪造检测领域具有较高代表性，能够为图像级和视频级伪造检测提供统一的数据基础。[1] 结合当前训练说明稿和脚本实现可知，系统优先关注 `original`、`Deepfakes`、`Face2Face`、`FaceSwap` 和 `NeuralTextures` 等核心子集，并将其作为当前阶段的主要训练资源。选择这一数据集的原因主要有三点：首先，它覆盖了典型的人脸篡改方式，便于训练具有代表性的视觉模型；其次，它的数据结构相对规范，便于编写自动化下载、抽帧和目录整理脚本；最后，它与当前系统的图像级检测任务高度契合，适合作为轻量级视觉模型的训练入口。

## 4.2 视频抽帧策略设计
系统并未直接使用完整视频序列训练，而是先将视频抽帧转换为图像级训练样本。根据 `extract_ffpp_frames.py` 的实现，当前默认采用 1 fps 的稀疏均匀抽帧策略，并限制每个视频最多保留 40 帧。选择这种策略而不是逐帧保留的原因在于：相邻视频帧高度相似，全量抽帧容易造成冗余样本过多；稀疏采样能够覆盖不同时间位置的人脸状态，同时控制数据规模和训练成本；在毕业设计阶段，这种方案更容易兼顾可重复性、数据管理效率和训练可执行性。

IMAGE: 图4.1 数据准备流程图。建议绘制“FF++ 子集下载 -> 视频抽帧 -> 真实/伪造目录映射 -> 训练集/验证集划分 -> 模型训练 -> metadata 固化 -> 后端接入”的流程。可使用泳道图或流程图形式，突出脚本文件名，如 `download_ffpp_subset.py`、`extract_ffpp_frames.py`、`prepare_visual_dataset.py`、`train_visual_model.py`、`export_fusion_features.py`、`train_fusion_model.py`。

## 4.3 训练集目录整理与工程化处理
完成抽帧后，系统还需要将样本组织为标准监督学习目录。根据 `prepare_visual_dataset.py` 的实现，来自 `original` 的图像被整理为 `real` 类，来自 `Deepfakes`、`Face2Face`、`FaceSwap` 和 `NeuralTextures` 的图像被归并为 `fake` 类，并进一步拆分为 `train/real`、`train/fake`、`val/real` 和 `val/fake`。脚本优先使用硬链接来减少存储冗余，在不支持硬链接的环境中再退回复制方式。这一细节体现了本文并非简单搬运数据，而是针对训练效率、空间使用和环境兼容性做了工程优化。

## 4.4 EfficientNet-B0 视觉模型训练设计
在视觉模型部分，系统采用 EfficientNet-B0 作为轻量级主干网络，并基于 ImageNet 预训练权重进行迁移学习。训练脚本通过 `ImageFolder` 和 `DataLoader` 构建标准图像分类训练流程，对输入图像执行统一尺寸缩放、张量化和归一化处理。随后，脚本将 EfficientNet-B0 的最后分类层替换为二分类线性层，使其输出 `real` 与 `fake` 两个类别。损失函数采用 `CrossEntropyLoss`，优化器采用 `AdamW`，每轮训练后会在验证集上计算 `accuracy`、`f1` 和 `auc` 等指标，并保留最佳 checkpoint。

从设计角度看，本文没有采用过于庞大的视觉主干，而是有意识地选择了更适合服务化与答辩展示的轻量级模型。这一选择体现了毕业设计场景下“准确率、训练成本、部署可行性”三者之间的平衡。通过 metadata 文件，系统进一步记录模型名称、输入尺寸、训练时间、数据版本、验证指标和阈值来源，为后续后端加载和结果展示提供依据。

## 4.5 XGBoost 融合模型训练设计
在融合模型部分，系统首先通过离线特征导出脚本复用在线检测逻辑，将每张图像转换为结构化特征行，再送入 XGBoost 训练器。当前定义的融合特征包括最大人脸风险、平均人脸风险、频域异常均值、图像质量均值、一致性均值、压缩痕迹均值、人脸数量以及语义分数等八个字段。该设计反映出本文对深度伪造检测的理解并不局限于视觉纹理，而是将检测任务视为多源证据融合问题。

XGBoost 训练脚本在读取 JSONL 样本后，会检验字段完整性并构造特征矩阵与标签数组，然后初始化 `XGBClassifier` 完成训练。若存在验证集，则在验证集上计算分类指标，并继续对真实阈值和伪造阈值进行校准；若验证集尚未齐备，则使用默认阈值作为临时方案。训练完成后，系统输出模型文件与 metadata 文件，以便后续后端接入和模型版本追踪。这样的设计表明本文的训练工作不是孤立的学术实验，而是与部署系统紧密耦合的工程闭环。

## 4.6 阈值策略与 metadata 固化
当前系统的一大特点是将阈值来源显式纳入运行逻辑，而不是在代码里硬编码一个固定的 0.5。模型注册表在运行时会优先读取融合模型 metadata 中的阈值信息；若不存在，则读取视觉模型 metadata；若两者都没有，则退回系统默认阈值。如此一来，系统不仅能进行三态判断，还能向前端明确说明当前阈值来自哪里，这在结果解释和系统调试中都十分重要。

metadata 的作用还不止于阈值。它记录了训练配置、数据版本、特征键顺序、评估指标和模型标识，使得前后端在运行时能够理解当前究竟加载了什么模型、模型是否完整、显示何种版本信息以及如何组织输入特征。可以说，metadata 是连接训练链路与部署链路的重要桥梁，也是本文区别于“只训练一个 pt 文件”的关键实现。

# 第5章 后端核心模块设计与实现
## 5.1 API 路由与任务调度
后端使用 FastAPI 构建统一 API 接口，主要路由包括健康检查、系统状态查询、任务创建、任务状态查询、任务结果查询以及检测产物访问等。与许多同步式预测接口不同，本系统在用户上传图像后首先创建一个检测任务，并将任务信息写入 SQLite 数据库；随后再将任务交给线程池异步执行。这样设计的原因有两点：一是可以在前端显示完整的执行阶段和进度信息，二是便于保存历史记录、追踪异常并支持演示中的多任务切换。

调度器 `DetectionOrchestrator` 是后端主流程的核心。它负责任务创建、文件保存、状态推进、智能体调用和异常处理。当任务运行时，系统依次推进 `queued`、`preprocessing`、`feature_extraction`、`decision`、`completed` 等阶段；若出现异常，则记录失败原因并保留任务上下文信息。对毕业设计而言，这种任务化调度不仅增强了工程结构的清晰度，也为论文中的流程分析提供了很好的叙述基础。

## 5.2 存储模块与数据库设计
系统在存储层同时管理两类信息。一类是结构化任务信息，保存在 SQLite 数据库 `tasks` 表中，包括任务 ID、文件名、当前状态、阶段、进度、当前智能体、输入路径、最终结果 JSON 和错误信息等。另一类是文件型产物，保存在任务专属目录下，包括用户上传原图、原图标注图、人脸裁剪图、热力图及其他可视化文件。通过这种设计，系统既能实现轻量级部署，又能支持历史任务回放与结果复查。

从论文写作角度看，这一部分有助于体现系统工程能力。因为一个成熟的检测系统不应仅在内存中短暂产生结果，而应能持久保存任务记录与关键中间产物。数据库让前端的历史任务页成为可能，而文件存储让热力图、人脸裁剪和标注图可以被统一访问与展示，这些内容都能在毕业论文中体现系统级思考。

## 5.3 特征提取智能体设计与实现
特征提取智能体对应代码中的 `FeatureExtractionAgent`。它首先加载输入图像，然后调用人脸检测后端获取人脸框。若未检测到有效人脸，则系统会生成对应证据条目和标注图，并将任务转入受限决策路径，而不是直接报错。若检测成功，系统则遍历每一张人脸，完成裁剪、对齐、热力图生成、视觉风险分析以及公众人物候选匹配，并将每一张脸的结果封装为 `FaceResult` 对象。最终，特征提取智能体输出 faces、artifacts、evidence、input summary 和 model versions 等信息，为决策智能体提供完整输入。

IMAGE: 图5.1 特征提取智能体流程图。建议绘制“图像读取 -> 人脸检测 -> 无脸分支 / 有脸分支 -> 人脸裁剪 -> 视觉分析 -> 图库匹配 -> 热力图生成 -> 证据条目构建 -> 输出结构化特征包”的流程，并突出 `FeatureExtractionAgent.run()` 的主线。

从代码实现可以看出，该智能体的核心定位不是“直接给出真假结论”，而是“把原始图像转化为结构化证据”。这一定位非常重要。因为一旦前端、融合模型或语义模块需要更多信息，系统可以直接在该阶段扩展，而不必重写整个检测管线。换句话说，特征提取智能体承担了系统中的感知层角色。

## 5.4 决策智能体设计与实现
决策智能体对应代码中的 `DecisionAgent`。它接收特征提取智能体输出后，首先解析 faces、evidence、artifacts、input summary 和 model versions，并从模型注册表中读取当前阈值策略。若输入中没有有效人脸，则直接构建 review only 结果；若存在有效人脸，则进一步构建任务画像 `TaskProfile`，计算最大人脸分数、平均人脸分数、质量分数、公众人物候选和冲突水平等指标。

在任务画像基础上，决策智能体进入策略选择阶段。系统优先使用大模型策略后端，当大模型不可用或返回异常时，则回退到规则策略。策略空间被严格限制为 `visual_only`、`visual_plus_semantic` 和 `review_only` 三类，不允许大模型自由决定超出边界的执行方式。这种设计体现了对智能体行为可控性的重视。随后，若策略要求语义分析，系统再通过语义后端生成语义分数和语义标记；之后，将视觉分数、统计特征和语义分数送入融合层，得到最终的图像级融合分数，并通过双阈值策略和若干保护条件确定输出标签与复核原因。整个过程会持续向 evidence chain 中写入新的证据条目，并在最终结果中整合为统一输出对象。

## 5.5 多级回退机制实现分析
多级回退机制在后端中的实现并不是一个单独函数，而是体现在各类 manager 与 backend 的协同逻辑中。FaceDetectorManager 负责在人脸检测后端之间顺序尝试，VisualBackendManager 负责在 EfficientNet 与启发式视觉后端之间切换，FusionBackendManager 负责在 XGBoost 与启发式融合之间切换，SemanticBackendManager 负责在远程视觉大模型、本地多模态模型和规则后端之间切换。只要高优先级方案失败，系统就自动进入下一层。

这种机制带来的直接收益是：即使真实模型资产尚未完全到位，系统仍可以输出可用的检测结果、热力图、证据链和前端页面效果。对于毕业设计原型系统来说，这种特性尤为重要，因为它保证了系统在不同运行环境中的适应能力，也为后续逐步替换和补强真实模型提供了稳定基础。

# 第6章 前端界面设计与可解释性展示
## 6.1 前端整体页面结构
前端采用 Vue3 + Vite 实现，主页面整体划分为左右两栏。左侧区域用于图像上传和历史任务展示，右侧区域用于显示当前任务的执行阶段、融合结论、图像产物、证据链和模型信息。这种布局遵循“输入与管理在左、结果与解释在右”的思路，使用户能够在一个页面内完成任务创建、历史切换和结果查看，适合答辩时进行流程演示。

## 6.2 上传区与历史任务区设计
上传区支持两种输入方式：用户本地选择图片和使用预置示例图。当前实现中，前端在用户提交后会将文件封装为 `FormData` 发送给任务创建接口，并在收到任务 ID 后立即进入轮询状态。历史任务区则显示最近任务的文件名、当前状态、阶段与置信度等信息，并支持点击切换查看。由于所有任务信息都持久化在数据库中，因此历史任务区并不是简单前端缓存，而是对系统任务管理能力的直接体现。

IMAGE: 图6.1 系统首页截图。建议在系统联调完成后截取一张包含“上传区域、历史任务列表、状态栏”的首页截图，用作系统交互界面展示。截图应分辨率清晰，建议使用浏览器全屏模式并隐藏不相关桌面元素。

## 6.3 结果面板设计
结果面板是前端可解释性展示的核心区域。其顶部会给出当前任务状态、执行阶段和进度；随后展示图像级标签、融合分数、置信度和复核原因；再向下显示视觉摘要、语义摘要、原图标注图、人脸裁剪图和热力图；最后给出模型版本信息和阈值来源。可以说，结果面板不仅是视觉展示区域，也是系统“把内部逻辑显式化”的界面载体。

从论文角度看，这一设计有两个亮点。其一，系统并未将判定结果藏在接口内部，而是把决策过程中的重要元素以页面形式透明呈现。其二，页面中的各个模块与后端返回结构一一对应，说明前后端联动并非临时拼接，而是围绕统一数据结构设计的。

## 6.4 证据链时间线设计
证据链时间线模块对应前端的 EvidenceTimeline 组件，其作用是将后端返回的 evidence chain 按时间顺序展示为“输入加载—人脸检测—特征提取—策略选择—语义分析—最终决策”的链式过程。如果某一证据节点包含 `thinking_chain`，前端会将其单独高亮，并支持折叠展开查看。这种展示方式使用户不仅能看到“系统做了什么”，还能看到“系统为什么这样做”，有效增强了原型系统的可解释性。

## 6.5 人机协同复核体验设计
当前系统虽然能够自动输出三态结果，但并不试图取代人类复核流程。当样本处于灰区、多脸、低质量或语义冲突场景时，系统会明确提示“需人工复核”，而不是继续掩盖不确定性。这种设计理念也体现在前端：页面会明确显示复核原因和证据基础，帮助使用者快速理解问题来源。对于深度伪造检测系统来说，这种人机协同机制比单纯追求更高的自动化程度更具现实意义。

# 第7章 系统测试与结果分析
## 7.1 测试目标与测试环境
结合当前仓库状态，本文测试重点并不放在夸大式的离线指标展示，而是围绕功能完整性、系统稳定性、回退链路有效性和可解释性展示效果展开。测试环境主要包括本地前后端联调环境、SQLite 任务存储环境以及若干模型资产缺失或服务不可用的回退场景。由于正式的跨数据集定量实验仍需在 FF++、Celeb-DF、DF40 等完整资产到位后进一步开展，因此本章更加关注当前阶段已经完成且可由仓库验证的部分。

## 7.2 功能测试
在功能测试方面，系统首先验证单张图像上传、任务创建、异步执行、结果查询和历史任务刷新是否正常。测试结果表明，前端能够成功发起任务并接收任务 ID，后端调度器能够依次推进任务阶段，并在任务完成后将结果持久化到数据库。用户点击历史任务时，系统也能够重新获取对应任务的状态和结果对象，说明“上传—执行—保存—展示—回看”的完整链路已经打通。

其次，系统验证了可视化产物生成是否正常。当前仓库中的特征提取模块能够在有人脸场景下生成原图标注图、人脸裁剪图和热力图，在无人脸场景下生成相应的提示型标注图。页面能够基于产物 URL 正常读取这些图像并展示在结果面板中。这说明系统不仅能输出文字型结论，还具备较完整的视觉证据呈现能力。

## 7.3 回退链路测试
回退链路测试是本系统最重要的测试内容之一。根据 README、训练说明稿和代码结构可知，当前仓库默认场景下可能并不总是具备完整的真实模型资产，例如 InsightFace 的缓存、EfficientNet 权重、XGBoost 融合模型或本地多模态模型可能处于未完全就绪状态。对此，系统通过 manager 机制自动选择可用后端。当高优先级后端不可用时，系统不会直接报错退出，而是自动回退到下一层方案继续执行。

从测试角度看，这一机制有效保障了系统演示稳定性。无论是人脸检测回退、视觉模型回退、融合模型回退还是语义分析回退，前端依然能够收到结果对象和证据链，并在页面中显示当前实际启用的后端名称。这说明多级回退并非表面设计，而是真正贯穿于运行链路之中的核心机制。

IMAGE: 图7.1 回退机制验证示意图。建议绘制一个分层决策图，展示 detector、visual、fusion、semantic 四条回退链，并在图中注明“优先后端失败后自动切换至备选后端”的条件关系。可使用树状图或分层箭头图，便于在论文中直观说明鲁棒性设计。

## 7.4 可解释性与交互效果分析
系统的可解释性主要体现在三方面。第一，结果面板中能够同时显示标签、融合分数、置信度、复核原因、模型版本和阈值来源，使结果具有可追溯性。第二，证据链时间线能够把内部执行过程显式展示给使用者，降低“黑盒感”。第三，热力图、人脸裁剪和原图标注图为视觉判断提供了直观支撑，有助于答辩中解释系统为何给出某个结果。

从交互体验来看，前端采用任务化模式而非阻塞式等待模式，使得用户能够清楚看到系统正在做什么，并支持在多次任务之间切换查看结果。这对于毕业设计展示尤为重要，因为评审教师往往希望观察系统的执行链路，而不是只看最终标签。

## 7.5 当前阶段结果分析
从当前仓库状态分析，系统已经完成了工程闭环、双智能体协作逻辑、多级回退机制、前后端联调、训练脚本与模型接入准备等关键工作。对于毕业设计而言，这意味着系统原型已经具备较高完成度，能够稳定演示其核心思想。与此同时，也需要客观指出：当前仓库默认状态强调的是“可运行、可解释、可回退”的原型系统，而不是已经完成全部正式实验指标的最终论文版本。特别是在跨数据集泛化验证、完整权重加载、语义特征离线注入和统一 benchmark 结果汇总方面，后续仍有补充空间。

## 7.6 存在不足与原因分析
首先，当前论文阶段尚未给出完备的正式量化实验表格，尤其是跨数据集评估、消融实验和不同回退模式的定量对比结果仍需在更多资产到位后补齐。其次，融合模型训练流程中对语义分数的离线导出目前仍使用占位值或较轻量方案，这意味着在线语义增强能力尚未完全在离线训练层面体现。再次，当前系统主要面向图像级 Deepfake 人脸检测，对视频时序建模和跨帧一致性利用仍有限。最后，虽然启发式回退机制提升了稳定性，但其精度天然无法与真实训练模型等同，因此后续仍应以真实模型资产替换为主、以回退逻辑保底为辅。

# 第8章 总结与展望
## 8.1 全文总结
本文围绕当前代码仓库实现，设计并分析了一套面向图像级 Deepfake 人脸检测的智能体协作系统。与单纯输出真假标签的实验型代码不同，本文从系统工程视角出发，完成了前端上传与结果展示、后端任务调度与状态管理、特征提取智能体与决策智能体协作、多后端视觉分析与融合判断、结构化证据链生成以及多级回退机制设计等一整套原型系统实现。

在数据与模型方面，本文基于 FaceForensics++ 构建了从下载、抽帧、样本整理到模型训练的工程流程，训练了 EfficientNet-B0 视觉模型和 XGBoost 融合模型，并通过 metadata 将训练资产与运行时逻辑连接起来。在运行机制方面，系统通过三态输出与人工复核机制处理不确定样本，通过回退设计应对模型资产缺失与环境异常，通过前端证据链与可视化产物提升结果透明度。从毕业设计目标来看，本文不仅完成了检测模型训练的相关工作，更重要的是完成了一个可运行、可展示、可解释、可扩展的原型系统。

## 8.2 后续工作展望
后续工作可从四个方向继续推进。第一，补充正式定量实验，在完整资产到位后完成 FF++、Celeb-DF、DF40 等数据集上的准确率、F1、AUC、复核率和时延对比，并进一步加入消融实验。第二，完善离线语义特征导出流程，使融合模型训练能够直接利用真实语义信号，进一步增强复杂场景下的判别能力。第三，扩展到视频级检测，引入跨帧一致性、时间注意力或时序建模机制，提高对短视频 Deepfake 的适应能力。第四，优化论文定稿阶段的图片与实验展示，通过系统架构图、页面截图、热力图案例、训练流程图和回退链路图进一步提升论文完整性和可读性。

# 致谢
在本次毕业设计与论文撰写过程中，我得到了导师、同学以及项目资料提供者的帮助与支持。导师在选题方向、系统设计思路、论文撰写方法和阶段性推进方面给予了耐心指导；同学和朋友在系统测试、界面体验反馈和资料整理方面提供了积极帮助。与此同时，本项目所使用的开源框架、公开数据集和相关研究成果为系统实现提供了重要支撑。在此，向所有给予帮助和支持的老师、同学及研究者表示衷心感谢。

# 参考文献
[1] Andreas Rossler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Niessner. FaceForensics++: Learning to Detect Manipulated Facial Images. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

[2] Darius Afchar, Vincent Nozick, Junichi Yamagishi, Isao Echizen. MesoNet: a Compact Facial Video Forgery Detection Network. 2018 IEEE International Workshop on Information Forensics and Security, 2018.

[3] Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. Proceedings of the 36th International Conference on Machine Learning, 2019.

[4] Tianqi Chen, Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.

[5] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. Generative Adversarial Nets. Advances in Neural Information Processing Systems, 2014.

[6] Luisa Verdoliva. Media Forensics and DeepFakes: An Overview. IEEE Journal of Selected Topics in Signal Processing, 2020, 14(5): 910-932.

[7] Ruben Tolosana, Ruben Vera-Rodriguez, Julian Fierrez, Aythami Morales, Javier Ortega-Garcia. Deepfakes and Beyond: A Survey of Face Manipulation and Fake Detection. Information Fusion, 2020, 64: 131-148.

[8] Yuezun Li, Siwei Lyu. Exposing DeepFake Videos By Detecting Face Warping Artifacts. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2019: 46-52.

[9] OpenCV Team. OpenCV: Open Source Computer Vision Library. https://opencv.org/

[10] FastAPI. FastAPI Framework Documentation. https://fastapi.tiangolo.com/

[11] Vue.js Team. Vue.js Documentation. https://vuejs.org/

[12] PyTorch Team. PyTorch Documentation. https://pytorch.org/

# 附录A 关键接口与核心数据结构说明
## A.1 后端核心接口
当前系统主要接口包括：`GET /api/health` 用于健康检查，`GET /api/status` 用于返回 detector、visual、fusion、semantic、llm 等模块的运行状态，`GET /api/tasks` 用于返回历史任务列表，`POST /api/tasks` 用于创建检测任务，`GET /api/tasks/{task_id}` 用于获取任务状态和进度，`GET /api/tasks/{task_id}/result` 用于获取最终结果，`GET /api/tasks/{task_id}/artifacts/{name}` 用于获取热力图、裁剪图与标注图等产物。

## A.2 关键数据结构
当前系统在结果组织上使用结构化对象。`FaceResult` 主要描述单张人脸的检测框、关键点、视觉风险分数、质量分数、频域分数、压缩分数、一致性分数和对应产物；`TaskProfile` 描述图像级任务画像，如人脸数量、最大风险、平均风险、冲突水平和是否需要语义分析；`DetectionResult` 则负责组织图像级标签、融合分数、置信度、复核原因、语义标记、证据链、模型版本和前端展示所需信息。通过这些对象，系统将底层执行过程转化为前后端都可以消费的统一结构。

# 附录B 论文图片补充建议与制作指南
## B.1 建议补充的图片清单
为了使论文篇幅达到更规范的展示效果，并增强阅读体验，建议至少补充以下 8 类图片：

1. 系统总体架构图：展示前端、后端、双智能体、多后端与存储层之间的关系。
2. 数据准备流程图：展示 FF++ 下载、抽帧、训练集整理、训练、部署接入的整体流程。
3. 特征提取智能体流程图：展示从图像输入到人脸检测、裁剪、视觉分析、热力图生成和结构化输出的过程。
4. 回退机制示意图：展示 detector、visual、fusion、semantic 四条回退链。
5. 前端首页截图：展示上传区、历史任务区与结果页联动。
6. 检测结果页截图：展示标签、融合分数、热力图、人脸裁剪图、证据链和模型信息。
7. 热力图案例图：选取一张典型伪造样本，对比原图、人脸裁剪图与热力图。
8. 证据链时间线截图：展示结构化证据如何在页面中按时间线展开。

## B.2 各类图片的制作方法
系统架构图、流程图和回退机制图建议使用 draw.io、ProcessOn 或 Visio 绘制。绘制时应保持统一字体和配色，使用矩形框表示模块、箭头表示数据流，图中尽量保留与代码一致的模块命名，如 `FeatureExtractionAgent`、`DecisionAgent`、`VisualBackendManager` 等。页面截图和结果图建议在前后端联调稳定后，用浏览器全屏截图方式采集，确保图像清晰，不出现无关界面元素。热力图案例建议从系统已经生成的 artifact 中选择典型样本，按“原图—裁剪图—热力图”三联图形式排版。若需要制作训练流程图，可在图中直接标注脚本文件名，增强论文与代码之间的对应关系。

## B.3 图片排版建议
按照撰写规范，图题应放在图下方，采用“图X.Y 图名”的形式。例如，“图3.1 系统总体架构图”“图4.1 数据准备流程图”“图6.1 系统首页截图”。若图中包含多个子图，可使用“(a)(b)(c)”进行标注。截图类图片建议统一宽度并保持页面居中，流程图和架构图建议使用矢量绘图工具导出为高清 PNG，避免模糊。论文定稿时，每章至少安排 1 至 2 幅图，既有助于解释系统，也有利于提升版面层次与篇幅表现。
