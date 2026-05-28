# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

UNPACKED = Path(r"F:\AgentDeepfakeFaceDet\AgentDeepfakeFaceDet\tmp_thesis_v3_unpacked\word\document.xml")


REPLACEMENTS = {
    "结合当前仓库状态，本文测试重点并不放在夸张的离线 benchmark 数值展示，而是围绕功能完整性、系统稳定性、回退链路有效性和可解释展示能力展开。测试环境主要包括本地前后端联调环境、SQLite 任务存储环境以及若干模型资产缺失或服务不可用的回退场景。由于正式的跨数据集定量实验仍需在 FF++ 、Celeb-DF 和 DF40 等完整资产到位后进一步开展，因此本章更关注当前阶段已经完成且可被仓库直接验证的部分。":
        "结合当前仓库状态，本文测试重点不仅包括功能完整性、系统稳定性、回退链路有效性和可解释展示能力，也补充了面向外部数据集的阶段性定量评测。测试环境主要包括本地前后端联调环境、SQLite 任务存储环境、模型权重接入环境以及 Celeb-DF 测试集抽帧后的离线批量评测场景。考虑到 LLM 策略模块在批量评测阶段会引入额外的策略波动和提示词规则不一致问题，本文在定量实验中关闭 LLM 策略选择，仅保留视觉模型、融合模型与规则式决策链路，以保证评测过程的可复现性和实验结果的一致性。",
    "从当前仓库状态出发，可以较为客观地得出以下结论。第一，系统工程闭环已经完成，具备稳定演示基础。第二， 双智能体协作主线、结构化证据链、三态输出与人工复核机制已经落地。第三， 视觉模型训练、融合模型训练及 metadata 接入流程已经具备可复用形态。第四， 前后端联调能力和历史任务追踪能力已经实现。第五， 多级回退机制使系统在资产不完备条件下依然具备可运行性。":
        "结合当前联调状态与外部测试结果，可以较为客观地得出以下结论。第一，系统工程闭环已经完成，具备稳定演示基础，上传、异步任务、结果展示、历史任务回看和证据链输出均已打通。第二，双智能体协作主线、结构化证据链、三态输出与人工复核机制已经落地，并能够在前端页面中被直观展示。第三，视觉模型训练、融合模型训练、阈值固化和 metadata 接入流程已经具备可复用形态。第四，在关闭 LLM 策略干扰并统一融合阈值与质量门控阈值后，系统在 Celeb-DF 平衡测试子集上完成了 100 张图片的批量评测，其中真实样本和伪造样本各 50 张，总体准确率为 57.0%，人工复核率为 5.0%，真实样本识别准确率为 36.0%，伪造样本识别准确率为 78.0%。这一结果说明当前系统已经摆脱了早期“大量 review 无法落判”的问题，但对真实样本仍存在较明显的误报倾向。第五，多级回退机制使系统在资产不完备条件下依然具备可运行性，这为毕业设计阶段的系统展示和后续迭代提供了现实基础。",
    "与此同时，也必须强调：当前仓库更接近“阶段一完成、阶段二待补齐正式实验”的原型系统，而不是已经完成全部正式 benchmark 的终版研究系统。论文写作必须据此保持客观表述，避免将阶段性成果夸写为全面完成。":
        "与此同时，也必须强调：当前仓库虽然已经补充了 Celeb-DF 外部测试集上的阶段性定量评测，但整体上仍更接近“原型系统已完成、正式 benchmark 仍待进一步扩充”的毕业设计实现，而不是已经完成多数据集全面对比与充分消融实验的终版研究系统。论文写作必须据此保持客观表述，即一方面如实呈现当前系统已经完成的训练、接入、联调与评测工作，另一方面也要明确指出现阶段结果仍具有外部数据集规模有限、真实样本误报偏高、部分模块依赖工程性开关控制等约束。",
    "当前系统仍存在若干不足。首先， 正式的跨数据集量化实验尚不完整，尤其是针对FF++、Celeb-DF 与 DF40 的统一对比表、消融实验和多种回退模式下的定量结果仍需在更多资产到位后补齐。其次，融合模型训练阶段的 semantic_score 仍未完全以真实离线语义特征支撑，这意味着“多模态特征参与训练”的闭环尚未彻底完成。再次， 当前系统主要面向图像级检测，对视频时序一致性建模、跨帧关系分析和视频级证据组织的支持仍然不足。最后， 启发式回退虽然保障了可用性，但其精度天花板无法与完整训练模型等同，因此后续仍应以真实模型资产逐步替换保底后端。":
        "当前系统仍存在若干不足。首先，从本次 Celeb-DF 平衡测试结果看，系统虽然将人工复核率压缩到了较低水平，但真实样本误报仍然偏高，100 张测试中共出现 30 个 false positive，而 false negative 为 8 个，说明当前模型更倾向于将样本判为伪造，跨数据集泛化能力仍然有限。其次，正式的跨数据集量化实验仍不完整，尤其是针对 FF++、Celeb-DF 与 DF40 的统一对比表、消融实验和更大样本规模下的稳健性验证仍需继续补齐。再次，融合模型训练阶段的 semantic_score 尚未完全由真实离线语义特征驱动，多模态特征参与训练的闭环仍未彻底完成。第四，系统在批量实验中暴露出 LLM 策略提示词与本地配置阈值不一致的问题，说明大模型辅助模块在工程配置一致性方面仍需进一步修正。最后，当前系统主要面向图像级检测，对视频时序一致性建模、跨帧关系分析和视频级证据组织的支持仍然不足，因此后续仍需从模型泛化、策略一致性和视频级扩展三个方向继续完善。",
    "结合当前不足，后续工作可从四个方向继续推进。第一，补齐正式定量实验，形成更完整的跨数据集 benchmark 与消融分析。第二， 完善离线语义特征导出流程，使融合模型训练能够真正利用语义信号。第三， 将系统扩展到视频级检测，引入时序建模与跨帧一致性分析。第四， 进一步丰富论文中的截图、流程图和案例图， 增强最终定稿的可读性与版面完整度。":
        "结合当前不足，后续工作可从四个方向继续推进。第一，继续补齐正式定量实验，形成覆盖 FF++、Celeb-DF 等数据集的统一 benchmark、阈值调优对比表和消融分析结果，并进一步验证不同样本规模下的稳定性。第二，针对当前真实样本误报偏高的问题，继续开展阈值校准、训练集扩充、外部数据域适配和误判样本针对性分析，以提升系统对真实样本的保持能力。第三，修复 LLM 策略模块中提示词规则与本地配置脱节的工程问题，使演示模式与实验模式在策略逻辑上保持一致，并进一步完善离线语义特征导出流程。第四，将系统从图像级检测扩展到视频级检测，引入时序建模、跨帧一致性分析和更完整的视频级证据组织，同时补充论文中的案例图、截图、流程图和对比表，增强最终定稿的可读性与完整性。",
}


def set_paragraph_text(p: ET.Element, text: str) -> None:
    ppr = p.find(f"{W}pPr")
    first_run = p.find(f"{W}r")
    run_props = None
    if first_run is not None:
        rpr = first_run.find(f"{W}rPr")
        if rpr is not None:
            run_props = copy.deepcopy(rpr)

    for child in list(p):
        if child.tag != f"{W}pPr":
            p.remove(child)

    r = ET.Element(f"{W}r")
    if run_props is not None:
        r.append(run_props)
    t = ET.SubElement(r, f"{W}t")
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    p.append(r)


def main() -> None:
    tree = ET.parse(UNPACKED)
    root = tree.getroot()
    paragraphs = root.findall(".//w:body/w:p", NS)
    replaced = 0
    for p in paragraphs:
        text = "".join((t.text or "") for t in p.findall(".//w:t", NS)).strip()
        if text in REPLACEMENTS:
            set_paragraph_text(p, REPLACEMENTS[text])
            replaced += 1
    tree.write(UNPACKED, encoding="utf-8", xml_declaration=True)
    print(f"replaced={replaced}")


if __name__ == "__main__":
    main()
