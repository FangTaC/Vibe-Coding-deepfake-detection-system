from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "质量阈值调优在线复验记录.docx"


def paragraph(text: str, *, style: str | None = None) -> str:
    text = escape(text)
    ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
    return (
        "<w:p>"
        f"{ppr}"
        "<w:r><w:rPr><w:rFonts w:ascii=\"宋体\" w:hAnsi=\"宋体\" w:eastAsia=\"宋体\"/>"
        "<w:sz w:val=\"24\"/><w:lang w:eastAsia=\"zh-CN\"/></w:rPr>"
        f"<w:t xml:space=\"preserve\">{text}</w:t></w:r>"
        "</w:p>"
    )


def build_document_xml() -> str:
    lines: list[str] = []
    lines.append(paragraph("质量阈值调优在线复验记录", style="Title"))
    lines.append(paragraph("实验编号：2026-04-27-quality-threshold-online-verify-01"))
    lines.append(paragraph(""))

    sections: list[tuple[str, list[str]]] = [
        (
            "一、实验目的",
            [
                "验证 Celeb-DF 外部测试集中大量 review_required 是否主要由系统质量门控阈值过高导致，并观察在下调质量阈值后的在线 API 批量评测效果。",
            ],
        ),
        (
            "二、已确认前提",
            [
                "1. 融合模型阈值已实际生效，threshold_source 为 fusion_metadata。",
                "2. 当前在线后端实际阈值为 true_threshold=0.05，fake_threshold=0.50。",
                "3. 单样本排查结果显示，大量样本进入 review 的主要原因是 strategy_selection 命中了 review_only。",
                "4. 对应的 review_reason 为：图像质量过低或输入不满足系统适用范围。",
            ],
        ),
        (
            "三、本轮配置调整",
            [
                "仅调整质量门控阈值。",
                "修改前：low_quality_threshold = 0.35",
                "修改后：low_quality_threshold = 0.15",
                "说明：融合阈值保持不变，本轮实验仅用于验证质量门控阈值对 review_required 的影响。",
            ],
        ),
        (
            "四、数据集与样本设置",
            [
                "数据集：Celeb-DF-v2 抽帧图像",
                "输入目录：D:\\datasets\\celebdf_eval_images",
                "本轮样本数：40",
                "真实样本：20",
                "伪造样本：20",
                "评测方式：调用现有单图检测 API 逐张评测",
            ],
        ),
        (
            "五、实验结果摘要",
            [
                "total_images = 40",
                "correct = 26",
                "incorrect = 14",
                "accuracy = 0.65",
                "real_accuracy = 0.45",
                "fake_accuracy = 0.85",
                "predicted_real = 11",
                "predicted_fake = 27",
                "false_positive = 10",
                "false_negative = 2",
                "review_required = 2",
                "failed_or_timeout = 0",
            ],
        ),
        (
            "六、结果分析",
            [
                "1. 下调质量阈值后，review_required 从先前的大量出现下降到 2/40，即 5%。",
                "2. 系统已经能够对绝大多数样本给出明确的二分类结果，说明高 review 现象的主因确实是质量门控阈值过高。",
                "3. 当前系统对 fake 更敏感，fake_accuracy 达到 0.85，但 real_accuracy 仅为 0.45，说明真实样本误报仍偏高。",
                "4. 这表明当前系统的主要剩余问题已从“大量人工复核”转为“真实样本误报偏高”。",
            ],
        ),
        (
            "七、可用于论文的结论",
            [
                "1. 在 Celeb-DF 外部测试集上，系统默认质量门控策略会导致大量样本提前进入人工复核流程。",
                "2. 将 low_quality_threshold 从 0.35 下调至 0.15 后，系统在线评测复核率显著下降至 5%，总体准确率提升至 65%。",
                "3. 该结果说明，系统最终检测表现不仅受融合模型影响，也显著受工程化门控策略影响。",
            ],
        ),
        (
            "八、需要继续保存的实验材料",
            [
                "1. D:\\datasets\\celebdf_balanced_eval.csv",
                "2. D:\\datasets\\celebdf_balanced_eval.json",
                "3. 本轮 PowerShell 终端输出",
                "4. 本轮运行命令文本",
                "5. 修改前后的 config.py 截图",
                "6. 若干张误判样本与正确识别样本图片",
            ],
        ),
        (
            "九、下一步建议",
            [
                "1. 继续做一轮 100 张平衡样本复验（50 real + 50 fake），验证 65% 的结果是否稳定。",
                "2. 在复验结果中单独统计 false_positive 和 false_negative，重点分析真实样本误报原因。",
                "3. 将“默认阈值结果”“仅校准 xgboost 阈值结果”“再下调质量阈值后的结果”整理成对比表，作为论文中的系统优化过程。",
                "4. 从结果集中各挑选若干张正确 real、正确 fake、误判 real、边界样本，作为论文案例图。",
                "5. 如果 100 张复验结果仍然偏向 fake，可再考虑后续做模型重训或域适配，但这一步不必阻塞当前论文初稿。",
            ],
        ),
    ]

    for title, paras in sections:
        lines.append(paragraph(title, style="Heading1"))
        for item in paras:
            lines.append(paragraph(item))
        lines.append(paragraph(""))

    body = "".join(lines)
    sect = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        f"<w:body>{body}{sect}</w:body></w:document>"
    )


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

DOCUMENT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/>
      <w:sz w:val="24"/>
      <w:lang w:eastAsia="zh-CN"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/>
      <w:b/><w:sz w:val="36"/><w:lang w:eastAsia="zh-CN"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="宋体" w:hAnsi="宋体" w:eastAsia="宋体"/>
      <w:b/><w:sz w:val="28"/><w:lang w:eastAsia="zh-CN"/>
    </w:rPr>
  </w:style>
</w:styles>
"""

CORE_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>质量阈值调优在线复验记录</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>
"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
</Properties>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("_rels/.rels", ROOT_RELS)
        zf.writestr("word/document.xml", build_document_xml())
        zf.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
        zf.writestr("word/styles.xml", STYLES)
        zf.writestr("docProps/core.xml", CORE_XML)
        zf.writestr("docProps/app.xml", APP_XML)


if __name__ == "__main__":
    main()
