# 公共人物演示图库

将公共人物演示图片或预计算嵌入放在这个目录下，用于答辩时演示“身份置换风险”场景。

- `gallery.json` 里的 `embedding` 字段可填写与系统 `feature_embedding` 维度一致的向量。
- 如果后续接入 `InsightFace` 识别嵌入，建议补一个 `build_gallery.py` 脚本批量生成嵌入并回写到 `gallery.json`。
- 当前仓库只预置了人物条目，不直接分发人脸图库图片。
