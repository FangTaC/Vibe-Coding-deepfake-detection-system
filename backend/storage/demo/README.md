# Demo 联调目录

该目录保存阶段一答辩演示所需的样本、联调报告和运行状态表。

## 结构

- `samples/`：演示样本图
- `reports/integration_status.json|md`：当前真实模型接入状态
- `reports/demo_checklist_report.json|md`：上传联调结果和截图记录
- `demo_samples_manifest.json`：样本清单

## 推荐流程

1. 运行 `python backend/scripts/generate_demo_assets.py`
2. 运行 `python backend/scripts/build_gallery.py`
3. 启动前后端
4. 运行 `python backend/scripts/report_integration_status.py`
5. 运行 `python backend/scripts/run_demo_checklist.py`

