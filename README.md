# AgentDeepfakeFaceDet-V2

基于智能体协作的 Deepfake 人脸图像检测系统。项目提供一个本地 Web 演示系统，支持上传单张图片，经过特征提取智能体和决策智能体处理后，输出检测结论、关键分数、策略选择、证据链和可视化结果。

## 实现功能

- 图片级 Deepfake 人脸检测演示，不包含视频检测。
- 前端上传图片、轮询任务状态、展示历史任务和检测结果。
- 后端异步执行检测流程，保存上传文件、任务记录、结果 JSON 和可视化产物。
- 双智能体流程：
  - `feature_agent`：负责人脸检测、图像质量评估、视觉/频域/一致性特征提取和可视化产物生成。
  - `decision_agent`：根据预设策略融合证据，输出 `疑似真实`、`疑似伪造` 或 `需人工复核`。
- 支持模型资产缺失时自动回退到启发式路径，保证本地演示流程可运行。
- 预留阶段二训练、融合、阈值调优和离线评估脚本。

## 技术栈

- 后端：Python、FastAPI、Uvicorn、SQLite、Pydantic、Pillow、NumPy。
- 机器学习相关：PyTorch、TorchVision、OpenCV、XGBoost、scikit-learn、InsightFace、ONNX Runtime、Transformers。
- 前端：Vue 3、Vite、JavaScript、CSS。
- 测试与验证：pytest、HTTPX、项目内 demo/checklist 脚本。

## 目录结构

```text
backend/
  app/                 后端应用、API、智能体、服务、测试
  scripts/             demo、训练、评估、阈值调优等脚本
  requirements.txt     后端基础依赖
  requirements-ml.txt  机器学习可选依赖
  run.py               后端启动入口
frontend/
  src/                 Vue 前端源码
  package.json         前端依赖与脚本
document/              项目说明、论文和实验记录
README.md              项目说明
```

## 环境要求

- Python 3.10 及以上。
- Node.js 18 及以上。
- npm。
- Git。

机器学习依赖较大，首次安装可能需要较长时间。如果只想先跑通基础后端和前端，可以先安装 `backend/requirements.txt`；需要完整模型相关能力时再安装 `backend/requirements-ml.txt`。

## 配置与运行

在项目根目录执行以下命令。

### 1. 创建并激活 Python 虚拟环境

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止激活脚本，可以先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 2. 安装后端依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

可选：安装机器学习相关依赖。

```powershell
python -m pip install -r backend\requirements-ml.txt
```

### 3. 安装前端依赖

```powershell
cd frontend
npm install
cd ..
```

### 4. 启动后端

```powershell
python backend\run.py
```

默认后端地址：

```text
http://127.0.0.1:8000
```

### 5. 启动前端

另开一个终端，在项目根目录执行：

```powershell
cd frontend
npm run dev
```

默认前端地址：

```text
http://127.0.0.1:5173
```

## 常用验证命令

运行后端测试：

```powershell
pytest backend\app\tests -q
```

生成 demo 样本和图廊：

```powershell
python backend\scripts\generate_demo_assets.py
python backend\scripts\build_gallery.py
```

查看当前模型接入状态：

```powershell
python backend\scripts\report_integration_status.py
```

单张图片离线评估：

```powershell
python backend\scripts\evaluate_image.py path\to\image.png
```

## 上传到 GitHub

建议不要上传 `venv`、缓存、运行产物、前端构建产物和本机 IDE 配置。本仓库已通过 `.gitignore` 排除这些内容。

如果这是一个还没有初始化 Git 的本地项目：

```powershell
git init
git add .
git commit -m "Initial project upload"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

如果本地已经是 Git 仓库，只需要关联远程并推送：

```powershell
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

如果远程仓库已经关联过：

```powershell
git add .
git commit -m "Update project documentation"
git push
```

## 当前说明

当前项目适合作为可运行的工程原型和演示系统。真实视觉模型权重、融合模型、本地多模态模型和正式数据集训练结果需要根据实际资源继续补充；在模型资产未到位时，系统会使用启发式后端保证演示链路可用。
