# Server Training Notes

## Upload these files

At minimum, upload the following:

- `backend/scripts/train_visual_model.py`
- `backend/scripts/train_visual_model_server.sh`
- `backend/scripts/SERVER_TRAINING_README.md`
- `backend/requirements.txt`
- `backend/requirements-ml.txt`

Upload your prepared dataset directory separately, for example:

- `/data/ffpp_visual_dataset/train/real`
- `/data/ffpp_visual_dataset/train/fake`
- `/data/ffpp_visual_dataset/val/real`
- `/data/ffpp_visual_dataset/val/fake`

## Minimal training environment

Recommended Python version:

- `Python 3.10+`

Install dependencies:

```bash
pip install -r backend/requirements.txt
pip install -r backend/requirements-ml.txt
```

## Run training with a live log file

```bash
bash backend/scripts/train_visual_model_server.sh \
  /data/ffpp_visual_dataset \
  backend/storage/models \
  ffpp_kaggle_c23_v1 \
  --epochs 3 \
  --batch-size 16 \
  --device cuda
```

This creates a log file under:

```text
backend/storage/models/logs/
```

## Watch training progress live

```bash
tail -f backend/storage/models/logs/train_visual_*.log
```

## Expected outputs

After training finishes, you should get:

- `backend/storage/models/efficientnet_b0_deepfake.pt`
- `backend/storage/models/efficientnet_b0_deepfake.meta.json`
