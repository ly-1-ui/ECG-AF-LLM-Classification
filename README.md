# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/qwq-y/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

generate and balance the dataset:

```shell
python -m src.task2.build_llm_dataset --cv 0
python -m src.task2.balance_llm_dataset \
  --input data/llm_cv0/mm_instructions_train_cv0.jsonl \
  --output data/llm_cv0/mm_instructions_train_cv0_posx10.jsonl \
  --factor 10
```

fine-tune:

```shell
export CUDA_VISIBLE_DEVICES=7

python -m src.task2.train \
  --data-root /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/llm_cv0 \
  --mat-dir /home/WangQingyang/Documents/ECG-AF-LLM-Classification/data/training2017 \
  --encoder-ckpt /home/WangQingyang/Documents/ECG-AF-LLM-Classification/outputs/mscnn/model.pth \
  --output-dir /home/WangQingyang/Documents/ECG-AF-LLM-Classification/outputs/llm_cv0 \
  --batch-size 64 \
  --epochs 1 \
  --cv 0 \
  --lr 1e-4
  # --resume outputs/llm_cv0/checkpoint-epoch140-20251214-1957.pt
```

evaluate:

```shell
python -m src.task2.eval \
  --ckpt outputs/llm_cv0/checkpoint-epoch110-20251214-1803.pt \
  --val data/llm_cv0/mm_instructions_train_cv0_100.jsonl \
  --mat-dir data/training2017 \
  --encoder-ckpt outputs/mscnn/model.pth
```

