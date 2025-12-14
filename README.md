# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/qwq-y/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

generate the dataset:

```shell
python -m src.task2.build_llm_dataset --cv 0
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
  --epochs 5 \
  --cv 0 \
  --ecg-len 2400
```

evaluate:

```shell
python -m src.task2.eval \
  --ckpt outputs/llm_cv0/checkpoint-epoch5-20251212-1825.pt \
  --val data/llm_cv0/mm_instructions_val_cv0.jsonl \
  --mat-dir data/training2017 \
  --encoder-ckpt outputs/mscnn/model.pth
```
