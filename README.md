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
python -m src.task2.train_llm \
  --cv 0 \
  --batch-size 4 \
  --epochs 5 \
  --grad-accum 8 \
  --lr 1e-4
```

evaluate:

