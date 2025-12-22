# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/ly-1-ui/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

generate and balance the dataset:

```shell
python -m src.task2.build_llm_dataset 
```

llm:

```shell

# 下载模型和tokenizer到指定目录
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir ./models/qwen2-1.5b --local-dir-use-symlinks False
```

train:

```shell
python -m src.task2.train
```

