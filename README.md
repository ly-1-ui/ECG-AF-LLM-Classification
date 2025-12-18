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
export HF_ENDPOINT=https://hf-mirror.com
python -m download
```

train:

```shell
python -m src.task2.train
```

