# ECG-AF-LLM-Classification

install:

```shell
git clone https://github.com/ly-1-ui/ECG-AF-LLM-Classification
conda env create -f environment.yml
conda activate ecg
```

```conda
# 下载 64 位版本（默认安装到用户目录，无需管理员权限）
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh -O miniconda.sh
# 运行安装程序
bash miniconda.sh
# 适配 bash 终端
source ~/.bashrc
```

generate and balance the dataset:

```shell
python -m src.task2.build_llm_dataset 
```

llm:

```shell
export HF_ENDPOINT=https://hf-mirror.com
python -m download

# 下载模型和tokenizer到指定目录
huggingface-cli download Qwen/Qwen2-1.5B-Instruct --local-dir ./models/qwen2-1.5b --local-dir-use-symlinks False
```

train:

```shell
python -m src.task2.train
```

