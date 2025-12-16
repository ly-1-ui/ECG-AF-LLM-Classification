import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from ..task1.encoder import Mscnn, freeze_module, load_mscnn_checkpoint

DEFAULT_QWEN_NAME = "Qwen/Qwen2-7B-Instruct"


class ECGQwenForAF(nn.Module):
    def __init__(
        self,
        llm_name: str = DEFAULT_QWEN_NAME,
        ecg_ch_in: int = 1,
        ecg_len: int = 3000,
        ecg_encoder_use_stream2: bool = True,
        ecg_encoder_stream2_kernel: int = 7,
        ecg_encoder_ckpt: Optional[str] = None,
        freeze_encoder: bool = True,
        ecg_adapter_dim: int = 256,
        ecg_adapter_dropout: float = 0.0,
        llm_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
    ):
        super().__init__()
        self.llm_name = llm_name

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=llm_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        self.encoder = Mscnn(
            ch_in=ecg_ch_in,
            ch_out=1,  # head unused; we use forward_features()
            use_stream2=ecg_encoder_use_stream2,
            stream2_kernel=ecg_encoder_stream2_kernel,
            input_len=ecg_len,
        )

        if ecg_encoder_ckpt is not None:
            load_mscnn_checkpoint(
                self.encoder,
                ecg_encoder_ckpt,
                map_location="cpu",
                strict=False,
            )

        if freeze_encoder:
            freeze_module(self.encoder)

        self.ecg_adapter = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, ecg_adapter_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=ecg_adapter_dropout) if ecg_adapter_dropout > 0 else nn.Identity(),
        )

        hidden_size = self.llm.config.hidden_size
        self.proj = nn.Linear(ecg_adapter_dim, hidden_size)

    def forward(self, ecg, input_ids, attention_mask=None, labels=None):
        inputs_embeds, full_attention_mask = self.prepare_inputs_for_generation(
            ecg=ecg,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        llm_device = inputs_embeds.device
        if labels is not None:
            labels = labels.to(llm_device)
            pad_ignore = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=llm_device)
            labels = torch.cat([pad_ignore, labels], dim=1)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )

    def prepare_inputs_for_generation(self, ecg, input_ids, attention_mask=None):
        llm_param = next(self.llm.parameters())
        llm_device = llm_param.device
        llm_dtype = llm_param.dtype

        if attention_mask is not None:
            attention_mask = attention_mask.to(llm_device)

        self.encoder = self.encoder.to(llm_device)
        self.ecg_adapter = self.ecg_adapter.to(llm_device)
        self.proj = self.proj.to(llm_device)

        ecg = ecg.to(device=llm_device, dtype=torch.float32)
        feat = self.encoder.forward_features(ecg)
        feat = feat.to(self.ecg_adapter[0].weight.dtype)
        ecg_vec = self.ecg_adapter(feat)
        ecg_vec = ecg_vec.to(self.proj.weight.dtype)
        ecg_emb = self.proj(ecg_vec).to(llm_dtype).unsqueeze(1)

        text_emb = self.llm.get_input_embeddings()(input_ids.to(llm_device))
        if text_emb.dtype != llm_dtype:
            text_emb = text_emb.to(llm_dtype)

        inputs_embeds = torch.cat([ecg_emb, text_emb], dim=1)

        if attention_mask is not None:
            ecg_mask = torch.ones(
                inputs_embeds.size(0),
                1,
                dtype=attention_mask.dtype,
                device=llm_device,
            )
            full_attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)
        else:
            full_attention_mask = None
            
        # DEBUG
        # if not hasattr(self, "_dbg_ecg_once"):
        #     self._dbg_ecg_once = True
            
        #     x = ecg_emb[0]  # [1, D]
        #     print("ecg_embeds:", x.shape)
        #     print("mean/std:", float(x.mean()), float(x.std()))
        #     print("abs-mean:", float(x.abs().mean()))
        #     print("l2-norm:", float(x.norm(p=2)))

        #     print("text_len:", int(attention_mask[0].sum().item()))
        #     print("inputs_embeds_len:", int(inputs_embeds.shape[1]))
        #     print("full_mask_len:", int(full_attention_mask.shape[1]))

        #     print("ecg_len:", int(ecg_emb.shape[1]))

        #     print("full_mask_first_16:", full_attention_mask[0, :16].int().tolist())
        #     print("full_mask_last_16:", full_attention_mask[0, -16:].int().tolist())
        #     print("=" * 60, flush=True)

        return inputs_embeds, full_attention_mask

    def encode_text(self, prompt_texts: List[str], add_special_tokens: bool = True, max_length: int = 256):
        batch = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return batch["input_ids"], batch["attention_mask"]


def apply_lora_to_llm(
    model: ECGQwenForAF,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
):
    for p in model.llm.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()
    return model
