import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from src.task1.encoder import Mscnn, load_mscnn_checkpoint, freeze_module


class ECGEncoder(nn.Module):
    def __init__(self, ckpt_path, device):
        super().__init__()
        self.encoder = Mscnn(ch_in=1, ch_out=1, use_stream2=True)

        load_mscnn_checkpoint(
            self.encoder,
            ckpt_path,
            map_location=device,
            strict=False
        )

        freeze_module(self.encoder)
        self.encoder.eval()
        self.encoder.to(device)

    @torch.no_grad()
    def forward(self, ecg):
        return self.encoder.forward_features(ecg)


class ECGProjector(nn.Module):
    def __init__(self, ecg_dim, llm_dim):
        super().__init__()
        self.linear = nn.Linear(ecg_dim, llm_dim)

    def forward(self, x):
        return self.linear(x).unsqueeze(1)


class MultimodalLLM(nn.Module):
    def __init__(
        self,
        cnn_ckpt,
        llm_name="Qwen/Qwen2-1.5B-Instruct",
        device="cuda",
        use_lora=True
    ):
        super().__init__()
        self.device = device

        # ===== Tokenizer =====
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            trust_remote_code=True
        )

        # ===== LLM =====
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to(device)

        # ===== LoRA =====
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm = get_peft_model(self.llm, lora_config)

        # ===== ECG Encoder =====
        self.ecg_encoder = ECGEncoder(cnn_ckpt, device)

        # ===== Projection =====
        ecg_dim = self.ecg_encoder.encoder.feature_dim
        llm_dim = self.llm.config.hidden_size
        self.projector = ECGProjector(ecg_dim, llm_dim).to(device)

    def forward(self, ecg, instruction, answer=None):
        B = ecg.size(0)

        # ECG → embedding
        ecg_feat = self.ecg_encoder(ecg)
        ecg_embed = self.projector(ecg_feat)

        # Instruction tokens
        instr = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        instr_embed = self.llm.get_input_embeddings()(instr.input_ids)

        # 拼接
        inputs_embeds = torch.cat([ecg_embed, instr_embed], dim=1)
        attention_mask = torch.cat(
            [torch.ones(B, 1, device=self.device), instr.attention_mask],
            dim=1
        )

        if answer is not None:
            ans = self.tokenizer(
                answer,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=ans.input_ids
            )

        else:
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=8
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
