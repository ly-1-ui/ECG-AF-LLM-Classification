import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QWEN_NAME = "Qwen/Qwen2-7B-Instruct" 


class ECGQwenForAF(nn.Module):
    """
    Multi-modal model for AF classification with ECG features and Qwen-7B.

    Input:
        ecg_feat:      [B, ecg_dim]
        input_ids:     [B, T] token identifiers of instruction (and optionally answer)
        attention_mask:[B, T] mask for text tokens
        labels:        [B, T'] language modeling labels for Qwen

    The ECG feature is projected to the LLM hidden size and prepended
    as the first token embedding.
    """

    def __init__(self, ecg_dim: int = 256, llm_name: str = DEFAULT_QWEN_NAME):
        super().__init__()
        self.llm_name = llm_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            trust_remote_code=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        hidden_size = self.llm.config.hidden_size
        self.proj = nn.Linear(ecg_dim, hidden_size)

    def forward(
        self,
        ecg_feat: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        """
        ecg_feat: [B, ecg_dim]
        input_ids: [B, T]
        """

        device = next(self.parameters()).device
        ecg_feat = ecg_feat.to(device)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        llm_dtype = next(self.llm.parameters()).dtype

        ecg_feat = ecg_feat.to(self.proj.weight.dtype)
        ecg_emb = self.proj(ecg_feat)                      # [B, H]
        ecg_emb = ecg_emb.to(llm_dtype).unsqueeze(1)       # [B, 1, H]

        text_emb = self.llm.get_input_embeddings()(input_ids)
        if text_emb.dtype != llm_dtype:
            text_emb = text_emb.to(llm_dtype)

        inputs_embeds = torch.cat([ecg_emb, text_emb], dim=1)  # [B, 1+T, H]

        if attention_mask is not None:
            ecg_mask = torch.ones(
                ecg_emb.size(0),
                1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)

        if labels is not None:
            pad_ignore = torch.full(
                (labels.size(0), 1),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([pad_ignore, labels], dim=1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def encode_text(
        self,
        prompt_texts: list[str],
        add_special_tokens: bool = True,
        max_length: int = 256,
    ):
        batch = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )
        return batch["input_ids"], batch["attention_mask"]
