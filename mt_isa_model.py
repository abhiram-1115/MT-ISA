"""
MT-ISA model (v2)

What changed vs. the previous version
-------------------------------------
1. D-AWL is actually applied:
     - input  : e_i = c_i * Emb(x_i)          (Eq. 4/5)  -> fed via inputs_embeds
     - output : per-INSTANCE weighting c_i * NLL_i (Eq. 6) instead of batch-mean
     - input_output : both (Eq. 7)
     - none   : no confidence weighting (ablation)
   D-AWL is applied to the auxiliary tasks only (aspect / opinion), as in the paper.
   Confidences are clipped to [0.5, 1.0] (the paper clips at 0.5).
2. Polarity is the PRIMARY task again, trained the way the paper describes:
   prompt-based Flan-T5, cross-entropy on the label word (positive/negative/neutral).
   Prediction = argmax over the first-decoder-step logits of the 3 label tokens.
   The randomly-initialised head on encoder token 0 (T5 has no CLS) is gone.
3. T-AWL is numerically stable:
     weight_k = exp(-s_k) with s_k = log(sigma_k^2)
     ALF1 reg = s_k            ALF2 reg = softplus(s_k) = ln(sigma_k^2 + 1)
   s_k is clamped to [-4, 4]. Logged weights are also reported normalised (sum = 1),
   which is how the paper's Table III reports them.
4. All losses are computed in fp32, with padded label positions ignored (-100)
   and a clamp on the token count, so an empty label can't produce NaN.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration

logger = logging.getLogger(__name__)

POLARITY_WORDS = ["positive", "negative", "neutral"]
TASK_NAMES = ("aspect", "opinion", "polarity")
CONF_MIN = 0.5
LOG_SIGMA_SQ_RANGE = (-4.0, 4.0)


class DataLevelAWL(nn.Module):
    """Data-level automatic weight learning driven by per-instance confidence."""

    def __init__(self, strategy: str = "input", conf_min: float = CONF_MIN):
        super().__init__()
        assert strategy in ("input", "output", "input_output", "none"), (
            f"Unknown D-AWL strategy: {strategy}"
        )
        self.strategy = strategy
        self.conf_min = conf_min

    @property
    def scales_input(self) -> bool:
        return self.strategy in ("input", "input_output")

    @property
    def scales_output(self) -> bool:
        return self.strategy in ("output", "input_output")

    def clean(self, conf: torch.Tensor) -> torch.Tensor:
        conf = torch.nan_to_num(
            conf.float(), nan=self.conf_min, posinf=1.0, neginf=self.conf_min
        )
        return conf.clamp(self.conf_min, 1.0)

    def scale_embeddings(
        self, emb: torch.Tensor, conf: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """e_i = c_i * Emb(x_i).  emb: [B, T, D], conf: [B]"""
        if not self.scales_input or conf is None:
            return emb
        return emb * self.clean(conf).view(-1, 1, 1).to(emb.dtype)

    def weight_sample_losses(
        self, per_sample: torch.Tensor, conf: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """c_i * NLL_i.  per_sample: [B], conf: [B]"""
        if not self.scales_output or conf is None:
            return per_sample
        return per_sample * self.clean(conf)


class TaskLevelAWL(nn.Module):
    """
    Homoscedastic-uncertainty task weighting.
      ALF1: sum_k  L_k / sigma_k^2 + log(sigma_k^2)
      ALF2: sum_k  L_k / sigma_k^2 + ln(sigma_k^2 + 1)
    Parametrised by s_k = log(sigma_k^2), initialised at 0 (sigma^2 = 1).
    Task order: (aspect, opinion, polarity).
    """

    def __init__(self, num_tasks: int = 3, alf_version: str = "alf2"):
        super().__init__()
        assert alf_version in ("alf1", "alf2"), f"Unknown ALF version: {alf_version}"
        self.num_tasks = num_tasks
        self.alf_version = alf_version
        self.log_sigma_sq = nn.Parameter(torch.zeros(num_tasks))

    def _s(self) -> torch.Tensor:
        return self.log_sigma_sq.clamp(*LOG_SIGMA_SQ_RANGE)

    def forward(self, losses: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        assert len(losses) == self.num_tasks
        s = self._s()
        total = 0.0
        for k, loss_k in enumerate(losses):
            total = total + torch.exp(-s[k]) * loss_k
        if self.alf_version == "alf1":
            reg = s.sum()
        else:
            reg = F.softplus(s).sum()  # == ln(exp(s) + 1) == ln(sigma^2 + 1)
        return total + reg

    @torch.no_grad()
    def get_task_weights(self, normalized: bool = True) -> Dict[str, float]:
        raw = torch.exp(-self._s())
        w = raw / raw.sum() if normalized else raw
        return {name: float(w[i]) for i, name in enumerate(TASK_NAMES)}

    @torch.no_grad()
    def get_sigma_sq(self) -> Dict[str, float]:
        sig = torch.exp(self._s())
        return {name: float(sig[i]) for i, name in enumerate(TASK_NAMES)}


class MTISAModel(nn.Module):
    """
    Shared Flan-T5 backbone, three tasks distinguished by their prompts:
      aspect  (aux)  : seq2seq NLL
      opinion (aux)  : seq2seq NLL
      polarity (main): seq2seq NLL on the label word, classified by label-token logits
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        d_awl_strategy: str = "input",
        t_awl_version: str = "alf2",
        backbone: Optional[nn.Module] = None,
        tokenizer=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.d_awl_strategy = d_awl_strategy
        self.t_awl_version = t_awl_version

        self.backbone = (
            backbone if backbone is not None
            else T5ForConditionalGeneration.from_pretrained(model_name)
        )
        self.tokenizer = (
            tokenizer if tokenizer is not None
            else AutoTokenizer.from_pretrained(model_name)
        )

        self.d_awl = DataLevelAWL(strategy=d_awl_strategy)
        self.t_awl = TaskLevelAWL(num_tasks=len(TASK_NAMES), alf_version=t_awl_version)

        first_ids = []
        for word in POLARITY_WORDS:
            ids = self.tokenizer(word, add_special_tokens=False).input_ids
            first_ids.append(ids[0])
            logger.info("Polarity word %-9s -> token ids %s", word, ids)
        if len(set(first_ids)) != len(POLARITY_WORDS):
            raise ValueError(
                f"Polarity words share a first token {first_ids}; "
                "pick different verbalizers."
            )
        self.register_buffer(
            "polarity_first_ids", torch.tensor(first_ids, dtype=torch.long), persistent=False
        )

        logger.info("MT-ISA model: %s | D-AWL=%s | T-AWL=%s",
                    model_name, d_awl_strategy, t_awl_version)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _per_sample_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean token NLL per sample, fp32, padding (-100) ignored.  -> [B]"""
        logits = logits.float()
        nll = F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )  # [B, T]; 0 at ignored positions
        n_tok = (labels != -100).float().sum(dim=1).clamp(min=1.0)
        return nll.sum(dim=1) / n_tok

    def _aux_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        confidence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        emb = self.backbone.get_input_embeddings()(input_ids)
        emb = self.d_awl.scale_embeddings(emb, confidence)          # input strategy
        out = self.backbone(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)
        per_sample = self._per_sample_nll(out.logits, labels)
        per_sample = self.d_awl.weight_sample_losses(per_sample, confidence)  # output strategy
        return per_sample.mean()

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        polarity_input_ids: torch.Tensor,
        polarity_attention_mask: torch.Tensor,
        polarity_labels: torch.Tensor,
        aspect_input_ids: Optional[torch.Tensor] = None,
        aspect_attention_mask: Optional[torch.Tensor] = None,
        aspect_labels: Optional[torch.Tensor] = None,
        aspect_confidence: Optional[torch.Tensor] = None,
        opinion_input_ids: Optional[torch.Tensor] = None,
        opinion_attention_mask: Optional[torch.Tensor] = None,
        opinion_labels: Optional[torch.Tensor] = None,
        opinion_confidence: Optional[torch.Tensor] = None,
    ) -> Dict:
        outputs: Dict = {}

        # ---- primary task: polarity (no D-AWL: gold labels, no confidence) ----
        pol = self.backbone(
            input_ids=polarity_input_ids,
            attention_mask=polarity_attention_mask,
            labels=polarity_labels,
        )
        polarity_loss = self._per_sample_nll(pol.logits, polarity_labels).mean()
        outputs["polarity_loss"] = polarity_loss
        # decoder step 0 predicts the first label token -> 3-way classification
        outputs["polarity_logits"] = (
            pol.logits[:, 0, :].index_select(-1, self.polarity_first_ids).float()
        )

        have_aux = aspect_input_ids is not None and opinion_input_ids is not None
        if not have_aux:
            outputs["combined_loss"] = polarity_loss
            return outputs

        # ---- auxiliary tasks ----
        aspect_loss = self._aux_loss(
            aspect_input_ids, aspect_attention_mask, aspect_labels, aspect_confidence
        )
        opinion_loss = self._aux_loss(
            opinion_input_ids, opinion_attention_mask, opinion_labels, opinion_confidence
        )
        outputs["aspect_loss"] = aspect_loss
        outputs["opinion_loss"] = opinion_loss

        # ---- task-level AWL ----
        outputs["combined_loss"] = self.t_awl((aspect_loss, opinion_loss, polarity_loss))
        return outputs

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_polarity(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Returns [B, 3] logits over (positive, negative, neutral)."""
        dec = torch.full(
            (input_ids.size(0), 1),
            self.backbone.config.decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec
        )
        return out.logits[:, 0, :].index_select(-1, self.polarity_first_ids).float()


def _self_test():
    """python mt_isa_model.py  ->  quick shape / NaN check with flan-t5-small."""
    logging.basicConfig(level=logging.INFO)
    model = MTISAModel("google/flan-t5-small", d_awl_strategy="input_output")
    tok = model.tokenizer
    enc = tok(["The food is great.", "Service was slow."], padding=True, return_tensors="pt")
    lab = tok(["positive", "negative"], padding=True, return_tensors="pt").input_ids
    lab[lab == tok.pad_token_id] = -100
    conf = torch.tensor([0.9, 0.6])
    out = model(
        polarity_input_ids=enc.input_ids, polarity_attention_mask=enc.attention_mask,
        polarity_labels=lab,
        aspect_input_ids=enc.input_ids, aspect_attention_mask=enc.attention_mask,
        aspect_labels=lab, aspect_confidence=conf,
        opinion_input_ids=enc.input_ids, opinion_attention_mask=enc.attention_mask,
        opinion_labels=lab, opinion_confidence=conf,
    )
    for k, v in out.items():
        print(k, tuple(v.shape) if v.dim() else float(v))
    assert torch.isfinite(out["combined_loss"])
    print("task weights:", model.t_awl.get_task_weights())
    print("OK")


if __name__ == "__main__":
    _self_test()
