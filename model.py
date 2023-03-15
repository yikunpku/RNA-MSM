from typing import Optional, Tuple, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import sklearn.linear_model
from tqdm import tqdm
from dataclasses import dataclass, field

from utils.tokenization import Vocab
from utils.metrics import rna_compute_precisions
from utils.tensor import symmetrize, apc

from modules import (
    AxialTransformerLayer,
    ContactPredictionHead,
    LearnedPositionalEmbedding,
    RobertaLMHead,
    RowSelfAttention,
    ColumnSelfAttention,
)
from product_key_memory import PKM

import lr_schedulers
from dataset import RNADataset


@dataclass
class TransformerLayerConfig:
    embed_dim: int = 768
    num_attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    attention_type: str = "standard"
    performer_attention_features: int = 256


@dataclass
class PKMLayerConfig(TransformerLayerConfig):
    pkm_attention_heads: int = 8
    num_product_keys: int = 1024
    pkm_topk: int = 32


@dataclass
class TransformerConfig:
    layer: TransformerLayerConfig = TransformerLayerConfig()
    pkm: PKMLayerConfig = PKMLayerConfig()
    num_layers: int = 12
    max_seqlen: int = 1024
    pkm_layers: List[int] = field(default_factory=list)


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "warmup_cosine"
    warmup_steps: int = 16000
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    max_steps: int = 1000000


class BaseProteinModel(pl.LightningModule, ABC):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[RNADataset] = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.optimizer_config = optimizer_config
        self.contact_train_data = contact_train_data

    @abstractmethod
    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        return NotImplemented

    @abstractmethod
    def get_sequence_attention(self, tokens):
        return NotImplemented

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

    def on_validation_epoch_start(self):
        self.train_contact_regression()

    def train_contact_regression(self, verbose=False):
        data = self.contact_train_data
        if data is None:
            raise RuntimeError(
                "Cannot train regression without trRosetta contact training set."
            )
        X = []
        y = []
        with torch.no_grad():
            iterable = data if not verbose else tqdm(data)
            for tokens,contacts,missing_nt_index in iterable:
                tokens = tokens.unsqueeze(0)
                attentions = self.get_sequence_attention(tokens)
                start_idx = int(self.vocab.prepend_bos)
                end_idx = attentions.size(-1) - int(self.vocab.append_eos)
                attentions = attentions[..., start_idx:end_idx, start_idx:end_idx]
                seqlen = attentions.size(-1)
                attentions = symmetrize(attentions)
                attentions = apc(attentions)
                attentions = attentions.view(-1, seqlen, seqlen).cpu().numpy()

                sep = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
                mask = sep >= 6
                if len(missing_nt_index) > 0:
                    for i in missing_nt_index:
                        mask[i, :] = False
                        mask[:, i] = False

                attentions = attentions[:, mask]
                attentions[np.isnan(attentions)] = 0
                contacts = contacts[mask]
                X.append(attentions.T)
                y.append(contacts)

        X = np.concatenate(X, 0)
        y = np.concatenate(y, 0)

        clf = sklearn.linear_model.LogisticRegression(
            penalty="l1",
            C=0.15,
            solver="liblinear",
            verbose=verbose,
            random_state=0,
        )
        clf.fit(X, y)

        self.contact_head.regression.load_state_dict(
            {
                "weight": torch.from_numpy(clf.coef_),
                "bias": torch.from_numpy(clf.intercept_),
            }
        )

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        logits = self(src)["logits"]
        valid_mask = tgt != self.vocab.pad_idx
        logits = logits[valid_mask]
        tgt = tgt[valid_mask]
        loss = nn.CrossEntropyLoss(reduction="none")(logits, tgt)
        perplexity = loss.float().exp().mean()
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_perp", perplexity, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        predictions = self.predict_contacts(batch["src_tokens"])

        result = {
            'predictions': predictions,
            'tgt': batch["tgt"],
            'missing_nt_index': batch["missing_nt_index"],
        }

        return result

    def validation_epoch_end(self, validation_step_outputs):

        metrics = rna_compute_precisions(
            validation_step_outputs,
            minsep=0,
            step=0.001,
        )

        for key, value in metrics.items():
            key = f"valid_{key}"
            self.log(key, value, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        no_decay = ["norm", "LayerNorm"]

        pkm_params = []
        for module in self.modules():
            if isinstance(module, PKM):
                pkm_params.append(module.values.weight)
        pkm_paramset = set(pkm_params)

        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if param in pkm_paramset:
                continue

            if any(nd in name for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.optimizer_config.weight_decay,
            },
            {"params": no_decay_params, "weight_decay": 0.0},
            {
                "params": pkm_params,
                "weight_decay": 0.0,
                "lr": 4 * self.optimizer_config.learning_rate,
            },
        ]

        if self.optimizer_config.name == "adam":
            optimizer_type = torch.optim.AdamW
        elif self.optimizer_config.name == "lamb":
            try:
                from apex.optimizers import FusedLAMB
            except ImportError:
                raise ImportError("Apex must be installed to use FusedLAMB optimizer.")
            optimizer_type = FusedLAMB
        optimizer = optimizer_type(
            optimizer_grouped_parameters,
            lr=self.optimizer_config.learning_rate,
            betas=self.optimizer_config.adam_betas,
        )
        scheduler = lr_schedulers.get(self.optimizer_config.lr_scheduler)(
            optimizer,
            self.optimizer_config.warmup_steps,
            self.optimizer_config.max_steps,
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict]


class MSATransformer(BaseProteinModel):
    def __init__(
        self,
        vocab: Vocab,
        optimizer_config: OptimizerConfig = OptimizerConfig(),
        contact_train_data: Optional[RNADataset] = None,
        embed_dim: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        embed_positions_msa: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_tokens_per_msa: int = 2 ** 14,
        max_seqlen: int = 1024,
    ):
        super().__init__(
            vocab=vocab,
            optimizer_config=optimizer_config,
            contact_train_data=contact_train_data,
        )
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.embed_positions_msa = embed_positions_msa
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_tokens_per_msa = max_tokens_per_msa

        self.embed_tokens = nn.Embedding(
            len(vocab), embed_dim, padding_idx=vocab.pad_idx
        )

        if embed_positions_msa:
            self.msa_position_embedding = nn.Parameter(
                0.01 * torch.randn(1, 1024, 1, 1),
                requires_grad=True,
            )
        else:
            self.register_parameter("msa_position_embedding", None)  # type: ignore

        self.dropout_module = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim=embed_dim,
                    ffn_embedding_dim=4 * embed_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    max_tokens_per_msa=max_tokens_per_msa,
                )
                for _ in range(num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            num_layers * num_attention_heads,
            vocab.prepend_bos,
            vocab.append_eos,
            eos_idx=vocab.eos_idx,
        )
        self.contact_head.requires_grad_(False)
        self.embed_positions = LearnedPositionalEmbedding(
            max_seqlen,
            embed_dim,
            vocab.pad_idx,
        )
        self.emb_layer_norm_before = nn.LayerNorm(embed_dim)
        self.emb_layer_norm_after = nn.LayerNorm(embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=embed_dim,
            output_dim=len(self.vocab),
            weight=self.embed_tokens.weight,
        )

        self.init_weights()

    def forward(
        self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False
    ):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.vocab.pad_idx)  # B, R, C
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(tokens.long())
        # x = self.embed_tokens(tokens)
        x += self.embed_positions(
            tokens.view(batch_size * num_alignments, seqlen)
        ).view(x.size())
        if self.msa_position_embedding is not None:
            if x.size(1) > 1024:
                raise RuntimeError(
                    "Using model with MSA position embedding trained on maximum MSA "
                    f"depth of 1024, but received {x.size(1)} alignments."
                )
            x += self.msa_position_embedding[:, :num_alignments]

        x = self.emb_layer_norm_before(x)

        x = self.dropout_module(x)

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            row_attn_weights = []
            col_attn_weights = []

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if need_head_weights:
                x, col_attn, row_attn = x
                # H x C x B x R x R -> B x H x C x R x R
                col_attn_weights.append(col_attn.permute(2, 0, 1, 3, 4))
                # H x B x C x C -> B x H x C x C
                row_attn_weights.append(row_attn.permute(1, 0, 2, 3))
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.permute(2, 0, 1, 3)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # col_attentions: B x L x H x C x R x R
            # col_attentions = torch.stack(col_attn_weights, 1)
            # row_attentions: B x L x H x C x C
            row_attentions = torch.stack(row_attn_weights, 1)
            # result["col_attentions"] = col_attentions
            result["row_attentions"] = row_attentions
            if return_contacts:
                contacts = self.contact_head(tokens, row_attentions)
                result["contacts"] = contacts

        return result

    def max_tokens_per_msa_(self, value: int) -> None:
        """The MSA Transformer automatically batches attention computations when
        gradients are disabled to allow you to pass in larger MSAs at test time than
        you can fit in GPU memory. By default this occurs when more than 2^14 tokens
        are passed in the input MSA. You can set this value to infinity to disable
        this behavior.
        """
        self.max_tokens_per_msa = value
        for module in self.modules():
            if isinstance(module, (RowSelfAttention, ColumnSelfAttention)):
                module.max_tokens_per_msa = value

    def get_sequence_attention(self, tokens):
        return self(tokens.to(device=self.device), need_head_weights=True)[
            "row_attentions"
        ]

