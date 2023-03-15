from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from pytorch_lightning import seed_everything
from utils.tokenization import Vocab
from model import MSATransformer
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from dataset import RNADataset, RandomCropDataset
import numpy as np
import msm
import torch
import os
import hydra

current_directory = Path(__file__).parent.absolute()
seed_everything(42)


@dataclass
class DataConfig:
    device: str = "cuda"
    root_path: str = current_directory
    MSA_path: str = "results"
    MSA_list: str = "rna_id.txt"  # rna id list
    model_path: str = str(current_directory / "pretrained/RNA_MSM_pretrained.ckpt")
    num_workers: int = 3
    architecture: str = "rna language"
    max_seqlen: int = 1024
    max_tokens: int = 16384
    max_seqs_per_msa: int = 512
    sample_method: str = "hhfilter"


@dataclass
class MSATransformerModelConfig:
    embed_dim: int = 768
    num_attention_heads: int = 12
    num_layers: int = 10
    embed_positions_msa: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1


@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    lr_scheduler: str = "warmup_cosine"
    warmup_steps: int = 16000
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    max_steps: int = 500000


@dataclass
class TrainConfig:
    pass


@dataclass
class MSATransformerSmallModelConfig(MSATransformerModelConfig):
    pass


@dataclass
class LoggingConfig:
    pass


@dataclass
class Config:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: MSATransformerModelConfig = MSATransformerModelConfig()
    logging: LoggingConfig = LoggingConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="data", name="default", node=DataConfig)
cs.store(group="train", name="default", node=TrainConfig)
cs.store(group="optimizer", name="default", node=OptimizerConfig)
cs.store(group="model", name="emb-transformer", node=MSATransformerModelConfig)
cs.store(group="logging", name="default", node=LoggingConfig)


@hydra.main(config_name="config")
def extract_feat(cfg: Config) -> None:
    model_path = cfg.data.model_path
    device = torch.device(cfg.data.device)
    alphabet = msm.data.Alphabet.from_architecture(cfg.data.architecture)
    vocab = Vocab.from_esm_alphabet(alphabet)

    print(f'Maximum Number of MSA Seqs:{cfg.data.max_seqs_per_msa}')
    print(f'Inference on: {device}')

    # data
    with open(Path(cfg.data.root_path) / cfg.data.MSA_list) as f:
        test_rnas = f.read().splitlines()
        test_rnas.sort()

    rna_data = RNADataset(
        data_path=cfg.data.root_path,
        msa_path=cfg.data.MSA_path,
        vocab=vocab,
        split_files=test_rnas,
        max_seqs_per_msa=cfg.data.max_seqs_per_msa,
        sample_method=cfg.data.sample_method,
    )

    rna_data = RandomCropDataset(
        rna_data,
        cfg.data.max_seqlen,
    )

    model = MSATransformer(
        vocab,
        optimizer_config=cfg.optimizer,
        contact_train_data=None,
        embed_dim=cfg.model.embed_dim,
        num_attention_heads=cfg.model.num_attention_heads,
        num_layers=cfg.model.num_layers,
        embed_positions_msa=cfg.model.embed_positions_msa,
        dropout=cfg.model.dropout,
        attention_dropout=cfg.model.attention_dropout,
        activation_dropout=cfg.model.activation_dropout,
        max_tokens_per_msa=cfg.data.max_tokens,
        max_seqlen=cfg.data.max_seqlen,
    )
    model.load_state_dict(torch.load(
        model_path,
        map_location=device)['state_dict'], strict=True)
    model = model.eval()
    model = model.to(device)

    # extract_feat
    with torch.no_grad():
        for rna_id, tokens in tqdm(rna_data):

            save_feat_path = os.path.join(cfg.data.root_path, cfg.data.MSA_path)
            if not os.path.exists(save_feat_path):
                os.makedirs(save_feat_path)

            tokens = tokens.unsqueeze(0)
            results = model(tokens.to(device), repr_layers=[10], need_head_weights=True)

            # extract attention map
            attentions = results["row_attentions"]
            start_idx = int(vocab.prepend_bos)
            end_idx = attentions.size(-1) - int(vocab.append_eos)
            attentions = attentions[..., start_idx:end_idx, start_idx:end_idx]
            seqlen = attentions.size(-1)
            attentions = attentions.view(-1, seqlen, seqlen).cpu().numpy()
            attentions_path = os.path.join(save_feat_path, rna_id + "_atp.npy")
            np.save(attentions_path, attentions)

            # extract embedding
            embedding = results["representations"][10]
            start_idx = int(vocab.prepend_bos)
            end_idx = embedding.size(-2) - int(vocab.append_eos)
            embedding = embedding[:, 0, start_idx:end_idx, :].squeeze(0).cpu().numpy()
            embedding_path = os.path.join(save_feat_path, rna_id + "_emb.npy")
            np.save(embedding_path, embedding)

        print(f"Done! Generated files are saved at {save_feat_path}")


if __name__ == "__main__":
    extract_feat()
