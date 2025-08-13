import glob
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tap import Tap
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bin.utils import create_transform, load_yaml
from data_utils.vqa_dataset import VQAMaskDataset
from models.multimodal_models import Uniferum
from models.utils import load_safetensors

mp.set_start_method("spawn", force=True)


class Arguments(Tap):
    run_dir: str
    ckpt_name: Optional[str] = (
        None  # a specific checkpoint for inference, will run all checkpoints if not give
    )
    val_file: str
    devices: str = "0,1,2,3,4,5,6,7"
    prefix: str = "prediction"


def get_ckpt_files(args):
    if args.ckpt_name is not None:
        ckpt_file = os.path.join(args.run_dir, args.ckpt_name, "model.safetensors")
        yield ckpt_file, args.ckpt_name
    else:
        search_str = os.path.join(args.run_dir, "checkpoint-*/model.safetensors")
        ckpt_files = glob.glob(search_str)
        for ckpt_file in ckpt_files:
            ckpt_name = os.path.basename(os.path.dirname(ckpt_file))
            yield ckpt_file, ckpt_name


def predict(rank: int, config, df_data: pd.DataFrame, ckpt_file: str, outputs):
    tokenizer = AutoTokenizer.from_pretrained(
        config["llm_args"]["model_id"], test_mode_fast=True
    )

    transform = None
    if "predict_transform" in config:
        transform = create_transform(config["predict_transform"])
    eval_dataset = VQAMaskDataset(
        df_data.question.to_list(),
        df_data.img_file_path.to_list(),
        tokenizer,
        transform=transform,
    )

    dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_dataset.collate_fn,
        num_workers=12,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    model = Uniferum(config).cuda(rank).eval()
    model = load_safetensors(model, ckpt_file)

    local_outputs = []
    with torch.no_grad():
        for k, d in enumerate(dataloader):
            d = {key: tensor.cuda(rank) for key, tensor in d.items()}
            out = model(**d).cpu().numpy()
            local_outputs.append(((rank, k), out[:, 0]))
    outputs.append(local_outputs)
    return


def main(
    output_file: str, ckpt_file: str, model_config: dict, val_file: str, devices=str
):
    gpus = sorted(
        set([int(r) for r in devices.split(",")])
    )  # ranks sorted to ensure order
    num_gpus = len(gpus)
    df_val = pd.read_parquet(val_file)
    chunk_size = int(np.ceil(len(df_val) / num_gpus))
    chunks = [
        df_val.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)
    ]
    print(f"main: {model_config}")
    manager = mp.Manager()
    outputs = manager.list()
    processes = []
    for i, rank in enumerate(gpus):
        p = mp.Process(
            target=predict, args=(rank, model_config, chunks[i], ckpt_file, outputs)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    all_outputs = [item for sublist in outputs for item in sublist]
    all_outputs.sort(key=lambda x: x[0])  # Sort by (rank, k)

    final_outputs = np.concatenate([out for _, out in all_outputs], axis=0)
    df_out = pd.DataFrame(final_outputs, columns=["predict"])
    # df_out["patient_id"] = df_val.patient_id.tolist()
    df_out["question"] = df_val.question.tolist()
    df_out["val_file"] = val_file
    df_out[["question", "predict"]].to_parquet(output_file)


if __name__ == "__main__":

    args = Arguments().parse_args()
    run_dir = args.run_dir
    config_file = os.path.join(run_dir, "train_config.yaml")
    model_config = load_yaml(config_file)

    for ckpt_file, ckpt_name in get_ckpt_files(args):
        output_file = os.path.join(run_dir, f"{args.prefix}_{ckpt_name}.parquet")
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping.")
            continue
        main(
            output_file=output_file,
            ckpt_file=ckpt_file,
            model_config=model_config,
            val_file=args.val_file,
            devices=args.devices,
        )
