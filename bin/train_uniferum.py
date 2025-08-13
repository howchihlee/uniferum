import os

import pandas as pd
from transformers import AutoTokenizer, Trainer, TrainingArguments

from bin.utils import (
    ConfigFileArgs,
    create_transform,
    load_checkpoint_from_config_if_available,
    load_yaml,
    save_yaml,
)
from data_utils.vqa_dataset import VQABinaryDataCollator, VQAMaskDataset
from models.multimodal_models import Uniferum


def create_datasets(
    input_file: str,
    tokenizer,
    train_transform=None,
    predict_transform=None,
    train_transform_imgonly=None,
):
    df_data = pd.read_parquet(input_file)
    df_train = df_data[df_data.split == "train"]
    df_val = df_data[df_data.split == "val"]

    train_dataset = VQAMaskDataset(
        df_train.question.to_list(),
        df_train.img_file_path.to_list(),
        tokenizer,
        df_train.label.to_list(),
        transform=train_transform,
        transform_imgonly=train_transform_imgonly,
    )

    eval_dataset = VQAMaskDataset(
        df_val.question.to_list(),
        df_val.img_file_path.to_list(),
        tokenizer,
        df_val.label.to_list(),
        transform=predict_transform,
    )

    collector = VQABinaryDataCollator(train_dataset.tokenizer, True)
    return train_dataset, eval_dataset, collector


def create_and_prepare_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["llm_args"]["model_id"])
    model = Uniferum(config)
    model = load_checkpoint_from_config_if_available(model, config)

    # Freeze LLM by default unless explicitly set to False
    freeze_llm = config.get("freeze_llm", True)
    freeze_llm = freeze_llm if isinstance(freeze_llm, bool) else True
    if "lora_args" not in config and freeze_llm:
        print("freezing llm")
        for param in model.model.parameters():
            param.requires_grad = False

    for param in model.lm_head_cls.parameters():
        param.requires_grad = True
    for param in model.lm_head_seg.parameters():
        param.requires_grad = True
    for param in model.model.pooler.parameters():
        param.requires_grad = True
    return model, tokenizer


def main():
    config_file = ConfigFileArgs().parse_args().config_file
    config = load_yaml(config_file)
    if "logging_dir" not in config["hf_TrainingArguments"]:
        out_dir = config["hf_TrainingArguments"].get("output_dir", "./")
        config["hf_TrainingArguments"]["logging_dir"] = os.path.join(out_dir, "logs")

    file_path = os.path.join(
        config["hf_TrainingArguments"]["output_dir"], "train_config.yaml"
    )
    save_yaml(config, file_path)

    train_transform = None
    if config.get("train_transform") is not None:
        train_transform = create_transform(config["train_transform"])
    train_transform_imgonly = None
    if config.get("train_transform") is not None:
        train_transform_imgonly = create_transform(config["train_transform_imgonly"])
    predict_transform = None
    if config.get("predict_transform") is not None:
        predict_transform = create_transform(config["predict_transform"])

    model, tokenizer = create_and_prepare_model(config)
    train_dataset, eval_dataset, collector = create_datasets(
        config["input_file"],
        tokenizer,
        train_transform=train_transform,
        predict_transform=predict_transform,
        train_transform_imgonly=train_transform_imgonly,
    )

    # Set training arguments
    training_args = TrainingArguments(**config["hf_TrainingArguments"])
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collector,
    )

    print("trainer.train")
    trainer.train()


if __name__ == "__main__":
    main()
