import os
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset, load_from_disk
import transformers
import trl
import torch
from bounding_trainers import BoundingTrainer

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="filtered_bc_s1")
    wandb_entity: Optional[str] = field(default="chongli-qin91-na")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)
    use_sft: bool = False
    ref_log_probs_in_input: bool = False
    # remove_unused_columns: bool = False

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


class CustomDataCollator(trl.DataCollatorForCompletionOnlyLM):
    def __call__(self, examples):
        # print(f"********KEYS IN CUSTOM DATA COLLATOR*******{examples[0].keys()}")

        examples_stripped = []
        for ex in examples:
            examples_stripped.append({
                "input_ids": ex["input_ids"],
                "attention_mask": ex["attention_mask"]
            })
        batch = super().__call__(examples_stripped)  # Call the parent class to get the default batch

        # Adding a new key, e.g., "custom_key"
        # print(f"Shape ids[0] {batch['input_ids'].shape}") 
        if "ref_log_probs" in examples[0].keys():
            batch["ref_log_probs"] = torch.tensor([example["ref_log_probs"] for example in examples])[:, :, None]
            batch["inputs_ref"] = torch.tensor([example["inputs_ref"] for example in examples])[:, :, None]
            batch["attn_mask"] = torch.tensor([example["attn_mask"] for example in examples])[:, :, None]
            batch["labels_ref"] = torch.tensor([example["labels_ref"] for example in examples])[:, :, None]
        # diff_inputs = torch.abs(batch["inputs_ref"] - batch['input_ids'])
        # diff_attn = torch.abs(batch["attention_mask"] - batch['attn_mask'])
        # print(f"******{diff_inputs.max()},*******")
        # print(f"******{diff_attn.max()},*******")
        return batch
    

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    ref_model = None
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {"device_map": "auto", "torch_dtype": "auto",
                  "attn_implementation": "flash_attention_2", "use_cache": False}
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
        if not config.use_sft and not config.ref_log_probs_in_input:
            logging.info("Setting up reference model since no ref_log_probs in input")
            ref_model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs).requires_grad_(False)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
        if not config.use_sft and not config.ref_log_probs_in_input:
            logging.info("Setting up reference model since no ref_log_probs in input")
            ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct").requires_grad_(False)
    
    if config.train_file_path.endswith(".hf"):
        dataset = load_from_disk(config.train_file_path)
        dataset = {'train': dataset}
    else:
        dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    # collator = trl.DataCollatorForCompletionOnlyLM(
    #     instruction_template=instruction_template,
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    #     mlm=False
    # )
    collator = CustomDataCollator(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size

    if config.use_sft:
        logging.info("Training with SFT!")
        trainer = trl.SFTTrainer(
            model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator
        )
    else:
        logging.info("Training with IW BoundingTrainer!")
        trainer = BoundingTrainer(
            ref_model=ref_model,
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
            args=args,
            data_collator=collator
        )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
