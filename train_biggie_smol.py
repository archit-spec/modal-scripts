import modal
import os
from typing import Optional

def download_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "nisten/Biggie-SmoLlm-0.15B-Base"
    
    # Download model and tokenizer
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForCausalLM.from_pretrained(model_name)

# Create Modal image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers>=4.31.0",
        "datasets>=2.14.0",
        "accelerate>=0.26.1",
        "bitsandbytes>=0.41.1",
        "wandb",
        "tqdm",
    )
    .env({
        "WANDB_API_KEY": "",
        "HF_TOKEN": "",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
    })
    .run_function(download_model)
)

# Create Modal app
app = modal.App("model-training", image=image)

# Create persistent volume for checkpoints and model artifacts
volume = modal.Volume.from_name("model-training-vol", create_if_missing=True)

@app.cls(
    gpu="L4",  # Request A100 GPU
    timeout=86400,  # 24 hour timeout
    volumes={"/model": volume}
)
class ModelTrainer:
    def __init__(self):
        self.grokking_signal = 0.0

    @modal.enter()
    def setup(self):
        import torch
        import logging
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.model_name = "nisten/Biggie-SmoLlm-0.15B-Base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to('cuda')

    def format_capybara_prompts(self, examples):
        texts = []
        for question, answers in zip(examples['question'], examples['answers']):
            formatted_text = f"Human: {question}<|endoftext|>\n\n"
            for answer in answers:
                formatted_text += f"Assistant: {answer['text']}<|endoftext|>\n\n"
            texts.append(formatted_text.strip())
        return {"text": texts}

    @modal.method()
    def train(
        self,
        output_dir: str = "/model/checkpoints",
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_steps: int = 800,
        gradient_accumulation_steps: int = 4,
        num_warmup_steps: int = 5,
        max_length: int = 1024
    ):
        import torch
        import torch.nn as nn
        from datasets import load_dataset
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        from torch.cuda.amp import autocast
        
        self.logger.info("Loading dataset...")
        dataset = load_dataset("archit11/worldbuilding", split="train[:30%]")
        dataset = dataset.map(
            self.format_capybara_prompts,
            batched=True,
            remove_columns=dataset.column_names
        )

        def tokenize_function(examples):
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            bf16=True,
            logging_steps=10,
            save_steps=300,
            save_total_limit=10,
            warmup_steps=num_warmup_steps,
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            eval_steps=50,
            max_steps=max_steps,
            fp16=False,
            optim="adamw_hf",
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset.select(range(min(1000, len(tokenized_dataset)))),
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False),
        )

        self.logger.info("Starting training...")
        trainer.train()

        self.logger.info("Saving model...")
        trainer.save_model(output_dir)
        trainer.push_to_hub("archit11/qwen-finetuned-model")

        return "Training completed successfully!"

@app.local_entrypoint()
def main(
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_steps: int = 800,
):
    trainer = ModelTrainer()
    result = trainer.train.remote(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps
    )
    print(result)
