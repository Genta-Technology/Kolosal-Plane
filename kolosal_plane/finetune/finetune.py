"""Finetuning LLM using Unsloth"""

from typing import Optional, List, Dict, Union

import pandas as pd
from trl import SFTTrainer
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq


class Finetuning():
    model = None
    tokenizer = None
    dataset = None
    max_sequence = None

    def __init__(self,
                 model_name: Optional[str] = "unsloth/Llama-3.2-1B-Instruct",
                 max_sequence: Optional[int] = 4096,
                 dtype: Optional[str] = None,
                 load_int4: Optional[bool] = False):

        # Load the model and tokenizer
        self.max_sequence = max_sequence
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_sequence,
            dtype=dtype,
            load_in_4bit=load_int4,
        )

    def load_dataset(self,
                     dataset: pd.DataFrame,
                     dataset_type: Optional[str] = "",
                     ):
        """
        Load and modify the dataset.
        This method takes a pandas DataFrame containing chat history and responses,
        modifies the chat history by appending the response, and stores the modified
        dataset in the class attribute.
        Args:
            dataset (pd.DataFrame): The dataset containing chat history and responses.
            dataset_type (str): The type of dataset, used to determine further processing steps.
        Returns:
            None
        """
        # Load and modify the dataset
        modified_dataset = []

        # Save it to class dataset
        if dataset_type == "simple_augmentation":
            for _, row in dataset.iterrows():
                # Copy the existing messages
                new_chat_history = row['chat_history'].copy()

                # Add response
                new_chat_history.append(
                    {"role": "assistant", "content": row['response']})

                # Store in the new list
                modified_dataset.append({"conversations": new_chat_history})

        else:
            for _, row in dataset.iterrows():
                # Copy the existing messages
                new_chat_history = [{"role": "system", "content": row['instruction']},
                                    {"role": "user", "content": row['input']},
                                    {"role": "assistant", "content": row['output']}]

                # Store in the new list
                modified_dataset.append({"conversations": new_chat_history})

        # Convert to a DataFrame
        modified_dataset = pd.DataFrame(modified_dataset)
        self.dataset = Dataset.from_pandas(modified_dataset)
        self.dataset = standardize_sharegpt(self.dataset)
        self.dataset = self.dataset.map(
            self.formatting_prompts, batched=True,)

    def finetune(self,
             rank: Optional[int] = 16,
             lora_alpha: Optional[int] = 16,
             lora_dropout: Optional[int] = 0,
             gradient_checkpointing: Optional[bool] = True,
             random_state: Optional[int] = 3407,
             use_rslora: Optional[bool] = False,
             loftq_config: Optional[any] = None,
             epoch: Optional[int] = 1,
             steps: Optional[int] = 60,
             training_args: Optional[dict] = None):
        """
        Fine-tunes the model with the given parameters.

        This function sets up a PEFT-modified model (using LoRA and optionally LoftQ)
        and fine-tunes it using a specialized SFTTrainer. In this setup, the loss is computed
        only on the assistant response tokens. The instruction part—which is now composed of both
        system and user tokens—is masked out during loss computation.

        Args:
            rank (Optional[int]): The rank for the PEFT model. Suggested values are 8, 16, 32, 64, 128. Default is 16.
            lora_alpha (Optional[int]): The alpha parameter for LoRA. Default is 16.
            lora_dropout (Optional[int]): The dropout rate for LoRA. Default is 0.
            gradient_checkpointing (Optional[bool]): Whether to use gradient checkpointing. Default is True.
            random_state (Optional[int]): The random seed for reproducibility. Default is 3407.
            use_rslora (Optional[bool]): Whether to use rank stabilized LoRA. Default is False.
            loftq_config (Optional[Any]): Configuration for LoftQ. Default is None.
            epoch (Optional[int]): Number of epochs for fine-tuning. Default is 1.
            steps (Optional[int]): Total number of training steps. Default is 60.
            training_args (Optional[dict]): Additional training arguments to override the default TrainingArguments.
                All parameters are optional and will be merged with the defaults.

        Returns:
            None
        """

        # Prepare the model with PEFT modifications.
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=gradient_checkpointing,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

        # Initialize training arguments, allowing for additional overrides.
        args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=epoch,
            max_steps=steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            **(training_args or {})
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_sequence,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            dataset_num_proc=2,
            packing=False,
            args=args,
        )

        # Combine system and user parts to form the complete instruction section.
        combined_instruction = (
            # "<|start_header_id|>system<|end_header_id|>\n\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part=combined_instruction,
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        trainer.train()

    def save_model(self,
                   path: str):
        """
        Save the current model and tokenizer to the specified path in GGUF format.
        Args:
            path (str): The directory path where the model and tokenizer will be saved.
        """

        self.model.save_pretrained_gguf(path, self.tokenizer)

    def push_model(self,
                   model_repository: str,
                   quantization: Union[str, List[str]],
                   huggingface_token: str) -> None:
        """
        Push the model to Hugging Face Hub with specified quantization methods.
        This method uploads the model to the Hugging Face Hub repository with the given 
        quantization method(s). The model is converted to GGUF format before uploading.
        Args:
            model_repository (str): The name of the repository on Hugging Face Hub where 
                the model will be pushed (format: "username/repo-name").
            quantization (Union[str, List[str]]): Quantization method(s) to be applied. 
                Can be a single string or a list of strings representing different 
                quantization methods.
            huggingface_token (str): Authentication token for Hugging Face Hub.
        Returns:
            None
        Raises:
            ValueError: If model_repository format is invalid or token is invalid.
            ConnectionError: If there's an error connecting to Hugging Face Hub.
        Example:
            >>> model.push_model(
            ...     "username/model-repo",
            ...     ["q4_k_m", "q5_k_m"],
            ...     "hf_xxxxx"
            ... )
        """

        # Ensure quantization_method is a list
        quantization_method = [quantization] if isinstance(
            quantization, str) else quantization

        self.model.push_to_hub_gguf(
            model_repository,
            self.tokenizer,
            quantization_method=quantization_method,
            token=huggingface_token,
        )

    def formatting_prompts(self, dataset):
        """
        Formats the prompts from a given dataset.
        Args:
            dataset (dict): A dictionary containing the dataset with a key "conversations" 
                            which holds a list of conversation data.
        Returns:
            dict: A dictionary with a single key "text" containing a list of formatted texts.
        """

        convos = dataset["conversations"]
        texts = [self.tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts, }
