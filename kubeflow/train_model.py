# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    Model
)


@component(base_image="python:3.12"',
          packages_to_install=["torch", "transformers>=4.30.0", "datasets", "accelerate"])
def train_model(
    dataset_dir: str,
    original_model_dir: str,
    finetuned_model_dir: str,
    hyperparameters: dict,
    finetuned_model: Output[Model]
):
    try:
        import os
        import random
        import torch
        import json
        import uuid
        import zipfile
        from pathlib import Path

        from transformers import (
            AutoTokenizer, # tokenizer model 
            AutoModelForSeq2SeqLM, # main seq2seq model
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            DataCollatorForSeq2Seq # dataset collator
        )
        from datasets import load_dataset, Dataset, DatasetDict
    except ImportError as e:
        print(f"Import error: {e}")

    # create output dir
    os.makedirs(finetuned_model_dir, exist_ok=True)

    # detect accelerator
    def detect_accelerator() -> (str, torch.dtype):
        # detect discrete accelerator
        if torch.cuda.is_available():
            accelerator = "cuda"
            dtype = torch.float16
        else:
            accelerator = "cpu"
            dtype = torch.float32

        return (accelerator, dtype)

    # create a new dataset class that holds training data information
    class CustomPIIDataset(torch.utils.data.Dataset):
        def __init__(self, dataset_path: str) -> None:
            from pathlib import Path

            # path that contains json datasets
            self.dataset_path: Path = Path(dataset_path)

            # find datafiles
            self.datasets: list = [f for f in self.dataset_path.glob("**/*.json")]

            # load dataset
            self.dataset: list = []
            for fname in self.datasets:
                print(f"Loading {fname}...")
                try:
                    with open(fname, "r") as json_dataset:
                        self.dataset.extend(json.load(json_dataset))
                except UnicodeDecodeError as ue:
                    print(f"Error decoding {fname} due to {ue}: Skipping file.")

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx) -> dict:
            # return datapoint
            return self.dataset[idx]

    # preprocessing stuff
    class DataPreprocessor():
        def __init__(self, tokenizer, max_length: int = 256):
            self.tokenizer = tokenizer
            self.max_length = max_length

        # preprocessing function to prepare dataset for training
        def data_preprocess(self, examples):
            """
            Convert PII anonymization data to T5 text-to-text format
            Input: "anonymize: [original text]"
            Output: "[anonymized text]"
            """
            inputs = []
            targets = []

            for original, anonymized in zip(examples['original'], examples['anonymized']):
                # Create input with task prefix
                input_text = f"anonymize: {original}"
                inputs.append(input_text)

                # Target is the anonymized text
                targets.append(anonymized)

            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=self.max_length,  # Increased for longer Italian sentences
                truncation=True,
                padding=False
            )

            # Tokenize targets
            labels = self.tokenizer(
                targets,
                max_length=self.max_length,
                truncation=True,
                padding=False
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    # detect the presence of an accelerator available for this task
    device, dtype = detect_accelerator()

    # parameters
    EPOCHS: int = hyperparameters.get("epochs", 1)
    LEARNING_RATE: float = float(hyperparameters.get("learning_rate", 1e-4))
    HAS_GPU: bool = (device == "cuda")
    MAX_LENGTH: int = int(hyperparameters.get("max_length", 256))
    OPTIMIZER: str = hyperparameters.get("optimizer", "AdamW")
    BATCH_SIZE: int = int(hyperparameters.get("batch_size", 4))
    TRAIN_VAL_SPLIT: float = float(hyperparameters.get("train_val_split", 0.8))
    RUN_NAME: str = f"flan-t5-it-finetune_{uuid.uuid4()}"

    # load dataset from disk...
    it_pii_dataset: CustomPIIDataset = CustomPIIDataset(dataset_dir)
    print(f"Dataset Loaded! -> Processed {len(it_pii_dataset)} datapoints")

    # prepare randomized splits
    random.shuffle(it_pii_dataset.dataset)
    train_val_split = int(len(it_pii_dataset) * float(TRAIN_VAL_SPLIT))
    train_examples: list = it_pii_dataset[:train_val_split]
    val_examples: list = it_pii_dataset[train_val_split:]

    # Create datasets
    train_data = {
        "original": [ex.get("source") for ex in train_examples],
        "anonymized": [ex.get("target") for ex in train_examples]
    }

    val_data = {
        "original": [ex.get("source") for ex in val_examples],
        "anonymized": [ex.get("target") for ex in val_examples]
    }

    # the complete rebuilt dataset. this is used for training
    dataset = DatasetDict({
        "train": Dataset.from_dict(train_data),
        "validation": Dataset.from_dict(val_data)
    })

    # Display dataset information
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(original_model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(original_model_dir).to(device)

    print(f"Model loaded: {original_model_dir} on {device}")
    print(f"Model parameters: {model.num_parameters()}")

    # Instantiate Preprocessor
    dp: DataPreprocessor = DataPreprocessor(tokenizer=tokenizer, max_length=MAX_LENGTH)

    # Process datasets
    print("Processing datasets...")
    tokenized_datasets = dataset.map(
        dp.data_preprocess,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    # print out final dataset
    print("Tokenized datasets:")
    print(tokenized_datasets)
    print("\nFirst tokenized example (input):")
    print(tokenizer.decode(tokenized_datasets['train'][0]['input_ids']))

    ## PREPARE TRAINING STEP ##

    # setup training parameters
    training_args = Seq2SeqTrainingArguments(
        output_dir=finetuned_model_dir,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,  # Higher learning rate for smaller dataset
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,  # More epochs for small dataset
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=HAS_GPU,  # Use mixed precision if GPU available
        dataloader_pin_memory=HAS_GPU, # only on GPU equipped systems. also silences warnings on MPS devices (Apple)
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none", run_name=RUN_NAME,
    )

    # Data collator:
    # - Build data batches
    # - dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator
    )

    print("Trainer initialized successfully!")

    ## TRAIN MODEL!
    print("Starting training...")
    train_result = trainer.train()

    # report information
    print("\nTraining completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")

    ## EVALUATE MODEL
    print("Evaluating model on validation set...")
    eval_result = trainer.evaluate()

    print("\nEvaluation Results:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    ## SAVE MODEL
    model.save_pretrained(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)

    # save finetuned model to S3
    finetuned_model._set_path(finetuned_model.path + ".zip")
    srcdir = Path(finetuned_model_dir)

    with zipfile.ZipFile(finetuned_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in srcdir.rglob("*"):
            zip_file.write(entry, entry.relative_to(srcdir))