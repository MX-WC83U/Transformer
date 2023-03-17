import tensorflow as tf
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, TextLineDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Download the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the training data
train_dataset = TextLineDataset('training_data.txt')

# Tokenize and encode the text
def tokenize_function(text):
    return tokenizer.encode(text, add_special_tokens=True)

train_dataset = train_dataset.map(lambda x: tf.py_function(tokenize_function, [x], tf.int64))

# Define the GPT-2 model
model = TFGPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=1000,
    save_steps=1000,
    evaluation_strategy='steps',
    eval_steps=5000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model weights
model.save_pretrained('./fine-tuned-model')
