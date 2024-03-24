import pandas as pd
# from torch.utils.data import Dataset
from datasets import load_dataset,Dataset
import torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from accelerate import PartialState, FullyShardedDataParallelPlugin, Accelerator

wandb.init(project='llm-finetune')

os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = "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"
model_id = "google/gemma-7b-it"

df1 = pd.read_csv('data/gemma2b.csv')
df2 = pd.read_csv('data/gemma7b_it.csv')

df = pd.concat([df1, df2], axis=0)
df = df.loc[~df['input'].isna(), :].reset_index(drop=True)


device_string = PartialState().process_index
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={'':device_string},
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    cache_dir='/scratch/npattab1/llms/',
    attn_implementation="flash_attention_2",
    # quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True, add_bos_token=True, padding_side='right', padding=True)


SYS_PROMPT = "Given the following context and a rewritten prompt, generate a new system prompt that combines the essential elements of both inputs to produce a coherent and engaging task description."

def format_prompt(cont, output, prompt):
    return f"""{SYS_PROMPT}\nTEXT: {cont}\n OUTPUT: {output}\nPROMPT: {prompt}"""


class PromptRecoveryDS(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        row = self.df.loc[idx, :].to_dict()
        context, output, prompt = (row['input'], 
                                   row['gemma_outputs'], 
                                   row['instruction'])
        input_text = format_prompt(context, output, prompt)
        # input_tokens = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
        # output_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        return {'text': input_text}

    def __len__(self):
        return self.df.shape[0]



# def collate_fn(batch):
#     x, y = zip(*batch.items())
#     input_tokens = tokenizer.batch_encode_plus(x, add_special_tokens=True)
#     output_tokens = tokenizer.batch_encode_plut(y, add_special_tokens=True);
#     return input_tokens, output_tokens


# data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
response_template = "PROMPT:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


train_ds = PromptRecoveryDS(df, tokenizer)
# print(train_ds[0])
acc_ds = [train_ds[i]['text'] for i in range(len(train_ds))]
train_ds = Dataset.from_pandas(pd.DataFrame({'text': acc_ds}))

training_arguments = TrainingArguments(
    output_dir='/scratch/npattab1/prompt_recovery',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    # optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=10,
    learning_rate=1e-4,
    weight_decay=1e-3,
    fp16=True,
    # bf16=bf16,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb"
)


trainer = SFTTrainer(
    model,
    train_dataset=train_ds,
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=False,
    max_seq_length=1024,
    args=training_arguments,
    # data_collator=collator
    # peft_config=peft_config,
)

trainer.train()
