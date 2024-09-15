import torch
from fastcore.script import *
from minai.core import *
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

@call_parse
def main(
    model_id: Param("Model ID", str) = 'HuggingFaceTB/SmolLM-360M',
    dataset_name: Param("Dataset name", str) = "answerdotai/tiny_programs",
    model_output_name: Param("Model output name", str) = "answerdotai/SmolLM-360M-finetuned-tiny_programs",
    batch_size: Param("Batch size", int) = 8,
    lr: Param("Learning rate", float) = 1e-3,
    num_epochs: Param("Number of epochs", int) = 5,
    filter_dataset: Param("Filter dataset", bool) = False,
    dataset_size: Param("Dataset size", int) = 754,
    dataset_column: Param("Dataset column", str) = "code",
    is_private: Param("Is private", bool) = True,
):
    set_seed(42)

    # Model and tokenizer setup
    m = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=0,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset preparation
    dataset = load_dataset(dataset_name, split="train").shuffle(42).select(range(dataset_size))
    if filter_dataset:
        dataset = dataset.filter(lambda x: x['score'] in [4, 5])

    def to_text(x):
        x['text'] = x[dataset_column]
        return x

    dataset = dataset.shuffle(42).map(to_text, remove_columns=dataset.column_names)
    train_dataset = dataset.select(range(0, len(dataset)-50))
    eval_dataset = dataset.select(range(len(dataset)-50, len(dataset)))

    # DataLoader setup
    def collate_fn(examples):
        input_ids = tokenizer([e['text'] for e in examples], return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids']
        return (input_ids[:, :-1], input_ids[:, 1:])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    dls = DataLoaders(train_dataloader, eval_dataloader)

    # Training setup
    loss_fn = lambda x, y: torch.nn.functional.cross_entropy(x.view(-1, x.shape[-1]), y.view(-1))
    # sz = len(dls.train) // 10

    cbs = [DeviceCB(), MetricsCB()]
    prog = ProgressCB(plot=True)
    learn = MomentumLearner(m, dls, loss_func=loss_fn, lr=lr, cbs=cbs, preds_nm='logits', mom=0.9)

    # Training
    learn.fit(num_epochs, cbs=prog)

    # push to the hub
    learn.model.push_to_hub(model_output_name, private=is_private)
    tokenizer.push_to_hub(model_output_name, private=is_private)

    # Test generation
    prompt = 'import requests\n'
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()

    with torch.inference_mode():
        output = m.generate(tokenized_prompt, max_new_tokens=90)

    print(prompt + tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True))

if __name__ == "__main__":
    main()