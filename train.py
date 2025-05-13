import torch
import tqdm
from transformers import AutoTokenizer
from transformers.models.llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import *
from datasets import load_dataset
from model import *
import gc
from accelerate import Accelerator


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
def tokenize_function(batch):
    conversations = batch["conversations"]
    texts = ["\n".join([turn["value"] for turn in convo]) for convo in conversations]
    tokenized = tokenizer(texts, truncation=True, max_length=128)
    return tokenized


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 2)


def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    padded = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")
    return padded


def main():
    global tokenized_dataset
    dtype = torch.bfloat16
    accelerator = Accelerator()


    device = accelerator.device
    base_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct", torch_dtype=dtype, device_map=device
    )
    base_model.eval()
    speculator_head = PredictorHead(base_model.model.config)
    speculator_head.load_state_dict(torch.load("speculator_head.pth", weights_only=True))
    speculator_head.to(device, dtype=dtype)
    specModel = TwoHeadModel(base_model, speculator_head)
    for param in specModel.base_model.parameters():
        param.requires_grad = False
    for param in specModel.main_head.parameters():
        param.requires_grad = False
    specModel.train()


    batch_size = 8
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    num_params = sum(p.numel() for p in specModel.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params / 1e6:.2f}M")


    ##########################
    # Training loop
    ##########################
    # We will optimize only the parameters of the speculator head.
    optimizer = torch.optim.Adam(speculator_head.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()


    num_epochs = 1
    torch.cuda.empty_cache()
    _ = gc.collect()


    specModel, optimizer, dataloader, scheduler = accelerator.prepare(
        specModel, optimizer, dataloader, scheduler
    )


    for epoch in range(num_epochs):
        idx = 0
        avg_loss = 0
        for batch in tqdm.tqdm(dataloader):
            input_ids = batch["input_ids"]
            head_logits, spec_logits = specModel(input_ids)
            head_logits = head_logits[:, 1:, :]
            spec_logits = spec_logits[:, :-1, :]
            head_logits = head_logits.reshape(-1, head_logits.shape[-1])
            spec_logits = spec_logits.reshape(-1, spec_logits.shape[-1])
            loss = loss_fn(spec_logits, torch.nn.functional.softmax(head_logits))


            avg_loss += loss.item()


            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            print(f"Epoch: {epoch} | Loss: {avg_loss:.4f}")
            avg_loss = 0
            idx += 1
            if idx % 1000 == 0 and accelerator.is_main_process:
                model = accelerator.unwrap_model(specModel)
                torch.save(model.speculator_head.state_dict(), "speculator_head.pth")


    if accelerator.is_main_process:
        model = accelerator.unwrap_model(specModel)
        torch.save(model.speculator_head.state_dict(), "speculator_head.pth")


if __name__ == "__main__":
    main()