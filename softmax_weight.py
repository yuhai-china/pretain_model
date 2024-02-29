from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=True)
print(model)
torch.save(model.lm_head, "lm.head.pt")
head1 = torch.load("lm.head.pt")
head = torch.nn.Linear(in_features=4096, out_features=32000, bias=False)
head.weight = head1.weight

print(head)
