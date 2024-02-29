from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=False)
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=False)
#print(model)
#embedding_model = model.model.embed_tokens
#print(embedding_model.weight)

#model2 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", local_files_only=False)
model2 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", local_files_only=False)
embedding_model2 = model2.model.embed_tokens
#print(embedding_model2.weight)

#embedding_model.weight.add(embedding_model2.weight)
avg = embedding_model2.weight
torch.save(avg,"avg.emb.pt")
print(avg)

