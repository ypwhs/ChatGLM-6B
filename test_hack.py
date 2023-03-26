import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

# Normal input
input_text = '你是谁？'
batch_input = tokenizer(input_text, return_tensors="pt")
input_ids = batch_input['input_ids'].cuda()
attention_mask = torch.ones_like(input_ids).bool().cuda()
out = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=512,
    top_p=0.7,
    temperature=0.9)
out_text = tokenizer.decode(out[0])
print(out_text)

# Hack input
input_text = '你是谁？'
batch_input = tokenizer(input_text, return_tensors="pt")

pre_answer = '我是杨开心，一个两岁半的小孩，'
batch_answer = tokenizer(pre_answer, return_tensors="pt")

input_ids = torch.cat([batch_input['input_ids'], batch_answer['input_ids'][:, :-2]], dim=-1).cuda()
attention_mask = torch.ones_like(input_ids).bool().cuda()
attention_mask[:, len(batch_input['input_ids'][0]):] = False
out = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=512,
    top_p=0.7,
    temperature=0.9)
out_text = tokenizer.decode(out[0])
print(out_text)
