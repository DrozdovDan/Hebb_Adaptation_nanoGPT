import time

out_dir = 'out-shakespeare-a-no-updates'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2-xl' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 40

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

hebb_updates = False 
bias=False
rank=8
alpha=32
hebb_dropout=0.1
lora_init={'lora_a' : 'normal', 'lora_b' : 'zeros'}
lora_frozen_layers = ['lora_a']
attn_modules=['c_attn']
hebb_lr=0.045
temperature=1.0
hebb_linears=[]
