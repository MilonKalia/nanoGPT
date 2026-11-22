# tiny CPU configuration for quick end-to-end checks (not used for results)
out_dir = 'out/smoke'
device = 'cpu'
compile = False
dtype = 'float32'
n_layer = 2
n_head = 2
n_embd = 32
mlp_hidden = 64
block_size = 64
batch_size = 4
gradient_accumulation_steps = 1
max_bytes = 0
max_iters = 20
eval_interval = 10
eval_iters = 2
log_interval = 5
attn_backend = 'reference'
