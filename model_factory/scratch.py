import numpy as np
import torch
latents = 5
outputs = 100
batches = 10
U = torch.rand((outputs, latents))
V = torch.randn((latents, outputs))
r = torch.randn((latents, batches))
cur_value = torch.randn((outputs, batches))
for batch in range(batches):
    r[:, batch] *= batch

v_batch = torch.einsum('jk, ji -> jik', r, V)
j_batch = torch.einsum('ij, jlk -> ilk', U, v_batch)
out_value = torch.einsum('ijk, jk -> ik', j_batch, cur_value)

truth_vals = []
difference = []
for batch in range(batches):
    true_val = j_batch[:, :, batch] @ cur_value[:, batch]
    true_valj = U @ torch.diag(r[:, batch]) @ V
    truth_vals.append(np.isclose(true_val, out_value[:, batch]).all())
    difference.append(torch.linalg.norm(true_val - out_value[:, batch]))
print(truth_vals)
print(difference)