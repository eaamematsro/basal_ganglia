import torch
import pdb
import numpy as np
from model_factory.factory_utils import torchify
from model_factory.networks import DecoderNetwork


cluster_dim, latent_dim, batch_size = 5, 10, 20

clusters = torchify(np.random.randn(batch_size, cluster_dim))
latents = torchify(np.random.randn(batch_size, latent_dim))

model = DecoderNetwork(
    input_dim=50, latent_dim=latent_dim, number_of_clusters=cluster_dim
)
model(latents, clusters)
out_dim, batch_size, latent_dim = 50, 10, 5

P = torch.randn(out_dim, latent_dim)
Q = torch.randn(latent_dim, out_dim)
z = torch.randn(batch_size, latent_dim)
input_vector = torch.randn(batch_size, out_dim)
z_test = torch.diag_embed(z)
a = torch.randn(batch_size, 15, latent_dim)
b = torch.randn(latent_dim, 30)
intermediate = torch.matmul(z_test, Q)
inp_mat = torch.matmul(P, intermediate)
test = torch.matmul(inp_mat, input_vector.unsqueeze(2)).squeeze()
# batch_test = torch.bmm(z_test, Q[None, :, :])
rec_input = torch.einsum("ij, kj, jl, ki -> kl", P, z, Q, input_vector)
ground_truths = []
for batch in range(batch_size):
    value = P @ torch.diag(z[batch]) @ Q @ input_vector[batch]
    ground_truths.append(value)
    print((value - test[batch]).sum())
pdb.set_trace()
