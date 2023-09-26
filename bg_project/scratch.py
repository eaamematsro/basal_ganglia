import torch

rthal = torch.randn((1, 10))
U = torch.randn((50, 10))
V = torch.randn((10, 50))
r = torch.randn((1, 50))

out_1 = r @ U @ torch.diag(rthal[0]) @ V

#
# J2 = torch.einsum('ij, kj, jl -> kil', U, rthal, V)
# out_2 = torch.einsum('kil, ki -> kl', J2, r)
out_3 = torch.einsum("ij, kj, jl, ki -> kl", U, rthal, V, r)
torch.isclose(out_1, out_3).all()
