"""
import torch
from make_clusters import euclidean_distance

# Two sets of tensors
set1 = torch.randn((12, 18, 512))
set2 = torch.randn((70000, 18, 512))

# Compute pairwise difference
vectors = set1.unsqueeze(1) - set2.unsqueeze(0)
vectors = vectors.view((12, 70000, -1))

direction = torch.randn((18, 512))

costheta = (vectors @ direction.view((1, -1)).T) / (vectors.norm(2) * direction.norm(2))
costheta = costheta.squeeze(2)

distance = euclidean_distance(set1.view((12, -1)), set2.view((70000, -1)))
# print(distance.shape)
# print(costheta.shape)
distance_along_direction = distance * costheta
print(distance_along_direction.shape)
"""