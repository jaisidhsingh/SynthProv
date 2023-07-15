from configs import configs as cfg
from utils.make_helpers import id_comparison_score, cosine_distance
import torch


ce = torch.load("/workspace/provenance/dataset/ffhq/synthetic/embeddings/ArcFace/embeddings_2.pt")
# ce = torch.load("/workspace/embeddings_2.pt")
re = torch.load("/workspace/provenance/dataset/ffhq/real/embeddings/ArcFace/embeddings_sources.pt")

sp = torch.load(
    "../results/ffhq/ArcFace/ffhq_st_full_2/ganprov_ffhq_st_full_2.pt"
)

cfg.matcher = "ArcFace"
# score = id_comparison_score(cfg, ce, re)
score = cosine_distance(ce, re)
print(score.shape)

# values = score.flatten()
values, indices = sp.topk(k=5, largest=False, dim=-1)
v = torch.stack([score[i][idx] for i, idx in enumerate(indices)])
print(v.shape)
print(v.flatten().mean().item())