from configs import configs as cfg
import torch
from time import perf_counter
import sys

s = perf_counter()

k = int(sys.argv[1])
factor = 10
start = int((k-1) * factor)
end = int(k * factor)

synth_latents = torch.load(cfg.synthetic_latents_path).cuda()[start:end, ...]
real_latents = torch.load(cfg.real_latents_path).cuda()[:, ...]
id_direction = torch.load(cfg.id_direction_path).cuda()

costhetas = torch.zeros((synth_latents.shape[0], real_latents.shape[0]))

vectors = synth_latents.unsqueeze(1) - real_latents.unsqueeze(0)
vectors = vectors.view((synth_latents.shape[0], real_latents.shape[0], -1))

costheta = (vectors @ id_direction.view((1, -1)).T) / (vectors.norm(2) * id_direction.norm(2))
costheta = costheta.squeeze(2)

costhetas = costheta
torch.save(costhetas, f"./res/costhetas_{k}.pt")

e = perf_counter()
print(costhetas.shape, e-s)