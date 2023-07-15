from tqdm import tqdm
import torch
import numpy as np
from utils.make_helpers import cosine_distance, euclidean_distance


datasets = ["ffhq"]
matchers = ["ArcFace"]
helpers_dir = f"../helpers"
results_dir = f"../results"
k = 6

for dataset in datasets:
	for matcher in matchers:
		synthprov_path = f"{results_dir}/{dataset}/{matcher}/{dataset}_st_full_{k}/ganprov_{dataset}_st_full_{k}.pt"

		ce = torch.load(f"/workspace/provenance/dataset/{dataset}/synthetic/embeddings/{matcher}/embeddings_{k}.pt")	
		re = torch.load(f"/workspace/provenance/dataset/{dataset}/real/embeddings/{matcher}/embeddings_sources.pt")	
		
		if matcher == "ElasticFace":
			comps2reals = euclidean_distance(ce, re)
		else:
			comps2reals = cosine_distance(ce, re)
		
		synthprov = torch.load(synthprov_path)

		dump = torch.zeros((10000, 5))
		values, indices = synthprov.topk(5, largest=False, dim=-1)
		for i, idx in tqdm(enumerate(indices)):
			dump[i] = comps2reals[i][idx]

		dump = dump.cpu().numpy()
		save_path = f"/workspace/provenance/dataset/{dataset}/synthetic/embeddings/{matcher}/lr_scores_{k}.npy"
		np.save(save_path, dump)
		print(dataset, matcher, "done", dump.flatten().mean())	
        