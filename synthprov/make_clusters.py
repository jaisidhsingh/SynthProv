import torch


# set up types of dissimilarity score per matcher
def cosine_distance(f1, f2):
    f1, f2 = f1/f1.norm(dim=-1, keepdim=True), f2/f2.norm(dim=-1, keepdim=True)
    return f1 @ f2.T

def euclidean_distance(f1, f2):
    f1, f2 = f1.T / torch.norm(f1, dim=1), f2.T/torch.norm(f2, dim=1)
    f1, f2 = f1.T.cpu(), f2.T.cpu()
    return torch.cdist(f1.unsqueeze(0), f2.unsqueeze(0)).squeeze(0)

def id_comparison_score(cfg, f1, f2):
    if cfg.matcher in ["ElasticArcFace+", "FaceNet"]:
        return euclidean_distance(f1, f2)
    else:
        return cosine_distance(f1, f2)


def select_assistants(cfg, parent_indices, comp2synths_id_scores, synthetic_embeddings):
	"""
	select the most similar faces to the composite from our synthetic set
	this is done to use the identity representatives of the composite without using 
	the composite itself in the retrieval (different from direct matchng) 
	"""
	num_synths = len(comp2synths_id_scores)

	non_parent_indices = list(set([i for i in range(num_synths)]) - set(parent_indices))
	threshold = min(comp2synths_id_scores[parent_indices].tolist())

	participant_indices = [j for j in non_parent_indices if comp2synths_id_scores[j] <= threshold]
	participant_scores = comp2synths_id_scores[participant_indices]

	assistant_scores, assistant_indices = participant_scores.topk(k=cfg.num_assistants, largest=False, dim=-1)
	# assistant_embeddings = synthetic_embeddings[participant_indices][assistant_indices]

	return assistant_indices #, assistant_embeddings


def get_clusters(cfg, parent_indices, parent_embeddings, assistant_indices, assistant_embeddings):
	"""
	assign assistants to each parent to make a family for each parent
	family size is kept constrained to 1 + cfg.parent_cluster_size per family/parent
	"""
	# find which parent each is assistant is most similar to on the basis of identity
	scores = id_comparison_score(cfg, parent_embeddings, assistant_embeddings)
	minned = scores.argmin(dim=0).view((cfg.num_assistants)).tolist()

	cluster_dict = {}
	for i in range(cfg.k):
		cluster_dict[i] = {
			"parent_index": parent_indices[i],
			"others_indices": [],
			"others_count": 0
		}

	# iteratively assign assistants to the parent which is most similar in identity to it.
	L = len(minned)
	for i in range(L):
		parent_idx = int(minned[i])

		if cluster_dict[parent_idx]["others_count"] < cfg.parent_cluster_size:
			cluster_dict[parent_idx]["others_indices"].append(int(assistant_indices[i]))
			cluster_dict[parent_idx]["others_count"] += 1

	return cluster_dict 

