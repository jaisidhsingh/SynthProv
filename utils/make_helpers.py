import os
import torch
from tqdm import tqdm


# set up types of dissimilarity score per matcher
def cosine_distance(feature1, feature2):
    f1, f2 = feature1.T / torch.norm(feature1, dim=1), feature2.T/torch.norm(feature2, dim=1)
    f1, f2 = f1.T, f2.T
    f1, f2 = f1.cpu(), f2.cpu()
    cos = f1 @ f2.T
    return (1 - cos).cpu()

def euclidean_distance(f1, f2):
    f1, f2 = f1.T / torch.norm(f1, dim=1), f2.T/torch.norm(f2, dim=1)
    f1, f2 = f1.T, f2.T
    return torch.cdist(f1.unsqueeze(0), f2.unsqueeze(0)).squeeze(0)

def id_comparison_score(cfg, f1, f2):
    if cfg.matcher in ["ElasticFace", "FaceNet"]:
        return euclidean_distance(f1, f2)
    else:
        return cosine_distance(f1, f2)


def make_id_scores(cfg):
	save_paths = [
		os.path.join(cfg.helpers_dir, cfg.matcher, "comps2reals_id_scores.pt"),
		os.path.join(cfg.helpers_dir, cfg.matcher, "comps2synths_id_scores.pt"),
		os.path.join(cfg.helpers_dir, cfg.matcher, "synths2reals_id_scores.pt"),
	]

	if os.path.exists(save_paths[0]):
		return save_paths

	else:
		comp_embs = torch.load(cfg.composite_embeddings_path)
		real_embs = torch.load(cfg.real_embeddings_path)
		synth_embs = torch.load(cfg.synthetic_embeddings_path)

		comps2reals = id_comparison_score(cfg, comp_embs, real_embs)
		comps2synths = id_comparison_score(cfg, comp_embs, synth_embs)
		synths2reals = id_comparison_score(cfg, synth_embs, real_embs)

		scores_list = [comps2reals, comps2synths, synths2reals]
		for i in range(len(scores_list)):
			torch.save(scores_list[i], save_paths[i])
		
		return save_paths

def make_unprojected_gan_scores(cfg):
	save_paths = [
		os.path.join(cfg.helpers_dir, "unprojected_synth2reals_gan_scores.pt"),
		" "
	]
    
	if os.path.exists(save_paths[0]):
		return save_paths
	
	else:
		synth_latents = torch.load(cfg.synthetic_latents_path)
		real_latents = torch.load(cfg.real_latents_path)

		synth_latents = synth_latents.view((synth_latents.shape[0], -1)).cpu()
		real_latents = real_latents.view((real_latents.shape[0], -1)).cpu()

		synths2reals_gan_scores = euclidean_distance(synth_latents, real_latents)
		torch.save(synths2reals_gan_scores, save_paths[0])

		return save_paths

def make_projected_gan_scores(cfg):
	save_paths = [
		os.path.join(cfg.helpers_dir, "sintheta_synth2reals_gan_scores.pt"),
		os.path.join(cfg.helpers_dir, "costhetas.pt")
	]

	if os.path.exists(save_paths[0]):
		return save_paths
	
	else:
		unprojected = torch.load(
			os.path.join(cfg.helpers_dir, "unprojected_synth2reals_gan_scores.pt")
		)
		costhetas = torch.load(save_paths[1])
		sinthetas = torch.sqrt(1 - costhetas**2)
		projected = unprojected * sinthetas
		print(projected.shape)
		torch.save(projected, save_paths[0])

		return save_paths

def setup_helpers(cfg):
	id_helper_paths = make_id_scores(cfg)

	if cfg.gan_score_mode == "projected":
		gan_helper_paths = make_projected_gan_scores(cfg)

	elif cfg.gan_score_mode == "unprojected":
		gan_helper_paths = make_unprojected_gan_scores(cfg)

	return id_helper_paths, gan_helper_paths
