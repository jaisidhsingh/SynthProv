import os
import torch
from tqdm import tqdm


# set up types of dissimilarity score per matcher
def cosine_distance(f1, f2):
    f1, f2 = f1/f1.norm(dim=-1, keepdim=True), f2/f2.norm(dim=-1, keepdim=True)
    return f1 @ f2.T

def euclidean_distance(f1, f2):
    f1, f2 = f1.T / torch.norm(f1, dim=1), f2.T/torch.norm(f2, dim=1)
    # f1, f2 = f1.T.cpu(), f2.T.cpu()
    f1, f2 = f1.T, f2.T
    return torch.cdist(f1.unsqueeze(0), f2.unsqueeze(0)).squeeze(0)

def id_comparison_score(cfg, f1, f2):
    if cfg.matcher in ["ElasticArcFace+", "FaceNet"]:
        return euclidean_distance(f1, f2)
    else:
        return cosine_distance(f1, f2)


def make_id_scores(cfg):
	save_paths = [
		os.path.join(cfg.helpers_dir, "comps2reals_id_scores.pt"),
		os.path.join(cfg.helpers_dir, "comps2synths_id_scores.pt"),
		os.path.join(cfg.helpers_dir, "synths2reals_id_scores.pt"),
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
		os.path.join(cfg.helpers_dir, "projected_synth2reals_gan_scores.pt"),
		os.path.join(cfg.helpers_dir, "costheta_with_id_direction.pt")
	]

	if os.path.exists(save_paths[0]):
		return save_paths
	
	else:
		synth_latents = torch.load(cfg.synthetic_latents_path).cpu()
		real_latents = torch.load(cfg.real_latents_path).cpu()
		id_direction = torch.load(cfg.id_direction_path).cpu()

		store = torch.zeros((synth_latents.shape[0], real_latents.shape[0]))
		costhetas = torch.zeros((synth_latents.shape[0], real_latents.shape[0]))

		old_i = 0
		step = 10	
		for i in tqdm(range(step, synth_latents.shape[0]+1, step)):
			iis = [x for x in range(old_i, i)]
			synth_latent = synth_latents[iis].view((step, 18, 512))

			vectors = synth_latent.unsqueeze(1) - real_latents.unsqueeze(0)
			vectors = vectors.view((synth_latent.shape[0], real_latents.shape[0], -1))

			costheta = (vectors @ id_direction.view((1, -1)).T) / (vectors.norm(2) * id_direction.norm(2))
			costheta = costheta.squeeze(2)

			synth_latent = synth_latent.view((synth_latent.shape[0], -1))
			real_latents = real_latents.view((real_latents.shape[0], -1))

			distances = euclidean_distance(synth_latent.cuda(), real_latents.cuda())
			projected_distance = distances.cuda() * costheta.cuda()

			store[iis] = projected_distance.cpu()
			costhetas[iis] = costheta.cpu()

			real_latents = real_latents.view((70000, 18, 512))

			old_i += step

		torch.save(store, save_paths[0])
		torch.save(costhetas, save_paths[1])

		return save_paths

def setup_helpers(cfg):
	id_helper_paths = make_id_scores(cfg)

	if cfg.gan_score_mode == "projected":
		gan_helper_paths = make_projected_gan_scores(cfg)

	elif cfg.gan_score_mode == "unprojected":
		gan_helper_paths = make_unprojected_gan_scores(cfg)

	return id_helper_paths, gan_helper_paths