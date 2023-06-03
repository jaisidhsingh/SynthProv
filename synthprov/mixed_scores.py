import torch


# construct our mixed scores using the families made in the previous step
def get_scores(cfg, clusters, synths2reals_id_scores, synths2reals_gan_scores):
	# get number of reals
	num_reals = synths2reals_id_scores.shape[1]

	# make separate lists for indices of each family member for convenience in indexing
	families = []
	for i in range(cfg.k):
		families.append([clusters[i]["parent_index"]] + clusters[i]["others_indices"])

	# make a holder for the cumulative family scores
	global_scores = torch.zeros((cfg.k, num_reals))
        
	# iterate over families and make mixed scores
	for i in range(cfg.k):
		id_scores = synths2reals_id_scores[families[i]]
		gan_scores = synths2reals_gan_scores[families[i]]
		
		# make mixed scores
		if cfg.version == 1:
			mixed_scores = id_scores + 0.3 * gan_scores
		else:
			mixed_scores = id_scores * gan_scores

		# cumulate them locally (or within each family) according to how we want
		cum_mixed_scores = cfg.cumulate_mode["family"](mixed_scores, dim=0)
		global_scores[i] = cum_mixed_scores

	# cumulate them globally according to how we want
	final_scores = cfg.cumulate_mode["global"](global_scores, dim=0)

	return final_scores
	
		
		
        
    