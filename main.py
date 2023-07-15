from configs import configs as cfg
from utils.get_results import *
from utils.make_helpers import *
from synthprov.make_clusters import *
from synthprov.mixed_scores import *

import torch
import os
from tqdm import tqdm


def make_readable(tensor):
	tmp = []
	tensor = tensor.tolist()
	for item in tensor:
		tmp.append([round(subitem, 4) for subitem in item])
	return tmp

def run_synthprov(cfg):
	# load precomputed matrices
	id_helpers_paths, gan_helpers_paths = setup_helpers(cfg)
	comps2synths_id_scores = torch.load(id_helpers_paths[1])
	synths2reals_id_scores = torch.load(id_helpers_paths[2])
	synths2reals_gan_scores = torch.load(gan_helpers_paths[0])

	# load additional data
	parents_data = torch.load(cfg.parents_list_path)['parents']
	synthetic_embeddings = torch.load(cfg.synthetic_embeddings_path)

	# setup progress bar
	bar = tqdm(total=cfg.iterations)

	# holder for the scores for efficiency
	store = torch.zeros((cfg.iterations, synths2reals_id_scores.shape[1]))

	# start the algorithm
	for i in range(cfg.iterations):
		
		# get the objects required for each composite
		parent_indices = parents_data[i]
		parent_embeddings = synthetic_embeddings[parent_indices]
		comp2synths_id_scores = comps2synths_id_scores[i]

		# get our assistants
		assistant_indices, _ = select_assistants(
			cfg,
			parent_indices,
			comp2synths_id_scores,
			synthetic_embeddings
		)

		# construct mixed scores
		scores = get_scores(
			cfg,
			parent_indices,
			assistant_indices,
			synths2reals_id_scores,
			synths2reals_gan_scores,
		)
		# store the scores
		store[i] = scores
		
		# log the progress
		logs = {"done": i+1}
		bar.update(1)
		bar.set_postfix(**logs)

	# make custom save directory
	save_dir = os.path.join(cfg.results_dir, cfg.run_name)
	os.makedirs(save_dir, exist_ok=True)

	# save scores
	torch.save(store, os.path.join(save_dir, f"ganprov_{cfg.run_name}.pt"))

	# get sample results for some query composites
	queries_donor_scores, queries_donor_indices = store[cfg.qualitative_queries].topk(
		k=cfg.qualitative_topk,
		largest=False,
		dim=-1
	)

	# get major donors from the SynthProv paradigm
	major_donor_scores, major_donor_indices = store.mean(dim=0).topk(
		k=cfg.qualitative_topk,
		largest=False,
		dim=-1
	)

	# results results
	results = {
		"queries": {
			"scores": make_readable(queries_donor_scores),
			"indices": queries_donor_indices.tolist()
		},
		"major": {
			"scores": [round(item, 4) for item in major_donor_scores.tolist()],
			"indices": major_donor_indices.tolist()
		}
	}

	return results


def run_naive_matching(cfg):
	# load precomputed matrices
	id_helpers_paths, _ = setup_helpers(cfg)
	comps2reals_id_scores = torch.load(id_helpers_paths[0])
	synths2reals_id_scores = torch.load(id_helpers_paths[2])

	# get sample results for some query composites
	queries_donor_scores, queries_donor_indices = comps2reals_id_scores[cfg.qualitative_queries].topk(
		k=cfg.qualitative_topk,
		largest=False,
		dim=-1
	)

	# get major donors from the naive paradigm
	major_donor_scores, major_donor_indices = synths2reals_id_scores.mean(dim=0).topk(
		k=cfg.qualitative_topk,
		largest=False,
		dim=-1
	)

	# return results
	results = {
		"queries": {
			"scores": make_readable(queries_donor_scores),
			"indices": queries_donor_indices.tolist()
		},
		"major": {
			"scores": [round(item, 4) for item in major_donor_scores.tolist()],
			"indices": major_donor_indices.tolist()
		}
	}

	return results


def main(cfg):
	synthprov_results = run_synthprov(cfg)
	naive_matching_results = run_naive_matching(cfg)

	save_dir = os.path.join(cfg.results_dir, cfg.run_name)
	sp_query_save_path = os.path.join(save_dir, f"query_synthprov_{cfg.run_name}.png")
	sp_major_save_path = os.path.join(save_dir, f"major_synthprov_{cfg.run_name}.png")

	nm_query_save_path = os.path.join(save_dir, f"query_naive_matching_{cfg.run_name}.png")
	nm_major_save_path = os.path.join(save_dir, f"major_naive_matching_{cfg.run_name}.png")

	report_save_path = os.path.join(save_dir, f"report_{cfg.run_name}.txt")

	# get and save qualitative results
	query_qualitative(
		cfg, 
		synthprov_results['queries']['indices'],
		sp_query_save_path
	)
	query_qualitative(
		cfg,
		naive_matching_results['queries']['indices'],
		nm_query_save_path
	)

	major_qualitative(
		cfg, 
		synthprov_results['major']['indices'],
		sp_major_save_path
	)
	major_qualitative(
		cfg,
		naive_matching_results['major']['indices'],
		nm_major_save_path
	)

	# generate and save report
	write_report(cfg, report_save_path, synthprov_results, naive_matching_results)
	
	# phew, finished
	print("Done")


if __name__ == "__main__":
	main(cfg)
