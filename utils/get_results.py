import torch
from PIL import Image
import os


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_all_real_paths(cfg):
	all_paths = [os.path.join(cfg.real_image_folder, fname) for fname in os.listdir(cfg.real_image_folder)]
	return sorted(all_paths)

def get_all_composite_paths(cfg):
	all_paths = [os.path.join(cfg.composite_image_folder, fname) for fname in os.listdir(cfg.composite_image_folder)]
	return sorted(all_paths)

def get_all_synthetic_paths(cfg):
	all_paths = [os.path.join(cfg.synthetic_image_folder, fname) for fname in os.listdir(cfg.synthetic_image_folder)]
	return sorted(all_paths)

def query_qualitative(cfg, donor_list, save_path):
	comp_paths = get_all_composite_paths(cfg)
	real_paths = get_all_real_paths(cfg)

	per_query_images = []
	for i, query_idx in enumerate(cfg.qualitative_queries):
		q_path = comp_paths[query_idx]
		donors = donor_list[i]
		d_paths = [real_paths[x] for x in donors]

		img_paths = [q_path] + d_paths
		imgs = [Image.open(path).convert('RGB') for path in img_paths]
		imgs = [img.resize((100, 100)) for img in imgs]

		tmp = imgs[0]
		for j in range(1, len(imgs)):
			tmp = get_concat_h(tmp, imgs[j])
		
		per_query_images.append(tmp)
	
	res = per_query_images[0]
	for i in range(1, len(per_query_images)):
		res = get_concat_v(res, per_query_images[i])
	
	res.save(save_path)

def major_qualitative(cfg, donor_list, save_path):
	real_paths = get_all_real_paths(cfg)
	img_paths = [real_paths[idx] for idx in donor_list]
	imgs = [Image.open(path).convert('RGB') for path in img_paths]
	imgs = [img.resize((100, 100)) for img in imgs]
	
	tmp = imgs[0]
	for i in range(1, len(imgs)):
		tmp = get_concat_h(tmp, imgs[i])
	
	tmp.save(save_path)

def write_report(cfg, report_path, synthprov_results, naive_matching_results):
	with open(report_path, "w") as f:
		f.writelines(
			[
				f"Experiment name: {cfg.run_name} \n",
				f"Number of composites used for this experiment: {cfg.iterations} \n",
				f"-------------------------------------------------------------------- \n",
				"  \n",
				"  \n",
				f"SynthProv results: \n",
				f"--- Real images leaking identity into composites {cfg.qualitative_queries}: \n",
				f"------- Indices of real images: {synthprov_results['queries']['indices']} \n"
				f"------- Scores of real images: {synthprov_results['queries']['scores']} \n"
				" \n",
				f"--- Real images leaking identity into all composites (major donors): \n",
				f"------- Indices of real images: {synthprov_results['major']['indices']} \n",
				f"------- Scores of real images: {synthprov_results['major']['scores']} \n",
				"  \n",
				"  \n",
				f"Naive Matching results: \n",
				f"--- Real images naively matching to composites {cfg.qualitative_queries}: \n",
				f"------- Indices of real images: {naive_matching_results['queries']['indices']} \n"
				f"------- Scores of real images: {naive_matching_results['queries']['scores']} \n"
				" \n",
				f"--- Real images acting as naive major donors for all synthetics (major donors): \n",
				f"------- Indices of real images: {naive_matching_results['major']['indices']} \n",
				f"------- Scores of real images: {naive_matching_results['major']['scores']} \n",
			]
		)
	print(f"Report generated at: {report_path}")

def inference_query_qualitative(cfg, query_list, donor_list, save_path):
	comp_paths = get_all_composite_paths(cfg)
	real_paths = get_all_real_paths(cfg)

	per_query_images = []
	for i, query_idx in enumerate(query_list):
		q_path = comp_paths[query_idx]
		donors = donor_list[i]
		d_paths = [real_paths[x] for x in donors]

		img_paths = [q_path] + d_paths
		imgs = [Image.open(path).convert('RGB') for path in img_paths]
		imgs = [img.resize((100, 100)) for img in imgs]

		tmp = imgs[0]
		for j in range(1, len(imgs)):
			tmp = get_concat_h(tmp, imgs[j])
		
		per_query_images.append(tmp)
	
	res = per_query_images[0]
	for i in range(1, len(per_query_images)):
		res = get_concat_v(res, per_query_images[i])
	
	res.save(save_path)

