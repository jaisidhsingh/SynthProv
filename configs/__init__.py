import torch
from types import SimpleNamespace


# initialize configs object
configs = SimpleNamespace(**{})

# running configs
configs.run_name = "ganprov_with_idd_0"
configs.version = 1

# dataset, matcher and k configs
configs.dataset = "ffhq"
configs.matcher = "ElasticArcFace+"
configs.k = 2

# number of composites to operate on
configs.iterations = 10

# how to use GAN space in our algorithm
configs.gan_score_mode = "projected"
configs.cumulate_mode = {
    "global": torch.mean,
    "family":  torch.min
}

# qualitative result configs
configs.qualitative_topk = 10
configs.qualitative_queries = [0, 1, 2]

# save configs
configs.results_dir = f"../results/{configs.dataset}/{configs.matcher}"
configs.helpers_dir = f"../helpers/{configs.dataset}/{configs.matcher}"



# homefolder for dataset
configs.homefolder = f"/workspace/provenance/dataset/{configs.dataset}"

# parent configs
configs.parents_list_path = f"{configs.homefolder}/synthetic/data/parents_{configs.k}.pt"

# id space configs
configs.composite_embeddings_path = f"{configs.homefolder}/synthetic/embeddings/{configs.matcher}/embeddings_{configs.k}.pt"
configs.synthetic_embeddings_path = f"{configs.homefolder}/synthetic/embeddings/{configs.matcher}/embeddings_sources.pt"
configs.real_embeddings_path = f"{configs.matcher}/real/embeddings/{configs.matcher}/embeddings_sources.pt"

# gan space configs
configs.synthetic_latents_path = f"{configs.homefolder}/synthetic/synthetic_source_latents.pt"
configs.real_latents_path = f"{configs.homefolder}/real/{configs.dataset}_inverted_latents.pt"
configs.id_direction_path = "/workspace/provenance/id-direction/results/idd-no-mse/all_directions.pt"

# image data configs
configs.composite_image_folder = f"{configs.homefolder}/synthetic/composites/{configs.k}"
configs.synthetic_image_folder = f"{configs.homefolder}/synthetic/synthetic_sources"
configs.real_image_folder = f"{configs.homefolder}/real/{configs.dataset}_data"

# algorithm configs
configs.num_assistants = 100
configs.parent_cluster_size = 5