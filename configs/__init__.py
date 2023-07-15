from types import SimpleNamespace


# initialize configs object
configs = SimpleNamespace(**{})

# running configs
configs.run_name = "ffhq_st_full_6"
configs.version = 1

# dataset, matcher and k configs
configs.dataset = "ffhq"
configs.matcher = "ArcFace"
configs.k = 6

# number of composites to operate on
configs.iterations = 10000

# how to use GAN space in our algorithm
configs.gan_score_mode = "projected" # keep this!
configs.id_coeff = 1.0
configs.w_coeff = 0.3

# qualitative result configs
configs.qualitative_topk = 5
configs.qualitative_queries = [0, 1, 2]

# save configs
configs.results_dir = f"../results/{configs.dataset}/{configs.matcher}"
configs.helpers_dir = f"../helpers/{configs.dataset}"
configs.inference_dir = f"../inference/{configs.dataset}"


# homefolder for dataset
configs.homefolder = f"/workspace/provenance/dataset/{configs.dataset}"

# parent configs
configs.parents_list_path = f"{configs.homefolder}/synthetic/data/parents_{configs.k}.pt"

# id space configs
configs.composite_embeddings_path = f"{configs.homefolder}/synthetic/embeddings/{configs.matcher}/embeddings_{configs.k}.pt"
configs.synthetic_embeddings_path = f"{configs.homefolder}/synthetic/embeddings/{configs.matcher}/embeddings_sources.pt"
configs.real_embeddings_path = f"{configs.homefolder}/real/embeddings/{configs.matcher}/embeddings_sources.pt"

# gan space configs
configs.synthetic_latents_path = f"{configs.homefolder}/synthetic/synthetic_source_latents.pt"
configs.real_latents_path = f"{configs.homefolder}/real/{configs.dataset}_inverted_latents.pt"
configs.id_direction_path = "/workspace/provenance/id-direction/results/idd-no-mse/all_directions.pt"

# image data configs
configs.composite_image_folder = f"{configs.homefolder}/synthetic/composites/{configs.k}"
configs.synthetic_image_folder = f"{configs.homefolder}/synthetic/synthetic_sources"
configs.real_image_folder = f"{configs.homefolder}/real/{configs.dataset}_data"

# algorithm configs
configs.num_assistants = 10
configs.parent_cluster_size = 5