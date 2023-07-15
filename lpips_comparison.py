import torch
import lpips
from utils.get_results import *
from utils.make_helpers import *
from configs import configs as cfg
from PIL import Image
from torchvision import transforms as tf
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")


def get_transforms():
    res = tf.Compose([
        tf.Resize((128, 128)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
	])
    return res

def load_image(path, transforms):
    image = Image.open(path).convert('RGB')
    image = transforms(image)
    image = image*2 - 1.0
    return image.unsqueeze(0)

store_paths = [
    os.path.join(cfg.results_dir, cfg.run_name, f"ganprov_{cfg.run_name}.pt"), 
    setup_helpers(cfg)[0][0]
]

print(cfg.run_name)

for store_path in store_paths:
	store = torch.load(store_path)

	queries = [x for x in range(1000)]
	queries_donor_scores, queries_donor_indices = store[queries].topk(5, largest=False, dim=-1)

	perceptual_loss = lpips.LPIPS(net='vgg')
	real_paths = get_all_real_paths(cfg)
	comp_paths = get_all_composite_paths(cfg)

	transforms = get_transforms()

	losses = np.zeros((1000, 5))
	for i in tqdm(range(len(queries))):
		q_idx = queries[i]
		q_path = comp_paths[q_idx]
		q_image = load_image(q_path, transforms)

		d_paths = [real_paths[u] for u in queries_donor_indices[i].tolist()]
		d_images = [load_image(p, transforms) for p in d_paths]

		loss_values = [perceptual_loss(q_image, img).item() for img in d_images]
		losses[i] = np.array(loss_values)

	print(losses.mean(axis=0))

