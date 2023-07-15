import torch
import os
from tqdm import tqdm


dump = torch.zeros((30000,  30000))
factor = 10
for i in tqdm(range(1, 3001)):
	filename = f"costhetas_{i}.pt"
	path = os.path.join("res", filename)
	x = torch.load(path)

	start = (i-1)*factor
	end = i*factor

	dump[start:end, :] = x
	del x

print("verify", dump[29999][0])
torch.save(dump, "../helpers/celebahq/costhetas.pt")

    
