import pathlib

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.v2 as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

class SimpleDataset(Dataset):
    def __init__(self, root, transform):
        self.root = pathlib.Path(root)
        self.transform = transform

        self.files = sorted(list(self.root.glob("*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        input_tensor = self.transform(image)
        return input_tensor

# Load the pre-trained AlexNet model
alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)

# Set the model to evaluation mode
alexnet.eval()

# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 150
dataset = SimpleDataset("gabor-filters", transform)
dataloader = DataLoader(dataset, batch_size)
num_img = len(dataset)
print(num_img)
features = np.zeros((num_img, 9216))

for i, p in tqdm(enumerate(dataloader), total=len(dataloader)):
    # Pass the input batch through the model to extract features
    with torch.no_grad():
        vector = alexnet.avgpool(alexnet.features(p.to(device))).flatten(start_dim=1)
        # print(vector.shape, torch.linalg.vector_norm(vector, dim=1).shape)
        # normalize the feature vector
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True)
        features[i*batch_size:(i+1)*batch_size, :] = vector.cpu().numpy()

np.save("features-gabor.npy", features)