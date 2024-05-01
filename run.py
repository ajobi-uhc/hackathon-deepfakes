import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.io import read_video
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from torchvision.utils import save_image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torchvision.transforms.functional as F
import timm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

DATASET_VIDEO_PATH = "./data/train_dataset"
DATASET_METADATA_PATH = "./data/train_dataset/metadata.json"
FRAME_RATE = 1  # Frame rate to sample (e.g., 1 frame per second)

# Load video metadata
df_labels = pd.read_json(DATASET_METADATA_PATH, orient='index')
df_labels.reset_index(inplace=True)
df_labels.columns = ['Filename', 'Label']
df_labels['label_value'] = np.where(df_labels['Label'] == 'real', 1, 0)

class VideoDataset(Dataset):
    def __init__(self, dataframe, root_dir, sequence_length=10, transform=None):
        """
        Args:
            dataframe (DataFrame): DataFrame containing video filenames and labels.
            root_dir (str): Directory path where video files are stored.
            sequence_length (int): Number of frames to extract from each video.
            transform (callable, optional): Optional transform to be applied on a frame.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        video_filename = self.dataframe.iloc[idx]['Filename']
        video_path = os.path.join(self.root_dir, video_filename)
        label = self.dataframe.iloc[idx]['label_value']
        # Read video and extract frames
        frames, _, _ = read_video(video_path, pts_unit='sec', start_pts=0, end_pts=10, output_format='TCHW')
        total_frames = len(frames)
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.sequence_length).long()
        selected_frames = frames[frame_indices]

        processed_frames = []
        for frame in selected_frames:
            if self.transform:
                frame = self.transform(frame)
            processed_frames.append(frame)

        frames_tensor = torch.stack(processed_frames)
        return frames_tensor, label

# Example of setting up the dataset and dataloader with transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Necessary to convert raw video frame to PIL Image for some transformations
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),  # Convert the PIL Image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
])

dataset = VideoDataset(df_labels, DATASET_VIDEO_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
model = timm.create_model('vit_large_patch14_clip_224', pretrained=True)

model.head = nn.Identity()

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze all the parameters in the model to prevent them from being updated during training
for param in model.parameters():
    param.requires_grad = False

# Define your classifier that will take the output features from the transformer
output_dims_of_CLIP = 1024  # This should match the output features of the last layer before the head
classifier = nn.Linear(in_features=output_dims_of_CLIP, out_features=1).to(device)
criterion = nn.BCEWithLogitsLoss()
initial_lr = 1e-4
optimizer = torch.optim.Adam(classifier.parameters(), lr=initial_lr)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
import torch
from tqdm import tqdm

num_epochs = 10  # Example epoch count
losses = []  # List to store all losses for visualization or further analysis

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    total_loss = 0.0
    num_batches = 0

    # Wrap the loader with tqdm for a progress bar
    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)

    for batch_idx, (videos, labels) in progress_bar:
        videos = videos.to(device)  # [batch_size, seq_length, channels, height, width]
        labels = labels.to(device).float().unsqueeze(1)  # [batch_size, seq_length, 1]
        labels = labels.repeat(1, videos.size(1)).unsqueeze(-1)  # [batch_size, seq_length, 1]

        optimizer.zero_grad()
        video_loss = 0.0  # Accumulate loss for the batch

        for i in range(videos.size(1)):  # Process each frame in the sequence
            frames = videos[:, i, :, :, :]
            frame_features = model(frames)
            logits = classifier(frame_features).view(-1, 1)
            frame_labels = labels[:, i]
            loss = criterion(logits, frame_labels)
            video_loss += loss

        # Average the loss over the number of frames, then backpropagate
        video_loss /= videos.size(1)
        video_loss.backward()
        optimizer.step()
        # Record and print the average loss
        current_loss = video_loss.item()
        total_loss += current_loss
        losses.append(current_loss)

        # Update tqdm postfix to display the loss at the current batch
        progress_bar.set_postfix(loss=f"{current_loss:.4f}")

average_loss = total_loss / len(loader)
print(f"Epoch {epoch+1} Completed. Average Loss: {average_loss:.4f}\n")
