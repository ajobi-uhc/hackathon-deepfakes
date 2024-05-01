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
import wandb
import torch
from tqdm import tqdm
print("Initializing the environment...")

DATASET_VIDEO_PATH = "/raid/datasets/hackathon2024/resized_dataset/train_dataset"
DATASET_METADATA_PATH = "/raid/datasets/hackathon2024/resized_dataset/train_dataset/metadata.json"
FRAME_RATE = 1  # Frame rate to sample (e.g., 1 frame per second)

# Load video metadata
df_labels = pd.read_json(DATASET_METADATA_PATH, orient='index')
df_labels.reset_index(inplace=True)
df_labels.columns = ['Filename', 'Label']
df_labels['label_value'] = np.where(df_labels['Label'] == 'real', 1, 0)
print("Initializing data loader...")
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
print("Generating Dataset...")
dataset = VideoDataset(df_labels, DATASET_VIDEO_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Fetching model...")
model = timm.create_model('vit_large_patch14_clip_224', pretrained=True)

model.head = nn.Identity()

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze all the parameters in the model to prevent them from being updated during training
for param in model.parameters():
    param.requires_grad = False
    
print("Setting the parameters...")
# Define your classifier that will take the output features from the transformer
output_dims_of_CLIP = 1024  # This should match the output features of the last layer before the head
classifier = nn.Linear(in_features=output_dims_of_CLIP, out_features=1).to(device)
criterion = nn.BCEWithLogitsLoss()
initial_lr = 1e-4
optimizer = torch.optim.Adam(classifier.parameters(), lr=initial_lr)
# scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

print("Login in wandb...")
wandb.login(key='1febd470895e910d7247a866fc41ab6966fe3476')
wandb.init(project="deepfake-v1-clip", entity="aryajakkli2002")
wandb.watch(classifier, log='all', log_freq=10)

print("Start training...")
num_epochs = 1  # Example epoch count
losses = []  # List to store all losses for visualization or further analysis
save_interval = len(loader) // 10
print("save interval", save_interval)
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
        wandb.log({
            "batch_loss": current_loss,
        })
        losses.append(current_loss)

        # Update tqdm postfix to display the loss at the current batch
        progress_bar.set_postfix(loss=f"{current_loss:.4f}")
        if batch_idx % save_interval == 0 or batch_idx == len(loader):
            print("saving at", batch_idx)
            checkpoint_path = f'checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pth'
            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

average_loss = total_loss / len(loader)
print(f"Epoch {epoch+1} Completed. Average Loss: {average_loss:.4f}\n")


torch.save({
    'classifier_state_dict': classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': video_loss.item()
}, 'model_checkpoint.pth')
