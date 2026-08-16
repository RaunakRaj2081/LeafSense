# ==============================================
# COMPLETE MEDICAL IMAGE CAPTIONING IMPLEMENTATION
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# NLP imports
import re
from collections import Counter, OrderedDict

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==============================================
# 1. SET DATASET PATH FROM KAGGLEHUB
# ==============================================

# Use the path from kagglehub download
dataset_path = "/root/.cache/kagglehub/datasets/virajbagal/roco-dataset/versions/1"
print(f"Dataset path: {dataset_path}")

# ==============================================
# 2. EXPLORE DATASET STRUCTURE
# ==============================================

def explore_dataset(base_path):
    """Explore the dataset structure"""
    print("Exploring dataset structure...")
    print("="*60)
    
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(base_path, 'all_data', split, 'radiology')
        
        if os.path.exists(split_path):
            images_dir = os.path.join(split_path, 'images')
            num_images = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
            
            # Check for CSV files
            csv_files = [f for f in os.listdir(split_path) if f.endswith('.csv')]
            
            print(f"{split.upper():12s} - Images: {num_images:6d}")
            print(f"  CSV files: {csv_files}")
            
            if csv_files:
                # Show first few rows of CSV
                csv_path = os.path.join(split_path, csv_files[0])
                try:
                    df = pd.read_csv(csv_path)
                    print(f"  CSV columns: {df.columns.tolist()}")
                    print(f"  CSV rows: {len(df)}")
                except:
                    pass
            print()

# Explore the dataset
explore_dataset(dataset_path)

# ==============================================
# 3. CREATE VOCABULARY CLASS
# ==============================================

class MedicalVocabulary:
    """Vocabulary class for medical image captioning"""
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
    
    def preprocess_text(self, text):
        """Clean and preprocess medical text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep medical notations
        text = re.sub(r'[^\w\s.,;:()\[\]-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = self.preprocess_text(text)
        # Split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocabulary(self, caption_list):
        """Build vocabulary from captions"""
        frequencies = Counter()
        idx = 4
        
        print(f"Building vocabulary from {len(caption_list)} captions...")
        
        for i, caption in enumerate(caption_list):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(caption_list)} captions...")
            
            tokens = self.tokenize(caption)
            for token in tokens:
                frequencies[token] += 1
                
                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
        
        print(f"Vocabulary built with {len(self.itos)} tokens")
        print(f"Top 20 most common tokens:")
        for word, freq in frequencies.most_common(20):
            print(f"  {word}: {freq}")
    
    def numericalize(self, text, max_length=50):
        """Convert text to numerical indices"""
        tokens = self.tokenize(text)
        
        # Limit tokens
        tokens = tokens[:max_length-2]
        
        # Convert to indices
        numericalized = [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
        
        # Add start and end tokens
        numericalized = [self.stoi["<start>"]] + numericalized + [self.stoi["<end>"]]
        
        # Pad if needed
        if len(numericalized) < max_length:
            numericalized = numericalized + [self.stoi["<pad>"]] * (max_length - len(numericalized))
        
        return numericalized[:max_length]
    
    def indices_to_text(self, indices):
        """Convert indices back to text"""
        tokens = []
        for idx in indices:
            if idx == self.stoi["<pad>"]:
                continue
            if idx == self.stoi["<end>"]:
                break
            
            token = self.itos.get(idx, "<unk>")
            if token not in ["<start>", "<end>", "<pad>", "<unk>"]:
                tokens.append(token)
        
        return " ".join(tokens)
    
    def __len__(self):
        return len(self.itos)

# ==============================================
# 4. CREATE DATASET CLASS
# ==============================================

class MedicalImageCaptioningDataset(Dataset):
    """Dataset class for medical images and captions"""
    def __init__(self, split='train', transform=None, max_length=50, sample_size=None):
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.base_path = os.path.join(dataset_path, 'all_data')
        
        # Load data
        self.image_paths, self.captions = self.load_data(sample_size)
        
        # Create vocabulary if this is training set
        if split == 'train':
            self.vocab = MedicalVocabulary(freq_threshold=2)
            self.vocab.build_vocabulary(self.captions)
        else:
            # For validation/test, we'll use vocabulary from training
            self.vocab = None
        
        print(f"{split.upper()} dataset: {len(self)} samples")
    
    def load_data(self, sample_size=None):
        """Load image paths and captions"""
        image_paths = []
        captions = []
        
        split_path = os.path.join(self.base_path, self.split, 'radiology')
        images_dir = os.path.join(split_path, 'images')
        
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return [], []
        
        # Find CSV file
        csv_files = [f for f in os.listdir(split_path) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV file found in {split_path}")
            return [], []
        
        csv_path = os.path.join(split_path, csv_files[0])
        print(f"Loading data from: {csv_path}")
        
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            print(f"CSV loaded with {len(df)} rows")
            
            # Find appropriate columns
            id_col = None
            caption_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'id' in col_lower or 'image' in col_lower:
                    id_col = col
                elif 'caption' in col_lower or 'text' in col_lower or 'title' in col_lower:
                    caption_col = col
            
            if not id_col or not caption_col:
                # Use first two columns as fallback
                id_col = df.columns[0]
                caption_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            
            print(f"Using columns: ID='{id_col}', Caption='{caption_col}'")
            
            # Process data
            processed = 0
            skipped = 0
            
            for idx, row in df.iterrows():
                if sample_size and processed >= sample_size:
                    break
                
                try:
                    # Get image ID
                    img_id = str(row[id_col]).strip()
                    if not img_id:
                        skipped += 1
                        continue
                    
                    # Get caption
                    caption = str(row[caption_col]).strip()
                    if not caption or pd.isna(caption):
                        skipped += 1
                        continue
                    
                    # Find image file
                    img_name = img_id
                    
                    # Try different extensions
                    possible_paths = [
                        os.path.join(images_dir, f"{img_name}"),
                        os.path.join(images_dir, f"{img_name}.jpg"),
                        os.path.join(images_dir, f"{img_name}.png"),
                    ]
                    
                    img_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            img_path = path
                            break
                    
                    if not img_path:
                        # Try to find by partial match
                        matching_files = [f for f in os.listdir(images_dir) 
                                       if img_id in f and f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                        if matching_files:
                            img_path = os.path.join(images_dir, matching_files[0])
                        else:
                            skipped += 1
                            continue
                    
                    image_paths.append(img_path)
                    captions.append(caption)
                    processed += 1
                    
                    if processed % 1000 == 0:
                        print(f"  Loaded {processed} samples...")
                        
                except Exception as e:
                    skipped += 1
                    continue
            
            print(f"Successfully loaded {processed} samples, skipped {skipped} samples")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return [], []
        
        return image_paths, captions
    
    def set_vocabulary(self, vocab):
        """Set vocabulary for validation/test sets"""
        self.vocab = vocab
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
                image = transform(image)
        except Exception as e:
            # Return blank image if loading fails
            image = torch.zeros((3, 224, 224))
        
        # Get caption
        caption = self.captions[idx]
        
        # Convert to indices
        if self.vocab:
            caption_indices = self.vocab.numericalize(caption, self.max_length)
            caption_tensor = torch.tensor(caption_indices, dtype=torch.long)
        else:
            caption_tensor = torch.zeros(self.max_length, dtype=torch.long)
        
        return image, caption_tensor, caption, img_path

# ==============================================
# 5. CREATE DATASETS
# ==============================================

print("\n" + "="*60)
print("CREATING DATASETS")
print("="*60)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Create datasets with reasonable sample sizes
TRAIN_SAMPLES = 10000  # Increased for better training
VAL_SAMPLES = 2000
TEST_SAMPLES = 2000

print("Creating training dataset...")
train_dataset = MedicalImageCaptioningDataset(
    split='train',
    transform=train_transform,
    max_length=50,
    sample_size=TRAIN_SAMPLES
)

print("\nCreating validation dataset...")
val_dataset = MedicalImageCaptioningDataset(
    split='validation',
    transform=val_test_transform,
    max_length=50,
    sample_size=VAL_SAMPLES
)

print("\nCreating test dataset...")
test_dataset = MedicalImageCaptioningDataset(
    split='test',
    transform=val_test_transform,
    max_length=50,
    sample_size=TEST_SAMPLES
)

# Use training vocabulary for validation and test
val_dataset.set_vocabulary(train_dataset.vocab)
test_dataset.set_vocabulary(train_dataset.vocab)

print(f"\nDataset Statistics:")
print(f"Training:   {len(train_dataset):6d} samples")
print(f"Validation: {len(val_dataset):6d} samples")
print(f"Test:       {len(test_dataset):6d} samples")
print(f"Vocabulary size: {len(train_dataset.vocab)}")

# ==============================================
# 6. VISUALIZE SAMPLES
# ==============================================

print("\n" + "="*60)
print("VISUALIZING SAMPLE DATA")
print("="*60)

def visualize_samples(dataset, num_samples=3):
    """Visualize sample images and captions"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        image, caption_tensor, caption, img_path = dataset[i]
        
        # Convert image for display
        img_np = image.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Image from: {os.path.basename(img_path)[:30]}", fontsize=9)
        axes[i].axis('off')
        
        # Show caption
        short_caption = caption[:80] + "..." if len(caption) > 80 else caption
        print(f"Sample {i+1} Caption: {short_caption}")
        print()
    
    plt.tight_layout()
    plt.show()

print("Training samples:")
visualize_samples(train_dataset, num_samples=3)

# ==============================================
# 7. CREATE DATALOADERS
# ==============================================

def collate_fn(batch):
    """Collate function for batching"""
    images = []
    captions = []
    raw_captions = []
    img_paths = []
    
    for image, caption, raw_caption, img_path in batch:
        images.append(image)
        captions.append(caption)
        raw_captions.append(raw_caption)
        img_paths.append(img_path)
    
    images = torch.stack(images, dim=0)
    captions = torch.stack(captions, dim=0)
    
    return images, captions, raw_captions, img_paths

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn
)

print(f"\nDataLoader Statistics:")
print(f"Batch size: {batch_size}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ==============================================
# 8. MODEL ARCHITECTURE
# ==============================================

class ImageEncoder(nn.Module):
    """CNN Encoder for images"""
    def __init__(self, embed_size, train_cnn=False):
        super(ImageEncoder, self).__init__()
        
        # Use ResNet50 pretrained on ImageNet
        resnet = torchvision.models.resnet50(pretrained=True)
        
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Freeze CNN if not training
        if not train_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Project to embedding space
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        """Extract features from images"""
        with torch.no_grad():
            features = self.resnet(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        features = self.dropout(features)
        
        return features

class Attention(nn.Module):
    """Attention Mechanism"""
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_out, decoder_hidden):
        """Calculate attention weights"""
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1)))
        alpha = self.softmax(att.squeeze(2))
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha

class DecoderRNN(nn.Module):
    """Decoder with Attention"""
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, encoder_dim=256, dropout=0.5):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
    
    def init_hidden_state(self, encoder_out):
        """Initialize hidden state"""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """Forward pass during training"""
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        embeddings = self.embedding(encoded_captions)
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold predictions
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        
        # Decode sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t],
                h[:batch_size_t]
            )
            
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.lstm(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            preds = self.fc(self.dropout_layer(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
    
    def sample(self, encoder_out, vocab, max_length=50):
        """Generate caption for a single image"""
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        h, c = self.init_hidden_state(encoder_out)
        
        start_token = vocab.stoi["<start>"]
        word = torch.tensor([[start_token]] * batch_size).to(device)
        
        captions = []
        alphas = []
        
        for t in range(max_length):
            embeddings = self.embedding(word).squeeze(1)
            
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            h, c = self.lstm(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )
            
            preds = self.fc(self.dropout_layer(h))
            predicted = preds.argmax(1)
            
            captions.append(predicted.item())
            alphas.append(alpha.cpu().detach().numpy())
            
            if predicted.item() == vocab.stoi["<end>"]:
                break
            
            word = predicted.unsqueeze(1)
        
        # Convert indices to words
        caption_words = [vocab.itos[idx] for idx in captions]
        
        # Remove special tokens
        final_caption = []
        for word in caption_words:
            if word in ["<start>", "<end>", "<pad>", "<unk>"]:
                continue
            final_caption.append(word)
        
        return " ".join(final_caption), alphas

class MedicalImageCaptionModel(nn.Module):
    """Complete Medical Image Captioning Model"""
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super(MedicalImageCaptionModel, self).__init__()
        
        self.encoder = ImageEncoder(embed_size, train_cnn=False)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_dim=embed_size
        )
    
    def forward(self, images, captions, lengths):
        """Forward pass"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs
    
    def generate_caption(self, image, vocab, max_length=50):
        """Generate caption for a single image"""
        features = self.encoder(image)
        caption, alphas = self.decoder.sample(features, vocab, max_length)
        return caption, alphas

# ==============================================
# 9. INITIALIZE MODEL
# ==============================================

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

vocab_size = len(train_dataset.vocab)
print(f"Vocabulary size: {vocab_size}")

model = MedicalImageCaptionModel(
    vocab_size=vocab_size,
    embed_size=256,
    hidden_size=512
).to(device)

print(f"Model created on {device}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ==============================================
# 10. TRAINING UTILITIES
# ==============================================

def train_epoch(model, dataloader, criterion, optimizer, epoch, grad_clip=5.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (images, captions, _, _) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        # Get caption lengths
        caption_lengths = (captions != 0).sum(dim=1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
        
        # Remove start token from targets
        targets = caps_sorted[:, 1:]
        
        # Create mask for valid positions
        mask = (targets != 0).float()
        
        # Calculate loss
        loss = criterion(outputs.permute(0, 2, 1), targets)
        loss = (loss * mask).sum() / mask.sum()
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Update statistics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / total_samples
    return avg_loss

def validate(model, dataloader, criterion):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, captions, _, _ in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            caption_lengths = (captions != 0).sum(dim=1)
            
            outputs, caps_sorted, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)
            
            targets = caps_sorted[:, 1:]
            mask = (targets != 0).float()
            
            loss = criterion(outputs.permute(0, 2, 1), targets)
            loss = (loss * mask).sum() / mask.sum()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    return avg_loss

def save_checkpoint(epoch, model, optimizer, vocab, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab': vocab,
        'vocab_size': len(vocab)
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# ==============================================
# 11. TRAIN MODEL
# ==============================================

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# Training parameters
num_epochs = 5
learning_rate = 1e-4

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1, verbose=True
)

train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    print("-" * 40)
    
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)
    train_losses.append(train_loss)
    print(f"Train Loss: {train_loss:.4f}")
    
    # Validate
    val_loss = validate(model, val_loader, criterion)
    val_losses.append(val_loss)
    print(f"Val Loss: {val_loss:.4f}")
    
    # Update learning rate
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            epoch, model, optimizer, train_dataset.vocab,
            '/kaggle/working/best_medical_caption_model.pth'
        )
    
    # Generate sample caption
    if epoch % 1 == 0:
        print("\nGenerating sample caption...")
        model.eval()
        with torch.no_grad():
            val_images, _, val_captions, _ = next(iter(val_loader))
            val_image = val_images[0:1].to(device)
            
            generated, _ = model.generate_caption(val_image, train_dataset.vocab)
            
            print(f"Generated: {generated}")
            print(f"True: {val_captions[0][:100]}...")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
plt.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==============================================
# 12. EVALUATE ON TEST SET
# ==============================================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60)

def evaluate_and_visualize(model, dataloader, vocab, num_samples=3):
    """Evaluate and visualize results"""
    model.eval()
    
    generated_captions = []
    true_captions = []
    images_list = []
    
    with torch.no_grad():
        for i, (images, captions, raw_captions, _) in enumerate(dataloader):
            if i >= 1:  # Use only first batch
                break
            
            images = images.to(device)
            
            for j in range(min(num_samples, len(images))):
                img = images[j:j+1]
                generated, _ = model.generate_caption(img, vocab)
                
                generated_captions.append(generated)
                true_captions.append(raw_captions[j])
                images_list.append(images[j].cpu())
    
    # Display results
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3*num_samples))
    
    for i in range(num_samples):
        # Show image
        img_np = images_list[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Test Image {i+1}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Show captions
        true_text = true_captions[i]
        gen_text = generated_captions[i]
        
        true_short = true_text[:120] + "..." if len(true_text) > 120 else true_text
        gen_short = gen_text[:120] + "..." if len(gen_text) > 120 else gen_text
        
        caption_text = f"TRUE CAPTION:\n{true_short}\n\n"
        caption_text += f"GENERATED:\n{gen_short}"
        
        axes[i, 1].text(0.05, 0.95, caption_text, fontsize=9,
                       verticalalignment='top', transform=axes[i, 1].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return generated_captions, true_captions

print("Generating captions for test images...")
generated, true = evaluate_and_visualize(model, test_loader, train_dataset.vocab, num_samples=3)

# ==============================================
# 13. SAVE FINAL MODEL
# ==============================================

print("\n" + "="*60)
print("SAVING FINAL MODEL")
print("="*60)

# Save final model
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'vocab': train_dataset.vocab,
    'vocab_size': len(train_dataset.vocab),
    'embed_size': 256,
    'hidden_size': 