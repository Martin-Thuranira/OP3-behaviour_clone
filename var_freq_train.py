#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# PATHS TO DATASET
DATASET_DIR = "/home/webots/webots_mujoco/Assets/Outputs/var_test1"
MODEL_DIR = "/home/webots/webots_mujoco/Assets/Model/"
PLOTS_DIR = "/home/webots/webots_mujoco/media/"

def compute_gait_frequency(linear_x):
    """
    Compute gait frequency based on linear_x velocity.
    Different frequencies for forward vs backward walking.
    MUST MATCH THE FUNCTION IN ros2bag2_align.py
    
    Args:
        linear_x: forward/backward velocity in m/s
    
    Returns:
        gait_freq: adapted gait frequency in Hz
    """
    MIN_GAIT_FREQ = 0.8        # Hz - stopped
    FORWARD_FREQ = 1.2         # Hz - forward walking at +0.2 m/s
    BACKWARD_FREQ = 1.0        # Hz - backward walking at -0.2 m/s
    MAX_FORWARD_FREQ = 1.6     # Hz - fast forward (for future)
    MAX_BACKWARD_FREQ = 1.3    # Hz - fast backward (for future)
    
    MIN_VEL = 0.0              # m/s
    TARGET_VEL = 0.2           # m/s - our test velocity
    MAX_VEL = 0.4              # m/s
    
    if abs(linear_x) <= MIN_VEL:
        # Stopped
        return MIN_GAIT_FREQ
    
    elif linear_x > 0:
        # FORWARD walking
        if linear_x <= TARGET_VEL:
            # 0 to +0.2 m/s: interpolate MIN to FORWARD_FREQ
            ratio = linear_x / TARGET_VEL
            return MIN_GAIT_FREQ + ratio * (FORWARD_FREQ - MIN_GAIT_FREQ)
        elif linear_x <= MAX_VEL:
            # +0.2 to +0.4 m/s: interpolate FORWARD to MAX_FORWARD
            ratio = (linear_x - TARGET_VEL) / (MAX_VEL - TARGET_VEL)
            return FORWARD_FREQ + ratio * (MAX_FORWARD_FREQ - FORWARD_FREQ)
        else:
            return MAX_FORWARD_FREQ
    
    else:
        # BACKWARD walking (linear_x < 0)
        abs_vel = abs(linear_x)
        if abs_vel <= TARGET_VEL:
            # 0 to -0.2 m/s: interpolate MIN to BACKWARD_FREQ
            ratio = abs_vel / TARGET_VEL
            return MIN_GAIT_FREQ + ratio * (BACKWARD_FREQ - MIN_GAIT_FREQ)
        elif abs_vel <= MAX_VEL:
            # -0.2 to -0.4 m/s: interpolate BACKWARD to MAX_BACKWARD
            ratio = (abs_vel - TARGET_VEL) / (MAX_VEL - TARGET_VEL)
            return BACKWARD_FREQ + ratio * (MAX_BACKWARD_FREQ - BACKWARD_FREQ)
        else:
            return MAX_BACKWARD_FREQ

#LOAD NORMALISED DATA
def load_norm_stats(dataset_dir):
    obs_mean = np.load(os.path.join(dataset_dir, "obs_mean.npy"))
    obs_std = np.load(os.path.join(dataset_dir, "obs_std.npy"))
    act_mean = np.load(os.path.join(dataset_dir, "act_mean.npy"))
    act_std = np.load(os.path.join(dataset_dir, "act_std.npy"))
    obs_std = np.maximum(obs_std, 1e-6)
    act_std = np.maximum(act_std, 1e-6)
    return obs_mean, obs_std, act_mean, act_std

class BC_RNN_WithPrevAction(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=obs_dim + act_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, act_dim),
        )
    
    def forward(self, obs, prev_action, hidden=None):
        x = torch.cat([obs, prev_action], dim=-1)
        out, hidden = self.lstm(x, hidden)
        return self.net(out), hidden

# DATASET
class TrajSeqDataset(Dataset):
    def __init__(self, obs, cmd, act):
        self.obs = obs
        self.cmd = cmd
        self.act = act
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        # Only use linear_x from cmd (first element)
        # Set linear_y and angular_z to zero
        cmd_x = self.cmd[idx]  # This should be (seq_len,) containing linear_x only
        
        # Create full cmd with zeros for y and angular_z
        cmd_full = np.zeros((len(cmd_x), 3), dtype=np.float32)
        cmd_full[:, 0] = cmd_x  # Set linear_x
        # cmd_full[:, 1] = 0.0  # linear_y (already zero)
        # cmd_full[:, 2] = 0.0  # angular_z (already zero)
        
        x = np.concatenate([self.obs[idx], cmd_full], axis=-1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.act[idx], dtype=torch.float32),
        )

def clean(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# DATA LOADING
def load_dataset(d):
    obs_tr = clean(np.load(f"{d}/observations_train.npy"))
    obs_va = clean(np.load(f"{d}/observations_val.npy"))
    act_tr = clean(np.load(f"{d}/actions_train.npy"))
    act_va = clean(np.load(f"{d}/actions_val.npy"))
    
    # Load only linear_x commands (ignore linear_y and angular_z)
    cmd_x_tr = clean(np.load(f"{d}/linear_x_train.npy"))
    cmd_x_va = clean(np.load(f"{d}/linear_x_val.npy"))
    
    print(f"Loaded dataset:")
    print(f"  Observations: train={obs_tr.shape}, val={obs_va.shape}")
    print(f"  Actions: train={act_tr.shape}, val={act_va.shape}")
    print(f"  Linear_x: train={cmd_x_tr.shape}, val={cmd_x_va.shape}")
    print(f"  Linear_x range: train=[{cmd_x_tr.min():.3f}, {cmd_x_tr.max():.3f}], "
          f"val=[{cmd_x_va.min():.3f}, {cmd_x_va.max():.3f}]")
    
    return obs_tr, cmd_x_tr, act_tr, obs_va, cmd_x_va, act_va

# MODEL TRAINING
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        torch.backends.cudnn.benchmark = True
        print(f"cuDNN benchmark enabled")
    
    obs_tr, cmd_tr, act_tr, obs_va, cmd_va, act_va = load_dataset(args.dataset)
    obs_mean, obs_std, act_mean, act_std = load_norm_stats(args.dataset)
    
    print(f"\nDataset shapes:")
    print(f"  Observations: {obs_tr.shape}")
    print(f"  Commands (linear_x): {cmd_tr.shape}")
    print(f"  Actions: {act_tr.shape}")
    
    train_ds = TrajSeqDataset(obs_tr, cmd_tr, act_tr)
    val_ds = TrajSeqDataset(obs_va, cmd_va, act_va)
    
    num_workers = 2 if device == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Input dimension: obs + cmd (cmd is now 3D: [linear_x, linear_y=0, angular_z=0])
    obs_dim = obs_tr.shape[-1] + 3  # +3 for full cmd_vel vector
    act_dim = act_tr.shape[-1]
    
    print(f"\nModel dimensions:")
    print(f"  Input: {obs_dim} (obs={obs_tr.shape[-1]} + cmd=3)")
    print(f"  Output: {act_dim}")
    
    model = BC_RNN_WithPrevAction(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_val = float("inf")
    train_losses, val_losses = [], []
    
    print("\n" + "="*70)
    print("TRAINING WITH ADAPTIVE GAIT FREQUENCY")
    print("="*70)
    print("Forward (+0.2 m/s): 1.2 Hz")
    print("Backward (-0.2 m/s): 1.0 Hz")
    print("Stopped (0.0 m/s): 0.8 Hz")
    print("="*70 + "\n")
    
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        
        # Scheduled sampling probability
        ss_prob = args.ss_start + (args.ss_end - args.ss_start) * (
            epoch / max(1, args.epochs - 1)
        )
        
        for obs_b, act_b in train_loader:
            obs_b, act_b = obs_b.to(device, non_blocking=True), act_b.to(device, non_blocking=True)
            B, T, _ = obs_b.shape
            
            hidden = None
            prev = torch.zeros((B, 1, act_dim), device=device)
            preds = []
            
            for t in range(T):
                obs_t = obs_b[:, t:t+1]
                pred_t, hidden = model(obs_t, prev, hidden)
                preds.append(pred_t)
                
                # Scheduled sampling
                if torch.rand(1).item() < ss_prob:
                    prev = pred_t.detach()
                else:
                    prev = act_b[:, t:t+1, :]
            
            pred = torch.cat(preds, dim=1)
            loss = loss_fn(pred, act_b)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            
            total += loss.item()
        
        train_losses.append(total / len(train_loader))
        
        # VALIDATION
        model.eval()
        total = 0.0
        with torch.no_grad():
            for obs_b, act_b in val_loader:
                obs_b, act_b = obs_b.to(device), act_b.to(device)
                B, T, _ = obs_b.shape
                
                prev = torch.zeros((B, 1, act_dim), device=device)
                hidden = None
                preds = []
                
                for t in range(T):
                    obs_t = obs_b[:, t:t+1, :]
                    pred_t, hidden = model(obs_t, prev, hidden)
                    preds.append(pred_t)
                    prev = pred_t
                
                pred = torch.cat(preds, dim=1)
                loss = loss_fn(pred, act_b)
                total += loss.item()
        
        val_loss = total / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"[{epoch+1}/{args.epochs}] train={train_losses[-1]:.4f} "
              f"val={val_loss:.4f} ss={ss_prob:.2f}")
        
        # Save checkpoint model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out.replace(".pth", "_best.pth"))
            print(f"  → New best model saved!")
    
    torch.save(model.state_dict(), args.out)
    print("\n✓ Training complete")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train", linewidth=2)
    plt.plot(val_losses, label="val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Training Loss - Adaptive Gait Frequency")
    plt.savefig(os.path.join(PLOTS_DIR, "losses.png"), dpi=150)
    plt.close()
    print(f"✓ Plot saved to {PLOTS_DIR}/losses.png")
    
    # Additional analysis plot
    plt.figure(figsize=(12, 4))
    
    # Plot velocity distribution in training data
    plt.subplot(1, 2, 1)
    cmd_flat = cmd_tr.flatten()
    plt.hist(cmd_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Linear X Velocity (m/s)')
    plt.ylabel('Frequency')
    plt.title('Training Data: Velocity Distribution')
    plt.grid(alpha=0.3)
    
    # Plot frequency mapping
    plt.subplot(1, 2, 2)
    velocities = np.linspace(-0.3, 0.3, 100)
    frequencies = [compute_gait_frequency(v) for v in velocities]
    plt.plot(velocities, frequencies, linewidth=2, label='Adaptive Frequency')
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Min (0.8 Hz)')
    plt.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5, label='Backward (1.0 Hz)')
    plt.axhline(y=1.2, color='green', linestyle='--', alpha=0.5, label='Forward (1.2 Hz)')
    plt.axvline(x=0.2, color='green', linestyle=':', alpha=0.5)
    plt.axvline(x=-0.2, color='orange', linestyle=':', alpha=0.5)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Gait Frequency (Hz)')
    plt.title('Adaptive Frequency Mapping')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "adaptive_frequency_analysis.png"), dpi=150)
    plt.close()
    print(f"✓ Analysis plot saved to {PLOTS_DIR}/adaptive_frequency_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--normalize_actions", action="store_true")
    parser.add_argument("--ss_start", type=float, default=0.05)
    parser.add_argument("--ss_end", type=float, default=0.3)
    parser.add_argument("--out", type=str, default=os.path.join(MODEL_DIR, "var_freq.pth"))
    
    args = parser.parse_args()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    train(args)