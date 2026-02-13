#!/usr/bin/env python3
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import op3_env
import matplotlib.pyplot as plt


MODEL_PATH = "/home/webots/webots_mujoco/Assets/Model/var_freq_best.pth"
DATASET_DIR = "/home/webots/webots_mujoco/Assets/Outputs/var_test1"
PLOT_DIR = "/home/webots/webots_mujoco/media/var_freq"
ENV_ID = "op3_env/OP3Env-V0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OBS_DIM = 30  # (qpos + imu + com + phase) - matches training data
ACT_DIM = 12
CMD_DIM = 3
DT = 0.02  # 50 Hz

# ACTUATOR ORDER MATCHES XML
LEG_ACTUATORS = [
    "l_hip_yaw_act","l_hip_roll_act","l_hip_pitch_act",
    "l_knee_act","l_ank_pitch_act","l_ank_roll_act",
    "r_hip_yaw_act","r_hip_roll_act","r_hip_pitch_act",
    "r_knee_act","r_ank_pitch_act","r_ank_roll_act",
]

# CMD_VEL TEST PRESETS - Only forward/backward, no lateral or angular
CMD_STOP = np.array([0.0, 0.0, 0.0], dtype=np.float32)
CMD_FORWARD = np.array([0.2, 0.0, 0.0], dtype=np.float32)
CMD_BACKWARD = np.array([-0.2, 0.0, 0.0], dtype=np.float32)

def compute_gait_frequency(linear_x):
    """
    Compute gait frequency based on linear_x velocity.
    Different frequencies for forward vs backward walking.
    
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

class AdaptivePhaseOscillator:
    """Phase oscillator with velocity-dependent frequency."""
    def __init__(self, dt=0.02):
        self.phase = 0.0
        self.dt = dt
        self.freq_history = []
    
    def update(self, linear_x):
        """
        Update phase based on current linear_x velocity.
        
        Args:
            linear_x: forward/backward velocity in m/s
        
        Returns:
            phase_sin: sine of current phase
            freq: current gait frequency in Hz
        """
        freq = compute_gait_frequency(linear_x)
        self.freq_history.append(freq)
        
        # Integrate phase
        self.phase += 2 * np.pi * freq * self.dt
        self.phase = self.phase % (2 * np.pi)
        
        return np.sin(self.phase), freq
    
    def reset(self):
        """Reset phase to zero."""
        self.phase = 0.0
        self.freq_history = []

# CMD FILTER
class CmdVelFilter:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.cmd = np.zeros(3, dtype=np.float32)
    
    def update(self, target):
        self.cmd = (1 - self.alpha) * self.cmd + self.alpha * target
        return self.cmd

# MODEL MATCHES TRAINING
class BC_RNN_WithPrevAction(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + act_dim, 256, 2, batch_first=True, dropout=0.1)
        self.net = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )
    
    def forward(self, obs, prev, hidden=None):
        x = torch.cat([obs, prev], dim=-1)
        y, hidden = self.lstm(x, hidden)
        return self.net(y), hidden

def main():
    env = gym.make(ENV_ID, render_mode="human", height=720, width=1280)
    obs_raw, _ = env.reset()
    mj = env.unwrapped
    
    # Actuator mapping
    act_names = [mj.model.actuator(i).name for i in range(mj.model.nu)]
    leg_act_ids = [act_names.index(n) for n in LEG_ACTUATORS]
    
    # Load normalization
    obs_mean = np.load(os.path.join(DATASET_DIR, "obs_mean.npy")).astype(np.float32)
    obs_std = np.load(os.path.join(DATASET_DIR, "obs_std.npy")).astype(np.float32)
    act_mean = np.load(os.path.join(DATASET_DIR, "act_mean.npy")).astype(np.float32)
    act_std = np.load(os.path.join(DATASET_DIR, "act_std.npy")).astype(np.float32)
    cmd_mean = np.load(os.path.join(DATASET_DIR, "cmd_mean.npy")).astype(np.float32)
    cmd_std = np.load(os.path.join(DATASET_DIR, "cmd_std.npy")).astype(np.float32)
    
    print(f"Loaded normalization stats:")
    print(f"  obs_mean: {obs_mean.shape}, obs_std: {obs_std.shape}")
    print(f"  act_mean: {act_mean.shape}, act_std: {act_std.shape}")
    print(f"  cmd_mean: {cmd_mean.shape}, cmd_std: {cmd_std.shape}")
    
    # Load model
    model = BC_RNN_WithPrevAction(OBS_DIM + CMD_DIM, ACT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✓ Model loaded from {MODEL_PATH}")
    
    prev_action = torch.zeros(1, 1, ACT_DIM, device=DEVICE)
    hidden = None
    cmd_filter = CmdVelFilter(alpha=0.05)
    
    # Adaptive phase oscillator
    phase_osc = AdaptivePhaseOscillator(dt=DT)
    
    ctrl_prev = mj.data.ctrl.copy()
    alpha_ctrl = 0.3
    
    # Joint order mapping (MuJoCo to dataset)
    MUJOCO_TO_DATASET_MAP = [
        8, 9, 10, 11, 12, 13,
        14, 15, 16, 17, 18, 19,
        2, 3, 4,
        5, 6, 7,
        0, 1
    ]
    
    print("\n" + "="*70)
    print("TESTING ADAPTIVE GAIT FREQUENCY")
    print("="*70)
    print("\nTest sequence:")
    print("  Steps 0-2000:    STOP (0.0 m/s)")
    print("  Steps 2000-5000: FORWARD (0.2 m/s)")
    print("  Steps 5000-8000: BACKWARD (-0.2 m/s)")
    print("  Steps 8000+:     STOP (0.0 m/s)")
    print("\nExpected frequencies:")
    print("  At 0.0 m/s:   ~0.8 Hz (stopped)")
    print("  At +0.2 m/s:  ~1.2 Hz (forward)")
    print("  At -0.2 m/s:  ~1.0 Hz (backward)")
    print("="*70 + "\n")

    # ---------------- DEBUG LOGGING ----------------
    log_time = []
    log_freq = []
    log_phase = []

    log_l_hip = []
    log_r_hip = []

    log_l_knee = []
    log_r_knee = []

    log_l_ankle = []
    log_r_ankle = []
    # ------------------------------------------------

    
    try:
        step = 0
        MAX_STEPS = 600
        while step < MAX_STEPS:
            # TEST SEQUENCE: Stop -> Forward -> Backward -> Stop
            if step < 100:
                cmd_target = CMD_FORWARD
                phase_name = "FORWARD"
            elif step < 250:
                cmd_target = CMD_STOP
                phase_name = "STOP"
            elif step < 400:
                cmd_target = CMD_BACKWARD
                phase_name = "BACKWARD"
            else:
                cmd_target = CMD_STOP
                phase_name = "STOP"
            
            cmd = cmd_filter.update(cmd_target)
            
            # ADAPTIVE PHASE - only based on linear_x (cmd[0])
            phase_sin, current_freq = phase_osc.update(cmd[0])
            
            # Extract and reorder joints
            qpos_mujoco = obs_raw[7:27]
            qpos = qpos_mujoco[MUJOCO_TO_DATASET_MAP]
            
            imu_a = obs_raw[40:43].astype(np.float32)
            imu_g = obs_raw[43:46].astype(np.float32)
            com = obs_raw[46:49].astype(np.float32)
            
            # Build full observation (matches training: 20 joints + 3 imu_a + 3 imu_g + 3 com + 1 phase = 30D)
            o_raw = np.concatenate([
                qpos,
                imu_a,
                imu_g,
                com,
                [phase_sin]
            ]).astype(np.float32)  # Total: 30D
            
            # Build input (obs + cmd) - NO normalization needed
            # Model was trained on raw data
            cmd_full = np.zeros(3, dtype=np.float32)
            cmd_full[0] = cmd[0]  # linear_x
            # cmd_full[1] = 0.0  # linear_y (already zero)
            # cmd_full[2] = 0.0  # angular_z (already zero)
            
            o_full = np.concatenate([o_raw, cmd_full]).astype(np.float32)  # 33D (30 obs + 3 cmd)
            
            o_t = torch.from_numpy(o_full).view(1, 1, -1).to(DEVICE)
            
            with torch.no_grad():
                a_t, hidden = model(o_t, prev_action, hidden)
                a = a_t[0, 0].cpu().numpy()  # Model outputs raw actions directly
            
            ctrl = ctrl_prev.copy()
            for i, aid in enumerate(leg_act_ids):
                ctrl[aid] = (1 - alpha_ctrl) * ctrl[aid] + alpha_ctrl * a[i]
            
            obs_raw, _, _, _, _ = env.step(ctrl)
            ctrl_prev[:] = ctrl
            prev_action = a_t

            # ---------- LOG DATA FOR DEBUG PLOTS ----------
            t = step * DT

            # Extract joint positions from MuJoCo
            # (base offset 7, matches your indexing earlier)
            l_hip = mj.data.qpos[7 + 10]     # l_hip_pitch
            r_hip = mj.data.qpos[7 + 16]     # r_hip_pitch

            l_knee = mj.data.qpos[7 + 11]
            r_knee = mj.data.qpos[7 + 17]

            l_ank = mj.data.qpos[7 + 12]     # l_ank_pitch
            r_ank = mj.data.qpos[7 + 18]     # r_ank_pitch

            log_time.append(t)
            log_freq.append(current_freq)
            log_phase.append(phase_sin)

            log_l_hip.append(l_hip)
            log_r_hip.append(r_hip)

            log_l_knee.append(l_knee)
            log_r_knee.append(r_knee)

            log_l_ankle.append(l_ank)
            log_r_ankle.append(r_ank)
            # ------------------------------------------------

            
            # Enhanced logging with phase information
            if step % 100 == 0:
                l_knee = mj.data.qpos[7+11]
                r_knee = mj.data.qpos[7+17]
                print(f"[{step:05d}] {phase_name:8s} | "
                      f"cmd={cmd[0]:+.2f} m/s | "
                      f"freq={current_freq:.2f} Hz | "
                      f"phase={phase_sin:+.2f} | "
                      f"knees: L={l_knee:+.3f} R={r_knee:+.3f}")
            
            # Phase transition notifications
            if step == 100:
                print("\n" + "→"*35)
                print("TRANSITION: Starting FORWARD walk (0.2 m/s)")
                print("Expected frequency: 0.8 Hz → 1.2 Hz")
                print("→"*35 + "\n")
            elif step == 250:
                print("\n" + "→"*35)
                print("TRANSITION: Starting BACKWARD walk (-0.2 m/s)")
                print("Expected frequency: 1.2 Hz → 1.0 Hz")
                print("→"*35 + "\n")
            elif step == 400:
                print("\n" + "→"*35)
                print("TRANSITION: Stopping")
                print("Expected frequency: 1.2 Hz → 0.8 Hz")
                print("→"*35 + "\n")
            
            step += 1
            
    finally:
        env.close()
        
        # Print frequency statistics
        if phase_osc.freq_history:
            print("\n" + "="*70)
            print("FREQUENCY STATISTICS")
            print("="*70)
            freq_array = np.array(phase_osc.freq_history)
            print(f"Mean frequency: {freq_array.mean():.3f} Hz")
            print(f"Min frequency:  {freq_array.min():.3f} Hz")
            print(f"Max frequency:  {freq_array.max():.3f} Hz")
            print(f"Std deviation:  {freq_array.std():.3f} Hz")
            print("="*70)
        if len(log_time) > 0:
            plot_debug_results(
                log_time, log_freq, log_phase,
                log_l_hip, log_r_hip,
                log_l_knee, log_r_knee,
                log_l_ankle, log_r_ankle
            )


def plot_debug_results(time, freq, phase,
                       l_hip, r_hip,
                       l_knee, r_knee,
                       l_ank, r_ank):
    
    time = np.array(time)
    freq = np.array(freq)

    def joint_plot(title, left_joint, right_joint):
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        ax1.set_title(title)
        ax1.plot(time, left_joint, label="Left Joint")
        ax1.plot(time, right_joint, label="Right Joint")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Joint Position (rad)")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Secondary axis for gait frequency
        ax2 = ax1.twinx()
        ax2.plot(time, freq, linestyle="--", alpha=0.7, label="Gait Frequency")
        ax2.set_ylabel("Gait Frequency (Hz)")
        
        plt.tight_layout()

        filename = title.replace(" ", "_") + ".png"
        filepath = os.path.join(PLOT_DIR, filename)

        plt.savefig(filepath, dpi=300)
        plt.close()

    joint_plot("Hip Pitch vs Gait Frequency",
            l_hip, r_hip)

    joint_plot("Knee Joint vs Gait Frequency",
            l_knee, r_knee)

    joint_plot("Ankle Pitch vs Gait Frequency",
            l_ank, r_ank)


if __name__ == "__main__":
    main()