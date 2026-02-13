Behaviour Cloning for Velocity-Conditioned Bipedal Locomotion
Overview

This project implements an end-to-end imitation learning pipeline for training a recurrent neural network policy to control a position-actuated biped robot.

Demonstration data is collected via ROS in Webots, processed into structured training sequences, and used to train a Behaviour Cloning (BC) model. The trained policy is deployed and evaluated in MuJoCo under matched control frequencies.

The project focuses on:

Sequence modelling for control

Velocity-conditioned locomotion

Simulator gap mitigation

Control frequency harmonisation

Quantitative rollout evaluation

System Architecture
Data Collection

Source: ROS topics from Webots simulation

Signals collected:

Joint positions

IMU data

Center of Mass (COM)

cmd_vel velocity commands

Data exported and converted into aligned NumPy datasets

Observation Space

Joint positions

IMU orientation + angular velocity

COM position/velocity

Commanded velocity (cmd_vel)

Previous action (explicit conditioning)

Action Space

Joint position targets (position-controlled actuators in MuJoCo)

Model Architecture

Recurrent Behaviour Cloning Policy

Multi-layer LSTM

Temporal sequence input (B, T, D)

Previous-action conditioning

Fully connected output head for joint targets

Why LSTM?

Captures gait phase implicitly

Maintains temporal continuity

Improves stability over feedforward BC

Simulator Deployment (Webots → MuJoCo)

A major focus of this project was mitigating sim-to-sim transfer issues.

Control Frequency Alignment

sim_dt: physics timestep

ctrl_dt: policy update rate

Ensured:

Identical effective control frequency across simulators

Proper actuator gain tuning (kp) in MuJoCo position actuators

Consistent observation sampling rate

This reduced instability caused by timing mismatches.

Evaluation & Diagnostics

Implemented rollout evaluation framework:

Trajectory logging

Joint tracking error plots

Velocity tracking performance

Stability metrics

Episode survival duration

Gait consistency analysis

Plots generated automatically from rollout logs.

Tech Stack

Python

PyTorch

ROS

MuJoCo

NumPy

Matplotlib

Engineering Challenges Addressed

Simulator gap between Webots and MuJoCo

Temporal credit assignment in imitation learning

Recurrent policy stabilisation

Frequency mismatch between physics and control loop

Velocity-conditioned control generalisation

Results

Stable forward locomotion across multiple commanded velocities

Reduced tracking error after frequency harmonisation

Improved gait smoothness with previous-action conditioning

Successful deployment of trained policy in MuJoCo

Results

Stable forward locomotion across multiple commanded velocities

Reduced tracking error after frequency harmonisation

Improved gait smoothness with previous-action conditioning

Successful deployment of trained policy in MuJoCo
