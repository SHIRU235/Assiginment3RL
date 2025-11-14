Deep Q-Learning on Pong 
1. Project Overview
This project presents a complete implementation of a Deep Q-Network (DQN) agent trained to play Atari Pong using the PongDeterministic-v5 environment. The objective is to explore how different hyperparameters—such as batch size and target network update frequency—affect the agent's performance. The agent learns directly from raw pixel observations using a convolutional neural network (CNN).
2. Installation & Setup
To install all required packages, run the following commands:

pip install gymnasium[atari]
pip install ale-py
pip install autorom
AutoROM --accept-license

Ensure that the Atari ROMs are installed correctly before running training.
3. Environment Details
- Environment: PongDeterministic-v4
- Observation Space: 210×160×3 RGB images
- Action Space (0–5)
- Difficulty: Default
- Mode: Default

The deterministic version reduces randomness, making evaluation more stable.
4. Preprocessing Steps
Each frame undergoes preprocessing to reduce computation and improve learning:
- Crop the image to remove unnecessary borders
- Convert to grayscale
- Resize to 84×80
- Normalize pixel values
- Stack 4 consecutive frames to form the input state

Final state tensor shape: 4×84×80
5. Deep Q-Network Hyperparameters
The experiments compare multiple hyperparameters.

- Mini-batch size: **8 (default)** and **16**
- Discount factor (γ): **0.95**
- Target network update frequency: **every 10 episodes (default)** and **every 3 episodes**
- Exploration schedule:
  ε_initial = 1.0
  ε_decay = 0.995
  ε_min = 0.05

Replay buffer size and learning rate are set according to typical DQN standards.
6. CNN Architecture Description
The agent uses a convolutional neural network.

Input: 4×84×80
Layers:
- Conv2D → ReLU activation
- Conv2D → ReLU activation
- Flatten layer
- Fully Connected layer → ReLU
- Output layer: Q-values for each of the 6 actions

This architecture allows the model to extract spatial-temporal features from stacked frames.
7. Training Loop Overview
The DQN training loop consists of the following steps:
1. Reset the environment
2. Preprocess first frame
3. Stack 4 frames to form the initial state
4. Select action using ε-greedy policy
5. Execute action in environment
6. Store transition in replay memory
7. Sample a batch and train the online network
8. Update target network periodically
9. Track and log scores, rewards, and losses

The loop continues for a fixed number of episodes.
8. Metrics to Report
Each experiment should produce the following:
- Episode scores
- Mean reward (last 5 episodes)
- Loss curves
- Comparative plots:
  • Batch size 8 vs batch size 16
  • Target updates every 10 vs every 3 episodes

Metrics help evaluate training stability and convergence.
9. Deliverables
- Full DQN Python implementation
- A PDF report containing plots, discussions, and conclusions
- Explanation of the CNN architecture
- Summary of the best-performing hyperparameters
10. Recommended Folder Structure
dqn_agent.py          – DQN model implementation
replay_buffer.py      – Experience replay memory
train.py              – Main training loop
pong_utils.py         – Preprocessing utilities (frame processing, stacking, etc.)
report.pdf            – Final project report with results
README.md             – Documentation with setup instructions
11. Command to Run Experiments
Run the following commands:

python train.py --batch_size 8 --target_update 10
python train.py --batch_size 16 --target_update 10
python train.py --batch_size 8 --target_update 3
python train.py --batch_size 16 --target_update 3

Each configuration should be tested independently for accurate comparison.

