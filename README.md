# Deep Reinforcement Learning Maze Solver (C++ & SDL3)

This repository contains a fully custom, dependency-free Reinforcement Learning backend written entirely in C++ that drives an AI agent to learn and solve dynamically generated mazes.

## Overview
Unlike typical Deep Reinforcement Learning (DRL) implementations that heavily rely on high-level Python libraries (such as TensorFlow or PyTorch), this project engineers a multi-layer Deep Q-Network (DQN) entirely from scratch using bare-metal C++ constructs. The environment leverages SDL3 for rendering the learning process in real-time, while Native Windows Multimedia (`winmm`) provides explicit audio feedback.

---

## 🧠 How the Artificial Intelligence Works

### 1. The Deep Q-Network (DQN) Architecture
The brain of the agent is a custom Multi-Layer Perceptron (MLP) trained via Stochastic Gradient Descent.
- **Topology:** The network has an Input Layer of 8 features, Two Hidden Layers of 128 nodes each (using ReLU activations to prevent vanishing gradients), and a Linear Output Layer of 4 nodes.
- **Action Space:** The 4 output nodes correspond to the approximate Q-Values (expected future rewards) for moving **Up, Down, Left,** or **Right**.

### 2. State Representation (What the Agent "Sees")
At every step, the agent converts its surroundings into an 8-dimensional mathematical state vector to feed into the neural network:
1. `X / Width`: Normalized horizontal coordinate.
2. `Y / Width`: Normalized vertical coordinate.
3. `Up Open`: Boolean (1 or 0) if the upward cell is a path.
4. `Down Open`: Boolean if the downward cell is a path.
5. `Left Open`: Boolean if the leftward cell is a path.
6. `Right Open`: Boolean if the rightward cell is a path.
7. `Distance X`: Normalized horizontal distance to the ultimate goal.
8. `Distance Y`: Normalized vertical distance to the ultimate goal.

### 3. Reward Engineering (How the Agent Learns)
Through trial and error, the agent receives discontinuous rewards based on its interactions. These rewards permanently alter its neural weights via Backpropagation:
- **Victory (+100):** Successfully navigating to the exact end coordinate.
- **Wall Collision (-5):** Attempting to walk into a solid wall.
- **Living Penalty (-0.1):** A minor penalty exactly every single step taken, forcing the agent to discover the mathematically shortest path rather than taking long scenic routes.
- **Start Backtracking (-10):** A heavy penalty to discourage walking all the way back to the initialization origin.

### 4. Experience Replay & Optimization
As the agent explores, it continuously saves its experiences `(state, action, reward, next_state)` into a cyclic Replay Buffer. During the neural training step, it randomly samples a "mini-batch" of 64 past experiences to train on. This breaks temporal correlations and heavily stabilizes the learning curve. It also operates a separate frozen "Target Network" to ensure gradient targets don't constantly shift during loss calculations.

---

## 🗺️ Environment & Maze Generation

### Recursive Backtracking
The environment utilizes a randomized Depth-First Search algorithm to carve out passages. This mathematically guarantees the generated maze is a "Spanning Tree" (meaning there are zero isolated islands and exactly one unique path between any two points).

### Maze Size Calculation (`2N + 1`)
When you input a size `n` (e.g., `5`), the program physically generates a grid that is **2n + 1** squares wide and tall (e.g., `11x11`). 
This arithmetic is required because the Recursive Backtracking algorithm creates discrete branching paths that are exactly 1 block wide, separated by walls that must also be 1 block wide, plus a solid boundary wall on all outer edges.

---

## 📉 Why does the Agent wander initially?
In Reinforcement Learning, the agent uses an **$\epsilon$-greedy policy**. 
- At the start of training (Episode 0), $\epsilon = 1.0$. This means the agent inherently takes **completely random actions 100% of the time** to blindly map out and explore the environment. Random walks often result in moving back and forth aimlessly in the same hall.
- As episodes pass, $\epsilon$ slowly decays geometrically down to a minimum floor of `0.05` (5%). 
- As $\epsilon$ drops, the agent abandons random guessing and relies entirely on its deep neural network weights to "exploit" what it has learned, magically calculating the route straight to the goal!

---

## 🎮 Controls & Interface

While the `maze_solver.exe` window is focused:
- **[SPACE]** - Toggle Fast-Forward Mode. Switches between visualizing the agent's movements exactly step-by-step (slow simulation), and bypassing the frame limits to aggressively calculate hundreds of neural gradients per second (Extremely Fast!).

### Visual Guide
- **Dark Grey**: Impassable Walls.
- **White**: Unexplored Valid Paths.
- **Red**: Cells the agent has explored during learning.
- **Green**: The current best known path to the exit. Note that the green path updates dynamically!
- **Cyan & Magenta**: Start and Target End blocks, respectively.
- **Blue Trail**: The visual trace of the exact route the agent took in its most recent steps.

### Audio Cues
Native Windows `winmm` audio is directly mapped to the RL loop:
- **Correct.mp3**: Plays immediately when the agent successfully exploits a path to the goal during normal visualization.
- **Wrong.mp3**: Plays if the agent exceeds the mathematical timeout step limit ($2 \times Width^2$) and is forcibly reset due to infinite loops in stochastic dead-ends.

---

## 🛠️ Building and Running

To compile natively on Windows via `g++`:
```powershell
g++ -O3 maze_solver.cpp -Iinclude -Llib -lmingw32 -lSDL3 -lwinmm -o maze_solver
```
Run the executable directly:
```powershell
.\maze_solver
```