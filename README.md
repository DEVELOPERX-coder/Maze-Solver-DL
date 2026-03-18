# Deep Reinforcement Learning Maze Solver (C++ \& SDL3)

This repository contains a fully custom Reinforcement Learning backend written entirely in C++ that drives an agent to learn and solve dynamically generated mazes.

## Features
- **Custom Neural Network**: A Multi-Layer Perceptron (MLP) built from scratch without external ML frameworks like PyTorch or TensorFlow.
- **Deep Q-Learning (DQN)**: Complete implementation of Experience Replay, Target Networks, and $\epsilon$-greedy exploration.
- **Procedural Maze Generation**: Uses Recursive Backtracking to generate guaranteed-solvable square mazes.
- **Live SDL3 Visualization**: The environment renders via hardware-accelerated SDL3 graphics, color coding walls, paths, and agent explorations.

## Why does the Agent wander on the same paths initially?
In Reinforcement Learning, the agent uses an **$\epsilon$-greedy policy**. 
- At the start of training (Episode 0), $\epsilon = 1.0$. This means the agent takes **completely random actions 100% of the time** to explore the environment. Random walks often result in moving back and forth in the same hall.
- As episodes pass, $\epsilon$ slowly decays down to `0.05` (5%). 
- As $\epsilon$ drops, the agent relies more heavily on the Neural Network approximations to "exploit" the shortest path it has learned, drastically reducing wandering!

## Controls
While the `maze_solver.exe` window is focused:
- **[SPACE]** - Toggle Fast-Forward Mode. Switches between visualizing the agent's movements exactly step-by-step (slow), and calculating multiple gradients per frame to aggressively speed up the training (fast!).

## Visual Guide
- **Dark Grey**: Impassable Walls
- **White**: Unexplored Valid Paths
- **Red**: Cells the agent has explored during learning.
- **Green**: The current best known path to the exit.
- **Cyan/Magenta**: Start and Target End Blocks, respectively.
- **Blue Trail**: The trace of the exact route the agent took in its most recent steps.

## Building and Running
To compile natively on Windows via `g++`:
```powershell
g++ -O3 maze_solver.cpp -Iinclude -Llib -lmingw32 -lSDL3 -o maze_solver
.\maze_solver
```