# 🐭 AI Smart Mouse: A Q-Learning Maze Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A small, visual reinforcement-learning project. A mouse starts with zero knowledge of a maze and learns, by trial and error with **Q-learning**, the shortest route to the cheese. Training takes about a second, then a Pygame window shows the mouse before and after learning.

```
 S # . . . . .        S = start     C = cheese
 . . . # # # .        # = wall      . = open cell
 . # # . . # .
 . . . # . # #        Learned path (10 steps, the optimal one):
 . # . . . . C        (0,0) ↓ ↓ ↓ ↓ (3,0) → → (3,2) ↓ (4,2) → → → → (4,6)
```

The maze has a trap on purpose: the top-right route looks promising but ends in a dead end at `(2,6)`, so the mouse has to discover that going down and around is the only way.

---

## 📚 Table of Contents
* [Quick Start](#-quick-start)
* [What You'll See](#-what-youll-see)
* [How It Works](#-how-it-works)
* [Understanding the Code](#-understanding-the-code)
* [Tuning the Parameters](#-tuning-the-parameters)
* [License](#-license)

---

## 🚀 Quick Start

```bash
pip install numpy pygame
python index.py
```

In the window you watch every episode live. **↑ / ↓** double/halve the speed, **Esc** quits.

Options:

| Flag       | Effect                                      |
| :--------- | :------------------------------------------ |
| `--seed N` | Make training reproducible.                 |

**Want to skip the animation?** In `main()` of `index.py`, comment out `train(screen, font)` and uncomment `train()` to run all 2000 episodes instantly, then watch the learned path.

> On very new Python versions (e.g. 3.14) `pygame` may have no prebuilt wheel yet. Use `pip install pygame-ce` instead; it is a drop-in replacement.

---

## 👀 What You'll See

The window shows the mouse moving one cell at a time, with a status bar: episode number, step, total reward so far and epsilon. The terminal prints one summary line per episode:

```
Episode 1: timeout in 200 steps, reward -636, epsilon 0.995
Episode 300: cheese in 14 steps, reward 87, epsilon 0.231
Episode 2000: cheese in 10 steps, reward 91, epsilon 0.010
```

If learning works, **steps fall toward 10** (the shortest path), **reward rises toward ~91**, and **epsilon decays** from 1.0 to 0.01. Once all episodes finish, the mouse runs the learned path with no random moves and the terminal prints it.

| Episode 1 (random) | Episode 300 (learning) | After training |
| :----------------: | :--------------------: | :------------: |
| ![Episode 1](docs/episode-1.png) | ![Episode 300](docs/episode-300.png) | ![Learned path](docs/learned-path.png) |

---

## 🧠 How It Works

**Reinforcement Learning (RL)** trains an **agent** to act in an **environment** so as to maximize cumulative **reward**.

| Concept     | In this project                                                                 |
| :---------- | :------------------------------------------------------------------------------ |
| **Agent**   | The mouse (gray square).                                                        |
| **State**   | Which maze cell the mouse is in (35 states for the 7×5 maze).                   |
| **Action**  | Move one cell **UP, DOWN, LEFT or RIGHT**.                                      |
| **Reward**  | See below.                                                                      |
| **Episode** | One attempt from the start cell until the cheese is found or 200 steps pass.    |

### Rewards

| Event                                  | Reward  | Why                                           |
| :------------------------------------- | :-----: | :-------------------------------------------- |
| Reach the cheese                       |  `+100` | The goal. Ends the episode.                   |
| Any normal move                        |   `-1`  | Makes shorter paths score higher.             |
| Bump into a wall or the map edge       |   `-5`  | Discourages pointless moves.                  |

There is deliberately **no hint** about where the cheese or the dead ends are. The mouse works that out from these three rules alone.

### The Q-Table

A table with one **row per state** and one **column per action**. Each entry, the **Q-value**, estimates the total future reward of taking that action in that state. It starts as all zeros, and training fills it in until the best action in each cell points along the shortest route.

### The Learning Loop

1. **Choose an action (epsilon-greedy).** With probability **ε** pick a random action (explore); otherwise pick the action with the highest Q-value (exploit). ε starts at 1.0 and decays each episode, so the mouse explores early and exploits later.
2. **Act.** Move, and receive a reward and the new state.
3. **Update the Q-table** with the Bellman update:

   ```
   Q(s, a) ← Q(s, a) + α · ( reward + γ · max Q(s', ·) − Q(s, a) )
   ```

   Each update pulls the old estimate toward "reward now + best value I expect from the next cell". Repeated thousands of times, the cheese's +100 propagates backwards along the path, one cell per pass. At the terminal (cheese) state there is no future value, so the `γ · max Q` term is dropped.
4. **Repeat** for 2000 episodes.

---

## 💻 Understanding the Code

Everything is in `index.py`, top to bottom:

| Piece | What it does |
| :---- | :----------- |
| `maze`, `unitSize` | The map (NumPy array) and the pixel size of one cell for drawing. |
| Hyperparameters & rewards | `learning_rate`, `discount_factor`, `epsilon_*`, `num_episodes`, `max_steps`, `REWARD_*`. |
| `step(state, action)` | The environment: applies a move and returns `(new_state, reward, done)`. |
| `choose_action()` / `greedy_action()` | Epsilon-greedy policy. Ties are broken randomly, so an untrained row isn't biased toward "UP". |
| `update_q_table(...)` | The Bellman update. |
| `train(screen, font)` | Runs all episodes. With a screen it draws every step; without one it runs instantly. |
| `run_greedy()` | Follows the learned policy (no randomness) and returns the path. |
| `draw()` | Draws the maze, mouse and status bar for one frame; handles the speed keys. |

---

## 🔧 Tuning the Parameters

All at the top of `index.py`. Try changing them and watching the episode log:

| Parameter | Default | Effect |
| :-------- | :-----: | :----- |
| `learning_rate` (α) | `0.1` | Higher learns faster but is noisier; lower is steadier but slower. |
| `discount_factor` (γ) | `0.95` | Closer to 1 makes the mouse value the distant cheese more. Too low and the +100 fades before it reaches the start. |
| `epsilon_decay_rate` | `0.005` | Higher stops exploring sooner. Too high and it may lock in a bad path; too low wastes episodes. |
| `min_epsilon` | `0.01` | Exploration floor. |
| `num_episodes` | `2000` | Plenty for this maze; it converges in roughly 500. |
| `max_steps` | `200` | Cut-off that ends an episode in which the mouse wanders too long. |
| `REWARD_*` | `100 / -1 / -5` | Reshaping rewards changes behavior. Try `REWARD_STEP = 0` and watch the paths get lazier. |

You can also edit the `maze` array to design your own, as long as `S` and `C` are connected.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
