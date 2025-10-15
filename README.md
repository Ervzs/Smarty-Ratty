# 🐭 AI Smart Mouse: A Q-Learning Maze Solver

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple and visual AI project where a "smart mouse" learns to solve a maze from scratch using **Reinforcement Learning**, specifically the **Q-Learning** algorithm.

The mouse starts with zero knowledge of the maze and, through trial and error, learns the optimal path from the start ('S') to the cheese ('C').

---

## 🚀 Demo

Watch the mouse learn! In the early episodes, its movement is completely random. After thousands of training episodes, it navigates the maze with purpose and efficiency.

**(WIP)**

| Before Training (Random Exploration) | After Training (Optimal Path) |
| :----------------------------------: | :-----------------------------: |
|  |  |

---

## 📚 Table of Contents
* [How It Works](#-how-it-works)
* [Tech Stack](#-tech-stack)
* [Understanding the Code](#-understanding-the-code)
* [Key Q-Learning Parameters](#-key-q-learning-parameters)
* [License](#-license)

---

## 🧠 How It Works

This project is a practical application of **Reinforcement Learning (RL)**. The goal of RL is to train an **agent** to make optimal decisions in an **environment** to maximize a cumulative **reward**.

### Core Concepts

* **Agent 🤖:** The learner. In this project, the **agent is the gray square (the mouse)**.
* **Environment 🗺️:** The world the agent interacts with. Here, it's the **2D maze**.
* **State 📍:** The agent's current position in the environment. A state is a specific grid cell in our maze.
* **Action 🕹️:** A possible move the agent can make. Our mouse can move **UP, DOWN, LEFT, or RIGHT**.
* **Reward 🏆:** Feedback from the environment after an action.
    * **Positive Reward (+100):** Finding the cheese.
    * **Negative Reward / Penalty (-10):** Hitting a wall.
    * **Small Negative Reward (-0.1):** Taking a step (encourages finding the shortest path).

### The Q-Learning Algorithm

The agent learns using **Q-Learning**. This algorithm uses a **Q-Table** to figure out the best action to take in any given state.

#### The Q-Table: The AI's Brain

The Q-Table is a simple lookup table. The **rows represent every state** (each cell in the maze), and the **columns represent every action** (UP, DOWN, etc.). The value in each cell, the **Q-value**, is a prediction of the total future reward the mouse can expect if it takes that action from that state.

Initially, this table is all zeros. The goal of training is to fill it with useful values that represent a winning strategy.



#### The Learning Process

1.  **Initialization:** The mouse is placed at the start ('S'). The Q-Table is all zeros.
2.  **Exploration vs. Exploitation:** The mouse needs to choose an action. It uses the **Epsilon-Greedy** policy:
    * With a probability of **epsilon ($\epsilon$)**, it chooses a **random action** (explores).
    * Otherwise, it chooses the **best-known action** from the Q-Table (exploits).
    We start with a high `epsilon` (lots of exploration) and slowly decrease it as the mouse learns.
3.  **Action & Reward:** The mouse performs the action and the environment gives back a reward and the new state.
4.  **Update Q-Table:** This is the magic step. We update the Q-value for the state-action pair we just experienced using the **Bellman equation**. In simple terms, this update rule adjusts our old Q-value based on the reward we just got and the best possible reward we could get from our new location. This slowly propagates information about good paths through the table.

This cycle repeats for thousands of "episodes" (an episode is one attempt from start to finish). Over time, the Q-Table becomes an accurate "cheat sheet" for solving the maze.

---

## 🛠️ Tech Stack

* **Python:** The core programming language.
* **Pygame:** A library used for creating the visual simulation (drawing the maze, mouse, etc.).
* **NumPy:** Used for creating and managing the Q-Table efficiently.

---

## 💻 Understanding the Code

The main logic is contained within `index.py`. Here is a breakdown of the key components:

* **Global Variables & Maze Setup:**
    * `maze`: A NumPy array that defines the layout of the environment.
    * `unitSize`: The size of each grid cell in pixels.
    * Q-Learning Hyperparameters (`learning_rate`, `discount_factor`, `epsilon`, etc.).

* **Helper Functions:**
    * `get_state_from_pos()`: Converts the mouse's pixel coordinates into a discrete state (a number from 0 to 34). This is essential for using the Q-Table.
    * `drawMap()` / `drawGrid()`: Pygame functions to render the maze on the screen.

* **Q-Learning Functions:**
    * `choose_action(state, epsilon)`: Implements the Epsilon-Greedy policy to select the next action.
    * `update_q_table(...)`: Executes the Bellman equation to update the Q-value after each step.

* **Main Game Loop (`main` function):**
    * This is the heart of the simulation. It loops through thousands of episodes.
    * Inside the loop, it handles the agent's movement, calculates rewards, calls the `update_q_table` function, and updates the Pygame display.
    * It also manages the episode logic, such as resetting the mouse's position and decaying `epsilon`.

---

## 🔧 Key Q-Learning Parameters

You can change these values at the top of `index.py` to see how they affect the AI's learning speed and performance.

* `learning_rate` (alpha): Controls how much the AI learns from new information. Higher values mean faster learning but can lead to instability.
* `discount_factor` (gamma): Determines how much the AI cares about future rewards. A value close to 1 makes the AI more "patient" and focused on the final goal.
* `epsilon`: The initial exploration rate.
* `epsilon_decay_rate`: Controls how quickly the AI stops exploring and starts exploiting its knowledge.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
