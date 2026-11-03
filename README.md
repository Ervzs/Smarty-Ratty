# Smarty Ratty

A mouse learning a maze with Q-learning. Made with NumPy and pygame.

![screenshot](docs/window.png)

## Run

```bash
pip install numpy pygame-ce
python index.py
```

Press Esc to stop.

Rewards: +100 for the cheese, -1 per move, -5 for hitting a wall. Each episode ends at the cheese or after 10 seconds. You can change the settings at the top of `index.py`.

## License

MIT
