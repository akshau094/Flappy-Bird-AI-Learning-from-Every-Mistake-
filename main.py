import argparse
import itertools
import os
import random
import json
from collections import deque
from datetime import datetime

import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from flappy_bird_gymnasium.envs.constants import (
    PIPE_HEIGHT,
    PIPE_WIDTH,
    PLAYER_HEIGHT,
    PLAYER_WIDTH,
)


PARAMS = {
    "flappybirdv0": {
        "env_id": "FlappyBird-v0",
        "epsilon_init": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "replay_memory_size": 100000,
        "mini_batch_size": 32,
        "network_sync_rate": 100,
        "alpha": 0.001,
        "gamma": 0.99,
        "reward_threshold": 1000,
    }
}

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
FAILURE_LOG_FILE = os.path.join(RUNS_DIR, "failure_log.json")


class FailureLogger:
    def __init__(self, log_file=FAILURE_LOG_FILE):
        self.log_file = log_file
        self.failures = []
        self.analysis = {}
        self._load_existing_log()

    def _load_existing_log(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    self.failures = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.failures = []
        self._analyze_failures()

    def _analyze_failures(self):
        self.analysis = {
            "total_failures": len(self.failures),
            "failed_pipe_heights": [],
            "failed_gap_sizes": [],
            "common_failure_heights": {},
            "common_failure_gaps": {}
        }
        
        for failure in self.failures:
            self.analysis["failed_pipe_heights"].append(failure["gap_height"])
            self.analysis["failed_gap_sizes"].append(failure["gap_size"])
            
            # Count frequency of gap heights
            height_key = round(failure["gap_height"], 0)
            if height_key not in self.analysis["common_failure_heights"]:
                self.analysis["common_failure_heights"][height_key] = 0
            self.analysis["common_failure_heights"][height_key] += 1
            
            # Count frequency of gap sizes
            gap_key = round(failure["gap_size"], 0)
            if gap_key not in self.analysis["common_failure_gaps"]:
                self.analysis["common_failure_gaps"][gap_key] = 0
            self.analysis["common_failure_gaps"][gap_key] += 1

    def log_failure(self, pipe_idx, gap_top, gap_bottom, bird_y, bird_vel, pipe_x, score):
        gap_size = gap_bottom - gap_top
        gap_height = (gap_top + gap_bottom) / 2
        
        failure_data = {
            "timestamp": datetime.now().isoformat(),
            "pipe_index": pipe_idx,
            "gap_top": gap_top,
            "gap_bottom": gap_bottom,
            "gap_size": gap_size,
            "gap_height": gap_height,
            "bird_y": bird_y,
            "bird_velocity": bird_vel,
            "pipe_x": pipe_x,
            "score_at_failure": score
        }
        self.failures.append(failure_data)
        self._save_log()
        self._analyze_failures()
        return failure_data

    def _save_log(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.failures, f, indent=4)

    def get_safe_zones(self):
        if not self.analysis["failed_pipe_heights"]:
            return None
        
        avg_gap_size = sum(self.analysis["failed_gap_sizes"]) / len(self.analysis["failed_gap_sizes"])
        avg_gap_height = sum(self.analysis["failed_pipe_heights"]) / len(self.analysis["failed_pipe_heights"])
        
        return {
            "average_gap_size": avg_gap_size,
            "average_gap_height": avg_gap_height,
            "recommended_safe_height": avg_gap_height
        }

    def find_similar_failures(self, gap_top, gap_bottom, tolerance=20):
        """Find past failures that are similar to the current pipe"""
        similar = []
        current_gap_height = (gap_top + gap_bottom) / 2
        current_gap_size = gap_bottom - gap_top
        
        for failure in self.failures:
            height_diff = abs(failure["gap_height"] - current_gap_height)
            size_diff = abs(failure["gap_size"] - current_gap_size)
            
            if height_diff < tolerance and size_diff < tolerance:
                similar.append(failure)
        return similar


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class DQN(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.model(x)


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, new_exp):
        self.memory.append(new_exp)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class SafePilot:
    def __init__(self, failure_logger=None):
        self.failure_logger = failure_logger
        self.distance_gain = 0.04
        self.velocity_gain = -1.08
        self.target_offset = 25.1
        self.gap_padding = 21.6
        self.next_pipe_blend_gain = 0.13
        self.next_pipe_blend_window = 140.0
        self.floor_emergency_margin = 22.3
        self.center_alignment_offset = PLAYER_HEIGHT / 2.0
        self.center_deadband = 1.5
        self.target_smoothing = 0.13
        self.flap_cooldown_frames = 2
        self.cooldown = 0
        self.previous_target_y = None
        self.safe_zones = None
        self.prediction_history = []  # Track prediction accuracy
        if self.failure_logger:
            self.safe_zones = self.failure_logger.get_safe_zones()
            if self.safe_zones:
                print(f"\nLoaded safe zones from past failures: {self.safe_zones}")
                print(f"   Total failures analyzed: {self.failure_logger.analysis['total_failures']}\n")

    def reset(self):
        self.cooldown = 0
        self.previous_target_y = None
        self.prediction_history = []

    def _get_pipe_sequence(self, env):
        game = env.unwrapped
        player_x = game._player_x
        pairs = list(zip(game._upper_pipes, game._lower_pipes))
        for idx, (up_pipe, low_pipe) in enumerate(pairs):
            if low_pipe["x"] + PIPE_WIDTH >= player_x:
                return idx, pairs
        return len(pairs) - 1, pairs

    def predict(self, env):
        game = env.unwrapped
        pipe_idx, pairs = self._get_pipe_sequence(env)
        up_pipe, low_pipe = pairs[pipe_idx]

        player_x = float(game._player_x)
        player_y = float(game._player_y)
        player_vel = float(game._player_vel_y)
        player_center_y = player_y + PLAYER_HEIGHT / 2.0
        player_bottom_y = player_y + PLAYER_HEIGHT
        gap_top = float(up_pipe["y"] + PIPE_HEIGHT)
        gap_bottom = float(low_pipe["y"])
        gap_center = (gap_top + gap_bottom) / 2.0
        distance = float(low_pipe["x"] - player_x)

        # Check for similar past failures and adjust!
        adjusted_gap_center = gap_center
        adjusted_gap_padding = self.gap_padding
        
        if self.failure_logger:
            similar_failures = self.failure_logger.find_similar_failures(gap_top, gap_bottom)
            if similar_failures:
                print(f"Found {len(similar_failures)} similar past failures! Adjusting...")
                # Learn from where the bird failed before!
                avg_failed_y = sum(f["bird_y"] for f in similar_failures) / len(similar_failures)
                avg_failed_vel = sum(f["bird_velocity"] for f in similar_failures) / len(similar_failures)
                
                # Adjust target to avoid the failed area!
                if avg_failed_y < gap_center:
                    adjusted_gap_center += 20  # Aim lower if failed high
                else:
                    adjusted_gap_center -= 20  # Aim higher if failed low
                # Increase safety padding!
                adjusted_gap_padding += 10
        
        if pipe_idx + 1 < len(pairs):
            next_up_pipe, next_low_pipe = pairs[pipe_idx + 1]
            next_gap_center = float(
                (next_up_pipe["y"] + PIPE_HEIGHT + next_low_pipe["y"]) / 2.0
            )
        else:
            next_gap_center = adjusted_gap_center

        blend = max(
            0.0,
            min(
                0.6,
                self.next_pipe_blend_gain
                * max(0.0, self.next_pipe_blend_window - distance)
                / self.next_pipe_blend_window,
            ),
        )
        target_center = (1.0 - blend) * adjusted_gap_center + blend * next_gap_center

        target_y = target_center + self.distance_gain * distance
        target_y += self.velocity_gain * player_vel + self.target_offset
        target_y = min(max(target_y, gap_top + adjusted_gap_padding), gap_bottom - adjusted_gap_padding)
        target_center_y = target_y + self.center_alignment_offset

        if self.previous_target_y is None:
            smoothed_target_y = target_center_y
        else:
            smoothed_target_y = (
                (1.0 - self.target_smoothing) * target_center_y
                + self.target_smoothing * self.previous_target_y
            )
        self.previous_target_y = smoothed_target_y

        projected_center_y = player_center_y
        action = 1 if player_center_y > smoothed_target_y + self.center_deadband else 0

        if player_bottom_y > gap_bottom - self.floor_emergency_margin:
            action = 1
        elif self.cooldown > 0:
            action = 0

        if action == 1:
            self.cooldown = self.flap_cooldown_frames
        elif self.cooldown > 0:
            self.cooldown -= 1
        
        # Track this prediction for accuracy!
        self.prediction_history.append({
            "gap_top": gap_top,
            "gap_bottom": gap_bottom,
            "target_y": smoothed_target_y,
            "player_y": player_center_y,
            "distance": distance
        })
        
        prediction = {
            "player_x": player_x + PLAYER_WIDTH / 2,
            "player_y": player_center_y,
            "target_x": float(low_pipe["x"] + PIPE_WIDTH / 2),
            "target_y": smoothed_target_y,
            "gap_top": gap_top,
            "gap_bottom": gap_bottom,
            "projected_y": projected_center_y,
        }
        return action, prediction

    def draw_prediction(self, env, prediction):
        game = env.unwrapped
        if not hasattr(game, "_surface") or game._display is None:
            return

        start_pos = (int(prediction["player_x"]), int(prediction["player_y"]))
        target_pos = (int(prediction["target_x"]), int(prediction["target_y"]))
        gap_top_pos = (int(prediction["target_x"]), int(prediction["gap_top"]))
        gap_bottom_pos = (int(prediction["target_x"]), int(prediction["gap_bottom"]))
        projected_pos = (int(prediction["player_x"]), int(prediction["projected_y"]))

        pygame.draw.line(game._surface, (255, 0, 0), start_pos, target_pos, 3)
        pygame.draw.line(game._surface, (255, 80, 80), gap_top_pos, gap_bottom_pos, 2)
        pygame.draw.circle(game._surface, (255, 0, 0), target_pos, 6)
        pygame.draw.circle(game._surface, (255, 160, 160), projected_pos, 4)
        game._update_display()


class Agent:
    def __init__(self, param_set, failure_logger=None):
        if param_set not in PARAMS:
            raise ValueError(f"Unknown parameter set: {param_set}")

        self.failure_logger = failure_logger
        
        params = PARAMS[param_set]
        self.param_set = param_set
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.log_file = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.model_file = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    def _make_env(self, render=False, score_limit=None):
        return gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None,
            use_lidar=False,
            score_limit=score_limit,
        )

    def run_train(self, render=False, score_limit=None):
        if self.failure_logger:
            print(f"\nTraining with failure analysis - {self.failure_logger.analysis['total_failures']} past failures loaded\n")
        
        env = self._make_env(render=render, score_limit=score_limit)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(DEVICE)
        target_dqn = DQN(num_states, num_actions).to(DEVICE)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        memory = ReplayMemory(self.replay_memory_size)
        epsilon = self.epsilon_init
        steps = 0
        best_reward = float("-inf")
        self.optimizer = optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        try:
            for episode in itertools.count():
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                episode_reward = 0.0
                terminated = False
                truncated = False
                last_pairs = None
                last_pipe_idx = 0

                while not (terminated or truncated) and episode_reward < self.reward_threshold:
                    game = env.unwrapped
                    pairs = list(zip(game._upper_pipes, game._lower_pipes))
                    last_pairs = pairs
                    
                    # Get current pipe index
                    player_x = game._player_x
                    for idx, (_, low_pipe) in enumerate(pairs):
                        if low_pipe["x"] + PIPE_WIDTH >= player_x:
                            last_pipe_idx = idx
                            break
                    
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                        action_tensor = torch.tensor(action, dtype=torch.long, device=DEVICE)
                    else:
                        with torch.no_grad():
                            action_tensor = policy_dqn(state.unsqueeze(0)).squeeze().argmax()
                        action = int(action_tensor.item())

                    next_state, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward

                    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
                    next_state_tensor = torch.tensor(
                        next_state, dtype=torch.float32, device=DEVICE
                    )
                    memory.append(
                        (state, action_tensor, next_state_tensor, reward_tensor, terminated)
                    )
                    steps += 1
                    state = next_state_tensor

                    if len(memory) > self.mini_batch_size:
                        self.optimize(memory.sample(self.mini_batch_size), policy_dqn, target_dqn)
                        if steps > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            steps = 0

                # Log failure if we terminated and have a FailureLogger
                if terminated and self.failure_logger and last_pairs:
                    pipe_idx = last_pipe_idx
                    up_pipe, low_pipe = last_pairs[pipe_idx] if pipe_idx < len(last_pairs) else last_pairs[-1]
                    gap_top = float(up_pipe["y"] + PIPE_HEIGHT)
                    gap_bottom = float(low_pipe["y"])
                    pipe_x = float(low_pipe["x"])
                    
                    game = env.unwrapped
                    bird_y = float(game._player_y)
                    bird_vel = float(game._player_vel_y)
                    
                    failure = self.failure_logger.log_failure(
                        pipe_idx, gap_top, gap_bottom, bird_y, bird_vel, pipe_x, info["score"]
                    )
                    print(f"Failure logged (episode {episode+1}): score={info['score']}")

                print(
                    f"episode={episode + 1} reward={episode_reward:.2f} score={info['score']}"
                )

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    with open(self.log_file, "a", encoding="utf-8") as log_handle:
                        log_handle.write(
                            f"best reward = {episode_reward:.2f} for episode={episode + 1}\n"
                        )
                    torch.save(policy_dqn.state_dict(), self.model_file)
        finally:
            env.close()

    def run_play(self, render=True, score_limit=None):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(
                f"Model file not found: {self.model_file}. Train first or use --mode safe."
            )

        env = self._make_env(render=render, score_limit=score_limit)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to(DEVICE)
        policy_dqn.load_state_dict(torch.load(self.model_file, map_location=DEVICE))
        policy_dqn.eval()

        try:
            state, _ = env.reset()
            terminated = False
            truncated = False

            while not (terminated or truncated):
                state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    action = int(policy_dqn(state_tensor.unsqueeze(0)).squeeze().argmax().item())
                state, _, terminated, truncated, info = env.step(action)

            print(f"finished score={info['score']}")
        finally:
            env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, next_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            target_q = rewards + (1 - terminations) * self.gamma * target_dqn(next_states).max(1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_safe_mode(render=True, episodes=1, score_limit=None):
    failure_logger = FailureLogger()
    env = gym.make(
        "FlappyBird-v0",
        render_mode="human" if render else None,
        use_lidar=False,
        score_limit=score_limit,
    )
    pilot = SafePilot(failure_logger)

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            pilot.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            last_prediction = None
            last_pipe_idx = 0
            last_pairs = None

            while not (terminated or truncated):
                game = env.unwrapped
                pairs = list(zip(game._upper_pipes, game._lower_pipes))
                last_pairs = pairs
                
                action, prediction = pilot.predict(env)
                last_prediction = prediction
                
                # Get current pipe index
                player_x = game._player_x
                for idx, (_, low_pipe) in enumerate(pairs):
                    if low_pipe["x"] + PIPE_WIDTH >= player_x:
                        last_pipe_idx = idx
                        break
                
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if render:
                    pilot.draw_prediction(env, prediction)

            # Calculate prediction accuracy for this episode!
            if pilot.prediction_history:
                accurate_predictions = 0
                for p in pilot.prediction_history:
                    # Check if target y was within the safe gap!
                    if p["gap_top"] < p["target_y"] < p["gap_bottom"]:
                        accurate_predictions += 1
                accuracy = (accurate_predictions / len(pilot.prediction_history)) * 100
                print(f"Prediction Accuracy: {accuracy:.1f}% ({accurate_predictions}/{len(pilot.prediction_history)})")

            # Log failure if we terminated
            if terminated and last_prediction and last_pairs:
                pipe_idx = last_pipe_idx
                up_pipe, low_pipe = last_pairs[pipe_idx] if pipe_idx < len(last_pairs) else last_pairs[-1]
                gap_top = float(up_pipe["y"] + PIPE_HEIGHT)
                gap_bottom = float(low_pipe["y"])
                pipe_x = float(low_pipe["x"])
                
                game = env.unwrapped
                bird_y = float(game._player_y)
                bird_vel = float(game._player_vel_y)
                
                failure = failure_logger.log_failure(
                    pipe_idx, gap_top, gap_bottom, bird_y, bird_vel, pipe_x, info["score"]
                )
                print(f"\nFailure logged! Details saved to {FAILURE_LOG_FILE}")
                print(f"   Failed at pipe {failure['pipe_index']}, score {failure['score_at_failure']}")
                print(f"   Gap size: {failure['gap_size']:.1f}, Gap height: {failure['gap_height']:.1f}\n")

            print(
                f"safe_episode={episode + 1} reward={total_reward:.2f} score={info['score']}"
            )
    finally:
        env.close()


def run_manual_mode():
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    pygame.init()

    try:
        _, _ = env.reset()
        done = False

        while not done:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    action = 1

            _, _, terminated, truncated, _ = env.step(action)
            env.render()
            done = done or terminated or truncated
    finally:
        env.close()
        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-file Flappy Bird runner with safe-pilot, training, play, and manual modes."
    )
    parser.add_argument(
        "--mode",
        choices=["safe", "train", "play", "manual"],
        default="safe",
        help="Choose how to run the merged script.",
    )
    parser.add_argument(
        "--param-set",
        default="flappybirdv0",
        help="Parameter set used by DQN train/play modes.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the game window.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes for safe mode.",
    )
    parser.add_argument(
        "--score-limit",
        type=int,
        default=None,
        help="Optional score limit for a quicker stop.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    failure_logger = FailureLogger()

    if args.mode == "safe":
        run_safe_mode(
            render=not args.headless,
            episodes=args.episodes,
            score_limit=args.score_limit,
        )
        return

    if args.mode == "manual":
        run_manual_mode()
        return

    agent = Agent(args.param_set, failure_logger)
    if args.mode == "train":
        agent.run_train(render=not args.headless, score_limit=args.score_limit)
    else:
        agent.run_play(render=not args.headless, score_limit=args.score_limit)


if __name__ == "__main__":
    main()
