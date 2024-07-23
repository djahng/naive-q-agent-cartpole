import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims

        self.fc1 = nn.Linear(*self.input_dims, 128)
        self.fc2 = nn.Linear(128, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = (
            "cuda" if T.cuda.is_available()
            else "mps" if T.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)
        # print(f"Using device `{self.device}`")

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class Agent:
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                 epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [n for n in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        if self.epsilon - self.eps_dec > self.eps_min:
            self.epsilon -= self.eps_dec
        else:
            self.epsilon = self.eps_min

    def learn(self, state, action, reward, state_):
        # Zero the gradients
        self.Q.optimizer.zero_grad()

        # Convert data to tensors
        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]
        q_next = self.Q.forward(states_).max()
        q_target = rewards + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)      # MSE
        loss.backward()         # Back propagate
        self.Q.optimizer.step()
        self.decrement_epsilon()


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    n_games = 10000
    scores = []
    eps_history = []

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                  n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        terminated = False
        observation, info = env.reset()

        while not terminated:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)

            agent.learn(observation, action, reward, observation_)

            observation = observation_
            score += reward

        scores.append(score)
        eps_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode: {i}\tScore: {score:.1f}\tAvg Score: {avg_score:.1f}\tEpsilon: {agent.epsilon:.2f}")

    filename = "cartpole_naive_dqn.png"
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
