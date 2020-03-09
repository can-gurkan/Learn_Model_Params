import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F


class policy_estimator():
    def __init__(self, n_outputs):
        self.n_inputs = 1
        self.n_outputs = n_outputs
        self.n_hidden = 128

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden),
            nn.Sigmoid(),
            nn.Linear(self.n_hidden, self.n_outputs * 2))

    def predict(self, x, y):
        res = self.network(x)
        means, sigs = res[:, :self.n_outputs], torch.exp(res[:, self.n_outputs:])
        return means, sigs


def reinforce(input, policy_estimator, num_episodes=2000, batch_size=10):
    opt = optim.Adam(policy_estimator.network.parameters(), lr=0.0005)

    for i in range(num_episodes):
        x = input
        y = get_abm_outputs()

        x, y = Variable(x), Variable(y)
        means, sigs = policy_estimator.predict(x,y)

        dists = torch.distributions.Normal(means, sigs)
        samples = dists.sample()

        mloss = F.mse_loss(samples, y, reduce=False)
        loss = - dists.log_prob(samples) * - mloss
        loss = loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 100 == 0:
            print('episode ', i)
            print('      ', 'loss', F.mse_loss(samples.data, y.data, reduce=False).mean(dim=0))
            print('      ', 'means', means.mean(dim=0))
            print('      ', 'sigs', sigs.mean(dim=0))
            print('      ', 'samples', samples)


def get_abm_outputs():
    # run the abm with given parameters and return outputs here
    return torch.tensor(np.array([[1,2,3,4]]),dtype=torch.float)


if __name__ == "__main__":

    num_params = 4
    rand_input = torch.randn(1, 1)
    policy_est = policy_estimator(num_params)
    rewards = reinforce(rand_input, policy_est)
