import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import abm_model_pynetlogo as abm_model


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

    def predict(self, x):
        res = self.network(x)
        means, sigs = res[:, :self.n_outputs], torch.exp(res[:, self.n_outputs:])
        return means, sigs


def reinforce(input, policy_estimator, model, num_episodes=10000, batch_size=10):
    opt = optim.Adam(policy_estimator.network.parameters(), lr=0.0005)

    for i in range(num_episodes):
        x = input
        y = get_abm_actual_outputs()

        x, y = Variable(x), Variable(y)
        means, sigs = policy_estimator.predict(x)

        dists = torch.distributions.Normal(means, sigs)
        samples = dists.sample()
        # Get outputs from model here
        output = get_abm_exp_outputs(samples.numpy()[0],batch_size, model)

        mloss = F.mse_loss(output, y)#, reduce=False)
        loss = - dists.log_prob(output) * - mloss
        loss = loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if i % 10 == 0:
            print('episode ', i)
            print('      ', 'loss', F.mse_loss(samples.data, y.data).mean(dim=0))
            print('      ', 'means', means.mean(dim=0))
            print('      ', 'sigs', sigs.mean(dim=0))
            print('      ', 'samples', samples)


def get_abm_actual_outputs():
    return torch.tensor(np.array([[507.17, 131.94, 0.0, 6.8999999999999995, 0., 0.]]),dtype=torch.float)


def get_abm_exp_outputs(params,batch,model):
    return torch.tensor(np.array([abm_model.run_model(model,params,batch)]),dtype=torch.float)


if __name__ == "__main__":
    netlogo = abm_model.init_model()
    num_params = 6
    rand_input = torch.randn(1, 1)
    policy_est = policy_estimator(num_params)
    reinforce(rand_input, policy_est, netlogo)
    netlogo.kill_workspace()
