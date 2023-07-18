import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

SAMPLES_LOWER_BOUND = -1
SAMPLES_UPPER_BOUND = 1
SAMPLES_DIM = 2
EMBEDDING_DIM = 5
CLASS_NUM = 9

LEARNING_RATE = 1e-3
BATCH_SIZE = -1  # essentially performs GD not SGD
EPOCHS_NUM = 3000
T = 1000

BIN_LIMITS = [-np.inf, -0.33, 0.33, np.inf]
COLORS = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']


def scheduler(t):
    return np.exp(5 * (t - 1))


def scheduler_der(t):
    return 5 * scheduler(t)


def mod_scheduler_q5(t):
    return t ** 2 + 0.0001


def mod_scheduler_der_q5(t):
    return 2 * t


class DiffusionModel(nn.Module):
    def __init__(self, input_dim=SAMPLES_DIM + 1,
                 hidden_dim=16, output_dim=SAMPLES_DIM, num_hidden_layers=2, classes_num=-1):
        super().__init__()
        self.classes_num = classes_num
        self.conditional = classes_num != -1
        self.name = ("C" if self.conditional else "Unc") + "onditional_Diffusion_Model.pkl"

        self.embedding = None
        if self.conditional:
            self.embedding = nn.Embedding(classes_num, EMBEDDING_DIM)
            input_dim += EMBEDDING_DIM

        # denoiser:
        layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU()]
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.denoiser = nn.Sequential(*layers)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def forward(self, x, t, c):
        c_ = self.embedding(c).reshape(-1, EMBEDDING_DIM) if self.conditional else c
        inp = [x, t, c_] if self.conditional else [x, t]
        inp = torch.cat(inp, axis=1)
        return self.denoiser(inp)

    @staticmethod
    def get_data(samples_num, samples_dim=SAMPLES_DIM, conditional=False, as_numpy=False):
        samples = torch.rand(samples_num, samples_dim) * 2 - 1
        classes = DiffusionModel.bin(samples) if conditional else None

        if as_numpy:
            samples = samples.numpy()
            classes = classes.numpy() if conditional else None

        return samples, classes

    @staticmethod
    def forward_noise_step(x, t):
        step_noise = torch.tensor(np.random.normal(loc=0, scale=1, size=x.shape),
                                  dtype=torch.float32)
        xt = torch.tensor(x, dtype=torch.float32) + scheduler(t) * step_noise
        return xt, step_noise

    def fit(self, samples_num, epochs_num=EPOCHS_NUM, batch_size=BATCH_SIZE, save_model=True, plot_loss=True):
        samples_, classes_ = self.get_data(samples_num=samples_num, conditional=self.conditional)
        classes_ = classes_ if self.conditional else torch.zeros((samples_.shape[0], 1))

        batch_size = batch_size if batch_size != -1 else samples_num
        dataset = TensorDataset(samples_, classes_)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = []
        for epoch in (pbar := tqdm(range(epochs_num))):
            running_loss = []
            for i, (x, y) in enumerate(data_loader):
                self.optimizer.zero_grad()
                t = torch.rand((x.shape[0], 1), dtype=torch.float32)
                xt, step_noise = self.forward_noise_step(x, t)
                loss = self.criterion(self.forward(xt, t, y), step_noise)

                loss.backward()
                running_loss.append(loss.item())

                self.optimizer.step()

            epoch_loss = np.mean(running_loss)
            pbar.set_description(f'Epoch {epoch + 1}/{epochs_num}, loss {epoch_loss:.6f}')
            loss_history.append(epoch_loss)

        if save_model:
            self.save()

        if plot_loss:
            plt.plot(np.arange(len(loss_history)), loss_history, color='blue', zorder=2)
            plt.title(f'Loss over training batches - {self.name}')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.show()

    def save(self, path=None):
        path = path if path is not None else self.name
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def plot_rectangle(show=False):
        plt.plot([-1, -1], [-1, 1], color='black')
        plt.plot([-1, 1], [1, 1], color='black')
        plt.plot([1, 1], [1, -1], color='black')
        plt.plot([1, -1], [-1, -1], color='black')
        if show:
            plt.show()

    @torch.no_grad()
    def sample(self, steps_=T, seed_=None, class_=None, non_deterministic=False, return_trajectory=False):
        if class_ is None and self.conditional:
            class_ = np.random.randint(0, CLASS_NUM)
        c = torch.tensor(class_, dtype=torch.int).reshape(-1, 1) if class_ is not None else None

        trajectory = []
        z = seed_ if seed_ is not None else np.random.normal(0, 1, 2)
        trajectory.append(z.reshape(-1, ))
        z = torch.tensor(z.reshape(1, -1), dtype=torch.float32)
        for t in np.linspace(1, 0, steps_):
            t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)
            sigma = scheduler(t)
            z_prime = z - (sigma * self.forward(z, t, c))
            score = (z_prime - z) / (sigma ** 2)
            dz = scheduler_der(t) * sigma * score * (1 / steps_)
            dz = dz + torch.tensor(np.random.normal(0, 0.005, size=z.shape),
                                   dtype=torch.float32) if non_deterministic else dz
            z = z + dz
            trajectory.append(z.detach().numpy().reshape(-1, ))

        z = trajectory[-1]
        res = z if not self.conditional else (z, class_)
        return res if not return_trajectory else (res, trajectory)

    @staticmethod
    def bin(samples):
        # Define the bin limits
        x_bins = torch.bucketize(samples[:, 0], torch.tensor(BIN_LIMITS)) - 1
        y_bins = torch.bucketize(samples[:, 1], torch.tensor(BIN_LIMITS)) - 1
        return (x_bins * 3 + y_bins).int()

    @staticmethod
    def SNR(t):
        return 1 / (scheduler(t) ** 2)
