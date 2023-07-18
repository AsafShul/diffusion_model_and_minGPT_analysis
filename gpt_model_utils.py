# imports:
import torch
from torch.utils.data import Dataset

from mingpt.bpe import get_encoder
from mingpt.trainer import Trainer
from mingpt.model import GPT

# constants:
VOCAB_SIZE = 50257
DATA_FILE_NAME = 'alice_in_wonderland.txt'
MODEL_WEIGHTS_PATH = 'alice_model.pth'


# classes:
class AlicDataset(Dataset):
    def __init__(self, block_size, vocab_size=VOCAB_SIZE, path=DATA_FILE_NAME):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.encoder = get_encoder()

        with open(path) as f:
            text = ' '.join((f.read().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')).split())

        self.tokens = torch.tensor(self.encoder.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.tokens) - (self.block_size + 1)

    def __getitem__(self, idx):
        slice_start = idx
        slice_end = idx + self.block_size

        input_seq = self.tokens[slice_start: slice_end]
        output_seq = self.tokens[slice_start + 1: slice_end + 1]

        return input_seq, output_seq


def init_dataset(block_size=64):
    return AlicDataset(block_size=block_size)


# functions:
def init_model(train_dataset):
    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt2'
    model_config.vocab_size = train_dataset.vocab_size
    model_config.block_size = train_dataset.block_size
    model = GPT(model_config)

    return model


def init_trainer(model, train_dataset, learning_rate=5e-4,
                 batch_size=32, max_iters=2000, num_workers=0):
    def batch_end_callback(trainer_):
        if trainer_.iter_num % 10 == 0:
            print(f"iter_dt {trainer_.iter_dt * 1000:.2f}ms; "
                  f"iter {trainer_.iter_num}: train loss {trainer_.loss.item():.5f}")

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = learning_rate  # the model we're using is so small that we can go a bit faster
    train_config.batch_size = batch_size
    train_config.max_iters = max_iters
    train_config.num_workers = num_workers

    trainer = Trainer(train_config, model, train_dataset)
    trainer.set_callback('on_batch_end', batch_end_callback)

    return trainer
