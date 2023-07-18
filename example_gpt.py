import torch
from gpt_model_utils import init_dataset, init_model, init_trainer, MODEL_WEIGHTS_PATH

SENTENCE = 'Alice said'


if __name__ == '__main__':
    re_train = True

    # init run:
    train_dataset = init_dataset()
    gpt = init_model(train_dataset)

    if not re_train:
        gpt.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
    else:
        trainer = init_trainer(gpt, train_dataset)
        trainer.run()
        torch.save(gpt.state_dict(), MODEL_WEIGHTS_PATH)

    # sample from model:
    with torch.no_grad():
        gpt.eval()
        inp = torch.tensor(train_dataset.encoder.encode(SENTENCE), dtype=torch.long).unsqueeze(0)
        sample = gpt.generate(inp, 15, do_sample=True)
        res = [train_dataset.encoder.decode(token) for token in sample.cpu().numpy()]
        print(res)
