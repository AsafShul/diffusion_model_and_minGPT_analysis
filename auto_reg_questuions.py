import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mingpt.bpe import get_encoder
from matplotlib.collections import PatchCollection
from gpt_model_utils import init_dataset, init_model, init_trainer, MODEL_WEIGHTS_PATH
from attention_utils import set_return_attention

LATENT_SHAPE = (1, 9, 768)
INVERSION_SENTENCE = 'I am a little squirrel holding a walnut'
Q345_SENTENCE = 'The little squirrel was around Alice but she kept saying'


class Questions:
    @staticmethod
    def question2(model_, encoder_decoder_):
        # more trials are in the dev notebook
        def inversion(model_, encoder_decoder, *, target_str, num_steps=500,
                      learning_rate=0.1, device: torch.device = torch.device('cpu')):

            target_sentence = torch.tensor(encoder_decoder.encode(target_str),
                                           dtype=torch.long)[None, ...].to(device)

            # sample a random latent vector
            input_sentence_ = torch.normal(0, 1, LATENT_SHAPE, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([input_sentence_], betas=(0.9, 0.999), lr=learning_rate)

            losses_ = []
            for step in tqdm(range(num_steps)):
                logits, loss = model_(None, targets=target_sentence, latent_vector=input_sentence_)
                losses_.append(float(loss))

                probs = F.softmax(logits, dim=-1)
                _, idx_next = torch.topk(probs, k=1, dim=-1)
                generated_sentence = encoder_decoder.decode(idx_next[0, :, 0].cpu().numpy())

                if step % 25 == 0:
                    print('current sentence for latent:', generated_sentence)

                if generated_sentence == target_str:
                    print('-' * 50)
                    print('Stopped !')
                    print('model output: "', generated_sentence, '"')
                    break

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            return input_sentence_, losses_

        input_sentence, losses = inversion(model_, encoder_decoder_, target_str=INVERSION_SENTENCE, num_steps=1000)
        print('the input for the model:')
        print(input_sentence)

        plt.plot(losses)
        plt.title('inversion loss')
        plt.show()

    @staticmethod
    def question3plus4(model_, encoder_decoder_):
        ## Question 3:
        model_.eval()

        # a 10-word sentence to check the attention matrix on:
        sentence = Q345_SENTENCE

        # run the model and collect the attention matrices
        with torch.no_grad():
            input_sec = torch.tensor([encoder_decoder_.encode(sentence)], dtype=torch.long)
            output_sec, att_mat_lists, _ = model_.generate(input_sec, 1, do_sample=True)

        decoded_sentence = [encoder_decoder_.decode(i) for i in output_sec.cpu().numpy()][0].split()

        # plot the attention matrix and the word importance:
        last_att_matrix = att_mat_lists[0][-1]

        plt.figure(figsize=(4, 4), dpi=200)
        ax_last_att = plt.axes()
        sns.heatmap(last_att_matrix, ax=ax_last_att, annot=True,
                    yticklabels=sentence.split(' '), xticklabels=sentence.split(' '),
                    cbar=False, cmap="Blues", fmt='.1f', annot_kws={"size": 10},
                    linewidths=0.5, linecolor='black', square=True)
        ax_last_att.set_title(f"Attention Matrix - Last transformer block\n",
                              fontweight='bold', fontsize=14)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(4, 4), dpi=200)

        word_importance_last_block = last_att_matrix[-1]
        color_thresh = np.quantile(word_importance_last_block, 0.8)

        ax_last_att_imp = plt.axes()
        colors = ['orange' if v > color_thresh else 'lightblue' for v in word_importance_last_block]
        sns.barplot(x=[f'{i}: {s}' for i, s in enumerate(sentence.split(' '))],
                    y=word_importance_last_block, ax=ax_last_att_imp, palette=colors)
        ax_last_att_imp.set_title(f"Word importance - Last transformer block\n",
                                  fontweight='bold', fontsize=14)
        ax_last_att_imp.set_ylim(0, 0.6)
        plt.xticks(rotation=90)
        plt.show()

        # question 4:
        plt.figure(figsize=(4, 4), dpi=200)

        first_att_matrix = att_mat_lists[0][0]

        ax_first_att = plt.axes()
        sns.heatmap(first_att_matrix, ax=ax_first_att, annot=True,
                    yticklabels=sentence.split(' '), xticklabels=sentence.split(' '),
                    cbar=False, cmap="Blues", fmt='.1f', annot_kws={"size": 10},
                    linewidths=0.5, linecolor='black', square=True)
        ax_first_att.set_title(f"Attention Matrix - First transformer block\n",
                               fontweight='bold', fontsize=14)
        plt.yticks(rotation=0)
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(4, 4), dpi=200)

        word_importance_first_block = first_att_matrix[-1]
        color_thresh_first = np.quantile(word_importance_first_block, 0.8)

        ax_first_att_imp = plt.axes()
        colors_first = ['orange' if v > color_thresh_first else 'lightblue' for v in word_importance_first_block]
        sns.barplot(x=[f'{i}: {s}' for i, s in enumerate(sentence.split(' '))],
                    y=word_importance_first_block, ax=ax_first_att_imp, palette=colors_first)
        ax_first_att_imp.set_title(f"Word importance - First transformer block\n",
                                   fontweight='bold', fontsize=14)
        ax_first_att_imp.set_ylim(0, 0.6)
        plt.xticks(rotation=90)
        plt.show()

        ylabels = sentence.split(' ')[::-1]
        xlabels = np.arange(12)

        N = len(ylabels)
        M = len(xlabels)

        x, y = np.meshgrid(np.arange(M), np.arange(N))

        fig, ax = plt.subplots(figsize=(4, 4), dpi=200)

        R = np.array([m[-1][::-1].round(2) for m in att_mat_lists[0]]).T
        circles = [plt.Circle((j, i), radius=R[i][j]) for i in range(N) for j in range(M)]
        col = PatchCollection(circles, array=R.flatten(), cmap='GnBu')
        ax.add_collection(col)

        ax.set(xticks=np.arange(M), yticks=np.arange(N),
               xticklabels=xlabels, yticklabels=ylabels)
        ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
        ax.grid(which='minor', linestyle='-', linewidth=2)
        ax.set_aspect('equal')
        ax.set_title('Attention of the last word in each \n\ntransformer block visualization',
                     pad=20, fontweight='bold', fontsize=14)
        ax.set_xlabel('\nBlock num')
        ax.set_ylabel('Word', rotation=0, labelpad=20)
        plt.show()

    @staticmethod
    def question5(model_, encoder_decoder_):
        # question 5:
        sentence_ = Q345_SENTENCE

        with torch.no_grad():
            input_sec = torch.tensor([encoder_decoder_.encode(sentence_)], dtype=torch.long)
            output_sec, att_mat_lists, log_probs = model_.generate(input_sec, 15)

        decoded = [encoder_decoder_.decode(i) for i in output_sec.cpu().numpy()][0]
        print(decoded)

        plt.figure(figsize=(10, 1))

        word_prob = np.hstack(log_probs)
        color_thresh_first = np.quantile(word_prob, 0.8)

        ax_prob = plt.axes()
        colors_first = ['orange' if v > color_thresh_first else 'lightblue' for v in word_prob]
        sns.barplot(x=[f'{i}: {s}' for i, s in enumerate(output_sec[0][-15:].numpy())],
                    y=np.exp(word_prob), ax=ax_prob, palette=colors_first)
        ax_prob.set_title(f"probabilities of generated output\n",
                          fontweight='bold', fontsize=14)
        plt.xticks(rotation=45)
        plt.show()

        print(f'sentence probability is: {np.exp(np.hstack(log_probs).sum())}')


if __name__ == '__main__':
    re_train = False  # need to have weights file in the same folder

    q = Questions()

    print('loading model')
    train_dataset = init_dataset()
    model = init_model(train_dataset)
    if not re_train:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
    else:
        trainer = init_trainer(model, train_dataset)
        trainer.run()
        torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    encoder_decoder = get_encoder()
    q.question2(model, encoder_decoder)
    set_return_attention()
    q.question3plus4(model, encoder_decoder)
