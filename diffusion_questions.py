from itertools import product
import matplotlib.pyplot as plt
from diffusion_model import *
from matplotlib.patches import Rectangle


class Part1:
    @staticmethod
    def q1(seed, steps_num=1000):
        # plot the trajectory of the generated point
        # part 1: Unconditional model
        DiffusionModel.plot_rectangle()

        plt.plot(seed[0], seed[1], 'ro', c=[1, 0, 0], markersize=0.5)

        point = seed.copy()
        for t in np.linspace(0, 1, steps_num):
            point, _ = DiffusionModel.forward_noise_step(point, t)
            color = [1 - t, t, 0]
            plt.plot(point[0], point[1], 'ro', c=color, markersize=0.5)
        plt.title(f'forward process trajectory for point {seed}')
        plt.xlim(-2, 2)
        plt.ylim(-1.5, 1.5)
        plt.show()

    @staticmethod
    def q2():
        # train model (and plot loss - question 2)
        model = DiffusionModel()
        model.fit(samples_num=1000, epochs_num=1000, save_model=True, plot_loss=True)
        return model

    @staticmethod
    def q3(model, samples_num=1000):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("9 Samples of 1000 points \nfor different random seeds", fontsize=18)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, j in product(range(3), range(3)):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1)
            ax.set_title(f"Sample {i * 3 + j + 1}")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            DiffusionModel.plot_rectangle()

            samples = np.array([model.sample() for _ in range(samples_num)])
            ax.scatter(x=samples[:, 0], y=samples[:, 1], s=1)
            ax.title.set_position([0.5, -0.2])
        plt.show()

    @staticmethod
    def q4(model, seed, steps_iter):
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Trajectory of the point for different steps", fontsize=18)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, j in product(range(2), range(2)):
            ax = fig.add_subplot(2, 2, i * 2 + j + 1)
            ax.set_title(f"Steps {steps_iter[i * 2 + j]}")
            DiffusionModel.plot_rectangle()
            steps_num = steps_iter[i * 2 + j]
            _, trajectory = model.sample(steps_=steps_num, seed_=seed, return_trajectory=True)
            trajectory = np.array(trajectory)
            # plot trajectory with color as a function of time
            for t in range(steps_num - 1):
                color = [1 - t / steps_num, t / steps_num, 0]
                ax.plot(trajectory[t][0], trajectory[t][1], 'ro', c=color, markersize=0.5)
            ax.title.set_position([0.5, -0.2])
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("samples of different step sizes", fontsize=18)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, j in product(range(2), range(2)):
            steps_num = steps_iter[i * 2 + j]
            ax = fig.add_subplot(2, 2, i * 2 + j + 1)
            ax.set_title(f"Sample {i * 2 + j + 1} - {steps_num} steps")
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            DiffusionModel.plot_rectangle()

            samples = np.array([model.sample(steps_=steps_num) for _ in range(500)])
            ax.scatter(x=samples[:, 0], y=samples[:, 1], s=1)
            ax.title.set_position([0.5, -0.2])
        plt.show()

    @staticmethod
    def q5():
        # question 5 - needs to be derivable, and monotonically increasing, and between 0 and 1
        steps = np.linspace(0, 1, T)
        plt.title('modifying sigma(t) function')
        plt.plot(steps, scheduler(steps), label='Sigma(t) = exp(5 * (1-t))')
        plt.plot(steps, mod_scheduler_q5(steps), label='Sigma(t) = x^2')
        plt.legend()
        plt.show()

    @staticmethod
    def q6(model, point, steps_num=1000):
        # plot the trajectory of the point, 10 times, with the same noise, to see the effect of the model (all axes on the same plot)
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Trajectory of the point for different steps", fontsize=18)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, j in product(range(3), range(3)):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1)
            ax.set_title(f"run {i * 3 + j + 1}")
            DiffusionModel.plot_rectangle()
            _, trajectory = model.sample(steps_=steps_num, seed_=point, return_trajectory=True)
            trajectory = np.array(trajectory)
            # plot trajectory with color as a function of time
            for t in range(steps_num - 1):
                color = [1 - t / steps_num, t / steps_num, 0]
                ax.plot(trajectory[t][0], trajectory[t][1], 'ro', c=color, markersize=0.5)
            ax.title.set_position([0.5, -0.2])
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Trajectory of the point for different steps with added noise", fontsize=18)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i, j in product(range(3), range(3)):
            ax = fig.add_subplot(3, 3, i * 3 + j + 1)
            ax.set_title(f"run {i * 3 + j + 1}")
            DiffusionModel.plot_rectangle()
            _, trajectory = model.sample(steps_=steps_num, seed_=point,
                                         return_trajectory=True, non_deterministic=True)
            trajectory = np.array(trajectory)
            # plot trajectory with color as a function of time
            for t in range(steps_num - 1):
                color = [1 - t / steps_num, t / steps_num, 0]
                ax.plot(trajectory[t][0], trajectory[t][1], 'ro', c=color, markersize=0.5)
            ax.title.set_position([0.5, -0.2])
        plt.show()

        # on the same plot, plot the trajectory of 4 points, with the same noise:
        fig = plt.figure(figsize=(8, 8))
        fig.suptitle("Trajectory of the point for different steps with added noise", fontsize=18)

        plt.plot([-1, -1], [0.5, 1], color='black')
        plt.plot([-1, -0.5], [1, 1], color='black')

        plt.xlim(-1.6, -0.5)
        plt.ylim(0.5, 1.6)

        # plot the initial point:
        plt.scatter(point[0], point[1], c='black', s=50)

        for i in range(4):
            _, trajectory = model.sample(steps_=steps_num, seed_=point,
                                         return_trajectory=True, non_deterministic=True)
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'ro', c=COLORS[i], markersize=0.5)

            # plot the last point:
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='black', s=50)

        plt.show()

    @staticmethod
    def run_unconditional_model_questions():
        model = Part1.q2()
        Part1.q1(seed=np.array([0, 0]), steps_num=100)
        Part1.q3(model)
        Part1.q4(model, seed=np.array([-1.5, 1.5]), steps_iter=[10, 100, 500, 1000])
        Part1.q5()
        Part1.q6(model, point=np.array([-1.5, 1.5]))


class Part2:
    @staticmethod
    def q1():
        samples, classes = DiffusionModel.get_data(samples_num=1000, conditional=True, as_numpy=True)
        plt.scatter(samples[:, 0], samples[:, 1], c=np.array(COLORS)[classes])
        plt.title('Samples with colors based on their classes')
        plt.show()

    @staticmethod
    def q3():
        model = DiffusionModel(classes_num=CLASS_NUM)
        model.fit(samples_num=3000, epochs_num=3000)
        classes = torch.arange(CLASS_NUM, dtype=torch.float32)

        fig, ax = plt.subplots()
        DiffusionModel.plot_rectangle()
        ax.add_patch(Rectangle((-1, -0.33), 0.66, -0.66, color=COLORS[0], alpha=0.5))
        ax.add_patch(Rectangle((-1, 0.33), 0.66, -0.66, color=COLORS[1], alpha=0.5))
        ax.add_patch(Rectangle((-1, 1), 0.66, -0.66, color=COLORS[2], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, -0.33), 0.66, -0.66, color=COLORS[3], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, 0.33), 0.66, -0.66, color=COLORS[4], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, 1), 0.66, -0.66, color=COLORS[5], alpha=0.5))
        ax.add_patch(Rectangle((0.33, -0.33), 0.66, -0.66, color=COLORS[6], alpha=0.5))
        ax.add_patch(Rectangle((0.33, 0.33), 0.66, -0.66, color=COLORS[7], alpha=0.5))
        ax.add_patch(Rectangle((0.33, 1), 0.66, -0.66, color=COLORS[8], alpha=0.5))

        for i in range(CLASS_NUM):
            _, trajectory = model.sample(steps_=2000, class_=classes[i], return_trajectory=True)
            trajectory = np.array(trajectory)
            ax.plot(trajectory[:, 0], trajectory[:, 1], label=f'class {i}', c=COLORS[i])
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', c=[0, 0, 0], markersize=5)
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
        plt.title('Trajectories of the points for different classes')
        plt.show()

        return model

    @staticmethod
    def q4(model, samples_num=1000):
        sc = [model.sample(steps_=1500) for _ in range(samples_num)]
        samples = np.array([s[0] for s in sc])
        classes = np.array([s[1] for s in sc])

        DiffusionModel.plot_rectangle()
        plt.scatter(samples[:, 0], samples[:, 1], c=np.array(COLORS)[classes])
        plt.title(f'Sample {samples_num} points from the model')
        plt.show()

    @staticmethod
    def q6(model, samples_num=500):
        """ Estimate the probability of at least 5 points.
            • Randomize many possible noise and time combinations.
            • Perform a forward process for all the combinations starting from x as the input (xt=x0+σ(t)·ε ε∼N(0,I)).
            • Estimate x0 for all combinations (xˆ0 =xt −σ(t)D(xt,t)).
            • Compute SNR differences for the sampled t values.
            • Average the results and multiply by T/2 .
            • −LT (x) is the lower bound for log p(x)
            """

        # plot data class areas:
        fig, ax = plt.subplots()
        DiffusionModel.plot_rectangle()
        ax.add_patch(Rectangle((-1, -0.33), 0.66, -0.66, color=COLORS[0], alpha=0.5))
        ax.add_patch(Rectangle((-1, 0.33), 0.66, -0.66, color=COLORS[1], alpha=0.5))
        ax.add_patch(Rectangle((-1, 1), 0.66, -0.66, color=COLORS[2], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, -0.33), 0.66, -0.66, color=COLORS[3], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, 0.33), 0.66, -0.66, color=COLORS[4], alpha=0.5))
        ax.add_patch(Rectangle((-0.33, 1), 0.66, -0.66, color=COLORS[5], alpha=0.5))
        ax.add_patch(Rectangle((0.33, -0.33), 0.66, -0.66, color=COLORS[6], alpha=0.5))
        ax.add_patch(Rectangle((0.33, 0.33), 0.66, -0.66, color=COLORS[7], alpha=0.5))
        ax.add_patch(Rectangle((0.33, 1), 0.66, -0.66, color=COLORS[8], alpha=0.5))

        data = [([-0.66, 0.66], 2), ([-0.66, 0.66], 6), ([4, -2], 1), ([0, 0], 4), ([1.5, 1.5], 3)]
        points = [torch.tensor(d[0], dtype=torch.float32).reshape(1, -1) for d in data]
        classes = [torch.tensor(d[1], dtype=torch.long).reshape(1, -1) for d in data]

        # plot each point with its class:
        for i in range(len(points)):
            ax.scatter(points[i][0, 0], points[i][0, 1], c=COLORS[classes[i][0]])

        # add title and show:
        plt.title('chosen points on the data (classes shown by color):')
        plt.show()

        with torch.no_grad():
            for x, c in zip(points, classes):
                L = []
                for _ in range(samples_num):
                    t = torch.rand(1, dtype=torch.float32).reshape(-1, 1)
                    xt, step_noise = model.forward_noise_step(x, t)
                    x_prime = xt - (scheduler(t) * model.forward(x, t, c))
                    x_L22 = torch.linalg.norm((x - x_prime), ord=2) ** 2
                    snr_t = model.SNR(t - (1/samples_num)) - model.SNR(t)
                    L.append(snr_t * x_L22)

                p_x = torch.exp(-torch.mean(torch.stack(L)) * (T / 2))
                print(f'p(x={x.numpy().reshape(-1, )}, c={c.numpy().reshape(-1, )}) = {p_x}')

    @staticmethod
    def run_conditional_model_questions():
        Part2.q1()
        model = Part2.q3()
        Part2.q4(model)
        Part2.q6(model)


if __name__ == '__main__':
    Part1.run_unconditional_model_questions()
    Part2.run_conditional_model_questions()
