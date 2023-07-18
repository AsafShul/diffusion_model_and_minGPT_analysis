from diffusion_model import *
from diffusion_questions import Part2

if __name__ == '__main__':
    model = DiffusionModel(classes_num=CLASS_NUM)
    model.fit(samples_num=1000, epochs_num=1000)
    Part2.q4(model)
