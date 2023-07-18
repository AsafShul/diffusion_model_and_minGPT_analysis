# Diffusion and minGPT analysis:
## Unconditional diffusion model:
this analysis uses 2D data for low computetional resource requirements, but the main ideas are the same for higher dimention data.
### plot of the forward noise process of a random sample:
![Screenshot 2023-07-18 at 11 27 42](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/210f8e96-44a3-4cbf-be04-841d717922dd)

### Denoiser train loss:
![Screenshot 2023-07-18 at 11 27 59](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/aeb8ffba-6ee3-4212-839b-89f22ebde280)

### DDIM sampling for 9 different seeds:
![Screenshot 2023-07-18 at 11 28 15](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/35a37a46-f4b5-40c7-91cf-77b7c17ddfeb)

### Effect of the number of denoising steps on the generated point (on a single point and for large nunber of points):
![Screenshot 2023-07-18 at 11 28 31](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/5d2bad18-244a-47d6-a8bb-4c4ae38be20f)

### trials with a different sampler:
![Screenshot 2023-07-18 at 11 28 46](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/219fd717-d3e5-4820-a8b6-333d903e97b2)

no noticable differance

### adding noise to the sampling process:
![Screenshot 2023-07-18 at 11 29 04](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/ea0aeabf-efac-4a6e-a933-7a2877dad722)

another view:

![Screenshot 2023-07-18 at 11 29 16](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/80f3a285-3558-44f6-986c-c8e32a453944)

--------
## conditional diffusion model:
### training

Generated train data:
![Screenshot 2023-07-18 at 11 29 29](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/0192a3f9-df34-404b-8cf6-7a4499e15f68)

Train loss:


![Screenshot 2023-07-18 at 11 35 08](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/8d8f64c6-e193-4732-9f8d-97a39dd77582)


### sampling one point from each class:
![Screenshot 2023-07-18 at 11 30 08](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/d093509a-91a4-4391-911b-e93c12cedf81)

### sampling 1000 random samples from the model:
![Screenshot 2023-07-18 at 11 30 17](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/405b730a-a61f-476c-a98e-5ca3a9d7bcae)

### point estimation for 5 representaive samples:
check for class variation, OOD to variable degrees and probable samples
![Screenshot 2023-07-18 at 11 31 11](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/87b20d96-edc5-4fe5-b36e-cfeb43ea6f6d)

--------
## minGPT
 to make the minGPT part work - clone the mingpt repo and replace the model.py file from there.

### train loss:
![Screenshot 2023-07-18 at 11 35 33](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/1938cf25-c85f-44e2-9ffb-7ab24063fbe1)

### latents optimization (LO) for making the model output a never seen before sentance:
the wanted sentance is "I am a little  squirrel holding a walnut"
![Screenshot 2023-07-18 at 11 31 52](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/e07c8777-38e2-488d-98f7-ef3fb3ed77f0)

### Attention analasys for the last transformer block:
![Screenshot 2023-07-18 at 11 36 03](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/280298aa-b5ce-49ca-a527-6fedf203fd70)

### Attention analasys for the first transformer block:
![Screenshot 2023-07-18 at 11 36 13](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/dcbbee97-9d32-4f1c-ac7a-8b1075549747)

### Attention visualization:
![Screenshot 2023-07-18 at 11 36 27](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/ff04abad-23eb-4765-9527-4bc39fb4524f)

### probability of an example generated output:

![Screenshot 2023-07-18 at 11 36 49](https://github.com/AsafShul/diffusion_model_and_minGPT_analysis/assets/44872433/4a107a96-526a-4a7a-8f79-2307ed6c9e57)
