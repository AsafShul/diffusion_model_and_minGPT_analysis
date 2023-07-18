# Diffusion and minGPT analysis:
## Unconditional diffusion model:
this analysis uses 2D data for low computetional resource requirements, but the main ideas are the same for higher dimention data.
### plot of the forward noise process of a random sample:
1

### Denoiser train loss:
2

### DDIM sampling for 9 different seeds:
3

### Effect of the number of denoising steps on the generated point (on a single point and for large nunber of points):

### trials with a different sampler:

no noticable differance

### adding noise to the sampling process:

another view:
--------
## conditional diffusion model:
### training
train data:

train loss:

### sampling one point from each class:

### sampling 1000 random samples from the model:

### point estimation for 5 representaive samples:
check for class variation, OOD to variable degrees and probable samples

--------
## minGPT
 to make the minGPT part work - clone the mingpt repo and replace the model.py file from there.

### train loss:

### latents optimization (LO) for making the model output a never seen before sentance:
the wanted sentance is "I am a little  squirrel holding a walnut"

### Attention analasys for the last transformer block:

### Attention analasys for the first transformer block:

### Attention visualization:

### probability of an example generated output:




