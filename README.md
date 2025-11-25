# Final Project
## Overview

One of the central challenges in meteorology is obtaining high-resolution gridded datasets that accurately represent atmospheric conditions. Traditionally, this has been achieved using numerical weather prediction (NWP) models, which can provide detailed fields but are computationally expensive.

A faster and more cost-effective alternative is the use of machine-learning–based downscaling, where ML models enhance coarse-resolution data by reconstructing the high-frequency details and sharp spatial gradients that are often smoothed out or lost during initial data generation. This process is commonly referred to as super-resolution or high-resolution reconstruction.

In this project, we apply a diffusion-based super-resolution model to meteorological fields to evaluate its ability to recover high-resolution atmospheric features.

For this purpose we use RTMA (Real-Time Mesoscale Analysis) data. RTMA is an operational, high-resolution, near-surface weather analysis produced by the National Centers for Environmental Prediction (NCEP), part of NOAA’s National Weather Service. It provides gridded fields of current weather conditions and serves as an excellent benchmark for evaluating ML-based downscaling methods.

## Diffusion model

A diffusion model is a generative machine learning model that learns to denoise data step-by-step.
It does two stages:

*(1)* Forward Diffusion

It gradually adds Gaussian noise to an HR image (or meteorological field), until the image becomes almost pure noise.

*(2)* Reverse Diffusion (Learning the Physics of the Data)

The model trains to reverse this process:

## STEPS
STEP 1: Take HR (2.5 km) data  
STEP 2: Apply degradation operator D → create LR (7.5 km)  
STEP 3: Train diffusion model on (LR → HR) pairs  
STEP 4: Use trained model on new LR data to generate super-resolution outputs

## Notes

Degradation operator D: Gaussian blur + block-average downsampling.
Converts high-resolution (2.5 km) data into low-resolution (7.5 km) data.

sigma_px is the standard deviation (σ) of the Gaussian blur kernel, measured in pixels (grid cells).
So when we call 
ndi.gaussian_filter(data, sigma=0.8)
the filter smooths the data with a Gaussian whose spread is 0.8 grid cells wide.

Physical interpretation for our case grid σ=0.8 pixels≈0.8×2.5=2.0km

That means the blur smooths over roughly a 2 km radius, mimicking how a 7.5 km model would not “see” details smaller than a few kilometers.

Without this blur, if you simply take every 3rd grid cell (downsampling factor = 3), high-frequency details would alias — small-scale variations fold into larger-scale ones and create unrealistic patterns.

The Gaussian filter acts as a low-pass (anti-alias) filter, removing features smaller than your new grid spacing before coarsening.

So your degradation operator is: 𝐷(𝑥𝐻𝑅)=Downsample(GaussianBlur(𝑥𝐻𝑅,𝜎=0.8))

How to choose σ:

σ ≈ 0.8–1.0 px → good starting range for mild smoothing (keeps patterns realistic).

σ < 0.5 px → almost no blur, aliasing may appear.

σ > 1.5 px → too smooth, removes too much small-scale detail (diffusion model learns an easier but less realistic task).