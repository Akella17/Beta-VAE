## dSprites Dataset

[dSprites](https://github.com/deepmind/dsprites-dataset) is a dataset of 2D shapes procedurally generated from 6 ground truth independent latent factors. These factors are color, shape, scale, rotation, x and y positions of a sprite.

All possible combinations of these latents are present exactly once, generating N = 737280 total images.

* Color: white
* Shape: square, ellipse, heart
* Scale: 6 values linearly spaced in [0.5, 1]
* Orientation: 40 values in [0, 2 pi]
* Position X: 32 values in [0, 1]
* Position Y: 32 values in [0, 1]
