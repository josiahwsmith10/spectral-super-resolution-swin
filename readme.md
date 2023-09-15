# Frequency Estimation Using Complex-Valued Shifted Window Transformer

The code accompanying our paper currently in preprint at [arXiv](arxiv.org) for complex-valued shifted window (Swin) transformer-based spectral super-resolution. 

## Publication and Citation
If you appreciate our work, please cite our work as
```
@article{smith2023frequency,
	title        = {Frequency Estimation Using Complex-Valued Shifted Window Transformer},
	author       = {Smith, J. W. and Torlak, M.},
	year         = 2023,
	month        = sep,
	journal      = {arXiv preprint arXiv:2305.02017}
}
```

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* PyTorch
    PyTorch must be install using the CPU/GPU configuration desired prior to attempting to run the code.

* requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

## Training

Both SwinFreq and CVSwinFreq can be easily trained from scratch using the included tools. 

* Training SwinFreq
    ```sh
    python train.py --model swinfreq
    ```

* Training CVSwinFreq
    ```sh
    python train.py --model cvswinfreq
    ```

## Reproducing Paper Experiments

The experiments shown in the paper can be reproduced by calling ```paper_experiments```. The trained models are contained in the folder ```saved/models/```.
    ```sh
    python paper_experiments.py
    ```
After running the experiments ```paper_results/make_figures.m``` can be run in MATLAB to reproduce the figures used to create the figures and save them to both .fig and .png files.

Note: experiment 4, showing a rotating target and demonstrating the improved resolution capacity of the proposed methods, was not used in the paper due to space constraits. 

## License

Distributed under the GPL-3.0 License. See `LICENSE.txt` for more information.

## Contact

Josiah W. Smith - josiah.radar@gmail.com

Project Link: [https://https://github.com/josiahwsmith10/spectral-super-resolution-swin](https://https://github.com/josiahwsmith10/spectral-super-resolution-swin)


