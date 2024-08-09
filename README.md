## Paper Link
Please find our paper at [https://arxiv.org/abs/2405.20628](https://arxiv.org/abs/2405.20628)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/justaguyalways/ToxVidLLM_ACL_2024.git
cd ToxVidLLM_ACL_2024
```

2. Create a conda environment and activate it:

```bash
conda create --name your-env-name python=3.8
conda activate your-env-name
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

1. Download the dataset from the following link: [ToxCMM Dataset Link](https://drive.google.com/drive/folders/1lAl6KpewLv9bO64Ad5fccBOImSZgRPPP?usp=sharing)


2. Unzip the downloaded file:

```bash
unzip dataset.zip
```

3. Move the unzipped folder to the `final_data` directory within the repository:

```bash
mv path_to_unzipped_folder final_data
```

## Usage

### Training

To train the model, run `train.py`. You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable. Replace `xxxx` with the appropriate GPU ID (e.g., `0` for the first GPU).

```bash
CUDA_VISIBLE_DEVICES=xxxx python train.py
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

### Testing

To test the model, run `test.py`. Similarly, you can specify the GPU with `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=xxxx python test.py
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

### Citation

If you use our work or find it useful, please cite:

### Citation

If you use our work or find it useful, please cite:

```plaintext
@misc{maity2024toxvidlmmultimodalframeworktoxicity,
  title={ToxVidLM: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos},
  author={Krishanu Maity and A. S. Poornash and Sriparna Saha and Pushpak Bhattacharyya},
  year={2024},
  eprint={2405.20628},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2405.20628}
}
```

