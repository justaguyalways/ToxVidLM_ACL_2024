## Paper Link
Please find our paper at [https://aclanthology.org/2024.findings-acl.663/](https://aclanthology.org/2024.findings-acl.663/)

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

```plaintext
@inproceedings{maity-etal-2024-toxvidlm,
    title = "{T}ox{V}id{LM}: A Multimodal Framework for Toxicity Detection in Code-Mixed Videos",
    author = "Maity, Krishanu  and
      Sangeetha, Poornash  and
      Saha, Sriparna  and
      Bhattacharyya, Pushpak",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.663",
    pages = "11130--11142",
}
```

