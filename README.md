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
