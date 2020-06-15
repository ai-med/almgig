# Adversarial Learned Molecular Graph Inference and Generation

This is a TensorFlow implementation of "Adversarial Learned Molecular Graph Inference and Generation".

## Installation

### Docker (Linux only)

1. Install [Docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
2. Build Docker image `almgig`:
```bash
cd dockerfiles/
./build-image.sh
```
3. **IMPORTANT:** Run all remaining scripts via the `run-docker.sh` script.
For instance, to run `python train_and_evaluate.py almgig --help`, run
```bash
./run-docker.sh python train_and_evaluate.py almgig --help
```

### Manually

1. Download and install [Miniconda](https://conda.io/en/latest/miniconda.html)
for Python 3.
2. Create a new conda environment `almgig` with all dependencies:
```bash
conda env create -n almgig --file dockerfiles/requirements.yaml
```

3. Activate the new environment:
```bash
conda activate almgig
```

4. Manually install GuacaMol without its dependencies:
```bash
pip install --no-deps 'guacamol==0.3.2'
```

5. Create fake fcd module which is imported by guacamol, but we don't use:
```bash
mkdir $(conda info --base)/envs/almgig/lib/python3.7/site-packages/fcd
touch $(conda info --base)/envs/almgig/lib/python3.7/site-packages/fcd/__init__.py
```

## Data

To download and preprocess the data, go to the `data` directory and
execute the `get-gdb9.sh` script:
```bash
cd data/
./get-gdb9.sh
```

This can take a while. If everything completed successfully, you should see

> All files have been created correctly.

Generated files will be stored in `data/gdb9/`.


## Training

To perform training with the same set of hyper-parameters as in the paper, run
```bash
./train_and_evaluate.sh
```

For more control, directly call he script `train_and_evaluate.py`. To see a full list
of options, run
```bash
python train_and_evaluate.py almgig --help
```

To monitor properties of generated molecules during training,
you can use [TensorBoard](https://www.tensorflow.org/tensorboard):
```bash
tensorboard --logdir models/gdb9/
```

## Evaluation

When performing training as above, statistics for each generated molecule will
be generated automatically, for other models, you can create a file with generated molecules
in SMILES representation (one per line), and execute the following script
to compute statistics:

```bash
python results/grammarVAE_asses_dist.py \
	--strict \
	--train_smiles data/gdb9/graphs/gdb9_train.smiles \
	-i "molecules-smiles.txt" \
	-o "outputs/other-model-distribution-learning.json"
```

To compute and compare descriptors of generated molecules, run

```bash
python -m gan.plotting.compare_descriptors \
    --dist 'emd' \
    --train_file data/gdb9/graphs/gdb9_train.smiles \
    --predict_file \
    "models/gdb9/almgig/distribution-learning_model.ckpt-51500.csv" \
    "outputs/other-model-distribution-learning.json" \
    --name "My Model" "Other Model" \
    --palette "stota" \
    -o "outputs/"
```

This will generate plots of the distribution of descriptors in the
`outputs` directory.

## Acknowledgements

This project contains modified code from the GuacaMol project, see
LICENSE.GUACAMOL for license information.