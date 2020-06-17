# Running Baselines

## Prerequisites

1. Make sure you generated the GDB9 data splits following the instructions
   in the README located in the root of this repository.
2. Initialize the repositories containing the code to run baselines by
   executing:

   ```bash
   git submodule update --init
   ```
3. Setup environments for the different models:

    - CGVAE and MolGAN: `conda env create -n cgvae --file requirements_cgvae.yaml`
    - GrammarVAE: `conda env create -n gvae --file requirements_grammarvae.yaml`
    - NeVAE: Use the same environment as for ALMGIG (see README located
      in the root of this repository).


## Constrained Graph Variational Autoencoder (CGVAE)

1. Go to the `CGVAE/data` directory and update `fname` at the bottom
   of the `get_qm9.py` file to point to the generated splits of the GDB9 data.
2. Create JSON data:

    ```bash
    python get_qm9.py
    ```

3. Go to the `CGVAE` directory and train the model:

    ```bash
    python CGVAE.py --dataset qm9
    ```

4. Sample molecules:

    ```bash
    python CGVAE.py --dataset qm9 \
        --restore "10_qm9.pickle" \
        --config '{"generation": true, "number_of_generation": 10000}'
    ```

    The file `generated_smiles_qm9.txt` will contain generated molecules
    in SMILES format.


## MolGAN

1. Go to the `MolGAN/data` directory and run:

    ```bash
    wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
    wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz
    ```

2. Go to the `MolGAN` directory and generate the data:

    ```bash
    python utils/sparse_molecular_dataset.py \
        --train "../../data/gdb9/graphs/gdb9_train.smiles" \
        --validation "../../data/gdb9/graphs/gdb9_valid.smiles" \
        --test "../../data/gdb9/graphs/gdb9_test.smiles" \
        --output "data/qm9-mysplits-data.pkl"
    ```

3. Train the model:

    ```bash
    python example.py
    ```

4. Sample molecules

    ```bash
    python predict.py \
        --model_dir "GraphGAN/norl/lam1/" \
        --number_samples 10000 \
        -o "generated_molecules.csv"
    ```

    Generated molecules in SMILES format will be written to
    `generated_molecules.csv`.


## NeVAE

1. Run the script `get-gdb9-with-hydrogens.sh` in the `data` directory
   located in the root of this repository.

2. Train the model by running `train_nevae.sh`. The script automatically
   samples a number of molecules once training was completed.
   Multiple CSV files with generated molecules in SMILES format will be
   located in the `models/nevae-poisson-masked` directory.


## GrammarVAE

1. Go the `data/gdb9/graphs` folder at the root of this repository and
   concatenate all GDB9 data:

    ```bash
    cat gdb9_test.smiles gdb9_train.smiles gdb9_valid.smiles > gdb9.smiles
    ```

2. Go to the `grammarVAE` directory, open the file
   `make_gdb9_dataset_grammar.py` and change `f` at the top
   of the file to point to `gdb9.smiles` created above, then run

    ```bash
    python make_gdb9_dataset_grammar.py
    ```

3. Train the model

    ```bash
    python train_gdb9.py
    ```

4. Sample molecules

    ```bash
    python sample_gdb9.py
    ```

    Generated SMILES strings will be written to `gdb9-generated.smi`.
    *Note that generated strings can be invalid SMILES*.


## Random Graph Generation

To generate molecules randomly, while imposing valence constraints, run:
```bash
python generate_random.py --output random_samples.csv
```
Molecules in SMILES format will be written to `random_samples.csv`.
