# Requirements

```
Python=3.7
PyTorch=1.4
```

# Training Process

Step 1. Prepare ACE data. Put the preprocessed ACE data under `data/ace`. The example of preprocessed ACE data is `example.json`in `Data.zip`. The preprocessing code is using http://blender.cs.illinois.edu/software/oneie/. If you have license of ACE data, please feel free to contact manling2@illinois.edu to get the preprocessed data.

Step 2. Generate paths.

```
cd data_utils/preprocessing/ace
python path_discover.py
```
Step 3. Generate training data for autoregressive language model and neighbor path classfication.

```
cd data_utils/preprocessing/ace
python path_tsv_vocab.py
``` 

Step 4. Train PathLM on two tasks,

```
sh path_xlnet_ft.sh
```
The variant of PathLM removing neighbor path classification can be trained as follows,

```
sh path_xlnet_ft_clm.sh
```

# Testing Process

```
cd data_utils/preprocessing/ace
python evaluate_path.py
```