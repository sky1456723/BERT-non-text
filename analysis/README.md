# Analysis

## negative

In the `negative` folder, there are python scripts corresponding to the following experiments:

* `dynamical_isometry.py` and `dynamical_isometry.sh`: [Resurrecting the sigmoid in deep learning through dynamical isometry: theory and practice](https://arxiv.org/abs/1711.04735). This code calculates the output-input jacobian matrices and their singular values using GLUE dataset. The input arguments of the .py file are: 
```
--model: the pretrained model to use (e.g. bert-base-uncased)
--task: the glue task to use (e.g. MNLI)
--pretrain: use the pre-trained model
--scratch: use the randomly initialized model
--save_dir: dir to save the results
--glue_dir: dir of glue data
--shift: value to shift the token as in the GLUE experiments (e.g. 1000)
--seed: random seed
```
* `gradient_confusion.py` and `grad_conf.sh`: [The Impact of Neural Network Overparameterization on Gradient Confusion and Stochastic Gradient Descent](https://arxiv.org/abs/1904.06963). This code calculates the gradient cosine similarity between batches using GLUE dataset. The input arguments are similar to `dynamical_isometry.py` except that:
```
--mode: use the pre-trained or randomly initialized model, default "scratch pretrain" runs both models.
--accumulation: gradient accumulation
--op: the operation to calculate similarity. Available options: cosine, dot (dot product), l2 (l2 distance).
```
* `perturbation.py` and `perturbation.sh`: [Understanding the Difficulty of Training Transformers](https://arxiv.org/abs/2004.08249). This code calculates the output variation of the model after adding noise to its parameters. The input arguments are similar to the above code except that `--std` means the std of the gaussian noise.
* `Lipschitz.py` and `run_lipschitz.sh`: [Lipschitz Constrained Parameter Initialization for Deep Transformers](https://arxiv.org/abs/1911.03179). This code calculates the std in layernorm of BERT. The input arguments are similar to the above.

You can execute the `.sh` files to run the examples.

## PWCCA_and_attn_match

In this folder, `BERT_and_PLUS_extract.py` can extract the hidden representations or attention maps of BERT and PLUS using protein data. The input arguments are:
```
--protein_dir: the dir of protein data
--save_dir: the dir to save the results
--pretrained_BERT: use the pre-trained BERT
--bert_ckpt: ckpt path to load the fine-tuned BERT. If this argument is not assigned, 
it will use the pre-trained model.
--scratch_ckpt: ckpt path to load the trained from scratch model. If this argument is not assigned, 
it will use the randomly initialized model.
--plus_ckpt: ckpt path to load PLUS model. If this argument is not assigned, PLUS model will not be used.
--task: the name of protein task to be used
--feature: this argument should be 'hidden' or 'attention'. 
'hidden' will extract the hidden representations. 'attention' will extract the attention maps.
```

You can execute `extract.sh` to run an example.

After extracting the hidden representations, you can run `feature_pwcca.py` to calculate the PWCCA similarity. The input arguments are the task name (`--task`) and the dir name that the extracted representtations are saved (`--data_dir`).

After extracting the attention maps, you can run `matching.py` to calculate the bipartite matching and the total l1 distance of the matching. The code will draw the results as a figure. The input argument is the dir name that the attention maps are saved (`--data_dir`).
