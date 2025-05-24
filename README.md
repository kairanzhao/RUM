# RUM
This is the official code repository of the following papers:

- **NeurIPS 2024: [What makes unlearning hard and what to do about it](https://arxiv.org/abs/2406.01257)**
- **NeurIPS 2024 FITML Workshop: [Scalability of memorization-based machine unlearning](https://openreview.net/pdf?id=VX9HGFiFF1)**

We also release our computed **memorization scores for the CIFAR-10 dataset**, available for download [here](https://drive.google.com/file/d/1RCTrrI8jbCk6n1AWOtWJS3IubY-jtjRl/view?usp=sharing).

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/kairanzhao/RUM.git
   cd RUM
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### 1. **Train an Original Model**
   ```bash
      python main_train.py \
      --dataset {DATASET} \
      --arch {ARCHITECTURE} \
      --epochs {NUM_EPOCHS} \
      --lr {LEARNING_RATE} \
      --batch_size {BATCH_SIZE}
   ```

### 2. **RUM Pipelines**
- **RUM**

   ```bash
   python main_rum.py \
   --unlearn {METHOD} \
   --mem mix \
   --num_indexes_to_replace 3000 \
   --dataset {DATASET} \
   --arch {ARCHITECTURE} \
   --epochs {ORIGINAL_MODEL_EPOCHS} \
   --lr 0.1 \
   --batch_size 256
   ```

- **RUM with memorization proxy**

   ```bash
   python main_rum_proxy.py \
   --unlearn {METHOD} \
   --mem mix \
   --num_indexes_to_replace 3000 \
   --dataset {DATASET} \
   --arch {ARCHITECTURE} \
   --epochs {ORIGINAL_MODEL_EPOCHS} \
   --lr 0.1 \
   --batch_size 256
   ```

### 3. **Other Unlearning Methods**

All other methods are grouped by their entry script. They share these core options:

- --dataset : dataset name (e.g., CIFAR10)

- --arch : model architecture (e.g., ResNet18)

- --num_indexes_to_replace : number of data examples to unlearn

- --mem : memorization group

- --group_index : ES group index

**3.1. Unlearning Methods via main_forget.py**

For Retrain, Fine-tune, L1-sparse, NegGrad, NegGrad+, SCRUB, Influence unlearning, etc.

   ```bash
   python main_forget.py \
   --unlearn {METHOD} \
   --dataset {DATASET} \
   --arch {ARCHITECTURE} \
   --num_indexes_to_replace 3000 \
   --unlearn_lr {LR} \
   --unlearn_epochs {EPOCHS} \
   --mem {MEM_GROUP} \
   --group_index {ES_GROUP} \
   [--alpha {ALPHA}] [--beta {BETA}] [--gamma {GAMMA}] [--msteps {M}] [--kd_T {T}]
   ```

|    Method |    Flag    | Optional Hyperparameters                                         |
| --------: | :--------: | :--------------------------------------------------------------- |
|   Retrain |  `retrain` | —                                                                |
| Fine-Tune |    `FT`    | —                                                                |
| L1-Sparse | `FT_prune` | `--alpha {ALPHA}`                                                |
|   NegGrad |    `GA`    | —                                                                |
|  NegGrad+ |    `NG`    | `--alpha {ALPHA}`                                                |
|     SCRUB |   `SCRUB`  | `--beta {BETA}`, `--gamma {GAMMA}`, `--msteps {M}`, `--kd_T {T}` |
| Influence |  `wfisher` | `--alpha {ALPHA}`                                                |


**3.2. Unlearning Methods via main_random.py**

For SalUn and Random-label.

   ```
   # (Optional) Generate saliency mask for SalUn
   python generate_mask.py \
   --save {MASK_PATH} \
   --num_indexes_to_replace 3000 \
   --unlearn_epochs 1 \
   --mem {MEM_GROUP} \
   --group_index {ES_GROUP}
   
   # Run SalUn or Random-label
   python main_random.py \
   --unlearn {RL|RL_og} \
   --unlearn_lr {LR} \
   --unlearn_epochs {EPOCHS} \
   --num_indexes_to_replace 3000 \
   --mem {MEM_GROUP} \
   --group_index {ES_GROUP} \
   [--path {MASK_PATH}]
   ```

| Method |   Flag  | Optional Hyperparameters |
| -----: | :-----: | :----------------------- |
|  SalUn |   `RL`  | `--path {MASK_PATH}`     |
| Random | `RL_og` | —                        |


### 4. Evaluation with ToW / ToW-MIA

- ToW
   ```bash
   python analysis.py \
   --no_aug \
   --dataset {DATASET} \
   --arch {ARCHITECTURE} \
   --unlearn {METHOD} \
   --mem_proxy {MEM_PROXY} \
   --mem {MEM_GROUP} \
   --num_indexes_to_replace 3000 
   ```

- ToW-MIA
   ```bash
   python analysis_mia.py \
   --no_aug \
   --dataset {DATASET} \
   --arch {ARCHITECTURE} \
   --unlearn {METHOD} \
   --mem_proxy {MEM_PROXY} \
   --mem {MEM_GROUP} \
   --num_indexes_to_replace 3000 
   ```

## Citation
If you find this work useful, please cite our paper(s):
```
@article{zhao2024makes,
  title={What makes unlearning hard and what to do about it},
  author={Zhao, Kairan and Kurmanji, Meghdad and B{\u{a}}rbulescu, George-Octavian and Triantafillou, Eleni and Triantafillou, Peter},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={12293--12333},
  year={2024}
}
```
```
@article{zhao2024scalability,
  title={Scalability of memorization-based machine unlearning},
  author={Zhao, Kairan and Triantafillou, Peter},
  journal={arXiv preprint arXiv:2410.16516},
  year={2024}
}
```

## References
We have used the code from the following repositories:

[SalUn] (https://github.com/OPTML-Group/Unlearn-Saliency)

[SCRUB] (https://github.com/meghdadk/SCRUB)

[Heldout Influence Estimation] (https://github.com/google-research/heldout-influence-estimation)

[Data Metrics] (https://github.com/meghdadk/data-metrics)

We have also used the public pre-computed memorization for CIFAR-100 dataset from https://pluskid.github.io/influence-memorization/


<!-- ### Retrain

```
python main_forget.py --unlearn retrain --num_indexes_to_replace 3000 --unlearn_epochs 30 --unlearn_lr 0.1 --mem {memorization group} --group_index {ES group} 
```

### Fine-tune

```
python main_forget.py --unlearn FT --num_indexes_to_replace 3000 --unlearn_lr 0.01 --unlearn_epochs 10 --mem {memorization group} --group_index {ES group} 
```

### l1-sparse

```
python main_forget.py --unlearn FT_prune --num_indexes_to_replace 3000 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 10 --mem {memorization group} --group_index {ES group} 
```

### NegGrad

```
python main_forget.py --unlearn GA --num_indexes_to_replace 3000 --unlearn_lr 0.0001 --unlearn_epochs 5 --mem {memorization group} --group_index {ES group} 
```

### NegGrad+

```
python main_forget.py --unlearn NG --num_indexes_to_replace 3000 --alpha ${alpha} --unlearn_lr 0.01 --unlearn_epochs 5 --mem {memorization group} --group_index {ES group} 
```

### SCRUB

```
python main_forget.py --unlearn SCRUB --num_indexes_to_replace 3000 --msteps 4 --kd_T 4.0 --beta ${beta} --gamma ${gamma} --unlearn_lr 5e-3 --unlearn_epochs 5 --mem {memorization group} --group_index {ES group} 
```

### Influence unlearning

```
python main_forget.py --unlearn wfisher --num_indexes_to_replace 3000 --alpha ${alpha} --mem {memorization group} --group_index {ES group} 
```


### SalUn

```
python generate_mask.py --save ${saliency_map_path} --num_indexes_to_replace 3000 --unlearn_epochs 1 --mem {memorization group} --group_index {ES group} 
```
```
python main_random.py --unlearn RL --unlearn_lr 0.1 --unlearn_epochs 10 --num_indexes_to_replace 3000 --path ${saliency_map_path} --mem {memorization group} --group_index {ES group} 
```

### Random-label

```
python main_random.py --unlearn RL_og --unlearn_lr 0.1 --unlearn_epochs 10 --num_indexes_to_replace 3000 --mem {memorization group} --group_index {ES group} 
``` -->