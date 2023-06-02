# MEFTs
Official code for paper **Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning**

## Features
- [x] BERT on GLUE
- [ ] RoBERTa on GLUE
- [ ] BART on GLUE
- [ ] OPT on Question-Answering

## Installation
```bash
conda create -n mefts python=3.8
conda activate mefts
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Fine-Tuning
#### Run GLUE Experiments with BERT
- Edit the **#TODO** places in [scripts/bert_glue.sh](/scripts/bert_glue.sh)
- Run as
  ```bash
  bash scripts/bert_glue.sh
  ```
  
## Citation
If you find our work or code useful, please cite as:
  ``` bibtex
  @misc{liao2023make,
      title={Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning}, 
      author={Baohao Liao and Shaomu Tan and Christof Monz},
      year={2023},
      eprint={2306.00477},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
  ```
