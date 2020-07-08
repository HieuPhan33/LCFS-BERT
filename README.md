# **LCFS-BERT**
### Requirement
* Pytorch >= 1.0
* Pytorch-transformer == 1.2.0 ```pip install pytorch-transformers==1.2.0```
### 1. Data preprocessing:
Download the SemEval dataset on http://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools

XML files are converted to text files, where each input sentence follows the format below:
```
Sentence: "$T$ is super fast , around anywhere from 35 seconds to 1 minute ."
Target word in place of $T$: "Boot time"
Polarity: 1
```

For exampled output, please find files in datasets/semeval14

### 2. Training
The command to train LCFS-BERT with context dynamic weight and SRD threshold = 4:
```python train.py --model_name lcfs_bert --dataset restaurant \
 --pretrained_bert_name bert-base-cased \
 --batch_size 32 --local_context_focus cdw --SRD 4
```

### Note
Some important scripts to note:
* datasets/semeval14/*.seg: Preprocessed training and testing sentences in SemEval2014.
* models/lcfs_bert.py: the source code of LCFS_BERT model.
* data_utils.py/ABSADataSet class: preprocess the tokens and calculates the shortest distance to target words via the syntax tree.

### Acknowledgement
We have based our model development on https://github.com/songyouwei/ABSA-PyTorch. Thanks for their contribution.
### Citation
If you found this repository is helpful, please cite our paper:
```
@inproceedings{phan2020modelling,
  title={Modelling Context and Syntactical Features for Aspect-based Sentiment Analysis},
  author={Phan, Minh Hieu and Ogunbona, Philip O},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3211--3220},
  year={2020}
}
```
