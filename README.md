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
```
python train.py --model_name lcfs_bert --dataset restaurant \
 --pretrained_bert_name bert-base-cased \
 --batch_size 32 --local_context_focus cdw --SRD 4
```

### Note
Some important scripts to note:
* datasets/semeval14/*.seg: Preprocessed training and testing sentences in SemEval2014.
* models/lcfs_bert.py: the source code of LCFS_BERT model.
* data_utils.py/ABSADataSet class: preprocess the tokens and calculates the shortest distance to target words via the syntax tree.

### CSAE script
You can find the source code of CSAE model from this link.
https://1drv.ms/u/s!AsJP8s8Vd4SChQ9XZLTgdCdOEzOt?e=cCjajJ

Please run the script src/run_ae.py as:

python run_ae.py \ --bert_model roberta-base --do_train --do_valid \ --max_seq_length 100 --train_batch_size 32 --learning_rate 3e-5 --num_train_epochs 20 \ --output_dir LAPTOP_OUT_DIR --data_dir ../ae/laptop
To validate:
python eval/evaluate.py --pred LAPTOP_OUT_DIR/prediction.json --target data/laptop/laptops--test.gold.xml

The CSAE model is the class RobertaPOSClassificationHead in model.py.
It uses the dependency-word embedding in "/ae/emb.npy".

Those are some basic usage about the model.

To compute and verify the data statistics we used for training the model, please run the script `data_check.py`.
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
