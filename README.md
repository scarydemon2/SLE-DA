# SLE-DA

the code of 《CAN WE GET MORE FROM LABELS? EVENT EXTRACTION WITHOUT TRIGGERS》IEEE ICASSP 2022

First of all you need to have the original data of ACE2005. Then refer to https://github.com/nlpcl-lab/ace2005-preprocessing for preprocessing. Then put the output file into SLE-DA/data
to train the model 
```
python run_args.py --do_train
```
to eval the model
```
python run_args.py --do_eval
```
results are shown at below table,ours(SLE-DA)

| models  | Event Detection F1 | Argument Extraction F1 |
| ------- | ------------------ | ---------------------- |
| DMCNN   | 69.1               | 53.5                   |
| PLMEE   | 80.7               | 58.0                   |
| JRNN    | 69.3               | 55.4                   |
| JMEE    | 73.7               | 60.3                   |
| BERT-QA | 72.4               | 53.5                   |
| MQAEE   | 73.8               | 55.0                   |
| TBNNAM  | 69.9               | -                      |
| **SLE-DA**  | 77.4               | 58.7                   |
