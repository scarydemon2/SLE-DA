# SLE-DA

the code of 《CAN WE GET MORE FROM LABELS? EVENT EXTRACTION WITHOUT TRIGGERS》IEEE ICASSP 2022

First of all you need to have the original data of ACE2005. Then refer to https://github.com/nlpcl-lab/ace2005-preprocessing for preprocessing. Then put the output file into SLE-DA/date
to train the model 
```
python run_args.py --do_train
```
to eval the model
```
python run_args.py --do_eval
```
