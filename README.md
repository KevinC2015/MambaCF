

1. install the mamba-ssm: https://github.com/state-spaces/mamba




## Run the code

1. go to model/graph and run

```shell
python setup1.py build_ext --inplace
python setup2.py build_ext --inplace
```

```shell
python3 main.py --dataset=pinterest --trainset=./dataset/pinterest/train.txt --testset=./dataset/pinterest/test.txt --model=MambaCF --loss=bpr --bidirection=False --pos_enc=False --gcn=False --walk_length=200 --sample_rate=0.01 --test_walk_length=200 --test_sample_rate=0.01

```
## Hyperparameter
```
choose
--dataset=[our 13 datasets.], then change the --trainset and --testset accordingly.
--bidirection = [False, True]
--pos_enc= [False. True]
--gcn= [False, True]
--walk_length=[20,50,100,200, 500, 1000]
--sample_rate=[0.01, 0.1, 0.2, 0.5] # large sample_rate will cause memory issue
--test_walk_length=[100,200, 500, 1000]
--test_sample_rate=[0.01, 0.1, 0.2, 0.5] 
```








