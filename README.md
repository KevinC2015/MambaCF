

1. install the mamba-ssm: https://github.com/state-spaces/mamba




## Run the code

1. go to model/graph and run

```shell
python setup1.py build_ext --inplace
python setup2.py build_ext --inplace
```
2. go to dataset, and unzip two dateset.zip
3. 
```shell
python3 main.py --dataset=pinterest --trainset=./dataset/pinterest/train.txt --testset=./dataset/pinterest/test.txt --model=MambaCF --loss=bpr --bidirection=True --pos_enc=True --gcn=False --walk_length=200 --sample_rate=0.01 --test_walk_length=200 --test_sample_rate=0.01

```
## Hyperparameter
```
choose
--dataset=[gowalla, yelp2018, amazon-book]then change the --trainset and --testset accordingly.
--bidirection = [True]
--pos_enc= [True]
--gcn= [False, True]
--walk_length=[50,100, 500]
--sample_rate=[0.1, 0.2, 0.5] # large sample_rate will cause memory issue
--test_walk_length=[100, 200, 500]
--test_sample_rate=[ 0.1, 0.2, 0.5] 
```








