# SimCE


## Training with SimCE

1. install the mamba-ssm: https://github.com/state-spaces/mamba




## Run the code

1. go to model/graph and run

```shell
python setup1.py
python setup2.py
```

```shell
python3 main.py --dataset=pinterest --trainset=./dataset/pinterest/train.txt --testset=./dataset/pinterest/test.txt --model=MambaCF --loss=bpr
```






