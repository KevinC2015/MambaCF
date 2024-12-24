import argparse
def get_params():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="MambaCF")
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--loss', type=str, default='bpr')
    parser.add_argument('--trainset', type=str, default="./dataset/yelp2018/train.txt")
    parser.add_argument('--testset', type=str, default="./dataset/yelp2018/test.txt")
    parser.add_argument('--batch_size', type=int, default=4096)
    
    parser.add_argument('--walk_length', type=int, default=500)
    parser.add_argument('--sample_rate', type=float, default=0.1)
    parser.add_argument('--test_walk_length', type=int, default=500)
    parser.add_argument('--test_sample_rate', type=float, default=0.1)
    parser.add_argument('--bidirection', type=bool, default=False)
    parser.add_argument('--pos_enc', type=bool, default=False)
    parser.add_argument('--gcn', type=bool, default=False)
    parser.add_argument('--m_layers', type=int, default=1)
    

    args, _ = parser.parse_known_args()
    
    return args
