import argparse
from train_augment_net_multiple import make_argss
from train_augment_net2 import experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slurm deployment')
    parser.add_argument('--deploy_num', type=int, default=0, help='The deployment number')
    args = parser.parse_args()

    deploy_argss = make_argss()
    deploy_args = deploy_argss[args.deploy_num]

    print(f"Launching {args.deploy_num}, {deploy_args}")
    experiment(deploy_args)
    print(f"Finished {args.deploy_num}, {deploy_args}")
