import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Text Generation")
    parser.add_argument('-f')
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=0.005)
    parser.add_argument("--hidden_dim", dest="hidden_dim", type=int, default=4)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--window", dest="window", type=int, default=5)
    parser.add_argument("--load_model", dest="load_model", type=bool, default=False)
    parser.add_argument("--model", dest="model", type=str, default='weights/textGenerator.pt')
 
    return parser.parse_args()