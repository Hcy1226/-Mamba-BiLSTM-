import argparse
import sys
import os

# 确保当前目录在 sys.path 中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from test import test_model

def main():
    parser = argparse.ArgumentParser(description="Mamba-BiLSTM DTI Framework Runner")
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or test')
    
    # Train Parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, default=r'e:\基于Mamba与BiLSTM混合架构\data\Davis.txt', help='Path to data file')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--folds', type=int, default=5, help='Number of folds for Cross Validation')
    train_parser.add_argument('--model_name', type=str, default='mamba_bilstm', 
                              choices=['mamba_bilstm', 'deepdta', 'transformer', 'graphdta', 'mcanet'],
                              help='Model architecture to use')
    train_parser.add_argument('--debug', action='store_true', help='Run in debug mode (fast, small data)')
    
    # Test Parser
    test_parser = subparsers.add_parser('test', help='Test the model')
    test_parser.add_argument('--weights', type=str, required=True, help='Path to model weights file')
    test_parser.add_argument('--data', type=str, default=r'e:\基于Mamba与BiLSTM混合架构\data\Davis.txt', help='Path to test data file')
    test_parser.add_argument('--model_name', type=str, default='mamba_bilstm', 
                            choices=['mamba_bilstm', 'deepdta', 'transformer', 'graphdta', 'mcanet'],
                            help='Model architecture to use')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Starting Training...")
        train_model(args.data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, folds=args.folds, model_name=args.model_name, debug=args.debug)
        
    elif args.mode == 'test':
        print("Starting Testing...")
        test_model(args.weights, args.data, model_name=args.model_name)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
