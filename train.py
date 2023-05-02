import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import Transformer
import numpy as np
from dataloader import TextDataset,Vocabulary 
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# python train.py --file_path "t8.shakespeare.txt" --embed_dim 512 --chunk_size 100 --num_layers 2 --batch_size 16 --num_epochs 200

parser = argparse.ArgumentParser(description='Train a Transformer model on text data')
# Add the command line arguments
parser.add_argument('--file_path', type=str, default='shakespeare.txt', help='Path to the text file')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size of each data sample')
parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
args = parser.parse_args()
# args = parser.parse_args()



def train(model, optimizer, criterion, dataloader, device,epoch,vocab):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    flag = True
    for i, batch in enumerate(dataloader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if flag and epoch % 20 == 0:
            outputs = outputs.cpu()
            max_indices = torch.argmax(outputs, dim=-1)
            tokens = [vocab.decode(list(max_indices[0].cpu().numpy()))]
            # join the tokens to form the output text
            output_text = " ".join(tokens)
            print(f"{output_text[:200]}")
            flag = False

        outputs = outputs.to(device)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(outputs, dim=-1)
        mask = (targets != 0)  # ignore padding
        num_correct = torch.sum(predictions[mask] == targets[mask])
        num_tokens = torch.sum(mask)
        total_correct += num_correct.item()
        total_tokens += num_tokens.item()
        total_loss += loss.item()

    accuracy = total_correct / total_tokens
    avg_loss = total_loss / len(dataloader)

    return avg_loss , accuracy




if __name__ == "__main__":
    # Now you can use these arguments in your code
    file_path = args.file_path
    embed_dim = args.embed_dim
    num_layers = args.num_layers
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # file_path = "t8shakespeare.txt"
    embed_dim = 512
    num_layers = 3
    batch_size = 20
    num_epochs = 80

    vocab = Vocabulary()
    chunk_size = 512
    # num_epochs = 1000
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextDataset(file_path, vocab, chunk_size, _type="word")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # # Hyperparameters
    vocab_size = len(dataset.vocab.stoi)
    vocab = dataset.vocab
    n_heads = 8
    embed_dim = 512
    num_layers = 3
    max_seq_len = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = np.zeros(num_epochs)
    train_accuracies = np.zeros(num_epochs)

    for epoch in range(1,num_epochs+1):
        train_loss , train_accuracy = train(model, optimizer, criterion, dataloader,device=device,epoch=epoch,vocab=vocab)
        train_losses[epoch-1] , train_accuracies[epoch-1] = train_loss , train_accuracy
        # save the model and training details
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy':train_accuracy
        }
        torch.save(checkpoint, f'pretrained_models/gpt2_on_{file_path}.pth')
        print(f"Epoch {epoch} | Loss {train_loss:>.3f} | Accuracy {train_accuracy * 100:>.2f}")
    
    # plot training loss
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    # return train_losses,train_accuracies