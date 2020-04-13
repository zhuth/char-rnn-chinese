import torch
from torch.utils.data import DataLoader
from utils import *
from model import *

from test import predict

np.random.seed(17)

n_step = 30
start_epoch = 0
end_epoch = 100
batch_size = 128

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint",
                        help="chekpoint file", dest="checkpoint", default="")
    parser.add_argument("-i", "--training",
                        help="training data", dest="training", default="data/存在与时间.txt")
    parser.add_argument("-b", "--batch-size",
                        help="batch size", dest="batch_size", type=int, default=128)
    parser.add_argument("-e", "--end-epoch",
                        help="epoches", dest="end_epoch", type=int, default=100)
    parser.add_argument("-d", "--checkpoint-dir",
                        help="default directory for saving checkpoints", dest="ckpt", default="ckpt")
    args = parser.parse_args()

    batch_size = args.batch_size
    end_epoch = args.end_epoch
    training_name = os.path.basename(args.training).rsplit('.', 1)[0]
    corpus_pth = args.training + '.pth'

    if os.path.exists(corpus_pth):
        corpus = Corpus(torch.load(corpus_pth))
    else:
        corpus = Corpus(args.training)
        corpus.save(corpus_pth)
    
    vocab_size = corpus.vocab_size
    num_seq = int(corpus.length / n_step)
    
    arr = corpus.text_to_arr(corpus.text[:num_seq*n_step]).reshape((num_seq, -1))
    arr = torch.from_numpy(arr).long()

    model, loss_function, optimizer, start_epoch = init_model(corpus, args.checkpoint)

    train_set = TextDataset(arr)
    train_data = DataLoader(train_set, batch_size, True, num_workers=4)

    min_loss = None
    
    for epoch in range(start_epoch, end_epoch):
        train_loss = 0
        for data in train_data:
            x, y = data
            y = y.long()
            x = x.to(device)
            y = y.to(device)

            x, y = Variable(x), Variable(y)

            # Forward.
            score, _ = model(x)
            loss = loss_function(score, y.view(-1))

            # Backward.
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        print(f'epoch: {epoch+1}, perplexity is: {np.exp(train_loss / len(train_data)):.3f}')
        predict(model, corpus, '', 100)
        if min_loss is None or min_loss > train_loss:
            min_loss = train_loss
            torch.save({
                'epoch': epoch,            
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'arr': arr,
                'vocab_size': corpus.vocab_size
            }, f'{args.ckpt}/checkpoint_{training_name}_{epoch}.pth')
