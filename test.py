from utils import *
from model import *


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


def predict(model, corpus, begin, sentence_len):
    if not begin:
        begin = corpus.text[np.random.randint(0, corpus.length-10):][:10]
    input_current = Variable(torch.LongTensor(
        [[corpus.word_to_int(c) for c in begin]]))
    _, init_state = model(input_current.to(device))
    result = input_current[0].tolist()

    for _ in range(sentence_len):
        out, _ = model(input_current.to(device), init_state)
        _, pred = torch.max(out[-1], 0)

        if len(input_current) < sentence_len:
            input_current = torch.LongTensor([input_current.tolist()[0] + [pred]])
        else:
            input_current = torch.LongTensor([input_current.tolist()[0][1:] + [pred]])

        result.append(pred.item())

    text = corpus.arr_to_text(result)
    print(begin + '|' + text[len(begin):])
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint",
                        help="chekpoint file", dest="checkpoint", default="")
    parser.add_argument("-i", "--corpus",
                        help="corpus file", dest="corpus", default="")
    parser.add_argument("-s", "--seed",
                        help="seeding sentence", dest="seed", default="")
    parser.add_argument("-l", "--length",
                        help="length", dest="length", default=100, type=int)
    args = parser.parse_args()

    corpus = Corpus(torch.load(args.corpus + '.pth'))
    model, *_ = init_model(corpus, args.checkpoint)
    model = model.eval()
    text = predict(model, corpus, args.seed, args.length)
    print(text)
