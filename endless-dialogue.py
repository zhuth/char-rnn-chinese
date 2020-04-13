from test import predict
from utils import Corpus
from model import init_model, torch
from flask import request, Flask, jsonify, redirect
import glob
import random


app = Flask(__name__)


class Persona:
    def __init__(self, name):
        self.name = name
        self.corpus = Corpus(torch.load(f'data/{name}.txt.pth'))
        model, *_ = init_model(self.corpus, f'{name}.pth')
        self.model = model.eval()
        
    def gen_text(self, seed, length):
        text = predict(self.model, self.corpus, seed, length).strip()
        if '。' in text: text = text[:text.rfind('。')+1]
        elif '，' in text: text = text[:text.rfind('，')] + '。'
        else: text += '……'
        return text


personas = []
for g in glob.glob('*.pth'):
    if 'checkpoint' in g: continue
    personas.append(Persona(g.rsplit('.', 1)[0]))


def dialogue(iters, seed, length):
    p0 = None
    for _ in range(iters):
        p = random.choice([p_ for p_ in personas if p_ != p0])
        p_ = p
        seed = p.gen_text(seed, length)[len(seed):]
        yield {
            'persona': p.name, 
            'text': seed
        }
    

@app.route('/gen')
def gen_text():
    length = int(request.args.get('length', 200))
    seed = request.args.get('seed', '')
    iters = int(request.args.get('iters', 5))
    return jsonify(list(dialogue(iters, seed, length)))


@app.route('/')
def index():
    return redirect('statics/index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8765, debug=True)