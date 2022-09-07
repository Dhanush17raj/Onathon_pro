from re import A
import torch
import numpy as np
from flask import Flask, request, render_template,url_for, redirect
import pickle


app = Flask(__name__)
from txtge.model import TextGenerator
from mains import Execution
from mod.parser import parameter_parser
args = parameter_parser()
execution = Execution(args)
execution.prepare_data()

idx_to_char = execution.idx_to_char
vocab_size = execution.vocab_size
char_to_idx = execution.char_to_idx
model = TextGenerator(args, vocab_size)
model_pi = pickle.load(open("model.pkl", "rb"))
# model_load = model.load_state_dict(torch.load('weights/textGenerator_model.pt'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        seq = request.form["textval"]
        seq = seq.lower()

        execution.generator(model_pi, seq, idx_to_char, char_to_idx, 15)

        textval = Execution.generator(model_pi, seq, idx_to_char, char_to_idx, 15)

        if seq == 'uthra':
            textact = 'Uthrada poonilaave vaa....'
            textlin = 'https://youtu.be/0QtNxqAZlXQ'
            
        elif seq == 'onamn':
            textact = 'Onam nilavile pole onam kinavithal pole...'
            textlin = 'https://youtu.be/W_lWEvwhlbU'
        elif seq == 'onamp':
            textact = 'Onam ponnonam poomala pongum...'
            textlin = ' https://youtu.be/kxw-EO-6Z8I'
        elif seq == 'onamv':
            textact = 'Onam vannallo oonjaalittallo...'
            textlin = 'https://youtu.be/oaSMBo7FYkM'
        elif seq == 'mavel':
            textact = 'Maveli nadu vaneedum kalam...'
            textlin = ' https://youtu.be/4XsmZvalkUY'
        elif seq == 'kutta':
            textact = 'Kuttanadan Punjayile...'
            textlin = ' https://youtu.be/s2R_geXB174'
        elif seq == 'Onapo':
            textact = 'Actual Song: Onapove omal pove Ppookudiyan...'
            textlin = ' https://youtu.be/c0FHgOkqe1U'
        else:
            textact = 'Unable to find the actual song!! Please try some other song.'
            textlin = 'No video available'

    return render_template("result.html", pred=textval, apred=textact, ypred=textlin)
    # return render_template("result.html", apred=textact)
    # return render_template("result.html", ypred=textlin)


@app.route('/vpredict', methods=['POST'])
def vpredict():
    if request.method == "POST":
        seq = request.form["vtext"]
        seq = seq.lower()

        execution.generator(model, seq, idx_to_char, char_to_idx, 15)

        vtext = Execution.generator(model_pi, seq, idx_to_char, char_to_idx, 15)

        if seq == 'uthra':
            texta = 'Uthrada poonilaave vaa....'
            textl = 'https://youtu.be/0QtNxqAZlXQ'
        elif seq == 'onamn':
            texta = 'Onam nilavile pole onam kinavithal pole...'
            textl = 'https://youtu.be/W_lWEvwhlbU'
        elif seq == 'onamp':
            texta = 'Onam ponnonam poomala pongum...'
            textl = ' https://youtu.be/kxw-EO-6Z8I'
        elif seq == 'onamv':
            texta = 'Onam vannallo oonjaalittallo...'
            textl = ' https://youtu.be/oaSMBo7FYkM'
        elif seq == 'mavel':
            texta = ' Maveli nadu vaneedum kalam...'
            textl = ' https://youtu.be/4XsmZvalkUY'
        elif seq == 'kutta':
            texta = 'Kuttanadan Punjayile...'
            textl = ' https://youtu.be/s2R_geXB174'
        elif seq == 'Onapo':
            texta = 'Onapove omal pove Ppookudiyan...'
            textl = 'https://youtu.be/c0FHgOkqe1U'
        else:
            texta = 'Unable to find the actual song!! Please try some other song.'
            textl = 'No video available'

        # return texta, textl

    return render_template("result.html", pred=vtext, apred=texta, ypred=textl)


if __name__ == "__main__":
    app.run(debug=True)
