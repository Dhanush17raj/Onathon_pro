from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)
from txtge.model import TextGenerator
from mains import Execution
from mod.parser import parameter_parser

model_pi = pickle.load(open("model.pkl", "rb"))

args = parameter_parser()

execution = Execution(args)
execution.prepare_data()
            

idx_to_char = execution.idx_to_char
vocab_size = execution.vocab_size
char_to_idx = execution.char_to_idx




model = TextGenerator(args, vocab_size)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == "POST":
        seq = request.form["textval"]
        seq = seq.lower()
        

        execution.generator(model,seq , idx_to_char, char_to_idx,15)

        textval = Execution.generator()


        if seq == 'uthra':
           textact = 'Actual Song: Uthrada poonilaave vaa....'
           textlin = 'Youtube link: https://youtu.be/0QtNxqAZlXQ'
        elif seq == 'onamn':
            textact ='Actual Song: Onam nilavile pole onam kinavithal pole...'
            textlin ='Youtube link: https://youtu.be/W_lWEvwhlbU'
        elif seq  == 'onamp':
            textact ='Actual Song: Onam ponnonam poomala pongum...'
            textlin ='Youtube link: https://youtu.be/kxw-EO-6Z8I'
        elif seq == 'onamv':
            textact ='Actual Song: Onam vannallo oonjaalittallo...'
            textlin ='Youtube link: https://youtu.be/oaSMBo7FYkM'
        elif seq == 'mavel':
            textact ='Actual Song: Maveli nadu vaneedum kalam...'
            textlin ='Youtube link: https://youtu.be/4XsmZvalkUY'
        elif seq == 'kutta':
            textact ='Actual Song: Kuttanadan Punjayile...'
            textlin ='Youtube link: https://youtu.be/s2R_geXB174'
        elif seq == 'Onapo':
            textact ='Actual Song: Onapove omal pove Ppookudiyan...'
            textlin ='Youtube link: https://youtu.be/c0FHgOkqe1U'
        else:
            textact ='Unable to find the actual song!! Please try some other song.'
            textlin ='No video available'





    return render_template("result.html", pred=textval, apred=textact, ypred=textlin)
    # return render_template("result.html", apred=textact)
    # return render_template("result.html", ypred=textlin)



@app.route('/vpredict', methods = ['POST'])
def vpredict():
    if request.method == "POST":
        seq = request.form["vtext"]
        seq = seq.lower()

        execution.generator(model,seq , idx_to_char, char_to_idx,15)

        vtext = Execution.generator()


        if seq == 'uthra':
           texta = 'Actual Song: Uthrada poonilaave vaa....'
           textl = 'Youtube link: https://youtu.be/0QtNxqAZlXQ'
        elif seq == 'onamn':
            texta ='Actual Song: Onam nilavile pole onam kinavithal pole...'
            textl ='Youtube link: https://youtu.be/W_lWEvwhlbU'
        elif seq  == 'onamp':
            texta ='Actual Song: Onam ponnonam poomala pongum...'
            textl ='Youtube link: https://youtu.be/kxw-EO-6Z8I'
        elif seq == 'onamv':
            texta ='Actual Song: Onam vannallo oonjaalittallo...'
            textl ='Youtube link: https://youtu.be/oaSMBo7FYkM'
        elif seq == 'mavel':
            texta ='Actual Song: Maveli nadu vaneedum kalam...'
            textl ='Youtube link: https://youtu.be/4XsmZvalkUY'
        elif seq == 'kutta':
            texta ='Actual Song: Kuttanadan Punjayile...'
            textl ='Youtube link: https://youtu.be/s2R_geXB174'
        elif seq == 'Onapo':
            texta ='Actual Song: Onapove omal pove Ppookudiyan...'
            textl ='Youtube link: https://youtu.be/c0FHgOkqe1U'
        else:
            texta ='Unable to find the actual song!! Please try some other song.'
            textl ='No video available'

        # return texta, textl



    return render_template("result.html", pred=vtext,apred=texta, ypred=textl)
    
if __name__ == "__main__":
    app.run(debug=True)