import speech_recognition as sr
from mains import Execution

while 1:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        
        try:
            filename =  "draft.txt"
            f = open(filename, "a+")
            
            recognized_text = r.recognize_google(audio)
            print(recognized_text)
            remainder = recognized_text.split()
            while remainder:
                line, remainder = remainder[:12], remainder[12:]
                f.write(' '.join(line))
            if recognized_text == 'stop':
                break
                
        except sr.UnknownValueError:
            print("Unable to Understand the audio")
        except sr.RequestErrror as o:
            print("Error: (o)".format(o))
            
        break  


with open('draft.txt', 'r') as f:
    fil = f.read()
    print(fil)
fil = fil.lower()
if len(fil) == 5 and(fil=='uthra'or fil=='onamn'or fil =='onamp'or fil =='onamv'or fil =='mavel'or fil =='kutta'or fil =='Onapo'):
    
    h = Execution(args)
    h.prepare_data()
        
    # Training the model
    h.train(args)
    sequences = h.sequences
    idx_to_char = h.idx_to_char
    vocab_size = h.vocab_size
    char_to_idx = h.char_to_idx
    # Initialize the model
    model = TextGenerator(args, vocab_size)
    h.generator(model, fil, idx_to_char, char_to_idx,15)
    

else:
    print("Please try ones again")

if fil.lower() == 'uthra':
    print('Actual Song: Uthrada poonilaave vaa....')
    print('Youtube link: https://youtu.be/0QtNxqAZlXQ')
elif fil.lower() == 'onamn':
    print('Actual Song: Onam nilavile pole onam kinavithal pole...')
    print('Youtube link: https://youtu.be/W_lWEvwhlbU')
elif fil.lower()  == 'onamp':
    print('Actual Song: Onam ponnonam poomala pongum...')
    print('Youtube link: https://youtu.be/kxw-EO-6Z8I')
elif fil.lower() == 'onamv':
    print('Actual Song: Onam vannallo oonjaalittallo...')
    print('Youtube link: https://youtu.be/oaSMBo7FYkM')
elif fil.lower() == 'mavel':
    print('Actual Song: Maveli nadu vaneedum kalam...')
    print('Youtube link: https://youtu.be/4XsmZvalkUY')
elif fil.lower() == 'kutta':
    print('Actual Song: Kuttanadan Punjayile...')
    print('Youtube link: https://youtu.be/s2R_geXB174')
elif fil.lower() == 'Onapo':
    print('Actual Song: Onapove omal pove Ppookudiyan...')
    print('Youtube link: https://youtu.be/c0FHgOkqe1U')
else:
    print('Unable to find the actual song!! Please try some other song.')
    
file = open('draft.txt', 'w')
file.close()