import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from txtge import TextGenerator
from mod import Preprocessing
from mod import parameter_parser

class Execution:

    def __init__(self, args):
        self.file = 'data/dataset.txt'
        self.window = args.window
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        
        self.targets = None
        self.sequences = None
        self.vocab_size = None
        self.char_to_idx = None
        self.idx_to_char = None

    def prepare_data(self):

        # Initialize preprocessor object
        preprocessing = Preprocessing()

        # The 'file' is loaded and split by char
        text = preprocessing.read_dataset(self.file)

        # Given 'text', it is created two dictionaries
        # a dictiornary about: from char to index
        # a dictorionary about: from index to char
        self.char_to_idx, self.idx_to_char = preprocessing.create_dictionary(text)
        
        # Given the 'window', it is created the set of training sentences as well as
        # the set of target chars
        self.sequences, self.targets = preprocessing.build_sequences_target(text, self.char_to_idx, window=self.window)
        
        # Gets the vocabuly size
        self.vocab_size = len(self.char_to_idx)


    def train(self, args):

        # Model initialization
        model = TextGenerator(args, self.vocab_size)
        # Optimizer initialization
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        # Defining number of batches
        num_batches = int(len(self.sequences) / self.batch_size)
        # Set model in training mode
        model.train()
        
        # Training pahse
        for epoch in range(self.num_epochs):
            # Mini batches
            for i in range(num_batches):

                # Batch definition
                try:
                    x_batch = self.sequences[i * self.batch_size : (i + 1) * self.batch_size]
                    y_batch = self.targets[i * self.batch_size : (i + 1) * self.batch_size]
                except:
                    x_batch = self.sequences[i * self.batch_size :]
                    y_batch = self.targets[i * self.batch_size :]

                # Convert numpy array into torch tensors
                x = torch.from_numpy(x_batch).type(torch.LongTensor)
                y = torch.from_numpy(y_batch).type(torch.LongTensor)
                
                # Feed the model
                y_pred = model(x)
                # Loss calculation
                loss = F.cross_entropy(y_pred, y.squeeze())
                # Clean gradients
                optimizer.zero_grad()
                # Calculate gradientes
                loss.backward()
                # Updated parameters

                optimizer.step()
        
            print("Epoch: %d,  loss: %.5f " % (epoch, loss.item()))


        torch.save(model.state_dict(), 'weights/textGenerator_model.pt')


    @staticmethod
    def generator(model, seq, idx_to_char, char_to_idx, n_chars):
        
        # Set the model in evalulation mode
        model.eval()
        
        # Define the softmax function
        softmax = nn.Softmax(dim=1)
        
        a = (char_to_idx[value] for value in seq)
        x = (tuple(a))
        #print(x)

        # The inputted text is converted to a tuple and is given to pattern
        pattern = x
        
        # By making use of the dictionaries, it is printed the pattern
        print("\nText entered:")
        print(''.join([idx_to_char[value] for value in pattern]))
        
        # In full_prediction we will save the complete prediction
        # tuple is converted to numpy array
        pattern = np.asarray(pattern)
        full_prediction = pattern
        # The prediction starts, it is going to be predicted a given
      # number of characters
        for i in range(n_chars):
        
            # The numpy patterns is transformed into a tesor-type and reshaped
            pattern = torch.from_numpy(pattern).type(torch.LongTensor)
            pattern = pattern.view(1,-1)
            
            # Make a prediction given the pattern
            prediction = model(pattern)
            # It is applied the softmax function to the predicted tensor
            prediction = softmax(prediction)
            
            # The prediction tensor is transformed into a numpy array
            prediction = prediction.squeeze().detach().numpy()
            # It is taken the idx with the highest probability
            arg_max = np.argmax(prediction)
            
            # The current pattern tensor is transformed into numpy array
            pattern = pattern.squeeze().detach().numpy()
            # The window is sliced 1 character to the right
            pattern = pattern[1:]
            # The new pattern is composed by the "old" pattern + the predicted character
            pattern = np.append(pattern, arg_max)
            
            # The full prediction is saved
            full_prediction = np.append(full_prediction, arg_max)
            
        print("\nSong prediction: ")
        print(''.join([idx_to_char[value] for value in full_prediction]) )

if __name__ == '__main__':

    args = parameter_parser()

    # If you already have the trained weights
    if args.load_model == True:
        if os.path.exists(args.model):
            
            # Load and prepare sequences
            execution = Execution(args)
            execution.prepare_data()
            
            seq = execution.seq
            idx_to_char = execution.idx_to_char
            vocab_size = execution.vocab_size
            char_to_idx = execution.char_to_idx

            # Initialize the model
            model = TextGenerator(args, vocab_size)
            
            # Load weights
            model.load_state_dict(torch.load('weights/textGenerator_model.pt'))
            seq = input("Enter the first 5 letters of the song: ")
            # Text generator
            
            execution.generator(model,seq , idx_to_char, char_to_idx,15)


            

    # If you will train the model 		
    else:
        # Load and preprare the sequences
        execution = Execution(args)
        execution.prepare_data()
        
        # Training the model
        execution.train(args)
        sequences = execution.sequences
        idx_to_char = execution.idx_to_char
        vocab_size = execution.vocab_size
        char_to_idx = execution.char_to_idx
        # Initialize the model
        model = TextGenerator(args, vocab_size)
        # Load weights
        model.load_state_dict(torch.load('weights/textGenerator_model.pt'))
        seq = input("Enter the first 5 letters of the song: ")
        # Text generator
        execution.generator(model, seq, idx_to_char, char_to_idx,15)
if seq.lower() == 'uthra':
    print('Actual Song: Uthrada poonilaave vaa....')
    print('Youtube link: https://youtu.be/0QtNxqAZlXQ')
elif seq.lower() == 'onamn':
    print('Actual Song: Onam nilavile pole onam kinavithal pole...')
    print('Youtube link: https://youtu.be/W_lWEvwhlbU')
elif seq.lower()  == 'onamp':
    print('Actual Song: Onam ponnonam poomala pongum...')
    print('Youtube link: https://youtu.be/kxw-EO-6Z8I')
elif seq.lower() == 'onamv':
    print('Actual Song: Onam vannallo oonjaalittallo...')
    print('Youtube link: https://youtu.be/oaSMBo7FYkM')
elif seq.lower() == 'mavel':
    print('Actual Song: Maveli nadu vaneedum kalam...')
    print('Youtube link: https://youtu.be/4XsmZvalkUY')
elif seq.lower() == 'kutta':
    print('Actual Song: Kuttanadan Punjayile...')
    print('Youtube link: https://youtu.be/s2R_geXB174')
elif seq.lower() == 'Onapo':
    print('Actual Song: Onapove omal pove Ppookudiyan...')
    print('Youtube link: https://youtu.be/c0FHgOkqe1U')
else:
    print('Unable to find the actual song!! Please try some other song.')

