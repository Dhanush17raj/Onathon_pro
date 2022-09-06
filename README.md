# Onathon_pro
This is an NLP model that is going to predict onam songs,  when we input the five 5 words of the song. The model is build on Bi_LSTM.we made use of pytorch for implementing the model. The input can be given either as text or voice command. The model will show the predicted song, actual song and the youtube link for the song.

<h1>MODEL ARCHITECTURE</h1>

![arch](https://user-images.githubusercontent.com/76429389/188706132-4762604d-6111-4ee6-b1e6-8b900f8a754b.jpeg)

<h1>Working of the Model</h1>

First we do the text preproceesing i.e convert all letters into lower case, remove punctuations etc. Then we cretate a dictionary of the processed characters and its numerical representation

Then we will slice through the sequence of data, with a fixed size. And the value to be predicted will be the next character of the sequence.Then the sequence is transformed into its numerical form. 
The sequence of characters is then passed through an embedding layer, output of it is given to Bi-LSTM. The outputs of the last hidden layers of the Bi-LSTM's are concatenated and given to the next LSTM. RMSProp is used as the optimizer. The weights of the model are adjusted by backpropagation to achieve best result. Then the output of the lSTM is given to the softmax layer. The outputted value is then decoded to give the text after the inputed sequence.



<h2>Model Prediction</h2>

<h3>i) When text is given</h3>

![prediction](https://user-images.githubusercontent.com/76429389/188712612-62910f90-d414-4da8-940a-b31365a06bb3.jpeg)



<h3>ii) when voice is given</h3>

![accu_vocal](https://user-images.githubusercontent.com/76429389/188716272-7408a64c-8067-4145-b64a-b322be8e1a13.jpeg)



<h2>Website</h2>

![frontend](https://user-images.githubusercontent.com/76429389/188716639-ec9d217f-76f6-4b83-92c6-d5cf653e1655.jpeg)





