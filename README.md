# KBot
KBot is a chatbot implemented in TensorFlow based on the sequence to sequence model with embedding and attention mechanism. The core of chatbot is mainly based on Google Translate TensorFlow model old version, however, you still can find corresponding files under the github history label of
https://github.com/tensorflow/models/commits/master/tutorials/rnn/translate

The origin of this model can be found in the following paper:
http://arxiv.org/abs/1412.7449
http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf
https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

# Dependency
This program is written in *Python*, and build the neural network model with *TensorFlow*. UI interface of this program is written in *Java*.

* Python 3.5 +
* Tensorflow 1.4.0 +
* Java

# Data Preprocessing
The preprocessed data has been included inside *data* folder.
If you want to process the data set from scratch. You can download **Cornell Movie-Dialogs Corpus** using the link provided inside *data* folder.
The data preprocessing can be done by:

```
cd KBot
python data_utils.py
```

Please make sure you have set up data folder inside *KBot* directory. Otherwise, you need to modify your own data set path inside *data_utils.py* file.

# Training
You have two ways to trigger the training mode of chatbot.

* Set model with "train" inside *chatbot_hparams.py*.

```
tf.flags.DEFINE_string("mode", "train", "Chatbot running mode")
```

Once you have set up the train mode, you can simply start up training as follows:

```
cd KBot
python chatbot.py
```

* Set model with "train" inside terminal command.

```
cd KBot
python chatbot.py --mode train
```

# Chat
You also have two ways to trigger the chatting mode of chatbot.

* Set model with "chat" inside *chatbot_hparams.py*.

```
tf.flags.DEFINE_string("mode", "chat", "Chatbot running mode")
```

Once you have set up the train mode, you can simply start up training as follows:

```
cd KBot
python chatbot.py
```

* Set model with "chat" inside terminal command.

```
cd KBot
python chatbot.py --mode chat
```

# UI Interface
You can also fire up a UI interface for chatbot. First, you need to change `./ui/config` file follow the instructions inside the file.

```
ABSOLUTE/PATH/TO/YOUR/PYTHON
ABSOLUTE/PATH/TO/CHATBOT/FOLDER
```

Once you have set up the *config* file. You can start up the UI interface by running `./ui/src/main/TalkFrame.java` inside *Eclipse*.

# Example

![1](media/1.jpg)



