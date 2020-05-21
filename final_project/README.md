# Transfer Learning for text reviews with Word2Vec and LSTM
#### By Jonathan Ellis for CS 344 at Calvin University

## Vision
This goal of this project is to design a machine learning model that will take a runner's text analysis of their race and
quantify their attitude and effort. This will allow coaches and team leaders to get a better idea of the mentality of
each runner and the team as a whole, helping them develop a better plan to help the team improve after a race. The
inspiration for this project came from my coaches on the Calvin University Cross Country team who emphasize the importance of focusing on attitude and effort in
competition since these two factors are some of the only controllable elements in running. Because of this emphasis, our team developed
a [post-race analysis website](calvinpostrace.herokuapp.com) so every runner could write about how their race went and rate themselves for attitude and
effort on a scale from 1-10. The data from this website are used to train the model. However, since there are so few samples
from the website, a model is first trained on an Amazon Reviews dataset before being applied to the post-race data so that the
post-race model has some insights before training even begins.

## Running the Project
The two main code modules are amazon.py and postrace.py located in resources/. The amazon.py module trains the Amazon
reviews model and the postrace.py trains the post-race model. There are a few things you need to run the model. The first
is the MONGO_URL located in postrace.py. I have redacted the url to keep the database secure. If you need the url, email
me at jde27@students.calvin.edu and I will provide it for you. Once you have the url, set the MONGO_URL variable to the
url I gave you and the module will be ready to run. It will load the amazon.h5 model from the resources folder. 

Additionally you will have to install the necessary modules in your Python environment. These include:
- numpy
- pandas
- keras
- tensorflow
- gensim
- sklearn
- matplotlib
- seaborn
- nltk
- pymongo

The dataset files for amazon.py are located on [Google Drive](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)
in amazon_review_full_csv.zip. Once you have the dataset files, the amazon.py module is also ready to run, but it will
not create new word vectors, it will simply load the ones stored in the resources folder. If you wish to create new word
vectors, then locate line 129 in amazon.py where create_embedding_matrix() is called and switch load to False.

Both of this files can be run using the command `python <<module-name>>` using Python 3.6.


