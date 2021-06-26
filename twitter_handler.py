from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import Row
from preprocessing import *
import csv
from joblib import load


def create_model(optimizer='Adam',dropout_rate=0.0, weight_constraint=0):
    print("Optimizer : ",optimizer,",Dropout : ",dropout_rate,"Weight Constraint : ",weight_constraint)
    model = Sequential()
    model.add(Dense(12, input_dim=len(x_train.columns), activation='relu',kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=optimizer)
    return model


def print_prediction(pred):
    return "hate speech " if pred==1 else "not hate speech"

model_name = "modelNN.joblib"
word2vec_model = "word2vec.model"


print("Loading Models...")
model = load(model_name)
word2vec_model = Doc2Vec.load("word2vec.model")
print("Models loaded correctly")


sc = SparkContext()
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 5)

socket_stream = ssc.socketTextStream("127.0.0.1", 5555)
tweets = socket_stream.window(10)


tokenized_tweets = tweets.map(lower)\
                        .map(remove_urls)\
                        .map(remove_tags)\
                        .map(remove_special_characters)\
                        .map(remove_extra_spaces)\
                        .map(remove_numbers)\
                        .map(lambda tweet : (tweet,tokenize(tweet)))\
                        .map(lambda t : (t[0],remove_stop_words(t[1])))\
                        .map(lambda t : (t[0],lemmatize(t[1])))\
                        .map(lambda t : (t[0],remove_duplicate_words(t[1])))\
                        .reduceByKey(lambda t1,t2 : t1)

tokenized_tweets.map(lambda t : (t[0],word2vec(word2vec_model,t[1])))\
                .map(lambda t : (t[0],model.predict([t[1]])))\
                .map(lambda t : '"'+str(t[0])+'"' + " is classified as " + print_prediction(t[1])+"\n")\
                .pprint()

ssc.start()
ssc.awaitTermination()
