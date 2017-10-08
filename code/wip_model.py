import json
import numpy as np
import re
import io
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dropout, Dense, Activation, Concatenate, concatenate, LSTM, Bidirectional, Merge, merge, Input,RepeatVector, dot, Flatten, Reshape
from keras.losses import sparse_categorical_crossentropy
from keras import backend as K

glove_dimensions = 50
max_context_length = 400
max_question_length = 30
max_answer_length = 10
batch_size = 64
slice_size = 6400 # slice of data to be used as one epoch training on full data is expensive

print("Loading Glove models...")
embedding_index = {}
f = open('../download/dwr/glove.6B.50d.txt','r') #50 dims glove file
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs 
f.close()
# print('Found {} word vectors.'.format(len(embedding_index)))

print('Creating index to word dictionary...')
vocab_dat = "../data/squad/vocab.dat"
vocab_opened = open(vocab_dat,'r')
index_to_word = {}
i = 0
for line in vocab_opened:
    word = line.split()
    index_to_word[i] = word[0].lower()
    i = i + 1
vocab_opened.close()

print('Creating word to index dictionary...')
word_to_index = { id: word for word, id in index_to_word.items() } 

# print("testing if this word exists {}".format(index_to_word['guomindang']))

print("Creating embedding matrix...")
num_words = len(index_to_word)+1 
embedding_dim = glove_dimensions
# max_sequence_length = max_context_length
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in index_to_word.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros??
        embedding_matrix[i] = embedding_vector


print("Creating training data stuff...")

# print('Creating answer uuids array...')
# f = "../download/squad/train-v1.1.json"
# with open(f) as train_f:
#     train_instances = json.load(train_f)
# train_data = train_instances["data"]

print('Creating train context ids array...')
f = open("../data/squad/train.ids.context",'r')
train_context_ids = []
for line in f:
    line = line.split()
    train_context_ids.append(line)
f.close()
train_context_ids = np.array(train_context_ids)
train_padded_context_ids = pad_sequences(train_context_ids, maxlen=max_context_length, padding='post')

print('Creating train question ids array...')
f = open("../data/squad/train.ids.question",'r')
train_question_ids = []
for line in f:
    line = line.split()
    train_question_ids.append(line)
f.close()
train_question_ids = np.array(train_question_ids)
train_padded_question_ids = pad_sequences(train_question_ids, maxlen=max_question_length, padding='post')

print('Creating train answer span array...')
f = open("../data/squad/train.span",'r')
train_answer_start = []
train_answer_end = []
for line in f:
    line = line.split()
    start = int(line[0])
    end = int(line[1])
    train_answer_start.append(start)
    train_answer_end.append(end)
f.close()
train_answer_start = np.array(train_answer_start)
train_answer_end = np.array(train_answer_end)

print('Train: Mapping the indexes of first and last words of answer span to one-hot representations...')
train_y_start = []
train_y_end = []
longer_than_400 = 0
for i in range(len(train_context_ids)):
    if train_answer_start[i] > 400:
        longer_than_400 = longer_than_400 + 1
        start = np.zeros(len(train_padded_context_ids[i]))
        start[399] = 2
        end = np.zeros(len(train_padded_context_ids[i]))
        end[399] = 2
    else:
        start = np.zeros(len(train_padded_context_ids[i]))
        start[train_answer_start[i]] = 1
        end = np.zeros(len(train_padded_context_ids[i]))
        end[train_answer_end[i]] = 1
    # start = np.zeros(len(context_ids_array[i]))
    # start[answer_start_array[i]] = 1
    # end = np.zeros(len(context_ids_array[i]))
    # end[answer_end_array[i]] = 1
    train_y_start.append(start)
    train_y_end.append(end)
train_y_start = np.array(train_y_start)
train_y_end = np.array(train_y_end)
# print("longer than 400 are {}".format( longer_than_400 ) )


# print('Creating train answer ids array...')
# f = open("../data/squad/train.answer",'r')
# answer_ids_array = []
# words_not_found_count = 0
# for line in f:
#     line = line.split()
#     lower_case = [word.lower() for word in line]
#     ids = []
#     for word in lower_case:
#         if word in word_to_index:
#             ids.append(word_to_index[word])
#         else:
#             ids.append(9999999999999)
#             words_not_found_count = words_not_found_count + 1
#     # ids = [index_to_word[word] for word in lower_case]
#     answer_ids_array.append(ids)
# f.close()
# padded_answer_ids_train = pad_sequences(answer_ids_array, maxlen=max_answer_length, padding='post')

# print("number of words not found is {}".format(words_not_found_count))


print("Creating testing data stuff...")

print('Creating test context ids array...')
f = open("../data/squad/val.ids.context",'r')
test_context_ids = []
for line in f:
    line = line.split()
    test_context_ids.append(line)
f.close()
test_context_ids = np.array(test_context_ids)
test_padded_context_ids = pad_sequences(test_context_ids, maxlen=max_context_length, padding='post')

print('Creating test question ids array...')
f = open("../data/squad/val.ids.question",'r')
test_question_ids = []
for line in f:
    line = line.split()
    test_question_ids.append(line)
f.close()
test_question_ids = np.array(test_question_ids)
test_padded_question_ids = pad_sequences(test_question_ids, maxlen=max_question_length, padding='post')

print('Creating test answer span array...')
f = open("../data/squad/val.span",'r')
test_answer_start = []
test_answer_end = []
for line in f:
    line = line.split()
    start = int(line[0])
    end = int(line[1])
    test_answer_start.append(start)
    test_answer_end.append(end)
f.close()
test_answer_start = np.array(test_answer_start)
test_answer_end = np.array(test_answer_end)

print('Test: Mapping the indexes of first and last words of answer span to one-hot representations...')
test_y_start = []
test_y_end = []
longer_than_400 = 0
for i in range(len(test_context_ids)):
    if test_answer_start[i] > 400:
        longer_than_400 = longer_than_400 + 1
        start = np.zeros(len(test_padded_context_ids[i]))
        start[399] = 2
        end = np.zeros(len(test_padded_context_ids[i]))
        end[399] = 2
    else:
        start = np.zeros(len(test_padded_context_ids[i]))
        start[test_answer_start[i]] = 1
        end = np.zeros(len(test_padded_context_ids[i]))
        end[test_answer_end[i]] = 1
    # start = np.zeros(len(context_ids_array[i]))
    # start[answer_start_array[i]] = 1
    # end = np.zeros(len(context_ids_array[i]))
    # end[answer_end_array[i]] = 1
    test_y_start.append(start)
    test_y_end.append(end)
test_y_start = np.array(test_y_start)
test_y_end = np.array(test_y_end)
# print("longer than 400 are {}".format( longer_than_400 ) )


print("Creating Neural Network...")

print("Loading context embeddings...") # note that we set trainable = False so as to keep the embeddings fixed

context_input = Input(shape=(max_context_length,), dtype='int32', name='context_input')
encoded_context = Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_context_length, mask_zero=True, trainable=False)(context_input)
encoded_context = Bidirectional(LSTM(embedding_dim, return_sequences=True))(encoded_context)
# encoded_context = Dropout(0.3)(encoded_context)
encoded_context = Bidirectional(LSTM(embedding_dim, return_sequences=False))(encoded_context)


print("Loading question embeddings...")

question_input = Input(shape=(max_question_length,), dtype='int32', name='question_input')
encoded_question = Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_question_length, mask_zero=True, trainable=False)(question_input)
encoded_question = Bidirectional(LSTM(embedding_dim, return_sequences=True))(encoded_question)
# encoded_question = Dropout(0.3)(encoded_question)
encoded_question = Bidirectional(LSTM(embedding_dim, return_sequences=False))(encoded_question)

# print("encoded_question shape {}".format(encoded_question.shape))

# print("Creating attention layer...")
# attention = merge([encoded_question, encoded_context], mode="dot", dot_axes=[1, 1])
# attention = Flatten()(attention)
# # 
# attention = Dense(max_context_length * embedding_dim)(attention)
# attention = Reshape((max_context_length, embedding_dim))(attention)
# # encoded_question_attention = merge([encoded_question, attention], mode="sum")
# encoded_question_attention = Flatten()(encoded_question_attention)
# softmax_output = Dense(2, activation="softmax")(encoded_question_attention)


print('Merging Model...')
# I SHOULD ADD THE ATTENTION LAYER HERE BUT KERAS DOESNT HAVE A DEFAULT ONE
# NOT SURE WHAT THE OUTPUT DIM SHOULD BE

# merged_layer = concatenate([encoded_context, encoded_question], axis=1)
merge1 = concatenate([encoded_context, encoded_question])
# merge1 = Bidirectional(LSTM(embedding_dim))(merge1)
output1 = Dense(max_context_length, activation='softmax')(merge1)
merge2 = concatenate([merge1, output1])
output2 = Dense(max_context_length, activation='softmax')(merge2)



print("Assembling Model...")
model = Model([context_input, question_input], [output1, output2])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# print("X context shape is {}".format(train_padded_context_ids.shape))
# print("X question shape is {}".format(train_padded_question_ids.shape))
# print("Y 1 shape is {}".format(train_y_start.shape))
# print("Y 2 shape is {}".format(train_y_end.shape))

print("Training Model...")
# model.fit([padded_context_ids_train, padded_question_ids_train], [y_start_train, y_end_train], batch_size=batch_size, epochs=10, validation_split=0.2)
# model.fit([train_padded_context_ids, train_padded_question_ids], [train_y_start, train_y_end], batch_size=batch_size, epochs=10, validation_data= ( [test_padded_context_ids, test_padded_question_ids], [test_y_start, test_y_end] ) )
model.fit([ train_padded_context_ids[:slice_size], train_padded_question_ids[:slice_size] ], [ train_y_start[:slice_size], train_y_end[:slice_size] ], batch_size=batch_size, epochs=10, validation_data= ( [ test_padded_context_ids[:slice_size], test_padded_question_ids[:slice_size] ], [ test_y_start[:slice_size], test_y_end[:slice_size] ] ) )

print("Finished training...")
# loss, acc = model.evaluate([padded_context_ids_test, padded_question_ids_test], [y_start_test, y_end_test], batch_size=batch_size, show_accuracy=True)
# loss, acc = model.evaluate([ padded_context_ids_test[:slice_size], padded_question_ids_test[:slice_size] ], [ y_start_test[:slice_size], y_end_test[:slice_size] ], batch_size=batch_size)

# print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

print("Predictions...")

predictions = model.predict([ test_padded_context_ids[:slice_size], test_padded_question_ids[:slice_size] ], batch_size=batch_size)
predictions = np.array(predictions)
predicted_answer_starts = predictions[0]
predicted_answer_ends = predictions[1]

# print("predictions shape is {}".format( predictions.shape ) )
# print("predicted_answer_starts shape is {}".format( predicted_answer_starts.shape ) )
# print("predicted_answer_starts values are {}".format( predicted_answer_starts ) )

# print("1 start values are {}".format( predicted_answer_starts[0] ) )
# print("2 start values are {}".format( predicted_answer_starts[1] ) )
# print("2 start values are {}".format( predicted_answer_starts[2] ) )

num_of_samples = predictions[0].shape[0]

# make class prediction
ansBegin = np.zeros((num_of_samples,), dtype=np.int32)
ansEnd = np.zeros((num_of_samples,),dtype=np.int32) 
for i in range(num_of_samples):
    # ansBegin[i] = predicted_answer_starts[i, :].argmax()
    ansBegin[i] = predicted_answer_starts[i].argmax()
    ansEnd[i] = predicted_answer_ends[i].argmax()
# print(ansBegin.min(), ansBegin.max(), ansEnd.min(), ansEnd.max())

answers = {}

num_answers_whose_start_exceeds_size = 0
num_answers_whose_end_exceeds_size = 0


# print("ansBegin shape is {}".format(ansBegin.shape) )
# print("ansEnd shape is {}".format(ansEnd.shape) )
print("ansBegin first 100 are {}".format(ansBegin[:100]) )
print("ansEnd first 100 are {}".format(ansEnd[:100]) )

for i in range(len(test_padded_context_ids[:slice_size])):
    if ansBegin[i] >= len(test_padded_context_ids[i]):
        answers[i] = "start past"
        num_answers_whose_start_exceeds_size = num_answers_whose_start_exceeds_size + 1
    elif ansEnd[i] >= len(test_padded_context_ids[i]):
        # answers[i] = context_original_test[i][ word_to_index[i][ ansBegin[i] ]: ]
        answers[i] = "end past"
        num_answers_whose_start_exceeds_size  = num_answers_whose_start_exceeds_size  + 1
    else:
        answer_sentence = ""
        for num in range( ansBegin[i], ansEnd[i] + 1 ):
            # print("num {}".format(num))
            context = test_padded_context_ids[i]
            word_index = context[num]
            word = index_to_word[word_index]
            answer_sentence = answer_sentence + " " + word
            answers[i] = answer_sentence

print("past start {}".format(num_answers_whose_start_exceeds_size) )
print("past end {}".format(num_answers_whose_end_exceeds_size) )
print("answers {}".format(answers) )














