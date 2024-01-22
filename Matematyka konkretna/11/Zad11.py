import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

text = "Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by non-human animals or by humans"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for i in range(1, len(text.split())):
    n_gram_sequence = text.split()[:i+1]
    input_sequences.append(" ".join(n_gram_sequence))

max_sequence_len = max([len(seq.split()) for seq in input_sequences])
input_sequences = pad_sequences(tokenizer.texts_to_sequences(input_sequences),
                                maxlen=max_sequence_len, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=1)

# Ocenianie dokładności na danych treningowych
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f'Treningowa dokładność: {accuracy * 100:.2f}%')
