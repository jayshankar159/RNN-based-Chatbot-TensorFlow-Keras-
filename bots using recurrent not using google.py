import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


questions = [
    "hi",
    "hello",
    "how are you",
    "what is your name",
    "bye"
]

answers = [
    "Hello!",
    "Hi there!",
    "I'm good, thanks!",
    "I am ChatBot.",
    "Goodbye!"
]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)

vocab_size = len(tokenizer.word_index) + 1


X = tokenizer.texts_to_sequences(questions)
y = tokenizer.texts_to_sequences(answers)


max_len = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_len, padding='post')
y = pad_sequences(y, maxlen=max_len, padding='post')


from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=vocab_size)


model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len))
model.add(SimpleRNN(50, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, y, epochs=500, verbose=0)


def chatbot_response(text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(seq)
    pred_word_index = np.argmax(pred[0], axis=1)
    words = []
    for idx in pred_word_index:
        for word, i in tokenizer.word_index.items():
            if i == idx:
                words.append(word)
                break
    return ' '.join(words)


print("RNN ChatBot: Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("RNN ChatBot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("RNN ChatBot:", response)
