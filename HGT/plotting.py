import matplotlib.pyplot as plt

epochs = list(range(10))  # Adjust the range based on the number of epochs

train_acc = [0.016560030109145654, 0.04528012279355334, 0.05624764061910155, 0.06064073226544622, 0.07936507936507936, 0.05518018018018018, 0.058091286307053944, 0.05635948210205636, 0.08073817762399077, 0.06603053435114503]
valid_acc = [0.006230529595015576, 0.029239766081871343, 0.04040404040404041, 0.04984423676012461, 0.01744186046511628, 0.0136986301369863, 0.038461538461538464, 0.03184713375796178, 0.05373134328358209, 0.046012269938650305]
test_acc = [0.011111111111111112, 0.04, 0.052980132450331126, 0.03934426229508197, 0.04672897196261682, 0.017123287671232876, 0.028070175438596492, 0.00974025974025974, 0.038461538461538464, 0.03642384105960265]

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, valid_acc, label='Validation Accuracy', marker='o')
plt.plot(epochs, test_acc, label='Test Accuracy', marker='o')

plt.title('FP16 Quantization Training, Validation, and Test Accuracies Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to be up to 1
plt.legend()
plt.grid(True)
plt.show()