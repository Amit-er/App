import pandas as pd
import matplotlib.pyplot as plt
import pickle
with open("train_history.pkl", "rb") as f:
    history = pickle.load(f)
def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

plt.plot(smooth_curve(history['loss']), label='Smoothed Training Loss')
plt.plot(smooth_curve(history['val_loss']), label='Smoothed Validation Loss')
plt.title("Smoothed Learning Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
