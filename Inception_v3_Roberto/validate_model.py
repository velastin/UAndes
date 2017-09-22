import data_feed
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.models import load_model
import numpy as np
from tqdm import tqdm

model = load_model('D:\\PAMELA-UANDES\\Inception_V3\\best_model.hdf5')

batch_size = 32
val_feed = data_feed.feed_data('test', samples=5000, batch_size=batch_size)

y_true = np.array([np.argmax(y[1]) for y in val_feed.data]) # onehot format
y_pred = []

keep_going = True
for _ in tqdm(range(len(y_true)/batch_size+1)):
	if not keep_going:
		break
	batch = val_feed.next_batch()
	X = batch[0]

	preds = model.predict(X)
	for pred in preds:
		if len(y_pred) == len(y_true):
			keep_going = False
			break
		y_pred.append(np.argmax(pred))

y_pred = np.array(y_pred)

print("")
print("Validation accuracy for {} images: {}".format(len(y_pred), accuracy_score(y_true, y_pred)))
print("")
print("Confusion matrix: \n{}".format(confusion_matrix(y_true, y_pred)))
print("")
print(classification_report(y_true, y_pred, target_names=['no head', 'head']))