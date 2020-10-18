import pickle, numpy as np
from sklearn.metrics import roc_auc_score

# file_dissimilarity_score.pkl contains a dictionary with key as video name and value as the sum of
# dissimilarity scores of all chunks of that video 
with open('file_dissimilarity_score.pkl', 'rb') as handle:
    test_dissimilarity_score = pickle.load(handle)

# file_target.pkl contains a dictionary with key as video name and value as the true target (real/fake)
# for that video
with open('file_target.pkl', 'rb') as handle:
    test_target = pickle.load(handle)

# file_number_of_chunks.pkl contains a dictionary with key as video name and value as the number of chunks 
# in that video
with open('file_number_of_chunks.pkl', 'rb') as handle:
    test_number_of_chunks = pickle.load(handle)

thresholds = [0.6]

for threshold in thresholds:
	y_tar = np.zeros((len(test_target),1))
	y_pred = np.zeros((len(test_target),1))
	count = 0
	for video,score in test_dissimilarity_score.items():
		tar = test_target[video]
		score = test_dissimilarity_score[video]
		num_chunks = test_number_of_chunks[video]
		mean_dissimilarity_score = (score.item()) / num_chunks

		if mean_dissimilarity_score >= threshold:
			# predicted target is fake
			pred = 0
		else:
			# predicted target is real
			pred = 1
		
		y_tar[count,0] = tar
		y_pred[count,0] = pred
		count += 1
	print('Video wise AUC is: '+str(roc_auc_score(y_tar, y_pred)))
