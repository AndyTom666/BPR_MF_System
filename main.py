import os
import time
import argparse
import numpy as np
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import bprmodel
import config
import data_utils
import evaluation

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
	type=float,
	default=0.00075,
	help="learning rate")
parser.add_argument("--lamda",
	type=float,
	default=0.003,
	help="model regularization rate")
parser.add_argument("--batch_size",
	type=int,
	default=4096,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=30,
	help="training epochs")
parser.add_argument("--top_k",
	type=int,
	default=10,
	help="top_k metrics")
parser.add_argument("--factor_num",
	type=int,
	default=36,
	help="latent factor numbers")
parser.add_argument("--num_ng",
	type=int,
	default=5,
	help="sample how many negative items one time for training")
parser.add_argument("--test_num",
	type=int,
	default=100,
	help="sample for testing")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


# prepare the data

train_data, user_ratings, test_data, test, vali_data, vali,  user_num, item_num, train_mat = data_utils.load_data()
# train_data, user_ratings, test_data, test, user_num, item_num, train_mat = data_utils.load_data()

# Generate the DataLoader
train_dataset = data_utils.BPRData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.BPRData(test_data, item_num, train_mat, 0, False)
vali_dataset = data_utils.BPRData(vali_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
vali_loader = data.DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num, shuffle=False, num_workers=0)

# build model
model = bprmodel.BPRModel(user_num, item_num, args.factor_num, args.lr, args.lamda, 0, 0.01)

#put the model into gpu
model.cuda()

# call the model's optimizer
optimizer = model.mf_optim

auc_score = -1
ndcg = -1
recall = -1
precision = -1
map = -1
savepath = 'C:/Users/10922/Desktop/pdfs/AIC/project/BPR_MF3/models/'
PATH = ''


def predict(u, i):
	'''
	The prediction matrix is obtained by inputting the decomposed User representations and item representations
	:param u: decomposed User representations
	:param i: decomposed item representations
	:return: the multiplied prediction matrix
	'''
	return np.inner(u.detach().numpy(), i.detach().numpy())

'''
	this method corrects the result, the user items that the user has generated an interaction are eliminated, 
    and only the data of the interactions that did not generate a user item are retained
    Ensure the recommendation cannot be positive items in the training set
    :param user_ratings: Set of user item dictionaries
    :param predict: one-dimensional prediction matrix
    :param item_no: item number
    :return: the corrected one-dimensional prediction matrix
'''
def handle_prediction(user_ratings, predict, item_no):
    for u in user_ratings.keys():
        for j in user_ratings[u]:
            predict[u * item_no + j] = 0
    return predict


########################### TRAINING #####################################
if __name__=="__main__":
	start_time0 = time.time()
	best_auc = 0
	epoch_list = []
	auc_list = []
	ndcg_list = []
	recall_list = []
	precision_list = []
	map_list = []
	loss_list = []
	epoch_list = [(i + 1) for i in range(args.epochs)]

	# start the training of model
	for epoch in range(args.epochs):
		model.train()
		start_time = time.time()
		# Training phase, this step generates real training samples <u,i,j>
		train_loader.dataset.ng_sample()

		# Use data_loader to fetch the data
		for user, item_i, item_j in train_loader:
			# put the data into the gpu
			user = user.cuda()
			item_i = item_i.cuda()
			item_j = item_j.cuda()
			# clear the gradient to 0
			optimizer.zero_grad()
			# Call the forward method
			loss = model(user, item_i, item_j)
			# get the gradient here
			loss.backward()
			# Gradient back propagation is performed based on the gradient obtained above
			optimizer.step()

		model.eval()
		t2 = time.time()
		# get the prediction matrix(Inner product of the trained matrix)
		predict_matrix = predict(model.U, model.V)
		# prediction, Convert the prediction matrix to a ndarray variable
		predicts = np.mat(predict_matrix).getA().reshape(-1)
		# Ensure the recommendation cannot be positive items in the training set
		predicts = handle_prediction(user_ratings, predicts, config.item_number)
		# Every 5 epochs, it will compare the current validation AUC with the best AUC and decide if we need to store the model
		if (epoch + 1) % 5 == 0:
			auc_score = roc_auc_score(vali, predicts)
			if auc_score > best_auc:
				 best_auc = auc_score
				 PATH = os.path.join(savepath, 'bestbprmodel_vail.pth')
				 torch.save(model.state_dict(), PATH)
			print('the validation AUC is :', auc_score)

		# Every 9 epochs, it will measure the metrics for the test data.
		elif (epoch+1) % 9 == 0:
			auc_score = roc_auc_score(test, predicts)
			if auc_score > best_auc:
				 best_auc = auc_score
			print('The test AUC is:', auc_score)

		else:
			print('================================training phase================================')
			auc_score = roc_auc_score(vali, predicts)
			if auc_score > best_auc:
				 best_auc = auc_score
			print('the validation AUC is :', auc_score)
		# print('================================training phase================================')
		# auc_score = roc_auc_score(test, predict_)
		# print('the validation AUC is :', auc_score)
		# if auc_score > best_auc:
		# 	best_auc = auc_score
		auc_list.append(auc_score)
		# Top-K evaluation
		ndcg, precision, recall, map = evaluation.topK_metrics(vali, predicts, 10, user_num, item_num)
		t1 = time.time()
		print(t1 - t2)
		ndcg_list.append(ndcg)
		recall_list.append(recall)
		precision_list.append(precision)
		map_list.append(map)
		elapsed_time = time.time() - start_time
		print("The time elapse of epoch {:03d}".format(epoch + 1) + " is: " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print('the best AUC is :', best_auc)
	print("Last Epoch's Data Recall -> ", recall_list[-1])
	print("Last Epoch's Data Precision -> ", precision_list[-1])
	print("Last Epoch's Data NDCG -> ", ndcg_list[-1])
	print("Last Epoch's Data MAP -> ", map_list[-1])
	elapsed_time = time.time() - start_time0
	print("The whole process takes : " + time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

	plt.plot(epoch_list, auc_list, label='AUC')
	plt.xlabel('Epoch')
	plt.ylabel('AUC')
	plt.savefig('C:/Users/10922/Desktop/pdfs/AIC/project/BPR_MF3/image/auc11.png')
	plt.legend()
	plt.show()

	plt.plot(epoch_list, recall_list, label='Recall@10')
	plt.plot(epoch_list, precision_list, label='Precision@10')
	plt.plot(epoch_list, ndcg_list, label='NDCG@10')
	plt.plot(epoch_list, map_list, label='MAP')
	plt.xlabel('Epoch')
	plt.ylabel('Metrics')
	plt.savefig('C:/Users/10922/Desktop/pdfs/AIC/project/BPR_MF3/image/metrics11.png')
	plt.legend()
	plt.show()


