# raw_path for Amazon data
raw_path = 'C:/Users/10922/Desktop/pdfs/AIC/project/BPR_MF3/Data/'

# raw_path for MovieLens100K data
# raw_path = 'C:/Users/10922/Desktop/pdfs/AIC/project/BPR_MF3/Data2/'

# exact path for Amazon data
train_data_path = raw_path + 'train.csv'
vali_data_path = raw_path + 'vali.csv'
test_data_path = raw_path + 'test.csv'

# exact path for MovieLens100K data
# train_data_path = raw_path + 'train.txt'
# test_data_path = raw_path + 'test.txt'

# the actual numbers of the users and items for amazon data
user_number = 3924
item_number = 3058

# the actual numbers of the users and items for movielens data
# user_number = 943
# item_number = 1682

# learning rate
lr = 0.00075
# regularization, lambda
reg = 0.001

# the number of the latent factors
latent_factors = 36

# the size of the user_item_relation matrix
relation_size = user_number * item_number

# the training time
train_count = 4096


