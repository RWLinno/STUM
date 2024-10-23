###################################
# 默认参数
###################################


default_dataset = './data/METR-LA'
default_device = 'cpu'
default_save = ''
default_adjdata = './data/sensor_graph/adj_mx.pkl'
default_adjtype = 'doubletransition'
default_pre_train = ''

default_lora = True
default_randomadj = True
default_gcn = True
default_aptonly = True
default_addaptadj = True

default_seq_length = 12
default_batch_size = 64
default_epochs = 5
default_dropout = 0.3

default_in_feats = 2
default_hidden_size = 32
default_num_classes = 207
default_learning_rate = 0.001
default_weight_decay = 0.0001

default_seed = 998244353
default_print_every = 50
default_expid = 1

default_aptonly_r = 10
default_lora_r = 8
default_lora_alpha = 16
default_lora_dropout = 0.05
default_kmerged = 4
default_kconv = 2