import tensorflow as tf
import numpy as np
import sys

#recurrent neural network
class the_rnn():
	def __init__(self, network_size, actuator_size, p_inh, p_con, KD, trial_len, taus):
		global RNDMSD, W_MASK
		self.sizeofrnncell = network_size
		self.T = trial_len
		self.p_inh = p_inh
		self.p_con = p_con
		self.KD = KD
		self.acsz = actuator_size
		self.taus = taus
		self.inh = None
		self.exc = None
		self.w_in_init = None
		self.w_init = None
		self.w_out_init = None
		self.mask = None
		self.taus_gaus_init = None
		self.xlist = []
		self.rlist = []
		self.x_start = None
		self.r_start = None
		self.olist = []

		DeltaT=1 # sampling rate

		self.u = tf.placeholder(tf.float32, [self.acsz, self.T], name='u') #placeholder for inputs

		#make the RNN - receives input and outputs to one fully connected output layer
		self.w_in = tf.get_variable('w_in', initializer = np.zeros([self.sizeofrnncell, self.acsz],dtype=np.float32), dtype=tf.float32, trainable=False)
		self.w = tf.get_variable('w', initializer = np.zeros([self.sizeofrnncell, self.sizeofrnncell],dtype=np.float32), dtype=tf.float32, trainable=True)
		self.w_out = tf.get_variable('w_out', initializer = np.zeros([1, self.sizeofrnncell],dtype=np.float32), dtype=tf.float32, trainable=True)
		self.m = tf.get_variable('m', initializer = np.zeros([self.sizeofrnncell, self.sizeofrnncell],dtype=np.float32), dtype=tf.float32, trainable=False)
		self.b_out = tf.Variable(0, dtype=tf.float32, name='b_out', trainable=True)
		self.taus_gaus = tf.Variable(tf.random_normal([self.sizeofrnncell, 1], seed=RNDMSD), dtype=tf.float32, name='taus_gaus', trainable=True)
		self.x_start = tf.Variable(tf.random_normal([self.sizeofrnncell, 1], seed=RNDMSD), dtype=tf.float32, name='x_start', trainable=False)
		self.r_start = tf.sigmoid(self.x_start)

		for t in range(1, self.T+1):
			if t==1:
				x = self.x_start
				r = self.r_start
				self.xlist.append(x)
				self.rlist.append(r)
			else:
				x = next_x
				r = tf.sigmoid(next_x)
			ww = tf.matmul(self.w, self.m) #weights are stored as all pos; mask must be applied each step to implement Dale's law
			taus_sig = tf.sigmoid(self.taus_gaus)*(self.taus[1] - self.taus[0]) + self.taus[0] #scale tau into the tau range given
			next_x = tf.multiply((1 - DeltaT/taus_sig), x) \
				+ tf.multiply((DeltaT/taus_sig), ((tf.matmul(ww, r)) \
				+ tf.matmul(self.w_in, tf.expand_dims(self.u[:, t-1], 1)))) \
				+ tf.random_normal([self.sizeofrnncell, 1], dtype=tf.float32)/10
			self.xlist.append(next_x) #x is synaptic current variable
			self.rlist.append(tf.sigmoid(next_x)) #rate is sigmoid of x
			tf.assign(self.w, tf.nn.relu(self.w)) #store weights as all positive values - will apply inh/exc mask later
			next_o = tf.matmul(self.w_out, tf.sigmoid(next_x)) + self.b_out
			self.olist.append(next_o)
			self.tauslist = taus_sig
			if t==(self.T-1):
				self.xlistfinal = self.xlist
				self.rlistfinal = self.rlist
				self.olistfinal = self.olist
				self.tauslistfinal = self.tauslist

	def get_weight_initialization(self,RNDMSD=None):
		global W_MASK
		np.random.seed(RNDMSD)

		self.inh = np.random.rand(self.sizeofrnncell, 1) < self.p_inh #boolean label of which is inhibitory
		self.exc = ~self.inh

		#INITIALIZE THE WEIGHTS
		#input weights (input to every neuron)
		self.w_in_init = np.float32(np.random.randn(self.sizeofrnncell, self.acsz))

		#initialize inter-neuron weights of the RNN
		w = np.zeros((self.sizeofrnncell, self.sizeofrnncell), dtype = np.float32) #make an N x N matrix of zeros for the weights
		idx = np.where(np.random.rand(self.sizeofrnncell, self.sizeofrnncell) < self.p_con) #row x column coors of connections
		w[idx[0], idx[1]] = np.random.normal(0, 1.0, len(idx[0])) #Gaussian dist of weights for the non-zero connections
		w = w*15/np.sqrt(self.sizeofrnncell*self.p_con) # scale by a gain to put into chaotic regime
		self.w_init = np.abs(w) #store all weights as positive

		self.w_out_init = np.float32(np.random.randn(1, self.sizeofrnncell)/100) #output weights

		# Mask matrix to implement Dale's Law (create exclusively inhibitory and excitatory populations)
		self.mask = np.eye(self.sizeofrnncell, dtype=np.float32) #nxn identity matrix
		self.mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] = -1 #set -1 for inhibitory neurons

		#synaptic perturbation masks
		if W_MASK=='GKD':
			self.mask = self.mask * self.KD #scale the all weights by KD
		elif W_MASK=='IKD':
			self.mask[np.where(self.inh==True)[0], np.where(self.inh==True)[0]] *= self.KD #scale only inhibitory neuron weights
		elif W_MASK=='EKD':
			self.mask[np.where(self.exc==True)[0], np.where(self.exc==True)[0]] *= self.KD #scale only excitatory neuron weights

		np.random.seed(None)

		return self.w_in_init, self.w_init, self.w_out_init, self.mask

	def get_tau_initialization(self,RNDMSD=None):
		np.random.seed(RNDMSD)
		self.taus_gaus_init = np.random.normal(0.0,1.0,(self.sizeofrnncell, 1))
		np.random.seed(None)

		return self.taus_gaus_init

	def get_initial_xro(self,RNDMSD=None):
		np.random.seed(RNDMSD)
		self.xlist = []
		self.x_start_init = np.random.normal(0.0,1.0,(self.sizeofrnncell, 1))/100
		self.rlist = []
		self.olist = []
		np.random.seed(None)

		return self.x_start_init, self.xlist, self.rlist, self.olist

#class to run network simulation
class sim_run():

	def __init__(self):
		self.bioANN_network = the_rnn(network_size=NETSZ, actuator_size=ACTUATOR_SIZE, p_inh=P_INH, p_con=P_CON, KD=KD, trial_len=TRIAL_LEN, taus=TAU)
		self.default_graph = tf.get_default_graph()

	def test(self,sess,testdatapath,getnumep):
		global RNDMSD
		print ("Starting bioANN for testing")
		with sess.as_default(), sess.graph.as_default():
			ep_num=0
			test_data = []
			while (ep_num <= getnumep):
				#begin episode; for each trial, reinitialize to random weights, random taus, random firing rates
				self.bioANN_network.get_weight_initialization(RNDMSD=RNDMSD)
				self.initialize_weights(sess)
				self.bioANN_network.get_tau_initialization(RNDMSD=RNDMSD) #time constants are drawn from gaussian and then scaled into given range later; creates taus_gaus
				self.initialize_taus(sess)
				self.bioANN_network.get_initial_xro(RNDMSD=RNDMSD)
				self.initialize_xr_start(sess)
				stim_in = np.zeros((1, TRIAL_LEN)) #zero stimulus applied
				
				#give network stim and get out the synaptic current variable and firing rates
				syncur, frates = sess.run([self.bioANN_network.xlistfinal,
					self.bioANN_network.rlistfinal],
					feed_dict={self.bioANN_network.u:stim_in})
				new_episode = [stim_in, syncur, frates]
				test_data.append(new_episode) #add to list of data observations; write data to testdatapath for analysis

				ep_num += 1

	def initialize_weights(self, sess):
		sess.run([tf.assign(self.bioANN_network.w_in, self.bioANN_network.w_in_init),
		tf.assign(self.bioANN_network.w, self.bioANN_network.w_init),
		tf.assign(self.bioANN_network.w_out, self.bioANN_network.w_out_init),
		tf.assign(self.bioANN_network.m, self.bioANN_network.mask),
		tf.assign(self.bioANN_network.b_out, 0)])

	def initialize_taus(self, sess):
		sess.run(tf.assign(self.bioANN_network.taus_gaus, self.bioANN_network.taus_gaus_init))

	def initialize_xr_start(self, sess):
		sess.run(tf.assign(self.bioANN_network.x_start, self.bioANN_network.x_start_init))

######################################################################################################
######################################################MAIN############################################

#main
tf.reset_default_graph()

#first arg is the network size: sys.argv[1]
#second arg is the pcon: sys.argv[2]
#third arg is p_inh: sys.argv[3]
#fourth arg is the subset of synapses to perturb (CTRL, GKD, EKD, IKD): sys.argv[4]
#fifth arg is perturbation amount, KD (0.6 means 0.6*CTRL, i.e. 60% of control aka 40% knockdown): sys.argv[5]

#cmd to run: python3 bioANN.py 512 0.5 0.5 CTRL 0.6 

RNDMSD = None #random seed; set if want to recreate same network configuration multiple times
ACTUATOR_SIZE = 1 #number of inputs into this network; default 1
TRIAL_LEN=200 # trial duration; each time step is 5ms; default 200
TAU = [4,20] #corresponds to 20-1000ms range
NETSZ = int(sys.argv[1]) #number of neurons in the RNN
P_CON = float(sys.argv[2]) # sparsity of connections; probability that any two neurons are connected
P_INH = float(sys.argv[3]) # % neurons in RNN that will be inhibitory
W_MASK = str(sys.argv[4]) #CTRL (no synaptic perturbation), GKD (all synapses), EKD (only excitatory), IKD (only inhibitory)
KD = float(sys.argv[5]) #degree of synaptic knockdown

#create simulation runner
the_sim_runner = sim_run()

#Give directory for output files------------------------------------------------------------------
testdatapath = '/home/bioANN/network_data' #give path and prefix for data to save
getnumep = 100 #number of episodes to get

#---------------------------------------------------------------------------------------------------------------------------

#create the tf.Session
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer()) #initialize variables
	the_sim_runner.test(sess=sess,testdatapath=testdatapath,getnumep=getnumep)



