from tensorflow.python.keras.layers.merge import concatenate
from astropy.cosmology import Planck15 as cosmo
import tensorflow as tf

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus( -x -5.0 ) 

def xsquared(x):
    return (x/4)**2


datatype = tf.float32


kernel_size = 5

class Conv_GRU(tf.keras.Model):
    def __init__(self , num_features):
        super(Conv_GRU, self).__init__()
        num_filters = num_features
        self.conv_1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='sigmoid',padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='sigmoid',padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='tanh',padding='same')
    def call(self, inputs, state):
        stacked_input = tf.concat([inputs , state], axis=3)
        z = self.conv_1(stacked_input)
        r = self.conv_2(stacked_input)
        r_state = tf.multiply(r , state)
        stacked_r_state = tf.concat([inputs , r_state], axis=3)
        update_info = self.conv_3(stacked_r_state)
        new_state = tf.multiply( 1-z , state) + tf.multiply(z , update_info)
        return new_state , new_state

initializer = tf.initializers.random_normal( stddev=0.06)
kernal_reg_amp = 0.0
bias_reg_amp = 0.0
kernel_size = 6

class Model(tf.keras.Model):
    def __init__(self,num_cell_features):
        super(Model, self).__init__()
        self.num_gru_features = num_cell_features/2
        num_filt_emb1_1 = self.num_gru_features
        num_filt_emb1_2 = self.num_gru_features
        num_filt_emb2 = self.num_gru_features
        num_filt_emb3_1 = self.num_gru_features
        num_filt_emb3_2 = self.num_gru_features
        self.conv1_1 = tf.keras.layers.Conv2D(filters = num_filt_emb1_1, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv1_2 = tf.keras.layers.Conv2D(filters = num_filt_emb1_2, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv1_3 = tf.keras.layers.Conv2D(filters = num_filt_emb1_2, kernel_size=[kernel_size,kernel_size], strides=2, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters = num_filt_emb2, kernel_size=[5,5], strides=1, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_1 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_1, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_2 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_2, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_3 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_2, kernel_size=[kernel_size,kernel_size], strides=2, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv4 = tf.keras.layers.Conv2D(filters = 1, kernel_size=[5,5], strides=1, activation='linear',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.gru1 = Conv_GRU(self.num_gru_features)
        self.gru2 = Conv_GRU(self.num_gru_features)
    def call(self, inputs, state, grad):
        stacked_input = tf.concat([inputs , grad], axis=3)
        xt_1E = self.conv1_1(stacked_input)
        xt_1E = self.conv1_2(xt_1E)
        xt_1E = self.conv1_3(xt_1E)
        ht_11 , ht_12 = tf.split(state, 2, axis=3)
        gru_1_out,_ = self.gru1( xt_1E ,ht_11)
        gru_1_outE = self.conv2(gru_1_out)
        gru_2_out,_ = self.gru2( gru_1_outE ,ht_12)
        delta_xt_1 = self.conv3_1(gru_2_out)
        delta_xt_1 = self.conv3_2(delta_xt_1)
        delta_xt_1 = self.conv3_3(delta_xt_1)
        delta_xt = self.conv4(delta_xt_1)
        xt = delta_xt + inputs
        ht = tf.concat([gru_1_out , gru_2_out], axis=3)
        return xt, ht

    
    
class GRU_COMPONENT(tf.keras.Model):
    def __init__(self,num_cell_features):
        super(GRU_COMPONENT, self).__init__()
        self.kernel_size = 5
        self.num_gru_features = num_cell_features/2
        self.conv1 = tf.keras.layers.Conv2D(filters = self.num_gru_features, kernel_size=self.kernel_size, strides=1, activation='relu',padding='same')
        self.gru1 = Conv_GRU(self.num_gru_features)
        self.gru2 = Conv_GRU(self.num_gru_features)
    def call(self, inputs, state):
        ht_11 , ht_12 = tf.split(state, 2, axis=3)
        gru_1_out,_ = self.gru1( inputs ,ht_11)
        gru_1_outE = self.conv1(gru_1_out)
        gru_2_out,_ = self.gru2( gru_1_outE ,ht_12)
        ht = tf.concat([gru_1_out , gru_2_out], axis=3)
        xt = gru_2_out
        return xt, ht

def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))




    
    
class RIM_CELL(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, num_steps ,num_pixels, state_size , input_size=None, activation=tf.tanh):
        self.num_pixels = num_pixels
        self.num_steps = num_steps
        self._num_units = state_size
        
        self.single_RIM_state_size = state_size/2
        self.gru_state_size = state_size/4
        self.gru_state_pixel_downsampled = 16*2
        self._activation = activation
        self.model_1 = Model(self.single_RIM_state_size)
            
        self.batch_size = batch_size
        self.initial_output_state()

    def initial_output_state(self):
        self.inputs_1 = tf.zeros(shape=(self.batch_size , self.num_pixels , self.num_pixels , 1))
        self.state_1 = tf.zeros(shape=(self.batch_size,  self.num_pixels/self.gru_state_pixel_downsampled, self.num_pixels/self.gru_state_pixel_downsampled , self.single_RIM_state_size ))


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs_1, state_1, grad_1, scope=None):
        xt_1, ht_1 = self.model_1(inputs_1, state_1 , grad_1)
        return xt_1, ht_1

    def forward_pass(self, data):

        if (data.shape[0] != self.batch_size):
           self.batch_size = data.shape[0]
           self.initial_output_state()


        output_series_1 = []
      

        with tf.GradientTape() as g:
            g.watch(self.inputs_1)
            
            y = log_likelihood(data,physical_model(self.inputs_1),noise_rms)
        grads = g.gradient(y, self.inputs_1 )

        output_1, state_1 = self.__call__(self.inputs_1, self.state_1 , grads)
        output_series_1.append(output_1)
        

        for current_step in range(self.num_steps-1):
            with tf.GradientTape() as g:
                g.watch(output_1)
                
                y = log_likelihood(data,physical_model(output_1),noise_rms)
            grads = g.gradient(y, output_1 )


            output_1, state_1 = self.__call__(output_1, state_1 , grads)
            output_series_1.append(output_1)
            
        final_log_L = log_likelihood(data,physical_model(output_1),noise_rms)
        return output_series_1 , final_log_L

    def cost_function( self, data,labels_x_1):
        output_series_1 , final_log_L = self.forward_pass(data)
        return tf.reduce_mean(tf.square(output_series_1 - labels_x_1)), output_series_1 ,output_series_1[-1].numpy() 

class Data_Generator(object):

    def __init__(self,train_batch_size=1,test_batch_size=1, impix_side = 192, im_size=128):
        self.im_size = im_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.impix_side = impix_side

    def gen_source(self):
        Im = np.ones((self.im_size,self.im_size))
        return Im

    def draw_im(self,train_or_test):
        if (train_or_test=="train"):
            np.random.seed(seed=None)
            num_samples = self.train_batch_size
        if (train_or_test=="test"):
            np.random.seed(seed=136)
            num_samples = self.test_batch_size
        
        self.IM_tr = np.zeros((num_samples,self.im_size,self.im_size,1))
        for i in range(num_samples):
            
            #parameters for im, here it's just an example  
            x = np.random.uniform(low=-1.0, high=1.)
            
            if (train_or_test=="train"):
                self.IM_tr[i,:,:,0] = self.gen_source()

            if (train_or_test=="test"):
                self.IM_ts[i,:,:,0] = self.gen_source()
 
        return



class Phys_Mod(object):

    def __init__(self, numpix_side = 192):
        self.numpix_side = numpix_side
                
    def physical_model(self, IM):
        
        sin_model = tf.tensordot(IM.ravel(),sin_tensor,axes=1)
        cos_model = tf.tensordot(IM.ravel(),cos_tensor,axes=1)
        vis2s = tf.abs(sin_model**2+cos_model**2)  
        phases = tf.angle(tf.complex(data_cos,data_sin))
        cps = tf.tensordot(bs_tensor,phases,axes=1)
        return tf.concat([vis2s,cps],axis=0)

    def simulate_noisy_image(self, IM, noise_rms=0.1):
        IM = self.physical_model(IM)
        noise = tf.random_normal(tf.shape(IM),mean=0.0,stddev = noise_rms,dtype=T)
        IM = IM + noise
        self.noise_rms = noise_rms
        return IM

    def log_likelihood(Data,Model,noise_rms):
        #logL = 0.5 * tf.reduce_mean(tf.reduce_mean((Data - Model)**2, axis=2 ), axis=1 )
        logL = 0.5 * tf.math.reduce_mean(tf.square(D - M), axis=1 )/ noise_sig**2
        return logL

