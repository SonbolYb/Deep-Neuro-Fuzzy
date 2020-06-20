import tensorflow as tf
from setting_mnist import setting
import numpy as np
import keras

def batch_data(data,label):
	assert len(data)==len(label)
	indices=np.arange(len(data))
	np.random.shuffle(indices)
	
	for start_idx in range(0,len(data)-setting.batch_size+1,setting.batch_size):
		excerpt=indices[start_idx:start_idx+setting.batch_size]
		
		yield data[excerpt],label[excerpt]

def augment(data,label):

	datagen = keras.preprocessing.image.ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		zca_epsilon=1e-06,  # epsilon for ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		# randomly shift images horizontally (fraction of total width)
		width_shift_range=0.1,
		# randomly shift images vertically (fraction of total height)
		height_shift_range=0.1,
		shear_range=0.,  # set range for random shear
		zoom_range=0.0,  # set range for random zoom
		channel_shift_range=0.,  # set range for random channel shifts
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		cval=0.,  # value used for fill_mode = "constant"
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False,  # randomly flip images
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=None,
		# image data format, either "channels_first" or "channels_last"
		data_format=None,
		# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)
	
	datagen.fit(data)
#	for x , y in datagen.flow(data, label,batch_size=128):
#		plt.imshow(x[0])
#		plt.show()
#		print(x.shape)
#	
#	plt.imshow(data[0])
#	plt.show()
	
	return datagen
	
	
def augment_test(data,label):

	datagen = keras.preprocessing.image.ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		zca_epsilon=1e-6,  # epsilon for ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		# randomly shift images horizontally (fraction of total width)
		width_shift_range=0,
		# randomly shift images vertically (fraction of total height)
		height_shift_range=0,
		shear_range=0.,  # set range for random shear
		zoom_range=0.0,  # set range for random zoom
		channel_shift_range=0.,  # set range for random channel shifts
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		cval=0.,  # value used for fill_mode = "constant"
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False,  # randomly flip images
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=None,
		# image data format, either "channels_first" or "channels_last"
		data_format=None,
		# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)
	
	datagen.fit(data)
#	for x , y in datagen.flow(data, label,batch_size=128):
#		plt.imshow(x[0])
#		plt.show()
#		print(x.shape)
#	
#	plt.imshow(data[0])
#	plt.show()
	
	return datagen

def conv2d_mf(inp,weight,bias,padding,stride,name):
	with tf.variable_scope(name):
	
		weight1=tf.get_variable('w',weight,initializer=tf.contrib.layers.xavier_initializer())
		
		#pass weights to Gaussian distributions
		#weight1=tf.clip_by_value(weight,0,1)		#gaussian
		
		bias1=tf.get_variable('b',bias,initializer=tf.constant_initializer(0.01))
		conv1=tf.nn.conv2d(inp,weight1,strides=[1,stride,stride,1],padding=padding)
		logit1=tf.nn.bias_add(conv1,bias1)
		
		#out1=tf.exp(-(tf.square(logit1)))		#gaussian
		out1=tf.clip_by_value(logit1,0,1)
		#out1=tf.nn.sigmoid(logit1)
		#out1=tf.nn.tanh(logit1)
		#out1=tf.nn.relu(logit1)
		#out1=logit1/(1+tf.abs(logit1))
	
		return  out1

def conv2d_depth(inp,kernel,outp,padding,stride,name):
	with tf.variable_scope(name):
		inp_size=inp.get_shape().as_list()
		weight1=tf.get_variable('w',[kernel,kernel,inp_size[-1],1],initializer=tf.contrib.layers.xavier_initializer())
		
		
		#pass weights to Gaussian distributions
		#weight1=tf.clip_by_value(weight,0,1)		#gaussian
		
		bias1=tf.get_variable('b',[inp_size[-1]],initializer=tf.constant_initializer(0.01))
		conv1=tf.nn.depthwise_conv2d(inp,weight1,strides=[1,stride,stride,1],padding=padding)		#we use depth-wise convolution to not combine depth information
		logit1=tf.nn.bias_add(conv1,bias1)
		out1=tf.nn.leaky_relu(logit1)
	
		return  out1

def conv2d_separable(inp,kernel,outp,padding,stride,name):
	with tf.variable_scope(name):
		inp_size=inp.get_shape().as_list()
		weight_depth=tf.get_variable('wd',[kernel,kernel,inp_size[-1],1],initializer=tf.contrib.layers.xavier_initializer())

		weight_point=tf.get_variable('wp',[1,1,inp_size[-1]*1,1],initializer=tf.contrib.layers.xavier_initializer())
		
		
		#pass weights to Gaussian distributions
		#weight1=tf.clip_by_value(weight,0,1)		#gaussian
		
		bias1=tf.get_variable('b',[inp_size[-1]],initializer=tf.constant_initializer(0.01))
		conv1=tf.nn.separable_conv2d(inp,depthwise_filter=weight_depth,pointwise_filter=weight_point,strides=[1,stride,stride,1],padding=padding)		#we use depth-wise convolution to not combine depth information
		logit1=conv1#tf.nn.bias_add(conv1,bias1)
		out1=tf.nn.leaky_relu(logit1)
	
		return  out1

def conv2d(inp,weight,bias,padding,stride,name):
	with tf.variable_scope(name):
	
		weight1=tf.get_variable('w',weight,initializer=tf.contrib.layers.xavier_initializer())
		
		bias1=tf.get_variable('b',bias,initializer=tf.constant_initializer(0.01))
		conv1=tf.nn.conv2d(inp,weight1,strides=[1,stride,stride,1],padding=padding)
		logit1=tf.nn.bias_add(conv1,bias1)
	
		out1=tf.nn.leaky_relu(logit1)
		
	
		return  out1

def norm_ops(inp,name):
	with tf.variable_scope(name):
		inp_size=inp.get_shape().as_list()
		out1=inp/(tf.sqrt(tf.reduce_sum(tf.square(inp),axis=-1,keepdims=True))+setting.epsilon)		#we normalize over layers that is depth here

		return out1

def avg_pooling(inp, kernel,padding,stride,name):
	with tf.variable_scope(name):
		pool=tf.nn.avg_pool(inp,[1,kernel,kernel,1],strides=[1,stride,stride,1],padding=padding)
		return(pool)

def conv2d_res(mf,inp,kernel,outp,stride,padding,name):
	#We apply nonlinearity to mf and to input1
	
	#divide mf to kernel*kernel. #TODO it makes edge effect; find a way to do a clever way of averaging
	#TODO; we don not need to do averaging because we are working with a nonlinear filter
	with tf.variable_scope(name):
		orig_size=inp.get_shape().as_list()
		size=orig_size[2]
		print(mf.get_shape())
		#pad mf to the padding size of the inp
		if padding =='SAME':#size=(size-kernel+2*p)/stride+1 ==> p=((size-1)*stride-size+kernel)/2
			pad_n= int(((size-1)*stride-size+kernel)/2)
			pad=tf.constant([[0,0],[pad_n,pad_n],[pad_n,pad_n],[0,0]])
			mf_p=tf.pad(mf,pad,'CONSTANT')
		else:	mf_p=mf		#TODO: not sure of this; it is used for pooling
		
		print('mf padded',mf_p.get_shape)
		mf_p_size=mf_p.get_shape().as_list()
		size_mf=mf_p_size[-1]
		
		outs=[0]*outp
		
		# if input is multi-dimensional, input should be converted to have depth of 1 with the same kernel for calculating mf. We consider it as nonlinearity for inpt
		inp_nl=conv2d(inp,[kernel,kernel,orig_size[-1],outp],[outp],padding=padding,stride=stride,name='inp_nl')
		
		bias=tf.get_variable('b',[outp],initializer=tf.constant_initializer(0.01))
		
		
		#1*1 filters and number of filters is equal to the outP; using this we are considering firing strength first and then multiplt with input
		mf_combine=conv2d(mf_p,[1,1,size_mf,outp],[outp],padding=padding,stride=stride,name='combine_firing')
		
		print(inp_nl.get_shape(),mf_combine.get_shape())
		
		#we can do either average pooling or a convolution
		#mf_nonlinear=conv2d_depth(mf_combine,kernel,size_mf,padding='VALID',stride=stride, name='mf_padding')
		mf_nonlinear=avg_pooling(mf_combine,kernel,padding='VALID',stride=stride, name='mf_padding')
		
		out=tf.multiply(inp_nl,mf_nonlinear)+bias
		
		return out
		
def conv2d_res_pooling(mf,inp,kernel,outp,stride,padding,name):
	#here we apply nonlinearity to input
	with tf.variable_scope(name):
		orig_size=inp.get_shape().as_list()
		mf_p_size=mf.get_shape().as_list()
		size_mf=mf_p_size[-1]
		
		# if input is multi-dimensional, input should be converted to have depth of 1 with the same kernel for calculating mf. We consider it as nonlinearity for inpt
		inp_nl=conv2d(inp,[kernel,kernel,orig_size[-1],outp],[outp],padding=padding,stride=stride,name='inp_nl')
		
		
		#1*1 filters and number of filters is equal to the outP; using this we are considering firing strength first and then multiplt with input
		mf_combine=conv2d(mf,[1,1,size_mf,outp],[outp],padding='SAME',stride=1,name='combine_firing')
		
		bias=tf.get_variable('b',[outp],initializer=tf.constant_initializer(0.01))
		
		out=tf.multiply(inp_nl,mf_combine)+bias

#		mf_size=mf.get_shape().as_list()
#		#w_mf=tf.get_variable('wmf',[1,1,mf_size[-1],1],initializer=tf.contrib.layers.xavier_initializer())
#		#conv_mf=tf.nn.conv2d(mf,w_mf,strides=[1,1,1,1],padding='SAME')
#		for i in range(outp):
#			inp_size=inp.get_shape().as_list()
#			
#		
#			weight1=tf.get_variable('w'+str(i),[kernel,kernel,inp_size[-1],outp],initializer=tf.contrib.layers.xavier_initializer())
#			bias1=tf.get_variable('b'+str(i),[1],initializer=tf.constant_initializer(0.01))

#			conv1=tf.nn.conv2d(inp,weight1,strides=[1,stride,stride,1],padding=padding)

#			logit=conv1+bias1
#			out_inp=tf.nn.leaky_relu(logit)
#			

#			out=tf.multiply(out_inp,mf)
#			
#			outf=tf.reduce_sum(out,keepdims=True,axis=-1)	
#			outs[i]=outf
#			
#		outfinal=tf.concat(outs,axis=-1)
			
		
		return out

def conv_ops(inp,kernel,rules,outp,stride,padding,name):
	with tf.variable_scope(name):
		inp_size=inp.get_shape().as_list()
		out1=conv2d_mf(inp,weight=[kernel,kernel,inp_size[-1],rules],bias=[rules],padding=padding,stride=stride,name=name+'_conv')
		print("out layer 1 in cov:\t\t",out1.get_shape())

		norm_1=norm_ops(out1,name+'_norm')
		print("out layer 2 in norm:\t\t",norm_1.get_shape())

		out1_res=conv2d_res(norm_1,inp,kernel,outp,stride,padding,name=name+'_res')
		print("out layer 3 in convres:\t\t",out1_res.get_shape())
		
		return out1_res

def pooling_ops(inp,kernel,rules,outp,name):
	with tf.variable_scope(name):
		inp_size=inp.get_shape().as_list()
		out1=conv2d_mf(inp,weight=[kernel,kernel,inp_size[-1],rules],bias=[rules],padding='VALID',stride=2,name=name+'_layer_1_p')
		print("out layer 1 in cov/p:\t\t",out1.get_shape())

		norm_1=norm_ops(out1,name+'_norm')
		print("out layer 2 in norm/p:\t\t",norm_1.get_shape())
		
		out1_res=conv2d_res_pooling(norm_1,inp,kernel,outp,padding='VALID',stride=2,name=name+'_layer_1_res')
		print("out layer 3 in convres/p:\t\t",out1_res.get_shape())
		
		return out1_res

def dense_layer(input1, shape_w,shape_b,name):
	with tf.variable_scope(name):
		w =tf.get_variable('w'+name,shape_w,initializer=tf.contrib.layers.xavier_initializer()) 
		b=tf.get_variable('b'+name,shape_b,initializer=tf.constant_initializer(0.1))
		pred_activation = tf.nn.xw_plus_b(input1,w,b)
		return pred_activation

def optimistic_restore(session, save_file):		#restoring all the variables
	ckpt_o=tf.train.get_checkpoint_state(save_file)
	#print(save_file,ckpt_o.model_checkpoint_path)
	reader = tf.train.NewCheckpointReader(ckpt_o.model_checkpoint_path)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
			if var.name.split(':')[0] in saved_shapes])
	#print(var_names)
	restore_vars = []
	name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
	with tf.variable_scope('', reuse=True):
		for var_name, saved_var_name in var_names:
			curr_var = name2var[saved_var_name]
			var_shape = curr_var.get_shape().as_list()
			if var_shape == saved_shapes[saved_var_name]:
				restore_vars.append(curr_var)
	saver = tf.train.Saver(restore_vars)
	saver.restore(session, ckpt_o.model_checkpoint_path)
