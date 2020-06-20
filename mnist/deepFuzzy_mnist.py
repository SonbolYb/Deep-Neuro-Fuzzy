'''
The first version of deep fuzzy. We have lots of new operations
https://github.com/openai/weightnorm/blob/master/tensorflow/nn.py
'''
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from setting_mnist import setting
from util_mnist import *

np.set_printoptions(threshold=sys.maxsize)


(x_train_combo, y_train_combo), (x_test, y_test) = mnist.load_data()
print('x_train shape:', x_train_combo.shape)
print(x_train_combo.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train_combo = keras.utils.to_categorical(y_train_combo, setting.n_class)
y_test = keras.utils.to_categorical(y_test, setting.n_class)


x_train=x_train_combo[:-10000]
y_train=y_train_combo[:-10000]

x_val=x_train_combo[-10000:]
y_val=y_train_combo[-10000:]

x_train=x_train/255.0
x_val=x_val/255.0
x_test=x_test/255.0

x_train=np.expand_dims(x_train,axis=-1)
x_val=np.expand_dims(x_val,axis=-1)
x_test=np.expand_dims(x_test,axis=-1)


x_size=x_train.shape[0]-x_train.shape[0]%setting.batch_size
x_train=x_train[:x_size]
y_train=y_train[:x_size]
datagen=augment(x_train,y_train)

x_test_size=x_test.shape[0]-x_test.shape[0]%setting.batch_size
x_test=x_test[:x_test_size]
y_test=y_test[:x_test_size]
datagen_test=augment_test(x_test,y_test)

x_val_size=x_val.shape[0]-x_val.shape[0]%setting.batch_size
x_val=x_val[:x_val_size]
y_val=y_val[:x_val_size]
datagen_val=augment_test(x_val,y_val)

setting.height_data=x_train.shape[1]
setting.width_data=x_train.shape[2]
setting.depth=x_train.shape[3]

print("train info:\t",x_train.shape,y_train.shape)
print("valid info:\t",x_val.shape,y_val.shape)
print("test info:\t",x_test.shape,y_test.shape)

#x_train=x_train[:2]
MDPATH='/home/sonbol/other/models/mnist/newmodel3/models1'
RESULT_PATH='/home/sonbol/other/models/mnist/newmodel3/results.csv'

BORROW='/home/sonbol/other/models/mnist/newmodel3/models15'


#label_train_oh=label_train_oh[:2]


inp=tf.placeholder(tf.float32,[setting.batch_size,setting.height_data,setting.width_data,setting.depth],name='inp')
target=tf.placeholder(tf.int32,[setting.batch_size,setting.n_class],name='target')
lr=tf.placeholder(tf.float32,())


print("*********layer1****************")
inp1=inp#tf.expand_dims(inp,axis=-1)
out1_res=conv_ops(inp1,kernel=3,rules=64,outp=32,stride=1,padding='SAME',name='layer1')


print("*************layer2*************")
out2_res=conv_ops(out1_res,kernel=3,rules=64,outp=32,stride=1,padding='SAME',name='layer2')#conv2d_res(norm_2,out1_res,weight=[kernel4,kernel4,nout3,nout4],bias=[nout4],padding='SAME',strid

print("*************layer3*************")
out3_res=pooling_ops(out2_res,kernel=2,rules=128,outp=32,name='layer3')


print("*************layer4*************")
out4_res=conv_ops(out3_res,kernel=3, rules=128,stride=1,outp=32, padding='SAME',name='layer4')


print("*************layer5*************")
out5_res=conv_ops(out4_res,kernel=3, rules=128,stride=1,outp=32, padding='SAME',name='layer5')

print("*************layer6*************")
out6_res=pooling_ops(out5_res,kernel=2, rules=128,outp=32,name='layer6')


#print("*************layer6*************")
#out6_res=conv_ops(out5_res,kernel=3, rules=128,stride=1,outp=256, padding='SAME',name='layer6')

#print("*************layer7*************")
#out7_res=pooling_ops(out6_res,kernel=2,rules=128,outp=1,name='layer7')

print("*************layer8*************")

out_f=out6_res

out_s=out_f.get_shape().as_list()
out7_flat=tf.reshape(out_f,(out_s[0],-1))

out8=dense_layer(out7_flat,shape_w=[out_s[1]*out_s[2]*out_s[3],512],shape_b=[512],name='dense')
out8_1=tf.nn.relu(out8)

prob = tf.placeholder(tf.float32, name='keep_prob')
#d1=out8_1
d1=tf.nn.dropout(out8_1, prob)

#out9=dense_layer(d1,shape_w=[512,512],shape_b=[512],name='dense2')
#out9_1=tf.nn.relu(out9)

#d2=tf.nn.dropout(out9_1, 0.5)
d2=d1#out9_1
print("*************layer9*************")
out_final=dense_layer(d2,shape_w=[512,setting.n_class],shape_b=[setting.n_class],name='last')


#out_final=tf.squeeze(out10_res)
print("prediction size:\t\t",out_final.get_shape())
pred=tf.nn.softmax(out_final)

err1=tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,logits=out_final)
err=tf.reduce_mean(err1)
print("err_size",err1.get_shape(),err.get_shape())

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(target,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

global_step=tf.train.get_or_create_global_step()

train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(err,global_step=global_step)


saver = tf.train.Saver(max_to_keep=10)
init=tf.global_variables_initializer()

gpu_options=tf.GPUOptions(visible_device_list="0")
#(config=tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	
	ckpt=tf.train.get_checkpoint_state(MDPATH)
	
	if ckpt and ckpt.model_checkpoint_path:
		print(ckpt,ckpt.model_checkpoint_path)
		sess.run(init)
		#saver.restore(sess,ckpt.model_checkpoint_path)
		optimistic_restore(sess,MDPATH)
	else:
		sess.run(init)
		#optimistic_restore(sess,BORROW)
	epoch=0
	while epoch < setting.num_epoch:
		epoch+=1
		loss_tr=0
		acc_tr=0
		num_batch_tr=0
		
		if epoch ==100:
			setting.lr=setting.lr*0.1
		
		if epoch==300:
			setting.lr=setting.lr*0.1
#		
		
		setting.lr=setting.lr*0.9995
		lr1=setting.lr
		
		for feat_tr,label_tr in  datagen.flow(x_train, y_train,batch_size=setting.batch_size,shuffle=True):#datagen.flow(x_train, y_train,batch_size=setting.batch_size,shuffle=True):#batch_data(x_train, y_train):#batch_data(x_train, y_train):#
			num_batch_tr+=1
#			print(num_batch_tr)
			if num_batch_tr > int(len(x_train_combo)/setting.batch_size):
				break

			args={inp:feat_tr,target:label_tr,lr:setting.lr,prob:0.75}
			_,loss,acc,step1,pred_v,err11,logit11=sess.run([train_op,err,accuracy,global_step,correct_pred,err1,out_final],feed_dict=args)
			loss_tr+=loss
			acc_tr+=acc
			
			if epoch <100:
				lr1=lr1*0.9995
			else:
				lr1=lr1*0.99995
			#print(logit11)
			#print(pred_v,err11,pred_v.shape,err11.shape)
			#print(num_batch_tr,acc,loss)

		loss_f_tr=loss_tr/num_batch_tr
		acc_f_tr=acc_tr/num_batch_tr
		#print("train",epoch,acc_f_tr,loss_f_tr)
		
		saver.save(sess, save_path=MDPATH + '/model_'+str(step1)+'.ckpt',global_step=step1,write_state=True)
		
		#validation
		num_batch_va=0
		loss_v=0
		acc_v=0
		for feat_v, label_v in batch_data(x_val, y_val):#batch_data(x_test,y_test):

#			if num_batch_va > int(len(x_val)/setting.batch_size):
#				break
			num_batch_va+=1
			args={inp:feat_v,target:label_v,prob:1.0}
			loss,acc,pred_v=sess.run([err,accuracy,correct_pred],feed_dict=args)
			loss_v+=loss
			acc_v+=acc
		
		loss_f_va=loss_v/num_batch_va
		acc_f_va=acc_v/num_batch_va
		
		#checking
		num_batch_ch=0
		loss_ch=0
		acc_ch=0
		for feat_ch, label_ch in batch_data(x_test, y_test):#batch_data(x_test,y_test):

#			if num_batch_ch > int(len(x_test)/setting.batch_size):
#				break
			num_batch_ch+=1
			args={inp:feat_ch,target:label_ch,prob:1.0}
			loss,acc,pred_ch=sess.run([err,accuracy,correct_pred],feed_dict=args)
			loss_ch+=loss
			acc_ch+=acc

		loss_f_ch=loss_ch/num_batch_ch
		acc_f_ch=acc_ch/num_batch_ch
		
		print("Epoch:",epoch,"   Tr_loss:",loss_f_tr,"   Tr_acc:",acc_f_tr,"   Val_loss:",loss_f_va,"   Val_acc:",acc_f_va,"   ch_loss:",loss_f_ch,"   ch_acc:",acc_f_ch,"   lr",lr1,"   num",num_batch_tr,num_batch_va,num_batch_ch)
		
		with open(RESULT_PATH, 'a') as f2:
			f2.write(str(epoch) + '  ' + str(loss_f_tr)  + '  ' + str(acc_f_tr) + '  ' + str(setting.lr) + '  ' + str(loss_f_va) + '  ' + str(acc_f_va)+ '  ' + str(loss_f_ch) + '  ' + str(acc_f_ch) +'\n')


