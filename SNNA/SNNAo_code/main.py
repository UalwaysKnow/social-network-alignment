# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import tensorflow as tf
import linecache
from gan import alignGAN

#########################################################################################
# os setting
#########################################################################################
os.environ["CUDA_VISIBLE_DEVICES"]="0"
if not os.path.exists('output/args'):
	os.makedirs('output/args')
if not os.path.exists('log'):
	os.makedirs('log')

#########################################################################################
# hyper-parameters
#########################################################################################

def parse_args():
	'''
	Parses the align-model arguments.
	'''
	parser = argparse.ArgumentParser(description="Run align GAN.")
	# for input
	#paixu:按节点顺序的deepwalk结果；gailv：三行---序号，度数，概率
	parser.add_argument('--input_netl_nodes', nargs='?', default='../example_embeddings/four-sorted.txt', help='Input left network embeddings path')
	parser.add_argument('--input_netr_nodes', nargs='?', default='../example_embeddings/twit-sorted.txt', help='Input right network embeddings path')
	parser.add_argument('--input_netl_weight', nargs='?', default='../example_embeddings/four-weight.txt', help='Input left sampling weight')
	parser.add_argument('--input_netr_weight', nargs='?', default='../example_embeddings/twit-weight.txt', help='Input right sampling weight')
	parser.add_argument('--target_model', nargs='?', default='deepwalk', help='Input target embedding model')
	#用于助于generator训练的A部分；以下为train_anchors相应的deepwalk结果
	#align：按初始顺序的deepwalk结果，gailv：三行---序号，初始概率，更新后概率
	parser.add_argument('--input_netl_anchors', nargs='?', default='../example_embeddings/four-align-train.txt', help='Input left train anchor embeddings file path')
	parser.add_argument('--input_netr_anchors', nargs='?', default='../example_embeddings/twit-align-train.txt', help='Input right train anchor embeddings file path')
	#parser.add_argument('--input_netl_anchors_weight', nargs='?', default='embeddings/four-gailv-align.txt', help='Input left anchor sampling weight file path')
	#parser.add_argument('--input_netr_anchors_weight', nargs='?', default='embeddings/twit-gailv-align.txt', help='Input right anchor sampling weight file path')
	parser.add_argument('--input_train_anchors', nargs='?', default='../example_embeddings/anchors_train.txt',help='Input train anchor file path')
	parser.add_argument('--input_test_anchors', nargs='?', default='../example_embeddings/anchors_test.txt',help='Input test anchor file path')
	parser.add_argument('--param_theta', nargs='?', default='None', help='Input parameters of theta path')
	parser.add_argument('--param_G', nargs='?', default='None', help='Input parameters of G path')
	parser.add_argument('--param_target_embeddings', nargs='?', default='None', help='Input parameters of generator path')
	# for test
	parser.add_argument('--input_test_netl_embeddings', nargs='?', default='../example_embeddings/four-align-test.txt',help='Input test anchor embeddings file path')
	parser.add_argument('--input_test_netr_embeddings', nargs='?', default='../example_embeddings/twit-align-test.txt',help='Input test anchor embeddings file path')
	# for model
	parser.add_argument('--left_nodes_num', nargs='?', default=5313, help='Input left network nodes num')
	parser.add_argument('--right_nodes_num', nargs='?', default=5120, help='Input right network nodes num')
	parser.add_argument('--train_anchors_num', nargs='?', default=1288,help='Input train anchor num')
	parser.add_argument('--test_anchors_num', nargs='?', default=323,help='Input test anchor num')
	parser.add_argument('--emb_dim', type=int, default=100, help='dim of source embedding.')
	parser.add_argument('--train_batch_size', type=int, default=256, help='Batch size for training.')
	parser.add_argument('--test_batch_size', type=int, default=128, help='Batch size for testing.')
	# for output
	parser.add_argument('--output_dir', nargs='?', default='output/', help='output path')
	parser.add_argument('--output_param_theta', nargs='?', default='output/gan_theta.pkl', help='output parameters of theta path')
	parser.add_argument('--output_param_G', nargs='?', default='output/gan_G.pkl', help='output parameters of G path')
	# for training
	parser.add_argument('--epoch_dis', type=int, default=5, help='the training epoch of discriminator.')
	parser.add_argument('--epoch_gen', type=int, default=1, help='the training epoch of generator.')
	parser.add_argument('--epoch_all', type=int, default=300000, help='the training epoch of model.')

	return parser.parse_args()

def get_batch_fromfile(file, trainset_num, index, size):
	item_real = []
	item_fake = []
	for i in range(index, index + size):
		line = linecache.getline(file, (i%trainset_num) + 1)
		line = line.strip()
		line = line.split()
		item_real.append(int(line[0]))
		item_fake.append(int(line[1]))
	return item_real, item_fake

def test_link(sess,batch_size,model,filename,filelen,begin_index,size,
			  left_anchors_embeddings,right_embeddings,right_anchors_embeddings,left_embeddings):
	read_index = 0
	result_l2r = np.array([0.] * 5)
	result_r2l = np.array([0.] * 5)
	while True:
		if read_index >= size:
			break
		elif read_index + batch_size <= size:
			test_anchor_left, test_anchor_right = get_batch_fromfile(
								filename,filelen,read_index + begin_index,batch_size)
			batch_test_left = left_anchors_embeddings[read_index+begin_index:read_index+begin_index+batch_size]
			batch_test_right = right_anchors_embeddings[read_index + begin_index:read_index + begin_index + batch_size]
			batch = batch_size
		else:
			test_anchor_left, test_anchor_right = get_batch_fromfile(
								filename,filelen,read_index + begin_index,size - read_index)
			batch_test_left = left_anchors_embeddings[read_index + begin_index: begin_index + size]
			batch_test_right = right_anchors_embeddings[read_index + begin_index: begin_index + size]
			batch = size - read_index
		read_index = read_index + batch_size

		feed_dict = {model.node_left:test_anchor_left,model.node_right:test_anchor_right,
					 model.batch_test_left: batch_test_left,model.batch_test_right:batch_test_right,
					 model.right_test_embedding: right_embeddings,model.left_test_embedding:left_embeddings}
		result_l2r_temp,result_r2l_temp= sess.run([model.l2r_acc,model.r2l_acc],feed_dict = feed_dict)

		summary = sess.run(model.test_summary_op, feed_dict=feed_dict)
		result_l2r += np.array(result_l2r_temp) * batch
		result_r2l += np.array(result_r2l_temp) * batch

	result = (result_l2r+result_r2l)/(2*size)
	return result,summary

def main(args):
	#set anchor training set and test set
	print("network edges size: ",args.left_nodes_num,args.right_nodes_num)
	print("train anchor size: ",args.train_anchors_num)
	print("test anchor size: ",args.test_anchors_num)
	#open log files
	test_log = open(args.output_dir + "test_log.txt", 'w')
	train_log = open(args.output_dir + "train_log.txt", 'w')
	generator_loss = open(args.output_dir + "generator_loss.txt", 'w')
	discriminator_loss = open(args.output_dir + "discriminator_loss.txt", 'w')
	#init model
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	model = alignGAN(args, sess)
	summary_writer = tf.summary.FileWriter('./log/model_logs', sess.graph)
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	localtime = time.asctime(time.localtime(time.time()))
	print('Begining Time :', localtime)
	#***********************************training************************************************
	count_all = 0
	best_p_test = 0.0

	#load input embeddings
	#for train
	netl = np.loadtxt(args.input_netl_nodes)
	netr = np.loadtxt(args.input_netr_nodes)
	netl_embeddings = np.delete(netl,0,1)
	netr_embeddings = np.delete(netr,0,1) #两个网络的embeddings结果（顺序：0-n）
	weight1 = np.loadtxt(args.input_netl_weight)
	weight2 = np.loadtxt(args.input_netr_weight)
	weight_l = weight1[2]
	weight_r = weight2[2] #采样概率，一维数组
	left = np.loadtxt(args.input_netl_anchors)
	right = np.loadtxt(args.input_netr_anchors)
	left_embeddings = np.delete(left, 0, 1)
	right_embeddings = np.delete(right, 0, 1)#已对齐的两个网络的embeddings结果（顺序：初始）
	#for test
	left_test = np.loadtxt(args.input_test_netl_embeddings)
	left_test_embeddings = np.delete(left_test, 0, 1)
	right_test = np.loadtxt(args.input_test_netr_embeddings)
	right_test_embeddings = np.delete(right_test, 0, 1)

	#用于求取每个batch的Mt
	netl_anchors,_ = get_batch_fromfile(
		args.input_train_anchors,args.train_anchors_num,0,args.train_anchors_num)
	#start training and testing
	for epoch_all in range(args.epoch_all):
		#discriminator
		for epoch_dis in range(args.epoch_dis):
			sampling1= np.random.choice(args.left_nodes_num, args.train_batch_size, replace=False, p=weight_l)
			netl_embedding = netl_embeddings[sampling1]
			sampling2 = np.random.choice(args.right_nodes_num, args.train_batch_size, replace=False, p=weight_r)
			netr_embedding = netr_embeddings[sampling2]

			feed_dict = {model.left_embedding_D:netl_embedding, model.right_embedding_D: netr_embedding}
			_, dis_loss_curr = sess.run([model.d_optim, model.d_loss],feed_dict=feed_dict)

			summary = sess.run(model.merged_summary_op, feed_dict=feed_dict)
			summary_writer.add_summary(summary, count_all)
			discriminator_loss.write("%s loss: %s\n" % (count_all, dis_loss_curr))
			discriminator_loss.flush()
		count_all += 1
		#genarator,args.epoch_gen==1
		flag = 0
		while flag<1:
			sampling3 = np.random.choice(args.left_nodes_num,args.train_batch_size, replace=False, p=weight_l)
			netl_embedding = netl_embeddings[sampling3]
			left_embedding = []
			right_embedding = []
			Mt = 0
			for i in sampling3:
				if i in netl_anchors:
					Mt+=1
					k = netl_anchors.index(i)
					left_embedding.append(left_embeddings[k])
					right_embedding.append(right_embeddings[k])
			if Mt == 0:
				continue
			else:
				left_embedding = np.array(left_embedding)
				right_embedding = np.array(right_embedding)
				flag += 1
		feed_dict = {model.left_embedding_D: netl_embedding,model.left_embedding_A:left_embedding, model.right_embedding_A: right_embedding}
		_,gen_loss_curr = sess.run([model.g_optim,model.g_loss],feed_dict=feed_dict)

		feed_dict = {model.left_embedding_D: netl_embedding, model.right_embedding_D: netr_embedding}
		summary = sess.run(model.merged_summary_op, feed_dict=feed_dict)
		summary_writer.add_summary(summary, count_all)
		generator_loss.write("%s loss: %s\n" % (count_all, gen_loss_curr))
		generator_loss.flush()
		count_all += 1

		model.save_model_theta(sess, 'output/args/gan_theta_temp.pkl')
		model.save_model_G(sess, 'output/args/gan_G_temp.pkl')

		#test
		if epoch_all % 10 == 0:
			# with open("output/test.log",'a')as f:
			# 	f.write('\n*******************test*****'+str(epoch_all + 1) + '***************************\n')
			result_test,summary = test_link(sess,args.test_batch_size,model,
							   args.input_test_anchors,args.test_anchors_num,
								0,args.test_anchors_num,left_test_embeddings,netr_embeddings,
											right_test_embeddings,netl_embeddings)
			buf = '\t'.join([str(x) for x in result_test])
			test_log.write(str(epoch_all) + "\t"
							  + "test accuracy:\t" + buf + '\n')
			test_log.flush()
			summary_writer.add_summary(summary, count_all)

			result_train,summary = test_link(sess,args.test_batch_size,model,
							   args.input_train_anchors,args.train_anchors_num,
								0,args.train_anchors_num,left_embeddings,netr_embeddings,
											 right_embeddings,netl_embeddings)
			buf = '\t'.join([str(x) for x in result_train])
			train_log.write(str(epoch_all) + "\t"
							  + "train accuracy:\t" + buf + '\n')
			train_log.flush()
			summary_writer.add_summary(summary, count_all)

			if result_test[4] > best_p_test:
				print("best p_test:" + str(epoch_all) +  '\t', result_test)
				print("and the p_train:" + str(epoch_all) +  '\t', result_train)
				best_p_test = result_test[4]
				model.save_model_theta(sess, args.output_param_theta)
				model.save_model_G(sess, args.output_param_G)
		if epoch_all%1000 == 0:
			print(epoch_all)

	#close log
	test_log.close()
	train_log.close()
	discriminator_loss.close()
	generator_loss.close()
	localtime = time.asctime(time.localtime(time.time()))
	print('Endding Time :', localtime)

if __name__ == "__main__":
	args = parse_args()
	main(args)
