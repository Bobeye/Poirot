import numpy as np
import tensorflow as tf
import config as cfg
import datetime
import os


class modelnet(object):

	def __init__(self):
		self.classes = cfg.CLASSES
		self.num_class = len(self.classes)
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.batch_size = cfg.BATCH_SIZE
		self.boxes_per_cell = cfg.BOXES_PER_CELL
		self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)  # output tensor size for the final layer
		self.scale = 1.0 * self.image_size / self.cell_size
		self.boundary1 = self.cell_size * self.cell_size * self.num_class 	# boundary for class probability prediction
		self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell 	# boundary for box prediction
		self.object_scale = cfg.OBJECT_SCALE
		self.noobject_scale = cfg.NOOBJECT_SCALE
		self.class_scale = cfg.CLASS_SCALE
		self.coord_scale = cfg.COORD_SCALE

		self. alpha = cfg.ALPHA
		self.disp_console = cfg.DISP_CONSOLE

		self.collection = []
		self.offset = np.transpose(
					  np.reshape(
					  np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
					  (self.boxes_per_cell, self.cell_size, self.cell_size)),
					  (1, 2, 0))

		# self.weights_file = 'data/weights/YOLO_small.ckpt'
		self.weights_file = None
		self.initial_learning_rate = cfg.LEARNING_RATE
		self.max_iter = cfg.MAX_ITER
		self.summary_iter = cfg.SUMMARY_ITER
		self.decay_steps = cfg.DECAY_STEPS
		self.decay_rate = cfg.DECAY_RATE
		self.staircase = cfg.STAIRCASE
		self.save_iter = cfg.SAVE_ITER
		self.output_dir = cfg.OUTPUT_DIR
		# self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
		self.save_cfg()

		self.build()

		self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
		self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, self.decay_steps,self.decay_rate, self.staircase, name='learning_rate')
		# self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
		self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
		self.averages_op = self.ema.apply(tf.trainable_variables())
		with tf.control_dependencies([self.optimizer]):
			self.train_op = tf.group(self.averages_op)
		self.saver = tf.train.Saver(self.collection, max_to_keep=None)
		self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
		self.sess = tf.Session()
		self.sess.run(tf.initialize_all_variables())
		if self.weights_file is not None:
			print ('Restoring weights from: ' + self.weights_file)
			self.saver.restore(self.sess, self.weights_file)
		else:
			# self.checkpoint = os.path.join(self.output_dir, 'save.ckpt')
			self.checkpoint = 'data/output/last/save.ckpt'
			print ('Restoring weights from last checkpoint')
			self.saver.restore(self.sess, self.checkpoint)



	def build(self):
		if self.disp_console:
			print ('building model graph for training')
		self.x = tf.placeholder('float32', [None, 448, 448, 3])
		self.conv_1 = self.conv_layer(1, self.x, 64, 7, 2)
		self.pool_2 = self.pooling_layer(2, self.conv_1, 2, 2)
		self.conv_3 = self.conv_layer(3, self.pool_2, 192, 3, 1)
		self.pool_4 = self.pooling_layer(4, self.conv_3, 2, 2)
		self.conv_5 = self.conv_layer(5, self.pool_4, 128, 1, 1)
		self.conv_6 = self.conv_layer(6, self.conv_5, 256, 3, 1)
		self.conv_7 = self.conv_layer(7, self.conv_6, 256, 1, 1)
		self.conv_8 = self.conv_layer(8, self.conv_7, 512, 3, 1)
		self.pool_9 = self.pooling_layer(9, self.conv_8, 2, 2)
		self.conv_10 = self.conv_layer(10, self.pool_9, 256, 1, 1)
		self.conv_11 = self.conv_layer(11, self.conv_10, 512, 3, 1)
		self.conv_12 = self.conv_layer(12, self.conv_11, 256, 1, 1)
		self.conv_13 = self.conv_layer(13, self.conv_12, 512, 3, 1)
		self.conv_14 = self.conv_layer(14, self.conv_13, 256, 1, 1)
		self.conv_15 = self.conv_layer(15, self.conv_14, 512, 3, 1)
		self.conv_16 = self.conv_layer(16, self.conv_15, 256, 1, 1)
		self.conv_17 = self.conv_layer(17, self.conv_16, 512, 3, 1)
		self.conv_18 = self.conv_layer(18, self.conv_17, 512, 1, 1)
		self.conv_19 = self.conv_layer(19, self.conv_18, 1024, 3, 1)
		self.pool_20 = self.pooling_layer(20, self.conv_19, 2, 2)
		self.conv_21 = self.conv_layer(21, self.pool_20, 512, 1, 1)
		self.conv_22 = self.conv_layer(22, self.conv_21, 1024, 3, 1)
		self.conv_23 = self.conv_layer(23, self.conv_22, 512, 1, 1)
		self.conv_24 = self.conv_layer(24, self.conv_23, 1024, 3, 1)
		self.conv_25 = self.conv_layer(25, self.conv_24, 1024, 3, 1)
		self.conv_26 = self.conv_layer(26, self.conv_25, 1024, 3, 2)
		self.conv_27 = self.conv_layer(27, self.conv_26, 1024, 3, 1)
		self.conv_28 = self.conv_layer(28, self.conv_27, 1024, 3, 1)
		self.fc_29 = self.fc_layer(29, self.conv_28, 512, flat=True, linear=False)
		self.fc_30 = self.fc_layer(30, self.fc_29, 4096, flat=False, linear=False)
		self.dropout_31 = tf.nn.dropout(self.fc_30, keep_prob=0.5)
		self.fc_32 = self.fc_layer(32, self.dropout_31, self.output_size, flat=False, linear=True)
		self.labels = tf.placeholder('float32', [None, self.cell_size, self.cell_size, 5 + self.num_class])
		self.loss = self.loss_layer(33, self.fc_32, self.labels)
		self.total_loss = self.loss

	def conv_layer(self, idx, inputs, filters, size, stride):
		channels = inputs.get_shape()[3]
		weight = tf.Variable(tf.truncated_normal(
			[size, size, int(channels), filters], stddev=0.1))
		biases = tf.Variable(tf.constant(1.0, shape=[filters]))
		self.collection.append(weight)
		self.collection.append(biases)
		
		pad_size = size // 2
		pad_mat = np.array([[0, 0], [pad_size, pad_size],
							[pad_size, pad_size], [0, 0]])
		inputs_pad = tf.pad(inputs, pad_mat)

		conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1],
							padding='VALID', name=str(idx) + '_conv')
		conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')

		if self.disp_console:
			print ('Layer  %d : Type = Conv, Size = %d * %d, Stride = %d, Filters = %d, Input channels = %d' % (idx, size, size, stride, filters, int(channels)))
		return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

	def pooling_layer(self, idx, inputs, size, stride):
		if self.disp_console:
			print ('Layer  %d : Type = Pool, Size = %d * %d, Stride = %d' % (idx, size, size, stride))
		return tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx) + '_pool')

	def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False):
		input_shape = inputs.get_shape().as_list()
		if flat:
			dim = input_shape[1] * input_shape[2] * input_shape[3]
			inputs_transposed = tf.transpose(inputs, (0, 3, 1, 2))
			inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
		else:
			dim = input_shape[1]
			inputs_processed = inputs
		weight = tf.Variable(tf.truncated_normal([dim, hiddens], stddev=0.1))
		biases = tf.Variable(tf.constant(1.0, shape=[hiddens]))
		self.collection.append(weight)
		self.collection.append(biases)
		if self.disp_console:
			print ('Layer  %d : Type = Full, Hidden = %d, Input dimension = %d, Flat = %d, Activation = %d' % (idx, hiddens, int(dim), int(flat), 1 - int(linear)))
		if linear:
			return tf.add(tf.matmul(inputs_processed, weight), biases, name=str(idx) + '_fc')
		ip = tf.add(tf.matmul(inputs_processed, weight), biases)
		return tf.maximum(self.alpha * ip, ip, name=str(idx) + '_fc')

	def calc_iou(self, boxes1, boxes2):
		boxes1 = tf.pack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
						  boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
						  boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
						  boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
		boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

		boxes2 = tf.pack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
						  boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
						  boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
						  boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2])
		boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

		# calculate the left up point & right down point
		lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
		rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

		# intersection
		intersection = tf.maximum(0.0, rd - lu)
		inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

		# calculate the boxs1 square and boxs2 square
		square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
			(boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
		square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
			(boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

		union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

		return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

	def loss_layer(self, idx, predicts, labels):

		predict_classes = tf.reshape(predicts[:, :self.boundary1],
			[self.batch_size, self.cell_size, self.cell_size, self.num_class])
		predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
			[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
		predict_boxes = tf.reshape(predicts[:, self.boundary2:],
			[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

		response = tf.reshape(labels[:, :, :, 0],
			[self.batch_size, self.cell_size, self.cell_size, 1])
		boxes = tf.reshape(labels[:, :, :, 1:5],
			[self.batch_size, self.cell_size, self.cell_size, 1, 4])
		boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
		classes = labels[:, :, :, 5:]

		offset = tf.constant(self.offset, dtype=tf.float32)
		offset = tf.reshape(offset,
			[1, self.cell_size, self.cell_size, self.boxes_per_cell])
		offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
		predict_boxes_tran = tf.pack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
									  (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
									  tf.square(predict_boxes[:, :, :, :, 2]),
									  tf.square(predict_boxes[:, :, :, :, 3])])
		predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

		iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

		# calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
		object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
		object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response
		noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

		boxes_tran = tf.pack([boxes[:, :, :, :, 0] * self.cell_size - offset,
							  boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
							  tf.sqrt(boxes[:, :, :, :, 2]),
							  tf.sqrt(boxes[:, :, :, :, 3])])
		boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

		# class_loss
		class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(response * (predict_classes - classes)),
			reduction_indices=[1, 2, 3]), name='class_loss') * self.class_scale

		# object_loss
		object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_mask * (predict_scales - iou_predict_truth)),
			reduction_indices=[1, 2, 3]), name='object_loss') * self.object_scale

		# noobject_loss
		noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_mask * predict_scales),
			reduction_indices=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

		# coord_loss
		coord_mask = tf.expand_dims(object_mask, 4)
		boxes_delta = coord_mask * (predict_boxes - boxes_tran)
		coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta),
			reduction_indices=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

		return class_loss + object_loss + noobject_loss + coord_loss
		# return coord_loss




	def save_cfg(self):
		with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
			cfg_dict = cfg.__dict__
			for key in sorted(cfg_dict.keys()):
				if key[0].isupper():
					cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
					f.write(cfg_str)

	def train(self):
		# load training dataset
		from data import data
		train_data = data()
		train_data.load()
		# train preparation     
		max_iter = self.max_iter

		from utils.timer import Timer 
		train_timer = Timer()
		load_timer = Timer()
		last_epoch = 0
		for step in range(1, max_iter+1):
			load_timer.tic()
			X_train, y_train = train_data.get()



			load_timer.toc()
			feed_dict = {self.x: X_train, self.labels: y_train}

			if step % self.summary_iter == 0:
				train_timer.tic()
				loss, _ = self.sess.run([self.loss, self.train_op],feed_dict=feed_dict)
				train_timer.toc()
				log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'' Loss: {}, Speed: {:.3f}s/iter,'' Load: {:.3f}s/iter, Remain: {}') \
						  .format(datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
							train_data.epoch,
							int(step),
							round(self.learning_rate.eval(session=self.sess), 6),
							loss,
							train_timer.average_time,
							load_timer.average_time,
							train_timer.remain(step, self.max_iter))
				print (log_str, end='\r')
				if train_data.epoch != last_epoch:
					print ('')
					last_epoch = train_data.epoch
				# else:
				# 	train_timer.tic()
				# 	_ = self.sess.run([self.train_op],feed_dict=feed_dict)
				# 	train_timer.toc()
			else:
				train_timer.tic()
				_ = self.sess.run(self.train_op, feed_dict=feed_dict)
				train_timer.toc()

			if step % self.save_iter == 0:

				print ('{} Saving checkpoint file to: {}'.format(
									datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
									self.output_dir))
				self.ckpt_file = 'data/output/'+str(step)+'/save.ckpt'
				if not os.path.exists('data/output/'+str(step)):
					os.makedirs('data/output/'+str(step))
				self.saver.save(self.sess, self.ckpt_file)

		# from data import data
		# train_data = data()
		# train_data.load()
		# X_train, y_trian = train_data.get()
		# feed_dict = {self.x: X_train, self.labels: y_trian}
		# self.sess.run(self.train_op, feed_dict=feed_dict)
		# self.saver.save(self.sess, self.ckpt_file)


if __name__ == '__main__':
	net = modelnet()
	net.train()

	print ('finished')