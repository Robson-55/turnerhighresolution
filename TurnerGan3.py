# -*- coding: utf-8 -*-
"""
turner-GAN
"""

#Si no funciona el entorno, mirar aquí: https://medium.com/@pushkarmandot/installing-tensorflow-theano-and-keras-in-spyder-84de7eb0f0df
#slim = tf.contrib.slim 
#https://medium.freecodecamp.org/how-ai-can-learn-to-generate-pictures-of-cats-ba692cb6eae4 

import os
import tensorflow as tf
import numpy as np
import random
import scipy.misc
from utils2 import *



HEIGHT, WIDTH, CHANNEL = 512, 512, 3
BATCH_SIZE = 64
EPOCH = 5000
version = 'newTurner2'
newPoke_path = './' + version

def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
 
def process_data():   
   
    current_dir = os.getcwd()
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir, 'PaintingsOfTurner')
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string) #se convierten el vector de imágenes en un tensor
    
    images_queue = tf.train.slice_input_producer(
                                        [all_images]) #divide el tensor en particiones
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1) #ajusta el brillo a un valor concreto aleatorio
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()
    
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)
									#está mezclando los tensores de imágenes [images] de cada epoch y dará como resultado image_batch, que es el lote de imagenes a tratar en esa epoch. 
									#capacity: maximo numero de elementos en la cola(pila). min_after_dequeue:minimo numero de elementos en la pila despues del dequeue(retirar elementos de la pila).
	

    return iamges_batch, num_images

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, 8192], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[8192], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')  																	     #matmul = matrix multiplication. flat_conv1=inputxw1 + b1 ----> dimensiones:(1x100)matmul(100x8096) + (1x8096)  =  (1x8096)
         #Convolution, bias, activation, repeat! 
        conv1 = tf.reshape(flat_conv1, shape=[-1, 4, 4, 512], name='conv1')  																	 #el -1 es porque desconocemos la dimension (aunque es 1x8096) y queremos que numpy la averigue. Esta transformando un tensor de 1x8096 en otro tensor de 4x4x512
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')       # adds a batch normalization layer. input:conv1/// is_training: Whether or not the layer is in training mode. In training mode it would accumulate the statistics of the moments/// epsilon: small float added to the variance to avoid dividing by zero/// decay=disminucion de la media
																																				 #Normalize the activations of the previous layer at each batch, i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
		  #se aplica ReLu
        # 8*8*256
        #Convolution, bias, activation, repeat! 
        act1 = tf.nn.relu(bn1, name='act1')
        conv2 = tf.layers.conv2d_transpose(act1, 256, kernel_size=[5, 5], strides=[2, 2], padding="SAME",           								 #input=act1; filters=c8=256 -> es la dimension del espacio de salida; kernel_size = dimensiones de los filtros/// strides=salto por el que se multiplica
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),        								 #inicializador del kernel		
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, 128, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, 64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, 32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
		# 128*128*16
		conv6 = tf.layers.conv2d_transpose(act5, 16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.relu(bn6, name='act6')
		# 256*256*8
		conv7 = tf.layers.conv2d_transpose(act6, 8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv7')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn7')
        act7 = tf.nn.relu(bn5, name='act7')
        
        #512*512*3
        conv8 = tf.layers.conv2d_transpose(act7, 3, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv8')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act8 = tf.nn.tanh(conv8, name='act8')
        return act8


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, 64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1') #se aplica Leaky Relu en lugar de Relu
         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, 128, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, 256, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, 512, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')																						 #el -1 es porque desconocemos la dimension y queremos que numpy la averigue
      
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')																				    #logits es el vector de predicciones que genera el discriminador antes de ser normalizado con sigmoid function (entre 1 y 0).
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits # acted_out


def train():
    random_dim = 100
    
    with tf.variable_scope('input'):
        #real and fake image placeholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image') 	#un placeholder(marcador de posicion) consiste en reservar espacio en memoria para una variable, pero no asignar el valor a esa variable aun. Se asigna en la sesión, de manera que la variable tendrá esos valores durante esa sesión. El placeholder es el método para introducir datos en los gráficos computacionales (tensores) de Tensorflow.
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input') 				#este placeholder será de tipo float con dimensiones (cualquiera(None) x random_dim) y con ese nombre
        is_train = tf.placeholder(tf.bool, name='is_train') 												#placeholder de tipo booleano: 1 o 0
    
    # wgan
    fake_image = generator(random_input, random_dim, is_train)
    
    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)
    
    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  									# This optimizes the discriminator. tf.reduce_mean-->hace la media de todos los valores. d_loss es la funcion de error del discriminador y se basa en la distancia entre el resultado verdadero y el falso.
    g_loss = -tf.reduce_mean(fake_result)  																	# This optimizes the generator. la funcion de error del generador, que se basa en la media del resultado falso generado
            

    t_vars = tf.trainable_variables()   																	#Devuelve todas las variables creadas con trainable=True
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)   			#Root Mean Squared Propagation: es una version adaptada del stochastic gradient descent. Va minimizando d_loss. Es un método de optimización adaptativa del learning rate.
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)   			#va minimizando g_loss 
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars] 									#las d_vars(variables del discriminador) inferiores a -0.01 serán 0.01 y las superiores a 0.01 serán 0.01

    
    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()  																#se obtienen las imágenes estandarizadas. samples_num es el numero de imagenes que se tratan, el numero de muestras. image_batch
    
    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()  
    sess.run(tf.global_variables_initializer())  															#antes de usar variables estas deben de ser inicializadas
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")   														#guarda las variables 
    ckpt = tf.train.latest_checkpoint('./model/' + version)  												#encuentra el nombre de del archivo del ultimo checkpoint guardado. './model/' es el directorio donde se encuentra el checkpoint
    saver.restore(sess, save_path) 																			#restaura las variables que se han guardado previamente
    coord = tf.train.Coordinator()
	 
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   										#threading implica multiples tareas ejecutandose asincronicamente. El metodo de threading es el queuing y ayuda para introducir los datos en el training set. Cuando Tensorflow está leyendo la entrada de datos necesita mantener muchas queues operando simultaneamente, por lo que el Coordinator  ayuda a gestionarlo. Multithreaded queues are a powerful and widely used mechanism supporting asynchronous computation. https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/how_tos/threading_and_queues/index.md 
																											# https://www.tensorflow.org/guide/graphs explica como funciona tensorflow y sus graficos y sesiones
																											#Like everything in TensorFlow, a queue is a node in a TensorFlow graph. It's a stateful node, like a variable: other nodes can modify its content. In particular, nodes can enqueue new items in to the queue, or dequeue existing items from the queue
	
	             
    print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('total training sample num:%d' % samples_num)
    print('start training...')
    for i in range(EPOCH):
        print("Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print(j)
            d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32) 				#el vector de ruido aleatorio normalizado de tamaño 64 x 100
            for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                #wgan clip weights
                sess.run(d_clip)  
                
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})

            # Update the generator
            for k in range(g_iters):
                
                _, gLoss = sess.run([trainer_g, g_loss],
                                    feed_dict={random_input: train_noise, is_train: True})

            # print 'train:[%d/%d],d_loss:%f,g_loss:%f' % (i, j, dLoss, gLoss)
            
        # save check point every 500 epoch
        if i%100 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        if i%10 == 0:
            # save images
            if not os.path.exists(newPoke_path):
                os.makedirs(newPoke_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32) 				#vector de ruido de 64 x 100
            imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})
            # imgtest = imgtest * 255.0
            # imgtest.astype(np.uint8)
            save_images(imgtest, [5,5] ,newPoke_path + '/epoch' + str(i) + '.jpg')
            
            print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))
			
    coord.request_stop()   																								#requests that threads should stop
    coord.join(threads)    																								#waits until the specified threads have stopped.


# def test():
    # random_dim = 100
    # with tf.variable_scope('input'):
        # real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        # random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        # is_train = tf.placeholder(tf.bool, name='is_train')
    
    # # wgan
    # fake_image = generator(random_input, random_dim, is_train)
    # real_result = discriminator(real_image, is_train)
    # fake_result = discriminator(fake_image, is_train, reuse=True)
    # sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())
    # variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    # print(variables_to_restore)
    # saver = tf.train.Saver(variables_to_restore) 
    # ckpt = tf.train.latest_checkpoint('./model/' + version)
    # saver.restore(sess, ckpt)

   #tf.reset_default_graph()
if __name__ == "__main__":
    train()
    # test()





































