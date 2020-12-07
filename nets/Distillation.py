import tensorflow as tf
from nets import SVP
import numpy as np

tcl = tf.contrib.layers

def Plane_mapping(X):
    with tf.variable_scope('Euc'):
        D = X.get_shape().as_list()[-1]
        center = tf.ones([1,D])/np.sqrt(D)
        scale = 1/tf.square(tf.cos(tf.acos(tf.reduce_sum(X,1,keepdims=True)/np.sqrt(D))/2))
        return (X+center)*scale - 2*center

def inter_cosine_sim(X):
    with tf.variable_scope('ICS'):
        X = tf.nn.l2_normalize(X,-1)
        return tf.matmul(X, X, transpose_b = True)
        
def EKI(student_feature_maps, teacher_feature_maps):
    with tf.variable_scope('EKI'):
        with tf.contrib.framework.arg_scope([tcl.fully_connected], weights_regularizer=None,
                                            variables_collections = [tf.GraphKeys.GLOBAL_VARIABLES,'MHA']):
            with tf.contrib.framework.arg_scope([tcl.batch_norm], activation_fn=None, param_regularizers = None,
                                                variables_collections=[tf.GraphKeys.GLOBAL_VARIABLES,'MHA']):
                GNN_losses0 = []
                GNN_losses1 = []
                GNN_losses2 = []
                T_F = S_F = D_F = None
                
                for i, (sfm, tfm) in enumerate(zip(student_feature_maps, teacher_feature_maps)):
                    tz = tfm.get_shape().as_list()
                    sz = sfm.get_shape().as_list()
                    D_B = tz[-1]
                    
                    with tf.variable_scope('Stacked_PCA%d'%i):
                        with tf.variable_scope('Teacher_PCA'):
                            Sigma_T, U_T, V_T = SVP.SVD_eid(tfm, 1, name = 'TSVD')
                            sign = tf.sign(tf.reduce_max(V_T,1,keepdims=True) + tf.reduce_min(V_T,1,keepdims=True))
                            V_T *= sign; U_T *= sign

                            T_B = tf.reshape(V_T,[-1,D_B])
                            T_B = Plane_mapping(T_B)
                            G_B, T_B, mean_V, P_B = PCA_Graph(T_B)

                        with tf.variable_scope('Student_PCA'):
                            sfm = tf.reshape(sfm, [-1,sz[1]*sz[2],sz[3]])
                            V_S = tf.nn.l2_normalize(tf.matmul(sfm, U_T, transpose_a = True),1)

                            S_B = tf.reshape(V_S,[-1,D_B])
                            S_B = Plane_mapping(S_B)
                            S_B = tf.matmul(S_B - mean_V, P_B)
                            
                    with tf.variable_scope('EKI_module%d'%i):
                        if i > 0:
                            with tf.variable_scope('MPNN'):
                                D = D_B//2
                                num_iter = 2
                                G_T, _   = MPNN(T_B, T_F, D, num_iter, 'MPNN')
                                tf.add_to_collection('MHA_loss', kld_loss(G_B, G_T))

                                # Update Graph Knowledge
                                G_T, M_T = MPNN(T_B, T_F, D, num_iter, 'MPNN', False, True)
                                G_S, M_S = MPNN(S_B, S_F, D, num_iter, 'MPNN', False, True)
                                
                                GNN_losses0.append(tf.reduce_mean(tf.reduce_sum(tf.abs(M_S-M_T),-1)))
                                GNN_losses1.append(kld_loss(G_T, G_S))

                            with tf.variable_scope('MHA'):
                                M_T = tf.reduce_mean(M_T,2)*num_iter
                                GNN_losses2.append(Attention_knowledge(T_F, T_B, S_F, S_B, num_head = 8, extrinsic = M_T))
                                
                    T_F, S_F, D_F = T_B, S_B, D_B
                
                tf.add_to_collection('dist', tf.add_n(GNN_losses0))
                tf.add_to_collection('dist', tf.add_n(GNN_losses1))
                tf.add_to_collection('dist', tf.add_n(GNN_losses2))
            
def MPNN(X_B, X_F, D, num_iter, scope, is_training = True, reuse = False):
    def Vertices(X_B, X_F, D, scope, reuse, is_training):
        with tf.variable_scope(scope, reuse = reuse):
            return X_B
            
    def Edge(X, D, scope, reuse, is_training):
        with tf.variable_scope(scope, reuse = reuse):
            X = tcl.batch_norm(tcl.fully_connected(X,D,scope='Efc'), scope='Ebn', is_training = is_training)
            X = tf.nn.l2_normalize(X,-1)
            return tf.expand_dims(X,1)*tf.expand_dims(X,0)*D

    def Message(H, E, D, scope, reuse, is_training):
        with tf.variable_scope(scope, reuse = reuse):
            with tf.contrib.framework.arg_scope([tcl.fully_connected, tcl.batch_norm], trainable = not(reuse), reuse = reuse):
                H = tf.concat([H,tf.reduce_mean(E,-1,keepdims=True)],-1)
                M = tcl.batch_norm(tcl.fully_connected(H,D,scope='Mfc'), scope='Mbn', is_training = is_training)
                G = tcl.batch_norm(tcl.fully_connected(H,D,scope='Gfc'), scope='Gbn', is_training = is_training, activation_fn = tf.nn.sigmoid)
                return M*G

    def Update(E, M, scope):
        with tf.variable_scope(scope):
            return E + M
            
    def Readout(E, scope):
        with tf.variable_scope(scope):
            return tf.reduce_mean(E,-1)
            
    with tf.variable_scope(scope):
        H = Vertices(X_B, X_F, D, 'Vertices', is_training = is_training, reuse = reuse)
        E = Edge(X_F, D, 'Edge', is_training = is_training, reuse = reuse)
        M_ = []
        H = tf.expand_dims(H,1)-tf.expand_dims(H,0)
        
        for i in range(num_iter):
            M = Message(H, E, D, 'Message', is_training = is_training, reuse = reuse if i == 0 else True)
            E = Update(E, M, 'Update')
            M_.append(M)
        
        R = Readout(E, 'Readout')
        return R, tf.concat(M_,-1)

def Incremental_PCA(X, D):
    with tf.variable_scope('Incremental_PCA'):
        rate = .9
        moving_M = tf.get_variable('M', [1,D], tf.float32, trainable = False, initializer = tf.zeros_initializer(),
                                   collections = [tf.GraphKeys.GLOBAL_VARIABLES,'MHA'])
        moving_S = tf.get_variable('S', [1,D//2], tf.float32, trainable = False, initializer = tf.zeros_initializer(),
                                   collections = [tf.GraphKeys.GLOBAL_VARIABLES,'MHA'])
        moving_V = tf.get_variable('V', [D,D//2], tf.float32, trainable = False, initializer = tf.zeros_initializer(),
                                   collections = [tf.GraphKeys.GLOBAL_VARIABLES,'MHA'])
        SV = tf.transpose(moving_S * moving_V,[1,0])
        
        current_mean = tf.reduce_mean(X, 0, keep_dims=True)
        X_ = X - current_mean

        init = tf.equal(tf.reduce_sum(SV), 0.)
        moving_M_ = tf.cond(init, lambda : current_mean, lambda : moving_M)
        X_        = tf.concat([X_, SV, moving_M_ - current_mean], 0)
        
        with tf.device('CPU'):
            s, U, V = tf.linalg.svd(X_)
        V *= tf.sign(tf.reduce_max(V,0,keepdims=True) + tf.reduce_min(V,0,keepdims=True))
        
        V = tf.slice(V, [0,0],[-1,D//2])
        s = tf.slice(s, [0],[D//2])
        
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(moving_M, moving_M*(1-rate) + current_mean*rate ))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(moving_S, tf.expand_dims(s,0) ))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(moving_V, V ))
        
        return tf.stop_gradient(moving_M) , tf.stop_gradient(moving_V)

def PCA_Graph(X):
    with tf.variable_scope('PCA_Graph'):
        D = X.get_shape().as_list()[-1]
        mean, V = Incremental_PCA(X, D)

        P = tf.matmul(X-mean, V)
        G = inter_cosine_sim(P)
        return G, P, mean, V

def Attention_knowledge(T_F, T_B, S_F, S_B, num_head, extrinsic = None):
    num_head -= extrinsic is not None
    with tf.variable_scope('Attention_Knowledge'):
        Df = T_F.get_shape().as_list()[-1]
        Db = T_B.get_shape().as_list()[-1]
        D = (Df+Db)//2
        G_T = Attention_head(T_B, T_F, D, num_head, 'Attention', is_training = True)
        T_B_ = Estimator(T_F, G_T, Db, num_head, 'Estimator', True, extrinsic)
        tf.add_to_collection('MHA_loss', tf.reduce_mean(1-tf.reduce_sum(T_B*T_B_, -1)) )
        
        G_T = Attention_head(T_B, T_F, D, num_head, 'Attention', reuse = True)
        G_S = Attention_head(S_B, S_F, D, num_head, 'Attention', reuse = True)
        return kld_loss(tf.tanh(G_S), tf.tanh(G_T))

def Attention_head(K, Q, D, num_head, name, is_training = False, reuse = False):
    with tf.variable_scope(name):
        with tf.contrib.framework.arg_scope([tcl.fully_connected], trainable = not(reuse), reuse = reuse):
            with tf.contrib.framework.arg_scope([tcl.batch_norm], trainable = not(reuse), is_training = is_training, reuse =  reuse):
                B = tf.squeeze(tf.slice(tf.shape(K),[0],[1]))
                
                X_sender   = tcl.batch_norm(tcl.fully_connected(K, D*num_head, scope = 'Sfc'), scope = 'Sbn')
                X_sender   = tf.reshape(X_sender,   [B, D, num_head])

                X_receiver = tcl.batch_norm(tcl.fully_connected(Q, D*num_head, scope = 'Rfc', reuse = reuse), scope = 'Rbn')
                X_receiver = tf.reshape(X_receiver, [B, D, num_head])
                
                X_sender   = tf.transpose(X_sender,  [2,0,1])
                X_receiver = tf.transpose(X_receiver,[2,1,0])
                X_ah = tf.matmul(X_sender, X_receiver)
                return X_ah

def Estimator(X, G, Dy, num_head, name, is_training, extrinsic = None):
    with tf.variable_scope(name):
        with tf.contrib.framework.arg_scope([tcl.fully_connected], trainable = True):
            with tf.contrib.framework.arg_scope([tcl.batch_norm], activation_fn=tf.nn.relu, trainable = True):
                B = tf.squeeze(tf.slice(tf.shape(G),[1],[1]))
                G = tf.nn.softmax(G)
                
                if extrinsic != None:
                    G = tf.concat([G,tf.expand_dims(tf.stop_gradient(extrinsic),0)],0)
                    num_head += 1

                G = drop_head(G, [num_head, B, 1])
                G = tf.reshape(G, [num_head*B, B])
                
                Dx = X.get_shape().as_list()[-1]
                D = (Dx+Dy)//2
                
                X = tcl.batch_norm(tcl.fully_connected(X, D, scope = 'fc0'), scope = 'bn0', is_training = is_training)
                X = tf.reshape(tf.matmul(G, X), [num_head, B, D])
                X = tf.reshape(tf.transpose(X,[1,0,2]),[B,D*num_head])
                
                X = tcl.fully_connected(X, Dy, biases_initializer=tf.zeros_initializer(), scope = 'fc1')
                X = tf.nn.l2_normalize(X, -1)
                return X
    
def drop_head(G, shape):
    with tf.variable_scope('Drop'):
        noise = tf.random.normal(shape)
        G *= tf.where(noise - tf.reduce_mean(noise, 0, keepdims=True) > 0, tf.ones_like(noise), tf.zeros_like(noise))
        return G

def kld_loss(X, Y):
    with tf.variable_scope('KLD'):
        return tf.reduce_sum( tf.nn.softmax(X)*(tf.nn.log_softmax(X)-tf.nn.log_softmax(Y)) )

