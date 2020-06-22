
####################-----Attention----------######################

def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))

    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
        
    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]),output_shape=(l, nv, nv))([q,k])# l, nv, nv
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]),  output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)
    
    out = Add()([out, q1])

    out = Dense(dout, activation = "relu")(out)

    return  Model(inputs=[q1,k1,v1], outputs=out)



##########################----MODEL-----##########################

class model_rgb2hs():

        def __init__(self, image_size):
                self.image_size = image_size

        def MODEL_hs(self):
                inp = Input((None, None,3))
                C1 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inp)                

                C2 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C1)
                C3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C2)
                
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(C3)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x)
                x = BatchNormalization()(x)
                x1 = concatenate([x, C3],axis=3)
                #2
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x1)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x)
                x = BatchNormalization()(x)
                x2 = concatenate([x, x1],axis=3)
                #3
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x2)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x)
                x = BatchNormalization()(x)
                x3 = concatenate([x, x2],axis=3)
                
                x = Reshape([6*6,64])(C3)    
                att = MultiHeadsAttModel(l=6*6, d=64 , dv=8*3, dout=32, nv = 8 )
                x = att([x,x,x])
                x = Reshape([6,6,32])(x)   
                model = Model(inp, x)
                return model
