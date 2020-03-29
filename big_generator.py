import pandas as pd
import numpy as np
import keras

random_sampling_size = 600
fraction_per_case = 0.2
true_false_ratio = 1

class file_generator():
    "Generating model"
    def __init__(self, window, time_it, feature,
                 true_X_time, true_X_static,true_y_total,
                 false_X_time, false_X_static, false_y_total,
                 list_time_Xn, list_time_yn, list_time_nstatic, 
                 random_sampling_size, fraction_per_case, true_false_ratio, repeat):
        self.window = window
        self.time_it = time_it
        self.feature = feature
        self.true_X_time = true_X_time
        self.true_X_static = true_X_static
        self.true_y_total = true_y_total
        self.false_X_time = false_X_time
        self.false_X_static = false_X_static
        self.false_y_total = false_y_total
        self.list_time_Xn = list_time_Xn
        self.list_time_yn = list_time_yn
        self.list_time_nstatic = list_time_nstatic
        self.random_sampling_size = random_sampling_size
        self.fraction_per_case = fraction_per_case
        self.true_false_ratio = true_false_ratio
        self.repeat = repeat


    def get_data(self, index):
        Xn_time = []
        Xn_static = []
        yn_time = []

        random_sampling = np.random.choice(len(self.list_time_Xn), size= self.random_sampling_size)
        ## negative data gathering
        for i in random_sampling:
            X_time_n = self.list_time_Xn[i]
            y_time_n = self.list_time_yn[i]
            Xn, yn = self.time_series_generator(X_time_n,y_time_n)
            time_static_data = self.list_time_nstatic[i]

            for n in range(len(Xn)):
                Xn_time.append(Xn[n])
                Xn_static.append(time_static_data)
                yn_time.append(yn[n])

        np_Xn_time = np.array(Xn_time, dtype="float32").reshape(-1,self.window,self.feature)
        np_Xn_static = np.array(Xn_static, dtype="float32").reshape(-1,38)
        np_yn_time = np.array(yn_time, dtype="float32").reshape(-1,)

        batch_false_list = np.random.choice(len(self.false_X_time), size=int(np.floor(len(self.true_X_time)*self.true_false_ratio)))

        batch_false_X = self.false_X_time[batch_false_list]
        batch_false_Xs = self.false_X_static[batch_false_list]
        batch_false_y = self.false_y_total[batch_false_list]

        repeat_true_X=np.repeat(np.concatenate([batch_false_X, self.true_X_time]), self.repeat, axis=0)
        repeat_true_Xs=np.repeat(np.concatenate([batch_false_Xs, self.true_X_static]), self.repeat, axis=0)
        repeat_true_y=np.repeat(np.concatenate([batch_false_y, self.true_y_total]), self.repeat, axis=0)
        
        batch_X=np.concatenate([np_Xn_time,repeat_true_X],axis=0)
        batch_static_X=np.concatenate([np_Xn_static,repeat_true_Xs], axis=0)
        batch_y=np.concatenate([np_yn_time,repeat_true_y],axis=0)

        batch_X, batch_static_X, batch_y = self.shuffling(batch_X, batch_static_X, batch_y)

        return batch_X, batch_static_X, batch_y


    def time_series_generator(self, x,y):
        Xn=[]
        yn=[]
        size_x = len(x)-self.time_it
        random_selection = np.random.choice(size_x, size = int(np.floor(size_x*self.fraction_per_case)))
        for n in random_selection:
            if n+1>self.window:
                X_train=x[n+1-self.window:n+1]
            else:
                X_train=x[0:n+1]
                X_train=np.pad(X_train, mode='constant', pad_width=((0,self.window-X_train.shape[0]),(0,0)),\
                               constant_values=-5)

            Xn.append(X_train)
            y_train=y[n+self.time_it]
            yn.append(y_train)

        return Xn, yn

    def shuffling(self, a, b, c):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p], c[p]

class file_generator_valid():
    "Generating model"
    def __init__(self, window, time_it, feature,
                 list_time_Xt, list_time_yt, list_time_tstatic,
                 list_time_Xn, list_time_yn, list_time_nstatic):
        self.window = window
        self.time_it = time_it
        self.feature = feature
        self.list_time_Xt = list_time_Xt
        self.list_time_yt = list_time_yt
        self.list_time_tstatic = list_time_tstatic
        self.list_time_Xn = list_time_Xn
        self.list_time_yn = list_time_yn
        self.list_time_nstatic = list_time_nstatic


    def get_data(self):
        Xt_time = []
        Xt_static = []
        yt_time = []

        Xn_time = []
        Xn_static = []
        yn_time = []


        ## positive data gathering
        for i in range(len(self.list_time_Xt)):
            Xt_time_n = self.list_time_Xt[i]
            yt_time_n = self.list_time_yt[i]
            Xt, yt = self.time_series_generator(Xt_time_n,yt_time_n)
            time_static_data = self.list_time_tstatic[i]

            for n in range(len(Xt)):
                Xt_time.append(Xt[n])
                Xt_static.append(time_static_data)
                yt_time.append(yt[n])

        positive_index = [i for i,result in enumerate(yt_time) if result==1]
        negative_index = [i for i,result in enumerate(yt_time) if result==0]

        positive_x = [Xt_time[i] for i in positive_index]
        positive_x_static = [Xt_static[i] for i in positive_index]
        positive_y = [yt_time[i] for i in positive_index]

        negative_x = [Xt_time[i] for i in negative_index]
        negative_x_static = [Xt_static[i] for i in negative_index]
        negative_y = [yt_time[i] for i in negative_index]


        ## negative data gathering
        for i in range(len(self.list_time_Xn)):
            X_time_n = self.list_time_Xn[i]
            y_time_n = self.list_time_yn[i]
            Xn, yn = self.time_series_generator(X_time_n,y_time_n)
            time_static_data = self.list_time_nstatic[i]

            for n in range(len(Xn)):
                Xn_time.append(Xn[n])
                Xn_static.append(time_static_data)
                yn_time.append(yn[n])

        true_X_time = np.array(positive_x, dtype="float32").reshape(-1,self.window,self.feature)
        true_X_static = np.array(positive_x_static, dtype="float32").reshape(-1,38)
        true_y_total = np.array(positive_y, dtype="float32").reshape(-1,)
        false_X_time = np.array(negative_x, dtype="float32").reshape(-1,self.window,self.feature)
        false_X_static = np.array(negative_x_static, dtype="float32").reshape(-1,38)
        false_y_total = np.array(negative_y, dtype="float32").reshape(-1,)
        Xn_time = np.array(Xn_time, dtype="float32").reshape(-1,self.window,self.feature)
        Xn_static = np.array(Xn_static, dtype="float32").reshape(-1,38)
        yn_time = np.array(yn_time, dtype="float32").reshape(-1,)

        batch_X=np.concatenate([Xn_time,false_X_time, true_X_time],axis=0)
        batch_static_X=np.concatenate([Xn_static,false_X_static, true_X_static], axis=0)
        batch_y=np.concatenate([yn_time,false_y_total, true_y_total],axis=0)

        return batch_X, batch_static_X, batch_y


    def time_series_generator(self, x,y):
        Xn=[]
        yn=[]
        for n in range(len(x)-self.time_it):
            if n+1>self.window:
                X_train=x[n+1-self.window:n+1]
            else:
                X_train=x[0:n+1]
                X_train=np.pad(X_train, mode='constant', pad_width=((0,self.window-X_train.shape[0]),(0,0)),\
                               constant_values=-5)

            Xn.append(X_train)
            y_train=y[n+self.time_it]
            yn.append(y_train)

        return Xn, yn


class file_generator_cul():
    "Generating model"
    def __init__(self, window, time_it, feature,
                 list_time_Xcul, list_time_ycul, list_time_culstatic):
        self.window = window
        self.time_it = time_it
        self.feature = feature
        self.list_time_Xn = list_time_Xcul
        self.list_time_yn = list_time_ycul
        self.list_time_nstatic = list_time_culstatic


    def get_data(self):
        Xt_time = []
        Xt_static = []
        yt_time = []

        Xn_time = []
        Xn_static = []
        yn_time = []


        ## negative data gathering
        for i in range(len(self.list_time_Xn)):
            X_time_n = self.list_time_Xn[i]
            y_time_n = self.list_time_yn[i]
            Xn, yn = self.time_series_generator(X_time_n,y_time_n)
            time_static_data = self.list_time_nstatic[i]

            for n in range(len(Xn)):
                Xn_time.append(Xn[n])
                Xn_static.append(time_static_data)
                yn_time.append(yn[n])

        Xn_time = np.array(Xn_time).reshape(-1,self.window,self.feature)
        Xn_static = np.array(Xn_static).reshape(-1,38)
        yn_time = np.array(yn_time).reshape(-1,)

        return Xn_time, Xn_static, yn_time


    def time_series_generator(self, x,y):
        Xn=[]
        yn=[]

        for n in range(len(x)-self.time_it):
            if n+1>self.window:
                X_train=x[n+1-self.window:n+1]
            else:
                X_train=x[0:n+1]
                X_train=np.pad(X_train, mode='constant', pad_width=((0,self.window-X_train.shape[0]),(0,0)),\
                               constant_values=-5)

            Xn.append(X_train)
            y_train=y[n+self.time_it]
            yn.append(y_train)

        return Xn, yn

class generator_cul():
    "Generating model"
    def __init__(self, window, time_it, feature,
                 list_time_Xcul, list_time_ycul, list_time_culstatic,
                cul_time_y):
        self.window = window
        self.time_it = time_it
        self.feature = feature
        self.list_time_Xn = list_time_Xcul
        self.list_time_yn = list_time_ycul
        self.list_time_nstatic = list_time_culstatic
        self.cul_time_y = cul_time_y

    def get_data(self):
        Xn_time = []
        Xn_static = []
        yn_time = []
        zn_time = []

        ## negative data gathering
        for i in range(len(self.list_time_Xn)):
            X_time_n = self.list_time_Xn[i]
            y_time_n = self.list_time_yn[i]
            cul_n = self.cul_time_y[i]
            Xn, yn,zn = self.time_series_generator_cul(X_time_n,y_time_n, cul_n)
            time_static_data = self.list_time_nstatic[i]

            for n in range(len(Xn)):
                Xn_time.append(Xn[n])
                Xn_static.append(time_static_data)
                yn_time.append(yn[n])
                zn_time.append(zn[n])

        Xn_time = np.array(Xn_time).reshape(-1,self.window,self.feature)
        Xn_static = np.array(Xn_static).reshape(-1,38)
        yn_time = np.array(yn_time).reshape(-1,)
        zn_time = np.array(zn_time).reshape(-1,)

        return Xn_time, Xn_static, yn_time, zn_time


    def time_series_generator_cul(self, x,y,z):
        Xn = []
        yn = []
        zn = []
        cul_loc = [i for i,t in enumerate(z) if t==1]
        for n in cul_loc:
            if n+1>self.window:
                X_train=x[n+1-self.window:n+1]
            else:
                X_train=x[0:n+1]
                X_train=np.pad(X_train, mode='constant', pad_width=((0,self.window-X_train.shape[0]),(0,0)),\
                               constant_values=-5)

            Xn.append(X_train)
            y_train=y[n]
            yn.append(y_train)
            z_train=z[n]
            zn.append(z_train)

        return Xn, yn, zn

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, window,  time_it, feature,
                 true_X_time, true_X_static,true_y_total,
                 false_X_time, false_X_static, false_y_total,
                 list_time_Xn, list_time_yn, list_time_nstatic):
        'Initialization'
        self.window = window
        self.time_it = time_it
        self.feature = feature
        self.true_X_time = true_X_time
        self.true_X_static = true_X_static
        self.true_y_total = true_y_total
        self.false_X_time = false_X_time
        self.false_X_static = false_X_static
        self.false_y_total = false_y_total
        self.list_time_Xn = list_time_Xn
        self.list_time_yn = list_time_yn
        self.list_time_nstatic = list_time_nstatic

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_time_Xn) / 40))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        file_generator1 = file_generator(window = self.window,
                                       time_it =self.time_it,
                                       feature = self.feature,
                                       true_X_time = self.true_X_time,
                                       true_X_static = self.true_X_static,
                                       true_y_total = self.true_y_total,

                                       false_X_time = self.false_X_time,
                                       false_X_static = self.false_X_static,
                                       false_y_total = self.false_y_total,

                                       list_time_Xn = self.list_time_Xn,
                                       list_time_yn = self.list_time_yn,
                                       list_time_nstatic = self.list_time_nstatic)
        # Generate data
        batch_X, batch_static_X, batch_y = file_generator1.get_data(index)

        return ({"time":batch_X, "static":batch_static_X}, batch_y)

from keras.callbacks import Callback
class roc_callback(Callback):
    def __init__(self,valid_x, valid_Xs, valid_y):
        self.x_val_time = valid_x
        self.x_val_static=valid_Xs
        self.y_val = valid_y
        self.list = {"roc_auc_val":[]}

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return self.list

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val =self.model.predict({"time":self.x_val_time, "static":self.x_val_static})
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\r roc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        self.list["roc_auc_val"].append(roc_val)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
