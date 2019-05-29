from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from collections import Counter
import tkinter as tk
import random
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_validation
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm
from sklearn.model_selection import KFold
#from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import xlwt
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

root = Tk()
# width x height + x_offset + y_offset:
root.geometry("160x125+550+160")


def comma():
    messagebox.showinfo(
        "CLUSTERING TOOL", "You have selected to proceed with the clustering tool")
    messagebox.showinfo(
        "CLUSTERING TOOL", "Please insert the dataset you want to perform clustering on.")
    filename = filedialog.askopenfilename()
    file = filename
    master = Tk()

    def show_values():
        print (w.get())
        # d=pd.read_excel(file)
        d = pd.read_csv(file)
        print('opened file')
    # - keep the names here of all the genes ......sos the column must be called 'gene '
        names = d['gene']
    # - keep the number of genes in your dataset
        m = len(names)-1
        print('\n-- Your genes are beeing processed ...')
    # take out the names
        pi = d.drop(['gene'], axis=1)
    # - keep  all the values in a variable called TI
        ΤΙ= pi.iloc[:, 0:].values
        name = np.array(names)
        X = np.array(ΤΙ)

# -------------------------------------------------------------------------------
# ----------------------------DO THE FITTING ----------------------------------
# ------------------------------------------------------------------------------

        kmeans = KMeans(n_clusters=int(w.get()), random_state=0).fit(X)
        agglom = AgglomerativeClustering(
            n_clusters=int(w.get()), linkage="ward").fit(X)
        m = kmeans.labels_
        n = agglom.labels_

        teliko = pd.DataFrame(list(zip(name, m)), columns=[
                              'GENES', 'CLUSTER '])
        #teliko.to_excel('kMEANS_cluster.xlsx', engine='xlsxwriter')
        export_csv = teliko.to_csv(
            r'/home/alexia/Downloads/kmeans_cluster.txt', index=None, header=True)

        teliko = pd.DataFrame(list(zip(name, n)), columns=[
                              'GENES', 'CLUSTER '])
        #teliko.to_excel('agglom_cluster.xlsx', engine='xlsxwriter')
        export_csv = teliko.to_csv(
            r'/home/alexia/Downloads/agglom_cluster.txt', index=None, header=True)

        messagebox.showinfo("CLUSTERING TOOL",
                            "You are ready to see your results!")

    w = Scale(master, from_=0, to=200, length=600, orient=HORIZONTAL)
    w.pack()
    Button(master, text='Selected clusters', command=show_values).pack()
    Button(master, text='Close', command=master.destroy).pack()


def comme():
    messagebox.showinfo("CLASSIFICATION TOOL",
                        "You have selected to proceed with the classification tool")
    messagebox.showinfo("CLASSIFICATION TOOL", "Select your training set.")

    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

        # ---------------------------------------------------------------------------
        # -------------------------OPEN DATA ----------------------------------------
        # --------------------------------------------------------------------------

    filename = filedialog.askopenfilename()
    messagebox.showinfo("CLASSIFICATION TOOL",
                        "Thank you,insert the data you want to classify.")
    filenam = filedialog.askopenfilename()
    df = pd.read_csv(filename)
    # df=pd.read_excel(filename)
    df = clean_dataset(df)
    # -keep in x the variable and as y the validation
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # -rename them
    data_input = x
    data_output = y
    print(data_output)
    # -set parameters for the kfold validation
    kf = KFold(10, shuffle=True)
    # -set parameters for the classifiers
    rf_class = RandomForestClassifier(n_estimators=10)
    log_class = LogisticRegression()
    svm_class = svm.SVC()
    nn_class = KNeighborsClassifier(n_neighbors=3)
    svc_class = SVC(kernel="linear", C=0.025)
    gausian_class = GaussianProcessClassifier(1.0 * RBF(1.0))
    dtc_class = DecisionTreeClassifier(max_depth=5)
    mpl_class = MLPClassifier(alpha=1)
    abc_class = AdaBoostClassifier()
    bnb_class = GaussianNB()

    accu = []  # -- here we will keep all the accuracies of each classifier

    #print("Random Forests: ")
    #print(cross_val_score(rf_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracy1 = cross_val_score(
        rf_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracy1)
    #print("Accuracy of Random Forests is: " , accuracy1)

    #print("\n\nsvm-linear: ")
    #print(cross_val_score(svc_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracysvc = cross_val_score(
        svc_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracysvc)
    #print("Accuracy of svm-linear is: " , accuracysvc)

    #print("\n\nGaussian process classifier: ")
    #print(cross_val_score(gausian_class, data_input, data_output, scoring='accuracy', cv = 10))
    #accuracygausian = cross_val_score(gausian_class, data_input, data_output, scoring='accuracy', cv = 10).mean() * 100
    # accu.append(accuracygausian)
    #print("Accuracy of  Gaussian process classifier is: " , accuracygausian)

    #print("\n\nDesicion tree classifier : ")
    #print(cross_val_score(dtc_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracydtc = cross_val_score(
        dtc_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracydtc)
    #print("Accuracy of Desicion tree classifier is: " , accuracydtc)

    #print("\n\nMPL: ")
    #print(cross_val_score(mpl_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracympl = cross_val_score(
        mpl_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracympl)
    #print("Accuracy of MPL Classifier is: " , accuracympl)

    #print("\n\nAdaBoostClassifier: ")
    #print(cross_val_score(abc_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracyabc = cross_val_score(
        abc_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracyabc)
    #print("Accuracy of AdaBoostClassifier is: " , accuracyabc)

    # print("\n\nGaussianNB:")# default is rbf
    #print(cross_val_score(bnb_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracybnb = cross_val_score(
        bnb_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracybnb)
    #print("Accuracy of GaussianNB is: " , accuracybnb)

    # print("\n\nSVM:")# default is rbf
    #print(cross_val_score(svm_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracy2 = cross_val_score(
        svm_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracy2)
    #print("Accuracy of SVM is: " , accuracy2)

    # print("\n\nLog:")
    #print(cross_val_score(log_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracy3 = cross_val_score(
        log_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracy3)
    #print("Accuracy of nLog is: " , accuracy3)

    # print("\n\nNN:")
    #print(cross_val_score(nn_class, data_input, data_output, scoring='accuracy', cv = 10))
    accuracy = cross_val_score(
        nn_class, data_input, data_output, scoring='accuracy', cv=10).mean() * 100
    accu.append(accuracy)
    #print("Accuracy of NN is: " , accuracy)

    # -- here we will display the user which classifier we suggest them to use for their analysis

    bac = max(accu)
    name = []

    if bac == accuracy1:
        name.append('Random Forest')
    elif bac == accuracy2:
        name.append('SVM')
    elif bac == accuracy3:
        name.append('nLog')
    elif bac == accuracysvc:
        name.append('SVC-LINEAR')
    elif bac == accuracy1:
        name.append('Gausian process')
    elif bac == accuracydtc:
        name.append('Decision Tree')
    elif bac == accuracympl:
        name.append('MPL')
    elif bac == accuracyabc:
        name.append('AdaBoost')
    elif bac == accuracybnb:
        name.append('GaussianNB')
    else:
        name.append('knn')

        # -------------------------------------------------------------------------------------------
        # -display all the results
    print("\n\n--Summing up :")

    classi = ['Random Forests', 'SVM', 'nLog', 'SVC-linear',
              'DecisionTree', 'MPL', 'AdaBoost', 'GaussianNB', 'NN']
    accur = [accuracy1, accuracy2, accuracy3, accuracysvc,
             accuracydtc, accuracympl, accuracyabc, accuracybnb, accuracy]
    tog = zip(classi, accur)
    teliko = pd.DataFrame(list(zip(classi, accur)), columns=[
                          'Classifier', 'Accuracy'])
    tel = teliko.sort_values(by=['Accuracy'], ascending=False)
    export_csv = tel.to_csv(
        r'/home/alexia/Downloads/accuracies.txt', index=None, header=True)

    print(tel)

    def printSomething():
         # if you want the button to disappear:
         # button.destroy() or button.pack_forget()
        label = Label(root, text=tel)
        # this creates a new label to the GUI
        label.pack()

    root = Tk()

    button = Button(root, text="CLICK HERE TO VIEW ACCURACIES",
                    command=printSomething)
    button.pack()
    root.mainloop()
    root = tk.Tk()
    v = tk.IntVar()
    v.set(1)  # initializing the choice, i.e. Python

    languages = [

        ("knn"),
        ("Randomforest"),
        ("LogisticRegretion"),
        ("SVM"),
        ("SVM-linear"),
        ("GaussianProcess"),
        ("DecisionTree"),
        ("MPL"),
        ("ΑdaBoost"),
        ("GaussianNB"),
        ("all")
    ]

    def ShowChoice():

        #df = pd.read_excel(filename)
        df = pd.read_csv(filename)
        # df=clean_dataset(df)
        # -keep in x the variable and as y the validation
        x = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values
        # -rename the variables
        data_input = x
        data_output = y

        # ---------------------------------------------------------------------------
        # -------------------------OPEN REAL DATA ----------------------------------------
        # --------------------------------------------------------------------------
        # -open the dataset that you want to run your analysis on
        #d = pd.read_excel(filenam)
        d = pd.read_csv(filenam)
        # d=clean_dataset(df)
        # - keep the names here of all the genes ......sos the column must be called 'gene '
        names = d['gene']
        # - format the matrix so than there will only be int inside
        m = len(names)-1
        print('\n-- There are in total', m, 'genes in your dataset')
        pi = d.drop(['gene'], axis=1)
        # - keep  all the values in a variable called X
        X = pi.iloc[:, 0:].values
        tixera = []  # here are saved in good form all the values for predict
        # mikos=1 # here i define how many i want to run the test on

        def stringator(listind):
            stri = " ".join(str(x) for x in listind)
            T = re.sub("\s+", ",", stri.strip())
            return T
        mikos = m-2
        for t in range(mikos):
            M = X[t+1]
            MM = (int(x) for x in M)
            l = stringator(MM)
            tixera.append(l)
        print('naiiiiiii')

        # ------------------------------------------------------------------------
        # --------------------------LEARNNG PART ---------------------------------
        # -------------------------------------------------------------------------

        # -Do the fitting on our data .
        knn = KNeighborsClassifier(n_neighbors=3)
        rf = RandomForestClassifier(n_estimators=10)
        lg = LogisticRegression()
        svmo = svm.SVC()
        svc = SVC(kernel="linear", C=0.025)
        dtc = DecisionTreeClassifier(max_depth=5)
        mpl = MLPClassifier(alpha=1)
        abc = AdaBoostClassifier()
        bnb = GaussianNB()

        svmo.fit(x, y)
        knn.fit(x, y)
        rf.fit(x, y)
        lg.fit(x, y)
        svc.fit(x, y)
        dtc.fit(x, y)
        mpl.fit(x, y)
        abc.fit(x, y)
        bnb.fit(x, y)
        print('done')

        # ------------------------------------------------------------------------
        # --------------------------PREDICTION PART KNN---------------------------------
        # -------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # --------------------------PREDICTION PART KNN---------------------------------
        # -------------------------------------------------------------------------

        if v.get() == 0 or v.get() == 10:

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = knn.predict([ti])
                pred.append(ni[0])
            # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
            print('knn-done')

            # for st in range(mikos):
            # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_knn.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputknn.txt', index=None, header=True)
            #messagebox.showinfo( "CLUSTERING TOOL", "You are ready to see your results!")

            if v.get() == 1 or v.get() == 10:

                 # ------------------------------------------------------------------------
                 # --------------------------PREDICTION PART RF---------------------------------
                 # -------------------------------------------------------------------------

                pred = []
                for st in range(mikos):
                    t = tixera[st]
                    ti = [int(s) for s in t.split(',')]
                    ni = rf.predict([ti])
                    pred.append(ni[0])
                 # ------------------------------------------------------------------------
                 # --------------------------Create final matrix ---------------------------------
                 # -------------------------------------------------------------------------
                 # given that in training 1 was as good target and as a bad one
                tava = []
                for pr in pred:
                    if pr > 0.95:
                        t = ' GOOD potential target'
                        tava.append(t)
                    else:
                        t = ' BAD potential target'
                        tava.append(t)
                name = []
                for st in range(mikos):
                    t = names[st]
                    name.append(t)
                    print('rf-done')

                 # for st in range(mikos):
                    # print('The gene',name[st],'is according to our algorithm',tava[st])

                teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                      'GENES', 'TARGET VALIDATION '])
                #teliko.to_excel('output_rf.xlsx', engine='xlsxwriter')
                export_csv = teliko.to_csv(
                    r'/home/alexia/Downloads/outputrf.txt', index=None, header=True)

                if v.get() == 2 or v.get() == 10:

                     # ------------------------------------------------------------------------
                     # --------------------------PREDICTION PART RF---------------------------------
                     # -------------------------------------------------------------------------

                    pred = []
                    for st in range(mikos):
                        t = tixera[st]
                        ti = [int(s) for s in t.split(',')]
                        ni = lg.predict([ti])
                        pred.append(ni[0])
                     # ------------------------------------------------------------------------
                     # --------------------------Create final matrix ---------------------------------
                     # -------------------------------------------------------------------------
                     # given that in training 1 was as good target and as a bad one
                    tava = []
                    for pr in pred:
                        if pr > 0.95:
                            t = ' GOOD potential target'
                            tava.append(t)
                        else:
                            t = ' BAD potential target'
                            tava.append(t)
                    name = []
                    for st in range(mikos):
                        t = names[st]
                        name.append(t)
                        print('logistic done')

                     # for st in range(mikos):
                        # print('The gene',name[st],'is according to our algorithm',tava[st])

                    teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                          'GENES', 'TARGET VALIDATION '])
                    #teliko.to_excel('output_lg.xlsx', engine='xlsxwriter')
                    export_csv = teliko.to_csv(
                        r'/home/alexia/Downloads/outputlg.txt', index=None, header=True)

                    if v.get() == 3 or v.get() == 10:

                         # ------------------------------------------------------------------------
                         # --------------------------PREDICTION PART RF---------------------------------
                         # -------------------------------------------------------------------------

                        pred = []
                        for st in range(mikos):
                            t = tixera[st]
                            ti = [int(s) for s in t.split(',')]
                            ni = svmo.predict([ti])
                            pred.append(ni[0])
                         # ------------------------------------------------------------------------
                         # --------------------------Create final matrix ---------------------------------
                         # -------------------------------------------------------------------------
                         # given that in training 1 was as good target and as a bad one
                        tava = []
                        for pr in pred:
                            if pr > 0.95:
                                t = ' GOOD potential target'
                                tava.append(t)
                            else:
                                t = ' BAD potential target'
                                tava.append(t)
                        name = []
                        for st in range(mikos):
                            t = names[st]
                            name.append(t)
                            print('svm-done')

                         # for st in range(mikos):
                            # print('The gene',name[st],'is according to our algorithm',tava[st])

                        teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                              'GENES', 'TARGET VALIDATION '])
                        #teliko.to_excel('output_svm.xlsx', engine='xlsxwriter')
                        export_csv = teliko.to_csv(
                            r'/home/alexia/Downloads/outputsvm.txt', index=None, header=True)
                        if v.get() == 4 or v.get() == 10:

                             # ------------------------------------------------------------------------
                             # --------------------------PREDICTION PART RF---------------------------------
                             # -------------------------------------------------------------------------

                            pred = []
                            for st in range(mikos):
                                t = tixera[st]
                                ti = [int(s) for s in t.split(',')]
                                ni = svc.predict([ti])
                                pred.append(ni[0])
                             # ------------------------------------------------------------------------
                             # --------------------------Create final matrix ---------------------------------
                             # -------------------------------------------------------------------------
                             # given that in training 1 was as good target and as a bad one
                            tava = []
                            for pr in pred:
                                if pr > 0.95:
                                    t = ' GOOD potential target'
                                    tava.append(t)
                                else:
                                    t = ' BAD potential target'
                                    tava.append(t)
                            name = []
                            for st in range(mikos):
                                t = names[st]
                                name.append(t)
                                print('svmlineardone')

                             # for st in range(mikos):
                                # print('The gene',name[st],'is according to our algorithm',tava[st])

                            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                                  'GENES', 'TARGET VALIDATION '])
                            #teliko.to_excel('output_svc.xlsx', engine='xlsxwriter')
                            export_csv = teliko.to_csv(
                                r'/home/alexia/Downloads/outputscv.txt', index=None, header=True)
                            if v.get() == 5 or v.get() == 10:

                                 # ------------------------------------------------------------------------
                                 # --------------------------PREDICTION PART RF---------------------------------
                                 # -------------------------------------------------------------------------

                                 # ------------------------------------------------------------------------
                                 # --------------------------Create final matrix ---------------------------------
                                 # -------------------------------------------------------------------------
                                 # given that in training 1 was as good target and as a bad one
                                print('paaame')
                                if v.get() == 6 or v.get() == 10:

                                     # ------------------------------------------------------------------------
                                     # --------------------------PREDICTION PART RF---------------------------------
                                     # -------------------------------------------------------------------------

                                    pred = []
                                    for st in range(mikos):
                                        t = tixera[st]
                                        ti = [int(s) for s in t.split(',')]
                                        ni = dtc.predict([ti])
                                        pred.append(ni[0])
                                     # ------------------------------------------------------------------------
                                     # --------------------------Create final matrix ---------------------------------
                                     # -------------------------------------------------------------------------
                                     # given that in training 1 was as good target and as a bad one
                                    tava = []
                                    for pr in pred:
                                        if pr > 0.95:
                                            t = ' GOOD potential target'
                                            tava.append(t)
                                        else:
                                            t = ' BAD potential target'
                                            tava.append(t)
                                    name = []
                                    for st in range(mikos):
                                        t = names[st]
                                        name.append(t)
                                        print('dtree done')

                                     # for st in range(mikos):
                                        # print('The gene',name[st],'is according to our algorithm',tava[st])

                                    teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                                          'GENES', 'TARGET VALIDATION '])
                                    #teliko.to_excel('output_dtc.xlsx', engine='xlsxwriter')
                                    export_csv = teliko.to_csv(
                                        r'/home/alexia/Downloads/outputdtc.txt', index=None, header=True)
                                    if v.get() == 7 or v.get() == 10:

                                         # ------------------------------------------------------------------------
                                         # --------------------------PREDICTION PART RF---------------------------------
                                         # -------------------------------------------------------------------------

                                        pred = []
                                        for st in range(mikos):
                                            t = tixera[st]
                                            ti = [int(s) for s in t.split(',')]
                                            ni = mpl.predict([ti])
                                            pred.append(ni[0])
                                         # ------------------------------------------------------------------------
                                         # --------------------------Create final matrix ---------------------------------
                                         # -------------------------------------------------------------------------
                                         # given that in training 1 was as good target and as a bad one
                                        tava = []
                                        for pr in pred:
                                            if pr > 0.95:
                                                t = ' GOOD potential target'
                                                tava.append(t)
                                            else:
                                                t = ' BAD potential target'
                                                tava.append(t)
                                        name = []
                                        for st in range(mikos):
                                            t = names[st]
                                            name.append(t)
                                            print('mpl done')

                                         # for st in range(mikos):
                                            # print('The gene',name[st],'is according to our algorithm',tava[st])

                                        teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                                              'GENES', 'TARGET VALIDATION '])
                                        #teliko.to_excel('output_mpl.xlsx', engine='xlsxwriter')
                                        export_csv = teliko.to_csv(
                                            r'/home/alexia/Downloads/outputmpl.txt', index=None, header=True)
                                        if v.get() == 8 or v.get() == 10:

                                             # ------------------------------------------------------------------------
                                             # --------------------------PREDICTION PART RF---------------------------------
                                             # -------------------------------------------------------------------------

                                            pred = []
                                            for st in range(mikos):
                                                t = tixera[st]
                                                ti = [int(s)
                                                      for s in t.split(',')]
                                                ni = abc.predict([ti])
                                                pred.append(ni[0])
                                            # ------------------------------------------------------------------------
                                             # --------------------------Create final matrix ---------------------------------
                                             # -------------------------------------------------------------------------
                                             # given that in training 1 was as good target and as a bad one
                                            tava = []
                                            for pr in pred:
                                                if pr > 0.95:
                                                    t = ' GOOD potential target'
                                                    tava.append(t)
                                                else:
                                                    t = ' BAD potential target'
                                                    tava.append(t)
                                            name = []
                                            for st in range(mikos):
                                                t = names[st]
                                                name.append(t)
                                                print('adaboost done')

                                             # for st in range(mikos):
                                                # print('The gene',name[st],'is according to our algorithm',tava[st])

                                            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                                                  'GENES', 'TARGET VALIDATION '])
                                            #teliko.to_excel('output_adaboost.xlsx', engine='xlsxwriter')
                                            export_csv = teliko.to_csv(
                                                r'/home/alexia/Downloads/outputadaboost.txt', index=None, header=True)
                                            if v.get() == 9 or v.get() == 10:

                                                 # ------------------------------------------------------------------------
                                                 # --------------------------PREDICTION PART RF---------------------------------
                                                 # -------------------------------------------------------------------------

                                                pred = []
                                                for st in range(mikos):
                                                    t = tixera[st]
                                                    ti = [int(s)
                                                          for s in t.split(',')]
                                                    ni = bnb.predict([ti])
                                                    pred.append(ni[0])
                                                 # ------------------------------------------------------------------------
                                                 # --------------------------Create final matrix ---------------------------------
                                                 # -------------------------------------------------------------------------
                                                 # given that in training 1 was as good target and as a bad one
                                                tava = []
                                                for pr in pred:
                                                    if pr > 0.95:
                                                        t = ' GOOD potential target'
                                                        tava.append(t)
                                                    else:
                                                        t = ' BAD potential target'
                                                        tava.append(t)
                                                name = []
                                                for st in range(mikos):
                                                    t = names[st]
                                                    name.append(t)
                                                    print('gausian nb. done')

                                                 # for st in range(mikos):
                                                    # print('The gene',name[st],'is according to our algorithm',tava[st])

                                                teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                                                      'GENES', 'TARGET VALIDATION '])
                                                #teliko.to_excel('output_bnb.xlsx', engine='xlsxwriter')
                                                export_csv = teliko.to_csv(
                                                    r'/home/alexia/Downloads/outputbnb.txt', index=None, header=True)

        elif v.get() == 1:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = rf.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('rf done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_rf.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputrf.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")

        elif v.get() == 2:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = lg.predict([ti])
                pred.append(ni[0])
            # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('lg done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_lg.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputlg.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 3:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = svmo.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('svm done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputsvm.txt', index=None, header=True)
            #teliko.to_excel('output_svm.xlsx', engine='xlsxwriter')
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 4:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = svc.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('svm linear done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
           # teliko.to_excel('output_svc.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputsvc.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 16:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = gausian.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            teliko.to_excel('output_gausianprocess.xlsx', engine='xlsxwriter')

        elif v.get() == 6:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = dtc.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('det.trees done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_dtc.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputdtc.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 7:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = mpl.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('mpl done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_mpl.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputmpl.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 8:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = abc.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('adaboost done')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_adaboost.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputadaboost.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        elif v.get() == 9:

             # ------------------------------------------------------------------------
             # --------------------------PREDICTION PART RF---------------------------------
             # -------------------------------------------------------------------------

            pred = []
            for st in range(mikos):
                t = tixera[st]
                ti = [int(s) for s in t.split(',')]
                ni = bnb.predict([ti])
                pred.append(ni[0])
             # ------------------------------------------------------------------------
             # --------------------------Create final matrix ---------------------------------
             # -------------------------------------------------------------------------
             # given that in training 1 was as good target and as a bad one
            tava = []
            for pr in pred:
                if pr > 0.95:
                    t = ' GOOD potential target'
                    tava.append(t)
                else:
                    t = ' BAD potential target'
                    tava.append(t)
            name = []
            for st in range(mikos):
                t = names[st]
                name.append(t)
                print('nb')

             # for st in range(mikos):
                # print('The gene',name[st],'is according to our algorithm',tava[st])

            teliko = pd.DataFrame(list(zip(name, tava)), columns=[
                                  'GENES', 'TARGET VALIDATION '])
            #teliko.to_excel('output_bnb.xlsx', engine='xlsxwriter')
            export_csv = teliko.to_csv(
                r'/home/alexia/Downloads/outputbnb.txt', index=None, header=True)
            messagebox.showinfo("CLASSIFICATION TOOL",
                                "You are ready to see your results!")
        if v.get() == 10:
            print('\n-- Since you have chosen to use all classifiers, a cross validation of all results will be done.')
            genes = []
            counter = 0

            outputs = ['outputadaboost.txt', 'outputrf.txt', 'outputsvm.txt', 'outputlg.txt',
                       'outputknn.txt', 'outputdtc.txt', 'outputscv.txt', 'outputmpl.txt', 'outputbnb.txt']
            for files in outputs:
                counter += 1
                df = pd.read_csv(files)
                d = np.array(df)
                for x in d:
                    if x[1] == ' GOOD potential target':
                        genes.append(x[0])
                print('done processing the :', counter, 'file')

            counted_genes = Counter(genes)
            name = []
            number = []
            a1_sorted_keys = sorted(
                counted_genes, key=counted_genes.get, reverse=True)
            for r in a1_sorted_keys:
                name.append(r)
                number.append(counted_genes[r])

            df = pd.DataFrame(list(zip(name, number)), columns=[
                              'GENES', 'NO OCCURRENCES '])
            #df=pd.DataFrame.from_dict(counted_genes,orient='index',columns=['number of times validated as a good potential target'])
            d = np.array(df)
            dd = []
            for x in d:
                if x[1] > 1:
                    dd.append(x)
            l = pd.DataFrame(dd, columns=['GENES', 'NO OCCURRENCES '])
            print(l)
            #df.to_csv('crossvalidated.xlsx', engine='xlsxwriter')
            export_csv = df.to_csv(
                r'/home/alexia/Downloads/crossvalidated.txt', index=None, header=True)
            messagebox.showinfo("CLUSTERING TOOL",
                                "You are ready to see your results!")

    tk.Label(root,
             text="""Choose your classifier:""",
             justify=tk.LEFT,
             padx=20).pack()

    for val, language in enumerate(languages):
        tk.Radiobutton(root,
                       text=language,
                       padx=20,
                       variable=v,
                       command=ShowChoice,
                       value=val).pack(anchor=tk.W)
    root.mainloop()


languages = ['CLASSIFICATION', 'CLUSTERING']
button = range(2)
for i in range(2):
    ct = [random.randrange(256) for x in range(3)]
    brightness = int(round(0.299*ct[0] + 0.587*ct[1] + 0.114*ct[2]))
    ct_hex = "%02x%02x%02x" % tuple(ct)
    bg_colour = '#' + "".join(ct_hex)
    l = tk.Button(root,
                  text=languages[i],
                  fg='White' if brightness < 120 else 'Black',
                  bg=bg_colour,
                  command=comma if i == 1 else comme)

    l.place(x=20, y=30 + i*30, width=120, height=25)


root.mainloop()
