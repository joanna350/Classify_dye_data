import time
import numpy as np
import pandas as pd
from math import log
import xgboost as xgb
from xgboost import DMatrix
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from utility import readin
from utility import filedescription
from utility import redye
from utility import colorcode
from utility import fastness
from utility import supplier
from utility import machine_manufacturer
from utility import subdetail
from utility import dyetype
from utility import shadedepth
from utility import finishtype
from utility import dyehouse


cnf = 0
cv = 1
others = 0

def preprocess():
    '''
    
    '''
#TODO see if this preprocesses test_df in the righ form
#TODO clean code and comment adequately. keep the good practice and upload on git
    print('----------------------reading in')
    train_d, train_o, test_df, test_tp = readin()
    mergemap = {0: 'parent.batch.id', 1: 'batch.id'} # not relevant for training
    for df, y, globalcount in zip([train_d], [train_o], [0]):
        df.drop_duplicates(inplace=True)
        y.drop_duplicates(inplace=True)
        if globalcount == 0:
            df = pd.merge(left=df, right=y, 
                    left_on=mergemap[globalcount], right_on=mergemap[globalcount])

        print('---------------------preprocessing')  
    #TODO: automation could be done by filtering cols with only one val and removing
    # only 1 value (first line) & majority empty (second line) 
        removal = ['recipe.type', 'lub.type.name', 'recipe.type.code','substrate.used.for',\
                    'dye.code.4', 'dye.code.5', 'dye.code.6', 'triangle.code.2','extra']
        print('previous', sum(df['has.passed']),len(df)-sum(df['has.passed'])) #11769, 2551
        df = df.drop(columns=removal)
    # separation is for just field checks. mapping str > int (category)
        tomap = ['supplier','colour.category', 'fastness.type', 'redye.flag', 'machine.manufacturer',
                'subdetail', 'shadedepth', 'dyetype', 'finish.type','dyehouse.code']
        mapfunc = {'supplier': supplier(), 
                    'colour.category': colorcode(), 
                    'fastness.type': fastness(), 
                    'machine.manufacturer': machine_manufacturer(), 
                    'subdetail':subdetail(), 
                    'dyetype':dyetype(), 
                    'shadedepth': shadedepth(),
                    'finish.type': finishtype(),
                    'dyehouse.code': dyehouse()}
        bands = ['stage.1.dye.conc.band','batch.weight.band', 'stage.1.dispense.wt.band']
        more = ['substrate.code'] #str that is equal to the first 3 chars of 'fibre.type'
                               #is usually in 'unfinished.standard.type'. encode 0 if not, 1 else
        to_enc= [tomap, bands, more]
        colmapkey = dict() # {col: {key: category}}
        for cols in to_enc:
            if cols == bands:
                for i in range(len(cols)):
                    col = cols[i]
                    if i == 1 or i ==2: # kg, g
                        df[col] = df[col].astype(str).str[0].astype(int)
                    elif i == 0: # pct
                        df[col] = df[col].replace({'=>3.0%<4%':'=>3.0%<4.0%'})
                        df[col] = df[col].apply(lambda x:x[-4:-1]).astype(float)
            elif cols == tomap:
                for col in cols:
                    if col == 'redye.flag':
                        cates = redye(df[col].unique()) 
                        df[col] = df[col].map(cates)
                    else:
                        df[col] = df[col].map(mapfunc[col])
            else:   # more> substrate.code > material
                df['material_check'] = [x[0] in x[1] for x in \
                         zip(df['substrate.code'], df['unfinished.standard.type'])]
                df['material_check'] = df['material_check'].astype(int) # mapping

                df['dyeclasses'] = [x[3:7] if len(x)>8 else x[3:8] \
                                            for x in df['dyeclasses']]
# material.code is unique per item (Article2, Ticket, aligned to the same length, '-' shade.name)
# thread.group is not fully correct combination string of {substrate + count.ply +etc}
# TODO: machine.model -until configured (one typo in spacing, the rest not informative), remove
                df.drop(columns=['machine.model', 'batch.id','parent.batch.id', 
                                        'thread.group','material.code', 
                                        'unfinished.standard.type','fibre.type'], inplace=True)
                tomap = ['dyeclasses','substrate.code', 'recipe.status', 'machine.name',
                            'shade.name', 'triangle.code.1']
                for col in tomap:
                    labels = df[col].unique() #handled nan in triangle.code.1
                    num = len(labels)
                    encode = list(range(num))
                    tomap = dict(zip(labels, encode))
                    df[col] = df[col].map(tomap)
                df['count'] = df['count.ply'].apply(lambda x:x[:-2]).astype(int)
                df['ply'] = df['count.ply'].apply(lambda x:x[-1]).astype(int)

        df['dyelot.date'] = df['dyelot.date'].apply(lambda x: 60*int(x.split(':')[0]) + int(x.split(':')[1].split('.')[0]) + int(x.split(':')[1].split('.')[1])/10).astype(float)
    #TODO: verify if separted features of time are more effective 
    #    df.rename(columns={'dyelot.year': 'year', 'dyelot.month': 'month', 
    #                                'dyelot.day': 'day', 'dyelot.date': 'ms'}, inplace=True)
    #    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'ms']]) 

        print('after', sum(df['has.passed']), 
                            len(df)-sum(df['has.passed']))

        ambiguous = ['shadedepth', 'Ticket', 'dyetype', 'failure.reason.group', 'count.ply', \
                        'Shade.name2', 'fibre.char', 'Article2', 'subdetail']
        df.drop(columns=ambiguous, inplace=True)
        
        if others:
            df = df.dropna()
        print(df.shape)
        print(df.dtypes)
        cols = list(df.columns)
        if globalcount == 0:
            cols.remove('has.passed')
            train_x, train_y = df[cols], df['has.passed']
       # else:
       #     test_x = df[cols]

 #TIP: machine.manufacturer, machine.model - same lines of na (retain all other info)
 # continuous numeric:   'weight.of.stage.1.dye.dispense.g',

    if others:    
     # .transform to X_test once read from the other file. 
     # not y and not in other modules    
        mm_scaler = MinMaxScaler()
        train_x = mm_scaler.fit_transform(train_x)

    return train_x, train_y

def colourmeasurement(df):
    '''
    :type dataframe: 
    :rtype dataframe: returned after removing NA rows as designed parameter 
    '''
    nans39 = ['L.value','A.value','B.value','chroma.value','hue.value',\
                'delta.l','delta.c','delta.h','max.colour.difference']
    #better perf retained
    #lower perf removing NA 
    #df.drop(columns = nans39, inplace=True)
    print('df that produced raw result', df.shape)
    na_free = df.dropna(subset=nans39)
#    only_na = df[~df.index.isin(na_free.index)]
#    only_na.to_csv('observe_dropped.csv', index=False)
    return na_free

def oversample(X, y):
    '''
    :type X, y: unbalanced data given
    :rtype X_re, y_re: balanced data (for each output category)
    '''
    # 11769, 2551 (1 vs 0)
    #print(sum(y), len(y) - sum(y))
    print('------------------balancing sample')
    ros= RandomOverSampler(random_state=0)
    X_re, y_re = ros.fit_resample(X,y)  
    print(sum(y_re), len(y_re) - sum(y_re))
    return X_re, y_re


def fit(train_x, train_y, test_x, test_y):
    '''
    '''
    models = []
    models.append((1,'xgbclassifier', XGBClassifier(objective='binary:logistic',
                                                    eval_metric='auc')))
 # binary:logistics is the default
 # can pass multiple as List[str] (eval_metric) 
 # random_state 
 # Note: using gblinear booster w shotgun updater is nondeterministic as it use Hogwild algo (=parallel sgd)

    # default solver: lbfgs from v 0.22
    # random_state in effect for solver of {'sag','saga','liblinear'}
    if others:
        models.append((0, 'logistic regression', LogisticRegression(penalty='l1',
                                                                solver='liblinear',
                                                                C=10.0,
                                                                random_state=0)))
        models.append((0, 'gaussian naive bayes', GaussianNB()))
        models.append((0, 'random forest classifier', RandomForestClassifier(max_depth=2, 
                                                                        random_state=0)))
        models.append((0, 'k neighbor classifier', KNeighborsClassifier(n_neighbors=3)))
    for flag, string, model in models:
        print('training model: ', string)
        if flag:
            model.fit(train_x, train_y,
                       early_stopping_rounds = 10,
                        eval_set = [(test_x, test_y)])
        else:
       #standardize (not necessary for decision trees, logreg unless regularized)
            model.fit(train_x, train_y)
        '''
        if flag:
            ax =plot_importance(model)
            ax.figure.tight_layout()
            ax.figure.savefig('xgb_plot_importance.png') 
            #dpi=fig.dpi
            plot_tree(model,
                    num_trees= model.best_iteration,
                    rankdir='LR',
                    ax = ax)
        '''
    return models

def select_feats(model, X, y, Xt, yt):
    '''
    model: those that support feature_importance
    X, y: for training purpose
    Xt, yt: testing purpose
    '''
    k= 6
    thresholds = sorted(list(model.feature_importances_))
    for i in range(k , len(thresholds) - k,k):
        thres=thresholds[i]
        # threshold applied for choice of feats
        sel = SelectFromModel(model, threshold= thres, prefit=True)
        # cross_val_score? not usable w prefit
        # reform X
        mod_X = sel.transform(X)
        # fit to train
        sel_model = XGBClassifier()
        sel_model.fit(mod_X, y)
        # evaluate w Xtest reformed the same
        mod_Xt = sel.transform(Xt)
        pred = sel_model.predict(mod_Xt)
        accuracy, precision, TPR, FPR, FNR = get_evaluation(yt, pred)
        print('threshold=%.3f, n=%d, \nacc: %.3f, prec: %.3f, tpr: %.3f, fpr: %.3f, fnr: %.3f'\
               %(thres, mod_X.shape[1], accuracy, precision, TPR, FPR, FNR))
        print('average: %.3f' %((accuracy + precision + TPR + (1-FPR))/4))

def get_prediction(model, test_x, test_y):
    '''
    will be used to print to test.template.csv
    :type [] test_x: test dataset to base accuracy result
    :type model: fitted model that would return prediction based on given x
    :rtype [np array] y_pred: prediction result the model returns
    :rtype [np array] pred_prob: predicted probability of each X sample being a given class [0,1] order
    '''
    y_pred = model.predict(test_x)
    pred_prob = model.predict_proba(test_x) #5638, 2
    print('--------------prediction')
    return y_pred, pred_prob


def get_evaluation(string, true, pred, pred_prob):

    '''
    
    '''
    print('--------------evaluation')
    cnf_matrix = confusion_matrix(true, pred)
    TP,FN,TN,FP = cnf_matrix[1][1], cnf_matrix[1][0], cnf_matrix[0][0], cnf_matrix[0][1]
    accu = (TP+TN)/(TP+FN+TN+FP)
    prec = TP/(TP+FP)
    tpr =  TP/(TP+FN)
    fpr = FP/(FP+TN)
    fnr = FN/(FN+TN)
    print('accuracy: {:.3f}\nprecision: {:.3f}\ntrue positive rate: {:.3f}\nfalse positive rate: {:.3f}\nfalse negative rate: {:.3f}'.format(accu, prec, tpr, fpr, fnr))
    #roc
    roc_auc= roc_auc_score(true, pred_prob[:,1])
    print('ROC AUC %.3f' %roc_auc)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fpr,tpr)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve')
    plt.savefig(string+'_ROC_curve.jpg')
    
    #pr-recall curve
    pres, recs, _ = precision_recall_curve(true, pred_prob[:,1])    
    plt.plot(recs, pres, marker='.', label=string)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig(string+'_pr_curve.jpg')

    if cnf:
        class_labels = [False, True] #by default
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_labels))
        plt.xticks(tick_marks, class_labels)
        plt.yticks(tick_marks, class_labels)

        #heatmap
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, fmt='g')

        ax.xaxis.set_label_position('top')

        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        plt.savefig('confusion_matrix.jpg')

''' 
    # get the roc curve
    n_classes = 2
    fpr, tpr, roc_auc  = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(pred, test_out, pos_label=2)
        roc_auc[i] = auc(fpr[i], tpr[i])
'''

def cv(X_t, y_t, X_test, y_test):
    dtrain = DMatrix(X_t, label=y_t)
    dtest = DMatrix(X_test, label=y_test)
    params={"objective":"binary:logistic",
            'max_depth': 5, 
            'alpha': 10,
            'eval_metric': 'auc'}
    numbr = 999
#fit
    model = xgb.train(params,
              dtrain,
              num_boost_round=numbr,
              evals=[(dtest,'Test')],
              early_stopping_rounds=10)
    print('best auc: {:.2f} with {} rounds'.format(model.best_score, 
                                                model.best_iteration+1))
    results = xgb.cv(params,    
                 dtrain,
                 num_boost_round=numbr,
                 seed=42,
                 nfold=4,
                 metrics={'auc'},
                 early_stopping_rounds=10)
    print(results)

#TODO: finetune hpp using xgb.cv
def finetune():
    pass

def execute():
    X, y = preprocess()
    X, y = oversample(X, y)
    X_t, X_test, y_t, y_test = train_test_split(X, y,
                                                test_size= 0.25,
                                                random_state = 0)
    if cv:
        cv(X_t, y_t, X_test, y_test)   
    else:
        models = fit(X_t, y_t, X_test, y_test)
        for flag, string, model in models:
            print('model in point: ', string)
            y_pred, pred_prob = get_prediction(model, X_test, y_test)
            #cnf matrix first
            get_evaluation(string, y_test, y_pred, pred_prob)

            #if flag:
               # print(model.feature_importances_)
               # np array[int] (46,) print(model.feature_importances_.shape)
            #    select_feats(model, X_t, y_t, X_test, y_test)
            

if __name__ == '__main__':
    
    execute()

   # train_x, train_y, test_x, test_y = preprocess()
   # model = fit(train_x, train_y, test_x, test_y)
   # pred, pred_prob = get_prediction(model, test_y) 
