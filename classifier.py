'''
script that handles preprocessing, training, predicting and evaluation
over a representative set of classifier models 
with sufficient number of metrics
'''
import os
import time
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from math import log
import xgboost as xgb
from xgboost import DMatrix
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from matplotlib import rcParams
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
from argparse import ArgumentParser
import logging

def argparser():
    '''
    accepts positional argument for setting based operation, ow run with default values
    :rtype: Bool: deciding to plot cnf / conduct gridesearch / run other models
    '''
    parser =ArgumentParser()
    parser.add_argument('plot_cnf', nargs= '?', action='store', default=0, type =int)
    parser.add_argument('gridsearch', nargs= '?',action='store', default=0, type =int)
    parser.add_argument('others', nargs= '?',action='store', default=0, type =int)

    args = vars(parser.parse_args())

    logging.info('arguments set at')
    logging.info(args)
    return args['plot_cnf'], args['gridsearch'], args['others']


def preprocess(others, category):
    '''
    invoke the functions that transform data to an adequate form for train/testing
    :type Bool others: whether to run other models, in which case NA rows would be dropped and data, scaled
    :rtype dataframes: X, y for training
    '''
    logging.info('----------------------reading in')
    train_d, train_o, test_df = readin()
    
    mergemap = {0: 'parent.batch.id', 1: 'batch.id'} # not relevant for training
    stringmap = {0: 'training data', 1: 'test columns'}
    dfmap = {0: train_d, 1: test_df}
    ymap = {0: train_o, 1: None}
    string = stringmap[category]
    df = dfmap[category]
    y = ymap[category]
    
    df.drop_duplicates(inplace=True)
    if category == 0:
        y.drop_duplicates(inplace=True)
        df = pd.merge(left=df, right=y, 
                left_on=mergemap[category], right_on=mergemap[category])

    logging.info('---------------------preprocessing %s', string)  
#TODO: automation could be done by filtering cols with only one val and removing
# only 1 value (first line) or majority empty (second line) 
    df = df.rename(columns={'weight.of.stage.1.dye.dispense..g.': 'weight.of.stage.1.dye.dispense.g.'})
    removal = ['recipe.type', 'lub.type.name', 'recipe.type.code','substrate.used.for',\
                'dye.code.4', 'dye.code.5', 'dye.code.6', 'triangle.code.2','extra']
    df = df.drop(columns=removal)
#  mapping str > int (category)
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
           # in 'unfinished.standard.type' to high proportion. encode 0 if not, 1 else
    to_enc= [tomap, bands, more]
    colmapkey = dict() # {col: {key: category}}
    for cols in to_enc:
        if cols == bands:
            for i in range(len(cols)):
                col = cols[i]
                if i == 1 or i ==2: # kg, g
                    df[col] = df[col].astype(str).str[0].astype(int) # has digit encoded as the 1st char
                elif i == 0: # pct
                    df[col] = df[col].replace({'=>3.0%<4%':'=>3.0%<4.0%'}) # data inconsistency
                    df[col] = df[col].apply(lambda x:x[-4:-1]).astype(float) # take the upper hand in interval[)
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
            df['material_check'] = df['material_check'].astype(int) # bool > 1/0
            df['dyeclasses'] = [x[3:7] if len(x) > 8 else x[3:8] \
                                            for x in df['dyeclasses']] 
    # material.code is unique per item (Article2, Ticket, aligned to the same length, '-' shade.name)
    # thread.group not a fully correct combination string of {substrate + count.ply +etc}
    # TODO: machine.model -until configured (one typo in spacing, the rest not informative), remove
            if category:
                batch_ids = df['batch.id'].to_list()
            df.drop(columns=['machine.model','parent.batch.id','batch.id', 
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
    if category:
        df['dyelot.date'] = df['dyelot.date'].apply(lambda x:int(x[-9:].split(':')[0])*60 + \
                                             int(x[-9:].split(':')[1].split('.')[0]) +\
                                             int(x[-9:].split(':')[1].split('.')[1])/1000)
    else:
        df['dyelot.date'] = df['dyelot.date'].apply(lambda x: 60*int(x.split(':')[0]) + \
                                                int(x.split(':')[1].split('.')[0]) + \
                                                int(x.split(':')[1].split('.')[1])/10).astype(float)

    ambiguous = ['shadedepth', 'Ticket', 'dyetype', 'failure.reason.group', 'count.ply', \
                    'Shade.name2', 'fibre.char', 'Article2', 'subdetail']
    df.drop(columns=ambiguous, inplace=True)
#keep batch.id <> index match, unless keeping batch.id in training retains perf
    if others:
        df = df.dropna() # results in 13596, 47 for training
    cols = list(df.columns)
    if category == 0:
        cols.remove('has.passed')
        train_x, train_y = df[cols], df['has.passed']
    else:
        df[cols].to_csv('preprocessed_test_df.csv')
        test_x = df[cols]
        return test_x, batch_ids
 #TIP: machine.manufacturer, machine.model - same lines of na (retain all other info)
#   print(cols == list(train_x.columns)) True
    if others:    
        mm_scaler = MinMaxScaler()
        train_x = mm_scaler.fit_transform(train_x)
        #if category: # not testing over other models.
           # test_x = mm_scaler.transform(test_x)
    return cols, train_x, train_y


def colourmeasurement(df):
    '''
    purposed to test the impact of the 8 column features that had equivalent number of missing rows > better to retain for performance
    :type dataframe: 
    :rtype dataframe: returned after removing NA rows as designed
    '''
    nans39 = ['L.value','A.value','B.value','chroma.value','hue.value',\
                'delta.l','delta.c','delta.h','max.colour.difference']
    logging.debug('df that produced raw result: %s, %s', df.shape[0], df.shape[1])
    na_free = df.dropna(subset=nans39)
    only_na = df[~df.index.isin(na_free.index)]
    only_na.to_csv('observe_dropped.csv', index=False)
    return na_free

def oversample(X, y):
    '''
    sampling with replacemet
    :type X, y: unbalanced data given
    :rtype X_re, y_re: balanced data (for each output category)
    '''
    logging.info('------------------balancing sample')
    ros= RandomOverSampler(random_state=0)
    X_re, y_re = ros.fit_resample(X,y)  
    logging.info('X: %s, %s', X_re.shape[0], X_re.shape[1])
    logging.info('%s, %s',sum(y_re), len(y_re) - sum(y_re))
    return X_re, y_re


def fit(others, cols, train_x, train_y, test_x, test_y):
    '''
    :type Bool others: decides to train other models or not
    :type dataframes: to fit into each model
    :rtype List[tup] models: each tuple consisting of Bool flag (xgboost or not), Str (model name), Model instance (trained)
    '''
    models = []
    models.append((1,'xgbclassifier', XGBClassifier(objective='binary:logistic', eval_metric='auc')))
    if others:
        models.append((0, 'logistic regression', LogisticRegression(penalty='l1', solver='liblinear', C=10.0, random_state=0)))
        models.append((0, 'gaussian naive bayes', GaussianNB()))
        models.append((0, 'random forest classifier', RandomForestClassifier(max_depth=2, random_state=0)))
        models.append((0, 'k neighbor classifier', KNeighborsClassifier(n_neighbors=3)))

    for flag, string, model in models:
        logging.info('training model: %s', string)
        if flag: 
            print(cols)
            model.fit(train_x, train_y, early_stopping_rounds = 10, eval_set = [(test_x, test_y)])
            plot_importance_w_featname(string, cols, model)
            import sys
            sys.exit()
            plt.clf()
        else: 
            model.fit(train_x, train_y)
    return models

def feature_importance_plot(model, cols): 
    '''
    legacy. tested and plotted for personal use
    :type model: xgboost (possible for all decision tree based models)
    :type List[Str] cols: features
    '''
    tups = sorted(list(zip(model.feature_importances_, cols)), reverse=True) 
    rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()
    barsize, padding =1/4, 1/4
    y_locs = np.arange(len(cols)) * (barsize + padding)
    ax.barh(y_locs, [i[0] for i in tups], align ='center', height = barsize, color='r')
    ax.set(yticks=np.arange(len(cols)), yticklabels=[i[1] for i in tups], ylim =[0-padding,len(cols)])
    ax.invert_yaxis()
    ax.set_xlabel('feature importance')
    fig.savefig('plots/temp_bar.png')
    plt.clf()
            
def select_feats(model, X, y, Xt, yt):
    '''
    legacy now, already done for the evaluators
    purposed to see if reducing features improves the XGBoost, the strongest model
    :type model: those that support feature_importance (tree based, but testing only for XGB in this case)
    :type X, y: for training purpose
    :type Xt, yt: testing purpose
    '''
    k = 6 # not automating as it is done for the evaluators
    thresholds = sorted(list(model.feature_importances_)) # total 46 feats, so not doing increment of 1
    for i in range(k , len(thresholds) - k, k):
        thres = thresholds[i]
        # threshold to filter feats
        sel = SelectFromModel(model, threshold= thres, prefit=True)
        # reform X
        mod_X = sel.transform(X)
        # fit to train
        sel_model = XGBClassifier()
        sel_model.fit(mod_X, y)
        # evaluate w Xtest reformed the same
        mod_Xt = sel.transform(Xt)
        pred = sel_model.predict(mod_Xt)
        accuracy, precision, TPR, FPR, FNR = get_evaluation(yt, pred)
        logging.info('threshold=%.3f, n=%d, \nacc: %.3f, prec: %.3f, tpr: %.3f, fpr: %.3f, fnr: %.3f'\
                                    %(thres, mod_X.shape[1], accuracy, precision, TPR, FPR, FNR))
        logging.info('average: %.3f' %((accuracy + precision + TPR + (1-FPR))/4))


def get_prediction(model, test_x):
    '''
    TODO: to output in the requestd format to test.template.csv
    :type dataframe test_x: test dataset, to produce output with the model
    :type model: fitted model that would return prediction based on given x
    :rtype [np array] y_pred: prediction result the model returns
    :rtype [np array] pred_prob: predicted probability of each X sample being a given class [0,1] order
    '''
    y_pred = model.predict(test_x)
    pred_prob = model.predict_proba(test_x) #5638, 2
    logging.info('--------------prediction')
    return y_pred, pred_prob


def get_evaluation(string, true, pred, pred_prob, plot_cnf, bst):

    '''
    :type Str string: logging purpose
    :type true: y output for test data
    :type pred: output from the model
    :type pred_prob: length of number of classes. for ROC AUC/PR-curve
    :type Bool plot_cnf: plot to the designated (curr) path or not
    
    '''
    logging.info('--------------evaluation')
    cnf_matrix = confusion_matrix(true, pred)
    TP,FN,TN,FP = cnf_matrix[1][1], cnf_matrix[1][0], cnf_matrix[0][0], cnf_matrix[0][1]
    accu = (TP+TN)/(TP+FN+TN+FP)
    prec = TP/(TP+FP)
    tpr =  TP/(TP+FN)
    fpr = FP/(FP+TN)
    fnr = FN/(FN+TP)
    logging.info('accuracy: %.3f\nprecision: %.3f\ntrue positive rate: %.3f\nfalse positive rate: %.3f\nfalse negative rate: %.3f', accu, prec, tpr, fpr, fnr)
    if bst == 0:
        pred_prob = pred_prob[:,1]
    #roc
    roc_auc= roc_auc_score(true, pred_prob)
    logging.info('ROC AUC %.3f' %roc_auc)
    fprs, tprs, thresh = roc_curve(true, pred_prob)
    plt.plot([0,1],[0,1], 'k--')
    plt.plot(fprs, tprs, color='red',
            lw=2, label='ROC curve (area=%0.2f)' %roc_auc)
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(string + ' Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig('plots/'+string+'_ROC_curve.png')
    plt.clf()
    #pr-recall curve - although I balanced the data
    pres, recs, _ = precision_recall_curve(true, pred_prob)    
    plt.plot(recs, pres, marker='.', label=string)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title(string + ' PR curve')
    plt.savefig('plots/' + string + '_pr_curve.png')
    plt.clf()
    if plot_cnf:
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
        plt.savefig('plots/' + string + '_confusion_matrix.png')
        plt.clf()


def cv(X_t, y_t, X_test, y_test):
    '''
    :type dataframes: will be using DMatrix, optimized datastructure offered from xgboost
    testing new APIs in xgboost with train/cv, to be used for hyperparameter tuning
    '''
    dtrain = DMatrix(X_t, label=y_t)
    dtest = DMatrix(X_test, label=y_test)
    params={"objective":"binary:logistic",
            'max_depth': 9,
            'min_child_weight': 2,
            'subsample': 0.8,
            'eta': 0.1,
            'alpha': 0.2,
            'lambda': 0.2,
            'eval_metric':'auc'}
    num_boost_round = 313 #model.best_iteration + 1
    #fit
    model = xgb.train(params,
              dtrain,
              num_boost_round=num_boost_round,
              evals=[(dtest,'Test')])

    return model

    '''
    logging.info('best auc: %.2f with %f rounds', model.best_score, model.best_iteration+1)
    cv_results = xgb.cv(params,    
                 dtrain,
                 num_boost_round=num_boost_round,
                 seed=42,
                 nfold=4,
                 metrics={'auc'},
                 early_stopping_rounds=10)

    logging.info(cv_results)

    gridsearch_params= [(lambda_, alpha) for lambda_ in [.5, .2, .1, .05] for alpha in [.2, .1, .05, .01, .005]] 
    max_auc = 0
    best_params = None
    for lambda_, alpha in gridsearch_params:
        logging.info('CV with lambda= {}, alpha={}'.format(lambda_, alpha))
        params['lambda'] = lambda_
        params['alpha'] = alpha

        cv_results = xgb.cv(params,
                            dtrain,
                            num_boost_round=num_boost_round,
                            seed=42,
                            nfold=5,
                            metrics={'auc'},
                            early_stopping_rounds=10)
        
        mean_auc = cv_results['test-auc-mean'].max()
        boost_rounds=cv_results['test-auc-mean'].argmax()
        logging.info('\tAUC {} for {} rounds'.format(mean_auc, boost_rounds))
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = (lambda_, alpha)
    logging.info('best params: {}, {}, AUC: {}'.format(best_params[0], best_params[1], max_auc))
   ''' 
    
def plot_importance_w_featname(string, cols, model):
    '''
    :type string: description of the given model
    :type cols: column(features) names to map when the data lost the info
    :type model: decision tree based models whose features will be plotted by importance
    ''' 
    ax = plot_importance(model)
    txt_yticklabels = list(ax.get_yticklabels())
    if txt_yticklabels[0].get_text()[1:].isdigit():
        nmapcol = dict(zip(range(len(cols)), cols))
        _yticklabels = [txt_yticklabels[i].get_text().lstrip('f') for i in range(len(txt_yticklabels))]
        _yticklabels = [nmapcol[int(i)] for i in _yticklabels] 
        ax.set_yticklabels(_yticklabels)
    ax.figure.tight_layout()
    ax.figure.savefig('plots/'+string+'_plot_importance.png') 
    plt.clf()

def execute():
    '''
    main
    '''
    logging.getLogger().setLevel(logging.INFO)
    plot_cnf, gridsearch, others = argparser()
    cols, X, y = preprocess(others, 0)
    X, y = oversample(X, y)
    X_t, X_val, y_t, y_val = train_test_split(X, y, test_size= 0.25, random_state = 0)
    if gridsearch:
        string = 'finetuned.booster'
        if Path(string).is_file():
            logging.info('---------loading %s', string)
            optmodel = xgb.Booster()
            optmodel.load_model(string)
            Vmatrix = DMatrix(X_val, label = y_val)
            logging.info('---------validation result from %s', string)
            pred, pred_prob = bst_prediction(optmodel, Vmatrix)
            get_evaluation(string, y_val, pred, pred_prob, plot_cnf, gridsearch)
            ax = plot_importance_w_featname(string, cols, optmodel)
            
            predict_w_optmodel(optmodel, string, others)
        else:
            optmodel = cv(X_t, y_t, X_val, y_val)
            logging.info('--------saving %s', string)
            optmodel.save_model(string)
    else:
        models = fit(others, cols, X_t, y_t, X_val, y_val)
        for flag, string, model in models:
            logging.info('model in point: %s', string)
            y_pred, pred_prob = get_prediction(model, X_val)
            get_evaluation(string, y_val, y_pred, pred_prob, plot_cnf, gridsearch)
            if flag and not others:
                logging.info('----------------real test returns per XGB')
                X_test, batch_ids = preprocess(others, 1)
                pred, pred_prob = get_prediction(model, X_test)
                out_df = pd.DataFrame(zip(batch_ids, pred, pred_prob[:,1]), columns=['batch.id', 'has.passed.prediction', 'has.passed.probability'])
                logging.info('------------------done and saving')
                out_df.to_csv('test.template.csv', index=False) 


def bst_prediction(optmodel, Dmatrix):
    '''
    :type booster model: different API, different output shape
    :type Dmatrix: input datatype optimized in efficiency
    :rtype List[int]: 1/0 for pass/fail
    :rtype List[int]: probability of pass
    '''
    pred_prob = optmodel.predict(Dmatrix)
    pred = [1 if i > 0.5 else 0 for i in pred_prob]
    return pred, pred_prob

def predict_w_optmodel(optmodel, string, others):
    '''
    :type booster model: will be used to make the final prediction and save
    :type Str string: description of the model in current operation
    :type int others: equivalent to boolean for not running other model
    '''
    X_test, batch_ids = preprocess(others, 1)
    dtest = DMatrix(X_test)   
    logging.info('---------------real test prediction with %s', string)
    pred, pred_prob = bst_prediction(optmodel, dtest)
    opt_df = pd.DataFrame(zip(batch_ids, pred, pred_prob), columns=['batch.id', 'has.passed.prediction', 'has.passed.probability'])
    logging.info('------------------saving the results at current timestamp')
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    outdir = Path.cwd() / timestamp
    if not outdir.is_dir():
        os.mkdir(outdir)
        opt_df.to_csv(outdir / 'test.template.csv', index=False)
    else:
        opt_df.to_csv(outdir / 'test.template.csv', index=False)


if __name__ == '__main__':
    execute()
