import pathlib
import os
import pandas as pd

def readin():
    currdir = pathlib.Path().absolute()
    for path, subdir, files in os.walk(currdir):
    # list of files under each subdir (will store the last in data)
        datapath = path
        files = files    

    trainf,testf = dict(), dict()
    for fn in files: 
        if 'train' in fn:
            if 'train.csv' == fn:
                trainf['base'] = fn
            else:
                trainf['out'] = fn
        elif 'test' in fn:
            if 'test.csv' == fn:
                testf['base'] = fn
            else:
                testf['temp'] = fn
   
    train_df = pd.read_csv(datapath +'/'+ trainf['base'])
    train_out = pd.read_csv(datapath +'/'+ trainf['out'])
    test_df = pd.read_csv(datapath +'/'+testf['base'])
    test_tp = pd.read_csv(datapath+'/'+testf['temp'])

    return train_df, train_out, test_df, test_tp

    
def filedescription():
    df = pd.read_excel('FieldDescription.xlsx')
    features = df['Column Name']
    features = features.tolist()
    for i in range(len(features)):
        features[i] = features[i].replace('_', '.')
    return features


def dyehouse():
    return {'VN_HAN':0, 'VN_HCM':1}

def redye(unique):
    cates = dict()
    for k in unique:
        cates[k] = int(k[-1])
    return cates

def colorcode():
    cates ={'DEFAULT': 1, 'NATURAL': 2, 'BLACK': 3, 'FLUORESCENT': 0}
    return cates

def fastness():
    return {'NORMAL':0, 'HALF BLEACH FAST':1, 'HIGH BLEACH FAST':2, 'HIGH WASH FAST':3}

def supplier():
    return {'G':0, 'L':1, 'U':2, 'Z':3}
    
def machine_manufacturer():
    return {'Fong\'s': 4, 'Ugolini':0, 'Longclose':2, 'Thies':3, 'N/A':1}

def subdetail():
    return {'H':0, 'N':1, 'P':2}

def dyetype():
    return {'B':0, 'D':1, 'E':2, 'P':3, 'T':4, 'W':5}

def shadedepth():
    return {'D':0, 'M':1, 'P':2, 'V':3, 'W':4}

def finishtype():
    return {
            'ANTIWICK': 0,
            'ANTIWICK PFC FREE':1,
            'FLAME RETARDANT':2,
            'LOW LUB':3,
            'HIGH LUB':4,
            'LUB AND BOND':5,
            'NO LUB':6,
            'NORMAL':7,
            'PU BONDED':8,
            'UNBONDED':9
            }


def fibertype(unique_set):
    cates = dict()
    for key in unique_set:
        if key[:4] in cates:
            cates[key[:4]] +=1
        else:
            cates[key[:4]] = 1
    return cates

def ply(unique_):
    cates = dict()
    for k in unique_:
        if int(k[:-2]) in cates:
            if int(k[-1]) in cates[int(k[:-2])]:
                cates[int(k[:-2])][int(k[-1])] +=1
            else:
                cates[int(k[:-2])][int(k[-1])] =1
        else:
            cates[int(k[:-2])] = dict()
            cates[int(k[:-2])][int(k[-1])] = 1
    return cates

def describefailure():
    original = pd.read_csv('data/train.csv')
    failed = pd.read_csv('not_passed_full.csv')
    observe = ['dyeing.method', 'fibre.type', 'count.ply']

    for col in observe:
        zero = set(original[col].unique())
        after = set(failed[col].unique())
        '''
        if col == 'fibre.type':
            origin_ = fibertype(zero)
            failed_ = fibertype(after)
            not_failed_ = fibertype(zero-after)
            print('original distrib of first four chars', origin_)
            print('failed distrib of first four chars', failed_)
            print('not filed distrib of the first four chars', not_failed_)


        elif col == 'count.ply':
            #take the prior of x, then make nested, div by the 1,2,3 after x
            origin_ = ply(zero)
            failed_ = ply(after)
            not_failed_ = ply(zero-after)
            print('original distrib of ply size', origin_)
            print('failed distrib of ply size', failed_)
            print('not failed distrib of ply size', not_failed_) 
        '''
        if col == 'dyeing.method':
            print('original varieties', len(zero))
            print('failure varieities', len(after))
            print('detail', sorted(list(after)))
            print('not failed', sorted(list(zero - after)))



if __name__ == '__main__':
    describefailure()
    #filedescription()
