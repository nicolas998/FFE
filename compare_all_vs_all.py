import numpy as np 
import pandas as pd 
import argparse 
#import peak analist as pk

parser = argparse.ArgumentParser(description='Compares all the realizations using a PRT as the only input')
parser.add_argument('path',type=str, help = 'path to the PRT with all the runs')
args = parser.parse_args()

def read_prt(path_prt, method = 'EMS'):
    f = open(path_prt,'r')
    l = f.readlines()
    f.close()
    flag = True
    positions = []
    d = []
    while flag:
        try:
            #Find the place with the regression
            if method == 'EMS':
                pos = l.index('PROBABILITY REG SKEW  REG SKEW       OF EST.    5.0% LOWER   95.0% UPPER\n')
            else:
                pos = l.index('PROBABILITY ESTIMATE   RECORD      OF EST.   5.0% LOWER  95.0% UPPER\n')
            positions.append(pos)
            l[pos] = 'changed'

            #Read the data
            if method == 'EMS':
                data = []
                for j in l[pos+2:pos+17]:
                    data.append([float(i) for i in j.split()])
                d.append(pd.DataFrame(np.array(data), columns=['prob','w_reg_skew','no_reg_skew','log_var','conf5','conf95']))
            else:
                data = []
                for j in l[pos+5:pos+17]:
                    t = []
                    for i in j.split():
                        try:
                            t.append(float(i))
                        except:
                            pass
                    data.append(t)
                d.append(pd.DataFrame(np.array(data), columns=['prob','estimate','record','conf5','conf95']))
        except:
            flag = False    
    return d

if __name__=='__main__':
    d = read_prt(args.path)
    rd = []
    for i in range(len(d)):
        for j in range(len(d)):
            if  i != j:
                a = (d[i]['w_reg_skew'] - d[j]['w_reg_skew'])/ d[i]['w_reg_skew']
                rd.append(a.values)

    rd = np.array(rd)
    name = args.path.split('/')[-1].split('.')[0]
    path_out = '/Users/nicolas/FFE/data/processed/all_vs_all/%s.npz' % name
    np.savez(path_out, rd=rd)