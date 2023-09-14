import pandas as pd 
import numpy as np 
import copy 
import pylab as pl 
from string import Template
import glob
from matplotlib import gridspec, cm

def read_synthetic_peaks(path):
    f = open(path,'r')
    l = f.readlines()
    f.close()
    nyears = int(path.split('_')[1][:-1])
    qs = []
    for i in np.arange(4,200000,nyears+5):
        a = l[i:i+nyears]
        data = [int(j[:31].split()[-1]) for j in a]
        if len(data)==nyears:    
            qs.append(np.array(data))
        else:
            print(i)
            break
    return qs

def create_cases(path, sigma = 0.1, skew = -0.34, skeError = 0.55, 
                 realizations = 50, skewOpt = 'Weighted', years = [90,80,70,60,50,40,30]):
    name = path.split('-')[-1].split('.')[0]
    print(name)
    sigma = 0.1
    cases = ['noNoise','Noise','NoiseNoShuffle']
    add_noise = [False, True, True]
    shuffle = [True, True, False]
    for case, noise, shuf in zip(cases, add_noise, shuffle):
        for nyears in years:
            l = write_mutated_peaks(path,
                                '%s_%dy_%s.txt' % (name, nyears,case),
                                '%s_%dy_%s.psf' % (name, nyears,case),
                                '%s_%dy_%s.prt'  % (name, nyears,case), 
                                nyears, realizations, skew,skeError,sigma, method = 'EMS', 
                                    add_noise=noise,
                                   shuffle = shuf, skewOpt=skewOpt)
            
forlabels = {'NOISE': 'Shuffle+Noise',
    'NONOISE': 'Shuffle',
    'NOISENOSHUFFLE':'Noise'}
prob = np.array([0.995,0.99,0.95,0.9,0.8,0.6667,0.5,0.4292,0.2,0.1,0.04,0.02,0.01,0.005,0.002])
tr = 1/prob
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

def plot_ratios_vs_tr(all_ratios, do, years = [90,60,30], path = None, ylim = [0.5,2], 
                      yticks = [0.5,1,1.5,2.0],which='wi',hline=0,ylabel='$MSD(Q_{peak})$', error=0.1):
    text = []
    for p in do['prob']:
        pro = p*100
        Tr=1/p
        text.append('%.1f\n%.1f' % (pro, Tr))
    letters = ['a','b','c','d','e','f']    
    fig = pl.figure(figsize=(20,20))
    for c,nyears in enumerate(years):
        ax = fig.add_subplot(6,1,c+1)
        ratios = all_ratios[nyears][which]
        vio1 = pl.violinplot(ratios['NOISE'], positions=np.arange(0,30,2), showmedians=True)
        vio2 = pl.violinplot(ratios['NONOISE'], positions=np.arange(0,30,2)+0.5, showmedians=True)
        vio3 = pl.violinplot(ratios['NOISENOSHUFFLE'], positions=np.arange(0,30,2)-0.5, showmedians=True)
        if c==0:
            pl.legend([vio1['bodies'][0],vio2['bodies'][0],vio3['bodies'][0]],['Bootstrap+Noise','Bootstrap','Noise'],
                     ncol = 3, fontsize = 'xx-large', loc = 9)
        pl.hlines(hline, -1,32, color = 'k', ls = '--')
        ax.set_xticks(np.arange(0,30,2))
        ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(labelsize = 16)
        ax.set_yticks(yticks)
        ax.hlines(error,-1,29,color = 'k',ls = '-',lw = 0.1)
        ax.hlines(-error,-1,29,color = 'k',ls = '-',lw = 0.1)
        ax.fill_between([-1,29],[-error,-error],[error,error], color = 'b',alpha = 0.05)
        ax.grid()
        if c == 1:
            ax.set_ylabel(ylabel, size = 18)
        if c == len(years)-1:
            ax.set_xticklabels(text)
        else:
            ax.set_xticklabels([])
        ax.set_xlim(-1,29)
        t = '%s) %d years' % (letters[c], nyears)
        ax.text(0.01,0.84,t, fontdict = {'size': 18, 'weight':'bold'}, transform = ax.transAxes,
                bbox=dict(facecolor='#d9d9d9', alpha=0.9, lw = 2))
    ax.text(0, -0.1, '$P$', fontdict={'size':15}, transform = ax.transAxes,)
    ax.text(0, -0.2, '$T_r$', fontdict={'size':15}, transform = ax.transAxes)
    if path is not None:
        pl.savefig(path, bbox_inches = 'tight')
    return ax

def plot_statistics_vs_mse(stats, all_mse,wi, pos=-1,
    years = [30,50,90],
    ylim = [-0.5,0.6],
    xlim = [-0.4,0.6],
    xticks = [-0.3,0.0,0.3,0.6],
    yticks = [-0.4,-0.2,0,0.2,0.4],path = None, error = 0.2,which='wi', figsize=(20,13)):
    fig = pl.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, fig, hspace=0.05, wspace = 0.05)
    forLabels = {'NOISE':'Bootstrap+Noise',
                'NONOISE':'Bootstrap',
                'NOISENOSHUFFLE':'Noise'}
    for c, sta in enumerate(years):        
        ax1= fig.add_subplot(gs[0,c])
        ax2= fig.add_subplot(gs[1,c])
        ax3= fig.add_subplot(gs[2,c])
        for k in stats[sta].keys():
            rat = (stats[sta][k][which] - wi)/wi
            ax1.scatter(rat[0], all_mse[sta][which][k][:,pos], label = forLabels[k], s = 120, edgecolor = 'k')
            ax2.scatter(rat[1], all_mse[sta][which][k][:,pos], label = forLabels[k], s = 120, edgecolor = 'k')
            ax3.scatter(rat[2], all_mse[sta][which][k][:,pos], label = forLabels[k], s = 120, edgecolor = 'k')
        if c==0:
            ax1.set_ylabel('$MSD(Q_{peak})$ vs $\Delta(\mu)$', size = 16)
            ax2.set_ylabel('$MSD(Q_{peak})$ vs $\Delta(\sigma)$', size = 16)
            ax3.set_ylabel('$MSD(Q_{peak})$ vs $\Delta(\gamma)$', size = 16)
            ax1.set_title('%d years' % sta, size = 20)            
        if c ==1:
            ax1.set_title('%d years' % sta, size = 20)
            ax3.set_xlabel('$\Delta M = M_i - M_r$', size = 20)
        if c ==2:
            ax1.set_title('%d years' % sta, size = 20)
            ax1.legend(loc = 0, fontsize = 'xx-large')
        if c>0:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
            ax3.set_yticklabels([])
        for ax in [ax1,ax2,ax3]:
            ax.tick_params(labelsize = 15)
            ax.grid()
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_xlim(xlim[0],xlim[1])            
            ax.hlines(error,xlim[0],xlim[1],color = 'k',ls = '-',lw = 1)
            ax.hlines(-error,xlim[0],xlim[1],color = 'k',ls = '-',lw = 1)
            ax.fill_between([xlim[0],xlim[1]],[-error,-error],[error,error], color = 'b',alpha = 0.15)
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
    if path is not None:
        pl.savefig(path, bbox_inches = 'tight')

def plot_statistics_vs_mse_compare(stats, all_mse, wi,
    pos=[-8,-3,-2,-1],
    st=2,
    years = [30,50,90],
    ylim = [-0.4,0.5],
    xlim = [-0.4,0.4],
    xticks = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3],
    yticks = [-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4],
    path = None,
    error = 0.1,
    which='wi',figsize=(20,6*3)):

    fig = pl.figure(figsize=figsize)
    gs = gridspec.GridSpec(len(pos), len(years), fig, hspace=0.05, wspace = 0.05)
    forLabels = {'NOISE':'Bootstrap+Noise',
                'NONOISE':'Bootstrap',
                'NOISENOSHUFFLE':'Noise'}
    cont = 0
    for c, year in enumerate(years):        
        for c2,p in enumerate(pos):
            ax1= fig.add_subplot(gs[c2,c])
    #         ax1.text(0.01,0.84,letters[cont], fontdict = {'size': 18, 'weight':'bold'}, transform = ax1.transAxes,
    #                 bbox=dict(facecolor='#d9d9d9', alpha=0.9, lw = 2))
    #         cont+=1
            for k in stats[year].keys():
                rat = (stats[year][k][which] - wi)/wi            
                ax1.scatter(rat[st], all_mse[year][which][k][:,p], label = forLabels[k], s = 120, edgecolor = 'k')   
                ax1.set_ylim(*ylim)
                ax1.set_xlim(*xlim)
                ax1.tick_params(labelsize = 15)
                ax1.set_xticks(xticks)
                ax1.set_yticks(yticks)
                ax1.grid()
                ax1.hlines(error,xlim[0],xlim[1],color = 'k',ls = '-',lw = 1)
                ax1.hlines(-error,xlim[0],xlim[1],color = 'k',ls = '-',lw = 1)
                ax1.fill_between([xlim[0],xlim[1]],[-error,-error],[error,error], color = 'b',alpha = 0.1)            
                if c != 0:
                    ax1.set_yticklabels([])
                if c == 0:
                    ax1.set_ylabel('$MSD(Q_{p,%d})$' % tr[p], size = 17)
                if c2 != len(pos)-1:
                    ax1.set_xticklabels([])
                if c2 == len(pos)-1 and c == 1:
                    ax1.set_xlabel('$\Delta \gamma = \gamma_i - \gamma_r$', size = 17)
                if c2 == 0:
                    ax1.set_title('%d years' % year, size = 20)
                if c == len(years)-1 and c2 == 0:
                    ax1.legend(loc = 0, fontsize = 'xx-large')
    if path is not None:
        pl.savefig(path, bbox_inches = 'tight')
    return ax1

def plot_median_ratio(all_ratios, years = [90,60,50,30], path = None, ylim = [-1,1], yticks = [-1,-0.5,0,0.5,1],ylabel='MSD($Q_{peak}$)'):
    text = []
    for p in do['prob']:
        pro = p*100
        Tr=1/p
        text.append('%.1f\n%1d' % (pro, Tr))
    cases = ['NOISE','NONOISE','NOISENOSHUFFLE']
    name_case = ['Bootstrap+Noise','Bootstrap','Noise']
    fig = pl.figure(figsize=(20,12))
    gs = gridspec.GridSpec(3, 3, fig, hspace=0.05)

    for c, case in enumerate(cases):

        ax = fig.add_subplot(gs[0,c])
        ax2 = fig.add_subplot(gs[1,c])
        ax3 = fig.add_subplot(gs[2,c])

        for k in years:
            if k == years[-1]:
                lab = '%d Years' % k
            else:
                lab = k
            p = ax.plot(np.percentile(all_ratios[k]['wi'][case], 50, axis = 0), lw = 3)                    
            ax.plot(np.percentile(all_ratios[k]['wo'][case], 50, axis = 0), lw = 3, c = p[0].get_color(), ls = '--')        
            key = 'c5%d' % k
            ax2.plot(np.percentile(all_ratios[key][case], 50, axis = 0), lw = 3,  c = p[0].get_color())        
            key = 'c95%d' % k
            ax3.plot(np.percentile(all_ratios[key][case], 50, axis = 0), lw = 3, c = p[0].get_color(), label = lab,)            
        if c == 1:
            ax3.legend(loc = 'lower right', fontsize = 'xx-large',ncol = 6,
                     bbox_to_anchor =[1.2,-0.38])
        for a in [ax, ax2, ax3]:
            a.hlines(0,0,14, color = 'k', ls = '-')
            a.set_xlim(0,14)
            a.tick_params(labelsize = 16)            
            a.set_xticks(np.arange(0,15,2))
            a.set_xticklabels(text[::2])
            a.set_yticks(yticks)
            a.set_ylim(*ylim)
            a.grid()
        if c == 0:
            ax2.set_ylabel(ylabel, size = 18)
            ax.text(0.01,0.05,'Dashed lines: EMA without reg-skew',transform = ax.transAxes,
                   fontdict = {'size': 16})
        ax.set_xticklabels([])
        ax.text(0.65,0.90,'EMA estimate', fontdict = {'size':16}, transform = ax.transAxes)
        ax2.text(0.62,0.90,'5$\%$ Confidence', fontdict = {'size':16}, transform = ax2.transAxes)
        ax3.text(0.59,0.90,'95$\%$ Confidence', fontdict = {'size':16}, transform = ax3.transAxes)
        ax2.set_xticklabels([])
        t = '%s' % (name_case[c])
        ax.text(0.03,1.0,t, fontdict = {'size': 18, 'weight':'bold'}, transform = ax.transAxes,
                bbox=dict(facecolor='#d9d9d9', alpha=0.9, lw = 2))
    if path is not None:
        pl.savefig(path,  bbox_inches = 'tight')
    return fig,ax,ax2,ax3

def get_ratios(path2prt, do, var = 'w_reg_skew'):
    dn = read_prt(path2prt)
    ratios = []
    for i in dn:
        a = i[var] / do[var]
        ratios.append(a.values)
    return np.array(ratios)

def get_mse(path2prt, do, var = 'w_reg_skew'):
    dn = read_prt(path2prt)
    ratios = []
    for i in dn:
        a = (i[var] - do[var]) / do[var]
        ratios.append(a.values)
    return np.array(ratios)

def ratios4cases_of_gauge(pattern, do):
    #Lists all files with the pattern
    l = glob.glob(pattern)
    #Get the names and years of the files in the pattern
    cases = np.unique([i.split('_')[-1].split('.')[0] for i in l]).tolist()    
    years = np.unique([int(i.split('_')[-2][:-1]) for i in l])
    #Read the data and get the ratios
    all_ratios = {}
    all_mse = {}
    stats={}
    pat = pattern[:-1]
    for nyears in years:
        ratios_wi = {}
        ratios_wo = {}
        mse_wi = {}
        mse_wo = {}
        con5 = {}
        con95 = {}
        mse5 = {}
        mse95 = {}
        st={}
        for i in cases:
            try:
                prt_name = '%s%dY_%s.PRT' % (pat, nyears, i)
                #Read the stats
                wo,wi = get_ema_stats(prt_name)
                st.update({i:{'wi':wi,'wo':wo}})
                #Compute the ratios
                ratios_wi.update({i:get_ratios(prt_name, do,)})
                ratios_wo.update({i:get_ratios(prt_name, do,'no_reg_skew')})
                #compute the mse error
                mse_wi.update({i:get_mse(prt_name, do,)})
                mse_wo.update({i:get_mse(prt_name, do,'no_reg_skew')})
                #Confidence intervals ratios
                con5.update({i:get_ratios(prt_name, do, 'conf5')})
                con95.update({i:get_ratios(prt_name, do, 'conf95')})
                mse5.update({i:get_mse(prt_name, do, 'conf5')})
                mse95.update({i:get_mse(prt_name, do, 'conf95')})
            except:
                pass
        all_ratios.update({nyears: {'wi':ratios_wi, 'wo':ratios_wo}, 'c5%d' % nyears: con5, 'c95%d' % nyears: con95})
        all_mse.update({nyears: {'wi':mse_wi, 'wo':mse_wo}, 'c5%d' % nyears: mse5, 'c95%d' % nyears: mse95})
        stats.update({nyears: st})
    return all_ratios, all_mse, stats

def fix_year_issue(path,path_out):
    f = open(path,'r')
    l = f.readlines()
    f.close()
    
    year1 = int(l[4:][0].split()[1][:4])
    for c,i in enumerate(l[4:]):
        y = year1+c
        yc = l[c+4].split()[1]
        t = '%d0505' % y
        l[c+4] = l[c+4].replace(yc,t)
        
    f = open(path_out,'w')
    f.writelines(l)
    f.close()
    
def constant_error_case(sigma2 = 0.16):
    '''if Xm are the measurments, the actual flood is X = Xm x S. 
    Here S is a random variable greater than zero distributed lognormal
    E[S] = 1 and V[S] = sigma2. If T = ln(S) then T is normal with the 
    following estimators:'''
    #Taken from Potter and walker 1981
    Et = -0.5*np.log(1+sigma2) #Sigma can also be 0.3
    Vt = np.log(1+sigma2)
    return np.exp(np.random.normal(Et, np.sqrt(Vt), 1))[0]

def qp_noise(path, out_path,out_psf,out_prt,ntimes,skewVal, skewError,sigma2 = 0.16, method = 'B17B'):
    '''Reads a peak file from https://nwis.waterdata.usgs.gov/usa/nwis/peak 
    creates a noise following a random function and writes the results 
    to out_path'''
    
    #Define templates to write the psf file
    header_template = Template('I ASCI $peaks_file\n\
    O File $out_peaks_file\n\
    O ConfInterval 0.9\n\
    O EMA YES\n')
    station_template = Template('Station $station\n\
    Analyze $method\n\
    PCPT_Thresh $year_start $year_end 0 1E+20 Default\n\
    BegYear $year_start\n\
    EndYear $year_end\n\
    HistPeriod $num_years\n\
    SkewOpt Weighted\n\
    UseSkewMap NO\n\
    GenSkew $skewVal\n\
    SkewSE $skewError\n\
    LOType MGBT\n')
    
    #Read the file
    f = open(path,'r')
    l = f.readlines()
    f.close()
    original_id = l[0].split()[0][1:] 
    
    #Gets stgart and end years and num years
    year_end = l[-1].split()[1][:4]
    year_start = l[5].split()[1][:4]
    num_years = len(l[4:-1])+1
    
    #Open the file to write into also opens the psf file
    f = open(out_path,'w')
    f_psf = open(out_psf,'w')
    
    #Writes the header of the psf 
    f_psf.write(header_template.safe_substitute({'peaks_file':out_path,
                                       'out_peaks_file':out_prt}))
    qpeaks = []
    #Creates ntimes series
    for z in range(ntimes):
        #Creates a copy of the orifinal list with the data
        lt = copy.copy(l)
        #Grab the data and not the header    
        a = lt[4:]
        new_id = '%d' % (100000000+z)
        new_id = new_id[1:]
        #Iterates through the data creating some noise
        qp_old = []
        qp_new = []
        for c, i in enumerate(a):
            try:
                #Grab the old record
                #t_old = i.split()[2]
                t_old = i[:31].split()[2]
                #Aplly random noise over the record
                #rn = np.random.normal(1,0.05)[0]
                rn = constant_error_case(sigma2)
                random = float(t_old) * rn
                #Replace the record in the list
                t_new = '%d'% random
                #Check lenghts
                dif_len = len(t_old) - len(t_new)
                if dif_len < 0:
                    for j in range(np.abs(dif_len)):
                        t_old = ' '+t_old
                elif dif_len > 0:
                    for j in range(dif_len):
                        t_new = ' '+t_new
                a[c] = i.replace(t_old, t_new) 
                #Change the height 
                try:
                    t_old2 = i.split()[3]
                    random = float(t_old2) * rn
                    #Replace the record in the list
                    t_new2 = '%.2f'% random
                    a[c] = a[c].replace(t_old2, t_new2)
                except:
                    pass
                qp_old.append(int(t_old))
                qp_new.append(int(t_new))
            except:
                pass
        #Writes the new file 
        lt[4:] = a
        
        #Replace the id of the time series
        for c, i in enumerate(lt):
            lt[c] = i.replace(original_id, new_id)
        f.writelines(lt)
        f.write('\n')
        
        #Updates the psf file 
        f_psf.write(station_template.safe_substitute({'station':new_id,
                                                      'skewVal':skewVal, 
                                                      'skewError':skewError,
                                                     'year_start':year_start,
                                                     'year_end':year_end,
                                                     'num_years':num_years,
                                                     'method':method}))
        #updates the series with noised peak flows 
        qpeaks.append(np.array(qp_new))
    
    f.close()
    return qp_old, qpeaks


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

def get_ema_stats(path):

    def search_pattern(pattern):    
        try:
            p = i.index(pattern)
            return [float(j) for j in l[c].split()[-3:]]
        except:
            return 1
    
    f = open(path,'r')
    l = f.readlines()
    f.close()
    
    data_wo = []
    data_wi = []
    for c, i in enumerate(l):
        we = search_pattern('EMA WITHOUT REG SKEW')
        if we != 1: data_wo.append(we)
        we = search_pattern('EMA WITH REG SKEW')
        if we != 1: data_wi.append(we)
    return np.array(data_wo).T, np.array(data_wi).T

def plot_freq(do, ax, dn = None, q = None, ema = None, path = None):
    #fig = pl.figure(figsize=(20,10))
    #ax = fig.add_subplot(111)    
    if dn is not None:
        pre = ''
        for d in dn:    
            ax.plot(1-d['prob'], d['conf5']*0.028, lw = 3, c = '#feb24c',zorder=1, label = pre+'Sce $5\%$ Confidence')
            ax.plot(1-d['prob'], d['conf95']*0.028, lw = 3, c = '#feb24c',zorder=1, label = pre+'Sce $95\%$ Confidence')
            pre='_'
        pre = ''
        for d in dn:    
            ax.plot(1-d['prob'], d['w_reg_skew']*0.028, lw = 3,c = '#9ecae1',zorder=1,label = pre+'Sce Fitted')
            pre = '_'
    ax.plot(1-do['prob'], do['w_reg_skew']*0.028, lw = 5, c = 'k',zorder=1, label = '$5\%$ Confidence')
    ax.plot(1-do['prob'], do['conf5']*0.028, ls = '--',lw = 3, c = 'r',zorder=1, label = '$5\%$ Confidence')
    ax.plot(1-do['prob'], do['conf95']*0.028,ls = '--', lw = 3, c = 'r',zorder=1, label = '$95\%$ Confidence')
    if q is not None:
        ax.scatter(1-ema, q*0.028, s = 250, edgecolor = 'k', zorder = 1, alpha = 0.7, label = 'Observed')
    ax.set_yscale('log')
    ax.set_xscale('logit')
    x = 1-do['prob'].values
    tr = 1/do['prob'].values
    trt = np.array(['%.2f' % i for i in 100*do['prob'].values])
    ax.set_xlim(x[0],x[-1])
    p = [0,2,6,8,9,11,-2,-1]
    ax.set_xticks(x[p])
    ax.legend(loc = 0, fontsize = 'xx-large')
    ax.set_xticklabels(trt[p])
    ax.tick_params(labelsize = 22)
    ax.set_xlabel('Annual exeedance probability [$\%$]',size = 24)
    ax.set_ylabel('Streamflow [cms]',size = 24)
    ax.grid(which='both')
    if path is not None:
        pl.savefig(path, bbox_inches = 'tight')
    return ax

def read_obs(path):
    f = open(path, 'r')
    l = f.readlines()
    f.close()

    to_find1 = '  TABLE 6 - EMPIRICAL FREQUENCY CURVES -- HIRSCH-STEDINGER PLOTTING POSITIONS\n'
    to_find2 = '                    TABLE 7 - EMA REPRESENTATION OF DATA\n'
    pos1 = l.index(to_find1)
    pos2 = l.index(to_find2)

    a = l[pos1+4:pos2-10]
    q = []
    ema = []
    pilf = []
    year = []
    for i in a:    
        try:
            if len(i.split()) == 3:
                year.append(float(i.split()[0]))
                q.append(float(i.split()[2]))
                ema.append(float(i.split()[1]))
                pilf.append(0)
            else:
                year.append(float(i.split()[1]))
                q.append(float(i.split()[3]))
                ema.append(float(i.split()[2]))
                pilf.append(1)
        except:
            pass
    return np.array(ema), np.array(q), np.array(pilf)

def replace(qpeaks, rec_len):    
    #Get the random positions to switch
    pos = np.random.choice(np.arange(0,len(qpeaks)),len(qpeaks), False)
    for p1, p2 in enumerate(pos):
        t1 = qpeaks[p1][24:]
        t2 = qpeaks[p2][24:]
        qpeaks[p1] = qpeaks[p1].replace(t1, t2)
        qpeaks[p2] = qpeaks[p2].replace(t2, t1)
    return qpeaks[:rec_len]

def mutate(qpeaks, sigma2, error = None):
    d = []
    for i in qpeaks:
        try:
            d.append(float(i[:31].split()[2]))
        except:
            pass
    d = np.array(d)
    
    a = copy.copy(qpeaks)
    b= []
    for c, i in enumerate(a):
        try:
            #Grab the old record
            t_old = i[:31].split()[2]
            #Aplly random noise over the record
            if error != None:
                sigma2 = (float(t_old)/d.mean()) * error
            rn = constant_error_case(sigma2)
            random = float(t_old) * rn
            #Replace the record in the list
            t_new = '%d'% random
            #Check lenghts
            dif_len = len(t_old) - len(t_new)
            if dif_len < 0:
                for j in range(np.abs(dif_len)):
                    t_old = ' '+t_old
            elif dif_len > 0:
                for j in range(dif_len):
                    t_new = ' '+t_new
            a[c] = i[:26] +  i[26:].replace(t_old, t_new)
            #Change the height 
            try:            
                t_old2 = i.split()[3]
                random = float(t_old2) * rn
                #Replace the record in the list
                t_new2 = '%.2f'% random
                a[c] = a[c].replace(t_old2, t_new2)
            except:
                pass        
        except:
            pass
    return a

def write_mutated_peaks(path_original, out_path, out_psf,out_prt, rec_len, ntimes, 
                        skewVal, skewError, sigma2=0.16, method = 'EMS', mapSkew = 'NO', add_noise = False,
                        shuffle = True, skewOpt = 'Weighted', error = None):
    #Read the original peak flow file
    f = open(path_original,'r')
    l = f.readlines()
    f.close()
    original_id = l[0].split()[0][1:] 
    
    #Define templates to write the psf file
    header_template = Template('I ASCI $peaks_file\n\
        O File $out_peaks_file\n\
        O ConfInterval 0.9\n\
        O EMA YES\n')
    station_template = Template('Station $station\n\
        Analyze $method\n\
        PCPT_Thresh $year_start $year_end 0 1E+20 Default\n\
        BegYear $year_start\n\
        EndYear $year_end\n\
        HistPeriod $num_years\n\
        SkewOpt $skewOpt\n\
        UseSkewMap $mapSkew\n\
        GenSkew $skewVal\n\
        SkewSE $skewError\n\
        LOType MGBT\n')
    
    
    #Open the file to write into also opens the psf file
    f = open(out_path,'w')
    f_psf = open(out_psf,'w')
    
    #Writes the header of the psf 
    f_psf.write(header_template.safe_substitute({'peaks_file':out_path,
                                       'out_peaks_file':out_prt}))

    for z in range(ntimes):
        #Grab the data and not the header    
        new_id = '%d' % (100000000+z)
        new_id = new_id[1:]
    
        #Get the mutate peaks 
        if shuffle:
            qpeaks = replace(l[4:], rec_len)
        else:
            qpeaks = l[4:rec_len+4]
        if add_noise:
            qpeaks = mutate(qpeaks, sigma2, error)
        
        #Fix end of line issue that I have no idea why happens
        for c,i in enumerate(qpeaks):
            if i[-1] !='\n':
                qpeaks[c]+='\n'

        #Replace the id of the time series
        lt = l[:4] + qpeaks
        for c, i in enumerate(lt):
            lt[c] = i.replace(original_id, new_id)
        f.writelines(lt)
        f.write('\n')
     
        #Updates the psf file 
        year_start = int(qpeaks[0].split()[1][:4])
        year_end = int(qpeaks[-1].split()[1][:4])
        num_years = year_end - year_start+1
        f_psf.write(station_template.safe_substitute({'station':new_id,
                                                      'skewVal':skewVal, 
                                                      'skewError':skewError,
                                                      'skewOpt':skewOpt,
                                                     'year_start':year_start,
                                                     'year_end':year_end,
                                                     'num_years':num_years,
                                                     'method':method,
                                                     'mapSkew':mapSkew}))
    f.close()
    f_psf.close()
    return qpeaks