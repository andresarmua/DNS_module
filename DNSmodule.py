# Module to explore DNS spc and stats files
#
# Note that this is not hardcoded to the number of file lines,
# but it depends on the names of the headers so
# if an error is found, check that the targets
# within the functions are correct
# Andres Armua 15/03/2022


import matplotlib.pyplot as plt
import sys
from glob import glob
from matplotlib import cm
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import numpy as np
import seaborn as sns



#------- get parameters from header -------------

def get_lattice(ufilename): 
    '''return lattice (k_max) from header''' 
    with open(ufilename) as readfile: 
        ulines = readfile.readlines() 
     
    target_parameter    = 'LATTICE' 
 
    for line in ulines: 
         
        try: 
            parameter   = line.split()[1] 
            value       = line.split()[2] 
        except: 
            print("spc file not formatted correctly in line {}".format(ulines.index(line))) 
            exit(1) 
         
        if parameter == target_parameter: 
            lattice = int(value) #why not integer? 
                        #there was a good reason I can't recall 
        elif parameter == 'Plot': 
            break 
 
    return lattice 

def get_viscosity(ufilename): 
    '''return viscosity from header''' 
    with open(ufilename) as readfile: 
        ulines = readfile.readlines() 
     
    target_parameter    = 'VISC' 
 
    for line in ulines: 
         
        try: 
            parameter   = line.split()[1] 
            value       = line.split()[2] 
        except: 
            print("spc file not formatted correctly in line {}".format(ulines.index(line))) 
            exit(1) 
         
        if parameter == target_parameter: 
            viscosity = float(value) #why not integer? 
                        #there was a good reason I can't recall 
        elif parameter == 'Plot': 
            break 
 
    return viscosity 

def get_kpeak(ufilename): 
    '''return viscosity from header''' 
    with open(ufilename) as readfile: 
        ulines = readfile.readlines() 
     
    target_parameter    = 'WAVENUMBER' 
 
    for line in ulines: 
         
        try: 
            parameter   = line.split()[1] 
            value       = line.split()[2] 
        except: 
            print("spc file not formatted correctly in line {}".format(ulines.index(line))) 
            exit(1) 
         
        if parameter == target_parameter: 
            kpeak = float(value) #why not integer? 
                        #there was a good reason I can't recall 
        elif parameter == 'Plot': 
            break 
 
    return kpeak 

def get_prandtl(ufilename): 
    '''return prandtl from header''' 
    with open(ufilename) as readfile: 
        ulines = readfile.readlines() 
     
    target_parameter    = 'PRANDTL' 
 
    for line in ulines: 
         
        try: 
            parameter   = line.split()[1] 
            value       = line.split()[2] 
        except: 
            print("spc file not formatted correctly in line {}".format(ulines.index(line))) 
            exit(1) 
         
        if parameter == target_parameter: 
            Pm = float(value) #why not integer? 
                        #there was a good reason I can't recall 
        elif parameter == 'Plot': 
            break 
 
    return Pm 



def get_dt(ufilename):
    '''return time interval of the output (not dt of the simulation) from header'''
    with open(ufilename) as readfile:
        ulines = readfile.readlines()

    target_parameter = 'INTERVAL'


    for line in ulines:

        try:
            parameter   = line.split()[1]
            value       = line.split()[2]
        except:
            print("spc file not formatted correctly in \
line {}".format(ulines.index(line)))
            exit(1)

        if parameter == target_parameter:
            dt = float(value)
        elif parameter == 'Plot':
            break

    return dt


def get_initial_time(ufilename):
    '''return initial time of spc file'''
    if '.stats' in ufilename:
        ufilename = ufilename.replace('stats','spc')
    with open(ufilename) as readfile:
        ulines = readfile.readlines()
    init_time = -1
    spectra_block_check = 'asd'
    for line in ulines:
        line_data = line.split()

        try:
            spectra_block_check = line_data[1]
        except:
            pass

        if spectra_block_check == "Plot":
            init_time = float(line_data[-1])

        if init_time != -1:
            break

    return init_time


def get_final_time(ufilename):
    '''return final time of spc file'''
    if '.stats' in ufilename:
        ufilename = ufilename.replace('stats','spc')
    with open(ufilename) as readfile:
        ulines = readfile.readlines()
    final_time = -1
    ulines.reverse()
    spectra_block_check = 'asd'
    for line in ulines:
        line_data = line.split()

        try:
            spectra_block_check = line_data[1]
        except:
            pass

        if spectra_block_check == "Plot":
            final_time = float(line_data[-1])

        if final_time != -1:
            break

    return final_time


# ------------------- MHD spectrums (spc) ----------------

def B_spectrum(ufilename):
    '''gets array with mag energy spectrum in time'''

    readfile = open(ufilename)
    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split('\t')
        for data in line_data:
            if 'Eb(k)' in data:
                idx = line_data.index(data)
    readfile = open(ufilename)

    init_time   = get_initial_time(ufilename)
    final_time  = get_final_time(ufilename)
    total_time  = final_time - init_time
    dt = get_dt(ufilename)
    n_steps     = int(total_time/dt) 
    k_max = int(get_lattice(ufilename)/3) -1
    spec_columns = np.loadtxt(readfile,usecols=idx)
    spectrums   = np.empty((0,k_max))
    for i in range(n_steps):
        idx_start   = i * k_max
        idx_end     = (i+1) * k_max
        spectrum    = spec_columns[idx_start:idx_end]
        spectrums   =  np.vstack((spectrums,spectrum))
    return spectrums





def E_spectrum(ufilename):
    '''gets array with kin energy spectrum in time'''
    if '.b_' in ufilename:
        ufilename = ufilename.replace('.b_','.u_')
    readfile = open(ufilename)
    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split('\t')
        for data in line_data:
            if 'Eu(k)' in data:
                idx = line_data.index(data)
    readfile = open(ufilename)

    init_time   = get_initial_time(ufilename)
    final_time  = get_final_time(ufilename)
    total_time  = final_time - init_time
    dt = get_dt(ufilename)
    n_steps     = int(total_time/dt)
    k_max = int(get_lattice(ufilename)/3) - 1
    spec_columns = np.loadtxt(readfile,usecols=idx)
    spectrums   = np.empty((0,k_max))
    for i in range(n_steps):
        idx_start   = i * k_max
        idx_end     = (i+1) * k_max
        spectrum    = spec_columns[idx_start:idx_end]
        spectrums   =  np.vstack((spectrums,spectrum))
    return spectrums


def Hm_spectrum(ufilename):
    '''gets array with mag energy spectrum in time'''

    readfile = open(ufilename)
    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split('\t')
        for data in line_data:
            if 'Hb(k)' in data:
                idx = line_data.index(data)
    readfile = open(ufilename)

    init_time   = get_initial_time(ufilename)
    final_time  = get_final_time(ufilename)
    total_time  = final_time - init_time
    dt = get_dt(ufilename)
    n_steps     = int(total_time/dt) 
    k_max = int(get_lattice(ufilename)/3) -1
    spec_columns = np.loadtxt(readfile,usecols=idx)
    spectrums   = np.empty((0,k_max))
    for i in range(n_steps):
        idx_start   = i * k_max
        idx_end     = (i+1) * k_max
        spectrum    = spec_columns[idx_start:idx_end]
        spectrums   =  np.vstack((spectrums,spectrum))
    return spectrums



def Tb_spectrum(ufilename):
    '''gets array with mag energy spectrum in time'''

    readfile = open(ufilename)
    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split('\t')
        for data in line_data:
            if 'Tb(k)' in data:
                idx = line_data.index(data)
    readfile = open(ufilename)

    init_time   = get_initial_time(ufilename)
    final_time  = get_final_time(ufilename)
    total_time  = final_time - init_time
    dt = get_dt(ufilename)
    n_steps     = int(total_time/dt) 
    k_max = int(get_lattice(ufilename)/3) -1
    spec_columns = np.loadtxt(readfile,usecols=idx)
    spectrums   = np.empty((0,k_max))
    for i in range(n_steps):
        idx_start   = i * k_max
        idx_end     = (i+1) * k_max
        spectrum    = spec_columns[idx_start:idx_end]
        spectrums   =  np.vstack((spectrums,spectrum))
    return spectrums

def Tu_spectrum(ufilename):
    '''gets array with kin energy spectrum in time'''
    if '.b_' in ufilename:
        ufilename = ufilename.replace('.b_','.u_')
    readfile = open(ufilename)
    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split('\t')
        for data in line_data:
            if 'Tu(k)' in data:
                idx = line_data.index(data)
    readfile = open(ufilename)

    init_time   = get_initial_time(ufilename)
    final_time  = get_final_time(ufilename)
    total_time  = final_time - init_time
    dt = get_dt(ufilename)
    n_steps  = int(total_time/dt) 
    k_max = int(get_lattice(ufilename)/3) - 1
    spec_columns = np.loadtxt(readfile,usecols=idx)
    spectrums   = np.empty((0,k_max))
    for i in range(n_steps):
        idx_start   = i * k_max
        idx_end     = (i+1) * k_max
        spectrum    = spec_columns[idx_start:idx_end]
        spectrums   =  np.vstack((spectrums,spectrum))
    return spectrums


def col_t(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' time' in data:
                idx = line_data.index(data)
    time = np.loadtxt(ufilename,usecols=idx)



    return time



def col_t_hyd(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' time' in data:
                idx = line_data.index(data)
    time = np.loadtxt(ufilename,usecols=idx)



    return time




#---------- MHD stats --------------------

def col_Lb(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' Lb' in data:
                idx = line_data.index(data)
    Lb = np.loadtxt(ufilename,usecols=idx)



    return Lb

def col_L(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' L' in data:
                idx = line_data.index(data)
    L = np.loadtxt(ufilename,usecols=idx)



    return L

def col_E(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' Ekin' in data:
                idx = line_data.index(data)
    E = np.loadtxt(ufilename,usecols=idx)
    

    return E

def col_B(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' Emag' in data:
                idx = line_data.index(data)
    B = np.loadtxt(ufilename,usecols=idx)



    return B



def col_Hm(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' a.b' in data:
                idx = line_data.index(data)
    Hm = np.loadtxt(ufilename,usecols=idx)
    

    return Hm


def col_epsb(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' epsmag+' in data:
                idx = line_data.index(data)
    epsb = np.loadtxt(ufilename,usecols=idx)
    

    return epsb


def col_epsu(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    if '.u_' or '.b_' in ufilename:
        ufilename = ufilename.replace('.u_','.ub_')
        ufilename = ufilename.replace('.b_','.ub_')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' epskin+' in data:
                idx = line_data.index(data)
    epsu = np.loadtxt(ufilename,usecols=idx)
    

    return epsu



# ---------------- hydrodynmic stats -------------------
def col_Ehyd(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' energy' in data:
                idx = line_data.index(data)
    E = np.loadtxt(ufilename,usecols=idx)

    return E


def col_Lhyd(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' L' in data:
                idx = line_data.index(data)
    L = np.loadtxt(ufilename,usecols=idx)

    return L

def col_thyd(ufilename):
    if '.spc' in ufilename:
        ufilename = ufilename.replace('.spc','.stats')
    try:
        readfile = open(ufilename)
    except:
        # in the future: add stats file creation from script
        print('Error: stats file missing, run script on spc file')
        exit(1)

    ulines = readfile.readlines()

    for line in ulines:
        line_data = line.split(',')
        for data in line_data:
            if ' time' in data:
                idx = line_data.index(data)
    t = np.loadtxt(ufilename,usecols=idx)

    return t



# ----- get all data MHD (TODO: HD)-----------    

def get_data_MHD(file_list):
    ks  = []
    Eks = []
    Bks = []
    times = []
    Es = []
    Bs = []
    Us  = []
    Vas = []
    Lus = []
    Lbs = []
    Tus  = []
    Tbs  = []
    epsus = []
    epsbs = []
    lus = []
    lbs = []
    viscs = []
    Pms = []
    Ns = []
    kps = []

    for f in file_list:
        N = get_lattice(f)
        k = np.arange(1,int(N/3),1)
        Ek = E_spectrum(f)
        Bk  = B_spectrum(f)
        time = col_t(f)
        Ekin = col_E(f)
        Emag = col_B(f)
        U = np.sqrt(2*Ekin/3)
        Va = np.sqrt(2*Emag/3) #Alfven vel
        Lu = col_L(f)
        Lb = col_Lb(f)
        try:
            Tu = Lu/U
        except:
            Tu = np.arange(Lu.shape,None)
        try:
            Tb = Lb/Va
        except:
            Tb = np.arange(Lb.shape,None)
        epsu = col_epsu(f)
        epsb = col_epsb(f)
        visc = get_viscosity(f)
        Pm = get_prandtl(f)
        kp = get_kpeak(f)
        try:
            lu = np.power(visc**3/epsu,1/4)
        except:
            lu = 0
        res = visc / Pm
        lb = np.power(res**3/epsb,1/4)
        
        
        ks.append(k)
        Eks.append(Ek)
        Bks.append(Bk)
        times.append(time)
        Es.append(Ekin)
        Bs.append(Emag)
        Us.append(U)
        Vas.append(Va)
        Lus.append(Lu)
        Lbs.append(Lb)
        Tus.append(Tu)
        Tbs.append(Tb)
        epsus.append(epsu)
        epsbs.append(epsb)
        lus.append(lu)
        lbs.append(lb)
        viscs.append(visc)
        Pms.append(Pm)
        Ns.append(N)
        kps.append(kp)
        
    Pms = np.array([int(Pm) if Pm.is_integer() else Pm for Pm in Pms])
    
    return np.array(ks), Eks, Bks, np.array(times), np.array(Es), np.array(Bs), np.array(Us),np.array(Vas), np.array(Lus), np.array(Lbs), np.array(Tus), np.array(Tbs), np.array(epsus),np.array(epsbs), np.array(lus), np.array(lbs), np.array(viscs), np.array(Pms),np.array(Ns),np.array(kps)


def arg_sort_MHD(ks, Eks, Bks, times, Es, Bs, Us, Vas, Lus, Lbs, Tus, Tbs, epsus, epsbs, lus, lbs, viscs, Pms, Ns, kps, files, idx_sort):
    
    ks  = ks[idx_sort]
    Eks = [Eks[i] for i in idx_sort]
    Bks = [Bks[i] for i in idx_sort]
    times = times[idx_sort]
    Es  = Es[idx_sort]
    Bs  = Bs[idx_sort]
    Us  = Us[idx_sort]
    Vas = Vas[idx_sort]
    Lus = Lus[idx_sort]
    Lbs = Lbs[idx_sort]
    Tus = Tus[idx_sort]
    Tbs = Tbs[idx_sort]
    epsus = epsus[idx_sort]
    epsbs = epsbs[idx_sort]
    lus = lus[idx_sort]
    lbs = lbs[idx_sort]
    viscs = viscs[idx_sort]
    Pms = Pms[idx_sort]
    Ns = Ns[idx_sort]
    kps = kps[idx_sort]
    files = np.array(files)
    files = files[idx_sort]
  
    
    return ks, Eks, Bks, Bks, times, Es, Bs, Us, Vas, Lus, Lbs, Tus, Tbs, epsus,epsbs, lus, lbs, viscs, Pms, Ns, kps, files


# --------- functions that can be useful ---------

def integrate(spec_field,_kmin,_kmax,_k):
    '''interate spectrums of the shape 
        Spec[time,k] '''
    integral = 0.0
    for idx,__k in enumerate(_k):
        if ((__k > _kmin) and (__k <= _kmax)):
            integral += spec_field[:,idx] # first element of field is time
        elif __k > _kmax:
            break
    return integral


def sch_function(t,f,p):
    df = np.gradient(f,t)
    return np.abs(df * np.power(f,-1-1/p))

def p(t,E):
    
    p,b = np.polyfit(np.log(t[-200:]),np.log(E[-200:]),1)
    
    p = -p
    p_inf = 0.8*p
    p_sup = 1.2*p
    
    p_range = np.linspace(p_inf,p_sup,6)
    y_end = []
    for idx, p in enumerate(p_range):
        y = sch_function(t,E,p)
        plt.plot(t,y,label='{:.2f}'.format(p))
        y_end.append(y[-1])
        
    y_end = np.array(y_end)
    y_min = y_end.min()
    y_max = y_end.max()
    plt.ylim(0.9*y_min,1.1*y_max)
    plt.xlim(0,5)    
    plt.ylabel('$\| E^{-1-1/p} dE/dt \|$')
    plt.xlabel('t')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.show()
    
def q(t,L):
    
    q,b = np.polyfit(np.log(t[-200:]),np.log(L[-200:]),1)
    
    q 
    q_inf = 0.8*q
    q_sup = 1.2*q
    
    q_range = np.linspace(q_inf,q_sup,6)
    y_end = []
    
    for idx, q in enumerate(q_range):
        y = sch_function(t,L,q)
        plt.plot(t,y,label='{:.2f}'.format(q))
        y_end.append(y[-1])
        
        
    y_end = np.array(y_end)
    y_min = y_end.min()
    y_max = y_end.max()
    plt.ylim(0.9*y_min,1.1*y_max)
    plt.xlim(0,5)    
    plt.ylabel('$\| L^{-1-1/q} dL/dt \|$')
    plt.xlabel('t')
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.show()

    
def fit(x,A,B):
    return A + B/x

def fit2(x,A,B):
    return A + B/np.power(x,2)

def fitm2(x,A,B):
    return A + B/np.power(x,1/2)


def fitm4(x,A,B):
    return A + B/np.power(x,1/4)

def fitm8(x,A,B):
    return A + B/np.power(x,1/8)



def fit_complete(x,A,B,exp):
    return A + B/np.power(x,exp)

def fit_complete2(x,B,exp):
    return 1 + B/np.power(x,exp)